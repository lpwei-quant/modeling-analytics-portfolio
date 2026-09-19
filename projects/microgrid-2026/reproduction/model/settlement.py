"""Physical settlement of already chosen Q2 actions, in kWh per interval."""
from __future__ import annotations

from typing import Any

import numpy as np

LEDGER_FIELDS = (
    "q", "a", "u", "v", "w", "c", "b", "e", "soc_start", "soc_end",
    "plan_cost", "emergency_cost", "cash_cost",
)

def _prepare_inputs(q, charge, discharge, load, pv, initial_soc, price,
                    eta_charge, eta_discharge, soc_min, soc_max,
                    bus_limit, emergency_multiplier, tolerance):
    arrays = {}
    for name, value in (("q", q), ("c", charge), ("b", discharge),
                        ("load", load), ("pv", pv), ("price", price)):
        array = np.array(value, dtype=float, copy=True)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a nonempty one-dimensional array")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains nonfinite values")
        arrays[name] = array
    if len({len(array) for array in arrays.values()}) != 1:
        raise ValueError("All input vectors must have the same length")
    scalars = {
        "initial_soc": initial_soc, "eta_charge": eta_charge,
        "eta_discharge": eta_discharge, "soc_min": soc_min,
        "soc_max": soc_max, "bus_limit": bus_limit,
        "emergency_multiplier": emergency_multiplier, "tolerance": tolerance,
    }
    for name, value in scalars.items():
        if not np.isscalar(value) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite scalar")
    if not (0 < eta_charge <= 1 and 0 < eta_discharge <= 1):
        raise ValueError("Charge and discharge efficiencies must lie in (0, 1]")
    if not 0 <= soc_min < soc_max:
        raise ValueError("Require 0 <= soc_min < soc_max")
    if bus_limit <= 0 or emergency_multiplier <= 0 or tolerance <= 0:
        raise ValueError("bus_limit, emergency_multiplier and tolerance must be positive")
    return arrays

def _reject(mask, message, values=None):
    positions = np.flatnonzero(mask)
    if positions.size:
        indices = positions[:8].tolist()
        detail = "" if values is None else f"; values={np.asarray(values)[positions[:8]].tolist()}"
        raise ValueError(f"{message}; interval indices={indices}{detail}")

def settle_schedule(q, charge, discharge, load, pv, initial_soc, price, *,
                    eta_charge=0.9, eta_discharge=0.9, soc_min=1200.,
                    soc_max=10800., bus_limit=5000/6,
                    emergency_multiplier=5., enforce_m0=False, tolerance=1e-6):
    """Settle a fixed schedule without modifying any submitted action."""
    z = _prepare_inputs(q, charge, discharge, load, pv, initial_soc, price,
                        eta_charge, eta_discharge, soc_min, soc_max,
                        bus_limit, emergency_multiplier, tolerance)
    for name, values in z.items():
        _reject(values < -tolerance, f"Negative {name}", values)
    q, c, b, load, pv, price = (z[k] for k in ("q", "c", "b", "load", "pv", "price"))
    if not soc_min - tolerance <= initial_soc <= soc_max + tolerance:
        raise ValueError(f"Initial SOC {initial_soc} outside [{soc_min}, {soc_max}]")
    _reject(c > bus_limit + tolerance, "Charge bus power/energy limit exceeded", c)
    _reject(b > bus_limit + tolerance, "Discharge bus power/energy limit exceeded", b)
    _reject(np.minimum(c, b) > tolerance, "Simultaneous charging and discharging", np.minimum(c, b))
    soc_end = float(initial_soc) + np.cumsum(eta_charge*c - b/eta_discharge)
    soc_start = np.r_[float(initial_soc), soc_end[:-1]]
    _reject(soc_end < soc_min - tolerance, "SOC below lower bound", soc_end)
    _reject(soc_end > soc_max + tolerance, "SOC above upper bound", soc_end)

    requirement = load + c - b
    _reject(requirement < -tolerance,
            "A = load + charge - discharge is negative; battery export/dumping is forbidden",
            requirement)
    v = np.minimum(pv, requirement)
    unserved_after_pv = requirement - v
    a = np.minimum(q, unserved_after_pv)
    e = unserved_after_pv - a
    u = q - a
    w = pv - v
    m0_product = e*c
    m0_mask = (e > tolerance) & (c > tolerance)
    if enforce_m0:
        _reject(m0_product > tolerance, "M0 forbids emergency energy while charging (e*c > tolerance)", m0_product)
    plan_cost = price*q
    emergency_cost = emergency_multiplier*price*e
    metadata = {
        "energy_unit": "kWh_per_interval", "soc_unit": "internal_kWh", "cost_unit": "yuan",
        "settlement_interpretation": "M0_enforced" if enforce_m0 else "approved_v2_general_bus",
        "m0_enforced": bool(enforce_m0),
        "emergency_charge_overlap_periods": int(np.count_nonzero(m0_mask)),
        "emergency_charge_overlap_fraction": float(np.mean(m0_mask)),
        "overlap_definition": "e > tolerance and c > tolerance",
        "max_charge_discharge_product_kwh2": float(np.max(c*b)),
        "m0_max_e_times_c_kwh2": float(np.max(m0_product)),
        "emergency_kwh_during_charging": float(np.sum(e[m0_mask])),
        "charging_kwh_with_emergency": float(np.sum(c[m0_mask])),
        "action_modification_count": 0,
        "action_clipping_count": 0,
        "signed_roundoff_values_retained": int(sum(np.count_nonzero(vv < 0) for vv in z.values())),
        "tolerance": float(tolerance),
        "n_periods": len(q),
    }
    return dict(q=q, a=a, u=u, v=v, w=w, c=c, b=b, e=e,
                soc_start=soc_start, soc_end=soc_end,
                plan_cost=plan_cost, emergency_cost=emergency_cost,
                cash_cost=plan_cost+emergency_cost, metadata=metadata)

def independent_recalculator(q, charge, discharge, load, pv, initial_soc, price,
                             ledger, *, eta_charge=0.9, eta_discharge=0.9,
                             soc_min=1200., soc_max=10800., bus_limit=5000/6,
                             emergency_multiplier=5., enforce_m0=False,
                             tolerance=1e-6) -> dict[str, Any]:
    """Independently replay a ledger using scalar allocation and SOC updates."""
    inputs = _prepare_inputs(q, charge, discharge, load, pv, initial_soc, price,
                             eta_charge, eta_discharge, soc_min, soc_max,
                             bus_limit, emergency_multiplier, tolerance)
    n = len(inputs["q"])
    expected = {field: [] for field in LEDGER_FIELDS}
    violations: dict[str, int] = {}

    def mark(name, condition):
        if condition:
            violations[name] = violations.get(name, 0) + 1

    state = float(initial_soc)
    mark("initial_soc_out_of_bounds", state < soc_min-tolerance or state > soc_max+tolerance)
    m0_count = 0
    m0_max = 0.
    for t in range(n):
        qt, ct, bt, lt, vt, pt = (float(inputs[k][t]) for k in ("q", "c", "b", "load", "pv", "price"))
        for key, value in (("q", qt), ("c", ct), ("b", bt), ("load", lt), ("pv", vt), ("price", pt)):
            mark("negative_"+key, value < -tolerance)
        mark("charge_limit", ct > bus_limit+tolerance)
        mark("discharge_limit", bt > bus_limit+tolerance)
        mark("charge_discharge_overlap", ct > tolerance and bt > tolerance)
        before = state
        state += float(eta_charge)*ct
        state -= bt/float(eta_discharge)
        mark("soc_bounds", state < soc_min-tolerance or state > soc_max+tolerance)
        required = lt + ct - bt
        mark("battery_export_or_dump", required < -tolerance)
        # Source-by-source scalar allocation, independently from the vector
        # min/residual expressions used by the production settlement.
        if vt >= required:
            pv_used = required
            pv_spill = vt-required
            outstanding = 0.
        else:
            pv_used = vt
            pv_spill = 0.
            outstanding = required-vt
        if qt >= outstanding:
            called = outstanding
            uncalled = qt-outstanding
            emergency = 0.
        else:
            called = qt
            uncalled = 0.
            emergency = outstanding-qt
        emergency_charge_product = emergency*ct
        m0_max = max(m0_max, emergency_charge_product)
        if emergency_charge_product > tolerance:
            m0_count += 1
            mark("M0_emergency_charging", enforce_m0)
        purchase_fee = qt*pt
        emergency_fee = emergency*pt*float(emergency_multiplier)
        values = (qt, called, uncalled, pv_used, pv_spill, ct, bt, emergency,
                  before, state, purchase_fee, emergency_fee, purchase_fee+emergency_fee)
        for field, value in zip(LEDGER_FIELDS, values):
            expected[field].append(value)

    errors = {}
    reported = {}
    for field in LEDGER_FIELDS:
        if field not in ledger:
            mark("missing_ledger_field_"+field, True)
            errors[field] = None
            continue
        value = np.asarray(ledger[field], dtype=float)
        if value.shape != (n,) or not np.isfinite(value).all():
            mark("invalid_ledger_field_"+field, True)
            errors[field] = None
            continue
        reported[field] = value
        error = float(np.max(np.abs(value-np.asarray(expected[field]))))
        errors[field] = error
        mark("mismatch_"+field, error > tolerance)

    identities = {}
    if len(reported) == len(LEDGER_FIELDS):
        for name, residue in {
            "bus_balance_kwh": reported["a"]+reported["e"]+reported["v"]+reported["b"]-inputs["load"]-reported["c"],
            "plan_split_kwh": reported["q"]-reported["a"]-reported["u"],
            "pv_split_kwh": inputs["pv"]-reported["v"]-reported["w"],
            "soc_recursion_kwh": reported["soc_end"]-reported["soc_start"]-eta_charge*reported["c"]+reported["b"]/eta_discharge,
            "cash_sum_yuan": reported["cash_cost"]-reported["plan_cost"]-reported["emergency_cost"],
            "soc_continuity_kwh": np.r_[reported["soc_start"][0]-initial_soc,
                                         reported["soc_start"][1:]-reported["soc_end"][:-1]],
        }.items():
            identities[name] = float(np.max(np.abs(residue)))
            mark("identity_"+name, identities[name] > tolerance)
    return {
        "passed": not violations, "n_periods": n, "tolerance": float(tolerance),
        "violations": violations, "max_error_by_field": errors,
        "max_identity_error": identities,
        "max_absolute_error": max([0.]+[v for v in errors.values() if v is not None]+list(identities.values())),
        "m0_enforced": bool(enforce_m0), "m0_violating_periods": m0_count,
        "m0_violating_fraction": m0_count/n, "m0_max_e_times_c_kwh2": m0_max,
        "totals": {name: float(sum(expected[name])) for name in
                   ("q", "a", "u", "v", "w", "c", "b", "e", "plan_cost", "emergency_cost", "cash_cost")},
        "final_soc_kwh": state,
    }
