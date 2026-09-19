"""Full-period perfect-information cash lower bound with common initial SOC."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from model.history import START_DATE, T, load_data
from model.planning import Physical
from model.settlement import settle_schedule

ROOT = Path(__file__).resolve().parents[1]

def formulate_oracle(load, pv, price, initial_soc, physical=Physical()):
    """Vectorized independent state-increment formulation; no Python row loop."""
    load, pv, price = (np.asarray(value, float) for value in (load, pv, price))
    if any(value.ndim != 1 or not len(value) or not np.isfinite(value).all()
           for value in (load, pv, price)):
        raise ValueError("Oracle requires finite nonempty one-dimensional inputs")
    if load.shape != pv.shape or load.shape != price.shape:
        raise ValueError("Oracle input shapes differ")
    if np.min(load) < 0 or np.min(pv) < 0 or np.min(price) <= 0:
        raise ValueError("Oracle requires nonnegative energies and positive prices")
    if physical.emergency_multiple <= 1:
        raise ValueError("Eliminating emergency purchases requires a multiplier above one")
    if not physical.s_min <= initial_soc <= physical.s_max:
        raise ValueError("Initial SOC outside physical limits")
    n = len(load)
    t = np.arange(n)
    state = n+t
    successor = state+1
    rows = np.concatenate([t, t, t, n+t, n+t, n+t,
                           2*n+t, 2*n+t, 3*n+t, 3*n+t])
    cols = np.concatenate([t, successor, state, t, successor, state,
                           successor, state, state, successor])
    values = np.concatenate([-np.ones(n), np.full(n, 1/physical.eta_c), np.full(n, -1/physical.eta_c),
                             -np.ones(n), np.full(n, physical.eta_d), np.full(n, -physical.eta_d),
                             np.ones(n), -np.ones(n), np.ones(n), -np.ones(n)])
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(4*n, 2*n+1)).tocsr()
    rhs = np.concatenate([pv-load, pv-load, np.full(n, physical.eta_c*physical.power_energy),
                          np.minimum(load, physical.power_energy)/physical.eta_d])
    lower = np.r_[np.zeros(n), np.full(n+1, physical.s_min)]
    upper = np.r_[np.full(n, np.inf), np.full(n+1, physical.s_max)]
    lower[n] = upper[n] = initial_soc
    return dict(matrix=matrix, rhs=rhs, lower=lower, upper=upper,
                objective=np.r_[price, np.zeros(n+1)], n=n)

def solve_oracle(load, pv, price, initial_soc, *, terminal_soc=None,
                 physical=Physical(), problem=None, time_limit=None):
    problem = formulate_oracle(load, pv, price, initial_soc, physical) if problem is None else problem
    n = problem["n"]
    lower, upper = problem["lower"].copy(), problem["upper"].copy()
    if terminal_soc is not None:
        if not physical.s_min-1e-6 <= terminal_soc <= physical.s_max+1e-6:
            raise ValueError("Matched terminal SOC outside physical tolerance")
        lower[-1] = upper[-1] = float(terminal_soc)
    options = {} if time_limit is None else {"time_limit": float(time_limit)}
    start = perf_counter()
    result = linprog(problem["objective"], A_ub=problem["matrix"], b_ub=problem["rhs"],
                     bounds=np.c_[lower, upper], method="highs", options=options)
    elapsed = perf_counter()-start
    if not result.success:
        raise RuntimeError(f"Full-period oracle failed: {result.status} {result.message}")
    stock = result.x[n:]
    increment = np.diff(stock)
    charge = np.maximum(increment, 0)/physical.eta_c
    discharge = np.maximum(-increment, 0)*physical.eta_d
    q = result.x[:n]
    ledger = settle_schedule(q, charge, discharge, load, pv, initial_soc, price,
                              eta_charge=physical.eta_c, eta_discharge=physical.eta_d,
                              soc_min=physical.s_min, soc_max=physical.s_max,
                              bus_limit=physical.power_energy,
                              emergency_multiplier=physical.emergency_multiple, enforce_m0=False)
    cash = float(np.sum(ledger["cash_cost"]))
    if abs(cash-result.fun) > 1e-5:
        raise AssertionError("Oracle settlement cash differs from the solver objective")
    if float(np.max(np.abs(ledger["soc_end"]-stock[1:]))) > 1e-6:
        raise AssertionError("Oracle state reconstruction differs from solver states")
    if terminal_soc is not None and abs(float(stock[-1])-terminal_soc) > 1e-6:
        raise AssertionError("Oracle did not satisfy the matched terminal equality")
    equality_target_error = None if terminal_soc is None else abs(float(stock[-1])-terminal_soc)
    # Full dual reconstruction supplies a separate optimality certificate.
    dual = float(problem["rhs"] @ result.ineqlin.marginals + lower @ result.lower.marginals)
    finite = np.isfinite(upper)
    dual += float(upper[finite] @ result.upper.marginals[finite])
    primal_violation = float(max(0., np.max(problem["matrix"] @ result.x-problem["rhs"]),
                                  np.max(lower-result.x), np.max(result.x-upper)))
    if primal_violation > 1e-6 or abs(float(result.fun)-dual) > max(1e-5, abs(float(result.fun))*1e-9):
        raise AssertionError("Oracle primal/dual certificate failed")
    info = dict(status="OPTIMAL", solver_status=int(result.status), solver_message=result.message,
                objective_yuan=float(result.fun), cash_cost_yuan=cash,
                dual_objective_yuan=dual, primal_dual_gap_yuan=float(result.fun)-dual,
                max_primal_violation=primal_violation, seconds=elapsed,
                iterations=int(result.nit), variables=2*n+1, inequalities=4*n,
                sparse_nonzeros=int(problem["matrix"].nnz),
                initial_soc_kwh=float(initial_soc), terminal_soc_kwh=float(stock[-1]),
                requested_terminal_soc_kwh=terminal_soc, terminal_equality_error_kwh=equality_target_error,
                terminal_constraint="free within physical bounds" if terminal_soc is None else "exact full-period equality",
                emergency_kwh=float(ledger["e"].sum()),
                daily_cyclic_constraints=0,
                e_elimination_proof="With free normal q and positive tariff, replace (q,e) by (q+e,0); cash decreases by (m-1)*p*e. Therefore an optimum has e=0, and M0 holds.")
    return ledger, info
