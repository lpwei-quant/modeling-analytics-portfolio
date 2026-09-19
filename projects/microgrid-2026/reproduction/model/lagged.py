"""Past-only reference LP; current observations enter execution protection only."""
from time import perf_counter

import numpy as np

from model.planning import Physical, solve_state_lp
from model.control import Config, make_plan, protect_intent
from model.settlement import settle_schedule, independent_recalculator

PRIMARY_CONFIGS = {
    'F_lagged_no_min_guard': Config('F_lagged_no_min_guard', controller='lagged', physical_guard=True),
}

def feedback_action(day, t, stock, plan, archive, observed_load, observed_pv,
                    config, physical=Physical()):
    if config.controller != 'lagged' or not config.physical_guard:
        raise ValueError('This experiment requires lagged planning and measured protection')
    loads, pvs, meta = archive.conditional(
        day, t, observed_load, observed_pv, observe_current=False,
        K=config.control_scenario_count, lag_anchor=config.lag_anchor,
        method=config.scenario_method)
    price = archive.data.price[t:]
    result = solve_state_lp(loads, pvs, price, stock, fixed_q=plan.q[t:],
        terminal=config.terminal, target=plan.end_target, lam=plan.lam,
        physical=physical, no_battery_dump=True, enforce_fixed_q_m0=False,
        first_discharge_cap=None)
    diagnostic = dict(t=t, observed_slots=len(observed_load),
        lp_objective=result['objective'], lp_seconds=result['seconds'],
        sample_count=len(loads), extra_first_discharge_bound=False,
        old_bound_for_diagnostic=max(0., float(np.min(loads[:,0]-pvs[:,0]-plan.q[t]))),
        metadata=meta, physical_projection_count=0)
    return float(result['c'][0]), float(result['b'][0]), diagnostic

def execute_day(day, initial_soc, plan, archive, config, physical=Physical(), *, require_feasible=True):
    load, pv, price = archive.data.load[day], archive.data.pv[day], archive.data.price
    n = len(price)
    c, b, pc, pb = (np.empty(n) for _ in range(4))
    stock = float(initial_soc)
    details, guards, guard_energy = [], 0, 0.
    start = perf_counter()
    for t in range(n):
        ct, bt, item = feedback_action(day, t, stock, plan, archive,
            load[:t], pv[:t], config, physical)
        pc[t], pb[t] = ct, bt
        ct, bt, guard = protect_intent(ct, bt, stock, plan.q[t], load[t], pv[t], physical)
        item.update(guard)
        item['physical_projection_count'] = int(guard['guard_applied'])
        guards += int(guard['guard_applied'])
        guard_energy += guard['guard_delta_bus_kwh']
        one = settle_schedule([plan.q[t]], [ct], [bt], [load[t]], [pv[t]], stock, [price[t]],
            eta_charge=physical.eta_c, eta_discharge=physical.eta_d,
            soc_min=physical.s_min, soc_max=physical.s_max, bus_limit=physical.power_energy,
            emergency_multiplier=physical.emergency_multiple, enforce_m0=False)
        c[t], b[t], stock = ct, bt, float(one['soc_end'][0])
        details.append(item)
    ledger = settle_schedule(plan.q, c, b, load, pv, initial_soc, price,
        eta_charge=physical.eta_c, eta_discharge=physical.eta_d,
        soc_min=physical.s_min, soc_max=physical.s_max, bus_limit=physical.power_energy,
        emergency_multiplier=physical.emergency_multiple, enforce_m0=False)
    independent = independent_recalculator(plan.q, c, b, load, pv, initial_soc, price, ledger,
        eta_charge=physical.eta_c, eta_discharge=physical.eta_d,
        soc_min=physical.s_min, soc_max=physical.s_max, bus_limit=physical.power_energy,
        emergency_multiplier=physical.emergency_multiple, enforce_m0=False)
    if not independent['passed']:
        raise AssertionError(independent)
    return ledger, dict(steps=details, proposed_c=pc, proposed_b=pb,
        independent_recalculation=independent, execution_seconds=perf_counter()-start,
        guard_count=guards, guard_energy_kwh=guard_energy,
        feedback_projection_count=guards,
        observation_contract='reference prefix [0,t); measured protection uses current aggregate')
