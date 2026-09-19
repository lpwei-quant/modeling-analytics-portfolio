"""Past-information planning and real-time control for the corrected Q2 run."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from model.history import ForecastArchive
from model.planning import Block, Physical, Plan, solve, solve_state_lp
from model.settlement import settle_schedule, independent_recalculator

@dataclass(frozen=True)
class Config:
    name: str
    plan_kind: str = 'S'
    controller: str = 'fast'
    terminal: str = 'value'
    scenario_count: int = 30
    control_scenario_count: int = 10
    scenario_method: str = 'residual'
    lag_anchor: str = 'corrected'
    physical_guard: bool = False
    flat_planning_price: bool = False
    no_battery: bool = False
    strict_delivery: bool = False
    mip_time_limit: float = 120.0
    mip_gap: float = 1e-8

PRIMARY_CONFIGS = {
    'D': Config('D',plan_kind='D',controller='fixed',physical_guard=True),
    'S': Config('S',controller='fixed',physical_guard=True),
    'F_fast': Config('F_fast'),
    'F_lagged': Config('F_lagged',controller='lagged'),
    'F_lagged_guard': Config('F_lagged_guard',controller='lagged',physical_guard=True),
    'No_storage': Config('No_storage',controller='fixed',no_battery=True,physical_guard=True),
}

def continuation_value(day: int, stock: float, archive: ForecastArchive, physical=Physical()):
    """Derived local continuation slope, using two-day deterministic LP only."""
    p=archive.data.price
    l0,v0=archive.point(day);l1,v1=archive.point(day,day+1)
    plan=solve([Block(l0,v0,p),Block(l1,v1,p)],stock,
               terminal='cyclic',no_battery_dump=True,physical=physical)
    return plan.lam,plan.end_target,dict(source='two_day_deterministic_LP_link_dual',
              lambda_yuan_per_internal_kwh=plan.lam,target_kwh=plan.end_target,
              forecast_origin_index=day,future_target_index=day+1,
              latest_observed_day=day-1,objective=plan.objective,seconds=plan.seconds)

def make_plan(day: int, stock: float, archive: ForecastArchive, config: Config,
              physical=Physical()):
    """Return a sealed day-ahead plan.  Never accepts current actual L/PV."""
    if config.strict_delivery:
        raise NotImplementedError('Corrected strict delivery is not a primary configuration; reference replay is labeled separately')
    if config.flat_planning_price:
        price=np.full(144,float(archive.data.price.mean()))
    else:price=archive.data.price
    lh,vh=archive.point(day)
    if config.plan_kind=='D':
        loads,pvs=lh[None,:],vh[None,:]
        scenario_meta={'method':'point','effective_history_days':None,'latest_observed_day':day-1}
    elif config.plan_kind=='S':
        loads,pvs,scenario_meta=archive.scenarios(day,config.scenario_count,config.scenario_method)
    else:raise ValueError(config.plan_kind)
    lam,target,dual_meta=continuation_value(day,stock,archive,physical)
    term=config.terminal
    if term=='free': lam=0.;target=None
    elif term=='cyclic': target=stock
    elif term!='value':raise ValueError(term)
    tic=perf_counter()
    if config.no_battery:
        # Exact minimizers of nonnegative empirical newsvendor loss.  The
        # lower 80th percentile is a valid choice within an optimal interval.
        net=loads-pvs
        q=np.maximum(0.,np.quantile(net,.8,axis=0,method='inverted_cdf'))
        c=np.zeros(144);b=np.zeros(144);s=np.full(144,stock)
        expectation=float(price@q+physical.emergency_multiple*price@np.maximum(net-q,0).mean(axis=0))
        plan=Plan(q,c,b,s,expectation,expectation,0.,stock,perf_counter()-tic,
                  {'formulation':'exact_no_storage_empirical_quantile','m0_enforced':False,
                   'scenario_count':len(loads),'variables':144,'constraints':0})
    else:
        r=solve_state_lp(loads,pvs,price,stock,terminal=term,target=target,lam=lam,
                         physical=physical,no_battery_dump=True)
        e=r['e'];c=r['c'];b=r['b'];s=r['soc'][1:]
        cash=float(price@r['q']+physical.emergency_multiple*price@e.mean(axis=0))
        plan=Plan(r['q'],c,b,s,r['objective'],cash,lam,float(s[-1]),r['seconds'],
                  {'formulation':'state_increment_exact_LP','m0_enforced':False,
                   'scenario_count':len(loads),'variables':len(r['result'].x),
                   'max_charge_discharge_product_kwh2':float(np.max(c*b)),
                   'emergency_charge_scenario_slots':int(np.count_nonzero((e>1e-6)&(c[None,:]>1e-6)))})
    plan.lam=lam
    # The terminal credit function is part of the frozen model. Its cap is
    # NOT the SOC that one particular day-ahead optimum happens to achieve.
    # Preserve the H2-derived cap (or cyclic floor) for every intraday LP.
    optimized_end_soc=float(plan.soc[-1])
    plan.end_target=float(target) if target is not None else physical.s_max
    meta={'day':day,'config':asdict(config),'initial_soc':stock,
          'forecast_origin_index':day,'max_actual_day_used':day-1,
          'scenario_metadata':scenario_meta,'continuation':dual_meta,
          'objective':plan.objective,'expected_cash':plan.expected_cash,
          'optimized_plan_end_soc':optimized_end_soc,'controller_terminal_target':plan.end_target,
          'plan_seconds':plan.seconds,'diagnostics':plan.diagnostics}
    return plan,meta

def protect_intent(c: float,b: float,stock: float,q: float,load: float,pv: float,
                   physical=Physical()):
    """Explicit common physical protection for fixed intents, never hidden."""
    x=physical.eta_c*c-b/physical.eta_d
    intended_c=max(0.,x)/physical.eta_c;intended_b=max(0.,-x)*physical.eta_d
    if x>=0:
        cap=max(0.,min(physical.power_energy,(physical.s_max-stock)/physical.eta_c))
        actual_c=min(intended_c,cap);actual_b=0.
    else:
        cap=max(0.,min(physical.power_energy,(stock-physical.s_min)*physical.eta_d,
                      load-pv-q))
        actual_c=0.;actual_b=min(intended_b,cap)
    delta=abs(actual_c-c)+abs(actual_b-b)
    return actual_c,actual_b,{'guard_applied':delta>1e-6,'guard_delta_bus_kwh':delta,
                           'charge_power_soc_reduction_kwh':max(0.,intended_c-actual_c),
                           'unneeded_discharge_reduction_kwh':max(0.,intended_b-actual_b)}

def feedback_action(day: int,t: int,stock: float,plan: Plan,archive: ForecastArchive,
                    observed_load,observed_pv,config: Config,physical=Physical()):
    """One remaining-horizon LP. Receives ONLY its allowed observation prefix."""
    fast=config.controller=='fast'
    if config.controller not in ('fast','lagged'):raise ValueError(config.controller)
    loads,pvs,meta=archive.conditional(day,t,observed_load,observed_pv,
                     observe_current=fast,K=config.control_scenario_count,
                     lag_anchor=config.lag_anchor,method=config.scenario_method)
    p=archive.data.price[t:]
    if config.flat_planning_price:p=np.full_like(p,float(archive.data.price.mean()))
    # Avoid gratuitous discharge in the current sample set. For fast, all
    # current samples are the actual measurement, so this is an exact limit.
    discharge_cap=max(0.,float(np.min(loads[:,0]-pvs[:,0]-plan.q[t])))
    r=solve_state_lp(loads,pvs,p,stock,fixed_q=plan.q[t:],terminal=config.terminal,
                     target=plan.end_target,lam=plan.lam,physical=physical,
                     no_battery_dump=True,enforce_fixed_q_m0=False,
                     first_discharge_cap=discharge_cap)
    c,b=float(r['c'][0]),float(r['b'][0])
    diagnostic={'t':t,'observed_slots':len(observed_load),'lp_objective':r['objective'],
                'lp_seconds':r['seconds'],'sample_count':len(loads),
                'current_load_min':float(loads[:,0].min()),'current_load_max':float(loads[:,0].max()),
                'current_pv_min':float(pvs[:,0].min()),'current_pv_max':float(pvs[:,0].max()),
                'planned_end_soc':plan.end_target,'continuation_lambda':plan.lam,
                'metadata':meta,'physical_projection_count':0}
    return c,b,diagnostic

def execute_day(day: int,initial_soc: float,plan: Plan,archive: ForecastArchive,
                config: Config,physical=Physical(), *, require_feasible=True):
    """Reveals the current row only after the caller has saved the plan."""
    load=archive.data.load[day];pv=archive.data.pv[day];price=archive.data.price
    c=np.empty(144);b=np.empty(144);proposed_c=np.empty(144);proposed_b=np.empty(144)
    stock=initial_soc;details=[];guard_count=0;guard_energy=0.
    tic=perf_counter()
    for t in range(144):
        if config.controller=='fixed':
            ct,bt=float(plan.c[t]),float(plan.b[t]);item={'t':t,'observed_slots':0,'lp_seconds':0.}
        else:
            prefix=t+int(config.controller=='fast')
            ct,bt,item=feedback_action(day,t,stock,plan,archive,load[:prefix],pv[:prefix],config,physical)
        proposed_c[t],proposed_b[t]=ct,bt
        if config.physical_guard:
            ct,bt,guard=protect_intent(ct,bt,stock,float(plan.q[t]),float(load[t]),float(pv[t]),physical)
            item.update(guard);guard_count+=int(guard['guard_applied']);guard_energy+=guard['guard_delta_bus_kwh']
        else:item.update(guard_applied=False,guard_delta_bus_kwh=0.)
        # This is a physical feasibility audit, not an action modification.
        try:
            one=settle_schedule([plan.q[t]],[ct],[bt],[load[t]],[pv[t]],stock,[price[t]],
                eta_charge=physical.eta_c,eta_discharge=physical.eta_d,soc_min=physical.s_min,
                soc_max=physical.s_max,bus_limit=physical.power_energy,
                emergency_multiplier=physical.emergency_multiple,enforce_m0=False)
        except ValueError as exc:
            failure={'day':day,'t':t,'config':asdict(config),'reason':str(exc),
                     'soc_before':stock,'q':float(plan.q[t]),'c':ct,'b':bt,
                     'actual_load':float(load[t]),'actual_pv':float(pv[t]),
                     'proposed_c':float(proposed_c[t]),'proposed_b':float(proposed_b[t]),
                     'completed_charge_actions':c[:t].tolist(),
                     'completed_discharge_actions':b[:t].tolist(),
                     'observation_count':len(load[:t+int(config.controller=='fast')]),
                     'controller_diagnostics':item,'previous_step_details':details}
            error=RuntimeError('Strict runtime physical infeasibility')
            error.failure_record=failure
            raise error from exc
        c[t],b[t]=ct,bt;stock=float(one['soc_end'][0]);details.append(item)
    ledger=settle_schedule(plan.q,c,b,load,pv,initial_soc,price,
                eta_charge=physical.eta_c,eta_discharge=physical.eta_d,soc_min=physical.s_min,
                soc_max=physical.s_max,bus_limit=physical.power_energy,
                emergency_multiplier=physical.emergency_multiple,enforce_m0=False)
    recalc=independent_recalculator(plan.q,c,b,load,pv,initial_soc,price,ledger,
                eta_charge=physical.eta_c,eta_discharge=physical.eta_d,soc_min=physical.s_min,
                soc_max=physical.s_max,bus_limit=physical.power_energy,
                emergency_multiplier=physical.emergency_multiple,enforce_m0=False)
    if not recalc['passed']:raise AssertionError(recalc)
    return ledger,dict(steps=details,proposed_c=proposed_c,proposed_b=proposed_b,
                      independent_recalculation=recalc,execution_seconds=perf_counter()-tic,
                      guard_count=guard_count,guard_energy_kwh=guard_energy,
                      feedback_projection_count=0 if not config.physical_guard else guard_count)
