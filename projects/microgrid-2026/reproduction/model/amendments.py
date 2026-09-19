"""One-signal planning and causal rolling final-contract execution for Q3."""
from dataclasses import asdict,dataclass
from time import perf_counter

import numpy as np

from model.history import ar1_coef
from model.planning import Block, Physical, solve
from model.control import protect_intent
from model.contracts import solve_shared, solve_signal_recourse
from model.contract_cash import settle, independent_recalculate

@dataclass(frozen=True)
class Config:
    name: str
    zero_model: str='Z1'
    enabled_hours: tuple=(6,12,18)
    amend: bool=True
    fee: str='B'
    controller: str='lagged'
    terminal18: bool=False
    scenario_count: int=30
    control_scenario_count: int=10

    def __post_init__(self):
        if self.zero_model not in ('Z1','Z2') or self.fee not in ('A','B'):
            raise ValueError('Invalid model or fee')
        if self.controller not in ('lagged','fast') or tuple(sorted(set(self.enabled_hours)))!=self.enabled_hours:
            raise ValueError('Invalid controller or publication sequence')
        if any(h not in (6,12,18) for h in self.enabled_hours):raise ValueError('Invalid issue hour')
        if self.terminal18 and 18 not in self.enabled_hours:raise ValueError('18 refresh requires 18 publication')

@dataclass(frozen=True)
class PriceInputs:
    """Day-indexed decision forecasts are distinct from realized settlement."""
    decision: np.ndarray
    settlement: np.ndarray
    next_day_forecast: np.ndarray
    provenance: str

    def __post_init__(self):
        arrays=[np.asarray(v,float) for v in (self.decision,self.settlement,self.next_day_forecast)]
        if any(v.shape!=(365,144) or not np.isfinite(v).all() or v.min()<=0 for v in arrays):
            raise ValueError('Each price input must be positive finite day x 144')
        for key,value in zip(('decision','settlement','next_day_forecast'),arrays):
            value=value.copy();value.setflags(write=False);object.__setattr__(self,key,value)

    @classmethod
    def q3(cls,data):
        matrix=np.tile(data.price,(365,1))
        return cls(matrix,matrix,matrix,'Observed attachment1 deterministic delivery-period tariff')

@dataclass
class DayPlan:
    q: np.ndarray
    c: np.ndarray
    b: np.ndarray
    soc: np.ndarray
    lam: float
    target: float
    initial_soc: float
    h2_load: np.ndarray
    h2_pv: np.ndarray
    h2_price: np.ndarray
    seconds: float
    metadata: dict

def effective_blocks(enabled_hours):
    starts=[h*6 for h in enabled_hours]
    return {a:b for a,b in zip(starts,starts[1:]+[144])}

def continuation_from_frozen(load,pv,price,s0,physical=Physical()):
    p=solve([Block(load[0],pv[0],price[0]),Block(load[1],pv[1],price[1])],
        s0,terminal='cyclic',no_battery_dump=True,physical=physical)
    return p.lam,p.end_target,dict(source='two-day deterministic link dual',
        lam=p.lam,cap=p.end_target,objective=p.objective,seconds=p.seconds)

def make_plan(day,stock,archive,prices,config,physical=Physical()):
    start=perf_counter();lh,vh,point_meta=archive.point(day,0)
    nextload,nextpv=archive.load_archive.point(day,day+1)
    h2load=np.stack((lh,nextload));h2pv=np.stack((vh,nextpv))
    h2price=np.stack((prices.decision[day],prices.next_day_forecast[day]))
    lam,target,dual=continuation_from_frozen(h2load,h2pv,h2price,stock,physical)
    loads,pvs,smeta=archive.scenarios(day,0,config.scenario_count)
    if len(loads)==0:
        loads,pvs=lh[None,:],vh[None,:]
        smeta.update(cold_start_fallback='Assumed deterministic point; no empirical residual invented')
    use_signal=config.zero_model=='Z2' and 6 in config.enabled_hours
    if use_signal:
        pairs=archive.z2_signal_pairs(day,config.scenario_count)
        groups=pairs['groups'] if len(pairs['groups']) else np.zeros(1,int)
        result=solve_signal_recourse(loads,pvs,prices.decision[day],stock,groups,
            allow_amend=config.amend,fee=config.fee,lam=lam,target=target,physical=physical)
        c=result['group_weights']@result['c_by_group'];b=result['group_weights']@result['b_by_group']
        soc=result['group_weights']@result['soc_by_group']
        extra=dict(signal=pairs['metadata'],signal_thresholds=pairs['thresholds'],
            group_weights=result['group_weights'],pre_signal_state_error=result['pre_signal_state_error'],
            hypothetical_group_contracts=result['m_by_group'],
            hypothetical_group_soc=result['soc_by_group'])
    else:
        result=solve_shared(loads,pvs,prices.decision[day],stock,lam=lam,target=target,physical=physical)
        c,b,soc=result['c'],result['b'],result['soc'];extra={}
    meta=dict(day=day,initial_soc=stock,config=asdict(config),point=point_meta,
        scenarios=smeta,continuation=dual,objective=result['objective'],
        expected_cash=result['expected_cost'],terminal_credit=result['terminal_credit'],
        effective_zero_model='Z2' if use_signal else 'Z1',
        q_latest_actual_slot=day*144-1,max_publication_hour_used=0,
        day_ahead_actions_are_reference_only=True,
        z2_contract_candidates_not_yet_committed=use_signal,**extra)
    return DayPlan(result['q'].copy(),c,b,soc,lam,target,stock,h2load,h2pv,h2price,
        perf_counter()-start,meta)

def terminal_refresh18(day,plan,archive,physical=Physical(),replace_pv=True):
    """Freeze the 00 H2 problem; substitute only tomorrow's published PV part."""
    _,vp,meta=archive.point(day,18,as_of_abs_slot=day*144+108)
    changed=plan.h2_pv.copy()
    if replace_pv:changed[1,:108]=vp[36:144]
    lam,target,dual=continuation_from_frozen(plan.h2_load,changed,plan.h2_price,
        plan.initial_soc,physical)
    dual.update(applied_from_slot=108,replace_tomorrow_pv=replace_pv,
        frozen_day_zero_initial_soc=plan.initial_soc,
        replacement_nextday_slots=[0,108],publication=meta['published_at'],
        changed_pv_l1=float(np.abs(changed-plan.h2_pv).sum()),
        interpretation='terminal approximation refresh, not realized-state continuation value')
    return lam,target,dual

class ConditionalArchive:
    def __init__(self,archive):self.archive=archive;self._base={}

    def base(self,day,hour,K):
        key=(day,hour,K)
        if key not in self._base:
            lh,vh,_=self.archive.point(day,hour)
            el,ev,indices,meta=self.archive.residuals(day,hour,30)
            rl,rv=(ar1_coef(el),ar1_coef(ev)) if len(el) else (0.,0.)
            order=np.arange(len(el));seed=day+365*hour
            if len(order)>K:order=np.random.default_rng(seed).choice(len(order),K,replace=False)
            if len(order):el,ev,indices=el[order],ev[order],indices[order]
            else:
                el,ev=np.zeros((1,144)),np.zeros((1,144))
                meta.update(cold_start_fallback='Assumed zero-error point path')
            self._base[key]=(lh,vh,el,ev,rl,rv,indices,meta,seed)
        return self._base[key]

    def remaining(self,day,t,hour,observed_load,observed_pv,config):
        fast=config.controller=='fast';allowed=t+int(fast)
        ol,ov=np.asarray(observed_load,float),np.asarray(observed_pv,float)
        if ol.shape!=(allowed,) or ov.shape!=(allowed,) or not np.isfinite(ol).all() or not np.isfinite(ov).all():
            raise ValueError('Controller must receive exactly its allowed completed/current prefix')
        u=hour*6;i=t-u;n=144-t
        if i<0:raise ValueError('Unpublished forecast requested')
        lh,vh,el,ev,rl,rv,indices,meta,seed=self.base(day,hour,config.control_scenario_count)
        r=t if fast else t-1;ri=r-u
        if ri<0:
            loads=lh[i:i+n][None,:]+el[:,i:i+n]
            pvs=vh[i:i+n][None,:]+ev[:,i:i+n]
        else:
            steps=np.arange(t,144)-r;fl=rl**steps;fv=rv**steps
            loads=lh[i:i+n][None,:]+el[:,i:i+n]+fl[None,:]*(ol[r]-lh[ri]-el[:,[ri]])
            pvs=vh[i:i+n][None,:]+ev[:,i:i+n]+fv[None,:]*(ov[r]-vh[ri]-ev[:,[ri]])
        loads,pvs=np.maximum(loads,0),np.maximum(pvs,0)
        pvs[:,vh[i:i+n]<=1e-9]=0
        if fast:loads[:,0]=ol[-1];pvs[:,0]=ov[-1]
        return loads,pvs,dict(issue_hour=hour,observed_slots=allowed,
            last_observed_slot=allowed-1,residual_anchor_slot=r if ri>=0 else None,
            historical_days=indices.tolist(),rho_load=rl,rho_pv=rv,seed=seed,
            historical_target_latest_abs_slot=meta['latest_target_actual_abs_slot'],
            sample_count=len(loads),reference_uses_current=fast)

def amendment(day,t,stop,stock,q,archive,prices,config,lam,target,physical=Physical()):
    """Return an effective current block and explicitly uncommitted candidates."""
    hour=t//6;loads,pvs,meta=archive.scenarios(day,hour,config.scenario_count)
    if len(loads)==0:
        lh,vh,_=archive.point(day,hour);loads,pvs=lh[None,:],vh[None,:]
        meta.update(cold_start_fallback='Assumed point path')
    r=solve_shared(loads[:,:144-t],pvs[:,:144-t],prices.decision[day,t:],stock,
        original_q=q[t:],fee=config.fee,lam=lam,target=target,physical=physical)
    return r['m'][:stop-t].copy(),dict(issue_hour=hour,committed_start=t,
        committed_stop=stop,candidate_start=stop,candidate_m=r['m'][stop-t:],
        committed_m=r['m'][:stop-t],lp_objective=r['objective'],
        expected_remaining_contract_cost=float(r['contract_cost'].sum()),
        seconds=r['seconds'],forecast_metadata=meta,lam=lam,target=target,
        original_plan_not_mutated=True)

def execute_day(day,stock,plan,archive,prices,config,physical=Physical(),conditional=None):
    start=perf_counter();initial=float(stock)
    conditional=conditional or ConditionalArchive(archive)
    load,pv=archive.data.load[day],archive.data.pv[day]
    q=plan.q;m=q.copy();c=np.empty(144);b=np.empty(144)
    pc=np.empty(144);pb=np.empty(144);steps=[];amendments=[];refresh=None
    blocks=effective_blocks(config.enabled_hours);hour=0;lam=plan.lam;target=plan.target
    guards=0;guard_energy=0.
    for t in range(144):
        if t in blocks:
            hour=t//6
            if t==108 and config.terminal18:
                lam,target,refresh=terminal_refresh18(day,plan,archive,physical)
            if config.amend:
                current,record=amendment(day,t,blocks[t],stock,q,archive,prices,config,lam,target,physical)
                m[t:blocks[t]]=current;amendments.append(record)
        prefix=t+int(config.controller=='fast')
        ls,vs,info=conditional.remaining(day,t,hour,load[:prefix],pv[:prefix],config)
        cap=max(0.,float(load[t]-pv[t]-m[t])) if config.controller=='fast' else None
        result=solve_shared(ls,vs,prices.decision[day,t:],stock,fixed_m=m[t:],
            lam=lam,target=target,physical=physical,first_discharge_cap=cap)
        pc[t],pb[t]=result['c'][0],result['b'][0]
        ct,bt,guard=protect_intent(pc[t],pb[t],stock,m[t],load[t],pv[t],physical)
        c[t],b[t]=ct,bt;stock+=physical.eta_c*ct-bt/physical.eta_d
        guards+=int(guard['guard_applied']);guard_energy+=guard['guard_delta_bus_kwh']
        steps.append(dict(t=t,lp_seconds=result['seconds'],lp_objective=result['objective'],
            lam=lam,target=target,reference=info,**guard))
    ledger=settle(q,m,c,b,load,pv,initial,prices.settlement[day],config.fee,physical)
    audit=independent_recalculate(q,m,c,b,load,pv,initial,prices.settlement[day],ledger,config.fee,physical)
    if not audit['passed']:raise AssertionError(audit)
    if not config.amend and np.max(np.abs(m-q))>1e-9:raise AssertionError('Information-only changed contract')
    return ledger,dict(steps=steps,amendments=amendments,terminal18=refresh,
        proposed_c=pc,proposed_b=pb,guard_count=guards,guard_energy_kwh=guard_energy,
        independent_recalculation=audit,execution_seconds=perf_counter()-start,
        current_effective_contracts_only_in_control=True,
        future_candidate_m_is_not_assumed_committed=True)

