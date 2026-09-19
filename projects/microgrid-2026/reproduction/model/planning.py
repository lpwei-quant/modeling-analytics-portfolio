"""Sparse Q2 planning models, explicit battery formulation and state LP."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

@dataclass(frozen=True)
class Physical:
    eta_c: float = 0.9
    eta_d: float = 0.9
    s_min: float = 1200.0
    s_max: float = 10800.0
    power_energy: float = 5000.0/6.0
    emergency_multiple: float = 5.0

@dataclass
class Block:
    load: np.ndarray
    pv: np.ndarray
    price: np.ndarray

    def __post_init__(self):
        self.load = np.atleast_2d(np.asarray(self.load, float))
        self.pv = np.atleast_2d(np.asarray(self.pv, float))
        self.price = np.asarray(self.price, float)
        if self.load.shape != self.pv.shape or self.price.shape != (self.load.shape[1],):
            raise ValueError('Inconsistent scenario/price shapes')
        if not np.isfinite(self.load).all() or not np.isfinite(self.pv).all():
            raise ValueError('Nonfinite scenario')
        if np.min(self.load) < 0 or np.min(self.pv) < 0 or np.min(self.price) <= 0:
            raise ValueError('Nonphysical input or nonpositive price')

@dataclass
class Plan:
    q: np.ndarray
    c: np.ndarray
    b: np.ndarray
    soc: np.ndarray
    objective: float
    expected_cash: float
    lam: float
    end_target: float
    seconds: float
    diagnostics: dict
    result: object = None
    formulation: dict | None = None

@lru_cache(maxsize=450)
def _layout(shapes: tuple, eta_c: float, eta_d: float, has_value: bool):
    """Matrix topology depends on shapes only; scenario values go in RHS."""
    offsets=[]; cursor=0
    for k,n in shapes:
        offsets.append(dict(q=cursor,c=cursor+n,b=cursor+2*n,s=cursor+3*n,
                            e=cursor+4*n,r=cursor+4*n+k*n))
        cursor += 4*n+2*k*n
    v_idx=cursor if has_value else None
    nv=cursor+int(has_value)
    rowparts=[]; colparts=[]; valparts=[]; rhs_slices=[]; link_rows=[]
    row=0; previous=None
    for (k,n),o in zip(shapes, offsets):
        t=np.arange(n)
        link_rows.append(row)
        # S_t - eta_c*c_t + b_t/eta_d - S_(t-1) = 0.
        rowparts.extend([row+t,row+t,row+t])
        colparts.extend([o['s']+t,o['c']+t,o['b']+t])
        valparts.extend([np.ones(n),np.full(n,-eta_c),np.full(n,1/eta_d)])
        if n>1:
            rowparts.append(row+t[1:]); colparts.append(o['s']+t[:-1]); valparts.append(-np.ones(n-1))
        if previous is not None:
            rowparts.append(np.array([row]));colparts.append(np.array([previous]));valparts.append(np.array([-1.]))
        row+=n
        # q+b+e-c-r = load-pv, scenario-major indexing.
        times=np.tile(t,k); rids=row+np.arange(k*n)
        rowparts.extend([rids]*5)
        colparts.extend([o['q']+times,o['b']+times,o['e']+np.arange(k*n),
                         o['c']+times,o['r']+np.arange(k*n)])
        valparts.extend([np.ones(k*n),np.ones(k*n),np.ones(k*n),-np.ones(k*n),-np.ones(k*n)])
        rhs_slices.append(slice(row,row+k*n)); row+=k*n
        previous=o['s']+n-1
    mat=sparse.coo_matrix((np.concatenate(valparts),(np.concatenate(rowparts),np.concatenate(colparts))),shape=(row,nv)).tocsr()
    return offsets,nv,mat,tuple(rhs_slices),tuple(link_rows),previous,v_idx

def formulate(blocks: list[Block], s0: float, *, terminal='free', lam=0.,
              target=None, fixed_q=None, physical=Physical(), strict_delivery=False,
              no_battery_dump=False, first_charge_cap=None, first_discharge_cap=None,
              explicit_mutual=False, enforce_m0=False):
    """Build explicit LP, or a small-case MILP when mutual/M0 is requested."""
    if terminal not in ('free','cyclic','equal','value','floor'):
        raise ValueError(terminal)
    if not physical.s_min-1e-7 <= s0 <= physical.s_max+1e-7:
        raise ValueError('Initial storage outside bounds')
    shapes=tuple(b.load.shape for b in blocks)
    offsets,nv,eq,ranges,links,last,vid=_layout(shapes,physical.eta_c,physical.eta_d,terminal=='value')
    obj=np.zeros(nv);lo=np.zeros(nv);hi=np.full(nv,np.inf);rhs=np.zeros(eq.shape[0]);rhs[0]=s0
    for block,(k,n),o,rs in zip(blocks,shapes,offsets,ranges):
        obj[o['q']:o['q']+n]=block.price
        obj[o['e']:o['e']+k*n]=np.tile(physical.emergency_multiple*block.price/k,k)
        hi[o['c']:o['c']+n]=physical.power_energy
        hi[o['b']:o['b']+n]=physical.power_energy
        if no_battery_dump:
            hi[o['b']:o['b']+n]=np.minimum(physical.power_energy,block.load.min(axis=0))
        lo[o['s']:o['s']+n]=physical.s_min;hi[o['s']:o['s']+n]=physical.s_max
        if strict_delivery: hi[o['r']:o['r']+k*n]=block.pv.ravel()
        rhs[rs]=(block.load-block.pv).ravel()
    first=offsets[0];n0=shapes[0][1]
    if fixed_q is not None:
        q=np.asarray(fixed_q,float)
        if q.shape!=(n0,) or np.min(q)<-1e-7: raise ValueError('Invalid fixed q')
        lo[first['q']:first['q']+n0]=q;hi[first['q']:first['q']+n0]=q
    if first_charge_cap is not None:
        hi[first['c']]=min(physical.power_energy,max(0.,float(first_charge_cap)))
    if first_discharge_cap is not None:
        hi[first['b']]=min(physical.power_energy,max(0.,float(first_discharge_cap)))
    ur=[];uc=[];uv=[];ub=[]
    if terminal in ('cyclic','equal','floor'):
        level=s0 if target is None else float(target)
        if terminal=='equal': lo[last]=hi[last]=level
        else: ur=[0];uc=[last];uv=[-1.];ub=[-level]
    elif terminal=='value':
        if target is None: target=physical.s_max
        if lam<0: raise ValueError('Negative continuation credit')
        obj[vid]=-lam;hi[vid]=float(target)
        ur=[0,0];uc=[vid,last];uv=[1.,-1.];ub=[0.]
    integrality=None
    if explicit_mutual or enforce_m0:
        # A mode z_t=1 allows charging and forces discharge/emergency to zero.
        # M for each e uses a known per-scenario physical bound; no q cap.
        n_modes=sum(n for _,n in shapes);base_nv=nv;nv+=n_modes
        obj=np.r_[obj,np.zeros(n_modes)];lo=np.r_[lo,np.zeros(n_modes)];hi=np.r_[hi,np.ones(n_modes)]
        integrality=np.r_[np.zeros(base_nv,int),np.ones(n_modes,int)]
        eq=sparse.hstack([eq,sparse.csr_matrix((eq.shape[0],n_modes))],format='csr')
        iz=base_nv
        for block,(k,n),o in zip(blocks,shapes,offsets):
            for t in range(n):
                r=len(ub);ur.extend([r,r]);uc.extend([o['c']+t,iz+t]);uv.extend([1.,-physical.power_energy]);ub.append(0.)
                r=len(ub);ur.extend([r,r]);uc.extend([o['b']+t,iz+t]);uv.extend([1.,physical.power_energy]);ub.append(physical.power_energy)
                if enforce_m0:
                    # z=1 (charging): q-c >= max_s(load_s-pv_s).
                    # z=0: c=0 and the inequality reduces to q>=0.
                    # With positive emergency cost, this forces every e_s=0
                    # whenever c>0, without a large arbitrary energy bound.
                    worst_net=float(np.max(block.load[:,t]-block.pv[:,t]))
                    r=len(ub);ur.extend([r,r,r]);uc.extend([o['c']+t,o['q']+t,iz+t]);uv.extend([1.,-1.,worst_net]);ub.append(0.)
            iz+=n
    ineq=None if not ub else sparse.coo_matrix((uv,(ur,uc)),shape=(len(ub),nv)).tocsr()
    return dict(objective=obj,lower=lo,upper=hi,eq=eq,rhs=rhs,ub=ineq,
                ub_rhs=None if not ub else np.asarray(ub),integrality=integrality,
                offsets=offsets,shapes=shapes,link_rows=links,value_idx=vid,
                terminal=terminal,lam=lam,target=target,physical=physical)

def solve(blocks: list[Block], s0: float, *, retain_problem=False, **options) -> Plan:
    f=formulate(blocks,s0,**options);tic=perf_counter()
    if f['integrality'] is None:
        result=linprog(f['objective'],A_ub=f['ub'],b_ub=f['ub_rhs'],
                       A_eq=f['eq'],b_eq=f['rhs'],bounds=np.c_[f['lower'],f['upper']],method='highs')
    else:
        constraints=[LinearConstraint(f['eq'],f['rhs'],f['rhs'])]
        if f['ub'] is not None: constraints.append(LinearConstraint(f['ub'],-np.inf,f['ub_rhs']))
        result=milp(f['objective'],integrality=f['integrality'],bounds=Bounds(f['lower'],f['upper']),
                    constraints=constraints,options={'mip_rel_gap':1e-9,'time_limit':120.})
    elapsed=perf_counter()-tic
    if not result.success:
        raise RuntimeError(f"Q2 solve failed: status={result.status} {result.message}")
    o=f['offsets'][0];k,n=f['shapes'][0];x=result.x
    arrays={name:x[o[name]:o[name]+n].copy() for name in ('q','c','b','s')}
    credit=0. if f['value_idx'] is None else f['lam']*x[f['value_idx']]
    marginal=0.
    if f['integrality'] is None:
        if len(blocks)>1: marginal=max(0.,-float(result.eqlin.marginals[f['link_rows'][1]]))
        elif f['terminal'] in ('cyclic','floor'):
            marginal=max(0.,-float(result.ineqlin.marginals[0]))
        elif f['terminal']=='value': marginal=f['lam']
    e=x[o['e']:o['e']+k*n].reshape(k,n)
    diag={'variables':len(x),'equalities':len(f['rhs']),'scenario_count':k,
          'max_simultaneous_charge_discharge':float(np.max(np.minimum(arrays['c'],arrays['b']))),
          'emergency_and_charge_scenario_slots':int(np.sum((e>1e-6)&(arrays['c'][None,:]>1e-6))),
          'm0_enforced':f['integrality'] is not None and options.get('enforce_m0',False),
          'solver_status':int(result.status),'solver_message':result.message,
          'primal_equality_error':float(np.max(np.abs(f['eq']@x-f['rhs']))),
          'min_bound_slack':float(min(np.min(x-f['lower']),np.min(f['upper']-x)))}
    if f['integrality'] is not None:
        diag.update(mip_gap=float(result.mip_gap),mip_dual_bound=float(result.mip_dual_bound))
    return Plan(arrays['q'],arrays['c'],arrays['b'],arrays['s'],float(result.fun),
                float(result.fun+credit),marginal,float(arrays['s'][-1]),elapsed,diag,
                result if retain_problem else None,f if retain_problem else None)

def solve_state_lp(load, pv, price, s0, *, fixed_q=None, terminal='free',
                   target=None, lam=0., physical=Physical(), no_battery_dump=False,
                   enforce_fixed_q_m0=False, first_discharge_cap=None):
    """Independently formulated exact piecewise state-increment LP for M1."""
    load=np.atleast_2d(np.asarray(load,float));pv=np.atleast_2d(np.asarray(pv,float));price=np.asarray(price,float)
    k,n=load.shape
    if pv.shape!=load.shape or len(price)!=n: raise ValueError('Shapes')
    # All n+1 states are variables, unlike the explicit formulation.
    q0=0;sidx=n;eidx=2*n+1;vid=eidx+k*n if terminal=='value' else None
    nv=eidx+k*n+int(vid is not None)
    obj=np.zeros(nv);obj[:n]=price;obj[eidx:eidx+k*n]=np.tile(physical.emergency_multiple*price/k,k)
    lo=np.zeros(nv);hi=np.full(nv,np.inf)
    lo[sidx:sidx+n+1]=physical.s_min;hi[sidx:sidx+n+1]=physical.s_max
    lo[sidx]=hi[sidx]=s0
    if fixed_q is not None:lo[:n]=hi[:n]=np.asarray(fixed_q,float)
    if enforce_fixed_q_m0 and fixed_q is None:
        raise ValueError('The convex M0 charge bound requires q already fixed')
    if terminal in ('cyclic','floor','equal'):
        level=s0 if target is None else target
        lo[sidx+n]=level
        if terminal=='equal':hi[sidx+n]=level
    if vid is not None:
        hi[vid]=physical.s_max if target is None else target;obj[vid]=-lam
    rows=[];cols=[];vals=[];bounds=[]
    for coefficient in (1/physical.eta_c,physical.eta_d):
        for s in range(k):
            for t in range(n):
                r=len(bounds); rows.extend([r]*4);cols.extend([t,sidx+t+1,sidx+t,eidx+s*n+t])
                vals.extend([-1.,coefficient,-coefficient,-1.]);bounds.append(pv[s,t]-load[s,t])
    for t in range(n):
        charge_cap=physical.power_energy
        if enforce_fixed_q_m0:
            # For fixed q and a common scenario action this bound is exactly
            # c<=max(0,q-max_s(net_s)), hence e_s*c=0 for every sample.
            charge_cap=min(charge_cap,max(0.,float(fixed_q[t]-np.max(load[:,t]-pv[:,t]))))
        r=len(bounds);rows.extend([r,r]);cols.extend([sidx+t+1,sidx+t]);vals.extend([1.,-1.]);bounds.append(physical.eta_c*charge_cap)
        discharge=physical.power_energy
        if no_battery_dump:discharge=min(discharge,float(load[:,t].min()))
        if t==0 and first_discharge_cap is not None:
            discharge=min(discharge,max(0.,float(first_discharge_cap)))
        r=len(bounds);rows.extend([r,r]);cols.extend([sidx+t,sidx+t+1]);vals.extend([1.,-1.]);bounds.append(discharge/physical.eta_d)
    if vid is not None:
        r=len(bounds);rows.extend([r,r]);cols.extend([vid,sidx+n]);vals.extend([1.,-1.]);bounds.append(0.)
    a=sparse.coo_matrix((vals,(rows,cols)),shape=(len(bounds),nv)).tocsr()
    tic=perf_counter();r=linprog(obj,A_ub=a,b_ub=np.asarray(bounds),bounds=np.c_[lo,hi],method='highs');elapsed=perf_counter()-tic
    if not r.success:raise RuntimeError(r.message)
    x=np.diff(r.x[sidx:sidx+n+1]);c=np.maximum(x,0)/physical.eta_c;b=np.maximum(-x,0)*physical.eta_d
    return dict(q=r.x[:n],c=c,b=b,soc=r.x[sidx:sidx+n+1],e=r.x[eidx:eidx+k*n].reshape(k,n),
                objective=float(r.fun),seconds=elapsed,result=r)
