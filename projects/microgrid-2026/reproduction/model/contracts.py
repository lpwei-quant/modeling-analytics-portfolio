"""Convex state-increment LPs with final-contract settlement and one signal split."""
from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from model.planning import Physical

def contract_cost(q, m, price, fee='B'):
    q, m, p = np.asarray(q,float), np.asarray(m,float), np.asarray(price,float)
    if fee == 'B':
        return p*np.minimum(q,m) + .5*p*np.maximum(q-m,0) + 1.5*p*np.maximum(m-q,0)
    if fee == 'A':
        return p*q + .5*p*np.maximum(q-m,0) + 1.5*p*np.maximum(m-q,0)
    raise ValueError('fee must be A or B')

def _inputs(load,pv,price,s0,physical):
    load, pv = np.atleast_2d(np.asarray(load,float)), np.atleast_2d(np.asarray(pv,float))
    p = np.asarray(price,float)
    if load.shape != pv.shape or p.shape != (load.shape[1],) or load.shape[0] < 1:
        raise ValueError('scenario/price shape mismatch or empty scenario set')
    if not all(np.isfinite(a).all() for a in (load,pv,p)) or min(load.min(),pv.min()) < -1e-8 or p.min() <= 0:
        raise ValueError('nonphysical scenario or nonpositive price')
    if not physical.s_min-1e-7 <= s0 <= physical.s_max+1e-7:
        raise ValueError('initial inventory outside bounds')
    return np.maximum(load,0), np.maximum(pv,0), p

@lru_cache(maxsize=500)
def _shared_topology(k,n,eta_c,eta_d,has_value):
    qidx=0; sidx=n; eidx=2*n+1; yidx=eidx+k*n if has_value else None
    nv=eidx+k*n+int(has_value)
    times=np.tile(np.arange(n),k)
    rows=[];cols=[];vals=[];cursor=0
    for alpha in (1/eta_c,eta_d):
        r=cursor+np.arange(k*n)
        rows.extend([r]*4)
        cols.extend([times,sidx+times+1,sidx+times,eidx+np.arange(k*n)])
        vals.extend([-np.ones(k*n),np.full(k*n,alpha),np.full(k*n,-alpha),-np.ones(k*n)])
        cursor+=k*n
    for sign in (1.,-1.):
        r=cursor+np.arange(n)
        rows.extend([r,r]);cols.extend([sidx+np.arange(n)+1,sidx+np.arange(n)])
        vals.extend([np.full(n,sign),np.full(n,-sign)]);cursor+=n
    if has_value:
        rows.extend([np.array([cursor]),np.array([cursor])])
        cols.extend([np.array([yidx]),np.array([sidx+n])])
        vals.extend([np.array([1.]),np.array([-1.])]);cursor+=1
    mat=sparse.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(cursor,nv)).tocsr()
    return mat,sidx,eidx,yidx,nv

def solve_shared(load,pv,price,s0,*,fixed_m=None,original_q=None,fee='B',
                 lam=0.,target=None,physical=Physical(),first_discharge_cap=None):
    """Day-ahead q, adjustable common m, or fixed-m storage subproblem."""
    if fixed_m is not None and original_q is not None:
        raise ValueError('choose fixed storage control or contract amendment')
    if fee not in ('A','B') or lam < 0:
        raise ValueError('unsupported fee or negative continuation value')
    load,pv,p=_inputs(load,pv,price,s0,physical)
    k,n=load.shape;has_value=lam>0
    mat,sidx,eidx,yidx,nv=_shared_topology(k,n,physical.eta_c,physical.eta_d,has_value)
    lower=np.zeros(nv);upper=np.full(nv,np.inf);obj=np.zeros(nv)
    lower[sidx:sidx+n+1]=physical.s_min;upper[sidx:sidx+n+1]=physical.s_max
    lower[sidx]=upper[sidx]=float(s0)
    obj[eidx:eidx+k*n]=np.tile(physical.emergency_multiple*p/k,k)
    if has_value:
        upper[yidx]=physical.s_max if target is None else float(target)
        obj[yidx]=-float(lam)
    dis=np.minimum(physical.power_energy,load.min(axis=0))
    if first_discharge_cap is not None:dis[0]=min(dis[0],max(0.,float(first_discharge_cap)))
    rhs=np.r_[(pv-load).ravel(),(pv-load).ravel(),
              np.full(n,physical.eta_c*physical.power_energy),dis/physical.eta_d,
              np.array([0.]) if has_value else np.array([])]
    if fixed_m is not None:
        m=np.asarray(fixed_m,float)
        if m.shape!=(n,) or m.min() < -1e-7:raise ValueError('invalid fixed contract')
        lower[:n]=upper[:n]=np.maximum(m,0)
    elif original_q is None:
        obj[:n]=p
    else:
        q=np.asarray(original_q,float)
        if q.shape!=(n,) or q.min() < -1e-7:raise ValueError('invalid original plan')
        mat=sparse.hstack((mat,sparse.csr_matrix((mat.shape[0],n))),format='csr')
        rows=np.r_[np.arange(n),np.arange(n),n+np.arange(n),n+np.arange(n)]
        cols=np.r_[np.arange(n),nv+np.arange(n),np.arange(n),nv+np.arange(n)]
        second=.5 if fee=='B' else -.5
        vals=np.r_[1.5*p,-np.ones(n),second*p,-np.ones(n)]
        fee_mat=sparse.coo_matrix((vals,(rows,cols)),shape=(2*n,nv+n)).tocsr()
        mat=sparse.vstack((mat,fee_mat),format='csr')
        rhs=np.r_[rhs,.5*p*q,(-.5 if fee=='B' else -1.5)*p*q]
        lower=np.r_[lower,np.zeros(n)];upper=np.r_[upper,np.full(n,np.inf)]
        obj=np.r_[obj,np.ones(n)]
    start=perf_counter()
    result=linprog(obj,A_ub=mat,b_ub=rhs,bounds=np.c_[lower,upper],method='highs')
    seconds=perf_counter()-start
    if not result.success:raise RuntimeError(f'Q3 shared LP: {result.status}: {result.message}')
    sol=result.x; m=sol[:n].copy();soc=sol[sidx:sidx+n+1].copy();x=np.diff(soc)
    c=np.maximum(x,0)/physical.eta_c;b=np.maximum(-x,0)*physical.eta_d
    e=sol[eidx:eidx+k*n].reshape(k,n)
    if original_q is not None:fee_values=contract_cost(original_q,m,p,fee)
    elif fixed_m is not None:fee_values=np.zeros(n)
    else:fee_values=p*m
    credit=0 if not has_value else lam*sol[yidx]
    expected=float(fee_values.sum()+physical.emergency_multiple*p@e.mean(axis=0))
    if abs(float(result.fun)-(expected-credit))>1e-5:
        raise AssertionError('shared LP objective/independent cost disagreement')
    return dict(m=m,q=m,c=c,b=b,soc=soc,e=e,objective=float(result.fun),
        expected_cost=expected,contract_cost=fee_values,terminal_credit=float(credit),
        seconds=seconds,variables=len(sol),max_primal_violation=float(max(0.,np.max(mat@sol-rhs))),
        solver_status=int(result.status),fixed_contract_constant_omitted=fixed_m is not None)

def solve_signal_recourse(load,pv,price,s0,groups,*,split=36,allow_amend=True,
                          fee='B',lam=0.,target=None,physical=Physical()):
    """One observed 06:00 signal, with group-shared post-signal decisions."""
    load,pv,p=_inputs(load,pv,price,s0,physical)
    k,n=load.shape
    raw=np.asarray(groups,int)
    if raw.shape!=(k,) or not 0<=split<=n or fee not in ('A','B') or lam<0:
        raise ValueError('invalid signal groups, split, or fee')
    labels,gidx=np.unique(raw,return_inverse=True);ng=len(labels)
    weights=np.bincount(gidx,minlength=ng)/k
    q0=0;m0=n;s0idx=m0+ng*n;e0=s0idx+ng*(n+1);z0=e0+k*n
    y0=z0+ng*n;nv=y0+(ng if lam>0 else 0)
    lo=np.zeros(nv);hi=np.full(nv,np.inf);obj=np.zeros(nv)
    for g in range(ng):
        lo[s0idx+g*(n+1):s0idx+(g+1)*(n+1)]=physical.s_min
        hi[s0idx+g*(n+1):s0idx+(g+1)*(n+1)]=physical.s_max
        lo[s0idx+g*(n+1)]=hi[s0idx+g*(n+1)]=float(s0)
        obj[z0+g*n:z0+(g+1)*n]=weights[g]
        if lam>0:
            obj[y0+g]=-lam*weights[g]
            hi[y0+g]=physical.s_max if target is None else target
    obj[e0:e0+k*n]=np.tile(physical.emergency_multiple*p/k,k)
    rr=[];cc=[];vv=[];rhs=[]
    def ub(cols,vals,bound):
        r=len(rhs);rr.extend([r]*len(cols));cc.extend(cols);vv.extend(vals);rhs.append(bound)
    for s in range(k):
        g=int(gidx[s]);si=s0idx+g*(n+1)
        for t in range(n):
            for alpha in (1/physical.eta_c,physical.eta_d):
                ub([si+t+1,si+t,m0+g*n+t,e0+s*n+t],
                   [alpha,-alpha,-1.,-1.],pv[s,t]-load[s,t])
    for g in range(ng):
        si=s0idx+g*(n+1);minload=load[gidx==g].min(axis=0)
        for t in range(n):
            ub([si+t+1,si+t],[1.,-1.],physical.eta_c*physical.power_energy)
            ub([si+t,si+t+1],[1.,-1.],min(physical.power_energy,minload[t])/physical.eta_d)
            mi,zi=m0+g*n+t,z0+g*n+t
            ub([mi,q0+t,zi],[1.5*p[t],-.5*p[t],-1.],0.)
            if fee=='B':ub([mi,q0+t,zi],[.5*p[t],.5*p[t],-1.],0.)
            else:ub([mi,q0+t,zi],[-.5*p[t],1.5*p[t],-1.],0.)
        if lam>0:ub([y0+g,si+n],[1.,-1.],0.)
    er=[];ec=[];ev=[];eb=[]
    def eq(cols,vals):
        r=len(eb);er.extend([r]*len(cols));ec.extend(cols);ev.extend(vals);eb.append(0.)
    for g in range(ng):
        for t in range(split if allow_amend else n):eq([m0+g*n+t,q0+t],[1.,-1.])
        if g:
            for t in range(1,split+1):eq([s0idx+g*(n+1)+t,s0idx+t],[1.,-1.])
    mat=sparse.coo_matrix((vv,(rr,cc)),shape=(len(rhs),nv)).tocsr()
    emat=sparse.coo_matrix((ev,(er,ec)),shape=(len(eb),nv)).tocsr()
    start=perf_counter()
    r=linprog(obj,A_ub=mat,b_ub=np.asarray(rhs),A_eq=emat,b_eq=np.asarray(eb),
              bounds=np.c_[lo,hi],method='highs')
    seconds=perf_counter()-start
    if not r.success:raise RuntimeError(f'Q3 signal LP: {r.status}: {r.message}')
    q=r.x[:n].copy();m=r.x[m0:m0+ng*n].reshape(ng,n).copy()
    states=r.x[s0idx:s0idx+ng*(n+1)].reshape(ng,n+1).copy()
    increments=np.diff(states,axis=1);c=np.maximum(increments,0)/physical.eta_c;b=np.maximum(-increments,0)*physical.eta_d
    e=r.x[e0:e0+k*n].reshape(k,n)
    fees=contract_cost(q[None,:],m,p[None,:],fee)
    expected=float(weights@fees.sum(axis=1)+physical.emergency_multiple*p@e.mean(axis=0))
    credit=float(lam*weights@r.x[y0:y0+ng]) if lam>0 else 0.
    if abs(r.fun-(expected-credit))>1e-5:raise AssertionError('signal objective double-counting or extraction error')
    pre_error=float(np.max(np.abs(states[:,:split+1]-states[0,:split+1])))
    if pre_error>1e-6:raise AssertionError('pre-signal storage anticipates future group')
    return dict(q=q,m_by_group=m,soc_by_group=states,c_by_group=c,b_by_group=b,
        e=e,group_labels=labels,scenario_groups=gidx,group_weights=weights,
        objective=float(r.fun),expected_cost=expected,terminal_credit=credit,
        contract_cost_by_group=fees,pre_signal_state_error=pre_error,
        seconds=seconds,variables=nv,solver_status=int(r.status),
        max_primal_violation=float(max(0.,np.max(mat@r.x-np.asarray(rhs)))))
