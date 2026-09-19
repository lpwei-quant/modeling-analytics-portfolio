"""Settle final effective contracts, keeping original q out of physical flow."""
import numpy as np

from model.planning import Physical
from model.settlement import settle_schedule
from model.contracts import contract_cost

LEDGER_FIELDS = ('q','m','a','u','v','w','c','b','e','soc_start','soc_end',
                 'plan_cost','adjustment_cost','contract_cost','emergency_cost','cash_cost')

def settle(q,m,c,b,load,pv,s0,price,fee='B',physical=Physical()):
    original=np.asarray(q,float)
    z=settle_schedule(m,c,b,load,pv,s0,price,eta_charge=physical.eta_c,
        eta_discharge=physical.eta_d,soc_min=physical.s_min,soc_max=physical.s_max,
        bus_limit=physical.power_energy,emergency_multiplier=physical.emergency_multiple,
        enforce_m0=False)
    if original.shape!=z['q'].shape or not np.isfinite(original).all() or original.min() < -1e-7:
        raise ValueError('Invalid original contract')
    z['m']=z.pop('q');z['q']=original.copy()
    z['plan_cost']=np.asarray(price)*original
    z['contract_cost']=contract_cost(original,z['m'],price,fee)
    z['adjustment_cost']=z['contract_cost']-z['plan_cost']
    z['cash_cost']=z['contract_cost']+z['emergency_cost']
    z['metadata'].update(fee_interpretation=fee,physical_contract='final effective m',
        original_q_charged_once=True,terminal_value_in_actual_cash=False,
        unused_definition='u=m-a',candidate_contracts_charged=False)
    return z

def independent_recalculate(q,m,c,b,load,pv,s0,price,ledger,fee='B',physical=Physical()):
    """Scalar branch/flow reconstruction, without calling either fee/settler."""
    arrays=[np.asarray(a,float) for a in (q,m,c,b,load,pv,price)]
    if any(a.ndim!=1 or len(a)!=len(arrays[0]) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Invalid audit arrays')
    if fee not in ('A','B'):raise ValueError('Invalid fee')
    rebuilt={k:[] for k in LEDGER_FIELDS};stock=float(s0);violations=[]
    for t,(qt,mt,ct,bt,lt,vt,pt) in enumerate(zip(*arrays)):
        need=lt+ct-bt;used_pv=min(vt,need);remaining=need-used_pv
        called=min(mt,remaining);emergency=remaining-called
        if mt>=qt:contract=pt*qt+1.5*pt*(mt-qt)
        elif fee=='B':contract=pt*mt+.5*pt*(qt-mt)
        else:contract=pt*qt+.5*pt*(qt-mt)
        end=stock+physical.eta_c*ct-bt/physical.eta_d
        vals=dict(q=qt,m=mt,a=called,u=mt-called,v=used_pv,w=vt-used_pv,
            c=ct,b=bt,e=emergency,soc_start=stock,soc_end=end,
            plan_cost=pt*qt,adjustment_cost=contract-pt*qt,contract_cost=contract,
            emergency_cost=physical.emergency_multiple*pt*emergency,
            cash_cost=contract+physical.emergency_multiple*pt*emergency)
        for k,x in vals.items():rebuilt[k].append(x)
        if (min(qt,mt,ct,bt,lt,vt,need,called,emergency)<-1e-6
            or min(ct,bt)>1e-6 or max(ct,bt)>physical.power_energy+1e-6
            or min(stock,end)<physical.s_min-1e-6 or max(stock,end)>physical.s_max+1e-6
            or abs(used_pv+called+emergency+bt-lt-ct)>1e-6):
            violations.append(t)
        stock=end
    errors={k:float(np.max(np.abs(np.asarray(v)-ledger[k]))) for k,v in rebuilt.items()}
    return dict(passed=not violations and max(errors.values())<1e-6,
        max_abs_error_by_field=errors,violating_slots=violations,
        method='independent scalar piecewise fee and physical flow',periods=len(arrays[0]))
