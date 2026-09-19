"""Re-solve the frozen Q1 case and independently reconcile public derived tables."""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'reproduction'))
from model import q1

def main():
    source = ROOT/'data/F04_q1_dispatch.csv'
    rows = list(csv.DictReader(source.open(encoding='utf-8-sig')))
    col = lambda key: np.array([float(r[key]) for r in rows])
    p,l,v = (col(k) for k in ('price_yuan_per_kwh','load_kwh','pv_kwh'))
    config=json.loads((ROOT/'reproduction/configs/q1_v2.json').read_text('utf-8'))
    solved=q1.solve(p,l,v,config)
    x=solved['solution']; g,c,d,w,s = (x[k] for k in ('g','c','d','spill','soc'))
    ref=json.loads((ROOT/'data/q1_reference.json').read_text('utf-8'))
    errors={
      'balance_kwh':float(np.max(np.abs(g+v+d-l-c-w))),
      'inventory_recursion_kwh':float(np.max(np.abs(np.diff(s)-.9*c+d/.9))),
      'initial_stock_kwh':float(abs(s[0]-6000)),
      'terminal_stock_kwh':float(abs(s[-1]-6000)),
      'below_stock_min_kwh':float(np.maximum(1200-s,0).max()),
      'above_stock_max_kwh':float(np.maximum(s-10800,0).max()),
      'above_charge_limit_kwh':float(np.maximum(c-5000/6,0).max()),
      'above_discharge_limit_kwh':float(np.maximum(d-5000/6,0).max()),
      'negative_grid_or_actions_kwh':float(max(np.maximum(-a,0).max() for a in (g,c,d,w))),
      'cost_difference_yuan':float(abs(p@g-ref['cost_yuan'])),
      'frozen_dispatch_cost_difference_yuan':float(abs(p@col('grid_kwh')-ref['cost_yuan'])),
    }
    simultaneous=int(((c>1e-6)&(d>1e-6)).sum())
    table=list(csv.DictReader((ROOT/'data/strategy_summary.csv').open(encoding='utf-8-sig')))
    identities={r['strategy']:abs(float(r['plan_cost_yuan'])+float(r['adjustment_cost_yuan'])+
                                float(r['emergency_cost_yuan'])-float(r['total_cash_yuan'])) for r in table}
    daily=list(csv.DictReader((ROOT/'data/F06_q2_annual.csv').open(encoding='utf-8-sig')))
    daily_errors={}
    for label in ['D','S','F_fast']:
        subset=[r for r in daily if r['strategy'].replace('-','_')==label]
        expected=next(r for r in table if r['strategy']=='q2/'+label)
        if len(subset)!=334: raise AssertionError(f'{label}: expected 334 days')
        daily_errors[label]=abs(sum(float(r['cash_yuan']) for r in subset)-float(expected['total_cash_yuan']))
    assert len(rows)==144 and simultaneous==0
    assert max(errors.values())<1e-6,errors
    assert max(identities.values())<1e-6,identities
    assert max(daily_errors.values())<1e-6,daily_errors
    receipt=dict(status='PASS',scope='Fresh Q1 solve, physical ledger, frozen 334-day aggregate reconciliation; no fresh annual optimization',
                 q1_slots=144,q1_cost_yuan=float(p@g),q1_without_storage_yuan=float(p@np.maximum(l-v,0)),
                 q1_saving_percent=100*(1-float(p@g)/float(p@np.maximum(l-v,0))),
                 simultaneous_slots=simultaneous,errors=errors,cash_identity_errors_yuan=identities,
                 daily_aggregate_errors_yuan=daily_errors,
                 versions=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__),
                 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    out=ROOT/'verification/representative.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
