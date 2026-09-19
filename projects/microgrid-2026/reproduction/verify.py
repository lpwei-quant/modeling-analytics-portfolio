"""Re-run mathematical checks and derived diagnostics from the supplied inputs."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from model import q1, planning, contracts
from model.explicit_check import explicit_milp, cases
from model.history import load_data, ForecastArchive
from model.fusion import FusedForecastArchive as FullWindow
from model.remaining import FusedForecastArchive as RemainingWindow
from model import amendments
from model.prices import load_prices
from model.amendment_prices import build_price_inputs
from model.oracle import solve_oracle
from run import ROOT, save, fingerprints, designs, one_day


def small_models():
    results = {'classification':'Simulated small cases; Q1 variants use Observed attachment 1'}
    config = json.loads((ROOT/'configs/q1_v2.json').read_text('utf-8'))
    source, _ = q1.read_inputs(config)
    q1_results = []
    names = q1.DEFAULT_CASES + ['upper_plus_10','lower_minus_10','charge_power_plus_100','discharge_power_plus_100']
    for name in names:
        cfg = q1.case_config(config,name)
        values = q1.interpreted(source,cfg['time_interpretation'])
        p,l,v = values[:,0],values[:,1]*cfg['step_hours'],values[:,2]*cfg['step_hours']
        actual = q1.solve(p,l,v,cfg)
        physical = planning.Physical(cfg['eta_charge'],cfg['eta_discharge'],cfg['soc_min_kwh'],
            cfg['soc_max_kwh'],cfg['max_charge_power_kw']*cfg['step_hours'],5.)
        independent = explicit_milp(l[None,:],v[None,:],p,cfg['initial_soc_kwh'],physical=physical,
            terminal_equal=True,charge_limit=cfg['max_charge_power_kw']*cfg['step_hours'],
            discharge_limit=cfg['max_discharge_power_kw']*cfg['step_hours'])
        gap = abs(actual['primary'].fun-independent['objective'])
        assert gap < 1e-6,(name,gap)
        q1_results.append(dict(case=name,lp_cost=float(actual['primary'].fun),milp_gap=gap))
    results['q1'] = q1_results
    rng = np.random.default_rng(20260910);constructed = []
    for index in range(24):
        n = (4,8,16,24)[index%4]
        p,l,v = rng.uniform(.1,2,n),rng.uniform(0,8,n),rng.uniform(0,12,n)
        l[rng.random(n)<.15] = 0.;v[rng.random(n)<.30] = 0.
        cfg = {**config,'step_hours':1.,'soc_min_kwh':0.,'soc_max_kwh':float(rng.uniform(3,15)),
            'max_charge_power_kw':float(rng.uniform(.2,8)),'max_discharge_power_kw':float(rng.uniform(.2,8)),
            'eta_charge':float(rng.uniform(.65,1)),'eta_discharge':float(rng.uniform(.65,1))}
        cfg['initial_soc_kwh'] = float(rng.uniform(0,cfg['soc_max_kwh']))
        actual = q1.solve(p,l,v,cfg)
        physical = planning.Physical(cfg['eta_charge'],cfg['eta_discharge'],0.,cfg['soc_max_kwh'],
                                     cfg['max_charge_power_kw'],5.)
        independent = explicit_milp(l[None,:],v[None,:],p,cfg['initial_soc_kwh'],physical=physical,
            terminal_equal=True,charge_limit=cfg['max_charge_power_kw'],
            discharge_limit=cfg['max_discharge_power_kw'])
        gap = abs(actual['primary'].fun-independent['objective'])
        assert gap < 1e-6,(index,gap)
        constructed.append(dict(case=index,periods=n,objective_gap=gap,classification='Simulated'))
    results['q1_constructed'] = constructed
    rng = np.random.default_rng(20260911)
    comparisons = []
    for index in range(18):
        l,v = rng.uniform(0,950,(3,6)),rng.uniform(0,1200,(3,6))
        p,s0 = rng.uniform(.37,1.4,6),float(rng.uniform(1200,9500))
        term = ('free','value','cyclic')[index%3]
        q = rng.uniform(0,300,6) if index%2 else None
        lam = .47 if term == 'value' else 0.
        target = s0 if term == 'cyclic' else 8500.
        actual = planning.solve_state_lp(l,v,p,s0,terminal=term,target=target,lam=lam,
                                          no_battery_dump=True,fixed_q=q)
        independent = explicit_milp(l,v,p,s0,physical=planning.Physical(),
            kind='dayahead' if q is None else 'fixed',fixed_m=q,lam=lam,target=target,
            terminal_floor=s0 if term == 'cyclic' else None)
        gap = abs(actual['objective']-independent['objective']-(0. if q is None else p@q))
        assert gap < 1e-6,(index,gap)
        comparisons.append(dict(case=index,objective_gap=float(gap)))
    results['q2'] = comparisons
    comparisons = []
    for case in cases():
        l,v,p,s0 = [case[k] for k in ('load','pv','price','s0')]
        common = {k:case[k] for k in ('physical','lam','target')}
        for kind,fee in (('dayahead','B'),('amend','B'),('amend','A'),('fixed','B')):
            kwargs = dict(common,fee=fee,first_discharge_cap=.2 if case['seed']%2 == 0 else None)
            if kind == 'amend': kwargs['original_q'] = case['original_q']
            if kind == 'fixed': kwargs['fixed_m'] = case['fixed_m']
            actual = contracts.solve_shared(l,v,p,s0,**kwargs)
            independent = explicit_milp(l,v,p,s0,kind=kind,**kwargs)
            gap = abs(actual['objective']-independent['objective'])
            assert gap < 2e-6,(case['name'],kind,gap)
            comparisons.append(dict(case=case['name'],kind=kind,fee=fee,objective_gap=gap))
        for fee,allowed in (('B',True),('A',True),('B',False),('A',False)):
            kwargs = dict(common,groups=case['groups'],split=case['split'],allow_amend=allowed,fee=fee)
            actual = contracts.solve_signal_recourse(l,v,p,s0,**kwargs)
            independent = explicit_milp(l,v,p,s0,kind='signal',**kwargs)
            gap = abs(actual['objective']-independent['objective'])
            assert gap < 2e-6,(case['name'],'signal',gap)
            comparisons.append(dict(case=case['name'],kind='signal',fee=fee,amend=allowed,objective_gap=gap))
    results['q3'] = comparisons
    return results


def information_boundary():
    data = load_data(ROOT);results = []
    for day in (31,100,264):
        cut = 72
        load,pv = data.load.copy(),data.pv.copy()
        load.reshape(-1)[day*144+cut+1:] *= 1.2
        pv.reshape(-1)[day*144+cut+1:] *= .7
        altered = replace(data,load=load,pv=pv)
        for case in ('q2/F_fast','q2/lagged_reference'):
            kind,config,physical,_ = designs()[case]
            ledgers = [one_day(kind,day,8657.424357634898,ForecastArchive(value,cold_start='scaled'),
                               config,physical)[0] for value in (data,altered)]
            errors = {key:float(np.max(np.abs(ledgers[0][key][:cut+1]-ledgers[1][key][:cut+1])))
                      for key in ('q','c','b','e','soc_end')}
            assert max(errors.values()) < 1e-6,(day,case,errors)
            assert np.array_equal(ledgers[0]['q'],ledgers[1]['q'])
            results.append(dict(day=day,case=case,observed_through_slot=cut,max_errors=errors))
        for hour in (0,6,12,18):
            load,pv = data.load.copy(),data.pv.copy()
            boundary = day*144+hour*6
            load.reshape(-1)[boundary:] *= 1.2;pv.reshape(-1)[boundary:] *= .7
            changed = replace(data,load=load,pv=pv)
            predictions = [RemainingWindow(value,root=ROOT,
                load_archive=ForecastArchive(value,cold_start='scaled')).point(day,hour)
                for value in (data,changed)]
            gap = max(float(np.max(np.abs(predictions[0][k]-predictions[1][k]))) for k in (0,1))
            assert gap < 1e-6,(day,hour,gap)
            results.append(dict(day=day,issue_hour=hour,prediction_error=gap))
    return dict(classification='Simulated future-data perturbation; fixed valid state',cases=results)


def forecast_diagnostics():
    data = load_data(ROOT)
    old = FullWindow(data,root=ROOT,load_archive=ForecastArchive(data,cold_start='scaled'))
    new = RemainingWindow(data,root=ROOT,load_archive=ForecastArchive(data,cold_start='scaled'))
    errors = {}
    for start,stop,period in ((0,31,'January'),(31,365,'February_December')):
        for day in range(start,stop):
            for hour in (0,6,12,18):
                boundary = day*144+hour*6
                _,history,official,_ = new.components(day,hour)
                previous = old.point(day,hour)[1]
                fused = new.point(day,hour)[1]
                for window,length in (('full',144),('remaining',144-hour*6)):
                    if boundary+length > stop*144: continue
                    actual = data.pv.reshape(-1)[boundary:boundary+length]
                    for name,prediction in (('history',history),('official',official),('previous',previous),('fused',fused)):
                        error = prediction[:length]-actual
                        errors.setdefault((period,window,name,'all'),[]).append(error)
                        errors.setdefault((period,window,name,str(hour)),[]).append(error)
    metrics = {}
    for keys,arrays in errors.items():
        values = np.concatenate(arrays)
        metrics['/'.join(keys)] = dict(targets=len(values),mae=float(np.abs(values).mean()),
            bias=float(values.mean()),p90=float(np.quantile(np.abs(values),.9)))
    weights = {str(hour):new.weight_at(31,hour)['official_weight'] for hour in (0,6,12,18)}
    return dict(weights=weights,metrics=metrics,classification='Derived; January numerical fitting only')


def optimal_face_and_marginals(output):
    config = json.loads((ROOT/'configs/q1_v2.json').read_text('utf-8'))
    data,_ = q1.read_inputs(config)
    p,l,v = data[:,0],data[:,1]/6,data[:,2]/6
    result = q1.solve(p,l,v,config)
    model,answer = result['model'],result['primary']
    ranges = [];widest = (0.,None,None)
    for index in list(range(len(answer.x)))+[None]:
        values = [];vectors = []
        for sign in (1.,-1.):
            objective = np.zeros(len(answer.x))
            if index is None: objective[:len(p)] = sign
            else: objective[index] = sign
            test = linprog(objective,A_ub=model['aub'],b_ub=model['bub'],
                A_eq=np.vstack((model['aeq'].toarray(),model['c'])),
                b_eq=np.r_[model['beq'],answer.fun],bounds=np.c_[model['lower'],model['upper']],method='highs')
            assert test.success
            values.append(float(objective@test.x/sign));vectors.append(test.x)
        width = values[1]-values[0]
        if width > 1e-6: ranges.append(dict(index=index,minimum=values[0],maximum=values[1]))
        if index is not None and index < len(p) and width > widest[0]:
            widest = (width,index,vectors)
    witnesses = {}
    for label,vector in zip(('minimum','maximum'),widest[2]):
        solution = q1.recover(vector,len(p),l,v,config)
        assert abs(p@solution['g']-answer.fun) < 1e-6
        witnesses.update({f'{label}_{k}':value for k,value in solution.items()})
    folder = output/'checks';folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'q1_face_witnesses.npz',**witnesses)
    n = len(p);step = config['step_hours']
    yb,yl,yu = answer.ineqlin.marginals,answer.lower.marginals,answer.upper.marginals
    limit = config['max_discharge_power_kw']*step
    changed = np.minimum(limit+100*step,l)-np.minimum(limit,l)
    predictions = [('upper_plus_10',10*yu[n+1:-1].sum()),
        ('lower_minus_10',-10*yl[n+1:-1].sum()),
        ('charge_power_plus_100',100*step*config['eta_charge']*yb[2*n:3*n].sum()),
        ('discharge_power_plus_100',yb[3*n:4*n]@(changed/config['eta_discharge']))]
    margins = []
    for case,prediction in predictions:
        altered = q1.solve(p,l,v,q1.case_config(config,case))['primary'].fun-answer.fun
        assert abs(altered-prediction) < 1e-6,(case,altered,prediction)
        margins.append(dict(case=case,predicted_change=float(prediction),actual_change=float(altered),
                            absolute_difference=float(abs(altered-prediction))))
    return dict(variable_count=len(answer.x),nonunique_coordinates=ranges,
                coordinate_minimizations=2*len(answer.x),total_purchase_minimizations=2,
                witness_coordinate=widest[1],witness_width=widest[0],
                primary_cost=float(answer.fun),marginal_checks=margins)


def quantile_replay(output,name):
    data = load_data(ROOT)
    archive = RemainingWindow(data,root=ROOT,load_archive=ForecastArchive(data,cold_start='scaled'))
    config = designs()['q3/Z2_111'][1]
    prices = amendments.PriceInputs.q3(data)
    if name.startswith('q4_3/'):
        selected = json.loads((output/'q3/selection.json').read_text('utf-8'))['selected']
        config = designs()[selected][1]
        prices = build_price_inputs(data,load_prices(ROOT,data.dates),name.split('/')[1])
    with np.load(output/name/'ledger.npz') as z: ledger = {k:z[k].copy() for k in z.files}
    maximum_violation = 0.;max_contract_error = 0.;count = positions = committed = 0
    physical = planning.Physical()
    fixed_action_bound = fixed_action_change = 0.;max_stock_roundoff = 0.
    for index,day in enumerate(ledger['d']):
        plan = amendments.make_plan(int(day),ledger['soc_start'][index,0],archive,prices,config)
        assert np.max(np.abs(plan.q-ledger['q'][index])) < 1e-6
        # Rebuild the controller's sequential arithmetic. The settled cumsum
        # can differ by 1e-11 kWh and select another degenerate LP vertex.
        runtime_stock = np.empty(144);stock = float(ledger['soc_start'][index,0])
        for slot in range(144):
            runtime_stock[slot] = stock
            stock += physical.eta_c*ledger['c'][index,slot]-ledger['b'][index,slot]/physical.eta_d
        max_stock_roundoff = max(max_stock_roundoff,float(np.max(np.abs(runtime_stock-ledger['soc_start'][index]))))
        for hour in config.enabled_hours:
            t = hour*6
            stop = amendments.effective_blocks(config.enabled_hours)[t]
            l,v,_ = archive.scenarios(int(day),hour,config.scenario_count)
            p = prices.decision[day,t:]
            old_q = plan.q[t:]
            result = contracts.solve_shared(l[:,:144-t],v[:,:144-t],p,runtime_stock[t],
                original_q=old_q,fee='B',lam=plan.lam,target=plan.target)
            m = result['m'];net = l[:,:144-t]-v[:,:144-t]+result['c']-result['b']
            # Left and right subgradients bracket zero; m=0 has only a right condition.
            eps = 1e-6
            left_contract = np.where(m <= old_q+eps,.5,1.5)*p
            right_contract = np.where(m >= old_q-eps,1.5,.5)*p
            left = left_contract-5*p*(net >= m[None,:]-eps).mean(axis=0)
            right = right_contract-5*p*(net > m[None,:]+eps).mean(axis=0)
            violation = max(float(np.max(left[m>eps],initial=0.)),float(np.max(-right,initial=0.)),0.)
            maximum_violation = max(maximum_violation,violation)
            error = float(np.max(np.abs(m[:stop-t]-ledger['m'][index,t:stop])))
            max_contract_error = max(max_contract_error,error)
            assert violation < 1e-6 and error < 1e-6,(name,int(day),hour,violation,error)
            count += 1;positions += len(m);committed += stop-t
        forecast = archive.point(int(day),18)[1][:36]
        actual = data.pv[day,108:]
        need = data.load[day,108:]+ledger['c'][index,108:]-ledger['b'][index,108:]-ledger['m'][index,108:]
        tariff = 5*prices.settlement[day,108:]
        fixed_action_bound += float(tariff@np.abs(forecast-actual))
        fixed_action_change += float(tariff@(np.maximum(need-forecast,0)-np.maximum(need-actual,0)))
        for owner in (archive,archive.load_archive):
            for key in ('_errors','_residuals','_conditional_bases'):
                cache = getattr(owner,key,None)
                if isinstance(cache,dict): cache.clear()
    assert max_stock_roundoff < 1e-6
    return dict(days=len(ledger['d']),amendment_lps=count,window_positions=positions,
        committed_positions=committed,max_subgradient_violation=maximum_violation,
        max_committed_contract_error=max_contract_error,fixed_action_bound=fixed_action_bound,
        fixed_action_cash_change=fixed_action_change,max_runtime_vs_settlement_stock_roundoff=max_stock_roundoff,
        state_reconstruction='Sequential arithmetic of execute_day',passed=True)


def oracle_and_shadow(output):
    data = load_data(ROOT)
    with np.load(output/'q2/F_fast/ledger.npz') as z: ledger = {k:z[k].copy() for k in z.files}
    price = np.tile(data.price,len(ledger['d']))
    entry = float(ledger['soc_start'][0,0])
    terminal = float(ledger['soc_end'][-1,-1])
    result = {}
    for label,target in (('free',None),('matched',terminal)):
        _,info = solve_oracle(ledger['load'].ravel(),ledger['pv'].ravel(),price,entry,terminal_soc=target)
        result[label] = info
    archive = ForecastArchive(data,cold_start='scaled')
    rows = []
    for day in (31,100,264):
        initial = float(ledger['soc_start'][np.flatnonzero(ledger['d'] == day)[0],0])
        blocks = [planning.Block(*archive.point(day,target),data.price) for target in (day,day+1)]
        baseline = planning.solve(blocks,initial,terminal='cyclic',no_battery_dump=True,retain_problem=True)
        f = baseline.formulation;values = []
        for change in (.1,-.1):
            rhs = f['rhs'].copy();rhs[f['link_rows'][1]] += change
            fit = linprog(f['objective'],A_ub=f['ub'],b_ub=f['ub_rhs'],A_eq=f['eq'],b_eq=rhs,
                bounds=np.c_[f['lower'],f['upper']],method='highs')
            assert fit.success
            values.append((baseline.objective-fit.fun)/change)
        assert max(abs(x-baseline.lam) for x in values) < 1e-6
        rows.append(dict(day=day,initial_soc=initial,shadow=baseline.lam,differences=values))
    result['shadow_checks'] = rows
    return result


def annual_accounts(output):
    data = load_data(ROOT);dynamic = load_prices(ROOT,data.dates)
    report = {};primary = {}
    for name,(kind,config,physical,_) in designs().items():
        with np.load(output/name/'ledger.npz') as source:z = {k:source[k].copy() for k in source.files}
        assert np.array_equal(z['d'],np.arange(31,365)),name
        load,pv = data.load[z['d']],data.pv[z['d']]
        assert np.array_equal(load,z['load']) and np.array_equal(pv,z['pv'])
        p = (np.stack([dynamic.settle_price(int(day)) for day in z['d']])
             if kind in ('q42','q43') else np.tile(data.price,(334,1)))
        m = z.get('m',z['q']);c,b,q = z['c'],z['b'],z['q']
        need = load+c-b;used_pv = np.minimum(pv,need)
        called = np.minimum(m,need-used_pv);emergency = need-used_pv-called
        fee = p*q
        if 'm' in z:
            fee += p*(1.5*np.maximum(m-q,0)-.5*np.maximum(q-m,0))
        cash = fee+physical.emergency_multiple*p*emergency
        errors = {
            'flow':float(np.max(np.abs(z['v']+z['a']+z['e']+b-load-c))),
            'state_step':float(np.max(np.abs(z['soc_start']+physical.eta_c*c-b/physical.eta_d-z['soc_end']))),
            'interday':float(np.max(np.abs(z['soc_start'][1:,0]-z['soc_end'][:-1,-1]))),
            'cash':float(np.max(np.abs(cash-z['cash_cost']))),
            'emergency':float(np.max(np.abs(emergency-z['e']))),
            'simultaneous':float(np.max(np.minimum(c,b))),
            'state_bounds':max(0.,float(physical.s_min-z['soc_end'].min()),float(z['soc_end'].max()-physical.s_max)),
            'power_bounds':max(0.,float(c.max()-physical.power_energy),float(b.max()-physical.power_energy)),
        }
        assert max(errors.values()) < 1e-6,(name,errors)
        report[name] = dict(periods=q.size,max_errors=errors,cash_yuan=float(cash.sum()),
                            terminal_soc=float(z['soc_end'][-1,-1]))
        if name in ('q2/F_fast','q3/Z2_111','q4_2/known_day_ahead','q4_3/known_day_ahead'):
            primary[name] = (z,p)
    z,p = primary['q2/F_fast'];overlap = (z['e']>1e-6)&(z['c']>1e-6)
    unchanged_need = np.maximum(z['load']-z['pv']-z['b']-z['q'],0)
    overlap_report = dict(periods=int(overlap.sum()),emergency_kwh=float(z['e'][overlap].sum()),
        reduced_gap_kwh=float((z['e']-unchanged_need)[overlap].sum()),
        remaining_gap_kwh=float(unchanged_need[overlap].sum()))
    repricing = {}
    for first,last in (('q2/F_fast','q4_2/known_day_ahead'),('q3/Z2_111','q4_3/known_day_ahead')):
        z,p0 = primary[first];future,p = primary[last]
        weight = z['q']+5*z['e']
        if 'm' in z:weight += 1.5*np.maximum(z['m']-z['q'],0)-.5*np.maximum(z['q']-z['m'],0)
        original,repriced = float((p0*weight).sum()),float((p*weight).sum())
        mean = float(334*((p.mean(axis=0)-p0[0])*weight.mean(axis=0)).sum())
        covariance = float(((p-p.mean(axis=0))*(weight-weight.mean(axis=0))).sum())
        assert abs(repriced-original-mean-covariance) < 1e-6
        repricing[first] = dict(original=original,repriced=repriced,mean_component=mean,
            covariance_component=covariance,reoptimized=float(future['cash_cost'].sum()))
    return dict(trajectories=report,emergency_charging_overlap=overlap_report,repricing=repricing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('small','boundary','forecast','face','annual','all'),default='all')
    parser.add_argument('--out',type=Path,default=ROOT/'output')
    args = parser.parse_args()
    selected = ('small','boundary','forecast','face','annual') if args.scope == 'all' else (args.scope,)
    for scope in selected:
        if scope == 'small': report = small_models()
        elif scope == 'boundary': report = information_boundary()
        elif scope == 'forecast': report = forecast_diagnostics()
        elif scope == 'face': report = optimal_face_and_marginals(args.out)
        else:
            report = {'oracle_and_shadow':oracle_and_shadow(args.out),'annual_accounts':annual_accounts(args.out)}
            for name in ('q3/Z2_111','q4_3/known_day_ahead','q4_3/previous_day_forecast'):
                report[name] = quantile_replay(args.out,name)
        hashes = fingerprints()
        hashes['verify.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        save(args.out/'checks'/f'{scope}.json',dict(passed=True,results=report,source_hashes=hashes))
        print(json.dumps(dict(scope=scope,passed=True)),flush=True)


if __name__ == '__main__':
    main()
