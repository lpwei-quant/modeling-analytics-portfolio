"""Recompute the paper from the four official workbooks (no saved results read)."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from model.history import ForecastArchive, load_data
from model.planning import Physical
from model import control as q2
from model import lagged as lagged
from model.remaining import FusedForecastArchive
from model import amendments as q3
from model import dynamic_prices as q42
from model.prices import load_prices
from model.amendment_prices import build_price_inputs

ROOT = Path(__file__).resolve().parent

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)), encoding='utf-8')

def fingerprints():
    paths = [Path(__file__), *sorted((ROOT/'model').rglob('*.py')),
             *sorted((ROOT/'problem').rglob('*.xlsx')), ROOT/'configs/q1_v2.json']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def designs():
    cases = {f'q2/{k}': ('q2', config, Physical(), False)
             for k, config in q2.PRIMARY_CONFIGS.items() if k != 'F_lagged_guard'}
    cases['q2/lagged_reference'] = ('lagged', lagged.PRIMARY_CONFIGS['F_lagged_no_min_guard'], Physical(), False)
    for label, options in [('G0_raw', dict(scenario_method='raw')),
                           ('G2_shuffled', dict(scenario_method='shuffled')),
                           ('terminal_free', dict(terminal='free')),
                           ('terminal_cyclic', dict(terminal='cyclic'))]:
        cases[f'q2/{label}'] = ('q2', replace(q2.PRIMARY_CONFIGS['F_fast'], **options), Physical(), False)
    for label, model, hours in [('Z1_000', 'Z1', ()), ('Z1_010', 'Z1', (12,)),
        ('Z1_011', 'Z1', (12,18)), ('Z1_111', 'Z1', (6,12,18)),
        ('Z2_110', 'Z2', (6,12)), ('Z2_111', 'Z2', (6,12,18))]:
        config = q3.Config(label, model, hours, bool(hours), controller='fast')
        cases[f'q3/{label}'] = ('q3', config, Physical(), False)
    for mode in ('known_day_ahead', 'previous_day_forecast'):
        cases[f'q4_2/{mode}'] = ('q42', mode, Physical(), False)
        # The selected Q3 configuration is filled from the fresh annual comparison.
        cases[f'q4_3/{mode}'] = ('q43', mode, Physical(), False)
    variants = [('base', {}, {})]
    variants += [(f'terminal_{x}', dict(terminal=x), {}) for x in ('free', 'cyclic')]
    variants += [(f'K_{k}', dict(scenario_count=k), {}) for k in (10,20,50)]
    variants += [(f'eta_{v}', {}, dict(eta_c=v, eta_d=v)) for v in (.85,.95)]
    variants += [(f'upper_{v}', {}, dict(s_max=v)) for v in (7200.,14400.)]
    variants += [(f'power_{v}', {}, dict(power_energy=v/6)) for v in (2500.,7500.)]
    variants += [(f'emergency_{v}', {}, dict(emergency_multiple=v)) for v in (2.,8.)]
    for label, config_values, physical_values in variants:
        cases[f'S_scan/{label}'] = ('q2', replace(q2.PRIMARY_CONFIGS['S'], **config_values),
                                  replace(Physical(), **physical_values), True)
    return cases

def one_day(kind, day, stock, archive, config, physical, prices=None, conditional=None):
    if kind in ('q2', 'lagged'):
        plan, _ = q2.make_plan(day, stock, archive, config, physical)
        plan.q.setflags(write=False)
        sealed = plan.q.copy()
        engine = lagged if kind == 'lagged' else q2
        ledger, execution = engine.execute_day(day, stock, plan, archive, config, physical)
    elif kind == 'q42':
        origin = prices.at_origin(day, config)
        plan, _, view = q42.make_plan(day, stock, archive, origin, physical)
        plan.q.setflags(write=False)
        sealed = plan.q.copy()
        actions, execution = q42.execute_actions(day, stock, plan, view, physical)
        ledger, audit = q42.settle_actual(plan, actions, archive.data.load[day],
            archive.data.pv[day], stock, prices.settle_price(day), physical)
        execution['independent_recalculation'] = audit
    else:
        plan = q3.make_plan(day, stock, archive, prices, config, physical)
        plan.q.setflags(write=False)
        sealed = plan.q.copy()
        ledger, execution = q3.execute_day(day, stock, plan, archive, prices, config,
                                          physical, conditional=conditional)
    assert np.array_equal(sealed, plan.q), 'The zero-time plan was changed'
    assert execution['independent_recalculation']['passed'], 'Independent settlement failed'
    return ledger, execution

def window(name, kind, config, physical, start, stop, initial_soc, destination):
    data = load_data(ROOT)
    archive = ForecastArchive(data, cold_start='scaled')
    prices = conditional = None
    if kind in ('q3', 'q43'):
        archive = FusedForecastArchive(data, root=ROOT, load_archive=archive)
        conditional = q3.ConditionalArchive(archive)
        prices = q3.PriceInputs.q3(data)
        if kind == 'q43':
            mode, config = config
            prices = build_price_inputs(data, load_prices(ROOT, data.dates), mode)
    elif kind == 'q42':
        prices = load_prices(ROOT, data.dates)
    folder = Path(destination)/name
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(name=name, engine=kind, config=asdict(config) if not isinstance(config,str) else config,
                    physical=asdict(physical), start=start, stop=stop, initial_soc=initial_soc,
                    input_and_source_hashes=fingerprints())
    manifest = folder/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text('utf-8')) == identity, 'Existing run has different inputs or code'
        completed = folder/'summary.json'
        if completed.exists():
            return json.loads(completed.read_text('utf-8'))
    save(manifest, identity)
    rows = []
    stock = float(initial_soc)
    guards = 0
    began = perf_counter()
    for day in range(start, stop):
        ledger, execution = one_day(kind, day, stock, archive, config, physical, prices, conditional)
        arrays = {k: v for k,v in ledger.items() if isinstance(v, np.ndarray) and v.shape == (144,)}
        arrays.update(load=data.load[day], pv=data.pv[day])
        if kind in ('q3','q43'):
            arrays['price'] = prices.settlement[day]
        elif kind == 'q42':
            origin = prices.at_origin(day, config)
            arrays.update(decision_price=origin.decision, settlement_price=prices.settle_price(day),
                          nextday_price_forecast=origin.next_day_forecast)
        rows.append(arrays)
        stock = float(ledger['soc_end'][-1])
        guards += execution['guard_count']
        # Residual and controller caches are keyed by the current day only.
        # Historical point forecasts stay cached; expired day buffers can go.
        for owner in (archive, getattr(archive,'load_archive',None), conditional):
            for key in ('_errors','_residuals','_conditional_bases','_base'):
                cache = getattr(owner,key,None)
                if isinstance(cache,dict): cache.clear()
        if day == start or (day+1)%30 == 0 or day+1 == stop:
            progress = dict(case=name, completed_days=len(rows), total_days=stop-start,
                            cash_yuan=float(sum(r['cash_cost'].sum() for r in rows)),
                            final_soc=stock, seconds=perf_counter()-began)
            save(folder/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    combined = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    combined['d'] = np.arange(start,stop)
    assert np.max(np.abs(combined['soc_end'][:-1,-1]-combined['soc_start'][1:,0]), initial=0.) < 1e-6
    np.savez_compressed(folder/'ledger.npz', **combined)
    summary = dict(status='COMPLETED', days=stop-start, initial_soc=initial_soc, final_soc=stock,
        totals={k:float(v.sum()) for k,v in combined.items() if k not in ('d','soc_start','soc_end')},
        daily_terminal_quantiles=np.quantile(combined['soc_end'][:,-1], [0,.05,.25,.5,.75,.95,1]),
        guard_count=guards, independent_recalculation_all_passed=True,
        interday_soc_continuous=True, sealed_plan_unchanged=True, seconds=perf_counter()-began)
    save(folder/'summary.json', summary)
    return summary

def run_case(name, destination, initial_soc, stop=365):
    kind, config, physical, own_warmup = designs()[name]
    if own_warmup:
        warm = window(name+'_warmup', kind, config, physical, 0,31,6000.,destination)
        initial_soc = warm['final_soc']
    if kind == 'q43':
        selected = json.loads((Path(destination)/'q3/selection.json').read_text('utf-8'))['selected']
        config = (config, designs()[selected][1])
    return window(name, kind, config, physical, 31,stop,initial_soc,destination)

def run_q1(destination):
    from model import q1 as model
    base = json.loads((ROOT/'configs/q1_v2.json').read_text('utf-8'))
    values, _ = model.read_inputs(base)
    summaries = {}
    for name in model.DEFAULT_CASES + ['upper_plus_10','lower_minus_10','charge_power_plus_100','discharge_power_plus_100']:
        config = model.case_config(base, name)
        data = model.interpreted(values, config['time_interpretation'])
        price, load, pv = data[:,0], data[:,1]*config['step_hours'], data[:,2]*config['step_hours']
        solved = model.solve(price, load, pv, config)
        folder = Path(destination)/'q1'/name
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'solution.npz', **solved['solution'], price=price, load=load, pv=pv)
        summaries[name] = model.metrics(solved['solution'], price, load, pv, config)
        save(folder/'summary.json', summaries[name])
    save(Path(destination)/'q1/comparison.json', summaries)
    return summaries

def select_q3(destination):
    names = [name for name in designs() if name.startswith('q3/') and name != 'q3/Z1_000']
    costs = {name: json.loads((Path(destination)/name/'summary.json').read_text('utf-8'))['totals']['cash_cost'] for name in names}
    result = dict(selected=min(costs,key=costs.get), annual_cash_yuan=costs,
                  rule='minimum verified complete annual cash among the five fixed configurations')
    save(Path(destination)/'q3/selection.json', result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='all', help='all, warmup, q1, or a case listed by --list')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=Path, default=ROOT/'output')
    parser.add_argument('--stop', type=int, default=365, help='exclusive day index; 365 is the full evaluation')
    args = parser.parse_args()
    save(args.out/'process.json',dict(pid=os.getpid(),case=args.case,workers=args.workers))
    if args.list:
        print('\n'.join(['q1','warmup',*designs()]))
        return
    if args.case in ('all','q1'):
        run_q1(args.out)
        if args.case == 'q1':
            return
    warm = window('warmup', 'q2', q2.PRIMARY_CONFIGS['F_fast'], Physical(), 0,31,6000.,args.out)
    if args.case == 'warmup':
        return
    if args.case != 'all':
        run_case(args.case,args.out,warm['final_soc'],args.stop)
        return
    cases = designs()
    # Q4-3 starts only after the five fresh Q3 annual results determine its configuration.
    for batch in ([n for n in cases if not n.startswith('q4_3/')], [n for n in cases if n.startswith('q4_3/')]):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(run_case,n,args.out,warm['final_soc'],args.stop):n for n in batch}
            for future in as_completed(jobs):
                summary = future.result()
                print(json.dumps(dict(finished=jobs[future],cash_yuan=summary['totals']['cash_cost'])),flush=True)
        if not batch[0].startswith('q4_3/'):
            select_q3(args.out)
    save(args.out/'complete.json', dict(status='COMPLETED', cases=len(cases),
        input_and_source_hashes=fingerprints(), evaluation_days=[31,args.stop]))

if __name__ == '__main__':
    main()
