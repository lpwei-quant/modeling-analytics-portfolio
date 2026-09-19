"""Price-only adapter; every planning/control decision delegates to Q2 F-fast."""
from dataclasses import replace

import numpy as np

from model.planning import Physical
from model.control import PRIMARY_CONFIGS, make_plan as q2_make_plan, execute_day as q2_execute_day
from model.settlement import LEDGER_FIELDS, settle_schedule, independent_recalculator

CONFIG = PRIMARY_CONFIGS['F_fast']
PHYSICAL_FIELDS = tuple(key for key in LEDGER_FIELDS if not key.endswith('cost'))

class DecisionArchive:
    """The Q2 data contract with only its deterministic price vector replaced."""
    def __init__(self, archive, price_day):
        self.data = replace(archive.data, price=price_day.decision)
        self.archive = archive
        self.price_day = price_day

    def point(self, *args, **kwargs):
        return self.archive.point(*args, **kwargs)

    def scenarios(self, *args, **kwargs):
        return self.archive.scenarios(*args, **kwargs)

    def conditional(self, *args, **kwargs):
        return self.archive.conditional(*args, **kwargs)

def make_plan(day, stock, archive, price_day, physical=Physical()):
    if price_day.day != day:
        raise ValueError('Price forecast is for another decision origin')
    view = DecisionArchive(archive, price_day)
    plan, meta = q2_make_plan(day, stock, view, CONFIG, physical)
    meta.update(price_information=price_day.metadata(),
        frozen_base_model='src.q2_v4.runtime.PRIMARY_CONFIGS[F_fast]',
        q4_changes='price profile only; forecast/controller/physical/terminal rules unchanged')
    meta['continuation'].update(price_day.metadata())
    return plan, meta, view

def execute_actions(day, stock, plan, decision_archive, physical=Physical()):
    """Execute Q2 F-fast under the decision price; no realized price is supplied."""
    return q2_execute_day(day, stock, plan, decision_archive, CONFIG, physical)

def settle_actual(plan, decision_ledger, load, pv, initial_soc, actual_price, physical=Physical()):
    """Independent actual-price cash, with exactly the already-fixed actions."""
    args = dict(eta_charge=physical.eta_c, eta_discharge=physical.eta_d,
        soc_min=physical.s_min, soc_max=physical.s_max, bus_limit=physical.power_energy,
        emergency_multiplier=physical.emergency_multiple, enforce_m0=False)
    ledger = settle_schedule(plan.q, decision_ledger['c'], decision_ledger['b'], load, pv,
        initial_soc, actual_price, **args)
    changes = {key: float(np.max(np.abs(ledger[key] - decision_ledger[key]))) for key in PHYSICAL_FIELDS}
    if max(changes.values()) != 0.:
        raise AssertionError(('Settlement changed physical actions/allocations', changes))
    audit = independent_recalculator(plan.q, ledger['c'], ledger['b'], load, pv,
        initial_soc, actual_price, ledger, **args)
    if not audit['passed']:
        raise AssertionError(audit)
    audit.update(physical_difference_vs_decision_price=changes,
        realized_price_not_supplied_to_action_function=True)
    return ledger, audit
