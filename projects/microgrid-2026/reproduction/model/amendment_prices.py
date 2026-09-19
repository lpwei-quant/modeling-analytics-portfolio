"""Adapt the existing attachment-4 origin views to the unchanged Q3 engine."""
from __future__ import annotations

import numpy as np

from model.amendments import PriceInputs
from model.prices import MODES, PriceArchive, load_prices

def availability_rules(mode: str) -> dict:
    if mode not in MODES:
        raise ValueError('Unknown attachment-4 information mode')
    return dict(
        mode=mode,
        decision_source='actual[d]' if mode == 'known_day_ahead' else 'actual[d-1]',
        decision_classification=('Assumed available in full at day d 00:00'
            if mode == 'known_day_ahead' else 'Derived previous-day persistence forecast'),
        settlement_source='Observed attachment-4 actual[d]; used only for ex-post settlement',
        nextday_source='the decision profile frozen at origin d; never actual[d+1]',
        nextday_classification='Assumed persistence forecast',
        price_profile_frozen_for_intraday_control=True,
        current_full_price_profile_known_day_ahead_assumed=mode == 'known_day_ahead',
        actual_nextday_price_not_read=True,
        day_zero_previous_price='unavailable; attachment-1 tariff is an interface placeholder only',
        day_zero_is_executable=False,
        evaluation_day_indices=[31, 364],
        initial_soc_source='unchanged common Q2 F-fast January warmup; not rerun with attachment 4',
        price_unit='yuan/kWh',
    )

def build_price_inputs(data, price_archive: PriceArchive, mode: str) -> PriceInputs:
    """Create three immutable 365 x 144 matrices for the existing Q3 API."""
    availability_rules(mode)
    if tuple(data.dates) != tuple(price_archive.dates):
        raise ValueError('Price dates do not align with load/PV observations')
    decision = np.empty((365, 144), dtype=float)
    for day in range(365):
        if day == 0 and mode == 'previous_day_forecast':
            # Required array-interface row only. There is no observed day -1,
            # and neither production nor the simulation helper may execute it.
            decision[day] = data.price
        else:
            decision[day] = price_archive.at_origin(day, mode).decision
    provenance = (
        'Q4-3; Observed attachment-4 actual settlement; '
        + ('Assumed entire current-day price curve known at 00:00; '
            if mode == 'known_day_ahead' else 'Derived previous-day decision-price persistence; ')
        + 'Assumed tomorrow-price persistence from this origin decision profile; '
        + 'full daily profile frozen intraday; no actual tomorrow-price access; '
        + 'day-zero previous-day row is an unexecuted attachment-1 interface placeholder'
    )
    return PriceInputs(decision, price_archive.actual, decision, provenance)
