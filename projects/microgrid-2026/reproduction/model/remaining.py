"""Remaining-day calibration with the unchanged complete 24h forecast API."""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import hashlib

import numpy as np

from model.history import START_DATE, T
from model.publication import ISSUE_HOURS, _time_at_boundary
from model.fusion import COLD_START_WEIGHT, FREEZE_BOUNDARY, JANUARY_DAYS, FusedForecastArchive as _FullWindowFusedArchive, _array_hash, fit_convex_l1_weight

FIT_WINDOW = "remaining_same_day"
REVISION_STATUS = "post_annual_diagnostics_revision_with_january_only_numeric_calibration"

class RemainingWindowFusedForecastArchive(_FullWindowFusedArchive):
    """V2-compatible centers/residuals with remaining-day January L1 weights."""

    def _fit_at(self, cutoff: int, issue_hour: int) -> dict:
        """Use January same-day windows with end=(j+1)*144 <= cutoff."""
        cutoff = min(int(cutoff), FREEZE_BOUNDARY)
        key = cutoff, issue_hour
        if key not in self._weight_fits:
            offset = issue_hour*6
            length = T-offset
            count = max(0, min(JANUARY_DAYS, cutoff//T))
            days = list(range(count))
            starts = [j*T+offset for j in days]
            ends = [(j+1)*T for j in days]
            if ends and max(ends) > min(cutoff, self.data.pv.size):
                raise ValueError("The requested calibration needs unavailable remaining-day actuals")
            if count:
                components = [self.components(j, issue_hour) for j in days]
                history = np.vstack([value[1][:length] for value in components])
                official = np.vstack([value[2][:length] for value in components])
                actual = np.vstack([self.data.pv.reshape(-1)[start:end] for start, end in zip(starts, ends)])
                record = fit_convex_l1_weight(history, official, actual)
                record["classification"] = "Derived from completed January historical-origin remaining-day forecasts"
            else:
                history = official = actual = np.empty((0, length))
                record = {
                    "official_weight": COLD_START_WEIGHT, "history_weight": 1-COLD_START_WEIGHT,
                    "optimal_weight_interval": [0.0, 1.0], "nonunique_optimum": True,
                    "selection_rule": "Assumed cold-start weight 0.5; no realized calibration target",
                    "solver": "none: no complete January remaining-day calibration windows",
                    "target_count": 0, "active_breakpoint_count": 0,
                    "distinct_clipped_breakpoints": 0, "zero_difference_target_count": 0,
                    "minimum_sae_kwh": None, "minimum_mae_kwh_per_interval": None,
                    "classification": "Assumed cold start",
                }
            record.update(issue_hour=issue_hour, fit_cutoff_abs_slot_exclusive=cutoff, sample_count=count, historical_days=days, latest_target_actual_abs_slot=max(ends) - 1 if ends else None, calibration_arrays_sha256=_array_hash(history, official, actual), objective=f'sum of absolute errors over all {length} remaining-day ten-minute targets of every eligible publication')
            self._weight_fits[key] = record
        return deepcopy(self._weight_fits[key])

# Compatible imports for the existing runtime and the new runner.
FusedForecastArchive = RemainingWindowFusedForecastArchive
