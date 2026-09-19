"""Causal historical/official PV fusion with January calibration only."""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import hashlib
import math
from pathlib import Path

import numpy as np

from model.history import DT, START_DATE, T, DailyData, ForecastArchive
from model.publication import ISSUE_HOURS, OfficialForecastArchive, OfficialForecastTable, _integer, _readonly, _time_at_boundary

JANUARY_DAYS = 31
FREEZE_BOUNDARY = JANUARY_DAYS * T
COLD_START_WEIGHT = 0.5

def fit_convex_l1_weight(history: np.ndarray, official: np.ndarray,
                         actual: np.ndarray) -> dict:
    """Exact piecewise-linear minimizer of sum |(1-w)h+w*o-y|, 0<=w<=1."""
    history, official, actual = (np.asarray(x, dtype=float) for x in (history, official, actual))
    if history.shape != official.shape or history.shape != actual.shape or not history.size:
        raise ValueError("L1 calibration needs nonempty equally shaped arrays")
    if not all(np.isfinite(x).all() for x in (history, official, actual)):
        raise ValueError("L1 calibration inputs must be finite")
    history, official, actual = history.ravel(), official.ravel(), actual.ravel()
    delta = official - history
    active = delta != 0
    if not np.isfinite(delta).all():
        raise ValueError("Calibration differences must be finite")
    if active.any():
        with np.errstate(over="ignore", divide="ignore"):
            points = np.clip((actual[active] - history[active]) / delta[active], 0.0, 1.0)
        masses = np.abs(delta[active])
        order = np.argsort(points, kind="stable")
        points, masses = points[order], masses[order]
        ratios = [float(m).as_integer_ratio() for m in masses]
        denominator = max(pair[1] for pair in ratios)
        integer_masses = [num * (denominator // den) for num, den in ratios]
        total, cumulative = sum(integer_masses), 0
        for index, mass in enumerate(integer_masses):
            cumulative += mass
            if 2 * cumulative >= total:
                low = float(points[index])
                high = float(points[index + 1]) if 2*cumulative == total and index+1 < len(points) else low
                break
        distinct = int(np.unique(points).size)
    else:
        low, high, distinct = 0.0, 1.0, 0
    weight = (low + high) / 2.0
    errors = (1.0-weight)*history + weight*official - actual
    sae = math.fsum(np.abs(errors).tolist())
    return {
        "official_weight": weight, "history_weight": 1.0-weight,
        "optimal_weight_interval": [low, high], "nonunique_optimum": high > low,
        "selection_rule": "midpoint of the complete optimal interval in [0,1]",
        "solver": "weighted median of clipped L1 breakpoints; exact binary-float mass sums",
        "target_count": int(actual.size), "active_breakpoint_count": int(active.sum()),
        "distinct_clipped_breakpoints": distinct,
        "zero_difference_target_count": int((~active).sum()),
        "minimum_sae_kwh": sae, "minimum_mae_kwh_per_interval": sae / actual.size,
    }

def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()

class FusedForecastArchive(OfficialForecastArchive):
    """V1-compatible archive with causal convex PV fusion and fixed load model."""

    def __init__(self, data: DailyData, root: str | Path | None = None,
                 load_archive: ForecastArchive | None = None,
                 official: OfficialForecastTable | None = None):
        super().__init__(data, root=root, load_archive=load_archive, official=official)
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
        self._components: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict]] = {}
        self._weight_fits: dict[tuple[int, int], dict] = {}

    def components(self, day: int, issue_hour: int = 0, *,
                   as_of_abs_slot: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Load, historical PV and official PV from this publication's history."""
        day, issue_hour, boundary = self._origin(day, issue_hour, as_of_abs_slot)
        key = day, issue_hour
        if key not in self._components:
            offset = issue_hour * 6
            load_today, pv_today = self.load_archive.point(day, day)
            if offset:
                load_next, pv_next = self.load_archive.point(day, day + 1)
                load_hat = np.r_[load_today[offset:], load_next[:offset]]
                historical = np.r_[pv_today[offset:], pv_next[:offset]]
            else:
                load_hat, historical = load_today.copy(), pv_today.copy()
            if boundary:
                anchor = float(self.data.pv.reshape(-1)[boundary - 1] / DT)
                source = "attachment2 actual PV at last completed interval endpoint"
            else:
                anchor = float(self.data.pv_template[-1] / DT)
                source = "Assumed cold start: attachment1 24:00 PV template endpoint"
            powers = self.official.hourly_kw[day, ISSUE_HOURS.index(issue_hour)]
            official_pv = np.interp(np.arange(1, T + 1, dtype=float) * DT,
                                    np.arange(25, dtype=float), np.r_[anchor, powers]) * DT
            if not all(np.isfinite(x).all() for x in (load_hat, historical, official_pv)):
                raise ValueError("A forecast consumed nonfinite available data")
            meta = self._metadata(day, issue_hour, boundary)
            pass
            self._components[key] = (_readonly(load_hat), _readonly(historical),
                                     _readonly(official_pv), meta)
        loads, historical, official_pv, meta = self._components[key]
        return loads, historical, official_pv, dict(meta)

    def _fit_at(self, cutoff: int, issue_hour: int) -> dict:
        """Fit only January paths fully observable strictly before cutoff."""
        cutoff = min(int(cutoff), FREEZE_BOUNDARY)
        key = cutoff, issue_hour
        if key not in self._weight_fits:
            offset = issue_hour * 6
            count = max(0, min(JANUARY_DAYS, (cutoff-offset-T)//T + 1))
            days = list(range(count))
            ends = [(j+1)*T+offset for j in days]
            if ends and max(ends) > min(cutoff, self.data.pv.size):
                raise ValueError("The requested calibration needs unavailable target actuals")
            if count:
                centers = [self.components(j, issue_hour) for j in days]
                history = np.vstack([value[1] for value in centers])
                official = np.vstack([value[2] for value in centers])
                actual = np.vstack([self.data.pv.reshape(-1)[j*T+offset:(j+1)*T+offset] for j in days])
                record = fit_convex_l1_weight(history, official, actual)
                record["classification"] = "Derived from completed January historical-origin forecasts"
            else:
                history = official = actual = np.empty((0, T))
                record = {
                    "official_weight": COLD_START_WEIGHT, "history_weight": 1-COLD_START_WEIGHT,
                    "optimal_weight_interval": [0.0, 1.0], "nonunique_optimum": True,
                    "selection_rule": "Assumed cold-start weight 0.5; no realized calibration target",
                    "solver": "none: no complete January calibration paths",
                    "target_count": 0, "active_breakpoint_count": 0,
                    "distinct_clipped_breakpoints": 0, "zero_difference_target_count": 0,
                    "minimum_sae_kwh": None, "minimum_mae_kwh_per_interval": None,
                    "classification": "Assumed cold start",
                }
            record.update(issue_hour=issue_hour, fit_cutoff_abs_slot_exclusive=cutoff, sample_count=count, historical_days=days, latest_target_actual_abs_slot=max(ends) - 1 if ends else None, calibration_arrays_sha256=_array_hash(history, official, actual), objective='sum of absolute errors over all 144 ten-minute targets of every eligible path')
            self._weight_fits[key] = record
        return deepcopy(self._weight_fits[key])

    def weight_at(self, day: int, issue_hour: int = 0, *, as_of_abs_slot: int | None = None) -> dict:
        """The historical weight actually available at this publication time."""
        day, issue_hour, boundary = self._origin(day, issue_hour, as_of_abs_slot)
        result = self._fit_at(min(boundary, FREEZE_BOUNDARY), issue_hour)
        result.update(origin_day=day, forecast_origin_abs_slot=boundary, regime='january_walk_forward' if boundary < FREEZE_BOUNDARY else 'february_frozen')
        return result

    def point(self, day: int, issue_hour: int = 0, *,
              as_of_abs_slot: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        day, issue_hour, _ = self._origin(day, issue_hour, as_of_abs_slot)
        key = day, issue_hour
        if key not in self._points:
            loads, historical, official, meta = self.components(day, issue_hour)
            fit = self.weight_at(day, issue_hour)
            weight = fit["official_weight"]
            fused = (1-weight)*historical + weight*official
            meta.update(official_weight=weight, history_weight=1 - weight)
            self._points[key] = loads, _readonly(fused), meta
        loads, fused, meta = self._points[key]
        return loads, fused, dict(meta)

