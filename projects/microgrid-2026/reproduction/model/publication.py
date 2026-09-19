"""Official PV forecast paths and causally available paired historical errors."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import hashlib
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from model.history import DT, START_DATE, T, DailyData, ForecastArchive

ISSUE_HOURS = (0, 6, 12, 18)
_EPOCH = datetime.combine(START_DATE, time.min)

def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array

def _time_at_boundary(boundary: int) -> str:
    return (_EPOCH + timedelta(minutes=10 * int(boundary))).isoformat(timespec="minutes")

def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)

@dataclass(frozen=True)
class OfficialForecastTable:
    """Read-only official table; the leading dimensions are day and issue hour."""

    hourly_kw: np.ndarray
    dates: tuple[date, ...]
    source_hashes: dict[str, str]
    sheet_name: str = "Sheet1"

def load_official_forecasts(root: str | Path | None = None) -> OfficialForecastTable:
    """Read attachment 3 without modifying it, validating every date/publication."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    folder = root / "problem" / "attachments" / "C题"
    workbook = load_workbook(folder / "附件3.xlsx", read_only=True, data_only=True)
    try:
        if len(workbook.worksheets) != 1:
            raise ValueError("Attachment 3 must contain one forecast sheet")
        sheet = workbook.worksheets[0]
        rows = list(sheet.iter_rows(values_only=True))
        if len(rows) != 1461 or len(rows[0]) != 26:
            raise ValueError("Attachment 3 differs from 365 days x 4 issues x 24 leads")
        if tuple(rows[0][2:]) != tuple(f"预报{h}小时" for h in range(1, 25)):
            raise ValueError("Attachment 3 lead-hour column labels are not 1 through 24")
        dates = tuple(START_DATE + timedelta(days=d) for d in range(365))
        hourly = np.empty((365, 4, 24), dtype=float)
        for d, expected_date in enumerate(dates):
            for issue_index, issue_hour in enumerate(ISSUE_HOURS):
                row = rows[1 + 4 * d + issue_index]
                raw_date = row[0]
                if isinstance(raw_date, datetime):
                    row_date = raw_date.date()
                elif isinstance(raw_date, date):
                    row_date = raw_date
                elif raw_date in (None, ""):
                    row_date = None
                else:
                    row_date = datetime.strptime(str(raw_date).strip(), "%Y-%m-%d").date()
                if (issue_index == 0 and row_date != expected_date) or (
                    issue_index > 0 and row_date not in (None, expected_date)
                ):
                    raise ValueError(f"Unexpected date in attachment 3 row {2 + 4*d + issue_index}")
                raw_hour = row[1]
                if isinstance(raw_hour, (datetime, time)):
                    parsed_hour = raw_hour.hour if raw_hour.minute == raw_hour.second == 0 else -1
                else:
                    parts = str(raw_hour).strip().split(":")
                    parsed_hour = int(parts[0]) if len(parts) == 2 and int(parts[1]) == 0 else -1
                if parsed_hour != issue_hour:
                    raise ValueError("Attachment 3 publication order differs from 00/06/12/18")
                hourly[d, issue_index] = np.asarray(row[2:], dtype=float)
        if not np.isfinite(hourly).all() or (hourly < 0).any():
            raise ValueError("Official forecast powers must be finite and nonnegative")
        sheet_name = sheet.title
    finally:
        workbook.close()
    hashes = {
        f"problem/attachments/C题/附件{n}.xlsx": hashlib.sha256(
            (folder / f"附件{n}.xlsx").read_bytes()
        ).hexdigest()
        for n in (1, 2, 3)
    }
    return OfficialForecastTable(_readonly(hourly), dates, hashes, sheet_name)

class OfficialForecastArchive:
    """Official PV centers plus Q2 load centers at the same historical origin."""

    def __init__(self, data: DailyData, root: str | Path | None = None,
                 load_archive: ForecastArchive | None = None,
                 official: OfficialForecastTable | None = None):
        self.data = data
        self.official = official if official is not None else load_official_forecasts(root)
        self.load_archive = load_archive if load_archive is not None else ForecastArchive(data)
        if data.load.shape != data.pv.shape or data.load.ndim != 2 or data.load.shape[1] != T:
            raise ValueError("Actual load/PV must be aligned day-by-slot arrays")
        if self.official.hourly_kw.shape != (len(self.official.dates), 4, 24):
            raise ValueError("Official forecasts must have shape (days, 4, 24)")
        expected_dates = tuple(START_DATE + timedelta(days=d)
                               for d in range(len(self.official.dates)))
        if self.official.dates != expected_dates:
            raise ValueError("Official forecast dates must be consecutive from START_DATE")
        if not np.isfinite(self.official.hourly_kw).all() or (self.official.hourly_kw < 0).any():
            raise ValueError("Official forecast powers must be finite and nonnegative")
        self._points: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict]] = {}
        self._errors: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict]] = {}

    def _origin(self, day: int, issue_hour: int,
                as_of_abs_slot: int | None = None) -> tuple[int, int, int]:
        day, issue_hour = _integer(day, "day"), _integer(issue_hour, "issue_hour")
        if not 0 <= day < len(self.official.dates) or issue_hour not in ISSUE_HOURS:
            raise ValueError("Need an available official forecast day and issue hour 0/6/12/18")
        boundary = day * T + issue_hour * 6
        as_of = boundary if as_of_abs_slot is None else _integer(as_of_abs_slot, "as_of_abs_slot")
        if as_of < boundary:
            raise ValueError("Requested forecast has not been published at this information boundary")
        if boundary > self.data.load.size:
            raise ValueError("The completed-interval anchor is absent from the supplied actual history")
        # A later as_of permits retrieval of an old publication, but never moves
        # that publication's actual-data fitting or residual-availability cutoff.
        return day, issue_hour, boundary

    def _metadata(self, day: int, issue_hour: int, boundary: int) -> dict:
        source_row = 2 + day * 4 + ISSUE_HOURS.index(issue_hour)
        return {'origin_day': day, 'issue_hour': issue_hour, 'published_at': _time_at_boundary(boundary), 'forecast_origin_abs_slot': boundary, 'latest_target_actual_abs_slot': None, 'latest_actual_abs_slot_used': boundary - 1 if boundary else None, 'historical_days': []}

    def residuals(self, day: int, issue_hour: int = 0, K: int = 30, *,
                  as_of_abs_slot: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Latest at most 30 same-publication complete 24-hour forecast errors."""
        day, issue_hour, boundary = self._origin(day, issue_hour, as_of_abs_slot)
        K = _integer(K, "K")
        if not 1 <= K <= 30:
            raise ValueError("K must be between 1 and the fixed 30-path history limit")
        key = day, issue_hour, K
        if key not in self._errors:
            offset = issue_hour * 6
            indices = np.arange(max(0, day - K), day, dtype=int)
            load_errors, pv_errors = [], []
            release_boundaries, completed_boundaries = [], []
            for j in indices:
                start = int(j) * T + offset
                end = start + T
                if end > boundary or end > self.data.load.size:
                    raise ValueError("A historical 24-hour target path is not fully observable")
                load_hat, pv_hat, _ = self.point(int(j), issue_hour)
                load_errors.append(self.data.load.reshape(-1)[start:end] - load_hat)
                pv_errors.append(self.data.pv.reshape(-1)[start:end] - pv_hat)
                release_boundaries.append(start)
                completed_boundaries.append(end)
            e_load = np.asarray(load_errors, dtype=float).reshape((-1, T))
            e_pv = np.asarray(pv_errors, dtype=float).reshape((-1, T))
            meta = self._metadata(day, issue_hour, boundary)
            meta.update(historical_days=indices.tolist(), latest_target_actual_abs_slot=max(completed_boundaries) - 1 if len(indices) else None, sample_count=len(indices))
            self._errors[key] = _readonly(e_load), _readonly(e_pv), _readonly(indices), meta
        e_load, e_pv, indices, meta = self._errors[key]
        return e_load, e_pv, indices, dict(meta)

    def scenarios(self, day: int, issue_hour: int = 0, K: int = 30, *,
                  as_of_abs_slot: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Add paired historical errors to the current center and clip at zero."""
        load_hat, pv_hat, point_meta = self.point(day, issue_hour, as_of_abs_slot=as_of_abs_slot)
        e_load, e_pv, _, residual_meta = self.residuals(day, issue_hour, K, as_of_abs_slot=as_of_abs_slot)
        meta = {**point_meta, **residual_meta, "scenario_method": "paired additive residuals with nonnegative clipping"}
        return np.maximum(load_hat[None, :] + e_load, 0), np.maximum(pv_hat[None, :] + e_pv, 0), meta

    def z2_signal_pairs(self, day: int, K: int = 30, *,
                        as_of_abs_slot: int | None = None) -> dict:
        """Past 00:00 errors paired with that same day's later 06:00 signal."""
        day, _, boundary = self._origin(day, 0, as_of_abs_slot)
        e_load, e_pv, indices, base_meta = self.residuals(day, 0, K, as_of_abs_slot=as_of_abs_slot)
        signals, pv00_remaining, pv06_remaining = [], [], []
        for j in indices:
            hist_signal_boundary = int(j) * T + 36
            if hist_signal_boundary > boundary or (int(j) + 1) * T > boundary:
                raise ValueError("Z2 signal and paired target residual must both already be historical")
            forecast00 = self.point(int(j), 0)[1]
            forecast06 = self.point(int(j), 6)[1]
            earlier, later = float(forecast00[36:].sum()), float(forecast06[:108].sum())
            pv00_remaining.append(earlier)
            pv06_remaining.append(later)
            signals.append(later - earlier)
        signal_array = np.asarray(signals, dtype=float)
        thresholds = np.quantile(signal_array, [1/3, 2/3]) if len(signals) else np.empty(0)
        groups = np.searchsorted(thresholds, signal_array, side="right").astype(int)
        counts = [int(np.sum(groups == g)) for g in range(3)]
        meta = dict(base_meta)
        pass
        return {
            "load_residuals": e_load, "pv_residuals": e_pv,
            "historical_days": indices, "signals": _readonly(signal_array),
            "thresholds": _readonly(np.asarray(thresholds)), "groups": _readonly(groups),
            "pv00_remaining_kwh": _readonly(np.asarray(pv00_remaining)),
            "pv06_remaining_kwh": _readonly(np.asarray(pv06_remaining)),
            "metadata": meta,
        }
