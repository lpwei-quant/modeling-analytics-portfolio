"""Authoritative Q1 data reconstruction and indexing.

This module reads the local CUMCM attachments without modifying them.  It
constructs crop/plot masters, land-season compatibility, land-specific
economic parameters, physical opportunity adjacency, and the 2023 initial
state required by the Question 1 MILP.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


YEARS = tuple(range(2024, 2031))
SEASON_SINGLE = "单季"
SEASON_FIRST = "第一季"
SEASON_SECOND = "第二季"
SPECIAL_SECOND_VEGETABLES = frozenset({"大白菜", "白萝卜", "红萝卜"})
OPEN_LAND_TYPES = frozenset({"平旱地", "梯田", "山坡地"})
GREENHOUSE_TYPES = frozenset({"普通大棚", "智慧大棚"})
IRRIGATED_LAND = "水浇地"
ORDINARY_GREENHOUSE = "普通大棚"
SMART_GREENHOUSE = "智慧大棚"


class Q1DataError(RuntimeError):
    """Raised when an authoritative input violates the Q1 data contract."""


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", "", str(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_price_interval(value: object) -> tuple[float, float]:
    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    if len(numbers) != 2:
        raise Q1DataError(f"Invalid price interval: {value!r}")
    lower, upper = map(float, numbers)
    if lower < 0 or upper < lower:
        raise Q1DataError(f"Invalid non-negative price interval: {value!r}")
    return lower, upper


@dataclass(frozen=True)
class Plot:
    name: str
    land_type: str
    area_mu: float


@dataclass(frozen=True)
class Crop:
    crop_id: int
    name: str
    crop_type: str

    @property
    def is_bean(self) -> bool:
        return "豆类" in self.crop_type

    @property
    def is_grain(self) -> bool:
        return self.crop_type.startswith("粮食")

    @property
    def is_vegetable(self) -> bool:
        return self.crop_type.startswith("蔬菜")

    @property
    def is_fungus(self) -> bool:
        return self.crop_type == "食用菌"


@dataclass(frozen=True)
class EconomicParameter:
    crop_id: int
    land_type: str
    season: str
    yield_jin_per_mu: float
    cost_yuan_per_mu: float
    price_lower_yuan_per_jin: float
    price_upper_yuan_per_jin: float
    source_excel_row: int
    provenance: str

    @property
    def midpoint_price_yuan_per_jin(self) -> float:
        return (self.price_lower_yuan_per_jin + self.price_upper_yuan_per_jin) / 2.0


@dataclass(frozen=True)
class Planting2023:
    source_excel_row: int
    plot: str
    crop_id: int
    season: str
    area_mu: float


@dataclass(frozen=True)
class OpportunityEdge:
    plot: str
    year_from: int
    season_from: str
    year_to: int
    season_to: str


@dataclass(frozen=True)
class Q1Data:
    project_root: Path
    plots: dict[str, Plot]
    crops: dict[int, Crop]
    parameters: dict[tuple[int, str, str], EconomicParameter]
    plantings_2023: tuple[Planting2023, ...]
    expected_sales_proxy: dict[tuple[int, str], float]
    compatible_cells: tuple[tuple[str, int, int, str], ...]
    available_plot_seasons: tuple[tuple[str, int, str], ...]
    adjacency_edges: tuple[OpportunityEdge, ...]
    initial_last_crops: dict[str, frozenset[int]]
    initial_first_seasons: dict[str, tuple[str, ...]]
    bean_area_2023: dict[str, float]
    source_hashes: dict[str, str]

    @property
    def bean_crop_ids(self) -> frozenset[int]:
        return frozenset(crop_id for crop_id, crop in self.crops.items() if crop.is_bean)

    @property
    def irrigated_plots(self) -> tuple[str, ...]:
        return tuple(name for name, plot in self.plots.items() if plot.land_type == IRRIGATED_LAND)

    def compatible_crops(self, plot: str, season: str) -> tuple[int, ...]:
        land_type = self.plots[plot].land_type
        return compatible_crop_ids(land_type, season, self.crops)


def compatible_crop_ids(
    land_type: str, season: str, crops: dict[int, Crop]
) -> tuple[int, ...]:
    result: list[int] = []
    for crop_id, crop in crops.items():
        allowed = False
        if land_type in OPEN_LAND_TYPES:
            allowed = season == SEASON_SINGLE and crop.is_grain and crop.name != "水稻"
        elif land_type == IRRIGATED_LAND:
            if season == SEASON_SINGLE:
                allowed = crop.name == "水稻"
            elif season == SEASON_FIRST:
                allowed = crop.is_vegetable and crop.name not in SPECIAL_SECOND_VEGETABLES
            elif season == SEASON_SECOND:
                allowed = crop.name in SPECIAL_SECOND_VEGETABLES
        elif land_type == ORDINARY_GREENHOUSE:
            if season == SEASON_FIRST:
                allowed = crop.is_vegetable and crop.name not in SPECIAL_SECOND_VEGETABLES
            elif season == SEASON_SECOND:
                allowed = crop.is_fungus
        elif land_type == SMART_GREENHOUSE:
            allowed = (
                season in {SEASON_FIRST, SEASON_SECOND}
                and crop.is_vegetable
                and crop.name not in SPECIAL_SECOND_VEGETABLES
            )
        else:
            raise Q1DataError(f"Unknown land type: {land_type!r}")
        if allowed:
            result.append(crop_id)
    return tuple(sorted(result))


def potential_seasons(land_type: str) -> tuple[str, ...]:
    if land_type in OPEN_LAND_TYPES:
        return (SEASON_SINGLE,)
    if land_type == IRRIGATED_LAND:
        return (SEASON_SINGLE, SEASON_FIRST, SEASON_SECOND)
    if land_type in GREENHOUSE_TYPES:
        return (SEASON_FIRST, SEASON_SECOND)
    raise Q1DataError(f"Unknown land type: {land_type!r}")


def read_plots(attachment_1: Path) -> dict[str, Plot]:
    frame = pd.read_excel(attachment_1, sheet_name="乡村的现有耕地", usecols="A:C")
    frame.columns = ["plot", "land_type", "area_mu"]
    plots: dict[str, Plot] = {}
    for row in frame.itertuples(index=False):
        name = normalize_text(row.plot)
        if not name:
            continue
        land_type = normalize_text(row.land_type)
        area = float(row.area_mu)
        if name in plots:
            raise Q1DataError(f"Duplicate plot: {name}")
        if not math.isfinite(area) or area < 0:
            raise Q1DataError(f"Invalid plot area for {name}: {area}")
        plots[name] = Plot(name=name, land_type=land_type, area_mu=area)
    if len(plots) != 54:
        raise Q1DataError(f"Expected 54 plots, found {len(plots)}")
    return plots


def read_crops(attachment_1: Path) -> dict[int, Crop]:
    frame = pd.read_excel(attachment_1, sheet_name="乡村种植的农作物", usecols="A:C")
    numeric_ids = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    frame = frame[numeric_ids.notna()].copy()
    frame.columns = ["crop_id", "name", "crop_type"]
    crops: dict[int, Crop] = {}
    for row in frame.itertuples(index=False):
        crop_id = int(row.crop_id)
        crop = Crop(
            crop_id=crop_id,
            name=normalize_text(row.name),
            crop_type=normalize_text(row.crop_type),
        )
        if crop_id in crops:
            raise Q1DataError(f"Duplicate crop id: {crop_id}")
        crops[crop_id] = crop
    if len(crops) != 41:
        raise Q1DataError(f"Expected 41 crops, found {len(crops)}")
    return crops


def read_parameters(
    attachment_2: Path, crops: dict[int, Crop]
) -> dict[tuple[int, str, str], EconomicParameter]:
    frame = pd.read_excel(
        attachment_2, sheet_name="2023年统计的相关数据", usecols="A:H"
    )
    frame.columns = [
        "sequence",
        "crop_id",
        "crop_name",
        "land_type",
        "season",
        "yield_value",
        "cost_value",
        "price_interval",
    ]
    frame["excel_row"] = frame.index + 2
    frame = frame[pd.to_numeric(frame["sequence"], errors="coerce").notna()].copy()
    parameters: dict[tuple[int, str, str], EconomicParameter] = {}
    for row in frame.itertuples(index=False):
        crop_id = int(row.crop_id)
        crop_name = normalize_text(row.crop_name)
        if crop_id not in crops or crops[crop_id].name != crop_name:
            raise Q1DataError(f"Parameter crop mismatch at Excel row {row.excel_row}")
        land_type = normalize_text(row.land_type)
        season = normalize_text(row.season)
        lower, upper = parse_price_interval(row.price_interval)
        parameter = EconomicParameter(
            crop_id=crop_id,
            land_type=land_type,
            season=season,
            yield_jin_per_mu=float(row.yield_value),
            cost_yuan_per_mu=float(row.cost_value),
            price_lower_yuan_per_jin=lower,
            price_upper_yuan_per_jin=upper,
            source_excel_row=int(row.excel_row),
            provenance="附件2直接参数行",
        )
        if min(
            parameter.yield_jin_per_mu,
            parameter.cost_yuan_per_mu,
            parameter.price_lower_yuan_per_jin,
            parameter.price_upper_yuan_per_jin,
        ) < 0:
            raise Q1DataError(f"Negative parameter at Excel row {row.excel_row}")
        key = (crop_id, land_type, season)
        if key in parameters:
            raise Q1DataError(f"Duplicate parameter key: {key}")
        parameters[key] = parameter

    ordinary_first = [
        value
        for value in parameters.values()
        if value.land_type == ORDINARY_GREENHOUSE and value.season == SEASON_FIRST
    ]
    for source in ordinary_first:
        key = (source.crop_id, SMART_GREENHOUSE, SEASON_FIRST)
        if key in parameters:
            raise Q1DataError(f"Unexpected direct smart-first parameter: {key}")
        parameters[key] = EconomicParameter(
            crop_id=source.crop_id,
            land_type=SMART_GREENHOUSE,
            season=SEASON_FIRST,
            yield_jin_per_mu=source.yield_jin_per_mu,
            cost_yuan_per_mu=source.cost_yuan_per_mu,
            price_lower_yuan_per_jin=source.price_lower_yuan_per_jin,
            price_upper_yuan_per_jin=source.price_upper_yuan_per_jin,
            source_excel_row=source.source_excel_row,
            provenance="附件2注释：智慧大棚第一季等同普通大棚第一季",
        )
    return parameters


def read_plantings_2023(
    attachment_2: Path, plots: dict[str, Plot], crops: dict[int, Crop]
) -> tuple[Planting2023, ...]:
    frame = pd.read_excel(
        attachment_2, sheet_name="2023年的农作物种植情况", usecols="A:F"
    )
    frame.columns = ["plot", "crop_id", "crop_name", "crop_type", "area_mu", "season"]
    frame["excel_row"] = frame.index + 2
    frame = frame[frame["crop_id"].notna()].copy()
    frame["plot"] = frame["plot"].ffill().map(normalize_text)
    records: list[Planting2023] = []
    for row in frame.itertuples(index=False):
        plot = normalize_text(row.plot)
        crop_id = int(row.crop_id)
        crop_name = normalize_text(row.crop_name)
        season = normalize_text(row.season)
        area = float(row.area_mu)
        if plot not in plots:
            raise Q1DataError(f"Unknown 2023 plot at row {row.excel_row}: {plot}")
        if crop_id not in crops or crops[crop_id].name != crop_name:
            raise Q1DataError(f"2023 crop mismatch at row {row.excel_row}")
        if not math.isfinite(area) or area < 0:
            raise Q1DataError(f"Invalid 2023 area at row {row.excel_row}")
        records.append(
            Planting2023(
                source_excel_row=int(row.excel_row),
                plot=plot,
                crop_id=crop_id,
                season=season,
                area_mu=area,
            )
        )
    if len(records) != 87:
        raise Q1DataError(f"Expected 87 planting rows, found {len(records)}")
    return tuple(records)


def read_expected_sales_proxy(
    baseline_csv: Path, crops: dict[int, Crop]
) -> dict[tuple[int, str], float]:
    frame = pd.read_csv(baseline_csv, encoding="utf-8-sig")
    required = {
        "crop_id",
        "crop_name",
        "planting_season",
        "theoretical_production_jin",
        "baseline_expected_sales_proxy_jin",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise Q1DataError(f"Baseline CSV missing columns: {sorted(missing)}")
    proxy: dict[tuple[int, str], float] = {}
    for row in frame.itertuples(index=False):
        crop_id = int(row.crop_id)
        season = normalize_text(row.planting_season)
        value = float(row.baseline_expected_sales_proxy_jin)
        production = float(row.theoretical_production_jin)
        if crop_id not in crops or crops[crop_id].name != normalize_text(row.crop_name):
            raise Q1DataError(f"Baseline crop mismatch for crop {crop_id}")
        if value < 0 or not math.isclose(value, production, rel_tol=0.0, abs_tol=1e-8):
            raise Q1DataError(f"Baseline proxy mismatch for {(crop_id, season)}")
        key = (crop_id, season)
        if key in proxy:
            raise Q1DataError(f"Duplicate baseline key: {key}")
        proxy[key] = value
    return proxy


def build_available_plot_seasons(plots: dict[str, Plot]) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (plot.name, year, season)
        for plot in plots.values()
        for year in YEARS
        for season in potential_seasons(plot.land_type)
    )


def build_compatible_cells(
    plots: dict[str, Plot],
    crops: dict[int, Crop],
    parameters: dict[tuple[int, str, str], EconomicParameter],
) -> tuple[tuple[str, int, int, str], ...]:
    cells: list[tuple[str, int, int, str]] = []
    for plot in plots.values():
        for season in potential_seasons(plot.land_type):
            for crop_id in compatible_crop_ids(plot.land_type, season, crops):
                parameter_key = (crop_id, plot.land_type, season)
                if parameter_key not in parameters:
                    raise Q1DataError(f"Missing economic parameter for {parameter_key}")
                for year in YEARS:
                    cells.append((plot.name, crop_id, year, season))
    return tuple(cells)


def build_adjacency_edges(plots: dict[str, Plot]) -> tuple[OpportunityEdge, ...]:
    edges: list[OpportunityEdge] = []
    for plot in plots.values():
        if plot.land_type in OPEN_LAND_TYPES:
            for year in YEARS[:-1]:
                edges.append(
                    OpportunityEdge(plot.name, year, SEASON_SINGLE, year + 1, SEASON_SINGLE)
                )
        elif plot.land_type in GREENHOUSE_TYPES:
            for year in YEARS:
                edges.append(
                    OpportunityEdge(plot.name, year, SEASON_FIRST, year, SEASON_SECOND)
                )
            for year in YEARS[:-1]:
                edges.append(
                    OpportunityEdge(plot.name, year, SEASON_SECOND, year + 1, SEASON_FIRST)
                )
        elif plot.land_type == IRRIGATED_LAND:
            for year in YEARS:
                edges.append(
                    OpportunityEdge(plot.name, year, SEASON_FIRST, year, SEASON_SECOND)
                )
            for year in YEARS[:-1]:
                for season_from in (SEASON_SINGLE, SEASON_SECOND):
                    for season_to in (SEASON_SINGLE, SEASON_FIRST):
                        edges.append(
                            OpportunityEdge(
                                plot.name, year, season_from, year + 1, season_to
                            )
                        )
        else:
            raise Q1DataError(f"Unknown land type: {plot.land_type}")
    return tuple(edges)


def build_initial_state(
    plots: dict[str, Plot],
    crops: dict[int, Crop],
    plantings: Iterable[Planting2023],
) -> tuple[dict[str, frozenset[int]], dict[str, tuple[str, ...]], dict[str, float]]:
    by_plot: dict[str, list[Planting2023]] = {plot: [] for plot in plots}
    for record in plantings:
        by_plot[record.plot].append(record)

    last_crops: dict[str, frozenset[int]] = {}
    first_seasons: dict[str, tuple[str, ...]] = {}
    bean_area: dict[str, float] = {}
    for plot_name, plot in plots.items():
        records = by_plot[plot_name]
        if not records:
            raise Q1DataError(f"Plot {plot_name} has no 2023 planting record")
        if plot.land_type in OPEN_LAND_TYPES:
            last_season = SEASON_SINGLE
            first_seasons[plot_name] = (SEASON_SINGLE,)
        elif plot.land_type == IRRIGATED_LAND:
            seasons = {record.season for record in records}
            last_season = SEASON_SINGLE if SEASON_SINGLE in seasons else SEASON_SECOND
            first_seasons[plot_name] = (SEASON_SINGLE, SEASON_FIRST)
        elif plot.land_type in GREENHOUSE_TYPES:
            last_season = SEASON_SECOND
            first_seasons[plot_name] = (SEASON_FIRST,)
        else:
            raise Q1DataError(f"Unknown land type: {plot.land_type}")
        last_crops[plot_name] = frozenset(
            record.crop_id
            for record in records
            if record.season == last_season and record.area_mu > 0
        )
        if not last_crops[plot_name]:
            raise Q1DataError(f"Plot {plot_name} has no crop in its 2023 last opportunity")
        bean_area[plot_name] = math.fsum(
            record.area_mu for record in records if crops[record.crop_id].is_bean
        )
    return last_crops, first_seasons, bean_area


def load_q1_data(project_root: Path | str | None = None) -> Q1Data:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    attachment_1 = root / "data" / "附件1 (1).xlsx"
    attachment_2 = root / "data" / "附件2 (2).xlsx"
    baseline_csv = root / "research" / "baseline_2023.csv"
    decisions = root / "research" / "modeling_decisions_v1.md"
    for path in (attachment_1, attachment_2, baseline_csv, decisions):
        if not path.is_file():
            raise Q1DataError(f"Required authoritative input not found: {path}")

    plots = read_plots(attachment_1)
    crops = read_crops(attachment_1)
    parameters = read_parameters(attachment_2, crops)
    plantings = read_plantings_2023(attachment_2, plots, crops)
    expected_sales = read_expected_sales_proxy(baseline_csv, crops)
    compatible_cells = build_compatible_cells(plots, crops, parameters)
    available_plot_seasons = build_available_plot_seasons(plots)
    adjacency_edges = build_adjacency_edges(plots)
    last_crops, first_seasons, bean_area = build_initial_state(
        plots, crops, plantings
    )

    total_area = math.fsum(plot.area_mu for plot in plots.values())
    open_area = math.fsum(
        plot.area_mu for plot in plots.values() if plot.land_type not in GREENHOUSE_TYPES
    )
    greenhouse_area = total_area - open_area
    if not math.isclose(total_area, 1213.0, abs_tol=1e-9):
        raise Q1DataError(f"Total area mismatch: {total_area}")
    if not math.isclose(open_area, 1201.0, abs_tol=1e-9):
        raise Q1DataError(f"Open area mismatch: {open_area}")
    if not math.isclose(greenhouse_area, 12.0, abs_tol=1e-9):
        raise Q1DataError(f"Greenhouse area mismatch: {greenhouse_area}")

    return Q1Data(
        project_root=root,
        plots=plots,
        crops=crops,
        parameters=parameters,
        plantings_2023=plantings,
        expected_sales_proxy=expected_sales,
        compatible_cells=compatible_cells,
        available_plot_seasons=available_plot_seasons,
        adjacency_edges=adjacency_edges,
        initial_last_crops=last_crops,
        initial_first_seasons=first_seasons,
        bean_area_2023=bean_area,
        source_hashes={
            "data/附件1 (1).xlsx": sha256(attachment_1),
            "data/附件2 (2).xlsx": sha256(attachment_2),
            "research/baseline_2023.csv": sha256(baseline_csv),
            "research/modeling_decisions_v1.md": sha256(decisions),
        },
    )
