"""Reproducible Q2 uncertainty paths under the approved E-class assumptions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from q1_data import Q1Data, YEARS
from q2_config import DEFAULT_Q2_CONFIG, Q2Config, UniformSupport


InnovationKey = tuple[str, int, int]
DemandKey = tuple[int, int, int, str]
MultiplierKey = tuple[int, int, int]


@dataclass(frozen=True)
class Q2ScenarioPaths:
    count: int
    seed: int
    sampling_method: str
    probabilities: np.ndarray
    dimension_order: tuple[InnovationKey, ...]
    unit_draws: np.ndarray
    innovations: dict[InnovationKey, np.ndarray]
    demand_jin: dict[DemandKey, float]
    yield_multiplier: dict[MultiplierKey, float]
    price_multiplier: dict[MultiplierKey, float]

    @property
    def scenario_ids(self) -> tuple[int, ...]:
        return tuple(range(self.count))


def _scale(unit_values: np.ndarray, support: UniformSupport) -> np.ndarray:
    return support.lower + (support.upper - support.lower) * unit_values


def _unit_design(count: int, dimensions: int, seed: int, method: str) -> np.ndarray:
    if count <= 0 or dimensions <= 0:
        raise ValueError("count and dimensions must be positive")
    rng = np.random.default_rng(seed)
    if method == "mc":
        return rng.random((count, dimensions))
    if method != "lhs":
        raise ValueError(f"Unsupported sampling method: {method}")
    result = np.empty((count, dimensions), dtype=float)
    for column in range(dimensions):
        strata = rng.permutation(count)
        result[:, column] = (strata + rng.random(count)) / count
    return result


def _dimension_specification(data: Q1Data) -> tuple[tuple[InnovationKey, UniformSupport], ...]:
    config = DEFAULT_Q2_CONFIG
    wheat_corn_ids = {
        crop.crop_id for crop in data.crops.values() if crop.name in {"小麦", "玉米"}
    }
    dimensions: list[tuple[InnovationKey, UniformSupport]] = []
    for crop_id in sorted(data.crops):
        for year in YEARS:
            family = "demand_growth" if crop_id in wheat_corn_ids else "demand_delta"
            support = (
                config.wheat_corn_growth
                if crop_id in wheat_corn_ids
                else config.other_demand_delta
            )
            dimensions.append(((family, crop_id, year), support))
    for crop_id in sorted(data.crops):
        for year in YEARS:
            dimensions.append((("yield_delta", crop_id, year), config.yield_delta))
    for crop_id in (38, 39, 40):
        for year in YEARS:
            dimensions.append((("mushroom_decline", crop_id, year), config.mushroom_decline))
    return tuple(dimensions)


def generate_scenarios(
    data: Q1Data,
    *,
    count: int,
    seed: int,
    sampling_method: str,
    config: Q2Config = DEFAULT_Q2_CONFIG,
) -> Q2ScenarioPaths:
    """Generate crop-year innovations and fully materialized parameter multipliers."""

    dimensions = _dimension_specification(data)
    unit = _unit_design(count, len(dimensions), seed, sampling_method)
    innovations: dict[InnovationKey, np.ndarray] = {}
    for column, (key, support) in enumerate(dimensions):
        innovations[key] = _scale(unit[:, column], support)

    probabilities = np.full(count, 1.0 / count, dtype=float)
    sales_crop_seasons = sorted(
        {
            (crop_id, season)
            for plot, crop_id, _year, season in data.compatible_cells
        }
    )
    wheat_corn_ids = {
        crop.crop_id for crop in data.crops.values() if crop.name in {"小麦", "玉米"}
    }

    demand: dict[DemandKey, float] = {}
    yield_multiplier: dict[MultiplierKey, float] = {}
    price_multiplier: dict[MultiplierKey, float] = {}
    for omega in range(count):
        for crop_id in sorted(data.crops):
            crop = data.crops[crop_id]
            recursive_demand_level = 1.0
            recursive_mushroom_price = 1.0
            for year in YEARS:
                if crop_id in wheat_corn_ids:
                    recursive_demand_level *= 1.0 + innovations[
                        ("demand_growth", crop_id, year)
                    ][omega]
                    demand_multiplier = recursive_demand_level
                else:
                    demand_multiplier = 1.0 + innovations[
                        ("demand_delta", crop_id, year)
                    ][omega]

                for demand_crop_id, season in sales_crop_seasons:
                    if demand_crop_id != crop_id:
                        continue
                    baseline = data.expected_sales_proxy.get((crop_id, season), 0.0)
                    demand[(omega, crop_id, year, season)] = baseline * demand_multiplier

                yield_multiplier[(omega, crop_id, year)] = 1.0 + innovations[
                    ("yield_delta", crop_id, year)
                ][omega]

                if crop.is_grain:
                    price_multiplier[(omega, crop_id, year)] = 1.0
                elif crop.is_vegetable:
                    price_multiplier[(omega, crop_id, year)] = (
                        1.0 + config.vegetable_price_growth
                    ) ** (year - 2023)
                elif crop_id in {38, 39, 40}:
                    recursive_mushroom_price *= 1.0 - innovations[
                        ("mushroom_decline", crop_id, year)
                    ][omega]
                    price_multiplier[(omega, crop_id, year)] = recursive_mushroom_price
                elif crop_id == 41:
                    price_multiplier[(omega, crop_id, year)] = (
                        1.0 - config.morel_price_decline
                    ) ** (year - 2023)
                else:
                    raise ValueError(f"Unsupported crop price class for crop {crop_id}")

    return Q2ScenarioPaths(
        count=count,
        seed=seed,
        sampling_method=sampling_method,
        probabilities=probabilities,
        dimension_order=tuple(key for key, _support in dimensions),
        unit_draws=unit,
        innovations=innovations,
        demand_jin=demand,
        yield_multiplier=yield_multiplier,
        price_multiplier=price_multiplier,
    )


def cost_multiplier(year: int, config: Q2Config = DEFAULT_Q2_CONFIG) -> float:
    if year not in YEARS:
        raise ValueError(f"Year outside Q2 horizon: {year}")
    return (1.0 + config.cost_growth) ** (year - 2023)


def lower_tail_cvar(
    profits: np.ndarray | list[float] | tuple[float, ...],
    beta: float,
    probabilities: np.ndarray | None = None,
) -> float:
    """Return the average of the worst (1-beta) probability mass of profits."""

    values = np.asarray(profits, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("profits must be a non-empty finite one-dimensional vector")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie strictly between zero and one")
    weights = (
        np.full(values.size, 1.0 / values.size, dtype=float)
        if probabilities is None
        else np.asarray(probabilities, dtype=float)
    )
    if weights.shape != values.shape or np.any(weights < 0.0):
        raise ValueError("probabilities must be non-negative and match profits")
    total_weight = math.fsum(float(value) for value in weights)
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"probabilities must sum to one, found {total_weight}")

    tail_mass = 1.0 - beta
    remaining = tail_mass
    weighted_sum = 0.0
    for index in np.argsort(values, kind="stable"):
        take = min(remaining, float(weights[index]))
        weighted_sum += take * float(values[index])
        remaining -= take
        if remaining <= 1e-15:
            break
    if remaining > 1e-12:
        raise ValueError("probability mass was insufficient for the requested tail")
    return weighted_sum / tail_mass

