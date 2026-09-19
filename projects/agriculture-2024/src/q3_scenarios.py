"""Bounded Q3 correlated scenarios with sparse price-demand interactions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from q1_data import Q1Data, YEARS
from q2_config import UniformSupport
from q3_config import (
    BASELINE_Q3_CONFIG,
    CROP_GROUPS,
    CROP_TO_GROUP,
    NONSTARCHY_VEGETABLE_GROUPS,
    Q3Config,
)


InnovationKey = tuple[str, int, int]
DemandKey = tuple[int, int, int, str]
MultiplierKey = tuple[int, int, int]
InteractionKey = tuple[int, int, int]


@dataclass(frozen=True)
class Q3ScenarioPaths:
    count: int
    seed: int
    sampling_method: str
    probabilities: np.ndarray
    dimension_order: tuple[InnovationKey, ...]
    unit_draws: np.ndarray
    innovations: dict[InnovationKey, np.ndarray]
    supports: dict[InnovationKey, UniformSupport]
    factor_draws: dict[tuple[str, str | int, int], np.ndarray]
    base_demand_jin: dict[DemandKey, float]
    demand_jin: dict[DemandKey, float]
    demand_interaction_multiplier: dict[InteractionKey, float]
    yield_multiplier: dict[MultiplierKey, float]
    price_multiplier: dict[MultiplierKey, float]
    cost_multiplier: dict[MultiplierKey, float]
    config: Q3Config

    @property
    def scenario_ids(self) -> tuple[int, ...]:
        return tuple(range(self.count))


def _scale(values: np.ndarray, support: UniformSupport) -> np.ndarray:
    return support.lower + (support.upper - support.lower) * values


def _lhs_marginal_from_latent(
    latent: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    count = latent.size
    marginal = (rng.permutation(count) + rng.random(count)) / count
    ordered_values = np.sort(marginal)
    output = np.empty(count, dtype=float)
    output[np.argsort(latent, kind="stable")] = ordered_values
    return output


def _unit_from_latent(
    latent: np.ndarray, method: str, rng: np.random.Generator
) -> np.ndarray:
    if method == "lhs":
        return _lhs_marginal_from_latent(latent, rng)
    if method != "mc":
        raise ValueError(f"Unsupported Q3 sampling method: {method}")
    return np.clip(norm.cdf(latent), np.finfo(float).eps, 1.0 - np.finfo(float).eps)


def _support_for(data: Q1Data, family: str, crop_id: int, config: Q3Config) -> UniformSupport:
    if family == "demand_growth":
        return config.wheat_corn_growth
    if family == "demand_delta":
        return config.other_demand_delta
    if family == "yield_delta":
        return config.yield_delta
    if family == "cost_growth":
        return config.cost_growth
    if family == "price_change":
        crop = data.crops[crop_id]
        if crop.is_grain:
            return config.grain_price_growth
        if crop.is_vegetable:
            return config.vegetable_price_growth
        if crop_id in {38, 39, 40}:
            return config.mushroom_decline
        if crop_id == 41:
            return config.morel_decline
    raise ValueError(f"Unsupported Q3 support family={family}, crop={crop_id}")


def _q2_reference_price_multiplier(data: Q1Data, crop_id: int, year: int) -> float:
    periods = year - 2023
    crop = data.crops[crop_id]
    if crop.is_grain:
        return 1.0
    if crop.is_vegetable:
        return 1.05**periods
    if crop_id in {38, 39, 40}:
        return 0.97**periods
    if crop_id == 41:
        return 0.95**periods
    raise ValueError(crop_id)


def _complement_crop_ids(crop_id: int) -> tuple[int, ...]:
    group = CROP_TO_GROUP[crop_id]
    if group == "staple_grains":
        return tuple(
            crop for name in NONSTARCHY_VEGETABLE_GROUPS for crop in CROP_GROUPS[name]
        )
    if group in NONSTARCHY_VEGETABLE_GROUPS:
        return CROP_GROUPS["staple_grains"]
    return ()


def generate_q3_scenarios(
    data: Q1Data,
    *,
    count: int,
    seed: int,
    sampling_method: str,
    config: Q3Config = BASELINE_Q3_CONFIG,
) -> Q3ScenarioPaths:
    if count <= 0:
        raise ValueError("Q3 scenario count must be positive")
    rng = np.random.default_rng(seed)
    factor_draws: dict[tuple[str, str | int, int], np.ndarray] = {}
    for year in YEARS:
        factor_draws[("weather", 0, year)] = rng.standard_normal(count)
        factor_draws[("input", 0, year)] = rng.standard_normal(count)
        for group in CROP_GROUPS:
            factor_draws[("market", group, year)] = rng.standard_normal(count)

    dimension_order: list[InnovationKey] = []
    units: list[np.ndarray] = []
    innovations: dict[InnovationKey, np.ndarray] = {}
    supports: dict[InnovationKey, UniformSupport] = {}
    for crop_id in sorted(data.crops):
        group = CROP_TO_GROUP[crop_id]
        for year in YEARS:
            idiosyncratic = {
                family: rng.standard_normal(count)
                for family in ("demand", "yield", "price", "cost")
            }
            demand_family = "demand_growth" if crop_id in {6, 7} else "demand_delta"
            latent_by_family = {
                demand_family: (
                    config.demand_market_loading * factor_draws[("market", group, year)]
                    + math.sqrt(1.0 - config.demand_market_loading**2)
                    * idiosyncratic["demand"]
                ),
                "yield_delta": (
                    config.yield_weather_loading * factor_draws[("weather", 0, year)]
                    + math.sqrt(1.0 - config.yield_weather_loading**2)
                    * idiosyncratic["yield"]
                ),
                "price_change": (
                    config.price_market_loading * factor_draws[("market", group, year)]
                    + config.price_input_loading * factor_draws[("input", 0, year)]
                    + math.sqrt(
                        1.0
                        - config.price_market_loading**2
                        - config.price_input_loading**2
                    )
                    * idiosyncratic["price"]
                ),
                "cost_growth": (
                    config.cost_input_loading * factor_draws[("input", 0, year)]
                    + math.sqrt(1.0 - config.cost_input_loading**2)
                    * idiosyncratic["cost"]
                ),
            }
            for family, latent in latent_by_family.items():
                key = (family, crop_id, year)
                unit = _unit_from_latent(latent, sampling_method, rng)
                support = _support_for(data, family, crop_id, config)
                dimension_order.append(key)
                units.append(unit)
                supports[key] = support
                innovations[key] = _scale(unit, support)

    unit_draws = np.column_stack(units)
    probabilities = np.full(count, 1.0 / count, dtype=float)
    yield_multiplier: dict[MultiplierKey, float] = {}
    price_multiplier: dict[MultiplierKey, float] = {}
    cost_multiplier: dict[MultiplierKey, float] = {}
    for omega in range(count):
        for crop_id in sorted(data.crops):
            price_level = 1.0
            cost_level = 1.0
            for year in YEARS:
                yield_multiplier[(omega, crop_id, year)] = 1.0 + innovations[
                    ("yield_delta", crop_id, year)
                ][omega]
                price_change = innovations[("price_change", crop_id, year)][omega]
                if crop_id >= 38:
                    price_level *= 1.0 - price_change
                else:
                    price_level *= 1.0 + price_change
                price_multiplier[(omega, crop_id, year)] = price_level
                cost_level *= 1.0 + innovations[("cost_growth", crop_id, year)][omega]
                cost_multiplier[(omega, crop_id, year)] = cost_level

    sales_crop_seasons = sorted(
        {(crop_id, season) for _plot, crop_id, _year, season in data.compatible_cells}
    )
    base_demand: dict[DemandKey, float] = {}
    demand: dict[DemandKey, float] = {}
    interaction_multiplier: dict[InteractionKey, float] = {}
    lower_log = math.log(config.demand_interaction_lower)
    upper_log = math.log(config.demand_interaction_upper)
    for omega in range(count):
        wheat_corn_level = {6: 1.0, 7: 1.0}
        for year in YEARS:
            relative_price = {
                crop_id: price_multiplier[(omega, crop_id, year)]
                / _q2_reference_price_multiplier(data, crop_id, year)
                for crop_id in data.crops
            }
            for crop_id in sorted(data.crops):
                if crop_id in {6, 7}:
                    wheat_corn_level[crop_id] *= 1.0 + innovations[
                        ("demand_growth", crop_id, year)
                    ][omega]
                    base_multiplier = wheat_corn_level[crop_id]
                else:
                    base_multiplier = 1.0 + innovations[
                        ("demand_delta", crop_id, year)
                    ][omega]

                group_peers = tuple(
                    peer for peer in CROP_GROUPS[CROP_TO_GROUP[crop_id]] if peer != crop_id
                )
                peer_log = (
                    math.fsum(math.log(relative_price[peer]) for peer in group_peers)
                    / len(group_peers)
                    if group_peers
                    else 0.0
                )
                complement_ids = _complement_crop_ids(crop_id)
                complement_log = (
                    math.fsum(math.log(relative_price[peer]) for peer in complement_ids)
                    / len(complement_ids)
                    if complement_ids
                    else 0.0
                )
                scale = config.elasticity_scale
                log_adjustment = scale * (
                    config.own_price_elasticity * math.log(relative_price[crop_id])
                    + config.substitute_elasticity * peer_log
                    + config.complement_elasticity * complement_log
                )
                adjustment = math.exp(min(upper_log, max(lower_log, log_adjustment)))
                interaction_multiplier[(omega, crop_id, year)] = adjustment
                for demand_crop_id, season in sales_crop_seasons:
                    if demand_crop_id != crop_id:
                        continue
                    baseline = data.expected_sales_proxy.get((crop_id, season), 0.0)
                    key = (omega, crop_id, year, season)
                    base_demand[key] = baseline * base_multiplier
                    demand[key] = base_demand[key] * adjustment

    return Q3ScenarioPaths(
        count=count,
        seed=seed,
        sampling_method=sampling_method,
        probabilities=probabilities,
        dimension_order=tuple(dimension_order),
        unit_draws=unit_draws,
        innovations=innovations,
        supports=supports,
        factor_draws=factor_draws,
        base_demand_jin=base_demand,
        demand_jin=demand,
        demand_interaction_multiplier=interaction_multiplier,
        yield_multiplier=yield_multiplier,
        price_multiplier=price_multiplier,
        cost_multiplier=cost_multiplier,
        config=config,
    )

