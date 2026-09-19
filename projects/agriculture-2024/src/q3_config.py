"""Isolated, explicit E-class configuration for Q3 dependence modeling."""

from __future__ import annotations

from dataclasses import dataclass, replace

from q2_config import UniformSupport


CROP_GROUPS: dict[str, tuple[int, ...]] = {
    "dry_beans": (1, 2, 3, 4, 5),
    "staple_grains": (6, 7, 8, 9, 10, 11, 14, 15, 16),
    "fresh_beans": (17, 18, 19),
    "tubers": (13, 20),
    "fruit_vegetables": (12, 21, 22, 24, 29, 31),
    "leafy_brassica": (23, 25, 26, 27, 28, 30, 32, 33, 34, 35),
    "root_vegetables": (36, 37),
    "mushrooms": (38, 39, 40, 41),
}
CROP_TO_GROUP = {
    crop_id: group for group, crop_ids in CROP_GROUPS.items() for crop_id in crop_ids
}
NONSTARCHY_VEGETABLE_GROUPS = (
    "fruit_vegetables",
    "leafy_brassica",
    "root_vegetables",
)


@dataclass(frozen=True)
class Q3Config:
    beta: float = 0.90
    surplus_discount: float = 0.50
    smoke_scenarios: int = 30
    main_scenarios: int = 60
    evaluation_paths: int = 3000
    optimization_seed: int = 20240820
    evaluation_seed: int = 20240821
    yield_weather_loading: float = 0.50
    demand_market_loading: float = 0.45
    price_market_loading: float = 0.35
    price_input_loading: float = 0.25
    cost_input_loading: float = 0.60
    own_price_elasticity: float = -0.20
    substitute_elasticity: float = 0.05
    complement_elasticity: float = -0.02
    elasticity_scale: float = 1.0
    demand_interaction_lower: float = 0.90
    demand_interaction_upper: float = 1.10
    grain_price_growth: UniformSupport = UniformSupport(-0.01, 0.01)
    vegetable_price_growth: UniformSupport = UniformSupport(0.03, 0.07)
    mushroom_decline: UniformSupport = UniformSupport(0.01, 0.05)
    morel_decline: UniformSupport = UniformSupport(0.03, 0.07)
    cost_growth: UniformSupport = UniformSupport(0.03, 0.07)
    wheat_corn_growth: UniformSupport = UniformSupport(0.05, 0.10)
    other_demand_delta: UniformSupport = UniformSupport(-0.05, 0.05)
    yield_delta: UniformSupport = UniformSupport(-0.10, 0.10)

    def __post_init__(self) -> None:
        if self.optimization_seed == self.evaluation_seed:
            raise ValueError("Q3 optimization and evaluation seeds must differ")
        if set(CROP_TO_GROUP) != set(range(1, 42)):
            raise ValueError("Q3 crop groups must partition crop ids 1..41")
        loadings = (
            self.yield_weather_loading,
            self.demand_market_loading,
            self.price_market_loading,
            self.price_input_loading,
            self.cost_input_loading,
        )
        if any(not 0.0 <= value < 1.0 for value in loadings):
            raise ValueError("all factor loadings must lie in [0, 1)")
        if self.price_market_loading**2 + self.price_input_loading**2 >= 1.0:
            raise ValueError("price loading squares must sum below one")
        if not 0.0 < self.demand_interaction_lower <= 1.0:
            raise ValueError("invalid lower interaction bound")
        if self.demand_interaction_upper < 1.0:
            raise ValueError("invalid upper interaction bound")
        if self.elasticity_scale < 0.0:
            raise ValueError("elasticity scale must be non-negative")

    def with_elasticity_scale(self, scale: float) -> "Q3Config":
        return replace(self, elasticity_scale=float(scale))


BASELINE_Q3_CONFIG = Q3Config()
WEAK_Q3_CONFIG = replace(
    BASELINE_Q3_CONFIG,
    yield_weather_loading=0.25,
    demand_market_loading=0.20,
    price_market_loading=0.15,
    price_input_loading=0.10,
    cost_input_loading=0.30,
)
STRONG_Q3_CONFIG = replace(
    BASELINE_Q3_CONFIG,
    yield_weather_loading=0.70,
    demand_market_loading=0.65,
    price_market_loading=0.50,
    price_input_loading=0.35,
    cost_input_loading=0.75,
)

