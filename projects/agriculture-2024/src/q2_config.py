"""Centralized Human-Gate-approved Q2 experimental configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class UniformSupport:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(f"Invalid support [{self.lower}, {self.upper}]")


@dataclass(frozen=True)
class Q2Config:
    beta: float = 0.90
    surplus_discount: float = 0.50
    smoke_scenarios: int = 30
    main_scenarios: int = 60
    stability_scenarios: int = 90
    evaluation_paths: int = 3000
    optimization_seed: int = 20240818
    evaluation_seed: int = 20240819
    frontier_theta: tuple[float, ...] = (0.25, 0.50, 0.75)
    wheat_corn_growth: UniformSupport = UniformSupport(0.05, 0.10)
    other_demand_delta: UniformSupport = UniformSupport(-0.05, 0.05)
    yield_delta: UniformSupport = UniformSupport(-0.10, 0.10)
    mushroom_decline: UniformSupport = UniformSupport(0.01, 0.05)
    cost_growth: float = 0.05
    vegetable_price_growth: float = 0.05
    grain_price_growth: float = 0.0
    morel_price_decline: float = 0.05
    optimization_sampling: str = "lhs"
    evaluation_sampling: str = "mc"
    scenario_weights: str = "equal"
    shock_granularity: str = "crop_year"
    innovation_dependence: str = "independent_across_crops_families_years"

    def __post_init__(self) -> None:
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must lie strictly between zero and one")
        if not 0.0 <= self.surplus_discount <= 1.0:
            raise ValueError("surplus_discount must lie in [0, 1]")
        if min(
            self.smoke_scenarios,
            self.main_scenarios,
            self.stability_scenarios,
            self.evaluation_paths,
        ) <= 0:
            raise ValueError("all scenario counts must be positive")
        if self.optimization_seed == self.evaluation_seed:
            raise ValueError("optimization and evaluation seeds must differ")
        if any(not 0.0 < theta < 1.0 for theta in self.frontier_theta):
            raise ValueError("frontier theta values must lie in (0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_Q2_CONFIG = Q2Config()

