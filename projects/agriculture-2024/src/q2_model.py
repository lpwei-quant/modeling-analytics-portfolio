"""Sparse finite-scenario MILP for CUMCM 2024 Problem C, Question 2."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, milp

from q1_data import Q1Data, YEARS
from q1_model import (
    NUMERICAL_ZERO,
    PlantKey,
    RegimeKey,
    SalesKey,
    SparseConstraintBuilder,
    VariableRegistry,
    add_activation_constraints,
    add_bean_window_constraints,
    add_capacity_constraints,
    add_initial_rotation_constraints,
    add_irrigated_regime_constraints,
    add_irrigated_second_season_choice,
    add_rotation_constraints,
    sales_groups,
)
from q2_config import DEFAULT_Q2_CONFIG, Q2Config
from q2_uncertainty import Q2ScenarioPaths, cost_multiplier, lower_tail_cvar


ScenarioSalesKey = tuple[int, SalesKey]


class Q2ModelError(RuntimeError):
    """Raised when a Q2 model violates its contract or fails to solve."""


@dataclass(frozen=True)
class Q2VariableBlock:
    x: dict[PlantKey, int]
    z: dict[PlantKey, int]
    regime: dict[RegimeKey, int]
    normal_sales: dict[ScenarioSalesKey, int]
    scenario_profit: dict[int, int]
    tau: int | None
    shortfall: dict[int, int]
    names: tuple[str, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    integrality: np.ndarray
    objective: np.ndarray


@dataclass(frozen=True)
class Q2BuiltModel:
    data: Q1Data
    scenarios: Q2ScenarioPaths
    config: Q2Config
    objective_kind: str
    cvar_floor_yuan: float | None
    expected_profit_floor_yuan: float | None
    variables: Q2VariableBlock
    constraints: object


@dataclass(frozen=True)
class Q2Solution:
    model: Q2BuiltModel
    result: OptimizeResult
    values: np.ndarray
    solve_seconds: float

    @property
    def x(self) -> dict[PlantKey, float]:
        return {
            key: 0.0 if abs(self.values[index]) <= NUMERICAL_ZERO else float(self.values[index])
            for key, index in self.model.variables.x.items()
        }

    @property
    def z(self) -> dict[PlantKey, float]:
        return {
            key: float(self.values[index])
            for key, index in self.model.variables.z.items()
        }

    @property
    def regimes(self) -> dict[RegimeKey, float]:
        return {
            key: float(self.values[index])
            for key, index in self.model.variables.regime.items()
        }

    @property
    def normal_sales(self) -> dict[ScenarioSalesKey, float]:
        return {
            key: 0.0 if abs(self.values[index]) <= NUMERICAL_ZERO else float(self.values[index])
            for key, index in self.model.variables.normal_sales.items()
        }

    @property
    def scenario_profits(self) -> np.ndarray:
        return np.asarray(
            [
                self.values[self.model.variables.scenario_profit[omega]]
                for omega in self.model.scenarios.scenario_ids
            ],
            dtype=float,
        )

    @property
    def expected_profit_yuan(self) -> float:
        return float(np.dot(self.model.scenarios.probabilities, self.scenario_profits))

    @property
    def empirical_cvar_yuan(self) -> float:
        return lower_tail_cvar(
            self.scenario_profits,
            beta=self.model.config.beta,
            probabilities=self.model.scenarios.probabilities,
        )

    @property
    def auxiliary_cvar_yuan(self) -> float | None:
        variables = self.model.variables
        if variables.tau is None:
            return None
        tau = float(self.values[variables.tau])
        penalty = math.fsum(
            float(self.model.scenarios.probabilities[omega])
            * float(self.values[variables.shortfall[omega]])
            for omega in self.model.scenarios.scenario_ids
        ) / (1.0 - self.model.config.beta)
        return tau - penalty


def _build_variables(
    data: Q1Data,
    scenarios: Q2ScenarioPaths,
    objective_kind: str,
    config: Q2Config,
) -> Q2VariableBlock:
    valid_objectives = {"risk_neutral", "max_cvar", "expected_with_cvar_floor"}
    if objective_kind not in valid_objectives:
        raise Q2ModelError(f"Unsupported objective kind: {objective_kind}")

    registry = VariableRegistry()
    x: dict[PlantKey, int] = {}
    z: dict[PlantKey, int] = {}
    regime: dict[RegimeKey, int] = {}
    normal_sales: dict[ScenarioSalesKey, int] = {}
    scenario_profit: dict[int, int] = {}
    shortfall: dict[int, int] = {}

    for key in data.compatible_cells:
        plot, crop_id, year, season = key
        area = data.plots[plot].area_mu
        x[key] = registry.add(f"x[{plot},{crop_id},{year},{season}]", upper=area)
        z[key] = registry.add(
            f"z[{plot},{crop_id},{year},{season}]", upper=1.0, integer=True
        )
    for plot in data.irrigated_plots:
        for year in YEARS:
            regime[(plot, year)] = registry.add(
                f"u_rice[{plot},{year}]", upper=1.0, integer=True
            )

    groups = sales_groups(data)
    for omega in scenarios.scenario_ids:
        for key in groups:
            crop_id, land_type, year, season = key
            normal_sales[(omega, key)] = registry.add(
                f"normal[{omega},{crop_id},{land_type},{year},{season}]"
            )
        scenario_profit[omega] = registry.add(
            f"profit[{omega}]",
            lower=-math.inf,
            upper=math.inf,
            objective=(
                -float(scenarios.probabilities[omega])
                if objective_kind in {"risk_neutral", "expected_with_cvar_floor"}
                else 0.0
            ),
        )

    tau: int | None = None
    if objective_kind in {"max_cvar", "expected_with_cvar_floor"}:
        tau = registry.add(
            "cvar_tau",
            lower=-math.inf,
            upper=math.inf,
            objective=-1.0 if objective_kind == "max_cvar" else 0.0,
        )
        for omega in scenarios.scenario_ids:
            shortfall[omega] = registry.add(
                f"cvar_shortfall[{omega}]",
                objective=(
                    float(scenarios.probabilities[omega])
                    / (1.0 - config.beta)
                    if objective_kind == "max_cvar"
                    else 0.0
                ),
            )

    return Q2VariableBlock(
        x=x,
        z=z,
        regime=regime,
        normal_sales=normal_sales,
        scenario_profit=scenario_profit,
        tau=tau,
        shortfall=shortfall,
        names=tuple(registry.names),
        lower_bounds=np.asarray(registry.lower, dtype=float),
        upper_bounds=np.asarray(registry.upper, dtype=float),
        integrality=np.asarray(registry.integrality, dtype=np.uint8),
        objective=np.asarray(registry.objective, dtype=float),
    )


def _add_agricultural_constraints(
    builder: SparseConstraintBuilder, data: Q1Data, variables: Q2VariableBlock
) -> None:
    add_capacity_constraints(builder, data, variables)
    add_activation_constraints(builder, data, variables)
    add_irrigated_regime_constraints(builder, data, variables)
    add_irrigated_second_season_choice(builder, data, variables)
    add_rotation_constraints(builder, data, variables)
    add_initial_rotation_constraints(builder, data, variables)
    add_bean_window_constraints(builder, data, variables)


def _add_scenario_economic_constraints(
    builder: SparseConstraintBuilder,
    data: Q1Data,
    scenarios: Q2ScenarioPaths,
    variables: Q2VariableBlock,
    config: Q2Config,
) -> None:
    groups = sales_groups(data)
    x_by_group: dict[SalesKey, list[tuple[PlantKey, int]]] = {key: [] for key in groups}
    for key, index in variables.x.items():
        plot, crop_id, year, season = key
        group = (crop_id, data.plots[plot].land_type, year, season)
        x_by_group[group].append((key, index))

    demand_groups: dict[tuple[int, int, str], list[SalesKey]] = {}
    for group in groups:
        crop_id, _land_type, year, season = group
        demand_groups.setdefault((crop_id, year, season), []).append(group)

    for omega in scenarios.scenario_ids:
        profit_coefficients: list[tuple[int, float]] = [
            (variables.scenario_profit[omega], 1.0)
        ]
        for group in groups:
            crop_id, land_type, year, season = group
            parameter = data.parameters[(crop_id, land_type, season)]
            yield_value = parameter.yield_jin_per_mu * scenarios.yield_multiplier[
                (omega, crop_id, year)
            ]
            price = parameter.midpoint_price_yuan_per_jin * scenarios.price_multiplier[
                (omega, crop_id, year)
            ]
            normal_index = variables.normal_sales[(omega, group)]
            production_coefficients = [
                (index, -yield_value) for _key, index in x_by_group[group]
            ]
            builder.add(
                [(normal_index, 1.0)] + production_coefficients,
                upper=0.0,
                category="scenario_normal_le_production",
                name=(
                    f"normal_le_production[{omega},{crop_id},{land_type},{year},{season}]"
                ),
            )
            profit_coefficients.append(
                (normal_index, -(1.0 - config.surplus_discount) * price)
            )
            for plant_key, index in x_by_group[group]:
                _plot, _crop, _year, _season = plant_key
                cost = parameter.cost_yuan_per_mu * cost_multiplier(year, config)
                profit_coefficients.append(
                    (index, cost - config.surplus_discount * price * yield_value)
                )

        for (crop_id, year, season), demand_sales_groups in demand_groups.items():
            demand = scenarios.demand_jin[(omega, crop_id, year, season)]
            builder.add(
                (
                    (variables.normal_sales[(omega, group)], 1.0)
                    for group in demand_sales_groups
                ),
                upper=demand,
                category="scenario_demand",
                name=f"demand[{omega},{crop_id},{year},{season}]",
            )

        builder.add(
            profit_coefficients,
            lower=0.0,
            upper=0.0,
            category="scenario_profit",
            name=f"profit_definition[{omega}]",
        )


def _add_cvar_constraints(
    builder: SparseConstraintBuilder,
    scenarios: Q2ScenarioPaths,
    variables: Q2VariableBlock,
    config: Q2Config,
    cvar_floor_yuan: float | None,
) -> None:
    if variables.tau is None:
        if cvar_floor_yuan is not None:
            raise Q2ModelError("CVaR floor requires CVaR auxiliary variables")
        return
    for omega in scenarios.scenario_ids:
        builder.add(
            {
                variables.tau: 1.0,
                variables.scenario_profit[omega]: -1.0,
                variables.shortfall[omega]: -1.0,
            },
            upper=0.0,
            category="cvar_shortfall",
            name=f"cvar_shortfall[{omega}]",
        )
    if cvar_floor_yuan is not None:
        coefficients: list[tuple[int, float]] = [(variables.tau, 1.0)]
        coefficients.extend(
            (
                variables.shortfall[omega],
                -float(scenarios.probabilities[omega]) / (1.0 - config.beta),
            )
            for omega in scenarios.scenario_ids
        )
        builder.add(
            coefficients,
            lower=float(cvar_floor_yuan),
            category="cvar_floor",
            name=f"cvar_floor[{cvar_floor_yuan:.12g}]",
        )


def _add_expected_profit_floor(
    builder: SparseConstraintBuilder,
    scenarios: Q2ScenarioPaths,
    variables: Q2VariableBlock,
    expected_profit_floor_yuan: float | None,
) -> None:
    if expected_profit_floor_yuan is None:
        return
    builder.add(
        (
            (variables.scenario_profit[omega], float(scenarios.probabilities[omega]))
            for omega in scenarios.scenario_ids
        ),
        lower=float(expected_profit_floor_yuan),
        category="expected_profit_floor",
        name=f"expected_profit_floor[{expected_profit_floor_yuan:.12g}]",
    )


def build_q2_model(
    data: Q1Data,
    scenarios: Q2ScenarioPaths,
    *,
    objective_kind: str,
    cvar_floor_yuan: float | None = None,
    expected_profit_floor_yuan: float | None = None,
    config: Q2Config = DEFAULT_Q2_CONFIG,
) -> Q2BuiltModel:
    if not math.isclose(
        math.fsum(float(value) for value in scenarios.probabilities),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise Q2ModelError("Scenario probabilities must sum to one")
    if objective_kind == "expected_with_cvar_floor" and cvar_floor_yuan is None:
        raise Q2ModelError("Expected-profit frontier model requires a CVaR floor")
    if objective_kind == "risk_neutral" and cvar_floor_yuan is not None:
        raise Q2ModelError("CVaR floor requires a model with CVaR auxiliaries")
    if cvar_floor_yuan is not None and not math.isfinite(cvar_floor_yuan):
        raise Q2ModelError("CVaR floor must be finite")
    if expected_profit_floor_yuan is not None and not math.isfinite(
        expected_profit_floor_yuan
    ):
        raise Q2ModelError("Expected-profit floor must be finite")

    variables = _build_variables(data, scenarios, objective_kind, config)
    builder = SparseConstraintBuilder(len(variables.names))
    _add_agricultural_constraints(builder, data, variables)
    _add_scenario_economic_constraints(builder, data, scenarios, variables, config)
    _add_cvar_constraints(builder, scenarios, variables, config, cvar_floor_yuan)
    _add_expected_profit_floor(
        builder, scenarios, variables, expected_profit_floor_yuan
    )
    constraints = builder.build()
    return Q2BuiltModel(
        data=data,
        scenarios=scenarios,
        config=config,
        objective_kind=objective_kind,
        cvar_floor_yuan=cvar_floor_yuan,
        expected_profit_floor_yuan=expected_profit_floor_yuan,
        variables=variables,
        constraints=constraints,
    )


def build_lexicographic_stage2_model(
    data: Q1Data,
    scenarios: Q2ScenarioPaths,
    *,
    incumbent_cvar_yuan: float,
    expected_profit_floor_yuan: float | None = None,
    config: Q2Config = DEFAULT_Q2_CONFIG,
) -> Q2BuiltModel:
    """Build the approved second-stage endpoint without an arbitrary lambda.

    Stage 1 supplies the feasible incumbent lower-tail profit CVaR. Stage 2
    preserves at least that achieved level and maximizes expected profit.
    """

    if not math.isfinite(incumbent_cvar_yuan):
        raise Q2ModelError("Stage-1 incumbent CVaR must be finite")
    return build_q2_model(
        data,
        scenarios,
        objective_kind="expected_with_cvar_floor",
        cvar_floor_yuan=float(incumbent_cvar_yuan),
        expected_profit_floor_yuan=expected_profit_floor_yuan,
        config=config,
    )


def build_q2_lp_relaxation(model: Q2BuiltModel) -> Q2BuiltModel:
    """Return the exact model with only integer restrictions relaxed.

    Every variable retains its original finite/infinite bounds.  Objective
    coefficients, variable indices, scenarios, and the full sparse constraint
    system are unchanged.  Copies prevent the relaxation from mutating the
    source MILP's arrays.
    """

    if model.objective_kind != "max_cvar":
        raise Q2ModelError("LP upper-bound relaxation requires a max_cvar model")
    variables = replace(
        model.variables,
        lower_bounds=model.variables.lower_bounds.copy(),
        upper_bounds=model.variables.upper_bounds.copy(),
        integrality=np.zeros_like(model.variables.integrality),
        objective=model.variables.objective.copy(),
    )
    return replace(model, variables=variables)


def solve_q2_model(
    model: Q2BuiltModel,
    *,
    time_limit_seconds: float,
    mip_relative_gap: float = 0.005,
    display_solver_log: bool = False,
) -> Q2Solution:
    import time

    started = time.perf_counter()
    result = milp(
        c=model.variables.objective,
        integrality=model.variables.integrality,
        bounds=Bounds(model.variables.lower_bounds, model.variables.upper_bounds),
        constraints=model.constraints.linear_constraint,
        options={
            "disp": display_solver_log,
            "presolve": True,
            "time_limit": float(time_limit_seconds),
            "mip_rel_gap": float(mip_relative_gap),
        },
    )
    solve_seconds = time.perf_counter() - started
    if result.status != 0 or result.x is None or result.fun is None:
        raise Q2ModelError(
            f"{model.objective_kind} failed: status={result.status}, "
            f"success={result.success}, message={result.message}"
        )
    values = np.asarray(result.x, dtype=float)
    if not np.all(np.isfinite(values)):
        raise Q2ModelError("Solver returned non-finite variable values")
    return Q2Solution(model=model, result=result, values=values, solve_seconds=solve_seconds)
