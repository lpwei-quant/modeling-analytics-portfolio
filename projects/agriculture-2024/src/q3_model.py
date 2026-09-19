"""Q3 stochastic model with scenario-specific dependent economics."""

from __future__ import annotations

import math

from q1_data import Q1Data
from q1_model import PlantKey, SalesKey, SparseConstraintBuilder, sales_groups
from q2_config import Q2Config
from q2_model import (
    Q2BuiltModel,
    Q2ModelError,
    Q2Solution,
    _add_agricultural_constraints,
    _add_cvar_constraints,
    _add_expected_profit_floor,
    _build_variables,
    build_q2_lp_relaxation,
    solve_q2_model,
)
from q3_config import BASELINE_Q3_CONFIG, Q3Config
from q3_scenarios import Q3ScenarioPaths


def _objective_config(config: Q3Config) -> Q2Config:
    return Q2Config(beta=config.beta, surplus_discount=config.surplus_discount)


def _add_q3_economic_constraints(
    builder: SparseConstraintBuilder,
    data: Q1Data,
    scenarios: Q3ScenarioPaths,
    variables,
    config: Q3Config,
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
            builder.add(
                [(normal_index, 1.0)]
                + [(index, -yield_value) for _key, index in x_by_group[group]],
                upper=0.0,
                category="scenario_normal_le_production",
                name=f"normal_le_production[{omega},{crop_id},{land_type},{year},{season}]",
            )
            profit_coefficients.append(
                (normal_index, -(1.0 - config.surplus_discount) * price)
            )
            for _plant_key, index in x_by_group[group]:
                cost = parameter.cost_yuan_per_mu * scenarios.cost_multiplier[
                    (omega, crop_id, year)
                ]
                profit_coefficients.append(
                    (index, cost - config.surplus_discount * price * yield_value)
                )

        for (crop_id, year, season), demand_sales_groups in demand_groups.items():
            builder.add(
                (
                    (variables.normal_sales[(omega, group)], 1.0)
                    for group in demand_sales_groups
                ),
                upper=scenarios.demand_jin[(omega, crop_id, year, season)],
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


def build_q3_model(
    data: Q1Data,
    scenarios: Q3ScenarioPaths,
    *,
    objective_kind: str,
    cvar_floor_yuan: float | None = None,
    expected_profit_floor_yuan: float | None = None,
    config: Q3Config | None = None,
) -> Q2BuiltModel:
    config = scenarios.config if config is None else config
    if config != scenarios.config:
        raise Q2ModelError("Q3 model config must match the materialized scenarios")
    if not math.isclose(
        math.fsum(float(value) for value in scenarios.probabilities),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise Q2ModelError("Q3 scenario probabilities must sum to one")
    objective_config = _objective_config(config)
    variables = _build_variables(data, scenarios, objective_kind, objective_config)
    builder = SparseConstraintBuilder(len(variables.names))
    _add_agricultural_constraints(builder, data, variables)
    _add_q3_economic_constraints(builder, data, scenarios, variables, config)
    _add_cvar_constraints(
        builder, scenarios, variables, objective_config, cvar_floor_yuan
    )
    _add_expected_profit_floor(
        builder, scenarios, variables, expected_profit_floor_yuan
    )
    return Q2BuiltModel(
        data=data,
        scenarios=scenarios,
        config=objective_config,
        objective_kind=objective_kind,
        cvar_floor_yuan=cvar_floor_yuan,
        expected_profit_floor_yuan=expected_profit_floor_yuan,
        variables=variables,
        constraints=builder.build(),
    )


def build_q3_lp_relaxation(model: Q2BuiltModel) -> Q2BuiltModel:
    return build_q2_lp_relaxation(model)


def solve_q3_model(
    model: Q2BuiltModel,
    *,
    time_limit_seconds: float,
    mip_relative_gap: float = 0.005,
    display_solver_log: bool = False,
) -> Q2Solution:
    return solve_q2_model(
        model,
        time_limit_seconds=time_limit_seconds,
        mip_relative_gap=mip_relative_gap,
        display_solver_log=display_solver_log,
    )
