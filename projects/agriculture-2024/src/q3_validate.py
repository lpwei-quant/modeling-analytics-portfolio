"""Independent Q3 fixed-plan economics and delivery diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from q1_data import Q1Data
from q1_model import PlantKey, SalesKey, sales_groups
from q2_model import Q2Solution
from q2_uncertainty import lower_tail_cvar
from q2_validate import validate_agricultural_plan
from q3_scenarios import Q3ScenarioPaths


@dataclass(frozen=True)
class Q3PlanEvaluation:
    profits_yuan: np.ndarray
    revenue_yuan: np.ndarray
    production_jin: np.ndarray
    normal_sales_jin: np.ndarray
    surplus_jin: np.ndarray
    planting_cost_yuan: np.ndarray
    surplus_discount: float


def evaluate_q3_plan(
    data: Q1Data,
    plan: dict[PlantKey, float],
    scenarios: Q3ScenarioPaths,
    *,
    surplus_discount: float,
) -> Q3PlanEvaluation:
    if not 0.0 <= surplus_discount <= 1.0:
        raise ValueError("surplus discount outside [0,1]")
    groups = sales_groups(data)
    base_production: dict[SalesKey, float] = defaultdict(float)
    positive_rows: list[tuple[int, int, float, float]] = []
    for (plot, crop_id, year, season), area in plan.items():
        if area <= 0.0:
            continue
        land_type = data.plots[plot].land_type
        parameter = data.parameters[(crop_id, land_type, season)]
        base_production[(crop_id, land_type, year, season)] += (
            area * parameter.yield_jin_per_mu
        )
        positive_rows.append((crop_id, year, area, parameter.cost_yuan_per_mu))
    demand_groups: dict[tuple[int, int, str], list[SalesKey]] = defaultdict(list)
    for group in groups:
        crop_id, _land_type, year, season = group
        demand_groups[(crop_id, year, season)].append(group)

    count = scenarios.count
    profits = np.zeros(count)
    revenues = np.zeros(count)
    productions = np.zeros(count)
    normal_totals = np.zeros(count)
    surplus_totals = np.zeros(count)
    costs = np.zeros(count)
    for omega in scenarios.scenario_ids:
        costs[omega] = math.fsum(
            area
            * base_cost
            * scenarios.cost_multiplier[(omega, crop_id, year)]
            for crop_id, year, area, base_cost in positive_rows
        )
        production = {}
        prices = {}
        for group in groups:
            crop_id, land_type, year, season = group
            parameter = data.parameters[(crop_id, land_type, season)]
            production[group] = base_production.get(group, 0.0) * scenarios.yield_multiplier[
                (omega, crop_id, year)
            ]
            prices[group] = parameter.midpoint_price_yuan_per_jin * scenarios.price_multiplier[
                (omega, crop_id, year)
            ]
        revenue = 0.0
        normal_total = 0.0
        for (crop_id, year, season), related_groups in demand_groups.items():
            remaining = scenarios.demand_jin[(omega, crop_id, year, season)]
            for group in sorted(related_groups, key=lambda key: (-prices[key], key[1])):
                normal = min(production[group], max(0.0, remaining))
                excess = production[group] - normal
                revenue += prices[group] * (normal + surplus_discount * excess)
                normal_total += normal
                remaining -= normal
        total_production = math.fsum(production.values())
        revenues[omega] = revenue
        productions[omega] = total_production
        normal_totals[omega] = normal_total
        surplus_totals[omega] = total_production - normal_total
        profits[omega] = revenue - costs[omega]
    return Q3PlanEvaluation(
        profits_yuan=profits,
        revenue_yuan=revenues,
        production_jin=productions,
        normal_sales_jin=normal_totals,
        surplus_jin=surplus_totals,
        planting_cost_yuan=costs,
        surplus_discount=surplus_discount,
    )


def validate_q3_solution_economics(
    solution: Q2Solution, *, tolerance_yuan: float = 0.1
) -> dict[str, object]:
    independent = evaluate_q3_plan(
        solution.model.data,
        solution.x,
        solution.model.scenarios,
        surplus_discount=solution.model.config.surplus_discount,
    )
    profit_diff = np.abs(independent.profits_yuan - solution.scenario_profits)
    expected_independent = float(
        np.dot(solution.model.scenarios.probabilities, independent.profits_yuan)
    )
    expected_difference = abs(expected_independent - solution.expected_profit_yuan)
    cvar_independent = lower_tail_cvar(
        independent.profits_yuan,
        solution.model.config.beta,
        solution.model.scenarios.probabilities,
    )
    cvar_difference = abs(cvar_independent - solution.empirical_cvar_yuan)
    agriculture = validate_agricultural_plan(solution.model.data, solution.x)
    passed = bool(
        agriculture.passed
        and float(profit_diff.max(initial=0.0)) <= tolerance_yuan
        and expected_difference <= tolerance_yuan
        and cvar_difference <= tolerance_yuan
    )
    return {
        "passed": passed,
        "agricultural_validation_passed": agriculture.passed,
        "maximum_scenario_profit_abs_difference_yuan": float(
            profit_diff.max(initial=0.0)
        ),
        "expected_profit_abs_difference_yuan": expected_difference,
        "cvar_abs_difference_yuan": cvar_difference,
        "independent_expected_profit_yuan": expected_independent,
        "independent_cvar_90_yuan": cvar_independent,
    }


def summarize_q3_evaluation(
    evaluation: Q3PlanEvaluation,
    *,
    beta: float = 0.90,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    values = evaluation.profits_yuan
    standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return {
        "mean_profit_yuan": float(np.mean(values)),
        "standard_deviation_yuan": standard_deviation,
        "quantile_05_yuan": float(np.quantile(values, 0.05)),
        "quantile_10_yuan": float(np.quantile(values, 0.10)),
        "quantile_20_yuan": float(np.quantile(values, 0.20)),
        "cvar_90_yuan": lower_tail_cvar(values, beta, probabilities),
        "minimum_profit_yuan": float(np.min(values)),
        "mean_cost_yuan": float(np.mean(evaluation.planting_cost_yuan)),
        "mean_production_jin": float(np.mean(evaluation.production_jin)),
        "mean_normal_sales_jin": float(np.mean(evaluation.normal_sales_jin)),
        "mean_surplus_jin": float(np.mean(evaluation.surplus_jin)),
    }


def audit_q3_scenarios(paths: Q3ScenarioPaths) -> dict[str, object]:
    support_violations = []
    for key, values in paths.innovations.items():
        support = paths.supports[key]
        violation = max(
            0.0,
            float(support.lower - np.min(values)),
            float(np.max(values) - support.upper),
        )
        if violation > 1e-12:
            support_violations.append({"key": key, "violation": violation})
    interaction_values = np.asarray(list(paths.demand_interaction_multiplier.values()))
    dependency_correlations = {
        "yield_wheat_corn_2024": float(
            np.corrcoef(
                paths.innovations[("yield_delta", 6, 2024)],
                paths.innovations[("yield_delta", 7, 2024)],
            )[0, 1]
        ),
        "demand_wheat_corn_2024": float(
            np.corrcoef(
                paths.innovations[("demand_growth", 6, 2024)],
                paths.innovations[("demand_growth", 7, 2024)],
            )[0, 1]
        ),
        "cost_wheat_tomato_2024": float(
            np.corrcoef(
                paths.innovations[("cost_growth", 6, 2024)],
                paths.innovations[("cost_growth", 21, 2024)],
            )[0, 1]
        ),
        "price_cost_tomato_2024": float(
            np.corrcoef(
                paths.innovations[("price_change", 21, 2024)],
                paths.innovations[("cost_growth", 21, 2024)],
            )[0, 1]
        ),
        "price_demand_tomato_2024": float(
            np.corrcoef(
                paths.innovations[("price_change", 21, 2024)],
                paths.innovations[("demand_delta", 21, 2024)],
            )[0, 1]
        ),
    }
    finite = bool(
        np.all(np.isfinite(paths.unit_draws))
        and np.all(np.isfinite(interaction_values))
        and all(math.isfinite(value) for value in paths.demand_jin.values())
        and all(math.isfinite(value) for value in paths.price_multiplier.values())
        and all(math.isfinite(value) for value in paths.cost_multiplier.values())
    )
    passed = bool(
        finite
        and not support_violations
        and np.min(paths.unit_draws) >= 0.0
        and np.max(paths.unit_draws) <= 1.0
        and np.min(interaction_values) >= paths.config.demand_interaction_lower - 1e-12
        and np.max(interaction_values) <= paths.config.demand_interaction_upper + 1e-12
        and min(paths.demand_jin.values()) >= 0.0
        and min(paths.price_multiplier.values()) > 0.0
        and min(paths.cost_multiplier.values()) > 0.0
    )
    return {
        "passed": passed,
        "finite": finite,
        "support_violation_count": len(support_violations),
        "support_violations": support_violations[:20],
        "minimum_unit_draw": float(np.min(paths.unit_draws)),
        "maximum_unit_draw": float(np.max(paths.unit_draws)),
        "minimum_demand_interaction_multiplier": float(np.min(interaction_values)),
        "maximum_demand_interaction_multiplier": float(np.max(interaction_values)),
        "dependency_correlations": dependency_correlations,
    }


def planting_pattern_difference(
    data: Q1Data,
    first: dict[PlantKey, float],
    second: dict[PlantKey, float],
    *,
    tolerance: float = 1e-6,
) -> dict[str, float | int]:
    first_crop: dict[int, float] = defaultdict(float)
    second_crop: dict[int, float] = defaultdict(float)
    all_keys = set(data.compatible_cells)
    cell_differences = []
    changed_positive = 0
    for key in all_keys:
        first_value = float(first.get(key, 0.0))
        second_value = float(second.get(key, 0.0))
        first_crop[key[1]] += first_value
        second_crop[key[1]] += second_value
        difference = abs(first_value - second_value)
        cell_differences.append(difference)
        if difference > tolerance:
            changed_positive += 1
    crop_l1 = math.fsum(
        abs(first_crop[crop_id] - second_crop[crop_id]) for crop_id in data.crops
    )
    cell_l1 = math.fsum(cell_differences)
    total = max(math.fsum(first.values()), math.fsum(second.values()), tolerance)
    first_vector = np.asarray([first_crop[crop] for crop in sorted(data.crops)])
    second_vector = np.asarray([second_crop[crop] for crop in sorted(data.crops)])
    correlation = float(np.corrcoef(first_vector, second_vector)[0, 1])
    return {
        "cell_area_l1_mu": cell_l1,
        "normalized_cell_area_l1": cell_l1 / total,
        "crop_total_area_l1_mu": crop_l1,
        "normalized_crop_total_area_l1": crop_l1 / total,
        "changed_cells": changed_positive,
        "crop_area_correlation": correlation,
    }
