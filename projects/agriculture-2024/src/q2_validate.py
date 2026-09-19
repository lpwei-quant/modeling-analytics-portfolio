"""Independent Q2 validation and fixed-plan economic evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from q1_data import (
    GREENHOUSE_TYPES,
    IRRIGATED_LAND,
    OPEN_LAND_TYPES,
    SEASON_FIRST,
    SEASON_SECOND,
    SEASON_SINGLE,
    Q1Data,
    YEARS,
)
from q1_model import PlantKey, SalesKey, sales_groups
from q2_config import DEFAULT_Q2_CONFIG, Q2Config
from q2_model import Q2BuiltModel, Q2Solution
from q2_uncertainty import Q2ScenarioPaths, cost_multiplier, lower_tail_cvar


TOLERANCE = 1e-6


def audit_q2_lp_relaxation(
    source_integer_model: Q2BuiltModel,
    relaxed_solution: Q2Solution,
    *,
    absolute_tolerance: float = 0.1,
    scaled_tolerance: float = TOLERANCE,
) -> dict[str, object]:
    """Independently audit an explicit continuous max-CVaR relaxation."""

    relaxed_model = relaxed_solution.model
    source_variables = source_integer_model.variables
    relaxed_variables = relaxed_model.variables
    source_constraint = source_integer_model.constraints.linear_constraint
    relaxed_constraint = relaxed_model.constraints.linear_constraint

    matrix_difference_nonzeros = int(
        (source_constraint.A != relaxed_constraint.A).nnz
    )
    structural_checks = {
        "objective_kind_equal": (
            source_integer_model.objective_kind
            == relaxed_model.objective_kind
            == "max_cvar"
        ),
        "cvar_floor_equal": (
            source_integer_model.cvar_floor_yuan == relaxed_model.cvar_floor_yuan
        ),
        "expected_profit_floor_equal": (
            source_integer_model.expected_profit_floor_yuan
            == relaxed_model.expected_profit_floor_yuan
        ),
        "same_scenario_object": source_integer_model.scenarios is relaxed_model.scenarios,
        "same_config": source_integer_model.config == relaxed_model.config,
        "same_variable_names": source_variables.names == relaxed_variables.names,
        "same_variable_index_maps": all(
            getattr(source_variables, name) == getattr(relaxed_variables, name)
            for name in (
                "x",
                "z",
                "regime",
                "normal_sales",
                "scenario_profit",
                "shortfall",
            )
        )
        and source_variables.tau == relaxed_variables.tau,
        "same_lower_bounds": np.array_equal(
            source_variables.lower_bounds, relaxed_variables.lower_bounds
        ),
        "same_upper_bounds": np.array_equal(
            source_variables.upper_bounds, relaxed_variables.upper_bounds
        ),
        "same_objective_coefficients": np.array_equal(
            source_variables.objective, relaxed_variables.objective
        ),
        "same_constraint_matrix": matrix_difference_nonzeros == 0,
        "same_constraint_lower_bounds": np.array_equal(
            source_constraint.lb, relaxed_constraint.lb
        ),
        "same_constraint_upper_bounds": np.array_equal(
            source_constraint.ub, relaxed_constraint.ub
        ),
        "same_constraint_names": (
            source_integer_model.constraints.names == relaxed_model.constraints.names
        ),
        "same_constraint_categories": (
            source_integer_model.constraints.category_counts
            == relaxed_model.constraints.category_counts
        ),
        "all_relaxed_integrality_zero": bool(
            np.count_nonzero(relaxed_variables.integrality) == 0
        ),
    }

    values = np.asarray(relaxed_solution.values, dtype=float)
    lower = relaxed_variables.lower_bounds
    upper = relaxed_variables.upper_bounds
    lower_violation = np.where(np.isfinite(lower), np.maximum(lower - values, 0.0), 0.0)
    upper_violation = np.where(np.isfinite(upper), np.maximum(values - upper, 0.0), 0.0)
    maximum_variable_bound_violation = float(
        max(np.max(lower_violation, initial=0.0), np.max(upper_violation, initial=0.0))
    )

    activity = np.asarray(relaxed_constraint.A @ values, dtype=float)
    row_lower_violation = np.where(
        np.isfinite(relaxed_constraint.lb),
        np.maximum(relaxed_constraint.lb - activity, 0.0),
        0.0,
    )
    row_upper_violation = np.where(
        np.isfinite(relaxed_constraint.ub),
        np.maximum(activity - relaxed_constraint.ub, 0.0),
        0.0,
    )
    row_violation = np.maximum(row_lower_violation, row_upper_violation)
    row_scale = np.maximum(
        1.0,
        np.maximum(
            np.where(np.isfinite(relaxed_constraint.lb), np.abs(relaxed_constraint.lb), 0.0),
            np.where(np.isfinite(relaxed_constraint.ub), np.abs(relaxed_constraint.ub), 0.0),
        ),
    )
    maximum_row_violation = float(np.max(row_violation, initial=0.0))
    maximum_scaled_row_violation = float(
        np.max(row_violation / row_scale, initial=0.0)
    )
    worst_row_index = int(np.argmax(row_violation / row_scale))
    worst_row = {
        "index": worst_row_index,
        "name": relaxed_model.constraints.names[worst_row_index],
        "activity": float(activity[worst_row_index]),
        "lower_bound": float(relaxed_constraint.lb[worst_row_index]),
        "upper_bound": float(relaxed_constraint.ub[worst_row_index]),
        "absolute_violation": float(row_violation[worst_row_index]),
        "scale": float(row_scale[worst_row_index]),
        "scaled_violation": float(
            row_violation[worst_row_index] / row_scale[worst_row_index]
        ),
    }

    recomputed_minimization_objective = float(
        np.dot(relaxed_variables.objective, values)
    )
    solver_minimization_objective = float(relaxed_solution.result.fun)
    objective_abs_difference = abs(
        recomputed_minimization_objective - solver_minimization_objective
    )
    cvar_upper_bound = -recomputed_minimization_objective
    auxiliary_cvar = relaxed_solution.auxiliary_cvar_yuan
    empirical_cvar = relaxed_solution.empirical_cvar_yuan
    auxiliary_objective_difference = (
        math.inf if auxiliary_cvar is None else abs(auxiliary_cvar - cvar_upper_bound)
    )
    empirical_auxiliary_difference = (
        math.inf if auxiliary_cvar is None else abs(empirical_cvar - auxiliary_cvar)
    )

    numerical_checks = {
        "finite_solution": bool(np.all(np.isfinite(values))),
        "variable_bounds_passed": maximum_variable_bound_violation <= absolute_tolerance,
        "linear_constraints_passed": bool(
            maximum_row_violation <= absolute_tolerance
            and maximum_scaled_row_violation <= scaled_tolerance
        ),
        "objective_recompute_passed": objective_abs_difference <= absolute_tolerance,
        "auxiliary_cvar_objective_passed": (
            auxiliary_objective_difference <= absolute_tolerance
        ),
        "empirical_cvar_recompute_passed": (
            empirical_auxiliary_difference <= absolute_tolerance
        ),
    }
    passed = all(structural_checks.values()) and all(numerical_checks.values())
    return {
        "passed": passed,
        "upper_bound_validated_yuan": cvar_upper_bound,
        "auxiliary_cvar_yuan": auxiliary_cvar,
        "empirical_cvar_yuan": empirical_cvar,
        "source_integer_variable_count": int(
            np.count_nonzero(source_variables.integrality)
        ),
        "relaxed_integer_variable_count": int(
            np.count_nonzero(relaxed_variables.integrality)
        ),
        "variable_count": len(relaxed_variables.names),
        "constraint_count": len(relaxed_model.constraints.names),
        "constraint_matrix_nonzeros": relaxed_model.constraints.matrix_nonzeros,
        "matrix_difference_nonzeros": matrix_difference_nonzeros,
        "maximum_variable_bound_violation": maximum_variable_bound_violation,
        "maximum_linear_row_violation": maximum_row_violation,
        "maximum_scaled_linear_row_violation": maximum_scaled_row_violation,
        "worst_scaled_linear_row": worst_row,
        "recomputed_minimization_objective": recomputed_minimization_objective,
        "solver_minimization_objective": solver_minimization_objective,
        "objective_abs_difference": objective_abs_difference,
        "auxiliary_objective_abs_difference_yuan": auxiliary_objective_difference,
        "empirical_auxiliary_abs_difference_yuan": empirical_auxiliary_difference,
        "structural_checks": structural_checks,
        "numerical_checks": numerical_checks,
        "absolute_tolerance": absolute_tolerance,
        "scaled_tolerance": scaled_tolerance,
    }


@dataclass(frozen=True)
class ValidationFinding:
    name: str
    passed: bool
    max_violation: float
    details: str


@dataclass(frozen=True)
class AgriculturalValidation:
    passed: bool
    findings: tuple[ValidationFinding, ...]


@dataclass(frozen=True)
class PlanEvaluation:
    profits_yuan: np.ndarray
    revenue_yuan: np.ndarray
    production_jin: np.ndarray
    normal_sales_jin: np.ndarray
    surplus_jin: np.ndarray
    planting_cost_yuan: float
    surplus_discount: float


@dataclass(frozen=True)
class EconomicValidation:
    passed: bool
    findings: tuple[ValidationFinding, ...]
    independent_evaluation: PlanEvaluation


def compare_approximate_endpoints(
    *,
    risk_neutral_expected_profit: float,
    risk_neutral_cvar: float,
    risk_neutral_mip_gap: float,
    risk_endpoint_expected_profit: float,
    risk_endpoint_cvar: float,
    risk_endpoint_mip_gap: float,
    tolerance_yuan: float = 0.1,
) -> dict[str, object]:
    """Diagnose ordering crossings without treating approximate incumbents as proofs."""

    expected_crossed = (
        risk_endpoint_expected_profit
        > risk_neutral_expected_profit + tolerance_yuan
    )
    cvar_crossed = risk_endpoint_cvar + tolerance_yuan < risk_neutral_cvar
    zero_gap_proved = (
        abs(risk_neutral_mip_gap) <= 1e-12
        and abs(risk_endpoint_mip_gap) <= 1e-12
    )
    blocking = bool(zero_gap_proved and (expected_crossed or cvar_crossed))
    interpretation = (
        "Both endpoint solves report zero MIP gap; an ordering crossing is a "
        "structural contradiction."
        if zero_gap_proved
        else (
            "Endpoint ordering is based on feasible incumbents under nonzero MIP "
            "stopping gaps. Crossings are reported as approximation diagnostics, "
            "not silently treated as exact-frontier contradictions."
        )
    )
    return {
        "risk_neutral_expected_profit_yuan": risk_neutral_expected_profit,
        "risk_endpoint_expected_profit_yuan": risk_endpoint_expected_profit,
        "expected_profit_difference_risk_minus_rn_yuan": (
            risk_endpoint_expected_profit - risk_neutral_expected_profit
        ),
        "risk_neutral_cvar_yuan": risk_neutral_cvar,
        "risk_endpoint_cvar_yuan": risk_endpoint_cvar,
        "cvar_difference_risk_minus_rn_yuan": risk_endpoint_cvar - risk_neutral_cvar,
        "risk_neutral_mip_gap": risk_neutral_mip_gap,
        "risk_endpoint_mip_gap": risk_endpoint_mip_gap,
        "expected_profit_order_crossed": expected_crossed,
        "cvar_order_crossed": cvar_crossed,
        "both_zero_gap_proved": zero_gap_proved,
        "blocking": blocking,
        "interpretation": interpretation,
    }


def assess_frontier_endpoint_consistency(
    *,
    expected_endpoint_expected_profit: float,
    expected_endpoint_cvar: float,
    cvar_endpoint_expected_profit: float,
    cvar_endpoint_cvar: float,
    tolerance_yuan: float = 0.1,
) -> dict[str, object]:
    """Apply the Human-Gate dominance test to independently evaluated endpoints.

    This is deliberately separate from solver-gap diagnostics.  Nonzero gaps
    explain how a dominated incumbent can occur, but a dominated plan still
    cannot serve as an authoritative endpoint for epsilon-target construction.
    """

    cvar_dominates_expected = bool(
        cvar_endpoint_expected_profit
        > expected_endpoint_expected_profit + tolerance_yuan
        and cvar_endpoint_cvar > expected_endpoint_cvar + tolerance_yuan
    )
    expected_dominates_cvar = bool(
        expected_endpoint_expected_profit
        > cvar_endpoint_expected_profit + tolerance_yuan
        and expected_endpoint_cvar > cvar_endpoint_cvar + tolerance_yuan
    )
    return {
        "expected_endpoint_expected_profit_yuan": expected_endpoint_expected_profit,
        "expected_endpoint_cvar_yuan": expected_endpoint_cvar,
        "cvar_endpoint_expected_profit_yuan": cvar_endpoint_expected_profit,
        "cvar_endpoint_cvar_yuan": cvar_endpoint_cvar,
        "delta_mean_cvar_minus_expected_yuan": (
            cvar_endpoint_expected_profit - expected_endpoint_expected_profit
        ),
        "delta_cvar_cvar_minus_expected_yuan": (
            cvar_endpoint_cvar - expected_endpoint_cvar
        ),
        "cvar_endpoint_strictly_dominates_expected_endpoint": cvar_dominates_expected,
        "expected_endpoint_strictly_dominates_cvar_endpoint": expected_dominates_cvar,
        "passed": not (cvar_dominates_expected or expected_dominates_cvar),
        "tolerance_yuan": tolerance_yuan,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_q1_freeze(project_root: Path) -> dict[str, object]:
    manifest_path = project_root / "outputs" / "q1_freeze_manifest_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "q1-freeze-manifest-v2":
        return {
            "passed": False,
            "checked": 0,
            "failures": [
                {
                    "path": str(manifest_path.relative_to(project_root)),
                    "expected": "schema_version=q1-freeze-manifest-v2",
                    "actual": f"schema_version={manifest.get('schema_version')}",
                }
            ],
        }
    failures: list[dict[str, str]] = []
    checked = 0
    for group_name in (
        "authoritative_input_hashes_sha256",
        "core_code_hashes_sha256",
        "result_hashes_sha256",
    ):
        for relative, expected in manifest[group_name].items():
            checked += 1
            actual = _sha256(project_root / relative)
            if actual != expected:
                failures.append(
                    {"path": relative, "expected": expected, "actual": actual}
                )
    return {"passed": not failures, "checked": checked, "failures": failures}


def load_plan_csv(path: Path) -> dict[PlantKey, float]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"plot", "crop_id", "year", "season", "area_mu"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Plan CSV missing columns: {sorted(missing)}")
    plan: dict[PlantKey, float] = {}
    for row in frame.itertuples(index=False):
        key = (str(row.plot), int(row.crop_id), int(row.year), str(row.season))
        if key in plan:
            raise ValueError(f"Duplicate plan key: {key}")
        plan[key] = float(row.area_mu)
    return plan


def _finding(name: str, violations: list[float], details: str) -> ValidationFinding:
    maximum = max(violations, default=0.0)
    return ValidationFinding(
        name=name,
        passed=maximum <= TOLERANCE,
        max_violation=float(maximum),
        details=details,
    )


def validate_agricultural_plan(
    data: Q1Data,
    plan: dict[PlantKey, float],
    *,
    tolerance: float = TOLERANCE,
) -> AgriculturalValidation:
    compatible = set(data.compatible_cells)
    invalid_keys = [key for key in plan if key not in compatible]
    missing_keys = compatible.difference(plan)
    key_violations = [1.0] * len(invalid_keys)
    nonfinite_or_negative = [
        abs(value) if not math.isfinite(value) else -value
        for value in plan.values()
        if not math.isfinite(value) or value < -tolerance
    ]

    area_by_opportunity: dict[tuple[str, int, str], float] = defaultdict(float)
    positive: set[PlantKey] = set()
    for key in compatible:
        value = float(plan.get(key, 0.0))
        plot, _crop, year, season = key
        area_by_opportunity[(plot, year, season)] += max(0.0, value)
        if value > tolerance:
            positive.add(key)

    capacity_violations = [
        max(0.0, area_by_opportunity[(plot, year, season)] - data.plots[plot].area_mu)
        for plot, year, season in data.available_plot_seasons
    ]

    regime_violations: list[float] = []
    second_choice_violations: list[float] = []
    for plot in data.irrigated_plots:
        for year in YEARS:
            rice_area = area_by_opportunity[(plot, year, SEASON_SINGLE)]
            first_area = area_by_opportunity[(plot, year, SEASON_FIRST)]
            second_area = area_by_opportunity[(plot, year, SEASON_SECOND)]
            regime_violations.append(
                min(rice_area, max(first_area, second_area))
            )
            active_second = sum(
                (plot, crop_id, year, SEASON_SECOND) in positive
                for crop_id in data.compatible_crops(plot, SEASON_SECOND)
            )
            second_choice_violations.append(max(0.0, active_second - 1.0))

    rotation_violations: list[float] = []
    for edge in data.adjacency_edges:
        common = set(data.compatible_crops(edge.plot, edge.season_from)).intersection(
            data.compatible_crops(edge.plot, edge.season_to)
        )
        for crop_id in common:
            first = (edge.plot, crop_id, edge.year_from, edge.season_from)
            second = (edge.plot, crop_id, edge.year_to, edge.season_to)
            rotation_violations.append(float(first in positive and second in positive))

    initial_rotation_violations: list[float] = []
    for plot, crop_ids in data.initial_last_crops.items():
        for season in data.initial_first_seasons[plot]:
            for crop_id in crop_ids.intersection(data.compatible_crops(plot, season)):
                initial_rotation_violations.append(
                    float((plot, crop_id, 2024, season) in positive)
                )

    bean_violations: list[float] = []
    for plot, plot_data in data.plots.items():
        first_total = data.bean_area_2023[plot] + math.fsum(
            plan.get((plot, crop_id, year, season), 0.0)
            for crop_id in data.bean_crop_ids
            for year in (2024, 2025)
            for season in (
                SEASON_SINGLE,
                SEASON_FIRST,
                SEASON_SECOND,
            )
            if (plot, crop_id, year, season) in compatible
        )
        bean_violations.append(max(0.0, plot_data.area_mu - first_total))
        for start in range(2024, 2029):
            total = math.fsum(
                plan.get((plot, crop_id, year, season), 0.0)
                for crop_id in data.bean_crop_ids
                for year in range(start, start + 3)
                for season in (SEASON_SINGLE, SEASON_FIRST, SEASON_SECOND)
                if (plot, crop_id, year, season) in compatible
            )
            bean_violations.append(max(0.0, plot_data.area_mu - total))

    findings = (
        _finding(
            "compatible_key_domain",
            key_violations,
            f"invalid={len(invalid_keys)}, omitted_zero_cells={len(missing_keys)}",
        ),
        _finding(
            "finite_nonnegative_area",
            nonfinite_or_negative,
            f"bad_values={len(nonfinite_or_negative)}",
        ),
        _finding("plot_season_capacity", capacity_violations, "independent capacity sums"),
        _finding("irrigated_whole_plot_regime", regime_violations, "rice versus vegetables"),
        _finding(
            "irrigated_second_season_single_crop",
            second_choice_violations,
            "positive-area crop count minus one",
        ),
        _finding("physical_opportunity_rotation", rotation_violations, "positive-area adjacency"),
        _finding("initial_2023_rotation", initial_rotation_violations, "2023 last opportunity"),
        _finding("rolling_three_year_beans", bean_violations, "aggregate-area approximation"),
    )
    return AgriculturalValidation(
        passed=all(finding.max_violation <= tolerance for finding in findings),
        findings=findings,
    )


def _base_group_production_and_cost(
    data: Q1Data, plan: dict[PlantKey, float], config: Q2Config
) -> tuple[dict[SalesKey, float], float]:
    base_production: dict[SalesKey, float] = defaultdict(float)
    total_cost = 0.0
    for key, area in plan.items():
        if area <= 0.0:
            continue
        plot, crop_id, year, season = key
        land_type = data.plots[plot].land_type
        parameter = data.parameters[(crop_id, land_type, season)]
        group = (crop_id, land_type, year, season)
        base_production[group] += area * parameter.yield_jin_per_mu
        total_cost += (
            area * parameter.cost_yuan_per_mu * cost_multiplier(year, config)
        )
    return dict(base_production), float(total_cost)


def evaluate_plan(
    data: Q1Data,
    plan: dict[PlantKey, float],
    scenarios: Q2ScenarioPaths,
    *,
    surplus_discount: float,
    config: Q2Config = DEFAULT_Q2_CONFIG,
) -> PlanEvaluation:
    if not 0.0 <= surplus_discount <= 1.0:
        raise ValueError("surplus_discount must lie in [0, 1]")
    base_production, total_cost = _base_group_production_and_cost(data, plan, config)
    groups = sales_groups(data)
    by_demand_group: dict[tuple[int, int, str], list[SalesKey]] = defaultdict(list)
    for group in groups:
        crop_id, _land_type, year, season = group
        by_demand_group[(crop_id, year, season)].append(group)

    profits = np.zeros(scenarios.count, dtype=float)
    revenues = np.zeros(scenarios.count, dtype=float)
    productions = np.zeros(scenarios.count, dtype=float)
    normal_totals = np.zeros(scenarios.count, dtype=float)
    surplus_totals = np.zeros(scenarios.count, dtype=float)
    for omega in scenarios.scenario_ids:
        production: dict[SalesKey, float] = {}
        prices: dict[SalesKey, float] = {}
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
        total_production = math.fsum(production.values())
        for (crop_id, year, season), demand_groups in by_demand_group.items():
            remaining = scenarios.demand_jin[(omega, crop_id, year, season)]
            for group in sorted(demand_groups, key=lambda key: (-prices[key], key[1])):
                normal = min(production[group], max(0.0, remaining))
                excess = production[group] - normal
                revenue += prices[group] * (normal + surplus_discount * excess)
                normal_total += normal
                remaining -= normal
        revenues[omega] = revenue
        productions[omega] = total_production
        normal_totals[omega] = normal_total
        surplus_totals[omega] = total_production - normal_total
        profits[omega] = revenue - total_cost

    return PlanEvaluation(
        profits_yuan=profits,
        revenue_yuan=revenues,
        production_jin=productions,
        normal_sales_jin=normal_totals,
        surplus_jin=surplus_totals,
        planting_cost_yuan=total_cost,
        surplus_discount=surplus_discount,
    )


def summarize_evaluation(
    evaluation: PlanEvaluation,
    *,
    beta: float = DEFAULT_Q2_CONFIG.beta,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    profits = evaluation.profits_yuan
    standard_deviation = float(np.std(profits, ddof=1)) if profits.size > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(profits.size)
    return {
        "mean_profit_yuan": float(np.mean(profits)),
        "standard_deviation_yuan": standard_deviation,
        "quantile_05_yuan": float(np.quantile(profits, 0.05)),
        "quantile_10_yuan": float(np.quantile(profits, 0.10)),
        "quantile_20_yuan": float(np.quantile(profits, 0.20)),
        "cvar_90_yuan": lower_tail_cvar(profits, beta, probabilities),
        "minimum_profit_yuan": float(np.min(profits)),
        "monte_carlo_standard_error_yuan": standard_error,
        "mean_ci95_lower_yuan": float(np.mean(profits) - 1.96 * standard_error),
        "mean_ci95_upper_yuan": float(np.mean(profits) + 1.96 * standard_error),
        "mean_production_jin": float(np.mean(evaluation.production_jin)),
        "mean_normal_sales_jin": float(np.mean(evaluation.normal_sales_jin)),
        "mean_surplus_jin": float(np.mean(evaluation.surplus_jin)),
        "planting_cost_yuan": evaluation.planting_cost_yuan,
    }


def validate_solution_economics(
    solution: Q2Solution,
    *,
    tolerance_yuan: float = 0.1,
) -> EconomicValidation:
    model = solution.model
    scenarios = model.scenarios
    independent = evaluate_plan(
        model.data,
        solution.x,
        scenarios,
        surplus_discount=model.config.surplus_discount,
        config=model.config,
    )
    profit_differences = np.abs(independent.profits_yuan - solution.scenario_profits)
    expected_difference = abs(
        float(np.dot(scenarios.probabilities, independent.profits_yuan))
        - solution.expected_profit_yuan
    )
    empirical_cvar = lower_tail_cvar(
        solution.scenario_profits, model.config.beta, scenarios.probabilities
    )
    cvar_recompute_difference = abs(empirical_cvar - solution.empirical_cvar_yuan)

    findings: list[ValidationFinding] = [
        ValidationFinding(
            "scenario_profit_independent_recompute",
            float(profit_differences.max(initial=0.0)) <= tolerance_yuan,
            float(profit_differences.max(initial=0.0)),
            "fixed-plan greedy sales allocation versus solver profit variables",
        ),
        ValidationFinding(
            "expected_profit_recompute",
            expected_difference <= tolerance_yuan,
            expected_difference,
            "equal-weight scenario-profit dot product",
        ),
        ValidationFinding(
            "empirical_cvar_recompute",
            cvar_recompute_difference <= tolerance_yuan,
            cvar_recompute_difference,
            "worst-tail probability-mass integration",
        ),
    ]
    if model.objective_kind == "max_cvar":
        auxiliary = solution.auxiliary_cvar_yuan
        difference = math.inf if auxiliary is None else abs(auxiliary - empirical_cvar)
        findings.append(
            ValidationFinding(
                "max_cvar_auxiliary_matches_empirical",
                difference <= tolerance_yuan,
                difference,
                "profit-CVaR sign and tight shortfall representation",
            )
        )
    if model.cvar_floor_yuan is not None:
        violation = max(0.0, model.cvar_floor_yuan - empirical_cvar)
        findings.append(
            ValidationFinding(
                "frontier_empirical_cvar_floor",
                violation <= tolerance_yuan,
                violation,
                f"floor={model.cvar_floor_yuan}",
            )
        )
    return EconomicValidation(
        passed=all(finding.passed for finding in findings),
        findings=tuple(findings),
        independent_evaluation=independent,
    )


def plan_diagnostics(data: Q1Data, plan: dict[PlantKey, float]) -> dict[str, float | int]:
    positive = {key: value for key, value in plan.items() if value > TOLERANCE}
    planted_area = math.fsum(positive.values())
    crop_area: dict[int, float] = defaultdict(float)
    for (_plot, crop_id, _year, _season), area in positive.items():
        crop_area[crop_id] += area
    shares = [area / planted_area for area in crop_area.values()] if planted_area > 0 else []
    active_capacity = math.fsum(
        data.plots[plot].area_mu
        for plot, year, season in data.available_plot_seasons
        if not (
            data.plots[plot].land_type == IRRIGATED_LAND
            and season == SEASON_SINGLE
            and any(
                plan.get((plot, crop_id, year, SEASON_FIRST), 0.0) > TOLERANCE
                for crop_id in data.compatible_crops(plot, SEASON_FIRST)
            )
        )
        if not (
            data.plots[plot].land_type == IRRIGATED_LAND
            and season in {SEASON_FIRST, SEASON_SECOND}
            and any(
                plan.get((plot, crop_id, year, SEASON_SINGLE), 0.0) > TOLERANCE
                for crop_id in data.compatible_crops(plot, SEASON_SINGLE)
            )
        )
    )
    return {
        "planted_area_mu": planted_area,
        "positive_planting_cells": len(positive),
        "crop_count_used": len(crop_area),
        "crop_area_hhi": math.fsum(share * share for share in shares),
        "maximum_crop_area_share": max(shares, default=0.0),
        "active_capacity_mu": active_capacity,
        "idle_area_mu": active_capacity - planted_area,
    }
