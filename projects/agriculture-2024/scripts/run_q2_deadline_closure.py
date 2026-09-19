#!/usr/bin/env python3
"""Human-Gate-approved near-utopia closure for CUMCM 2024 C, Q2."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for directory in (SRC_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from q1_data import load_q1_data  # noqa: E402
from q2_config import DEFAULT_Q2_CONFIG  # noqa: E402
from q2_model import (  # noqa: E402
    build_q2_lp_relaxation,
    build_q2_model,
    solve_q2_model,
)
from q2_outputs import scenario_fingerprint, sha256, write_json  # noqa: E402
from q2_uncertainty import generate_scenarios  # noqa: E402
from q2_validate import (  # noqa: E402
    audit_q2_lp_relaxation,
    evaluate_plan,
    load_plan_csv,
    plan_diagnostics,
    summarize_evaluation,
    validate_agricultural_plan,
)
from run_q2 import (  # noqa: E402
    _independently_evaluate_saved_plan,
    require_freeze,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "q2"
EXPECTED_CANDIDATE_ID = "q2_main_expected_endpoint_polished"
APPROVED_EXPECTED_PROFIT = 69_640_628.7089308
APPROVED_CVAR = 68_312_890.8155968
APPROVED_EXPECTED_UPPER_BOUND = 69_646_477.5038949
APPROVED_SCENARIO_FINGERPRINT = (
    "7C8C9CBBAAE2494842EB31B83F92A0FCC91F39020A90A0BA1D3D4EC97ED43272"
)
CVaR_FLOOR_TOLERANCE_YUAN = 0.1
CERTIFICATION_RELATIVE_TOLERANCE = 0.005
Q1_CASE1_PROFIT = 41_044_527.130780034
Q1_CASE2_PROFIT = 63_909_448.53304118


def _require_close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.1):
        raise RuntimeError(f"{name} changed: actual={actual}, expected={expected}")


def _write_lp_solution(solution, source_integrality: np.ndarray) -> dict[str, str]:
    variables_path = OUTPUT_DIR / "q2_max_cvar_lp_relaxation_variables.csv"
    profits_path = OUTPUT_DIR / "q2_max_cvar_lp_relaxation_scenario_profits.csv"
    pd.DataFrame(
        {
            "variable": solution.model.variables.names,
            "value": solution.values,
            "lower_bound": solution.model.variables.lower_bounds,
            "upper_bound": solution.model.variables.upper_bounds,
            "source_milp_integrality": source_integrality,
            "relaxed_integrality": solution.model.variables.integrality,
        }
    ).to_csv(
        variables_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.15g",
    )
    pd.DataFrame(
        {
            "scenario": solution.model.scenarios.scenario_ids,
            "probability": solution.model.scenarios.probabilities,
            "lp_scenario_profit_yuan": solution.scenario_profits,
        }
    ).to_csv(
        profits_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.15g",
    )
    return {
        "variables": str(variables_path.relative_to(PROJECT_ROOT)),
        "variables_sha256": sha256(variables_path),
        "scenario_profits": str(profits_path.relative_to(PROJECT_ROOT)),
        "scenario_profits_sha256": sha256(profits_path),
    }


def run_certificate(args: argparse.Namespace) -> dict[str, object]:
    freeze_before = require_freeze()
    if not freeze_before["passed"] or freeze_before["checked"] != 25:
        raise RuntimeError(f"Q1 v2 freeze gate failed: {freeze_before}")

    data = load_q1_data(PROJECT_ROOT)
    scenarios = generate_scenarios(
        data,
        count=DEFAULT_Q2_CONFIG.main_scenarios,
        seed=DEFAULT_Q2_CONFIG.optimization_seed,
        sampling_method=DEFAULT_Q2_CONFIG.optimization_sampling,
    )
    fingerprint = scenario_fingerprint(scenarios.unit_draws)
    if fingerprint != APPROVED_SCENARIO_FINGERPRINT:
        raise RuntimeError(f"Scenario fingerprint changed: {fingerprint}")

    candidate = _independently_evaluate_saved_plan(
        data,
        scenarios,
        candidate_id=EXPECTED_CANDIDATE_ID,
        require_saved_profit_reconciliation=True,
    )
    mean_plan = float(candidate["independently_recomputed_expected_profit_yuan"])
    cvar_plan = float(candidate["independently_recomputed_cvar_90_yuan"])
    _require_close(mean_plan, APPROVED_EXPECTED_PROFIT, name="Expected profit")
    _require_close(cvar_plan, APPROVED_CVAR, name="CVaR")

    receipt_path = OUTPUT_DIR / f"{EXPECTED_CANDIDATE_ID}_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    dual_bound_minimization = float(receipt["solver"]["mip_dual_bound"])
    mean_upper_bound = -dual_bound_minimization
    _require_close(
        mean_upper_bound,
        APPROVED_EXPECTED_UPPER_BOUND,
        name="Expected-profit solver upper bound",
    )
    if mean_upper_bound + 0.1 < mean_plan:
        raise RuntimeError("Expected-profit solver upper bound is below feasible value")

    cvar_floor = cvar_plan - CVaR_FLOOR_TOLERANCE_YUAN
    integer_model = build_q2_model(
        data,
        scenarios,
        objective_kind="max_cvar",
        cvar_floor_yuan=cvar_floor,
    )
    relaxed_model = build_q2_lp_relaxation(integer_model)
    lp_solution = solve_q2_model(
        relaxed_model,
        time_limit_seconds=args.lp_time_limit,
        mip_relative_gap=0.0,
        display_solver_log=args.solver_log,
    )
    lp_audit = audit_q2_lp_relaxation(integer_model, lp_solution)
    if not lp_audit["passed"]:
        raise RuntimeError(f"LP-relaxation independent audit failed: {lp_audit}")
    cvar_upper_bound = float(lp_audit["upper_bound_validated_yuan"])
    if cvar_upper_bound + 0.1 < cvar_plan:
        raise RuntimeError("LP CVaR upper bound is below feasible candidate CVaR")
    lp_paths = _write_lp_solution(
        lp_solution, integer_model.variables.integrality
    )

    mean_absolute_gap = mean_upper_bound - mean_plan
    cvar_absolute_gap = cvar_upper_bound - cvar_plan
    mean_relative_gap = mean_absolute_gap / abs(mean_plan)
    cvar_relative_gap = cvar_absolute_gap / abs(cvar_plan)
    mean_pass = bool(mean_relative_gap <= CERTIFICATION_RELATIVE_TOLERANCE)
    cvar_pass = bool(cvar_relative_gap <= CERTIFICATION_RELATIVE_TOLERANCE)
    certificate_pass = mean_pass and cvar_pass

    freeze_after = require_freeze()
    if freeze_after != freeze_before:
        raise RuntimeError("Q1 v2 freeze result changed during LP certification")
    proof = (
        "All source MILP constraints, bounds, coefficients, scenario equations, "
        "economic equations, and CVaR equations are retained. Only integrality is "
        "relaxed to continuous variables within the same bounds, so the LP feasible "
        "region contains the MILP feasible region. The incumbent-derived CVaR floor "
        "does not exclude any maximum-CVaR MILP optimum because a validated feasible "
        "plan already attains a value above that floor. Therefore the maximum LP "
        "CVaR is a valid upper bound on the maximum achievable MILP CVaR."
    )
    payload: dict[str, object] = {
        "status": "PASS" if certificate_pass else "BLOCKED",
        "classification": (
            "SOLVER-CERTIFIED 0.5%-NEAR-UTOPIA BICRITERIA SOLUTION"
            if certificate_pass
            else "NEAR-UTOPIA CERTIFICATION FAILED"
        ),
        "terminology": (
            "The plan is within solver-certified upper-bound tolerances of both "
            "single-objective ideals; this is not a claim of exact Pareto optimality "
            "or global optimality in both objectives."
        ),
        "scenario": {
            "count": scenarios.count,
            "seed": scenarios.seed,
            "sampling_method": scenarios.sampling_method,
            "fingerprint_sha256": fingerprint,
        },
        "candidate_revalidation": candidate,
        "expected_profit_ideal": {
            "feasible_plan_value_yuan": mean_plan,
            "solver_upper_bound_yuan": mean_upper_bound,
            "solver_minimization_dual_bound": dual_bound_minimization,
            "absolute_gap_yuan": mean_absolute_gap,
            "relative_gap": mean_relative_gap,
            "threshold": CERTIFICATION_RELATIVE_TOLERANCE,
            "passed": mean_pass,
        },
        "maximum_cvar_ideal": {
            "feasible_plan_cvar_yuan": cvar_plan,
            "lp_relaxation_upper_bound_yuan": cvar_upper_bound,
            "absolute_gap_yuan": cvar_absolute_gap,
            "relative_gap": cvar_relative_gap,
            "threshold": CERTIFICATION_RELATIVE_TOLERANCE,
            "passed": cvar_pass,
            "source_milp_cvar_floor_yuan": cvar_floor,
            "lp_solve_seconds": lp_solution.solve_seconds,
            "lp_solver_status": int(lp_solution.result.status),
            "lp_solver_message": str(lp_solution.result.message),
            "lp_audit": lp_audit,
            "files": lp_paths,
        },
        "upper_bound_proof": proof,
        "frontier_decision": (
            "Cancel further maximum-CVaR integer polishing, theta points, and knee "
            "selection under the approved deadline rule."
            if certificate_pass
            else "STOP; do not invent or approximate the frontier."
        ),
        "q1_freeze": freeze_after,
    }
    output_path = OUTPUT_DIR / "q2_near_utopia_certificate.json"
    write_json(output_path, payload)
    if not certificate_pass:
        raise RuntimeError(
            f"Near-utopia certificate failed: mean={mean_relative_gap}, "
            f"cvar={cvar_relative_gap}"
        )
    return payload


def _finding_payload(validation) -> list[dict[str, object]]:
    return [
        {
            "name": finding.name,
            "passed": finding.passed,
            "max_violation": finding.max_violation,
            "details": finding.details,
        }
        for finding in validation.findings
    ]


def _downside_metrics(profits: np.ndarray) -> dict[str, float]:
    return {
        "probability_profit_below_zero": float(np.mean(profits < 0.0)),
        "probability_below_q1_case1_deterministic_profit": float(
            np.mean(profits < Q1_CASE1_PROFIT)
        ),
        "probability_below_q1_case2_deterministic_profit": float(
            np.mean(profits < Q1_CASE2_PROFIT)
        ),
    }


def run_evaluation() -> dict[str, object]:
    certificate_path = OUTPUT_DIR / "q2_near_utopia_certificate.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if certificate.get("status") != "PASS":
        raise RuntimeError("Near-utopia certificate is absent or not PASS")

    freeze_before = require_freeze()
    if not freeze_before["passed"] or freeze_before["checked"] != 25:
        raise RuntimeError(f"Q1 v2 freeze gate failed: {freeze_before}")

    source_plan_path = OUTPUT_DIR / f"{EXPECTED_CANDIDATE_ID}_plan.csv"
    final_plan_path = OUTPUT_DIR / "q2_final_plan.csv"
    shutil.copy2(source_plan_path, final_plan_path)
    data = load_q1_data(PROJECT_ROOT)
    scenarios = generate_scenarios(
        data,
        count=DEFAULT_Q2_CONFIG.evaluation_paths,
        seed=DEFAULT_Q2_CONFIG.evaluation_seed,
        sampling_method=DEFAULT_Q2_CONFIG.evaluation_sampling,
    )
    evaluation_fingerprint = scenario_fingerprint(scenarios.unit_draws)

    plan_paths = {
        "q2_final_near_utopia": final_plan_path,
        "q1_case1_frozen": PROJECT_ROOT / "outputs" / "q1_case1_long.csv",
        "q1_case2_frozen": PROJECT_ROOT / "outputs" / "q1_case2_long.csv",
    }
    plans = {name: load_plan_csv(path) for name, path in plan_paths.items()}
    agricultural = {
        name: validate_agricultural_plan(data, plan)
        for name, plan in plans.items()
    }
    failed_agricultural = [
        name for name, result in agricultural.items() if not result.passed
    ]
    if failed_agricultural:
        raise RuntimeError(
            f"Agricultural validation failed for: {failed_agricultural}"
        )

    rules = {
        "half_price_surplus": 0.5,
        "waste_all_surplus": 0.0,
    }
    summary_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    evaluations = {}
    for rule_name, discount in rules.items():
        for plan_name, plan in plans.items():
            evaluation = evaluate_plan(
                data,
                plan,
                scenarios,
                surplus_discount=discount,
            )
            evaluations[(rule_name, plan_name)] = evaluation
            summary = summarize_evaluation(
                evaluation,
                probabilities=scenarios.probabilities,
            )
            row = {
                "surplus_rule": rule_name,
                "surplus_discount": discount,
                "plan": plan_name,
                **summary,
                **_downside_metrics(evaluation.profits_yuan),
                **plan_diagnostics(data, plan),
            }
            summary_rows.append(row)
            for scenario_id in scenarios.scenario_ids:
                path_rows.append(
                    {
                        "surplus_rule": rule_name,
                        "surplus_discount": discount,
                        "plan": plan_name,
                        "scenario": scenario_id,
                        "profit_yuan": evaluation.profits_yuan[scenario_id],
                        "revenue_yuan": evaluation.revenue_yuan[scenario_id],
                        "production_jin": evaluation.production_jin[scenario_id],
                        "normal_sales_jin": evaluation.normal_sales_jin[scenario_id],
                        "surplus_jin": evaluation.surplus_jin[scenario_id],
                        "planting_cost_yuan": evaluation.planting_cost_yuan,
                    }
                )

    summary_frame = pd.DataFrame(summary_rows)
    path_frame = pd.DataFrame(path_rows)
    summary_path = OUTPUT_DIR / "q2_oos_3000_summary.csv"
    path_path = OUTPUT_DIR / "q2_oos_3000_pathwise.csv"
    summary_frame.to_csv(
        summary_path, index=False, encoding="utf-8-sig", float_format="%.15g"
    )
    path_frame.to_csv(
        path_path, index=False, encoding="utf-8-sig", float_format="%.15g"
    )

    pathwise_dominance = {}
    ranking = {}
    for plan_name in plans:
        half = evaluations[("half_price_surplus", plan_name)].profits_yuan
        waste = evaluations[("waste_all_surplus", plan_name)].profits_yuan
        difference = half - waste
        pathwise_dominance[plan_name] = {
            "minimum_half_minus_waste_yuan": float(np.min(difference)),
            "maximum_half_minus_waste_yuan": float(np.max(difference)),
            "mean_half_minus_waste_yuan": float(np.mean(difference)),
            "half_price_pathwise_dominates_waste": bool(
                np.all(difference >= -0.1)
            ),
        }
    for rule_name in rules:
        selected = summary_frame[summary_frame["surplus_rule"] == rule_name]
        ranking[rule_name] = {
            "mean_profit_descending": selected.sort_values(
                "mean_profit_yuan", ascending=False
            )["plan"].tolist(),
            "cvar_90_descending": selected.sort_values(
                "cvar_90_yuan", ascending=False
            )["plan"].tolist(),
        }
    ranking_robust = bool(
        ranking["half_price_surplus"]["mean_profit_descending"]
        == ranking["waste_all_surplus"]["mean_profit_descending"]
        and ranking["half_price_surplus"]["cvar_90_descending"]
        == ranking["waste_all_surplus"]["cvar_90_descending"]
    )
    sensitivity_pass = all(
        item["half_price_pathwise_dominates_waste"]
        for item in pathwise_dominance.values()
    )

    freeze_after = require_freeze()
    if freeze_after != freeze_before:
        raise RuntimeError("Q1 v2 freeze result changed during OOS evaluation")
    payload: dict[str, object] = {
        "status": "PASS" if sensitivity_pass else "BLOCKED",
        "evaluation_design": {
            "scope": "independent_out_of_sample_fixed_plan_evaluation",
            "scenario_count": scenarios.count,
            "seed": scenarios.seed,
            "sampling_method": scenarios.sampling_method,
            "common_random_numbers": True,
            "scenario_fingerprint_sha256": evaluation_fingerprint,
            "optimization_performed": False,
            "in_sample_reference": {
                "scenario_count": DEFAULT_Q2_CONFIG.main_scenarios,
                "seed": DEFAULT_Q2_CONFIG.optimization_seed,
                "sampling_method": DEFAULT_Q2_CONFIG.optimization_sampling,
            },
        },
        "plans": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
                "agricultural_validation_passed": agricultural[name].passed,
                "agricultural_findings": _finding_payload(agricultural[name]),
                "concentration_diagnostics": plan_diagnostics(data, plans[name]),
            }
            for name, path in plan_paths.items()
        },
        "q1_deterministic_reference_profits_yuan": {
            "q1_case1": Q1_CASE1_PROFIT,
            "q1_case2": Q1_CASE2_PROFIT,
        },
        "rankings": ranking,
        "ranking_robust_across_surplus_rules": ranking_robust,
        "pathwise_surplus_rule_checks": pathwise_dominance,
        "sensitivity_validation_passed": sensitivity_pass,
        "files": {
            "summary": str(summary_path.relative_to(PROJECT_ROOT)),
            "summary_sha256": sha256(summary_path),
            "pathwise": str(path_path.relative_to(PROJECT_ROOT)),
            "pathwise_sha256": sha256(path_path),
            "final_plan": str(final_plan_path.relative_to(PROJECT_ROOT)),
            "final_plan_sha256": sha256(final_plan_path),
        },
        "q1_freeze": freeze_after,
    }
    result_path = OUTPUT_DIR / "q2_oos_3000_evaluation.json"
    write_json(result_path, payload)
    if not sensitivity_pass:
        raise RuntimeError("Pathwise surplus sensitivity validation failed")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("certificate", "evaluation", "all"), default="certificate"
    )
    parser.add_argument("--lp-time-limit", type=float, default=1800.0)
    parser.add_argument("--solver-log", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "certificate":
        result = run_certificate(args)
    elif args.stage == "evaluation":
        result = run_evaluation()
    else:
        run_certificate(args)
        result = run_evaluation()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
