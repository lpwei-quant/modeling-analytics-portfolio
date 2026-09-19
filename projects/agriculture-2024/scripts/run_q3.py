#!/usr/bin/env python3
"""Deadline-mode Q3 solve, certificate, OOS comparison and sensitivity."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from create_q2_freeze_manifest_v1 import verify_manifest as verify_q2_freeze  # noqa: E402
from q1_data import load_q1_data  # noqa: E402
from q2_outputs import scenario_fingerprint, sha256, write_json, write_plan_csv  # noqa: E402
from q2_validate import (  # noqa: E402
    audit_q2_lp_relaxation,
    load_plan_csv,
    plan_diagnostics,
    validate_agricultural_plan,
)
from q3_config import (  # noqa: E402
    BASELINE_Q3_CONFIG,
    STRONG_Q3_CONFIG,
    WEAK_Q3_CONFIG,
    Q3Config,
)
from q3_model import (  # noqa: E402
    build_q3_lp_relaxation,
    build_q3_model,
    solve_q3_model,
)
from q3_scenarios import generate_q3_scenarios  # noqa: E402
from q3_validate import (  # noqa: E402
    audit_q3_scenarios,
    evaluate_q3_plan,
    planting_pattern_difference,
    summarize_q3_evaluation,
    validate_q3_solution_economics,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "q3"
CERTIFICATE_TOLERANCE = 0.005
ECONOMIC_TOLERANCE_YUAN = 0.1


def require_freezes() -> dict[str, object]:
    result = verify_q2_freeze()
    if not result["passed"]:
        raise RuntimeError(f"Q2 frozen baseline changed: {result['failures']}")
    return result


def _write_solution(candidate_id: str, solution, validation, scenario_audit, freeze):
    plan_path = OUTPUT_DIR / f"{candidate_id}_plan.csv"
    profits_path = OUTPUT_DIR / f"{candidate_id}_scenario_profits.csv"
    receipt_path = OUTPUT_DIR / f"{candidate_id}_receipt.json"
    write_plan_csv(
        plan_path,
        solution.model.data,
        solution.x,
        candidate_id=candidate_id,
        z=solution.z,
        regimes=solution.regimes,
    )
    pd.DataFrame(
        {
            "scenario": solution.model.scenarios.scenario_ids,
            "probability": solution.model.scenarios.probabilities,
            "profit_yuan": solution.scenario_profits,
        }
    ).to_csv(profits_path, index=False, encoding="utf-8-sig", float_format="%.15g")
    result = solution.result
    receipt = {
        "status": "PASS",
        "candidate_id": candidate_id,
        "scenario_count": solution.model.scenarios.count,
        "scenario_seed": solution.model.scenarios.seed,
        "sampling_method": solution.model.scenarios.sampling_method,
        "scenario_fingerprint_sha256": scenario_fingerprint(
            solution.model.scenarios.unit_draws
        ),
        "expected_profit_yuan": solution.expected_profit_yuan,
        "empirical_cvar_90_yuan": solution.empirical_cvar_yuan,
        "solver": {
            "status": int(result.status),
            "message": str(result.message),
            "mip_gap": float(getattr(result, "mip_gap", math.nan)),
            "mip_dual_bound": float(getattr(result, "mip_dual_bound", math.nan)),
            "solve_seconds": solution.solve_seconds,
        },
        "model_size": {
            "variables": len(solution.model.variables.names),
            "integer_variables": int(np.count_nonzero(solution.model.variables.integrality)),
            "constraints": len(solution.model.constraints.names),
            "matrix_nonzeros": solution.model.constraints.matrix_nonzeros,
        },
        "independent_validation": validation,
        "scenario_audit": scenario_audit,
        "q2_freeze": freeze,
        "q3_config": asdict(solution.model.scenarios.config),
    }
    write_json(receipt_path, receipt)
    return {
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": sha256(plan_path),
        "scenario_profits": str(profits_path.relative_to(PROJECT_ROOT)),
        "scenario_profits_sha256": sha256(profits_path),
        "receipt": str(receipt_path.relative_to(PROJECT_ROOT)),
        "receipt_sha256": sha256(receipt_path),
    }


def solve_expected_candidate(*, count: int, candidate_id: str, time_limit: float, log: bool):
    freeze_before = require_freezes()
    data = load_q1_data(PROJECT_ROOT)
    scenarios = generate_q3_scenarios(
        data,
        count=count,
        seed=BASELINE_Q3_CONFIG.optimization_seed,
        sampling_method="lhs",
    )
    scenario_audit = audit_q3_scenarios(scenarios)
    if not scenario_audit["passed"]:
        raise RuntimeError(f"Q3 scenario audit failed: {scenario_audit}")
    model = build_q3_model(data, scenarios, objective_kind="risk_neutral")
    solution = solve_q3_model(
        model,
        time_limit_seconds=time_limit,
        mip_relative_gap=0.005,
        display_solver_log=log,
    )
    validation = validate_q3_solution_economics(solution)
    if not validation["passed"]:
        raise RuntimeError(f"Q3 independent economic validation failed: {validation}")
    freeze_after = require_freezes()
    if freeze_after != freeze_before:
        raise RuntimeError("Q2 freeze result changed during Q3 solve")
    files = _write_solution(
        candidate_id, solution, validation, scenario_audit, freeze_after
    )
    return data, scenarios, solution, files


def run_smoke(args) -> dict[str, object]:
    _data, scenarios, solution, files = solve_expected_candidate(
        count=BASELINE_Q3_CONFIG.smoke_scenarios,
        candidate_id="q3_smoke_expected",
        time_limit=args.smoke_time_limit,
        log=args.solver_log,
    )
    payload = {
        "status": "PASS",
        "scenario_count": scenarios.count,
        "expected_profit_yuan": solution.expected_profit_yuan,
        "cvar_90_yuan": solution.empirical_cvar_yuan,
        "files": files,
    }
    write_json(OUTPUT_DIR / "q3_smoke_gate.json", payload)
    return payload


def run_certificate(args) -> dict[str, object]:
    data, scenarios, candidate, files = solve_expected_candidate(
        count=BASELINE_Q3_CONFIG.main_scenarios,
        candidate_id="q3_main_expected",
        time_limit=args.main_time_limit,
        log=args.solver_log,
    )
    result = candidate.result
    minimization_bound = float(getattr(result, "mip_dual_bound", math.nan))
    if not math.isfinite(minimization_bound):
        raise RuntimeError("Expected-profit solver did not provide a finite dual bound")
    expected_upper = -minimization_bound
    expected_value = candidate.expected_profit_yuan
    expected_gap = (expected_upper - expected_value) / abs(expected_value)

    source_cvar_model = build_q3_model(
        data,
        scenarios,
        objective_kind="max_cvar",
        cvar_floor_yuan=candidate.empirical_cvar_yuan - ECONOMIC_TOLERANCE_YUAN,
    )
    relaxed_model = build_q3_lp_relaxation(source_cvar_model)
    relaxed_solution = solve_q3_model(
        relaxed_model,
        time_limit_seconds=args.lp_time_limit,
        mip_relative_gap=0.0,
        display_solver_log=args.solver_log,
    )
    lp_audit = audit_q2_lp_relaxation(source_cvar_model, relaxed_solution)
    if not lp_audit["passed"]:
        raise RuntimeError(f"Q3 maximum-CVaR LP audit failed: {lp_audit}")
    cvar_upper = float(lp_audit["upper_bound_validated_yuan"])
    cvar_value = candidate.empirical_cvar_yuan
    cvar_gap = (cvar_upper - cvar_value) / abs(cvar_value)
    passed = bool(
        expected_gap <= CERTIFICATE_TOLERANCE
        and cvar_gap <= CERTIFICATE_TOLERANCE
        and expected_gap >= -1e-10
        and cvar_gap >= -1e-10
    )

    lp_profit_path = OUTPUT_DIR / "q3_max_cvar_lp_scenario_profits.csv"
    pd.DataFrame(
        {
            "scenario": scenarios.scenario_ids,
            "probability": scenarios.probabilities,
            "profit_yuan": relaxed_solution.scenario_profits,
        }
    ).to_csv(lp_profit_path, index=False, encoding="utf-8-sig", float_format="%.15g")
    final_plan_path = OUTPUT_DIR / "q3_final_plan.csv"
    shutil.copy2(PROJECT_ROOT / files["plan"], final_plan_path)
    freeze = require_freezes()
    payload = {
        "status": "PASS" if passed else "BLOCKED",
        "classification": (
            "SOLVER-CERTIFIED 0.5%-NEAR-UTOPIA BICRITERIA"
            if passed
            else "Q3 BICRITERIA CERTIFICATE FAILED"
        ),
        "optimality_caveat": (
            "The certificate bounds distance to both single-objective ideals; it "
            "does not prove exact Pareto optimality or exact integer maximum CVaR."
        ),
        "scenario": {
            "count": scenarios.count,
            "seed": scenarios.seed,
            "sampling_method": scenarios.sampling_method,
            "fingerprint_sha256": scenario_fingerprint(scenarios.unit_draws),
        },
        "candidate": {
            "expected_profit_yuan": expected_value,
            "cvar_90_yuan": cvar_value,
            "files": files,
            "final_plan": str(final_plan_path.relative_to(PROJECT_ROOT)),
            "final_plan_sha256": sha256(final_plan_path),
        },
        "expected_profit_ideal": {
            "solver_upper_bound_yuan": expected_upper,
            "relative_gap": expected_gap,
            "threshold": CERTIFICATE_TOLERANCE,
            "passed": expected_gap <= CERTIFICATE_TOLERANCE,
        },
        "maximum_cvar_ideal": {
            "lp_relaxation_upper_bound_yuan": cvar_upper,
            "relative_gap": cvar_gap,
            "threshold": CERTIFICATE_TOLERANCE,
            "passed": cvar_gap <= CERTIFICATE_TOLERANCE,
            "lp_audit": lp_audit,
            "scenario_profits": str(lp_profit_path.relative_to(PROJECT_ROOT)),
            "scenario_profits_sha256": sha256(lp_profit_path),
        },
        "q2_freeze": freeze,
    }
    write_json(OUTPUT_DIR / "q3_near_utopia_certificate.json", payload)
    if not passed:
        raise RuntimeError(
            f"Q3 certificate failed: expected_gap={expected_gap}, cvar_gap={cvar_gap}"
        )
    return payload


def _downside(profits: np.ndarray, q2_reference: float) -> dict[str, float]:
    return {
        "probability_profit_below_zero": float(np.mean(profits < 0.0)),
        "probability_below_q2_in_sample_expected_profit": float(
            np.mean(profits < q2_reference)
        ),
    }


def run_evaluation() -> dict[str, object]:
    certificate = json.loads(
        (OUTPUT_DIR / "q3_near_utopia_certificate.json").read_text(encoding="utf-8")
    )
    if certificate["status"] != "PASS":
        raise RuntimeError("Q3 certificate is not PASS")
    freeze_before = require_freezes()
    data = load_q1_data(PROJECT_ROOT)
    plans = {
        "q3_final": load_plan_csv(OUTPUT_DIR / "q3_final_plan.csv"),
        "q2_frozen": load_plan_csv(PROJECT_ROOT / "outputs/q2/q2_final_plan.csv"),
        "q1_case1_frozen": load_plan_csv(PROJECT_ROOT / "outputs/q1_case1_long.csv"),
        "q1_case2_frozen": load_plan_csv(PROJECT_ROOT / "outputs/q1_case2_long.csv"),
    }
    agriculture = {
        name: validate_agricultural_plan(data, plan) for name, plan in plans.items()
    }
    if not all(result.passed for result in agriculture.values()):
        raise RuntimeError("A fixed comparison plan failed agricultural validation")
    baseline_paths = generate_q3_scenarios(
        data,
        count=BASELINE_Q3_CONFIG.evaluation_paths,
        seed=BASELINE_Q3_CONFIG.evaluation_seed,
        sampling_method="mc",
    )
    baseline_audit = audit_q3_scenarios(baseline_paths)
    dependency_signs_passed = all(
        value > 0.05
        for value in baseline_audit["dependency_correlations"].values()
    )
    if not baseline_audit["passed"] or not dependency_signs_passed:
        raise RuntimeError(f"Q3 OOS dependence audit failed: {baseline_audit}")

    summary_rows = []
    path_rows = []
    baseline_evaluations = {}
    q2_reference = float(certificate["candidate"]["expected_profit_yuan"])
    for rule, discount in (("half_price_surplus", 0.5), ("waste_all_surplus", 0.0)):
        for plan_name, plan in plans.items():
            evaluation = evaluate_q3_plan(
                data, plan, baseline_paths, surplus_discount=discount
            )
            baseline_evaluations[(rule, plan_name)] = evaluation
            summary_rows.append(
                {
                    "environment": "baseline_dependence_baseline_elasticity",
                    "surplus_rule": rule,
                    "surplus_discount": discount,
                    "plan": plan_name,
                    **summarize_q3_evaluation(
                        evaluation, probabilities=baseline_paths.probabilities
                    ),
                    **_downside(evaluation.profits_yuan, q2_reference),
                    **plan_diagnostics(data, plan),
                }
            )
            for omega in baseline_paths.scenario_ids:
                path_rows.append(
                    {
                        "surplus_rule": rule,
                        "plan": plan_name,
                        "scenario": omega,
                        "profit_yuan": evaluation.profits_yuan[omega],
                        "revenue_yuan": evaluation.revenue_yuan[omega],
                        "cost_yuan": evaluation.planting_cost_yuan[omega],
                        "production_jin": evaluation.production_jin[omega],
                        "normal_sales_jin": evaluation.normal_sales_jin[omega],
                        "surplus_jin": evaluation.surplus_jin[omega],
                    }
                )

    dependence_configs = {
        "weak": WEAK_Q3_CONFIG,
        "baseline": BASELINE_Q3_CONFIG,
        "strong": STRONG_Q3_CONFIG,
    }
    elasticity_scales = {"zero": 0.0, "baseline": 1.0, "strong": 1.5}
    sensitivity_rows = []
    pathwise_surplus_checks = []
    for dependence_name, base_config in dependence_configs.items():
        for elasticity_name, scale in elasticity_scales.items():
            config = base_config.with_elasticity_scale(scale)
            paths = generate_q3_scenarios(
                data,
                count=config.evaluation_paths,
                seed=config.evaluation_seed,
                sampling_method="mc",
                config=config,
            )
            for plan_name in ("q3_final", "q2_frozen"):
                plan = plans[plan_name]
                half = evaluate_q3_plan(data, plan, paths, surplus_discount=0.5)
                waste = evaluate_q3_plan(data, plan, paths, surplus_discount=0.0)
                difference = half.profits_yuan - waste.profits_yuan
                pathwise_surplus_checks.append(
                    {
                        "dependence": dependence_name,
                        "elasticity": elasticity_name,
                        "plan": plan_name,
                        "minimum_half_minus_waste_yuan": float(np.min(difference)),
                        "passed": bool(np.all(difference >= -0.1)),
                    }
                )
                for rule, evaluation in (
                    ("half_price_surplus", half),
                    ("waste_all_surplus", waste),
                ):
                    sensitivity_rows.append(
                        {
                            "dependence_strength": dependence_name,
                            "elasticity_scale": elasticity_name,
                            "elasticity_multiplier": scale,
                            "surplus_rule": rule,
                            "plan": plan_name,
                            **summarize_q3_evaluation(
                                evaluation, probabilities=paths.probabilities
                            ),
                        }
                    )
    sensitivity_passed = all(row["passed"] for row in pathwise_surplus_checks)
    if not sensitivity_passed:
        raise RuntimeError("Q3 surplus sensitivity pathwise dominance failed")

    summary_path = OUTPUT_DIR / "q3_oos_3000_summary.csv"
    pathwise_path = OUTPUT_DIR / "q3_oos_3000_pathwise.csv"
    sensitivity_path = OUTPUT_DIR / "q3_sensitivity_summary.csv"
    pd.DataFrame(summary_rows).to_csv(
        summary_path, index=False, encoding="utf-8-sig", float_format="%.15g"
    )
    pd.DataFrame(path_rows).to_csv(
        pathwise_path, index=False, encoding="utf-8-sig", float_format="%.15g"
    )
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    sensitivity_frame.to_csv(
        sensitivity_path, index=False, encoding="utf-8-sig", float_format="%.15g"
    )
    half_summary = pd.DataFrame(summary_rows)
    half_summary = half_summary[half_summary["surplus_rule"] == "half_price_surplus"]
    mean_ranking = half_summary.sort_values("mean_profit_yuan", ascending=False)[
        "plan"
    ].tolist()
    cvar_ranking = half_summary.sort_values("cvar_90_yuan", ascending=False)[
        "plan"
    ].tolist()
    pattern = planting_pattern_difference(
        data, plans["q3_final"], plans["q2_frozen"]
    )
    freeze_after = require_freezes()
    if freeze_after != freeze_before:
        raise RuntimeError("Q2 freeze changed during Q3 OOS evaluation")
    payload = {
        "status": "PASS",
        "evaluation_design": {
            "scenario_count": baseline_paths.count,
            "seed": baseline_paths.seed,
            "sampling_method": baseline_paths.sampling_method,
            "common_random_numbers": True,
            "scenario_fingerprint_sha256": scenario_fingerprint(
                baseline_paths.unit_draws
            ),
            "optimization_performed": False,
        },
        "scenario_audit": baseline_audit,
        "dependency_signs_passed": dependency_signs_passed,
        "agricultural_validation": {
            name: result.passed for name, result in agriculture.items()
        },
        "baseline_half_price_mean_ranking": mean_ranking,
        "baseline_half_price_cvar_ranking": cvar_ranking,
        "q3_vs_q2_planting_pattern": pattern,
        "sensitivity_pathwise_surplus_checks": pathwise_surplus_checks,
        "sensitivity_passed": sensitivity_passed,
        "files": {
            "summary": str(summary_path.relative_to(PROJECT_ROOT)),
            "summary_sha256": sha256(summary_path),
            "pathwise": str(pathwise_path.relative_to(PROJECT_ROOT)),
            "pathwise_sha256": sha256(pathwise_path),
            "sensitivity": str(sensitivity_path.relative_to(PROJECT_ROOT)),
            "sensitivity_sha256": sha256(sensitivity_path),
        },
        "q2_freeze": freeze_after,
    }
    write_json(OUTPUT_DIR / "q3_oos_3000_evaluation.json", payload)
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("smoke", "certificate", "evaluation", "all"), default="all"
    )
    parser.add_argument("--smoke-time-limit", type=float, default=300.0)
    parser.add_argument("--main-time-limit", type=float, default=1800.0)
    parser.add_argument("--lp-time-limit", type=float, default=1800.0)
    parser.add_argument("--solver-log", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        result = run_smoke(args)
    elif args.stage == "certificate":
        result = run_certificate(args)
    elif args.stage == "evaluation":
        result = run_evaluation()
    else:
        run_smoke(args)
        run_certificate(args)
        result = run_evaluation()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
