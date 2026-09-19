#!/usr/bin/env python3
"""Fail-fast staged Q2 stochastic optimization and independent evaluation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from q1_data import IRRIGATED_LAND, SEASON_FIRST, SEASON_SECOND, SEASON_SINGLE, load_q1_data  # noqa: E402
from q1_model import PlantKey  # noqa: E402
from q2_config import DEFAULT_Q2_CONFIG  # noqa: E402
from q2_model import (  # noqa: E402
    Q2Solution,
    build_lexicographic_stage2_model,
    build_q2_model,
    solve_q2_model,
)
from q2_outputs import (  # noqa: E402
    scenario_fingerprint,
    sha256,
    write_candidate_outputs,
    write_json,
    write_plan_csv,
)
from q2_uncertainty import generate_scenarios, lower_tail_cvar  # noqa: E402
from q2_validate import (  # noqa: E402
    assess_frontier_endpoint_consistency,
    compare_approximate_endpoints,
    evaluate_plan,
    load_plan_csv,
    plan_diagnostics,
    summarize_evaluation,
    validate_agricultural_plan,
    validate_solution_economics,
    verify_q1_freeze,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "q2"


def require_freeze() -> dict[str, object]:
    result = verify_q1_freeze(PROJECT_ROOT)
    if not result["passed"]:
        raise RuntimeError(f"Q1 freeze hash failure: {result['failures']}")
    return result


def _saved_candidate_paths(candidate_id: str) -> dict[str, str]:
    """Reconstruct the write_candidate_outputs path contract from disk."""
    plan_path = OUTPUT_DIR / f"{candidate_id}_plan.csv"
    profits_path = OUTPUT_DIR / f"{candidate_id}_scenario_profits.csv"
    receipt_path = OUTPUT_DIR / f"{candidate_id}_receipt.json"
    missing = [
        str(path) for path in (plan_path, profits_path, receipt_path) if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"Saved candidate artifacts are incomplete: {missing}")
    return {
        "plan": str(plan_path),
        "scenario_profits": str(profits_path),
        "receipt": str(receipt_path),
        "plan_sha256": sha256(plan_path),
        "scenario_profits_sha256": sha256(profits_path),
        "receipt_sha256": sha256(receipt_path),
    }


def solve_and_record(
    data,
    scenarios,
    *,
    candidate_id: str,
    objective_kind: str,
    cvar_floor_yuan: float | None,
    expected_profit_floor_yuan: float | None = None,
    time_limit_seconds: float,
    mip_relative_gap: float,
    display_solver_log: bool,
) -> tuple[Q2Solution, dict[str, str]]:
    freeze_before = require_freeze()
    model = build_q2_model(
        data,
        scenarios,
        objective_kind=objective_kind,
        cvar_floor_yuan=cvar_floor_yuan,
        expected_profit_floor_yuan=expected_profit_floor_yuan,
    )
    solution = solve_q2_model(
        model,
        time_limit_seconds=time_limit_seconds,
        mip_relative_gap=mip_relative_gap,
        display_solver_log=display_solver_log,
    )
    agricultural = validate_agricultural_plan(data, solution.x)
    if not agricultural.passed:
        raise RuntimeError(
            f"{candidate_id} agricultural validation failed: "
            f"{[f for f in agricultural.findings if not f.passed]}"
        )
    economic = validate_solution_economics(solution)
    if not economic.passed:
        raise RuntimeError(
            f"{candidate_id} economic validation failed: "
            f"{[f for f in economic.findings if not f.passed]}"
        )
    freeze_after = require_freeze()
    if freeze_before != freeze_after:
        raise RuntimeError("Q1 freeze result changed during Q2 solve")
    paths = write_candidate_outputs(
        OUTPUT_DIR,
        solution,
        candidate_id=candidate_id,
        agricultural=agricultural,
        economic=economic,
        q1_freeze=freeze_after,
    )
    return solution, paths


def solve_lexicographic_endpoint(
    data,
    scenarios,
    *,
    candidate_id: str,
    time_limit_seconds: float,
    mip_relative_gap: float,
    display_solver_log: bool,
    stage1_cvar_floor_yuan: float | None = None,
    stage2_expected_profit_floor_yuan: float | None = None,
) -> tuple[Q2Solution, dict[str, str], dict[str, object]]:
    """Run the approved max-CVaR incumbent then expected-profit tie-break."""

    freeze_before = require_freeze()
    stage1_model = build_q2_model(
        data,
        scenarios,
        objective_kind="max_cvar",
        cvar_floor_yuan=stage1_cvar_floor_yuan,
    )
    stage1 = solve_q2_model(
        stage1_model,
        time_limit_seconds=time_limit_seconds,
        mip_relative_gap=mip_relative_gap,
        display_solver_log=display_solver_log,
    )
    stage1_agricultural = validate_agricultural_plan(data, stage1.x)
    if not stage1_agricultural.passed:
        raise RuntimeError(
            f"{candidate_id} Stage 1 agricultural validation failed: "
            f"{[f for f in stage1_agricultural.findings if not f.passed]}"
        )
    if stage1.auxiliary_cvar_yuan is None:
        raise RuntimeError(f"{candidate_id} Stage 1 did not expose CVaR auxiliaries")
    stage1_cvar_difference = abs(
        stage1.auxiliary_cvar_yuan - stage1.empirical_cvar_yuan
    )
    if stage1_cvar_difference > 0.1:
        raise RuntimeError(
            f"{candidate_id} Stage 1 CVaR mismatch: {stage1_cvar_difference} yuan"
        )
    incumbent_cvar = float(stage1.auxiliary_cvar_yuan)

    stage1_economic_audit = validate_solution_economics(stage1)
    stage1_independent_profits = (
        stage1_economic_audit.independent_evaluation.profits_yuan
    )
    stage1_independent_expected = float(
        np.dot(scenarios.probabilities, stage1_independent_profits)
    )
    stage1_independent_cvar = lower_tail_cvar(
        stage1_independent_profits,
        DEFAULT_Q2_CONFIG.beta,
        scenarios.probabilities,
    )
    if stage1_independent_cvar + 0.1 < incumbent_cvar:
        raise RuntimeError(
            f"{candidate_id} Stage 1 fixed-plan economics does not preserve its "
            f"CVaR incumbent: independent={stage1_independent_cvar}, "
            f"incumbent={incumbent_cvar}"
        )
    stage1_id = f"{candidate_id}_stage1"
    stage1_plan_path = OUTPUT_DIR / f"{stage1_id}_plan.csv"
    stage1_profits_path = OUTPUT_DIR / f"{stage1_id}_scenario_profits.csv"
    stage1_receipt_path = OUTPUT_DIR / f"{stage1_id}_receipt.json"
    write_plan_csv(
        stage1_plan_path,
        data,
        stage1.x,
        candidate_id=stage1_id,
        z=stage1.z,
        regimes=stage1.regimes,
    )
    pd.DataFrame(
        {
            "scenario": scenarios.scenario_ids,
            "probability": scenarios.probabilities,
            "solver_profit_yuan": stage1.scenario_profits,
            "independent_profit_for_same_plan_yuan": (
                stage1_economic_audit.independent_evaluation.profits_yuan
            ),
        }
    ).to_csv(
        stage1_profits_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12f",
    )
    stage1_audit: dict[str, object] = {
        "status": "stage1_feasible_incumbent_not_operational_endpoint",
        "candidate_id": stage1_id,
        "terminology": (
            "Stage 1 obtains a feasible incumbent CVaR under the configured MIP "
            "stopping tolerance; it is not described as a zero-gap global maximum."
        ),
        "incumbent_cvar_yuan": incumbent_cvar,
        "empirical_cvar_from_solver_profit_vector_yuan": stage1.empirical_cvar_yuan,
        "auxiliary_empirical_cvar_abs_difference_yuan": stage1_cvar_difference,
        "solver_reported_expected_profit_yuan": stage1.expected_profit_yuan,
        "independently_recomputed_expected_profit_yuan": stage1_independent_expected,
        "independently_recomputed_cvar_yuan": stage1_independent_cvar,
        "non_tail_degeneracy_audit": {
            "full_economic_reconciliation_passed": stage1_economic_audit.passed,
            "findings": [
                {
                    "name": finding.name,
                    "passed": finding.passed,
                    "max_violation": finding.max_violation,
                    "details": finding.details,
                }
                for finding in stage1_economic_audit.findings
            ],
            "interpretation": (
                "A Stage-1 full-profit mismatch is retained as audit evidence and "
                "must be eliminated by the operational Stage-2 endpoint."
            ),
        },
        "agricultural_validation_passed": stage1_agricultural.passed,
        "solver": {
            "status_code": int(stage1.result.status),
            "success": bool(stage1.result.success),
            "message": str(stage1.result.message),
            "mip_gap": float(getattr(stage1.result, "mip_gap", math.nan)),
            "mip_node_count": int(getattr(stage1.result, "mip_node_count", -1)),
            "solve_seconds": stage1.solve_seconds,
        },
        "files": {
            "plan": str(stage1_plan_path),
            "scenario_profits": str(stage1_profits_path),
        },
        "q1_freeze": require_freeze(),
    }
    write_json(stage1_receipt_path, stage1_audit)
    stage1_audit["receipt"] = str(stage1_receipt_path)

    safe_stage2_expected_floor = stage1_independent_expected - 0.1
    if stage2_expected_profit_floor_yuan is not None:
        safe_stage2_expected_floor = max(
            safe_stage2_expected_floor, stage2_expected_profit_floor_yuan
        )
    stage2_model = build_lexicographic_stage2_model(
        data,
        scenarios,
        incumbent_cvar_yuan=incumbent_cvar,
        expected_profit_floor_yuan=safe_stage2_expected_floor,
    )
    stage2 = solve_q2_model(
        stage2_model,
        time_limit_seconds=time_limit_seconds,
        mip_relative_gap=mip_relative_gap,
        display_solver_log=display_solver_log,
    )
    stage2_agricultural = validate_agricultural_plan(data, stage2.x)
    if not stage2_agricultural.passed:
        raise RuntimeError(
            f"{candidate_id} Stage 2 agricultural validation failed: "
            f"{[f for f in stage2_agricultural.findings if not f.passed]}"
        )
    stage2_economic = validate_solution_economics(stage2)
    if not stage2_economic.passed:
        raise RuntimeError(
            f"{candidate_id} Stage 2 economic validation failed: "
            f"{[f for f in stage2_economic.findings if not f.passed]}"
        )
    stage2_cvar_shortfall = max(0.0, incumbent_cvar - stage2.empirical_cvar_yuan)
    if stage2_cvar_shortfall > 0.1:
        raise RuntimeError(
            f"{candidate_id} Stage 2 failed to preserve incumbent CVaR: "
            f"shortfall={stage2_cvar_shortfall} yuan"
        )
    freeze_after = require_freeze()
    if freeze_before != freeze_after:
        raise RuntimeError("Q1 freeze result changed during lexicographic endpoint solve")
    paths = write_candidate_outputs(
        OUTPUT_DIR,
        stage2,
        candidate_id=candidate_id,
        agricultural=stage2_agricultural,
        economic=stage2_economic,
        q1_freeze=freeze_after,
    )
    lexicographic_audit = {
        "stage1": stage1_audit,
        "solver_acceleration_bounds": {
            "stage1_cvar_floor_yuan": stage1_cvar_floor_yuan,
            "stage2_expected_profit_floor_yuan": safe_stage2_expected_floor,
            "stage2_floor_source": (
                "Stage-1 planting plan independently evaluated with economically "
                "maximal recourse; 0.1 yuan numerical tolerance subtracted."
            ),
            "interpretation": (
                "Bounds are derived from independently validated feasible 60-scenario "
                "plans and are solver-acceleration cutoffs, not modeling assumptions."
            ),
        },
        "stage2": {
            "candidate_id": candidate_id,
            "objective": "maximize expected profit",
            "cvar_floor_yuan": incumbent_cvar,
            "expected_profit_yuan": stage2.expected_profit_yuan,
            "empirical_cvar_yuan": stage2.empirical_cvar_yuan,
            "cvar_floor_shortfall_yuan": stage2_cvar_shortfall,
            "agricultural_validation_passed": stage2_agricultural.passed,
            "economic_validation_passed": stage2_economic.passed,
            "solve_seconds": stage2.solve_seconds,
            "mip_gap": float(getattr(stage2.result, "mip_gap", math.nan)),
        },
        "total_solve_seconds": stage1.solve_seconds + stage2.solve_seconds,
        "no_arbitrary_lambda": True,
    }
    write_json(
        OUTPUT_DIR / f"{candidate_id}_lexicographic_audit.json",
        lexicographic_audit,
    )
    return stage2, paths, lexicographic_audit


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    data = load_q1_data(PROJECT_ROOT)
    scenarios = generate_scenarios(
        data,
        count=DEFAULT_Q2_CONFIG.smoke_scenarios,
        seed=DEFAULT_Q2_CONFIG.optimization_seed,
        sampling_method=DEFAULT_Q2_CONFIG.optimization_sampling,
    )
    risk_neutral, rn_paths = solve_and_record(
        data,
        scenarios,
        candidate_id="q2_smoke_risk_neutral",
        objective_kind="risk_neutral",
        cvar_floor_yuan=None,
        time_limit_seconds=args.smoke_time_limit,
        mip_relative_gap=args.mip_gap,
        display_solver_log=args.solver_log,
    )
    max_cvar, cvar_paths, cvar_lexicographic = solve_lexicographic_endpoint(
        data,
        scenarios,
        candidate_id="q2_smoke_max_cvar",
        time_limit_seconds=args.smoke_time_limit,
        mip_relative_gap=args.mip_gap,
        display_solver_log=args.solver_log,
    )
    endpoint_diagnostic = compare_approximate_endpoints(
        risk_neutral_expected_profit=risk_neutral.expected_profit_yuan,
        risk_neutral_cvar=risk_neutral.empirical_cvar_yuan,
        risk_neutral_mip_gap=float(getattr(risk_neutral.result, "mip_gap", math.nan)),
        risk_endpoint_expected_profit=max_cvar.expected_profit_yuan,
        risk_endpoint_cvar=max_cvar.empirical_cvar_yuan,
        risk_endpoint_mip_gap=float(getattr(max_cvar.result, "mip_gap", math.nan)),
    )
    if endpoint_diagnostic["blocking"]:
        raise RuntimeError(f"30-scenario exact endpoint contradiction: {endpoint_diagnostic}")
    summary = {
        "status": "ok",
        "stage": "smoke",
        "scenario_count": scenarios.count,
        "seed": scenarios.seed,
        "scenario_fingerprint_sha256": scenario_fingerprint(scenarios.unit_draws),
        "risk_neutral": {
            "expected_profit_yuan": risk_neutral.expected_profit_yuan,
            "cvar_90_yuan": risk_neutral.empirical_cvar_yuan,
            "solve_seconds": risk_neutral.solve_seconds,
            "paths": rn_paths,
        },
        "max_cvar": {
            "expected_profit_yuan": max_cvar.expected_profit_yuan,
            "cvar_90_yuan": max_cvar.empirical_cvar_yuan,
            "auxiliary_cvar_yuan": max_cvar.auxiliary_cvar_yuan,
            "solve_seconds": max_cvar.solve_seconds,
            "paths": cvar_paths,
            "lexicographic": cvar_lexicographic,
        },
        "approximate_endpoint_ordering_diagnostic": endpoint_diagnostic,
        "q1_freeze": require_freeze(),
    }
    write_json(OUTPUT_DIR / "q2_smoke_summary.json", summary)
    return summary


def run_smoke_finalize(_args: argparse.Namespace) -> dict[str, object]:
    """Recover a completed smoke solve through fresh artifact-only validation."""

    freeze_before = require_freeze()
    data = load_q1_data(PROJECT_ROOT)
    scenarios = generate_scenarios(
        data,
        count=DEFAULT_Q2_CONFIG.smoke_scenarios,
        seed=DEFAULT_Q2_CONFIG.optimization_seed,
        sampling_method=DEFAULT_Q2_CONFIG.optimization_sampling,
    )
    candidate_ids = ("q2_smoke_risk_neutral", "q2_smoke_max_cvar")
    validated: dict[str, dict[str, object]] = {}
    for candidate_id in candidate_ids:
        plan_path = OUTPUT_DIR / f"{candidate_id}_plan.csv"
        profits_path = OUTPUT_DIR / f"{candidate_id}_scenario_profits.csv"
        receipt_path = OUTPUT_DIR / f"{candidate_id}_receipt.json"
        if not all(path.is_file() for path in (plan_path, profits_path, receipt_path)):
            raise RuntimeError(f"Saved smoke artifacts are incomplete for {candidate_id}")
        plan = load_plan_csv(plan_path)
        agricultural = validate_agricultural_plan(data, plan)
        if not agricultural.passed:
            raise RuntimeError(f"Saved {candidate_id} plan failed agricultural validation")
        independent = evaluate_plan(
            data,
            plan,
            scenarios,
            surplus_discount=DEFAULT_Q2_CONFIG.surplus_discount,
        )
        saved_profits = pd.read_csv(profits_path, encoding="utf-8-sig")[
            "profit_yuan"
        ].to_numpy(dtype=float)
        if saved_profits.shape != independent.profits_yuan.shape:
            raise RuntimeError(f"Saved {candidate_id} profit vector has wrong shape")
        maximum_profit_difference = float(
            np.max(np.abs(saved_profits - independent.profits_yuan), initial=0.0)
        )
        if maximum_profit_difference > 0.1:
            raise RuntimeError(
                f"Saved {candidate_id} profit reconciliation failed: "
                f"{maximum_profit_difference} yuan"
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_profit = float(np.dot(scenarios.probabilities, saved_profits))
        cvar = lower_tail_cvar(
            saved_profits,
            DEFAULT_Q2_CONFIG.beta,
            scenarios.probabilities,
        )
        expected_difference = abs(expected_profit - float(receipt["expected_profit_yuan"]))
        cvar_difference = abs(cvar - float(receipt["empirical_cvar_90_yuan"]))
        if max(expected_difference, cvar_difference) > 0.1:
            raise RuntimeError(
                f"Saved {candidate_id} receipt reconciliation failed: "
                f"expected_diff={expected_difference}, cvar_diff={cvar_difference}"
            )
        validated[candidate_id] = {
            "expected_profit_yuan": expected_profit,
            "cvar_90_yuan": cvar,
            "maximum_scenario_profit_abs_difference_yuan": maximum_profit_difference,
            "expected_profit_abs_difference_yuan": expected_difference,
            "cvar_abs_difference_yuan": cvar_difference,
            "agricultural_validation_passed": agricultural.passed,
            "solver_mip_gap": float(receipt["solver"]["mip_gap"]),
            "solver_solve_seconds": float(receipt["solver"]["solve_seconds"]),
            "receipt": str(receipt_path),
            "receipt_sha256": sha256(receipt_path),
        }

    stage1_receipt_path = OUTPUT_DIR / "q2_smoke_max_cvar_stage1_receipt.json"
    lexicographic_path = OUTPUT_DIR / "q2_smoke_max_cvar_lexicographic_audit.json"
    stage1 = json.loads(stage1_receipt_path.read_text(encoding="utf-8"))
    lexicographic = json.loads(lexicographic_path.read_text(encoding="utf-8"))
    incumbent_cvar = float(stage1["incumbent_cvar_yuan"])
    stage2_cvar = float(validated["q2_smoke_max_cvar"]["cvar_90_yuan"])
    floor_shortfall = max(0.0, incumbent_cvar - stage2_cvar)
    if floor_shortfall > 0.1:
        raise RuntimeError(
            f"Saved Stage 2 does not preserve incumbent CVaR: {floor_shortfall} yuan"
        )

    endpoint_diagnostic = compare_approximate_endpoints(
        risk_neutral_expected_profit=float(
            validated["q2_smoke_risk_neutral"]["expected_profit_yuan"]
        ),
        risk_neutral_cvar=float(
            validated["q2_smoke_risk_neutral"]["cvar_90_yuan"]
        ),
        risk_neutral_mip_gap=float(
            validated["q2_smoke_risk_neutral"]["solver_mip_gap"]
        ),
        risk_endpoint_expected_profit=float(
            validated["q2_smoke_max_cvar"]["expected_profit_yuan"]
        ),
        risk_endpoint_cvar=stage2_cvar,
        risk_endpoint_mip_gap=float(
            validated["q2_smoke_max_cvar"]["solver_mip_gap"]
        ),
    )
    if endpoint_diagnostic["blocking"]:
        raise RuntimeError(f"Saved exact endpoint contradiction: {endpoint_diagnostic}")
    freeze_after = require_freeze()
    if freeze_before != freeze_after:
        raise RuntimeError("Q1 freeze changed during smoke artifact finalization")
    summary = {
        "status": "ok",
        "stage": "smoke_artifact_revalidation",
        "recovery_reason": (
            "A later repeat solve was interrupted by the execution session after a "
            "previous complete Stage-1/Stage-2 run. Saved artifacts were accepted "
            "only after fresh scenario regeneration and independent reconciliation."
        ),
        "scenario_count": scenarios.count,
        "seed": scenarios.seed,
        "scenario_fingerprint_sha256": scenario_fingerprint(scenarios.unit_draws),
        "validated_candidates": validated,
        "stage1_incumbent_cvar_yuan": incumbent_cvar,
        "stage2_cvar_yuan": stage2_cvar,
        "stage2_cvar_floor_shortfall_yuan": floor_shortfall,
        "lexicographic_audit": lexicographic,
        "approximate_endpoint_ordering_diagnostic": endpoint_diagnostic,
        "q1_freeze": freeze_after,
    }
    write_json(OUTPUT_DIR / "q2_smoke_summary.json", summary)
    return summary


def _normalize(values: list[float]) -> list[float]:
    lower, upper = min(values), max(values)
    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-9):
        return [1.0] * len(values)
    return [(value - lower) / (upper - lower) for value in values]


def _select_knee(rows: list[dict[str, object]]) -> str:
    expected = [float(row["expected_profit_yuan"]) for row in rows]
    cvars = [float(row["cvar_90_yuan"]) for row in rows]
    normalized_expected = _normalize(expected)
    normalized_cvar = _normalize(cvars)
    for row, norm_e, norm_c in zip(rows, normalized_expected, normalized_cvar):
        row["normalized_expected_profit"] = norm_e
        row["normalized_cvar"] = norm_c
        row["distance_to_utopia"] = math.hypot(1.0 - norm_e, 1.0 - norm_c)
    best = min(
        rows,
        key=lambda row: (
            float(row["distance_to_utopia"]),
            -float(row["expected_profit_yuan"]),
            -float(row["cvar_90_yuan"]),
            str(row["candidate_id"]),
        ),
    )
    return str(best["candidate_id"])


def _validation_rows(report) -> list[dict[str, object]]:
    return [
        {
            "name": finding.name,
            "passed": finding.passed,
            "max_violation": finding.max_violation,
            "details": finding.details,
        }
        for finding in report.findings
    ]


def _independently_evaluate_saved_plan(
    data,
    scenarios,
    *,
    candidate_id: str,
    require_saved_profit_reconciliation: bool,
) -> dict[str, object]:
    plan_path = OUTPUT_DIR / f"{candidate_id}_plan.csv"
    receipt_path = OUTPUT_DIR / f"{candidate_id}_receipt.json"
    if not plan_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"Missing saved candidate evidence for {candidate_id}")
    plan = load_plan_csv(plan_path)
    agricultural = validate_agricultural_plan(data, plan)
    if not agricultural.passed:
        raise RuntimeError(f"Saved plan {candidate_id} fails agricultural validation")
    evaluation = evaluate_plan(
        data,
        plan,
        scenarios,
        surplus_discount=DEFAULT_Q2_CONFIG.surplus_discount,
    )
    expected = float(np.dot(scenarios.probabilities, evaluation.profits_yuan))
    cvar = lower_tail_cvar(
        evaluation.profits_yuan,
        DEFAULT_Q2_CONFIG.beta,
        scenarios.probabilities,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    probability_difference = math.nan
    maximum_profit_difference = math.nan
    profits_path = OUTPUT_DIR / f"{candidate_id}_scenario_profits.csv"
    if require_saved_profit_reconciliation:
        saved = pd.read_csv(profits_path, encoding="utf-8-sig")
        probability_difference = float(
            np.max(
                np.abs(saved["probability"].to_numpy(dtype=float) - scenarios.probabilities),
                initial=0.0,
            )
        )
        maximum_profit_difference = float(
            np.max(
                np.abs(saved["profit_yuan"].to_numpy(dtype=float) - evaluation.profits_yuan),
                initial=0.0,
            )
        )
        if probability_difference > 1e-12 or maximum_profit_difference > 0.1:
            raise RuntimeError(
                f"Saved candidate {candidate_id} is not economically comparable: "
                f"probability_diff={probability_difference}, "
                f"profit_diff={maximum_profit_difference}"
            )
    return {
        "candidate_id": candidate_id,
        "plan_path": str(plan_path),
        "receipt_path": str(receipt_path),
        "scenario_count": receipt.get("scenario_count", scenarios.count),
        "scenario_seed": receipt.get("scenario_seed", scenarios.seed),
        "sampling_method": receipt.get("sampling_method", scenarios.sampling_method),
        "scenario_fingerprint_sha256": receipt.get("scenario_fingerprint_sha256"),
        "config": receipt.get("config"),
        "independently_recomputed_expected_profit_yuan": expected,
        "independently_recomputed_cvar_90_yuan": cvar,
        "saved_probability_max_abs_difference": probability_difference,
        "saved_scenario_profit_max_abs_difference_yuan": maximum_profit_difference,
        "agricultural_validation_passed": agricultural.passed,
        "agricultural_findings": _validation_rows(agricultural),
    }


def run_frontier_gate(args: argparse.Namespace) -> dict[str, object]:
    """Reconcile dominated incumbents and polish both 60-scenario endpoints."""

    freeze_before = require_freeze()
    data = load_q1_data(PROJECT_ROOT)
    scenarios = generate_scenarios(
        data,
        count=DEFAULT_Q2_CONFIG.main_scenarios,
        seed=DEFAULT_Q2_CONFIG.optimization_seed,
        sampling_method=DEFAULT_Q2_CONFIG.optimization_sampling,
    )
    generated_fingerprint = scenario_fingerprint(scenarios.unit_draws)
    old_rn = _independently_evaluate_saved_plan(
        data,
        scenarios,
        candidate_id="q2_main_risk_neutral",
        require_saved_profit_reconciliation=True,
    )
    old_cvar = _independently_evaluate_saved_plan(
        data,
        scenarios,
        candidate_id="q2_main_max_cvar",
        require_saved_profit_reconciliation=True,
    )
    old_stage1 = _independently_evaluate_saved_plan(
        data,
        scenarios,
        candidate_id="q2_main_max_cvar_stage1",
        require_saved_profit_reconciliation=False,
    )
    approved_config_json = json.loads(
        json.dumps(DEFAULT_Q2_CONFIG.to_dict(), ensure_ascii=False)
    )

    comparable = bool(
        old_rn["scenario_count"] == old_cvar["scenario_count"] == scenarios.count
        and old_rn["scenario_seed"] == old_cvar["scenario_seed"] == scenarios.seed
        and old_rn["sampling_method"]
        == old_cvar["sampling_method"]
        == scenarios.sampling_method
        and old_rn["scenario_fingerprint_sha256"]
        == old_cvar["scenario_fingerprint_sha256"]
        == generated_fingerprint
        and old_rn["config"] == old_cvar["config"] == approved_config_json
        and old_rn["saved_probability_max_abs_difference"] <= 1e-12
        and old_cvar["saved_probability_max_abs_difference"] <= 1e-12
        and old_rn["agricultural_validation_passed"]
        and old_cvar["agricultural_validation_passed"]
    )
    if not comparable:
        payload = {
            "status": "blocked_not_comparable",
            "generated_scenario_fingerprint_sha256": generated_fingerprint,
            "risk_neutral": old_rn,
            "max_cvar_operational": old_cvar,
            "q1_freeze": require_freeze(),
        }
        write_json(OUTPUT_DIR / "q2_frontier_consistency_precheck.json", payload)
        raise RuntimeError("The two saved 60-scenario endpoints are not comparable")

    initial_dominance = assess_frontier_endpoint_consistency(
        expected_endpoint_expected_profit=float(
            old_rn["independently_recomputed_expected_profit_yuan"]
        ),
        expected_endpoint_cvar=float(old_rn["independently_recomputed_cvar_90_yuan"]),
        cvar_endpoint_expected_profit=float(
            old_cvar["independently_recomputed_expected_profit_yuan"]
        ),
        cvar_endpoint_cvar=float(old_cvar["independently_recomputed_cvar_90_yuan"]),
    )
    existing = (old_rn, old_cvar, old_stage1)
    best_expected = max(
        float(row["independently_recomputed_expected_profit_yuan"])
        for row in existing
    )
    best_cvar = max(
        float(row["independently_recomputed_cvar_90_yuan"])
        for row in existing
    )
    precheck = {
        "status": "comparable",
        "same_60_scenario_realizations": True,
        "same_equal_weights": True,
        "same_seed_and_lhs_dynamics": True,
        "same_surplus_discount_rule": True,
        "same_independent_economic_evaluator": True,
        "same_agricultural_feasible_set": True,
        "same_here_and_now_planting_structure": True,
        "generated_scenario_fingerprint_sha256": generated_fingerprint,
        "old_endpoints": {"risk_neutral": old_rn, "max_cvar_operational": old_cvar},
        "old_stage1_feasible_plan": old_stage1,
        "initial_dominance_test": initial_dominance,
        "old_risk_neutral_classification": (
            "VALID FEASIBLE INCUMBENT, BUT DOMINATED / NOT AUTHORITATIVE FRONTIER ENDPOINT"
            if initial_dominance["cvar_endpoint_strictly_dominates_expected_endpoint"]
            else "VALID FEASIBLE INCUMBENT"
        ),
        "best_verified_expected_profit_yuan": best_expected,
        "best_verified_empirical_cvar_yuan": best_cvar,
        "cutoff_interpretation": (
            "Each cutoff is derived from an independently validated feasible plan; "
            "it is a solver-acceleration bound, not a modeling assumption."
        ),
        "q1_freeze": require_freeze(),
    }
    write_json(OUTPUT_DIR / "q2_frontier_consistency_precheck.json", precheck)

    rn_id = "q2_main_expected_endpoint_polished"
    rn_receipt_path = OUTPUT_DIR / f"{rn_id}_receipt.json"
    if rn_receipt_path.is_file():
        rn_record = _independently_evaluate_saved_plan(
            data,
            scenarios,
            candidate_id=rn_id,
            require_saved_profit_reconciliation=True,
        )
        rn_receipt = json.loads(rn_receipt_path.read_text(encoding="utf-8"))
        rn_expected = float(rn_record["independently_recomputed_expected_profit_yuan"])
        rn_cvar = float(rn_record["independently_recomputed_cvar_90_yuan"])
        rn_paths = _saved_candidate_paths(rn_id)
        rn_solver = rn_receipt["solver"]
        rn_reused = True
    else:
        rn_solution, rn_paths = solve_and_record(
            data,
            scenarios,
            candidate_id=rn_id,
            objective_kind="risk_neutral",
            cvar_floor_yuan=None,
            expected_profit_floor_yuan=best_expected - 0.1,
            time_limit_seconds=args.polish_time_limit,
            mip_relative_gap=args.mip_gap,
            display_solver_log=args.solver_log,
        )
        rn_expected = rn_solution.expected_profit_yuan
        rn_cvar = rn_solution.empirical_cvar_yuan
        rn_solver = {
            "mip_gap": float(getattr(rn_solution.result, "mip_gap", math.nan)),
            "mip_dual_bound": float(
                getattr(rn_solution.result, "mip_dual_bound", math.nan)
            ),
            "mip_node_count": int(getattr(rn_solution.result, "mip_node_count", -1)),
            "solve_seconds": rn_solution.solve_seconds,
        }
        rn_reused = False
    current_best_cvar = max(best_cvar, rn_cvar)
    precheck["best_verified_empirical_cvar_after_expected_endpoint_polish_yuan"] = (
        current_best_cvar
    )
    precheck["expected_endpoint_reused_from_validated_artifact"] = rn_reused
    write_json(OUTPUT_DIR / "q2_frontier_consistency_precheck.json", precheck)

    cvar_id = "q2_main_cvar_endpoint_polished"
    cvar_solution, cvar_paths, lexicographic = solve_lexicographic_endpoint(
        data,
        scenarios,
        candidate_id=cvar_id,
        stage1_cvar_floor_yuan=current_best_cvar - 0.1,
        time_limit_seconds=args.polish_time_limit,
        mip_relative_gap=args.mip_gap,
        display_solver_log=args.solver_log,
    )
    final_consistency = assess_frontier_endpoint_consistency(
        expected_endpoint_expected_profit=rn_expected,
        expected_endpoint_cvar=rn_cvar,
        cvar_endpoint_expected_profit=cvar_solution.expected_profit_yuan,
        cvar_endpoint_cvar=cvar_solution.empirical_cvar_yuan,
    )
    summary = {
        "status": "pass" if final_consistency["passed"] else "blocked_dominance",
        "terminology": (
            "Approximate solver-certified frontier endpoints under mip_rel_gap=0.005; "
            "no zero-gap global-optimality claim is made."
        ),
        "scenario_count": scenarios.count,
        "scenario_seed": scenarios.seed,
        "scenario_fingerprint_sha256": generated_fingerprint,
        "precheck": str(OUTPUT_DIR / "q2_frontier_consistency_precheck.json"),
        "expected_profit_endpoint": {
            "candidate_id": rn_id,
            "expected_profit_yuan": rn_expected,
            "cvar_90_yuan": rn_cvar,
            "mip_gap": float(rn_solver["mip_gap"]),
            "mip_dual_bound": float(rn_solver["mip_dual_bound"]),
            "mip_node_count": int(rn_solver["mip_node_count"]),
            "solve_seconds": float(rn_solver["solve_seconds"]),
            "objective_cutoff_yuan": best_expected - 0.1,
            "reused_from_validated_artifact": rn_reused,
            "files": rn_paths,
        },
        "cvar_endpoint": {
            "candidate_id": cvar_id,
            "expected_profit_yuan": cvar_solution.expected_profit_yuan,
            "cvar_90_yuan": cvar_solution.empirical_cvar_yuan,
            "mip_gap": float(getattr(cvar_solution.result, "mip_gap", math.nan)),
            "mip_dual_bound": float(
                getattr(cvar_solution.result, "mip_dual_bound", math.nan)
            ),
            "mip_node_count": int(
                getattr(cvar_solution.result, "mip_node_count", -1)
            ),
            "solve_seconds": cvar_solution.solve_seconds,
            "stage1_cvar_cutoff_yuan": current_best_cvar - 0.1,
            "files": cvar_paths,
            "lexicographic": lexicographic,
        },
        "final_dominance_consistency": final_consistency,
        "q1_freeze": require_freeze(),
    }
    if freeze_before != summary["q1_freeze"]:
        raise RuntimeError("Q1 freeze changed during frontier consistency gate")
    write_json(OUTPUT_DIR / "q2_frontier_consistency_gate.json", summary)
    if not final_consistency["passed"]:
        raise RuntimeError(f"Polished endpoints remain dominated: {final_consistency}")
    return summary


def run_main(args: argparse.Namespace) -> dict[str, object]:
    data = load_q1_data(PROJECT_ROOT)
    scenarios = generate_scenarios(
        data,
        count=DEFAULT_Q2_CONFIG.main_scenarios,
        seed=DEFAULT_Q2_CONFIG.optimization_seed,
        sampling_method=DEFAULT_Q2_CONFIG.optimization_sampling,
    )
    solutions: dict[str, Q2Solution] = {}
    output_paths: dict[str, dict[str, str]] = {}

    rn_id = "q2_main_risk_neutral"
    solutions[rn_id], output_paths[rn_id] = solve_and_record(
        data,
        scenarios,
        candidate_id=rn_id,
        objective_kind="risk_neutral",
        cvar_floor_yuan=None,
        time_limit_seconds=args.main_time_limit,
        mip_relative_gap=args.mip_gap,
        display_solver_log=args.solver_log,
    )
    cvar_id = "q2_main_max_cvar"
    solutions[cvar_id], output_paths[cvar_id], cvar_lexicographic = solve_lexicographic_endpoint(
        data,
        scenarios,
        candidate_id=cvar_id,
        time_limit_seconds=args.main_time_limit,
        mip_relative_gap=args.mip_gap,
        display_solver_log=args.solver_log,
    )

    cvar_rn = solutions[rn_id].empirical_cvar_yuan
    cvar_max = solutions[cvar_id].empirical_cvar_yuan
    endpoint_diagnostic = compare_approximate_endpoints(
        risk_neutral_expected_profit=solutions[rn_id].expected_profit_yuan,
        risk_neutral_cvar=cvar_rn,
        risk_neutral_mip_gap=float(
            getattr(solutions[rn_id].result, "mip_gap", math.nan)
        ),
        risk_endpoint_expected_profit=solutions[cvar_id].expected_profit_yuan,
        risk_endpoint_cvar=cvar_max,
        risk_endpoint_mip_gap=float(
            getattr(solutions[cvar_id].result, "mip_gap", math.nan)
        ),
    )
    if endpoint_diagnostic["blocking"]:
        raise RuntimeError(f"60-scenario exact endpoint contradiction: {endpoint_diagnostic}")

    candidate_order = [rn_id]
    for theta in DEFAULT_Q2_CONFIG.frontier_theta:
        candidate_id = f"q2_main_theta_{int(round(theta * 100)):03d}"
        target = cvar_rn + theta * (cvar_max - cvar_rn)
        solutions[candidate_id], output_paths[candidate_id] = solve_and_record(
            data,
            scenarios,
            candidate_id=candidate_id,
            objective_kind="expected_with_cvar_floor",
            cvar_floor_yuan=target,
            time_limit_seconds=args.main_time_limit,
            mip_relative_gap=args.mip_gap,
            display_solver_log=args.solver_log,
        )
        candidate_order.append(candidate_id)
    candidate_order.append(cvar_id)

    rows: list[dict[str, object]] = []
    for candidate_id in candidate_order:
        solution = solutions[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "objective_kind": solution.model.objective_kind,
                "cvar_floor_yuan": solution.model.cvar_floor_yuan,
                "expected_profit_yuan": solution.expected_profit_yuan,
                "cvar_90_yuan": solution.empirical_cvar_yuan,
                "solve_seconds": solution.solve_seconds,
                "mip_gap": float(getattr(solution.result, "mip_gap", math.nan)),
                "plan_path": output_paths[candidate_id]["plan"],
            }
        )
    selected_id = _select_knee(rows)
    for row in rows:
        row["selected_knee"] = str(row["candidate_id"] == selected_id).lower()
    frontier_path = OUTPUT_DIR / "q2_frontier.csv"
    pd.DataFrame(rows).to_csv(
        frontier_path, index=False, encoding="utf-8-sig", float_format="%.12f"
    )
    selected_solution = solutions[selected_id]
    selected_plan_path = OUTPUT_DIR / "q2_selected_plan.csv"
    write_plan_csv(
        selected_plan_path,
        data,
        selected_solution.x,
        candidate_id=selected_id,
        z=selected_solution.z,
        regimes=selected_solution.regimes,
    )
    summary = {
        "status": "ok",
        "stage": "main_frontier",
        "scenario_count": scenarios.count,
        "seed": scenarios.seed,
        "scenario_fingerprint_sha256": scenario_fingerprint(scenarios.unit_draws),
        "cvar_risk_neutral_yuan": cvar_rn,
        "cvar_max_yuan": cvar_max,
        "max_cvar_lexicographic": cvar_lexicographic,
        "approximate_endpoint_ordering_diagnostic": endpoint_diagnostic,
        "selected_knee_candidate_id": selected_id,
        "selected_plan": str(selected_plan_path),
        "selected_plan_sha256": sha256(selected_plan_path),
        "frontier": rows,
        "frontier_csv": str(frontier_path),
        "frontier_csv_sha256": sha256(frontier_path),
        "q1_freeze": require_freeze(),
    }
    write_json(OUTPUT_DIR / "q2_main_summary.json", summary)
    return summary


def _area_l1(first: dict[PlantKey, float], second: dict[PlantKey, float]) -> float:
    return math.fsum(abs(first.get(key, 0.0) - second.get(key, 0.0)) for key in set(first) | set(second))


def _infer_irrigated_regime(data, plan: dict[PlantKey, float], plot: str, year: int) -> str:
    rice = math.fsum(
        plan.get((plot, crop_id, year, SEASON_SINGLE), 0.0)
        for crop_id in data.compatible_crops(plot, SEASON_SINGLE)
    )
    vegetables = math.fsum(
        plan.get((plot, crop_id, year, season), 0.0)
        for season in (SEASON_FIRST, SEASON_SECOND)
        for crop_id in data.compatible_crops(plot, season)
    )
    if rice > 1e-6:
        return "rice_single"
    if vegetables > 1e-6:
        return "two_season_vegetables"
    return "idle"


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    freeze = require_freeze()
    data = load_q1_data(PROJECT_ROOT)
    frontier_path = OUTPUT_DIR / "q2_frontier.csv"
    if not frontier_path.is_file():
        raise RuntimeError("Main frontier is missing; run --stage main first")
    frontier = pd.read_csv(frontier_path, encoding="utf-8-sig")
    plans: dict[str, dict[PlantKey, float]] = {}
    for row in frontier.itertuples(index=False):
        plans[str(row.candidate_id)] = load_plan_csv(Path(str(row.plan_path)))
    q1_case1 = load_plan_csv(PROJECT_ROOT / "outputs" / "q1_case1_long.csv")
    q1_case2 = load_plan_csv(PROJECT_ROOT / "outputs" / "q1_case2_long.csv")
    plans["frozen_q1_case1"] = q1_case1
    plans["frozen_q1_case2"] = q1_case2

    scenarios = generate_scenarios(
        data,
        count=DEFAULT_Q2_CONFIG.evaluation_paths,
        seed=DEFAULT_Q2_CONFIG.evaluation_seed,
        sampling_method=DEFAULT_Q2_CONFIG.evaluation_sampling,
    )
    fingerprint = scenario_fingerprint(scenarios.unit_draws)
    q1_benchmarks = {
        "case1": 41044527.130780034,
        "case2": 63909448.53304118,
    }
    comparison_rows: list[dict[str, object]] = []
    evaluations = {}
    for candidate_id, plan in plans.items():
        agricultural = validate_agricultural_plan(data, plan)
        if not agricultural.passed:
            raise RuntimeError(f"{candidate_id} failed agricultural evaluation gate")
        evaluation = evaluate_plan(
            data,
            plan,
            scenarios,
            surplus_discount=DEFAULT_Q2_CONFIG.surplus_discount,
        )
        evaluations[candidate_id] = evaluation
        row: dict[str, object] = {"candidate_id": candidate_id}
        row.update(summarize_evaluation(evaluation, probabilities=scenarios.probabilities))
        row.update(plan_diagnostics(data, plan))
        row["probability_below_frozen_q1_case1_deterministic_profit"] = float(
            np.mean(evaluation.profits_yuan < q1_benchmarks["case1"])
        )
        row["probability_below_frozen_q1_case2_deterministic_profit"] = float(
            np.mean(evaluation.profits_yuan < q1_benchmarks["case2"])
        )
        row["area_l1_vs_q1_case1_mu"] = _area_l1(plan, q1_case1)
        row["area_l1_vs_q1_case2_mu"] = _area_l1(plan, q1_case2)
        row["evaluation_seed"] = scenarios.seed
        row["common_path_fingerprint_sha256"] = fingerprint
        comparison_rows.append(row)

    comparison_path = OUTPUT_DIR / "q2_oos_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(
        comparison_path, index=False, encoding="utf-8-sig", float_format="%.12f"
    )

    main_summary = json.loads(
        (OUTPUT_DIR / "q2_main_summary.json").read_text(encoding="utf-8")
    )
    selected_id = str(main_summary["selected_knee_candidate_id"])
    selected_plan = plans[selected_id]
    waste = evaluate_plan(
        data, selected_plan, scenarios, surplus_discount=0.0
    )
    waste_summary = summarize_evaluation(waste, probabilities=scenarios.probabilities)
    half_summary = summarize_evaluation(
        evaluations[selected_id], probabilities=scenarios.probabilities
    )
    if np.any(waste.profits_yuan > evaluations[selected_id].profits_yuan + 1e-6):
        raise RuntimeError("Waste sensitivity exceeds half-price profit on a common path")
    waste_payload = {
        "candidate_id": selected_id,
        "common_path_fingerprint_sha256": fingerprint,
        "half_price": half_summary,
        "waste_all_surplus": waste_summary,
        "mean_profit_difference_yuan": (
            half_summary["mean_profit_yuan"] - waste_summary["mean_profit_yuan"]
        ),
    }
    write_json(OUTPUT_DIR / "q2_waste_surplus_sensitivity.json", waste_payload)

    beta_sensitivity = {
        str(beta): lower_tail_cvar(
            evaluations[selected_id].profits_yuan,
            beta,
            scenarios.probabilities,
        )
        for beta in (0.85, 0.90, 0.95)
    }
    write_json(
        OUTPUT_DIR / "q2_beta_sensitivity.json",
        {
            "candidate_id": selected_id,
            "evaluation_only_no_reoptimization": True,
            "cvar_yuan": beta_sensitivity,
        },
    )

    q2_ids = [str(value) for value in frontier["candidate_id"].tolist()]
    stability_rows: list[dict[str, object]] = []
    for key in data.compatible_cells:
        areas = [plans[candidate_id].get(key, 0.0) for candidate_id in q2_ids]
        if all(area > 1e-6 for area in areas):
            stability_rows.append(
                {
                    "kind": "planting_cell_positive_in_all_frontier_plans",
                    "plot": key[0],
                    "crop_id": key[1],
                    "year": key[2],
                    "season": key[3],
                    "state": "positive",
                    "minimum_area_mu": min(areas),
                    "maximum_area_mu": max(areas),
                }
            )
    for plot in data.irrigated_plots:
        for year in range(2024, 2031):
            states = [_infer_irrigated_regime(data, plans[candidate_id], plot, year) for candidate_id in q2_ids]
            if len(set(states)) == 1:
                stability_rows.append(
                    {
                        "kind": "irrigated_regime_same_in_all_frontier_plans",
                        "plot": plot,
                        "crop_id": "",
                        "year": year,
                        "season": "annual",
                        "state": states[0],
                        "minimum_area_mu": "",
                        "maximum_area_mu": "",
                    }
                )
    stability_path = OUTPUT_DIR / "q2_frontier_stability.csv"
    pd.DataFrame(stability_rows).to_csv(
        stability_path, index=False, encoding="utf-8-sig", float_format="%.12f"
    )

    summary = {
        "status": "ok",
        "stage": "out_of_sample_evaluation",
        "evaluation_paths": scenarios.count,
        "evaluation_seed": scenarios.seed,
        "optimization_seed": DEFAULT_Q2_CONFIG.optimization_seed,
        "seeds_differ": scenarios.seed != DEFAULT_Q2_CONFIG.optimization_seed,
        "common_path_fingerprint_sha256": fingerprint,
        "candidate_count": len(plans),
        "selected_candidate_id": selected_id,
        "comparison_csv": str(comparison_path),
        "comparison_csv_sha256": sha256(comparison_path),
        "frontier_stability_csv": str(stability_path),
        "frontier_stability_csv_sha256": sha256(stability_path),
        "stable_record_count": len(stability_rows),
        "waste_sensitivity": waste_payload,
        "beta_sensitivity_cvar_yuan": beta_sensitivity,
        "q1_freeze": require_freeze(),
    }
    if freeze != summary["q1_freeze"]:
        raise RuntimeError("Q1 freeze result changed during out-of-sample evaluation")
    write_json(OUTPUT_DIR / "q2_evaluation_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "smoke",
            "smoke_finalize",
            "frontier_gate",
            "main",
            "evaluate",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--smoke-time-limit", type=float, default=900.0)
    parser.add_argument("--main-time-limit", type=float, default=1800.0)
    parser.add_argument("--polish-time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=0.005)
    parser.add_argument("--solver-log", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stages = (
        ("smoke", "main", "evaluate") if args.stage == "all" else (args.stage,)
    )
    results: dict[str, object] = {}
    for stage in stages:
        if stage == "smoke":
            results[stage] = run_smoke(args)
        elif stage == "smoke_finalize":
            results[stage] = run_smoke_finalize(args)
        elif stage == "frontier_gate":
            results[stage] = run_frontier_gate(args)
        elif stage == "main":
            results[stage] = run_main(args)
        elif stage == "evaluate":
            results[stage] = run_evaluation(args)
    print(json.dumps({"status": "ok", "completed_stages": list(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
