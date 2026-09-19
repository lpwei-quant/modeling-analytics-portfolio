from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


MODEL_ROOT = Path(__file__).resolve().parents[1]
SRC = MODEL_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q1_data import load_q1_data  # noqa: E402
from q1_model import build_q1_model, solve_q1_model  # noqa: E402
from q2_config import DEFAULT_Q2_CONFIG  # noqa: E402
from q2_uncertainty import generate_scenarios, lower_tail_cvar  # noqa: E402
from q2_validate import evaluate_plan, load_plan_csv, validate_agricultural_plan  # noqa: E402
from q3_config import BASELINE_Q3_CONFIG  # noqa: E402
from q3_scenarios import generate_q3_scenarios  # noqa: E402
from q3_validate import evaluate_q3_plan  # noqa: E402


def summarize(values: np.ndarray, beta: float = 0.90) -> dict[str, float]:
    return {
        "mean_profit_yuan": float(np.mean(values)),
        "cvar_90_yuan": float(lower_tail_cvar(values, beta=beta)),
        "minimum_profit_yuan": float(np.min(values)),
        "maximum_profit_yuan": float(np.max(values)),
    }


def evaluate_frozen_plans(paths: int) -> dict[str, object]:
    data = load_q1_data(MODEL_ROOT)
    q2_plan = load_plan_csv(MODEL_ROOT / "results" / "q2_final_plan.csv")
    q3_plan = load_plan_csv(MODEL_ROOT / "results" / "q3_final_plan.csv")
    q2_feasibility = validate_agricultural_plan(data, q2_plan)
    q3_feasibility = validate_agricultural_plan(data, q3_plan)
    if not q2_feasibility.passed or not q3_feasibility.passed:
        raise RuntimeError("Frozen plan failed an agricultural feasibility check")

    q2_paths = generate_scenarios(
        data,
        count=paths,
        seed=DEFAULT_Q2_CONFIG.evaluation_seed,
        sampling_method=DEFAULT_Q2_CONFIG.evaluation_sampling,
        config=DEFAULT_Q2_CONFIG,
    )
    q2_eval = evaluate_plan(
        data,
        q2_plan,
        q2_paths,
        surplus_discount=DEFAULT_Q2_CONFIG.surplus_discount,
        config=DEFAULT_Q2_CONFIG,
    )

    q3_paths = generate_q3_scenarios(
        data,
        count=paths,
        seed=BASELINE_Q3_CONFIG.evaluation_seed,
        sampling_method="mc",
        config=BASELINE_Q3_CONFIG,
    )
    q2_in_q3 = evaluate_q3_plan(data, q2_plan, q3_paths, surplus_discount=0.5)
    q3_in_q3 = evaluate_q3_plan(data, q3_plan, q3_paths, surplus_discount=0.5)
    paired = q2_in_q3.profits_yuan - q3_in_q3.profits_yuan

    return {
        "paths": paths,
        "q2_environment": {
            "seed": DEFAULT_Q2_CONFIG.evaluation_seed,
            "sampling": DEFAULT_Q2_CONFIG.evaluation_sampling,
            "q2_plan": summarize(q2_eval.profits_yuan),
        },
        "q3_correlated_environment": {
            "seed": BASELINE_Q3_CONFIG.evaluation_seed,
            "sampling": "mc",
            "q2_plan": summarize(q2_in_q3.profits_yuan),
            "q3_plan": summarize(q3_in_q3.profits_yuan),
            "q2_better_path_count": int(np.sum(paired > 0)),
            "paired_mean_difference_yuan": float(np.mean(paired)),
        },
    }


def solve_q1() -> dict[str, object]:
    data = load_q1_data(MODEL_ROOT)
    output: dict[str, object] = {}
    for name, discount in (("case1_waste", 0.0), ("case2_half_price", 0.5)):
        model = build_q1_model(data, case_name=name, excess_price_factor=discount)
        solution = solve_q1_model(model, time_limit_seconds=300.0, mip_relative_gap=0.005)
        if not bool(solution.result.success) or not math.isfinite(solution.total_profit_yuan):
            raise RuntimeError(f"Q1 solve failed: {name}")
        output[name] = {
            "profit_yuan": solution.total_profit_yuan,
            "mip_gap": float(getattr(solution.result, "mip_gap", math.nan)),
            "solver_status": int(solution.result.status),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce the paper's core model checks from the anonymous support package.")
    parser.add_argument("--paths", type=int, default=3000, help="Evaluation paths; use 30 for a quick smoke run.")
    parser.add_argument("--solve-q1", action="store_true", help="Also rerun the two Q1 MILPs.")
    parser.add_argument("--output", type=Path, default=MODEL_ROOT / "reproduction_summary.json")
    args = parser.parse_args()
    if args.paths <= 0:
        raise SystemExit("--paths must be positive")
    result: dict[str, object] = {"frozen_plan_evaluation": evaluate_frozen_plans(args.paths)}
    if args.solve_q1:
        result["q1_resolve"] = solve_q1()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "paths": args.paths}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
