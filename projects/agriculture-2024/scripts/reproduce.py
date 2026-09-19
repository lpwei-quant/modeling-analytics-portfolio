"""Reevaluate two frozen plans on the SAME newly generated 3000 Q3 paths; never optimize."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portable_data import load_portable_data
from q2_validate import load_plan_csv, validate_agricultural_plan
from q3_config import BASELINE_Q3_CONFIG
from q3_scenarios import generate_q3_scenarios
from q3_validate import evaluate_q3_plan, summarize_q3_evaluation


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".reproduction" / "verification.json")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    for entry in manifest["records"]:
        if file_hash(ROOT / entry["public_relative_path"]) != entry["public_sha256"]:
            raise RuntimeError(f"Frozen public artifact changed: {entry['public_relative_path']}")
    data = load_portable_data(ROOT)
    if len(data.plots) != 54 or len(data.crops) != 41 or not math.isclose(sum(p.area_mu for p in data.plots.values()), 1213):
        raise RuntimeError("Input dimensions differ from the study")
    plans = {
        "q2_frozen": load_plan_csv(ROOT / "data" / "q2_final_plan.csv"),
        "q3_final": load_plan_csv(ROOT / "data" / "q3_final_plan.csv"),
    }
    feasibility = {name: validate_agricultural_plan(data, plan) for name, plan in plans.items()}
    if not all(v.passed for v in feasibility.values()):
        raise RuntimeError("Frozen plans failed agricultural feasibility")
    print("Generating one shared set of 3000 Q3 correlated Monte Carlo paths (Simulated).", flush=True)
    paths = generate_q3_scenarios(
        data, count=3000, seed=BASELINE_Q3_CONFIG.evaluation_seed,
        sampling_method="mc", config=BASELINE_Q3_CONFIG,
    )
    expected_summary = pd.read_csv(ROOT / "data" / "q3_common_paths_summary.csv")
    expected_profits = pd.read_csv(ROOT / "data" / "q3_common_paths_profits.csv")
    if not np.array_equal(expected_profits["scenario"].to_numpy(), np.arange(3000)):
        raise RuntimeError("Frozen path indices are not the complete ordered common path set")
    results = {}
    evaluations = {}
    tolerance_yuan = 0.1
    for name, plan in plans.items():
        print(f"Evaluating fixed {name}; no optimizer is called.", flush=True)
        evaluation = evaluate_q3_plan(data, plan, paths, surplus_discount=0.5)
        evaluations[name] = evaluation
        summary = summarize_q3_evaluation(evaluation, probabilities=paths.probabilities)
        frozen = expected_summary.loc[(expected_summary.plan == name) & (expected_summary.surplus_discount == 0.5)]
        if len(frozen) != 1:
            raise RuntimeError("Frozen summary lookup is not unique")
        differences = {key: abs(value - float(frozen.iloc[0][key])) for key, value in summary.items()}
        path_difference = float(np.max(np.abs(evaluation.profits_yuan - expected_profits[f"{name}_profit_yuan"].to_numpy())))
        if max(differences.values()) > tolerance_yuan or path_difference > tolerance_yuan:
            raise RuntimeError(f"Frozen comparison failed for {name}: max path delta {path_difference}")
        results[name] = {
            "metrics": summary,
            "maximum_summary_absolute_difference": max(differences.values()),
            "maximum_path_profit_absolute_difference_yuan": path_difference,
            "agricultural_validation": asdict(feasibility[name]),
        }
    sensitivity = pd.read_csv(ROOT / "data" / "q3_sensitivity_summary.csv")
    keys = ["dependence_strength", "elasticity_scale", "surplus_rule", "plan"]
    if len(sensitivity) != 36 or sensitivity.duplicated(keys).any():
        raise RuntimeError("Frozen sensitivity grid is not 36 unique rows")
    expected_grid = {(d,e,r,p) for d in ("weak","baseline","strong") for e in ("zero","baseline","strong") for r in ("half_price_surplus","waste_all_surplus") for p in plans}
    if set(sensitivity[keys].itertuples(index=False, name=None)) != expected_grid:
        raise RuntimeError("Frozen sensitivity grid is incomplete")
    canonical = np.ascontiguousarray(paths.unit_draws, dtype="<f8")
    digest = hashlib.sha256(str(canonical.shape).encode("ascii") + canonical.tobytes()).hexdigest().upper()
    reference = json.loads((ROOT / "data" / "frozen_reference.json").read_text(encoding="utf-8"))
    paired = evaluations["q2_frozen"].profits_yuan - evaluations["q3_final"].profits_yuan
    imported_local = {}
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("q1_", "q2_", "q3_")) or name == "portable_data":
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(ROOT / "src"):
                raise RuntimeError(f"Nonportable source import: {name}")
            imported_local[name] = path.relative_to(ROOT).as_posix()
    result = {
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "fixed Q2/Q3 plans on shared Q3 half-price-surplus evaluation paths; no reoptimization",
        "classification": "Simulated out-of-sample evaluation, not observed agricultural performance",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "design": {"paths": 3000, "seed": paths.seed, "sampling": "mc", "surplus_discount": 0.5, "cvar_beta": 0.9, "common_random_numbers": True, "optimization_performed": False},
        "dimensions": {"plots": len(data.plots), "crops": len(data.crops), "land_area_mu": sum(p.area_mu for p in data.plots.values())},
        "numeric_tolerance_yuan": tolerance_yuan,
        "plans": results,
        "paired_q2_minus_q3": {"mean_profit_difference_yuan": float(paired.mean()), "q2_better_path_count": int((paired > 0).sum()), "q3_better_path_count": int((paired < 0).sum()), "ties": int((paired == 0).sum())},
        "scenario_fingerprint_sha256": digest,
        "historical_scenario_fingerprint_sha256": reference["evaluation_design"]["scenario_fingerprint_sha256"],
        "scenario_fingerprint_matches": digest == reference["evaluation_design"]["scenario_fingerprint_sha256"],
        "sensitivity": {"rows": 36, "grid_complete": True, "source_hash_verified": True, "rerun": False},
        "portable_imports": imported_local,
        "public_input_hashes": {e["public_relative_path"]:e["public_sha256"] for e in manifest["records"]},
        "execution_source_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"):file_hash(p) for p in [Path(__file__), ROOT / "src" / "portable_data.py"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "q2_mean_profit_yuan": results["q2_frozen"]["metrics"]["mean_profit_yuan"], "q3_mean_profit_yuan": results["q3_final"]["metrics"]["mean_profit_yuan"], "maximum_path_delta_yuan": max(r["maximum_path_profit_absolute_difference_yuan"] for r in results.values()), "fingerprint_matches": result["scenario_fingerprint_matches"]}))


if __name__ == "__main__":
    main()
