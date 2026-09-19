"""Machine-auditable Q2 plan, scenario-profit, and receipt outputs."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from q1_data import Q1Data
from q1_model import PlantKey
from q2_model import Q2Solution
from q2_validate import (
    AgriculturalValidation,
    EconomicValidation,
    plan_diagnostics,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def scenario_fingerprint(unit_draws: np.ndarray) -> str:
    canonical = np.ascontiguousarray(unit_draws, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes())
    return digest.hexdigest().upper()


def plan_frame(
    data: Q1Data,
    plan: dict[PlantKey, float],
    *,
    candidate_id: str,
    z: dict[PlantKey, float] | None = None,
    regimes: dict[tuple[str, int], float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key in data.compatible_cells:
        plot, crop_id, year, season = key
        parameter = data.parameters[(crop_id, data.plots[plot].land_type, season)]
        rows.append(
            {
                "candidate_id": candidate_id,
                "plot": plot,
                "land_type": data.plots[plot].land_type,
                "plot_capacity_mu": data.plots[plot].area_mu,
                "crop_id": crop_id,
                "crop_name": data.crops[crop_id].name,
                "crop_type": data.crops[crop_id].crop_type,
                "year": year,
                "season": season,
                "area_mu": float(plan.get(key, 0.0)),
                "activation_binary": (
                    float(z[key]) if z is not None else float(plan.get(key, 0.0) > 1e-6)
                ),
                "irrigated_regime": (
                    float(regimes[(plot, year)])
                    if regimes is not None and (plot, year) in regimes
                    else "not_supplied"
                ),
                "base_yield_jin_per_mu": parameter.yield_jin_per_mu,
                "base_cost_yuan_per_mu": parameter.cost_yuan_per_mu,
                "base_midpoint_price_yuan_per_jin": parameter.midpoint_price_yuan_per_jin,
                "parameter_provenance": parameter.provenance,
            }
        )
    return pd.DataFrame(rows)


def write_plan_csv(
    path: Path,
    data: Q1Data,
    plan: dict[PlantKey, float],
    *,
    candidate_id: str,
    z: dict[PlantKey, float] | None = None,
    regimes: dict[tuple[str, int], float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = plan_frame(
        data, plan, candidate_id=candidate_id, z=z, regimes=regimes
    )
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12f")


def _validation_rows(report: AgriculturalValidation | EconomicValidation) -> list[dict[str, object]]:
    return [
        {
            "name": finding.name,
            "passed": finding.passed,
            "max_violation": finding.max_violation,
            "details": finding.details,
        }
        for finding in report.findings
    ]


def solution_receipt(
    solution: Q2Solution,
    *,
    candidate_id: str,
    agricultural: AgriculturalValidation,
    economic: EconomicValidation,
    q1_freeze: dict[str, object],
) -> dict[str, Any]:
    model = solution.model
    result = solution.result
    return {
        "status": "ok",
        "candidate_id": candidate_id,
        "objective_kind": model.objective_kind,
        "cvar_floor_yuan": model.cvar_floor_yuan,
        "expected_profit_floor_yuan": model.expected_profit_floor_yuan,
        "scenario_count": model.scenarios.count,
        "scenario_seed": model.scenarios.seed,
        "sampling_method": model.scenarios.sampling_method,
        "scenario_fingerprint_sha256": scenario_fingerprint(model.scenarios.unit_draws),
        "expected_profit_yuan": solution.expected_profit_yuan,
        "empirical_cvar_90_yuan": solution.empirical_cvar_yuan,
        "auxiliary_cvar_yuan": solution.auxiliary_cvar_yuan,
        "solver": {
            "status_code": int(result.status),
            "success": bool(result.success),
            "message": str(result.message),
            "scipy_minimization_objective": float(result.fun),
            "mip_gap": float(getattr(result, "mip_gap", math.nan)),
            "mip_node_count": int(getattr(result, "mip_node_count", -1)),
            "mip_dual_bound": float(getattr(result, "mip_dual_bound", math.nan)),
            "solve_seconds": solution.solve_seconds,
        },
        "model_size": {
            "variables": len(model.variables.names),
            "binary_variables": int(np.count_nonzero(model.variables.integrality)),
            "constraints": len(model.constraints.names),
            "matrix_nonzeros": model.constraints.matrix_nonzeros,
            "constraint_categories": model.constraints.category_counts,
        },
        "plan_diagnostics": plan_diagnostics(model.data, solution.x),
        "validation": {
            "agricultural_passed": agricultural.passed,
            "agricultural_findings": _validation_rows(agricultural),
            "economic_passed": economic.passed,
            "economic_findings": _validation_rows(economic),
            "q1_freeze": q1_freeze,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "config": model.config.to_dict(),
    }


def write_candidate_outputs(
    output_dir: Path,
    solution: Q2Solution,
    *,
    candidate_id: str,
    agricultural: AgriculturalValidation,
    economic: EconomicValidation,
    q1_freeze: dict[str, object],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / f"{candidate_id}_plan.csv"
    profits_path = output_dir / f"{candidate_id}_scenario_profits.csv"
    receipt_path = output_dir / f"{candidate_id}_receipt.json"
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
    ).to_csv(profits_path, index=False, encoding="utf-8-sig", float_format="%.12f")
    write_json(
        receipt_path,
        solution_receipt(
            solution,
            candidate_id=candidate_id,
            agricultural=agricultural,
            economic=economic,
            q1_freeze=q1_freeze,
        ),
    )
    return {
        "plan": str(plan_path),
        "scenario_profits": str(profits_path),
        "receipt": str(receipt_path),
        "plan_sha256": sha256(plan_path),
        "scenario_profits_sha256": sha256(profits_path),
        "receipt_sha256": sha256(receipt_path),
    }
