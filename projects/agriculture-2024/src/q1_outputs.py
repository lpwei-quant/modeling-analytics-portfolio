"""Machine-readable outputs and metrics for solved Q1 cases."""

from __future__ import annotations

import json
import math
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import scipy

from q1_data import OPEN_LAND_TYPES, SEASON_FIRST, SEASON_SECOND, SEASON_SINGLE, YEARS
from q1_model import Q1Solution
from q1_validate import ValidationReport


LONG_COLUMNS = [
    "case",
    "year",
    "season",
    "physical_opportunity",
    "plot",
    "land_type",
    "plot_capacity_mu",
    "crop_id",
    "crop_name",
    "crop_type",
    "is_bean",
    "area_mu",
    "activation_binary",
    "irrigated_regime",
    "yield_jin_per_mu",
    "production_jin",
    "cost_yuan_per_mu",
    "planting_cost_yuan",
    "price_lower_yuan_per_jin",
    "price_upper_yuan_per_jin",
    "midpoint_price_yuan_per_jin",
    "expected_sales_proxy_crop_season_jin",
    "normal_sales_allocated_jin",
    "excess_production_allocated_jin",
    "discount_sales_allocated_jin",
    "wasted_production_allocated_jin",
    "revenue_yuan",
    "profit_yuan",
    "parameter_provenance",
]


def _production_by_group(solution: Q1Solution) -> dict[tuple[int, str, int, str], float]:
    data = solution.model.data
    production: dict[tuple[int, str, int, str], float] = defaultdict(float)
    for (plot, crop_id, year, season), area in solution.x.items():
        land_type = data.plots[plot].land_type
        parameter = data.parameters[(crop_id, land_type, season)]
        production[(crop_id, land_type, year, season)] += (
            area * parameter.yield_jin_per_mu
        )
    return dict(production)


def solution_long_frame(solution: Q1Solution) -> pd.DataFrame:
    data = solution.model.data
    production_by_group = _production_by_group(solution)
    normal = solution.normal_sales
    excess = solution.excess_sales
    rows: list[dict[str, Any]] = []
    for key in data.compatible_cells:
        plot, crop_id, year, season = key
        plot_data = data.plots[plot]
        crop = data.crops[crop_id]
        parameter = data.parameters[(crop_id, plot_data.land_type, season)]
        area = solution.x[key]
        production = area * parameter.yield_jin_per_mu
        group = (crop_id, plot_data.land_type, year, season)
        group_production = production_by_group[group]
        share = production / group_production if group_production > 0 else 0.0
        normal_allocated = normal[group] * share
        excess_allocated = excess[group] * share
        discount_sales = (
            excess_allocated if solution.model.excess_price_factor > 0 else 0.0
        )
        wasted = excess_allocated if solution.model.excess_price_factor == 0 else 0.0
        revenue = parameter.midpoint_price_yuan_per_jin * (
            normal_allocated
            + solution.model.excess_price_factor * excess_allocated
        )
        planting_cost = area * parameter.cost_yuan_per_mu
        regime = "not_applicable"
        if plot_data.land_type == "水浇地":
            regime = (
                "rice_single"
                if solution.regimes[(plot, year)] >= 0.5
                else "two_season_vegetables"
            )
        rows.append(
            {
                "case": solution.model.case_name,
                "year": year,
                "season": season,
                "physical_opportunity": f"{year}:{season}",
                "plot": plot,
                "land_type": plot_data.land_type,
                "plot_capacity_mu": plot_data.area_mu,
                "crop_id": crop_id,
                "crop_name": crop.name,
                "crop_type": crop.crop_type,
                "is_bean": int(crop.is_bean),
                "area_mu": area,
                "activation_binary": solution.z[key],
                "irrigated_regime": regime,
                "yield_jin_per_mu": parameter.yield_jin_per_mu,
                "production_jin": production,
                "cost_yuan_per_mu": parameter.cost_yuan_per_mu,
                "planting_cost_yuan": planting_cost,
                "price_lower_yuan_per_jin": parameter.price_lower_yuan_per_jin,
                "price_upper_yuan_per_jin": parameter.price_upper_yuan_per_jin,
                "midpoint_price_yuan_per_jin": parameter.midpoint_price_yuan_per_jin,
                "expected_sales_proxy_crop_season_jin": data.expected_sales_proxy.get(
                    (crop_id, season), 0.0
                ),
                "normal_sales_allocated_jin": normal_allocated,
                "excess_production_allocated_jin": excess_allocated,
                "discount_sales_allocated_jin": discount_sales,
                "wasted_production_allocated_jin": wasted,
                "revenue_yuan": revenue,
                "profit_yuan": revenue - planting_cost,
                "parameter_provenance": parameter.provenance,
            }
        )
    frame = pd.DataFrame(rows, columns=LONG_COLUMNS)
    return frame.sort_values(
        ["year", "plot", "season", "crop_id"], kind="stable"
    ).reset_index(drop=True)


def _active_capacity(solution: Q1Solution, year: int) -> float:
    data = solution.model.data
    total = 0.0
    for plot, plot_data in data.plots.items():
        if plot_data.land_type in OPEN_LAND_TYPES:
            total += plot_data.area_mu
        elif plot_data.land_type == "水浇地":
            total += plot_data.area_mu * (
                1.0 if solution.regimes[(plot, year)] >= 0.5 else 2.0
            )
        else:
            total += 2.0 * plot_data.area_mu
    return total


def _highs_version() -> str:
    try:
        from scipy.optimize._highspy._core import (  # type: ignore[attr-defined]
            HIGHS_VERSION_MAJOR,
            HIGHS_VERSION_MINOR,
            HIGHS_VERSION_PATCH,
        )

        return f"{HIGHS_VERSION_MAJOR}.{HIGHS_VERSION_MINOR}.{HIGHS_VERSION_PATCH}"
    except Exception:
        return "embedded in SciPy; exact version unavailable"


def build_metrics(
    solution: Q1Solution,
    frame: pd.DataFrame,
    validation: ValidationReport,
    *,
    solve_seconds: float,
) -> dict[str, Any]:
    annual: list[dict[str, Any]] = []
    for year in YEARS:
        subset = frame[frame["year"] == year]
        planted = float(subset["area_mu"].sum())
        active_capacity = _active_capacity(solution, year)
        annual.append(
            {
                "year": year,
                "profit_yuan": float(subset["profit_yuan"].sum()),
                "revenue_yuan": float(subset["revenue_yuan"].sum()),
                "planting_cost_yuan": float(subset["planting_cost_yuan"].sum()),
                "planted_area_mu": planted,
                "active_capacity_mu": active_capacity,
                "unused_area_mu": active_capacity - planted,
                "production_jin": float(subset["production_jin"].sum()),
                "normal_sales_jin": float(
                    subset["normal_sales_allocated_jin"].sum()
                ),
                "excess_production_jin": float(
                    subset["excess_production_allocated_jin"].sum()
                ),
                "wasted_production_jin": float(
                    subset["wasted_production_allocated_jin"].sum()
                ),
                "discount_sales_jin": float(
                    subset["discount_sales_allocated_jin"].sum()
                ),
            }
        )

    crop_summary = (
        frame.groupby(["crop_id", "crop_name", "season"], as_index=False)
        .agg(
            planted_area_mu=("area_mu", "sum"),
            production_jin=("production_jin", "sum"),
            normal_sales_jin=("normal_sales_allocated_jin", "sum"),
            excess_production_jin=("excess_production_allocated_jin", "sum"),
            wasted_production_jin=("wasted_production_allocated_jin", "sum"),
            discount_sales_jin=("discount_sales_allocated_jin", "sum"),
            revenue_yuan=("revenue_yuan", "sum"),
            planting_cost_yuan=("planting_cost_yuan", "sum"),
            profit_yuan=("profit_yuan", "sum"),
        )
        .sort_values(["crop_id", "season"])
    )
    result = solution.result
    total_planted = float(frame["area_mu"].sum())
    total_active_capacity = math.fsum(item["active_capacity_mu"] for item in annual)
    return {
        "case": solution.model.case_name,
        "excess_price_factor": solution.model.excess_price_factor,
        "solver": {
            "library": "scipy.optimize.milp",
            "scipy_version": scipy.__version__,
            "highs_version": _highs_version(),
            "python_version": platform.python_version(),
            "status_code": int(result.status),
            "success": bool(result.success),
            "message": str(result.message),
            "mip_relative_gap_tolerance": 0.005,
            "achieved_mip_gap": float(getattr(result, "mip_gap", math.nan)),
            "mip_node_count": int(getattr(result, "mip_node_count", 0)),
            "dual_bound_profit_yuan": -float(getattr(result, "mip_dual_bound", result.fun)),
            "solve_seconds": solve_seconds,
        },
        "model": {
            "variable_count": len(solution.model.variables.names),
            "binary_variable_count": int(solution.model.variables.integrality.sum()),
            "constraint_count": len(solution.model.constraints.names),
            "constraint_category_counts": solution.model.constraints.category_counts,
            "constraint_matrix_nonzeros": solution.model.constraints.matrix_nonzeros,
        },
        "source_hashes": solution.model.data.source_hashes,
        "assumptions": {
            "expected_sales_proxy": "2023 theoretical production by crop-season",
            "representative_price": "midpoint of observed 2023 price interval",
            "bean_requirement": "rolling three-year cumulative bean area >= plot area",
            "management_thresholds": "deferred; none imposed",
        },
        "totals": {
            "profit_yuan": float(frame["profit_yuan"].sum()),
            "solver_profit_yuan": solution.total_profit_yuan,
            "revenue_yuan": float(frame["revenue_yuan"].sum()),
            "planting_cost_yuan": float(frame["planting_cost_yuan"].sum()),
            "planted_area_mu": total_planted,
            "active_capacity_mu": total_active_capacity,
            "unused_area_mu": total_active_capacity - total_planted,
            "production_jin": float(frame["production_jin"].sum()),
            "normal_sales_jin": float(frame["normal_sales_allocated_jin"].sum()),
            "excess_production_jin": float(
                frame["excess_production_allocated_jin"].sum()
            ),
            "wasted_production_jin": float(
                frame["wasted_production_allocated_jin"].sum()
            ),
            "discount_sales_jin": float(
                frame["discount_sales_allocated_jin"].sum()
            ),
        },
        "annual": annual,
        "crop_season": crop_summary.to_dict(orient="records"),
        "validation": validation.to_dict(),
    }


def write_case_outputs(
    output_dir: Path,
    case_stem: str,
    frame: pd.DataFrame,
    metrics: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{case_stem}_long.csv"
    metrics_path = output_dir / f"{case_stem}_metrics.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return csv_path, metrics_path


def build_template_payload(
    solution: Q1Solution, output_path: Path
) -> Path:
    data = solution.model.data
    payload: dict[str, Any] = {
        "case": solution.model.case_name,
        "years": list(YEARS),
        "crop_order": [data.crops[crop_id].name for crop_id in sorted(data.crops)],
        "plot_order": list(data.plots),
        "second_plot_order": [
            plot
            for plot, plot_data in data.plots.items()
            if plot_data.land_type not in OPEN_LAND_TYPES
        ],
    }
    payload["values"] = {}
    crop_ids = sorted(data.crops)
    second_plots = payload["second_plot_order"]
    for year in YEARS:
        first = [
            [
                math.fsum(
                    solution.x.get((plot, crop_id, year, season), 0.0)
                    for season in (SEASON_SINGLE, SEASON_FIRST)
                )
                for crop_id in crop_ids
            ]
            for plot in payload["plot_order"]
        ]
        second = [
            [solution.x.get((plot, crop_id, year, SEASON_SECOND), 0.0) for crop_id in crop_ids]
            for plot in second_plots
        ]
        payload["values"][str(year)] = {"first": first, "second": second}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path
