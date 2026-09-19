"""Portable serialization layer; frozen modeling rules remain in q1_data.py."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from q1_data import (
    Crop, EconomicParameter, Planting2023, Plot, Q1Data,
    build_adjacency_edges, build_available_plot_seasons, build_compatible_cells,
    build_initial_state,
)


def basis_payload(plots, crops, parameters, plantings, expected_sales, source_hashes):
    return {
        "schema_version": 1,
        "classification": "Derived input basis; demand proxy is an Assumption",
        "plots": [asdict(value) for value in plots.values()],
        "crops": [asdict(value) for value in crops.values()],
        "parameters": [asdict(value) for value in parameters.values()],
        "plantings_2023": [asdict(value) for value in plantings],
        "expected_sales_proxy": [
            {"crop_id": crop, "season": season, "quantity_jin": quantity}
            for (crop, season), quantity in expected_sales.items()
        ],
        "source_hashes": source_hashes,
    }


def load_portable_data(root: Path) -> Q1Data:
    obj = json.loads((root / "data" / "input_basis.json").read_text(encoding="utf-8"))
    if obj["schema_version"] != 1:
        raise ValueError("Unsupported input basis schema")
    plots = {v["name"]: Plot(**v) for v in obj["plots"]}
    crops = {v["crop_id"]: Crop(**v) for v in obj["crops"]}
    parameters = {
        (v["crop_id"], v["land_type"], v["season"]): EconomicParameter(**v)
        for v in obj["parameters"]
    }
    plantings = tuple(Planting2023(**v) for v in obj["plantings_2023"])
    expected = {(v["crop_id"], v["season"]): v["quantity_jin"] for v in obj["expected_sales_proxy"]}
    last, first, bean_area = build_initial_state(plots, crops, plantings)
    return Q1Data(
        project_root=root, plots=plots, crops=crops, parameters=parameters,
        plantings_2023=plantings, expected_sales_proxy=expected,
        compatible_cells=build_compatible_cells(plots, crops, parameters),
        available_plot_seasons=build_available_plot_seasons(plots),
        adjacency_edges=build_adjacency_edges(plots), initial_last_crops=last,
        initial_first_seasons=first, bean_area_2023=bean_area,
        source_hashes=obj["source_hashes"],
    )
