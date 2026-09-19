"""Sparse deterministic MILP for CUMCM 2024 Problem C, Question 1."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, milp
from scipy.sparse import coo_matrix

from q1_data import (
    IRRIGATED_LAND,
    SEASON_FIRST,
    SEASON_SECOND,
    SEASON_SINGLE,
    YEARS,
    Q1Data,
)


class Q1ModelError(RuntimeError):
    """Raised when the Q1 model cannot be built or solved to optimality."""


PlantKey = tuple[str, int, int, str]
RegimeKey = tuple[str, int]
SalesKey = tuple[int, str, int, str]
NUMERICAL_ZERO = 1e-8


def _clean_continuous(value: float) -> float:
    """Suppress solver-scale signed zeros without changing material decisions."""
    return 0.0 if abs(value) <= NUMERICAL_ZERO else float(value)


@dataclass(frozen=True)
class VariableBlock:
    x: dict[PlantKey, int]
    z: dict[PlantKey, int]
    regime: dict[RegimeKey, int]
    normal_sales: dict[SalesKey, int]
    excess_sales: dict[SalesKey, int]
    names: tuple[str, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    integrality: np.ndarray
    objective: np.ndarray


@dataclass(frozen=True)
class ConstraintSystem:
    linear_constraint: LinearConstraint
    names: tuple[str, ...]
    category_counts: dict[str, int]
    matrix_nonzeros: int


@dataclass(frozen=True)
class Q1BuiltModel:
    data: Q1Data
    case_name: str
    excess_price_factor: float
    variables: VariableBlock
    constraints: ConstraintSystem


@dataclass(frozen=True)
class Q1Solution:
    model: Q1BuiltModel
    result: OptimizeResult
    values: np.ndarray
    total_profit_yuan: float

    @property
    def x(self) -> dict[PlantKey, float]:
        return {
            key: _clean_continuous(self.values[index])
            for key, index in self.model.variables.x.items()
        }

    @property
    def z(self) -> dict[PlantKey, float]:
        return {key: float(self.values[index]) for key, index in self.model.variables.z.items()}

    @property
    def regimes(self) -> dict[RegimeKey, float]:
        return {
            key: float(self.values[index])
            for key, index in self.model.variables.regime.items()
        }

    @property
    def normal_sales(self) -> dict[SalesKey, float]:
        return {
            key: _clean_continuous(self.values[index])
            for key, index in self.model.variables.normal_sales.items()
        }

    @property
    def excess_sales(self) -> dict[SalesKey, float]:
        return {
            key: _clean_continuous(self.values[index])
            for key, index in self.model.variables.excess_sales.items()
        }


class VariableRegistry:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integrality: list[int] = []
        self.objective: list[float] = []

    def add(
        self,
        name: str,
        *,
        lower: float = 0.0,
        upper: float = math.inf,
        integer: bool = False,
        objective: float = 0.0,
    ) -> int:
        index = len(self.names)
        self.names.append(name)
        self.lower.append(float(lower))
        self.upper.append(float(upper))
        self.integrality.append(1 if integer else 0)
        self.objective.append(float(objective))
        return index


class SparseConstraintBuilder:
    def __init__(self, variable_count: int) -> None:
        self.variable_count = variable_count
        self.row_indices: list[int] = []
        self.column_indices: list[int] = []
        self.values: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.names: list[str] = []
        self.categories: Counter[str] = Counter()

    def add(
        self,
        coefficients: Mapping[int, float] | Iterable[tuple[int, float]],
        *,
        lower: float = -math.inf,
        upper: float = math.inf,
        category: str,
        name: str,
    ) -> None:
        row = len(self.lower)
        items = coefficients.items() if isinstance(coefficients, Mapping) else coefficients
        nonzero_count = 0
        for column, value in items:
            coefficient = float(value)
            if abs(coefficient) <= 0.0:
                continue
            self.row_indices.append(row)
            self.column_indices.append(int(column))
            self.values.append(coefficient)
            nonzero_count += 1
        if nonzero_count == 0:
            if lower <= 0 <= upper:
                return
            raise Q1ModelError(f"Infeasible empty constraint {name}: {lower} <= 0 <= {upper}")
        self.lower.append(float(lower))
        self.upper.append(float(upper))
        self.names.append(name)
        self.categories[category] += 1

    def build(self) -> ConstraintSystem:
        matrix = coo_matrix(
            (self.values, (self.row_indices, self.column_indices)),
            shape=(len(self.lower), self.variable_count),
            dtype=float,
        ).tocsr()
        return ConstraintSystem(
            linear_constraint=LinearConstraint(
                matrix,
                np.asarray(self.lower, dtype=float),
                np.asarray(self.upper, dtype=float),
            ),
            names=tuple(self.names),
            category_counts=dict(sorted(self.categories.items())),
            matrix_nonzeros=int(matrix.nnz),
        )


def sales_groups(data: Q1Data) -> tuple[SalesKey, ...]:
    groups = {
        (crop_id, data.plots[plot].land_type, year, season)
        for plot, crop_id, year, season in data.compatible_cells
    }
    return tuple(sorted(groups, key=lambda key: (key[2], key[3], key[0], key[1])))


def build_variables(data: Q1Data, excess_price_factor: float) -> VariableBlock:
    if excess_price_factor not in {0.0, 0.5}:
        raise Q1ModelError(f"Unsupported excess price factor: {excess_price_factor}")
    registry = VariableRegistry()
    x: dict[PlantKey, int] = {}
    z: dict[PlantKey, int] = {}
    regime: dict[RegimeKey, int] = {}
    normal_sales: dict[SalesKey, int] = {}
    excess_sales: dict[SalesKey, int] = {}

    for key in data.compatible_cells:
        plot, crop_id, year, season = key
        plot_data = data.plots[plot]
        parameter = data.parameters[(crop_id, plot_data.land_type, season)]
        x[key] = registry.add(
            f"x[{plot},{crop_id},{year},{season}]",
            lower=0.0,
            upper=plot_data.area_mu,
            objective=parameter.cost_yuan_per_mu,
        )
        z[key] = registry.add(
            f"z[{plot},{crop_id},{year},{season}]",
            lower=0.0,
            upper=1.0,
            integer=True,
        )

    for plot in data.irrigated_plots:
        for year in YEARS:
            regime[(plot, year)] = registry.add(
                f"u_rice[{plot},{year}]", lower=0.0, upper=1.0, integer=True
            )

    for key in sales_groups(data):
        crop_id, land_type, year, season = key
        parameter = data.parameters[(crop_id, land_type, season)]
        price = parameter.midpoint_price_yuan_per_jin
        normal_sales[key] = registry.add(
            f"normal[{crop_id},{land_type},{year},{season}]",
            lower=0.0,
            objective=-price,
        )
        excess_sales[key] = registry.add(
            f"excess[{crop_id},{land_type},{year},{season}]",
            lower=0.0,
            objective=-excess_price_factor * price,
        )

    return VariableBlock(
        x=x,
        z=z,
        regime=regime,
        normal_sales=normal_sales,
        excess_sales=excess_sales,
        names=tuple(registry.names),
        lower_bounds=np.asarray(registry.lower, dtype=float),
        upper_bounds=np.asarray(registry.upper, dtype=float),
        integrality=np.asarray(registry.integrality, dtype=np.uint8),
        objective=np.asarray(registry.objective, dtype=float),
    )


def add_capacity_constraints(
    builder: SparseConstraintBuilder, data: Q1Data, variables: VariableBlock
) -> None:
    cells_by_plot_season: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for (plot, _crop, year, season), index in variables.x.items():
        cells_by_plot_season[(plot, year, season)].append(index)
    for plot, year, season in data.available_plot_seasons:
        builder.add(
            ((index, 1.0) for index in cells_by_plot_season[(plot, year, season)]),
            upper=data.plots[plot].area_mu,
            category="capacity",
            name=f"capacity[{plot},{year},{season}]",
        )


def add_activation_constraints(
    builder: SparseConstraintBuilder, data: Q1Data, variables: VariableBlock
) -> None:
    for key, x_index in variables.x.items():
        plot = key[0]
        builder.add(
            {x_index: 1.0, variables.z[key]: -data.plots[plot].area_mu},
            upper=0.0,
            category="activation",
            name=f"activation[{plot},{key[1]},{key[2]},{key[3]}]",
        )


def add_irrigated_regime_constraints(
    builder: SparseConstraintBuilder, data: Q1Data, variables: VariableBlock
) -> None:
    x_by_plot_year_season: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for (plot, _crop, year, season), index in variables.x.items():
        x_by_plot_year_season[(plot, year, season)].append(index)
    for plot in data.irrigated_plots:
        area = data.plots[plot].area_mu
        for year in YEARS:
            u = variables.regime[(plot, year)]
            builder.add(
                [(index, 1.0) for index in x_by_plot_year_season[(plot, year, SEASON_SINGLE)]]
                + [(u, -area)],
                upper=0.0,
                category="irrigated_regime",
                name=f"rice_regime[{plot},{year}]",
            )
            for season in (SEASON_FIRST, SEASON_SECOND):
                builder.add(
                    [(index, 1.0) for index in x_by_plot_year_season[(plot, year, season)]]
                    + [(u, area)],
                    upper=area,
                    category="irrigated_regime",
                    name=f"vegetable_regime[{plot},{year},{season}]",
                )


def add_irrigated_second_season_choice(
    builder: SparseConstraintBuilder, data: Q1Data, variables: VariableBlock
) -> None:
    for plot in data.irrigated_plots:
        for year in YEARS:
            indices = [
                variables.z[(plot, crop_id, year, SEASON_SECOND)]
                for crop_id in data.compatible_crops(plot, SEASON_SECOND)
            ]
            builder.add(
                ((index, 1.0) for index in indices),
                upper=1.0,
                category="irrigated_second_choice",
                name=f"irrigated_second_choice[{plot},{year}]",
            )


def add_rotation_constraints(
    builder: SparseConstraintBuilder, data: Q1Data, variables: VariableBlock
) -> None:
    for edge in data.adjacency_edges:
        first_crops = set(data.compatible_crops(edge.plot, edge.season_from))
        second_crops = set(data.compatible_crops(edge.plot, edge.season_to))
        for crop_id in sorted(first_crops.intersection(second_crops)):
            first_key = (edge.plot, crop_id, edge.year_from, edge.season_from)
            second_key = (edge.plot, crop_id, edge.year_to, edge.season_to)
            builder.add(
                {variables.z[first_key]: 1.0, variables.z[second_key]: 1.0},
                upper=1.0,
                category="rotation",
                name=(
                    f"rotation[{edge.plot},{crop_id},{edge.year_from},{edge.season_from},"
                    f"{edge.year_to},{edge.season_to}]"
                ),
            )


def add_initial_rotation_constraints(
    builder: SparseConstraintBuilder, data: Q1Data, variables: VariableBlock
) -> None:
    for plot, crop_ids in data.initial_last_crops.items():
        for season in data.initial_first_seasons[plot]:
            compatible = set(data.compatible_crops(plot, season))
            for crop_id in sorted(crop_ids.intersection(compatible)):
                key = (plot, crop_id, 2024, season)
                builder.add(
                    {variables.z[key]: 1.0},
                    upper=0.0,
                    category="initial_rotation",
                    name=f"initial_rotation[{plot},{crop_id},{season}]",
                )


def add_bean_window_constraints(
    builder: SparseConstraintBuilder, data: Q1Data, variables: VariableBlock
) -> None:
    bean_ids = data.bean_crop_ids
    for plot, plot_data in data.plots.items():
        first_indices = [
            index
            for (cell_plot, crop_id, year, _season), index in variables.x.items()
            if cell_plot == plot and crop_id in bean_ids and year in {2024, 2025}
        ]
        required_after_2023 = plot_data.area_mu - data.bean_area_2023[plot]
        builder.add(
            ((index, 1.0) for index in first_indices),
            lower=required_after_2023,
            category="bean_window",
            name=f"bean_window[{plot},2023-2025]",
        )
        for start in range(2024, 2029):
            indices = [
                index
                for (cell_plot, crop_id, year, _season), index in variables.x.items()
                if cell_plot == plot
                and crop_id in bean_ids
                and start <= year <= start + 2
            ]
            builder.add(
                ((index, 1.0) for index in indices),
                lower=plot_data.area_mu,
                category="bean_window",
                name=f"bean_window[{plot},{start}-{start + 2}]",
            )


def add_production_balance_constraints(
    builder: SparseConstraintBuilder, data: Q1Data, variables: VariableBlock
) -> None:
    x_by_group: dict[SalesKey, list[tuple[int, float]]] = defaultdict(list)
    for (plot, crop_id, year, season), index in variables.x.items():
        land_type = data.plots[plot].land_type
        parameter = data.parameters[(crop_id, land_type, season)]
        x_by_group[(crop_id, land_type, year, season)].append(
            (index, -parameter.yield_jin_per_mu)
        )
    for key, normal_index in variables.normal_sales.items():
        coefficients = x_by_group[key] + [
            (normal_index, 1.0),
            (variables.excess_sales[key], 1.0),
        ]
        builder.add(
            coefficients,
            lower=0.0,
            upper=0.0,
            category="production_balance",
            name=f"production_balance[{key[0]},{key[1]},{key[2]},{key[3]}]",
        )


def add_expected_sales_constraints(
    builder: SparseConstraintBuilder, data: Q1Data, variables: VariableBlock
) -> None:
    normal_by_crop_season_year: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for (crop_id, _land_type, year, season), index in variables.normal_sales.items():
        normal_by_crop_season_year[(crop_id, year, season)].append(index)
    for (crop_id, year, season), indices in sorted(normal_by_crop_season_year.items()):
        demand = data.expected_sales_proxy.get((crop_id, season), 0.0)
        builder.add(
            ((index, 1.0) for index in indices),
            upper=demand,
            category="expected_sales",
            name=f"expected_sales[{crop_id},{year},{season}]",
        )


def build_constraints(data: Q1Data, variables: VariableBlock) -> ConstraintSystem:
    builder = SparseConstraintBuilder(len(variables.names))
    add_capacity_constraints(builder, data, variables)
    add_activation_constraints(builder, data, variables)
    add_irrigated_regime_constraints(builder, data, variables)
    add_irrigated_second_season_choice(builder, data, variables)
    add_rotation_constraints(builder, data, variables)
    add_initial_rotation_constraints(builder, data, variables)
    add_bean_window_constraints(builder, data, variables)
    add_production_balance_constraints(builder, data, variables)
    add_expected_sales_constraints(builder, data, variables)
    return builder.build()


def build_q1_model(
    data: Q1Data, *, case_name: str, excess_price_factor: float
) -> Q1BuiltModel:
    variables = build_variables(data, excess_price_factor)
    constraints = build_constraints(data, variables)
    return Q1BuiltModel(
        data=data,
        case_name=case_name,
        excess_price_factor=excess_price_factor,
        variables=variables,
        constraints=constraints,
    )


def solve_q1_model(
    model: Q1BuiltModel,
    *,
    time_limit_seconds: float = 300.0,
    mip_relative_gap: float = 0.005,
    display_solver_log: bool = False,
) -> Q1Solution:
    result = milp(
        c=model.variables.objective,
        integrality=model.variables.integrality,
        bounds=Bounds(model.variables.lower_bounds, model.variables.upper_bounds),
        constraints=model.constraints.linear_constraint,
        options={
            "disp": display_solver_log,
            "presolve": True,
            "time_limit": float(time_limit_seconds),
            "mip_rel_gap": float(mip_relative_gap),
        },
    )
    if result.status != 0 or result.x is None or result.fun is None:
        raise Q1ModelError(
            f"{model.case_name} did not solve within the declared MIP gap: "
            f"status={result.status}, success={result.success}, message={result.message}"
        )
    values = np.asarray(result.x, dtype=float)
    if not np.all(np.isfinite(values)):
        raise Q1ModelError(f"{model.case_name} returned non-finite variable values")
    return Q1Solution(
        model=model,
        result=result,
        values=values,
        total_profit_yuan=-float(result.fun),
    )
