"""Pandas-only versions of the tabular tools used by synthetic training cases."""

from __future__ import annotations

import ast
from collections.abc import Callable
from io import StringIO
from typing import Any

from corecoder.tools.base import Tool


def _frame(state: dict, name: str):
    return (state.get("data_store") or {}).get(name)


def _frame_info(frame) -> tuple[str, str]:
    buffer = StringIO()
    frame.info(buf=buffer)
    non_empty = (frame.count() / len(frame) * 100).round(2)
    non_empty_info = "\n\nNon-empty values (%):\n"
    for column, percentage in non_empty.items():
        non_empty_info += f"{column}: {percentage}%\n"
    return buffer.getvalue(), non_empty_info


def _merged_description(output_name: str, frame) -> str:
    info_string, non_empty_info = _frame_info(frame)
    return (
        f"Created merged DataFrame named {output_name}.\n"
        f"Description of merged DataFrame:\n{info_string}\n"
        f"Amount of non-empty values in this geodataframe:\n{non_empty_info}"
    )


def _merge(state: dict, dataframe_name: str, geodataframe_name: str,
           statkey: str, geokey: str, output_dataframe_name: str) -> str:
    data = _frame(state, dataframe_name)
    geo = _frame(state, geodataframe_name)
    merged = geo.merge(data, left_on=geokey, right_on=statkey, how="left")
    state.setdefault("data_store", {})[output_dataframe_name] = merged
    return _merged_description(output_dataframe_name, merged)


def _unique_values(state: dict, dataframe_name: str, column: str) -> str:
    frame = _frame(state, dataframe_name)
    if column not in frame.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    values = frame[column].unique().tolist()
    return f"Unique values in column '{column}' of {dataframe_name}:\n{values}"


def _filter_categorical(state: dict, dataframe_name: str, filters: Any,
                        output_dataframe_name: str) -> str:
    if isinstance(filters, str):
        filters = ast.literal_eval(filters)
    frame = _frame(state, dataframe_name)
    filtered = frame.copy()
    if filters:
        for column, values in filters.items():
            if column not in frame.columns:
                raise ValueError(f"Column '{column}' not found in dataframe")
            if isinstance(values, (list, tuple, set)):
                filtered = filtered[filtered[column].isin(values)]
            else:
                filtered = filtered[filtered[column] == values]
    state.setdefault("data_store", {})[output_dataframe_name] = filtered
    return (
        f"Created filtered DataFrame named {output_dataframe_name} from DataFrame {dataframe_name}.\n"
        f"Filters applied: {filters}\n"
        f"Columns in the new dataframe  {output_dataframe_name} are the same as in the original DataFrame {dataframe_name}.\n"
        f"Number of rows in the new {output_dataframe_name} is {len(filtered)}, different from the original {len(frame)}."
    )


def _statistics(state: dict, dataframe_name: str, column_name: str,
                output_variable_name: str, include_quantiles: bool = True,
                additional_quantiles: Any = None) -> str:
    if isinstance(additional_quantiles, str):
        additional_quantiles = ast.literal_eval(additional_quantiles)
    frame = _frame(state, dataframe_name)
    if frame is None:
        return f"Error: DataFrame '{dataframe_name}' not found in data store"
    if column_name not in frame.columns:
        return f"Error: Column '{column_name}' not found in DataFrame"
    column = frame[column_name]
    stats = {
        "summary": {
            "count": len(column),
            "valid_count": column.count(),
            "missing_count": column.isnull().sum(),
            "min": float(column.min()) if not column.empty else None,
            "max": float(column.max()) if not column.empty else None,
            "mean": float(column.mean()) if not column.empty else None,
            "std": float(column.std()) if not column.empty else None,
        }
    }
    if include_quantiles:
        stats["quantiles"] = {
            "q25": float(column.quantile(0.25)),
            "q50": float(column.quantile(0.50)),
            "q75": float(column.quantile(0.75)),
        }
    if additional_quantiles:
        stats["additional_quantiles"] = {
            f"q{int(q * 100)}": float(column.quantile(q))
            for q in additional_quantiles
        }
    state.setdefault("data_store", {})[output_variable_name] = stats
    summary = stats["summary"]
    parts = [
        f"Statistics for column '{column_name}' in {dataframe_name}:",
        "\nBasic Statistics:",
        f"- Count: {summary['count']:,} (Valid: {summary['valid_count']:,}, Missing: {summary['missing_count']:,})",
        f"- Range: {summary['min']:,.2f} to {summary['max']:,.2f}",
        f"- Mean: {summary['mean']:,.2f}",
        f"- Standard Deviation: {summary['std']:,.2f}",
    ]
    if include_quantiles:
        quantiles = stats["quantiles"]
        parts.extend([
            "\nQuantile:",
            f"- 25th percentile (Q1): {quantiles['q25']:,.2f}",
            f"- 50th percentile (Q2/median): {quantiles['q50']:,.2f}",
            f"- 75th percentile (Q3): {quantiles['q75']:,.2f}",
        ])
    if additional_quantiles:
        parts.append("\nAdditional Quantiles:")
        for quantile in additional_quantiles:
            key = f"q{int(quantile * 100)}"
            parts.append(f"- {int(quantile * 100)}th percentile: {stats['additional_quantiles'][key]:,.2f}")
    parts.append(f"\nResults stored in variable '{output_variable_name}'")
    return "\n".join(parts)


class SyntheticDataTool(Tool):
    concurrency_safe = False

    def __init__(self, name: str, description: str, parameters: dict,
                 function: Callable[..., str], state: dict):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._function = function
        self._state = state

    def execute(self, **kwargs: Any) -> str:
        try:
            return str(self._function(self._state, **kwargs))
        except Exception as exc:  # noqa: BLE001 - tools return errors as text
            return f"Error: {type(exc).__name__}: {exc}"


def synthetic_tabular_tools(state: dict) -> dict[str, SyntheticDataTool]:
    return {
        "get_unique_values": SyntheticDataTool(
            "get_unique_values",
            "Gets unique values from a specified column in a DataFrame/GeoDataFrame.",
            {"type": "object", "properties": {
                "dataframe_name": {"type": "string"},
                "column": {"type": "string"},
            }, "required": ["dataframe_name", "column"]},
            _unique_values, state),
        "filter_categorical": SyntheticDataTool(
            "filter_categorical",
            "Filters DataFrame/GeoDataFrame by categorical values in specified columns.",
            {"type": "object", "properties": {
                "dataframe_name": {"type": "string"},
                "filters": {"type": "string"},
                "output_dataframe_name": {"type": "string"},
            }, "required": ["dataframe_name", "filters", "output_dataframe_name"]},
            _filter_categorical, state),
        "merge_dataframes": SyntheticDataTool(
            "merge_dataframes",
            "Merge tabular dataframes using specified key columns.",
            {"type": "object", "properties": {
                "dataframe_name": {"type": "string"},
                "geodataframe_name": {"type": "string"},
                "statkey": {"type": "string"},
                "geokey": {"type": "string"},
                "output_dataframe_name": {"type": "string"},
            }, "required": ["dataframe_name", "geodataframe_name", "statkey", "geokey",
                               "output_dataframe_name"]},
            _merge, state),
        "calculate_column_statistics": SyntheticDataTool(
            "calculate_column_statistics",
            "Calculate summary statistics for a numerical column.",
            {"type": "object", "properties": {
                "dataframe_name": {"type": "string"},
                "column_name": {"type": "string"},
                "output_variable_name": {"type": "string"},
                "include_quantiles": {"type": "boolean", "default": True},
                "additional_quantiles": {"type": ["string", "null"]},
            }, "required": ["dataframe_name", "column_name", "output_variable_name"]},
            _statistics, state),
    }
