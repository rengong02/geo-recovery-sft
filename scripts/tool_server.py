#!/usr/bin/env python
"""A tool executor on a pipe, so the model and the GIS stack can live apart.

This exists because of an environment fact, not a design preference: the
interpreter that has torch cannot import geopandas here, and the one that has
geopandas has no torch. Installing the GIS stack into the model environment to
work around that would change the user's environment for a convenience, and this
project has already paid for treating environment differences as noise.

So the model process stays in its own interpreter and this one stays in its own.
JSON lines in, JSON lines out:

    -> {"id": 1, "tool": "load_data", "arguments": {...}}
    <- {"id": 1, "observation": "...", "failed": false}

The state lives here, which is what makes the conversation sequential: step N+1
sees what step N stored, exactly as it does inside a normal run. A protocol that
forgot that would let a model "recover" from an error it never saw.

Reads until stdin closes. One tool error is a message, never a crash -- a dead
server would look like a model that stopped calling tools.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from strata.train.executor import render_visible


def load_catalog_tools():
    from strata.train.build import fresh_tools

    return fresh_tools()


def build_synthetic_tools(records: list[dict]):
    from strata.geobench.tools import benchmark_state
    from strata.train.synth import SynthTableSpec, fresh_synth_tools

    if not isinstance(records, list) or not records:
        raise ValueError("synthetic profile requires at least one table spec")
    specs = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("synthetic table spec must be an object")
        specs.append(SynthTableSpec(
            handle=str(record["handle"]),
            columns=tuple(str(column) for column in record["columns"]),
            n_rows=int(record["n_rows"]),
            seed=int(record["seed"]),
            kind=str(record.get("kind", "stats")),
            id_column=record.get("id_column"),
            variant_family=str(record.get("variant_family", "synth/v1")),
            entity_namespace=record.get("entity_namespace"),
        ))
    state = benchmark_state()
    return fresh_synth_tools(specs, state), state


def build_real_tools(records: list[dict], sources_dir: str):
    """The real-data profile: the published-table loader plus the tabular tools.

    The snapshots are read here, in the process that has geopandas, and their
    hashes are re-checked -- a server pointed at a different snapshot would
    otherwise score the model against data the sample never described.
    """
    from pathlib import Path

    from strata.geobench.tools import benchmark_state
    from strata.train.real_data import RealTableSpec, fresh_real_tools

    if not isinstance(records, list) or not records:
        raise ValueError("real profile requires at least one table spec")
    specs = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("real table spec must be an object")
        specs.append(RealTableSpec.from_dict(record))
    state = benchmark_state()
    return fresh_real_tools(specs, state, Path(sources_dir)), state


def real_fill_rate(records: list[dict], sources_dir: str, data_frame: str, geo_frame: str,
                   value_column: str, data_handle: str, geo_handle: str,
                   pair: tuple[str, str]) -> float:
    from strata.train.reference import fill_rate

    tools, state = build_real_tools(records, sources_dir)
    tools["load_real_table"].execute(handle=data_handle, output_dataframe_name=data_frame)
    tools["load_real_table"].execute(handle=geo_handle, output_dataframe_name=geo_frame)
    tools["merge_dataframes"].execute(
        dataframe_name=data_frame, geodataframe_name=geo_frame,
        statkey=pair[0], geokey=pair[1], output_dataframe_name="__score_probe__",
    )
    return fill_rate((state.get("data_store") or {}).get("__score_probe__"), value_column)


def synthetic_fill_rate(records: list[dict], data_frame: str, geo_frame: str,
                        value_column: str, data_handle: str, geo_handle: str,
                        pair: tuple[str, str]) -> float:
    from strata.train.reference import fill_rate

    tools, state = build_synthetic_tools(records)
    tools["load_synthetic_table"].execute(
        handle=data_handle, output_dataframe_name=data_frame,
    )
    tools["load_synthetic_table"].execute(
        handle=geo_handle, output_dataframe_name=geo_frame,
    )
    tools["merge_dataframes"].execute(
        dataframe_name=data_frame, geodataframe_name=geo_frame,
        statkey=pair[0], geokey=pair[1], output_dataframe_name="__score_probe__",
    )
    return fill_rate((state.get("data_store") or {}).get("__score_probe__"), value_column)


def f2_fill_rates() -> dict:
    """How much of each F2 case's value column survives each candidate key pair.

    Computed here rather than in the driving process because this is the process
    with geopandas. The alternative -- importing the tool layer into the model's
    interpreter just to measure a join -- is what broke the first baseline run:
    the model process pulled in the whole GIS stack it exists to avoid.
    """
    from strata.train.reference import fill_rate
    from strata.train.templates import F2_CASES

    tools, state = load_catalog_tools()
    rates: dict[str, float] = {}
    for case in F2_CASES:
        tools["load_data"].execute(dataset=case.data_dataset,
                                   output_dataframe_name=case.data_frame)
        tools["load_geodata"].execute(geodataset=case.geo_dataset,
                                      output_geodataframe_name=case.geo_frame)
        pairs = [(case.data_key, case.geo_key), *case.wrong_pairs]
        for pair in pairs:
            tools["merge_dataframes"].execute(
                dataframe_name=case.data_frame, geodataframe_name=case.geo_frame,
                statkey=pair[0], geokey=pair[1], output_dataframe_name="probe")
            rates[f"{case.case_id}|{pair[0]}|{pair[1]}"] = fill_rate(
                state["data_store"]["probe"], case.value_column)
    return rates


def main() -> int:
    tools = None
    _state = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as exc:
            print(json.dumps({"id": None, "observation": f"bad request: {exc}",
                              "failed": True}), flush=True)
            continue

        if request.get("op") == "configure_synthetic":
            try:
                tools, _state = build_synthetic_tools(request.get("specs") or [])
                response = {"id": request.get("id"), "configured": "synthetic",
                            "failed": False}
            except Exception as exc:  # noqa: BLE001 - report bad profile data
                response = {"id": request.get("id"), "configured": None,
                            "observation": f"Error: {type(exc).__name__}: {exc}",
                            "failed": True}
            print(json.dumps(response, ensure_ascii=False), flush=True)
            continue

        if request.get("op") == "configure_real":
            try:
                tools, _state = build_real_tools(request.get("specs") or [],
                                                 str(request.get("sources_dir") or ""))
                response = {"id": request.get("id"), "configured": "real", "failed": False}
            except Exception as exc:  # noqa: BLE001 - report bad profile data
                response = {"id": request.get("id"), "configured": None,
                            "observation": f"Error: {type(exc).__name__}: {exc}",
                            "failed": True}
            print(json.dumps(response, ensure_ascii=False), flush=True)
            continue

        if request.get("op") == "real_fill_rate":
            try:
                rate = real_fill_rate(
                    request.get("specs") or [], str(request.get("sources_dir") or ""),
                    str(request["data_frame"]), str(request["geo_frame"]),
                    str(request["value_column"]), str(request["data_handle"]),
                    str(request["geo_handle"]),
                    (str(request["statkey"]), str(request["geokey"])),
                )
                response = {"id": request.get("id"), "fill_rate": rate, "failed": False}
            except Exception as exc:  # noqa: BLE001 - report bad score inputs
                response = {"id": request.get("id"), "fill_rate": 0.0,
                            "observation": f"Error: {type(exc).__name__}: {exc}",
                            "failed": True}
            print(json.dumps(response, ensure_ascii=False), flush=True)
            continue

        if request.get("op") == "configure_catalog":
            try:
                tools, _state = load_catalog_tools()
                response = {"id": request.get("id"), "configured": "catalog",
                            "failed": False}
            except Exception as exc:  # noqa: BLE001 - report setup failures
                response = {"id": request.get("id"), "configured": None,
                            "observation": f"Error: {type(exc).__name__}: {exc}",
                            "failed": True}
            print(json.dumps(response, ensure_ascii=False), flush=True)
            continue

        if request.get("op") == "synthetic_fill_rate":
            try:
                rate = synthetic_fill_rate(
                    request.get("specs") or [],
                    str(request["data_frame"]), str(request["geo_frame"]),
                    str(request["value_column"]), str(request["data_handle"]),
                    str(request["geo_handle"]),
                    (str(request["statkey"]), str(request["geokey"])),
                )
                response = {"id": request.get("id"), "fill_rate": rate,
                            "failed": False}
            except Exception as exc:  # noqa: BLE001 - report bad score inputs
                response = {"id": request.get("id"), "fill_rate": 0.0,
                            "observation": f"Error: {type(exc).__name__}: {exc}",
                            "failed": True}
            print(json.dumps(response, ensure_ascii=False), flush=True)
            continue

        if request.get("op") == "fill_rates":
            try:
                print(json.dumps({"id": request.get("id"), "fill_rates": f2_fill_rates()},
                                 ensure_ascii=False), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"id": request.get("id"), "fill_rates": {},
                                  "error": f"{type(exc).__name__}: {exc}"}), flush=True)
            continue

        call_id = request.get("id")
        name = request.get("tool")
        arguments = request.get("arguments") or {}
        tool = tools.get(name) if tools is not None else None

        if tool is None:
            raw, failed = f"Error: tool {name!r} is not in this profile", True
        else:
            try:
                raw = str(tool.execute(**arguments))
                from corecoder.tool_errors import is_error_result
                failed = is_error_result(raw)
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                raw, failed = f"Error: {type(exc).__name__}: {exc}", True

        print(json.dumps({"id": call_id, "observation": render_visible(raw),
                          "raw_length": len(raw), "failed": failed},
                         ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
