import json

from scripts import replay_trajectory as replay


def test_start_at_rebuilds_prefix_but_records_only_requested_suffix(monkeypatch, tmp_path):
    source = tmp_path / "run.jsonl"
    source.write_text(json.dumps({
        "task_ID": "TASK_TEST",
        "task_text": "test replay",
        "provenance": {"evidence_schema_preview": "head_tail"},
        "final_response": "original",
        "steps": [
            {"function_name": "first", "arguments": {}},
            {"function_name": "second", "arguments": {}},
        ],
    }) + "\n", encoding="utf-8")

    state = {"data_store": {}, "image_store": [], "html_store": [], "visualize": False}
    calls = []

    class StubTool:
        def __init__(self, name):
            self.name = name

        def execute(self, **arguments):
            calls.append(self.name)
            state["data_store"][self.name] = {"value": self.name}
            return f"ran {self.name}"

    monkeypatch.setattr(replay, "benchmark_state", lambda: state)
    monkeypatch.setattr(replay, "build_geobench_tools",
                        lambda **arguments: [StubTool("first"), StubTool("second")])
    monkeypatch.setattr(replay.sys, "argv", [
        "replay", "--task-id", "TASK_TEST", "--from", str(source),
        "--out-dir", str(tmp_path / "out"), "--start-at", "1",
    ])

    assert replay.main() == 0
    record = json.loads((tmp_path / "out" / "replay.json").read_text())
    assert calls == ["first", "second"]
    assert record["steps_replayed"] == 2
    assert record["steps_recorded_from"] == 1
    assert record["steps_recorded"] == 1
    assert [step["index"] for step in record["steps"]] == [1]
    assert record["final_tables"]["first"]["type"] == "dict"


def test_start_at_rejects_out_of_range(monkeypatch, tmp_path):
    source = tmp_path / "run.jsonl"
    source.write_text(json.dumps({"task_ID": "TASK_TEST", "steps": []}) + "\n",
                      encoding="utf-8")
    monkeypatch.setattr(replay.sys, "argv", [
        "replay", "--task-id", "TASK_TEST", "--from", str(source),
        "--out-dir", str(tmp_path / "out"), "--start-at", "1",
    ])

    try:
        replay.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("out-of-range --start-at was accepted")
