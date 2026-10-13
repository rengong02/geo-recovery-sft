"""Subprocess execution and CPU tool preflight shared by training runners."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def baseline_command(samples: Path, output: Path, model: Path, kind: str,
                     manifest: Path, tool_python: str, budget_seconds: float) -> list[str]:
    return [sys.executable, "-u", str(ROOT / "scripts" / "run_baseline.py"),
            "--samples", str(samples), "--out", str(output), "--model", str(model),
            "--evaluation-kind", kind, "--device", "cuda", "--attn-implementation", "sdpa",
            "--tool-python", tool_python, "--max-turns", "6", "--max-new-tokens", "1024",
            "--resume", "--source-manifest", str(manifest),
            "--budget-seconds", str(budget_seconds)]


def run_logged(command: list[str], path: Path, lock_descriptor: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        print("running:", " ".join(command), flush=True)
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   pass_fds=(lock_descriptor,))
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def check_tool_runtime(samples: list[dict], tool_python: str) -> None:
    from scripts.run_baseline import ToolServer

    server = ToolServer(tool_python)
    try:
        sample = samples[0]
        server.configure_for_sample(sample)
        for step in sample["steps"]:
            result = server.call(step["tool"], step["arguments"])
            if bool(result.get("failed")) != bool(step["failed"]):
                raise ValueError(f"CPU tool preflight disagrees with {sample['sample_id']}")
    finally:
        server.close()
