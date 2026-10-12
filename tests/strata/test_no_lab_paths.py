"""The repository must not carry the lab machine's absolute path root.

The frozen reports under `evals/train/` were produced on a shared machine, so
they recorded that machine's paths -- including a home directory that identifies
a person. Redacting them once is not enough: the rule was re-broken three times
while it was being applied, twice by the very files that documented it, because
writing the string down is the natural way to explain it.

So it is a test rather than a convention. The prefix is read from the redaction
script rather than repeated here, for the same reason.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.redact_lab_paths import LAB_PREFIX, PLACEHOLDER

ROOT = Path(__file__).resolve().parents[2]


def _kept_text_files() -> list[Path]:
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout
    others = subprocess.run(["git", "ls-files", "-o", "--exclude-standard", "-z"],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout
    names = sorted({name for name in (listed + others).split("\0") if name})
    return [ROOT / name for name in names]


def test_the_redaction_rule_is_not_itself_the_placeholder():
    """A tool whose pattern equals its replacement silently reports clean."""
    assert LAB_PREFIX != PLACEHOLDER
    assert LAB_PREFIX.startswith("/")


def test_no_kept_file_contains_the_lab_path_root():
    offenders = []
    for path in _kept_text_files():
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if LAB_PREFIX in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"lab path root present in: {offenders}"


def test_the_redaction_check_gate_passes():
    """The published form of the same check, so a clone can run exactly it."""
    proc = subprocess.run([sys.executable, "scripts/redact_lab_paths.py", "--check"],
                          cwd=ROOT, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
