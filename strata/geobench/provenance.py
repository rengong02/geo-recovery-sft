"""What produced a run's artifacts.

An artifact that cannot name its inputs is an anecdote. Concretely: a rerun
produced a file that still showed 30 `fmt` errors while the working tree already
had the fix applied, and **nothing in the file said which checkout, revision or
patch state it came from** -- so the result could not be trusted either way, and
the run had to be repeated to settle it.

Every row a runner writes now carries this instead. It is a few hundred bytes
against a run that costs minutes and real money.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .tools import geobenchx_root


def _git(cwd: Path, *args: str) -> str | None:
    """Run a git command, or return None if git or the repo is unavailable."""
    try:
        proc = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            check=False, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_revision(repo: Path) -> str | None:
    return _git(repo, "rev-parse", "HEAD")


def working_tree_digest(repo: Path) -> str | None:
    """A hash of the uncommitted diff, so "dirty" is a specific state.

    `None` when the tree is clean or git is unavailable. The digest is over
    `git diff HEAD` -- tracked modifications only. A patch that has been applied
    and *committed* shows up in the revision instead, so between the two the
    state is pinned.
    """
    diff = _git(repo, "diff", "HEAD")
    if diff is None:
        return None
    if not diff:
        return ""
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()[:16]


def matplotlib_backend() -> str:
    """The backend in force *now*.

    Worth recording because the first baseline's 15 GUI-thread failures were a
    property of the backend, and nothing in that run's output said which one it
    was. Importing matplotlib only to ask is fine -- it is already imported by
    the time a GIS run starts.
    """
    try:
        import matplotlib

        return matplotlib.get_backend()
    except Exception as e:  # noqa: BLE001 - provenance must never break a run
        return f"unknown ({type(e).__name__})"


def patch_state(patches_dir: Path, upstream: Path) -> dict[str, bool]:
    """Which patches in `patches_dir` are currently applied to `upstream`.

    `git apply --reverse --check` succeeds exactly when a patch is already in
    the tree, which is the cheapest honest test: it asks the same program that
    would apply the patch, rather than pattern-matching the source.
    """
    state: dict[str, bool] = {}
    if not patches_dir.is_dir():
        return state
    for patch in sorted(patches_dir.glob("*.patch")):
        proc = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch.resolve())],
            cwd=upstream, capture_output=True, text=True, check=False,
        )
        state[patch.name] = proc.returncode == 0
    return state


def geobenchx_provenance(repo_root: Path) -> dict:
    """Everything needed to say which upstream state a run used."""
    upstream = geobenchx_root()
    return {
        "geobenchx_root": str(upstream),
        "geobenchx_head": git_revision(upstream),
        "geobenchx_dirty": working_tree_digest(upstream),
        "patches": patch_state(repo_root / "patches", upstream),
        "matplotlib_backend": matplotlib_backend(),
    }
