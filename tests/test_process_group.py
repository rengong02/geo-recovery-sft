"""Shell timeouts must take out the whole process group, not just the shell.

`subprocess.run(timeout=N)` kills the direct child and re-raises. When that
child is a shell, its own children survive as orphans: `npm run dev` leaves a
node process, `pytest --watch` keeps a watcher, a backgrounded `sleep` keeps
running. Those orphans hold ports and burn CPU long after the tool call has
returned "timed out".

The fix is to put the shell in its own session (`start_new_session=True`) so it
leads a fresh process group, then signal the *group* on timeout.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest

from corecoder.shell import run_shell

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="process groups are a POSIX concept"
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - exists but not ours
        return True
    return True


def _wait_dead(pid: int, deadline_s: float = 5.0) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def test_timeout_kills_grandchildren(tmp_path):
    """The regression this file exists for: a backgrounded grandchild must die."""
    pidfile = tmp_path / "grandchild.pid"
    # The shell backgrounds a long sleep, records its pid, then waits on it.
    command = f"sleep 300 & echo $! > {pidfile}; wait"

    with pytest.raises(subprocess.TimeoutExpired):
        run_shell(command, capture_output=True, text=True, timeout=1)

    assert pidfile.exists(), "grandchild never started; test is not exercising the fix"
    grandchild = int(pidfile.read_text().strip())

    assert _wait_dead(grandchild), (
        f"grandchild pid {grandchild} survived the timeout -- "
        "the process group was not cleaned up"
    )


def test_timeout_still_raises_timeout_expired():
    """Callers (tools/bash.py) catch TimeoutExpired; keep that contract."""
    with pytest.raises(subprocess.TimeoutExpired):
        run_shell("sleep 300", capture_output=True, text=True, timeout=1)


def test_completed_process_contract_is_unchanged():
    """The no-timeout path must behave exactly like subprocess.run."""
    proc = run_shell("echo hello", capture_output=True, text=True, timeout=30)

    assert isinstance(proc, subprocess.CompletedProcess)
    assert proc.returncode == 0
    assert proc.stdout.strip() == "hello"


def test_stdout_and_stderr_are_both_captured():
    proc = run_shell("echo out; echo err 1>&2", capture_output=True, text=True, timeout=30)

    assert proc.stdout.strip() == "out"
    assert proc.stderr.strip() == "err"


def test_check_true_raises_called_process_error():
    """`check=True` must keep working after the Popen rewrite."""
    with pytest.raises(subprocess.CalledProcessError):
        run_shell("exit 3", check=True, capture_output=True, text=True, timeout=30)


def test_cwd_is_honoured(tmp_path):
    proc = run_shell("pwd", capture_output=True, text=True, timeout=30, cwd=tmp_path)

    assert Path(proc.stdout.strip()).resolve() == tmp_path.resolve()


def test_fast_command_is_not_disturbed():
    """A command that finishes well inside the timeout returns normally."""
    start = time.monotonic()
    proc = run_shell("echo quick", capture_output=True, text=True, timeout=30)

    assert proc.stdout.strip() == "quick"
    assert time.monotonic() - start < 5


def test_nonzero_exit_is_reported_not_raised():
    proc = run_shell("exit 7", capture_output=True, text=True, timeout=30)

    assert proc.returncode == 7


def test_kill_is_forceful_enough_for_a_stubborn_child(tmp_path):
    """The group kill must stop a child that ignores SIGTERM.

    `sleep` dies on SIGTERM, so this runs a shell that traps TERM and then
    blocks -- only SIGKILL gets rid of it.
    """
    pidfile = tmp_path / "stubborn.pid"
    command = f"trap '' TERM; echo $$ > {pidfile}; while true; do sleep 0.1; done"

    with pytest.raises(subprocess.TimeoutExpired):
        run_shell(command, capture_output=True, text=True, timeout=1)

    assert pidfile.exists(), "shell never started; test is not exercising the fix"
    pid = int(pidfile.read_text().strip())
    assert _wait_dead(pid), f"pid {pid} ignored SIGTERM and was not SIGKILLed"
