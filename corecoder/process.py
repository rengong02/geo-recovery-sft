"""Stopping a subprocess *and everything it spawned*.

Two places in CoreCoder spawn child processes: the shell tool and the MCP
client. Both used to kill only the direct child, which is not enough -- the
child is usually an interpreter or a shell with its own children:

* a shell running ``npm run dev`` leaves a node process holding a port
* a backgrounded job in a tool command keeps running
* an MCP server that shells out to GDAL leaves the GDAL process behind

The orphan outlives the call that started it, so a run that reports "timed out"
or "server closed" can leave the machine busier than before it started.

Putting the child in its own session makes it a process-group leader, and
signalling the *group* reaches the whole tree. This module is the single place
that knows how; callers just spawn through `spawn_in_own_group` and stop
through `terminate_process_group`.
"""

import logging
import os
import signal
import subprocess

log = logging.getLogger(__name__)


def spawn_in_own_group(
    argv: str | list[str], *, shell: bool = False, **kwargs
) -> subprocess.Popen:
    """`subprocess.Popen` with the child leading a fresh process group.

    ``start_new_session`` is POSIX-only and is silently ignored on Windows,
    where `terminate_process_group` therefore degrades to a single-process kill.
    """
    return subprocess.Popen(argv, shell=shell, start_new_session=True, **kwargs)


def process_group_of(proc: subprocess.Popen) -> int | None:
    """The group `proc` leads, or None if it does not lead a safe one.

    Callers that need to sweep a group *after* the leader may have exited must
    capture this first -- `getpgid` stops answering once the process is reaped.

    Two conditions, both load-bearing:
    ``pgid == proc.pid`` proves the child is a group leader, so it came from
    `spawn_in_own_group` and not from some other library's process; and
    ``pgid != os.getpgrp()`` keeps us from signalling our own group, which would
    take the agent process down with it.
    """
    if os.name != "posix":
        return None
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return None
    if pgid == proc.pid and pgid != os.getpgrp():
        return pgid
    return None


def terminate_process_group(
    proc: subprocess.Popen, sig: int = signal.SIGKILL, pgid: int | None = None
) -> bool:
    """Stop `proc` and every process in its group.

    Returns True when a whole group was signalled, False when only the direct
    child could be stopped -- already reaped, not a group leader, or a platform
    without process groups. The False case is logged at debug level rather than
    silently collapsed, so "cleaned up the group" and "only got the one
    process" are distinguishable in a trace.

    Pass `pgid` (from `process_group_of`) to sweep a group whose leader has
    already exited but whose children are still running.
    """
    if pgid is None:
        pgid = process_group_of(proc)
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False  # raced with the whole group exiting
        except OSError as e:
            log.debug("killpg(%d) failed, falling back to the direct child: %s", pgid, e)
    # Windows, or a child that never led its own group: best effort.
    # A complete Windows fix needs CREATE_NEW_PROCESS_GROUP plus
    # `taskkill /T /F`; that path is untested here and deliberately not claimed.
    log.debug("no process group for pid %s; killing the direct child only", proc.pid)
    proc.kill()
    return False


def reap(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    """Wait for a process we just killed, so it does not linger as a zombie."""
    try:
        proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):  # pragma: no cover - SIGKILL wins
        pass
