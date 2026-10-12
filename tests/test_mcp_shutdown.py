"""Shutting MCP servers down: deterministically, and all the way down.

Two gaps in the upstream lifecycle:

1. Cleanup was `atexit` only. A runner that evaluates many tasks wants to
   release the servers between phases, and anything driving the loop as a
   library wants it deterministically rather than whenever the interpreter
   happens to exit.
2. `close()` signalled the direct child only. An MCP server that shells out --
   to GDAL, to a language server -- leaves those processes behind, the same
   orphan problem the shell tool had.

The fake server here records its own pid and starts a long-lived grandchild, so
both halves are checkable.
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

from corecoder import mcp
from corecoder.mcp import load_mcp_tools, shutdown_mcp_servers

# Writes its own pid, backgrounds a grandchild, then serves MCP over stdio.
FAKE_SERVER = """
import json, os, subprocess, sys

pidfile = sys.argv[1]
gcfile = sys.argv[2]

# a grandchild that outlives a naive kill of the server itself
gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
with open(gcfile, "w") as f:
    f.write(str(gc.pid))
with open(pidfile, "w") as f:
    f.write(str(os.getpid()))

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8", newline="\\n")

TOOLS = [{"name": "echo", "description": "Echo", "inputSchema":
          {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]

for line in sys.stdin:
    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "id" not in req:
        continue
    if req.get("method") == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake", "version": "0.1"}}
    elif req.get("method") == "tools/list":
        result = {"tools": TOOLS}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}) + "\\n")
    sys.stdout.flush()
"""


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    shutdown_mcp_servers()


@pytest.fixture
def two_servers(tmp_path):
    """Two configured servers, each recording its pid and grandchild pid."""
    script = tmp_path / "server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")

    servers = {}
    for name in ("alpha", "beta"):
        servers[name] = {
            "pidfile": tmp_path / f"{name}.pid",
            "gcfile": tmp_path / f"{name}.gc",
        }
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {
        name: {"command": sys.executable,
               "args": [str(script), str(v["pidfile"]), str(v["gcfile"])]}
        for name, v in servers.items()
    }}), encoding="utf-8")
    return cfg, servers


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_dead(pid: int, deadline_s: float = 5.0) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def _read_pid(path: Path) -> int:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text().strip())
        time.sleep(0.05)
    raise AssertionError(f"{path} was never written; the fake server did not start")


def test_shutdown_closes_every_configured_server(two_servers):
    cfg, _ = two_servers
    load_mcp_tools(cfg)

    assert shutdown_mcp_servers() == 2


def test_shutdown_is_safe_to_call_twice(two_servers):
    cfg, _ = two_servers
    load_mcp_tools(cfg)

    assert shutdown_mcp_servers() == 2
    assert shutdown_mcp_servers() == 0  # nothing left to close


def test_shutdown_stops_the_server_process(two_servers):
    cfg, servers = two_servers
    load_mcp_tools(cfg)
    pid = _read_pid(servers["alpha"]["pidfile"])

    shutdown_mcp_servers()

    assert _wait_dead(pid), f"server pid {pid} survived shutdown"


def test_shutdown_reaches_the_servers_own_children(two_servers):
    """The orphan fix: a server's grandchild must go too, not just the server."""
    cfg, servers = two_servers
    load_mcp_tools(cfg)
    grandchild = _read_pid(servers["alpha"]["gcfile"])

    shutdown_mcp_servers()

    assert _wait_dead(grandchild), (
        f"grandchild pid {grandchild} survived -- close() only killed the server"
    )


def test_shutdown_clears_the_registry_so_a_reload_starts_fresh(two_servers):
    """Otherwise every load would leak another set of servers."""
    cfg, _ = two_servers
    load_mcp_tools(cfg)
    shutdown_mcp_servers()

    assert mcp._live_clients == []

    load_mcp_tools(cfg)  # a second round must start clean, not stack up
    assert len(mcp._live_clients) == 2


def test_a_server_that_already_exited_is_not_an_error(two_servers):
    cfg, _ = two_servers
    load_mcp_tools(cfg)
    for client in mcp._live_clients:
        client._proc.kill()
        client._proc.wait(timeout=5)

    assert shutdown_mcp_servers() == 2  # still counted, but close() is a no-op


# --- the leader exiting does not mean its children did -----------------------

STUBBORN_SERVER = """
import json, os, subprocess, sys

pidfile, gcfile = sys.argv[1], sys.argv[2]

# A grandchild that ignores SIGTERM. It writes its pid only *after* installing
# the handler, so the test can wait for that file and be sure it is in place.
child_code = (
    "import os, signal, time\\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
    f"open({gcfile!r}, 'w').write(str(os.getpid()))\\n"
    "time.sleep(300)\\n"
)
child = subprocess.Popen([sys.executable, "-c", child_code])
with open(pidfile, "w") as f:
    f.write(str(os.getpid()))

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8", newline="\\n")
for line in sys.stdin:
    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "id" not in req:
        continue
    if req.get("method") == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "s", "version": "1"}}
    elif req.get("method") == "tools/list":
        result = {"tools": []}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}) + "\\n")
    sys.stdout.flush()
# default disposition: SIGTERM exits this process, leaving the child behind
"""


def test_close_sweeps_children_that_outlive_their_parent(tmp_path):
    """The case a plain `terminate()` + `wait()` misses.

    This server honours SIGTERM and exits; its child ignores SIGTERM and keeps
    running. The leader exiting used to look like a clean shutdown.
    """
    script = tmp_path / "stubborn.py"
    script.write_text(STUBBORN_SERVER, encoding="utf-8")
    pidfile, gcfile = tmp_path / "s.pid", tmp_path / "s.gc"
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"stubborn": {
        "command": sys.executable, "args": [str(script), str(pidfile), str(gcfile)]}}}),
        encoding="utf-8")

    load_mcp_tools(cfg)
    leader = _read_pid(pidfile)
    child = _read_pid(gcfile)  # exists only once SIG_IGN is installed

    shutdown_mcp_servers()

    assert _wait_dead(leader), "the server itself should have exited"
    assert _wait_dead(child), (
        f"child pid {child} survived: close() stopped at the leader"
    )


def test_one_failing_server_does_not_strand_the_others(two_servers):
    """A close() that blows up must not leave the rest running."""
    cfg, servers = two_servers
    load_mcp_tools(cfg)

    class Boom:
        name = "boom"
        _proc = mcp._live_clients[0]._proc  # keep a real pid so we can assert

        def close(self):
            raise RuntimeError("close exploded")

    mcp._live_clients.append(Boom())

    assert shutdown_mcp_servers() == 3  # counted, failure swallowed
    assert mcp._live_clients == []

    pid = _read_pid(servers["alpha"]["pidfile"])
    assert _wait_dead(pid), "a sibling server was stranded by the failure"
