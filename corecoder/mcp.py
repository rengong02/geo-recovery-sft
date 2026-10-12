"""MCP stdio client, distilled to the slice of the protocol an agent uses.

Servers are configured in ~/.corecoder/mcp.json:

    {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "some-fs-server", "/tmp"]}}}

Each server is a subprocess speaking JSON-RPC 2.0, one message per line over
stdin/stdout. Startup handshakes (`initialize`), pulls `tools/list`, and every
remote tool joins the agent as `mcp__<server>__<tool>`, so consent, hooks and
the main loop treat them exactly like the built-ins. A server that hangs or
dies fails that one call as an ordinary error string; it never kills the loop.
"""

import atexit
import json
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from . import __version__
from .process import process_group_of, reap, spawn_in_own_group, terminate_process_group
from .tools.base import Tool

log = logging.getLogger(__name__)

CONFIG_FILE = Path.home() / ".corecoder" / "mcp.json"
PROTOCOL_VERSION = "2025-06-18"
INIT_TIMEOUT = 15  # seconds for initialize + tools/list at startup
CALL_TIMEOUT = 60  # seconds for one tools/call


class MCPError(RuntimeError):
    """Transport or protocol failure talking to one server."""


class MCPClient:
    """One stdio server process: handshake, list, call, shut down.

    A daemon thread owns stdout and parks each response under its request id,
    so calls from parallel tool execution match their own replies; the write
    lock keeps two threads' requests from interleaving on stdin.
    """

    def __init__(self, name: str, command: str, args: list = (), env: dict | None = None):
        self.name = name
        self.call_timeout = CALL_TIMEOUT
        self._closed = False
        self._proc = spawn_in_own_group(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env={**os.environ, **(env or {})},  # servers inherit the user env, config overrides
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # line buffered: the transport is newline-delimited JSON
        )
        # Captured now: getpgid stops answering once the leader is reaped, and
        # close() deliberately waits for the leader.
        self._pgid = process_group_of(self._proc)
        self._next_id = 0
        self._dead: MCPError | None = None
        self._responses: dict[int, dict] = {}
        self._cond = threading.Condition()
        self._write_lock = threading.Lock()
        threading.Thread(target=self._read_loop, daemon=True).start()
        try:
            self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "corecoder", "version": __version__},
            }, INIT_TIMEOUT)
            self._notify("notifications/initialized")
            listed = self._request("tools/list", {}, INIT_TIMEOUT)
            self.tools = [MCPTool(self, t) for t in listed.get("tools", [])]
        except BaseException:
            self.close()  # a half-started server must not leak
            raise

    def call_tool(self, tool_name: str, arguments: dict,
                  deadline_at: float | None = None) -> str:
        """Run one remote tool and return its text content.

        `deadline_at` is the run's absolute deadline. The call gets the smaller
        of its own allowance and what is left of the run -- without that, a task
        bounded at five seconds still sat here for the fixed sixty waiting to
        find out, which makes the task's own deadline a decoration.
        """
        timeout = self.call_timeout
        if deadline_at is not None:
            left = deadline_at - time.monotonic()
            if left <= 0:
                # Identifiable as a deadline stop rather than a slow server: the
                # two call for different responses, and a report needs to say
                # which happened.
                raise MCPError(
                    f"{tool_name!r} was not started: the run's deadline had already passed"
                )
            timeout = min(timeout, left)
        result = self._request(
            "tools/call", {"name": tool_name, "arguments": arguments}, timeout
        )
        text = "\n".join(
            part.get("text", "")
            for part in result.get("content", [])
            if part.get("type") == "text"
        )
        if result.get("isError"):
            raise MCPError(text or f"{tool_name} reported an error")
        return text or json.dumps(result)  # non-text content: hand the model the raw result

    def close(self):
        """Shut the server down. Safe to call twice.

        The server leads its own process group, so the signal reaches anything
        it spawned -- an MCP server that shells out to GDAL, say. Closing stdin
        first gives it the EOF it is watching for; SIGTERM gives it a chance to
        flush; a SIGKILL sweep catches children that ignored the SIGTERM even
        though their parent honoured it.

        The pgid is captured at spawn because `getpgid` stops answering once the
        leader is reaped -- and the leader is exactly what we wait on.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        terminate_process_group(self._proc, sig=signal.SIGTERM, pgid=self._pgid)
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Sweep regardless: a child that ignores SIGTERM outlives a parent that
        # does not. An empty group makes this a no-op (ProcessLookupError).
        terminate_process_group(self._proc, sig=signal.SIGKILL, pgid=self._pgid)
        reap(self._proc)

    def _notify(self, method: str):
        try:
            self._write({"jsonrpc": "2.0", "method": method, "params": {}})
        except MCPError:
            pass  # a notification has no reply to lose; the next request meets the dead server

    def _request(self, method: str, params: dict, timeout: float) -> dict:
        with self._cond:
            if self._dead is not None:
                raise self._dead
            self._next_id += 1
            req_id = self._next_id
        self._write({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        with self._cond:
            while req_id not in self._responses:
                if self._dead is not None:
                    raise self._dead
                left = deadline - time.monotonic()
                if left <= 0:
                    raise MCPError(f"MCP server {self.name!r} gave no answer to {method} in {timeout:g}s")
                self._cond.wait(left)
            msg = self._responses.pop(req_id)
        if "error" in msg:
            raise MCPError(f"MCP server {self.name!r} rejected {method}: {msg['error'].get('message', msg['error'])}")
        return msg.get("result", {})

    def _write(self, msg: dict):
        try:
            with self._write_lock:
                self._proc.stdin.write(json.dumps(msg) + "\n")
                self._proc.stdin.flush()
        except OSError as e:
            raise MCPError(f"MCP server {self.name!r} is not writable: {e}") from e

    def _read_loop(self):
        try:
            for line in self._proc.stdout:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("MCP server %r sent a non-JSON line, skipped", self.name)
                    continue
                if "id" not in msg:
                    continue  # server notification: nothing here needs an answer
                with self._cond:
                    self._responses[msg["id"]] = msg
                    self._cond.notify_all()
        except (OSError, ValueError):
            pass
        with self._cond:
            # stdout closing means the server is gone for good
            if self._dead is None:
                self._dead = MCPError(f"MCP server {self.name!r} exited")
            self._cond.notify_all()


class MCPTool(Tool):
    """A tool living in an MCP server, registered under a mcp__ name so it can
    never collide with a built-in. Side effects are unknown, so it stays out of
    Permission.READ_ONLY and the consent gate asks first, like any mutating tool.
    """

    #: Enforceable: the call waits on another process with a bounded timeout, so
    #: it cannot outlive the wait it was given.
    deadline_mode = "enforced"

    def __init__(self, client: MCPClient, spec: dict):
        self._client = client
        self._remote_name = spec["name"]
        self.name = f"mcp__{client.name}__{spec['name']}"
        self.description = spec.get("description") or ""
        self.parameters = spec.get("inputSchema") or {"type": "object", "properties": {}}

    def execute(self, **kwargs) -> str:
        return self._client.call_tool(self._remote_name, kwargs)

    def execute_with_deadline(self, deadline_at: float | None, **kwargs) -> str:
        """A remote call is exactly the kind of tool that needs a bound: it waits
        on another process, and the run's remaining time is the only thing that
        should decide how long."""
        return self._client.call_tool(self._remote_name, kwargs, deadline_at=deadline_at)


_live_clients: list[MCPClient] = []


def load_mcp_tools(path: Path = CONFIG_FILE) -> list[Tool]:
    """Start the configured servers and return their tools. A missing file
    means no MCP; a broken file or a server that won't start gets one clear
    warning and the agent carries on with whatever loaded."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as e:
        log.warning("ignoring %s: %s", path, e)
        return []
    tools: list[Tool] = []
    for name, spec in (data.get("mcpServers") or {}).items():
        try:
            client = MCPClient(name, spec["command"], spec.get("args", []), spec.get("env"))
        except (MCPError, OSError, KeyError, TypeError) as e:
            log.warning("MCP server %r skipped: %s", name, e)
            continue
        _live_clients.append(client)
        tools.extend(client.tools)
    return tools


def shutdown_mcp_servers() -> int:
    """Close every server `load_mcp_tools` started.

    Returns the number of servers *processed*, which is not the number that
    shut down cleanly: a failing `close()` is logged and the count still
    includes it. Reading this as "servers closed" would overstate the result.

    `atexit` covers a clean interpreter exit, but a benchmark that runs many
    tasks wants to release the servers *between* phases -- and anything running
    the loop as a library (or under a test runner) wants it deterministically
    rather than whenever the process happens to end. Safe to call twice.

    One server failing to close must not strand the rest, so each is isolated
    and the failure is logged rather than raised. Not concurrency-safe: do not
    call it while another thread is inside `load_mcp_tools`.
    """
    closed = 0
    while _live_clients:
        client = _live_clients.pop()
        try:
            client.close()
        except Exception:
            log.warning("MCP server %r did not shut down cleanly", client.name, exc_info=True)
        closed += 1
    return closed


@atexit.register
def _shutdown():
    """Backstop for processes that never call `shutdown_mcp_servers`."""
    shutdown_mcp_servers()
