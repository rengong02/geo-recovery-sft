#!/usr/bin/env python
"""A stdio MCP server that answers the handshake and then never answers a call.

The point of the fixture is the silence: against a server that simply never
replies, the only way a tool call returns promptly is if it is bounded by the
*run's* deadline rather than by the client's fixed allowance. A mock client
would prove the timeout arithmetic and nothing about whether a real wait is
actually interruptible.
"""

from __future__ import annotations

import json
import sys

TOOL = {
    "name": "hang",
    "description": "answers the handshake, then never answers again",
    "inputSchema": {"type": "object", "properties": {}},
}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        method = message.get("method")

        if method == "initialize":
            result = {
                "protocolVersion": message.get("params", {}).get("protocolVersion", "2024-11-05"),
                "capabilities": {},
                "serverInfo": {"name": "fake-hanging-server", "version": "0"},
            }
        elif method == "tools/list":
            result = {"tools": [TOOL]}
        else:
            # `tools/call`, and `notifications/initialized`: nothing comes back.
            continue

        if "id" in message:
            sys.stdout.write(
                json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}) + "\n"
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
