"""Base class for all tools, plus the workspace boundary check."""

import os
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

# File-mutating tools (write_file/edit_file) serialize on this, so a parallel
# batch can't interleave read-modify-write on the same file and lose an edit.
FILE_MUTATION_LOCK = threading.Lock()

# --- workspace boundary -----------------------------------------------------
#
# Upstream, file tools resolve whatever path the model hands them and use it.
# That is fine for a human-driven REPL: the person at the keyboard is the trust
# boundary and can see what the agent is about to touch. It is not fine for an
# unattended run, where the *model* chooses the path and may be steered by
# untrusted content it read a moment ago.
#
# So the boundary is opt-in rather than always-on:
#
#   * no root configured  -> upstream behaviour, byte for byte (interactive CLI)
#   * root configured     -> every path is resolved and must land inside it
#
# A runner that evaluates against a benchmark sets the root once at startup; a
# human typing at the prompt never does.

#: Distinguishes "never configured" from "configured to nothing". Without this
#: sentinel an explicit `set_workspace_root(None)` is indistinguishable from the
#: initial state, and the environment variable would keep winning -- so the
#: documented "None disables it" would be false.
_UNSET = object()

_WORKSPACE_ROOT: Path | None | object = _UNSET

ENV_VAR = "CORECODER_WORKSPACE"


def set_workspace_root(root: str | Path | None) -> None:
    """Configure the boundary. `None` disables it (the interactive default).

    An explicit call always wins over `CORECODER_WORKSPACE`; use
    `clear_workspace_root()` to hand control back to the environment.
    """
    global _WORKSPACE_ROOT
    if root is None:
        _WORKSPACE_ROOT = None
        return
    # Resolve once so that a symlinked root still matches its real children.
    _WORKSPACE_ROOT = Path(root).expanduser().resolve()


def clear_workspace_root() -> None:
    """Forget the explicit setting so `CORECODER_WORKSPACE` applies again."""
    global _WORKSPACE_ROOT
    _WORKSPACE_ROOT = _UNSET


@contextmanager
def workspace_root(root: str | Path | None):
    """Set the boundary for a block, restoring the previous setting on exit.

    A runner that scopes a boundary to one task needs the *previous* state back,
    not "the environment rule applies again": `clear_workspace_root()` would
    hand control to `CORECODER_WORKSPACE`, so a process with that variable set
    would end a task still bounded by it -- which is not what "the task's
    workspace is released" is supposed to mean.
    """
    global _WORKSPACE_ROOT
    previous = _WORKSPACE_ROOT
    set_workspace_root(root)
    try:
        yield
    finally:
        _WORKSPACE_ROOT = previous


def get_workspace_root() -> Path | None:
    """The active boundary: the explicit setting, else the environment, else None.

    The environment value is resolved **once** and cached. Resolving it on every
    call would make the boundary move whenever the process changed directory -- a
    relative `CORECODER_WORKSPACE` would quietly point somewhere new.
    """
    global _WORKSPACE_ROOT
    if _WORKSPACE_ROOT is not _UNSET:
        return _WORKSPACE_ROOT  # type: ignore[return-value]
    raw = os.getenv(ENV_VAR)
    _WORKSPACE_ROOT = Path(raw).expanduser().resolve() if raw else None
    return _WORKSPACE_ROOT


def ensure_within_workspace(p: Path) -> Path:
    """Verify an already-resolved path sits inside the workspace root.

    Raises `ValueError` with a message the model can act on. The error is a
    normal return value by the time it reaches the loop -- see the module
    docstring convention that errors never escape as exceptions past a tool.
    """
    root = get_workspace_root()
    if root is None:
        return p
    if not p.is_relative_to(root):
        raise ValueError(
            f"path {p} is outside the workspace root {root}; "
            f"file tools are restricted to that directory"
        )
    return p


def resolve_in_workspace(raw: str | Path) -> Path:
    """Resolve a model-supplied path and enforce the workspace boundary.

    With no root configured this is exactly `Path(raw).expanduser().resolve()`,
    so callers that used to do that inline keep their old behaviour.

    With a root configured, a *relative* path is anchored to the root rather
    than to the process cwd -- an unattended run should not depend on where the
    runner happened to be launched from.
    """
    root = get_workspace_root()
    if root is None:
        return Path(raw).expanduser().resolve()

    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = root / p
    try:
        p = p.resolve()
    except (RuntimeError, OSError) as e:
        # resolve() raises RuntimeError on a symlink loop (ENAMETOOLONG-ish
        # cases land here too). Surface it as a plain error string rather than
        # letting it reach the loop as a traceback.
        raise ValueError(f"could not resolve path {raw}: {e}") from e
    return ensure_within_workspace(p)


class Tool(ABC):
    """Minimal tool interface. Subclass this to add new capabilities."""

    name: str
    description: str
    parameters: dict  # JSON Schema for the function args

    #: Whether this tool may run beside its siblings when one response asks for
    #: several calls at once. The default is `True` because upstream's tools are
    #: independent and the pool is a win there -- the test suite pins that, so
    #: changing the default would be a silent behaviour change.
    #:
    #: Set it to `False` for a tool that reads or writes state its siblings may
    #: touch. Order in the assistant message is the model's *intent*, and a
    #: thread pool does not honour it: a batch holding a writer and a reader of
    #: the same key has the reader racing the writer, and losing. Measured, not
    #: theorised -- see `GeoBenchTool`, where 27 of 30 such batches failed.
    concurrency_safe: bool = True

    #: How this tool relates to the run's deadline.
    #:
    #: `"cooperative"` (the default) -- it can be *told* the deadline but cannot
    #: enforce it. An in-process function that blocks returns when it returns,
    #: and nothing outside it can stop that. The Agent records the overrun
    #: instead of pretending there wasn't one.
    #:
    #: `"enforced"` -- the tool bounds its own wait: a subprocess timeout with a
    #: process-group kill, or a network read timeout. Claiming this without
    #: doing it is worse than claiming nothing, which is why the attribute is
    #: asserted against behaviour in the tests rather than trusted.
    deadline_mode: str = "cooperative"

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Run the tool and return a text result."""
        ...

    def execute_with_deadline(self, deadline_at: float | None, **kwargs) -> str:
        """Run the tool, aware of when the run that asked for it must stop.

        **Not part of the tool schema on purpose.** The model chooses arguments;
        when the *run* ends is the harness's business, and a model deciding to
        extend its own deadline is not a thing anyone wants. So this is a
        separate entry point the Agent uses, not a parameter the model sees.

        Default: ignore it. A tool that can bound itself -- one waiting on a
        network round trip or a child process -- overrides this; a tool that
        computes for a moment and returns has nothing to bound.
        """
        return self.execute(**kwargs)

    def schema(self) -> dict:
        """OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
