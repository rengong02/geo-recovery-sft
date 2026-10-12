"""Session-scoped undo for file mutations.

edit_file and write_file record a checkpoint before touching a file; /undo
pops the latest one and restores the previous bytes (or removes the file if
it did not exist). In-memory only: undo history dies with the process, and
bash side effects are not tracked, only the two file-writing tools.
"""

from pathlib import Path

# (path, prior bytes or None if the file did not exist)
_stack: list[tuple[str, bytes | None]] = []


def record(path: Path) -> None:
    """Capture the pre-mutation state of path. Call right before writing."""
    _stack.append((str(path), path.read_bytes() if path.exists() else None))


def undo() -> str:
    """Restore the most recent checkpoint."""
    if not _stack:
        return "Nothing to undo."
    path_str, prior = _stack.pop()
    p = Path(path_str)
    if prior is None:
        p.unlink(missing_ok=True)
        return f"Removed {path_str} (created this session)."
    # the parent tree may be gone by now: bash side effects are untracked, so
    # recreate it instead of dying on FileNotFoundError
    recreated = not p.parent.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(prior)
    if recreated:
        return f"Restored {path_str} (recreated missing parent directories)."
    return f"Restored {path_str}."


def pending() -> int:
    return len(_stack)


def clear() -> None:
    _stack.clear()
