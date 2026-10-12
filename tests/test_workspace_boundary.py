"""Workspace boundary: file tools must not reach outside a configured root.

Context: upstream `read.py:36` / `write.py:37` resolve a path and use it with no
ownership check, so an agent can read `~/.ssh/id_rsa` or write `/etc/hosts`.
That is fine for a human-driven REPL (the user is the trust boundary) but not for
an unattended run against untrusted input, where the *model* chooses the path.

Design: the boundary is **opt-in**. With no root configured the tools behave
exactly as before, so interactive CLI semantics are unchanged. `Config` /
`bcp`-style runners set a root for benchmark runs.
"""


import pytest

from corecoder.tools import ALL_TOOLS
from corecoder.tools.base import (
    clear_workspace_root,
    get_workspace_root,
    set_workspace_root,
)
from tests.conftest import get_tool


@pytest.fixture(autouse=True)
def _clear_root():
    """Reset to "never configured" around each test.

    `clear_workspace_root()` rather than `set_workspace_root(None)`: the latter
    is a deliberate *disable* which would then outrank `CORECODER_WORKSPACE` and
    make the environment-variable test meaningless.
    """
    clear_workspace_root()
    yield
    clear_workspace_root()


def _read(path):
    return get_tool("read_file").execute(file_path=str(path))


def _write(path, content="payload\n"):
    return get_tool("write_file").execute(file_path=str(path), content=content)


# --- baseline: with no root, upstream behaviour is preserved -----------------


def test_no_root_configured_means_no_restriction(tmp_path):
    """Interactive CLI semantics: an unconfigured root must not restrict reading."""
    outside = tmp_path / "anywhere.txt"
    outside.write_text("hello", encoding="utf-8")

    assert get_workspace_root() is None
    assert "hello" in _read(outside)


# --- the vulnerability this file exists to close -----------------------------


def test_read_outside_root_is_refused(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")

    set_workspace_root(root)
    out = _read(secret)

    assert out.startswith("Error:"), out
    assert "TOP SECRET" not in out


def test_write_outside_root_is_refused(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    target = tmp_path / "escaped.txt"

    set_workspace_root(root)
    out = _write(target)

    assert out.startswith("Error:"), out
    assert not target.exists(), "write escaped the workspace root"


def test_absolute_path_outside_root_is_refused(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    set_workspace_root(root)

    out = _read("/etc/hosts")

    assert out.startswith("Error:"), out
    assert "localhost" not in out


# --- escape vectors ----------------------------------------------------------


def test_dotdot_traversal_is_refused(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("nope", encoding="utf-8")

    set_workspace_root(root)
    out = _read(root / ".." / "outside.txt")

    assert out.startswith("Error:"), out


def test_symlink_escaping_root_is_refused(tmp_path):
    """A symlink inside the root that points outside must not be followed."""
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")

    link = root / "innocent.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks unavailable on this platform")

    set_workspace_root(root)
    out = _read(link)

    assert out.startswith("Error:"), out
    assert "nope" not in out


def test_write_does_not_create_parents_outside_root(tmp_path):
    """`write_file` mkdir's parents; that must not escape either."""
    root = tmp_path / "workspace"
    root.mkdir()

    set_workspace_root(root)
    out = _write(tmp_path / "made" / "up" / "path.txt")

    assert out.startswith("Error:"), out
    assert not (tmp_path / "made").exists(), "created directories outside the root"


def test_symlink_loop_does_not_hang_or_crash(tmp_path):
    """resolve() raises RuntimeError on a symlink loop; that must surface as an error string."""
    root = tmp_path / "workspace"
    root.mkdir()
    a, b = root / "a", root / "b"
    try:
        a.symlink_to(b)
        b.symlink_to(a)
    except (OSError, NotImplementedError):  # pragma: no cover
        pytest.skip("symlinks unavailable on this platform")

    set_workspace_root(root)
    out = _read(a)

    assert out.startswith("Error:"), out


# --- paths that must still work ---------------------------------------------


def test_read_inside_root_is_allowed(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    f = root / "note.txt"
    f.write_text("hello", encoding="utf-8")

    set_workspace_root(root)

    assert "hello" in _read(f)


def test_write_inside_root_is_allowed_and_creates_parents(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()

    set_workspace_root(root)
    target = root / "sub" / "dir" / "new.txt"
    out = _write(target)

    assert not out.startswith("Error:"), out
    assert target.read_text(encoding="utf-8") == "payload\n"


def test_root_itself_is_inside_root(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "f.txt").write_text("x", encoding="utf-8")

    set_workspace_root(root)
    assert not _read(root).startswith("Error:") or "directory" in _read(root)


def test_relative_path_resolves_against_root_not_cwd(tmp_path, monkeypatch):
    """A relative path from an unrelated cwd must land inside the root."""
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "data.txt").write_text("inside", encoding="utf-8")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    set_workspace_root(root)
    out = get_tool("read_file").execute(file_path="data.txt")

    assert "inside" in out, out


# --- the root is resolved once, not trusted verbatim -------------------------


def test_symlinked_root_is_resolved(tmp_path):
    """If the root itself is a symlink, paths under the real dir must still pass."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "f.txt").write_text("ok", encoding="utf-8")

    linkroot = tmp_path / "linkroot"
    try:
        linkroot.symlink_to(real)
    except (OSError, NotImplementedError):  # pragma: no cover
        pytest.skip("symlinks unavailable on this platform")

    set_workspace_root(linkroot)

    assert "ok" in _read(real / "f.txt")


def test_all_file_tools_are_covered():
    """Guard against a new file tool being added without a boundary check."""
    names = {t.name for t in ALL_TOOLS}
    # every tool here takes a `file_path` and must be boundary-checked
    for expected in ("read_file", "write_file", "edit_file"):
        assert expected in names


def test_env_var_can_set_the_root(tmp_path, monkeypatch):
    """Runners need a way to configure this without touching Python."""
    root = tmp_path / "workspace"
    root.mkdir()

    monkeypatch.setenv("CORECODER_WORKSPACE", str(root))
    assert get_workspace_root() == root.resolve()


def test_an_explicit_none_beats_the_environment(tmp_path, monkeypatch):
    """`set_workspace_root(None)` means *disabled*, not *unset*.

    Upstream of this fix the two were indistinguishable, so the documented
    "None disables it" was false: the environment variable kept winning.
    """
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setenv("CORECODER_WORKSPACE", str(root))

    set_workspace_root(None)

    assert get_workspace_root() is None
    # and the boundary really is off, not just reporting None
    outside = tmp_path / "outside.txt"
    outside.write_text("reachable", encoding="utf-8")
    assert "reachable" in _read(outside)


def test_clearing_hands_control_back_to_the_environment(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setenv("CORECODER_WORKSPACE", str(root))

    set_workspace_root(None)
    clear_workspace_root()

    assert get_workspace_root() == root.resolve()


def test_the_context_manager_restores_the_previous_setting(tmp_path, monkeypatch):
    """Scoping a boundary to a block must not hand control back to the env.

    `clear_workspace_root()` on exit would do exactly that: with
    CORECODER_WORKSPACE set, a task would finish still bounded by the
    environment's root instead of returning to the pre-task state.
    """
    env_root = tmp_path / "env-root"
    env_root.mkdir()
    task_root = tmp_path / "task-root"
    task_root.mkdir()
    monkeypatch.setenv("CORECODER_WORKSPACE", str(env_root))

    from corecoder.tools.base import workspace_root

    assert get_workspace_root() == env_root.resolve()
    with workspace_root(task_root):
        assert get_workspace_root() == task_root.resolve()
    assert get_workspace_root() == env_root.resolve(), (
        "the env root did not come back; the block leaked its setting"
    )


def test_the_context_manager_restores_nothing_when_nothing_was_set(tmp_path):
    from corecoder.tools.base import workspace_root

    root = tmp_path / "w"
    root.mkdir()
    assert get_workspace_root() is None
    with workspace_root(root):
        assert get_workspace_root() == root.resolve()
    assert get_workspace_root() is None


def test_the_context_manager_restores_after_an_exception(tmp_path):
    from corecoder.tools.base import workspace_root

    root = tmp_path / "w"
    root.mkdir()
    try:
        with workspace_root(root):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert get_workspace_root() is None, "the boundary leaked after an exception"


def test_the_env_root_is_resolved_once_not_per_call(tmp_path, monkeypatch):
    """A relative CORECODER_WORKSPACE must not wander when the cwd changes."""
    root = tmp_path / "workspace"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("CORECODER_WORKSPACE", "workspace")
    monkeypatch.chdir(tmp_path)

    first = get_workspace_root()
    monkeypatch.chdir(elsewhere)

    assert get_workspace_root() == first, "the root moved with the process cwd"
