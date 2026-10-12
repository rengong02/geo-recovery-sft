"""Replace the lab's absolute path root with a neutral placeholder.

The runs recorded in `evals/train/` happened on a shared lab machine, so their
reports, logs and manifests carry that machine's absolute paths -- including a
home directory that identifies a person. A repository that is meant to be read
by strangers should not ship someone's account name in every frozen artifact.

The substitution is deliberately **mechanical and documented** rather than a
hand-edit of the interesting-looking fields: a reader who has the original file
can apply the same rule and get the same bytes back, which is what keeps the
frozen reports checkable after redaction. Redacting by hand would leave the
hashes unverifiable and the rule unknown.

Consequence, stated where it is used and not only here: the on-disk copies are
**not byte-identical to the server's** any more. `results/` records the hashes of
the redacted copies and says so.

Run with `--check` to report without writing -- that is the form to use as a
gate before publishing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The rule cannot be written as one literal here: a tool that removes a string
# may not itself contain it, or the repository still ships the thing it exists
# to delete -- and it would rewrite its own source on the first run, which is
# exactly what happened the first time this file was written.
LAB_PREFIX = "/" + "g3240100218" + "hbj"
PLACEHOLDER = "<lab-root>"

#: Binary or generated files that must not be rewritten in place. Nothing in the
#: kept tree is expected to contain the prefix, but a text substitution over a
#: shapefile would corrupt it, so the rule is explicit rather than implied.
SKIP_SUFFIXES = frozenset({".zip", ".shp", ".dbf", ".shx", ".prj", ".cpg", ".tif",
                           ".png", ".gz", ".safetensors"})


def candidate_paths() -> list[Path]:
    """Every file the repository keeps: tracked ones plus untracked-and-not-ignored."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    others = subprocess.run(["git", "ls-files", "-o", "--exclude-standard", "-z"],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout
    names = {name for name in (listed + others).split("\0") if name}
    return sorted(ROOT / name for name in names)


def redact(path: Path) -> int:
    """Return the number of replacements; raise if the file is not UTF-8 text."""
    if path.suffix in SKIP_SUFFIXES:
        return 0
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return 0
    if LAB_PREFIX not in text:
        return 0
    count = text.count(LAB_PREFIX)
    path.write_text(text.replace(LAB_PREFIX, PLACEHOLDER), encoding="utf-8")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report occurrences and exit nonzero; write nothing")
    args = parser.parse_args()

    if LAB_PREFIX == PLACEHOLDER or not LAB_PREFIX.startswith("/"):
        # A no-op that reports clean is worse than no tool: it would let this
        # gate pass while the paths stay in the tree.
        print(f"refusing to run: LAB_PREFIX is {LAB_PREFIX!r}, which cannot match "
              f"a real path", file=sys.stderr)
        return 2

    total = 0
    touched = 0
    for path in candidate_paths():
        if not path.is_file():
            continue
        if args.check:
            if path.suffix in SKIP_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            found = text.count(LAB_PREFIX)
        else:
            found = redact(path)
        if found:
            total += found
            touched += 1
            print(f"{'found' if args.check else 'redacted'} {found:>3}  "
                  f"{path.relative_to(ROOT)}")
    print(f"\n{touched} files, {total} occurrences")
    if args.check and total:
        print("the lab path root is still present; run without --check to redact")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
