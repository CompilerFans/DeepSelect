#!/usr/bin/env python3
"""Which `deep_select` answers, and which artifact it loads -- one definition.

Every runner here has to settle the same two questions before it can measure
anything, and `CLAUDE.md` states the rule for both once:

  - **which package answers.**  `$REPO` goes on `sys.path` only when the
    checkout has a built extension to answer with.  A tree with none must let
    an installed wheel answer, because a path entry is a claim about which
    package ran -- and a checkout that cannot load an extension imports fine
    and then fails at call time, since `_binding` resolves the `.so` lazily.
  - **which artifact that package loads.**  Asked of the loader itself
    (`_binding.extension_path`), never of a path under the checkout: it is the
    same glob and the same sorted-path-last rule `load()` uses, so a receipt
    cannot name a file other than the one a call would load.

Neither can live inside `deep_select`: the first decides which `deep_select`
to import, so it has to run before the package is reachable.  It lives here
instead, and every consumer imports it -- `run_test.sh`, `run_bench.sh`,
`scripts/perf_snapshot.py`, `scripts/wheel_smoke.py`, `tools/sample_gate.py`,
`tools/overflow_exactfit.py`.

Consumers differ in two ways only, and both are arguments rather than copies:
whether the grid's directory is wanted (`with_tests` -- `tests/` is in the
repository, not in the wheel), and how many md5 characters a receipt prints.
"""
import glob
import hashlib
import os
import sys
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The extension's own spelling, and the one place it is globbed for the
# "does this checkout answer" question.  `_binding` holds the other copy --
# it resolves a *file*, which is a different question (see
# `checkout_answers`).
EXTENSION_GLOB = "deep_select_maca*.so"

# Human-facing receipts print 12 characters: it is the length commit messages
# cite and it still distinguishes two builds of one tree.  A data column (the
# snapshot manifest's `extension_md5`) keeps all 32 -- `md5_of(path, None)`.
RECEIPT_MD5_CHARS = 12


def checkout_answers(repo: str = REPO) -> bool:
    """Whether the checkout has a built extension -- a boolean, on purpose.

    Which file it is is `_binding.extension_path`'s question.  Answering it
    here is exactly how a snapshot record once named the wrong family
    (`provenance()` took the first `.so` in `deep_select/`, which on a
    three-family build is always xcore1000, while a C600U loads xcore1600).
    """
    return bool(glob.glob(os.path.join(repo, "deep_select", EXTENSION_GLOB)))


def use_checkout_if_built(repo: str = REPO, with_tests: bool = False) -> bool:
    """Apply the path rule; return whether the checkout is the answering tree.

    Entries that would resolve to the checkout come off first: `python -` and
    `python scripts/x.py` can leave one behind, and on its own it makes
    `import deep_select` land on a tree whose extension may not load.  The
    grid's directory is re-inserted only when asked for, and the checkout's
    own root only when it has an extension.
    """
    here = os.path.abspath(os.getcwd())
    repo = os.path.abspath(repo)
    drop = {here, repo, os.path.join(repo, "tests")}
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or here) not in drop]
    if with_tests:
        sys.path.insert(0, os.path.join(repo, "tests"))
    answers = checkout_answers(repo)
    if answers:
        sys.path.insert(0, repo)
    return answers


def answering_extension() -> Optional[str]:
    """The artifact the answering package loads, or None when none resolves.

    A probe, so a failure to import is "no artifact" rather than an exception:
    the callers are printing a receipt or deciding whether to refuse, and both
    want that answer rather than a traceback.
    """
    try:
        from deep_select.interface import _KERNEL_NAME, _binding
        return _binding.extension_path(_KERNEL_NAME)
    except Exception:  # noqa: BLE001 - a probe, not a gate
        return None


def md5_of(path: str, chars: Optional[int] = RECEIPT_MD5_CHARS) -> str:
    """The file's md5, truncated for a human-facing line unless `chars` is None."""
    digest = hashlib.md5(open(path, "rb").read()).hexdigest()
    return digest if chars is None else digest[:chars]


def package_receipt(module, chars: Optional[int] = RECEIPT_MD5_CHARS) -> str:
    """Where the package came from and which artifact it resolves.

    The two facts a record is worthless without: a count without the binary
    that produced it is not a receipt (`CLAUDE.md`: the `.so` md5 is the
    "which artifact did I measure" receipt).
    """
    where = os.path.dirname(os.path.abspath(module.__file__))
    line = f"deep_select {module.__version__} from {where}"
    so = answering_extension()
    if so:
        line += f"\nartifact {so}\n         md5 {md5_of(so, chars)}"
    else:
        line += "\nartifact: none resolves (a torch-only run)"
    return line
