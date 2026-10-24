#!/usr/bin/env python3
"""The kernel tree's include graph, as a command rather than a convention.

`CLAUDE.md`'s include discipline states three rules for `csrc/maca_kernels/`,
and each one rests on a measurement rather than on taste:

  * **Angle brackets, repo-root-prefixed** -- DeepJIT's include parser tracks
    only `<csrc/...>`: quoted includes are rejected outright and its digest
    covers only prefixed paths, so a header written the other way silently
    leaves the JIT feed.
  * **`routing.cuh` and `radix_configs.inc` are included by `maca_topk.cu`
    and by nothing else.**  mxcc's device pass compiles every function body
    and every template instantiation present in a TU -- an `extern template`
    declaration is honored by the host pass only -- so a header that included
    either one would drag every launcher, or all 60 row kernels, into the
    device image of every TU that includes the header.  Measured: a probe TU
    went 540,669 B -> 26,221 B when the four non-template dispatchers moved
    out of `radix_chunked.cuh`, and the row table cost each of 89 TUs
    7,626,616 B until it moved into `maca_topk.cu`.
  * **One project include per instantiation TU** -- a generated file declares
    the one explicit instantiation its table row names, and the source list
    `setup.py` compiles is that table.  A second include is either a dataflow
    the row does not name or a header that will be walked into the device
    pass for nothing.

A fourth is a consequence rather than a rule: **every project include must
resolve.**  A rename that leaves a stale path behind compiles anyway, because
nothing includes it, and then breaks on the day someone does.

Usage:
    python3 tools/include_gate.py              # check this checkout, exit 1 on any violation
    python3 tools/include_gate.py --tree DIR   # check another checkout
    python3 tools/include_gate.py --self-test  # 6 controls, 5 of them negative

The check is textual and cheap on purpose -- it is meant to run beside
`scripts/generate_instantiations.py --check` before a build, not to be a
compiler.  It says nothing about whether an include is *needed*, only that the
ones present are spelled and placed the way the tree documents.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_TREE = os.path.join("csrc", "maca_kernels")

# The two files whose every inclusion is a device-image cost, and the one TU
# allowed to include them.  See the docstring for the measurements.
SINGLE_INCLUDER = {
    os.path.join("csrc", "maca_kernels", "common", "routing.cuh"):
        {os.path.join("csrc", "maca_kernels", "common", "maca_topk.cu")},
    os.path.join("csrc", "maca_kernels", "common", "radix_configs.inc"):
        {os.path.join("csrc", "maca_kernels", "common", "maca_topk.cu")},
}

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*(?P<open>[<"])(?P<target>[^>"]+)(?P<close>[>"])\s*$')


def strip_comments(text: str) -> str:
    """Blank out `//` and `/* */` while keeping every newline in place.

    Line numbers have to survive, because a violation is reported as
    `path:line` -- so this replaces comment characters with spaces rather than
    deleting them.  String literals are not special-cased: an include line
    carries no string, and `//` or `/*` inside one on some other line would
    only ever cause the gate to miss, never to fire falsely.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def is_project(target: str) -> bool:
    """Whether an include names this tree's own sources.

    `csrc/` is unambiguous.  The suffixes are what makes a *quoted* project
    include -- `"radix_common.cuh"` -- count as project too: the point of the
    form rule is exactly those, and a rule that only looked at `csrc/` would
    have nothing to say about the spelling it forbids.
    """
    return target.startswith("csrc/") or target.endswith((".cuh", ".inc"))


def kernel_sources(root: str):
    base = os.path.join(root, KERNEL_TREE)
    for dirpath, _dirnames, filenames in os.walk(base):
        for name in sorted(filenames):
            if name.endswith((".cu", ".cuh")):
                yield os.path.join(dirpath, name)


def rel(root: str, path: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def check_tree(root: str):
    """Return the violations, as `path:line: reason` strings."""
    violations = []
    includers = {}
    for path in kernel_sources(root):
        text = strip_comments(open(path, encoding="utf-8", errors="replace").read())
        project_includes = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            m = INCLUDE_RE.match(line)
            if not m:
                continue
            target = m.group("target").strip()
            if not is_project(target):
                continue
            project_includes += 1
            where = f"{rel(root, path)}:{lineno}"
            if m.group("open") == '"' or m.group("close") == '"':
                violations.append(
                    f"{where}: quoted include of a project header ({target}) -- "
                    f"DeepJIT tracks only angle-bracket, repo-root-prefixed paths")
                continue
            if not target.startswith("csrc/"):
                violations.append(
                    f"{where}: project include is not repo-root-prefixed "
                    f"({target}) -- spell it <csrc/...>")
                continue
            if not os.path.exists(os.path.join(root, target)):
                violations.append(f"{where}: include does not resolve ({target})")
            includers.setdefault(target, set()).add(rel(root, path))
        if "/instantiations/" in rel(root, path) and project_includes != 1:
            violations.append(
                f"{rel(root, path)}: an instantiation TU carries exactly one "
                f"project include, found {project_includes}")

    for target, allowed in SINGLE_INCLUDER.items():
        got = includers.get(target, set())
        extra = sorted(got - allowed)
        missing = sorted(allowed - got)
        if extra:
            violations.append(
                f"{target}: included by {', '.join(extra)} -- this file may be "
                f"included only by {', '.join(sorted(allowed))} (the device "
                f"pass compiles every body a TU sees)")
        if missing:
            violations.append(
                f"{target}: expected to be included by "
                f"{', '.join(missing)}, but nothing includes it")
    return violations


# ── controls ───────────────────────────────────────────────────────────────
CLEAN_MAIN = """\
#include <csrc/maca_kernels/common/routing.cuh>
#include <csrc/maca_kernels/common/radix_core.cuh>
#include <csrc/maca_kernels/common/radix_configs.inc>
"""
CLEAN_HEADER = "// a header\n#include <csrc/maca_kernels/common/launch.cuh>\n"
CLEAN_INST = "#include <csrc/maca_kernels/common/contract.cuh>\n"


def _tree(files):
    root = tempfile.mkdtemp(prefix="include_gate_")
    for name, text in files.items():
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    return root


def _base():
    return {
        "csrc/maca_kernels/common/maca_topk.cu": CLEAN_MAIN,
        "csrc/maca_kernels/common/routing.cuh": "// routing\n",
        "csrc/maca_kernels/common/radix_configs.inc": "X(a)\n",
        "csrc/maca_kernels/common/radix_core.cuh": CLEAN_HEADER,
        "csrc/maca_kernels/common/launch.cuh": "// launch\n",
        "csrc/maca_kernels/common/contract.cuh": "// contract\n",
        "csrc/maca_kernels/xcore1000/instantiations/row.cu": CLEAN_INST,
    }


def self_test() -> int:
    controls = []

    def control(name, files, want):
        root = _tree(files)
        got = check_tree(root)
        ok = (not got) if want is None else any(want in v for v in got)
        controls.append((name, ok, got))

    control("clean tree", _base(), None)

    files = _base()
    files["csrc/maca_kernels/common/radix_core.cuh"] += '#include "quoted.cuh"\n'
    control("quoted include", files, "quoted include")

    files = _base()
    files["csrc/maca_kernels/common/radix_core.cuh"] += '#include <radix_common.cuh>\n'
    control("not repo-root-prefixed", files, "not repo-root-prefixed")

    files = _base()
    files["csrc/maca_kernels/common/radix_core.cuh"] += \
        "#include <csrc/maca_kernels/common/routing.cuh>\n"
    control("routing.cuh from a header", files, "included by")

    files = _base()
    files["csrc/maca_kernels/xcore1000/instantiations/row.cu"] += \
        "#include <csrc/maca_kernels/common/radix_core.cuh>\n"
    control("two includes in an instantiation TU", files, "exactly one project include")

    files = _base()
    files["csrc/maca_kernels/common/radix_core.cuh"] += \
        "#include <csrc/maca_kernels/common/does_not_exist.cuh>\n"
    control("stale include", files, "does not resolve")

    width = max(len(name) for name, _, _ in controls)
    failed = 0
    for name, ok, got in controls:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<{width}}  "
              f"({len(got)} violation{'s' if len(got) != 1 else ''})")
        if not ok:
            failed += 1
            for v in got:
                print(f"          {v}")
    print(f"{len(controls) - failed}/{len(controls)} controls passed")
    return 0 if not failed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", default=REPO, help="checkout to check (default: this one)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the gate's own controls instead of the tree")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    root = os.path.abspath(a.tree)
    if not os.path.isdir(os.path.join(root, KERNEL_TREE)):
        sys.exit(f"include_gate.py: {root} has no {KERNEL_TREE}")
    violations = check_tree(root)
    files = sum(1 for _ in kernel_sources(root))
    for v in violations:
        print(v)
    print(f"{files} files, {len(violations)} violation"
          f"{'' if len(violations) == 1 else 's'}")
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
