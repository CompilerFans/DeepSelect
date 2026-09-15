#!/usr/bin/env python3
"""Decide whether a revision changed code or only its documentation.

A comment is invisible to the compiler and to review at this size, so "this
diff is comments only" is a claim that has to be *gated*, not asserted.  This
does the gating: strip comments AND docstrings from both revisions, collapse
whitespace, and require the remainder to be byte-identical.  That is the
contract `CLAUDE.md`'s comment discipline is written against.

    # one pair
    tools/comment_gate.py OLD_FILE NEW_FILE

    # every file a commit touched
    tools/comment_gate.py --commit e3f73cb

    # every file the working tree has modified against HEAD
    tools/comment_gate.py --worktree

    # the gate's own controls (run this after changing the gate)
    tools/comment_gate.py --self-test

Exit status is 0 when every pair is documentation-only, 1 otherwise -- so it
reads as a plain pass/fail in a shell `&&` chain.

Two ways the gate itself lies, both measured, both defended against here:

  * **Dispatch on the real extension.**  A copy staged as `/tmp/x/h.bak` does
    not end in `.sh`, so it reaches the C stripper, which strips `//` and
    `/* */` and NOT `#` -- and every shell file then reads as "code changed"
    while nothing did.  Stage both sides under the same basename; `--commit`
    and `--worktree` do this for you.
  * **Docstrings are documentation, not code.**  Python's tokenizer emits them
    as STRING, not COMMENT, so tokenize-only stripping treats a rewritten
    docstring as a behaviour change.  `strip_py` drops the spans `ast` reports
    for Module/ClassDef/FunctionDef bodies as well.  Runtime strings -- assert
    messages, log lines, format strings -- are NOT docstrings and stay in the
    comparison, because changing one can change behaviour.

The gate proves the *files* did not change code.  It says nothing about
whether the comment that survived is still true, or whether a table it now
points at still exists -- those are the reviewer's job, and `CLAUDE.md` states
them as the pointer-must-resolve rule.
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import re
import subprocess
import sys
import tempfile
import tokenize


# ── C / C++ ────────────────────────────────────────────────────────────────
def strip_c(src: str) -> str:
    """Drop `//` and `/* */`, leaving string and char literals alone."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in "\"'":
            q = c
            out.append(c)
            i += 1
            while i < n:
                out.append(src[i])
                if src[i] == "\\":
                    if i + 1 < n:
                        out.append(src[i + 1])
                        i += 2
                        continue
                elif src[i] == q:
                    i += 1
                    break
                i += 1
            continue
        if c == "/" and i + 1 < n:
            if src[i + 1] == "/":
                j = src.find("\n", i)
                i = n if j < 0 else j
                continue
            if src[i + 1] == "*":
                j = src.find("*/", i + 2)
                i = n if j < 0 else j + 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


# ── shell ──────────────────────────────────────────────────────────────────
def strip_sh(src: str) -> str:
    """Drop `#` comments, honouring quotes and heredocs.

    A `#` starts a comment only at a line start or after whitespace, which is
    what protects `${x#y}` and `foo#bar`.  Quoting is tracked so a `#` inside a
    string survives.  Heredoc bodies pass through untouched: they carry shell
    and inline-python fragments that must be compared as code.
    """
    out = []
    heredoc = None
    for line in src.split("\n"):
        if heredoc is not None:
            out.append(line)
            if line.strip() == heredoc:
                heredoc = None
            continue
        m = re.search(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", line)
        if m:
            heredoc = m.group(2)
        res = []
        i, n = 0, len(line)
        q = None
        while i < n:
            c = line[i]
            if q is None and c in "'\"":
                q = c
                res.append(c)
                i += 1
                continue
            if q is not None:
                res.append(c)
                if c == "\\" and q == '"' and i + 1 < n:
                    res.append(line[i + 1])
                    i += 2
                    continue
                if c == q:
                    q = None
                i += 1
                continue
            if c == "#" and (i == 0 or line[i - 1] in " \t"):
                break
            res.append(c)
            i += 1
        out.append("".join(res))
    return "\n".join(out)


# ── Python ─────────────────────────────────────────────────────────────────
def _docstring_lines(src: str) -> set:
    """Line numbers covered by a docstring.

    A docstring is an `ast.Expr` wrapping a bare string constant, so `ast` can
    name exactly those and leave every other string alone.
    """
    spans = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return spans
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None) or []
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            spans.update(range(first.value.lineno, first.value.end_lineno + 1))
    return spans


def strip_py(src: str) -> str:
    drop = _docstring_lines(src)
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and tok.start[0] in drop:
            continue
        out.append(tok.string)
    return "\n".join(out)


def norm(s: str) -> str:
    """Collapse whitespace -- indentation and line breaks move when comments
    shrink, and neither is code."""
    return re.sub(r"\s+", " ", s).strip()


CODE_EXT = (".py", ".sh", ".bash", ".cu", ".cuh", ".c", ".h", ".cpp", ".hpp",
            ".cc", ".cxx")


def is_code(path: str) -> bool:
    """Whether this gate can say anything about the file at all.

    A `.md` run through `strip_c` is not a conservative default, it is a wrong
    answer dressed as one: there is no comment syntax to strip and every line
    reads as code, so the gate would report DIFFERS on a file that contains no
    code to change.  Say "not gated" instead of inventing a verdict.
    """
    return path.endswith(CODE_EXT)


def strip_text(path: str, src: str) -> str:
    if path.endswith(".py"):
        return norm(strip_py(src))
    if path.endswith((".sh", ".bash")):
        return norm(strip_sh(src))
    return norm(strip_c(src))


def compare(old_src: str, new_src: str, path: str) -> tuple[bool, str]:
    """(identical?, explanation)."""
    try:
        x = strip_text(path, old_src)
        y = strip_text(path, new_src)
    except (tokenize.TokenError, IndentationError) as exc:
        return False, f"could not tokenize: {exc}"
    if x == y:
        return True, ""
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return False, (f"first difference at char {i}:\n"
                           f"      before: ...{x[max(0, i - 120):i + 120]!r}\n"
                           f"      after : ...{y[max(0, i - 120):i + 120]!r}")
    return False, (f"length differs: {len(x)} vs {len(y)}\n"
                   f"      tail before: {x[min(len(x), len(y)):][:300]!r}\n"
                   f"      tail after : {y[min(len(x), len(y)):][:300]!r}")


# ── drivers ────────────────────────────────────────────────────────────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args: str) -> str:
    """Run git against THIS repo, not the caller's cwd.

    Without the `-C`, a run from anywhere else gates whatever repository that
    directory happens to be in -- and this tree is vendored inside another one,
    so "anywhere else" is a real place to be standing.
    """
    return subprocess.run(["git", "-C", _REPO, *args], capture_output=True,
                          text=True, check=True).stdout


def _pairs_from_commit(rev: str):
    """(path, old, new) for every file a commit touched."""
    paths = [p for p in _git("show", "--name-only", "--pretty=format:", rev).split("\n") if p]
    for p in paths:
        old = _git("show", f"{rev}^:{p}")
        try:
            new = _git("show", f"{rev}:{p}")
        except subprocess.CalledProcessError:
            continue  # deleted by this commit
        yield p, old, new


def _pairs_from_worktree(rev: str):
    """(path, old, new) for every file the working tree modified against `rev`.

    Both sides are staged under the same basename -- the old revision in a temp
    tree mirroring the repo's layout -- because the stripper dispatches on the
    extension and a flattened copy would silently take the wrong one.
    """
    paths = [p for p in _git("diff", "--name-only", rev).split("\n") if p]
    with tempfile.TemporaryDirectory() as td:
        for p in paths:
            dest = os.path.join(td, p)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                old = _git("show", f"{rev}:{p}")
            except subprocess.CalledProcessError:
                continue  # new in the working tree; nothing to compare
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write(old)
            try:
                with open(os.path.join(_REPO, p), encoding="utf-8") as fh:
                    new = fh.read()
            except OSError:
                continue
            yield p, old, new


# ── controls ───────────────────────────────────────────────────────────────
SELF_TEST = [
    # (path, old, new, expect_identical)
    ("a.py", "import os\nfrom typing import Optional, Tuple\nx = f(a, b)  # hi\n",
     "import os\nfrom typing import Optional,Tuple\nx=f(a,b)\n", True),
    ("a.py", 'def f():\n    """old doc."""\n    return {"a": 1}  # c\n',
     'def f():\n    """new."""\n    return {"a":1}\n', True),
    # a docstring rewrite is documentation-only...
    ("a.py", 'def f():\n    """old."""\n    return 1\n',
     'def f():\n    """new."""\n    return 1\n', True),
    # ...but a runtime string is code: changing it can change behaviour.
    ("a.py", 'def f():\n    raise ValueError("old msg")\n    return 1\n',
     'def f():\n    raise ValueError("new msg")\n    return 1\n', False),
    ("a.py", 'def f():\n    assert x, "a real message"\n    return 1\n',
     'def f():\n    return 1\n', False),
    ("a.sh", 'x=1  # trailing\n# full line\necho "a # b"\n',
     'x=1\necho "a # b"\n', True),
    # `${x#y}` and `foo#bar` are not comments
    ("a.sh", 'echo ${x#y}\nfoo#bar\n', 'echo ${x#y}\nfoo#bar\n', True),
    ("a.sh", 'x=1  # different comment\necho "a # b"\n',
     'x=1\necho "a # b"\n', True),
    ("a.sh", 'x=1\necho "a # b"\n', 'x=2\necho "a # b"\n', False),
    ("a.cu", '// c\nint f() { return 1; } /* x */\n', 'int f() { return 1; }\n', True),
    ("a.cu", 'const char *s = "// not a comment";\n', 'const char *s = "// not a comment";\n', True),
    ("a.cu", 'int f() { return 1; }\n', 'int f() { return 2; }\n', False),
]


def self_test() -> int:
    bad = 0
    for path, old, new, expect in SELF_TEST:
        got, why = compare(old, new, path)
        mark = "ok  " if got == expect else "FAIL"
        if got != expect:
            bad += 1
        print(f"  {mark} {path:<8} expect={'identical' if expect else 'differs':<9} "
              f"got={'identical' if got else 'differs'}")
        if got != expect and why:
            print(f"        {why}")
    print(f"\n  {len(SELF_TEST) - bad}/{len(SELF_TEST)} controls pass")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", nargs="?", help="baseline file")
    ap.add_argument("new", nargs="?", help="candidate file")
    ap.add_argument("--commit", metavar="REV", help="every file REV touched vs REV^")
    ap.add_argument("--worktree", nargs="?", const="HEAD", metavar="REV",
                    help="every file modified vs REV (default HEAD)")
    ap.add_argument("--self-test", action="store_true", help="run the gate's own controls")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.commit:
        pairs = list(_pairs_from_commit(args.commit))
    elif args.worktree:
        pairs = list(_pairs_from_worktree(args.worktree))
    elif args.old and args.new:
        pairs = [(args.new, open(args.old, encoding="utf-8").read(),
                  open(args.new, encoding="utf-8").read())]
    else:
        ap.error("give OLD NEW, --commit REV, --worktree [REV], or --self-test")

    same = skipped = 0
    for path, old, new in pairs:
        if not is_code(path):
            skipped += 1
            print(f"  not gated        {path}  (not a code file)")
            continue
        ok, why = compare(old, new, path)
        if ok:
            same += 1
            print(f"  IDENTICAL CODE   {path}")
        else:
            print(f"  !! CODE DIFFERS  {path}")
            print(f"    {why}")
    failed = len(pairs) - same - skipped
    print(f"\n{len(pairs)} files: {same} documentation-only, {failed} differ in code, "
          f"{skipped} not gated (not code)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
