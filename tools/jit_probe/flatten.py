#!/usr/bin/env python3
"""Flatten a DeepSelect kernel TU (instantiation .cu + its quoted-include
header tree) into one self-contained source for DeepJIT, whose include parser
accepts only angle-bracket, prefixed includes.

Usage: flatten.py <repo-root> <relative .cu path> <out path>
The inliner resolves `#include "..."` relative to the including file's own
directory, inlines recursively with cycle prevention, and leaves every other
line untouched.  `#ifndef`/`#pragma once` guards keep double inclusions empty
at compile time, which is why textual duplication is safe here.
"""
import re
import sys
from pathlib import Path

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
EXTERN_RE = re.compile(r'^\s*extern\s+template\s')

def inline(path: Path, out: list, seen: set, repo: Path) -> None:
    path = path.resolve()
    if path in seen:
        return
    seen.add(path)
    skipping = False
    for line in path.read_text().splitlines(keepends=True):
        # DeepJIT expects exactly one device kernel in the artifact; the
        # headers' `extern template` declarations would materialize the whole
        # declared set under -device-bin, so strip them from the JIT feed
        # (only the explicit instantiation at the bottom then emits).  These
        # declarations span several lines, so skip to the terminating ';'.
        if EXTERN_RE.match(line):
            out.append("// stripped extern template declaration\n")
            skipping = True
            continue
        if skipping:
            if line.rstrip().endswith(";"):
                skipping = False
            continue
        m = INCLUDE_RE.match(line)
        if m:
            inc = (path.parent / m.group(1)).resolve()
            if not inc.exists():
                # our headers resolve through the build's -I list (csrc/,
                # csrc/maca_kernels/common) rather than only the file's dir
                for base in (repo / "csrc", repo / "csrc/maca_kernels/common"):
                    cand = (base / m.group(1)).resolve()
                    if cand.exists():
                        inc = cand
                        break
            if inc.exists():
                out.append(f"// begin inline {inc}\n")
                inline(inc, out, seen, repo)
                out.append(f"// end inline {inc}\n")
                continue
        out.append(line)

def main() -> None:
    repo = Path(sys.argv[1]).resolve()
    src = (repo / sys.argv[2]).resolve()
    dst = Path(sys.argv[3])
    out = [f"// flattened from {src}\n"]
    inline(src, out, set(), repo)
    dst.write_text("".join(out))
    print(f"{dst}: {len(out)} lines")

if __name__ == "__main__":
    main()
