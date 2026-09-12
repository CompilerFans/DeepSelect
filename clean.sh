#!/usr/bin/env bash
#
# Remove this tree's generated build, test, and Python artifacts.
#
# Mirrors the host repository's `clean.sh` in shape (always clean the directory
# the script lives in, even when invoked from elsewhere; prune by directory
# name; then a second pass for loose files), with the differences this tree
# needs spelled out below.  What is removed is a superset of what `build.sh
# --clean` and `install.sh --clean` do, because those two only clear what their
# own build would otherwise reuse; this clears the tree.
#
# Usage:
#     ./clean.sh               # dry-run by default: print what would go
#     ./clean.sh --yes         # actually remove it
#     ./clean.sh --help
#
# ── Why this is a dry run unless --yes ──────────────────────────────────────
#
# A wildcard delete of `*.so` and `build/` is exactly what install.sh's symlink
# step sits next to, and this tree has been bitten by build-artifact mistakes
# three separate times in one session: an in-place extension that was silently
# the *previous* variant, a stale object reused by an incremental build, and a
# probe tree whose "baseline" binary was a wrong-answer variant.  Deletion is
# the one step here that cannot be undone by rebuilding, so it prints first and
# removes on request.  `--yes` makes it scriptable; `clean.sh --yes` in a
# pipeline behaves like the host's `clean.sh`.
#
# ── What this deliberately does NOT do ──────────────────────────────────────
#
# The host `clean.sh` also deletes `$HOME/.triton`, `$HOME/.tilelang`,
# `$HOME/.deep_gemm` and `$HOME/.metax`.  Nothing in this tree writes them:
# its build is `mxcc` driven from `setup.py` (no JIT cache), and the only
# `~/.deep_gemm` reference anywhere in the repo is prose in docs that is
# describing the *host* repository's cache.  Deleting another project's cache
# from here would be a side effect with no counterpart in this tree, so it is
# left alone.
#
# The `find` passes also skip `csrc/3rdparty/**` (vendored kerutils, and the
# cutlass submodule from `.gitmodules`).  Neither has anything matching these
# names today, but a prune by name over vendored third-party source is a trap
# waiting for the first `__pycache__` that appears inside one.
#
set -euo pipefail

# ── project root ────────────────────────────────────────────────────────────
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

usage() {
    cat >&2 <<'EOF'
Usage: clean.sh [--yes] [-h|--help]

  --yes        actually remove the artifacts (default: print what would go)
  -h, --help   this message

Removes: build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/ and friends,
loose *.pyc /*.pyo, and every *.so in the tree (all of which are build
products -- checked, there is no vendored binary here).
Leaves: the installed site-packages copy, and ~/.deep_gemm, ~/.triton,
~/.tilelang, ~/.metax (this tree writes none of them).
EOF
}

dry_run=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes)     dry_run=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *)         echo "clean.sh: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

echo "clean.sh: ${script_dir}"
if [[ "$dry_run" == 1 ]]; then
    echo "clean.sh: dry run -- nothing will be removed.  Re-run with --yes."
fi

# ── directories ─────────────────────────────────────────────────────────────
# `-prune` on the 3rdparty subtree first so it is neither descended into nor
# matched; `-prune ... -exec rm -rf` then removes what was matched.
dirs=$(find . \
    \( -path './csrc/3rdparty' -o -path './.git' \) -prune -o \
    -type d \( \
        -name '__pycache__' \
        -o -name '.pytest_cache' \
        -o -name '.mypy_cache' \
        -o -name '.ruff_cache' \
        -o -name '.tox' \
        -o -name '.nox' \
        -o -name '.hypothesis' \
        -o -name 'build' \
        -o -name 'dist' \
        -o -name '*.egg-info' \
    \) -print | sort)

# ── loose files ─────────────────────────────────────────────────────────────
# `-type l` as well as `-type f`: install.sh links the wheel's extension back
# under `deep_select/` as a symlink, and `-type f` does not match a symlink.
# The link is a build artifact of this tree (`install.sh` recreates it) and its
# *target* is in site-packages, which `rm -f` on a symlink does not touch.
files=$(find . \
    \( -path './csrc/3rdparty' -o -path './.git' \) -prune -o \
    \( -type f -o -type l \) \( \
        -name '*.pyc' \
        -o -name '*.pyo' \
        -o -name '.coverage' \
        -o -name '.coverage.*' \
        -o -name 'coverage.xml' \
        -o -name '*.so' \
    \) -print | sort)

count_dirs=$(printf '%s' "$dirs" | grep -c . || true)
count_files=$(printf '%s' "$files" | grep -c . || true)

if [[ "$count_dirs" == 0 && "$count_files" == 0 ]]; then
    echo "clean.sh: nothing to remove."
else
    echo "clean.sh: directories ($count_dirs):"
    printf '%s\n' "$dirs" | sed 's/^/    /'
    echo "clean.sh: files ($count_files):"
    printf '%s\n' "$files" | sed 's/^/    /'

    if [[ "$dry_run" == 0 ]]; then
        # Re-run the finds with -exec rather than deleting the captured paths,
        # so a directory already removed by an earlier pass (an egg-info inside
        # build/, say) does not produce a spurious error mid-delete.
        #
        # The directory pass uses `-prune -exec rm -rf`: `-prune` is a no-op
        # under `-delete` (which implies `-depth`), so the 3rdparty subtree
        # would be descended into.  In this pass it is not, and that matters --
        # `dist/` and `build/` are plain names a vendored tree could contain.
        find . \
            \( -path './csrc/3rdparty' -o -path './.git' \) -prune -o \
            -type d \( \
                -name '__pycache__' -o -name '.pytest_cache' \
                -o -name '.mypy_cache' -o -name '.ruff_cache' \
                -o -name '.tox' -o -name '.nox' -o -name '.hypothesis' \
                -o -name 'build' -o -name 'dist' -o -name '*.egg-info' \
            \) -prune -exec rm -rf -- {} +
        # The file pass cannot use -prune with -delete (GNU find warns and
        # ignores the prune), so the vendored subtree is excluded with an
        # explicit `! -path` guard instead, which -delete honors.
        find . \
            ! -path './csrc/3rdparty/*' ! -path './.git/*' \
            \( -type f -o -type l \) \( \
                -name '*.pyc' -o -name '*.pyo' -o -name '.coverage' \
                -o -name '.coverage.*' -o -name 'coverage.xml' -o -name '*.so' \
            \) -delete
        echo "clean.sh: removed."
    fi
fi

# ── what the environment still resolves ─────────────────────────────────────
# This script does not touch site-packages -- that is `pip uninstall`'s job, and
# removing files out from under an installed wheel leaves pip believing it is
# still there.  But a stale installed copy shadows the repo for anything not run
# with `PYTHONPATH=.`, which is a real hazard for this tree (install.sh links
# the wheel's extension back under `deep_select/`, so the two are normally the
# same build).  Report which one a plain `import` resolves to, from a directory
# that is not the repo.
echo "clean.sh: installed copy (this script does not remove it):"
( cd /tmp && python - <<'PY' 2>/dev/null || echo "    (deep_select is not importable outside the repo)"
import os
try:
    import deep_select
except Exception as exc:                 # not installed, or an arch mismatch
    raise SystemExit(f"    import failed: {exc}")
print(f"    {deep_select.__version__}  from {os.path.dirname(deep_select.__file__)}")
PY
)

echo "clean.sh: done.  Rebuild with ./build.sh or ./install.sh."
cd "$original_dir"
