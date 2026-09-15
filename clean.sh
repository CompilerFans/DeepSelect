#!/usr/bin/env bash
#
# Remove this tree's generated build, test, and Python artifacts.  Mirrors the
# host repository's `clean.sh` in shape, and is a superset of what `build.sh
# --clean`/`install.sh --clean` clear.
#
#     ./clean.sh               # remove the artifacts (same as the host's)
#     ./clean.sh --dry-run     # print what would go, remove nothing
#     ./clean.sh --help
#
# ── Why `--dry-run` is opt-in ───────────────────────────────────────────────
#
# Do not put a confirmation gate back: `./clean.sh && ./build.sh` then prints
# nothing alarming, the build succeeds anyway (build.sh does its own `rm`), and
# the full rebuild silently does not happen.  A listing would not have caught
# this tree's real mistakes either -- those were wrong *content* at a correct
# path.  So the listing is on request, and the default is what it says.
#
# ── What this deliberately does NOT do ──────────────────────────────────────
#
# The host `clean.sh` also deletes `~/.triton`, `~/.tilelang`, `~/.deep_gemm`
# and `~/.metax`; nothing here writes them.  The `find` passes also skip
# `csrc/3rdparty/**` -- nothing there matches today, but a prune by name over
# vendored source is a trap for the first `__pycache__` that appears inside one.
#
set -euo pipefail

# ── project root ────────────────────────────────────────────────────────────
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

usage() {
    cat >&2 <<'EOF'
Usage: clean.sh [--dry-run] [-h|--help]

  --dry-run    print what would be removed, remove nothing
  -h, --help   this message

Removes: build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/ and friends,
loose *.pyc /*.pyo, and every *.so in the tree (all of which are build
products -- checked, there is no vendored binary here).
Leaves: the installed site-packages copy, and ~/.deep_gemm, ~/.triton,
~/.tilelang, ~/.metax (this tree writes none of them).

`--yes` is accepted and ignored: it was this script's first spelling, when the
default was to remove nothing.  The default is now to remove, so there is
nothing left for it to answer.
EOF
}

dry_run=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        # Redundant now that removing is the default.  Accepted so an older
        # invocation still works and is visibly ignored rather than mistyped.
        --yes)     echo "clean.sh: note: --yes is redundant (removing is the" >&2
                   echo "          default now); ignoring it.  Use --dry-run to" >&2
                   echo "          preview instead." >&2
                   shift ;;
        -h|--help) usage; exit 0 ;;
        *)         echo "clean.sh: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

echo "clean.sh: ${script_dir}"
if [[ "$dry_run" == 1 ]]; then
    echo "clean.sh: dry run -- nothing will be removed."
fi

# ── directories ─────────────────────────────────────────────────────────────
# `-prune` first, so the 3rdparty subtree is neither descended into nor matched.
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
# `-type l` as well as `-type f`: `install.sh` links the wheel's extension back
# under `deep_select/`, and `-type f` does not match a symlink.  Removing the
# link does not touch its target in site-packages.
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
        # so a path already removed by an earlier pass does not error here.
        #
        # `-prune` is a no-op under `-delete` (which implies `-depth`), so the
        # directory pass cannot use `-delete`: the 3rdparty subtree would be
        # descended into, and `dist/`/`build/` are plain names it could carry.
        find . \
            \( -path './csrc/3rdparty' -o -path './.git' \) -prune -o \
            -type d \( \
                -name '__pycache__' -o -name '.pytest_cache' \
                -o -name '.mypy_cache' -o -name '.ruff_cache' \
                -o -name '.tox' -o -name '.nox' -o -name '.hypothesis' \
                -o -name 'build' -o -name 'dist' -o -name '*.egg-info' \
            \) -prune -exec rm -rf -- {} +
        # The same prune is unavailable with `-delete` here, so the subtree is
        # excluded with an explicit `! -path` guard instead, which -delete honors.
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
# removing files under an installed wheel leaves pip thinking it is still there.
# But a stale installed copy shadows the repo for anything run without
# `PYTHONPATH=.`, so report which one a plain `import` resolves to.
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
