#!/usr/bin/env bash
#
# Remove this tree's generated build, test, and Python artifacts.
#
#     ./clean.sh               # remove the artifacts
#     ./clean.sh --dry-run     # print what would go, remove nothing
#     ./clean.sh --help
#
# The default is to remove, with no confirmation: `./clean.sh && ./develop.sh`
# would otherwise print nothing alarming, succeed anyway (`develop.sh` does its
# own `rm`), and silently not rebuild.  The listing is on request because it is
# a second traversal.
#
# The `find` passes skip `csrc/3rdparty/**`, and delete nothing outside this
# tree -- in particular not `~/.triton`, `~/.tilelang`, `~/.deep_gemm` or
# `~/.metax`, which nothing here writes.
#
set -euo pipefail

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
EOF
}

dry_run=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)         echo "clean.sh: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

echo "clean.sh: ${script_dir}"

# ── what counts as an artifact ──────────────────────────────────────────────
# One copy of each predicate list, shared by the listing and the removal below.
# This file used to carry the same two lists four times (a pass to collect and a
# pass to delete, twice over), which is four places for one of them to be edited.
dir_pred=(
    -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache'
    -o -name '.ruff_cache' -o -name '.tox' -o -name '.nox'
    -o -name '.hypothesis' -o -name 'build' -o -name 'dist'
    -o -name '*.egg-info'
)
# `-type l` as well as `-type f`: `-type f` is false for a symlink (find lstats),
# so a hand-made `deep_select/deep_select_maca.so -> build/lib.../...so` would
# survive a clean and shadow the next build.  Nothing here creates one.
file_pred=(
    -name '*.pyc' -o -name '*.pyo' -o -name '.coverage' -o -name '.coverage.*'
    -o -name 'coverage.xml' -o -name '*.so'
)
# `-prune` first, so the 3rdparty subtree is neither descended into nor matched.
dirs_find=( . \( -path './csrc/3rdparty' -o -path './.git' \) -prune -o
            -type d \( "${dir_pred[@]}" \) )
files_find=( . \( -path './csrc/3rdparty' -o -path './.git' \) -prune -o
             \( -type f -o -type l \) \( "${file_pred[@]}" \) )

if [[ "$dry_run" == 1 ]]; then
    echo "clean.sh: dry run -- nothing will be removed."
    dirs=$(find "${dirs_find[@]}" -print | sort)
    files=$(find "${files_find[@]}" -print | sort)
    count_dirs=$(printf '%s' "$dirs" | grep -c . || true)
    count_files=$(printf '%s' "$files" | grep -c . || true)
    if [[ "$count_dirs" == 0 && "$count_files" == 0 ]]; then
        echo "clean.sh: nothing to remove."
    else
        echo "clean.sh: directories ($count_dirs):"
        printf '%s\n' "$dirs" | sed 's/^/    /'
        echo "clean.sh: files ($count_files):"
        printf '%s\n' "$files" | sed 's/^/    /'
    fi
else
    # The removal is a second traversal rather than a replay of the listing, so
    # a path already gone between the two cannot error here.
    #
    # `-prune` is a no-op under `-delete` (which implies `-depth`), so the
    # directory pass cannot use `-delete`: the 3rdparty subtree would be
    # descended into, and `dist/`/`build/` are plain names it could carry.
    find "${dirs_find[@]}" -prune -exec rm -rf -- {} +
    # The same prune is unavailable with `-delete`, so the subtree is excluded
    # with an explicit `! -path` guard instead, which `-delete` honors.
    find . \
        ! -path './csrc/3rdparty/*' ! -path './.git/*' \
        \( -type f -o -type l \) \( "${file_pred[@]}" \) -delete
    echo "clean.sh: removed."
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

echo "clean.sh: done.  Rebuild with ./develop.sh (in place) or ./build.sh (a wheel)."
cd "$original_dir"
