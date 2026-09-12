#!/usr/bin/env bash
#
# Build DeepSelect for this device and install it into the active environment.
#
# Mirrors the host repository's `install.sh` in shape (script_dir cd, MACA_PATH,
# CUCC_TARGETS, bdist_wheel, pip install --force-reinstall --no-deps, restore
# the caller's directory).  One thing the host script does not need and this
# tree does: a *single* `setup.py` invocation per wheel.
#
# `setup.py` stamps the version with `datetime.now()`.  A PEP 517 install runs
# `setup.py` twice -- once for metadata, once for the wheel -- and when the two
# runs straddle a second boundary pip rejects the result as misnamed
# (`Wheel has unexpected file name`); build isolation adds a second, unrelated
# failure (torch is not in pip's isolated environment).  So the wheel is built
# here by one `bdist_wheel` run, and pip is handed the finished file.  See
# README.md, "Installation" / "MACA (MetaX)".
#
# Usage:
#     ./install.sh                             # build for this device, install
#     ./install.sh --targets xcore1600         # build for another architecture
#     ./install.sh --all                       # xcore1000,xcore1500,xcore1600
#     ./install.sh --build-only                # leave the wheel in dist/
#     ./install.sh --clean                     # rm -rf build dist first
#
# Env:
#     CUCC_TARGETS   targets to build (default native); explicit value wins
#     MACA_PATH      MACA toolkit root (default /opt/maca)
#     MAX_JOBS       forwarded to ninja as -j
#
set -euo pipefail

# ── project root ────────────────────────────────────────────────────────────
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

usage() {
    cat >&2 <<'EOF'
Usage: install.sh [options]

  --targets LIST   architectures to build (default: $CUCC_TARGETS or native)
  --all            xcore1000,xcore1500,xcore1600 -- every family this tree names
  --build-only     build the wheel but do not install it
  --clean          rm -rf build dist before building
  --no-link        do not symlink the built extension into deep_select/
  -h, --help       this message

Env: CUCC_TARGETS (default native), MACA_PATH (default /opt/maca), MAX_JOBS
EOF
}

# ── environment ─────────────────────────────────────────────────────────────
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"

# See build.sh: a second `--offload-arch` in one extension would carry a
# 128 KiB-sized config into a 64 KiB build, and setup.py refuses outright.
if [[ -n "${TORCH_EXTENSION_ENABLE_XC1500_COMPILE:-}" ]]; then
    echo "install.sh: clearing TORCH_EXTENSION_ENABLE_XC1500_COMPILE -- it makes" >&2
    echo "            torch append an architecture to every source; name the" >&2
    echo "            architectures with CUCC_TARGETS instead" >&2
    unset TORCH_EXTENSION_ENABLE_XC1500_COMPILE
fi

# ── arguments ───────────────────────────────────────────────────────────────
do_clean=0
build_only=0
do_link=1
targets_spec="${CUCC_TARGETS:-native}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --targets)
            [[ $# -ge 2 ]] || { echo "install.sh: --targets needs a value" >&2; exit 2; }
            targets_spec="$2"; shift 2 ;;
        --targets=*) targets_spec="${1#*=}"; shift ;;
        --all)       targets_spec="xcore1000,xcore1500,xcore1600"; shift ;;
        --build-only) build_only=1; shift ;;
        --no-link)   do_link=0; shift ;;
        --clean)     do_clean=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *)           echo "install.sh: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

# Resolve through the repo's own `_arch`, as build.sh does, so the targets this
# script reports and the targets setup.py builds are the same list.
resolved=$(CUCC_TARGETS="$targets_spec" python - <<'PY'
import os
from deep_select._arch import CAPACITY_BYTES, family_of_target, kernel_directory, resolve_targets

for target in resolve_targets(os.environ.get("CUCC_TARGETS")):
    family = family_of_target(target)
    print(f"{target} {family} {CAPACITY_BYTES[family] // 1024} "
          f"{kernel_directory(family)}")
PY
) || { echo "install.sh: could not resolve CUCC_TARGETS='$targets_spec'" >&2; exit 1; }

echo "install.sh: MACA_PATH=$MACA_PATH"
echo "install.sh: CUCC_TARGETS=$targets_spec"
while read -r target family kibs tree; do
    echo "install.sh:   $target -> xcore$family (${kibs} KiB/SM, csrc/$tree/)"
done <<<"$resolved"

# ── clean ───────────────────────────────────────────────────────────────────
if [[ "$do_clean" == 1 ]]; then
    echo "install.sh: rm -rf build dist"
    rm -rf build dist
fi
rm -rf ./*.egg-info

# ── stale in-place extensions ───────────────────────────────────────────────
# The install lands in site-packages, but a leftover in-place `.so` under
# `deep_select/` sits ahead of it on sys.path for anything run from the repo
# root (`PYTHONPATH=.` is how every test here is invoked), so it would shadow
# what was just installed.  Remove it, and build.sh will not be confused by it
# either.
while read -r target _family _kibs _tree; do
    for so in deep_select/deep_select_${target}*.so \
              build/lib.*/deep_select/deep_select_${target}*.so; do
        if [[ -e "$so" ]]; then
            echo "install.sh: rm -f $so"
            rm -f "$so"
        fi
    done
done <<<"$resolved"

# ── build the wheel ─────────────────────────────────────────────────────────
export CUCC_TARGETS="$targets_spec"

which python
which pip
python -c 'import sys, torch
print(f"install.sh: python {sys.version.split()[0]}, torch {torch.__version__}")'

# `setup.py` derives the wheel's version from `datetime.now()`, so hoisting the
# version off the freshly written filename and putting it back with `mv` makes
# pip's filename-vs-metadata check independent of how long the build took.
# (`pip install .` cannot do this: it is the two-invocation split that breaks.)
dist_dir="$script_dir/dist"
rm -rf "$dist_dir"
mkdir -p "$dist_dir"

if [[ -n "${MAX_JOBS:-}" ]]; then
    python setup.py bdist_wheel --dist-dir "$dist_dir" -j "$MAX_JOBS"
else
    python setup.py bdist_wheel --dist-dir "$dist_dir"
fi

wheel=$(ls "$dist_dir"/deep_select-*.whl 2>/dev/null | head -1 || true)
if [[ -z "$wheel" ]]; then
    echo "install.sh: ERROR no wheel was produced in $dist_dir" >&2
    exit 1
fi

# The version a run records is `__version__+git_rev.datetime_rev`; the two
# short derived pieces are what must agree with the metadata.
version=$(python - <<'PY'
import os
from deep_select.__version__ import __version__

rev = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or "unknown"
print(f"{__version__}+{rev}")
PY
)
# Compare on the version's own fields rather than by string-stripping a prefix:
# the local version segment contains dots, so a prefix strip does not stop at
# the field boundary.  Splitting on "-" does -- the name is the first field,
# the version is *either* field 1 (a build normalizes it, dropping
# `setup.py`'s `+git_rev.datetime_rev` local segment) *or* field 1 plus the
# second field (when it survives).  Only the first shape needs repairing, and
# then only to drop a stale `datetime_rev`; the second already matches the
# metadata and is left alone.
base=$(basename "$wheel")
fixed=$(python - "$base" "$version" <<'PY'
import sys

fields = sys.argv[1].split("-")
if len(fields) < 3 or "." in fields[1]:
    sys.exit(0)                        # version already spans the field; leave
print("-".join(["deep_select", sys.argv[2]] + fields[2:]))
PY
)
if [[ -n "$fixed" ]]; then
    echo "install.sh: wheel version drift, renaming to match metadata:"
    echo "install.sh:   $fixed"
    mv "$wheel" "$dist_dir/$fixed"
    wheel="$dist_dir/$fixed"
fi
echo "install.sh: built $(basename "$wheel")"

# Verify the targets got in before installing anything.  A wheel built for one
# architecture and deployed to another fails at import, not here.
missing=0
for line in "${resolved[@]}"; do
    read -r target _family _kibs _tree <<<"$line"
    if unzip -l "$wheel" | grep -q "deep_select/deep_select_${target}"; then
        echo "install.sh: OK  wheel contains $target"
    else
        echo "install.sh: ERROR wheel has no extension for $target" >&2
        missing=1
    fi
done
[[ "$missing" == 0 ]] || exit 1

if [[ "$build_only" == 1 ]]; then
    echo "install.sh: --build-only, leaving the wheel at $wheel"
    exit 0
fi

# ── install ─────────────────────────────────────────────────────────────────
# --force-reinstall so a rebuild overwrites the previous one (the wheel's local
# version segment changes every build, but a same-second rebuild would not) and
# --no-deps because the MACA-patched torch is the environment's job, as in the
# host repository's install.sh.
pip install "$wheel" --force-reinstall --no-deps

# Confirm what the environment now resolves, from a directory that is not the
# repo -- otherwise the check reports the repo's own package and passes even
# when the install did nothing.
( cd /tmp && python -c '
import deep_select, os
print("install.sh: installed", deep_select.__version__)
print("install.sh: from     ", os.path.dirname(deep_select.__file__))
' )

# ── link into the repo ──────────────────────────────────────────────────────
# Every test in this repository is run with `PYTHONPATH=.` from the repo root
# (README "Testing"), which resolves `deep_select` to the *repo* copy first --
# so an installed wheel alone would not be what the suites exercise.  Symlink
# the wheel's extension back under `deep_select/` so the repo and the
# environment are the same build.  A symlink, not a copy: the build.sh stale-
# binary trap (an in-place `.so` newer than its source, skipped by
# `build_ext --inplace`'s timestamp check) needs the file to be visibly a
# link, and a re-`build.sh` overwrites the target rather than fighting it.
if [[ "$do_link" == 1 ]]; then
    # Ask pip where the install actually went.  Not `import deep_select` from
    # the repo root: `PYTHONPATH=.` puts the repo's own (uninstalled) package
    # first on sys.path, so that reports the repo back and the link becomes a
    # symlink onto itself.
    site_pkgs=$(pip show deep_select | sed -n 's/^Location: //p' | head -1)
    if [[ -z "$site_pkgs" || ! -d "$site_pkgs/deep_select" ]]; then
        echo "install.sh: ERROR cannot locate the installed package to link" >&2
        exit 1
    fi
    while read -r target _family _kibs _tree; do
        target_so=$(ls "$site_pkgs/deep_select/deep_select_${target}"*.so 2>/dev/null | head -1 || true)
        if [[ -z "$target_so" ]]; then
            echo "install.sh: ERROR the installed package has no $target extension" >&2
            exit 1
        fi
        name=$(basename "$target_so")
        link="$script_dir/deep_select/$name"
        if [[ "$target_so" == "$link" ]]; then
            echo "install.sh: ERROR refusing to link $name onto itself" >&2
            exit 1
        fi
        rm -f "$link"
        ln -s "$target_so" "$link"
        echo "install.sh: linked deep_select/$name -> $site_pkgs"
    done <<<"$resolved"
fi

echo "install.sh: done"
cd "$original_dir"
