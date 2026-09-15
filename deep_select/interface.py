import functools
import os
import torch

from typing import Optional, Tuple

from . import _binding
from ._arch import FAMILY_OF_TARGET, native_target


# The public backend names.  `maca_c` is the MACA kernel for this device.
# Which kernel that is -- the hand-written one on a 64 KiB part, the ported one
# on a 128 KiB part -- is a property of the device, not a choice a caller
# makes, so the name does not name an architecture (the host repository's
# `backend=` names implementations the same way, e.g. `maca_c`, `mctlassEx`).
# `torch` is not a kernel: it is a reference implementation of the same
# contract, and runs anywhere.  `deep_gemm` is the host repository's own MACA
# selector, which implements a subset of the contract -- see
# `topk_deep_gemm` for exactly which part.
_BACKENDS = ("maca_c", "torch", "deep_gemm")

# What `topk` runs when the caller does not choose.  `torch` is deliberate: it
# is the implementation of this contract that does not depend on a kernel
# being correct on the device in front of you, and the kernel path is validated
# *against* it -- so making it the default costs no coverage and removes the
# last route by which a caller who asked for nothing in particular could reach
# a kernel defect.  The cost is speed: the torch arm is the slow one (its
# largest official cell peaks at 16.4 GiB and runs in seconds, not milliseconds).
def _default_backend() -> str:
    """`DS_TOPK_BACKEND` if it names a backend, else `"torch"`.

    Explicit rather than implied, and stated in the docstring, because the
    production route and the default route are now different: a caller who
    wants the kernel this device has asks for `"maca_c"`, or sets
    `DS_TOPK_BACKEND=maca_c` for a whole process.  An unrecognized value is
    ignored rather than raised -- this is a default, and a typo in an
    environment variable must not break every call.
    """
    v = os.environ.get("DS_TOPK_BACKEND")
    if v and v in _BACKENDS:
        return v
    return "torch"


class UnsupportedByBackend(ValueError):
    """This backend does not implement this part of the contract.

    Distinct from the `ValueError` a bad argument gets: the request is valid,
    the operator offers it, and the chosen implementation is simply narrower --
    `deep_gemm` is float32-only, for instance, while the operator takes
    bfloat16 too.  Keeping them apart lets a caller (or a test table) report
    "outside this backend's contract" instead of "wrong call", and it is still
    a `ValueError` for anyone who only cares that the call was refused.
    """

# The kernels the build produces, one per architecture -- setup.py builds one
# extension per architecture, `deep_select.deep_select_xcore<N>`, holding the
# kernel whose shared memory capacity fits it.  Internal: this is how `maca_c`
# resolves on a given device, and it is the vocabulary of the build, not of the
# API.
_KERNEL_NAMES = tuple(f"xcore{family}"
                      for family in sorted(set(FAMILY_OF_TARGET.values())))


@functools.lru_cache(maxsize=None)
def _backend_for(name: str):
    """The loaded tvm-ffi extension that implements the kernel `name`.

    Loaded on demand and cached (both here and in `_binding`): the module
    holds a device binary, and a process that never calls `topk` on a kernel
    should not load one.
    """
    assert name in _KERNEL_NAMES, name
    return _binding.load(name)


# `structs.h`'s INPUT_STRIDE_ALIGNMENT_REQUIREMENT and OUTPUT_STRIDE_ALIGNMENT_
# REQUIREMENT.  A built kernel exports them through `get_alignment_requirement()`
# and that is where the value comes from whenever one is available; these are
# the same numbers for when none is, so that `backend="torch"` works on a
# machine with no kernel built at all.  A fallback to a constant is sound here
# because the alignment is the operator's contract -- identical for every
# kernel -- rather than any one build's property.
_ALIGNMENT_REQUIREMENT_BYTES = (1024, 32)


@functools.lru_cache(maxsize=1)
def get_stride_requirement() -> Tuple[int, int]:
    """
    Returns the stride requirement for input / output tensors, in bytes
    """
    try:
        # The FFI entry returns `Array<int64_t>` (a python list); the pybind11
        # one returned a C++ pair, which pybind converted to a tuple for free.
        # Normalize here so the rest of this module sees the tuple either way.
        pair = _backend_for(native_target()).get_alignment_requirement()
        return (int(pair[0]), int(pair[1]))
    except RuntimeError:
        return _ALIGNMENT_REQUIREMENT_BYTES


def topk(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    begin: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
    indices_type: torch.dtype = torch.int64,
    sorted_index: bool = False,
    hint: Optional[torch.Tensor] = None,
    output_idx: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
    backend: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Arguments:
        input: (b, vocab_size), dtype=torch.bfloat16/torch.float. stride(0) must be a multiple of `deep_select.get_stride_requirement()[0]` bytes, and stride(1) must be 1.
        topk: int. Select topk elements for each row.
        sorted: bool. Whether to return sorted **output_val**. fp32 on the
                upstream contract; the MACA kernel also supports bfloat16.
        begin(optional): (b,), dtype=int32. CURRENTLY NOT SUPPORTED. The left(inclusive) range for input row, default is 0.
        end(optional): (b,), dtype=int32. The right(exclusive) range for input row, default is vocab_size. The stride of this tensor must be 1.
                       Note when end[i] <= topk, valid elements will be gathered at the beginning of values and indices returned. The rest of `values` will be filled with `value_oob_fill_value`, while the rest of `indices` will be filled with `idx_oob_fill_value` (won't be plused by `output_idx_offset`).
                       `end` <= `vocab_size` must be held
        indices_type: torch.dtype. The output indices dtype, only support torch.int32 and torch.int64.
        sorted_index: bool. Whether to return sorted **output_idx**.
        hint(optional): CURRENTLY NOT SUPPORTED
        output_idx(optional): (b, topk), dtype=indices_type. A contiguous tensor to store output.
        output_idx_offset(optional): (b,), dtype=int32. If provided, all output_idx (`idx_oob_fill_value` not included) will += output_idx_offset.
        idx_oob_fill_value: int. See comments above when end[i]-begin[i]<topk.
        return_value: bool. If False, only return indices without values to accelerate the kernel. The return value is still a Tuple, but the first element will be None.
        abort_when_nan_found: bool. When a NaN is found, if True, aborts the whole kernel; if False, writes 0x3F3F3F3F to the corresponding output_idx[batch_idx][0] and exits.
                The NaN check itself is always enabled. Exception: when the row's length <= topk, it is skipped.
        backend: str. Which implementation to run.  `"torch"` (the default) is a
                reference implementation of the contract built on torch ops:
                it runs on any device and dtype, including where no kernel is
                built, and it is the arm to trust when the question is what
                the *answer* should be.  `"maca_c"` is the MACA kernel this
                device has -- the hand-written kernel on a 64 KiB part, the
                ported one on a 128 KiB part, which of the two being a
                property of the device rather than a choice, so the name does
                not name an architecture.  It is the production path and the
                fast one, and it is what `DS_TOPK_BACKEND=maca_c` restores
                (see below).  `"deep_gemm"` is the host repository's selector,
                faster again on long rows, implementing a subset of this
                contract; what it cannot serve raises `UnsupportedByBackend`
                (see `topk_deep_gemm`).

                **The default is correctness-first, not speed-first.**  The
                kernel path is validated against this one (a differential
                check, plus the official contract suite with `backend=` bound
                by `scripts/official_slice.py`), so leaving it out of the
                default costs nothing in coverage and removes the last place a
                kernel defect could be reached by a caller who did not ask for
                one.  `DS_TOPK_BACKEND` overrides the default for a whole
                process -- the escape hatch for production, for a perf run
                (the benchmark arms set it rather than editing any call site;
                `scripts/perf_snapshot.py` names its backend per arm by
                construction), and for bisecting.  An unrecognized value is
                ignored, so the default still governs.

    Return:
        output_val: (b, topk), dtype=input.dtype.
        output_idx: (b, topk), dtype=indices_type.
                    The output tensors may not be contiguous, when topk * sizeof(input.dtype or indices_dtype) is not a multiple of 32 Bytes
    """

    # `None` means "take the process default", which is `torch` unless
    # `DS_TOPK_BACKEND` says otherwise -- see the docstring.  Read here rather
    # than in the signature so the environment is consulted per call (a
    # process that sets it late still gets it) and so `backend=None` cannot be
    # mistaken for a caller error.
    if backend is None:
        backend = _default_backend()
    # Checked before anything is read off `input`: a caller who mistyped a
    # backend name should hear about that, not about whatever the None they
    # passed in place of a tensor does to the next line.
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unsupported backend: {backend!r}. Expected one of "
            f"{', '.join(_BACKENDS)} -- 'maca_c' runs the kernel this device "
            f"has, 'torch' the reference implementation"
        )

    N = input.shape[0]

    def get_empty_and_aligned_tensor(dim0: int, dim1: int, device: torch.device, dtype: torch.dtype):
        """
        Return a tensor with shape (dim0, dim1), and stride (X, 1), where X is a multiple of 32B
        """
        output_stride_requirement_bytes = get_stride_requirement()[1]
        output_stride_requirement = output_stride_requirement_bytes // dtype.itemsize
        assert output_stride_requirement > 0
        dim1_rounded = (dim1+output_stride_requirement-1) // output_stride_requirement * output_stride_requirement
        return torch.empty((dim0, dim1_rounded), device=device, dtype=dtype)[:, :dim1]
    
    output_val = get_empty_and_aligned_tensor(N, topk, device=input.device, dtype=input.dtype) if return_value else None
    if output_idx is None:
        output_idx = get_empty_and_aligned_tensor(N, topk, device=input.device, dtype=indices_type)
    else:
        assert output_idx.dtype == indices_type

    assert begin is None, "`begin` is not supported now"
    assert hint is None, "`hint` is not supported now"

    if backend == "torch":
        return topk_torch(
            input, topk, sorted=sorted, end=end, indices_type=indices_type,
            sorted_index=sorted_index, output_idx=output_idx,
            output_idx_offset=output_idx_offset,
            idx_oob_fill_value=idx_oob_fill_value,
            value_oob_fill_value=value_oob_fill_value,
            return_value=return_value,
            abort_when_nan_found=abort_when_nan_found,
        )
    if backend == "deep_gemm":
        return topk_deep_gemm(
            input, topk, end=end, sorted_value=sorted,
            sorted_index=sorted_index, indices_type=indices_type,
            output_idx=output_idx, output_idx_offset=output_idx_offset,
            idx_oob_fill_value=idx_oob_fill_value,
            value_oob_fill_value=value_oob_fill_value,
            return_value=return_value,
            abort_when_nan_found=abort_when_nan_found,
        )
    else:
        # `kernels=[...]` is the public `aoti_torch`-style indirection a caller
        # may pass to select particular implementations; it is not part of the
        # contract this operator publishes, and the kernel pick is a property
        # of the device, so nothing here reads it.
        #
        # The tvm-ffi entry takes 12 positional arguments, all DLTensor or
        # plain scalar: `begin` is rejected by this function above and `hint` is
        # not supported, so neither crosses the boundary -- the old pybind11
        # entry passed them as empty optionals and the kernel layer ignored
        # them.  `output_val` is ``None`` when `return_value` is False, which is
        # what `Optional<TensorView>` carries.
        backend_args = (
            input,
            topk,
            end,
            sorted, sorted_index,
            output_val, output_idx,
            output_idx_offset,
            idx_oob_fill_value,
            value_oob_fill_value,
            return_value,
            abort_when_nan_found,
        )
        # `maca_c` means this device's kernel: which extension that is gets
        # resolved here rather than in the signature, which keeps `torch`
        # usable on a machine with no MACA device at all.
        # `launching()` installs torch's current stream into the FFI
        # environment for the duration of the call.  The C++ side reads it
        # through TVMFFIEnvGetStream; without this it would see the null
        # handle and launch on the legacy default stream.
        # The env variable is the explicit escape hatch: the workaround is
        # correct but unusual, so it is named rather than silent.
        if os.environ.get("DEEP_SELECT_NO_STREAM_GUARD"):
            _backend_for(native_target()).topk(*backend_args)
        else:
            with _binding.launching():
                _backend_for(native_target()).topk(*backend_args)
        return output_val, output_idx


# `deep_gemm.kernels.fp32_topk`'s `kMaxTopK`.  Its selector refuses anything
# larger, so this backend refuses first, with a message that names the gap
# rather than the kernel's assert.
_DEEP_GEMM_MAX_TOPK = 2048


@functools.lru_cache(maxsize=1)
def _deep_gemm():
    """The host repository's package, imported on first use.

    A soft dependency: this repository is standalone -- it is also vendored as
    a submodule of that same host -- so this is the only place it is imported,
    and only when `backend="deep_gemm"` is actually asked for.
    """
    import deep_gemm

    return deep_gemm


def topk_deep_gemm(
    input: torch.Tensor,
    topk: int,
    end: Optional[torch.Tensor] = None,
    sorted_value: bool = False,
    sorted_index: bool = False,
    indices_type: torch.dtype = torch.int64,
    output_idx: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """`topk` served by the host repository's indexer selector.

    That kernel (`deep_gemm.fp32_indexer_topk_selector`, `topk_coarse12` /
    `topk_chunks`) solves the same problem this repository does, and faster on
    the long-row shapes -- it is the obvious thing to route to when the shapes
    suit it.  It implements a strict subset of the contract, so it is offered
    as a backend rather than as a replacement, and what it cannot serve raises
    `UnsupportedByBackend` instead of quietly answering something narrower:

        * float32 scores only (no bfloat16);
        * `topk <= 2048`;
        * an unordered selection (`sorted_value` / `sorted_index` need the
          kernels);
        * nothing else: `end`, `output_idx`, `output_idx_offset`, the
          out-of-band fills, `return_value` and the NaN contract are all
          implemented here, on top of it.

    Exactness is the host kernel's, and it has one known hole: a kernel
    collects the members of the threshold *coarse* bin -- the half-precision
    ordered key >> 6, so everything inside one 64-half-ULP bucket, which a row
    of near-tied scores fills end to end -- before refining, and the chunked
    kernel (small batch, long row) silently drops members past its staging
    capacity instead of re-scanning.  A row with more than 4096 values in one
    such bucket can therefore get a top-k of an arbitrary subset of it, varying
    run to run.  `maca_c` has no such hole; in the host repository it is filed
    as a strict `xfail` in
    `deep_gemm/tests/test_indexer_topk_selector.py::test_selector_candidate_overflow`.

    The window is the same one: the selector's `seq_lens` with a single row per
    group is a per-row exclusive upper bound (`torch_topk_selector.py:23`),
    which is what `end` is, and its indices are relative to a `seq_starts` of
    zero -- i.e. absolute -- so only the padding needs translating (`-1` there,
    `idx_oob_fill_value` here).  Its values are a gather of the input, not a
    kernel output, which is also how this returns them.
    """
    try:
        module = _deep_gemm()
    except ImportError as exc:
        raise RuntimeError(
            "backend 'deep_gemm' needs the mcDeepGEMM package importable "
            "(install it, or put its repository root on PYTHONPATH); this "
            "repository runs standalone too, and that is the only case where "
            "the import is missing"
        ) from exc

    # The operator's own contract rejections, as `topk_torch` enforces them:
    # they are properties of the contract, not of an implementation, so a
    # caller must hear the same thing whichever backend it picked.
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if input.dim() != 2:
        raise ValueError(f"input must be 2-D, got {input.dim()} dimensions")
    if input.stride(1) != 1:
        raise ValueError("input.stride(1) must be 1")
    if indices_type not in (torch.int32, torch.int64):
        raise ValueError(
            f"indices_type must be int32 or int64, got {indices_type}")
    if input.dtype != torch.float32:
        raise UnsupportedByBackend(
            f"backend 'deep_gemm' ranks float32 scores, and was given "
            f"{input.dtype}; 'maca_c' and 'torch' take bfloat16 as well")
    if topk > _DEEP_GEMM_MAX_TOPK:
        raise UnsupportedByBackend(
            f"backend 'deep_gemm' selects at most {_DEEP_GEMM_MAX_TOPK} per "
            f"row and was asked for {topk}; 'maca_c' goes to 4096")
    if sorted_value or sorted_index:
        raise UnsupportedByBackend(
            "backend 'deep_gemm' returns an unordered selection; "
            "`sorted_value`/`sorted_index` need backend 'maca_c'")

    n_rows, vocab_size = input.shape
    device = input.device
    if input.stride(0) != vocab_size:
        # The host kernel walks a row as `scores + row * n_cols`, i.e. it
        # assumes tightly packed rows; a padded row stride would quietly rank
        # the wrong elements.  Padding is legal in this contract
        # (`get_stride_requirement`), so normalize instead of refusing.
        input = input.contiguous()
    lengths = (torch.full((n_rows,), vocab_size, dtype=torch.int64,
                          device=device)
               if end is None
               else end.to(torch.int64).clamp(min=0, max=vocab_size))

    # The NaN contract, checked as `topk_torch` checks it: always on, and
    # skipped for a row whose window is not longer than `topk`.
    if bool((lengths > topk).any().item()):
        cols = torch.arange(vocab_size, device=device)
        nan_rows = (torch.isnan(input)
                    & (cols.unsqueeze(0) < lengths.unsqueeze(1))).any(dim=1)
        if bool(nan_rows.any().item()):
            if abort_when_nan_found:
                raise RuntimeError("NaN detected in the input")
            if output_idx is None:
                output_idx = torch.empty((n_rows, topk), dtype=indices_type,
                                         device=device)
            output_idx[nan_rows, 0] = 0x3F3F3F3F
            values = None
            if return_value:
                values = torch.full((n_rows, topk), value_oob_fill_value,
                                    dtype=input.dtype, device=device)
            return values, output_idx

    selected = module.fp32_indexer_topk_selector(
        input, lengths.to(torch.int32), topk, return_val=False,
        backend="maca_c")["indices"]
    valid = selected >= 0

    # `-1` marks the padding; everything else is already the column index.
    # (`.to` because the selector answers in int32 whatever `indices_type` is.)
    columns = selected.to(torch.int64)
    if output_idx_offset is not None:
        columns = columns + output_idx_offset.to(torch.int64).unsqueeze(1)
    out_idx = torch.where(valid, columns,
                          torch.full_like(columns, idx_oob_fill_value)
                          ).to(indices_type)

    out_val = None
    if return_value:
        gathered = input.gather(1, selected.clamp_min(0).to(torch.int64))
        out_val = torch.where(valid, gathered,
                              torch.full_like(gathered, value_oob_fill_value))

    if output_idx is not None:
        output_idx.copy_(out_idx)
        out_idx = output_idx
    return out_val, out_idx


def topk_torch(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    end: Optional[torch.Tensor] = None,
    indices_type: torch.dtype = torch.int64,
    sorted_index: bool = False,
    output_idx: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Reference `torch` implementation of the `topk` contract.

    Same contract as `topk`, expressed with torch ops only, so it runs on any
    device and any dtype.  Rows longer than `topk` are handled by masking the
    out-of-window tail to `-inf` and taking one `torch.topk` over the padded
    row, which reproduces the per-row window without a loop over rows.
    """
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if topk > 4096:
        raise ValueError(f"topk must be <= 4096, got {topk}")
    # The same contract rejections the kernel path enforces (upstream:
    # csrc/xcore1600/api.cu).  They are part of the operator's contract, not of any one
    # implementation -- a caller that passes a strided view must get an error
    # rather than a silently different answer just because it chose this
    # backend.
    #
    # Note what is *not* here: upstream restricts `sorted` to float32, and the
    # ported kernel keeps that restriction, but the MACA-native one
    # (`csrc/xcore1000/maca_topk.cu`) implements it for bfloat16 too.  The
    # reference describes the operator this repository ships, so it follows the
    # wider contract; `torch.topk` orders bfloat16 natively, so nothing extra
    # is needed for it.
    if sorted and not return_value:
        raise ValueError("`return_value` must be enabled when `sorted` is True")
    if sorted and sorted_index:
        raise ValueError("`sorted` and `sorted_index` cannot be used at the same time")
    if input.dim() != 2:
        raise ValueError(f"input must be 2-D, got {input.dim()} dimensions")
    if input.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"input dtype must be float32 or bfloat16, got {input.dtype}")
    if input.stride(1) != 1:
        raise ValueError("input.stride(1) must be 1")
    if indices_type not in (torch.int32, torch.int64):
        raise ValueError(f"indices_type must be int32 or int64, got {indices_type}")
    if output_idx is not None:
        if output_idx.dtype != indices_type:
            raise ValueError(f"output_idx dtype must be {indices_type}, got {output_idx.dtype}")
        if output_idx.stride(1) != 1:
            raise ValueError("output_idx.stride(1) must be 1")
        if (output_idx.size(0) < input.shape[0]
                or output_idx.size(1) < topk):
            raise ValueError(
                f"output_idx must be at least (batch_size, topk) = "
                f"({input.shape[0]}, {topk}), got {tuple(output_idx.shape)}")

    n_rows, vocab_size = input.shape
    device = input.device

    # `end` is an exclusive per-row upper bound.  The kernel never NaN-checks
    # a row whose visible length is <= topk; the check itself is always on.
    if end is not None:
        lengths = end.to(torch.int64).clamp(min=0, max=vocab_size)
    else:
        lengths = torch.full((n_rows,), vocab_size, dtype=torch.int64, device=device)

    nan_rows = torch.zeros(n_rows, dtype=torch.bool, device=device)
    checked = lengths > topk
    if bool(checked.any().item()):
        cols = torch.arange(vocab_size, device=device)
        visible = torch.isnan(input.float()) & (cols.unsqueeze(0) < lengths.unsqueeze(1))
        nan_rows = visible.any(dim=1)
    if abort_when_nan_found and bool(nan_rows.any().item()):
        raise RuntimeError("NaN detected in the input")
    # A NaN row is answered by the guard in its first index slot and is
    # otherwise left unspecified, which is how the kernel path leaves it -- but
    # *that row only*.  Returning here for the whole batch, as this did, leaves
    # every clean row of the batch undefined too: the checks that matter run
    # per row, so a batch that contains one NaN row was answering every other
    # row with whatever the output buffer happened to hold.

    # Rows shorter than topk select their whole visible prefix and are padded
    # with the fill values; masking the hidden tail to -inf makes a single
    # torch.topk reproduce that.
    work = input
    if bool((lengths < vocab_size).any().item()):
        cols = torch.arange(vocab_size, device=device)
        work = input.masked_fill(cols.unsqueeze(0) >= lengths.unsqueeze(1),
                                 float("-inf"))

    k_eff = min(topk, vocab_size)
    values, indices = torch.topk(work, k_eff, dim=1, sorted=bool(sorted))

    # The window mask is `-inf`, which is also a value a row can hold: a hidden
    # slot then ties with the visible ones and `torch.topk`'s tie order is
    # unspecified, so it can hand back an index outside the window.
    #
    # Such a row is answered by re-picking from its window alone.  Rewriting the
    # row's indices to the window's first `k_eff` *positions* instead -- which
    # is what this did -- keeps the values sorted and the indices in window, but
    # splits the two: `value_i` is `torch.topk`'s i-th largest while `index_i`
    # is position i, so the row reports a value it does not hold at the index it
    # reports (`3.0, 2.0, 1.0` against indices `0, 1, 2` of the row
    # `3.0, 1.0, 2.0`).  A hidden slot can only be selected when the window
    # holds fewer than `k_eff` values above `-inf`, so `min(k_eff, length)`
    # entries is all the window can answer with, and the rest of the row stays
    # padding, masked into the fill values below.
    in_window = indices < lengths.unsqueeze(1)
    if bool((~in_window).any().item()):
        for row in (~in_window).any(dim=1).nonzero().flatten().tolist():
            length = int(lengths[row])
            picked = min(k_eff, length)
            if picked == 0:
                continue
            row_values, row_indices = torch.topk(
                input[row, :length], picked, sorted=bool(sorted)
            )
            values[row, :picked] = row_values
            indices[row, :picked] = row_indices

    # Padding picked up by torch.topk on short rows becomes the fill value.
    valid = torch.arange(k_eff, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    out_idx = torch.full((n_rows, topk), idx_oob_fill_value,
                         dtype=indices_type, device=device)
    idx_sel = indices.to(indices_type)
    if output_idx_offset is not None:
        idx_sel = idx_sel + output_idx_offset.to(indices_type).unsqueeze(1)
    out_idx[:, :k_eff] = torch.where(valid, idx_sel,
                                     torch.full_like(idx_sel, idx_oob_fill_value))

    out_val = None
    if return_value:
        out_val = torch.full((n_rows, topk), value_oob_fill_value,
                             dtype=input.dtype, device=device)
        out_val[:, :k_eff] = torch.where(valid, values.to(input.dtype),
                                         out_val[:, :k_eff])

    if sorted_index:
        # `torch.gather` requires an int64 index regardless of the output dtype.
        #
        # Order the *selected* indices only, and send the padding to the end
        # explicitly: a selected slot holds `index + output_idx_offset` while a
        # padding slot holds the raw `idx_oob_fill_value`, so sorting the row as
        # it stands orders two different currencies -- and whenever the offset
        # makes a selected value larger than the fill (or, for a negative
        # offset, smaller) the padding lands in front, in the slots the contract
        # requires to be real in-window indices.
        sel = (torch.arange(topk, device=device).unsqueeze(0)
               < lengths.unsqueeze(1))
        key = torch.where(sel, out_idx.to(torch.int64),
                          torch.iinfo(torch.int64).max)
        order = torch.argsort(key, dim=1, stable=True)
        out_idx = torch.gather(out_idx, 1, order)
        if return_value:
            out_val = torch.gather(out_val, 1, order)

    if output_idx is not None:
        output_idx.copy_(out_idx)
        out_idx = output_idx

    # Stamped on the tensor that goes back to the caller, whichever it is.
    if bool(nan_rows.any().item()):
        out_idx[nan_rows, 0] = 0x3F3F3F3F

    return out_val, out_idx
