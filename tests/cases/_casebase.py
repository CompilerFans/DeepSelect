"""The bootstrap and helpers the `tests/cases/` scripts share.

**These scripts measure the checkout, not the installed wheel.**  Importing
this module puts the *repository's* `tests/` directory on `sys.path`, which is
the only way a `deep_select` import can resolve to a tree that also carries
`kernelkit` and `lib` -- the two modules every case here is built on.  That is
deliberate, and it is also the trap: a bare `python tests/cases/foo.py` run
from the repository root gets `.` on `sys.path` from the interpreter, and with
the package importable from site-packages the name can resolve there instead.
Run these with `PYTHONPATH=.` and read the `artifact=` line every one of them
prints, or `run_bench.sh`, which resolves the extension the same way and
records its md5 in the run header.

Two helpers, because both scripts had copied them:

* `build_case` -- the seeded `(TestParam, Testcase)` pair, on the device.
* `call_topk` -- `tests/test.py`'s `run_topk_select`: a case exercises the same
  call the official suite makes, not a hand-written one.  The same argument as
  the official *predicate* these scripts already use.

The device setup and the path insert run at **import** time: they have to
precede `import kernelkit`/`lib`/`test`, which a module offering only functions
could not do.
"""
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "tests"))
torch.set_default_device("cuda")

# Imported here and re-exported: the order is the bootstrap (the path insert
# above has to precede all four), and a consumer that imported them itself
# would be reproducing it.
import kernelkit as kk                                    # noqa: E402, F401
import deep_select                                        # noqa: E402, F401
import lib                                                # noqa: E402, F401
import test as suite                                      # noqa: E402, F401
from lib import (TestParam, NormalFloatDistribution,      # noqa: E402, F401
                 UniformUIntDistribution,
                 UintDistributionWithHotspotAndSpecifiedPivot, get_fp_config)

REPO = _REPO


def build_case(b, V, k, *, end=False, si=True, sv=False, rv=False,
               idx_dtype=torch.int32, distrib=None, seed=3):
    """The `(TestParam, Testcase)` pair, seeded and on the device.

    fp32 only: both scripts measure the fp32 routes (`f32_chunks_applies`,
    `f32_coarse12_applies`), and `sv` is the one fp32-only ordering.
    """
    p = TestParam(batch_size=b, vocab_size=V, topk=k, sorted_value=sv,
                  sorted_index=si, return_value=rv, dtype=torch.float32,
                  out_idx_dtype=idx_dtype, enable_end_position=end,
                  input_distrib=distrib or NormalFloatDistribution(), seed=seed,
                  idx_oob_fill_value=-2000000 + V)
    kk.set_random_seed(seed)
    t = lib.generate_testcase(p)
    t.input = t.input.to("cuda")
    if t.end is not None:
        t.end = t.end.to("cuda")
    return p, t


def call_topk(p, t, backend="maca_c"):
    """`tests/test.py`'s official call site for this case."""
    return suite.run_topk_select(p, t, backend)
