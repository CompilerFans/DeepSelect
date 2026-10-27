"""`deep_select_cuda` on this platform: an alias of `deep_select_maca`.

The official wheel names its backends for the platform (`deep_select_cuda`,
`deep_select_npu`).  A framework written against that wheel imports, or probes
for, the CUDA name; on a MACA build the same backend answers to it, so the
framework's platform arm needs no fork-specific spelling.
"""
from .deep_select_maca import get_alignment_requirement, topk

__all__ = ["get_alignment_requirement", "topk"]
