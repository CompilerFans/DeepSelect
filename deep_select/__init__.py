from .__version__ import __version__

from ._backends import deep_gemm_available, topk_deep_gemm, topk_torch
from .interface import UnsupportedByBackend, get_stride_requirement, topk
