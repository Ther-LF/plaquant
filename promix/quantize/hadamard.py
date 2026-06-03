"""Hadamard transform utilities.

Thin wrapper that imports from project-resq's implementation to avoid
duplicating hundreds of lines of hardcoded Hadamard matrix constants.
"""

import sys
import os

# Add project-resq to path for importing
_resq_path = os.path.join(os.path.dirname(__file__), '../../project-resq/fake_quant')
if _resq_path not in sys.path:
    sys.path.insert(0, _resq_path)

from utils.hadamard_utils import (
    get_hadK,
    matmul_hadU_cuda,
    random_orthogonal_matrix,
    is_pow2,
)
from utils.utils import HadamardTransform

__all__ = [
    'get_hadK',
    'matmul_hadU_cuda',
    'random_orthogonal_matrix',
    'is_pow2',
    'HadamardTransform',
]
