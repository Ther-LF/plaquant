"""Rotation utilities — apply PCA basis and optimized rotation to model.

Wraps project-resq's rotation_utils which handles:
- Loading PCA basis (U matrix)
- Loading optimized rotation (R matrix)
- Fusing basis into model weights
- Rearranging columns by variance ordering
"""

import sys
import os

_resq_path = os.path.join(os.path.dirname(__file__), '../../project-resq/fake_quant')
if _resq_path not in sys.path:
    sys.path.insert(0, _resq_path)

from eval_utils.rotation_utils import (
    fuse_basis_to_model,
    rotate_model,
    rearrange_columns,
)

__all__ = [
    'fuse_basis_to_model',
    'rotate_model',
    'rearrange_columns',
]
