"""RMSNorm fusion utilities.

Fuses RMSNorm/LayerNorm weights into adjacent linear layers
so that the model can operate without explicit norm computation
(the norm is absorbed into the rotation + quantization pipeline).
"""

import sys
import os

_resq_path = os.path.join(os.path.dirname(__file__), '../../project-resq/fake_quant')
if _resq_path not in sys.path:
    sys.path.insert(0, _resq_path)

from utils.fuse_norm_utils import fuse_layer_norms

__all__ = ['fuse_layer_norms']
