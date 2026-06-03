"""PCA basis computation — identify high-variance subspace per layer.

Wraps project-resq's get_basis.py logic.
"""

import sys
import os

_resq_path = os.path.join(os.path.dirname(__file__), '../../project-resq/fake_quant')
if _resq_path not in sys.path:
    sys.path.insert(0, _resq_path)

# get_basis.py is a script, not a module — we import the key function if available
# For Phase 1, basis computation uses project-resq directly via scripts/run_basis.sh

__all__ = []
