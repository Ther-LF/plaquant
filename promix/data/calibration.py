"""Calibration data loading utilities."""

import sys
import os

_resq_path = os.path.join(os.path.dirname(__file__), '../../project-resq/fake_quant')
if _resq_path not in sys.path:
    sys.path.insert(0, _resq_path)

from utils.data_utils import get_loaders

__all__ = ['get_loaders']
