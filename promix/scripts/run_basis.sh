#!/bin/bash
# Run PCA basis computation
# Uses project-resq's get_basis.py (Phase 1: delegate)
cd $(dirname $0)/../../project-resq/fake_quant
bash 0_get_basis.sh
