#!/bin/bash
# Run rotation matrix optimization
# Uses project-resq's optimize_rotation.py (Phase 1: delegate)
cd $(dirname $0)/../../project-resq/fake_quant
bash 1_optimize_rotation.sh
