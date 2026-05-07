"""Pytest configuration for plaquant kernel tests."""
import os
import pytest

# GEMM test data paths
CEPH_BASE = "/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_libra/user_spanaluo/plaquant"
GEMM_DATA_DIR = os.path.join(CEPH_BASE, "gemm_data")


@pytest.fixture
def gemm_data_dir():
    """Path to GEMM test data on ceph."""
    if not os.path.exists(GEMM_DATA_DIR):
        pytest.skip(f"GEMM data not found at {GEMM_DATA_DIR}")
    return GEMM_DATA_DIR


@pytest.fixture
def q_proj_data(gemm_data_dir):
    """Path to q_proj layer 0 test data."""
    path = os.path.join(gemm_data_dir, "model_layers_0_self_attn_q_proj")
    if not os.path.exists(path):
        pytest.skip(f"q_proj data not found at {path}")
    return path
