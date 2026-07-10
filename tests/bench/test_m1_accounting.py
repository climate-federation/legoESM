"""M1 increment-2 accounting helpers (audit items 4/8/9): pure arithmetic.

Direct unit tests for ``scripts/bench/metadata.py``'s
``calibrated_bound`` / ``comm_accounting`` / ``wet_cell_metrics`` — the
honest-null contract is the point: a missing ingredient must surface as
``None`` + a named reason, NEVER a fabricated number.  CPU/pure-python
(the helpers import no JAX).  The end-to-end JSONL wiring of these fields
is asserted by ``test_bench_ocean_latlon_spmd_gates.py``'s smoke run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))
from metadata import (  # noqa: E402
    DEFAULT_COMM_BANDWIDTH_GBS,
    DEFAULT_COMM_LATENCY_US,
    calibrated_bound,
    comm_accounting,
    wet_cell_metrics,
)


# --------------------------------------------------------------------------
# calibrated_bound (audit item 8)
# --------------------------------------------------------------------------

def test_bound_complete_arithmetic():
    """Hand-checked: T_bound = max(compute, comm) + red + launch + imbalance."""
    out = calibrated_bound(
        measured_fused_step_ms=16.0,
        single_device_fused_step_ms=10.0,
        halo_messages_per_step=100,
        halo_bytes_per_step=1_000_000,
        n_reductions_per_step=50,
        rank_imbalance=1.2,
        latency_us=10.0,
        bandwidth_GBs=10.0,
        launch_host_ms=0.3,
    )
    # comm = 100*10us + 1e6 B / 10 GB/s = 1.0 + 0.1 = 1.1 ms (< compute)
    assert out["bound_ingredients"]["comm_ms"] == pytest.approx(1.1)
    # reduction = 50 * 10us = 0.5 ms; imbalance = 0.2 * 10 = 2.0 ms
    assert out["bound_ingredients"]["reduction_ms"] == pytest.approx(0.5)
    assert out["bound_ingredients"]["imbalance_ms"] == pytest.approx(2.0)
    # T_bound = max(10, 1.1) + 0.5 + 0.3 + 2.0 = 12.8; measured/bound = 1.25
    assert out["t_bound_ms"] == pytest.approx(12.8)
    assert out["measured_over_bound"] == pytest.approx(16.0 / 12.8)
    assert out["bound_calibrated"] is True
    assert out["bound_incomplete_reason"] is None


def test_bound_comm_dominates_via_max():
    """When comm > compute the overlappable max picks comm, not the sum."""
    out = calibrated_bound(
        measured_fused_step_ms=None,
        single_device_fused_step_ms=1.0,
        halo_messages_per_step=1000,
        halo_bytes_per_step=0,
        n_reductions_per_step=0,
        rank_imbalance=1.0,
        latency_us=10.0,          # comm = 10 ms > compute = 1 ms
        bandwidth_GBs=10.0,
    )
    assert out["t_bound_ms"] == pytest.approx(10.0)
    assert out["measured_over_bound"] is None   # no measurement passed


def test_bound_missing_single_device_is_null_and_named():
    out = calibrated_bound(
        measured_fused_step_ms=8.0,
        single_device_fused_step_ms=None,       # the nd=1 row not provided
        halo_messages_per_step=10,
        halo_bytes_per_step=100,
        n_reductions_per_step=5,
        rank_imbalance=1.1,
    )
    assert out["t_bound_ms"] is None
    assert out["measured_over_bound"] is None
    assert "single_device_fused_step_ms" in out["bound_incomplete_reason"]
    # defaults used -> NOT calibrated
    assert out["bound_calibrated"] is False
    assert out["bound_ingredients"]["latency_us"] == DEFAULT_COMM_LATENCY_US
    assert (out["bound_ingredients"]["bandwidth_GBs"]
            == DEFAULT_COMM_BANDWIDTH_GBS)


def test_bound_missing_comm_census_is_null_and_named():
    out = calibrated_bound(
        measured_fused_step_ms=8.0,
        single_device_fused_step_ms=4.0,
        halo_messages_per_step=None,            # no census for this config
        halo_bytes_per_step=None,
        n_reductions_per_step=None,
        rank_imbalance=1.05,
    )
    assert out["t_bound_ms"] is None
    for name in ("halo_messages_per_step", "halo_bytes_per_step",
                 "n_reductions_per_step"):
        assert name in out["bound_incomplete_reason"]


def test_bound_single_device_trivial():
    """nd=1: zero comm is a fact -> bound == compute, ratio == 1."""
    out = calibrated_bound(
        measured_fused_step_ms=5.0,
        single_device_fused_step_ms=5.0,
        halo_messages_per_step=0,
        halo_bytes_per_step=0,
        n_reductions_per_step=0,
        rank_imbalance=1.0,
    )
    assert out["t_bound_ms"] == pytest.approx(5.0)
    assert out["measured_over_bound"] == pytest.approx(1.0)
    assert out["bound_incomplete_reason"] is None


def test_bound_partial_calibration_not_calibrated():
    """Passing only ONE of latency/bandwidth still uses a placeholder."""
    out = calibrated_bound(
        single_device_fused_step_ms=1.0,
        halo_messages_per_step=0, halo_bytes_per_step=0,
        n_reductions_per_step=0, rank_imbalance=1.0,
        latency_us=3.0,           # bandwidth left to the placeholder
    )
    assert out["bound_calibrated"] is False


def test_bound_rejects_bad_fabric_numbers():
    with pytest.raises(ValueError):
        calibrated_bound(bandwidth_GBs=0.0, latency_us=1.0)
    with pytest.raises(ValueError):
        calibrated_bound(bandwidth_GBs=1.0, latency_us=-1.0)


# --------------------------------------------------------------------------
# comm_accounting (audit item 4)
# --------------------------------------------------------------------------

def test_comm_bytes_slab_arithmetic():
    """bytes = messages x (n_lon x nlev x dtype x rows)."""
    out = comm_accounting(
        halo_messages_per_step=380, n_lon=32, nlev=1, dtype_bytes=8,
        rows_per_message=2,
    )
    assert out["halo_bytes_per_message"] == 32 * 1 * 8 * 2
    assert out["halo_bytes_per_step"] == 380 * 512
    assert out["full_state_gathers_per_step"] == 0


def test_comm_explicit_bytes_per_message_wins():
    out = comm_accounting(
        halo_messages_per_step=10, bytes_per_message=1000,
        full_state_gathers_per_step=2,     # run_omip-style host-loop driver
    )
    assert out["halo_bytes_per_step"] == 10_000
    assert out["full_state_gathers_per_step"] == 2


def test_comm_3d_slab_levels():
    out = comm_accounting(
        halo_messages_per_step=4, n_lon=100, nlev=20, dtype_bytes=4,
    )
    assert out["halo_bytes_per_message"] == 100 * 20 * 4
    assert out["halo_bytes_per_step"] == 4 * 8000


def test_comm_unknown_census_is_all_null_never_fabricated():
    out = comm_accounting(halo_messages_per_step=None,
                          scope_note="no census for this config")
    assert out["halo_messages_per_step"] is None
    assert out["halo_bytes_per_message"] is None
    assert out["halo_bytes_per_step"] is None
    assert out["comm_scope_note"] == "no census for this config"


def test_comm_zero_messages_zero_bytes():
    out = comm_accounting(halo_messages_per_step=0, bytes_per_message=0)
    assert out["halo_messages_per_step"] == 0
    assert out["halo_bytes_per_step"] == 0


def test_comm_message_count_without_size_raises():
    with pytest.raises(ValueError, match="payload size"):
        comm_accounting(halo_messages_per_step=5)


# --------------------------------------------------------------------------
# wet_cell_metrics (audit item 9)
# --------------------------------------------------------------------------

def test_wet_metric_on_masked_toy():
    """Toy 2-band mask (z-star column semantics): counts from the mask."""
    # band 0: 4 wet of 6 columns; band 1: 3 wet of 6.
    mask = np.array([[1, 1, 0, 1], [1, 0, 0, 0],       # device 0 (rows 0-1)
                     [1, 1, 0, 0], [0, 0, 1, 0]],      # device 1 (rows 2-3)
                    dtype=float)
    nlev, nd = 3, 2
    per_dev = mask.reshape(nd, 2, 4).sum(axis=(1, 2))  # the bench reduction
    out = wet_cell_metrics(
        wet_columns=float(mask.sum()), nlev=nlev,
        total_cells=mask.size * nlev, n_devices=nd,
        wet_columns_per_device=[float(w) for w in per_dev],
    )
    assert out["wet_cell_levels"] == 7 * 3
    assert out["wet_cell_levels_per_device"] == pytest.approx(10.5)
    assert out["wet_fraction"] == pytest.approx(21.0 / 48.0)
    assert out["wet_equals_total"] is False
    assert out["wet_cell_levels_per_device_min"] == 3 * 3
    assert out["wet_cell_levels_per_device_max"] == 4 * 3


def test_wet_metric_all_wet_flags_non_informative():
    out = wet_cell_metrics(wet_columns=12, nlev=3, total_cells=36,
                           n_devices=4)
    assert out["wet_cell_levels"] == 36
    assert out["wet_equals_total"] is True


def test_wet_metric_input_validation():
    with pytest.raises(ValueError):
        wet_cell_metrics(wet_columns=1, nlev=0, total_cells=1, n_devices=1)
    with pytest.raises(ValueError):
        wet_cell_metrics(wet_columns=1, nlev=1, total_cells=1, n_devices=0)
    with pytest.raises(ValueError, match="entries"):
        wet_cell_metrics(wet_columns=4, nlev=2, total_cells=8, n_devices=2,
                         wet_columns_per_device=[4.0])
