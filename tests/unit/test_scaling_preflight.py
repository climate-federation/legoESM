"""Parse-time scale-out preflight (#1361).

Each test encodes one of the three multi-node allocations burned on 2026-07-27
by a constraint that was already decidable from the arguments.
"""

from __future__ import annotations

import pytest

from legoesm.scaling_preflight import (
    CS_SPMD_DEVICE_COUNTS,
    DEVICE_HBM_BYTES,
    estimate_bytes_per_device,
    nearest_divisible,
    preflight_or_exit,
    tiled_device_counts,
    validate_device_count,
    validate_divisibility,
    validate_memory,
)


class TestDivisibility:
    def test_job_26497323_regression(self):
        """`--n-lat 720 --n-devices 64` died AFTER the 16-GPU arm ran."""
        with pytest.raises(ValueError, match="not divisible"):
            validate_divisibility(720, 64, axis="n_lat")

    def test_message_offers_nearest_valid_resolutions(self):
        with pytest.raises(ValueError) as exc:
            validate_divisibility(720, 64, axis="n_lat")
        msg = str(exc.value)
        # actionable: the user should not have to do the arithmetic
        assert "704" in msg and "768" in msg

    def test_exact_multiple_passes(self):
        validate_divisibility(768, 64, axis="n_lat")
        validate_divisibility(128, 8, axis="n_lat")

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="n_devices must be positive"):
            validate_divisibility(128, 0)
        with pytest.raises(ValueError, match="must be positive"):
            validate_divisibility(0, 8)

    @pytest.mark.parametrize("n,d", [(720, 64), (100, 7), (33, 4), (1, 8)])
    def test_nearest_divisible_are_actually_divisible(self, n, d):
        near = nearest_divisible(n, d)
        assert near, "must never be empty"
        assert all(v % d == 0 and v > 0 for v in near)


class TestDeviceCount:
    def test_job_26495955_regression(self):
        """cs-spmd at 96 GPUs silently replicated, then died on an 83 GB arg."""
        with pytest.raises(ValueError, match="cs-spmd face-shards only"):
            validate_device_count("cs-spmd", 96)

    @pytest.mark.parametrize("n", CS_SPMD_DEVICE_COUNTS)
    def test_cs_spmd_supported_counts_pass(self, n):
        validate_device_count("cs-spmd", n)

    @pytest.mark.parametrize("n", [6, 24, 54, 96])
    def test_tiled_6ktsq_passes(self, n):
        validate_device_count("tiled", n)

    @pytest.mark.parametrize("n", [8, 16, 64, 100])
    def test_tiled_rejects_non_6ktsq(self, n):
        with pytest.raises(ValueError, match=r"6\*kt\^2"):
            validate_device_count("tiled", n)

    def test_unknown_path_raises(self):
        """Dispatch hardening: an unknown path must not validate silently."""
        with pytest.raises(ValueError, match="unknown cube parallel path"):
            validate_device_count("bogus", 6)

    def test_tiled_device_counts_are_6ktsq(self):
        assert tiled_device_counts(4) == (6, 24, 54, 96)


class TestMemory:
    def test_job_26497294_regression(self):
        """C1152 L60 kt=3 -> 105.7 GB/device vs 80 GB HBM, found at compile.

        The cube lanes allocate GLOBAL-sized buffers per device (#1370), so the
        estimate must NOT divide by n_devices -- with sharded=True this config
        estimates 2.4 GB/device and is silently waved through, which is the bug
        this preflight exists to prevent.
        """
        with pytest.raises(ValueError, match="exceeds the 80 GB HBM"):
            validate_memory(n_columns=6 * 1152 * 1152, nlev=60,
                            n_devices=6 * 3 * 3, device="a100-80",
                            sharded=False)

    def test_calibration_brackets_the_observed_footprint(self):
        """Estimate must be >= the measured 105.7 GB (conservative) and within
        ~1.5x of it (not so loose it rejects everything real)."""
        est = estimate_bytes_per_device(
            n_columns=6 * 1152 * 1152, nlev=60, n_devices=54, sharded=False)
        observed = 105.7 * 1024**3
        assert est >= observed
        assert est <= 1.5 * observed

    def test_sharded_true_would_have_missed_it(self):
        """Documents WHY the sharded flag exists: the same config passes when
        the divide-by-n_devices assumption is (wrongly) applied."""
        est = validate_memory(n_columns=6 * 1152 * 1152, nlev=60,
                              n_devices=54, device="a100-80", sharded=True)
        assert est < DEVICE_HBM_BYTES["a100-80"]

    def test_message_carries_the_estimate(self):
        with pytest.raises(ValueError) as exc:
            validate_memory(n_columns=6 * 1152 * 1152, nlev=60,
                            n_devices=54, device="a100-80", sharded=False)
        assert "GB/device" in str(exc.value)

    def test_fitting_config_passes_and_returns_estimate(self):
        est = validate_memory(n_columns=128 * 256, nlev=30, n_devices=8,
                              device="a100-80")
        assert 0 < est < DEVICE_HBM_BYTES["a100-80"]

    def test_no_device_means_estimate_only(self):
        """We do not guess hardware: no device -> no gate, even if huge."""
        est = validate_memory(n_columns=6 * 1152 * 1152, nlev=60, n_devices=6)
        assert est > 0

    def test_more_devices_lowers_per_device_bytes(self):
        a = estimate_bytes_per_device(n_columns=10_000, nlev=30, n_devices=4)
        b = estimate_bytes_per_device(n_columns=10_000, nlev=30, n_devices=8)
        assert b == pytest.approx(a / 2)

    def test_fp32_halves_the_estimate(self):
        f64 = estimate_bytes_per_device(n_columns=10_000, nlev=30, n_devices=4)
        f32 = estimate_bytes_per_device(n_columns=10_000, nlev=30, n_devices=4,
                                        bytes_per_value=4)
        assert f32 == pytest.approx(f64 / 2)

    def test_unknown_device_raises(self):
        with pytest.raises(ValueError, match="unknown device"):
            validate_memory(n_columns=100, nlev=1, n_devices=1, device="tpu9")


class TestPreflightOrExit:
    def test_converts_valueerror_to_systemexit(self):
        with pytest.raises(SystemExit, match=r"\[preflight\] .*not divisible"):
            preflight_or_exit(validate_divisibility, 720, 64, axis="n_lat")

    def test_passes_through_return_value(self):
        est = preflight_or_exit(validate_memory, n_columns=100, nlev=2,
                                n_devices=1)
        assert est > 0


class TestValidateBandRowsGpu:
    """Thin-band NCCL-init deadlock guard (empirical floor, 2026-08-18)."""

    def test_refuses_twelve_row_bands(self):
        from legoesm.scaling_preflight import validate_band_rows_gpu
        with pytest.raises(ValueError, match="deadlock"):
            validate_band_rows_gpu(2304, 192)   # the receipted hang config

    def test_accepts_fifteen_row_bands(self):
        from legoesm.scaling_preflight import validate_band_rows_gpu
        validate_band_rows_gpu(2880, 192)       # the receipted clean config

    def test_escape_hatch(self, monkeypatch):
        from legoesm.scaling_preflight import validate_band_rows_gpu
        monkeypatch.setenv("LEGOESM_ALLOW_THIN_BANDS", "1")
        validate_band_rows_gpu(2304, 192)

    def test_floor_is_the_named_constant(self):
        # The guard must key off MIN_GPU_BAND_ROWS, not a re-hardcoded 15:
        # tightening the constant must tighten the guard.
        from legoesm import scaling_preflight as sp
        rows_at_floor = sp.MIN_GPU_BAND_ROWS
        sp.validate_band_rows_gpu(rows_at_floor * 8, 8)
        with pytest.raises(ValueError):
            sp.validate_band_rows_gpu((rows_at_floor - 1) * 8, 8)
