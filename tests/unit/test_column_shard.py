"""Tests for ``legoesm.parallel.column_shard``.

Issue #273 Phase-3 follow-up: column sharding helpers for per-column
physics on arbitrary device counts (4-GPU unblock that does not
depend on cubed-sphere face-divisibility).

The tests exercise the helpers on a single test device (everything
degenerates to ``n_devices=1``) — the contract assertions (axis
divisibility math, mesh axis name, numerical correctness of the
per-column gray-radiation call) hold there.  Multi-device hardware
validation is downstream cluster work.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# ---------------------------------------------------------------------------
# pad_to_shardable
# ---------------------------------------------------------------------------


class TestPadToShardable:
    @pytest.mark.parametrize("ncol,n_dev,expected_padded,expected_pad", [
        (216, 4, 216, 0),
        (217, 4, 220, 3),
        (13824, 3, 13824, 0),
        (13824, 4, 13824, 0),
        (13825, 4, 13828, 3),
        (0, 4, 0, 0),
    ])
    def test_padding(self, ncol, n_dev, expected_padded, expected_pad):
        from legoesm.parallel.column_shard import pad_to_shardable
        padded, pad = pad_to_shardable(ncol, n_dev)
        assert padded == expected_padded
        assert pad == expected_pad
        assert padded % n_dev == 0
        assert pad < n_dev

    def test_invalid_n_devices(self):
        from legoesm.parallel.column_shard import pad_to_shardable
        with pytest.raises(ValueError, match="n_devices must be >= 1"):
            pad_to_shardable(10, 0)

    def test_invalid_ncol(self):
        from legoesm.parallel.column_shard import pad_to_shardable
        with pytest.raises(ValueError, match="ncol must be >= 0"):
            pad_to_shardable(-1, 4)


# ---------------------------------------------------------------------------
# create_column_mesh
# ---------------------------------------------------------------------------


class TestColumnMesh:
    def test_creates_single_axis_mesh(self):
        from legoesm.parallel.column_shard import create_column_mesh
        mesh = create_column_mesh(n_devices=1)
        assert mesh.axis_names == ("col",)
        assert mesh.shape["col"] >= 1

    def test_auto_uses_available_devices(self):
        from legoesm.parallel.column_shard import create_column_mesh
        mesh = create_column_mesh(n_devices="auto")
        assert mesh.shape["col"] == len(jax.devices())

    def test_invalid_n_devices(self):
        from legoesm.parallel.column_shard import create_column_mesh
        with pytest.raises(ValueError, match="n_devices must be >= 1"):
            create_column_mesh(n_devices=0)


# ---------------------------------------------------------------------------
# shard_columns / replicate
# ---------------------------------------------------------------------------


class TestShardColumns:
    def test_shard_preserves_values(self):
        from legoesm.parallel.column_shard import (
            create_column_mesh,
            shard_columns,
        )
        mesh = create_column_mesh(n_devices=1)
        x = jnp.asarray(np.arange(8 * 4, dtype=np.float64).reshape(8, 4))
        sharded = shard_columns(x, mesh)
        np.testing.assert_array_equal(np.asarray(sharded), np.asarray(x))

    def test_replicate_preserves_values(self):
        from legoesm.parallel.column_shard import (
            create_column_mesh,
            replicate,
        )
        mesh = create_column_mesh(n_devices=1)
        x = jnp.asarray(np.arange(8, dtype=np.float64))
        rep = replicate(x, mesh)
        np.testing.assert_array_equal(np.asarray(rep), np.asarray(x))

    def test_shard_scalar_falls_back_to_replicate(self):
        from legoesm.parallel.column_shard import (
            create_column_mesh,
            shard_columns,
        )
        mesh = create_column_mesh(n_devices=1)
        x = jnp.asarray(3.14)
        out = shard_columns(x, mesh)
        assert float(out) == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# End-to-end: gray_radiation runs unchanged on a column-sharded mesh
# ---------------------------------------------------------------------------


class TestGrayRadiationOnColumnMesh:
    """Issue #273 contract: ``gray_radiation`` produces the same output
    whether the column axis is sharded across an arbitrary device count
    or run on a single device.  Validates the unblock works end-to-end
    on the radiation hot path."""

    def _setup_columns(self, ncol: int, nlev: int):
        rng = np.random.default_rng(0)
        T = jnp.asarray(
            250.0 + 20.0 * rng.standard_normal((ncol, nlev)),
        )
        # Hydrostatic decreasing pressure profile per column.
        sigma_full = np.linspace(0.05, 0.95, nlev)
        sigma_half = np.concatenate(
            [[0.0], 0.5 * (sigma_full[:-1] + sigma_full[1:]), [1.0]],
        )
        p_s = jnp.asarray(
            1.0e5 + 1.0e3 * rng.standard_normal((ncol,)),
        )
        p_full = jnp.asarray(sigma_full[None, :] * np.asarray(p_s)[:, None])
        p_half = jnp.asarray(sigma_half[None, :] * np.asarray(p_s)[:, None])
        T_sfc = jnp.asarray(
            290.0 + 5.0 * rng.standard_normal((ncol,)),
        )
        lat = jnp.asarray(
            (rng.uniform(-1, 1, ncol) * (np.pi / 2)),
        )
        q_v = jnp.asarray(
            0.01 * rng.uniform(0.0, 1.0, (ncol, nlev)),
        )
        insolation = jnp.asarray(
            340.0 * rng.uniform(0.0, 1.0, (ncol,)),
        )
        return T, p_full, p_half, T_sfc, lat, q_v, insolation

    def test_sharded_matches_single_device(self):
        from legoesm.atmosphere.physics.radiation.config import (
            GrayRadiationConfig,
        )
        from legoesm.atmosphere.physics.radiation.gray import gray_radiation
        from legoesm.parallel.column_shard import (
            create_column_mesh,
            pad_to_shardable,
            replicate,
            shard_columns,
        )

        ncol, nlev = 216, 10
        T, p_full, p_half, T_sfc, lat, q_v, insolation = (
            self._setup_columns(ncol, nlev)
        )
        config = GrayRadiationConfig()

        # Baseline: run on single device, no sharding.
        ref = gray_radiation(
            T, p_full, p_half, T_sfc, lat, q_v, insolation, config,
        )

        # Sharded path.  On a single test device the mesh is just
        # ``n_dev=1`` but the helper API still routes through
        # ``device_put`` so the test catches any sharding-spec bugs
        # (wrong axis name, mis-shaped P, etc.).
        mesh = create_column_mesh(n_devices=1)
        n_dev = mesh.shape["col"]
        padded_ncol, pad = pad_to_shardable(ncol, n_dev)

        def _pad(x):
            if x.ndim == 1:
                return jnp.pad(x, (0, pad))
            return jnp.pad(x, ((0, pad),) + ((0, 0),) * (x.ndim - 1))

        T_s = shard_columns(_pad(T), mesh)
        p_full_s = shard_columns(_pad(p_full), mesh)
        p_half_s = shard_columns(_pad(p_half), mesh)
        T_sfc_s = shard_columns(_pad(T_sfc), mesh)
        lat_s = shard_columns(_pad(lat), mesh)
        q_v_s = shard_columns(_pad(q_v), mesh)
        ins_s = shard_columns(_pad(insolation), mesh)

        out = gray_radiation(
            T_s, p_full_s, p_half_s, T_sfc_s, lat_s, q_v_s, ins_s, config,
        )

        # Strip padding before comparison.
        def _crop(x):
            return np.asarray(x)[:ncol]

        np.testing.assert_allclose(
            _crop(out.heating_rate), np.asarray(ref.heating_rate),
            rtol=1.0e-12, atol=1.0e-14,
        )
        np.testing.assert_allclose(
            _crop(out.lw_flux_up), np.asarray(ref.lw_flux_up),
            rtol=1.0e-12, atol=1.0e-14,
        )
        np.testing.assert_allclose(
            _crop(out.sw_flux_down), np.asarray(ref.sw_flux_down),
            rtol=1.0e-12, atol=1.0e-14,
        )

    def test_multidevice_emulation_matches_single_device(self):
        """Issue #273: under CPU 4-device emulation (the closest
        local proxy for a 4×A100 node), column-sharded
        gray_radiation must match the single-device baseline
        bit-for-bit.  Run only when at least 2 devices are
        available (skipped on the default single-device test
        host).  To exercise locally:

            XLA_FLAGS="--xla_force_host_platform_device_count=4"
            JAX_PLATFORMS=cpu pytest tests/unit/test_column_shard.py
        """
        if len(jax.devices()) < 2:
            pytest.skip(
                "single-device host — set XLA_FLAGS to emulate 4 devices"
            )
        from legoesm.atmosphere.physics.radiation.config import (
            GrayRadiationConfig,
        )
        from legoesm.atmosphere.physics.radiation.gray import gray_radiation
        from legoesm.parallel.column_shard import (
            create_column_mesh,
            pad_to_shardable,
            shard_columns,
        )

        ncol, nlev = 217, 10  # not divisible by 4 — exercises padding
        T, p_full, p_half, T_sfc, lat, q_v, insolation = (
            self._setup_columns(ncol, nlev)
        )
        config = GrayRadiationConfig()

        ref = gray_radiation(
            T, p_full, p_half, T_sfc, lat, q_v, insolation, config,
        )

        mesh = create_column_mesh(n_devices=len(jax.devices()))
        n_dev = mesh.shape["col"]
        padded_ncol, pad = pad_to_shardable(ncol, n_dev)
        assert pad > 0, (
            f"this test wants pad > 0 to exercise padding logic; "
            f"ncol={ncol}, n_dev={n_dev} produced pad={pad}"
        )

        def _pad(x):
            if x.ndim == 1:
                return jnp.pad(x, (0, pad))
            return jnp.pad(x, ((0, pad),) + ((0, 0),) * (x.ndim - 1))

        T_s = shard_columns(_pad(T), mesh)
        p_full_s = shard_columns(_pad(p_full), mesh)
        p_half_s = shard_columns(_pad(p_half), mesh)
        T_sfc_s = shard_columns(_pad(T_sfc), mesh)
        lat_s = shard_columns(_pad(lat), mesh)
        q_v_s = shard_columns(_pad(q_v), mesh)
        ins_s = shard_columns(_pad(insolation), mesh)

        out = gray_radiation(
            T_s, p_full_s, p_half_s, T_sfc_s, lat_s, q_v_s, ins_s, config,
        )

        def _crop(x):
            return np.asarray(x)[:ncol]

        np.testing.assert_allclose(
            _crop(out.heating_rate), np.asarray(ref.heating_rate),
            rtol=1.0e-12, atol=1.0e-14,
        )
