"""Serial unit tests for the fused multi-field lat-lon halo exchange.

Pins the value contract of the O4 scaling lever: the batched pads
(``pad_with_pole_bc_lat_multi`` / ``pad_ns_zero_multi`` /
``interp_to_v_points_multi`` / ``pad_with_pole_bc_lat_multi_mpi``) are
bit-identical to the per-field single pads on the local backend and on
a single-rank band layout.  The MPI fused-message schedule itself is
exercised by ``tests/distributed/test_fused_halo_mpi.py`` (mpirun) and
the bench ``--parity-gate``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.halo_latlon import (
    pad_with_pole_bc_lat,
    pad_with_pole_bc_lat_multi,
)
from legoesm.grids.operators_latlon_cgrid import pad_ns_zero, pad_ns_zero_multi
from legoesm.parallel.latlon_mpi import (
    LatLonBandLayout,
    pad_with_pole_bc_lat_mpi,
    pad_with_pole_bc_lat_multi_mpi,
)


def _fields(key=0):
    rng = np.random.default_rng(key)
    f2d = jnp.asarray(rng.standard_normal((8, 16)))
    f3d = jnp.asarray(rng.standard_normal((8, 16, 5)))
    g3d = jnp.asarray(rng.standard_normal((8, 17, 5)))  # u-face n_lon+1
    return f2d, f3d, g3d


class TestLocalBackendBitIdentical:
    def test_multi_matches_singles_zero_bc(self):
        fields = _fields()
        fused = pad_with_pole_bc_lat_multi(fields, halo=1)
        for f, fz in zip(fields, fused):
            single = pad_with_pole_bc_lat(f, halo=1)
            np.testing.assert_array_equal(np.asarray(fz), np.asarray(single))

    def test_multi_matches_singles_mixed_constants_halo2(self):
        fields = _fields(1)
        sv, nv = (1.0e30, 0.0, -2.5), (1.0e30, 0.0, 7.0)
        fused = pad_with_pole_bc_lat_multi(
            fields, halo=2, south_values=sv, north_values=nv,
        )
        for f, fz, s, n in zip(fields, fused, sv, nv):
            single = pad_with_pole_bc_lat(
                f, halo=2, south_value=s, north_value=n,
            )
            np.testing.assert_array_equal(np.asarray(fz), np.asarray(single))

    def test_pad_ns_zero_multi_matches_singles(self):
        fields = _fields(2)
        fused = pad_ns_zero_multi(*fields)
        for f, fz in zip(fields, fused):
            np.testing.assert_array_equal(
                np.asarray(fz), np.asarray(pad_ns_zero(f)),
            )

    def test_mixed_dtypes_preserved(self):
        f64 = jnp.asarray(np.random.default_rng(3).standard_normal((6, 4)))
        f32 = f64.astype(jnp.float32)
        a, b = pad_with_pole_bc_lat_multi((f64, f32), halo=1)
        assert a.dtype == jnp.float64
        assert b.dtype == jnp.float32
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(pad_with_pole_bc_lat(f64, halo=1)),
        )
        np.testing.assert_array_equal(
            np.asarray(b), np.asarray(pad_with_pole_bc_lat(f32, halo=1)),
        )

    def test_env_gate_off_matches(self, monkeypatch):
        monkeypatch.setenv("LEGOESM_LATLON_FUSED_HALO", "0")
        fields = _fields(4)
        fused = pad_with_pole_bc_lat_multi(fields, halo=1)
        for f, fz in zip(fields, fused):
            np.testing.assert_array_equal(
                np.asarray(fz),
                np.asarray(pad_with_pole_bc_lat(f, halo=1)),
            )

    def test_empty_and_validation(self):
        assert pad_with_pole_bc_lat_multi(()) == ()
        with pytest.raises(ValueError, match="south_values"):
            pad_with_pole_bc_lat_multi(
                _fields(), halo=1, south_values=(0.0,),
            )

    def test_interp_to_v_points_multi_matches_single(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            interp_to_v_points,
            interp_to_v_points_multi,
        )
        grid = create_latlon_grid(n_lat=8, n_lon=16)
        rng = np.random.default_rng(5)
        a = jnp.asarray(rng.standard_normal((8, 16, 5)))
        b = jnp.asarray(rng.standard_normal((8, 16, 3)))
        fa, fb = interp_to_v_points_multi((a, b), grid)
        np.testing.assert_array_equal(
            np.asarray(fa), np.asarray(interp_to_v_points(a, grid)),
        )
        np.testing.assert_array_equal(
            np.asarray(fb), np.asarray(interp_to_v_points(b, grid)),
        )


class TestSingleRankBandLayout:
    """np=1 layout (both neighbours None): multi_mpi == per-field mpi."""

    def _layout(self, n_lat=8, n_lon=16):
        return LatLonBandLayout(
            rank=0, n_ranks=1,
            n_lat_global=n_lat, n_lon_global=n_lon,
            n_lat_local=n_lat, lat_start=0, lat_end=n_lat,
            south_rank=None, north_rank=None,
        )

    def test_multi_mpi_matches_singles(self):
        layout = self._layout()
        fields = _fields(6)
        sv, nv = (3.0, 0.0, -1.0), (0.5, 2.0, 0.0)
        fused = pad_with_pole_bc_lat_multi_mpi(
            fields, layout, halo=1, south_values=sv, north_values=nv,
        )
        for f, fz, s, n in zip(fields, fused, sv, nv):
            single = pad_with_pole_bc_lat_mpi(
                f, layout, halo=1, south_value=s, north_value=n,
            )
            np.testing.assert_array_equal(np.asarray(fz), np.asarray(single))

    def test_multi_mpi_validation(self):
        layout = self._layout()
        f_a = jnp.zeros((8, 16))
        f_b = jnp.zeros((6, 16))  # mismatched n_lat
        with pytest.raises(ValueError, match="n_lat_local"):
            pad_with_pole_bc_lat_multi_mpi((f_a, f_b), layout, halo=1)
        with pytest.raises(ValueError, match="halo=9"):
            pad_with_pole_bc_lat_multi_mpi((f_a,), layout, halo=9)

    def test_grad_flows_through_multi(self):
        layout = self._layout()

        def loss(x, y):
            a, b = pad_with_pole_bc_lat_multi_mpi((x, y), layout, halo=1)
            return jnp.sum(a**2) + jnp.sum(b**3)

        x = jnp.ones((8, 16))
        y = jnp.full((8, 16, 2), 2.0)
        gx, gy = jax.grad(loss, argnums=(0, 1))(x, y)
        np.testing.assert_allclose(np.asarray(gx), 2.0)
        np.testing.assert_allclose(np.asarray(gy), 3.0 * 4.0)
