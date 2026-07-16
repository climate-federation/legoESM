"""Unit tests for the rank-aware pole-v BC helper in
``primitive_eq_latlon_cgrid``.

The helper replaces a previously hardcoded
``jnp.pad(v[1:-1, :, :], ((1,1),(0,0),(0,0)))`` pattern that zeroed v
at both lat-axis ends unconditionally.  Under latitude-band MPI, only
the boundary ranks own the actual global poles; interior ranks must
NOT zero their band-edge v-rows (those are interior v-faces shared
with the neighbour rank).  The helper makes the zeroing per-end
opt-in, so the serial ``pole_v_bc=(True, True)`` configuration
preserves bit-exact behaviour and MPI ranks can pass
``(layout.south_rank is None, layout.north_rank is None)``.

These tests are deliberately tiny and pure — they do not import jax
config / sigma coords / model class — so they run in <1s on the
login-node CPU and serve as a fast guard against accidental
regression of the pole-BC contract.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationConfig,
    _zero_v_at_pole,
)


@pytest.fixture
def v_random_3d():
    """A (n_lat+1, n_lon, nlev) array with nonzero values at every row,
    so any zeroing is observable.
    """
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.standard_normal((9, 6, 4)))


class TestZeroVAtPole:
    """Direct unit tests on the helper."""

    def test_default_both_ends_zeroed(self, v_random_3d):
        """``(south=True, north=True)`` is the serial default."""
        out = _zero_v_at_pole(v_random_3d, south=True, north=True)
        # Both end rows are zero
        assert jnp.all(out[0] == 0)
        assert jnp.all(out[-1] == 0)
        # Interior is unchanged
        np.testing.assert_allclose(out[1:-1], v_random_3d[1:-1], rtol=0, atol=0)

    def test_default_matches_legacy_pad(self, v_random_3d):
        """Bit-exact equivalence to the pre-refactor
        ``jnp.pad(v[1:-1], ((1,1),(0,0),(0,0)))`` line.  Locks in
        serial behaviour.
        """
        legacy = jnp.pad(v_random_3d[1:-1, :, :], ((1, 1), (0, 0), (0, 0)))
        out = _zero_v_at_pole(v_random_3d, south=True, north=True)
        np.testing.assert_allclose(out, legacy, rtol=0, atol=0)

    def test_neither_pole_is_passthrough(self, v_random_3d):
        """Interior MPI ranks (south_rank and north_rank both not None)
        pass v through unchanged.
        """
        out = _zero_v_at_pole(v_random_3d, south=False, north=False)
        np.testing.assert_allclose(out, v_random_3d, rtol=0, atol=0)

    def test_south_only(self, v_random_3d):
        """Rank that owns the south pole but has a north neighbour:
        zero v[0], leave v[-1] alone.
        """
        out = _zero_v_at_pole(v_random_3d, south=True, north=False)
        assert jnp.all(out[0] == 0)
        np.testing.assert_allclose(out[1:], v_random_3d[1:], rtol=0, atol=0)

    def test_north_only(self, v_random_3d):
        """Rank that owns the north pole but has a south neighbour:
        leave v[0] alone, zero v[-1].
        """
        out = _zero_v_at_pole(v_random_3d, south=False, north=True)
        np.testing.assert_allclose(out[:-1], v_random_3d[:-1], rtol=0, atol=0)
        assert jnp.all(out[-1] == 0)

    def test_shape_preserved(self, v_random_3d):
        """All four (south, north) combinations preserve the input shape."""
        for south, north in [(True, True), (True, False),
                              (False, True), (False, False)]:
            out = _zero_v_at_pole(v_random_3d, south=south, north=north)
            assert out.shape == v_random_3d.shape, (
                f"(south={south}, north={north}) changed the shape from "
                f"{v_random_3d.shape} to {out.shape}"
            )


class TestOffset:
    """``offset>0`` zeros the actual pole rows when the array is
    padded by ``offset`` halo rows on each side (MPI use case)."""

    def test_offset_halo2_zeros_interior_indices(self, v_random_3d):
        """With halo=2 padding, the actual pole rows sit at index 2 and
        ``-3``; halo rows at 0, 1, -1, -2 must remain unchanged."""
        out = _zero_v_at_pole(v_random_3d, south=True, north=True, offset=2)
        # Halo rows untouched
        np.testing.assert_allclose(out[0], v_random_3d[0], rtol=0, atol=0)
        np.testing.assert_allclose(out[1], v_random_3d[1], rtol=0, atol=0)
        np.testing.assert_allclose(out[-2], v_random_3d[-2], rtol=0, atol=0)
        np.testing.assert_allclose(out[-1], v_random_3d[-1], rtol=0, atol=0)
        # Pole rows zeroed
        assert jnp.all(out[2] == 0)
        assert jnp.all(out[-3] == 0)
        # Interior unchanged
        np.testing.assert_allclose(out[3:-3], v_random_3d[3:-3], rtol=0, atol=0)

    def test_offset_halo1_zeros_interior_indices(self, v_random_3d):
        out = _zero_v_at_pole(v_random_3d, south=True, north=True, offset=1)
        np.testing.assert_allclose(out[0], v_random_3d[0], rtol=0, atol=0)
        np.testing.assert_allclose(out[-1], v_random_3d[-1], rtol=0, atol=0)
        assert jnp.all(out[1] == 0)
        assert jnp.all(out[-2] == 0)

    def test_offset_zero_reproduces_legacy(self, v_random_3d):
        """Belt-and-braces: offset=0 must still bit-match the legacy
        single-Pad expression (the kwarg default)."""
        legacy = jnp.pad(v_random_3d[1:-1, :, :], ((1, 1), (0, 0), (0, 0)))
        out = _zero_v_at_pole(
            v_random_3d, south=True, north=True, offset=0,
        )
        np.testing.assert_allclose(out, legacy, rtol=0, atol=0)


class TestConfigDefault:
    """The config's ``pole_v_bc`` default must remain ``(True, True)``
    so the existing AMIP / Held-Suarez / Williamson runs keep their
    serial behaviour."""

    def test_default_is_both_true(self):
        cfg = CGridLatLonPrimitiveEquationConfig()
        assert cfg.pole_v_bc == (True, True), (
            "Serial bit-exactness depends on the default being "
            "(True, True).  See _zero_v_at_pole's docstring for why "
            "changing this default would break the AMIP regression "
            "baselines."
        )

    def test_pole_v_bc_passes_through_replace(self):
        """``_replace`` of the config carries the override through —
        the MPI wrapper uses this pattern to inject rank-aware flags."""
        cfg = CGridLatLonPrimitiveEquationConfig()
        for flags in [(False, False), (True, False), (False, True)]:
            mpi_cfg = cfg._replace(pole_v_bc=flags)
            assert mpi_cfg.pole_v_bc == flags
            # All other fields are preserved
            assert mpi_cfg.fix_mass == cfg.fix_mass
            assert mpi_cfg.time_integrator == cfg.time_integrator
