"""Grid-agnostic ocean column physics (convective adjustment).

Convective adjustment is a per-column operation, so the SAME density
profile must yield the SAME interface diffusivity regardless of the
horizontal grid layout (cubed-sphere, lat-lon / tripole, MPAS).  These
tests pin that property and the static-instability response.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.column import (
    convective_adjustment_K,
    extract_cell_center_velocity,
)
from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
from legoesm.ocean.vertical import create_ocean_z_star

NLEV = 5
# Surface->bottom density [kg/m^3].  rho DROPS from level 1 to level 2
# (denser water sitting over lighter) => that interface is statically
# UNSTABLE (N^2 < 0); every other interface is stable.
RHO_COL = jnp.asarray([1027.0, 1028.0, 1026.5, 1029.0, 1030.0])
UNSTABLE_IFACE = 1  # interface between level 1 and level 2


@pytest.fixture
def zcoord():
    return create_ocean_z_star(n_levels=NLEV, H_max=4000.0)


def _broadcast(col, horiz_shape):
    """Broadcast a length-nlev column to (horiz_shape..., nlev)."""
    return jnp.broadcast_to(col, tuple(horiz_shape) + (NLEV,))


GRID_SHAPES = {
    "cubed_sphere": (6, 4, 4),
    "latlon": (8, 16),
    "mpas": (50,),
    "single_column": (1,),
}


class TestConvectiveAdjustmentGridAgnostic:
    def test_same_column_same_K_across_grids(self, zcoord):
        """The per-column K is identical on every grid shape."""
        ref = None
        for name, hshape in GRID_SHAPES.items():
            rho = _broadcast(RHO_COL, hshape)
            T = jnp.zeros_like(rho)
            S = jnp.zeros_like(rho)
            jac = jnp.ones(hshape)
            K = convective_adjustment_K(T, S, rho, zcoord, jac)
            assert K.shape == tuple(hshape) + (NLEV - 1,)
            # Collapse the horizontal dims — every column is identical.
            K_col = np.asarray(K).reshape(-1, NLEV - 1)[0]
            if ref is None:
                ref = K_col
            else:
                np.testing.assert_allclose(
                    K_col, ref, rtol=1e-12, atol=0,
                    err_msg=f"grid {name} column K differs from reference")

    def test_unstable_interface_gets_convective_K(self, zcoord):
        cfg = EnhancedDiffusionConfig()
        rho = RHO_COL[None, :]               # (1, nlev)
        K = convective_adjustment_K(
            jnp.zeros_like(rho), jnp.zeros_like(rho), rho, zcoord,
            jnp.ones((1,)), cfg)[0]
        # Unstable interface → ~K_conv; stable interfaces → ~K_bg.
        assert float(K[UNSTABLE_IFACE]) > 0.5 * cfg.K_conv, (
            f"unstable interface K={float(K[UNSTABLE_IFACE]):.3g} not enhanced")
        for k in range(NLEV - 1):
            if k != UNSTABLE_IFACE:
                assert float(K[k]) < 1e-3, (
                    f"stable interface {k} K={float(K[k]):.3g} too large")

    def test_stable_column_stays_background(self, zcoord):
        """A monotonically densifying column → no convection anywhere."""
        cfg = EnhancedDiffusionConfig()
        rho = jnp.asarray([1025.0, 1026.0, 1027.0, 1028.0, 1029.0])[None, :]
        K = convective_adjustment_K(
            jnp.zeros_like(rho), jnp.zeros_like(rho), rho, zcoord,
            jnp.ones((1,)), cfg)[0]
        assert float(jnp.max(K)) < 1e-3, "stable column should not convect"


class TestExtractCellCenterVelocity:
    def test_cgrid_averages_faces(self):
        n_lat, n_lon, nlev = 4, 6, 3
        u = jnp.asarray(
            np.random.default_rng(0).standard_normal((n_lat, n_lon + 1, nlev)))
        v = jnp.asarray(
            np.random.default_rng(1).standard_normal((n_lat + 1, n_lon, nlev)))
        u_c, v_c = extract_cell_center_velocity(u, v, "tripole")
        assert u_c.shape == (n_lat, n_lon, nlev)
        assert v_c.shape == (n_lat, n_lon, nlev)
        np.testing.assert_allclose(u_c, 0.5 * (u[:, :-1, :] + u[:, 1:, :]))
        np.testing.assert_allclose(v_c, 0.5 * (v[:-1, :, :] + v[1:, :, :]))

    def test_agrid_identity(self):
        u = jnp.ones((6, 4, 4, 2))
        v = jnp.ones((6, 4, 4, 2))
        u_c, v_c = extract_cell_center_velocity(u, v, "cubed_sphere")
        assert u_c is u and v_c is v

    def test_mpas_requires_mesh(self):
        with pytest.raises(NotImplementedError, match="MPAS"):
            extract_cell_center_velocity(
                jnp.ones((10, 3)), jnp.ones((10, 3)), "mpas")

    def test_unknown_grid_raises(self):
        with pytest.raises(ValueError, match="unknown grid_type"):
            extract_cell_center_velocity(jnp.ones((4, 4)), jnp.ones((4, 4)), "xyz")
