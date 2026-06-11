"""Unit tests for the per-level (3-D) face masks in the flux-divergence
lateral viscosity + its K_diss_h EKE credit (the 2026-06 mask-bug fix).

The bug class (same as the nine global_4deg variable-bathymetry fixes): the
operator masked gradients with the 2-D face masks only, so at topographic
STEPS (face closed at depth, open above) it computed a flux across the
closed face — a spurious no-slip wall stress (momentum leak into rock) whose
positive-definite dissipation was credited to EKE (+37% on the global yr-3
state).  Veros multiplies every flux by the per-level maskU/maskV
(friction.py:390-403): free-slip at steps, zero flux, zero credit.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks_3d,
    flux_divergence_viscosity_cgrid,
)
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
)

N_LAT, N_LON, NLEV = 8, 12, 6


def _setup(stepped: bool):
    grid = create_latlon_grid(N_LAT, N_LON)
    z = create_ocean_z_star(n_levels=NLEV, H_max=3000.0)
    H = np.full((N_LAT, N_LON), 3000.0)
    if stepped:
        H[:, : N_LON // 2] = float(-np.asarray(z.z_half_ref)[3])
    zp = create_partial_cell_coordinate(z, jnp.asarray(H))
    mask = jnp.ones((N_LAT, N_LON))
    u_mask = jnp.ones((N_LAT, N_LON + 1))
    v_mask = jnp.zeros((N_LAT + 1, N_LON)).at[1:-1, :].set(1.0)
    rng = np.random.default_rng(7)
    u = jnp.asarray(rng.normal(0, 0.1, (N_LAT, N_LON + 1, NLEV)))
    v = jnp.asarray(rng.normal(0, 0.1, (N_LAT + 1, N_LON, NLEV)))
    return grid, zp, mask, u_mask, v_mask, u, v


class TestSteppedBathymetry:
    def test_no_flux_through_closed_faces_and_lower_kdiss(self):
        grid, zp, mask, u_mask, v_mask, u, v = _setup(True)
        um3, vm3 = compute_face_masks_3d(zp.is_active, grid)
        um3 = um3.astype(u.dtype)
        vm3 = vm3.astype(v.dtype)

        # 2-D masks (the bug): flux crosses faces closed at depth.
        _, _, kd2 = flux_divergence_viscosity_cgrid(
            u, v, grid, 1.0e4, mask=mask, u_mask=u_mask, v_mask=v_mask,
            want_dissipation=True)
        # 3-D masks (the fix): per-level free-slip.
        vu3, vv3, kd3 = flux_divergence_viscosity_cgrid(
            u * um3, v * vm3, grid, 1.0e4, mask=mask, u_mask=um3,
            v_mask=vm3, want_dissipation=True)

        # (a) the viscous tendency is zero on faces below the seafloor
        #     (no momentum injected into rock).
        np.testing.assert_array_equal(
            np.asarray(vu3) * (1.0 - np.asarray(um3)), 0.0)
        np.testing.assert_array_equal(
            np.asarray(vv3) * (1.0 - np.asarray(vm3)), 0.0)
        # (b) the spurious closed-face credit is strictly removed: the
        #     positive-definite K_diss_h integral DROPS with 3-D masks.
        assert float(jnp.sum(kd3)) < float(jnp.sum(kd2))

    def test_wet_tendency_independent_of_subseafloor_velocity(self):
        """With per-level masks (+ per-level-masked velocities), junk u/v
        below the seafloor cannot reach the wet-face tendencies or the
        dissipation; with 2-D masks it can (the bug)."""
        grid, zp, mask, u_mask, v_mask, u, v = _setup(True)
        um3, vm3 = compute_face_masks_3d(zp.is_active, grid)
        um3 = um3.astype(u.dtype)
        vm3 = vm3.astype(v.dtype)
        u2 = u + 5.0 * (1.0 - um3)   # junk only below the seafloor
        v2 = v + 5.0 * (1.0 - vm3)

        out_a = flux_divergence_viscosity_cgrid(
            u * um3, v * vm3, grid, 1.0e4, mask=mask, u_mask=um3,
            v_mask=vm3, want_dissipation=True)
        out_b = flux_divergence_viscosity_cgrid(
            u2 * um3, v2 * vm3, grid, 1.0e4, mask=mask, u_mask=um3,
            v_mask=vm3, want_dissipation=True)
        for a, b in zip(out_a, out_b):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

        # The 2-D-mask path IS sensitive to the junk — the synthetic
        # violation that proves this test is non-vacuous.
        _, _, kd_a = flux_divergence_viscosity_cgrid(
            u, v, grid, 1.0e4, mask=mask, u_mask=u_mask, v_mask=v_mask,
            want_dissipation=True)
        _, _, kd_b = flux_divergence_viscosity_cgrid(
            u2, v2, grid, 1.0e4, mask=mask, u_mask=u_mask, v_mask=v_mask,
            want_dissipation=True)
        assert not np.allclose(np.asarray(kd_a), np.asarray(kd_b))


class TestFlatBottomDegeneracy:
    def test_3d_masks_bit_identical_on_full_depth_columns(self):
        """Flat bottom: the 3-D face masks are all ones, so the fixed call
        (3-D masks + masked velocities) is BIT-IDENTICAL to the legacy 2-D
        call — the production-path safety gate."""
        grid, zp, mask, u_mask, v_mask, u, v = _setup(False)
        assert bool(np.all(np.asarray(zp.is_active)))
        um3, vm3 = compute_face_masks_3d(zp.is_active, grid)
        um3 = um3.astype(u.dtype)
        vm3 = vm3.astype(v.dtype)
        out2 = flux_divergence_viscosity_cgrid(
            u, v, grid, 1.0e4, cos_power=1, mask=mask, u_mask=u_mask,
            v_mask=v_mask, want_dissipation=True)
        out3 = flux_divergence_viscosity_cgrid(
            u * um3, v * vm3, grid, 1.0e4, cos_power=1, mask=mask,
            u_mask=um3, v_mask=vm3, want_dissipation=True)
        for a, b in zip(out2, out3):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
