"""Unit tests for the EKE-runaway GM/Redi fixes (2026-06 tier-2 isolation).

Covers, with synthetic configurations where the expected behavior is
hand-derivable:

1. PER-TRIAD face masking (canonical fix): on stepped bathymetry the W-face
   triad fluxes, K_33 and the tracer tendency on WET cells are INDEPENDENT of
   the junk values stored in sub-seafloor cells (Veros's masked-gradient
   mechanism), and bit-identical to legacy on a full-depth (flat-bottom)
   partial-cell domain.
2. ``double_redi_diagonal`` option: OFF is the default (bit-identical path);
   ON adds EXACTLY one extra taper-weighted diagonal ``K·∂q/∂x`` to the
   horizontal fluxes (verified against a hand-computed uniform-κ case).
3. ``veros_triad_weights`` option: on a UNIFORM vertical grid the interior
   weights reduce to the legacy 1/4 (interior levels bit-class identical);
   the option redistributes only the boundary-level weights.
4. Constant-κ sampling self-test: a spatially UNIFORM 3-D interface κ field
   must reproduce the scalar-κ tendency exactly (κ uniform ⇒ vertical
   structure irrelevant ⇒ the 3-D dispatch is invisible).
5. Manufactured two-level κ(z): the horizontal u-face κ must be the Veros
   ``diffloc`` 0.25-average of the two W-levels straddling the cell over the
   two adjacent columns (one-sided at the top/bottom cells), and the W-face
   flux must consume the interface κ DIRECTLY.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    _kappa_center_uvw,
    compute_isoneutral_K33_latlon,
    gm_redi_tracer_tendency_triads_latlon_cgrid,
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
        # A shelf: half the domain only 3 levels deep (a topographic step).
        H[:, : N_LON // 2] = float(np.asarray(z.z_half_ref)[3] * -1.0)
    zp = create_partial_cell_coordinate(z, jnp.asarray(H))
    mask = jnp.ones((N_LAT, N_LON))
    u_mask = jnp.ones((N_LAT, N_LON + 1))
    v_mask = jnp.zeros((N_LAT + 1, N_LON)).at[1:-1, :].set(1.0)
    lat = np.degrees(np.asarray(grid.lat))
    zc = np.asarray(z.z_full_ref)
    # Smooth stratified T with weak horizontal structure (stable, gentle
    # slopes -> taper ~= 1 interior).
    T = (20.0 + 8.0 * (zc[None, None, :] / 3000.0)
         + 0.5 * np.sin(np.radians(lat))[:, None, None]
         + 0.2 * np.cos(2 * np.pi * np.arange(N_LON) / N_LON)[None, :, None])
    S = np.full(T.shape, 35.0)
    rho = 1026.0 - 0.2 * (T - 10.0)   # linear-EOS-like density
    jac = jnp.ones((N_LAT, N_LON))
    return grid, z, zp, mask, u_mask, v_mask, jnp.asarray(T), jnp.asarray(
        S), jnp.asarray(rho), jac


def _triads(q, rho, mask, u_mask, v_mask, zc, jac, grid, kgm, kredi,
            **kw):
    return gm_redi_tracer_tendency_triads_latlon_cgrid(
        q, rho, mask, u_mask, v_mask, zc, jac, grid, kgm, kredi,
        S_max=0.01, taper_width_frac=0.5, implicit_K33=True,
        K_iso_steep=0.0, return_fluxes=True, **kw)


class TestPerTriadFaceMasking:
    def test_wet_tendency_independent_of_subseafloor_junk(self):
        """Perturbing sub-seafloor cell values must not change the tendency,
        fluxes, or K33 on wet cells (Veros: masked gradients kill those
        triads; pre-fix the Neumann-filled gradients leaked them)."""
        (grid, z, zp, mask, u_mask, v_mask, T, S, rho, jac) = _setup(True)
        ia = np.asarray(zp.is_active)
        dry = ~ia
        T2 = np.asarray(T).copy()
        rho2 = np.asarray(rho).copy()
        T2[dry] += 7.5          # junk perturbation below the seafloor
        rho2[dry] += 3.0
        out1 = _triads(T, rho, mask, u_mask, v_mask, zp, jac, grid,
                       1000.0, 1000.0)
        out2 = _triads(jnp.asarray(T2), jnp.asarray(rho2), mask, u_mask,
                       v_mask, zp, jac, grid, 1000.0, 1000.0)
        tend1, fx1, fy1, fz1 = out1
        tend2, fx2, fy2, fz2 = out2
        act = ia.astype(float)
        # Tendency on ACTIVE cells must be unchanged.
        np.testing.assert_allclose(
            np.asarray(tend1) * act, np.asarray(tend2) * act,
            rtol=0, atol=1e-13)
        # W-face fluxes on wet interfaces likewise.
        wif = (ia[..., :-1] & ia[..., 1:]).astype(float)
        np.testing.assert_allclose(
            np.asarray(fz1) * wif, np.asarray(fz2) * wif,
            rtol=0, atol=1e-13)

    def test_flat_bottom_partial_cell_matches_pure_zstar(self):
        """Full-depth partial-cell coord (flat bottom): the per-triad masks
        are all ones, so the result is bit-identical to the pure z-star
        path (the legacy code)."""
        (grid, z, zp, mask, u_mask, v_mask, T, S, rho, jac) = _setup(False)
        assert bool(np.all(np.asarray(zp.is_active)))
        o_star = _triads(T, rho, mask, u_mask, v_mask, z, jac, grid,
                         1000.0, 1000.0)
        o_part = _triads(T, rho, mask, u_mask, v_mask, zp, jac, grid,
                         1000.0, 1000.0)
        for a, b in zip(o_star, o_part):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_k33_independent_of_subseafloor_junk(self):
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        (grid, z, zp, mask, u_mask, v_mask, T, S, rho, jac) = _setup(True)
        cfg = GMRediConfig(kappa_Redi=1000.0, S_max=0.01,
                           taper_width_frac=0.5, implicit_K33=True)
        ia = np.asarray(zp.is_active)
        T2 = np.asarray(T).copy()
        T2[~ia] -= 5.0
        eta = jnp.zeros((N_LAT, N_LON))
        H = jnp.sum(zp.h_partial, axis=-1)
        k1 = compute_isoneutral_K33_latlon(
            T, S, eta, H, grid, zp, cfg, eos="linear", mask=mask)
        k2 = compute_isoneutral_K33_latlon(
            jnp.asarray(T2), S, eta, H, grid, zp, cfg, eos="linear",
            mask=mask)
        wif = (ia[..., :-1] & ia[..., 1:]).astype(float)
        np.testing.assert_allclose(
            np.asarray(k1) * wif, np.asarray(k2) * wif, rtol=0, atol=1e-13)


class TestDoubleRediDiagonal:
    def test_off_is_default_and_none_is_noop(self):
        (grid, z, zp, mask, u_mask, v_mask, T, S, rho, jac) = _setup(False)
        o1 = _triads(T, rho, mask, u_mask, v_mask, z, jac, grid,
                     1000.0, 1000.0)
        o2 = _triads(T, rho, mask, u_mask, v_mask, z, jac, grid,
                     1000.0, 1000.0, double_diag_kappa=None)
        for a, b in zip(o1, o2):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_on_adds_exactly_one_extra_diagonal(self):
        """With uniform κ and gentle slopes (taper ≈ 1 interior), the option
        must add ≈ K·∂q/∂x to F_x (and K·∂q/∂y to F_y): F_on − F_off ==
        K·Σw·taper·∇q, hand-checkable at interior faces where all four
        triads are alive and untapered."""
        (grid, z, zp, mask, u_mask, v_mask, T, S, rho, jac) = _setup(False)
        K = 700.0
        _, fx0, fy0, fz0 = _triads(T, rho, mask, u_mask, v_mask, z, jac,
                                   grid, 1000.0, K)
        _, fx1, fy1, fz1 = _triads(T, rho, mask, u_mask, v_mask, z, jac,
                                   grid, 1000.0, K, double_diag_kappa=K)
        # F_z must be untouched (the option is horizontal-diagonal only).
        np.testing.assert_array_equal(np.asarray(fz0), np.asarray(fz1))
        # The extra is a pure diagonal: compare with K·dq/dx at interior
        # faces/levels (taper ~ 1 for this gentle field).
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            gradient_x_cgrid,
        )
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            neumann_fill_cgrid,
        )
        dqdx = np.asarray(gradient_x_cgrid(neumann_fill_cgrid(T, mask), grid))
        extra = np.asarray(fx1) - np.asarray(fx0)
        # Hand expectation: the dm95 taper at near-zero slope is
        # 0.5·(1+tanh(1/frac)) (NOT exactly 1), so the extra diagonal is
        # K·taper0·dq/dx to within the small slope variation of this field.
        from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
            dm95_taper_scalar,
        )
        t0 = float(dm95_taper_scalar(jnp.asarray(0.0), 0.01,
                                     transition_width_frac=0.5)[1])
        sl = (slice(2, -2), slice(2, -2), slice(1, NLEV - 1))
        np.testing.assert_allclose(extra[sl], (K * t0 * dqdx)[sl], rtol=0.05)

    def test_orchestrator_rejects_centered_scheme(self):
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            gm_redi_tracer_tendency_latlon,
        )
        (grid, z, zp, mask, u_mask, v_mask, T, S, rho, jac) = _setup(False)
        cfg = GMRediConfig(slope_scheme="centered",
                           double_redi_diagonal=True)
        eta = jnp.zeros((N_LAT, N_LON))
        H = jnp.full((N_LAT, N_LON), 3000.0)
        with pytest.raises(ValueError, match="triads"):
            gm_redi_tracer_tendency_latlon(
                T, S, eta, H, grid, z, cfg, eos="linear", mask=mask,
                u_mask=u_mask, v_mask=v_mask)


class TestVerosTriadWeights:
    def test_uniform_grid_interior_levels_match_legacy(self):
        """Uniform dz ⇒ dzw(pair)/(4·dzt) = 1/4 = legacy interior weight.
        Interior levels (away from the surface/bottom edge) must agree;
        only the boundary levels may differ (Veros's no-renormalization
        convention)."""
        from legoesm.ocean.vertical import OceanZStarCoordinate
        (grid, _z, zp, mask, u_mask, v_mask, T, S, rho, jac) = _setup(False)
        # A genuinely UNIFORM vertical grid (create_ocean_z_star stretches).
        dz = jnp.full((NLEV,), 500.0)
        zh = jnp.concatenate([jnp.zeros(1), -jnp.cumsum(dz)])
        zf = 0.5 * (zh[:-1] + zh[1:])
        z = OceanZStarCoordinate(
            n_levels=NLEV, H_max=float(NLEV * 500.0), z_full_ref=zf,
            z_half_ref=zh, dz_ref=dz,
            dz_half_ref=jnp.abs(zf[:-1] - zf[1:]))
        _, fx0, fy0, _ = _triads(T, rho, mask, u_mask, v_mask, z, jac, grid,
                                 1000.0, 800.0)
        _, fx1, fy1, _ = _triads(T, rho, mask, u_mask, v_mask, z, jac, grid,
                                 1000.0, 800.0, veros_triad_weights=True)
        sl = (slice(None), slice(None), slice(1, NLEV - 1))
        np.testing.assert_allclose(np.asarray(fx0)[sl], np.asarray(fx1)[sl],
                                   rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.asarray(fy0)[sl], np.asarray(fy1)[sl],
                                   rtol=0, atol=1e-14)
        # Boundary levels DO differ (renormalized vs dzw-weighted + edge
        # death) — guard against the option silently becoming a no-op.
        assert not np.allclose(np.asarray(fx0)[..., 0], np.asarray(fx1)[..., 0])


class TestConstantKappaSampling:
    def test_uniform_interface_kappa_equals_scalar(self):
        """κ uniform ⇒ vertical structure irrelevant: a constant-valued 3-D
        interface κ must reproduce the scalar-κ result exactly (the sharp
        κ-sampling self-test)."""
        (grid, z, zp, mask, u_mask, v_mask, T, S, rho, jac) = _setup(False)
        K = 1000.0
        k3 = jnp.full((N_LAT, N_LON, NLEV - 1), K)
        o_scalar = _triads(T, rho, mask, u_mask, v_mask, z, jac, grid,
                           K, K)
        o_field = _triads(T, rho, mask, u_mask, v_mask, z, jac, grid,
                          k3, k3)
        for a, b in zip(o_scalar, o_field):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                       rtol=0, atol=1e-12)


class TestKappaPlacementVerosDiffloc:
    def test_two_level_kappa_horizontal_sampling(self):
        """Manufactured two-distinct-level interface κ: the u-face κ must be
        Veros's diffloc — interior cell: 0.25·(κ_w_above + κ_w_below) over
        the two columns; top/bottom cell: the one-sided interface copy —
        hand-computed here from first principles."""
        kw = np.zeros((N_LAT, N_LON, NLEV - 1))
        kw[:, :, 0] = 400.0      # shallow interface value
        kw[:, :, 1:] = 1200.0    # deep value
        kc, ku, kv, kw_out = _kappa_center_uvw(jnp.asarray(kw), NLEV)
        np.testing.assert_array_equal(np.asarray(kw_out), kw)  # W direct
        kc = np.asarray(kc)
        # Hand-derived centres: top cell = iface0 (one-sided) = 400;
        # cell 1 = 0.5(400+1200) = 800; interior = 1200; bottom = 1200.
        assert np.allclose(kc[..., 0], 400.0)
        assert np.allclose(kc[..., 1], 800.0)
        assert np.allclose(kc[..., 2:], 1200.0)
        # u-face = horizontal 0.5-average of adjacent centres (uniform field
        # here, so equals the centre value = Veros's 0.25 four-point diffloc).
        ku = np.asarray(ku)
        assert np.allclose(ku[..., 1], 800.0)
        assert np.allclose(ku[..., 0], 400.0)
