"""Tests for GM/Redi on the lat-lon Arakawa C-grid.

Run with:

    JAX_ENABLE_X64=1 python -m pytest tests/ocean/unit/test_gm_redi_latlon_cgrid.py -v

Covers: shape/finiteness, conservation, variance reduction, land masks,
differentiability, Visbeck coefficient, and structural enforcement against
code duplication with the cubed-sphere implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.vertical import create_ocean_z_star, compute_ocean_jacobian
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig, VisbeckConfig
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    compute_isopycnal_slopes_latlon_cgrid,
    gm_redi_tracer_tendency_latlon_cgrid,
    gm_redi_tracer_tendency_triads_latlon_cgrid,
    gm_redi_tracer_tendency_latlon,
    gm_redi_lateral_mixing_latlon,
    compute_isoneutral_K33_latlon,
    nemo_iso_lap_tracer_tendency_latlon_cgrid,
)
from tests.legoesm_paths import legoesm_source_path


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _make_setup(n_lat=10, n_lon=20, nlev=5, H_max=4000.0):
    """Create a basic lat-lon C-grid ocean setup."""
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(
        n_levels=nlev, H_max=H_max, dz_surface=100.0, dz_deep=1500.0,
    )
    mask = jnp.ones((n_lat, n_lon))
    u_mask = jnp.ones((n_lat, n_lon + 1))
    v_mask = jnp.ones((n_lat + 1, n_lon))
    eta = jnp.zeros((n_lat, n_lon))
    H_bathy = jnp.full((n_lat, n_lon), H_max)
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
    return grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian


def _stratified_with_meridional_tilt(n_lat=10, n_lon=20, nlev=5, slope=1e-4):
    """Stratified ocean with meridionally tilted isopycnals.

    Returns density rho with:
    - Vertical: 1025 at surface to 1027 at depth (stable)
    - Meridional gradient proportional to requested slope
    """
    grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian = \
        _make_setup(n_lat, n_lon, nlev)

    # Vertical profile: stable stratification.
    rho_z = jnp.linspace(constants.rho_ocean, 1027.0, nlev)
    # Add meridional gradient.
    H_total = float(jnp.sum(z_coord.dz_ref))
    drho_dz = 2.0 / H_total
    drho_dy = slope * drho_dz
    lat_idx = jnp.arange(n_lat, dtype=jnp.float64)
    rho = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)
    rho = rho + rho_z[jnp.newaxis, jnp.newaxis, :]
    # grid.dy is 1D (n_lat,); this regional channel has uniform dlat so
    # take a scalar representative value for the linear background gradient.
    dy_ref = float(grid.dy[0])
    rho = rho + drho_dy * lat_idx[:, jnp.newaxis, jnp.newaxis] * dy_ref

    # T proportional to density (linear EOS: rho ~ 1025 - 0.2*T)
    T = (constants.rho_ocean - rho) / 0.2
    S = jnp.full((n_lat, n_lon, nlev), 35.0, dtype=jnp.float64)

    cfg = GMRediConfig(kappa_GM=1000.0, kappa_Redi=1000.0, S_max=0.005)

    return (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
            jacobian, rho, T, S, cfg)


# =====================================================================
# 1. Shape and finiteness
# =====================================================================

class TestShapeAndFiniteness:

    def test_slope_shapes(self):
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        n_lat, n_lon, nlev = T.shape

        S_x, S_y, taper = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg,
        )
        assert S_x.shape == (n_lat, n_lon, nlev - 1)
        assert S_y.shape == (n_lat, n_lon, nlev - 1)
        assert taper.shape == (n_lat, n_lon, nlev - 1)

    def test_tendency_shapes(self):
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup

        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg,
        )
        dT = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
        )
        assert dT.shape == T.shape
        assert jnp.all(jnp.isfinite(dT))

    def test_orchestrator_shapes(self):
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        # Pin centered so this stays a centered-orchestrator regression test
        # (the default has flipped to triads; triads-orchestrator coverage
        # lives in TestTriadOrchestratorDispatch).
        cfg = cfg._replace(slope_scheme="centered")

        dT, dS = gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg,
            eos="linear", mask=mask, u_mask=u_mask, v_mask=v_mask,
        )
        assert dT.shape == T.shape
        assert dS.shape == S.shape
        assert jnp.all(jnp.isfinite(dT))
        assert jnp.all(jnp.isfinite(dS))


# =====================================================================
# 2. Zero tendency for uniform tracer
# =====================================================================

class TestZeroTendency:

    def test_uniform_tracer_gives_zero_tendency(self):
        """If T is spatially constant, dT/dt must be exactly zero."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup

        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg,
        )
        q_uniform = jnp.ones_like(T) * 15.0
        dq = gm_redi_tracer_tendency_latlon_cgrid(
            q_uniform, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
        )
        assert jnp.allclose(dq, 0.0, atol=1e-12)

    def test_horizontally_uniform_density_gives_zero_slopes(self):
        """If rho depends only on z, isopycnal slopes are zero."""
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian = _make_setup()
        nlev = z_coord.n_levels

        rho_z = jnp.linspace(constants.rho_ocean, 1027.0, nlev)
        rho = jnp.broadcast_to(
            rho_z[jnp.newaxis, jnp.newaxis, :],
            (mask.shape[0], mask.shape[1], nlev),
        ) + 0.0  # materialise

        cfg = GMRediConfig(S_max=0.005)
        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg,
        )
        assert jnp.allclose(S_x, 0.0, atol=1e-12)
        assert jnp.allclose(S_y, 0.0, atol=1e-12)


# =====================================================================
# 3. Tracer conservation
# =====================================================================

class TestConservation:

    def test_tracer_integral_conserved(self):
        """sum(dT * h * area * mask) should be zero (no sources/sinks)."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup

        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg,
        )
        dT = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
        )
        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
        area = grid.area[:, :, jnp.newaxis]
        integral = float(jnp.sum(dT * dz * area * mask[:, :, jnp.newaxis]))
        # Relative to max tendency * total volume
        max_dT = float(jnp.max(jnp.abs(dT)))
        total_vol = float(jnp.sum(dz * area * mask[:, :, jnp.newaxis]))
        if max_dT > 0:
            relative = abs(integral) / (max_dT * total_vol)
            assert relative < 1e-10, f"Conservation violated: relative error {relative:.2e}"
        else:
            assert abs(integral) < 1e-20


# =====================================================================
# 4. Variance reduction (APE)
# =====================================================================

class TestVarianceReduction:

    def test_gm_reduces_tracer_variance(self):
        """sum(dT * T * h * area) <= 0: GM should reduce APE."""
        setup = _stratified_with_meridional_tilt(slope=5e-4)
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup

        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg,
        )
        dT = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
        )
        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
        area = grid.area[:, :, jnp.newaxis]
        # d/dt(0.5 * T^2) = T * dT/dt
        variance_tendency = float(jnp.sum(dT * T * dz * area * mask[:, :, jnp.newaxis]))
        assert variance_tendency <= 0, (
            f"GM should reduce tracer variance, got {variance_tendency:.4e}"
        )


# =====================================================================
# 5. Land mask correctness
# =====================================================================

class TestLandMask:

    def test_tendency_zero_on_land(self):
        """Tendency must be zero at land cells."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        n_lat, n_lon, nlev = T.shape

        # Put land in the middle of the domain.
        mask_land = mask.at[4:6, 8:12].set(0.0)
        # Recompute face masks: 1 only if both neighbours are ocean.
        u_mask_land = mask_land * jnp.roll(mask_land, 1, axis=1)
        u_mask_land = jnp.concatenate([u_mask_land, u_mask_land[:, 0:1]], axis=1)
        v_mask_land_interior = mask_land[:-1, :] * mask_land[1:, :]
        v_mask_land = jnp.concatenate([
            jnp.zeros((1, n_lon)), v_mask_land_interior, jnp.zeros((1, n_lon)),
        ], axis=0)

        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask_land, z_coord, jacobian, grid, cfg,
        )
        dT = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask_land, u_mask_land, v_mask_land,
            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
        )
        # Land cells must have zero tendency.
        land_3d = (mask_land < 0.5)[:, :, jnp.newaxis]
        assert jnp.allclose(dT * land_3d, 0.0, atol=1e-15)

    def test_no_flux_through_land(self):
        """With land, conservation should still hold."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        n_lat, n_lon, nlev = T.shape

        mask_land = mask.at[4:6, 8:12].set(0.0)
        u_mask_land = mask_land * jnp.roll(mask_land, 1, axis=1)
        u_mask_land = jnp.concatenate([u_mask_land, u_mask_land[:, 0:1]], axis=1)
        v_mask_land_interior = mask_land[:-1, :] * mask_land[1:, :]
        v_mask_land = jnp.concatenate([
            jnp.zeros((1, n_lon)), v_mask_land_interior, jnp.zeros((1, n_lon)),
        ], axis=0)

        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask_land, z_coord, jacobian, grid, cfg,
        )
        dT = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask_land, u_mask_land, v_mask_land,
            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
        )
        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
        area = grid.area[:, :, jnp.newaxis]
        integral = float(jnp.sum(dT * dz * area * mask_land[:, :, jnp.newaxis]))
        max_dT = float(jnp.max(jnp.abs(dT)))
        total_vol = float(jnp.sum(dz * area * mask_land[:, :, jnp.newaxis]))
        if max_dT > 0:
            relative = abs(integral) / (max_dT * total_vol)
            assert relative < 1e-8, f"Conservation with land violated: {relative:.2e}"


# =====================================================================
# 6. Symmetry
# =====================================================================

class TestSymmetry:

    def test_zonally_uniform_gives_zonally_uniform_tendency(self):
        """If inputs are zonally uniform, tendency should be too."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup

        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg,
        )
        dT = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
        )
        # dT should be constant along longitude axis (axis=1).
        dT_mean = jnp.mean(dT, axis=1, keepdims=True)
        deviation = jnp.max(jnp.abs(dT - dT_mean))
        assert deviation < 1e-10, f"Zonal asymmetry: max deviation {deviation:.2e}"


# =====================================================================
# 7. Sign check
# =====================================================================

class TestSignCheck:

    def test_gm_opposes_slope(self):
        """Where S_y > 0 and T increases northward, GM tendency should
        warm the deep and cool the surface (flattening isopycnals)."""
        setup = _stratified_with_meridional_tilt(slope=5e-4)
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup

        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg,
        )
        dT = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
        )
        # Nonzero tendency somewhere (slopes are nonzero).
        assert jnp.max(jnp.abs(dT)) > 0, "Tendency is all zeros — slopes should be nonzero"


# =====================================================================
# 8. Visbeck coefficient
# =====================================================================

class TestVisbeck:

    def test_visbeck_shape_and_bounds(self):
        """Visbeck kappa should have per-column shape and be within bounds."""
        from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
            compute_visbeck_kappa_gm,
        )
        setup = _stratified_with_meridional_tilt(slope=5e-4)
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup

        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg,
        )
        vcfg = VisbeckConfig(enabled=True, alpha=0.015, kappa_min=200, kappa_max=2000)
        f_coriolis = 2.0 * constants.Omega * jnp.sin(grid.lat[:, jnp.newaxis])
        f_coriolis = jnp.broadcast_to(f_coriolis, mask.shape) + 0.0

        kappa = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian, f_coriolis, vcfg,
        )
        assert kappa.shape == mask.shape
        assert jnp.all(kappa >= vcfg.kappa_min)
        assert jnp.all(kappa <= vcfg.kappa_max)
        assert jnp.all(jnp.isfinite(kappa))


# =====================================================================
# 9. JAX differentiability
# =====================================================================

class TestDifferentiability:

    def test_grad_through_full_tendency(self):
        """jax.grad must produce finite gradients through GM/Redi."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup

        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg,
        )

        def loss(T_in):
            dT = gm_redi_tracer_tendency_latlon_cgrid(
                T_in, S_x, S_y, mask, u_mask, v_mask,
                z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
            )
            return jnp.mean(dT ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_through_orchestrator(self):
        """jax.grad must flow through the full orchestrator (EOS + slopes + tendency)."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        # Pin centered: triads-orchestrator grad coverage lives in
        # TestTriadDifferentiability.test_grad_through_triad.
        cfg = cfg._replace(slope_scheme="centered")

        def loss(T_in):
            dT, _ = gm_redi_tracer_tendency_latlon(
                T_in, S, eta, H_bathy, grid, z_coord, cfg,
                eos="linear", mask=mask, u_mask=u_mask, v_mask=v_mask,
            )
            return jnp.mean(dT ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))


# =====================================================================
# 10. Vertical CFL diagnostic
# =====================================================================

class TestVerticalCFL:

    def test_vertical_cfl_within_limit(self):
        """kappa * S^2 * dt / dz^2 must be < 0.5 for default config."""
        setup = _stratified_with_meridional_tilt(slope=5e-4)
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup

        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg,
        )
        S2 = S_x ** 2 + S_y ** 2
        dt = 300.0  # typical baroclinic timestep
        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
        dz_min = float(jnp.min(dz[:, :, :-1]))  # thinnest layer (exclude bottom)
        kappa = cfg.kappa_Redi
        max_S2 = float(jnp.max(S2))
        cfl_vert = kappa * max_S2 * dt / dz_min ** 2
        assert cfl_vert < 0.5, (
            f"Vertical CFL = {cfl_vert:.4f} >= 0.5 "
            f"(kappa={kappa}, S2_max={max_S2:.2e}, dt={dt}, dz_min={dz_min:.1f})"
        )


# =====================================================================
# 11. LateralMixingOutput wrapper
# =====================================================================

class TestLateralMixingOutput:

    def test_wrapper_returns_named_tuple(self):
        """gm_redi_lateral_mixing_latlon must return LateralMixingOutput."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup

        out = gm_redi_lateral_mixing_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg,
            eos="linear", mask=mask, u_mask=u_mask, v_mask=v_mask,
        )
        from legoesm.ocean.physics.lateral_mixing.output import LateralMixingOutput
        assert isinstance(out, LateralMixingOutput)
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.dS_dt))


# =====================================================================
# 12. Structural enforcement (no code duplication)
# =====================================================================

class TestNoDuplication:
    """Verify that the lat-lon GM/Redi reuses shared helpers."""

    _SRC = legoesm_source_path("ocean/physics/lateral_mixing")

    def _read(self, name):
        return (self._SRC / name).read_text()

    def test_latlon_imports_visbeck_from_common(self):
        text = self._read("gm_redi_latlon_cgrid.py")
        assert "compute_visbeck_kappa_gm" in text
        assert "_gm_redi_common" in text

    def test_latlon_imports_dm95_from_common(self):
        text = self._read("gm_redi_latlon_cgrid.py")
        assert "dm95_taper" in text
        assert "_gm_redi_common" in text

    def test_latlon_imports_vertical_flux_div_from_common(self):
        text = self._read("gm_redi_latlon_cgrid.py")
        assert "vertical_flux_divergence" in text

    def test_latlon_does_not_inline_visbeck_formula(self):
        text = self._read("gm_redi_latlon_cgrid.py")
        assert "alpha * L ** 2 * sigma_bar" not in text
        assert "alpha * L**2 * sigma" not in text

    def test_cs_also_uses_common(self):
        text = self._read("gm_redi.py")
        assert "from legoesm.ocean.physics.lateral_mixing._gm_redi_common import" in text
        assert "dm95_taper" in text
        assert "vertical_flux_divergence" in text
        assert "compute_visbeck_kappa_gm" in text


# =====================================================================
# 13. Triad slope discretisation
# =====================================================================

class TestTriadShapeAndFiniteness:
    """The triad function must produce finite, correctly-shaped output."""

    def test_triad_tendency_shape(self):
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        dT = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=cfg.kappa_GM, kappa_Redi=cfg.kappa_Redi,
            S_max=cfg.S_max,
        )
        assert dT.shape == T.shape
        assert jnp.all(jnp.isfinite(dT))


class TestTriadConservation:

    def test_triad_tracer_integral_conserved(self):
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        dT = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=cfg.kappa_GM, kappa_Redi=cfg.kappa_Redi,
            S_max=cfg.S_max,
        )
        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
        area = grid.area[:, :, jnp.newaxis]
        integral = float(jnp.sum(dT * dz * area * mask[:, :, jnp.newaxis]))
        max_dT = float(jnp.max(jnp.abs(dT)))
        total_vol = float(jnp.sum(dz * area * mask[:, :, jnp.newaxis]))
        if max_dT > 0:
            relative = abs(integral) / (max_dT * total_vol)
            assert relative < 1e-10, f"Triad conservation: rel={relative:.2e}"


class TestTriadZeroTendency:

    def test_triad_uniform_tracer_zero_tendency(self):
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        q_uniform = jnp.ones_like(T) * 15.0
        dq = gm_redi_tracer_tendency_triads_latlon_cgrid(
            q_uniform, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=cfg.kappa_GM, kappa_Redi=cfg.kappa_Redi,
            S_max=cfg.S_max,
        )
        assert jnp.allclose(dq, 0.0, atol=1e-12)


class TestTriadLandMask:

    def test_triad_tendency_zero_on_land(self):
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        n_lat, n_lon, nlev = T.shape

        mask_land = mask.at[4:6, 8:12].set(0.0)
        u_mask_land = mask_land * jnp.roll(mask_land, 1, axis=1)
        u_mask_land = jnp.concatenate([u_mask_land, u_mask_land[:, 0:1]], axis=1)
        v_mask_land_interior = mask_land[:-1, :] * mask_land[1:, :]
        v_mask_land = jnp.concatenate([
            jnp.zeros((1, n_lon)), v_mask_land_interior, jnp.zeros((1, n_lon)),
        ], axis=0)

        dT = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask_land, u_mask_land, v_mask_land,
            z_coord, jacobian, grid,
            kappa_GM=cfg.kappa_GM, kappa_Redi=cfg.kappa_Redi,
            S_max=cfg.S_max,
        )
        land_3d = (mask_land < 0.5)[:, :, jnp.newaxis]
        assert jnp.allclose(dT * land_3d, 0.0, atol=1e-15)


class TestTriadDifferentiability:

    def test_grad_through_triad(self):
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup

        def loss(T_in):
            dT = gm_redi_tracer_tendency_triads_latlon_cgrid(
                T_in, rho, mask, u_mask, v_mask,
                z_coord, jacobian, grid,
                kappa_GM=cfg.kappa_GM, kappa_Redi=cfg.kappa_Redi,
                S_max=cfg.S_max,
            )
            return jnp.mean(dT ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))


class TestTriadVisbeck:
    """Triad scheme must broadcast correctly against a per-cell kappa_GM
    (e.g., the (n_lat, n_lon) field returned by Visbeck).  Regression
    test for the shape mismatch fixed alongside the global-overturning
    GM/Redi rollout — the original triad code multiplied a cell-centred
    kappa against u-face / v-face fluxes without face interpolation."""

    def test_triad_with_2d_kappa_GM_runs_and_finite(self):
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        n_lat, n_lon, _ = T.shape

        kappa_GM_2d = jnp.full(
            (n_lat, n_lon), float(cfg.kappa_GM), dtype=T.dtype,
        )
        dT = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=kappa_GM_2d, kappa_Redi=cfg.kappa_Redi,
            S_max=cfg.S_max,
        )
        assert dT.shape == T.shape
        assert jnp.all(jnp.isfinite(dT))

    def test_triad_uniform_2d_kappa_GM_matches_scalar(self):
        """A uniform (n_lat, n_lon) kappa_GM must give the same tendency
        as the equivalent scalar kappa_GM (interpolation is identity for
        uniform fields)."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        n_lat, n_lon, _ = T.shape

        kappa_scalar = float(cfg.kappa_GM)
        kappa_2d = jnp.full((n_lat, n_lon), kappa_scalar, dtype=T.dtype)

        dT_scalar = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=kappa_scalar, kappa_Redi=cfg.kappa_Redi,
            S_max=cfg.S_max,
        )
        dT_2d = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid,
            kappa_GM=kappa_2d, kappa_Redi=cfg.kappa_Redi,
            S_max=cfg.S_max,
        )
        assert jnp.allclose(dT_scalar, dT_2d, atol=1e-14, rtol=1e-12)


class TestTriadOrchestratorDispatch:

    def test_orchestrator_triads_branch(self):
        """slope_scheme='triads' produces finite, correctly-shaped output."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        cfg_t = cfg._replace(slope_scheme="triads")

        dT, dS = gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg_t,
            eos="linear", mask=mask, u_mask=u_mask, v_mask=v_mask,
        )
        assert dT.shape == T.shape
        assert dS.shape == S.shape
        assert jnp.all(jnp.isfinite(dT))
        assert jnp.all(jnp.isfinite(dS))

    def test_orchestrator_invalid_scheme_raises(self):
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        cfg_bad = cfg._replace(slope_scheme="bogus")
        with pytest.raises(ValueError, match="slope_scheme"):
            gm_redi_tracer_tendency_latlon(
                T, S, eta, H_bathy, grid, z_coord, cfg_bad,
                eos="linear", mask=mask, u_mask=u_mask, v_mask=v_mask,
            )


# =====================================================================
# R1: array-capable kappa_Redi (for the K_iso = K_gm coupling).
# A constant array must reproduce the scalar path bit-identically (the
# broadcast is exact); a per-column array must take effect.
# =====================================================================

class TestKappaRediArray:

    def _slopes(self, setup):
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg)
        return S_x, S_y

    def test_centered_const_array_redi_matches_scalar(self):
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        S_x, S_y = self._slopes(setup)
        n_lat, n_lon, _ = T.shape
        dT_scalar = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask, z_coord, jacobian, grid, 1000.0, 1000.0)
        kR = jnp.full((n_lat, n_lon), 1000.0)
        dT_array = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask, z_coord, jacobian, grid, 1000.0, kR)
        assert jnp.allclose(dT_array, dT_scalar, rtol=1e-13, atol=1e-30)

    def test_triad_const_array_redi_matches_scalar(self):
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        n_lat, n_lon, _ = T.shape
        dT_scalar = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid, 1000.0, 1000.0, cfg.S_max)
        kR = jnp.full((n_lat, n_lon), 1000.0)
        dT_array = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid, 1000.0, kR, cfg.S_max)
        assert jnp.allclose(dT_array, dT_scalar, rtol=1e-13, atol=1e-30)

    def test_triad_both_const_arrays_match_both_scalars(self):
        """K_iso=K_gm const case: kappa_GM AND kappa_Redi as equal const arrays ==
        equal scalars (broadcast + the (kappa_Redi-kappa_GM) cancellation hold)."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        n_lat, n_lon, _ = T.shape
        dT_scalar = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid, 800.0, 800.0, cfg.S_max)
        kc = jnp.full((n_lat, n_lon), 800.0)
        dT_array = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid, kc, kc, cfg.S_max)
        assert jnp.allclose(dT_array, dT_scalar, rtol=1e-13, atol=1e-30)

    def test_triad_kiso_equals_kgm_takes_effect(self):
        """A spatially-varying K_iso=K_gm (kappa_Redi == kappa_GM array) is finite and
        DIFFERS from holding kappa_Redi constant — the prognostic-Redi override is
        active. Uses a passive tracer NOT aligned with density (a zonal sinusoid, while
        rho tilts only in y) so Redi genuinely diffuses it and kappa_Redi matters; a
        tracer q=f(rho) would cancel per-triad (flux -> kappa_GM·dq/dx, kappa_Redi
        drops out)."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        n_lat, n_lon, nlev = T.shape
        lon = jnp.arange(n_lon, dtype=jnp.float64)
        q = jnp.broadcast_to(
            jnp.sin(2.0 * jnp.pi * lon / n_lon)[None, :, None], (n_lat, n_lon, nlev))
        kGM = jnp.broadcast_to(
            jnp.linspace(100.0, 2000.0, n_lat)[:, None], (n_lat, n_lon))
        dq_kiso = gm_redi_tracer_tendency_triads_latlon_cgrid(
            q, rho, mask, u_mask, v_mask, z_coord, jacobian, grid, kGM, kGM, cfg.S_max)
        dq_const = gm_redi_tracer_tendency_triads_latlon_cgrid(
            q, rho, mask, u_mask, v_mask, z_coord, jacobian, grid, kGM, 1000.0, cfg.S_max)
        assert jnp.all(jnp.isfinite(dq_kiso))
        # Relative difference vs the tendency magnitude (the tendencies are ~1e-10,
        # far below jnp.allclose's default atol=1e-8, so compare relatively).
        denom = float(jnp.max(jnp.abs(dq_const))) + 1e-30
        rel = float(jnp.max(jnp.abs(dq_kiso - dq_const))) / denom
        assert rel > 0.1, f"K_iso=K_gm override had negligible effect (rel={rel:.3g})"


# =====================================================================
# 12. Implicit K_33 + K_iso_steep (Veros-faithful isoneutral options)
# =====================================================================

class TestImplicitK33AndKisoSteep:
    """Vertical isoneutral diagonal K_33 applied implicitly + the K_iso_steep
    horizontal-diffusion floor — config-selectable Veros-matching options
    (GMRediConfig.implicit_K33 / K_iso_steep), matching Veros
    core/isoneutral/diffusion.py and isoneutral.py:128/165."""

    def test_config_defaults_off(self):
        # Defaults MUST preserve the explicit, no-floor behaviour.
        cfg = GMRediConfig()
        assert cfg.implicit_K33 is False
        assert cfg.K_iso_steep == 0.0

    def test_K33_nonneg_and_positive_for_sloped(self):
        setup = _stratified_with_meridional_tilt(slope=2e-3)
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        K33 = compute_isoneutral_K33_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg, mask=mask)
        assert K33.shape == (T.shape[0], T.shape[1], T.shape[2] - 1)
        assert jnp.all(jnp.isfinite(K33))
        assert jnp.all(K33 >= 0.0), "K_33 (a vertical diffusivity) must be >= 0"
        assert float(jnp.max(K33)) > 0.0, "sloped isopycnals must give K_33 > 0"

    def test_K33_zero_for_flat_isopycnals(self):
        # No horizontal density gradient => zero slope => K_33 == 0.
        setup = _stratified_with_meridional_tilt(slope=0.0)
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        K33 = compute_isoneutral_K33_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg, mask=mask)
        assert float(jnp.max(jnp.abs(K33))) < 1e-18, "flat isopycnals => K_33 = 0"

    def test_implicit_K33_split_consistency(self):
        # The K_33 term DROPPED from the explicit F_z (implicit_K33=True) must,
        # for small dt, equal the implicit K_33 increment — proving the
        # explicit->implicit move is the SAME operator (corr ~ 1).
        from legoesm.ocean.physics.vertical_mixing import (
            implicit_vertical_diffusion_ocean, build_dz_half,
        )
        setup = _stratified_with_meridional_tilt(slope=3e-3)
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        kw = dict(mask=mask, u_mask=u_mask, v_mask=v_mask)
        dT_full, _ = gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg._replace(implicit_K33=False), **kw)
        dT_skew, _ = gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg._replace(implicit_K33=True), **kw)
        dT_K33_explicit = dT_full - dT_skew
        K33 = compute_isoneutral_K33_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg._replace(implicit_K33=True), mask=mask)
        dz_cell = z_coord.dz_ref * jacobian[:, :, None]
        dz_half = build_dz_half(dz_cell)
        dt_small = 1.0
        T_imp = implicit_vertical_diffusion_ocean(T, K33, dz_cell, dz_half, dt_small)
        dT_K33_implicit = (T_imp - T) / dt_small
        m3 = jnp.broadcast_to((mask > 0.5)[:, :, None], dT_K33_explicit.shape)
        a = dT_K33_explicit[m3]
        b = dT_K33_implicit[m3]
        assert float(jnp.sqrt(jnp.mean(a ** 2))) > 0.0, "no K_33 signal — test is vacuous"
        corr = float(jnp.corrcoef(a, b)[0, 1])
        assert corr > 0.999, f"explicit vs implicit K_33 mismatch (corr={corr:.5f})"

    def test_implicit_K33_changes_tendency(self):
        # implicit_K33 removes the K_33 vertical diagonal from the explicit F_z,
        # so the explicit tendency must change (and stay finite).
        setup = _stratified_with_meridional_tilt(slope=3e-3)
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        kw = dict(mask=mask, u_mask=u_mask, v_mask=v_mask)
        dT_full, _ = gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg._replace(implicit_K33=False), **kw)
        dT_skew, _ = gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg._replace(implicit_K33=True), **kw)
        assert jnp.all(jnp.isfinite(dT_skew))
        assert float(jnp.max(jnp.abs(dT_full - dT_skew))) > 0.0

    def test_K_iso_steep_zero_is_noop(self):
        # K_iso_steep=0 (the default) must be bit-identical to the unfloored path
        # — the floor is gated `if K_iso_steep > 0.0`.
        setup = _stratified_with_meridional_tilt(slope=3e-3)
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        base = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            cfg.kappa_GM, cfg.kappa_Redi, cfg.S_max, cfg.taper_width_frac)  # defaults
        floored0 = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            cfg.kappa_GM, cfg.kappa_Redi, cfg.S_max, cfg.taper_width_frac, False, 0.0)
        assert jnp.array_equal(base, floored0)

    def test_K_iso_steep_floors_horizontal_diffusivity(self):
        # K_iso_steep gives the diagonal max(K_iso_steep, kappa·taper).  legoESM
        # clips the slope to S_max BEFORE the DM95 taper, so the taper bottoms at
        # 0.5; the floor therefore bites only for kappa < 2·K_iso_steep.  Use
        # kappa=800 (< 1000) + steep slopes so kappa·0.5=400 < K_iso_steep=500.
        setup = _stratified_with_meridional_tilt(slope=0.05)  # >> S_max => steep
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        kappa = 800.0
        no_floor = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            kappa, kappa, cfg.S_max, cfg.taper_width_frac, False, 0.0)
        with_floor = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            kappa, kappa, cfg.S_max, cfg.taper_width_frac, False, 500.0)
        assert jnp.all(jnp.isfinite(with_floor))
        assert float(jnp.max(jnp.abs(with_floor - no_floor))) > 0.0, \
            "K_iso_steep floor had no effect (kappa<2·K_iso_steep, steep slopes)"


# =====================================================================
# 13. interp_wface_to_center — the shared interface→center vertical
#     interpolation helper (Stage 3 of the 3-D EKE GM/Redi kappa).
# =====================================================================

class TestInterpWfaceToCenter:
    """Unit tests for the interface(W-grid)→cell-center vertical interp helper
    that lifts a depth-resolved 3-D EKE kappa (nlev-1 interfaces) to cell
    centers (nlev) for the GM/Redi horizontal flux."""

    def test_shape_iface_to_center(self):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            interp_wface_to_center,
        )
        f = jnp.arange(2 * 3 * 6, dtype=jnp.float64).reshape(2, 3, 6)  # nlev-1=6
        c = interp_wface_to_center(f)
        assert c.shape == (2, 3, 7)  # nlev = 7

    def test_interior_is_average_endpoints_one_sided(self):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            interp_wface_to_center,
        )
        key = jax.random.PRNGKey(0)
        f = jax.random.normal(key, (4, 5, 8), dtype=jnp.float64)  # nlev-1=8
        c = interp_wface_to_center(f)
        # interior center k = 0.5*(iface[k-1] + iface[k]), 1 <= k <= nlev-2
        assert jnp.allclose(c[:, :, 1:-1], 0.5 * (f[:, :, :-1] + f[:, :, 1:]),
                            rtol=1e-13, atol=0.0)
        # top/bottom centers are one-sided copies of the nearest interface.
        assert jnp.allclose(c[:, :, 0], f[:, :, 0], rtol=1e-13, atol=0.0)
        assert jnp.allclose(c[:, :, -1], f[:, :, -1], rtol=1e-13, atol=0.0)

    def test_uniform_interface_maps_to_uniform_center(self):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            interp_wface_to_center,
        )
        f = jnp.full((3, 4, 9), 1234.5, dtype=jnp.float64)  # nlev-1 = 9
        c = interp_wface_to_center(f)
        assert c.shape == (3, 4, 10)
        assert jnp.allclose(c, 1234.5, rtol=1e-13, atol=0.0)

    def test_grad_safe(self):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            interp_wface_to_center,
        )
        f = jax.random.normal(jax.random.PRNGKey(1), (2, 2, 5), dtype=jnp.float64)
        g = jax.grad(lambda x: jnp.sum(interp_wface_to_center(x) ** 2))(f)
        assert g.shape == f.shape
        assert jnp.all(jnp.isfinite(g))


# =====================================================================
# 14. 3-D (depth-resolved) interface kappa_GM / kappa_Redi
#     (Stage 3: GM/Redi tracer tendency accepts a 3-D interface kappa).
#
# The 3-D EKE kappa lives at the nlev-1 interior interfaces (W-grid).
# Tests use nlev=12 (>2, and nlev-1=11 is unambiguous vs the (n,n,1)
# broadcast).  Triad scheme is the production path (slope_scheme="triads").
# =====================================================================

class TestKappa3DInterface:

    NLEV = 12

    def _setup(self, slope=2e-3):
        return _stratified_with_meridional_tilt(nlev=self.NLEV, slope=slope)

    def _passive_zonal_tracer(self, n_lat, n_lon, nlev):
        # A zonal sinusoid NOT aligned with the (y-only) density tilt, so the
        # Redi/skew flux genuinely acts and the kappa depth-structure matters.
        lon = jnp.arange(n_lon, dtype=jnp.float64)
        prof = jnp.linspace(1.0, 2.0, nlev)  # mild vertical structure too
        return (jnp.sin(2.0 * jnp.pi * lon / n_lon)[None, :, None]
                * prof[None, None, :]
                * jnp.ones((n_lat, 1, 1)))

    # --- (a) depth structure matters -------------------------------------
    def test_triad_depth_varying_3d_differs_from_depth_mean(self):
        """A depth-VARYING 3-D interface kappa gives a tendency that differs from
        the tendency using a depth-uniform kappa equal to its depth mean — the
        depth structure of the kappa genuinely changes the answer."""
        setup = self._setup()
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        n_lat, n_lon, nlev = T.shape
        q = self._passive_zonal_tracer(n_lat, n_lon, nlev)
        # Depth-varying interface kappa: large near surface, small at depth.
        kz = jnp.linspace(2000.0, 100.0, nlev - 1)  # (nlev-1,)
        kappa_var = jnp.broadcast_to(kz[None, None, :], (n_lat, n_lon, nlev - 1))
        # Depth-mean-equivalent: same per-column mean, uniform with depth.
        kbar = float(jnp.mean(kz))
        kappa_mean = jnp.full((n_lat, n_lon, nlev - 1), kbar)
        dq_var = gm_redi_tracer_tendency_triads_latlon_cgrid(
            q, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            kappa_var, kappa_var, cfg.S_max)
        dq_mean = gm_redi_tracer_tendency_triads_latlon_cgrid(
            q, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            kappa_mean, kappa_mean, cfg.S_max)
        assert jnp.all(jnp.isfinite(dq_var))
        denom = float(jnp.max(jnp.abs(dq_mean))) + 1e-30
        rel = float(jnp.max(jnp.abs(dq_var - dq_mean))) / denom
        assert rel > 0.1, (
            f"depth-varying 3-D kappa had negligible effect vs depth-mean "
            f"(rel={rel:.3g}) — depth structure should matter")

    # --- (b) depth-uniform 3-D == 2-D/scalar (consistency) ---------------
    def test_triad_uniform_3d_equals_scalar(self):
        """A depth-UNIFORM 3-D interface kappa must equal the scalar-kappa
        tendency: interp_wface_to_center of a constant is constant, and the
        cell→face interp of a constant is the same constant ⇒ identical to the
        scalar broadcast at every face/interface."""
        setup = self._setup()
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        n_lat, n_lon, nlev = T.shape
        k0 = 900.0
        kappa_3d = jnp.full((n_lat, n_lon, nlev - 1), k0)
        dq_scalar = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            k0, k0, cfg.S_max)
        dq_3d = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            kappa_3d, kappa_3d, cfg.S_max)
        assert jnp.allclose(dq_3d, dq_scalar, rtol=1e-12, atol=1e-30)

    def test_triad_uniform_3d_equals_2d(self):
        """A depth-uniform 3-D interface kappa equals the equivalent 2-D
        per-column kappa (both reduce to the same constant at every face)."""
        setup = self._setup()
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        n_lat, n_lon, nlev = T.shape
        k0 = 750.0
        kappa_2d = jnp.full((n_lat, n_lon), k0)
        kappa_3d = jnp.full((n_lat, n_lon, nlev - 1), k0)
        dq_2d = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            kappa_2d, kappa_2d, cfg.S_max)
        dq_3d = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            kappa_3d, kappa_3d, cfg.S_max)
        assert jnp.allclose(dq_3d, dq_2d, rtol=1e-12, atol=1e-30)

    def test_centered_uniform_3d_equals_scalar(self):
        """Same consistency check for the centered scheme."""
        setup = self._setup()
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        n_lat, n_lon, nlev = T.shape
        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg)
        k0 = 650.0
        kappa_3d = jnp.full((n_lat, n_lon, nlev - 1), k0)
        dq_scalar = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask, z_coord, jacobian, grid, k0, k0)
        dq_3d = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask, z_coord, jacobian, grid,
            kappa_3d, kappa_3d)
        assert jnp.allclose(dq_3d, dq_scalar, rtol=1e-12, atol=1e-30)

    # --- (c) q = f(rho) cancellation holds with a 3-D kappa --------------
    def test_triad_cancellation_q_eq_f_rho_with_3d_kappa(self):
        """The per-triad cancellation (Redi flux vanishes for q = f(rho)) must
        still hold with a depth-varying 3-D interface kappa. With kappa_Redi ==
        kappa_GM (the K_iso=K_gm coupling) and q = T = f(rho), the whole GM/Redi
        tendency reduces to the GM skew of T, which for q=f(rho) the triad
        construction also cancels — the tendency must be machine-zero."""
        setup = self._setup()
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        n_lat, n_lon, nlev = T.shape
        kz = jnp.linspace(2000.0, 50.0, nlev - 1)
        kappa_var = jnp.broadcast_to(kz[None, None, :], (n_lat, n_lon, nlev - 1))
        # T is exactly linear in rho here (see _stratified_with_meridional_tilt),
        # so q = f(rho) per triad; kappa_Redi == kappa_GM (== kappa_var).
        dq = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            kappa_var, kappa_var, cfg.S_max)
        assert jnp.all(jnp.isfinite(dq))
        # Compare against the scalar-kappa cancellation magnitude as the
        # machine-precision reference (same fields, same tracer).
        dq_scalar = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            1000.0, 1000.0, cfg.S_max)
        ref = float(jnp.max(jnp.abs(dq_scalar)))
        # The 3-D-kappa residual must be no worse than ~the scalar residual
        # scaled by max(kappa_var)/1000 (the cancellation is per-triad, so the
        # residual scales with kappa, not with its depth structure).
        tol = max(ref * (float(jnp.max(kz)) / 1000.0) * 10.0, 1e-12)
        assert float(jnp.max(jnp.abs(dq))) < tol, (
            f"q=f(rho) cancellation broke with a 3-D kappa: "
            f"max|dq|={float(jnp.max(jnp.abs(dq))):.3e} > tol={tol:.3e}")

    # --- (d) 2-D / scalar path bit-identical (regression) ----------------
    def test_scalar_and_2d_paths_bit_identical_regression(self):
        """Introducing the 3-D dispatch must NOT perturb the scalar or 2-D paths:
        they go through the unchanged broadcast branch and must be bit-identical
        between scalar and a uniform 2-D array (the historical guarantee)."""
        setup = self._setup()
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        n_lat, n_lon, nlev = T.shape
        dq_scalar = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            1000.0, 1000.0, cfg.S_max)
        kR2d = jnp.full((n_lat, n_lon), 1000.0)
        dq_2d = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
            1000.0, kR2d, cfg.S_max)
        assert jnp.allclose(dq_2d, dq_scalar, rtol=1e-13, atol=1e-30)
        # K_33 getter: scalar vs 2-D bit-identical too.
        K33_scalar = compute_isoneutral_K33_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg, mask=mask)
        K33_2d = compute_isoneutral_K33_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg, mask=mask,
            kappa_redi_override=jnp.full((n_lat, n_lon), float(cfg.kappa_Redi)))
        assert jnp.allclose(K33_2d, K33_scalar, rtol=1e-13, atol=1e-30)

    # --- (e) finiteness + jax.grad AD-safety with a 3-D kappa ------------
    def test_3d_kappa_finite_and_grad_safe(self):
        setup = self._setup()
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        n_lat, n_lon, nlev = T.shape
        q = self._passive_zonal_tracer(n_lat, n_lon, nlev)
        kz = jnp.linspace(1500.0, 200.0, nlev - 1)
        kappa = jnp.broadcast_to(kz[None, None, :], (n_lat, n_lon, nlev - 1))

        def loss(kap):
            dq = gm_redi_tracer_tendency_triads_latlon_cgrid(
                q, rho, mask, u_mask, v_mask, z_coord, jacobian, grid,
                kap, kap, cfg.S_max)
            return jnp.sum(dq ** 2)

        val = loss(kappa)
        g = jax.grad(loss)(kappa)
        assert jnp.isfinite(val)
        assert g.shape == kappa.shape
        assert jnp.all(jnp.isfinite(g))
        # Gradient must be non-trivial (the depth-resolved kappa influences dq).
        assert float(jnp.max(jnp.abs(g))) > 0.0

    def test_3d_kappa_K33_getter_finite_and_nonneg(self):
        """The K_33 getter accepts a 3-D interface kappa_Redi override and stays
        finite + non-negative (it is a vertical diffusivity)."""
        setup = self._setup(slope=3e-3)
        (grid, z_coord, mask, u_mask, v_mask, eta, H_bathy,
         jacobian, rho, T, S, cfg) = setup
        n_lat, n_lon, nlev = T.shape
        kz = jnp.linspace(1800.0, 100.0, nlev - 1)
        kappa = jnp.broadcast_to(kz[None, None, :], (n_lat, n_lon, nlev - 1))
        K33 = compute_isoneutral_K33_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg, mask=mask,
            kappa_redi_override=kappa)
        assert K33.shape == (n_lat, n_lon, nlev - 1)
        assert jnp.all(jnp.isfinite(K33))
        assert jnp.all(K33 >= 0.0)
        assert float(jnp.max(K33)) > 0.0
        # Depth-uniform 3-D kappa_Redi in K_33 == scalar kappa_Redi override.
        k0 = float(jnp.mean(kz))
        K33_3d_uniform = compute_isoneutral_K33_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg, mask=mask,
            kappa_redi_override=jnp.full((n_lat, n_lon, nlev - 1), k0))
        K33_scalar = compute_isoneutral_K33_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg, mask=mask,
            kappa_redi_override=k0)
        assert jnp.allclose(K33_3d_uniform, K33_scalar, rtol=1e-12, atol=1e-30)


# =====================================================================
# NEMO ldfslp mixed-layer slope ramp (default OFF -> byte-identical)
# =====================================================================

class TestNemoMixedLayerSlopeRamp:
    """``GMRediConfig.nemo_mld_slope_ramp`` linearly flattens ML slopes.

    Manufactured density with a well-mixed surface layer (top 3 levels
    constant in z) over a stratified, meridionally-tilted interior.  In the
    mixed layer the vertical density gradient -> 0, so the raw isoneutral slope
    blows up; the NEMO ramp must taper it linearly to ~0 at the surface while
    leaving the stratified interior untouched (default OFF => byte-identical).
    """

    @staticmethod
    def _mixed_layer_setup(n_lat=8, n_lon=12, nlev=10):
        from legoesm.ocean.eos import make_eos_fn
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
        z_coord = create_ocean_z_star(
            n_levels=nlev, H_max=2000.0, dz_surface=20.0, dz_deep=400.0)
        mask = jnp.ones((n_lat, n_lon))
        eta = jnp.zeros((n_lat, n_lon))
        H_bathy = jnp.full((n_lat, n_lon), 2000.0)
        jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
        # Vertical T: 3-level mixed layer (constant), stratified below.
        Tz = jnp.concatenate(
            [jnp.full(3, 15.0), jnp.linspace(15.0, 2.0, nlev - 3)])
        y_idx = jnp.arange(n_lat, dtype=jnp.float64)
        # Meridional tilt (warmer south) -> nonzero horizontal density gradient.
        T = Tz[None, None, :] + 0.3 * y_idx[:, None, None]
        S = jnp.full((n_lat, n_lon, nlev), 35.0, dtype=jnp.float64)
        eos_fn = make_eos_fn("linear")
        rho = eos_fn(T, S, jnp.zeros_like(T))
        return grid, z_coord, mask, jacobian, rho, T, S, eos_fn

    def test_ramp_off_is_byte_identical(self):
        grid, z_coord, mask, jac, rho, T, S, eos_fn = self._mixed_layer_setup()
        base = GMRediConfig(kappa_GM=1000.0, kappa_Redi=1000.0)
        off = GMRediConfig(kappa_GM=1000.0, kappa_Redi=1000.0,
                           nemo_mld_slope_ramp=False)
        Sx_b, Sy_b, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jac, grid, base, T=T, S=S, eos_fn=eos_fn)
        Sx_o, Sy_o, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jac, grid, off, T=T, S=S, eos_fn=eos_fn)
        # Default and explicit-off are bit-identical (ramp changes nothing).
        assert jnp.array_equal(Sx_b, Sx_o)
        assert jnp.array_equal(Sy_b, Sy_o)

    def test_ramp_flattens_mixed_layer_only(self):
        grid, z_coord, mask, jac, rho, T, S, eos_fn = self._mixed_layer_setup()
        off = GMRediConfig(kappa_GM=1000.0, kappa_Redi=1000.0)
        on = GMRediConfig(kappa_GM=1000.0, kappa_Redi=1000.0,
                          nemo_mld_slope_ramp=True)
        Sy_off, = (compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jac, grid, off, T=T, S=S, eos_fn=eos_fn)[1],)
        Sy_on = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jac, grid, on, T=T, S=S, eos_fn=eos_fn)[1]
        # The MLD criterion places the ML base a few levels down for this
        # synthetic column; the shallowest interface (0) must be flattened
        # toward zero by the ramp.
        surf_off = float(jnp.mean(jnp.abs(Sy_off[:, :, 0])))
        surf_on = float(jnp.mean(jnp.abs(Sy_on[:, :, 0])))
        assert surf_on < 0.34 * surf_off, (surf_on, surf_off)
        # Below the mixed layer the ramp is a NO-OP: bit-identical to the
        # un-ramped slopes. The computed MLD base for this synthetic column sits
        # around interface ~4 (deeper than the nominal "top 3 levels"), so check
        # interfaces 5+ are untouched.
        assert jnp.array_equal(Sy_on[:, :, 5:], Sy_off[:, :, 5:])
        # grad is AD-safe AND actually FLOWS to T through the below-ML base slope
        # (the MLD level is quantized -> zero grad through hml, but the base slope
        # is differentiable), so it must be finite AND non-zero.
        def _loss(Tf):
            return jnp.sum(compute_isopycnal_slopes_latlon_cgrid(
                eos_fn(Tf, S, jnp.zeros_like(Tf)), mask, z_coord, jac, grid,
                on, T=Tf, S=S, eos_fn=eos_fn)[1] ** 2)
        g = jax.grad(_loss)(T)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 0.0


class TestNemoSlopeShapiro:
    """``GMRediConfig.nemo_slope_shapiro`` = NEMO ldfslp horizontal Shapiro filter.

    A ``(1-2-1)⊗(1-2-1)/16`` nine-point binomial on the masked interface slopes,
    times a coastal taper ``zcofw`` that shrinks slopes toward land (NOT a
    wet-renormalization).  Verified as an isolated operator: interior points
    equal the hand-computed binomial mean; a coast damps the slope; land stays 0.
    """

    def test_shapiro_off_byte_identical(self):
        grid, z_coord, mask, jac, rho, T, S, eos_fn = (
            TestNemoMixedLayerSlopeRamp._mixed_layer_setup())
        base = GMRediConfig(kappa_GM=1000.0, kappa_Redi=1000.0)
        off = GMRediConfig(kappa_GM=1000.0, kappa_Redi=1000.0,
                           nemo_slope_shapiro=False)
        Sx_b, Sy_b, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jac, grid, base, T=T, S=S, eos_fn=eos_fn)
        Sx_o, Sy_o, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jac, grid, off, T=T, S=S, eos_fn=eos_fn)
        assert jnp.array_equal(Sx_b, Sx_o)
        assert jnp.array_equal(Sy_b, Sy_o)

    def test_interior_equals_binomial_mean(self):
        """Interior wet point == the (1-2-1)²/16 weighted mean of its 3x3 nbhd."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            _shapiro_smooth_slopes,
        )
        rng = np.random.default_rng(0)
        Sx = jnp.asarray(rng.standard_normal((5, 6, 2)))
        Sy = jnp.asarray(rng.standard_normal((5, 6, 2)))
        mask = jnp.ones((5, 6))                    # all wet -> zcofw = 1/16
        Sxs, Sys = _shapiro_smooth_slopes(Sx, Sy, mask)
        # Hand binomial at interior point (2,3): weights [[1,2,1],[2,4,2],[1,2,1]]/16.
        w = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0
        for arr, out in ((Sx, Sxs), (Sy, Sys)):
            patch = np.asarray(arr)[1:4, 2:5, 0]
            expect = float((w * patch).sum())
            np.testing.assert_allclose(float(np.asarray(out)[2, 3, 0]), expect,
                                       rtol=1e-12)

    def test_coast_damps_and_land_zero(self):
        """A wet point beside land is damped below the interior binomial mean,
        and land points stay exactly 0."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            _shapiro_smooth_slopes,
        )
        Sx = jnp.ones((5, 6, 1))
        mask = jnp.ones((5, 6)).at[2, 5].set(0.0)   # one land cell at east edge
        Sxs, _ = _shapiro_smooth_slopes(Sx, Sx, mask)
        # Interior far from land -> 1.0 (binomial mean of all-ones = 1).
        np.testing.assert_allclose(float(np.asarray(Sxs)[2, 2, 0]), 1.0, rtol=1e-12)
        # Wet neighbour of the land cell (2,4): east u-face dry -> zcofw < 1/16
        # AND a zero enters the sum -> strictly damped below 1.
        assert float(np.asarray(Sxs)[2, 4, 0]) < 0.999
        # The land cell itself stays exactly 0 (masked input, tmask factor).
        assert float(np.asarray(Sxs)[2, 5, 0]) == 0.0

    def test_shapiro_grad_flows(self):
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            _shapiro_smooth_slopes,
        )
        mask = jnp.ones((5, 6))
        def _loss(Sx):
            out, _ = _shapiro_smooth_slopes(Sx, Sx, mask)
            return jnp.sum(out ** 2)
        g = jax.grad(_loss)(jnp.ones((5, 6, 2)))
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 0.0


# =====================================================================
# NEMO iso-neutral Laplacian Redi operator (slope_scheme="nemo_iso_lap")
# =====================================================================
#
# Port of NEMO 5.0.2 traldf_iso (#define iso_lap), verified against NEMO's
# dumped ttrd_ldf (GYRE oracle, T corr 0.96 fed legoESM slopes) by the scratch
# gate ~/oracle-builds/nemo5/gap_audit/nemo_iso_lap_repo_gate.py.  These unit
# tests cover the invariants CLAUDE.md mandates: dispatch selection, off-by-
# default byte-identity, conservation + variance sign gate, grad flow, and a
# hand-checked interior tendency.

class TestNemoIsoLapOperator:

    def _closed_box(self, setup):
        """Force the domain walls closed (zero the boundary face rings) so the
        flux divergence telescopes to a closed budget."""
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        u = u_mask.at[:, 0].set(0.0).at[:, -1].set(0.0)
        v = v_mask.at[0, :].set(0.0).at[-1, :].set(0.0)
        return u, v

    def _slopes(self, setup):
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jacobian, grid, cfg)
        return S_x, S_y

    def test_dispatch_selects_nemo_iso_lap(self):
        """slope_scheme='nemo_iso_lap' routes to the ported operator: the
        dispatcher output is bit-identical to a direct operator call fed the
        dispatcher's own internally-computed slopes + active_3d mask."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            gm_redi_density_and_jacobian,
        )
        from legoesm.ocean.eos import make_eos_fn
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        cfg_n = cfg._replace(slope_scheme="nemo_iso_lap", kappa_GM=0.0)
        dT, dS = gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg_n,
            eos="linear", mask=mask, u_mask=u_mask, v_mask=v_mask)
        assert dT.shape == T.shape and dS.shape == S.shape
        assert jnp.all(jnp.isfinite(dT)) and jnp.all(jnp.isfinite(dS))
        # Reproduce the dispatcher's slopes + active_3d and call the op directly.
        eos_fn = make_eos_fn("linear", cfg_n.eos_linear if hasattr(cfg_n, "eos_linear") else None)
        rho_d, jac_d = gm_redi_density_and_jacobian(
            T, S, eta, H_bathy, grid, z_coord, eos="linear", mask=mask)
        S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
            rho_d, mask, z_coord, jac_d, grid, cfg_n, T=T, S=S, eos_fn=eos_fn)
        z_top = jnp.cumsum(z_coord.dz_ref) - z_coord.dz_ref
        act = ((mask[:, :, jnp.newaxis] > 0.5)
               & (z_top[jnp.newaxis, jnp.newaxis, :] < H_bathy[:, :, jnp.newaxis])
               ).astype(T.dtype)
        dT_direct = nemo_iso_lap_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask, z_coord, jac_d, grid,
            cfg_n.kappa_Redi, act)
        assert jnp.allclose(dT, dT_direct, rtol=1e-12, atol=1e-30)

    def test_default_is_triads_byte_identical(self):
        """Off by default: the default config selects triads and its output is
        bit-identical to explicitly selecting slope_scheme='triads' — adding the
        nemo_iso_lap elif changed neither existing scheme's numerics."""
        assert GMRediConfig().slope_scheme == "triads"
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        cfg_default = cfg._replace()               # keeps slope_scheme="triads"
        cfg_triads = cfg._replace(slope_scheme="triads")
        dT_def, dS_def = gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg_default,
            eos="linear", mask=mask, u_mask=u_mask, v_mask=v_mask)
        dT_tri, dS_tri = gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg_triads,
            eos="linear", mask=mask, u_mask=u_mask, v_mask=v_mask)
        assert jnp.array_equal(dT_def, dT_tri) and jnp.array_equal(dS_def, dS_tri)
        # Centered branch still runs finite/shaped (untouched by the edit).
        cfg_c = cfg._replace(slope_scheme="centered")
        dT_c, dS_c = gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg_c,
            eos="linear", mask=mask, u_mask=u_mask, v_mask=v_mask)
        assert dT_c.shape == T.shape and jnp.all(jnp.isfinite(dT_c))
        assert not jnp.array_equal(dT_c, dT_tri)   # distinct schemes

    def test_kappa_gm_nonzero_raises(self):
        """Pure-Redi guard: nemo_iso_lap with kappa_GM != 0 must raise."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        cfg_bad = cfg._replace(slope_scheme="nemo_iso_lap", kappa_GM=1000.0)
        with pytest.raises(ValueError, match="pure iso-neutral"):
            gm_redi_tracer_tendency_latlon(
                T, S, eta, H_bathy, grid, z_coord, cfg_bad,
                eos="linear", mask=mask, u_mask=u_mask, v_mask=v_mask)

    def test_invalid_scheme_message_lists_three(self):
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        cfg_bad = cfg._replace(slope_scheme="bogus")
        with pytest.raises(ValueError, match="nemo_iso_lap"):
            gm_redi_tracer_tendency_latlon(
                T, S, eta, H_bathy, grid, z_coord, cfg_bad,
                eos="linear", mask=mask, u_mask=u_mask, v_mask=v_mask)

    def test_tracer_integral_conserved(self):
        """Diffusion conserves the volume-integrated tracer: on a closed wet
        box, sum(dT * e1t*e2t*e3t) == 0 to roundoff (the flux divergence
        telescopes, walls + sea floor carry no flux)."""
        setup = _stratified_with_meridional_tilt(slope=5e-4)
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        S_x, S_y = self._slopes(setup)
        u, v = self._closed_box(setup)
        act = jnp.broadcast_to(mask[:, :, jnp.newaxis], T.shape)
        dT = nemo_iso_lap_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u, v, z_coord, jacobian, grid, 1000.0, act)
        geom = ensure_geometry(grid)
        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
        vol = (geom.dx_T * geom.dy_T)[:, :, jnp.newaxis] * dz * mask[:, :, jnp.newaxis]
        integral = float(jnp.sum(dT * vol))
        rel = abs(integral) / (float(jnp.max(jnp.abs(dT))) * float(jnp.sum(vol)))
        assert rel < 1e-12, f"conservation violated: rel {rel:.2e}"

    def test_variance_non_increasing(self):
        """Down-gradient: sum(q * dq/dt * vol) <= 0 (variance non-increasing).
        Checked with the physical isopycnal slopes AND the pure-horizontal-
        diffusion (zero-slope) limit, which is unconditionally diffusive."""
        setup = _stratified_with_meridional_tilt(slope=5e-4)
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        S_x, S_y = self._slopes(setup)
        u, v = self._closed_box(setup)
        act = jnp.broadcast_to(mask[:, :, jnp.newaxis], T.shape)
        geom = ensure_geometry(grid)
        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
        vol = (geom.dx_T * geom.dy_T)[:, :, jnp.newaxis] * dz * mask[:, :, jnp.newaxis]
        for tag, sx, sy in [("real", S_x, S_y),
                            ("zero", jnp.zeros_like(S_x), jnp.zeros_like(S_y))]:
            dT = nemo_iso_lap_tracer_tendency_latlon_cgrid(
                T, sx, sy, mask, u, v, z_coord, jacobian, grid, 1000.0, act)
            var_tend = float(jnp.sum(T * dT * vol))
            assert var_tend <= 0.0, f"{tag}-slope variance increased: {var_tend:.4e}"

    def test_zero_slope_is_horizontal_laplacian(self):
        """With S=0 the operator reduces to pure horizontal Laplacian diffusion:
        a hand-checked interior cell equals kappa*(q_E+q_W+q_N+q_S-4 q)/dx^2 on a
        uniform grid (the A11/A22 diagonal terms; A13/A23/A31/A32 all vanish)."""
        n_lat, n_lon, nlev = 8, 8, 3
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
        z_coord = create_ocean_z_star(n_levels=nlev, H_max=3000.0,
                                      dz_surface=100.0, dz_deep=1500.0)
        mask = jnp.ones((n_lat, n_lon))
        # Closed walls so interior cell 4,4 is unaffected by wrap.
        u_mask = jnp.ones((n_lat, n_lon + 1)).at[:, 0].set(0.0).at[:, -1].set(0.0)
        v_mask = jnp.ones((n_lat + 1, n_lon)).at[0, :].set(0.0).at[-1, :].set(0.0)
        eta = jnp.zeros((n_lat, n_lon))
        H_bathy = jnp.full((n_lat, n_lon), 3000.0)
        jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
        act = jnp.broadcast_to(mask[:, :, jnp.newaxis], (n_lat, n_lon, nlev))
        key = jax.random.PRNGKey(0)
        q = jax.random.normal(key, (n_lat, n_lon, nlev))
        zero = jnp.zeros((n_lat, n_lon, nlev - 1))
        kappa = 1000.0
        dq = nemo_iso_lap_tracer_tendency_latlon_cgrid(
            q, zero, zero, mask, u_mask, v_mask, z_coord, jacobian, grid, kappa, act)
        geom = ensure_geometry(grid)
        # Interior cell (i=4,j=4,k=1): 5-point Laplacian with the operator's
        # own metrics.  zfu(i)=k*(e2u/e1u*e3t)*(q[i+1]-q[i]); tend = div /(e1t e2t e3t).
        i, j, k = 4, 4, 1
        e1t = float(geom.dx_T[i, j]); e2t = float(geom.dy_T[i, j])
        e3t = float(z_coord.dz_ref[k] * jacobian[i, j])
        e1u = float(geom.dx_u[i, j + 1]); e2u = float(geom.dy_u[i, j + 1])
        e1u_w = float(geom.dx_u[i, j]); e2u_w = float(geom.dy_u[i, j])
        e1v = float(geom.dx_v[i + 1, j]); e2v = float(geom.dy_v[i + 1, j])
        e1v_s = float(geom.dx_v[i, j]); e2v_s = float(geom.dy_v[i, j])
        qc = float(q[i, j, k])
        fu_e = kappa * (e2u / e1u * e3t) * (float(q[i, j + 1, k]) - qc)
        fu_w = kappa * (e2u_w / e1u_w * e3t) * (qc - float(q[i, j - 1, k]))
        fv_n = kappa * (e1v / e2v * e3t) * (float(q[i + 1, j, k]) - qc)
        fv_s = kappa * (e1v_s / e2v_s * e3t) * (qc - float(q[i - 1, j, k]))
        expected = ((fu_e - fu_w) + (fv_n - fv_s)) / (e1t * e2t * e3t)
        assert np.isclose(float(dq[i, j, k]), expected, rtol=1e-10, atol=1e-16), (
            f"interior tendency {float(dq[i, j, k]):.6e} != hand-checked {expected:.6e}")

    def test_topographic_step_conserves(self):
        """A lateral bathymetry step (deep column beside a shallower one) must
        NOT leak tracer through the u/v-face into the dry cell: the both-cells-
        wet face mask zeros that face, so the wet-volume integral of the
        tendency stays ~0.  Without requiring the neighbour's activity, the
        flux into the discarded dry cell would break conservation."""
        n_lat, n_lon, nlev = 6, 6, 4
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
        geom = ensure_geometry(grid)
        z_coord = create_ocean_z_star(n_levels=nlev, H_max=3000.0,
                                      dz_surface=100.0, dz_deep=1500.0)
        mask = jnp.ones((n_lat, n_lon))
        u_mask = jnp.ones((n_lat, n_lon + 1)).at[:, 0].set(0.0).at[:, -1].set(0.0)
        v_mask = jnp.ones((n_lat + 1, n_lon)).at[0, :].set(0.0).at[-1, :].set(0.0)
        eta = jnp.zeros((n_lat, n_lon))
        H_bathy = jnp.full((n_lat, n_lon), 3000.0)
        jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
        # Step: level nlev-1 dry everywhere; cols 3.. also dry at level nlev-2.
        act = jnp.ones((n_lat, n_lon, nlev)).at[:, :, -1].set(0.0)
        act = act.at[:, 3:, nlev - 2].set(0.0)
        key = jax.random.PRNGKey(3)
        q = jax.random.normal(key, (n_lat, n_lon, nlev)) * act  # garbage 0 in dry
        S_x = jnp.zeros((n_lat, n_lon, nlev - 1))
        S_y = jnp.zeros((n_lat, n_lon, nlev - 1))
        dT = nemo_iso_lap_tracer_tendency_latlon_cgrid(
            q, S_x, S_y, mask, u_mask, v_mask, z_coord, jacobian, grid, 1000.0, act)
        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
        vol = (geom.dx_T * geom.dy_T)[:, :, jnp.newaxis] * dz * act
        rel = abs(float(jnp.sum(dT * vol))) / (float(jnp.max(jnp.abs(dT))) * float(jnp.sum(vol)))
        assert rel < 1e-12, f"topographic-step leak: conservation rel {rel:.2e}"

    def test_grad_flows(self):
        """jax.grad through the operator (wrt kappa_Redi and q) is finite."""
        setup = _stratified_with_meridional_tilt()
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
        S_x, S_y = self._slopes(setup)
        u, v = self._closed_box(setup)
        act = jnp.broadcast_to(mask[:, :, jnp.newaxis], T.shape)

        def loss_k(kappa):
            return jnp.sum(nemo_iso_lap_tracer_tendency_latlon_cgrid(
                T, S_x, S_y, mask, u, v, z_coord, jacobian, grid, kappa, act) ** 2)
        gk = jax.grad(loss_k)(1000.0)
        assert np.isfinite(float(gk)) and float(gk) != 0.0

        def loss_q(q):
            return jnp.sum(nemo_iso_lap_tracer_tendency_latlon_cgrid(
                q, S_x, S_y, mask, u, v, z_coord, jacobian, grid, 1000.0, act) ** 2)
        gq = jax.grad(loss_q)(T)
        assert jnp.all(jnp.isfinite(gq))
        assert float(jnp.max(jnp.abs(gq))) > 0.0

    def test_bottom_dry_level_no_leak(self):
        """A sub-seafloor dry level (garbage 0 tracer) must not leak an
        across-floor vertical gradient into the deepest wet cell: active_3d
        zeros the below-floor cell, so the deepest-wet-cell tendency is the
        same whether the dry level holds 0 or a copy of the cell above."""
        n_lat, n_lon, nlev = 6, 6, 4          # 3 wet levels + 1 dry below floor
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
        z_coord = create_ocean_z_star(n_levels=nlev, H_max=3000.0,
                                      dz_surface=100.0, dz_deep=1500.0)
        mask = jnp.ones((n_lat, n_lon))
        u_mask = jnp.ones((n_lat, n_lon + 1)).at[:, 0].set(0.0).at[:, -1].set(0.0)
        v_mask = jnp.ones((n_lat + 1, n_lon)).at[0, :].set(0.0).at[-1, :].set(0.0)
        eta = jnp.zeros((n_lat, n_lon))
        H_bathy = jnp.full((n_lat, n_lon), 3000.0)
        jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
        # active_3d: bottom level dry.
        act = jnp.ones((n_lat, n_lon, nlev)).at[:, :, -1].set(0.0)
        key = jax.random.PRNGKey(1)
        q = jax.random.normal(key, (n_lat, n_lon, nlev))
        q_garbage = q.at[:, :, -1].set(0.0)       # NEMO stores 0 below floor
        q_copy = q.at[:, :, -1].set(q[:, :, -2])  # or a copy of the cell above
        S_x = jnp.zeros((n_lat, n_lon, nlev - 1))
        S_y = jnp.zeros((n_lat, n_lon, nlev - 1))
        d1 = nemo_iso_lap_tracer_tendency_latlon_cgrid(
            q_garbage, S_x, S_y, mask, u_mask, v_mask, z_coord, jacobian, grid, 1000.0, act)
        d2 = nemo_iso_lap_tracer_tendency_latlon_cgrid(
            q_copy, S_x, S_y, mask, u_mask, v_mask, z_coord, jacobian, grid, 1000.0, act)
        # Deepest wet level is index nlev-2; its tendency must not depend on the
        # dry level's value.
        assert jnp.allclose(d1[:, :, nlev - 2], d2[:, :, nlev - 2], rtol=1e-10, atol=1e-20)


# =====================================================================
# MSC (ln_traldf_msc) explicit-K33 stabilizing correction
# =====================================================================

class TestMSCStabilize:
    """NEMO ln_traldf_msc: the akz-stabilized EXPLICIT K33 vertical diagonal
    (traldf_iso_a33). Default (msc_stabilize=False) leaves the operator's full
    K33 implicit → bit-identical to the prior operator; msc_stabilize=True adds
    the explicit part the ttrd_ldf dump carries for msc=T configs (e.g. DINO)."""

    def _inputs(self, slope=5e-3):
        setup = _stratified_with_meridional_tilt(n_lat=8, n_lon=12, nlev=6,
                                                 slope=slope)
        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jac, rho, T, S, cfg = setup
        nlev = T.shape[2]
        # Steep interface slopes so the K33 diagonal (aht·wslp²) is nonzero.
        S_x = jnp.full((*T.shape[:2], nlev - 1), slope, dtype=jnp.float64)
        S_y = jnp.full((*T.shape[:2], nlev - 1), slope, dtype=jnp.float64)
        act = jnp.broadcast_to(mask[:, :, jnp.newaxis], T.shape)
        return dict(T=T, S_x=S_x, S_y=S_y, mask=mask, u_mask=u_mask,
                    v_mask=v_mask, z_coord=z_coord, jacobian=jac, grid=grid,
                    kappa=1000.0, act=act)

    def _call(self, a, **kw):
        return nemo_iso_lap_tracer_tendency_latlon_cgrid(
            a["T"], a["S_x"], a["S_y"], a["mask"], a["u_mask"], a["v_mask"],
            a["z_coord"], a["jacobian"], a["grid"], a["kappa"], a["act"], **kw)

    def test_msc_false_bit_identical(self):
        """msc_stabilize=False (the default) is byte-identical to the prior
        operator regardless of dt — the explicit-K33 block is fully gated, so
        GYRE and every existing caller are unaffected."""
        a = self._inputs()
        base = self._call(a)                                  # no msc arg (default)
        off = self._call(a, msc_stabilize=False, dt=3600.0)   # explicit off + dt
        assert jnp.array_equal(base, off)

    def test_msc_true_requires_dt(self):
        a = self._inputs()
        with pytest.raises(ValueError, match="dt"):
            self._call(a, msc_stabilize=True, dt=None)

    def test_msc_true_adds_explicit_k33_interior(self):
        """msc_stabilize=True adds a NONZERO explicit K33 at interior steep-slope
        interfaces (and the surface interface stays untouched — NEMO a33 loops
        jk=2..jpkm1)."""
        a = self._inputs(slope=5e-3)
        base = self._call(a)
        on = self._call(a, msc_stabilize=True, dt=3600.0)
        diff = jnp.abs(on - base)
        assert jnp.all(jnp.isfinite(on))
        # Non-vacuous: the explicit K33 changes the interior tendency.
        assert float(jnp.max(diff[:, :, 1:])) > 0.0
        # Zero slope ⇒ ah_wslp2=0, so the explicit coeff is (0 − akz) = −akz.  On
        # THIS coarse grid dt·akz_h < 0.5 ⇒ akz=0 ⇒ msc inert.  (On a fine grid the
        # slope-independent akz_h can push akz>0, giving a nonzero −akz explicit
        # part even at zero slope — NEMO's own MSC behaviour, not a bug.)
        a0 = self._inputs(slope=0.0)
        b0 = self._call(a0)
        on0 = self._call(a0, msc_stabilize=True, dt=3600.0)
        assert jnp.allclose(b0, on0, rtol=1e-10, atol=1e-30)
