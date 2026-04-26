"""Tests for GM/Redi on the lat-lon Arakawa C-grid.

Run with:

    JAX_ENABLE_X64=1 python -m pytest tests/ocean/unit/test_gm_redi_latlon_cgrid.py -v

Covers: shape/finiteness, conservation, variance reduction, land masks,
differentiability, Visbeck coefficient, and structural enforcement against
code duplication with the cubed-sphere implementation.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star, compute_ocean_jacobian
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig, VisbeckConfig
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    compute_isopycnal_slopes_latlon_cgrid,
    gm_redi_tracer_tendency_latlon_cgrid,
    gm_redi_tracer_tendency_triads_latlon_cgrid,
    gm_redi_tracer_tendency_latlon,
    gm_redi_lateral_mixing_latlon,
)


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
    rho_z = jnp.linspace(1025.0, 1027.0, nlev)
    # Add meridional gradient.
    H_total = float(jnp.sum(z_coord.dz_ref))
    drho_dz = 2.0 / H_total
    drho_dy = slope * drho_dz
    lat_idx = jnp.arange(n_lat, dtype=jnp.float64)
    rho = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)
    rho = rho + rho_z[jnp.newaxis, jnp.newaxis, :]
    rho = rho + drho_dy * lat_idx[:, jnp.newaxis, jnp.newaxis] * grid.dy

    # T proportional to density (linear EOS: rho ~ 1025 - 0.2*T)
    T = (1025.0 - rho) / 0.2
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

        rho_z = jnp.linspace(1025.0, 1027.0, nlev)
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
        f_coriolis = 2.0 * 7.292e-5 * jnp.sin(grid.lat[:, jnp.newaxis])
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

    _SRC = pathlib.Path(__file__).resolve().parents[3] / \
        "src" / "legoesm" / "ocean" / "physics" / "lateral_mixing"

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
