"""Slab-land interface flux-law unification (``land_interface_flux``).

Defect (measured on the compiled lat-lon AMIP lane): ``_step_slab_land``
advanced the land skin with its OWN constant-C_H/C_E no-stability bulk fluxes
while the atmosphere debited sensible/latent through the turbulence scheme's
surface layer (``holtslag_boville_turbulence`` -> ``surface_fluxes_at_lowest_level``
with ``config.surface``, coare3 + stability) — two different flux laws at one
interface, same-state mismatch +75..+152 W/m^2 (a spurious skin heat source).

``land_interface_flux="unified"`` makes the slab consume THE SAME law.  The
lane-level test below runs the RUNNING SYMBOLS of the compiled AMIP lane:
``PhysicsPipeline.build_step_unified()``'s ``step_unified`` (the function
``compiled_segments`` executes), whose ``_rad_branch`` calls
``compute_radiation_core`` -> ``_step_slab_land`` (the slab-side energy
credit) and then ``physics_step_no_rad`` -> ``holtslag_boville_turbulence``
-> ``surface_fluxes_at_lowest_level`` (the atmosphere-side debit) — both evaluated at
the SAME pre-step state (the slab is advanced from the old ``T_land`` and the
atmosphere fluxes are computed with ``T_land=T_land`` old, physics_pipeline
``_rad_branch``).

Run with ``JAX_ENABLE_X64=1`` (numerics/conservation test).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.thermo import saturation_specific_humidity
from legoesm.atmosphere.physics._shared import compute_heights_from_sigma
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    surface_fluxes_at_lowest_level,
)

NLEV = 6
N_CS = 4
SHAPE_2D = (6, N_CS, N_CS)
SHAPE_3D = (*SHAPE_2D, NLEV)


def _sigma(nlev=NLEV):
    class _S:
        sigma_full = jnp.linspace(0.1, 0.95, nlev)
        sigma_half = jnp.linspace(0.05, 1.0, nlev + 1)
        dsigma = jnp.diff(jnp.linspace(0.05, 1.0, nlev + 1))

        def pressure_at_full(self, p_s):
            return p_s[..., None] * self.sigma_full

        def pressure_at_half(self, p_s):
            return p_s[..., None] * self.sigma_half

        def layer_thickness_dp(self, p_s):
            return p_s[..., None] * self.dsigma
    return _S()


def _pipeline(**cfg_overrides):
    """All-land pipeline on the zero-radiation stub (sw_down = lw_down = 0
    exactly, so the slab SEB terms are known without trusting a solver)."""
    grid = create_cubed_sphere(N_CS)
    kwargs = dict(
        radiation="none", microphysics="none", convection="none",
        turbulence="holtslag_boville", surface_bulk_scheme="coare3",
        diurnal_cycle=False, slab_land_active=True, topography="etopo",
    )
    kwargs.update(cfg_overrides)
    cfg = ExperimentConfig(**kwargs)
    cfg.validate_strict()
    pipe = build_physics_pipeline(grid, _sigma(), cfg)
    pipe.f_land = jnp.ones(SHAPE_2D)          # f_land == 1 everywhere
    pipe.albedo_land = jnp.full(SHAPE_2D, 0.20)
    pipe.rad_update_steps = 1
    pipe.slab_land_active = True
    return pipe


def _state(T_land_val=278.0, T_air_val=285.0, q_air_val=0.004):
    """Synthetic STABLE column (land colder than air) over full land."""
    T = jnp.full(SHAPE_3D, T_air_val)
    p_s = jnp.full(SHAPE_2D, 1.0e5)
    q_v = jnp.full(SHAPE_3D, q_air_val)
    u = jnp.full(SHAPE_3D, 5.0)
    v = jnp.zeros(SHAPE_3D)
    sst = jnp.full(SHAPE_2D, 290.0)   # inert at f_land == 1
    sic = jnp.zeros(SHAPE_2D)
    lat = jnp.full(SHAPE_2D, 0.7)
    lon = jnp.full(SHAPE_2D, 1.0)
    T_land = jnp.full(SHAPE_2D, T_land_val)
    return T, p_s, q_v, u, v, sst, sic, lat, lon, T_land


def _law_fluxes(pipe, T, p_s, q_v, u, v, T_land, sst, sic):
    """The atmosphere-side surface-flux law, from the PUBLIC API with the
    pipeline's own SurfaceLayerConfig — the reference both sides must match.

    Mirrors physics_step_no_rad's non-tiled HB inputs at ANY f_land: the
    BLENDED T_sfc, q_sfc = q_sat(T_sfc), rho = p_full_low/(R_d*T_low), and
    the lowest-level height the pipeline hands the surface layer (dry
    hydrostatic heights from the same T and p_half), so the reference applies
    the same reference-height correction production does (#1783).
    """
    from legoesm.forcing.surface_utils import blend_surface_temperature
    T_air = T[..., -1]
    q_air = q_v[..., -1]
    rho_low = (p_s * pipe.sigma_full[-1]) / (constants.R_d * T_air)
    T_sfc = pipe._blend_land(
        blend_surface_temperature(sst, sic, pipe.T_ice), T_land)
    q_sfc = saturation_specific_humidity(T_sfc, p_s)
    _, _, sh, lh, _ = surface_fluxes_at_lowest_level(
        u[..., -1], v[..., -1], T_air, q_air, T_sfc, q_sfc, rho_low,
        pipe.turbulence_config.surface, _z_low(pipe, T, p_s),
    )
    return sh, lh


def _z_low(pipe, T, p_s):
    """Lowest full level's height above the surface [m], built as the
    pipeline builds it (dry hydrostatic heights from T and p_half)."""
    ad = pipe.adapter
    z_full, z_half = compute_heights_from_sigma(
        ad.flatten_3d(T), ad.flatten_3d(pipe.sigma_coord.pressure_at_half(p_s)))
    z_low = ad.unflatten_2d(z_full[:, -1] - z_half[:, -1])
    assert bool(jnp.all(jnp.isfinite(z_low) & (z_low > 0.0)))
    return z_low


# ---------------------------------------------------------------------------
# Lane-level: one flux law at the interface (unified)
# ---------------------------------------------------------------------------

class TestUnifiedLaneOneFluxLaw:
    def test_both_sides_on_one_law_through_step_unified(self):
        """One jitted ``step_unified`` (static_need_rad=True) call yields BOTH
        sides at the same state: PhysicsOutput.shflx/lhflx (the flux the HB
        bottom boundary actually credits the column) and T_land_new (the slab
        debit).  Both must sit on the SAME law: (i) the atmosphere flux
        equals the reference law fluxes; (ii) the slab energy change obeys
        C_land*dT = dt_rad*(F + F'*dT) with F from that SAME law.  This pins
        the LAW, not exact conservation — the discrete residual is measured
        in test_window_integrated_residual_beats_legacy."""
        pipe = _pipeline(land_interface_flux="unified")
        T, p_s, q_v, u, v, sst, sic, lat, lon, T_land = _state()
        step_fn = pipe.build_step_unified(static_need_rad=True)
        ad = pipe.adapter
        held_3d = jnp.zeros(SHAPE_3D)
        held_2d = jnp.zeros(SHAPE_2D)
        o3 = jnp.zeros((ad.ncol, NLEV))
        aerosol = jnp.zeros((ad.ncol, NLEV))
        dt = 600.0

        phys_out, _held, T_land_new, _ml = step_fn(
            jnp.bool_(True),
            T, p_s, q_v, jnp.zeros(SHAPE_3D), jnp.zeros(SHAPE_3D),
            jnp.zeros((ad.ncol,)), u, v, sst, sic, lat, lon,
            100.0, 43200.0, dt,
            jnp.array([]), constants.S_0, o3, aerosol,
            held_3d, held_2d, held_2d, held_2d, held_2d, held_2d,
            T_land=T_land,
        )

        sh_ref, lh_ref = _law_fluxes(pipe, T, p_s, q_v, u, v, T_land, sst, sic)

        # (i) ATMOSPHERE DEBIT == LAW.  PhysicsOutput.shflx/lhflx are the HB
        # kernel's surface fluxes — the values its implicit diffusion bottom
        # boundary actually applies to the column (storage-dtype cast).
        assert jnp.allclose(phys_out.shflx, sh_ref, rtol=1e-4, atol=1e-2), (
            "HB surface SH != reference law SH — the atmosphere side is not "
            "on the law the test assumes")
        assert jnp.allclose(phys_out.lhflx, lh_ref, rtol=1e-4, atol=1e-2)

        # (ii) SLAB CREDIT == LAW (semi-implicit energy identity).  With the
        # zero-radiation stub: sw_down = lw_down = 0, so
        #   F  = -eps*sb*T^4 - SH - LE          (flux INTO the skin, +down)
        #   F' = -4*eps*sb*T^3 - max(d(SH+LE)/dT, 0)
        #   C_land*(T_new - T) = dt_rad*(F + F'*(T_new - T))
        eps = pipe.emissivity_land
        sb = constants.sigma_sb

        def _turb_total(T_l):
            sh, lh = _law_fluxes(pipe, T, p_s, q_v, u, v, T_l, sst, sic)
            return sh + lh

        turb, d_turb = jax.jvp(
            _turb_total, (T_land,), (jnp.ones_like(T_land),))
        F = -eps * sb * T_land ** 4 - turb
        Fp = -4.0 * eps * sb * T_land ** 3 - jnp.maximum(d_turb, 0.0)
        dT = T_land_new.astype(jnp.float64) - T_land
        lhs = pipe.C_land * dT / (dt * pipe.rad_update_steps)
        rhs = F + Fp * dT
        # T_land_new is storage-dtype (f32) cast: |dT| ~ 0.3 K resolves to
        # ~2e-5 K -> ~0.007 W/m^2 on lhs; 0.05 W/m^2 is well above that noise
        # and far below the >75 W/m^2 defect this guards against.
        assert jnp.allclose(lhs, rhs, atol=0.05), (
            "slab energy change does not equal the applied unified-law flux")

        # Teeth: on this same state the LEGACY slab law differs from the
        # unified law by a physically large margin — the mismatch this fix
        # removes is real on this state, so the assertions above are not
        # vacuously satisfied by any flux law.
        rho_leg, wind_leg = pipe._land_surface_bulk(
            T[..., -1], u[..., -1], v[..., -1], p_s)
        sh_leg = rho_leg * constants.c_pd * pipe.C_H * wind_leg * (
            T_land - T[..., -1])
        lh_leg = rho_leg * constants.L_v * pipe.C_E * wind_leg * (
            saturation_specific_humidity(T_land, p_s) - q_v[..., -1])
        mismatch = jnp.abs((sh_leg + lh_leg) - (sh_ref + lh_ref))
        assert float(jnp.min(mismatch)) > 5.0, (
            "legacy and unified laws agree on this state — pick a state "
            "where the defect is visible or the test proves nothing")

    def test_tight_identity_through_compute_radiation_core_x64(self):
        """The same semi-implicit energy identity at full input precision via
        ``compute_radiation_core`` (the exact symbol ``_rad_branch`` calls),
        before the step's storage-dtype cast."""
        if not jax.config.read("jax_enable_x64"):
            pytest.skip("x64 required for the tight fp identity")
        pipe = _pipeline(land_interface_flux="unified")
        T, p_s, q_v, u, v, sst, sic, lat, lon, T_land = _state()
        dt = 600.0
        out = pipe.compute_radiation_core(
            T, p_s, q_v, sst, sic, lat, lon, 100.0, 43200.0,
            jnp.array([]), constants.S_0, None, None,
            u=u, v=v, dt=dt, T_land=T_land,
        )
        T_land_new = out[6]
        eps = pipe.emissivity_land
        sb = constants.sigma_sb

        def _turb_total(T_l):
            sh, lh = _law_fluxes(pipe, T, p_s, q_v, u, v, T_l, sst, sic)
            return sh + lh

        turb, d_turb = jax.jvp(
            _turb_total, (T_land,), (jnp.ones_like(T_land),))
        F = -eps * sb * T_land ** 4 - turb
        Fp = -4.0 * eps * sb * T_land ** 3 - jnp.maximum(d_turb, 0.0)
        dT = T_land_new - T_land
        residual = pipe.C_land * dT - dt * (F + Fp * dT)
        assert float(jnp.max(jnp.abs(residual))) < 1e-6, (
            f"fp64 energy-identity residual {float(jnp.max(jnp.abs(residual)))}"
        )

    def test_fractional_cells_share_one_law(self):
        """Codex R1 blocker regression: on MIXED f_land cells the unified
        non-tiled law is evaluated on the BLENDED surface — exactly what the
        one-tile atmosphere applies — so the per-unit-area LAW matches at
        every land fraction, not only f_land == 1.  Both sides are taken
        from the RUNNING lane symbols (codex R2: the atmosphere side must be
        the actual ``physics_step_no_rad`` output, not a reconstruction)."""
        if not jax.config.read("jax_enable_x64"):
            pytest.skip("x64 required for the tight fp identity")
        pipe = _pipeline(land_interface_flux="unified")
        # Heterogeneous mask: 0, 0.3, 0.7, 1 tiled across the grid.
        fracs = jnp.array([0.0, 0.3, 0.7, 1.0])
        pipe.f_land = jnp.tile(fracs, SHAPE_2D[:-1] + (1,))[..., :N_CS]
        T, p_s, q_v, u, v, sst, sic, lat, lon, T_land = _state()
        dt = 600.0

        sh_ref, lh_ref = _law_fluxes(pipe, T, p_s, q_v, u, v, T_land, sst, sic)

        # ATMOSPHERE DEBIT at fractional f_land — the actual lane symbol.
        zero2 = jnp.zeros(SHAPE_2D)
        phys_out = pipe.physics_step_no_rad(
            T, p_s, q_v, jnp.zeros(SHAPE_3D), jnp.zeros(SHAPE_3D),
            jnp.zeros((pipe.adapter.ncol,)), u, v, sst, sic, lat, dt,
            jnp.zeros(SHAPE_3D), zero2, zero2, zero2, zero2, zero2,
            T_land=T_land,
        )
        assert jnp.allclose(phys_out.shflx, sh_ref, rtol=1e-6, atol=1e-6), (
            "physics_step_no_rad SH at fractional f_land != blended law")
        assert jnp.allclose(phys_out.lhflx, lh_ref, rtol=1e-6, atol=1e-6)

        # SLAB CREDIT — the semi-implicit identity with the SAME law.
        out = pipe.compute_radiation_core(
            T, p_s, q_v, sst, sic, lat, lon, 100.0, 43200.0,
            jnp.array([]), constants.S_0, None, None,
            u=u, v=v, dt=dt, T_land=T_land,
        )
        T_land_new = out[6]
        eps = pipe.emissivity_land
        sb = constants.sigma_sb

        def _turb_total(T_l):
            sh, lh = _law_fluxes(pipe, T, p_s, q_v, u, v, T_l, sst, sic)
            return sh + lh

        turb, d_turb = jax.jvp(
            _turb_total, (T_land,), (jnp.ones_like(T_land),))
        F = -eps * sb * T_land ** 4 - turb
        Fp = -4.0 * eps * sb * T_land ** 3 - jnp.maximum(d_turb, 0.0)
        dT = T_land_new - T_land
        residual = pipe.C_land * dT - dt * (F + Fp * dT)
        assert float(jnp.max(jnp.abs(residual))) < 1e-6, (
            "unified slab does not debit the blended-surface law on "
            "fractional cells")

    def test_subcycled_slab_debits_held_flux_times_window(self):
        """Radiation subcycling (rad_update_steps=N) pins the DOCUMENTED
        time discretization: the slab debits the unified-law flux evaluated
        at the refresh state, held over the whole N*dt window (the same
        held-forcing semantics as the radiation fluxes; the atmosphere's
        within-window re-evaluation against the held T_land is a
        time-truncation residual, not a flux-law mismatch — codex R3)."""
        if not jax.config.read("jax_enable_x64"):
            pytest.skip("x64 required for the tight fp identity")
        pipe = _pipeline(land_interface_flux="unified")
        pipe.rad_update_steps = 2
        T, p_s, q_v, u, v, sst, sic, lat, lon, T_land = _state()
        dt = 600.0
        out = pipe.compute_radiation_core(
            T, p_s, q_v, sst, sic, lat, lon, 100.0, 43200.0,
            jnp.array([]), constants.S_0, None, None,
            u=u, v=v, dt=dt, T_land=T_land,
        )
        T_land_new = out[6]
        eps = pipe.emissivity_land
        sb = constants.sigma_sb

        def _turb_total(T_l):
            sh, lh = _law_fluxes(pipe, T, p_s, q_v, u, v, T_l, sst, sic)
            return sh + lh

        turb, d_turb = jax.jvp(
            _turb_total, (T_land,), (jnp.ones_like(T_land),))
        F = -eps * sb * T_land ** 4 - turb
        Fp = -4.0 * eps * sb * T_land ** 3 - jnp.maximum(d_turb, 0.0)
        dT = T_land_new - T_land
        dt_window = dt * 2                       # the held window, not dt
        residual = pipe.C_land * dT - dt_window * (F + Fp * dT)
        assert float(jnp.max(jnp.abs(residual))) < 1e-6

    def test_unified_matches_tiled_land_law_when_surface_tiled(self):
        """With surface_tiled the unified slab must sit on the LAND TILE law
        (fixed-roughness MOST) — checked against ``_tiled_surface_flux``
        itself at f_land == 1 (frac_land = 1 blend == the land tile)."""
        pipe = _pipeline(turbulence="louis", land_interface_flux="unified",
                         surface_tiled=True, slab_land_active=True)
        pipe.surface_tiled = True
        T, p_s, q_v, u, v, sst, sic, _lat, _lon, T_land = _state()
        ad = pipe.adapter
        T_air = T[..., -1]
        q_air = q_v[..., -1]
        rho_low = (p_s * pipe.sigma_full[-1]) / (constants.R_d * T_air)
        f2 = ad.flatten_2d
        z_low = _z_low(pipe, T, p_s)
        _, _, sh_tile, lh_tile, _ = pipe._tiled_surface_flux(
            f2(u[..., -1]), f2(v[..., -1]), f2(T_air), f2(q_air), f2(rho_low),
            sst, sic, T_land, p_s, z_low=f2(z_low),
        )
        sh_uni, lh_uni, _d = pipe._unified_land_fluxes(
            T_land, T_air, q_air, u[..., -1], v[..., -1], p_s, z_low=z_low)
        assert jnp.allclose(f2(sh_uni), sh_tile, rtol=1e-6, atol=1e-8)
        assert jnp.allclose(f2(lh_uni), lh_tile, rtol=1e-6, atol=1e-8)
        # Teeth: the height correction is live on this state, so a call site
        # that dropped z_low would not match the tile law above.
        _, _, sh_bare, _, _ = pipe._tiled_surface_flux(
            f2(u[..., -1]), f2(v[..., -1]), f2(T_air), f2(q_air), f2(rho_low),
            sst, sic, T_land, p_s,
        )
        assert float(jnp.max(jnp.abs(sh_bare - sh_tile))) > 1.0


    def test_window_integrated_residual_beats_legacy(self):
        """THE conservation measurement (codex R6): integrate the ACTUAL
        atmosphere-side turbulent flux (``PhysicsOutput.shflx+lhflx`` from
        the running ``step_unified``) over a FULL radiation window —
        substep 0 with need_rad=True at the old skin, substeps 1..N-1 with
        need_rad=False at the held new skin — and difference it against the
        turbulent energy the slab actually applied over that same window.

        This is the real land-air exchange residual, not a slab-algebra
        identity.  It is NOT zero (the update is semi-implicit and the
        atmosphere re-evaluates against the updated skin on the no-rad
        substeps — both time-discretization terms).  The claim under test is
        the CONTROLLED one: unified beats legacy_dual by a large factor on
        the SAME state, same window, same everything else.  Measured on this
        C4 stable column with the lowest-level height correction on (#1783):
        legacy 116.8 -> unified 1.02 W/m^2 at N=1 (115x), 251.3 -> 3.75 W/m^2
        at N=24 (4 h, 67x).  The bounds below are about half the measured
        ratios; with the correction pinned off on the unified lane the N=24
        ratio is 7x, so restoring that pin fails here."""
        if not jax.config.read("jax_enable_x64"):
            pytest.skip("x64 required for the flux integration")
        dt = 600.0

        def _residual(flux_mode, n_rad):
            pipe = _pipeline(land_interface_flux=flux_mode)
            pipe.rad_update_steps = n_rad
            T, p_s, q_v, u, v, sst, sic, lat, lon, T_land = _state()
            ad = pipe.adapter
            h3 = jnp.zeros(SHAPE_3D)
            h2 = jnp.zeros(SHAPE_2D)
            base = (T, p_s, q_v, jnp.zeros(SHAPE_3D), jnp.zeros(SHAPE_3D),
                    jnp.zeros((ad.ncol,)), u, v, sst, sic, lat, lon,
                    100.0, 43200.0, dt, jnp.array([]), constants.S_0,
                    jnp.zeros((ad.ncol, NLEV)), jnp.zeros((ad.ncol, NLEV)),
                    h3, h2, h2, h2, h2, h2)
            rad = pipe.build_step_unified(static_need_rad=True, jit=False)
            no_rad = pipe.build_step_unified(static_need_rad=False, jit=False)
            po, _h, T_new, _ = rad(True, *base, T_land=T_land)
            credit = (po.shflx + po.lhflx).astype(jnp.float64) * dt
            for _ in range(n_rad - 1):
                po2, _h2, _tl, _ = no_rad(False, *base, T_land=T_new)
                credit = credit + (
                    po2.shflx + po2.lhflx).astype(jnp.float64) * dt
            dT = T_new.astype(jnp.float64) - T_land
            T_air = T[..., -1]
            q_air = q_v[..., -1]
            if flux_mode == "unified":
                def _turb(T_l):
                    sh, lh = _law_fluxes(
                        pipe, T, p_s, q_v, u, v, T_l, sst, sic)
                    return sh + lh
                F0, dF = jax.jvp(
                    _turb, (T_land,), (jnp.ones_like(T_land),))
                debit = (F0 + jnp.maximum(dF, 0.0) * dT) * (dt * n_rad)
            else:
                rho_l, ws = pipe._land_surface_bulk(
                    T_air, u[..., -1], v[..., -1], p_s)
                sh = rho_l * constants.c_pd * pipe.C_H * ws * (T_land - T_air)
                q_sat = saturation_specific_humidity(T_land, p_s)
                lh = rho_l * constants.L_v * pipe.C_E * ws * (q_sat - q_air)
                dqs = q_sat * constants.L_v / (constants.R_v * T_land ** 2)
                dS = (rho_l * constants.c_pd * pipe.C_H * ws
                      + rho_l * constants.L_v * pipe.C_E * ws * dqs)
                debit = (sh + lh + dS * dT) * (dt * n_rad)
            return float(
                jnp.max(jnp.abs(debit - credit))) / (dt * n_rad)

        for n_rad, min_factor in ((1, 50.0), (24, 30.0)):
            leg = _residual("legacy_dual", n_rad)
            uni = _residual("unified", n_rad)
            assert uni < leg / min_factor, (
                f"N={n_rad}: unified window-mean residual {uni:.2f} W/m2 is "
                f"not >{min_factor}x better than legacy {leg:.2f} W/m2")
            # Not claimed to be zero — only much smaller.
            assert uni > 0.0

    def test_stomatal_beta_par_input_mirrored_tiled_fractional(self):
        """Codex R4 (HIGH): tiled + bucket + stomatal beta at FRACTIONAL
        f_land — the slab's Jarvis beta must consume the RECONSTRUCTED PAR
        (sw_net/(1-albedo_land)) that the atmosphere-side beta uses
        (physics_step_no_rad), not the true sw_down.  Discriminating: the
        semi-implicit identity holds with the reconstructed-PAR beta and
        fails with the true-sw_down beta."""
        if not jax.config.read("jax_enable_x64"):
            pytest.skip("x64 required for the tight fp identity")
        from legoesm.atmosphere.physics.radiation.output import RadiationOutput
        from legoesm.land.stomata import StomataConfig
        pipe = _pipeline(turbulence="louis", land_interface_flux="unified",
                         surface_tiled=True)
        pipe.surface_tiled = True
        pipe.land_soil_bucket = True
        pipe.land_stomatal_beta = True
        pipe.stomata_config = StomataConfig()
        fracs = jnp.array([0.25, 0.5, 0.75, 1.0])
        pipe.f_land = jnp.tile(fracs, SHAPE_2D[:-1] + (1,))[..., :N_CS]

        # Constant-flux radiation stub: KNOWN sw/lw down at the surface, in
        # the LOW-light regime where the Jarvis light response is steep (a
        # saturated light term would make both PAR conventions agree and the
        # test vacuous).
        sw_down_val, lw_down_val = 60.0, 340.0

        def const_rad(T_col, *a, **kw):
            ncol, nlev = T_col.shape
            zf = jnp.zeros((ncol, nlev), dtype=T_col.dtype)
            zh = jnp.zeros((ncol, nlev + 1), dtype=T_col.dtype)
            return RadiationOutput(
                lw_flux_up=zh, lw_flux_down=zh.at[:, -1].set(lw_down_val),
                sw_flux_up=zh, sw_flux_down=zh.at[:, -1].set(sw_down_val),
                heating_rate=zf, lw_heating_rate=zf, sw_heating_rate=zf)

        pipe.radiation_fn = const_rad
        # WARM, MOIST state: the Jarvis f_T term is a parabola about 25 degC
        # with a 20 degC range (a 278 K skin zeroes gs) and f_VPD collapses
        # beta to ~1e-3 in dry air — both crush the PAR difference below
        # resolution, and the test would not discriminate.
        T, p_s, q_v, u, v, sst, sic, lat, lon, T_land = _state(
            T_land_val=295.0, T_air_val=292.0, q_air_val=0.012)
        w_land = jnp.full(SHAPE_2D, 140.0)   # near-full bucket: canopy binds
        dt = 600.0
        out = pipe.compute_radiation_core(
            T, p_s, q_v, sst, sic, lat, lon, 100.0, 43200.0,
            jnp.array([]), constants.S_0, None, None,
            u=u, v=v, dt=dt, T_land=T_land, w_land=w_land,
        )
        T_land_new = out[6]

        alb_land = pipe.albedo_land
        alb_blend = pipe.f_land * alb_land + (1.0 - pipe.f_land) * 0.06
        sw_net_blend = sw_down_val * (1.0 - alb_blend)
        q_air = q_v[..., -1]
        par_recon = sw_net_blend / jnp.maximum(1.0 - alb_land, 1e-3)
        par_true = jnp.full(SHAPE_2D, sw_down_val)
        beta_recon = pipe._land_beta(w_land, T_land=T_land,
                                     sw_down_sfc=par_recon, q_air=q_air,
                                     p_s=p_s)
        beta_true = pipe._land_beta(w_land, T_land=T_land,
                                    sw_down_sfc=par_true, q_air=q_air,
                                    p_s=p_s)
        # Discrimination precondition: the two PAR conventions give
        # materially different beta somewhere fractional.
        assert float(jnp.max(jnp.abs(beta_recon - beta_true))) > 1e-3, (
            "PAR conventions agree on this state — pick a steeper-light "
            "regime or the test proves nothing")

        T_air = T[..., -1]
        rho_low = (p_s * pipe.sigma_full[-1]) / (constants.R_d * T_air)
        eps = pipe.emissivity_land
        sb = constants.sigma_sb
        land_cfg = pipe._land_tile_surface_cfg()
        z_low = _z_low(pipe, T, p_s)

        def _residual(beta):
            def _turb(T_l):
                q_sfc = q_air + beta * (
                    saturation_specific_humidity(T_l, p_s) - q_air)
                _, _, sh, lh, _ = surface_fluxes_at_lowest_level(
                    u[..., -1], v[..., -1], T_air, q_air, T_l, q_sfc,
                    rho_low, land_cfg, z_low)
                return sh + lh
            turb, d_turb = jax.jvp(
                _turb, (T_land,), (jnp.ones_like(T_land),))
            F = (sw_down_val * (1.0 - alb_land)
                 + eps * lw_down_val - eps * sb * T_land ** 4 - turb)
            Fp = -4.0 * eps * sb * T_land ** 3 - jnp.maximum(d_turb, 0.0)
            dT = T_land_new - T_land
            return float(jnp.max(jnp.abs(
                pipe.C_land * dT - dt * (F + Fp * dT))))

        res_recon = _residual(beta_recon)
        res_true = _residual(beta_true)
        assert res_recon < 1e-6, (
            f"identity fails with the reconstructed-PAR beta: {res_recon}")
        assert res_true > max(1e3 * res_recon, 1e-3), (
            "true-sw_down beta also satisfies the identity — the test does "
            "not discriminate the PAR convention")


# ---------------------------------------------------------------------------
# Legacy default: byte-identical pin
# ---------------------------------------------------------------------------

class TestLegacyDefaultPin:
    def test_default_flag_is_legacy_dual(self):
        pipe = _pipeline()
        assert pipe.land_interface_flux == "legacy_dual"

    def test_legacy_step_bitwise_equals_legacy_formula(self):
        """The default path reproduces the pre-change slab arithmetic
        EXACTLY (jnp.array_equal — a byte-identical pin, not a tolerance)."""
        pipe = _pipeline()   # default land_interface_flux
        T, p_s, q_v, u, v, _sst, _sic, _lat, _lon, T_land = _state()
        sw_down = jnp.full(SHAPE_2D, 400.0)
        lw_down = jnp.full(SHAPE_2D, 340.0)
        dt = 600.0
        T_new = pipe._step_slab_land(
            T_land, sw_down, lw_down, T, p_s, q_v, u, v, dt)

        # The legacy formula, replicated op-for-op from the pre-change code.
        T_air = T[..., -1]
        q_air = q_v[..., -1]
        rho_low, wind_speed = pipe._land_surface_bulk(
            T_air, u[..., -1], v[..., -1], p_s)
        sh_coef = rho_low * constants.c_pd * pipe.C_H * wind_speed
        lh_coef = rho_low * constants.L_v * pipe.C_E * wind_speed
        eps = pipe.emissivity_land
        sb = constants.sigma_sb
        q_sat_land = saturation_specific_humidity(T_land, p_s)
        sw_net = sw_down * (1.0 - pipe.albedo_land)
        lw_net = eps * lw_down - eps * sb * T_land ** 4
        shflx = sh_coef * (T_land - T_air)
        lhflx = 1.0 * lh_coef * (q_sat_land - q_air)
        flux = sw_net + lw_net - shflx - lhflx
        dqsat_dT = q_sat_land * constants.L_v / (constants.R_v * T_land ** 2)
        dflux_dT = (-4.0 * eps * sb * T_land ** 3
                    - sh_coef - 1.0 * lh_coef * dqsat_dT)
        expected = T_land + dt * flux / (pipe.C_land - dt * dflux_dT)
        assert jnp.array_equal(T_new, expected)

    def test_unified_step_reverse_differentiable(self):
        """jax.grad flows through the unified slab step (jvp of the MOST
        iteration inside a reverse-mode trace) — the slab sits in the
        differentiable rollout unless rad_stop_gradient."""
        pipe = _pipeline(land_interface_flux="unified")
        T, p_s, q_v, u, v, _sst, _sic, _lat, _lon, T_land = _state()
        T_ocean = jnp.full(SHAPE_2D, 290.0)   # inert at f_land == 1

        def _loss(T_l):
            T_new = pipe._step_slab_land(
                T_l, jnp.zeros(SHAPE_2D), jnp.zeros(SHAPE_2D),
                T, p_s, q_v, u, v, dt=600.0, T_sfc_ocean=T_ocean)
            return jnp.mean(T_new)

        g = jax.grad(_loss)(T_land)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 0.0

    def test_unified_differs_from_legacy(self):
        """Sanity: the two laws produce different skin updates on the same
        state (the flag is not inert)."""
        T, p_s, q_v, u, v, _sst, _sic, _lat, _lon, T_land = _state()
        args = (T_land, jnp.zeros(SHAPE_2D), jnp.zeros(SHAPE_2D),
                T, p_s, q_v, u, v)
        leg = _pipeline()._step_slab_land(*args, dt=600.0)
        uni = _pipeline(land_interface_flux="unified")._step_slab_land(
            *args, dt=600.0, T_sfc_ocean=jnp.full(SHAPE_2D, 290.0))
        assert not jnp.allclose(leg, uni)


# ---------------------------------------------------------------------------
# Config validation + threading
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_validate_strict_rejects_unknown_option(self):
        with pytest.raises(ValueError, match="land_interface_flux"):
            ExperimentConfig(land_interface_flux="both").validate_strict()

    def test_validate_strict_rejects_unified_without_turbulence(self):
        with pytest.raises(ValueError, match="land_interface_flux"):
            ExperimentConfig(turbulence="none",
                             land_interface_flux="unified").validate_strict()

    def test_validate_strict_rejects_unified_with_multilayer_land(self):
        with pytest.raises(ValueError, match="land_interface_flux"):
            ExperimentConfig(
                turbulence="holtslag_boville",
                land_interface_flux="unified",
                use_multilayer_land=True,
                land_mask_path="x.nc",
            ).validate_strict()

    @pytest.mark.parametrize("topography", ["flat", "gaussian"])
    @pytest.mark.parametrize("slab_land_active", [False, True])
    def test_validate_strict_rejects_unified_without_active_land_tile(
            self, topography, slab_land_active):
        """Idealized terrain cannot activate the requested slab without a mask."""
        cfg = ExperimentConfig(
            turbulence="holtslag_boville", land_interface_flux="unified",
            slab_land_active=slab_land_active, topography=topography,
        )
        with pytest.raises(ValueError, match=f"topography={topography!r}"):
            cfg.validate_strict()
        cfg._replace(land_mask_path="land_mask.nc").validate_strict()

    def test_validate_strict_rejects_unified_on_mpas_and_spectral(self):
        """The MPAS/spectral lanes run combined.make_physics — the driver
        pipeline slab never steps there (codex R2: land_mask_path satisfied
        the tile gate on MPAS without tripping the slab-flag rejection)."""
        from legoesm.driver.config import DycoreConfig, GridConfig
        with pytest.raises(ValueError, match="land_interface_flux"):
            ExperimentConfig(
                grid=GridConfig(grid_type="mpas", resolution=3, nlev=5),
                dycore=DycoreConfig(discretization="mpas"),
                turbulence="holtslag_boville",
                land_interface_flux="unified",
                land_mask_path="x.nc",       # tile gate satisfied — must
            ).validate_strict()              # still reject the lane
        with pytest.raises(ValueError, match="land_interface_flux"):
            ExperimentConfig(
                dycore=DycoreConfig(discretization="spectral"),
                turbulence="holtslag_boville",
                land_interface_flux="unified",
                land_mask_path="x.nc",
            ).validate_strict()

    def test_validate_strict_rejects_unified_on_latlon_spmd(self):
        """The lat-lon operator-split SPMD lane builds a FRESH per-band
        pipeline that never receives f_land / slab_land_active, so the slab
        SEB never steps and 'unified' is silently inert (codex R1 P1)."""
        from legoesm.driver.config import GridConfig
        with pytest.raises(ValueError, match="land_interface_flux"):
            ExperimentConfig(
                grid=GridConfig(grid_type="latlon"),
                turbulence="holtslag_boville",
                land_interface_flux="unified",
                land_mask_path="x.nc",       # tile gate satisfied — must
                enable_latlon_spmd=True,     # still reject the lane
            ).validate_strict()

    def test_validate_strict_rejects_unified_with_ml_physics(self):
        """physics_parameterization='ml' computes its own surface fluxes,
        bypassing the turbulence surface layer — 'unified' would unify with
        a law the atmosphere does not apply (codex R1 finding 3)."""
        cfg = ExperimentConfig(
            turbulence="holtslag_boville",
            land_interface_flux="unified",
            slab_land_active=True,
            physics_parameterization="ml",
        )
        with pytest.raises(ValueError, match="land_interface_flux"):
            cfg.validate_strict()

    def test_build_raises_on_unknown_value_bypassing_validate(self):
        """A typo'd selector through a DIRECT builder (no validate_strict)
        must raise, never silently run the legacy law (codex R1 finding 6)."""
        grid = create_cubed_sphere(N_CS)
        cfg = ExperimentConfig(
            radiation="none", microphysics="none", convection="none",
            turbulence="holtslag_boville",
            land_interface_flux="unfied")   # typo on purpose
        with pytest.raises(ValueError, match="land_interface_flux"):
            build_physics_pipeline(grid, _sigma(), cfg)

    def test_step_slab_land_raises_on_mutated_unknown_value(self):
        """Trace-time dispatch hardening on the pipeline attribute itself."""
        pipe = _pipeline()
        pipe.land_interface_flux = "both"   # simulate a mutated pipeline
        T, p_s, q_v, u, v, _sst, _sic, _lat, _lon, T_land = _state()
        with pytest.raises(ValueError, match="land_interface_flux"):
            pipe._step_slab_land(
                T_land, jnp.zeros(SHAPE_2D), jnp.zeros(SHAPE_2D),
                T, p_s, q_v, u, v, dt=600.0)

    def test_driver_runtime_guard_unified_without_land(self):
        """A config can pass validate_strict (slab_land_active + non-flat
        topography) yet load NO land at runtime (e.g. an all-zero mask).
        The driver must then refuse 'unified' rather than silently never
        applying it (codex R3)."""
        from types import SimpleNamespace
        from legoesm.driver.model_driver import ModelDriver
        cfg = ExperimentConfig(
            radiation="none", microphysics="none", convection="none",
            turbulence="holtslag_boville",
            land_interface_flux="unified",
            slab_land_active=True, topography="etopo",
        )
        fake = SimpleNamespace(
            config=cfg, grid=create_cubed_sphere(N_CS), sigma=_sigma(),
            _f_land=None,                      # runtime loaded no land
        )
        with pytest.raises(ValueError, match="did not activate"):
            ModelDriver._create_physics(fake)

    def test_build_raises_on_bad_C_land_bypassing_validate(self):
        """Direct builders skipping validate_strict must still refuse a
        sign-flipping C_land (codex R3)."""
        grid = create_cubed_sphere(N_CS)
        cfg = ExperimentConfig(
            radiation="none", microphysics="none", convection="none",
            turbulence="holtslag_boville", C_land=-2.0e5)
        with pytest.raises(ValueError, match="C_land"):
            build_physics_pipeline(grid, _sigma(), cfg)

    def test_from_amip_config_defaults_to_legacy(self):
        """Legacy AMIP checkpoints predate the flag; the deserialization
        adapter must restart them on the legacy law with the default C_land
        (codex R1 finding 4)."""
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        cfg = ExperimentConfig.from_amip_config(AMIPExperimentConfig())
        assert cfg.land_interface_flux == "legacy_dual"
        assert cfg.C_land == 2.0e5

    def test_validate_strict_rejects_bad_C_land(self):
        for bad in (0.0, -2.0e5, 1.0e3, 1.0e9, float("nan")):
            with pytest.raises(ValueError, match="C_land"):
                ExperimentConfig(C_land=bad).validate_strict()

    def test_build_raises_unified_without_surface_layer(self):
        """Direct pipeline builders that skip validate_strict still fail at
        BUILD time (not at trace time inside the compiled step)."""
        grid = create_cubed_sphere(N_CS)
        cfg = ExperimentConfig(
            radiation="none", microphysics="none", convection="none",
            turbulence="none", land_interface_flux="unified")
        with pytest.raises(ValueError, match="land_interface_flux"):
            build_physics_pipeline(grid, _sigma(), cfg)

    def test_C_land_threads_to_pipeline(self):
        """ExperimentConfig.C_land now reaches the pipeline (it was silently
        inert — the pipeline always ran the constructor default 2e5)."""
        pipe = _pipeline(C_land=5.0e5)
        assert pipe.C_land == 5.0e5
        assert _pipeline().C_land == 2.0e5   # default unchanged


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
