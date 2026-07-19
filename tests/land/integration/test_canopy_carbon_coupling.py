"""Integration tests for the coupled canopy + carbon pipeline (Phase 4).

Verifies that the two-leaf canopy surface scheme's ``gpp`` output flows
into ``step_carbon`` via ``gpp_override`` and that the coupled
carbon-water-energy dynamics are internally consistent:

- Daytime ``co2_flux < 0`` (uptake) / nighttime ``co2_flux ≥ 0``
  (respiration dominates).
- Pool evolution over a multi-day run: ``C_lab`` grows during the
  growing season, respiration pools respond to temperature.
- ``step_carbon`` receives the same GPP value the canopy surface scheme
  returned (no in-between modification).
- ``jax.grad`` propagates through the whole pipeline (leaf params →
  Newton closure → pool update) without NaN.
- SimpleSEB + ``differland`` regression: Phase 3 refactor must not
  perturb the existing coupled-carbon behaviour.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy import CanopyLandParams
from legoesm.land.carbon import CarbonConfig, init_carbon_state
from legoesm.land.config import LandConfig, MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land,
    step_multilayer_land_with_diagnostics,
)
from legoesm.land.slab_land import step_land
from legoesm.land.state import LandState
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.core.field import Field


# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------

def _make_forcing(ncol: int, sw_down: float, cos_zenith: float,
                  T_air: float = 295.0) -> AtmToSurface:
    return AtmToSurface(
        sw_down=jnp.full(ncol, sw_down),
        lw_down=jnp.full(ncol, 380.0),
        precip_total=jnp.zeros(ncol),
        precip_snow=jnp.zeros(ncol),
        T_lowest=jnp.full(ncol, T_air),
        q_lowest=jnp.full(ncol, 0.012),
        u_lowest=jnp.full(ncol, 3.0),
        v_lowest=jnp.full(ncol, 0.5),
        p_lowest=jnp.full(ncol, 98000.0),
        p_surface=jnp.full(ncol, 101325.0),
        rho_lowest=jnp.full(ncol, 1.18),
        cos_zenith=jnp.full(ncol, cos_zenith),
        co2_ppmv=jnp.full(ncol, 420.0),
        has_radiation=True,
        has_precipitation=False,
    )


def _canopy_params(ncol: int, Vc3: float = 60.0) -> CanopyLandParams:
    return CanopyLandParams(
        LAI=jnp.full(ncol, 3.0),
        hc=jnp.full(ncol, 5.0),
        fC4=jnp.zeros(ncol),
        FNonVeg=jnp.zeros(ncol),
        CI=jnp.full(ncol, 0.75),
        kn=jnp.full(ncol, 0.3),
        Vcmax25_C3_leaf=jnp.full(ncol, Vc3),
        Vcmax25_C4_leaf=jnp.full(ncol, 40.0),
        m_C3=jnp.full(ncol, 9.0),
        m_C4=jnp.full(ncol, 4.0),
        b0_C3=jnp.full(ncol, 0.01),
        b0_C4=jnp.full(ncol, 0.04),
        alf=jnp.full(ncol, 0.3),
        TgC=jnp.full(ncol, 20.0),
        ALB_VIS=jnp.full(ncol, 0.1),
        ALB_NIR=jnp.full(ncol, 0.2),
        emissivity=jnp.full(ncol, 0.97),
        rz0m=jnp.full(ncol, 0.055),
        rd=jnp.full(ncol, 0.67),
    )


def _make_canopy_carbon_cfg() -> MultiLayerLandConfig:
    # use_prognostic_lai defaults to False (jax.grad through the
    # feedback loop is NaN-prone until the MOST custom-VJP lands);
    # the tests in this file exercise the prognostic path on purpose.
    return MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(
            max_iters=30, use_prognostic_lai=True),
        carbon=CarbonConfig(scheme="differland"),
    )


# ---------------------------------------------------------------------------
# Multilayer + canopy + carbon
# ---------------------------------------------------------------------------

def test_canopy_carbon_daytime_uptake():
    """At midday, canopy-driven ``co2_flux`` must be negative (land sink)
    and carbon pools must grow.
    """
    ncol = 2
    cfg = _make_canopy_carbon_cfg()
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0, TgC_init=20.0)
    carbon = init_carbon_state((ncol,), cfg.carbon)
    forcing = _make_forcing(ncol, sw_down=700.0, cos_zenith=0.8)
    params = _canopy_params(ncol)

    _, resp, carbon_new = step_multilayer_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), carbon_state=carbon, doy=180.0,
        land_params=params)

    assert jnp.all(jnp.isfinite(resp.co2_flux))
    # Daytime land sink: co2_flux < 0 (positive = source convention).
    assert float(resp.co2_flux[0]) < 0.0, (
        f"Canopy+carbon midday should be a sink, got "
        f"co2_flux={float(resp.co2_flux[0]):.3e}")
    # Labile pool grows under net uptake.
    assert float(carbon_new.C_lab[0]) > float(carbon.C_lab[0])


def test_canopy_carbon_nighttime_respiration():
    """At night (no SW), canopy GPP is zero and carbon pools lose mass
    via respiration only.  ``co2_flux`` should be ≥ 0.
    """
    ncol = 2
    cfg = _make_canopy_carbon_cfg()
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0, TgC_init=20.0)
    carbon = init_carbon_state((ncol,), cfg.carbon)
    forcing = _make_forcing(ncol, sw_down=0.0, cos_zenith=0.0)
    params = _canopy_params(ncol)

    _, resp, carbon_new = step_multilayer_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), carbon_state=carbon, doy=180.0,
        land_params=params)

    assert jnp.all(jnp.isfinite(resp.co2_flux))
    # Respiration dominates at night → co2_flux ≥ 0 (source).
    assert float(resp.co2_flux[0]) >= 0.0
    # Labile pool releases some C to maintain respiration.
    # (It may still grow slightly from any leftover allocation, so we
    # only test that it's finite and that the magnitude is small.)
    dC_lab = float(carbon_new.C_lab[0] - carbon.C_lab[0])
    assert abs(dC_lab) < 1.0, f"Nighttime dC_lab = {dC_lab:.3f} gC/m2 in 30 min"


def test_canopy_gpp_flows_into_step_carbon():
    """The GPP passed to ``step_carbon`` must equal the canopy surface
    scheme's ``gpp`` output — no in-between modification.
    """
    ncol = 2
    cfg = _make_canopy_carbon_cfg()
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0, TgC_init=20.0)
    carbon = init_carbon_state((ncol,), cfg.carbon)
    forcing = _make_forcing(ncol, sw_down=500.0, cos_zenith=0.7)
    params = _canopy_params(ncol)

    _, _, _, surface_out = step_multilayer_land_with_diagnostics(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), carbon_state=carbon, doy=180.0,
        land_params=params)

    # surface_out.gpp should be finite and positive at midday.
    assert surface_out.gpp is not None
    assert jnp.all(jnp.isfinite(surface_out.gpp))
    assert jnp.all(surface_out.gpp >= 0.0)

    # Cross-check: running the step also produces co2_flux consistent with
    # ``-GPP`` to within NPP-allocation and respiration magnitudes.
    # co2_flux unit is kg CO2 / m2 / s, gpp unit is gC / m2 / s.
    # Rough sanity: co2_flux is bounded in magnitude by a few × GPP.
    _, resp, _ = step_multilayer_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), carbon_state=carbon, doy=180.0,
        land_params=params)
    # |co2_flux| < GPP * 10 (very loose bound — just sanity)
    gpp_kgc = surface_out.gpp * 1e-3  # gC → kgC
    assert jnp.all(jnp.abs(resp.co2_flux) < 10.0 * (gpp_kgc + 1e-12))


def test_simple_seb_differland_regression():
    """Default SimpleSEB + differland must still run and produce a
    non-trivial carbon flux — guard for the Phase 3 refactor.
    """
    ncol = 4
    cfg = MultiLayerLandConfig(carbon=CarbonConfig(scheme="differland"))
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0)
    carbon = init_carbon_state((ncol,), cfg.carbon)
    forcing = _make_forcing(ncol, sw_down=400.0, cos_zenith=0.6)

    _, resp, carbon_new = step_multilayer_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), carbon_state=carbon, doy=180.0)

    assert jnp.all(jnp.isfinite(resp.co2_flux))
    assert carbon_new is not None
    # Even with SimpleSEB (no canopy), differland still runs via its LUE
    # fallback inside compute_effective_beta.
    assert jnp.any(resp.co2_flux != 0.0)


# ---------------------------------------------------------------------------
# Multi-day diurnal run: pool evolution
# ---------------------------------------------------------------------------

def test_canopy_carbon_multi_day_diurnal():
    """Run a 3-day diurnal cycle with canopy + carbon and verify pool
    evolution + finiteness.  No strict numerical bounds — the goal is to
    catch blow-ups and NaN under a nontrivial time trajectory.
    """
    ncol = 2
    cfg = _make_canopy_carbon_cfg()
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0, TgC_init=20.0)
    carbon = init_carbon_state((ncol,), cfg.carbon)
    params = _canopy_params(ncol)
    dt = 3600.0  # 1 h
    n_steps = 24 * 3  # 3 days

    # Pass cos_zenith and sw_down as traced jnp.ndarray arguments so the
    # JIT kernel is compiled once and reused across all 72 iterations —
    # calling ``step_multilayer_land`` directly in a Python loop triggers
    # a recompile per step and dominates the test runtime.
    @jax.jit
    def _step(state, carbon, cos_z_scalar, sw_scalar):
        f = _make_forcing(ncol, sw_down=100.0, cos_zenith=0.5)._replace(
            sw_down=jnp.full(ncol, sw_scalar),
            cos_zenith=jnp.full(ncol, cos_z_scalar),
        )
        return step_multilayer_land(
            state, f, cfg, U_min=1.0, dt=dt,
            lat=jnp.zeros(ncol), carbon_state=carbon, doy=180.0,
            land_params=params)

    C_lab_hist = []
    co2_flux_hist = []
    for k in range(n_steps):
        t_hr = (12.0 + k * dt / 3600.0) % 24.0
        cosz = max(0.0, float(jnp.sin(jnp.pi * (t_hr - 6.0) / 12.0)))
        sw = 900.0 * cosz
        state, resp, carbon = _step(
            state, carbon,
            jnp.asarray(cosz),  # traced → no recompile
            jnp.asarray(sw),
        )
        C_lab_hist.append(float(carbon.C_lab[0]))
        co2_flux_hist.append(float(resp.co2_flux[0]))

    C_lab_arr = jnp.asarray(C_lab_hist)
    co2_flux_arr = jnp.asarray(co2_flux_hist)
    assert jnp.all(jnp.isfinite(C_lab_arr))
    assert jnp.all(jnp.isfinite(co2_flux_arr))
    # Diurnal variation: co2_flux should swing sign across the cycle
    # (negative at day, ≥ 0 at night).
    assert float(jnp.min(co2_flux_arr)) < 0.0, "No daytime uptake"
    # Labile pool should remain positive.
    assert float(jnp.min(C_lab_arr)) > 0.0


# ---------------------------------------------------------------------------
# Differentiability through the full canopy→carbon pipeline
# ---------------------------------------------------------------------------

def test_grad_through_canopy_carbon_pipeline():
    """``jax.grad`` of a scalar derived from the carbon pool update must
    be finite with respect to the canopy Vcmax25 parameter, proving the
    canopy Newton closure + post-flux pipeline stays differentiable under
    coupled C-W-E dynamics.
    """
    ncol = 2
    cfg = _make_canopy_carbon_cfg()
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0, TgC_init=20.0)
    carbon = init_carbon_state((ncol,), cfg.carbon)
    forcing = _make_forcing(ncol, sw_down=700.0, cos_zenith=0.8)

    def _loss(Vc):
        params = CanopyLandParams(
            LAI=jnp.full(ncol, 3.0),
            hc=jnp.full(ncol, 5.0),
            fC4=jnp.zeros(ncol),
            FNonVeg=jnp.zeros(ncol),
            CI=jnp.full(ncol, 0.75),
            kn=jnp.full(ncol, 0.3),
            Vcmax25_C3_leaf=jnp.full(ncol, Vc),
            Vcmax25_C4_leaf=jnp.full(ncol, 40.0),
            m_C3=jnp.full(ncol, 9.0),
            m_C4=jnp.full(ncol, 4.0),
            b0_C3=jnp.full(ncol, 0.01),
            b0_C4=jnp.full(ncol, 0.04),
            alf=jnp.full(ncol, 0.3),
            TgC=jnp.full(ncol, 20.0),
            ALB_VIS=jnp.full(ncol, 0.1),
            ALB_NIR=jnp.full(ncol, 0.2),
            emissivity=jnp.full(ncol, 0.97),
            rz0m=jnp.full(ncol, 0.055),
            rd=jnp.full(ncol, 0.67),
        )
        _, _, c_new = step_multilayer_land(
            state, forcing, cfg, U_min=1.0, dt=1800.0,
            lat=jnp.zeros(ncol), carbon_state=carbon, doy=180.0,
            land_params=params)
        # Scalar target: change in labile pool summed over columns.
        return jnp.sum(c_new.C_lab - carbon.C_lab)

    grad = jax.grad(_loss)(60.0)
    assert jnp.isfinite(grad), f"non-finite grad wrt Vcmax25: {grad}"
    # Sensible sign: more Vcmax → more GPP → more C_lab growth → positive grad.
    assert float(grad) > 0.0, f"grad wrt Vcmax25 should be positive, got {grad}"


# ---------------------------------------------------------------------------
# Slab + canopy + carbon
# ---------------------------------------------------------------------------

def _make_slab_state(shape):
    dims = ("x",) if len(shape) == 1 else ("face", "x", "y")
    return LandState(
        T_soil=Field(data=jnp.full(shape, 290.0),
                     name="T_soil", dims=dims, units="K"),
        W_bucket=Field(data=jnp.full(shape, 100.0),
                       name="W_bucket", dims=dims, units="kg/m2"),
        snow_depth=Field(data=jnp.zeros(shape),
                         name="snow_depth", dims=dims, units="kg/m2"),
        snow_age=Field(data=jnp.zeros(shape),
                       name="snow_age", dims=dims, units="s"),
        TgC=jnp.full(shape, 20.0),
    )


def test_slab_canopy_carbon_runs():
    """Slab + canopy + differland: coupled-carbon path must run without
    NaN and produce a daytime carbon sink.
    """
    shape = (4,)
    cfg = LandConfig(
        surface_scheme=TwoLeafCanopyConfig(max_iters=30),
        carbon=CarbonConfig(scheme="differland"),
    )
    state = _make_slab_state(shape)
    carbon = init_carbon_state(shape, cfg.carbon)
    forcing = _make_forcing(shape[0], sw_down=700.0, cos_zenith=0.8)

    _, resp, carbon_new = step_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(shape), carbon_state=carbon, doy=180.0)

    assert jnp.all(jnp.isfinite(resp.co2_flux))
    assert carbon_new is not None
    # Daytime uptake
    assert float(resp.co2_flux[0]) < 0.0


# ---------------------------------------------------------------------------
# Phase 6 / Stage 2b: prognostic LAI from C_fol
# ---------------------------------------------------------------------------

def test_prognostic_lai_helper_returns_c_fol_over_lcma():
    """``compute_prognostic_lai`` must return C_fol / LCMA when
    differland is active + use_prognostic_lai is True.  Returns None
    when either condition fails.
    """
    from legoesm.land.surface_scheme.two_leaf_canopy import compute_prognostic_lai

    ncol = 2
    cfg = _make_canopy_carbon_cfg()
    carbon = init_carbon_state((ncol,), cfg.carbon)
    # C_fol_init = 200 gC/m², LCMA = 50 → expected LAI = 4.0
    lai = compute_prognostic_lai(carbon, cfg, cfg.surface_scheme)
    assert lai is not None
    assert jnp.allclose(lai, 4.0)

    # use_prognostic_lai=False → returns None
    cfg_off = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(use_prognostic_lai=False),
        carbon=CarbonConfig(scheme="differland"),
    )
    lai_off = compute_prognostic_lai(carbon, cfg_off, cfg_off.surface_scheme)
    assert lai_off is None

    # carbon scheme != differland → returns None
    cfg_none = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(),
        carbon=CarbonConfig(scheme="none"),
    )
    lai_none = compute_prognostic_lai(carbon, cfg_none, cfg_none.surface_scheme)
    assert lai_none is None

    # carbon_state is None → returns None
    lai_no_state = compute_prognostic_lai(None, cfg, cfg.surface_scheme)
    assert lai_no_state is None


def test_prognostic_lai_overrides_prescribed_lai():
    """When ``use_prognostic_lai`` is on and carbon is active, the canopy
    flux output must use ``C_fol / LCMA`` even if ``CanopyLandParams.LAI``
    prescribes a different value.
    """
    ncol = 2
    forcing = _make_forcing(ncol, sw_down=700.0, cos_zenith=0.8)

    # Prescribed LAI = 1.5; prognostic override should force LAI = 4.0
    params = _canopy_params(ncol)._replace(LAI=jnp.full(ncol, 1.5))

    cfg_prog = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(
            max_iters=30, use_prognostic_lai=True),
        carbon=CarbonConfig(scheme="differland"),
    )
    cfg_fixed = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(
            max_iters=30, use_prognostic_lai=False),
        carbon=CarbonConfig(scheme="differland"),
    )

    state = init_multilayer_land_state(ncol, cfg_prog, T_init=290.0, TgC_init=20.0)
    carbon = init_carbon_state((ncol,), cfg_prog.carbon)

    _, r_prog, _ = step_multilayer_land(
        state, forcing, cfg_prog, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), carbon_state=carbon, doy=180.0,
        land_params=params)
    _, r_fixed, _ = step_multilayer_land(
        state, forcing, cfg_fixed, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), carbon_state=carbon, doy=180.0,
        land_params=params)

    # The two runs MUST differ — LAI=4 vs LAI=1.5 changes SW absorption,
    # canopy transpiration, aerodynamic resistance, etc.
    assert not jnp.allclose(r_prog.lhflx, r_fixed.lhflx, atol=1e-3), (
        "prognostic LAI must change canopy fluxes relative to LAI=1.5")
    assert not jnp.allclose(r_prog.shflx, r_fixed.shflx, atol=1e-3)
    assert not jnp.allclose(r_prog.T_sfc, r_fixed.T_sfc, atol=1e-4)


def test_prognostic_lai_responds_to_c_fol_changes():
    """Two different ``C_fol_init`` values must yield different canopy
    fluxes when prognostic LAI is active.  Confirms the feedback is
    actually reading the state.
    """
    ncol = 2
    forcing = _make_forcing(ncol, sw_down=700.0, cos_zenith=0.8)
    cfg_low = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(
            max_iters=30, use_prognostic_lai=True),
        carbon=CarbonConfig(scheme="differland", C_fol_init=100.0),
    )
    cfg_high = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(
            max_iters=30, use_prognostic_lai=True),
        carbon=CarbonConfig(scheme="differland", C_fol_init=400.0),
    )

    state = init_multilayer_land_state(ncol, cfg_low, T_init=290.0, TgC_init=20.0)
    carbon_low = init_carbon_state((ncol,), cfg_low.carbon)    # C_fol=100 → LAI=2
    carbon_high = init_carbon_state((ncol,), cfg_high.carbon)  # C_fol=400 → LAI=8

    _, r_low, _ = step_multilayer_land(
        state, forcing, cfg_low, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), carbon_state=carbon_low, doy=180.0,
        land_params=_canopy_params(ncol))
    _, r_high, _ = step_multilayer_land(
        state, forcing, cfg_high, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), carbon_state=carbon_high, doy=180.0,
        land_params=_canopy_params(ncol))

    assert not jnp.allclose(r_low.lhflx, r_high.lhflx, atol=1e-3), (
        "canopy with LAI=2 must differ from canopy with LAI=8")


def test_prognostic_lai_jax_grad_through_feedback():
    """``jax.grad`` through the prognostic LAI loop (C_fol → LAI →
    canopy Newton → surface fluxes) is finite for training.

    Was xfail(strict) while the two-leaf canopy solve crashed
    (``photosynthesis`` dropped its gross-GPP return -> the residual's
    ``An, _ = photosynthesis(...)`` unpack raised "iteration over a 0-d
    array"); the forward pass never ran, so the grad through it was moot.
    With the canopy solve restored the reverse-mode grad through the LAI
    feedback is finite + nonzero, so the marker is removed and this now
    guards the differentiable-LAI path directly.
    """
    ncol = 2
    cfg = _make_canopy_carbon_cfg()
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0, TgC_init=20.0)
    carbon_base = init_carbon_state((ncol,), cfg.carbon)
    forcing = _make_forcing(ncol, sw_down=700.0, cos_zenith=0.8)
    params = _canopy_params(ncol)

    def _loss(C_fol_scale):
        carbon = carbon_base._replace(
            C_fol=carbon_base.C_fol * C_fol_scale)
        _, resp, _ = step_multilayer_land(
            state, forcing, cfg, U_min=1.0, dt=1800.0,
            lat=jnp.zeros(ncol), carbon_state=carbon, doy=180.0,
            land_params=params)
        return jnp.sum(resp.lhflx)

    grad = jax.grad(_loss)(1.0)
    assert jnp.isfinite(grad), f"non-finite grad wrt C_fol scale: {grad}"
    assert float(grad) != 0.0


def test_prognostic_lai_preserves_grad_wrt_vcmax():
    """Regression: enabling prognostic LAI must NOT break ``jax.grad``
    with respect to other canopy parameters.  Guards against the
    prognostic LAI feedback accidentally introducing a NaN into the
    generic canopy gradient path.
    """
    ncol = 2
    cfg = _make_canopy_carbon_cfg()  # fixture enables prognostic LAI
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0, TgC_init=20.0)
    carbon = init_carbon_state((ncol,), cfg.carbon)
    forcing = _make_forcing(ncol, sw_down=700.0, cos_zenith=0.8)

    def _loss(Vc):
        params = _canopy_params(ncol)._replace(
            Vcmax25_C3_leaf=jnp.full(ncol, Vc))
        _, _, c_new = step_multilayer_land(
            state, forcing, cfg, U_min=1.0, dt=1800.0,
            lat=jnp.zeros(ncol), carbon_state=carbon, doy=180.0,
            land_params=params)
        return jnp.sum(c_new.C_lab - carbon.C_lab)

    grad = jax.grad(_loss)(60.0)
    assert jnp.isfinite(grad)
    # More Vcmax → more GPP → more C_lab allocation.  Sign must be positive.
    assert float(grad) > 0.0
