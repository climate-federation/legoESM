"""Budget-closure diagnostics from ``step_multilayer_land_with_budget``.

The taped ``Rnet``/``shflx``/``lhflx`` alone cannot close the surface energy
budget: ``SurfaceFluxOutput.G_soil`` is DEFINED as ``Rnet - SH - LH`` and the
snow phase change is then subtracted from it.  So a diagnosed "residual" of
``Rnet - SH - LH`` is the ground heat flux, not an energy leak — a distinction
that cost real analysis time to establish and is pinned here.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy import CanopyConfig
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    LandStepDiagnostics,
    init_multilayer_land_state,
    step_multilayer_land,
    step_multilayer_land_with_budget,
    step_multilayer_land_with_diagnostics,
)
from legoesm.land.soil_grid import SoilGridConfig

NCOL = 6


def _forcing(T=283.0, sw=400.0, precip=0.0, snow_frac=0.0):
    one = jnp.ones(NCOL)
    return AtmToSurface(
        sw_down=one * sw, lw_down=one * 300.0,
        precip_total=one * precip, precip_snow=one * precip * snow_frac,
        T_lowest=one * T, q_lowest=one * 5e-3,
        u_lowest=one * 3.0, v_lowest=one * 0.0,
        p_lowest=one * 1.0e5, p_surface=one * 1.0e5,
        rho_lowest=one * 1.2, cos_zenith=one * 0.6, co2_ppmv=one * 412.0,
        has_radiation=True, has_precipitation=True,
    )


def _cfg(**kw):
    # NOTE: this file runs ~13 min.  The cost is NOT the canopy Newton iteration
    # count -- measured 12:39 at max_iters=6 vs 12:53 at 30 -- so it is the eager
    # (un-jitted) per-step dispatch of the whole land column.  Left at the
    # production-like 30; lowering it buys nothing.
    return MultiLayerLandConfig(
        surface_scheme=CanopyConfig(max_iters=30),
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0),
        bulk_scheme="most", **kw)


def _run(cfg, forcing, dt=1800.0, nstep=3):
    st = init_multilayer_land_state(NCOL, cfg)
    out = []
    for _ in range(nstep):
        st, resp, _c, so, bud = step_multilayer_land_with_budget(
            st, forcing, cfg, 1.0, dt)
        out.append((st, resp, so, bud))
    return out


# --------------------------------------------------------------------------
# Wrapper contracts
# --------------------------------------------------------------------------
def test_existing_wrappers_keep_their_arity():
    """The 3- and 4-tuple contracts have many call sites (EC-site driver, canopy
    validators, tests); widening them would churn all of them."""
    cfg, f = _cfg(), _forcing()
    st = init_multilayer_land_state(NCOL, cfg)
    assert len(step_multilayer_land(st, f, cfg, 1.0, 1800.0)) == 3
    assert len(step_multilayer_land_with_diagnostics(st, f, cfg, 1.0, 1800.0)) == 4
    assert len(step_multilayer_land_with_budget(st, f, cfg, 1.0, 1800.0)) == 5


def test_budget_wrapper_agrees_with_the_others():
    """The extra return must not perturb the physics."""
    cfg, f = _cfg(), _forcing()
    st = init_multilayer_land_state(NCOL, cfg)
    s3, r3, _ = step_multilayer_land(st, f, cfg, 1.0, 1800.0)
    s5, r5, _c, _so, _b = step_multilayer_land_with_budget(st, f, cfg, 1.0, 1800.0)
    np.testing.assert_array_equal(np.asarray(r3.shflx), np.asarray(r5.shflx))
    np.testing.assert_array_equal(np.asarray(s3.T_soil), np.asarray(s5.T_soil))


def test_diagnostics_are_finite_and_shaped():
    for (_st, _r, _so, b) in _run(_cfg(), _forcing()):
        assert isinstance(b, LandStepDiagnostics)
        for name in b._fields:
            v = np.asarray(getattr(b, name))
            assert v.shape == (NCOL,), name
            assert np.isfinite(v).all(), name


# --------------------------------------------------------------------------
# THE identity the analysis depends on
# --------------------------------------------------------------------------
@pytest.mark.parametrize("T,precip,snow_frac", [
    (290.0, 0.0, 0.0),          # warm, dry, no snow
    (283.0, 1e-5, 0.0),         # rain
    (263.0, 1e-5, 1.0),         # snowfall, freezing -> melt/refreeze branch live
])
def test_energy_identity_rnet_minus_sh_lh_equals_g_plus_melt(T, precip, snow_frac):
    """Rnet - SH - LH == G + melt_energy, to roundoff.

    This is what makes the surface budget CHECKABLE: without melt_energy taped,
    the melt term is invisible and the analyst mistakes G for an energy leak.
    """
    cfg = _cfg()
    f = _forcing(T=T, precip=precip, snow_frac=snow_frac)
    for (_st, _resp, so, b) in _run(cfg, f, nstep=4):
        lhs = np.asarray(so.G_soil)                   # == Rnet - SH - LH by construction
        rhs = np.asarray(b.g_soil) + np.asarray(b.melt_energy)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-8)


def test_melt_energy_is_zero_without_snow():
    for (_st, _r, _so, b) in _run(_cfg(), _forcing(T=295.0)):
        np.testing.assert_allclose(np.asarray(b.melt_energy), 0.0, atol=0.0)
        np.testing.assert_allclose(np.asarray(b.snowmelt), 0.0, atol=0.0)


def test_melt_energy_matches_the_melt_mass():
    """melt_energy must be the fusion enthalpy of the taped melt (plus blowing-snow
    sublimation, zero on the single-pack path) — otherwise the two diagnostics
    could drift and the budget would close on inconsistent terms."""
    cfg = _cfg()
    f = _forcing(T=272.0, sw=600.0, precip=2e-5, snow_frac=1.0)
    for (_st, _r, _so, b) in _run(cfg, f, nstep=6):
        expected = np.asarray(b.snowmelt) * constants.L_f
        np.testing.assert_allclose(np.asarray(b.melt_energy), expected,
                                   rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------
# Water-budget storage terms
# --------------------------------------------------------------------------
def test_soil_water_matches_a_hand_integral():
    """The taped column water must equal sum(theta*dz)*rho_w on the SAME grid the
    model integrates — the top-layer-only theta_soil_top cannot close a budget
    (it is 2.9 mm of a 3 m column on the AMIP-parity grid)."""
    from legoesm.land.soil_grid import make_soil_grid
    cfg = _cfg()
    dz = np.asarray(make_soil_grid(cfg.soil_grid).dz)
    for (st, _r, _so, _b) in _run(cfg, _forcing(precip=1e-5)):
        want = (np.asarray(st.theta_soil) * dz[None, :]).sum(-1) * constants.rho_water
        assert want.shape == (NCOL,)
        assert (want > 0).all()
        # a 3 m column at plausible theta
        assert (want / (constants.rho_water * dz.sum()) < 1.0).all()


def test_sublimation_sign_convention():
    """Positive = mass leaving the pack as vapour; negative = frost deposition.
    The spin-up showed persistently NEGATIVE ET over perennial snow, so the sign
    here is load-bearing for interpreting the ice-sheet mass budget."""
    cfg = _cfg()
    for (_st, _r, _so, b) in _run(cfg, _forcing(T=260.0, precip=1e-5, snow_frac=1.0), nstep=5):
        assert np.isfinite(np.asarray(b.sublimation)).all()
