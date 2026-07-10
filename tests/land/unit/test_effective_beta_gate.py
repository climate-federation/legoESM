"""compute_effective_beta enabled-gate: stomata OFF returns the bare soil-
moisture beta (no physiological control); stomata ON applies stomatal
conductance limitation to land ET.  Guards the run_coupled --stomata wiring."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import LandConfig
from legoesm.land.stomata_utils import compute_effective_beta


def _forcing(co2_ppmv=400.0):
    shape = (4, 4)
    z = jnp.zeros(shape)
    return AtmToSurface(
        sw_down=jnp.full(shape, 600.0), lw_down=jnp.full(shape, 350.0),
        precip_total=z, precip_snow=z,
        T_lowest=jnp.full(shape, 295.0), q_lowest=jnp.full(shape, 8e-3),
        u_lowest=jnp.full(shape, 2.0), v_lowest=z,
        p_lowest=jnp.full(shape, 1.0e5), p_surface=jnp.full(shape, 1.0e5),
        rho_lowest=jnp.full(shape, 1.15), cos_zenith=jnp.full(shape, 0.7),
        co2_ppmv=jnp.array(co2_ppmv), has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(0.0),
    )


def test_stomata_off_returns_bare_soil_beta():
    """enabled=False => beta == beta_soil (no stomatal/physiological control)."""
    cfg = LandConfig()
    assert cfg.stomata.enabled is False               # default
    beta_soil = jnp.full((4, 4), 0.7)
    beta, gpp, _sif = compute_effective_beta(
        jnp.full((4, 4), 298.0), _forcing(), beta_soil, cfg,
        carbon_state=None, dt=3600.0)
    np.testing.assert_allclose(np.asarray(beta), np.asarray(beta_soil))
    assert gpp is None


def test_stomata_on_applies_conductance_limit():
    """enabled=True (Jarvis, no carbon) => beta is stomatally limited
    (<= beta_soil) and DIFFERS from the bare soil beta => ET is physiologically
    controlled, not the no-stomata case."""
    cfg = LandConfig()._replace(
        stomata=LandConfig().stomata._replace(enabled=True))
    beta_soil = jnp.full((4, 4), 0.7)
    beta, _, _ = compute_effective_beta(
        jnp.full((4, 4), 298.0), _forcing(), beta_soil, cfg,
        carbon_state=None, dt=3600.0)
    beta = np.asarray(beta)
    assert np.all(np.isfinite(beta))
    assert np.all(beta <= np.asarray(beta_soil) + 1e-9)   # stomata can only limit
    assert not np.allclose(beta, np.asarray(beta_soil))   # genuinely active


def _land_params(shape, fc4):
    """Minimal shape-matched LandSurfaceParams with a prescribed C4 fraction."""
    from legoesm.land.surface_params import LandSurfaceParams
    f = lambda v: jnp.full(shape, v)
    return LandSurfaceParams(
        albedo_veg=f(0.15), emissivity=f(0.97), z0=f(0.1),
        W_max=f(200.0), C_soil=f(2.0e6), d_soil=f(1.0),
        root_depth=f(1.0), theta_wp=f(0.15), theta_fc=f(0.30),
        Vc_max25=f(60.0), LCMA=f(50.0), g1=f(9.0),
        fC4=f(fc4),
    )


def test_c4_fraction_changes_gpp_through_dispatch():
    """A C4-dominated column (land_params.fC4=1) yields a DIFFERENT coupled-
    Farquhar GPP than a C3 column (fC4=0) through compute_effective_beta —
    proving the canonical C3/C4 blend is wired end-to-end, not silently pure-C3
    (the pre-canonical-FvCB behaviour ran every column as C3)."""
    from legoesm.land.carbon.carbon_cycle import init_carbon_state
    shape = (4, 4)
    base = LandConfig()
    cfg = base._replace(
        stomata=base.stomata._replace(enabled=True),
        carbon=base.carbon._replace(scheme="differland"),
    )
    carbon = init_carbon_state(shape, cfg.carbon)

    def _gpp(fc4):
        _beta, gpp, _sif = compute_effective_beta(
            jnp.full(shape, 298.0), _forcing(), jnp.full(shape, 0.7), cfg,
            carbon_state=carbon, dt=3600.0, land_params=_land_params(shape, fc4))
        return np.asarray(gpp)

    gpp_c3 = _gpp(0.0)
    gpp_c4 = _gpp(1.0)
    assert np.all(np.isfinite(gpp_c3)) and np.all(np.isfinite(gpp_c4))
    # The C4 branch (Collatz) genuinely changes GPP vs pure C3 at these
    # conditions — the blend is live, not a silent fC4=0 fallback.
    assert not np.allclose(gpp_c3, gpp_c4)


def test_c4_fraction_defaults_to_c3_when_unset():
    """land_params with fC4=None (prescribed-PFT path) runs pure C3 — the
    documented default, matching the no-land_params path."""
    from legoesm.land.carbon.carbon_cycle import init_carbon_state
    from legoesm.land.surface_params import LandSurfaceParams
    shape = (4, 4)
    base = LandConfig()
    cfg = base._replace(
        stomata=base.stomata._replace(enabled=True),
        carbon=base.carbon._replace(scheme="differland"),
    )
    carbon = init_carbon_state(shape, cfg.carbon)
    lp_none = _land_params(shape, 0.0)._replace(fC4=None)
    lp_zero = _land_params(shape, 0.0)
    args = (jnp.full(shape, 298.0), _forcing(), jnp.full(shape, 0.7), cfg)
    _, gpp_none, _ = compute_effective_beta(
        *args, carbon_state=carbon, dt=3600.0, land_params=lp_none)
    _, gpp_zero, _ = compute_effective_beta(
        *args, carbon_state=carbon, dt=3600.0, land_params=lp_zero)
    np.testing.assert_allclose(np.asarray(gpp_none), np.asarray(gpp_zero))
