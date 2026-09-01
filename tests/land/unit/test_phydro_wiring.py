"""P-hydro wiring: switch dispatch, refusals, step-level integration."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.canopy.config import CanopyConfig, CLMMLCanopyConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state, step_multilayer_land)


def _forcing(ncol):
    ones = jnp.ones(ncol)
    vals = dict(
        sw_down=400.0, lw_down=300.0, precip_total=1e-5, precip_snow=0.0,
        T_lowest=292.0, q_lowest=6e-3, u_lowest=3.0, v_lowest=1.0,
        p_lowest=1e5, p_surface=1.013e5, rho_lowest=1.2, cos_zenith=0.6,
        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0,
    )
    return AtmToSurface(**{k: v * ones for k, v in vals.items()})


def _cfg(**kw):
    return MultiLayerLandConfig(
        surface_scheme=CanopyConfig(
            stomatal_model="medlyn", capacity_scheme="p_model",
            g1_source="p_model").validate(),
        **kw)


def test_unknown_transpiration_stress_raises():
    cfg = _cfg(transpiration_stress="weibull")
    st = init_multilayer_land_state(2, cfg)
    with pytest.raises(ValueError, match="transpiration_stress"):
        step_multilayer_land(st, _forcing(2), cfg, U_min=1.0, dt=1800.0)


def test_phydro_requires_pmodel_switch():
    cfg = MultiLayerLandConfig(surface_scheme=CanopyConfig(),
                               transpiration_stress="phydro")
    st = init_multilayer_land_state(2, cfg)
    with pytest.raises(ValueError, match="P-model switch"):
        step_multilayer_land(st, _forcing(2), cfg, U_min=1.0, dt=1800.0)


def test_phydro_refused_on_clm_ml():
    cfg = MultiLayerLandConfig(
        surface_scheme=CLMMLCanopyConfig(
            stomatal_model="medlyn", capacity_scheme="p_model").validate(),
        transpiration_stress="phydro")
    st = init_multilayer_land_state(2, cfg)
    with pytest.raises(ValueError, match="plant hydraulics"):
        step_multilayer_land(st, _forcing(2), cfg, U_min=1.0, dt=1800.0)


def test_phydro_two_leaf_step_runs_and_switch_binds():
    """One step runs finitely under phydro, and the switch CHANGES the
    fluxes relative to beta_theta on a water-limited column (the empirical
    multiplier is replaced by the profit optimum — a step-level flux
    inequality between wet/dry columns is confounded by soil evaporation
    and the energy partition, so the binding-switch check is the honest
    step-level assertion; the optimum-level drought response is pinned in
    test_phydro.py)."""
    cfg_ph = _cfg(transpiration_stress="phydro")
    cfg_bt = _cfg()
    st = init_multilayer_land_state(2, cfg_ph, theta_init=0.14)
    _, resp_ph, _ = step_multilayer_land(
        st, _forcing(2), cfg_ph, U_min=1.0, dt=1800.0)
    _, resp_bt, _ = step_multilayer_land(
        st, _forcing(2), cfg_bt, U_min=1.0, dt=1800.0)
    for r in (resp_ph, resp_bt):
        assert np.all(np.isfinite(np.asarray(r.T_sfc)))
        assert np.all(np.isfinite(np.asarray(r.lhflx)))
    assert not np.allclose(np.asarray(resp_ph.lhflx),
                           np.asarray(resp_bt.lhflx), atol=1e-3)
    # And the SUPPLY responds to drying: a drier column has lower lsc.
    from legoesm.land.phydro import soil_root_supply
    from legoesm.land.soil_grid import make_soil_grid
    grid = make_soil_grid(cfg_ph.soil_grid)
    st_dry = init_multilayer_land_state(2, cfg_ph, theta_init=0.08)
    nlev = st.theta_soil.shape[1]
    rf = jnp.full((2, nlev), 1.0 / nlev)
    lai = jnp.full((2,), 3.0)
    sup_wet = soil_root_supply(st.psi_soil, st.theta_soil, grid.dz, rf, lai,
                               cfg_ph.hydraulics, cfg_ph.phydro)
    sup_dry = soil_root_supply(st_dry.psi_soil, st_dry.theta_soil, grid.dz,
                               rf, lai, cfg_ph.hydraulics, cfg_ph.phydro)
    assert float(sup_dry.lsc_mol[0]) < float(sup_wet.lsc_mol[0])


def test_beta_theta_default_unchanged():
    cfg = _cfg()  # transpiration_stress default
    assert cfg.transpiration_stress == "beta_theta"
    st = init_multilayer_land_state(2, cfg)
    _, resp, _ = step_multilayer_land(st, _forcing(2), cfg, U_min=1.0,
                                      dt=1800.0)
    assert np.all(np.isfinite(np.asarray(resp.lhflx)))


def test_cli_round_trip():
    import pytest as _pt
    from scripts.run.run_lmip import _parse_args, build_config_from_args

    land = build_config_from_args(_parse_args(
        ["--lat", "45.0", "--land-surface-scheme", "two_leaf",
         "--canopy-stomatal-model", "medlyn",
         "--canopy-capacity-scheme", "p_model",
         "--canopy-g1-source", "p_model",
         "--transpiration-stress", "phydro"])).land
    assert land.transpiration_stress == "phydro"
    land2 = build_config_from_args(_parse_args(
        ["--lat", "45.0", "--land-surface-scheme", "two_leaf"])).land
    assert land2.transpiration_stress == "beta_theta"
    with _pt.raises(SystemExit):
        _parse_args(["--lat", "45.0", "--transpiration-stress", "weibull"])
    with _pt.raises(ValueError, match="clm_ml"):
        build_config_from_args(_parse_args(
            ["--lat", "45.0", "--land-surface-scheme", "clm_ml",
             "--clm-ml-stomatal-model", "medlyn",
             "--canopy-capacity-scheme", "p_model",
             "--transpiration-stress", "phydro"]))
