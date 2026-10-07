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
    # And the SUPPLY responds to drying: a drier column has a higher
    # soil-to-root resistance.
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
    assert float(sup_dry.rsoil_leaf[0]) > float(sup_wet.rsoil_leaf[0])


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


def test_g1_nonmedlyn_requires_phydro_at_the_step():
    """Least-cost g1 is a Medlyn quantity: p_model g1 + ball_berry WITHOUT
    phydro is refused at the consumer; WITH phydro it runs (the slope maps
    onto the active model)."""
    cfg_bad = MultiLayerLandConfig(surface_scheme=CanopyConfig(
        stomatal_model="ball_berry", capacity_scheme="p_model",
        g1_source="p_model").validate())
    st = init_multilayer_land_state(2, cfg_bad)
    with pytest.raises(ValueError, match="MEDLYN"):
        step_multilayer_land(st, _forcing(2), cfg_bad, U_min=1.0, dt=1800.0)

    cfg_ok = MultiLayerLandConfig(
        surface_scheme=CanopyConfig(
            stomatal_model="ball_berry", capacity_scheme="p_model",
            g1_source="p_model").validate(),
        transpiration_stress="phydro")
    st2 = init_multilayer_land_state(2, cfg_ok)
    _, resp, _ = step_multilayer_land(st2, _forcing(2), cfg_ok, U_min=1.0,
                                      dt=1800.0)
    assert np.all(np.isfinite(np.asarray(resp.lhflx)))


def _incident_ppfd(forcing):
    from legoesm.land.canopy.radiative_transfer import (
        PAR_W_TO_UMOL, split_sw_components)
    par_dir, par_diff, _, _, _ = split_sw_components(
        forcing.sw_down, forcing.cos_zenith)
    return (par_dir + par_diff) * PAR_W_TO_UMOL


def test_sunlit_leaf_apar_is_leaf_level_absorbed_light():
    """APAR_Sun / (fSun * LAI): below the incident flux under high sun, zero at
    night, finite and bounded under diffuse-only light (where the dusk ramp on
    fSun would otherwise blow the ratio up), and only weakly dependent on LAI
    (a per-leaf-area quantity, not a canopy total)."""
    from legoesm.land.surface_scheme.two_leaf_canopy import sunlit_leaf_apar
    f = _forcing(4)._replace(
        sw_down=jnp.asarray([800.0, 0.0, 25.0, 800.0]),
        cos_zenith=jnp.asarray([0.8, 0.0, 0.3, 0.8]))
    lai = jnp.asarray([2.0, 2.0, 2.0, 4.0])
    leaf = np.asarray(sunlit_leaf_apar(f, None, LAI_override=lai))
    inc = np.asarray(_incident_ppfd(f))
    assert np.all(np.isfinite(leaf))
    assert 0.25 * inc[0] < leaf[0] < 0.9 * inc[0]      # clear midday
    assert leaf[1] == 0.0                              # night
    assert 0.0 < leaf[2] < 2.0 * inc[2]                # dim diffuse light
    assert abs(leaf[3] / leaf[0] - 1.0) < 0.25         # LAI 2 -> 4


def test_two_leaf_acclimation_light_is_sunlit_leaf_apar():
    """The P-model acclimation mean under the two-leaf scheme advances with
    the PAR a sunlit leaf absorbs — not the incident top-of-canopy PPFD."""
    from legoesm.land.canopy.energy_balance import canopy_met_variables
    from legoesm.land.p_model import advance_pmodel_acclim
    from legoesm.land.surface_scheme.two_leaf_canopy import sunlit_leaf_apar
    cfg = _cfg()
    st = init_multilayer_land_state(2, cfg)
    f = _forcing(2)
    new, _, _ = step_multilayer_land(st, f, cfg, U_min=1.0, dt=1800.0)

    def expected(ppfd):
        vpd = canopy_met_variables(f.p_surface, f.T_lowest, f.q_lowest)[2]
        return advance_pmodel_acclim(
            st.pmodel_acclim, T_K=f.T_lowest, ppfd=ppfd, vpd_pa=vpd,
            co2_ppm=f.co2_ppmv, ps_pa=f.p_surface, cfg=cfg.p_model,
            dt=1800.0).iabs_mean

    got = np.asarray(new.pmodel_acclim.iabs_mean)
    np.testing.assert_allclose(
        got, np.asarray(expected(sunlit_leaf_apar(f, None))), rtol=1e-5)
    assert not np.allclose(got, np.asarray(expected(_incident_ppfd(f))), rtol=1e-3)
