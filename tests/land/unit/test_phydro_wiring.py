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
    """Least-cost arm, or the one-configuration P-hydro arm when
    ``transpiration_stress='phydro'`` (hydraulic stress + optimum capacities
    + the instantaneous optimum as the stomatal model, no slope)."""
    if kw.get("transpiration_stress") == "phydro":
        cc = CanopyConfig(stomatal_model="phydro", capacity_scheme="phydro",
                          g1_source="table").validate()
    else:
        cc = CanopyConfig(stomatal_model="medlyn", capacity_scheme="p_model",
                          g1_source="p_model").validate()
    return MultiLayerLandConfig(surface_scheme=cc, **kw)


def test_unknown_transpiration_stress_raises():
    cfg = _cfg(transpiration_stress="weibull")
    st = init_multilayer_land_state(2, cfg)
    with pytest.raises(ValueError, match="transpiration_stress"):
        step_multilayer_land(st, _forcing(2), cfg, U_min=1.0, dt=1800.0)


def test_phydro_requires_pmodel_switch():
    cfg = MultiLayerLandConfig(surface_scheme=CanopyConfig(),
                               transpiration_stress="phydro")
    st = init_multilayer_land_state(2, cfg)
    with pytest.raises(ValueError, match="together|P-model switch"):
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


def test_g1_nonmedlyn_is_refused_and_the_phydro_arm_is_one_configuration():
    """Least-cost g1 is a Medlyn quantity: p_model g1 + ball_berry is refused
    at the consumer.  The P-hydro arm is one configuration: the hydraulic
    stress without the phydro stomatal model (or with a p_model slope) is
    refused, the full arm runs."""
    cfg_bad = MultiLayerLandConfig(surface_scheme=CanopyConfig(
        stomatal_model="ball_berry", capacity_scheme="p_model",
        g1_source="p_model").validate())
    st = init_multilayer_land_state(2, cfg_bad)
    with pytest.raises(ValueError, match="MEDLYN"):
        step_multilayer_land(st, _forcing(2), cfg_bad, U_min=1.0, dt=1800.0)

    for cc, msg in ((CanopyConfig(stomatal_model="medlyn", capacity_scheme="p_model",
                                  g1_source="p_model"), "together"),
                    (CanopyConfig(stomatal_model="phydro", capacity_scheme="p_model",
                                  g1_source="table"), "together"),
                    (CanopyConfig(stomatal_model="phydro", capacity_scheme="phydro",
                                  g1_source="p_model"), "g1_source must be 'table'")):
        cfg_half = MultiLayerLandConfig(surface_scheme=cc.validate(),
                                        transpiration_stress="phydro")
        st_h = init_multilayer_land_state(2, cfg_half)
        with pytest.raises(ValueError, match=msg):
            step_multilayer_land(st_h, _forcing(2), cfg_half, U_min=1.0, dt=1800.0)

    cfg_ok = _cfg(transpiration_stress="phydro")
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


def _two_leaf_fluxes(cc, w_frac_rz, *, transpiration_stress="beta_theta",
                     canopy_params=None):
    """Direct two-leaf flux call at one sunny column (no soil model)."""
    from legoesm.land.p_model import PModelConfig, init_pmodel_acclim
    from legoesm.land.phydro import PhydroSupply
    from legoesm.land.surface_scheme.two_leaf_canopy import (
        compute_two_leaf_canopy_fluxes)
    one = jnp.ones(1)
    f = _forcing(1)._replace(sw_down=700.0 * one, cos_zenith=0.9 * one)
    acclim = init_pmodel_acclim(
        1, t_init_K=292.0, ps_init_pa=1.013e5, cfg=PModelConfig(),
        soil_init=((-0.4, 20.0) if transpiration_stress == "phydro" else None))
    out = compute_two_leaf_canopy_fluxes(
        T_soil_top=293.0 * one, forcing=f, canopy_config=cc,
        land_config=MultiLayerLandConfig(
            surface_scheme=cc, transpiration_stress=transpiration_stress),
        canopy_params=canopy_params, w_frac_rz=w_frac_rz * one,
        wind_speed=3.0 * one, wind_dir_x=one, wind_dir_y=0.0 * one,
        soil_thermal_fn=lambda G, dt_: 293.0 * one, dt=1800.0,
        w_frac_soil_evap=0.5 * one, pmodel_acclim=acclim,
        phydro_supply=PhydroSupply(rsoil_leaf=20.0 * one, psi_s_mpa=-0.4 * one))
    return np.asarray([out.gpp[0], out.lhflx[0], out.shflx[0]], dtype=float)


def test_phydro_two_leaf_applies_no_empirical_stress():
    """Under phydro the root-zone factor reaches NEITHER Vcmax25 nor the
    stomatal slope/intercept: with the hydraulic supply and the soil-evaporation
    stress held fixed, the solved GPP / LE / H are identical for a wet and a
    dry root-zone factor.  The same change moves the beta_theta fluxes."""
    cc_ph = CanopyConfig(stomatal_model="phydro", capacity_scheme="phydro",
                         g1_source="table").validate()
    wet = _two_leaf_fluxes(cc_ph, 1.0, transpiration_stress="phydro")
    dry = _two_leaf_fluxes(cc_ph, 0.2, transpiration_stress="phydro")
    assert np.all(np.isfinite(wet)) and wet[0] > 0.0
    np.testing.assert_allclose(dry, wet, rtol=1e-9, atol=1e-9)
    cc_lc = CanopyConfig(stomatal_model="medlyn", capacity_scheme="p_model",
                         g1_source="p_model").validate()
    bt_wet, bt_dry = _two_leaf_fluxes(cc_lc, 1.0), _two_leaf_fluxes(cc_lc, 0.2)
    assert bt_dry[0] < 0.6 * bt_wet[0]


def test_least_cost_slope_is_not_stressed():
    """rpmodel structure: the empirical stress scales the capacities only.
    With prescribed capacities, a P-model slope under stress w gives exactly
    the fluxes of a TABULATED slope g1/w under the same stress (whose stressed
    product is g1) — i.e. the predicted slope itself was left unstressed."""
    from types import SimpleNamespace
    from legoesm.land.p_model import (
        PModelConfig, acclimated_capacities, init_pmodel_acclim, optimal_chi)
    from legoesm.land.phydro import slope_for_model
    w = 0.5
    acclim = init_pmodel_acclim(1, t_init_K=292.0, ps_init_pa=1.013e5,
                                cfg=PModelConfig())
    caps = acclimated_capacities(acclim, PModelConfig())
    gamma_star = optimal_chi(acclim.t_mean_K, acclim.vpd_mean_pa,
                             acclim.co2_mean_ppm, acclim.ps_ema, PModelConfig())[3]
    g1 = slope_for_model("medlyn", caps.chi, acclim, gamma_star)
    pm_slope = _two_leaf_fluxes(
        CanopyConfig(stomatal_model="medlyn", g1_source="p_model").validate(), w)
    table = _two_leaf_fluxes(
        CanopyConfig(stomatal_model="medlyn").validate(), w,
        canopy_params=SimpleNamespace(m_C3=g1 / w))
    # rtol: the two runs start the Newton solve from different Ci seeds
    # (chi* vs the fixed 0.7) and agree to the solver tolerance.
    np.testing.assert_allclose(pm_slope, table, rtol=1e-4)
    stressed = _two_leaf_fluxes(
        CanopyConfig(stomatal_model="medlyn").validate(), w,
        canopy_params=SimpleNamespace(m_C3=g1))
    assert not np.allclose(pm_slope, stressed, rtol=1e-3)


def test_least_cost_slope_reproduces_the_full_least_cost_chi():
    """The two-leaf scheme converts the least-cost chi with the same mapping
    as P-hydro: Medlyn's ci/ca = g1/(g1 + sqrt(D)) then equals the FULL
    least-cost chi at the mean VPD (the textbook g1 = xi drops the Gamma*/ca
    term and gives a lower chi), and the fluxes are those of a tabulated
    slope of that value — not of xi."""
    from types import SimpleNamespace
    from legoesm.land.p_model import (
        PModelConfig, acclimated_capacities, init_pmodel_acclim, optimal_chi)
    from legoesm.land.phydro import slope_for_model
    pcfg = PModelConfig()
    acclim = init_pmodel_acclim(1, t_init_K=292.0, ps_init_pa=1.013e5, cfg=pcfg)
    caps = acclimated_capacities(acclim, pcfg)
    gamma_star = optimal_chi(acclim.t_mean_K, acclim.vpd_mean_pa,
                             acclim.co2_mean_ppm, acclim.ps_ema, pcfg)[3]
    g1 = slope_for_model("medlyn", caps.chi, acclim, gamma_star)
    sqrt_d = np.sqrt(float(acclim.vpd_mean_pa[0]) / 1000.0)
    np.testing.assert_allclose(
        float(g1[0]) / (float(g1[0]) + sqrt_d), float(caps.chi[0]), rtol=1e-6)
    assert float(g1[0]) > 1.05 * float(caps.g1_kpa[0])   # exceeds xi
    medlyn = CanopyConfig(stomatal_model="medlyn").validate()
    got = _two_leaf_fluxes(
        CanopyConfig(stomatal_model="medlyn", g1_source="p_model").validate(), 1.0)
    np.testing.assert_allclose(
        got, _two_leaf_fluxes(medlyn, 1.0, canopy_params=SimpleNamespace(m_C3=g1)),
        rtol=1e-4)  # different Ci seeds -> agreement to the solver tolerance
    assert not np.allclose(
        got, _two_leaf_fluxes(medlyn, 1.0, canopy_params=SimpleNamespace(m_C3=caps.g1_kpa)),
        rtol=1e-3)


def test_solver_ci_seed_uses_the_pmodel_optimum():
    """Newton Ci seed: Ca * chi* when the P model supplies an optimum (clipped
    to the kernel's Ci/Ca range, no gradient), else the fixed C3/C4 ratios."""
    import jax
    from legoesm.land.surface_scheme.two_leaf_canopy import ci_seed
    ca, fc4 = jnp.asarray([400.0, 400.0, 400.0]), jnp.asarray([0.0, 1.0, 0.5])
    np.testing.assert_allclose(np.asarray(ci_seed(ca, fc4)), [280.0, 160.0, 220.0])
    chi = jnp.asarray([0.62, 0.30, 1.0])
    np.testing.assert_allclose(np.asarray(ci_seed(ca, fc4, chi)), [248.0, 200.0, 360.0])
    g = jax.grad(lambda c: jnp.sum(ci_seed(ca, fc4, c)))(chi)
    assert np.all(np.asarray(g) == 0.0)


def test_least_cost_jmax_ratio_is_rederived_under_stress(monkeypatch):
    """Two-leaf least-cost lane: under the empirical stress the Jmax25/Vcmax25
    ratio is reduced as in rpmodel (wired, not just defined) — neutralising
    the factor changes the stressed fluxes and leaves the unstressed ones."""
    import legoesm.land.surface_scheme.two_leaf_canopy as tl
    cc = CanopyConfig(stomatal_model="medlyn", capacity_scheme="p_model",
                      g1_source="p_model").validate()
    dry, wet = _two_leaf_fluxes(cc, 0.3), _two_leaf_fluxes(cc, 1.0)
    monkeypatch.setattr(tl, "stressed_rjv25_factor",
                        lambda jmaxlim, stress: jnp.ones_like(stress))
    dry_off, wet_off = _two_leaf_fluxes(cc, 0.3), _two_leaf_fluxes(cc, 1.0)
    np.testing.assert_allclose(wet_off, wet, rtol=1e-6)
    assert dry[0] < dry_off[0] * (1.0 - 1e-4)       # lower Jmax -> lower GPP
