"""``transpiration_stress="hydraulic"``: the Medlyn supply-demand closure.

Design: docs/land/hydraulic_stress_design.md, step 1.  Pins, on one midday
column of the two-leaf canopy Newton (``canopy/solver.py``):

* with the vulnerability curves switched off (psi50 -> -inf, so the xylem
  conducts freely and the slope is unstressed) the hydraulic baseline
  reproduces today's unstressed Medlyn column: same six shared unknowns,
  same fluxes;
* the supply-demand row of each class is zero at convergence, i.e. the
  water the xylem delivers at the solved draw-down equals the Medlyn demand;
* sunlit leaves draw down more than shaded ones at midday;
* the conductance falls monotonically with drying soil;
* ``beta_theta`` is untouched (6-state, no supply fields);
* the switch is accepted by the land config / step and refused where its
  row is not written (non-Medlyn stomata, big leaf, CLM-ML).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy.config import CanopyConfig, CLMMLCanopyConfig
from legoesm.land.canopy.solver import (
    CanopyForcingBundle, _canopy_residual, canopy_forward,
    solve_canopy_closure, state_size)
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state, step_multilayer_land)
from legoesm.land.plant_hydraulics import (
    PlantHydraulicsConfig, leaf_psi, supply_demand_residual, weibull_stress)

_a = jnp.asarray
_OFF = -1.0e6  # psi50 -> -inf: W = 1 at every leaf potential, P = 1 on the xylem path


def _medlyn_bundle(**over):
    """One midday column, Medlyn slope 4, no supply fields (6-state)."""
    kw = dict(
        LAI=_a(3.0), SZA=_a(30.0), La=_a(350.0), epsf=_a(0.97), epss=_a(0.96),
        fSun=_a(0.6), APAR_Sun=_a(800.0), APAR_Sh=_a(200.0),
        Vcmax25_Sun=_a(60.0), Vcmax25_Sh=_a(30.0),
        Vcmax25_C4Sun=_a(0.0), Vcmax25_C4Sh=_a(0.0),
        ASW_Sun=_a(400.0), ASW_Sh=_a(100.0), ASW_Soil=_a(50.0),
        Ts_bc=_a(300.0), Ca=_a(400.0), Ps=_a(101325.0), Ta=_a(298.0),
        lam=_a(2.5e6), Cp=_a(1005.0), rhoa=_a(1.2), Tv_atm=_a(298.5), q_atm=_a(0.012),
        m=_a(4.0), b0=_a(0.01), alf=_a(0.3), TgC=_a(20.0), fC4=_a(0.0),
        fStress_soil=_a(0.8), ur=_a(3.0), CI=_a(0.75), z0m=_a(0.5), displa=_a(3.0),
        z0=_a(10.0), cv=_a(0.0135), d_leaf=_a(0.025), r_soil_surface=_a(0.0),
    )
    kw.update(over)
    return CanopyForcingBundle(**kw)


def _hydraulic_bundle(psi_s=-0.5, psi50_xylem=-2.0, psi50_g1=-2.0, **over):
    """The same column with the supply fields of the hydraulic mode."""
    return _medlyn_bundle(
        psi_s_mpa=_a(psi_s), rsoil_leaf=_a(0.0), gplant_mol=_a(4e-3),
        psi50_xylem_mpa=_a(psi50_xylem), ck_xylem=_a(1.5),
        psi50_g1_mpa=_a(psi50_g1), ck_g1=_a(1.5), **over)


_CFG = CanopyConfig(stomatal_model="medlyn", tol=1e-8, max_iters=200)
_X0_6 = jnp.array([298.0, 298.0, 280.0, 280.0, 298.0, 0.013])
_X0_8 = jnp.concatenate([_X0_6, jnp.array([0.1, 0.1])])


def _solve(bundle, x0):
    x, _, conv = solve_canopy_closure(x0, bundle, _CFG)
    f = canopy_forward(x, bundle, _CFG.LE_module, _CFG.stomatal_model,
                       _CFG.le_cap_mode, _CFG.use_ta_for_photosynthesis)
    return x, bool(conv), f


def _residual(x, bundle):
    return _canopy_residual(x, bundle, _CFG.LE_module, _CFG.stomatal_model,
                            _CFG.le_cap_mode, _CFG.use_ta_for_photosynthesis)


def test_state_size_follows_the_stress_mode():
    assert state_size("medlyn") == 6
    assert state_size("medlyn", "beta_theta") == 6
    assert state_size("medlyn", "hydraulic") == 8
    assert state_size("phydro", "phydro") == 8


def test_curves_off_reproduce_the_unstressed_medlyn_column():
    """psi50 -> -inf: the hydraulic closure's six shared unknowns solve the
    plain Medlyn column (same root; the draw-down is then only the water the
    free xylem must move).  Two checks: the hydraulic solution restarted in
    the 6-state closure is already its fixed point (no further step), and
    the independently solved 6-state column agrees to the damped Newton's
    own stopping tolerance."""
    x6, c6, f6 = _solve(_medlyn_bundle(), _X0_6)
    x8, c8, f8 = _solve(_hydraulic_bundle(psi50_xylem=_OFF, psi50_g1=_OFF), _X0_8)
    assert c6 and c8
    assert float(x8[6]) > 0.0 and float(x8[7]) > 0.0
    # Same root: polishing both stopped iterates in the plain 6-state closure
    # (a restart near a root converges to it tightly, the LM's tolerance being
    # relative to the restart residual) lands on one point.
    x6p, _, c6p = solve_canopy_closure(x6, _medlyn_bundle(), _CFG)
    x8p, _, c8p = solve_canopy_closure(x8[:6], _medlyn_bundle(), _CFG)
    assert c6p and c8p
    np.testing.assert_allclose(np.asarray(x8p), np.asarray(x6p), rtol=1e-6, atol=1e-8)
    # Independent solves agree to the solver's stopping tolerance (the LM
    # stops on a relative residual norm, ~1e-3 of the K-sized rows).
    np.testing.assert_allclose(np.asarray(x8[:6]), np.asarray(x6), rtol=1e-3, atol=1e-6)
    for k in ("gs_Sun", "gs_Sh", "An_Sun", "An_Sh", "LE_Sun", "LE_Sh", "H_Sun", "H_Sh"):
        np.testing.assert_allclose(float(f8[k]), float(f6[k]), rtol=1e-3, atol=1e-4)


def test_supply_demand_row_is_zero_at_convergence():
    b = _hydraulic_bundle()
    x, conv, f = _solve(b, _X0_8)
    assert conv
    r = np.asarray(_residual(x, b))
    # The damped Newton stops on a relative residual norm; the two rows are in
    # mol H2O/m2/s against demands of ~0.2, so 1e-4 is < 0.1 % of the demand.
    assert abs(r[6]) < 1e-4 and abs(r[7]) < 1e-4
    # The row IS supply + intercept - demand for each class: recompute it from
    # the forward's conductances (mol/m2/s) at the solved draw-downs.
    from legoesm.land.canopy.energy_balance import canopy_met_variables, gs_mol_per_ms
    from legoesm.land.plant_hydraulics import floored_vpd
    _, _, vpd, _, _, _, _ = canopy_met_variables(b.Ps, x[4], x[5])
    d_frac = floored_vpd(vpd) / b.Ps
    for k, cls in ((6, "Sun"), (7, "Sh")):
        hyd = f["dpsi_" + cls]
        assert hyd is not None and float(hyd) == float(x[k])
    from legoesm.land.canopy.solver import _leaf_hydraulics
    hyd_sun, hyd_sh = _leaf_hydraulics(x, b)
    for hyd, gs, tf in ((hyd_sun, f["gs_Sun"], x[0]), (hyd_sh, f["gs_Sh"], x[1])):
        demand = gs * gs_mol_per_ms(tf, b.Ps)
        sd = supply_demand_residual(hyd, d_frac, demand, b.b0)
        assert abs(float(sd)) < 1e-3 * float(demand)


def test_sunlit_leaf_draws_down_more_than_shaded_at_midday():
    x, conv, _ = _solve(_hydraulic_bundle(), _X0_8)
    assert conv
    psi_sun = float(leaf_psi(x[6], _a(-0.5)))
    psi_sh = float(leaf_psi(x[7], _a(-0.5)))
    assert psi_sun < psi_sh < -0.5


def test_conductance_falls_monotonically_with_drying_soil():
    gs = []
    for psi_s in (-0.3, -0.8, -1.3, -1.8):
        x, conv, f = _solve(_hydraulic_bundle(psi_s=psi_s), _X0_8)
        assert conv
        gs.append(float(f["gs_Sun"]))
        # The slope factor is what closes the stomata: W_g1 < 1 on a dry leaf.
        assert 0.0 < float(weibull_stress(leaf_psi(x[6], _a(psi_s)), -2.0, 1.5)) <= 1.0
    assert all(a > b for a, b in zip(gs[:-1], gs[1:]))


def test_beta_theta_closure_is_untouched():
    """The 6-state Medlyn solve neither reads nor needs the supply fields."""
    b = _medlyn_bundle()
    x, conv, f = _solve(b, _X0_6)
    assert conv and x.shape == (6,)
    assert f["dpsi_Sun"] is None and f["dpsi_Sh"] is None
    assert b.psi50_g1_mpa is None and b.psi_s_mpa is None


# ---- land-config / step level -------------------------------------------

def _forcing(ncol):
    ones = jnp.ones(ncol)
    vals = dict(
        sw_down=400.0, lw_down=300.0, precip_total=1e-5, precip_snow=0.0,
        T_lowest=292.0, q_lowest=6e-3, u_lowest=3.0, v_lowest=1.0,
        p_lowest=1e5, p_surface=1.013e5, rho_lowest=1.2, cos_zenith=0.6,
        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0,
    )
    return AtmToSurface(**{k: v * ones for k, v in vals.items()})


def test_hydraulic_baseline_step_runs_and_the_switch_binds():
    """One land step runs finitely under the hydraulic baseline, and on a
    water-limited column its fluxes differ from the beta_theta baseline's
    (the empirical multiplier is replaced by the closure)."""
    cc = CanopyConfig(stomatal_model="medlyn").validate()
    cfg_hy = MultiLayerLandConfig(
        surface_scheme=cc, transpiration_stress="hydraulic",
        plant_hydraulics=PlantHydraulicsConfig(psi50_g1_mpa=-1.5))
    cfg_bt = MultiLayerLandConfig(surface_scheme=cc)
    st = init_multilayer_land_state(2, cfg_hy, theta_init=0.14)
    _, resp_hy, _ = step_multilayer_land(st, _forcing(2), cfg_hy, U_min=1.0, dt=1800.0)
    _, resp_bt, _ = step_multilayer_land(st, _forcing(2), cfg_bt, U_min=1.0, dt=1800.0)
    for r in (resp_hy, resp_bt):
        assert np.all(np.isfinite(np.asarray(r.T_sfc)))
        assert np.all(np.isfinite(np.asarray(r.lhflx)))
    assert not np.allclose(np.asarray(resp_hy.lhflx), np.asarray(resp_bt.lhflx), atol=1e-3)


def test_hydraulic_mode_refusals():
    with pytest.raises(ValueError, match="MEDLYN demand"):
        cfg = MultiLayerLandConfig(
            surface_scheme=CanopyConfig(stomatal_model="ball_berry").validate(),
            transpiration_stress="hydraulic")
        st = init_multilayer_land_state(2, cfg)
        step_multilayer_land(st, _forcing(2), cfg, U_min=1.0, dt=1800.0)
    with pytest.raises(ValueError, match="CLM-ML"):
        cfg = MultiLayerLandConfig(
            surface_scheme=CLMMLCanopyConfig(stomatal_model="medlyn").validate(),
            transpiration_stress="hydraulic")
        st = init_multilayer_land_state(2, cfg)
        step_multilayer_land(st, _forcing(2), cfg, U_min=1.0, dt=1800.0)


def test_lmip_config_accepts_hydraulic_and_refuses_non_medlyn():
    import copy
    from legoesm.land.lmip_config import validate_config
    base = {
        "grid": {"type": "latlon", "resolution": 4},
        "physics": {"land_mode": "multilayer", "surface_scheme": "two_leaf_canopy",
                    "bulk_scheme": "constant", "stomatal_model": "medlyn",
                    "transpiration_stress": "hydraulic"},
        "forcing": {"source": "synthetic", "year_start": 2000, "year_end": 2000},
        "surfdata": {"path": "/tmp/sd.nc"},
        "time": {"dt": 3600.0, "n_steps": 4, "start_doy": 0.0},
        "output": {"tapes": [{"name": "step", "freq": "step",
                              "average": "inst", "vars": ["T_sfc"]}]},
    }
    validate_config(copy.deepcopy(base))
    bad = copy.deepcopy(base)
    bad["physics"]["stomatal_model"] = "ball_berry"
    with pytest.raises(ValueError, match="hydraulic"):
        validate_config(bad)
