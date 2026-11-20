"""Two-leaf canopy ground heat flux handed to the soil with its derivative.

``CanopyConfig.implicit_ground_flux`` (default True since 2026-11-20) gives the final soil
solve ``lambda = -dG/dT_ground`` so its top boundary is semi-implicit, the CLM5
treatment (Technical Description eqs. 2.5.123-2.5.124 for the derivative,
2.6.15 for the linearised flux).  Defect it removes (I. Di Carlo, legoESM-enso
16f7ac09a7): with the explicit flux a converged midday solve over sparse dry canopy
put 290-390 W/m2 into the 2.9 mm top layer and raised it 17-43 K in one 1800 s
step, 24-40 K past the ground temperature the canopy had used.

Tests:
* the soil solve reports the flux it applied, and the soil heat content
  changes by exactly that (sub-stepped too);
* lambda equals a finite difference of the ground flux with the canopy air
  re-solved and the leaves held, CLM5's linearisation;
* with the flag on the land column closes Rn_ext = SH + LE + G_applied to the
  scheme's own residual, and the soil receives G - lambda (T_new - T_ref);
* with the flag off nothing changes: no conductance, no reference, the plain
  explicit solve.
float64.
"""
from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land import multilayer_land as ml
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_thermal import solve_soil_thermal, compute_heat_capacity
from legoesm.land.surface_scheme import TwoLeafCanopyConfig

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="float64 energy gates; run with JAX_ENABLE_X64=1")


def _forcing(n, T=300.0, q=0.008, sw=900.0, u=1.0):
    o = jnp.ones(n)
    ps = 1e5 * o
    return AtmToSurface(
        sw_down=sw * o, lw_down=400.0 * o, precip_total=0 * o, precip_snow=0 * o,
        T_lowest=T * o, q_lowest=q * o, u_lowest=u * o, v_lowest=0 * o,
        p_lowest=0.99 * ps, p_surface=ps, rho_lowest=ps / (constants.R_d * T),
        cos_zenith=0.85 * o, co2_ppmv=340 * o, has_radiation=o,
        has_precipitation=o)


def _cfg(implicit):
    sch = TwoLeafCanopyConfig(max_iters=10, implicit_ground_flux=implicit)
    return MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, dz_top=0.025, growth_factor=2.0,
                                 total_depth=3.0),
        surface_scheme=sch)


@pytest.mark.parametrize("n_sub", [1, 3])
def test_soil_solve_applies_the_flux_it_reports(n_sub):
    cfg = _cfg(True)
    grid = make_soil_grid(cfg.soil_grid)
    n = 3
    T0 = jnp.linspace(300.0, 320.0, 10)[None, :] * jnp.ones((n, 1))
    theta = jnp.full((n, 10), 0.15)
    G = jnp.array([350.0, -120.0, 40.0])
    lam = jnp.array([12.0, 30.0, 0.0])
    T1, robin = solve_soil_thermal(T0, theta, grid, cfg.hydraulics, cfg.thermal, G,
                                   1800.0, surface_conductance=lam, n_substeps=n_sub,
                                   return_robin_flux=True)
    Gap = G - robin
    T_plain = solve_soil_thermal(T0, theta, grid, cfg.hydraulics, cfg.thermal, G,
                                 1800.0, surface_conductance=lam, n_substeps=n_sub)
    np.testing.assert_array_equal(np.asarray(T1), np.asarray(T_plain))
    C = compute_heat_capacity(theta, cfg.hydraulics, cfg.thermal) * grid.dz[None, :]
    dE = jnp.sum(C * (T1 - T0), axis=1) / 1800.0
    np.testing.assert_allclose(np.asarray(dE),
                               np.asarray(Gap + cfg.thermal.Q_geothermal),
                               rtol=0, atol=1e-8)
    # lambda = 0 column applies G unchanged; lambda > 0 columns differ from G
    assert float(Gap[2]) == float(G[2])
    assert np.all(np.abs(np.asarray(Gap - G)[:2]) > 1.0)


def _step(monkeypatch, implicit, T_init=307.0, n=4, u=(0.7, 1.4, 2.8, 3.6)):
    cfg = _cfg(implicit)
    st = ml.init_multilayer_land_state(n, cfg, T_init=T_init, theta_init=0.12)
    f = _forcing(n)._replace(u_lowest=jnp.asarray(u))
    seen = {}
    real = ml.solve_soil_thermal
    sig = inspect.signature(real)

    def spy(*a, **k):
        args = sig.bind(*a, **k).arguments
        out = real(*a, **k)
        seen["G_in"] = args["G_surface"]
        seen["lam"] = args.get("surface_conductance")
        seen["ret"] = args.get("return_robin_flux", False)
        seen["G_applied"] = (args["G_surface"] - out[1]) if seen["ret"] else args["G_surface"]
        return out

    monkeypatch.setattr(ml, "solve_soil_thermal", spy)
    step = jax.jit(lambda s: ml.step_multilayer_land_with_diagnostics(
        s, f, cfg, 1.0, 1800.0, lat=jnp.full(n, -0.55))
        + (seen["G_applied"], seen["G_in"]))
    new, r, _c, so, G_ap, G_in = step(st)
    return st, new, r, so, G_ap, G_in, seen


def test_column_energy_closes_with_the_implicit_flux(monkeypatch):
    st, new, r, so, G_ap, G_in, seen = _step(monkeypatch, True)
    assert seen["ret"] and seen["lam"] is not None
    assert bool(jnp.all(so.converged)) and int(so.n_held) == 0
    lam = np.asarray(so.surface_conductance)
    assert np.all(np.isfinite(lam)) and np.all(lam > 0.0)
    # the land boundary closes to the scheme's own residual with the flux the
    # soil actually received
    E = so.Rn_ext - r.shflx - r.lhflx - G_ap
    np.testing.assert_allclose(np.asarray(E), np.asarray(so.residual_ext),
                               rtol=0, atol=1e-9)
    # applied = G_scheme - lambda (T_new_top - T_ref) for one sub-step solves
    # (freeze/thaw off in this config -> one sub-step)
    want = so.G_soil - so.surface_conductance * (new.T_soil[:, 0]
                                                 - so.surface_conductance_T_ref)
    np.testing.assert_allclose(np.asarray(G_ap), np.asarray(want), rtol=0, atol=1e-9)


def test_flag_off_is_the_explicit_path(monkeypatch):
    st, new, r, so, G_ap, G_in, seen = _step(monkeypatch, False)
    assert so.surface_conductance is None and so.surface_conductance_T_ref is None
    assert seen["lam"] is None and not seen["ret"]
    np.testing.assert_array_equal(np.asarray(G_ap), np.asarray(G_in))
    # and the fix changes the answer (non-vacuity of the flag)
    _st, new1, r1, *_ = _step(monkeypatch, True)
    assert not np.array_equal(np.asarray(new1.T_soil), np.asarray(new.T_soil))


def test_default_is_the_implicit_flux():
    assert TwoLeafCanopyConfig().implicit_ground_flux is True


def test_conductance_matches_a_finite_difference():
    """lambda against central differences of G with the canopy air re-solved
    (Newton on the two canopy-air rows) and the leaves held fixed."""
    from legoesm.land.canopy import solver as S
    cc = TwoLeafCanopyConfig(max_iters=60, implicit_ground_flux=True)
    x, bundle = _converged_column(cc)

    def G_at(ts):
        b = bundle._replace(Ts_bc=ts)

        def Fa(xa):
            return S._canopy_residual(
                x.at[4:6].set(xa), b, LE_module=cc.LE_module,
                stomatal_model=cc.stomatal_model, le_cap_mode=cc.le_cap_mode,
                use_ta_for_photosynthesis=cc.use_ta_for_photosynthesis,
                rh_cap_width=cc.rh_cap_smoothing_width,
                zeta_cap_width=cc.zeta_cap_smoothing_width,
                most_n_iters=cc.most_n_iters)[4:6]
        xa = x[4:6]
        for _ in range(30):
            xa = xa - jnp.linalg.solve(jax.jacfwd(Fa)(xa), Fa(xa))
        return S.canopy_forward(
            x.at[4:6].set(xa), b, cc.LE_module, cc.stomatal_model, cc.le_cap_mode,
            cc.use_ta_for_photosynthesis, cc.rh_cap_smoothing_width,
            cc.zeta_cap_smoothing_width, cc.most_n_iters)["G"]

    h = 1e-3
    ts = bundle.Ts_bc
    lam_fd = -(G_at(ts + h) - G_at(ts - h)) / (2 * h)
    lam = S.ground_heat_conductance(x, bundle, cc, jnp.array(True))
    assert float(lam) > 0.0
    np.testing.assert_allclose(float(lam), float(lam_fd), rtol=1e-5)
    # the partial (canopy air held) is larger: the canopy air follows the ground
    lam_part = S.ground_heat_conductance(x, bundle, cc, jnp.array(False))
    assert float(lam_part) > float(lam)


def _converged_column(cc):
    """One converged column with leaves (LAI 1.0, C4 grass), the bundle built
    the way test_canopy_solver_grad builds it."""
    from legoesm.land.canopy import CanopyLandParams
    from legoesm.land.canopy.radiative_transfer import (
        split_sw_components, canopy_shortwave_rt)
    from legoesm.land.canopy.stability import compute_aerodynamics, sat_specific_humidity
    from legoesm.land.canopy.solver import CanopyForcingBundle, solve_canopy_closure
    from legoesm.land.surface_scheme import two_leaf_canopy as tlc
    from legoesm.thermo import latent_heat_vaporization
    land_config = MultiLayerLandConfig(surface_scheme=cc)
    f = _forcing(1, u=1.5)
    p = CanopyLandParams(
        LAI=jnp.full(1, 1.0), hc=jnp.full(1, 0.5), fC4=jnp.zeros(1),
        FNonVeg=jnp.zeros(1), CI=jnp.full(1, 0.75), kn=jnp.full(1, 0.3),
        Vcmax25_C3_leaf=jnp.full(1, 60.0), Vcmax25_C4_leaf=jnp.full(1, 40.0),
        m_C3=jnp.full(1, 9.0), m_C4=jnp.full(1, 4.0), b0_C3=jnp.full(1, 0.01),
        b0_C4=jnp.full(1, 0.04), alf=jnp.full(1, 0.3), TgC=jnp.full(1, 25.0),
        ALB_VIS=jnp.full(1, 0.15), ALB_NIR=jnp.full(1, 0.3),
        emissivity=jnp.full(1, 0.97), rz0m=jnp.full(1, 0.055), rd=jnp.full(1, 0.67))
    Ts = jnp.full(1, 312.0)
    wind = jnp.full(1, 1.5)
    z0m, displa = compute_aerodynamics(p.hc, p.LAI, p.rz0m, p.rd)
    z_ref = jnp.maximum(jnp.full(1, land_config.z_ref), displa + 10.0 * z0m + 2.0)
    SZA = jnp.degrees(jnp.arccos(jnp.clip(f.cos_zenith, 0.0, 1.0)))
    sw = canopy_shortwave_rt(*split_sw_components(f.sw_down, f.cos_zenith), SZA,
                             p.LAI, p.CI, p.ALB_VIS, p.ALB_NIR, p.Vcmax25_C3_leaf,
                             p.Vcmax25_C4_leaf, p.kn)
    Ta, Ps, q_atm, Ca = f.T_lowest, f.p_surface, f.q_lowest, f.co2_ppmv
    q_c = 0.5 * (sat_specific_humidity(Ts, Ps) + q_atm)
    Ci = Ca * tlc._CI_CA_C3
    x0 = jnp.stack([Ta, Ta, Ci, Ci, Ta, q_c], axis=-1)
    b = CanopyForcingBundle(
        LAI=p.LAI, SZA=SZA, La=f.lw_down, epsf=jnp.full(1, cc.epsf),
        epss=jnp.full(1, cc.epss), fSun=sw.fSun, APAR_Sun=sw.APAR_Sun,
        APAR_Sh=sw.APAR_Sh, Vcmax25_Sun=sw.Vcmax25_C3Sun, Vcmax25_Sh=sw.Vcmax25_C3Sh,
        Vcmax25_C4Sun=sw.Vcmax25_C4Sun, Vcmax25_C4Sh=sw.Vcmax25_C4Sh,
        ASW_Sun=sw.ASW_Sun, ASW_Sh=sw.ASW_Sh, ASW_Soil=sw.ASW_Soil, Ts_bc=Ts, Ca=Ca,
        Ps=Ps, Ta=Ta, lam=latent_heat_vaporization(Ts), Cp=jnp.full(1, constants.c_pd),
        rhoa=f.rho_lowest, Tv_atm=Ta * (1.0 + tlc._VIRT_T_COEF * q_atm), q_atm=q_atm,
        m=p.m_C3, b0=p.b0_C3, alf=p.alf, TgC=p.TgC, fC4=p.fC4,
        fStress_soil=jnp.full(1, 0.3), ur=wind, CI=p.CI, z0m=z0m, displa=displa,
        z0=z_ref, cv=jnp.full(1, cc.cv), d_leaf=jnp.full(1, tlc._DEFAULT_D_LEAF),
        r_soil_surface=jnp.zeros(1), fwet=jnp.zeros(1))
    x, _n_it, conv = jax.vmap(lambda x_, b_: solve_canopy_closure(x_, b_, cc))(x0, b)
    assert bool(jnp.all(conv))
    return x[0], jax.tree_util.tree_map(lambda a: a[0], b)


def test_config_rejects_a_non_bool_flag():
    with pytest.raises(ValueError, match="implicit_ground_flux"):
        TwoLeafCanopyConfig(implicit_ground_flux=1).validate()
