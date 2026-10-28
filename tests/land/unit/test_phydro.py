"""P-hydro (Joshi profit strategy on a vulnerable xylem): pure-module behaviour.

Parity with the rphydro R package, optimality certificate (brute-force grid),
drought response, Weibull guards, slope mappings for all three stomatal
models, implicit-gradient correctness, loud no-state error.

Parity, finite-difference and certificate tests need ``JAX_ENABLE_X64=1``;
the drought, finiteness and gradient-finiteness tests also run in float32,
the model's default precision.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land import phydro as ph
from legoesm.land.canopy.photosynthesis import (
    jmax_temperature_response,
    vcmax_temperature_response,
)
from legoesm.land.p_model import (
    PModelAcclimState,
    PModelConfig,
    init_pmodel_acclim,
    optimal_chi,
)
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig

needs_x64 = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs JAX_ENABLE_X64=1")

# rpmodel + rphydro reference table (scripts/validate/optimality_reference.R).
_REFERENCE = Path(__file__).parent / "data" / "optimality_reference.csv"


def _acclim(ncol=1, t_c=25.0, vpd=1000.0):
    a = init_pmodel_acclim(ncol, t_init_K=constants.T_freeze + t_c,
                           ps_init_pa=constants.p_atm_std,
                           cfg=PModelConfig())
    return a._replace(vpd_mean_pa=jnp.full((ncol,), vpd))


def _supply(psi_s=-0.2, rsoil=0.0, ncol=1):
    return ph.PhydroSupply(
        rsoil_leaf=jnp.full((ncol,), rsoil),
        psi_s_mpa=jnp.full((ncol,), psi_s))


def _soil_column(psi_m=-5.0, nlev=6, ncol=2):
    hyd = SoilHydraulicsConfig()
    psi = jnp.full((ncol, nlev), psi_m)
    theta = jnp.full((ncol, nlev), 0.3)
    dz = jnp.full((nlev,), 0.25)
    rf = jnp.full((ncol, nlev), 1.0 / nlev)
    lai = jnp.full((ncol,), 3.0)
    return psi, theta, dz, rf, lai, hyd


@needs_x64
def test_supply_tracks_soil_potential_and_resistance_rises_when_dry():
    cfg = ph.PHydroConfig()
    mpa_per_m = constants.rho_water * constants.g * 1e-6
    psi, theta, dz, rf, lai, hyd = _soil_column(psi_m=-1.0)  # wet (~-0.01 MPa)
    sup = ph.soil_root_supply(psi, theta, dz, rf, lai, hyd, cfg)
    # The soil-to-root resistance is masked to zero (rphydro supply).
    assert np.all(np.asarray(sup.rsoil_leaf) == 0.0)
    # Wetter than field capacity (and the Richards lane's positive overshoot
    # on rain) is capped smoothly at the cap, with a finite gradient.
    cap, width = ph._PSI_SOIL_CAP_MPA, ph._PSI_SOIL_CAP_WIDTH_MPA
    for wet_m in (-1.0, 350.0):  # ~-0.01 MPa and ~+3.4 MPa
        psi_w = jnp.full_like(psi, wet_m)
        psi_s = float(ph.soil_root_supply(psi_w, theta, dz, rf, lai, hyd, cfg).psi_s_mpa[0])
        assert cap - width < psi_s <= cap
        g = jax.grad(lambda p: ph.soil_root_supply(
            p, theta, dz, rf, lai, hyd, cfg).psi_s_mpa[0])(psi_w)
        assert np.all(np.isfinite(np.asarray(g)))
    # Very dry soil: the bulk potential is the soil's own (no floor).
    psi_d, theta_d, *_ = _soil_column(psi_m=-300.0)  # ~ -2.9 MPa
    sup_d = ph.soil_root_supply(psi_d, jnp.full_like(theta_d, 0.12), dz, rf,
                                lai, hyd, cfg)
    np.testing.assert_allclose(float(sup_d.psi_s_mpa[0]), -300.0 * mpa_per_m, rtol=1e-12)
    assert float(sup_d.rsoil_leaf[0]) == 0.0
    # A dried-out layer drops out of the bulk potential through its own
    # collapsing conductance: one wet layer dominates five dry ones.
    psi_mix = psi_d.at[:, -1].set(-1.0)
    theta_mix = jnp.full_like(theta_d, 0.12).at[:, -1].set(0.3)
    sup_mix = ph.soil_root_supply(psi_mix, theta_mix, dz, rf, lai, hyd, cfg)
    assert float(sup_mix.psi_s_mpa[0]) > -0.5


def _reference_case(alpha):
    """The reference grid run through phydro_optimum in rphydro's own setting:
    no soil resistance, gamma = 1."""
    pd = pytest.importorskip("pandas")
    ref = pd.read_csv(_REFERENCE)
    col = lambda c: jnp.asarray(ref[c].to_numpy(), dtype=jnp.float64)
    n = len(ref)
    tk = col("tc") + constants.T_freeze
    acclim = PModelAcclimState(
        iabs_mean=col("ppfd"), t_mean_K=tk, vpd_mean_pa=col("vpd"),
        co2_mean_ppm=jnp.full((n,), 400.0),
        ps_ema=jnp.full((n,), constants.p_atm_std))
    sup = ph.PhydroSupply(rsoil_leaf=jnp.zeros((n,)), psi_s_mpa=col("psi_soil"))
    cfg = ph.PHydroConfig(
        gplant_mol=col("k_scaled"), psi50_xylem_mpa=col("psi50"), ck_xylem=1.5,
        alpha_cost=alpha, gamma_cost=1.0)
    caps = ph.phydro_optimum(acclim, sup, cfg, PModelConfig())
    tg_c = tk - constants.T_freeze
    vcmax = caps.vcmax25_leaf * vcmax_temperature_response(tk, tg_c)
    jmax = caps.rjv25 * caps.vcmax25_leaf * jmax_temperature_response(tk, tg_c)
    return ref, caps, np.asarray(vcmax), np.asarray(jmax)


@needs_x64
def test_reproduces_rphydro_reference():
    """With no soil resistance the optimum IS rphydro's
    (jaideep777/phydro @a59e2c6, acclimated analytical solver), on every row of
    the T x PPFD x VPD x soil-psi x psi50 grid."""
    ref, caps, vcmax, jmax = _reference_case(alpha=0.1)
    assert bool(np.all(np.asarray(caps.converged)))
    np.testing.assert_allclose(np.asarray(caps.chi), ref.r2acc_chi, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(caps.dpsi_mpa), ref.r2acc_dpsi, rtol=1e-3)
    np.testing.assert_allclose(jmax, ref.r2acc_jmax, rtol=1e-3)
    # Vcmax carries the repo's own Michaelis-Menten constants (~1% from
    # rphydro's literals; module departure 4).
    np.testing.assert_allclose(vcmax, ref.r2acc_vcmax, rtol=2e-2)


@needs_x64
def test_reference_parity_needs_the_jmax_cost():
    """Non-vacuity of the parity test: a different alpha must NOT match."""
    ref, caps, _, jmax = _reference_case(alpha=0.05)
    assert np.median(np.abs(jmax / ref.r2acc_jmax - 1.0)) > 0.05
    assert np.median(np.abs(np.asarray(caps.chi) / ref.r2acc_chi - 1.0)) > 0.01


@needs_x64
@pytest.mark.parametrize("psi_s", [-0.05, -0.6, -1.0])
@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_optimum_dominates_brute_force_grid(psi_s, gamma):
    """Optimality certificate with the Jmax stress ON (no external reference
    exists for it): the returned (chi, dpsi) beats every point of a dense
    grid of the same profit."""
    cfg = ph.PHydroConfig(gamma_cost=gamma)
    pcfg = PModelConfig()
    acclim = _acclim()
    sup = _supply(psi_s=psi_s, rsoil=20.0)
    caps = ph.phydro_optimum(acclim, sup, cfg, pcfg)
    _, _, _, gamma_star, _ = optimal_chi(
        acclim.t_mean_K, acclim.vpd_mean_pa, acclim.co2_mean_ppm,
        acclim.ps_ema, pcfg)
    ca_pa = acclim.co2_mean_ppm * acclim.ps_ema * 1e-6
    env = ph._Env(
        psi_s=sup.psi_s_mpa, rsoil=sup.rsoil_leaf,
        d_frac=acclim.vpd_mean_pa / acclim.ps_ema,
        ca_umol=acclim.co2_mean_ppm, g=gamma_star / ca_pa,
        jlim=4.0 * ph._phi0(acclim.t_mean_K, pcfg) * acclim.iabs_mean)
    p_star = float(ph._profit(caps.chi, caps.dpsi_mpa, env, cfg)[0])
    assert p_star > 0.0
    best = -np.inf
    for dpsi in np.linspace(1e-3, 6.0, 240):
        d = jnp.full((1,), float(dpsi))
        gs = ph._supply_side(d, env, cfg)
        lo = float(ph._chi_min(gs, env)[0])
        chis = jnp.linspace(lo + 1e-6, 1.0, 400)
        vals = np.asarray(ph._profit(chis, d, env, cfg))
        best = max(best, float(np.nanmax(vals)))
    assert p_star >= best - 1e-4 * max(abs(best), 1.0)


def test_drought_response_is_gradual_and_shuts_down():
    """Capacities fall monotonically and smoothly with drying soil (no cliff
    at a prescribed leaf-potential floor), and a column whose capacity can no
    longer pay for itself is shut down with finite outputs."""
    cfg = ph.PHydroConfig()
    pcfg = PModelConfig()
    psi = np.array([-0.05, -0.2, -0.4, -0.6, -0.8, -1.0, -3.0, -30.0])
    n = len(psi)
    caps = ph.phydro_optimum(
        _acclim(ncol=n),
        ph.PhydroSupply(rsoil_leaf=jnp.zeros((n,)), psi_s_mpa=jnp.asarray(psi)),
        cfg, pcfg)
    v = np.asarray(caps.vcmax25_leaf)
    for field in caps:
        assert np.all(np.isfinite(np.asarray(field, dtype=float)))
    assert np.all(np.diff(v) <= 1e-9)
    assert v[0] > 20.0
    # Gradual: no single step of the wet-to-dry sweep removes most of the capacity.
    assert np.max(-np.diff(v[:6])) < 0.5 * v[0]
    # rphydro closure (gamma = 1, no soil resistance): still active at -3 MPa
    # (the xylem keeps some conductance; ~80% of the wet capacity), shut down
    # at -30 MPa.
    assert v[-1] < 1e-6 and 0.0 < v[-2] < v[0]
    assert np.all(np.asarray(caps.psi_leaf_mpa) <= psi + 1e-12)


def test_gamma_reduces_drawdown():
    pcfg = PModelConfig()
    acclim = _acclim()
    d1 = float(ph.phydro_optimum(
        acclim, _supply(), ph.PHydroConfig(gamma_cost=0.5), pcfg).dpsi_mpa[0])
    d2 = float(ph.phydro_optimum(
        acclim, _supply(), ph.PHydroConfig(gamma_cost=5.0), pcfg).dpsi_mpa[0])
    assert d2 < d1


@needs_x64
def test_weibull_curve_and_its_gradients_are_finite_everywhere():
    """P = 0.5 at psi50, and no NaN/inf in the value or in the gradient w.r.t.
    psi, psi50 and ck at psi = 0 (0*log 0), at psi50, and deep in the flat
    tail where dP/dpsi underflows to zero."""
    np.testing.assert_allclose(float(ph.weibull_vulnerability(-2.0, -2.0, 1.5)), 0.5)
    np.testing.assert_allclose(float(ph.weibull_vulnerability(0.0, -2.0, 1.5)), 1.0)
    for psi in (0.0, -1e-9, -2.0, -50.0, -1.0e4):
        for ck in (1.0, 1.5, 6.0):
            val, grads = jax.value_and_grad(
                ph.weibull_vulnerability, argnums=(0, 1, 2))(psi, -2.0, ck)
            assert np.isfinite(float(val))
            assert all(np.isfinite(float(g)) for g in grads)


_GRAD_ARGS = ("alpha", "gamma", "gplant", "psi50_x", "ck_x",
              "psi_s", "rsoil", "vpd", "t_k", "iabs", "co2")
_GRAD_BASE = dict(alpha=0.1, gamma=0.3, gplant=4e-3, psi50_x=-2.0, ck_x=1.5,
                  psi_s=-0.3, rsoil=20.0, vpd=1200.0,
                  t_k=constants.T_freeze + 22.0, iabs=700.0, co2=400.0)


def _grad_loss(*x):
    a = dict(zip(_GRAD_ARGS, x))
    one = lambda v: jnp.reshape(jnp.asarray(v, dtype=float), (1,))
    acclim = PModelAcclimState(
        iabs_mean=one(a["iabs"]), t_mean_K=one(a["t_k"]),
        vpd_mean_pa=one(a["vpd"]), co2_mean_ppm=one(a["co2"]),
        ps_ema=one(constants.p_atm_std))
    cfg = ph.PHydroConfig(
        gplant_mol=a["gplant"], psi50_xylem_mpa=a["psi50_x"], ck_xylem=a["ck_x"],
        alpha_cost=a["alpha"], gamma_cost=a["gamma"])
    caps = ph.phydro_optimum(
        acclim, ph.PhydroSupply(rsoil_leaf=one(a["rsoil"]), psi_s_mpa=one(a["psi_s"])),
        cfg, PModelConfig())
    return (caps.vcmax25_leaf[0] + 10.0 * caps.rjv25[0]
            + 30.0 * caps.chi[0] + 20.0 * caps.dpsi_mpa[0])


@needs_x64
@pytest.mark.parametrize("override", [{}, {"psi_s": -0.9}, {"gamma": 1.0}])
def test_implicit_gradients_match_finite_differences(override):
    """IFT gradients of every output w.r.t. every input and parameter equal
    central finite differences at an interior optimum."""
    base = {**_GRAD_BASE, **override}
    x = [base[k] for k in _GRAD_ARGS]
    grads = jax.grad(_grad_loss, argnums=tuple(range(len(x))))(*x)
    for i, name in enumerate(_GRAD_ARGS):
        h = 1e-5 * max(abs(x[i]), 1e-3)
        up = [*x]; up[i] = x[i] + h
        dn = [*x]; dn[i] = x[i] - h
        fd = (float(_grad_loss(*up)) - float(_grad_loss(*dn))) / (2.0 * h)
        np.testing.assert_allclose(
            float(grads[i]), fd, rtol=2e-3, atol=1e-6 * max(1.0, abs(fd)),
            err_msg=f"d/d{name}")


@pytest.mark.parametrize("psi_s", [0.0, -1.3, -3.0, -100.0])
def test_gradients_finite_on_wet_and_shut_down_columns(psi_s):
    """Saturated soil (psi = 0), the shut-down threshold and bone-dry soil
    (vulnerability curves flat at 0, Hessian singular): gradients are finite
    — zero where the plant is shut down — never NaN/inf."""
    x = [{**_GRAD_BASE, "psi_s": psi_s, "gamma": 0.0}[k] for k in _GRAD_ARGS]
    grads = jax.grad(_grad_loss, argnums=tuple(range(len(x))))(*x)
    assert all(np.isfinite(float(g)) for g in grads)


def test_slope_mappings_reproduce_target_gs():
    """Each model's gs with the mapped slope equals 1.6*A/(ca*(1-chi))."""
    from legoesm.land.stomata import ball_berry_gs, leuning_gs, medlyn_gs
    from legoesm.thermo import saturation_vapor_pressure
    acclim = _acclim()
    pcfg = PModelConfig()
    chi = jnp.full((1,), 0.75)
    _, _, _, gamma_star, _ = optimal_chi(
        acclim.t_mean_K, acclim.vpd_mean_pa, acclim.co2_mean_ppm,
        acclim.ps_ema, pcfg)
    A = jnp.full((1,), 12.0)
    ca_ppm = acclim.co2_mean_ppm
    gs_target = 1.6 * A / (ca_ppm * (1.0 - chi))
    d_kpa = acclim.vpd_mean_pa / 1000.0
    es = saturation_vapor_pressure(acclim.t_mean_K)
    rh = 1.0 - acclim.vpd_mean_pa / es
    gamma_ppm = gamma_star / (1e-6 * acclim.ps_ema)

    g1 = ph.slope_for_model("medlyn", chi, acclim, gamma_star)
    gs_med = medlyn_gs(A, d_kpa, ca_ppm, g1, 0.0)
    np.testing.assert_allclose(np.asarray(gs_med), np.asarray(gs_target),
                               rtol=1e-6)
    m = ph.slope_for_model("ball_berry", chi, acclim, gamma_star)
    gs_bb = ball_berry_gs(A, rh, ca_ppm, m, 0.0)
    np.testing.assert_allclose(np.asarray(gs_bb), np.asarray(gs_target),
                               rtol=1e-6)
    a1 = ph.slope_for_model("leuning", chi, acclim, gamma_star,
                            d0_leuning_kpa=1.5)
    gs_le = leuning_gs(A, d_kpa, ca_ppm, gamma_ppm, a1, 1.5, 0.0)
    np.testing.assert_allclose(np.asarray(gs_le), np.asarray(gs_target),
                               rtol=1e-6)
    with pytest.raises(ValueError, match="stomatal_model"):
        ph.slope_for_model("wue", chi, acclim, gamma_star)
    with pytest.raises(ValueError, match="d0_leuning"):
        ph.slope_for_model("leuning", chi, acclim, gamma_star)


def test_no_state_raises():
    with pytest.raises(ValueError, match="acclimation state"):
        ph.phydro_optimum(None, _supply(), ph.PHydroConfig(), PModelConfig())


def _rphydro_assim(vcmax, jmax, iabs, tk, pcfg):
    """rphydro's instantaneous assimilation (umol/m2/s) as a function of ci
    [umol/mol]: the smaller of the Rubisco- and light-limited rates, Smith
    light response, no dark respiration (the reference table's rdark = 0)."""
    from legoesm.land.p_model import _phi0
    _, _, _, gamma_star_pa, k_pa = optimal_chi(
        tk, jnp.full_like(tk, 1000.0), jnp.full_like(tk, 400.0),
        jnp.full_like(tk, constants.p_atm_std), pcfg)
    to_ppm = 1e6 / constants.p_atm_std
    gstar, km = gamma_star_pa * to_ppm, k_pa * to_ppm
    phi0_i = _phi0(tk, pcfg) * iabs
    jj = 4.0 * phi0_i / jmax
    jlim = phi0_i / jnp.sqrt(1.0 + jj * jj)

    def assim(ci):
        ac = vcmax * (ci - gstar) / (ci + km)
        aj = jlim * (ci - gstar) / (ci + 2.0 * gstar)
        return jnp.minimum(ac, aj)
    return assim


def _instantaneous_solve(assim, hyd, d_frac, ca, n_iter=60):
    """Root of the instantaneous stationarity condition by bisection on
    dpsi in [0, 20] MPa with the diffusion balance solved inside (test-side
    reference solver; the model solves the same condition inside the canopy
    Newton closure).  ``fori_loop`` so the trace stays small."""
    def ci_of(dpsi):
        h = hyd._replace(dpsi_mpa=dpsi)
        gs_c = ph.leaf_supply_gs_w(h, d_frac) / 1.6

        def body(_, br):   # assim(ci) - gs_c (ca - ci) is increasing in ci
            lo, hi = br
            mid = 0.5 * (lo + hi)
            up = assim(mid) - gs_c * (ca - mid) < 0.0
            return jnp.where(up, mid, lo), jnp.where(up, hi, mid)
        lo, hi = jax.lax.fori_loop(0, n_iter, body, (jnp.zeros_like(ca), ca))
        return 0.5 * (lo + hi), gs_c

    def foc(dpsi):
        ci, gs_c = ci_of(dpsi)
        h = hyd._replace(dpsi_mpa=dpsi)
        an_prime = jax.jvp(assim, (ci,), (jnp.ones_like(ci),))[1]
        return ph.instantaneous_foc(h, d_frac, an_prime, ci, ca, 1.6 * gs_c)

    def body(_, br):
        lo, hi = br
        mid = 0.5 * (lo + hi)
        rising = foc(mid) > 0.0
        return jnp.where(rising, mid, lo), jnp.where(rising, hi, mid)
    lo, hi = jax.lax.fori_loop(
        0, n_iter, body, (jnp.zeros_like(ca), jnp.full_like(ca, 20.0)))
    dpsi = 0.5 * (lo + hi)
    ci, gs_c = ci_of(dpsi)
    return dpsi, ci / ca, gs_c


@needs_x64
def test_instantaneous_reproduces_rphydro_reference():
    """The instantaneous stationarity condition (instantaneous_foc) with
    rphydro's own assimilation form reproduces rphydro_instantaneous_
    analytical on every reference row: capacities from the weekly (VPD =
    1 kPa) acclimated row of the same (T, PPFD, soil psi, psi50), this row's
    instantaneous VPD, no soil resistance, gamma = 1."""
    pd = pytest.importorskip("pandas")
    ref = pd.read_csv(_REFERENCE)
    key = ["tc", "ppfd", "psi_soil", "psi50"]
    weekly = ref[ref.vpd == 1000].set_index(key)[["r2acc_vcmax", "r2acc_jmax"]]
    ref = ref.join(weekly, on=key, rsuffix="_wk")
    # The VPD = 1 kPa rows ARE the acclimated conditions, where coordination
    # makes A_c = A_j exactly: the limiting-rate assimilation has a derivative
    # kink at the optimum, rphydro's finite-difference derivative straddles it
    # and this solver takes one side (0.05-0.12 MPa apart at 30 degC).  Off
    # that measure-zero point the two agree.
    ref = ref[np.isfinite(ref.r2ins_chi) & np.isfinite(ref.r2acc_vcmax_wk)
              & (ref.r2ins_dpsi < 19.0) & (ref.vpd != 1000)].reset_index(drop=True)
    assert len(ref) > 800
    col = lambda c: jnp.asarray(ref[c].to_numpy(), dtype=jnp.float64)
    n = len(ref)
    tk = col("tc") + constants.T_freeze
    assim = _rphydro_assim(col("r2acc_vcmax_wk"), col("r2acc_jmax_wk"),
                           col("ppfd"), tk, PModelConfig())
    hyd = ph.LeafHydraulics(
        dpsi_mpa=jnp.zeros((n,)), psi_s_mpa=col("psi_soil"),
        rsoil_leaf=jnp.zeros((n,)), lai_class=jnp.ones((n,)),
        gplant_mol=col("k_scaled"), psi50_xylem_mpa=col("psi50"),
        ck_xylem=jnp.full((n,), 1.5), gamma_cost=jnp.ones((n,)))
    dpsi, chi, gs_c = _instantaneous_solve(
        assim, hyd, col("vpd") / constants.p_atm_std, jnp.full((n,), 400.0))
    np.testing.assert_allclose(np.asarray(dpsi), ref.r2ins_dpsi, rtol=1e-2, atol=1e-4)
    np.testing.assert_allclose(np.asarray(chi), ref.r2ins_chi, rtol=1e-2)
    np.testing.assert_allclose(np.asarray(gs_c), ref.r2ins_gs, rtol=1e-2, atol=1e-6)


def test_instantaneous_foc_closes_stomata_without_gain():
    """A leaf with nothing to gain (A' = 0, e.g. night) has its stationarity
    residual equal to -2 gamma dpsi: the only root is dpsi = 0, with a
    well-conditioned slope."""
    one = jnp.ones((1,))
    hyd = ph.LeafHydraulics(
        dpsi_mpa=0.7 * one, psi_s_mpa=-0.5 * one, rsoil_leaf=0.0 * one,
        lai_class=2.0 * one, gplant_mol=4e-3 * one, psi50_xylem_mpa=-2.0 * one,
        ck_xylem=1.5 * one, gamma_cost=1.0 * one)
    d = 0.01 * one
    gs_w = ph.leaf_supply_gs_w(hyd, d)
    r = ph.instantaneous_foc(hyd, d, 0.0 * one, 300.0 * one, 400.0 * one, gs_w)
    np.testing.assert_allclose(float(r[0]), -2.0 * 0.7, rtol=1e-12)
    slope = jax.jvp(lambda dp: ph.instantaneous_foc(
        hyd._replace(dpsi_mpa=dp), d, 0.0 * one, 300.0 * one, 400.0 * one, gs_w),
        (hyd.dpsi_mpa,), (one,))[1]
    np.testing.assert_allclose(float(slope[0]), -2.0, rtol=1e-12)
    # The supply is linear in a small draw-down (no floor: the canopy solver
    # projects its trials onto dpsi >= 0).
    g2 = ph.leaf_supply_gs_w(hyd._replace(dpsi_mpa=0.02 * one), d)
    g1 = ph.leaf_supply_gs_w(hyd._replace(dpsi_mpa=0.01 * one), d)
    np.testing.assert_allclose(float(g2[0] / g1[0]), 2.0, rtol=1e-2)
