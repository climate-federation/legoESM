"""Oracle match: radiation droplet/ice effective radius on CAM6 MG2's in-cloud footing.

Production path (MPAS ``_run_column``): ``radiation.integration._make_hydrostatic_radiation``
physics_fn overrides the droplet number with the aerosol-activated (specified, IN-CLOUD)
number and calls ``_call_radiation_backend(..., n_cloud_in_cloud=True)`` ->
``compute_cloud_properties``.  CAM6 forms REL from that kind of number (MG2 ``nccons``)
with the in-cloud water ``qc / max(ast, mincld)``; before #1521 was fixed legoESM paired it
with the grid-mean q_c, so r_eff came out low by cf^(1/3).

Fixture ``oracle_fixtures/reff_incloud.npz``: outputs from RUNNING CAM cam_cesm2_1_rel_60
(a03b84b7c4e34f965b115686f22a043b85739e56) micro_mg2_0.F90 L877-879, 2739-2768,
2772-2808, 2811-2865 and micro_mg_cam.F90 L2296, 2534-2554 (the grid REL radiation reads)
verbatim with micro_mg_utils.F90 compiled unmodified (dcs = 500e-6).  Inputs: the 5 tier-1
CAM6-L32 columns with ast < mincld edge levels, cap-bound levels, a grid q_c below 1e-14
whose in-cloud value is above qsmall, two dead-droplet-number levels (oracle number =
CloudConfig.Nc_default), and a deep convective mass flux (radiative cover != ast).
Production CloudConfig (amip_production.yaml, resolved through run_amip + model_driver) is
stored in the fixture.
Generator: scripts/validate/oracle/reff_incloud/{build.sh,oracle_driver.F90,make_fixture.py}.

Liquid: asserted on EVERY level against ``rel_cam`` (CAM6 REL from the post-MG2 number,
i.e. after MG2's mean-size re-adjust: in-cloud water capped at 5 g/kg, MG2
size_dist_param_liq at the dry density, qsmall gate, 10 um fill).  ``effc_wrf`` (legoESM's
former M2005 PSD on the same inputs) is kept for the PSD gap only.
Ice: asserted on every tracer-ice level against MG2 ``effi``, including where the ice-slope
bounds adjust the number (``ni_adj``) and where the 5 g/kg in-cloud cap binds.
The oracle match runs MG2's hard limits (RULE 5 widths 0); the production widths are
pinned as measured gaps with gradient checks across each limit.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics._shared import compute_rho
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    KK2000_CAM6_QC_MAX,
    MG2_MINCLD,
    mg2_size_bounded_nc,
)
from legoesm.atmosphere.physics.microphysics.config import MG2_DCS_M, MorrisonConfig
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig

_FIX = Path(__file__).parent / "oracle_fixtures" / "reff_incloud.npz"
_TOL = 1e-10  # user-approved oracle-match tolerance (max relative error, fp64);
# where CAM's own number update loses more digits, its computed round-off bound
_MG2_QC_CAP = 5.0e-3  # micro_mg2_0.F90:2762, micro_mg_cam.F90:2296 in-cloud cap


@pytest.fixture(scope="module")
def z():
    return np.load(_FIX)


# Droplet numbers [#/m^3] at which the hard MG2 slope reaches 1/50 um and 1/2 um
# (q = 1e-4 kg/kg, rho = 1), and the production-width gaps to the oracle on the
# fixture, all measured 2026-10-27 (a change to a smooth form or width re-measures).
_N_LAM_LO, _N_LAM_HI = 1175671.9776878892, 10742958658.70296
_PROD_GAPS = (6.112631957471963e-10, 2.303176760420911e-09)

# MG2's hard limits (RULE 5 widths 0) for the oracle match; dcs = the fixture's.
_HARD = dict(mg2_drop_number_cap_log_width=0.0, mg2_drop_slope_log_width=0.0,
             mg2_ice_cap_log_width=0.0)


def _kwargs(z, hard=True):
    cfg = CloudConfig(**json.loads(str(z["meta"]))["cloud_config"])
    if hard:
        cfg = cfg._replace(**_HARD)
    return dict(
        mg2_ice_dcs_m=MG2_DCS_M,
        mg2_ice_slope_log_width=0.0 if hard else MorrisonConfig().mg2_ice_slope_log_width,
        T=jnp.asarray(z["T"]), p_full=jnp.asarray(z["p_full"]), q_v=jnp.asarray(z["q_v"]),
        dp=jnp.asarray(np.diff(z["p_half"], axis=1)), config=cfg,
        q_cloud=jnp.asarray(z["q_c"]), q_ice=jnp.asarray(z["q_i"]),
        n_ice=jnp.asarray(z["n_i"]), n_cloud=jnp.asarray(z["n_ccn"]),
        cloud_fraction_override=jnp.asarray(z["alst"]), lat=jnp.asarray(z["lat"]),
        conv_mass_flux_up=jnp.asarray(z["conv_mass_flux_up"]),
        conv_icwmr=jnp.asarray(z["conv_icwmr"]), p_half=jnp.asarray(z["p_half"]))


def _max_rel(ours, ref, mask):
    assert mask.sum() > 0
    return float(np.max(np.abs(ours[mask] - ref[mask]) / np.abs(ref[mask])))


def _rel_roundoff_bound(z):
    """Round-off of CAM's post-MG2 number ``nc + nctend*dt`` (oracle_driver.F90),
    ``nctend*dt = nc_adj - nc`` (driver deltat = 1, so /dt and *dt are exact):
    relative error <= eps (|R - 1| + 2) with
    ``R = nc / nc_adj``, carried 1:1 into r_eff (its true sensitivity to the
    number is ~1/3).  ``R`` comes from the inputs through MG2's hard re-adjust."""
    kw = _kwargs(z)
    n = jnp.where(kw["n_cloud"] > 1.0, kw["n_cloud"], kw["config"].Nc_default)
    rho_d = compute_rho(kw["T"], kw["p_full"])
    pre = n / compute_rho(kw["T"], kw["p_full"], kw["q_v"]) * rho_d
    q_ic = jnp.minimum(kw["q_cloud"] / jnp.maximum(jnp.asarray(z["ast"]), MG2_MINCLD),
                       KK2000_CAM6_QC_MAX)
    post = mg2_size_bounded_nc(q_ic, pre, rho_d, number_cap_log_width=0.0,
                               slope_log_width=0.0)
    return (np.abs(np.asarray(pre / post) - 1.0) + 2.0) * np.finfo(np.float64).eps / 2


def test_liquid_reff_matches_cam6_rel(z):
    props = compute_cloud_properties(**_kwargs(z), n_cloud_in_cloud=True,
                                     clubb_config=CLUBBConfig())
    ours = np.asarray(props.r_eff_liq) * 1e6  # m -> um (oracle REL units)
    q_ic = z["q_c"] / z["lcldm"]
    dead = z["n_ccn"] <= 1.0
    live = ~dead & (z["q_c"] > 0.0)
    # the fixture exercises both halves of lcldm = max(ast, mincld), ast != cover,
    # the 5 g/kg cap, the in-cloud qsmall gate below legoESM's old 1e-14 grid gate,
    # the 10 um fill, and the dead-number floor between mincld and 1e-3 and below mincld
    assert np.any(live & (z["ast"] < 1.0e-4)) and np.any(live & (z["ast"] > 0.05))
    assert np.any(live & (np.asarray(props.cloud_fraction) > z["ast"] + 1e-6))
    assert np.any(live & (q_ic > _MG2_QC_CAP))
    assert np.any(live & (z["q_c"] <= 1.0e-14) & (q_ic >= 1.0e-18))
    assert np.any(z["q_c"] <= 0.0)
    a = np.asarray(props.cloud_fraction)[dead]
    assert np.any((a > 1.0e-4) & (a < 1.0e-3)) and np.any(a < 1.0e-4)
    err = np.abs(ours - z["rel_cam"]) / np.abs(z["rel_cam"])
    allowed = np.maximum(_TOL, _rel_roundoff_bound(z))
    assert np.all(err <= allowed), (
        f"r_eff_liq rel {err.max():.3e}; worst level err/allowed {np.max(err / allowed):.3e}")
    # the no-liquid fill is CAM's 10 um, not the CloudConfig.r_eff_liq knob
    kw = _kwargs(z)
    kw["config"] = kw["config"]._replace(r_eff_liq=14.0e-6)
    ours14 = np.asarray(compute_cloud_properties(
        **kw, n_cloud_in_cloud=True, clubb_config=CLUBBConfig()).r_eff_liq) * 1e6
    np.testing.assert_array_equal(ours14, ours)


def test_dead_number_reff_ignores_deep_convective_cover(z):
    """Follow-up row 46: the dead-number REL divides the water by ast, as
    CAM does for every level (micro_mg_cam.F90:2539), not by a radiative
    cover that also holds deep cu.  The fixture's dead levels sit in a column
    without convection, so give that column the deepest column's convective
    mass flux: cover rises above ast there and CAM's REL (which never sees
    the convective cover) must still be matched."""
    kw = _kwargs(z)
    dead = np.asarray(z["n_ccn"]) <= 1.0
    cols = np.unique(np.nonzero(dead)[0])
    mf = np.asarray(z["conv_mass_flux_up"]).copy()
    icw = np.asarray(z["conv_icwmr"]).copy()
    # deep convection through the whole of those columns
    mf[cols] = np.max(mf)
    icw[cols] = np.max(icw)
    kw["conv_mass_flux_up"] = jnp.asarray(mf)
    kw["conv_icwmr"] = jnp.asarray(icw)
    props = compute_cloud_properties(**kw, n_cloud_in_cloud=True,
                                     clubb_config=CLUBBConfig())
    cover = np.asarray(props.cloud_fraction)
    assert np.any(dead & (cover > np.asarray(z["ast"]) + 1e-6))
    ours = np.asarray(props.r_eff_liq) * 1e6
    rel = _max_rel(ours, z["rel_cam"], dead)
    assert rel <= _TOL, f"dead-number r_eff_liq max rel {rel:.3e} > {_TOL}"

def test_ice_reff_matches_mg2_effi(z):
    props = compute_cloud_properties(**_kwargs(z), n_cloud_in_cloud=True,
                                     clubb_config=CLUBBConfig())
    ours = np.asarray(props.r_eff_ice) * 1e6
    m = z["q_i"] > 1.0e-14
    # rows 48/52: the slope bounds re-adjust the number and the 5 g/kg cap binds somewhere
    assert np.any(m & (z["ni_adj"] != 0.0)) and np.any(m & (z["q_i"] / z["lcldm"] > _MG2_QC_CAP))
    rel = _max_rel(ours, z["effi_mg2"], m)
    assert rel <= _TOL, f"r_eff_ice max rel {rel:.3e} > {_TOL}"


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_reff_gradient_finite(z, dtype):
    kw = {k: (jnp.asarray(v, dtype=dtype) if isinstance(v, jnp.ndarray) else v)
          for k, v in _kwargs(z).items()}

    def loss(q_c, n_c, alst):
        p = compute_cloud_properties(**{**kw, "q_cloud": q_c, "n_cloud": n_c,
                                        "cloud_fraction_override": alst},
                                     n_cloud_in_cloud=True,
                                     clubb_config=CLUBBConfig())
        return jnp.sum(p.r_eff_liq) + jnp.sum(p.lwp / p.r_eff_liq)

    grads = jax.grad(loss, argnums=(0, 1, 2))(kw["q_cloud"], kw["n_cloud"],
                                               kw["cloud_fraction_override"])
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.asarray(grads[2]) != 0.0)  # r_eff depends on the cloud fraction


def test_production_widths_measured_gaps(z):
    """KNOWN DIFFERENCE (RULE 5) at the production widths (CloudConfig 0.01, the
    microphysics ice-slope width): max relative gap to the hard-limit oracle."""
    props = compute_cloud_properties(**_kwargs(z, hard=False), n_cloud_in_cloud=True,
                                     clubb_config=CLUBBConfig())
    liq = _max_rel(np.asarray(props.r_eff_liq) * 1e6, z["rel_cam"], np.ones(z["q_c"].shape, bool))
    ice = _max_rel(np.asarray(props.r_eff_ice) * 1e6, z["effi_mg2"], z["q_i"] > 1.0e-14)
    print("production-width gaps: liquid", repr(liq), "ice", repr(ice))
    np.testing.assert_allclose([liq, ice], _PROD_GAPS, rtol=1e-6, atol=1e-14)


def test_droplet_limits_gradient_across_each_bound():
    """Each smooth droplet limit at 0.01: second-order gradient check and a finite,
    nonzero slope through the bound (N <= q/1e-20, and lambda at 1/[50, 2] um)."""
    from jax.test_util import check_grads

    from legoesm.atmosphere.physics.microphysics import _warm_rain as W

    w = dict(number_cap_log_width=0.01, slope_log_width=0.01)

    def r_eff(lq, ln):
        return jnp.sum(W.mg2_droplet_effective_radius(
            jnp.exp(lq), W.mg2_size_bounded_nc(jnp.exp(lq), jnp.exp(ln), 1.0, **w), 1.0, **w))

    q = 1.0e-4
    for n in (_N_LAM_LO, _N_LAM_HI):   # slope bounds (50 um, 2 um mean diameter)
        a = (jnp.log(q), jnp.log(n))
        check_grads(r_eff, a, order=2, modes=["rev"], eps=1e-4, atol=1e-5, rtol=1e-4)
        g = jax.grad(r_eff, argnums=1)(*a)
        assert np.isfinite(float(g)) and float(g) != 0.0, (n, float(g))
    # The number cap N <= q/1e-20 kg sits ~1e6 above the 2 um slope bound, so the
    # radius is already on that bound there: finite (and checked) gradient, but
    # the cap's own transition is not visible in the radius.
    a = (jnp.log(q), jnp.log(q / 1.0e-20))
    check_grads(r_eff, a, order=2, modes=["rev"], eps=1e-4, atol=1e-5, rtol=1e-4)
    assert np.isfinite(float(jax.grad(r_eff, argnums=1)(*a)))
    assert q / 1.0e-20 > 1.0e5 * _N_LAM_HI


def test_ice_cap_gradient_across_the_cap(z):
    """The smooth 5 g/kg in-cloud ice cap at 0.01: r_eff_ice is differentiable in q_i
    with a finite, nonzero slope on both sides of the cap."""
    kw = _kwargs(z, hard=False)
    lev = np.nonzero(np.asarray(z["q_i"]) > 1.0e-14)
    i, k = lev[0][0], lev[1][0]
    icldm = float(z["lcldm"][i, k])
    for frac in (0.9, 1.0, 1.1):
        def f(qik):
            qi = kw["q_ice"].at[i, k].set(qik)
            return compute_cloud_properties(**{**kw, "q_ice": qi}, n_cloud_in_cloud=True,
                                            clubb_config=CLUBBConfig()).r_eff_ice[i, k]
        g = jax.grad(f)(jnp.asarray(frac * _MG2_QC_CAP * icldm))
        assert np.isfinite(float(g)) and float(g) != 0.0, (frac, float(g))


def test_droplet_number_cap_is_smooth_in_isolation():
    """The number cap's own transition (the radius cannot show it: the slope
    bound dominates): second-order gradient check at the cap, finite slope
    strictly between 0 and 1 just past it at 0.01, 0 for the hard cap."""
    from jax.test_util import check_grads

    from legoesm.atmosphere.physics.microphysics import _warm_rain as W

    q = jnp.asarray(1.0e-4)
    ln_cap = jnp.log(q / 1.0e-20)

    def f(ln_n, w):
        return W.mg2_drop_ln_number_cap(jnp.exp(ln_n), q, w)

    check_grads(lambda x: f(x, 0.01), (ln_cap,), order=2, modes=["rev"],
                eps=1e-4, atol=1e-5, rtol=1e-4)
    for dx, w, lo, hi in ((0.005, 0.01, 0.0, 1.0), (0.005, 0.0, -1e-12, 1e-12),
                          (-1.0, 0.01, 1.0 - 1e-6, 1.0 + 1e-6)):
        g = float(jax.grad(f)(ln_cap + dx, w))
        assert np.isfinite(g) and lo < g < hi, (dx, w, g)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ice_radius_gradients_finite_at_production_widths(z, dtype):
    """Ice radius and ice path through the new MG2 limits: every input
    gradient finite in fp32 and fp64, including levels with no ice."""
    kw = {k: (jnp.asarray(v, dtype=dtype) if isinstance(v, jnp.ndarray) else v)
          for k, v in _kwargs(z, hard=False).items()}

    def loss(q_i, n_i, alst):
        p = compute_cloud_properties(**{**kw, "q_ice": q_i, "n_ice": n_i,
                                        "cloud_fraction_override": alst},
                                     n_cloud_in_cloud=True, clubb_config=CLUBBConfig())
        return jnp.sum(p.r_eff_ice) + jnp.sum(p.iwp / p.r_eff_ice)

    grads = jax.grad(loss, argnums=(0, 1, 2))(kw["q_ice"], kw["n_ice"],
                                               kw["cloud_fraction_override"])
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.asarray(grads[1]) != 0.0)  # r_eff_ice depends on the ice number


def test_ice_radius_differentiable_in_dcs(z):
    """dcs is a tunable microphysics parameter; the CAM6 ice radius reads it and
    must stay differentiable in it (finite, nonzero where the lower slope bound
    binds)."""
    kw = _kwargs(z, hard=False)

    def f(dcs):
        return jnp.sum(compute_cloud_properties(**{**kw, "mg2_ice_dcs_m": dcs},
                                                n_cloud_in_cloud=True,
                                                clubb_config=CLUBBConfig()).r_eff_ice)

    g = float(jax.grad(f)(jnp.asarray(MG2_DCS_M)))
    assert np.isfinite(g) and g != 0.0, g
