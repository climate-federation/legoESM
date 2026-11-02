"""CAM6 MG2 density factors in thin air (rho < 0.1 kg/m3) against MG2.

MG2 has no density floor: rhof = (rhosu/rho)^0.54, arn = ar rhof,
asn = as rhof, ain = ai (rhosu/rho)^0.35 (micro_mg2_0.F90:895-901).  The point
grid of ``oracle_fixtures/mg2_thin_air.npz``
(scripts/validate/oracle/morrison_mg2_thin_air/make_fixture.py; unmodified
micro_mg_utils.F90 kernels with MG2's rhof, arn, asn, ain lines verbatim)
spans rho 0.03 to 0.4 kg/m3.  Three places in the MG2 mode used
rho floored at 0.1 kg/m3: the rain evaporation ventilation (arn), the snow
density factor (rhof, asn: fall speeds, aggregation, riming, Bergeron,
sublimation) and the cloud-ice fall speed (ain).  Bound 1e-10 relative, x64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig  # noqa: E402
from legoesm.atmosphere.physics.microphysics.morrison import resolve_morrison_flavor  # noqa: E402
from legoesm.thermo import (saturation_specific_humidity_goff_gratch,  # noqa: E402
                            saturation_specific_humidity_ice_goff_gratch)

from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import TOL, _rel  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_thin_air.npz"


def _fx():
    return dict(np.load(FIXTURE))


def _cfg():
    return resolve_morrison_flavor(MorrisonConfig(
        morrison_flavor="mg", ice_snow_d_auto=500.0e-6, melt_sharpness=1.0e9))


def _points(fx, kind):
    i = {k[6:]: jnp.asarray(v) for k, v in fx.items() if k.startswith("pt_in_")}
    o = {k[7:]: v for k, v in fx.items() if k.startswith("pt_out_")}
    sel = np.asarray(i["qr"] > 0) if kind == "rain" else np.asarray(i["qs"] > 0)
    return {k: v[sel] for k, v in i.items()}, {k: v[sel] for k, v in o.items()}


def _report(errs, rho):
    thin = np.asarray(rho) < 0.1
    worst = {k: (float(np.max(v[thin])), float(np.max(v[~thin]))) for k, v in errs.items()}
    print("max rel err (rho < 0.1, rho >= 0.1)", worst)
    return max(max(w) for w in worst.values()), worst


def test_fixture_spans_the_floor():
    fx = _fx()
    for g in ("pt_in_rho", "ice_in_rho"):
        assert fx[g].min() < 0.05 and fx[g].max() > 0.2
    i, o = _points(fx, "rain")
    assert np.count_nonzero(o["pre"][np.asarray(i["rho"]) < 0.1]) > 20
    i, o = _points(fx, "snow")
    for k in ("ums", "nsagg", "psacws", "bergs", "prds"):
        assert np.count_nonzero(o[k][np.asarray(i["rho"]) < 0.1]) > 10, k


def test_rain_evaporation_thin_air():
    """rain_evaporation_mg2 against evaporate_sublimate_precip (arn)."""
    fx = _fx()
    cfg = _cfg()
    i, o = _points(fx, "rain")
    rho, pf = i["rho"], i["precip_frac"]
    qvl = saturation_specific_humidity_goff_gratch(i["T"], i["p"])
    q_ric, N_ric, lamr, n0r = W.mg2_inprecip_rain(i["qr"], i["nr"] * rho, rho, pf,
                                                  cfg.lamr_min, cfg.lamr_max)
    errs = {"qric": _rel(q_ric, o["qric"]), "lamr": _rel(lamr, o["lamr"])}
    evap = W.rain_evaporation_mg2(i["qfrac"] * qvl, qvl, i["T"], i["p"], rho, q_ric, lamr, n0r,
                                  i["lcldm"], pf, i["qcic"], i["qiic"], cfg)
    errs["evap"] = _rel(evap, -o["pre"] * pf)
    worst, w = _report(errs, rho)
    assert worst <= TOL, w


def test_snow_kernels_thin_air():
    """MG2 snow kernels with the density factor (rhof, asn) in thin air."""
    fx = _fx()
    cfg = _cfg()
    i, o = _points(fx, "snow")
    rho, T, p, pf = i["rho"], i["T"], i["p"], i["precip_frac"]
    qvl = saturation_specific_humidity_goff_gratch(T, p)
    qvi = saturation_specific_humidity_ice_goff_gratch(T, p)
    qsic, nsic, lams, n0s = W.mg2_inprecip_snow(i["qs"], i["ns"], pf, cfg, psd_sharpness=1.0e9)
    errs = {"lams": _rel(lams, o["lams"])}
    ums, uns = W.mg2_snow_fall_speeds(lams, rho, cfg)
    errs["ums"] = _rel(ums, o["ums"])
    errs["uns"] = _rel(uns, o["uns"])
    errs["nsagg"] = _rel(W.mg2_snow_self_aggregation(qsic, nsic, T, rho, cfg), o["nsagg"])
    nc_adj, pgam, lamc = W.mg2_size_dist_liq(i["qcic"], i["ncic"] * rho, rho)
    eps_s, mu = W.mg2_snow_vent_eps(n0s, lams, T, p, rho, cfg)
    psacws, npsacws = W.mg2_accrete_cloud_water_snow(
        T, rho, mu, uns, i["qcic"], nc_adj / rho, qsic, pgam, lamc, lams, n0s, cfg, 1.0e-9)
    errs["psacws"] = _rel(psacws, o["psacws"])
    errs["npsacws"] = _rel(npsacws, o["npsacws"])
    errs["bergs"] = _rel(W.mg2_bergeron_snow(T, i["qcic"], qsic, qvl, qvi, eps_s, 1.0e-9),
                         o["bergs"])
    errs["prds"] = _rel(W.mg2_snow_sublimation(
        i["qfrac"] * qvl, qvl, qvi, T, eps_s, qsic, i["lcldm"], pf, i["qcic"], i["qiic"]),
        o["prds"])
    worst, w = _report(errs, rho)
    assert worst <= TOL, w


def _lego_ice_fall_speeds(fx, monkeypatch):
    """legoESM's MG2-mode ice fall speeds (V_t_i, V_n_i) for the oracle's ice
    state, read at the sedimentation call (one level per column, ice only,
    ice-subsaturated so nothing nucleates)."""
    from legoesm.atmosphere.physics.microphysics import morrison as M
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import (
        HARD_LIMIT, _production_morrison)
    T = jnp.asarray(fx["ice_in_T"])[:, None]
    p = jnp.asarray(fx["ice_in_p"])[:, None]
    rho = jnp.asarray(fx["ice_in_rho"])[:, None]
    q_i = jnp.asarray(fx["ice_in_dumi"])[:, None]
    N_i = jnp.asarray(fx["ice_in_dumni"])[:, None]
    ph = jnp.concatenate([p - 50.0, p + 50.0], axis=1)
    dz = (ph[:, 1:] - ph[:, :-1]) / (9.80616 * rho)
    z = jnp.zeros_like(T)
    qv = 0.5 * saturation_specific_humidity_ice_goff_gratch(T, p)
    seen = []
    orig = M.sedimentation_tendency

    def spy(q, rho_, V, dz_, **kw):
        seen.append((np.asarray(q), np.asarray(V)))
        return orig(q, rho_, V, dz_, **kw)
    monkeypatch.setattr(M, "sedimentation_tendency", spy)
    cfg = _production_morrison()._replace(**HARD_LIMIT, warm_rain_scheme="kk2000_cam6")
    hyd = HydrometeorState(q_c=z, q_r=z, q_i=q_i, q_s=z, q_g=z, N_c=z + 1e8, N_r=z, N_i=N_i,
                           N_s=z)
    M.morrison_microphysics(T, qv, hyd, p, ph, rho, dz, 600.0, cfg, cloud_fraction=z + 1.0)
    vt = [V for q, V in seen if np.array_equal(q, np.asarray(q_i))]
    vn = [V for q, V in seen if np.array_equal(q, np.asarray(jnp.clip(N_i, 0.0)))]
    assert len(vt) == 1 and len(vn) == 1, (len(vt), len(vn))
    return vt[0][:, 0], vn[0][:, 0]


def _ice_subset_density_only(fx):
    """Points where MG2's speeds are the plain power laws: mean radius
    1.5/lami >= 20 um (no small-ice blend, :2133-2135) and both speeds below
    the 1.2 rhof cap.  There the only density dependence is ain."""
    r_um = 1.5 / fx["ice_out_lami"] * 1.0e6
    cap = 1.2 * fx["ice_out_rhof"]
    return (r_um >= 20.0) & (fx["ice_out_vtrmi"] < cap) & (fx["ice_out_vni"] < cap)


def test_ice_fall_speed_density_factor_thin_air(monkeypatch):
    """MG2-mode cloud-ice fall speeds (ain = ai (rhosu/rho)^0.35, :900)
    against micro_mg2_0.F90:2119-2147 for the same ice state, on the points
    without the small-ice blend and below the cap."""
    fx = _fx()
    vt, vn = _lego_ice_fall_speeds(fx, monkeypatch)
    m = _ice_subset_density_only(fx)
    assert m.sum() >= 20 and (fx["ice_in_rho"][m] < 0.1).sum() >= 10
    errs = {"vtrmi": _rel(vt[m], fx["ice_out_vtrmi"][m]),
            "vni": _rel(vn[m], fx["ice_out_vni"][m])}
    worst, w = _report(errs, fx["ice_in_rho"][m])
    assert worst <= TOL, w


@pytest.mark.xfail(strict=True, reason="MG2 mode keeps the SAM ice fall speed: no small-ice "
                   "blend to aj, bj below 20 um (micro_mg2_0.F90:2133-2147) and a 1.2 "
                   "(rhosu/rho)^0.35 cap where MG2 caps at 1.2 rhof (:2121-2128); not ported")
def test_ice_fall_speed_all_points(monkeypatch):
    """All ice points, small ice and capped speeds included."""
    fx = _fx()
    vt, vn = _lego_ice_fall_speeds(fx, monkeypatch)
    errs = {"vtrmi": _rel(vt, fx["ice_out_vtrmi"]), "vni": _rel(vn, fx["ice_out_vni"])}
    worst, w = _report(errs, fx["ice_in_rho"])
    assert worst <= TOL, w
