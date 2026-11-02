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
The cloud-ice fall speeds are also checked at every point (MG2's small-ice
blend and 1.2 rhof cap, micro_mg2_0.F90:2119-2147) and on the post-process
ice MG2 sizes them from (:2063-2064, :2088).
"""
from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig  # noqa: E402
from legoesm.atmosphere.physics.microphysics.morrison import resolve_morrison_flavor  # noqa: E402
from legoesm import constants  # noqa: E402
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


# MG2 sizes the ice fall speeds from the post-process ice dumi = qi + qitend dt
# (micro_mg2_0.F90:2063-2064, :2088), and so does the MG2 mode.  The oracle's
# state is the input state, so the form tests run one step this short: the
# process change of the ice over it moves the speeds by 2.7e-12 relative.
_DT_FORM_S = 1.0e-12


# The 600 s columns are one layer 0.8 p thick, so the ice falls out at MG2's
# sub-step count below the model's cap (a 100 Pa layer would empty many times
# over in one step).
_THICK_HALF = 0.4


def _lego_ice_fall_speeds(fx, monkeypatch, dt=_DT_FORM_S, qv_frac=0.5, full=False,
                          half_dp=None):
    """legoESM's MG2-mode ice fall speeds (V_t_i, V_n_i) for the oracle's ice
    state, read where the MG2 ice sedimentation computes them (one level per
    column, ice only, ice-subsaturated so nothing nucleates, unless
    ``qv_frac > 1``).  ``full`` also returns the ice mass and number the
    sedimentation transports, its mass and number tendencies, the outputs
    and the resolved config.  A model that sediments the ice through the
    generic ``sedimentation_tendency`` (the order before the MG2 port) is
    read there instead."""
    from legoesm.atmosphere.physics.microphysics import _warm_rain as W
    from legoesm.atmosphere.physics.microphysics import morrison as M
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import (
        HARD_LIMIT, _production_morrison)
    T = jnp.asarray(fx["ice_in_T"])[:, None]
    p = jnp.asarray(fx["ice_in_p"])[:, None]
    rho = jnp.asarray(fx["ice_in_rho"])[:, None]
    q_i = jnp.asarray(fx["ice_in_dumi"])[:, None]
    N_i = jnp.asarray(fx["ice_in_dumni"])[:, None]
    h = 50.0 if half_dp is None else half_dp * p
    ph = jnp.concatenate([p - h, p + h], axis=1)
    dz = (ph[:, 1:] - ph[:, :-1]) / (constants.g * rho)
    z = jnp.zeros_like(T)
    qv = qv_frac * saturation_specific_humidity_ice_goff_gratch(T, p)
    seen, speeds, mg2 = [], [], []
    orig, orig_v = M.sedimentation_tendency, getattr(W, "mg2_ice_fall_speeds", None)
    orig_s = getattr(M, "mg2_ice_sedimentation", None)

    def spy(q, rho_, V, dz_, **kw):
        res = orig(q, rho_, V, dz_, **kw)
        seen.append((np.asarray(q), np.asarray(V), res))
        return res

    def spy_v(*a):
        res = orig_v(*a)
        speeds.append(tuple(np.asarray(v) for v in res))
        return res

    def spy_s(q, n, *a, **k):
        res = orig_s(q, n, *a, **k)
        mg2.append((np.asarray(q), np.asarray(n), res))
        return res
    monkeypatch.setattr(M, "sedimentation_tendency", spy)
    if orig_s is not None:
        monkeypatch.setattr(W, "mg2_ice_fall_speeds", spy_v)
        monkeypatch.setattr(M, "mg2_ice_sedimentation", spy_s)
    cfg = _production_morrison()._replace(**HARD_LIMIT, warm_rain_scheme="kk2000_cam6")
    hyd = HydrometeorState(q_c=z, q_r=z, q_i=q_i, q_s=z, q_g=z, N_c=z + 1e8, N_r=z, N_i=N_i,
                           N_s=z)
    out = M.morrison_microphysics(T, qv, hyd, p, ph, rho, dz, dt, cfg,
                                  cloud_fraction=z + 1.0)
    if mg2:
        assert len(mg2) == 1 and len(speeds) == 1, (len(mg2), len(speeds))
        (q_tr, n_tr, res), (vt, vn) = mg2[0], speeds[0]
        sed_q, sed_n = res[0], res[1]
    else:   # the generic path: the ice read at its sedimentation call
        vt = [(V, r) for q, V, r in seen if np.array_equal(q, np.asarray(q_i))]
        vn = [(V, r) for q, V, r in seen if np.array_equal(q, np.asarray(jnp.clip(N_i, 0.0)))]
        assert len(vt) == 1 and len(vn) == 1, (len(vt), len(vn))
        q_tr, n_tr, sed_q, sed_n = q_i, jnp.clip(N_i, 0.0), vt[0][1][0], vn[0][1]
        vt, vn = vt[0][0], vn[0][0]
    if not full:
        return vt[:, 0], vn[:, 0]
    return dict(vt=vt[:, 0], vn=vn[:, 0], q_tr=np.asarray(q_tr)[:, 0],
                n_tr=np.asarray(n_tr)[:, 0], sed_q=np.asarray(sed_q)[:, 0],
                sed_n=np.asarray(sed_n)[:, 0], out=out, cfg=M.resolve_morrison_flavor(cfg))


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


def test_ice_fall_speed_all_points(monkeypatch):
    """All ice points, small ice (the blend to aj, bj below 20 um,
    :2133-2147) and capped speeds (1.2 rhof, :2121-2128) included."""
    fx = _fx()
    vt, vn = _lego_ice_fall_speeds(fx, monkeypatch)
    errs = {"vtrmi": _rel(vt, fx["ice_out_vtrmi"]), "vni": _rel(vn, fx["ice_out_vni"])}
    worst, w = _report(errs, fx["ice_in_rho"])
    assert worst <= TOL, w


def test_ice_fall_speed_post_process_state(monkeypatch):
    """Over a 600 s step the speeds are MG2's form (checked above) on the
    post-process ice, rebuilt here from the step's outputs as
    q + (tendency - sedimentation tendency) dt for mass and number.  The
    column is ice-supersaturated (q_v = 1.2 q_sat,i), so the ice grows by up
    to 4466 times.  In a subsaturated column the step's final
    ice-number adjustments rewrite N_i after the ice has nearly all
    sublimated, so the outputs no longer carry the sizing state."""
    from legoesm.atmosphere.physics.microphysics import _warm_rain as W
    fx = _fx()
    dt = 600.0
    r = _lego_ice_fall_speeds(fx, monkeypatch, dt=dt, qv_frac=1.2, full=True,
                              half_dp=_THICK_HALF)
    out, cfg = r["out"], r["cfg"]
    q_post = fx["ice_in_dumi"] + (np.asarray(out.dq_i_dt)[:, 0] - r["sed_q"]) * dt
    n_post = fx["ice_in_dumni"] + (np.asarray(out.dN_i_dt)[:, 0] - r["sed_n"]) * dt
    _, lami = W.mg2_ice_slope(jnp.asarray(q_post), jnp.maximum(jnp.asarray(n_post), 0.0),
                              cfg.rho_cloud_ice, cfg.ice_snow_d_auto)
    vt, vn = W.mg2_ice_fall_speeds(lami, jnp.asarray(fx["ice_in_rho"]), cfg)
    assert np.max(_rel(r["vt"], fx["ice_out_vtrmi"])) > 0.5   # the state did change
    m = _number_kept(fx, r, dt)
    errs = {"vtrmi": _rel(r["vt"][m], np.asarray(vt)[m]),
            "vni": _rel(r["vn"][m], np.asarray(vn)[m])}
    worst, w = _report(errs, fx["ice_in_rho"][m])
    assert worst <= TOL, w


def _number_kept(fx, r, dt):
    """Cells whose ice number the step's last size re-adjustment left alone:
    where the end-of-step slope sits on an MG2 bound (1/[2 dcs, 1 um]), that
    re-adjustment (micro_mg2_0.F90:2778) rewrites N_i after sedimentation,
    so the outputs no longer give the transported number there (32 of the
    180 cells, ice that sedimentation left with too many crystals)."""
    from legoesm.atmosphere.physics.microphysics import _warm_rain as W
    out, cfg = r["out"], r["cfg"]
    q_end = fx["ice_in_dumi"] + np.asarray(out.dq_i_dt)[:, 0] * dt
    n_end = fx["ice_in_dumni"] + np.asarray(out.dN_i_dt)[:, 0] * dt
    lam = (np.pi * cfg.rho_cloud_ice * n_end / q_end) ** (1.0 / 3.0)
    lo, hi = W.mg2_ice_lambda_bounds(cfg.ice_snow_d_auto)
    m = (lam < hi * (1.0 - 1.0e-9)) & (lam > lo * (1.0 + 1.0e-9))
    assert m.sum() >= 120, m.sum()
    return m


def test_ice_sediments_the_post_process_state(monkeypatch):
    """MG2 order (micro_mg2_0.F90:2206-2214): the ice transported is the
    post-process dumi = qi + qitend dt, dumni = max(ni + nitend dt, 0), with
    no sink reserved; the same 600 s supersaturated column as above, the
    post-process state rebuilt from the step's outputs.  The order before
    (the input ice, its sinks reserved) transports q_i and fails."""
    fx = _fx()
    dt = 600.0
    r = _lego_ice_fall_speeds(fx, monkeypatch, dt=dt, qv_frac=1.2, full=True,
                              half_dp=_THICK_HALF)
    out = r["out"]
    q_post = fx["ice_in_dumi"] + (np.asarray(out.dq_i_dt)[:, 0] - r["sed_q"]) * dt
    n_post = np.maximum(fx["ice_in_dumni"] + (np.asarray(out.dN_i_dt)[:, 0] - r["sed_n"]) * dt,
                        0.0)
    assert np.max(_rel(q_post, fx["ice_in_dumi"])) > 0.5      # the ice changed in the step
    m = _number_kept(fx, r, dt)
    errs = {"dumi": _rel(r["q_tr"], q_post)}
    worst, w = _report(errs, fx["ice_in_rho"])
    assert worst <= TOL, w
    errs = {"dumni": _rel(r["n_tr"][m], n_post[m])}
    worst, w = _report(errs, fx["ice_in_rho"][m])
    assert worst <= TOL, w


def _col_rel(a, b):
    """|a - b| over the column's largest |b| (tendencies cancel level by level)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    s = np.max(np.abs(b), axis=1, keepdims=True)
    return np.abs(a - b) / np.where(s > 0, s, 1.0)


def _mg2_ice_sed_columns(fx, q, n, tag="isc"):
    cfg = _cfg()
    pdel = fx[f"{tag}_in_pdel"]
    p_half = jnp.asarray(np.concatenate([np.zeros_like(pdel[:, :1]), np.cumsum(pdel, axis=1)],
                                        axis=1))
    return W.mg2_ice_sedimentation(
        jnp.asarray(q), jnp.asarray(n), jnp.asarray(fx[f"{tag}_in_icldm"]),
        jnp.asarray(fx[f"{tag}_in_rho"]), p_half, float(fx["isc_dt"]),
        int(fx[f"{tag}_out_nstep"].max()) + 1, cfg, cfg.ice_snow_d_auto,
        presence_top=W.MG2_QSMALL, slope_log_width=0.0)


def test_ice_sedimentation_columns_against_mg2():
    """mg2_ice_sedimentation against MG2's own ice sedimentation of 18
    columns (isc_*, mg2_ice_sed_cols.F90: micro_mg2_0.F90:2063-2147,
    :2206-2316 verbatim), three cloud-fraction profiles (ice falling into
    clear air sublimates), slow and fast ice (1 and 7 sub-steps at 1800 s),
    process tendencies that leave, grow or deplete the ice, thin air on top.
    Fed MG2's post-process ice it reproduces the mass and number
    sedimentation tendencies, the vapour and heat of the sublimated inflow,
    the surface ice flux and the sub-step count.  Fed the pre-process ice
    (the order before) it misses wherever the processes act."""
    fx = _fx()
    dt = float(fx["isc_dt"])
    q_post = fx["isc_in_qi"] + fx["isc_in_qitend"] * dt
    n_post = np.maximum(fx["isc_in_ni"] + fx["isc_in_nitend"] * dt, 0.0)
    dq, dn, dqv, dT, sfc, req = _mg2_ice_sed_columns(fx, q_post, n_post)
    assert (fx["isc_out_nstep"] > 1).sum() >= 6 and (np.abs(fx["isc_out_qvlat"]) > 0).any()
    np.testing.assert_array_equal(np.asarray(req), fx["isc_out_nstep"])
    errs = {"qisedten": _col_rel(dq, fx["isc_out_qisedten"]),
            "nised": _col_rel(dn, fx["isc_out_nitend"] - fx["isc_in_nitend"]),
            "qvlat": _col_rel(dqv, fx["isc_out_qvlat"]),
            "tlat": _col_rel(dT, fx["isc_out_tlat"] / constants.c_pd)}
    worst = {k: float(v.max()) for k, v in errs.items()}
    worst["preci"] = float(np.max(_rel(sfc, fx["isc_out_preci"] * constants.rho_water)))
    print("max column-relative error", worst)
    assert max(worst.values()) <= TOL, worst
    # the order before: the input ice transported
    dq0 = _mg2_ice_sed_columns(fx, fx["isc_in_qi"], fx["isc_in_ni"])[0]
    acts = np.any(fx["isc_in_qitend"] != 0, axis=1)
    miss = _col_rel(dq0, fx["isc_out_qisedten"]).max(axis=1)
    assert acts.sum() == 12 and (miss[acts] > 0.05).all() and (miss[~acts] <= TOL).all(), miss


def _snow_melt_after_sedimentation(fx, with_sed_heat):
    """The MG2 helpers chained as micro_mg2_0.F90 runs them on the ism
    columns: ice sedimentation, then the +2 C snow melt tested on
    t + tlat/cpp dt with (MG2's order) or without the sedimentation heating."""
    dt = float(fx["isc_dt"])
    dT_sed = np.asarray(_mg2_ice_sed_columns(fx, fx["ism_in_qi"], fx["ism_in_ni"], "ism")[3])
    T_test = fx["ism_in_t"] + (fx["ism_in_tlat"] / constants.c_pd
                               + (dT_sed if with_sed_heat else 0.0)) * dt
    m, n = W.mg2_instant_snow_melt(jnp.asarray(T_test), jnp.asarray(fx["ism_in_dums"]),
                                   jnp.asarray(fx["ism_in_dumns"]), _DT_FORM_S, math.inf)
    return np.asarray(m) / dt, np.asarray(n) / dt, dT_sed


def test_snow_melt_sees_the_sedimentation_heating_as_mg2():
    """MG2 order against MG2 (ism_*: mg2_ice_sed_cols.F90, the ice
    sedimentation loop :2232-2316 then the snow melt :2536-2566, verbatim):
    27 two-level columns near 0 C, ice falling from a full cloud into a 5 %
    one where 95 % of the inflow sublimates and cools the snowy air by 0.12
    or 0.49 K, the air 0.2, 0.6 or 3 K above snowmelt.  The melt rates and
    the final heating match with the sedimentation cooling in the melt's
    temperature; without it (the order before) the melt is wrong in 8
    columns, by up to the whole melt."""
    fx = _fx()
    m, n, dT_sed = _snow_melt_after_sedimentation(fx, True)
    terms = (fx["ism_in_tlat"], constants.c_pd * dT_sed, -constants.L_f * m)
    tlat = terms[0] + terms[1] + terms[2]
    # scale: the column's largest heating term (a heat-limited melt cancels
    # the rest to round-off)
    scale = np.max(sum(np.abs(t) for t in terms), axis=1, keepdims=True)
    errs = {"smelt": float(np.max(_rel(m, fx["ism_out_smelt"]))),
            "nsmelt": float(np.max(_rel(n, fx["ism_out_nsmelt"]))),
            "tlat": float(np.max(np.abs(tlat - fx["ism_out_tlat"]) / scale))}
    print("max relative error", errs)
    assert max(errs.values()) <= TOL, errs
    m0 = _snow_melt_after_sedimentation(fx, False)[0]
    miss = np.max(_rel(m0, fx["ism_out_smelt"]), axis=1)
    assert (miss > 0.05).sum() == 8 and miss.max() == 1.0, miss


def _melt_column(monkeypatch, zero=()):
    """One two-level MG2-mode column near 0 C: cloud at +1 C with small ice
    on top (cloud fraction 1), snowy air 0.1 K above snowmelt below it
    (cloud fraction 0.05), where falling ice sublimates and falling droplets
    evaporate.  Returns the temperature the post-sedimentation snow melt
    tests, its melt, and the ice and droplet sedimentation heating; the
    sedimentation heating named in ``zero`` ("ice", "drop") is set to 0."""
    from legoesm.atmosphere.physics.microphysics import morrison as M
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import (
        HARD_LIMIT, _production_morrison)
    T = jnp.array([[constants.T_freeze + 1.0, constants.T_freeze + 2.1]])
    p = jnp.array([[60000.0, 64000.0]])
    ph = jnp.array([[58500.0, 61500.0, 66500.0]])
    rho = p / (constants.R_d * T)
    dz = (ph[:, 1:] - ph[:, :-1]) / (constants.g * rho)
    cf = jnp.array([[1.0, 0.05]])
    lam_i, lam_s = 1.5 / 40.0e-6, 2000.0   # small ice: little of it turns to snow
    q_i = jnp.array([[2.0e-4, 0.0]])
    q_s = jnp.array([[0.0, 2.0e-3]])
    hyd = HydrometeorState(
        q_c=jnp.array([[5.0e-4, 0.0]]), q_r=jnp.zeros_like(T), q_i=q_i, q_s=q_s,
        q_g=jnp.zeros_like(T), N_c=jnp.full_like(T, 1.0e8), N_r=jnp.zeros_like(T),
        N_i=lam_i ** 3 * q_i / (np.pi * 500.0), N_s=lam_s ** 3 * q_s / (np.pi * 250.0))
    qv = saturation_specific_humidity_goff_gratch(T, p) * jnp.array([[1.0, 0.9]])
    seen = {"T": [], "melt": []}
    real_m, real_i, real_c = (M.mg2_instant_snow_melt, M.mg2_ice_sedimentation,
                              M.mg2_cloud_sedimentation)

    def spy_m(T_, q, n, *a, **k):
        res = real_m(T_, q, n, *a, **k)
        seen["T"].append(np.asarray(T_))
        seen["melt"].append(np.asarray(res[0]))
        return res

    def wrap(real, key):
        def f(*a, **k):
            res = list(real(*a, **k))
            seen[key] = np.asarray(res[3])
            if key in zero:
                res[3] = 0.0 * res[3]
            return tuple(res)
        return f
    monkeypatch.setattr(M, "mg2_instant_snow_melt", spy_m)
    monkeypatch.setattr(M, "mg2_ice_sedimentation", wrap(real_i, "ice"))
    monkeypatch.setattr(M, "mg2_cloud_sedimentation", wrap(real_c, "drop"))
    cfg = _production_morrison()._replace(**HARD_LIMIT, warm_rain_scheme="kk2000_cam6",
                                          specified_nc_droplet_sedimentation=True)
    M.morrison_microphysics(T, qv, hyd, p, ph, rho, dz, 1800.0, cfg, cloud_fraction=cf)
    assert len(seen["T"]) == 2            # start-of-step melt, post-sedimentation melt
    return seen["T"][1][0], seen["melt"][1][0], seen["ice"][0], seen["drop"][0]


def test_post_sedimentation_snow_melt_temperature_has_the_sedimentation_heating(monkeypatch):
    """MG2 order in the scheme (micro_mg2_0.F90:2296, :2367 before :2543):
    the temperature the post-sedimentation snow melt tests carries the ice
    and droplet sedimentation heating; removing either heating moves it by
    exactly that heating x dt.  In the order before, the melt saw neither,
    and this fails.  (The start-of-step melt brings the snowy air back to
    snowmelt and no process here warms it at +2 C, so the melt after
    sedimentation is 0 in both; the melt itself is checked against MG2
    above.)"""
    dt = 1800.0
    T0, m0, dT_ice, dT_drop = _melt_column(monkeypatch)
    assert dT_ice[1] < -1.0e-5 and dT_drop[1] < -1.0e-5, (dT_ice, dT_drop)
    for key, dT in (("ice", dT_ice), ("drop", dT_drop)):
        with monkeypatch.context() as mp:
            T1, m1, _, _ = _melt_column(mp, zero=(key,))
        np.testing.assert_allclose(T0 - T1, dT * dt, rtol=1.0e-9, atol=1.0e-12, err_msg=key)


def _rain_freeze(fx, width_K, cap, ramp_frac):
    """legoESM's post-sedimentation rain freezing on the rfz points: the
    frozen rain and number rates, their cloud-ice and snow shares, the rain
    slope and the heating after it."""
    dt = float(fx["isc_dt"])
    cfg = _cfg()
    T_test = fx["rfz_in_t"] + fx["rfz_in_tlat"] / constants.c_pd * dt
    m, n = W.mg2_instant_rain_freeze(jnp.asarray(T_test), jnp.asarray(fx["rfz_in_dumr"]),
                                     jnp.asarray(fx["rfz_in_dumnr"]), width_K, cap)
    one = jnp.ones_like(m)
    n_adj, lamr = W.mg2_rain_size_dist(jnp.asarray(fx["rfz_in_dumr"]),
                                       jnp.asarray(fx["rfz_in_dumnr"]), one,
                                       cfg.lamr_min, cfg.lamr_max)
    nd = W.mg2_instant_rain_freeze(jnp.asarray(T_test), jnp.asarray(fx["rfz_in_dumr"]),
                                   n_adj, width_K, cap)[1]
    f = np.asarray(W.mg2_frozen_rain_ice_fraction(lamr, jnp.asarray(fx["rfz_in_dcs"]),
                                                  ramp_frac, ice_at_dcs=True))
    m, n, nd = np.asarray(m) / dt, np.asarray(n) / dt, np.asarray(nd) / dt
    return dict(qrfrz=m, nrfrz=n, qifrz=f * m, nifrz=f * nd, qsfrz=m - f * m,
                nsfrz=nd - f * nd, lamr=np.asarray(lamr),
                tlat=fx["rfz_in_tlat"] + constants.L_f * m)


def test_rain_freeze_after_sedimentation_against_mg2():
    """MG2's post-sedimentation rain freezing (rfz_*: mg2_rain_frz_pts.F90,
    micro_mg2_0.F90:2571-2610 verbatim) at the hard limits: 48 points from
    5 K below rainfrze = tmelt - 40 K (:331) to 0.5 K above, the rain the
    grid pool after sedimentation (:2511-2512, :2527), heat-limited freezing
    near rainfrze, mean drops of 100 um, 480 um and 1 mm (slope-bounded to
    500 um) at Dcs 500, 450 and 200 um.  MG2 sends the frozen rain to snow
    where lamr < 1/Dcs and to cloud ice otherwise (:2596-2602, the
    ice_at_dcs test of the size switch), so a drop at the slope bound
    1/lamr = 500 um goes to ice at CAM6's Dcs of 500 um.  Covered: drops at
    the bound and on each side of Dcs (480 um below Dcs 500, 480 um and the
    bound above Dcs 450).  Frozen rain and number, rain slope, the snow and
    ice shares and the heating match on every point."""
    fx = _fx()
    lg = _rain_freeze(fx, _DT_FORM_S, math.inf, 0.0)
    on = fx["rfz_out_qrfrz"] > 0
    assert (fx["rfz_out_qrfrz"] == 0).sum() >= 12        # above rainfrze: no freezing
    d, dcs = 1.0 / np.where(on, fx["rfz_out_lamr"], 1.0), fx["rfz_in_dcs"]
    bound = on & (np.abs(d * 2000.0 - 1.0) < 1e-12)      # 1/lamr at the 500 um bound
    for sel, dest in ((bound & (dcs == 500.0e-6), "qifrz"),   # at the bound = Dcs: ice
                      (on & (d < dcs), "qifrz"),             # below Dcs: ice
                      (on & (d > dcs), "qsfrz")):            # above Dcs: snow
        assert sel.sum() >= 9 and (fx[f"rfz_out_{dest}"][sel] > 0).all(), dest
    assert (on & (d > dcs) & (d < 1.1 * dcs)).sum() >= 9  # just above Dcs (480 um at 450)
    errs = {k: float(np.max(_rel(lg[k], fx[f"rfz_out_{k}"]))) for k in
            ("qrfrz", "nrfrz", "tlat", "qifrz", "nifrz", "qsfrz", "nsfrz")}
    errs["lamr"] = float(np.max(_rel(lg["lamr"][on], fx["rfz_out_lamr"][on])))
    print("max relative error", errs)
    assert max(errs.values()) <= TOL, errs
    # each pair of shares sums to the rain loss bit for bit (the scheme takes
    # the loss as the sum of its shares)
    np.testing.assert_array_equal(lg["qifrz"] + lg["qsfrz"], lg["qrfrz"])


# Default-switch gap per rfz point (relative error of the frozen rain, and of
# its cloud-ice share, against MG2): the 0.2 K switch-on of the
# homogeneous-freezing width, the rounded heat cap (cap_sharpness 100) and the
# size ramp from Dcs to 1.1 Dcs.  Measured 2026-11-02 on this fixture.  Point
# order: Dcs (500, 450, 200 um) > offset (-5, -0.5, -0.05, +0.5 K) > rain
# (1e-8, 1e-5, 1e-3) > drop (100 um, 480 um, bound).  Frozen rain: 0.8965 of
# it missing on the nine points 0.05 K below rainfrze per Dcs (the switch-on),
# 0 elsewhere.  Ice share: those points, wherever MG2 makes ice, and the whole
# share (1.0) on the 480 um drops at Dcs 450 um, 1.067 Dcs, which MG2 sends to
# snow and the ramp partly to ice.
_SW = 0.8964843749999399


def _rfz_gap(ice):
    g = np.zeros((3, 4, 3, 3))
    g[:, 2] = _SW
    if ice:
        g[1, 2, :, 2] = 0.0             # Dcs 450: the bound drop goes to snow
        g[2, 2, :, 1:] = 0.0            # Dcs 200: 480 um and the bound to snow
        g[1, :3, :, 1] = 1.0            # Dcs 450, 480 um, frozen: in the ramp
    return g.ravel()


_RFZ_DEFAULT_GAP = _rfz_gap(False)
_RFZ_DEFAULT_GAP_ICE = _rfz_gap(True)


def test_rain_freeze_default_gap_per_point():
    """At the production widths the frozen rain departs from MG2 only where
    the switch-on and the rounded heat cap act; the gap per point is the
    recorded one (RULE 5)."""
    fx = _fx()
    from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import _production_morrison
    cfg = resolve_morrison_flavor(_production_morrison())
    width = W.RAMP_WIDTH_PER_SHARPNESS / cfg.homogeneous_freeze_sharpness
    lg = _rain_freeze(fx, width, W.MG2_CAP_SHARPNESS, cfg.mg2_frozen_rain_ice_ramp_frac)
    gap = _rel(lg["qrfrz"], fx["rfz_out_qrfrz"])
    gap_ice = _rel(lg["qifrz"], fx["rfz_out_qifrz"])
    print("default gap", repr(gap.tolist()), "ice", repr(gap_ice.tolist()))
    np.testing.assert_allclose(gap, _RFZ_DEFAULT_GAP, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(gap_ice, _RFZ_DEFAULT_GAP_ICE, rtol=1e-6, atol=1e-12)


def test_rain_freeze_gradient_at_default():
    """Finite, nonzero derivative of the frozen rain through the -40 C switch
    and the heat cap at the production width."""
    from jax.test_util import check_grads
    width = W.RAMP_WIDTH_PER_SHARPNESS / 20.0

    def f(T, q):
        return W.mg2_instant_rain_freeze(T, q, 1.0e6 * q, width)[0]
    T0 = constants.T_freeze - W.MG2_RAINFRZE_OFFSET_K - 0.1
    for q in (1.0e-5, 1.0e-3):                 # switch-on, then the heat cap
        g = jax.grad(f, argnums=(0, 1))(jnp.asarray(T0), jnp.asarray(q))
        assert all(np.isfinite(x) for x in g) and abs(float(g[0])) > 0.0, g
        check_grads(f, (jnp.asarray(T0), jnp.asarray(q)), order=2, modes=("fwd", "rev"),
                    eps=1.0e-6 * min(q, 1e-2), atol=1e-6, rtol=1e-4)


def _rain_prefreeze(fx, tag, width_K, cap):
    """legoESM's start-of-step rain freezing on the rfz or rfp points: frozen
    rain and number over the call and the heating."""
    dt = float(fx["isc_dt"])
    m, n = W.mg2_instant_rain_freeze(jnp.asarray(fx[f"{tag}_in_t"]),
                                     jnp.asarray(fx[f"{tag}_in_dumr"]),
                                     jnp.asarray(fx[f"{tag}_in_dumnr"]), width_K, cap)
    m, n = np.asarray(m), np.asarray(n)
    return dict(minstrf=m, ninstrf=n, tlat0=constants.L_f * m / dt)


def test_rain_prefreeze_against_mg2():
    """MG2's start-of-step rain freezing at -40 C (micro_mg2_0.F90:1182-1212,
    verbatim in mg2_rain_frz_pts.F90; it runs after the start-of-step snow
    melt :1147-1169 and tests the bare t) at the hard limits: the 48 rfz
    points and the 8 rfp points 0.01 to 1 K below rainfrze with 3e-3 kg/kg of
    rain, where the freezing is heat-limited.  Frozen rain, number and
    heating match; all of it goes to snow (:1207-1210)."""
    fx = _fx()
    errs = {}
    for tag in ("rfz", "rfp"):
        lg = _rain_prefreeze(fx, tag, _DT_FORM_S, math.inf)
        for k in ("minstrf", "ninstrf", "tlat0"):
            errs[f"{tag}_{k}"] = float(np.max(_rel(lg[k], fx[f"{tag}_out_{k}"])))
    lim = fx["rfp_out_minstrf"] < 0.99 * fx["rfp_in_dumr"]
    assert (lim & (fx["rfp_out_minstrf"] > 0)).sum() == 3           # heat-limited
    assert (fx["rfp_out_minstrf"] == 0).sum() == 2                  # above rainfrze
    print("max relative error", errs)
    assert max(errs.values()) <= TOL, errs


# Default-switch gap per rfp point (relative error of the start-of-step frozen
# rain against MG2) at the 0.2 K switch-on and the rounded heat cap.  Measured
# 2026-11-02 on this fixture: 1 K below rainfrze 0, and 0.0053 where the
# rounded cap meets the full freezing; 0.1 K below half the freezing is
# missing (the switch-on), 0.01 K below 0.9988; above rainfrze 0.
_RFP_DEFAULT_GAP = np.array([0.0, 0.005318910006701753, 0.5000000000000533, 0.5000000000000533,
                             0.9988418750000031, 0.9988418750000031, 0.0, 0.0])


def test_rain_prefreeze_default_gap_per_point():
    """At the production widths the start-of-step freezing departs from MG2
    only inside the switch-on and where the rounded cap acts (RULE 5)."""
    fx = _fx()
    from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import _production_morrison
    cfg = resolve_morrison_flavor(_production_morrison())
    width = W.RAMP_WIDTH_PER_SHARPNESS / cfg.homogeneous_freeze_sharpness
    lg = _rain_prefreeze(fx, "rfp", width, W.MG2_CAP_SHARPNESS)
    gap = _rel(lg["minstrf"], fx["rfp_out_minstrf"])
    print("default gap", repr(gap.tolist()))
    np.testing.assert_allclose(gap, _RFP_DEFAULT_GAP, rtol=1e-6, atol=1e-12)
