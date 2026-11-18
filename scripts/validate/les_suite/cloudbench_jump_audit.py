"""Audit of the sharp theta_l / q_t steps in the CloudBench day-5 profiles.

Numbers only, no verdict. For each case it writes:

* ``init``: the SCM starting column (theta_l, q_t on its own hydrostatic heights)
  against the LES frame 0 interpolated to those heights, and the hydrostatic
  heights against the sounding heights the grid was laid on;
* ``forcing_check``: the SCM forcing tendency of theta_l and q_t at the start
  (``compute_forcing_tendencies``) against an independent numpy rebuild from the
  artifact (first-order upwind subsidence, recorded advection, nudging to frame 0,
  LES radiation as the window-width-weighted mean of the LES frames over Pi);
* ``budget_<arm>``: per step, the column water and moist-energy change rebuilt from
  the recorded states against dt * (surface law + forcing - rain at the ground),
  plus the run totals of each supply term;
* ``les_budget``: the LES's own day-5 budget per height band: the fitted
  d<X>/dt against -(1/rho) d(rho F)/dz + subsidence + advection + nudging
  (+ radiation / Pi for theta_l), with rho from the sounding; rain fallout is not
  recorded, so it is part of the residual;
* profiles (npz): raw LES day-5 mean, LES frame 0, LES day-5 mean sampled on the
  SCM levels and re-interpolated, SCM forcing-only (no turbulence, no surface flux,
  microphysics on) and tuned YSU, all day-5 means on the LES heights; plus the
  height and size of the largest theta_l gradient between 0.3 and 5 km.

Run with JAX_ENABLE_X64=1 from the repo worktree on a compute node.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "scripts" / "validate" / "les_suite"))
import cosci_eval as ce  # noqa: E402
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel  # noqa: E402
from legoesm.atmosphere.forcing.scm.scm_forcing import compute_forcing_tendencies  # noqa: E402
from legoesm.atmosphere.les_suite.candidate_eval import _frames, e2_nsteps, e2_profiles  # noqa: E402
from legoesm.atmosphere.les_suite.cloudbench import parse_case_name, read_sounding  # noqa: E402
from legoesm.atmosphere.les_suite.cloudbench_scm import build_cloudbench_scm  # noqa: E402
from legoesm.atmosphere.les_suite.conservation import check_conservation  # noqa: E402
from legoesm.atmosphere.les_suite.scm_coupling import interp_profile  # noqa: E402
from legoesm.atmosphere.les_suite.scm_runner import scm_physics_config  # noqa: E402
from legoesm.atmosphere.physics._shared import compute_heights_from_sigma  # noqa: E402

from legoesm import constants  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/glade/derecho/scratch/pg2328/cosci/round2/audit")
ART = Path("/glade/derecho/scratch/pg2328/les_suite/artifacts/cloudbench")
FORCING = Path("/glade/derecho/scratch/pg2328/les_suite/external/cloudbench/forcing")
STATS = Path("/glade/derecho/scratch/pg2328/les_suite/external/cloudbench/cloudbench_statistics.nc")
NLEV, DT = 32, 600.0
YSU_TUNED = {"Ri_crit": 0.6371469884649477, "countergrad_coeff": 16.817622352747875,
             "entrainment_ratio": 0.4525497225274592}
CASES = {"stratocumulus": "cloudbench_s292_m01_amip.npz",
         "shallow_cumulus": "cloudbench_s118_m10_amip.npz"}
BANDS = ((0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 6000))


def _np(x):
    return np.asarray(x, dtype=np.float64)


def _col(field):
    return _np(field.data).reshape(-1)          # top-to-bottom


def theta_l_qt(temp, q_v, q_c, q_r, exner):
    q_r = 0.0 if q_r is None else q_r
    theta_l = temp / exner - constants.L_v * q_c / (constants.c_pd * exner)
    return theta_l, q_v + q_c + q_r


def max_gradient(z, x, lo=300.0, hi=5000.0):
    g = np.diff(x) / np.diff(z)
    zc = 0.5 * (z[1:] + z[:-1])
    m = (zc >= lo) & (zc <= hi)
    if not m.any():
        raise ValueError("no interval inside the gradient window")
    i = np.argmax(np.where(m, np.abs(g), -1.0))
    return {"z_m": float(zc[i]), "dxdz_per_km": float(g[i] * 1e3)}


def init_check(scm, grid, art):
    st = scm.state
    temp, q_v = _col(st.T), _col(st.tracers["q_v"])
    q_c = _col(st.tracers["q_c"])
    q_r = _col(st.tracers["q_r"]) if "q_r" in st.tracers else None
    sc = scm.sigma_coord
    p_s = st.p_s.data.reshape(())
    p_full, p_half = _np(sc.pressure_at_full(p_s)), _np(sc.pressure_at_half(p_s))
    exner = (p_full / constants.p_ref) ** constants.kappa
    thl, qt = theta_l_qt(temp, q_v, q_c, q_r, exner)
    z_h, _ = compute_heights_from_sigma(jnp.asarray(temp)[None], jnp.asarray(p_half)[None],
                                        q_v=jnp.asarray(q_v)[None])
    z_h = _np(z_h).reshape(-1)
    z_les = _np(art.heights_m)
    les_thl = np.interp(z_h, z_les, _np(art.theta)[0])
    les_qt = np.interp(z_h, z_les, _np(art.qt)[0])
    return {"max_abs_theta_l_K": float(np.max(np.abs(thl - les_thl))),
            "max_abs_q_t_g_kg": float(1e3 * np.max(np.abs(qt - les_qt))),
            "max_abs_height_hydrostatic_minus_sounding_m":
                float(np.max(np.abs(z_h - _np(grid.z_scm_m)))),
            "lowest_level_m": float(z_h[-1]), "top_level_m": float(z_h[0])}, z_h, exner


def rad_mean(art):
    """Width-weighted mean of the LES radiation frames (frame 0 is NaN)."""
    t, r = _np(art.times_s)[1:], _np(art.rad_T_tendency)[1:]
    edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
    w = np.diff(edges)
    return (r * w[:, None]).sum(0) / w.sum()


def forcing_check(scm, art, z_h, z_grid, exner):
    st = scm.state
    f = compute_forcing_tendencies(st, scm.sigma_coord, scm.forcing_arg, 0.0)
    code_thl = _np(f.dT_dt.data).reshape(-1) / exner          # q_c held: dtheta_l = dT/Pi
    code_qt = _np(f.tracer_tendencies["q_v"].data).reshape(-1)
    q_c = _col(st.tracers["q_c"])
    q_r = _col(st.tracers["q_r"]) if "q_r" in st.tracers else None
    thl, qt = theta_l_qt(_col(st.T), _col(st.tracers["q_v"]), q_c, q_r, exner)
    qt_noqr = _col(st.tracers["q_v"]) + q_c                  # the forcing's q_t (no rain)
    z_les = _np(art.heights_m)

    def on(p):                     # the builder puts every forcing profile on the grid heights
        return np.interp(z_grid, z_les, _np(p))
    w = on(art.subsidence_w)
    zf = z_h                                                   # top-to-bottom, decreasing

    def subs(x):
        out = np.zeros_like(x)
        for k in range(len(x)):
            if w[k] < 0.0:                                     # donor above
                out[k] = 0.0 if k == 0 else -w[k] * (x[k - 1] - x[k]) / (zf[k - 1] - zf[k])
            else:                                              # donor below
                out[k] = 0.0 if k == len(x) - 1 else -w[k] * (x[k] - x[k + 1]) / (zf[k] - zf[k + 1])
        return out
    rate = on(art.nudge_rate)
    thl0, qt0 = on(_np(art.theta)[0]), on(_np(art.qt)[0])
    ref_thl = (subs(thl) + on(art.theta_adv) + rate * (thl0 - thl)
               + on(rad_mean(art)) / exner)
    ref_qt = subs(qt_noqr) + on(art.qv_adv) + rate * (qt0 - qt_noqr)
    scale_t = np.max(np.abs(ref_thl))
    scale_q = np.max(np.abs(ref_qt))
    return {"max_abs_diff_theta_l_K_day": float(86400 * np.max(np.abs(code_thl - ref_thl))),
            "max_theta_l_tendency_K_day": float(86400 * scale_t),
            "max_abs_diff_q_t_g_kg_day": float(86400e3 * np.max(np.abs(code_qt - ref_qt))),
            "max_q_t_tendency_g_kg_day": float(86400e3 * scale_q),
            "profile_z_m": z_h.tolist(),
            "code_theta_l_K_day": (86400 * code_thl).tolist(),
            "ref_theta_l_K_day": (86400 * ref_thl).tolist(),
            "code_q_t_g_kg_day": (86400e3 * code_qt).tolist(),
            "ref_q_t_g_kg_day": (86400e3 * ref_qt).tolist()}


def state_budget(rec, dt):
    """Column water / moist energy from the recorded states vs the declared supply."""
    b = {k: _np(v) for k, v in rec.budget.items()}
    n = b["rate_water"].shape[0]
    cols = [rec.at(k) for k in range(n + 1)]
    lv, cp = constants.L_v, constants.c_pd
    W = np.array([float(jnp.sum(c.mass * c.q_t)) for c in cols])
    E = np.array([float(jnp.sum(c.mass * (cp * c.temp + lv * c.q_v
                                          + 0.5 * (c.u ** 2 + c.v ** 2)))) for c in cols])
    sup_w = b["sfc_water"] + b["forcing_water"] - b["precip"]
    sup_e = b["sfc_energy"] + b["forcing_energy"]
    rw = np.diff(W) - dt * sup_w
    re = np.diff(E) - dt * sup_e
    return {"steps": int(n),
            "water_max_step_residual_rel": float(np.max(np.abs(rw)) / np.max(np.abs(W))),
            "water_run_residual_rel": float(abs(rw.sum()) / np.max(np.abs(W))),
            "energy_max_step_residual_rel": float(np.max(np.abs(re)) / np.max(np.abs(E))),
            "energy_run_residual_rel": float(abs(re.sum()) / np.max(np.abs(E))),
            "run_total_kg_m2": {"surface": float(dt * b["sfc_water"].sum()),
                                "forcing": float(dt * b["forcing_water"].sum()),
                                "rain": float(-dt * b["precip"].sum()),
                                "column_change": float(W[-1] - W[0])},
            "run_total_MJ_m2": {"surface": float(dt * b["sfc_energy"].sum() / 1e6),
                                "forcing": float(dt * b["forcing_energy"].sum() / 1e6),
                                "column_change": float((E[-1] - E[0]) / 1e6)}}


def _upwind_dxdz(x, z, w):
    """First-order upwind d x/dz on surface-first rows (..., nz); w < 0 donor above,
    w > 0 donor below; a missing donor at an end gives 0."""
    g = np.diff(x, axis=-1) / np.diff(z)
    above = np.concatenate([g, np.zeros_like(g[..., :1])], axis=-1)
    below = np.concatenate([np.zeros_like(g[..., :1]), g], axis=-1)
    return np.where(w < 0.0, above, below)


def les_budget(art, sounding):
    """Day-5 LES budget: (X_last - X_first)/(t_last - t_first) against the trapezoid
    time mean of the per-frame supplies over the same frames."""
    z = _np(art.heights_m)
    t = _np(art.times_s)[1:]
    rho = np.interp(z, sounding["z"], sounding["rho"])
    exner = (np.interp(z, sounding["z"], sounding["p"]) / constants.p_ref) ** constants.kappa
    w = _np(art.subsidence_w)
    rate = _np(art.nudge_rate)
    rad = _np(art.rad_T_tendency)[1:] / exner

    def tmean(a):
        return np.trapezoid(a, t, axis=0) / (t[-1] - t[0])
    out = {}
    for key, fr, fs, adv, extra in (
            ("theta_l", "wtheta_resolved", "wtheta_sgs", _np(art.theta_adv), rad),
            ("q_t", "wqt_resolved", "wqt_sgs", _np(art.qv_adv), np.zeros_like(rad))):
        x = _np(getattr(art, "theta" if key == "theta_l" else "qt"))
        x0, xs = x[0], x[1:]
        observed = (xs[-1] - xs[0]) / (t[-1] - t[0])
        flux = (_np(getattr(art, fr)) + _np(getattr(art, fs)))[1:]
        conv = tmean(-np.gradient(rho * flux, z, axis=1) / rho)
        subs = tmean(-w * _upwind_dxdz(xs, z, w))
        nud = tmean(rate * (x0 - xs))
        extra_m = tmean(extra)
        res = observed - (conv + subs + adv + nud + extra_m)
        bands = {}
        for lo, hi in BANDS:
            m = (z >= lo) & (z < hi)
            wt = rho[m]

            def mean(a):
                return float((a[m] * wt).sum() / wt.sum() * 86400 * (1e3 if key == "q_t" else 1))
            bands[f"{lo}-{hi}"] = {"observed": mean(observed), "turb_flux_conv": mean(conv),
                                   "subsidence": mean(subs), "advection": mean(adv),
                                   "nudging": mean(nud), "radiation": mean(extra_m),
                                   "residual": mean(res)}
        out[key] = {"units": "K/day" if key == "theta_l" else "g/kg/day",
                    "any_w_up": bool(np.any(w > 0.0)), "bands": bands}
    return out


def run_forcing_only(scm_ref, art, turb):
    """No turbulence, no surface flux; forcing and microphysics as in the scored run."""
    st = scm_ref.state
    turb_none = turb._replace(scheme="none")
    forcing = scm_ref.forcing_arg._replace(prescribe="none", flux_to_closure=False,
                                           surface_flux_fn=None)
    p_s = float(st.p_s.data.reshape(()))
    sc = scm_ref.sigma_coord
    col = SingleColumnModel.create(
        physics_config=scm_physics_config(art, turb_none), nlev=NLEV, dt=DT,
        T_profile=jnp.asarray(_col(st.T)), q_v_profile=jnp.asarray(_col(st.tracers["q_v"])),
        q_c_profile=jnp.asarray(_col(st.tracers["q_c"])),
        q_r_profile=(jnp.asarray(_col(st.tracers["q_r"])) if "q_r" in st.tracers else None),
        u=jnp.asarray(_col(st.u)), v=jnp.asarray(_col(st.v)), p_s=p_s,
        sigma_top=float(_np(sc.sigma_half)[0]), forcing=forcing,
        time_integrator="forward_euler", dtype=jnp.float64)
    if not np.allclose(_np(col.sigma_coord.sigma_full), _np(sc.sigma_full), rtol=0, atol=1e-14):
        raise SystemExit("forcing-only column is not on the scored column's levels")
    steps = set(_frames(art, DT))
    thl_s, qt_s, z_s = [], [], []
    p_full = _np(sc.pressure_at_full(jnp.asarray(p_s)))
    p_half = _np(sc.pressure_at_half(jnp.asarray(p_s)))
    exner = (p_full / constants.p_ref) ** constants.kappa
    # rec.at(s) is the state AFTER s steps; match that
    for k in range(max(steps) + 1):
        if k in steps:
            s = col.state
            q_v, q_c = _col(s.tracers["q_v"]), _col(s.tracers["q_c"])
            q_r = _col(s.tracers["q_r"]) if "q_r" in s.tracers else None
            thl, qt = theta_l_qt(_col(s.T), q_v, q_c, q_r, exner)
            zf, _ = compute_heights_from_sigma(jnp.asarray(_col(s.T))[None],
                                               jnp.asarray(p_half)[None],
                                               q_v=jnp.asarray(q_v)[None])
            thl_s.append(thl)
            qt_s.append(qt)
            z_s.append(_np(zf).reshape(-1))
        col.step()
    z_les = _np(art.heights_m)
    th = np.mean([np.interp(z_les, z[::-1], x[::-1]) for z, x in zip(z_s, thl_s)], 0)
    qq = np.mean([np.interp(z_les, z[::-1], x[::-1]) for z, x in zip(z_s, qt_s)], 0)
    return th, qq


def main() -> int:
    if not jax.config.read("jax_enable_x64"):
        raise SystemExit("run with JAX_ENABLE_X64=1")
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [ART / f for f in CASES.values()]
    arts = [(str(p), ce.load_artifact(p)) for p in paths]
    cb_prov = ce.configure_cloudbench(arts, FORCING, STATS)
    base, key = ce._roster_base("ysu")
    fields = [m.field for m in ce._scheme_tunable_metas(key, (1,))]
    yarm = ce.Arm(base)
    summary = {"git_sha": ce._code_id(), "nlev": NLEV, "dt": DT, "cloudbench": cb_prov,
               "cases": {}}
    for (path, art), regime in zip(arts, CASES):
        name = Path(path).name
        site = parse_case_name(art.case_name)[0]
        par = art.les_setup["parameters"]
        sounding = read_sounding(FORCING / str(site) / str(int(par["month"]))
                                 / par["experiment"] / "sounding.csv")
        scm, grid = build_cloudbench_scm(art, ce._case_turb(base, art),
                                         forcing_dir=FORCING, nlev=NLEV, dt=DT)
        case = {"regime": regime}
        case["init"], z_h, exner = init_check(scm, grid, art)
        case["forcing_check"] = forcing_check(scm, art, z_h, _np(grid.z_scm_m), exner)
        case["les_budget"] = les_budget(art, sounding)
        n = e2_nsteps(art, DT)
        prep = yarm.prepare(art, nlev=NLEV, dt=DT)
        rec = yarm.run_with_turb(prep, art, n,
                                 ce.apply_overrides_to_base(base, dict(
                                     zip(fields, [YSU_TUNED[f] for f in fields]))))[0]
        gate = check_conservation(rec, DT)
        case["gate_ysu_tuned"] = {"passed": bool(gate.passed), "reason": gate.reason}
        case["budget_ysu_tuned"] = state_budget(rec, DT)
        prof = e2_profiles(art, rec, DT)
        z_les = _np(art.heights_m)
        save = {"heights_m": z_les,
                "les_day5_theta_l": _np(prof.truth["theta"]).mean(0),
                "les_day5_q_t": _np(prof.truth["qt"]).mean(0),
                "les_frame0_theta_l": _np(art.theta)[0], "les_frame0_q_t": _np(art.qt)[0],
                "ysu_tuned_theta_l": _np(prof.model["theta"]).mean(0),
                "ysu_tuned_q_t": _np(prof.model["qt"]).mean(0)}
        zs = z_h[::-1]
        for v in ("theta_l", "q_t"):
            save[f"les_on_scm_{v}"] = np.interp(z_les, zs, np.interp(zs, z_les, save[f"les_day5_{v}"]))
        save["forcing_only_theta_l"], save["forcing_only_q_t"] = run_forcing_only(
            scm, art, ce._case_turb(base, art))
        case["max_theta_l_gradient"] = {k[:-8]: max_gradient(z_les, save[k])
                                        for k in save if k.endswith("_theta_l")}
        case["max_q_t_gradient_g_kg"] = {k[:-4]: {kk: (vv * 1e3 if kk != "z_m" else vv)
                                                  for kk, vv in max_gradient(z_les, save[k]).items()}
                                         for k in save if k.endswith("_q_t")}
        save["provenance"] = np.asarray(json.dumps({"git_sha": summary["git_sha"], "case": name}))
        np.savez(OUT / f"audit_{name}", **save)
        summary["cases"][name] = case
        print(name, json.dumps({k: case[k] for k in ("init", "gate_ysu_tuned")}), flush=True)
        jax.clear_caches()
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
