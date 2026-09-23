#!/usr/bin/env python3
"""Polar-night surface downward longwave: how much of the deficit is cloud.

The Arctic surface (ice skin AND 2 m air) runs ~10 K colder than ERA5 while the
model's downward longwave at the surface is ~30 W/m2 below ERA5.  A 33 W/m2
deficit lowers a 248 K skin by ~9.5 K (4 sigma T^3 = 3.5 W/m2/K), so the
deficit is large enough to be the whole surface bias -- but it can be cloud
opacity (cause) or the cold, dry air itself (consequence).  This replays the
run's OWN radiation on one checkpoint and separates the two:

  all-sky     resolved cloud config, exactly what the run's radiation saw
  clear-sky   no cloud kwargs at all
  q_ref=X     the condensate-aware cover floor (CloudConfig.cover_condensate_q_ref,
              off in production) so prognostic ice becomes radiatively visible
  sundqvist   the RH-only closure (production before 2026-09)

Cloud effect at the surface = all-sky - clear-sky, per cap.  Observed winter
Arctic surface longwave cloud effect is ~30-50 W/m2 under ~70 % cloud (SHEBA).
Run the same probe on an EARLY checkpoint (air still near ERA5) and a late one:
the all-sky difference between them at the same cloud settings bounds the
"cold air" share.

CONTROL: the all-sky replay's outgoing longwave over 70-90N is compared with
the run's published rlut for the same cap; a replay outside --control-tol
aborts (regime check only: single instant vs monthly mean).

Polar night: cos_zenith is pinned to the solver floor, shortwave is not
reported.  Surface temperature handed to the solver is the lowest air level,
as the sibling probes do; it enters downward longwave only through the 3 %
surface reflection.

Usage:
  polar_dlw_arms.py <run> --day 50 [--q-ref 1e-4 1e-5] [--control-tol 20]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

_VAL = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_VAL))
_spec = importlib.util.spec_from_file_location("cloud_layers", _VAL / "cloud_layers.py")
cl = importlib.util.module_from_spec(_spec)
sys.modules["cloud_layers"] = cl
_spec.loader.exec_module(cl)
rb = cl.rb

CAPS = (("75-90N", 75.0), ("70-90N", 70.0), ("60-90N", 60.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--q-ref", type=float, nargs="*", default=[1.0e-4, 1.0e-5])
    ap.add_argument("--q-c-diag", type=float, nargs="*", default=[],
                    help="arms: radiative in-cloud condensate floor q_c_diagnostic [kg/kg]")
    ap.add_argument("--t-ice-only", type=float, nargs="*", default=[],
                    help="arms: T_ice_only [K] of the diagnostic ice-fraction ramp")
    ap.add_argument("--pin-reff-um", type=float, default=None,
                    help="extra arms: repeat all-sky and every q_c_diag arm with the "
                         "liquid effective radius pinned [um] (number-convention audit)")
    ap.add_argument("--cap-floor-qc", type=float, nargs="*", default=[],
                    help="arms: poleward of --cap-floor-lat and below --cap-floor-p, raise the "
                         "layer cloud fraction to --cap-floor-cf and the grid-mean liquid path to "
                         "cf * QC * dp/g with QC [kg/kg] in-cloud liquid (radiation only): the "
                         "'opaque cap clouds' attribution arm, measured offline before it is run")
    ap.add_argument("--cap-floor-cf", type=float, default=0.8)
    ap.add_argument("--cap-floor-lat", type=float, default=70.0)
    ap.add_argument("--cap-floor-p", type=float, default=70000.0, help="[Pa] floor applies below this")
    ap.add_argument("--no-sundqvist", action="store_true")
    ap.add_argument("--control-tol", type=float, default=20.0,
                    help="max |replay - published| 70-90N rlut [W/m2]")
    args = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from legoesm import constants
    from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

    rundir = f"{rb.ROOT}/{args.run}"
    exp = json.load(open(f"{rundir}/experiment_config.json"))
    ccfg = cl.resolved_cloud_config(exp)
    lat_deg, lon_deg, area = cl.mesh_coords(exp)
    z = np.load(f"{rundir}/checkpoint_day_{args.day:04d}.npz", allow_pickle=True)
    order = cl.cell_order(z, lat_deg.size)

    def col(name):
        return np.asarray(z[name])[order] if name in z.files else None

    vg = np.asarray(z["meta_vgrid"], dtype=np.float64)
    ps = col("p_s")
    p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * ps[:, None]
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    dp = np.diff(p_half, axis=1)
    T, q_v = col("T"), col("trc_q_v")
    q_c, q_i = col("trc_q_c"), col("trc_q_i")
    if q_c is None or q_i is None:
        raise SystemExit("FATAL: checkpoint carries no q_c/q_i tracers")
    T_sfc = T[:, -1]
    skin = col("ice_T_skin")

    solver = RRTMGP.from_legoesm_config(RRTMGPConfig(
        gpoint_batch_size=16, gpoint_checkpoint=False, include_clouds=True))
    mu = jnp.full(T.shape[0], 1e-4)          # polar night; SW not reported

    def solve(kw, n_sub=1):
        # n_sub > 1: kw already holds (n_sub*ncol, nlev) subcolumn paths, so the
        # column inputs are expanded the same way and the fluxes averaged back,
        # exactly as radiation/integration.py does under max_random.  Solving
        # the grid-mean path instead (one solve) overstates the LW effect of a
        # partial cover, because LW is opaque at a few tens of g/m2.
        base = dict(T=jnp.asarray(T), p_full=jnp.asarray(p_full), p_half=jnp.asarray(p_half),
                    sfc_temperature=jnp.asarray(T_sfc), q_v=jnp.asarray(q_v),
                    cos_zenith=mu, sfc_albedo=0.6, sfc_emissivity=0.97)
        ncol = T.shape[0]
        if n_sub > 1:
            from legoesm.atmosphere.physics.clouds import subcolumns as sub
            base = sub.expand_kwargs(base, n_sub, ncol)
        out = solver.solve_columns(**base, **kw)
        if n_sub > 1:
            out = sub.average_output(out, n_sub, ncol)
        return (np.asarray(out.lw_flux_down[:, -1]), np.asarray(out.lw_flux_up[:, 0]))

    def cloud_kwargs(cfg, cap_qc=None):
        # Same call as cloud_layers.analyse_checkpoint (raw number tracers,
        # mirroring the live radiation entry), then the run's overlap treatment.
        conv = col("physstate_conv_precip") if cfg.convective_cloud else None
        props = compute_cloud_properties(T, p_full, q_v, dp, cfg,
                                         q_cloud=q_c, q_ice=q_i,
                                         n_cloud=col("trc_N_c"), n_ice=col("trc_N_i"),
                                         conv_precip=conv)
        if cap_qc is not None:
            capm = (lat_deg >= args.cap_floor_lat)[:, None] & (p_full > args.cap_floor_p)
            cf_new = jnp.where(capm, jnp.maximum(props.cloud_fraction, args.cap_floor_cf), props.cloud_fraction)
            lwp_floor = cf_new * cap_qc * jnp.asarray(dp) / constants.g
            lwp_new = jnp.where(capm, jnp.maximum(props.lwp, lwp_floor), props.lwp)
            props = props._replace(cloud_fraction=cf_new, lwp=lwp_new,
                                   lwp_lw=None if props.lwp_lw is None else jnp.where(capm, jnp.maximum(props.lwp_lw, lwp_floor), props.lwp_lw))
        kw = props.to_rrtmg_kwargs()
        lwp, iwp = cl.radiation_paths(props, cfg)          # grid-mean, for the table
        n_sub = 1
        if cfg.cloud_vertical_overlap_optics == "max_random":
            from legoesm.atmosphere.physics.clouds import subcolumns as sub
            n_sub = int(cfg.cloud_n_subcolumns)
            mask = sub.generate_subcolumns(props.cloud_fraction, n_sub)
            kw["cloud_path_liq"], kw["cloud_path_ice"] = sub.subcolumn_paths(
                mask, props.cloud_fraction, props.lwp, props.iwp)
            kw = sub.expand_kwargs(kw, n_sub, props.cloud_fraction.shape[0],
                                   keys={"cloud_r_eff_liq", "cloud_r_eff_ice"})
        else:
            kw["cloud_path_liq"], kw["cloud_path_ice"] = jnp.asarray(lwp), jnp.asarray(iwp)
        cf = np.asarray(props.cloud_fraction)
        return kw, cf, lwp, iwp, n_sub

    def cap(v, lo):
        m = lat_deg >= lo
        return float((v[m] * area[m]).sum() / area[m].sum())

    arms = [("all-sky", ccfg), ("clear-sky", None)]
    arms += [(f"q_ref={q:g}", ccfg._replace(cover_condensate_q_ref=q)) for q in args.q_ref]
    arms += [(f"q_c_diag={q:g}", ccfg._replace(q_c_diagnostic=q)) for q in args.q_c_diag]
    arms += [(f"T_ice_only={t:g}", ccfg._replace(T_ice_only=t)) for t in args.t_ice_only]
    if args.q_c_diag and args.t_ice_only:
        arms.append((f"qcd={max(args.q_c_diag):g}+Tio={min(args.t_ice_only):g}",
                     ccfg._replace(q_c_diagnostic=max(args.q_c_diag),
                                   T_ice_only=min(args.t_ice_only))))
    if ccfg.scheme != "sundqvist" and not args.no_sundqvist:
        arms.append(("sundqvist", ccfg._replace(scheme="sundqvist")))
    arms += [(f"cap_floor qc={q:g}", {"cfg": ccfg, "qc": q}) for q in args.cap_floor_qc]   # CloudConfig IS a tuple: mark with a dict
    if args.pin_reff_um is not None:
        arms += [(f"{n}|re={args.pin_reff_um:g}um", c) for n, c in arms
                 if c is not None and not isinstance(c, dict) and (n == "all-sky" or n.startswith("q_c_diag"))]

    print(f"=== {args.run} day {args.day}: surface downward LW arms "
          f"(resolved cover {ccfg.scheme}, saturation {ccfg.saturation_scheme}, "
          f"q_ref {ccfg.cover_condensate_q_ref:g}) ===")
    print(f"{'arm':>20s} " + " ".join(f"{'DLW ' + c:>12s} {'OLR ' + c:>12s} {'cover':>6s} {'LWP':>6s} {'IWP':>6s}"
                                     for c, _ in CAPS))
    res = {}
    for name, cfg in arms:
        if cfg is None:
            dlw, olr = solve({})
            cf_tot = np.zeros(T.shape[0]); iwp = np.zeros(T.shape[0]); lwp = np.zeros(T.shape[0])
        else:
            kw, cf, lwp_l, iwp_l, n_sub = (cloud_kwargs(cfg["cfg"], cfg["qc"]) if isinstance(cfg, dict)
                                           else cloud_kwargs(cfg))
            if "|re=" in name:
                kw["cloud_r_eff_liq"] = jnp.full_like(kw["cloud_r_eff_liq"], args.pin_reff_um * 1e-6)
            dlw, olr = solve(kw, n_sub)
            cf_tot = 1.0 - np.prod(1.0 - np.clip(cf, 0.0, 1.0), axis=1)
            iwp = iwp_l.sum(1) * 1e3; lwp = lwp_l.sum(1) * 1e3
        res[name] = (dlw, olr)
        print(f"{name:>20s} " + " ".join(
            f"{cap(dlw, lo):12.1f} {cap(olr, lo):12.1f} {cap(cf_tot, lo):6.3f} {cap(lwp, lo):6.1f} {cap(iwp, lo):6.1f}"
            for _, lo in CAPS))

    # CONTROL: all-sky OLR vs the run's published rlut, 70-90N
    pub = rb._load_model(args.run, "rlut")
    if pub is not None:
        plat = np.asarray(pub.lat)
        pv = np.asarray(pub["rlut"]).mean(axis=0)
        sel = plat >= 70.0
        wl = np.cos(np.deg2rad(plat[sel]))
        published = float((pv[sel].mean(axis=1) * wl).sum() / wl.sum())
        replay = cap(res["all-sky"][1], 70.0)
        d = replay - published
        flag = "OK" if abs(d) <= args.control_tol else "FAIL"
        print(f"\ncontrol 70-90N rlut: replay {replay:.1f}  published {published:.1f}  "
              f"diff {d:+.1f}  [{flag}]")
        if flag == "FAIL":
            raise SystemExit("FATAL: all-sky replay is not in the run's regime; "
                             "arm numbers withheld")
    else:
        print("\ncontrol SKIPPED: run publishes no rlut")

    print("\nsurface longwave cloud effect (all-sky - clear-sky), W/m2:")
    for c, lo in CAPS:
        print(f"  {c}: {cap(res['all-sky'][0], lo) - cap(res['clear-sky'][0], lo):+.1f}")
    if skin is not None:
        for c, lo in CAPS:
            m = (lat_deg >= lo) & (skin < constants.T_freeze_ocean - 0.5)
            if m.any():
                w = area[m]
                sk = float((skin[m] * w).sum() / w.sum())
                emit = 0.97 * float((constants.sigma_sb * skin[m] ** 4 * w).sum() / w.sum())
                dl = float((res["all-sky"][0][m] * w).sum() / w.sum())
                print(f"  {c} ice cells: skin {sk:.1f} K, emits {emit:.1f}, DLW {dl:.1f}, "
                      f"net surface LW eps*(DLW-sigmaT4) {0.97 * dl - emit:+.1f} W/m2  (n={int(m.sum())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
