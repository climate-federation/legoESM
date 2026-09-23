"""Closed cloud-water budget over a pressure band, both arms, one state.

Every source and sink of ``q_c`` that the scheme applies, named, area- and
mass-weighted over a pressure band, with the residual against the scheme's own
``dq_c_dt`` printed so nothing can hide in it.  The terms are the APPLIED
(post-donor-clamp) ones published by ``MorrisonConfig.publish_qc_budget``; a
re-derivation outside the scheme would report PRE-clamp rates and could not
close (codex review, 2026-09-23).

Also measures the three named departures from the MG2 oracle, because each is a
candidate for the CAM6 arm's cloud-water deficit and each is cheap to test on
the same state:

1. MG2 caps in-cloud cloud water at 5e-3 kg/kg (micro_mg2_0.F90:1226); we do
   not.  Where the cap would bite, our KK2000 rates (autoconversion ~ q_c^2.47)
   are evaluated at an in-cloud water content MG2 would never feed them.
2. MG2 scales vapour deposition by the liquid-lifetime fraction its limiter
   produced (:1588-1592); our clamp does not couple to deposition.
3. Our sedimentation acts on PRE-source pools (morrison.py:29-36); MG2
   sediments the POST-source hydrometeors (:2204-2211).

NUMBERS ONLY -- no verdict.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

_DAY = 86400.0


def _load_run_amip():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_ra_qcb", root / "scripts" / "run" / "run_amip.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["_ra_qcb"] = m
    spec.loader.exec_module(m)
    return m


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--restart", required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--band", nargs=2, type=float, default=[500.0, 800.0],
                    metavar=("P_LO_HPA", "P_HI_HPA"))
    ap.add_argument("--no-graupel", action="store_true",
                    help="MG2-faithful counterfactual: CAM6's MG2 carries no "
                         "graupel category at all (micro_mg_cam.F90:139), so "
                         "run the same state with do_graupel off and compare.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return ap


def band_rate(rate, dp, area_w, g, mask):
    """Area-weighted, mass-weighted column integral inside ``mask``."""
    return float(np.sum(area_w[:, None] * np.where(mask, rate * dp, 0.0)) / g)


def main():
    a = build_arg_parser().parse_args()
    import jax.numpy as jnp
    from legoesm import constants

    ra = _load_run_amip()
    argv = ["--config", a.config] + list(a.extra)
    parser = ra.build_arg_parser()
    from legoesm.driver.run_config_yaml import load_yaml_config
    keys = load_yaml_config(a.config, parser)
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    args = parser.parse_args(argv)
    args = ra._postprocess_args(args, parser, argv)
    ra._apply_spectral_scheme_fallback(args, argv, parser)
    config = ra.build_config_from_args(args)

    from legoesm.driver.model_driver import ModelDriver
    driver = ModelDriver(config)
    driver.setup()
    step0, day0 = driver.load_checkpoint(a.restart)
    print(f"[{a.label}] restored step={step0} day={day0}", flush=True)

    state, model = driver.state, driver.model
    sc = model.sigma_coord
    T = jnp.asarray(state.T.data)
    ncol, nlev = T.shape
    area = np.asarray(model.mesh.areaCell).reshape(ncol)
    A = area / area.sum()
    p_s = jnp.asarray(state.p_s.data)
    p_full = sc.pressure_at_full(p_s).reshape(ncol, nlev)
    p_half = sc.pressure_at_half(p_s).reshape(ncol, nlev + 1)
    dp = np.asarray(p_half[:, 1:] - p_half[:, :-1])
    g = constants.g

    def trac(n):
        if state.tracers is None or n not in state.tracers:
            return jnp.zeros_like(T)
        r = state.tracers[n]
        d = r.data if hasattr(r, "data") else r
        return jnp.asarray(d).reshape(ncol, nlev)

    q_v, q_c, q_r = trac("q_v"), trac("q_c"), trac("q_r")
    from legoesm.atmosphere.physics._shared import (
        compute_rho as _rho_of, compute_layer_dz as _dz_of)
    rho = _rho_of(T, p_full, q_v=q_v)
    dz = _dz_of(T, p_half, q_v=q_v)

    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    from legoesm.atmosphere.physics.microphysics.integration import (
        number_per_mass_to_per_volume)
    hyd = HydrometeorState(
        q_c=q_c, q_r=q_r, q_i=trac("q_i"), q_s=trac("q_s"), q_g=trac("q_g"),
        N_c=number_per_mass_to_per_volume(trac("N_c"), rho),
        N_r=number_per_mass_to_per_volume(trac("N_r"), rho),
        N_i=trac("N_i"))

    from legoesm.driver.physics_pipeline import _resolve_microphysics
    micro_fn, mcfg = _resolve_microphysics(config)
    dt_phys = float(config.dycore.dt) * int(
        getattr(config, "physics_update_steps", 1) or 1)
    n_macmic = int(getattr(config, "cld_macmic_num_steps", 1) or 1)
    dt_micro = dt_phys / max(1, n_macmic)
    print(f"[{a.label}] micro dt {dt_micro:.1f} s "
          f"({dt_phys:.1f} s physics / {n_macmic} macro-micro sub-step(s))")

    mcfg_b = mcfg._replace(publish_qc_budget=True)
    if a.no_graupel:
        mcfg_b = mcfg_b._replace(do_graupel=False)
        print(f"[{a.label}] COUNTERFACTUAL: do_graupel=False (MG2 carries no "
              f"graupel category; its riming of cloud water by snow, psacws, "
              f"sends the rimed mass to SNOW -- micro_mg2_0.F90:1894)")
    out = micro_fn(T, q_v, hyd, p_full, p_half, rho, dz, dt_micro, mcfg_b)
    if out.qc_budget is None:
        raise SystemExit(f"[{a.label}] scheme published no qc_budget; this "
                         f"probe requires the Morrison family")

    lo, hi = a.band[0] * 100.0, a.band[1] * 100.0
    pfn = np.asarray(p_full)
    band = (pfn >= lo) & (pfn < hi)

    # Temperature sectors: riming is gated below freezing, so a band-integrated
    # comparison dilutes it with warm layers where it is structurally zero.
    Tn = np.asarray(T)
    T_FREEZE = float(constants.T_freeze)
    sectors = (("whole band", band),
               ("cold (T<273.15 K)", band & (Tn < T_FREEZE)),
               ("warm (T>=273.15 K)", band & (Tn >= T_FREEZE)))
    for sname, mask in sectors:
        res = band_rate(np.asarray(q_c), dp, A, g, mask)
        if res <= 0.0:
            print(f"[{a.label}] --- {sname}: no cloud water")
            continue
        print(f"[{a.label}] --- {a.band[0]:.0f}-{a.band[1]:.0f} hPa, {sname}: "
              f"cloud water {res:.6e} kg/m2, "
              f"rain {band_rate(np.asarray(q_r), dp, A, g, mask):.6e} kg/m2, "
              f"{int(np.sum(mask))} cells")
        print(f"[{a.label}]   CLOSED cloud-water budget [kg/m2/day, "
              f"+ = source of cloud water]")
        tot = 0.0
        sink_rate = 0.0
        for name, term in out.qc_budget.items():
            v = band_rate(np.asarray(term), dp, A, g, mask) * _DAY
            tot += v
            if v < 0.0:
                sink_rate += -v / res
            print(f"[{a.label}]     {name:22s} {v:+13.6e}   "
                  f"{v / res:+9.4f} per day of the reservoir")
        net = band_rate(np.asarray(out.dq_c_dt), dp, A, g, mask) * _DAY
        denom = max(abs(band_rate(np.asarray(t), dp, A, g, mask) * _DAY)
                    for t in out.qc_budget.values()) or 1.0
        print(f"[{a.label}]     {'SUM OF TERMS':22s} {tot:+13.6e}")
        print(f"[{a.label}]     {'scheme dq_c_dt':22s} {net:+13.6e}")
        print(f"[{a.label}]     {'RESIDUAL':22s} {tot - net:+13.6e}   "
              f"= {abs(tot - net) / denom:.3e} of the largest term")
        print(f"[{a.label}]     total fractional sink {sink_rate:8.3f} per day "
              f"=> cloud-water residence time {24.0 / max(sink_rate, 1e-30):.3f} h")

    res = band_rate(np.asarray(q_c), dp, A, g, band)
    # --- departure 1: MG2 in-cloud cloud-water cap ------------------------
    from legoesm.atmosphere.physics.microphysics.morrison import (
        resolve_morrison_flavor)
    from legoesm.thermo import saturation_mixing_ratio
    cfg = resolve_morrison_flavor(mcfg)
    if getattr(cfg, "subgrid_autoconversion", False):
        q_sat = saturation_mixing_ratio(T, p_full)
        arg = (1.0 - q_v / jnp.maximum(q_sat, 1.0e-10)) / max(
            1.0 - cfg.subgrid_rh_crit, 1.0e-6)
        cf = jnp.clip(jnp.where(arg > 0.0, 1.0 - jnp.sqrt(jnp.where(
            arg > 0.0, arg, 1.0)), 1.0), cfg.subgrid_cf_min, 1.0)
    else:
        cf = jnp.ones_like(q_c)
    q_c_ic = np.asarray(q_c / cf)
    MG2_CAP = 5.0e-3   # micro_mg2_0.F90:1226
    over = (q_c_ic > MG2_CAP) & band
    n_band = int(np.sum(band))
    qc_np = np.asarray(q_c)
    # Cloud water sitting in cells whose in-cloud value exceeds MG2's cap.
    mass_over = band_rate(np.where(over, qc_np, 0.0), dp, A, g,
                          np.ones_like(band, dtype=bool))
    print(f"[{a.label}] --- departure 1: MG2 in-cloud cap {MG2_CAP:.1e} kg/kg")
    print(f"[{a.label}]   in-cloud q_c exceeds the cap in {int(np.sum(over))} "
          f"of {n_band} band cells "
          f"({100.0 * np.sum(over) / max(n_band, 1):.3f} %)")
    print(f"[{a.label}]   cloud water in those cells: {mass_over:.6e} kg/m2 "
          f"= {100.0 * mass_over / max(res, 1e-30):.3f} % of the band reservoir")
    print(f"[{a.label}]   in-cloud q_c: mean {np.mean(q_c_ic[band]):.4e}  "
          f"p99 {np.percentile(q_c_ic[band], 99):.4e}  "
          f"max {np.max(q_c_ic[band]):.4e} kg/kg")
    print(f"[{a.label}]   cloud fraction in band: mean "
          f"{float(jnp.mean(cf[jnp.asarray(band)])):.4f}  "
          f"min {float(jnp.min(cf[jnp.asarray(band)])):.4f}")


if __name__ == "__main__":
    main()
