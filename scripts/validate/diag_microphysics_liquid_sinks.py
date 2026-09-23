"""Term-by-term microphysical sinks on cloud water, both arms, same state.

    diag_microphysics_liquid_sinks.py --config <deck> --restart <ckpt.npz> \
        [--label X] [--band 500 800] [--dt-split]

Four candidates for the CAM6 AMIP arm's 46% cloud-water deficit have been
measured and excluded: vertical resolution, the diagnostic condensate floor, the
cloud-fraction path as a sink, and convective detrainment.  The microphysical
sink is what is left.

The probe calls the deck's OWN microphysics on the restored state through the
production resolver, and reports, per term, the rate on cloud water AND that rate
as a fraction of the cloud water present, because a larger absolute sink on a
smaller reservoir is the signature being looked for and a rate alone hides it.
Autoconversion and accretion are evaluated by calling the SAME repository
functions the scheme calls, with the scheme's own in-cloud scaling; the remaining
sinks (the transfer to ice, evaporation, sedimentation) are reported as one
residual closed against the scheme's net tendency, so nothing is unaccounted for.

``--dt-split`` separates scheme from timestep: the SAME scheme on the SAME
columns, once at the long step against five short steps of the same total
elapsed time.  Autoconversion is non-linear in cloud water, so an apparent scheme
difference can be a timestep difference.  NUMBERS ONLY -- no verdict.

TRAP, 2026-09-23: this split originally printed ONLY the net cloud-water
tendency, which carries the saturation-adjustment condensation SOURCE
alongside the removal sinks (morrison.py:1418).  A 2.76x net difference was
read as the warm-rain sinks over-stripping a long step; resolving the terms
showed the gross sinks move by <=6.5%, and at the CAM6 deck's real 600 s step
the band is a net cloud-water SOURCE at both step lengths.  The per-term
columns below exist so that source can never again be mistaken for a sink.
The per-term rates are PRE-donor-clamp, so the binding count is printed too:
where the clamp binds the applied removal is smaller than the rate shown.
"""
from __future__ import annotations
import argparse, importlib.util, os, sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_DAY = 86400.0


def _load_run_amip():
    path = _ROOT / "scripts" / "run" / "run_amip.py"
    spec = importlib.util.spec_from_file_location("run_amip_sinkprobe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_amip_sinkprobe"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--restart", required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--band", nargs=2, type=float, default=[500.0, 800.0],
                    metavar=("P_LO_HPA", "P_HI_HPA"))
    ap.add_argument("--dt-split", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return ap


def weighted_rate(rate, dp, area_w, g):
    """Area-weighted column integral of a mixing-ratio rate [kg/m^2/s]."""
    return float(np.sum(area_w * np.sum(rate * dp, axis=1) / g))


def masked_rate(rate, dp, area_w, g, mask):
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

    state = driver.state
    model = driver.model
    mesh = model.mesh
    sc = model.sigma_coord
    T = jnp.asarray(state.T.data)
    ncol, nlev = T.shape
    area = np.asarray(mesh.areaCell).reshape(ncol)
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

    q_v = trac("q_v"); q_c = trac("q_c"); q_r = trac("q_r")
    from legoesm.atmosphere.physics._shared import (
        compute_rho as _rho_of, compute_layer_dz as _dz_of)
    rho = _rho_of(T, p_full, q_v=q_v)
    dz = _dz_of(T, p_half, q_v=q_v)

    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    from legoesm.atmosphere.physics.microphysics.integration import (
        number_per_mass_to_per_volume,
    )
    hyd = HydrometeorState(
        q_c=q_c, q_r=q_r, q_i=trac("q_i"), q_s=trac("q_s"), q_g=trac("q_g"),
        N_c=number_per_mass_to_per_volume(trac("N_c"), rho),
        N_r=number_per_mass_to_per_volume(trac("N_r"), rho),
        N_i=trac("N_i"),
    )

    from legoesm.driver.physics_pipeline import _resolve_microphysics
    micro_fn, mcfg = _resolve_microphysics(config)
    dt_phys = float(config.dycore.dt) * int(getattr(config, "physics_update_steps", 1) or 1)
    n_macmic = int(getattr(config, "cld_macmic_num_steps", 1) or 1)
    dt_micro = dt_phys / max(1, n_macmic)
    print(f"[{a.label}] microphysics = {config.microphysics}; physics step {dt_phys:.1f} s, "
          f"{n_macmic} macro/micro sub-step(s) => micro dt {dt_micro:.1f} s")

    out = micro_fn(T, q_v, hyd, p_full, p_half, rho, dz, dt_micro, mcfg)
    net = np.asarray(out.dq_c_dt)
    src = out.dq_v_to_qc_dt
    src = np.asarray(src) if src is not None else None

    # --- the two warm-rain sinks, via the scheme's own functions -------------
    from legoesm.atmosphere.physics.microphysics._warm_rain import (
        effective_Nc, autoconversion_kk2000, accretion_kk2000,
        autoconversion_sb, accretion as accretion_sb_fn,
        autoconversion_sb2001, accretion_sb2001,
    )
    from legoesm.atmosphere.physics.microphysics.morrison import (
        resolve_morrison_flavor,
    )
    from legoesm.thermo import saturation_mixing_ratio
    cfg = resolve_morrison_flavor(mcfg)
    N_c_eff = effective_Nc(hyd.N_c, cfg.Nc_0,
                           predict_Nc=getattr(cfg, "predict_Nc", False),
                           nc_specified_field=getattr(cfg, "nc_from_aerosol", False))
    if getattr(cfg, "subgrid_autoconversion", False):
        q_sat = saturation_mixing_ratio(T, p_full)
        RH = q_v / jnp.maximum(q_sat, 1.0e-10)
        arg = (1.0 - RH) / max(1.0 - cfg.subgrid_rh_crit, 1.0e-6)
        arg_safe = jnp.where(arg > 0.0, arg, 1.0)
        cf_sg = jnp.where(arg > 0.0, 1.0 - jnp.sqrt(arg_safe), 1.0)
        cf_eff = jnp.clip(cf_sg, cfg.subgrid_cf_min, 1.0)
    else:
        cf_eff = jnp.ones_like(q_c)
    q_c_ic = q_c / cf_eff
    q_r_ic = q_r / cf_eff
    scheme = getattr(cfg, "warm_rain_scheme", "kk2000")
    if scheme == "kk2000":
        dq_au, _, _ = autoconversion_kk2000(q_c_ic, N_c_eff, rho, dt_micro)
        dq_au = dq_au * cf_eff
        dq_ac = accretion_kk2000(q_c_ic, q_r_ic) * cf_eff
    elif scheme == "seifert_beheng_sb2001":
        dq_au, _, _ = autoconversion_sb2001(q_c_ic, q_r_ic, N_c_eff, rho)
        dq_au = dq_au * cf_eff
        dq_ac = accretion_sb2001(q_c_ic, q_r_ic, rho) * cf_eff
    elif scheme == "seifert_beheng":
        dq_au, _, _ = autoconversion_sb(q_c_ic, N_c_eff, rho, cfg.k_au, cfg.x_star,
                                        cfg.autoconversion_sharpness)
        dq_au = dq_au * cf_eff
        dq_ac = accretion_sb_fn(q_c_ic, q_r_ic, rho, cfg.k_ac) * cf_eff
    else:
        # Dispatch hardening: the probe must never silently score a DIFFERENT
        # closure than the scheme runs (morrison.py:301-338 raises likewise).
        raise SystemExit(
            f"diag_microphysics_liquid_sinks: unknown warm_rain_scheme "
            f"{scheme!r}; the probe must mirror morrison.py's dispatch exactly")
    print(f"[{a.label}] warm-rain scheme = {scheme}; "
          f"sub-grid in-cloud closure {'ON' if getattr(cfg,'subgrid_autoconversion',False) else 'OFF'}")

    au = np.asarray(dq_au); ac = np.asarray(dq_ac)
    qcn = np.asarray(q_c)
    pfn = np.asarray(p_full)
    lo, hi = a.band[0] * 100.0, a.band[1] * 100.0
    band = (pfn >= lo) & (pfn < hi)
    allm = np.ones_like(band, dtype=bool)

    for name, mask in (("global", allm), (f"{a.band[0]:.0f}-{a.band[1]:.0f} hPa", band)):
        res = masked_rate(qcn, dp, A, g, mask)
        qrres = masked_rate(np.asarray(q_r), dp, A, g, mask)
        print(f"[{a.label}] --- {name}: cloud water present {res:.6e} kg/m2, "
              f"rain present {qrres:.6e} kg/m2")
        netv = masked_rate(-net, dp, A, g, mask) * _DAY
        auv = masked_rate(au, dp, A, g, mask) * _DAY
        acv = masked_rate(ac, dp, A, g, mask) * _DAY
        # Closure: net sink = autoconversion + accretion + everything else,
        # where "everything else" nets the condensation SOURCE against the
        # transfer to ice, evaporation and sedimentation.  The scheme does not
        # publish those terms separately on this path, so they stay combined
        # and the budget closes by construction.
        other = netv - auv - acv
        print(f"[{a.label}]   {'autoconversion':30s} {auv:12.6e} kg/m2/day   "
              f"{auv/max(res,1e-30):9.4f} per day of the reservoir")
        print(f"[{a.label}]   {'accretion':30s} {acv:12.6e} kg/m2/day   "
              f"{acv/max(res,1e-30):9.4f} per day of the reservoir")
        print(f"[{a.label}]   {'all other terms, net':30s} {other:12.6e} kg/m2/day   "
              f"{other/max(res,1e-30):9.4f} per day  "
              f"(negative = net source: condensation beats ice+evaporation+sedimentation)")
        print(f"[{a.label}]   {'NET sink on cloud water':30s} {netv:12.6e} kg/m2/day   "
              f"{netv/max(res,1e-30):9.4f} per day")


    def _warm_rain_terms(Tx, qvx, qcx, qrx, rhox, dtx):
        """Autoconversion and accretion rates alone, same closure as above."""
        if getattr(cfg, "subgrid_autoconversion", False):
            qs_ = saturation_mixing_ratio(Tx, p_full)
            rh_ = qvx / jnp.maximum(qs_, 1.0e-10)
            ar_ = (1.0 - rh_) / max(1.0 - cfg.subgrid_rh_crit, 1.0e-6)
            as_ = jnp.where(ar_ > 0.0, ar_, 1.0)
            cfx = jnp.clip(jnp.where(ar_ > 0.0, 1.0 - jnp.sqrt(as_), 1.0),
                           cfg.subgrid_cf_min, 1.0)
        else:
            cfx = jnp.ones_like(qcx)
        qci, qri = qcx / cfx, qrx / cfx
        if scheme == "kk2000":
            d_au, _, _ = autoconversion_kk2000(qci, N_c_eff, rhox, dtx)
            d_ac = accretion_kk2000(qci, qri)
        elif scheme == "seifert_beheng_sb2001":
            d_au, _, _ = autoconversion_sb2001(qci, qri, N_c_eff, rhox)
            d_ac = accretion_sb2001(qci, qri, rhox)
        else:
            d_au, _, _ = autoconversion_sb(qci, N_c_eff, rhox, cfg.k_au,
                                           cfg.x_star, cfg.autoconversion_sharpness)
            d_ac = accretion_sb_fn(qci, qri, rhox, cfg.k_ac)
        return d_au * cfx, d_ac * cfx

    if a.dt_split:
        print(f"[{a.label}] --- timestep split, same scheme, same columns, same elapsed time")
        for label, dts, nrep in ((f"one call at {dt_micro:.1f} s", dt_micro, 1),
                                 (f"five calls at {dt_micro/5:.1f} s", dt_micro / 5.0, 5)):
            Tc, qvc, qcc, qrc = T, q_v, q_c, q_r
            hy = hyd
            acc = np.zeros_like(qcn)
            acc_ac = np.zeros_like(qcn)
            acc_au = np.zeros_like(qcn)
            acc_cd = np.zeros_like(qcn)
            cd_available = [True]
            n_clamped = [0]
            for _ in range(nrep):
                o = micro_fn(Tc, qvc, hy, p_full, p_half, rho, dz, dts, mcfg)
                _au_i, _ac_i = _warm_rain_terms(Tc, qvc, qcc, qrc, rho, dts)
                acc_au = acc_au + np.asarray(_au_i) * dts
                acc_ac = acc_ac + np.asarray(_ac_i) * dts
                acc = acc + np.asarray(o.dq_c_dt) * dts
                # Condensation SOURCE, published by the scheme, so the budget
                # closes without re-deriving it.
                _s = o.dq_v_to_qc_dt
                if _s is None:
                    cd_available[0] = False
                else:
                    acc_cd = acc_cd + np.asarray(_s) * dts
                # WARM-RAIN-ONLY reservoir exceedance count.  The scheme's
                # actual donor clamp (morrison.py:985) scales a LARGER sink set
                # -- evaporation, Bergeron, riming, homogeneous freezing -- so
                # this is a lower bound on clamp activity, not the clamp itself.
                # Where it binds, the per-term rates printed above are PRE-clamp
                # and overstate the applied removal.
                _tot = np.asarray(_au_i) + np.asarray(_ac_i)
                clamped_cells = int(np.sum((_tot * dts > np.asarray(qcc))
                                           & (np.asarray(qcc) > 0.0) & band))
                n_clamped[0] += clamped_cells
                Tc = Tc + dts * o.dT_dt
                qvc = qvc + dts * o.dq_v_dt
                qcc = qcc + dts * o.dq_c_dt
                qrc = qrc + dts * o.dq_r_dt
                hy = hy._replace(q_c=qcc, q_r=qrc)
            tot = dts * nrep
            v = masked_rate(-acc / tot, dp, A, g, band) * _DAY
            va = masked_rate(acc_ac / tot, dp, A, g, band) * _DAY
            vu = masked_rate(acc_au / tot, dp, A, g, band) * _DAY
            print(f"[{a.label}]   {label:26s} net sink over {tot:.1f} s: "
                  f"{v:12.6e} kg/m2/day in the band")
            vc = (f"{masked_rate(acc_cd / tot, dp, A, g, band) * _DAY:12.6e}"
                  if cd_available[0] else "  UNAVAILABLE")
            print(f"[{a.label}]   {'':26s}   accretion alone: {va:12.6e}   "
                  f"autoconversion alone: {vu:12.6e} kg/m2/day")
            print(f"[{a.label}]   {'':26s}   condensation SOURCE: {vc} kg/m2/day "
                  f"(UNAVAILABLE = the scheme does not publish it; NOT zero)")
            print(f"[{a.label}]   {'':26s}   warm-rain-only reservoir exceedances: "
                  f"{n_clamped[0]} band cell-steps.  The scheme's donor clamp "
                  f"(morrison.py:985) also carries evaporation, freezing and ice "
                  f"collection, so this is a LOWER bound on clamp activity and "
                  f"says nothing about how much mass each arm loses to limiting.")


if __name__ == "__main__":
    main()
