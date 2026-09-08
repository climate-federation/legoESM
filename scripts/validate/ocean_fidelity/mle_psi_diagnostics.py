"""Localise the MLE d90 surface freshening: WHERE is it, and is the MLE
streamfunction there consistent with Fox-Kemper or numerically unstable?

Two discriminating measurements, both offline on existing snapshots:

1. FRESHENING MAP: native-cell SSS/SST difference (MLE arm minus control) at a
   matched day.  The two arms share IC, mesh and every flag except --mle, so
   the difference field IS the MLE effect (plus 30-90 d of divergent
   trajectory).  Its geography discriminates the candidate mechanisms:
     * colocated with the known deep-MLD excess columns (Greenland/Labrador,
       H ~ 2 km)  -> Psi ~ H^2 amplification by our MLD bias;
     * river mouths / tropical rain belts -> interaction with the freshwater
       forcing;
     * broad tropics with grid-scale noise -> centered-flux instability
       (bolus_cfl_cap defaults to 0 = no clamp).

2. PSI + CFL FROM THE ACTUAL CODE PATH: the run's own MLE configuration is
   applied to the CONTROL day-30 state (the state MLE would have acted on) via
   the same functions the model calls -- ``mle_mld_and_buoyancy`` and the
   lat-lon C-grid streamfunction assembly -- not a re-derived lookalike (a
   proxy that cannot reproduce a known answer is the classic trap).  Reported:
   the Psi distribution, its MLD^2 scaling, and the implied per-step tracer
   Courant number  |utr| * dt / V_cell  whose exceedance of ~1 marks a
   numerically unstable explicit centered flux.

Committed per #1492 Phase 0.3: provenance (git SHA, inputs, config) stamped in
the JSON; PNG maps saved for visual inspection.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

_NATIVE_J = slice(0, 331)
_NATIVE_I = slice(1, 361)


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _native2(a):
    a = np.asarray(a, dtype=np.float64)
    if a.shape != (332, 362):
        raise SystemExit(f"expected (332, 362), got {a.shape}")
    return a[_NATIVE_J, _NATIVE_I]


def _native3(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 3 or a.shape[:2] != (332, 362):
        raise SystemExit(f"expected (332, 362, nlev), got {a.shape}")
    return a[_NATIVE_J, _NATIVE_I, :]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mle-snapshot", required=True)
    p.add_argument("--control-snapshot", required=True,
                   help="Matched-day control (identical command minus --mle).")
    p.add_argument("--mle-ce", type=float, default=0.06)
    p.add_argument("--dt-s", type=float, required=True,
                   help="Model timestep [s] from the run manifest -- used for "
                        "the bolus Courant number.")
    p.add_argument("--mesh-mask", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    zm = np.load(a.mle_snapshot)
    zc = np.load(a.control_snapshot)
    lat = _native2(zm["lat_T"])
    lon = _native2(zm["lon_T"])

    sss_m = _native3(zm["S"])[..., 0]
    sss_c = _native3(zc["S"])[..., 0]
    sst_m = _native3(zm["T"])[..., 0]
    sst_c = _native3(zc["T"])[..., 0]
    wet = _native2(zm["land_mask"]) > 0.5      # 1.0 == OCEAN in these snapshots
    dS = np.where(wet, sss_m - sss_c, np.nan)
    dT = np.where(wet, sst_m - sst_c, np.nan)

    # --- 1. freshening geography -------------------------------------------
    flat = np.argsort(np.nan_to_num(dS, nan=0.0), axis=None)
    worst = []
    for idx in flat[:40]:
        j, i = np.unravel_index(idx, dS.shape)
        worst.append({"lat": float(lat[j, i]), "lon": float(lon[j, i]),
                      "dSSS": float(dS[j, i]), "dSST": float(dT[j, i])})
    absS = np.abs(dS[wet & np.isfinite(dS)])
    geo = {
        "n_wet": int(wet.sum()),
        "dSSS_mean": float(np.nanmean(dS)),
        "dSSS_p50_abs": float(np.percentile(absS, 50)),
        "dSSS_p99_abs": float(np.percentile(absS, 99)),
        "frac_dSSS_below_-1": float(np.nanmean(dS[wet] < -1.0)),
        "frac_dSSS_below_-1_tropics": float(np.nanmean(
            dS[wet & (np.abs(lat) < 23.0)] < -1.0)),
        "frac_dSSS_below_-1_arctic": float(np.nanmean(
            dS[wet & (lat >= 60.0)] < -1.0)),
        "worst_40_freshening_cells": worst,
    }
    print(f"[geo] mean dSSS {geo['dSSS_mean']:+.3f}, p99|dSSS| "
          f"{geo['dSSS_p99_abs']:.3f}; frac<-1 psu: global "
          f"{geo['frac_dSSS_below_-1']:.3f}, tropics "
          f"{geo['frac_dSSS_below_-1_tropics']:.3f}, arctic "
          f"{geo['frac_dSSS_below_-1_arctic']:.3f}")

    # --- 2. Psi and bolus Courant from the ACTUAL code path -----------------
    # The control state is what MLE would act on; the run's config is rebuilt
    # with the run's ce.  Uses the same jitted functions the model calls.
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.physics.lateral_mixing.mle import (
        MLEConfig, mle_coefficient, mle_mld_and_buoyancy)

    try:
        import netCDF4 as nc
        ds = nc.Dataset(a.mesh_mask)

        def v(name):
            return np.asarray(ds.variables[name][:], dtype=np.float64).squeeze()
        e1t = v("e1t")[_NATIVE_J, _NATIVE_I]
        e2t = v("e2t")[_NATIVE_J, _NATIVE_I]
        e3t = np.transpose(v("e3t_0")[:, _NATIVE_J, _NATIVE_I], (1, 2, 0))
        tmask3 = np.transpose(v("tmask")[:, _NATIVE_J, _NATIVE_I], (1, 2, 0))
        gdepw_1d = v("gdepw_1d")
        gdept_1d = v("gdept_1d")
        e3t_1d = v("e3t_1d")
        ds.close()
        # NEMO's gdepw_1d has nlev entries (w-point at the TOP of each cell);
        # the shared core wants all nlev+1 faces, so append the bottom face.
        z_faces_1d = np.concatenate([gdepw_1d, [gdepw_1d[-1] + e3t_1d[-1]]])
    except Exception as exc:
        raise SystemExit(f"mesh metrics unreadable: {exc}")

    cfg = MLEConfig(ce=a.mle_ce)
    T3 = _native3(zc["T"])
    S3 = _native3(zc["S"])
    eta = _native2(zc["eta"])
    wet3 = tmask3 > 0.5

    # NEMO rhop: the run's default EOS at ZERO pressure (same construction as
    # combined.py's rho_pot input to the MLE).
    rho_pot = np.asarray(make_eos_fn()(
        jnp.asarray(T3), jnp.asarray(S3), jnp.zeros_like(jnp.asarray(T3))))

    # GEOMETRY APPROXIMATION (declared): dz_live from the NEMO mesh reference
    # partial-cell thickness with the uniform z-star dilation (H+eta)/H, and
    # face metrics from averaged T-cell metrics -- the runtime geom.dx_u is
    # not reconstructable from a snapshot alone.  Adequate for a magnitude/
    # geography verdict, NOT for a bit-exact Psi.
    Hcol = (e3t * wet3).sum(axis=-1)
    colw = Hcol > 0
    dil = np.where(colw, (Hcol + eta * colw) / np.where(colw, Hcol, 1.0), 0.0)
    dz_live = e3t * wet3 * dil[..., None]

    zmld, bm, _in_ml = mle_mld_and_buoyancy(
        jnp.asarray(rho_pot), jnp.asarray(dz_live),
        jnp.asarray(wet3.astype(np.float64)),
        z_faces=jnp.asarray(z_faces_1d),
        z_centers_ref=jnp.asarray(gdept_1d),
        rho_c_mle=cfg.rho_c_mle,
        ref_depth_m=cfg.ref_depth_m,
    )
    zmld = np.asarray(zmld)
    bm = np.asarray(bm)

    # Streamfunction magnitude on interior x-faces, same algebra as
    # mle_latlon_cgrid (rc_f * H_face^2 * e2u * dbm/dx * min(cap, e1u)).
    rc_f = float(mle_coefficient(cfg.ce, cfg.lat_ref_deg))
    H_face = np.minimum(zmld[:, :-1], zmld[:, 1:])
    dbdx = (bm[:, 1:] - bm[:, :-1]) / (0.5 * (e1t[:, 1:] + e1t[:, :-1]))
    e2u = 0.5 * (e2t[:, 1:] + e2t[:, :-1])
    cap = np.minimum(cfg.max_grid_scale_m, 0.5 * (e1t[:, 1:] + e1t[:, :-1]))
    psi = rc_f * H_face**2 * e2u * dbdx * cap          # [m^3/s]

    ok = np.isfinite(psi)
    ap = np.abs(psi[ok])
    # Per-step tracer Courant of the bolus transport through the SURFACE cell
    # of the downwind column (thinnest cell -> worst case): |psi| * mu_max
    # acts over roughly the top model cell; V_top = e1t*e2t*dz_top.
    dz_top = float(gdepw_1d[1])                       # thickness of level 0
    V_top = e1t * e2t * dz_top
    V_face = 0.5 * (V_top[:, 1:] + V_top[:, :-1])
    courant = np.abs(psi) * a.dt_s / V_face
    stats = {
        "rc_f": rc_f, "dt_s": a.dt_s, "dz_top_m": dz_top,
        "psi_p50_m3s": float(np.percentile(ap, 50)),
        "psi_p99_m3s": float(np.percentile(ap, 99)),
        "psi_max_m3s": float(ap.max()),
        "mld_p50_m": float(np.percentile(zmld[wet], 50)),
        "mld_p99_m": float(np.percentile(zmld[wet], 99)),
        "mld_max_m": float(zmld[wet].max()),
        "courant_p99": float(np.percentile(courant[ok], 99)),
        "courant_max": float(np.nanmax(courant[ok])),
        "frac_courant_gt_1": float(np.mean(courant[ok] > 1.0)),
        "frac_courant_gt_0p5": float(np.mean(courant[ok] > 0.5)),
    }
    print(f"[psi] p50 {stats['psi_p50_m3s']:.3e}  p99 {stats['psi_p99_m3s']:.3e}"
          f"  max {stats['psi_max_m3s']:.3e} m3/s;  MLD p50/p99/max "
          f"{stats['mld_p50_m']:.0f}/{stats['mld_p99_m']:.0f}/{stats['mld_max_m']:.0f} m")
    print(f"[cfl] surface-cell bolus Courant p99 {stats['courant_p99']:.3f}  "
          f"max {stats['courant_max']:.3f}  frac>1 {stats['frac_courant_gt_1']:.4f}"
          f"  frac>0.5 {stats['frac_courant_gt_0p5']:.4f}")

    # --- maps ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.4))
    im0 = ax[0].pcolormesh(dS, cmap="RdBu", vmin=-3, vmax=3)
    ax[0].set_title("SSS: MLE - control [psu]")
    plt.colorbar(im0, ax=ax[0], shrink=0.85)
    im1 = ax[1].pcolormesh(dT, cmap="RdBu_r", vmin=-3, vmax=3)
    ax[1].set_title("SST: MLE - control [C]")
    plt.colorbar(im1, ax=ax[1], shrink=0.85)
    lg = np.log10(np.maximum(np.abs(psi), 1.0))
    im2 = ax[2].pcolormesh(np.where(ok, lg, np.nan), cmap="magma")
    ax[2].set_title("log10 |Psi| [m3/s] on control state")
    plt.colorbar(im2, ax=ax[2], shrink=0.85)
    fig.suptitle("MLE freshening geography and streamfunction (native cells, index space)")
    fig.tight_layout()
    fig.savefig(out / "mle_freshening_psi.png", dpi=110)
    print(f"[map] {out / 'mle_freshening_psi.png'}")

    report = {"generated_by": str(Path(__file__).resolve()),
              "git_sha": _git_sha(Path(__file__).resolve().parents[3]),
              "mle_snapshot": a.mle_snapshot,
              "control_snapshot": a.control_snapshot,
              "mle_ce": a.mle_ce,
              "geography": geo, "psi_stats": stats}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[report] {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
