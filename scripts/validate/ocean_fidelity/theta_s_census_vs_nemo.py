"""Volumetric theta-S census, ours vs NEMO, on identical eORCA1 mesh volumes.

WHY THIS AND NOT ANOTHER SECTION.  Every fidelity figure so far is a map or a
section, and both are vulnerable to cancellation: a global zonal mean cancels
opposite-signed basin anomalies, and a section only sees the water it cuts
through.  A volumetric census is immune to both -- it asks how much OCEAN
VOLUME sits at each temperature and salinity, with no geography at all -- and
at 30 days from rest it is the cleanest available probe of SPURIOUS DIAPYCNAL
MIXING, because a numerically diffusive model collapses volume out of the
distribution's tails toward its centre whether or not any surface flux differs.

PRE-REGISTERED FALSIFIER, with the sign stated before the number is read.
If our model mixes more than NEMO over these 30 days, volume must move from
the EXTREMES toward the MIDDLE: the coldest/warmest and freshest/saltiest bins
must hold LESS volume than NEMO's and the central bins MORE, and the tail
deficit must roughly balance the central surplus.  A difference that is large
but UNSTRUCTURED -- no consistent tail-to-centre transfer -- is not spurious
mixing and must not be reported as such.  A difference at or below the census's
own discretisation noise is a null result, which at day 30 is the expected and
desirable outcome.

CONTROLLED COMPARISON.  Both models are binned with the SAME reference cell
volumes (e1t*e2t*e3t_0*tmask on the native eORCA1 frame), so a difference in
the census is a difference in the T/S DISTRIBUTION and nothing else.  The
z-star dilation (H+eta)/H is deliberately NOT applied: it is order 3e-4 here,
it differs between the two models, and including it would let a sea-level
difference masquerade as a water-mass difference.  That is a choice, so it is
stated rather than defaulted.

CONVENTIONS.  NEMO's 3-D fields in this file are named ``to`` and ``so`` and,
despite CF standard_names saying potential/practical, hold CONSERVATIVE
temperature and ABSOLUTE salinity under ln_teos10 -- the same pair our model
carries, so no conversion is applied.  See the loader in
plot_sections_vs_nemo, which names the variables it used in its output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2] / "plot"))
from global_tracer_content import (  # noqa: E402
    _native as native_frame, load_mesh_metrics)
from plot_sections_vs_nemo import load_nemo_3d  # noqa: E402


def census(T, S, dV, t_edges, s_edges):
    """Volume [m3] binned by (temperature, salinity), plus the volume that
    fell OUTSIDE the requested bin ranges.

    The out-of-range volume is returned rather than silently clipped: a census
    whose bins miss real water understates the tails, which is exactly the
    quantity the falsifier reads.
    """
    wet = np.isfinite(T) & np.isfinite(S) & (dV > 0.0)
    t, s, w = T[wet], S[wet], dV[wet]
    inside = ((t >= t_edges[0]) & (t <= t_edges[-1])
              & (s >= s_edges[0]) & (s <= s_edges[-1]))
    H, _, _ = np.histogram2d(t[inside], s[inside], bins=[t_edges, s_edges],
                             weights=w[inside])
    return H, float(w[~inside].sum()), float(w.sum())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, help="tripole day-30 snapshot")
    p.add_argument("--mesh-mask", required=True)
    p.add_argument("--nemo-gridt", required=True)
    p.add_argument("--nemo-time-idx", type=int, required=True,
                   help="0-based record; the window ENDING at snapshot day D "
                        "is D/5-1. REQUIRED, never defaulted: inheriting a "
                        "default of -1 once scored a day-30 state against the "
                        "day-90 mean.")
    p.add_argument("--label", default="tripole")
    p.add_argument("--t-range", default="-2.5,32.0")
    p.add_argument("--s-range", default="30.0,38.0")
    p.add_argument("--n-bins", type=int, default=140)
    p.add_argument("--tail-pct", type=float, default=5.0,
                   help="percentile defining the distribution TAILS for the "
                        "pre-registered tail-to-centre test")
    p.add_argument("--nemo-self-ref-idx", type=int, default=None,
                   help="second NEMO record used as a SCALE CONTROL. Without "
                        "it, a tail deficit has no yardstick and 'we mix more' "
                        "is unquantified. NEMO's own evolution between this "
                        "record and --nemo-time-idx is a known-real change of "
                        "the same census, in the same units, so our "
                        "ours-minus-NEMO difference can be reported as a "
                        "FRACTION of it instead of as a bare volume.")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0, t1 = (float(v) for v in a.t_range.split(","))
    s0, s1 = (float(v) for v in a.s_range.split(","))
    t_edges = np.linspace(t0, t1, a.n_bins + 1)
    s_edges = np.linspace(s0, s1, a.n_bins + 1)

    e1t, e2t, e3t, tmask = load_mesh_metrics(a.mesh_mask)
    dV = e1t[None] * e2t[None] * e3t * tmask
    print(f"[mesh] native frame {dV.shape}, total wet volume "
          f"{dV.sum():.6e} m3")

    z = np.load(a.snapshot)
    Tm, Sm = native_frame(z["T"]), native_frame(z["S"])
    if Tm.shape != dV.shape:
        raise SystemExit(f"FATAL: snapshot native {Tm.shape} != mesh {dV.shape}")

    nem = load_nemo_3d(a.nemo_gridt, a.nemo_time_idx)
    Tn, Sn = nem["T3d"], nem["S3d"]
    if Tn.shape != dV.shape:
        raise SystemExit(
            f"FATAL: NEMO 3-D {Tn.shape} != mesh native frame {dV.shape}. "
            "The census weights both models with the SAME volumes, so a shape "
            "mismatch means they are not on the same cells; refusing to guess "
            "a window.")

    # A wet cell for the census must be wet for BOTH models, or the two
    # histograms integrate different oceans and their difference is a domain
    # difference wearing a water-mass costume.
    both = np.isfinite(Tm) & np.isfinite(Sm) & np.isfinite(Tn) & np.isfinite(Sn)
    dVc = np.where(both, dV, 0.0)
    print(f"[mask] cells wet in both: {int(both.sum())}; shared volume "
          f"{dVc.sum():.6e} m3 ({100.0 * dVc.sum() / dV.sum():.2f}% of mesh)")

    Hm, out_m, tot_m = census(Tm, Sm, dVc, t_edges, s_edges)
    Hn, out_n, tot_n = census(Tn, Sn, dVc, t_edges, s_edges)
    print(f"[census] {a.label}: {100.0 * out_m / tot_m:.4f}% of volume "
          f"outside the bins; NEMO: {100.0 * out_n / tot_n:.4f}%")
    if max(out_m / tot_m, out_n / tot_n) > 0.01:
        print("[census] WARNING: >1% of volume falls outside the bin ranges; "
              "the tails are understated and the falsifier is weakened")

    # PRE-REGISTERED TEST. Tails and centre are defined on NEMO's OWN
    # distribution so the partition does not move with our answer.
    tmar_n, smar_n = Hn.sum(axis=1), Hn.sum(axis=0)
    tmar_m, smar_m = Hm.sum(axis=1), Hm.sum(axis=0)

    def tail_centre(mar_n, mar_m, edges):
        c = 0.5 * (edges[:-1] + edges[1:])
        cum = np.cumsum(mar_n) / max(mar_n.sum(), 1e-30)
        lo = c <= np.interp(a.tail_pct / 100.0, cum, c)
        hi = c >= np.interp(1.0 - a.tail_pct / 100.0, cum, c)
        tails = lo | hi
        return (float(mar_m[tails].sum() - mar_n[tails].sum()),
                float(mar_m[~tails].sum() - mar_n[~tails].sum()))

    dT_tail, dT_cent = tail_centre(tmar_n, tmar_m, t_edges)
    dS_tail, dS_cent = tail_centre(smar_n, smar_m, s_edges)
    verdict = {}
    for nm, (dt, dc) in (("theta", (dT_tail, dT_cent)), ("salt", (dS_tail, dS_cent))):
        mixing_like = (dt < 0.0) and (dc > 0.0)
        rel = abs(dt) / max(tot_n, 1e-30)
        verdict[nm] = {"tail_volume_diff_m3": dt, "centre_volume_diff_m3": dc,
                       "tail_deficit_frac_of_ocean": rel,
                       "mixing_signature": bool(mixing_like)}
        print(f"[{nm}] ours-minus-NEMO volume: tails {dt:+.4e} m3, centre "
              f"{dc:+.4e} m3 -> tail deficit {100.0 * rel:.5f}% of ocean; "
              f"spurious-mixing signature (tails DOWN and centre UP): "
              f"{mixing_like}")

    # SCALE CONTROL. A tail deficit of "0.011% of the ocean" is unreadable on
    # its own. NEMO's own change over one output window is a known-real motion
    # of the SAME census, so expressing our difference as a fraction of it says
    # whether we are looking at a model discrepancy or at nothing.
    if a.nemo_self_ref_idx is not None:
        ref = load_nemo_3d(a.nemo_gridt, a.nemo_self_ref_idx)
        Hr, _, _ = census(ref["T3d"], ref["S3d"], dVc, t_edges, s_edges)
        rt, rs = Hr.sum(axis=1), Hr.sum(axis=0)
        rT_tail, rT_cent = tail_centre(tmar_n, rt, t_edges)
        rS_tail, rS_cent = tail_centre(smar_n, rs, s_edges)
        for nm, ours, theirs in (("theta", dT_tail, rT_tail),
                                 ("salt", dS_tail, rS_tail)):
            ratio = abs(ours) / max(abs(theirs), 1e-30)
            verdict[nm]["nemo_self_tail_diff_m3"] = theirs
            verdict[nm]["ours_over_nemo_self"] = ratio
            print(f"[{nm}] SCALE: NEMO record {a.nemo_self_ref_idx} vs "
                  f"{a.nemo_time_idx} moves {theirs:+.4e} m3 of tail volume; "
                  f"our ours-minus-NEMO tail difference is {ratio:.2f}x that")
        rep_extra = {"nemo_self_ref_idx": a.nemo_self_ref_idx,
                     "scale_control": "tail-volume change across NEMO's own "
                                      "two records, same bins and volumes"}
    else:
        rep_extra = {"nemo_self_ref_idx": None,
                     "scale_control": "NOT RUN -- the tail deficit below has "
                                      "no yardstick and must not be called "
                                      "large or small"}

    rep = {**rep_extra, "snapshot": a.snapshot, "nemo_gridt": a.nemo_gridt,
           "nemo_time_idx": a.nemo_time_idx, "label": a.label,
           "t_range": [t0, t1], "s_range": [s0, s1], "n_bins": a.n_bins,
           "tail_pct": a.tail_pct,
           "shared_volume_m3": float(dVc.sum()),
           "outside_bins_frac": {"ours": out_m / tot_m, "nemo": out_n / tot_n},
           "verdict": verdict,
           "note": "reference volumes (no z-star dilation) applied identically "
                   "to both models; NEMO to/so are CT/SA under ln_teos10"}
    (out / "theta_s_census.json").write_text(json.dumps(rep, indent=2))
    print(f"[report] {out / 'theta_s_census.json'}")

    _plot(out, a.label, t_edges, s_edges, Hm, Hn, tmar_m, tmar_n,
          smar_m, smar_n)
    return 0


def _plot(out, label, t_edges, s_edges, Hm, Hn, tmar_m, tmar_n, smar_m, smar_n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tc = 0.5 * (t_edges[:-1] + t_edges[1:])
    sc = 0.5 * (s_edges[:-1] + s_edges[1:])
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    with np.errstate(divide="ignore"):
        lm = np.log10(np.where(Hm > 0, Hm, np.nan))
        ln_ = np.log10(np.where(Hn > 0, Hn, np.nan))
    vmax = np.nanmax([np.nanmax(lm), np.nanmax(ln_)])
    vmin = vmax - 6.0
    for axi, dat, ttl in ((ax[0], lm, f"{label}"), (ax[1], ln_, "NEMO")):
        im = axi.pcolormesh(sc, tc, dat, vmin=vmin, vmax=vmax, cmap="viridis",
                            shading="auto")
        axi.set_title(f"{ttl}  log10 volume [m3]")
        axi.set_xlabel("salinity"); axi.set_ylabel("temperature [degC]")
        plt.colorbar(im, ax=axi, shrink=0.85)
    d = np.where((Hm > 0) | (Hn > 0), Hm - Hn, np.nan)
    dmax = np.nanpercentile(np.abs(d), 99) or 1.0
    im = ax[2].pcolormesh(sc, tc, d, vmin=-dmax, vmax=dmax, cmap="RdBu_r",
                          shading="auto")
    ax[2].set_title(f"{label} - NEMO  volume [m3]")
    ax[2].set_xlabel("salinity"); ax[2].set_ylabel("temperature [degC]")
    plt.colorbar(im, ax=ax[2], shrink=0.85)
    fig.suptitle("Volumetric theta-S census on identical eORCA1 cell volumes "
                 "(red = we hold more water at that T/S than NEMO)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "theta_s_census.png", dpi=110)
    plt.close(fig)
    print(f"[fig] {out / 'theta_s_census.png'}")

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    ax[0].plot(tc, tmar_m, label=label, lw=1.4)
    ax[0].plot(tc, tmar_n, label="NEMO", lw=1.6, color="k", ls="--")
    ax[0].set_yscale("log"); ax[0].set_xlabel("temperature [degC]")
    ax[0].set_ylabel("volume [m3]"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(sc, smar_m, label=label, lw=1.4)
    ax[1].plot(sc, smar_n, label="NEMO", lw=1.6, color="k", ls="--")
    ax[1].set_yscale("log"); ax[1].set_xlabel("salinity")
    ax[1].set_ylabel("volume [m3]"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle("Volumetric census marginals — tails are where spurious "
                 "mixing shows first", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "theta_s_marginals.png", dpi=110)
    plt.close(fig)
    print(f"[fig] {out / 'theta_s_marginals.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
