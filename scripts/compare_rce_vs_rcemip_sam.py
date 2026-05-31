"""Compare a legoESM plane-CRM RCE run's TIME-AVERAGED profiles + bulk
magnitudes against published RCEMIP (Wing et al. 2020, GMD) 300 K reference
values — the apples-to-apples faithfulness check the CRM-vs-gSAM task asks for
("magnitudes of anomalies and profiles, NOT snapshots").

Reads the per-snapshot vertical profiles written by ``run_rcemip_plane.py
--snapshot-every N`` (``<output>/snapshots/profile_step_*.npz``), TIME-AVERAGES
them over the convective window (steps >= --window-start, default skips the
laminar spin-up), and emits:
  - a fixed-width PASS/CHECK table (bulk magnitudes vs RCEMIP 300 K ranges),
  - a 6-panel profile figure vs the RCEMIP reference bands.

The reference numbers are the well-established RCEMIP 300 K CRM signatures
(Wing et al. 2020 "RCEMIP", GMD 13, 793-813; Reed et al. 2021): domain-mean
column water vapour, near-surface T/qv, cold-point tropopause, mid-troposphere
convective w-variance, and the bimodal (boundary-layer + anvil) cloud fraction.
They are inter-model RANGES, not a single digitized SAM profile (the RCEMIP
archive is not available offline here) — so this validates faithful MAGNITUDES
and profile SHAPES, which is exactly the stated comparison protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from legoesm import constants as C
from legoesm.thermo import saturation_mixing_ratio


# --- RCEMIP 300 K reference ranges (Wing et al. 2020, GMD) ------------------
# (low, high, units, note). Generous to reflect inter-model/domain spread.
BULK_REF = {
    "CWV (PW)":            (35.0, 55.0, "mm",     "RCEMIP 300 K precipitable water"),
    "T near-surface":      (295.0, 298.5, "K",    "lowest-level air T over 300 K SST"),
    "qv near-surface":     (14.0, 18.5, "g/kg",   "lowest-level water-vapour mixing ratio"),
    "T cold-point":        (188.0, 198.0, "K",    "tropopause cold-point temperature"),
    "z cold-point":        (13.5, 17.5, "km",     "tropopause height"),
    "w_RMS mid-trop peak": (0.25, 0.85, "m/s",    "convective vertical-velocity RMS, mid-troposphere"),
    "cloud frac peak":     (0.08, 0.30, "-",      "bimodal BL+anvil peak domain cloud fraction"),
}


def _load_window(snap_dir: Path, window_start: int):
    files = sorted(snap_dir.glob("profile_step_*.npz"))
    if not files:
        raise SystemExit(f"no profile_step_*.npz in {snap_dir}")
    kept = []
    for f in files:
        d = np.load(f)
        if int(d["step"]) >= window_start:
            kept.append(d)
    if not kept:
        raise SystemExit(
            f"no snapshots with step >= {window_start} (have "
            f"{[int(np.load(f)['step']) for f in files]})")
    return kept


def _time_mean(snaps, key):
    return np.mean(np.stack([s[key] for s in snaps], axis=0), axis=0)


def compute_metrics(snaps) -> dict:
    """Time-average the per-snapshot profiles and derive RCEMIP bulk magnitudes
    + profile-shape signatures. PURE (no I/O) so it is unit-testable; ``snaps``
    is a list of mapping-like objects exposing the keys written by
    ``run_rcemip_plane._emit_profile_npz``. Everything is sorted ascending in z
    (surface = index 0) so the logic is robust to the run's internal
    surface-first/last level convention.
    """
    order = np.argsort(np.asarray(snaps[0]["z"]))
    z = np.asarray(snaps[0]["z"])[order]   # full-level height [m], ascending
    dz = np.abs(np.diff(np.sort(np.asarray(snaps[0]["z_half"]))))  # thickness, len nlev
    T = _time_mean(snaps, "T_mean")[order]        # [K]
    theta = _time_mean(snaps, "theta_mean")[order]
    qv = _time_mean(snaps, "qv_mean")[order]      # [kg/kg]
    qc = _time_mean(snaps, "qc_mean")[order]
    w_RMS = _time_mean(snaps, "w_RMS")[order]     # [m/s]
    cloud = _time_mean(snaps, "cloud_fraction")[order]
    rho = _time_mean(snaps, "rho_mean")[order]    # [kg/m3]
    qprecip = _time_mean(snaps, "q_precip_mean")[order]

    # Pressure from hydrostatic Exner reference (T/theta = pi_0), then RH via library qsat.
    p = float(C.p_ref) * (T / theta) ** (1.0 / C.kappa)
    qsat = np.asarray(saturation_mixing_ratio(T, p))
    RH = 100.0 * np.clip(qv / np.maximum(qsat, 1e-12), 0.0, 2.0)

    # Bulk magnitudes (ascending z: index 0 = surface).
    # CWV = integral of specific humidity * total density (qv is a MIXING ratio,
    # so q = qv/(1+qv); using qv*rho_total directly overestimates by ~2%).
    cwv = float(np.sum(rho * (qv / (1.0 + qv)) * dz))   # kg/m2 = mm
    k_cp = int(np.argmin(T))
    kml = int(np.argmin(np.abs(z - 5000.0)))   # ~5 km mid-trop index
    got = {
        "CWV (PW)": cwv, "T near-surface": float(T[0]),
        "qv near-surface": float(qv[0] * 1e3),
        "T cold-point": float(T[k_cp]), "z cold-point": float(z[k_cp] / 1e3),
        "w_RMS mid-trop peak": float(np.max(w_RMS)),
        "cloud frac peak": float(np.max(cloud)),
    }
    shapes = {
        "T decreasing surface->cold point": bool(np.all(np.diff(T[:k_cp + 1]) <= 0.5)),
        "qv decreases with height (exp-like)": bool(qv[0] > qv[kml] > qv[k_cp]),
        "w_RMS mid-trop-peaked (not surface)": bool(3 < int(np.argmax(w_RMS)) < len(z) - 4),
    }
    profiles = dict(z=z, T=T, qv=qv, qc=qc, w_RMS=w_RMS, cloud=cloud, RH=RH,
                    qprecip=qprecip)
    return dict(got=got, shapes=shapes, profiles=profiles)


def evaluate(got) -> tuple:
    """Compare bulk magnitudes vs ``BULK_REF``. PURE. Returns
    ``(rows, n_pass, n_edge)`` where each row is
    ``(name, value, lo, hi, unit, ok, where, note)`` and ``where`` is one of
    BELOW / lo-edge / mid / hi-edge / ABOVE; ``n_edge`` counts passes that sit
    at a range edge (so an edge-pass is not oversold as a tight match).
    """
    rows = []
    n_pass = n_edge = 0
    for name, (lo, hi, unit, note) in BULK_REF.items():
        v = got[name]
        ok = lo <= v <= hi
        n_pass += ok
        pos = (v - lo) / (hi - lo) if hi > lo else 0.5
        where = ("BELOW" if v < lo else "ABOVE" if v > hi
                 else "lo-edge" if pos <= 0.15 else "hi-edge" if pos >= 0.85
                 else "mid")
        if ok and where in ("lo-edge", "hi-edge"):
            n_edge += 1
        rows.append((name, v, lo, hi, unit, ok, where, note))
    return rows, n_pass, n_edge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path,
                    help="RCE run output dir (contains snapshots/)")
    ap.add_argument("--window-start", type=int, default=4000,
                    help="first step (inclusive) of the convective averaging "
                         "window; skips the laminar spin-up")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    snap_dir = args.output / "snapshots"
    snaps = _load_window(snap_dir, args.window_start)
    steps = [int(s["step"]) for s in snaps]
    print(f"Averaging {len(snaps)} snapshots over steps {steps}")

    m = compute_metrics(snaps)
    got, shapes, prof = m["got"], m["shapes"], m["profiles"]
    rows, n_pass, n_edge = evaluate(got)

    print("\n  quantity              legoESM     RCEMIP-300K ref     status where     note")
    print("  " + "-" * 92)
    for name, v, lo, hi, unit, ok, where, note in rows:
        print(f"  {name:21s} {v:8.2f}  [{lo:7.2f},{hi:7.2f}] {unit:5s} "
              f"{'PASS ' if ok else 'CHECK'} {where:7s} {note}")
    print("  " + "-" * 92)
    print(f"  {n_pass}/{len(BULK_REF)} within RCEMIP 300 K model-spread range "
          f"({n_edge} of those sit at a range EDGE — read as 'consistent with the "
          f"RCEMIP spread', not a tight match)")

    print("\n  profile-shape signatures:")
    for k, v in shapes.items():
        print(f"    {'OK  ' if v else 'CHK '} {k}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z = prof["z"]; zk = z / 1e3
        fig, ax = plt.subplots(2, 3, figsize=(13, 8))
        fig.suptitle(f"legoESM plane-CRM RCE (dx=3 km) time-mean over steps "
                     f"{steps[0]}-{steps[-1]} vs RCEMIP 300 K", fontsize=11)
        ax[0, 0].plot(prof["T"], zk, "b-"); ax[0, 0].axvspan(188, 198, color="grey", alpha=0.15)
        ax[0, 0].set_xlabel("T [K]"); ax[0, 0].set_ylabel("z [km]"); ax[0, 0].set_title("Temperature")
        ax[0, 1].plot(prof["qv"] * 1e3, zk, "g-")
        ax[0, 1].set_xlabel("q_v [g/kg]"); ax[0, 1].set_title("Water vapour")
        ax[0, 2].plot(prof["RH"], zk, "c-"); ax[0, 2].axvspan(40, 80, color="grey", alpha=0.15)
        ax[0, 2].set_xlabel("RH [%]"); ax[0, 2].set_title("Relative humidity")
        ax[1, 0].plot(prof["w_RMS"], zk, "r-"); ax[1, 0].axvspan(0.25, 0.85, color="grey", alpha=0.15)
        ax[1, 0].set_xlabel("w_RMS [m/s]"); ax[1, 0].set_ylabel("z [km]")
        ax[1, 0].set_title("Vertical-velocity RMS")
        ax[1, 1].plot(prof["cloud"], zk, "m-"); ax[1, 1].axvspan(0.08, 0.30, color="grey", alpha=0.15)
        ax[1, 1].set_xlabel("cloud fraction"); ax[1, 1].set_title("Cloud fraction")
        ax[1, 2].plot(prof["qc"] * 1e3, zk, "k-", label="q_cloud")
        ax[1, 2].plot(prof["qprecip"] * 1e3, zk, "b--", label="q_precip")
        ax[1, 2].set_xlabel("condensate [g/kg]"); ax[1, 2].set_title("Condensate"); ax[1, 2].legend()
        for a in ax.ravel():
            a.set_ylim(0, min(20, zk.max())); a.grid(alpha=0.3)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = args.output / "rce_vs_rcemip_sam.png"
        fig.savefig(out, dpi=110)
        print(f"\n  figure -> {out}")


if __name__ == "__main__":
    main()
