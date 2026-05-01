#!/usr/bin/env python
"""End-state cross-scheme comparison from a 1-year RCE sweep.

For each scheme, find the *latest non-blowup* 3D snapshot and plot:

* lat-mean T(σ) profile (vs moist adiabat from SST)
* lat-mean q_v(σ)
* lat-mean RH(σ)
* zonal-mean precipitation vs latitude

All on a single 2x2 figure for direct cross-scheme comparison.  Also
verifies physics consistency (sign, range, NaN) at the picked snapshot.

Usage:
    .venv/bin/python scripts/plot_rce_endstate_comparison.py \\
        [results/rce_convection_sweep] [--no-strict]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCHEME_ORDER = (
    "sbm", "tiedtke", "zhang_mcfarlane", "emanuel",
    "bechtold", "kuo", "kain_fritsch", "mass_flux",
)
SCHEME_COLOURS = {
    "sbm": "k", "tiedtke": "tab:blue", "zhang_mcfarlane": "tab:orange",
    "emanuel": "tab:green", "bechtold": "tab:red", "kuo": "tab:purple",
    "kain_fritsch": "tab:brown", "mass_flux": "tab:cyan",
}
_CONFIG_STAMPS = ("days", "N", "nlev", "dt", "sst", "T_init", "RH_init",
                  "run_id", "snap3d_days")


def _scalar(v):
    if hasattr(v, "ndim") and v.ndim == 0:
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode()
    return v


def _moist_adiabat(T_sfc: float, p_full: np.ndarray) -> np.ndarray:
    R_d, c_pd, L_v, R_v = 287.0, 1004.0, 2.5e6, 461.5
    eps = R_d / R_v
    T = np.empty_like(p_full)
    T[-1] = T_sfc
    for k in range(len(p_full) - 2, -1, -1):
        Tk = T[k + 1]
        es = 611.2 * np.exp(17.67 * (Tk - 273.15) / (Tk - 29.65))
        q_sat = eps * es / (p_full[k + 1] - (1 - eps) * es)
        num = R_d * Tk + L_v * q_sat
        den = c_pd + L_v ** 2 * q_sat * eps / (R_d * Tk * Tk)
        dTdp = num / (p_full[k + 1] * den)
        T[k] = Tk + dTdp * (p_full[k] - p_full[k + 1])
    return T


def _last_good_snapshot(snap: dict) -> tuple[int, str]:
    """Return (time_idx, label) of the last snapshot with finite T.

    Falls back to the last snapshot if everything is NaN."""
    T_stack = snap["T"]   # (n_snaps, n_lat, n_lon, nlev)
    n = T_stack.shape[0]
    for i in range(n - 1, -1, -1):
        if np.all(np.isfinite(T_stack[i])):
            day = float(snap["snap_day"][i])
            return i, f"day {int(day)}"
    return n - 1, "BLOWUP"


def _saturation_mixing_ratio(T_K, p_Pa):
    es = 611.2 * np.exp(17.67 * (T_K - 273.15) / (T_K - 29.65))
    eps = 0.622
    return eps * es / np.clip(p_Pa - (1 - eps) * es, 1.0, None)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    strict = "--no-strict" not in sys.argv[1:]
    root = Path(args[0] if args else "results/rce_convection_sweep")
    snaps: dict[str, dict] = {}
    reference: dict | None = None
    issues: list[str] = []
    for name in SCHEME_ORDER:
        p = root / f"snapshot3d_{name}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        snap = {k: d[k] for k in d.keys()}
        missing = [k for k in _CONFIG_STAMPS if k not in snap]
        if missing:
            issues.append(f"{p.name} missing {missing}")
            continue
        cfg = {k: _scalar(snap[k]) for k in _CONFIG_STAMPS}
        if reference is None:
            reference = cfg
        else:
            diffs = {k: (cfg[k], reference[k])
                     for k in _CONFIG_STAMPS
                     if cfg[k] != reference[k]}
            if diffs:
                issues.append(f"{p.name} stamps disagree: {list(diffs)}")
                continue
        snaps[name] = snap
    if issues and strict:
        print(f"  ERROR: {len(issues)} issue(s):", file=sys.stderr)
        for m in issues:
            print(f"    - {m}", file=sys.stderr)
        sys.exit(2)
    if not snaps:
        print(f"no snapshot3d_*.npz in {root}", file=sys.stderr)
        sys.exit(1)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    p_full = next(iter(snaps.values()))["sigma_full"] * 1e5
    T_sfc_ref = float(_scalar(next(iter(snaps.values()))["sst"]))
    T_madiabat = _moist_adiabat(T_sfc_ref, p_full)
    sigma = next(iter(snaps.values()))["sigma_full"]

    physics_violations = []
    for name in SCHEME_ORDER:
        if name not in snaps:
            continue
        snap = snaps[name]
        idx, tag = _last_good_snapshot(snap)
        Tz = np.nanmean(snap["T"][idx], axis=(0, 1))      # (nlev,)
        qvz = np.nanmean(snap["q_v"][idx], axis=(0, 1))
        qsat = _saturation_mixing_ratio(Tz, p_full)
        rh = qvz / np.clip(qsat, 1e-12, None)
        precip_lat = np.nanmean(snap["precip"][idx], axis=1) * 86400.0
        lat_deg = np.rad2deg(snap["lat"][:, 0])

        # Physics consistency
        if Tz.min() < 180 or Tz.max() > 340:
            physics_violations.append(
                f"{name} {tag}: T zonal-mean profile out of [180,340] K"
            )
        if qvz.min() < -1e-12 or qvz.max() > 0.05:
            physics_violations.append(
                f"{name} {tag}: q_v zonal-mean out of [0,0.05] kg/kg"
            )
        if rh.min() < -1e-6:
            physics_violations.append(f"{name} {tag}: RH negative")
        if np.any(precip_lat < -1e-12):
            physics_violations.append(f"{name} {tag}: precip negative")

        col = SCHEME_COLOURS[name]
        lw = 2.0 if name == "sbm" else 1.2
        ls = ":" if "BLOWUP" in tag else "-"
        label = f"{name} ({tag})"
        axes[0, 0].plot(Tz, sigma, color=col, lw=lw, ls=ls, label=label)
        axes[0, 1].plot(qvz * 1e3, sigma, color=col, lw=lw, ls=ls)
        axes[1, 0].plot(rh, sigma, color=col, lw=lw, ls=ls)
        axes[1, 1].plot(lat_deg, precip_lat, color=col, lw=lw, ls=ls)

    # Overlay moist adiabat
    axes[0, 0].plot(T_madiabat, sigma, "k:", alpha=0.4, lw=1.0,
                    label=f"moist adiabat ({T_sfc_ref:.0f} K)")
    axes[0, 0].set_xlabel("T (K)")
    axes[0, 0].set_ylabel("σ")
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(loc="best", fontsize=7, ncols=1)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title("Domain-mean T profile")

    axes[0, 1].set_xlabel("q_v (g/kg)")
    axes[0, 1].set_ylabel("σ")
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title("Domain-mean q_v profile")

    axes[1, 0].axvline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
    axes[1, 0].set_xlabel("RH")
    axes[1, 0].set_ylabel("σ")
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("Domain-mean RH profile")

    axes[1, 1].axhline(0, color="gray", lw=0.6, alpha=0.5)
    axes[1, 1].set_xlabel("lat (°N)")
    axes[1, 1].set_ylabel("Precip (mm/day)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title("Zonal-mean precipitation")

    fig.suptitle(f"RCE end-state comparison "
                 f"(SST={T_sfc_ref:.0f} K, run_id={reference['run_id']})")
    fig.tight_layout()
    out = root / "endstate_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    if physics_violations:
        print(f"\n  PHYSICS CHECK: {len(physics_violations)} violation(s):")
        for m in physics_violations:
            print(f"    - {m}")
        if strict:
            sys.exit(1)
    else:
        print("  PHYSICS CHECK: end-state profiles within plausible bounds.")


if __name__ == "__main__":
    main()
