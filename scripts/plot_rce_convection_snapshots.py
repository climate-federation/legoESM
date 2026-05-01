#!/usr/bin/env python
"""Plot per-scheme snapshots from ``run_rce_convection_sweep.py`` output.

Reads ``results/rce_convection_sweep/snapshot_<scheme>.npz`` for every
scheme that has a snapshot present and emits two PNG plots:

* ``timeseries.png`` — domain-mean ``T_atm``, ``CWV``, ``precip``,
  ``max|v|`` vs time, all schemes overlaid (SBM in black as the
  reference, others in colour).
* ``profiles.png`` — final-day vertical profiles of ``T``, ``q_v``,
  ``RH``, and the cached convective heating rate ``dT/dt`` (K/day),
  all schemes overlaid against the moist adiabat from the surface
  parcel.

The figures are written next to the snapshots.

Usage:
    .venv/bin/python scripts/plot_rce_convection_snapshots.py \\
        [results/rce_convection_sweep]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCHEME_ORDER = (
    "sbm",
    "tiedtke",
    "zhang_mcfarlane",
    "emanuel",
    "bechtold",
    "kuo",
    "kain_fritsch",
    "mass_flux",
)
SCHEME_COLOURS = {
    "sbm": "k",
    "tiedtke": "tab:blue",
    "zhang_mcfarlane": "tab:orange",
    "emanuel": "tab:green",
    "bechtold": "tab:red",
    "kuo": "tab:purple",
    "kain_fritsch": "tab:brown",
    "mass_flux": "tab:cyan",
}


# Fields that must match across all loaded snapshots.  If any differ
# we refuse to combine — the snapshots came from different sweep
# invocations and plotting them on the same axes would be misleading.
_CONFIG_STAMPS = ("days", "N", "nlev", "dt", "sst", "T_init", "RH_init",
                  "run_id")


def _scalar(v):
    """Coerce a 0-d numpy array (how np.savez stores scalars) to a
    plain Python value for stable comparison."""
    if hasattr(v, "ndim") and v.ndim == 0:
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode()
    return v


def _load_snapshots(root: Path, *, strict: bool = True) -> dict[str, dict]:
    """Load ``snapshot_<scheme>.npz`` files from ``root``.

    Two-phase validation:

    1. **Completeness**: every snapshot must carry the full set of
       run-config stamps (``days``, ``N``, ``nlev``, ``dt``, ``sst``,
       ``T_init``, ``RH_init``, ``run_id``).  A snapshot missing any
       of these fields is from a pre-stamping version of the sweep
       script (or has been hand-edited) and is therefore rejected
       *before* it can be used as the reference — otherwise an
       unstamped legacy file could become the comparison baseline and
       cause every newer (correctly stamped) snapshot to be silently
       skipped as "the mismatch".

    2. **Consistency**: among the snapshots that pass step 1, every
       stamp must match the first-loaded one.  Snapshots that disagree
       are skipped with a clear warning.

    In ``strict`` mode (the default) any rejection or mismatch is a
    fatal error (rc=2) so a CI pipeline calling this plotter never
    silently mixes snapshots from different runs.  ``--no-strict``
    downgrades to warnings (skipped snapshots still excluded).
    """
    snaps: dict[str, dict] = {}
    reference: dict | None = None
    reference_scheme: str | None = None
    issues: list[str] = []
    for name in SCHEME_ORDER:
        p = root / f"snapshot_{name}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        snap = {k: d[k] for k in d.keys()}
        # Phase 1: completeness — reject any snapshot missing a stamp.
        missing = [k for k in _CONFIG_STAMPS if k not in snap]
        if missing:
            msg = (
                f"snapshot {p.name} missing stamp field(s) "
                f"{missing} — likely from a pre-stamping version of "
                f"the sweep script; rejecting (re-run the sweep)"
            )
            issues.append(msg)
            print(f"  REJECT: {msg}", file=sys.stderr)
            continue
        cfg = {k: _scalar(snap[k]) for k in _CONFIG_STAMPS}
        if reference is None:
            reference = cfg
            reference_scheme = name
            snaps[name] = snap
            continue
        # Phase 2: consistency with the first-loaded reference.
        diffs = {k: (cfg[k], reference[k])
                 for k in _CONFIG_STAMPS
                 if cfg[k] != reference[k]}
        if diffs:
            msg = (
                f"snapshot {p.name} disagrees with reference "
                f"({reference_scheme}) on: " +
                ", ".join(f"{k}={cfg[k]!r} vs {reference[k]!r}"
                          for k in diffs)
            )
            issues.append(msg)
            print(f"  SKIP: {msg}", file=sys.stderr)
            continue
        snaps[name] = snap
    if issues and strict:
        print(
            f"\n  ERROR: {len(issues)} snapshot(s) failed "
            f"completeness/consistency validation.  Refusing to "
            f"combine inconsistent data.  Re-run the sweep so all "
            f"snapshots share a run_id, or pass --no-strict to plot "
            f"anyway (rejected/mismatched snapshots are still excluded).",
            file=sys.stderr,
        )
        sys.exit(2)
    return snaps


def _moist_adiabat(T_sfc: float, p_full: np.ndarray) -> np.ndarray:
    """Reversible saturated moist adiabat from a surface parcel.

    Uses a simple integration of dT/dp = R_d T / (c_p p) corrected
    by the latent-heat / Clausius-Clapeyron term.  Approximation only —
    intended as a visual reference, not a precise sounding.
    """
    R_d, c_pd, L_v, R_v = 287.0, 1004.0, 2.5e6, 461.5
    eps = R_d / R_v
    T = np.empty_like(p_full)
    T[-1] = T_sfc                       # surface (high-p end)
    for k in range(len(p_full) - 2, -1, -1):
        Tk = T[k + 1]
        # Clausius-Clapeyron: e_sat ≈ 611.2 exp(17.67 (T-273)/(T-29.65))
        e_sat = 611.2 * np.exp(17.67 * (Tk - 273.15) / (Tk - 29.65))
        q_sat = eps * e_sat / (p_full[k + 1] - (1 - eps) * e_sat)
        # Pseudo-adiabatic dT/dp formula (Iribarne–Godson).
        num = R_d * Tk + L_v * q_sat
        den = c_pd + L_v ** 2 * q_sat * eps / (R_d * Tk * Tk)
        dTdp = num / (p_full[k + 1] * den)
        dp = p_full[k] - p_full[k + 1]   # negative going up
        T[k] = Tk + dTdp * dp
    return T


def plot_timeseries(snaps: dict, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    metrics = [
        ("ts_T_atm", "Domain-mean T_atm (K)", axes[0, 0]),
        ("ts_cwv", "Column water vapor (kg/m²)", axes[0, 1]),
        ("ts_precip", "Surface precipitation (mm/day)", axes[1, 0]),
        ("ts_max_v", "max |v| (m/s)", axes[1, 1]),
    ]
    for key, label, ax in metrics:
        for name in SCHEME_ORDER:
            if name not in snaps:
                continue
            d = snaps[name]
            day = d["ts_day"]
            y = d[key]
            if len(day) == 0:
                continue
            blowup = bool(d.get("blowup", False))
            ls = "--" if blowup else "-"
            ax.plot(day, y, ls=ls, color=SCHEME_COLOURS[name],
                    label=name, lw=2 if name == "sbm" else 1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Day")
    axes[1, 1].set_xlabel("Day")
    axes[0, 0].legend(loc="best", fontsize=8, ncols=2)
    fig.suptitle("RCE convection-scheme sweep — time series")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_profiles(snaps: dict, out_path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    # Use SBM's sigma + p_s reference for the moist adiabat overlay.
    if "sbm" in snaps:
        sigma = snaps["sbm"]["sigma_full"]
    else:
        sigma = next(iter(snaps.values()))["sigma_full"]
    p_s = 1e5
    p_full = sigma * p_s
    T_sfc_ref = float(snaps[next(iter(snaps))]["sst"])
    T_madiabat = _moist_adiabat(T_sfc_ref, p_full)

    panels = [
        ("T_profile", "T (K)", axes[0]),
        ("qv_profile", "q_v (kg/kg)", axes[1]),
        ("rh_profile", "RH", axes[2]),
        ("dT_conv_profile", "Conv heating dT/dt (K/day)", axes[3]),
    ]
    for key, label, ax in panels:
        for name in SCHEME_ORDER:
            if name not in snaps:
                continue
            d = snaps[name]
            blowup = bool(d.get("blowup", False))
            if blowup:
                continue   # skip blowup vertical profile (NaN-laden)
            y = d[key]
            if key == "dT_conv_profile":
                y = y * 86400.0    # K/s → K/day
            ax.plot(y, sigma, color=SCHEME_COLOURS[name], label=name,
                    lw=2 if name == "sbm" else 1.2)
        if key == "T_profile":
            ax.plot(T_madiabat, sigma, "k:", lw=1.0, alpha=0.7,
                    label="moist adiabat (SST)")
        if key == "rh_profile":
            ax.axvline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("σ (=p/p_s)")
    axes[0].legend(loc="best", fontsize=7, ncols=1)
    fig.suptitle(f"RCE convection-scheme sweep — final-day vertical "
                 f"profiles (SST={T_sfc_ref:.1f} K)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_conv_tendencies(snaps: dict, out_path: Path):
    """Per-scheme conv tendency profiles, each on its own axis (free
    scale).  Highlights what each scheme is *actually* contributing
    even when the M_b cap suppresses cross-scheme magnitudes."""
    n = len(snaps)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, [k for k in SCHEME_ORDER if k in snaps]):
        d = snaps[name]
        sigma = d["sigma_full"]
        if bool(d.get("blowup", False)):
            ax.text(0.5, 0.5, "BLOWUP", transform=ax.transAxes,
                    ha="center", va="center", color="red",
                    fontsize=14, fontweight="bold")
        else:
            dT = d["dT_conv_profile"] * 86400.0   # K/day
            dq = d["dqv_conv_profile"] * 86400.0e3  # g/kg/day
            ax.plot(dT, sigma, color=SCHEME_COLOURS[name], lw=1.5,
                    label="dT/dt (K/day)")
            ax2 = ax.twiny()
            ax2.plot(dq, sigma, color=SCHEME_COLOURS[name], lw=1.0,
                     ls="--", label="dq_v/dt (g/kg/day)")
            ax2.tick_params(axis="x", labelsize=7,
                            colors=SCHEME_COLOURS[name])
        ax.set_title(name, fontsize=10)
        ax.axvline(0, color="gray", lw=0.6, alpha=0.5)
        ax.grid(True, alpha=0.3)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("σ")
    fig.suptitle("Convection tendency profiles (last refresh, "
                 "domain-mean) — solid: dT/dt K/day,  dashed: "
                 "dq_v/dt g/kg/day")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    # CLI: optional positional <root> dir, optional --no-strict flag.
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    strict = "--no-strict" not in sys.argv[1:]
    root = Path(args[0] if args else "results/rce_convection_sweep")
    if not root.exists():
        print(f"no snapshots in {root}; run the sweep first", file=sys.stderr)
        sys.exit(1)
    snaps = _load_snapshots(root, strict=strict)
    if not snaps:
        print(f"no snapshot_*.npz files in {root}", file=sys.stderr)
        sys.exit(1)
    print(f"  loaded {len(snaps)} schemes from {root}")
    plot_timeseries(snaps, root / "timeseries.png")
    plot_profiles(snaps, root / "profiles.png")
    plot_conv_tendencies(snaps, root / "conv_tendencies.png")


if __name__ == "__main__":
    main()
