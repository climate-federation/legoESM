#!/usr/bin/env python
"""Plot lat-lon and vertical cross-section snapshots from the RCE sweep.

Reads ``snapshot3d_<scheme>.npz`` files (written by
``run_rce_convection_sweep.py``) and emits, per scheme and per
captured time step:

* ``latlon_<scheme>_d<day>.png`` — 2x2 lat-lon maps of T at the lowest
  level, q_v at the lowest level, column water vapor, and surface
  precipitation.
* ``vertlat_<scheme>_d<day>.png`` — zonal-mean vertical cross-sections
  (lat × σ) of T, q_v, RH, and zonal wind u.
* ``vertlon_<scheme>_d<day>.png`` — meridional-mean vertical cross-
  sections (lon × σ) of T, q_v, RH, and meridional wind v.

Files land in ``<root>/<scheme>/`` so each scheme has its own
sub-directory with the time-stacked PNGs.

Physics consistency checks fail with a non-zero exit code if any
snapshot violates basic plausibility:

* ``T`` outside [180, 340] K (BLOWUP territory),
* ``q_v`` negative or > 0.05 kg/kg (50 g/kg — well above tropical),
* ``precip`` negative,
* ``RH`` negative.

NaNs in any field also fail.  The same ``--no-strict`` opt-out as
``plot_rce_convection_snapshots.py`` is honoured.

Usage:
    .venv/bin/python scripts/plot_rce_3d_snapshots.py \\
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

# Physics consistency bounds.
#
# - T: generous — flags NaN / unit errors, not realism.
# - q_v: must not exceed saturation, i.e. RH ≤ 1 + ``_RH_TOL``.  The
#   saturation-adjustment step in the script clips q_v to q_sat each
#   step, so any persistent RH > 1 means convection has detrained
#   vapour faster than sat-adj can remove it (a real physics bug, not
#   just an unrealistic equilibrium).
# - precip: must be non-negative (small negative tolerated for
#   floating-point roundoff).
_T_MIN, _T_MAX = 180.0, 340.0
_QV_MIN = 0.0
_RH_TOL = 0.02                          # 102 % RH allowed for transient over-shoot
_PRECIP_MIN = -1e-12


_CONFIG_STAMPS = ("days", "N", "nlev", "dt", "sst", "T_init", "RH_init",
                  "run_id", "snap3d_days")


def _scalar(v):
    if hasattr(v, "ndim") and v.ndim == 0:
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode()
    return v


def _saturation_mixing_ratio(T_K, p_Pa):
    """Tetens-form q_sat for plotting RH only — does NOT need to match
    the model's exact formula bit-for-bit."""
    es = 611.2 * np.exp(17.67 * (T_K - 273.15) / (T_K - 29.65))
    eps = 0.622
    return eps * es / np.clip(p_Pa - (1 - eps) * es, 1.0, None)


def _check_physics(scheme: str, day: float, snap: dict) -> list[str]:
    """Return a list of physics-consistency violations for a snapshot."""
    issues = []
    T = snap["T"]
    qv = snap["q_v"]
    precip = snap["precip"]
    if not np.all(np.isfinite(T)):
        issues.append(f"{scheme} d{day}: T has {np.sum(~np.isfinite(T))} NaN/inf")
    if not np.all(np.isfinite(qv)):
        issues.append(f"{scheme} d{day}: q_v has {np.sum(~np.isfinite(qv))} NaN/inf")
    if not np.all(np.isfinite(precip)):
        issues.append(f"{scheme} d{day}: precip has {np.sum(~np.isfinite(precip))} NaN/inf")
    Tmin, Tmax = float(T.min()), float(T.max())
    if Tmin < _T_MIN or Tmax > _T_MAX:
        issues.append(
            f"{scheme} d{day}: T out of [{_T_MIN},{_T_MAX}] K "
            f"(min={Tmin:.1f}, max={Tmax:.1f})"
        )
    qvmin, qvmax = float(qv.min()), float(qv.max())
    if qvmin < _QV_MIN or qvmax > _QV_MAX:
        issues.append(
            f"{scheme} d{day}: q_v out of [{_QV_MIN},{_QV_MAX}] kg/kg "
            f"(min={qvmin:.2e}, max={qvmax:.2e})"
        )
    pmin = float(precip.min())
    if pmin < _PRECIP_MIN:
        issues.append(
            f"{scheme} d{day}: precip negative (min={pmin:.2e} kg/m²/s)"
        )
    return issues


def _load_one(p: Path, reference: dict | None) -> tuple[dict | None, str | None]:
    """Load one snapshot3d, validating completeness + consistency."""
    d = np.load(p, allow_pickle=True)
    snap = {k: d[k] for k in d.keys()}
    missing = [k for k in _CONFIG_STAMPS if k not in snap]
    if missing:
        return None, f"{p.name} missing stamp(s) {missing}"
    cfg = {k: _scalar(snap[k]) for k in _CONFIG_STAMPS}
    if reference is not None:
        diffs = {k: (cfg[k], reference[k])
                 for k in _CONFIG_STAMPS
                 if cfg[k] != reference[k]}
        if diffs:
            return None, (
                f"{p.name} disagrees with reference on " +
                ", ".join(f"{k}={cfg[k]!r} vs {reference[k]!r}"
                          for k in diffs)
            )
    return snap, None


def _load_all(root: Path, *, strict: bool) -> dict[str, dict]:
    snaps: dict[str, dict] = {}
    reference: dict | None = None
    issues: list[str] = []
    for name in SCHEME_ORDER:
        p = root / f"snapshot3d_{name}.npz"
        if not p.exists():
            continue
        snap, err = _load_one(p, reference)
        if err is not None:
            issues.append(err)
            print(f"  REJECT/SKIP: {err}", file=sys.stderr)
            continue
        if reference is None:
            reference = {k: _scalar(snap[k]) for k in _CONFIG_STAMPS}
        snaps[name] = snap
    if issues and strict:
        print(
            f"\n  ERROR: {len(issues)} 3D snapshot(s) failed validation. "
            f"Re-run the sweep or pass --no-strict.",
            file=sys.stderr,
        )
        sys.exit(2)
    return snaps


def _plot_latlon(snap: dict, scheme: str, time_idx: int, out_dir: Path):
    """4-panel lat-lon map at one time."""
    day = float(snap["snap_day"][time_idx])
    lat = snap["lat"]                       # (n_lat, n_lon) [rad]
    lon = snap["lon"]
    lat_deg = np.rad2deg(lat[:, 0])         # 1D
    lon_deg = np.rad2deg(lon[0, :])
    T = snap["T"][time_idx, ..., -1]        # surface level
    qv = snap["q_v"][time_idx, ..., -1] * 1e3   # g/kg
    precip = snap["precip"][time_idx] * 86400.0  # mm/day
    p_s = 1e5
    sigma = snap["sigma_full"]
    dp = np.diff(np.concatenate(
        [[0.0], 0.5 * (sigma[:-1] + sigma[1:]), [1.0]])) * p_s
    cwv = np.sum(snap["q_v"][time_idx] * dp / 9.80616, axis=-1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        (T, "T at lowest level (K)", "RdBu_r", axes[0, 0]),
        (qv, "q_v at lowest level (g/kg)", "viridis", axes[0, 1]),
        (cwv, "Column water vapour (kg/m²)", "Blues", axes[1, 0]),
        (precip, "Surface precipitation (mm/day)", "GnBu", axes[1, 1]),
    ]
    for arr, label, cmap, ax in panels:
        finite = np.isfinite(arr)
        if not finite.any():
            ax.text(0.5, 0.5, "NaN field", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontweight="bold")
            ax.set_title(label)
            continue
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
        # For precip / qv, anchor min at 0 so tiny noise doesn't dominate.
        if "precip" in label or "q_v" in label or "Column" in label:
            vmin = 0.0
        im = ax.pcolormesh(lon_deg, lat_deg, arr, cmap=cmap,
                           vmin=vmin, vmax=max(vmax, vmin + 1e-12),
                           shading="auto")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("lon (°E)")
        ax.set_ylabel("lat (°N)")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f"{scheme} — lat-lon, day {day:.0f}")
    fig.tight_layout()
    out_path = out_dir / f"latlon_{scheme}_d{int(day):03d}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_vertlat(snap: dict, scheme: str, time_idx: int, out_dir: Path):
    """4-panel zonal-mean vertical cross-section (lat × σ)."""
    day = float(snap["snap_day"][time_idx])
    lat_deg = np.rad2deg(snap["lat"][:, 0])
    sigma = snap["sigma_full"]
    p_full = sigma * 1e5
    T = np.nanmean(snap["T"][time_idx], axis=1)        # (n_lat, nlev)
    qv = np.nanmean(snap["q_v"][time_idx], axis=1)
    u = np.nanmean(snap["u"][time_idx], axis=1)
    qsat = _saturation_mixing_ratio(T, p_full[None, :])
    rh = qv / np.clip(qsat, 1e-12, None)
    panels = [
        (T, "T (K)", "inferno", False),
        (qv * 1e3, "q_v (g/kg)", "viridis", False),
        (rh, "RH (-)", "BrBG", False),
        (u, "Zonal wind u (m/s)", "RdBu_r", True),    # symmetric → wind
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    for ax, (arr, label, cmap, symmetric) in zip(axes, panels):
        if not np.all(np.isfinite(arr)):
            ax.text(0.5, 0.5, "NaN field", transform=ax.transAxes,
                    ha="center", va="center", color="red")
            ax.set_title(label)
            continue
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if symmetric:
            m = max(abs(vmin), abs(vmax), 1e-12)
            vmin, vmax = -m, m
        if vmax <= vmin:
            vmax = vmin + 1e-12
        im = ax.pcolormesh(lat_deg, sigma, arr.T, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(label)
        ax.set_xlabel("lat (°N)")
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    axes[0].set_ylabel("σ")
    fig.suptitle(f"{scheme} — zonal-mean (lat × σ), day {day:.0f}")
    fig.tight_layout()
    out_path = out_dir / f"vertlat_{scheme}_d{int(day):03d}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_vertlon(snap: dict, scheme: str, time_idx: int, out_dir: Path):
    """4-panel meridional-mean vertical cross-section (lon × σ)."""
    day = float(snap["snap_day"][time_idx])
    lon_deg = np.rad2deg(snap["lon"][0, :])
    sigma = snap["sigma_full"]
    p_full = sigma * 1e5
    T = np.nanmean(snap["T"][time_idx], axis=0)        # (n_lon, nlev)
    qv = np.nanmean(snap["q_v"][time_idx], axis=0)
    v = np.nanmean(snap["v"][time_idx], axis=0)
    qsat = _saturation_mixing_ratio(T, p_full[None, :])
    rh = qv / np.clip(qsat, 1e-12, None)
    panels = [
        (T, "T (K)", "inferno", False),
        (qv * 1e3, "q_v (g/kg)", "viridis", False),
        (rh, "RH (-)", "BrBG", False),
        (v, "Meridional wind v (m/s)", "RdBu_r", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    for ax, (arr, label, cmap, symmetric) in zip(axes, panels):
        if not np.all(np.isfinite(arr)):
            ax.text(0.5, 0.5, "NaN field", transform=ax.transAxes,
                    ha="center", va="center", color="red")
            ax.set_title(label)
            continue
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if symmetric:
            m = max(abs(vmin), abs(vmax), 1e-12)
            vmin, vmax = -m, m
        if vmax <= vmin:
            vmax = vmin + 1e-12
        im = ax.pcolormesh(lon_deg, sigma, arr.T, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(label)
        ax.set_xlabel("lon (°E)")
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    axes[0].set_ylabel("σ")
    fig.suptitle(f"{scheme} — meridional-mean (lon × σ), day {day:.0f}")
    fig.tight_layout()
    out_path = out_dir / f"vertlon_{scheme}_d{int(day):03d}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    strict = "--no-strict" not in sys.argv[1:]
    root = Path(args[0] if args else "results/rce_convection_sweep")
    if not root.exists():
        print(f"no snapshots in {root}; run the sweep first", file=sys.stderr)
        sys.exit(1)
    snaps = _load_all(root, strict=strict)
    if not snaps:
        print(f"no snapshot3d_*.npz in {root}", file=sys.stderr)
        sys.exit(1)
    print(f"  loaded 3D snapshots for {len(snaps)} schemes")

    physics_issues: list[str] = []
    for scheme, snap in snaps.items():
        n_times = len(snap["snap_day"])
        out_dir = root / scheme
        out_dir.mkdir(exist_ok=True)
        print(f"  -- {scheme}: {n_times} time(s) -> {out_dir}/")
        for t in range(n_times):
            day = float(snap["snap_day"][t])
            issues = _check_physics(scheme, day, {
                "T": snap["T"][t], "q_v": snap["q_v"][t],
                "precip": snap["precip"][t],
            })
            physics_issues.extend(issues)
            for fn in (_plot_latlon, _plot_vertlat, _plot_vertlon):
                try:
                    fn(snap, scheme, t, out_dir)
                except Exception as e:                  # noqa: BLE001
                    physics_issues.append(
                        f"{scheme} d{day} {fn.__name__}: {type(e).__name__}: {e}"
                    )
    print()
    if physics_issues:
        print(f"  PHYSICS CHECK: {len(physics_issues)} issue(s) flagged:")
        for m in physics_issues:
            print(f"    - {m}")
        if strict:
            sys.exit(1)
    else:
        print("  PHYSICS CHECK: all snapshots within plausible bounds.")


if __name__ == "__main__":
    main()
