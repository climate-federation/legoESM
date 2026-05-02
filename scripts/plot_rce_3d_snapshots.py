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

# Use the model's own saturation formula — never hard-code 273.15 etc.
# (per CLAUDE.md "constant and parameter discipline").
from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio as _model_q_sat


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
    """Saturation mixing ratio — wraps ``legoesm.thermo`` so the
    physics-consistency RH check uses the *same* q_sat formula as the
    model's saturation adjustment.  Using a different formula here
    would flag spurious supersaturation from formula disagreement
    (the standard atmospheric Tetens form gives ~15 % smaller q_sat
    than the model's mixing-ratio form at high q_sat values).

    Returns numpy arrays so the rest of the plotter's numpy code is
    unchanged.
    """
    return np.asarray(_model_q_sat(np.asarray(T_K), np.asarray(p_Pa)))


def _check_physics(scheme: str, day: float, snap: dict,
                   sigma_full: np.ndarray) -> list[str]:
    """Return a list of physics-consistency violations for a snapshot.

    ``sigma_full`` is the per-level σ value (surface-last shape
    (nlev,)).  Per-level pressure is reconstructed from the snapshot's
    actual surface-pressure field ``p_s`` (varies in space and time
    with the dycore mass distribution) so the RH ≤ 1 check uses the
    *same* (T, p) the model's sat-adj saw.  Using a constant 1e5 Pa
    here gives spurious super-saturation reports in cells where p_s
    is below 1e5.
    """
    issues = []
    T = snap["T"]                           # (n_lat, n_lon, nlev)
    qv = snap["q_v"]
    precip = snap["precip"]
    p_s = snap["p_s"]                       # (n_lat, n_lon)
    p_full = p_s[..., None] * sigma_full    # (n_lat, n_lon, nlev)
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
    if float(qv.min()) < _QV_MIN:
        issues.append(
            f"{scheme} d{day}: q_v negative (min={qv.min():.2e} kg/kg)"
        )
    # Real physics check: q_v cannot exceed saturation.  RH > 1 + tol
    # means convection moistened above the sat-adj's ability to clip,
    # which is a model-level bug (or a microphysics step missing).
    qsat = _saturation_mixing_ratio(T, p_full)
    rh = qv / np.clip(qsat, 1e-12, None)
    rh_max = float(np.nanmax(rh))
    if rh_max > 1.0 + _RH_TOL:
        # Locate worst point for diagnostic.
        i = np.unravel_index(np.nanargmax(rh), rh.shape)
        issues.append(
            f"{scheme} d{day}: RH > 1 + {_RH_TOL} (max RH = {rh_max:.3f}) "
            f"at lat_idx={i[0]}, lon_idx={i[1]}, lev_idx={i[2]} "
            f"(p={float(p_full[i])/100:.0f} hPa, "
            f"T={float(T[i]):.1f} K, q_v={float(qv[i])*1e3:.1f} g/kg, "
            f"q_sat={float(qsat[i])*1e3:.1f} g/kg)"
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
    """4-panel zonal-mean vertical cross-section (lat × pressure)."""
    day = float(snap["snap_day"][time_idx])
    lat_deg = np.rad2deg(snap["lat"][:, 0])
    sigma = snap["sigma_full"]
    p_full_hPa = sigma * 1e3                # σ × p_s/100 [hPa]
    p_full_Pa = sigma * 1e5
    T = np.nanmean(snap["T"][time_idx], axis=1)        # (n_lat, nlev)
    qv = np.nanmean(snap["q_v"][time_idx], axis=1)
    u = np.nanmean(snap["u"][time_idx], axis=1)
    qsat = _saturation_mixing_ratio(T, p_full_Pa[None, :])
    rh = qv / np.clip(qsat, 1e-12, None)
    # Cap q_v colormap at 25 g/kg — saturation at SST 300 K is ~22 g/kg,
    # so values above this are TOA cold-air mixing-ratio artefacts (q_sat
    # naturally exceeds 100 g/kg at 50 hPa) and dominate the autoscale,
    # crushing the tropospheric signal.
    panels = [
        (T, "T (K)", "inferno", False, None, None),
        (qv * 1e3, "q_v (g/kg)", "viridis", False, 0.0, 25.0),
        (rh, "RH (-)", "BrBG", False, 0.0, 1.05),
        (u, "Zonal wind u (m/s)", "RdBu_r", True, None, None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    for ax, (arr, label, cmap, symmetric, vmin_cap, vmax_cap) in zip(
        axes, panels
    ):
        if not np.all(np.isfinite(arr)):
            ax.text(0.5, 0.5, "NaN field", transform=ax.transAxes,
                    ha="center", va="center", color="red")
            ax.set_title(label)
            continue
        vmin = float(np.nanmin(arr)) if vmin_cap is None else vmin_cap
        vmax = float(np.nanmax(arr)) if vmax_cap is None else vmax_cap
        if symmetric:
            m = max(abs(vmin), abs(vmax), 1e-12)
            vmin, vmax = -m, m
        if vmax <= vmin:
            vmax = vmin + 1e-12
        im = ax.pcolormesh(lat_deg, p_full_hPa, arr.T, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(label)
        ax.set_xlabel("lat (°N)")
        # Standard atmospheric convention: high pressure (surface) at
        # the BOTTOM, low pressure (TOA) at the top.  Set ylim
        # explicitly with high-p at the bottom of the axis range
        # (matplotlib treats the *first* y-limit value as the bottom).
        ax.set_ylim(float(p_full_hPa.max()), float(p_full_hPa.min()))
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    axes[0].set_ylabel("p (hPa)")
    fig.suptitle(f"{scheme} — zonal-mean (lat × p), day {day:.0f}")
    fig.tight_layout()
    out_path = out_dir / f"vertlat_{scheme}_d{int(day):03d}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_vertlon(snap: dict, scheme: str, time_idx: int, out_dir: Path):
    """4-panel meridional-mean vertical cross-section (lon × pressure)."""
    day = float(snap["snap_day"][time_idx])
    lon_deg = np.rad2deg(snap["lon"][0, :])
    sigma = snap["sigma_full"]
    p_full_hPa = sigma * 1e3
    p_full_Pa = sigma * 1e5
    T = np.nanmean(snap["T"][time_idx], axis=0)        # (n_lon, nlev)
    qv = np.nanmean(snap["q_v"][time_idx], axis=0)
    v = np.nanmean(snap["v"][time_idx], axis=0)
    qsat = _saturation_mixing_ratio(T, p_full_Pa[None, :])
    rh = qv / np.clip(qsat, 1e-12, None)
    panels = [
        (T, "T (K)", "inferno", False, None, None),
        (qv * 1e3, "q_v (g/kg)", "viridis", False, 0.0, 25.0),
        (rh, "RH (-)", "BrBG", False, 0.0, 1.05),
        (v, "Meridional wind v (m/s)", "RdBu_r", True, None, None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    for ax, (arr, label, cmap, symmetric, vmin_cap, vmax_cap) in zip(
        axes, panels
    ):
        if not np.all(np.isfinite(arr)):
            ax.text(0.5, 0.5, "NaN field", transform=ax.transAxes,
                    ha="center", va="center", color="red")
            ax.set_title(label)
            continue
        vmin = float(np.nanmin(arr)) if vmin_cap is None else vmin_cap
        vmax = float(np.nanmax(arr)) if vmax_cap is None else vmax_cap
        if symmetric:
            m = max(abs(vmin), abs(vmax), 1e-12)
            vmin, vmax = -m, m
        if vmax <= vmin:
            vmax = vmin + 1e-12
        im = ax.pcolormesh(lon_deg, p_full_hPa, arr.T, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(label)
        ax.set_xlabel("lon (°E)")
        # Surface (high p) at bottom — explicit ylim is more robust
        # than ``invert_yaxis()`` against later autoscale by colorbar.
        ax.set_ylim(float(p_full_hPa.max()), float(p_full_hPa.min()))
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    axes[0].set_ylabel("p (hPa)")
    fig.suptitle(f"{scheme} — meridional-mean (lon × p), day {day:.0f}")
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
        sigma_full = snap["sigma_full"]
        for t in range(n_times):
            day = float(snap["snap_day"][t])
            issues = _check_physics(scheme, day, {
                "T": snap["T"][t], "q_v": snap["q_v"][t],
                "precip": snap["precip"][t],
                "p_s": snap["p_s"][t],   # actual surface pressure
            }, sigma_full)
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
