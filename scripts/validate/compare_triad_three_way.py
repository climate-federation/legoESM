"""Three-way Eady GM/Redi comparison at high kappa using triads.

Runs the same Eady uniform setup (no perturbation, 20x10 latlon channel,
beta_S = 0 so rho = f(T) cleanly) under three configurations:

    1. baseline   — kappa_GM = 0, kappa_Redi = 0
    2. redi_only  — kappa_GM = 0, kappa_Redi = 5e4   (expect ~baseline)
    3. gm_only    — kappa_GM = 5e4, kappa_Redi = 0   (expect isopycnal flattening)

All three use slope_scheme="triads".  kappa is boosted 50x above the
production default so the GM signature is visible at this coarse
resolution.

Default duration is 120 days — long enough to expose any cumulative
divergence between Redi-only and baseline, and to let GM appreciably
flatten the isopycnals.

After the runs finish, generates two custom comparison plots that the
default test-matrix output doesn't provide:

  - ``compare_T_initial_vs_final.png`` — T(y,z) at t=0 and t=final for
    each case, with BOTH initial (dashed) and current (solid) isotherm
    contours overlaid.  This makes isopycnal flattening directly
    visible (the default matrix plot only overlays the initial
    isotherms as a static reference, which is easy to misread).
  - ``gm_only_streamfunction.png`` — the GM bolus streamfunction
    ψ_GM(y,z) = κ_GM · S_y(zonally averaged), plus the meridional
    bolus velocity ⟨v*⟩ = -∂ψ/∂z.  Should show a single overturning
    cell that flattens the tilted isopycnals.

Outputs land in results/ocean/triad_three_way/<case>/...
"""

from __future__ import annotations

import os
import sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "matrix"))  # ocean_test_matrix package home

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from legoesm import constants
from ocean_test_matrix.experiments import run_eady_gm_redi
from ocean_test_matrix.testcase import TestCase


KAPPA = 5.0e4
DURATION_DAYS = 120.0
QUICK_DAYS = 10.0


def _make_tc(name: str, gm_mode: str,
             kappa_GM: float, kappa_Redi: float,
             duration: float = DURATION_DAYS) -> TestCase:
    return TestCase(
        case=name,
        grid_type="latlon_channel",
        resolution="20x10",
        duration_days=duration,
        quick_days=QUICK_DAYS,
        run_kwargs={
            "gm_mode": gm_mode,
            "slope_scheme": "triads",
            "kappa_GM_override": kappa_GM,
            "kappa_Redi_override": kappa_Redi,
            "beta_S_override": 0.0,
        },
    )


def _zonal_mean(field_yxz: np.ndarray, mask_yx: np.ndarray) -> np.ndarray:
    """Zonal-mean a (n_lat, n_lon, n_lev) field with a 2D wet mask."""
    m = mask_yx[:, :, None].astype(field_yxz.dtype)
    num = np.sum(field_yxz * m, axis=1)
    den = np.sum(m, axis=1)
    den = np.where(den > 0, den, 1.0)
    return num / den  # (n_lat, n_lev)


def _plot_T_initial_vs_final(out_root: Path) -> None:
    cases = ["baseline", "redi_only", "gm_only"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)

    # Same colour scale across all panels for fair comparison.
    T_lo, T_hi = None, None
    iso_levels = None
    panels = {}
    for case in cases:
        d = np.load(out_root / case / "snapshots_native.npz")
        T = np.asarray(d["T_3d"])             # (n_snap, n_lat, n_lon, n_lev)
        mask = np.asarray(d["land_mask"][0])  # (n_lat, n_lon)
        T0 = _zonal_mean(T[0], mask)          # (n_lat, n_lev)
        T1 = _zonal_mean(T[-1], mask)
        panels[case] = (T0, T1)
        if T_lo is None:
            T_lo, T_hi = float(T.min()), float(T.max())
            iso_levels = np.linspace(T_lo, T_hi, 9)[1:-1]

    # Pull lat/depth coords from the first case (all three share grid).
    d0 = np.load(out_root / cases[0] / "snapshots_native.npz")
    n_lat = d0["T_3d"].shape[1]
    n_lev = d0["T_3d"].shape[3]
    # Channel goes from 15S→35S equivalent in degrees; the matrix plot
    # uses the same lat range. Plot in row-index for simplicity, label
    # as latitude index since true lats aren't carried in the npz.
    lat_idx = np.arange(n_lat)
    z_top_to_bot = np.linspace(0, 5000, n_lev)  # nominal depths

    for col, case in enumerate(cases):
        T0, T1 = panels[case]
        for row, (T_field, label) in enumerate([(T0, "t=0 d"),
                                                (T1, f"t={DURATION_DAYS:g} d")]):
            ax = axes[row, col]
            pc = ax.pcolormesh(lat_idx, z_top_to_bot, T_field.T,
                               cmap="RdBu_r", vmin=T_lo, vmax=T_hi,
                               shading="auto")
            # initial isotherms (dashed gray) — fixed reference
            ax.contour(lat_idx, z_top_to_bot, T0.T, levels=iso_levels,
                       colors="0.3", linewidths=0.6, linestyles="--")
            # current isotherms (solid black) — live
            ax.contour(lat_idx, z_top_to_bot, T_field.T, levels=iso_levels,
                       colors="k", linewidths=1.0)
            ax.invert_yaxis()
            if row == 0:
                ax.set_title(case)
            if col == 0:
                ax.set_ylabel(f"Depth (m)\n{label}")
            if row == 1:
                ax.set_xlabel("Latitude index")

    fig.suptitle(
        f"Eady triad three-way @ kappa={KAPPA:g}, beta_S=0, "
        f"{DURATION_DAYS:g} days\n"
        f"dashed = initial isotherms (reference); solid = current isotherms"
    )
    cbar = fig.colorbar(pc, ax=axes, shrink=0.85, label="T [degC]")
    out = out_root / "compare_T_initial_vs_final.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _plot_gm_streamfunction(out_root: Path) -> None:
    """Compute and plot ψ_GM(y,z) = κ_GM · ⟨S_y⟩_x for the gm_only case.

    Computes the meridional isopycnal slope from the saved zonal-mean
    density via finite differences (good enough for visualization).
    Density is reconstructed from T using the same linear EOS as the
    run (β_S = 0, α_T = 1.7e-4, rho_0 = 1027.5).

    Plots three snapshots (initial / mid / final), top row: ψ_GM in
    Sverdrups-equivalent (m²/s); bottom row: zonal-mean T with
    isotherms overlaid.
    """
    d = np.load(out_root / "gm_only" / "snapshots_native.npz")
    T = np.asarray(d["T_3d"])                # (n_snap, n_lat, n_lon, n_lev)
    mask_yx = np.asarray(d["land_mask"][0])  # (n_lat, n_lon)
    n_snap, n_lat, n_lon, n_lev = T.shape

    # Reproduce z-star levels used by the run (matches
    # create_ocean_z_star defaults from EadyUniformConfig).
    from legoesm.ocean.vertical import create_ocean_z_star
    z_coord = create_ocean_z_star(n_levels=n_lev, H_max=5000.0,
                                  dz_surface=10.0, dz_deep=500.0)
    dz = np.asarray(z_coord.dz_ref)              # (n_lev,)
    z_centers = np.cumsum(dz) - 0.5 * dz         # cell-centre depths
    z_interfaces = np.cumsum(dz)[:-1]            # internal interfaces (n_lev-1)

    # Approximate dy from the channel extent (15→35°S = 20° lat,
    # n_lat - 2 active rows).  R_earth = 6.371e6 m.
    R_E = constants.R_earth
    n_lat_active = n_lat - 2
    dlat_deg = 20.0 / n_lat_active
    dy = R_E * np.deg2rad(dlat_deg)

    # Linear EOS parameters used by the run (Eady default + β_S=0).
    alpha_T = 1.7e-4
    rho_0 = 1027.5
    T_ref = 10.0
    S_max = 0.01

    snap_idxs = [0, n_snap // 2, n_snap - 1]
    times = np.asarray(d["times_days"])

    psis = []
    Tms = []
    psi_max = 0.0
    # Active wet rows are 1..n_lat-2 (rows 0 and n_lat-1 are wall ghost
    # cells with T=0).  Compute slopes only on the wet subdomain.
    wet = slice(1, n_lat - 1)
    n_wet = n_lat - 2
    for i in snap_idxs:
        T_i = T[i]
        Tm = _zonal_mean(T_i, mask_yx)            # (n_lat, n_lev)
        Tm_wet = Tm[wet, :]                       # (n_wet, n_lev)
        rho = rho_0 * (1.0 - alpha_T * (Tm_wet - T_ref))   # (n_wet, n_lev)

        # Centered drho/dy with one-sided differences at the wet
        # boundary (no ghost-row contamination).
        drho_dy = np.zeros_like(rho)
        drho_dy[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / (2.0 * dy)
        drho_dy[0, :] = (rho[1, :] - rho[0, :]) / dy
        drho_dy[-1, :] = (rho[-1, :] - rho[-2, :]) / dy

        # drho/dz at internal interfaces.  legoesm convention: z is
        # positive downward, so stable stratification gives drho/dz > 0.
        drho_dz = (rho[:, 1:] - rho[:, :-1]) / (z_centers[1:] - z_centers[:-1])

        # drho/dy averaged onto the same internal interfaces.
        drho_dy_int = 0.5 * (drho_dy[:, 1:] + drho_dy[:, :-1])

        # Guard drho/dz with a small POSITIVE floor (stable strat).
        drho_dz_safe = np.where(drho_dz > 1e-12, drho_dz, 1e-12)
        S_y = -drho_dy_int / drho_dz_safe

        # DM95 tanh taper.
        taper = 0.5 * (1.0 + np.tanh((S_max - np.abs(S_y)) / (0.1 * S_max)))
        psi_wet = KAPPA * S_y * taper            # (n_wet, n_lev-1)

        # Embed back into the full (n_lat, n_lev-1) grid for plotting.
        psi = np.zeros((n_lat, n_lev - 1))
        psi[wet, :] = psi_wet

        psis.append(psi)
        Tms.append(Tm)
        psi_max = max(psi_max, float(np.abs(psi_wet).max()))

    iso_levels = np.linspace(np.min(Tms), np.max(Tms), 9)[1:-1]
    lat_idx = np.arange(n_lat)

    fig, axes = plt.subplots(2, len(snap_idxs), figsize=(14, 7),
                             sharex=True, sharey=True)
    for col, (i, psi, Tm) in enumerate(zip(snap_idxs, psis, Tms)):
        ax = axes[0, col]
        pc = ax.pcolormesh(lat_idx, z_interfaces, psi.T,
                           cmap="RdBu_r", vmin=-psi_max, vmax=psi_max,
                           shading="auto")
        ax.contour(lat_idx, z_centers, Tm.T,
                   levels=iso_levels, colors="k", linewidths=0.7)
        ax.invert_yaxis()
        ax.set_title(f"t={times[i]:.1f} d")
        if col == 0:
            ax.set_ylabel("Depth (m)\nψ_GM [m²/s]")
        fig.colorbar(pc, ax=ax, shrink=0.85)

        ax2 = axes[1, col]
        pc2 = ax2.pcolormesh(lat_idx, z_centers, Tm.T,
                             cmap="RdBu_r", shading="auto",
                             vmin=float(np.min(Tms)),
                             vmax=float(np.max(Tms)))
        ax2.contour(lat_idx, z_centers, Tm.T,
                    levels=iso_levels, colors="k", linewidths=0.7)
        # Reference: initial isotherms (dashed)
        ax2.contour(lat_idx, z_centers, Tms[0].T,
                    levels=iso_levels, colors="0.4",
                    linewidths=0.6, linestyles="--")
        ax2.invert_yaxis()
        ax2.set_xlabel("Latitude index")
        if col == 0:
            ax2.set_ylabel("Depth (m)\nT [degC]")
        fig.colorbar(pc2, ax=ax2, shrink=0.85)

    fig.suptitle(
        f"GM-only triad: bolus streamfunction ψ_GM = κ_GM · ⟨S_y⟩_x · taper  "
        f"(κ_GM={KAPPA:g}, β_S=0)"
    )
    fig.tight_layout()
    out = out_root / "gm_only_streamfunction.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main() -> None:
    out_root = _REPO / "results" / "ocean" / "triad_three_way"
    out_root.mkdir(parents=True, exist_ok=True)

    cases = [
        _make_tc("baseline", "baseline", 0.0, 0.0),
        _make_tc("redi_only", "redi_only", 0.0, KAPPA),
        _make_tc("gm_only", "gm_only", KAPPA, 0.0),
    ]

    print("=" * 78)
    print(f" Triad three-way Eady comparison (κ={KAPPA:g}, β_S=0, "
          f"{DURATION_DAYS:g} days)")
    print("=" * 78)

    results = []
    for tc in cases:
        out = out_root / tc.case
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {tc.case} ---")
        status, wall, notes = run_eady_gm_redi(tc, out, days=tc.duration_days)
        print(f"  {status}  ({wall:.1f}s)  {notes}")
        results.append((tc.case, status, wall, notes))

    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    for case, status, wall, notes in results:
        print(f"  {status}  {case:12s}  {wall:6.1f}s  {notes}")
    print(f"\n  Outputs in: {out_root}")

    print("\n--- post-processing ---")
    _plot_T_initial_vs_final(out_root)
    _plot_gm_streamfunction(out_root)

    # Quantify Redi vs baseline divergence at the final snapshot.
    d_b = np.load(out_root / "baseline" / "snapshots_native.npz")
    d_r = np.load(out_root / "redi_only" / "snapshots_native.npz")
    d_g = np.load(out_root / "gm_only" / "snapshots_native.npz")
    T_b = d_b["T_3d"][-1]
    T_r = d_r["T_3d"][-1]
    T_g = d_g["T_3d"][-1]
    mask = d_b["land_mask"][0].astype(bool)
    T_b_ocean = T_b[mask]
    T_r_ocean = T_r[mask]
    T_g_ocean = T_g[mask]
    rms_redi = float(np.sqrt(np.mean((T_r_ocean - T_b_ocean) ** 2)))
    rms_gm = float(np.sqrt(np.mean((T_g_ocean - T_b_ocean) ** 2)))
    max_redi = float(np.max(np.abs(T_r_ocean - T_b_ocean)))
    max_gm = float(np.max(np.abs(T_g_ocean - T_b_ocean)))
    print(f"\n  RMS(Redi - baseline) = {rms_redi:.3e} K   (max {max_redi:.3e})")
    print(f"  RMS(GM   - baseline) = {rms_gm:.3e} K   (max {max_gm:.3e})")
    if rms_redi > 0:
        print(f"  GM/Redi divergence ratio: {rms_gm / rms_redi:.1f}x")


if __name__ == "__main__":
    main()
