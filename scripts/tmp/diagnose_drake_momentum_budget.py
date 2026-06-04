#!/usr/bin/env python
"""Drake Passage zonal-momentum budget diagnostic.

Computes the depth-and-zonal-integrated band-mean zonal-momentum budget
for the Drake band (j=2..6) of the global-overturning + GM/Redi
configuration, expressed as an equivalent stress in Pa for direct
comparison with the surface wind stress.

The depth-integrated zonal-momentum equation, zonally averaged in a
periodic channel and assuming steady state, is:

    0 = F_wind  +  F_drag_bot  +  F_drag_baro  -  F_adv  +  F_visc

Where, all in Pa (force per unit area) when divided by the band area:

    F_wind     = τ_x  applied at the surface (eastward = +)
    F_drag_bot = -ρ·r·u_bot   (path 1: explicit bottom-cell drag)
    F_drag_baro= -ρ·r·U_baro  (path 3: depth-mean drag on barotropic
                                substep — currently active and redundant
                                with path 1+2 when bottom_drag_r > 0)
    F_adv      =  ∂_y⟨∫ v·u dz⟩, signed so it appears with a minus
                  in the LHS budget — i.e. positive F_adv means the band
                  loses eastward momentum to the meridional advective flux
                  divergence (so it acts like a westward force on U_baro).
    F_visc     = A_h·∂²U_baro/∂y² · H   (lateral momentum diffusion)

Residual = F_wind + F_drag_bot + F_drag_baro - F_adv + F_visc.  Should be
small if the budget closes and our time-mean approximation is adequate.

Usage:
    JAX_ENABLE_X64=1 python scripts/diagnose_drake_momentum_budget.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.experiments.global_overturning import GlobalOverturningConfig


J_DRAKE = np.arange(2, 7)         # cell rows  (lat -77.5 .. -57.5)
J_V_S   = int(J_DRAKE[0])         # 2 — v-face index south of band
J_V_N   = int(J_DRAKE[-1] + 1)    # 7 — v-face index north of band

RHO_0 = 1027.0


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_time_mean_from_npz(time_mean_path: Path):
    d = np.load(time_mean_path, allow_pickle=False)
    return {
        "T": d["T"], "S": d["S"],
        "u": d["u"], "v": d["v"], "w": d["w"], "eta": d["eta"],
        "n_samples": int(d["n_samples"]),
        "label_days": (float(d["day_start"]), float(d["day_end"])),
    }


def _load_time_mean_from_restarts(restart_paths):
    """Average u, v, T, eta over a list of restart files (used for the
    50-yr run which only has 5-yr-spaced restarts)."""
    sum_T = sum_S = sum_u = sum_v = sum_eta = sum_w = None
    days = []
    n = 0
    for p in restart_paths:
        d = np.load(p, allow_pickle=False)
        T = d["T"]; S = d["S"]; u = d["u"]; v = d["v"]; eta = d["eta"]
        w = d["w"] if "w" in d.files else np.zeros_like(T)
        if sum_T is None:
            sum_T = np.zeros_like(T, dtype=np.float64)
            sum_S = np.zeros_like(S, dtype=np.float64)
            sum_u = np.zeros_like(u, dtype=np.float64)
            sum_v = np.zeros_like(v, dtype=np.float64)
            sum_w = np.zeros_like(w, dtype=np.float64)
            sum_eta = np.zeros_like(eta, dtype=np.float64)
        sum_T += T; sum_S += S; sum_u += u; sum_v += v
        sum_w += w; sum_eta += eta
        days.append(float(d["time_days"]))
        n += 1
    return {
        "T":   sum_T / n, "S": sum_S / n,
        "u":   sum_u / n, "v": sum_v / n, "w": sum_w / n,
        "eta": sum_eta / n,
        "n_samples": n,
        "label_days": (days[0], days[-1]),
    }, n


# ---------------------------------------------------------------------------
# Geometry / forcing helpers
# ---------------------------------------------------------------------------

def _build_geometry():
    cfg = GlobalOverturningConfig(use_gm_redi=True)
    z_coord = create_ocean_z_star(
        n_levels=cfg.n_levels, H_max=cfg.H_max,
        dz_surface=cfg.dz_surface, dz_deep=cfg.dz_deep,
    )
    grid = create_latlon_grid(36, 72)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    H_total = float(dz.sum())
    lat = np.asarray(grid.lat, dtype=np.float64)
    lon = np.asarray(grid.lon, dtype=np.float64)
    dlat = float(grid.dlat); dlon = float(grid.dlon)
    R = float(grid.radius)
    dy = R * dlat
    # u-face longitudes: at cell west/east interfaces (n_lon+1).  For x-length
    # of a u-face we use cos(lat_centre[j])·R·dlon.
    cos_lat_c = np.cos(np.clip(lat, -np.pi/2 + 1e-9, np.pi/2 - 1e-9))
    dx_u = cos_lat_c * R * dlon                                 # (n_lat,)
    # v-face latitudes (n_lat+1)
    lat_v = np.concatenate([
        [lat[0] - 0.5 * dlat],
        0.5 * (lat[:-1] + lat[1:]),
        [lat[-1] + 0.5 * dlat],
    ])
    cos_lat_v = np.cos(np.clip(lat_v, -np.pi/2 + 1e-9, np.pi/2 - 1e-9))
    dx_v = cos_lat_v * R * dlon                                 # (n_lat+1,)
    return cfg, grid, z_coord, dz, H_total, lat, lat_v, dx_u, dx_v, dy, R


def _two_belt_wind(lat_rad, tau_max=0.1):
    phi_jet = np.radians(50.0); sigma_w = np.radians(12.0)
    sigma_t = np.radians(15.0)
    abs_lat = np.abs(lat_rad)
    return (
        -0.5 * tau_max * np.exp(-(lat_rad / sigma_t) ** 2)
        + tau_max * np.exp(-((abs_lat - phi_jet) / sigma_w) ** 2)
    )


# ---------------------------------------------------------------------------
# Budget computation
# ---------------------------------------------------------------------------

def compute_budget(state, *, r, A_h, dz, lat, lat_v, dx_u, dx_v, dy,
                   land_mask, u_mask, v_mask, has_path3_drag=True):
    """Compute the depth-integrated zonal-momentum budget for the Drake
    band.  All quantities are *total forces* in N (force in the +x
    direction).  The caller can divide by band area to get equivalent
    stress in Pa.

    Parameters
    ----------
    state : dict  with keys "u", "v", "T", "eta" (3D / 2D arrays).
    r : float, linear bottom-drag rate [m/s].
    A_h : float, lateral viscosity [m^2/s].
    has_path3_drag : bool
        Whether the depth-mean barotropic-substep drag is active.  In our
        current code this is True whenever bottom_drag_r > 0.

    Returns
    -------
    dict with each force term in N, plus diagnostics.
    """
    u   = state["u"]                 # (n_lat,    n_lon+1, nlev)
    v   = state["v"]                 # (n_lat+1,  n_lon,   nlev)
    nlev = u.shape[-1]
    n_lat = u.shape[0]
    n_lon = v.shape[1]

    # ------- zonal-mean u(j, k) at cell-centre lat (averaging u-faces) ----
    m_uw = u_mask[:, :, None]                                   # (n_lat, n_lon+1, 1)
    den_u = np.sum(m_uw, axis=1)                                # (n_lat, 1)
    zm_u = np.where(den_u > 0,
                    (u * m_uw).sum(axis=1) / np.where(den_u > 0, den_u, 1.0),
                    0.0)                                        # (n_lat, nlev)
    # depth-mean U_baro per latitude
    H_local = dz.sum()
    U_baro_lat = np.sum(zm_u * dz[None, :], axis=1) / H_local   # (n_lat,)
    u_bot_lat = zm_u[:, -1]                                     # (n_lat,)

    # Wet u-face area per row j (m^2): n_wet × dx_u(j) × dy.  Use this
    # to integrate face-quantities (drag, wind on u-face) over the row.
    n_wet_u_per_row = np.sum(u_mask, axis=1)                    # (n_lat,)

    # ------- F_wind: τ_x at the surface, integrated over wet OCEAN cells -
    # We apply τ at cell centres (mask = land_mask).  Total wind force on
    # the Drake band:
    tau_x_lat = _two_belt_wind(lat)                             # (n_lat,)
    n_wet_cell_per_row = np.sum(land_mask, axis=1)              # (n_lat,)
    F_wind_per_row = tau_x_lat * dx_u * n_wet_cell_per_row * dy
    F_wind = float(np.sum(F_wind_per_row[J_DRAKE]))

    # ------- F_drag_bot (path 1+2): bottom-cell drag, all wet u-faces ----
    # Drag stress per face (Pa, eastward) = -ρ·r·u_bot_face.  Force
    # contribution per face = -ρ·r·u_bot · dx_u(j) · dy.  Sum over wet
    # faces in band rows.
    drag_bot_per_face = (-RHO_0 * r * u[:, :, -1]) * u_mask     # (n_lat, n_lon+1)
    F_drag_bot_per_row = np.sum(drag_bot_per_face, axis=1) * dx_u * dy
    F_drag_bot = float(np.sum(F_drag_bot_per_row[J_DRAKE]))

    # ------- F_drag_baro (path 3): depth-mean drag projection -----------
    # Each wet u-face contributes -ρ·r·U_baro_face · dx · dy to total force.
    # Approximate U_baro_face from u-face values: depth-mean of u at face.
    u_baro_face = np.sum(u * dz[None, None, :], axis=-1) / H_local  # (n_lat, n_lon+1)
    drag_baro_per_face = (-RHO_0 * r * u_baro_face) * u_mask
    F_drag_baro_per_row = np.sum(drag_baro_per_face, axis=1) * dx_u * dy
    F_drag_baro = float(np.sum(F_drag_baro_per_row[J_DRAKE])) if has_path3_drag else 0.0

    # ------- F_adv: meridional flux divergence of zonal momentum --------
    # Compute ⟨∫ v·u dz⟩ at v-face latitudes.  v is on v-faces (n_lat+1, n_lon).
    # Need u co-located at v-faces.  Standard C-grid: u_at_vface(j_v, i) =
    # 0.25·( u[j_v-1, i] + u[j_v-1, i+1] + u[j_v, i] + u[j_v, i+1] ),
    # for interior j_v.  At j_v=0 or n_lat we don't care (out of domain).
    # We only need j_v = J_V_S, J_V_N which are well in the interior.
    def u_at_vface(j_v):
        # u shape (n_lat, n_lon+1, nlev).  Use cells j_v-1 and j_v.
        u_n = u[j_v,     :-1, :]          # (n_lon, nlev), i-th cell west face
        u_e = u[j_v,      1:, :]          # i-th cell east face
        u_s = u[j_v - 1, :-1, :]
        u_se= u[j_v - 1,  1:, :]
        return 0.25 * (u_n + u_e + u_s + u_se)                  # (n_lon, nlev)

    def vu_flux_through(j_v):
        u_v = u_at_vface(j_v)                                   # (n_lon, nlev)
        vu  = v[j_v] * u_v                                      # (n_lon, nlev)
        # Mask: v_mask[j_v] applies to v-faces.
        mask_v = v_mask[j_v][:, None]
        # Per face zonal momentum transport [m^3/s · m/s] times dx_v(j_v)·dz:
        # contribution to total force = ρ · ⟨vu⟩ · dx · dz
        # Sum zonally and over depth, multiply by ρ.
        per_face = vu * mask_v * dz[None, :]                    # (n_lon, nlev)
        return RHO_0 * np.sum(per_face) * dx_v[j_v]

    F_in_south  = vu_flux_through(J_V_S)         # northward flow advecting u into band
    F_in_north  = vu_flux_through(J_V_N)
    F_adv = F_in_north - F_in_south              # net export of u-momentum out of band

    # ------- F_visc: pointwise A_h · ∇²u summed over wet cells in band ---
    # The model applies viscosity to u_prime (= u - U_bar) on the
    # C-grid pointwise.  Since u_prime has zero depth-mean by
    # construction, A_h·∇²u_prime integrated over depth is *exactly*
    # zero — lateral viscosity contributes nothing to the depth-mean
    # budget.  We still report a "diagnostic" value computed as
    # A_h·∂²U_baro/∂y² (the smoothing of zonal-mean depth-mean profile)
    # for inspection, but it does NOT enter the residual sum.
    d2U_dy2 = (U_baro_lat[2:] - 2 * U_baro_lat[1:-1] + U_baro_lat[:-2]) / dy**2
    d2U_full = np.concatenate([[0.0], d2U_dy2, [0.0]])
    F_visc_per_row = (RHO_0 * A_h * d2U_full * H_local) * dx_u * n_wet_cell_per_row * dy
    F_visc_diag = float(np.sum(F_visc_per_row[J_DRAKE]))
    F_visc = 0.0   # exactly-zero contribution to the depth-mean budget

    # ------- F_vortcor: depth-integrated zonal-and-zonally-summed
    # vorticity-Coriolis term ⟨ζ × v⟩ in the u-equation, on the band's
    # wet u-faces.  ζ lives at vertices; we average to u-points.
    # ----------------------------------------------------------------
    # ζ_v[j_v, i, k] = (v[j_v, i] - v[j_v, i-1])/dx_v(j_v) - (u[j, i, k] - u[j-1, i, k])/dy
    # for j_v on (n_lat+1) lat axis.
    # Then avg to u-faces: ζ_u[j, i_u, k] = 0.5*(ζ_v[j, i_u, k] + ζ_v[j+1, i_u, k]).
    # v_at_u: average of 4 v-faces around u-face.
    # Compute pointwise ζ_u × v_at_u and integrate.
    # We do a SIMPLIFIED beta-plane-style: -∂U/∂y at each u-face from
    # finite differences.  Periodic in lon → ∂v/∂x is well defined.
    n_lat = u.shape[0]
    n_lon_p1 = u.shape[1]
    # ∂v/∂x at vertex points (n_lat+1, n_lon+1) per level: roll-based
    # (treat lon as periodic).  v has shape (n_lat+1, n_lon, nlev).
    v_east = np.roll(v, -1, axis=1)
    dvdx = (v_east - v) / dx_v[:, None, None]                    # (n_lat+1, n_lon, nlev)
    # Make it (n_lat+1, n_lon+1, nlev) by appending the rolled column at i=n_lon
    dvdx_full = np.concatenate([dvdx, dvdx[:, 0:1, :]], axis=1)  # periodic
    # ∂u/∂y at vertex points (n_lat+1, n_lon+1) per level
    u_full = u                                                    # (n_lat, n_lon+1, nlev)
    dudy = np.zeros((n_lat + 1, n_lon_p1, nlev))
    dudy[1:n_lat] = (u_full[1:] - u_full[:-1]) / dy
    zeta_v = dvdx_full - dudy                                    # (n_lat+1, n_lon+1, nlev)
    # zeta at u-faces: average over the two adjacent vertex rows
    zeta_u = 0.5 * (zeta_v[:-1] + zeta_v[1:])                    # (n_lat, n_lon+1, nlev)
    # v at u-faces: 4-point average from surrounding v-faces
    # v has shape (n_lat+1, n_lon, nlev).  For u-face (j, i_u, k):
    #   neighbours are v[j, i_u-1], v[j, i_u], v[j+1, i_u-1], v[j+1, i_u]
    # with periodic lon.
    v_west = np.roll(v, 1, axis=1)
    v_at_u_centre = 0.25 * (v[:-1] + v[1:] + v_west[:-1] + v_west[1:])  # (n_lat, n_lon, nlev)
    # extend to n_lon+1 by adding periodic east column
    v_at_u_full = np.concatenate(
        [v_at_u_centre, v_at_u_centre[:, 0:1, :]], axis=1)        # (n_lat, n_lon+1, nlev)
    # vortcor pointwise tendency on u: zeta_u × v_at_u_full  (m/s²)
    vortcor_u = zeta_u * v_at_u_full                              # (n_lat, n_lon+1, nlev)
    # Total force on band wet u-faces, integrated over depth:
    # F = ρ · Σ_face vortcor × dx_u · dy · dz × u_mask
    contrib = RHO_0 * vortcor_u * u_mask[:, :, None] \
              * dx_u[:, None, None] * dy * dz[None, None, :]
    F_vortcor = float(np.sum(contrib[J_DRAKE]))

    # ------- residual ----------------------------------------------------
    residual = F_wind + F_drag_bot + F_drag_baro - F_adv + F_vortcor

    # band area for converting to equivalent Pa
    area_band = float(np.sum((dx_u * n_wet_cell_per_row * dy)[J_DRAKE]))

    return {
        "F_wind":       F_wind,
        "F_drag_bot":   F_drag_bot,
        "F_drag_baro":  F_drag_baro,
        "F_adv":        F_adv,
        "F_vortcor":    F_vortcor,
        "F_visc_diag":  F_visc_diag,  # diagnostic only, NOT in residual
        "F_visc":       F_visc,
        "residual":     residual,
        "area_band":    area_band,
        # band-mean diagnostics
        "U_baro_band": float(np.mean(U_baro_lat[J_DRAKE])),
        "u_bot_band":  float(np.mean(u_bot_lat[J_DRAKE])),
        "drake_T_Sv":  float(np.sum(U_baro_lat[J_DRAKE] * H_local) * dy / 1e6),
    }


def _print_budget(label, b):
    A = b["area_band"]
    def to_Pa(F):
        return F / A
    print(f"\n--- {label} ---")
    print(f"  band area: {A:.3e} m²    U_baro_band = {b['U_baro_band']*100:+.3f} cm/s "
          f"   u_bot_band = {b['u_bot_band']*100:+.3f} cm/s "
          f"   Drake T = {b['drake_T_Sv']:+.1f} Sv")
    print(f"  {'term':<22} {'TN':>10} {'Pa':>10}")
    for k in ["F_wind", "F_drag_bot", "F_drag_baro", "F_adv", "F_vortcor",
              "F_visc_diag", "residual"]:
        print(f"  {k:<22} {b[k]/1e12:+10.3f} {to_Pa(b[k]):+10.4f}")


def _plot_budgets(budgets, out_path):
    labels = list(budgets.keys())
    keys = ["F_wind", "F_drag_bot", "F_drag_baro", "F_adv (-sign)", "F_visc", "residual"]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.13
    x = np.arange(len(labels))
    for i, k in enumerate(keys):
        if k == "F_adv (-sign)":
            vals = [-b["F_adv"] / b["area_band"] for b in budgets.values()]
        else:
            vals = [b[k] / b["area_band"] for b in budgets.values()]
        ax.bar(x + (i - len(keys)/2 + 0.5) * width, vals, width=width, label=k)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Equivalent stress (Pa)")
    ax.set_title("Drake band depth-integrated zonal-momentum budget\n"
                 "F_adv shown with -sign so all terms add to ~residual")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg, grid, z_coord, dz, H_total, lat, lat_v, dx_u, dx_v, dy, R = \
        _build_geometry()

    r_default = cfg.bottom_drag_coeff
    A_h = cfg.A_h
    print(f"Defaults: r = {r_default}, A_h = {A_h}, H = {H_total} m")

    # land/u/v masks shared across runs (geometry doesn't change)
    base_restart = np.load(
        "results/ocean/global_overturning_50yr_gmredi/restart_day018250.npz",
        allow_pickle=False)
    land_mask = np.asarray(base_restart["land_mask"], dtype=np.float64)
    u_mask    = np.asarray(base_restart["u_mask"],    dtype=np.float64)
    v_mask    = np.asarray(base_restart["v_mask"],    dtype=np.float64)

    runs = []

    # ---- 50-yr GM/Redi (no ridge): mean of last 5 restarts (yrs 30-50) ---
    rd = sorted(Path(
        "results/ocean/global_overturning_50yr_gmredi"
    ).glob("restart_day*.npz"))[-5:]
    state, n = _load_time_mean_from_restarts(rd)
    runs.append(("50yr GM/Redi (yrs 30-50 mean)", state, r_default, True))

    # ---- bottom-drag x0.2 -----------------------------------------------
    p = Path("results/ocean/drake_sensitivity_lower_drag_x0p2/time_mean.npz")
    if p.exists():
        s = _load_time_mean_from_npz(p)
        runs.append(("drag x0.2 (5-yr mean)", s, r_default * 0.2, True))

    # ---- ridge mid-Pacific ----------------------------------------------
    p = Path("results/ocean/drake_sensitivity_ridge_2000m_lon30/time_mean.npz")
    # the 2000m run blew up — only the 3000m one is valid
    p3000 = Path("results/ocean/drake_sensitivity_ridge_3000m_lon30/time_mean.npz")
    if p3000.exists():
        s = _load_time_mean_from_npz(p3000)
        runs.append(("ridge 3000m lon30 (5-yr mean)", s, r_default, True))

    # ---- ridge Drake position -------------------------------------------
    p = Path("results/ocean/drake_sensitivity_ridge_3000m_lon8/time_mean.npz")
    if p.exists():
        s = _load_time_mean_from_npz(p)
        runs.append(("ridge 3000m lon8 (5-yr mean)", s, r_default, True))

    budgets = {}
    for label, state, r, has_path3 in runs:
        b = compute_budget(
            state, r=r, A_h=A_h, dz=dz, lat=lat, lat_v=lat_v,
            dx_u=dx_u, dx_v=dx_v, dy=dy,
            land_mask=land_mask, u_mask=u_mask, v_mask=v_mask,
            has_path3_drag=has_path3,
        )
        _print_budget(label, b)
        budgets[label] = b

    out = Path("results/ocean/drake_momentum_budget.png")
    _plot_budgets(budgets, out)


if __name__ == "__main__":
    main()
