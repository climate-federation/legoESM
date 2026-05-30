#!/usr/bin/env python
"""legoESM ocean forced by CORE-II Normal-Year Forcing on the eORCA1 tripole
grid -- the faithful counterpart to the NEMO ORCA1 reference (see
``OMIP_faithful.md``).

Both models use the SAME CORE-II forcing: NEMO reads the raw COREv2 files; here
``load_core2_nyf`` reads ``nyf.zarr`` built from those same files by
``scripts/build_core2_nyf_zarr.py`` (native 6-hourly winds, so the nonlinear
bulk fluxes match). The eORCA1 grid + land mask + bathymetry come from NEMO's
own ``eORCA1.2_mesh_mask.nc`` (tmaskutil / e3t_0), so the geometry matches too.

Loop: ``load_core2_nyf`` -> per step pick the 6-hourly record -> apply CORE-II
bulk fluxes via ``apply_omip2_surface_fluxes`` (ocean-reaction sign, tripole
rotation) -> ``model.step``. Annual snapshots + scalar diagnostics are written
for scoring against the NEMO climatology.

NOTE the applicator is host-side NumPy, so each step round-trips the state
device<->host. ``--smoke`` reports steps/s so the real run length can be sized;
if throughput is too low, apply forcing every N steps (forcing is 6-hourly).

Usage (GPU sbatch):
    JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 python scripts/run_omip_core2.py --smoke
    JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 python scripts/run_omip_core2.py --years 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_SEC_PER_DAY = 86400.0
_SEC_PER_6H = 21600.0
_YEAR_S = 365.0 * _SEC_PER_DAY
_MESH = "data/grids/eORCA1.2_mesh_mask.nc"


def _squeeze2d(a: np.ndarray) -> np.ndarray:
    """Drop leading singleton (t, z) dims from a NEMO mesh field -> 2-D (y, x)."""
    a = np.asarray(a)
    while a.ndim > 2:
        a = a[0]
    return a


def read_mesh_mask_bathy(mesh_path: str):
    """Derive the 2-D ocean land mask and total bathymetric depth from NEMO's
    own eORCA1 mesh_mask -- the most faithful geometry for the comparison.

    Returns
    -------
    land_mask : (n_lat, n_lon) float64, 1 = ocean, 0 = land  (tmaskutil)
    H_bathy   : (n_lat, n_lon) float64, total wet-column depth [m]
                (sum_k e3t_0 * tmask)
    """
    import xarray as xr
    ds = xr.open_dataset(mesh_path)
    # Surface ocean/land mask (1 = ocean). Prefer the 2-D util mask.
    if "tmaskutil" in ds:
        land_mask = _squeeze2d(ds["tmaskutil"].values).astype(np.float64)
    else:
        land_mask = _squeeze2d(ds["tmask"].values).astype(np.float64)
    # Total wet-column depth = sum over z of e3t_0 where tmask is wet.
    e3t = np.asarray(ds["e3t_0"].values)            # (t,z,y,x) or (z,y,x)
    tmask = np.asarray(ds["tmask"].values)
    while e3t.ndim > 3:
        e3t = e3t[0]
    while tmask.ndim > 3:
        tmask = tmask[0]
    H_bathy = (e3t * tmask).sum(axis=0).astype(np.float64)   # (y, x)
    # Guard: dry columns get 0 depth (they are land via land_mask anyway).
    return land_mask, H_bathy


def compute_woa_3d(grid, z_coord, woa_t, woa_s, H_bathy, land_mask):
    """WOA18 T,S interpolated to the model grid (°C / PSU), masked below the
    local seafloor (``bathymetry_depth``) and zeroed on land. Used both as the
    WOA initial condition and as the nudging target. Shared so the two paths
    cannot diverge."""
    from legoesm.ocean.init_woa import init_ocean_from_woa
    from legoesm.ocean.vertical import OceanPartialCellCoordinate
    _is_pc = isinstance(z_coord, OceanPartialCellCoordinate)
    # For a partial-cell coord, do NOT pass bathymetry_depth: init_ocean_from_woa
    # masks levels by ``|z_full_ref| > bathymetry_depth`` (reference full-cell
    # centres), which deep-fills ACTIVE bottom partial cells whose reference
    # centre lies below the snapped bathymetry — corrupting the IC at the exact
    # topographic-step region this coord stabilises (codex adversarial-review).
    # Instead keep interpolated WOA at every reference level and mask only the
    # genuinely-inactive (below-seafloor) cells with ``~z_coord.is_active`` below.
    T_woa, S_woa = init_ocean_from_woa(
        grid, z_coord, woa_t, woa_s,
        bathymetry_depth=(None if _is_pc else np.maximum(np.asarray(H_bathy), 1.0)),
    )
    T_woa = np.array(T_woa, dtype=np.float64)   # writable copy (not a view)
    S_woa = np.array(S_woa, dtype=np.float64)
    m2 = np.asarray(land_mask) > 0.5
    # NEMO's ocean mask (tmaskutil) includes cells WOA has NO data for (WOA-land
    # / outside coverage / marginal seas), where init_ocean_from_woa leaves
    # S~0. Those S=0 cells sit next to real S~35 ocean -> ~40-PSU horizontal
    # jumps -> a spurious O(10 m/s) baroclinic PGF on step 1 -> blowup (root
    # cause of the "WOA cold-start instability"). Flood-fill every such ocean
    # cell (surface S < 1 PSU) from its nearest valid-WOA ocean column.
    bad = m2 & (S_woa[..., 0] < 1.0)
    if bad.any():
        from scipy.spatial import cKDTree
        valid = m2 & ~bad
        jj, ii = np.where(valid)
        jb, ib = np.where(bad)
        _, idx = cKDTree(np.c_[jj, ii]).query(np.c_[jb, ib], k=1)
        T_woa[jb, ib, :] = T_woa[jj[idx], ii[idx], :]
        S_woa[jb, ib, :] = S_woa[jj[idx], ii[idx], :]
        print(f"[setup] flood-filled {int(bad.sum())} NEMO-ocean cells "
              f"lacking WOA data (S<1) from nearest valid column")
    # Re-apply the RECEIVER bathymetry deep-fill (codex C1): a donor column
    # copied above may carry levels below the receiver's own seafloor; replace
    # every level deeper than the local H_bathy with the deep-ocean fill, so the
    # filled columns match init_ocean_from_woa's bathymetry_depth treatment and
    # do not reintroduce a hidden below-seafloor T/S bias. Idempotent for the
    # original (already-filled) columns.
    from legoesm import constants as _const
    T_fill = float(getattr(_const, "T_deep_ocean_ref_C", 1.5))
    S_fill = float(getattr(_const, "S_deep_ocean_ref_psu", 34.7))
    if _is_pc:
        # Partial-cell coord: ``is_active`` already accounts for the active
        # bottom PARTIAL cell. Using the reference full-cell centres
        # (cumsum(dz_ref)) here would mark active bottom partial cells whose
        # thickness is < 50% of dz_ref as below-seafloor and overwrite their
        # real WOA T/S with deep fill — corrupting the IC/nudge target at the
        # exact topographic-step region this coord is meant to stabilise
        # (codex adversarial-review). Use the coord's own active mask.
        below = ~np.asarray(z_coord.is_active)
    else:
        dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
        z_cen = np.cumsum(dz) - 0.5 * dz                  # (nlev,) cell-centre depths
        below = z_cen[None, None, :] > np.asarray(H_bathy)[..., None]
    T_woa = np.where(below, T_fill, T_woa)
    S_woa = np.where(below, S_fill, S_woa)
    m3 = m2[..., None]
    return T_woa * m3, S_woa * m3


def make_partial_cell(z_coord, H_bathy, land_mask, thin_threshold=0.3,
                      smoothing_passes=0):
    """Convert a z* reference coord + bathymetry to an ``OceanPartialCellCoordinate``,
    snapping ``H_bathy`` DOWN to the interface above whenever the bottom partial cell
    would be thinner than ``thin_threshold * dz_ref`` (MOM6/MITgcm thin-cell fix).

    ``smoothing_passes`` > 0 applies that many Laplacian smoothing passes to the
    OCEAN ``H_bathy`` field first (land held fixed), reducing the bathymetric slope
    (r-factor ``|H_i-H_j|/(H_i+H_j)``).  NEMO/ROMS smooth their bathymetry for exactly
    this reason: the spurious partial-cell pressure-gradient seed that blows up the
    WOA cold-start (per-term diag: KE_PGF at the S-Atlantic / Indonesian continental
    SLOPES, the steepest cells) scales with the slope, so gentler topography shrinks
    it.  The max r-factor is reported before/after so the geometry cost is explicit.

    Mirrors the documented-stable ``run_omip.py`` partial-cell setup
    (``run_omip_single``, ~line 3181). The OMIP-faithful runner previously passed the
    plain ``OceanZStarCoordinate`` from ``_create_setup`` straight to the model. With
    that coord, ``J = (eta + H_bathy)/H_max`` uniformly stretches all levels to the
    local depth (terrain-following / sigma-like), AND the Adcroft/SMC03 partial-cell
    PGF correction is gated OFF (``isinstance(z_coord, OceanPartialCellCoordinate)``
    is False in ``ocean_pe_latlon_cgrid``). Over steep equatorial topography (f≈0)
    that drives the spurious bottom meridional-PGF seed that blows up the WOA cold
    start (per-term diag job 8106208: KE_PGF_v ~8.6e-3 m/s2 @ Indonesian seas, bottom
    level). The z-level partial-cell coord (this function) puts levels at FIXED
    reference depths, activates the PGF correction, and applies the thin-cell snap
    that prior work found necessary at the equator.

    Returns ``(z_coord_partial, H_snapped_np, land_mask_np)``.
    """
    from legoesm.ocean.vertical import create_partial_cell_coordinate
    H_np = np.asarray(H_bathy, dtype=np.float64)
    lm0 = np.asarray(land_mask, dtype=np.float64)

    if smoothing_passes and smoothing_passes > 0:
        from legoesm.ocean.bathymetry import _laplacian_smooth_2d, _r_factor_max
        ocean = lm0 > 0.5
        r_before = float(_r_factor_max(H_np, lm0))
        H_s = H_np.copy()
        # Smooth ocean cells only; hold land fixed and re-impose it each
        # pass so the smoother never bleeds land depths into the ocean.
        for _ in range(int(smoothing_passes)):
            H_sm = np.asarray(_laplacian_smooth_2d(H_s, 1, is_cubed=False))
            H_s = np.where(ocean, H_sm, H_np)
        H_np = np.where(ocean, H_s, H_np)
        r_after = float(_r_factor_max(H_np, lm0))
        print(f"[setup] bathymetry smoothing: {smoothing_passes} Laplacian "
              f"passes, max r-factor {r_before:.3f} -> {r_after:.3f}")

    abs_z_half = np.abs(np.asarray(z_coord.z_half_ref))   # (nlev+1,) positive depths
    dz_ref_np = np.asarray(z_coord.dz_ref)                # (nlev,) positive
    H_snapped = H_np.copy()
    n_snapped = 0
    for k in range(z_coord.n_levels):
        top, bot = abs_z_half[k], abs_z_half[k + 1]
        in_layer = (H_np > top) & (H_np <= bot)
        too_thin = in_layer & ((H_np - top) < thin_threshold * dz_ref_np[k])
        H_snapped = np.where(too_thin, top, H_snapped)
        n_snapped += int(np.sum(too_thin))
    lm = np.asarray(land_mask, dtype=np.float64)
    new_land = (H_snapped <= 0.0) & (lm > 0.5)
    n_new_land = int(np.sum(new_land))
    lm_out = np.where(new_land, 0.0, lm)
    print(f"[setup] partial-cell snap (cutoff {thin_threshold*100:.0f}%): "
          f"{n_snapped} cells snapped, {n_new_land} -> land")
    zc = create_partial_cell_coordinate(
        z_coord, jnp.asarray(H_snapped, dtype=jnp.float64),
    )
    return zc, H_snapped, lm_out


def build_tripole(nlev: int, H_max: float, mesh_path: str,
                  woa_init: bool = False, woa_t=None, woa_s=None,
                  pgf_scheme=None, A_h=None, B_h=None, K_bih=None, flat_bottom=False, A_h_eq_boost=None,
                  ke_gradient_scheme=None, partial_cell=False,
                  adaptive_implicit_vertadv=None, bathy_smoothing_passes=0):
    """Build the eORCA1 tripole grid + model + initial state with NEMO's mask/bathy.

    Reuses run_omip's validated tripole setup. ``forcing_mode='jra55_do_tropical'``
    selects the surface-forcing ``scheme='none'`` config so the model applies NO
    internal restoring -- CORE-II forcing is applied externally by the applicator.

    ``woa_init`` initialises T/S from the WOA18 climatology (``init_ocean_from_woa``)
    instead of the idealised rest state -- ESSENTIAL for a faithful comparison, since
    NEMO starts from the Gouretski/WOCE climatology; a rest-state vs climatology IC
    confounds model differences with IC differences over a few-year spinup. (WOA18 is
    a close stand-in for NEMO's exact Gouretski IC, which is the further refinement.)
    """
    import run_omip
    grid, z_coord, config, model, _ = run_omip._create_setup(
        "tripole", "eorca1", nlev, H_max,
        physics_preset="full", water_type="II",
        forcing_mode="jra55_do_tropical",
    )
    # Optional dycore-stability overrides (for WOA cold-start tuning): rebuild
    # the config + model from run_omip's validated tripole base, changing only
    # the requested knobs (e.g. pgf_scheme="smc03", higher A_h/B_h).
    _ovr = {k: v for k, v in (("pgf_scheme", pgf_scheme), ("A_h", A_h),
                              ("B_h", B_h), ("K_bih", K_bih),
                              ("A_h_eq_boost", A_h_eq_boost),
                              ("ke_gradient_scheme", ke_gradient_scheme),
                              ("adaptive_implicit_vertadv", adaptive_implicit_vertadv),
                              ) if v is not None}
    if _ovr:
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        config = config._replace(**_ovr)
        model = LatLonCGridOceanModel(grid, z_coord, config)
        print(f"[setup] tripole config override: {_ovr}")
    land_mask, H_bathy = read_mesh_mask_bathy(mesh_path)
    if flat_bottom:
        H_bathy = np.where(land_mask > 0.5, H_max, 0.0)
        print("[setup] FLAT BOTTOM (topography removed -- PGF-over-topo control)")
    n_lat, n_lon = int(grid.lat_T.shape[0]), int(grid.lat_T.shape[1])
    if land_mask.shape != (n_lat, n_lon):
        raise ValueError(
            f"mesh mask shape {land_mask.shape} != grid {(n_lat, n_lon)}"
        )
    if partial_cell:
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        z_coord, H_bathy, land_mask = make_partial_cell(
            z_coord, H_bathy, land_mask, smoothing_passes=bathy_smoothing_passes)
        model = LatLonCGridOceanModel(grid, z_coord, config)
    state = run_omip._init_rest_state(
        "tripole", grid, z_coord, H_max,
        H_bathy=jnp.asarray(H_bathy), land_mask=jnp.asarray(land_mask),
    )
    if woa_init:
        from legoesm.core.field import Field
        T_woa, S_woa = compute_woa_3d(grid, z_coord, woa_t, woa_s,
                                      H_bathy, land_mask)
        if T_woa.shape != state.T.data.shape:
            raise ValueError(
                f"WOA T shape {T_woa.shape} != state T {state.T.data.shape}"
            )
        state = state._replace(
            T=Field(jnp.asarray(T_woa), name=state.T.name,
                    dims=state.T.dims, units=state.T.units),
            S=Field(jnp.asarray(S_woa), name=state.S.name,
                    dims=state.S.dims, units=state.S.units),
        )
        print(f"[setup] T/S initialised from WOA18 ({Path(woa_t).name})")
    return grid, z_coord, model, state, np.asarray(H_bathy)


def build_latlon_bathy(nlev: int, H_max: float, mesh_path: str,
                       n_lat: int = 180, n_lon: int = 360,
                       woa_init: bool = False, woa_t=None, woa_s=None,
                       pgf_scheme=None, A_h=None, B_h=None, K_bih=None, flat_bottom=False, A_h_eq_boost=None,
                       ke_gradient_scheme=None, partial_cell=False,
                       adaptive_implicit_vertadv=None, bathy_smoothing_passes=0):
    """Build a regular lat-lon C-grid with REALISTIC bathymetry + the run_omip
    production config (smc03 PGF, biharmonic, implicit-CN barotropic, GM/Redi,
    KPP) -- documented to run STABLE 50+ yr with real geometry, unlike the
    tripole (adcroft) path which blows up on a realistic cold-start.

    Bathymetry + land mask are NEMO's OWN eORCA1 fields (tmaskutil / e3t_0)
    regridded to the lat-lon grid (nearest-neighbour, periodic) -- so the
    geometry still matches the NEMO reference.
    """
    import run_omip
    import xarray as xr
    from compare_omip_nemo import regrid_curv_to_latlon
    res = f"{n_lat}x{n_lon}"
    grid, z_coord, config, model, _ = run_omip._create_setup(
        "latlon", res, nlev, H_max, physics_preset="full", water_type="II",
        use_bathymetry=True, pgf_scheme=pgf_scheme,
        A_h_override=A_h, B_h_override=B_h,
    )
    _ovr = {k: v for k, v in (("K_bih", K_bih),
                              ("ke_gradient_scheme", ke_gradient_scheme),
                              ("adaptive_implicit_vertadv", adaptive_implicit_vertadv),
                              ) if v is not None}
    if _ovr:
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        config = config._replace(**_ovr)
        model = LatLonCGridOceanModel(grid, z_coord, config)
        print(f"[setup] latlon config override: {_ovr}")
    e_mask, e_H = read_mesh_mask_bathy(mesh_path)
    ds = xr.open_dataset(mesh_path)
    src_lat = _squeeze2d(ds["gphit"].values)
    src_lon = _squeeze2d(ds["glamt"].values)
    tgt_lat = np.rad2deg(np.asarray(grid.lat))
    tgt_lon = np.rad2deg(np.asarray(grid.lon))
    H_ll, ocean_ll = regrid_curv_to_latlon(
        e_H, src_lat, src_lon, e_mask, tgt_lat, tgt_lon, max_deg=3.0,
    )
    land_mask = (ocean_ll > 0.5).astype(np.float64)
    H_bathy = np.where(land_mask > 0.5, np.maximum(H_ll, 50.0), 0.0)
    if flat_bottom:
        H_bathy = np.where(land_mask > 0.5, H_max, 0.0)
        print("[setup] FLAT BOTTOM (topography removed -- PGF-over-topo control)")
    print(f"[setup] latlon {n_lat}x{n_lon}: ocean cells {int(land_mask.sum())}, "
          f"H_bathy [{H_bathy[land_mask>0.5].min():.0f},{H_bathy.max():.0f}] m")
    if partial_cell:
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        z_coord, H_bathy, land_mask = make_partial_cell(
            z_coord, H_bathy, land_mask, smoothing_passes=bathy_smoothing_passes)
        model = LatLonCGridOceanModel(grid, z_coord, config)
    state = run_omip._init_rest_state(
        "latlon", grid, z_coord, H_max,
        H_bathy=jnp.asarray(H_bathy), land_mask=jnp.asarray(land_mask),
    )
    if woa_init:
        from legoesm.core.field import Field
        T_woa, S_woa = compute_woa_3d(grid, z_coord, woa_t, woa_s,
                                      H_bathy, land_mask)
        state = state._replace(
            T=Field(jnp.asarray(T_woa), name=state.T.name,
                    dims=state.T.dims, units=state.T.units),
            S=Field(jnp.asarray(S_woa), name=state.S.name,
                    dims=state.S.dims, units=state.S.units),
        )
        print(f"[setup] latlon-bathy T/S initialised from WOA18 ({Path(woa_t).name})")
    return grid, z_coord, model, state, np.asarray(H_bathy)


def _idx_t(step: int, dt: float, n_rec: int) -> int:
    """Nearest 6-hourly CORE-II record for the current model time (perpetual yr)."""
    t = (step * dt) % _YEAR_S
    return int(round(t / _SEC_PER_6H)) % n_rec


def _diag(state, lat2d=None) -> dict:
    """Cheap scalar diagnostics over ocean cells (one device->host pull).

    ``umax_lat`` = latitude of the surface max|u| — localises WHERE velocity
    grows/blows up (equator f->0 vs western boundaries vs poles), the key
    pin-point for the baroclinic-dynamics instability.
    """
    T = np.asarray(state.T.data)[..., 0]
    S = np.asarray(state.S.data)[..., 0]
    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    m = np.asarray(state.land_mask.data) > 0.5
    usurf = np.abs(u[..., 0])
    max_speed = float(np.nanmax(usurf)) if u.size else 0.0
    umax_lat = float("nan")
    if lat2d is not None and np.isfinite(usurf).any():
        ju = int(np.unravel_index(np.nanargmax(usurf), usurf.shape)[0])
        lat2d = np.asarray(lat2d)
        umax_lat = round(float(lat2d[min(ju, lat2d.shape[0] - 1), 0]), 1)
    return {
        "mean_sst_C": float(np.nanmean(T[m])) if m.any() else float("nan"),
        "mean_sss": float(np.nanmean(S[m])) if m.any() else float("nan"),
        "max_abs_u": max_speed,
        "max_abs_v": float(np.nanmax(np.abs(v))) if v.size else 0.0,
        "umax_lat": umax_lat,
        "finite": bool(np.isfinite(T).all() and np.isfinite(u).all()),
    }


def _grid_lat2d_deg(grid, grid_type):
    """2-D (lat, lon) in degrees for snapshots/scoring, for either grid type."""
    if grid_type == "tripole":
        return (np.rad2deg(np.asarray(grid.lat_T)),
                np.rad2deg(np.asarray(grid.lon_T)))
    # regular lat-lon: 1-D radian axes -> 2-D degree meshgrid
    lon2d, lat2d = np.meshgrid(np.rad2deg(np.asarray(grid.lon)),
                               np.rad2deg(np.asarray(grid.lat)))
    return lat2d, lon2d


def _save_snapshot(out_dir: Path, tag: str, state, lat2d, lon2d):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"snapshot_{tag}.npz",
        T=np.asarray(state.T.data), S=np.asarray(state.S.data),
        u=np.asarray(state.u.data), v=np.asarray(state.v.data),
        land_mask=np.asarray(state.land_mask.data),
        lat_T=np.asarray(lat2d), lon_T=np.asarray(lon2d),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=float, default=5.0)
    p.add_argument("--smoke", action="store_true",
                   help="Short 10-day benchmark run (reports steps/s).")
    p.add_argument("--dt", type=float, default=3600.0,
                   help="Timestep [s] (default 3600 = NEMO ORCA1).")
    p.add_argument("--nlev", type=int, default=20)
    p.add_argument("--H-max", type=float, default=5500.0)
    p.add_argument("--mesh", type=str, default=_MESH)
    p.add_argument("--grid", type=str, default="tripole",
                   choices=["tripole", "latlon_bathy"],
                   help="tripole (eORCA1, ideal same-grid but cold-start-unstable) "
                        "or latlon_bathy (regular lat-lon + NEMO bathy + smc03; the "
                        "documented-stable production config).")
    p.add_argument("--latlon-res", type=str, default="180x360",
                   help="lat-lon resolution NxM for --grid latlon_bathy.")
    p.add_argument("--woa-init", action="store_true",
                   help="Initialise T/S from WOA18 (faithful IC) vs rest state.")
    p.add_argument("--woa-t", type=str, default="data/woa18/woa18_decav_t00_01.nc")
    p.add_argument("--woa-s", type=str, default="data/woa18/woa18_decav_s00_01.nc")
    p.add_argument("--ke-gradient-scheme", type=str, default=None,
                   choices=["centered", "hollingsworth"],
                   help="KE-gradient discretization for the vector-invariant "
                        "momentum advection. 'hollingsworth' (NEMO nkeg_HW) is "
                        "consistent with the AL81 PV-flux Coriolis term; "
                        "'centered' is the legacy scheme. Default (None) PRESERVES "
                        "the config default, currently 'centered' for BOTH tripole "
                        "and latlon_bathy -- no production default is changed. This "
                        "is a diagnostic A/B knob: job 8106193 showed hollingsworth "
                        "does NOT fix the WOA cold-start blowup (both schemes go "
                        "non-finite by day 0.5 on both grids). For tripole the "
                        "hollingsworth KE stencil also still lacks a fold-aware "
                        "north halo (see run_omip.py tripole config note).")
    p.add_argument("--partial-cell", action="store_true",
                   help="Use OceanPartialCellCoordinate (z-level + partial bottom "
                        "steps, NEMO-faithful) with thin-cell snapping, instead of "
                        "the plain sigma-like z* coord from _create_setup. Activates "
                        "the Adcroft/SMC03 partial-cell PGF correction (gated off for "
                        "plain z*) and matches NEMO's vertical coordinate. Fixes the "
                        "spurious equatorial-bottom PGF cold-start blowup (job 8106208).")
    p.add_argument("--pgf-scheme", type=str, default=None, choices=[None, "adcroft", "smc03"],
                   help="Override tripole PGF scheme (default: run_omip's adcroft).")
    p.add_argument("--A-h", type=float, default=None, help="Override Laplacian viscosity [m2/s].")
    p.add_argument("--B-h", type=float, default=None, help="Override biharmonic viscosity [m4/s].")
    p.add_argument("--K-bih", type=float, default=None,
                   help="Biharmonic tracer hyperdiffusion [m4/s] -- scale-selectively "
                        "damp a grid-scale baroclinic T/S mode (preserves large-scale gradients).")
    p.add_argument("--flat-bottom", action="store_true",
                   help="Replace bathymetry with a flat bottom (H_max) over ocean cells "
                        "-- controlled test isolating the PGF-over-topography error.")
    p.add_argument("--A-h-eq-boost", type=float, default=None,
                   help="Equatorial Laplacian-viscosity boost factor -- damps the f->0 "
                        "velocity growth (A_h *= 1+(boost-1)*exp(-(lat/sigma)^2)).")
    p.add_argument("--adaptive-implicit-vertadv", action="store_true",
                   help="Enable adaptive-implicit vertical momentum advection "
                        "(Shchepetkin 2015 / NEMO ln_zad_Aimp) -- removes the vertical-CFL "
                        "limit so the spurious-w 'vertadv' runaway cannot amplify. The "
                        "NEMO-faithful fix for the OMIP cold-start blowup (eORCA OMIP "
                        "production runs set ln_zad_Aimp=.true.).")
    p.add_argument("--bathy-smoothing-passes", type=int, default=0,
                   help="Laplacian smoothing passes on the OCEAN bathymetry before "
                        "building the partial-cell coord -- reduces the bathymetric "
                        "slope (r-factor) and hence the spurious partial-cell PGF seed "
                        "at steep continental slopes (NEMO/ROMS smooth for this). "
                        "Requires --partial-cell. 0=off.")
    p.add_argument("--output", type=str, default="results/omip_nemo/legoesm_tripole")
    p.add_argument("--diag-every-days", type=float, default=30.0)
    p.add_argument("--forcing-ramp-days", type=float, default=0.0,
                   help="Ramp the surface forcing 0->full over N days "
                        "(cold-start shock mitigation).")
    p.add_argument("--nudge-woa-tau-days", type=float, default=0.0,
                   help="Nudge T,S toward WOA with this timescale [days] from a "
                        "rest start -- gradual cold-start spinup that avoids the "
                        "WOA-IC geostrophic-imbalance blowup. 0=off.")
    p.add_argument("--nudge-release-day", type=float, default=0.0,
                   help="Stop nudging after this model day (0=nudge throughout).")
    p.add_argument("--spinup-drag-tau-days", type=float, default=0.0,
                   help="Rayleigh velocity-damping timescale [days] during the "
                        "spin-up phase -- bleeds off the cold-start geostrophic "
                        "adjustment OVERSHOOT (the ~5-10 m/s transients that trip "
                        "the nonlinear advective blowup) while the stratification "
                        "settles. 0=off.")
    p.add_argument("--spinup-drag-days", type=float, default=0.0,
                   help="Duration [days] of the spin-up velocity-damping phase "
                        "(drag removed afterwards -> free run).")
    args = p.parse_args()

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    from legoesm.ocean.forcing import load_core2_nyf
    from legoesm.ocean.coupler import compute_omip2_surface_forcing
    from legoesm.core.field import Field

    print(f"[setup] building {args.grid} (nlev={args.nlev}, "
          f"woa_init={args.woa_init}) ...")
    if args.grid == "tripole":
        grid, z_coord, model, state, H_bathy = build_tripole(
            args.nlev, args.H_max, args.mesh,
            woa_init=args.woa_init, woa_t=args.woa_t, woa_s=args.woa_s,
            pgf_scheme=args.pgf_scheme, A_h=args.A_h, B_h=args.B_h, K_bih=args.K_bih,
            flat_bottom=args.flat_bottom, A_h_eq_boost=args.A_h_eq_boost,
            ke_gradient_scheme=args.ke_gradient_scheme,
            partial_cell=args.partial_cell,
            adaptive_implicit_vertadv=(True if args.adaptive_implicit_vertadv else None),
            bathy_smoothing_passes=args.bathy_smoothing_passes,
        )
        app_grid_type = "tripole"
    else:
        _nlat, _nlon = (int(x) for x in args.latlon_res.split("x"))
        grid, z_coord, model, state, H_bathy = build_latlon_bathy(
            args.nlev, args.H_max, args.mesh, n_lat=_nlat, n_lon=_nlon,
            woa_init=args.woa_init, woa_t=args.woa_t, woa_s=args.woa_s,
            pgf_scheme=args.pgf_scheme, A_h=args.A_h, B_h=args.B_h, K_bih=args.K_bih,
            flat_bottom=args.flat_bottom, A_h_eq_boost=args.A_h_eq_boost,
            ke_gradient_scheme=args.ke_gradient_scheme,
            partial_cell=args.partial_cell,
            adaptive_implicit_vertadv=(True if args.adaptive_implicit_vertadv else None),
            bathy_smoothing_passes=args.bathy_smoothing_passes,
        )
        app_grid_type = "latlon"
    lat2d, lon2d = _grid_lat2d_deg(grid, args.grid)
    forcing = load_core2_nyf()
    n_rec = int(forcing.u10.shape[0])
    print(f"[setup] grid {lat2d.shape}, forcing records {n_rec}, dt={args.dt}s")

    dt = float(args.dt)
    ramp_s = float(args.forcing_ramp_days) * _SEC_PER_DAY
    total_days = 10.0 if args.smoke else args.years * 365.0
    n_steps = int(total_days * _SEC_PER_DAY / dt)
    diag_every = max(1, int(args.diag_every_days * _SEC_PER_DAY / dt))
    steps_per_year = int(365.0 * _SEC_PER_DAY / dt)
    out_dir = Path(args.output)

    nudge_tau_s = float(args.nudge_woa_tau_days) * _SEC_PER_DAY
    nudge_release_s = float(args.nudge_release_day) * _SEC_PER_DAY
    drag_tau_s = float(args.spinup_drag_tau_days) * _SEC_PER_DAY
    drag_days_s = float(args.spinup_drag_days) * _SEC_PER_DAY
    if drag_tau_s > 0:
        print(f"[setup] spin-up velocity drag: tau={args.spinup_drag_tau_days}d "
              f"for first {args.spinup_drag_days}d")
    nudge_T = nudge_S = nudge_m3 = None
    if nudge_tau_s > 0:
        nudge_T, nudge_S = compute_woa_3d(
            grid, z_coord, args.woa_t, args.woa_s,
            H_bathy, np.asarray(state.land_mask.data),
        )
        nudge_m3 = np.asarray(state.land_mask.data)[..., None]
        print(f"[setup] nudging T,S -> WOA: tau={args.nudge_woa_tau_days}d, "
              f"release day={args.nudge_release_day or 'never'}")

    print(f"[run] {total_days:.0f} days = {n_steps} steps "
          f"(diag every {diag_every} steps)")
    d0 = _diag(state, lat2d)
    print(f"[diag] step 0: {d0}")

    t_wall = time.time()
    for step in range(1, n_steps + 1):
        it = _idx_t(step, dt, n_rec)
        ramp = min(1.0, (step * dt) / ramp_s) if ramp_s > 0 else 1.0
        # Build CORE-II surface forcing and integrate it INSIDE model.step (the
        # dynamics-core external-tau block) -- energetically consistent, unlike
        # the operator-split applicator (which pumped the runaway). Optional
        # cold-start ramp scales the forcing fields.
        sf = compute_omip2_surface_forcing(
            state, forcing=forcing, idx_t=it,
            grid=grid, grid_type=app_grid_type,
        )
        if ramp < 1.0:
            sf = sf._replace(tau_x=sf.tau_x * ramp, tau_y=sf.tau_y * ramp,
                             q_net=sf.q_net * ramp, sw_down=sf.sw_down * ramp)
        state = model.step(state, dt, surface_forcing=sf)
        if nudge_tau_s > 0 and (nudge_release_s <= 0 or step * dt < nudge_release_s):
            a = dt / nudge_tau_s
            Tn = np.asarray(state.T.data)
            Sn = np.asarray(state.S.data)
            Tn = Tn + a * (nudge_T - Tn) * nudge_m3
            Sn = Sn + a * (nudge_S - Sn) * nudge_m3
            state = state._replace(
                T=Field(jnp.asarray(Tn), name=state.T.name,
                        dims=state.T.dims, units=state.T.units),
                S=Field(jnp.asarray(Sn), name=state.S.name,
                        dims=state.S.dims, units=state.S.units),
            )
        if drag_tau_s > 0 and step * dt < drag_days_s:
            df = float(np.exp(-dt / drag_tau_s))   # Rayleigh decay factor
            state = state._replace(
                u=Field(jnp.asarray(np.asarray(state.u.data) * df),
                        name=state.u.name, dims=state.u.dims, units=state.u.units),
                v=Field(jnp.asarray(np.asarray(state.v.data) * df),
                        name=state.v.name, dims=state.v.dims, units=state.v.units),
            )
        if step % diag_every == 0 or step == n_steps:
            state = jax.block_until_ready(state)
            d = _diag(state, lat2d)
            rate = step / (time.time() - t_wall)
            print(f"[diag] step {step} (day {step*dt/_SEC_PER_DAY:.0f}): "
                  f"{d} | {rate:.2f} steps/s")
            if not d["finite"]:
                print("[ABORT] non-finite state"); _save_snapshot(out_dir, f"blowup_step{step}", state, lat2d, lon2d)
                return 1
        if not args.smoke and steps_per_year > 0 and step % steps_per_year == 0:
            yr = step // steps_per_year
            _save_snapshot(out_dir, f"year{yr:03d}", state, lat2d, lon2d)
            print(f"[snapshot] year {yr} saved")

    state = jax.block_until_ready(state)
    _save_snapshot(out_dir, "final", state, lat2d, lon2d)
    rate = n_steps / (time.time() - t_wall)
    print(f"[done] {n_steps} steps @ {rate:.2f} steps/s; final: {_diag(state, lat2d)}")
    if args.smoke:
        yr_est = steps_per_year / rate / 3600.0
        print(f"[smoke] projected wall-time: {yr_est:.2f} h/yr  "
              f"({args.years:.0f}yr -> {yr_est*args.years:.1f} h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
