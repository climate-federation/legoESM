#!/usr/bin/env python
"""legoESM ocean forced by CORE-II Normal-Year Forcing on the eORCA1 tripole
grid -- the faithful counterpart to the NEMO ORCA1 reference (see
``OMIP_faithful.md``).

Both models use the SAME CORE-II forcing: NEMO reads the raw COREv2 files; here
``load_core2_nyf`` reads ``nyf.zarr`` built from those same files by
``scripts/data/build_core2_nyf_zarr.py`` (native 6-hourly winds, so the nonlinear
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
_ROOT = str(Path(__file__).resolve().parents[2])
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
        # Multi-donor INVERSE-DISTANCE blend (k=8) instead of a wholesale
        # nearest-COLUMN copy.  A single-donor copy leaves a 1-cell T/S step
        # at EVERY depth (incl. the deep k15) along the flood-fill seam; from
        # rest that step is a spurious baroclinic-PGF seed that vertadv pumps
        # to the surface -> the Brazil-Malvinas-region cold-start runaway
        # (mechanism workflow wshsnjjm3).  Blending the 8 nearest valid
        # columns by inverse distance smooths the seam.
        nlev = T_woa.shape[-1]
        if land_mask.ndim == 2:
            # 2-D structured grid (latlon/tripole): index-distance kNN (UNCHANGED
            # — keeps the validated tripole/latlon IC bit-for-bit).
            jj, ii = np.where(valid)
            jb, ib = np.where(bad)
            kdt = cKDTree(np.c_[jj, ii])
            K = int(min(8, len(jj)))
            dist, idx = kdt.query(np.c_[jb, ib], k=K)
            if K == 1:
                dist = dist[:, None]; idx = idx[:, None]
            w = 1.0 / np.maximum(dist, 1e-6)          # (n_bad, K)
            w = w / w.sum(axis=1, keepdims=True)
            T_don = T_woa[jj[idx], ii[idx], :]        # (n_bad, K, nlev)
            S_don = S_woa[jj[idx], ii[idx], :]
            T_woa[jb, ib, :] = np.einsum("nk,nkl->nl", w, T_don)
            S_woa[jb, ib, :] = np.einsum("nk,nkl->nl", w, S_don)
        else:
            # N-D horizontal layout (cube (6,n,n)): index distance is meaningless
            # across faces, so use GREAT-CIRCLE distance on the grid's own
            # lat/lon (radians) between valid and bad columns. Flatten the
            # horizontal dims; donors are the nearest valid columns by chord.
            horiz = bad.shape
            Tf = T_woa.reshape(-1, nlev); Sf = S_woa.reshape(-1, nlev)
            vflat = valid.ravel(); bflat = bad.ravel()
            latr = np.asarray(grid.lat).ravel()
            lonr = np.asarray(grid.lon).ravel()
            cl = np.cos(latr)
            xyz = np.stack([cl * np.cos(lonr), cl * np.sin(lonr),
                            np.sin(latr)], axis=-1)
            vidx = np.where(vflat)[0]                  # global flat idx of valids
            kdt = cKDTree(xyz[vidx])
            K = int(min(8, vidx.size))
            dist, idx = kdt.query(xyz[bflat], k=K)
            if K == 1:
                dist = dist[:, None]; idx = idx[:, None]
            w = 1.0 / np.maximum(dist, 1e-9)
            w = w / w.sum(axis=1, keepdims=True)
            T_don = Tf[vidx[idx], :]                   # (n_bad, K, nlev)
            S_don = Sf[vidx[idx], :]
            Tf[bflat] = np.einsum("nk,nkl->nl", w, T_don)
            Sf[bflat] = np.einsum("nk,nkl->nl", w, S_don)
            T_woa = Tf.reshape(*horiz, nlev)
            S_woa = Sf.reshape(*horiz, nlev)
        print(f"[setup] flood-filled {int(bad.sum())} NEMO-ocean cells "
              f"lacking WOA data (S<1) via inverse-distance blend of {K} "
              f"nearest valid columns (smooths the seam)")
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


def smooth_woa_ts(state, grid, passes):
    """Horizontal Laplacian smoothing of the WOA T,S initial condition over
    OCEAN cells, per level.  Removes the spurious grid-scale / over-sharp
    fronts that interpolating + flood-filling WOA onto the tripole introduces
    -- those imply unphysically large geostrophic velocities (>>2 m/s) and the
    cold-start cannot carry them.  The dynamics ARE stable on a smooth
    stratification (uniform-strat rest test), so smoothing the IC toward that
    regime is the natural conditioning."""
    from legoesm.ocean.bathymetry import _laplacian_smooth_2d
    mask = np.asarray(state.land_mask.data) > 0.5
    # Cube horizontal fields are (6, n, n) -> _laplacian_smooth_2d needs the
    # cube-topology stencil (cross-face neighbours); 2-D grids are (n_lat, n_lon).
    is_cubed = (mask.ndim == 3)
    T = np.array(state.T.data, dtype=np.float64)
    S = np.array(state.S.data, dtype=np.float64)
    nlev = T.shape[-1]
    for arr in (T, S):
        for k in range(nlev):
            orig_k = arr[..., k].copy()
            cur = arr[..., k]
            for _ in range(int(passes)):
                sm = np.asarray(_laplacian_smooth_2d(cur, 1, is_cubed=is_cubed))
                cur = np.where(mask, sm, orig_k)
            arr[..., k] = cur
    print(f"[setup] WOA T,S horizontal smoothing: {passes} Laplacian passes/level "
          f"(removes spurious grid-scale fronts)")
    return state._replace(
        T=state.T.replace(data=jnp.asarray(T)),
        S=state.S.replace(data=jnp.asarray(S)),
    )


def apply_balanced_init(state, grid, z_coord, config,
                        taper_lat_deg=8.0, ref_depth_m=1500.0,
                        max_speed=2.5, with_ssh=True):
    """Initialise the cold-start in geostrophic / thermal-wind balance.

    The OMIP WOA cold-start blows up because it starts from REST (u=0) with a
    flat free surface (eta=0): the full baroclinic pressure-gradient force from
    WOA's density fronts is then UNBALANCED, and the violent geostrophic
    adjustment goes nonlinear (conclusively diagnosed -- the partial-cell PGF
    itself is NEMO-class, ~1e-6 m/s2 on a uniform-stratification rest test).

    This puts the flow in balance at t=0 so there is no adjustment shock:

    1. Baroclinic pressure anomaly ``p'`` from WOA T,S (surface-referenced),
       via the SAME ``iterate_eos_and_pressure_anomaly`` the dycore uses.
    2. LEVEL-OF-NO-MOTION reference: subtract the deepest-active ``p'_bottom``
       so the total pressure is flat at the seafloor (deep flow -> 0,
       surface-intensified ~1 m/s -- physical).  ``p_ref = p' - p'_bottom``.
    3. Geostrophic velocity from ``p_ref`` at cell centres,
       ``u_g = -(1/rho_0 f) dp_ref/dy``, ``v_g = +(1/rho_0 f) dp_ref/dx``,
       with the Coriolis singularity regularised near the equator
       ``1/f -> f/(f^2 + f_eps^2)`` (``f_eps = 2 Omega sin(taper_lat)`` -> the
       geostrophic velocity tapers smoothly to zero within ~|lat|<taper_lat).
       Mapped to the C-grid faces with ``cell_to_cgrid_winds`` (fold-aware).
    4. (with_ssh) Balanced free surface ``eta = -p'_bottom/(rho_0 g)`` (area-
       demeaned), so the dycore's total PGF ``-(1/rho_0) grad(p' + rho_0 g eta)
       = -(1/rho_0) grad(p_ref)`` exactly balances the geostrophic velocity.

    Pure IC change -- no dynamics-core modification.
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        gradient_x_cgrid, gradient_y_cgrid, cell_to_cgrid_winds,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _neumann_fill_cgrid
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        iterate_eos_and_pressure_anomaly,
    )
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.vertical import OceanPartialCellCoordinate
    from legoesm import constants

    T = state.T.data
    S = state.S.data
    mask = state.land_mask.data
    rho_0 = float(config.rho_0)
    g_val = float(config.g)
    is_pc = isinstance(z_coord, OceanPartialCellCoordinate)
    eos_fn = make_eos_fn(config.eos, getattr(config, "eos_linear", None))
    h_actual = z_coord.h_partial if is_pc else None

    _, _, p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask,
        lambda fld: _neumann_fill_cgrid(fld, mask, grid=grid),
        eos_fn, z_coord.dz_ref, rho_0, g_val,
        n_iter=2, hi_precision_pressure=True, h_actual=h_actual,
    )                                                    # (n_lat, n_lon, nlev)

    # Reference level for the level-of-no-motion: a FIXED depth (~ref_depth_m)
    # common to all sufficiently-deep columns -- NOT the per-column seafloor
    # (whose depth varies with bathymetry, so a seafloor reference makes p_ref
    # and eta scale with DEPTH instead of the dynamic steric signal -> O(100 m)
    # spurious SSH).  Shallow columns reference their own bottom level.
    z_full = np.asarray(z_coord.z_full_ref)                 # (nlev,), negative
    k_ref = int(np.argmin(np.abs(z_full + ref_depth_m)))    # level nearest ref_depth
    if is_pc:
        bl = jnp.clip(z_coord.bottom_level, 0, z_coord.n_levels - 1)
        k_use = jnp.minimum(bl, k_ref)                      # (n_lat, n_lon)
    else:
        k_use = jnp.full(p_prime.shape[:-1], k_ref, dtype=jnp.int32)
    p_at_ref = jnp.take_along_axis(p_prime, k_use[..., None], axis=-1)  # (...,1)
    p_ref = p_prime - p_at_ref                              # 0 at the reference

    p_ref_filled = _neumann_fill_cgrid(p_ref, mask, grid=grid)
    gx_u = gradient_x_cgrid(p_ref_filled, grid)            # (n_lat, n_lon+1, nlev)
    gy_v = gradient_y_cgrid(p_ref_filled, grid)            # (n_lat+1, n_lon, nlev)
    gx_T = 0.5 * (gx_u[:, :-1] + gx_u[:, 1:])              # (n_lat, n_lon, nlev)
    gy_T = 0.5 * (gy_v[:-1] + gy_v[1:])

    # grid-agnostic Coriolis: tripole LatLonCGridGeometry -> f_T, plain
    # LatLonGrid -> f; both expose the grid_coriolis @property -> (n_lat, n_lon).
    f_T = grid.grid_coriolis                                # (n_lat, n_lon)
    f_eps = 2.0 * constants.Omega * float(np.sin(np.deg2rad(taper_lat_deg)))
    inv_f = (f_T / (f_T ** 2 + f_eps ** 2))[..., None]     # -> 0 at the equator

    u_g = -(1.0 / rho_0) * inv_f * gy_T
    v_g = +(1.0 / rho_0) * inv_f * gx_T
    # Safety clip: geostrophy is invalid in the (tapered) equatorial band and
    # at any residual sharp IC front (e.g. flood-fill seams); bound the speed
    # to a physical maximum so those cells start bounded rather than at
    # tens of m/s.  Mid-latitude balanced flow is well below this.
    u_g = jnp.clip(u_g, -max_speed, max_speed)
    v_g = jnp.clip(v_g, -max_speed, max_speed)
    m3 = mask[..., None]
    if is_pc:
        m3 = m3 * z_coord.is_active.astype(m3.dtype)
    u_g = u_g * m3
    v_g = v_g * m3

    u_face, v_face = cell_to_cgrid_winds(u_g, v_g, grid)
    u_face = u_face * state.u_mask.data[..., None]
    u_face = u_face.at[:, -1].set(u_face[:, 0])            # periodic wrap column
    v_face = v_face * state.v_mask.data[..., None]

    repl = dict(
        u=state.u.replace(data=u_face),
        v=state.v.replace(data=v_face),
    )
    umax = float(jnp.nanmax(jnp.abs(u_face)))
    vmax = float(jnp.nanmax(jnp.abs(v_face)))
    if with_ssh:
        eta = -p_at_ref[..., 0] / (rho_0 * g_val)          # (n_lat, n_lon)
        area = grid.area * mask   # grid-agnostic (both grids expose .area)
        eta_mean = jnp.sum(eta * area) / jnp.maximum(jnp.sum(area), 1.0)
        eta = jnp.clip(eta - eta_mean, -5.0, 5.0) * mask   # physical SSH bound
        repl["eta"] = state.eta.replace(data=eta)
        print(f"[setup] balanced init: ref level k={k_ref} (~{ref_depth_m:.0f} m), "
              f"geostrophic u,v (taper {taper_lat_deg} deg, clip {max_speed} m/s) "
              f"max|u|={umax:.3f} max|v|={vmax:.3f} m/s + SSH "
              f"eta[{float(jnp.min(eta)):.2f},{float(jnp.max(eta)):.2f}] m")
    else:
        print(f"[setup] balanced init: ref level k={k_ref} (~{ref_depth_m:.0f} m), "
              f"geostrophic u,v (taper {taper_lat_deg} deg, clip {max_speed} m/s) "
              f"max|u|={umax:.3f} max|v|={vmax:.3f} m/s (NO balanced SSH)")
    return state._replace(**repl)


def make_partial_cell(z_coord, H_bathy, land_mask, thin_threshold=0.3,
                      smoothing_passes=0, min_levels=1):
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
    # rn_hmin-style conditioning: mask ocean columns with fewer than
    # ``min_levels`` active reference levels. Single-active-level coastal cells
    # (H <= dz_ref[0]) are a 1/h instability seed at 1/4 deg, where resolved
    # Arctic/coastal shelves collapse to one thin partial layer and a tiny
    # smc03 PGF residual is amplified into a blowup (eORCA025 cold-start seed at
    # 67N/107.5W). ``abs_z_half[k]`` is the top of level k, so n_active =
    # #levels whose top lies above the local seafloor.
    n_masked_shallow = 0
    if min_levels and int(min_levels) > 1:
        n_active = (abs_z_half[None, None, :z_coord.n_levels]
                    < H_snapped[..., None]).sum(axis=2)
        too_shallow = (n_active < int(min_levels)) & (lm > 0.5) & (H_snapped > 0.0)
        n_masked_shallow = int(np.sum(too_shallow))
        H_snapped = np.where(too_shallow, 0.0, H_snapped)
        new_land = new_land | too_shallow
    n_new_land = int(np.sum(new_land))
    lm_out = np.where(new_land, 0.0, lm)
    msg = (f"[setup] partial-cell snap (cutoff {thin_threshold*100:.0f}%): "
           f"{n_snapped} cells snapped, {n_new_land} -> land")
    if min_levels and int(min_levels) > 1:
        msg += f" ({n_masked_shallow} masked for <{int(min_levels)} active levels)"
    print(msg)
    zc = create_partial_cell_coordinate(
        z_coord, jnp.asarray(H_snapped, dtype=jnp.float64),
    )
    return zc, H_snapped, lm_out


def build_tripole(nlev: int, H_max: float, mesh_path: str,
                  woa_init: bool = False, woa_t=None, woa_s=None,
                  pgf_scheme=None, A_h=None, B_h=None, K_bih=None, flat_bottom=False, A_h_eq_boost=None,
                  ke_gradient_scheme=None, partial_cell=False,
                  adaptive_implicit_vertadv=None, bathy_smoothing_passes=0,
                  momentum_time_integrator=None, barotropic_solver=None,
                  barotropic_diffusion_alpha=None, n_barotropic_substeps=None,
                  barotropic_time_filter=None, bottom_drag_r=None,
                  C_smag=None, C_leith=None, C_smag_lap=None,
                  momentum_advection=None, slope_foot_alpha=None,
                  slope_foot_n_levels=None, slope_foot_threshold=None,
                  min_levels=1, div_damp_2=None, div_damp_4=None,
                  smag_cfl_safety=None, convection="none",
                  convection_K_conv=1.0, convection_K_bg=1e-5,
                  freeze_floor=None):
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
    from scripts.run import run_omip
    # Pick the tripole resolution from the mesh file: eORCA025 (1/4 deg) vs the
    # default eORCA1 (1 deg). create_tripole_grid reads the grid (glamt/e1t.../
    # tmask + fold) from this SAME file, so the grid and the land_mask/bathy
    # (read below) provably come from one mesh -- assert it to kill any drift.
    resolution = "eorca025" if "025" in Path(mesh_path).name else "eorca1"
    _grid_mesh = run_omip._parse_resolution("tripole", resolution)["mesh_path"]
    if Path(_grid_mesh).resolve() != Path(mesh_path).resolve():
        raise ValueError(
            f"tripole mesh mismatch: grid built from {_grid_mesh!r} but "
            f"mask/bathy read from {mesh_path!r}. Pass --mesh {_grid_mesh}."
        )
    grid, z_coord, config, model, _ = run_omip._create_setup(
        "tripole", resolution, nlev, H_max,
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
                              ("momentum_time_integrator", momentum_time_integrator),
                              ("barotropic_solver", barotropic_solver),
                              ("barotropic_diffusion_alpha", barotropic_diffusion_alpha),
                              ("n_barotropic_substeps", n_barotropic_substeps),
                              ("barotropic_time_filter", barotropic_time_filter),
                              ("bottom_drag_r", bottom_drag_r),
                              ("C_smag", C_smag), ("C_leith", C_leith),
                              ("C_smag_lap", C_smag_lap),
                              ("momentum_advection", momentum_advection),
                              ("slope_foot_alpha", slope_foot_alpha),
                              ("slope_foot_n_levels", slope_foot_n_levels),
                              ("slope_foot_threshold", slope_foot_threshold),
                              ("div_damp_2", div_damp_2),
                              ("div_damp_4", div_damp_4),
                              ("smag_cfl_safety", smag_cfl_safety),
                              ("freeze_floor", freeze_floor),
                              ) if v is not None}
    # Grid-agnostic convective adjustment (Oceananigans-style enhanced
    # vertical diffusivity where N^2 < 0).  The tripole base config ships
    # physics=None; opting in attaches an OceanPhysicsConfig whose
    # convective K flows through the SAME grid-agnostic
    # compute_vertical_K_profiles -> implicit backward-Euler vertical solve
    # the cubed-sphere / lat-lon-bathy paths already use.  Default "none"
    # leaves the validated faithful config untouched.
    if convection and convection != "none":
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.convection.config import (
            OceanConvectionConfig, EnhancedDiffusionConfig,
        )
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig,
        )
        from legoesm.ocean.physics.lateral_mixing.config import (
            LateralMixingConfig,
        )
        from legoesm.ocean.physics.surface_forcing.config import (
            SurfaceForcingConfig,
        )
        from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
        # CONVECTION-ONLY physics pipeline.  Every other module is
        # explicitly disabled: the OceanPhysicsConfig defaults are NOT
        # inert (lateral_mixing defaults to harmonic -- assumes a
        # cubed-sphere 4-D layout and would crash on the tripole 3-D
        # state; shortwave_penetration defaults ON -- would double-count
        # the shortwave that the dynamics-core external-tau block already
        # applies).  The C-grid model's own config-level A_h/B_h/K_h,
        # bottom_drag_r and the external CORE-II forcing are untouched;
        # the pipeline contributes ONLY the convective K.
        _ovr["physics"] = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="none"),
            lateral_mixing=LateralMixingConfig(scheme="none"),
            surface_forcing=SurfaceForcingConfig(scheme="none"),
            bottom_drag=BottomDragConfig(scheme="none"),
            convection=OceanConvectionConfig(
                scheme=convection,
                enhanced_diffusion=EnhancedDiffusionConfig(
                    K_conv=convection_K_conv, K_bg=convection_K_bg,
                ),
            ),
            shortwave_penetration=None,
        )
        # Convective adjustment must apply through the implicit vertical
        # solve (backward-Euler is unconditionally stable; an explicit
        # K_conv would violate CFL at ocean dt).
        _ovr["implicit_vertical_mixing"] = True
        print(f"[setup] tripole convection ENABLED (convection-only physics): "
              f"scheme={convection} K_conv={convection_K_conv} "
              f"K_bg={convection_K_bg}")
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
            z_coord, H_bathy, land_mask, smoothing_passes=bathy_smoothing_passes,
            min_levels=min_levels)
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
                       adaptive_implicit_vertadv=None, bathy_smoothing_passes=0,
                  momentum_time_integrator=None, barotropic_solver=None,
                  barotropic_diffusion_alpha=None, n_barotropic_substeps=None,
                  barotropic_time_filter=None, bottom_drag_r=None,
                  C_smag=None, C_leith=None, C_smag_lap=None,
                  momentum_advection=None, slope_foot_alpha=None,
                  slope_foot_n_levels=None, slope_foot_threshold=None,
                  min_levels=1, div_damp_2=None, div_damp_4=None,
                  smag_cfl_safety=None, freeze_floor=None,
                  use_polar_filter=None, polar_filter_cutoff_lat_deg=None,
                  polar_filter_max_wave_speed=None,
                  polar_filter_safety_factor=None):
    """Build a regular lat-lon C-grid with REALISTIC bathymetry + the run_omip
    production config (smc03 PGF, biharmonic, implicit-CN barotropic, GM/Redi,
    KPP) -- documented to run STABLE 50+ yr with real geometry, unlike the
    tripole (adcroft) path which blows up on a realistic cold-start.

    Bathymetry + land mask are NEMO's OWN eORCA1 fields (tmaskutil / e3t_0)
    regridded to the lat-lon grid (nearest-neighbour, periodic) -- so the
    geometry still matches the NEMO reference.
    """
    from scripts.run import run_omip
    import xarray as xr
    from scripts.validate.compare_omip_nemo import regrid_curv_to_latlon
    res = f"{n_lat}x{n_lon}"
    grid, z_coord, config, model, _ = run_omip._create_setup(
        "latlon", res, nlev, H_max, physics_preset="full", water_type="II",
        use_bathymetry=True, pgf_scheme=pgf_scheme,
        A_h_override=A_h, B_h_override=B_h,
    )
    _ovr = {k: v for k, v in (("K_bih", K_bih),
                              ("ke_gradient_scheme", ke_gradient_scheme),
                              ("adaptive_implicit_vertadv", adaptive_implicit_vertadv),
                              ("momentum_time_integrator", momentum_time_integrator),
                              ("barotropic_solver", barotropic_solver),
                              ("barotropic_diffusion_alpha", barotropic_diffusion_alpha),
                              ("n_barotropic_substeps", n_barotropic_substeps),
                              ("barotropic_time_filter", barotropic_time_filter),
                              ("bottom_drag_r", bottom_drag_r),
                              ("C_smag", C_smag), ("C_leith", C_leith),
                              ("C_smag_lap", C_smag_lap),
                              ("momentum_advection", momentum_advection),
                              ("slope_foot_alpha", slope_foot_alpha),
                              ("slope_foot_n_levels", slope_foot_n_levels),
                              ("slope_foot_threshold", slope_foot_threshold),
                              ("div_damp_2", div_damp_2),
                              ("div_damp_4", div_damp_4),
                              ("smag_cfl_safety", smag_cfl_safety),
                              ("freeze_floor", freeze_floor),
                              ("use_polar_filter", use_polar_filter),
                              ("polar_filter_cutoff_lat_deg",
                               polar_filter_cutoff_lat_deg),
                              ("polar_filter_max_wave_speed",
                               polar_filter_max_wave_speed),
                              ("polar_filter_safety_factor",
                               polar_filter_safety_factor),
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
            z_coord, H_bathy, land_mask, smoothing_passes=bathy_smoothing_passes,
            min_levels=min_levels)
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


def _regrid_curv_to_points(field2d, src_lat_deg, src_lon_deg, ocean_mask,
                           tgt_lat_deg, tgt_lon_deg, k=4, max_deg=3.0):
    """IDW-regrid a curvilinear 2-D field (ocean cells only) onto ARBITRARY target
    points (any shape, e.g. cube (6,n,n)) using great-circle (chord) kNN. Mirrors
    ``compare_omip_nemo.regrid_curv_to_latlon`` but for point targets (that one
    meshgrids 1-D axes, so it can't take cube cell centres). Returns (values,
    ocean_flag) with the target's shape; ocean_flag=0 where the nearest source
    ocean cell is farther than ``max_deg``."""
    from scipy.spatial import cKDTree

    def _xyz(lat_r, lon_r):
        cl = np.cos(lat_r)
        return np.stack([cl * np.cos(lon_r), cl * np.sin(lon_r),
                         np.sin(lat_r)], axis=-1)

    m = np.asarray(ocean_mask).ravel() > 0.5
    if not m.any():
        raise ValueError("no ocean source cells")
    src_xyz = _xyz(np.deg2rad(np.asarray(src_lat_deg).ravel()[m]),
                   np.deg2rad(np.asarray(src_lon_deg).ravel()[m]))
    vals = np.asarray(field2d, dtype=np.float64).ravel()[m]
    tshape = np.asarray(tgt_lat_deg).shape
    tgt_xyz = _xyz(np.deg2rad(np.asarray(tgt_lat_deg).ravel()),
                   np.deg2rad(np.asarray(tgt_lon_deg).ravel()))
    tree = cKDTree(src_xyz)
    d, idx = tree.query(tgt_xyz, k=k)
    d = np.maximum(d, 1e-12)
    w = (1.0 / d) / (1.0 / d).sum(axis=1, keepdims=True)
    out = (vals[idx] * w).sum(axis=1).reshape(tshape)
    chord = 2.0 * np.sin(np.deg2rad(max_deg) / 2.0)
    ocean = (d[:, 0].reshape(tshape) < chord).astype(np.float64)
    return out, ocean


def build_cubed_sphere(nlev: int, H_max: float, mesh_path: str, n: int = 48,
                       woa_init: bool = False, woa_t=None, woa_s=None,
                       flat_bottom: bool = False, A_h=None, hyperdiff_coeff=None,
                       div_damp_2=None, div_damp_4=None, baroclinic_rk3=None):
    """Build a cubed-sphere ocean (FC-Gram spectral baroclinic backend) with NEMO's
    OWN eORCA1 bathymetry/land-mask regridded onto the cube cell centres, for the
    faithful CORE-II comparison. The 3rd grid; reuses run_omip._create_setup (FC +
    fv3sw barotropic + face-edge-stability A_h/K_h) and the OMIP-2 applicator
    (grid_type='cubed_sphere'). NOTE: the cube OceanModel still has the documented
    PGF-over-bathy instability (docs/ocean_experiments/cubed_sphere_pgf_stability.md)
    that FC + elevated diffusion only delay; this builder is the harness to drive
    the dycore fix, not a finished faithful path."""
    from scripts.run import run_omip
    from legoesm.ocean.init import rest_state_ocean
    grid, z_coord, config, model, _ = run_omip._create_setup(
        "cubed_sphere", f"C{n}", nlev, H_max, physics_preset="full",
        water_type="II",
    )
    # _create_setup builds the cube with physics=None (face-edge stability), so
    # model.step(surface_forcing=sf) would DROP the CORE-II forcing (the FC cube
    # path applies surface forcing only via physics_fn). Configure EXTERNAL surface
    # forcing so the coupler-provided tau/q_net are applied (atmosphere convention,
    # ocean reaction = -tau -- exactly what compute_omip2_surface_forcing returns).
    # compute_omip2_surface_forcing folds shortwave INTO q_net, and the external
    # scheme deposits the FULL q_net in the surface layer, so DISABLE shortwave
    # penetration (=None) to avoid double-counting solar.
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    from legoesm.core.operators_fc import build_fc_config
    phys = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp"),
        lateral_mixing=LateralMixingConfig(scheme="harmonic"),
        surface_forcing=SurfaceForcingConfig(scheme="external"),
        bottom_drag=BottomDragConfig(scheme="none"),  # drag via model config, not physics
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
        shortwave_penetration=None,
    )
    config = config._replace(physics=phys)
    # Optional momentum-dissipation overrides (debug the wind-stress-driven
    # grid-scale momentum instability on the cube A/CD-grid: the doc's FC-stable
    # cube was tested under thermal restoring only, NOT wind tau).
    _ovr = {k: v for k, v in (("A_h", A_h),
                              ("hyperdiff_coeff", hyperdiff_coeff),
                              ("div_damp_2", div_damp_2),
                              ("div_damp_4", div_damp_4),
                              ("baroclinic_rk3", baroclinic_rk3)) if v is not None}
    if _ovr:
        config = config._replace(**_ovr)
        print(f"[setup] cube config override: {_ovr}")
    model = OceanModel(grid, z_coord, config,
                       fc_config=build_fc_config(dtype=jnp.float64))
    # NEMO bathy/mask -> cube cell centres (point-target IDW; the curvilinear
    # mesh is the same faithful geometry tripole/latlon use).
    import xarray as xr
    e_mask, e_H = read_mesh_mask_bathy(mesh_path)
    ds = xr.open_dataset(mesh_path)
    src_lat = _squeeze2d(ds["gphit"].values)
    src_lon = _squeeze2d(ds["glamt"].values)
    tgt_lat = np.rad2deg(np.asarray(grid.lat))   # (6, n, n)
    tgt_lon = np.rad2deg(np.asarray(grid.lon))
    H_cs, ocean_cs = _regrid_curv_to_points(
        e_H, src_lat, src_lon, e_mask, tgt_lat, tgt_lon, max_deg=3.0)
    land_mask = (ocean_cs > 0.5).astype(np.float64)
    H_bathy = np.where(land_mask > 0.5, np.maximum(H_cs, 50.0), 0.0)
    if flat_bottom:
        H_bathy = np.where(land_mask > 0.5, H_max, 0.0)
        print("[setup] FLAT BOTTOM (cube; topography removed)")
    print(f"[setup] cubed_sphere C{n}: ocean cells {int(land_mask.sum())}/"
          f"{land_mask.size}, H_bathy [{H_bathy[land_mask>0.5].min():.0f},"
          f"{H_bathy.max():.0f}] m")
    state = rest_state_ocean(grid, z_coord, H_max=H_max)
    state = state._replace(
        land_mask=state.land_mask.replace(data=jnp.asarray(land_mask)),
        H_bathy=state.H_bathy.replace(data=jnp.asarray(H_bathy)),
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
        print(f"[setup] cube T/S initialised from WOA18 ({Path(woa_t).name})")
    return grid, z_coord, model, state, np.asarray(H_bathy)


def _idx_t(step: int, dt: float, n_rec: int) -> int:
    """Nearest 6-hourly CORE-II record for the current model time (perpetual yr)."""
    t = (step * dt) % _YEAR_S
    # CORE-II 6-hourly records are cell-CENTRED at (k+0.5)*6h (see
    # build_core2_nyf_zarr time_s=(arange+0.5)*6h), so the record whose centre is
    # nearest model time t is floor(t/6h), not round(t/6h) -- the latter applies
    # each record with a +3 h phase lead.
    return int(t // _SEC_PER_6H) % n_rec


def _diag(state, lat2d=None, lon2d=None) -> dict:
    """Cheap scalar diagnostics over ocean cells (one device->host pull).

    ``umax_lat``/``umax_lon``/``umax_lev`` = location of the 3-D max|u| —
    localises WHERE velocity grows/blows up (equator f->0 vs western
    boundaries vs the bipolar cap/fold vs at depth), the key pin-point for
    the dynamics instability seed.
    """
    T = np.asarray(state.T.data)[..., 0]
    S = np.asarray(state.S.data)[..., 0]
    u = np.asarray(state.u.data)            # (nlat, nlon+1, nlev)
    v = np.asarray(state.v.data)
    m = np.asarray(state.land_mask.data) > 0.5
    au = np.abs(u)
    has_u = bool(au.size and np.isfinite(au).any())
    max_speed = float(np.nanmax(au)) if has_u else float("nan")
    umax_lat = umax_lon = float("nan")
    umax_lev = -1
    # Cube u is 4-D (6, n, n, nlev); the C-grid u is 3-D (n_lat, n_lon+1, nlev).
    # The umax-location pin-point below only makes sense for the 2-D-mappable
    # C-grid case, so skip it (keep max_speed + finite, which are shape-agnostic)
    # when the field is the cube layout or the coord arrays don't match.
    if lat2d is not None and has_u and u.ndim == 4 and np.asarray(lat2d).ndim == 3:
        # Cube A-grid: u (6, n, n, nlev) collocated with T -> lat2d (6, n, n).
        fu, ju, iu, ku = (int(x) for x in
                          np.unravel_index(np.nanargmax(au), au.shape))
        lat2d = np.asarray(lat2d)
        umax_lat = round(float(lat2d[fu, ju, iu]), 1)
        umax_lev = ku
        if lon2d is not None:
            umax_lon = round(float(np.asarray(lon2d)[fu, ju, iu]), 1)
    elif lat2d is not None and has_u and u.ndim == 3:
        ju, iu, ku = (int(x) for x in
                      np.unravel_index(np.nanargmax(au), au.shape))
        lat2d = np.asarray(lat2d)
        jj = min(ju, lat2d.shape[0] - 1)
        # u is a u-FACE field (n_lat, n_lon+1): column iu spans [0, n_lon]. The
        # T-centre coord arrays have n_lon columns and the wrap column n_lon is a
        # copy of column 0, so fold iu back with % (NOT clamp to n_lon-1, which
        # would report the cyclic-seam max ~360 deg away at the far edge).
        ii = iu % lat2d.shape[1]
        umax_lat = round(float(lat2d[jj, ii]), 1)
        umax_lev = ku
        if lon2d is not None:
            umax_lon = round(float(np.asarray(lon2d)[jj, ii]), 1)
    return {
        "mean_sst_C": float(np.nanmean(T[m])) if m.any() else float("nan"),
        "mean_sss": float(np.nanmean(S[m])) if m.any() else float("nan"),
        "max_abs_u": max_speed,
        "max_abs_v": float(np.nanmax(np.abs(v))) if v.size else 0.0,
        "umax_lat": umax_lat,
        "umax_lon": umax_lon,
        "umax_lev": umax_lev,
        # Guard ALL prognostic fields -- a blowup that goes non-finite first in
        # S or v (not just T/u) must still trip the ABORT, else a NaN state is
        # silently snapshotted.
        "finite": bool(np.isfinite(T).all() and np.isfinite(S).all()
                       and np.isfinite(u).all() and np.isfinite(v).all()),
    }


def _grid_lat2d_deg(grid, grid_type):
    """Lat/lon in degrees for snapshots/scoring, per grid type. Cube returns the
    (6,n,n) per-face arrays as-is (the scorer flattens source points), tripole the
    2-D curvilinear T arrays, regular lat-lon the meshgridded 2-D axes."""
    if grid_type == "cubed_sphere":
        return (np.rad2deg(np.asarray(grid.lat)),
                np.rad2deg(np.asarray(grid.lon)))
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
                   choices=["tripole", "latlon_bathy", "cubed_sphere"],
                   help="tripole (eORCA1 same-grid), latlon_bathy (regular lat-lon + "
                        "NEMO bathy + smc03 + polar filter), or cubed_sphere (FC-Gram "
                        "backend + NEMO bathy on cube cells; grid 3, dycore WIP).")
    p.add_argument("--latlon-res", type=str, default="180x360",
                   help="lat-lon resolution NxM for --grid latlon_bathy.")
    p.add_argument("--cube-n", type=int, default=48,
                   help="cubed-sphere face resolution n (C-n) for --grid cubed_sphere.")
    p.add_argument("--cube-Ah", type=float, default=None,
                   help="cube horizontal viscosity A_h override [m^2/s].")
    p.add_argument("--cube-hyperdiff", type=float, default=None,
                   help="cube biharmonic hyperdiffusion coeff override.")
    p.add_argument("--cube-divdamp2", type=float, default=None,
                   help="cube 2nd-order divergence damping [m^2/s].")
    p.add_argument("--cube-divdamp4", type=float, default=None,
                   help="cube 4th-order divergence damping [m^4/s].")
    p.add_argument("--cube-rk3", action="store_true",
                   help="cube: 3-stage SSP-RK3 baroclinic update (vs forward-Euler) "
                        "— the tripole cold-start fix ported to the cube OceanModel.")
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
    p.add_argument("--slope-foot-alpha", type=float, default=None,
                   help="MOM6 slope-foot viscosity enhancement at topographic slopes "
                        "(WBCs hug slopes -- the built-in WBC-enhanced-viscosity, NEMO-like). "
                        "Multiplies A_h by ~alpha where slope>threshold. 0=off, prod 3.")
    p.add_argument("--slope-foot-n-levels", type=int, default=None,
                   help="Levels from bottom to apply slope-foot (default 5; set =nlev for full column).")
    p.add_argument("--slope-foot-threshold", type=float, default=None,
                   help="Slope r-factor threshold above which slope-foot fires (MOM6 default 0.1).")
    p.add_argument("--momentum-advection", default=None,
                   choices=[None,"vector_invariant","weno5","weno7"],
                   help="Momentum advection scheme. weno5/weno7 = upstream-biased "
                        "(dissipative at sharp jets, NEMO-UP3-like); vector_invariant "
                        "(default) = energy-conserving AL81 (NON-dissipative).")
    p.add_argument("--C-smag", type=float, default=None,
                   help="Biharmonic Smagorinsky coeff (self-activating ~strain, scale-selective).")
    p.add_argument("--C-leith", type=float, default=None,
                   help="Leith biharmonic coeff (self-activating ~|grad vorticity| -- targets the sharp-jet edge).")
    p.add_argument("--C-smag-lap", type=float, default=None,
                   help="Laplacian Smagorinsky coeff (tripole OMIP default 0.33).")
    p.add_argument("--bottom-drag-r", type=float, default=None,
                   help="Linear bottom drag coefficient [m/s] (du/dt|drag=-r*u/h_bot). "
                        "tripole OMIP default is 0 (OFF); NEMO uses implicit quadratic drag.")
    p.add_argument("--barotropic-solver", default=None, choices=[None,"explicit_substep","implicit_cn"],
                   help="Override barotropic solver. NEMO uses split-explicit forward-backward "
                        "(=explicit_substep here, with a dissipative cosine time filter); OMIP "
                        "default is implicit_cn (Crank-Nicolson, NEUTRAL -- no fast-gravity-wave damping).")
    p.add_argument("--barotropic-diffusion-alpha", type=float, default=None,
                   help="Barotropic 2D Laplacian damping coefficient (explicit_substep).")
    p.add_argument("--n-barotropic-substeps", type=int, default=None,
                   help="Number of barotropic substeps (explicit_substep).")
    p.add_argument("--barotropic-time-filter", default=None, choices=[None,"box","cosine"],
                   help="Barotropic time-average filter (cosine = more dissipative for fast modes).")
    p.add_argument("--momentum-rk3", action="store_true",
                   help="Use 3-stage SSP-RK3 for the outer baroclinic momentum step "
                        "(mirrors NEMO's RK3 / key_RK3) instead of forward-Euler -- the "
                        "NEMO-faithful fix for the cold-start adjustment blowup. 3x tendency cost.")
    p.add_argument("--freeze-floor", action="store_true",
                   help="Floor ocean T at the seawater freezing point (~-1.8 C) each "
                        "step -- a sea-ice thermodynamic surrogate. legoESM has no "
                        "prognostic ice, so high-lat (esp. Arctic) cells over-cool "
                        "3-5 C below NEMO (LIM ice caps SST). NEMO-faithful; removes "
                        "~half the Arctic SST RMSE. Off = bit-exact legacy.")
    p.add_argument("--polar-filter", action="store_true",
                   help="Enable the mask-aware Fourier polar filter (lat-lon grid only): "
                        "truncate the zonal modes exceeding the per-latitude CFL near the "
                        "converging-meridian poles, where a global lat-lon ocean otherwise "
                        "blows up ~day 0.25. Mask-aware (land filled with ocean zonal mean "
                        "before the FFT, restored after) so continents are not smeared into "
                        "ocean; tracer-conservative (k=0 mode kept). NOTE: lat-lon stays "
                        "imperfect in the land-locked Arctic -- ORCA tripole is the faithful "
                        "path; this is for lat-lon stability + a tropics/mid-lat/SH compare.")
    p.add_argument("--polar-filter-cutoff-lat", type=float, default=None,
                   help="Latitude (deg) poleward of which the polar filter acts (default 60).")
    p.add_argument("--polar-filter-max-wave-speed", type=float, default=None,
                   help="Max wave speed [m/s] setting the CFL wavenumber cap (default 300).")
    p.add_argument("--polar-filter-safety", type=float, default=None,
                   help="Fraction of the CFL wavenumber kept, <1 for margin (default 0.85).")
    p.add_argument("--woa-smoothing-passes", type=int, default=0,
                   help="Horizontal Laplacian smoothing passes/level on the WOA T,S IC "
                        "-- removes spurious grid-scale fronts from interpolating/flood-"
                        "filling WOA onto the tripole that imply >>2 m/s geostrophic flow "
                        "and break the cold start. Requires --woa-init. 0=off.")
    p.add_argument("--balanced-init", action="store_true",
                   help="Initialise the cold-start in geostrophic/thermal-wind balance "
                        "(level-of-no-motion velocity from the WOA p' field, equator-tapered) "
                        "+ balanced SSH -- removes the unbalanced-rest adjustment shock that is "
                        "the conclusively-diagnosed cause of the WOA cold-start blowup. "
                        "Requires --woa-init.")
    p.add_argument("--balanced-init-taper-lat", type=float, default=8.0,
                   help="Equatorial taper latitude [deg] for the geostrophic init: the "
                        "1/f singularity is regularised as f/(f^2+f_eps^2) with "
                        "f_eps=2*Omega*sin(taper_lat), so u_g,v_g taper to 0 within ~this "
                        "latitude of the equator.")
    p.add_argument("--no-balanced-ssh", action="store_true",
                   help="With --balanced-init, set only the geostrophic velocity (surface-"
                        "referenced inconsistency control); default also sets the balanced SSH.")
    p.add_argument("--bathy-smoothing-passes", type=int, default=0,
                   help="Laplacian smoothing passes on the OCEAN bathymetry before "
                        "building the partial-cell coord -- reduces the bathymetric "
                        "slope (r-factor) and hence the spurious partial-cell PGF seed "
                        "at steep continental slopes (NEMO/ROMS smooth for this). "
                        "Requires --partial-cell. 0=off.")
    p.add_argument("--output", type=str, default="results/omip_nemo/legoesm_tripole")
    p.add_argument("--diag-every-days", type=float, default=30.0)
    p.add_argument("--scan-block", type=int, default=0,
                   help="Issue #354: wrap the time loop in jax.lax.scan, "
                        "fusing this many steps per block (CORE-II forcing "
                        "sampled on-device, no per-step host roundtrip). "
                        "0 (default) = the bit-identical Python loop. "
                        "Tripole only; incompatible with --nudge-woa / "
                        "--spinup-drag. Diagnostics run at block boundaries.")
    p.add_argument("--snapshot-every-days", type=float, default=0.0,
                   help="Write a state snapshot every N sim-days (in addition "
                        "to yearly + final). 0=off. Lets a long run be scored "
                        "mid-flight (e.g. day-30 SST vs NEMO) without waiting "
                        "for the full integration.")
    p.add_argument("--forcing-ramp-days", type=float, default=0.0,
                   help="Ramp the surface forcing 0->full over N days "
                        "(cold-start shock mitigation).")
    p.add_argument("--min-levels", type=int, default=1,
                   help="rn_hmin-style conditioning: mask ocean columns with "
                        "fewer than this many active reference levels (partial-"
                        "cell path only). 1=off. Use 2 at 1/4 deg to remove the "
                        "single-thin-layer coastal cells that seed a 1/h blowup.")
    p.add_argument("--convection", type=str, default="none",
                   choices=["none", "enhanced_diffusion"],
                   help="Grid-agnostic convective adjustment (Oceananigans-"
                        "style enhanced vertical diffusivity where N^2<0), "
                        "applied via the implicit backward-Euler vertical "
                        "solve. Default 'none' preserves the validated "
                        "faithful config (tripole base ships physics=None).")
    p.add_argument("--convection-K-conv", type=float, default=1.0,
                   help="Convective diffusivity K_conv [m^2/s] for "
                        "--convection enhanced_diffusion (default 1.0).")
    p.add_argument("--convection-K-bg", type=float, default=1e-5,
                   help="Background diffusivity K_bg [m^2/s] for convection.")
    p.add_argument("--div-damp-2", type=float, default=None,
                   help="2nd-order divergence damping [m^2/s] -- suppresses "
                        "grid-scale divergent (checkerboard) modes at small "
                        "high-lat coastal cells. CFL: dt < dx^2/(2*nu).")
    p.add_argument("--div-damp-4", type=float, default=None,
                   help="4th-order (scale-selective) divergence damping [m^4/s] "
                        "-- damps the grid-scale mode far more than the resolved "
                        "flow; gentler CFL than 2nd-order. Try ~1e9-1e10 at 1/4 deg.")
    p.add_argument("--smag-cfl-safety", type=float, default=None,
                   help="Cap the Laplacian-Smagorinsky coeff at smag_cfl_safety*"
                        "area*cos^2(lat)/dt (per-cell tuned viscosity ceiling). "
                        "Lets --C-smag-lap "
                        "be cranked high to damp WBC jets WITHOUT self-CFL at the "
                        "sharp jet. ~0.125 is a safe 2-D Laplacian cap.")
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
            momentum_time_integrator=("rk3" if args.momentum_rk3 else None),
            freeze_floor=(True if args.freeze_floor else None),
            barotropic_solver=args.barotropic_solver,
            barotropic_diffusion_alpha=args.barotropic_diffusion_alpha,
            n_barotropic_substeps=args.n_barotropic_substeps,
            barotropic_time_filter=args.barotropic_time_filter,
            bottom_drag_r=args.bottom_drag_r,
            C_smag=args.C_smag, C_leith=args.C_leith, C_smag_lap=args.C_smag_lap,
            momentum_advection=args.momentum_advection,
            slope_foot_alpha=args.slope_foot_alpha,
            slope_foot_n_levels=args.slope_foot_n_levels,
            slope_foot_threshold=args.slope_foot_threshold,
            min_levels=args.min_levels,
            div_damp_2=args.div_damp_2, div_damp_4=args.div_damp_4,
            smag_cfl_safety=args.smag_cfl_safety,
            convection=args.convection,
            convection_K_conv=args.convection_K_conv,
            convection_K_bg=args.convection_K_bg,
        )
        app_grid_type = "tripole"
    elif args.grid == "cubed_sphere":
        grid, z_coord, model, state, H_bathy = build_cubed_sphere(
            args.nlev, args.H_max, args.mesh, n=args.cube_n,
            woa_init=args.woa_init, woa_t=args.woa_t, woa_s=args.woa_s,
            flat_bottom=args.flat_bottom,
            A_h=args.cube_Ah, hyperdiff_coeff=args.cube_hyperdiff,
            div_damp_2=args.cube_divdamp2, div_damp_4=args.cube_divdamp4,
            baroclinic_rk3=(True if args.cube_rk3 else None),
        )
        app_grid_type = "cubed_sphere"
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
            momentum_time_integrator=("rk3" if args.momentum_rk3 else None),
            freeze_floor=(True if args.freeze_floor else None),
            barotropic_solver=args.barotropic_solver,
            barotropic_diffusion_alpha=args.barotropic_diffusion_alpha,
            n_barotropic_substeps=args.n_barotropic_substeps,
            barotropic_time_filter=args.barotropic_time_filter,
            bottom_drag_r=args.bottom_drag_r,
            C_smag=args.C_smag, C_leith=args.C_leith, C_smag_lap=args.C_smag_lap,
            momentum_advection=args.momentum_advection,
            slope_foot_alpha=args.slope_foot_alpha,
            slope_foot_n_levels=args.slope_foot_n_levels,
            slope_foot_threshold=args.slope_foot_threshold,
            min_levels=args.min_levels,
            div_damp_2=args.div_damp_2, div_damp_4=args.div_damp_4,
            smag_cfl_safety=args.smag_cfl_safety,
            use_polar_filter=(True if args.polar_filter else None),
            polar_filter_cutoff_lat_deg=args.polar_filter_cutoff_lat,
            polar_filter_max_wave_speed=args.polar_filter_max_wave_speed,
            polar_filter_safety_factor=args.polar_filter_safety,
        )
        app_grid_type = "latlon"

    if args.woa_smoothing_passes and args.woa_smoothing_passes > 0:
        if not args.woa_init:
            raise ValueError("--woa-smoothing-passes requires --woa-init.")
        state = smooth_woa_ts(state, grid, args.woa_smoothing_passes)

    if args.balanced_init:
        if not args.woa_init:
            raise ValueError("--balanced-init requires --woa-init (it balances the WOA IC).")
        state = apply_balanced_init(
            state, grid, z_coord, model.config,
            taper_lat_deg=args.balanced_init_taper_lat,
            with_ssh=(not args.no_balanced_ssh),
        )

    lat2d, lon2d = _grid_lat2d_deg(grid, args.grid)
    # allow_synthetic=False: this NEMO-faithful pipeline MUST use the real
    # 6-hourly CORE-II nyf.zarr; a silent fallback to 365 daily synthetic forcing
    # would corrupt the comparison invisibly.
    forcing = load_core2_nyf(allow_synthetic=False)
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

    snap_every = (int(args.snapshot_every_days * _SEC_PER_DAY / dt)
                  if args.snapshot_every_days > 0 else 0)

    print(f"[run] {total_days:.0f} days = {n_steps} steps "
          f"(diag every {diag_every} steps"
          f"{f', snapshot every {snap_every} steps' if snap_every else ''})")
    d0 = _diag(state, lat2d, lon2d)
    print(f"[diag] step 0: {d0}", flush=True)

    # Progress time-series CSV, flushed each diag -> observable mid-run even when
    # stdout is pipe-buffered, and a record for post-hoc analysis.
    out_dir.mkdir(parents=True, exist_ok=True)
    _csv_cols = ["step", "day", "mean_sst_C", "mean_sss", "max_abs_u",
                 "max_abs_v", "umax_lat", "umax_lon", "umax_lev", "steps_per_s"]
    _csv = open(out_dir / "diag_timeseries.csv", "w")
    _csv.write(",".join(_csv_cols) + "\n")

    def _log_diag_csv(step, day, d, rate):
        _csv.write(
            f"{step},{day:.3f},{d['mean_sst_C']:.4f},{d['mean_sss']:.4f},"
            f"{d['max_abs_u']:.6e},{d['max_abs_v']:.6e},{d['umax_lat']},"
            f"{d['umax_lon']},{d['umax_lev']},{rate:.3f}\n")
        _csv.flush()

    _log_diag_csv(0, 0.0, d0, 0.0)

    t_wall = time.time()

    # ------------------------------------------------------------------
    # Issue #354: optional lax.scan block-stepping (tripole; no nudge/drag).
    # The CORE-II forcing is sampled on-device (no per-step host roundtrip),
    # so XLA fuses each block of ``--scan-block`` steps.  Default
    # (--scan-block 0) keeps the bit-identical Python loop below.
    # Diagnostics / snapshots / non-finite abort run at BLOCK BOUNDARIES.
    # ------------------------------------------------------------------
    _tti = getattr(getattr(model, "config", None),
                   "tracer_time_integrator", "euler")
    use_scan = (int(args.scan_block) > 0 and app_grid_type == "tripole"
                and nudge_tau_s == 0.0 and drag_tau_s == 0.0
                and _tti != "ab2")
    if int(args.scan_block) > 0 and not use_scan:
        why = ("AB2 tracer time integrator (None->Field carry breaks "
               "lax.scan)" if _tti == "ab2"
               else "grid!=tripole or WOA-nudging / spin-up-drag enabled "
                    "(those need per-step host updates)")
        print(f"[scan] --scan-block ignored: {why}.", flush=True)
    if use_scan:
        from legoesm.ocean.coupler.omip2_applicator import (
            build_core2_forcing_device_stack, build_omip2_scan_block_fn,
        )
        f_stack, nn_i, nn_j, gshape = build_core2_forcing_device_stack(
            forcing, grid, "tripole")
        block_fn = build_omip2_scan_block_fn(model, dt, gshape, ramp_s=ramp_s)
        bsz = int(args.scan_block)
        print(f"[run] lax.scan block-stepping: block<={bsz} steps, split at "
              f"diag/snapshot/year boundaries so output cadence matches the "
              f"Python loop (CORE-II forcing fused on-device)", flush=True)

        def _block_steps(step):
            # Cap the block so it ENDS on the next diagnostic / snapshot /
            # year boundary -> the modulo-gated I/O below fires at exactly
            # the same cadence as the Python loop (codex #354 finding 2).
            nb = min(bsz, n_steps - step)
            for period in (diag_every, snap_every, steps_per_year):
                if period and period > 0:
                    nb = min(nb, period - (step % period))
            return max(1, nb)

        step = 0
        while step < n_steps:
            nb = _block_steps(step)
            idx_block = jnp.asarray(
                [_idx_t(step + 1 + k, dt, n_rec) for k in range(nb)],
                dtype=jnp.int32)
            state = block_fn(state, f_stack, nn_i, nn_j, idx_block,
                             jnp.int32(step + 1))
            step += nb
            day = step * dt / _SEC_PER_DAY
            if step % diag_every == 0 or step == n_steps:
                state = jax.block_until_ready(state)
                d = _diag(state, lat2d, lon2d)
                rate = step / (time.time() - t_wall)
                print(f"[diag] step {step} (day {day:.0f}): {d} | "
                      f"{rate:.2f} steps/s", flush=True)
                _log_diag_csv(step, day, d, rate)
                if not d["finite"]:
                    print("[ABORT] non-finite state", flush=True)
                    _save_snapshot(out_dir, f"blowup_step{step}",
                                   state, lat2d, lon2d)
                    _csv.close()
                    return 1
            if snap_every > 0 and step % snap_every == 0 and step != n_steps:
                _save_snapshot(out_dir, f"day{int(round(day)):04d}",
                               state, lat2d, lon2d)
                print(f"[snapshot] day {day:.0f} saved", flush=True)
            if not args.smoke and steps_per_year > 0 and step % steps_per_year == 0:
                yr = step // steps_per_year
                _save_snapshot(out_dir, f"year{yr:03d}", state, lat2d, lon2d)
                print(f"[snapshot] year {yr} saved", flush=True)
        state = jax.block_until_ready(state)
        _save_snapshot(out_dir, "final", state, lat2d, lon2d)
        _csv.close()
        rate = n_steps / (time.time() - t_wall)
        print(f"[done] {n_steps} steps @ {rate:.2f} steps/s (scan); "
              f"final: {_diag(state, lat2d, lon2d)}")
        if args.smoke:
            yr_est = steps_per_year / rate / 3600.0
            print(f"[smoke] projected wall-time: {yr_est:.2f} h/yr  "
                  f"({args.years:.0f}yr -> {yr_est*args.years:.1f} h)")
        return 0

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
            d = _diag(state, lat2d, lon2d)
            rate = step / (time.time() - t_wall)
            day = step * dt / _SEC_PER_DAY
            print(f"[diag] step {step} (day {day:.0f}): {d} | {rate:.2f} steps/s",
                  flush=True)
            _log_diag_csv(step, day, d, rate)
            if not d["finite"]:
                print("[ABORT] non-finite state", flush=True)
                _save_snapshot(out_dir, f"blowup_step{step}", state, lat2d, lon2d)
                _csv.close()
                return 1
        if snap_every > 0 and step % snap_every == 0 and step != n_steps:
            day = step * dt / _SEC_PER_DAY
            _save_snapshot(out_dir, f"day{int(round(day)):04d}", state, lat2d, lon2d)
            print(f"[snapshot] day {day:.0f} saved", flush=True)
        if not args.smoke and steps_per_year > 0 and step % steps_per_year == 0:
            yr = step // steps_per_year
            _save_snapshot(out_dir, f"year{yr:03d}", state, lat2d, lon2d)
            print(f"[snapshot] year {yr} saved", flush=True)

    state = jax.block_until_ready(state)
    _save_snapshot(out_dir, "final", state, lat2d, lon2d)
    _csv.close()
    rate = n_steps / (time.time() - t_wall)
    print(f"[done] {n_steps} steps @ {rate:.2f} steps/s; final: {_diag(state, lat2d, lon2d)}")
    if args.smoke:
        yr_est = steps_per_year / rate / 3600.0
        print(f"[smoke] projected wall-time: {yr_est:.2f} h/yr  "
              f"({args.years:.0f}yr -> {yr_est*args.years:.1f} h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
