"""NEMO eORCA tripole geometry + WOA18 initial-condition loaders.

These helpers read the faithful eORCA1 geometry (land mask + bathymetry) from
NEMO's own ``mesh_mask`` file and build the WOA18 T/S initial condition on the
model grid.  They were originally defined inside
``scripts/run/run_omip_core2.py``; promoting them into the ocean package lets
BOTH the OMIP forced-ocean driver AND the coupled-ESM driver (the Phase-2
tripole coupler) build the SAME validated tripole cold-start without the coupler
importing from ``scripts/`` (a layering violation).

Pure host-side NumPy / xarray geometry + IC construction — no traced JAX
numerics, no autodiff concern (these produce the static initial state).
"""

from __future__ import annotations

import numpy as np


def squeeze_nemo_field_2d(a: np.ndarray) -> np.ndarray:
    """Drop leading singleton (t, z) dims from a NEMO mesh field -> 2-D (y, x)."""
    a = np.asarray(a)
    while a.ndim > 2:
        a = a[0]
    return a


# Backward-compatible private alias so the moved function bodies below read
# exactly as they did in ``run_omip_core2.py`` (no behaviour change).
_squeeze2d = squeeze_nemo_field_2d


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
            # Cube exposes .lat/.lon; MPAS VoronoiMesh exposes .latCell/.lonCell
            # (radians).  Same grid-type dispatch as init_ocean_from_woa.
            latr = np.asarray(getattr(grid, "lat",
                                      getattr(grid, "latCell", None))).ravel()
            lonr = np.asarray(getattr(grid, "lon",
                                      getattr(grid, "lonCell", None))).ravel()
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
        # Broadcast z_cen against ANY horizontal layout: cube H_bathy is (6,n,n)
        # (ndim 3), MPAS is (nCells,) (ndim 1).  reshape z_cen to (1,...,1,nlev)
        # so z_cen[...] > H_bathy[...,None] gives (*horiz, nlev) for both — the
        # old (None,None,:) hardcoded a 2-D horizontal and mis-broadcast on the
        # 1-D MPAS layout (silent IC corruption).
        Hb = np.asarray(H_bathy)
        z_cen_b = z_cen.reshape((1,) * Hb.ndim + (-1,))
        below = z_cen_b > Hb[..., None]
    T_woa = np.where(below, T_fill, T_woa)
    S_woa = np.where(below, S_fill, S_woa)
    m3 = m2[..., None]
    return T_woa * m3, S_woa * m3
