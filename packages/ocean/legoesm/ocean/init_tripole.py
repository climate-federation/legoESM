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


def mask_to_nemo_domain(land_mask, lat_deg, lon_deg, domain_cfg_path, *,
                        coord_tol_deg: float = 1e-3):
    """Set to LAND every cell that is wet in ``land_mask`` but dry in the
    oracle's ``domain_cfg.nc`` (``top_level == 0``), so the wet domain matches
    the NEMO run being compared against.

    Why: the eORCA1.2 ``mesh_mask.nc`` keeps three closed basins wet -- the
    Great Lakes, the Caspian Sea and Lake Victoria (207 cells) -- that the
    ORCA1 ``domain_cfg`` NEMO actually runs has as land (``namclo`` defaults:
    ``ln_mask_csundef = .true.``).  Kept wet they carry WOA-extrapolated
    ocean salinity and are restored to it forever, and they are the whole
    wet-for-us / dry-for-the-oracle set on every scorecard.

    The domain_cfg frame is NEMO's inner domain (``(331, 360)`` for ORCA1)
    while the mesh frame carries the cyclic halo columns and the north-fold
    row (``(332, 362)``).  The offset between the two is FOUND by matching
    the coordinates themselves (``gphit``/``glamt`` on both files) and the
    call refuses to guess: no offset with every |dlat|, |dlon| below
    ``coord_tol_deg`` -> ``ValueError``.  Cells dry in the mesh but wet in
    domain_cfg are NEVER wetted (there is no bathymetry for them); their
    count is printed so a mesh/domain mismatch in that direction is visible.

    Returns a masked copy of ``land_mask`` with the same shape, always as
    ``float64`` (the comparison against ``top_level`` is done in float64), so
    a bool or float32 mask comes back widened.
    """
    import xarray as xr

    ds = xr.open_dataset(domain_cfg_path, mask_and_scale=False)
    try:
        top = _squeeze2d(np.asarray(ds["top_level"].values))
        dlat = _squeeze2d(np.asarray(ds["gphit"].values, dtype=np.float64))
        dlon = _squeeze2d(np.asarray(ds["glamt"].values, dtype=np.float64))
    finally:
        ds.close()
    mask = np.asarray(land_mask, dtype=np.float64).copy()
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    if lat.shape != mask.shape or lon.shape != mask.shape:
        raise ValueError(
            f"lat/lon {lat.shape}/{lon.shape} do not match land_mask {mask.shape}")
    nj, ni = top.shape
    ny, nx = mask.shape
    if nj > ny or ni > nx:
        raise ValueError(
            f"domain_cfg frame {top.shape} is larger than the mesh frame "
            f"{mask.shape}; it must be the mesh's inner domain")

    def _agrees(j0, i0):
        sl = lat[j0:j0 + nj, i0:i0 + ni]
        sn = lon[j0:j0 + nj, i0:i0 + ni]
        if sl.shape != top.shape:
            return False
        dphi = np.abs(sl - dlat)
        dlam = np.abs((sn - dlon + 180.0) % 360.0 - 180.0)
        return bool(np.all(dphi < coord_tol_deg) and np.all(dlam < coord_tol_deg))

    offset = next(((j0, i0) for j0 in range(ny - nj + 1)
                   for i0 in range(nx - ni + 1) if _agrees(j0, i0)), None)
    if offset is None:
        raise ValueError(
            f"no offset places the domain_cfg frame {top.shape} inside the mesh "
            f"frame {mask.shape} with coordinates agreeing to {coord_tol_deg} deg; "
            f"{domain_cfg_path} is not the domain_cfg of this mesh")
    j0, i0 = offset
    inner = mask[j0:j0 + nj, i0:i0 + ni]
    nemo_wet = top > 0
    ours_wet = inner > 0.5
    n_to_land = int(np.sum(ours_wet & ~nemo_wet))
    n_dry_here_wet_there = int(np.sum(~ours_wet & nemo_wet))
    inner[ours_wet & ~nemo_wet] = 0.0
    print(f"[setup] wet domain matched to NEMO domain_cfg {domain_cfg_path}: "
          f"frame offset (j0={j0}, i0={i0}); {n_to_land} cells wet in the mesh "
          f"but dry for NEMO -> LAND; {n_dry_here_wet_there} cells dry in the "
          f"mesh but wet for NEMO (left dry: no bathymetry for them)")
    return mask


# Enclosed seas the observed IC cannot fill (PHC3 has no profile at any depth
# in the Marmara or the Black Sea, so every column there is stitched from
# Aegean and Black-Sea donors and the stitch is a density wall); masked as
# land, as the FESOM FORCA20 mesh does.  (lat_min, lat_max, lon_min, lon_max)
# in degrees, lon in [-180, 180).
# ponytail: the Dardanelles west of 26.9E stay ocean as a dead-end channel;
# a flood fill from the Bosporus replaces the boxes if that ever matters.
CLOSED_SEAS = {
    "marmara": (40.3, 41.1, 26.9, 30.0),
    "black_sea": (40.9, 47.5, 27.3, 42.0),   # includes the Bosporus and the Azov
}


def read_mesh_mask_bathy(mesh_path, *, strip_north_rows: int = 0,
                         nemo_domain_cfg: str | None = None,
                         closed_seas=()):
    """Derive the 2-D ocean land mask and total bathymetric depth from NEMO's
    own mesh files (eORCA1 ``mesh_mask.nc``, or the split ``mesh_hgr.nc`` +
    ``mesh_zgr.nc`` + ``mask.nc`` of older runs such as NOC ORCA0083) -- the most
    faithful geometry for the comparison.

    ``mesh_path`` is one path, a sequence of paths, or an ``os.pathsep``-joined
    string (see :func:`legoesm.grids.tripole.mesh_file_list`); variables are
    looked up across all files, first hit wins.  ``strip_north_rows`` drops that
    many NORTH rows (must match the ``create_tripole_grid`` call for the same
    mesh -- see its docstring for the ORCA12 dead-halo-row case).
    ``nemo_domain_cfg`` (a ``domain_cfg.nc`` path) additionally sets to land
    every cell the oracle runs dry -- see :func:`mask_to_nemo_domain`; the
    mask is applied BEFORE the bathymetry is masked, so ``H_bathy`` is 0 there.

    Returns
    -------
    land_mask : (n_lat, n_lon) float64, 1 = ocean, 0 = land  (tmaskutil)
    H_bathy   : (n_lat, n_lon) float64, total wet-column depth [m]
                (sum_k e3t_0 * tmask; NEMO 3.x mesh_zgr names it ``e3t``)
    """
    import xarray as xr
    from legoesm.grids.tripole import mesh_file_list
    # mask_and_scale=False: NEMO writes e3t_0 with _FillValue = 0.0, so xarray's
    # default decoding turns every LAND scale factor (exactly 0) into NaN and
    # the masked column sum below becomes NaN on land.  Read the raw values.
    dss = [xr.open_dataset(p, mask_and_scale=False)
           for p in mesh_file_list(mesh_path)]

    def _var(*names):
        for ds in dss:
            for nm in names:
                if nm in ds:
                    return ds[nm].values
        raise KeyError(
            f"none of {names} found in mesh files {mesh_file_list(mesh_path)}")

    # Surface ocean/land mask (1 = ocean). Prefer the 2-D util mask.
    land_mask = _squeeze2d(_var("tmaskutil", "tmask")).astype(np.float64)
    if nemo_domain_cfg:
        land_mask = mask_to_nemo_domain(
            land_mask,
            _squeeze2d(_var("gphit")).astype(np.float64),
            _squeeze2d(_var("glamt")).astype(np.float64),
            nemo_domain_cfg)
    # Total wet-column depth = sum over z of e3t_0 where tmask is wet.
    e3t = np.asarray(_var("e3t_0", "e3t"))            # (t,z,y,x) or (z,y,x)
    tmask = np.asarray(_var("tmask"))
    while e3t.ndim > 3:
        e3t = e3t[0]
    while tmask.ndim > 3:
        tmask = tmask[0]
    H_bathy = (e3t * tmask).sum(axis=0).astype(np.float64)   # (y, x)
    # Dry columns get exactly 0 depth; a non-finite depth on a WET column is a
    # broken mesh, never something to carry into the dynamics.
    H_bathy = np.where(land_mask > 0.5, H_bathy, 0.0)
    if not np.all(np.isfinite(H_bathy[land_mask > 0.5])):
        raise ValueError(
            f"non-finite wet-column depth in mesh {mesh_file_list(mesh_path)}")
    unknown = sorted(set(closed_seas) - set(CLOSED_SEAS))
    if unknown:
        raise ValueError(f"unknown closed_seas {unknown}; known: {sorted(CLOSED_SEAS)}")
    if closed_seas:
        lat = _squeeze2d(_var("gphit", "nav_lat"))
        lon = _squeeze2d(_var("glamt", "nav_lon"))
        lon = np.mod(lon + 180.0, 360.0) - 180.0
        closed = np.zeros(land_mask.shape, dtype=bool)
        for name in closed_seas:
            lat_min, lat_max, lon_min, lon_max = CLOSED_SEAS[name]
            closed |= ((lat >= lat_min) & (lat <= lat_max)
                       & (lon >= lon_min) & (lon <= lon_max))
        land_mask = np.where(closed, 0.0, land_mask)
        H_bathy = np.where(closed, 0.0, H_bathy)
    if strip_north_rows < 0:
        raise ValueError(f"strip_north_rows must be >= 0, got {strip_north_rows}")
    if strip_north_rows >= land_mask.shape[0]:
        raise ValueError(
            f"strip_north_rows={strip_north_rows} would remove every row of the "
            f"{land_mask.shape} mask")
    if strip_north_rows:
        land_mask = land_mask[:-strip_north_rows]
        H_bathy = H_bathy[:-strip_north_rows]
    return land_mask, H_bathy


def read_mesh_vertical_1d(mesh_path):
    """Read NEMO's 1-D reference vertical grid from the mesh files.

    Returns ``(e3t_1d, gdept_1d, gdepw_1d)`` in metres: the reference layer
    thicknesses, the T-point depths and the W-point (interface) depths that
    NEMO's own ``zgr_zps`` uses to place the bottom level.  The T-point depths
    are what ``create_partial_cell_coordinate(..., bottom_index_rule=
    "nemo_tpoint")`` needs; deriving them from the thicknesses instead changes
    which level is the bottom one on a stretched grid, so they are read, never
    reconstructed.

    Same file-scan contract as :func:`read_mesh_mask_bathy` (first hit wins
    across the listed mesh files).
    """
    import xarray as xr
    from legoesm.grids.tripole import mesh_file_list

    files = mesh_file_list(mesh_path)
    dss = [xr.open_dataset(p, mask_and_scale=False) for p in files]
    try:
        def _var1d(name, *alts):
            for ds in dss:
                for nm in (name, *alts):
                    if nm in ds:
                        v = ds[nm]
                        if not np.issubdtype(np.asarray(v.values).dtype,
                                             np.floating):
                            # mask_and_scale=False (needed because NEMO writes
                            # _FillValue=0 on e3t) also disables unpacking, so a
                            # packed integer variable would arrive as raw counts
                            # and pass every range check below.
                            raise ValueError(
                                f"{nm} in {files} is {np.asarray(v.values).dtype}, "
                                f"not floating point; this reader cannot unpack "
                                f"scaled integer storage")
                        a = np.asarray(v.values, dtype=np.float64)
                        squeezed = a.squeeze()
                        if squeezed.ndim != 1:
                            raise ValueError(
                                f"{nm} in {files} has shape {a.shape}; a 1-D "
                                f"reference ladder is required (a leading time "
                                f"or ensemble dimension is not collapsed here "
                                f"because concatenating it would silently "
                                f"multiply the level count)")
                        return np.atleast_1d(squeezed)
            raise KeyError(f"none of {(name, *alts)} found in mesh files {files}")

        e3t_1d = _var1d("e3t_1d", "e3t_0_1d")
        gdept_1d = _var1d("gdept_1d")
        gdepw_1d = _var1d("gdepw_1d")
    finally:
        for ds in dss:
            ds.close()
    n = e3t_1d.size
    if gdept_1d.size != n or gdepw_1d.size not in (n, n + 1):
        raise ValueError(
            f"inconsistent NEMO 1-D vertical arrays in {files}: "
            f"e3t_1d {e3t_1d.size}, gdept_1d {gdept_1d.size}, "
            f"gdepw_1d {gdepw_1d.size}")
    for nm, a in (("e3t_1d", e3t_1d), ("gdept_1d", gdept_1d),
                  ("gdepw_1d", gdepw_1d)):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{nm} in {files} is not finite")
    # Strictly positive: a zero reference thickness would become a
    # zero-thickness level and divide the tracer tendency by nothing.
    if np.any(e3t_1d <= 0.0):
        raise ValueError(f"e3t_1d in {files} has a non-positive thickness")
    if np.any(np.diff(gdept_1d) <= 0.0):
        raise ValueError(f"gdept_1d in {files} is not strictly increasing")
    if np.any(gdepw_1d < 0.0):
        raise ValueError(f"gdepw_1d in {files} has a negative depth")
    # The two ladders must describe the SAME grid: every T point sits inside
    # its own layer.  Meshes whose thicknesses and depths come from different
    # NEMO configurations have equal level counts and would otherwise build a
    # sheared coordinate in silence.
    interfaces = np.concatenate([[0.0], np.cumsum(e3t_1d)])
    if np.any(gdept_1d <= interfaces[:-1]) or np.any(gdept_1d >= interfaces[1:]):
        bad = int(np.argmax((gdept_1d <= interfaces[:-1])
                            | (gdept_1d >= interfaces[1:])))
        raise ValueError(
            f"gdept_1d and e3t_1d in {files} describe different vertical "
            f"grids: level {bad} has T depth {gdept_1d[bad]:.4f} m outside its "
            f"layer [{interfaces[bad]:.4f}, {interfaces[bad + 1]:.4f}] m")
    return e3t_1d, gdept_1d, gdepw_1d


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
