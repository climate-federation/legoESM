"""OMIP-2 / CORE-II / JRA55-do surface forcing for the ocean drivers.

:func:`compute_omip2_surface_forcing` samples the atmospheric forcing onto the
model grid, evaluates the NCAR bulk fluxes (``legoesm.ocean.bulk_flux_omip``)
and assembles NEMO's ``blk_oce`` non-solar heat flux, returning an
``OceanSurfaceForcing`` that ``model.step(state, dt, surface_forcing=...)``
integrates inside the timestep (the dynamics core applies the ``-tau`` ocean
reaction, the geographic->grid rotation and the shortwave penetration).
:func:`compute_omip2_freshwater_forcing` builds the matching P - E channel
for ``model.step(..., freshwater=...)``.

Sign conventions: ``tau_x``/``tau_y`` in the ATMOSPHERIC convention as
``air_sea_fluxes`` returns them; ``q_net`` positive INTO the ocean and
including shortwave.

Shortwave albedo is applied only when a sea-ice concentration is passed
(``ice_albedo``); with ``ice_albedo=None`` the downwelling SW enters unchanged.

The operator-split forward-Euler applicator (``apply_omip2_surface_fluxes``)
was deleted for issue #1820: it was applied outside the timestep, did not
weight downward longwave by emissivity, and added geographic stress to
cube-local velocities unrotated.
"""

from __future__ import annotations


import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.ocean.bulk_flux_omip import air_sea_fluxes
from legoesm.ocean.eos import VALID_FREEZE_SCHEMES, freezing_point

# --- per-step host-transfer ledger (scaling-M2 honest-cost accounting) ------
# The forcing builders below each pull ONE 2-D surface-T slice of the ocean
# state to the host per call (device-side slice FIRST, so only the surface
# layer crosses — never the full 3-D, possibly lat-band-sharded, leaf; codex
# batch4 HIGH: ``np.asarray(full leaf)[..., 0]`` assembled the whole field).
# Each pull is recorded here so the persistent-sharded OMIP driver
# (``run_omip_core2 --spmd-persistent-state``) can report the TRUE per-step
# leaf-transfer cost next to its full-state gather counter instead of
# implying zero transfer cost.  Counters are process-global and MONOTONIC:
# consumers snapshot :func:`host_pull_ledger` and take deltas.  Pure Python
# int side effect at host level — the builders are host-loop functions and
# are never jitted/traced, so this cannot leak into a trace.
_HOST_PULL_LEDGER = {"surface_slice_pulls": 0}


def host_pull_ledger() -> dict:
    """Copy of the module's monotonic host-transfer counters (accessor per
    the mutable-singleton rule — never import the dict itself)."""
    return dict(_HOST_PULL_LEDGER)


def _record_surface_slice_pull() -> None:
    _HOST_PULL_LEDGER["surface_slice_pulls"] += 1


# Cache of precomputed nearest-neighbour (lat_idx, lon_idx) maps, keyed by a
# (src, dst) signature. The target points (grid cell centres) are fixed within
# a run, so the O(src*dst) index search runs once; per-step calls are a single
# fancy-index.
_NN_INDEX_CACHE: dict = {}

def _coord_key(*arrays):
    """Collision-safe cache key from coordinate arrays: full-content hash
    (shape + dtype + bytes).  The float-checksum keys previously used
    (first/last/sum) could collide between two different same-length target
    sets and silently reuse another grid's indices/weights (codex YELLOW,
    2026-08-26).  Coordinate arrays are small and hashed once per (src, dst)
    pair, so the full hash costs microseconds."""
    import hashlib
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()



def _nn_interp_to_points(field, src_lat_deg, src_lon_deg,
                         dst_lat_deg_pts, dst_lon_deg_pts):
    """True nearest-neighbour lookup from a regular 1-D ``(src_lat, src_lon)``
    grid to a 1-D set of target points ``(dst_lat_pts, dst_lon_pts)``.

    Latitude: nearest by ``|Δlat|``. Longitude: nearest by *periodic*
    (wrap-around) distance, so a target near the 0/360 seam picks the truly
    closest source column. Source axes need not be pre-sorted. Used for cube,
    MPAS, and tripole targets (unstructured / 2-D cell centres); the regular
    lat-lon path uses ``conservative_regrid`` instead.
    """
    src_lat = np.asarray(src_lat_deg, dtype=np.float64)
    src_lon = np.asarray(src_lon_deg, dtype=np.float64) % 360.0
    dst_lat = np.asarray(dst_lat_deg_pts, dtype=np.float64)
    dst_lon = np.asarray(dst_lon_deg_pts, dtype=np.float64) % 360.0
    key = _coord_key(src_lat, src_lon, dst_lat, dst_lon)
    idx = _NN_INDEX_CACHE.get(key)
    if idx is None:
        # nearest source latitude (absolute difference)
        i = np.argmin(np.abs(src_lat[:, None] - dst_lat[None, :]), axis=0)
        # nearest source longitude (periodic distance on the circle)
        dlon = np.abs(src_lon[:, None] - dst_lon[None, :])
        dlon = np.minimum(dlon, 360.0 - dlon)
        j = np.argmin(dlon, axis=0)
        idx = (i, j)
        _NN_INDEX_CACHE[key] = idx
    i, j = idx
    return np.asarray(field)[i, j]


# Cache of precomputed 4-point bilinear (i4, j4, w4) maps, same keying scheme
# as _NN_INDEX_CACHE.
_BILIN_MAP_CACHE: dict = {}


def _bilinear_point_maps(src_lat_deg, src_lon_deg,
                         dst_lat_deg_pts, dst_lon_deg_pts):
    """4-point bilinear interpolation maps from a regular 1-D
    ``(src_lat, src_lon)`` grid to target points: ``(i4, j4, w4)``, each
    ``(4, n_pts)``, with ``sum_k w4[k] == 1``.

    Nearest-neighbour sampling of the ~1.9-degree CORE-II grid onto the
    1-degree tripole copies each coarse source ROW onto 1-2 adjacent target
    rows, printing zonally aligned bands into every forcing channel (wind,
    shortwave, precip) that surface mixing then inherits — the unphysical
    zonal MLD banding reported 2026-08-26.  NEMO interpolates CORE-II
    bilinearly (its weights files), so bilinear is also the faithful choice.

    Latitude: linear between the two bracketing source rows, clamped to the
    edge row beyond the source's outermost latitudes (CORE-II stops ~89.5
    deg short of the pole).  Handles ascending or descending source lat.
    Longitude: periodic bracketing on the circle (the seam target between
    the last and first source column interpolates across the wrap).
    """
    src_lat = np.asarray(src_lat_deg, dtype=np.float64)
    src_lon = np.asarray(src_lon_deg, dtype=np.float64) % 360.0
    dst_lat = np.asarray(dst_lat_deg_pts, dtype=np.float64)
    dst_lon = np.asarray(dst_lon_deg_pts, dtype=np.float64) % 360.0
    key = _coord_key(src_lat, src_lon, dst_lat, dst_lon)
    maps = _BILIN_MAP_CACHE.get(key)
    if maps is not None:
        return maps

    # --- latitude bracket (sorted ascending view; indices mapped back) ---
    order = np.argsort(src_lat)
    lat_sorted = src_lat[order]
    hi = np.searchsorted(lat_sorted, dst_lat)            # first >= target
    hi = np.clip(hi, 1, lat_sorted.size - 1)
    lo = hi - 1
    span = lat_sorted[hi] - lat_sorted[lo]
    wlat = np.where(span > 0.0,
                    (dst_lat - lat_sorted[lo]) / np.where(span > 0.0, span, 1.0),
                    0.0)
    wlat = np.clip(wlat, 0.0, 1.0)      # clamp: beyond edges -> edge row
    i_lo = order[lo].astype(np.int32)
    i_hi = order[hi].astype(np.int32)

    # --- longitude bracket, periodic ---
    lorder = np.argsort(src_lon)
    lon_sorted = src_lon[lorder]
    jhi = np.searchsorted(lon_sorted, dst_lon)
    at_wrap = (jhi == 0) | (jhi == lon_sorted.size)
    jhi_in = np.clip(jhi, 1, lon_sorted.size - 1)
    jlo_in = jhi_in - 1
    # wrap bracket: last column .. first column across the seam
    j_lo = np.where(at_wrap, lorder[-1], lorder[jlo_in]).astype(np.int32)
    j_hi = np.where(at_wrap, lorder[0], lorder[jhi_in]).astype(np.int32)
    lon_lo = np.where(at_wrap, lon_sorted[-1], lon_sorted[jlo_in])
    lon_hi = np.where(at_wrap, lon_sorted[0], lon_sorted[jhi_in])
    dspan = (lon_hi - lon_lo) % 360.0
    dpos = (dst_lon - lon_lo) % 360.0
    wlon = np.where(dspan > 0.0, dpos / np.where(dspan > 0.0, dspan, 1.0), 0.0)
    wlon = np.clip(wlon, 0.0, 1.0)

    i4 = np.stack([i_lo, i_lo, i_hi, i_hi])              # (4, n)
    j4 = np.stack([j_lo, j_hi, j_lo, j_hi])
    w4 = np.stack([(1 - wlat) * (1 - wlon), (1 - wlat) * wlon,
                   wlat * (1 - wlon), wlat * wlon])
    maps = (i4, j4, w4)
    _BILIN_MAP_CACHE[key] = maps
    return maps


def _bilinear_interp_to_points(field, src_lat_deg, src_lon_deg,
                               dst_lat_deg_pts, dst_lon_deg_pts):
    """Bilinear sample of one 2-D field at target points (see
    :func:`_bilinear_point_maps`)."""
    i4, j4, w4 = _bilinear_point_maps(src_lat_deg, src_lon_deg,
                                      dst_lat_deg_pts, dst_lon_deg_pts)
    f = np.asarray(field, dtype=np.float64)
    return (f[i4, j4] * w4).sum(axis=0)


# --- gap 13: bicubic, because ORCA1 remaps its WINDS bicubically ------------
#
# The docstring above says "NEMO interpolates CORE-II bilinearly (its weights
# files)". That is true of the SCALAR channels and NOT of the winds. From the
# run's own namelist_cfg:
#
#   sn_wndi / sn_wndj  (148-149)  weights_coreII_2_eORCA1.4.2_BICUBIC.nc
#   sn_qsr/qlw/tair    (150-152)  weights_coreII_2_eORCA1.4.2_BILINEAR.nc
#
# So a single uniform method cannot be faithful whatever it is set to: the
# oracle deliberately uses a higher-order remap for momentum and a linear one
# for the thermodynamic fields.
#
# The oracle's own weight FILES are not on this machine, so this reproduces
# NEMO's METHOD, not its exact weights. Catmull-Rom (a = -0.5) is the cubic
# convolution SCRIPS/NEMO's bicubic remapping is built on; it is interpolating
# (passes through the data) and reduces to the same stencil bilinear would use
# when the field is linear.
_BICUBIC_MAP_CACHE: dict = {}

# Catmull-Rom cubic convolution kernel, a = -0.5.
_CUBIC_A = -0.5


def _cubic_weights(t):
    """Cubic-convolution weights for the four points at offsets -1,0,1,2.

    ``t`` in [0,1) is the fractional position between points 0 and 1. The four
    weights sum to 1 for every t, which is what keeps a constant field exactly
    constant through the remap.
    """
    t = np.asarray(t, dtype=np.float64)
    a = _CUBIC_A
    t1 = 1.0 + t                       # distance to the -1 point
    t2 = t                             # to 0
    t3 = 1.0 - t                       # to 1
    t4 = 2.0 - t                       # to 2
    w1 = a * (t1 ** 3) - 5 * a * (t1 ** 2) + 8 * a * t1 - 4 * a
    w2 = (a + 2) * (t2 ** 3) - (a + 3) * (t2 ** 2) + 1.0
    w3 = (a + 2) * (t3 ** 3) - (a + 3) * (t3 ** 2) + 1.0
    w4 = a * (t4 ** 3) - 5 * a * (t4 ** 2) + 8 * a * t4 - 4 * a
    return w1, w2, w3, w4


def _bicubic_point_maps(src_lat_deg, src_lon_deg,
                        dst_lat_deg_pts, dst_lon_deg_pts):
    """16-point bicubic maps ``(i16, j16, w16)``, each ``(16, n_pts)``.

    Longitude is PERIODIC (the stencil wraps across the seam); latitude is
    CLAMPED at the source's outermost rows, matching what the bilinear map
    already does there — CORE-II stops ~0.5 deg short of the pole, and a cubic
    stencil that ran off the end would otherwise extrapolate into the gap.
    """
    src_lat = np.asarray(src_lat_deg, dtype=np.float64)
    src_lon = np.asarray(src_lon_deg, dtype=np.float64)
    # Full-content hash of EVERY coordinate array, via the collision-safe
    # helper this module already grew for precisely this bug: a key built from
    # endpoints alone can match two different destination grids and silently
    # hand one of them the other's weights (codex found my first version
    # repeating that August mistake).
    key = _coord_key(src_lat, src_lon,
                     np.asarray(dst_lat_deg_pts, dtype=np.float64),
                     np.asarray(dst_lon_deg_pts, dtype=np.float64))
    cached = _BICUBIC_MAP_CACHE.get(key)
    if cached is not None:
        return cached

    dlat = np.asarray(dst_lat_deg_pts, dtype=np.float64).ravel()
    dlon = np.asarray(dst_lon_deg_pts, dtype=np.float64).ravel()

    ascending = src_lat[1] > src_lat[0]
    lat_axis = src_lat if ascending else src_lat[::-1]
    n_lat = lat_axis.size

    # Bracketing row and fractional position, clamped to the interior.
    i1 = np.clip(np.searchsorted(lat_axis, dlat, side="right") - 1,
                 0, n_lat - 2)
    tlat = (dlat - lat_axis[i1]) / (lat_axis[i1 + 1] - lat_axis[i1])
    tlat = np.clip(tlat, 0.0, 1.0)
    rows = [np.clip(i1 + k, 0, n_lat - 1) for k in (-1, 0, 1, 2)]
    if not ascending:
        rows = [n_lat - 1 - r for r in rows]

    n_lon = src_lon.size
    lon0 = float(src_lon[0])
    dlon_step = float(src_lon[1] - src_lon[0])
    x = (dlon - lon0) / dlon_step
    j1 = np.floor(x).astype(int)
    tlon = x - j1
    cols = [np.mod(j1 + k, n_lon) for k in (-1, 0, 1, 2)]

    wlat = _cubic_weights(tlat)
    wlon = _cubic_weights(tlon)

    i16, j16, w16 = [], [], []
    for a_i in range(4):
        for b_j in range(4):
            i16.append(rows[a_i])
            j16.append(cols[b_j])
            w16.append(wlat[a_i] * wlon[b_j])
    maps = (np.asarray(i16), np.asarray(j16), np.asarray(w16))
    _BICUBIC_MAP_CACHE[key] = maps
    return maps


def _bicubic_interp_to_points(field, src_lat_deg, src_lon_deg,
                              dst_lat_deg_pts, dst_lon_deg_pts):
    """Bicubic sample of one 2-D field (see :func:`_bicubic_point_maps`)."""
    i16, j16, w16 = _bicubic_point_maps(src_lat_deg, src_lon_deg,
                                        dst_lat_deg_pts, dst_lon_deg_pts)
    f = np.asarray(field, dtype=np.float64)
    return (f[i16, j16] * w16).sum(axis=0)


# The oracle's per-field split: momentum bicubic, everything else bilinear.
_NEMO_BICUBIC_CHANNELS = ("u10", "v10")


# === gap 13, exact reproduction: read NEMO's OWN SCRIP weight files ==========
#
# The user chose exact reproduction over the method-equivalent Catmull-Rom
# above. The oracle's weight files ARE on this machine (an earlier claim that
# they were not was wrong and is retracted in the gap 13 commit):
#
#   cfgs/ORCA1/INPUTS/weights_coreII_2_eORCA1.4.2_bicubic.nc     (winds)
#   cfgs/ORCA1/INPUTS/weights_coreII_2_eORCA1.4.2_bilinear.nc    (scalars)
#   .../INPUTS/orca1_inputs/data_repository/input_fields/
#       weights_reg05_bilinear.nc                                (chlorophyll)
#
# The copies sitting in EXP00/RUN_GATEWAY* are BROKEN SYMLINKS; use INPUTS.
#
# CONVENTIONS, every one of them MEASURED before any of this was written
# (probe jobs recorded in scripts/cluster/omip_nemo/_scrip_*.sbatch):
#   * variables srcNN / dstNN / wgtNN; destination shape (331, 360).
#   * src holds 1-BASED FLAT indices in C order (lat-major) into the 94x192
#     CORE-II grid, stored as FLOAT -- they must be cast before use.
#   * the source latitude axis is used AS-IS (ascending). Proven with
#     ANTISYMMETRIC test fields: a symmetric field such as cos(lat) cannot
#     detect a latitude flip at all, which is how the first probe missed it.
#   * the destination is the INTERIOR of our (332, 362) mesh, python
#     [0:331, 1:361]. That is not inferred from arithmetic -- it is in the
#     file's own history attribute, `ncks -F -d lon,2,361 -d lat,1,331`,
#     i.e. drop both cyclic-overlap columns and the north-fold row.
#   * ACCURACY: over the 113761 full 4-point stencils (95.5% of points) a
#     LINEAR field is reproduced to a median 2.18e-08. The residual lives in
#     5399 partial 2-point stencils and in 180 points in the ten northernmost
#     rows, i.e. the tripolar fold.
#
# CAVEAT worth carrying: the files are named for eORCA1.4.2 while our mesh is
# eORCA1.2. They are DIMENSIONALLY identical, which is why the indices line up,
# but the coastline/land mask may differ in detail.
_SCRIP_CACHE: dict = {}


def load_scrip_weights(path):
    """Read a NEMO/SCRIP remapping weights file.

    Returns ``(src0, wgt, n)`` with ``src0`` the ZERO-based flat source indices
    and ``wgt`` the raw weights, both ``(n, ny, nx)``. ``n`` is 4 for a
    bilinear file and 16 for a bicubic one, and that count is what selects how
    the weights are APPLIED -- see :func:`apply_scrip_weights`.
    """
    import xarray as xr
    key = str(path)
    cached = _SCRIP_CACHE.get(key)
    if cached is not None:
        return cached
    with xr.open_dataset(path, decode_times=False) as d:
        n = sum(1 for v in d.variables if str(v).startswith("src"))
        if n not in (4, 16):
            raise ValueError(
                f"{path}: expected 4 (bilinear) or 16 (bicubic) weight "
                f"triples, found {n}")
        src = np.stack([np.asarray(d[f"src{k:02d}"]) for k in range(1, n + 1)])
        wgt = np.stack([np.asarray(d[f"wgt{k:02d}"]) for k in range(1, n + 1)])
    out = (src.astype(np.int64) - 1, np.asarray(wgt, dtype=np.float64), n)
    _SCRIP_CACHE[key] = out
    return out


def _pad_source(field):
    """One-cell halo: PERIODIC in longitude, edge-replicated in latitude.

    NEMO builds the same padded array (``fly_dta``) so its bicubic stencil can
    reach ni-1 and ni+1 around every corner. Longitude wraps because the
    CORE-II grid is global; latitude replicates because it is not.
    """
    f = np.asarray(field, dtype=np.float64)
    # Longitude: PERIODIC. NEMO first replicates the east-west edges and then,
    # "if data grid is cyclic we can do better on east-west edges"
    # (fldread.F90:1503), re-reads the wrap columns. CORE-II is a cyclic global
    # grid, so the cyclic branch is the one that applies.
    f = np.pad(f, ((0, 0), (1, 1)), mode="wrap")
    # Latitude is NOT symmetric in NEMO, and this is easy to get wrong:
    #   south edge  fly_dta(:,jpj1-1) = fly_dta(:,jpj1)            REPLICATE
    #   north edge  fly_dta(:,jpj2+1) = 2*fly_dta(:,jpj2)
    #                                   - fly_dta(:,jpj2-1)        EXTRAPOLATE
    # (fldread.F90:1495-1499). Replicating BOTH ends -- which is what this
    # function did first -- flattens the gradient in the northernmost source
    # row and therefore perturbs the Arctic derivative stencils.
    south = f[:1]
    north = 2.0 * f[-1:] - f[-2:-1]
    return np.concatenate([south, f, north], axis=0)


def apply_scrip_weights(field, src0, wgt, n, src_shape):
    """Remap one 2-D source field through NEMO's own weights.

    For a BILINEAR file (n=4) this is the obvious weighted sum. For a BICUBIC
    file (n=16) IT IS NOT: NEMO's fld_interp (fldread.F90:1543-1574) uses the
    first four weights on the VALUES and the other twelve on CENTRED
    DERIVATIVES of the source field --

        5-8   x  0.5  * (f[i+1, j] - f[i-1, j])          d/di
        9-12  x  0.5  * (f[i, j+1] - f[i, j-1])          d/dj
        13-16 x  0.25 * ((f[i+1,j+1] - f[i-1,j+1])
                         - (f[i+1,j-1] - f[i-1,j-1]))    cross

    -- which is why those sixteen weights do NOT sum to one (measured range
    0.8168 to 1.2017). Treating them as value weights would produce a
    plausible, wrong field rather than an error.
    """
    ny_s, nx_s = src_shape
    p = _pad_source(np.asarray(field).reshape(ny_s, nx_s))
    # ALL FOUR weight groups address the SAME four corners: NEMO reuses
    # data_jpi/data_jpj across its four `DO jn = 1,4` loops and only the weight
    # slice changes. So the geometry comes from the first four triples, and
    # triples 5-16 must repeat those same source indices -- asserted here
    # rather than assumed, because if they did not this whole reading of the
    # file would be wrong.
    corners = src0[:4]
    if n == 16 and not np.array_equal(np.tile(corners, (4, 1, 1)), src0):
        raise ValueError(
            "bicubic weights do not repeat the same four source corners "
            "across their four weight groups; the file layout is not what "
            "fld_interp assumes")
    j = corners // nx_s
    i = corners - j * nx_s
    # Corner value in padded coordinates.
    val = p[j + 1, i + 1]
    if n == 4:
        return (val * wgt).sum(axis=0)
    out = (val * wgt[:4]).sum(axis=0)
    d_di = 0.5 * (p[j + 1, i + 2] - p[j + 1, i])
    d_dj = 0.5 * (p[j + 2, i + 1] - p[j, i + 1])
    cross = 0.25 * ((p[j + 2, i + 2] - p[j + 2, i])
                    - (p[j, i + 2] - p[j, i]))
    out = out + (d_di * wgt[4:8]).sum(axis=0)
    out = out + (d_dj * wgt[8:12]).sum(axis=0)
    out = out + (cross * wgt[12:16]).sum(axis=0)
    return out


# Cache of pre-computed conservative-regrid weights, keyed by
# (src_lat_shape, src_lon_shape, src_lat_first, src_lon_first,
#  dst_lat_shape, dst_lon_shape, dst_lat_first, dst_lon_first).
_REGRID_WEIGHTS_CACHE: dict = {}

# Real forcing stops short of the pole (CORE-II NYF's inferred outer edge is
# +-89.486 deg, a 0.514 deg gap), so the polar destination row is partly covered --
# or, once the target is finer than the gap, not covered at all.  Under the default
# 'dstarea' normalisation that returns coverage x field: measured outer-row coverage
# for CORE-II is 0.736 at the production 1 deg (a 250 K air temperature arriving as
# 184 K) and exactly 0 at 0.5 deg and finer.  Every channel on this path is
# INTENSIVE -- T_air, q_air, slp, winds, and the radiative/precip flux DENSITIES --
# and the shortfall is a DATA GAP, not a region of zero flux, so the correct
# treatment is the field's own value, not a diluted one:
#   * 'fracarea' renormalises a partly covered row to the area-weighted mean of the
#     source that does overlap;
#   * polar_fill gives a row beyond the source's band its outermost row, zonally
#     resolved (a zeroth-order poleward extrapolation).
# Together they make every destination cell sum to 1, so the coverage check is back
# to the strict invariant and a real seam/ghost deficit still raises.  Both
# deliberately trade strict global conservation for correct magnitude; that trade is
# right here and WRONG for flux coupling, which is why coupler/grid_remap.py keeps
# 'dstarea'.  See regrid_polar_coverage_2026-07-24.md.
_FORCING_NORMALIZATION = "fracarea"


def _edges_from_centers_deg(centers_deg):
    """Derive uniform-spaced cell edges (radians) from cell centres.

    Assumes the centres are uniformly spaced.  The 0/360 seam is closed by the
    ghost columns in :func:`_conservative_regrid_to_latlon`, not here.
    """
    c = np.asarray(centers_deg, dtype=np.float64)
    if c.size < 2:
        raise ValueError("Need >= 2 cell centres to infer edges.")
    dc = float(c[1] - c[0])
    edges = np.empty(c.size + 1, dtype=np.float64)
    edges[:-1] = c - 0.5 * dc
    edges[-1] = c[-1] + 0.5 * dc
    return np.radians(edges)


def _conservative_regrid_to_latlon(
    field_2d, src_lat_deg, src_lon_deg, dst_lat_deg, dst_lon_deg,
):
    """Conservatively regrid a 2-D ``(n_src_lat, n_src_lon)`` field to a
    regular lat-lon model grid.

    Builds + caches the overlap weights on first call for each
    (src_shape, dst_shape) pair; subsequent calls are a sparse matmul.
    """
    from legoesm.grids.conservative_regrid import (
        check_axis_span, compute_overlap_weights, apply_conservative_regrid,
    )
    import jax.numpy as jnp_local
    key = (
        np.asarray(src_lat_deg).shape,
        np.asarray(src_lon_deg).shape,
        float(src_lat_deg[0]), float(src_lon_deg[0]),
        np.asarray(dst_lat_deg).shape,
        np.asarray(dst_lon_deg).shape,
        float(dst_lat_deg[0]), float(dst_lon_deg[0]),
    )
    if key not in _REGRID_WEIGHTS_CACHE:
        # LONGITUDE WRAP: pad the source with one ghost column on each side
        # (last column shifted -360, first column shifted +360) so a
        # destination cell straddling the 0/360 seam sees full source
        # coverage.  Without this the seam-adjacent destination column was
        # only partially covered (under-weighted forcing stripe at the last
        # longitude; with the coarse synthetic test forcing the column came
        # back HALVED, and the 10-m pressure iteration then NaN'd on the
        # resulting garbage air temperature).
        src_lon = np.asarray(src_lon_deg, dtype=np.float64)
        n_src_lon = src_lon.size
        # PRECONDITION the wrap-pad relies on, checked BEFORE padding: the RAW
        # source must tile the full 360 deg.  The +-360 ghosts below would turn a
        # partial-longitude source into one enormous cell spanning the whole
        # missing sector, which then reports COMPLETE longitude coverage to the
        # weight builder -- so this is the only point where the difference is
        # still visible.
        check_axis_span(_edges_from_centers_deg(src_lon), 2.0 * np.pi,
                        name="omip2 forcing source longitude")
        ds = abs(float(src_lon[1] - src_lon[0]))
        dd = abs(float(np.asarray(dst_lon_deg)[1] - np.asarray(dst_lon_deg)[0]))
        n_ghost = max(1, int(np.ceil(dd / ds)))
        n_ghost = min(n_ghost, n_src_lon)
        src_lon_padded = np.concatenate(
            [src_lon[-n_ghost:] - 360.0, src_lon, src_lon[:n_ghost] + 360.0])
        src_lat_edges = _edges_from_centers_deg(src_lat_deg)
        src_lon_edges = _edges_from_centers_deg(src_lon_padded)
        dst_lat_edges = _edges_from_centers_deg(dst_lat_deg)
        dst_lon_edges = _edges_from_centers_deg(dst_lon_deg)
        # Clamp lat edges into [-pi/2, pi/2] in case the inferred edge
        # spills over the pole due to rounding.
        src_lat_edges = np.clip(src_lat_edges, -np.pi / 2, np.pi / 2)
        dst_lat_edges = np.clip(dst_lat_edges, -np.pi / 2, np.pi / 2)
        # fracarea + polar_fill treat the physical polar gap (see the constant's
        # rationale above), after which every destination cell sums to 1 and
        # require_full_coverage is the STRICT invariant again -- so a longitude
        # seam/ghost deficit still raises.  That is the defect worth guarding: a
        # single ghost once left the seam column HALVED and the 10-m pressure
        # iteration NaN'd on it.  See regrid_polar_coverage_2026-07-24.md
        _REGRID_WEIGHTS_CACHE[key] = compute_overlap_weights(
            src_lat_edges, src_lon_edges,
            dst_lat_edges, dst_lon_edges,
            require_full_coverage=True,
            normalization=_FORCING_NORMALIZATION,
            polar_fill=True,
        )
    weights = _REGRID_WEIGHTS_CACHE[key]
    # Ghost-column count matches the weight build below (width ratio, NOT
    # count ratio: this caller's dst may be a sub-global regional grid where
    # count ratio over-estimates). Clamp so slicing can't over-wrap the src.
    src_lon = np.asarray(src_lon_deg, dtype=np.float64)
    n_src_lon = src_lon.size
    ds = abs(float(src_lon[1] - src_lon[0]))
    dd = abs(float(np.asarray(dst_lon_deg)[1] - np.asarray(dst_lon_deg)[0]))
    n_ghost = max(1, int(np.ceil(dd / ds)))
    n_ghost = min(n_ghost, n_src_lon)
    f = jnp_local.asarray(field_2d)
    f_padded = jnp_local.concatenate(
        [f[:, -n_ghost:], f, f[:, :n_ghost]], axis=1)
    return np.asarray(apply_conservative_regrid(f_padded, weights))


_BASE_FORCING_CHANNELS = ("u10", "v10", "T_air", "q_air",
                          "sw_down", "lw_down", "precip")
# Channels added for NEMO-parity surface fluxes (2026-06): solid
# precipitation (snow-fusion + snow heat-content terms of q_ns) and
# sea-level pressure (moist-air density + Goff saturation humidity).
# OPTIONAL: forcing sets built before the schema extension lack them and
# the flux assembly falls back to snow=0 / slp=standard atmosphere.
_OPTIONAL_FORCING_CHANNELS = ("snow", "slp")


def _forcing_channels(forcing):
    """Base channels + whichever optional channels ``forcing`` carries."""
    names = list(_BASE_FORCING_CHANNELS)
    for name in _OPTIONAL_FORCING_CHANNELS:
        if getattr(forcing, name, None) is not None:
            names.append(name)
    return tuple(names)


def _sample_forcing_latlon(forcing, idx_t, dst_lat_deg, dst_lon_deg):
    """Conservative-regrid the forcing channels at time ``idx_t`` onto a
    regular destination lat-lon grid."""
    out = {}
    for name in _forcing_channels(forcing):
        out[name] = _conservative_regrid_to_latlon(
            getattr(forcing, name)[idx_t],
            forcing.lat, forcing.lon,
            dst_lat_deg, dst_lon_deg,
        )
    return out


def _sample_forcing_points(forcing, idx_t, lat_pts_deg, lon_pts_deg,
                           method: str = "nearest"):
    """Sample the forcing channels at a set of points (cube / MPAS / tripole
    cell centres).  ``method``: "nearest" (legacy) or "bilinear" (the tripole
    default — see :func:`_bilinear_point_maps` for why nearest-neighbour onto
    a finer structured grid prints zonal forcing bands)."""
    if method not in ("nearest", "bilinear", "nemo_weights"):
        raise ValueError(f"_sample_forcing_points: unknown method {method!r}")
    _uniform = {"bilinear": _bilinear_interp_to_points,
                "nearest": _nn_interp_to_points}.get(method)
    out = {}
    for name in _forcing_channels(forcing):
        if method == "nemo_weights":
            # PER-CHANNEL, because the oracle is per-channel: the winds carry
            # bicubic weights files and every other channel carries bilinear
            # ones. A uniform method cannot reproduce that whichever one it
            # picks.
            interp = (_bicubic_interp_to_points
                      if name in _NEMO_BICUBIC_CHANNELS
                      else _bilinear_interp_to_points)
        else:
            interp = _uniform
        out[name] = interp(
            getattr(forcing, name)[idx_t],
            forcing.lat, forcing.lon,
            lat_pts_deg, lon_pts_deg,
        )
    return out


def scrip_interior_to_full_tripole(interior, ny=332, nx=362):
    """Place a (331, 360) SCRIP result into a full (ny, nx) tripole field.

    The oracle's weight files cover the INTERIOR only -- their own history
    attribute records `ncks -F -d lon,2,361 -d lat,1,331`. The entries they
    drop are not missing data, they are the grid's own redundancy, so they are
    FILLED rather than left as zeros; a silent zero in a forcing field is the
    failure mode this repo keeps hitting.

      * columns 0 and 361 are the CYCLIC OVERLAP of columns 360 and 1;
      * row 331 is the NORTH-FOLD row, filled from the row below.

    The fold fill is an APPROXIMATION: a true eORCA fold maps the row onto
    itself with a reversal. For a smooth surface forcing field at a single row
    the difference is small, and this is stated rather than hidden.
    """
    interior = np.asarray(interior, dtype=np.float64)
    if interior.shape != (ny - 1, nx - 2):
        raise ValueError(
            f"SCRIP interior {interior.shape} does not fit a ({ny}, {nx}) "
            f"tripole; expected {(ny - 1, nx - 2)}")
    out = np.zeros((ny, nx), dtype=np.float64)
    out[0:ny - 1, 1:nx - 1] = interior
    out[0:ny - 1, 0] = out[0:ny - 1, nx - 2]
    out[0:ny - 1, nx - 1] = out[0:ny - 1, 1]
    out[ny - 1, :] = out[ny - 2, :]
    return out


# ORCA1's own weight files, per sn_* in namelist_cfg. The RUN_GATEWAY copies
# are BROKEN SYMLINKS; these INPUTS paths are the real files.
_ORACLE_WEIGHTS_DIR = ("/burg-archive/glab/users/pg2328/nemo_orca1/"
                       "nemo_5.0.1/cfgs/ORCA1/INPUTS/")
_CORE2_BICUBIC = _ORACLE_WEIGHTS_DIR + "weights_coreII_2_eORCA1.4.2_bicubic.nc"
_CORE2_BILINEAR = _ORACLE_WEIGHTS_DIR + "weights_coreII_2_eORCA1.4.2_bilinear.nc"


def sample_forcing_tripole_scrip(forcing, idx_t):
    """Remap every forcing channel onto the eORCA1 tripole with ORACLE WEIGHTS.

    Per-channel, because the oracle is per-channel: sn_wndi/sn_wndj name the
    BICUBIC weights file (namelist_cfg:148-149) and every other channel names
    the BILINEAR one (150-152). Returns full (332, 362) fields.

    This is a GRID-level routine, deliberately not a `method` on
    :func:`_sample_forcing_points`: that function samples arbitrary POINT
    lists, whereas a weights file encodes one fixed destination grid, and
    pretending otherwise would let it be called for a grid it cannot serve.
    """
    out = {}
    for name in _forcing_channels(forcing):
        path = (_CORE2_BICUBIC if name in _NEMO_BICUBIC_CHANNELS
                else _CORE2_BILINEAR)
        src0, wgt, n = load_scrip_weights(path)
        field = np.asarray(getattr(forcing, name)[idx_t], dtype=np.float64)
        interior = apply_scrip_weights(field, src0, wgt, n, field.shape)
        out[name] = scrip_interior_to_full_tripole(interior)
    return out


def _sample_omip2_forcing(forcing, idx_t, grid, grid_type,
                          forcing_remap="bilinear"):
    """Sample the CORE-II/JRA forcing channels onto the model grid for a record.

    Shared by :func:`compute_omip2_surface_forcing` (momentum/heat) and
    :func:`compute_omip2_freshwater_forcing` (P - E) so both see the SAME
    spatially-sampled fields.  ``latlon``/``latlon_regional`` use the
    conservative regrid onto the regular T grid; ``tripole``/``cubed_sphere``/
    ``mpas`` use nearest-neighbour onto the (possibly 2-D / 1-D) cell centres.
    """
    if forcing_remap not in ("bilinear", "nemo_scrip"):
        raise ValueError(
            f"unknown forcing_remap {forcing_remap!r}; expected 'bilinear' "
            f"or 'nemo_scrip'")
    if forcing_remap == "nemo_scrip" and grid_type != "tripole":
        raise ValueError(
            f"forcing_remap='nemo_scrip' is wired for the tripole only; the "
            f"oracle's weights target that grid. Got grid_type={grid_type!r}.")
    if grid_type in ("latlon", "latlon_regional"):
        lat_deg = np.degrees(np.asarray(grid.lat))
        lon_deg = np.degrees(np.asarray(grid.lon))
        return _sample_forcing_latlon(forcing, idx_t, lat_deg, lon_deg)
    if grid_type == "tripole":
        lat_pts = np.degrees(np.asarray(grid.lat_T))
        lon_pts = np.degrees(np.asarray(grid.lon_T))
        shp = lat_pts.shape
        if forcing_remap == "nemo_scrip":
            if shp != (332, 362):
                raise ValueError(
                    f"forcing_remap='nemo_scrip' needs the (332, 362) eORCA1 "
                    f"tripole; this grid is {shp}. The weight files encode one "
                    f"destination grid and remapping another through them "
                    f"would give a plausible wrong field, not an error.")
            return sample_forcing_tripole_scrip(forcing, idx_t)
        forc = _sample_forcing_points(
            forcing, idx_t, lat_pts.reshape(-1), lon_pts.reshape(-1), method="bilinear",
        )
        return {k: v.reshape(shp) for k, v in forc.items()}
    if grid_type == "cubed_sphere":
        lat_pts = np.degrees(np.asarray(grid.lat))
        lon_pts = np.degrees(np.asarray(grid.lon))
        shp = lat_pts.shape
        forc = _sample_forcing_points(
            forcing, idx_t, lat_pts.reshape(-1), lon_pts.reshape(-1),
        )
        return {k: v.reshape(shp) for k, v in forc.items()}
    if grid_type in ("mpas", "mpas_regional"):
        lat_pts = np.degrees(np.asarray(grid.latCell))
        lon_pts = np.degrees(np.asarray(grid.lonCell))
        return _sample_forcing_points(forcing, idx_t, lat_pts, lon_pts)
    if grid_type == "fesom":
        # FESOM triangular mesh: 1-D PAIRED node (lat, lon) points (radians
        # on the FesomOceanGrid facade) — same nearest-neighbour treatment
        # as the MPAS Voronoi cell centres.
        lat_pts = np.degrees(np.asarray(grid.lat))
        lon_pts = np.degrees(np.asarray(grid.lon))
        return _sample_forcing_points(forcing, idx_t, lat_pts, lon_pts)
    raise NotImplementedError(
        f"_sample_omip2_forcing does not support grid_type={grid_type!r}"
    )


def sample_omip2_forcing(forcing, idx_t, grid, grid_type):
    """Public wrapper for :func:`_sample_omip2_forcing`.

    Returns the CORE-II / JRA forcing channels (``u10``, ``v10``, ``T_air`` [K],
    ``q_air``, ``sw_down``, ``lw_down``, ``precip``, and the optional ``snow`` /
    ``slp``) spatially sampled onto the model grid for one time record — the
    SAME sampling :func:`compute_omip2_surface_forcing` and
    :func:`compute_omip2_freshwater_forcing` consume.  Exposed so an external
    coupler driver (e.g. the prognostic sea-ice glue in
    ``scripts/run/run_omip_core2.py``) can populate an ``AtmToSurface`` from the
    identical sampled fields rather than re-deriving the regrid (cross-module
    callers must not import the private ``_sample_omip2_forcing``)."""
    return _sample_omip2_forcing(forcing, idx_t, grid, grid_type)


def compute_omip2_freshwater_forcing(state, *, forcing, idx_t: int,
                                     grid, grid_type: str,
                                     runoff_R=None,
                                     emp: bool = True,
                                     ramp: float = 1.0,
                                     rho_air: float = constants.rho_air,
                                     u_oce=None, v_oce=None,
                                     wind_current_feedback_vfac: float = 0.0,
                                     forcing_remap: str = "bilinear"):
    """Build a :class:`FreshwaterForcing` (P, E, runoff) for the OMIP-2 run.

    Delivered to the ocean via the in-core channel
    ``model.step(state, dt, surface_forcing=sf, freshwater=fw)`` so the surface
    freshwater enters the SAME tested path the model uses for its own freshwater:
    the virtual-salt tendency ``dS_top/dt = -S_ref * F_fw / (rho_0 * dz_0)`` with
    ``config.S_ref`` AND the free-surface source ``deta/dt = F_fw/rho_0`` inside
    the barotropic solve (NOT an operator-split post-step NumPy update, which
    would bypass the eta solve, break AD/JIT device-residency, and double-apply
    on the cube's 'external' physics that already consumes
    ``surface_forcing.freshwater``).

    Components (all [kg/m^2/s], + INTO ocean):
    * ``precip`` -- prescribed CORE-II precipitation (rain+snow).
    * ``evap``   -- INTERACTIVE evaporation, positive UP, back-derived from the
      latent heat flux (``air_sea_fluxes`` returns ``lh`` +INTO ocean, so an
      evaporating column has lh<0 and E = -lh/L_v >= 0).  ``net_freshwater_flux``
      forms ``precip - evap + runoff`` => ``precip + lh/L_v + runoff``.
    * ``runoff`` -- optional Dai-Trenberth river/ice-shelf/iceberg field.

    Without P - E the ocean only sees runoff (a one-sided freshwater SOURCE) and
    freshens ~0.3 PSU/yr -- the multi-year drift that strong SSS restoring was
    masking by injecting net salt (-> AMOC suppression).

    Parameters
    ----------
    emp : bool
        When False, zero the P - E contribution (ablation / runoff-only).
    ramp : float
        Cold-start spin-up scale in [0, 1] applied to ALL freshwater components
        (matches the tau/q_net ramp).
    u_oce, v_oce, wind_current_feedback_vfac :
        NEMO ``ln_crt_dwn`` / ``rn_vfac`` relative-wind current feedback, passed
        straight through to :func:`air_sea_fluxes` (geographic-frame currents;
        see :func:`compute_omip2_surface_forcing`).  MUST match the ``vfac`` /
        currents used for the heat + momentum forcing so the evaporative MASS
        flux ``E`` (P - E salinity) stays the SAME physical flux as the latent
        HEAT flux in ``q_net`` (``E = -lhflx / L_vap``).  ``vfac == 0.0``
        (default) is byte-identical.
    forcing_remap :
        The atmosphere-to-ocean sampling, as for
        :func:`compute_omip2_surface_forcing`; MUST be the same value so P and
        E here are the same sampled fields as the heat flux's (review
        2026-10-10 F19: this builder ignored ``nemo_scrip``).
    """
    from legoesm.ocean.freshwater import FreshwaterForcing

    forc = _sample_omip2_forcing(forcing, idx_t, grid, grid_type,
                                 forcing_remap=forcing_remap)
    # MPAS state is (nCells,) at the surface; cube/latlon/tripole are (..., 0).
    # Slice FIRST (device-side), THEN convert: np.asarray on the full leaf
    # would assemble the entire 3-D (possibly lat-band-sharded) T field on
    # the host every step (codex batch4 HIGH); the [..., 0] child moves only
    # the 2-D surface layer.  Value-identical — slicing commutes with the
    # elementwise f64 upcast.  Counted in the module host-pull ledger.
    _record_surface_slice_pull()
    T_sfc_K = np.asarray(state.T.data[..., 0], dtype=np.float64) + float(constants.T_freeze)
    slp = forc.get("slp")
    # NCAR algo (NEMO-faithful): q_sfc = 0.98*q_sat_goff(SST, slp) computed
    # internally; evap returned directly (= -lh / L_vap(SST), consistent with
    # the latent flux by construction -- no separate constants.L_v division).
    _, _, _, _, evap_j = air_sea_fluxes(
        u10=jnp.asarray(forc["u10"]), v10=jnp.asarray(forc["v10"]),
        T_air_K=jnp.asarray(forc["T_air"]), q_air=jnp.asarray(forc["q_air"]),
        T_sfc_K=jnp.asarray(T_sfc_K),
        slp_Pa=None if slp is None else jnp.asarray(slp),
        u_oce=u_oce, v_oce=v_oce, vfac=wind_current_feedback_vfac,
    )
    if emp:
        precip = np.asarray(forc["precip"], dtype=np.float64)
        evap = np.asarray(evap_j, dtype=np.float64)   # positive UP
    else:
        precip = np.zeros_like(np.asarray(forc["precip"], dtype=np.float64))
        evap = np.zeros_like(precip)
    if runoff_R is not None:
        runoff = np.asarray(runoff_R, dtype=np.float64)
    else:
        runoff = np.zeros_like(precip)
    r = float(ramp)
    z = jnp.zeros_like(jnp.asarray(precip))
    return FreshwaterForcing(
        precip=jnp.asarray(precip * r),
        evap=jnp.asarray(evap * r),
        runoff=jnp.asarray(runoff * r),
        ice_fw=z,
        restoring=z,
    )


def dm2dc_sw_factor(grid, dm2dc_window):
    """Analytic diurnal-cycle shortwave factor field (NEMO ln_dm2dc / sbcdcy,
    Bernie et al. 2007) for one step window, on the model grid's T points.

    ``dm2dc_window = (day_of_year, year_len_days, t_frac_lo, t_frac_up)``.
    Mean-preserving over a day by construction.  ONE shared implementation so
    every consumer of the CORE-II daily-mean SW (the ocean surface forcing AND
    the prognostic sea-ice AtmToSurface) sees the SAME diurnal modulation —
    codex r1 #2: the ice tile previously received the raw daily-mean SW while
    the ocean saw the modulated one.  Lat-lon / tripole C-grid families and the
    MPAS Voronoi mesh (matches the run drivers' --dm2dc gate)."""
    from legoesm.ocean.forcing.diurnal_cycle import diurnal_sw_factor
    _day_of_year, _year_len, _t_lo, _t_up = dm2dc_window
    _lat_T = getattr(grid, "lat_T", None)
    _lat_cell = getattr(grid, "latCell", None)
    if _lat_T is not None:            # tripole family (2-D, radians)
        _lat_deg = np.degrees(np.asarray(_lat_T))
        _lon_deg = np.degrees(np.asarray(grid.lon_T))
    elif _lat_cell is not None:       # MPAS Voronoi mesh: 1-D PAIRED cell
        # centres (radians) -- each cell carries its OWN (lat, lon), so pass the
        # 1-D arrays directly, NOT a meshgrid.  diurnal_sw_factor is elementwise,
        # so this returns the (nCells,) per-cell factor, matching the sw_down /
        # q_net fields the MPAS lane modulates.
        _lat_deg = np.degrees(np.asarray(_lat_cell))
        _lon_deg = np.degrees(np.asarray(grid.lonCell))
    else:                             # regular lat-lon (1-D, radians)
        _lat_deg = np.degrees(np.asarray(grid.lat))[:, None]
        _lon_deg = np.degrees(np.asarray(grid.lon))[None, :]
    return np.asarray(diurnal_sw_factor(
        _lon_deg, _lat_deg,
        day_of_year=_day_of_year, year_len_days=_year_len,
        t_frac_lo=_t_lo, t_frac_up=_t_up))


def _sw_albedo_factor(sw_down, ice_albedo):
    """Reduce downwelling SW by the effective surface albedo.

    ``albedo_eff = alpha_ocean*(1-siconc) + alpha_ice*siconc`` where ``siconc``
    (= ``ice_albedo`` arg, the prescribed sea-ice concentration in [0,1]) weights
    open-ocean vs sea-ice broadband albedo.  Returns ``sw_down*(1-albedo_eff)``.
    ``ice_albedo=None`` -> unchanged ``sw_down`` (bit-exact default).  This is the
    ONLY place SW albedo is applied; the cores' 0.94 penetration/surface split is
    a vertical-distribution split (NOT an albedo) and is left untouched, so the
    albedoed ``sw_net`` flows once into both ``q_net`` and the penetrating SW.
    """
    if ice_albedo is None:
        return sw_down
    a_oc = float(constants.alpha_ocean_broadband)
    a_ice = float(constants.alpha_ice_broadband_cold)
    # Defensive clip: the loader already returns siconc in [0,1], but guard the
    # API contract (albedo_eff in [0,1], 0<=sw_net<=sw_down) against a NaN /
    # out-of-range caller (codex) so a bad siconc can never amplify SW.
    sic = np.clip(np.nan_to_num(np.asarray(ice_albedo, dtype=np.float64),
                                nan=0.0), 0.0, 1.0)
    albedo_eff = a_oc * (1.0 - sic) + a_ice * sic
    return np.asarray(sw_down, dtype=np.float64) * (1.0 - albedo_eff)


def _ice_surface_heat(sw_down, q_non_sw, ice_albedo, *,
                      under_ice: bool = False, tau_ice_sw: float = 0.03):
    """Combine SW + non-SW surface heat into ``(sw_ocean, q_net)`` [W/m², +into
    ocean] with an optional PRESCRIBED-ICE thermodynamic boundary (codex HIGH).

    Three regimes (``ice_albedo`` = prescribed siconc in [0,1], or ``None``):

    * ``ice_albedo is None`` -> NO albedo (bit-exact legacy default):
      ``sw_ocean = sw_down``; ``q_net = q_non_sw + sw_down``.
    * ``under_ice=False`` (albedo-only surrogate): ``sw_ocean`` = open/ice albedo-
      weighted SW (:func:`_sw_albedo_factor`); full open-ocean ``q_non_sw``.
    * ``under_ice=True`` (prescribed-ice boundary): under sea ice almost no SW
      reaches the ocean and the open-ocean turbulent+LW fluxes do not act on the
      ice-covered fraction.  Per-cell ice fraction ``sic`` blends open water and
      ice::

          sw_ocean = (1-sic)·sw_down·(1-α_ocean) + sic·sw_down·τ_ice_sw
          q_net    = sw_ocean + (1-sic)·q_non_sw

      ``τ_ice_sw`` (~0.03) is the small SW transmittance through ice/snow into the
      ocean (vs the albedo-only surrogate's ``1-α_ice=0.35``, which over-warms the
      under-ice ocean -- the Southern-Ocean warm bias).  At ``sic=0`` this equals
      the albedo-only open-water value, so only ice-covered cells change.  The
      under-ice relaxation toward the freezing point is applied as a post-step
      state nudge (:func:`under_ice_freeze_relax`), NOT here, so it needs no
      top-layer thickness in this flux producer."""
    q_non_sw = np.asarray(q_non_sw, dtype=np.float64)
    if not under_ice or ice_albedo is None:
        sw_ocean = _sw_albedo_factor(sw_down, ice_albedo)
        return sw_ocean, q_non_sw + sw_ocean
    a_oc = float(constants.alpha_ocean_broadband)
    sic = np.clip(np.nan_to_num(np.asarray(ice_albedo, dtype=np.float64),
                                nan=0.0), 0.0, 1.0)
    swd = np.asarray(sw_down, dtype=np.float64)
    sw_open = swd * (1.0 - a_oc)
    sw_ice = swd * float(tau_ice_sw)
    sw_ocean = (1.0 - sic) * sw_open + sic * sw_ice
    q_net = sw_ocean + (1.0 - sic) * q_non_sw
    return sw_ocean, q_net


def under_ice_freeze_relax(T_top_C, ice_concentration, dt: float, *,
                           tau_ice_days: float = 20.0, T_freeze_C=None,
                           S_top=None, freeze_scheme: str = "constant"):
    """Relax the under-ice surface ocean temperature toward the freezing point
    (prescribed-ice thermodynamic boundary; codex HIGH).  Returns the updated
    top-cell temperature [°C].

        T_top <- T_top + (dt/τ_ice)·sic·(T_freeze - T_top)

    The ice-ocean interface sits at the freezing point, so under prescribed ice
    the surface ocean is nudged toward ``T_freeze`` with timescale ``τ_ice`` (days)
    weighted by the ice fraction ``sic``.  TWO-SIDED: it COOLS a too-warm under-ice
    cell (the Southern-Ocean / Antarctic warm bias) and HOLDS a cold Arctic cell
    near freezing, while ``sic=0`` open water is untouched -> leaves the seasonal-
    albedo NH-summer warming intact.  Intentionally non-conservative for the ocean
    alone (the missing reservoir is the prescribed ice's latent heat).  Host-loop
    helper (NumPy; the ``--ice-thermo`` host path, NOT the lax.scan path -- scan
    refuses it).  A convex relaxation, unconditionally stable: ``dt/τ`` is clipped
    to <=1 so the update never overshoots ``T_freeze``.

    Freezing-point target (MED-1 follow-up): ``T_freeze_C`` may be a SCALAR or a
    PER-CELL array [°C].  With ``freeze_scheme != "constant"`` the target is
    computed per cell from the LOCAL surface salinity ``S_top`` [PSU] via the
    liquidus ``eos.freezing_point(S_top, scheme)`` (KELVIN, so the 0 °C
    reference ``constants.T_freeze`` is subtracted) — the Arctic-relevant path:
    fresher shelf water relaxes toward a warmer freezing point, saline water
    colder (~-1.92 °C at S=35, not -1.8).  ``freeze_scheme="constant"``
    (default) with ``T_freeze_C=None`` keeps the historical fixed
    ``T_freeze_ocean - T_freeze`` scalar byte-identical.  Passing BOTH an
    explicit ``T_freeze_C`` and a liquidus scheme raises (ambiguous), as does a
    liquidus scheme without ``S_top`` (never a silent constant fallback)."""
    # Dispatch hardening: validate the static selector at function entry
    # against the single source of truth (eos.VALID_FREEZE_SCHEMES) so a typo
    # can never silently run the constant path.
    if freeze_scheme not in VALID_FREEZE_SCHEMES:
        raise ValueError(
            f"Unknown freezing-point scheme: {freeze_scheme!r}. "
            f"Valid schemes: {sorted(VALID_FREEZE_SCHEMES)}."
        )
    if freeze_scheme != "constant":
        if T_freeze_C is not None:
            raise ValueError(
                "under_ice_freeze_relax: pass EITHER an explicit T_freeze_C "
                f"OR a liquidus freeze_scheme ({freeze_scheme!r}), not both "
                "(ambiguous freezing-point target)."
            )
        if S_top is None:
            raise ValueError(
                f"under_ice_freeze_relax: freeze_scheme={freeze_scheme!r} "
                "computes the per-cell liquidus from the LOCAL surface "
                "salinity — pass S_top (PSU, shape of T_top_C)."
            )
        # eos.freezing_point returns KELVIN; this helper works in °C.
        T_freeze_C = np.asarray(
            freezing_point(np.asarray(S_top, dtype=np.float64), 0.0,
                           scheme=freeze_scheme),
            dtype=np.float64,
        ) - float(constants.T_freeze)
    elif T_freeze_C is None:
        T_freeze_C = float(constants.T_freeze_ocean) - float(constants.T_freeze)
    sic = np.clip(np.nan_to_num(np.asarray(ice_concentration, dtype=np.float64),
                                nan=0.0), 0.0, 1.0)
    tau_s = float(tau_ice_days) * 86400.0
    alpha = np.minimum(float(dt) / max(tau_s, 1.0e-30), 1.0) * sic
    T = np.asarray(T_top_C, dtype=np.float64)
    # Scalar T_freeze_C -> 0-d float64 array: IEEE-identical arithmetic to the
    # historical float(...) path (byte-identical constant default).
    return T + alpha * (np.asarray(T_freeze_C, dtype=np.float64) - T)


def compute_omip2_surface_forcing(state, *, forcing, idx_t: int,
                                  grid, grid_type: str,
                                  rho_air: float = constants.rho_air,
                                  ice_albedo=None,
                                  under_ice: bool = False,
                                  tau_ice_sw: float = 0.03,
                                  dm2dc_window=None,
                                  u_oce=None, v_oce=None,
                                  wind_current_feedback_vfac: float = 0.0,
                                  forcing_remap: str = "bilinear"):
    """Build an :class:`OceanSurfaceForcing` (tau_x, tau_y, q_net, sw_down) on
    the model grid from CORE-II / JRA55 forcing, for INTEGRATION INSIDE
    ``model.step(state, dt, surface_forcing=...)`` -- the dynamics-core
    external-tau block.  Integrating the forcing within the timestep is
    energetically consistent with the dynamics (the operator-split forward-Euler
    applicator this replaced pumped spurious KE -> runaway velocities over a
    multi-month global run; deleted for issue #1820).

    Conventions (matching the dynamics-core consumer, ocean_pe_latlon_cgrid):
    * ``tau_x``/``tau_y`` are left in the ATMOSPHERIC convention exactly as
      ``air_sea_fluxes`` returns them; the core applies the ``-tau`` ocean
      reaction + the geographic->grid rotation (so no sign/rotation here).
    * ``q_net`` is the TOTAL net surface heat flux into the ocean (turbulent +
      longwave + shortwave); the core subtracts the penetrating ``sw_down`` and
      distributes it over depth.

    ``ice_albedo`` (optional, shape of the model surface field): prescribed
    sea-ice concentration in [0,1].  When given, the downwelling SW is reduced by
    the effective open-ocean/sea-ice albedo (see :func:`_sw_albedo_factor`) in
    BOTH ``q_net`` and the returned ``sw_down``.  ``None`` is bit-exact default.

    ``under_ice`` (prescribed-ice thermodynamic boundary; default False keeps the
    albedo-only surrogate): under sea ice, cut the SW reaching the ocean to
    ``tau_ice_sw`` (~0.03, vs the albedo-only 0.35) and suppress the open-ocean
    turbulent+LW fluxes by ``(1-sic)`` (see :func:`_ice_surface_heat`).  The
    companion under-ice freezing relaxation (:func:`under_ice_freeze_relax`) is a
    post-step state nudge.  Together they cool the over-warm under-ice Southern
    Ocean while leaving seasonal open water to benefit from the albedo correction.

    Supports the lat-lon C-grid family (``latlon`` / ``latlon_regional`` via
    conservative regrid; ``tripole`` via nearest-neighbour on the 2-D T grid).

    ``u_oce`` / ``v_oce`` / ``wind_current_feedback_vfac`` (NEMO ``ln_crt_dwn`` /
    ``rn_vfac``): when ``vfac > 0`` the caller-supplied ocean surface current is
    subtracted from the wind inside :func:`air_sea_fluxes` (relative-wind stress
    + turbulent fluxes).  The current MUST already be in the wind frame
    (GEOGRAPHIC east/north) -- the caller does the grid->geographic rotation (the
    ``legoesm.ocean`` package may not import the coupler rotation helper under the
    import-linter layering contract, so the top-layer run driver rotates and
    passes geographic currents here).  ``vfac == 0.0`` (default) with
    ``u_oce=v_oce=None`` is BYTE-IDENTICAL to the absolute-wind behaviour.

    ``forcing_remap`` selects how the atmospheric fields reach the model grid:
    ``"bilinear"`` (default, unchanged) or ``"nemo_scrip"``, which reads NEMO's
    own SCRIP weight files so the winds arrive through the oracle's exact
    interpolation -- bicubic for u10/v10, bilinear for the rest, as ORCA1's
    namsbc_blk specifies. Tripole only; the sampler raises otherwise.
    """
    from legoesm.ocean.state import OceanSurfaceForcing
    sigma_sb = float(constants.sigma_sb)
    T_freeze = float(constants.T_freeze)

    forc = _sample_omip2_forcing(forcing, idx_t, grid, grid_type,
                                 forcing_remap=forcing_remap)

    if dm2dc_window is not None:
        # NEMO ln_dm2dc (sbcdcy, Bernie et al. 2007): modulate the DAILY-MEAN
        # downwelling SW with the analytic diurnal shape for this step's
        # window.  Mean-preserving by construction; applied to the raw
        # sw_down so BOTH q_net and the penetrative channel see it (exactly
        # where NEMO applies sbc_dcy to qsr).
        _fac = dm2dc_sw_factor(grid, dm2dc_window)
        forc = dict(forc)
        forc["sw_down"] = np.asarray(forc["sw_down"], dtype=np.float64) * _fac

    # Top-cell ocean temperature (state stored in degC) -> K.  Slice FIRST
    # (device-side), THEN convert — np.asarray on the full leaf would
    # assemble the entire 3-D (possibly lat-band-sharded) T field on the
    # host every step (codex batch4 HIGH); only the 2-D surface layer
    # crosses.  Value-identical.  Counted in the module host-pull ledger.
    _record_surface_slice_pull()
    T_sfc_K = np.asarray(state.T.data[..., 0], dtype=np.float64) + T_freeze
    slp = forc.get("slp")
    tau_x, tau_y, sh, lh, evap = air_sea_fluxes(
        u10=jnp.asarray(forc["u10"]),
        v10=jnp.asarray(forc["v10"]),
        T_air_K=jnp.asarray(forc["T_air"]),
        q_air=jnp.asarray(forc["q_air"]),
        T_sfc_K=jnp.asarray(T_sfc_K),
        slp_Pa=None if slp is None else jnp.asarray(slp),
        u_oce=u_oce, v_oce=v_oce, vfac=wind_current_feedback_vfac,
    )
    # Non-solar open-water heat flux, assembled EXACTLY like NEMO blk_oce_2:
    #   qns = eps_w*(LW_down - sigma*T_s^4)            net LW (Kirchhoff: the
    #                                                  SAME eps_w=0.98 weights
    #                                                  absorption AND emission)
    #       + sensible + latent                        NCAR turbulent fluxes
    #       - snow*L_fus                               melt freshly-fallen snow
    #       - evap*c_p_sw*T_s[degC]                    evap removes heat at SST
    #       + rain*c_p_sw*(theta_air[degC])            rain heat content
    #       + snow*c_p_ice*min(theta_air[degC], 0)     snow heat content
    # (NEMO blk_oce_2 receives theta_air_zt for the rain/snow terms and the
    # ABSOLUTE skin/bulk SST for LW + evap heat content; rcp/rcpi/rLfus are
    # the NEMO-parity values.)
    lw_net = constants.emissivity_seawater_lw * (
        np.asarray(forc["lw_down"], dtype=np.float64) - sigma_sb * T_sfc_K ** 4
    )
    precip_np = np.asarray(forc["precip"], dtype=np.float64)
    snow_np = (np.asarray(forc["snow"], dtype=np.float64)
               if forc.get("snow") is not None else np.zeros_like(precip_np))
    # CORE-II precip = RAIN+SNOW; guard tiny negative rain from regridding.
    rain_np = np.maximum(precip_np - snow_np, 0.0)
    from legoesm.ocean.bulk_flux_omip import potential_air_temperature_10m
    theta_air_j, _ = potential_air_temperature_10m(
        jnp.asarray(forc["T_air"]), jnp.asarray(forc["q_air"]),
        None if slp is None else jnp.asarray(slp))
    theta_air_C = np.asarray(theta_air_j, dtype=np.float64) - T_freeze
    T_sfc_C = T_sfc_K - T_freeze
    q_precip_evap = (
        - snow_np * float(constants.L_fus_nemo)
        - np.asarray(evap, dtype=np.float64) * float(constants.c_p_seawater) * T_sfc_C
        + rain_np * float(constants.c_p_seawater) * theta_air_C
        + snow_np * float(constants.c_p_ice_nemo) * np.minimum(theta_air_C, 0.0)
    )
    # Under a prescribed-ice boundary (under_ice) ``_ice_surface_heat``
    # suppresses this on the ice-covered fraction (snow falling on ice does
    # NOT cool the ocean) and cuts the under-ice SW; otherwise it is the
    # albedo-only surrogate.
    q_non_sw = np.asarray(sh) + np.asarray(lh) + lw_net + q_precip_evap
    sw_net, q_net = _ice_surface_heat(
        forc["sw_down"], q_non_sw, ice_albedo,
        under_ice=under_ice, tau_ice_sw=tau_ice_sw)
    # NOTE: the surface freshwater flux P - E is NOT returned here.  It is built
    # by :func:`compute_omip2_freshwater_forcing` and delivered to the ocean via
    # the in-core ``model.step(..., freshwater=FreshwaterForcing)`` channel (which
    # applies the virtual salt with ``config.S_ref`` + the eta free-surface source
    # in the barotropic solve).  Routing it through ``surface_forcing.freshwater``
    # would double-apply on the cube ('external' physics consumes that field) and
    # bypass the eta/free-surface treatment on latlon/MPAS.
    return OceanSurfaceForcing(
        tau_x=jnp.asarray(np.asarray(tau_x)),
        tau_y=jnp.asarray(np.asarray(tau_y)),
        q_net=jnp.asarray(q_net),
        sw_down=jnp.asarray(np.asarray(sw_net)),
    )


# ===========================================================================
# JAX-traceable / lax.scan-fusable forcing path (issue #354)
# ===========================================================================
#
# ``compute_omip2_surface_forcing`` (above) pulls ``state.T`` to the host
# every step (np.asarray) to compute the saturation humidity, which forces
# a device sync per step and prevents wrapping the time loop in
# ``jax.lax.scan``.  The functions below provide a pure-JAX equivalent for
# the tripole grid: the forcing is lifted to device arrays ONCE and the
# spatial nearest-neighbour sample is a static gather, so the whole block
# of steps fuses on-device.  At 1/4 deg the raw CORE-II forcing
# (1460 x 94 x 192 x 7ch, ~1.4 GB) fits on device, whereas pre-sampling all
# records to the model grid (1206 x 1440) would not — hence the gather.


def core2_forcing_nn_indices(forcing, lat_pts_deg, lon_pts_deg):
    """Fixed nearest-neighbour source indices ``(i, j)`` from the CORE-II
    forcing grid to a set of model points.

    Geometry-only (time-independent), so the spatial sample inside a
    ``lax.scan`` becomes a static gather.  Replicates the ``(i, j)``
    computed in :func:`_nn_interp_to_points` EXACTLY (nearest latitude by
    ``|Δlat|``; nearest longitude by periodic wrap distance).
    """
    src_lat = np.asarray(forcing.lat, dtype=np.float64)
    src_lon = np.asarray(forcing.lon, dtype=np.float64) % 360.0
    dst_lat = np.asarray(lat_pts_deg, dtype=np.float64)
    dst_lon = np.asarray(lon_pts_deg, dtype=np.float64) % 360.0
    i = np.argmin(np.abs(src_lat[:, None] - dst_lat[None, :]), axis=0)
    dlon = np.abs(src_lon[:, None] - dst_lon[None, :])
    dlon = np.minimum(dlon, 360.0 - dlon)
    j = np.argmin(dlon, axis=0)
    return i.astype(np.int32), j.astype(np.int32)


def build_core2_forcing_device_stack(forcing, grid, grid_type: str):
    """Lift CORE-II forcing to device arrays + fixed NN indices for the
    scan kernel (issue #354).  Tripole only (the faithful OMIP path);
    other grids keep the host per-step sampling.

    Returns
    -------
    forcing_stack : dict[str, jax.Array]
        6 channels (u10, v10, T_air, q_air, sw_down, lw_down), each
        ``(n_rec, n_lat_f, n_lon_f)`` on device.
    nn_i, nn_j : jax.Array (int32, ``(n_lat*n_lon,)``)
        Nearest-neighbour source indices into the forcing grid.
    grid_shape : tuple
        ``(n_lat, n_lon)`` of the model T grid.
    """
    if grid_type != "tripole":
        raise NotImplementedError(
            "build_core2_forcing_device_stack: the lax.scan forcing path is "
            "implemented for grid_type='tripole' (the faithful OMIP path); "
            f"got {grid_type!r}.  Use the host compute_omip2_surface_forcing."
        )
    lat_pts = np.degrees(np.asarray(grid.lat_T)).reshape(-1)
    lon_pts = np.degrees(np.asarray(grid.lon_T)).reshape(-1)
    # BILINEAR maps (i4, j4, w4), matching the host tripole sample exactly
    # (the scan lane and the host loop must apply the SAME forcing; nearest
    # printed zonal bands -- see _bilinear_point_maps).
    nn_i, nn_j, nn_w = _bilinear_point_maps(
        np.asarray(forcing.lat), np.asarray(forcing.lon), lat_pts, lon_pts)
    channels = ("u10", "v10", "T_air", "q_air", "sw_down", "lw_down",
                "precip")
    forcing_stack = {
        name: jnp.asarray(np.asarray(getattr(forcing, name)))
        for name in channels
    }
    # Optional NEMO-parity channels: snow (q_ns fusion/heat-content terms)
    # and slp (moist-air density + Goff saturation).  Missing -> the same
    # fallbacks as the host path (snow=0, slp=standard atmosphere), built
    # as full records so the scan gather stays shape-uniform.
    ref = np.asarray(forcing.precip)
    snow = getattr(forcing, "snow", None)
    slp = getattr(forcing, "slp", None)
    forcing_stack["snow"] = jnp.asarray(
        np.zeros_like(ref) if snow is None else np.asarray(snow))
    forcing_stack["slp"] = jnp.asarray(
        np.full_like(ref, float(constants.p_atm_std)) if slp is None
        else np.asarray(slp))
    grid_shape = tuple(np.asarray(grid.lat_T).shape)
    return (forcing_stack, jnp.asarray(nn_i), jnp.asarray(nn_j),
            jnp.asarray(nn_w), grid_shape)


def compute_omip2_surface_forcing_jax(
    state, *, forcing_stack, nn_i, nn_j, grid_shape, idx_t, rho_air=constants.rho_air,
    nn_w=None,
):
    """Pure-JAX, ``lax.scan``-traceable form of
    :func:`compute_omip2_surface_forcing` for the tripole grid (issue #354).

    ``idx_t`` is the (possibly traced) 6-hourly CORE-II record index.  The
    spatial sample is a static nearest-neighbour gather via the fixed
    ``(nn_i, nn_j)`` map, so there is NO host roundtrip per step.
    Bit-equivalent (to fp tolerance) to the host function for tau/q_net/sw_down:
    identical ``air_sea_fluxes`` (NCAR), identical NEMO-form non-solar assembly
    (Kirchhoff LW + snow fusion + precip/evap heat content), identical
    nearest-neighbour spatial sample.  NO ice (``ice_albedo``/``under_ice``)
    support — the prescribed-ice runs use the host loop.

    LIMITATION: this scan variant covers ONLY tau/q_net/sw_down.  The device
    stack DOES carry precip/snow/slp (the q_ns heat-content terms need them),
    but the P - E / runoff ``FreshwaterForcing`` is still NOT built here: the
    faithful surface-freshwater forcing is applied only on the HOST run loop
    (``compute_omip2_freshwater_forcing`` -> ``model.step(freshwater=...)``),
    and the run driver refuses ``--scan-block`` for salinity-faithful runs
    until the freshwater channel is wired into the scan body.
    """
    from legoesm.ocean.state import OceanSurfaceForcing
    sigma_sb = float(constants.sigma_sb)
    T_freeze = float(constants.T_freeze)

    def _sample(name):
        rec = forcing_stack[name][idx_t]            # (n_lat_f, n_lon_f)
        if nn_w is None:                       # legacy pure-NN indices (N,)
            return rec[nn_i, nn_j].reshape(grid_shape)
        # 4-point bilinear: (4, N) index/weight stacks -> (n_lat, n_lon)
        return (rec[nn_i, nn_j] * nn_w).sum(axis=0).reshape(grid_shape)

    u10 = _sample("u10")
    v10 = _sample("v10")
    T_air = _sample("T_air")
    q_air = _sample("q_air")
    sw_down = _sample("sw_down")
    lw_down = _sample("lw_down")
    precip = _sample("precip")
    snow = _sample("snow")
    slp = _sample("slp")

    # Top-cell ocean temperature (state in degC) -> K, pure JAX (no host pull).
    T_sfc_K = state.T.data[..., 0] + T_freeze
    tau_x, tau_y, sh, lh, evap = air_sea_fluxes(
        u10=u10, v10=v10, T_air_K=T_air, q_air=q_air,
        T_sfc_K=T_sfc_K, slp_Pa=slp,
    )
    # Non-solar assembly: EXACT jnp mirror of the host
    # ``compute_omip2_surface_forcing`` (NEMO blk_oce_2 form) -- net-LW
    # Kirchhoff eps_w, snow fusion, rain/snow heat content at theta_air,
    # evap heat content at the absolute SST.
    from legoesm.ocean.bulk_flux_omip import potential_air_temperature_10m
    lw_net = constants.emissivity_seawater_lw * (
        lw_down - sigma_sb * T_sfc_K ** 4)
    rain = jnp.maximum(precip - snow, 0.0)
    theta_air, _ = potential_air_temperature_10m(T_air, q_air, slp)
    theta_air_C = theta_air - T_freeze
    T_sfc_C = T_sfc_K - T_freeze
    q_precip_evap = (
        - snow * constants.L_fus_nemo
        - evap * constants.c_p_seawater * T_sfc_C
        + rain * constants.c_p_seawater * theta_air_C
        + snow * constants.c_p_ice_nemo * jnp.minimum(theta_air_C, 0.0)
    )
    q_net = sh + lh + lw_net + q_precip_evap + sw_down
    return OceanSurfaceForcing(
        tau_x=tau_x, tau_y=tau_y, q_net=q_net, sw_down=sw_down,
    )


def build_omip2_scan_block_fn(
    model, dt, grid_shape, *, rho_air=constants.rho_air, ramp_s=0.0,
):
    """Build a JIT-compiled ``lax.scan`` block-step function for the tripole
    OMIP time loop (issue #354).

    The returned
    ``block_fn(state, forcing_stack, nn_i, nn_j, idx_t_block, step0)``
    advances ``state`` over ``len(idx_t_block)`` steps; each step samples
    the CORE-II forcing on-device via
    :func:`compute_omip2_surface_forcing_jax`, optionally applies the
    spin-up wind/heat ramp, and calls ``model._step_impl`` (the un-wrapped
    core step, so no nested JIT).  Because the forcing has no host
    roundtrip, XLA fuses the whole block.

    ``forcing_stack`` / ``nn_i`` / ``nn_j`` are explicit JIT arguments
    (NOT closure captures), so the ~1.3 GB 1/4-deg forcing is a device
    INPUT rather than an embedded compiled constant.  ``grid_shape`` is a
    static (closure) shape used only in ``reshape``.

    Host-side per-step pieces of the Python loop (diagnostics, snapshots,
    non-finite abort, optional WOA nudging / spin-up drag) are intentionally
    NOT inside the scan — the caller runs them at block boundaries and must
    not enable nudging/drag on the scan path.
    """
    import jax
    from jax import lax

    apply_ramp = ramp_s > 0.0  # static gate (CLAUDE.md feature-gating)

    @jax.jit
    def block_fn(state, forcing_stack, nn_i, nn_j, nn_w, idx_t_block, step0):
        def _body(carry, idx_t):
            st, step = carry
            sf = compute_omip2_surface_forcing_jax(
                st, forcing_stack=forcing_stack, nn_i=nn_i, nn_j=nn_j,
                nn_w=nn_w,
                grid_shape=grid_shape, idx_t=idx_t, rho_air=rho_air,
            )
            if apply_ramp:
                ramp = jnp.minimum(
                    1.0, (step.astype(sf.tau_x.dtype) * dt) / ramp_s)
                sf = sf._replace(
                    tau_x=sf.tau_x * ramp, tau_y=sf.tau_y * ramp,
                    q_net=sf.q_net * ramp, sw_down=sf.sw_down * ramp,
                )
            st = model._step_impl(st, dt, surface_forcing=sf)
            # The scan body calls ``_step_impl`` directly (no nested JIT), so it
            # bypasses ``model.step``'s post-step freezing-point floor.  Re-apply
            # it here under the same STATIC config gate so ``--scan-block`` and
            # the default Python loop are physically identical (issue #354 +
            # freeze_floor). ``_step_impl`` returns a plain state here
            # (outer_integrator='forward_euler'; scan does not support ab2).
            if getattr(model.config, "freeze_floor", False):
                st = model._apply_freeze_floor(st)
            return (st, step + 1), None

        # store_mass_flux (#1442, codex round-6 RED 2): ``_step_impl`` turns
        # the mass_flux_* slots from None into Fields when the flag is on, so
        # an unseeded carry aborts this scan on iteration 1 with a carry
        # structure mismatch.  Seed at the scan boundary.  STATIC gate on a
        # config bool (never a traced value), and a no-op for every model whose
        # config lacks the field -- so the historical path is untouched.
        #
        # This also fixes the RETURNED structure: without it block_fn would
        # return a state whose slots are Fields while its input's were None,
        # retracing this jit on the host loop's second block.
        if getattr(model.config, "store_mass_flux", False):
            from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
                seed_mass_flux_carry,
            )
            state = seed_mass_flux_carry(state, True)
        if getattr(model.config, "store_salt_flux", False):
            from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
                seed_salt_flux_carry,
            )
            state = seed_salt_flux_carry(state, True)
        (state, _), _ = lax.scan(_body, (state, step0), idx_t_block)
        return state

    return block_fn


__all__ = [
    "compute_omip2_surface_forcing",
    "compute_omip2_freshwater_forcing",
    "compute_omip2_surface_forcing_jax",
    "build_core2_forcing_device_stack",
    "build_omip2_scan_block_fn",
    "core2_forcing_nn_indices",
]
