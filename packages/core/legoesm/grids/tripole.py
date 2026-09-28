"""Tripolar grid support for the lat-lon C-grid ocean.

Provides:
- ``create_tripole_grid``: load a tripolar grid from a NEMO mesh_mask
  NetCDF file (e.g. ORCA1) into a ``LatLonCGridGeometry``.
- ``create_synthetic_tripole``: generate a synthetic tripolar-like grid
  for unit testing of fold halo exchange and metric-array operators,
  without requiring a 484 MB grid file download.

The synthetic grid is a regular lat-lon grid with an active
``FoldDescriptor`` at the northern boundary.  South of the cap latitude
the metrics are identical to ``create_latlon_geometry``; above the cap
the metrics remain regular lat-lon (not a true bipolar cap) but the
fold is correctly configured so that halo exchange, operator boundary
handling, and vector sign flips can be tested.

References
----------
- Murray, R. J. (1996). Explicit generation of orthogonal grids for
  ocean models. J. Comput. Phys. 126, 251--273.
- Madec, G. and Imbard, M. (1996). A global ocean mesh to overcome
  the North Pole singularity. Climate Dyn. 12, 381--388.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.latlon import (
    FoldDescriptor,
    LatLonCGridGeometry,
    compute_v_face_coords,
    create_latlon_geometry,
)


# =========================================================================
# NEMO mesh_mask reader
# =========================================================================


def mesh_file_list(path) -> list[str]:
    """Normalise a NEMO mesh spec to a list of file paths.

    Accepts one path, a sequence of paths, or a single string with
    ``os.pathsep``-joined paths.  Older NEMO runs (e.g. NOC ORCA0083) ship
    the grid as THREE files -- ``mesh_hgr.nc`` (metrics/coordinates),
    ``mesh_zgr.nc`` (vertical scale factors) and ``mask.nc`` (masks) -- while
    newer ones bake everything into one ``mesh_mask.nc``/``domain_cfg.nc``.
    """
    import os
    if isinstance(path, (str, Path)):
        parts = str(path).split(os.pathsep)
    else:
        parts = [str(p) for p in path]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("mesh path list is empty")
    return parts


def _read_nemo_mesh_mask(path) -> dict:
    """Read a NEMO mesh_mask (or domain_cfg) NetCDF file, or a LIST of files
    that together hold the mesh (``mesh_hgr.nc`` + ``mesh_zgr.nc`` + ``mask.nc``).

    Returns a flat dict of 2D JAX arrays keyed by NEMO variable name.
    Only the surface-level slice of 3D fields is kept.  With several files
    the FIRST file holding a variable wins; a variable in none of them is
    simply absent (as before).

    Parameters
    ----------
    path : str, Path, or sequence of them
        Path(s) to NEMO ``mesh_mask*.nc`` / ``domcfg*.nc`` / ``mesh_hgr.nc`` /
        ``mesh_zgr.nc`` / ``mask.nc`` files (see :func:`mesh_file_list`).

    Returns
    -------
    dict
        Keys include ``glamt``, ``gphit``, ``e1t``, ``e2t``,
        ``e1u``, ``e2u``, ``e1v``, ``e2v``, ``tmask``, etc.
    """
    try:
        import netCDF4  # noqa: N813
    except ImportError as exc:
        raise ImportError(
            "netCDF4 is required for reading NEMO grid files. "
            "Install with: pip install netCDF4"
        ) from exc

    out: dict[str, jax.Array] = {}

    # Standard NEMO mesh_mask variables (2D or 3D with time/depth dims)
    wanted_2d = [
        "glamt", "gphit", "glamu", "gphiu", "glamv", "gphiv",
        "glamf", "gphif",
        "e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f",
    ]
    wanted_masks = ["tmask", "umask", "vmask", "fmask"]

    for one in mesh_file_list(path):
        ds = netCDF4.Dataset(one, "r")
        for name in wanted_2d:
            if name in out:
                continue
            # Try with and without _0 suffix (domain_cfg convention)
            for suffix in ["", "_0"]:
                key = name + suffix
                if key in ds.variables:
                    arr = ds.variables[key][:]
                    # Squeeze singleton dims (time, depth)
                    while arr.ndim > 2:
                        arr = arr[0]
                    out[name] = jnp.array(arr, dtype=jnp.float64)
                    break

        for name in wanted_masks:
            if name in out:
                continue
            if name in ds.variables:
                arr = ds.variables[name][:]
                while arr.ndim > 2:
                    arr = arr[0]
                out[name] = jnp.array(arr, dtype=jnp.float64)
        ds.close()

    return out


def strip_north_rows_raw(raw: dict, n: int) -> dict:
    """Drop ``n`` rows from the NORTH (last axis-0 rows) of every 2-D mesh
    field.  ``n == 0`` returns ``raw`` unchanged; negative raises."""
    if n < 0:
        raise ValueError(f"strip_north_rows must be >= 0, got {n}")
    if n == 0:
        return raw
    out = {}
    for k, v in raw.items():
        if v.shape[0] <= n:
            raise ValueError(
                f"strip_north_rows={n} would remove every row of {k!r} "
                f"(shape {tuple(v.shape)})")
        out[k] = v[:-n]
    return out


def _detect_cap_j(gphit, n_lat, fold_j, cap_dlat_rel_deviation):
    """Row where the grid starts deviating from regular lat-lon (bipolar
    cap start): first row whose dlat deviates from the southern-half median
    by more than ``cap_dlat_rel_deviation`` (relative)."""
    n_lon = gphit.shape[1]
    lat_col = gphit[:, n_lon // 4]  # sample column away from fold poles
    dlat = jnp.diff(lat_col)
    median_dlat = jnp.median(dlat[:n_lat // 2])  # use southern half
    deviation = jnp.abs(dlat - median_dlat) / jnp.abs(median_dlat)
    cap_candidates = jnp.where(deviation > cap_dlat_rel_deviation,
                               size=n_lat - 1)
    if len(cap_candidates[0]) > 0:
        return int(cap_candidates[0][0])
    return fold_j  # no cap detected (very regular grid)


def _detect_fold(
    glamt: jax.Array,
    gphit: jax.Array,
    n_lat: int,
    n_lon: int,
    *,
    max_fold_asym_deg: float = 0.1,
    cap_dlat_rel_deviation: float = 0.1,
    fold_convention: str = "auto",
    fold_tie_tol_deg: float = 1e-6,
    fold_pivot: str = "legacy",
    vf_coords=None,
) -> FoldDescriptor:
    """Detect the tripolar fold from the T-point coordinates.

    NEMO ORCA grids have a T-fold at the last j-row: cell (i, j_max)
    is identified with cell (n_lon - 1 - i, j_max).  We verify this
    by checking that the longitude pattern at j_max is self-reversing.

    Parameters
    ----------
    glamt, gphit : (n_lat, n_lon) arrays of T-point lon/lat [degrees].
    n_lat, n_lon : grid dimensions.
    max_fold_asym_deg : float, default 0.1
        Maximum allowed absolute latitude asymmetry across the fold
        permutation [deg]. Larger values are treated as a non-NEMO grid
        and raise.
    cap_dlat_rel_deviation : float, default 0.1
        Relative deviation of per-row dlat from the southern-half median
        used to detect where the bipolar cap begins (dimensionless).
    fold_convention : {"auto", "n_lon-1-i", "(n_lon-i)%n_lon"}, default "auto"
        Fold index convention. ``"auto"`` picks whichever convention makes the
        fold-row latitude most self-symmetric, and RAISES when both conventions
        fit equally well (a true tie, e.g. a near-constant fold-row latitude),
        since symmetry then cannot disambiguate the E-W wrap origin and a silent
        guess could be the WRONG origin for a de-haloed mesh. Pass the
        convention EXPLICITLY (``"(n_lon-i)%n_lon"`` for de-haloed meshes such
        as eORCA025, ``"n_lon-1-i"`` for halo-inclusive meshes such as
        eORCA1.2); the explicit choice is still verified against
        ``max_fold_asym_deg``.
    fold_tie_tol_deg : float, default 1e-6
        ``"auto"`` tie threshold [deg]: if the two candidate fold-row latitude
        asymmetries differ by no more than this (and both are valid fits), the
        detection is ambiguous and raises rather than silently guessing.
    fold_pivot : {"legacy", "F"}, default "legacy"
        ``"legacy"`` = every existing detection path, byte-identical.  ``"F"``
        = explicit opt-in to the NEMO F-point-pivot descriptor for a mesh whose
        duplicated fold-halo row has been STRIPPED (see
        ``FoldDescriptor.fpivot``).  The maps are VERIFIED, not assumed: the
        stored top v row and top F row must each be self-coincident under
        their fold maps (``vf_coords`` required), else this raises.
    vf_coords : (glamv, gphiv, glamf, gphif) arrays in degrees, optional
        NEMO-layout V and F coordinates (east-of-cell ``glamf`` columns),
        required for ``fold_pivot="F"``.

    Returns
    -------
    FoldDescriptor
    """
    fold_j = n_lat - 1
    lat_fold = gphit[fold_j]

    if fold_pivot not in ("legacy", "F"):
        raise ValueError(
            f"fold_pivot must be 'legacy' or 'F', got {fold_pivot!r}")
    if fold_pivot == "F":
        return _detect_fpivot_fold(glamt, gphit, n_lat, n_lon, vf_coords,
                                   cap_dlat_rel_deviation)

    # NEMO tripole meshes use two index conventions for the T-fold
    # self-permutation of the fold (last) row, differing only by the cyclic
    # E-W wrap-halo origin (both are jperio=4 T-point folds):
    #   - halo-inclusive (e.g. the eORCA1.2 mesh_mask, which bakes in two
    #     cyclic E-W halo columns):  perm[i] = n_lon - 1 - i
    #   - pure / de-haloed (e.g. the eORCA025 mesh_mask): perm[i] = (n_lon - i) % n_lon
    # Auto-detect by choosing the permutation that makes the fold-row latitude
    # self-symmetric, so a single code path reads either mesh. perm_v shares
    # the chosen origin (V-points stagger with T along i on the ORCA T-fold).
    perm_candidates = {
        "n_lon-1-i": jnp.arange(n_lon - 1, -1, -1, dtype=jnp.int32),
        "(n_lon-i)%n_lon": (n_lon - jnp.arange(n_lon, dtype=jnp.int32)) % n_lon,
    }

    # --- EXACT storage-layout classification (measured convention, 2026-08-26;
    # see scripts/validate/ocean_fidelity/check_tripole_fold_pairing.py and the
    # NEMO T-pivot reference lbc_nfd_generic.h90).  Real ORCA meshes hit one of
    # these to <1e-6 deg; synthetic/legacy grids fall through to the lat-only
    # auto-detect below (byte-identical legacy behaviour).
    #   pivot_row_stored (de-haloed, e.g. eORCA025): stored top T row is the
    #     SELF-symmetric pivot row under P_T=(n_lon-i)%n_lon.
    #   halo_row_stored (e.g. eORCA1.2): stored top T row == permuted copy of
    #     the row below under n_lon-1-i.
    def _wrap_dlon(a, b):
        d = jnp.abs(a - b) % 360.0
        return jnp.minimum(d, 360.0 - d)

    lon_fold = glamt[fold_j]
    _exact = 1.0e-5
    _idx = jnp.arange(n_lon, dtype=jnp.int32)
    _p_self = (n_lon - _idx) % n_lon
    _m_self = _p_self != _idx
    _self_ok = bool(
        (jnp.max(jnp.where(_m_self, jnp.abs(lat_fold - lat_fold[_p_self]), 0.0))
         < _exact)
        and (jnp.max(jnp.where(_m_self,
                               _wrap_dlon(lon_fold, lon_fold[_p_self]), 0.0))
             < _exact))
    if _self_ok:
        # De-haloed T-pivot mesh (eORCA025 class): per-point-type maps from
        # the measured coincidences (T/U self on the pivot row; V/F pair with
        # the row below): P_T=(-i)%n, P_U=(-i-1)%n; V uses P_T, F uses P_U.
        return FoldDescriptor(
            is_active=True, fold_j=fold_j,
            cap_j=_detect_cap_j(gphit, n_lat, fold_j, cap_dlat_rel_deviation),
            perm_T=_p_self,
            perm_v=_p_self,
            vector_sign_u=-1.0, vector_sign_v=-1.0,
            pivot_row_stored=True,
            perm_u=(n_lon - _idx - 1) % n_lon,
            perm_f=(n_lon - _idx - 1) % n_lon,
        )
    # Halo-row-stored meshes (eORCA1.2 class: stored top row duplicates the
    # row below; validated by the 1-degree campaign) intentionally fall
    # through to the LEGACY lat-symmetry auto-detect below — byte-identical
    # behaviour for every existing working configuration.
    if fold_convention not in ("auto", *perm_candidates):
        raise ValueError(
            f"fold_convention must be 'auto' or one of {list(perm_candidates)}, "
            f"got {fold_convention!r}."
        )
    # Unfilled cyclic-halo columns: NOC ORCA0083 mesh_hgr stores (lat, lon) =
    # (0, 0) in the last two columns of its fold rows (NEMO never filled them).
    # No Arctic fold row passes through (0 N, 0 E), so such a column is a
    # placeholder, not geometry; it is excluded from the symmetry check (for
    # itself and as a partner) and reported.  Real asymmetries still fail.
    lon_fold = glamt[fold_j]
    unset = (lat_fold == 0.0) & (lon_fold == 0.0)
    n_unset = int(jnp.sum(unset))
    if n_unset:
        if n_unset > n_lon // 4:
            raise ValueError(
                f"Fold row j={fold_j}: {n_unset} of {n_lon} columns are unset "
                "(0,0) placeholders -- more than a cyclic halo; this is not a "
                "usable fold row (wrong strip_north_rows or a broken mesh).")
        print(f"  tripole fold check: ignoring {n_unset} unset (0,0) placeholder "
              f"column(s) on the fold row j={fold_j}")

    def _asym(p):
        keep = ~(unset | unset[p])
        d = jnp.abs(lat_fold - lat_fold[p])
        return float(jnp.max(jnp.where(keep, d, 0.0)))

    asym_by_perm = {name: _asym(p) for name, p in perm_candidates.items()}
    if fold_convention == "auto":
        # Symmetry-based auto-detect. A near-tie means BOTH conventions fit the
        # fold-row latitude equally well (e.g. a near-constant fold-row
        # latitude), so symmetry cannot disambiguate the E-W wrap origin.
        # Silently taking min(...) would pick the dict-first key and could
        # corrupt every ORCA-seam fold halo / vector-sign flip — require an
        # explicit convention instead.
        # A tie = BOTH conventions are valid fits (each asymmetry <=
        # max_fold_asym_deg) AND they are indistinguishable (their asymmetries
        # differ by <= fold_tie_tol_deg) — e.g. a near-constant fold-row
        # latitude is self-symmetric under either origin. Symmetry then cannot
        # disambiguate the E-W wrap origin, so silently taking min(...) could
        # pick the wrong ORCA seam. A real mesh leaves the WRONG origin with an
        # O(deg) asymmetry (>> the right one), so the normal case — exactly one
        # convention fitting — falls through to the min() below.
        _asyms = list(asym_by_perm.values())
        if (max(_asyms) <= max_fold_asym_deg
                and abs(_asyms[0] - _asyms[1]) <= fold_tie_tol_deg):
            raise ValueError(
                f"Ambiguous fold_convention='auto' at j={fold_j}: both index "
                f"conventions fit the fold-row latitude equally well "
                f"(asymmetries {asym_by_perm}, within {fold_tie_tol_deg} deg). "
                f"Pass fold_convention explicitly ('(n_lon-i)%n_lon' for "
                f"de-haloed meshes such as eORCA025, 'n_lon-1-i' for "
                f"halo-inclusive meshes such as eORCA1.2)."
            )
        best_perm = min(asym_by_perm, key=asym_by_perm.get)
    else:
        best_perm = fold_convention
    max_asym = asym_by_perm[best_perm]
    if max_asym > max_fold_asym_deg:
        raise ValueError(
            f"Fold symmetry check failed: lat asymmetry for perm {best_perm!r} "
            f"= {max_asym:.3f} deg at j={fold_j} (all candidates: "
            f"{asym_by_perm}; fold_convention={fold_convention!r}). This may "
            f"not be a standard NEMO T-fold grid, or the explicit "
            f"fold_convention is wrong for this mesh."
        )
    perm_T = perm_candidates[best_perm]
    # For v/q stagger the permutation is the same for the ORCA T-fold.
    perm_v = perm_T

    cap_j = _detect_cap_j(gphit, n_lat, fold_j, cap_dlat_rel_deviation)

    return FoldDescriptor(
        is_active=True,
        fold_j=fold_j,
        cap_j=cap_j,
        perm_T=perm_T,
        perm_v=perm_v,
        vector_sign_u=-1.0,
        vector_sign_v=-1.0,
    )


def _detect_fpivot_fold(glamt, gphit, n_lat, n_lon, vf_coords,
                        cap_dlat_rel_deviation, tol_deg: float = 1e-6):
    """F-point-pivot descriptor for a mesh with its fold-halo row stripped.

    Measured maps (``measure_fpivot_fold_perms.py`` on eORCA1.2, this model's
    staggering; NEMO ``lbc_nfd_generic.h90`` F branch): ``P_T = P_V = n-1-i``,
    ``P_U = P_F = (n-i) % n``.  Verified here on the mesh itself: the stored
    top V row (the fold line) must be self-coincident under ``P_V`` and the
    top F row under ``P_F`` (our SW-vertex column ``k`` = NEMO F column
    ``k-1``), every column except the two cyclic-halo columns, to
    ``tol_deg``.  A mesh whose top row is still the duplicated halo row, or a
    T-pivot mesh, fails and raises.
    """
    if vf_coords is None:
        raise ValueError("fold_pivot='F' needs vf_coords=(glamv, gphiv, "
                         "glamf, gphif) to verify the fold line")
    glamv, gphiv, glamf, gphif = (jnp.asarray(a) for a in vf_coords)
    idx = jnp.arange(n_lon, dtype=jnp.int32)
    p_t = (n_lon - 1 - idx).astype(jnp.int32)
    p_u = ((n_lon - idx) % n_lon).astype(jnp.int32)

    def _wrap(a, b):
        d = jnp.abs(a - b) % 360.0
        return jnp.minimum(d, 360.0 - d)

    def _self_err(lon_row, lat_row, perm, keep):
        d = jnp.maximum(_wrap(lon_row, lon_row[perm]),
                        jnp.abs(lat_row - lat_row[perm]))
        return float(jnp.max(jnp.where(keep, d, 0.0)))

    # Exclude self-paired columns and (when the mesh carries them) the two
    # cyclic-halo columns, whose stored coordinates NEMO does not fold.
    halo = (idx == 0) | (idx == n_lon - 1)
    keep_v = (p_t != idx) & ~halo & ~halo[p_t]
    err_v = _self_err(glamv[-1], gphiv[-1], p_t, keep_v)
    # Our vertex column k = NEMO F column k-1 (SW corner of cell k).
    lon_f = jnp.concatenate([glamf[-1, -1:], glamf[-1, :-1]])
    lat_f = jnp.concatenate([gphif[-1, -1:], gphif[-1, :-1]])
    src = idx - 1
    halo_f = (src <= 0) | (src >= n_lon - 1)
    keep_f = (p_u != idx) & ~halo_f & ~halo_f[p_u]
    err_f = _self_err(lon_f, lat_f, p_u, keep_f)
    if not (err_v < tol_deg and err_f < tol_deg):
        raise ValueError(
            "fold_pivot='F' refused: the stored top row is not an F-pivot "
            f"fold line (V self-mismatch {err_v:.3e} deg under n-1-i, F "
            f"self-mismatch {err_f:.3e} deg under (n-i)%n; need < {tol_deg}). "
            "Strip the duplicated fold-halo row first "
            "(strip_north_rows=1 on eORCA1.2), or this is not an F-pivot mesh.")
    # NEMO's cyclic E-W halo columns (eORCA1.2): col 0 == col n-2, col n-1 ==
    # col 1 on every row below the fold.
    glamt = jnp.asarray(glamt)
    ew_halo = bool(
        n_lon > 4
        and float(jnp.max(_wrap(glamt[:-1, 0], glamt[:-1, n_lon - 2]))) < tol_deg
        and float(jnp.max(_wrap(glamt[:-1, n_lon - 1], glamt[:-1, 1]))) < tol_deg)
    return fpivot_descriptor(n_lat, n_lon, ew_halo=ew_halo,
                             cap_j=_detect_cap_j(gphit, n_lat, n_lat - 1,
                                                 cap_dlat_rel_deviation))


def fpivot_perms(n_lon: int, ew_halo: bool):
    """F-pivot fold maps ``(P_T, P_U)`` in this model's staggering
    (``P_V = P_T``, ``P_F = P_U``).

    Pure periodic layout: ``P_T = n-1-i``, ``P_U = (n-i) % n``.  With NEMO's
    two cyclic halo columns (``ew_halo``) the maps are NEMO ``lbc_nfd``'s own
    index sets (``ihls = 1``), which read interior columns only:
    T/V column 0 <- 1 and n-1 <- n-2; U/F (our west face k = NEMO column
    k-1) face 0 <- n-2, face n-1 <- n-1 (self: NEMO's point ``ipi-ihls``).
    """
    idx = jnp.arange(n_lon, dtype=jnp.int32)
    p_t = (n_lon - 1 - idx).astype(jnp.int32)
    p_u = ((n_lon - idx) % n_lon).astype(jnp.int32)
    if ew_halo:
        p_t = p_t.at[0].set(1).at[n_lon - 1].set(n_lon - 2)
        p_u = p_u.at[0].set(n_lon - 2).at[n_lon - 1].set(n_lon - 1)
    return p_t, p_u


def fpivot_descriptor(n_lat: int, n_lon: int, *, ew_halo: bool = False,
                      cap_j: int | None = None) -> FoldDescriptor:
    """The F-pivot ``FoldDescriptor`` (see ``FoldDescriptor.fpivot``)."""
    if n_lon % 2:
        raise ValueError(f"F-pivot fold needs an even n_lon, got {n_lon}")
    p_t, p_u = fpivot_perms(n_lon, ew_halo)
    return FoldDescriptor(
        is_active=True, fold_j=n_lat - 1,
        cap_j=max(0, n_lat - n_lat // 4) if cap_j is None else int(cap_j),
        perm_T=p_t, perm_v=p_t,
        vector_sign_u=-1.0, vector_sign_v=-1.0,
        pivot_row_stored=False,
        perm_u=p_u, perm_f=p_u,
        fpivot=True, ew_halo=bool(ew_halo),
    )


def _compute_rotation_angles(
    glamu: jax.Array,
    gphiu: jax.Array,
    glamv: jax.Array,
    gphiv: jax.Array,
    cap_j: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute rotation angles from local grid axes to geographic east/north.

    Below ``cap_j`` the grid is regular lat-lon and rotation angles are
    (cos=1, sin=0).  Above ``cap_j`` the local i-axis deviates from
    geographic east; the angle is estimated from the coordinate
    gradients.

    Returns
    -------
    cos_alpha_u, sin_alpha_u, cos_alpha_v, sin_alpha_v
    """
    n_lat_u, n_lon_u = glamu.shape
    n_lat_v, n_lon_v = glamv.shape

    # u-point rotation: direction of the local i-axis at u-points.
    # This is approximated by the coordinate gradient along i.
    cos_alpha_u = jnp.ones((n_lat_u, n_lon_u))
    sin_alpha_u = jnp.zeros((n_lat_u, n_lon_u))

    # v-point rotation
    cos_alpha_v = jnp.ones((n_lat_v, n_lon_v))
    sin_alpha_v = jnp.zeros((n_lat_v, n_lon_v))

    if cap_j >= n_lat_u - 1:
        # No bipolar cap — all regular lat-lon
        return cos_alpha_u, sin_alpha_u, cos_alpha_v, sin_alpha_v

    # Inside the bipolar cap, compute the angle between local i-axis
    # and geographic east using coordinate gradients.
    # For u-points: delta_lon and delta_lat along the i-direction
    # at the u-point gives the orientation of the local i-axis.
    dlon_u = jnp.diff(glamu, axis=1)  # along i
    # H6: unwrap the along-i longitude difference across the +-180 branch cut so
    # a cell straddling the seam does not get a spurious ~360 deg dlon and a
    # garbage angle.  180/360 are branch-cut geometry, not physical constants.
    dlon_u = ((dlon_u + 180.0) % 360.0) - 180.0
    dlat_u = jnp.diff(gphiu, axis=1)
    # Angle of local i-axis relative to east
    alpha_u_interior = jnp.arctan2(
        jnp.deg2rad(dlat_u),
        jnp.deg2rad(dlon_u) * jnp.cos(jnp.deg2rad(gphiu[:, :-1])),
    )
    # Pad to full shape (wrap)
    alpha_u_full = jnp.concatenate(
        [alpha_u_interior, alpha_u_interior[:, 0:1]], axis=1
    )
    # Only apply in the cap region
    mask_u = jnp.arange(n_lat_u)[:, None] >= cap_j
    cos_alpha_u = jnp.where(mask_u, jnp.cos(alpha_u_full), 1.0)
    sin_alpha_u = jnp.where(mask_u, jnp.sin(alpha_u_full), 0.0)

    # v-points: same approach along j-direction
    dlon_v = jnp.diff(glamv, axis=0)
    # H6: unwrap across the +-180 branch cut (see the u-point note above).
    dlon_v = ((dlon_v + 180.0) % 360.0) - 180.0
    dlat_v = jnp.diff(gphiv, axis=0)
    alpha_v_interior = jnp.arctan2(
        jnp.deg2rad(dlon_v) * jnp.cos(jnp.deg2rad(gphiv[:-1, :])),
        jnp.deg2rad(dlat_v),
    )
    # Pad north row
    alpha_v_full = jnp.concatenate(
        [alpha_v_interior, alpha_v_interior[-1:, :]], axis=0
    )
    mask_v = jnp.arange(n_lat_v)[:, None] >= cap_j
    cos_alpha_v = jnp.where(mask_v, jnp.cos(alpha_v_full), 1.0)
    # H5: the v-point arctan2 args are swapped vs the u-point, so
    # alpha_v_full = atan2(E_j, N_j) = -alpha (the i-axis->east angle).
    # cos(-alpha)=+cos(alpha) is already correct, but sin(-alpha)=-sin(alpha) is
    # sign-flipped relative to the SINGLE convention every consumer uses
    # (+sin(alpha) at BOTH u- and v-faces: the ocean/ice/omip inverse stress
    # rotation j-row tau_j = -tau_e*sin_alpha_v + tau_n*cos_alpha_v with an
    # EXPLICIT minus and the +cos_alpha_v above, and the fold-halo relative
    # rotation).  Negate to restore +sin(alpha); cos is left unchanged.
    sin_alpha_v = jnp.where(mask_v, -jnp.sin(alpha_v_full), 0.0)

    return cos_alpha_u, sin_alpha_u, cos_alpha_v, sin_alpha_v


# Floor clamped onto every per-cell length metric [m]: keeps degenerate
# fold / land cells from producing zero or negative spacings.  Named so a
# caller deriving a coefficient from the metrics can detect saturation on
# it rather than pass a literal.
DEFAULT_MIN_DX_M: float = 1000.0


def create_tripole_grid(
    grid_file,
    *,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
    min_dx_m: float = DEFAULT_MIN_DX_M,
    fold_convention: str = "auto",
    allow_ambiguous_legacy_fold: bool = False,
    strip_north_rows: int = 0,
    fold_pivot: str = "legacy",
) -> LatLonCGridGeometry:
    """Load a tripolar grid from a NEMO mesh_mask NetCDF file.

    Parameters
    ----------
    grid_file : str, Path, or sequence of them
        Path to a NEMO ``mesh_mask*.nc`` / ``domcfg*.nc`` file, or the list
        ``[mesh_hgr.nc, mesh_zgr.nc, mask.nc]`` of an older split mesh (see
        :func:`mesh_file_list`).
    radius : float
        Sphere radius [m].  Overrides grid-file values for consistency
        with the rest of legoESM.
    omega : float
        Rotation rate [rad/s].
    dtype : optional
        Storage dtype.
    min_dx_m : float, optional
        Floor (m) clamped onto every per-cell length metric
        (``dx_T``, ``dy_T``, ``dx_u``, ``dy_u``, ``dx_v``, ``dy_v``)
        and squared onto every area metric (``area_T``, ``area_q``).
        ORCA1 has ~2.4% of cells with raw ``dx`` down to 1–4 m at
        the bipolar-cap convergence, which crushes the CFL even
        with implicit barotropics. Default 1 000 m matches the
        runner-side floor used in the 20-yr ORCA1 production run.
        Pass 0.0 to disable.
    strip_north_rows : int, default 0
        Drop this many rows from the NORTH end of every mesh field before
        building the geometry.  NEMO ``jperio=4`` (T-point pivot) meshes such
        as ORCA0083/ORCA12 end with a DEAD halo row above the self-dual pivot
        row; dropping it makes the pivot row the last row, which is the only
        fold layout the operators implement (last row identified with its own
        permutation).  0 = the mesh as stored (eORCA1.2 / eORCA025 layout).
    fold_convention : {"auto", "n_lon-1-i", "(n_lon-i)%n_lon"}, default "auto"
        T-fold index convention forwarded to ``_detect_fold``. ``"auto"``
        symmetry-detects it; on a genuinely ambiguous (near-constant) fold row
        where both conventions fit equally it RAISES (fail loud) rather than
        guess a seam origin that could silently corrupt the northern-halo
        exchange. Pass it EXPLICITLY for a de-haloed mesh whose fold row is too
        flat to disambiguate (``"(n_lon-i)%n_lon"`` for eORCA025-style de-haloed
        meshes, ``"n_lon-1-i"`` for halo-inclusive eORCA1.2-style meshes); the
        choice is still verified against the symmetry tolerance.
    fold_pivot : {"legacy", "F"}, default "legacy"
        ``"F"`` = explicit opt-in to the NEMO F-point-pivot fold descriptor
        (``FoldDescriptor.fpivot``); requires the duplicated halo row to be
        stripped (``strip_north_rows=1`` on eORCA1.2) and is verified against
        the mesh's V/F coordinates.  ``"legacy"`` = unchanged detection.
    allow_ambiguous_legacy_fold : bool, default False
        Explicit opt-in: when ``True`` and ``fold_convention="auto"`` hits an
        ambiguous fold row, fall back to the historical ``"n_lon-1-i"`` origin
        (with a warning) instead of raising. For callers who knowingly accept
        the legacy tie-break; prefer an explicit ``fold_convention``.

    Returns
    -------
    LatLonCGridGeometry
        With per-cell metrics, fold descriptor, and rotation angles.
    """
    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().storage
        except Exception:
            dtype = jnp.float32

    raw = _read_nemo_mesh_mask(grid_file)
    raw = strip_north_rows_raw(raw, strip_north_rows)

    # Coordinates at T-points [degrees -> radians]
    glamt = raw["glamt"]
    gphit = raw["gphit"]
    n_lat, n_lon = glamt.shape

    lat_T = jnp.deg2rad(gphit).astype(dtype)
    lon_T = jnp.deg2rad(glamt).astype(dtype)

    # Metric arrays [metres] — NEMO stores these as e1t, e2t etc.
    dx_T = raw["e1t"].astype(dtype)
    dy_T = raw["e2t"].astype(dtype)
    area_T = dx_T * dy_T
    total_area = jnp.sum(area_T)

    # u-point metrics.  NEMO e1u/e2u have shape (n_lat, n_lon) but on a C-grid
    # u-points have shape (n_lat, n_lon+1).
    #
    # NEMO's u-point ``i`` lies EAST of T-cell ``i`` -- between T(i) and T(i+1).
    # MEASURED off the mesh rather than taken from documentation: on the
    # 1-degree part of eORCA1, ``glamu - glamt = +0.5000`` deg at four
    # consecutive equatorial columns (and ``gphiv - gphit = +0.32`` deg, the
    # matching statement for v).  Our face ``i`` lies WEST of cell ``i`` --
    # ``f_u_inner`` just below averages ``f_T[i-1]`` and ``f_T[i]``, which is
    # the same statement.  So our face ``i`` must take NEMO's ``e1u[i-1]``, and
    # face 0 wraps to the LAST column by periodicity.
    #
    # This block previously appended ``e1u[:, 0:1]`` instead, which gave every
    # face the metric of the face one column EAST.  That is exactly a no-op
    # wherever the mesh does not vary along a row -- the whole southern
    # hemisphere and the tropics -- which is why it survived; ORCA's
    # quasi-isotropic northern grid begins near 20N, so it was a several- to
    # twenty-percent error on every zonal face north of 30N, including the
    # Gulf Stream and the Arctic.  Confirmed to reach the continuity operator
    # (20-31% of the local divergence in the worst percentile north of 30N,
    # and exactly zero in 30S-60S where the two builds are bit-identical) by
    # ``scripts/validate/ocean_fidelity/tripole_metric_divergence.py``.
    e1u = raw["e1u"].astype(dtype)
    e2u = raw["e2u"].astype(dtype)
    dx_u = jnp.concatenate([e1u[:, -1:], e1u], axis=1)  # (n_lat, n_lon+1)
    dy_u = jnp.concatenate([e2u[:, -1:], e2u], axis=1)

    # v-point metrics.  NEMO v-points have shape (n_lat, n_lon); the
    # extra row at the fold boundary needs special handling.
    e1v = raw["e1v"].astype(dtype)
    e2v = raw["e2v"].astype(dtype)
    # Prepend a south zero row (wall BC); the NEMO e1v/e2v arrays
    # already include the fold row as their last (northernmost) row,
    # so no north padding is needed — just the south zero gives
    # (n_lat+1, n_lon).
    dx_v = jnp.concatenate(
        [jnp.zeros((1, n_lon), dtype=dtype), e1v], axis=0
    )  # (n_lat+1, n_lon)
    dy_v = jnp.concatenate(
        [jnp.zeros((1, n_lon), dtype=dtype), e2v], axis=0
    )

    # Vertex area (f-point area in NEMO).  If e1f/e2f available, use them.
    if "e1f" in raw and "e2f" in raw:
        e1f = raw["e1f"].astype(dtype)
        e2f = raw["e2f"].astype(dtype)
        area_q_inner = e1f * e2f  # (n_lat, n_lon), NEMO F(ji,jj) = NE corner of T(ji,jj)
        # Model vertex (j, i) is the SW corner of T(j, i), i.e. NEMO F(ji=i-1,
        # jj=j-1): shift one column east (longitude-periodic) and one row
        # north (south wall row copies its neighbour), then append the wrap
        # column (vertex column n_lon == column 0).
        _aq = jnp.roll(area_q_inner, 1, axis=1)
        _aq = jnp.concatenate([_aq[:1], _aq], axis=0)            # (n_lat+1, n_lon)
        area_q = jnp.concatenate([_aq, _aq[:, :1]], axis=1)      # (n_lat+1, n_lon+1)
    else:
        # Estimate from T-point areas
        area_q = jnp.pad(area_T, ((0, 1), (0, 1)), mode="edge")

    # Coriolis
    f_T = (2.0 * omega * jnp.sin(lat_T)).astype(dtype)

    # f at u-points
    f_u_inner = 0.5 * (jnp.roll(f_T, 1, axis=1) + f_T)
    f_u = jnp.concatenate([f_u_inner, f_u_inner[:, 0:1]], axis=1)

    # f at v-points
    f_v_inner = 0.5 * (f_T[:-1] + f_T[1:])
    f_v = jnp.concatenate([f_T[0:1], f_v_inner, f_T[-1:]], axis=0)
    if fold_pivot == "F":
        # F-pivot: the top v row is the fold line between the top cell and
        # its mirror image (perm_T = n-1-i), so it takes the same two-cell
        # average as every interior v row (self-symmetric under perm_v).
        _pt = (n_lon - 1 - jnp.arange(n_lon)).astype(jnp.int32)
        f_v = f_v.at[-1].set(0.5 * (f_T[-1] + f_T[-1][_pt]))

    # Fold descriptor. ``_detect_fold`` raises on a genuinely ambiguous
    # (near-constant) fold row under "auto" because BOTH seam origins fit, and a
    # silent guess would corrupt the northern-boundary halo exchange / vector
    # sign-flip for the whole run (not just at startup). The default here is to
    # let that error propagate — fail loud and require an explicit
    # ``fold_convention``. Real ORCA meshes have a curved (non-constant) fold
    # row and never hit this. ``allow_ambiguous_legacy_fold=True`` is an
    # explicit opt-in that restores the historical "n_lon-1-i" tie-break (with a
    # warning) for callers who knowingly accept the legacy behaviour. Non-
    # ambiguity errors always propagate.
    if fold_pivot == "F":
        fold = _detect_fold(
            raw["glamt"], raw["gphit"], n_lat, n_lon, fold_pivot="F",
            vf_coords=(raw["glamv"], raw["gphiv"], raw["glamf"],
                       raw["gphif"]))
    else:
        fold = None
    try:
        if fold is None:
            fold = _detect_fold(raw["glamt"], raw["gphit"], n_lat, n_lon,
                                fold_convention=fold_convention,
                                fold_pivot=fold_pivot)
    except ValueError as exc:
        is_ambiguity = (
            fold_convention == "auto"
            and "Ambiguous fold_convention" in str(exc)
        )
        if not (is_ambiguity and allow_ambiguous_legacy_fold):
            raise
        import warnings
        warnings.warn(
            f"{exc} allow_ambiguous_legacy_fold=True → using the legacy "
            "'n_lon-1-i' fold origin, which may be WRONG for a de-haloed mesh; "
            "pass fold_convention explicitly for a guaranteed-correct origin.",
            RuntimeWarning,
            stacklevel=2,
        )
        fold = _detect_fold(raw["glamt"], raw["gphit"], n_lat, n_lon,
                            fold_convention="n_lon-1-i")

    # Rotation angles
    glamu = raw.get("glamu", glamt)
    gphiu = raw.get("gphiu", gphit)
    glamv = raw.get("glamv", glamt)
    gphiv = raw.get("gphiv", gphit)
    cos_alpha_u, sin_alpha_u, cos_alpha_v, sin_alpha_v = (
        _compute_rotation_angles(glamu, gphiu, glamv, gphiv, fold.cap_j)
    )
    cos_alpha_u = cos_alpha_u.astype(dtype)
    sin_alpha_u = sin_alpha_u.astype(dtype)
    cos_alpha_v = cos_alpha_v.astype(dtype)
    sin_alpha_v = sin_alpha_v.astype(dtype)

    # Pad rotation angles for staggered shapes
    cos_alpha_u = jnp.concatenate(
        [cos_alpha_u, cos_alpha_u[:, 0:1]], axis=1
    )
    sin_alpha_u = jnp.concatenate(
        [sin_alpha_u, sin_alpha_u[:, 0:1]], axis=1
    )
    cos_alpha_v = jnp.concatenate(
        [cos_alpha_v[0:1], cos_alpha_v], axis=0
    )
    sin_alpha_v = jnp.concatenate(
        [sin_alpha_v[0:1], sin_alpha_v], axis=0
    )

    # Legacy 1D fields (approximate — use column averages)
    lat_1d = jnp.mean(lat_T, axis=1)
    lon_1d = lon_T[0, :]
    cos_lat_1d = jnp.maximum(jnp.cos(lat_1d), 1e-10)
    # Legacy 1-D v-face REPRESENTATIVE, mirroring lat_1d/cos_lat_1d: a tripolar
    # grid has no true 1-D v-face axis (the fold rows are curvilinear), so this
    # is the same half-cell reconstruction on zonal-mean latitudes that lat_1d
    # already is.  Consumers needing the REAL v-face metric must use the 2-D
    # dx_v; the one lat-scaling consumer refuses on fold.is_active.  Deliberately
    # NOT a NaN sentinel: this leaf flows through jit, where a value-inspecting
    # guard is impossible and a NaN can only poison silently.
    cos_lat_v_1d = compute_v_face_coords(lat_1d, lat_1d[1] - lat_1d[0])[1]
    sin_lat_1d = jnp.sin(lat_1d)

    if min_dx_m > 0.0:
        dx_T = jnp.maximum(dx_T, min_dx_m)
        dy_T = jnp.maximum(dy_T, min_dx_m)
        dx_u = jnp.maximum(dx_u, min_dx_m)
        dy_u = jnp.maximum(dy_u, min_dx_m)
        dx_v = jnp.maximum(dx_v, min_dx_m)
        dy_v = jnp.maximum(dy_v, min_dx_m)
        area_T = jnp.maximum(area_T, min_dx_m * min_dx_m)
        area_q = jnp.maximum(area_q, min_dx_m * min_dx_m)

    return LatLonCGridGeometry(
        n_lat=n_lat,
        n_lon=n_lon,
        radius=float(radius),
        lat_T=lat_T,
        lon_T=lon_T,
        dx_T=dx_T,
        dy_T=dy_T,
        area_T=area_T,
        total_area=total_area,
        dx_u=dx_u,
        dy_u=dy_u,
        dx_v=dx_v,
        dy_v=dy_v,
        area_q=area_q,
        f_T=f_T,
        f_u=f_u,
        f_v=f_v,
        cos_alpha_u=cos_alpha_u,
        sin_alpha_u=sin_alpha_u,
        cos_alpha_v=cos_alpha_v,
        sin_alpha_v=sin_alpha_v,
        fold=fold,
        cos_lat=cos_lat_1d,
        sin_lat=sin_lat_1d,
        cos_lat_v=cos_lat_v_1d,
        lat=lat_1d,
        lon=lon_1d,
        dlon=0.0,   # sentinel: tripole grids have non-uniform spacing
        dlat=0.0,
        omega=float(omega),   # (#521) so grid.omega matches the f_T/f_u/f_v build
    )


def pad_tripole_grid_south(grid: LatLonCGridGeometry,
                           n_pad: int) -> LatLonCGridGeometry:
    """Prepend ``n_pad`` LAND rows to the SOUTH of a (tri)polar C-grid geometry.

    The lat-band SPMD ocean step requires ``n_lat % n_devices == 0`` (one uniform
    ``shard_map`` program per band).  A real mesh (eORCA025 ``n_lat=1207``) is not
    divisible by an arbitrary device count, so the driver appends land rows.  They
    go at the SOUTH because the bipolar fold is NORTH-relative
    (``fold_j = n_lat-1``, ``cap_j`` measured down from the north): padding the
    south leaves the fold seam at the (new) northernmost row and the bipolar cap
    band physically unchanged — only its ROW INDEX shifts by ``+n_pad``.

    Staggering (the C-grid convention in ``LatLonCGridGeometry``):
      * T/u-row arrays lead with ``n_lat``  -> prepend ``n_pad`` rows.
      * v/q-row arrays lead with ``n_lat+1`` -> prepend ``n_pad`` rows (the south
        wall row ``[0]`` stays a wall; the new rows are further-south walls).
      * the 1-D legacy ``lat``/``cos_lat``/``sin_lat`` (``n_lat``) -> prepend.

    Added-row VALUES: the south rows are LAND (the driver pads ``land_mask=0`` /
    ``H_bathy=0`` to match, so every operator masks them out), so their metric
    values only need to be FINITE + positive + monotone in latitude.  The metric
    lengths/areas/rotation copy the current southern EDGE row (positive, dynamics
    never sees them through the wet mask); ``lat_T`` extends southward by the
    southern row's meridional spacing (``dy_T / radius``) so latitude stays
    strictly increasing northward (Coriolis ``f_T = 2Ω sin(lat)`` follows); the
    rotation angles south of the cap are regular lat-lon (cos=1, sin=0) — and the
    edge row already is, below ``cap_j``.

    ``total_area`` is PRESERVED exactly (the area-weighted-mean denominator must
    not pick up the spurious land padding — the same GLOBAL-``total_area``
    invariant the band slicer keeps).  ``fold_j``/``cap_j`` shift ``+n_pad``;
    ``perm_T``/``perm_v`` (lon permutations) and the vector signs are unchanged.

    ``n_pad == 0`` returns the grid unchanged.  Requires an ACTIVE fold (the
    regular lat-lon path does not need this — it is divisor-padded differently /
    not at all); raises otherwise so a mis-wired caller fails loud.
    """
    if n_pad < 0:
        raise ValueError(f"n_pad must be >= 0, got {n_pad}")
    if n_pad == 0:
        return grid
    fold = grid.fold
    if fold is None or not bool(fold.is_active):
        raise ValueError(
            "pad_tripole_grid_south requires an ACTIVE bipolar fold (the south "
            "pad keeps the north fold at the new top row); got an inactive fold. "
            "A regular lat-lon grid does not use this helper.")

    n_lat = int(grid.n_lat)
    n_lon = int(grid.n_lon)
    dtype = grid.lat_T.dtype

    def _prepend_rows(arr, rows):
        return jnp.concatenate([rows.astype(arr.dtype), arr], axis=0)

    def _edge_pad(arr):
        """Prepend ``n_pad`` copies of the southern edge row ``arr[0]``."""
        edge = jnp.broadcast_to(arr[0:1], (n_pad,) + arr.shape[1:])
        return _prepend_rows(arr, edge)

    # The n_pad new south rows are LAND (the driver pads land_mask=0 / H_bathy=0),
    # so dynamics never sees them through the wet mask.  Two requirements: the
    # ORIGINAL rows stay BIT-EXACT (so a padded run reproduces the unpadded one on
    # the wet domain), and the latitude stays STRICTLY INCREASING northward (no
    # zero/negative dlat at the seam).  => PREPEND the new rows (preserving the
    # originals) and compute the new rows' latitude-derived fields FROM the new
    # latitude; do NOT recompute the whole array (recomputing f_T=2Ω·sin(lat) or
    # the 1-D lat off a rebuilt lat_T perturbs the originals off the build's
    # values — f_T is O(1e-4), the bit-exactness break the pad test caught).
    radius = float(grid.radius)
    dlat_row = (jnp.asarray(grid.dy_T)[0] / radius).astype(dtype)   # (n_lon,) [rad]
    south_offsets = (jnp.arange(n_pad, 0, -1, dtype=dtype)[:, None]
                     * dlat_row[None, :])                           # (n_pad, n_lon)
    lat_new = (grid.lat_T[0:1] - south_offsets).astype(dtype)       # monotone south
    # Past -90 deg the cosine goes NEGATIVE and the 1e-10 clamps below flatten
    # it (GLM review 2026-08-24).  The pad's contract for LAND rows is only
    # finite + positive, which the clamp preserves — and the synthetic
    # full-sphere test grid legitimately extrapolates past the pole (12-deg
    # rows), so this is a LOUD WARNING, not an error.  Production eORCA
    # meshes (south edge ~-80 deg, <=3 pad rows of ~0.25 deg) never trigger
    # it; if a run log shows this line, inspect the mesh before trusting any
    # diagnostic that reads latitude off the padded rows.
    _lat_min = float(jnp.min(lat_new))
    if _lat_min <= -0.5 * float(jnp.pi):
        print(f"[pad_tripole_grid_south] WARNING: extrapolated south "
              f"latitude {_lat_min:.4f} rad crosses the pole; the {n_pad} "
              f"pad rows are LAND and their clamped metrics stay finite and "
              f"positive, but their latitude values are not physical.",
              flush=True)
    lat_T_pad = _prepend_rows(grid.lat_T, lat_new)
    lon_T_pad = _edge_pad(grid.lon_T)                               # lon unchanged

    # Coriolis on the NEW rows from their latitude; ORIGINAL f_T/f_u preserved.
    f_T_new = (2.0 * constants.Omega * jnp.sin(lat_new)).astype(dtype)
    f_T_pad = _prepend_rows(grid.f_T, f_T_new)
    fu_inner_new = 0.5 * (jnp.roll(f_T_new, 1, axis=1) + f_T_new)   # 4-pt zonal avg
    f_u_new = jnp.concatenate([fu_inner_new, fu_inner_new[:, 0:1]], axis=1)
    f_u_pad = _prepend_rows(grid.f_u, f_u_new)
    f_v_pad = _edge_pad(grid.f_v)              # v-row (n_lat+1 leading); wall, unused

    # 1-D legacy fields: prepend the new rows' zonal-mean latitude (originals kept).
    lat_new_1d = jnp.mean(lat_new, axis=1)                         # (n_pad,)
    lat_1d_pad = jnp.concatenate([lat_new_1d, jnp.asarray(grid.lat, dtype)])
    cos_lat_pad = jnp.concatenate(
        [jnp.maximum(jnp.cos(lat_new_1d), 1e-10),
         jnp.asarray(grid.cos_lat, dtype)])
    sin_lat_pad = jnp.concatenate(
        [jnp.sin(lat_new_1d), jnp.asarray(grid.sin_lat, dtype)])

    # cos_lat_v is the (n_lat+1,) v-FACE profile — it must grow with the v
    # rows or the SPMD band slicer's [s:e+1] on the padded grid hands the
    # NORTH band one row fewer than the interior bands, and the per-field
    # band stack fails with "All input arrays must have the same shape"
    # (the eORCA025 full-card 4-GPU smoke, job 9471878: this field was
    # MISSED when it was added to the geometry after this pad was written,
    # and eORCA1's n_lat=332 divides evenly so the pad — and the miss —
    # never fired there).  New face values = clamped cos of the new rows'
    # latitudes (land rows; finite + positive is all dynamics requires).
    cos_lat_v_pad = jnp.concatenate(
        [jnp.maximum(jnp.cos(lat_new_1d), 1e-10).astype(dtype),
         jnp.asarray(grid.cos_lat_v, dtype)])

    # seam_wall_rows is an OPTIONAL (n_lat,) per-row profile (None on the
    # eORCA builds today, set by the DINO bridge).  If present it must grow
    # too, or the same band-slice raggedness bites; the new rows are LAND,
    # so the seam face there is WALLED (1.0).
    seam_pad = grid.seam_wall_rows
    if seam_pad is not None:
        seam_pad = jnp.concatenate(
            [jnp.ones((n_pad,), dtype=jnp.asarray(seam_pad).dtype),
             jnp.asarray(seam_pad)])

    native_lat_pad = grid.native_lat_T_deg
    if native_lat_pad is not None:
        native_lat_pad = jnp.concatenate(
            [jnp.degrees(lat_new).astype(jnp.asarray(native_lat_pad).dtype),
             jnp.asarray(native_lat_pad)], axis=0)

    return grid._replace(
        n_lat=n_lat + n_pad,
        lat_T=lat_T_pad,
        lon_T=lon_T_pad,
        dx_T=_edge_pad(grid.dx_T),
        dy_T=_edge_pad(grid.dy_T),
        area_T=_edge_pad(grid.area_T),
        total_area=grid.total_area,            # PRESERVED (no spurious land area)
        dx_u=_edge_pad(grid.dx_u),
        dy_u=_edge_pad(grid.dy_u),
        dx_v=_edge_pad(grid.dx_v),
        dy_v=_edge_pad(grid.dy_v),
        area_q=_edge_pad(grid.area_q),
        f_T=f_T_pad,
        f_u=f_u_pad,
        f_v=f_v_pad,
        cos_alpha_u=_edge_pad(grid.cos_alpha_u),
        sin_alpha_u=_edge_pad(grid.sin_alpha_u),
        cos_alpha_v=_edge_pad(grid.cos_alpha_v),
        sin_alpha_v=_edge_pad(grid.sin_alpha_v),
        fold=fold._replace(fold_j=int(fold.fold_j) + n_pad,
                           cap_j=int(fold.cap_j) + n_pad),
        cos_lat=cos_lat_pad,
        sin_lat=sin_lat_pad,
        cos_lat_v=cos_lat_v_pad,
        lat=lat_1d_pad,
        seam_wall_rows=seam_pad,
        native_lat_T_deg=native_lat_pad,
        # lon (n_lon,) unchanged; dlon/dlat sentinels unchanged.
    )


# =========================================================================
# Synthetic tripolar grid for testing
# =========================================================================


def create_synthetic_tripole(
    n_lat: int = 36,
    n_lon: int | None = None,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
) -> LatLonCGridGeometry:
    """Create a synthetic tripolar grid for testing.

    This generates a regular lat-lon grid with an **active fold
    descriptor** at the northern boundary.  The metrics are regular
    lat-lon everywhere (no actual bipolar cap distortion), but the
    fold is correctly configured so that:

    - Fold halo exchange can be tested (scalar and vector round-trips).
    - Operator boundary handling via ``pad_ns_*`` dispatches to the
      fold branch rather than the wall-BC branch.
    - Vector sign flips at the fold are exercised.

    This is sufficient for developing and testing Phases 1A through 3
    of the tripolar implementation without requiring a 484 MB ORCA1
    grid file download.

    Parameters
    ----------
    n_lat : int
        Number of latitude cells.
    n_lon : int, optional
        Number of longitude cells (default: 2 * n_lat).
    radius : float
        Sphere radius [m].
    omega : float
        Rotation rate [rad/s].
    dtype : optional
        Storage dtype.

    Returns
    -------
    LatLonCGridGeometry
        Regular lat-lon metrics but with ``fold.is_active = True``.
    """
    # Start from a regular lat-lon geometry
    geom = create_latlon_geometry(n_lat, n_lon, radius, omega, dtype)
    n_lon_eff = geom.n_lon

    # Create an active fold descriptor at the northern boundary
    fold_j = n_lat - 1
    cap_j = max(0, n_lat - n_lat // 4)  # cap covers top ~25% of grid

    perm_T = jnp.arange(n_lon_eff - 1, -1, -1, dtype=jnp.int32)
    perm_v = perm_T.copy()

    fold = FoldDescriptor(
        is_active=True,
        fold_j=fold_j,
        cap_j=cap_j,
        perm_T=perm_T,
        perm_v=perm_v,
        vector_sign_u=-1.0,
        vector_sign_v=-1.0,
    )

    # Replace the inactive fold with the active one.
    # NamedTuple._replace creates a shallow copy with the specified
    # field changed.
    return geom._replace(fold=fold)


def create_synthetic_tripole_pivot(n_lat: int, n_lon: int | None = None,
                                   radius: float = constants.R_earth,
                                   omega: float = constants.Omega,
                                   dtype=None):
    """Synthetic tripole with the DE-HALOED pivot-row-stored fold layout.

    The eORCA025 storage convention (measured 2026-08-26, see
    ``check_tripole_fold_pairing.py``): the stored top T row is the
    SELF-symmetric T-pivot row — cell ``(i, j_max)`` and
    ``(perm_T[i], j_max)`` are the same physical cell under
    ``perm_T = (n_lon - i) % n_lon`` — the duplicated halo row was
    stripped, U points self-map under ``(n_lon - i - 1) % n_lon``, and
    V/F rows pair with the row BELOW.  ``create_synthetic_tripole``
    models the OTHER layout (eORCA1.2, halo row stored); the two fixtures
    together pin both fold code paths.

    Returns a regular lat-lon geometry with the pivot-layout fold
    descriptor attached (same idealization as the halo-layout fixture:
    metrics stay regular; only the fold BC dispatch is exercised).
    """
    geom = create_latlon_geometry(n_lat, n_lon, radius, omega, dtype)
    n_lon_eff = geom.n_lon
    idx = jnp.arange(n_lon_eff, dtype=jnp.int32)
    fold = FoldDescriptor(
        is_active=True,
        fold_j=n_lat - 1,
        cap_j=max(0, n_lat - n_lat // 4),
        perm_T=(n_lon_eff - idx) % n_lon_eff,
        perm_v=(n_lon_eff - idx) % n_lon_eff,
        vector_sign_u=-1.0,
        vector_sign_v=-1.0,
        pivot_row_stored=True,
        perm_u=(n_lon_eff - idx - 1) % n_lon_eff,
        perm_f=(n_lon_eff - idx - 1) % n_lon_eff,
    )
    return geom._replace(fold=fold)


def download_orca1_grid(
    dest_dir: str | Path = "data/grids",
    *,
    connect_timeout_s: float = 30.0,
) -> Path:
    """Download the eORCA1 mesh_mask from Zenodo.

    Parameters
    ----------
    dest_dir : str or Path
        Directory to save the file.
    connect_timeout_s : float, default 30.0
        Socket connect/read timeout in seconds. Prevents an unreachable
        Zenodo mirror from hanging the caller indefinitely.

    Returns
    -------
    Path
        Path to the downloaded ``eORCA1.2_mesh_mask.nc`` file.
    """
    import shutil
    import socket
    import urllib.request

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    filepath = dest / "eORCA1.2_mesh_mask.nc"

    if filepath.exists():
        print(f"ORCA1 grid already exists at {filepath}")
        return filepath

    url = (
        "https://zenodo.org/records/4436658/files/"
        "eORCA1.2_mesh_mask.nc?download=1"
    )
    print(f"Downloading eORCA1 mesh_mask (~484 MB) to {filepath}...")
    # Stream to a temp file and rename on success so a partial download
    # does not masquerade as a finished grid file on disk.
    tmp_path = filepath.with_suffix(filepath.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=connect_timeout_s) as response, \
                open(tmp_path, "wb") as fout:
            shutil.copyfileobj(response, fout)
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Failed to download eORCA1 mesh_mask from {url}: {e}. "
            f"Increase connect_timeout_s or download manually to {filepath}."
        ) from e
    tmp_path.rename(filepath)
    print(f"Download complete: {filepath}")
    return filepath


def south_pad_rows(n_lat: int, n_devices: int) -> int:
    """Number of LAND rows to prepend at the SOUTH so ``n_lat`` is a multiple
    of ``n_devices`` (the lat-band SPMD step needs one uniform band per device).

    eORCA025 ``n_lat=1207`` is odd: for ``n_devices=2`` this returns 1 (-> 1208).
    Returns 0 when already divisible (or ``n_devices <= 1``).
    """
    if n_devices <= 1:
        return 0
    rem = n_lat % n_devices
    return 0 if rem == 0 else (n_devices - rem)


def pad_mask_bathy_south(land_mask, H_bathy, n_pad: int):
    """Prepend ``n_pad`` LAND rows (mask=0, bathy=0) to the SOUTH of the cell
    ``(n_lat, n_lon)`` land-mask + bathymetry arrays.

    Pairs with :func:`pad_tripole_grid_south` (which pads the GRID geometry the
    same way + keeps the north fold): the padded mask/bathy + grid are fed to
    the SAME rest-state / WOA-fill path, so the state is built on the padded
    grid with the added rows masked LAND (inert dynamics).  The wet rows are
    preserved bit-exact, shifted ``+n_pad`` in the lat index.
    """
    import numpy as np
    if n_pad <= 0:
        return land_mask, H_bathy
    lm = np.asarray(land_mask)
    hb = np.asarray(H_bathy)
    n_lon = lm.shape[1]
    zeros_lm = np.zeros((n_pad, n_lon), dtype=lm.dtype)
    zeros_hb = np.zeros((n_pad, n_lon), dtype=hb.dtype)
    return (np.concatenate([zeros_lm, lm], axis=0),
            np.concatenate([zeros_hb, hb], axis=0))


def create_synthetic_tripole_fpivot(n_lat: int, n_lon: int | None = None,
                                    radius: float = constants.R_earth,
                                    omega: float = constants.Omega,
                                    dtype=None, ew_halo: bool = False,
                                    lat_1d=None):
    """Synthetic tripole with the NEMO F-POINT-pivot fold (halo row stripped).

    The eORCA1 layout after ``strip_north_rows=1`` (measured 2026-09-28,
    ``measure_fpivot_fold_perms.py``): the fold line is the north v-face of
    the stored top row; ``perm_T = perm_v = n-1-i`` and
    ``perm_u = perm_f = (n-i) % n`` (see ``FoldDescriptor.fpivot``).  Regular
    lat-lon metrics (each row uniform in longitude, hence fold-symmetric);
    only the fold dispatch is exercised.  ``n_lon`` must be even.
    ``ew_halo=True`` selects NEMO's halo-column index sets (the geometry
    itself stays purely periodic).  ``lat_1d`` (radians, optional) gives a
    regional band so the fold line sits away from the geographic pole; the
    top v-face width and vertex area are opened into a fold line
    (:func:`fpivot_fold_line_geometry`).
    """
    geom = create_latlon_geometry(n_lat, n_lon, radius, omega, dtype,
                                  lat_1d=lat_1d)
    return fpivot_fold_line_geometry(
        geom._replace(fold=fpivot_descriptor(n_lat, geom.n_lon,
                                             ew_halo=ew_halo)))


def fpivot_fold_line_geometry(geom):
    """Open the regular builder's north WALL row into an F-pivot fold line.

    ``create_latlon_geometry`` zeroes the top v-face width (a pole wall) and
    keeps only the southern half of the top vertex dual cell.  On the F-pivot
    fold the top v-face is an interior face shared with the mirror cell, so
    its width is the face-latitude ``R cos(phi) dlon`` and the dual cell is
    twice the stored half (its northern half is the mirror of the southern).
    Test-grid helper for the synthetic fixtures only (a mesh file supplies
    these metrics directly)."""
    lat_top_face = float(geom.lat[-1]) + 0.5 * float(geom.lat[-1] - geom.lat[-2])
    dlon = 2.0 * jnp.pi / geom.n_lon
    dxv_top = jnp.full((1, geom.n_lon),
                       geom.radius * jnp.cos(lat_top_face) * dlon,
                       dtype=geom.dx_v.dtype)
    return geom._replace(
        dx_v=jnp.concatenate([geom.dx_v[:-1], dxv_top], axis=0),
        area_q=jnp.concatenate([geom.area_q[:-1], 2.0 * geom.area_q[-1:]],
                               axis=0))
