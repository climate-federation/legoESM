"""NEMO restart + mesh_mask reader for the differentiable-NEMO fidelity harness.

The NEMO counterpart of :mod:`mitgcm_io`. NEMO writes plain NetCDF (via the
native ``iom_nf90`` path — no XIOS needed), so this is a thin xarray reader; the
only conventions it reconciles are:

* **Halo strip.** ``nn_hls`` here is the FILE's halo, not the run's: NEMO
  <= 4.0 wrote global arrays WITH an ``nn_hls``-cell halo per side
  (``nn_hls=1`` for GYRE: 30x20 stored 32x22); NEMO 4.2+/5.x writes the
  COMPUTE domain WITHOUT halos (pass ``nn_hls=0`` — DINO 5.0.2 files are
  52x199 all-real; stripping a phantom halo discards the land-wall and
  ridge columns, #1226 root cause). Interior is ``[h:-h, h:-h]`` for h>0.
* **Axis order.** NEMO 3-D fields are ``(z, y, x)`` on disk; legoESM wants the
  vertical LAST — ``(y=lat, x=lon, z=lev)`` — a single ``moveaxis(0, -1)``.
* **Vertical order.** NEMO ``k=1`` is the surface, ``k`` increasing downward —
  the SAME top-down order legoESM uses, so there is NO z-reversal.

This module only READS + reshapes into plain NumPy; the staggering / grid
construction (NEMO east-face ``U`` -> legoESM ``u_face``, the beta-plane
geometry) lives in :mod:`nemo_state_bridge`, per the harness/model split in
``docs/ocean/fidelity/oracle_recipe_strategy.md``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import xarray as xr


class NemoGrid(NamedTuple):
    """Halo-stripped NEMO horizontal+vertical grid from a ``mesh_mask.nc``.

    Horizontal fields are ``(n_lat, n_lon)``; 3-D masks are ``(n_lat, n_lon,
    nlev)``; 1-D vertical fields are ``(nlev,)``. All in NumPy float64.
    """
    glamt: np.ndarray        # T-point longitude [deg] (n_lat, n_lon)
    gphit: np.ndarray        # T-point latitude  [deg]
    e1t: np.ndarray          # zonal T-cell width  [m]
    e2t: np.ndarray          # merid T-cell height [m]
    e1u: np.ndarray          # zonal U-cell width  [m]
    e2v: np.ndarray          # merid V-cell height [m]
    ff_t: np.ndarray         # Coriolis at T-points [1/s]
    ff_f: np.ndarray         # Coriolis at F-points (NE corner) [1/s]
    e3t_1d: np.ndarray       # reference T-cell thickness [m] (nlev,)
    gdept_1d: np.ndarray     # reference T-point depth [m] (nlev,)
    gdepw_1d: np.ndarray     # reference W-point (interface) depth [m] (nlev,)
    tmask: np.ndarray        # T-point wet mask (n_lat, n_lon, nlev)
    umask: np.ndarray        # U-point wet mask
    vmask: np.ndarray        # V-point wet mask
    # V-point latitude (north cell faces) [deg] (n_lat, n_lon) — only needed by
    # the Mercator/topography bridge for exact meridional cell faces; optional so
    # existing flat-bottom NemoGrid constructors (GYRE) stay valid.
    # NEMO's ACTUAL 3-D vertical scale factors (key_vco_3d). NEMO integrates
    # with THESE, not with the 1-D reference ladder above: for DINO, e3t_1d is
    # the unstretched analytic ladder (sums to 4506.375 m) while e3t_0 is
    # stretched so the deepest wet column is exactly the domain depth
    # (4000.000 m). They agree in the upper ocean and diverge below ~2000 m by
    # up to 12.9% (#1226). Optional so existing GYRE constructors stay valid --
    # GYRE is key_linssh where the two coincide, which is why this went
    # unnoticed.
    e3t_0: np.ndarray | None = None      # (n_lat, n_lon, nlev) [m]
    gdept_0: np.ndarray | None = None    # (n_lat, n_lon, nlev) [m]
    gphiv: np.ndarray | None = None
    # #1226 item 2 (dom_qco_r3c r3u/r3v): NEMO's reference u-/v-column depths
    # ``hu_0 = sum_k(e3u_0*umask)`` / ``hv_0 = sum_k(e3v_0*vmask)``
    # (dom_oce.F90:345-346, domain.F90:140-146) — the denominator of the
    # face-point z-star ratios ``r3u = ssh_face/hu_0``, ``r3v = ssh_face/hv_0``
    # (domqco.F90:166-169). Derived here from ``e3u_0``/``e3v_0`` (already in
    # mesh_mask.nc) + ``umask``/``vmask``, not re-dumped from NEMO.
    e3u_0: np.ndarray | None = None      # (n_lat, n_lon, nlev) [m]
    e3v_0: np.ndarray | None = None      # (n_lat, n_lon, nlev) [m]
    hu_0: np.ndarray | None = None       # (n_lat, n_lon) [m]
    hv_0: np.ndarray | None = None       # (n_lat, n_lon) [m]
    # Partial-periodic seam-wall profile, shape ``(n_lat,)``, 1.0 = the
    # zonal periodic-seam u-face is WALLED at that latitude row, 0.0 =
    # open/re-entrant.  Derived from the RAW (un-stripped) surface
    # ``tmask`` west-outer-halo column (col 0), which NEMO fills with land
    # outside a partial channel (DINO: land everywhere except the ACC
    # band) while the interior stays all-wet.  ``None`` = fully periodic
    # (no partial seam) or a config whose halo carries no wall.
    seam_wall_rows: np.ndarray | None = None


class NemoState(NamedTuple):
    """Halo-stripped NEMO prognostic state from a restart file.

    Tracers/density are ``(n_lat, n_lon, nlev)`` at T-points; velocities are at
    their NEMO faces (still T-shaped here — the face-index convention is applied
    in the bridge); ``ssh`` is ``(n_lat, n_lon)``.
    """
    T: np.ndarray            # potential temperature [degC]
    S: np.ndarray            # practical salinity [PSU]
    u: np.ndarray            # zonal velocity at NEMO U-points (east face)
    v: np.ndarray            # meridional velocity at NEMO V-points (north face)
    ssh: np.ndarray          # sea-surface height [m]
    rhd: np.ndarray | None   # in-situ density anomaly (rho-rho0)/rho0, if dumped


def _check_hls(nn_hls: int) -> None:
    # nn_hls must match the FILE, not the run: NEMO <= 4.0 wrote global
    # arrays WITH the halo (GYRE 30x20 stored 32x22 -> nn_hls=1), but NEMO
    # 4.2+/5.x writes the COMPUTE domain WITHOUT halos (DINO 5.0.2:
    # jpiglo=56 at runtime with nn_hls=2, files 52 wide -> nn_hls=0).
    # Stripping a phantom halo discards REAL boundary columns/rows (the
    # DINO land-wall + ridge columns; #1226 root cause) — pass 0 for
    # halo-free files.
    if nn_hls < 0:
        raise ValueError(f"nn_hls must be >= 0; got {nn_hls}.")


def _strip_halo_2d(a: np.ndarray, nn_hls: int) -> np.ndarray:
    _check_hls(nn_hls)
    h = nn_hls
    a = np.asarray(a, dtype=np.float64)
    if h == 0:      # a[0:-0] would be an empty slice — no-op explicitly
        return a
    out = a[h:-h, h:-h]
    assert out.shape == (a.shape[0] - 2 * h, a.shape[1] - 2 * h)
    return out


def _to_latlon_lev(a: np.ndarray, nn_hls: int) -> np.ndarray:
    """``(z, y, x)`` with halo -> ``(y-h, x-h, z)`` interior, vertical last."""
    _check_hls(nn_hls)
    a = np.asarray(a, dtype=np.float64)
    h = nn_hls
    if h == 0:
        return np.moveaxis(a, 0, -1)
    return np.moveaxis(a[:, h:-h, h:-h], 0, -1)


def read_nemo_mesh_mask(path: str, *, nn_hls: int = 1) -> NemoGrid:
    """Read + halo-strip a NEMO ``mesh_mask.nc`` into a :class:`NemoGrid`."""
    m = xr.open_dataset(path, decode_times=False)

    def h2(name: str) -> np.ndarray:
        return _strip_halo_2d(np.asarray(m[name].values).squeeze(), nn_hls)

    def m3(name: str) -> np.ndarray:
        return _to_latlon_lev(np.asarray(m[name].values).squeeze(), nn_hls)

    def v1(name: str) -> np.ndarray:
        return np.asarray(m[name].values).ravel().astype(np.float64)

    # Partial-periodic seam wall from the RAW halo (before the strip): the
    # surface T-mask west-outer-halo column (col 0) is land at latitudes
    # where the periodic seam is closed (DINO: outside the ACC channel);
    # strip it to the interior rows.  Only a GENUINE partial seam (some
    # walled, some open) sets the field — a fully re-entrant or fully
    # walled config leaves it ``None`` (byte-identical downstream).
    if nn_hls == 0:
        # Halo-free file (NEMO 4.2+/5.x): column 0 is a REAL domain column
        # (DINO: the land-wall continent itself), not a halo probe — the
        # wall is carried by the land mask, so no seam fabrication.
        _seam_wall = None
    else:
        _tmask_raw = np.asarray(m["tmask"].values).squeeze()  # (z,y,x) w/ halo
        _tsurf_west_halo = _tmask_raw[0, nn_hls:-nn_hls, 0]   # interior rows
        _seam_wall = (_tsurf_west_halo < 0.5).astype(np.float64)  # 1 = walled
        if not (_seam_wall.any() and (_seam_wall < 0.5).any()):
            _seam_wall = None

    umask_3d = m3("umask")
    vmask_3d = m3("vmask")
    # #1226 item 2: hu_0/hv_0 = sum_k(e3u_0*umask) / sum_k(e3v_0*vmask)
    # (domain.F90:140-146), computed here from e3u_0/e3v_0 (in mesh_mask.nc)
    # -- only when the thickness field is present (umask/vmask are mandatory
    # NemoGrid fields, read unconditionally above; a mesh_mask.nc missing
    # them KeyErrors earlier, so no separate guard is needed here).
    if "e3u_0" in m:
        e3u_0_arr = m3("e3u_0")
        hu_0_arr = (e3u_0_arr * umask_3d).sum(axis=-1)
    else:
        e3u_0_arr = None
        hu_0_arr = None
    if "e3v_0" in m:
        e3v_0_arr = m3("e3v_0")
        hv_0_arr = (e3v_0_arr * vmask_3d).sum(axis=-1)
    else:
        e3v_0_arr = None
        hv_0_arr = None

    return NemoGrid(
        glamt=h2("glamt"), gphit=h2("gphit"),
        e1t=h2("e1t"), e2t=h2("e2t"), e1u=h2("e1u"), e2v=h2("e2v"),
        ff_t=h2("ff_t"), ff_f=h2("ff_f"),
        e3t_1d=v1("e3t_1d"), gdept_1d=v1("gdept_1d"), gdepw_1d=v1("gdepw_1d"),
        tmask=m3("tmask"), umask=umask_3d, vmask=vmask_3d,
        e3t_0=(m3("e3t_0") if "e3t_0" in m else None),
        gdept_0=(m3("gdept_0") if "gdept_0" in m else None),
        gphiv=(h2("gphiv") if "gphiv" in m else None),
        seam_wall_rows=_seam_wall,
        e3u_0=e3u_0_arr, e3v_0=e3v_0_arr, hu_0=hu_0_arr, hv_0=hv_0_arr,
    )


def read_nemo_restart(path: str, *, nn_hls: int = 1) -> NemoState:
    """Read + halo-strip a NEMO restart. Uses the ``tn/sn/un/vn`` (Kbb) fields.

    ``rhd`` is included when the restart carries it (the MY_SRC EOS dump);
    otherwise ``None``.
    """
    r = xr.open_dataset(path, decode_times=False)

    def m3(name: str) -> np.ndarray:
        return _to_latlon_lev(np.asarray(r[name].values).squeeze(), nn_hls)

    return NemoState(
        T=m3("tn"), S=m3("sn"), u=m3("un"), v=m3("vn"),
        ssh=_strip_halo_2d(np.asarray(r["sshn"].values).squeeze(), nn_hls),
        rhd=(m3("rhd") if "rhd" in r else None),
    )


class NemoBeforeState(NamedTuple):
    """Halo-stripped NEMO leap-frog BEFORE-level (Nbb, ``tb/sb/ub/vb``) state.

    Same shape/axis conventions as :class:`NemoState`. Separate from
    :class:`NemoState` (not every caller needs the before level — only the
    MLF twin bridging NEMO's leap-frog integrator memory, #1317
    ``--bridge-before``). ``ssh``/``tau_x``/``tau_y`` are ``None`` when the
    restart does not carry ``sshb``/``utau_b``/``vtau_b`` (older NEMO
    restarts write only the tracer/velocity before-fields).
    """
    T: np.ndarray
    S: np.ndarray
    u: np.ndarray
    v: np.ndarray
    ssh: np.ndarray | None
    tau_x: np.ndarray | None   # utau_b, T-point wind stress [Pa], before-level
    tau_y: np.ndarray | None   # vtau_b


def read_nemo_restart_before(path: str, *, nn_hls: int = 1) -> NemoBeforeState:
    """Read + halo-strip NEMO's leap-frog BEFORE-level fields (Nbb).

    Uses ``tb/sb/ub/vb`` (+ ``sshb``/``utau_b``/``vtau_b`` when present) --
    the modified-leap-frog integrator's THIRD time level, one full step
    behind the ``tn/sn/un/vn`` (Kbb) read by :func:`read_nemo_restart`. Only
    meaningful for a restart written under ``outer_integrator="leapfrog"``
    (NEMO ``stp_MLF``/key_qco); a forward-Euler/AB2 restart still writes
    these fields (NEMO always carries a before-level in the restart file)
    but a legoESM twin only reads them via ``--bridge-before``
    (``kamm_twin_90d.py``), which requires the leapfrog card.
    """
    r = xr.open_dataset(path, decode_times=False)

    def m3(name: str) -> np.ndarray:
        return _to_latlon_lev(np.asarray(r[name].values).squeeze(), nn_hls)

    def h2(name: str) -> np.ndarray:
        return _strip_halo_2d(np.asarray(r[name].values).squeeze(), nn_hls)

    return NemoBeforeState(
        T=m3("tb"), S=m3("sb"), u=m3("ub"), v=m3("vb"),
        ssh=(h2("sshb") if "sshb" in r else None),
        tau_x=(h2("utau_b") if "utau_b" in r else None),
        tau_y=(h2("vtau_b") if "vtau_b" in r else None),
    )


def read_nemo_restart_en(path: str, *, nn_hls: int = 1) -> np.ndarray:
    """Read + halo-strip NEMO's TKE restart field ``en`` (w-levels, T-points).

    Not part of :class:`NemoState` (``en`` is closure-scheme integrator memory,
    not a core prognostic field every caller needs) -- a standalone reader for
    fidelity harnesses that want to bridge the TKE closure's cold-start memory
    specifically (e.g. the #1317 ``--bridge-tke`` twin experiment).

    Returns ``(n_lat, n_lon, jpk)`` with ``jpk`` NEMO w-levels, index 0 = the
    surface w-level (``gdepw_1d[0]=0``), same convention as ``gdepw_1d``.
    ``jpk == nlev`` (NEMO's w-levels and T-levels share the same
    ``nav_lev``-length axis). The caller maps this onto legoESM's ``nlev-1``
    interior interfaces (``state.tke``, dims ``("lat","lon","level")``) by
    dropping only the surface w-level (index 0): ``en[..., 1:jpk]`` has
    exactly ``jpk-1 == nlev-1`` levels, aligned index-for-index with lego's
    interior interfaces 0..nlev-2 -- see ``kamm_twin_90d.py``'s
    ``--bridge-tke``.
    """
    r = xr.open_dataset(path, decode_times=False)
    return _to_latlon_lev(np.asarray(r["en"].values).squeeze(), nn_hls)


__all__ = ("NemoGrid", "NemoState", "NemoBeforeState", "read_nemo_mesh_mask",
           "read_nemo_restart", "read_nemo_restart_before", "read_nemo_restart_en")
