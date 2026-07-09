#!/usr/bin/env python
"""Ocean single-column "twins": legoESM SCM vs NEMO ORCA1 columns.

Science goal — disconnect DYNAMICS from PHYSICS for debugging-via-comparison
against NEMO.  At a set of named diagnostic points, the legoESM dynamics-free
:class:`legoesm.ocean.scm.OceanColumnModel` (the SAME physics factory as the
3-D model: KPP / Richardson / constant vertical mixing, convective adjustment,
optional shortwave penetration, implicit vertical diffusion) is initialised
from NEMO's own monthly initial-condition profile and driven by the SAME
CORE-II NYF forcing + NCAR bulk formulas the 3-D OMIP run uses
(:func:`legoesm.ocean.coupler.compute_omip2_surface_forcing`).  Two tiers:

* **T0** — zero dynamics: ``f_c = 0``, no wind stress => ``u = v = 0``
  exactly; pure buoyancy-driven column physics (heat + freshwater).
* **T1** — Ekman: ``f_c = 2 Omega sin(lat)`` (``legoesm.constants.Omega``),
  wind stress on; prognostic column ``u, v`` with the Crank–Nicolson
  Coriolis rotation of the SCM.

REFEREE CAVEAT (printed on every output): until a NEMO C1D build exists, the
referee is NEMO's 3-D monthly output at the same points.  Those columns
INCLUDE lateral+vertical ADVECTION (and sea ice, SSS restoring, runoff),
which the SCM does not represent — differences are (advection + missing
forcings) + physics, not physics alone.

Forcing wiring (reuse, not re-derivation)
-----------------------------------------
``load_core2_nyf`` provides atmospheric STATE (u10, v10, T_air, q_air,
sw_down, lw_down, precip[, snow, slp]) — not fluxes — so fluxes must be
derived per step from the NCAR bulk formulas, which need the evolving model
SST.  The hot loop is ONE ``jax.jit``-compiled step per (point, tier, vmix)
run (:func:`build_jitted_step`, built ONCE before the loop): the per-point
channel time series is extracted once with numpy
(:func:`extract_point_forcing_series`, same nearest-neighbour convention as
the 3-D producers via :func:`core2_forcing_nn_indices`) and the step derives
the fluxes IN-TRACE from the SCM's own current state through the SAME
pure-JAX producer the 3-D ``--scan-block`` path uses
(:func:`compute_omip2_surface_forcing_jax` -> NCAR :func:`air_sea_fluxes`)
plus the freshwater mirror of :func:`compute_omip2_freshwater_forcing`
(runoff EXCLUDED).  The CORE-II record index is a TRACED argument
(SegmentForcing doctrine: per-step values are args, never closure captures)
sampled at the END-of-step time exactly like the 3-D host loop (1-based
``_idx_t(step, dt)``; the 2-line floor convention is mirrored in
:func:`record_index` with a direct test, ``scripts/run`` not being
importable from here).

:class:`Core2PointFluxes` retains the closure-based REFERENCE
implementation of the same wiring (``OceanSCMForcing`` ``f(t)->scalar``
closures fed by the host :func:`compute_omip2_surface_forcing`); the unit
tests lock the jitted path to it step-for-step.  The reference path is NOT
used in the hot loop: driving ``scm.step()`` eagerly re-traces AND
re-compiles the implicit-solver ``lax.fori_loop`` closures EVERY step
(~seconds/step — job 8889618 produced zero output in 2.7 h).

Sign conventions (walked per CLAUDE.md sign-check):
* ``air_sea_fluxes`` returns wind stress in the ATMOSPHERIC convention
  (``-rho Cd |U| u``, opposing the wind); the 3-D dynamics core applies the
  ``-tau`` ocean reaction.  The SCM's prescribed path is OCEAN_DIRECT
  (``+tau`` accelerates the surface layer), so this driver applies
  ``tau_scm = -tau_atm`` (:func:`atm_to_ocean_stress`).
* ``q_net`` [W/m^2] positive INTO the ocean everywhere.
* ``e_minus_p`` [m/s] positive = net evaporation = salinifying virtual salt
  flux; built as ``(evap - precip)/rho_water`` from the freshwater producer
  (evap positive UP, precip positive INTO ocean).  RUNOFF IS EXCLUDED
  (the 3-D run injects Dai–Trenberth runoff with coastal spreading, which has
  no single-column analogue) — significant at ``siberian_shelf`` (Lena).
* Salt application mirrors the 3-D OMIP step EXACTLY (codex 2026-07 #2):
  the SHARED :func:`legoesm.ocean.freshwater.virtual_salt_flux_from_net`
  leaf, ``dS/dt = -S_ref * F_fw / (rho_0 * dz_0)`` with
  ``F_fw = -(E-P) * rho_water`` [kg/m^2/s, + INTO ocean] and the 3-D
  config-default ``S_ref``/``rho_0`` (``LatLonCGridOceanConfig``: 35.0 PSU /
  1025 kg/m^3) — NOT ``prescribed_surface_forcing``'s local-S form
  ``S_top*(E-P)/dz_0``.  Net evaporation => F_fw < 0 => dS/dt > 0
  (salinifies), same sign either way; the S_ref form is what the 3-D
  ``virtual_salt_flux`` closure applies.  The closure REFERENCE path
  pre-compensates the ``e_minus_p`` channel by ``S_ref*rho_w/(rho_0*S_top)``
  so the prescribed formula inside ``scm.step()`` lands on the identical
  closure (see :meth:`Core2PointFluxes.update`).
* Forcing + Coriolis are evaluated at the SELECTED nearest-wet NEMO cell
  (``ic["cell_lat"]``, ``ic["cell_lon"]``), never at the requested point
  (codex 2026-07 #1) — the referee column lives at the cell; the requested
  coordinates are metadata only.
* Shortwave: ``compute_omip2_surface_forcing`` folds SW INTO ``q_net``.  In
  ``--sw-mode penetrate`` (default, mirrors the 3-D core's two-band split
  EXACTLY) the SCM applies ``q_net - 0.94*sw_net`` to the top layer (the
  0.06 skin fraction stays at the surface, as in ``ocean_pe_latlon_cgrid``)
  and lets the physics pipeline's Jerlov two-band scheme deposit
  ``0.94*sw_net`` over depth; in ``--sw-mode top`` the full ``q_net`` lands
  in the surface layer (penetration disabled).

NEMO file conventions found by inspection (2026-07):
* IC:  ``woce_temp_monthly_init_4p2.nc`` var ``contemp`` [Celsius degrees]
  (Conservative Temperature), ``woce_salt_monthly_init_4p2.nc`` var
  ``presalt`` [g/kg]; dims (time=12, z=75, y=331, x=360); 2-D nav_lat/nav_lon,
  ``nav_lev`` centre depths; fill ``-1e34`` below the seabed (=> wet mask).
* Referee: ``ORCA1_1m_..._grid_T.nc`` 3-D vars ``to`` [degC] / ``so`` [1e-3]
  (40 monthly time records, CF time ``seconds since 1900-01-01``, noleap —
  of which the LAST is an UNWRITTEN all-``_FillValue`` payload from the
  killed run: 40 time entries, 39 written to/so/e3t records; data-less
  records are auto-excluded from month selection).  Layer thicknesses come
  from the per-point ``e3t(time, z, y, x)`` on the SAME selected records
  (partial-cell/shelf-aware — the global ``deptht_bounds`` misstates the
  wet-column depth/heat capacity on shelf columns like ``siberian_shelf``);
  ``deptht_bounds`` is the fallback when ``e3t`` is absent.
  TEOS-10 caveat: NEMO carries CT / (near-)SA while the legoESM column labels
  T/S as potential temperature / PSU; the ~0.05 degC / ~0.1 g kg^-1 scale
  offsets are second order for a 90-day physics comparison but are noted in
  the output metadata.

Usage (compute node / sbatch only — NOT the login node):
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -u \
        scripts/validate/ocean_fidelity/run_scm_column_twins.py \
        --days 90 --dt 3600 --vmix kpp        # add ,constant for the A/B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
_VALIDATE_DIR = _HERE.parents[1]          # scripts/validate (compare_omip_nemo)

# --- diagnostic points (name -> (lat_deg N, lon_deg E)) --------------------
DEFAULT_POINTS: dict[str, tuple[float, float]] = {
    "siberian_shelf": (76.0, 125.0),   # Laptev; +2.6 psu salty-bias hotspot
    "barents":        (73.0, 40.0),    # -1.3 psu over-freshened box
    "nh_midlat":      (35.0, -40.0),   # NH-midlat winter-ML / mode water
    "subtrop_pac":    (20.0, -150.0),  # quiet reference
}

ADVECTION_CAVEAT = (
    "CAVEAT: referee = NEMO 3-D monthly output — columns INCLUDE advection "
    "(+ sea ice, SSS restoring, runoff) that the SCM does not; differences "
    "are NOT physics-only. NEMO C1D twin comes later."
)

# NYF perpetual-year calendar (noleap), matching run_omip_core2._YEAR_S and
# the CORE-II zarr layout (time_s = (k+0.5)*record_spacing).
_YEAR_S = 365.0 * 86400.0
_NOLEAP_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

# Fraction of the net surface shortwave that PENETRATES the column in the
# 3-D dynamics core's two-band branch; the remaining 0.06 "skin" fraction is
# absorbed with the non-solar flux in the surface layer.  MIRROR of the
# inline 0.94 in ocean_pe_latlon_cgrid.py (``sw_absorbed = sw_T * 0.94``,
# "legacy two-band Jerlov; 0.94 skin split unchanged") — must track that
# value so the SCM heating PROFILE matches the 3-D core, not only the
# column-integrated heat (codex 2026-07 finding #1).
_SW_PENETRATION_FRACTION = 0.94

_NEMO_BASE = Path(
    "/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1"
)
DEFAULT_IC_TEMP = (
    _NEMO_BASE / "INPUTS/orca1_inputs/data_repository/initial_conditions/"
    "woce_temp_monthly_init_4p2.nc"
)
DEFAULT_IC_SALT = (
    _NEMO_BASE / "INPUTS/orca1_inputs/data_repository/initial_conditions/"
    "woce_salt_monthly_init_4p2.nc"
)
DEFAULT_NEMO_GRIDT = (
    _NEMO_BASE / "EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc"
)

# Variable-name candidates (first match wins; actual names verified by
# inspection are listed first).
_TEMP_VAR_CANDIDATES = ("contemp", "votemper", "thetao", "to")
_SALT_VAR_CANDIDATES = ("presalt", "vosaline", "so")
_NEMO_T3D_CANDIDATES = ("to", "thetao", "votemper")
_NEMO_S3D_CANDIDATES = ("so", "vosaline")

_VMIX_CHOICES = ("kpp", "constant", "richardson")  # SCM rejects tke/catke


class PointTooFarError(RuntimeError):
    """Nearest wet ORCA1 column is farther than the allowed distance."""


# ===========================================================================
# Pure helpers (numpy only — unit-tested directly)
# ===========================================================================


def parse_point_spec(spec: str) -> tuple[str, float, float]:
    """Parse ``name=lat,lon`` (degrees; lon east, negative = west)."""
    if "=" not in spec:
        raise ValueError(f"point spec must be name=lat,lon, got {spec!r}")
    name, _, coords = spec.partition("=")
    parts = coords.split(",")
    if len(parts) != 2 or not name.strip():
        raise ValueError(f"point spec must be name=lat,lon, got {spec!r}")
    lat, lon = float(parts[0]), float(parts[1])
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"latitude out of range in point spec {spec!r}")
    return name.strip(), lat, lon


def great_circle_deg(lat_deg, lon_deg, lat0_deg: float, lon0_deg: float):
    """Great-circle separation [deg] between grid points and one target.

    Unit-sphere chord -> central angle (same inline-trivial pattern as
    ``compare_omip_nemo._xyz``; handles the lon wrap and both [-180,180]
    and [0,360) conventions with no branch).
    """
    la = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lo = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    la0 = np.deg2rad(float(lat0_deg))
    lo0 = np.deg2rad(float(lon0_deg))
    cos_ang = (
        np.sin(la) * np.sin(la0) + np.cos(la) * np.cos(la0) * np.cos(lo - lo0)
    )
    return np.rad2deg(np.arccos(np.clip(cos_ang, -1.0, 1.0)))


def nearest_wet_column(
    nav_lat, nav_lon, wet_mask, lat0: float, lon0: float,
    max_dist_deg: float = 2.0,
) -> tuple[int, int, float]:
    """Index ``(j, i)`` of the nearest WET cell to ``(lat0, lon0)``.

    Raises :class:`PointTooFarError` when the nearest wet cell is farther
    than ``max_dist_deg`` (great-circle degrees) — the caller warns + skips.
    """
    wet = np.asarray(wet_mask, dtype=bool)
    if not wet.any():
        raise PointTooFarError("no wet cells in the domain")
    dist = great_circle_deg(nav_lat, nav_lon, lat0, lon0)
    dist = np.where(wet, dist, np.inf)
    j, i = np.unravel_index(int(np.argmin(dist)), dist.shape)
    d = float(dist[j, i])
    if d > max_dist_deg:
        raise PointTooFarError(
            f"nearest wet ORCA1 column to ({lat0:.2f}N, {lon0:.2f}E) is "
            f"{d:.2f} deg away (> {max_dist_deg:.2f} deg) — refusing to "
            "compare a column that far from the requested point"
        )
    return int(j), int(i), d


def dz_from_center_depths(z_centers) -> np.ndarray:
    """Layer thicknesses from centre depths (midpoint-interface rule).

    Fallback when ``deptht_bounds`` is unavailable: interfaces at midpoints
    between centres, top interface at 0, bottom interface reflected so the
    last centre sits mid-cell.  Exact for uniform grids; a small (<~2%)
    thickness error on stretched grids vs NEMO's analytic e3t — the referee
    file's ``deptht_bounds`` is preferred when present.
    """
    zc = np.asarray(z_centers, dtype=np.float64)
    if zc.ndim != 1 or zc.size < 2:
        raise ValueError(f"z_centers must be 1-D with >= 2 levels, got {zc.shape}")
    if not np.all(np.diff(zc) > 0.0):
        raise ValueError("z_centers must be positive-down monotone increasing")
    interfaces = np.empty(zc.size + 1, dtype=np.float64)
    interfaces[0] = 0.0
    interfaces[1:-1] = 0.5 * (zc[:-1] + zc[1:])
    interfaces[-1] = 2.0 * zc[-1] - interfaces[-2]
    dz = np.diff(interfaces)
    if not np.all(dz > 0.0):
        raise ValueError("derived thicknesses must be positive")
    return dz


def record_index(t_seconds: float, n_rec: int, year_seconds: float = _YEAR_S) -> int:
    """CORE-II record index at model time ``t`` (perpetual year).

    Mirrors ``scripts/run/run_omip_core2._idx_t`` EXACTLY (that script bucket
    is not importable from here): records are cell-CENTRED at
    ``(k + 0.5) * spacing`` (see ``build_core2_nyf_zarr``), so the record
    whose centre is nearest to ``t`` is ``floor(t / spacing)`` — NOT
    ``round``, which would apply each record with a +spacing/2 phase lead.
    """
    if n_rec <= 0:
        raise ValueError(f"n_rec must be positive, got {n_rec}")
    spacing = year_seconds / float(n_rec)
    t = float(t_seconds) % year_seconds
    return int(t // spacing) % n_rec


def month_start_seconds(month: int) -> float:
    """Seconds from Jan-1 00:00 to the start of ``month`` (1-12, noleap)."""
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1..12, got {month}")
    return float(sum(_NOLEAP_MONTH_DAYS[: month - 1])) * 86400.0


def atm_to_ocean_stress(tau_atm: float) -> float:
    """ATMOSPHERIC-convention stress -> on-ocean (OCEAN_DIRECT) stress.

    ``air_sea_fluxes`` returns ``tau = -rho Cd |U| u`` (drag opposing the
    wind); the 3-D dynamics core applies the ``-tau`` ocean reaction
    (``ocean_pe_latlon_cgrid``).  The SCM's prescribed surface forcing is
    OCEAN_DIRECT (``+tau`` accelerates the surface layer in the stress
    direction — see ``prescribed.py __physics_contract__``), so the ocean
    twin must receive ``-tau_atm``.  Westerly wind (u10 > 0) => tau_atm < 0
    => on-ocean stress > 0 => eastward surface acceleration.  Locked by a
    unit test (tests/ocean/unit/test_scm_column_twins.py).
    """
    return -float(tau_atm)


def month_sequence(start_month: int, n_months: int) -> list[int]:
    """Calendar months (1-12) covered by ``n_months`` 30-day windows."""
    return [(start_month - 1 + k) % 12 + 1 for k in range(n_months)]


# ===========================================================================
# NEMO file extraction (netCDF via xarray)
# ===========================================================================


def _find_data_var(ds, candidates) -> str:
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(
        f"none of {candidates} found; data vars = "
        f"{[v for v in ds.data_vars]}"
    )


def extract_ic_profile(
    temp_path: Path,
    salt_path: Path,
    month_idx: int,
    lat0: float,
    lon0: float,
    max_dist_deg: float = 2.0,
) -> dict:
    """Extract the nearest-wet NEMO IC column (T, S) at one point.

    Returns a dict with the wet-truncated profiles (surface -> bottom),
    the (j, i) grid indices (shared by IC and referee — same ORCA1 T grid),
    the great-circle distance to the requested point, and the full centre
    depths ``nav_lev``.  Wetness = finite (non-fill) values; the profile is
    truncated at the first non-finite level so a mid-column fill can never
    be silently interpolated over.
    """
    import xarray as xr

    with xr.open_dataset(temp_path, decode_times=False) as dt:
        tvar = _find_data_var(dt, _TEMP_VAR_CANDIDATES)
        T4 = np.asarray(dt[tvar].isel(time=month_idx).values, dtype=np.float64)
        nav_lat = np.asarray(dt["nav_lat"].values, dtype=np.float64)
        nav_lon = np.asarray(dt["nav_lon"].values, dtype=np.float64)
        nav_lev = np.asarray(dt["nav_lev"].values, dtype=np.float64)
    with xr.open_dataset(salt_path, decode_times=False) as dsalt:
        svar = _find_data_var(dsalt, _SALT_VAR_CANDIDATES)
        S4 = np.asarray(
            dsalt[svar].isel(time=month_idx).values, dtype=np.float64
        )

    if T4.shape != S4.shape:
        raise ValueError(
            f"IC temp {T4.shape} and salt {S4.shape} shapes differ"
        )
    wet2d = np.isfinite(T4[0]) & np.isfinite(S4[0])
    j, i, dist = nearest_wet_column(
        nav_lat, nav_lon, wet2d, lat0, lon0, max_dist_deg
    )

    T_col = T4[:, j, i]
    S_col = S4[:, j, i]
    finite = np.isfinite(T_col) & np.isfinite(S_col)
    # contiguous wet levels from the surface down
    n_wet = int(np.argmin(finite)) if not finite.all() else int(finite.size)
    if n_wet < 2:
        raise PointTooFarError(
            f"column at ({lat0:.2f}N,{lon0:.2f}E) -> (j={j},i={i}) has only "
            f"{n_wet} wet level(s); the SCM needs >= 2"
        )
    return {
        "T": T_col[:n_wet].copy(),
        "S": S_col[:n_wet].copy(),
        "n_wet": n_wet,
        "j": j,
        "i": i,
        "dist_deg": dist,
        "nav_lev": nav_lev,
        "cell_lat": float(nav_lat[j, i]),
        "cell_lon": float(nav_lon[j, i]),
        "temp_var": tvar,
        "salt_var": svar,
    }


def _import_compare_omip_nemo():
    """Import scripts/validate/compare_omip_nemo.py (month-decode reuse).

    ``scripts/validate`` is on PYTHONPATH in the sbatch environment; insert
    it explicitly so the driver also works standalone.  Private-symbol reuse
    across *scripts* is the sanctioned white-box pattern (the
    ``test_no_private_cross_imports`` ratchet scopes packages only).
    """
    if str(_VALIDATE_DIR) not in sys.path:
        sys.path.insert(0, str(_VALIDATE_DIR))
    import compare_omip_nemo

    return compare_omip_nemo


def extract_nemo_reference(
    gridt_path: Path,
    j: int,
    i: int,
    months: list[int],
    clim: bool = False,
) -> dict:
    """Referee columns: NEMO 3-D monthly ``to``/``so`` at ``(j, i)``.

    Month selection reuses ``compare_omip_nemo._nemo_record_months`` (CF
    ``units`` + ``calendar`` decode — records are NOT assumed Jan-first).
    ``clim=False`` (default) takes the FIRST record of each requested
    calendar month — the year-1 trajectory that actually started from the
    IC; ``clim=True`` averages every occurrence (climatological month).

    Robustness (codex 2026-07 findings #3/#4):

    * ``nt`` is the DATA variable's time length (``ds[tvar].sizes[tdim]``),
      not the dataset-level dim, and ``so`` (and ``e3t`` when used) must
      match — a merged/aggregated time axis longer than a payload can never
      index past the data.  The decoded record months are sliced to ``nt``.
    * A record can EXIST on the time axis with an UNWRITTEN payload (killed
      run: the real RUN_REF file carries 40 ``time_counter`` entries but the
      40th ``to``/``so``/``e3t`` record is all-``_FillValue``).  Candidate
      records whose (T, S) column is entirely non-finite at ``(j, i)`` are
      dropped LOUDLY before the first/clim selection (``dropped_records``),
      so ``--nemo-clim`` never averages fill into the referee; a month with
      zero valid records raises.
    * Layer thickness ``dz``: per-point ``e3t[:, :, j, i]`` on the SAME
      selected records (per-month mean, then across months — consistent
      with ``clim``), preferred over the GLOBAL ``deptht_bounds``, which
      misstates the wet-column depth/heat capacity on shelf/partial-bottom
      columns.  ``dz_source`` records which was used.  Below the local
      seabed ``e3t`` is fill => NaN; the caller truncates to the IC's wet
      depth and validates finiteness.

    The advection caveat applies to everything returned here.
    """
    import xarray as xr

    cmp_mod = _import_compare_omip_nemo()

    out: dict = {"months": list(months), "clim": bool(clim)}
    with xr.open_dataset(gridt_path, decode_times=False) as ds:
        tvar = _find_data_var(ds, _NEMO_T3D_CANDIDATES)
        svar = _find_data_var(ds, _NEMO_S3D_CANDIDATES)
        tdim = "time_counter"
        if tdim not in ds[tvar].dims:
            raise ValueError(f"{gridt_path} has no {tdim} dim on {tvar}")
        # Time length of the DATA variable, not ds.sizes[tdim]: sibling
        # variables (or a merged view) can carry a longer time axis than the
        # T/S payloads (codex finding #4).
        nt = int(ds[tvar].sizes[tdim])
        if int(ds[svar].sizes.get(tdim, -1)) != nt:
            raise ValueError(
                f"referee {gridt_path}: {svar} time length "
                f"{ds[svar].sizes.get(tdim)} != {tvar} length {nt} — "
                "refusing to pair mismatched T/S records"
            )
        e3t_var = "e3t" if "e3t" in ds.variables else None
        if e3t_var is not None and int(ds[e3t_var].sizes.get(tdim, -1)) != nt:
            raise ValueError(
                f"referee {gridt_path}: e3t time length "
                f"{ds[e3t_var].sizes.get(tdim)} != {tvar} length {nt} — "
                "refusing to pair mismatched thickness records"
            )
        rec_months = cmp_mod._nemo_record_months(ds, tdim, nt)
        if rec_months is None:
            raise ValueError(
                f"cannot decode CF time of {gridt_path} (cftime missing or "
                "bad metadata) — refusing to guess which records are which "
                "calendar month"
            )
        rec_months = list(rec_months)[:nt]
        T_profiles, S_profiles, rec_used, e3t_profiles = [], [], [], []
        dropped: list[int] = []
        for m in months:
            idxs = [k for k in range(nt) if rec_months[k] == m]
            if not idxs:
                raise ValueError(
                    f"referee {gridt_path} has no record with calendar "
                    f"month {m} (n_time={nt})"
                )
            # lazy column index BEFORE .values so only (n_cand, 75) loads
            T_cand = np.asarray(
                ds[tvar].isel({tdim: idxs})[:, :, j, i].values,
                dtype=np.float64,
            )
            S_cand = np.asarray(
                ds[svar].isel({tdim: idxs})[:, :, j, i].values,
                dtype=np.float64,
            )
            # Data-less records: time entry written, payload never flushed
            # (all-fill => all-NaN after CF masking; a WET column always has
            # a finite surface value in a written record).
            has_data = (
                np.isfinite(T_cand).any(axis=1)
                & np.isfinite(S_cand).any(axis=1)
            )
            bad = [idxs[p] for p in range(len(idxs)) if not has_data[p]]
            if bad:
                dropped.extend(bad)
                print(f"[twins] WARN: referee records {bad} (month {m}) "
                      "carry no data (unwritten trailing record) — excluded",
                      flush=True)
            valid_pos = [p for p in range(len(idxs)) if has_data[p]]
            if not valid_pos:
                raise ValueError(
                    f"referee {gridt_path} month {m}: every candidate "
                    f"record {idxs} is data-less at (j={j}, i={i})"
                )
            pos_sel = valid_pos if clim else valid_pos[:1]
            sel = [idxs[p] for p in pos_sel]
            T_profiles.append(T_cand[pos_sel].mean(axis=0))
            S_profiles.append(S_cand[pos_sel].mean(axis=0))
            rec_used.append(sel)
            if e3t_var is not None:
                # Per-point partial-cell thickness on the SAME selected
                # records, averaged like the T/S profiles (codex finding #3).
                e3t_profiles.append(
                    np.asarray(
                        ds[e3t_var].isel({tdim: sel})[:, :, j, i].values,
                        dtype=np.float64,
                    ).mean(axis=0)
                )
        out["T"] = np.stack(T_profiles)          # (n_months, 75)
        out["S"] = np.stack(S_profiles)
        out["records_used"] = rec_used
        out["dropped_records"] = dropped
        out["deptht"] = np.asarray(ds["deptht"].values, dtype=np.float64)
        if e3t_var is not None:
            # across-month mean of the per-month means (mirrors the T/S
            # month weighting; e3t time variation is mm-scale SSH — the
            # signal is the static partial-cell geometry at (j, i))
            out["dz"] = np.stack(e3t_profiles).mean(axis=0)
            out["dz_source"] = "e3t"
        elif "deptht_bounds" in ds.variables:
            b = np.asarray(ds["deptht_bounds"].values, dtype=np.float64)
            out["dz"] = b[:, 1] - b[:, 0]
            out["dz_source"] = "deptht_bounds"
        else:
            out["dz"] = None
            out["dz_source"] = None
        out["cell_lat"] = float(np.asarray(ds["nav_lat"].values)[j, i])
        out["cell_lon"] = float(np.asarray(ds["nav_lon"].values)[j, i])
        out["t_var"] = tvar
        out["s_var"] = svar
    return out


# ===========================================================================
# CORE-II point forcing (reuses the 3-D producers verbatim)
# ===========================================================================


def _virtual_salt_ref_constants() -> tuple[float, float]:
    """``(S_ref [PSU], rho_0 [kg/m^3])`` of the 3-D virtual-salt closure.

    Read from the 3-D lat-lon C-grid config DEFAULTS
    (``LatLonCGridOceanConfig``: ``S_ref=35.0``, ``rho_0=constants.rho_ocean``)
    so the SCM twin tracks the EXACT convention of the OMIP step's
    ``virtual_salt_flux(freshwater, S_ref=config.S_ref, dz_0,
    rho_0=config.rho_0)`` (``ocean_model_latlon_cgrid``) instead of
    hardcoding the values here.
    """
    from legoesm.ocean.state import LatLonCGridOceanConfig

    d = LatLonCGridOceanConfig._field_defaults
    return float(d["S_ref"]), float(d["rho_0"])


class _PointGrid(NamedTuple):
    """1x1 shim exposing the ``lat_T``/``lon_T`` [rad] attributes that the
    ``grid_type="tripole"`` branch of ``sample_omip2_forcing`` reads.
    Mimicry-only glue -> lives in this harness, never in the model
    (oracle-recipe doctrine)."""

    lat_T: np.ndarray  # (1, 1) [rad]
    lon_T: np.ndarray  # (1, 1) [rad]


class Core2PointFluxes:
    """Per-step CORE-II bulk fluxes at one point, from the SCM's own state.

    ``update(state, t)`` calls the SAME producers as the 3-D OMIP run —
    :func:`compute_omip2_surface_forcing` (NCAR bulk stress + NEMO-assembled
    q_ns + SW) and :func:`compute_omip2_freshwater_forcing` (P, interactive
    E; runoff EXCLUDED here) — with the live column state, and caches Python
    floats that the ``OceanSCMForcing`` closures return.  Call it BEFORE each
    ``scm.step()``: the closures are then evaluated at exactly the state/time
    the cache was built from (the SCM tendency is sampled once per step at
    the start-of-step time; the 3-D run uses the same current-state fluxes).

    ROLE (job 8889618): this closure path is the validated REFERENCE — the
    unit tests lock :func:`build_jitted_step` (the production hot loop) to
    it step-for-step.  Do NOT drive long integrations through it: each
    eager ``scm.step()`` re-compiles the implicit-solver loops.
    """

    def __init__(self, forcing, lat_deg: float, lon_deg: float,
                 sw_mode: str = "penetrate"):
        if sw_mode not in ("penetrate", "top"):
            raise ValueError(
                f"sw_mode must be 'penetrate' or 'top', got {sw_mode!r}"
            )
        self.forcing = forcing
        self.sw_mode = sw_mode
        self.grid = _PointGrid(
            lat_T=np.deg2rad(np.asarray([[float(lat_deg)]])),
            lon_T=np.deg2rad(np.asarray([[float(lon_deg)]])),
        )
        time_s = np.asarray(forcing.time_s, dtype=np.float64)
        self.n_rec = int(time_s.size)
        if self.n_rec >= 2:
            spacings = np.diff(time_s)
            if not np.allclose(spacings, spacings[0], rtol=1e-6):
                raise ValueError(
                    "CORE-II forcing records are not uniformly spaced; the "
                    "floor record-index convention requires uniform spacing"
                )
        # 3-D virtual-salt convention (S_ref, rho_0) — see update().
        self.S_ref, self.rho_0 = _virtual_salt_ref_constants()
        # cache holds on-ocean-convention scalars; zeros until first update()
        self._cache = {
            "tau_x": 0.0, "tau_y": 0.0, "q_net_total": 0.0,
            "sw_net": 0.0, "e_minus_p": 0.0, "e_minus_p_prescribed": 0.0,
        }
        self.last_idx: int | None = None

    # -- closure targets ----------------------------------------------------
    def update(self, state, t_seconds: float) -> None:
        from legoesm import constants
        from legoesm.ocean.coupler import (
            compute_omip2_freshwater_forcing,
            compute_omip2_surface_forcing,
        )

        idx = record_index(t_seconds, self.n_rec)
        self.last_idx = idx
        sf = compute_omip2_surface_forcing(
            state, forcing=self.forcing, idx_t=idx,
            grid=self.grid, grid_type="tripole",
        )
        fw = compute_omip2_freshwater_forcing(
            state, forcing=self.forcing, idx_t=idx,
            grid=self.grid, grid_type="tripole",
            runoff_R=None,          # runoff EXCLUDED (documented caveat)
        )

        def _scalar(x) -> float:
            return float(np.asarray(x).reshape(-1)[0])

        # SIGN: bulk tau is ATMOSPHERIC convention; SCM is OCEAN_DIRECT.
        self._cache["tau_x"] = atm_to_ocean_stress(_scalar(sf.tau_x))
        self._cache["tau_y"] = atm_to_ocean_stress(_scalar(sf.tau_y))
        self._cache["q_net_total"] = _scalar(sf.q_net)   # + into ocean, incl SW
        self._cache["sw_net"] = _scalar(sf.sw_down)      # post-albedo SW
        # E - P [m/s], positive = net evaporation (salinifies); evap is
        # positive UP and precip positive INTO the ocean [kg/m^2/s].
        e_minus_p = (
            _scalar(fw.evap) - _scalar(fw.precip)
        ) / float(constants.rho_water)
        self._cache["e_minus_p"] = e_minus_p
        # 3-D-parity virtual salt (codex 2026-07 #2): the OMIP step applies
        #   dS/dt = -S_ref * F_fw / (rho_0 * dz_0)      [virtual_salt_flux]
        # with F_fw = -(E-P)*rho_water [kg/m^2/s, + INTO ocean], while the
        # SCM's only salt channel is prescribed_surface_forcing's LOCAL-S
        # form dS/dt = S_top * E_minus_P / dz_0.  PRE-COMPENSATE the channel:
        #   E' = (E-P) * S_ref * rho_water / (rho_0 * S_top)
        # => S_top * E' / dz_0 == -S_ref * F_fw / (rho_0 * dz_0) exactly.
        # SIGN: net evaporation (E-P > 0, S_top > 0) keeps E' > 0 =>
        # salinifies, identical to the 3-D closure.  NOTE: the KPP surface
        # struct built from this channel inside scm.step() then carries the
        # ~S_ref*rho_w/(rho_0*S_top) factor in its freshwater — this closure
        # path is the parity REFERENCE (constant-mixing tests lock
        # build_jitted_step to it), never the production kpp hot loop, which
        # feeds KPP the RAW F_fw.
        S_top = float(np.asarray(state.S.data)[..., 0].reshape(-1)[0])
        if not np.isfinite(S_top) or S_top <= 0.0:
            raise ValueError(
                "top-layer salinity must be positive/finite for the "
                f"virtual-salt compensation, got {S_top!r}"
            )
        self._cache["e_minus_p_prescribed"] = (
            e_minus_p * self.S_ref * float(constants.rho_water)
            / (self.rho_0 * S_top)
        )

    # closure reads (ignore t: cache is refreshed each step at start time)
    def _tau_x(self, t: float) -> float:
        return self._cache["tau_x"]

    def _tau_y(self, t: float) -> float:
        return self._cache["tau_y"]

    def _q_net(self, t: float) -> float:
        if self.sw_mode == "penetrate":
            # Mirror the 3-D core's two-band split EXACTLY
            # (ocean_pe_latlon_cgrid): the surface layer receives
            # q_net - 0.94*sw (i.e. non-solar + the 0.06 skin fraction);
            # the penetrating 0.94*sw goes through the pipeline's Jerlov
            # scheme (which deposits 100% of what it receives, so the
            # column-total heat stays q_net_total).
            return (self._cache["q_net_total"]
                    - _SW_PENETRATION_FRACTION * self._cache["sw_net"])
        return self._cache["q_net_total"]

    def _sw_down(self, t: float) -> float:
        return _SW_PENETRATION_FRACTION * self._cache["sw_net"]

    def _e_minus_p(self, t: float) -> float:
        # PRE-COMPENSATED value: routed through the prescribed local-S
        # formula this yields the 3-D S_ref virtual-salt closure (see
        # update()); the RAW E-P stays in _cache["e_minus_p"].
        return self._cache["e_minus_p_prescribed"]

    def make_scm_forcing(self, tier: str, lat_deg: float):
        """Build the :class:`OceanSCMForcing` for tier ``T0`` or ``T1``.

        T0: f_c = 0 and NO wind-stress channels (u, v stay exactly 0 —
        buoyancy-only physics).  T1: full Ekman column, f_c = 2 Omega sin(lat)
        from ``legoesm.constants.Omega``.
        """
        from legoesm import constants
        from legoesm.ocean.scm_forcing import OceanSCMForcing

        if tier not in ("T0", "T1"):
            raise ValueError(f"tier must be 'T0' or 'T1', got {tier!r}")
        sw = self._sw_down if self.sw_mode == "penetrate" else None
        if tier == "T0":
            return OceanSCMForcing(
                f_c=0.0, tau_x=None, tau_y=None,
                q_net=self._q_net, e_minus_p=self._e_minus_p, sw_down=sw,
            )
        f_c = 2.0 * float(constants.Omega) * float(np.sin(np.deg2rad(lat_deg)))
        return OceanSCMForcing(
            f_c=f_c, tau_x=self._tau_x, tau_y=self._tau_y,
            q_net=self._q_net, e_minus_p=self._e_minus_p, sw_down=sw,
        )


# ===========================================================================
# Jitted hot path (job 8889618 timeout fix)
# ===========================================================================

_FORCING_CHANNELS = ("u10", "v10", "T_air", "q_air", "sw_down", "lw_down",
                     "precip", "snow", "slp")


def extract_point_forcing_series(forcing, lat_deg: float, lon_deg: float) -> dict:
    """Column time series of the 9 CORE-II atmospheric-STATE channels.

    Pure array extraction (numpy only, ONCE per point — no JAX): the
    nearest forcing cell is selected with
    :func:`core2_forcing_nn_indices` (the EXACT nearest-neighbour
    convention of the 3-D producers / ``_nn_interp_to_points``) and each
    channel is sliced to a ``(n_rec, 1, 1)`` series, i.e. a single-point
    ``forcing_stack`` consumable by :func:`compute_omip2_surface_forcing_jax`.
    Same layout as ``build_core2_forcing_device_stack`` — which lifts the
    FULL forcing grid to device (~2 GB) that a column driver does not need —
    including its optional-channel fallbacks: ``snow=0``,
    ``slp=constants.p_atm_std`` (identical to the host producers' internal
    defaults, so jitted and closure paths see the same values).
    """
    from legoesm import constants
    from legoesm.ocean.coupler.omip2_applicator import core2_forcing_nn_indices

    nn_i, nn_j = core2_forcing_nn_indices(
        forcing, np.asarray([float(lat_deg)]), np.asarray([float(lon_deg)]),
    )
    i0, j0 = int(nn_i[0]), int(nn_j[0])
    series = {
        name: np.asarray(getattr(forcing, name), dtype=np.float64)[:, i0, j0]
        .reshape(-1, 1, 1)
        for name in _FORCING_CHANNELS
        if getattr(forcing, name, None) is not None
    }
    ref = series["precip"]
    if "snow" not in series:
        series["snow"] = np.zeros_like(ref)
    if "slp" not in series:
        series["slp"] = np.full_like(ref, float(constants.p_atm_std))
    return series


def build_jitted_step(scm, series: dict, *, tier: str, sw_mode: str):
    """ONE jitted SCM step per run (closure built ONCE before the loop).

    WHY (job 8889618, cancelled at 4 h with zero output): ``scm.step()`` is
    an eager host loop; its backward-Euler solves build fresh
    ``lax.fori_loop`` body closures on every call (``thomas_solve``:
    ``forward_body``/``backward_body`` x 4 fields), so EVERY step re-traced
    and re-COMPILED ~8 XLA loops (~seconds/step; 2160 steps x 16 runs never
    finished).  Wrapping the whole step in a single ``jax.jit`` traces that
    control flow ONCE; the CORE-II record index is a traced argument
    (SegmentForcing doctrine — per-step values are ARGS, never closure
    captures), so each run compiles exactly once.

    Flux reuse (no re-derivation):

    * tau / q_net / sw come from the SHARED pure-JAX 3-D producer
      :func:`compute_omip2_surface_forcing_jax` (NCAR
      :func:`air_sea_fluxes` + NEMO ``blk_oce_2`` non-solar assembly) with
      the EVOLVING SST from the traced state and the ``series`` mini-stack.
    * the freshwater channel mirrors :func:`compute_omip2_freshwater_forcing`
      (runoff EXCLUDED, documented caveat): evaporation from one more call
      to the same :func:`air_sea_fluxes` leaf on the same sampled record
      (XLA CSEs the repeated pure evaluation).
    * the explicit top-layer wind + heat tendencies reuse
      :func:`prescribed_surface_forcing` VERBATIM (tau / Q_net ride the
      config as traced scalars; ``E_minus_P=0.0`` is a STATIC zero, so its
      local-S salt branch is OFF).  Salt is applied through the SHARED 3-D
      leaf :func:`legoesm.ocean.freshwater.virtual_salt_flux_from_net`
      instead: ``dS/dt = -S_ref * F_fw / (rho_0 * dz_0)`` with
      ``F_fw = sfc.freshwater = -(E-P)*rho_water`` [kg/m^2/s, + INTO ocean]
      and the 3-D config-default ``S_ref``/``rho_0``
      (:func:`_virtual_salt_ref_constants`) — exactly the OMIP step's
      ``virtual_salt_flux(freshwater, S_ref=config.S_ref, dz_0,
      rho_0=config.rho_0)``, NOT prescribed's local-S form (codex 2026-07
      finding).  Net evaporation => F_fw < 0 => dS/dt > 0 (salinifies).
      Parity with the (equally pre-compensated) closure reference is locked
      by ``test_jitted_step_matches_closure_reference``; the closure form
      itself by ``test_jitted_step_salt_is_3d_virtual_salt_closure``.
    * the state advance reuses the SCM's own operators in the exact
      ``scm.step()`` order: ``_apply_ocean_tendencies`` (forward Euler) ->
      ``scm._apply_implicit_vertical_mixing`` (backward-Euler tridiagonal)
      -> ``scm._apply_coriolis_rotation`` (Crank-Nicolson).

    Returns ``step_fn(state, idx_t) -> (new_state, K_v)``; ``idx_t`` is the
    traced int32 CORE-II record index which the caller samples at the
    END-of-step time (3-D host-loop convention, codex 2026-07 finding #2).
    ``K_v`` is the diffusivity profile that ACTED during the step (may be
    ``None`` for schemes that only use the background floors).
    """
    import jax
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.ocean.bulk_flux_omip import air_sea_fluxes
    from legoesm.ocean.coupler.omip2_applicator import (
        compute_omip2_surface_forcing_jax,
    )
    from legoesm.ocean.freshwater import virtual_salt_flux_from_net
    from legoesm.ocean.physics.surface_forcing.config import (
        PrescribedForcingConfig,
    )
    from legoesm.ocean.physics.surface_forcing.prescribed import (
        prescribed_surface_forcing,
    )
    from legoesm.ocean.physics.tendencies import wrap_ocean_tendencies
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.ocean.vertical import compute_ocean_jacobian
    # Private-symbol reuse across scripts is the sanctioned white-box pattern
    # (see _import_compare_omip_nemo); _apply_ocean_tendencies is the SCM's
    # own Euler-update operator — re-deriving it here would be worse.
    from legoesm.ocean.scm import _apply_ocean_tendencies, add_ocean_tendencies

    if tier not in ("T0", "T1"):
        raise ValueError(f"tier must be 'T0' or 'T1', got {tier!r}")
    if sw_mode not in ("penetrate", "top"):
        raise ValueError(
            f"sw_mode must be 'penetrate' or 'top', got {sw_mode!r}"
        )
    if set(series) != set(_FORCING_CHANNELS):
        raise ValueError(
            f"series channels {sorted(series)} != expected "
            f"{sorted(_FORCING_CHANNELS)}"
        )

    # Device put ONCE; the (n_rec, 1, 1) mini-stack IS the point, so the
    # nearest-neighbour gather indices are identically zero.
    stack = {k: jnp.asarray(np.asarray(v, dtype=np.float64))
             for k, v in series.items()}
    nn0 = jnp.zeros((1,), dtype=jnp.int32)
    physics_fn = scm.physics_fn
    grid, z_coord, dt = scm.grid, scm.z_coord, scm.dt
    wind_on = tier == "T1"            # STATIC gates (feature-gating doctrine:
    penetrate = sw_mode == "penetrate"  # Python if on static bool, not where)
    rho_w = float(constants.rho_water)
    T_freeze = float(constants.T_freeze)
    s_ref_3d, rho_0_3d = _virtual_salt_ref_constants()

    @jax.jit
    def step_fn(state, idx_t):
        horiz = state.eta.data.shape          # (1, 1, 1)
        dtype = state.T.data.dtype

        def _b(x):                            # any channel -> (1, 1, 1)
            return jnp.broadcast_to(jnp.asarray(x, dtype), horiz)

        def _g(name):                         # sampled record -> (1, 1)
            return stack[name][idx_t][nn0, nn0].reshape(1, 1)

        # -- bulk fluxes at the traced record from the EVOLVING SST --------
        sf = compute_omip2_surface_forcing_jax(
            state, forcing_stack=stack, nn_i=nn0, nn_j=nn0,
            grid_shape=(1, 1), idx_t=idx_t,
        )
        # Freshwater (mirror of compute_omip2_freshwater_forcing, runoff
        # EXCLUDED): interactive evaporation from the SAME shared NCAR leaf.
        T_sfc_K = state.T.data[..., 0] + T_freeze
        _, _, _, _, evap = air_sea_fluxes(
            u10=_g("u10"), v10=_g("v10"), T_air_K=_g("T_air"),
            q_air=_g("q_air"), T_sfc_K=T_sfc_K, slp_Pa=_g("slp"),
        )
        # E - P [m/s], positive = net evaporation = salinifying (evap
        # positive UP, precip positive INTO the ocean [kg/m^2/s]).
        e_minus_p = _b((evap - _g("precip")) / rho_w)

        # -- SW split + sign flip (mirrors the Core2PointFluxes closures) --
        q_tot = _b(sf.q_net)
        sw_net = _b(sf.sw_down)
        if penetrate:
            # 3-D core two-band split: surface layer gets non-solar + the
            # 0.06 skin fraction; 0.94*sw penetrates via Jerlov.
            q_top = q_tot - _SW_PENETRATION_FRACTION * sw_net
            sw_pen = _SW_PENETRATION_FRACTION * sw_net
        else:
            q_top = q_tot
            sw_pen = None
        # SIGN: the producer returns ATMOSPHERIC-convention tau; the SCM
        # prescribed path is OCEAN_DIRECT => apply -tau (atm_to_ocean_stress).
        if wind_on:
            tau_x_o = -_b(sf.tau_x)
            tau_y_o = -_b(sf.tau_y)
        else:
            # T0: no wind channels — u = v stay exactly 0.
            tau_x_o = jnp.zeros(horiz, dtype)
            tau_y_o = jnp.zeros(horiz, dtype)

        # Struct for KPP / shortwave penetration (mirror of
        # build_ocean_surface_forcing_struct: freshwater positive INTO the
        # ocean = -e_minus_p * rho_water).
        sfc = OceanSurfaceForcing(
            sw_down=sw_pen, q_net=q_top, tau_x=tau_x_o, tau_y=tau_y_o,
            freshwater=-e_minus_p * rho_w,
        )
        tend = physics_fn(state, grid, z_coord, surface_forcing=sfc)

        # Explicit top-layer wind + heat tendencies: prescribed_surface_forcing
        # REUSED (traced tau/Q_net scalars; E_minus_P=0.0 STATIC => its
        # local-S salt branch is OFF).
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        cfg = PrescribedForcingConfig(
            tau_x=tau_x_o[0, 0, 0], tau_y=tau_y_o[0, 0, 0],
            Q_net=q_top[0, 0, 0], E_minus_P=0.0, wind_profile="constant",
        )
        out = prescribed_surface_forcing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            z_coord, J, grid, cfg,
        )
        # Salt: SHARED 3-D virtual-salt leaf (codex 2026-07 #2), NOT the
        # local-S form.  dS/dt = -S_ref * F_fw / (rho_0 * dz_0) with
        # F_fw = sfc.freshwater [kg/m^2/s, + INTO ocean]; net evaporation
        # (F_fw < 0) => dS/dt > 0 salinifies — the 3-D OMIP step's closure
        # with the same config-default S_ref/rho_0.  dz_0 is the identical
        # top-layer thickness prescribed_surface_forcing uses.
        dz_0 = z_coord.dz_ref[0] * J
        dS_fw = virtual_salt_flux_from_net(
            sfc.freshwater, S_ref=s_ref_3d, dz_0=dz_0, rho_0=rho_0_3d,
        )
        nlev = state.T.data.shape[-1]
        pad_axes_T = ((0, 0),) * (state.T.data.ndim - 1)
        dS_dt_fw = jnp.pad(
            dS_fw[..., None], (*pad_axes_T, (0, nlev - 1)),
        ).astype(state.S.data.dtype)
        ftend = wrap_ocean_tendencies(
            out.du_dt, out.dv_dt, out.dT_dt, dS_dt_fw, state,
        )
        tend = add_ocean_tendencies(tend, ftend)

        # scm.step() sequence, reused verbatim (traced once inside this jit):
        # forward-Euler explicit update -> backward-Euler implicit vertical
        # diffusion -> Crank-Nicolson Coriolis rotation.
        state_star = _apply_ocean_tendencies(state, tend, dt)
        new_state = scm._apply_implicit_vertical_mixing(
            state_star, tend.K_v, tend.A_v, dt,
        )
        new_state = scm._apply_coriolis_rotation(new_state)
        return new_state, tend.K_v

    return step_fn


# ===========================================================================
# SCM construction + integration
# ===========================================================================


def build_physics_config(vmix: str, sw_mode: str, convection: str):
    """Column physics config mirroring the faithful 3-D OMIP choices.

    Same factory as the 3-D model (``make_ocean_physics``): vertical mixing
    scheme selectable, surface forcing 'none' (the SCM applies the CORE-II
    fluxes itself — 'prescribed'/'bulk' would double-count), lateral mixing /
    bottom drag off (no horizontal neighbours; drag is a dynamics-level
    concern), Jerlov two-band SW penetration in 'penetrate' mode (water type
    II — the ``_create_setup`` default of the 3-D runner), optional
    enhanced-diffusion convective adjustment (nu_conv = nu_bg = 0 under KPP:
    KPP owns convective momentum, the factory rejects anything else).
    """
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.convection.config import (
        EnhancedDiffusionConfig,
        OceanConvectionConfig,
    )
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.shortwave_penetration import (
        ShortwavePenetrationConfig,
    )
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig

    if vmix not in _VMIX_CHOICES:
        raise ValueError(
            f"unknown vmix scheme {vmix!r}; expected one of {_VMIX_CHOICES} "
            "(tke/catke are rejected by the ocean SCM)"
        )
    if convection not in ("none", "enhanced_diffusion"):
        raise ValueError(
            f"unknown convection scheme {convection!r}; expected 'none' or "
            "'enhanced_diffusion'"
        )
    if convection == "enhanced_diffusion":
        conv = OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(nu_conv=0.0, nu_bg=0.0),
        )
    else:
        conv = OceanConvectionConfig(scheme="none")
    sw_cfg = (
        ShortwavePenetrationConfig(scheme="jerlov_2band", water_type="II")
        if sw_mode == "penetrate" else None
    )
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme=vmix),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=conv,
        shortwave_penetration=sw_cfg,
    )


def build_scm(
    *,
    T_profile: np.ndarray,
    S_profile: np.ndarray,
    dz: np.ndarray,
    lat_deg: float,
    lon_deg: float,
    dt: float,
    tier: str,
    vmix: str,
    sw_mode: str,
    convection: str,
    forcing,
    t0_seconds: float = 0.0,
):
    """Construct the (SCM, per-step flux updater) pair for one run.

    ``lat_deg``/``lon_deg`` are the coordinates the CORE-II forcing and the
    Coriolis parameter are evaluated at — production (:func:`run_point_tier`)
    passes the SELECTED NEMO cell coords, never the requested point.

    The vertical grid is the EXACT NEMO wet-column grid
    (:func:`create_z_star_from_thicknesses` on the reference thicknesses),
    so profiles compare level-by-level with the referee without regridding.
    """
    import jax.numpy as jnp
    from legoesm.ocean.scm import OceanColumnModel
    from legoesm.ocean.vertical import create_z_star_from_thicknesses

    dz = np.asarray(dz, dtype=np.float64)
    n_wet = int(dz.size)
    if len(T_profile) != n_wet or len(S_profile) != n_wet:
        raise ValueError(
            f"profile/dz length mismatch: T={len(T_profile)} "
            f"S={len(S_profile)} dz={n_wet}"
        )
    z_coord = create_z_star_from_thicknesses(dz)
    fluxes = Core2PointFluxes(forcing, lat_deg, lon_deg, sw_mode=sw_mode)
    scm_forcing = fluxes.make_scm_forcing(tier, lat_deg)
    scm = OceanColumnModel.create(
        nlev=n_wet,
        dt=dt,
        T_profile=jnp.asarray(np.asarray(T_profile, dtype=np.float64)),
        S_profile=jnp.asarray(np.asarray(S_profile, dtype=np.float64)),
        physics_config=build_physics_config(vmix, sw_mode, convection),
        z_coord=z_coord,
        latitude_deg=float(lat_deg),
        longitude_deg=float(lon_deg),
        forcing=scm_forcing,
        implicit_vertical_mixing=True,   # forward-Euler + backward-Euler solve
        t0_seconds=float(t0_seconds),
    )
    return scm, fluxes


def run_point_tier(
    *,
    ic: dict,
    dz: np.ndarray,
    forcing,
    tier: str,
    vmix: str,
    sw_mode: str,
    convection: str,
    days: int,
    dt: float,
    t0_seconds: float,
    progress_label: str = "",
    speed_floor_steps_s: float = 5.0,
) -> dict:
    """Integrate one (point, tier, vmix) SCM run; daily profile history.

    Forcing sampling AND the Coriolis parameter use the SELECTED nearest-wet
    NEMO cell coordinates (``ic["cell_lat"]``, ``ic["cell_lon"]`` from
    :func:`extract_ic_profile`) — NOT the requested point, which can sit on
    land one or more cells away (codex 2026-07 finding #1): the IC and the
    referee column live at the cell, so CORE-II fluxes and ``f_c`` must too.
    The requested coordinates are metadata only (handled in :func:`main`).

    The hot loop drives the ONE jitted step from :func:`build_jitted_step`;
    the host only computes the CORE-II record index and pulls profiles at
    day boundaries.  Each simulated day prints a progress line (label, day,
    SST) and day 1 prints a steps/s SELF-CHECK measured EXCLUDING the first
    step (which pays the one-off jit compile): below
    ``speed_floor_steps_s`` a loud RETRACE-SUSPECTED warning fires — job
    8889618 ran 2.7 h with zero output because every eager step recompiled
    the implicit-solver loops.  ``K_v`` snapshots are the diffusivity that
    ACTED during the last step of each day (returned by the jitted step;
    the previous design recomputed physics on the post-step state).
    """
    import time as _time

    import jax.numpy as jnp

    # SELECTED cell coords (the column's true location on the NEMO grid).
    cell_lat = float(ic["cell_lat"])
    cell_lon = float(ic["cell_lon"])
    scm, fluxes = build_scm(
        T_profile=ic["T"], S_profile=ic["S"], dz=dz,
        lat_deg=cell_lat, lon_deg=cell_lon, dt=dt, tier=tier, vmix=vmix,
        sw_mode=sw_mode, convection=convection, forcing=forcing,
        t0_seconds=t0_seconds,
    )
    steps_per_day = int(round(86400.0 / dt))
    if abs(steps_per_day * dt - 86400.0) > 1e-6:
        raise ValueError(
            f"dt={dt} s does not divide one day; daily snapshots need "
            "86400 % dt == 0"
        )
    nsteps = int(days) * steps_per_day
    label = progress_label or f"{tier}/{vmix}"

    series = extract_point_forcing_series(forcing, cell_lat, cell_lon)
    step_fn = build_jitted_step(scm, series, tier=tier, sw_mode=sw_mode)
    n_rec = fluxes.n_rec          # Core2PointFluxes validated record spacing
    state = scm.state
    nlev = int(state.T.data.shape[-1])

    day, Ts, Ss, us, vs, Ks = [], [], [], [], [], []
    wall_start = _time.perf_counter()
    wall_after_first = wall_start
    for k in range(nsteps):
        # Record-selection convention: the 3-D host loop is 1-based and
        # samples the forcing at the END-of-step time (`for step in
        # range(1, n_steps+1): it = _idx_t(step, dt, ...)`, i.e. t=step*dt
        # for the step integrating [(step-1)dt, step*dt]).  Mirror it,
        # otherwise every record transition lags the 3-D run by exactly one
        # step (codex 2026-07 finding #2).
        idx = record_index(t0_seconds + (k + 1) * dt, n_rec)
        state, kv = step_fn(state, jnp.asarray(idx, dtype=jnp.int32))
        if k == 0:
            state.T.data.block_until_ready()      # compile boundary (timing)
            wall_after_first = _time.perf_counter()
        if (k + 1) % steps_per_day == 0:
            d = (k + 1) // steps_per_day
            day.append(d)
            Ts.append(np.asarray(state.T.data[0, 0, 0], dtype=np.float64))
            Ss.append(np.asarray(state.S.data[0, 0, 0], dtype=np.float64))
            us.append(np.asarray(state.u.data[0, 0, 0], dtype=np.float64))
            vs.append(np.asarray(state.v.data[0, 0, 0], dtype=np.float64))
            Ks.append(
                np.full(nlev, np.nan) if kv is None
                else np.asarray(kv, dtype=np.float64).reshape(-1)
            )
            print(f"[twins]   {label} day {d}/{int(days)}: "
                  f"SST={float(Ts[-1][0]):+.3f} degC "
                  f"SSS={float(Ss[-1][0]):.3f} g/kg", flush=True)
            if d == 1 and steps_per_day >= 2:
                # SPEED SELF-CHECK: steps/s over day 1 EXCLUDING step 1
                # (the jit compile lands there).  A healthy jitted column
                # runs O(100+) steps/s; < speed_floor means the step is
                # being re-traced/re-compiled or fell back to eager.
                span = max(_time.perf_counter() - wall_after_first, 1e-9)
                rate = (steps_per_day - 1) / span
                print(f"[twins]   {label} speed self-check: {rate:.1f} "
                      f"steps/s (first step incl. jit compile: "
                      f"{wall_after_first - wall_start:.1f} s)", flush=True)
                if rate < speed_floor_steps_s:
                    print(f"[twins]   !!! RETRACE-SUSPECTED: {rate:.2f} "
                          f"steps/s < {speed_floor_steps_s:.0f} — the step "
                          "function is likely re-traced/re-compiled per "
                          "step (per-step closure rebuild or eager lax "
                          "control flow); inspect build_jitted_step wiring "
                          "before burning walltime.", flush=True)
    # keep the SCM object coherent with the integrated trajectory
    scm.state = state
    scm.t_seconds = t0_seconds + nsteps * dt
    z_center = -np.asarray(scm.z_coord.z_full_ref, dtype=np.float64)
    return {
        "time_days": np.asarray(day, dtype=np.float64),
        "T": np.stack(Ts),
        "S": np.stack(Ss),
        "u": np.stack(us),
        "v": np.stack(vs),
        "K_v": np.stack(Ks),
        "z_center": z_center,       # positive-down [m]
        "f_c": float(scm.forcing.f_c),
        "cell_lat": cell_lat,       # coords the forcing + f_c were built at
        "cell_lon": cell_lon,
        "tier": tier,
        "vmix": vmix,
    }


# ===========================================================================
# Plotting
# ===========================================================================


def plot_point(
    name: str,
    lat: float,
    lon: float,
    ic: dict,
    runs: dict,
    nemo: dict | None,
    snapshot_days: list[int],
    out_png: Path,
) -> None:
    """One figure per point: T (left) and S (right), depth axis DOWN.

    Overlays: IC (black), lego T0 (dashed) / T1 (solid) at the snapshot days
    for every vmix scheme (primary = first key, full alpha; others thinner),
    NEMO monthly referee columns (red tones).  The title carries the
    ADVECTION CAVEAT verbatim.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axT, axS) = plt.subplots(1, 2, figsize=(11.5, 7.5), sharey=True)
    zc_ic = ic["nav_lev"][: ic["n_wet"]]
    axT.plot(ic["T"], zc_ic, color="k", lw=2.0, label="IC (NEMO woce init)")
    axS.plot(ic["S"], zc_ic, color="k", lw=2.0)

    tier_style = {"T0": "--", "T1": "-"}
    # matplotlib >= 3.9 removed plt.get_cmap — use the colormaps registry.
    cmap = {"T0": matplotlib.colormaps["Blues"],
            "T1": matplotlib.colormaps["Greens"]}
    vmix_names = list(dict.fromkeys(k[1] for k in runs))
    for (tier, vmix), r in sorted(runs.items()):
        primary = vmix == vmix_names[0]
        lw = 1.8 if primary else 1.0
        alpha = 1.0 if primary else 0.55
        for nd, d in enumerate(snapshot_days):
            sel = np.argmin(np.abs(r["time_days"] - d))
            frac = 0.35 + 0.55 * (nd + 1) / max(len(snapshot_days), 1)
            color = cmap[tier](frac)
            lbl = (
                f"lego {tier} {vmix} d{int(r['time_days'][sel])}"
                if primary else None
            )
            axT.plot(r["T"][sel], r["z_center"], tier_style[tier], color=color,
                     lw=lw, alpha=alpha, label=lbl)
            axS.plot(r["S"][sel], r["z_center"], tier_style[tier], color=color,
                     lw=lw, alpha=alpha)

    if nemo is not None:
        reds = matplotlib.colormaps["Reds"]
        for k, m in enumerate(nemo["months"]):
            frac = 0.4 + 0.5 * (k + 1) / max(len(nemo["months"]), 1)
            fin = np.isfinite(nemo["T"][k])
            axT.plot(nemo["T"][k][fin], nemo["deptht"][fin], ":",
                     color=reds(frac), lw=2.2, label=f"NEMO 3-D mon {m}")
            finS = np.isfinite(nemo["S"][k])
            axS.plot(nemo["S"][k][finS], nemo["deptht"][finS], ":",
                     color=reds(frac), lw=2.2)

    zmax = float(zc_ic[-1]) * 1.05
    for ax, xlab in ((axT, "T [degC]"), (axS, "S [g/kg]")):
        ax.set_ylim(zmax, 0.0)          # depth axis DOWN
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.3)
    axT.set_ylabel("depth [m]")
    axT.legend(fontsize=7, loc="lower left")
    fig.suptitle(
        f"SCM column twin — {name} ({lat:.1f}N, {lon:.1f}E), "
        f"ORCA1 cell ({ic['cell_lat']:.2f}N, {ic['cell_lon']:.2f}E)\n"
        + ADVECTION_CAVEAT,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ===========================================================================
# CLI
# ===========================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--point", action="append", default=None, metavar="NAME=LAT,LON",
        help="diagnostic point (repeatable); default = the 4 built-ins: "
             + ", ".join(f"{k}={v[0]},{v[1]}" for k, v in DEFAULT_POINTS.items()),
    )
    p.add_argument("--start-month", type=int, default=1,
                   help="IC month index (1=Jan) + forcing season start")
    p.add_argument("--days", type=int, default=90, help="integration length")
    p.add_argument("--dt", type=float, default=3600.0, help="SCM step [s]")
    p.add_argument("--vmix", type=str, default="kpp",
                   help="comma list of vertical-mixing schemes to run "
                        f"(subset of {_VMIX_CHOICES}; e.g. 'kpp,constant' "
                        "for the A/B)")
    p.add_argument("--tiers", type=str, default="T0,T1",
                   help="comma list of tiers (T0=no dynamics, T1=Ekman)")
    p.add_argument("--convection", type=str, default="none",
                   choices=("none", "enhanced_diffusion"),
                   help="convective-adjustment scheme (mirrors the 3-D "
                        "run's opt-in --convection)")
    p.add_argument("--sw-mode", type=str, default="penetrate",
                   choices=("penetrate", "top"),
                   help="'penetrate': non-solar q_net on top + Jerlov-II SW "
                        "over depth (3-D core split); 'top': full q_net in "
                        "the surface layer")
    p.add_argument("--ic-temp", type=Path, default=DEFAULT_IC_TEMP,
                   help="NEMO monthly T initial-condition file")
    p.add_argument("--ic-salt", type=Path, default=DEFAULT_IC_SALT,
                   help="NEMO monthly S initial-condition file")
    p.add_argument("--nemo-gridt", type=Path, default=DEFAULT_NEMO_GRIDT,
                   help="NEMO 3-D monthly grid_T referee file")
    p.add_argument("--nemo-clim", action="store_true",
                   help="referee = climatological calendar-month mean over "
                        "all records (default: FIRST occurrence = year-1 "
                        "trajectory)")
    p.add_argument("--forcing-path", type=str, default=None,
                   help="CORE-II NYF cache dir (load_core2_nyf cache_dir=); "
                        "default = the standard ocean-fidelity cache")
    p.add_argument("--allow-synthetic-forcing", action="store_true",
                   help="permit the synthetic-forcing fallback (NOT "
                        "OMIP-comparable; default refuses loudly)")
    p.add_argument("--max-point-distance-deg", type=float, default=2.0,
                   help="skip a point when the nearest wet column is "
                        "farther than this")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/omip_nemo/scm_twins"))
    return p


def _resolve_points(args) -> dict[str, tuple[float, float]]:
    if not args.point:
        return dict(DEFAULT_POINTS)
    pts: dict[str, tuple[float, float]] = {}
    for spec in args.point:
        name, lat, lon = parse_point_spec(spec)
        pts[name] = (lat, lon)
    return pts


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    points = _resolve_points(args)
    vmix_list = [v.strip() for v in args.vmix.split(",") if v.strip()]
    for v in vmix_list:
        if v not in _VMIX_CHOICES:
            raise SystemExit(
                f"--vmix {v!r} is not one of {_VMIX_CHOICES}"
            )
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    for t in tiers:
        if t not in ("T0", "T1"):
            raise SystemExit(f"--tiers entry {t!r} must be T0 or T1")
    if not (1 <= args.start_month <= 12):
        raise SystemExit(f"--start-month must be 1..12, got {args.start_month}")

    from legoesm.ocean.forcing import load_core2_nyf

    forcing = load_core2_nyf(
        cache_dir=(Path(args.forcing_path) if args.forcing_path else None),
        allow_synthetic=bool(args.allow_synthetic_forcing),
    )
    n_months = max(1, int(round(args.days / 30.0)))
    months = month_sequence(args.start_month, n_months)
    t0 = month_start_seconds(args.start_month)
    snapshot_days = [30 * (k + 1) for k in range(n_months)
                     if 30 * (k + 1) <= args.days]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(ADVECTION_CAVEAT)
    print(f"[twins] points={list(points)} tiers={tiers} vmix={vmix_list} "
          f"days={args.days} dt={args.dt} start_month={args.start_month} "
          f"sw_mode={args.sw_mode} convection={args.convection}")
    print("[twins] EXCLUDED vs the 3-D run: runoff, SSS restoring, sea ice "
          "(expect below-freezing drift at winter high-lat points).")

    n_ok = 0
    for name, (lat, lon) in points.items():
        print(f"\n=== point {name} ({lat:.2f}N, {lon:.2f}E) ===")
        try:
            ic = extract_ic_profile(
                args.ic_temp, args.ic_salt, args.start_month - 1, lat, lon,
                max_dist_deg=args.max_point_distance_deg,
            )
        except PointTooFarError as exc:
            print(f"[twins] WARN: skipping {name}: {exc}")
            continue
        print(f"[twins] IC vars: T={ic['temp_var']!r} S={ic['salt_var']!r}; "
              f"cell (j={ic['j']}, i={ic['i']}) at "
              f"({ic['cell_lat']:.2f}N, {ic['cell_lon']:.2f}E), "
              f"{ic['dist_deg']:.2f} deg from request; n_wet={ic['n_wet']}")

        nemo = None
        if Path(args.nemo_gridt).exists():
            nemo = extract_nemo_reference(
                args.nemo_gridt, ic["j"], ic["i"], months,
                clim=bool(args.nemo_clim),
            )
            if abs(nemo["cell_lat"] - ic["cell_lat"]) > 0.05 or (
                abs(nemo["cell_lon"] - ic["cell_lon"]) > 0.05
            ):
                raise SystemExit(
                    f"IC and referee grids disagree at (j={ic['j']}, "
                    f"i={ic['i']}): IC cell ({ic['cell_lat']:.3f}, "
                    f"{ic['cell_lon']:.3f}) vs referee "
                    f"({nemo['cell_lat']:.3f}, {nemo['cell_lon']:.3f})"
                )
            print(f"[twins] referee vars: T={nemo['t_var']!r} "
                  f"S={nemo['s_var']!r}; months={nemo['months']} "
                  f"records={nemo['records_used']} clim={nemo['clim']}")
            print(f"[twins] {ADVECTION_CAVEAT}")
        else:
            print(f"[twins] WARN: referee {args.nemo_gridt} missing — "
                  "running lego-only (no NEMO overlay)")

        # NEMO reference layer thicknesses: per-point e3t (shelf/partial-
        # bottom aware) when present, global deptht_bounds otherwise;
        # midpoint-rule fallback from the IC centre depths without a referee.
        if nemo is not None and nemo["dz"] is not None:
            dz_full = nemo["dz"]
            dz_src = nemo["dz_source"]
            if not np.allclose(
                nemo["deptht"], ic["nav_lev"], rtol=0.0, atol=0.5,
            ):
                raise SystemExit(
                    "IC nav_lev and referee deptht differ by > 0.5 m — "
                    "not the same 75-level grid; refusing to mix them"
                )
        else:
            dz_full = dz_from_center_depths(ic["nav_lev"])
            dz_src = "ic_nav_lev_midpoint"
        dz = np.asarray(dz_full[: ic["n_wet"]], dtype=np.float64)
        # Per-point e3t is fill (NaN) below the LOCAL seabed: the IC's wet
        # depth and the referee's must agree, else fail loudly rather than
        # integrate on a corrupt grid.
        if not (np.all(np.isfinite(dz)) and np.all(dz > 0.0)):
            raise SystemExit(
                f"layer thicknesses for {name} (source={dz_src}) are "
                f"non-finite/non-positive within the IC's {ic['n_wet']} wet "
                "levels — IC and referee wet columns disagree; refusing"
            )
        print(f"[twins] layer thicknesses: {dz_src} "
              f"(n_wet={ic['n_wet']}, dz_0={dz[0]:.3f} m, "
              f"dz_bot={dz[-1]:.3f} m)")

        runs: dict[tuple[str, str], dict] = {}
        for vmix in vmix_list:
            for tier in tiers:
                print(f"[twins] run {name} tier={tier} vmix={vmix} ...",
                      flush=True)
                # forcing + f_c at the SELECTED cell (ic["cell_lat"/
                # "cell_lon"]); the requested (lat, lon) is metadata only.
                r = run_point_tier(
                    ic=ic, dz=dz, forcing=forcing,
                    tier=tier, vmix=vmix, sw_mode=args.sw_mode,
                    convection=args.convection, days=args.days, dt=args.dt,
                    t0_seconds=t0,
                    progress_label=f"{name} {tier} {vmix}",
                )
                runs[(tier, vmix)] = r
                sst0, sstN = float(ic["T"][0]), float(r["T"][-1][0])
                print(f"[twins]   day {int(r['time_days'][-1])}: "
                      f"SST {sst0:+.2f} -> {sstN:+.2f} degC, "
                      f"SSS {float(ic['S'][0]):.2f} -> "
                      f"{float(r['S'][-1][0]):.2f} g/kg, "
                      f"|u|max {np.abs(r['u']).max():.3f} m/s")
                meta = {
                    "caveat": ADVECTION_CAVEAT,
                    "point": name, "lat": lat, "lon": lon,
                    "cell": [ic["j"], ic["i"]],
                    "cell_lat": ic["cell_lat"], "cell_lon": ic["cell_lon"],
                    "forcing_coords_note": (
                        "CORE-II forcing + f_c sampled at the SELECTED NEMO "
                        "cell (cell_lat, cell_lon); requested (lat, lon) is "
                        "metadata only"
                    ),
                    "dist_deg": ic["dist_deg"], "n_wet": ic["n_wet"],
                    "dz_source": dz_src,
                    "tier": tier, "vmix": vmix, "f_c": r["f_c"],
                    "sw_mode": args.sw_mode, "convection": args.convection,
                    "days": args.days, "dt": args.dt,
                    "start_month": args.start_month,
                    "t0_seconds": t0,
                    "ic_vars": [ic["temp_var"], ic["salt_var"]],
                    "nemo_vars": (
                        [nemo["t_var"], nemo["s_var"]] if nemo else None
                    ),
                    "nemo_months": months,
                    "excluded_forcings": [
                        "runoff", "sss_restoring", "sea_ice",
                    ],
                    "sign_notes": (
                        "tau closures are OCEAN_DIRECT = -tau_atm(bulk); "
                        "q_net + into ocean; e_minus_p>0 = net evaporation; "
                        "salt = 3-D virtual-salt closure "
                        "dS/dt=-S_ref*F_fw/(rho_0*dz_0), S_ref/rho_0 from "
                        "LatLonCGridOceanConfig defaults"
                    ),
                    "ts_convention_note": (
                        "NEMO IC/referee carry TEOS-10 CT / (near-)SA; "
                        "legoESM column labels are potential T / PSU"
                    ),
                }
                npz = out_dir / f"scm_{name}_{tier}_{vmix}.npz"
                np.savez_compressed(
                    npz,
                    time_days=r["time_days"], z_center=r["z_center"],
                    T=r["T"], S=r["S"], u=r["u"], v=r["v"], K_v=r["K_v"],
                    ic_T=ic["T"], ic_S=ic["S"],
                    ic_z=ic["nav_lev"][: ic["n_wet"]],
                    nemo_T=(nemo["T"] if nemo else np.zeros((0, 0))),
                    nemo_S=(nemo["S"] if nemo else np.zeros((0, 0))),
                    nemo_z=(nemo["deptht"] if nemo else np.zeros(0)),
                    nemo_months=np.asarray(months),
                    meta=json.dumps(meta),
                )
                print(f"[twins]   wrote {npz}")

        out_png = out_dir / f"scm_twin_{name}.png"
        plot_point(name, lat, lon, ic, runs, nemo, snapshot_days, out_png)
        print(f"[twins] wrote {out_png}")
        print(f"[twins] {ADVECTION_CAVEAT}")
        n_ok += 1

    if n_ok == 0:
        raise SystemExit("no point produced a comparison (all skipped)")
    print(f"\n[twins] done: {n_ok}/{len(points)} points. {ADVECTION_CAVEAT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
