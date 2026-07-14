"""Reader for SAM CASE forcing files (snd / lsf / sfc) — GATE, LBA, DYNAMO…

SAM drives a CRM case from three ASCII files in ``CASES/<NAME>/`` (parsed by
``setdata.f90`` / ``forcing.f90``); this module ingests them so the legoESM
plane CRM can run the SAME deep-convection cases.  All parsing + interpolation
is host-side NumPy (it runs once at setup), returning plain arrays the driver
converts to JAX and feeds to the D3 large-scale-forcing operator
(:mod:`legoesm.atmosphere.forcing.plane_large_scale_forcing`).

File formats
------------
``snd`` (initial sounding) and ``lsf`` (large-scale forcing) share a layout::

    <text column header>
    <day>, <levels>, <pres0[mb]>   day,levels,pres0      # time-block header
    <levels rows of numeric columns>
    <day>, <levels>, <pres0>   ...                        # next time block
    ...

* ``snd`` columns: ``z[m]  p[mb]  θ[K]  q[g/kg]  u[m/s]  v[m/s]``  (θ is
  potential temperature — SAM's ``tabssnd`` absolute-T variant, signalled by a
  negative first value, is also handled).
* ``lsf`` columns: ``z[m]  p[mb]  dθ/dt[K/s]  dq/dt[kg/kg/s]  u_ls  v_ls
  w_ls[m/s]``.

``sfc`` (surface) is a flat table after a one-line header::

    day  sst(K)  H(W/m2)  LE(W/m2)  TAU(m2/s2)

Vertical interpolation is linear in height (matching ``forcing.f90``); values
outside the sounding range clamp to the nearest endpoint.  Time interpolation
(for multi-block ``lsf`` / ``sfc``) is linear in ``day``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Repo-local forcing cache populated by ``scripts/data/fetch_les_forcing.py``.
# This module lives at packages/atmosphere/legoesm/atmosphere/; the repo root is
# four parents up (…/legoESM/).
_LOCAL_FORCING_CACHE = Path(__file__).resolve().parents[4] / "data" / "les_cases"


def resolve_sam_case_dir(case_name: str) -> str:
    """Resolve the ``CASES/<case_name>`` deck directory for a gSAM-backed case.

    Priority:

    1. ``$LEGOESM_GSAM_ROOT/CASES/<case_name>`` — an explicit external gSAM
       checkout (set by the user) wins, if the deck is present there.
    2. The repo-local cache ``data/les_cases/<case_name>`` (override the root
       with ``$LEGOESM_LES_FORCING``), populated by
       ``scripts/data/fetch_les_forcing.py``.

    The cache path is returned even when absent, so a caller's existence check
    raises an actionable error (run the fetch script, pass ``--case-dir``, or set
    ``LEGOESM_GSAM_ROOT``).  ``case_name`` is the gSAM directory name, e.g.
    ``"BOMEX"`` / ``"DYCOMS_RF01"`` / ``"GATE_IDEAL"``.
    """
    env_root = os.environ.get("LEGOESM_GSAM_ROOT")
    if env_root:
        cand = Path(env_root).expanduser() / "CASES" / case_name
        if cand.is_dir():
            return str(cand)
    cache_root = os.environ.get("LEGOESM_LES_FORCING")
    cache = Path(cache_root).expanduser() if cache_root else _LOCAL_FORCING_CACHE
    return str(cache / case_name)


class SAMSounding(NamedTuple):
    """Initial sounding on the file's native levels (bottom-to-top)."""
    z: np.ndarray            # [m]
    p: np.ndarray            # [mb]
    theta: np.ndarray        # potential temperature [K]
    q_v: np.ndarray          # water-vapour mixing ratio [kg/kg] (converted)
    u: np.ndarray            # [m/s]
    v: np.ndarray            # [m/s]
    pres0: float             # surface pressure [mb]


class SAMForcing(NamedTuple):
    """Large-scale forcing time-series on the file's native levels.

    ``days`` is ``(n_time,)``; every profile is ``(n_time, n_lev)``.
    """
    days: np.ndarray
    z: np.ndarray            # (n_time, n_lev) [m]
    T_ls: np.ndarray         # dT_abs/dt|_ls [K/s] (SAM forcing.f90 adds tls to
                             # liquid static energy t≈tabs — an ABSOLUTE-T
                             # tendency, NOT dθ/dt; convert via /exner to use it)
    qv_ls: np.ndarray        # dq_v/dt|_ls [(kg/kg)/s]
    u_ls: np.ndarray         # large-scale / target u [m/s]
    v_ls: np.ndarray         # large-scale / target v [m/s]
    w_ls: np.ndarray         # large-scale vertical velocity [m/s] (positive up)


class SAMSurface(NamedTuple):
    """Surface boundary time-series."""
    days: np.ndarray
    sst: np.ndarray          # [K]
    shf: np.ndarray          # sensible heat flux [W/m^2]  (0 ⇒ interactive)
    lhf: np.ndarray          # latent heat flux [W/m^2]    (0 ⇒ interactive)
    tau: np.ndarray          # surface stress [m^2/s^2]    (0 ⇒ interactive)


def _parse_blocks(path: Path, n_cols: int):
    """Parse the SAM block format → (days(n_time,), data(n_time,n_lev,n_cols)).

    Requires every block to declare the same ``levels`` count (true for all
    SAM CASE files — the vertical grid is fixed across forcing times).
    """
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"SAM case file {path} is empty.")
    i = 1  # line 0 is the text column header
    days, blocks = [], []
    n_lev_ref = None
    while i < len(lines):
        hdr = lines[i].split()
        try:
            day = float(hdr[0].rstrip(","))
            levels = int(float(hdr[1].rstrip(",")))
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"{path}: expected a 'day, levels, pres0' block header at "
                f"line {i + 1}, got {lines[i]!r}."
            ) from exc
        if n_lev_ref is None:
            n_lev_ref = levels
        elif levels != n_lev_ref:
            raise ValueError(
                f"{path}: block at line {i + 1} declares {levels} levels but "
                f"an earlier block had {n_lev_ref}; SAM CASE files keep the "
                f"vertical grid fixed across times."
            )
        i += 1
        rows = []
        for _ in range(levels):
            if i >= len(lines):
                raise ValueError(
                    f"{path}: block starting near line {i} is truncated "
                    f"(expected {levels} data rows)."
                )
            toks = lines[i].split()
            if len(toks) < n_cols:
                raise ValueError(
                    f"{path}: line {i + 1} has {len(toks)} columns, "
                    f"expected >= {n_cols}: {lines[i]!r}."
                )
            rows.append([float(x) for x in toks[:n_cols]])
            i += 1
        days.append(day)
        blocks.append(np.asarray(rows, dtype=np.float64))
    return np.asarray(days, dtype=np.float64), np.stack(blocks, axis=0)


def _validate_z(z: np.ndarray, path) -> None:
    """Reject pressure-coordinate / non-monotonic soundings (codex iter-32 D).

    The supported path is a HEIGHT-coordinate sounding: ``z`` strictly
    increasing and non-negative (SAM bottom-to-top).  SAM also accepts
    pressure-coordinate soundings (signalled by ``zsnd[1] <= zsnd[0]`` / negative
    z); those need the p-grid branch of ``forcing.f90`` and are not supported.
    """
    if np.any(z < 0.0):
        raise NotImplementedError(
            f"{path}: negative heights — pressure-coordinate SAM soundings are "
            "not supported (only strictly-increasing height grids)."
        )
    if not np.all(np.diff(z) > 0.0):
        raise ValueError(
            f"{path}: sounding heights must be STRICTLY INCREASING "
            "(bottom-to-top); got a non-monotonic / pressure-coordinate grid."
        )


def read_sam_snd(path) -> SAMSounding:
    """Read a SAM ``snd`` initial-sounding file (first time block).

    The ``θ`` column is potential temperature (returned as-is for the θ-based
    plane CRM); absolute-temperature soundings (SAM ``tabssnd``, negative first
    value) are rejected. The ``q`` column is mixing ratio [g/kg]; SAM's
    relative-humidity convention (negative q) is rejected (see below).
    """
    _, data = _parse_blocks(Path(path), n_cols=6)
    blk = data[0]  # (n_lev, 6) — soundings use the first (day-0) block
    # Finite guard BEFORE the sign checks (codex iter-56 LOW): a NaN slips
    # through ``theta[0] < 0`` / ``q < 0`` (both False for NaN) and would silently
    # propagate into the IC; reject non-finite sounding values up front.
    if not np.all(np.isfinite(blk)):
        raise ValueError(
            f"{path}: sounding contains non-finite (NaN/Inf) values.")
    _validate_z(blk[:, 0], path)
    theta = blk[:, 2]
    # SAM's tabssnd convention: a negative first θ entry flags an ABSOLUTE-T
    # sounding stored negated (forcing.f90: ``tabssnd = tsnd(1,1) < 0``).
    if theta[0] < 0.0:
        raise NotImplementedError(
            f"{path}: absolute-temperature sounding (tabssnd, negative θ) is "
            "not yet supported — only potential-temperature soundings are."
        )
    # SAM's RH-sounding convention (forcing.f90:111 ``if(qg0(k).lt.0.)``): a
    # NEGATIVE q entry is RELATIVE HUMIDITY [%], converted to mixing ratio via
    # ``-q/100·qsatw(T,p)``. The GATE/LBA/RCE target soundings are all g/kg
    # (q≥0); an RH sounding (DYNAMO/TOGA-COARE) is REJECTED rather than silently
    # turned into a NEGATIVE mixing ratio (the same silent-units trap class as
    # the iter-55 FORCING-T tls bug).
    q_raw = blk[:, 3]
    if np.any(q_raw < 0.0):
        raise NotImplementedError(
            f"{path}: relative-humidity sounding (SAM negative-q convention) is "
            "not supported — only g/kg mixing-ratio soundings (q≥0). Convert via "
            "-q/100·qsat(T,p) at model levels if an RH case is added."
        )
    return SAMSounding(
        z=blk[:, 0], p=blk[:, 1], theta=theta,
        q_v=q_raw * 1.0e-3,               # g/kg → kg/kg
        u=blk[:, 4], v=blk[:, 5],
        pres0=_read_pres0(path),
    )


def us_standard_atmosphere_temperature(z_m: np.ndarray) -> np.ndarray:
    """US Standard Atmosphere 1976 temperature [K] vs geometric height [m].

    The piecewise-linear base-layer profile SAM's ``atmosphere()`` returns and
    uses for the above-sounding T-ratio extrapolation: 6.5 K/km tropospheric
    lapse to 11 km, ISOTHERMAL 216.65 K (11-20 km), +1.0 K/km (20-32 km),
    +2.8 K/km (32-47 km), isothermal 270.65 K (47-51 km). Good to ~80 km; the
    CRM only extrapolates the 11-32 km band.

    Argument is GEOMETRIC height (matching SAM, which calls
    ``atmosphere(z(iz)/1000.)`` with the geometric model height — `forcing.f90:92`);
    the geopotential-vs-geometric difference is ~0.5 % at 30 km ⇒ ~0.1 K, below the
    faithfulness margin, so no geopotential conversion is applied.
    """
    z = np.asarray(z_m, dtype=np.float64) / 1000.0           # → km
    # (base height km, base T K, lapse K/km with T increasing upward)
    layers = [(0.0, 288.15, -6.5), (11.0, 216.65, 0.0), (20.0, 216.65, 1.0),
              (32.0, 228.65, 2.8), (47.0, 270.65, 0.0)]
    T = np.full_like(z, layers[-1][1] + layers[-1][2] * (z - layers[-1][0]))
    for (zb, Tb, lapse), (zt, _, _) in zip(layers, layers[1:]):
        in_layer = (z >= zb) & (z < zt)
        T = np.where(in_layer, Tb + lapse * (z - zb), T)
    T = np.where(z < layers[0][0], layers[0][1], T)          # below sea level
    return T


def extend_sounding_to_top(snd: SAMSounding, z_top_model: float,
                           *, margin: float = 2000.0, dz: float = 250.0
                           ) -> SAMSounding:
    """Extend a sounding above its top to cover the model top (SND-TOP).

    SAM (``forcing.f90:92-99``) extrapolates the background ABOVE the sounding
    top up the US-standard-atmosphere temperature RATIO
    ``tt(iz)=ratio_t2/ratio_t1·tt(iz-1)`` (telescopes to ``T(z)=T_top·T_std(z)/
    T_std(z_top)``), with the moisture decaying ``q(z)=q_top·exp(-Δz/3000)`` and
    the wind held CONSTANT. We replicate that EXACTLY (codex iter-57): the
    sounding-top absolute temperature ``T_top`` is scaled by the std-atm ratio
    (so it stays isothermal through 11-20 km AND warms +1 K/km above 20 km, which
    a flat-isothermal fill missed by several K / ~5-10 % N² at 30 km). The
    hydrostatic pressure is integrated incrementally with this T profile and
    ``θ(z)=T(z)/Π(z)`` (rises ⇒ STABLE), replacing the old ``jnp.interp``
    constant-θ clamp (a dry-NEUTRAL stratosphere, ``T_ref→`` tens of K aloft).

    Extending the sounding ONCE makes the reference θ
    (``build_sam_case_height_coord``) and the IC (``interp_sounding_to_levels``)
    use the SAME profile, so θ'≈0 at init. Levels run every ``dz`` from the snd
    top to ``z_top_model+margin`` so the model top is always COVERED (no silent
    re-clamp); ``build_sam_case_height_coord`` asserts the coverage.
    """
    from legoesm import constants
    z_top = float(snd.z[-1])
    z_needed = float(z_top_model) + float(margin)
    if z_top >= z_needed:
        return snd
    p_top_mb = float(snd.p[-1])
    if not 1.0 < p_top_mb < 1100.0:        # codex LOW: guard the hPa assumption
        raise ValueError(
            f"extend_sounding_to_top: sounding top pressure {p_top_mb} is not a "
            "plausible hPa value (expected ~1-1100 mb); reader unit mismatch?")
    p_top_pa = p_top_mb * 100.0
    exner_top = (p_top_pa / constants.p_ref) ** constants.kappa
    T_top = float(snd.theta[-1]) * exner_top
    # codex MEDIUM: cover the model top robustly (ceil, not a fragile arange end)
    n = int(np.ceil((z_needed - z_top) / dz))
    z_new = z_top + dz * np.arange(1, n + 1)
    # SAM std-atm-ratio absolute temperature above the top
    T_std = us_standard_atmosphere_temperature(z_new)
    T_std_top = us_standard_atmosphere_temperature(np.array([z_top]))[0]
    T_new = T_top * T_std / T_std_top
    # incremental hydrostatic p with this (non-isothermal) T: integrate
    # dln p = -g dz/(R_d T) over each [z_{k-1}, z_k] using the layer-mean T.
    z_edges = np.concatenate([[z_top], z_new])
    T_edges = np.concatenate([[T_top], T_new])
    T_mid = 0.5 * (T_edges[:-1] + T_edges[1:])
    dln = -constants.g * np.diff(z_edges) / (constants.R_d * T_mid)
    p_new_pa = p_top_pa * np.exp(np.cumsum(dln))
    exner_new = (p_new_pa / constants.p_ref) ** constants.kappa
    theta_new = T_new / exner_new                            # rises (stable)
    q_new = float(snd.q_v[-1]) * np.exp(-(z_new - z_top) / 3000.0)
    return SAMSounding(
        z=np.concatenate([snd.z, z_new]),
        p=np.concatenate([snd.p, p_new_pa / 100.0]),         # back to mb
        theta=np.concatenate([snd.theta, theta_new]),
        q_v=np.concatenate([snd.q_v, q_new]),
        u=np.concatenate([snd.u, np.full_like(z_new, float(snd.u[-1]))]),
        v=np.concatenate([snd.v, np.full_like(z_new, float(snd.v[-1]))]),
        pres0=snd.pres0,
    )


def _read_pres0(path) -> float:
    """Surface pressure [mb] from the first block header (``…, pres0``)."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    hdr = lines[1].split()
    return float(hdr[2].rstrip(","))


def read_sam_lsf(path, *, wls_kind: str = "w") -> SAMForcing:
    """Read a SAM ``lsf`` large-scale-forcing file (all time blocks).

    ``wls_kind`` (codex iter-32 E): the last column is large-scale vertical
    velocity in m/s (``"w"``, the default and the GATE/LBA/RCEMIP convention).
    SAM's ``wgls_holds_omega`` mode instead stores ω=dp/dt [Pa/s] there and
    converts ``w = -ω/(ρ g)``; that needs the column density, so ``"omega"`` is
    explicitly rejected rather than silently mis-read as m/s.

    ``tls`` is the ABSOLUTE-temperature tendency dT/dt|_ls [K/s] — SAM's
    ``forcing.f90`` adds it straight to the prognostic liquid/ice static-energy
    ``t`` (``=tabs+gamaz-L·qcond/cp``) with NO ``/prespot`` conversion, so it is
    a T tendency, NOT a θ tendency (the consumer must divide by exner to obtain
    dθ/dt). ``qls`` is dq_v/dt|_ls [(kg/kg)/s]; ``u_ls``/``v_ls`` are the
    geostrophic/reference wind (Coriolis ref, not a direct momentum source).
    """
    if wls_kind not in ("w", "omega"):
        raise ValueError(
            f"read_sam_lsf: wls_kind={wls_kind!r} invalid; use 'w' or 'omega'."
        )
    if wls_kind == "omega":
        raise NotImplementedError(
            "read_sam_lsf: wgls_holds_omega (ω=dp/dt [Pa/s]) soundings need a "
            "column density to convert w=-ω/(ρg); not yet supported. Supply an "
            "lsf with w in m/s (wls_kind='w')."
        )
    days, data = _parse_blocks(Path(path), n_cols=7)
    if not np.all(np.diff(days) > 0.0):
        raise ValueError(
            f"{path}: lsf block days must be STRICTLY INCREASING; got {days} "
            "(duplicate/decreasing time blocks are ambiguous for time interp)."
        )
    _validate_z(data[0, :, 0], path)
    return SAMForcing(
        days=days, z=data[..., 0],
        T_ls=data[..., 2], qv_ls=data[..., 3],
        u_ls=data[..., 4], v_ls=data[..., 5], w_ls=data[..., 6],
    )


def read_sam_sfc(path) -> SAMSurface:
    """Read a SAM ``sfc`` surface-boundary file."""
    arr = np.loadtxt(Path(path), skiprows=1, dtype=np.float64)
    arr = np.atleast_2d(arr)
    return SAMSurface(
        days=arr[:, 0], sst=arr[:, 1],
        shf=arr[:, 2], lhf=arr[:, 3], tau=arr[:, 4],
    )


class SAMRadForcing(NamedTuple):
    """Prescribed radiative-heating time-series (SAM ``doradforcing``).

    ``days`` is ``(n_time,)``; ``z`` and ``dTdt_rad`` are ``(n_time, n_lev)``.
    """
    days: np.ndarray
    z: np.ndarray            # (n_time, n_lev) [m]
    dTdt_rad: np.ndarray     # prescribed radiative dT/dt [K/s]


def read_sam_rad(path) -> SAMRadForcing:
    """Read a SAM ``rad`` prescribed-radiative-cooling file (LBA-style).

    Block format: a header line, then per time block ``<day>, <levels>`` and
    ``levels`` rows of ``z[m]  (dT/dt)rad[K/s]`` (a fixed radiative heating
    profile, e.g. ≈−1.4 K/day in the BL → 0 in the stratosphere). Used when the
    case ``prm`` sets ``doradforcing=.true.`` (LBA) instead of interactive RRTM.
    """
    days, data = _parse_blocks(Path(path), n_cols=2)
    return SAMRadForcing(days=days, z=data[..., 0], dTdt_rad=data[..., 1])


class SAMGrid(NamedTuple):
    """SAM ``grd``-file vertical grid.

    The two arrays are in DIFFERENT orientations on purpose (codex iter-52 G:
    the field names make this explicit so neither is mis-consumed):
    ``z_full_bottom_up`` is the raw grd scalar levels (bottom→top, SAM order);
    ``z_half`` is the legoESM-convention interfaces (TOP→bottom) ready to feed
    :func:`create_height_coordinate_from_z_half`.
    """
    z_full_bottom_up: np.ndarray   # (nlev,) cell-CENTRE heights, BOTTOM-TO-TOP [m]
    z_half: np.ndarray             # (nlev+1,) interfaces TOP-TO-BOTTOM (z_half[0]=top)


def read_sam_grd(path) -> SAMGrid:
    """Parse a SAM ``grd`` vertical-grid file (the EXACT SAM levels).

    Format: each line ``z[m]  level_index  spacing[m]``, bottom-to-top, where
    ``z`` is the scalar (cell-CENTRE) height. SAM's GATE ``grd`` is dz=50 m in
    the boundary layer, ~100 m uniform through the deep-convection layer
    (5-17 km), then stretched to ~30 km (266 levels) — a profile NO geometric
    stretch matches.

    Interfaces follow SAM's convention ``zi(k)=½(z(k-1)+z(k))`` (the MIDPOINTS
    of the scalar levels), ``zi(0)=0`` (surface), top extrapolated — so the
    cell centre is the midpoint of its interfaces (NOT ``z±spacing/2``; the 3rd
    column is the centre spacing, not the layer thickness). REVERSED to
    legoESM's top-to-bottom convention. Feed ``z_half`` to
    :func:`legoesm.grids.vertical.create_height_coordinate_from_z_half`.
    """
    z_rows = []
    for line in Path(path).read_text().splitlines():
        toks = line.split()
        if len(toks) < 3:
            continue
        try:
            z_rows.append(float(toks[0]))
        except ValueError:
            continue
    if len(z_rows) < 2:
        raise ValueError(f"read_sam_grd: <2 valid 'z idx spacing' rows in {path}")
    z_full_bu = np.asarray(z_rows, dtype=np.float64)   # cell centres, bottom→top
    if np.any(np.diff(z_full_bu) <= 0.0) or z_full_bu[0] <= 0.0:
        raise ValueError("read_sam_grd: scalar levels must be strictly "
                         "increasing and positive.")
    zi = np.empty(z_full_bu.shape[0] + 1, dtype=np.float64)   # surface→top
    zi[0] = 0.0
    zi[1:-1] = 0.5 * (z_full_bu[:-1] + z_full_bu[1:])
    zi[-1] = 2.0 * z_full_bu[-1] - zi[-2]                # top interface
    # NOTE: legoESM's HeightCoordinate then takes z_full = midpoint(z_half),
    # which differs from SAM's scalar z by ≤~1.5 m where the grid stretches
    # (SAM's z is not the geometric midpoint of its w-levels) — negligible for
    # the reference-state interpolation.
    return SAMGrid(z_full_bottom_up=z_full_bu, z_half=zi[::-1].copy())


def interp_rad_to_levels(rad: SAMRadForcing, z_model, day: float = 0.0):
    """Interpolate the prescribed radiative dT/dt [K/s] onto ``z_model`` at
    time ``day`` (linear in height + time, like the lsf forcing)."""
    z_model = np.asarray(z_model, dtype=np.float64)
    n_time = rad.days.shape[0]
    per_time = np.stack(
        [_interp_z(rad.z[t], rad.dTdt_rad[t], z_model) for t in range(n_time)],
        axis=0)
    if n_time == 1:
        return per_time[0]
    return np.array([np.interp(day, rad.days, per_time[:, j])
                     for j in range(z_model.shape[0])])


def _interp_z(z_data: np.ndarray, vals: np.ndarray,
              z_model: np.ndarray) -> np.ndarray:
    """Linear interpolation in height onto ``z_model``, SAM-faithful at edges.

    ``z_data`` must be strictly increasing (SAM soundings are bottom-to-top);
    ``z_model`` may be in any order (the plane CRM stores it top-to-bottom) and
    the result follows ``z_model``'s order.

    Edge handling matches ``forcing.f90`` (codex iter-32 A): the interior is
    linear between bracketing levels; BELOW the lowest level the value is
    LINEARLY EXTRAPOLATED from the bottom two levels (SAM's ``i=2`` branch gives
    ``coef<0``); ABOVE the top level SAM leaves the highest-bracket value, i.e.
    it HOLDS the top — which is ``np.interp``'s right-clamp.
    """
    out = np.interp(z_model, z_data, vals)          # interior linear + clamped
    below = np.asarray(z_model) < z_data[0]
    if np.any(below) and z_data.shape[0] >= 2:
        slope = (vals[1] - vals[0]) / (z_data[1] - z_data[0])
        out = np.where(below, vals[0] + slope * (z_model - z_data[0]), out)
    return out


def interp_sounding_to_levels(snd: SAMSounding, z_model) -> dict:
    """Interpolate the initial sounding onto the model levels ``z_model``."""
    z_model = np.asarray(z_model, dtype=np.float64)
    return {
        "theta": _interp_z(snd.z, snd.theta, z_model),
        "q_v": _interp_z(snd.z, snd.q_v, z_model),
        "u": _interp_z(snd.z, snd.u, z_model),
        "v": _interp_z(snd.z, snd.v, z_model),
        "p": _interp_z(snd.z, snd.p, z_model),
    }


def interp_forcing_to_levels(lsf: SAMForcing, z_model, day: float = 0.0) -> dict:
    """Interpolate the large-scale forcing onto ``z_model`` at time ``day``.

    Each time block is first interpolated in height onto ``z_model``; the
    resulting per-time profiles are then linearly interpolated in ``day``
    (clamped to the first/last block outside the time range).  Returns the
    five profiles the D3 operator consumes: ``w_ls`` (subsidence), ``T_adv``
    (prescribed ABSOLUTE-temperature advective tendency [K/s] — divide by exner
    to get dθ/dt) + ``qv_adv`` (prescribed q_v tendency), ``u_ls`` + ``v_ls``
    (nudging targets).
    """
    z_model = np.asarray(z_model, dtype=np.float64)
    n_time = lsf.days.shape[0]
    fields = {
        "w_ls": lsf.w_ls, "T_adv": lsf.T_ls, "qv_adv": lsf.qv_ls,
        "u_ls": lsf.u_ls, "v_ls": lsf.v_ls,
    }
    out = {}
    for name, src in fields.items():
        # (n_time, nlev_model): interp each block in height.
        per_time = np.stack(
            [_interp_z(lsf.z[t], src[t], z_model) for t in range(n_time)],
            axis=0,
        )
        if n_time == 1:
            out[name] = per_time[0]
        else:
            # linear in day, per model level
            out[name] = np.array([
                np.interp(day, lsf.days, per_time[:, j])
                for j in range(z_model.shape[0])
            ])
    return out


def surface_at_day(sfc: SAMSurface, day: float = 0.0) -> dict:
    """Linearly interpolate the surface boundary to ``day`` (clamped)."""
    return {
        "sst": float(np.interp(day, sfc.days, sfc.sst)),
        "shf": float(np.interp(day, sfc.days, sfc.shf)),
        "lhf": float(np.interp(day, sfc.days, sfc.lhf)),
        "tau": float(np.interp(day, sfc.days, sfc.tau)),
    }
