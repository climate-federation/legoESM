"""Surface blending utilities for AMIP-style experiments.

Provides common surface-property computations shared across scripts:
- Sea-ice / ocean temperature blending
- Sea-ice / ocean albedo and emissivity blending
- Column AOD distribution to model layers
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants


def blend_surface_temperature(
    sst: jnp.ndarray,
    sic: jnp.ndarray,
    T_ice: "float | jnp.ndarray",
) -> jnp.ndarray:
    """Blend ocean SST and sea-ice temperature by ice concentration.

    Parameters
    ----------
    sst : array
        Sea surface temperature [K].
    sic : array
        Sea-ice concentration [0, 1].
    T_ice : float or array
        Sea-ice temperature [K].  A scalar constant (prescribed-ice default)
        or a per-cell array (the prognostic ice skin); broadcasts against
        ``sst``/``sic`` either way.

    Returns
    -------
    T_sfc : array
        Blended surface temperature [K].
    """
    return sic * T_ice + (1.0 - sic) * sst


def land_lapse_adjusted_surface_temperature(
    T_sfc: jnp.ndarray,
    f_land: jnp.ndarray,
    z_sfc: jnp.ndarray,
    lapse_K_per_m: float,
) -> jnp.ndarray:
    """Lower the LAND fraction of a surface-temperature anchor by a lapse rate.

    AMIP SST loaders fill land cells with the nearest-ocean SST (a sea-level
    temperature).  Anchoring surface fluxes / radiation to that value at an
    elevated cell overheats the surface by ``lapse * z`` (e.g. ~+13 K at 2 km
    for 6.5 K/km), driving spurious surface fluxes and convection over
    highlands.  This applies the standard-atmosphere correction on the land
    fraction only::

        T_eff = T_sfc - f_land * lapse_K_per_m * max(z_sfc, 0)

    Sign convention: ``z_sfc`` positive up [m]; ``lapse_K_per_m > 0`` cools
    with height.  The ocean fraction (``f_land = 0``) is unchanged; negative
    elevations (below-sea-level basins) are clipped to zero rather than
    warmed.

    Parameters
    ----------
    T_sfc : array
        Surface-temperature anchor [K] (ocean/ice blended, land = nearest-
        ocean fill).
    f_land : array
        Land fraction in [0, 1], same shape as ``T_sfc``.
    z_sfc : array
        Surface elevation [m] (e.g. ``phis / g``), same shape.
    lapse_K_per_m : float
        Lapse rate [K/m]; 0 disables (returns ``T_sfc`` unchanged up to
        floating-point identity).

    Returns
    -------
    T_eff : array
        Lapse-adjusted surface temperature [K].
    """
    return T_sfc - f_land * lapse_K_per_m * jnp.maximum(z_sfc, 0.0)


def blend_surface_property(
    sic: jnp.ndarray,
    value_ice: float | jnp.ndarray,
    value_ocean: float | jnp.ndarray,
) -> jnp.ndarray:
    """Blend an ice and ocean surface property by ice concentration.

    Useful for albedo, emissivity, or any linearly-blended quantity.

    Parameters
    ----------
    sic : array
        Sea-ice concentration [0, 1].
    value_ice : float or array
        Property value over ice.
    value_ocean : float or array
        Property value over ocean.

    Returns
    -------
    blended : array
        Ice-concentration-weighted blend.
    """
    return sic * value_ice + (1.0 - sic) * value_ocean


def blended_surface_albedo(
    sic: jnp.ndarray,
    f_land: "jnp.ndarray | None",
    albedo_ice: "float | jnp.ndarray",
    albedo_ocean: "float | jnp.ndarray",
    albedo_land: "jnp.ndarray | None" = None,
) -> jnp.ndarray:
    """Tile-blended surface shortwave albedo (ocean/ice, then land).

    The single place the ocean/ice/land albedo blend is formed for the
    surface-albedo boundary condition handed to radiation::

        alpha_sea  = sic * alpha_ice + (1 - sic) * alpha_ocean
        alpha_sfc  = f_land * alpha_land + (1 - f_land) * alpha_sea

    Both steps are linear area-weighted tile blends, so the result is the
    area-mean albedo of the cell — the quantity a single-column radiation
    solver needs to reproduce the cell-mean reflected shortwave.

    ``albedo_land`` is REQUIRED whenever ``f_land`` is not None: falling back
    to the ocean value over land is the defect this helper exists to prevent
    (it silently applied ``albedo_ocean = 0.06`` to every land column on the
    MPAS lane).  A land fraction with no land albedo therefore raises rather
    than quietly returning an ocean-albedo field.

    Parameters
    ----------
    sic : array
        Sea-ice concentration [0, 1].
    f_land : array or None
        Land area fraction [0, 1].  ``None`` ⇒ pure ocean/ice surface.
    albedo_ice, albedo_ocean : float or array
        Ice / open-ocean shortwave albedo [0, 1].
    albedo_land : array or None
        Land shortwave albedo [0, 1] (snow-brightened by the caller when a
        snow-albedo feedback is active).  Required when ``f_land`` is given.

    Returns
    -------
    albedo : array
        Area-blended surface shortwave albedo [0, 1].

    Raises
    ------
    ValueError
        If a land fraction is supplied without a land albedo.
    """
    albedo = blend_surface_property(sic, albedo_ice, albedo_ocean)
    if f_land is None:
        return albedo
    if albedo_land is None:
        raise ValueError(
            "blended_surface_albedo: f_land was supplied without an "
            "albedo_land field. Refusing to apply the OCEAN albedo over land "
            "— that silently reflects far too little shortwave from every "
            "land column. Provide a land albedo map (--albedo-land-file), a "
            "surfdata-derived albedo, or the latitude-vegetation default "
            "(legoesm.surface_albedo.land_vegetation_albedo)."
        )
    return f_land * albedo_land + (1.0 - f_land) * albedo


def surface_temperature_for_lw_boundary(
    radiation: str,
    *,
    T_rad: jnp.ndarray,
    lw_up: jnp.ndarray,
) -> jnp.ndarray:
    """Surface temperature to feed a scheme's longwave boundary.

    A scheme reproduces the surface's true upward LW flux ``LW_out`` only if it
    is fed the temperature consistent with its OWN emissivity convention:

    * ``rrtmgp`` / ``rrtmg`` — use the radiative-equivalent ``T_rad`` paired with
      the tile-blended ``eps_col`` (``eps_col*sigma*T_rad^4 = LW_emit``); the
      ``(1-eps_col)*La`` reflection term then completes ``LW_out``.
    * ``gray`` / ``none`` — gray emits as a BLACK surface (``eps = 1``) and
      cannot honour ``eps_col``, so feeding it ``T_rad`` would emit
      ``sigma*T_rad^4 = LW_emit/eps_col`` and OVERSTATE the flux by ``1/eps_col``.
      Instead derive a black-surface BRIGHTNESS temperature from the complete
      upward flux, ``T_bb = (lw_up/sigma)^0.25``, so ``sigma*T_bb^4 = LW_out``
      exactly.

    Parameters
    ----------
    radiation : str
        Active radiation scheme.
    T_rad : array
        Radiative-equivalent skin temperature (``eps_col*sigma*T_rad^4 =
        LW_emit``).
    lw_up : array
        Total upward LW flux at the surface (``LW_out``) [W/m^2].

    Returns
    -------
    T : array
        Surface temperature to hand the LW boundary.
    """
    if radiation in ("rrtmgp", "rrtmg"):
        return T_rad
    # gray / none: black-surface brightness temperature from the full upward flux.
    return (jnp.maximum(lw_up, 1.0e-6) / constants.sigma_sb) ** 0.25


def surface_emissivity_for_lw_inversion(
    radiation: str,
    *,
    dynamic_emissivity: jnp.ndarray | None,
    static_sfc_emissivity: float | jnp.ndarray,
) -> float | jnp.ndarray:
    """Surface emissivity to invert a held ``lw_net`` back to gross ``lw_down``.

    The coupled drivers reconstruct gross ``lw_down`` from the held net surface
    longwave via ``lw_down = (lw_net + eps*sigma*T^4) / eps``.  For that round
    trip to be exact, ``eps`` MUST equal the emissivity the radiation scheme
    actually EMITTED the boundary with — otherwise a persistent O(1 W/m^2)
    surface-energy bias leaks in (the emissivity mismatch is independent of the
    skin temperature used).  This returns that matching emissivity:

    * ``rrtmgp`` / ``rrtmg`` WITH the dynamic surface-radiation feedback — the
      tile-blended ``eps_col`` (``dynamic_emissivity``, not ``None``), exactly
      what ``solve_columns(sfc_emissivity=emis_col)`` used.
    * ``rrtmgp`` / ``rrtmg`` WITHOUT the feedback (or before the first coupler
      response) — the static config surface emissivity it emitted with.
    * ``gray`` / ``none`` — an idealized BLACK surface (``eps = 1.0``).  Gray
      radiation keeps ``GrayRadiationConfig.sfc_emissivity = 1.0`` and the
      dynamic ``emis_col`` is intentionally NOT threaded into it, so the
      reconstruction must also use ``1.0`` regardless of any config emissivity.

    Parameters
    ----------
    radiation : str
        Active radiation scheme (``"rrtmgp"``, ``"rrtmg"``, ``"gray"``,
        ``"none"``, ...).
    dynamic_emissivity : array or None
        The coupler's tile-blended surface emissivity for this segment, or
        ``None`` when the dynamic feedback is off / unavailable.
    static_sfc_emissivity : float or array
        The static config surface emissivity the scheme falls back to.

    Returns
    -------
    eps : float or array
        Emissivity to use for the ``lw_net`` -> ``lw_down`` inversion.
    """
    if radiation in ("rrtmgp", "rrtmg"):
        if dynamic_emissivity is not None:
            return dynamic_emissivity
        return static_sfc_emissivity
    # gray / none: idealized black surface (emit with eps = 1.0).
    return 1.0
def snow_fraction(
    T_low: jnp.ndarray,
    T_freeze: float,
    transition_center_offset_K: float = 2.0,
    transition_width_K: float = 4.0,
) -> jnp.ndarray:
    """Smooth rain/snow partition fraction (Wigmosta 1994 / Dai 2008).

    Linear ramp from 0 (all rain) at
    ``T_low = T_freeze + transition_center_offset_K`` down to 1 (all snow)
    at ``T_low = T_freeze + transition_center_offset_K - transition_width_K``.
    Replaces the hard step ``where(T_low < T_freeze, 1, 0)``, which zeroes
    ``d(snow)/d(T_low)`` on training/DA paths and miscounts mixed-phase
    precipitation in the 0–4 °C band. Single source of truth shared by the
    coupled and earth-system drivers so the two cannot silently diverge.

    Parameters
    ----------
    T_low : array
        Lowest-model-level air temperature [K].
    T_freeze : float
        Freezing point [K] (pass ``constants.T_freeze``).
    transition_center_offset_K : float
        Upper edge of the mixed-phase band above freezing [K] (default 2.0).
    transition_width_K : float
        Total width of the linear ramp [K] (default 4.0).

    Returns
    -------
    snow_frac : array
        Snow fraction in [0, 1].
    """
    return jnp.clip(
        (T_freeze + transition_center_offset_K - T_low) / transition_width_K,
        0.0,
        1.0,
    )


def distribute_column_aod_to_layers(
    aod_col: jnp.ndarray,
    p_half_col: jnp.ndarray,
) -> jnp.ndarray:
    """Distribute column aerosol optical depth to layers by pressure thickness.

    Parameters
    ----------
    aod_col : array, shape (ncol,)
        Total column AOD per column.
    p_half_col : array, shape (ncol, nlev+1)
        Half-level pressures.

    Returns
    -------
    aod_layers : array, shape (ncol, nlev)
        Layer-distributed AOD.
    """
    dp = jnp.clip(p_half_col[:, 1:] - p_half_col[:, :-1], 1.0e-12, None)
    w = dp / jnp.sum(dp, axis=1, keepdims=True)
    return jnp.clip(aod_col, 0.0, None)[:, None] * w


def place_stratospheric_aod_profile_to_layers(
    aod_profile: jnp.ndarray,
    p_edges: jnp.ndarray,
    p_half_col: jnp.ndarray,
) -> jnp.ndarray:
    """Conservatively remap a fixed-pressure-edge AOD profile onto model layers.

    Volcanic stratospheric aerosol is supplied as a per-layer absorption
    optical depth on the forcing file's OWN (altitude -> pressure) grid.  This
    bins it onto the model's pressure layers by fractional pressure overlap,
    so the aerosol lands at its true stratospheric pressure -- unlike
    :func:`distribute_column_aod_to_layers`, which spreads a single column AOD
    by full-column pressure mass and thus dumps ~90 % of a stratospheric layer
    into the troposphere.

    Method: the cumulative-OD curve is sampled at each model half level and the
    per-layer OD is the difference of the cumulative at the layer's two edges.
    The cumulative is interpolated in LOG-pressure, because the source OD is
    ``ext*dz`` -- uniform in geometric height (~ log-pressure) within a source
    layer, NOT uniform in pressure; a linear-pressure CDF would misallocate ~1 %
    of a 0.5 km source layer's OD across a cutting model interface (Codex review
    iter-1).  The remap is exact at source-layer edges and conserves the total
    column OD that falls within the model's pressure range (OD above the model
    top or below the surface edge is dropped -- it cannot be represented).

    The op set (searchsorted / gather / clip / min / max) is piecewise-linear:
    a.e.-differentiable in both ``aod_profile`` and ``p_half_col`` with
    well-defined one-sided gradients (kinks at source/model edges), and
    JIT/vmap-safe.

    Sign/units: optical depth is >= 0 and additive; pressure [Pa].

    Parameters
    ----------
    aod_profile : array, shape (ncol, nsrc)
        Per-source-layer absorption optical depth [-] (>= 0).
    p_edges : array, shape (nsrc+1,)
        Source-layer pressure edges [Pa], ASCENDING (index 0 = top / lowest
        pressure), aligned so ``aod_profile[:, j]`` occupies
        ``[p_edges[j], p_edges[j+1]]``.
    p_half_col : array, shape (ncol, nlev+1)
        Model half-level pressures [Pa] (index 0 = top).

    Returns
    -------
    aod_layers : array, shape (ncol, nlev)
        Layer absorption optical depth [-].
    """
    pe = jnp.asarray(p_edges)                                   # (nsrc+1,)
    aod = jnp.clip(jnp.asarray(aod_profile), 0.0, None)         # (ncol, nsrc)
    # Cumulative OD from the top: cum[:, k] = OD in source layers 0..k-1, so
    # cum aligns to the nsrc+1 source edges ``pe``.
    cum = jnp.concatenate(
        [jnp.zeros((aod.shape[0], 1), aod.dtype), jnp.cumsum(aod, axis=1)],
        axis=1,
    )                                                          # (ncol, nsrc+1)
    # Interpolate the CDF in log-pressure (source OD is uniform in geometric
    # height ~ log-pressure within a layer).
    l_edges = jnp.log(jnp.clip(pe, 1.0e-12, None))             # (nsrc+1,)

    ph = jnp.asarray(p_half_col)
    p_lo = jnp.minimum(ph[:, :-1], ph[:, 1:])                  # (ncol, nlev)
    p_hi = jnp.maximum(ph[:, :-1], ph[:, 1:])

    def _cdf_at(p):
        # Per-column log-pressure interp of the shared-edge cumulative at
        # pressures ``p``; clamps to [0, total] outside [pe[0], pe[-1]].
        lp = jnp.log(jnp.clip(p, 1.0e-12, None))
        idx = jnp.clip(jnp.searchsorted(l_edges, lp) - 1, 0, pe.shape[0] - 2)
        l0 = l_edges[idx]
        l1 = l_edges[idx + 1]
        frac = jnp.clip((lp - l0) / jnp.clip(l1 - l0, 1.0e-12, None), 0.0, 1.0)
        c0 = jnp.take_along_axis(cum, idx, axis=1)
        c1 = jnp.take_along_axis(cum, idx + 1, axis=1)
        return c0 + frac * (c1 - c0)

    return jnp.clip(_cdf_at(p_hi) - _cdf_at(p_lo), 0.0, None)


# --- prognostic sea-ice skin temperature (Semtner 1976 zero-layer + thermal
#     inertia; conductivity Untersteiner 1961 via constants.k_ice_default) ---

# Numerics floor for the skin update [K]: below the coldest observed polar
# surface (~185 K is the terrestrial record, Antarctic plateau) — a guard
# against a pathological flux spike, never active in normal operation.
_ICE_SKIN_FLOOR_K = 185.0


def prognostic_ice_skin_temperature(
    T_skin: jnp.ndarray,
    F_net_down_W_m2: jnp.ndarray,
    sic: jnp.ndarray,
    dt_s: float,
    h_ice_m: float,
    T_freeze_K: float = constants.T_freeze_ocean,
    k_ice_W_m_K: float = constants.k_ice_default,
    T_melt_surface_K: float = constants.T_freeze,
) -> jnp.ndarray:
    """Advance a prescribed-ice AMIP skin temperature one integration step.

    Semtner (1976) zero-layer thermodynamics — conductive flux through a
    climatological ice slab of thickness ``h_ice_m`` toward the seawater
    freezing point ``T_freeze_K`` at the BASE (ice/ocean interface) — plus
    the slab's half thermal inertia (``C = rho_ice * c_pi * h/2``), which
    turns the diagnostic zero-layer balance into a stably integrable
    prognostic skin:

        C dT_s/dt = F_net_down(atm) + (k_i / h) * (T_base - T_s)

    Signs (surface conventions of the exported ``_sfc_diag`` fluxes):
    ``F_net_down_W_m2 = sw_net_sfc + lw_net_sfc - shflx - lhflx`` — sw/lw net
    positive INTO the surface, turbulent fluxes positive UPWARD out of it, so
    ``F_net_down`` is the net energy gain of the skin from the atmosphere.
    Winter polar night: F_net_down < 0, the skin cools below ``T_freeze_K``
    until conduction from the ocean balances the loss — the equilibrium
    ``T_s = T_base + F_net_down * h / k_i`` of the classic zero-layer model.

    Discretization: backward-Euler in the CONDUCTIVE term (unconditionally
    stable for any ``dt_s``; the atmospheric flux is the lagged explicit
    forcing, refreshed by the physics each step):

        T_new = (T_s + (dt/C)*(F_net + (k_i/h)*T_base)) / (1 + (dt/C)*k_i/h)

    The FULL coupled system (skin + atmosphere recomputing F_net at the new
    skin) is only CONDITIONALLY stable, and the threshold depends on WHEN the
    atmosphere sees the new skin.  If the caller refreshes the surface anchor
    at the SAME cadence as this advance — the MPAS driver re-blends T_sfc every
    model step — the per-step amplification is ``|1 - r*lambda|/(1 + r*g)`` for
    a surface-flux feedback ``lambda = -dF_net/dT_s > 0`` (LW + sensible),
    ``r = dt/C``, ``g = k_i/h``; at the short model step ``dt_s = DT`` this
    keeps ``r*lambda << 1``, stable and monotone for any realistic ``lambda``.
    If instead the anchor is HELD while the skin advances sub-cadence (e.g.
    per-step advance under a once-daily anchor), the effective map reverts to
    the coarse-cadence thresholds (order tens of W/m^2/K, tightening as h
    shrinks) and the melt cap only BOUNDS — does not cure — a cap/floor
    oscillation.  Advance and anchor-refresh cadence MUST match.

    Bounds: melt cap ``T_new <= T_melt_surface_K`` — the FRESH-ICE SURFACE
    melting point (``constants.T_freeze``, i.e. 0 C), NOT the basal seawater
    freezing point ``T_freeze_K`` (``constants.T_freeze_ocean``).  Distinct
    boundary conditions: conduction is toward the warmer seawater-freezing
    base, the surface melts at 0 C (mirrors ``ice/sea_ice.py`` ``T_base`` vs
    ``T_melt_surface``).  A melting skin sheds energy at 0 C — Semtner's melt
    branch, whose meltwater bookkeeping a prescribed-ice run does not carry.
    ``_ICE_SKIN_FLOOR_K`` guards a pathological cold spike.  Open water
    (``sic <= 0``) snaps to ``T_freeze_K`` (seawater freezing) so a cell
    freezing later starts from the freezing point, not a stale skin.

    Limitations. (1) Single-tile: ``F_net_down`` is the BLENDED-cell surface
    flux (computed at ``sic*T_skin + (1-sic)*sst``), used here as the ice-tile
    forcing.  Exact only at ``sic=1``; at marginal ice the true ice-tile
    LW/turbulent flux differs — a documented closure error, most consequential
    during seasonal advance/retreat, second-order because the skin enters the
    anchor weighted by ``sic``.  (2) ``h_ice_m`` is a single global
    climatological thickness (no Arctic~2 m / Antarctic~1 m asymmetry).
    (3) Differentiability: the interior is smooth (``dT_new/dF = r/(1+r*g)``,
    exact), but ``clip`` has zero gradient AT the two caps and ``where(sic>0)``
    zeroes the flux gradient over open water — the physically correct
    dead-gradient of a saturated cap / inactive mask, not a defect.

    Parameters
    ----------
    T_skin : (ncol,) current skin temperature [K].
    F_net_down_W_m2 : (ncol,) net downward atmospheric energy flux [W/m^2].
    sic : (ncol,) sea-ice concentration [0-1].
    dt_s : integration step [s] (the model step DT in the MPAS AMIP loop).
    h_ice_m : climatological slab thickness [m] (tunable; ~2 m Arctic mean).
    T_freeze_K : basal seawater freezing point [K] (conduction base +
        open-water snap).
    k_ice_W_m_K : ice thermal conductivity [W/m/K].
    T_melt_surface_K : fresh-ice surface melting point [K] (upper cap).

    Returns
    -------
    (ncol,) updated skin temperature [K].
    """
    C_areal = 0.5 * constants.rho_ice * constants.c_pi * h_ice_m  # [J/m^2/K]
    g_cond = k_ice_W_m_K / h_ice_m                                # [W/m^2/K]
    r = dt_s / C_areal
    T_new = (T_skin + r * (F_net_down_W_m2 + g_cond * T_freeze_K)) \
        / (1.0 + r * g_cond)
    T_new = jnp.clip(T_new, _ICE_SKIN_FLOOR_K, T_melt_surface_K)
    return jnp.where(sic > 0.0, T_new, T_freeze_K)
