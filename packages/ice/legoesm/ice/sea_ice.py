"""Sea ice model: thermodynamics, dynamics, and multi-category ice.

Modes (controlled by ``SeaIceConfig``):
- **Slab** (``dynamics="none"``, ``n_categories=1``): Original thermodynamic
  slab with diagnostic free-drift velocity. Fully backward compatible.
- **Free drift** (``dynamics="free_drift"``): Heuristic linear-combination
  velocity (not a force-balance solver) with optional tracer advection.
- **EVP** (``dynamics="evp"``): Elastic-Viscous-Plastic rheology with
  subcycled momentum solver (Hunke & Dukowicz 1997).
- **mEVP** (``dynamics="mevp"``): Modified-EVP pseudo-time relaxation
  (Bouillon 2013 / Kimmritz 2015). Converges to the implicit VP
  solution without the EVP elastic CFL constraint.

Multi-category ice (``n_categories > 1``) uses a simplified category
transfer scheme to redistribute ice across thickness bins.  This is
*not* a full Lipscomb (2001) linear remapping.  Snow depth is not
tracked.

Thermodynamics:
    Surface energy balance determines T_ice.
    Conductive flux through ice: F_cond = k_ice * (T_freeze - T_ice) / (h + h_min)
    Growth/melt: dh/dt = (F_cond - F_ocean) / (rho_ice * L_f)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio_ice
from legoesm.core.bulk_flux import (
    simple_bulk_fluxes,
    compute_most_fluxes,
    validate_bulk_scheme,
)
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.ice.dynamics import evp_solver, mevp_solver, free_drift_velocity
from legoesm.ice.transport import advect_ice_tracers
from legoesm.ice.itd import (
    aggregate_state,
    linear_remap,
    lipscomb_2001_remap,
)
from legoesm.ice.snow import (
    accumulate_snowfall,
    combined_conductance,
    consume_from_snow_then_ice,
    consume_sublimation_from_snow_then_ice,
    snow_ice_flooding,
)
from legoesm.ice.brine import update_salinity_and_salt_flux
from legoesm.ice.ridging import apply_ridging
from legoesm.ice.shortwave import compute_ice_sw
from legoesm.ice.ponds import step_ponds
from legoesm.core.surface_energy import surface_radiation_fluxes
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import (
    SeaIceState,
    DynamicSeaIceState,
    dynamic_to_slab,
)
from legoesm.surface_albedo import ice_albedo as compute_ice_albedo


def grid_supports_ice_dynamics(grid) -> bool:
    """True when ``grid`` has implemented sea-ice dynamics/transport ops.

    Sea-ice dynamics (EVP/mEVP via ``strain_rates`` + ``stress_divergence``)
    and tracer transport (``advect_ice_tracers`` via the FV flux-divergence
    operators) are implemented for cubed-sphere, lat-lon, and MPAS/Voronoi
    grids.  Other grid objects — ``GaussianGrid`` (spectral atmosphere),
    ``PlaneGrid``, ``CubedSphereCDGrid`` — have no matching operators and
    would otherwise crash with an ``AttributeError`` deep inside the EVP /
    FV kernels.  Function-scoped imports keep ``legoesm.ice`` importable in
    isolation (no eager ``legoesm.grids`` dependency at module load).

    Public so an external coupler driver can pick a supported dynamics scheme
    (or fall back to ``free_drift``) before calling :func:`step_sea_ice`.
    """
    from legoesm.grids.cubed_sphere import CubedSphereGrid
    from legoesm.grids.latlon import LatLonGrid
    from legoesm.grids.voronoi import VoronoiMesh
    return isinstance(grid, (CubedSphereGrid, LatLonGrid, VoronoiMesh))


def grid_supports_ice_transport(grid) -> bool:
    """True when ``grid`` has implemented sea-ice TRACER-TRANSPORT ops.

    Superset of :func:`grid_supports_ice_dynamics`: every dynamics-capable
    grid can also transport, and the curvilinear lat-lon C-grid
    (``LatLonCGridGeometry`` — the tripole eORCA geometry) additionally
    supports transport via the fold-aware donor-cell C-grid advection
    (``transport.fv_flux_divergence_latlon_cgrid``, built on the core
    ``upwind_cell_to_uface/vface`` + ``divergence_cgrid`` operators) while
    remaining dynamics-INcapable (no curvilinear strain-rate/stress-
    divergence — EVP/mEVP still degrade to ``free_drift`` there).
    """
    if grid_supports_ice_dynamics(grid):
        return True
    from legoesm.grids.latlon import LatLonCGridGeometry
    return isinstance(grid, LatLonCGridGeometry)


# Backward-compatible private alias (internal call sites below + any importer
# predating the public promotion).
_grid_supports_ice_dynamics = grid_supports_ice_dynamics


def _base_spatial_ndim(grid):
    """Spatial rank of a per-cell ice field for ``grid`` (no category axis).

    MPAS Voronoi → 1 ``(nCells,)``; lat-lon / tripole C-grid → 2
    ``(n_lat, n_lon)``; cubed-sphere → 3 ``(6, n, n)``.  Returns ``None``
    for an unknown / None grid so callers fall back to the cubed-sphere
    convention.  Mirrors the per-grid base rank in
    ``transport.advect_ice_tracers``; lets the multi-category trailing axis
    be detected correctly on every grid rather than via the
    cubed-sphere-only ``h.ndim > 3`` heuristic.
    """
    from legoesm.grids.latlon import LatLonCGridGeometry, LatLonGrid
    from legoesm.grids.voronoi import VoronoiMesh
    if isinstance(grid, VoronoiMesh):
        return 1
    if isinstance(grid, (LatLonGrid, LatLonCGridGeometry)):
        return 2
    return 3 if grid is not None else None


def _validate_dynamic_state_shape(h_dyn_shape, config, grid) -> None:
    """Validate a dynamic ice-state thickness shape vs ``config.n_categories``.

    Grid-aware: uses the spatial base rank so the trailing category axis is
    checked correctly on cubed-sphere / lat-lon / MPAS.  Shared by BOTH the
    new-physics (``_step_dynamic_v2``) and legacy (``_step_dynamic``) paths
    so ``config.n_categories`` and the state array can never silently
    disagree (multi-category branching keys off ``config.n_categories``).
    When ``grid`` is None the spatial rank is unknown, so only the trailing
    category-axis size is validated (multi-cat) / the cubed-sphere heuristic
    is used (single-cat).
    """
    base = _base_spatial_ndim(grid)
    # When the grid is unknown (grid=None, thermo-only), default the spatial
    # base rank to the cubed-sphere convention (3) rather than skipping the
    # rank check — otherwise a stale single-category cubed-sphere state
    # (6, n, n) would pass as multi-category whenever n_categories == n,
    # and _step_dynamic_v2 would treat a spatial axis as the category axis
    # (codex finding).
    base_eff = base if base is not None else 3
    if config.n_categories > 1:
        rank_ok = len(h_dyn_shape) == base_eff + 1
        if h_dyn_shape[-1] != config.n_categories or not rank_ok:
            raise ValueError(
                f"step_sea_ice: dynamic ice-state h_ice has shape "
                f"{h_dyn_shape}, inconsistent with config.n_categories="
                f"{config.n_categories} on a grid with spatial base rank "
                f"{base}.  Rebuild via init_dynamic_ice_state(shape, "
                "n_categories=...) and distribute_to_categories."
            )
    else:
        has_extra_axis = len(h_dyn_shape) > base_eff
        if has_extra_axis:
            raise ValueError(
                f"step_sea_ice: dynamic ice-state h_ice has shape "
                f"{h_dyn_shape}, which carries a trailing category axis, "
                "but config.n_categories=1.  Drop the trailing axis or "
                "raise n_categories to match."
            )


def _uses_new_physics(config: SeaIceConfig) -> bool:
    """True when any Tier-1/Tier-2 new physics gate is enabled."""
    return (
        config.snow.enabled
        or config.brine.enabled
        or config.ridging.enabled
        or config.ponds.enabled
        or config.shortwave_scheme != "constant"
        # Constant-scheme SW transmittance is a v2-path feature (the legacy
        # _step_dynamic surface EB has no penetration channel); route to v2 so
        # a nonzero setting is never a silent no-op.
        or config.sw_transmittance_const > 0.0
        or config.itd_remap != "simple"
    )


# ==============================================================================
# Main entry point
# ==============================================================================

def step_sea_ice(
    state,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
    grid=None,
):
    """Step the sea ice model forward by dt seconds.

    Dispatches to slab or dynamic model based on config.

    Parameters
    ----------
    state : SeaIceState or DynamicSeaIceState
        Current ice state.
    forcing : AtmToSurface
        Atmospheric forcing fields.
    ocean_sst : jnp.ndarray
        Ocean SST [K], shape (6, n, n).
    ocean_u, ocean_v : jnp.ndarray
        Ocean surface currents [m/s], shape (6, n, n).
    config : SeaIceConfig
        Sea ice model parameters.
    U_min : float
        Minimum wind speed floor [m/s].
    dt : float
        Time step [s].
    grid : CubedSphereGrid, optional
        Required when ``dynamics != "none"`` or ``transport != "none"``.

    Returns
    -------
    new_state : SeaIceState or DynamicSeaIceState
    response : TileResponse
    """
    # Validate: dynamics literal + grid requirement
    if config.dynamics not in ("none", "free_drift", "evp", "mevp"):
        raise ValueError(
            f"Unknown sea-ice dynamics scheme: {config.dynamics!r}. "
            "Expected one of: 'none', 'free_drift', 'evp', 'mevp'."
        )
    if config.dynamics in ("evp", "mevp") and grid is None:
        raise ValueError(
            f"dynamics={config.dynamics!r} requires a grid argument. "
            "Pass grid=<CubedSphereGrid> to step_sea_ice()."
        )
    if config.transport not in ("none", "advect"):
        raise ValueError(
            f"Unknown sea-ice transport scheme: {config.transport!r}. "
            "Expected one of: 'none', 'advect'."
        )
    if config.transport == "advect" and grid is None:
        raise ValueError(
            "transport='advect' requires a grid argument. "
            "Pass grid=<CubedSphereGrid> to step_sea_ice()."
        )

    # Grid-TYPE guards: dynamics (EVP/mEVP) + ridging call grid-specific
    # STRAIN-RATE operators (cubed-sphere / lat-lon / MPAS only); tracer
    # transport calls FLUX-DIVERGENCE operators, which additionally exist on
    # the tripole/curvilinear C-grid (grid_supports_ice_transport).  Reject an
    # unsupported non-None grid up front with a clear message rather than
    # letting it fall through to the cubed-sphere branch and raise an opaque
    # AttributeError deep in the EVP loop (e.g. Gaussian spectral / Plane).
    if grid is not None:
        _needs_strain_ops = (
            config.dynamics in ("evp", "mevp") or config.ridging.enabled
        )
        if _needs_strain_ops and not _grid_supports_ice_dynamics(grid):
            raise ValueError(
                "Sea-ice dynamics/ridging require a grid with implemented "
                "strain-rate operators (CubedSphereGrid, LatLonGrid, or "
                f"VoronoiMesh); got {type(grid).__name__}.  Use "
                "dynamics='none'/'free_drift' and ridging.enabled=False on "
                "this grid."
            )
        if config.transport == "advect" and not grid_supports_ice_transport(grid):
            raise ValueError(
                "Sea-ice tracer transport requires a grid with implemented "
                "flux-divergence operators (CubedSphereGrid, LatLonGrid, "
                "VoronoiMesh, or the tripole LatLonCGridGeometry); got "
                f"{type(grid).__name__}.  Use transport='none' for "
                "thermodynamics-only on this grid."
            )
    if config.ridging.enabled and grid is None:
        raise ValueError(
            "ridging.enabled=True requires a grid argument: mechanical "
            "ridging needs the convergence (strain-rate) operator.  Pass a "
            "supported grid, or set ridging.enabled=False."
        )

    # F10 guard: lat-lon EVP/mEVP strain-rate + stress-divergence use the
    # spherical metric tanθ/r, singular at the geographic pole (cosθ → 0).
    # Cell-centered lat-lon grids place rows strictly inside the poles, so the
    # metric is finite; reject a grid whose rows actually reach ±90° (e.g. a
    # hand-built node grid) with a clear message instead of emitting inf/NaN
    # forces deep in the EVP loop.  Pole-reaching polar-cap dynamics should use
    # the cubed-sphere / MPAS backend.
    #
    # The check reads ``grid.lat`` VALUES, so it runs only when they are
    # concrete (eager call / grid construction).  It uses host NumPy so no
    # tracer is created; under ``jax.jit`` (grid arrays traced, whether passed
    # as an argument or lifted from a closure into the trace) the host
    # conversion raises and we skip the eager value check — the grid has
    # already been validated on its first eager use, and ``create_latlon_grid``
    # never produces pole-reaching rows.
    if config.dynamics in ("evp", "mevp") and grid is not None:
        import numpy as _np
        from legoesm.grids.latlon import LatLonGrid
        if isinstance(grid, LatLonGrid):
            try:
                lat_host = _np.asarray(grid.lat)
            except jax.errors.TracerArrayConversionError:
                lat_host = None  # tracing under jit; skip the eager guard
            if (lat_host is not None
                    and float(_np.min(_np.abs(_np.cos(lat_host)))) < 1e-6):
                raise ValueError(
                    "Sea-ice dynamics={!r} on a LatLonGrid uses the spherical "
                    "metric tanθ/r, singular at the geographic pole (cosθ → 0); "
                    "this grid has a row at/within ~1e-6 of ±90°.  Use a "
                    "cell-centered lat-lon grid (create_latlon_grid places rows "
                    "strictly inside the poles) or the cubed-sphere / MPAS "
                    "backend for polar-cap dynamics.".format(config.dynamics)
                )

    # Multi-category tracer conservation: the 'simple' (linear) ITD remap
    # transfers only h / concentration / temperature across category bins, NOT
    # the snow / bulk-salinity / pond tracers, so moving ice volume between
    # categories that hold different tracer values would break salt / snow /
    # pond conservation (the per-category tracers would sit on a stale bin
    # structure).  Require the tracer-aware Lipscomb (2001) remap whenever a
    # tracer is active in multi-category mode.
    if config.n_categories > 1:
        # Validate the ITD-remap scheme up front (no silent fallback): the
        # dispatch routes any value other than 'lipscomb2001' to the legacy
        # linear remap, so an unrecognised string must NOT slip through.
        if config.itd_remap not in ("simple", "lipscomb2001"):
            raise ValueError(
                f"Unknown config.itd_remap={config.itd_remap!r}; expected "
                "'simple' or 'lipscomb2001'."
            )
        # Tracer conservation: only 'lipscomb2001' carries the snow /
        # bulk-salinity / pond tracers across category bins.  Require it (NOT
        # merely reject 'simple') so a typo cannot route to the legacy linear
        # remap, which transfers only h / concentration / temperature and would
        # violate salt / snow / pond conservation on category transfers.
        if (config.itd_remap != "lipscomb2001"
                and (config.brine.enabled or config.snow.enabled
                     or config.ponds.enabled)):
            raise ValueError(
                "Multi-category sea ice with brine/snow/pond tracers requires "
                "config.itd_remap='lipscomb2001': the 'simple' linear ITD "
                "remap carries only h / concentration / temperature across "
                "category bins, not the tracer fields (bulk salinity, snow, "
                "ponds), so category transfers would violate salt / snow / "
                "pond conservation.  Set config.itd_remap='lipscomb2001', or "
                "use n_categories=1."
            )

    if config.dynamics == "none" and config.n_categories == 1 and not _uses_new_physics(config):
        # Original slab path — fully backward compatible
        if isinstance(state, DynamicSeaIceState):
            state = dynamic_to_slab(state)
        return _step_slab(state, forcing, ocean_sst, ocean_u, ocean_v,
                          config, U_min, dt)
    elif _uses_new_physics(config):
        # Extended physics path: snow / brine / ridging / ponds /
        # delta-Eddington / Lipscomb 2001 remap.  Requires
        # DynamicSeaIceState carrier so the new state fields are
        # available.  Slab inputs are lifted; multi-category configs
        # must arrive with a state that already carries the category
        # axis — silently expanding a 3-D slab to 4-D would lose
        # the per-category distribution.
        if isinstance(state, SeaIceState):
            if config.n_categories > 1:
                raise ValueError(
                    f"step_sea_ice: new-physics path with n_categories="
                    f"{config.n_categories} requires a DynamicSeaIceState "
                    "with shape (..., n_categories).  Build one via "
                    "init_dynamic_ice_state(shape, n_categories=...) and "
                    "distribute_to_categories before calling step_sea_ice."
                )
            from legoesm.ice.state import slab_to_dynamic
            state = slab_to_dynamic(state)
        # Validate dynamic-state shape against the configured number of
        # categories (grid-aware) so a stale state cannot silently run a
        # multi-category config (skipping ITD remap / ridging) or vice versa.
        _validate_dynamic_state_shape(state.h_ice.data.shape, config, grid)
        return _step_dynamic_v2(state, forcing, ocean_sst, ocean_u, ocean_v,
                                config, U_min, dt, grid)
    else:
        # Legacy dynamic path (free_drift / EVP / mEVP without new physics):
        # apply the same grid-aware shape validation as the new-physics path
        # so config.n_categories and the state array cannot disagree on any
        # grid.
        if isinstance(state, DynamicSeaIceState):
            _validate_dynamic_state_shape(state.h_ice.data.shape, config, grid)
        return _step_dynamic(state, forcing, ocean_sst, ocean_u, ocean_v,
                             config, U_min, dt, grid)


# ==============================================================================
# Shared bulk-flux dispatch
# ==============================================================================

def _bulk_flux_dispatch(
    T_ice: jnp.ndarray,
    forcing: AtmToSurface,
    config: SeaIceConfig,
    U_min: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(tau_x, tau_y, shflx, lhflx)`` from configured scheme.

    The slab path duplicates this dispatch inline; the multi-category
    thermodynamics path used to skip it and silently use
    ``simple_bulk_fluxes`` regardless of ``config.bulk_scheme``.  This
    helper centralises the choice so both paths and the diagnostic
    ``_build_response`` produce consistent values.
    """
    valid_schemes = ("constant", "most", "coare3", "large_yeager")
    if config.bulk_scheme not in valid_schemes:
        raise ValueError(
            f"Unknown sea-ice bulk_scheme {config.bulk_scheme!r}; "
            f"expected one of {valid_schemes}."
        )
    wind_speed = jnp.sqrt(
        jnp.maximum(
            forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2, 1e-12
        )
    )
    rho = forcing.rho_lowest
    q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)
    validate_bulk_scheme(config.bulk_scheme)
    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        tau_x, tau_y, shflx, lhflx, _ = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_ice, q_sfc, rho,
            z_ref=config.z_ref,
            z0_init=config.z0_ice,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
            L_latent=constants.L_s,
            stability_scheme=config.stability_scheme,
        )
    else:
        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_ice, q_sfc, rho, wind_speed,
            config.Cd_ice, config.Ch_ice,
            L_latent=constants.L_s,
        )
    return tau_x, tau_y, shflx, lhflx


# ==============================================================================
# Slab thermodynamics (original implementation)
# ==============================================================================

def _step_slab(
    state: SeaIceState,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
) -> tuple[SeaIceState, TileResponse]:
    """Original slab sea ice model (backward compatible)."""
    h = state.h_ice.data
    T_ice = state.T_ice.data
    conc = state.concentration.data

    # ---------- Surface fluxes (configured scheme via shared dispatch) ----------
    tau_x, tau_y, shflx, lhflx = _bulk_flux_dispatch(T_ice, forcing, config, U_min)

    # ---------- Thermodynamics (delegate to shared routine) ----------
    h_new, T_ice_new, conc_new, diag = _thermo_single(
        h, T_ice, conc, forcing, ocean_sst, config, U_min, dt,
        shflx=shflx, lhflx=lhflx, return_diagnostics=True,
    )

    # ---------- Ice velocity (free drift, diagnostic) ----------
    # Use the shared Zubov-style drag balance instead of the dimensionally
    # broken slab formula (iter-86): the previous inline expression
    # ``drag_ocean*U_w + (drag_atm*rho_air/rho_ice)*U_a`` produced ice
    # drift two orders of magnitude smaller than the physical Nansen /
    # Zubov estimate.
    u_ice, v_ice = free_drift_velocity(
        ocean_u, ocean_v,
        forcing.u_lowest, forcing.v_lowest,
        drag_ocean=config.drag_ocean,
        drag_atm=config.drag_atm,
        rho_air=config.rho_air_ref,
        rho_ocean=config.rho_ocean_ref,
    )

    new_state = SeaIceState(
        h_ice=state.h_ice.replace(data=h_new),
        T_ice=state.T_ice.replace(data=T_ice_new),
        concentration=state.concentration.replace(data=conc_new),
    )

    # ---------- Build response ----------
    # ``jnp.full`` is one ``Broadcast`` HLO op vs the
    # ``broadcast_to(jnp.array(scalar), ...)`` form.  Same micro-fix
    # as the loop-18/20 lake / coupler / land tile responses.
    _h_dtype = h.dtype
    if config.temp_dependent_albedo:
        alpha_ice = compute_ice_albedo(T_ice_new, config.ice_albedo)
    else:
        alpha_ice = jnp.full(h.shape, config.albedo_ice, dtype=_h_dtype)

    # Direct ``ε σ T⁴ + (1-ε)·lw_down`` instead of the full
    # ``surface_radiation_fluxes`` call (which discards ``sw_net``
    # / ``lw_net``) — same direct-expression rewrite as the
    # ``ocean_tile_response`` and ``two_layer_lake`` fixes.
    lw_up_new = (
        config.emissivity_ice * constants.sigma_sb * T_ice_new ** 4
        + (1.0 - config.emissivity_ice) * forcing.lw_down
    )
    q_sfc_new = saturation_mixing_ratio_ice(T_ice_new, forcing.p_surface)

    # Sea-ice -> ocean freshwater and heat exchange (F2 + F11).
    #
    # ``_thermo_single`` returns per-PROCESS, PER-GRID-CELL volume rates
    # [m_ice/m2_cell/s]: existing-ice melt/growth/sublimation already weighted
    # by the ice fraction ``conc``, lead freeze by the open-water lead area.
    # We assemble the per-grid-cell exchange budget from those explicit terms
    # instead of the old per-ICE-area ``(h_new - h)/dt`` rate, which (a)
    # silently omitted lead-freeze exchange on partial-cover cells and (b) had
    # to back out sublimation from a lumped thickness change.  Per-process
    # terms make melt (-> freshwater), basal congelation and lead freeze
    # (extract ocean water), and the atmosphere-only sublimation explicit and
    # individually testable.
    #
    # These FW/heat fluxes are returned PER-GRID-CELL (per-water-area).  The
    # coupler's ``blend_tiles`` weights ice->ocean exchange terms by the WATER
    # fraction ``f_water`` (not ``f_ice``), so the per-cell melt/freeze budget
    # is delivered intact for any concentration history — melt retreat,
    # terminal melt-out, new-ice formation — with no 1/conc normalisation that
    # could dilute the flux or blow up as a cell melts out (F11).
    #
    # SCOPE: this fixes the FLUX bookkeeping (what crosses the ice-ocean
    # interface).  It does NOT make the prognostic slab inventory strictly
    # ``V = h*conc``-conserving: ``_thermo_single`` still evolves ``h`` and
    # ``conc`` semi-independently (lateral melt-retreat and lead-area growth are
    # heuristics, not a CICE V=h*A state update).  That state-variable rework
    # is tracked separately; the ocean FW here intentionally reflects the
    # energy-based melt/growth/lead-freeze mass, not the lateral area
    # redistribution (which would double-count the same dh_dt-driven melt).

    # Positive = freshwater INTO ocean (melt > freeze); negative = water
    # extracted to build ice.  Sublimation excluded (atmosphere via lhflx).
    freshwater_to_ocean = config.rho_ice * (
        diag["vmelt_ice"] - diag["vgrowth_basal"] - diag["vlead_freeze"]
    )

    # Heat extracted from the ocean (positive = ocean loses energy to ice):
    #   1) basal turbulent heat over the ice area; the ocean loses it whenever
    #      SST > T_freeze_ocean under ice.  Uses ``ocean_heat_basal_per_ice_area``
    #      = F_ocean minus the latent of any UNREALIZED basal melt: when the
    #      over-ablation cap fires the ocean cannot melt ice that is gone, so the
    #      unrealized latent stays in the ocean and is not extracted (#28).
    #   2) lead-freeze latent heat ``L_f*rho_ice`` per m^3 of new lead ice;
    #      ``vlead_freeze`` is per-cell so this also charges partial-cover cells
    #      that the old ``~ice_mask`` gate silently skipped.
    # Basal-growth latent heat is released UPWARD through the ice (conduction),
    # not drawn from the ocean, so it is intentionally absent.  Audit F8.
    ocean_heat_extraction = (
        diag["ocean_heat_basal_per_ice_area"] * conc
        + config.rho_ice * config.L_f * diag["vlead_freeze"]
        # Surplus surface-melt heat WARMS the ocean (energy closure, finding
        # #6): negative contribution to extraction (+sign = ocean loses heat).
        - diag["surface_melt_ocean_gain_per_ice_area"] * conc
    )

    # Sea-ice → ocean back-reaction stress (Newton's third law).
    # The ocean→ice drag tau_oi accelerates the ice; the ice exerts
    # −tau_oi on the ocean column.  Compute the proper Cauchy drag using the
    # ocean-ice drag coefficient (config.drag_ocean = C_oi) and the relative
    # velocity.  Audit F9 / F11.
    #
    # Returned PER-ICE-TILE (no ``* conc`` here): the ice-ocean drag is a
    # continuous force proportional to the instantaneous ice area, so the
    # coupler's ``blend_tiles`` applies the single area weight ``f_ice =
    # f_water*conc``.  The previous ``* conc`` double-counted the ice fraction
    # (``blend_tiles`` multiplies by ``f_ice`` again).  Ice-free cells get
    # ``f_ice = 0`` in the blend, so they contribute no stress automatically.
    du_oi = ocean_u - u_ice
    dv_oi = ocean_v - v_ice
    speed_oi = jnp.sqrt(du_oi ** 2 + dv_oi ** 2 + 1e-10)
    tau_oi_x = config.rho_ocean_ref * config.drag_ocean * speed_oi * du_oi
    tau_oi_y = config.rho_ocean_ref * config.drag_ocean * speed_oi * dv_oi
    ocean_stress_x = -tau_oi_x
    ocean_stress_y = -tau_oi_y

    response = TileResponse(
        T_sfc=T_ice_new,
        albedo=alpha_ice,
        emissivity=jnp.full(h.shape, config.emissivity_ice, dtype=_h_dtype),
        z0=jnp.full(h.shape, config.z0_ice, dtype=_h_dtype),
        q_surface=q_sfc_new,
        shflx=shflx,
        # Realized latent (== L_s * surface_mass_flux / conc): the atmosphere
        # latent energy matches the skin solve and the moisture mass (#28).
        lhflx=diag["lhflx_realized"],
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up_new,
        u_ocean_sfc=u_ice,
        v_ocean_sfc=v_ice,
        co2_flux=jnp.zeros_like(h),
        freshwater_flux=freshwater_to_ocean,
        ocean_heat_extraction=ocean_heat_extraction,
        ocean_stress_x=ocean_stress_x,
        ocean_stress_y=ocean_stress_y,
        # Sea-ice surface moisture exchange is sublimation/deposition (L_s).
        # Use the REALIZED ice->atmosphere mass (capped sublimation minus
        # deposition), returned PER-GRID-CELL (weighted by the input ice
        # fraction ``conc``) so ``blend_tiles`` can weight it by ``f_water`` and
        # the atmosphere receives the full pulse even at terminal melt-out.
        # Equals lhflx/L_s * conc in the no-clamp regime; bounded to the water
        # the ice actually lost when the over-ablation cap fires (#28, codex).
        surface_mass_flux=diag["sublim_mass_per_ice_area"] * conc,
        # Slab path has no bulk-ice salinity tracking — salt channel
        # is inert.  The brine-aware path uses the dynamic state and
        # populates ``salt_flux`` in ``_build_response``.
        salt_flux=jnp.zeros_like(h),
    )

    return new_state, response


# ==============================================================================
# Dynamic sea ice (EVP + optional multi-category)
# ==============================================================================

def _step_dynamic(
    state,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
    grid=None,
):
    """Dynamic sea ice step: dynamics → transport → thermodynamics → ITD remap.

    Parameters
    ----------
    state : DynamicSeaIceState
    grid : CubedSphereGrid, required for dynamics/transport
    """
    h = state.h_ice.data
    T_ice = state.T_ice.data
    conc = state.concentration.data
    u_ice = state.u_ice.data
    v_ice = state.v_ice.data
    s11 = state.sigma_11.data
    s22 = state.sigma_22.data
    s12 = state.sigma_12.data

    # Cap an already-overfilled multicat state (e.g. a restart) to
    # sum_k a_k <= 1 BEFORE the dynamics aggregate, so EVP/mEVP rheology — whose
    # ice strength is exponential in concentration — does not run on > 1 cell
    # area for a step.  No-op when sum <= 1.  (Transport and the ITD remap can
    # re-introduce overfill; they are re-capped downstream.)  Codex.
    if config.n_categories > 1:
        conc, h, _, _ = _cap_multicat_concentration(conc, h)

    # For multi-category: aggregate for coupler response and dynamics
    if config.n_categories > 1:
        h_agg, T_agg, conc_agg = aggregate_state(h, T_ice, conc)
    else:
        h_agg, T_agg, conc_agg = h, T_ice, conc

    # ---- 1. Dynamics ----
    if config.dynamics == "evp" and grid is not None:
        u_ice, v_ice, s11, s22, s12 = evp_solver(
            u_ice, v_ice, s11, s22, s12,
            h_agg, conc_agg,
            forcing.u_lowest, forcing.v_lowest,
            ocean_u, ocean_v,
            grid, dt,
            N_evp=config.N_evp,
            e_yield=config.e_yield,
            P_star=config.P_star,
            C_strength=config.C_strength,
            T_evp=config.T_evp,
            Delta_min=config.Delta_min,
            rho_ice=config.rho_ice,
            rho_air=config.rho_air_ref,
            rho_ocean=config.rho_ocean_ref,
            C_ai=config.drag_atm,
            C_oi=config.drag_ocean,
            h_ice_min=config.h_ice_min,
            differentiable=config.differentiable_dynamics,
        )
    elif config.dynamics == "mevp" and grid is not None:
        u_ice, v_ice, s11, s22, s12 = mevp_solver(
            u_ice, v_ice, s11, s22, s12,
            h_agg, conc_agg,
            forcing.u_lowest, forcing.v_lowest,
            ocean_u, ocean_v,
            grid, dt,
            N_mevp=config.N_mevp,
            e_yield=config.e_yield,
            P_star=config.P_star,
            C_strength=config.C_strength,
            alpha_mevp=config.alpha_mevp,
            beta_mevp=config.beta_mevp,
            Delta_min=config.Delta_min,
            rho_ice=config.rho_ice,
            rho_air=config.rho_air_ref,
            rho_ocean=config.rho_ocean_ref,
            C_ai=config.drag_atm,
            C_oi=config.drag_ocean,
            h_ice_min=config.h_ice_min,
            differentiable=config.differentiable_dynamics,
        )
    elif config.dynamics == "free_drift":
        u_ice, v_ice = free_drift_velocity(
            ocean_u, ocean_v,
            forcing.u_lowest, forcing.v_lowest,
            drag_ocean=config.drag_ocean,
            drag_atm=config.drag_atm,
            rho_air=config.rho_air_ref,
            rho_ocean=config.rho_ocean_ref,
        )

    # ---- 2. Transport ----
    # ``advect_ice_tracers`` handles both 2D ``(6, n, n)`` and multi-category
    # 3D ``(6, n, n, n_cat)`` shapes natively, with a single 4D vector halo
    # exchange across all quantities and categories — the prior vmap-over-
    # categories pattern issued ``n_cat × 3`` halo exchanges per timestep.
    if config.transport == "advect" and grid is not None:
        h, conc, T_ice = advect_ice_tracers(
            h, conc, T_ice, u_ice, v_ice, grid, dt,
            T_max=config.T_melt_surface,
            n_subcycles=config.transport_subcycles,
        )

    # Snapshot of aggregated ice thickness AFTER transport but BEFORE
    # thermodynamics.  This is the reference for the ice → ocean
    # freshwater / heat exchange — only the thermodynamic ΔV is an
    # ocean exchange; horizontal transport conserves ice mass and
    # should not show up as melt/freezing in the coupler response.
    # Codex iter-3 finding #2.
    if config.n_categories > 1:
        h_agg_post_transport, _, conc_agg_post_transport = aggregate_state(
            h, T_ice, conc)
    else:
        h_agg_post_transport = h
        conc_agg_post_transport = conc

    # ---- 3. Thermodynamics (per category or single) ----
    if config.n_categories > 1:
        # Multi-category: apply thermodynamics per category via vmap.
        # ``jax.vmap`` accepts negative ``in_axes`` / ``out_axes`` and
        # vmaps over the trailing category axis directly — skipping the
        # six ``jnp.moveaxis`` round-trips used by the prior pattern.
        # JAX still produces one batched kernel for ``_thermo_single``,
        # so the savings are layout/intermediate eliminations rather
        # than fewer kernel launches; the diff is one less buffer copy
        # per category-vmap invocation under XLA fusion.
        n_cat = h.shape[-1]
        # Restore sum_k a_k <= 1 BEFORE recording the remap reference and the
        # per-process exchange area, so both the diag and the budgets use the
        # post-compaction (sum<=1) concentration.  A second cap follows the
        # linear remap (which can re-split a compacted bin and reintroduce
        # sum>1).  Applied UNCONDITIONALLY (the cap is a no-op when sum<=1) so
        # an already-overfilled restart / external state is handled even
        # without transport.  Not bin-accurate for large overfill (deferred).
        # Codex.
        conc, h, _, _ = _cap_multicat_concentration(conc, h)
        h_old = h
        conc_old = conc

        # Open-water fraction at the aggregate level — the total lead
        # area available for refreezing this step.  Per-category
        # ``(1 - conc_k)`` sums to ``n_cat - sum_k conc_k`` and badly
        # over-counts when more than one category is occupied.  We pass
        # this aggregated value into ``_thermo_single`` so the lead
        # freeze is driven by the true lead area.  Codex iter-3 #3.
        open_water_agg = jnp.clip(1.0 - jnp.sum(conc_old, axis=-1), 0.0, 1.0)

        # Compute bulk fluxes per category with the configured scheme
        # BEFORE thermodynamics, so the dynamic multi-category path uses
        # the same bulk-flux closure that the slab path and diagnostic
        # ``_build_response`` use.  Previously ``_thermo_single`` was
        # called without ``shflx``/``lhflx``, silently falling back to
        # ``simple_bulk_fluxes`` regardless of ``config.bulk_scheme`` —
        # state and diagnostics could disagree.  (Codex finding #7.)
        # ``enable_lead_freeze`` is per-category: True only for cat 0
        # so the open-water → new-ice deposit happens once, into the
        # thinnest bin.  Cats 1+ still do their own basal / surface
        # melt-growth from F_cond / F_ocean / sublim.
        def _thermo_cat(h_k, T_k, conc_k, enable_lead_freeze_k):
            _, _, shflx_k, lhflx_k = _bulk_flux_dispatch(
                T_k, forcing, config, U_min,
            )
            return _thermo_single(
                h_k, T_k, conc_k,
                forcing, ocean_sst, config, U_min, dt,
                shflx=shflx_k, lhflx=lhflx_k,
                open_water_fraction=open_water_agg,
                enable_lead_freeze=bool(enable_lead_freeze_k),
                return_diagnostics=True,
            )

        # cat-0 deposits the lead freeze; cats 1+ skip it.  Implemented
        # by building separate forward calls (cat 0 vs the others) so
        # the static bool can flow through ``_thermo_single``.
        def _thermo_cat_with_lead(args):
            h_k, T_k, c_k = args
            return _thermo_cat(h_k, T_k, c_k, True)

        def _thermo_cat_no_lead(args):
            h_k, T_k, c_k = args
            return _thermo_cat(h_k, T_k, c_k, False)

        # Split: process cat 0 with lead-freeze, then vmap cats 1+
        # without.  ``jax.lax.map``-style splits keep the static bool
        # as a Python-level constant inside each branch.  ``_thermo_single``
        # returns per-process diagnostics (return_diagnostics=True); we build
        # the per-cell ice->ocean exchange from them (F11).
        h_0, T_0, c_0, diag_0 = _thermo_cat_with_lead(
            (h[..., 0], T_ice[..., 0], conc[..., 0]),
        )
        # Per-cell freshwater / heat / atmosphere-mass from cat 0 (input conc =
        # conc_old[..., 0]).
        fw_exch, heat_exch, subl_exch = _ocean_exchange_from_diag(
            diag_0, conc_old[..., 0], config)
        # Per-ice-area realized latent flux, aggregated conc-weighted over cats
        # (the atmosphere sees the aggregate ice surface): accumulate the
        # numerator sum_k(lhflx_realized_k * conc_k) and the conc denominator,
        # normalise after all cats are summed.  Matches the realized sublimation
        # mass + the skin solve so the atmosphere latent energy closes (#28).
        lhflx_num = diag_0["lhflx_realized"] * conc_old[..., 0]
        conc_sum_lh = conc_old[..., 0]
        if n_cat > 1:
            h_rest, T_rest, c_rest, diag_rest = jax.vmap(
                _thermo_cat_no_lead, in_axes=0, out_axes=0,
            )((
                jnp.moveaxis(h[..., 1:], -1, 0),
                jnp.moveaxis(T_ice[..., 1:], -1, 0),
                jnp.moveaxis(conc[..., 1:], -1, 0),
            ))
            h = jnp.concatenate(
                [h_0[..., None], jnp.moveaxis(h_rest, 0, -1)], axis=-1,
            )
            T_ice = jnp.concatenate(
                [T_0[..., None], jnp.moveaxis(T_rest, 0, -1)], axis=-1,
            )
            conc = jnp.concatenate(
                [c_0[..., None], jnp.moveaxis(c_rest, 0, -1)], axis=-1,
            )
            # Sum the per-cat per-cell exchange over cats 1+ (vmap stacked on
            # axis 0; input conc = conc_old[..., 1:] in the same layout).
            fw_rest, heat_rest, subl_rest = _ocean_exchange_from_diag(
                diag_rest, jnp.moveaxis(conc_old[..., 1:], -1, 0), config)
            fw_exch = fw_exch + jnp.sum(fw_rest, axis=0)
            heat_exch = heat_exch + jnp.sum(heat_rest, axis=0)
            subl_exch = subl_exch + jnp.sum(subl_rest, axis=0)
            lhflx_num = lhflx_num + jnp.sum(
                diag_rest["lhflx_realized"]
                * jnp.moveaxis(conc_old[..., 1:], -1, 0),
                axis=0,
            )
            conc_sum_lh = conc_sum_lh + jnp.sum(conc_old[..., 1:], axis=-1)
        else:
            h = h_0[..., None]
            T_ice = T_0[..., None]
            conc = c_0[..., None]
        # Conc-weighted aggregate per-ice-area realized latent for the response.
        lhflx_exch = lhflx_num / jnp.maximum(conc_sum_lh, 1e-30)

        # Open-water ice growth should only be deposited into category 0
        # (thinnest). Zero out new-ice growth in empty higher categories
        # to prevent spurious ice creation in all empty categories.
        was_empty = h_old <= 0.0  # (..., n_cat) True where category had no ice
        cat_mask = jnp.arange(n_cat) > 0  # False for cat 0, True for cats 1+
        suppress = was_empty & cat_mask  # suppress growth in empty non-zero cats
        h = jnp.where(suppress, h_old, h)
        conc = jnp.where(suppress, conc_old, conc)

        # ---- 4. ITD remap (including temperature for enthalpy conservation) ----
        h, conc, T_ice = linear_remap(
            h_old, conc_old, h, conc, n_cat, T_new=T_ice,
            T_max=config.T_melt_surface,
        )
        # The incremental linear remap clips each target bin independently and
        # can re-split a compacted category across bins, reintroducing
        # sum_k a_k > 1; re-cap (unconditionally; no-op when sum<=1) so the
        # returned state respects the aggregate-area invariant
        # (volume-conserving).  Codex.
        conc, h, _, _ = _cap_multicat_concentration(conc, h)
    else:
        # Single-category dynamic-path thermo.  Use the same bulk-flux
        # dispatch (MOST / COARE / Large-Yeager / simple_bulk) as the
        # slab path + multi-cat path + diagnostic ``_build_response``
        # so state and diagnostics use the same closure.  Earlier this
        # branch fell through to ``simple_bulk_fluxes`` regardless of
        # ``config.bulk_scheme``.  Codex iter-39 #1.
        _, _, shflx_sc, lhflx_sc = _bulk_flux_dispatch(
            T_ice, forcing, config, U_min,
        )
        conc_in_sc = conc  # input ice fraction (pre-thermo) for the exchange
        h, T_ice, conc, diag_sc = _thermo_single(
            h, T_ice, conc, forcing, ocean_sst, config, U_min, dt,
            shflx=shflx_sc, lhflx=lhflx_sc, return_diagnostics=True,
        )
        # Per-cell ice->ocean freshwater / heat / atmosphere-mass from the
        # per-process diagnostics (F11) — same decomposition as the slab path.
        fw_exch, heat_exch, subl_exch = _ocean_exchange_from_diag(
            diag_sc, conc_in_sc, config)
        # Per-ice-area realized latent (single cat: conc cancels) (#28).
        lhflx_exch = diag_sc["lhflx_realized"]

    # Re-aggregate for coupler response
    if config.n_categories > 1:
        h_agg, T_agg, conc_agg = aggregate_state(h, T_ice, conc)
    else:
        h_agg, T_agg, conc_agg = h, T_ice, conc

    # ---- Build response ----
    # Use the post-transport snapshot as the reference for FW / heat
    # so transport (which conserves ice mass globally) doesn't appear
    # as melt/freezing in the ocean coupler response.
    response = _build_response(
        h_agg, T_agg, conc_agg, u_ice, v_ice,
        forcing, config, U_min,
        h_old=h_agg_post_transport,
        conc_old=conc_agg_post_transport,
        ocean_sst=ocean_sst,
        ocean_u=ocean_u,
        ocean_v=ocean_v,
        dt=dt,
        # Per-process per-cell ice->ocean exchange from _thermo_single's
        # diagnostics (F11) — separates basal growth from lead freeze, which
        # the aggregate (h - h_old)/dt fallback cannot.
        freshwater_override=fw_exch,
        ocean_heat_override=heat_exch,
        # Realized (capped) per-cell ice->atmosphere sublimation mass (#28).
        surface_mass_override=subl_exch,
        # Realized per-ice-area latent flux, conc-weighted across cats (#28).
        lhflx_override=lhflx_exch,
    )

    # Legacy path: snow / brine / pond fields are pass-through (zero-
    # initialised in the input state when ``_uses_new_physics`` is False).
    new_state = DynamicSeaIceState(
        h_ice=state.h_ice.replace(data=h),
        T_ice=state.T_ice.replace(data=T_ice),
        concentration=state.concentration.replace(data=conc),
        u_ice=state.u_ice.replace(data=u_ice),
        v_ice=state.v_ice.replace(data=v_ice),
        sigma_11=state.sigma_11.replace(data=s11),
        sigma_22=state.sigma_22.replace(data=s22),
        sigma_12=state.sigma_12.replace(data=s12),
        h_snow=state.h_snow,
        S_ice=state.S_ice,
        pond_area=state.pond_area,
        pond_depth=state.pond_depth,
    )

    return new_state, response


# ==============================================================================
# Thermodynamics for a single thickness class
# ==============================================================================

def _thermo_single(
    h: jnp.ndarray,
    T_ice: jnp.ndarray,
    conc: jnp.ndarray,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
    shflx: jnp.ndarray | None = None,
    lhflx: jnp.ndarray | None = None,
    open_water_fraction: jnp.ndarray | None = None,
    enable_lead_freeze: bool = True,
    return_diagnostics: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Thermodynamic update for a single thickness class.

    Parameters
    ----------
    shflx, lhflx : jnp.ndarray or None
        Pre-computed sensible and latent heat fluxes.  When *None*
        (the default), ``simple_bulk_fluxes`` is called internally.
        Pass pre-computed values when the caller uses a different
        bulk-flux scheme (e.g. MOST).
    open_water_fraction : jnp.ndarray or None
        Aggregated ``(1 - sum_k conc_k)`` from the multi-category
        dispatch.  When supplied, replaces the per-category
        ``(1 - conc)`` in the concentration-growth term so the lead
        freeze is driven by *total* lead area rather than per-category
        lead area (which sums to > 1 across categories).  Default None
        preserves the slab/single-category behaviour.
    enable_lead_freeze : bool
        When False, suppress the open-water freezing (volume and
        concentration) contribution.  Multi-category callers set this
        True for category 0 only so the lead freeze isn't double-
        counted across categories.  Codex iter-3 finding #3.
    """
    ice_mask = h > 0.0
    h_eff = jnp.maximum(h, config.h_ice_min)

    # Bulk fluxes — use caller-supplied values when available.
    if shflx is None or lhflx is None:
        wind_speed = jnp.sqrt(
            jnp.maximum(
                forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2, 1e-12
            )
        )
        rho = forcing.rho_lowest
        q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)
        _, _, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_ice, q_sfc, rho, wind_speed,
            config.Cd_ice, config.Ch_ice,
            L_latent=constants.L_s,  # sublimation over ice
        )

    # Albedo: use ice albedo over ice, ocean albedo over open water
    if config.temp_dependent_albedo:
        alpha_ice = compute_ice_albedo(T_ice, config.ice_albedo)
    else:
        alpha_ice = jnp.full(h.shape, config.albedo_ice, dtype=h.dtype)
    alpha = jnp.where(ice_mask, alpha_ice, config.albedo_ocean)

    # Radiation
    sw_net, lw_net, lw_up = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_ice, alpha,
        config.emissivity_ice,
    )
    Q_sfc = sw_net + lw_net - shflx - lhflx

    # Conductance through the ice [W/m^2/K]: K = k_ice / h, so the
    # conductive flux is F_cond = K*(T_base - T_sfc), heat arriving at the
    # surface from the warm ice base.
    K_cond = jnp.where(ice_mask, config.k_ice / h_eff, 0.0)
    T_base = config.T_freeze_ocean

    # Surface skin-temperature update with the conductive term treated
    # SEMI-IMPLICITLY (evaluated at the new T).  Explicit Euler is stiff
    # for thin ice (K = k_ice/h_ice_min ~ 200 W/m^2/K at h = 1 cm, so
    # K*dt/skin_cap >> 1) and drives a 180 K<->273 K limit cycle that
    # prevents ice from accumulating.  Backward-Euler:
    #   cap*(T_new - T)/dt = Q_sfc + K*(T_base - T_new)
    #   => T_new = (cap_dt*T + Q_sfc + K*T_base) / (cap_dt + K)
    # is unconditionally stable, reduces to the explicit update for thick
    # ice (K -> 0), and has the same steady state (Q_sfc + K*(T_base-T)=0).
    skin_cap = config.rho_ice * config.c_ice * h_eff * 0.5
    cap_dt = skin_cap / dt
    T_implicit = (cap_dt * T_ice + Q_sfc + K_cond * T_base) / (cap_dt + K_cond)
    T_trial = jnp.where(ice_mask, T_implicit, T_ice)
    T_new = jnp.where(
        ice_mask,
        jnp.clip(T_trial, config.T_ice_min, config.T_melt_surface),
        jnp.full(T_ice.shape, config.T_freeze_ocean, dtype=T_ice.dtype),
    )

    # Conductive flux consistent with the updated (implicit) surface T, so
    # the basal-growth Stefan balance below uses the same bounded flux that
    # warmed the skin rather than the stiff old-temperature value.
    F_cond = jnp.where(ice_mask, K_cond * (T_base - T_new), 0.0)

    # Surface melt: if T_trial exceeds freezing, the excess enthalpy melts
    # ice from the top instead of being discarded by the temperature clamp.
    # Excess surface energy when the implicit T_trial overshoots the melt
    # point.  The implicit operator gives the residual at the melt point as
    # (cap_dt + K_cond)*(T_trial - T_melt_surface); using skin_cap/dt =
    # cap_dt alone (the explicit coefficient) would under-count melt by the
    # conductive term K_cond and silently delete heat (large for thin ice).
    excess_energy_raw = (cap_dt + K_cond) * jnp.maximum(
        T_trial - config.T_melt_surface, 0.0
    )  # [W/m²]
    # Cap the surface-melt RATE at the available ice (per unit time): the
    # excess energy can exceed the column's latent capacity for thin ice, and
    # an uncapped rate would drive h negative (clamped) while the conc / ocean
    # diagnostics over-report the melt.  rho_ice*L_f*h/dt is the energy that
    # fully ablates the existing ice this step.
    max_surface_melt_W = config.rho_ice * config.L_f * h / dt
    excess_energy = jnp.minimum(excess_energy_raw, max_surface_melt_W)
    dh_dt_surface_melt = -excess_energy / (config.rho_ice * config.L_f)
    # NOTE: the surplus surface-melt heat that must warm the ocean (energy
    # closure, finding #6) is computed BELOW, AFTER ``removal_scale`` is applied
    # to ``surface_melt_rate`` — it must be based on the REALIZED surface melt
    # (post first-cap AND post multi-process competition), not this first cap
    # alone (codex adversarial review).

    # Growth/melt — turbulent ocean heat transfer (not conductive scaling)
    F_ocean = config.ocean_heat_transfer_coeff * jnp.maximum(
        ocean_sst - config.T_freeze_ocean, 0.0,
    )
    dh_dt_basal = (F_cond - F_ocean) / (config.rho_ice * config.L_f)

    # Sublimation mass loss: lhflx > 0 means moisture leaves the
    # surface into the atmosphere via L_s, so the equivalent ice mass
    # is removed from the column.  When lhflx < 0 (deposition), mass
    # is added.  Without this term the surface energy budget closes
    # but the ice mass budget is open: thin polar ice would grow
    # endlessly under sublimation, biased high by ~tens of cm/year.
    # (Coupler-conservation audit F7.)
    dh_dt_sublim = jnp.where(
        ice_mask,
        -lhflx / (config.rho_ice * constants.L_s),
        0.0,
    )

    # ---- Realized (capped) existing-ice thickness increments (#28) ----------
    # Removal (surface + basal melt + sublimation loss) cannot exceed the ice
    # available this step, ``h/dt``; scale ALL removal terms by one factor.  The
    # SAME realized increments drive both the prognostic volume below AND the
    # ocean-exchange diagnostics, so the state and the freshwater/heat fluxes
    # agree even when the cap fires.  Previously the cap scaled only the
    # returned diagnostics while ``h`` advanced on the UNCAPPED rate, so in the
    # clamp branch the reported ocean exchange did not match the state volume
    # change (codex).  ``eps`` guards the divide (numerical floor, not tunable).
    basal_growth_rate = jnp.maximum(dh_dt_basal, 0.0)    # >=0 basal congelation
    deposition_rate = jnp.maximum(dh_dt_sublim, 0.0)     # >=0 vapor -> ice (atm)
    surface_melt_rate = jnp.maximum(-dh_dt_surface_melt, 0.0)
    basal_melt_rate = jnp.maximum(-dh_dt_basal, 0.0)
    sublim_loss_rate = jnp.maximum(-dh_dt_sublim, 0.0)   # >0 = ice -> atmosphere
    total_removal = surface_melt_rate + basal_melt_rate + sublim_loss_rate
    # The removal can consume the ice present at the START plus whatever basal
    # congelation / deposition ADDS during the step: removal_cap = h/dt + growth
    # + deposition.  Capping at h/dt alone over-caps (and over-scales F_ocean)
    # when growth/deposition replenish the column, leaving the cell with too
    # much ice and the ocean heat scaled as if the base had vanished (codex #28).
    eps = 1e-30
    removal_cap = h / dt + basal_growth_rate + deposition_rate
    removal_scale = jnp.where(
        total_removal > removal_cap,
        removal_cap / jnp.maximum(total_removal, eps),
        1.0,
    )
    # Heat / atmosphere-mass channels must be capped CONSISTENTLY with the
    # state (codex #28): the turbulent ocean->base flux and the bulk latent
    # flux demand more melt / sublimation than thin ice can supply, but the
    # ice can only lose what it holds.
    #   (a) Ocean heat: the ocean->base turbulent flux operates only while an
    #       ice base exists, so when the column over-ablates within the step it
    #       stops early.  The realized ocean heat loss is F_ocean scaled by the
    #       same survived fraction ``removal_scale`` (= available ice / demanded
    #       removal); this keeps the ocean heat consistent with the realized
    #       melt mass and bounded in [0, F_ocean].  (Subtracting the unrealized
    #       LATENT instead would over-debit by the conductive share when the
    #       surface is warmer than the base, F_cond < 0.)  No-op at scale == 1.
    #   (b) Atmosphere mass: the realized ice->atmosphere sublimation is the
    #       CAPPED loss minus deposition; equals lhflx/L_s in the no-clamp
    #       regime (rho_ice*(max(-x,0)-max(x,0)) = -rho_ice*dh_dt_sublim).
    ocean_heat_basal_per_ice_area = F_ocean * removal_scale
    surface_melt_rate = surface_melt_rate * removal_scale
    basal_melt_rate = basal_melt_rate * removal_scale
    sublim_loss_rate = sublim_loss_rate * removal_scale
    sublim_mass_per_ice_area = config.rho_ice * (sublim_loss_rate - deposition_rate)
    # Surplus surface-melt heat that REALIZED ablation could not consume must
    # warm the ocean mixed layer (energy closure, finding #6), not be discarded.
    # Uses the REALIZED surface melt (post first-cap AND post ``removal_scale``
    # competition with basal melt / sublimation), so
    # ``realized_surface_melt_W + gain == excess_energy_raw`` exactly (energy
    # closes).  Gated to ice-present cells; CREDITED to the ocean (subtracted
    # from ocean_heat_extraction) in the callers.  [W/m² per ice-area]
    realized_surface_melt_W = config.rho_ice * config.L_f * surface_melt_rate
    surface_melt_ocean_gain_per_ice_area = jnp.where(
        ice_mask, jnp.maximum(excess_energy_raw - realized_surface_melt_W, 0.0), 0.0,
    )

    # ---- Energy-closing latent flux + single skin re-solve (#28) ----
    # The latent the ATMOSPHERE receives must equal L_s * the REALIZED (post
    # removal_scale) sublimation mass, otherwise the over-ablation cap cools the
    # skin with more latent enthalpy than the atmosphere gets and the coupled
    # surface energy budget does not close.  Guard with ``removal_scale < 1`` so
    # thick / deposition / no-cap cells return the bulk ``lhflx`` BITWISE (the
    # L_s/rho_ice round-trip is bit-exact for typical |lhflx| but leaves a 1-ULP
    # residual near zero; the guard removes it and protects validated baselines).
    lhflx_realized = jnp.where(
        removal_scale < 1.0,
        constants.L_s * sublim_mass_per_ice_area,
        lhflx,
    )
    if config.latent_skin_resolve:
        # Re-solve the implicit skin balance ONCE with the realized latent so the
        # returned T_new / F_cond / lead-freeze flux are cooled by exactly the
        # latent the atmosphere receives.  NO iteration -- the fixed point drives
        # the realized latent toward zero in mixed melt+sublim clamp cells
        # (surface melt saturates its own h/dt share and starves sublimation).
        # The melt/removal PARTITION above is kept from the first (bulk-latent)
        # pass; that 2nd-order residual is bounded by L_s*rho_ice*h/dt (sub-cm
        # clamped ice).  sw_net/lw_net/K_cond/T_base/cap_dt are latent-independent
        # and reused; for thick ice lhflx_realized == lhflx so Q_sfc / T_new /
        # F_cond / freeze_flux_open are byte-identical to the first pass.
        Q_sfc = sw_net + lw_net - shflx - lhflx_realized
        T_implicit = (cap_dt * T_ice + Q_sfc + K_cond * T_base) / (cap_dt + K_cond)
        T_trial = jnp.where(ice_mask, T_implicit, T_ice)
        T_new = jnp.where(
            ice_mask,
            jnp.clip(T_trial, config.T_ice_min, config.T_melt_surface),
            jnp.full(T_ice.shape, config.T_freeze_ocean, dtype=T_ice.dtype),
        )
        # F_cond is NOT recomputed: the basal exchange above already consumed
        # the first-pass F_cond; only T_new (returned skin temp -> lw_up / q_sfc)
        # and the re-solved Q_sfc (-> freeze_flux_open below) feed downstream.
    # Net realized existing-ice thickness rate.  In the no-clamp regime
    # (removal_scale == 1) this is identically dh_dt_basal + dh_dt_surface_melt
    # + dh_dt_sublim (max(x,0) - max(-x,0) = x).  When the cap fires the removal
    # is bounded at removal_cap so h + dt*dh_dt_ice = h + dt*(growth + depo) -
    # dt*removal_cap = 0 exactly (full exhaustion); otherwise it stays > 0.
    # Either way h_existing >= 0 by construction (the max(.,0) floor never binds).
    dh_dt_ice = (
        basal_growth_rate + deposition_rate
        - surface_melt_rate - basal_melt_rate - sublim_loss_rate
    )

    freeze_flux_open = jnp.maximum(-Q_sfc, 0.0)
    dh_dt_open_raw = freeze_flux_open / (config.rho_ice * config.L_f)
    # Multi-category: only deposit lead-freeze in category 0.  Cats
    # with ``enable_lead_freeze=False`` see ``dh_dt_open = 0`` so the
    # same open-water freeze does not fire per-category.
    dh_dt_open = dh_dt_open_raw if enable_lead_freeze else jnp.zeros_like(dh_dt_open_raw)
    dh_dt = jnp.where(ice_mask, dh_dt_ice, dh_dt_open)

    # ---- Volume-based area<->thickness update (V = h*A self-consistent, #28) --
    # Earlier this kernel set ``h_new = h + dt*dh_dt`` and evolved ``conc_new``
    # INDEPENDENTLY, so the lead-freeze ice volume was never folded into the
    # thickness: partial-cover cells inherited the existing category thickness
    # for the newly-frozen area (a thick floe MANUFACTURED volume, a thin floe
    # DROPPED freeze volume), and melt double-counted (volume fell by BOTH
    # thinning and area retreat) while ``_ocean_exchange_from_diag`` reported the
    # true per-process volume.  Now mirror ``_thermo_v2``: build the post-process
    # ice VOLUME, then recover thickness from the new area so V = h*A holds and
    # the prognostic state matches the ocean-exchange diagnostics (codex).
    #
    # Existing-ice thickness after the vertical processes (basal/surface melt,
    # basal growth, sublimation), applied over the CURRENT ice area; open-water
    # cells carry no existing ice.
    h_existing = jnp.where(ice_mask, jnp.maximum(h + dt * dh_dt_ice, 0.0), 0.0)
    V_existing = h_existing * conc

    # Concentration evolution (CICE / Icepack ``add_new_ice`` convention).
    #
    # Growth: areal concentration only increases from NEW-ICE FORMATION in
    # OPEN-WATER portions of the cell.  The driver is ``dh_dt_open`` (the
    # lead-freezing rate from a destabilizing surface flux), NOT ``dh_dt_ice``
    # (vertical growth of existing floes).  New ice forms at the nominal
    # thickness ``h_new_ice``, so the area filled is ``delta_V_lead/h_new_ice``;
    # this holds whether the cell is fully open (ice_mask=False) or partially
    # ice-covered (A<1, the ``(1 − A)`` lead fraction refreezes).
    #
    # Melt: concentration decreases as floes shrink in area while their
    # thickness stays roughly constant — ``dh_dt · A / h_eff`` (sign carries
    # through, dh_dt < 0 in melt).
    # ``lead_area`` is the lead area available for refreezing.  In single-
    # category mode this is ``(1 - conc)`` of the local cell; in multi-category
    # mode the caller supplies the aggregated ``(1 - sum_k conc_k)`` to avoid
    # summing more than 1 across cats.
    lead_area = (
        (1.0 - conc) if open_water_fraction is None else open_water_fraction
    )
    # New lead-freeze ice VOLUME deposited in the open-water (lead) fraction.
    delta_V_lead = jnp.maximum(dh_dt_open * lead_area * dt, 0.0)
    V_after = V_existing + delta_V_lead

    dconc_growth = dh_dt_open * lead_area / config.h_new_ice
    # Cap new lead-ice area at the available lead area: a lead can freeze over
    # fully but not beyond.  Unbounded, ``dh_dt_open*dt/h_new_ice`` can exceed
    # 1 under strong freezing / long dt, letting cat 0 add more area than the
    # lead holds so sum_k a_k > 1 reaches the ITD remap.  The excess freeze
    # volume is preserved in ``V_after`` and thickens the new ice below (codex).
    dconc_growth_area = jnp.minimum(dt * dconc_growth, lead_area)
    dconc_melt = jnp.minimum(dh_dt, 0.0) * conc / h_eff
    conc_new = jnp.clip(conc + dconc_growth_area + dt * dconc_melt, 0.0, 1.0)

    # Volume-consistent thickness + ablation zero-out: recover h from the
    # conserved post-process volume so V = h*A.  When the post-step volume is
    # non-positive the column fully ablated, so conc MUST be zero — otherwise an
    # h=0 / conc>0 "zombie" tile persists and keeps exporting basal ocean heat
    # (F_ocean*conc) on later steps, a systematic bias.
    has_vol = V_after > 1e-12
    conc_new = jnp.where(has_vol, conc_new, 0.0)
    h_new = jnp.where(
        (conc_new > 1e-12) & has_vol,
        V_after / jnp.maximum(conc_new, 1e-12),
        0.0,
    )

    if not return_diagnostics:
        return h_new, T_new, conc_new

    # ----- Per-grid-cell ice-VOLUME RATES [m_ice / m2_cell / s] (F2) -----
    # Per-PROCESS volume rates for assembling the ocean freshwater/heat
    # exchange in ``_step_slab``.  These reuse the SAME capped
    # ``surface_melt_rate`` / ``basal_melt_rate`` / ``sublim_loss_rate`` /
    # ``basal_growth_rate`` that advanced ``h`` above, so the reported ocean
    # exchange matches the prognostic volume change EXACTLY, including in the
    # over-ablation clamp branch (#28, codex).  Existing-ice processes act over
    # the ice fraction ``conc``; the lead-freeze term is deposited over the
    # open-water ``lead_area`` (already per-cell).
    vmelt_ice = (surface_melt_rate + basal_melt_rate) * conc  # ice melt -> ocean
    vgrowth_basal = basal_growth_rate * conc                  # basal congelation
    vsublim = sublim_loss_rate * conc                         # ice -> atmosphere
    vlead_freeze = jnp.maximum(dh_dt_open * lead_area, 0.0)   # new lead ice
    diag = {
        "vmelt_ice": vmelt_ice,
        "vgrowth_basal": vgrowth_basal,
        "vsublim": vsublim,
        "vlead_freeze": vlead_freeze,
        "F_ocean_per_ice_area": F_ocean,
        # Ocean basal heat with the unrealized (clamped) basal-melt latent
        # removed, so ocean heat matches the realized state change (#28).
        "ocean_heat_basal_per_ice_area": ocean_heat_basal_per_ice_area,
        # Surplus surface-melt heat (per ice-area) that ablation could not
        # consume; credited to the ocean by the callers so energy closes.
        "surface_melt_ocean_gain_per_ice_area": (
            surface_melt_ocean_gain_per_ice_area
        ),
        # Realized ice->atmosphere sublimation mass [kg/m2(ice)/s] (#28).
        "sublim_mass_per_ice_area": sublim_mass_per_ice_area,
        # Latent flux PER-ICE-AREA consistent with the realized sublimation mass
        # (== L_s * sublim_mass_per_ice_area where the cap fires, else the bulk
        # lhflx); fed to the coupler so the atmosphere latent energy matches the
        # skin solve and the moisture mass (#28).
        "lhflx_realized": lhflx_realized,
    }
    return h_new, T_new, conc_new, diag


def _ocean_exchange_from_diag(diag, conc_in, config):
    """Per-grid-cell ice->ocean freshwater [kg/m2/s] and heat extraction
    [W/m2] from ``_thermo_single``'s per-process diagnostics.

    Identical decomposition to ``_step_slab``: melt adds freshwater; basal
    congelation and lead freeze extract ocean water; sublimation is excluded
    (atmosphere).  ``vmelt_ice`` / ``vgrowth_basal`` / ``vlead_freeze`` are
    already per-grid-cell (weighted by the ice fraction / lead area inside the
    kernel); ``ocean_heat_basal_per_ice_area`` is per-ice-area so it is weighted
    here by the INPUT ice fraction ``conc_in`` (the area over which the ocean
    lost turbulent heat during the step).  Lead-freeze latent heat is drawn from
    the ocean; basal-growth latent heat is released upward by conduction (absent).

    With the volume-based V=h*A thermo-state update (#28) ``_thermo_single`` now
    folds the lead-freeze volume into ``h`` and uses ONE set of capped process
    increments for both the state and these diagnostics, so the reported
    exchange matches the prognostic state change ``delta(h*conc)`` -- including
    in the over-ablation clamp branch (the basal-heat field already nets out the
    unrealized capped melt; ``vmelt_ice`` / ``vsublim`` use the capped rates).
    """
    rho, Lf = config.rho_ice, config.L_f
    fw = rho * (diag["vmelt_ice"] - diag["vgrowth_basal"] - diag["vlead_freeze"])
    heat = (
        diag["ocean_heat_basal_per_ice_area"] * conc_in
        # Surplus surface-melt heat WARMS the ocean (energy closure, finding
        # #6): negative contribution (+sign = ocean loses heat to the ice).
        - diag["surface_melt_ocean_gain_per_ice_area"] * conc_in
        + rho * Lf * diag["vlead_freeze"]
    )
    # Realized ice->atmosphere sublimation mass, PER-GRID-CELL: the capped
    # per-ice-area mass weighted by the INPUT ice fraction so it survives
    # terminal melt-out under the f_water blend (#28, codex; mirrors the FW
    # exchange convention).
    atmos_mass = diag["sublim_mass_per_ice_area"] * conc_in
    return fw, heat, atmos_mass


def _cap_multicat_concentration(conc, h, h_snow=None, pond_depth=None):
    """Restore the aggregate-area invariant ``sum_k a_k <= 1`` after a
    multi-category transport step that advected each category independently.

    Where the total concentration exceeds 1, scale the excess area into
    THICKNESS — a uniform mechanical compaction that fits the ice into the
    cell: ``a_k /= S``, ``h_k *= S`` (and ``h_snow_k *= S``, ``pond_depth_k *=
    S`` when present), with ``S = max(sum_k a_k, 1)``.  This conserves ice
    volume ``h*a``, snow volume ``h_snow*a``, pond water ``area*depth*a``, and
    salt ``S_ice*h*a`` (the intensive ``S_ice`` is unchanged) while capping
    ``sum_k a_k`` at 1.  ``S = 1`` (a no-op) where the cell is not overfilled.

    BIN ACCURACY (#28, verified): for REALISTIC sub-CFL overfill (<<1%; tested
    up to ~3%) the compaction is a sub-percent thickening, so the subsequent
    Lipscomb ITD remap re-sorts every occupied category back WITHIN its bin
    bounds while conserving volume / snow / pond / salt
    (``test_cap_realistic_overfill_stays_bin_accurate``).  For PATHOLOGICAL
    LARGE overfill (e.g. sum_k a_k -> 2) the compacted thickness can span more
    than one ITD bin and the linear Lipscomb g(h) reconstruction smears the
    per-category mean thickness slightly outside its index's bounds; the remap
    still conserves volume / snow / pond / salt and total area exactly
    (``test_cap_large_overfill_conserves``), so the result is SAFE, only the ITD
    SHAPE degrades.  A hard bin-cascade re-sort would restore strict per-bin
    membership but is intentionally NOT used: it is a step-function in ``h``
    (non-differentiable) and the large-overfill regime is unreachable in
    practice (the root-cause lead-freeze area bound + sub-CFL transport keep
    overfill tiny).
    """
    sum_conc = jnp.sum(conc, axis=-1, keepdims=True)
    overfill = jnp.maximum(sum_conc, 1.0)
    conc = conc / overfill
    h = h * overfill
    if h_snow is not None:
        h_snow = h_snow * overfill
    if pond_depth is not None:
        pond_depth = pond_depth * overfill
    return conc, h, h_snow, pond_depth


def _pond_area_depth_from_volume(V_pond_per_cell, conc, config):
    """Reconstruct per-ice-area pond ``(area, depth)`` from the per-GRID-CELL
    pond water volume after a category-conserving transfer (ITD remap or
    ridging), conserving the volume EXACTLY even when the area cap binds.

    ``pond_area * pond_depth`` equals the per-ice-area pond volume
    (``V_pond_per_cell / conc``).  The initial sqrt split honours
    ``depth_to_area_ratio`` and conserves volume only while the area is below
    ``pond_to_ice_max_area``; at the cap, ``depth`` is recomputed from the
    clipped area so ``area * depth`` stays equal to the volume (otherwise pond
    water above ``depth_to_area_ratio * max_area**2`` would be silently lost).
    The 1e-14 sqrt floor keeps no-pond cells no-pond and the gradient finite.
    """
    conc_safe = jnp.maximum(conc, 1e-12)
    vol_per_ice_area = jnp.where(conc > 1e-12, V_pond_per_cell / conc_safe, 0.0)
    depth_init = jnp.sqrt(
        jnp.maximum(config.ponds.depth_to_area_ratio * vol_per_ice_area, 1e-14)
    )
    area = jnp.where(
        depth_init > 1e-6,
        jnp.clip(vol_per_ice_area / jnp.maximum(depth_init, 1e-6), 0.0,
                 config.ponds.pond_to_ice_max_area),
        0.0,
    )
    depth = jnp.where(
        area > 1e-6, vol_per_ice_area / jnp.maximum(area, 1e-12), depth_init,
    )
    return area, depth


# ==============================================================================
# Build TileResponse
# ==============================================================================

def _build_response(
    h: jnp.ndarray,
    T_ice: jnp.ndarray,
    conc: jnp.ndarray,
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    forcing: AtmToSurface,
    config: SeaIceConfig,
    U_min: float,
    *,
    h_old: jnp.ndarray | None = None,
    conc_old: jnp.ndarray | None = None,
    ocean_sst: jnp.ndarray | None = None,
    ocean_u: jnp.ndarray | None = None,
    ocean_v: jnp.ndarray | None = None,
    dt: float | None = None,
    freshwater_override: jnp.ndarray | None = None,
    ocean_heat_override: jnp.ndarray | None = None,
    surface_mass_override: jnp.ndarray | None = None,
    lhflx_override: jnp.ndarray | None = None,
) -> TileResponse:
    """Build coupler response from aggregated ice fields.

    The optional ``h_old``/``ocean_*``/``dt`` arguments enable the
    dynamic multi-category path to compute the freshwater flux, ocean
    heat extraction, and ocean stress feedbacks — the slab path
    computes these inline.  Without them the response carries zero
    placeholders (legacy behaviour) which silently breaks ice→ocean
    feedback.  See codex finding #8.
    """
    q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)

    if config.temp_dependent_albedo:
        alpha_ice = compute_ice_albedo(T_ice, config.ice_albedo)
    else:
        alpha_ice = jnp.full(h.shape, config.albedo_ice, dtype=h.dtype)

    lw_up = (
        config.emissivity_ice * constants.sigma_sb * T_ice ** 4
        + (1.0 - config.emissivity_ice) * forcing.lw_down
    )

    # Recompute surface fluxes from aggregated state via shared dispatch
    tau_x, tau_y, shflx, lhflx = _bulk_flux_dispatch(
        T_ice, forcing, config, U_min,
    )
    # Realized latent override (per-ice-area) from the kernel diagnostics, so
    # the atmosphere latent energy matches the realized sublimation mass and the
    # skin solve (#28).  Falls back to the recomputed bulk latent for external
    # callers that do not thread it.
    if lhflx_override is not None:
        lhflx = lhflx_override

    # Ice → ocean feedbacks.  When the dynamic path threads
    # ``h_old`` + ``ocean_*`` + ``dt`` we compute the per-ice-area
    # thickness-rate FW flux that the slab path produces inline;
    # otherwise expose zero placeholders.  See the matching slab-
    # path note on the deferred d(h·conc)/dt formulation.
    # Returned PER-GRID-CELL (per-water-area): the per-ice-area thickness-rate
    # terms are multiplied by the ice fraction ``conc`` so blend_tiles can
    # weight ice->ocean exchange by ``f_water`` (F11).
    #
    # Preferred path: the caller (``_step_dynamic``) supplies the per-PROCESS,
    # per-grid-cell freshwater/heat budget aggregated from ``_thermo_single``'s
    # diagnostics — identical decomposition to the slab path (melt / basal
    # growth / lead freeze, sublimation excluded, F_ocean over the input ice
    # area).  This correctly separates basal growth (over the existing-ice
    # area) from lead freeze (over the open-water lead area), which a single
    # aggregate ``(h - h_old)/dt`` weight cannot.  F11.
    if freshwater_override is not None:
        freshwater_flux = freshwater_override
    elif h_old is not None and dt is not None:
        # Fallback aggregate budget (used only when diagnostics are not
        # threaded, e.g. external callers).  Per-cell exchange area =
        # ``max(conc_old, conc)`` (the participating ice fraction) so terminal
        # melt-out (conc -> 0, h_old > 0) is not zeroed; reduces to ``conc``
        # for steady ice and new-ice formation.  NOTE: this aggregate cannot
        # separate basal growth from lead freeze and may mis-weight mixed
        # partial-cover growth — prefer ``freshwater_override``.
        exch_conc = (
            jnp.maximum(conc_old, conc) if conc_old is not None else conc
        )
        dh_dt_total = (h - h_old) / dt
        ice_mask_init = h_old > config.h_ice_min
        dh_dt_sublim = jnp.where(
            ice_mask_init,
            -lhflx / (config.rho_ice * constants.L_s),
            0.0,
        )
        freshwater_flux = -config.rho_ice * (dh_dt_total - dh_dt_sublim) * exch_conc
    else:
        freshwater_flux = jnp.zeros_like(h)

    if ocean_heat_override is not None:
        ocean_heat_extraction = ocean_heat_override
    elif (
        h_old is not None and ocean_sst is not None and dt is not None
    ):
        exch_conc = (
            jnp.maximum(conc_old, conc) if conc_old is not None else conc
        )
        ice_mask_init = h_old > config.h_ice_min
        F_ocean = jnp.where(
            ice_mask_init,
            config.ocean_heat_transfer_coeff
            * jnp.maximum(ocean_sst - config.T_freeze_ocean, 0.0),
            0.0,
        )
        dh_dt_freeze_open = jnp.where(~ice_mask_init, h / dt, 0.0)
        open_freeze_flux = config.rho_ice * config.L_f * dh_dt_freeze_open
        ocean_heat_extraction = (F_ocean + open_freeze_flux) * exch_conc
    else:
        ocean_heat_extraction = jnp.zeros_like(h)

    if ocean_u is not None and ocean_v is not None:
        du_oi = ocean_u - u_ice
        dv_oi = ocean_v - v_ice
        speed_oi = jnp.sqrt(du_oi ** 2 + dv_oi ** 2 + 1e-10)
        tau_oi_x = config.rho_ocean_ref * config.drag_ocean * speed_oi * du_oi
        tau_oi_y = config.rho_ocean_ref * config.drag_ocean * speed_oi * dv_oi
        # Per-ice-tile (no ``* conc``): blend_tiles applies the single area
        # weight ``f_ice``.  F11 — see the slab-path note.
        ocean_stress_x = -tau_oi_x
        ocean_stress_y = -tau_oi_y
    else:
        ocean_stress_x = jnp.zeros_like(h)
        ocean_stress_y = jnp.zeros_like(h)

    return TileResponse(
        T_sfc=T_ice,
        albedo=alpha_ice,
        emissivity=jnp.full(h.shape, config.emissivity_ice, dtype=h.dtype),
        z0=jnp.full(h.shape, config.z0_ice, dtype=h.dtype),
        q_surface=q_sfc,
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up,
        u_ocean_sfc=u_ice,
        v_ocean_sfc=v_ice,
        co2_flux=jnp.zeros_like(h),
        freshwater_flux=freshwater_flux,
        ocean_heat_extraction=ocean_heat_extraction,
        ocean_stress_x=ocean_stress_x,
        ocean_stress_y=ocean_stress_y,
        # Sublimation mass flux from the ice surface, PER-GRID-CELL (weighted
        # by the input ice fraction) so blend_tiles weights it by f_water and
        # the atmosphere mass survives terminal melt-out.  Preferred path: the
        # realized (capped) ``surface_mass_override`` from _thermo_single's
        # diagnostics; fallback to the uncapped bulk lhflx/L_s * conc (#28).
        surface_mass_flux=(
            surface_mass_override if surface_mass_override is not None
            else lhflx / constants.L_s * conc
        ),
        # Salt-flux placeholder — populated by the brine-aware path
        # in ``_step_dynamic`` when ``config.brine.enabled``.  The
        # zero default keeps the freshwater convention for legacy
        # configs.
        salt_flux=jnp.zeros_like(h),
    )


# ==============================================================================
# Extended-physics dynamic sea ice step (Tier 1 + Tier 2)
# ==============================================================================

def _closing_rate_from_velocity(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    grid,
    cap: float,
) -> jnp.ndarray:
    """Net convergence rate ``max(0, −div(u_ice))`` from the velocity field.

    The grid is validated for operator support at ``step_sea_ice`` entry
    (``_grid_supports_ice_dynamics``), so ``strain_rates`` is called
    directly — no exception-swallowing fallback, which would silently
    disable ridging and detach the gradient w.r.t. velocity (forbidden by
    the repo AD rules).
    """
    from legoesm.ice.rheology import strain_rates
    eps_11, eps_22, _eps_12 = strain_rates(u_ice, v_ice, grid)
    div = eps_11 + eps_22
    return jnp.clip(-div, 0.0, cap)


def _thermo_v2(
    h: jnp.ndarray,
    T_ice: jnp.ndarray,
    conc: jnp.ndarray,
    h_snow: jnp.ndarray,
    S_ice: jnp.ndarray,
    pond_area: jnp.ndarray,
    pond_depth: jnp.ndarray,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
    *,
    open_water_fraction: jnp.ndarray | None = None,
    enable_lead_freeze: bool = True,
) -> dict:
    """Extended thermodynamics: snow + brine + SW scheme + ponds aware.

    Returns a dictionary of post-step per-category fields plus
    diagnostics needed by the caller to build the coupler response
    (delta volumes for the salt budget, SW penetration, etc.).
    """
    ice_mask = h > 0.0
    h_eff = jnp.maximum(h, config.h_ice_min)

    # Bulk fluxes via shared dispatch (consistent across paths).
    tau_x, tau_y, shflx, lhflx = _bulk_flux_dispatch(T_ice, forcing, config, U_min)

    # 1. Snow accumulation from precipitation (positive on ice cells).
    # ``config.snow.enabled`` is a static Python bool — safe to branch
    # on under jit; the ``has_precipitation`` array flag is folded
    # into ``accumulate_snowfall`` via array masking so JIT/scan can
    # trace the branch without lowering a tracer-boolean conversion.
    if config.snow.enabled:
        precip_snow_gated = forcing.precip_snow * jnp.asarray(
            forcing.has_precipitation, dtype=forcing.precip_snow.dtype,
        )
        # Snowfall on the ICE fraction accumulates in ``h_snow``.  The second
        # return (snow on the OPEN-water fraction) is intentionally discarded
        # here: open-water precipitation is the OCEAN tile's responsibility
        # (it receives ``precip_total``, which includes snow), so routing it
        # through the ice tile too would double-count it after blend_tiles
        # weights ice->ocean exchange by ``f_water`` (F11).
        # KNOWN second-order limitation: precip over the lead area that
        # FREEZES within the step (open-water -> ice transition sliver) is
        # weighted by the post-step ocean fraction, so ~precip*dt*Δconc of it
        # is missed for one step.  This O(dt*Δconc) error is inherent to
        # weighting any CONTINUOUS atmospheric flux (shflx/lhflx/precip) by the
        # instantaneous tile fractions and is not specific to snow; the large
        # DISCRETE ice melt/freeze exchange is handled conservatively by the
        # per-cell + f_water budget above.
        h_snow, _snow_to_ocean_open_unused = accumulate_snowfall(
            h_snow, precip_snow_gated, ice_mask, dt,
            rho_snow=config.snow.rho_snow,
        )

    # 2. Shortwave: alpha + absorbed + penetrated.
    sw_result = compute_ice_sw(
        forcing.sw_down, T_ice, h, h_snow,
        pond_area, pond_depth,
        scheme=config.shortwave_scheme,
        albedo_const=config.albedo_ice,
        sw_transmittance_const=config.sw_transmittance_const,
    )
    alpha = sw_result.albedo_eff
    sw_absorbed = sw_result.sw_absorbed_surface
    sw_penetrated = sw_result.sw_penetrated

    # Longwave net (gray-radiation surface BC; positive = into surface).
    # Upward LW = thermal emission + reflected downwelling:
    #   lw_up = e*sigma*T^4 + (1-e)*lw_down.
    # Net LW at the skin is the full downwelling minus the full upwelling,
    #   lw_net = lw_down - lw_up = e*lw_down - e*sigma*T^4,
    # i.e. (absorbed downwelling) - (emitted).  Writing it as
    #   e*lw_down - lw_up
    # double-removes the reflected (1-e)*lw_down term (already inside lw_up),
    # under-counting lw_net by (1-e)*lw_down (~8 W/m^2 at e=0.97).  This
    # matches core.surface_radiation_fluxes used by _thermo_single.
    lw_up = (
        config.emissivity_ice * constants.sigma_sb * T_ice ** 4
        + (1.0 - config.emissivity_ice) * forcing.lw_down
    )
    lw_net = forcing.lw_down - lw_up
    # Q_sfc is the net heat available at the surface skin.
    Q_sfc = sw_absorbed + lw_net - shflx - lhflx

    # 3. Conductance through the snow+ice column [W/m^2/K].  Kept
    #    separate from the flux so the surface balance below can treat the
    #    conductive term semi-implicitly (stable for thin ice).
    T_base_arr = jnp.full_like(T_ice, config.T_freeze_ocean)
    K_cond = jnp.where(
        ice_mask,
        combined_conductance(
            h_ice=h,
            h_snow=h_snow,
            k_ice=config.k_ice,
            k_snow=config.snow.k_snow,
            h_ice_min=config.h_ice_min,
            h_snow_min=config.snow.h_snow_min,
        ),
        0.0,
    )

    # 4. Surface energy balance to update T_ice.
    skin_cap = (
        config.rho_ice * config.c_ice * h_eff * 0.5
        + config.snow.rho_snow * config.snow.c_snow * jnp.maximum(h_snow, 0.0)
    )
    skin_cap = jnp.maximum(skin_cap, 1.0)
    cap_dt = skin_cap / dt
    # Semi-implicit conductive term (see _thermo_single): unconditionally
    # stable for thin ice where K_cond*dt >> skin_cap.
    T_implicit = (cap_dt * T_ice + Q_sfc + K_cond * T_base_arr) / (cap_dt + K_cond)
    T_trial = jnp.where(ice_mask, T_implicit, T_ice)
    T_new = jnp.where(
        ice_mask,
        jnp.clip(T_trial, config.T_ice_min, config.T_melt_surface),
        jnp.full(T_ice.shape, config.T_freeze_ocean, dtype=T_ice.dtype),
    )
    # Conductive flux consistent with the updated surface T.
    F_cond = jnp.where(ice_mask, K_cond * (T_base_arr - T_new), 0.0)

    # NOTE (energy-closure approximation, tracked): when a thin column is fully
    # ablated within a step, surface-melt energy above the column's latent
    # capacity is not re-routed back to the open-water tile.  Per-step magnitude
    # is bounded by the column latent capacity (rho_ice*L_f*h) and is small
    # except for sub-mm ice.  The BASAL turbulent heat is now scaled by the
    # realized basal-melt fraction (``ocean_heat_scale`` below), so the ocean is
    # no longer charged F_ocean for a column that surface melt already removed
    # (#28).  The mass budget IS closed (melt/sublimation are capped at
    # available ice above).
    # 5. Excess surface energy → melt: snow first then ice.
    # Implicit-operator melt residual: (cap_dt + K_cond)*(T_trial - T_melt),
    # not skin_cap/dt alone (see _thermo_single) — otherwise thin-ice melt
    # energy is under-counted by the conductive term and heat is lost.
    excess_W = jnp.maximum(T_trial - config.T_melt_surface, 0.0) * (cap_dt + K_cond)
    energy_for_melt = excess_W * dt  # [J/m²]
    (
        h_snow_after_melt, h_after_melt, snow_melt_m, ice_melt_m,
        surface_melt_unconsumed_J,
    ) = consume_from_snow_then_ice(
        energy_for_melt, h_snow, h, config.snow.rho_snow, config.rho_ice, config.L_f,
    )
    # Surplus surface-melt energy left after the column fully ablated [J/m^2,
    # per-ice-area].  It must warm the ocean mixed layer (finding #6): convert
    # to a per-grid-cell flux and CREDIT the ocean below (negative contribution
    # to ocean_heat_extraction, whose +sign means ocean LOSES heat to the ice).
    # Weighted by the INPUT ``conc`` (the same ice-area snapshot used for
    # ``F_ocean * conc`` and the penetrated-SW channel).
    surface_melt_ocean_gain = surface_melt_unconsumed_J / dt * conc  # [W/m^2]

    # 6. Basal exchange (ocean side).  After clipping ``h`` at zero
    #    we recover the *actual* basal mass change so the FW + salt
    #    budgets don't ship more ice than the column held.
    F_ocean = config.ocean_heat_transfer_coeff * jnp.maximum(
        ocean_sst - config.T_freeze_ocean, 0.0,
    )
    dh_dt_basal = (F_cond - F_ocean) / (config.rho_ice * config.L_f)
    h_after_basal = jnp.maximum(h_after_melt + dt * dh_dt_basal, 0.0)
    basal_delta = h_after_basal - h_after_melt
    basal_growth_m = jnp.maximum(basal_delta, 0.0)
    basal_melt_m = jnp.maximum(-basal_delta, 0.0)
    # Fraction of the DEMANDED basal melt that the (post-surface-melt) column
    # could actually supply: when surface melt has already thinned the column,
    # basal_melt_m is capped below the F_ocean-driven demand, and the ocean
    # turbulent heat charged below must be scaled by the same realized fraction
    # so the ocean is not over-cooled for ice that was no longer there (#28,
    # codex; matches the _thermo_single survived-fraction treatment).  == 1 when
    # the demand is fully met or there is no basal melt (growth / no melt).
    basal_melt_demand = jnp.maximum(-dt * dh_dt_basal, 0.0)
    ocean_heat_scale = jnp.where(
        basal_melt_demand > 1e-30,
        basal_melt_m / jnp.maximum(basal_melt_demand, 1e-30),
        1.0,
    )

    # 7. Sublimation: snow first, then ice.  ``lhflx > 0`` → mass loss.
    sublim_mass_per_area = (lhflx / constants.L_s) * dt
    h_snow_after_sub, h_after_sub, snow_sub_m, ice_sub_m = consume_sublimation_from_snow_then_ice(
        sublim_mass_per_area,
        h_snow_after_melt,
        h_after_basal,
        rho_snow=config.snow.rho_snow,
        rho_ice=config.rho_ice,
        sublim_partition=config.snow.sublim_partition,
    )

    # ---- Energy-closing latent flux + single skin re-solve (#28) ----
    # The realized snow+ice sublimation mass (capped at available snow/ice by
    # consume_sublimation_from_snow_then_ice; deposition via min(..., 0) since
    # the helper grows snow and returns zero sub-depths under lhflx<0) is known
    # in ONE pass.  Build the latent CONSISTENT with it so the atmosphere latent
    # energy == L_s*realized mass and the skin is cooled by that same latent.
    # Use snow+ice TOTAL (snow sublimation is not bounded by ice h).  GUARD: only
    # rebase when the loss demand was not fully met, so thick / fully-supplied
    # cells return the bulk ``lhflx`` bitwise (no-op for validated baselines).
    realized_sublim_kg = (
        snow_sub_m * config.snow.rho_snow
        + ice_sub_m * config.rho_ice
        + jnp.minimum(lhflx / constants.L_s * dt, 0.0)
    )
    demanded_sublim_kg = lhflx / constants.L_s * dt
    sublim_capped = realized_sublim_kg < demanded_sublim_kg - 1e-30
    lhflx_realized = jnp.where(
        sublim_capped,
        constants.L_s * realized_sublim_kg / dt,
        lhflx,
    )
    if config.latent_skin_resolve:
        # Re-solve the implicit skin balance ONCE with the realized latent (same
        # closed form as _thermo_single; no iteration).  The melt/basal/sublim
        # PARTITION from the first (bulk-latent) pass is kept; the re-solve only
        # corrects the returned T_new and Q_sfc (-> lead-freeze flux).  No-op for
        # thick ice (lhflx_realized == lhflx).  sw_absorbed / lw_net / cap_dt /
        # K_cond / T_base_arr are latent-independent and reused.
        Q_sfc = sw_absorbed + lw_net - shflx - lhflx_realized
        T_implicit = (cap_dt * T_ice + Q_sfc + K_cond * T_base_arr) / (cap_dt + K_cond)
        T_trial = jnp.where(ice_mask, T_implicit, T_ice)
        T_new = jnp.where(
            ice_mask,
            jnp.clip(T_trial, config.T_ice_min, config.T_melt_surface),
            jnp.full(T_ice.shape, config.T_freeze_ocean, dtype=T_ice.dtype),
        )

    # 8. Lead freezing (open-water freeze) — only when enabled (cat 0
    #    in multi-cat).  This deposits new ice at thickness
    #    ``h_new_ice`` into the lead area.  Partial-cover cells DO
    #    contribute: ``(1 − conc)`` of the cell is open water and can
    #    refreeze.
    freeze_flux_open = jnp.maximum(-Q_sfc, 0.0)
    lead_area = jnp.clip(
        (1.0 - conc) if open_water_fraction is None else open_water_fraction,
        0.0, 1.0,
    )
    # ``dh_dt_open`` is the local lead-ice thickening rate per unit
    # lead area, driven by the destabilising surface flux.
    dh_dt_open_raw = freeze_flux_open / (config.rho_ice * config.L_f)
    dh_dt_open = dh_dt_open_raw if enable_lead_freeze else jnp.zeros_like(dh_dt_open_raw)
    # Per-grid-cell new volume contributed by lead-freezing this step.
    delta_V_lead_freeze = jnp.maximum(dh_dt_open * lead_area * dt, 0.0)

    # Update concentration and mean thickness so the lead-freeze ice
    # volume is captured even in partial-cover cells:
    #   V_after = V_before + delta_V_lead_freeze
    #   conc_after = conc_before + dconc_growth
    #   h_after   = V_after / max(conc_after, eps)
    V_before_lead = h_after_sub * conc
    V_after_lead = V_before_lead + delta_V_lead_freeze

    dconc_growth = dh_dt_open * lead_area / config.h_new_ice
    # Cap the NEW lead-ice area at the available lead: a lead can freeze over
    # completely but not beyond.  Unbounded, ``dh_dt_open*dt/h_new_ice`` can
    # exceed 1 under strong freezing flux / long dt, so cat 0 would add MORE
    # area than the lead holds and sum_k a_k > 1 reaches the ITD remap (the
    # per-cat clip to [0,1] does not bound the aggregate).  Bounding the area
    # to ``lead_area`` keeps sum_k a_k <= 1 by construction: cat 0 grows by at
    # most the open-water fraction while every other cat only shrinks.  The
    # excess freeze volume is preserved in ``V_after_lead`` and thickens the
    # new ice (``h_new = V_after_lead / conc_new``) instead of over-spreading
    # it — volume-conserving (codex).
    dconc_growth_area = jnp.minimum(dt * dconc_growth, lead_area)
    # Area retreat from ICE-volume loss only.  Melting SNOW removes the snow
    # layer, not the ice floe area, so snow_melt_m must NOT drive dconc_melt
    # (it previously did, over-shrinking concentration on snow-laden cells).
    dh_dt_melt_for_conc = (
        - (
            ice_melt_m * config.rho_ice
            + basal_melt_m * config.rho_ice
            + ice_sub_m * config.rho_ice
        )
        / (config.rho_ice * dt)
    )
    dconc_melt = jnp.minimum(dh_dt_melt_for_conc, 0.0) * conc / jnp.maximum(h, config.h_ice_min)
    conc_new = jnp.clip(conc + dconc_growth_area + dt * dconc_melt, 0.0, 1.0)

    # Enforce area<->volume consistency: if the post-thermo ice volume is
    # non-positive the column has fully ablated, so the concentration MUST be
    # zero — otherwise an h=0 / conc>0 "zombie" tile persists and keeps
    # exporting basal ocean heat (F_ocean*conc) on later steps, a systematic
    # bias.  Conversely keep conc only where real ice volume remains.
    has_vol = V_after_lead > 1e-12
    conc_new = jnp.where(has_vol, conc_new, 0.0)
    h_new = jnp.where(
        (conc_new > 1e-12) & has_vol,
        V_after_lead / jnp.maximum(conc_new, 1e-12),
        0.0,
    )

    # 9b. Snow/pond AREA-CHANGE split — done BEFORE the flooding/pond
    #     conversions so every tracer<->ice conversion below operates on the
    #     single new-area (``conc_new``) frame.  Snow depth and pond
    #     (area,depth) are per-ice-area tracers (conserved per-cell volume =
    #     depth*area).  The GROSS old-ice melt retreat (``dconc_melt`` <= 0,
    #     independent of lead-freeze growth) sheds its snow + pond water to
    #     the ocean; the surviving tracer is DILUTED onto the new ice area so
    #     newly-frozen lead ice carries zero snow/pond.  Exact per-cell mass
    #     identity: shed + surviving = conc*depth (codex #22).
    conc_surv = jnp.maximum(conc + dt * dconc_melt, 0.0)   # surviving old-ice area
    # The RETAINED snowy/ponded area cannot exceed either the surviving old
    # area (conc_surv) OR the new total ice area (conc_new): for thin ice
    # (h < h_ice_min) a step can drive conc_new -> 0 (full ablation) while
    # conc_surv stays > 0, and using conc_surv alone would keep tracer mass
    # in-column that the ablation zero-out then silently deletes (codex).
    # retained = min(conc_surv, conc_new) makes shed_area = conc - retained
    # capture ALL retired-area mass (= conc on full ablation), and
    # dilute = retained/conc_new <= 1 (no manufacture).  Exact identity:
    # shed + dilute*conc_new = (conc - retained) + retained = conc.
    retained_area = jnp.where(
        conc_new > 1e-12, jnp.minimum(conc_surv, conc_new), 0.0,
    )
    shed_area = jnp.maximum(conc - retained_area, 0.0)     # gross retreat (>=0)
    dilute = jnp.where(conc_new > 1e-12, retained_area / jnp.maximum(conc_new, 1e-12), 0.0)
    # Per-grid-cell snow + pond mass shed to the ocean from the retired area,
    # taken from the PRE-conversion tracers (post-sublimation snow; old pond).
    shed_snow_kg = shed_area * jnp.maximum(h_snow_after_sub, 0.0) * config.snow.rho_snow
    shed_pond_kg = shed_area * pond_area * pond_depth * constants.rho_water
    # Surviving tracers diluted onto the new ice area (depth-diluted; the pond
    # area FRACTION is preserved, the water VOLUME per ice-area is diluted).
    h_snow_dil = jnp.maximum(h_snow_after_sub, 0.0) * dilute
    pond_depth_dil = pond_depth * dilute

    # 10. Snow-ice flooding (white-ice formation) — on the diluted, new-area snow.
    if config.snow.enabled and config.snow.flooding:
        h_ice_after_flood, h_snow_after_flood, h_si_formed = snow_ice_flooding(
            h_new, h_snow_dil,
            rho_ice=config.rho_ice,
            rho_snow=config.snow.rho_snow,
            rho_ocean=config.rho_ocean_ref,
        )
        h_new = h_ice_after_flood
        h_snow_new = h_snow_after_flood
        delta_V_white_ice = h_si_formed
    else:
        h_snow_new = h_snow_dil
        delta_V_white_ice = jnp.zeros_like(h_new)

    # 11. Pond update — ordered BEFORE the brine budget so refrozen
    #     pond water is included in ``V_new_cat`` when computing the
    #     salt-mass change.  Pond refreeze is liquid water (m of
    #     H₂O); converting to ice-thickness via ``rho_water/rho_ice``
    #     gives mass-equivalent ice gain.
    if config.ponds.enabled:
        rain_m_liquid = jnp.maximum(
            (forcing.precip_total - forcing.precip_snow) * dt / constants.rho_water,
            0.0,
        )
        # Surface-melt thickness is in *layer* meters (snow, ice).
        # Ponds collect liquid-water equivalent — convert with the
        # density ratio before passing to ``step_ponds``.
        melt_water_m_liquid = (
            snow_melt_m * config.snow.rho_snow / constants.rho_water
            + ice_melt_m * config.rho_ice / constants.rho_water
        )
        # Snow above ice blocks pond formation.  ``step_ponds`` uses
        # the same gate internally; we mirror it here to know how
        # much melt + rain water was actually captured (so we can
        # avoid double-routing the captured water to the ocean via
        # ``fw_from_melt``).
        snow_blocks = h_snow_new > config.ponds.snow_block_threshold
        pond_captured_m_liquid = jnp.where(
            ice_mask & (~snow_blocks),
            melt_water_m_liquid + rain_m_liquid,
            0.0,
        )
        # Of the captured water, the melt portion is the one that
        # would otherwise have appeared in ``fw_from_melt``.  Split
        # by source-water fraction.
        total_input_safe = jnp.maximum(
            melt_water_m_liquid + rain_m_liquid, 1.0e-30,
        )
        melt_fraction_of_capture = melt_water_m_liquid / total_input_safe
        pond_captured_melt_m_liquid = (
            pond_captured_m_liquid * melt_fraction_of_capture
        )

        pond_area_new, pond_depth_new, drain_to_ocean_m, refreeze_m = step_ponds(
            pond_area, pond_depth_dil,
            melt_water_m=melt_water_m_liquid,
            rain_water_m=rain_m_liquid,
            ice_mask=ice_mask,
            h_snow=h_snow_new,
            T_air=forcing.T_lowest,
            dt=dt,
            drainage_timescale=config.ponds.drainage_timescale_s,
            refreeze_threshold=config.ponds.refreeze_threshold,
            pond_to_ice_max_area=config.ponds.pond_to_ice_max_area,
            depth_to_area_ratio=config.ponds.depth_to_area_ratio,
            snow_block_threshold=config.ponds.snow_block_threshold,
            refreeze_width_K=config.ponds.refreeze_width_K,
        )
        # Convert refrozen pond water [m of liquid] to equivalent ice
        # thickness [m of ice] using density.  Ice gain is
        # mass-neutral with the ocean: the water came from earlier
        # melt that we've already debited from the ice column.
        refreeze_ice_m = refreeze_m * constants.rho_water / config.rho_ice
        h_new = h_new + refreeze_ice_m
        pond_drain_to_ocean_kg_s = drain_to_ocean_m * constants.rho_water / dt
    else:
        pond_area_new = pond_area
        pond_depth_new = pond_depth_dil
        pond_drain_to_ocean_kg_s = jnp.zeros_like(h_new)
        pond_captured_melt_m_liquid = jnp.zeros_like(h_new)
        refreeze_ice_m = jnp.zeros_like(h_new)

    # 11b. Final ablation zero-out (the snow/pond area split + dilution was
    #      done at 9b on the new-area frame, before flooding/ponds).  Route
    #      the retired-area shed (9b) + any residual ice on a fully-ablated
    #      cell to the ocean as freshwater, then zero the column so no
    #      h>0/conc=0 or h=0/conc>0 zombie tile persists.
    ablated = conc_new <= 1e-12
    orphan_ice_kg = jnp.where(ablated, h_new * conc_new, 0.0) * config.rho_ice
    ablation_fw_per_cell = (
        shed_snow_kg + shed_pond_kg + orphan_ice_kg
    ) / dt
    h_new = jnp.where(ablated, 0.0, h_new)
    h_snow_new = jnp.where(ablated, 0.0, h_snow_new)
    pond_area_new = jnp.where(ablated, 0.0, pond_area_new)
    pond_depth_new = jnp.where(ablated, 0.0, pond_depth_new)

    # 12. Brine + salt budget update — uses the *final* per-cat
    #     volume (including white-ice and refrozen-pond contributions)
    #     so the salt budget is fully closed.
    if config.brine.enabled:
        V_old = h * conc
        V_new_cat = h_new * conc_new
        brine = update_salinity_and_salt_flux(
            S_ice_old=S_ice,
            V_ice_old=V_old,
            V_ice_new=V_new_cat,
            delta_V_lead_freeze=delta_V_lead_freeze,
            delta_V_white_ice=delta_V_white_ice * conc_new,
            rho_ice=config.rho_ice,
            dt=dt,
            S_lead_ice=config.brine.S_ice_new,
            S_white_ice=0.5 * config.brine.S_ocean_ref,
            # Basal congelation freezes seawater onto the ice base — a
            # salty-ice source.  Pass the per-cell basal-freeze volume
            # (per-ice-area growth × ice fraction) at the first-year
            # congelation salinity so the salt budget closes against the
            # basal-growth freshwater extraction (codex finding).
            delta_V_basal_freeze=basal_growth_m * conc,
            S_basal_ice=config.brine.S_ice_new,
            # Ice lost to sublimation leaves to the ATMOSPHERE, not the
            # ocean — its salt stays in the column (concentrating it).
            # Pass the per-cell sublimated ice volume so the brine budget
            # does not ship that salt to the ocean as a phantom flux
            # (codex finding).
            delta_V_sublim=ice_sub_m * conc,
            # Refrozen melt-pond water is FRESH new ice (the salt drained to
            # the ocean when the old ice melted): declare it as a fresh-ice
            # source so it is not mistaken for surviving old ice (which would
            # bury the old salt and under-report the ocean salt flux — codex
            # mixed melt/refreeze finding).
            delta_V_fresh_refreeze=refreeze_ice_m * conc_new,
            S_fresh_ice=config.brine.S_ice_min,
            S_ice_min=config.brine.S_ice_min,
            S_ice_max=config.brine.S_ice_max,
        )
        S_ice_new = brine.S_ice_new
        salt_flux_to_ocean = brine.salt_flux_to_ocean
    else:
        S_ice_new = S_ice
        salt_flux_to_ocean = jnp.zeros_like(h_new)

    # 13. Freshwater flux to ocean [kg/m²/s, PER-GRID-CELL-AREA].
    #
    #     Per-category-area contributions (melt of snow / ice /
    #     basal) are multiplied by ``conc`` to convert to per-cell
    #     mass loss.  Per-cell contributions (lead-freeze
    #     removal, open-water snowfall, pond drainage) flow through
    #     as-is.  When ponds are enabled, the meltwater captured
    #     into ponds is subtracted from the ocean FW channel — that
    #     mass leaves the ice column into ponds, not into the ocean
    #     (the ocean eventually receives it through ``pond_drain``).
    # Basal congelation growth (``basal_growth_m`` > 0 when F_cond >
    # F_ocean) freezes seawater onto the ice base, extracting
    # rho_ice*basal_growth_m of water from the ocean column — a
    # NEGATIVE freshwater flux to the ocean.  Omitting it makes winter
    # growth columns under-report ocean freshwater extraction; the slab
    # path captures the same effect implicitly via the net thickness
    # rate dh_dt_total = (h_new - h)/dt.
    fw_from_melt_per_cell = (
        (snow_melt_m * config.snow.rho_snow
         + ice_melt_m * config.rho_ice
         + basal_melt_m * config.rho_ice
         - basal_growth_m * config.rho_ice)
        * conc / dt
    )
    if config.ponds.enabled:
        # ``pond_captured_melt_m_liquid`` and ``pond_drain_to_ocean_kg_s``
        # are per-ice-area quantities (pond state is stored relative to
        # ice area).  Multiply by ``conc`` to convert to per-grid-cell-
        # area before mixing with the per-cell fluxes here.
        fw_from_melt_per_cell = fw_from_melt_per_cell - (
            pond_captured_melt_m_liquid * constants.rho_water * conc_new / dt
        )
        pond_drain_per_cell = pond_drain_to_ocean_kg_s * conc_new
    else:
        pond_drain_per_cell = pond_drain_to_ocean_kg_s  # zero array
    fw_from_lead_freeze_per_cell = -delta_V_lead_freeze * config.rho_ice / dt
    freshwater_to_ocean = (
        fw_from_melt_per_cell
        # NOTE: open-water snowfall is intentionally NOT included here — the
        # ocean tile delivers it via ``precip_total`` (F11; see snow step).
        + pond_drain_per_cell
        + fw_from_lead_freeze_per_cell
        # Orphaned ice/snow/pond mass from fully-ablated cells (11b) — sent
        # to the ocean as freshwater so the column mass budget closes.
        + ablation_fw_per_cell
    )

    # Ocean heat extraction [W/m², PER-GRID-CELL-AREA].
    #   * Turbulent basal heat (``F_ocean``) is per-cat-area
    #     → multiply by ``conc``.
    #   * Open-water lead freeze (``delta_V_lead_freeze`` already
    #     per-cell) releases latent heat L_f back to / removes from
    #     the ocean — sign convention: positive = ocean LOSES energy
    #     to the ice tile.  The lead-freeze contribution removes L_f
    #     per kg of ice formed from the ocean.
    ocean_heat_extraction = (
        F_ocean * conc * ocean_heat_scale
        + delta_V_lead_freeze * config.rho_ice * config.L_f / dt
        # Surplus surface-melt heat from a melt-out step warms the ocean
        # (ocean GAINS -> NEGATIVE extraction); previously this energy was
        # dropped on the floor (finding #6).
        - surface_melt_ocean_gain
    )

    # Pack diagnostics for the caller.
    return {
        "h": h_new,
        "T": T_new,
        "conc": conc_new,
        "h_snow": h_snow_new,
        "S_ice": S_ice_new,
        "pond_area": pond_area_new,
        "pond_depth": pond_depth_new,
        "alpha": alpha,
        "lw_up": lw_up,
        "shflx": shflx,
        # Realized latent (== L_s * sublim_mass_to_atmos / conc): the atmosphere
        # latent energy matches the realized sublimation mass + the skin solve
        # (#28).  Equals the bulk lhflx where the loss demand was fully met.
        "lhflx": lhflx_realized,
        "tau_x": tau_x,
        "tau_y": tau_y,
        "freshwater_to_ocean": freshwater_to_ocean,
        "salt_flux_to_ocean": salt_flux_to_ocean,
        "ocean_heat_extraction": ocean_heat_extraction,
        # Realized ice->atmosphere sublimation mass, PER-GRID-CELL [kg/m2/s]:
        # the SAME realized_sublim_kg used for lhflx_realized above (snow + ice
        # sublimated this step, capped at available snow/ice; deposition added
        # via min(lhflx/L_s*dt, 0) since the helper grows snow and returns zero
        # sub-depths under lhflx<0), weighted by the input ice fraction.  Reusing
        # the same numerator guarantees L_s*sublim_mass_to_atmos == lhflx_realized
        # * conc exactly, so the atmosphere water + energy pair (#28).
        "sublim_mass_to_atmos": realized_sublim_kg / dt * conc,
        # PER-GRID-CELL: weight by the INPUT ``conc`` (the ice area through
        # which shortwave penetrated DURING the step), the same snapshot used
        # for ``F_ocean * conc`` above.  This keeps the heat channel on a
        # single concentration basis and retains the penetrated SW at terminal
        # melt-out (where the post-step conc is 0).  F11.
        "sw_penetrated_to_ocean": sw_penetrated * conc,
        "delta_V_lead_freeze": delta_V_lead_freeze,
        "delta_V_white_ice": delta_V_white_ice,
    }


def _step_dynamic_v2(
    state: DynamicSeaIceState,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
    grid=None,
) -> tuple[DynamicSeaIceState, TileResponse]:
    """Extended dynamic sea-ice step with Tier-1 + Tier-2 physics.

    Sequence: dynamics → transport (incl. snow + salt + pond
    tracers) → thermo (snow / brine / SW / ponds) → ITD remap
    (Lipscomb 2001) → ridging → coupler response.
    """
    h = state.h_ice.data
    T_ice = state.T_ice.data
    conc = state.concentration.data
    u_ice = state.u_ice.data
    v_ice = state.v_ice.data
    s11 = state.sigma_11.data
    s22 = state.sigma_22.data
    s12 = state.sigma_12.data
    h_snow = state.h_snow.data
    S_ice = state.S_ice.data
    pond_area = state.pond_area.data
    pond_depth = state.pond_depth.data

    # Cap an already-overfilled multicat state (e.g. a restart) to
    # sum_k a_k <= 1 BEFORE the dynamics aggregate and the conc_pre snapshot,
    # so EVP/mEVP rheology and the F11 per-cell normalisation never see > 1
    # cell area.  No-op when sum <= 1; transport / ITD remap can re-introduce
    # overfill and are re-capped downstream.  Codex.
    if config.n_categories > 1:
        conc, h, h_snow, pond_depth = _cap_multicat_concentration(
            conc, h, h_snow, pond_depth)

    # Pre-step concentration snapshot — used when normalising
    # per-grid-cell fluxes to per-ice-tile fluxes so a cell that
    # FULLY melts during the step (post-step conc = 0, pre-step conc
    # > 0) still divides by a finite denominator.  Note that the
    # downstream tile blender re-multiplies by post-step f_ice, so
    # terminal-melt cells will still lose their final pulse — see
    # README in this module + the pre-existing slab-path TODO.  The
    # max(pre, post) denominator at least keeps the divide finite and
    # consistent across pre→post.
    conc_pre = conc

    # Multi-category is determined by config (grid-independent); the
    # step_sea_ice shape validation guarantees the state axis matches.
    is_multicat = config.n_categories > 1

    if is_multicat:
        h_agg, T_agg, conc_agg = aggregate_state(h, T_ice, conc)
    else:
        h_agg, T_agg, conc_agg = h, T_ice, conc

    # ---- 1. Dynamics ----
    if config.dynamics == "evp" and grid is not None:
        u_ice, v_ice, s11, s22, s12 = evp_solver(
            u_ice, v_ice, s11, s22, s12,
            h_agg, conc_agg,
            forcing.u_lowest, forcing.v_lowest,
            ocean_u, ocean_v,
            grid, dt,
            N_evp=config.N_evp, e_yield=config.e_yield,
            P_star=config.P_star, C_strength=config.C_strength,
            T_evp=config.T_evp, Delta_min=config.Delta_min,
            rho_ice=config.rho_ice, rho_air=config.rho_air_ref,
            rho_ocean=config.rho_ocean_ref,
            C_ai=config.drag_atm, C_oi=config.drag_ocean,
            h_ice_min=config.h_ice_min,
            differentiable=config.differentiable_dynamics,
        )
    elif config.dynamics == "mevp" and grid is not None:
        u_ice, v_ice, s11, s22, s12 = mevp_solver(
            u_ice, v_ice, s11, s22, s12,
            h_agg, conc_agg,
            forcing.u_lowest, forcing.v_lowest,
            ocean_u, ocean_v,
            grid, dt,
            N_mevp=config.N_mevp, e_yield=config.e_yield,
            P_star=config.P_star, C_strength=config.C_strength,
            alpha_mevp=config.alpha_mevp, beta_mevp=config.beta_mevp,
            Delta_min=config.Delta_min,
            rho_ice=config.rho_ice, rho_air=config.rho_air_ref,
            rho_ocean=config.rho_ocean_ref,
            C_ai=config.drag_atm, C_oi=config.drag_ocean,
            h_ice_min=config.h_ice_min,
            differentiable=config.differentiable_dynamics,
        )
    elif config.dynamics == "free_drift":
        u_ice, v_ice = free_drift_velocity(
            ocean_u, ocean_v,
            forcing.u_lowest, forcing.v_lowest,
            drag_ocean=config.drag_ocean, drag_atm=config.drag_atm,
            rho_air=config.rho_air_ref, rho_ocean=config.rho_ocean_ref,
        )

    # ---- 2. Transport ----
    # Advect ice tracers (h, conc, T) AND the new physics tracers
    # (h_snow, S_ice, pond_area, pond_depth) at the same velocity.
    if config.transport == "advect" and grid is not None:
        # Save the PRE-transport thickness/concentration: the new-physics
        # tracers must be advected against the SAME concentration and flux as
        # the ice, as CONSERVED INVENTORIES.  ``advect_ice_tracers`` conserves
        # ``first_arg * concentration``, so:
        #   * snow  V_snow = h_snow*a        -> advect h_snow;
        #   * salt  ∝ S_ice*h*a             -> advect (S_ice*h0), recover S=./h;
        #   * pond  V_pond = area*depth*a    -> advect (area*depth), recover.
        # Advecting S_ice / pond_area / pond_depth on their OWN (and against the
        # already-transported ``conc``) conserved the wrong quantities
        # (S_ice*a, area*a, depth*a separately) and double-advected the
        # concentration, corrupting salt and pond water (Codex transport
        # tracer-conservation finding).
        h0 = h
        conc0 = conc
        h, conc, T_ice = advect_ice_tracers(
            h, conc, T_ice, u_ice, v_ice, grid, dt,
            T_max=config.T_melt_surface,
            n_subcycles=config.transport_subcycles,
        )
        # Snow volume (h_snow is already a per-ice-area thickness): the h-slot
        # conserves h_snow*conc0 = V_snow.
        h_snow, _, _ = advect_ice_tracers(
            h_snow, conc0, T_ice, u_ice, v_ice, grid, dt,
            n_subcycles=config.transport_subcycles,
        )
        # Salinity is an intensive ice tracer like temperature: advect it
        # through the ENTHALPY (T) channel, which conserves the extensive
        # ``S_ice * h0 * conc0`` (= salt mass) and reconstructs ``S = enth/vol``
        # with the channel's monotone PPM.  The channel's defensive clip is set
        # to a WIDE, NON-BINDING range (not the physical ``S_ice_max``): a
        # monotone-PPM ratio can overshoot the input max by a hair (e.g. 12.003
        # from a 12.0 cap), and clipping that at the physical cap HERE would
        # delete salt with no accounting.  Instead we let the tiny overshoot
        # pass through; the brine budget downstream applies the physical
        # [S_ice_min, S_ice_max] clamp CONSERVATIVELY (its residual is routed
        # to the ocean via salt_flux = salt_old - salt_stored).  Open water gets
        # ``S_ice_min`` (ice-free cells carry no salt, not the freezing temp).
        # ``_S_TRANSPORT_LO/HI`` are numerical safety bounds, never physically
        # binding for sea-ice salinity in [0, ~12].
        _S_TRANSPORT_LO, _S_TRANSPORT_HI = -10.0, 100.0
        _, _, S_ice = advect_ice_tracers(
            h0, conc0, S_ice, u_ice, v_ice, grid, dt,
            T_ice_min=_S_TRANSPORT_LO,
            T_freeze_ocean=config.brine.S_ice_min,
            T_max=_S_TRANSPORT_HI,
            n_subcycles=config.transport_subcycles,
        )
        if config.ponds.enabled:
            # Pond water: advect the pond-water thickness (area*depth), recover
            # (area, depth) from the conserved per-cell volume.
            pond_thick, _, _ = advect_ice_tracers(
                pond_area * pond_depth, conc0, T_ice, u_ice, v_ice, grid, dt,
                n_subcycles=config.transport_subcycles,
            )
            pond_area, pond_depth = _pond_area_depth_from_volume(
                jnp.maximum(pond_thick, 0.0) * conc, conc, config)

    # Multi-category aggregate-area invariant: independent per-category
    # advection (and overfilled restarts) do not guarantee ``sum_k a_k <= 1``;
    # restore it (volume / snow / pond / salt conserving) BEFORE the
    # thermo/brine/ridging budgets so they act on at most one grid-cell area,
    # and again AFTER the ITD remap (below), since the incremental remap can
    # re-split a compacted category across bins and reintroduce sum > 1.  The
    # per-process exchange (built inside ``_thermo_v2``) thus uses the
    # post-compaction (sum<=1) concentration.  Applied UNCONDITIONALLY for
    # multicat (no-op when sum<=1) so an overfilled restart is handled without
    # transport.  NOTE: a uniform mechanical thickening, NOT a bin-accurate ITD
    # redistribution — realistic sub-CFL overfill is <<1% and stays within its
    # category; bin-accurate large-overfill handling is the deferred item.
    if is_multicat:
        conc, h, h_snow, pond_depth = _cap_multicat_concentration(
            conc, h, h_snow, pond_depth)

    # ---- 3. Pre-thermo per-category state (save for ITD remap) ----
    h_old = h
    conc_old = conc

    # ---- 4. Thermodynamics per category ----
    if is_multicat:
        n_cat = h.shape[-1]
        open_water_agg = jnp.clip(1.0 - jnp.sum(conc, axis=-1), 0.0, 1.0)

        h_list = []
        T_list = []
        conc_list = []
        snow_list = []
        S_list = []
        pa_list = []
        pd_list = []
        salt_flux_list = []
        fw_flux_list = []
        ocean_heat_list = []
        sublim_mass_list = []
        sw_pen_list = []
        delta_lead_list = []
        delta_white_list = []
        shflx_aggsum_components = []
        tau_x_components = []
        tau_y_components = []
        for k in range(n_cat):
            result = _thermo_v2(
                h[..., k], T_ice[..., k], conc[..., k],
                h_snow[..., k], S_ice[..., k],
                pond_area[..., k], pond_depth[..., k],
                forcing, ocean_sst, config, U_min, dt,
                open_water_fraction=open_water_agg,
                enable_lead_freeze=(k == 0),
            )
            h_list.append(result["h"])
            T_list.append(result["T"])
            conc_list.append(result["conc"])
            snow_list.append(result["h_snow"])
            S_list.append(result["S_ice"])
            pa_list.append(result["pond_area"])
            pd_list.append(result["pond_depth"])
            salt_flux_list.append(result["salt_flux_to_ocean"])
            fw_flux_list.append(result["freshwater_to_ocean"])
            ocean_heat_list.append(result["ocean_heat_extraction"])
            sublim_mass_list.append(result["sublim_mass_to_atmos"])
            sw_pen_list.append(result["sw_penetrated_to_ocean"])
            delta_lead_list.append(result["delta_V_lead_freeze"])
            delta_white_list.append(result["delta_V_white_ice"])
            shflx_aggsum_components.append(result["shflx"] * result["conc"])
            tau_x_components.append(result["tau_x"] * result["conc"])
            tau_y_components.append(result["tau_y"] * result["conc"])

        h = jnp.stack(h_list, axis=-1)
        T_ice = jnp.stack(T_list, axis=-1)
        conc = jnp.stack(conc_list, axis=-1)
        h_snow = jnp.stack(snow_list, axis=-1)
        S_ice = jnp.stack(S_list, axis=-1)
        pond_area = jnp.stack(pa_list, axis=-1)
        pond_depth = jnp.stack(pd_list, axis=-1)

        # Ice->ocean EXCHANGE fluxes (freshwater / heat / salt / SW
        # penetration): ``_thermo_v2`` already multiplied the per-cat-area
        # melt/heat terms by ``conc``, so summing across categories gives the
        # PER-GRID-CELL budget directly.  These are returned per-grid-cell
        # (NOT divided by the ice fraction): ``blend_tiles`` weights ice->ocean
        # exchange by ``f_water``, so the full per-cell budget is delivered for
        # any concentration history without a 1/conc normalisation (F11).
        sum_conc_pre = jnp.sum(conc_pre, axis=-1, keepdims=False)
        sum_conc_post = jnp.sum(conc, axis=-1, keepdims=False)
        # ``sum_conc_safe`` (== max(pre, post)) is the LATENT basis only: it
        # matches the single-cat ``conc_basis`` contract so resp.lhflx *
        # max(pre,post) == L_s * sublim_mass_total identically across paths
        # (sublim_mass_total is on the INPUT-conc basis).  The SH/STRESS
        # numerators are kept per-grid-cell and divided by ``conc_agg`` at the
        # response build (findings #9 + #4), NOT by this latent basis.
        sum_conc_safe = jnp.maximum(
            jnp.maximum(sum_conc_pre, sum_conc_post), 1e-30,
        )
        fw_per_cat = jnp.stack(fw_flux_list, axis=-1)
        heat_per_cat = jnp.stack(ocean_heat_list, axis=-1)
        fw_flux_total = jnp.sum(fw_per_cat, axis=-1)
        ocean_heat_total = jnp.sum(heat_per_cat, axis=-1)
        salt_flux_total = jnp.sum(jnp.stack(salt_flux_list, axis=-1), axis=-1)
        # Realized ice->atmosphere sublimation mass is already PER-GRID-CELL
        # (weighted by the input conc inside _thermo_v2); sum across categories.
        sublim_mass_total = jnp.sum(jnp.stack(sublim_mass_list, axis=-1), axis=-1)
        # SW penetrated to the ocean is already PER-GRID-CELL (weighted by the
        # input conc inside _thermo_v2); sum across categories directly.
        sw_pen_total = jnp.sum(jnp.stack(sw_pen_list, axis=-1), axis=-1)
        ocean_heat_total = ocean_heat_total - sw_pen_total

        # Atmosphere-coupler SH / stress: keep the PER-GRID-CELL numerators
        # (Sum_k flux_k * conc_post_k) here and defer the per-ice-tile division
        # to the response build, where it is divided by ``conc_agg`` -- the SAME
        # aggregated concentration ``f_ice`` later multiplies by.  This makes the
        # delivered flux ``shflx_resp * f_ice`` recover the per-cell total
        # EXACTLY even when ITD remap / ridging change the aggregate area between
        # the thermo step and the response (findings #9 + #4).  ``sum_conc_post``
        # equals ``conc_agg`` only when no area-changing ridging fires.
        shflx_pergrid_num = jnp.sum(
            jnp.stack(shflx_aggsum_components, axis=-1), axis=-1)
        tau_x_pergrid_num = jnp.sum(
            jnp.stack(tau_x_components, axis=-1), axis=-1)
        tau_y_pergrid_num = jnp.sum(
            jnp.stack(tau_y_components, axis=-1), axis=-1)
        # LATENT: derive the per-ice-area latent from the REALIZED sublimation
        # MASS (same INPUT-conc basis as ``sublim_mass_total``), NOT from the
        # post-thermo conc-weighted result["lhflx"] -- otherwise melt/retreat/
        # clamp cells under-report the latent and break
        # resp.lhflx*sum_conc == L_s*sublim_mass_total (codex).  This keeps the
        # atmosphere latent ENERGY paired to the moisture MASS on one basis.
        lhflx_resp = constants.L_s * sublim_mass_total / sum_conc_safe

    else:
        result = _thermo_v2(
            h, T_ice, conc, h_snow, S_ice, pond_area, pond_depth,
            forcing, ocean_sst, config, U_min, dt,
        )
        h = result["h"]
        T_ice = result["T"]
        conc = result["conc"]
        h_snow = result["h_snow"]
        S_ice = result["S_ice"]
        pond_area = result["pond_area"]
        pond_depth = result["pond_depth"]
        # Ice->ocean EXCHANGE fluxes (freshwater / heat / salt) returned by
        # ``_thermo_v2`` are PER-GRID-CELL (per-cat-area terms already
        # multiplied by ``conc`` inside the kernel).  Returned per-grid-cell:
        # ``blend_tiles`` weights ice->ocean exchange by ``f_water`` (F11), so
        # the full per-cell budget is delivered for any concentration history
        # without a 1/conc normalisation that could blow up at melt-out.
        salt_flux_total = result["salt_flux_to_ocean"]
        fw_flux_total = result["freshwater_to_ocean"]
        sublim_mass_total = result["sublim_mass_to_atmos"]
        # SW penetration is already PER-GRID-CELL (weighted by the input conc
        # inside _thermo_v2); subtract directly from the per-cell ocean heat.
        ocean_heat_total = (
            result["ocean_heat_extraction"]
            - result["sw_penetrated_to_ocean"]
        )
        # Atmosphere fluxes stay per-ice-tile (blend_tiles re-multiplies by
        # the post-step f_ice).
        shflx_resp = result["shflx"]
        # LATENT on the realized-MASS basis, using the SAME conc basis as the
        # multicat path -- max(PRE-DYNAMICS conc_pre, post-thermo conc) -- so
        # single-cat and multicat expose an IDENTICAL latent<->mass invariant to
        # TileResponse consumers regardless of transport: resp.lhflx *
        # max(conc_pre, conc_post) == L_s * surface_mass_flux.  (Using the
        # post-transport thermo-input conc here would diverge from the multicat
        # ``sum_conc_safe`` basis under transport='advect'.)  #28, codex.
        conc_basis = jnp.maximum(jnp.maximum(conc_pre, conc), 1e-30)
        lhflx_resp = constants.L_s * sublim_mass_total / conc_basis
        tau_x_resp = result["tau_x"]
        tau_y_resp = result["tau_y"]

    # ---- 5. ITD remap ----
    if is_multicat:
        # Cap the aggregate area BEFORE the ITD remap so neither Lipscomb nor
        # the simple remap ever consumes sum_k a_k > 1.  With the lead-freeze
        # area now bounded inside the thermo kernels this is a no-op for
        # thermo-generated overfill, but it still guards against an overfilled
        # restart / transport remainder reaching the remapper (volume / snow /
        # pond / salt conserving; no-op when sum <= 1).  Codex.
        conc, h, h_snow, pond_depth = _cap_multicat_concentration(
            conc, h, h_snow, pond_depth)
        n_cat = h.shape[-1]
        # Volume of snow per cat (h_snow * a) for remap.
        V_snow_cat = h_snow * conc
        if config.itd_remap == "lipscomb2001":
            remap = lipscomb_2001_remap(
                h_old=h_old, a_old=conc_old,
                h_new=h, a_new=conc,
                n_cat=n_cat, dt=dt,
                T_new=T_ice, S_new=S_ice,
                V_snow_new=V_snow_cat,
                V_pond_new=pond_area * pond_depth * conc,
                T_max=config.T_melt_surface,
            )
            h = remap["h"]
            conc = remap["a"]
            T_ice = remap["T"]
            S_ice = remap["S"]
            # Recover h_snow from V_snow and conc.
            conc_safe = jnp.maximum(conc, 1e-12)
            h_snow = jnp.where(conc > 1e-12, remap["V_snow"] / conc_safe, 0.0)
            # Recover per-ice-area pond (area, depth) from the conserved
            # per-cell pond volume (V-conserving even at the area cap).
            pond_area, pond_depth = _pond_area_depth_from_volume(
                remap["V_pond"], conc, config)
        elif config.itd_remap == "simple":
            # Legacy linear remap for h, a, T only (valid only when there are
            # no snow/brine/pond tracers — enforced by the multicat guard in
            # ``step_sea_ice``).
            h, conc, T_ice = linear_remap(
                h_old, conc_old, h, conc, n_cat, T_new=T_ice,
                T_max=config.T_melt_surface,
            )
        else:
            # Unreachable: step_sea_ice validates itd_remap up front.  Guard
            # against a silent tracer-dropping fallback for an unknown scheme.
            raise ValueError(
                f"Unknown config.itd_remap={config.itd_remap!r}; expected "
                "'simple' or 'lipscomb2001'."
            )

        # The incremental ITD remap can re-split a category across bins and
        # reintroduce sum_k a_k > 1; re-cap BEFORE ridging so the ridging kernel
        # (donor area, snow/pond drainage, freshwater fluxes) never sees an
        # overfilled column.  No-op when sum <= 1.  Codex.
        conc, h, h_snow, pond_depth = _cap_multicat_concentration(
            conc, h, h_snow, pond_depth)

    # ---- 6. Ridging ----
    if config.ridging.enabled and is_multicat and grid is not None:
        closing_rate = _closing_rate_from_velocity(
            u_ice, v_ice, grid, cap=config.ridging.closing_rate_max,
        )
        ridge_result = apply_ridging(
            conc, h, h_snow * conc, S_ice, closing_rate,
            n_cat=h.shape[-1], dt=dt,
            # Carry per-cell pond volume so ridging transports/drains pond
            # water conservatively instead of silently dropping it (codex).
            V_pond_cat=pond_area * pond_depth * conc,
            e_star=config.ridging.e_star,
            mu_rdg=config.ridging.mu_rdg,
            H_star=config.ridging.H_star,
            snow_fraction_retained=config.ridging.snow_fraction_retained,
        )
        conc = ridge_result["a"]
        h = ridge_result["h"]
        conc_safe = jnp.maximum(conc, 1e-12)
        h_snow = jnp.where(conc > 1e-12, ridge_result["V_snow"] / conc_safe, 0.0)
        S_ice = ridge_result["S_ice"]
        # Reconstruct per-ice-area pond (area, depth) from the post-ridging
        # per-cell pond volume (donor pond water was drained to the ocean).
        pond_area, pond_depth = _pond_area_depth_from_volume(
            ridge_result["V_pond"], conc, config)
        # Snow + pond shed by ridging → ocean as freshwater.  Both
        # ``snow_to_ocean`` and ``pond_to_ocean`` are PER-GRID-CELL-AREA fluxes
        # (built from V_snow_cat / V_pond_cat = tracer*conc).  ``fw_flux_total``
        # is also per-grid-cell (F11), so they add directly — no ice-fraction
        # normalisation; ``blend_tiles`` weights the combined per-cell
        # freshwater by ``f_water``.
        fw_flux_total = (
            fw_flux_total
            + ridge_result["snow_to_ocean"]
            + ridge_result["pond_to_ocean"]
        )

    # Final aggregate-area invariant: the incremental ITD remap can re-split a
    # compacted/grown category across bins and reintroduce sum_k a_k > 1, so
    # re-cap the post-remap/post-ridging state (volume / snow / pond / salt
    # conserving; no-op when sum <= 1).  Codex.
    if is_multicat:
        conc, h, h_snow, pond_depth = _cap_multicat_concentration(
            conc, h, h_snow, pond_depth)

    # ---- 7. Build response ----
    if is_multicat:
        h_agg, T_agg, conc_agg = aggregate_state(h, T_ice, conc)
        # Per-ice-tile SH / stress: divide the per-grid-cell numerators by the
        # FINAL aggregated concentration (the one ``f_ice`` multiplies by), so
        # ``resp * f_ice`` recovers the per-cell total exactly even after ITD
        # remap / ridging changed the aggregate area (findings #9 + #4).
        conc_agg_safe = jnp.maximum(conc_agg, 1e-30)
        shflx_resp = shflx_pergrid_num / conc_agg_safe
        tau_x_resp = tau_x_pergrid_num / conc_agg_safe
        tau_y_resp = tau_y_pergrid_num / conc_agg_safe
    else:
        h_agg, T_agg, conc_agg = h, T_ice, conc

    q_sfc = saturation_mixing_ratio_ice(T_agg, forcing.p_surface)
    lw_up_total = (
        config.emissivity_ice * constants.sigma_sb * T_agg ** 4
        + (1.0 - config.emissivity_ice) * forcing.lw_down
    )
    # Tile albedo the atmosphere sees.  ``maykut_untersteiner`` and
    # ``delta_eddington`` are NONLINEAR in thickness / snow / pond state, so
    # evaluating the albedo on the AREA-AGGREGATED state (α(mean state)) is NOT
    # the area-mean albedo (mean(α_k)) in multi-category mode — a thin+thick mix
    # biased the tile albedo by tens of W/m² (e.g. 0.5/0.5 area, h=[0.05, 2.0]:
    # MU 0.700 vs the correct 0.461 → ~72 W/m² at SW=300; codex finding).
    # Compute the SW kernel PER CATEGORY and area-weight the resulting albedo so
    # the coupler's f_ice blend receives the physically correct mean reflectance.
    # The ``constant`` scheme is linear in state so per-cat == aggregate (the
    # weighted mean of a constant is the constant); this fix is exact for it too.
    if is_multicat:
        sw_cat = compute_ice_sw(
            forcing.sw_down[..., None], T_ice, h,
            h_snow, pond_area, pond_depth,
            scheme=config.shortwave_scheme,
            albedo_const=config.albedo_ice,
            sw_transmittance_const=config.sw_transmittance_const,
        )
        alpha_resp = (
            jnp.sum(sw_cat.albedo_eff * conc, axis=-1)
            / jnp.maximum(conc_agg, 1e-12)
        )
    else:
        sw_agg = compute_ice_sw(
            forcing.sw_down, T_agg, h_agg,
            h_snow, pond_area, pond_depth,
            scheme=config.shortwave_scheme,
            albedo_const=config.albedo_ice,
            sw_transmittance_const=config.sw_transmittance_const,
        )
        alpha_resp = sw_agg.albedo_eff

    # Ice → ocean back-reaction stress.  Per-ice-tile (no ``* conc_agg``):
    # blend_tiles applies the single area weight ``f_ice``.  F11.
    du_oi = ocean_u - u_ice
    dv_oi = ocean_v - v_ice
    speed_oi = jnp.sqrt(du_oi ** 2 + dv_oi ** 2 + 1e-10)
    tau_oi_x = config.rho_ocean_ref * config.drag_ocean * speed_oi * du_oi
    tau_oi_y = config.rho_ocean_ref * config.drag_ocean * speed_oi * dv_oi
    ocean_stress_x = -tau_oi_x
    ocean_stress_y = -tau_oi_y

    response = TileResponse(
        T_sfc=T_agg,
        albedo=alpha_resp,
        emissivity=jnp.full(h_agg.shape, config.emissivity_ice, dtype=h_agg.dtype),
        z0=jnp.full(h_agg.shape, config.z0_ice, dtype=h_agg.dtype),
        q_surface=q_sfc,
        shflx=shflx_resp,
        lhflx=lhflx_resp,
        tau_x=tau_x_resp,
        tau_y=tau_y_resp,
        lw_up=lw_up_total,
        u_ocean_sfc=u_ice,
        v_ocean_sfc=v_ice,
        co2_flux=jnp.zeros_like(h_agg),
        freshwater_flux=fw_flux_total,
        ocean_heat_extraction=ocean_heat_total,
        ocean_stress_x=ocean_stress_x,
        ocean_stress_y=ocean_stress_y,
        # Realized (capped) ice->atmosphere sublimation mass, PER-GRID-CELL, so
        # blend_tiles weights it by f_water and the atmosphere water budget
        # matches the state through melt-out (#28, codex).
        surface_mass_flux=sublim_mass_total,
        salt_flux=salt_flux_total,
        # Aggregate concentration at the THERMO time level (post-transport,
        # pre-thermo ``conc_old``): the ice area the atmospheric fluxes were
        # integrated over this step.  Forced-ocean drivers partition the
        # open-water forcing with (1 - A) at THIS level (codex r5 #1 — the
        # pre-call concentration is one transport substep stale under
        # transport='advect').
        ice_concentration_thermo=jnp.clip(
            (jnp.sum(conc_old, axis=-1) if is_multicat else conc_old),
            0.0, 1.0),
    )

    new_state = DynamicSeaIceState(
        h_ice=state.h_ice.replace(data=h),
        T_ice=state.T_ice.replace(data=T_ice),
        concentration=state.concentration.replace(data=conc),
        u_ice=state.u_ice.replace(data=u_ice),
        v_ice=state.v_ice.replace(data=v_ice),
        sigma_11=state.sigma_11.replace(data=s11),
        sigma_22=state.sigma_22.replace(data=s22),
        sigma_12=state.sigma_12.replace(data=s12),
        h_snow=state.h_snow.replace(data=h_snow),
        S_ice=state.S_ice.replace(data=S_ice),
        pond_area=state.pond_area.replace(data=pond_area),
        pond_depth=state.pond_depth.replace(data=pond_depth),
    )
    return new_state, response
