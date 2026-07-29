"""Simple ocean modes for idealized experiments.

Three modes:
- Fixed SST: prescribed constant or spatial map, no state evolution.
- Slab ocean: single mixed-layer with energy balance and optional Q-flux.
- Two-layer slab: mixed layer + deep layer with vertical mixing and
  optional deep-layer restoring.

Factory ``make_ocean`` dispatches on ``SimpleOceanConfig.mode`` and returns
a uniform step function compatible with the coupler.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.core.bulk_flux import apply_gustiness, ocean_surface_q_sat
from legoesm.ocean.eos import FreezingPointConfig, slab_freeze_point_K


# ============================================================================
# Configuration
# ============================================================================

class SimpleOceanConfig(NamedTuple):
    """Configuration for simplified ocean modes."""
    mode: str = "fixed"              # "fixed" | "slab" | "two_layer"
    # Fixed SST
    sst_constant: float = 300.0      # Global constant SST [K]
    # Slab ocean
    h_mix: float = 50.0              # Mixed-layer depth [m]
    rho_ocean: float = constants.rho_ocean
    c_ocean: float = constants.c_sw
    Q_flux: float = 0.0              # Prescribed OHT convergence [W/m2]
    # Optional spatially+seasonally varying q-flux (ocean-heat-transport
    # convergence) CLIMATOLOGY file [W/m2], sign +INTO the mixed layer (same
    # sign as the scalar Q_flux above).  Empty (default) => the scalar Q_flux
    # is used everywhere (byte-identical).  When set, the coupled driver loads
    # the (12, ...) monthly climatology, regrids to the ocean grid, and threads
    # the calendar-month-interpolated map into the slab step each coupling
    # interval as the ``q_flux`` step argument (see coupled_esm_driver.
    # _step_ocean).  Standard CMIP slab calibration: q_flux(x, month) =
    # -(monthly-mean net downward surface heat flux from an AMIP-SST-forced
    # run), so the slab reproduces the observed SST climatology.
    q_flux_path: str = ""
    albedo_ocean: float = 0.06
    emissivity_ocean: float = 0.97
    Cd_ocean: float = 1.5e-3         # Drag coefficient
    Ch_ocean: float = 1.5e-3         # Heat transfer coefficient
    U_min: float = 1.0               # Numerical wind floor [m/s]
    # Sub-grid convective gustiness floor [m/s] for the air-sea bulk fluxes:
    # |U|_eff = sqrt(|U|^2 + gustiness^2).  DEFAULT 1.0 (= the old numerical
    # floor): in an INTERACTIVE slab/two-layer ocean the equilibrium evaporation
    # is ENERGY-limited, not wind-limited — a larger gustiness boosts E only
    # transiently, then the latent cooling drops SST -> lowers q_sat(SST) ->
    # E settles LOWER and the slab cold-drifts (measured: gustiness=5 gave
    # hfls 35->31, CWV 15->13, SST drift -20 K/yr).  Keep ~1.0 for the coupled
    # slab; gustiness ~5 (Wing 2018) is correct only for PRESCRIBED-SST paths
    # (SCM rce_surface_flux / AMIP) where the SST cannot cool away.  See the
    # CAM surface-energy audit: the coupled dry column is an LW/cloud-opacity
    # (atmospheric emissivity) problem, not a surface-wind problem.
    gustiness: float = 1.0
    # Air-sea turbulent bulk-flux algorithm for the slab/two-layer heat budget:
    # "constant" (neutral Ch_ocean + gustiness floor; DEFAULT, byte-identical) |
    # "coare3" | "large_yeager".  The MOST schemes add the free-convection
    # velocity scale w* (computed self-consistently from the buoyancy flux), so
    # the slab heat LOSS matches the heat the atmosphere surface layer GAINS when
    # both are set to the same scheme (interface energy consistency; run_coupled
    # wires --surface-bulk-scheme into both).  z_ref / n_iter use the
    # ``compute_most_fluxes`` defaults (10 m / 5), matching SurfaceLayerConfig.
    bulk_scheme: str = "constant"
    # COARE convective-gustiness BL depth z_i [m] for the slab heat budget's
    # coare3/large_yeager fluxes.  None (default) = scheme-native (600 m for
    # coare3, off otherwise — AeroBulk parity); explicit 0.0 = off.  Kept
    # consistent with the atmosphere SurfaceLayerConfig.gustiness_w_zi by
    # run_coupled.
    gustiness_w_zi: float | None = None
    # Thermodynamic constants set for the slab's coare3/large_yeager fluxes
    # (#762): "legoesm" (default) = constant L_v / dry c_pd; "aerobulk" =
    # NEMO/AeroBulk/COARE parity (SST-dependent L_vap(T_sfc), moist
    # cp_air(q_atm)).  Kept consistent with the atmosphere
    # SurfaceLayerConfig.thermo_convention by run_coupled.
    thermo_convention: str = "legoesm"
    T_freeze: float = constants.T_freeze_ocean
    # Seawater freezing-point (liquidus) scheme for the freeze clamp below.
    # "constant" (default) => the fixed T_freeze above, byte-identical.  The slab
    # carries no salinity, so a liquidus scheme is evaluated at the reference
    # ocean salinity constants.S_ocean_ref (see _step).  MED-1.
    freezing: FreezingPointConfig = FreezingPointConfig()
    # Two-layer additions
    h_deep: float = 200.0            # Deep layer depth [m]
    k_mix: float = 1.0e-4            # Vertical mixing coefficient [m2/s]
    restore_deep: bool = False       # Restore deep layer toward T_deep_ref?
    T_deep_ref: float = 278.0        # Deep restoring target [K]
    tau_deep: float = 365.25 * 86400.0  # Restoring timescale [s] (1 year)


# ============================================================================
# State
# ============================================================================

class SlabOceanState(NamedTuple):
    """State for slab and two-layer ocean modes.

    ``Q_freeze`` is a diagnostic field [W/m²] populated when the
    freezing clamp fires.  It captures the energy that would have
    pushed SST below ``T_freeze`` and represents the latent heat
    released to ice formation; consumers wishing to close the
    sea-ice freezing budget should integrate this term.  Defaults
    to ``None`` so legacy callers that ignore freezing energy keep
    working unchanged.
    """
    T_sfc: Field    # SST [K], shape (6, n, n)
    T_deep: Field   # Deep layer temperature [K], shape (6, n, n)
    Q_freeze: Field | None = None  # latent-heat-of-fusion flux at freezing clamp


# ============================================================================
# Initialization
# ============================================================================

DIMS_2D = ("face", "x", "y")


def init_slab_state(
    shape: tuple[int, ...],
    T_sfc_init: float = 300.0,
    T_deep_init: float = 278.0,
) -> SlabOceanState:
    """Initialize a SlabOceanState with uniform temperatures.

    ``Q_freeze`` is initialised to a zero Field so the pytree shape
    is invariant across timesteps (matches ``LakeState.Q_freeze``
    convention).  Step functions update the values in place via
    ``state.Q_freeze.replace(data=...)``.
    """
    return SlabOceanState(
        T_sfc=Field(
            jnp.full(shape, T_sfc_init),
            name="T_sfc", dims=DIMS_2D, units="K",
        ),
        T_deep=Field(
            jnp.full(shape, T_deep_init),
            name="T_deep", dims=DIMS_2D, units="K",
        ),
        Q_freeze=Field(
            jnp.zeros(shape),
            name="Q_freeze", dims=DIMS_2D, units="W/m2",
        ),
    )


# ============================================================================
# Slab ocean physics
# ============================================================================

def _ocean_turbulent_fluxes(
    T_sfc: jnp.ndarray,
    q_sfc: jnp.ndarray,
    forcing: AtmToSurface,
    config: SimpleOceanConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sensible + latent surface fluxes (positive upward) for the slab/two-layer
    ocean heat budget.

    Mirrors the atmosphere surface-layer dispatch
    (``turbulence/surface_layer.compute_surface_fluxes``) so the turbulent heat
    LEAVING the slab matches the heat ENTERING the atmosphere when both are set
    to the same ``bulk_scheme`` (interface energy consistency).  The slab has no
    surface current, so the relative wind equals the atmospheric wind.

    * ``"constant"`` — neutral coefficient ``Ch_ocean`` with the sub-grid
      gustiness floor (the legacy path; byte-identical default).
    * ``"coare3"`` / ``"large_yeager"`` — stability-dependent MOST.  MOST
      derives its own convective gustiness (w*) from the buoyancy flux, so
      ``apply_gustiness`` is NOT applied here (it would double-count).
    """
    rho = forcing.rho_lowest
    if config.bulk_scheme == "constant":
        wind = apply_gustiness(
            forcing.u_lowest, forcing.v_lowest, config.gustiness,
        )
        shflx = rho * constants.c_pd * config.Ch_ocean * wind * (T_sfc - forcing.T_lowest)
        lhflx = rho * constants.L_v * config.Ch_ocean * wind * (q_sfc - forcing.q_lowest)
        return shflx, lhflx
    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        # "most" = generic iterative MOST with fixed roughness (no Charnock);
        # "coare3"/"large_yeager" = ocean-specific stability-dependent MOST.
        from legoesm.core.bulk_flux import compute_most_fluxes
        _tx, _ty, shflx, lhflx, _ust = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest, T_sfc, q_sfc, rho,
            scheme=config.bulk_scheme,
            gustiness_w_zi=getattr(config, "gustiness_w_zi", None),
            thermo_convention=getattr(config, "thermo_convention", "legoesm"),
        )
        return shflx, lhflx
    raise ValueError(
        f"Unknown SimpleOceanConfig.bulk_scheme {config.bulk_scheme!r}; "
        f"expected 'constant', 'most', 'coare3', or 'large_yeager'."
    )


def _slab_step(
    state: SlabOceanState,
    forcing: AtmToSurface,
    config: SimpleOceanConfig,
    dt: float,
    q_flux: jnp.ndarray | None = None,
) -> tuple[SlabOceanState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single mixed-layer energy balance step.

    ``q_flux`` (optional, [W/m2], sign +INTO the mixed layer) is the
    spatially+seasonally varying ocean-heat-transport convergence for this
    step; ``None`` (default) falls back to the scalar ``config.Q_flux``, so a
    run without a q-flux climatology is byte-identical.  Threaded per step
    (not baked into the config) so the climatology can vary seasonally without
    a recompile — the slab step runs eagerly on the host.
    """
    T_sfc = state.T_sfc.data

    # Surface humidity: saturated, on the same thermodynamic convention as the
    # flux formation (#762) — Goff under aerobulk+MOST, else Tetens (default
    # legoesm / 'constant' scheme byte-identical).  Matches the coupler ocean
    # tile so the air-sea interface q_sfc is single-valued.
    q_sfc = ocean_surface_q_sat(
        T_sfc, forcing.p_surface,
        thermo_convention=getattr(config, "thermo_convention", "legoesm"),
        bulk_scheme=config.bulk_scheme)

    # Bulk turbulent fluxes (positive upward); scheme-consistent with the
    # atmosphere surface layer (see _ocean_turbulent_fluxes).
    shflx, lhflx = _ocean_turbulent_fluxes(T_sfc, q_sfc, forcing, config)

    # Radiation
    sw_net = (1.0 - config.albedo_ocean) * forcing.sw_down
    lw_net = (config.emissivity_ocean * forcing.lw_down
              - config.emissivity_ocean * constants.sigma_sb * T_sfc ** 4)

    # Energy balance.  Sign convention: all terms are the net heat flux INTO
    # the mixed layer [W/m2].  sw_net/lw_net are +into-ocean (absorbed SW,
    # net LW); shflx/lhflx are +upward (out of the ocean), hence subtracted;
    # q_flux_eff is the prescribed ocean-heat-transport convergence, +into the
    # mixed layer (same sign as the scalar config.Q_flux it replaces).
    q_flux_eff = config.Q_flux if q_flux is None else q_flux
    C_mix = config.rho_ocean * config.c_ocean * config.h_mix
    dT_dt = (sw_net + lw_net - shflx - lhflx + q_flux_eff) / C_mix
    T_sfc_trial = T_sfc + dt * dT_dt

    # Freezing clamp.  When the trial SST is below T_freeze, the energy
    # the column would have lost to push it below freezing is the heat
    # that *should* go into latent heat of fusion (creating sea ice).
    # Diagnose this as ``Q_freeze`` on the new state instead of letting
    # the clamp silently destroy the energy.  The two-layer lake uses
    # the same pattern (two_layer_lake.py:84-99).  Coupler-conservation
    # audit F6.  Freeze point: shared single owner ``eos.slab_freeze_point_K``
    # ("constant" default returns config.T_freeze byte-identical; a liquidus
    # scheme is evaluated at constants.S_ocean_ref -- no prognostic S).  MED-1.
    T_freeze_eff = slab_freeze_point_K(config.T_freeze, config.freezing.scheme)
    T_sfc_new = jnp.maximum(T_sfc_trial, T_freeze_eff)
    Q_freeze = C_mix * jnp.maximum(T_freeze_eff - T_sfc_trial, 0.0) / dt

    new_state = SlabOceanState(
        T_sfc=state.T_sfc.replace(data=T_sfc_new),
        T_deep=state.T_deep,  # unchanged in slab mode
        Q_freeze=(state.Q_freeze.replace(data=Q_freeze)
                  if state.Q_freeze is not None
                  else Field(data=Q_freeze, name="Q_freeze",
                             dims=state.T_sfc.dims, units="W/m2")),
    )
    u_sfc = jnp.zeros_like(T_sfc)
    v_sfc = jnp.zeros_like(T_sfc)
    return new_state, T_sfc_new, u_sfc, v_sfc


# ============================================================================
# Two-layer ocean physics
# ============================================================================

def _two_layer_step(
    state: SlabOceanState,
    forcing: AtmToSurface,
    config: SimpleOceanConfig,
    dt: float,
    q_flux: jnp.ndarray | None = None,
) -> tuple[SlabOceanState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Two-layer slab ocean: mixed layer + deep layer.

    ``q_flux`` (optional, [W/m2], +INTO the mixed layer) — see ``_slab_step``;
    ``None`` falls back to the scalar ``config.Q_flux`` (byte-identical).
    """
    T_sfc = state.T_sfc.data
    T_deep = state.T_deep.data

    # Surface humidity: saturated, on the same thermodynamic convention as the
    # flux formation (#762) — Goff under aerobulk+MOST, else Tetens (default
    # legoesm / 'constant' scheme byte-identical).  Matches the coupler ocean
    # tile so the air-sea interface q_sfc is single-valued.
    q_sfc = ocean_surface_q_sat(
        T_sfc, forcing.p_surface,
        thermo_convention=getattr(config, "thermo_convention", "legoesm"),
        bulk_scheme=config.bulk_scheme)

    # Bulk turbulent fluxes (positive upward); scheme-consistent with the
    # atmosphere surface layer (see _ocean_turbulent_fluxes).
    shflx, lhflx = _ocean_turbulent_fluxes(T_sfc, q_sfc, forcing, config)

    # Radiation
    sw_net = (1.0 - config.albedo_ocean) * forcing.sw_down
    lw_net = (config.emissivity_ocean * forcing.lw_down
              - config.emissivity_ocean * constants.sigma_sb * T_sfc ** 4)

    # Vertical mixing flux (positive downward = heat from surface to deep)
    d_mid = 0.5 * (config.h_mix + config.h_deep)
    F_mix = (config.rho_ocean * config.c_ocean * config.k_mix
             * (T_sfc - T_deep) / d_mid)

    # Mixed layer energy balance.  Sign convention: net heat flux INTO the
    # mixed layer [W/m2] — sw_net/lw_net +into-ocean, shflx/lhflx +upward
    # (subtracted), q_flux_eff +into the layer, F_mix +downward to the deep
    # layer (subtracted from the mixed layer).
    q_flux_eff = config.Q_flux if q_flux is None else q_flux
    C_mix = config.rho_ocean * config.c_ocean * config.h_mix
    dT_sfc_dt = (sw_net + lw_net - shflx - lhflx + q_flux_eff - F_mix) / C_mix
    T_sfc_trial = T_sfc + dt * dT_sfc_dt

    # Deep layer
    C_deep = config.rho_ocean * config.c_ocean * config.h_deep
    restore = jnp.where(
        config.restore_deep,
        C_deep * (T_deep - config.T_deep_ref) / config.tau_deep,
        0.0,
    )
    dT_deep_dt = (F_mix - restore) / C_deep
    T_deep_new = T_deep + dt * dT_deep_dt

    # Freezing clamp on surface — diagnose Q_freeze (see _slab_step
    # docstring + audit F6).  Shared owner: eos.slab_freeze_point_K.
    T_freeze_eff = slab_freeze_point_K(config.T_freeze, config.freezing.scheme)
    T_sfc_new = jnp.maximum(T_sfc_trial, T_freeze_eff)
    Q_freeze = C_mix * jnp.maximum(T_freeze_eff - T_sfc_trial, 0.0) / dt

    new_state = SlabOceanState(
        T_sfc=state.T_sfc.replace(data=T_sfc_new),
        T_deep=state.T_deep.replace(data=T_deep_new),
        Q_freeze=(state.Q_freeze.replace(data=Q_freeze)
                  if state.Q_freeze is not None
                  else Field(data=Q_freeze, name="Q_freeze",
                             dims=state.T_sfc.dims, units="W/m2")),
    )
    u_sfc = jnp.zeros_like(T_sfc)
    v_sfc = jnp.zeros_like(T_sfc)
    return new_state, T_sfc_new, u_sfc, v_sfc


# ============================================================================
# Factory
# ============================================================================

def make_ocean(config: SimpleOceanConfig, sst_map=None):
    """Factory: returns a step function for the configured ocean mode.

    Parameters
    ----------
    config : SimpleOceanConfig
        Ocean configuration with mode selection.
    sst_map : jax.Array or None
        Prescribed SST map of shape (6, n, n) for fixed mode.
        If None, ``config.sst_constant`` is broadcast.

    Returns
    -------
    step_ocean : callable
        ``(state, forcing, dt) -> (state, sst, u_sfc, v_sfc)``
    """
    mode = config.mode

    # All step closures accept an optional ``q_flux`` array [W/m2] (the
    # per-step spatially+seasonally varying ocean-heat-transport convergence,
    # +into the mixed layer).  ``None`` (the default, and the only value the
    # fixed-SST mode ignores) preserves byte-identical behavior via the scalar
    # ``config.Q_flux``.  The coupled driver interpolates the monthly
    # climatology and passes it in each coupling interval (eager path, no
    # recompile); a caller that never sets it is unchanged.
    if mode == "fixed":
        def step_fixed(state, forcing, dt, q_flux=None):
            if sst_map is not None:
                sst = sst_map
            else:
                sst = jnp.broadcast_to(
                    jnp.array(config.sst_constant), forcing.sw_down.shape,
                )
            u_sfc = jnp.zeros_like(sst)
            v_sfc = jnp.zeros_like(sst)
            return state, sst, u_sfc, v_sfc
        return step_fixed

    elif mode == "slab":
        def step_slab(state, forcing, dt, q_flux=None):
            return _slab_step(state, forcing, config, dt, q_flux=q_flux)
        return step_slab

    elif mode == "two_layer":
        def step_two_layer(state, forcing, dt, q_flux=None):
            return _two_layer_step(state, forcing, config, dt, q_flux=q_flux)
        return step_two_layer

    else:
        raise ValueError(f"Unknown ocean mode: {mode!r}")
