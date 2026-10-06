"""Shared output types for all surface schemes.

A surface scheme is a pure function from ``(state, forcing, config, params)``
to a ``SurfaceFluxOutput``.  Downstream land-model post-processing
(snow, Richards, soil thermal, carbon, TileResponse) consumes this
NamedTuple without knowing which scheme produced it.
"""

from __future__ import annotations

from typing import NamedTuple

import jax


class SurfaceFluxOutput(NamedTuple):
    """Output of a surface scheme (``SimpleSEB`` or ``TwoLeafCanopy``).

    All energy fluxes in [W / m^2].  ``shflx`` / ``lhflx`` follow the
    coupler sign convention (positive = INTO the atmosphere).  ``G_soil``
    is positive INTO the soil.  ``sw_net`` / ``lw_net`` are positive INTO
    the surface (absorbed).

    The output contains everything the shared post-flux pipeline in
    ``step_multilayer_land`` / ``step_land`` needs to run snow / Richards
    / soil thermal / carbon / TileResponse — no scheme-specific branches
    are required downstream.

    Fields that only apply to the two-leaf canopy scheme (``Tf_Sun``,
    ``gs_Sun``, ``n_iters``, ``f_veg_canopy``, etc.) are ``None`` for
    SimpleSEB.  Conversely, ``beta_soil_post`` is populated only by
    SimpleSEB (which uses it in the post-step ``q_surface`` recomputation)
    — for canopy the skin-layer bulk-beta concept does not apply.
    """

    # ---- Turbulent fluxes (atmosphere-surface interface) ----
    shflx: jax.Array             # sensible heat to atmosphere [W/m^2]
    lhflx: jax.Array             # latent heat to atmosphere [W/m^2] (demanded)
    tau_x: jax.Array             # zonal momentum flux [N/m^2]
    tau_y: jax.Array             # meridional momentum flux [N/m^2]

    # ---- Radiation ----
    sw_net: jax.Array            # absorbed shortwave [W/m^2]
    lw_net: jax.Array            # net longwave (down - up) [W/m^2]
    lw_up: jax.Array             # upward longwave (for coupler) [W/m^2]

    # ---- Ground heat flux ----
    G_soil: jax.Array            # positive INTO soil (top BC for solve_soil_thermal) [W/m^2]

    # ---- Surface state for coupler / radiation ----
    T_surface: jax.Array         # emission-weighted surface T [K]
    q_surface: jax.Array         # surface specific humidity [kg/kg]
    albedo: jax.Array            # broadband shortwave albedo [-]
    emissivity: jax.Array        # broadband longwave emissivity [-]
    z0: jax.Array                # roughness length [m]

    # ---- Photosynthesis (None if stomata disabled / carbon off) ----
    gpp: jax.Array | None = None  # gross primary production [gC/m^2/s]

    # ---- Solar-induced fluorescence (None unless a SIFConfig is attached) ----
    # Observed top-of-canopy SIF photon flux [umol/m^2/s] (sunlit+shaded sum for
    # the two-leaf canopy, single leaf for SimpleSEB), scaled by the escape
    # probability fesc.  Passive diagnostic — no feedback into the land state.
    sif: jax.Array | None = None
    # Last CONVERGED canopy Newton solution per column, (ncol, 6) in the solver's
    # own variable order, NaN where this column has never converged.  Purely a
    # warm-start cache for the next call (the fixed point does not depend on the
    # seed, and the solve's adjoint returns a zero cotangent for it); None for
    # every scheme that does not run the canopy closure.
    canopy_x: jax.Array | None = None
    # Canopy solver TERMINAL DIAGNOSTICS (None for every other scheme).  The
    # solver's own squared residual at exit, that value relative to the seed's
    # (the ratio the relative convergence gate tests), and 1.0 where the loop
    # left by the ITERATION CAP rather than the damping ceiling.  These exist
    # because ``converged`` alone cannot distinguish a stalled column from a
    # nearly-solved one; non-differentiable, no feedback into the land state.
    canopy_resid_sq: jax.Array | None = None
    canopy_resid_rel: jax.Array | None = None
    canopy_hit_cap: jax.Array | None = None

    # ---- Canopy-specific diagnostics (None for SimpleSEB) ----
    Tf_Sun: jax.Array | None = None        # sunlit leaf T [K]
    Tf_Sh: jax.Array | None = None         # shaded leaf T [K]
    # Canopy air-space temperature Tc [K] = conductance-weighted blend of the
    # above-canopy air, sunlit/shaded leaves, and ground.  This is the canopy's
    # AERODYNAMIC surface temperature (the exchange node for sensible heat,
    # H_tot = rho*cp*(Tc - Ta)/Ra), and is what the coupler reports as the tile
    # ``T_sfc`` for the atmosphere's sensible-heat coupling.  Distinct from the
    # LW-derived radiometric ``T_surface``.  None for SimpleSEB.
    T_canopy_air: jax.Array | None = None
    gs_Sun: jax.Array | None = None        # sunlit stomatal conductance [m/s]
    gs_Sh: jax.Array | None = None         # shaded stomatal conductance [m/s]
    n_iters: jax.Array | None = None       # canopy Newton iteration count
    # Per-column flag: did the canopy Newton closure reach a root?  ``False``
    # means the accompanying fluxes come from a stopped iterate (iteration cap
    # hit, or a singular Jacobian whose step had to be discarded) and are NOT a
    # solution of the surface energy balance.  ``None`` for schemes that do not
    # iterate.  Consumers must not spend a False column's fluxes as they are:
    # ``multilayer_land`` closes their energy and accepts or reverts the column.
    f_veg: jax.Array | None = None         # vegetation cover fraction [0-1]
    fSun: jax.Array | None = None          # sunlit canopy fraction [0-1]
    # Ts_solve: converged soil skin T from the canopy Picard loop (for
    # SimpleSEB this equals ``T_surface``).
    Ts_solve: jax.Array | None = None
    # Per-component fluxes (canopy only).  ``LE_canopy`` = LE_Sun + LE_Sh,
    # ``LE_soil`` = LE_Soil; etc.  Useful for offline diagnostic drivers
    # that want to inspect the canopy internal partitioning.
    LE_canopy: jax.Array | None = None
    # Wet-leaf evaporation part of ``LE_canopy`` [W/m^2] — the interception-loss
    # flux sourced from the canopy-water store (two-leaf interception path).
    # ``None`` when interception is off.
    LE_wet_canopy: jax.Array | None = None
    LE_soil: jax.Array | None = None
    # SimpleSEB over a layered pack (fractional cover f): the pack's share of
    # ``lhflx`` [W/m^2 per cell area], f times the snow surface's own latent
    # flux.  ``None`` on the binary snow path and for the canopy schemes.
    LE_snow: jax.Array | None = None
    # SimpleSEB over a layered pack: cell vapour conductance [kg m-2 s-1 per
    # kg/kg] (> 0), so the caller can export the humidity q_air + E / g that
    # implies the REALISED vapour flux E.  ``None`` elsewhere.
    vapour_conductance: jax.Array | None = None
    H_canopy: jax.Array | None = None
    H_soil: jax.Array | None = None
    Rn_canopy: jax.Array | None = None
    Rn_soil: jax.Array | None = None
    # Internal canopy energy-balance closure diagnostics.  For the canopy
    # scheme, ``Rn_int = Rn_canopy + Rn_soil`` and
    # ``residual_int = Rn_int - (LE_tot + H_tot + G)`` should be within
    # ~1e-6 W/m^2 if the Newton closure converged.  None for SimpleSEB
    # (where the budget is exact by construction).
    Rn_int: jax.Array | None = None
    residual_int: jax.Array | None = None
    # External (boundary-condition) radiation balance: what a downstream
    # observer sees from the forcing and the canopy-mean surface T used
    # for LW emission.  ``residual_ext = Rn_ext - (LE + H + G)`` and the
    # difference ``residual_ext - residual_int`` indicates RT/LW
    # accounting drift.  Only populated by canopy.
    Rn_ext: jax.Array | None = None
    residual_ext: jax.Array | None = None

    # ---- Carry for post-step q_surface recomputation ----
    # SimpleSEB needs ``stomatal_ratio`` = beta_eff / beta_soil to apply
    # stomatal limitation to the post-step q_sat.  The canopy scheme
    # sets this to 1.0 because LE is computed from leaf-level humidity
    # gradients directly and q_surface is derived from the converged
    # skin T, not from a beta * q_sat product.
    stomatal_ratio: jax.Array | None = None

    # ---- Semi-implicit surface conductance for the soil-thermal Robin BC ----
    # lambda = -dG_surface/dT_sfc (>= 0) from the SimpleSEB linearisation
    # (longwave + finite-difference sensible/latent).  Passed to
    # ``solve_soil_thermal(surface_conductance=...)`` to make the T_sfc-dependence
    # of the surface energy balance implicit, removing the explicit-coupling
    # instability (large dt + thin top layer + stiff surface -> NaN).  ``None`` for
    # the two-leaf canopy (its own Newton closure handles the coupling) and for a
    # SimpleSEB build that leaves it unset -> solve_soil_thermal falls back to the
    # explicit BC (lambda = 0), byte-identical to before.
    surface_conductance: jax.Array | None = None

    # ---- Canopy heat storage (CLM-ML only) ----
    # stflx_air: canopy air-space sensible heat storage [W/m^2], positive when
    #   the air is warming (net heat stored in the canopy air column).
    # stflx_veg: vegetation biomass sensible heat storage [W/m^2].
    # Both are required for full energy balance closure:
    #   Rnet = SH + LH + G_soil + stflx_air + stflx_veg
    # None for SimpleSEB (negligible for thin canopies without explicit storage).
    stflx_air: jax.Array | None = None
    stflx_veg: jax.Array | None = None

    # ---- Solver / containment status (appended: adding these anywhere earlier
    #      would shift every following field's POSITION, so a positional
    #      constructor would silently bind the wrong values) ----
    # Did the surface scheme's iterative closure reach a root on this column?
    # ``False`` means the fluxes above come from a stopped iterate (iteration
    # cap, or a singular Jacobian whose step had to be discarded) and are NOT a
    # solution of the surface energy balance.  ``None`` for schemes that do not
    # iterate.
    converged: jax.Array | None = None
    # Per-column containment result, filled in by the land step (not by the
    # surface scheme): ``held`` marks columns whose new state was rejected and
    # reverted, ``n_held`` counts them.  A non-zero ``n_held`` means the land
    # model could not solve somewhere — a defect report, not a diagnostic to
    # ignore.
    held: jax.Array | None = None
    n_held: jax.Array | None = None
    # Unsolved-but-finite columns ACCEPTED with energy-closed fallback fluxes
    # (``fallback`` mask, ``n_fallback`` count), and unsolved finite columns the
    # fallback guards rejected and reverted instead (``fallback_rejected`` mask,
    # ``n_fallback_rejected`` count; already included in ``held``/``n_held``).
    # Filled in by the land step.
    fallback: jax.Array | None = None
    n_fallback: jax.Array | None = None
    fallback_rejected: jax.Array | None = None
    n_fallback_rejected: jax.Array | None = None

    # Layered snowpack only (``snow_scheme == "layered"``): excess of the pack-top
    # temperature over T_freeze after the implicit solve and BEFORE the enthalpy
    # re-equilibration [K], (ncol,).  The solve carries sensible heat only, so this
    # measures how far one step overshoots the melting point (an accuracy, not a
    # conservation, diagnostic).  None for the bulk snowpack.
    snow_T_top_excess: jax.Array | None = None
    # Layered snowpack only, for the column energy budget: enthalpy carried into
    # the pack by mass (snowfall, frost, rain) minus that leaving it (sublimated
    # ice, drainage incl. any liquid above T_freeze) [J/m^2, relative to ice at
    # T_freeze]; and the ground heat flux the pack+soil column received, Robin
    # term included [W/m^2, positive into the ground].  Pack enthalpy + soil
    # sensible energy change by dt*(snow_ground_heat_applied + geothermal) +
    # snow_advected_heat.  Melt water enters the soil carrying no heat (same
    # convention as rain infiltration), so the drainage enthalpy leaves the column.
    snow_advected_heat: jax.Array | None = None
    snow_ground_heat_applied: jax.Array | None = None
    # The scheme's solved surface stress MAGNITUDE rho*u*^2 [Pa] (appended
    # last, same positional reason).  ``tau_x``/``tau_y`` lay it along a
    # wind-speed-floored direction, so their length is smaller in light wind;
    # a consumer handing the land's stress to the atmosphere reads this.
    # ``None`` for schemes that solve no friction velocity.
    tau_mag: jax.Array | None = None
