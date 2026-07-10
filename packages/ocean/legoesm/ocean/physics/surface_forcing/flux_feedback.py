"""Veros-style "flux + feedback" surface tracer forcing (global_4deg).

Replicates the Veros ``global_4deg`` ``set_forcing_kernel`` heat/salt block
(veros/setups/global_4deg/global_4deg.py:249-266) as a config-selectable
surface-forcing scheme:

    forc_temp = (qnet + qnec·(sst* − T_surf)) · maskT / cp_0 / rho_0  [K·m/s]
    forc_salt = (1/t_rest) · (sss* − S_surf) · maskT · dzt_surf       [PSU·m/s]
    ice: where (T_surf·maskT < −1.8 °C) & (forc_temp < 0) BOTH are zeroed

Veros's tracer core then divides both by the surface-cell thickness
(``thermodynamics.py:276-282``: ``dt_tracer · forc / dzt[-1]`` on the implicit
tridiagonal RHS).  legoESM physics functions return RATES directly, so this
scheme folds that division in ONCE:

    dT/dt = (q_prescribed + q_feedback·(T_target − T_surf))
            / (rho_0 · c_sw · dz_0)                                   [K/s]
    dS/dt = (S_target − S_surf) / tau_restore_s                       [PSU/s]

i.e. for salinity the Veros ``× dzt[-1]`` (setup) and ``/ dzt[-1]`` (core)
cancel EXACTLY — the net Veros salt forcing is the plain restoring rate, so no
thickness factor appears here.  With the per-cell PISTON form
(``FluxFeedbackConfig.salt_restore_piston`` + ``S_restore_piston`` [m/s];
Veros north_atlantic ``sss_rest`` = file/100, north_atlantic.py:245-249) there
is NO setup-side dzt factor to cancel, so the core's ``/ dzt[-1]`` REMAINS:

    dS/dt = S_restore_piston · (S_target − S_surf) / dz_0              [PSU/s]

(``forc_salt_surface = sss_rest·(sss_clim − S)·maskT`` [PSU·m/s],
north_atlantic.py:336-340, divided by the top-cell thickness in the implicit
RHS, core/thermodynamics.py:282).  For heat, ``dz_0`` is the ACTUAL top-layer
thickness (partial-cell aware); under a rigid lid with a full top cell it
equals ``dz_ref[0]`` = Veros's fixed ``dzt[-1]`` exactly.

Unit seam (W/m² ⇄ K·m/s): Veros carries ``forc_temp_surface`` in K·m/s
(= Q[W/m²]/(cp·rho_0)); legoESM forcing channels carry Q in W/m² (positive
INTO the ocean = warming, same sign convention as ``q_net``).  The single
conversion ``Q/(rho_0·c_sw·dz_0)`` happens here, nowhere else.

The ice mask is evaluated on the heat flux BEFORE zeroing (Veros order:
compute ``forc_temp``, then ``mask = (T·maskT < −1.8) & (forc_temp < 0)``,
then zero both heat AND salt).  The condition ``forc_temp < 0`` is
sign-equivalent to ``q_total·maskT < 0`` since cp·rho_0 > 0.  State-dependent
(traced T_surf) ⇒ ``jnp.where``, by design.

Wind stress / TKE surface forcing are NOT handled here: tau flows through the
``OceanSurfaceForcing.tau_x/tau_y`` channels into the dynamics seam
(``_bc_external_surface_forcing``) and the TKE closure reads tau directly —
exactly the ACC-recipe seam.  Veros likewise does NOT ice-mask the wind/TKE
forcing.

With ``LatLonCGridOceanConfig.surface_forcing_implicit=True`` this scheme's
dT/dS rates are automatically routed into the backward-Euler vertical-mixing
solve (``surface_tracer_forcing_fn`` seam, ``ocean_pe_latlon_cgrid.py`` stage
10b''), matching Veros's implicit placement.  Masking semantics are unchanged
by the routing: the ice mask is applied to the RATE inside this scheme, before
the seam multiplies by the land mask.

Penetrative shortwave (``FluxFeedbackConfig.penetrative_shortwave``, the Veros
global_flexible / global_1deg ``qsol`` channel — set_forcing_kernel solar
block, global_flexible.py:392-405 / global_1deg.py:333-346): the optional
``OceanSurfaceForcing.q_solar`` [W/m²] is deposited through the FULL column
via the shared two-band Jerlov kernel (``shortwave_penetration_tendency``,
type "I" ≡ the Veros literals 0.58/0.35/23.0), converted with cfg's
``c_sw``/``rho_0``.  Heat-ownership contract: ``q_prescribed`` then carries
the NON-SOLAR remainder only (legoESM I(0)=1 full-deposition convention; the
Veros pen(0)=0 zero-column-sum redistribution on the solar-inclusive qnet is
cell-by-cell algebraically identical — see ``OceanSurfaceForcing.q_solar``).
The ice mask is evaluated on the TOTAL flux (non-solar + feedback + solar,
matching Veros's ``forc_temp_surface`` whose qnet includes solar) and zeroes
the surface deposit AND the full solar column (Veros ``ice[..., None]``).
The column rate flows through the SAME implicit/explicit placement seam as
the surface heat: under ``surface_forcing_implicit=True`` it reaches the
solve input at weight 1.0 — Veros's forward-Euler-at-taup1-before-vmix
``temp_source`` placement (thermodynamics.py:419 → diffusion.py:142).
Per-column seafloor masking (Veros ``maskT``) is the caller-supplied
``wet_3d``; the W/m²→K/s division uses the REFERENCE dz (Veros divides by the
full ``dzt`` too — Veros has no partial cells, and the 4deg-recipe kbot snap
makes columns full cells).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import freezing_point
from legoesm.ocean.physics.shortwave_penetration import (
    ShortwavePenetrationConfig,
    shortwave_penetration_tendency,
)
from legoesm.ocean.physics.surface_forcing._shared import (
    linear_relaxation,
    surface_tendency_factors,
)
from legoesm.ocean.physics.surface_forcing.config import FluxFeedbackConfig
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput

# Machine-checked scheme contract (see tests/test_physics_contracts.py and
# docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md).
__physics_contract__ = {
    "summary": (
        "Veros-style surface tracer forcing: prescribed heat flux plus a linear "
        "SST feedback deposited in the top layer, surface-salinity restoring, "
        "and a simple sea-ice mask zeroing both where the surface is below "
        "freezing and the net heat flux is cooling."
    ),
    "inputs": {
        "T": "degC",
        "S": "PSU",
        "dz_0": "m (actual top-layer thickness; 0 on land)",
        "surface_forcing.q_prescribed": "W/m^2 (positive into ocean; NON-solar when q_solar given)",
        "surface_forcing.q_feedback": "W/m^2/K (>=0 damps SST anomalies)",
        "surface_forcing.T_feedback_target": "degC",
        "surface_forcing.S_restore_target": "PSU",
        "surface_forcing.S_restore_piston": "m/s (per-cell SSS restoring piston velocity, >=0 damps)",
        "surface_forcing.q_solar": "W/m^2 (positive into ocean; penetrative Jerlov column)",
        "cfg.c_sw": "J/(kg K)",
        "cfg.rho_0": "kg/m^3",
        "cfg.tau_restore_s": "s",
        "cfg.ice_threshold_C": "degC (fixed ice-mask threshold; used when cfg.freezing.scheme == 'constant')",
        "cfg.freezing.scheme": "1 (liquidus selector; != 'constant' replaces ice_threshold_C with the local freezing_point(S_surf) [degC])",
        "dz_ref": "m", "z_half_ref": "m", "jacobian": "1", "wet_3d": "1",
    },
    "outputs": {
        "dT_dt": "K/s (surface layer; plus the full Jerlov column when q_solar is given)",
        "dS_dt": "PSU/s (surface layer only)",
    },
    "sign_convention": (
        "Heat flux positive INTO the ocean = surface warming (dT_dt > 0); the "
        "feedback term q_feedback*(T_target - T_surf) warms when the surface is "
        "colder than the target; salinity restoring drives S toward the target "
        "(scalar 1/tau_restore_s form, or piston*(S_target - S_surf)/dz_0 when "
        "the per-cell piston channel is active). "
        "Ice mask zeroes BOTH tendencies (and the solar column) where "
        "(T_surf < threshold) AND (total heat flux incl. q_solar < 0); the "
        "threshold is the fixed cfg.ice_threshold_C under "
        "cfg.freezing.scheme='constant' (byte-identical default) or the "
        "local per-cell liquidus freezing_point(S_surf) [degC] otherwise. "
        "q_solar >= 0 deposits 100% of its energy in the wet column "
        "(rho_0*c_sw*sum_k dz_k*dT_k == q_solar on full-depth columns, the "
        "shared Jerlov kernel's closure)."
    ),
    # Boundary source/sink of heat and salt by construction (air-sea exchange
    # + restoring): interior-conservation does not apply.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Veros global_4deg set_forcing_kernel "
        "(veros/setups/global_4deg/global_4deg.py:249-266) + implicit source "
        "placement in veros/core/thermodynamics.py:276-282; penetrative solar: "
        "global_flexible.py:292-301,392-405 / global_1deg.py:229-240,333-346 "
        "(qsol * divpen_shortwave * ice * maskT / cp_0 / rho_0, applied at "
        "taup1 pre-vmix via thermodynamics.py:419 -> diffusion.py:142); "
        "piston SSS restoring: north_atlantic.py:245-249 (sss_rest = file/100 "
        "[m/s]) + :336-344 (forc_salt_surface = sss_rest*(sss_clim - S)*maskT, "
        "ice-masked) + core/thermodynamics.py:282 (/dzt[-1] in the implicit "
        "RHS)."
    ),
    "idealized_test": (
        "All channels None -> zero tendencies; uniform q_prescribed=Q with "
        "q_feedback=0 -> dT_dt = Q/(rho_0 c_sw dz_0) in every wet surface cell; "
        "T_surf = T_target and S_surf = S_target -> only the prescribed part "
        "remains; cold surface (T < -1.8 degC) under cooling -> both zeroed. "
        "Piston channel (tests/ocean/unit/test_flux_feedback_piston.py): "
        "hand-computed piston*(S*-S)/dz_0, default-None bit-identity, "
        "double-specification guards, ice gating, Veros NA kernel replica, "
        "AD finiteness. "
        "Solar channel (tests/ocean/unit/test_flux_feedback_solar.py): Jerlov "
        "column vs hand two-band exponential, exact column heat closure, ice "
        "quadrants gating the full column, Veros global-setup kernel replica "
        "to 1e-14, double-count guards, implicit/explicit placement equality."
    ),
}


def flux_feedback_surface_forcing(
    T: jnp.ndarray,
    S: jnp.ndarray,
    dz_0: jnp.ndarray,
    surface_forcing,
    cfg: FluxFeedbackConfig,
    *,
    dz_ref: jnp.ndarray | None = None,
    z_half_ref: jnp.ndarray | None = None,
    jacobian: jnp.ndarray | None = None,
    wet_3d: jnp.ndarray | None = None,
) -> SurfaceForcingOutput:
    """Compute the flux+feedback surface tracer forcing rates.

    Parameters
    ----------
    T, S : array, shape (..., nlev)
        Tracer fields; index 0 of the trailing axis is the surface.
    dz_0 : array, shape (...)
        ACTUAL top-layer thickness [m] at T points (partial-cell aware),
        e.g. ``compute_layer_thickness(eta, H_bathy, z_coord)[..., 0]``.
        Zero on land columns (used as the wet/dry mask, Veros maskT).
        NOTE (same convention as the ``external`` scheme): on inits that
        keep nonzero ``H_bathy`` under the land mask, dz_0 is nonzero on
        land and this mask is then wet there — the dynamics seams zero
        the land TENDENCIES downstream (stage 11 / stage 10b'' masking),
        so only the standalone ``Q_net`` diagnostic differs over land.
    surface_forcing : OceanSurfaceForcing or None
        Carries ``q_prescribed`` / ``q_feedback`` / ``T_feedback_target`` /
        ``S_restore_target`` / ``q_solar`` / ``S_restore_piston`` (any may be
        None ⇒ that term is inert; ``q_solar`` additionally requires
        ``cfg.penetrative_shortwave=True`` and the column arguments below;
        ``S_restore_piston`` [m/s] requires ``cfg.salt_restore_piston=True``
        AND ``S_restore_target`` and replaces the scalar ``tau_restore_s``
        rate with ``piston·(S* − S)/dz₀`` — Veros north_atlantic sss_rest).
    cfg : FluxFeedbackConfig
    dz_ref : array, shape (nlev,), optional
        Reference layer thicknesses [m] for the q_solar Jerlov column
        (``z_coord.dz_ref``).  Required iff the solar channel is active.
    z_half_ref : array, shape (nlev+1,), optional
        Reference interface depths [m, negative; z_half_ref[0]=0]
        (``z_coord.z_half_ref``).  Required iff the solar channel is active.
    jacobian : array, shape (...), optional
        z* column Jacobian (eta + H)/H (``compute_ocean_jacobian``; ≡ 1
        under a rigid lid).  Required iff the solar channel is active.
    wet_3d : array, shape (..., nlev), optional
        Per-cell wet mask in {0,1} (Veros maskT; e.g.
        ``compute_layer_thickness(...) > 0``) zeroing the solar deposit
        below the local seafloor.  Required iff the solar channel is active.

    Returns
    -------
    SurfaceForcingOutput
        ``dT_dt`` [K/s] non-zero in the surface layer (plus the full Jerlov
        column when ``q_solar`` is active); ``dS_dt`` [PSU/s] surface layer
        only; ``du_dt``/``dv_dt`` are zero (wind stress is a separate
        channel).  ``Q_net`` carries the ice-masked total heat flux [W/m²]
        (including ``q_solar``, 100% of which enters the column) as a
        diagnostic.
    """
    nlev = T.shape[-1]
    dtype = T.dtype
    shape_3d = T.shape

    is_ocean = dz_0 > cfg.min_wet_cell_thickness_m
    mask = is_ocean.astype(dtype)               # Veros maskT (surface level)
    dz_safe = jnp.maximum(dz_0, 1.0e-10)
    zT = jnp.zeros_like(mask)

    q_p = getattr(surface_forcing, "q_prescribed", None) if surface_forcing else None
    q_f = getattr(surface_forcing, "q_feedback", None) if surface_forcing else None
    T_t = getattr(surface_forcing, "T_feedback_target", None) if surface_forcing else None
    S_t = getattr(surface_forcing, "S_restore_target", None) if surface_forcing else None
    q_sol = getattr(surface_forcing, "q_solar", None) if surface_forcing else None
    S_p = (getattr(surface_forcing, "S_restore_piston", None)
           if surface_forcing else None)

    if q_sol is not None and not cfg.penetrative_shortwave:
        raise ValueError(
            "flux_feedback: q_solar was provided but "
            "FluxFeedbackConfig.penetrative_shortwave is False — the channel "
            "would be silently ignored (and a top-cell fallback would change "
            "the heat-ownership contract). Set penetrative_shortwave=True or "
            "fold the solar flux into q_prescribed."
        )
    if q_sol is not None and (
        dz_ref is None or z_half_ref is None or jacobian is None or wet_3d is None
    ):
        raise ValueError(
            "flux_feedback: the q_solar penetrative-shortwave column requires "
            "dz_ref, z_half_ref, jacobian and wet_3d (supplied by the "
            "make_surface_forcing_physics factory from z_coord/state)."
        )
    if q_f is not None and T_t is None:
        raise ValueError(
            "flux_feedback: q_feedback requires T_feedback_target "
            "(the feedback term is q_feedback*(T_target - T_surf))."
        )
    if T_t is not None and q_f is None:
        raise ValueError(
            "flux_feedback: T_feedback_target requires q_feedback — without "
            "it the target would be silently ignored."
        )
    if not cfg.tau_restore_s > 0.0:
        raise ValueError(
            f"flux_feedback: tau_restore_s must be > 0 s "
            f"(got {cfg.tau_restore_s!r})."
        )
    # --- Piston-velocity SSS restoring guards (EXT-N3): the gate and the
    #     channel must be specified together — a channel without the gate
    #     would be silently ignored; the gate without the channel would
    #     silently fall back to the scalar tau_restore_s form (the
    #     double-specification hazard). Trace-time (None-ness is static). ---
    if S_p is not None and not cfg.salt_restore_piston:
        raise ValueError(
            "flux_feedback: S_restore_piston was provided but "
            "FluxFeedbackConfig.salt_restore_piston is False — the channel "
            "would be silently ignored and the scalar tau_restore_s rate "
            "applied instead. Set salt_restore_piston=True (the Veros "
            "north_atlantic per-cell sss_rest form) or drop the channel."
        )
    if S_p is not None and S_t is None:
        raise ValueError(
            "flux_feedback: S_restore_piston requires S_restore_target "
            "(the restoring is piston*(S_target - S_surf)/dz_0)."
        )
    if cfg.salt_restore_piston and S_p is None and S_t is not None:
        raise ValueError(
            "flux_feedback: salt_restore_piston=True but no S_restore_piston "
            "channel was provided while S_restore_target is given — the "
            "restoring would silently fall back to the scalar tau_restore_s "
            "form. Provide S_restore_piston, or set salt_restore_piston=False."
        )

    T_surf = T[..., 0]
    S_surf = S[..., 0]

    # --- Total surface heat flux [W/m², + into ocean], masked (Veros
    #     multiplies forc_temp by maskT; cp·rho_0 > 0 preserves the sign). ---
    q_total = zT
    if q_p is not None:
        q_total = q_total + jnp.asarray(q_p, dtype)
    if q_f is not None:
        q_total = q_total + jnp.asarray(q_f, dtype) * (
            jnp.asarray(T_t, dtype) - T_surf
        )
    q_total = q_total * mask

    # --- W/m² → K/s: the ONE conversion (Veros: /cp_0/rho_0 in the setup
    #     kernel, /dzt[-1] in the tracer core).  #518: shared helper, with the
    #     Veros-faithful config-pinned cfg.rho_0/cfg.c_sw (deliberately NOT the
    #     eos module constants the other schemes use). ---
    _, inv_rho_csw_dz = surface_tendency_factors(
        is_ocean, dz_0, cfg.rho_0, cfg.c_sw)
    inv_rho_csw_dz = inv_rho_csw_dz.astype(dtype)
    dT_top = q_total * inv_rho_csw_dz

    # --- Penetrative solar column [K/s] (Veros qsol·divpen·maskT/cp_0/rho_0):
    #     100% of q_solar through the SHARED two-band Jerlov kernel (type "I"
    #     ≡ the Veros literals), cfg-pinned cp/rho_0, per-cell wet (maskT)
    #     gating.  Python-if on channel None-ness (static pytree structure):
    #     the default path is structurally untouched. ---
    if q_sol is not None:
        q_sol_m = jnp.asarray(q_sol, dtype) * mask
        dT_solar = shortwave_penetration_tendency(
            q_sol_m,
            jnp.asarray(dz_ref, dtype),
            jnp.asarray(z_half_ref, dtype),
            jnp.asarray(jacobian, dtype),
            ShortwavePenetrationConfig(
                scheme="jerlov_2band", water_type=cfg.shortwave_water_type,
            ),
            rho_0=cfg.rho_0,
            c_sw=cfg.c_sw,
        ) * jnp.asarray(wet_3d, dtype)
    else:
        q_sol_m = None
        dT_solar = None

    # --- SSS restoring [PSU/s].  Scalar form: Veros's ×dzt[-1] (setup) /
    #     ÷dzt[-1] (core) cancel exactly ⇒ the plain rate.  Piston form
    #     (EXT-N3): no setup-side dzt factor, so the core's ÷dzt[-1] remains —
    #     dz_0 here is the ACTUAL top-layer thickness, the same convention as
    #     the heat conversion above (≡ Veros's fixed dzt[-1] under a rigid
    #     lid with full top cells, e.g. the NA kbot snap). ---
    if S_t is not None:
        if S_p is not None:
            inv_dz = jnp.where(is_ocean, 1.0 / dz_safe, 0.0).astype(dtype)
            dS_top = (jnp.asarray(S_p, dtype)
                      * (jnp.asarray(S_t, dtype) - S_surf) * inv_dz)
        else:
            # Scalar SSS restoring (shared kernel, #518 item 9).
            dS_top = linear_relaxation(
                S_surf, jnp.asarray(S_t, dtype), cfg.tau_restore_s) * mask
    else:
        dS_top = zT

    # --- Simple ice mask (Veros order: evaluated on the PRE-ZEROING heat
    #     flux; zeroes BOTH heat and salt).  The mask quantity is the TOTAL
    #     flux INCLUDING q_solar: Veros's ``forc_temp_surface > 0`` uses the
    #     solar-inclusive qnet, and the open-water indicator multiplies the
    #     full 3-D solar source (``ice[..., None]``).  State-dependent ⇒
    #     jnp.where. ---
    if cfg.ice_mask:
        q_for_ice = q_total if q_sol_m is None else q_total + q_sol_m
        # Ice threshold [°C]: the fixed cfg.ice_threshold_C (Veros comparison,
        # byte-identical default) or, under a liquidus scheme (MED-1
        # follow-up), the LOCAL freezing point from the surface salinity.
        # ``freezing_point`` returns KELVIN and T here is °C, so subtract
        # ``constants.T_freeze`` (the 0 °C reference).  ``scheme`` is a static
        # config field ⇒ feature-gating Python branch, not a traced select
        # (an unknown scheme raises inside freezing_point at trace time).
        if cfg.freezing.scheme == "constant":
            ice_threshold_C = cfg.ice_threshold_C
        else:
            ice_threshold_C = (
                freezing_point(S_surf, 0.0, scheme=cfg.freezing.scheme)
                - constants.T_freeze
            ).astype(dtype)
        ice = jnp.logical_and(
            T_surf * mask < ice_threshold_C, q_for_ice < 0.0
        )
        dT_top = jnp.where(ice, 0.0, dT_top)
        dS_top = jnp.where(ice, 0.0, dS_top)
        q_total = jnp.where(ice, 0.0, q_total)
        if dT_solar is not None:
            dT_solar = jnp.where(ice[..., None], 0.0, dT_solar)
            q_sol_m = jnp.where(ice, 0.0, q_sol_m)

    pad_T = ((0, 0),) * (len(shape_3d) - 1)
    dT_dt = jnp.pad(dT_top[..., None], (*pad_T, (0, nlev - 1)))
    dS_dt = jnp.pad(dS_top[..., None], (*pad_T, (0, nlev - 1)))
    if dT_solar is not None:
        dT_dt = dT_dt + dT_solar
        q_total = q_total + q_sol_m  # Q_net diagnostic = true total heat in
    z3 = jnp.zeros(shape_3d, dtype=dtype)

    return SurfaceForcingOutput(
        du_dt=z3, dv_dt=z3, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=q_total, tau_x=zT, tau_y=zT,
    )
