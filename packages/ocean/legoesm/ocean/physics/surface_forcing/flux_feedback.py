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
thickness factor appears here.  For heat, ``dz_0`` is the ACTUAL top-layer
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
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.surface_forcing.config import FluxFeedbackConfig
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput

# Machine-checked scheme contract (see tests/test_physics_contracts.py and
# docs/ai_guardrails/domain_architect_vs_syntax_engine.md).
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
        "surface_forcing.q_prescribed": "W/m^2 (positive into ocean)",
        "surface_forcing.q_feedback": "W/m^2/K (>=0 damps SST anomalies)",
        "surface_forcing.T_feedback_target": "degC",
        "surface_forcing.S_restore_target": "PSU",
        "cfg.c_sw": "J/(kg K)",
        "cfg.rho_0": "kg/m^3",
        "cfg.tau_restore_s": "s",
        "cfg.ice_threshold_C": "degC",
    },
    "outputs": {"dT_dt": "K/s (surface layer only)", "dS_dt": "PSU/s (surface layer only)"},
    "sign_convention": (
        "Heat flux positive INTO the ocean = surface warming (dT_dt > 0); the "
        "feedback term q_feedback*(T_target - T_surf) warms when the surface is "
        "colder than the target; salinity restoring drives S toward the target. "
        "Ice mask zeroes BOTH tendencies where (T_surf < ice_threshold_C) AND "
        "(net heat flux < 0)."
    ),
    # Boundary source/sink of heat and salt by construction (air-sea exchange
    # + restoring): interior-conservation does not apply.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Veros global_4deg set_forcing_kernel "
        "(veros/setups/global_4deg/global_4deg.py:249-266) + implicit source "
        "placement in veros/core/thermodynamics.py:276-282."
    ),
    "idealized_test": (
        "All channels None -> zero tendencies; uniform q_prescribed=Q with "
        "q_feedback=0 -> dT_dt = Q/(rho_0 c_sw dz_0) in every wet surface cell; "
        "T_surf = T_target and S_surf = S_target -> only the prescribed part "
        "remains; cold surface (T < -1.8 degC) under cooling -> both zeroed."
    ),
}


def flux_feedback_surface_forcing(
    T: jnp.ndarray,
    S: jnp.ndarray,
    dz_0: jnp.ndarray,
    surface_forcing,
    cfg: FluxFeedbackConfig,
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
        ``S_restore_target`` (any may be None ⇒ that term is inert).
    cfg : FluxFeedbackConfig

    Returns
    -------
    SurfaceForcingOutput
        ``dT_dt`` [K/s] / ``dS_dt`` [PSU/s] non-zero only in the surface
        layer; ``du_dt``/``dv_dt`` are zero (wind stress is a separate
        channel).  ``Q_net`` carries the ice-masked total heat flux [W/m²]
        as a diagnostic.
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
    #     kernel, /dzt[-1] in the tracer core). ---
    inv_rho_csw_dz = jnp.where(
        is_ocean, 1.0 / (cfg.rho_0 * cfg.c_sw * dz_safe), 0.0
    ).astype(dtype)
    dT_top = q_total * inv_rho_csw_dz

    # --- SSS restoring [PSU/s]: Veros's ×dzt[-1] (setup) / ÷dzt[-1] (core)
    #     cancel exactly ⇒ the plain rate. ---
    if S_t is not None:
        dS_top = (jnp.asarray(S_t, dtype) - S_surf) / cfg.tau_restore_s * mask
    else:
        dS_top = zT

    # --- Simple ice mask (Veros order: evaluated on the PRE-ZEROING heat
    #     flux; zeroes BOTH heat and salt).  State-dependent ⇒ jnp.where. ---
    if cfg.ice_mask:
        ice = jnp.logical_and(
            T_surf * mask < cfg.ice_threshold_C, q_total < 0.0
        )
        dT_top = jnp.where(ice, 0.0, dT_top)
        dS_top = jnp.where(ice, 0.0, dS_top)
        q_total = jnp.where(ice, 0.0, q_total)

    pad_T = ((0, 0),) * (len(shape_3d) - 1)
    dT_dt = jnp.pad(dT_top[..., None], (*pad_T, (0, nlev - 1)))
    dS_dt = jnp.pad(dS_top[..., None], (*pad_T, (0, nlev - 1)))
    z3 = jnp.zeros(shape_3d, dtype=dtype)

    return SurfaceForcingOutput(
        du_dt=z3, dv_dt=z3, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=q_total, tau_x=zT, tau_y=zT,
    )
