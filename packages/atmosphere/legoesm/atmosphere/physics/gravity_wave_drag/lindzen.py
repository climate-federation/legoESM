"""Smoothed Lindzen (1981) orographic gravity wave drag.

Orographic GWD with smooth sigmoid activation for wave breaking,
fully differentiable via jax.lax.scan for the vertical stress profile.

.. note::

   **Critical-level treatment (disclosed single-wave simplification).** Like
   ``mcfarlane.py``, this single-wave (c = 0) scheme deposits most launched
   momentum below the critical level by saturation breaking; residual stress
   reaching the critical level (where ``U_proj`` reverses) is absorbed/radiated
   rather than deposited on the opposing flow, so it slightly under-deposits at
   a sharp critical level relative to E3SM's spectral solver.  The drag remains
   a physically-signed, differentiable momentum sink with no spurious
   acceleration.  See ``mcfarlane.py`` for the full discussion.

References
----------
- Lindzen, R. S. (1981). Turbulence and stress owing to gravity wave and
  tidal breakdown. J. Geophys. Res., 86, 9707-9714.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import safe_divide
from legoesm.atmosphere.physics.gravity_wave_drag.config import LindzenConfig
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput


def lindzen_gwd(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    rho: jax.Array,
    lat: jax.Array,
    dt: float,
    config: LindzenConfig,
    h_topo_col: jax.Array | None = None,
) -> GWDOutput:
    """Compute Lindzen orographic GWD tendencies.

    Parameters
    ----------
    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
        Standard GWD backend signature. All column arrays (ncol, nlev).
    h_topo_col : jax.Array, shape (ncol,) or None
        Optional per-column subgrid orographic standard deviation [m].
        When provided, overrides ``config.h_topo`` and produces a
        spatially varying launch stress ``tau_0 ∝ h_topo^2`` driven by
        the actual orography rather than a single global value
        (audit 2026-05-12 MEDIUM #9).  Pass ``None`` to keep the
        scalar fallback.

    Returns
    -------
    GWDOutput
    """
    ncol, nlev = u.shape

    # Brunt-Väisälä frequency at full levels
    # theta_v = T * (p_ref/p)^kappa
    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    dz_full = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    dz_full = jnp.clip(dz_full, 1.0, None)
    dtheta_dz = (theta[:, :-1] - theta[:, 1:]) / dz_full
    theta_bar = 0.5 * (theta[:, :-1] + theta[:, 1:])
    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
    N2_half = jnp.clip(N2_half, 1e-8, None)
    N_half = jnp.sqrt(N2_half)  # (ncol, nlev-1)

    # Extrapolate N to full levels by padding
    N_full = jnp.concatenate([
        N_half[:, :1],
        0.5 * (N_half[:, :-1] + N_half[:, 1:]),
        N_half[:, -1:],
    ], axis=1)  # (ncol, nlev)

    # Low-level wind at surface level
    u_sfc = u[:, -1]
    v_sfc = v[:, -1]
    U_ll = jnp.sqrt(u_sfc ** 2 + v_sfc ** 2 + 1e-10)
    cos_a = u_sfc / U_ll
    sin_a = v_sfc / U_ll

    # Wind projection along wave direction at each level
    U_proj = u * cos_a[:, None] + v * sin_a[:, None]  # (ncol, nlev)

    # Source stress at surface: tau_0 = rho * N * k * h^2 * U
    # ``h_topo_col`` (if provided) supplies a per-column subgrid
    # orographic stddev so that mountainous columns generate stress
    # and oceanic columns generate ≈0 stress — replacing the single
    # global ``config.h_topo`` placeholder.
    rho_sfc = rho[:, -1]
    N_sfc = N_full[:, -1]
    if h_topo_col is None:
        h_topo_sq = config.h_topo ** 2
    else:
        h_topo_sq = jnp.clip(h_topo_col, 0.0, None) ** 2
    tau_0 = rho_sfc * N_sfc * config.k_wave * h_topo_sq * U_ll
    tau_0 = jnp.clip(tau_0, 0.0, None)

    # Saturation stress per level: tau_sat = rho * U^3 * k / N
    # Wave breaks where carried stress exceeds local saturation.
    # AD-safe divide by ``N`` (issue #249): ``N_full`` can hit the
    # ``1e-8`` clip floor in nearly neutral layers, where the prior
    # ``clip + divide`` form left ``-rho*U^3*k / N**2`` cotangents that
    # blow up under reverse-mode AD.
    U_proj_abs = jnp.clip(jnp.abs(U_proj), 0.1, None)  # coeff-ok: projected-wind floor [m/s]
    tau_sat = rho * U_proj_abs ** 3 * config.k_wave * safe_divide(
        jnp.ones_like(N_full), N_full, eps=1e-6,
    )
    tau_sat = jnp.clip(tau_sat, 1e-10, None)

    # Smooth critical-level absorption gate (E3SM gw_common.F90:492
    # ``where ubmc*(ubi_above - c) > 0``).  The orographic wave has phase
    # speed c = 0, so a critical level is where the source-projected wind
    # ``U_proj`` reverses sign.  ``tau_sat ~ |U_proj|^3`` is symmetric in
    # ``U_proj`` and so does NOT by itself absorb the wave through a reversal
    # — it merely drops to a small value near ``U = 0`` and recovers above,
    # which is not the physical critical-level filter (codex round-1 #2).  The
    # gate is applied to the carried-forward stress inside the scan so the
    # propagated stress is driven to ~0 AT the critical level regardless of the
    # saturation ratio, and the upward ``jnp.minimum`` monotonicity keeps it
    # zero above.  ``crit_gate -> 1`` well below any critical level, so the
    # forward path is unchanged there.
    crit_gate = jax.nn.sigmoid(
        config.crit_level_sharpness * (U_proj - config.crit_level_floor)
    )

    # Top-down scan: propagate stress from surface upward
    # Levels: 0=top, -1=surface. Scan from surface to top (reversed).
    # Breaking occurs smoothly where tau_carry exceeds tau_sat
    def scan_fn(carry, k_rev):
        tau_carry = carry
        k = nlev - 1 - k_rev
        # Smooth breaking: sigmoid activation where stress exceeds saturation.
        # AD-safe ratio (issue #249).  ``tau_sat`` is already pre-clipped
        # to ``≥ 1e-10`` upstream so its VJP is already zero in the
        # floor-active cells.  ``safe_divide`` is used here with ``eps``
        # strictly *below* the pre-clip floor (``1e-12`` vs ``1e-10``) so
        # the mask is never triggered for any physical input — the
        # forward path stays bit-identical to the legacy
        # ``tau_carry / tau_sat`` (including in floor-clipped cells,
        # where the breaking transition must keep firing once
        # ``tau_carry`` overtakes the floor).  ``safe_divide`` is kept
        # here defensively in case the upstream pre-clip is later
        # removed or relaxed.
        excess = safe_divide(
            tau_carry, tau_sat[:, k], eps=1e-12,
        ) - config.critical_Fr
        f_break = jax.nn.sigmoid(config.Fr_sharpness * excess)
        tau_new = tau_carry * (1.0 - f_break) + tau_sat[:, k] * f_break
        tau_new = jnp.minimum(tau_new, tau_carry)
        drag_sat = tau_carry - tau_new
        # Critical-level absorption (orographic c = 0).  Where the source-
        # projected wind reverses: (a) stop propagating the stress upward via
        # the SMOOTH ``crit_gate`` (differentiable absorption), and (b) do NOT
        # deposit the absorbed pseudomomentum as a force on the reversed flow.
        # The single-wave drag is rigidly along the source direction, so a HARD
        # ``U_proj > 0`` positivity mask makes the deposited drag EXACTLY zero in
        # any reversed layer (the smooth sigmoid alone leaves a tiny accelerating
        # leak at weakly-negative ``U_proj``) — du/dt·u <= 0 is STRICT.
        gate_k = crit_gate[:, k]
        pos_mask = (U_proj[:, k] > 0.0).astype(tau_carry.dtype)
        tau_new = tau_new * gate_k
        drag = drag_sat * gate_k * pos_mask
        return tau_new, drag

    _, drag_stack = jax.lax.scan(scan_fn, tau_0, jnp.arange(nlev))
    # drag_stack: (nlev, ncol) — reverse to get (ncol, nlev) top-first
    drag_all = drag_stack.T  # (ncol, nlev)
    drag_all = drag_all[:, ::-1]  # back to top-first ordering

    # Convert stress deposit to tendency: drag / (rho * dz)
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz = jnp.clip(dz, 1.0, None)
    accel = -drag_all / (jnp.clip(rho * dz, 1e-10, None))

    # Tendency limiters (E3SM gw_common.F90:642-643).  ``accel`` is a pure
    # deceleration along the source direction (``drag_all >= 0``), so it always
    # opposes the flow; cap its MAGNITUDE without touching its sign:
    #   1. ``|du/dt| <= umcfac * |c - U_proj| / dt``  (orographic c = 0), so a
    #      single step never reverses the wind past the (zero) phase speed; and
    #   2. ``|du/dt| <= tndmax`` an absolute ceiling that kills the
    #      ``stress/(rho*dz)`` blow-up where the launched stress saturates
    #      abruptly in a thin / weak-wind surface layer.
    # AD-safe (``jnp.minimum``/``jnp.abs`` subgradient ops; no NaN/dead grad).
    # NOTE on conservation: a *post-flux* limiter — where it binds the column
    # drag no longer exactly equals the stress-flux divergence, but stays a
    # momentum SINK bounded by the launched surface stress (no source).
    tndmax = config.tndmax_per_day / 86400.0
    accel_cap = jnp.minimum(config.umcfac * jnp.abs(U_proj) / dt, tndmax)
    accel = -jnp.minimum(jnp.abs(accel), accel_cap)

    # Project back to (du_dt, dv_dt)
    du_dt = accel * cos_a[:, None]
    dv_dt = accel * sin_a[:, None]

    # Frictional heating
    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd

    # Column dissipation (positive-definite: KE lost by the mean flow)
    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
