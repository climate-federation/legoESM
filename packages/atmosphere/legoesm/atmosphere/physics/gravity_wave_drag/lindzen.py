"""Smoothed Lindzen (1981)-saturation orographic gravity wave drag.

Orographic GWD with smooth sigmoid activation for wave breaking, AD-safe via
jax.lax.scan for the vertical stress profile — differentiable ALMOST
EVERYWHERE (the smooth sigmoids are C-infinity, but the HARD ``U_proj > 0``
critical-level mask is a step with zero a.e. gradient, and the ``clip`` /
``minimum`` saturation + tendency limiters are subgradient kinks).

.. note::

   **Critical-level treatment (disclosed single-wave simplification).** Like
   ``mcfarlane.py``, this single-wave (c = 0) scheme deposits most launched
   momentum below the critical level by saturation breaking; residual stress
   reaching the critical level (where ``U_proj`` reverses) is absorbed/radiated
   rather than deposited on the opposing flow, so it slightly under-deposits at
   a sharp critical level relative to E3SM's spectral solver.  The drag remains
   a physically-signed, AD-safe momentum sink with no spurious acceleration.
   See ``mcfarlane.py`` for the full discussion.

Faithfulness to Lindzen (1981)
------------------------------
Reference oracle: Lindzen (1981) for the SATURATION mechanism; the on-disk
E3SM path ``e3sm_cam.py`` (``gw_oro_src`` McFarlane + ``gw_drag_prof`` Lindzen
saturation + WKB damping) as a faithful sibling for the orographic launch +
solver. This module is a Lindzen-style SINGLE-WAVE scheme with a McFarlane-type
``h²`` launch — NOT a parameter-restricted SUBSET of ``e3sm_cam.py`` (E3SM
builds a source REGION with ``hdsp = 2·sgh`` and a ``0.5·k·min(hdsp², …)`` cap,
then WKB-damps; this uses a plain surface-level ``ρ·N·k·h²·U`` and saturation
only).
FAITHFUL to Lindzen — the saturation core:
  * ``tau_sat = 0.5·ρ·k·|U_proj|³ / N`` is Lindzen's marginal-convective-
    instability saturation stress (the ½ and the ``k·U³/N`` form are Lindzen's;
    matches E3SM ``effkwv·rhoi·ubmc³/(2·ni)``);
  * the saturation-breaking HYPOTHESIS: the sigmoid ``f_break`` turns ON where
    the carried stress exceeds ``fcrit2·tau_sat`` (= ``tau_sat_eff``, the
    fcrit2-scaled cap — oracle effkwv semantics); the broken wave sheds stress
    as the momentum-flux divergence ``accel = −(tau_carry − tau_new)/(ρ·dz)``
    (a deceleration; equivalently ``+∂τ/∂z / ρ`` for the code's ``≥ 0`` stress
    ``τ`` that DECREASES upward — a monotone-non-increasing, physically-signed
    sink). The saturation is SOFT:
    ``tau_new = tau_carry·(1−f_break) + tau_sat_eff·f_break`` then
    ``min(tau_new, tau_carry)`` with ``tau_sat_eff = fcrit2·tau_sat`` — with
    finite ``Fr_sharpness`` (``f_break < 1``) and ``tau_carry > tau_sat_eff``
    the blend stays ABOVE ``tau_sat_eff`` (approaching it only as
    ``f_break → 1``) — there is NO exact hard cap.
DEPARTURES / DESIGN:
  * **``fcrit2`` scales the saturation CAP VALUE — the oracle semantics**
    (E3SM ``effkwv = kwv·fcrit2`` feeding ``tausat``, gw_common.F90:153,
    493-494; Lindzen's ``F_sat = Fr_c²·ρ·k·U³/2N``): ``tau_sat_eff =
    fcrit2·tau_sat`` with the breaking sigmoid activating at the FIXED
    threshold ``tau_carry > tau_sat_eff``.  (Formerly named ``critical_Fr``
    and wired only as the sigmoid ACTIVATION center with an unscaled
    relaxation target — the ``min(tau_new, tau_carry)`` guard made that knob
    exactly inert wherever ``tau_carry ≤ tau_sat``, so half its tuning range
    was dead; the GWD-recon audit quantified 1.4% response vs the oracle's
    50%.)  The saturation remains SOFT (finite ``Fr_sharpness`` sigmoid, no
    exact hard cap — measured ~1.4% local-tendency perturbation vs the hard
    cap at the default sharpness, total deposited stress preserved to 1e-4);
  * **LAUNCH is McFarlane (1987), not Lindzen (1981)**: ``tau_0 =
    ρ_sfc·N_sfc·k·h_topo²·U_ll``. Lindzen (1981) is a saturation/breakdown
    theory (tidal + upward-propagating waves), NOT an orographic source; the
    ``h²`` launch is the McFarlane/Pierrehumbert orographic form;
  * **NO ``sghmax`` Froude cap on the launch** (E3SM caps the source-region
    displacement; here ``tau_0`` is only clipped ≥ 0, so it grows without bound
    as ``h²``) and **NO WKB radiative damping** (E3SM ``min(taudmp, tausat)``;
    here saturation breaking + critical-level absorption are the only sinks);
  * **critical-level treatment is DESIGN, not deposition**: at ``c = 0`` (where
    ``U_proj`` reverses) the smooth ``crit_gate`` drives the CARRIED stress → ~0
    but the DEPOSITED drag is ``drag_sat·crit_gate·pos_mask`` (NOT all of
    ``drag_sat`` — the gate + hard mask scale it down toward the critical
    level), so the gate-removed RESIDUAL stress is radiated/discarded (never
    deposited on the
    reversed flow), so the scheme under-deposits at a sharp critical level;
  * ``crit_level_floor = 0.5 m/s``: the smooth gate ``sigmoid(sharpness·(U_proj
    − 0.5))`` is half-on at SIGNED ``U_proj = +0.5 m/s`` (not ``|U_proj|``), so
    absorption ramps in as the wind DROPS toward +0.5 — marginally BEFORE the
    true ``c = 0`` reversal; the sigmoid only ASYMPTOTICALLY → 0 for reversed
    ``U_proj < 0`` (never identically 0), so it is the SEPARATE hard ``U_proj > 0``
    mask that makes the DEPOSITED drag EXACTLY zero on the reversed flow;
  * the antiparallel "deceleration rigidly along the source direction" + the
    KE→heat closure (all mean-flow KE loss returned as local frictional
    heating, ``dT_dt = −(u·du_dt+v·dv_dt)/c_pd``) are single-wave DESIGN
    choices. NOTE: ``dT_dt`` is computed FROM the final tendency, so
    ``c_pd·Σρ·dT·dz == eps_gwd`` holds BY CONSTRUCTION — and for a c=0 wave
    (zero vertical wave-energy flux, F_E = c·F_momentum = 0) that IS the exact
    resolved-energy conservation the ``conserves = ["energy"]`` note declares.
  * the ``tndmax`` / ``umcfac`` tendency limiters (E3SM ``gw_common.F90``
    provenance) are post-flux magnitude caps that break the exact
    stress-divergence balance where they bind (a bounded sink, never a source).
NUMERICS (AD-safety): the smooth sigmoid breaking (``Fr_sharpness``) and smooth
critical-level gate (``crit_level_sharpness``) replace hard on/off switches; a
HARD ``U_proj > 0`` positivity mask forces the VECTOR sink
``u·du_dt + v·dv_dt ≤ 0`` strictly (the drag is antiparallel to the SOURCE-wind
direction, so componentwise ``du_dt·u`` can be positive for an oblique/veering
column wind; only the vector projection is guaranteed) — the smooth gate alone
leaves a tiny accelerating leak at weakly-negative ``U_proj``;
``safe_divide`` for ``1/N`` (issue #249); the ``|U_proj|`` floor (0.1 m/s), the
``tau_sat`` floor (1e-10), the ``min(tau_new, tau_carry)`` monotonicity clamp,
and the ``dz`` / ``ρ·dz`` floors.
Non-behavioral pins: ``tests/atmosphere/hydrostatic/unit/test_lindzen_gwd_faithful.py``.

References
----------
- Lindzen, R. S. (1981). Turbulence and stress owing to gravity wave and
  tidal breakdown. J. Geophys. Res., 86, 9707-9714.
- McFarlane, N. A. (1987). The effect of orographically excited gravity wave
  drag on the general circulation of the lower stratosphere and troposphere.
  J. Atmos. Sci., 44, 1775-1800 (the orographic launch).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import brunt_vaisala_n_full, safe_divide
from legoesm.atmosphere.physics.gravity_wave_drag.config import LindzenConfig
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "Lindzen (1981) orographic (c=0) gravity-wave drag: a launched "
        "subgrid-orography wave stress saturates upward against the local "
        "convective-overturning stress (tau_sat = 0.5*rho*k*U^3/N) and "
        "deposits its stress-divergence as a momentum sink on the flow."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "rho": "kg/m^3", "lat": "rad", "dt": "s",
        "h_topo_col": "m (optional per-column subgrid orographic stddev)",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "eps_gwd": "W/m^2",
    },
    "sign_convention": (
        "z up; orographic phase speed c=0. Drag is a deceleration directed "
        "along the SOURCE (surface-wind) direction, so it opposes the "
        "source-projected wind: the VECTOR sink u*du_dt + v*dv_dt <= 0 (== "
        "accel*U_proj) is made strict by a hard U_proj>0 mask. (Componentwise "
        "du_dt*u can be >0 for an oblique/veering column wind; only the vector "
        "projection is guaranteed.) Carried stress is monotone non-increasing "
        "upward and bounded by the launched stress. eps_gwd>=0 is the column KE "
        "loss returned as frictional heating dT_dt = -(u*du_dt + v*dv_dt)/c_pd."
    ),
    # conserves=["energy"]: a STATIONARY orographic wave (phase speed c=0) carries
    # ZERO vertical wave-energy flux (F_E = c*F_momentum = 0, Eliassen-Palm), so
    # it transports momentum WITHOUT transporting energy. Hence ALL mean-flow KE
    # removed where the wave breaks is returned LOCALLY as heat: dT_dt =
    # -(u*du+v*dv)/c_pd from the FINAL applied tendency gives c_pd*sum(rho*dT*dz)
    # == eps_gwd EXACTLY, so resolved KE + internal energy is conserved pointwise.
    # The identity is "definitional" (dT_dt SET from the tendency) but that makes
    # it conservation BY CONSTRUCTION, which for c=0 (zero wave-energy flux) is the
    # COMPLETE energy story -- not merely bookkeeping. MOMENTUM is NOT conserved
    # (absorbed by the subgrid mountain / at a critical level); the critical-level
    # radiation + post-flux tendency limiter break MOMENTUM closure, not energy
    # (both rescale dT_dt with the same limited du_dt). The discriminator is
    # "does the launched wave carry vertical energy flux (c!=0)?": c=0 orographic
    # (this scheme, mcfarlane.py #1045, rayleigh direct-drag) => ["energy"]; c!=0
    # launched spectra (hines.py, prognostic_spectral.py) whose waves carry energy
    # the limiter discards => ["none"]; e3sm_cam.py => ["none"] via its c=0(+)c!=0
    # multi-source intersection. (Corrects the prior ["none"] ruling, which
    # over-applied the hines c!=0 precedent to this c=0 scheme -- codex #1045.)
    "conserves": ["energy"],
    "differentiable": True,
    "reference": (
        "Lindzen (1981), J. Geophys. Res. 86, 9707-9714, "
        "doi:10.1029/JC086iC10p09707"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py: rest / "
        "zero-orography column -> zero tendency; u*du_dt + v*dv_dt <= 0 at every "
        "level (vector sink); c_pd*sum(rho*dT_dt*dz) == eps_gwd >= 0 (exact "
        "KE->heat energy closure for the c=0 wave)"
    ),
}


def _lindzen_launch_stress(rho_sfc, N_sfc, k_wave, h_topo_sq, U_ll):
    """McFarlane (1987) orographic launch stress ``tau_0 = ρ·N·k·h²·U`` (clipped ≥ 0).

    The ``h²`` orographic source is McFarlane/Pierrehumbert, NOT Lindzen (1981); there is NO
    ``sghmax`` Froude cap (cf. E3SM ``gw_oro_src``), so ``tau_0`` grows unbounded in ``h``.
    """
    return jnp.clip(rho_sfc * N_sfc * k_wave * h_topo_sq * U_ll, 0.0, None)


def _lindzen_saturation_stress(rho, U_proj, k_wave, N_full):
    """Lindzen (1981) marginal-instability saturation stress ``tau_sat = 0.5·ρ·k·|U_proj|³ / N``.

    ``U_proj`` is floored at 0.1 m/s (near-zero-wind / AD guard) and the result at 1e-10 Pa. This
    is the E3SM ``effkwv·rhoi·ubmc³/(2·ni)`` form; ``safe_divide`` keeps the ``1/N`` VJP finite in
    nearly-neutral layers (issue #249).
    """
    U_proj_abs = jnp.clip(jnp.abs(U_proj), 0.1, None)  # coeff-ok: projected-wind floor [m/s]
    tau_sat = 0.5 * rho * U_proj_abs ** 3 * k_wave * safe_divide(
        jnp.ones_like(N_full), N_full, eps=1e-6,
    )
    return jnp.clip(tau_sat, 1e-10, None)


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
    N_full = brunt_vaisala_n_full(T, p_full, z_full)  # (ncol, nlev)

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
    tau_0 = _lindzen_launch_stress(rho_sfc, N_sfc, config.k_wave, h_topo_sq, U_ll)

    # Saturation stress per level: tau_sat = 0.5 * rho * k * |U_proj|^3 / N
    # (Lindzen 1981; identical to the faithful E3SM path in e3sm_cam.py,
    # which uses ``effkwv*rhoi*ubmc**3/(2*ni)``).  The factor of 1/2 was
    # previously missing here, making the saturation stress ~2x too large.
    # Wave breaks where carried stress exceeds local saturation.  See
    # ``_lindzen_saturation_stress`` for the AD-safe divide by ``N`` (#249).
    tau_sat = _lindzen_saturation_stress(rho, U_proj, config.k_wave, N_full)
    # fcrit2 scales the SATURATION CAP VALUE itself — the oracle semantics
    # (E3SM effkwv = kwv*fcrit2 feeding tausat, gw_common.F90:153,493-494;
    # Lindzen-1981 F_sat ~ Fr_c^2*rho*k*U^3/2N).  The former ``critical_Fr``
    # only shifted the sigmoid ACTIVATION center while the relaxation target
    # stayed the unscaled tau_sat, and the min(tau_new, tau_carry) guard made
    # the knob EXACTLY inert wherever tau_carry <= tau_sat — the lower half
    # of its tuning range was provably dead (blend >= tau_carry whenever
    # tau_sat >= tau_carry, so the min always returned tau_carry).  With the
    # cap-scaling form the activation threshold is pinned at 1 (excess =
    # tau_carry/tau_sat_eff - 1): bit-identical at the default fcrit2 = 1.
    tau_sat = config.fcrit2 * tau_sat

    # Smooth critical-level absorption gate (E3SM gw_common.F90:492
    # ``where ubmc*(ubi_above - c) > 0``).  The orographic wave has phase
    # speed c = 0, so a critical level is where the source-projected wind
    # ``U_proj`` reverses sign.  ``tau_sat ~ |U_proj|^3`` is symmetric in
    # ``U_proj`` and so does NOT by itself absorb the wave through a reversal
    # — it merely drops to a small value near ``U = 0`` and recovers above,
    # which is not the physical critical-level filter (codex round-1 #2).  The
    # gate is applied to the carried-forward stress inside the scan so the
    # propagated stress is driven SMALL (asymptotically ~0, since the sigmoid is
    # never identically 0 at finite reversed ``U_proj``) at the critical level
    # regardless of the saturation ratio; the upward ``jnp.minimum`` monotonicity
    # then only prevents the (small) carried stress from re-growing above — it
    # does not force it exactly to 0.  ``crit_gate -> 1`` well below any critical
    # level, so the forward path is unchanged there.
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
        ) - 1.0
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
        # leak at weakly-negative ``U_proj``) — the VECTOR sink
        # ``u*du_dt + v*dv_dt <= 0`` is STRICT (componentwise ``du_dt*u`` can be
        # >0 for an oblique wind; only the source-direction projection is signed).
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
