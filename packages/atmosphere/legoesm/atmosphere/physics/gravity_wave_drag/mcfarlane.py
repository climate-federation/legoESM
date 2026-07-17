"""McFarlane (1987)-INSPIRED single-wave orographic gravity-wave drag.

A compact, mostly-smooth single-wave (c = 0) orographic drag: a
Froude-capped launch stress saturates upward (Lindzen breaking) and deposits
its stress-divergence as a momentum sink, with frictional heating.

Faithfulness (read before using as an oracle)
----------------------------------------------
This is **McFarlane-INSPIRED, not a faithful E3SM ``gw_oro`` port** — use
``e3sm_cam`` for the E3SM-faithful orographic GWD.  Faithful ONLY in the
**Froude-capped source FORM** ``min(h_disp², fcrit2·(U/N)²)`` (E3SM
``gw_oro_src``, verified by ``_mcfarlane_launch_stress`` + tests; the
displacement ``h_disp`` is ``h`` by default, E3SM's ``2·sgh`` with
``use_e3sm_hdsp=True``).  Documented DEPARTURES from
E3SM ``gw_oro``/``gw_common``:

* **Source amplitude** (flag-selectable, ``config.use_e3sm_hdsp``): the
  DEFAULT uses ``h_topo`` (a subgrid-orography std dev) DIRECTLY as the
  displacement, i.e. ``0.5·k·h²``; E3SM forms the displacement
  ``hdsp = 2·sgh`` (gw_oro.F90:117) and launches ``0.5·k·hdsp² = 2·k·sgh²``
  — so at equal ``sgh`` the default launches ~4× LESS **below the Froude
  cap** (above the cap both use the same ``fcrit2·(U/N)²`` limit and agree;
  1–4× in the band between).  ``use_e3sm_hdsp=True`` applies the E3SM
  doubling inside ``_mcfarlane_launch_stress`` (the ``h_topo_col`` units
  contract stays "sgh stddev", shared with ``e3sm_cam``) and REQUIRES the
  per-column ``h_topo_col`` — on the scalar ``config.h_topo`` fallback it
  raises, because this scheme has NO landfrac factor (E3SM's driver applies
  ``utgw *= landfrac``, gw_drag.F90:904-906) and a quadrupled uniform 500 m
  pseudo-mountain would silently drag over every ocean column.  Flip is
  behavioral → retune ``G_0``/``directional_spread``/``tau_max``,
  RCE/AMIP-gated.
* **Surface-only source** — the source ``ρ``, ``N``, ``U`` are taken at the
  bottom level, not E3SM's depth-averaged low-level source.
* **Saturation** is a Lindzen-style ``τ_sat ∝ ρ·U³·k/N`` smooth cap, not the
  E3SM ``gw_common`` spectral ``gw_drag_prof`` solver.
* **Critical level** (c = 0): the gated residual stress is RADIATED (removed),
  not deposited in the crossing layer as E3SM does — because the rigid
  single-wave drag is directed along the source and depositing it on the
  reversed flow would accelerate it (``u·du_dt + v·dv_dt > 0``).  This UNDER-deposits at a
  sharp reversal (in one discontinuous-reversal experiment ≈17 % of the launched
  stress; the exact fraction depends on grid, profile, and gate parameters).
* **Numerics margins** (sub-percent, kept): (a) ``tau_sat`` is FLOORED at
  1e-10 Pa (a dead wave carries the floor; the crit gate does the killing)
  where E3SM ZEROES ``tausat``/``taudmp`` ≤ ``taumin`` (gw_common.F90:495,541);
  (b) the ``|U_proj|`` 1e-2 m/s clip in ``tau_sat`` has no oracle analog
  (E3SM's ``ubmc2mn = 0.01 m²/s²`` guards the damping term, not ``tausat``);
  (c) ``N`` is the shared midpoint θ-gradient ``brunt_vaisala_n_full`` (E3SM
  ``gw_prof`` uses an interface T-gradient form; both floor ``N² ≥ 1e-8``).
* **No displacement gate**: E3SM zeroes the source unless ``hdsp > 10 m``
  AND ``U > 2 m/s`` (gw_oro.F90:165-172, hard); ours has only the smooth
  min-wind sigmoid (which MULTIPLIES ``U``, also weakening the Froude cap —
  a smooth-scheme choice with no oracle analog) and no ``h`` threshold —
  ``h²`` scaling keeps tiny-``h`` launches tiny.

**Conservation** (``conserves = ["energy"]``): a STATIONARY orographic wave
(``c = 0``) carries ZERO vertical wave-energy flux (``F_E = c·F_momentum = 0``,
Eliassen–Palm), so it transports momentum without transporting energy.  All
mean-flow KE removed where the wave breaks is therefore returned LOCALLY as heat
— ``dT_dt = −(u·du_dt + v·dv_dt)/c_pd`` from the FINAL applied tendency gives the
EXACT closure ``c_pd·∫ρ (dT/dt) dz == eps_gwd``, conserving resolved KE +
internal energy pointwise.  This is conservation BY CONSTRUCTION, which for
``c = 0`` (no wave-energy flux) is the COMPLETE energy budget, not mere
bookkeeping.  MOMENTUM is NOT conserved (absorbed by the subgrid mountain / at a
critical level); the critical-level radiation and the post-flux tendency limiter
break MOMENTUM closure, not energy (both rescale ``dT_dt`` with the same limited
``du_dt``).  Contrast the ``c ≠ 0`` launched-spectrum schemes ``hines.py`` /
``prognostic_spectral.py`` (``["none"]``: their waves carry energy the limiter
discards).

The drag is a physically-signed momentum sink bounded by the launched stress.
It is **AD-traceable (finite ``jax.grad`` in tested regimes) but NOT
mathematically differentiable** at the hard ``U_proj > 0`` mask discontinuity or
the ``jnp.minimum`` kinks (the mask's gradient is zero on the reversed side,
which is correct: no source-direction physics there).

References
----------
- McFarlane, N. A. (1987). The effect of orographically excited gravity
  wave drag on the general circulation of the lower stratosphere and
  troposphere. J. Atmos. Sci., 44, 1775-1800.
- E3SM ``gw_oro.F90`` (``gw_oro_src``) + ``gw_common.F90`` (``gw_drag_prof``):
  launch ``tauoro = 0.5*k*min(hdsp^2, fcrit2*(U/N)^2)*rho*N*U`` with
  ``hdsp = 2*sgh``; critical-level filter (gw_common.F90:492); tendency limiters
  (gw_common.F90:642-643).  The Froude-cap FORM is shared; the rest is not — see
  ``e3sm_cam`` for the faithful port.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import brunt_vaisala_n_full, safe_divide
from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "McFarlane (1987)-INSPIRED single-wave (c=0) orographic gravity-wave "
        "drag (NOT a faithful E3SM gw_oro port — see e3sm_cam): a launched "
        "subgrid-orography wave stress (Froude-capped, E3SM gw_oro_src form) "
        "saturates upward (Lindzen breaking) and deposits its stress-divergence "
        "as a momentum sink on the resolved flow, with frictional heating."
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
        "along the source (surface-wind) direction (accel<=0 along the fixed "
        "unit vector (cos_a,sin_a)); the guaranteed KE-sink invariant is the "
        "VECTOR projection u*du_dt + v*dv_dt = accel*U_proj <= 0 (NOT the "
        "componentwise du_dt*u, which can be >0 for a wind that VEERS with "
        "height), made strict by a hard U_proj>0 mask. Carried stress is "
        "monotone non-increasing upward and bounded by the launched stress. "
        "eps_gwd = -sum(rho*(u*du_dt+v*dv_dt)*dz) >= 0 (a pure sink here). The "
        "frictional heating dT_dt = -(u*du_dt + v*dv_dt)/c_pd returns that KE "
        "loss locally as heat, so c_pd*sum(rho*dT*dz) == eps_gwd by construction "
        "-- exact energy conservation for a c=0 wave (zero wave-energy flux, so "
        "nothing is radiated; conserves=energy)."
    ),
    # conserves=["energy"]: a STATIONARY orographic wave (phase speed c=0)
    # carries ZERO vertical wave-energy flux (F_E = c * F_momentum = 0, Eliassen-
    # Palm), so it transports momentum WITHOUT transporting energy. Hence ALL
    # mean-flow KE removed where the wave breaks is returned LOCALLY as heat with
    # nothing radiated: dT_dt = -(u*du+v*dv)/c_pd computed from the FINAL applied
    # tendency gives c_pd*sum(rho*dT*dz) == eps_gwd EXACTLY, so resolved KE +
    # internal energy is conserved pointwise. The identity is "definitional"
    # (dT_dt is SET from the tendency) but that makes it conservation BY
    # CONSTRUCTION, which for c=0 (zero wave-energy flux) is the COMPLETE energy
    # story -- not merely bookkeeping. (Contrast the c!=0 launched-spectrum
    # schemes hines.py / prognostic_spectral.py, whose waves carry energy that
    # the saturation/limiter discard -> ["none"]; and e3sm_cam.py, whose OWN
    # docstring calls its c=0 orographic subpath energy-conserving in-atmosphere,
    # taking ["none"] only via the multi-source union.) MOMENTUM is NOT conserved
    # (a sink to the subgrid mountain / absorbed at a critical level); the
    # critical-level radiation + post-flux tendency limiter break MOMENTUM
    # closure, not energy (both scale dT_dt with the same limited du_dt). The
    # drag stays a sink (never adds momentum).
    "conserves": ["energy"],
    "differentiable": True,
    "reference": (
        "McFarlane (1987), J. Atmos. Sci. 44, 1775-1800; E3SM gw_oro.F90 "
        "(gw_oro_src) launch stress + gw_common.F90 saturation/limiters"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py: rest / "
        "zero-orography column -> zero tendency; u*du_dt + v*dv_dt <= 0 at "
        "every level (vector KE sink); c_pd*sum(rho*dT_dt*dz) == eps_gwd >= 0 "
        "(exact KE->heat energy closure for the c=0 wave)"
    ),
}


def _mcfarlane_launch_stress(
    rho_sfc: jax.Array,
    N_sfc: jax.Array,
    U_activated: jax.Array,
    h_topo_sq: jax.Array,
    config: McFarlaneConfig,
) -> jax.Array:
    """Froude-capped orographic launch stress ``tau_0`` [Pa].

    The E3SM ``gw_oro_src`` source FORM (the one piece this scheme is faithful
    to)::

        tau_0 = G_0 * rho * N * k * min(h^2, fcrit2 * (U / N)^2) * U

    ``min(h^2, fcrit2*(U/N)^2)`` is the Froude cap: the streamline-displacement
    amplitude saturates at the value that makes the low-level flow marginally
    unstable (Fr = 1), so above the cap ``tau_0`` is INDEPENDENT of ``h`` and
    scales as ``U^3 / N``.  Returned BEFORE the optional ``directional_spread``
    scaling and the operational ``tau_max`` clip so the cap can be probed
    directly (the total deposited drag is downstream of the Lindzen saturation
    ``tau_sat`` and cannot isolate it).

    Displacement convention (``config.use_e3sm_hdsp``): E3SM forms the
    streamline displacement ``hdsp = 2*sgh`` (gw_oro.F90:117) and launches
    ``oroko2*min(hdsp^2, sghmax) = 0.5*k*min((2*sgh)^2, fcrit2*(U/N)^2)``
    (gw_oro.F90:166-168, ``oroko2 = 0.5*kwv`` at :33).  With the flag ON the
    doubling is applied here, INSIDE the cap (the h_topo_col units contract
    stays "sgh stddev", shared with e3sm_cam) — exactly 4x the legacy launch
    below the Froude cap, identical above it (both Froude-limited), 1-4x in
    the band ``h^2 < fcrit2*(U/N)^2 < 4*h^2``.  Default OFF keeps the legacy
    direct-``h`` displacement (~4x smaller at equal ``sgh``; ``h_topo``
    effectively a tuned amplitude — see the module docstring).
    """
    froude_h_sq = config.fcrit2 * safe_divide(
        U_activated ** 2, N_sfc ** 2, eps=1e-30,
    )
    h_disp_sq = 4.0 * h_topo_sq if config.use_e3sm_hdsp else h_topo_sq
    h_eff_sq = jnp.minimum(h_disp_sq, froude_h_sq)
    return config.G_0 * rho_sfc * N_sfc * config.k_wave * h_eff_sq * U_activated


def mcfarlane_gwd(
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
    config: McFarlaneConfig,
    h_topo_col: jax.Array | None = None,
) -> GWDOutput:
    """Compute McFarlane orographic GWD tendencies.

    Parameters
    ----------
    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
        Standard GWD backend signature. All column arrays (ncol, nlev).
    h_topo_col : jax.Array, shape (ncol,) or None
        Optional per-column subgrid orographic standard deviation [m]
        overriding the global ``config.h_topo`` (audit 2026-05-12
        MEDIUM #9).  When ``None`` the scalar config value is used
        everywhere (legacy behaviour).  REQUIRED (not None) when
        ``config.use_e3sm_hdsp`` is set — see the ValueError below.

    Returns
    -------
    GWDOutput
    """
    # Dispatch hardening (static config + static None-ness, jit-safe):
    # ``use_e3sm_hdsp`` quadruples the below-cap launch; on the scalar
    # ``config.h_topo = 500 m`` fallback that pseudo-mountain exists over
    # every OCEAN column too (this scheme has no landfrac factor, unlike
    # the E3SM driver's ``utgw *= landfrac``), so the flag would silently
    # quadruple ocean drag planet-wide.  Demand the real per-column sgh.
    if config.use_e3sm_hdsp and h_topo_col is None:
        raise ValueError(
            "McFarlaneConfig.use_e3sm_hdsp=True requires a per-column "
            "h_topo_col (wire grid.subgrid_topo_stddev or pass h_topo_col "
            "explicitly): the E3SM hdsp=2*sgh displacement quadruples the "
            "below-cap launch, and applying it to the uniform scalar "
            "config.h_topo fallback would quadruple drag over ocean "
            "columns (no landfrac factor in this scheme)."
        )
    ncol, nlev = u.shape

    # Brunt-Väisälä frequency at full levels
    N_full = brunt_vaisala_n_full(T, p_full, z_full)

    # Low-level wind
    u_sfc = u[:, -1]
    v_sfc = v[:, -1]
    U_ll = jnp.sqrt(u_sfc ** 2 + v_sfc ** 2 + 1e-10)
    cos_a = u_sfc / U_ll
    sin_a = v_sfc / U_ll

    # Smooth minimum wind activation
    U_activated = jax.nn.sigmoid(config.min_wind_sharpness * (U_ll - config.min_wind)) * U_ll

    # Wind projection along wave direction.  ``U_proj`` is the SIGNED
    # source-projected wind; it is positive at launch (the source direction is
    # the surface wind) and a critical level is where it falls to zero / reverses
    # (the orographic phase speed is c = 0).
    U_proj = u * cos_a[:, None] + v * sin_a[:, None]
    U_proj_abs = jnp.clip(jnp.abs(U_proj), 1e-2, None)  # coeff-ok: projected-wind floor [m/s]

    # Launch flux: orographic gravity-wave stress
    #     tau_0 = G_0 * rho * N * k * h^2 * U
    # (after McFarlane 1987 / Palmer 1986).  ``G_0`` is *dimensionless*;
    # the dimensional factors that turn the formula into a stress
    # [Pa = kg/(m·s²)] are ``rho * N * k * h^2 * U``.  The earlier
    # implementation omitted ``k_wave`` and clipped the result to
    # ``[0, 10] Pa`` — which combined with the missing wavenumber gave
    # the formula units of ``kg²/(m²·s⁴)`` and a magnitude of order
    # ``10⁴`` (numerically) → clip truncated to 10 → drag ~1e-21 m/s²
    # (audit's "McFarlane stress dimensionally suspect").
    rho_sfc = rho[:, -1]
    N_sfc = N_full[:, -1]
    if h_topo_col is None:
        h_topo_sq = config.h_topo ** 2
    else:
        h_topo_sq = jnp.clip(h_topo_col, 0.0, None) ** 2
    # McFarlane (1987) / E3SM ``gw_oro_src`` (gw_oro.F90:166-168) cap the
    # displacement height by the Froude-number limit before forming the
    # launch stress:
    #     tau_0 = 0.5*k * min(hdsp^2, fcrit2*(U/N)^2) * rho * N * U
    # i.e. the streamline-displacement amplitude saturates at the value that
    # would make the low-level flow marginally unstable (Fr = 1).  The earlier
    # form used the raw ``h^2`` with no Froude cap, so tall mountains in weak
    # winds launched an unphysically large stress that only the operational
    # ``tau_max`` clip masked.  ``fcrit2`` lives in config; ``N_sfc`` is the
    # source-level Brunt-Väisälä frequency.  ``oroko2 = 0.5*k`` is folded into
    # the ``G_0`` prefactor (G_0 defaults to 0.5).
    tau_0 = _mcfarlane_launch_stress(
        rho_sfc, N_sfc, U_activated, h_topo_sq, config,
    )
    tau_0 = tau_0 * config.directional_spread
    tau_0 = jnp.clip(tau_0, 0.0, config.tau_max)

    # Saturation stress per level (Lindzen 1981 / McFarlane 1987):
    #     tau_sat = efficiency * rho * U^3 * k_wave / (N * envelope)   [Pa]
    # The earlier formulation omitted ``k_wave`` and had units
    # ``kg/s^2`` rather than ``Pa = kg/(m·s^2)`` — together with the
    # missing ``k_wave`` in the launch stress (fixed earlier in this
    # file) the scheme produced dimensionally inconsistent stresses
    # whose numerical magnitudes were off by a factor of ~k_wave that
    # the ``tau_0`` clip then masked operationally.  Lindzen
    # (``lindzen.py:85``) implements the correct form; McFarlane is
    # now aligned with it.
    envelope = config.envelope_scale
    # fcrit2 couples the SAME marginal-instability Froude criterion into the
    # saturation stress AND the launch cap, exactly as the oracle: E3SM forms
    # effkwv = kwv*fcrit2 (gw_common.F90:153) entering tausat =
    # effkwv*rhoi*ubmc^3/(2*ni) (gw_common.F90:493-494), the sibling of the
    # sghmax = fcrit2*(U/N)^2 launch cap (gw_oro.F90:166) this scheme already
    # applies; McFarlane (1987) couples them likewise (tau_sat ~ E*k*Fc^2/2 *
    # rho*U^3/N).  Attached only to the launch, a tuned fcrit2 moved the
    # source-strength limit while leaving the breaking limit untouched — a
    # coefficient faithful in isolation but wired to a DIFFERENT structural
    # object (the F6 lesson).  Identity at the default fcrit2 = 1.
    tau_sat = (
        config.efficiency
        * config.fcrit2
        * rho
        * U_proj_abs ** 3
        * config.k_wave
        / (jnp.clip(N_full, 1e-6, None) * envelope)
    )
    tau_sat = jnp.clip(tau_sat, 1e-10, None)

    # Smooth critical-level absorption gate (E3SM gw_common.F90:492
    # ``where ubmc*(ubi_above - c) > 0``).  The orographic wave has phase
    # speed c = 0, so a critical level is where the source-projected wind
    # ``U_proj`` falls to zero / reverses sign.  The earlier ``|U_proj|^3``
    # saturation stress was symmetric in ``U_proj`` and therefore stayed large
    # through a wind reversal, letting the wave TRANSMIT past the critical
    # level and ACCELERATE the reversed flow (u·du_dt + v·dv_dt > 0).
    #
    # The gate is applied to the CARRIED-FORWARD stress inside the scan (not to
    # ``tau_sat``).  Gating ``tau_sat`` was unsafe: ``crit_gate`` can push
    # ``tau_sat`` far below the ``safe_divide`` ``eps=1e-12`` mask floor on a
    # sharp reversal, at which point the ratio masks to 0, ``excess = -1`` and
    # the wave does NOT break (codex round-1 finding #1).  Multiplying
    # ``tau_new`` by ``crit_gate`` instead forces the propagated stress to ~0
    # AT the critical level unconditionally, and the upward ``jnp.minimum``
    # monotonicity then keeps it zero above — robust to arbitrarily abrupt
    # reversals.  ``crit_gate -> 1`` where ``U_proj`` is well above the floor,
    # so the forward path is unchanged below any critical level.
    crit_gate = jax.nn.sigmoid(
        config.crit_level_sharpness * (U_proj - config.crit_level_floor)
    )

    # Top-down scan: cap the carried stress at the local saturation stress.
    #
    # The earlier ``softmin(tau_carry, tau_sat) = -logsumexp(-α·[a,b])/α``
    # introduced a ``-log(2)/α`` FLOOR BIAS when the two arguments are nearly
    # equal: above a critical level both ``tau_carry`` and ``tau_sat`` collapse
    # to ~0, the softmin then returns a small NEGATIVE value, and
    # ``drag = tau_carry - tau_k`` came out positive at every level — a
    # persistent spurious drag (≈ log(2)/α) that ACCELERATED the reversed flow
    # above the critical level (u·du_dt + v·dv_dt > 0).  We replace it with the same
    # bias-free smooth saturation cap that ``lindzen.py`` uses: a sigmoid blend
    # toward ``tau_sat`` once the carried stress exceeds it, hard-floored by
    # ``jnp.minimum(·, tau_carry)`` so the stress is monotone non-increasing
    # upward and the per-level drag is ``>= 0`` (a genuine momentum sink).
    # ``jnp.minimum`` of equal arguments is exactly zero — no floor bias.
    sat_sharpness = config.softmin_sharpness
    def scan_fn(carry, k_rev):
        tau_carry = carry
        k = nlev - 1 - k_rev
        tau_sat_k = tau_sat[:, k]
        # Smooth breaking on the DIMENSIONLESS excess ratio (as in lindzen.py):
        # ``tau_carry / tau_sat - 1``.  ``tau_sat`` is upstream pre-clipped to
        # ``>= 1e-10`` (so its VJP is already zero in the floor-active cells).
        # Near a critical level ``tau_sat ∝ |U_proj|^3`` (with ``|U_proj|``
        # floored at 1e-2) is itself tiny, so once ``tau_carry`` overtakes it the
        # blend caps ``tau_new`` down to ``tau_sat`` ≈ 0.  (The critical-level
        # gate does NOT touch ``tau_sat``; it multiplies the CARRIED stress
        # ``tau_new`` at the end of this scan step, which is what forces the
        # propagated stress to ~0 AT the critical level unconditionally — the
        # upward ``jnp.minimum`` then keeps it zero above.)  ``safe_divide`` uses
        # ``eps`` below the ``1e-10`` pre-clip floor so the forward path is
        # bit-identical to ``tau_carry / tau_sat`` for any physical input.
        excess = safe_divide(tau_carry, tau_sat_k, eps=1e-12) - 1.0
        f_break = jax.nn.sigmoid(sat_sharpness * excess)
        tau_new = tau_carry * (1.0 - f_break) + tau_sat_k * f_break
        # Stress can only decrease upward (and never go negative): this kills
        # the floor-bias leak that the logsumexp softmin produced.
        tau_new = jnp.minimum(tau_new, tau_carry)
        # Saturation breaking deposits its convergence on the mean flow.
        drag_sat = tau_carry - tau_new

        # Critical-level absorption (orographic c = 0): the wave is removed from
        # the propagated stress where the source-projected wind reverses
        # (``crit_gate -> 0``).  Two distinct things must happen there:
        #   (a) the wave stops propagating upward -> multiply ``tau_new`` by the
        #       gate so no stress is carried into the reversed layer (robust to
        #       an arbitrarily sharp reversal because it acts on the CARRIED
        #       stress, not on ``tau_sat``); and
        #   (b) the absorbed pseudomomentum is NOT deposited as a force on the
        #       reversed flow.  The single-wave orographic drag is always a
        #       deceleration along the source direction (``accel = -|...|·cos_a``);
        #       applying it at a level whose local wind has already reversed
        #       would ACCELERATE that reversed flow (u·du_dt + v·dv_dt > 0).  The absorbed
        #       momentum is treated as radiated rather than dumped onto the
        #       opposing flow — the standard single-wave critical-level
        #       treatment.
        #
        # The carried stress (a) is gated by the SMOOTH ``crit_gate`` so the
        # absorption is differentiable.  The deposited drag (b) is additionally
        # masked by a HARD ``U_proj > 0`` positivity mask so it is EXACTLY zero
        # in any reversed layer — the smooth sigmoid alone is never identically
        # zero, leaving a tiny accelerating leak at weakly-negative ``U_proj``
        # (codex round-3: ~2 m/s/day worst case).  The hard mask makes
        # ``u·du_dt + v·dv_dt <= 0`` STRICT.  Its gradient is zero on the reversed side,
        # which is correct: there is no source-direction physics there.
        gate_k = crit_gate[:, k]
        pos_mask = (U_proj[:, k] > 0.0).astype(tau_carry.dtype)
        tau_new = tau_new * gate_k
        drag = drag_sat * gate_k * pos_mask
        return tau_new, drag

    _, drag_stack = jax.lax.scan(scan_fn, tau_0, jnp.arange(nlev))
    drag_all = drag_stack.T[:, ::-1]  # (ncol, nlev), top-first

    # Convert to tendency
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz = jnp.clip(dz, 1.0, None)
    accel = -drag_all / jnp.clip(rho * dz, 1e-10, None)

    # Tendency limiters (E3SM gw_common.F90:642-643).  ``accel`` is a pure
    # deceleration along the source direction (``drag_all >= 0``), so it always
    # opposes the flow; we cap its MAGNITUDE without touching its sign:
    #   1. ``|du/dt| <= umcfac * |c - U_proj| / dt``  (c = 0) so a single step
    #      never changes the wind by more than ``umcfac`` of the wind-to-phase-
    #      speed gap — i.e. the drag cannot reverse the wind past the (zero)
    #      phase speed.
    #   2. ``|du/dt| <= tndmax``  an absolute ceiling that kills the
    #      ridiculously large ``stress/(rho*dz)`` accelerations in thin,
    #      low-density upper layers.
    # The limiters are AD-safe (``jnp.minimum``/``jnp.abs`` are subgradient
    # operations with a well-defined VJP a.e.; no NaN, no dead gradient).
    # NOTE on conservation: this is a *post-flux* tendency limiter — where it
    # binds, the column-integrated drag no longer exactly equals the diagnosed
    # stress-flux divergence ``g·Δtau``.  It still leaves the drag a momentum
    # SINK bounded by the launched surface stress (the property the harness
    # asserts); it does NOT introduce a momentum source.
    tndmax = config.tndmax_per_day / 86400.0
    accel_cap = jnp.minimum(
        config.umcfac * jnp.abs(U_proj) / dt, tndmax
    )
    accel_mag = jnp.minimum(jnp.abs(accel), accel_cap)
    accel = -accel_mag  # always a deceleration along the source direction

    du_dt = accel * cos_a[:, None]
    dv_dt = accel * sin_a[:, None]

    # Frictional heating
    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd

    # Column dissipation (positive-definite: KE lost by the mean flow)
    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
