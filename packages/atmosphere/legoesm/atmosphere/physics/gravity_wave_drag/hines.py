"""Single-bulk-amplitude WKB gravity wave drag (Hines-1997-inspired).

.. warning::

   This is **NOT** the full Hines (1997) Doppler-spread *spectrum*.  It is
   a single-bulk-amplitude WKB approximation: one rms wave amplitude is
   propagated and saturated per column (see the disclosed-simplification
   note below), rather than the full azimuthal + vertical-wavenumber
   Doppler-spread spectrum.

Non-orographic GWD scheme: a single bulk wave amplitude grows with
decreasing density and is capped by a smooth saturation amplitude. Uses
bottom-up propagation with smooth sigmoid activation for AD-safety —
differentiable ALMOST EVERYWHERE (the ``clip``/``minimum`` drag-floor and
tendency limiters are subgradient KINKS, not everywhere-smooth).

.. note::

   **Disclosed simplification / limitation (F-GWD-2, defer).** This is a
   *single bulk-amplitude* approximation of the Hines spectrum: one rms
   wave amplitude ``sigma`` is propagated and saturated per column rather
   than the full azimuthal + vertical-wavenumber spectrum.  Consequently
   the whole packet saturates at once and the per-level deposition
   ``(sigma_grown^2 - sigma_new^2)·rho`` is large; the ``Fmax`` clip then
   *binds* over a substantial fraction of a realistic column (≈22-40 % of
   levels on 30-60 level grids at the default ``total_rms_wind``), so the
   cap — not the Doppler-spread physics — shapes the drag profile there.
   The drag remains a physically-signed momentum sink and is AD-safe
   (differentiable a.e.; the ``clip``/``minimum`` limiters are subgradient
   kinks, not everywhere-smooth); only its *vertical distribution* is
   cap-dominated.  A
   faithful upgrade (spectral/azimuthal Hines) plus a QBO / momentum-flux
   benchmark is required before retuning ``Fmax`` / ``total_rms_wind`` —
   retuning without a benchmark would be guessing.  Default GWD scheme is
   ``none``; this scheme is opt-in.

Faithfulness to Hines (1997)
----------------------------
This is a Hines-INSPIRED bulk heuristic, NOT a faithful Hines Doppler-spread parameterization.
The Hines (1997) MECHANISM — an incident azimuthal + vertical-wavenumber SPECTRUM whose
Doppler-shifted cutoff vertical wavenumber evolves with the background wind and the wave-induced
rms wind (eq. 9 is that cutoff-wavenumber relation, NOT a velocity-amplitude cap) — is ABSENT.
This scheme propagates ONE bulk rms amplitude and saturates it against a scalar (per-column, but
locally ``N(z)``-dependent) scale ``σ_sat = N(z)/m_*`` — one amplitude, one scale per level.
FAITHFUL to Hines — only the undamped-WKB amplitude scaling:
  * the amplitude grows with ``1/√ρ`` between levels, ``σ_grown = σ·√(ρ[k+1]/ρ[k])`` — the WKB
    scaling for a conserved ``ρσ²`` (NOTE: ``ρσ²`` is the code's ASSUMED invariant, NOT the true
    momentum flux ``ρ⟨u'w'⟩``, which carries the wave-geometry ``k/m`` correlation omitted here).
DEPARTURES from Hines — the mechanism is gone:
  * **NO Doppler-spread SPECTRUM**: a SINGLE bulk rms amplitude, not the azimuthal +
    vertical-wavenumber spectrum whose cutoff evolution IS the Hines mechanism (F-GWD-2);
  * **NO background-wind coupling in propagation**: the wind is NOT used for Doppler shifting,
    cutoff evolution, critical-level filtering, or refraction — it enters ONLY as (a) the
    antiparallel sink DIRECTION and (b) the tendency limiter (``lat`` and ``p_half`` are unused;
    a resting column can still saturate the packet internally, the projection just zeroes the
    RETURNED vector tendency and heating);
  * ``σ_sat = N/m_*`` is a **bulk saturation-amplitude HEURISTIC** (a velocity scale ≈ the
    intrinsic phase speed ``N/m_*``), NOT Hines eq. 9's cutoff wavenumber;
  * the deposition ``drag = ρ(σ²_grown − σ²_new)`` (where ``σ_new`` is the smoothly-saturated
    amplitude ``σ_grown(1−f)+σ_sat·f``, equal to ``σ_sat`` only in the hard-saturation limit
    ``f→1``) is a per-layer STRESS DECREMENT [Pa] (``accel = −decrement/(ρ·dz)``). Under the
    code's ASSUMED fixed wave geometry/proportionality ``F ∝ ρσ²`` this reads as a stress-flux
    divergence, but it is NOT Hines' spectral vector momentum flux — the azimuthal-spectrum vector
    covariance ``ρ⟨u'w'⟩`` (with its ``k/m`` sign/geometry) that Hines deposits is absent here;
  * the **antiparallel "pure deceleration along U" and the KE→heat closure** (all mean-flow KE
    loss returned as local frictional heating) are bulk-closure DESIGN choices — Hines deposits
    the vector momentum/energy of the dissipating spectrum, which need not be antiparallel to the
    local ``U`` and distinguishes wave-energy dissipation from mean-flow KE dissipation and
    heat-flux (Becker & McLandress 2009);
  * the ``Fmax`` cap and the tendency limiter are **UNCLOSED sinks**: the scan removes the FULL
    uncapped amplitude loss from the wave carry but deposits only the CAPPED stress on the mean
    flow, so excess wave momentum/energy is lost rather than transferred — so even in the
    interior the mean-flow drag no longer equals the wave stress-flux divergence (momentum is
    not conserved; energy of the mean flow that IS removed is still heated back exactly).
NUMERICS (AD-safety): the smooth sigmoid saturation (``doppler_sharpness``) — NOTE this smooth
blend is a two-sided relaxation toward ``σ_sat``, so BELOW threshold (``σ_grown < σ_sat``, where
``f_diss`` is small but nonzero) it nudges the carry amplitude slightly UP toward ``σ_sat`` (a weak
spurious SOURCE); the drag clamp ``[0, Fmax]`` floors the resulting negative decrement to 0, so this
sub-threshold gain is neither drag nor heat but an unbudgeted carry perturbation of the AD smoothing
(a hard ``σ_grown > σ_sat`` gate would not have it); the ``accel`` magnitude limiter
(E3SM-PROVENANCE constants ``tndmax``/``umcfac``, a LOOSE analog of E3SM's per-phase-speed
``|c−u|/dt`` limiter — NOT
E3SM-equivalent: no phase speed ``c``, one scalar magnitude, no post-limit stress reconstruction);
density / ``U_mag`` floors; the scan-carry dtype pin.
Non-behavioral pins: ``tests/atmosphere/hydrostatic/unit/test_hines_gwd_faithful.py``.

References
----------
- Hines, C. O. (1997). Doppler-spread parameterization of gravity-wave
  momentum deposition in the middle atmosphere. 1. Basic formulation.
  J. Atmos. Solar-Terr. Phys., 59, 371-386.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import brunt_vaisala_n_full
from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "Hines-INSPIRED non-orographic gravity-wave drag (single bulk-amplitude "
        "heuristic, NOT the Doppler-spread spectrum): a launched rms wave amplitude "
        "grows with 1/sqrt(rho) upward, saturates against a bulk scalar scale "
        "N/m_*, and deposits the per-layer stress decrement as a drag on the "
        "flow.  With ``launch_p`` set the wave is held at its launch amplitude "
        "at and below that level (no source, no deposition) and starts "
        "propagating there, so the stress-decrement statement applies only "
        "ABOVE the launch level."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "rho": "kg/m^3", "lat": "rad", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "eps_gwd": "W/m^2",
    },
    "sign_convention": (
        "z up. Drag opposes the wind: accel is a pure deceleration along the "
        "local wind vector (du_dt*u + dv_dt*v <= 0); stress decrement "
        "drag = rho*(sigma_grown^2 - sigma_new^2) is floored at 0 (a sink). "
        "eps_gwd>=0 is the column KE loss returned as frictional heating "
        "dT_dt = -(u*du_dt + v*dv_dt)/c_pd."
    ),
    # KE->heat is an EXACT local closure: the resolved mean-flow KE removed
    # (-U.a) is returned exactly as frictional heating (c_pd*sum(rho*dT_dt*dz)
    # == eps_gwd). But total WAVE + mean-flow ENERGY is NOT conserved: the
    # launched wave is an EXTERNAL/unbudgeted source, and the Fmax cap + the
    # tendency limiter discard the implied carry loss (the scan removes the
    # full uncapped amplitude loss from the wave but deposits only the capped
    # stress on the mean flow). Momentum is not conserved either. Hence
    # "none" -- the only exact invariant is the resolved-KE->heat closure,
    # pinned by the idealized test, not a total-energy conservation law.
    "conserves": ["none"],
    # True = AD-compatible (a defined VJP everywhere, incl. the clip/minimum
    # subgradient kinks); NOT everywhere-smooth (see NUMERICS in the docstring).
    "differentiable": True,
    "reference": (
        "Hines (1997), J. Atmos. Solar-Terr. Phys. 59, 371-386 "
        "(Doppler-spread parameterization, basic formulation)"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py: rest "
        "state -> zero tendency; du_dt*u + dv_dt*v <= 0 at every level; "
        "c_pd*sum(rho*dT_dt*dz) == eps_gwd >= 0 (KE->heat closure)"
    ),
}


def hines_gwd(
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
    config: HinesConfig,
) -> GWDOutput:
    """Compute Hines-inspired bulk GWD tendencies.

    Parameters
    ----------
    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
        Standard GWD backend signature. All column arrays (ncol, nlev).

    Returns
    -------
    GWDOutput
    """
    ncol, nlev = u.shape

    # Brunt-Väisälä frequency at full levels
    N_full = brunt_vaisala_n_full(T, p_full, z_full)

    # Wind magnitude at each level
    U_mag = jnp.sqrt(u ** 2 + v ** 2 + 1e-10)

    # Layer thickness
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz = jnp.clip(dz, 1.0, None)

    # Saturation amplitude per level:
    # As the bulk wave propagates upward its amplitude grows with decreasing
    # density (the code's assumed WKB invariant rho*sigma^2 = const; this is
    # NOT the true momentum flux rho*<u'w'>).  Saturation occurs when the
    # wave-induced velocity perturbation reaches a BULK saturation-amplitude
    # scale ``sigma_sat = N / m_*`` (a velocity ~ the intrinsic phase speed;
    # a heuristic, NOT Hines 1997 eq. 9's cutoff wavenumber), constant per
    # column at fixed N and m_*.  An earlier formulation divided this
    # by ``rho_ratio = sqrt(rho_sfc/rho) ≥ 1`` ⇒ ``sigma_sat`` *decreased*
    # with altitude, the opposite of physical expectation: amplitudes
    # grow with altitude (1/sqrt(rho)) so the cap should remain at least
    # constant.  The /rho_ratio factor caused premature saturation aloft
    # and biased the drag deposition lower in the column (audit GWD-B2).
    sigma_sat = N_full / jnp.clip(config.m_star, 1e-6, None)

    # Per-level WKB growth factor for the bottom-up scan.  Going from
    # level (k+1) to level k (one step upward), the amplitude grows by
    # ``sqrt(rho[k+1] / rho[k])`` (energy conservation rho * sigma^2).
    # The carry already contains the integrated WKB amplitude from the
    # surface to level k+1, so we multiply by the *inter-level* ratio,
    # not the cumulative ``sqrt(rho_sfc/rho_k)``.  Multiplying by the
    # cumulative factor at every step compounds the growth and
    # over-amplifies the wave by a product of cumulative ratios — a
    # bug masked in operational use only because the sigma_sat cap
    # truncates the runaway.
    rho_ratio_step = jnp.ones_like(rho)
    rho_ratio_step = rho_ratio_step.at[:, :-1].set(
        jnp.sqrt(jnp.clip(
            rho[:, 1:] / jnp.clip(rho[:, :-1], 0.01, None), 1.0, None,  # coeff-ok: density-ratio floor
        ))
    )

    # --- Launch level ---------------------------------------------------
    # STATIC Python branch on a build-time config constant (the JAX
    # feature-gating exception): ``launch_p is None`` keeps the legacy
    # surface-launch path with no extra HLO and byte-identical output.
    #
    # A non-orographic wave launched at the SURFACE is born supersaturated
    # wherever the launch amplitude exceeds ``sigma_sat = N/m_star``, and N
    # is SMALLEST in the well-mixed boundary layer: on the 2.5 deg AMIP
    # state 2.0 m/s exceeds the 1.25 m/s sigma_sat at 140 m over 78.5% of
    # the area, so the wave breaks AT its own launch level (55% of its
    # momentum below 1 km, only 35% above 12 km).
    #
    # Level selection is PER COLUMN, not from a column-mean profile: the
    # documented units are [Pa], so a nominal 700 hPa source must sit at
    # 700 hPa over a mountain as well as over the ocean.  (The sibling
    # e3sm_cam.py:1544 picks ONE global level from the column mean, which
    # makes its threshold a reference-grid convention rather than a
    # pressure; codex review 2026-07-31.)  ``jnp.argmin`` keeps the index
    # TRACED (never ``int(...)``) so the kernel stays jit-safe; the index
    # itself is not differentiated, matching E3SM's static selection.
    #
    # Columns whose SURFACE pressure is already below ``launch_p`` (high
    # terrain) get NO source rather than a silently relocated one: the
    # launch level is pushed past the bottom so every level reads as "below
    # launch" and the column contributes zero drag.
    #
    # NB it is NOT enough to zero the drag below the launch level: the
    # amplitude carry would still propagate up through the BL and SATURATE
    # there, so the wave would arrive at the launch level already clipped
    # and the drag ALOFT would be unchanged (verified: bit-identical above
    # the launch level with output-masking alone).  The carry must be HELD
    # AT the launch amplitude until the launch level is reached, so the
    # wave genuinely starts there.
    _k_launch = None
    if config.launch_p is not None:
        # (ncol,) index of the level closest to launch_p in EACH column.
        _k_launch = jnp.argmin(
            jnp.abs(p_full - config.launch_p), axis=1
        ).astype(jnp.int32)
        # No source where the whole column lies above the launch pressure
        # (p_s < launch_p, i.e. high terrain).  The scan gate is
        # ``below = k >= k_launch``, so the sentinel that marks EVERY level
        # as below-launch is 0 (k >= 0 always holds) — NOT nlev, which
        # would make the condition never true and mask nothing.
        _p_sfc = p_full[:, -1]
        _k_launch = jnp.where(
            _p_sfc < config.launch_p, jnp.int32(0), _k_launch)

    # Bottom-up scan: propagate sigma_gw upward from surface.
    # ``rho_ratio_step[:, k]`` carries amplitude from level k+1 to level k;
    # at the surface (k = nlev-1) the step factor is 1 (initial condition).
    def scan_fn(carry, k_rev):
        sigma_gw = carry
        k = nlev - 1 - k_rev

        # Amplitude growth from density decrease (single-layer step)
        sigma_grown = sigma_gw * rho_ratio_step[:, k]

        # Smooth blend toward the saturation amplitude. ABOVE threshold
        # (sigma_grown > sigma_sat) this dissipates (sigma_new < sigma_grown,
        # a sink). BELOW threshold the sigmoid is small but NONZERO, so
        # sigma_new = sigma_grown*(1-f)+sigma_sat*f is pulled slightly UP
        # toward sigma_sat (sigma_new > sigma_grown): a weak spurious SOURCE
        # in the carry. The drag floor clip(., 0, Fmax) zeroes the resulting
        # negative decrement, so this sub-threshold amplitude gain is neither
        # drag nor heat -- an unbudgeted carry perturbation of the smooth
        # (AD) sigmoid, not a physical process (see NUMERICS in the docstring).
        f_diss = jax.nn.sigmoid(
            config.doppler_sharpness * (sigma_grown - sigma_sat[:, k])
        )
        sigma_new = sigma_grown * (1.0 - f_diss) + sigma_sat[:, k] * f_diss

        # Stress decrement: ``ΔF = ρ · (σ²_grown - σ²_new)`` [Pa].  Under the
        # code's ASSUMED fixed wave geometry/proportionality ``F ∝ ρσ²`` this
        # reads as a stress-flux divergence between two levels of the WKB-grown
        # wave; it is NOT the physical spectral flux ``F = ρ·<u'w'>`` — Hines'
        # azimuthal spectrum supplies the vector covariance (the ``k/m`` sign/
        # geometry) that this single-amplitude surrogate lacks.  Acceleration
        # of the mean flow is then ``-ΔF / (ρ·dz)`` [m/s²].
        #
        # An earlier formulation used ``ρ · (σ_grown - σ_new)``, which
        # has units ``kg/(m²·s)`` rather than Pa, so the downstream
        # ``accel = drag / (ρ·dz)`` came out as ``1/s`` rather than
        # ``m/s²`` (audit cycle 2 P1: "Hines drag dimensional
        # inconsistency").  Operationally the two forms gave near-
        # identical drag because Fmax saturates the upper levels in
        # both, but they differ by a factor of ``σ_grown + σ_new``
        # (typically 2-4×) in the sub-saturation troposphere.
        #
        # Clamp the lower bound to zero: the smooth ``f_diss`` sigmoid
        # does not vanish exactly when ``sigma_grown < sigma_sat``, so
        # without the floor a small "anti-drag" leak can appear in the
        # transition region.  GWD on the mean flow is always a sink.
        drag = (sigma_grown ** 2 - sigma_new ** 2) * rho[:, k]
        drag = jnp.clip(drag, 0.0, config.Fmax)

        # Launch gate: at and BELOW the launch level (arrays are top-down, so
        # k >= k_launch) the wave does not exist yet — hold the carry at the
        # launch amplitude and deposit no drag, so the wave genuinely STARTS
        # at the launch level with an unclipped amplitude.  Static Python
        # branch: absent for the legacy path (byte-identical HLO).
        if _k_launch is not None:
            _below = k >= _k_launch
            sigma_new = jnp.where(_below, config.total_rms_wind, sigma_new)
            drag = jnp.where(_below, 0.0, drag)

        # Pin the carry back to the launch-wind precision: under x64 the
        # Python-float ``config.*`` constants promote ``sigma_new`` to f64, but
        # the scan carry init (sigma_gw_init) is ``u.dtype`` (f32) — lax.scan
        # requires carry in/out dtypes to match, so cast the carry output.
        return sigma_new.astype(sigma_gw.dtype), drag

    # Pin the carry dtype so the scan body stays at the input precision
    # (defaulting allows x64 to silently promote the launch wind to f64).
    sigma_gw_init = jnp.full((ncol,), config.total_rms_wind, dtype=u.dtype)
    _, drag_stack = jax.lax.scan(scan_fn, sigma_gw_init, jnp.arange(nlev))
    drag_all = drag_stack.T[:, ::-1]  # (ncol, nlev), top-first

    # Convert to acceleration
    accel = -drag_all / jnp.clip(rho * dz, 1e-10, None)

    # Tendency limiters (E3SM gw_common.F90:642-643).  ``drag_all >= 0`` so
    # ``accel`` is a pure deceleration along the wind; cap its MAGNITUDE
    # without touching its sign.  The fixed ``Fmax`` momentum-flux cap divided
    # by a tiny ``rho*dz`` in thin, low-density upper layers produces
    # physically-implausible accelerations (hundreds of m/s/day); ``tndmax``
    # is the absolute ceiling, and ``umcfac*U_mag/dt`` is a LOCAL no-reversal
    # limiter (Hines is amplitude-based with no explicit phase speed ``c``, so
    # this is the bulk analog of E3SM's ``umcfac*|c-u|/dt``, not the literal
    # phase-speed limiter).  AD-safe (``jnp.minimum``/``jnp.abs`` subgradient
    # ops; no NaN/dead grad).  NOTE: a *post-flux* limiter — where it binds the
    # column drag no longer exactly equals the stress-flux divergence, but
    # stays a momentum SINK (no source).
    tndmax = config.tndmax_per_day / 86400.0
    accel_cap = jnp.minimum(config.umcfac * U_mag / dt, tndmax)
    accel = -jnp.minimum(jnp.abs(accel), accel_cap)

    cos_a = u / jnp.clip(U_mag, config.U_mag_floor, None)
    sin_a = v / jnp.clip(U_mag, config.U_mag_floor, None)
    du_dt = accel * cos_a
    dv_dt = accel * sin_a

    # Frictional heating
    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd

    # Column dissipation (positive-definite: KE lost by the mean flow)
    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
