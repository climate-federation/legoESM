"""Prognostic spectral gravity wave drag parameterization (EXPERIMENTAL).

Multi-azimuthal, multi-wavenumber spectral GWD with a prognostic
wave spectrum. Carries the wave flux array forward in time with a
relaxation timescale back to the launch source.

Uses jax.lax.scan for the vertical propagation, fully differentiable.

.. warning::

   **Experimental / not validated; opt-in only (the default GWD scheme is
   ``none``).**  This scheme is *bespoke* — it is NOT a faithful
   implementation of a published spectral GWD parameterization (e.g.
   Alexander-Dunkerton 1999 or Scinocca 2003) and carries no
   ``__physics_contract__`` yet (tracked in ``CONTRACT_TODO``).

   **F-GWD-1 (deposition sign) — FIXED; energetics caveat remains.** The
   momentum deposition originally omitted the ``sign(c - U_proj)`` factor,
   so a symmetric launch spectrum produced a force independent of the mean
   wind.  The deposition now carries ``s0 = sign(c - U_launch)`` fixed at
   the launch level (see the sign-convention block in
   ``__physics_contract__``), which relaxes ``U_proj`` toward ``c`` — a
   true drag whenever the spectrum is slower than the wind.  ``eps_gwd``
   is the column wave-dissipation (frictional-heating) integral and is
   ``>= 0`` by construction.  NOTE: with a prognostic two-sided spectrum
   the mean flow can still legitimately gain kinetic energy from waves
   faster than the wind, so the mean-flow KE loss
   ``-sum(rho*(u*du_dt + v*dv_dt)*dz)`` is NOT guaranteed positive and is
   deliberately NOT what ``eps_gwd`` reports; the saturation energetics
   still need a proper spectral-GWD review plus a momentum-flux / QBO
   benchmark (``parameterization_checks.md`` F-GWD-1).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import brunt_vaisala_n_full
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    PrognosticSpectralConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "Prognostic multi-azimuthal / multi-wavenumber spectral gravity-wave "
        "drag: carries a wave momentum-flux spectrum forward in time (relaxed "
        "toward a launch source), saturates it upward, and deposits the "
        "stress-divergence as wind tendencies carrying the constant launch-"
        "level directional factor s0 = tanh((c - U_launch)/w) ~ "
        "sign(c - U_launch) (opt-in)."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "rho": "kg/m^3", "lat": "rad", "dt": "s",
        "spectrum_in": "Pa (per-column wave momentum-flux spectrum, (n_az, n_wn))",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "eps_gwd": "W/m^2",
        "spectrum_new": "Pa (updated wave momentum-flux spectrum)",
    },
    "sign_convention": (
        "z up; waves launched at the surface propagate upward. The vertical "
        "flux of azimuth-projected momentum carried by a wave of phase speed "
        "c is s0*F with F>=0 the carried spectrum and s0 = sign(c - U_launch) "
        "FIXED at the launch level (a wave's pseudomomentum sign does not "
        "change with height until absorbed), so the deposition du_proj/dt = "
        "+g*(dF/dp)*s0 relaxes U_proj TOWARD c (a true drag whenever the "
        "spectrum is slower than the wind; F-GWD-1 fixed). Using the CONSTANT "
        "s0 (not the local-layer sign) makes the column deposition telescope "
        "to g*s0*(F_launch - F_top), so momentum is conserved even across a "
        "critical level where the local sign flips. Frictional heating uses "
        "the intrinsic form dT_dt = (g/c_pd)*sum(drag*|c - U_proj|) >= 0 "
        "(gated by config.thermal_tendency) and eps_gwd >= 0 BY CONSTRUCTION "
        "(|.| >= 0 and the carried flux may not grow above the flux entering "
        "from below)."
    ),
    # Conserves NOTHING robustly: the spectrum is RELAXED toward a launch source
    # (a source/sink, not conserved wave momentum flux) and the wave-energy flux
    # through the column top is not tracked, so the column budget stays OPEN;
    # the thermal (energy) tie-back is config-gated (thermal_tendency).
    # Declaring any conserved quantity would over-claim.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Multi-azimuthal spectral non-orographic GWD (prognostic wave "
        "spectrum), legoESM; directional momentum-flux sign per Warner & "
        "McIntyre (1996) / Scinocca (2003); intrinsic-frequency heating as in "
        "E3SM gw_common dttke (sum_l (c_l - ubm)*gwut_l)"
    ),
    "idealized_test": (
        "rest state -> zero tendency and spectrum relaxes toward launch_flux "
        "on timescale tau_decay; du_dt/dv_dt shapes (ncol, nlev); symmetric "
        "two-wave spectrum (+-c, c < U) in mean flow U > 0 -> net force "
        "opposing U, zero force at U = 0 by symmetry, eps_gwd >= 0 "
        "(tests/atmosphere/hydrostatic/unit/test_gwd_physical_realism.py)"
    ),
}

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def prognostic_spectral_gwd(
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
    config: PrognosticSpectralConfig,
    spectrum_in: jax.Array,
) -> Tuple[GWDOutput, jax.Array]:
    """Compute prognostic spectral GWD tendencies.

    Parameters
    ----------
    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
        Standard GWD backend signature.
    spectrum_in : jax.Array
        Input wave spectrum, shape (ncol, n_azimuths, n_wavenumbers).

    Returns
    -------
    GWDOutput
        Tendencies.
    spectrum_new : jax.Array
        Updated wave spectrum.
    """
    ncol, nlev = u.shape
    n_az = config.n_azimuths
    n_wn = config.n_wavenumbers

    # Brunt-Väisälä frequency at full levels
    N_full = brunt_vaisala_n_full(T, p_full, z_full)  # (ncol, nlev)

    # Wavenumber grid (log-spaced)
    k_grid = jnp.exp(jnp.linspace(
        jnp.log(config.k_min), jnp.log(config.k_max), n_wn
    ))  # (n_wn,)

    # Azimuthal directions
    azimuths = jnp.linspace(0.0, 2.0 * jnp.pi, n_az, endpoint=False)  # (n_az,)
    cos_az = jnp.cos(azimuths)  # (n_az,)
    sin_az = jnp.sin(azimuths)

    # Wind projection per azimuth: (ncol, n_az, nlev)
    U_proj = (
        u[:, None, :] * cos_az[None, :, None]
        + v[:, None, :] * sin_az[None, :, None]
    )

    # Phase speed per wavenumber: c = N / k.  ``k_grid`` is a static
    # config-derived array — its values never participate in AD — and
    # the legacy ``clip`` floor preserves the divide's forward
    # semantics for misconfigured ``k_min ≤ 1e-10`` configs (codex
    # round 3 flagged that ``safe_divide`` would silently zero
    # ``c_phase`` and ``wavelength`` in that boundary case).
    c_phase = N_full[:, :, None] / jnp.clip(k_grid[None, None, :], 1e-10, None)

    # Layer thickness
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz = jnp.clip(dz, 1.0, None)  # (ncol, nlev)

    # Pressure thickness
    dp = jnp.abs(p_half[:, 1:] - p_half[:, :-1])  # (ncol, nlev)
    dp = jnp.clip(dp, 1.0, None)

    # Bottom-up scan per (azimuth, wavenumber)
    # For each spectral component, propagate flux upward
    # Saturation: tau_sat = threshold * rho * (c - U_proj)^3 / (N * 2*pi/k)
    # We scan over levels from surface to top

    # Precompute saturation stress: (ncol, n_az, n_wn, nlev)
    # c_phase: (ncol, nlev, n_wn) -> (ncol, n_wn, nlev) via transpose
    c_phase_t = jnp.transpose(c_phase, (0, 2, 1))  # (ncol, n_wn, nlev)

    # intrinsic: (ncol, n_az, n_wn, nlev)
    intrinsic = (
        c_phase_t[:, None, :, :]     # (ncol, 1, n_wn, nlev)
        - U_proj[:, :, None, :]       # (ncol, n_az, 1, nlev)
    )
    intrinsic_abs = jnp.clip(jnp.abs(intrinsic), 0.1, None)  # coeff-ok: intrinsic-freq floor

    wavelength = 2.0 * jnp.pi / jnp.clip(k_grid, 1e-10, None)  # (n_wn,)
    N_4d = N_full[:, None, None, :]  # (ncol, 1, 1, nlev)
    rho_4d = rho[:, None, None, :]

    # ``N`` is upstream-clipped (``N2_half ≥ 1e-8`` → ``N ≥ 1e-4``) and
    # ``wavelength`` floors at ``2π/k_max`` ≥ 1e3 m for the default
    # config, so the legacy ``clip(N, 1e-6) * wavelength`` denominator
    # is bounded well above zero in normal operation.  The clip on
    # ``N_4d`` keeps the divide AD-safe via the clip's zero VJP in any
    # misconfigured neutral layer.  Issue #249 codex round 3:
    # ``safe_divide`` here would mask trace-but-valid configurations
    # rather than fall back to the clipped-denominator divide.
    tau_sat = (
        config.breaking_threshold * rho_4d * intrinsic_abs ** 3
        / (jnp.clip(N_4d, 1e-6, None) * wavelength[None, None, :, None])
    )  # (ncol, n_az, n_wn, nlev)
    tau_sat = jnp.clip(tau_sat, _EPS, None)

    # Scan from surface (level -1) to top (level 0)
    # Flatten spectral dims for scan: (ncol * n_az * n_wn,)
    F_init = spectrum_in.reshape(ncol * n_az * n_wn)
    tau_sat_flat = tau_sat.reshape(ncol * n_az * n_wn, nlev)
    dp_flat = jnp.broadcast_to(
        dp[:, None, None, :], (ncol, n_az, n_wn, nlev)
    ).reshape(ncol * n_az * n_wn, nlev)

    def scan_fn(carry, k_rev):
        F_carry = carry
        k = nlev - 1 - k_rev
        f_break = jax.nn.sigmoid(
            config.breaking_sharpness * (F_carry - tau_sat_flat[:, k])
        )
        F_new = F_carry * (1.0 - f_break) + tau_sat_flat[:, k] * f_break
        # No wave source aloft: the carried flux may not GROW above the flux
        # entering the layer from below.  The smooth blend alone overshoots by
        # up to ~0.28/breaking_sharpness [Pa] when tau_sat > F_carry (the
        # sigmoid tail pulls F toward the larger saturation value), which
        # would make drag_deposit < 0 — a spurious momentum/energy source that
        # breaks the eps_gwd >= 0 guarantee.  minimum() keeps the blend where
        # the wave actually breaks (F_carry > tau_sat) and is exact (deposit
        # = 0) where it does not.
        F_new = jnp.minimum(F_new, F_carry)
        drag_deposit = (F_carry - F_new) / jnp.clip(dp_flat[:, k], 1.0, None)
        return F_new, drag_deposit

    _, drag_stack = jax.lax.scan(scan_fn, F_init, jnp.arange(nlev))
    # drag_stack: (nlev, ncol*n_az*n_wn)
    drag_4d = drag_stack.T.reshape(ncol, n_az, n_wn, nlev)
    drag_4d = drag_4d[:, :, :, ::-1]  # reverse to top-first

    # Sum over spectrum to get (ncol, nlev) tendencies.
    # ``drag_deposit = (F_carry - F_new) / dp`` is dimensionless
    # (both F and dp are in Pa).  To convert to a per-mass force we
    # use the hydrostatic identity ``dp = -ρ·g·dz`` to get
    # ``F/(ρ·dz) = F·g/(-dp)`` → the conversion factor is ``g``,
    # not ``1`` as the earlier comment claimed.  Audit cycle iter-26
    # finding P0: missing this factor under-counted GWD acceleration
    # by a factor of ~9.8 in the prognostic-spectral path.
    #
    # Sign convention (z up; waves launched at the surface propagate up;
    # F-GWD-1 fix).  The vertical flux of azimuth-projected momentum carried
    # by a wave of phase speed c in projected wind U_proj is
    #   tau_wave = s0 * F,   F >= 0 the carried spectrum,
    # where s0 = sign(c - U_launch) is FIXED at the launch (surface) level: a
    # wave's pseudomomentum sign is set at launch and does NOT change with
    # height until the wave is absorbed.  Deposition in a layer (flux in at the
    # bottom minus flux out at the top, F_carry - F_new = dp*drag >= 0 by the
    # no-growth clamp) puts that momentum into the flow:
    #   du_proj/dt = -(1/rho) d(tau_wave)/dz = +g * s0 * drag
    # (hydrostatic dp = -rho*g*dz).  Using the CONSTANT launch sign s0 (not the
    # local-layer tanh(c - U_proj)) is what conserves momentum: the column sum
    # telescopes to g*s0*(F_launch - F_top).  The earlier local-layer sign
    # dropped the momentum deposited within a critical-level band (|c-U_proj| <
    # w), where the local sign -> 0 exactly as the deposition peaks (codex
    # conservation fix).  s0 is smoothed as tanh((c - U_launch)/w) for
    # differentiability; w = config.direction_sign_width [m/s]; nlev-1 is the
    # surface (launch) level.
    launch_sign = jnp.tanh(
        intrinsic[:, :, :, nlev - 1:nlev] / config.direction_sign_width
    )  # (ncol, n_az, n_wn, 1) -> broadcast over levels
    _trig_stack = jnp.stack([cos_az, sin_az], axis=-1)[None, :, None, None, :]
    _duv_spec = (drag_4d * launch_sign)[..., None] * _trig_stack
    _duv = jnp.sum(_duv_spec, axis=(1, 2)) * constants.g  # (ncol, nlev, 2)
    du_dt = _duv[..., 0]
    dv_dt = _duv[..., 1]

    # Frictional heating from wave dissipation.  The vertical energy flux of a
    # gravity wave is c_intrinsic * (momentum flux), so the dissipation rate is
    #   |c - U_proj| * (-d(momentum flux)/dz) = |intrinsic| * g * drag  [W/kg],
    # non-negative BY CONSTRUCTION (drag >= 0 from the no-growth clamp,
    # |intrinsic| >= 0).  At a critical level |intrinsic| -> 0: momentum still
    # deposits but ~zero energy dissipates (it goes to the mean-flow KE), which
    # is physically correct.  Independent of the (constant) sign s0 used for
    # the momentum tendency, so heating and eps_gwd stay >= 0 even when s0
    # differs from the local (c - U_proj).
    heating = constants.g * jnp.sum(
        drag_4d * jnp.abs(intrinsic), axis=(1, 2)
    )  # (ncol, nlev) [W/kg]
    dT_dt = heating / constants.c_pd
    if not config.thermal_tendency:
        dT_dt = jnp.zeros_like(dT_dt)

    # Column dissipation [W/m^2] (>= 0 by construction, see heating above).
    # Deliberately the WAVE-DISSIPATION integral, not the mean-flow KE loss
    # -sum(rho*(u*du_dt + v*dv_dt)*dz): with a prognostic two-sided spectrum
    # waves faster than the wind legitimately accelerate the mean flow toward
    # c, so the KE-loss form is NOT guaranteed positive — see the module
    # docstring note (F-GWD-1 history).
    eps_gwd = jnp.sum(rho * heating * dz, axis=1)

    # Prognostic spectrum update: relax toward launch source.  Pin the
    # broadcast dtype to the input spectrum dtype so the relaxation
    # stays at the input precision (defaulting to ``jnp.full`` allows
    # x64 mode to silently promote the spectrum to f64 even when the
    # state is f32).
    launch_source = jnp.full(
        (ncol, n_az, n_wn), config.launch_flux, dtype=spectrum_in.dtype,
    )
    spectrum_new = spectrum_in + dt * (launch_source - spectrum_in) / config.tau_decay

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd), spectrum_new
