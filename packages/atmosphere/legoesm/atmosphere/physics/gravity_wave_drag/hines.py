"""Single-bulk-amplitude WKB gravity wave drag (Hines-1997-inspired).

.. warning::

   This is **NOT** the full Hines (1997) Doppler-spread *spectrum*.  It is
   a single-bulk-amplitude WKB approximation: one rms wave amplitude is
   propagated and saturated per column (see the disclosed-simplification
   note below), rather than the full azimuthal + vertical-wavenumber
   Doppler-spread spectrum.

Non-orographic GWD scheme: a single bulk wave amplitude grows with
decreasing density and is capped by a smooth saturation amplitude. Uses
bottom-up propagation with smooth sigmoid activation for full
differentiability.

.. note::

   **Disclosed simplification / limitation (F-GWD-2, defer).** This is a
   *single bulk-amplitude* approximation of the Hines spectrum: one rms
   wave amplitude ``sigma`` is propagated and saturated per column rather
   than the full azimuthal + vertical-wavenumber spectrum.  Consequently
   the whole packet saturates at once and the per-level deposition
   ``(sigma_grown^2 - sigma_sat^2)·rho`` is large; the ``Fmax`` clip then
   *binds* over a substantial fraction of a realistic column (≈22-40 % of
   levels on 30-60 level grids at the default ``total_rms_wind``), so the
   cap — not the Doppler-spread physics — shapes the drag profile there.
   The drag remains a physically-signed momentum sink and is fully
   differentiable; only its *vertical distribution* is cap-dominated.  A
   faithful upgrade (spectral/azimuthal Hines) plus a QBO / momentum-flux
   benchmark is required before retuning ``Fmax`` / ``total_rms_wind`` —
   retuning without a benchmark would be guessing.  Default GWD scheme is
   ``none``; this scheme is opt-in.

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
        "Hines (1997) Doppler-spread non-orographic gravity-wave drag "
        "(single bulk-amplitude approximation): a launched rms wave amplitude "
        "grows with 1/sqrt(rho) upward, saturates at the Doppler-broadening "
        "cap, and deposits its momentum-flux divergence as a drag on the flow."
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
        "local wind vector (du_dt*u + dv_dt*v <= 0); momentum-flux deposition "
        "drag = rho*(sigma_grown^2 - sigma_sat^2) is floored at 0 (a sink). "
        "eps_gwd>=0 is the column KE loss returned as frictional heating "
        "dT_dt = -(u*du_dt + v*dv_dt)/c_pd."
    ),
    # KE removed from the mean flow is returned exactly as frictional heating,
    # so total ENERGY is conserved. Momentum is NOT conserved (wave source at
    # launch, absorption aloft). The Fmax momentum-flux cap and the post-flux
    # tendency limiter reshape the vertical distribution but keep the drag a
    # sink; they do not add momentum.
    "conserves": ["energy"],
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
    """Compute Hines Doppler-spread GWD tendencies.

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
    # As gravity waves propagate upward, their amplitude grows with
    # decreasing density (energy conservation: F ~ rho * sigma^2 = const).
    # Saturation occurs when the wave-induced velocity perturbation
    # reaches the critical Doppler-broadening amplitude
    # ``sigma_critical = N / m_*`` (Hines 1997 eq. 9), which is constant
    # per column at fixed N and m_*.  An earlier formulation divided this
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

    # Bottom-up scan: propagate sigma_gw upward from surface.
    # ``rho_ratio_step[:, k]`` carries amplitude from level k+1 to level k;
    # at the surface (k = nlev-1) the step factor is 1 (initial condition).
    def scan_fn(carry, k_rev):
        sigma_gw = carry
        k = nlev - 1 - k_rev

        # Amplitude growth from density decrease (single-layer step)
        sigma_grown = sigma_gw * rho_ratio_step[:, k]

        # Dissipation where grown amplitude exceeds saturation
        f_diss = jax.nn.sigmoid(
            config.doppler_sharpness * (sigma_grown - sigma_sat[:, k])
        )
        sigma_new = sigma_grown * (1.0 - f_diss) + sigma_sat[:, k] * f_diss

        # Momentum deposited: ``ΔF = ρ · (σ²_grown - σ²_new)`` [Pa] — this
        # is the wave momentum-flux divergence between two levels of the
        # WKB-grown wave, where ``F = ρ · <u'w'> ∝ ρ · σ²`` for an
        # upward-propagating gravity wave.  Acceleration of the mean
        # flow is then ``-ΔF / (ρ·dz)`` [m/s²].
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
