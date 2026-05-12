"""Microphysics output containers.

HydrometeorState holds the prognostic hydrometeor fields passed to backends.
MicrophysicsOutput is the common interface returned by all backends.

All backends accept and return the same containers so that integration
code can be backend-agnostic.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class HydrometeorState(NamedTuple):
    """Hydrometeor state for backends. All fields shape (ncol, nlev).

    **Number-concentration conventions** are NOT uniform across species
    — different schemes inherit different SB / Morrison / Thompson
    historical conventions:

    * ``N_c`` (cloud droplets) — **per-volume** ``[1/m³]``.  The
      Seifert-Beheng autoconversion ``x_c = q_c · ρ / N_c`` depends
      on this so the result is in ``[kg]`` (mean droplet mass)
      comparable to ``x_star = 2.6e-10 kg``.  Default
      ``Nc_0 = 1e8 /m³`` is the maritime SB value.
    * ``N_r`` (rain drops) — **per-volume** ``[1/m³]``.  The
      self-collection ``-k_sc · N_r · q_r · ρ`` and breakup-diameter
      ``D = (q_r · ρ / N_r / (π/6 · ρ_w))^(1/3)`` both rely on the
      per-volume form (D in ``[m]``).
    * ``N_i`` (ice crystals) — **per-mass** ``[1/kg]``.  Cooper (1986)
      nucleation ``N_target = N_i0 · exp(…) / ρ`` divides the
      per-volume Cooper expression by ρ to obtain a per-mass
      concentration (``N_i0 = 5e3 /m³`` from the Cooper fit, but the
      stored ``N_i`` is per-mass).

    Mixing the two conventions in the same NamedTuple is a known
    historical artifact (audit Codex cycle 2) — each formula was
    written for the convention native to its scheme of origin.
    Converting either at the boundary would change the numerics; the
    docstring drift was the actionable fix.
    """
    q_c: jax.Array    # cloud water [kg/kg]
    q_r: jax.Array    # rain water [kg/kg]
    q_i: jax.Array    # cloud ice [kg/kg]
    q_s: jax.Array    # snow [kg/kg]
    q_g: jax.Array    # graupel [kg/kg]
    N_c: jax.Array    # cloud droplet number [1/m³] (Seifert-Beheng per-volume)
    N_r: jax.Array    # rain drop number     [1/m³] (Seifert-Beheng per-volume)
    N_i: jax.Array    # ice crystal number   [1/kg] (Morrison/Thompson per-mass)


class MicrophysicsOutput(NamedTuple):
    """Backend-agnostic output. All (ncol, nlev) except precipitation (ncol,).

    Number tendencies match the per-species convention of
    ``HydrometeorState`` — see that class's docstring for the
    cloud-vs-rain (per-volume) vs ice (per-mass) split.
    """
    dT_dt: jax.Array          # latent heating [K/s]
    dq_v_dt: jax.Array        # vapor tendency [kg/kg/s]
    dq_c_dt: jax.Array        # cloud water tendency
    dq_r_dt: jax.Array        # rain tendency
    dq_i_dt: jax.Array        # ice tendency
    dq_s_dt: jax.Array        # snow tendency
    dq_g_dt: jax.Array        # graupel tendency
    dN_c_dt: jax.Array        # cloud number tendency [1/(m³·s)] per-volume
    dN_r_dt: jax.Array        # rain number tendency  [1/(m³·s)] per-volume
    dN_i_dt: jax.Array        # ice number tendency   [1/(kg·s)] per-mass
    precipitation: jax.Array  # surface precip [kg/m^2/s]


def make_zero_hydrometeors(
    ncol: int, nlev: int, dtype=None,
) -> HydrometeorState:
    """Create a zero-initialized HydrometeorState.

    ``dtype`` defaults to the JAX default float (``float64`` under x64,
    ``float32`` otherwise).  Callers integrating with the column physics
    pipeline should pass the upstream state dtype explicitly so this
    fallback never silently promotes a float32 column path to float64.
    """
    z = jnp.zeros((ncol, nlev), dtype=dtype)
    return HydrometeorState(
        q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=z, N_i=z,
    )


def make_zero_output(
    ncol: int, nlev: int, dtype=None,
) -> MicrophysicsOutput:
    """Create a zero-initialized MicrophysicsOutput.

    ``dtype`` is forwarded to ``jnp.zeros`` for the same reason as
    ``make_zero_hydrometeors``: defaulting allows x64 mode to silently
    promote the precip path.
    """
    z2 = jnp.zeros((ncol, nlev), dtype=dtype)
    z1 = jnp.zeros((ncol,), dtype=dtype)
    return MicrophysicsOutput(
        dT_dt=z2, dq_v_dt=z2, dq_c_dt=z2, dq_r_dt=z2,
        dq_i_dt=z2, dq_s_dt=z2, dq_g_dt=z2,
        dN_c_dt=z2, dN_r_dt=z2, dN_i_dt=z2,
        precipitation=z1,
    )


def sedimentation_tendency(
    q: jax.Array,
    rho: jax.Array,
    V_t: jax.Array,
    dz: jax.Array,
    dt: float | jax.Array | None = None,
    return_surface_flux: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute sedimentation tendency from vertical flux divergence.

    When ``dt`` is supplied the outgoing flux at each level is capped
    by the layer's in-column mass per step
    (``q · rho · dz / dt``), which guarantees positivity of
    ``q_new = q + dt · tendency`` for any Courant number ``V_t·dt/dz``
    (Bott / explicit-FCT positivity).  Without the limiter explicit
    sedimentation can drive ``q`` negative when ``V_t·dt/dz > 1``.

    Parameters
    ----------
    q : jax.Array
        Hydrometeor mixing ratio [kg/kg], shape (ncol, nlev).
    rho : jax.Array
        Air density [kg/m^3], shape (ncol, nlev).
    V_t : jax.Array
        Terminal velocity [m/s], shape (ncol, nlev).
    dz : jax.Array
        Layer thickness [m], shape (ncol, nlev).
    dt : float, optional
        Physics step [s].  When provided, applies CFL-aware positivity
        limiter (recommended).
    return_surface_flux : bool, default False
        When True returns ``(tendency, surface_flux)`` where
        ``surface_flux`` is the dt-limited outgoing mass flux at the
        bottom interface [kg/m²/s].  Precipitation diagnostics MUST use
        this — using the raw ``V_t · q · rho`` at the surface breaks
        column water conservation whenever the limiter fires.

    Returns
    -------
    jax.Array or tuple
        Sedimentation tendency [kg/kg/s], shape (ncol, nlev); or
        ``(tendency, surface_flux)`` if ``return_surface_flux=True``.
    """
    q_pos = jnp.clip(q, 0.0, None)
    flux = V_t * q_pos * rho  # (ncol, nlev) outgoing flux density [kg/m^2/s]

    if dt is not None:
        # Positivity-preserving flux limiter: outgoing flux at level k
        # cannot exceed the mass available in that layer per step.
        # ``q*rho*dz/dt`` is the maximum sustainable flux density that
        # leaves ``q_new ≥ 0`` for ANY local Courant number.
        max_outflux = q_pos * rho * dz / jnp.maximum(dt, 1.0e-12)
        flux = jnp.minimum(flux, max_outflux)

    # Flux from above: zero at top, flux[k-1] enters level k.  Use
    # ``jnp.pad`` (single Pad HLO) instead of allocating a fresh
    # zero buffer + concatenate.
    flux_in = jnp.pad(flux[:, :-1], ((0, 0), (1, 0)))
    dz_safe = jnp.clip(dz, 1.0, None)
    tendency = (flux_in - flux) / (rho * dz_safe)

    if return_surface_flux:
        # Bottom outgoing flux is the precipitation reaching the surface.
        return tendency, flux[:, -1]
    return tendency
