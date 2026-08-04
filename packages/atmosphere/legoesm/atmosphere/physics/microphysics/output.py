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
    # Optional prognostic SNOW number [1/kg] (per-mass, like N_i / SAM NS3D).
    # ``None`` ⇒ single-moment snow (bulk fall speed, no snow PSD); an array
    # ⇒ double-moment snow (PSD slope LAMS=(π·ρ_sn·N_s/q_s)^⅓, PSD fall speed).
    # Carried in tracer slot [9] by the plane CRM when allocated with ≥10 slots.
    N_s: jax.Array | None = None
    # Optional prognostic GRAUPEL number [1/kg] (per-mass, like N_s / SAM NG3D).
    # ``None`` ⇒ single-moment graupel (fixed intercept N0G, LAMG=(π·ρ_g·N0G/
    # (ρ·q_g))^¼); an array ⇒ double-moment graupel (LAMG=(π·ρ_g·N_g/q_g)^⅓).
    # Carried in tracer slot [10] by the plane CRM when allocated with ≥11 slots.
    N_g: jax.Array | None = None


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
    # Optional SNOW number tendency [1/(kg·s)] per-mass; None for single-moment
    # snow (matches HydrometeorState.N_s). Written to tracer slot [9].
    dN_s_dt: jax.Array | None = None
    # Optional GRAUPEL number tendency [1/(kg·s)] per-mass; None for single-
    # moment graupel (matches HydrometeorState.N_g). Written to tracer slot [10].
    dN_g_dt: jax.Array | None = None
    # Optional: the POSITIVE saturation-adjustment condensation rate
    # [kg/kg/s, >= 0] — the part of the scheme's vapour sink that becomes cloud
    # water (distinct from the net ``dq_c_dt``, which also carries autoconversion
    # / accretion sinks).  The coupled pipeline's JOINT vapour donor clamp needs
    # this isolated condensation to scale it consistently against the
    # convective vapour sink (both draw the same pre-physics q_v).  ``None`` for
    # schemes that do not expose it (the joint clamp then skips the micro term).
    dq_v_to_qc_dt: jax.Array | None = None
    # Optional SOLID-PHASE part of ``precipitation`` [kg/m^2/s, same
    # positive-into-the-surface sense], i.e. the ice + snow + graupel
    # sedimentation flux at the surface.  This is CMIP6 ``prsn``
    # ("precipitation of all forms of water in the SOLID phase"), and it is
    # a SUBSET of ``precipitation``: 0 <= precipitation_solid <=
    # precipitation by construction, since both are sums of the SAME
    # per-species dt-limited surface fluxes.
    #
    # ``None`` for schemes that do not resolve frozen precipitation
    # separately (warm-rain and bulk schemes).  It is left None rather than
    # set to zero there: a zero would be published as "this model never
    # snows anywhere", which is a claim, not an absence.
    precipitation_solid: jax.Array | None = None


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
    extra_sink: jax.Array | None = None,
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
    extra_sink : jax.Array, optional
        Additional per-level sink rate [kg/kg/s] (e.g. rain evaporation
        in the same step).  When supplied with ``dt``, the outgoing-flux
        cap becomes ``(q - extra_sink·dt) · ρ · dz / dt`` so the
        combined per-step removal by sedimentation plus the external
        sink cannot exceed available ``q``.  Codex iter-29 #1.

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
        # cannot exceed the mass available in that layer per step,
        # net of any other per-step sink (``extra_sink·dt``).  Without
        # this joint accounting, separately-capped sedimentation and
        # rain evaporation can each remove q/dt, summing to 2·q/dt
        # over one step and driving q < 0.
        if extra_sink is not None:
            q_for_cap = jnp.maximum(q_pos - extra_sink * dt, 0.0)
        else:
            q_for_cap = q_pos
        max_outflux = q_for_cap * rho * dz / jnp.maximum(dt, 1.0e-12)
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
