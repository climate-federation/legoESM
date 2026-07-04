"""Sundqvist large-scale diagnostic condensation scheme.

A diagnostic scheme that activates condensation when relative humidity
exceeds a critical threshold. Produces large-scale (non-convective)
precipitation through autoconversion and sub-cloud evaporation.

All operations use smooth (differentiable) approximations.

References
----------
- Sundqvist et al. (1989): Condensation and cloud parameterization
  studies with a mesoscale numerical weather prediction model.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
)


class SundqvistProcessRates(NamedTuple):
    """Intermediate Sundqvist process rates used to assemble tendencies."""

    condensation: jax.Array
    autoconversion: jax.Array
    evaporation: jax.Array
    precipitation: jax.Array


def diagnose_sundqvist_process_rates(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: SundqvistConfig = SundqvistConfig(),
) -> SundqvistProcessRates:
    """Diagnose the Sundqvist condensation, rain conversion, and evaporation terms."""
    del p_half  # Included for signature parity with ``sundqvist_microphysics``.
    q_c = hydrometeors.q_c
    sharpness = config.sigmoid_sharpness

    # Saturation
    q_sat = saturation_mixing_ratio(T, p_full)
    RH = q_v / jnp.clip(q_sat, 1e-10)

    # 1. Smooth condensation activation — convert increment [kg/kg] to tendency [kg/kg/s]
    #
    # Sundqvist (1989) gates condensation on RH > RH_crit (partial
    # cloud-fraction regime), but the *thermodynamic target* the
    # condensation drives ``q_v`` toward is ``q_sat``, not
    # ``RH_crit * q_sat``.  The previous code used
    # ``max(q_v - RH_crit * q_sat, 0.0)`` which removed any vapor
    # above ``0.8 * q_sat`` in a single step: at RH=1.0 the column
    # lost ``0.2 * q_sat`` of vapor per call (verified with a
    # T=290 K, p=80 kPa probe: dq_v_dt = -1e-5 kg/kg/s, dropping RH
    # from 1.00 → 0.80 in one 300-s step).  Kessler at the same
    # conditions removed zero (no supersat).  The fix removes only
    # the *supersaturation* (``q_v - q_sat``), with ``f`` keeping
    # the smooth RH_crit *onset* gating intact — Sundqvist's
    # partial-cloud-fraction subgrid variance is diagnosed
    # separately by :func:`legoesm.atmosphere.physics.clouds.cloud_fraction.sundqvist_cloud_fraction`
    # and is not the microphysics tendency's concern.
    f = jax.nn.sigmoid(sharpness * (RH - config.rh_crit))
    condensation = (
        f * jnp.maximum(q_v - q_sat, 0.0) / dt
    )  # [kg/kg/s]

    # 2. Autoconversion
    # condensation is a tendency [kg/kg/s]; multiply by dt to get increment [kg/kg]
    # Cap the autoconversion against the available cloud water + new
    # condensation so an explicit Euler step (``q_c_new = q_c + dt·dq_c_dt``)
    # never drives q_c below zero.  Without this, default
    # ``auto_rate · dt = 1e-3·1800 = 1.8`` over ~30 min for typical
    # ``q_c ≈ 1e-4 kg/kg`` overshoots the available mass by ~80 %.
    qc_avail = jnp.maximum(q_c + condensation * dt, 0.0)
    # Sundqvist (1989) autoconversion: P_auto = c_0·q_c·(1−exp(−(q_c/q_c,crit)²)).
    # The threshold factor suppresses autoconversion below the critical
    # cloud water (drizzle forms only when cloud droplets are large
    # enough) — the previous code dropped it, autoconverting linearly at
    # any q_c despite the docstring's "exceeds a critical threshold".
    # The factor ∈ [0,1) is smooth + AD-safe (exp of a non-positive arg;
    # → 0 as q_c → 0, → 1 for q_c ≫ q_c,crit).
    threshold = 1.0 - jnp.exp(
        -(qc_avail / jnp.maximum(config.qc_crit, 1e-12)) ** 2
    )
    P_auto_demand = config.auto_rate * qc_avail * threshold
    # Donor cap: rate · dt ≤ qc_avail → rate ≤ qc_avail / dt.
    dt_safe = jnp.maximum(dt, 1.0e-12)
    P_auto = jnp.minimum(P_auto_demand, qc_avail / dt_safe)

    # 3. Sub-cloud evaporation
    #
    # legoESM column layout convention: level index 0 = TOA, level
    # index ``nlev-1`` = surface (``sigma_full`` runs 0→1 top→bottom;
    # ``p_full[..., 0]`` is the lowest pressure).  ``moveaxis(..., 1,
    # 0)`` puts the vertical axis first so :func:`jax.lax.scan`
    # iterates TOA → surface — the correct direction for falling
    # rain: ``P_above`` starts at zero (no rain above TOA),
    # accumulates the autoconversion source ``P_local`` layer-by-
    # layer on the way down, and lands at the surface as the final
    # carry ``P_final``.  Sub-cloud evaporation reduces ``P_total``
    # in sub-saturated layers (``evap_mask`` peaks where ``RH < RH_crit``).
    evap_mask = jax.nn.sigmoid(sharpness * (config.rh_crit - RH))
    P_flux_layer = P_auto * rho * dz

    def scan_fn(carry, x):
        P_above = carry
        P_local, evap_m, rho_k, dz_k = x
        P_total = P_above + P_local
        evap = config.evap_coeff * evap_m * P_total / jnp.clip(rho_k * dz_k, 1.0)
        evap = jnp.minimum(evap, P_total / jnp.clip(rho_k * dz_k, 1.0))
        P_out = jnp.clip(P_total - evap * rho_k * dz_k, 0.0)
        return P_out, evap

    # Pick a working dtype that ``scan`` can carry without promotion.
    # Under ``JAX_ENABLE_X64=1`` ``jnp.zeros``/``jnp.ones`` default to
    # f64, so a state assembled from a mix of (f32) ``T`` and (f64)
    # tracers ends up with f64 ``q_v``/``q_c``.  ``P_flux_layer``
    # then inherits the f64 promotion from ``q_c + condensation * dt``,
    # while a carry pinned to ``T.dtype`` (f32) would mismatch the
    # f64 scan output.  Promoting to the wider of carry/input dtype
    # keeps ``scan`` happy without silently downcasting precipitation
    # mass.
    _scan_dtype = jnp.promote_types(T.dtype, P_flux_layer.dtype)
    inputs = (
        jnp.moveaxis(P_flux_layer.astype(_scan_dtype), 1, 0),
        jnp.moveaxis(evap_mask.astype(_scan_dtype), 1, 0),
        jnp.moveaxis(rho.astype(_scan_dtype), 1, 0),
        jnp.moveaxis(dz.astype(_scan_dtype), 1, 0),
    )
    P_init = jnp.zeros(T.shape[0], dtype=_scan_dtype)
    P_final, evap_col = jax.lax.scan(scan_fn, P_init, inputs)
    evaporation = jnp.moveaxis(evap_col, 0, 1)
    return SundqvistProcessRates(
        condensation=condensation,
        autoconversion=P_auto,
        evaporation=evaporation,
        precipitation=P_final,
    )


__physics_contract__ = {
    "summary": (
        "Sundqvist et al. (1989) large-scale diagnostic condensation: "
        "fractional-cloud RH-based condensation/evaporation of cloud water and "
        "a Sundqvist-Berge autoconversion + accretion precipitation rate."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "hydrometeors.q_c": "kg/kg",
        "hydrometeors.q_r": "kg/kg", "p_full": "Pa", "rho": "kg/m^3",
        "dz": "m", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "precipitation": "kg/m^2/s",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Latent heating dT_dt is tied to the "
        "diagnosed condensation rate via L_v. Condensate above the threshold is "
        "converted to precipitation which leaves the column (precipitation "
        ">= 0), so column moisture is NOT conserved -- no contract-level "
        "conservation is claimed. Cloud water and q_v stay >= 0."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Sundqvist, Berge & Kristjansson (1989), Mon. Wea. Rev. 117, 1641-1657",
    "idealized_test": (
        "tests/unit/test_physics_microphysics.py — RH above the critical "
        "threshold forms cloud with L_v heating; supersaturation is removed; "
        "condensate above the conversion threshold rains out; q_c, q_v >= 0."
    ),
}


def sundqvist_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: SundqvistConfig = SundqvistConfig(),
) -> MicrophysicsOutput:
    """Compute Sundqvist diagnostic condensation tendencies.

    Parameters
    ----------
    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
        Same interface as all microphysics backends.

    Returns
    -------
    MicrophysicsOutput
    """
    ncol, nlev = T.shape
    rates = diagnose_sundqvist_process_rates(
        T=T,
        q_v=q_v,
        hydrometeors=hydrometeors,
        p_full=p_full,
        p_half=p_half,
        rho=rho,
        dz=dz,
        dt=dt,
        config=config,
    )

    # 4. Latent heating
    net_cond = rates.condensation - rates.evaporation
    dT_dt = constants.L_v * net_cond / constants.c_pd

    # Tendencies.  Sundqvist is a *diagnostic* large-scale precipitation
    # scheme: rain produced by autoconversion is treated as falling
    # instantly through the column (the bottom-up scan in
    # ``diagnose_sundqvist_process_rates`` accumulates the layer
    # autoconversion source into a downward mass flux ``P_total`` and
    # subtracts sub-cloud evaporation, so ``rates.precipitation`` is the
    # surface flux).  Adding ``autoconversion - evaporation`` to ``dq_r_dt``
    # would also accumulate that mass as a ``q_r`` tracer, double-counting
    # it: the column would lose water to surface precipitation AND grow
    # ``q_r`` per step.
    #
    # Diagnostic-rain semantics (full): the scheme should *own* the q_r
    # tracer, not just leave it untouched.  Any q_r passed in (from a
    # prior step under a prognostic scheme like Kessler, or from a
    # warm-start) is treated as already-falling rain and drained to the
    # surface in one step.  The drained mass is added to the surface
    # precipitation flux so the column water budget closes (with the
    # ``rho*dz`` mass weighting used consistently by the tendencies and the
    # autoconversion precip — NOT ``dp/g``):
    #     int (dq_v + dq_c + dq_r) rho dz  =  -precipitation
    # In a steady state with q_r = 0 input, ``dq_r_dt = 0`` and the
    # column budget reduces to ``int (dq_v + dq_c) rho dz = -precipitation``.
    dq_v_dt = -rates.condensation + rates.evaporation
    dq_c_dt = rates.condensation - rates.autoconversion
    dt_safe = jnp.maximum(dt, 1e-10)
    q_r_in = jnp.clip(hydrometeors.q_r, 0.0, None)
    dq_r_dt = -q_r_in / dt_safe
    # Drained mass [kg/m^2/s] is added to the surface precipitation
    # diagnostic so total column water exits the column at the correct
    # rate.  ``rates.precipitation`` is the autoconversion-driven surface
    # flux; ``q_r_drain_flux`` is the column-integrated drain.
    #
    # The drain MUST use the SAME ``rho*dz`` mass weighting as the
    # tracer tendencies (``dq_r_dt`` removes ``q_r_in`` per unit air,
    # column-integrated as ``int dq_r * rho * dz``) and as the
    # autoconversion-driven precipitation (``rates.precipitation`` comes
    # from ``P_flux_layer = P_auto * rho * dz``).  An earlier form used
    # ``int q_r * dp / (g * dt)`` (the ``dp/g`` hydrostatic mass), which
    # closes the column water budget only when ``dp/g == rho*dz``
    # (hydrostatic balance) and leaks on a non-hydrostatically-consistent
    # profile — a latent mass-weighting inconsistency within one scheme.
    q_r_drain_flux = jnp.sum(q_r_in * rho * dz, axis=1) / dt_safe
    precipitation = rates.precipitation + q_r_drain_flux

    # Pin dtype to the input precision so we never silently promote
    # the unused-tendency placeholders to f64 under x64 mode.
    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=z,
        dq_s_dt=z,
        dq_g_dt=z,
        dN_c_dt=z,
        dN_r_dt=z,
        dN_i_dt=z,
        precipitation=precipitation,
    )
