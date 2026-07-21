"""Sundqvist large-scale diagnostic condensation scheme.

A diagnostic scheme that activates condensation when relative humidity
exceeds a critical threshold. Produces large-scale (non-convective)
precipitation through autoconversion — enhanced per SBK89 by coalescence
with precipitation falling from above (F1) and the Bergeron-Findeisen
process in mixed phase (F2) — and sub-cloud evaporation.

All operations are AD-safe (differentiable ALMOST everywhere; the ``max``/``min``/
``clip`` guards and the donor cap introduce measure-zero kinks, not smoothness).

Faithfulness to Sundqvist (1978) / SBK89 (oracle = the published equations;
pinned in ``tests/atmosphere/hydrostatic/unit/test_sundqvist_faithful.py``)
--------------------------------------------------------------------------
FAITHFUL (forms reproduce the published autoconversion algebra):
  * The UNCAPPED, no-new-condensate release kernel
    ``P_auto = c_0·q_c·(1 - exp(-(q_c/q_c,crit)²))`` has the Sundqvist (1978) /
    SBK89 algebra (``c_0``=``auto_rate``, ``q_c,crit``=``qc_crit``). The
    IMPLEMENTED rate additionally uses ``qc_avail = max(q_c + condensation·dt, 0)``
    and a donor cap ``min(rate, qc_avail/dt)``, so the published algebra holds
    only where no new condensate enters and the cap does not bind.
  * The SBK89 (Sec. 5) enhancement STRUCTURE: the coalescence F1 and
    Bergeron-Findeisen F2 factors BOTH multiply the rate (``c_0 → c_0·F1·F2``)
    AND lower the onset threshold (``q_c,crit → q_c,crit/(F1·F2)``), i.e. the
    ``enh`` factor appears in the numerator of ``(q_c·enh/q_c,crit)`` and as the
    rate prefactor. F1 = ``1 + c1·√P_above`` is the SBK89 collection form.
DEPARTURES / SURROGATES (documented; NOT the SBK89 closed forms):
  * F2 uses a smooth GAUSSIAN-in-T window ``1 + c2·exp(-((T-T_peak)/T_width)²)``
    peaking near -15 °C as a proxy for SBK89's Bergeron driver, which is tied to
    the actual liquid-ice saturation difference ``e_sw - e_si`` (a ``1 + c2·√Δ``
    form), NOT a Gaussian. Same qualitative mixed-phase peak, different algebra.
  * Condensation is a SIMPLIFIED sigmoid-gated removal of SUPERSATURATION
    (``sigmoid(k·(RH-RH_crit))·max(q_v-q_sat,0)/dt``), driving q_v toward q_sat
    — not SBK89's full condensation-rate closure. The SBK89 partial cloud
    fraction ``b`` is diagnosed SEPARATELY in
    ``clouds.cloud_fraction.sundqvist_cloud_fraction`` and is not this
    microphysics tendency's concern.
  * Sub-cloud evaporation is a simplified ``evap_coeff·(RH<RH_crit gate)·P``
    proxy, not SBK89's full evaporation-rate expression.
  * F1's argument is the RAW precip flux ``√P_above`` (no SBK89 reference-flux
    normalisation); ``c1``/``c2`` are re-tunable closure coefficients, not the
    SBK89 paper values.
  * Precipitation is treated as DIAGNOSTIC: the autoconversion source is
    accumulated into a one-step downward flux (with the evaporation proxy and a
    one-step ``q_r`` drain). This transport treatment is an IMPLEMENTATION choice,
    not one of the supplied SBK89 forms — no paper faithfulness is claimed for it.

References
----------
- Sundqvist, H. (1978): A parameterization scheme for non-convective
  condensation including prediction of cloud water content. Quart. J. Roy.
  Meteor. Soc., 104, 677-690. (Base autoconversion release form.)
- Sundqvist, Berge & Kristjansson (1989): Condensation and cloud
  parameterization studies with a mesoscale numerical weather prediction
  model. Mon. Wea. Rev., 117, 1641-1657. (F1/F2 enhancements, Sec. 5.)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.microphysics._warm_rain import safe_pow
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

    # 2. Autoconversion (computed level-by-level INSIDE the downward scan
    # below, because the SBK89 coalescence enhancement F1 depends on the
    # precipitation flux falling in from above).
    # Cap the autoconversion against the available cloud water + new
    # condensation so an explicit Euler step (``q_c_new = q_c + dt·dq_c_dt``)
    # never drives q_c below zero.  Without this, default
    # ``auto_rate · dt = 1e-3·1800 = 1.8`` over ~30 min for typical
    # ``q_c ≈ 1e-4 kg/kg`` overshoots the available mass by ~80 %.
    qc_avail = jnp.maximum(q_c + condensation * dt, 0.0)
    dt_safe = jnp.maximum(dt, 1.0e-12)

    # SBK89 Bergeron-Findeisen enhancement F2 (per level, T-dependent): a
    # smooth Gaussian window centred near −15 °C where the liquid-ice
    # saturation difference e_sw − e_si (the Bergeron growth driver) peaks;
    # ≈ 1 above freezing by construction (the Gaussian tail).  Smooth in T
    # (AD-safe everywhere).
    bergeron_f2 = 1.0 + config.bergeron_enh_coeff * jnp.exp(
        -((T - config.bergeron_T_peak_K) / config.bergeron_T_width_K) ** 2
    )

    # 3. Precipitation release + sub-cloud evaporation (single downward scan)
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
    # Sign convention: precipitation flux P is positive DOWNWARD [kg/m²/s].
    evap_mask = jax.nn.sigmoid(sharpness * (config.rh_crit - RH))

    def scan_fn(carry, x):
        P_above = carry
        qc_k, berg_k, evap_m, rho_k, dz_k = x
        # SBK89 coalescence enhancement F1 = 1 + c1·√P_above from the
        # precipitation flux entering the layer from above: falling
        # precipitation collects cloud water, accelerating release.
        # safe_pow guards the √P AD trap at P_above = 0 (no-precip columns).
        coal_f1 = 1.0 + config.coalescence_enh_coeff * safe_pow(P_above, 0.5)
        enh = coal_f1 * berg_k
        # SBK89 release: c_0 → c_0·F1·F2 AND q_c,crit → q_c,crit/(F1·F2)
        # (both the rate and the onset threshold are enhanced), applied to
        # the Sundqvist (1989) base form
        #   P_auto = c_0·q_c·(1 − exp(−(q_c/q_c,crit)²)).
        # The threshold factor ∈ [0,1) is smooth + AD-safe (exp of a
        # non-positive arg; → 0 as q_c → 0, → 1 for q_c ≫ q_c,crit).
        threshold = 1.0 - jnp.exp(
            -(qc_k * enh / jnp.maximum(config.qc_crit, 1e-12)) ** 2
        )
        # Donor cap: rate · dt ≤ qc_avail → rate ≤ qc_avail / dt.
        P_auto_k = jnp.minimum(
            config.auto_rate * enh * qc_k * threshold, qc_k / dt_safe,
        )
        P_total = P_above + P_auto_k * rho_k * dz_k
        evap = config.evap_coeff * evap_m * P_total / jnp.clip(rho_k * dz_k, 1.0)
        evap = jnp.minimum(evap, P_total / jnp.clip(rho_k * dz_k, 1.0))
        P_out = jnp.clip(P_total - evap * rho_k * dz_k, 0.0)
        return P_out, (P_auto_k, evap)

    # Pick a working dtype that ``scan`` can carry without promotion.
    # Under ``JAX_ENABLE_X64=1`` ``jnp.zeros``/``jnp.ones`` default to
    # f64, so a state assembled from a mix of (f32) ``T`` and (f64)
    # tracers ends up with f64 ``q_v``/``q_c``.  ``qc_avail`` inherits
    # the f64 promotion from ``q_c + condensation * dt``, while a carry
    # pinned to ``T.dtype`` (f32) would mismatch the f64 scan output.
    # Promoting to the wider of carry/input dtype keeps ``scan`` happy
    # without silently downcasting precipitation mass.
    _scan_dtype = jnp.promote_types(T.dtype, qc_avail.dtype)
    inputs = (
        jnp.moveaxis(qc_avail.astype(_scan_dtype), 1, 0),
        jnp.moveaxis(bergeron_f2.astype(_scan_dtype), 1, 0),
        jnp.moveaxis(evap_mask.astype(_scan_dtype), 1, 0),
        jnp.moveaxis(rho.astype(_scan_dtype), 1, 0),
        jnp.moveaxis(dz.astype(_scan_dtype), 1, 0),
    )
    P_init = jnp.zeros(T.shape[0], dtype=_scan_dtype)
    P_final, (auto_col, evap_col) = jax.lax.scan(scan_fn, P_init, inputs)
    autoconversion = jnp.moveaxis(auto_col, 0, 1)
    evaporation = jnp.moveaxis(evap_col, 0, 1)
    return SundqvistProcessRates(
        condensation=condensation,
        autoconversion=autoconversion,
        evaporation=evaporation,
        precipitation=P_final,
    )


__physics_contract__ = {
    "summary": (
        "Sundqvist (1978)/SBK89 large-scale diagnostic condensation: simplified "
        "RH-gated vapor condensation and sub-cloud evaporation of diagnostic "
        "precipitation, plus the Sundqvist (1978) "
        "autoconversion release P_auto=c_0*q_c*(1-exp(-(q_c/q_c,crit)^2)), enhanced "
        "per SBK89 by coalescence with precipitation from above (F1=1+c1*sqrt(P)) "
        "and a Bergeron-Findeisen mixed-phase factor (F2, a smooth Gaussian-in-T "
        "proxy for SBK89's e_sw-e_si driver). See the module docstring's "
        "Faithfulness section for the faithful-form vs surrogate split."
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
