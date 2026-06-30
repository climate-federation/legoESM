"""Bucket-hydrology runoff partition: infiltration excess (Green-Ampt-STYLE /
Hortonian) + saturation excess (Dunne), shared by the slab land tile
(``slab_land.step_land``) and the atmosphere-pipeline soil-water bucket
(``physics_pipeline._bucket_update``).

Saturation- and infiltration-excess ARE the appropriate runoff mechanisms for a
bucket/slab store (unlike the Richards column, where ponding is resolved by the
coupled surface cell).  One pure function so the two single-bucket codepaths never
re-derive the partition.

Green-Ampt-STYLE infiltration capacity, evaluated per step from the CURRENT bucket
deficit:

    f_max = K_s * (1 + B * (1 - W/W_max))      [m/s]

This is a Green-Ampt-INSPIRED bucket cap, NOT a faithful Green-Ampt update: it uses
the current bucket deficit as a static suction proxy and does NOT integrate the
within-storm cumulative infiltration F (a physics-timestep bucket cannot resolve
the storm hyetograph).  So it cannot reproduce the falling-rate Green-Ampt curve
within a step (capacity stays at the start-of-step value).  K_s is the saturated
infiltration capacity (Green-Ampt K), and B = psi_f/L_f the dimensionless suction
enhancement (wetting-front suction head over front depth): dry soil (deficit -> 1)
infiltrates at K_s*(1+B), saturated soil at K_s.  Rain above f_max is Hortonian
(infiltration-excess) runoff; rain that infiltrates but overfills the bucket is
Dunne (saturation-excess) runoff.  Reference (parameter values, not the per-step
form): Rawls, Brakensiek & Miller (1983) for K_s / psi_f; Mein & Larson (1973).

Preconditions, not re-guarded here for the JAX hot path: the SCALAR config params
W_max > 0, K_infiltration > 0, suction_boost >= 0 are enforced by the caller's
config validation (ExperimentConfig.validate_strict / LandConfig); the runtime
array P_input >= 0 is a caller-site precondition (precip + snowmelt are nonnegative
by construction).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants


def partition_bucket_runoff(W, P_input, evap_demand, dt, W_max,
                            K_infiltration, suction_boost,
                            infiltration_excess: bool = True,
                            limit_evaporation: bool = True):
    """Advance a soil-water bucket one step, partitioning precip into infiltration,
    Hortonian (infiltration-excess) runoff and Dunne (saturation-excess) runoff.

    Parameters
    ----------
    W : array            Current bucket water [kg/m2], in [0, W_max].
    P_input : array      Surface water input (rain + snowmelt) [kg/m2/s], >= 0.
    evap_demand : array  Bucket evaporation sink [kg/m2/s] (may be < 0 = dew).
    dt : float           Time step [s].
    W_max : array/float  Bucket capacity [kg/m2].
    K_infiltration : array/float  Saturated infiltration capacity K_s [m/s].
    suction_boost : array/float   Green-Ampt suction enhancement B = psi_f/L_f [-], >= 0.
    infiltration_excess : bool    If False the infiltration cap is disabled (all
        P_input infiltrates) — only saturation excess remains (fill-and-spill).
    limit_evaporation : bool      If True (the standalone slab tile), the bucket
        evaporation is throttled to the available water (W/dt + infiltration) so
        the lower clamp never creates water AND the caller MUST use the returned
        ``evap_actual`` for the latent-heat / surface-moisture flux (slab_land
        recomputes ``lhflx`` from it).  If False (the atmosphere-pipeline bucket,
        whose BL/energy flux is computed UPSTREAM from the beta-limited demand and
        cannot be revised here), the bucket consumes ``evap_demand`` verbatim so it
        stays CONSISTENT with that already-applied flux; the bucket may then clip
        at 0 under the beta-floor over-evaporation (the standard Manabe behaviour,
        identical to the prior bucket).  IN ISOLATION the False path is NOT water-
        conserving when evap_demand exceeds W/dt + infiltration (the clamp absorbs
        the deficit) — column conservation for that path rests on the caller
        keeping the demand reachable via the beta ramp (beta -> beta_min as W->0),
        NOT on re-limiting the bucket here.

    Returns
    -------
    (W_new, evap_actual, runoff, runoff_infil, runoff_sat)
        W_new [kg/m2], the evaporation actually applied to the bucket [kg/m2/s],
        total runoff and its infiltration- and saturation-excess parts [kg/m2/s].

    With ``limit_evaporation=True`` and valid inputs this conserves exactly:
    ``P_input == dW/dt + evap_actual + runoff``.
    """
    rho_w = constants.rho_water

    # Green-Ampt-style infiltration capacity from the current deficit [kg/m2/s].
    deficit = jnp.clip(1.0 - W / W_max, 0.0, 1.0)
    infil_cap = K_infiltration * rho_w * (1.0 + suction_boost * deficit)
    if infiltration_excess:
        infiltration = jnp.minimum(P_input, infil_cap)
    else:
        infiltration = P_input
    runoff_infil = P_input - infiltration                       # Hortonian, >= 0

    if limit_evaporation:
        # Throttle to what is in the bucket plus what just infiltrated, so the
        # lower clamp never creates water (dew, evap_demand < 0, is a source and is
        # never throttled).  The caller must apply evap_actual to its energy flux.
        max_evap = jnp.maximum(W / dt + infiltration, 0.0)
        evap_actual = jnp.minimum(evap_demand, max_evap)
    else:
        # Consume the demand verbatim to stay consistent with an externally-applied
        # latent flux; the clamp below may then bite at 0 (pre-existing behaviour).
        evap_actual = evap_demand

    W_unclamped = W + dt * (infiltration - evap_actual)
    runoff_sat = jnp.maximum(W_unclamped - W_max, 0.0) / dt     # Dunne, >= 0
    W_new = jnp.clip(W_unclamped, 0.0, W_max)

    runoff = runoff_infil + runoff_sat
    return W_new, evap_actual, runoff, runoff_infil, runoff_sat
