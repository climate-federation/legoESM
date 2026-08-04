"""Fast closed-form differentiable SOM equilibrium for carbon calibration.

Stage-B (B2-fast) forward map.  The full B1 forward
(:func:`legoesm.land.carbon.global_init.equilibrate_archetypes_traced`)
differentiates the per-archetype equilibrium SOC through the coupled multi-year
land+carbon spin-up + reverse-mode remat — correct but ~tens of minutes per
optimizer step, so a calibration cannot converge.  This module replaces that
forward with a CLOSED FORM that is differentiable in the SOM parameters at ~zero
cost, exploiting one structural fact of DifferLand:

    the tunable SOM parameters (``tor_som_active/slow/passive``,
    ``f_active_to_slow`` / ``f_slow_to_passive``, ``som_freeze_floor``,
    ``Q10_het_exp``, ``cwd_humification_eff``) act on the SOM cascade with NO
    SOM -> GPP feedback, so the wood allocation ``a_wood`` and the soil-temperature
    trajectory the decomposition modifier sees are INDEPENDENT of them.  The
    litter -> SOM decomposition input ``lit_to_som`` is independent of the SEVEN
    fast-trained params; ``Q10_het_exp`` DOES also enter the litter modifier
    (``_temperate_modifier`` sets ``lit_to_som``), so freezing ``lit_to_som`` at the
    defaults gives ``Q10_het_exp`` only a PARTIAL fast-mode gradient -- which is why
    the calibration's fast trainable set excludes it (see below).

So a SINGLE default-parameter forward spin-up per archetype
(:func:`legoesm.land.carbon.global_init.precompute_fast_analytic_inputs`, no
grad) records the param-independent stationary inputs + the stationary top-soil
temperature trajectory (:class:`FastAnalyticInputs`); thereafter
:func:`analytic_som_soc` computes the per-archetype equilibrium SOM SOC DIRECTLY
from the config rates, forward-substituting the CENTURY cascade steady state —
no spin-up in the gradient loop.

Fidelity to the model.  This is an ANNUAL-EQUILIBRIUM approximation of the
stepped SOM cascade, NOT an exact closed form of its periodic fixed point: it
holds each pool constant within the year when balancing input against output
(``C_X = I_annual / k_annual``), which is the SAME constant-pool cascade algebra
:func:`analytic_slow_pool_equilibrium` uses.  The per-pool turnover reuses the
model's EXACT kinetics
(:func:`legoesm.land.carbon.carbon_cycle.som_decomposition_rate` =
``_effective_rate(_som_decomp_modifier * tor, dt_days)``) evaluated on the
recorded soil-temperature trajectory, so ``k_X = sum_t r_X(t) * dt_days`` is the
SAME realised annual turnover ``k_X = D_X / C_X`` inferred from a spun-up pool's
loss/stock.  Capturing the full seasonal + diurnal temperature trajectory (rather
than a single mean-annual T) is REQUIRED: the modifier is convex in T (Q10
exponential + freeze sigmoid), so ``<m(T(t))>`` far exceeds ``m(<T>)`` for the
large seasonal amplitude of cold archetypes — a mean-T closed form would
over-predict cold-soil SOC by a large factor.  The residual vs the stepped model
is the dropped within-year COVARIANCE between ``C_X(t)``, the input timing, and
``r_X(t)`` — largest for the ~annual active pool, but that pool is only ~5% of
total SOC and its ~3-yr residence averages the annual cycle so its covariance
correction is a fraction of a % of SOC (an EXACT periodic active-pool solve was
verified fidelity-neutral).  NOT a sign/unit error; it does not compromise the
gradient directions the calibration follows.

The DOMINANT residual is instead the CWD-humification input consistency, now
fixed (see ``a_wood_annual``): the reference reset
:func:`analytic_slow_pool_equilibrium` builds ``I_active = lit_to_som +
cwd_humification_eff * a_wood`` from the LAST-TRANSIENT-year fluxes, and the closed
form now records ``lit_to_som``/``a_wood`` from THOSE SAME reset fluxes rather than
a post-verify stationary year.  Sampling ``a_wood`` post-verify had shifted it
~8-11% below the reset value on the still-equilibrating ~27-yr wood pool (the reset
boosts ``C_wood`` -> higher maintenance respiration -> lower NPP/allocation), which
propagated through the whole active->slow->passive cascade and made the closed form
systematically UNDER-predict the passive pool (the bulk of SOC) for woody
archetypes — the passive/slow bias this record-source fix removes.

Which parameters are fast-differentiable.  The closed form is differentiable in
all EIGHT SOM leaves (verified: :func:`jax.grad` of the SOC loss is finite and
non-zero for each).  The calibration's fast trainable SET is the SEVEN excluding
``Q10_het_exp`` (see ``train_carbon_params.SOM_FIELDS``): ``Q10_het_exp`` also
enters the PRECOMPUTED (frozen) ``lit_to_som`` litter input, so its fast-mode
gradient is only PARTIAL (the live SOM-cascade path, not the frozen-litter path)
-- the seven kept fields have their FULL gradient.  ``lit_to_som`` is
param-independent at the litter steady state (its throughput is set by the
litterfall input and the ``decomp_rate``/``tor_litter`` split, in which the
shared temperature modifier cancels), so it is precomputed once.  The
CWD-humification input is precomputed as the SEPARATE ``a_wood`` allocation, so
``cwd_humification_eff`` enters the closed form analytically
(``I_active = lit_to_som + cwd_humification_eff * a_wood``, mirroring
:func:`analytic_slow_pool_equilibrium`, which drives the reference equilibrium;
this reproduces the model's own reset -- using the raw per-step ``wood_litter``
turnover instead regresses the match, since the coupled wood pool is a small net
sink in the recorded year).  The two remaining approximations are small and
documented on :func:`analytic_som_soc`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.land.carbon.carbon_cycle import som_decomposition_rate
from legoesm.land.carbon.config import CarbonConfig

# Divide-safety floor on an annual turnover ``k`` [1/yr] (<= 1e-6 numerics
# guard; the sanctioned training producer maps ``som_freeze_floor`` through a
# sigmoid onto a strictly-positive bound so ``k > 0`` always -- this floor only
# guards a hand-crafted degenerate input, never the calibration path).
_K_FLOOR_PER_YR = 1e-12


class FastAnalyticInputs(NamedTuple):
    """Per-archetype inputs for the fast closed-form forwards -- the SOM-SOC cascade
    (:func:`analytic_som_soc`) AND the live-pool biomass/LAI equilibrium
    (:func:`legoesm.land.carbon.live_pool_forward.build_live_pool_forward`) --
    independent of the SEVEN fast-trained SOM params.  (``Q10_het_exp`` also enters
    the frozen ``lit_to_som`` litter modifier, so it is NOT in the fast trainable
    set -- a documented partial gradient; see the module docstring.)

    Recorded ONCE by
    :func:`legoesm.land.carbon.global_init.precompute_fast_analytic_inputs` from a
    single default-parameter forward spin-up (no grad).  Every field is a per-
    archetype array (leading axis ``n_arch``) except ``dt_days`` (scalar):

    lit_to_som_annual : (n_arch,) [gC/m2/yr]
        Annual litter -> active-SOM decomposition flux, taken from the
        LAST-TRANSIENT-year fluxes the reference reset
        :func:`analytic_slow_pool_equilibrium` consumes (``SlowPoolFluxes.lit_to_som``,
        exposed by :func:`~legoesm.land.carbon.spinup.run_semi_analytic_spinup`), NOT
        a separate post-verify stationary year.  Recording the SAME flux the reset
        uses makes the closed form reproduce the reset's active-pool INPUT chain
        exactly, removing the input-phase bias (the residual vs the spin-up is only
        the ``k_X`` turnover, evaluated on the post-verify soil-T -- ~0.5% on the
        tiny-world CRUX test, was ~7.5%).  Independent of the SEVEN fast-trained SOM
        params (litter-pool throughput = litterfall * the fixed ``decomp_rate``/
        ``tor_litter`` split; the shared modifier cancels).  ``Q10_het_exp`` DOES
        scale the litter modifier, so freezing this input gives it only a partial
        fast gradient -- hence its exclusion from the fast trainable set.
    a_wood_annual : (n_arch,) [gC/m2/yr]
        Annual NPP allocation to wood the wood->SOM humification input driver
        ``wood_to_som = cwd_humification_eff * a_wood`` uses, taken from the SAME
        LAST-TRANSIENT-year reset fluxes (``SlowPoolFluxes.a_wood``) so it MATCHES
        the value the reference reset :func:`analytic_slow_pool_equilibrium` built
        ``real_som`` from (the reset takes ``a_wood`` as the wood turnover at wood
        equilibrium).  Kept SEPARATE from ``lit_to_som_annual`` so
        ``cwd_humification_eff`` enters the active-pool input analytically.  Two
        reasons this exact source matters (both verified):
          * Using the raw per-step wood TURNOVER ``wood_litter`` instead of ``a_wood``
            regresses the match (the reference reset is built from ``a_wood``, so the
            surrogate must use ``a_wood`` too; at TRUE wood equilibrium they coincide).
          * Recording ``a_wood`` from a POST-VERIFY stationary year instead of the
            reset year sampled a DIFFERENT phase of the ~27-yr (``tor_wood=1e-4/day``)
            wood pool: the reset boosts ``C_wood`` -> raises maintenance respiration ->
            lowers NPP/allocation, so post-verify ``a_wood`` ran ~8-11% BELOW the
            transient value the reset used, and the surrogate systematically
            UNDER-predicted the CWD-fed active->slow->passive cascade (a passive-pool
            bias of ~10-15% cover-weighted for woody archetypes).  Recording the reset
            flux removes that inconsistency; grasses (``a_wood==0``) are unaffected.
    soil_T_traj : (n_arch, n_samples) [K]
        Top-soil-layer temperature the SOM decomposition modifier sees, sampled
        every sub-daily step over ONE stationary year (the full seasonal +
        diurnal cycle).  SOM-parameter-independent (no SOM -> energy feedback).
    soil_frozen_fraction : (n_arch,) [-]
        Per-archetype ANNUAL frozen fraction in [0, 1] -- the perennial-frost
        index driving the permafrost/anaerobic SOM protection
        (:func:`legoesm.land.carbon.carbon_cycle.perennial_frost_protection`).
        Computed by :func:`~legoesm.land.carbon.global_init.
        precompute_fast_analytic_inputs` from the climatological annual
        temperature cycle (:func:`~legoesm.land.carbon.carbon_cycle.
        annual_frozen_fraction` on the archetype ``mat_k`` /
        ``t_seasonal_amp_k``) -- the SAME per-column value the coupled spin-up's
        ``step_carbon_differland`` applies, so the closed form and the spin-up
        rescale each pool's turnover by a BYTE-IDENTICAL ``f_perma`` (the
        surrogate-vs-spin-up fidelity gate stays ratio-invariant).  Climate-only,
        so param-independent and frozen like ``soil_T_traj``.
    precip : (n_arch,) [kg/m2/s]
        Per-archetype precipitation rate (constant over the year in the archetype
        forcing) driving the moisture modifier ``f_moist``.
    dt_days : float
        Sub-daily timestep [day], used both by ``_effective_rate`` and to weight
        the per-step turnover into an annual total (``k_X = sum_t r_X * dt_days``).
    npp_pos_annual : (n_arch,) [gC/m2/yr]
        Stationary annual ALLOCATABLE NPP ``sum_t max(NPP_day, 0) dt`` -- the
        positive net primary production the model PARTITIONS in
        ``step_carbon_differland`` (``A_x = a_x * max(NPP_day, 0)``).  The frozen
        driver of the closed-form LIVE pools
        (:func:`legoesm.land.carbon.live_pool_forward.compute_live_pools`): NPP is
        upstream of the trained allocation/residence/LCMA leaves, so freezing it at
        the defaults is EXACT for the DIRECT pool effect their gradient follows
        (mirroring ``lit_to_som_annual`` and the SIF forward's frozen leaf state); a
        SECOND-ORDER pools->LAI/respiration->NPP feedback is omitted (a documented
        partial gradient, like ``Q10_het_exp`` in the fast SOM set -- see
        ``live_pool_forward``).
    is_woody : (n_arch,) [-]
        Per-archetype woody flag (``1.0`` woody / ``0.0`` herbaceous), the STATIC
        per-group ``CarbonConfig.woody`` used to route the structural allocation
        remainder in the live-pool forward (woody -> wood; herbaceous -> roots).
    is_evergreen : (n_arch,) [-]
        Per-archetype evergreen flag (``1.0`` evergreen / ``0.0`` deciduous), the STATIC
        per-group ``CarbonConfig.evergreen`` selecting the live-pool leaf-turnover branch
        (continuous ``1/leaf_lifespan`` vs the DALEC990 Gaussian leaf-fall year-integral).
    live_ref_C_fol, live_ref_C_root, live_ref_C_wood : each (n_arch,) [gC/m2]
        The default-parameter spin-up's ANNUAL-MEAN equilibrium live pools (averaged
        over the recorded stationary year), the REFERENCE the closed-form live-pool
        forward is checked against (the biomass/LAI fidelity gate) -- the live-pool
        analogue of ``som_total_equilibrium`` for SOC.  Annual-MEAN (not an
        end-of-year snapshot) because the closed form is a throughput x residence
        annual-mean stock, so the fair comparison for the seasonal foliage pool is
        the annual mean.
    """
    lit_to_som_annual: jax.Array
    a_wood_annual: jax.Array
    soil_T_traj: jax.Array
    soil_frozen_fraction: jax.Array
    precip: jax.Array
    dt_days: float
    npp_pos_annual: jax.Array
    is_woody: jax.Array
    is_evergreen: jax.Array
    live_ref_C_fol: jax.Array
    live_ref_C_root: jax.Array
    live_ref_C_wood: jax.Array


def _forward_substitute_cascade(
    i_active: jax.Array,
    k_active: jax.Array,
    k_slow: jax.Array,
    k_passive: jax.Array,
    f_active_to_slow: jax.Array,
    f_slow_to_passive: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Forward-substitution steady state of the lower-triangular CENTURY cascade.

    The SAME recurrence :func:`analytic_slow_pool_equilibrium` implements, but
    parameterised by each pool's annual turnover ``k_X`` [1/yr] DIRECTLY (from the
    config rates + recorded climate) instead of inferring it from a spun-up
    pool's loss/stock.  For a linear pool ``dC/dt = I - k C`` the steady state is
    ``C = I / k``; at that steady state the pool's loss equals its input, so the
    humified transfer to the next-slower pool is ``f * I`` (forward cascade, no
    back-transfer):

        C_active  = i_active / k_active           ; loss_active  = i_active
        i_slow    = f_active_to_slow  * i_active   ; C_slow = i_slow / k_slow
        i_passive = f_slow_to_passive * i_slow     ; C_passive = i_passive / k_passive

    Unlike :func:`analytic_slow_pool_equilibrium` there is NO degenerate
    leave-unchanged gating: the calibration constrains ``som_freeze_floor`` (hence
    the modifier ``m``) strictly positive, so every ``k_X > 0`` and each pool has
    a finite equilibrium (``_K_FLOOR_PER_YR`` only guards a hand-crafted input).

    All arguments broadcast over the archetype axis; returns
    ``(C_active, C_slow, C_passive)`` [gC/m2].
    """
    c_active = i_active / jnp.maximum(k_active, _K_FLOOR_PER_YR)
    i_slow = f_active_to_slow * i_active            # equilibrium: loss_active == i_active
    c_slow = i_slow / jnp.maximum(k_slow, _K_FLOOR_PER_YR)
    i_passive = f_slow_to_passive * i_slow          # equilibrium: loss_slow == i_slow
    c_passive = i_passive / jnp.maximum(k_passive, _K_FLOOR_PER_YR)
    return c_active, c_slow, c_passive


def analytic_som_soc(
    precomputed: FastAnalyticInputs,
    config: CarbonConfig,
) -> jax.Array:
    """Closed-form per-archetype total SOM SOC [gC/m2], differentiable in *config*.

    Computes the CENTURY active/slow/passive cascade equilibrium DIRECTLY from the
    ``config`` SOM rates + the precomputed param-independent inputs, with NO
    spin-up in the gradient loop.  Reuses the model's exact per-pool kinetics
    (:func:`legoesm.land.carbon.carbon_cycle.som_decomposition_rate`) on the
    recorded stationary soil-temperature trajectory:

        r_X(t)  = _effective_rate(
                      _som_decomp_modifier(T(t), precip, config, phi_frozen) * tor_X, dt)
        k_X     = sum_t r_X(t) * dt_days                        [1/yr] annual turnover
        i_active = lit_to_som_annual + cwd_humification_eff * a_wood_annual
        (C_active, C_slow, C_passive) = _forward_substitute_cascade(i_active, k_X..., f...)
        SOC = C_active + C_slow + C_passive

    ``k_X = sum_t r_X(t) dt_days`` is the realised annual turnover ``D_X / C_X``
    :func:`analytic_slow_pool_equilibrium` infers from a spun-up pool, and the
    constant-within-year balance ``C_X = i_X / k_X`` is the SAME algebra it uses.
    This is an ANNUAL-EQUILIBRIUM APPROXIMATION of the stepped periodic fixed
    point (see the module docstring), NOT an exact solve; with the SOM active-pool
    inputs recorded from the reset's own fluxes (see ``FastAnalyticInputs.
    a_wood_annual``) it tracks ``som_total(equilibrate_archetypes)`` per archetype to
    ~0.5-1.5% at the default parameters (verified; cover-weighted ~1.2% on the 24-
    archetype world, was ~13-16% when ``a_wood`` was sampled post-verify).  The small
    residual is the ``k_X`` turnover, evaluated on the post-verify soil-T rather than
    the reset's realised loss (the dropped within-year covariance, chiefly the
    ~annual active pool -- a fraction of a % of SOC).

    Differentiability.  ``config`` carries the eight TRACED SOM leaves
    (``tor_som_active`` / ``tor_som_slow`` / ``tor_som_passive`` /
    ``f_active_to_slow`` / ``f_slow_to_passive`` / ``som_freeze_floor`` /
    ``Q10_het_exp`` / ``cwd_humification_eff``) spliced in by
    ``legoesm.core.param_overrides.apply_param_overrides`` INSIDE the loss; ``jax.grad`` of any
    SOC-based loss flows to all eight at ~zero cost.  ``som_freeze_floor`` and
    ``Q10_het_exp`` enter through the per-sample modifier ``m(T(t))``, so the full
    seasonal/diurnal temperature integral (not a mean-T proxy) sets their effect.

    Approximations (both small; see the module docstring for why they hold):
      1. The SOM pools are held at their within-year-CONSTANT equilibrium when
         forming ``k_X`` — exact for the decadal/centennial slow+passive pools
         (which dominate SOC) and a small covariance error for the ~annual active
         pool (a few % of total SOC).
      2. ``lit_to_som_annual`` is treated as SOM-parameter-independent — exact for
         the turnover/transfer/freeze parameters (absent from the litter
         equations) and a negligible residual for ``Q10_het_exp`` (the modifier
         cancels in the deeply-linear litter split ratio).

    Parameters
    ----------
    precomputed : FastAnalyticInputs
        Param-independent inputs from
        :func:`~legoesm.land.carbon.global_init.precompute_fast_analytic_inputs`.
    config : CarbonConfig
        Carbon config carrying the (possibly traced) SOM parameters.  Only the SOM
        turnover/transfer/freeze/Q10/CWD fields + the modifier reference fields
        (``T_ref`` / ``precip_ref`` / ``moisture_factor`` /
        ``moist_modifier_min`` / ``moist_modifier_max`` / ``som_freeze_width_K``)
        are read; GPP / phenology / allocation fields are ignored (their effect is
        baked into the precomputed inputs).

    Returns
    -------
    jax.Array
        Per-archetype total SOM SOC ``(n_arch,)`` [gC/m2]
        (``C_som_active + C_som_slow + C_som_passive``), differentiable in the
        ``config`` SOM leaves.
    """
    dt_days = precomputed.dt_days
    temp = precomputed.soil_T_traj                    # (n_arch, n_samples) [K]
    precip = precomputed.precip[:, None]              # (n_arch, 1) -> broadcasts over samples
    # Per-column perennial-frost index (annual frozen fraction) -> the SAME
    # per-column protection factor ``f_perma`` the coupled ``step_carbon_differland``
    # applies (byte-identical, so the fidelity gate is ratio-invariant).  Broadcast
    # over the sample axis; ``f_perma`` scales the modifier inside every r_X(t).
    frozen_fraction = precomputed.soil_frozen_fraction[:, None]   # (n_arch, 1)

    # Realised annual turnover per pool [1/yr] = sum_t r_X(t) * dt_days, using the
    # model's EXACT modifier + finite-dt kinetics on the recorded trajectory.
    def _annual_turnover(tor_som: jax.Array) -> jax.Array:
        r = som_decomposition_rate(
            temp, precip, tor_som, config, dt_days,
            frozen_fraction=frozen_fraction)
        return jnp.sum(r, axis=1) * dt_days           # (n_arch,)

    k_active = _annual_turnover(config.tor_som_active)
    k_slow = _annual_turnover(config.tor_som_slow)
    k_passive = _annual_turnover(config.tor_som_passive)

    # Active-pool input: litter -> SOM decomposition + humified CWD, mirroring the
    # reference reset ``analytic_slow_pool_equilibrium``'s
    # ``i_active = lit_to_som + cwd_humification_eff * a_wood`` (wood turnover taken
    # at wood equilibrium == a_wood) so the closed form reproduces the model's own
    # equilibrium; see FastAnalyticInputs.a_wood_annual for why a_wood (not the
    # raw per-step wood_litter) is the faithful choice.
    i_active = (
        precomputed.lit_to_som_annual
        + config.cwd_humification_eff * precomputed.a_wood_annual
    )
    c_active, c_slow, c_passive = _forward_substitute_cascade(
        i_active, k_active, k_slow, k_passive,
        config.f_active_to_slow, config.f_slow_to_passive)
    return c_active + c_slow + c_passive
