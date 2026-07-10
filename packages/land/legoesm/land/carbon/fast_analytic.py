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
    ``Q10_het_exp``, ``cwd_humification_eff``) affect ONLY the SOM decomposition.
    There is NO SOM -> GPP feedback, so the SOM active-pool INPUTS
    (litter -> SOM decomposition ``lit_to_som`` + wood allocation ``a_wood``) and
    the soil-temperature trajectory the decomposition modifier sees are
    INDEPENDENT of those parameters.

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
(~6-8% of total SOC at the defaults; verified) is the dropped within-year
COVARIANCE between ``C_X(t)``, the input timing, and ``r_X(t)`` — chiefly the
~annual active pool — NOT a sign/unit error; it does not compromise the gradient
directions the calibration follows.

Which parameters are fast-trainable.  ALL EIGHT SOM fields.  ``lit_to_som`` is
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
    """Per-archetype, tunable-parameter-INDEPENDENT inputs for the fast closed-form
    forwards -- the SOM-SOC cascade (:func:`analytic_som_soc`) AND the live-pool
    biomass/LAI equilibrium
    (:func:`legoesm.land.carbon.live_pool_forward.build_live_pool_forward`).

    Recorded ONCE by
    :func:`legoesm.land.carbon.global_init.precompute_fast_analytic_inputs` from a
    single default-parameter forward spin-up (no grad).  Every field is a per-
    archetype array (leading axis ``n_arch``) except ``dt_days`` (scalar):

    lit_to_som_annual : (n_arch,) [gC/m2/yr]
        Stationary annual litter -> active-SOM decomposition flux.  Independent of
        the tunable SOM parameters (litter-pool throughput = litterfall * the
        fixed ``decomp_rate``/``tor_litter`` split; the shared modifier cancels).
    a_wood_annual : (n_arch,) [gC/m2/yr]
        Stationary annual NPP allocation to wood, used as the wood->SOM input
        driver ``wood_to_som = cwd_humification_eff * a_wood`` -- EXACTLY as the
        reference reset :func:`analytic_slow_pool_equilibrium` does (it takes
        ``a_wood`` as the wood turnover at wood equilibrium).  Kept SEPARATE from
        ``lit_to_som_annual`` so ``cwd_humification_eff`` enters the active-pool
        input analytically.  Using the raw per-step wood TURNOVER instead
        REGRESSES the analytic-vs-spin-up match (7.5% -> 13.9%): the coupled wood
        pool is a small net sink in the recorded year (``a_wood > wood_litter``,
        wood not perfectly equilibrated), and the reference equilibrium is built
        from ``a_wood`` -- so ``a_wood`` is the faithful reproduction.  At TRUE
        wood equilibrium the two coincide.
    soil_T_traj : (n_arch, n_samples) [K]
        Top-soil-layer temperature the SOM decomposition modifier sees, sampled
        every sub-daily step over ONE stationary year (the full seasonal +
        diurnal cycle).  SOM-parameter-independent (no SOM -> energy feedback).
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

        r_X(t)  = _effective_rate(_som_decomp_modifier(T(t), precip, config) * tor_X, dt)
        k_X     = sum_t r_X(t) * dt_days                        [1/yr] annual turnover
        i_active = lit_to_som_annual + cwd_humification_eff * a_wood_annual
        (C_active, C_slow, C_passive) = _forward_substitute_cascade(i_active, k_X..., f...)
        SOC = C_active + C_slow + C_passive

    ``k_X = sum_t r_X(t) dt_days`` is the realised annual turnover ``D_X / C_X``
    :func:`analytic_slow_pool_equilibrium` infers from a spun-up pool, and the
    constant-within-year balance ``C_X = i_X / k_X`` is the SAME algebra it uses.
    This is an ANNUAL-EQUILIBRIUM APPROXIMATION of the stepped periodic fixed
    point (see the module docstring), NOT an exact solve; at the default
    parameters it tracks ``som_total(equilibrate_archetypes)`` per archetype to
    ~6-8% (verified), the residual being the dropped within-year covariance.

    Differentiability.  ``config`` carries the eight TRACED SOM leaves
    (``tor_som_active`` / ``tor_som_slow`` / ``tor_som_passive`` /
    ``f_active_to_slow`` / ``f_slow_to_passive`` / ``som_freeze_floor`` /
    ``Q10_het_exp`` / ``cwd_humification_eff``) spliced in by
    ``param_collector.apply_param_overrides`` INSIDE the loss; ``jax.grad`` of any
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

    # Realised annual turnover per pool [1/yr] = sum_t r_X(t) * dt_days, using the
    # model's EXACT modifier + finite-dt kinetics on the recorded trajectory.
    def _annual_turnover(tor_som: jax.Array) -> jax.Array:
        r = som_decomposition_rate(temp, precip, tor_som, config, dt_days)
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
