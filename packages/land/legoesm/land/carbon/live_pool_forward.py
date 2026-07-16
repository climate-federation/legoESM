"""Per-archetype closed-form LIVE-pool (biomass + LAI) forward for Stage-B carbon calibration.

The MODELLED counterpart of the CLM ``MONTHLY_LAI`` / gridded above-ground-biomass
observations (:mod:`legoesm.land.carbon.lai_observations` and the biomass fetcher
follow-up): a per-archetype simulated live-pool biomass [kgC/m2] and leaf-area index
[m2/m2], differentiable in the DifferLand allocation / residence / leaf-mass-per-area
:class:`~legoesm.land.carbon.config.CarbonConfig` parameters.

Like the SIF forward (:mod:`legoesm.land.carbon.sif_forward`) and UNLIKE the SOM-SOC
forward (:func:`legoesm.land.carbon.global_init.equilibrate_archetypes_traced`), this is a
**single-step** diagnostic -- there is NO multi-year spin-up scan in the gradient loop.  It
exploits one structural fact of the DifferLand LIVE pools (foliage, root, wood): at
steady state each is a LINEAR pool ``dC/dt = I - k C`` fed by an NPP allocation input
``I = a_pool * NPP`` and drained by a first-order turnover ``k`` [1/yr], so its stationary
stock is the closed form ``C_pool = I / k`` (Olson 1963; stock = throughput x residence)::

    C_fol   = a_fol  * NPP / k_leaf              [gC/m2]
    C_root  = a_root * NPP / k_root              [gC/m2]  ; k_root = tor_root * days_per_year
    C_wood  = a_wood * NPP / k_wood              [gC/m2]  ; k_wood = tor_wood * days_per_year
    LAI     = C_fol / LCMA                        [m2/m2]
    biomass = (C_fol + C_root + C_wood) / 1000    [kgC/m2]

The allocation fractions ``a_pool`` reuse the model's OWN sequential DALEC partition
(:func:`legoesm.land.carbon.carbon_cycle.sequential_allocation`) -- never re-derived here
-- with the same woody/herbaceous routing of the structural remainder that
:func:`~legoesm.land.carbon.carbon_cycle.step_carbon_differland` uses (woody: to wood;
herbaceous: invested belowground).

Foliage input = the DIRECT allocation ``a_fol * NPP`` only.  The labile allocation
``a_lab`` funds a SEPARATE buffer pool (spring bud-burst release + nighttime
respiration-deficit draw) that is NOT part of the standing biomass (``biomass`` excludes
``C_lab``); over the annual cycle its NET contribution to the standing foliage stock is
second-order and archetype-dependent (in a warm canopy the labile is largely drained to
cover 24 h maintenance respiration, ``lab_release_annual << A_lab``), so the leading-order
annual-mean foliage is set by the direct allocation -- verified against the
default-parameter spin-up equilibrium (the fidelity gate); the residual labile->foliage
routing is a documented approximation.

Leaf turnover ``k_leaf`` [1/yr] reuses the model's OWN phenology
(:func:`legoesm.land.carbon.carbon_cycle.compute_phenology`) integrated over one year --
NOT a re-derived ``1/leaf_lifespan``, which holds ONLY for the evergreen (continuous)
branch.  For the DECIDUOUS DALEC990 Gaussian leaf-fall pulse the realised annual turnover
is the year-integral of the leaf-fall fraction ``k_leaf = sum_doy lff(doy)``, which the
compact leaf-fall window makes SUBSTANTIALLY larger than ``1/leaf_lifespan`` (the canopy is
shed over ~a leaf-fall period, not spread across the whole year) -- so a ``1/leaf_lifespan``
residence over-predicts the deciduous annual-mean foliage by ~2-3x.  ``k_leaf`` is selected
per archetype from the recorded static ``is_evergreen`` flag; both branches are the model's
own ``compute_phenology`` output summed, so the leaf residence is never re-derived.  Root
and wood keep the first-order ``tor_root``/``tor_wood`` turnover the prognostic step drains
them with.

NPP source.  ``NPP`` is the model's OWN annual allocatable NPP
``npp_pos_annual = sum_t max(NPP_day(t), 0) dt`` [gC/m2/yr] -- the positive net primary
production that the model actually PARTITIONS in ``step_carbon_differland`` (``A_x = a_x *
max(NPP_day, 0)``); the negative-NPP deficit is drawn from the pools separately and does
not feed allocation.  It is recorded ONCE from a single default-parameter spin-up
(:func:`legoesm.land.carbon.global_init.precompute_fast_analytic_inputs`, stored on
:class:`~legoesm.land.carbon.fast_analytic.FastAnalyticInputs`) and held FIXED while the
trained leaves vary, exactly as the fast SOC forward freezes ``lit_to_som_annual`` and the
SIF forward freezes its coupled-Farquhar leaf state.  No GPP / autotrophic respiration is
re-derived here -- the model's own NPP is reused wholesale.

Because NPP is UPSTREAM of allocation, freezing it is EXACT for the DIRECT
allocation/residence/LCMA -> pool effect the calibration gradient follows.  It omits a
SECOND-ORDER feedback: the pools set ``LAI = C_fol/LCMA`` (hence GPP via fAPAR) and the
maintenance respiration (hence NPP), so a trained leaf that changes the pools would, in the
fully-coupled model, slightly change NPP too.  This is the SAME partial-gradient
approximation that excludes ``Q10_het_exp`` from the fast SOM-calibration set; the trainer's
fidelity gate (``_live_pool_match``) confirms the closed form reproduces the spin-up
equilibrium at the defaults, so the calibration follows the correct leading-order gradient.
A fully-coupled NPP feedback would require the slow grad-through-spin-up forward (out of
scope -- this forward is single-step by design).

Fidelity.  This is an ANNUAL-MEAN equilibrium approximation of the stepped live pools, NOT
an exact closed form of their periodic fixed point.  It is EXACT for the non-seasonal wood
and (near-non-seasonal) root pools; the ~annual foliage pool carries a small within-year
COVARIANCE residual (the deciduous leaf-fall pulse sheds when the canopy is full, so the
realised annual leaf turnover exceeds ``1/leaf_lifespan`` slightly) -- the SAME class of
residual the fast SOC forward documents for its active pool, NOT a sign/unit error.  The
closed form is validated against the default-parameter spin-up equilibrium live pools in
:func:`legoesm.land.carbon.global_init.precompute_fast_analytic_inputs`'s recorded
reference (the ``live_ref_*`` fields) and the trainer's fidelity gate.

All returned quantities are ``>= 0`` and every function is pure JAX, differentiable in the
``CarbonConfig`` allocation (``f_fol`` / ``f_lab`` / ``f_root``), residence
(``leaf_lifespan`` / ``tor_root`` / ``tor_wood``) and ``LCMA`` leaves.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.land.carbon.carbon_cycle import compute_phenology, sequential_allocation

# --- calendar / unit conversions (numerics, not tunable) ----------------------
# Year length [days] for the annual NPP integration + the per-day -> per-year
# residence conversion.  MUST match the precompute's stationary-year integration
# (global_init._YEAR_DAYS = 365.0, over which npp_pos_annual is summed) so
# C_root = A_root_annual / (tor_root * _DAYS_PER_YEAR) is self-consistent.
_DAYS_PER_YEAR = 365.0
# Number of daily samples for the phenology leaf-turnover year-integral (the leaf-fall
# Gaussian varies slowly over ~a leaf-fall period, so daily resolution captures it).
_N_PHENOLOGY_DAYS = 365
_G_PER_KG = 1000.0            # gC/m2 -> kgC/m2 (exact conversion)
# Divide-safety floor on an annual turnover k [1/yr]; the sanctioned training producer maps
# tor_root/tor_wood through a sigmoid onto a strictly-positive bound so k > 0 always -- this
# floor only guards a hand-crafted degenerate config, never the calibration path.
_K_FLOOR_PER_YR = 1e-12


__physics_contract__ = {
    "summary": (
        "Closed-form per-archetype LIVE-pool (foliage/root/wood) equilibrium for the "
        "Stage-B carbon calibration's biomass + LAI observation streams.  Each live pool "
        "is a linear steady state C_pool = a_pool * tau_pool * NPP (allocation fraction x "
        "residence time x annual allocatable NPP); LAI = C_fol / LCMA and biomass = "
        "(C_fol + C_root + C_wood)/1000 [kgC/m2].  Reuses the model's own DALEC sequential "
        "allocation (carbon_cycle.sequential_allocation) and residence leaves "
        "(leaf_lifespan, tor_root, tor_wood); NPP is the model's own annual allocatable "
        "NPP, recorded once at default physiology (param-independent).  Single-step, no "
        "spin-up scan; does not re-derive photosynthesis, respiration, or allocation."
    ),
    "inputs": {
        "npp_annual": "gC/m^2/yr",      # model's own annual allocatable NPP (>= 0)
        "is_woody": "1",                # per-archetype woody flag {0, 1}
        "is_evergreen": "1",            # per-archetype evergreen flag {0, 1}
        "config": "CarbonConfig (allocation/residence/LCMA leaves)",
    },
    "outputs": {"biomass": "kgC/m^2", "lai": "m^2/m^2"},
    "sign_convention": (
        "All fluxes are POSITIVE carbon inputs to their pool; all stocks C_pool >= 0 for "
        "NPP >= 0 and fractions/rates in [0,1]/(0,inf).  biomass >= 0 and LAI >= 0 "
        "(LAI = C_fol/LCMA, LCMA > 0).  Every turnover k > 0 so the linear steady state "
        "C = I/k is finite and positive."
    ),
    "conserves": [],  # diagnostic equilibrium of the live pools; not a step-wise budget
    "differentiable": True,
    "reference": (
        "DALEC990 / DifferLand v1.0 (Fang & Gentine, Columbia) live-pool allocation + "
        "first-order turnover; linear-pool steady state C = I/k (Olson 1963 residence "
        "time).  NPP allocation cascade: carbon_cycle.step_carbon_differland."
    ),
    "idealized_test": (
        "tests/land/unit/test_live_pool_forward.py: biomass/LAI >= 0 and finite; LAI "
        "scales 1/LCMA; C_wood scales 1/tor_wood; biomass rises with NPP; herbaceous "
        "archetype grows no wood; closed form matches the default-parameter spin-up "
        "equilibrium live pools to ~10-15%; finite non-zero grad wrt f_fol/leaf_lifespan/"
        "tor_wood/LCMA."
    ),
}


def annual_leaf_turnover(config):
    """Model's OWN annual leaf turnover ``(k_evergreen, k_deciduous)`` [1/yr] -- the
    year-integral of the DALEC990 leaf-fall fraction from
    :func:`legoesm.land.carbon.carbon_cycle.compute_phenology`, for BOTH leaf habits.

    ``compute_phenology`` returns a per-day leaf-fall fraction ``lff`` [1/day]; the realised
    annual turnover ``k = sum_doy lff(doy)`` [1/yr] is the fraction of the foliage pool shed
    per year (``leaf_litter_annual = k * <C_fol>`` for a slowly-varying canopy).  Evaluated
    NH-phased (``lat = 0``, matching the archetype forcing) over one year at daily
    resolution.  Both habits are the model's OWN ``compute_phenology`` (selected by a
    ``config._replace(evergreen=...)`` on the STATIC flag), never re-derived here:

    * evergreen: ``lff = 1/(leaf_lifespan * days_per_year)`` constant -> ``k ~ 1/leaf_lifespan``;
    * deciduous: the Gaussian leaf-fall pulse -> ``k`` = its year-integral, SUBSTANTIALLY
      larger than ``1/leaf_lifespan`` (the canopy sheds over a compact leaf-fall window).

    Differentiable in the phenology leaves (``leaf_lifespan`` / ``leaf_fall_period`` /
    ``Fday``); the caller selects the habit per archetype from the recorded ``is_evergreen``
    flag.  Returns two scalars (the config is shared across archetypes)."""
    doys = jnp.arange(_N_PHENOLOGY_DAYS, dtype=jnp.asarray(config.leaf_lifespan).dtype) + 0.5
    lat = jnp.zeros(_N_PHENOLOGY_DAYS, dtype=doys.dtype)      # NH-phased (lat = 0)
    _lrf_e, lff_ever = compute_phenology(doys, lat, config._replace(evergreen=True))
    _lrf_d, lff_decid = compute_phenology(doys, lat, config._replace(evergreen=False))
    return jnp.sum(lff_ever), jnp.sum(lff_decid)


def compute_live_pools(npp_annual, is_woody, is_evergreen, config):
    """Closed-form ANNUAL-MEAN live pools ``(C_fol, C_root, C_wood)`` [gC/m2].

    Steady state of each linear live pool ``dC/dt = I - k C`` -> ``C = I / k``:

    * foliage input ``I_fol = A_fol`` -- the DIRECT foliage allocation only (the labile
      allocation ``A_lab`` funds a separate buffer, excluded from standing biomass; see the
      module docstring); ``k_leaf`` is the model's own annual phenology turnover
      (:func:`annual_leaf_turnover`) selected per archetype by ``is_evergreen``.
    * ``A_root`` / ``A_wood`` -- the woody/herbaceous-routed structural allocation
      (:func:`legoesm.land.carbon.carbon_cycle.sequential_allocation` + the same routing as
      ``step_carbon_differland``); ``k_root = tor_root * days_per_year`` and
      ``k_wood = tor_wood * days_per_year`` [1/yr] (the first-order per-day turnover the
      prognostic step drains the pools with).

    Parameters
    ----------
    npp_annual : array (n_arch,)
        Model's OWN annual allocatable NPP [gC/m2/yr] (``>= 0``), recorded param-
        independently by the precompute.
    is_woody : array (n_arch,)
        Per-archetype woody flag (``1.0`` woody / ``0.0`` herbaceous); a herbaceous
        archetype invests the structural remainder belowground (no wood pool), matching
        ``step_carbon_differland``'s Python-branch routing but as a per-archetype arithmetic
        gate (the flag is an array here, not a static scalar).
    is_evergreen : array (n_arch,)
        Per-archetype evergreen flag (``1.0`` evergreen / ``0.0`` deciduous), selecting the
        leaf-turnover branch (continuous ``1/leaf_lifespan`` vs the Gaussian leaf-fall
        integral) -- the static per-group ``CarbonConfig.evergreen`` recorded at precompute.
    config : CarbonConfig
        Carbon config carrying the (possibly TRACED) allocation / residence / phenology leaves.

    Returns
    -------
    (C_fol, C_root, C_wood) : each array (n_arch,) [gC/m2]
        Annual-mean equilibrium live-pool carbon, differentiable in the config leaves.
    """
    npp = jnp.asarray(npp_annual)
    woody = jnp.asarray(is_woody)
    evergreen = jnp.asarray(is_evergreen)
    # Model's own DALEC sequential partition (absolute annual fluxes).  A_wood_raw is the
    # structural remainder BEFORE routing; it is >= 0 for npp >= 0 and fractions in [0,1].
    a_fol, _a_lab, a_root_base, a_wood_raw = sequential_allocation(
        npp, config.f_fol, config.f_lab, config.f_root)
    # Woody/herbaceous routing (per-archetype flag): woody keeps the remainder as wood;
    # herbaceous invests it belowground (roots) -- same as step_carbon_differland.
    A_wood = woody * a_wood_raw
    A_root = a_root_base + (1.0 - woody) * a_wood_raw

    # Leaf turnover [1/yr]: the model's OWN phenology integral, selected per archetype.
    k_ever, k_decid = annual_leaf_turnover(config)
    k_leaf = jnp.maximum(jnp.where(evergreen > 0.5, k_ever, k_decid), _K_FLOOR_PER_YR)
    # Structural turnover [1/yr]: first-order per-day rate x days_per_year.
    k_root = jnp.maximum(config.tor_root * _DAYS_PER_YEAR, _K_FLOOR_PER_YR)
    k_wood = jnp.maximum(config.tor_wood * _DAYS_PER_YEAR, _K_FLOOR_PER_YR)

    C_fol = a_fol / k_leaf          # foliage input = A_fol (direct allocation only)
    C_root = A_root / k_root
    C_wood = A_wood / k_wood
    return C_fol, C_root, C_wood


def compute_biomass_lai(npp_annual, is_woody, is_evergreen, config):
    """Per-archetype ``(biomass [kgC/m2], lai [m2/m2])`` from the closed-form live pools.

    ``biomass = (C_fol + C_root + C_wood)/1000`` (total live-pool carbon; excludes the
    separate labile buffer) and ``lai = C_fol / LCMA``.  Differentiable in the
    ``CarbonConfig`` allocation / residence / phenology / ``LCMA`` leaves via
    :func:`compute_live_pools`.
    """
    C_fol, C_root, C_wood = compute_live_pools(npp_annual, is_woody, is_evergreen, config)
    lai = C_fol / config.LCMA
    biomass = (C_fol + C_root + C_wood) / _G_PER_KG
    return biomass, lai


def build_live_pool_forward(precomputed):
    """Precompute-bound closure ``fn(carbon_config) -> (biomass, lai)`` [kgC/m2, m2/m2].

    The model's annual allocatable NPP (``precomputed.npp_pos_annual``) and the per-archetype
    woody / evergreen flags (``precomputed.is_woody`` / ``precomputed.is_evergreen``) are
    recorded ONCE, param-independent of every trained leaf, by
    :func:`legoesm.land.carbon.global_init.precompute_fast_analytic_inputs`; this returns a
    closure that applies only the differentiable :func:`compute_biomass_lai` algebra, so the
    allocation/residence/phenology/LCMA gradient is a pure single-step graph (no spin-up
    scan) -- the trainer's ``biomass`` / ``lai`` loss terms call this.  Mirrors
    :func:`legoesm.land.carbon.sif_forward.build_sif_forward` (a static precomputed state +
    a differentiable kernel).

    Parameters
    ----------
    precomputed : FastAnalyticInputs
        Carries ``npp_pos_annual`` ``(n_arch,)`` [gC/m2/yr] and the ``is_woody`` /
        ``is_evergreen`` ``(n_arch,)`` flags.

    Returns
    -------
    fn : callable
        ``fn(config: CarbonConfig) -> (biomass (n_arch,), lai (n_arch,))``.
    """
    npp_annual = precomputed.npp_pos_annual
    is_woody = precomputed.is_woody
    is_evergreen = precomputed.is_evergreen

    def fn(config):
        return compute_biomass_lai(npp_annual, is_woody, is_evergreen, config)

    return fn
