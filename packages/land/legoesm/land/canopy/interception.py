"""Canonical canopy-water interception — shared by the two-leaf and CLM-ML canopies.

Precipitation falling on a vegetated surface splits into three fates:

1. **Throughfall** — the part that misses the canopy (direct) or drips off once
   the leaves are saturated (canopy drip).  It reaches the soil.
2. **Interception storage** — water held on leaf/stem surfaces, a prognostic
   store ``W_canopy`` [kg m-2] carried between steps.
3. **Wet-leaf evaporation** — intercepted water that evaporates directly off the
   wet leaf surface, at the potential rate scaled by the wetted fraction.  This
   ET component is separate from transpiration (which draws root-zone water
   through stomata); transpiration acts on the *dry* leaf fraction.

The formulation is the CLM-ML / CLM5 one (``MLCanopyWaterMod``), aggregated to a
single big-leaf layer so the two-leaf canopy and the multi-layer canopy share
the same math.  The plant-area used is ``LAI + SAI``; a two-leaf canopy with no
stem-area term passes ``sai = 0``.

SCOPE — water cycle, not a wet-leaf energy balance.  This is a WATER-CYCLE
interception: it advances the store, routes throughfall to the soil, and lets
the wet leaf evaporate from the store.  On the two-leaf coupling the wet-leaf
evaporation is a *substitution* — stored water is drawn instead of root-zone
water, up to the transpiration the canopy is already demanding, leaving the
reported surface latent flux unchanged.  It does NOT re-solve the leaf energy
balance with a wetted fraction (which would recompute leaf temperature, sensible
heat, and stomatal conductance for the wet surface); that fuller wet/dry-leaf
energy partition is a documented follow-up.  The multi-layer (CLM-ML) canopy
keeps its own internal per-layer interception with the same constants.

Sign convention: all fluxes are rates [kg m-2 s-1] positive in the direction of
the name (``throughfall`` down onto soil, ``wet_evap`` up off the leaf; a
negative ``wet_evap`` is dew deposition adding to the store).  Water conserved:

    precip = throughfall + d(W_canopy)/dt + wet_evap

verified to machine precision by ``water_balance_residual`` and its test.

References
----------
Bonan et al. (2018) CLM-ML; Lawrence et al. (2019) CLM5 tech note, canopy
hydrology.  Constants (``dewmx``, ``fwet_exponent``, ``interception_fraction``,
``maximum_leaf_wetted_fraction``) match ``clm-ml-jax`` ``MLclm_varcon``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.smooth import min_rel, positive_part_c1, positive_part_rel, smoothstep_cubic

__physics_contract__ = {
    "units": {
        "W_canopy": "kg m-2",
        "precip": "kg m-2 s-1",
        "throughfall": "kg m-2 s-1",
        "wet_evap": "kg m-2 s-1",
        "fwet": "1",
    },
    "signs": "throughfall positive down onto soil; wet_evap >= 0 (evaporation "
             "up); dew (downward flux) excluded from the store; W_canopy in "
             "[0, h2ocanmx]",
    "conserves": "water: precip = throughfall + dW_canopy/dt + wet_evap",
    "differentiable": True,
    "reference": "Bonan et al. 2018 CLM-ML MLCanopyWaterMod; CLM5 tech note",
    "idealized_test": "tests/land/unit/test_interception.py "
                      "(closure to machine precision; drip caps storage; "
                      "wetted fraction bounded)",
}

# Floors that keep gradients finite where a physical quantity may hit zero
# (empty canopy, dry store).  Not tunable — pure numerics.
_EPS_PAI = 1.0e-12       # m2 m-2, guards divide by plant area
# Floor on the wetted-fraction base ``W/h2ocanmx``.  A tiny floor (1e-30) makes
# ``base**(p-1)`` ~1e10 — a finite but optimisation-destroying gradient at the
# dry limit.  1e-6 keeps ``d(base**p)`` <~100 while ``fwet(W=0) = 1e-6**0.667
# ~1e-4`` stays negligibly small (dry canopy is effectively dry).
_EPS_POW = 1.0e-6

__param_spec__ = {
    "InterceptionConfig": {
        "scheme_key": "land.canopy.interception",
        "excluded": {
            # CLM5 interception efficiency defaults to 1.0 — the physical
            # fraction CEILING. A sigmoid tunable needs its default STRICTLY
            # interior to (lo, hi<=1) to round-trip through the collector's
            # logit inverse, which a default of exactly 1.0 cannot satisfy (and
            # hi>1 is an invalid fraction domain). So it is fixed (tier 0) at the
            # CLM5 reference rather than made a downward-only sigmoid knob.
            "interception_fraction": "CLM5 interception efficiency = 1.0, the "
                                     "physical fraction ceiling; not a well-posed "
                                     "sigmoid tunable (default on the bound).",
        },
        "params": {
            # CLM5 canopy hydrology closures — per-site tunable (tier 2), bounds
            # interior to the physical range so they round-trip through the
            # trainable collector's transform inverse.
            "dewmx": {"units": "kg m-2", "bounds": (0.05, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "canopy_hydrology", "reference": "CLM5 max leaf water per unit plant area (0.1)", "shape": None},
            "fwet_exponent": {"units": "1", "bounds": (0.3, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "canopy_hydrology", "reference": "CLM5 wetted-fraction exponent (0.667)", "shape": None},
            "maximum_leaf_wetted_fraction": {"units": "1", "bounds": (0.01, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "canopy_hydrology", "reference": "CLM5 cap on the wet leaf fraction (0.05)", "shape": None},
        },
    },
    "CanopySnowConfig": {
        "scheme_key": "land.canopy.canopy_snow",
        "excluded": {
            "temperature_width_K": "RULE 5 smoothing width of the CLM5 "
                                   "temperature switches (numerics, not physics).",
            "relative_width": "RULE 5 smoothing width of the CLM5 caps "
                              "(numerics, not physics).",
            "fcansno_ramp": "RULE 5 C1 ramp of fcansno's 0.15 power at an "
                            "empty store (numerics, not physics).",
        },
        "params": {},
    },
}


class InterceptionConfig(NamedTuple):
    """Canopy-water interception parameters (CLM-ML / CLM5 defaults).

    Shared by both canopy schemes so a tuned value applies consistently.  Three
    are per-site tunable closures (``__param_spec__`` tier 2); ``interception_
    fraction`` is fixed at the CLM5 reference 1.0 (the physical fraction ceiling,
    not a well-posed sigmoid knob — see ``__param_spec__`` excluded).
    """
    # --- canopy hydrology (CLM5 / Bonan MLCanopyWaterMod) ---
    dewmx: float = 0.1                          # max leaf water [kg m-2 / plant-area]
    interception_fraction: float = 1.0          # fraction of precip intercepted at high LAI
    fwet_exponent: float = 0.667                # wetted-fraction power
    maximum_leaf_wetted_fraction: float = 0.05  # cap on wet leaf fraction


def max_canopy_water(pai: jnp.ndarray, cfg: InterceptionConfig) -> jnp.ndarray:
    """Maximum intercepted-water storage ``h2ocanmx = dewmx * PAI`` [kg m-2]."""
    return cfg.dewmx * jnp.maximum(pai, 0.0)


def wetted_fraction(W_canopy: jnp.ndarray, pai: jnp.ndarray,
                    cfg: InterceptionConfig) -> jnp.ndarray:
    """Wet fraction of the canopy ``fwet in [0, max_leaf_wetted]``.

    ``fwet = min((W / h2ocanmx)**fwet_exponent, maximum_leaf_wetted_fraction)``.
    Transpiration later acts on ``1 - fwet``; wet-leaf evaporation on ``fwet``.
    Empty canopy (``pai == 0``) is fully dry.
    """
    h2ocanmx = max_canopy_water(pai, cfg)
    has_pai = pai > _EPS_PAI
    # 0**p has an infinite gradient at 0; floor the base so grad stays finite.
    base = jnp.where(has_pai,
                     jnp.maximum(W_canopy / jnp.maximum(h2ocanmx, _EPS_POW),
                                 _EPS_POW),
                     _EPS_POW)
    fwet = jnp.minimum(base ** cfg.fwet_exponent,
                       cfg.maximum_leaf_wetted_fraction)
    # A DRY store is exactly dry (fwet == 0), so an interception-enabled but
    # empty canopy is bit-identical to interception-off — the base floor is only
    # to keep the gradient finite for W > 0, not to nucleate a spurious wet
    # fraction at W = 0.
    return jnp.where(has_pai & (W_canopy > 0.0), fwet, 0.0)


def intercept_rain(W_canopy: jnp.ndarray, precip: jnp.ndarray,
                   pai: jnp.ndarray, dt: float, cfg: InterceptionConfig):
    """Phase 1: intercept rain into the store; return ``(W_int, throughfall)``.

    ``fpi = clip(interception_fraction * tanh(PAI), 0, 1)`` of the precip lands
    on the leaves; the rest is direct throughfall.  The store is capped at
    ``h2ocanmx`` and the excess drips.  No evaporation here — call
    :func:`evaporate_wet_leaf` after the caller knows the transpiration demand,
    so the store evaporation can be capped by it (closure).  Split so the
    THROUGHFALL is known before the soil-evaporation water-availability limiter
    (which must see throughfall, not raw precip).
    """
    pai = jnp.maximum(pai, 0.0)
    h2ocanmx = max_canopy_water(pai, cfg)
    fpi = jnp.clip(cfg.interception_fraction * jnp.tanh(pai), 0.0, 1.0)
    W_int = W_canopy + precip * fpi * dt
    drip = jnp.maximum(W_int - h2ocanmx, 0.0) / dt
    W_int = jnp.minimum(W_int, h2ocanmx)
    throughfall = precip * (1.0 - fpi) + drip
    return W_int, throughfall


def evaporate_wet_leaf(W_int: jnp.ndarray, le_pot_wet: jnp.ndarray,
                       pai: jnp.ndarray, dt: float, cfg: InterceptionConfig,
                       L_v: float, transp_avail: jnp.ndarray | None = None):
    """Phase 2: evaporate the wet leaf from the store; return ``(W_new, wet_evap, fwet)``.

    ``wet_evap = min(fwet * le_pot_wet / L_v, W_int/dt, transp_avail)``, positive
    only (dew is excluded — it stays in the caller's surface-dew path, else the
    atmospheric water is double-counted).  ``transp_avail`` (the root-zone
    transpiration the caller will reduce by ``wet_evap``) caps the store
    evaporation so the substitution never removes more surface water than the
    atmosphere is taking (closure).
    """
    pai = jnp.maximum(pai, 0.0)
    fwet = wetted_fraction(W_int, pai, cfg)
    evap_demand = jnp.maximum(fwet * le_pot_wet / L_v, 0.0)
    max_evap = W_int / dt
    if transp_avail is not None:
        max_evap = jnp.minimum(max_evap, jnp.maximum(transp_avail, 0.0))
    wet_evap = jnp.minimum(evap_demand, max_evap)
    W_new = jnp.maximum(W_int - wet_evap * dt, 0.0)
    return W_new, wet_evap, fwet


def update_canopy_water(W_canopy: jnp.ndarray, precip: jnp.ndarray,
                        le_pot_wet: jnp.ndarray, pai: jnp.ndarray, dt: float,
                        cfg: InterceptionConfig, L_v: float,
                        transp_avail: jnp.ndarray | None = None):
    """Advance the canopy-water store one step; return the water partition.

    Ordering mirrors CLM-ML: intercept precip (cap storage, spill to drip),
    then evaporate the wet leaf at the potential rate scaled by ``fwet``.

    **Substitution model (see module docstring / caller).**  The wet-leaf
    evaporation is a re-sourcing of canopy latent heat from the store rather
    than the root zone; it is therefore capped at ``transp_avail`` (the
    root-zone transpiration mass the caller is about to draw) so that shifting
    it onto the store never removes MORE water from the surface than the
    atmosphere is taking.  Without a cap, closure would fail when the wet demand
    exceeds the available transpiration (codex).  This is a water-cycle
    interception; it does NOT re-solve the leaf energy balance for a wet leaf.

    Dew is NOT deposited into the store here: a downward (negative) potential
    flux is clamped to zero so canopy dew stays entirely in the caller's
    existing surface-dew path (routing it here too would double-count the
    atmospheric water — codex).  The store is therefore bounded in
    ``[0, h2ocanmx]`` every step.

    Parameters
    ----------
    W_canopy : (ncol,) intercepted-water store at the start of the step [kg m-2].
    precip   : (ncol,) rate of liquid precip onto the canopy [kg m-2 s-1] (>= 0).
    le_pot_wet : (ncol,) latent-heat flux the wet leaf would sustain [W m-2],
        positive up.  The wet-leaf evaporation demand is ``fwet * le_pot_wet``;
        a downward (negative) value contributes no store evaporation.
    pai      : (ncol,) plant area index ``LAI + SAI`` [m2 m-2].
    dt       : timestep [s].
    cfg      : :class:`InterceptionConfig`.
    L_v      : latent heat of vaporization [J kg-1] (``constants.L_v``).
    transp_avail : (ncol,) upper bound on the store evaporation [kg m-2 s-1] —
        the root-zone transpiration the caller reduces by ``wet_evap``.  ``None``
        imposes no cap (stand-alone use).

    Returns
    -------
    W_new : (ncol,) updated store [kg m-2], in ``[0, h2ocanmx]``.
    throughfall : (ncol,) water reaching the soil [kg m-2 s-1] = direct
        throughfall + canopy drip.
    wet_evap : (ncol,) wet-leaf evaporation [kg m-2 s-1], >= 0, capped by the
        store AND by ``transp_avail``.
    fwet : (ncol,) wetted fraction used [-].
    """
    W_int, throughfall = intercept_rain(W_canopy, precip, pai, dt, cfg)
    W_new, wet_evap, fwet = evaporate_wet_leaf(
        W_int, le_pot_wet, pai, dt, cfg, L_v, transp_avail)
    return W_new, throughfall, wet_evap, fwet


def water_balance_residual(W_old: jnp.ndarray, W_new: jnp.ndarray,
                           precip: jnp.ndarray, throughfall: jnp.ndarray,
                           wet_evap: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Closure residual [kg m-2 s-1]: ``precip - throughfall - dW/dt - wet_evap``.

    Zero to machine precision (no store clamp fires now that dew is excluded and
    the store is bounded in ``[0, h2ocanmx]`` each step).
    """
    return precip - throughfall - (W_new - W_old) / dt - wet_evap


# ---------------------------------------------------------------------------
# Canopy snow (CLM5.0 snowveg_flag = 'ON_RAD', the CLM5.0 default)
# ---------------------------------------------------------------------------
# Fixed published coefficients of CLM5.0 CanopyHydrologyMod.F90 (release-
# clm5.0.34), not tunables.
_SNOCANMX_FACTOR = 60.0        # snocanmx = 60 dewmx (elai + esai)       (:329)
_FPISNOW_K = 0.5               # fpisnow = 1 - exp(-0.5 (elai + esai))     (:332)
_UNLOAD_T_BELOW_FREEZE_K = 3.0  # unloading starts at 270.15 K = tfrz - 3 K (:426)
_UNLOAD_T_SCALE = 1.87e5       # [K s] temperature unloading scale          (:426)
_UNLOAD_WIND_COEF = 0.5        # wind unloading coefficient                 (:427)
_UNLOAD_WIND_SCALE = 1.56e5    # [m] wind unloading length                  (:427)
_FCANSNO_EXPONENT = 0.15       # fcansno = (snocan / snocanmx)**0.15        (:726)


# CLM5.0 CanopyHydrologyMod canopy-water constants (dewmx 0.1, fwet exponent
# 0.666666666666 at :719, maximum_leaf_wetted_fraction 0.05): what the canopy
# snow store reads when liquid interception is off.  The liquid store's own
# default exponent is 0.667 (clm-ml-jax MLclm_varcon), used when it is on.
CLM5_CANOPY_WATER = InterceptionConfig(fwet_exponent=0.666666666666)


class CanopySnowConfig(NamedTuple):
    """CLM5 intercepted canopy snow: RULE 5 widths of its state switches.

    ``temperature_width_K`` smooths the three temperature switches (snow
    fall-off only at or below freezing, the 270.15 K unloading ramp, and the
    leaf-phase split of the wet-leaf evaporation); ``relative_width`` the caps
    (fall-off of the excess over ``snocanmx``, unloading at most the store,
    ``fcansno <= 1``, store draws); ``fcansno_ramp`` the store fraction
    ``snocan/snocanmx`` below which ``fcansno``'s 0.15 power (infinite slope at
    an empty store) is replaced by its C1 quadratic continuation from 0.
    0 = CLM5's hard form.  Legal ranges: :func:`validate_canopy_snow_config`.
    """
    temperature_width_K: float = 0.5
    relative_width: float = 0.05
    fcansno_ramp: float = 1.0e-3


def validate_canopy_snow_config(cfg: CanopySnowConfig) -> None:
    """One legal range per width: each is 0 (CLM5's hard switch) or a RULE 5
    width; ``relative_width <= 1`` (above it a smooth min of non-negative
    operands can go negative, e.g. unloading turns into loading) and
    ``fcansno_ramp <= 1``."""
    from legoesm.core.smooth import validate_smoothing_width
    for name in cfg._fields:
        validate_smoothing_width(f"canopy_snow.{name}", getattr(cfg, name))
    for name in ("relative_width", "fcansno_ramp"):
        if getattr(cfg, name) > 1.0:
            raise ValueError(f"canopy_snow.{name} must be <= 1 (got "
                             f"{getattr(cfg, name)!r}): a larger width breaks "
                             "the non-negativity of the store transfers.")


def canopy_cold_weight(T, T_switch, width):
    """Weight of ``T <= T_switch``: exactly 1 at and below, exactly 0 from
    ``T_switch + width`` (C1); ``width = 0`` is the hard test."""
    if width == 0:
        return (T <= T_switch).astype(jnp.result_type(T, float))
    return 1.0 - smoothstep_cubic((T - T_switch) / width)


def canopy_snow_capacity(vai, ic: InterceptionConfig):
    """``snocanmx = 60 dewmx (elai + esai)`` [kg m-2] (CLM5 :329)."""
    return _SNOCANMX_FACTOR * ic.dewmx * jnp.maximum(vai, 0.0)


def canopy_snow_step(snocan, snow_rate, T_air, wind, vai, dt,
                     ic: InterceptionConfig, cfg: CanopySnowConfig):
    """Intercept snowfall, shed the excess below freezing, unload (CLM5 ON_RAD).

    CLM5.0 CanopyHydrologyMod.F90 :328-347 (interception ``fpisnow``), :361-378
    (below-freezing fall-off of the excess over ``snocanmx``) and :421-437
    (temperature and wind unloading, limited to the store), in that order.
    ``vai`` is the exposed plant area ``elai + esai``.  Returns ``(snocan_new,
    ground_snow_rate)``; every kilogram that leaves the store reaches the
    ground snow, so ``snow_rate = ground_snow_rate + (snocan_new - snocan)/dt``
    exactly (sublimation is drawn later, by the caller).

    KNOWN DIFFERENCES vs CLM5.0 (recorded, not hidden):
    * smooth defaults (RULE 5): the three switches carry ``cfg`` widths
      (0 = CLM5); per-case gaps at the default are pinned by
      tests/land/unit/oracle_fixtures/clm5_canopy_snow_gaps.json;
    * CLM5 skips the whole block where ``frac_veg_nosno = 0`` (canopy buried),
      freezing ``snocan``; here a buried canopy (``vai = 0``) intercepts
      nothing and its store still unloads;
    * CLM5 sheds the excess over ``snocanmx`` only in a step with
      precipitation; here it is shed in any step below freezing (same mass,
      reaches the ground snow at most one dry spell earlier);
    * the land step (not this function): CLM5's one ``t_veg`` picks the
      evaporation phase; the two-leaf canopy uses its sunlit-fraction-weighted
      leaf temperature; canopy frost/dew keeps the land's existing routing
      (CLM5 deposits it into ``snocan`` below freezing); canopy sublimation
      pays L_s (CLM5 charges hvap, dropping L_f -- RULE 0);
    * an evaporation demand above both stores (CLM5 caps it upstream, then
      drives ``liqcan`` negative on an over-draw) empties both stores here.
    """
    vai = jnp.maximum(vai, 0.0)
    fpisnow = 1.0 - jnp.exp(-_FPISNOW_K * vai)
    through = snow_rate * (1.0 - fpisnow)
    s1 = snocan + dt * snow_rate * fpisnow
    cap = canopy_snow_capacity(vai, ic)
    cold = canopy_cold_weight(T_air, constants.T_freeze, cfg.temperature_width_K)
    falloff = cold * positive_part_rel(s1 - cap, cap, cfg.relative_width)
    s2 = s1 - falloff
    w = cfg.temperature_width_K
    dT = T_air - (constants.T_freeze - _UNLOAD_T_BELOW_FREEZE_K)
    warm = jnp.maximum(dT, 0.0) if w == 0 else positive_part_c1(dT, 1.0 / w)
    k = warm / _UNLOAD_T_SCALE + _UNLOAD_WIND_COEF * wind / _UNLOAD_WIND_SCALE
    frac = min_rel(k * dt, 1.0, cfg.relative_width)
    unload = frac * s2
    s3 = s2 - unload
    return s3, through + (falloff + unload) / dt


def canopy_snow_fraction(snocan, vai, ic: InterceptionConfig,
                         cfg: CanopySnowConfig):
    """Snow-covered canopy fraction ``fcansno = min(1, (snocan/snocanmx)**0.15)``
    (CLM5 :723-729); 0 for an empty store or no exposed plant area.  At
    ``cfg.fcansno_ramp = 0`` (CLM5's form) the power's infinite slope at an
    empty store is guarded by a double ``where`` (zero gradient there); the
    default ramp makes it C1 with a finite positive slope from 0."""
    cap = canopy_snow_capacity(vai, ic)
    p, x0 = _FCANSNO_EXPONENT, cfg.fcansno_ramp
    if x0 > 0.0:
        # C1 quadratic g(x) = a x + b x^2 on [0, x0] matching x**p and its slope
        # at x0 (a = (2 - p) x0**(p-1), b = (p - 1) x0**(p-2)): monotone, 0 at
        # an empty store with the finite slope a (so the empty store is not a
        # gradient dead end).  The power only sees x >= x0 (no infinite slope).
        on = cap > 0.0
        base = snocan / jnp.where(on, cap, 1.0)     # snocan >= 0 by construction
        xs = jnp.minimum(base, x0)
        quad = (2.0 - p) * x0 ** (p - 1.0) * xs + (p - 1.0) * x0 ** (p - 2.0) * xs * xs
        power = jnp.where(base < x0, quad, jnp.where(base < x0, x0, base) ** p)
    else:
        on = (snocan > 0.0) & (cap > 0.0)
        base = jnp.where(on, snocan / jnp.where(on, cap, 1.0), 1.0)
        power = base ** p
    f = min_rel(power, 1.0, cfg.relative_width)
    return jnp.where(on, f, 0.0)


def split_wet_evap(demand_cold, demand_warm, snow_avail, liq_avail,
                   relative_width):
    """Draw the wet-leaf evaporation from the two canopy stores, CLM5 order
    (CanopyFluxesMod.F90 :1257-1272): leaves at or below freezing take snow
    first then liquid, warmer leaves liquid first then snow.  All arguments
    are masses per step [kg m-2] (>= 0).  Returns ``(from_snow, from_liq)``;
    their sum is the demand capped by the two stores together.
    """
    rw = relative_width
    s_cold = min_rel(demand_cold, snow_avail, rw)
    l_cold = min_rel(demand_cold - s_cold, liq_avail, rw)
    l_warm = min_rel(demand_warm, liq_avail - l_cold, rw)
    s_warm = min_rel(demand_warm - l_warm, snow_avail - s_cold, rw)
    return s_cold + s_warm, l_cold + l_warm
