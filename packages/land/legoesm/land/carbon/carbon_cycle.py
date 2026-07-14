"""Land carbon cycle: DifferLand prognostic model and seasonal cycle.

DifferLand (Fang & Gentine, Columbia) — DALEC990-based:
    8 carbon pools (labile, foliage, root, wood, litter, and a 3-pool CENTURY
    SOM cascade: active/slow/passive) driven by a light-use-efficiency GPP with
    temperature/moisture responses.  Phenology follows DALEC990 Gaussian
    seasonal forcing.

Seasonal cycle:
    Prescribed sinusoidal NEE with latitude-dependent amplitude and phase.
    No prognostic pools — useful for testing carbon transport.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.land.carbon.config import (
    CarbonConfig,
    CarbonDiagnostics,
    CarbonState,
    is_concrete,
    validate_som_transfer_fractions,
)

# Fixed calendar / radiation constants (not tunable).
_PAR_FRACTION_OF_SW = 0.48     # photosynthetically-active fraction of shortwave
_DAYS_PER_YEAR = 365.25        # Julian year length [days]
_HALF_YEAR_OFFSET_DAYS = 182.5  # half-year phenology phase offset [days]
_SEASONAL_YEAR_DAYS = 365.0    # year length for the NEE seasonal phase [days]
# Number of within-year samples for the annual-frozen-fraction integral
# (:func:`annual_frozen_fraction`); daily resolution is ample for a smooth
# seasonal cosine.  A NUMERICS sample count (never trained), fixed so the
# coupled spin-up and the closed-form surrogate integrate an IDENTICAL index.
_FROST_INDEX_SAMPLES = 365

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
_GC_TO_KG_CO2 = (44.0 / 12.0) * 1e-3   # gC -> kgCO2
_SPD = 86400.0                            # seconds per day

# ---------------------------------------------------------------------------
# CENTURY-like SOM equilibrium partition (Parton et al. 1987; Koven et al. 2013)
# ---------------------------------------------------------------------------
# Steady-state stock FRACTIONS used to split ``C_som_init`` across the three SOM
# pools at initialisation: the active pool is small and fast, the passive
# (mineral-stabilised) pool holds the bulk.  The three sum to 1 so ``som_total``
# equals ``C_som_init`` exactly (the partition moves carbon between pools, never
# creates or destroys it).
_SOM_INIT_FRAC_ACTIVE = 0.03
_SOM_INIT_FRAC_SLOW = 0.32
_SOM_INIT_FRAC_PASSIVE = 0.65

# ---------------------------------------------------------------------------
# Phenology offset polynomial coefficients (DALEC990)
# ---------------------------------------------------------------------------
_OFFSET_COEFFS = (
    0.000023599784710,
    0.000332730053021,
    0.000901865258885,
    -0.005437736864888,
    -0.020836027517787,
    0.126972018064287,
    -0.188459767342504,
)


# ===================================================================
# GPP — Light-Use Efficiency model
# ===================================================================

def compute_gpp(
    sw_down: jnp.ndarray,
    T: jnp.ndarray,
    LAI: jnp.ndarray,
    co2_ppmv: jnp.ndarray,
    beta: jnp.ndarray,
    config: CarbonConfig,
) -> jnp.ndarray:
    """Gross primary production via light-use efficiency.

    Parameters
    ----------
    sw_down : Downward shortwave radiation [W/m2].
    T       : Surface temperature [K].
    LAI     : Leaf area index [m2/m2].
    co2_ppmv: Atmospheric CO2 [ppmv].
    beta    : Moisture availability [0-1].
    config  : CarbonConfig.

    Returns
    -------
    gpp : Instantaneous GPP [gC/m2/s].
    """
    # PAR = 48 % of SW, convert W/m2 -> MJ/m2/s
    PAR_MJ = _PAR_FRACTION_OF_SW * sw_down * 1e-6

    # Absorbed fraction (Beer's law)
    fAPAR = 1.0 - jnp.exp(-config.k_ext * LAI)
    APAR = fAPAR * PAR_MJ

    T_C = T - constants.T_freeze
    f_T = jnp.exp(-0.5 * ((T_C - config.T_opt_C) / config.T_width_C) ** 2)

    # CO2 fertilization — Michaelis-Menten.  Use ``jnp.full`` (single
    # ``Broadcast`` HLO op) instead of
    # ``broadcast_to(jnp.asarray(scalar, dtype), shape)`` which
    # additionally forces a ``ConvertElementType``.
    if hasattr(co2_ppmv, "shape"):
        co2 = jnp.broadcast_to(co2_ppmv.astype(T.dtype), T.shape)
    else:
        co2 = jnp.full(T.shape, co2_ppmv, dtype=T.dtype)
    f_CO2 = co2 / (co2 + config.K_CO2)

    # GPP = epsilon * APAR * f_T * f_CO2 * beta  [gC/m2/s].  The cold-deciduous
    # freeze-dormancy gate is applied in step_carbon_differland (on the FINAL
    # gpp, whether an override or this computed value) so the coupled Farquhar
    # gpp_override path is gated too -- see there.
    return jnp.maximum(config.epsilon * APAR * f_T * f_CO2 * beta, 0.0)


# ===================================================================
# Phenology (DALEC990 Gaussian seasonal forcing)
# ===================================================================

def _phenology_offset(lifespan: float, width: float) -> jnp.ndarray:
    """Empirical DALEC990 offset correction (polynomial in ln(L-1))."""
    x = jnp.log(jnp.maximum(lifespan - 1.0, 1e-6))
    # Horner evaluation of degree-6 polynomial
    val = jnp.asarray(_OFFSET_COEFFS[0])
    for c in _OFFSET_COEFFS[1:]:
        val = val * x + c
    return width * val


def compute_phenology(
    doy: jnp.ndarray,
    lat: jnp.ndarray,
    config: CarbonConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Labile-release and leaf-fall fractions [day^-1].

    Two leaf-habit modes, selected by the STATIC ``config.evergreen`` flag (a
    Python ``if`` on a static bool -- the repo's feature-gating convention, NOT
    a traced ``jnp.where`` that would evaluate both branches):

    * DECIDUOUS (``evergreen=False``, default): the DALEC990 Gaussian seasonal
      forcing -- labile release peaks near ``Bday``, leaf fall near ``Fday``,
      the canopy dropping toward ~0 LAI between.
    * EVERGREEN (``evergreen=True``): near-CONTINUOUS turnover -- a steady,
      season-independent leaf-fall fraction ``1 / (leaf_lifespan * yr)`` and a
      steady labile release ``1 / (lab_lifespan * yr)``, so the canopy sheds
      and replaces a small constant fraction each day and never defoliates.
      Correct for broadleaf/needleleaf evergreen PFTs.

    Only the ``(lrf, lff)`` RATES differ between modes; carbon conservation is
    identical -- ``step_carbon_differland`` routes ``lab_release`` C_lab->C_fol
    and ``leaf_litter`` C_fol->C_lit exactly regardless of the rate values, and
    ``_effective_rate`` clips any rate into [0, 1) so no pool is over-drained.

    Parameters
    ----------
    doy  : Day of year [0-365].
    lat  : Latitude [radians].
    config : CarbonConfig.

    Returns
    -------
    lrf : Labile release fraction [day^-1].
    lff : Leaf fall fraction [day^-1].
    """
    if config.evergreen:
        # Evergreen (continuous) phenology: shed + replace a small constant
        # fraction of the canopy every day instead of the deciduous Bday/Fday
        # pulse.  Leaf residence time == leaf_lifespan years, so the steady
        # per-day leaf-fall fraction is 1 / (leaf_lifespan * days_per_year); the
        # labile release is likewise steady (1 / (lab_lifespan * days_per_year))
        # to feed the continuous regrowth.  Both are CONSTANT in doy/lat, use
        # the SAME [day^-1] convention as the deciduous branch, and broadcast to
        # lat.shape to match its output shape.  (leaf_lifespan / lab_lifespan
        # are years and strictly positive by construction -- the deciduous
        # branch's jnp.log(lifespan) below likewise assumes positivity.)
        lrf_val = 1.0 / (config.lab_lifespan * _DAYS_PER_YEAR)
        lff_val = 1.0 / (config.leaf_lifespan * _DAYS_PER_YEAR)
        lrf = jnp.full(lat.shape, lrf_val, dtype=lat.dtype)
        lff = jnp.full(lat.shape, lff_val, dtype=lat.dtype)
        return lrf, lff

    # Deciduous (default): DALEC990 Gaussian seasonal forcing.  Byte-identical
    # to the pre-evergreen implementation (the evergreen early-return above
    # leaves every line below untouched).
    sf = _DAYS_PER_YEAR / jnp.pi
    inv_sqrt_pi = 2.0 / jnp.sqrt(jnp.pi)
    sqrt2_half = jnp.sqrt(2.0) / 2.0

    # Labile release parameters
    fl = 0.5 * (
        jnp.log(config.lab_lifespan)
        - jnp.log(jnp.maximum(config.lab_lifespan - 1.0, 1e-6))
    )
    wl = config.clab_release_period * sqrt2_half
    osl = _phenology_offset(config.lab_lifespan, wl)

    # Leaf fall parameters
    ff = 0.5 * (
        jnp.log(config.leaf_lifespan)
        - jnp.log(jnp.maximum(config.leaf_lifespan - 1.0, 1e-6))
    )
    wf = config.leaf_fall_period * sqrt2_half
    osf = _phenology_offset(config.leaf_lifespan, wf)

    # Hemisphere-aware effective doy (shift by half year for SH).
    # ``jnp.full`` is one ``Broadcast`` HLO op vs the
    # ``broadcast_to(jnp.asarray(...))`` form's
    # ``ConvertElementType + Broadcast`` pair.
    if hasattr(doy, "shape"):
        doy_arr = jnp.broadcast_to(doy.astype(lat.dtype), lat.shape)
    else:
        doy_arr = jnp.full(lat.shape, doy, dtype=lat.dtype)
    if config.hemisphere_aware:
        doy_eff = jnp.where(
            lat < 0,
            jnp.mod(doy_arr + _HALF_YEAR_OFFSET_DAYS, _DAYS_PER_YEAR),
            doy_arr,
        )
    else:
        doy_eff = doy_arr

    # Labile release factor — Gaussian pulse around Bday
    arg_l = jnp.sin((doy_eff - config.Bday + osl) / sf) * sf / wl
    lrf = inv_sqrt_pi * (fl / wl) * jnp.exp(-arg_l ** 2)

    # Leaf fall factor — Gaussian pulse around Fday
    arg_f = jnp.sin((doy_eff - config.Fday + osf) / sf) * sf / wf
    lff = inv_sqrt_pi * (ff / wf) * jnp.exp(-arg_f ** 2)

    return lrf, lff


# ===================================================================
# Decomposition modifier
# ===================================================================

def _temperate_modifier(
    T: jnp.ndarray,
    precip: jnp.ndarray,
    config: CarbonConfig,
) -> jnp.ndarray:
    """Temperature-moisture modifier for heterotrophic (soil) respiration.

    Uses ``Q10_het_exp`` (soil-decomposition Q10 ~2.5), decoupled from the
    autotrophic ``Q10_exp``: warm soils turn SOM/litter over fast (low
    equilibrium SOM) while cold soils retain carbon.

    Returns an ENVIRONMENTAL ACCELERATION factor that legitimately EXCEEDS 1
    for warm/wet conditions (faster than the ``T_ref``/``precip_ref``
    reference-condition rate) -- see :func:`_som_decomp_modifier` for why
    this is intended, numerically safe, and must NOT be clipped to [0, 1].
    """
    temp_factor = jnp.exp(config.Q10_het_exp * (T - config.T_ref))
    precip_ratio = precip / jnp.maximum(config.precip_ref, 1e-10)
    moist = (precip_ratio - 1.0) * config.moisture_factor + 1.0
    moist = jnp.clip(moist, config.moist_modifier_min, config.moist_modifier_max)
    return temp_factor * moist


def _freeze_modifier(
    T: jnp.ndarray,
    config: CarbonConfig,
) -> jnp.ndarray:
    """Freeze suppression of soil decomposition, ``f_freeze`` in
    ``[som_freeze_floor, 1]``.

    Reuses the smooth sigmoid freezing characteristic of
    ``soil_thermal.liquid_water_content`` (the unfrozen-liquid-water fraction
    that microbial decomposition tracks) — NOT a re-derived curve — then FLOORS
    it so a frozen soil still decomposes a small nonzero fraction of its
    unfrozen rate::

        f_freeze(T) = som_freeze_floor
                      + (1 - som_freeze_floor)
                        * sigmoid((T - T_freeze) / som_freeze_width_K)

    -> 1 for warm soil (T >> T_freeze, no suppression), -> ``som_freeze_floor``
    for frozen soil (T << T_freeze, decomposition floored, NOT shut off),
    monotonically INCREASING in T and bounded in ``[som_freeze_floor, 1]``.
    Frozen soils therefore turn SOM over slowly and RETAIN carbon (the
    high-latitude / grassland SOC fix), but the nonzero floor keeps the
    millennial slow/passive-pool equilibrium ``I / (m * k)`` FINITE — without it
    ``f_freeze -> 0`` (``m -> 0``) drove a runaway cold-soil SOC accumulation
    (CLM4.5 / CENTURY cold-soil decomposition floor).  ``som_freeze_width_K`` is
    a fixed NUMERICS half-width [K]; ``som_freeze_floor`` is a tunable closure
    fraction in ``[0, 1)``.  Both validated on the STATIC Python config values
    (a bare Python branch is safe — feature-gating exception).
    """
    w = config.som_freeze_width_K
    if not w > 0.0:
        raise ValueError(f"som_freeze_width_K must be > 0, got {w!r}.")
    floor = config.som_freeze_floor
    # ``floor`` may be a sigmoid-CONSTRAINED JAX leaf in the offline
    # differentiable carbon-calibration path
    # (global_init.equilibrate_archetypes_traced): the bound is then guaranteed by
    # the transform and a Python bool on it -- even a concrete array lifted inside
    # the scan -- raises, so the fail-early check enforces on concretely-knowable
    # values only (host scalars AND concrete arrays; see is_concrete).
    if is_concrete(floor) and not 0.0 <= floor < 1.0:
        raise ValueError(
            f"som_freeze_floor must be in [0, 1), got {floor!r}.")
    # Floored sigmoid: f_freeze = floor + (1 - floor) * sigmoid(...), ranging
    # [floor, 1].  floor in [0, 1) keeps (1 - floor) > 0 so the curve stays
    # monotonically increasing and bounded, and floor <= f_freeze <= 1.
    return floor + (1.0 - floor) * jax.nn.sigmoid((T - constants.T_freeze) / w)


def _nsc_respiration_factor(
    C_lab: jnp.ndarray,
    C_fol: jnp.ndarray,
    C_root: jnp.ndarray,
    C_wood: jnp.ndarray,
    config: CarbonConfig,
) -> jnp.ndarray:
    """Substrate (NSC) limitation of maintenance respiration, ``f_nsc`` in
    ``[r_maint_floor_frac, 1]``.

    Respiratory downregulation under carbon starvation (Atkin & Tjoelker 2003):
    when the labile / non-structural-carbon reserve ``C_lab`` is depleted
    relative to a fraction of LIVE BIOMASS, maintenance respiration throttles
    toward a small basal floor instead of demanding the full biomass-proportional
    amount and cannibalising structural pools to death::

        C_lab_ref = nsc_ref_labile_frac * (C_fol + C_root + C_wood)
        f_nsc = r_maint_floor_frac
                + (1 - r_maint_floor_frac) * smoothstep(C_lab / C_lab_ref)

    ``smoothstep`` is the C1 Hermite ``x^2 (3 - 2x)`` on a ``[0, 1]``-clamped
    argument -> differentiable; ``f_nsc -> 1`` for ample reserve
    (``C_lab >= C_lab_ref``; healthy plants unaffected -> temperate/tropical
    no-regression) and ``-> r_maint_floor_frac`` as ``C_lab -> 0``.

    The reference scales with LIVE BIOMASS, not foliage: a winter-leafless plant
    has ``C_fol -> 0``, so a foliage-only reference would collapse and leave the
    root+wood winter drain ungated (the exact death-spiral case).  ``C_root`` /
    ``C_wood`` persist through the leafless season and keep the reference finite.
    A ``1e-10`` epsilon guards ``ref`` so a fully bare column yields the floor,
    not ``0/0``.
    """
    # Fail-early on the STATIC config values (dispatch-hardening; the
    # __param_spec__ bounds only constrain training, so a direct
    # CarbonConfig(r_maint_floor_frac=1.5) would otherwise give f_nsc > 1 and
    # INCREASE R_maint, violating the pure-reduction invariant).  ``is_concrete``
    # so a sigmoid-constrained TRACED leaf in the calibration path is skipped.
    floor = config.r_maint_floor_frac
    if is_concrete(floor) and not 0.0 <= floor <= 1.0:
        raise ValueError(f"r_maint_floor_frac must be in [0, 1], got {floor!r}.")
    ref_frac = config.nsc_ref_labile_frac
    if is_concrete(ref_frac) and not ref_frac > 0.0:
        raise ValueError(f"nsc_ref_labile_frac must be > 0, got {ref_frac!r}.")
    ref = ref_frac * (C_fol + C_root + C_wood)
    x = jnp.clip(C_lab / jnp.maximum(ref, 1e-10), 0.0, 1.0)
    smoothstep = x * x * (3.0 - 2.0 * x)
    return floor + (1.0 - floor) * smoothstep


def _cold_deciduous_dormancy_factor(
    T: jnp.ndarray,
    config: CarbonConfig,
) -> jnp.ndarray:
    """Cold-deciduous winter-dormancy factor ``d`` in ``(0, 1]``.

    Smooth freeze-onset sigmoid ``d = sigmoid((T - freeze_dormancy_threshold_K)
    / dormancy_transition_width_K)`` -> ``1`` for warm (active canopy), ``-> 0``
    when frozen (leaves shed / metabolically dormant).  Callers multiply it onto
    the FOLIAR GPP and FOLIAR maintenance-respiration terms so a cold-deciduous
    PFT neither photosynthesises nor pays foliar respiration through the frozen
    season (larch/tundra strategy).  Differentiable; the ``cold_deciduous`` /
    ``cold_deciduous_dormancy`` STATIC gates are applied by the caller (feature
    gating), so this returns the smooth factor unconditionally.
    """
    w = config.dormancy_transition_width_K
    if not w > 0.0:
        raise ValueError(
            f"dormancy_transition_width_K must be > 0, got {w!r}.")
    z = (T - config.freeze_dormancy_threshold_K) / w
    return jax.nn.sigmoid(z)


def perennial_frost_protection(
    frozen_fraction: jnp.ndarray,
    config: CarbonConfig,
) -> jnp.ndarray:
    """Perennial-frost / anaerobic SOM protection factor ``f_perma`` in
    ``[permafrost_protection_min, 1]``, driven by the ANNUAL frozen fraction.

    Permafrost carbon protection (Koven et al. 2013; CLM4.5 cold-soil
    biogeochemistry; Hugelius et al. 2014 NCSCD stocks).  ``_freeze_modifier``
    already suppresses decomposition per-timestep whenever the soil is frozen,
    flooring it at the AEROBIC ``som_freeze_floor`` -- but the ANNUAL turnover of
    a cold column is dominated by its brief unfrozen (thaw-season) window, so
    that instantaneous floor alone leaves high-latitude SOC capped far below the
    observed 100-300 kgC/m2 permafrost/peat stocks.  A PERENNIALLY-frozen,
    waterlogged column protects its SOM ALL YEAR -- anaerobic (O2-limited)
    decomposition in the meltwater-saturated active layer + cryoturbation
    burying carbon into the perennially-frozen, decomposition-shielded permafrost
    -- so this is a SEPARATE, whole-column suppression keyed on the PERENNIAL-
    frost STATE (an annual statistic), not the instantaneous temperature.

    SIGN / UNITS (carbon RETAINED, never created; see ``step_carbon_differland``
    SOM cascade).  ``frozen_fraction`` in ``[0, 1]`` (dimensionless annual frozen
    fraction).  Returns a dimensionless suppression factor.  Direction, walked
    at the term: MORE frost (``frozen_fraction`` UP) -> ``sigmoid`` UP ->
    ``f_perma`` DOWN -> the SOM decomposition modifier ``m`` DOWN -> per-step
    respiration/humification loss ``D_X = C_X*eff_rate(m*k_X)`` DOWN -> the pool
    RETAINS more carbon -> equilibrium ``C_X = I_X/(m*k_X)`` UP.  No carbon is
    created: the suppressed loss simply stays as pool storage, and the SOM
    sub-column budget ``d(sum C_som) = (input - R_het_som)*dt`` still closes
    exactly (``f_perma`` only rescales ``m``, a factor already inside the loss).
    ``frozen_fraction = 0`` (temperate/tropical, never frozen) ->
    ``f_perma`` ~ 1 (UNCHANGED, since the threshold sits well above 0), so the
    already-correct warm-soil SOC is preserved.  ``frozen_fraction -> 1``
    (perennial permafrost) -> ``f_perma -> permafrost_protection_min``, which
    (multiplying through ``_freeze_modifier``'s ``som_freeze_floor``) drives the
    effective rate BELOW the aerobic floor -- the coherent reconciliation of the
    two: ``som_freeze_floor`` is the unfrozen-season aerobic minimum, ``f_perma``
    the perennial-frost/anaerobic protection.

        f_perma = 1 - (1 - permafrost_protection_min)
                      * sigmoid((frozen_fraction - threshold) / width)

    -> monotonically DECREASING in ``frozen_fraction``, bounded in
    ``[permafrost_protection_min, 1]`` (``permafrost_protection_min`` in ``(0, 1]``
    keeps the millennial ``I/(m*k)`` equilibrium FINITE and the factor a pure
    SUPPRESSION, never an amplification).  Validated on the STATIC config values
    (concrete host scalars / concrete arrays only; a sigmoid-CONSTRAINED traced
    calibration leaf is guaranteed in-bounds by its transform -- see
    ``_freeze_modifier``).
    """
    p_min = config.permafrost_protection_min
    thr = config.permafrost_frozen_fraction_threshold
    w = config.permafrost_frozen_fraction_width
    if not w > 0.0:
        raise ValueError(
            f"permafrost_frozen_fraction_width must be > 0, got {w!r}.")
    if is_concrete(p_min) and not 0.0 < p_min <= 1.0:
        raise ValueError(
            f"permafrost_protection_min must be in (0, 1], got {p_min!r}.")
    if is_concrete(thr) and not 0.0 < thr < 1.0:
        raise ValueError(
            f"permafrost_frozen_fraction_threshold must be in (0, 1), "
            f"got {thr!r}.")
    return 1.0 - (1.0 - p_min) * jax.nn.sigmoid((frozen_fraction - thr) / w)


def annual_frozen_fraction(
    mat_k: jnp.ndarray,
    t_seasonal_amp_k: jnp.ndarray,
    config: CarbonConfig,
) -> jnp.ndarray:
    """Annual frozen fraction ``phi`` in ``[0, 1]`` of the climatological
    near-surface temperature cycle -- the perennial-frost INDEX driving
    :func:`perennial_frost_protection`.

    Physical choice (documented per requirement).  The perennial-frost state
    (permafrost vs seasonally-frozen) is an ANNUAL property that a single
    timestep cannot resolve, so the protection is keyed on the FRACTION OF THE
    YEAR the soil is frozen -- the standard climatological permafrost predictor
    (mean-annual air temperature / freezing index; Gruber 2012 permafrost
    zonation index).  It is computed here from the climatological annual
    temperature cycle ``T(doy) = mat_k + t_seasonal_amp_k * cos(2*pi*doy/year)``
    (the same seasonal cycle :func:`legoesm.land.climate_forcing.
    make_climatological_forcing` drives the archetype spin-up with, and that the
    recorded ``fast_analytic.FastAnalyticInputs.soil_T_traj`` top-soil
    temperature tracks) rather than the spin-up ``soil_T_traj`` itself for ONE
    decisive reason: the coupled spin-up applies the protection per-timestep
    (it cannot see the whole year) while the closed-form surrogate applies it to
    the annual turnover, and BOTH must use a BYTE-IDENTICAL per-column ``phi`` or
    the surrogate-vs-spin-up fidelity gate breaks (``f_perma`` would rescale the
    two forwards by different factors).  Deriving ``phi`` from the deterministic
    climatological cycle -- available a priori to both paths from the SAME
    per-archetype ``(mat_k, t_seasonal_amp_k)`` -- guarantees that identity, and
    is robust to spin-up-transient / snow-insulation contamination that a
    ``soil_T_traj`` estimate would carry.  The fraction is invariant to the
    seasonal PHASE (a full-year integral), so the peak-day offset is irrelevant.

    Smooth / differentiable: the hard ``T < T_freeze`` indicator is the freeze
    sigmoid ``sigmoid((T_freeze - T)/som_freeze_width_K)`` (reusing the SOM
    freeze-curve half-width -- no new curve), averaged over the year::

        phi = mean_doy[ sigmoid((T_freeze - T(doy)) / som_freeze_width_K) ]

    ``mat_k`` / ``t_seasonal_amp_k`` broadcast together; returns a per-column
    ``phi`` (same shape).  ``phi`` is SOM-parameter-independent (climate only),
    so it is precomputed ONCE and frozen (like ``soil_T_traj``); the trainable
    perennial-frost parameters enter only through
    :func:`perennial_frost_protection`.
    """
    w = config.som_freeze_width_K
    if not w > 0.0:
        raise ValueError(f"som_freeze_width_K must be > 0, got {w!r}.")
    mat = jnp.asarray(mat_k)[..., None]
    amp = jnp.asarray(t_seasonal_amp_k)[..., None]
    # Uniform within-year samples of the seasonal cosine (phase-invariant for a
    # full-year mean).  ``2*pi``: math; ``_FROST_INDEX_SAMPLES``: numerics count.
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, _FROST_INDEX_SAMPLES, endpoint=False)
    T_cycle = mat + amp * jnp.cos(theta)                # (..., n_samples) [K]
    frozen_indicator = jax.nn.sigmoid((constants.T_freeze - T_cycle) / w)
    return jnp.mean(frozen_indicator, axis=-1)          # (...,) in [0, 1]


def _som_decomp_modifier(
    T: jnp.ndarray,
    precip: jnp.ndarray,
    config: CarbonConfig,
    frozen_fraction: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """SOM decomposition-rate modifier ``m = f_temp * f_moist * f_freeze``.

    EXTENDS the shared litter/soil :func:`_temperate_modifier` (temperature Q10
    + moisture) with the :func:`_freeze_modifier` freeze suppression, so cold /
    frozen soils turn their organic matter over slowly.  Applied to all three
    SOM cascade pools; the surface-litter path keeps the freeze-free
    ``_temperate_modifier`` (fresh litter is not the deep freeze-protected SOC
    reservoir this fix targets).

    ``m`` is an ENVIRONMENTAL ACCELERATION factor, NOT a bounded [0, 1]
    fraction, and ROUTINELY EXCEEDS 1: warm/wet soils decompose FASTER than
    the reference-condition base rate ``tor_som_X`` (at ``T_ref``/
    ``precip_ref``) -- this is the intended Q10/moisture control that
    produces the observed warm-fast / cold-slow SOC gradient (see
    ``TestColdWarmSocRealism``).  Only ``f_freeze in [som_freeze_floor, 1]`` is a
    bounded SUPPRESSION term; ``f_temp`` (Q10 exponential, unbounded above) and
    ``f_moist`` (clipped to ``[moist_modifier_min, moist_modifier_max]``,
    default max 3.0) are both free to exceed 1, so ``m`` has no upper bound.

    DO NOT clip ``m`` to [0, 1] -- that would remove the temperature
    acceleration and give every soil the same turnover regardless of
    climate, which is wrong.  Numerical safety does NOT depend on bounding
    ``m``: :func:`_effective_rate` clips ``rate = m * tor_som_X`` to
    ``[0, 1 - 1e-10]`` before computing ``1 - (1 - rate)**dt_days``, so the
    DECOMPOSED FRACTION of a pool lost per step is always in ``[0, 1)``
    regardless of how large ``m`` grows -- a modifier > 1 can never overdraw
    a pool.

    ``frozen_fraction`` (optional, per-column annual frozen fraction from
    :func:`annual_frozen_fraction`): when provided, ``m`` is additionally scaled
    by the perennial-frost / anaerobic protection ``f_perma in
    [permafrost_protection_min, 1]`` (:func:`perennial_frost_protection`), a
    SECOND bounded SUPPRESSION (like ``f_freeze``) so cold PERENNIALLY-frozen
    columns turn SOM over even more slowly (permafrost carbon; Koven et al.
    2013).  ``None`` (default) -> no protection (``f_perma == 1``), so every
    existing caller is byte-identical.  A per-column suppression <= 1 keeps ``m``
    upper-bound-free reasoning unchanged (``_effective_rate`` still guards
    safety).
    """
    m = _temperate_modifier(T, precip, config) * _freeze_modifier(T, config)
    if frozen_fraction is not None:
        m = m * perennial_frost_protection(frozen_fraction, config)
    return m


# ===================================================================
# Effective turnover for finite timestep
# ===================================================================

def _effective_rate(rate: jnp.ndarray, dt_days: jnp.ndarray) -> jnp.ndarray:
    """Convert daily rate *r* to effective flux fraction per day.

    Returns (1 - (1-r)^dt) / dt  so that  C * eff * dt = C * (1-(1-r)^dt),
    the exact exponential-decay loss over the timestep.
    """
    rate = jnp.clip(rate, 0.0, 1.0 - 1e-10)
    return (1.0 - (1.0 - rate) ** dt_days) / jnp.maximum(dt_days, 1e-10)


def som_decomposition_rate(
    T: jnp.ndarray,
    precip: jnp.ndarray,
    tor_som: jnp.ndarray,
    config: CarbonConfig,
    dt_days: jnp.ndarray,
    frozen_fraction: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Effective per-day SOM decomposition FRACTION for a pool whose base
    (reference-T, unfrozen) turnover is *tor_som* under conditions ``(T, precip)``.

    Public wrapper reusing the EXACT kinetics the prognostic step applies to each
    SOM pool (see :func:`step_carbon_differland`)::

        r = _effective_rate(
                _som_decomp_modifier(T, precip, config, frozen_fraction) * tor_som,
                dt_days)

    i.e. the shared ``f_temp * f_moist * f_freeze [* f_perma]`` environmental
    modifier (:func:`_som_decomp_modifier`) scales the base rate, then
    :func:`_effective_rate` maps it through the exact finite-dt exponential-decay
    handling.  Exposed so the offline differentiable fast-analytic SOC calibration
    (:mod:`legoesm.land.carbon.fast_analytic`) computes the per-pool turnover from
    the SAME modifier + finite-dt convention as the model, rather than importing
    the private helpers or re-deriving the numerics.  ``T``/``precip`` broadcast
    together; ``tor_som`` is a scalar (or broadcastable) base turnover [1/day].
    ``frozen_fraction`` (optional, per-column annual frozen fraction) threads the
    perennial-frost protection ``f_perma`` through so the surrogate applies the
    IDENTICAL suppression the coupled step does.
    """
    return _effective_rate(
        _som_decomp_modifier(T, precip, config, frozen_fraction) * tor_som,
        dt_days)


def sequential_allocation(
    npp_pos: jnp.ndarray,
    f_fol: jnp.ndarray,
    f_lab: jnp.ndarray,
    f_root: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """DALEC990 sequential NPP partition: absolute allocation fluxes
    ``(A_fol, A_lab, A_root_base, A_wood_raw)`` from a positive NPP throughput.

    The single definition of the DifferLand allocation cascade, shared by the
    prognostic step (:func:`step_carbon_differland`, which multiplies the per-day
    ``NPP_pos`` rate) and the offline closed-form live-pool equilibrium
    (:func:`legoesm.land.carbon.live_pool_forward.compute_live_pools`, which
    multiplies the ANNUAL allocatable NPP) -- one cascade, never re-derived.  Each
    fraction takes a slice of the REMAINDER after the previous pool, so the four
    fluxes sum to ``npp_pos`` EXACTLY (telescoping)::

        A_fol       = npp_pos * f_fol
        A_lab       = (npp_pos - A_fol) * f_lab
        A_root_base = (npp_pos - A_fol - A_lab) * f_root
        A_wood_raw  = npp_pos - A_fol - A_lab - A_root_base      # = the remainder

    ``A_wood_raw`` is the STRUCTURAL remainder BEFORE the woody/herbaceous routing
    (the caller decides whether it forms wood or is invested belowground); it is
    ``>= 0`` for ``npp_pos >= 0`` and fractions in ``[0, 1]``.  All arguments
    broadcast together; ``f_*`` may be static Python floats (production) or traced
    ``CarbonConfig`` overrides (calibration).  Sign convention: every returned flux
    is a POSITIVE carbon INPUT to its pool [same units as ``npp_pos``].
    """
    A_fol = npp_pos * f_fol
    A_lab = (npp_pos - A_fol) * f_lab
    A_root_base = (npp_pos - A_fol - A_lab) * f_root
    A_wood_raw = npp_pos - A_fol - A_lab - A_root_base
    return A_fol, A_lab, A_root_base, A_wood_raw


# ===================================================================
# DifferLand prognostic step
# ===================================================================

def step_carbon_differland(
    state: CarbonState,
    sw_down: jnp.ndarray,
    T: jnp.ndarray,
    co2_ppmv: jnp.ndarray,
    beta: jnp.ndarray,
    lat: jnp.ndarray,
    doy: float,
    precip: jnp.ndarray,
    config: CarbonConfig,
    dt: float,
    gpp_override: jnp.ndarray | None = None,
    return_diagnostics: bool = False,
    soil_frozen_fraction: jnp.ndarray | None = None,
) -> (
    tuple[CarbonState, jnp.ndarray]
    | tuple[CarbonState, jnp.ndarray, CarbonDiagnostics]
):
    """Advance all eight carbon pools by *dt* seconds.

    Parameters
    ----------
    state    : Current carbon pools [gC/m2].
    sw_down  : Downward SW radiation [W/m2].
    T        : Surface temperature [K].
    co2_ppmv : Atmospheric CO2 [ppmv].
    beta     : Moisture availability [0-1].
    lat      : Latitude [radians].
    doy      : Day of year.
    precip   : Total precipitation [kg/m2/s].
    config   : CarbonConfig.
    dt       : Timestep [s].
    return_diagnostics : When True, also return a :class:`CarbonDiagnostics`
        with the full GPP / NPP / respiration / allocation / turnover
        breakdown (per-day rates [gC/m2/day]).  Purely diagnostic — the
        state update is byte-identical either way.
    soil_frozen_fraction : Optional per-column annual frozen fraction
        (:func:`annual_frozen_fraction`) enabling the perennial-frost /
        anaerobic SOM protection (:func:`perennial_frost_protection`) on the
        three SOM cascade pools.  ``None`` (default) -> no protection, so every
        existing caller (production coupled ESM, tests) is byte-identical; the
        archetype spin-up passes it so perennially-frozen high-latitude columns
        accumulate the observed deep permafrost SOC.  NOT applied to the surface
        litter path (fresh litter is not the deep freeze/frost-protected
        reservoir), matching the freeze-modifier treatment.

    Returns
    -------
    new_state : Updated CarbonState.
    co2_flux  : Net CO2 flux [kgCO2/m2/s], positive up.
    diag      : (only when ``return_diagnostics``) CarbonDiagnostics.
    """
    # Fail-early validation on the STATIC config value (dispatch-hardening
    # discipline; see config.validate_som_transfer_fractions for why this
    # cannot rely on the __param_spec__ training bounds alone).
    validate_som_transfer_fractions(
        config.f_active_to_slow, config.f_slow_to_passive,
        config.cwd_humification_eff)
    dt_days = dt / _SPD

    # LAI from foliar carbon
    LAI = state.C_fol / config.LCMA

    # --- GPP ---------------------------------------------------------------
    if gpp_override is not None:
        gpp = gpp_override
    else:
        gpp = compute_gpp(sw_down, T, LAI, co2_ppmv, beta, config)  # gC/m2/s
    # Cold-deciduous freeze dormancy (Mechanism 2): zero foliar GPP when the
    # canopy is frozen/dormant.  Applied HERE, on the FINAL gpp, so the coupled
    # archetype path's nonzero Farquhar gpp_override is gated too (codex P1 --
    # gating only compute_gpp left the override path un-suppressed).  STATIC
    # gate -> Python if (feature gating); d == 1.0 when off => byte-identical.
    if config.cold_deciduous_dormancy and config.cold_deciduous:
        gpp = gpp * _cold_deciduous_dormancy_factor(T, config)
    gpp_day = gpp * _SPD  # gC/m2/day rate

    # --- Autotrophic respiration & NPP -------------------------------------
    # Maintenance respiration: biomass-proportional, temperature-dependent,
    # always active (including nighttime and dormant seasons).
    # Autotrophic maintenance-respiration temperature factor.  Reference is the
    # autotrophic normalization (config.T_ref_ra = 25 degC), SEPARATE from the
    # heterotrophic reference (config.T_ref, used for soil decomposition in
    # _temperate_modifier above) so the autotrophic calibration is independent of the
    # heterotrophic config.  T_ref_ra sets ONLY the overall magnitude (it is
    # confounded with the bulk r_maint_* amplitudes; see config) -- it does NOT
    # change the warm/cold shape, which is set by Q10_exp.  Sign convention (carbon,
    # Ra is a LOSS to the atmosphere; NPP = GPP - R_auto): Q10_exp > 0, so T warmer
    # than the reference => factor > 1 (more maintenance loss).  Raising the
    # reference multiplies temp_factor_ra by exp(-Q10_exp*dref) at EVERY fixed
    # temperature, UNIFORMLY lowering R_maint and RAISING NPP.
    temp_factor_ra = jnp.exp(config.Q10_exp * (T - config.T_ref_ra))
    # High-latitude productivity rescue (opt-in, static-flag feature gates).
    # Mechanism 1 (NSC gate) throttles ALL maintenance terms as the labile
    # reserve depletes; Mechanism 2 (cold-deciduous dormancy) additionally zeros
    # the FOLIAR term when frozen.  f_nsc == 1.0 and d == 1.0 when off =>
    # byte-identical to the ungated (r_maint_fol*C_fol + ...)·temp_factor_ra.
    # Sign (carbon, positive-out): R_maint is a LOSS plant->atmosphere; f_nsc, d
    # in (0, 1] REDUCE the loss (survival), never increase it.
    if config.nsc_gated_respiration:
        f_nsc = _nsc_respiration_factor(
            state.C_lab, state.C_fol, state.C_root, state.C_wood, config)
    else:
        f_nsc = 1.0
    if config.cold_deciduous_dormancy and config.cold_deciduous:
        d = _cold_deciduous_dormancy_factor(T, config)
    else:
        d = 1.0
    R_maint_day = (
        config.r_maint_fol * state.C_fol * f_nsc * d
        + config.r_maint_root * state.C_root * f_nsc
        + config.r_maint_wood * state.C_wood * f_nsc
    ) * temp_factor_ra  # gC/m2/day

    # Growth respiration: fraction of net assimilation (GPP minus maintenance)
    R_growth_day = config.f_auto * jnp.maximum(gpp_day - R_maint_day, 0.0)
    R_auto_day = R_maint_day + R_growth_day
    NPP_day = gpp_day - R_auto_day

    # --- NPP allocation (sequential partition) -----------------------------
    # Allocation only occurs when NPP > 0 (growth); maintenance losses are
    # already accounted for in R_auto_day and flow directly to atmosphere.
    NPP_pos = jnp.maximum(NPP_day, 0.0)
    # Shared DALEC cascade (identical to the closed-form live-pool forward's
    # allocation) -- the telescoping A_wood_raw = NPP_pos - A_fol - A_lab -
    # A_root_base is the same expression as before, so the pool update is
    # byte-identical; only a defensive non-negativity clip is applied here.
    A_fol, A_lab, A_root_base, A_wood_raw = sequential_allocation(
        NPP_pos, config.f_fol, config.f_lab, config.f_root)
    A_wood_raw = jnp.maximum(A_wood_raw, 0.0)
    # Woodiness is a static per-PFT flag -> Python branch (feature-gating
    # exception, not a data-dependent jnp.where).  Herbaceous PFTs (grass,
    # crop, tundra) have no wood: the structural fraction that would form wood
    # is invested belowground (roots) instead, so a grassland does not silently
    # grow a phantom multi-kgC tree pool.  Allocation still closes exactly
    # (A_fol + A_lab + A_root + A_wood == NPP_pos) either way.
    if config.woody:
        A_root = A_root_base
        A_wood = A_wood_raw
    else:
        A_root = A_root_base + A_wood_raw
        A_wood = jnp.zeros_like(A_wood_raw)

    # --- Phenology ---------------------------------------------------------
    lrf, lff = compute_phenology(jnp.asarray(doy), lat, config)

    lab_release = state.C_lab * _effective_rate(lrf, dt_days)   # gC/m2/day
    leaf_litter = state.C_fol * _effective_rate(lff, dt_days)

    # --- Cold-deciduous leaf bootstrap (Mechanism 2, opt-in) ----------------
    # A cold-deciduous plant that lost its canopy (C_fol -> 0) but kept a labile
    # reserve (the NSC gate preserved it) must REGROW a minimum leaf area from
    # labile in the growing season; otherwise GPP -- computed once per step from
    # the ENTRY C_fol (LAI = C_fol / LCMA) -- stays 0, so NPP < 0, A_fol = 0, and
    # the plant is locked dead (larch / tundra).  Draw EXTRA labile toward a
    # minimum leaf mass when ACTIVE (dormancy factor ~1, growing season),
    # LEAFLESS (C_fol below target), and the reserve allows it.  This is a PURE
    # C_lab -> C_fol transfer: it is added to lab_release, which the pool update
    # DEBITS from C_lab and CREDITS to C_fol symmetrically, and it is capped at
    # the reserve, so carbon is CONSERVED (NOT a jnp.maximum C_fol floor, which
    # would create leaf mass).  STATIC gate -> Python if (feature gating); a
    # no-op when the flag is off => byte-identical.
    if config.cold_deciduous_dormancy and config.cold_deciduous:
        # Input hardening (codex): the __param_spec__ bounds only constrain
        # training, so validate the concrete config -- a NEGATIVE frac/lai would
        # make the transfer negative and drive C_fol below 0 (clipped by
        # _soft_pos -> CREATED carbon).
        frac = config.leaf_bootstrap_frac
        if is_concrete(frac) and not frac >= 0.0:
            raise ValueError(f"leaf_bootstrap_frac must be >= 0, got {frac!r}.")
        lai_min = config.leaf_bootstrap_lai
        if is_concrete(lai_min) and not lai_min >= 0.0:
            raise ValueError(f"leaf_bootstrap_lai must be >= 0, got {lai_min!r}.")
        d_boot = _cold_deciduous_dormancy_factor(T, config)     # ~1 active, ~0 dormant
        fol_target = lai_min * config.LCMA                      # min leaf mass [gC/m2]
        fol_gap = jnp.maximum(fol_target - state.C_fol, 0.0)    # leaf shortfall [gC/m2]
        # Cap the extra transfer to the reserve REMAINING after the natural
        # labile release, as a MASS subtraction (not a C_lab/dt_days rate
        # round-trip), so C_lab - (lab_release + boot)*dt_days >= 0 without an
        # fp32 ULP underflow (codex P3; exact closure holds in the x64 carbon
        # regime).  dt_days is guarded > 0 (codex P2: a direct dt=0 caller).
        reserve_left = jnp.maximum(state.C_lab - lab_release * dt_days, 0.0)   # gC/m2
        boot_mass = d_boot * jnp.minimum(
            jnp.minimum(fol_gap, state.C_lab * frac), reserve_left)           # gC/m2
        lab_release = lab_release + boot_mass / jnp.maximum(dt_days, 1e-10)

    # --- Structural turnover -----------------------------------------------
    wood_litter = state.C_wood * _effective_rate(
        jnp.asarray(config.tor_wood), dt_days,
    )
    root_litter = state.C_root * _effective_rate(
        jnp.asarray(config.tor_root), dt_days,
    )

    # NPP deficit (NPP_day < 0, GPP cannot cover R_auto).  Cascade the draw
    # C_lab → C_wood → C_root → C_fol so the atmosphere gain via NEE is matched
    # exactly by biomass loss, even if any single pool is exhausted.
    #
    # DRAW ORDER = mobile reserve first, PHOTOSYNTHETIC ORGANS last.  A
    # DALEC-style daily respiration model run at the land model's sub-daily dt
    # sees GPP=0 every night while the biomass-proportional maintenance
    # respiration stays on, so every night NPP_day<0 and this cascade fires to
    # pay maintenance from stored carbon.  Which pool absorbs that recurring
    # nighttime draw matters:
    #   * Draining C_fol (the leaves) collapses LAI → GPP → the whole column
    #     (a carbon death-spiral once the labile reserve empties at leaf-fall).
    #   * Draining C_root (a functional, relatively small pool) empties it in
    #     weeks — a grassland root system cannot buffer a forest's respiration.
    # So after the labile (NSC) reserve, the draw goes to WOOD — the largest,
    # most allocation-fed pool (also the plant's structural NSC store) — which
    # absorbs the diurnal deficit yet still net-accumulates because daytime
    # allocation (the wood share of NPP) exceeds the nighttime draw whenever
    # daily NPP>0.  Roots and leaves are touched only in genuine starvation
    # (labile+wood both exhausted, e.g. a leafless herbaceous column in polar
    # winter), and the leaves strictly last.  This keeps every functional pool
    # at a physical steady state instead of one pool cannibalising to zero.
    #
    # Each draw is capped at the **net** pool contents after natural turnover
    # flows (lab_release / leaf_litter / etc.) so the pool cannot go below 0
    # once both the natural drain AND the deficit are applied (codex iter-63:
    # the cap must subtract the natural turnover drain, not use raw state.C_x).
    _inv_dt_days = 1.0 / jnp.maximum(dt_days, 1e-10)
    npp_deficit_day = jnp.maximum(-NPP_day, 0.0)
    # Net pool capacity per day after subtracting natural turnover drain
    # (and adding natural turnover gain, e.g. ``lab_release → C_fol``).
    # ``A_x`` contributions are zero in deficit regime (NPP_pos = 0).
    lab_net_avail = jnp.maximum(
        state.C_lab + (A_lab - lab_release) * dt_days, 0.0,
    ) * _inv_dt_days
    fol_net_avail = jnp.maximum(
        state.C_fol + (A_fol + lab_release - leaf_litter) * dt_days, 0.0,
    ) * _inv_dt_days
    root_net_avail = jnp.maximum(
        state.C_root + (A_root - root_litter) * dt_days, 0.0,
    ) * _inv_dt_days
    wood_net_avail = jnp.maximum(
        state.C_wood + (A_wood - wood_litter) * dt_days, 0.0,
    ) * _inv_dt_days

    # Draw order: labile (mobile NSC) → wood (structural store/buffer)
    # → root → foliage (photosynthetic organ, strictly last).
    lab_deficit_draw = jnp.minimum(npp_deficit_day, lab_net_avail)
    remaining_after_lab = npp_deficit_day - lab_deficit_draw
    wood_deficit_draw = jnp.minimum(remaining_after_lab, wood_net_avail)
    remaining_after_wood = remaining_after_lab - wood_deficit_draw
    root_deficit_draw = jnp.minimum(remaining_after_wood, root_net_avail)
    remaining_after_root = remaining_after_wood - root_deficit_draw
    fol_deficit_draw = jnp.minimum(remaining_after_root, fol_net_avail)
    # Unmet NPP deficit: the part of the maintenance-respiration demand that
    # NO pool could supply (all biomass exhausted).  R_auto is charged to
    # the atmosphere via NEE below, so the flux MUST be reduced by this
    # undrawn remainder or NEE reports carbon that left no pool (atmosphere
    # gain > biomass loss).  See finding #5.
    unmet_npp_deficit_day = jnp.maximum(
        remaining_after_root - fol_deficit_draw, 0.0,
    )

    # --- Heterotrophic respiration & decomposition -------------------------
    tempmod = _temperate_modifier(T, precip, config)

    # The litter pool is drained by TWO independent first-order sinks:
    # heterotrophic respiration (-> atmosphere) and decomposition to SOM
    # (-> C_som).  Each ``_effective_rate`` is the exact single-sink decay,
    # but their SUM can exceed the litter net availability for large dt,
    # which ``_soft_pos`` would clip while NEE still reported the full,
    # uncapped R_het_lit -> atmosphere gain > litter loss.  Apply a joint
    # net-availability cap (cf. the biomass deficit cascade): scale both
    # demanded sinks by the same factor so their realised sum never exceeds
    # ``C_lit + (leaf_litter + root_litter)*dt`` (the litter content after
    # this step's litter inputs).  The atm/SOM partition ratio is preserved.
    # See finding #4.
    R_het_lit_demand = state.C_lit * _effective_rate(
        tempmod * config.tor_litter, dt_days,
    )
    lit_to_som_demand = state.C_lit * _effective_rate(
        tempmod * config.decomp_rate, dt_days,
    )
    lit_total_demand = R_het_lit_demand + lit_to_som_demand  # gC/m2/day
    lit_net_avail = jnp.maximum(
        state.C_lit + (leaf_litter + root_litter) * dt_days, 0.0,
    ) * _inv_dt_days  # gC/m2/day available to drain over the step
    # scale in [0, 1]; == 1 (no-op) whenever demand <= availability.
    lit_scale = jnp.where(
        lit_total_demand > lit_net_avail,
        lit_net_avail / jnp.maximum(lit_total_demand, 1e-30),
        1.0,
    )
    R_het_lit = lit_scale * R_het_lit_demand
    lit_to_som = lit_scale * lit_to_som_demand

    # Coarse woody debris: wood turnover does NOT humify 100 % into the
    # millennial SOM pool.  Split it like the litter pathway — a fraction
    # ``cwd_humification_eff`` becomes stable SOM, the rest respires to the
    # atmosphere (CWD heterotrophic respiration).  Routing all of wood_litter
    # to C_som (the previous behaviour) drove an unphysically large soil-carbon
    # stock (SOM_eq ~ input × 550-yr turnover).  Conserves exactly:
    # wood_litter = wood_to_som + R_het_cwd, so C_wood loses wood_litter,
    # C_som gains wood_to_som, and the atmosphere gains R_het_cwd (added to NEE).
    wood_to_som = config.cwd_humification_eff * wood_litter
    R_het_cwd = wood_litter - wood_to_som

    # --- Multi-pool SOM cascade (active -> slow -> passive) -----------------
    # SIGN CONVENTION: carbon INTO a pool is POSITIVE; every term below is walked
    # against it.  Each SOM pool decomposes at its base rate ``k_X`` scaled by the
    # shared modifier ``som_mod = f_temp * f_moist * f_freeze`` (temperature Q10 *
    # moisture * FREEZE suppression).  Of that decomposition ``D_X`` a humified
    # fraction transfers to the NEXT-SLOWER pool and the remainder respires to the
    # atmosphere; there is NO back-transfer (lower-triangular, forward cascade):
    #   active input  = lit_to_som + wood_to_som          (litter + CWD humif.)
    #   slow   input  = f_active_to_slow  * D_active       (active -> slow)
    #   passive input = f_slow_to_passive * D_slow         (slow -> passive)
    # ``_effective_rate`` is the EXACT exponential-decay loss fraction over dt, so
    # ``D_X * dt_days = C_X * (1 - (1 - som_mod*k_X)^dt_days) <= C_X``.  With every
    # input >= 0 this guarantees each pool stays >= 0 WITHOUT clipping (the
    # ``_soft_pos`` below is a defensive no-op for the SOM pools), so the SOM
    # sub-column conserves to machine precision:
    #   d(C_active + C_slow + C_passive) == (som_active_input - R_het_som) * dt.
    # Frozen/cold soils have a SMALL ``som_mod`` (freeze suppression) -> small
    # decomposition -> carbon accumulates, chiefly in the millennial passive pool
    # = the high-latitude / grassland SOC fix
    # (docs/land/multipool_som_phenology_plan.md).  ``soil_frozen_fraction``
    # (per-column annual frozen fraction, when supplied) additionally scales
    # ``som_mod`` by the perennial-frost/anaerobic protection ``f_perma`` <= 1
    # (permafrost carbon), so a PERENNIALLY-frozen column's effective rate drops
    # BELOW the aerobic ``som_freeze_floor`` and its deep permafrost SOM
    # accumulates -- carbon RETAINED (a smaller loss term), not created.
    som_mod = _som_decomp_modifier(
        T, precip, config, frozen_fraction=soil_frozen_fraction)
    D_active = state.C_som_active * _effective_rate(
        som_mod * config.tor_som_active, dt_days)    # gC/m2/day (active out)
    D_slow = state.C_som_slow * _effective_rate(
        som_mod * config.tor_som_slow, dt_days)      # gC/m2/day (slow out)
    D_passive = state.C_som_passive * _effective_rate(
        som_mod * config.tor_som_passive, dt_days)   # gC/m2/day (passive out)
    # Humification transfers to the next-slower pool (fraction of D_X, positive).
    som_active_to_slow = config.f_active_to_slow * D_active
    som_slow_to_passive = config.f_slow_to_passive * D_slow
    # Heterotrophic respiration = the decomposition NOT transferred onward.  The
    # three respiration terms sum to ``R_het_som`` (kept as this SUM so the
    # r_het == r_het_lit + r_het_som + r_het_cwd identity and NEE still hold).
    R_het_active = D_active - som_active_to_slow      # (1 - f_active_to_slow)*D_active
    R_het_slow = D_slow - som_slow_to_passive         # (1 - f_slow_to_passive)*D_slow
    R_het_passive = D_passive                         # passive fully respires
    R_het_som = R_het_active + R_het_slow + R_het_passive
    # Active-pool input: litter decomposition + humified coarse woody debris.
    som_active_input = lit_to_som + wood_to_som

    # --- Pool updates (Euler, gC/m2/day rates * dt_days) -------------------
    # Hard non-negativity ``jnp.maximum(x, 0)``.  The iter-64 cap
    # (``lab_net_avail`` etc., computed against the pool AFTER natural
    # turnover) guarantees ``state.C + (A − drain − deficit) · dt ≥ 0``
    # exactly, so no smoothing is required.  Previously a softplus
    # (``_alpha · logaddexp(x/_alpha, 0)`` with ``_alpha = 0.01``) was
    # used; even when ``raw_new_C == 0`` exactly, ``softplus(0) =
    # _alpha · log(2) ≈ 0.007 gC/m2`` of phantom carbon was added per
    # pool per step (codex iter-63 stop-time review).  ``jnp.maximum``
    # has subgradient 0 below 0 and 1 above — well-defined for both
    # forward simulation and AD when crossing zero is unphysical
    # anyway.
    def _soft_pos(x):
        return jnp.maximum(x, 0.0)

    new_state = CarbonState(
        C_lab=_soft_pos(
            state.C_lab + (A_lab - lab_release - lab_deficit_draw) * dt_days),
        C_fol=_soft_pos(
            state.C_fol + (A_fol + lab_release - leaf_litter
                           - fol_deficit_draw) * dt_days),
        C_root=_soft_pos(
            state.C_root + (A_root - root_litter
                            - root_deficit_draw) * dt_days),
        C_wood=_soft_pos(
            state.C_wood + (A_wood - wood_litter
                            - wood_deficit_draw) * dt_days),
        C_lit=_soft_pos(
            state.C_lit + (leaf_litter + root_litter
                           - R_het_lit - lit_to_som) * dt_days),
        # SOM cascade (phase A2, live).  Each pool gains its input and loses its
        # own total decomposition D_X; the slow/passive pools gain the humified
        # transfer from the pool above.  Sign walk (carbon in = +):
        #   active : + som_active_input (litter+CWD in), - D_active (out)
        #   slow   : + som_active_to_slow (from active),  - D_slow (out)
        #   passive: + som_slow_to_passive (from slow),   - D_passive (out)
        C_som_active=_soft_pos(
            state.C_som_active + (som_active_input - D_active) * dt_days),
        C_som_slow=_soft_pos(
            state.C_som_slow + (som_active_to_slow - D_slow) * dt_days),
        C_som_passive=_soft_pos(
            state.C_som_passive + (som_slow_to_passive - D_passive) * dt_days),
    )

    # --- NEE: positive = source to atmosphere ------------------------------
    # R_het_lit is the litter-availability-CAPPED respiration (finding #4);
    # ``unmet_npp_deficit_day`` removes the maintenance respiration that no
    # biomass pool could supply (finding #5); ``R_het_cwd`` is the coarse
    # woody-debris respiration split off from wood turnover.  With these
    # corrections the column closes exactly: sum(dC_pools) == -NEE_day*dt
    # (verified by test_carbon_*_conservation).
    NEE_day = (
        R_auto_day - unmet_npp_deficit_day
        + R_het_lit + R_het_som + R_het_cwd - gpp_day
    )
    co2_flux = NEE_day / _SPD * _GC_TO_KG_CO2

    if not return_diagnostics:
        return new_state, co2_flux

    diag = CarbonDiagnostics(
        gpp=gpp_day,
        npp=NPP_day,
        r_maint=R_maint_day,
        r_growth=R_growth_day,
        r_auto=R_auto_day,
        r_het_lit=R_het_lit,
        r_het_som=R_het_som,
        r_het_cwd=R_het_cwd,
        r_het=R_het_lit + R_het_som + R_het_cwd,
        nee=NEE_day,
        unmet_npp_deficit=unmet_npp_deficit_day,
        a_fol=A_fol,
        a_lab=A_lab,
        a_root=A_root,
        a_wood=A_wood,
        lab_release=lab_release,
        leaf_litter=leaf_litter,
        root_litter=root_litter,
        wood_litter=wood_litter,
        wood_to_som=wood_to_som,
        lit_to_som=lit_to_som,
        som_active_loss=D_active,
        som_slow_loss=D_slow,
        som_passive_loss=D_passive,
        lai=LAI,
    )
    return new_state, co2_flux, diag


# ===================================================================
# Seasonal prescribed cycle
# ===================================================================

def seasonal_co2_flux(
    doy: float,
    lat: jnp.ndarray,
    config: CarbonConfig,
) -> jnp.ndarray:
    """Prescribed repeating seasonal NEE cycle.

    NH: peak uptake at *nee_peak_day* (~July).  SH: opposite phase.
    Amplitude peaks at mid-latitudes (~45 deg), falls toward equator/poles.

    Returns
    -------
    co2_flux : kgCO2/m2/s, positive up (source to atmosphere).
    """
    phase = 2.0 * jnp.pi * (doy - config.nee_peak_day) / _SEASONAL_YEAR_DAYS

    # -cos peaks at phase=0  -->  uptake maximum at nee_peak_day
    seasonal = -jnp.cos(phase)

    # Flip sign for Southern Hemisphere
    seasonal = jnp.where(lat >= 0, seasonal, -seasonal)

    # Latitude weighting: sin(2|lat|) peaks at 45 deg
    lat_weight = jnp.sin(jnp.clip(2.0 * jnp.abs(lat), 0.0, jnp.pi))

    return config.nee_amplitude * seasonal * lat_weight


# ===================================================================
# Top-level dispatch
# ===================================================================

def step_carbon(
    carbon_state: CarbonState | None,
    sw_down: jnp.ndarray,
    T: jnp.ndarray,
    co2_ppmv: jnp.ndarray,
    beta: jnp.ndarray,
    lat: jnp.ndarray,
    doy: float,
    precip: jnp.ndarray,
    config: CarbonConfig,
    dt: float,
    gpp_override: jnp.ndarray | None = None,
    return_diagnostics: bool = False,
    soil_frozen_fraction: jnp.ndarray | None = None,
) -> tuple[CarbonState | None, jnp.ndarray] | tuple[
    CarbonState | None, jnp.ndarray, CarbonDiagnostics
]:
    """Dispatch carbon step based on *config.scheme*.

    Parameters
    ----------
    gpp_override : jnp.ndarray, optional
        When provided (e.g. from Farquhar photosynthesis), replaces the
        internal LUE-based GPP computation.
    return_diagnostics : bool, optional
        Return the CarbonDiagnostics breakdown as a third element.  Only
        the ``differland`` scheme produces diagnostics; requesting them for
        another scheme raises ``ValueError`` (there is no NPP/allocation
        breakdown for a prescribed or disabled carbon cycle).
    soil_frozen_fraction : jnp.ndarray, optional
        Per-column annual frozen fraction enabling the perennial-frost / anaerobic
        SOM protection in the ``differland`` step (see
        :func:`step_carbon_differland`).  ``None`` (default) -> no protection.

    Returns
    -------
    carbon_state : Updated state (None for "none"/"seasonal").
    co2_flux     : kgCO2/m2/s, positive up.
    diag         : (only when ``return_diagnostics``) CarbonDiagnostics.
    """
    if config.scheme == "differland":
        if carbon_state is None:
            raise ValueError("differland scheme requires a CarbonState")
        return step_carbon_differland(
            carbon_state, sw_down, T, co2_ppmv, beta, lat, doy,
            precip, config, dt, gpp_override=gpp_override,
            return_diagnostics=return_diagnostics,
            soil_frozen_fraction=soil_frozen_fraction,
        )
    if return_diagnostics:
        raise ValueError(
            f"return_diagnostics is only supported for the 'differland' "
            f"carbon scheme, not {config.scheme!r}."
        )
    if config.scheme == "seasonal":
        return carbon_state, seasonal_co2_flux(doy, lat, config)
    elif config.scheme == "none":
        return carbon_state, jnp.zeros_like(T)
    else:
        raise ValueError(
            f"Unknown carbon scheme {config.scheme!r}; expected one of "
            "'none', 'differland', 'seasonal'."
        )


# ===================================================================
# Initialization
# ===================================================================

def init_carbon_state(
    shape: tuple[int, ...],
    config: CarbonConfig,
) -> CarbonState:
    """Create initial carbon pool state with uniform values from config.

    An herbaceous config (``woody=False``) starts with NO wood pool: the flag
    disables both wood allocation AND the initial wood stock, so a grassland
    never carries a phantom tree that would otherwise respire, turn over, and
    feed CWD/SOM for decades from the ``C_wood_init`` default.  Woody configs
    use ``C_wood_init`` unchanged.
    """
    C_wood_init = config.C_wood_init if config.woody else 0.0
    # Partition ``C_som_init`` across the three SOM pools by CENTURY-like
    # steady-state stock fractions (active small, passive dominant; Parton et al.
    # 1987, Koven et al. 2013).  The fractions sum to 1, so ``som_total`` of the
    # returned state equals ``C_som_init`` exactly.  Seeding all three pools
    # non-zero also lets the semi-analytic slow-pool solve infer each pool's
    # turnover (its degenerate-column guard needs a non-zero loss flux).
    C_som = config.C_som_init
    return CarbonState(
        C_lab=jnp.full(shape, config.C_lab_init),
        C_fol=jnp.full(shape, config.C_fol_init),
        C_root=jnp.full(shape, config.C_root_init),
        C_wood=jnp.full(shape, C_wood_init),
        C_lit=jnp.full(shape, config.C_lit_init),
        C_som_active=jnp.full(shape, C_som * _SOM_INIT_FRAC_ACTIVE),
        C_som_slow=jnp.full(shape, C_som * _SOM_INIT_FRAC_SLOW),
        C_som_passive=jnp.full(shape, C_som * _SOM_INIT_FRAC_PASSIVE),
    )
