"""P model: optimality-based acclimated photosynthesis parameters (Stocker 2020).

Least-cost + coordination optimality (Prentice et al. 2014; Wang et al. 2017;
Stocker et al. 2020 GMD "rpmodel"; Mengoli et al. 2022 dual-timescale) supplying
ACCLIMATED PARAMETERS to the existing FvCB/Medlyn machinery — it does not
compute fluxes itself:

- leaf-top ``Vcmax25`` and the ratio ``rjv25 = Jmax25/Vcmax25`` for the canopy
  FvCB kernel (``canopy/photosynthesis.py``), and
- a predicted Medlyn slope ``g1 = xi`` for ``stomata.medlyn_gs``.

The instantaneous ("fast") temperature response stays in the FvCB kernel; this
module handles only the slow (acclimated) optimum, evaluated at whole-day
running means of the drivers (plain exponential moving averages, see
:func:`advance_pmodel_acclim`).  Everything is closed form — no ci iteration —
and smooth, so the whole path is differentiable.

Faithfulness
------------
Transcribed from Stocker et al. (2020) GMD 13, 1545-1581 and the rpmodel R
source (geco-bern/rpmodel, R/rpmodel.R + R/subroutines.R), pinned by
``tests/land/unit/test_p_model_faithful.py`` against an independent scalar
oracle.  Chi: rpmodel ``calc_optimal_chi`` (Stocker eq. 8-9); Jmax limitation:
``wang17`` method, ``mprime = mj*sqrt(1-(c*/mj)^(2/3))`` with c* = 0.41 (Wang
et al. 2017); ``Vcmax = phi0(T)*Iabs*mprime/mc``; ``Jmax = 4*phi0(T)*Iabs /
sqrt((1/fact_jmaxlim)^2 - 1)``, algebraically rearranged here to
``4*phi0*Iabs*sqrt((1-k)/k)`` with ``k = (c*/mj)^(2/3)`` (identical, since
``fact_jmaxlim = sqrt(1-k)``); ``phi0(T) = kphio*(0.352 + 0.022 Tc
- 3.4e-4 Tc^2)`` (Bernacchi 2003 fit, rpmodel ``ftemp_kphio``), kphio default
0.081785 (rpmodel's temperature-dependent, no-soil-moisture calibration).

DELIBERATE DEPARTURES from rpmodel, for internal consistency with the host
kernel (each documented where it happens):

1. **Leaf-level area basis.** rpmodel returns capacities per unit GROUND area
   using ``Iabs = fAPAR * PPFD``; here the output is the LEAF-TOP Vcmax25 the
   existing kn nitrogen-profile integral (``canopy/radiative_transfer.py``)
   expects, so ``Iabs`` is light per unit LEAF area.  Under the two-leaf
   scheme the caller passes the PAR a sunlit leaf ABSORBS,
   ``APAR_Sun / (fSun * LAI)`` from the scheme's own radiative transfer
   (``two_leaf_canopy.sunlit_leaf_apar``) — the same absorbed light the host
   kernel later runs the capacities with.  Lanes without that sunlit/shaded
   transfer (CLM-ML, simple SEB) still pass the incident top-of-canopy PPFD,
   which over-sizes the capacities by roughly the incident-to-absorbed ratio.
2. **Kinetics from the shared repo constants.** Kc/Ko/Gamma* mole fractions
   from ``leaf_biophysics`` (Bernacchi 2001, umol/mol), converted to Pa with
   the acclimated-mean surface pressure — not rpmodel's Pa-basis literals
   (~1% different for Kc).
3. **25 degC normalisation with the host kernel's Kattge & Knorr response**
   (``vcmax_temperature_response`` / ``jmax_temperature_response``), not
   rpmodel's constants, so the growth-T optimum survives the kernel's own
   temperature response on the way back up.
4. **Smooth mj floor.** Where rpmodel hard-zeroes capacities when
   ``mj <= c*`` (Jmax investment infeasible), we keep ``mj`` smoothly above
   ``c*`` (softplus floor) so capacities decay smoothly to ~0 with finite
   gradients (no NaN/dead adjoints).

References
----------
- Stocker et al. (2020) GMD 13, 1545-1581 (rpmodel).
- Wang et al. (2017) Nature Plants 3, 734-741 (least cost + Jmax limitation).
- Prentice et al. (2014) Ecol. Lett. 17, 82-91 (least-cost hypothesis).
- Mengoli et al. (2022) JAMES 14, e2021MS002767 (dual-timescale acclimation).
- Bernacchi et al. (2003) Plant Cell Environ. 26, 1419-1430 (phi0(T) fit).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.land.leaf_biophysics import (
    DIFFUSIVITY_RATIO_H2O_CO2,
    GAMMA_STAR25_UMOL_MOL,
    HA_GAMMA,
    HA_KC,
    HA_KO,
    KC25_UMOL_MOL,
    KO25_UMOL_MOL,
    O2_UMOL_MOL,
    arrhenius_factor,
    water_viscosity_ratio,
)
# NB: the Kattge & Knorr 25C-normalisation responses are imported at FUNCTION
# scope inside acclimated_capacities: a top-level import closes the cycle
# land.config -> p_model -> canopy.photosynthesis -> canopy.__init__ ->
# canopy.config -> p_model (CLAUDE.md: function-scope deferred imports for
# legitimate cross-module cycles).

__physics_contract__ = {
    "summary": (
        "P-model optimality (Stocker et al. 2020 rpmodel): closed-form "
        "least-cost chi = Gamma*/ca + (1-Gamma*/ca)*xi/(xi+sqrt(D)) with "
        "xi = sqrt(beta*(K+Gamma*)/(1.6*eta*)), Wang-2017 Jmax limitation, "
        "and coordination Vcmax = phi0(T)*Iabs*mj'/mc — evaluated at "
        "whole-day running means (plain EMA) of the drivers and "
        "normalised to 25 degC with the host kernel's Kattge & Knorr "
        "response. Supplies leaf-top Vcmax25, rjv25 = Jmax25/Vcmax25 and a "
        "predicted Medlyn slope g1 = xi to the existing FvCB/Medlyn path."
    ),
    "inputs": {
        "t_mean_K": "K", "vpd_mean_pa": "Pa", "co2_mean_ppm": "umol/mol",
        "ps_ema": "Pa", "iabs_mean": "umol/m^2/s", "dt": "s",
    },
    "outputs": {
        "vcmax25_leaf": "umol/m^2/s", "rjv25": "1", "g1_kpa": "kPa^0.5",
        "chi": "1",
    },
    "sign_convention": (
        "All capacities >= 0; chi in (0, 1); g1 > 0. Acclimation-state "
        "update is a convex combination (gain in [0, 1)), so means stay "
        "inside the historical driver range."
    ),
    "conserves": [],  # parameter prediction, not a flux
    "differentiable": True,
    "reference": (
        "Stocker et al. (2020) GMD 13 1545-1581; Wang et al. (2017) Nature "
        "Plants 3 734-741; Prentice et al. (2014) Ecol. Lett. 17 82-91; "
        "Mengoli et al. (2022) JAMES 14 e2021MS002767; Bernacchi et al. "
        "(2003) Plant Cell Environ. 26 1419-1430."
    ),
    "idealized_test": (
        "tests/land/unit/test_p_model.py: chi in (0,1) and decreasing in "
        "VPD; g1 ~ 2-6 kPa^0.5 at 25 degC / 1 kPa / 400 ppm / sea level; "
        "capacities -> 0 smoothly as Iabs -> 0 with finite rjv25 and finite "
        "gradients; rjv25 independent of kphio and Iabs; the running means "
        "converge to the 24-hour average and stay finite through polar "
        "night. tests/land/unit/"
        "test_p_model_faithful.py: rel-1e-9 scalar-oracle pin."
    ),
}

__param_spec__ = {
    "PModelConfig": {
        "scheme_key": "land.p_model",
        "excluded": {
            "vpd_min_pa": "numerics: smooth floor under sqrt(VPD)",
            "mj_floor_eps": "numerics: softplus floor offset above c*",
            "mj_floor_width": "numerics: softplus floor width (AD safety)",
            "init_ppfd": "initialisation constant, not a physical closure",
            "init_vpd_pa": "initialisation constant, not a physical closure",
            "init_co2_ppm": "initialisation constant, not a physical closure",
        },
        "params": {
            "beta_cost": {
                "units": "1", "bounds": (30.0, 500.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "photosynthesis",
                "reference": "Stocker et al. 2020 GMD sect. 2.4.1 (beta = 146)",
                "shape": None,
            },
            "kphio": {
                "units": "mol/mol", "bounds": (0.02, 0.15), "tunable_tier": 2,
                "transform": "sigmoid", "category": "photosynthesis",
                "reference": (
                    "Stocker et al. 2020 GMD Table 1 (0.081785, temperature-"
                    "dependent setup, no soil-moisture stress)"
                ),
                "shape": None,
            },
            "beta_cost_c4": {
                "units": "1", "bounds": (3.0, 60.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "photosynthesis",
                "reference": "rpmodel c4 beta = 146/9 (Cai & Prentice 2020)",
                "shape": None,
            },
            "kphio_c4": {
                "units": "mol/mol", "bounds": (0.2, 2.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "photosynthesis",
                "reference": (
                    "rpmodel c4 base kphio = 1.0 (the Cai & Prentice 2020 "
                    "temperature quadratic carries the magnitude)"
                ),
                "shape": None,
            },
            "tau_acclim_s": {
                "units": "s", "bounds": (86400.0, 10368000.0),
                "tunable_tier": 2, "transform": "sigmoid",
                "category": "photosynthesis",
                "reference": "Mengoli et al. 2022 (~15 d acclimation memory)",
                "shape": None,
            },
        },
    },
}

# --- Wang et al. (2017) Jmax-limitation cost (rpmodel wang17) ---
C_STAR = 0.41  # [-] unit cost of Jmax maintenance; k = (c*/mj)^(2/3)

# --- Bernacchi (2003) quantum-yield temperature fit (rpmodel ftemp_kphio, C3) ---
# phi0(T)/kphio = a + b*Tc + c*Tc^2, ascending powers of leaf T [degC].
_KPHIO_QUAD_C3 = (0.352, 0.022, -3.4e-4)

# --- C4 quantum-yield temperature fit (rpmodel ftemp_kphio, c4 branch) ---
# "Cai & Prentice (2020), corrected by David Orme" (rpmodel source comment);
# the C4 base kphio is 1.0 in rpmodel — this quadratic carries the magnitude
# (peaks ~0.42 near 32 degC), floored at 0 like the C3 fit.
_KPHIO_QUAD_C4 = (-0.064, 0.03, -0.000464)
# rpmodel evaluates the C4 quadratic at a FIXED 15 degC when the temperature
# dependence is switched off (do_ftemp_kphio = FALSE branch).
_KPHIO_C4_REF_TC = 15.0

# --- numerics (smooth guards) ---
_PHI0_FLOOR_WIDTH = 0.01  # coeff-ok: softplus width keeping phi0 >= 0 smooth

# --- LUNA nitrogen use efficiencies (CTSM LunaMod.F90:76-77, verbatim) ---
# Diagnostic-only (PR5): the model carries NO nitrogen cycle; these convert
# C3 capacities into implied N pools for output tapes and NOTHING reads them
# back (no-feedback guarantee).  CTSM pins the ROUNDED values (comments there
# give the products 6.22*47.3 = 294.206 and 8.06*156 = 1257.36; the code uses
# 294.2 / 1257.0) — we copy the code values exactly.  C3 ONLY: the constants
# do not apply to the C4 pathway.  Provenance is secondhand in CTSM (Rogers
# 2014; Coste 2005 via Xu et al 2012) and 47.3 umol/gRubisco/s sits at the
# high end of specific-activity estimates — carry ~+/-50% structural
# uncertainty when interpreting the pools.
_F_C25_UMOL_GN_S = 294.2   # umol CO2 /s /gN in Rubisco (= 6.22 gRub/gN * 47.3)
_F_J25_UMOL_GN_S = 1257.0  # umol e- /s /gN in electron transport (= 8.06 * 156)


class PModelConfig(NamedTuple):
    """Configuration for the P-model optimality parameter source.

    ``beta_cost``/``kphio``/``tau_acclim_s`` are the tunables (see
    ``__param_spec__``); the rest are numeric guards and recorded
    initialisation constants (excluded from calibration).
    """

    # --- optimality (Stocker et al. 2020 GMD) ---
    beta_cost: float = 146.0     # [-] carboxylation:transpiration cost ratio
    kphio: float = 0.081785      # [mol/mol] apparent quantum yield (rpmodel default)
    kphio_temp: bool = True      # Bernacchi 2003 phi0(T) quadratic on (compile-time)
    # --- C4 optimality (rpmodel c4; Cai & Prentice 2020) ---
    beta_cost_c4: float = 146.0 / 9.0  # [-] C4 cost ratio (rpmodel: beta/9)
    kphio_c4: float = 1.0        # [mol/mol] C4 base quantum yield (rpmodel 1.0)
    # --- acclimation (Mengoli et al. 2022) ---
    tau_acclim_s: float = 1296000.0  # [s] = 15 days
    # --- numerics (smooth guards; excluded from calibration) ---
    vpd_min_pa: float = 50.0     # [Pa] smooth floor under sqrt(D)
    mj_floor_eps: float = 0.01   # [-] mj kept above C_STAR + eps
    mj_floor_width: float = 0.01  # [-] softplus width of that floor
    # --- initialisation constants (recorded here, no hidden choice) ---
    init_ppfd: float = 400.0     # [umol/m^2/s] light-mean cold-start
    init_vpd_pa: float = 800.0   # [Pa] VPD-mean cold-start
    init_co2_ppm: float = 400.0  # [umol/mol] CO2 cold-start (driver may override)


# Valid selector values for the CanopyConfig switches wired in
# surface_scheme/two_leaf_canopy.py; imported by canopy/config.py.
# capacity_scheme: "prescribed" (table) | "p_model" (least-cost acclimated
# optimum) | "phydro" (P-hydro acclimated optimum, two-leaf canopy only).
VALID_CAPACITY_SCHEMES = ("prescribed", "p_model", "phydro")
OPTIMALITY_CAPACITY_SCHEMES = ("p_model", "phydro")
VALID_G1_SOURCES = ("table", "p_model")


class PModelAcclimState(NamedTuple):
    """Whole-day running-mean acclimation drivers, all shape ``(ncol,)``.

    Every field is a plain exponential moving average over day AND night
    with e-folding time ``PModelConfig.tau_acclim_s``
    (:func:`advance_pmodel_acclim`); ``iabs_mean`` is the mean of the light
    the caller passes (sunlit-leaf absorbed PAR under the two-leaf scheme),
    night zeros included.
    """

    iabs_mean: jax.Array
    t_mean_K: jax.Array
    vpd_mean_pa: jax.Array
    co2_mean_ppm: jax.Array
    ps_ema: jax.Array
    # Acclimated soil state seen by the roots (P-hydro only; ``None`` when
    # transpiration_stress != 'phydro'): the same running mean of the
    # conductance-weighted soil water potential [MPa] and of the soil-to-root
    # resistance per leaf area [MPa m2 s/mol] (land/phydro.py PhydroSupply).
    psi_s_mean_mpa: jax.Array | None = None
    rsoil_mean: jax.Array | None = None


class PModelCapacities(NamedTuple):
    """Acclimated parameter predictions (leaf-top; diagnostics included)."""

    vcmax25_leaf: jax.Array  # [umol/m^2/s] leaf-top Vcmax25
    rjv25: jax.Array         # [-] Jmax25 / Vcmax25
    g1_kpa: jax.Array        # [kPa^0.5] predicted Medlyn slope (= xi)
    jmaxlim: jax.Array       # [-] Jmax limitation factor L = sqrt(1 - (c*/mj)^(2/3))
    chi: jax.Array           # [-] ci/ca at the acclimated optimum (diagnostic)
    xi_sqrt_pa: jax.Array    # [Pa^0.5] least-cost xi (diagnostic)


def pmodel_switches_active(surface_scheme=None, stomata=None) -> bool:
    """True iff any P-model switch is selected on the given config objects.

    Reads ``capacity_scheme``/``g1_source`` where present (canopy configs) and
    the same fields on a big-leaf ``StomataConfig``.  Shared by every lane's
    state-initialisation gate and driver guard so "is the P model on?" has one
    definition.
    """
    for obj in (surface_scheme, stomata):
        if obj is None:
            continue
        if getattr(obj, "capacity_scheme", "prescribed") in OPTIMALITY_CAPACITY_SCHEMES:
            return True
        if getattr(obj, "g1_source", "table") == "p_model":
            return True
    return False


def _smooth_floor(x: jax.Array, floor: jax.Array, width: jax.Array) -> jax.Array:
    """Smooth (softplus) lower bound: ~x for x >> floor, -> floor from above.

    C-infinity everywhere (no `where`/`max` kink), so adjoints stay finite
    as x crosses the floor.
    """
    return floor + width * jax.nn.softplus((x - floor) / width)


def optimal_chi(
    T_K: jax.Array,
    vpd_pa: jax.Array,
    co2_ppm: jax.Array,
    ps_pa: jax.Array,
    cfg: PModelConfig,
    beta: jax.Array | float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Least-cost optimal chi (Stocker 2020 eq. 8-9), all pressures in Pa.

    Returns ``(chi, xi, ci_pa, gamma_star_pa, K_pa)``.  Kinetic mole
    fractions come from the shared ``leaf_biophysics`` Bernacchi constants,
    converted to partial pressures with ``ps_pa`` (departure 2 in the module
    docstring).
    """
    frac = 1e-6  # exact conversion umol/mol -> mol/mol
    gamma_star = GAMMA_STAR25_UMOL_MOL * frac * ps_pa * arrhenius_factor(T_K, HA_GAMMA)
    kc = KC25_UMOL_MOL * frac * ps_pa * arrhenius_factor(T_K, HA_KC)
    ko = KO25_UMOL_MOL * frac * ps_pa * arrhenius_factor(T_K, HA_KO)
    p_o2 = O2_UMOL_MOL * frac * ps_pa
    big_k = kc * (1.0 + p_o2 / ko)  # Michaelis coefficient K [Pa]

    eta_star = water_viscosity_ratio(T_K, ps_pa)
    _beta = cfg.beta_cost if beta is None else beta
    xi = jnp.sqrt(
        _beta * (big_k + gamma_star) / (DIFFUSIVITY_RATIO_H2O_CO2 * eta_star)
    )  # [Pa^0.5]

    vpd = _smooth_floor(vpd_pa, cfg.vpd_min_pa, cfg.vpd_min_pa)
    ca = co2_ppm * frac * ps_pa  # [Pa]
    g_over_ca = gamma_star / ca
    chi = g_over_ca + (1.0 - g_over_ca) * xi / (xi + jnp.sqrt(vpd))
    ci = chi * ca
    return chi, xi, ci, gamma_star, big_k


def _phi0(T_K: jax.Array, cfg: PModelConfig) -> jax.Array:
    """Apparent quantum yield phi0(T) [mol/mol] (rpmodel ftemp_kphio, C3).

    The Bernacchi quadratic goes negative below ~ -14 degC; a smooth floor at
    0 keeps deep-cold acclimated capacities at ~0 with finite gradients.
    """
    if not cfg.kphio_temp:
        return jnp.asarray(cfg.kphio) * jnp.ones_like(T_K)
    tc = T_K - constants.T_freeze
    a0, a1, a2 = _KPHIO_QUAD_C3
    quad = a0 + a1 * tc + a2 * tc * tc
    quad = _PHI0_FLOOR_WIDTH * jax.nn.softplus(quad / _PHI0_FLOOR_WIDTH)
    return cfg.kphio * quad


def acclimated_capacities(
    acclim: PModelAcclimState | None, cfg: PModelConfig
) -> PModelCapacities:
    """Acclimated leaf-top Vcmax25, rjv25 and predicted Medlyn g1.

    Coordination (Stocker eq. 16 / rpmodel wang17): at the acclimated-mean
    drivers, ``Vcmax_growth = phi0(Tg)*Iabs*mj'/mc`` and ``Jmax_growth =
    4*phi0(Tg)*Iabs*sqrt((1-k)/k)``; both are normalised to 25 degC with the
    host kernel's own Kattge & Knorr responses evaluated at the same growth
    temperature.  ``mj'/mc`` is computed via the exact algebraic ratio
    ``mj/mc = (ci + K)/(ci + 2 Gamma*)`` so the ci -> Gamma* cancellation is
    preserved analytically (no 0/0), and ``rjv25`` is likewise formed from
    the analytic ``Jmax/Vcmax`` ratio in which phi0 and Iabs cancel — so
    rjv25 stays finite as Iabs -> 0.
    """
    if acclim is None:
        raise ValueError(
            "P-model capacities requested but no acclimation state was "
            "supplied (pmodel_acclim=None): this land lane does not carry "
            "PModelAcclimState. Initialise it (init_pmodel_acclim) or select "
            "capacity_scheme='prescribed' / g1_source='table'."
        )
    tg_k = acclim.t_mean_K
    chi, xi, ci, gamma_star, big_k = optimal_chi(
        tg_k, acclim.vpd_mean_pa, acclim.co2_mean_ppm, acclim.ps_ema, cfg
    )
    vcmax25, rjv25, jmaxlim = capacities_from_chi(
        chi=chi, ci_pa=ci, gamma_star_pa=gamma_star, big_k_pa=big_k,
        tg_k=tg_k, iabs=acclim.iabs_mean, phi0=_phi0(tg_k, cfg), cfg=cfg)

    # Medlyn slope: matching ci/ca = g1/(g1 + sqrt(D)) to the xi term of
    # Stocker eq. 8 (Gamma*/ca term neglected — standard) gives g1 = xi;
    # sqrt(1000) converts Pa^0.5 -> kPa^0.5 (medlyn_gs takes VPD in kPa).
    g1_kpa = xi / jnp.sqrt(1000.0)
    return PModelCapacities(
        vcmax25_leaf=vcmax25, rjv25=rjv25, g1_kpa=g1_kpa, jmaxlim=jmaxlim,
        chi=chi, xi_sqrt_pa=xi
    )


def stressed_rjv25_factor(jmaxlim: jax.Array, stress: jax.Array) -> jax.Array:
    """Factor on Jmax25/Vcmax25 when an empirical stress scales Vcmax.

    rpmodel's structure (``rpmodel.R``): the soil-moisture stress ``s``
    multiplies Vcmax, and Jmax is then re-derived from the stressed Vcmax
    through the Jmax limitation ``L = s * L0`` on the Smith light response,
    ``Jmax = 4 phi0 Iabs / sqrt(1/L^2 - 1)``.  Jmax therefore falls MORE than
    Vcmax: ``Jmax_s / Jmax_0 = s * sqrt((1 - L0^2) / (1 - s^2 L0^2))``, and the
    ratio Jmax/Vcmax is multiplied by the square root returned here (1 at
    s = 1, ``sqrt(1 - L0^2)`` as s -> 0).
    """
    l0_sq = jmaxlim * jmaxlim
    return jnp.sqrt((1.0 - l0_sq) / (1.0 - stress * stress * l0_sq))


def capacities_from_chi(
    *,
    chi: jax.Array,
    ci_pa: jax.Array,
    gamma_star_pa: jax.Array,
    big_k_pa: jax.Array,
    tg_k: jax.Array,
    iabs: jax.Array,
    phi0: jax.Array,
    cfg: PModelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Coordination capacities (Vcmax25, rjv25, L) at a GIVEN optimal chi.

    The chi -> capacity algebra shared by the least-cost optimum
    (:func:`acclimated_capacities`) and the P-hydro profit optimum
    (``land/phydro.py``): Wang-2017 limitation on the smooth-floored mj,
    ``Vcmax_growth = phi0*Iabs*mj'/mc`` via the analytic ``mj/mc`` ratio,
    both normalised to 25 degC with the host Kattge & Knorr responses.
    """
    from legoesm.land.canopy.photosynthesis import (
        jmax_temperature_response,
        vcmax_temperature_response,
    )

    del chi  # capacity algebra runs on ci/Gamma*/K directly
    tg_c = tg_k - constants.T_freeze
    mj = (ci_pa - gamma_star_pa) / (ci_pa + 2.0 * gamma_star_pa)
    mj_safe = _smooth_floor(mj, C_STAR + cfg.mj_floor_eps, cfg.mj_floor_width)
    k = (C_STAR / mj_safe) ** (2.0 / 3.0)  # in (0, 1) by the floor
    sqrt_1mk = jnp.sqrt(1.0 - k)

    # mj'/mc = sqrt(1-k) * (mj/mc), with mj/mc = (ci + K)/(ci + 2 Gamma*).
    mj_over_mc = (ci_pa + big_k_pa) / (ci_pa + 2.0 * gamma_star_pa)

    f_v = vcmax_temperature_response(tg_k, tg_c)
    f_j = jmax_temperature_response(tg_k, tg_c)

    vcmax25 = phi0 * iabs * sqrt_1mk * mj_over_mc / f_v
    # Jmax_growth / Vcmax_growth = 4*sqrt((1-k)/k) / (sqrt(1-k)*mj/mc)
    #                            = 4 / (sqrt(k) * mj/mc)   (phi0, Iabs cancel)
    rjv25 = (4.0 / (jnp.sqrt(k) * mj_over_mc)) * (f_v / f_j)
    return vcmax25, rjv25, sqrt_1mk


def init_pmodel_acclim(
    ncol: int,
    *,
    t_init_K: jax.Array | float,
    ps_init_pa: jax.Array | float,
    cfg: PModelConfig,
    co2_init_ppm: jax.Array | float | None = None,
    dtype: Any = jnp.float32,
    soil_init: tuple[float, float] | None = None,
) -> PModelAcclimState:
    """Cold-start acclimation state from the recorded config constants.

    ``t_init_K``/``ps_init_pa`` come from the caller's initial condition;
    PPFD/VPD (and CO2 unless overridden with the run's value) come from the
    ``PModelConfig`` init fields so every cold-start choice is in the run's
    resolved config.  ``soil_init = (psi_s_mpa, rsoil_leaf)`` (the
    ``PHydroConfig`` init fields) switches the P-hydro soil-state means on;
    ``None`` leaves them absent.
    """
    full = lambda v: jnp.full((ncol,), v, dtype=dtype)  # noqa: E731
    co2 = cfg.init_co2_ppm if co2_init_ppm is None else co2_init_ppm
    return PModelAcclimState(
        iabs_mean=full(cfg.init_ppfd),
        t_mean_K=full(0.0) + jnp.asarray(t_init_K, dtype=dtype),
        vpd_mean_pa=full(cfg.init_vpd_pa),
        co2_mean_ppm=full(0.0) + jnp.asarray(co2, dtype=dtype),
        ps_ema=full(0.0) + jnp.asarray(ps_init_pa, dtype=dtype),
        psi_s_mean_mpa=None if soil_init is None else full(soil_init[0]),
        rsoil_mean=None if soil_init is None else full(soil_init[1]),
    )


class PModelCapacitiesC4(NamedTuple):
    """C4 acclimated parameter predictions (leaf-top; rpmodel c4 method)."""

    vcmax25_c4_leaf: jax.Array  # [umol/m^2/s] leaf-top C4 Vcmax25
    g1_c4_kpa: jax.Array        # [kPa^0.5] predicted C4 Medlyn slope (= xi_c4)
    chi_c4: jax.Array           # [-] C4 ci/ca at the optimum (diagnostic)
    xi_c4_sqrt_pa: jax.Array    # [Pa^0.5] C4 least-cost xi (diagnostic)


def _phi0_c4(T_K: jax.Array, cfg: PModelConfig) -> jax.Array:
    """C4 apparent quantum yield [mol/mol] (rpmodel ftemp_kphio, c4 branch).

    ``kphio_c4 * max(quad_c4(Tc), 0)`` with the smooth softplus floor; when
    the temperature dependence is off, rpmodel freezes the quadratic at
    15 degC rather than dropping it (the magnitude lives in the quadratic).
    """
    a0, a1, a2 = _KPHIO_QUAD_C4
    if not cfg.kphio_temp:
        tc = jnp.full_like(T_K, _KPHIO_C4_REF_TC)
    else:
        tc = T_K - constants.T_freeze
    quad = a0 + a1 * tc + a2 * tc * tc
    quad = _PHI0_FLOOR_WIDTH * jax.nn.softplus(quad / _PHI0_FLOOR_WIDTH)
    return cfg.kphio_c4 * quad


def acclimated_capacities_c4(
    acclim: PModelAcclimState | None, cfg: PModelConfig
) -> PModelCapacitiesC4:
    """C4 acclimated leaf-top Vcmax25 and predicted Medlyn g1 (rpmodel c4).

    rpmodel's c4 method verbatim: chi/xi use the FULL kinetics (real Gamma*
    and K) with the C4 cost ratio ``beta_cost_c4`` (= 146/9); the CO2-
    saturated bundle sheath sets mj = mc = 1, so the Wang-2017 limitation
    factor is the constant ``sqrt(1 - c*^(2/3))`` and
    ``Vcmax_growth = phi0_c4(Tg) * Iabs * sqrt(1 - c*^(2/3))``.  The 25 degC
    inversion divides by the HOST Collatz kernel's RAW temperature factor
    (``c4_vcmax_temperature_response``, which is ~0.87 at 25 degC by the
    kernel's own convention), so the kernel reproduces the optimum at the
    growth temperature.  The returned value is therefore the EFFECTIVE
    25 degC base the kernel multiplies — not a literal 25 degC rate (GLM
    review) — exactly the semantics of the prescribed ``Vcmax25_C4`` it
    replaces.  No Jmax ratio: the Collatz kernel has none.
    """
    if acclim is None:
        raise ValueError(
            "P-model C4 capacities requested but no acclimation state was "
            "supplied (pmodel_acclim=None): this land lane does not carry "
            "PModelAcclimState. Initialise it (init_pmodel_acclim) or select "
            "capacity_scheme='prescribed' / g1_source='table'."
        )
    from legoesm.land.canopy.photosynthesis import (
        c4_vcmax_temperature_response,
    )

    tg_k = acclim.t_mean_K
    chi, xi, _ci, _gs, _K = optimal_chi(
        tg_k, acclim.vpd_mean_pa, acclim.co2_mean_ppm, acclim.ps_ema, cfg,
        beta=cfg.beta_cost_c4)
    mprime_c4 = jnp.sqrt(1.0 - C_STAR ** (2.0 / 3.0))  # mj = 1 (CO2-saturated)
    v_growth = _phi0_c4(tg_k, cfg) * acclim.iabs_mean * mprime_c4
    vcmax25_c4 = v_growth / c4_vcmax_temperature_response(tg_k)
    g1_c4 = xi / jnp.sqrt(1000.0)
    return PModelCapacitiesC4(
        vcmax25_c4_leaf=vcmax25_c4, g1_c4_kpa=g1_c4, chi_c4=chi,
        xi_c4_sqrt_pa=xi)


def nitrogen_diagnostics(vcmax25_leaf: jax.Array,
                         jmax25_leaf: jax.Array):
    """Implied C3 photosynthetic nitrogen pools, per m2 LEAF (diagnostic only).

    Returns ``(n_rubisco, n_et)`` in g N m-2 leaf: nitrogen bound in Rubisco
    implied by ``vcmax25_leaf`` and in the electron-transport chain implied by
    ``jmax25_leaf`` (both leaf-top acclimated capacities, umol m-2 leaf s-1),
    via the LUNA nitrogen use efficiencies (see the constants block above).
    Deliberately NO total: light-capture N needs chlorophyll-allocation
    assumptions this model does not carry, and a sum omitting it would read
    as leaf N (GLM design review).  C3 only — the constants do not apply to
    C4.  Nothing in the model reads these values back; they exist for tapes
    and for a future N closure to compare against.
    """
    return (vcmax25_leaf / _F_C25_UMOL_GN_S,
            jmax25_leaf / _F_J25_UMOL_GN_S)


def acclim_trajectory(
    init: "PModelAcclimState",
    *,
    T_K: jax.Array,
    ppfd: jax.Array,
    vpd_pa: jax.Array,
    co2_ppm: jax.Array,
    ps_pa: jax.Array,
    cfg: "PModelConfig",
    dt: float,
) -> "PModelAcclimState":
    """Scan the acclimation update over a driver time series.

    Drivers have a leading time axis ((n_time, ...) per leaf); returns the
    PER-STEP state trajectory (leaves (n_time, ...)) so a diagnostic
    (state-free) driver can evaluate the acclimated capacities at every
    timestep with the causal running mean up to that step (offline pre-pass;
    e.g. run_ec_site --mode diagnostic).
    """
    def _f(carry, x):
        t_k, w, d_pa, ca, ps = x
        new = advance_pmodel_acclim(
            carry, T_K=t_k, ppfd=w, vpd_pa=d_pa, co2_ppm=ca, ps_pa=ps,
            cfg=cfg, dt=dt)
        return new, new

    # Promote the carry to the drivers' dtype up front: an f32 init state
    # (e.g. built from float32 NetCDF driver values) against f64 forcing
    # under JAX_ENABLE_X64 makes the scan carry change dtype across the
    # body, which lax.scan rejects.
    _dt = jnp.result_type(T_K, ppfd, vpd_pa, co2_ppm, ps_pa)
    init = jax.tree_util.tree_map(lambda x: x.astype(_dt), init)
    _, traj = jax.lax.scan(_f, init, (T_K, ppfd, vpd_pa, co2_ppm, ps_pa))
    return traj


def advance_pmodel_acclim(
    acclim: PModelAcclimState,
    *,
    T_K: jax.Array,
    ppfd: jax.Array,
    vpd_pa: jax.Array,
    co2_ppm: jax.Array,
    ps_pa: jax.Array,
    cfg: PModelConfig,
    dt: float,
    psi_s_mpa: jax.Array | None = None,
    rsoil_leaf: jax.Array | None = None,
) -> PModelAcclimState:
    """One acclimation step: whole-day running means of every driver.

    A plain exponential moving average with e-folding time ``tau_acclim_s``
    — every timestep, day and night, carries the same weight, so the means
    are the 24-hour (multi-day) averages the optimality literature sizes
    capacities for (daily / weekly / monthly means; Stocker et al. 2020).
    The light mean therefore includes the zero night values and decays
    toward zero through a polar night (capacities -> 0, finite).  The
    P-hydro soil-state means advance with the same gain when the step's
    ``psi_s_mpa``/``rsoil_leaf`` are given and the state carries them.
    """
    gain = 1.0 - jnp.exp(-dt / cfg.tau_acclim_s)
    step = lambda mean, x: mean + gain * (x - mean)  # noqa: E731
    soil = {}
    if acclim.psi_s_mean_mpa is not None:
        if psi_s_mpa is None or rsoil_leaf is None:
            raise ValueError(
                "advance_pmodel_acclim: the acclimation state carries the "
                "P-hydro soil-state means but this step supplied no "
                "psi_s_mpa / rsoil_leaf; they would silently freeze.")
        soil = dict(psi_s_mean_mpa=step(acclim.psi_s_mean_mpa, psi_s_mpa),
                    rsoil_mean=step(acclim.rsoil_mean, rsoil_leaf))
    return PModelAcclimState(
        iabs_mean=step(acclim.iabs_mean, ppfd),
        t_mean_K=step(acclim.t_mean_K, T_K),
        vpd_mean_pa=step(acclim.vpd_mean_pa, vpd_pa),
        co2_mean_ppm=step(acclim.co2_mean_ppm, co2_ppm),
        ps_ema=step(acclim.ps_ema, ps_pa),
        **soil,
    )
