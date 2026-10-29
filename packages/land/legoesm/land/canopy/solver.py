"""Newton-Raphson canopy closure solver.

Solves the 6-variable canopy energy balance system:
  x = [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c]
or, under ``stomatal_model="phydro"``, the 8-variable system with the
soil-to-leaf draw-down of each leaf class appended:
  x = [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c, dpsi_Sun, dpsi_Sh]
whose two extra residuals are the stationarity of the instantaneous
P-hydro profit (``land/phydro.py::instantaneous_foc``); the hydraulic
supply at that draw-down IS the leaf's stomatal conductance.

The soil skin temperature ``Ts`` is **not** a Newton variable of this
solver — it is supplied by the caller as ``bundle.Ts_bc`` and kept fixed
during the inner Newton.  Stability of the canopy↔soil coupling is
provided by the **outer Picard iteration** in ``canopy_land.py``: the
caller runs [canopy closure → thermal solve → update Ts_bc] for a
fixed number of passes per timestep, which converges the coupled
fast-turbulent / slow-conductive system without the feedback-induced
oscillation that a single explicit pass produces.

After convergence ``G = Rn_soil − LE_soil − H_soil`` is passed to
``solve_soil_thermal`` as the top BC exactly as in
``multilayer_land.py``.

Newton step control (CLM5-inspired, section 2.5.3.2):
  * per-component absolute caps on each iteration (constant, not
    decaying) — prevents dusk overshoot that a scalar clamp cannot
  * per-component convergence check on the raw (unclipped) Newton delta
  * wide ``Ci`` cap so the Ball-Berry / Farquhar inner fixed point is
    resolved in 1–2 iterations and does not dominate the outer loop
  * sunlit-leaf degeneracy anchor when ``fSun < 0.05``

Uses jax.lax.scan for JIT-compatible fixed-point iteration with a
forward-mode autodiff Jacobian (jax.jacfwd).  The LE_module is captured
as a static Python string in a functools.partial closure — it is never
traced.

**Coupling scheme**: only the DifferBESS FULLY_COUPLED formulation is
implemented — leaves and soil share the same canopy air space (Tc, q_c)
with a clumping-index-weighted below-canopy resistance (see
``compute_below_canopy_resistance``).  The ``VEG_ONLY`` and
``LEAVES_ATMO`` variants from DifferBESS were removed to keep the
Newton residual shape minimal; they can be added back later if a
multi-scheme comparison study needs them.

Source: adapted from DifferBESS/algo/newton_root.py;
        coupling architecture rewritten for legoESM's multilayer soil
        thermal solver via outer Picard iteration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

from legoesm.land.canopy.config import CanopyConfig, VALID_LE_MODULES
from legoesm.land.canopy.radiative_transfer import canopy_longwave_rt
from legoesm.land.canopy.photosynthesis import photosynthesis
from legoesm.land.phydro import LeafHydraulics, floored_vpd, instantaneous_foc
from legoesm.land.leaf_biophysics import DIFFUSIVITY_RATIO_H2O_CO2
from legoesm.land.canopy.stability import (
    monin_obukhov_stability,
    compute_boundary_layer_resistance,
    compute_below_canopy_resistance,
    sat_specific_humidity,
)
from legoesm.land.canopy.energy_balance import (
    canopy_met_variables,
    gs_mol_per_ms,
    saturation_specific_humidity,
    leaf_energy_balance_bt,
    leaf_energy_balance_pm,
    soil_energy_balance_bt,
    soil_energy_balance_pm,
    canopy_air_update,
)


# --- Sunlit-leaf degeneracy anchor smoother (numerics; see solver notes) ---
# tanh blend Tf_Sun -> Tf_Sh as the sunlit fraction shrinks; centred at
# fSun = 0.08 with transition half-width 0.03 (differentiable replacement for
# the old hard ``where(fSun < 0.05, ...)`` kink).
_ANCHOR_FSUN_CENTER = 0.08   # [-] tanh centre in sunlit fraction
_ANCHOR_FSUN_WIDTH  = 0.03   # [-] tanh transition half-width

# --- Damped-least-squares canopy solve (numerics; never tunable) ------------
# WHY NOT PLAIN NEWTON.  A small share of columns -- hot, bright and CALM --
# never converged at any iteration count: the unsolved share was flat at 0.66%
# from 25 iterations through 50.  They do not stall, they OSCILLATE, moving
# ~3.9 K per iteration forever while solved columns move exactly 0.0.  In calm
# air the aerodynamic conductance collapses and one direction of the Jacobian
# goes nearly flat; Newton divides by that tiny slope, the step overshoots the
# root, and a FIXED step clamp of the same magnitude as the overshoot turns the
# overshoot into a stable period-2 cycle.  The root sits BETWEEN consecutive
# iterates and the iteration can never land on it.
#
# Shrinking the clamp cannot fix that: the bulk of columns legitimately need
# steps of order 1 K early on, and a clamp of 1.0 left 80-92% of ALL columns
# unconverged.  One constant cannot serve both regimes.
#
# The cure is damping that ADAPTS: the step is the solution of a least-squares
# problem whose damping rises whenever a step fails to reduce the residual and
# falls when it succeeds, so ordinary columns never feel it and a cycling column
# is damped until it lands inside the bracket.  Steps that fail are REJECTED
# rather than taken, which is what makes the residual monotone.
#
# The damped system is solved in AUGMENTED form via QR, never through the normal
# equations: this Jacobian's condition number is ~1e5, and squaring it to 1e10
# exceeds single precision, which is what this path runs in -- that would trade
# an oscillation for numerical noise.
_LM_LAMBDA_0 = 1.0e-2       # [-] initial damping, relative to the scaled system
_LM_LAMBDA_MAX = 1.0e10     # [-] damping at which a column is declared unsolved
_LM_LAMBDA_UP = 4.0         # [-] damping growth after a rejected step
_LM_LAMBDA_DOWN = 3.0       # [-] damping decay after a good step
_LM_RHO_BAD, _LM_RHO_GOOD = 0.25, 0.75   # [-] Nielsen gain-ratio thresholds
_LM_RTOL, _LM_ATOL = 1.0e-8, 1.0e-12     # [-] on the SQUARED residual norm

# Characteristic increment of each unknown, in its own units:
#   [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c] = [K, K, umol/mol, umol/mol, K, kg/kg]
# The solver previously CLAMPED every component to the same +/-5, which is 5 K on
# a temperature and 5 umol/mol on a CO2 concentration -- both sensible -- but
# FIVE HUNDRED TIMES its own magnitude on a specific humidity of ~0.01 kg/kg,
# i.e. no constraint at all on the one variable that most needed one.  These
# values are that characteristic magnitude per unknown; solving in them
# (z = x/xscale) makes the damping act evenly across components.
# NOTE: this is a SCALE, not the old hard clip -- there is no longer a fixed
# +/-5 cap on the step.  The step is instead bounded by the adaptive damping and
# the accept-only-if-the-residual-drops rule; a component's move is
# xscale[i]*dz[i], which the damping shrinks whenever a trial is rejected.
_LM_XSCALE = jnp.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0e-3])
# The two P-hydro draw-down unknowns [MPa] appended under stomatal_model="phydro".
_LM_XSCALE_DPSI = 0.5
# Residual weight of the two draw-down rows.  Left at 1: the stationarity
# residual is already O(10-100) umol/m2/s/MPa at the start of a daytime solve,
# on a par with the kelvin rows; a larger weight lets those rows dominate the
# RELATIVE convergence test (n_sq <= rtol * n_sq_0) and loosens every other
# row (measured: LE/H skill fell at two sites with 1e2).
_FOC_ROW_SCALE = 1.0
_ANCHOR_DPSI_LO = 0.12     # [-] anchor weight below which the draw-down row is pure stationarity (and 1-this above which it is pure anchor)
_CI_PROJ_MIN_FRAC = 0.25   # [-] lower projection bound of the Ci unknowns, as a fraction of Ca (phydro)
_CI_PROJ_MAX_FRAC = 1.5    # [-] upper projection bound of the Ci unknowns, as a fraction of Ca (phydro)
_N_STATE = 6
_N_STATE_PHYDRO = 8


def state_size(stomatal_model: str) -> int:
    """Length of the Newton state vector for a stomatal model."""
    return _N_STATE_PHYDRO if stomatal_model == "phydro" else _N_STATE


def _xscale(n: int) -> jax.Array:
    if n == _N_STATE:
        return _LM_XSCALE
    if n == _N_STATE_PHYDRO:
        return jnp.concatenate([_LM_XSCALE, jnp.full((2,), _LM_XSCALE_DPSI)])
    raise ValueError(f"canopy Newton state of length {n}; expected 6 or 8")

# The residual is already weighted where it is built: its humidity row carries a
# factor of 1e3, which puts a kg/kg difference on the same footing as a kelvin
# one.  So no further residual weighting is applied here -- adding a second one
# would double-count it, and putting it on the wrong row would weight CO2
# instead.


# ---------------------------------------------------------------------------
# Forcing bundle NamedTuple — groups all per-column inputs for the closure
# ---------------------------------------------------------------------------

class CanopyForcingBundle(NamedTuple):
    """All column-level inputs needed inside the Newton-Raphson residual.

    Shapes are (ncol,) scalar per column (solver operates on single columns).
    """
    # Radiation
    LAI: jax.Array
    SZA: jax.Array
    La: jax.Array        # incoming LW from atmosphere [W m-2]
    epsf: jax.Array      # leaf emissivity
    epss: jax.Array      # soil emissivity
    fSun: jax.Array      # sunlit fraction
    APAR_Sun: jax.Array  # [μmol m-2 s-1]
    APAR_Sh: jax.Array
    Vcmax25_Sun: jax.Array  # canopy-integrated C3 Vcmax25, stressed [μmol m-2 s-1]
    Vcmax25_Sh: jax.Array
    Vcmax25_C4Sun: jax.Array
    Vcmax25_C4Sh: jax.Array
    ASW_Sun: jax.Array   # [W m-2]
    ASW_Sh: jax.Array
    ASW_Soil: jax.Array

    # Prescribed soil skin temperature for this inner canopy closure
    # (the outer Picard loop in ``canopy_land.py`` updates it between
    # passes).  Not a Newton variable.
    Ts_bc: jax.Array

    # Atmosphere
    Ca: jax.Array        # CO2 concentration [μmol mol-1]
    Ps: jax.Array        # pressure [Pa]
    Ta: jax.Array        # air temperature [K]
    lam: jax.Array       # latent heat of vaporisation [J kg-1]
    Cp: jax.Array        # specific heat [J kg-1 K-1]
    rhoa: jax.Array      # air density [kg m-3]
    Tv_atm: jax.Array    # virtual temperature [K]
    q_atm: jax.Array     # specific humidity [kg kg-1]

    # Stomatal
    m: jax.Array         # Ball-Berry slope (soil-moisture-stressed)
    b0: jax.Array        # Ball-Berry intercept (soil-stressed iff CanopyConfig.stress_b0)
    alf: jax.Array       # quantum yield
    TgC: jax.Array       # growth temperature [°C]
    fC4: jax.Array       # C4 fraction [-]
    fStress_soil: jax.Array  # soil evaporation stress [0-1]

    # Aerodynamics
    ur: jax.Array        # wind speed at reference height [m/s]
    CI: jax.Array        # clumping index
    z0m: jax.Array       # roughness length [m]
    displa: jax.Array    # displacement height [m]
    z0: jax.Array        # reference height [m]
    cv: jax.Array        # leaf BL forced-convection coefficient [m^-0.5 s^0.5]
    d_leaf: jax.Array    # characteristic leaf width [m]
    # Below-canopy soil-surface resistance to evaporation [s/m], added in SERIES
    # with the aerodynamic ``raw_below`` in the soil energy balance (Sellers 1992
    # r_ss + Sakaguchi-Zeng 2009 litter; see energy_balance.soil_surface_evap_
    # resistance).  Zeros recover the pure-aerodynamic (legacy beta) behaviour.
    r_soil_surface: jax.Array
    # Wetted leaf fraction [0-1] from the canopy-water store (interception).  The
    # wet part evaporates at the boundary-layer limit (no stomatal resistance),
    # so LE rises with wetness (interception loss).  0.0 = dry (no interception,
    # the default so every existing bundle construction is unchanged).
    fwet: jax.Array = 0.0
    # Optional Jmax25/Vcmax25 ratio override (P model, land/p_model.py).  None
    # keeps the kernel's Kattge & Knorr acclimated ratio bit-for-bit.  Appended
    # last so positional bundle constructions stay valid.
    rjv25: jax.Array | None = None
    # Leuning D0 [kPa] (CanopyConfig.d0_leuning_kpa) — consumed only when
    # stomatal_model == "leuning"; None otherwise.  Appended last.
    d0_leuning_kpa: jax.Array | float | None = None
    # P-hydro instantaneous supply (stomatal_model == "phydro" only; None
    # otherwise): THIS step's conductance-weighted soil water potential [MPa]
    # and soil-to-root resistance per leaf area, plus the hydraulic constants
    # as (ncol,) arrays so the implicit backward pass differentiates them.
    psi_s_mpa: jax.Array | None = None
    rsoil_leaf: jax.Array | None = None
    gplant_mol: jax.Array | None = None
    psi50_xylem_mpa: jax.Array | None = None
    ck_xylem: jax.Array | None = None
    gamma_cost: jax.Array | None = None


def _leaf_hydraulics(x, bundle: CanopyForcingBundle, stomatal_model: str):
    """``(hyd_Sun, hyd_Sh)`` from the state's draw-downs; ``(None, None)``
    unless the stomatal model is ``"phydro"``."""
    if stomatal_model != "phydro":
        return None, None
    b = bundle
    if b.psi_s_mpa is None:
        raise ValueError(
            "stomatal_model='phydro' but the forcing bundle carries no "
            "P-hydro supply (psi_s_mpa=None); the caller must thread the "
            "step's PhydroSupply and PHydroConfig into the bundle")
    _LAI_CLASS_FLOOR = 1.0e-3  # coeff-ok: leaf-area floor of an empty leaf class [m2/m2]
    def _hyd(dpsi, lai_class):
        return LeafHydraulics(
            dpsi_mpa=dpsi, psi_s_mpa=b.psi_s_mpa, rsoil_leaf=b.rsoil_leaf,
            lai_class=jnp.maximum(lai_class, _LAI_CLASS_FLOOR),
            gplant_mol=b.gplant_mol, psi50_xylem_mpa=b.psi50_xylem_mpa,
            ck_xylem=b.ck_xylem, gamma_cost=b.gamma_cost)
    return (_hyd(x[6], b.fSun * b.LAI), _hyd(x[7], (1.0 - b.fSun) * b.LAI))


# ---------------------------------------------------------------------------
# Residual function (defines the closure equations)
# ---------------------------------------------------------------------------

_KKT_FB_EPS = 1e-6  # [umol m-2 s-1 MPa-1] smoothing of the Fischer-Burmeister function at the origin


def _kkt_fischer_burmeister(a: jax.Array, b: jax.Array) -> jax.Array:
    """Fischer-Burmeister complementarity residual ``a + b - sqrt(a^2 + b^2)``.

    Zero iff ``a >= 0``, ``b >= 0`` and ``a b = 0``; smooth everywhere except
    the origin, where ``_KKT_FB_EPS`` regularises the square root.
    """
    return a + b - jnp.sqrt(a * a + b * b + _KKT_FB_EPS ** 2)


def _canopy_residual(
    x: jax.Array,
    bundle: CanopyForcingBundle,
    LE_module: str,
    stomatal_model: str,
    le_cap_mode: str,
    use_ta_for_photosynthesis: bool,
) -> jax.Array:
    """Compute the residual vector F(x) for the FULLY_COUPLED canopy closure.

    x = [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c]

    Leaves and soil share the canopy air space (Tc, q_c); soil communicates
    with Tc/q_c via the clumping-weighted below-canopy resistance.  ``Ts``
    is read from ``bundle.Ts_bc`` (prescribed; updated by the outer Picard
    loop between passes).

    Returns F such that F(x*) = 0 at the solution.
    """
    Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c = (
        x[0], x[1], x[2], x[3], x[4], x[5])
    hyd_Sun, hyd_Sh = _leaf_hydraulics(x, bundle, stomatal_model)

    b = bundle
    Ts = b.Ts_bc
    zldis = b.z0 - b.displa

    # ---- MOST stability ----
    ustar, rah_above, raw_above, uav, _ = monin_obukhov_stability(
        b.ur, b.Ta, b.Tv_atm, Tc, b.q_atm, q_c, zldis, b.z0m)

    # ---- Boundary and below-canopy resistances ----
    Rb_Sun, Rb_Sh = compute_boundary_layer_resistance(uav, b.LAI, b.fSun, b.cv, b.d_leaf)
    rah_below, raw_below = compute_below_canopy_resistance(uav, b.CI, b.LAI)
    # Soil evaporation (vapour) path adds the soil-surface resistance in SERIES
    # (Sellers 1992 r_ss + Sakaguchi-Zeng 2009 litter); the sensible-heat path
    # (rah_below) is a direct skin-to-canopy-air conduction and takes NO soil-side
    # resistance.  r_soil_surface = 0 recovers the pure-aerodynamic (legacy) form.
    raw_soil_evap = raw_below + b.r_soil_surface

    # Soil evaporation uses the beta efficiency b.fStress_soil (= soil pore RH
    # h_r from the prognostic top-layer matric potential) as a conductance
    # multiplier — see soil_energy_balance_bt / canopy_air_update.

    # ---- Longwave radiation ----
    lw_out  = canopy_longwave_rt(
        b.LAI, b.CI, b.SZA, Ts, Tf_Sun, Tf_Sh, b.La, b.epsf, b.epss)
    ALW_Sun, ALW_Sh, ALW_Soil = lw_out.ALW_Sun, lw_out.ALW_Sh, lw_out.ALW_Soil

    # ---- Photosynthesis ----
    T_phot_sun = b.Ta if use_ta_for_photosynthesis else Tf_Sun
    T_phot_sh  = b.Ta if use_ta_for_photosynthesis else Tf_Sh
    # Only NET An enters the residual (stomatal conductance + leaf CO2/energy
    # flux); the GROSS assimilation (carbon-model GPP) is consumed downstream
    # in ``canopy_forward``, so discard it here.
    An_Sun, _ = photosynthesis(
        T_phot_sun, Ci_Sun, b.APAR_Sun,
        b.Vcmax25_Sun, b.Vcmax25_C4Sun, b.fC4, b.Ps, b.alf, b.TgC,
        rjv25=b.rjv25)
    An_Sh, _ = photosynthesis(
        T_phot_sh,  Ci_Sh,  b.APAR_Sh,
        b.Vcmax25_Sh, b.Vcmax25_C4Sh, b.fC4, b.Ps, b.alf, b.TgC,
        rjv25=b.rjv25)

    # ---- Leaf microclimate (FULLY_COUPLED: leaves use canopy air space) ----
    e_c, es_c, VPD_c, RH_c, desTc, ddesTc, gamma_c = canopy_met_variables(
        b.Ps, Tc, q_c)

    # ---- Leaf energy balance ----
    if LE_module == "BT":
        q_f_Sun = saturation_specific_humidity(Tf_Sun, b.Ps)
        q_f_Sh  = saturation_specific_humidity(Tf_Sh,  b.Ps)
        _, LE_Sun, H_Sun, Tf_Sun_new, gs_Sun, Ci_Sun_new = leaf_energy_balance_bt(
            An_Sun, b.ASW_Sun, ALW_Sun, Tf_Sun, b.Ps, b.Ca,
            Tc, q_f_Sun, q_c, RH_c, VPD_c,
            b.lam, b.Cp, b.rhoa, Rb_Sun, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode,
            d0_leuning_kpa=b.d0_leuning_kpa, hyd=hyd_Sun)
        _, LE_Sh, H_Sh, Tf_Sh_new, gs_Sh, Ci_Sh_new = leaf_energy_balance_bt(
            An_Sh, b.ASW_Sh, ALW_Sh, Tf_Sh, b.Ps, b.Ca,
            Tc, q_f_Sh, q_c, RH_c, VPD_c,
            b.lam, b.Cp, b.rhoa, Rb_Sh, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode,
            d0_leuning_kpa=b.d0_leuning_kpa, hyd=hyd_Sh)
    else:  # PM
        _, LE_Sun, H_Sun, Tf_Sun_new, gs_Sun, Ci_Sun_new = leaf_energy_balance_pm(
            An_Sun, b.ASW_Sun, ALW_Sun, Tf_Sun, b.Ps, b.Ca,
            Tc, VPD_c, RH_c, desTc, ddesTc, gamma_c,
            b.Cp, b.rhoa, Rb_Sun, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode,
            d0_leuning_kpa=b.d0_leuning_kpa, hyd=hyd_Sun)
        _, LE_Sh, H_Sh, Tf_Sh_new, gs_Sh, Ci_Sh_new = leaf_energy_balance_pm(
            An_Sh, b.ASW_Sh, ALW_Sh, Tf_Sh, b.Ps, b.Ca,
            Tc, VPD_c, RH_c, desTc, ddesTc, gamma_c,
            b.Cp, b.rhoa, Rb_Sh, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode,
            d0_leuning_kpa=b.d0_leuning_kpa, hyd=hyd_Sh)

    # ---- Soil energy balance (prescribed Ts; G diagnosed as residual) ----
    q_s = saturation_specific_humidity(Ts, b.Ps)
    if LE_module == "BT":
        _, LE_Soil, H_Soil, _G = soil_energy_balance_bt(
            Ts, Tc, q_s, q_c,
            b.lam, b.rhoa, b.Cp,
            rah_below, raw_soil_evap, b.fStress_soil,
            b.ASW_Soil, ALW_Soil, le_cap_mode=le_cap_mode)
    else:
        _, LE_Soil, H_Soil, _G = soil_energy_balance_pm(
            Ts, Tc, q_s, q_c,
            b.lam, b.rhoa, b.Cp,
            rah_below, raw_soil_evap, b.fStress_soil,
            b.ASW_Soil, ALW_Soil, le_cap_mode=le_cap_mode)

    # ---- Canopy air update (FULLY_COUPLED: soil included) ----
    Tc_new, q_c_new = canopy_air_update(
        b.Ta, b.q_atm,
        Tf_Sun_new, Tf_Sh_new, Ts,
        gs_Sun, gs_Sh,
        Rb_Sun, Rb_Sh,
        rah_above, raw_above,
        rah_below, raw_soil_evap,
        b.fStress_soil, b.Ps, fwet=b.fwet)

    # ---- Sunlit-leaf anchor when fSun is too small for two-leaf split ----
    # When ``fSun`` is small, ``Rb_Sun = rb / (LAI · fSun)`` is large, the
    # sunlit-leaf equation is ill-conditioned and Newton has no attractor
    # for ``Tf_Sun`` or ``Ci_Sun``.  Collapse to a single-leaf model by
    # smoothly blending ``Tf_Sun → Tf_Sh`` as fSun shrinks (the two-leaf
    # partition is physically marginal anyway when direct beam is tiny).
    #
    # NOTE (2026-08-21): retargeting this trigger to the sunlit LEAF AREA was
    # tried and REVERTED.  It fires in the leafless limit, but it replaces the
    # residual of the leaf that still HAS area with equality to the empty one —
    # the wrong direction, and both reviewers flagged it as the mechanism that
    # spreads a degenerate leaf state into the canopy air.  Measured, it moved
    # the failure count only 186 -> 172 of 4500.  If this is revisited, the
    # EMPTY class must lose its equation and the active class must keep its own.
    #
    # The blend weight is a tanh smoother centred at ``fSun = 0.08`` with
    # transition half-width 0.03: anchor_weight ≈ 1 at fSun ≲ 0.05, ≈ 0
    # at fSun ≳ 0.11, and smoothly in between.  The earlier hard
    # ``where(fSun < 0.05, ...)`` produced a visible kink in H and LE
    # at the transition (seen in diagnostic plots) and made the residual
    # non-differentiable there, causing ``jax.grad`` to return NaN.
    anchor_weight = 0.5 * (1.0 - jnp.tanh((b.fSun - _ANCHOR_FSUN_CENTER) / _ANCHOR_FSUN_WIDTH))
    res_Tf_Sun = (anchor_weight * (Tf_Sun - Tf_Sh)
                  + (1.0 - anchor_weight) * (Tf_Sun - Tf_Sun_new))
    if stomatal_model == "phydro":
        # Under the hydraulic supply gs is independent of An, so the fixed-point
        # form ``Ci - (Ca - 1.6 An/gs)`` is stiff where gs is small (night, dry
        # soil: dCi/dgs ~ An/gs^2) and the damped Newton creeps.  Write the SAME
        # diffusion balance in flux form instead, ``gs_c (Ca - Ci) - An = 0``
        # [umol/m2/s], which is smooth and O(An) everywhere.  (Medlyn ties gs
        # to An, so its fixed-point form is a constant chi and never stiff.)
        gs_c_Sun = gs_Sun * gs_mol_per_ms(Tf_Sun, b.Ps) / DIFFUSIVITY_RATIO_H2O_CO2
        gs_c_Sh = gs_Sh * gs_mol_per_ms(Tf_Sh, b.Ps) / DIFFUSIVITY_RATIO_H2O_CO2
        bal_Sun = An_Sun - gs_c_Sun * (b.Ca - Ci_Sun)
        bal_Sh = An_Sh - gs_c_Sh * (b.Ca - Ci_Sh)
        # Ci is projected onto [_CI_PROJ_MIN_FRAC, _CI_PROJ_MAX_FRAC] Ca, and
        # the balance's root can lie outside (a dark leaf with cuticular
        # conductance only has Ci = Ca + 1.6 Rd/g0, above the ceiling for
        # Rd ~ 1 umol/m2/s).  The row is therefore the KKT condition of the
        # bounded balance: zero at an interior root, or at a bound with the
        # residual pointing outward.  Nested Fischer-Burmeister, slack in
        # flux units (gs_c times the Ci room left to the bound); the balance
        # rises with Ci, so below its root it is negative and above positive.
        def _bounded(bal, gs_c, Ci):
            lo = _CI_PROJ_MIN_FRAC * b.Ca
            hi = _CI_PROJ_MAX_FRAC * b.Ca
            at_floor = _kkt_fischer_burmeister(gs_c * (Ci - lo), bal)
            return _kkt_fischer_burmeister(gs_c * (hi - Ci), -at_floor)
        bal_Sun = _bounded(bal_Sun, gs_c_Sun, Ci_Sun)
        bal_Sh = _bounded(bal_Sh, gs_c_Sh, Ci_Sh)
    else:
        bal_Sun, bal_Sh = Ci_Sun - Ci_Sun_new, Ci_Sh - Ci_Sh_new
    res_Ci_Sun = (anchor_weight * (Ci_Sun - Ci_Sh)
                  + (1.0 - anchor_weight) * bal_Sun)

    rows = [
        res_Tf_Sun,
        Tf_Sh  - Tf_Sh_new,
        res_Ci_Sun,
        bal_Sh,
        Tc     - Tc_new,
        (q_c   - q_c_new) * 1e3,   # scale humidity residual
    ]
    if stomatal_model == "phydro":
        # Stationarity of each class's instantaneous profit A - gamma*dpsi^2
        # in its draw-down, at the current leaf state: A' = dAn/dCi of the
        # kernel, gs the class's conductance from the supply (back in mol).
        # The sunlit row takes the same degeneracy anchor as Tf/Ci.
        d_frac = floored_vpd(VPD_c) / b.Ps
        def _an_prime(T, Ci, APAR, Vc3, Vc4):
            f = lambda c: photosynthesis(T, c, APAR, Vc3, Vc4, b.fC4, b.Ps,
                                         b.alf, b.TgC, rjv25=b.rjv25)[0]
            return jax.jvp(f, (Ci,), (jnp.ones_like(Ci),))[1]
        foc_Sun = instantaneous_foc(
            hyd_Sun, d_frac,
            _an_prime(T_phot_sun, Ci_Sun, b.APAR_Sun, b.Vcmax25_Sun, b.Vcmax25_C4Sun),
            Ci_Sun, b.Ca, gs_Sun * gs_mol_per_ms(Tf_Sun, b.Ps))
        foc_Sh = instantaneous_foc(
            hyd_Sh, d_frac,
            _an_prime(T_phot_sh, Ci_Sh, b.APAR_Sh, b.Vcmax25_Sh, b.Vcmax25_C4Sh),
            Ci_Sh, b.Ca, gs_Sh * gs_mol_per_ms(Tf_Sh, b.Ps))
        # Scaled like the humidity row: the stationarity residual is O(1e-2)
        # umol/m2/s/MPa for a 0.01 MPa miss, far below the K-sized rows, so
        # without the factor the residual-norm convergence test would accept
        # draw-downs 0.01 MPa off (negative at night).
        # The draw-down row's anchor must SATURATE at 1: the tanh blend used for
        # Tf/Ci leaves ~1% of the stationarity residual in the row, and at a
        # degenerate sunlit fraction (low sun: dawn, dusk, winter) that leak is
        # as large as the anchor term itself and puts the row's root below the
        # dpsi >= 0 bound, where the column can never converge (measured: the
        # unsolved site steps were all low-sun).  Piecewise-linear remap of the
        # same weight: exactly 1 for fSun below ~0.05, 0 above ~0.11.
        # The anchor term is expressed in the stationarity residual's own
        # units, 2 gamma (dpsi_sun - dpsi_sh): the draw-down "distance" the
        # risk-cost slope converts a residual into.  Blending two quantities
        # of the same scale keeps the row's root between the shaded draw-down
        # and the sunlit optimum, hence feasible, at every weight.
        w_dpsi = jnp.clip((anchor_weight - _ANCHOR_DPSI_LO) / (1.0 - 2.0 * _ANCHOR_DPSI_LO), 0.0, 1.0)
        anchor_dpsi = 2.0 * b.gamma_cost * (x[6] - x[7])
        # The draw-down is bounded below by zero, so its optimality condition
        # is the KKT pair "dpsi >= 0, dP/ddpsi <= 0, one of them zero", not
        # the bare stationarity dP/ddpsi = 0: a respiring leaf (night, Ci >
        # Ca, An < 0) LOSES profit for any opening, its optimum is dpsi = 0
        # with a strictly negative slope, and a residual that insists on
        # slope = 0 there can never converge.  Fischer-Burmeister form
        # a + b - sqrt(a^2 + b^2), zero exactly on the KKT set, with a = 2 gamma
        # dpsi (the draw-down in the slope's units) and b = -dP/ddpsi; the eps
        # keeps it differentiable at the origin.
        kkt_Sun = _kkt_fischer_burmeister(2.0 * b.gamma_cost * x[6], -foc_Sun)
        kkt_Sh = _kkt_fischer_burmeister(2.0 * b.gamma_cost * x[7], -foc_Sh)
        rows += [_FOC_ROW_SCALE * (w_dpsi * anchor_dpsi + (1.0 - w_dpsi) * kkt_Sun),
                 _FOC_ROW_SCALE * kkt_Sh]
    return jnp.array(rows)


# ---------------------------------------------------------------------------
# Diagnostic forward pass — same logic but returns all fluxes
# ---------------------------------------------------------------------------

def canopy_forward(
    x: jax.Array,
    bundle: CanopyForcingBundle,
    LE_module: str,
    stomatal_model: str,
    le_cap_mode: str,
    use_ta_for_photosynthesis: bool,
) -> dict:
    """Evaluate the FULLY_COUPLED canopy state and return all fluxes.

    Used after the Newton solver has converged to extract final diagnostics.
    The returned ``G`` is the surface-energy-budget residual
    ``Rn_soil - LE_soil - H_soil``, intended to be passed as the top BC to
    ``solve_soil_thermal`` in the caller.
    """
    Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c = (
        x[0], x[1], x[2], x[3], x[4], x[5])
    hyd_Sun, hyd_Sh = _leaf_hydraulics(x, bundle, stomatal_model)

    b = bundle
    Ts = b.Ts_bc
    zldis = b.z0 - b.displa

    ustar, rah_above, raw_above, uav, zeta = monin_obukhov_stability(
        b.ur, b.Ta, b.Tv_atm, Tc, b.q_atm, q_c, zldis, b.z0m)

    Rb_Sun, Rb_Sh = compute_boundary_layer_resistance(uav, b.LAI, b.fSun, b.cv, b.d_leaf)
    rah_below, raw_below = compute_below_canopy_resistance(uav, b.CI, b.LAI)
    # Soil evaporation (vapour) path adds the soil-surface resistance in SERIES
    # (Sellers 1992 r_ss + Sakaguchi-Zeng 2009 litter); the sensible-heat path
    # (rah_below) is a direct skin-to-canopy-air conduction and takes NO soil-side
    # resistance.  r_soil_surface = 0 recovers the pure-aerodynamic (legacy) form.
    raw_soil_evap = raw_below + b.r_soil_surface

    lw_out  = canopy_longwave_rt(
        b.LAI, b.CI, b.SZA, Ts, Tf_Sun, Tf_Sh, b.La, b.epsf, b.epss)
    ALW_Sun, ALW_Sh, ALW_Soil = lw_out.ALW_Sun, lw_out.ALW_Sh, lw_out.ALW_Soil
    Ls = lw_out.Ls
    Lcanopy_up = lw_out.Lcanopy_up
    gap_LW = lw_out.gap_LW
    LW_out = lw_out.LW_out
    eps_col = lw_out.eps_col
    LW_emit = lw_out.LW_emit

    T_phot_sun = b.Ta if use_ta_for_photosynthesis else Tf_Sun
    T_phot_sh  = b.Ta if use_ta_for_photosynthesis else Tf_Sh
    # NET An (An_Sun/An_Sh) drives the leaf energy/CO2 flux + SIF; GROSS A
    # (Agross_Sun/Agross_Sh) is the carbon-model GPP (uptake BEFORE dark
    # respiration).  Both are returned so ``two_leaf_canopy`` can report GROSS
    # GPP while the leaf coupling stays NET (avoids double-counting foliar
    # respiration against the carbon model's r_maint_fol*C_fol).
    An_Sun, Agross_Sun = photosynthesis(
        T_phot_sun, Ci_Sun, b.APAR_Sun,
        b.Vcmax25_Sun, b.Vcmax25_C4Sun, b.fC4, b.Ps, b.alf, b.TgC,
        rjv25=b.rjv25)
    An_Sh, Agross_Sh = photosynthesis(
        T_phot_sh,  Ci_Sh,  b.APAR_Sh,
        b.Vcmax25_Sh, b.Vcmax25_C4Sh, b.fC4, b.Ps, b.alf, b.TgC,
        rjv25=b.rjv25)

    # FULLY_COUPLED: leaves and soil share the canopy air space (Tc, q_c).
    e_c, es_c, VPD_c, RH_c, desTc, ddesTc, gamma_c = canopy_met_variables(
        b.Ps, Tc, q_c)

    if LE_module == "BT":
        q_f_Sun = saturation_specific_humidity(Tf_Sun, b.Ps)
        q_f_Sh  = saturation_specific_humidity(Tf_Sh,  b.Ps)
        Rn_Sun, LE_Sun, H_Sun, _, gs_Sun, _ = leaf_energy_balance_bt(
            An_Sun, b.ASW_Sun, ALW_Sun, Tf_Sun, b.Ps, b.Ca,
            Tc, q_f_Sun, q_c, RH_c, VPD_c,
            b.lam, b.Cp, b.rhoa, Rb_Sun, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode,
            d0_leuning_kpa=b.d0_leuning_kpa, hyd=hyd_Sun)
        Rn_Sh,  LE_Sh,  H_Sh,  _, gs_Sh, _  = leaf_energy_balance_bt(
            An_Sh, b.ASW_Sh, ALW_Sh, Tf_Sh, b.Ps, b.Ca,
            Tc, q_f_Sh, q_c, RH_c, VPD_c,
            b.lam, b.Cp, b.rhoa, Rb_Sh, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode,
            d0_leuning_kpa=b.d0_leuning_kpa, hyd=hyd_Sh)
    else:
        Rn_Sun, LE_Sun, H_Sun, _, gs_Sun, _ = leaf_energy_balance_pm(
            An_Sun, b.ASW_Sun, ALW_Sun, Tf_Sun, b.Ps, b.Ca,
            Tc, VPD_c, RH_c, desTc, ddesTc, gamma_c,
            b.Cp, b.rhoa, Rb_Sun, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode,
            d0_leuning_kpa=b.d0_leuning_kpa, hyd=hyd_Sun)
        Rn_Sh,  LE_Sh,  H_Sh,  _, gs_Sh,  _ = leaf_energy_balance_pm(
            An_Sh, b.ASW_Sh, ALW_Sh, Tf_Sh, b.Ps, b.Ca,
            Tc, VPD_c, RH_c, desTc, ddesTc, gamma_c,
            b.Cp, b.rhoa, Rb_Sh, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode,
            d0_leuning_kpa=b.d0_leuning_kpa, hyd=hyd_Sh)

    # Wet-leaf evaporation (interception loss): the fwet share of each leaf's
    # latent flux, which is sourced from the canopy-water store rather than
    # transpired from the root zone.  Attribution matches the BT conductance
    # blend g_lh_eff = (1-fwet)*g_lh + fwet/Rb (exact for BT; a close proxy for
    # PM, whose rc_eff uses the same vapour-conductance blend).  The caller
    # routes ``LE_wet_canopy`` to the store and ``LE_canopy - LE_wet_canopy`` to
    # the transpiration sink.
    def _le_wet(LE, gs, Rb):
        Rb_s = jnp.maximum(Rb, 1e-9)             # guard 1/Rb (matches leaf LE)
        g_lh = gs / (gs * Rb_s + 1.0)
        g_lh_wet = 1.0 / Rb_s
        g_lh_eff = (1.0 - b.fwet) * g_lh + b.fwet * g_lh_wet
        return LE * jnp.where(g_lh_eff > 0.0,
                              b.fwet * g_lh_wet / g_lh_eff, 0.0)
    LE_wet_Sun = _le_wet(LE_Sun, gs_Sun, Rb_Sun)
    LE_wet_Sh = _le_wet(LE_Sh, gs_Sh, Rb_Sh)

    q_s = saturation_specific_humidity(Ts, b.Ps)
    if LE_module == "BT":
        Rn_Soil, LE_Soil, H_Soil, G = soil_energy_balance_bt(
            Ts, Tc, q_s, q_c,
            b.lam, b.rhoa, b.Cp,
            rah_below, raw_soil_evap, b.fStress_soil,
            b.ASW_Soil, ALW_Soil, le_cap_mode=le_cap_mode)
    else:
        Rn_Soil, LE_Soil, H_Soil, G = soil_energy_balance_pm(
            Ts, Tc, q_s, q_c,
            b.lam, b.rhoa, b.Cp,
            rah_below, raw_soil_evap, b.fStress_soil,
            b.ASW_Soil, ALW_Soil, le_cap_mode=le_cap_mode)

    return dict(
        An_Sun=An_Sun, An_Sh=An_Sh,
        Agross_Sun=Agross_Sun, Agross_Sh=Agross_Sh,
        LE_Sun=LE_Sun, LE_Sh=LE_Sh, LE_Soil=LE_Soil,
        LE_wet_Sun=LE_wet_Sun, LE_wet_Sh=LE_wet_Sh,
        H_Sun=H_Sun,   H_Sh=H_Sh,   H_Soil=H_Soil,
        Rn_Sun=Rn_Sun, Rn_Sh=Rn_Sh, Rn_Soil=Rn_Soil,
        G=G, Ls=Ls, Lcanopy_up=Lcanopy_up, gap_LW=gap_LW, LW_out=LW_out,
        eps_col=eps_col, LW_emit=LW_emit,
        gs_Sun=gs_Sun, gs_Sh=gs_Sh,
        ustar=ustar, zeta=zeta,
        rah_above=rah_above, rah_below=rah_below,
        Rb_Sun=Rb_Sun, Rb_Sh=Rb_Sh,
        ALW_Sun=ALW_Sun, ALW_Sh=ALW_Sh, ALW_Soil=ALW_Soil,
        # P-hydro draw-downs of the two classes (None unless "phydro").
        dpsi_Sun=(None if hyd_Sun is None else hyd_Sun.dpsi_mpa),
        dpsi_Sh=(None if hyd_Sh is None else hyd_Sh.dpsi_mpa),
    )


# ---------------------------------------------------------------------------
# Newton-Raphson solver
# ---------------------------------------------------------------------------

def solve_canopy_closure(
    initial_state: jax.Array,
    bundle: CanopyForcingBundle,
    config: CanopyConfig,
) -> tuple[jax.Array, jax.Array]:
    """Newton-Raphson canopy closure via jax.lax.scan.

    Parameters
    ----------
    initial_state : shape (6,), or (8,) under stomatal_model="phydro"
        [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c(, dpsi_Sun, dpsi_Sh)].  Soil
        skin T is passed as ``bundle.Ts_bc`` — see module docstring.
    bundle : CanopyForcingBundle
        All per-column forcing inputs (including ``Ts_bc``).
    config : CanopyConfig
        Solver settings (max_iters, tol, LE_module, stomatal_model).

    Returns
    -------
    x_final : same shape as ``initial_state`` — the final Newton iterate
    n_iters : scalar int  — iteration count when the loop stopped
    converged : scalar bool — whether the loop stopped because the Newton step
        fell below ``tol`` (True) rather than because it ran out of iterations
        or its Jacobian went singular (False).  A False here means ``x_final``
        is NOT a root of the canopy residual, so the fluxes derived from it are
        not physics; callers must not consume them as if they were.
    """
    # Fail-early on a typo'd LE_module: the internal leaf-energy dispatch is a
    # bare ``if LE_module == "BT": ... else: # PM``, so an unknown value would
    # SILENTLY run Penman-Monteith.  Guards the direct-call path (callers that
    # skip CanopyConfig.validate).  Shares config.VALID_LE_MODULES (no drift).
    if config.LE_module not in VALID_LE_MODULES:
        raise ValueError(
            f"unknown LE_module {config.LE_module!r}; the leaf-energy method "
            f"must be one of {VALID_LE_MODULES} ('BT'=bulk transfer, "
            "'PM'=Penman-Monteith)")
    solver = _make_implicit_newton_solver(
        LE_module=config.LE_module,
        stomatal_model=config.stomatal_model,
        le_cap_mode=config.le_cap_mode,
        use_ta_for_photosynthesis=config.use_ta_for_photosynthesis,
        max_iters=config.max_iters,
        tol=config.tol,
    )
    return solver(initial_state, bundle)


# ---------------------------------------------------------------------------
# Implicit-function-theorem Newton solver
# ---------------------------------------------------------------------------

def _make_implicit_newton_solver(
    LE_module: str,
    stomatal_model: str,
    le_cap_mode: str,
    use_ta_for_photosynthesis: bool,
    max_iters: int,
    tol: float,
):
    """Create a custom_vjp Newton solver bound to the static config args.

    Forward pass:
        Damped Newton via ``jax.lax.while_loop`` with true early stopping —
        the iteration halts as soon as ``||delta|| < tol``, saving wasted
        work when convergence is fast.

    Backward pass (implicit function theorem):
        At the fixed point ``x*`` with ``F(x*; θ) = 0``,
            ``dx*/dθ = −(∂F/∂x)⁻¹ · ∂F/∂θ``
        which gives the adjoint
            ``λ = (∂F/∂x)⁻ᵀ g_x``,  ``grad_θ = −(∂F/∂θ)ᵀ λ``.
        Solved with an exact ``jnp.linalg.solve`` (verified to match central
        finite differences); the canopy Jacobian is stiff (cond ~ 1e5) but
        nonsingular at converged columns, and the earlier ``lstsq(rcond=1e-4)``
        truncation biased the gradient.  Without IFT-based gradients,
        ``jax.grad`` through the scan-based Newton produces NaN from
        second-order tangents at near-singular Jacobians.

    Guards (backward only):
        * Convergence: if Newton did not converge (hit max_iters), the IFT
          identity F(x*)=0 does not hold, so all gradients are zeroed.
        * Divergence: if ``x*`` is NaN or the adjoint ``λ`` is NaN/Inf, all
          gradients are zeroed (the whole solve is invalid).
        * Per-leaf: otherwise only individual non-finite cotangent leaves are
          zeroed.  The Monin-Obukhov scan is not differentiable w.r.t. its
          aerodynamic forcing (Ta/Tv_atm/q_atm/ur/z0m/displa/z0 come back NaN),
          but every trainable physics-parameter gradient is finite and kept.

    Adapted from DifferBESS ``algo.newton_root._make_implicit_newton_solver``.
    """
    def _F(x, bundle):
        return _canopy_residual(
            x, bundle,
            LE_module=LE_module,
            stomatal_model=stomatal_model,
            le_cap_mode=le_cap_mode,
            use_ta_for_photosynthesis=use_ta_for_photosynthesis,
        )

    def _forward(x0, bundle):
        """Damped least-squares solve; see the module constants for the why.

        Each iteration proposes a step, accepts it only if it reduces the
        residual, and adjusts the damping from how well the actual reduction
        matched the one the local model predicted.  A rejected step leaves the
        iterate untouched and raises the damping, so the residual can never
        increase and an overshooting column is progressively pulled back inside
        the bracket it was jumping over.
        """
        # ``max_iters`` comes from CanopyConfig and is honoured as the cap, so
        # a hard case can still be given more budget from config.  ``tol`` was
        # the old STEP-norm tolerance; the damped solve converges on the
        # RESIDUAL (_LM_RTOL/_LM_ATOL) because a step can be small purely from
        # damping — reusing ``tol``'s value with a different meaning would be a
        # silent semantic change, so it is deliberately unused on this path.
        F = partial(_F, bundle=bundle)
        Jac = jax.jacfwd(F)
        xscale = _xscale(int(x0.shape[0]))

        def _res_jac(x):
            """Residual and Jacobian in SCALED variables (z = x / xscale)."""
            return F(x), Jac(x) * xscale              # columns scaled

        def _step(J_z, resid, lam):
            """Damped step, via QR of the augmented system.

            Solves  min || [J ; sqrt(lam) I] dz - [-F ; 0] ||.  Stacking the
            damping as extra ROWS keeps the conditioning at the Jacobian's own,
            where forming the normal equations would square it -- fatal here in
            single precision at a condition number of ~1e5.
            """
            n = resid.shape[0]
            aug = jnp.vstack([J_z, jnp.sqrt(lam) * jnp.eye(n)])
            rhs = jnp.concatenate([-resid, jnp.zeros(n, resid.dtype)])
            q, r = jnp.linalg.qr(aug)
            dz = jax.scipy.linalg.solve_triangular(r, q.T @ rhs, lower=False)
            # Reduction the local linear model predicts for ||F||^2, computed
            # from the model residual rather than from a normal-equations
            # quadratic, so it cannot disagree with the step actually taken.
            pred_resid = resid + J_z @ dz
            return dz, jnp.sum(pred_resid * pred_resid)

        def cond(state):
            _, _, _, _, _, _, done = state
            return ~done

        def _bounds():
            """Lower/upper bounds of the phydro unknowns (None otherwise).

            The draw-downs are non-negative by definition (a leaf cannot be
            wetter than its soil) and no stationary point lies below zero (the
            gain term is >= 0, the risk term restoring).  Ci is kept inside
            [0.25, 1.5] Ca: the floor sits above Gamma* for leaf temperatures
            up to ~43 degC, so dAn/dCi > 0 and the draw-down row keeps its
            gain (below Gamma* the kernel's gross floors make A' = 0 and the
            leaf freezes in a non-transpiring state); the ceiling lies above
            the respiring root Ci = Ca + 1.6 Rd/g0 (night, An < 0), so that
            root is not a corner of the projection.
            """
            lo = jnp.full((_N_STATE_PHYDRO,), -jnp.inf, x0.dtype)
            hi = jnp.full((_N_STATE_PHYDRO,), jnp.inf, x0.dtype)
            lo = lo.at[_N_STATE:].set(0.0)
            lo = lo.at[2:4].set(_CI_PROJ_MIN_FRAC * bundle.Ca)
            hi = hi.at[2:4].set(_CI_PROJ_MAX_FRAC * bundle.Ca)
            return lo, hi

        def body(state):
            x, resid, J_z, n_sq, lam, i, _ = state
            dz, n_sq_pred = _step(J_z, resid, lam)
            if x0.shape[0] == _N_STATE_PHYDRO:
                # Bound-constrained step (active set): an unknown sitting on a
                # bound whose unconstrained step points outward is frozen (its
                # Jacobian column zeroed) and the step re-solved, so the local
                # model does not bank on a move the projection then cancels —
                # the mismatch otherwise reads as a failed step, inflates the
                # damping and stalls the column (measured: night leaves with
                # dpsi = 0 never reached Ci = Ca).  The trial is projected
                # onto the bounds and the predicted reduction is taken at the
                # step actually made.
                lo, hi = _bounds()
                outward = ((x <= lo) & (dz < 0.0)) | ((x >= hi) & (dz > 0.0))
                free = jnp.where(outward, 0.0, 1.0)
                dz, _ = _step(J_z * free[None, :], resid, lam)
                x_trial = jnp.clip(x + xscale * dz, lo, hi)
                pred_resid = resid + J_z @ ((x_trial - x) / xscale)
                n_sq_pred = jnp.sum(pred_resid * pred_resid)
            else:
                x_trial = x + xscale * dz
            resid_t = F(x_trial)
            n_sq_t = jnp.sum(resid_t * resid_t)

            # Gain ratio: actual reduction over predicted.  Both terms come from
            # THIS step, so the first iteration is a real measurement rather
            # than a fiction inherited from an initialised-to-infinity carry.
            rho = (n_sq - n_sq_t) / jnp.maximum(n_sq - n_sq_pred, 1e-30)
            accept = (rho > 0.0) & jnp.all(jnp.isfinite(resid_t))

            # A REJECTED STEP IS NOT TAKEN.  Raising the damping for next time
            # while still moving is what makes an oscillation persist; keeping
            # the iterate is what makes the residual monotone.
            x = jnp.where(accept, x_trial, x)
            resid = jnp.where(accept, resid_t, resid)
            n_sq_new = jnp.where(accept, n_sq_t, n_sq)
            J_z = jnp.where(accept, Jac(x_trial) * xscale, J_z)

            # A non-finite trial residual makes ``rho`` NaN, and every NaN
            # comparison is False — written naively that freezes the damping
            # and retries the SAME divergent trial until the iteration cap
            # (codex).  Phrase the bad-step test so NaN lands in it: damping
            # must RISE on any step that is not demonstrably adequate.
            step_bad = ~(rho >= _LM_RHO_BAD)          # True for NaN
            step_good = rho > _LM_RHO_GOOD            # False for NaN
            lam = jnp.clip(
                jnp.where(step_bad, lam * _LM_LAMBDA_UP,
                          jnp.where(step_good, lam / _LM_LAMBDA_DOWN, lam)),
                0.0, _LM_LAMBDA_MAX)

            # CONVERGENCE IS THE RESIDUAL, NOT THE STEP.  A step can be small
            # merely because the damping is large, which is exactly the state a
            # stalled column ends in -- calling that converged would hand the
            # backward pass a point that is not a root and unmask a gradient
            # that must be zeroed.  A stalled column instead leaves the loop by
            # the damping ceiling, with converged False.
            converged = n_sq_new <= (_LM_ATOL + _LM_RTOL * n_sq_0)
            done = converged | (lam >= _LM_LAMBDA_MAX) | (i + 1 >= max_iters)
            return (x, resid, J_z, n_sq_new, lam, i + 1, done)

        resid_0, J_z_0 = _res_jac(x0)
        n_sq_0 = jnp.sum(resid_0 * resid_0)
        conv_0 = n_sq_0 <= _LM_ATOL
        x_final, _r, _j, _n, _lam, n_iters, _done = jax.lax.while_loop(
            cond, body,
            (x0, resid_0, J_z_0, n_sq_0, jnp.asarray(_LM_LAMBDA_0, x0.dtype),
             jnp.array(0), conv_0))
        # Recompute the flag on the returned iterate so what the caller reads is
        # a statement about the state it is handed, not about a loop variable.
        resid_f = F(x_final)
        converged = jnp.sum(resid_f * resid_f) <= (_LM_ATOL + _LM_RTOL * n_sq_0)
        return x_final, n_iters, converged

    @jax.custom_vjp
    def solve(x0, bundle):
        x_final, n_iters, converged = _forward(x0, bundle)
        return x_final, n_iters, converged

    def solve_fwd(x0, bundle):
        x_final, n_iters, converged = _forward(x0, bundle)
        return (x_final, n_iters, converged), (x_final, bundle, converged)

    def solve_bwd(res, g):
        x_star, bundle, converged = res
        # Cotangents for the iteration count and the convergence flag are
        # ignored: both are integer/boolean diagnostics, not differentiable
        # outputs.
        g_x, _, _ = g

        # NaN / divergence guard: substitute zeros so the adjoint solve
        # is well-defined even if Newton diverged.
        x_safe = jnp.where(jnp.isnan(x_star), jnp.zeros_like(x_star), x_star)
        had_nan = jnp.any(jnp.isnan(x_star)) | jnp.any(jnp.isnan(g_x))
        # Convergence guard: the IFT adjoint dx*/dθ = -(∂F/∂x)^{-1} ∂F/∂θ is
        # only valid at a true root F(x*) = 0.  If the forward solve hit
        # max_iters (or the damping ceiling) without the RESIDUAL norm reaching
        # tolerance, x* is not a root and the adjoint is
        # inconsistent — zero the cotangent (mirrors the NaN guard) rather than
        # emit a misleading gradient.  ``converged`` is the forward loop's own
        # convergence flag, so genuinely-converged columns are never masked.
        not_converged = ~converged

        # Jacobian ∂F/∂x at the fixed point
        J = jax.jacfwd(partial(_F, bundle=bundle))(x_safe)

        # Adjoint solve: J^T λ = g_x.  Use an exact linear solve, not
        # ``lstsq(rcond=1e-4)``: the canopy Jacobian is routinely stiff
        # (cond(J) ~ 1e5 even at a well-converged midday column), and the
        # rcond truncation silently *biased* the gradient (~5x too small vs
        # central finite differences).  A genuinely singular J (low fSun,
        # wilting, freezing) gives inf/nan here and is caught by the guard
        # below.
        lam = jnp.linalg.solve(J.T, g_x)

        # Gradient w.r.t. bundle via VJP of F at x_safe
        _, vjp_fn = jax.vjp(partial(_F, x_safe), bundle)
        grad_bundle = vjp_fn(-lam)[0]

        # ``solve_failed`` invalidates the WHOLE adjoint (Newton diverged, did
        # not converge, or the adjoint solve produced inf/nan) → zero every
        # gradient.  Otherwise mask only individual non-finite leaves: the
        # Monin-Obukhov scan is not differentiable w.r.t. its aerodynamic
        # forcing (Ta, Tv_atm, q_atm, ur, z0m, displa, z0), whose cotangents
        # come back NaN — zero those alone and keep the finite gradients of
        # every trainable physics parameter (Vcmax25, m, b0, TgC, CI, cv,
        # d_leaf, emissivities, ...).  The old all-or-nothing mask let a single
        # NaN forcing leaf zero the entire gradient (so jax.grad returned 0).
        solve_failed = (had_nan | not_converged
                        | jnp.any(jnp.isnan(lam)) | jnp.any(jnp.isinf(lam)))

        def _mask_leaf(v):
            bad = solve_failed | jnp.isnan(v) | jnp.isinf(v)
            return jnp.where(bad, jnp.zeros_like(v), v)

        grad_bundle = jax.tree.map(_mask_leaf, grad_bundle)
        return jnp.zeros_like(x_star), grad_bundle

    solve.defvjp(solve_fwd, solve_bwd)
    return solve
