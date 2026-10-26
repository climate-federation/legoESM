"""Newton-Raphson canopy closure solver.

Solves the 6-variable canopy energy balance system:
  x = [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c]

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

The closure uses the core package's scaled Levenberg--Marquardt helper: an
augmented-QR forward step with Nielsen gain-ratio rejection and an exact,
column-equilibrated implicit adjoint.  Its convergence flag is based on the
residual norm, and non-converged solves have zero gradient.

**Coupling scheme**: only the DifferBESS FULLY_COUPLED formulation is
implemented — leaves and soil share the same canopy air space (Tc, q_c)
with a clumping-index-weighted below-canopy resistance (see
``compute_below_canopy_resistance``).  The ``VEG_ONLY`` and
``LEAVES_ATMO`` variants from DifferBESS were removed to keep the
Newton residual shape minimal; they can be added back later if a
multi-scheme comparison study needs them.

The coupling architecture was adapted for legoESM's multilayer soil thermal
solver via an outer Picard iteration.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.core.nonlinear import make_implicit_newton_solver
from legoesm.land.canopy.config import (
    RH_CAP_WIDTH_MAX, VALID_LE_MODULES, ZETA_CAP_WIDTH_MAX, CanopyConfig)
from legoesm.land.canopy.energy_balance import (
    canopy_air_update,
    canopy_met_variables,
    leaf_energy_balance_bt,
    leaf_energy_balance_pm,
    saturation_specific_humidity,
    soil_energy_balance_bt,
)
from legoesm.land.canopy.photosynthesis import photosynthesis
from legoesm.land.canopy.radiative_transfer import canopy_longwave_rt
from legoesm.land.canopy.stability import (
    LEAF_AREA_FLOOR,
    compute_below_canopy_resistance,
    compute_boundary_layer_resistance,
    monin_obukhov_stability,
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
_LM_FSCALE = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0e-3])

# --- Root certificate (numerics; containment thresholds, not tunable physics) --
# The relative gate above (residual reduced 1e8x from the SEED) cannot certify a
# root on its own: from a far-off warm-start seed the seed residual is huge and
# the gate admits a residual of O(10) K, and the residual also has spurious
# fixed points far outside the range its saturation / radiation formulae are
# valid in (measured 2026-09-29: leaf roots at 462, 891, 2300 and -1301 K,
# q_c = -0.11 kg/kg, from seeds copied out of AMIP checkpoints).  Such a state,
# once cached as the next step's seed, perpetuated itself: ~200-320 of 20709
# land columns carried leaf temperatures of 13-5655 K in every AMIP run, and one
# ran the soil and lowest air down to ~90 K.  A converged state must therefore
# also sit inside a broad physical box and below an absolute residual ceiling.
# Cold-started solves reach n_sq ~1e-7..1e-9; the relative gate allows <~1e-4.
_ROOT_T_MIN_K = 150.0      # [K] below any terrestrial leaf / canopy air
_ROOT_T_MAX_K = 360.0      # [K] above any terrestrial leaf / canopy air
_ROOT_QC_MIN = -1.0e-10    # [kg/kg] q_c >= 0 up to round-off at the dry boundary
_ROOT_NSQ_MAX = 1.0e-2     # [-] absolute ceiling on the scaled squared residual


def canopy_state_admissible(x: jax.Array) -> jax.Array:
    """True where ``x[..., :6]`` = [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c] is a
    finite state inside the physical box a canopy root may occupy.

    Containment, not a full physical check: Ci is only required finite (no
    garbage state seen so far had an out-of-range Ci; the leaf temperatures and
    q_c are what went wrong)."""
    T = x[..., jnp.array([0, 1, 4])]
    return (jnp.all(jnp.isfinite(x), axis=-1)
            & jnp.all((T >= _ROOT_T_MIN_K) & (T <= _ROOT_T_MAX_K), axis=-1)
            & (x[..., 5] >= _ROOT_QC_MIN))


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
    # Latent heat [J kg-1] charged to the SOIL latent flux; None = ``lam``.  Over
    # snow the driver passes the snow-weighted (1-w) L_v + w L_s (#1875); leaves
    # keep ``lam``.
    lam_soil: jax.Array | None = None


# ---------------------------------------------------------------------------
# Residual function (defines the closure equations)
# ---------------------------------------------------------------------------

def _canopy_residual(
    x: jax.Array,
    bundle: CanopyForcingBundle,
    LE_module: str,
    stomatal_model: str,
    le_cap_mode: str,
    use_ta_for_photosynthesis: bool,
    rh_cap_width: float,
    zeta_cap_width: float,
    most_n_iters: int,
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

    b = bundle
    Ts = b.Ts_bc
    zldis = b.z0 - b.displa

    # ---- MOST stability ----
    ustar, rah_above, raw_above, uav, _ = monin_obukhov_stability(
        b.ur, b.Ta, b.Tv_atm, Tc, b.q_atm, q_c, zldis, b.z0m,
        n_iters=most_n_iters, zeta_cap_width=zeta_cap_width)

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
        b.Vcmax25_Sun, b.Vcmax25_C4Sun, b.fC4, b.Ps, b.alf, b.TgC)
    An_Sh, _ = photosynthesis(
        T_phot_sh,  Ci_Sh,  b.APAR_Sh,
        b.Vcmax25_Sh, b.Vcmax25_C4Sh, b.fC4, b.Ps, b.alf, b.TgC)

    # ---- Leaf microclimate (FULLY_COUPLED: leaves use canopy air space) ----
    e_c, es_c, VPD_c, RH_c, desTc, ddesTc, gamma_c = canopy_met_variables(
        b.Ps, Tc, q_c, rh_cap_width)

    # ---- Leaf energy balance ----
    if LE_module == "BT":
        q_f_Sun = saturation_specific_humidity(Tf_Sun, b.Ps)
        q_f_Sh  = saturation_specific_humidity(Tf_Sh,  b.Ps)
        _, LE_Sun, H_Sun, Tf_Sun_new, gs_Sun, Ci_Sun_new = leaf_energy_balance_bt(
            An_Sun, b.ASW_Sun, ALW_Sun, Tf_Sun, b.Ps, b.Ca,
            Tc, q_f_Sun, q_c, RH_c, VPD_c,
            b.lam, b.Cp, b.rhoa, Rb_Sun, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode)
        _, LE_Sh, H_Sh, Tf_Sh_new, gs_Sh, Ci_Sh_new = leaf_energy_balance_bt(
            An_Sh, b.ASW_Sh, ALW_Sh, Tf_Sh, b.Ps, b.Ca,
            Tc, q_f_Sh, q_c, RH_c, VPD_c,
            b.lam, b.Cp, b.rhoa, Rb_Sh, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode)
    else:  # PM
        _, LE_Sun, H_Sun, Tf_Sun_new, gs_Sun, Ci_Sun_new = leaf_energy_balance_pm(
            An_Sun, b.ASW_Sun, ALW_Sun, Tf_Sun, b.Ps, b.Ca,
            Tc, VPD_c, RH_c, desTc, ddesTc, gamma_c,
            b.Cp, b.rhoa, Rb_Sun, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode)
        _, LE_Sh, H_Sh, Tf_Sh_new, gs_Sh, Ci_Sh_new = leaf_energy_balance_pm(
            An_Sh, b.ASW_Sh, ALW_Sh, Tf_Sh, b.Ps, b.Ca,
            Tc, VPD_c, RH_c, desTc, ddesTc, gamma_c,
            b.Cp, b.rhoa, Rb_Sh, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode)

    # ---- Soil energy balance (prescribed Ts; G diagnosed as residual) ----
    q_s = saturation_specific_humidity(Ts, b.Ps)
    # PM and BT coincide at the soil level (Ts prescribed), so one helper serves both.
    _, LE_Soil, H_Soil, _G = soil_energy_balance_bt(
        Ts, Tc, q_s, q_c,
        b.lam if b.lam_soil is None else b.lam_soil, b.rhoa, b.Cp,
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
    res_Ci_Sun = (anchor_weight * (Ci_Sun - Ci_Sh)
                  + (1.0 - anchor_weight) * (Ci_Sun - Ci_Sun_new))

    diff = jnp.array([
        res_Tf_Sun,
        Tf_Sh  - Tf_Sh_new,
        res_Ci_Sun,
        Ci_Sh  - Ci_Sh_new,
        Tc     - Tc_new,
        q_c   - q_c_new,
    ])

    # ---- Bare ground: no leaves, so no leaf equations ----
    # At LAI == 0 the leaf rows above still get solved, through boundary-layer
    # resistances of ~1e7 s/m.  Nothing then holds the leaf temperature: it
    # drifts ~100 K from the canopy air (measured median 96 K on the failing
    # columns) and the saturation humidity evaluated there makes the q_c row
    # non-smooth, so the damped Newton steps are rejected and the solve stalls
    # at the iteration cap.  Those columns were 90% of all non-converged solves
    # (10.7% -> 2.0% with the pins; gradient gate -0.21 -> 1.14).  Pin the empty
    # leaf state to the canopy air and ambient CO2; the Tc and q_c rows keep
    # their full balance, in which the leaf terms are already ~0 at LAI == 0.
    # The trigger is the leaf-area floor of the boundary-layer resistance: at
    # or below it both leaf classes are bit-identical to bare ground, so a
    # column with LAI = 1e-7 is pinned too, and no column with distinct leaf
    # physics is.
    bare = b.LAI <= LEAF_AREA_FLOOR
    pinned = jnp.array([
        Tf_Sun - Tc,
        Tf_Sh  - Tc,
        Ci_Sun - b.Ca,
        Ci_Sh  - b.Ca,
        diff[4],
        diff[5],
    ])
    return jnp.where(bare, pinned, diff)


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
    rh_cap_width: float,
    zeta_cap_width: float,
    most_n_iters: int,
) -> dict:
    """Evaluate the FULLY_COUPLED canopy state and return all fluxes.

    Used after the Newton solver has converged to extract final diagnostics.
    The returned ``G`` is the surface-energy-budget residual
    ``Rn_soil - LE_soil - H_soil``, intended to be passed as the top BC to
    ``solve_soil_thermal`` in the caller.
    """
    Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c = (
        x[0], x[1], x[2], x[3], x[4], x[5])

    b = bundle
    Ts = b.Ts_bc
    zldis = b.z0 - b.displa

    ustar, rah_above, raw_above, uav, zeta = monin_obukhov_stability(
        b.ur, b.Ta, b.Tv_atm, Tc, b.q_atm, q_c, zldis, b.z0m,
        n_iters=most_n_iters, zeta_cap_width=zeta_cap_width)

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
        b.Vcmax25_Sun, b.Vcmax25_C4Sun, b.fC4, b.Ps, b.alf, b.TgC)
    An_Sh, Agross_Sh = photosynthesis(
        T_phot_sh,  Ci_Sh,  b.APAR_Sh,
        b.Vcmax25_Sh, b.Vcmax25_C4Sh, b.fC4, b.Ps, b.alf, b.TgC)

    # FULLY_COUPLED: leaves and soil share the canopy air space (Tc, q_c).
    e_c, es_c, VPD_c, RH_c, desTc, ddesTc, gamma_c = canopy_met_variables(
        b.Ps, Tc, q_c, rh_cap_width)

    if LE_module == "BT":
        q_f_Sun = saturation_specific_humidity(Tf_Sun, b.Ps)
        q_f_Sh  = saturation_specific_humidity(Tf_Sh,  b.Ps)
        Rn_Sun, LE_Sun, H_Sun, _, gs_Sun, _ = leaf_energy_balance_bt(
            An_Sun, b.ASW_Sun, ALW_Sun, Tf_Sun, b.Ps, b.Ca,
            Tc, q_f_Sun, q_c, RH_c, VPD_c,
            b.lam, b.Cp, b.rhoa, Rb_Sun, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode)
        Rn_Sh,  LE_Sh,  H_Sh,  _, gs_Sh, _  = leaf_energy_balance_bt(
            An_Sh, b.ASW_Sh, ALW_Sh, Tf_Sh, b.Ps, b.Ca,
            Tc, q_f_Sh, q_c, RH_c, VPD_c,
            b.lam, b.Cp, b.rhoa, Rb_Sh, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode)
    else:
        Rn_Sun, LE_Sun, H_Sun, _, gs_Sun, _ = leaf_energy_balance_pm(
            An_Sun, b.ASW_Sun, ALW_Sun, Tf_Sun, b.Ps, b.Ca,
            Tc, VPD_c, RH_c, desTc, ddesTc, gamma_c,
            b.Cp, b.rhoa, Rb_Sun, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode)
        Rn_Sh,  LE_Sh,  H_Sh,  _, gs_Sh,  _ = leaf_energy_balance_pm(
            An_Sh, b.ASW_Sh, ALW_Sh, Tf_Sh, b.Ps, b.Ca,
            Tc, VPD_c, RH_c, desTc, ddesTc, gamma_c,
            b.Cp, b.rhoa, Rb_Sh, b.m, b.b0,
            fwet=b.fwet, stomatal_model=stomatal_model, le_cap_mode=le_cap_mode)

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
    Rn_Soil, LE_Soil, H_Soil, G = soil_energy_balance_bt(
        Ts, Tc, q_s, q_c,
        b.lam if b.lam_soil is None else b.lam_soil, b.rhoa, b.Cp,
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
    initial_state : shape (6,)
        [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c].  Soil skin T is passed
        as ``bundle.Ts_bc`` — see module docstring.
    bundle : CanopyForcingBundle
        All per-column forcing inputs (including ``Ts_bc``).
    config : CanopyConfig
        Solver settings (max_iters, tol, LE_module, stomatal_model).

    Returns
    -------
    x_final : shape (6,)  — the final Newton iterate
    n_iters : scalar int  — iteration count when the loop stopped
    converged : scalar bool — whether ``x_final`` is a certified root: the
        residual fell 1e8x below the seed's, below an absolute ceiling, and the
        state lies in the physical box (:func:`canopy_state_admissible`).  A
        False here means ``x_final`` is NOT a usable root, so the fluxes derived
        from it are not physics; callers must not consume them as if they were.
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
    if not 0.0 < config.rh_cap_smoothing_width <= RH_CAP_WIDTH_MAX:
        raise ValueError("rh_cap_smoothing_width must be in "
                         f"(0, {RH_CAP_WIDTH_MAX}] (the smooth relative-humidity "
                         f"cap divides by it), got {config.rh_cap_smoothing_width!r}")
    if not 0.0 < config.zeta_cap_smoothing_width <= ZETA_CAP_WIDTH_MAX:
        raise ValueError("zeta_cap_smoothing_width must be in "
                         f"(0, {ZETA_CAP_WIDTH_MAX}] (the smooth stability "
                         f"cap divides by it), got {config.zeta_cap_smoothing_width!r}")
    solver = _make_implicit_newton_solver(
        LE_module=config.LE_module,
        stomatal_model=config.stomatal_model,
        le_cap_mode=config.le_cap_mode,
        use_ta_for_photosynthesis=config.use_ta_for_photosynthesis,
        rh_cap_width=config.rh_cap_smoothing_width,
        zeta_cap_width=config.zeta_cap_smoothing_width,
        most_n_iters=config.most_n_iters,
        max_iters=config.max_iters,
        tol=config.tol,
    )
    return solver(initial_state, bundle)[:3]


def solve_canopy_closure_diag(
    initial_state: jax.Array,
    bundle: CanopyForcingBundle,
    config: CanopyConfig,
):
    """As :func:`solve_canopy_closure`, plus the solver's terminal diagnostics.

    Returns ``(x_final, n_iters, converged, n_sq_final, n_sq_rel, lam_final,
    hit_cap)``.  ``n_sq_final`` is the solver's OWN squared residual norm at
    exit; ``n_sq_rel`` is that divided by its value at the seed — the ratio the
    RELATIVE convergence gate tests.  ``lam_final`` is the terminal damping and
    ``hit_cap`` is 1.0 when the loop left by the iteration cap rather than by
    the damping ceiling.

    Diagnostics only: same solve, same convergence test, same gradients (the
    extra outputs carry zero cotangent, exactly like ``n_iters``).
    """
    if config.LE_module not in VALID_LE_MODULES:
        raise ValueError(
            f"unknown LE_module {config.LE_module!r}; the leaf-energy method "
            f"must be one of {VALID_LE_MODULES} ('BT'=bulk transfer, "
            "'PM'=Penman-Monteith)")
    if not 0.0 < config.rh_cap_smoothing_width <= RH_CAP_WIDTH_MAX:
        raise ValueError("rh_cap_smoothing_width must be in "
                         f"(0, {RH_CAP_WIDTH_MAX}] (the smooth relative-humidity "
                         f"cap divides by it), got {config.rh_cap_smoothing_width!r}")
    if not 0.0 < config.zeta_cap_smoothing_width <= ZETA_CAP_WIDTH_MAX:
        raise ValueError("zeta_cap_smoothing_width must be in "
                         f"(0, {ZETA_CAP_WIDTH_MAX}] (the smooth stability "
                         f"cap divides by it), got {config.zeta_cap_smoothing_width!r}")
    solver = _make_implicit_newton_solver(
        LE_module=config.LE_module,
        stomatal_model=config.stomatal_model,
        le_cap_mode=config.le_cap_mode,
        use_ta_for_photosynthesis=config.use_ta_for_photosynthesis,
        rh_cap_width=config.rh_cap_smoothing_width,
        zeta_cap_width=config.zeta_cap_smoothing_width,
        most_n_iters=config.most_n_iters,
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
    rh_cap_width: float,
    zeta_cap_width: float,
    most_n_iters: int,
    max_iters: int,
    tol: float,
):
    """Bind the shared implicit root solver to the canopy residual."""
    del tol  # Legacy step tolerance; convergence is the residual contract.

    def residual(x, bundle):
        return _canopy_residual(
            x, bundle,
            LE_module=LE_module,
            stomatal_model=stomatal_model,
            le_cap_mode=le_cap_mode,
            use_ta_for_photosynthesis=use_ta_for_photosynthesis,
            rh_cap_width=rh_cap_width,
            zeta_cap_width=zeta_cap_width,
            most_n_iters=most_n_iters,
        )

    return make_implicit_newton_solver(
        residual,
        x_scale=_LM_XSCALE,
        f_scale=_LM_FSCALE,
        max_iters=max_iters,
        rtol=_LM_RTOL,
        atol=_LM_ATOL,
        n_sq_max=_ROOT_NSQ_MAX,
        admissible=canopy_state_admissible,
    )
