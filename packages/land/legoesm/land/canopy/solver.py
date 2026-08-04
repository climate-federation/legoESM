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
from legoesm.land.canopy.stability import (
    monin_obukhov_stability,
    compute_boundary_layer_resistance,
    compute_below_canopy_resistance,
    sat_specific_humidity,
)
from legoesm.land.canopy.energy_balance import (
    canopy_met_variables,
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
        b.Vcmax25_Sun, b.Vcmax25_C4Sun, b.fC4, b.Ps, b.alf, b.TgC)
    An_Sh, _ = photosynthesis(
        T_phot_sh,  Ci_Sh,  b.APAR_Sh,
        b.Vcmax25_Sh, b.Vcmax25_C4Sh, b.fC4, b.Ps, b.alf, b.TgC)

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
        (q_c   - q_c_new) * 1e3,   # scale humidity residual
    ])
    return diff


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
        b.Vcmax25_Sun, b.Vcmax25_C4Sun, b.fC4, b.Ps, b.alf, b.TgC)
    An_Sh, Agross_Sh = photosynthesis(
        T_phot_sh,  Ci_Sh,  b.APAR_Sh,
        b.Vcmax25_Sh, b.Vcmax25_C4Sh, b.fC4, b.Ps, b.alf, b.TgC)

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
    x_final : shape (6,)  — converged state
    n_iters : scalar int  — iteration count when convergence was first reached
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
        F = partial(_F, bundle=bundle)
        Jac = jax.jacfwd(F)

        def cond(state):
            _, i, converged = state
            return (~converged) & (i < max_iters)

        def body(state):
            x, i, _ = state
            delta = jnp.linalg.solve(Jac(x), -F(x))
            # Sanitise a non-finite Newton step BEFORE the clip: a singular
            # Jacobian / non-finite residual at the calm cold-night MOST edge
            # makes ``solve`` return NaN, and ``jnp.clip(nan)`` stays NaN — a
            # single non-finite step then poisons the whole coupled land state
            # (every soil leaf goes NaN, forcing the atomic land guard to revert
            # the step).  Replace it with 0 (no update this iteration) so the
            # closure returns the last finite iterate instead, keeping the flux
            # finite.  inf is already mapped to +/-5 by the clip below.
            delta = jnp.where(jnp.isfinite(delta), delta, 0.0)
            # Constant scalar clamp on the Newton step.  The earlier
            # decaying clamp (10 → 0.1) starved late iterations of step
            # size during dusk transitions; constant 5.0 lets the solver
            # traverse the radiation-collapse smoothly while still
            # preventing catastrophic overshoot.
            delta = jnp.clip(delta, -5.0, 5.0)  # coeff-ok: constant Newton step cap (see note above)
            x_new = x + delta
            new_converged = jnp.linalg.norm(delta) < tol
            return (x_new, i + 1, new_converged)

        x_final, n_iters, converged = jax.lax.while_loop(
            cond, body, (x0, jnp.array(0), jnp.array(False)))
        return x_final, n_iters, converged

    @jax.custom_vjp
    def solve(x0, bundle):
        x_final, n_iters, _converged = _forward(x0, bundle)
        return x_final, n_iters

    def solve_fwd(x0, bundle):
        x_final, n_iters, converged = _forward(x0, bundle)
        return (x_final, n_iters), (x_final, bundle, converged)

    def solve_bwd(res, g):
        x_star, bundle, converged = res
        g_x, _ = g  # ignore cotangent for the integer iteration count

        # NaN / divergence guard: substitute zeros so the adjoint solve
        # is well-defined even if Newton diverged.
        x_safe = jnp.where(jnp.isnan(x_star), jnp.zeros_like(x_star), x_star)
        had_nan = jnp.any(jnp.isnan(x_star)) | jnp.any(jnp.isnan(g_x))
        # Convergence guard: the IFT adjoint dx*/dθ = -(∂F/∂x)^{-1} ∂F/∂θ is
        # only valid at a true root F(x*) = 0.  If the forward Newton hit
        # max_iters without ||Δx|| < tol, x* is not a root and the adjoint is
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
