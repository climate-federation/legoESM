"""Newton-Raphson canopy closure solver.

Solves the 7-variable canopy energy balance system:
  x = [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Ts, Tc, q_c]

The residual function F(x) is defined by the physical consistency conditions
that leaf temperatures, intercellular CO2, soil temperature, and canopy air
state must simultaneously satisfy the photosynthesis, energy balance, and
aerodynamic equations.

Uses jax.lax.scan for JIT-compatible fixed-point iteration with a forward-mode
autodiff Jacobian (jax.jacfwd).  The coupling_scheme and LE_module are captured
as static Python strings in a functools.partial closure — they are never traced.

Source: adapted from DifferBESS/algo/newton_root.py and
        DifferBESS/process/vector_state_variable_difference.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

from legoesm.land.canopy.config import CanopyConfig
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
    G_alpha: jax.Array

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
    m: jax.Array         # Ball-Berry slope (stress-applied)
    b0: jax.Array        # Ball-Berry intercept (stress-applied)
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


# ---------------------------------------------------------------------------
# Residual function (defines the closure equations)
# ---------------------------------------------------------------------------

def _canopy_residual(
    x: jax.Array,
    bundle: CanopyForcingBundle,
    coupling_scheme: str,
    LE_module: str,
    use_ta_for_photosynthesis: bool,
) -> jax.Array:
    """Compute the residual vector F(x) for the canopy closure.

    x = [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Ts, Tc, q_c]

    Returns F such that F(x*) = 0 at the solution.
    """
    Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Ts, Tc, q_c = (
        x[0], x[1], x[2], x[3], x[4], x[5], x[6])

    b = bundle
    zldis = b.z0 - b.displa

    # ---- MOST stability ----
    ustar, rah_above, raw_above, uav, _ = monin_obukhov_stability(
        b.ur, b.Ta, b.Tv_atm, Tc, b.q_atm, q_c, zldis, b.z0m)

    # ---- Boundary and below-canopy resistances ----
    Rb_Sun, Rb_Sh = compute_boundary_layer_resistance(uav, b.LAI, b.fSun)
    rah_below, raw_below = compute_below_canopy_resistance(uav, b.CI, b.LAI)

    # ---- Soil evaporation resistance ----
    Rsoil = raw_below * (1.0 / jnp.maximum(b.fStress_soil, 1e-6) - 1.0)

    # ---- Longwave radiation ----
    Tf_mean = (Tf_Sun**4 * b.fSun + Tf_Sh**4 * (1.0 - b.fSun))**0.25
    lw_out  = canopy_longwave_rt(
        b.LAI, b.SZA, Ts, Tf_mean, Tf_Sun, Tf_Sh, b.La, b.epsf, b.epss)
    ALW_Sun, ALW_Sh, ALW_Soil = lw_out.ALW_Sun, lw_out.ALW_Sh, lw_out.ALW_Soil

    # ---- Photosynthesis ----
    T_phot_sun = b.Ta if use_ta_for_photosynthesis else Tf_Sun
    T_phot_sh  = b.Ta if use_ta_for_photosynthesis else Tf_Sh
    An_Sun = photosynthesis(
        T_phot_sun, Ci_Sun, b.APAR_Sun,
        b.Vcmax25_Sun, b.Vcmax25_C4Sun, b.fC4, b.Ps, b.alf, b.TgC)
    An_Sh = photosynthesis(
        T_phot_sh,  Ci_Sh,  b.APAR_Sh,
        b.Vcmax25_Sh, b.Vcmax25_C4Sh, b.fC4, b.Ps, b.alf, b.TgC)

    # ---- Leaf microclimate (scheme-dependent) ----
    if coupling_scheme == "LEAVES_ATMO":
        Tc_leaf = b.Ta
        q_c_leaf = b.q_atm
    else:
        Tc_leaf = Tc
        q_c_leaf = q_c

    e_c, es_c, VPD_c, RH_c, desTc, ddesTc, gamma_c = canopy_met_variables(
        b.Ps, Tc_leaf, q_c_leaf)

    # ---- Leaf energy balance ----
    if LE_module == "BT":
        q_f_Sun = saturation_specific_humidity(Tf_Sun, b.Ps)
        q_f_Sh  = saturation_specific_humidity(Tf_Sh,  b.Ps)
        _, LE_Sun, H_Sun, Tf_Sun_new, gs_Sun, Ci_Sun_new = leaf_energy_balance_bt(
            An_Sun, b.ASW_Sun, ALW_Sun, Tf_Sun, b.Ps, b.Ca,
            Tc_leaf, q_f_Sun, q_c_leaf, RH_c,
            b.lam, b.Cp, b.rhoa, Rb_Sun, b.m, b.b0)
        _, LE_Sh, H_Sh, Tf_Sh_new, gs_Sh, Ci_Sh_new = leaf_energy_balance_bt(
            An_Sh, b.ASW_Sh, ALW_Sh, Tf_Sh, b.Ps, b.Ca,
            Tc_leaf, q_f_Sh, q_c_leaf, RH_c,
            b.lam, b.Cp, b.rhoa, Rb_Sh, b.m, b.b0)
    else:  # PM
        _, LE_Sun, H_Sun, Tf_Sun_new, gs_Sun, Ci_Sun_new = leaf_energy_balance_pm(
            An_Sun, b.ASW_Sun, ALW_Sun, Tf_Sun, b.Ps, b.Ca,
            Tc_leaf, VPD_c, RH_c, desTc, ddesTc, gamma_c,
            b.Cp, b.rhoa, Rb_Sun, b.m, b.b0)
        _, LE_Sh, H_Sh, Tf_Sh_new, gs_Sh, Ci_Sh_new = leaf_energy_balance_pm(
            An_Sh, b.ASW_Sh, ALW_Sh, Tf_Sh, b.Ps, b.Ca,
            Tc_leaf, VPD_c, RH_c, desTc, ddesTc, gamma_c,
            b.Cp, b.rhoa, Rb_Sh, b.m, b.b0)

    # ---- Soil energy balance ----
    if coupling_scheme == "VEG_ONLY":
        Tc_soil   = b.Ta
        q_c_soil  = b.q_atm
        rah_s     = rah_above + 1.0 / jnp.maximum(
            0.13 / 0.4 * (b.z0m * uav / 1.5e-5)**0.45 * uav, 1e-9)
        raw_s     = rah_s
        Rsoil_veg = raw_s * (1.0 / jnp.maximum(b.fStress_soil, 1e-6) - 1.0)
    else:
        Tc_soil   = Tc
        q_c_soil  = q_c
        rah_s     = rah_below
        raw_s     = raw_below
        Rsoil_veg = Rsoil

    if LE_module == "BT":
        q_s = saturation_specific_humidity(Ts, b.Ps)
        _, LE_Soil, H_Soil, Ts_new, G = soil_energy_balance_bt(
            Ts, Tc_soil, q_s, q_c_soil,
            b.lam, b.rhoa, b.Cp,
            rah_s, raw_s, Rsoil_veg,
            b.ASW_Soil, ALW_Soil, b.G_alpha)
    else:
        e_, es_, VPD_s, RH_s, des_s, ddes_s, gam_s = canopy_met_variables(
            b.Ps, Tc_soil, q_c_soil)
        _, LE_Soil, H_Soil, Ts_new, G = soil_energy_balance_pm(
            Ts, Tc_soil, VPD_s, des_s, ddes_s, gam_s, b.rhoa, b.Cp,
            rah_s, raw_s, Rsoil_veg,
            b.ASW_Soil, ALW_Soil, b.G_alpha)

    # ---- Canopy air update ----
    Tc_new, q_c_new = canopy_air_update(
        b.Ta, b.q_atm,
        Tf_Sun_new, Tf_Sh_new, Ts_new,
        gs_Sun, gs_Sh,
        Rb_Sun, Rb_Sh,
        rah_above, raw_above,
        rah_below, raw_below,
        Rsoil_veg, b.Ps, coupling_scheme)

    # ---- Residuals ----
    diff = jnp.array([
        Tf_Sun - Tf_Sun_new,
        Tf_Sh  - Tf_Sh_new,
        Ci_Sun - Ci_Sun_new,
        Ci_Sh  - Ci_Sh_new,
        Ts     - Ts_new,
        Tc     - Tc_new,
        (q_c   - q_c_new) * 1e3,   # scale humidity residual
    ])
    return diff


# ---------------------------------------------------------------------------
# Diagnostic forward pass — same logic but returns all fluxes
# ---------------------------------------------------------------------------

def _canopy_forward(
    x: jax.Array,
    bundle: CanopyForcingBundle,
    coupling_scheme: str,
    LE_module: str,
    use_ta_for_photosynthesis: bool,
) -> dict:
    """Evaluate the canopy state and return all fluxes (no residual).

    Used after the solver has converged to extract final diagnostics.
    """
    Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Ts, Tc, q_c = (
        x[0], x[1], x[2], x[3], x[4], x[5], x[6])

    b = bundle
    zldis = b.z0 - b.displa

    ustar, rah_above, raw_above, uav, zeta = monin_obukhov_stability(
        b.ur, b.Ta, b.Tv_atm, Tc, b.q_atm, q_c, zldis, b.z0m)

    Rb_Sun, Rb_Sh = compute_boundary_layer_resistance(uav, b.LAI, b.fSun)
    rah_below, raw_below = compute_below_canopy_resistance(uav, b.CI, b.LAI)
    Rsoil = raw_below * (1.0 / jnp.maximum(b.fStress_soil, 1e-6) - 1.0)

    Tf_mean = (Tf_Sun**4 * b.fSun + Tf_Sh**4 * (1.0 - b.fSun))**0.25
    lw_out  = canopy_longwave_rt(
        b.LAI, b.SZA, Ts, Tf_mean, Tf_Sun, Tf_Sh, b.La, b.epsf, b.epss)
    ALW_Sun, ALW_Sh, ALW_Soil = lw_out.ALW_Sun, lw_out.ALW_Sh, lw_out.ALW_Soil
    Ls = lw_out.Ls

    T_phot_sun = b.Ta if use_ta_for_photosynthesis else Tf_Sun
    T_phot_sh  = b.Ta if use_ta_for_photosynthesis else Tf_Sh
    An_Sun = photosynthesis(
        T_phot_sun, Ci_Sun, b.APAR_Sun,
        b.Vcmax25_Sun, b.Vcmax25_C4Sun, b.fC4, b.Ps, b.alf, b.TgC)
    An_Sh = photosynthesis(
        T_phot_sh,  Ci_Sh,  b.APAR_Sh,
        b.Vcmax25_Sh, b.Vcmax25_C4Sh, b.fC4, b.Ps, b.alf, b.TgC)

    if coupling_scheme == "LEAVES_ATMO":
        Tc_leaf, q_c_leaf = b.Ta, b.q_atm
    else:
        Tc_leaf, q_c_leaf = Tc, q_c

    e_c, es_c, VPD_c, RH_c, desTc, ddesTc, gamma_c = canopy_met_variables(
        b.Ps, Tc_leaf, q_c_leaf)

    if LE_module == "BT":
        q_f_Sun = saturation_specific_humidity(Tf_Sun, b.Ps)
        q_f_Sh  = saturation_specific_humidity(Tf_Sh,  b.Ps)
        Rn_Sun, LE_Sun, H_Sun, _, gs_Sun, _ = leaf_energy_balance_bt(
            An_Sun, b.ASW_Sun, ALW_Sun, Tf_Sun, b.Ps, b.Ca,
            Tc_leaf, q_f_Sun, q_c_leaf, RH_c,
            b.lam, b.Cp, b.rhoa, Rb_Sun, b.m, b.b0)
        Rn_Sh,  LE_Sh,  H_Sh,  _, gs_Sh, _  = leaf_energy_balance_bt(
            An_Sh, b.ASW_Sh, ALW_Sh, Tf_Sh, b.Ps, b.Ca,
            Tc_leaf, q_f_Sh, q_c_leaf, RH_c,
            b.lam, b.Cp, b.rhoa, Rb_Sh, b.m, b.b0)
    else:
        Rn_Sun, LE_Sun, H_Sun, _, gs_Sun, _ = leaf_energy_balance_pm(
            An_Sun, b.ASW_Sun, ALW_Sun, Tf_Sun, b.Ps, b.Ca,
            Tc_leaf, VPD_c, RH_c, desTc, ddesTc, gamma_c,
            b.Cp, b.rhoa, Rb_Sun, b.m, b.b0)
        Rn_Sh,  LE_Sh,  H_Sh,  _, gs_Sh,  _ = leaf_energy_balance_pm(
            An_Sh, b.ASW_Sh, ALW_Sh, Tf_Sh, b.Ps, b.Ca,
            Tc_leaf, VPD_c, RH_c, desTc, ddesTc, gamma_c,
            b.Cp, b.rhoa, Rb_Sh, b.m, b.b0)

    if coupling_scheme == "VEG_ONLY":
        Tc_soil, q_c_soil = b.Ta, b.q_atm
        nu       = 1.5e-5
        Csbare   = 0.4 / 0.13 * (b.z0m * jnp.maximum(uav, 1e-3) / nu)**(-0.45)
        rah_s    = rah_above + 1.0 / jnp.maximum(Csbare * uav, 1e-9)
        raw_s    = rah_s
        Rsoil_s  = raw_s * (1.0 / jnp.maximum(b.fStress_soil, 1e-6) - 1.0)
    else:
        Tc_soil, q_c_soil = Tc, q_c
        rah_s, raw_s = rah_below, raw_below
        Rsoil_s = Rsoil

    if LE_module == "BT":
        q_s = saturation_specific_humidity(Ts, b.Ps)
        Rn_Soil, LE_Soil, H_Soil, _, G = soil_energy_balance_bt(
            Ts, Tc_soil, q_s, q_c_soil,
            b.lam, b.rhoa, b.Cp,
            rah_s, raw_s, Rsoil_s,
            b.ASW_Soil, ALW_Soil, b.G_alpha)
    else:
        _, _, VPD_s, _, des_s, ddes_s, gam_s = canopy_met_variables(
            b.Ps, Tc_soil, q_c_soil)
        Rn_Soil, LE_Soil, H_Soil, _, G = soil_energy_balance_pm(
            Ts, Tc_soil, VPD_s, des_s, ddes_s, gam_s, b.rhoa, b.Cp,
            rah_s, raw_s, Rsoil_s,
            b.ASW_Soil, ALW_Soil, b.G_alpha)

    return dict(
        An_Sun=An_Sun, An_Sh=An_Sh,
        LE_Sun=LE_Sun, LE_Sh=LE_Sh, LE_Soil=LE_Soil,
        H_Sun=H_Sun,   H_Sh=H_Sh,   H_Soil=H_Soil,
        Rn_Sun=Rn_Sun, Rn_Sh=Rn_Sh, Rn_Soil=Rn_Soil,
        G=G, Ls=Ls,
        gs_Sun=gs_Sun, gs_Sh=gs_Sh,
        ustar=ustar, zeta=zeta,
        rah_above=rah_above, rah_below=rah_below,
        Rb_Sun=Rb_Sun, Rb_Sh=Rb_Sh,
        ALW_Sun=ALW_Sun, ALW_Sh=ALW_Sh, ALW_Soil=ALW_Soil,
        Rsoil=Rsoil,
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
    initial_state : shape (7,)
        [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Ts, Tc, q_c]
    bundle : CanopyForcingBundle
        All per-column forcing inputs.
    config : CanopyConfig
        Solver settings (max_iters, tol, coupling_scheme, LE_module).

    Returns
    -------
    x_final : shape (7,)  — converged state
    n_iters : scalar int  — iteration count when convergence was first reached
    """
    # Bind static Python strings into the residual closure — not traced
    F = partial(
        _canopy_residual,
        bundle=bundle,
        coupling_scheme=config.coupling_scheme,
        LE_module=config.LE_module,
        use_ta_for_photosynthesis=config.use_ta_for_photosynthesis,
    )
    J_F = jax.jacfwd(F)

    max_iters = config.max_iters
    tol       = config.tol

    def body(carry, _):
        x, i, converged, stop_iter = carry

        def update(_):
            F_val  = F(x)
            J_val  = J_F(x)
            delta_x = jnp.linalg.solve(J_val, -F_val)
            return delta_x

        def no_update(_):
            return jnp.zeros_like(x)

        delta_x = jax.lax.cond(converged, no_update, update, operand=None)

        # Damping: wide window early, tighter near convergence
        clamp = 10.0 - 9.9 * i / max_iters
        x_new = x + jnp.clip(delta_x, -clamp, clamp)

        new_converged = jnp.linalg.norm(delta_x) < tol
        stop_iter = jax.lax.cond(
            new_converged & (~converged),
            lambda: i + 1,
            lambda: stop_iter,
        )
        return (x_new, i + 1, new_converged, stop_iter), None

    init = (initial_state, jnp.array(0), jnp.array(False), jnp.array(max_iters))
    (x_final, _, _, n_iters), _ = jax.lax.scan(body, init, xs=None, length=max_iters)
    return x_final, n_iters
