"""Monin-Obukhov bulk air-sea flux algorithms.

Provides stability-dependent transfer coefficients using iterative
Monin-Obukhov similarity theory (MOST). Three schemes:

1. ``constant`` — Fixed neutral transfer coefficients (no iteration)
2. ``coare3`` — COARE 3.0 (Fairall et al. 2003): Charnock + smooth-flow
   roughness, Businger-Dyer stability functions
3. ``large_yeager`` — Large & Yeager 2004 (CORE): empirical C_DN(U_10N),
   stability-dependent Stanton/Dalton number

The iterative Obukhov length loop uses ``jax.lax.fori_loop`` for
full JAX differentiability (compatible with jax.grad, jax.jit).

References
----------
- Fairall, C. W., et al. (2003). Bulk parameterization of air-sea fluxes:
  Updates and verification for the COARE algorithm. J. Climate, 16, 571-591.
- Large, W. G., & Yeager, S. G. (2004). Diurnal to decadal global forcing
  for ocean and sea-ice models. NCAR Tech. Note, NCAR/TN-460+STR.
- Businger, J. A., et al. (1971). Flux-profile relationships in the
  atmospheric surface layer. J. Atmos. Sci., 28, 181-189.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants

# Physical constants
KAPPA = constants.kappa_vk  # von Kármán constant (0.4)
G = constants.g
NU_AIR = 1.5e-5  # kinematic viscosity of air [m²/s]


# ============================================================================
# Stability functions (Businger-Dyer)
# ============================================================================

def psi_m(zeta):
    """MOST momentum stability function.

    Unstable (ζ < 0): Businger-Dyer
        ψ_m = 2 ln((1+x)/2) + ln((1+x²)/2) − 2 arctan(x) + π/2
        where x = (1 − 16ζ)^{1/4}
    Stable (ζ > 0): Dyer (1974)
        ψ_m = −5ζ

    Uses safe branching (min/max on inputs) to avoid NaN gradients
    in the inactive branch.
    """
    zeta_c = jnp.clip(zeta, -10.0, 10.0)
    # Safe inputs: each branch only sees valid arguments
    zeta_neg = jnp.minimum(zeta_c, -1e-10)
    zeta_pos = jnp.maximum(zeta_c, 1e-10)

    x = jnp.power(1.0 - 16.0 * zeta_neg, 0.25)
    unstable = (
        2.0 * jnp.log((1.0 + x) / 2.0)
        + jnp.log((1.0 + x ** 2) / 2.0)
        - 2.0 * jnp.arctan(x)
        + jnp.pi / 2.0
    )
    stable = -5.0 * zeta_pos

    return jnp.where(zeta_c < 0.0, unstable, stable)


def psi_h(zeta):
    """MOST heat/moisture stability function.

    Unstable (ζ < 0): Businger-Dyer
        ψ_h = 2 ln((1+y)/2)  where y = (1 − 16ζ)^{1/2}
    Stable (ζ > 0): Dyer (1974)
        ψ_h = −5ζ
    """
    zeta_c = jnp.clip(zeta, -10.0, 10.0)
    zeta_neg = jnp.minimum(zeta_c, -1e-10)
    zeta_pos = jnp.maximum(zeta_c, 1e-10)

    y = jnp.sqrt(1.0 - 16.0 * zeta_neg)
    unstable = 2.0 * jnp.log((1.0 + y) / 2.0)
    stable = -5.0 * zeta_pos

    return jnp.where(zeta_c < 0.0, unstable, stable)


# ============================================================================
# Main MOST flux computation
# ============================================================================

def compute_most_fluxes(
    u_rel,
    v_rel,
    T_atm,
    q_atm,
    T_sfc,
    q_sfc,
    rho,
    z_ref=10.0,
    z0_init=1e-4,
    scheme="coare3",
    n_iter=5,
    charnock=0.011,
):
    """Compute stability-dependent bulk fluxes via iterative MOST.

    Uses ``jax.lax.fori_loop`` for the Obukhov length iteration,
    ensuring full JAX differentiability (jax.grad, jax.jit).

    Parameters
    ----------
    u_rel, v_rel : array
        Wind components relative to surface [m/s].
    T_atm : array
        Atmospheric temperature at reference height [K].
    q_atm : array
        Atmospheric specific humidity at reference height [kg/kg].
    T_sfc : array
        Surface temperature [K].
    q_sfc : array
        Surface saturation specific humidity [kg/kg].
    rho : array
        Air density at reference height [kg/m³].
    z_ref : float
        Reference height for bulk formulas [m] (default 10).
    z0_init : float
        Initial momentum roughness length [m] (default 1e-4).
    scheme : str
        ``"coare3"`` or ``"large_yeager"``.
    n_iter : int
        Number of MOST iterations (default 5).
    charnock : float
        Charnock coefficient (COARE only, default 0.011).

    Returns
    -------
    tau_x, tau_y : array
        Surface stress [Pa] (opposes wind direction).
    shflx : array
        Sensible heat flux [W/m²] (positive upward = surface warmer).
    lhflx : array
        Latent heat flux [W/m²] (positive upward = surface moister).
    ustar : array
        Friction velocity [m/s].
    """
    wind_speed = jnp.sqrt(u_rel ** 2 + v_rel ** 2 + 1e-4)
    dT = T_sfc - T_atm
    dq = q_sfc - q_atm
    T_v = T_atm * (1.0 + 0.61 * q_atm)

    # Initialize with neutral log-law profile
    z0 = jnp.full_like(wind_speed, z0_init)
    z0_t = z0 * 0.1
    z0_q = z0_t

    ln_z_z0 = jnp.log(z_ref / jnp.maximum(z0, 1e-12))
    denom_init = jnp.maximum(ln_z_z0, 0.5)
    u_star = KAPPA * wind_speed / denom_init
    theta_star = KAPPA * dT / denom_init
    q_star_val = KAPPA * dq / denom_init

    carry = (u_star, z0, z0_t, z0_q, theta_star, q_star_val)

    def body_fn(i, carry):
        u_star, z0, z0_t, z0_q, theta_star, q_star_val = carry
        u_star_safe = jnp.maximum(u_star, 1e-6)

        # Virtual potential temperature scale
        theta_v_star = theta_star + 0.61 * T_atm * q_star_val

        # Obukhov length: L = −u*² T_v / (κ g θ_v*)
        L_denom = KAPPA * G * theta_v_star
        L = jnp.where(
            jnp.abs(L_denom) > 1e-10,
            -u_star_safe ** 2 * T_v / L_denom,
            jnp.where(L_denom > 0.0, -1e6, 1e6),
        )

        zeta = jnp.clip(z_ref / L, -10.0, 10.0)
        psi_m_val = psi_m(zeta)
        psi_h_val = psi_h(zeta)

        # --- Roughness update (Python if resolved at trace time) ---
        if scheme == "coare3":
            # COARE 3.0: Charnock + smooth-flow regime
            z0_new = (
                charnock * u_star_safe ** 2 / G
                + 0.11 * NU_AIR / u_star_safe
            )
            Re = u_star_safe * z0_new / NU_AIR
            z0_t_new = jnp.minimum(
                1.15e-4,
                5.5e-5 * jnp.power(jnp.maximum(Re, 0.01), -0.6),
            )
            z0_q_new = z0_t_new

        elif scheme == "large_yeager":
            # Large & Yeager 2004: empirical C_DN(U_10N)
            U_10N = u_star_safe / KAPPA * jnp.log(
                10.0 / jnp.maximum(z0, 1e-12)
            )
            U_10N = jnp.clip(U_10N, 0.5, 50.0)

            C_DN = (2.7 / U_10N + 0.142 + 0.0764 * U_10N) * 1e-3
            C_DN = jnp.clip(C_DN, 0.5e-3, 3.0e-3)

            z0_new = 10.0 / jnp.exp(KAPPA / jnp.sqrt(C_DN))

            # Stability-dependent Stanton / Dalton coefficient
            ch_coeff = jnp.where(zeta < 0.0, 32.7e-3, 18.0e-3)
            ce_coeff = jnp.where(zeta < 0.0, 34.6e-3, 34.6e-3)
            z0_t_new = 10.0 / jnp.exp(KAPPA / ch_coeff)
            z0_q_new = 10.0 / jnp.exp(KAPPA / ce_coeff)

        else:
            z0_new = z0
            z0_t_new = z0_t
            z0_q_new = z0_q

        # Transfer coefficients with stability correction
        ln_z_z0 = jnp.log(z_ref / jnp.maximum(z0_new, 1e-12))
        ln_z_z0t = jnp.log(z_ref / jnp.maximum(z0_t_new, 1e-12))
        ln_z_z0q = jnp.log(z_ref / jnp.maximum(z0_q_new, 1e-12))

        denom_m = jnp.maximum(ln_z_z0 - psi_m_val, 0.5)
        denom_h = jnp.maximum(ln_z_z0t - psi_h_val, 0.5)
        denom_q = jnp.maximum(ln_z_z0q - psi_h_val, 0.5)

        u_star_new = KAPPA * wind_speed / denom_m
        theta_star_new = KAPPA * dT / denom_h
        q_star_new = KAPPA * dq / denom_q

        return (u_star_new, z0_new, z0_t_new, z0_q_new,
                theta_star_new, q_star_new)

    carry = jax.lax.fori_loop(0, n_iter, body_fn, carry)
    u_star, z0, z0_t, z0_q, theta_star, q_star_val = carry

    # Fluxes from scaling parameters
    tau_x = -rho * u_star ** 2 * u_rel / wind_speed
    tau_y = -rho * u_star ** 2 * v_rel / wind_speed
    shflx = rho * constants.c_pd * u_star * theta_star
    lhflx = rho * constants.L_v * u_star * q_star_val

    return tau_x, tau_y, shflx, lhflx, u_star
