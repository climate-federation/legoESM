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
    L_latent=None,
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
    # Virtual-T moisture coefficient = 1/ε − 1 ≈ 0.6078 (canonical, not 0.61).
    _vT_coef = 1.0 / constants.epsilon - 1.0
    T_v = T_atm * (1.0 + _vT_coef * q_atm)

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

        # Virtual potential temperature scale (1/ε − 1 ≈ 0.6078)
        theta_v_star = theta_star + _vT_coef * T_atm * q_star_val

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
            # Large & Yeager 2004 (CORE): iterate in coefficient space.
            # 1) Neutral 10-m wind from current u_star and z0
            U_10N = u_star_safe / KAPPA * jnp.log(
                10.0 / jnp.maximum(z0, 1e-12)
            )
            U_10N = jnp.clip(U_10N, 0.5, 50.0)

            # 2) Empirical neutral 10-m drag coefficient C_DN(U_10N)
            C_DN = (2.7 / U_10N + 0.142 + 0.0764 * U_10N) * 1e-3
            C_DN = jnp.clip(C_DN, 0.5e-3, 3.0e-3)

            # 3) Neutral exchange coefficients at 10 m
            rdn = jnp.sqrt(C_DN)
            # Stability-dependent 10-m Stanton/Dalton number
            CHN10 = jnp.where(zeta < 0.0, 32.7e-3, 18.0e-3) * rdn
            CEN10 = 34.6e-3 * rdn
            rhn = CHN10 / rdn  # = ch_coeff
            ren = CEN10 / rdn  # = ce_coeff

            # 4) Shift coefficients from 10 m to measurement height z_ref
            #    with stability corrections (LY04 Eq. 9-11):
            #    rd = rdn / (1 + rdn/kappa * (ln(z_ref/10) - psi_m))
            #    rh = rhn / (1 + rhn/kappa * (ln(z_ref/10) - psi_h))
            ln_z_ratio = jnp.log(z_ref / 10.0)
            zeta_10 = jnp.clip(10.0 / L, -10.0, 10.0)
            psi_m_10 = psi_m(zeta_10)
            psi_h_10 = psi_h(zeta_10)
            # Stability correction difference between z_ref and 10 m
            dpsi_m = psi_m_val - psi_m_10
            dpsi_h = psi_h_val - psi_h_10

            rd = rdn / jnp.maximum(
                1.0 + rdn / KAPPA * (ln_z_ratio - dpsi_m), 0.2
            )
            rh = rhn / jnp.maximum(
                1.0 + rhn / KAPPA * (ln_z_ratio - dpsi_h), 0.2
            )
            re = ren / jnp.maximum(
                1.0 + ren / KAPPA * (ln_z_ratio - dpsi_h), 0.2
            )

            # 5) Update scaling parameters directly from coefficients
            u_star_new = rd * wind_speed
            theta_star_new = rh * dT
            q_star_new = re * dq

            # Still need z0 for the next iteration's U_10N estimate
            z0_new = z_ref / jnp.exp(KAPPA / rd + psi_m_val)
            z0_new = jnp.clip(z0_new, 1e-12, 1.0)
            z0_t_new = z0_t  # not used in coefficient path
            z0_q_new = z0_q  # not used in coefficient path

            return (u_star_new, z0_new, z0_t_new, z0_q_new,
                    theta_star_new, q_star_new)

        else:
            z0_new = z0
            z0_t_new = z0_t
            z0_q_new = z0_q

        # For COARE and constant: transfer coefficients via log-law + stability
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
    _L = constants.L_v if L_latent is None else L_latent
    tau_x = -rho * u_star ** 2 * u_rel / wind_speed
    tau_y = -rho * u_star ** 2 * v_rel / wind_speed
    shflx = rho * constants.c_pd * u_star * theta_star
    lhflx = rho * _L * u_star * q_star_val

    return tau_x, tau_y, shflx, lhflx, u_star


def simple_bulk_fluxes(
    u_lowest: jnp.ndarray,
    v_lowest: jnp.ndarray,
    T_lowest: jnp.ndarray,
    q_lowest: jnp.ndarray,
    T_sfc: jnp.ndarray,
    q_sfc: jnp.ndarray,
    rho: jnp.ndarray,
    wind_speed: jnp.ndarray,
    Cd: float,
    Ch: float,
    L_latent: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Simple bulk-aerodynamic surface fluxes with constant coefficients.

    Parameters
    ----------
    u_lowest, v_lowest : array
        Lowest-level wind components [m/s].
    T_lowest : array
        Lowest-level air temperature [K].
    q_lowest : array
        Lowest-level specific humidity [kg/kg].
    T_sfc : array
        Surface temperature [K].
    q_sfc : array
        Surface specific humidity [kg/kg].
    rho : array
        Air density at lowest level [kg/m3].
    wind_speed : array
        Wind speed (with minimum floor applied) [m/s].
    Cd : float
        Drag coefficient for momentum.
    Ch : float
        Transfer coefficient for heat and moisture.

    Returns
    -------
    tau_x, tau_y : array
        Surface stress [Pa] (opposes wind).
    shflx : array
        Sensible heat flux [W/m2] (positive upward = surface warmer).
    lhflx : array
        Latent heat flux [W/m2] (positive upward = surface moister).
    """
    _L = constants.L_v if L_latent is None else L_latent
    tau_x = -rho * Cd * wind_speed * u_lowest
    tau_y = -rho * Cd * wind_speed * v_lowest
    shflx = rho * constants.c_pd * Ch * wind_speed * (T_sfc - T_lowest)
    lhflx = rho * _L * Ch * wind_speed * (q_sfc - q_lowest)
    return tau_x, tau_y, shflx, lhflx
