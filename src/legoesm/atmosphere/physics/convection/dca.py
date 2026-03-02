"""Deep Convective Adjustment (DCA) scheme.

The simplest convection parameterization: scans from bottom to top,
adjusting adjacent layer pairs toward moist-adiabatic neutrality.
Excess moisture is removed as precipitation.

Uses jax.lax.scan for JIT-friendliness and differentiability.
Smooth sigmoid triggers ensure continuous gradients.

References
----------
- Manabe, S., Smagorinsky, J., & Strickler, R. F. (1965).
  Simulated climatology of a general circulation model with a
  hydrological cycle. Mon. Wea. Rev., 93, 769-798.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import (
    saturation_mixing_ratio,
    moist_adiabat_lapse_rate,
    compute_cape,
)
from legoesm.atmosphere.physics.convection.config import DCAConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput


def _adjust_one_iteration(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    dp: jax.Array,
    mixing_fraction: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One bottom-to-top sweep adjusting unstable layer pairs.

    Scans from the bottom-most pair upward using jax.lax.scan.
    For each adjacent pair (k, k-1) with k being lower:
    - Compare actual lapse rate to moist adiabatic
    - If unstable, adjust toward neutral with smooth blending
    - Redistribute excess moisture as precipitation

    Parameters
    ----------
    T : jax.Array
        Temperature [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    dp : jax.Array
        Layer thickness [Pa], shape (ncol, nlev).
    mixing_fraction : float
        Fraction of adjustment per iteration.

    Returns
    -------
    T_new : jax.Array
        Adjusted temperature, shape (ncol, nlev).
    q_v_new : jax.Array
        Adjusted moisture, shape (ncol, nlev).
    precip_col : jax.Array
        Precipitation from this sweep [kg/m^2/s equivalent: kg/kg * Pa/g],
        shape (ncol,).
    """
    ncol, nlev = T.shape

    # Reverse to scan from bottom to top
    # Level indices: 0=top, nlev-1=bottom
    # Reversed: 0=bottom, nlev-1=top
    T_rev = T[:, ::-1]          # (ncol, nlev)
    q_v_rev = q_v[:, ::-1]      # (ncol, nlev)
    p_rev = p_full[:, ::-1]     # (ncol, nlev)
    dp_rev = dp[:, ::-1]        # (ncol, nlev)

    def scan_step(carry, k):
        """Adjust pair (k-1, k) where k is the upper level in reversed indexing.

        carry: (T_below, q_below, p_below, dp_below, precip_accum)
        k indexes into reversed arrays for the upper level.
        """
        T_below, q_below, p_below, dp_below, precip_accum = carry

        # Upper level values from reversed arrays
        T_upper = T_rev[:, k]
        q_upper = q_v_rev[:, k]
        p_upper = p_rev[:, k]
        dp_upper = dp_rev[:, k]

        # Midpoint for lapse rate
        p_mid = 0.5 * (p_below + p_upper)
        T_mid = 0.5 * (T_below + T_upper)
        dp_pair = p_below - p_upper  # pressure difference (positive)
        dp_pair = jnp.clip(dp_pair, 1.0, None)

        # Actual lapse rate: dT/dp (temperature decrease per pressure decrease)
        actual_dTdp = (T_below - T_upper) / dp_pair

        # Moist adiabatic lapse rate
        gamma_m = moist_adiabat_lapse_rate(T_mid, p_mid)

        # Dry adiabatic lapse rate for normalization
        gamma_dry = constants.R_d * T_mid / (constants.c_pd * p_mid)

        # Dimensionless instability: positive means superadiabatic
        instability = (actual_dTdp - gamma_m) / jnp.clip(gamma_dry, 1e-10, None)

        # Smooth trigger: sigmoid with steep transition on dimensionless metric
        blend = jax.nn.sigmoid(10.0 * instability) * mixing_fraction

        # Target temperature for upper level: T_target = T_below - gamma_m * dp_pair
        T_target_upper = T_below - gamma_m * dp_pair

        # Adjusted temperatures: weighted average preserving layer enthalpy
        # Weight by layer dp for energy conservation
        total_dp = dp_below + dp_upper
        T_mean_weighted = (T_below * dp_below + T_upper * dp_upper) / total_dp
        T_target_mean = (T_below * dp_below + T_target_upper * dp_upper) / total_dp

        # Shift both layers to preserve mean while achieving target lapse rate
        delta_mean = T_mean_weighted - T_target_mean
        T_new_upper = T_target_upper + delta_mean
        T_new_below = T_below + (T_mean_weighted - (T_new_upper * dp_upper + T_below * dp_below) / total_dp) * total_dp / dp_below

        # Actually, simpler: preserve total enthalpy exactly
        # T_new_below = (total_dp * T_mean_weighted - dp_upper * T_new_upper) / dp_below
        T_new_below = (T_mean_weighted * total_dp - dp_upper * T_target_upper) / dp_below

        # Blend between original and adjusted
        T_adj_upper = T_upper + blend * (T_target_upper - T_upper)
        T_adj_below = T_below + blend * (T_new_below - T_below)

        # Moisture adjustment: saturate at the new temperature
        q_sat_upper = saturation_mixing_ratio(T_adj_upper, p_upper)
        q_sat_below = saturation_mixing_ratio(T_adj_below, p_below)

        # Remove excess moisture (precipitation)
        q_new_upper = jnp.minimum(q_upper, q_sat_upper)
        q_new_below = jnp.minimum(q_below, q_sat_below)

        # Blend moisture adjustment
        q_adj_upper = q_upper + blend * (q_new_upper - q_upper)
        q_adj_below = q_below + blend * (q_new_below - q_below)

        # Accumulate precipitation from moisture removal
        dq_upper = (q_upper - q_adj_upper) * dp_upper
        dq_below = (q_below - q_adj_below) * dp_below
        precip_new = precip_accum + (dq_upper + dq_below) / constants.g

        new_carry = (T_adj_below, q_adj_below, p_below, dp_below, precip_new)

        # Store the adjusted upper level values
        return new_carry, (T_adj_upper, q_adj_upper)

    # Initial carry: bottom level values
    init_carry = (
        T_rev[:, 0],      # T_below (bottom level)
        q_v_rev[:, 0],    # q_below
        p_rev[:, 0],      # p_below
        dp_rev[:, 0],     # dp_below
        jnp.zeros(ncol),  # precipitation accumulator
    )

    # Scan over levels 1..nlev-1 (moving upward)
    level_indices = jnp.arange(1, nlev)
    final_carry, (T_upper_stack, q_upper_stack) = jax.lax.scan(
        scan_step, init_carry, level_indices,
    )
    # T_upper_stack: (nlev-1, ncol) — levels 1..nlev-1 in reversed order
    # final_carry contains the adjusted bottom level

    T_adj_bottom, q_adj_bottom, _, _, precip_col = final_carry

    # Reconstruct full arrays (still in reversed order)
    T_adj_rev = jnp.concatenate(
        [T_adj_bottom[:, None], T_upper_stack.T],  # (ncol, nlev)
        axis=1,
    )
    q_adj_rev = jnp.concatenate(
        [q_adj_bottom[:, None], q_upper_stack.T],
        axis=1,
    )

    # Un-reverse to original top-to-bottom ordering
    T_new = T_adj_rev[:, ::-1]
    q_new = q_adj_rev[:, ::-1]

    return T_new, q_new, precip_col


def dca_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    dt: float,
    config: DCAConfig = DCAConfig(),
) -> ConvectionOutput:
    """Compute Deep Convective Adjustment tendencies.

    Parameters
    ----------
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    dt : float
        Model time step [s].
    config : DCAConfig
        Convection configuration.

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.
    """
    ncol, nlev = T.shape
    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)

    # Apply adjustment iterations
    T_adj = T
    q_adj = q_v
    precip_total = jnp.zeros(ncol)

    def body_fn(carry, _):
        T_c, q_c, prec = carry
        T_new, q_new, prec_iter = _adjust_one_iteration(
            T_c, q_c, p_full, dp, config.mixing_fraction,
        )
        return (T_new, q_new, prec + prec_iter), None

    (T_adj, q_adj, precip_total), _ = jax.lax.scan(
        body_fn,
        (T_adj, q_adj, precip_total),
        jnp.arange(config.n_iterations),
    )

    # Convert to tendencies
    dT_dt = (T_adj - T) / dt
    dq_v_dt = (q_adj - q_v) / dt
    precipitation = jnp.clip(precip_total / dt, 0.0, None)

    # CAPE diagnostic (using original profiles)
    # Use the adjusted profile as parcel temperature for CAPE
    cape = compute_cape(T, T_adj, p_full, p_half)

    # Convective mask: any column that was adjusted
    adjustment_magnitude = jnp.sum(jnp.abs(T_adj - T) * dp, axis=1)
    convective_mask = jax.nn.sigmoid(1000.0 * adjustment_magnitude)

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        precipitation=precipitation,
        cape=cape,
        convective_mask=convective_mask,
    )
