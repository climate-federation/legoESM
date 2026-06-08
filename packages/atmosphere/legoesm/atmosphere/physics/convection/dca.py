"""Deep Convective Adjustment (DCA) scheme.

The simplest convection parameterization: scans from bottom to top,
adjusting adjacent layer pairs toward moist-adiabatic neutrality.
Excess moisture is removed as precipitation.

Uses jax.lax.scan for JIT-friendliness and differentiability.
Smooth sigmoid triggers ensure continuous gradients.

Planned: a DRY convective-adjustment mode (DCAConfig.dry=False default)
--------------------------------------------------------------------
The gray radiative–convective-equilibrium column
(``atmosphere.idealized.radiative_convective_column``) needs a *dry* adjustment,
but this scheme currently always targets the SATURATED MOIST adiabat (the
``gamma_m = moist_adiabat_lapse_rate(T_mid, p_mid)`` target in
``_adjust_one_iteration`` uses ``q_sat(T,p)``, NOT the supplied ``q_v``) and gates
by moist CAPE — so ``q_v=0`` does NOT yield a dry adjustment.  To add a correct
dry mode (additive, default-off, existing moist path byte-identical):

1. ``DCAConfig.dry: bool = False``.
2. In ``_adjust_one_iteration`` (thread ``dry`` through), when ``dry``:
   * use the dry-adiabatic target ``gamma = gamma_dry = R_d*T_mid/(c_pd*p_mid)``
     (already computed) for both the instability metric and ``T_target_upper``;
   * SKIP the moisture branch entirely (no ``q_sat`` saturation/removal, no
     ``delta_T_lh`` latent warming) — ``q`` unchanged, precip 0.  This makes the
     pair adjustment conserve dry static energy ``c_p*T*dp`` exactly.
3. In ``dca_convection``: when ``dry``, replace the moist-CAPE gate
   (``compute_cape`` vs ``cape_threshold``) with a DRY static-stability gate —
   e.g. a smooth sigmoid on the column's max super-adiabatic excess
   ``(actual_dTdp - gamma_dry)`` — so the adjustment actually fires for a dry
   super-adiabatic column (moist CAPE is ~0 there and would suppress it).
4. Validate (new test): a super-adiabatic dry column relaxes to dry-adiabatic
   NEUTRALITY (constant potential temperature ``theta`` to tol), column dry
   enthalpy ``sum(c_p*T*dp)`` conserved (no precip), jax.grad finite; then
   re-enable the column's ``convective_adjustment=True`` path against it.

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
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.thermodynamics import (
    moist_adiabat_lapse_rate,
    compute_cape,
)
from legoesm.atmosphere.physics.convection.config import DCAConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.atmosphere.physics.convection.mass_flux import (
    stratosphere_mass_flux_gate,
)
from legoesm.atmosphere.physics._shared import safe_divide


def _adjust_one_iteration(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    dp: jax.Array,
    mixing_fraction: float,
    instability_blend_sharpness: float,
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
        Water vapor specific humidity [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    dp : jax.Array
        Layer thickness [Pa], shape (ncol, nlev).
    mixing_fraction : float
        Fraction of adjustment per iteration.
    instability_blend_sharpness : float
        Sigmoid sharpness on the dimensionless superadiabatic-instability
        metric controlling adjustment blending.

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
        """Adjust adjacent pair (k-1, k) using progressively updated profiles."""
        T_work, q_work, precip_accum = carry

        T_below = T_work[:, k - 1]
        q_below = q_work[:, k - 1]
        p_below = p_rev[:, k - 1]
        dp_below = dp_rev[:, k - 1]

        T_upper = T_work[:, k]
        q_upper = q_work[:, k]
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

        # Smooth trigger: sigmoid on the dimensionless super-adiabatic
        # metric, times the per-iteration mixing fraction, AND gated out
        # of the stratosphere.  The pressure gate multiplies ``blend``
        # BEFORE the thermodynamics so the thin upper-model layers (small
        # Δp/g) are never adjusted: an ungated solve concentrated the
        # pair's compensating heat in the top layer and spiked it to
        # ~519 K → NaN within the first RCE day (codex adversarial review).
        strat_gate_pair = stratosphere_mass_flux_gate(p_mid)
        blend = (
            jax.nn.sigmoid(instability_blend_sharpness * instability)
            * mixing_fraction * strat_gate_pair
        )

        # Simultaneous two-level solve enforcing BOTH the moist-adiabatic
        # target lapse and mass-weighted (dry) enthalpy conservation:
        #     T_below_new − T_upper_new = gamma_m · dp_pair        (lapse)
        #     dp_b·T_below_new + dp_u·T_upper_new
        #         = dp_b·T_below + dp_u·T_upper                  (enthalpy)
        # The earlier code derived ``T_target_upper`` from the OLD
        # ``T_below`` and then moved ``T_below`` independently, so at
        # ``blend = 1`` the achieved lapse overshot/inverted the target and
        # dumped the compensating heat into the thin top layer (codex
        # must-fix; the discarded ``delta_mean``/``T_new_upper`` lines were
        # also dead code).  Solving the 2×2 system makes ``blend = 1``
        # impose the target lapse exactly while conserving pair enthalpy.
        total_dp = dp_below + dp_upper
        enthalpy = T_below * dp_below + T_upper * dp_upper
        T_new_upper = (enthalpy - dp_below * gamma_m * dp_pair) / total_dp
        T_new_below = T_new_upper + gamma_m * dp_pair

        # Blend between original and adjusted
        T_adj_upper = T_upper + blend * (T_new_upper - T_upper)
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
        dq_upper_pa = (q_upper - q_adj_upper) * dp_upper  # kg/kg · Pa
        dq_below_pa = (q_below - q_adj_below) * dp_below  # kg/kg · Pa
        precip_new = precip_accum + (dq_upper_pa + dq_below_pa) / constants.g

        # Moist static energy conservation: condensed water releases L_v
        # energy per unit mass.  The previous implementation conserved
        # only dry static energy (mass-weighted T preserved), losing
        # L_v · ⟨Δq⟩ ≈ 2.5 K per g/kg of column-mean condensed water.
        # Adding the latent warming uniformly to the pair preserves the
        # moist-adiabatic lapse rate just imposed via T_target while
        # closing the moist static energy budget:
        #   c_p ⟨ΔT⟩ + L_v ⟨Δq⟩ = 0  (column mean over the pair).
        delta_T_lh = (
            constants.L_v * (dq_upper_pa + dq_below_pa)
            / (constants.c_pd * (dp_below + dp_upper))
        )
        T_adj_upper = T_adj_upper + delta_T_lh
        T_adj_below = T_adj_below + delta_T_lh

        T_work = T_work.at[:, k - 1].set(T_adj_below)
        T_work = T_work.at[:, k].set(T_adj_upper)
        q_work = q_work.at[:, k - 1].set(q_adj_below)
        q_work = q_work.at[:, k].set(q_adj_upper)

        return (T_work, q_work, precip_new), None

    # Pin the precip carry dtype to whatever ``q * dp`` actually
    # produces inside the scan body — under standard promotion the
    # compute precision wins when ``q_v`` is at storage precision but
    # ``dp_rev`` comes from sigma-coord arrays at compute precision.
    # ``jnp.result_type`` resolves this without materializing a scalar.
    _precip_dtype = jnp.result_type(q_v_rev, dp_rev)
    init_carry = (T_rev, q_v_rev, jnp.zeros(ncol, dtype=_precip_dtype))
    level_indices = jnp.arange(1, nlev)
    (T_adj_rev, q_adj_rev, precip_col), _ = jax.lax.scan(
        scan_step, init_carry, level_indices,
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
        Water vapor specific humidity [kg/kg], shape (ncol, nlev).
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

    # Apply adjustment iterations.  ``prec_iter`` returned by the inner
    # scan inherits ``q * dp`` precision (compute precision wins when
    # state is f32 but sigma-coord-derived dp is f64), so pin
    # ``precip_total`` to the same result-type so the outer scan carry
    # input matches its output.
    T_adj = T
    q_adj = q_v
    precip_total = jnp.zeros(ncol, dtype=jnp.result_type(q_v, dp))

    def body_fn(carry, _):
        T_c, q_c, prec = carry
        T_new, q_new, prec_iter = _adjust_one_iteration(
            T_c, q_c, p_full, dp, config.mixing_fraction,
            config.instability_blend_sharpness,
        )
        return (T_new, q_new, prec + prec_iter), None

    (T_adj, q_adj, precip_total), _ = jax.lax.scan(
        body_fn,
        (T_adj, q_adj, precip_total),
        jnp.arange(config.n_iterations),
    )

    # CAPE diagnostic BEFORE gating (using original profiles)
    cape = compute_cape(T, T_adj, p_full, p_half)

    # Gate tendencies by CAPE: only adjust where CAPE exceeds threshold.
    # Smooth sigmoid gating preserves differentiability.
    cape_gate = jax.nn.sigmoid(
        config.cape_sharpness * (cape - config.cape_threshold)
    )  # (ncol,)

    # Convert to tendencies, gated by CAPE (a per-column scalar, so it
    # preserves the per-pair MSE balance below).  The stratosphere is
    # already suppressed inside the sweep, where ``blend`` is multiplied
    # by the pressure gate *per adjusting pair* (uniformly across the
    # pair's two levels) — that keeps ``c_p·ΔT + L_v·Δq = 0`` per pair, so
    # column MSE is conserved.  Gating the OUTPUT tendencies by the
    # per-level pressure factor instead would break that conservation
    # (∫ gate·[c_p·dT + L_v·dq] dp ≠ 0 for a level-varying gate).
    dT_dt = cape_gate[:, None] * (T_adj - T) / dt
    dq_v_dt = cape_gate[:, None] * (q_adj - q_v) / dt
    # Convective source for cloud water — column-conservative
    # rescaling so that ∫ dq_c_conv_dt dp/g equals the column-net
    # drying (matches the legacy ``precipitation`` formula). Naive
    # per-level ``max(-dq_v_dt, 0)`` would create water column-wide
    # whenever the adjustment has mixed-sign vapor tendencies; this
    # rescaling removes that bug while keeping the field non-negative
    # at every level. ``precip_total`` (the scan-accumulated column
    # total) is no longer surfaced — microphysics owns the surface
    # precipitation diagnostic.
    del precip_total
    local_cond = jnp.maximum(-dq_v_dt, 0.0)
    # Both column reductions share the ``* dp / g`` weight on the level
    # axis — stack the two integrands and reduce once.
    _col_pair = jnp.sum(
        jnp.stack([local_cond, dq_v_dt], axis=-1) * (dp / constants.g)[..., None],
        axis=-2,
    )
    col_local_cond = _col_pair[..., 0:1]
    col_net_drying = jnp.clip(-_col_pair[..., 1:2], 0.0, None)
    # AD-safe column rescaling — see sbm.py for derivation; issue #249.
    dq_c_conv_dt = local_cond * safe_divide(
        col_net_drying, col_local_cond, eps=1e-20,
    )  # (ncol, nlev) [kg/kg/s]

    # Convective mask: CAPE-gated
    convective_mask = cape_gate

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=convective_mask,
    )
