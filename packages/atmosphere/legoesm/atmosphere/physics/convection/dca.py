"""Deep Convective Adjustment (DCA) scheme.

Two variants, routed by ``DCAConfig.variant``:

* ``"manabe"`` (default): the Manabe-Smagorinsky-Strickler (1965)
  pairwise moist-adiabatic adjustment.  Scans from bottom to top,
  adjusting adjacent layer pairs toward moist-adiabatic neutrality;
  excess moisture is removed as precipitation.  Uses ``jax.lax.scan``
  for JIT-friendliness and differentiability with smooth sigmoid
  triggers.

* ``"ahmed_neelin"``: the Ahmed-Neelin-Adames (2020) lower-tropospheric-
  buoyancy (B_L) precipitation-buoyancy closure (``ahmed_neelin_dca``).
  The operative closure is the empirical eq-(8) precipitation–buoyancy
  relation ``P = a·(B_L − B_c)+``; the implied column latent heating is
  partitioned heating-up / drying-down along the eq-(41) direction
  (latent cooling cancels sensible heating per level), which conserves
  column moist static energy and drives ``B_L`` toward ``B_c``.  The
  column relaxation time scale is emergent (the paper's nominal ≈2 h is
  derived from observational EOF structures unavailable in-model).  See
  :func:`ahmed_neelin_dca`.

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
- Ahmed, F., Adames, A. F., & Neelin, J. D. (2020). Deep convective
  adjustment of temperature and moisture. J. Atmos. Sci., 77, 2163-2186.

Faithfulness
------------
The Manabe path (``_manabe_dca_convection`` / ``_adjust_one_iteration``) is
oracle-faithfulness-pinned in
``tests/atmosphere/hydrostatic/unit/test_dca_manabe_faithful.py``: the full
``ConvectionOutput`` matches an independent (original-coord Python-loop) oracle
to rel 1e-12 on a nonuniform mass grid; column moist static energy
(``c_pd*dT_dt + L_v*dq_v_dt``) and total water (``dq_v_dt + dq_c_conv_dt``) are
conserved to round-off; and the per-column CAPE-gate application (output ==
gate * pre-gate sweep, gated across the (33,300) J/kg tunable range), the
latent-heat closure, the moist-adiabatic target, and the pair level assignment
each carry a load-bearing canary.  (Ahmed-Neelin variant: see
``test_dca_ahmed_neelin.py``.)
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
from legoesm.atmosphere.physics.convection.config import (
    AhmedNeelinDCAConfig,
    DCAConfig,
)
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.atmosphere.physics.convection._triggers import cape_trigger
from legoesm.atmosphere.physics.convection.mass_flux import (
    stratosphere_mass_flux_gate,
)
from legoesm.atmosphere.physics._shared import safe_divide

__physics_contract__ = {
    "summary": (
        "Deep convective adjustment (variant dispatch): Manabe-Smagorinsky-"
        "Strickler (1965) pairwise moist-adiabatic adjustment or the "
        "Ahmed-Neelin-Adames (2020) lower-tropospheric-buoyancy precipitation "
        "closure; both relax the column toward neutrality and convert the "
        "dried vapor to cloud water."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_conv_dt": "kg/kg/s",
        "cape": "J/kg (B_L in m/s^2 for the ahmed_neelin variant)",
        "convective_mask": "1 (0-1 convective indicator)",
    },
    "sign_convention": (
        "Warms and dries where it stabilizes (dT_dt>0, dq_v_dt<0 in the "
        "convecting layer); the condensed vapor becomes cloud water "
        "(dq_c_conv_dt>=0, handed to microphysics); column moist static "
        "energy is conserved per adjusted pair/level (c_pd*dT_dt + "
        "L_v*dq_v_dt integrates to 0) and column total water is closed "
        "(integral of dq_v_dt + dq_c_conv_dt = 0); surface at the last "
        "vertical index."
    ),
    "conserves": ["energy", "moisture"],
    "differentiable": True,
    "reference": (
        "Manabe, Smagorinsky & Strickler (1965), Mon. Wea. Rev. 93, 769-798; "
        "Ahmed, Adames & Neelin (2020), J. Atmos. Sci. 77, 2163-2186"
    ),
    "idealized_test": (
        "A super-adiabatic saturated column relaxes toward moist-adiabatic "
        "neutrality with column moist static energy conserved and total water "
        "(vapor+cloud) closed; a stable / rest column yields zero tendency."
    ),
}


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


def _manabe_dca_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    dt: float,
    config: DCAConfig = DCAConfig(),
) -> ConvectionOutput:
    """Compute Manabe-style Deep Convective Adjustment tendencies.

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
    # cape_trigger == sigmoid(sharpness * (CAPE - threshold)) — differentiable.
    cape_gate = cape_trigger(
        cape, config.cape_threshold, config.cape_sharpness
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


# ===========================================================================
# Ahmed-Neelin-Adames (2020) lower-tropospheric-buoyancy (B_L) closure
# ===========================================================================
#
# Reference: Ahmed, Adames & Neelin (2020), J. Atmos. Sci. 77, 2163-2186
# (ANA20).  Equation numbers below refer to that paper.
#
# Moist enthalpy (ANA20, after eq 6):  e = T + (L_v/c_p)·q   [K]
#   (q is specific humidity; e* uses saturation specific humidity q*).
# Exner function:                       Π(p) = (p/p0)^κ        [-]
# Buoyancy (eq 7):
#   B_L = (g·Π_L/e_L*)·[ w_b·(e_B/Π_B) + w_L·(e_L/Π_L) − e_L*/Π_L ]  [m/s²]
# Precipitation (eq 8):  P = a·(B_L − B_c)·H(B_L − B_c)               [kg/m²/s]


def _exner(p: jax.Array) -> jax.Array:
    """Exner function Π(p) = (p/p_ref)^κ [-]; p_ref, κ from constants."""
    return (p / constants.p_ref) ** constants.kappa


def _layer_membership(
    p_full: jax.Array,
    p_top,
    p_bot,
    edge_width_pa: float,
) -> jax.Array:
    """Smooth (sigmoid) 0–1 membership weight for a pressure layer.

    Returns ~1 for ``p_top < p < p_bot`` and ~0 outside, with sigmoid
    transitions of half-width ``edge_width_pa`` at each edge.  A hard
    boolean pressure mask kills ``jax.grad`` through the layer boundaries;
    the product of two sigmoids keeps the layer averages differentiable.

    Parameters
    ----------
    p_full : jax.Array
        Full-level pressure [Pa], shape (ncol, nlev).
    p_top, p_bot : float
        Top (lower pressure) and bottom (higher pressure) of the layer
        [Pa], with ``p_top < p_bot``.
    edge_width_pa : float
        Sigmoid transition half-width [Pa].

    Returns
    -------
    jax.Array
        Membership weight in [0, 1], shape (ncol, nlev).
    """
    # below_bot ≈ 1 where p <= p_bot (inside, on the high-pressure side)
    below_bot = jax.nn.sigmoid((p_bot - p_full) / edge_width_pa)
    # above_top ≈ 1 where p >= p_top (inside, on the low-pressure side)
    above_top = jax.nn.sigmoid((p_full - p_top) / edge_width_pa)
    return below_bot * above_top


def _layer_average(
    field: jax.Array,
    dp: jax.Array,
    membership: jax.Array,
    eps: float = 1.0,
) -> jax.Array:
    """Mass-weighted layer average of ``field`` over a smooth layer.

    ⟨X⟩ = Σ_k X_k · w_k · Δp_k / Σ_k w_k · Δp_k, where ``w_k`` is the
    smooth layer membership and ``Δp_k`` the layer thickness [Pa].
    AD-safe: the denominator carries the same membership so it never
    vanishes when the layer is populated, and an ``eps`` [Pa] floor guards
    the empty-layer limit.

    Parameters
    ----------
    field : jax.Array
        Field to average [arbitrary unit], shape (ncol, nlev).
    dp : jax.Array
        Layer thickness [Pa], shape (ncol, nlev), positive.
    membership : jax.Array
        Smooth 0–1 layer membership, shape (ncol, nlev).
    eps : float
        Denominator floor [Pa] for the empty-layer limit.

    Returns
    -------
    jax.Array
        Mass-weighted layer mean [same unit as field], shape (ncol,).
    """
    w = membership * dp
    num = jnp.sum(field * w, axis=-1)
    den = jnp.sum(w, axis=-1)
    return num / jnp.clip(den, eps, None)


def _compute_BL(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    dp: jax.Array,
    cfg: AhmedNeelinDCAConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Lower-tropospheric buoyancy B_L (ANA20 eq 7) and its building blocks.

    Parameters
    ----------
    T : jax.Array
        Temperature [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor specific humidity [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Full-level pressure [Pa], shape (ncol, nlev).
    dp : jax.Array
        Layer thickness [Pa], shape (ncol, nlev), positive.
    cfg : AhmedNeelinDCAConfig
        Ahmed-Neelin configuration.

    Returns
    -------
    BL : jax.Array
        Lower-tropospheric buoyancy [m/s²], shape (ncol,).
    member_bl : jax.Array
        Boundary-layer membership, shape (ncol, nlev).
    member_lft : jax.Array
        Lower-free-troposphere membership, shape (ncol, nlev).
    Pi_L : jax.Array
        LFT-averaged Exner function [-], shape (ncol,).
    e_L_star : jax.Array
        LFT-averaged saturation moist enthalpy [K], shape (ncol,).
    """
    Lv_cp = constants.L_v / constants.c_pd  # [K per (kg/kg)]

    # Moist enthalpy fields (ANA20 after eq 6).  ``q_sat`` is the saturation
    # MIXING RATIO, matching the model's ``q_v`` convention: legoESM
    # initialises and carries ``q_v`` as a mixing ratio (it is built from
    # ``saturation_mixing_ratio``), so e = T + (L_v/c_p)·q_v and
    # e* = T + (L_v/c_p)·q_sat use the SAME water variable.  (ANA20 writes
    # ``q`` loosely as "specific humidity"; at tropical q≈20 g/kg the
    # mixing-ratio vs specific-humidity difference is ~2% in e and far
    # smaller in B_L, but using one consistent variable avoids a spurious
    # systematic offset between e and e* — codex review-1 finding.)
    q_sat = saturation_mixing_ratio(T, p_full)  # [kg/kg] mixing ratio
    e = T + Lv_cp * q_v          # [K]
    e_star = T + Lv_cp * q_sat   # [K]
    Pi = _exner(p_full)          # [-]

    # Surface-aware layer edges.  ANA20 defines the BL as "surface → 850 hPa"
    # (Δp_B ≈ 150 hPa).  Over high topography / low surface pressure a FIXED
    # 850-hPa BL top can sit above the surface, leaving the BL empty and the
    # ratios e_B/Π_B meaningless (codex review-1 finding).  We therefore
    # clamp the BL top to stay a minimum depth below the surface pressure
    # (``p_s − layer_floor``) and the LFT top a minimum depth below the BL
    # top, so both layers always retain mass and degrade gracefully on a
    # shallow column.  On a standard p_s ≈ 1000 hPa column the clamps are
    # inactive and the edges are exactly the config 850 / 500 hPa.
    # Surface pressure ≈ bottom interface = lowest full level + half its
    # thickness (p_full ordered top→bottom, so index −1 is the surface
    # layer).  Avoids changing ``_compute_BL``'s signature for all callers.
    p_s = (p_full[:, -1:] + 0.5 * dp[:, -1:])            # (ncol, 1) [Pa]
    layer_floor = cfg.layer_min_depth_pa                 # min layer depth [Pa]
    p_bl_top = jnp.minimum(cfg.p_bl_top_pa, p_s - layer_floor)        # (ncol,1)
    p_lft_top = jnp.minimum(cfg.p_lft_top_pa, p_bl_top - layer_floor)  # (ncol,1)
    # BL high-pressure edge left open above the surface so the lowest model
    # levels are always counted.
    p_surface_ceiling = p_s + 10.0 * cfg.layer_edge_width_pa
    member_bl = _layer_membership(
        p_full, p_bl_top, p_surface_ceiling, cfg.layer_edge_width_pa,
    )
    member_lft = _layer_membership(
        p_full, p_lft_top, p_bl_top, cfg.layer_edge_width_pa,
    )

    # Layer-averaged moist enthalpies and Exner functions.
    e_B = _layer_average(e, dp, member_bl)            # [K]
    e_L = _layer_average(e, dp, member_lft)           # [K]
    e_L_star = _layer_average(e_star, dp, member_lft)  # [K]
    Pi_B = _layer_average(Pi, dp, member_bl)          # [-]
    Pi_L = _layer_average(Pi, dp, member_lft)         # [-]

    # ANA20 eq (7):
    #   B_L = (g·Π_L/e_L*)·[ w_b·(e_B/Π_B) + w_L·(e_L/Π_L) − e_L*/Π_L ]
    bracket = (
        cfg.w_b * (e_B / Pi_B)
        + cfg.w_l * (e_L / Pi_L)
        - e_L_star / Pi_L
    )
    BL = (constants.g * Pi_L / e_L_star) * bracket  # [m/s²]
    return BL, member_bl, member_lft, Pi_L, e_L_star


def _a_si(cfg: AhmedNeelinDCAConfig) -> float:
    """Slope ``a`` of the P–B_L line in SI mass-flux units.

    ANA20 Table 1 reports ``a`` in mm h⁻¹ per (m s⁻²).  Convert to
    [kg m⁻² s⁻¹ per (m s⁻²)] using water density ρ_w = 1000 kg/m³ and
    3600 s/h: ``a_SI = a_mm_per_hr · ρ_w / (1000 · 3600)`` since 1 mm of
    water = 1 kg/m² (ρ_w · 1e-3 m).
    """
    rho_w = constants.rho_water  # density of liquid water [kg/m³]
    # 1 mm/h of rain = (rho_w * 1e-3 m) / 3600 s = rho_w / 3.6e6 kg/m²/s
    return cfg.a_mm_per_hr * rho_w / 3.6e6  # coeff-ok: mm/hr -> kg/m^2/s


def _precip_from_BL(
    BL: jax.Array,
    cfg: AhmedNeelinDCAConfig,
) -> jax.Array:
    """Precipitation P = a·(B_L − B_c)·H(B_L − B_c) (ANA20 eq 8).

    The Heaviside is smoothed to a softplus so the ramp is differentiable
    at B_c:  ``softplus(s·x)/s → max(x, 0)`` as ``s → ∞``.  Output is a
    non-negative mass flux [kg/m²/s].

    Parameters
    ----------
    BL : jax.Array
        Lower-tropospheric buoyancy [m/s²], shape (ncol,).
    cfg : AhmedNeelinDCAConfig
        Configuration (slope ``a``, critical ``B_c``, sharpness).

    Returns
    -------
    jax.Array
        Precipitation [kg/m²/s], shape (ncol,), >= 0.
    """
    s = cfg.precip_heaviside_sharpness
    excess = jax.nn.softplus(s * (BL - cfg.b_c)) / s  # smooth max(BL-Bc, 0)
    return _a_si(cfg) * excess


def ahmed_neelin_dca(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    dt: float,
    config: AhmedNeelinDCAConfig = AhmedNeelinDCAConfig(),
) -> ConvectionOutput:
    """Ahmed-Neelin-Adames (2020) B_L convective-adjustment tendencies.

    The OPERATIVE closure is the empirical precipitation-buoyancy relation
    eq (8) — the central result the paper validates against TRMM/ERA data
    (Fig 3).  The column heating and drying are set so that the
    column-integrated precipitation equals
    ``P = a·(B_L − B_c)·H(B_L − B_c)`` (eq 8), moved along the eq-(41)
    direction that conserves column moist static energy:

    * **Precipitation (eq 8, OPERATIVE).** ``P = a·(B_L − B_c)+`` with the
      empirical slope ``a`` (config ``a_mm_per_hr``) and critical buoyancy
      ``B_c`` (config ``b_c``).  The Heaviside is softplus-smoothed for
      differentiability.  The implied column latent heating is
      ``Q̂_c = L_v·P`` (ANA20 eq 11), so ``∫ c_p·dT/dt dp/g = L_v·P``
      holds to machine precision.

    * **Adjustment direction (eq 41).** The heating is moved along the
      second eigenvector of the linearised system (slope −1 in the
      moist-enthalpy q̂–T̂ plane): per level the latent cooling exactly
      cancels the sensible heating, ``L_v·dq_v/dt + c_p·dT/dt = 0``.  Hence
      column moist static energy is conserved during the adjustment, and
      drying coincides with warming.

    * **Adjustment time scale (eqs 38-42).** The buoyancy excess relaxes
      toward zero with an EMERGENT time scale set by eq (8) and the
      column's own buoyancy sensitivity to the heating direction.  ANA20
      report ``τ_c ≈ 2 h`` (eq 42); that value is itself DERIVED from
      ``a`` together with the observational EOF vertical structures
      (``L_{TL}``, ``L_{qL}``, …; eqs 25-27) that are not available
      in-model, so we surface the emergent τ as a diagnostic rather than
      imposing it.  See `config.tau_adjust_s` — it is retained for
      reference / optional rescaling but is NOT used to set the operative
      rate (which is eq 8).

    Conventions: ``dq_v_dt < 0`` (drying), ``dT_dt > 0`` (latent heating)
    where convecting; ``q_v`` stays non-negative (the per-step adjustment
    is capped at the available vapor, preserving the column MSE balance).

    Parameters
    ----------
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).  (legoESM
        carries ``q_v`` as a mixing ratio; ``e`` and ``e*`` use the same
        variable — see `_compute_BL`.)
    p_full : jax.Array
        Full-level pressure [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Half-level pressure [Pa], shape (ncol, nlev+1).
    dt : float
        Model time step [s].
    config : AhmedNeelinDCAConfig
        Ahmed-Neelin configuration.

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.  ``cape`` carries the
        diagnostic ``B_L`` [m/s²] (not Joules) for this scheme;
        ``convective_mask`` is the smooth precipitating indicator.
    """
    ncol, nlev = T.shape
    dp = p_half[:, 1:] - p_half[:, :-1]            # (ncol, nlev) [Pa] > 0
    g = constants.g
    Lv = constants.L_v
    c_pd = constants.c_pd

    # --- B_L and the OPERATIVE precipitation (ANA20 eqs 7, 8) -------------
    BL, member_bl, member_lft, _Pi_L, _e_L_star = _compute_BL(
        T, q_v, p_full, dp, config,
    )                                              # BL: (ncol,) [m/s²]
    precip = _precip_from_BL(BL, config)           # (ncol,) [kg/m²/s] >= 0  (eq 8)
    # Implied column-integrated latent heating Q̂_c = L_v·P (ANA20 eq 11).
    Qc_col = Lv * precip                           # (ncol,) [W/m²]

    # --- Vertical structure: BL+LFT mass shape ---------------------------
    # ``member_union`` is the union of the BL and LFT memberships (the
    # layers that define B_L).  ``shape_k`` [1/Pa] is normalised so that
    # Σ_k shape_k·Δp_k = 1 on a populated column; it vanishes on a
    # degenerate all-stratosphere column.
    member_union = member_bl + member_lft           # (ncol, nlev) [-]
    member_mass = jnp.sum(member_union * dp, axis=-1)  # (ncol,) [Pa]
    shape = member_union * safe_divide(
        jnp.ones_like(member_mass), member_mass, eps=1.0,
    )[:, None]                                       # (ncol, nlev) [1/Pa]

    # --- Heating from eq (8), distributed; drying along eq (41) ----------
    # Heating per level [K/s]:  dT_dt_k = (Q̂_c / c_p) · g · shape_k.
    #   Σ_k c_p·dT_dt_k·Δp_k/g = Q̂_c·Σ_k shape_k·Δp_k = Q̂_c   (eq 11). ✓
    heating_rate = (Qc_col / c_pd)[:, None] * g * shape   # (ncol, nlev) [K/s]
    # Slope −1 in moist-enthalpy units (eq 41): per level latent cooling
    # cancels sensible heating ⇒ column MSE conserved and
    # ∫ L_v·dq dp/g = −∫ c_p·dT dp/g = −L_v·P exactly.
    drying_rate = -(c_pd / Lv) * heating_rate        # (ncol, nlev) [kg/kg/s]

    # --- Positivity: cap the per-step drying at the available vapor ------
    # A single step must not drive q_v negative.  Scale the WHOLE column's
    # heating+drying by one factor so the MSE balance (eq 41) and the
    # latent-heating ↔ precip closure stay exact under the cap.
    max_drying = -q_v / dt                           # (ncol, nlev) <= 0
    frac_level = jnp.where(
        drying_rate < max_drying,                    # would over-dry
        safe_divide(max_drying, drying_rate, eps=1e-30, fill=1.0),
        jnp.ones_like(drying_rate),
    )
    frac = jnp.min(frac_level, axis=-1, keepdims=True)  # (ncol, 1) in (0, 1]

    dT_dt = heating_rate * frac                      # (ncol, nlev) [K/s]
    dq_v_dt = drying_rate * frac                     # (ncol, nlev) [kg/kg/s]

    # Diagnostic precipitation P [kg/m²/s] = column latent heating / L_v
    # (equals eq-(8) P before the positivity cap engages).
    precip = jnp.sum(c_pd * dT_dt * dp, axis=-1) / (g * Lv)  # (ncol,) >= 0

    # Convective condensate source for the cloud-water bucket: the column
    # net drying becomes cloud water (microphysics owns surface precip).
    # By construction dq_v_dt <= 0, so -dq_v_dt >= 0 everywhere.
    dq_c_conv_dt = jnp.maximum(-dq_v_dt, 0.0)        # (ncol, nlev) [kg/kg/s]

    # Smooth precipitating indicator in [0, 1].
    convective_mask = jax.nn.sigmoid(
        config.precip_heaviside_sharpness * (BL - config.b_c)
    )                                                # (ncol,)
    del precip  # microphysics owns the surface precip diagnostic

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=BL,                                     # B_L [m/s²] diagnostic
        convective_mask=convective_mask,
    )


def dca_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    dt: float,
    config: DCAConfig = DCAConfig(),
) -> ConvectionOutput:
    """Compute Deep Convective Adjustment tendencies (variant dispatch).

    Routes on ``config.variant``:

    * ``"manabe"`` (default): :func:`_manabe_dca_convection` — the
      Manabe-Smagorinsky-Strickler (1965) pairwise moist-adiabatic
      adjustment (legacy legoESM ``dca`` behaviour, unchanged).
    * ``"ahmed_neelin"``: :func:`ahmed_neelin_dca` — the
      Ahmed-Neelin-Adames (2020) B_L precipitation-buoyancy closure,
      using ``config.ahmed_neelin``.

    Dispatch is on a static Python string (the config ``variant`` field),
    so only the selected branch is traced — no ``jnp.where`` over both
    schemes.  Unknown variants raise ``ValueError`` (no silent fallback).

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
        Convection configuration (selects the variant).

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.
    """
    if config.variant == "manabe":
        return _manabe_dca_convection(T, q_v, p_full, p_half, dt, config)
    elif config.variant == "ahmed_neelin":
        return ahmed_neelin_dca(
            T, q_v, p_full, p_half, dt, config.ahmed_neelin,
        )
    else:
        raise ValueError(
            f"Unknown DCAConfig.variant: {config.variant!r}. "
            f"Choose 'manabe' or 'ahmed_neelin'."
        )
