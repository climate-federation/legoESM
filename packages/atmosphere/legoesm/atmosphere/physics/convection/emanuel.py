"""Emanuel (1991) convection: faithful port of CONVECT v4.3c.

The scheme IS Kerry Emanuel's ``convect43c.f`` (CONVECT version 4.3c, 20 May
2002): parcel source at the level of maximum moist static energy below its
lowest local minimum, a prognostic cloud-base mass flux relaxed toward
sub-cloud quasi-equilibrium (the DTMA closure), an ensemble of mixtures of
cloud and environmental air sorted by buoyancy (the ``(i,j)`` episodic-mixing
matrix), and the precipitating unsaturated downdraught (on by default,
``enable_unsaturated_downdraft``; off takes the oracle's downdraught skip).  Each piece is ported line by line:

* :mod:`._emanuel_mixing` -- column setup, source/cloud-base/cloud-top
  diagnosis, rejection gates, TLIFT, the closure and the mixing matrix with its
  tendencies (oracle lines 338-938, 1004-1075);
* :mod:`._emanuel_downdraft` -- the downdraught (lines 713-830);
* this module -- the downdraught tendencies added to the mixing tendencies,
  the cloud-top ``FRAC`` adjustment (lines 940-964) and the ``ENTS``
  enthalpy-conservation pass (lines 969-984), in that order, as the oracle.

``tests/atmosphere/test_emanuel_convect43c_oracle.py`` checks FT, FQ and CBMF
against the running oracle (fixture generator in
``scripts/validate/oracle/emanuel_convect43c/``).

What is NOT the oracle (stated, not hidden):

* No convective momentum transport (CONVECT's FU/FV): this scheme's signature
  takes no winds, so ``du_dt_conv = dv_dt_conv = None``.
* The oracle's WD/TPRIME/QPRIME surface-flux scales are not returned.
* Water routing: CONVECT returns a scalar PRECIP.  Here the whole column
  water deficit ``-sum(FQ dp/g)`` (the precipitating ``EP*CLW`` and the mixture
  condensate ``AWAT`` detrained below its origin, both the oracle's WDTRAIN,
  minus downdraught re-evaporation) is emitted as the in-updraught rain source
  ``dq_r_conv_dt``, placed by the column drying profile; ``dq_c_conv_dt`` is 0.
  A column whose net vapour tendency is POSITIVE emits no rain and the gain
  is not balanced (pre-existing; measured 2026-10-28 at the default
  smoothing on all 40962 columns of an MPAS AMIP checkpoint: 234 columns
  with the downdraught off, 1488 with it on, max gain 6.8e-6 kg/m2/s vs a
  max column vapour loss of 7.8e-4; none on the 1535 ECMWF profiles).
* ``cape``/``convective_mask`` are diagnostics (surface-parcel moist-adiabat
  CAPE and a smooth CAPE indicator); they do not gate the scheme.

References
----------
* Emanuel, K. A. (1991). A scheme for representing cumulus convection
  in large-scale models.  *J. Atmos. Sci.*, 48, 2313-2335.
* Emanuel, K. A. and M. Zivkovic-Rothman (1999). *J. Atmos. Sci.*, 56,
  1766-1782.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics.convection._emanuel_downdraft import (
    emanuel_downdraft,
)
from legoesm.atmosphere.physics.convection._emanuel_mixing import (
    emanuel_mixing_tendencies,
)
from legoesm.atmosphere.physics.convection._triggers import (
    cape_trigger,
)
from legoesm.atmosphere.physics.convection.config import EmanuelConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.atmosphere.physics.thermodynamics import (
    compute_cape,
    compute_moist_adiabat,
)
from legoesm.thermo import latent_heat_vaporization

from legoesm import constants

__all__ = ("emanuel_convection",)


__physics_contract__ = {
    "summary": (
        "Emanuel (1991) convection, a faithful port of CONVECT v4.3c "
        "(convect43c.f): buoyancy-sorting episodic mixing with a prognostic "
        "cloud-base mass flux (DTMA quasi-equilibrium closure) and the "
        "precipitating downdraught (default on); returns convective T/q_v/q_c/q_r "
        "tendencies plus the updated CBMF carry."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa",
        "conv_prog_profile": "kg/m^2/s (prognostic CBMF carry in slot -1)",
        "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_conv_dt": "kg/kg/s (0)",
        "dq_r_conv_dt": "kg/kg/s",
        "cape": "J/kg (diagnostic)",
        "convective_mask": "1 (0-1 diagnostic CAPE indicator)",
        "conv_prog_profile_new": "kg/m^2/s (updated prognostic CBMF carry)",
    },
    "sign_convention": (
        "Oracle FT/FQ: warms and dries the convecting layer; the ENTS pass "
        "closes c_pn*FT + L_v(T)*FQ over levels 1..INB exactly; the column "
        "vapour loss is a non-negative rain source (no cloud-water source; "
        "a net column vapour gain is not balanced); "
        "the prognostic cloud-base mass flux is "
        ">=0; surface at the last vertical index."
    ),
    "conserves": ["energy"],
    "differentiable": True,
    "reference": "Emanuel (1991), J. Atmos. Sci. 48, 2313-2335; CONVECT v4.3c",
    "idealized_test": (
        "tests/atmosphere/test_emanuel_convect43c_oracle.py: matches the "
        "running convect43c.f oracle to <= 1e-10 (fp64) at the hard limit of "
        "the smoothing fields; a stable column returns zero tendencies and a "
        "zero carry."
    ),
}


# --- unit conversion (oracle PH in hPa) ---
_HPA_PER_PA = 0.01

# Fields of the retired single-sigmoid surrogate (and of its smooth source
# selector).  CONVECT has no counterpart; they are kept so existing configs
# load, and any non-default value is refused rather than silently ignored.
_RETIRED_FIELDS = (
    "n_mixing_fractions", "cu_coefficient", "precip_efficiency_water",
    "precip_efficiency_lcl", "parcel_perturb_T", "parcel_perturb_q",
    "smooth_trigger_sharpness", "epsilon_0", "delta_0",
    "mse_min_search_offset",
    # mass-flux caps with no CONVECT counterpart (removed, #2213 split):
    "M_b_max", "cbmf_carry_max",
)


def _is_concrete(value) -> bool:
    return not isinstance(value, jax.core.Tracer)


def _check_retired_fields(config: EmanuelConfig) -> None:
    """Refuse non-default retired fields (scalars, NumPy or concrete JAX
    values; a traced value is left to the trainer's inert-leaf gate)."""
    g = config.use_genuine_mixing
    if _is_concrete(g) and not bool(np.all(np.asarray(g))):
        raise ValueError(
            "EmanuelConfig.use_genuine_mixing=False selected the retired "
            "single-sigmoid surrogate; the Emanuel scheme is now the CONVECT "
            "v4.3c port only.")
    default = EmanuelConfig()
    for name in _RETIRED_FIELDS:
        value = getattr(config, name)
        if _is_concrete(value) and np.any(np.asarray(value) != getattr(default, name)):
            raise ValueError(
                f"EmanuelConfig.{name}={value!r}: this field is retired and "
                "has no counterpart in CONVECT "
                "v4.3c; it no longer affects the scheme.  Leave it at its "
                f"default ({getattr(default, name)!r}).")


def _div_floor(num, den, floor):
    """``num / max(den, floor)`` with no derivative through the floor: a
    floored entry divides by the constant, so 1/floor**2 (which overflows in
    fp32) never enters a gradient or tangent.  Same value."""
    ok = den > floor
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), num / floor)


def _apply_frac(f, inb, frac, dph, scale):
    """Oracle l.940-950 on a SURFACE-FIRST profile: move FRAC of the INB
    tendency to INB-1, times the layer-thickness ratio and
    ``scale(INB)/scale(INB-1)`` (LV for FQ, CPN for FT)."""
    idx = jnp.arange(f.shape[-1])[None, :]

    def at(x, i):
        return jnp.take_along_axis(x, i[:, None], axis=-1)

    f_old = at(f, inb)
    thick = at(dph, inb) / at(dph, inb - 1)
    f = jnp.where(idx == inb[:, None], f * (1.0 - frac[:, None]), f)
    return jnp.where(idx == (inb - 1)[:, None],
                     f + frac[:, None] * f_old * thick * at(scale, inb)
                     / at(scale, inb - 1), f)


def emanuel_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    conv_prog_profile: jax.Array,
    dt: float,
    config: EmanuelConfig = EmanuelConfig(),
) -> tuple[ConvectionOutput, jax.Array]:
    """Emanuel CONVECT v4.3c convection.

    Parameters
    ----------
    T : jax.Array, shape (ncol, nlev)
        Environmental temperature [K], surface-LAST.
    q_v : jax.Array, shape (ncol, nlev)
        Water-vapour specific humidity [kg/kg].
    p_full, p_half : jax.Array
        Full / half-level pressures [Pa].
    conv_prog_profile : jax.Array, shape (ncol, nlev)
        Prognostic carry: CONVECT's cloud-base mass flux CBMF, which the
        caller must store between calls, is read from ``[:, -1]`` and the
        updated value written back there.  Must NOT be advanced with a stale
        stage carry under multi-stage integrators (the SCM rejects it under
        RK).
    dt : float
        Time step [s] (CONVECT DELT).
    config : EmanuelConfig

    Returns
    -------
    out : ConvectionOutput
        ``du_dt_conv = dv_dt_conv = None`` (no convective momentum transport).
    conv_prog_profile_new : jax.Array, shape (ncol, nlev)

    Known limitation
    ----------------
    Left as is by user decision.  The rain is placed by the column drying
    weight ``drying / drying_col``.  A column whose drying ``drying_col`` lies
    in (1e-30, ~1e-19] kg/m2/s is not floored, and the weight's derivative
    carries ``1/drying_col**2``, which overflows fp32 there (fp64 is fine).
    Measured never reached (SLURM job 28511370): 0 such columns over the 1535
    ECMWF oracle profiles and 40962 AMIP columns x 8 configs; smallest
    positive drying 3e-8; 0 non-finite fp32 gradients.
    """
    _check_retired_fields(config)
    # The scheme runs in the state temperature's precision: a float64 tracer
    # or pressure in a float32 run would make the downdraught's scan carry and
    # outputs differ in dtype (reviewer decision, #2213 split).
    q_v, p_full, p_half = (jnp.asarray(x, T.dtype) for x in (q_v, p_full, p_half))
    ncol, nlev = T.shape
    dp = p_half[:, 1:] - p_half[:, :-1]

    # -- Diagnostics only (no oracle role): surface-parcel CAPE + indicator --
    T_moist = compute_moist_adiabat(T[:, -1], p_full)
    cape = compute_cape(T, T_moist, p_full, p_half)
    cape_weight = cape_trigger(cape, config.cape_threshold, config.cape_sharpness)

    # -- CONVECT through the mixing tendencies (carry: oracle CBMF in) --------
    cbmf_old = jnp.maximum(conv_prog_profile[:, -1], 0.0).astype(T.dtype)
    mixing = emanuel_mixing_tendencies(
        T, q_v, p_full, p_half, cbmf_old, dt,
        c_l=config.c_l_emanuel,
        elcrit=config.elcrit,
        tlcrit=config.tlcrit,
        entp=config.entp,
        dtmax=config.dtmax,
        alpha_closure=config.alpha_closure,
        damp_coefficient=config.damp_coefficient,
        level_window_sharpness=config.level_window_sharpness,
        sij_gate_sharpness=config.sij_gate_sharpness,
        sij_upper_gate=config.sij_upper_gate,
        denom_floor=config.denom_floor,
        sat_branch_sharpness=config.sat_branch_sharpness,
        lcl_pressure_sharpness=config.lcl_pressure_sharpness,
        strict_index_sharpness=config.strict_index_sharpness,
        epmax=config.precip_efficiency_max,
    )
    _rev = lambda x: x[:, ::-1]  # noqa: E731 - surface-last <-> surface-first
    ft = _rev(mixing.dT_dt)                       # surface-FIRST from here
    fq = _rev(mixing.dq_v_dt)

    # -- Precipitating downdraught (oracle l.713-830 and its FT/FQ terms) -----
    # Static Python branch: no trace overhead when disabled.  Disabled is the
    # oracle with its downdraught skipped (the EP(INB)<1e-4 branch taken).
    if config.enable_unsaturated_downdraft:
        downdraft = emanuel_downdraft(
            T=_rev(T), q=_rev(q_v), qs=mixing.qs,
            p_full=_rev(p_full), p_half=_rev(p_half[:, 1:]),
            h_dry_static=mixing.h_dry_static, gz=mixing.gz, lv=mixing.lv,
            cpn=mixing.cpn,
            m_profile=mixing.m_profile, ment=mixing.ment, elij=mixing.elij,
            clw=mixing.clw, ep=mixing.ep, inb=mixing.inb,
            c_l=config.c_l_emanuel,
            sigd=config.downdraft_sigd,
            sigs=config.downdraft_sigs,
            omtrain=config.downdraft_omtrain_pa_s,
            omtsnow=config.downdraft_omtsnow_pa_s,
            coeffr=config.downdraft_coeffr,
            coeffs=config.downdraft_coeffs,
            freeze_transition_K=config.downdraft_freeze_transition_K,
            inertia_scale_hPa=config.downdraft_inertia_scale_hPa,
            taper_p_fraction=config.downdraft_taper_p_fraction,
            dhdp_min=config.downdraft_dhdp_min,
            ep_gate_threshold=config.downdraft_ep_gate_threshold,
            ep_gate_width=config.downdraft_ep_gate_width,
        )
        ft = ft + downdraft.dT_dt
        fq = fq + downdraft.dq_v_dt

    # -- Cloud-top adjustment (oracle l.940-950), on FT+FQ incl. downdraught --
    phf = _rev(p_half) * _HPA_PER_PA
    dph = phf[:, :-1] - phf[:, 1:]
    fq = _apply_frac(fq, mixing.inb, mixing.frac, dph, mixing.lv)
    ft = _apply_frac(ft, mixing.inb, mixing.frac, dph, mixing.cpn)
    dT_dt = _rev(ft)
    dq_v_dt = _rev(fq)

    # -- Column water -> rain ---------------------------------------------------
    # ``net_water`` < 0 is the precipitating condensate (WDTRAIN: EP*CLW and
    # AWAT, minus any downdraught re-evaporation): CONVECT's PRECIP.  It is
    # emitted as the in-updraught rain source ``dq_r_conv_dt`` (column-
    # integrated into same-step surface precipitation by the physics
    # pipeline), distributed by the column drying weight.  No heating
    # accompanies it: the latent heat is already in FT.
    net_water = jnp.sum(dq_v_dt * dp, axis=-1) / constants.g
    drying = jnp.maximum(-dq_v_dt, 0.0)
    drying_col = jnp.sum(drying * dp, axis=-1) / constants.g
    # The 1e-30 divisor floor goes through _div_floor (same value; a floored,
    # non-convecting column then has finite fp32 derivatives).
    # fp32 derivative limitation of this weight: see the docstring.
    add_weight = _div_floor(drying, drying_col[:, None], 1e-30)  # ∫w dp/g = 1
    _residual = jnp.maximum(-net_water, 0.0)
    dq_r_conv_dt = _residual[:, None] * add_weight    # ∫dq_r dp/g = _residual
    # A residual that cannot be placed (no drying anywhere) is spread uniformly
    # in mass rather than dropped.
    _placed = jnp.sum(dq_r_conv_dt * dp, axis=-1) / constants.g
    _col_dp = jnp.sum(dp, axis=-1, keepdims=True)
    dq_r_conv_dt = dq_r_conv_dt + jnp.where(
        (drying_col <= 0.0)[:, None],
        (_residual - _placed)[:, None] * constants.g / _col_dp,
        0.0,
    )

    # -- ENTS: exact enthalpy conservation over levels 1..INB (l.969-984) -----
    # ENTS = sum (CPN*FT + LV*FQ) dp / (PH(1)-PH(INB+1)); FT -= ENTS/CPN.
    cpn = constants.c_pd * (1.0 - q_v) + constants.c_pv * q_v
    lv_eff = latent_heat_vaporization(T, c_liquid=config.c_l_emanuel)
    ents_mask = mixing.convective_layer_mask
    layer_mass = jnp.sum(ents_mask * dp, axis=-1)
    mse_tend = cpn * dT_dt + lv_eff * dq_v_dt
    ents = jnp.sum(ents_mask * mse_tend * dp, axis=-1) / jnp.maximum(
        layer_mass, 1.0,
    )
    dT_dt = dT_dt - ents_mask * (ents[:, None] / cpn)

    out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=jnp.zeros_like(dq_v_dt),
        dq_r_conv_dt=dq_r_conv_dt,
        precip_frozen_sfc=jnp.zeros_like(cape),  # L_v-only: liquid at sfc (F29)
        cape=cape,
        convective_mask=cape_weight,
        du_dt_conv=None,
        dv_dt_conv=None,
    )
    # Carry CONVECT's CBMF (>= 0 by the oracle's own MAX; no cap).
    cbmf_carry = mixing.cbmf
    conv_prog_profile_new = (
        jnp.zeros((ncol, nlev), dtype=T.dtype).at[:, -1].set(cbmf_carry)
    )
    return out, conv_prog_profile_new
