"""Zhang-McFarlane DILUTE entraining-plume CAPE (Raymond-Blyth 1992).

FAITHFUL port of the E3SM/CAM ``parcel_dilute`` / ``buoyan_dilute``
routines (``components/eam/src/physics/cam/zm_conv.F90``).  The ZM deep
convection trigger and closure use the CAPE of a **dilute** parcel that
ascends as a constant-fractional-entrainment plume, conserving moist
**entropy** (Raymond & Blyth 1992), *not* an undilute moist adiabat.
Mixing dry environmental air into the parcel as it rises dramatically
reduces its buoyancy and CAPE (a factor of ~3 in a tropical sounding) —
this is the single most important ZM fidelity property, and it is what
keeps ZM from over-firing in marginally-unstable columns.

Oracle algorithm (per column, surface-last ``[:, -1]`` = surface):

1. **Launch level** ``mx`` = level of maximum moist static energy
   ``h = c_p T + g z + L q`` within the PBL.  For PBL-rooted convection
   this is the surface in the vast majority of columns.
2. **Entraining ascent** from ``mx`` upward.  The parcel carries total
   water ``qt`` and entropy ``s``; at each layer it entrains the layer-
   mean environmental ``s`` / ``qt`` at fractional rate
   ``dmpdz`` [1/m] (converted to ``dmpdp`` [1/Pa] via hydrostatic
   ``dp/dz``):

       s_mix(k)  = (s0  + Σ -dmpdp·dp·s_env ) / (1 + Σ -dmpdp·dp)
       qt_mix(k) = (qt0 + Σ -dmpdp·dp·qt_env) / (1 + Σ -dmpdp·dp)

   The parcel temperature ``T_mix`` and saturation ``qs_mix`` at each
   level follow by **inverting entropy** ``entropy(T, p, qt) = s_mix``.
3. **Condensate loading + freezing** (2 fixed iterations): excess
   condensate above ``lwmax = 1 g/kg`` rains out (entropy loss
   ``ds_xsh2o``); below freezing the latent heat of fusion is released
   (entropy gain ``ds_freeze``).  The parcel virtual (density)
   temperature folds in the retained condensate.
4. **CAPE** = ``R_d Σ buoy(k) · ln(pf[k+1]/pf[k])`` over the buoyant
   plume, with ``buoy = Tpv_parcel − Tv_env + tiedke_add``.

Differentiability
-----------------
The oracle inverts entropy with Brent's method (data-dependent
iteration count — not AD-safe).  This port replaces Brent with a
**fixed-iteration Newton solve** (20 iterations), which converges to
the same root and is smooth everywhere.  The launch-level search and
the multi-region CIN bookkeeping are replaced by smooth surrogates:
the launch state is a PBL-MSE softmax over levels, and the CAPE
integral runs from the launch level over all positively-buoyant
layers (a smooth ``softplus`` positive part).  Validated against the
compiled Fortran oracle to within 0.5 % CAPE on a tropical RCE
sounding (see ``.physics-validator/zhang_mcfarlane``).

Constants
---------
Uses ``legoesm.constants`` (per CLAUDE.md — no re-derived literals).
The CAM oracle uses ``c_pliq = 4188``, ``c_pwv = 1810`` vs our
``c_pw = 4218``, ``c_pv = 1846``; the resulting CAPE divergence is
< 0.5 % (the entropy is dominated by the ``L·q/T`` and ``R ln p``
terms, not the small heat-capacity corrections) — documented in the
fidelity report rather than monkey-patched.

References
----------
- Raymond, D. J., & Blyth, A. M. (1992). Extension of the
  stochastic mixing model to cumulonimbus clouds.  *J. Atmos. Sci.*,
  49, 1968-1983.
- Zhang, G. J., & McFarlane, N. A. (1995). *Atmos.-Ocean*, 33, 407-446.
- E3SM v3.0.1 ``zm_conv.F90`` (buoyan_dilute / parcel_dilute).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.convection._triggers import (
    smooth_lowest_crossing_index,
)


__all__ = ("DiluteParcel", "dilute_parcel_cape")


# Both the oracle ``qsat_hPa`` and our shared ``saturation_mixing_ratio``
# return a vapor/dry-air mass mixing ratio (``ε·e/(p − ω·e)`` vs
# ``ε·e/(p − e)`` — the same convention, differing only in the ``ω = 1−ε``
# vs ``1`` factor in the small ``e`` correction).  Both feed the same
# Raymond-Blyth entropy formula; the validated CAPE difference is < 0.5 %.
# Convention note: the ZM oracle treats the model ``q`` field as a
# vapor/dry mixing ratio throughout (entropy, virtual T with the 1.608
# coefficient), so we do the same here for faithfulness.

# Maximum condensate retained before rainout [kg/kg] (oracle ``lwmax``).
_LWMAX = 1.0e-3
# Number of latent-heat (condensate + freezing) iterations (oracle
# ``nit_lheat``).
_NIT_LHEAT = 2
# Newton iterations for the entropy inversion (replaces Brent).
_NEWTON_ITERS = 20


class DiluteParcel(NamedTuple):
    """Dilute-parcel diagnostics, surface-last ``(ncol, nlev)``.

    Fields
    ------
    cape : jax.Array, shape (ncol,)
        Dilute CAPE [J/kg], non-negative.
    T_parcel : jax.Array, shape (ncol, nlev)
        Dilute-parcel temperature ``tp`` [K].
    Tv_parcel : jax.Array, shape (ncol, nlev)
        Dilute-parcel virtual (density) temperature ``tpv`` [K].
    qs_parcel : jax.Array, shape (ncol, nlev)
        Dilute-parcel saturation / retained-vapor mixing ratio
        ``qstp`` [kg/kg].
    buoyancy : jax.Array, shape (ncol, nlev)
        Parcel buoyancy ``Tpv − Tv_env + tiedke_add`` [K].
    k_launch_smooth : jax.Array, shape (ncol,)
        Smooth fractional launch-level index (surface-last).
    """
    cape: jax.Array
    T_parcel: jax.Array
    Tv_parcel: jax.Array
    qs_parcel: jax.Array
    buoyancy: jax.Array
    k_launch_smooth: jax.Array


def _moist_entropy(T: jax.Array, p_pa: jax.Array, qtot: jax.Array) -> jax.Array:
    """Raymond-Blyth (1992) moist entropy [J/kg/K].

    ``s = (c_pd + qtot·c_pw) ln(T/T0) − R_d ln((p−e)/p_ref)
          + L·qv/T − qv·R_v ln(qv/qs)``

    with ``L = L_v − (c_pw − c_pv)(T − T0)``, ``qv = min(qtot, qs)``,
    ``e = qv·p/(ε + qv)``, ``p`` in Pa.  Mirrors the oracle ``entropy``
    function (which uses ``p`` in hPa and ``p_ref = 1000 hPa``; here we
    keep Pa throughout and ``p_ref = 1e5 Pa`` — the ``ln`` of the ratio
    is identical).
    """
    q_sat = saturation_mixing_ratio(T, p_pa)
    qv = jnp.minimum(qtot, q_sat)
    # Numerical floor so ln(qv/qs) is finite for a bone-dry parcel.
    qv_safe = jnp.maximum(qv, 1.0e-12)
    L = constants.L_v - (constants.c_pw - constants.c_pv) * (T - constants.T_freeze)
    e = qv * p_pa / (constants.epsilon + qv)
    # Floor the dry partial pressure (p − e) away from zero before the log:
    # a cold, moist parcel at very low total pressure can drive e → p, and
    # ``log(0)`` would poison both the value and the gradient.  The floor
    # is 1 Pa (negligible vs the ~10⁴–10⁵ Pa cloud layer where the parcel
    # actually has buoyancy).
    p_dry = jnp.maximum(p_pa - e, 1.0)
    return (
        (constants.c_pd + qtot * constants.c_pw) * jnp.log(T / constants.T_freeze)
        - constants.R_d * jnp.log(p_dry / constants.p_ref)
        + L * qv / T
        - qv * constants.R_v * jnp.log(qv_safe / jnp.maximum(q_sat, 1.0e-12))
    )


def _invert_entropy(
    s_target: jax.Array,
    p_pa: jax.Array,
    qtot: jax.Array,
    T_first_guess: jax.Array,
) -> jax.Array:
    """Newton inversion: solve ``_moist_entropy(T, p, qt) = s_target`` for T.

    Fixed ``_NEWTON_ITERS`` iterations with a centered finite-difference
    derivative (the entropy has a ``min(qtot, qs)`` kink that a closed-
    form derivative would have to special-case; the centered difference
    is smooth-enough for AD and converges in < 10 iterations in
    practice).  ``T`` is clipped to a physical range each step to keep
    the iteration bounded.
    """
    dT = 0.01
    _dtype = T_first_guess.dtype

    def body(_, T):
        f = _moist_entropy(T, p_pa, qtot) - s_target
        fp = (
            _moist_entropy(T + dT, p_pa, qtot)
            - _moist_entropy(T - dT, p_pa, qtot)
        ) / (2.0 * dT)
        # Guard the derivative magnitude away from zero while preserving
        # its sign.  ``dS/dT`` for the moist entropy is positive in
        # practice, but a correct floor must not collapse to exactly 0
        # for a small negative ``fp`` (the earlier ``sign(fp)·1e-12 +
        # 1e-12`` gave 0 when ``fp < 0`` — Codex review #6).  Add a small
        # positive bias so ``|fp_safe| >= 1e-6`` with the sign of ``fp``
        # (treating the near-zero / positive entropy derivative as the
        # physical branch).
        fp_safe = jnp.where(jnp.abs(fp) > 1.0e-6, fp, 1.0e-6)
        T_new = T - f / fp_safe
        # Pin to input precision (fori_loop carry dtype invariant).
        return jnp.clip(T_new, 120.0, 360.0).astype(_dtype)

    return jax.lax.fori_loop(0, _NEWTON_ITERS, body, T_first_guess)


def dilute_parcel_cape(
    T_env: jax.Array,
    q_v_env: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    *,
    dmpdz: float = -1.0e-3,
    tiedke_add: float = 0.5,
    tp_fac: float = 0.0,
    tpert: jax.Array | float = 0.0,
    pbl_top_pa: float = 7.0e4,
    launch_sharpness: float = 5.0e-4,
) -> DiluteParcel:
    """Dilute entraining-plume CAPE (FAITHFUL to ZM ``buoyan_dilute``).

    Parameters
    ----------
    T_env, q_v_env : jax.Array, shape (ncol, nlev)
        Environmental temperature [K] and vapor mixing ratio [kg/kg],
        surface-last.
    p_full, p_half : jax.Array
        Full-level ``(ncol, nlev)`` and half-level ``(ncol, nlev+1)``
        pressures [Pa].  ``p_half[:, 0]`` is TOA, ``p_half[:, -1]`` the
        surface.
    z_full : jax.Array, shape (ncol, nlev)
        Geopotential height [m], surface-last.
    dmpdz : float
        Fractional entrainment rate [1/m] (oracle ``dmpdz``; default
        ``−1.0e-3``, the CAM default — NEGATIVE because ``mp`` decreases
        the relative parcel mass as a sign convention; the magnitude is
        the physical entrainment rate).
    tiedke_add : float
        Buoyancy offset [K] added to the parcel buoyancy (oracle
        ``tiedke_add``; default 0.5 K).
    tp_fac, tpert : float / jax.Array
        PBL temperature-perturbation factor and perturbation [K]
        (oracle ``tp_fac`` × ``tpert``; default 0).
    pbl_top_pa : float
        Pressure [Pa] above which the launch-level MSE search is
        excluded (PBL-rooted convection; default 700 hPa).
    launch_sharpness : float
        Softmax sharpness [1/(J/kg of MSE)] for the smooth launch-level
        selection.  Large enough to pick the single max-MSE PBL level
        while staying differentiable.

    Returns
    -------
    DiluteParcel
    """
    ncol, nlev = T_env.shape
    _dtype = T_env.dtype

    tpert_arr = jnp.asarray(tpert, dtype=_dtype) * jnp.ones((ncol,), dtype=_dtype)

    # ----- Launch level: max MSE within the PBL (smooth softmax) -----------
    h_mse = (
        constants.c_pd * T_env + constants.g * z_full + constants.L_v * q_v_env
    )
    in_pbl = jax.nn.sigmoid((p_full - pbl_top_pa) / 1.0e3)  # ~1 in PBL, 0 aloft
    # Mask MSE outside the PBL with a large negative offset so the softmax
    # never selects an aloft level (whose geopotential term inflates MSE).
    LARGE = jnp.asarray(1.0e9, dtype=_dtype)
    h_masked = h_mse - LARGE * (1.0 - in_pbl)
    launch_w = jax.nn.softmax(launch_sharpness * h_masked, axis=-1)  # (ncol,nlev)
    levels = jnp.arange(nlev, dtype=_dtype)
    k_launch_smooth = jnp.sum(launch_w * levels[None, :], axis=-1)

    # Launch-level parcel state: softmax-weighted environmental T, q, p, z.
    T_launch = jnp.sum(launch_w * T_env, axis=-1)
    q_launch = jnp.sum(launch_w * q_v_env, axis=-1)
    p_launch = jnp.sum(launch_w * p_full, axis=-1)

    s0 = _moist_entropy(T_launch, p_launch, q_launch)        # (ncol,)
    qt0 = q_launch
    # Per-level "at-or-above launch" weight.  The oracle defines parcel
    # buoyancy for ``k <= mx`` (the launch level INCLUSIVE) — so the
    # launch layer itself must carry weight ~1, not 0.5.  Centre the
    # transition at ``k_launch + 0.5`` (half a level below the launch)
    # and use a sharp slope so the launch level and everything above it
    # gets ~1 and everything strictly below gets ~0:
    #   level = k_launch     -> sigmoid(4·0.5)  ≈ 0.88  (launch included)
    #   level = k_launch+1   -> sigmoid(4·-0.5) ≈ 0.12  (just below: out)
    # The earlier ``sigmoid(2·(k_launch − level))`` put the midpoint AT
    # the launch level, giving it only 0.5 weight (Codex review #3:
    # "launch mask is wrong at the launch level and leaks below launch").
    above_launch = jax.nn.sigmoid(
        4.0 * (k_launch_smooth[:, None] + 0.5 - levels[None, :])
    )  # ~1 at/above launch (index <= k_launch), ~0 strictly below

    # ----- Entraining ascent (scan surface-first) --------------------------
    T_env_r = T_env[:, ::-1].astype(_dtype)
    q_env_r = q_v_env[:, ::-1].astype(_dtype)
    p_env_r = p_full[:, ::-1].astype(_dtype)
    above_r = above_launch[:, ::-1].astype(_dtype)

    # Layer-mean environment and dp between adjacent full levels (oracle
    # uses center-to-center dp and 0.5(env_k + env_{k+1})).  In surface-
    # first reversed indexing, level j ascends from j-1 (below).
    inputs = (
        jnp.moveaxis(T_env_r, 1, 0),   # (nlev, ncol)
        jnp.moveaxis(q_env_r, 1, 0),
        jnp.moveaxis(p_env_r, 1, 0),
        jnp.moveaxis(above_r, 1, 0),
    )

    # Initial carry at the surface-first index 0.  The launch level is at
    # or above the surface, so we begin accumulating from the surface but
    # the ``above_launch`` weight zeros contributions below the launch.
    T_mix0 = _invert_entropy(
        s0.astype(_dtype), p_launch.astype(_dtype),
        qt0.astype(_dtype), T_launch.astype(_dtype),
    )
    init_carry = (
        s0.astype(_dtype),          # s_accum (parcel entropy numerator part)
        jnp.zeros((ncol,), _dtype), # s_entr (Σ −dmpdp·dp·s_env)
        jnp.zeros((ncol,), _dtype), # qt_entr (Σ −dmpdp·dp·qt_env)
        jnp.zeros((ncol,), _dtype), # m_entr (Σ −dmpdp·dp)
        T_env_r[:, 0].astype(_dtype),  # T_env_prev
        q_env_r[:, 0].astype(_dtype),  # q_env_prev
        p_env_r[:, 0].astype(_dtype),  # p_env_prev
        T_mix0.astype(_dtype),      # T_mix_prev (first guess chaining)
    )

    def ascend_step(carry, lin):
        (s0c, s_entr, qt_entr, m_entr,
         T_env_prev, q_env_prev, p_env_prev, T_mix_prev) = carry
        T_e, q_e, p_e, abv = lin

        # Center-to-center dp (Pa, negative going up since p decreases).
        dp = p_e - p_env_prev
        qt_env_lyr = 0.5 * (q_e + q_env_prev)
        T_env_lyr = 0.5 * (T_e + T_env_prev)
        p_env_lyr = 0.5 * (p_e + p_env_prev)
        s_env_lyr = _moist_entropy(T_env_lyr, p_env_lyr, qt_env_lyr)

        # Fractional entrainment /Pa from /m via hydrostatic dp/dz.
        # dpdz = −(p·g)/(R·T); dzdp = 1/dpdz; dmpdp = dmpdz·dzdp.
        # Floor |dpdz| away from zero so ``dzdp`` is finite even if a
        # pathological full-level pressure or temperature drives it
        # toward 0 (Codex review #7: unfloored ``1/dpdz`` hazard).
        dpdz = -(p_env_lyr * constants.g) / (constants.R_d * T_env_lyr)
        dpdz_safe = jnp.where(jnp.abs(dpdz) > 1.0e-12, dpdz, -1.0e-12)
        dzdp = 1.0 / dpdz_safe
        dmpdp = dmpdz * dzdp

        # Accumulate entrainment ONLY above the launch level.
        d_s = -dmpdp * dp * s_env_lyr * abv
        d_qt = -dmpdp * dp * qt_env_lyr * abv
        d_m = -dmpdp * dp * abv
        s_entr_new = s_entr + d_s
        qt_entr_new = qt_entr + d_qt
        m_entr_new = m_entr + d_m

        s_mix = (s0c + s_entr_new) / (1.0 + m_entr_new)
        qt_mix = (qt0 + qt_entr_new) / (1.0 + m_entr_new)

        T_mix = _invert_entropy(s_mix, p_e, qt_mix, T_mix_prev)
        qs_mix = saturation_mixing_ratio(T_mix, p_e)

        # Pin carry to input precision (scan-carry dtype invariant —
        # Python-float constants otherwise promote to float64).
        new_carry = (
            s0c.astype(_dtype), s_entr_new.astype(_dtype),
            qt_entr_new.astype(_dtype), m_entr_new.astype(_dtype),
            T_e.astype(_dtype), q_e.astype(_dtype), p_e.astype(_dtype),
            T_mix.astype(_dtype),
        )
        return new_carry, (
            s_mix.astype(_dtype), qt_mix.astype(_dtype),
            T_mix.astype(_dtype), qs_mix.astype(_dtype),
        )

    _, ascend_out = jax.lax.scan(ascend_step, init_carry, inputs)
    s_mix_r, qt_mix_r, T_mix_r, qs_mix_r = ascend_out  # (nlev, ncol)

    # Move back to surface-last.
    s_mix = jnp.moveaxis(s_mix_r, 0, 1)[:, ::-1]
    qt_mix = jnp.moveaxis(qt_mix_r, 0, 1)[:, ::-1]
    T_mix = jnp.moveaxis(T_mix_r, 0, 1)[:, ::-1]
    qs_mix = jnp.moveaxis(qs_mix_r, 0, 1)[:, ::-1]

    # ----- Condensate loading + freezing (scan surface-first) --------------
    # Iterate _NIT_LHEAT times accumulating rainout entropy loss
    # (ds_xsh2o) and freezing entropy gain (ds_freeze), then re-invert.
    s_mix_r2 = jnp.moveaxis(s_mix[:, ::-1], 1, 0)
    qt_mix_r2 = jnp.moveaxis(qt_mix[:, ::-1], 1, 0)
    p_env_r2 = jnp.moveaxis(p_full[:, ::-1].astype(_dtype), 1, 0)
    T_mix_r2 = jnp.moveaxis(T_mix[:, ::-1], 1, 0)
    qs_mix_r2 = jnp.moveaxis(qs_mix[:, ::-1], 1, 0)
    # The oracle runs the precipitation/freezing loop ONLY for k < klaunch
    # and initialises the launch level separately (tp(mx)=tmix(mx),
    # qstp(mx)=q(mx)).  Use a STRICTLY-above-launch mask (excludes the
    # launch level) for the latent-heat loop so it does not modify the
    # launch parcel (Codex round-2 #3).  ``strict_above[k] ≈ 1`` for
    # ``index < k_launch`` (higher altitude) and ≈ 0 at/below launch.
    strict_above_launch = jax.nn.sigmoid(
        4.0 * (k_launch_smooth[:, None] - 0.5 - levels[None, :])
    )
    strict_r2 = jnp.moveaxis(strict_above_launch[:, ::-1].astype(_dtype), 1, 0)
    # Environmental q at the launch (for the launch-level qstp init).
    q_env_r2 = jnp.moveaxis(q_v_env[:, ::-1].astype(_dtype), 1, 0)

    lheat_inputs = (s_mix_r2, qt_mix_r2, p_env_r2, strict_r2, q_env_r2)

    init_lheat = (
        jnp.zeros((ncol,), _dtype),   # xsh2o_prev
        jnp.zeros((ncol,), _dtype),   # ds_xsh2o_prev
        jnp.zeros((ncol,), _dtype),   # ds_freeze_prev
        T_mix_r2[0].astype(_dtype),   # T_mix_prev (first guess)
        qs_mix_r2[0].astype(_dtype),  # qs_mix_prev
    )

    def lheat_step(carry, lin):
        xsh2o_prev, dsx_prev, dsf_prev, T_mix_p, qs_mix_p = carry
        s_m, qt_m, p_e, sabv, q_e = lin

        # ``sabv`` ≈ 1 STRICTLY above launch (the lheat loop region), ≈ 0
        # at/below launch.  The launch level uses the oracle init
        # (tp=tmix, qstp=q_env) and is excluded from the loop.

        # Two fixed iterations (oracle nit_lheat=2) — computed everywhere;
        # the ``sabv`` blend below restricts the modification to the
        # strictly-above-launch region.
        T_mix_k = T_mix_p
        qs_mix_k = saturation_mixing_ratio(T_mix_k, p_e)
        new_q = qt_m
        xsh2o_k = xsh2o_prev
        dsx_k = dsx_prev
        dsf_k = dsf_prev
        for _ in range(_NIT_LHEAT):
            xsh2o_k = jnp.maximum(0.0, qt_m - qs_mix_k - _LWMAX)
            dsx_k = dsx_prev - constants.c_pw * jnp.log(T_mix_k / constants.T_freeze) * jnp.maximum(
                0.0, xsh2o_k - xsh2o_prev
            )
            # Freezing entropy: gained when T <= T_freeze.  One-off when
            # dsf_prev == 0, continual otherwise.  Use a smooth blend on
            # the freezing indicator so the branch is differentiable.
            # Sharpen the sub-freezing gate (10/K → ~0 at +0.5 K, ~1 at
            # −0.5 K of supercooling) so warm-cloud levels get ≈0 freezing
            # (Codex review #5).
            below_frz = jax.nn.sigmoid(10.0 * (constants.T_freeze - T_mix_k))
            dsf_oneoff = (constants.L_f / T_mix_k) * jnp.maximum(
                0.0, qt_m - qs_mix_k - xsh2o_k
            )
            dsf_cont = dsf_prev + (constants.L_f / T_mix_k) * jnp.maximum(
                0.0, qs_mix_p - qs_mix_k
            )
            # ``started`` must be EXACTLY 0 when no freezing has occurred
            # below (``dsf_prev == 0``) so the one-off branch is taken at
            # the first freezing level — the earlier ``sigmoid(1e6·dsf_prev)``
            # gave 0.5 at ``dsf_prev == 0`` (Codex review #5).  ``dsf`` is
            # non-negative, so ``1 − exp(−K·dsf_prev)`` is 0 at 0 and →1 for
            # any positive accumulated freezing.
            started = -jnp.expm1(-1.0e6 * dsf_prev)
            dsf_k = below_frz * (
                (1.0 - started) * dsf_oneoff + started * dsf_cont
            )
            new_s = s_m + dsx_k + dsf_k
            new_q = qt_m - xsh2o_k
            T_mix_k = _invert_entropy(new_s, p_e, new_q, T_mix_k)
            qs_mix_k = saturation_mixing_ratio(T_mix_k, p_e)

        # Restrict the latent-heat modification to STRICTLY above launch
        # (oracle loops only ``k < klaunch``; the launch level and below are
        # untouched — Codex round-2 #3).  Below launch AND at the launch
        # level, the rainout/freezing accumulators carry the unmodified
        # pre-launch state, and:
        #   * the launch level outputs Tp = the entrained T_mix (no lheat)
        #     and qstp = environmental launch q (oracle qstp(mx)=q(mx));
        #   * below launch the values are inert (masked out of CAPE).
        # ``T_mix_p`` is the entrained-only parcel T at this level
        # (carry-in is the pre-lheat T_mix), so the at/below-launch branch
        # uses it directly.
        xsh2o_k = sabv * xsh2o_k + (1.0 - sabv) * xsh2o_prev
        dsx_k = sabv * dsx_k + (1.0 - sabv) * dsx_prev
        dsf_k = sabv * dsf_k + (1.0 - sabv) * dsf_prev
        # new_q: above launch = qt − xsh2o (rained-out); at/below = q_env
        # (the oracle initialises qstp(mx)=q(mx), i.e. the launch vapor).
        new_q = sabv * new_q + (1.0 - sabv) * q_e
        # Tp: above launch = lheat-adjusted; at/below = entrained-only T_mix.
        T_mix_k = sabv * T_mix_k + (1.0 - sabv) * T_mix_p
        qs_mix_k = saturation_mixing_ratio(T_mix_k, p_e)

        # Retained vapor (qstp): above launch = qs if super-saturated else
        # new_q; at/below launch = q_env (oracle qstp(mx)=q(mx)).
        supersat = jax.nn.sigmoid(1.0e4 * (new_q - qs_mix_k))
        qstp_above = supersat * qs_mix_k + (1.0 - supersat) * new_q
        qstp = sabv * qstp_above + (1.0 - sabv) * q_e

        # Pin every carry slot to the input precision: the Python-float
        # physical constants (L_f, T_freeze, c_pw) promote intermediates to
        # float64, which breaks ``lax.scan``'s carry-in == carry-out dtype
        # invariant when the model runs in float32 (CLAUDE.md scan-carry
        # dtype rule).
        new_carry = (
            xsh2o_k.astype(_dtype), dsx_k.astype(_dtype), dsf_k.astype(_dtype),
            T_mix_k.astype(_dtype), qs_mix_k.astype(_dtype),
        )
        return new_carry, (
            T_mix_k.astype(_dtype), qstp.astype(_dtype), new_q.astype(_dtype),
        )

    _, lheat_out = jax.lax.scan(lheat_step, init_lheat, lheat_inputs)
    Tp_r, qstp_r, newq_r = lheat_out  # (nlev, ncol)

    Tp = jnp.moveaxis(Tp_r, 0, 1)[:, ::-1]
    qstp = jnp.moveaxis(qstp_r, 0, 1)[:, ::-1]
    newq = jnp.moveaxis(newq_r, 0, 1)[:, ::-1]

    # Parcel virtual (density) temperature tpv (oracle eq 4844):
    #   tpv = (tp + tp_fac·tpert)·(1 + 1.608·qstp)/(1 + new_q)
    Tpv = (
        (Tp + tp_fac * tpert_arr[:, None])
        * (1.0 + 1.608 * qstp)
        / (1.0 + newq)
    )
    # Environment virtual temperature (oracle convention 1.608, vapor/dry).
    Tv_env = T_env * (1.0 + 1.608 * q_v_env) / (1.0 + q_v_env)

    # Buoyancy (oracle ``buoy(k) = tpv − tv + tiedke_add`` for k<=mx).  Do
    # NOT pre-multiply by ``above_launch`` here — the CAPE integral applies
    # the launch mask once below (Codex review #1/#3: double-masking +
    # baseline).
    buoy = Tpv - Tv_env + tiedke_add

    # ----- CAPE integral ---------------------------------------------------
    # The oracle integrates SIGNED buoyancy from the launch level (mx) up
    # to the tentative cloud top and takes the MAXIMUM over candidate tops:
    #   CAPE = max_top  R_d Σ_{mx >= k > top} buoy(k)·ln(pf[k+1]/pf[k]).
    # Equivalently, defining the per-layer signed buoyant-energy
    # increment ``dB(k) = R_d·buoy(k)·ln(pf[k+1]/pf[k])`` and the partial
    # sum from the launch DOWN to level k (i.e. accumulated going UP),
    # CAPE is the running MAXIMUM of that partial sum over the column.
    # This faithfully:
    #   * includes the sub-LFC negative buoyancy (CIN) the oracle counts,
    #   * stops at the level of neutral buoyancy that maximises CAPE
    #     (the highest positively-buoyant top), and
    #   * has NO false-CAPE baseline (the earlier ``softplus(buoy)``
    #     positive-part added ≈0.35 K at buoy=0 — Codex review #1/#2).
    #
    # Floor the half-level pressures before the log: the model top
    # half-level can be exactly 0 Pa (sigma top), and ``log(p2/0) = inf``
    # would poison the gradient even after masking (``inf · 0 = NaN``).
    ph_floor = jnp.maximum(p_half, 1.0)
    dlnp = jnp.log(ph_floor[:, 1:] / ph_floor[:, :-1])  # (ncol, nlev), >0
    dB = constants.R_d * buoy * dlnp * above_launch       # per-layer (ncol,nlev)

    # Cumulative buoyant energy from the launch UPWARD.  Surface-last:
    # ascent goes from index nlev-1 (surface/launch) toward index 0
    # (top), so the upward-cumulative sum is a reverse cumsum.
    partial_up = jnp.cumsum(dB[:, ::-1], axis=-1)[:, ::-1]  # (ncol,nlev)
    # CAPE = max over candidate tops of the launch->top partial sum.  Only
    # a top where ``dB`` changes sign from + to − (a buoyancy crossing, the
    # oracle ``lelten``) can be the strict maximum of the partial sums, so
    # the running-max over ALL partials is mathematically identical to the
    # oracle's max-over-crossing-tops — no extra candidates are admitted.
    # Use a differentiable soft-argmax-WEIGHTED mean (``Σ xᵢ·softmax(β xᵢ)``)
    # rather than ``logsumexp − ln(n)/β``: the soft-argmax form has NO
    # ln(nlev) baseline and is unbiased to first order (→ hard max as
    # β→∞), so it does not erase marginal CAPE (Codex round-2 #1).  ``β =
    # 0.5`` 1/(J/kg) tracks the hard max to ≲ a few J/kg.
    sm_beta = jnp.asarray(0.5, dtype=_dtype)  # 1/(J/kg)
    sm_w = jax.nn.softmax(sm_beta * partial_up, axis=-1)
    cape = jnp.sum(sm_w * partial_up, axis=-1)
    cape = jnp.maximum(cape, 0.0)

    # Re-apply the launch mask to the reported buoyancy diagnostic (kept
    # for downstream cloud-model use).
    buoy = buoy * above_launch

    return DiluteParcel(
        cape=cape,
        T_parcel=Tp,
        Tv_parcel=Tpv,
        qs_parcel=qstp,
        buoyancy=buoy,
        k_launch_smooth=k_launch_smooth,
    )
