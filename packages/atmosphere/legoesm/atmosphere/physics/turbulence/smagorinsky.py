"""Smagorinsky turbulence scheme.

DEFAULT ARM ``stability_form="wrf_km3"``: port of WRF v4.8.0 ``km_opt = 3``
(``dyn_em/module_diffusion_em.F``: ``smag_km`` with ``mix_isotropic = 0``,
``calculate_N2``, and the vertical-shear terms of ``cal_deform_and_div`` for a
flat, horizontally homogeneous column), see :func:`wrf_km3_coefficients`.
``K_m = min(max(c_s² dz² √max(0, D² − N²/Pr), 1e-6 dz²), mub dz²/dt)``,
``K_h = min(K_m/Pr, mub dz²/dt)``, ``c_s = 0.25``, ``Pr = 1/3``, ``mub = 0.1``,
``dz`` the layer thickness, ``D²`` the mass-level average of the w-level shear
(zero at the surface and top w-levels), ``N²`` WRF's moist/dry buoyancy
frequency.  Running-oracle receipt:
``tests/atmosphere/hydrostatic/unit/test_smagorinsky_wrf_km3_oracle.py``.
Declared departures: the implicit Thomas solve and our surface-flux boundary
condition replace WRF's explicit ``vertical_diffusion_2`` (unstable at GCM time
steps); WRF's eta extrapolation coefficients use ``eta = (p − p_top)/(p_s − p_top)``
of our hydrostatic column; ``q_c = q_i = 0`` where a lane does not pass them;
the switches are smooth (repo helpers) outside the oracle test.

The ``"lilly"`` and ``"deardorff"`` arms below are the earlier Smagorinsky–Lilly
closure, kept selectable (the LES-tuned SCM overlay runs ``"lilly"``).

Strain-dependent (deformation-based) eddy viscosity

    K_m = (C_s · l)^2 · |S| · √(max(0, 1 − Ri/Pr_t)),   K_h = K_m / Pr_t

following Smagorinsky (1963), with a Lilly-type buoyancy cutoff that shuts
mixing off in strongly stable layers (Ri ≥ Pr_t).  Lilly (1962) left the Ri
dependence and ``K_h/K_m`` as undetermined functions in his GENERAL theory,
but his equilibrium EXPERIMENT already gives THIS ``√(1 − Ri/Pr_t)`` stability
factor with ``K_h/K_m = 1`` (hence ``Pr_t = 1``) and ``K_m = 0`` for ``Ri > 1``
— all reproduced here.  Only the JAX AD-safe NUMERICS are modern: the
``max(0, ·)`` clamp and the double-``where`` guard keep the ``√`` and its
reverse-mode cotangent finite at the ``Ri = Pr_t`` kink (plus a ``S²+1e-10``
floor); the ``√`` ramp itself is Lilly's, not a modern replacement of a hard
on/off.  The deformation is a 1-D PROXY: the resolved vertical shear |S| =
√((∂u/∂z)² + (∂v/∂z)²) is SUBSTITUTED for Smagorinsky's horizontal deformation
(a single column carries no horizontal strain); the mixing length ``l`` is the
Blackadar (1962) asymptotic form shared with the other turbulence closures.

Vertical mixing is applied implicitly using the Thomas algorithm
to ensure numerical stability at any time step.

Faithfulness to Smagorinsky (1963) / Lilly (1962)
-------------------------------------------------
This is a MODERN 1-D specialization of the Smagorinsky-Lilly idea, NOT the literal
published scheme: Smagorinsky (1963) used the HORIZONTAL deformation and a GRID
length scale (reporting ``k_s ≈ 0.28``).  Lilly (1962) left ``K_h/K_m`` and the
Richardson dependence as undetermined functions in his GENERAL theory, but his
EXPERIMENTS fixed ``K_h/K_m = 1`` and ``K_m → 0`` for ``Ri > 1``.
FAITHFUL (Smagorinsky deformation structure — the only truly-faithful piece):
  * **Deformation eddy viscosity** ``K_m = (C_s·l)²·|S|`` — the Smagorinsky
    length²·strain form (with the length/strain modernized; see below).
LILLY-EXPERIMENT-CONSISTENT (matches Lilly's specific 1962 equilibrium experiment, not
his undetermined general theory):
  * **Buoyancy stability factor + cutoff + Prandtl closure**: the ``√(1 − Ri/Pr_t)``
    stability factor, the ``Ri ≥ Pr_t`` shut-off, and the default ``Pr_t = 1`` (so
    ``K_h = K_m/Pr_t = K_m``) all match Lilly's equilibrium experiment — his ``√``
    stability factor with ``K_h/K_m = 1`` and ``K_m → 0`` for ``Ri > 1``.  The ``√``
    ramp is Lilly's OWN form (not a modern replacement); only the AD-safe numerics
    (below) are modern.  ``Pr_t`` is exposed as a tunable that generalizes Lilly's
    fixed unity.
MODERN additions (from other authors / not in either 1962-63 paper):
  * **Blackadar (1962) master mixing length** ``l = κz/(1 + κz/l_∞)`` — a separate
    author's length scale, not from Smagorinsky/Lilly.
DEPARTURES / RE-TUNED (vs the published papers):
  * **Modern deformation constant**: ``C_s = 0.2`` (Smagorinsky reported
    ``k_s ≈ 0.28``) and ``l_mix_max = 100 m`` are modern atmospheric single-column
    choices, NOT the 1962/1963 constants.
  * **1-D vertical-shear PROXY for the horizontal deformation**: ``|S| = √((∂u/∂z)² +
    (∂v/∂z)²)`` is the resolved VERTICAL shear.  Smagorinsky (1963) used the
    HORIZONTAL strain + a horizontal grid length; the full strain-rate tensor
    ``√(2·S_ij·S_ij)`` is the modern 3-D LES extension.  Vertical shear is not a
    surviving *component* of the horizontal deformation — it is a CHOSEN 1-D
    substitution for a single-column model (no horizontal strain is available).
  * **AD-safety numerics** (not Smagorinsky): the ``S² + 1e-10`` floor slightly
    perturbs ``K_m`` and ``Ri`` even where mixing is active (negligibly at resolved
    shear), and the double-``where`` at the Lilly-type cutoff yields a finite SELECTED
    reverse-mode cotangent at ``Ri = Pr_t``.  This is the ``differentiable: True``
    AD-safety sense — the forward is continuous but NON-C¹ at the cutoff (the on-side
    √ slope diverges); a bare ``√(max(·,0))`` would instead leak a 0·∞ NaN.
Behavior/assembly pins: ``tests/atmosphere/hydrostatic/unit/test_smagorinsky_faithful.py``.

Running-oracle comparison (tier-2 oracle audit, les_sgs_closures, 2026-10-19)
-------------------------------------------------------------------------------
The audit fed legoESM's own ``S²`` and ``N²`` to two running kernels (fp64):
microhh received the length ``C_s·l``, SAM received ``l`` and ``C_s`` separately.  The algebra matches; only the floors differ, and those
departures are KEPT here pending a user decision (NOT faithful to either kernel):
  * ``stability_form="lilly"`` vs microhh 2.0.2 (736a2f5e) ``diff_smag2.cxx``
    ``calc_evisc``: microhh adds ``dsmall = 1e-9`` to ``2 S_ij S_ij``
    (``include/diff_kernels.h:140``) where this module adds 1e-10, and clamps
    ``Ri/Pr_t`` at ``1 − 1e-9`` instead of cutting ``K_m`` to exactly 0 at
    ``Ri ≥ Pr_t``.  Measured: up to 2.0e-4 / 5.2e-4 m²/s (CBL, default /
    LES-tuned config), 7.7e-3 / 2.6e-2 m²/s at near-zero-shear levels (ratio
    √0.1), 7.3e-5 / 1.9e-4 m²/s in stable levels (ours 0, microhh its √1e-9
    residual).  With both microhh floors applied the result is bitwise.
  * ``stability_form="deardorff"``: see
    :func:`legoesm.atmosphere.physics._shared.deardorff_stable_eddy_viscosity`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import (
    buoyancy_coefficient,
    exner_function,
    deardorff_stable_eddy_viscosity,
    lilly_buoyancy_factor,
    mixing_length,
    virtual_temperature,
    half_to_iface_zero_ends,
)
from legoesm.atmosphere.physics.turbulence.config import SmagorinskyConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import diagnose_pbl_height
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    latent_enthalpy_correction,
    surface_moisture_flux,
    compute_surface_fluxes,
    surface_fluxes_at_lowest_level,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    diagnostic_heat_flux_full,
    implicit_vertical_diffusion,
    implicit_vertical_diffusion_theta,
)

from legoesm.core.smooth import positive_part_c1, smooth_max, smooth_min, smoothstep_cubic
from legoesm.thermo import saturation_vapor_pressure

# --- WRF v4.8.0 calculate_N2 / smag_km fixed literals (dyn_em/module_diffusion_em.F) ---
_WRF_N2_VAPOUR_COEF = 1.61   # calculate_N2 dry-branch vapour buoyancy coefficient
_WRF_QC_CR = 1.0e-5          # calculate_N2 cloud-water threshold [kg/kg]
_WRF_KM_FLOOR = 1.0e-6       # smag_km floor coefficient: K >= 1e-6 dz^2 [1/s]
# --- RULE 5 smoothing (VOTES D6, D11): relative widths ---
_WRF_SMOOTH_REL = 0.01       # floor / cap / moist-switch relative width


__physics_contract__ = {
    "summary": (
        "Default: WRF v4.8.0 km_opt=3 Smagorinsky vertical eddy coefficients "
        "(smag_km, mix_isotropic=0, calculate_N2). Lilly/Deardorff arms: "
        "Smagorinsky-Lilly strain-dependent eddy-viscosity turbulence: "
        "K_m = (C_s l)^2 |S| sqrt(max(0, 1 - Ri/Pr_t)), K_h = K_m/Pr_t, "
        "applied by implicit vertical diffusion with surface-flux boundary "
        "conditions."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "T_sfc": "K", "q_sfc": "kg/kg", "rho": "kg/m^3", "dt": "s", "q_c": "kg/kg", "q_i": "kg/kg",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m",
    },
    "sign_convention": (
        "Down-gradient mixing: du_dt ~ (1/rho) d/dz(rho Km du/dz); Km, Kh >= 0 "
        "and vanish where Ri >= Pr_t (Lilly-type stable cutoff). shflx, lhflx are "
        "positive UPWARD from the surface and are injected as the lower "
        "boundary condition (a source/sink), so the resolved column budget is "
        "NOT closed. z increases upward; level index -1 is the surface."
    ),
    "conserves": ["none"],
    # AD-safe (finite selected reverse-mode VJP via the double-where guard); NOT a
    # claim of a C1-smooth forward -- the Lilly-type cutoff is a non-C1 point at
    # Ri = Pr_t (the on-side sqrt ramp's slope diverges).
    "differentiable": True,
    "reference": (
        "WRF v4.8.0 (06d4240ae9) dyn_em/module_diffusion_em.F km_opt=3; "
        "Smagorinsky (1963), Mon. Wea. Rev. 91, 99-164; "
        "Lilly (1962), Tellus 14, 148-172"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_turbulence.py: a strongly "
        "stable column (Ri >= Pr_t) shuts mixing off (Km -> 0); with zero "
        "surface flux a neutral column gives ~zero tendencies."
    ),
}


def _wrf_eta_extrapolation(p_half_b):
    """WRF cf1, cf2, cf3 (module_initialize_scm_xy.F 333-349) from the column's
    eta = (p_half - p_top) / (p_s - p_top) at w-levels, bottom-up (VOTE D10)."""
    znw = (p_half_b - p_half_b[:, -1:]) / (p_half_b[:, :1] - p_half_b[:, -1:])
    dnw = znw[:, 1:] - znw[:, :-1]                       # dnw(k), 0-based k = 0..n-1
    dn1 = 0.5 * (dnw[:, 1] + dnw[:, 0])
    dn2 = 0.5 * (dnw[:, 2] + dnw[:, 1])
    fnp1 = .5 * dnw[:, 1] / dn1
    fnm1 = .5 * dnw[:, 0] / dn1
    cof1 = (2. * dn1 + dn2) / (dn1 + dn2) * dnw[:, 0] / dn1
    cof2 = dn1 / (dn1 + dn2) * dnw[:, 0] / dn2
    return fnp1 + cof1, fnm1 - cof1 - cof2, cof2


def wrf_km3_coefficients(u, v, T, q_v, q_c, q_i, p_full, p_half, z_half, dt,
                         config: SmagorinskyConfig, *, hard: bool = False):
    """WRF v4.8.0 ``km_opt = 3`` vertical eddy coefficients for a flat,
    horizontally homogeneous column (``w = 0``, map factors 1).

    Port of ``dyn_em/module_diffusion_em.F``: ``compute_diff_metrics`` (rdz,
    rdzw), the vertical-shear part of ``cal_deform_and_div`` (``defor13``,
    ``defor23`` at w-levels, zero at the surface and model top), ``calculate_N2``
    (dry and moist branches, special lowest level, top copies the level below)
    and ``smag_km`` with ``mix_isotropic = 0`` (``mlen_v = 1/rdzw``).  WRF's
    floating-point operation order is kept.

    Inputs are top-down (level -1 at the surface), ``q_*`` SPECIFIC [kg/kg];
    WRF's mixing ratios are formed as ``q_x / (1 - q_v - q_c - q_i)``.  Outputs
    are top-down: ``defor13``/``defor23`` (ncol, nlev+1), ``bn2``, ``xkmv``,
    ``xkhv`` (ncol, nlev).

    ``hard=True`` gives WRF's discontinuous form (oracle tests only); the default
    replaces each switch by the repo smooth helper: the moist-branch test by a
    smooth OR of two C1 ramps over ``[-1 %, 0]`` of ``r_v/r_vs - 1`` and
    ``r_c/qc_cr - 1`` (exact outside that band), the
    floor / cap by ``smooth_max`` / ``smooth_min`` at 1 % of the floor / cap,
    and ``max(0, def2 - BN2/pr)`` by ``positive_part_c1`` of width
    ``(1e-6/c_s^2)^2`` (below it the floor binds anyway).
    """
    g, R_d, R_v, cp, xlv = (constants.g, constants.R_d, constants.R_v,
                            constants.c_pd, constants.L_v)  # latent-ok: WRF km_opt=3 oracle constant
    flip = lambda a: a[:, ::-1]  # noqa: E731  top-down <-> WRF bottom-up
    qt_spec = q_v + q_c + q_i
    dry = 1.0 - qt_spec
    qv, qc, qi = flip(q_v / dry), flip(q_c / dry), flip(q_i / dry)
    t, p, ub, vb = flip(T), flip(p_full), flip(u), flip(v)
    zw, pw = flip(z_half), flip(p_half)
    if t.shape[1] < 3:
        raise ValueError("stability_form='wrf_km3' needs >= 3 levels (WRF's "
                         "surface extrapolation reads three).")
    pr, c_s, mub = config.wrf_prandtl, config.wrf_c_s, config.mix_upper_bound

    # compute_diff_metrics
    rdzw = 1.0 / (zw[:, 1:] - zw[:, :-1])                       # mass levels
    rdz_in = 2.0 / (zw[:, 2:] - zw[:, :-2])                     # w-levels 1..n-1

    # cal_deform_and_div, vertical-shear terms (w = 0, flat, mix_full_fields)
    zero = jnp.zeros_like(t[:, :1])
    d13 = jnp.concatenate([zero, (ub[:, 1:] - ub[:, :-1]) * (0.5 * (rdz_in + rdz_in)), zero], 1)
    d23 = jnp.concatenate([zero, (vb[:, 1:] - vb[:, :-1]) * (0.5 * (rdz_in + rdz_in)), zero], 1)

    # calculate_N2
    th = t / (p / constants.p_ref) ** (R_d / cp)               # phy_prep th_phy
    es = saturation_vapor_pressure(t)
    qvs = constants.epsilon * es / (p - es)
    tmp1 = qv + qc + qi
    cf1, cf2, cf3 = (c[:, None] for c in _wrf_eta_extrapolation(pw))
    tmp1sfc = cf1 * qv[:, :1]
    for k, sp in enumerate((qv, qc, qi)):     # WRF species loop, its add order
        if k:
            tmp1sfc = tmp1sfc + cf1 * sp[:, :1]
        tmp1sfc = tmp1sfc + cf2 * sp[:, 1:2] + cf3 * sp[:, 2:3]
    if hard:
        wet = ((qv >= qvs) | (qc >= _WRF_QC_CR)).astype(t.dtype)
    else:
        # C1 ramp on [-1 %, 0] (VOTE S2 b): exactly WRF's >= at and above the
        # threshold, exactly dry below 0.99 of it.
        s1 = smoothstep_cubic((qv / qvs - 1.0) / _WRF_SMOOTH_REL + 1.0)
        s2 = smoothstep_cubic((qc / _WRF_QC_CR - 1.0) / _WRF_SMOOTH_REL + 1.0)
        wet = 1.0 - (1.0 - s1) * (1.0 - s2)
    xlvqv = xlv * qv
    coefa = (1.0 + xlvqv / R_d / t) / (1.0 + xlv * xlvqv / cp / R_v / t / t) / th
    the = th * (1.0 + xlv * qvs / cp / t)
    # interior mass levels 1..n-2
    tmpdz = 1.0 / rdz_in[:, :-1] + 1.0 / rdz_in[:, 1:]
    dqt = (tmp1[:, 2:] - tmp1[:, :-2]) / tmpdz
    bn2_m = g * (coefa[:, 1:-1] * (the[:, 2:] - the[:, :-2]) / tmpdz - dqt)
    bn2_d = g * ((th[:, 2:] - th[:, :-2]) / th[:, 1:-1] / tmpdz
                 + _WRF_N2_VAPOUR_COEF * (qv[:, 2:] - qv[:, :-2]) / tmpdz - dqt)
    bn2_in = wet[:, 1:-1] * bn2_m + (1.0 - wet[:, 1:-1]) * bn2_d
    # lowest mass level
    # phy_prep: surface T and p extrapolated linearly in z from the two lowest
    # mass levels (z at mass points = mid-layer of the w-level heights)
    z1 = 0.5 * (zw[:, :1] + zw[:, 1:2])
    z2 = 0.5 * (zw[:, 1:2] + zw[:, 2:3])
    w1 = (zw[:, :1] - z2) / (z1 - z2)
    w2 = 1. - w1
    p8w = w1 * p[:, :1] + w2 * p[:, 1:2]
    t8w = w1 * t[:, :1] + w2 * t[:, 1:2]
    thetasfc = t8w / (p8w / constants.p_ref) ** (R_d / cp)
    qvsfc_m = cf1 * qvs[:, :1] + cf2 * qvs[:, 1:2] + cf3 * qvs[:, 2:3]
    tmpdz_m = 1.0 / rdz_in[:, :1] + 0.5 / rdzw[:, :1]
    thetaesfc = thetasfc * (1.0 + xlv * qvsfc_m / cp / t8w)
    b0_m = g * (coefa[:, :1] * (the[:, 1:2] - thetaesfc) / tmpdz_m
                - (tmp1[:, 1:2] - tmp1sfc) / tmpdz_m)
    qvsfc_d = cf1 * qv[:, :1] + cf2 * qv[:, 1:2] + cf3 * qv[:, 2:3]
    tmpdz_d = 1. / rdzw[:, :1]
    b0_d = g * ((th[:, 1:2] - th[:, :1]) / th[:, :1] / tmpdz_d
                + _WRF_N2_VAPOUR_COEF * (qv[:, 1:2] - qvsfc_d) / tmpdz_d
                - (tmp1[:, 1:2] - tmp1sfc) / tmpdz_d)
    b0 = wet[:, :1] * b0_m + (1.0 - wet[:, :1]) * b0_d
    bn2 = jnp.concatenate([b0, bn2_in, bn2_in[:, -1:]], 1)

    # smag_km, mix_isotropic = 0
    t13 = 0.25 * (d13[:, 1:] + d13[:, :-1] + d13[:, 1:] + d13[:, :-1])
    t23 = 0.25 * (d23[:, 1:] + d23[:, :-1] + d23[:, 1:] + d23[:, :-1])
    def2 = t13 * t13 + t23 * t23
    mlen = 1. / rdzw
    arg = def2 - bn2 / pr
    floor = _WRF_KM_FLOOR * mlen * mlen
    cap = mub * mlen * mlen / dt
    if hard:
        tmp = jnp.maximum(0., arg) ** 0.5
        xkmv = jnp.minimum(jnp.maximum(c_s * c_s * mlen * mlen * tmp, floor), cap)
        xkhv = jnp.minimum(xkmv / pr, cap)
    else:
        w0 = (_WRF_KM_FLOOR / (c_s * c_s)) ** 2
        pos = positive_part_c1(arg / w0, 1.0) * w0
        ok = pos > 0.0
        tmp = jnp.where(ok, jnp.sqrt(jnp.where(ok, pos, 1.0)), 0.0)
        xkmv = c_s * c_s * mlen * mlen * tmp
        xkmv = smooth_max(xkmv / floor, 1.0, 1.0 / _WRF_SMOOTH_REL) * floor
        xkmv = smooth_min(xkmv / cap, 1.0, 1.0 / _WRF_SMOOTH_REL) * cap
        xkhv = smooth_min(xkmv / pr / cap, 1.0, 1.0 / _WRF_SMOOTH_REL) * cap
    return {"defor13": flip(d13), "defor23": flip(d23), "bn2": flip(bn2),
            "xkmv": flip(xkmv), "xkhv": flip(xkhv)}


def smagorinsky_turbulence(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: SmagorinskyConfig,
    q_c: jax.Array | None = None,
    q_i: jax.Array | None = None,
) -> TurbulenceOutput:
    """Compute turbulence tendencies from the deformation/stability-dependent eddy diffusivity.

    Parameters
    ----------
    u, v : jax.Array
        Wind components at full levels [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    z_full : jax.Array
        Height at full levels [m], shape (ncol, nlev).
    z_half : jax.Array
        Height at half levels [m], shape (ncol, nlev+1).
    T_sfc : jax.Array
        Surface temperature [K], shape (ncol,).
    q_sfc : jax.Array
        Surface saturation mixing ratio [kg/kg], shape (ncol,).
    rho : jax.Array
        Air density at full levels [kg/m^3], shape (ncol, nlev).
    dt : float
        Time step [s].
    config : SmagorinskyConfig
    q_c, q_i : jax.Array, optional
        Cloud liquid / ice specific contents [kg/kg], (ncol, nlev); read only by
        the ``"wrf_km3"`` arm (WRF's ``qv + qc + qi`` buoyancy term and its
        ``qc >= 1e-5`` moist test).  ``None`` = zero (declared departure where a
        lane does not pass them).

    Returns
    -------
    TurbulenceOutput
    """
    ncol, nlev = T.shape

    # --- Strain-dependent Smagorinsky–Lilly eddy viscosity ---------------
    # Heights and shear at the (nlev-1) interior half-level interfaces.
    z_half_inner = 0.5 * (z_full[:, :-1] + z_full[:, 1:])  # (ncol, nlev-1)
    l_mix = mixing_length(z_half_inner, config.l_mix_max)   # Blackadar (1962)

    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])       # (ncol, nlev-1)
    dz_half = jnp.clip(dz_half, 1.0, None)

    # Resolved deformation: the vertical shear of the horizontal wind is
    # CHOSEN as a 1-D proxy for Smagorinsky's horizontal deformation (a
    # single column carries no horizontal strain).  Floor S2 so both
    # ``S = √S2`` and ``Ri = N²/S2`` stay finite and differentiable.
    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2 = du_dz ** 2 + dv_dz ** 2 + 1e-10
    S = jnp.sqrt(S2)

    # Gradient Richardson number at the interfaces (virtual θ buoyancy),
    # via the canonical inverse-Exner + shared buoyancy-coefficient helpers
    # (clips kept at the call sites).
    exner_pref = 1.0 / exner_function(p_full)
    theta_v = virtual_temperature(T, q_v) * exner_pref
    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    N2 = buoyancy_coefficient(jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz
    Ri = N2 / S2

    # Lilly (1962) buoyancy stability factor √(max(0, 1 − Ri/Pr_t)) — the √
    # form is Lilly's equilibrium result (K_h/K_m = 1, K_m→0 for Ri>1); the
    # AD-safe double-``where`` cutoff (finite cotangent at Ri = Pr_t) lives in the
    # shared helper (with the S²+1e-10 floor above keeping Ri finite). Enhances
    # mixing when unstable (Ri<0), shuts it off at Ri ≥ Pr_t.
    # Stable-stratification treatment.  Dispatch on the STATIC config value at
    # function entry (never a silent ``else``): an unknown value must not run
    # different physics under a typo.
    if config.stability_form == "wrf_km3":
        # WRF km_opt=3 K at mass levels; vertical_diffusion_2 hands the solver
        # the w-level average 0.5 (K(k) + K(k-1)) (VOTE D5).
        zeros = jnp.zeros_like(q_v)
        co = wrf_km3_coefficients(
            u, v, T, q_v, zeros if q_c is None else q_c, zeros if q_i is None else q_i,
            p_full, p_half, z_half, dt, config)
        Km_half = 0.5 * (co["xkmv"][:, :-1] + co["xkmv"][:, 1:])
        Kh_half = 0.5 * (co["xkhv"][:, :-1] + co["xkhv"][:, 1:])
    elif config.stability_form == "lilly":
        f_buoy = lilly_buoyancy_factor(Ri, config.Pr_t)
        # K_m = (C_s · l)^2 · |S| · f_buoy ;  K_h = K_m / Pr_t.
        Km_half = (config.C_s * l_mix) ** 2 * S * f_buoy    # (ncol, nlev-1)
    elif config.stability_form == "deardorff":
        # Deardorff (1980) stable-length limit — stratification shrinks the
        # mixing length instead of driving a stability factor to zero.  Takes
        # the RAW S² (the N²/Pr_t subtraction happens inside the helper);
        # passing the already-corrected strain would apply it twice.
        Km_half = deardorff_stable_eddy_viscosity(
            S2, N2, l_mix, config.C_s, config.Pr_t)
    else:
        raise ValueError(
            f"Unknown SmagorinskyConfig.stability_form "
            f"{config.stability_form!r}; expected 'wrf_km3', 'lilly' or 'deardorff'."
        )
    if config.stability_form != "wrf_km3":
        Kh_half = Km_half / config.Pr_t

    # Interpolate to full levels for diagnostics (single concat; same
    # pattern as Louis/TKE/Holtslag-Boville/YSU).
    Km_interior = 0.5 * (Km_half[:, :-1] + Km_half[:, 1:])
    Km_full = jnp.concatenate(
        [Km_half[:, :1], Km_interior, Km_half[:, -1:]], axis=1,
    )
    Kh_interior = 0.5 * (Kh_half[:, :-1] + Kh_half[:, 1:])
    Kh_full = jnp.concatenate(
        [Kh_half[:, :1], Kh_interior, Kh_half[:, -1:]], axis=1,
    )

    # Aliases for the implicit-diffusion solve below.
    K_half_m = Km_half
    K_half_h = Kh_half

    # Layer thicknesses
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])  # (ncol, nlev)
    dz = jnp.clip(dz, 1.0, None)

    # Surface fluxes
    tau_x, tau_y, shflx, lhflx, ustar = surface_fluxes_at_lowest_level(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface, z_full[:, -1] - z_half[:, -1])

    # Convert surface fluxes to boundary conditions for diffusion
    # Momentum: flux = tau / rho ~ Cd * |V| * u  (already has rho in it)
    # For the Thomas solver, surface_flux units = [phi_units * kg/m^2/s]
    # tau_x = -rho * Cd * |V| * u  [Pa = kg/(m*s^2)]
    # For u diffusion: surface_flux = tau_x (positive = upward flux of u)
    sflx_u = tau_x   # [Pa]
    sflx_v = tau_y
    # Heat: shflx = rho * c_pd * Ch * |V| * (T_sfc - T) [W/m^2]
    # For T diffusion: surface_flux = shflx / c_pd [kg/(m^2*s) * K]
    sflx_q = surface_moisture_flux(config.surface, lhflx, T_sfc)
    # Heat BC carries the latent enthalpy correction (water at L(T) vs L_v).
    sflx_T = (shflx + latent_enthalpy_correction(lhflx, sflx_q)) / constants.c_pd

    # Apply implicit vertical diffusion.  Heat is mixed in θ-space so a
    # dry adiabat stays neutral; momentum and moisture are conserved on
    # adiabatic motion and use raw T-style diffusion.
    u_new = implicit_vertical_diffusion(u, K_half_m, rho, dz, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, K_half_m, rho, dz, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion_theta(
        T, K_half_h, rho, dz, dz_half, p_full, dt, sflx_T,
    )
    q_new = implicit_vertical_diffusion(q_v, K_half_h, rho, dz, dz_half, dt, sflx_q)

    # Tendencies
    du_dt = (u_new - u) / dt
    dv_dt = (v_new - v) / dt
    dT_dt = (T_new - T) / dt
    dq_v_dt = (q_new - q_v) / dt

    h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)

    # Q1 diagnostic (LES-suite): local down-gradient heat flux F = -Kh·∂θ/∂z
    # (γ=0), θ = T·exner_pref [=T/Π].  Pure diagnostic — does NOT feed tendencies.
    wtheta_flux = diagnostic_heat_flux_full(T * exner_pref, dz_half, Kh_half)

    return TurbulenceOutput(
        du_dt=du_dt,
        dv_dt=dv_dt,
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        Km=Km_full,
        Kh=Kh_full,
        shflx=shflx,
        lhflx=lhflx, evap_sfc=sflx_q,
        ustar=ustar,
        h_pbl=h_pbl,
        wtheta_flux=wtheta_flux,
        Kh_iface=half_to_iface_zero_ends(Kh_half),
    )
