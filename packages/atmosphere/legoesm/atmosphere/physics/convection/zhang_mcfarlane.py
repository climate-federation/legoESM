"""Zhang & McFarlane (1995) deep-convection scheme.

A single-plume mass-flux scheme with a quasi-equilibrium CAPE-relaxation
closure: the cloud-base mass flux ``M_b`` is diagnosed from the column
CAPE excess over a threshold and relaxed (implicit Euler) toward the
diagnosed equilibrium value over a tunable timescale.  The plume itself
is integrated using the shared
:func:`legoesm.atmosphere.physics.convection._plume.entraining_detraining_plume`
helper; the environmental tendencies (compensating subsidence +
detrainment) are computed by the existing
:func:`legoesm.atmosphere.physics.convection.mass_flux.apply_mass_flux_kernel`
so this scheme reuses every piece of column physics rather than
re-implementing it.

The scheme is **smooth-everywhere** in the AD sense:

* ``CAPE > threshold`` is replaced by ``cape_trigger`` (sigmoid).
* ``(CAPE - threshold)+`` uses ``smooth_positive_part``.
* Cloud-base / LFC / LNB localization uses the smooth-fractional
  level diagnostics from ``_plume``.
* The plume integrator's mass-flux profile is gated by a sigmoid on
  buoyancy, not a hard cut.

Convective momentum transport (CMT) is enabled by default via the
Gregory et al. 1997 closure
(:func:`legoesm.atmosphere.physics.convection._plume.cmt_gregory_1997`).

Faithfulness to Zhang-McFarlane (1995) / E3SM ``zm_conv.F90`` (oracle)
---------------------------------------------------------------------
FAITHFUL (ports of / matched to the E3SM/CAM ``zm_conv.F90`` algorithm):
  * The DILUTE-parcel CAPE (:func:`._zm_dilute.dilute_parcel_cape`) is a
    faithful port of the oracle ``parcel_dilute``/``buoyan_dilute`` (Raymond &
    Blyth 1992 entropy-conserving entraining plume): max-MSE PBL launch,
    fractional-entrainment ascent, entropy inversion, condensate loading +
    freezing. Its quantitative agreement with the compiled E3SM/CAM Fortran was
    checked offline against a local, untracked oracle harness — no committed test
    certifies a specific percentage (see the note in ``test_zm_dilute_parcel``).
    What IS CI-pinned is the qualitative dilute-parcel behavior (dilute CAPE
    strictly < undilute, < 0.6× on a tropical sounding) in
    ``tests/unit/test_zm_dilute_parcel.py``.
  * The dilute-parcel entrainment constants are the E3SM/CAM defaults:
    ``dmpdz=-1e-3`` 1/m and ``tiedke_add=0.5`` K. (The ``cape_threshold=70`` J/kg
    ZM95 value is a TRIGGER/closure setting, not a dilute-parcel constant.)
DEPARTURES / SURROGATES (documented; NOT the ZM95 closed forms):
  * The cloud-base mass-flux closure is a GENERIC first-order CAPE-relaxation
    SURROGATE. With ``Δ = CAPE - cape_threshold`` and sharpness ``s``, the
    equilibrium flux is
    ``M_b_eq = sigmoid(s·Δ) · rho_BL · softplus(s·Δ)/(s · g · tau_cape)``
    (a smooth trigger ``sigmoid(s·Δ)`` times the ``(Δ)+`` softplus positive part),
    implicit-Euler relaxed toward ``M_b_eq`` and clipped to ``[0, M_b_max]``. This
    is NOT the ZM95 cloud-work-function / quasi-equilibrium closure (which sets the
    CAPE-consumption rate from a work-function sensitivity), and NOT a Kain-2004
    iterated M_b. The ``g/rho_BL`` factor is a dimensional stand-in for the
    CAPE-consumption sensitivity. See the inline note at the closure.
  * The plume (single bulk entraining/detraining plume, constant ``epsilon_0``/
    ``delta_0``) and the environmental subsidence+detrainment use the SHARED
    ``_plume``/``mass_flux`` kernels (advective solve, conservative only to
    truncation order on the default path), not a ZM-specific microphysics/
    downdraft package. CMT is Gregory et al. (1997), not the ZM95 momentum term.
Scheme-level use of the faithful dilute CAPE + the surrogate closure form are
pinned in ``tests/atmosphere/hydrostatic/unit/test_zhang_mcfarlane_faithful.py``.

References
----------
- Zhang, G. J., & McFarlane, N. A. (1995). Sensitivity of climate
  simulations to the parameterization of cumulus convection in the
  Canadian Climate Centre general circulation model.  *Atmos.-Ocean*,
  33(3), 407–446.
- Gregory, D., Kershaw, R., & Inness, P. M. (1997). Parametrization of
  momentum transport by convection. II.  *Quart. J. Roy. Meteor. Soc.*,
  123, 1153–1183.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import compute_rho
from legoesm.atmosphere.physics.thermodynamics import (
    parcel_profile_and_cape,
)

from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig
from legoesm.atmosphere.physics.convection.output import (
    ConvectionOutput,
    split_convective_rain,
)
from legoesm.atmosphere.physics.convection.mass_flux import (
    apply_mass_flux_kernel,
    compute_column_geometry,
)
from legoesm.atmosphere.physics.convection._triggers import (
    cape_trigger,
    smooth_positive_part,
)
from legoesm.atmosphere.physics.convection._plume import (
    cmt_gregory_1997,
    compute_lcl,
    entraining_detraining_plume,
)
from legoesm.atmosphere.physics.convection._zm_dilute import (
    dilute_parcel_cape,
)


__all__ = ("zhang_mcfarlane_convection",)


__physics_contract__ = {
    "summary": (
        "Zhang-McFarlane (1995) deep convection: a single entraining-detraining "
        "plume with a dilute-CAPE quasi-equilibrium closure (M_b relaxed toward "
        "the CAPE-consumption equilibrium) and optional Gregory-97 CMT. Uses "
        "the shared subsidence+detrainment kernel; condensate handed to "
        "microphysics. Smooth (differentiable)."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa",
        "u": "m/s", "v": "m/s",
        "conv_prog_profile": "kg/m^2/s (cloud-base mass-flux carry M_b at [:, -1])",
        "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "dq_c_conv_dt": "kg/kg/s (detrained cloud-water source to microphysics, >=0)",
        "cape": "J/kg", "convective_mask": "1 (0-1 CAPE trigger)",
        "du_dt_conv": "m/s^2 (CMT; None if disabled)",
        "dv_dt_conv": "m/s^2 (CMT; None if disabled)",
        "conv_prog_profile_new": "kg/m^2/s (relaxed M_b at [:, -1])",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Where dilute CAPE>threshold the plume warms "
        "aloft and dries the lower column via compensating subsidence + "
        "detrainment; dq_c_conv_dt >= 0 is a cloud-water SOURCE to microphysics "
        "(precip deferred). The kernel conserves column moist static energy and "
        "total water (advective default: truncation order). Optional Gregory-97 "
        "CMT redistributes momentum vertically (transport-dominant, not exactly "
        "conserving, so momentum is not claimed)."
    ),
    # The DEFAULT public path uses the shared kernel's advective subsidence
    # solve, conservative only to TRUNCATION ORDER (exact only in the opt-in
    # implicit_flux path), so no contract-level conservation is guaranteed; the
    # column budget is closed downstream.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Zhang & McFarlane (1995), Atmos.-Ocean 33, 407-446; "
        "Gregory et al. (1997), Q. J. R. Meteorol. Soc. 123, 1153-1183"
    ),
    "idealized_test": (
        "tests/unit/test_zhang_mcfarlane.py; CAPE<=threshold -> zero mass flux "
        "and zero tendency; a conditionally-unstable tropical column -> heating "
        "aloft + low-level drying with a positive dq_c source and M_b relaxing "
        "toward the CAPE-closure equilibrium; column MSE and total water "
        "conserved by the shared kernel."
    ),
}


def zhang_mcfarlane_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    u: jax.Array,
    v: jax.Array,
    conv_prog_profile: jax.Array,
    dt: float,
    config: ZhangMcFarlaneConfig = ZhangMcFarlaneConfig(),
) -> tuple[ConvectionOutput, jax.Array]:
    """Zhang-McFarlane deep convection (smooth, differentiable).

    Parameters
    ----------
    T : jax.Array, shape (ncol, nlev)
        Environmental temperature [K].  Surface at ``[:, -1]``.
    q_v : jax.Array, shape (ncol, nlev)
        Water-vapor specific humidity [kg/kg].
    p_full, p_half : jax.Array
        Full / half-level pressures [Pa].  Shapes ``(ncol, nlev)`` and
        ``(ncol, nlev+1)`` respectively.
    u, v : jax.Array, shape (ncol, nlev)
        Environmental wind components [m/s].  Used by the Gregory et
        al. 1997 CMT closure when ``config.enable_cmt = True``.
    conv_prog_profile : jax.Array, shape (ncol, nlev)
        Convection prognostic carry (PR-0 schema).  ZM uses only the
        surface-adjacent slot ``[:, -1]`` to remember the previous
        cloud-base mass flux ``M_b`` for implicit-Euler relaxation
        toward the diagnosed equilibrium value; aloft slots are
        unused (zeros in / zeros out).
    dt : float
        Time step [s].
    config : ZhangMcFarlaneConfig
        Scheme tunables — see :class:`ZhangMcFarlaneConfig`.

    Returns
    -------
    out : ConvectionOutput
        Tendencies on environment T, q_v, q_c plus the CAPE diagnostic
        and (when CMT is enabled) du/dt, dv/dt.
    conv_prog_profile_new : jax.Array, shape (ncol, nlev)
        Updated carry with the relaxed ``M_b_new`` packed at
        ``[:, -1]``; aloft entries are zero.
    """
    ncol, nlev = T.shape

    # -- Column geometry, moist adiabat, CAPE --------------------------------
    # Use virtual-T moist hydrostatic geometry (clean_physics iter-2 #2).
    dz, rho, z = compute_column_geometry(T, p_full, p_half, q_v=q_v)
    T_base = T[:, -1]
    q_base = q_v[:, -1]
    p_base = p_full[:, -1]

    # ZM uses the CAPE of a DILUTE entraining plume (Raymond-Blyth 1992;
    # ``buoyan_dilute``/``parcel_dilute`` in zm_conv.F90), NOT an undilute
    # moist adiabat.  The launch parcel ascends entraining environmental
    # air at fractional rate ``dmpdz`` [1/m]; CAPE is the buoyancy integral
    # of that DILUTE parcel.  Entraining dry air reduces buoyancy and CAPE
    # by a factor ~3 in a tropical sounding — the single most important ZM
    # fidelity property (without it ZM over-fires in marginal columns).
    # The dilute CAPE's quantitative agreement with the compiled E3SM/CAM
    # Fortran oracle was checked offline against a local, untracked harness;
    # CI pins only the qualitative bounds (see test_zm_dilute_parcel).
    if config.use_dilute_cape:
        dparcel = dilute_parcel_cape(
            T, q_v, p_full, p_half, z,
            dmpdz=config.dmpdz,
            tiedke_add=config.tiedke_add,
            tp_fac=config.tp_fac,
            tpert=config.parcel_tpert,
            pbl_top_pa=config.pbl_top_pa,
        )
        cape = dparcel.cape
        # The dilute parcel temperature is the physically-correct cloud
        # model temperature; keep it for diagnostics / future closure work.
        T_moist = dparcel.T_parcel
    else:
        # Legacy undilute moist-adiabat CAPE (use_dilute_cape=False).
        T_moist, cape = parcel_profile_and_cape(T, p_full, p_half, q_v=q_v)

    # -- Smooth CAPE trigger and cloud-base mass-flux closure ---------------
    cape_weight = cape_trigger(
        cape, config.cape_threshold, config.cape_sharpness,
    )
    # Generic first-order CAPE-relaxation SURROGATE (NOT a published
    # closure).  This is *not* the Zhang-McFarlane (1995) closure, which
    # consumes CAPE at a rate set by a cloud-work-function / quasi-equilibrium
    # sensitivity, and it is *not* a Kain (2004) formula (Kain 2004 has no
    # closed-form M_b — it iterates M_b to remove CAPE over TIMEC).  Here the
    # ``g / rho_BL`` factor is a dimensional stand-in for that CAPE-consumption
    # sensitivity, giving a kg/m^2/s mass flux:
    #     M_b = rho_BL * (CAPE - threshold)+ / (g * tau)   [kg/m^2/s]
    # The earlier formula ``(CAPE - threshold)+ / tau`` had units
    # ``m^2/s^3`` — wrong by a factor of ``rho_BL/g``.  At sea level
    # this made M_b ~8x larger than the dimensionally-correct value;
    # the runaway was masked operationally only by the ``M_b_max`` cap,
    # but the gradient w.r.t. CAPE was off by the same factor and ``M_b``
    # did not scale with the surface air density at all.
    # Dry boundary-layer density via the shared ideal-gas helper (same
    # 1 K temperature clip as the previous inline form).
    rho_BL = compute_rho(T[:, -1], p_full[:, -1])
    M_b_eq = (
        cape_weight
        * rho_BL
        * smooth_positive_part(
            cape - config.cape_threshold, config.cape_sharpness,
        )
        / (constants.g * config.tau_cape)
    )
    # Implicit-Euler relaxation toward equilibrium — stable for any
    # ratio ``r = dt / max(tau_cape, 1e-30)`` (the max() floors tau away
    # from 0; see the final sentence):
    #     M_b_new = (M_b_old + r * M_b_eq) / (1 + r).
    # For ``dt >> tau`` this approaches ``M_b_eq`` (full
    # equilibration); for ``dt << tau`` it approaches a small
    # fractional adjustment ``r * (M_b_eq - M_b_old)``.  An
    # earlier form ``dt / max(tau, dt)`` clamped the ratio to ≤ 1 —
    # under-stepping by up to ``r/(r+1) - 1/2 ≈ 41%`` at ``r=10`` —
    # which is *not* what the comment claims (audit Codex finding:
    # "the documented implicit-Euler factor is not what is
    # implemented").  Protect against ``tau == 0`` only.
    M_b_old = conv_prog_profile[:, -1]
    dt_over_tau = dt / jnp.maximum(config.tau_cape, 1e-30)
    M_b = (M_b_old + dt_over_tau * M_b_eq) / (1.0 + dt_over_tau)
    # Bound M_b to a fraction of the literature peak tropical value
    # (config.M_b_max, default 0.05 kg/m²/s ~ half the ~0.1 peak).
    # Without this cap a column with very large
    # CAPE drives M_b unboundedly and emits column heating that breaks
    # the next dynamics step on the lat-lon FV pole-cell CFL.
    M_b = jnp.clip(M_b, 0.0, config.M_b_max)

    # -- Plume launch / cloud-base index ------------------------------------
    # Surface parcel perturbed slightly per Zhang & McFarlane 1995 §3a;
    # this avoids zero-perturbation degeneracies and gives a smooth
    # cloud-base diagnosis.
    T_parcel = T_base + config.parcel_dT
    q_parcel = q_base + config.parcel_dq
    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
    k_base_smooth = lcl.k_lcl_smooth

    # Constant entrainment / detrainment profiles in the surface-last
    # convention (level index nlev-1 = surface, 0 = top).  Tunable per
    # scheme but identical across levels in the standard Zhang-McFarlane
    # bulk plume.
    eps_profile = jnp.full_like(T, config.epsilon_0)
    dlt_profile = jnp.full_like(T, config.delta_0)

    plume = entraining_detraining_plume(
        T, q_v, p_full, p_half, z,
        T_parcel, q_parcel, k_base_smooth,
        eps_profile, dlt_profile, M_b,
        buoyancy_death_memory=config.buoyancy_death_memory,
    )

    # Cap plume.M_u once at the source so every downstream use (kernel
    # tendencies, CMT, q_c sources) sees the same bounded value.  The
    # kernel's internal cap is now redundant but kept for safety.
    plume_M_u_capped = jnp.clip(plume.M_u, 0.0, config.M_b_max)
    plume = plume._replace(M_u=plume_M_u_capped)

    # -- Environmental tendencies via the shared mass-flux kernel ----------
    # Plume splits vapor (``plume.q_u`` — saturation-clipped per level)
    # and cloud water (``plume.q_c_u`` — accumulated condensation)
    # explicitly, so the kernel's ``q_c_conv_dt`` is now the correct
    # detrainment of plume cloud water and we use it directly.
    dT_dt, dq_v_dt, dq_c_conv_dt = apply_mass_flux_kernel(
        T, q_v, p_full,
        plume.T_u, plume.q_u, plume.q_c_u, plume.M_u,
        z, rho, config.delta_0, M_u_max=config.M_b_max,
    )

    # -- Convective momentum transport --------------------------------------
    if config.enable_cmt:
        du_dt_conv, dv_dt_conv = cmt_gregory_1997(
            u, v, plume.M_u, None,
            p_full, p_half, rho,
            c_u=config.cmt_c_u, c_d=config.cmt_c_d,
        )
    else:
        du_dt_conv = None
        dv_dt_conv = None

    # -- Convective mask (column-mean diagnostic) ---------------------------
    convective_mask = cape_weight  # already a smooth (ncol,) indicator

    # In-updraft precipitation: shared rain-split (same knob + mass proof as
    # Tiedtke/Bechtold). precip_efficiency=0 (default) => no split, byte-identical.
    dq_c_conv_dt, dq_r_conv_dt = split_convective_rain(
        dq_c_conv_dt, config.precip_efficiency)

    out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=convective_mask,
        du_dt_conv=du_dt_conv,
        dv_dt_conv=dv_dt_conv,
        dq_r_conv_dt=dq_r_conv_dt,
    )

    # Pack the relaxed M_b back into the surface-adjacent carry slot
    # for downstream visibility (training diagnostics, conservation
    # checks).  Aloft slots are zero-filled — the orchestrator schema
    # is uniform across all schemes.
    conv_prog_profile_new = jnp.zeros_like(conv_prog_profile).at[:, -1].set(M_b)

    return out, conv_prog_profile_new
