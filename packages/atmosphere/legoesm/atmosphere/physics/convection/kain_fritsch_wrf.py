"""Kain-Fritsch (KF-Eta) deep and shallow convection: WRF v4.8.0 port (OPT-IN).

NOT RULE 5 COMPLIANT (user decision 2026-10-30, PR #2249; precedent PR #2229 /
issue #2231).  Every traced-state switch of the oracle (trigger, candidate
search, deep/shallow choice, level selections, downdraft and closure exits,
piecewise formulas, clips) is HARD at the default, so the gradient ignores
them and the output jumps when one flips.  Selected only by
``KainFritschConfig.variant = "wrf_kfeta"`` (``ExperimentConfig.
kain_fritsch_variant``); the default ``"smooth"`` runs the smooth surrogate in
``kain_fritsch.py``.  Smoothing every switch is required before any
production lane uses this variant: tracked in issue #2280 (switch inventory).

``kain_fritsch_wrf_convection`` runs the faithful JAX port of WRF v4.8.0
(06d4240ae9) ``phys/module_cu_kfeta.F`` ``KF_eta_PARA`` (in ``_kf_eta.py``,
oracle-matched to the compiled Fortran at round-off, see
``tests/atmosphere/test_kain_fritsch_oracle.py``) on legoESM columns.  The
scheme is the oracle's, hard switches included: the updraft-source-layer
search over the lowest 300 hPa, the Fritsch-Chappell trigger (``kf_trigger``
3, or 1 with ``enable_rh_trigger_perturb=False``), the entraining/detraining
updraft with CONDLOAD fallout and glaciation, the RH-controlled downdraft,
the iterated ABE closure (deep) or the TKE closure (shallow), and the upstream
compensating subsidence over the convective time scale ``TIMEC``.

What this wrapper adds, and nothing else (each is a recorded seam choice):

* levels flipped (legoESM top-down -> KF bottom-up) and back;
* ``QV0`` = the model's specific humidity ``q_v``, unchanged;
* ``DZQ`` = the hypsometric layer depth of :func:`compute_column_geometry`
  and ``rhoe = dp/(g*DZQ)``, so the oracle's ``DP = rhoe*g*DZQ`` is the
  model's layer pressure thickness (KF column mass = model column mass);
* ``W0AVG1D`` = the instantaneous ``w_grid`` (no running mean);
* ``DX`` = 25 km x ``dtlcl_dx_scale`` (the existing field's meaning:
  ``WKL = w*DX/25 km``);
* constants: legoESM ``g, c_pd, R_d, L_v`` with ``XLV1 = 0`` (constant
  latent heat) and the ``legoesm.thermo`` Tetens curve
  (``SVP1..SVPT0``);
* ``warm_rain = True`` branch of the hydrometeor feedback (frozen
  condensate melted into ``dq_c``, legoESM has no convective ice output);
* the oracle tendencies are applied as instantaneous rates every call (the
  WRF ``NCA`` hold over ``TIMEC`` is not carried);
* convective precipitation ``PRATEC`` leaves through ``dq_r_conv_dt``,
  distributed over the levels in proportion to the oracle's own net
  precipitation production ``(PPTLIQ+PPTICE)*AINC*(1-FBFRC)*FRC2`` (its rain
  feedback profile); the column integral equals ``PRATEC``; the oracle's
  resolved-rain tendency ``DQRDT`` (shallow fallback with ``FBFRC=1``) is
  added to it;
* ``cape`` = the oracle's buoyant energy ``ABE`` where convection fires,
  ``convective_mask`` = 1 where it fires, and the carry packs the
  cloud-base mass flux ``UMF(LCL)/DX**2`` [kg m-2 s-1] at ``[:, -1]``.

KainFritschConfig fields that are the oracle's own numbers stay live
(``usl_depth_pa`` = DPMIN, ``dtlcl_coeff``, ``dtlcl_exponent``,
``wklcl_ref``, ``wklcl_zref``, ``rh_trigger_u00/rhmax/slope``,
``entrain_const``, ``rad_min_m``, ``rad_max_m``, ``rad_wkl_ref``,
``timec_min_s``, ``timec_max_s``, ``dtlcl_dx_scale``,
``enable_rh_trigger_perturb``); their defaults are the oracle values.  The
fields of the former surrogate that have no counterpart in KF-Eta are
accepted only at their defaults (a non-default or traced value raises).

The scheme needs the grid winds ``u``, ``v`` (TIMEC = DX/VCONV and the
shear-dependent precipitation efficiency); calling without them raises.

References
----------
* Kain, J. S. (2004). The Kain-Fritsch convective parameterization: An update.
  *J. Appl. Meteor.*, 43, 170-181.
* Kain, J. S. & Fritsch, J. M. (1990). *J. Atmos. Sci.*, 47, 2784-2802.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.convection._kf_eta import KFConstants, kf_eta_para
from legoesm.atmosphere.physics.convection.config import KainFritschConfig
from legoesm.atmosphere.physics.convection.mass_flux import compute_column_geometry
from legoesm.atmosphere.physics.convection.output import ConvectionOutput

from legoesm import constants, thermo

__all__ = ("kain_fritsch_wrf_convection", "check_wrf_kfeta_config")

__physics_contract__ = {
    "summary": (
        "Kain-Fritsch (WRF v4.8.0 KF-Eta) deep/shallow mass-flux convection: "
        "faithful port of KF_eta_PARA applied to legoESM columns. OPT-IN "
        "(KainFritschConfig.variant='wrf_kfeta'); NOT RULE 5 compliant: "
        "every traced-state switch is hard at the default."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa",
        "w_grid": "m/s (resolved grid-scale vertical velocity)",
        "u": "m/s", "v": "m/s", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "dq_c_conv_dt": "kg/kg/s (detrained cloud condensate)",
        "dq_r_conv_dt": "kg/kg/s (convective precipitation, column integral = PRATEC)",
        "cape": "J/kg (oracle ABE)", "convective_mask": "1",
        "conv_prog_profile_new": "kg/m^2/s (cloud-base mass flux at [:, -1])",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Positive tendency = increase. Condensation "
        "warms and dries; dq_r_conv_dt >= 0 is precipitation that leaves the "
        "column at the surface."
    ),
    "conserves": ["moisture"],
    "differentiable": True,  # AD runs, but NOT RULE 5 compliant: hard switches carry no gradient
    "reference": (
        "WRF v4.8.0 phys/module_cu_kfeta.F (06d4240ae9); Kain (2004), J. Appl. "
        "Meteor. 43, 170-181"
    ),
    "idealized_test": (
        "tests/atmosphere/test_kain_fritsch_oracle.py (oracle match of the "
        "port and of this wrapper on the fixture columns, column water budget, "
        "gradients); tests/unit/test_kain_fritsch_wrf.py"
    ),
}

# WKL = W0AVG*DX/25 km (module_cu_kfeta.F:983): the grid length the DTLCL
# formula is written for; DX = this x dtlcl_dx_scale.
_KF_REF_DX_M = 25.0e3
_KF_SEC_PER_MIN = 60.0

# KainFritschConfig fields of the former surrogate with no KF-Eta counterpart.
_SURROGATE_ONLY = (
    "w_thresh_offset", "w_thresh_scale", "trigger_sharpness", "cape_consumption_time",
    "parcel_perturb_T", "parcel_perturb_q", "epsilon_0", "delta_0", "cloud_depth_min",
    "cloud_depth_sharpness", "enable_shallow", "buoyancy_death_memory", "cape_threshold",
    "cape_sharpness", "cape_or_threshold", "cape_or_sharpness", "cape_or_w_ref", "M_b_max",
    "faithful_trigger", "condload_fresh_retention_fraction", "wkl_floor",
    "wkl_softplus_sharpness", "dtlcl_pos_sharpness", "faithful_entrainment",
    "cloud_top_detrainment_fraction", "cloud_top_detrainment_width_levels",
    "cape_removal_fraction", "precip_efficiency", "subsidence_solve", "theta_implicit",
)


def check_wrf_kfeta_config(config: KainFritschConfig) -> None:
    """Raise unless every surrogate-only field is at its default."""
    defaults = KainFritschConfig._field_defaults
    for name in _SURROGATE_ONLY:
        val = getattr(config, name)
        if isinstance(val, jax.core.Tracer) or val != defaults[name]:
            raise ValueError(
                f"KainFritschConfig.{name} has no counterpart in WRF KF-Eta "
                f"(module_cu_kfeta.F); the faithful port only accepts its default "
                f"{defaults[name]!r}, got {val!r}")


def _lego_constants() -> KFConstants:
    return KFConstants(
        g=constants.g, cp=constants.c_pd, r=constants.R_d,
        xlv0=constants.L_v,  # latent-ok: KF-Eta XLV0 argument (constant L_v, XLV1 = 0)
        xlv1=0.0,
        svp1=thermo.TETENS_E0_PA / 1000.0,  # kPa: KF reads ALIQ = SVP1*1000
        svp2=thermo.TETENS_A,
        svp3=constants.T_freeze - thermo.TETENS_B_K,
        svpt0=constants.T_freeze,
    )


def _oracle_params(config: KainFritschConfig) -> dict:
    return dict(
        dpmin=config.usl_depth_pa, wklcl_ref=config.wklcl_ref, wklcl_zref=config.wklcl_zref,
        dtlcl_coeff=config.dtlcl_coeff, dtlcl_exponent=config.dtlcl_exponent,
        rh_u00=config.rh_trigger_u00, rh_max=config.rh_trigger_rhmax,
        rh_slope=config.rh_trigger_slope, rad_min=config.rad_min_m, rad_max=config.rad_max_m,
        rad_wkl_ref=config.rad_wkl_ref, entrain_const=config.entrain_const,
        timec_min=config.timec_min_s, timec_max=config.timec_max_s,
    )


def kain_fritsch_wrf_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    w_grid: jax.Array,
    conv_prog_profile: jax.Array,
    dt: float,
    config: KainFritschConfig = KainFritschConfig(),
    *,
    u: jax.Array | None = None,
    v: jax.Array | None = None,
) -> tuple[ConvectionOutput, jax.Array]:
    """WRF KF-Eta convection on ``(ncol, nlev)`` top-down columns.

    Parameters
    ----------
    T, q_v, p_full : (ncol, nlev)
        Temperature [K], specific humidity [kg/kg], full-level pressure [Pa].
    p_half : (ncol, nlev+1)
        Interface pressure [Pa].
    w_grid : (ncol, nlev)
        Grid-scale vertical velocity [m/s] (KF's ``W0AVG1D``).
    conv_prog_profile : (ncol, nlev)
        Unused input carry (KF holds no state between calls here).
    dt : float
        Time step [s] (KF rounds ``TIMEC`` to a multiple of it).
    config : KainFritschConfig
    u, v : (ncol, nlev)
        Grid winds [m/s]; required.

    Returns
    -------
    (ConvectionOutput, conv_prog_profile_new)
    """
    del conv_prog_profile
    check_wrf_kfeta_config(config)
    if u is None or v is None:
        raise ValueError(
            "kain_fritsch variant='wrf_kfeta' needs the grid winds u and v (KF-Eta sets "
            "TIMEC = DX/VCONV and the precipitation efficiency from the shear)")
    ncol, nlev = T.shape
    dz, _, _ = compute_column_geometry(T, p_full, p_half, q_v=q_v)
    dp = p_half[:, 1:] - p_half[:, :-1]
    kc = _lego_constants()
    rhoe = dp / (kc.g * dz)
    dx = _KF_REF_DX_M * config.dtlcl_dx_scale

    def flip(a):
        return a[:, ::-1]

    o = kf_eta_para(
        flip(u), flip(v), flip(T), flip(q_v), flip(p_full), flip(dz), flip(w_grid),
        flip(rhoe), dt=dt, dx=dx, constants=kc, params=_oracle_params(config),
        trigger=3 if config.enable_rh_trigger_perturb else 1, warm_rain=True,
        cudt=dt / _KF_SEC_PER_MIN)
    # convective precipitation over the levels of its production (see module docstring)
    ems = o["ems"]
    has_mass = ems > 0.0
    rain_prod = (o["pptliq"] + o["pptice"]) * (o["ainc"] * (1.0 - o["fbfrc"]) * o["frc2"])[:, None]
    rain = jnp.where(has_mass, rain_prod / jnp.where(has_mass, ems, 1.0), 0.0)
    rain = jnp.where(o["fired"][:, None], rain, 0.0)
    dq_r = flip(rain + o["dqrdt"])
    fired = o["fired"]
    lcl = jnp.clip(o["lcl"] - 1, 0, nlev - 1)
    m_b = jnp.where(fired, jnp.take_along_axis(o["umf"], lcl[:, None], axis=1)[:, 0] / (dx * dx),
                    0.0)
    out = ConvectionOutput(
        dT_dt=flip(o["dtdt"]),
        dq_v_dt=flip(o["dqdt"]),
        dq_c_conv_dt=flip(o["dqcdt"]),
        cape=o["abe"],
        convective_mask=fired.astype(T.dtype),
        du_dt_conv=None,
        dv_dt_conv=None,
        dq_r_conv_dt=dq_r,
        precip_frozen_sfc=jnp.zeros((ncol,), T.dtype),
    )
    carry = jnp.zeros((ncol, nlev), dtype=T.dtype).at[:, -1].set(m_b)
    return out, carry
