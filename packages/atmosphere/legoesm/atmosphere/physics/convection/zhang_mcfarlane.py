"""Zhang-McFarlane deep convection: the CAM6 algorithm end to end.

``scheme="zhang_mcfarlane"`` runs the CESM2.1 ``zm_conv_tend`` sequence
(``zm_conv_intr.F90``) on the faithful kernels of
:mod:`legoesm.atmosphere.physics.convection._zm_cam6`:

1. ``zm_convr`` -- dilute-parcel CAPE (``buoyan_dilute``), cloud model
   (``cldprp``), quasi-equilibrium closure ``mb`` capped at the CFL bound,
   ``q1q2_pjr`` heating/moistening/detrainment;
2. ``physics_update`` -- the oracle applies the ``zm_convr`` tendencies to
   the state BEFORE the evaporation step, so the evaporation below sees
   ``T + heat/cp*dt`` and ``q + dqdt*dt``;
3. ``zm_conv_evap`` -- Sundqvist rain evaporation with snow/melt fusion
   heating (``cldfrc_fice``), on the updated state, against CAM's mixed-phase
   ``qsat`` (``svp_trans`` blend to ice between 0 and -20 degC);
4. ``momtran`` -- Richter-Rasch convective momentum transport with the
   Boville-Bretherton kinetic-energy dissipation heating.

CAM6 has NO cloud-base mass-flux carry: ``mb`` is diagnosed every step, so
``conv_prog_profile`` is not read.  The returned carry packs the diagnosed
``mb`` [kg/m^2/s] at ``[:, -1]`` for diagnostics only (zeros aloft), which
keeps the orchestrator's uniform carry schema.

5. ``convtran`` -- convective transport of the CAM6 ``convtran1``
   constituents the host carries (``CONVTRAN1_TRACERS``; ``tracers=``), on the
   input state, returned as signed column-conserving tendencies in
   ``ConvectionOutput.tracer_transport``.  ``tracers=None`` or ``config.transport_cloud_tracers=False`` transports nothing.

Departures inside the kernels are listed in ``_zm_cam6``.

The rain field ``dq_r_conv_dt`` is CAM's ``ntprprd``: the NET divergence of
the falling-precipitation flux (``prdprec - evpprec``), negative where rain
from above evaporates.  It column-integrates to the surface rain and is not
a per-layer condensate source; hosts must sum it (the bridges do, or refuse).

Host inputs beyond the convection contract: ``pblh`` (PBL height; every
bridge passes the turbulence scheme's lagged ``PhysicsState.pblh``, CAM's
pbuf ``pblh``; ``None`` raises), the total cloud
fraction for the evaporation (``cld_frac``; ``None`` -> 0, i.e. CAM's
``(1 - cldfrc)`` factor at its maximum), and ``land_frac``, which is required
unless ``config.land_fraction == "none"`` (an explicit aquaplanet: ocean
coefficients everywhere).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.convection._zm_cam6 import (
    Q_MIN_VAPOR,
    convtran,
    momtran,
    zm_conv_evap,
    zm_convr,
)
from legoesm.atmosphere.physics.convection._zm_dilute import LWMAX
from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig
from legoesm.atmosphere.physics.convection.mass_flux import compute_column_geometry
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.core.smooth import smooth_clamp, smooth_relu


__all__ = ("CONVTRAN1_TRACERS", "cam6_detrain_condensate",
           "validate_detrain_widths", "zhang_mcfarlane_convection")

# CAM6 constituents with ``is_convtran1`` (micro_mg_cam.F90:416-435, MG2) that
# zm_conv_intr.F90:829-846 transports with ``convtran``, mapped to legoESM
# tracer names: CLDLIQ, CLDICE, NUMLIQ, NUMICE, RAINQM, SNOWQM, NUMRAI.  CAM's
# NUMSNO has no legoESM tracer.
CONVTRAN1_TRACERS = ("q_c", "q_i", "N_c", "N_i", "q_r", "q_s", "N_r")


# --- CAM6 detrainment of ZM condensate (clubb_intr.F90:2345-2376) ---
# The CLUBB host books ZM's detrained condensate ``dlf`` on the post-CLUBB
# state of every macmic sub-step.  Literals exactly as the oracle writes them
# (cam_cesm2_1_rel_60, a03b84b7c4e34f965b115686f22a043b85739e56): the
# all-liquid / all-ice temperatures and ramp width (:2347-2352), and the
# deep-convection droplet / crystal radii and densities of the number source
# (:2365-2372, CAM's ``3.14``, not pi).  The all-ice temperature 238.15 K is
# ``_DETRAIN_T_LIQ_K - _DETRAIN_RAMP_K``: the clamp's upper end.
_DETRAIN_T_LIQ_K = 268.15
_DETRAIN_RAMP_K = 30.0
_DETRAIN_PI = 3.14
_DETRAIN_R_LIQ_M = 8.0e-6
_DETRAIN_R_ICE_M = 25.0e-6
_DETRAIN_RHO_LIQ = 997.0
_DETRAIN_RHO_ICE = 500.0


__physics_contract__ = {
    "summary": (
        "CAM6 Zhang-McFarlane deep convection (zm_conv_tend port): dilute-"
        "parcel CAPE, entraining/detraining updraft with downdraft, quasi-"
        "equilibrium closure, rain evaporation, momentum transport."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa",
        "u": "m/s", "v": "m/s",
        "conv_prog_profile": "unused (CAM6 ZM carries no state)",
        "dt": "s", "land_frac": "1", "cld_frac": "1", "pblh": "m",
        "tracers": "kg/kg or 1/kg (CAM6 convtran1 constituents, optional)",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "dq_c_conv_dt": "kg/kg/s (detrained cloud water dlf, >=0)",
        "dq_r_conv_dt": "kg/kg/s (NET rain flux divergence: production minus "
                        "evaporation of rain from above; signed per layer, "
                        "column-integrates to the surface rain; never a "
                        "per-layer tracer source)",
        "cape": "J/kg", "convective_mask": "1 (cape > capelmt)",
        "du_dt_conv": "m/s^2 (None if enable_cmt=False)",
        "dv_dt_conv": "m/s^2 (None if enable_cmt=False)",
        "conv_prog_profile_new": "kg/m^2/s (diagnosed mb at [:, -1])",
        "mass_flux_up": "kg/m^2/s (CAM cmfmc: net deep mass flux mu+md on "
                        "interfaces, top->bottom, bottom face 0)",
        "icwmr": "kg/kg (CAM ICWMRDP: in-cloud updraft condensate, 0 outside "
                 "convecting columns)",
        "tracer_transport": "tracer units/s (convtran; signed, column-"
                            "conserving; None without tracers)",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. dT_dt > 0 warms (includes the fusion "
        "heating of convective snow and the KE-dissipation heating); "
        "dq_v_dt > 0 moistens; dq_c_conv_dt >= 0 is a cloud-water source; "
        "dq_r_conv_dt integrates over the column to the surface convective "
        "precipitation (kg/m^2/s = -sum dp (dq_v_dt + dq_c_conv_dt)/g)."
    ),
    # moisture: vapour + cloud + rain sources close the column to rounding,
    # up to CAM's qneg3 floor (Q_MIN_VAPOR, 1e-12 kg/kg) applied to the
    # state the kernels read, never to the returned tendency (as in CAM);
    # momentum: momtran is flux-form (exact column conservation).
    "conserves": ["moisture", "momentum"],
    "differentiable": True,
    "reference": (
        "Zhang & McFarlane (1995), Atmos.-Ocean 33; Neale, Richter & Jochum "
        "(2008), J. Climate 21; Richter & Rasch (2008), J. Climate 21; "
        "CESM2.1 zm_conv.F90 / zm_conv_intr.F90"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_zm_cam6_oracle.py (Fortran "
        "transcription pin, water closure, jit parity, gradients); "
        "tests/unit/test_zhang_mcfarlane.py (host-level behaviour)."
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
    *,
    land_frac: jax.Array | None = None,
    cld_frac: jax.Array | None = None,
    pblh: jax.Array | None = None,
    pref_edge: jax.Array | None = None,
    tracers: dict | None = None,
) -> tuple[ConvectionOutput, jax.Array]:
    """CAM6 Zhang-McFarlane deep convection tendencies for one step.

    Parameters
    ----------
    T, q_v : jax.Array, shape (ncol, nlev)
        Temperature [K] and water vapour [kg/kg], surface at ``[:, -1]``.
    p_full, p_half : jax.Array
        Full ``(ncol, nlev)`` / half ``(ncol, nlev+1)`` pressures [Pa].
    u, v : jax.Array, shape (ncol, nlev)
        Winds [m/s] for ``momtran``.
    conv_prog_profile : jax.Array, shape (ncol, nlev)
        Not read (CAM6 ZM is diagnostic); see the module docstring.
    dt : float
        Physics time step [s] (the oracle's ``ztodt``).
    config : ZhangMcFarlaneConfig
    land_frac, cld_frac, pblh : optional
        Column land fraction ``(ncol,)``, total cloud fraction
        ``(ncol, nlev)`` and PBL height ``(ncol,)`` [m]; see the module
        docstring for the defaults when absent.
    pref_edge : optional
        Reference interface pressures ``(nlev+1,)`` [Pa]; CAM's
        40 hPa convection cap is fixed from these once.  ``None`` -> the cap
        follows each column's own interfaces (see ``_zm_cam6.zm_convr``).
    tracers : dict, optional
        ``{name: (ncol, nlev)}`` moist tracers to transport with ``convtran``
        (the host passes the ``CONVTRAN1_TRACERS`` it carries).
    """
    ncol, nlev = T.shape
    dtype = T.dtype
    q_v, p_full, p_half, u, v = (jnp.asarray(a, dtype) for a in (q_v, p_full, p_half, u, v))
    cp = constants.c_pd
    # CAM's physics_update floors Q at qmin before zm_conv_tend runs (qneg3);
    # the host hands exact zeros, on which the Fortran itself would NaN.
    q_v = jnp.maximum(q_v, Q_MIN_VAPOR)

    dz, _, z = compute_column_geometry(T, p_full, p_half, q_v=q_v)
    zf = jnp.concatenate(
        [jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1], jnp.zeros((ncol, 1), dtype)], axis=1)
    if config.land_fraction == "required":
        if land_frac is None:
            raise ValueError(
                "zhang_mcfarlane_convection: land_frac is required (it selects "
                "the land/ocean autoconversion coefficient per column). Pass "
                "the column land fraction, or set ZhangMcFarlaneConfig."
                "land_fraction='none' for an aquaplanet.")
        lf = jnp.asarray(land_frac, dtype).reshape(ncol)
    elif config.land_fraction == "none":
        if land_frac is not None:
            raise ValueError(
                "zhang_mcfarlane_convection: land_fraction='none' (aquaplanet) "
                "but a land_frac was passed; drop one of the two.")
        lf = jnp.zeros((ncol,), dtype)
    else:
        raise ValueError(
            f"Unknown ZhangMcFarlaneConfig.land_fraction scheme "
            f"{config.land_fraction!r}; choose 'required' or 'none'.")
    cf = (jnp.zeros_like(T) if cld_frac is None
          else jnp.asarray(cld_frac, dtype).reshape(ncol, nlev))

    conv = zm_convr(
        T, q_v, p_full, p_half, z, zf, lf, dt,
        pblh=None if pblh is None else jnp.asarray(pblh, dtype).reshape(ncol),
        tpert=config.parcel_tpert, capelmt=config.capelmt, tau=config.tau,
        num_cin=int(config.num_cin), dmpdz=config.dmpdz,
        tiedke_add=config.tiedke_add, lwmax=LWMAX, c0_lnd=config.c0_lnd,
        c0_ocn=config.c0_ocn, alfa=config.alfa, limcnv_p_pa=config.limcnv_p_pa,
        pref_edge=pref_edge)

    # zm_conv_tend: physics_update before zm_conv_evap (zm_conv_intr.F90:686).
    T1 = T + conv.heat / cp * dt
    q1 = jnp.maximum(q_v + conv.dqdt * dt, Q_MIN_VAPOR)   # qneg3 after the update, too
    pdel = p_half[:, 1:] - p_half[:, :-1]
    evap = zm_conv_evap(T1, p_full, pdel, q1, conv.rprd, cf, dt, conv.prec, ke=config.ke)

    dT_dt = (conv.heat + evap.tend_s) / cp
    dq_v_dt = conv.dqdt + evap.tend_q

    if config.enable_cmt:
        mom = momtran(u, v, conv.mu, conv.md, conv.du, conv.eu, conv.ed, conv.dp,
                      conv.jt, conv.maxg, conv.msg, dt,
                      momcu=config.momcu, momcd=config.momcd)
        dT_dt = dT_dt + mom.seten / cp
        du_dt_conv, dv_dt_conv = mom.dudt, mom.dvdt
    else:
        du_dt_conv = dv_dt_conv = None

    transport = None
    if tracers is not None and config.transport_cloud_tracers:
        # zm_conv_intr.F90:842: convtran on state1 (ZM's T/Q/U/V updates do
        # not touch these constituents, so this is the input state).
        transport = {
            name: convtran(jnp.asarray(c, dtype).reshape(ncol, nlev), conv.mu, conv.md,
                           conv.du, conv.eu, conv.ed, conv.dp, conv.jt, conv.maxg, conv.msg)
            for name, c in tracers.items()}

    out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=conv.dlf,
        cape=conv.cape,
        convective_mask=conv.ideep.astype(dtype),
        du_dt_conv=du_dt_conv,
        dv_dt_conv=dv_dt_conv,
        dq_r_conv_dt=evap.ntprprd,
        # surface snow flux of zm_conv_evap (its fusion heat is in tend_s):
        # the land's convective snowfall (F29)
        precip_frozen_sfc=evap.snow,
        # pbuf fields clubb_intr's deepcu reads (clubb_intr.F90:2501-2504):
        # CMFMC = zm_convr's net mass flux mc (mu + md, mb-scaled) scattered
        # to interface k = the top face of layer k, bottom face 0, hPa/s ->
        # kg/m^2/s (zm_conv.F90:1189, zm_conv_intr.F90:661); ICWMRDP = the
        # in-cloud updraft condensate ql, zeroed for every column before the
        # gather (zm_conv.F90:538) and rewritten in gathered ones (:1193).
        mass_flux_up=jnp.concatenate(
            [conv.mc * 100.0 / constants.g, jnp.zeros((ncol, 1), dtype)], axis=1),
        icwmr=jnp.where(conv.ideep[:, None], conv.ql, 0.0),
        tracer_transport=transport,
    )
    mb_kg = conv.mb * 100.0 / constants.g
    conv_prog_profile_new = jnp.zeros_like(conv_prog_profile).at[:, -1].set(mb_kg)
    return out, conv_prog_profile_new


def validate_detrain_widths(w_T, w_n) -> None:
    """The one legal range of the two detrainment smoothing widths: 0 (CAM's
    hard switch) or finite and >= sqrt(finfo(float32).tiny) (~1.08e-19).
    Shared by :func:`cam6_detrain_condensate` and
    ``ExperimentConfig.validate_strict``.  Far enough below the floor the
    divisions ``30 / w_T`` (the ramp sharpness) and ``dlf / w_n`` overflow (measured: fp32
    outputs non-finite at w_T = 1.2e-38 and w_n = 1e-100, fp64 NaN at
    w_n = 1e-320); the run's dtype is unknown here, so the floor keeps both
    quotients <~ 1e20 x their numerators, far below the float32 maximum."""
    floor = math.sqrt(float(jnp.finfo(jnp.float32).tiny))
    if not all(w == 0 or (math.isfinite(w) and w >= floor) for w in (w_T, w_n)):
        raise ValueError(
            "ZhangMcFarlaneConfig detrain_phase_width_K / "
            "detrain_number_width_kgkg_s must be finite and >= 0, and either 0 "
            f"(CAM's hard switch) or >= {floor:.3e}; got {w_T!r}, {w_n!r}.")


def cam6_detrain_condensate(T: jax.Array, dlf: jax.Array,
                            config: ZhangMcFarlaneConfig):
    """CAM6 detrainment of ZM condensate, clubb_intr.F90:2345-2376.

    ``T`` is the state the CLUBB host holds after its own update in the
    macmic sub-step (``state1``, clubb_intr.F90:2251); ``dlf`` [kg/kg/s] is
    ZM's detrained condensate (pbuf DLF).  The ``zmconv_microp = .false.``
    branch (namelist default) with CAM6's shallow detrainment ``dlf2 = 0``
    (clubb_intr.F90:2327), whose terms therefore vanish:

    ``dum1 = 0`` above 268.15 K, ``1`` below 238.15 K, ``(268.15 - T)/30``
    between; ``dq_c = dlf (1 - dum1)``, ``dq_i = dlf dum1``,
    ``dN_c = 3 max(0, dlf) (1 - dum1) / (4 3.14 (8e-6)^3 997)``,
    ``dN_i = 3 max(0, dlf) dum1 / (4 3.14 (25e-6)^3 500)``,
    ``s = dlf dum1 latice`` (fusion heating, applied as ``s/cpair``).

    KNOWN DIFFERENCE (RULE 5): both switches depend on traced state and are
    smoothed by default.  The ice fraction is
    ``smooth_clamp((268.15 - T)/30, 0, 1, sharpness=30/w_T)`` with
    ``w_T = config.detrain_phase_width_K`` (0.1 K), and ``max(0, dlf)`` in the
    number sources is ``w_n smooth_relu(dlf/w_n, 1)`` with
    ``w_n = config.detrain_number_width_kgkg_s`` (1e-12 kg/kg/s).  A width
    of 0 recovers CAM's hard switch exactly (oracle match 1e-10).  At the
    default: ``dum1`` departs from CAM by up to ``ln 2 w_T/30`` (~2.3e-3) at
    the 268.15 / 238.15 K corners; it never exceeds 1 but dips ~1e-133 below
    0 well above 268.15 K (fp64), so ``dq_i`` can be that fraction of ``dlf``
    negative there;
    the number sources gain ``~ln 2 w_n`` of ``dlf`` where CAM's ``dlf`` is 0
    (a spurious droplet source ~0.3 /kg/s).  ``dq_c + dq_i = dlf`` still
    holds.  Measured on the 15 CAM oracle columns (max |ours - CAM| / max |CAM|
    per field): heating and dq_c / dq_i <= 1.6e-8, dN_c 1.1e-4 and dN_i
    1.3e-4 (the dlf = 0 levels).  Per-case gaps are stored in
    tests/atmosphere/test_zm_detrain_oracle.py.

    Returns ``(dT_dt [K/s], dq_c_dt, dq_i_dt [kg/kg/s], dN_c_dt, dN_i_dt
    [1/kg/s])``; signed as sources (+ = gain), same shape as ``T``.
    """
    w_T = config.detrain_phase_width_K
    w_n = config.detrain_number_width_kgkg_s
    validate_detrain_widths(w_T, w_n)
    dum1 = smooth_clamp((_DETRAIN_T_LIQ_K - T) / _DETRAIN_RAMP_K, 0.0, 1.0,
                        math.inf if w_T == 0 else _DETRAIN_RAMP_K / w_T)
    dlf_pos = (jnp.maximum(0.0, dlf) if w_n == 0
               else w_n * smooth_relu(dlf / w_n, 1.0))
    dq_c = dlf * (1.0 - dum1)
    dq_i = dlf * dum1
    dN_c = 3.0 * (dlf_pos * (1.0 - dum1)) / (
        4.0 * _DETRAIN_PI * _DETRAIN_R_LIQ_M**3 * _DETRAIN_RHO_LIQ)
    dN_i = 3.0 * (dlf_pos * dum1) / (
        4.0 * _DETRAIN_PI * _DETRAIN_R_ICE_M**3 * _DETRAIN_RHO_ICE)
    dT = dlf * dum1 * constants.L_f / constants.c_pd
    return dT, dq_c, dq_i, dN_c, dN_i
