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
   heating (``cldfrc_fice``), on the updated state;
4. ``momtran`` -- Richter-Rasch convective momentum transport with the
   Boville-Bretherton kinetic-energy dissipation heating.

CAM6 has NO cloud-base mass-flux carry: ``mb`` is diagnosed every step, so
``conv_prog_profile`` is not read.  The returned carry packs the diagnosed
``mb`` [kg/m^2/s] at ``[:, -1]`` for diagnostics only (zeros aloft), which
keeps the orchestrator's uniform carry schema.

Not wired (host contract): the oracle also runs ``convtran`` on cloud
liquid/ice.  ``_zm_cam6.convtran`` is ported and pinned, but this scheme
receives no ``q_c``/``q_i`` and ``ConvectionOutput`` has no signed cloud
transport slot (``dq_c_conv_dt`` is a non-negative source), so that call is
left out.  Departures inside the kernels are listed in ``_zm_cam6``.

The rain field ``dq_r_conv_dt`` is CAM's ``ntprprd``: the NET divergence of
the falling-precipitation flux (``prdprec - evpprec``), negative where rain
from above evaporates.  It column-integrates to the surface rain and is not
a per-layer condensate source; hosts must sum it (the bridges do, or refuse).

Host inputs the oracle has that the convection contract lacks: ``pblh``
(PBL height; ``None`` -> ``pbl_top_pa`` launch bound), the total cloud
fraction for the evaporation (``cld_frac``; ``None`` -> 0, i.e. CAM's
``(1 - cldfrc)`` factor at its maximum), and ``land_frac`` (``None`` -> 0,
ocean coefficients everywhere; the bridge passes it when the grid has it).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.convection._zm_cam6 import (
    Q_MIN_VAPOR,
    momtran,
    zm_conv_evap,
    zm_convr,
)
from legoesm.atmosphere.physics.convection._zm_dilute import LWMAX
from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig
from legoesm.atmosphere.physics.convection.mass_flux import compute_column_geometry
from legoesm.atmosphere.physics.convection.output import ConvectionOutput


__all__ = ("zhang_mcfarlane_convection",)


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
    lf = (jnp.zeros((ncol,), dtype) if land_frac is None
          else jnp.asarray(land_frac, dtype).reshape(ncol))
    cf = (jnp.zeros_like(T) if cld_frac is None
          else jnp.asarray(cld_frac, dtype).reshape(ncol, nlev))

    conv = zm_convr(
        T, q_v, p_full, p_half, z, zf, lf, dt,
        pblh=None if pblh is None else jnp.asarray(pblh, dtype).reshape(ncol),
        tpert=config.parcel_tpert, capelmt=config.capelmt, tau=config.tau,
        num_cin=int(config.num_cin), dmpdz=config.dmpdz,
        tiedke_add=config.tiedke_add, lwmax=LWMAX, c0_lnd=config.c0_lnd,
        c0_ocn=config.c0_ocn, alfa=config.alfa, limcnv_p_pa=config.limcnv_p_pa,
        pbl_top_pa=config.pbl_top_pa, pref_edge=pref_edge)

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

    out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=conv.dlf,
        cape=conv.cape,
        convective_mask=conv.ideep.astype(dtype),
        du_dt_conv=du_dt_conv,
        dv_dt_conv=dv_dt_conv,
        dq_r_conv_dt=evap.ntprprd,
        # pbuf fields clubb_intr's deepcu reads (clubb_intr.F90:2501-2504):
        # CMFMC = zm_convr's net mass flux mc (mu + md, mb-scaled) scattered
        # to interface k = the top face of layer k, bottom face 0, hPa/s ->
        # kg/m^2/s (zm_conv.F90:1189, zm_conv_intr.F90:661); ICWMRDP = the
        # in-cloud updraft condensate ql, zeroed for every column before the
        # gather (zm_conv.F90:538) and rewritten in gathered ones (:1193).
        mass_flux_up=jnp.concatenate(
            [conv.mc * 100.0 / constants.g, jnp.zeros((ncol, 1), dtype)], axis=1),
        icwmr=jnp.where(conv.ideep[:, None], conv.ql, 0.0),
    )
    mb_kg = conv.mb * 100.0 / constants.g
    conv_prog_profile_new = jnp.zeros_like(conv_prog_profile).at[:, -1].set(mb_kg)
    return out, conv_prog_profile_new
