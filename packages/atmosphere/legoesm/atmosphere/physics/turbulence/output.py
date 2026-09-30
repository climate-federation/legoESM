"""Turbulence output container.

TurbulenceOutput is the common interface between all turbulence backends.
All schemes produce the same NamedTuple so that downstream integration
code can be backend-agnostic.
"""

from __future__ import annotations

from typing import NamedTuple

import jax


class TurbulenceOutput(NamedTuple):
    """Output from a turbulence scheme (backend-agnostic).

    Tendency fields have shape (ncol, nlev).
    Diffusivity fields have shape (ncol, nlev).
    Surface fields have shape (ncol,).

    Fields
    ------
    du_dt : jax.Array
        Zonal wind tendency [m/s^2], shape (ncol, nlev).
    dv_dt : jax.Array
        Meridional wind tendency [m/s^2], shape (ncol, nlev).
    dT_dt : jax.Array
        Temperature tendency [K/s], shape (ncol, nlev).
    dq_v_dt : jax.Array
        Water vapor mixing ratio tendency [kg/kg/s], shape (ncol, nlev).
    Km : jax.Array
        Eddy diffusivity for momentum [m^2/s], shape (ncol, nlev).
    Kh : jax.Array
        Eddy diffusivity for heat [m^2/s], shape (ncol, nlev).
    shflx : jax.Array
        Surface sensible heat flux [W/m^2], shape (ncol,).
    lhflx : jax.Array
        Surface latent heat flux [W/m^2], shape (ncol,).
    ustar : jax.Array
        Friction velocity [m/s], shape (ncol,).
    h_pbl : jax.Array
        Diagnosed PBL height [m], shape (ncol,).
    cloud_fraction : jax.Array or None
        Optional sub-grid LIQUID cloud fraction [-] diagnosed from the scheme's
        own PDF (CLUBB ADG1 double-Gaussian; clubb_lite single-Gaussian
        ``0.5 erfc(-s/(sqrt2 sigma_s))``), shape (ncol, nlev).  ``None`` (the
        default) for schemes that carry no PDF cloud closure (louis, tke, ...),
        in which case radiation keeps using the grid-scale cloud scheme
        (sundqvist / xu_randall).  A moist higher-order closure's cloud fraction
        is physically less overcast than the RH-diagnosed one over a saturated
        marine BL; ``cloud_scheme="clubb"`` routes THIS field to RRTMGP.
    dq_c_dt : jax.Array or None
        Optional CLOUD LIQUID mixing-ratio tendency [kg/kg/s], shape
        ``(ncol, nlev)``.  ``None`` (the default) for every scheme that does not
        exchange condensate with the host, which is all of them unless CLUBB's
        ``liquid_partition`` is on.  When present it REPLACES the host's cloud
        liquid with the closure's own (``(rcm − q_c)/dt``, CAM
        ``clubb_intr.F90:2160``) and is paired with a ``dq_v_dt`` that has had
        that same liquid removed, so the two together conserve total water.
        Dropping it while keeping ``dq_v_dt`` would therefore DESTROY water; a
        lane that cannot route it must refuse the configuration rather than
        ignore the field.
    wtheta_flux : jax.Array or None
        Optional DIAGNOSTIC kinematic heat flux ``⟨w'θ'⟩`` [K m/s] the scheme would
        transport at the given mean state, shape (ncol, nlev), on FULL levels. For a
        local K-closure this is ``−Kh·∂θ/∂z`` (down-gradient); for a nonlocal closure
        it INCLUDES the counter-gradient term (``−Kh·(∂θ/∂z − γ)``). ``None`` (the
        default) for schemes that do not expose it. It is a pure diagnostic — it does
        NOT feed the tendencies (those come from the flux DIVERGENCE / implicit solve)
        and so cannot change any run. Consumed only by the LES-suite Q1 diagnostic
        score (``les_suite`` compares it to the LES flux at the LES mean state).
    """
    du_dt: jax.Array
    dv_dt: jax.Array
    dT_dt: jax.Array
    dq_v_dt: jax.Array
    Km: jax.Array
    Kh: jax.Array
    shflx: jax.Array
    lhflx: jax.Array
    ustar: jax.Array
    h_pbl: jax.Array
    cloud_fraction: jax.Array | None = None
    wtheta_flux: jax.Array | None = None
    dq_c_dt: jax.Array | None = None
    # Surface water flux the kernel actually used as its moisture BC [kg/m2/s,
    # positive up]: the prescribed tile water or lhflx / L_v(T_sfc).  The water
    # ledgers and CMOR evspsbl read THIS, never lhflx / L_v.
    evap_sfc: jax.Array | None = None
