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
    tau_x, tau_y : jax.Array or None
        Optional surface wind stress components [Pa], shape (ncol,), in the
        MODEL convention ``tau = -rho C_d |V| u`` (OPPOSES the wind — the
        stress felt BY the atmosphere; see
        ``core.bulk_flux.simple_bulk_fluxes``).  CMOR ``tauu``/``tauv``
        (surface DOWNWARD momentum flux, positive with the wind) are the
        NEGATIVE of these — the CMOR feed flips the sign.  ``None`` (the
        default) for schemes that do not export their surface stress;
        trailing optional fields so every existing constructor is unaffected.
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
    tau_x: jax.Array | None = None
    tau_y: jax.Array | None = None
