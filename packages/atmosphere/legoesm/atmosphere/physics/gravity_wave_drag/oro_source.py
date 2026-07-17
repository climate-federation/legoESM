"""Shared E3SM ``gw_oro_src`` depth-averaged orographic source region.

The one canonical implementation of the E3SM low-level source machinery
(gw_oro.F90:119-145), factored out of ``e3sm_cam.gw_oro_src`` so the
lightweight ``mcfarlane`` scheme can reuse it (GWD-recon G3; pre-impl-search
doctrine: extend, never re-derive):

* source-region selection — the surface midpoint always, plus every
  bottom-half midpoint the mountain penetrates
  (``hdsp > sqrt(zm[k]*zm[k+1])``, gw_oro.F90:126-128);
* ``src_level`` — the interface above the topmost included midpoint
  (E3SM ``src_level = kk-1``, 1-based);
* dp-weighted source averages ``X_src = sum(X_k*dpm_k)/dpsrc`` with the
  PRESSURE-INTERVAL normalisation ``dpsrc = pint(surface) - pint(src_level)``
  (gw_oro.F90:139 — exactly E3SM, not ``sum(dpm)``).

Index convention: k = 0 model top, k = nlev-1 surface; interface k sits
ABOVE midpoint k.  The penetration inequality is E3SM's sharp test (its
sub-gradient is benign; see ``e3sm_cam.gw_oro_src``).

No physics constants are used here — all inputs are prepared by the caller
(``e3sm_cam`` builds ``rho_mid = pmid/(R_d*T)`` and its ``gw_prof`` midpoint
``nm``; ``mcfarlane`` passes its diagnostic ``rho`` and the shared
``brunt_vaisala_n_full`` — the midpoint-N departure its header documents).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

__physics_contract__ = {
    "summary": (
        "E3SM gw_oro_src low-level source region: penetration-mask level "
        "selection + dp-weighted source averages (rho, u, v, N) with the "
        "pressure-interval dpsrc normalisation."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "rho_mid": "kg/m^3", "hdsp": "m",
        "pint": "Pa", "dpm": "Pa", "zm": "m", "nm": "1/s",
    },
    "outputs": {
        "rsrc": "kg/m^3", "usrc": "m/s", "vsrc": "m/s", "nsrc": "1/s",
        "src_level": "1 (interface index, 0-based)",
    },
    "sign_convention": (
        "k=0 model top, k=nlev-1 surface; interface k above midpoint k; "
        "src_level is the interface above the topmost included midpoint "
        "(deposition belongs strictly ABOVE it; E3SM holds tau constant "
        "at and below it)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "E3SM v3.0.1 gw_oro.F90:119-145 (gw_oro_src)",
    "idealized_test": "tests/atmosphere/hydrostatic/unit/test_gwd_oro_source.py",
}


def depth_averaged_oro_source(
    u: jax.Array,
    v: jax.Array,
    rho_mid: jax.Array,
    hdsp: jax.Array,
    pint: jax.Array,
    dpm: jax.Array,
    zm: jax.Array,
    nm: jax.Array,
):
    """E3SM source-region selection + dp-weighted averages (gw_oro.F90:119-145).

    Parameters
    ----------
    u, v, rho_mid, zm, nm, dpm : (ncol, nlev)
        Midpoint winds, density, heights, Brunt-Väisälä frequency, and
        layer pressure thicknesses.
    hdsp : (ncol,)
        Streamline-displacement height [m] (E3SM ``2*sgh``; whatever
        displacement convention the caller runs).
    pint : (ncol, nlev+1)
        Interface pressures.

    Returns
    -------
    (rsrc, usrc, vsrc, nsrc, src_level)
        dp-weighted source averages (each (ncol,)) and the 0-based source
        interface index (ncol,) int32.
    """
    ncol, nlev = u.shape
    k_idx = jnp.arange(nlev)

    # Penetration test hdsp > sqrt(zm[i]*zm[i+1]) for bottom-half midpoints
    # i in [nlev//2 - 1, nlev-2] (E3SM loop kk = pver-1 .. pver/2, 1-based).
    gm = jnp.sqrt(jnp.abs(zm[:, :-1] * zm[:, 1:]))       # (ncol, nlev-1)
    lo = (nlev // 2) - 1
    in_loop = (k_idx >= lo) & (k_idx <= nlev - 2)        # (nlev,)
    gm_full = jnp.concatenate([gm, gm[:, -1:]], axis=1)  # pad idx nlev-1 (unused)
    penetrates = hdsp[:, None] > gm_full                 # (ncol, nlev)
    include = jnp.where(
        k_idx[None, :] == (nlev - 1),
        True,
        in_loop[None, :] & penetrates,
    )                                                     # (ncol, nlev) bool
    w = include.astype(u.dtype) * dpm                     # dp-weights

    # src_level: interface above the topmost included midpoint.
    big = nlev
    top_inc = jnp.min(jnp.where(include, k_idx[None, :], big), axis=1)
    src_level = top_inc.astype(jnp.int32)                 # (ncol,) interface idx

    # dpsrc = pint(surface) - pint(src_level)  (gw_oro.F90:139).
    pint_at_src = jnp.take_along_axis(
        pint, src_level[:, None].astype(jnp.int32), axis=1
    )[:, 0]
    dpsrc = pint[:, nlev] - pint_at_src
    dpsrc = jnp.where(dpsrc > 0.0, dpsrc, 1.0)

    rsrc = jnp.sum(w * rho_mid, axis=1) / dpsrc
    usrc = jnp.sum(w * u, axis=1) / dpsrc
    vsrc = jnp.sum(w * v, axis=1) / dpsrc
    nsrc = jnp.sum(w * nm, axis=1) / dpsrc
    return rsrc, usrc, vsrc, nsrc, src_level
