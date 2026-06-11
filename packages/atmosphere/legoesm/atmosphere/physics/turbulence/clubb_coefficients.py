"""CLUBB skewness-dependent C-coefficient family (CAM-default tree).

The xm/wpxp advance needs the pressure-term coefficients ``C6rt_Skw_fnc``,
``C6thl_Skw_fnc`` and ``C7_Skw_fnc``. For the CAM-default flags
(``l_diag_Lscale_from_tau = .false.`` and ``l_use_C7_Richardson = .false.``)
these are **skewness functions** of ``Skw_zm`` (NOT the ARM Richardson/constant
that the CLUBB-JAX reference hard-wires), so they are ported from the CLUBB
Fortran (``advance_xm_wpxp_module.F90``):

  * C6rt/C6thl: ``Cb + (C - Cb)·exp(-½(Skw/Cc)^2)`` (:func:`clubb_wp23.compute_skw_fnc`)
    then the Lscale-based stable-region damping (:func:`damp_coefficient`);
  * C7: the same skewness function, no damping. With the CAM default ``C7 = C7b``
    this reduces to the constant ``C7b``.

The C1/C11 skewness functions (wp2/wp3) are computed inside ``advance_wp2_wp3``;
this module supplies the xm/wpxp C6/C7. Pure / JIT-safe / differentiable. The
CAM branch has no CLUBB-JAX oracle (the reference is ARM) — validated against the
Fortran formula + the parity-tested ``compute_skw_fnc`` sub-piece.
"""

from __future__ import annotations

import jax.numpy as jnp
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid
from legoesm.atmosphere.physics.turbulence.clubb_wp23 import compute_skw_fnc


def damp_coefficient(coefficient, Cx_Skw_fnc, max_coeff_value, altitude_threshold,
                     threshold, Lscale_zm, gr: CLUBBGrid):
    """Lscale-based damping of a skewness coefficient (``damp_coefficient``).

    In stably stratified regions (``Lscale_zm < threshold`` AND ``zm >
    altitude_threshold``) the coefficient is ramped linearly toward
    ``max_coeff_value`` as ``Lscale_zm -> 0``:
    ``max_coeff_value + (coefficient - max_coeff_value)/threshold · Lscale_zm``;
    elsewhere the input ``Cx_Skw_fnc`` is unchanged. ``coefficient``/
    ``max_coeff_value``/``altitude_threshold``/``threshold`` are ``(ncol,)``;
    ``Cx_Skw_fnc``/``Lscale_zm`` are ``(ncol, nzm)``.
    """
    thr = threshold[:, None]
    mx = max_coeff_value[:, None]
    cond = (Lscale_zm < thr) & (gr.zm > altitude_threshold[:, None])
    damped = mx + ((coefficient[:, None] - mx) / thr) * Lscale_zm
    return jnp.where(cond, damped, Cx_Skw_fnc)


def compute_C6_C7_Skw_fnc(Skw_zm, Lscale_zm, config, gr: CLUBBGrid):
    """The xm/wpxp ``C6rt``/``C6thl``/``C7`` skewness coefficients (CAM branch).

    ``C6rt``/``C6thl`` are skewness functions of ``Skw_zm`` then Lscale-damped;
    ``C7`` is the skewness function (no damping; CAM ``C7 = C7b`` → constant).
    All zm-level ``(ncol, nzm)``. Returns ``(C6rt_Skw_fnc, C6thl_Skw_fnc,
    C7_Skw_fnc)``.
    """
    p = config.params
    ng = Skw_zm.shape[0]

    def col(v):
        return jnp.full((ng,), v, dtype=Skw_zm.dtype)

    C6rt_raw = compute_skw_fnc(col(p.C6rt), col(p.C6rtb), col(p.C6rtc), Skw_zm)
    C6rt = damp_coefficient(col(p.C6rt), C6rt_raw, col(p.C6rt_Lscale0),
                            col(p.altitude_threshold), col(p.wpxp_L_thresh), Lscale_zm, gr)
    C6thl_raw = compute_skw_fnc(col(p.C6thl), col(p.C6thlb), col(p.C6thlc), Skw_zm)
    C6thl = damp_coefficient(col(p.C6thl), C6thl_raw, col(p.C6thl_Lscale0),
                             col(p.altitude_threshold), col(p.wpxp_L_thresh), Lscale_zm, gr)
    C7 = compute_skw_fnc(col(p.C7), col(p.C7b), col(p.C7c), Skw_zm)
    return C6rt, C6thl, C7


__all__ = ["damp_coefficient", "compute_C6_C7_Skw_fnc"]
