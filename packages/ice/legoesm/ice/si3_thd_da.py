"""SI3 lateral melt ``ice_thd_da``, NEMO-literal (rung O11).

Transcribes ``ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:97-218``
(``-fdefault-real-8 -O3``, no FMA in the object code) on NEMO's gathered
``ice1D`` cells of one category, as called at ``icethd.f90:210`` when
``ln_icedA``.  Every operation is rounded in source order
(``nemo_source_round``); ``pow`` follows the precision policy (glibc arm under
``transcendentals="libm"``).  MIN/MAX operand order and the ``a_i`` tests are
taken from the compiled ``icethd_da.o``, not from the source text.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.core import transcendentals
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ice.bitz_lipscomb import ieee_lt_zero
from legoesm.ice.constants_config import IceConstantsConfig
from legoesm.ice.si3_surface import fortran_max, fortran_min, fortran_sum0, ieee_is_zero

# --- ice_thd_da PARAMETERs (icethd_da.f90:167-170) and phycst (phycst.f90:36) ---
_DA_ZDMAX_M = 300.0  # zdmax, maximum floe diameter
_DA_ZCS = 0.66  # zcs, floe shape factor (Rothrock and Thorndike 1984)
_DA_ZM1_M_S = 3.0e-6  # zm1, melt-speed coefficient (Maykut and Perovich 1987)
_DA_ZM2 = 1.36  # zm2, melt-speed exponent
_RPI = 3.141592653589793  # rpi, phycst.f90:36 (binary64 == math.pi)
_SIGN_MASK = 1 << 63


def _ieee_gt_zero(value):
    """``value > 0._wp`` (``comisd``/``ja``) on the bit pattern: ``-value < 0``.

    The sign flip is an integer XOR, so a positive subnormal stays true under
    XLA CPU ``jit`` (a float compare flushes it); ``+0`` and NaN are false.
    """
    bits = jax.lax.bitcast_convert_type(value, jnp.uint64) ^ jnp.uint64(_SIGN_MASK)
    return ieee_lt_zero(jax.lax.bitcast_convert_type(bits, value.dtype))


def _da_zastar(rn_dmin, rn_beta):
    """``icethd_da.f90:174`` ``zastar = 1 / ( 1 - (rn_dmin / zdmax)**(1/rn_beta) )``.

    ``zdmax`` goes through the source rounding so XLA keeps ``/ 300`` a division
    (it otherwise multiplies by ``1/300``, which moves ``zastar`` at e.g. ``rn_dmin = 6.2``).
    """
    sr = nemo_source_round
    zdmax = sr(jnp.float64(_DA_ZDMAX_M))
    return sr(1.0 / sr(1.0 - transcendentals.pow(sr(rn_dmin / zdmax), sr(1.0 / rn_beta))))


class SI3ThdDa(NamedTuple):
    """``ice_thd_da`` outputs on the ``ice1D`` gathered cells of one category."""

    a_i: jax.Array
    h_i: jax.Array
    h_s: jax.Array
    sfx_lam: jax.Array
    hfx_thd: jax.Array
    wfx_lam: jax.Array


def si3_ice_thd_da(
    *,
    at_i,
    a_i,
    h_i,
    h_s,
    e_i,
    e_s,
    s_i,
    sz_i,
    sst,
    t_bo,
    sfx_lam,
    hfx_thd,
    wfx_lam,
    rn_beta,
    rn_dmin,
    rDt_ice,
    nn_icesal,
    constants: IceConstantsConfig,
) -> SI3ThdDa:
    """``ice_thd_da`` lateral melt (``icethd_da.f90:97-218``, ``ORCA2_SAS_ICE_O7_REF``).

    Inputs are NEMO's ``*_1d`` arrays at the call (``icethd.f90:210``): any cell
    shape, layer arrays ``e_i``/``sz_i`` (``..., nlay_i``) and ``e_s`` (``..., nlay_s``),
    ``e_*`` already in J/m3.  ``r1_nlay_*`` and ``r1_Dt_ice`` are NEMO's
    reciprocals (``icestp.f90:419,423-424``).  Every operation is rounded in source
    order; ``pow`` follows the precision policy (glibc arm under ``"libm"``).
    MIN/MAX operand order and the ``a_i > 0`` / ``a_i == 0`` tests follow the
    object code (``icethd_da.o``): the branch tests read the bit pattern.
    Non-differentiable: the two MIN, the MAX, both IF; forward bits are NEMO's,
    the ``at_i`` divisor of untaken cells is replaced by 1 for the gradient only.
    """
    sr = nemo_source_round
    named = dict(
        at_i=at_i,
        a_i=a_i,
        h_i=h_i,
        h_s=h_s,
        e_i=e_i,
        e_s=e_s,
        s_i=s_i,
        sz_i=sz_i,
        sst=sst,
        t_bo=t_bo,
        sfx_lam=sfx_lam,
        hfx_thd=hfx_thd,
        wfx_lam=wfx_lam,
        rn_beta=rn_beta,
        rn_dmin=rn_dmin,
        rDt_ice=rDt_ice,
    )
    missing = [k for k, v in named.items() if v is None]
    if missing or constants is None:
        raise ValueError(f"ice_thd_da needs every NEMO input; missing {missing}; no defaults")
    if type(nn_icesal) is not int or nn_icesal not in (1, 2, 3, 4):
        raise ValueError(f"nn_icesal must be the namelist integer 1..4, got {nn_icesal!r}")
    f64 = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in named.items()}
    at_i, a_i, h_i, h_s = f64["at_i"], f64["a_i"], f64["h_i"], f64["h_s"]
    e_i, e_s, sst, t_bo = f64["e_i"], f64["e_s"], f64["sst"], f64["t_bo"]
    nlay_i, nlay_s = e_i.shape[-1], e_s.shape[-1]
    if f64["sz_i"].shape != e_i.shape or e_i.shape[:-1] != a_i.shape or e_s.shape[:-1] != a_i.shape:
        raise ValueError("ice_thd_da layer arrays must be (cells..., nlay)")
    r1_nlay_i = 1.0 / float(nlay_i)  # icestp.f90:423
    r1_nlay_s = 1.0 / float(nlay_s)  # icestp.f90:424
    rdt = f64["rDt_ice"]
    r1_dt = sr(1.0 / rdt)  # icestp.f90:419
    rn_beta, rn_dmin = f64["rn_beta"], f64["rn_dmin"]
    rt0, rhoi, rhos = constants.T0, constants.rho_ice, constants.rho_snow
    pow_ = transcendentals.pow

    zastar = _da_zastar(rn_dmin, rn_beta)
    # :178-180 zs_i
    if nn_icesal == 4:
        zs_i = f64["sz_i"]
    else:
        zs_i = jnp.broadcast_to(f64["s_i"][..., None], e_i.shape)
    # :192 IF( a_i_1d(ji) > 0._wp ): comisd/ja; the object code computes the floe block inside it.
    active = _ieee_gt_zero(a_i)
    # :183 zdfloe = rn_dmin * ( zastar / ( zastar - at_i ) )**rn_beta
    zdfloe = sr(rn_dmin * pow_(sr(zastar / sr(zastar - at_i)), rn_beta))
    # :185 zperi = at_i * rpi / ( zcs * zdfloe )
    zperi = sr(sr(at_i * _RPI) / sr(_DA_ZCS * zdfloe))
    # :187 zwlat = zm1 * ( MAX( 0, sst - ( t_bo - rt0 ) ) )**zm2; maxsd: y > 0 ? y : +0
    zdt = fortran_max(jnp.zeros_like(sst), sr(sst - sr(t_bo - rt0)))
    zwlat = sr(_DA_ZM1_M_S * pow_(zdt, _DA_ZM2))
    # :189 zda_tot = MIN( zwlat*zperi*rDt_ice, at_i ); minsd keeps x only when x < at_i
    zda_tot = fortran_min(at_i, sr(sr(zwlat * zperi) * rdt))
    # :195 zda = MIN( a_i, zda_tot * a_i / at_i ); minsd keeps q only when q < a_i
    at_div = jnp.where(active, at_i, 1.0)
    zda = fortran_min(a_i, sr(sr(zda_tot * a_i) / at_div))
    zda_dt = sr(zda * r1_dt)
    # :198 sfx_lam += rhoi * zda * r1_Dt_ice * h_i * r1_nlay_i * SUM( zs_i(:) )
    sfx_new = sr(
        f64["sfx_lam"]
        + sr(sr(sr(sr(sr(rhoi * zda) * r1_dt) * h_i) * r1_nlay_i) * fortran_sum0(zs_i))
    )
    # :201-202 hfx_thd -= zda*r1_Dt_ice*( h_i*r1_nlay_i*SUM(e_i) + h_s*r1_nlay_s*SUM(e_s) )
    ice_heat = sr(sr(h_i * r1_nlay_i) * fortran_sum0(e_i))
    snow_heat = sr(sr(h_s * r1_nlay_s) * fortran_sum0(e_s))
    hfx_new = sr(f64["hfx_thd"] - sr(zda_dt * sr(ice_heat + snow_heat)))
    # :205 wfx_lam += zda*r1_Dt_ice*( rhoi*h_i + rhos*h_s )
    wfx_new = sr(f64["wfx_lam"] + sr(zda_dt * sr(sr(rhoi * h_i) + sr(rhos * h_s))))
    # :208 a_i -= zda; :211-214 IF( a_i == 0 ) h_i = h_s = 0 (ucomisd; movq $0 -> +0)
    a_new = sr(a_i - zda)
    emptied = ieee_is_zero(a_new)
    zero = jnp.zeros_like(h_i)
    h_i_new = jnp.where(emptied, zero, h_i)
    h_s_new = jnp.where(emptied, zero, h_s)

    def pick(new, old):
        return jnp.where(active, new, old)

    return SI3ThdDa(
        a_i=pick(a_new, a_i),
        h_i=pick(h_i_new, h_i),
        h_s=pick(h_s_new, h_s),
        sfx_lam=pick(sfx_new, f64["sfx_lam"]),
        hfx_thd=pick(hfx_new, f64["hfx_thd"]),
        wfx_lam=pick(wfx_new, f64["wfx_lam"]),
    )


__all__ = ("SI3ThdDa", "si3_ice_thd_da")
