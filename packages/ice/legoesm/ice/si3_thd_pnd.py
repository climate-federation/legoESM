"""SI3 level-ice melt-pond thermodynamics ``ice_thd_pnd`` / ``pnd_LEV`` / ``ice_perm_eff``, NEMO-literal (rung O12).

Transcribes ``ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:162-264``
(``ice_thd_pnd``), ``:329-582`` (``pnd_LEV``), ``:1372-1422`` (``ice_perm_eff``),
and ``:1424-1487`` (``ice_thd_pnd_init`` selector resolution), plus the category
exit to pond entry bridge ``icethd.f90:176-223,484-588`` (``-fdefault-real-8 -O3``,
no FMA in the object code).  Every arithmetic operation is rounded in compiled
instruction order via ``nemo_source_round``; ``EXP`` follows the precision policy
(glibc ``libm`` arm under ``transcendentals="libm"``); ``MIN``/``MAX`` operand
roles and the fused Darcy-flushing ``minsd`` follow ``icethd_pnd.o``.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
from legoesm.core import transcendentals
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ice.bitz_lipscomb import (
    SI3DHFluxes,
    SI3O10CategoryStages,
    ieee_lt_zero,
    si3_ice_thd_o10_category_stages,
)
from legoesm.ice.constants_config import IceConstantsConfig
from legoesm.ice.si3_surface import fortran_sum0
from legoesm.ice.si3_thd_da import SI3ThdDa, si3_ice_thd_da

# --- pnd_LEV / ice_perm_eff PARAMETERs (icethd_pnd.f90:57,385-387) and par_ice (par_ice.f90:51-52) ---
_PND_ZASPECT = 0.8  # zaspect, pond aspect ratio (icethd_pnd.f90:385)
_PND_ZTP_C = -0.15  # zTp, reference freezing temperature [degC] (icethd_pnd.f90:386)
_PND_ZVISC = 1.79e-3  # zvisc, water dynamic viscosity [Pa.s] (icethd_pnd.f90:387)
_PND_TINY_AI = 0.01  # minimum ice or pond fraction threshold (icethd_pnd.f90:437,533)
_PND_DEPTH_CAP_FACTOR = 0.25  # 0.5**2 depth-driven overflow factor (icethd_pnd.f90:471)
_PND_FREEZE_HALF = 0.5  # quadratic formula 1/2 factor (icethd_pnd.f90:490)
_PND_FREEZE_FOUR = 4.0  # quadratic formula 4*a*c factor (icethd_pnd.f90:491)
_PND_NO_LID_DECAY = 0.01  # Holland 2012 exponential decay scale (icethd_pnd.f90:500)
_PND_LID_RATIO_MAX = 10.0  # h_il > 10 * h_ip pond-removal ratio (icethd_pnd.f90:537)
_PERM_VC19_C1 = 18.7  # VC19 linear brine-salinity coefficient (icethd_pnd.f90:1393)
_PERM_VC19_C2 = 0.519  # VC19 quadratic brine-salinity coefficient (icethd_pnd.f90:1393)
_PERM_VC19_C3 = 0.00535  # VC19 cubic brine-salinity coefficient (icethd_pnd.f90:1393)
_PERM_SCALE = 3.0e-8  # permeability prefactor [m2] (icethd_pnd.f90:1416)
_EPSI06 = 1.0e-6  # epsi06 (par_ice.f90:51)
_EPSI10 = 1.0e-10  # epsi10 (par_ice.f90:52)
_SIGN_MASK = 1 << 63

_PND_NAMELIST = (
    "ln_icedH",
    "rn_himin",
    "ln_pnd",
    "ln_pnd_LEV",
    "rn_apnd_min",
    "rn_apnd_max",
    "rn_pnd_flush",
    "ln_pnd_lids",
    "ln_pnd_rain",
    "ln_pnd_CST",
    "rn_apnd",
    "rn_hpnd",
    "ln_pnd_TOPO",
    "ln_pnd_alb",
    "rn_pnd_hl_min",
    "rn_pnd_hl_max",
    "ln_icediachk",
)


def _maxsd(dst: jax.Array, src: jax.Array) -> jax.Array:
    """x86 ``maxsd src, dst``: returns ``dst`` when ``dst > src``, else ``src`` (tie/NaN -> ``src``)."""
    dst = jnp.asarray(dst, dtype=jnp.float64)
    src = jnp.asarray(src, dtype=jnp.float64)
    return jnp.where(dst > src, dst, src)


def _minsd(dst: jax.Array, src: jax.Array) -> jax.Array:
    """x86 ``minsd src, dst``: returns ``dst`` when ``dst < src``, else ``src`` (tie/NaN -> ``src``)."""
    dst = jnp.asarray(dst, dtype=jnp.float64)
    src = jnp.asarray(src, dtype=jnp.float64)
    return jnp.where(dst < src, dst, src)


def _ieee_gt_zero(value: jax.Array) -> jax.Array:
    """``value > 0._wp`` (``comisd``/``ja``) on the bit pattern: ``-value < 0``."""
    bits = jax.lax.bitcast_convert_type(value, jnp.uint64) ^ jnp.uint64(_SIGN_MASK)
    return ieee_lt_zero(jax.lax.bitcast_convert_type(bits, value.dtype))


@jax.custom_jvp
def _nemo_sqrt(x: jax.Array) -> jax.Array:
    """x86 ``sqrtsd`` with source-rounding barrier and finite zero-radicand JVP."""
    return nemo_source_round(jnp.sqrt(nemo_source_round(x)))


@_nemo_sqrt.defjvp
def _nemo_sqrt_jvp(primals, tangents):
    (x,) = primals
    (xdot,) = tangents
    y = _nemo_sqrt(x)
    pos = _ieee_gt_zero(x)
    safe_two_y = jnp.where(pos, 2.0 * y, 1.0)
    ydot = jnp.where(pos, xdot / safe_two_y, jnp.zeros_like(y))
    return y, ydot


def _required(name: str, value):
    if value is None:
        raise ValueError(f"SI3 O12 pond stage requires {name!r}; no default")
    return jnp.asarray(value, dtype=jnp.float64)


class SI3PndLevOutputs(NamedTuple):
    """``pnd_LEV`` outputs on gathered cells (``icethd_pnd.f90:565-573``)."""

    a_ip: jax.Array
    h_ip: jax.Array
    h_il: jax.Array
    v_ip: jax.Array
    v_il: jax.Array
    wfx_pnd: jax.Array


class SI3PondOutputs(NamedTuple):
    """``ice_thd_pnd`` outputs on the full 2D/3D domain (``icethd_pnd.f90:203-264``)."""

    at_i: jax.Array
    a_ip: jax.Array
    v_ip: jax.Array
    h_ip: jax.Array
    v_il: jax.Array
    h_il: jax.Array
    a_ip_frac: jax.Array
    wfx_pnd: jax.Array


class SI3O12CategoryOutput(NamedTuple):
    """One category's O9 + O10 + O11 + ``ice_thd_1d2d(jl, 2)`` exit."""

    o10: SI3O10CategoryStages
    da: SI3ThdDa | None
    a_i: jax.Array
    h_i: jax.Array
    h_s: jax.Array
    v_i: jax.Array
    v_s: jax.Array
    t_su: jax.Array
    t_i: jax.Array
    e_i: jax.Array
    t_s: jax.Array
    e_s: jax.Array
    sz_i: jax.Array
    szv_i: jax.Array
    s_i: jax.Array
    sv_i: jax.Array
    oa_i: jax.Array
    dh_i_sum: jax.Array
    dh_s_sum: jax.Array


class SI3O12Stages(NamedTuple):
    """Composed ``DO jl = 1, jpl`` category exits followed by ``ice_thd_pnd``."""

    categories: tuple[SI3O12CategoryOutput, ...]
    pnd: SI3PondOutputs


def si3_ice_perm_eff(
    t_i,
    sz_i,
    *,
    constants: IceConstantsConfig,
) -> jax.Array:
    """``ice_perm_eff`` harmonic-mean brine permeability (``icethd_pnd.f90:1372-1422``).

    ``t_i`` and ``sz_i`` have shape ``(..., nlay_i)`` where ``nlay_i`` is the
    ice-layer axis.  Every statement is rounded in the instruction order of
    ``ORCA2_SAS_ICE_O7_REF/BLD/obj/icethd_pnd.o`` (``__icethd_pnd_MOD_ice_perm_eff``).
    """
    if t_i is None or sz_i is None or constants is None:
        raise ValueError("si3_ice_perm_eff requires t_i, sz_i, and constants; no defaults")
    t_i = jnp.asarray(t_i, dtype=jnp.float64)
    sz_i = jnp.asarray(sz_i, dtype=jnp.float64)
    if t_i.shape != sz_i.shape or t_i.ndim < 1:
        raise ValueError(
            f"si3_ice_perm_eff requires matching (..., nlay_i) arrays; got {t_i.shape} vs {sz_i.shape}"
        )
    sr = nemo_source_round
    nlay_i = t_i.shape[-1]
    rt0 = sr(jnp.float64(constants.T0))
    c1 = sr(jnp.float64(_PERM_VC19_C1))
    c2 = sr(jnp.float64(_PERM_VC19_C2))
    c3 = sr(jnp.float64(_PERM_VC19_C3))
    epsi10 = sr(jnp.float64(_EPSI10))
    epsi06 = sr(jnp.float64(_EPSI06))
    perm_scale = sr(jnp.float64(_PERM_SCALE))
    one = sr(jnp.float64(1.0))

    # :1389-1401 brine salinity and liquid fraction zv_br(1:nlay_i)
    zt1 = sr(t_i - rt0)
    zt2 = sr(zt1 * zt1)
    zt3 = sr(zt2 * zt1)
    # :1393 object code: mulsd 18.7, zt1; xorpd -0.0; subsd (0.519*zt2); subsd (0.00535*zt3)
    term1 = -sr(c1 * zt1)
    term2 = sr(c2 * zt2)
    term3 = sr(c3 * zt3)
    zs_br = sr(sr(term1 - term2) - term3)
    # :1398 maxsd(dst=zs_br, src=epsi10)
    denom = _maxsd(zs_br, epsi10)
    cold = zt1 < -epsi06
    zv_br = jnp.where(cold, sr(sz_i / denom), jnp.zeros_like(sz_i))

    # :1408-1420 np_perm_eff == 2 harmonic mean across jk = 1..nlay_i
    cell_shape = t_i.shape[:-1]
    zero = sr(jnp.zeros(cell_shape, dtype=jnp.float64))
    ztmp = zero
    stopped = jnp.zeros(cell_shape, dtype=bool)
    for jk in range(nlay_i):
        zv = zv_br[..., jk]
        low = zv < epsi06  # :1412 IF( zv_br(jk) < epsi06 ) THEN ztmp = 0._wp ; EXIT
        stopped = stopped | low
        # :1416 object code: ((3.e-8_wp * zv) * zv) * zv
        zperm = sr(sr(sr(perm_scale * zv) * zv) * zv)
        zperm_safe = jnp.where(stopped, one, zperm)
        inv_zperm = sr(one / zperm_safe)
        ztmp = jnp.where(stopped, zero, sr(ztmp + inv_zperm))

    # :1419 IF( ztmp > 0._wp ) ice_perm_eff = REAL( nlay_i, wp ) / ztmp
    valid = (~stopped) & _ieee_gt_zero(ztmp)
    ztmp_safe = jnp.where(valid, ztmp, one)
    nlay_f = sr(jnp.float64(float(nlay_i)))
    return jnp.where(valid, sr(nlay_f / ztmp_safe), zero)


def _pnd_lev_single_category(
    *,
    at_i: jax.Array,
    wfx_pnd: jax.Array,
    a_i: jax.Array,
    h_i: jax.Array,
    t_su: jax.Array,
    a_ip: jax.Array,
    h_ip: jax.Array,
    h_il: jax.Array,
    dh_i_sum: jax.Array,
    dh_s_sum: jax.Array,
    sz_i: jax.Array,
    t_i: jax.Array,
    rDt_ice: jax.Array,
    r1_Dt_ice: jax.Array,
    rn_himin: jax.Array,
    rn_apnd_min: jax.Array,
    rn_apnd_max: jax.Array,
    rn_pnd_flush: jax.Array,
    ln_pnd_lids: bool,
    rhow: jax.Array,
    grav: jax.Array,
    constants: IceConstantsConfig,
    z1_rhow: jax.Array,
    z1_aspect: jax.Array,
    z1_Tp: jax.Array,
) -> SI3PndLevOutputs:
    """One category ``jl`` of ``pnd_LEV`` (``icethd_pnd.f90:433-570``)."""
    sr = nemo_source_round
    zero = sr(jnp.zeros_like(a_i))
    one = sr(jnp.ones_like(a_i))
    zaspect = sr(jnp.float64(_PND_ZASPECT))
    zTp = sr(jnp.float64(_PND_ZTP_C))
    zvisc = sr(jnp.float64(_PND_ZVISC))
    tiny_ai = sr(jnp.float64(_PND_TINY_AI))
    depth_cap_factor = sr(jnp.float64(_PND_DEPTH_CAP_FACTOR))
    freeze_half = sr(jnp.float64(_PND_FREEZE_HALF))
    freeze_four = sr(jnp.float64(_PND_FREEZE_FOUR))
    no_lid_decay = sr(jnp.float64(_PND_NO_LID_DECAY))
    lid_ratio_max = sr(jnp.float64(_PND_LID_RATIO_MAX))

    rt0 = sr(jnp.float64(constants.T0))
    rhoi = sr(jnp.float64(constants.rho_ice))
    rhos = sr(jnp.float64(constants.rho_snow))
    rho0 = sr(jnp.float64(constants.rho_ocean))
    r1_rho0 = sr(jnp.float64(1.0) / rho0)
    rcnd_i = sr(jnp.float64(constants.k_ice))
    rLfus = sr(jnp.float64(constants.latent_fusion))

    # :435 zdv_pnd = ( h_ip_1d(ii) + h_il_1d(ii) ) * a_ip_1d(ii)
    zdv_pnd_old = sr(sr(h_ip + h_il) * a_ip)

    # :437 IF( h_i_1d(ii) < rn_himin .OR. a_i_1d(ii) < 0.01_wp ) THEN
    thin_or_tiny = (h_i < rn_himin) | (a_i < tiny_ai)
    h_i_safe = jnp.where(thin_or_tiny, one, h_i)
    a_i_safe = jnp.where(thin_or_tiny, one, a_i)

    # :446-447 retrieve volume from thickness
    v_ip = sr(h_ip * a_ip)
    v_il = sr(h_il * a_ip)

    # :454 zdv_avail = -( dh_i_sum(ii)*rhoi + dh_s_sum(ii)*rhos ) * z1_rhow * a_i_1d(ii)
    # Object code: ((dh_i_sum*rhoi + dh_s_sum*rhos) * z1_rhow) * a_i_1d, then xorpd sign bit
    melt_mass = sr(sr(dh_i_sum * rhoi) + sr(dh_s_sum * rhos))
    zdv_avail = -sr(sr(melt_mass * z1_rhow) * a_i)

    # :455 zfr_mlt = rn_apnd_min + ( rn_apnd_max - rn_apnd_min ) * at_i_1d(ii)
    zfr_mlt = sr(rn_apnd_min + sr(sr(rn_apnd_max - rn_apnd_min) * at_i))

    # :456 zdv_mlt = MAX( 0._wp, zfr_mlt * zdv_avail ); maxsd(dst=prod, src=ZERO)
    zdv_mlt = _maxsd(sr(zfr_mlt * zdv_avail), zero)

    # :464-465 area-driven overflow
    # Object code at :465: minsd(dst = zv_ip_max - v_ip, src = zdv_mlt), then maxsd(dst = min, src = ZERO)
    zv_ip_max_area = sr(sr(sr(zfr_mlt * zfr_mlt) * a_i) * zaspect)
    zdv_mlt = _maxsd(_minsd(sr(zv_ip_max_area - v_ip), zdv_mlt), zero)

    # :471-472 depth-driven overflow
    # Object code at :472: minsd(dst = zdv_mlt, src = zv_ip_max - v_ip), then maxsd(dst = min, src = ZERO)
    zv_ip_max_depth = sr(sr(sr(sr(z1_aspect * a_i) * depth_cap_factor) * h_i) * h_i)
    zdv_mlt = _maxsd(_minsd(zdv_mlt, sr(zv_ip_max_depth - v_ip)), zero)

    # :475 Pond growing
    v_ip = sr(v_ip + zdv_mlt)

    # :478 Lid melting
    if ln_pnd_lids:
        v_il = _maxsd(sr(v_il - zdv_mlt), zero)

    # :484 zdT = MAX( zTp+rt0 - t_su_1d(ii), 0._wp ); maxsd(dst=diff, src=ZERO)
    zdT = _maxsd(sr(sr(zTp + rt0) - t_su), zero)

    # :487-503 Pond contraction (due to refreezing)
    if ln_pnd_lids:
        # :490-491 object code:
        # p1 = a_ip*a_ip; p2 = p1*4.0; p3 = p2*rcnd_i; p4 = p3*zdT; p5 = p4*rDt_ice;
        # denom = rLfus*rhow; term = p5/denom; rad = v_il*v_il + term; root = sqrtsd(rad);
        # mx = maxsd(dst = root - v_il, src = ZERO); pos_frz = 0.5 * mx;
        # :494 v_il = maxsd(dst = v_il + pos_frz, src = ZERO)
        # :497 v_ip = maxsd(dst = v_ip - pos_frz, src = ZERO)
        p1 = sr(a_ip * a_ip)
        p2 = sr(p1 * freeze_four)
        p3 = sr(p2 * rcnd_i)
        v_il_sq = sr(v_il * v_il)
        p4 = sr(p3 * zdT)
        denom_frz = sr(rLfus * rhow)
        p5 = sr(p4 * rDt_ice)
        term_frz = sr(p5 / denom_frz)
        rad_frz = sr(v_il_sq + term_frz)
        root_frz = _nemo_sqrt(rad_frz)
        mx_frz = _maxsd(sr(root_frz - v_il), zero)
        pos_frz = sr(freeze_half * mx_frz)
        v_il = _maxsd(sr(v_il + pos_frz), zero)
        v_ip = _maxsd(sr(v_ip - pos_frz), zero)
    else:
        # :500-502 zdv_frz = v_ip_1d(ii) * ( EXP( 0.01_wp * zdT * z1_Tp ) - 1._wp )
        arg_exp = sr(sr(no_lid_decay * zdT) * z1_Tp)
        zdv_frz = sr(v_ip * sr(transcendentals.exp(arg_exp) - one))
        v_ip = _maxsd(sr(v_ip + zdv_frz), zero)

    # :508-509 Set new pond area and depth
    # Object code: arg = (z1_aspect * v_ip) * a_i; rt = sqrtsd(arg); minsd(dst=rt, src=a_i)
    rad_1 = sr(sr(v_ip * z1_aspect) * a_i)
    a_ip_1 = _minsd(_nemo_sqrt(rad_1), a_i)
    h_ip_1 = sr(sr(zaspect * a_ip_1) / a_i_safe)

    # :516 zhp = ( h_i_1d(ii) * ( rho0 - rhoi ) + h_ip_1d(ii) * ( rho0 - rhow * a_ip_1d(ii) / a_i_1d(ii) ) ) * r1_rho0
    t1_hp = sr(sr(rhow * a_ip_1) / a_i_safe)
    t2_hp = sr(h_i * sr(rho0 - rhoi))
    t3_hp = sr(h_ip_1 * sr(rho0 - t1_hp))
    zhp = sr(sr(t2_hp + t3_hp) * r1_rho0)

    # :519 zperm = ice_perm_eff( t_i_1d(ii,:), sz_i_1d(ii,:) )
    zperm = si3_ice_perm_eff(t_i, sz_i, constants=constants)

    # :522-524 Darcy flushing: gfortran -O3 computes positive pos_flush,
    # capped = minsd(dst=pos_flush, src=v_ip), and v_ip = v_ip - capped.
    q1 = sr(zperm * rho0)
    q2 = sr(q1 * grav)
    q3 = sr(q2 * zhp)
    q4 = sr(q3 * rDt_ice)
    denom_fl = sr(zvisc * h_i_safe)
    q5 = sr(q4 / denom_fl)
    q6 = sr(q5 * a_ip_1)
    pos_flush = sr(q6 * rn_pnd_flush)
    capped_flush = _minsd(pos_flush, v_ip)
    v_ip = sr(v_ip - capped_flush)

    # :527-528 Set new pond area and depth after flushing
    rad_2 = sr(sr(v_ip * z1_aspect) * a_i)
    a_ip_2 = _minsd(_nemo_sqrt(rad_2), a_i)
    h_ip_2 = sr(sr(zaspect * a_ip_2) / a_i_safe)

    # :531-542 Corrections and lid thickness
    if ln_pnd_lids:
        # :533-535 IF( a_ip_1d(ii) > 0.01_wp ) THEN h_il_1d = v_il_1d / a_ip_1d ELSE 0._wp
        a_ip_gt_001 = a_ip_2 > tiny_ai
        a_ip_div = jnp.where(a_ip_gt_001, a_ip_2, one)
        h_il_2 = jnp.where(a_ip_gt_001, sr(v_il / a_ip_div), zero)
        # :537-541 IF ( h_il_1d(ii) > h_ip_1d(ii) * 10._wp ) THEN zero a_ip, h_ip, h_il
        lid_too_thick = h_il_2 > sr(h_ip_2 * lid_ratio_max)
        a_ip_thick = jnp.where(lid_too_thick, zero, a_ip_2)
        h_ip_thick = jnp.where(lid_too_thick, zero, h_ip_2)
        h_il_thick = jnp.where(lid_too_thick, zero, h_il_2)
    else:
        a_ip_thick = a_ip_2
        h_ip_thick = h_ip_2
        h_il_thick = h_il

    # :437-442 vs :444-552
    a_ip_out = jnp.where(thin_or_tiny, zero, a_ip_thick)
    h_ip_out = jnp.where(thin_or_tiny, zero, h_ip_thick)
    h_il_out = jnp.where(thin_or_tiny, zero, h_il_thick)

    # :554-558 final volume recovery and freshwater flux update
    v_ip_out = sr(h_ip_out * a_ip_out)
    v_il_out = sr(h_il_out * a_ip_out)
    zdv_pnd = sr(sr(sr(h_ip_out + h_il_out) * a_ip_out) - zdv_pnd_old)
    wfx_pnd_out = sr(wfx_pnd - sr(sr(zdv_pnd * rhow) * r1_Dt_ice))

    return SI3PndLevOutputs(
        a_ip=a_ip_out,
        h_ip=h_ip_out,
        h_il=h_il_out,
        v_ip=v_ip_out,
        v_il=v_il_out,
        wfx_pnd=wfx_pnd_out,
    )


def si3_pnd_lev(
    *,
    at_i,
    wfx_pnd,
    a_i,
    h_i,
    t_su,
    a_ip,
    h_ip,
    h_il,
    dh_i_sum,
    dh_s_sum,
    sz_i,
    t_i,
    rDt_ice,
    r1_Dt_ice,
    rn_himin,
    rn_apnd_min,
    rn_apnd_max,
    rn_pnd_flush,
    ln_pnd_lids: bool,
    rho_w,
    gravity,
    constants: IceConstantsConfig,
) -> SI3PndLevOutputs:
    """``pnd_LEV`` level-ice melt pond evolution (``icethd_pnd.f90:329-582``).

    Accepts either single-category gathered arrays (``a_i`` of shape ``(cells...,)``,
    ``sz_i``/``t_i`` of shape ``(cells..., nlay_i)``) or multi-category arrays
    (``a_i`` of shape ``(cells..., jpl)``, ``sz_i``/``t_i`` of shape
    ``(cells..., nlay_i, jpl)``), executing ``DO jl = 1, jpl`` sequentially so
    ``wfx_pnd`` accumulates across categories in NEMO's exact loop order.
    """
    if not isinstance(ln_pnd_lids, bool):
        raise ValueError(f"ln_pnd_lids must be a bool; got {ln_pnd_lids!r}")
    if constants is None:
        raise ValueError("si3_pnd_lev requires constants; no default")
    req_arrays = {
        "at_i": at_i,
        "wfx_pnd": wfx_pnd,
        "a_i": a_i,
        "h_i": h_i,
        "t_su": t_su,
        "a_ip": a_ip,
        "h_ip": h_ip,
        "h_il": h_il,
        "dh_i_sum": dh_i_sum,
        "dh_s_sum": dh_s_sum,
        "sz_i": sz_i,
        "t_i": t_i,
    }
    arr = {k: _required(k, v) for k, v in req_arrays.items()}
    req_scalars = {
        "rDt_ice": rDt_ice,
        "r1_Dt_ice": r1_Dt_ice,
        "rn_himin": rn_himin,
        "rn_apnd_min": rn_apnd_min,
        "rn_apnd_max": rn_apnd_max,
        "rn_pnd_flush": rn_pnd_flush,
        "rho_w": rho_w,
        "gravity": gravity,
    }
    scl = {k: _required(k, v) for k, v in req_scalars.items()}

    sr = nemo_source_round
    one = sr(jnp.float64(1.0))
    # :401-403
    z1_rhow = sr(one / scl["rho_w"])
    z1_aspect = sr(one / sr(jnp.float64(_PND_ZASPECT)))
    z1_Tp = sr(one / sr(jnp.float64(_PND_ZTP_C)))

    if arr["a_i"].shape == arr["at_i"].shape:
        if arr["sz_i"].shape[:-1] != arr["a_i"].shape or arr["t_i"].shape != arr["sz_i"].shape:
            raise ValueError("Single-category si3_pnd_lev requires sz_i/t_i shape (cells..., nlay_i)")
        return _pnd_lev_single_category(
            **arr,
            rDt_ice=scl["rDt_ice"],
            r1_Dt_ice=scl["r1_Dt_ice"],
            rn_himin=scl["rn_himin"],
            rn_apnd_min=scl["rn_apnd_min"],
            rn_apnd_max=scl["rn_apnd_max"],
            rn_pnd_flush=scl["rn_pnd_flush"],
            ln_pnd_lids=ln_pnd_lids,
            rhow=scl["rho_w"],
            grav=scl["gravity"],
            constants=constants,
            z1_rhow=z1_rhow,
            z1_aspect=z1_aspect,
            z1_Tp=z1_Tp,
        )

    if (
        arr["a_i"].shape[:-1] != arr["at_i"].shape
        or arr["sz_i"].shape[:-2] != arr["at_i"].shape
        or arr["sz_i"].shape[-1] != arr["a_i"].shape[-1]
        or arr["t_i"].shape != arr["sz_i"].shape
    ):
        raise ValueError(
            "Multi-category si3_pnd_lev requires a_i shape (cells..., jpl) and sz_i/t_i shape (cells..., nlay_i, jpl)"
        )

    jpl = arr["a_i"].shape[-1]
    wfx = arr["wfx_pnd"]
    a_ip_list, h_ip_list, h_il_list, v_ip_list, v_il_list = [], [], [], [], []
    for jl in range(jpl):
        cat_out = _pnd_lev_single_category(
            at_i=arr["at_i"],
            wfx_pnd=wfx,
            a_i=arr["a_i"][..., jl],
            h_i=arr["h_i"][..., jl],
            t_su=arr["t_su"][..., jl],
            a_ip=arr["a_ip"][..., jl],
            h_ip=arr["h_ip"][..., jl],
            h_il=arr["h_il"][..., jl],
            dh_i_sum=arr["dh_i_sum"][..., jl],
            dh_s_sum=arr["dh_s_sum"][..., jl],
            sz_i=arr["sz_i"][..., :, jl],
            t_i=arr["t_i"][..., :, jl],
            rDt_ice=scl["rDt_ice"],
            r1_Dt_ice=scl["r1_Dt_ice"],
            rn_himin=scl["rn_himin"],
            rn_apnd_min=scl["rn_apnd_min"],
            rn_apnd_max=scl["rn_apnd_max"],
            rn_pnd_flush=scl["rn_pnd_flush"],
            ln_pnd_lids=ln_pnd_lids,
            rhow=scl["rho_w"],
            grav=scl["gravity"],
            constants=constants,
            z1_rhow=z1_rhow,
            z1_aspect=z1_aspect,
            z1_Tp=z1_Tp,
        )
        wfx = cat_out.wfx_pnd
        a_ip_list.append(cat_out.a_ip)
        h_ip_list.append(cat_out.h_ip)
        h_il_list.append(cat_out.h_il)
        v_ip_list.append(cat_out.v_ip)
        v_il_list.append(cat_out.v_il)

    return SI3PndLevOutputs(
        a_ip=jnp.stack(a_ip_list, axis=-1),
        h_ip=jnp.stack(h_ip_list, axis=-1),
        h_il=jnp.stack(h_il_list, axis=-1),
        v_ip=jnp.stack(v_ip_list, axis=-1),
        v_il=jnp.stack(v_il_list, axis=-1),
        wfx_pnd=wfx,
    )


def si3_ice_thd_pnd(
    *,
    a_i,
    v_i,
    h_i,
    t_su,
    a_ip,
    v_ip,
    h_ip,
    v_il,
    h_il,
    a_ip_frac,
    dh_i_sum_2d,
    dh_s_sum_2d,
    t_i,
    sz_i,
    wfx_pnd,
    namelist,
    rDt_ice,
    r1_Dt_ice,
    rho_w,
    gravity,
    nn_hls: int,
    constants: IceConstantsConfig,
) -> SI3PondOutputs:
    """``ice_thd_pnd`` production melt-pond stage (``icethd_pnd.f90:162-264``).

    Replays the master-record projection from Kind 10 (``pond_entry_category_done``)
    to Kind 11 (``pond_exit_itd_entry``) at ``jpl = 5, nlay_i = 10``.  Every
    physical input and namelist selector is required without defaults.
    """
    arrays = {
        "a_i": a_i,
        "v_i": v_i,
        "h_i": h_i,
        "t_su": t_su,
        "a_ip": a_ip,
        "v_ip": v_ip,
        "h_ip": h_ip,
        "v_il": v_il,
        "h_il": h_il,
        "a_ip_frac": a_ip_frac,
        "dh_i_sum_2d": dh_i_sum_2d,
        "dh_s_sum_2d": dh_s_sum_2d,
        "t_i": t_i,
        "sz_i": sz_i,
        "wfx_pnd": wfx_pnd,
    }
    arr = {k: _required(k, v) for k, v in arrays.items()}
    for name, value in (
        ("rDt_ice", rDt_ice),
        ("r1_Dt_ice", r1_Dt_ice),
        ("rho_w", rho_w),
        ("gravity", gravity),
    ):
        if value is None:
            raise ValueError(f"SI3 ice_thd_pnd requires {name!r}; no default")
    if constants is None:
        raise ValueError("SI3 ice_thd_pnd requires constants; no default")
    if namelist is None:
        raise ValueError("SI3 ice_thd_pnd requires the resolved namelist; no default")
    if type(nn_hls) is not int or nn_hls < 0:
        raise ValueError(f"SI3 ice_thd_pnd requires integer nn_hls >= 0; got {nn_hls!r}")

    nml = dict(namelist)
    missing = [name for name in _PND_NAMELIST if nml.get(name) is None]
    if missing:
        raise ValueError(f"SI3 ice_thd_pnd missing namelist values {missing}")

    # :1471-1477 ice_thd_pnd_init option resolution and production selector check
    expected = {
        "ln_icedH": True,
        "ln_pnd": True,
        "ln_pnd_LEV": True,
        "ln_pnd_CST": False,
        "ln_pnd_TOPO": False,
        "ln_pnd_lids": True,
        "ln_pnd_rain": True,
        "ln_pnd_alb": True,
        "ln_icediachk": False,
    }
    wrong = {name: (nml[name], val) for name, val in expected.items() if nml[name] is not val}
    if wrong:
        raise ValueError(f"SI3 ice_thd_pnd transcribes only the production selectors; got {wrong}")

    if arr["a_i"].ndim != 3:
        raise ValueError(f"SI3 ice_thd_pnd requires 3D (jpi, jpj, jpl) a_i; got {arr['a_i'].shape}")
    jpi, jpj, jpl = arr["a_i"].shape
    if jpl != 5:
        raise ValueError(f"SI3 O12 requires the production 5 ice categories (jpl=5); got {jpl}")
    if 2 * nn_hls >= jpi or 2 * nn_hls >= jpj:
        raise ValueError(f"nn_hls={nn_hls} leaves no interior in grid {(jpi, jpj)}")
    for k in ("v_i", "h_i", "t_su", "a_ip", "v_ip", "h_ip", "v_il", "h_il", "a_ip_frac", "dh_i_sum_2d", "dh_s_sum_2d"):
        if arr[k].shape != (jpi, jpj, jpl):
            raise ValueError(f"SI3 ice_thd_pnd requires {k} shape {(jpi, jpj, jpl)}; got {arr[k].shape}")
    if arr["t_i"].shape != (jpi, jpj, 10, jpl) or arr["sz_i"].shape != (jpi, jpj, 10, jpl):
        raise ValueError(
            f"SI3 O12 requires t_i and sz_i shape {(jpi, jpj, 10, jpl)} (nlay_i=10); got {arr['t_i'].shape}, {arr['sz_i'].shape}"
        )
    if arr["wfx_pnd"].shape != (jpi, jpj):
        raise ValueError(f"SI3 ice_thd_pnd requires wfx_pnd shape {(jpi, jpj)}; got {arr['wfx_pnd'].shape}")

    sr = nemo_source_round
    epsi10 = sr(jnp.float64(_EPSI10))
    rhow = _required("rho_w", rho_w)
    r1_dt = _required("r1_Dt_ice", r1_Dt_ice)

    # :203 at_i(:,:) = SUM( a_i, dim=3 ) over the full (jpi, jpj) domain
    at_i = fortran_sum0(arr["a_i"])

    interior = (slice(nn_hls, jpi - nn_hls), slice(nn_hls, jpj - nn_hls))
    at_int = at_i[interior]
    v_i_int = arr["v_i"][interior]
    a_ip_int = arr["a_ip"][interior]
    v_ip_int = arr["v_ip"][interior]
    v_il_int = arr["v_il"][interior]
    h_ip_int = arr["h_ip"][interior]
    h_il_int = arr["h_il"][interior]
    a_ip_frac_int = arr["a_ip_frac"][interior]
    wfx_int = arr["wfx_pnd"][interior]

    # :205-217 DO jl = 1, jpl; DO_2D(0,0,0,0) remove ponds where ice has vanished
    zero_2d = sr(jnp.zeros_like(at_int))
    a_ip_clean, v_ip_clean, v_il_clean, h_ip_clean, h_il_clean, a_ip_frac_clean = [], [], [], [], [], []
    for jl in range(jpl):
        vanished = (v_i_int[..., jl] < epsi10) | (at_int < epsi10)
        # :208 wfx_pnd(ji,jj) = wfx_pnd(ji,jj) + ( v_ip(ji,jj,jl) + v_il(ji,jj,jl) ) * rhow * r1_Dt_ice
        # Object code: ((v_ip + v_il) * rhow) * r1_Dt_ice + wfx_pnd
        flux_inc = sr(sr(sr(v_ip_int[..., jl] + v_il_int[..., jl]) * rhow) * r1_dt)
        wfx_int = jnp.where(vanished, sr(wfx_int + flux_inc), wfx_int)
        a_ip_clean.append(jnp.where(vanished, zero_2d, a_ip_int[..., jl]))
        v_ip_clean.append(jnp.where(vanished, zero_2d, v_ip_int[..., jl]))
        v_il_clean.append(jnp.where(vanished, zero_2d, v_il_int[..., jl]))
        h_ip_clean.append(jnp.where(vanished, zero_2d, h_ip_int[..., jl]))
        h_il_clean.append(jnp.where(vanished, zero_2d, h_il_int[..., jl]))
        a_ip_frac_clean.append(jnp.where(vanished, zero_2d, a_ip_frac_int[..., jl]))

    a_ip_int = jnp.stack(a_ip_clean, axis=-1)
    v_ip_int = jnp.stack(v_ip_clean, axis=-1)
    v_il_int = jnp.stack(v_il_clean, axis=-1)
    h_ip_int = jnp.stack(h_ip_clean, axis=-1)
    h_il_int = jnp.stack(h_il_clean, axis=-1)
    a_ip_frac_int = jnp.stack(a_ip_frac_clean, axis=-1)

    # :222-240 Gather interior cells with at_i(ji,jj) >= epsi10 and call pnd_LEV
    has_ice = at_int >= epsi10
    has_ice_3d = has_ice[..., None]
    lev_out = si3_pnd_lev(
        at_i=at_int,
        wfx_pnd=wfx_int,
        a_i=arr["a_i"][interior],
        h_i=arr["h_i"][interior],
        t_su=arr["t_su"][interior],
        a_ip=a_ip_int,
        h_ip=h_ip_int,
        h_il=h_il_int,
        dh_i_sum=arr["dh_i_sum_2d"][interior],
        dh_s_sum=arr["dh_s_sum_2d"][interior],
        sz_i=arr["sz_i"][interior],
        t_i=arr["t_i"][interior],
        rDt_ice=rDt_ice,
        r1_Dt_ice=r1_Dt_ice,
        rn_himin=nml["rn_himin"],
        rn_apnd_min=nml["rn_apnd_min"],
        rn_apnd_max=nml["rn_apnd_max"],
        rn_pnd_flush=nml["rn_pnd_flush"],
        ln_pnd_lids=nml["ln_pnd_lids"],
        rho_w=rho_w,
        gravity=gravity,
        constants=constants,
    )

    a_ip_int_out = jnp.where(has_ice_3d, lev_out.a_ip, a_ip_int)
    h_ip_int_out = jnp.where(has_ice_3d, lev_out.h_ip, h_ip_int)
    h_il_int_out = jnp.where(has_ice_3d, lev_out.h_il, h_il_int)
    v_ip_int_out = jnp.where(has_ice_3d, lev_out.v_ip, v_ip_int)
    v_il_int_out = jnp.where(has_ice_3d, lev_out.v_il, v_il_int)
    wfx_int_out = jnp.where(has_ice, lev_out.wfx_pnd, wfx_int)

    return SI3PondOutputs(
        at_i=at_i,
        a_ip=arr["a_ip"].at[interior].set(a_ip_int_out),
        v_ip=arr["v_ip"].at[interior].set(v_ip_int_out),
        h_ip=arr["h_ip"].at[interior].set(h_ip_int_out),
        v_il=arr["v_il"].at[interior].set(v_il_int_out),
        h_il=arr["h_il"].at[interior].set(h_il_int_out),
        a_ip_frac=arr["a_ip_frac"].at[interior].set(a_ip_frac_int),
        wfx_pnd=arr["wfx_pnd"].at[interior].set(wfx_int_out),
    )


def si3_ice_thd_o12_stages(
    *,
    category_entries: Sequence[dict],
    category_fluxes: Sequence[SI3DHFluxes],
    sfx_bri,
    sfx_lam,
    hfx_thd,
    wfx_lam,
    a_ip,
    v_ip,
    h_ip,
    v_il,
    h_il,
    a_ip_frac,
    wfx_pnd,
    namelist: dict,
    rDt_ice,
    r1_Dt_ice,
    r1_nlay_i,
    r1_nlay_s,
    rho_i,
    rho_s,
    rho_w,
    gravity,
    nn_hls: int,
    constants: IceConstantsConfig,
) -> SI3O12Stages:
    """Selectable O9 + O10 + O11 + ``ice_thd_1d2d(jl, 2)`` + O12 pond path (``icethd.f90:176-223``).

    Executes the ``jpl = 5`` category loop (``si3_ice_thd_o10_category_stages``
    -> ``si3_ice_thd_da`` when ``ln_icedA`` -> ``ice_thd_1d2d(jl, 2)`` extensive
    volume/content recovery ``icethd.f90:488-501``), packs the 5 category exits
    into ``(jpi, jpj, jpl)`` / ``(jpi, jpj, nlay_i, jpl)`` arrays, and calls
    :func:`si3_ice_thd_pnd` (``icethd.f90:222-223``).
    """
    if namelist is None:
        raise ValueError("si3_ice_thd_o12_stages requires the resolved namelist; no default")
    nml = dict(namelist)
    if len(category_entries) != 5 or len(category_fluxes) != 5:
        raise ValueError("SI3 O12 requires 5 category entries and flux tuples (jpl=5)")
    ln_iceda = nml.get("ln_icedA")
    if not isinstance(ln_iceda, bool):
        raise ValueError(f"si3_ice_thd_o12_stages requires bool ln_icedA in namelist; got {ln_iceda!r}")
    r1_nlay_s_arr = _required("r1_nlay_s", r1_nlay_s)
    r1_nlay_i_arr = _required("r1_nlay_i", r1_nlay_i)
    sr = nemo_source_round

    cat_outputs: list[SI3O12CategoryOutput] = []
    cur_sfx_bri = _required("sfx_bri", sfx_bri)
    cur_sfx_lam = _required("sfx_lam", sfx_lam)
    cur_hfx_thd = _required("hfx_thd", hfx_thd)
    cur_wfx_lam = _required("wfx_lam", wfx_lam)

    for jl in range(5):
        entry = category_entries[jl]
        fluxes = category_fluxes[jl]
        o10 = si3_ice_thd_o10_category_stages(
            entry=entry,
            fluxes=fluxes,
            sfx_bri=cur_sfx_bri,
            namelist=nml,
            rDt_ice=rDt_ice,
            r1_Dt_ice=r1_Dt_ice,
            r1_nlay_i=r1_nlay_i,
            rho_i=rho_i,
            rho_s=rho_s,
            rho_w=rho_w,
            gravity=gravity,
            constants=constants,
        )
        cur_sfx_bri = o10.sal.sfx_bri
        dh = o10.o9.dh
        a_i_cat = dh.a_i
        h_i_cat = dh.h_i
        h_s_cat = dh.h_s

        if ln_iceda:
            da_out = si3_ice_thd_da(
                at_i=entry["at_i"],
                a_i=a_i_cat,
                h_i=h_i_cat,
                h_s=h_s_cat,
                e_i=dh.e_i,
                e_s=dh.e_s,
                s_i=dh.s_i,
                sz_i=o10.sal.sz_i,
                sst=entry["sst"],
                t_bo=entry["t_bo"],
                sfx_lam=cur_sfx_lam,
                hfx_thd=cur_hfx_thd,
                wfx_lam=cur_wfx_lam,
                rn_beta=nml.get("rn_beta"),
                rn_dmin=nml.get("rn_dmin"),
                rDt_ice=rDt_ice,
                nn_icesal=nml.get("nn_icesal"),
                constants=constants,
            )
            a_i_cat = da_out.a_i
            h_i_cat = da_out.h_i
            h_s_cat = da_out.h_s
            cur_sfx_lam = da_out.sfx_lam
            cur_hfx_thd = da_out.hfx_thd
            cur_wfx_lam = da_out.wfx_lam
        else:
            da_out = None

        # icethd.f90:488-501 ice_thd_1d2d(jl, 2) unit & extensive conversions:
        # e_i_1d(jk) = ((e_i_1d(jk) * h_i_1d) * a_i_1d) * r1_nlay_i
        # e_s_1d(jk) = ((e_s_1d(jk) * h_s_1d) * a_i_1d) * r1_nlay_s
        # v_i_1d = h_i_1d * a_i_1d; v_s_1d = h_s_1d * a_i_1d;
        # sv_i_1d = s_i_1d * v_i_1d; oa_i_1d = o_i_1d * a_i_1d;
        # szv_i_1d(jk) = (sz_i_1d(jk) * v_i_1d) * r1_nlay_i
        e_i_2d = sr(sr(sr(dh.e_i * h_i_cat[..., None]) * a_i_cat[..., None]) * r1_nlay_i_arr)
        e_s_2d = sr(sr(sr(dh.e_s * h_s_cat[..., None]) * a_i_cat[..., None]) * r1_nlay_s_arr)
        v_i_cat = sr(h_i_cat * a_i_cat)
        v_s_cat = sr(h_s_cat * a_i_cat)
        sv_i_cat = sr(dh.s_i * v_i_cat)
        o_i_entry = _required("o_i", entry.get("o_i"))
        oa_i_cat = sr(o_i_entry * a_i_cat)
        szv_i_cat = sr(sr(o10.sal.sz_i * v_i_cat[..., None]) * r1_nlay_i_arr)

        cat_outputs.append(
            SI3O12CategoryOutput(
                o10=o10,
                da=da_out,
                a_i=a_i_cat,
                h_i=h_i_cat,
                h_s=h_s_cat,
                v_i=v_i_cat,
                v_s=v_s_cat,
                t_su=dh.t_su,
                t_i=o10.t_i,
                e_i=e_i_2d,
                t_s=dh.t_s,
                e_s=e_s_2d,
                sz_i=o10.sal.sz_i,
                szv_i=szv_i_cat,
                s_i=dh.s_i,
                sv_i=sv_i_cat,
                oa_i=oa_i_cat,
                dh_i_sum=dh.dh.dh_i_sum,
                dh_s_sum=dh.dh.dh_s_sum,
            )
        )

    def _pack_3d(field_name: str) -> jax.Array:
        vals = [getattr(c, field_name) for c in cat_outputs]
        stacked = jnp.stack(vals, axis=-1)
        return stacked if stacked.ndim == 3 else stacked.reshape((-1, 1, 5))

    def _pack_4d(field_name: str) -> jax.Array:
        vals = [getattr(c, field_name) for c in cat_outputs]
        stacked = jnp.stack(vals, axis=-1)
        return stacked if stacked.ndim == 4 else stacked.reshape((-1, 1, stacked.shape[-2], 5))

    a_i_3d = _pack_3d("a_i")
    v_i_3d = _pack_3d("v_i")
    h_i_3d = _pack_3d("h_i")
    t_su_3d = _pack_3d("t_su")
    dh_i_sum_3d = _pack_3d("dh_i_sum")
    dh_s_sum_3d = _pack_3d("dh_s_sum")
    t_i_4d = _pack_4d("t_i")
    sz_i_4d = _pack_4d("sz_i")

    pnd_out = si3_ice_thd_pnd(
        a_i=a_i_3d,
        v_i=v_i_3d,
        h_i=h_i_3d,
        t_su=t_su_3d,
        a_ip=a_ip,
        v_ip=v_ip,
        h_ip=h_ip,
        v_il=v_il,
        h_il=h_il,
        a_ip_frac=a_ip_frac,
        dh_i_sum_2d=dh_i_sum_3d,
        dh_s_sum_2d=dh_s_sum_3d,
        t_i=t_i_4d,
        sz_i=sz_i_4d,
        wfx_pnd=wfx_pnd,
        namelist=nml,
        rDt_ice=rDt_ice,
        r1_Dt_ice=r1_Dt_ice,
        rho_w=rho_w,
        gravity=gravity,
        nn_hls=nn_hls,
        constants=constants,
    )
    return SI3O12Stages(categories=tuple(cat_outputs), pnd=pnd_out)


__all__ = (
    "SI3PndLevOutputs",
    "SI3PondOutputs",
    "SI3O12CategoryOutput",
    "SI3O12Stages",
    "si3_ice_perm_eff",
    "si3_pnd_lev",
    "si3_ice_thd_pnd",
    "si3_ice_thd_o12_stages",
)
