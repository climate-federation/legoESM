"""Independent scalar checks for SI3 O12 melt-pond stage (ice_thd_pnd / pnd_LEV / ice_perm_eff)."""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import hashlib
import inspect
import math
import textwrap
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants as canonical_constants
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice import si3_thd_da as da_mod
from legoesm.ice import si3_thd_pnd as pnd_mod
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
from tests.ice.unit import test_si3_o9_zdf_dh_temp as o9_fixture
from tests.ice.unit import test_si3_o10_salinity as o10_fixture


@contextlib.contextmanager
def precision_scope(policy: PrecisionPolicy):
    prev = get_policy()
    set_policy(policy)
    try:
        yield
    finally:
        set_policy(prev)


_LIBM = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
_LIBM_EXP = _LIBM.exp
_LIBM_EXP.argtypes = (ctypes.c_double,)
_LIBM_EXP.restype = ctypes.c_double

RDT = 10800.0
R1_RDT = 1.0 / RDT
R1_NLAY_I = 1.0 / 10.0
R1_NLAY_S = 1.0 / 5.0
RHO_I = 917.0
RHO_S = 330.0
RHO_W = 1000.0
RHO_0 = 1026.0
GRAVITY = canonical_constants.g_nemo

DECK_PND_NML = {
    "ln_icedH": True,
    "rn_himin": 0.1,
    "ln_pnd": True,
    "ln_pnd_LEV": True,
    "rn_apnd_min": 0.15,
    "rn_apnd_max": 0.80,
    "rn_pnd_flush": 0.1,
    "ln_pnd_lids": True,
    "ln_pnd_rain": True,
    "ln_pnd_CST": False,
    "rn_apnd": 0.2,
    "rn_hpnd": 0.05,
    "ln_pnd_TOPO": False,
    "ln_pnd_alb": True,
    "rn_pnd_hl_min": 0.005,
    "rn_pnd_hl_max": 0.015,
    "ln_icediachk": False,
}


def _fmax(dst: float, src: float) -> float:
    """x86 ``maxsd src, dst`` tie survivor, independent of JAX/model code."""
    return dst if dst > src else src


def _fmin(dst: float, src: float) -> float:
    """x86 ``minsd src, dst`` tie survivor, independent of JAX/model code."""
    return dst if dst < src else src


def _bits(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).view(np.uint64)


def _oracle_ice_perm_eff(
    t_i: list[float],
    sz_i: list[float],
    *,
    T0: float,
    events: dict[str, int] | None = None,
) -> float:
    """Independent scalar translation of ``ice_perm_eff`` (``icethd_pnd.f90:1372-1422``)."""
    nlay_i = len(t_i)
    assert len(sz_i) == nlay_i
    zv_br = []
    for jk in range(nlay_i):
        zt1 = t_i[jk] - T0
        zt2 = zt1 * zt1
        zt3 = zt2 * zt1
        zs_br = ((-18.7 * zt1) - (0.519 * zt2)) - (0.00535 * zt3)
        if events is not None and zs_br < 1.0e-10:
            events["perm_zs_br_floor"] += 1
        zs_br = _fmax(zs_br, 1.0e-10)
        if zt1 < -1.0e-6:
            zv_br.append(sz_i[jk] / zs_br)
        else:
            if events is not None:
                events["perm_warm_layer_zero"] += 1
            zv_br.append(0.0)

    zsum = 0.0
    for jk in range(nlay_i):
        if zv_br[jk] < 1.0e-6:
            if events is not None:
                events["perm_impermeable_exit"] += 1
            zsum = 0.0
            break
        zk_br = ((3.0e-8 * zv_br[jk]) * zv_br[jk]) * zv_br[jk]
        zsum = zsum + (1.0 / zk_br)

    if zsum > 0.0:
        if events is not None:
            events["perm_positive_return"] += 1
        return float(nlay_i) / zsum
    return 0.0


def _oracle_pnd_lev_category(
    *,
    at_i: float,
    a_i: float,
    h_i: float,
    t_su: float,
    a_ip: float,
    h_ip: float,
    h_il: float,
    dh_i_sum: float,
    dh_s_sum: float,
    sz_i: list[float],
    t_i: list[float],
    wfx_pnd: float,
    rDt_ice: float,
    r1_Dt_ice: float,
    rn_himin: float,
    rn_apnd_min: float,
    rn_apnd_max: float,
    rn_pnd_flush: float,
    ln_pnd_lids: bool,
    rho_w: float,
    gravity: float,
    constants,
    events: dict[str, int] | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Independent scalar translation of one category in ``pnd_LEV`` (``icethd_pnd.f90:384-559``)."""
    zaspect = 0.8
    zTp = -0.15
    zvisc = 1.79e-3
    rt0 = float(constants.T0)
    rho0 = float(constants.rho_ocean)
    rhoi = float(constants.rho_ice)
    rhos = float(constants.rho_snow)
    rhow = float(rho_w)
    grav = float(gravity)
    rLfus = float(constants.latent_fusion)
    rcnd_i = float(constants.k_ice)
    r1_rho0 = 1.0 / rho0

    z1_Tp = 1.0 / zTp
    z1_aspect = 1.0 / zaspect
    z1_rhow = 1.0 / rhow

    zdv_pnd_old = (h_ip + h_il) * a_ip
    v_ip = h_ip * a_ip
    v_il = h_il * a_ip

    if h_i < rn_himin or a_i < 0.01:
        if events is not None:
            if h_i < rn_himin:
                events["lev_thin_ice_removed"] += 1
            if a_i < 0.01:
                events["lev_tiny_ai_removed"] += 1
        a_ip_out = 0.0
        h_ip_out = 0.0
        h_il_out = 0.0
    else:
        melt_mass = (dh_i_sum * rhoi) + (dh_s_sum * rhos)
        zdv_avail = -((melt_mass * z1_rhow) * a_i)
        zfr_mlt = rn_apnd_min + ((rn_apnd_max - rn_apnd_min) * at_i)
        zdv_mlt = _fmax(zfr_mlt * zdv_avail, 0.0)
        if events is not None and zdv_mlt > 0.0:
            events["lev_melt_growth"] += 1

        zv_ip_max_area = ((zfr_mlt * zfr_mlt) * a_i) * zaspect
        diff_area = zv_ip_max_area - v_ip
        if events is not None and zdv_mlt > diff_area:
            events["lev_area_cap_bound"] += 1
        # Object code at line 465: minsd(dst=zv_ip_max - v_ip_1d, src=zdv_mlt), then maxsd(..., src=0)
        zdv_mlt = _fmax(_fmin(diff_area, zdv_mlt), 0.0)

        zv_ip_max_depth = (((z1_aspect * a_i) * 0.25) * h_i) * h_i
        diff_depth = zv_ip_max_depth - v_ip
        if events is not None and zdv_mlt > diff_depth:
            events["lev_depth_cap_bound"] += 1
        # Object code at line 472: minsd(dst=zdv_mlt, src=zv_ip_max - v_ip_1d), then maxsd(..., src=0)
        zdv_mlt = _fmax(_fmin(zdv_mlt, diff_depth), 0.0)

        v_ip = v_ip + zdv_mlt

        if ln_pnd_lids:
            if events is not None and v_il > 0.0 and zdv_mlt > 0.0:
                events["lev_lid_melted"] += 1
            v_il = _fmax(v_il - zdv_mlt, 0.0)

        zdT = _fmax((zTp + rt0) - t_su, 0.0)
        if ln_pnd_lids:
            p1 = a_ip * a_ip
            p2 = p1 * 4.0
            p3 = p2 * rcnd_i
            v_il_sq = v_il * v_il
            p4 = p3 * zdT
            denom_frz = rLfus * rhow
            p5 = p4 * rDt_ice
            term_frz = p5 / denom_frz
            rad_frz = v_il_sq + term_frz
            root = math.sqrt(rad_frz)
            pos_frz = 0.5 * _fmax(root - v_il, 0.0)
            if events is not None and pos_frz > 0.0:
                events["lev_lid_freeze_growth"] += 1
                if pos_frz >= v_ip:
                    events["lev_lid_freeze_exhausts_pond"] += 1
            v_il = _fmax(v_il + pos_frz, 0.0)
            v_ip = _fmax(v_ip - pos_frz, 0.0)
        else:
            arg_exp = (0.01 * zdT) * z1_Tp
            zdv_frz = v_ip * (float(_LIBM_EXP(arg_exp)) - 1.0)
            if events is not None and zdv_frz < 0.0:
                events["lev_nolid_exp_shrink"] += 1
            v_ip = _fmax(v_ip + zdv_frz, 0.0)

        rad_1 = (v_ip * z1_aspect) * a_i
        rt_1 = math.sqrt(rad_1)
        if events is not None and rt_1 > a_i:
            events["lev_pre_flush_aip_capped"] += 1
        a_ip_1 = _fmin(rt_1, a_i)
        h_ip_1 = (zaspect * a_ip_1) / a_i

        t1_hp = (rhow * a_ip_1) / a_i
        t2_hp = h_i * (rho0 - rhoi)
        t3_hp = h_ip_1 * (rho0 - t1_hp)
        zhp = (t2_hp + t3_hp) * r1_rho0

        zperm = _oracle_ice_perm_eff(t_i, sz_i, T0=rt0, events=events)

        q1 = zperm * rho0
        q2 = q1 * grav
        q3 = q2 * zhp
        q4 = q3 * rDt_ice
        denom_fl = zvisc * h_i
        q5 = q4 / denom_fl
        q6 = q5 * a_ip_1
        pos_flush = q6 * rn_pnd_flush
        if events is not None and pos_flush > 0.0:
            if pos_flush >= v_ip and v_ip > 0.0:
                events["lev_flush_exhausts_pond"] += 1
            elif v_ip > 0.0:
                events["lev_flush_partial"] += 1
        capped_flush = _fmin(pos_flush, v_ip)
        v_ip = v_ip - capped_flush

        rad_2 = (v_ip * z1_aspect) * a_i
        rt_2 = math.sqrt(rad_2)
        if events is not None and rt_2 > a_i:
            events["lev_post_flush_aip_capped"] += 1
        a_ip_2 = _fmin(rt_2, a_i)
        h_ip_2 = (zaspect * a_ip_2) / a_i

        if ln_pnd_lids:
            if a_ip_2 > 0.01:
                h_il_2 = v_il / a_ip_2
                if events is not None:
                    events["lev_lid_thickness_recovered"] += 1
            else:
                h_il_2 = 0.0
                if events is not None and v_il > 0.0:
                    events["lev_lid_small_aip_zeroed"] += 1
            if h_il_2 > (h_ip_2 * 10.0):
                if events is not None:
                    events["lev_lid_ratio_removed"] += 1
                a_ip_out = 0.0
                h_ip_out = 0.0
                h_il_out = 0.0
            else:
                a_ip_out = a_ip_2
                h_ip_out = h_ip_2
                h_il_out = h_il_2
        else:
            a_ip_out = a_ip_2
            h_ip_out = h_ip_2
            h_il_out = h_il

    v_ip_out = h_ip_out * a_ip_out
    v_il_out = h_il_out * a_ip_out
    zdv_pnd = ((h_ip_out + h_il_out) * a_ip_out) - zdv_pnd_old
    wfx_pnd_out = wfx_pnd - ((zdv_pnd * rhow) * r1_Dt_ice)
    return a_ip_out, h_ip_out, h_il_out, v_ip_out, v_il_out, wfx_pnd_out


def _oracle_pnd_lev(
    *,
    at_i: np.ndarray,
    wfx_pnd: np.ndarray,
    a_i: np.ndarray,
    h_i: np.ndarray,
    t_su: np.ndarray,
    a_ip: np.ndarray,
    h_ip: np.ndarray,
    h_il: np.ndarray,
    dh_i_sum: np.ndarray,
    dh_s_sum: np.ndarray,
    sz_i: np.ndarray,
    t_i: np.ndarray,
    rDt_ice: float,
    r1_Dt_ice: float,
    rn_himin: float,
    rn_apnd_min: float,
    rn_apnd_max: float,
    rn_pnd_flush: float,
    ln_pnd_lids: bool,
    rho_w: float,
    gravity: float,
    constants,
    events: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent scalar translation of multi-category ``pnd_LEV`` (``icethd_pnd.f90:329-582``)."""
    n_cells, jpl = a_i.shape
    a_ip_out = np.array(a_ip, dtype=np.float64, copy=True)
    h_ip_out = np.array(h_ip, dtype=np.float64, copy=True)
    h_il_out = np.array(h_il, dtype=np.float64, copy=True)
    v_ip_out = np.zeros_like(a_ip_out)
    v_il_out = np.zeros_like(a_ip_out)
    wfx_out = np.array(wfx_pnd, dtype=np.float64, copy=True)

    for jl in range(jpl):
        for idx in range(n_cells):
            res = _oracle_pnd_lev_category(
                at_i=float(at_i[idx]),
                a_i=float(a_i[idx, jl]),
                h_i=float(h_i[idx, jl]),
                t_su=float(t_su[idx, jl]),
                a_ip=float(a_ip[idx, jl]),
                h_ip=float(h_ip[idx, jl]),
                h_il=float(h_il[idx, jl]),
                dh_i_sum=float(dh_i_sum[idx, jl]),
                dh_s_sum=float(dh_s_sum[idx, jl]),
                sz_i=[float(x) for x in sz_i[idx, :, jl]],
                t_i=[float(x) for x in t_i[idx, :, jl]],
                wfx_pnd=float(wfx_out[idx]),
                rDt_ice=rDt_ice,
                r1_Dt_ice=r1_Dt_ice,
                rn_himin=rn_himin,
                rn_apnd_min=rn_apnd_min,
                rn_apnd_max=rn_apnd_max,
                rn_pnd_flush=rn_pnd_flush,
                ln_pnd_lids=ln_pnd_lids,
                rho_w=rho_w,
                gravity=gravity,
                constants=constants,
                events=events,
            )
            a_ip_out[idx, jl] = res[0]
            h_ip_out[idx, jl] = res[1]
            h_il_out[idx, jl] = res[2]
            v_ip_out[idx, jl] = res[3]
            v_il_out[idx, jl] = res[4]
            wfx_out[idx] = res[5]
    return a_ip_out, h_ip_out, h_il_out, v_ip_out, v_il_out, wfx_out


def _oracle_ice_thd_pnd(
    *,
    a_i: np.ndarray,
    v_i: np.ndarray,
    h_i: np.ndarray,
    t_su: np.ndarray,
    a_ip: np.ndarray,
    v_ip: np.ndarray,
    h_ip: np.ndarray,
    v_il: np.ndarray,
    h_il: np.ndarray,
    a_ip_frac: np.ndarray,
    dh_i_sum_2d: np.ndarray,
    dh_s_sum_2d: np.ndarray,
    t_i: np.ndarray,
    sz_i: np.ndarray,
    wfx_pnd: np.ndarray,
    namelist: dict,
    rDt_ice: float,
    r1_Dt_ice: float,
    rho_w: float,
    gravity: float,
    constants,
    nn_hls: int = 2,
    events: dict[str, int] | None = None,
) -> tuple[np.ndarray, ...]:
    """Independent scalar translation of ``ice_thd_pnd`` (``icethd_pnd.f90:162-264``)."""
    jpj, jpi, jpl = a_i.shape
    a_ip_w = np.array(a_ip, dtype=np.float64, copy=True)
    v_ip_w = np.array(v_ip, dtype=np.float64, copy=True)
    h_ip_w = np.array(h_ip, dtype=np.float64, copy=True)
    v_il_w = np.array(v_il, dtype=np.float64, copy=True)
    h_il_w = np.array(h_il, dtype=np.float64, copy=True)
    a_ip_frac_w = np.array(a_ip_frac, dtype=np.float64, copy=True)
    wfx_w = np.array(wfx_pnd, dtype=np.float64, copy=True)

    # :203 at_i(:,:) = SUM( a_i, dim=3 )
    at_i_w = np.zeros((jpj, jpi), dtype=np.float64)
    for jj in range(jpj):
        for ji in range(jpi):
            acc = 0.0
            for jl in range(jpl):
                acc = acc + float(a_i[jj, ji, jl])
            at_i_w[jj, ji] = acc

    # :205-217 DO jl = 1, jpl / DO_2D( 0, 0, 0, 0 )
    for jl in range(jpl):
        for jj in range(nn_hls, jpj - nn_hls):
            for ji in range(nn_hls, jpi - nn_hls):
                if float(v_i[jj, ji, jl]) < 1.0e-10 or float(at_i_w[jj, ji]) < 1.0e-10:
                    if events is not None and (v_ip_w[jj, ji, jl] > 0.0 or v_il_w[jj, ji, jl] > 0.0):
                        events["pnd_vanished_ice_conserved"] += 1
                    dv = float(v_ip_w[jj, ji, jl]) + float(v_il_w[jj, ji, jl])
                    flux_inc = (dv * float(rho_w)) * float(r1_Dt_ice)
                    wfx_w[jj, ji] = float(wfx_w[jj, ji]) + flux_inc
                    a_ip_w[jj, ji, jl] = 0.0
                    v_ip_w[jj, ji, jl] = 0.0
                    v_il_w[jj, ji, jl] = 0.0
                    h_ip_w[jj, ji, jl] = 0.0
                    h_il_w[jj, ji, jl] = 0.0
                    a_ip_frac_w[jj, ji, jl] = 0.0

    # :222-240 Gather active interior cells (at_i >= epsi10) and call pnd_LEV (:565-573 tab_1d_2d)
    for jl in range(jpl):
        for jj in range(nn_hls, jpj - nn_hls):
            for ji in range(nn_hls, jpi - nn_hls):
                if float(at_i_w[jj, ji]) >= 1.0e-10:
                    res = _oracle_pnd_lev_category(
                        at_i=float(at_i_w[jj, ji]),
                        a_i=float(a_i[jj, ji, jl]),
                        h_i=float(h_i[jj, ji, jl]),
                        t_su=float(t_su[jj, ji, jl]),
                        a_ip=float(a_ip_w[jj, ji, jl]),
                        h_ip=float(h_ip_w[jj, ji, jl]),
                        h_il=float(h_il_w[jj, ji, jl]),
                        dh_i_sum=float(dh_i_sum_2d[jj, ji, jl]),
                        dh_s_sum=float(dh_s_sum_2d[jj, ji, jl]),
                        sz_i=[float(x) for x in sz_i[jj, ji, :, jl]],
                        t_i=[float(x) for x in t_i[jj, ji, :, jl]],
                        wfx_pnd=float(wfx_w[jj, ji]),
                        rDt_ice=rDt_ice,
                        r1_Dt_ice=r1_Dt_ice,
                        rn_himin=float(namelist["rn_himin"]),
                        rn_apnd_min=float(namelist["rn_apnd_min"]),
                        rn_apnd_max=float(namelist["rn_apnd_max"]),
                        rn_pnd_flush=float(namelist["rn_pnd_flush"]),
                        ln_pnd_lids=bool(namelist["ln_pnd_lids"]),
                        rho_w=rho_w,
                        gravity=gravity,
                        constants=constants,
                        events=events,
                    )
                    a_ip_w[jj, ji, jl] = res[0]
                    h_ip_w[jj, ji, jl] = res[1]
                    h_il_w[jj, ji, jl] = res[2]
                    v_ip_w[jj, ji, jl] = res[3]
                    v_il_w[jj, ji, jl] = res[4]
                    wfx_w[jj, ji] = res[5]
                else:
                    if events is not None and jl == 0:
                        events["lev_skipped_inactive_cell"] += 1

    return at_i_w, a_ip_w, v_ip_w, h_ip_w, v_il_w, h_il_w, a_ip_frac_w, wfx_w


def _pnd_2d_fixture() -> dict[str, np.ndarray]:
    """2D/3D (jpj=6, jpi=7, jpl=5, nlay_i=10, nn_hls=2) fixture covering every O12 branch."""
    T0 = float(NEMO_SI3_CONSTANTS_CONFIG.T0)
    jpj, jpi, jpl, nlay_i = 6, 7, 5, 10
    rng = np.random.default_rng(20261012)

    # Start with non-trivial halo values to verify halo invariance (nn_hls = 2)
    a_i = rng.uniform(0.08, 0.25, (jpj, jpi, jpl))
    h_i = rng.uniform(0.5, 2.5, (jpj, jpi, jpl))
    v_i = a_i * h_i
    t_su = T0 - rng.uniform(0.05, 6.0, (jpj, jpi, jpl))
    dh_i_sum_2d = -rng.uniform(0.0, 0.03, (jpj, jpi, jpl))
    dh_s_sum_2d = -rng.uniform(0.0, 0.015, (jpj, jpi, jpl))
    sz_i = rng.uniform(1.5, 3.0, (jpj, jpi, nlay_i, jpl))
    t_i = T0 - rng.uniform(2.0, 4.0, (jpj, jpi, nlay_i, jpl))
    a_ip = rng.uniform(0.04, 0.12, (jpj, jpi, jpl))
    h_ip = rng.uniform(0.06, 0.16, (jpj, jpi, jpl))
    h_il = rng.uniform(0.0, 0.025, (jpj, jpi, jpl))
    v_ip = h_ip * a_ip
    v_il = h_il * a_ip
    a_ip_frac = a_ip / np.maximum(a_i, 1.0e-10)
    wfx_pnd = rng.uniform(-1.0e-5, 1.0e-5, (jpj, jpi))

    # Interior cells are (jj in {2, 3}, ji in {2, 3, 4}) -> 6 interior cells x 5 categories = 30 category columns:
    # Cell (2, 2): completely ice-free cell (at_i < epsi10) with leftover ponds -> vanished-ice conservation cleanup
    a_i[2, 2, :] = 0.0
    h_i[2, 2, :] = 0.0
    v_i[2, 2, :] = 0.0
    a_ip[2, 2, :] = [0.05, 0.02, 0.0, 0.03, 0.01]
    h_ip[2, 2, :] = [0.10, 0.08, 0.0, 0.06, 0.04]
    h_il[2, 2, :] = [0.02, 0.01, 0.0, 0.005, 0.0]
    v_ip[2, 2, :] = h_ip[2, 2, :] * a_ip[2, 2, :]
    v_il[2, 2, :] = h_il[2, 2, :] * a_ip[2, 2, :]

    # Cell (2, 3): active cell with mixed category edge cases:
    # jl=0: vanished v_i < epsi10 on an active cell (cleaned in :207-215, then thin/tiny in pnd_LEV)
    a_i[2, 3, 0] = 0.0
    h_i[2, 3, 0] = 0.0
    v_i[2, 3, 0] = 0.0
    a_ip[2, 3, 0] = 0.04
    h_ip[2, 3, 0] = 0.08
    h_il[2, 3, 0] = 0.01
    v_ip[2, 3, 0] = h_ip[2, 3, 0] * a_ip[2, 3, 0]
    v_il[2, 3, 0] = h_il[2, 3, 0] * a_ip[2, 3, 0]
    # jl=1: h_i < rn_himin (0.05 < 0.1) with active pond -> removed in pnd_LEV (:437)
    a_i[2, 3, 1] = 0.25
    h_i[2, 3, 1] = 0.05
    v_i[2, 3, 1] = a_i[2, 3, 1] * h_i[2, 3, 1]
    # jl=2: a_i < 0.01 (0.005) with h_i = 0.5 -> removed in pnd_LEV (:437)
    a_i[2, 3, 2] = 0.005
    h_i[2, 3, 2] = 0.5
    v_i[2, 3, 2] = a_i[2, 3, 2] * h_i[2, 3, 2]
    # jl=3: strong surface melt driving area-cap overflow (:464-465) and partial lid melt (:478)
    a_i[2, 3, 3] = 0.30
    h_i[2, 3, 3] = 2.50
    v_i[2, 3, 3] = a_i[2, 3, 3] * h_i[2, 3, 3]
    a_ip[2, 3, 3] = 0.12
    h_ip[2, 3, 3] = 0.40  # v_ip = 0.048
    h_il[2, 3, 3] = 0.30  # v_il = 0.036 > zdv_mlt -> partial lid melt
    v_ip[2, 3, 3] = h_ip[2, 3, 3] * a_ip[2, 3, 3]
    v_il[2, 3, 3] = h_il[2, 3, 3] * a_ip[2, 3, 3]
    dh_i_sum_2d[2, 3, 3] = -0.80
    dh_s_sum_2d[2, 3, 3] = -0.30
    t_su[2, 3, 3] = T0  # warm surface, no freeze
    # jl=4: moderate melt on thinner ice driving depth-cap overflow (:471-472)
    a_i[2, 3, 4] = 0.40
    h_i[2, 3, 4] = 0.25  # zv_ip_max_depth = 1.25 * 0.4 * 0.25 * 0.0625 = 0.0078125 << zv_ip_max_area
    v_i[2, 3, 4] = a_i[2, 3, 4] * h_i[2, 3, 4]
    a_ip[2, 3, 4] = 0.05
    h_ip[2, 3, 4] = 0.10  # v_ip = 0.005
    h_il[2, 3, 4] = 0.01
    v_ip[2, 3, 4] = h_ip[2, 3, 4] * a_ip[2, 3, 4]
    v_il[2, 3, 4] = h_il[2, 3, 4] * a_ip[2, 3, 4]
    dh_i_sum_2d[2, 3, 4] = -0.20
    dh_s_sum_2d[2, 3, 4] = -0.10
    t_su[2, 3, 4] = T0

    # Cell (2, 4): freezing regimes, lid ratio removal, and Darcy flushing regimes
    # jl=0: cold surface freezing partially shrinking pond and growing lid (:490-497), permeable Darcy partial flush
    a_i[2, 4, 0] = 0.35
    h_i[2, 4, 0] = 1.20
    v_i[2, 4, 0] = a_i[2, 4, 0] * h_i[2, 4, 0]
    a_ip[2, 4, 0] = 0.18
    h_ip[2, 4, 0] = 0.22
    h_il[2, 4, 0] = 0.02
    v_ip[2, 4, 0] = h_ip[2, 4, 0] * a_ip[2, 4, 0]
    v_il[2, 4, 0] = h_il[2, 4, 0] * a_ip[2, 4, 0]
    dh_i_sum_2d[2, 4, 0] = 0.0
    dh_s_sum_2d[2, 4, 0] = 0.0
    t_su[2, 4, 0] = T0 - 4.5
    t_i[2, 4, :, 0] = T0 - 2.5
    sz_i[2, 4, :, 0] = 2.0

    # jl=1: very cold surface freezing that completely exhausts a shallow pond (pos_frz >= v_ip) and leaves a_ip <= 0.01
    a_i[2, 4, 1] = 0.25
    h_i[2, 4, 1] = 1.00
    v_i[2, 4, 1] = a_i[2, 4, 1] * h_i[2, 4, 1]
    a_ip[2, 4, 1] = 0.08
    h_ip[2, 4, 1] = 0.002  # tiny v_ip = 0.00016
    h_il[2, 4, 1] = 0.01
    v_ip[2, 4, 1] = h_ip[2, 4, 1] * a_ip[2, 4, 1]
    v_il[2, 4, 1] = h_il[2, 4, 1] * a_ip[2, 4, 1]
    dh_i_sum_2d[2, 4, 1] = 0.0
    dh_s_sum_2d[2, 4, 1] = 0.0
    t_su[2, 4, 1] = T0 - 25.0

    # jl=2: thick lid with h_il > 10 * h_ip -> pond/lid removed at :537-541
    a_i[2, 4, 2] = 0.30
    h_i[2, 4, 2] = 1.50
    v_i[2, 4, 2] = a_i[2, 4, 2] * h_i[2, 4, 2]
    a_ip[2, 4, 2] = 0.10
    h_ip[2, 4, 2] = 0.015
    h_il[2, 4, 2] = 0.30  # h_il > 10 * h_ip
    v_ip[2, 4, 2] = h_ip[2, 4, 2] * a_ip[2, 4, 2]
    v_il[2, 4, 2] = h_il[2, 4, 2] * a_ip[2, 4, 2]
    dh_i_sum_2d[2, 4, 2] = 0.0
    dh_s_sum_2d[2, 4, 2] = 0.0
    t_su[2, 4, 2] = T0 - 0.5
    t_i[2, 4, :, 2] = T0 - 25.0  # cold impermeable ice (zperm = 0)
    sz_i[2, 4, :, 2] = 0.1

    # jl=3: very permeable warm salty ice where Darcy flushing exhausts the entire pond (pos_flush >= v_ip)
    a_i[2, 4, 3] = 0.30
    h_i[2, 4, 3] = 0.40
    v_i[2, 4, 3] = a_i[2, 4, 3] * h_i[2, 4, 3]
    a_ip[2, 4, 3] = 0.05
    h_ip[2, 4, 3] = 0.02
    h_il[2, 4, 3] = 0.0
    v_ip[2, 4, 3] = h_ip[2, 4, 3] * a_ip[2, 4, 3]
    v_il[2, 4, 3] = 0.0
    dh_i_sum_2d[2, 4, 3] = 0.0
    dh_s_sum_2d[2, 4, 3] = 0.0
    t_su[2, 4, 3] = T0
    t_i[2, 4, :, 3] = T0 - 0.15  # very high brine fraction -> huge permeability
    sz_i[2, 4, :, 3] = 15.0

    # jl=4: warm layer (zt1 >= -1e-6 -> zv_br = 0) and zs_br < 1e-10 floor in ice_perm_eff, plus initial v_ip > zaspect*a_i so a_ip_1 is capped at a_i (:508)
    a_i[2, 4, 4] = 0.20
    h_i[2, 4, 4] = 2.00
    v_i[2, 4, 4] = a_i[2, 4, 4] * h_i[2, 4, 4]
    a_ip[2, 4, 4] = 0.20
    h_ip[2, 4, 4] = 1.20  # v_ip = 0.24 > zaspect * a_i (0.16), so sqrt(v_ip*z1_aspect*a_i) > a_i -> capped at a_i!
    h_il[2, 4, 4] = 0.0
    v_ip[2, 4, 4] = h_ip[2, 4, 4] * a_ip[2, 4, 4]
    v_il[2, 4, 4] = 0.0
    dh_i_sum_2d[2, 4, 4] = 0.0
    dh_s_sum_2d[2, 4, 4] = 0.0
    t_su[2, 4, 4] = T0
    t_i[2, 4, :, 4] = T0 - 2.0
    t_i[2, 4, 5, 4] = T0 + 1.0e-4  # warm layer: zt1 > 0 -> zs_br < 1e-10 floor & zv_br = 0

    return dict(
        a_i=a_i,
        v_i=v_i,
        h_i=h_i,
        t_su=t_su,
        a_ip=a_ip,
        v_ip=v_ip,
        h_ip=h_ip,
        v_il=v_il,
        h_il=h_il,
        a_ip_frac=a_ip_frac,
        dh_i_sum_2d=dh_i_sum_2d,
        dh_s_sum_2d=dh_s_sum_2d,
        t_i=t_i,
        sz_i=sz_i,
        wfx_pnd=wfx_pnd,
    )


_PND_ARRAY_NAMES = tuple(_pnd_2d_fixture())


def _pnd_array_args() -> tuple[jax.Array, ...]:
    fx = _pnd_2d_fixture()
    return tuple(jnp.asarray(fx[name], dtype=jnp.float64) for name in _PND_ARRAY_NAMES)


def _call_si3_ice_thd_pnd(*values: jax.Array, namelist: dict = DECK_PND_NML) -> pnd_mod.SI3PondOutputs:
    arrays = dict(zip(_PND_ARRAY_NAMES, values, strict=True))
    return pnd_mod.si3_ice_thd_pnd(
        **arrays,
        namelist=namelist,
        rDt_ice=RDT,
        r1_Dt_ice=R1_RDT,
        rho_w=RHO_W,
        gravity=GRAVITY,
        constants=NEMO_SI3_CONSTANTS_CONFIG,
        nn_hls=2,
    )


_JIT_SI3_ICE_THD_PND = jax.jit(_call_si3_ice_thd_pnd)


def _source_mutant(function, old: str, new: str):
    """Compile a one-site mutant from the real production function source."""
    source = textwrap.dedent(inspect.getsource(function))
    assert source.count(old) == 1, f"expected 1 occurrence of {old!r}, found {source.count(old)}"
    namespace = dict(vars(pnd_mod))
    namespace["_barrier_identity"] = lambda value: value
    exec(
        compile(source.replace(old, new), inspect.getsourcefile(function), "exec"),
        namespace,
    )
    return namespace[function.__name__]


def _without_source_rounding(function):
    source = textwrap.dedent(inspect.getsource(function))
    assert "sr = nemo_source_round" in source
    namespace = dict(vars(pnd_mod))
    namespace["_barrier_identity"] = lambda value: value
    exec(
        compile(
            source.replace("sr = nemo_source_round", "sr = _barrier_identity"),
            inspect.getsourcefile(function),
            "exec",
        ),
        namespace,
    )
    return namespace[function.__name__]


def test_ice_perm_eff_is_bit_exact_to_independent_scalar_under_jit_and_eager():
    T0 = float(NEMO_SI3_CONSTANTS_CONFIG.T0)
    rng = np.random.default_rng(12001)
    # 8 columns x 10 layers covering permeable, impermeable interior layer, warm layer (zt1 >= -1e-6), and zs_br floor
    t_i = T0 - rng.uniform(0.1, 3.0, (8, 10))
    sz_i = rng.uniform(4.0, 14.0, (8, 10))
    # Column 1: cold interior layer making zv_br < 1e-6 -> early exit 0.0
    t_i[1, 6] = T0 - 25.0
    sz_i[1, 6] = 0.0001
    # Column 2: warm layer zt1 = 0.0 -> zs_br = 0.0 (hits maxsd(zs_br, 1e-10) floor) and zv_br = 0.0
    t_i[2, 3] = T0
    # Column 3: warm layer zt1 > 0 -> zs_br < 0 (hits maxsd(zs_br, 1e-10) floor)
    t_i[3, 0] = T0 + 0.05

    events = {
        "perm_zs_br_floor": 0,
        "perm_warm_layer_zero": 0,
        "perm_impermeable_exit": 0,
        "perm_positive_return": 0,
    }
    oracle = np.asarray(
        [
            _oracle_ice_perm_eff(
                t_i[row].tolist(),
                sz_i[row].tolist(),
                T0=T0,
                events=events,
            )
            for row in range(t_i.shape[0])
        ],
        dtype=np.float64,
    )
    assert events["perm_zs_br_floor"] >= 2
    assert events["perm_warm_layer_zero"] >= 2
    assert events["perm_impermeable_exit"] >= 3
    assert events["perm_positive_return"] >= 5

    t_j = jnp.asarray(t_i, dtype=jnp.float64)
    s_j = jnp.asarray(sz_i, dtype=jnp.float64)
    fn = lambda t, s: pnd_mod.si3_ice_perm_eff(
        t, s, constants=NEMO_SI3_CONSTANTS_CONFIG
    )
    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        with jax.disable_jit():
            eager = fn(t_j, s_j)
        compiled = jax.jit(fn)(t_j, s_j)
    np.testing.assert_array_equal(_bits(eager), _bits(oracle))
    np.testing.assert_array_equal(_bits(compiled), _bits(oracle))


@pytest.mark.parametrize("ln_pnd_lids", [True, False])
def test_pnd_lev_is_bit_exact_to_independent_scalar_under_jit_and_eager(ln_pnd_lids):
    fx = _pnd_2d_fixture()
    # Flatten the 6 interior cells (jj in 2..3, ji in 2..4)
    sl = (slice(2, 4), slice(2, 5))
    a_i = fx["a_i"][sl].reshape(-1, 5)
    h_i = fx["h_i"][sl].reshape(-1, 5)
    t_su = fx["t_su"][sl].reshape(-1, 5)
    a_ip = fx["a_ip"][sl].reshape(-1, 5)
    h_ip = fx["h_ip"][sl].reshape(-1, 5)
    h_il = fx["h_il"][sl].reshape(-1, 5)
    dh_i_sum = fx["dh_i_sum_2d"][sl].reshape(-1, 5)
    dh_s_sum = fx["dh_s_sum_2d"][sl].reshape(-1, 5)
    sz_i = fx["sz_i"][sl].reshape(-1, 10, 5)
    t_i = fx["t_i"][sl].reshape(-1, 10, 5)
    wfx_pnd = fx["wfx_pnd"][sl].reshape(-1)
    at_i = np.sum(a_i, axis=-1)

    events = {
        name: 0
        for name in (
            "perm_zs_br_floor",
            "perm_warm_layer_zero",
            "perm_impermeable_exit",
            "perm_positive_return",
            "lev_thin_ice_removed",
            "lev_tiny_ai_removed",
            "lev_melt_growth",
            "lev_area_cap_bound",
            "lev_depth_cap_bound",
            "lev_lid_melted",
            "lev_lid_freeze_growth",
            "lev_lid_freeze_exhausts_pond",
            "lev_nolid_exp_shrink",
            "lev_pre_flush_aip_capped",
            "lev_flush_exhausts_pond",
            "lev_flush_partial",
            "lev_post_flush_aip_capped",
            "lev_lid_thickness_recovered",
            "lev_lid_small_aip_zeroed",
            "lev_lid_ratio_removed",
        )
    }
    oracle = _oracle_pnd_lev(
        at_i=at_i,
        wfx_pnd=wfx_pnd,
        a_i=a_i,
        h_i=h_i,
        t_su=t_su,
        a_ip=a_ip,
        h_ip=h_ip,
        h_il=h_il,
        dh_i_sum=dh_i_sum,
        dh_s_sum=dh_s_sum,
        sz_i=sz_i,
        t_i=t_i,
        rDt_ice=RDT,
        r1_Dt_ice=R1_RDT,
        rn_himin=DECK_PND_NML["rn_himin"],
        rn_apnd_min=DECK_PND_NML["rn_apnd_min"],
        rn_apnd_max=DECK_PND_NML["rn_apnd_max"],
        rn_pnd_flush=DECK_PND_NML["rn_pnd_flush"],
        ln_pnd_lids=ln_pnd_lids,
        rho_w=RHO_W,
        gravity=GRAVITY,
        constants=NEMO_SI3_CONSTANTS_CONFIG,
        events=events,
    )

    def run_model(at_i_a, wfx_a, ai_a, hi_a, tsu_a, aip_a, hip_a, hil_a, dhi_a, dhs_a, szi_a, ti_a):
        return pnd_mod.si3_pnd_lev(
            at_i=at_i_a,
            wfx_pnd=wfx_a,
            a_i=ai_a,
            h_i=hi_a,
            t_su=tsu_a,
            a_ip=aip_a,
            h_ip=hip_a,
            h_il=hil_a,
            dh_i_sum=dhi_a,
            dh_s_sum=dhs_a,
            sz_i=szi_a,
            t_i=ti_a,
            rDt_ice=RDT,
            r1_Dt_ice=R1_RDT,
            rn_himin=DECK_PND_NML["rn_himin"],
            rn_apnd_min=DECK_PND_NML["rn_apnd_min"],
            rn_apnd_max=DECK_PND_NML["rn_apnd_max"],
            rn_pnd_flush=DECK_PND_NML["rn_pnd_flush"],
            ln_pnd_lids=ln_pnd_lids,
            rho_w=RHO_W,
            gravity=GRAVITY,
            constants=NEMO_SI3_CONSTANTS_CONFIG,
        )

    args = tuple(
        jnp.asarray(x, dtype=jnp.float64)
        for x in (at_i, wfx_pnd, a_i, h_i, t_su, a_ip, h_ip, h_il, dh_i_sum, dh_s_sum, sz_i, t_i)
    )
    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        with jax.disable_jit():
            eager = run_model(*args)
        compiled = jax.jit(run_model)(*args)
    for actual in (eager, compiled):
        for got, want in zip(actual, oracle, strict=True):
            np.testing.assert_array_equal(_bits(got), _bits(want))

    if ln_pnd_lids:
        assert events["lev_thin_ice_removed"] >= 1
        assert events["lev_tiny_ai_removed"] >= 1
        assert events["lev_melt_growth"] >= 1
        assert events["lev_area_cap_bound"] >= 1
        assert events["lev_depth_cap_bound"] >= 1
        assert events["lev_lid_melted"] >= 1
        assert events["lev_lid_freeze_growth"] >= 1
        assert events["lev_lid_freeze_exhausts_pond"] >= 1
        assert events["lev_pre_flush_aip_capped"] >= 1
        assert events["lev_flush_exhausts_pond"] >= 1
        assert events["lev_flush_partial"] >= 1
        assert events["lev_lid_thickness_recovered"] >= 1
        assert events["lev_lid_small_aip_zeroed"] >= 1
        assert events["lev_lid_ratio_removed"] >= 1
    else:
        assert events["lev_nolid_exp_shrink"] >= 1


def test_ice_thd_pnd_is_bit_exact_to_independent_scalar_and_preserves_halo():
    fx = _pnd_2d_fixture()
    events = {
        name: 0
        for name in (
            "perm_zs_br_floor",
            "perm_warm_layer_zero",
            "perm_impermeable_exit",
            "perm_positive_return",
            "lev_thin_ice_removed",
            "lev_tiny_ai_removed",
            "lev_melt_growth",
            "lev_area_cap_bound",
            "lev_depth_cap_bound",
            "lev_lid_melted",
            "lev_lid_freeze_growth",
            "lev_lid_freeze_exhausts_pond",
            "lev_nolid_exp_shrink",
            "lev_pre_flush_aip_capped",
            "lev_flush_exhausts_pond",
            "lev_flush_partial",
            "lev_post_flush_aip_capped",
            "lev_lid_thickness_recovered",
            "lev_lid_small_aip_zeroed",
            "lev_lid_ratio_removed",
            "lev_skipped_inactive_cell",
            "pnd_vanished_ice_conserved",
        )
    }
    oracle = _oracle_ice_thd_pnd(
        **fx,
        namelist=DECK_PND_NML,
        rDt_ice=RDT,
        r1_Dt_ice=R1_RDT,
        rho_w=RHO_W,
        gravity=GRAVITY,
        constants=NEMO_SI3_CONSTANTS_CONFIG,
        nn_hls=2,
        events=events,
    )
    args = _pnd_array_args()
    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        with jax.disable_jit():
            eager = _call_si3_ice_thd_pnd(*args)
        compiled = _JIT_SI3_ICE_THD_PND(*args)
    for actual in (eager, compiled):
        for got, want in zip(actual, oracle, strict=True):
            np.testing.assert_array_equal(_bits(got), _bits(want))

    assert events["pnd_vanished_ice_conserved"] >= 4
    assert events["lev_skipped_inactive_cell"] >= 1

    # Halo ring (nn_hls = 2) must be bit-identical to entry for all pond state and wfx_pnd
    for field_name in ("a_ip", "v_ip", "h_ip", "v_il", "h_il", "a_ip_frac", "wfx_pnd"):
        got = np.asarray(getattr(compiled, field_name))
        entry = fx[field_name]
        np.testing.assert_array_equal(_bits(got[:2]), _bits(entry[:2]))
        np.testing.assert_array_equal(_bits(got[-2:]), _bits(entry[-2:]))
        np.testing.assert_array_equal(_bits(got[:, :2]), _bits(entry[:, :2]))
        np.testing.assert_array_equal(_bits(got[:, -2:]), _bits(entry[:, -2:]))


def test_o12_inexact_constants_and_output_digest_are_pinned():
    assert np.float64(pnd_mod._PND_ZASPECT).hex() == "0x1.999999999999ap-1"
    assert np.float64(pnd_mod._PND_ZTP_C).hex() == "-0x1.3333333333333p-3"
    assert np.float64(pnd_mod._PND_ZVISC).hex() == "0x1.d53cddd6e04c0p-10"
    assert np.float64(pnd_mod._PND_TINY_AI).hex() == "0x1.47ae147ae147bp-7"
    assert np.float64(pnd_mod._PERM_VC19_C1).hex() == "0x1.2b33333333333p+4"
    assert np.float64(pnd_mod._PERM_VC19_C2).hex() == "0x1.09ba5e353f7cfp-1"
    assert np.float64(pnd_mod._PERM_VC19_C3).hex() == "0x1.5e9e1b089a027p-8"
    assert np.float64(pnd_mod._PERM_SCALE).hex() == "0x1.01b2b29a4692bp-25"
    assert np.float64(DECK_PND_NML["rn_apnd_min"]).hex() == "0x1.3333333333333p-3"
    assert np.float64(DECK_PND_NML["rn_apnd_max"]).hex() == "0x1.999999999999ap-1"
    assert np.float64(DECK_PND_NML["rn_pnd_flush"]).hex() == "0x1.999999999999ap-4"
    assert np.float64(DECK_PND_NML["rn_himin"]).hex() == "0x1.999999999999ap-4"
    assert np.float64(R1_RDT).hex() == "0x1.845c8a0ce5129p-14"
    assert np.float64(1.0 / RHO_0).hex() == "0x1.ff007fc01ff00p-11"

    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        out = _JIT_SI3_ICE_THD_PND(*_pnd_array_args())
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(out):
        digest.update(np.asarray(leaf, dtype=np.float64).tobytes())
    assert (
        digest.hexdigest()
        == "5332d71c74a660ea60985d1046bea4f2c24cc60895d711f9dd5039e42b203e89"
    )


@pytest.mark.parametrize("site", ["ice_perm_eff", "pnd_lev_core", "ice_thd_pnd"])
def test_real_model_barrier_groups_have_bit_changing_stripped_plants(site, monkeypatch):
    """Stripping nemo_source_round from the real model function changes bits under jit."""
    rng = np.random.default_rng(202612)
    T0 = float(NEMO_SI3_CONSTANTS_CONFIG.T0)

    if site == "ice_perm_eff":
        t_i = jnp.asarray(T0 - rng.uniform(0.15, 1.5, (1500, 10)), dtype=jnp.float64)
        sz_i = jnp.asarray(rng.uniform(3.0, 15.0, (1500, 10)), dtype=jnp.float64)
        fn = lambda t, s: pnd_mod.si3_ice_perm_eff(
            t, s, constants=NEMO_SI3_CONSTANTS_CONFIG
        )
        stripped_perm = _without_source_rounding(pnd_mod.si3_ice_perm_eff)
        stripped_fn = lambda t, s: stripped_perm(
            t, s, constants=NEMO_SI3_CONSTANTS_CONFIG
        )
        with jax.disable_jit():
            eager = fn(t_i, sz_i)
        protected = jax.jit(fn)(t_i, sz_i)
        stripped = jax.jit(stripped_fn)(t_i, sz_i)
        np.testing.assert_array_equal(_bits(protected), _bits(eager))
        assert np.count_nonzero(_bits(stripped) != _bits(eager)) > 0
        return

    if site == "pnd_lev_core":
        n = 800
        kwargs = dict(
            at_i=jnp.asarray(rng.uniform(0.3, 0.95, n), dtype=jnp.float64),
            wfx_pnd=jnp.asarray(rng.uniform(-1.0e-5, 1.0e-5, n), dtype=jnp.float64),
            a_i=jnp.asarray(rng.uniform(0.15, 0.85, n), dtype=jnp.float64),
            h_i=jnp.asarray(rng.uniform(0.3, 2.5, n), dtype=jnp.float64),
            t_su=jnp.asarray(T0 - rng.uniform(0.2, 6.0, n), dtype=jnp.float64),
            a_ip=jnp.asarray(rng.uniform(0.03, 0.25, n), dtype=jnp.float64),
            h_ip=jnp.asarray(rng.uniform(0.03, 0.20, n), dtype=jnp.float64),
            h_il=jnp.asarray(rng.uniform(0.005, 0.04, n), dtype=jnp.float64),
            dh_i_sum=jnp.asarray(-rng.uniform(0.001, 0.05, n), dtype=jnp.float64),
            dh_s_sum=jnp.asarray(-rng.uniform(0.001, 0.03, n), dtype=jnp.float64),
            sz_i=jnp.asarray(rng.uniform(1.5, 4.0, (n, 10)), dtype=jnp.float64),
            t_i=jnp.asarray(T0 - rng.uniform(1.5, 3.5, (n, 10)), dtype=jnp.float64),
            rDt_ice=jnp.asarray(RDT, dtype=jnp.float64),
            r1_Dt_ice=jnp.asarray(R1_RDT, dtype=jnp.float64),
            rn_himin=jnp.asarray(0.1, dtype=jnp.float64),
            rn_apnd_min=jnp.asarray(0.15, dtype=jnp.float64),
            rn_apnd_max=jnp.asarray(0.80, dtype=jnp.float64),
            rn_pnd_flush=jnp.asarray(0.1, dtype=jnp.float64),
            ln_pnd_lids=True,
            rhow=jnp.asarray(RHO_W, dtype=jnp.float64),
            grav=jnp.asarray(GRAVITY, dtype=jnp.float64),
            constants=NEMO_SI3_CONSTANTS_CONFIG,
            z1_rhow=jnp.asarray(1.0 / RHO_W, dtype=jnp.float64),
            z1_aspect=jnp.asarray(1.0 / 0.8, dtype=jnp.float64),
            z1_Tp=jnp.asarray(1.0 / -0.15, dtype=jnp.float64),
        )
        fn = lambda: pnd_mod._pnd_lev_single_category(**kwargs)
        with jax.disable_jit():
            eager = fn()
        protected = jax.jit(fn)()
        stripped_core = _without_source_rounding(pnd_mod._pnd_lev_single_category)
        stripped = jax.jit(lambda: stripped_core(**kwargs))()
        for got, want in zip(protected, eager, strict=True):
            np.testing.assert_array_equal(_bits(got), _bits(want))
        assert sum(
            np.count_nonzero(_bits(got) != _bits(want))
            for got, want in zip(stripped, eager, strict=True)
        ) > 0
        return

    args = _pnd_array_args()
    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        with jax.disable_jit():
            eager = _call_si3_ice_thd_pnd(*args)
        protected = jax.jit(lambda *v: _call_si3_ice_thd_pnd(*v))(*args)
        monkeypatch.setattr(
            pnd_mod,
            "_pnd_lev_single_category",
            _without_source_rounding(pnd_mod._pnd_lev_single_category),
        )
        monkeypatch.setattr(
            pnd_mod,
            "si3_ice_thd_pnd",
            _without_source_rounding(pnd_mod.si3_ice_thd_pnd),
        )
        stripped = jax.jit(lambda *v: _call_si3_ice_thd_pnd(*v))(*args)
    for got, want in zip(protected, eager, strict=True):
        np.testing.assert_array_equal(_bits(got), _bits(want))
    assert sum(
        np.count_nonzero(_bits(got) != _bits(want))
        for got, want in zip(stripped, eager, strict=True)
    ) > 0


@pytest.mark.parametrize(
    ("target_name", "old", "new"),
    [
        (
            "si3_ice_perm_eff",
            "denom = _maxsd(zs_br, epsi10)",
            "denom = _maxsd(zs_br, jnp.float64(100.0))",
        ),
        (
            "si3_ice_perm_eff",
            "low = zv < epsi06",
            "low = zv < jnp.float64(0.05)",
        ),
        (
            "_pnd_lev_single_category",
            "zdv_mlt = _maxsd(_minsd(sr(zv_ip_max_area - v_ip), zdv_mlt), zero)",
            "zdv_mlt = _maxsd(zdv_mlt, zero)",
        ),
        (
            "_pnd_lev_single_category",
            "zdv_mlt = _maxsd(_minsd(zdv_mlt, sr(zv_ip_max_depth - v_ip)), zero)",
            "zdv_mlt = _maxsd(zdv_mlt, zero)",
        ),
        (
            "_pnd_lev_single_category",
            "v_il = _maxsd(sr(v_il + pos_frz), zero)",
            "v_il = _maxsd(sr(v_il - pos_frz), zero)",
        ),
        (
            "_pnd_lev_single_category",
            "v_ip = sr(v_ip - capped_flush)",
            "v_ip = v_ip",
        ),
        (
            "_pnd_lev_single_category",
            "lid_too_thick = h_il_2 > sr(h_ip_2 * lid_ratio_max)",
            "lid_too_thick = h_il_2 > sr(h_ip_2 * 100.0)",
        ),
        (
            "si3_ice_thd_pnd",
            "wfx_int = jnp.where(vanished, sr(wfx_int + flux_inc), wfx_int)",
            "wfx_int = wfx_int",
        ),
    ],
)
def test_production_code_path_plants_fire_through_si3_ice_thd_pnd(
    target_name, old, new, monkeypatch
):
    args = _pnd_array_args()
    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        baseline = _JIT_SI3_ICE_THD_PND(*args)
        target_fn = getattr(pnd_mod, target_name)
        mutant_fn = _source_mutant(target_fn, old, new)
        monkeypatch.setattr(pnd_mod, target_name, mutant_fn)
        mutated = _call_si3_ice_thd_pnd(*args)
    changed = sum(
        np.count_nonzero(_bits(got) != _bits(want))
        for got, want in zip(mutated, baseline, strict=True)
    )
    assert changed > 0, f"plant ({target_name}, {old!r} -> {new!r}) failed to change output"


def test_ice_perm_eff_epsi06_early_exit_boundary_plant():
    """Column with epsi10 < zv_br < epsi06 returns 0.0; mutating epsi06 -> epsi10 returns > 0."""
    T0 = float(NEMO_SI3_CONSTANTS_CONFIG.T0)
    t_i = jnp.full((1, 10), T0 - 20.0, dtype=jnp.float64)
    sz_i = jnp.full((1, 10), 1.0e-4, dtype=jnp.float64)  # zv_br ~ 3.7e-7 in (epsi10, epsi06)
    base = jax.jit(
        lambda t, s: pnd_mod.si3_ice_perm_eff(
            t, s, constants=NEMO_SI3_CONSTANTS_CONFIG
        )
    )(t_i, sz_i)
    assert float(base[0]) == 0.0
    mutant = _source_mutant(
        pnd_mod.si3_ice_perm_eff,
        "low = zv < epsi06",
        "low = zv < epsi10",
    )
    planted = jax.jit(
        lambda t, s: mutant(t, s, constants=NEMO_SI3_CONSTANTS_CONFIG)
    )(t_i, sz_i)
    assert float(planted[0]) > 0.0


def test_masked_and_signed_zero_paths_are_verified_through_pnd_lev(monkeypatch):
    """Verify the 456 and 478 MASKED zero-tie sites and zero-pond outputs."""
    T0 = float(NEMO_SI3_CONSTANTS_CONFIG.T0)
    # Column 0: dh_i_sum = +0.0, dh_s_sum = +0.0 -> zdv_avail = -0.0, 456 ties with +0.0
    # Column 1: dh_i_sum = -0.0, dh_s_sum = -0.0 -> zdv_avail = +0.0, 456 has +0.0 vs +0.0
    # Both have v_ip = 0.0, v_il = 0.0, t_su = T0 -> 478 has +0.0 - +0.0 = +0.0
    kwargs = dict(
        at_i=jnp.asarray([0.8, 0.8], dtype=jnp.float64),
        wfx_pnd=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        a_i=jnp.asarray([0.8, 0.8], dtype=jnp.float64),
        h_i=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        t_su=jnp.asarray([T0, T0], dtype=jnp.float64),
        a_ip=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        h_ip=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        h_il=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        dh_i_sum=jnp.asarray([0.0, -0.0], dtype=jnp.float64),
        dh_s_sum=jnp.asarray([0.0, -0.0], dtype=jnp.float64),
        sz_i=jnp.full((2, 10), 5.0, dtype=jnp.float64),
        t_i=jnp.full((2, 10), T0 - 2.0, dtype=jnp.float64),
        rDt_ice=RDT,
        r1_Dt_ice=R1_RDT,
        rn_himin=0.1,
        rn_apnd_min=0.15,
        rn_apnd_max=0.80,
        rn_pnd_flush=0.1,
        ln_pnd_lids=True,
        rho_w=RHO_W,
        gravity=GRAVITY,
        constants=NEMO_SI3_CONSTANTS_CONFIG,
    )
    out = jax.jit(lambda: pnd_mod.si3_pnd_lev(**kwargs))()
    for leaf in out:
        bits = _bits(leaf)
        np.testing.assert_array_equal(bits, np.zeros_like(bits))

    # Even if 456's and 478's dst/src roles are flipped in _pnd_lev_single_category,
    # 465/472 and 490/494 mask the zero sign so all outputs remain +0.0.
    flipped_456 = _source_mutant(
        pnd_mod._pnd_lev_single_category,
        "zdv_mlt = _maxsd(sr(zfr_mlt * zdv_avail), zero)",
        "zdv_mlt = _maxsd(zero, sr(zfr_mlt * zdv_avail))",
    )
    monkeypatch.setattr(pnd_mod, "_pnd_lev_single_category", flipped_456)
    out_flipped = jax.jit(lambda: pnd_mod.si3_pnd_lev(**kwargs))()
    for got, want in zip(out_flipped, out, strict=True):
        np.testing.assert_array_equal(_bits(got), _bits(want))


def test_five_category_o9_o10_o11_1d2d_o12_composition_matches_scalar_oracle(monkeypatch):
    namelist = {
        **o9_fixture._deck(),
        **o10_fixture.NML,
        **DECK_PND_NML,
        "ln_icedA": True,
        "rn_beta": 0.5,
        "rn_dmin": 8.0,
    }
    categories = o9_fixture._categories(seed=19)
    assert len(categories) == 5
    # Build a 2D grid (jpj=5, jpi=5, nn_hls=2 -> 1x1=1 interior cell) with 5 categories
    jpj, jpi, jpl = 5, 5, 5
    n_cells = jpj * jpi
    rng = np.random.default_rng(20261205)

    entries = []
    fluxes_list = []
    for jl in range(jpl):
        base_cells = categories[jl]
        repeated_cells = [base_cells[idx % len(base_cells)] for idx in range(n_cells)]
        entry_1d, fluxes_1d = o9_fixture._entry(repeated_cells)
        entry_2d = {
            k: v.reshape((jpj, jpi) + v.shape[1:]) if hasattr(v, "reshape") else v
            for k, v in entry_1d.items()
        }
        entry_2d["at_i"] = jnp.full((jpj, jpi), 0.75, dtype=jnp.float64)
        entry_2d["sst"] = jnp.full((jpj, jpi), 0.2, dtype=jnp.float64)
        entry_2d["o_i"] = jnp.full((jpj, jpi), 10.0, dtype=jnp.float64)
        fluxes_2d = bl.SI3DHFluxes(
            **{
                f: getattr(fluxes_1d, f).reshape(jpj, jpi)
                for f in fluxes_1d._fields
            }
        )
        entries.append(entry_2d)
        fluxes_list.append(fluxes_2d)

    sfx_bri = jnp.zeros((jpj, jpi), dtype=jnp.float64)
    sfx_lam = jnp.zeros((jpj, jpi), dtype=jnp.float64)
    hfx_thd = jnp.zeros((jpj, jpi), dtype=jnp.float64)
    wfx_lam = jnp.zeros((jpj, jpi), dtype=jnp.float64)
    a_ip = jnp.asarray(rng.uniform(0.02, 0.08, (jpj, jpi, jpl)), dtype=jnp.float64)
    h_ip = jnp.asarray(rng.uniform(0.05, 0.14, (jpj, jpi, jpl)), dtype=jnp.float64)
    h_il = jnp.asarray(rng.uniform(0.0, 0.02, (jpj, jpi, jpl)), dtype=jnp.float64)
    v_ip = jnp.asarray(np.asarray(h_ip) * np.asarray(a_ip), dtype=jnp.float64)
    v_il = jnp.asarray(np.asarray(h_il) * np.asarray(a_ip), dtype=jnp.float64)
    a_ip_frac = jnp.asarray(
        np.asarray(a_ip) / np.maximum(np.stack([np.asarray(e["a_i"]) for e in entries], axis=-1), 1.0e-10),
        dtype=jnp.float64,
    )
    wfx_pnd = jnp.zeros((jpj, jpi), dtype=jnp.float64)

    # Compile single-category O10 once and reuse across all 5 categories
    jit_o10 = jax.jit(
        lambda e, f, b: bl.si3_ice_thd_o10_category_stages(
            entry=e,
            fluxes=f,
            sfx_bri=b,
            namelist=namelist,
            rDt_ice=RDT,
            r1_Dt_ice=R1_RDT,
            r1_nlay_i=R1_NLAY_I,
            rho_i=RHO_I,
            rho_s=RHO_S,
            rho_w=RHO_W,
            gravity=GRAVITY,
            constants=NEMO_SI3_CONSTANTS_CONFIG,
        )
    )
    monkeypatch.setattr(
        pnd_mod,
        "si3_ice_thd_o10_category_stages",
        lambda **kw: jit_o10(kw["entry"], kw["fluxes"], kw["sfx_bri"]),
    )

    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        out = pnd_mod.si3_ice_thd_o12_stages(
            category_entries=tuple(entries),
            category_fluxes=tuple(fluxes_list),
            sfx_bri=sfx_bri,
            sfx_lam=sfx_lam,
            hfx_thd=hfx_thd,
            wfx_lam=wfx_lam,
            a_ip=a_ip,
            v_ip=v_ip,
            h_ip=h_ip,
            v_il=v_il,
            h_il=h_il,
            a_ip_frac=a_ip_frac,
            wfx_pnd=wfx_pnd,
            namelist=namelist,
            rDt_ice=RDT,
            r1_Dt_ice=R1_RDT,
            r1_nlay_i=R1_NLAY_I,
            r1_nlay_s=R1_NLAY_S,
            rho_i=RHO_I,
            rho_s=RHO_S,
            rho_w=RHO_W,
            gravity=GRAVITY,
            constants=NEMO_SI3_CONSTANTS_CONFIG,
            nn_hls=2,
        )

    assert len(out.categories) == 5
    oracle_pnd = _oracle_ice_thd_pnd(
        a_i=np.stack([np.asarray(c.a_i) for c in out.categories], axis=-1),
        v_i=np.stack([np.asarray(c.v_i) for c in out.categories], axis=-1),
        h_i=np.stack([np.asarray(c.h_i) for c in out.categories], axis=-1),
        t_su=np.stack([np.asarray(c.t_su) for c in out.categories], axis=-1),
        a_ip=np.asarray(a_ip),
        v_ip=np.asarray(v_ip),
        h_ip=np.asarray(h_ip),
        v_il=np.asarray(v_il),
        h_il=np.asarray(h_il),
        a_ip_frac=np.asarray(a_ip_frac),
        dh_i_sum_2d=np.stack([np.asarray(c.dh_i_sum) for c in out.categories], axis=-1),
        dh_s_sum_2d=np.stack([np.asarray(c.dh_s_sum) for c in out.categories], axis=-1),
        t_i=np.stack([np.asarray(c.t_i) for c in out.categories], axis=-1),
        sz_i=np.stack([np.asarray(c.sz_i) for c in out.categories], axis=-1),
        wfx_pnd=np.asarray(wfx_pnd),
        namelist=namelist,
        rDt_ice=RDT,
        r1_Dt_ice=R1_RDT,
        rho_w=RHO_W,
        gravity=GRAVITY,
        constants=NEMO_SI3_CONSTANTS_CONFIG,
        nn_hls=2,
    )
    for got, want in zip(out.pnd, oracle_pnd, strict=True):
        np.testing.assert_array_equal(_bits(got), _bits(want))


def test_o12_composer_calls_o10_then_da_then_1d2d_then_pnd(monkeypatch):
    """Spy on si3_ice_thd_o12_stages to verify exact call sequence and argument wiring."""
    calls = []
    jpj, jpi, jpl, nlay_i, nlay_s = 5, 5, 5, 10, 5
    one_2d = jnp.ones((jpj, jpi), dtype=jnp.float64)
    one_i = jnp.ones((jpj, jpi, nlay_i), dtype=jnp.float64)
    one_s = jnp.ones((jpj, jpi, nlay_s), dtype=jnp.float64)
    one_3d = jnp.ones((jpj, jpi, jpl), dtype=jnp.float64)

    def o10_spy(**kwargs):
        calls.append("o10")
        dh_ns = SimpleNamespace(
            a_i=one_2d * 0.2,
            h_i=one_2d * 1.5,
            h_s=one_2d * 0.1,
            t_su=one_2d * 270.0,
            t_s=one_s * 268.0,
            s_i=one_2d * 4.0,
            e_i=one_i * 1.0e8,
            e_s=one_s * 5.0e7,
            dh=SimpleNamespace(dh_i_sum=-one_2d * 0.01, dh_s_sum=-one_2d * 0.005),
            fluxes=SimpleNamespace(sfx_res=one_2d),
        )
        o9_ns = SimpleNamespace(dh=dh_ns, t_i=one_i * 271.0)
        sal_ns = SimpleNamespace(sz_i=one_i * 5.0, sfx_bri=kwargs["sfx_bri"] + 1.0)
        return SimpleNamespace(o9=o9_ns, sal=sal_ns, t_i=one_i * 271.0)

    def da_spy(**kwargs):
        calls.append("da")
        return SimpleNamespace(
            a_i=kwargs["a_i"] * 0.95,
            h_i=kwargs["h_i"],
            h_s=kwargs["h_s"],
            sfx_lam=kwargs["sfx_lam"] + 2.0,
            hfx_thd=kwargs["hfx_thd"] + 3.0,
            wfx_lam=kwargs["wfx_lam"] + 4.0,
        )

    def pnd_spy(**kwargs):
        calls.append("pnd")
        # Verify ice_thd_1d2d(jl, 2) computed v_i = h_i * a_i (1.5 * 0.19 = 0.285)
        np.testing.assert_allclose(kwargs["v_i"], 1.5 * (0.2 * 0.95))
        return pnd_mod.SI3PondOutputs(
            at_i=one_2d,
            a_ip=one_3d,
            v_ip=one_3d,
            h_ip=one_3d,
            v_il=one_3d,
            h_il=one_3d,
            a_ip_frac=one_3d,
            wfx_pnd=one_2d * 42.0,
        )

    monkeypatch.setattr(pnd_mod, "si3_ice_thd_o10_category_stages", o10_spy)
    monkeypatch.setattr(pnd_mod, "si3_ice_thd_da", da_spy)
    monkeypatch.setattr(pnd_mod, "si3_ice_thd_pnd", pnd_spy)

    namelist = {
        **o9_fixture._deck(),
        **o10_fixture.NML,
        **DECK_PND_NML,
        "ln_icedA": True,
        "rn_beta": 0.5,
        "rn_dmin": 8.0,
    }
    res = pnd_mod.si3_ice_thd_o12_stages(
        category_entries=tuple(
            {"t_bo": one_2d, "sss": one_2d, "at_i": one_2d, "sst": one_2d, "o_i": one_2d}
            for _ in range(jpl)
        ),
        category_fluxes=tuple(object() for _ in range(jpl)),
        sfx_bri=one_2d,
        sfx_lam=one_2d,
        hfx_thd=one_2d,
        wfx_lam=one_2d,
        a_ip=one_3d,
        v_ip=one_3d,
        h_ip=one_3d,
        v_il=one_3d,
        h_il=one_3d,
        a_ip_frac=one_3d,
        wfx_pnd=one_2d,
        namelist=namelist,
        rDt_ice=RDT,
        r1_Dt_ice=R1_RDT,
        r1_nlay_i=R1_NLAY_I,
        r1_nlay_s=R1_NLAY_S,
        rho_i=RHO_I,
        rho_s=RHO_S,
        rho_w=RHO_W,
        gravity=GRAVITY,
        constants=NEMO_SI3_CONSTANTS_CONFIG,
        nn_hls=2,
    )
    assert calls == ["o10", "da"] * 5 + ["pnd"]
    np.testing.assert_array_equal(res.pnd.wfx_pnd, one_2d * 42.0)


def test_pnd_lev_and_ice_thd_pnd_gradients_are_finite_nonzero_and_match_central_difference():
    """Smooth interior columns have finite nonzero AD gradients matching central finite differences."""
    T0 = float(NEMO_SI3_CONSTANTS_CONFIG.T0)
    # 2 smooth interior cells x 5 categories away from piecewise kinks:
    # cell 0: melting regime (zdv_avail > 0, uncapped, partial lid melt, partial Darcy flush)
    # cell 1: freezing regime (t_su < zTp + rt0, lid freeze growth, partial Darcy flush)
    a_i = jnp.asarray(
        [
            [0.22, 0.25, 0.18, 0.20, 0.15],
            [0.20, 0.24, 0.19, 0.21, 0.16],
        ],
        dtype=jnp.float64,
    )
    h_i = jnp.asarray(
        [
            [1.4, 1.6, 1.8, 2.0, 2.2],
            [1.3, 1.5, 1.7, 1.9, 2.1],
        ],
        dtype=jnp.float64,
    )
    t_su = jnp.asarray(
        [
            [T0 - 0.05] * 5,  # above zTp + T0 (-0.15 degC), so zdT = 0
            [T0 - 2.50] * 5,  # below zTp + T0, so zdT > 0
        ],
        dtype=jnp.float64,
    )
    a_ip = jnp.asarray(
        [
            [0.08, 0.09, 0.07, 0.10, 0.06],
            [0.09, 0.10, 0.08, 0.11, 0.07],
        ],
        dtype=jnp.float64,
    )
    h_ip = jnp.asarray(
        [
            [0.12, 0.13, 0.11, 0.14, 0.10],
            [0.14, 0.15, 0.13, 0.16, 0.12],
        ],
        dtype=jnp.float64,
    )
    h_il = jnp.asarray(
        [
            [0.03, 0.03, 0.025, 0.035, 0.02],
            [0.015, 0.02, 0.015, 0.02, 0.01],
        ],
        dtype=jnp.float64,
    )
    dh_i_sum = jnp.asarray(
        [
            [-0.012, -0.015, -0.010, -0.014, -0.008],
            [0.005, 0.004, 0.006, 0.003, 0.005],
        ],
        dtype=jnp.float64,
    )
    dh_s_sum = jnp.asarray(
        [
            [-0.004, -0.005, -0.003, -0.006, -0.002],
            [0.001, 0.001, 0.002, 0.001, 0.001],
        ],
        dtype=jnp.float64,
    )
    sz_i = jnp.full((2, 10, 5), 2.0, dtype=jnp.float64)
    t_i = jnp.full((2, 10, 5), T0 - 2.5, dtype=jnp.float64)
    wfx_pnd = jnp.zeros(2, dtype=jnp.float64)

    def loss(ai, hi, tsu, aip, hip, hil, dhi, dhs, szi, ti):
        res = pnd_mod.si3_pnd_lev(
            at_i=jnp.sum(ai, axis=-1),
            wfx_pnd=wfx_pnd,
            a_i=ai,
            h_i=hi,
            t_su=tsu,
            a_ip=aip,
            h_ip=hip,
            h_il=hil,
            dh_i_sum=dhi,
            dh_s_sum=dhs,
            sz_i=szi,
            t_i=ti,
            rDt_ice=RDT,
            r1_Dt_ice=R1_RDT,
            rn_himin=0.1,
            rn_apnd_min=0.15,
            rn_apnd_max=0.80,
            rn_pnd_flush=0.1,
            ln_pnd_lids=True,
            rho_w=RHO_W,
            gravity=GRAVITY,
            constants=NEMO_SI3_CONSTANTS_CONFIG,
        )
        return (
            jnp.sum(res.v_ip)
            + 2.0 * jnp.sum(res.v_il)
            + 0.3 * jnp.sum(res.a_ip)
            + 0.1 * jnp.sum(res.h_ip)
            + 0.2 * jnp.sum(res.h_il)
            + 1.0e2 * jnp.sum(res.wfx_pnd)
        )

    primals = (a_i, h_i, t_su, a_ip, h_ip, h_il, dh_i_sum, dh_s_sum, sz_i, t_i)
    grads = jax.jit(jax.grad(loss, argnums=tuple(range(len(primals)))))(*primals)
    for idx, g in enumerate(grads):
        arr = np.asarray(g)
        assert np.all(np.isfinite(arr)), f"non-finite grad for arg {idx}"
        assert np.any(arr != 0.0), f"all-zero grad for arg {idx}"

    # Directional JVP and central finite difference along a random unit perturbation of all 10 inputs
    rng = np.random.default_rng(9120)
    tangents = tuple(
        jnp.asarray(rng.normal(scale=1.0e-3, size=p.shape), dtype=jnp.float64)
        for p in primals
    )
    _, jvp_val = jax.jit(lambda p, t: jax.jvp(loss, p, t))(primals, tangents)
    dot_grad = sum(float(jnp.sum(g * t)) for g, t in zip(grads, tangents, strict=True))
    np.testing.assert_allclose(float(jvp_val), dot_grad, rtol=1.0e-12, atol=1.0e-12)

    eps = 1.0e-4
    plus = tuple(p + eps * t for p, t in zip(primals, tangents, strict=True))
    minus = tuple(p - eps * t for p, t in zip(primals, tangents, strict=True))
    fd_val = float((loss(*plus) - loss(*minus)) / (2.0 * eps))
    np.testing.assert_allclose(dot_grad, fd_val, rtol=1.0e-5, atol=1.0e-7)

    # Plant: stop_gradient on sz_i and t_i must break the sz_i and t_i gradients
    grads_planted = jax.jit(
        jax.grad(
            lambda szi, ti: loss(
                a_i, h_i, t_su, a_ip, h_ip, h_il, dh_i_sum, dh_s_sum,
                jax.lax.stop_gradient(szi), jax.lax.stop_gradient(ti),
            ),
            argnums=(0, 1),
        )
    )(sz_i, t_i)
    assert np.all(np.asarray(grads_planted[0]) == 0.0)
    assert np.all(np.asarray(grads_planted[1]) == 0.0)


def test_edge_case_zero_pond_thin_ice_and_vanished_ice_have_finite_gradients():
    """Verify jax.grad produces no NaN on zero ponds (rad=0, disc=0), thin ice, or vanished ice."""
    args = _pnd_array_args()

    def total_loss(*values):
        out = _call_si3_ice_thd_pnd(*values)
        return (
            jnp.sum(out.a_ip)
            + jnp.sum(out.v_ip)
            + jnp.sum(out.h_ip)
            + jnp.sum(out.v_il)
            + jnp.sum(out.h_il)
            + jnp.sum(out.a_ip_frac)
            + jnp.sum(out.wfx_pnd)
        )

    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        grads = jax.jit(jax.grad(total_loss, argnums=tuple(range(len(args)))))(*args)
    for name, grad in zip(_PND_ARRAY_NAMES, grads, strict=True):
        assert np.all(np.isfinite(np.asarray(grad))), f"NaN/Inf gradient in {name}"


@pytest.mark.parametrize(
    "missing",
    [
        *_PND_ARRAY_NAMES,
        "namelist",
        "rDt_ice",
        "r1_Dt_ice",
        "rho_w",
        "gravity",
        "constants",
        "nn_hls",
    ],
)
def test_ice_thd_pnd_has_no_default_for_physical_inputs(missing):
    fx = _pnd_2d_fixture()
    kwargs = {
        **{k: jnp.asarray(v, dtype=jnp.float64) for k, v in fx.items()},
        "namelist": DECK_PND_NML,
        "rDt_ice": RDT,
        "r1_Dt_ice": R1_RDT,
        "rho_w": RHO_W,
        "gravity": GRAVITY,
        "constants": NEMO_SI3_CONSTANTS_CONFIG,
        "nn_hls": 2,
    }
    kwargs[missing] = None
    with pytest.raises(ValueError, match="requires|missing|nn_hls"):
        pnd_mod.si3_ice_thd_pnd(**kwargs)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("ln_icedH", False),
        ("ln_pnd", False),
        ("ln_pnd_LEV", False),
        ("ln_pnd_CST", True),
        ("ln_pnd_TOPO", True),
        ("ln_pnd_lids", False),
        ("ln_pnd_rain", False),
        ("ln_pnd_alb", False),
        ("ln_icediachk", True),
    ],
)
def test_ice_thd_pnd_refuses_nonproduction_selectors(key, bad_value):
    fx = _pnd_2d_fixture()
    kwargs = {
        **{k: jnp.asarray(v, dtype=jnp.float64) for k, v in fx.items()},
        "namelist": {**DECK_PND_NML, key: bad_value},
        "rDt_ice": RDT,
        "r1_Dt_ice": R1_RDT,
        "rho_w": RHO_W,
        "gravity": GRAVITY,
        "constants": NEMO_SI3_CONSTANTS_CONFIG,
        "nn_hls": 2,
    }
    with pytest.raises(ValueError, match="production selectors"):
        pnd_mod.si3_ice_thd_pnd(**kwargs)
