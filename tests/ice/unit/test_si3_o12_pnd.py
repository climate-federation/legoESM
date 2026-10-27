"""Independent scalar checks for SI3 O12 melt-pond stage (ice_thd_pnd / pnd_LEV / ice_perm_eff)."""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import hashlib
import inspect
import math
import pathlib
import re
import textwrap
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice import ponds as ponds_mod
from legoesm.ice import sea_ice as sea_ice_mod
from legoesm.ice import si3_thd_da as da_mod
from legoesm.ice import si3_thd_pnd as pnd_mod
from legoesm.ice.config import MeltPondConfig, SeaIceConfig, SI3PondConfig, SnowConfig
from legoesm.ice.constants_config import IceConstantsConfig, NEMO_SI3_CONSTANTS_CONFIG
from legoesm.ice.state import init_dynamic_ice_state
from tests.ice.unit import test_si3_o9_zdf_dh_temp as o9_fixture
from tests.ice.unit import test_si3_o10_salinity as o10_fixture


@pytest.fixture(autouse=True)
def _cpu_device_scope(monkeypatch):
    """Place all array creation and JIT execution on CPU so CPU-only libm tests pass on GPU hosts."""
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@contextlib.contextmanager
def precision_scope(policy: PrecisionPolicy):
    prev = get_policy()
    orig_backend = jax.default_backend
    jax.default_backend = lambda: "cpu"
    with jax.default_device(jax.devices("cpu")[0]):
        set_policy(policy)
        try:
            yield
        finally:
            set_policy(prev)
            jax.default_backend = orig_backend


_LIBM = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
_LIBM_EXP = _LIBM.exp
_LIBM_EXP.argtypes = (ctypes.c_double,)
_LIBM_EXP.restype = ctypes.c_double

# Independent oracle physical constants transcribed directly from the NEMO 5.0.2
# record build (ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/phycst.f90:49-50,68-75,
# eosbn2.f90:1945, par_ice.f90:134-135) — NEVER imported from legoesm.constants
# or NEMO_SI3_CONSTANTS_CONFIG.
_ORACLE_RT0 = 273.15  # const-ok: independent oracle transcribes the record build's phycst.f90:50 literal, never the model's constants
_ORACLE_GRAV = 9.80665  # phycst.f90:49 grav
_ORACLE_RHOW = 1000.0  # phycst.f90:68 rhow
_ORACLE_RHO0 = 1026.0  # eosbn2.f90:1945 rn_rho0
_ORACLE_RHOI = 917.0  # phycst.f90:70 rhoi
_ORACLE_RHOS = 330.0  # phycst.f90:71 rhos
_ORACLE_RLFUS = 0.3333601e6  # const-ok: independent oracle transcribes the record build's phycst.f90:75 literal, never the model's constants
_ORACLE_RCND_I = 2.034396  # par_ice.f90:135 rcnd_i

RDT = 10800.0
R1_RDT = 1.0 / RDT
R1_NLAY_I = 1.0 / 10.0
R1_NLAY_S = 1.0 / 5.0
RHO_I = _ORACLE_RHOI
RHO_S = _ORACLE_RHOS
RHO_W = _ORACLE_RHOW
RHO_0 = _ORACLE_RHO0
GRAVITY = _ORACLE_GRAV


def _parse_independent_pnd_namelist() -> dict[str, float | bool]:
    """Parse &namthd and &namthd_pnd directly from the committed NEMO namelist files."""
    data_dir = pathlib.Path(__file__).resolve().parent / "data" / "orca2_sas_ice_expref"
    values: dict[str, float | bool] = {}
    keys = {
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
    }
    for fname in ("namelist_ice_ref", "namelist_ice_cfg"):
        for raw_line in (data_dir / fname).read_text().splitlines():
            code = raw_line.split("!", 1)[0].strip()
            m = re.match(r"^([A-Za-z0-9_]+)\s*=\s*(\S+)", code)
            if not m or m.group(1) not in keys:
                continue
            key, tok = m.group(1), m.group(2).rstrip(",")
            if tok.lower() in (".true.", ".false."):
                values[key] = tok.lower() == ".true."
            else:
                values[key] = float(tok.replace("d", "e").replace("D", "E"))
    return values


DECK_PND_NML = _parse_independent_pnd_namelist()


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
    events: dict[str, int] | None = None,
) -> float:
    """Independent scalar translation of ``ice_perm_eff`` (``icethd_pnd.f90:1372-1422``)."""
    nlay_i = len(t_i)
    assert len(sz_i) == nlay_i
    zv_br = []
    for jk in range(nlay_i):
        zt1 = t_i[jk] - _ORACLE_RT0
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
    events: dict[str, int] | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Independent scalar translation of one category in ``pnd_LEV`` (``icethd_pnd.f90:384-559``)."""
    zaspect = 0.8
    zTp = -0.15
    zvisc = 1.79e-3
    rt0 = _ORACLE_RT0
    rho0 = _ORACLE_RHO0
    rhoi = _ORACLE_RHOI
    rhos = _ORACLE_RHOS
    rhow = _ORACLE_RHOW
    grav = _ORACLE_GRAV
    rLfus = _ORACLE_RLFUS
    rcnd_i = _ORACLE_RCND_I
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

        zperm = _oracle_ice_perm_eff(t_i, sz_i, events=events)

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
                    flux_inc = (dv * _ORACLE_RHOW) * float(r1_Dt_ice)
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
    T0 = _ORACLE_RT0
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
    T0 = _ORACLE_RT0
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
    assert compiled.a_ip.devices().pop().platform == "cpu"
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


_CORE_SINGLE_LINE_SITES: dict[str, tuple[str, str]] = {
    "lev_zdv_pnd_old": (
        "zdv_pnd_old = sr(sr(h_ip + h_il) * a_ip)",
        "zdv_pnd_old = ((h_ip + h_il) * a_ip)",
    ),
    "lev_v_il_init": (
        "v_il = sr(h_il * a_ip)",
        "v_il = (h_il * a_ip)",
    ),
    "lev_melt_mass": (
        "melt_mass = sr(sr(dh_i_sum * rhoi) + sr(dh_s_sum * rhos))",
        "melt_mass = ((dh_i_sum * rhoi) + (dh_s_sum * rhos))",
    ),
    "lev_zfr_mlt": (
        "zfr_mlt = sr(rn_apnd_min + sr(sr(rn_apnd_max - rn_apnd_min) * at_i))",
        "zfr_mlt = (rn_apnd_min + ((rn_apnd_max - rn_apnd_min) * at_i))",
    ),
    "lev_zv_ip_max_area": (
        "zv_ip_max_area = sr(sr(sr(zfr_mlt * zfr_mlt) * a_i) * zaspect)",
        "zv_ip_max_area = (((zfr_mlt * zfr_mlt) * a_i) * zaspect)",
    ),
    "lev_zv_ip_max_depth": (
        "zv_ip_max_depth = sr(sr(sr(sr(z1_aspect * a_i) * depth_cap_factor) * h_i) * h_i)",
        "zv_ip_max_depth = ((((z1_aspect * a_i) * depth_cap_factor) * h_i) * h_i)",
    ),
    "lev_lid_freeze_quad": (
        "v_il_sq = sr(v_il * v_il)",
        "v_il_sq = (v_il * v_il)",
    ),
    "lev_head_and_darcy": (
        "t3_hp = sr(h_ip_1 * sr(rho0 - t1_hp))",
        "t3_hp = (h_ip_1 * (rho0 - t1_hp))",
    ),
    "lev_zdv_pnd_out": (
        "zdv_pnd = sr(sr(sr(h_ip_out + h_il_out) * a_ip_out) - zdv_pnd_old)",
        "zdv_pnd = (((h_ip_out + h_il_out) * a_ip_out) - zdv_pnd_old)",
    ),
    "lev_wfx_pnd_out": (
        "wfx_pnd_out = sr(wfx_pnd - sr(sr(zdv_pnd * rhow) * r1_Dt_ice))",
        "wfx_pnd_out = (wfx_pnd - ((zdv_pnd * rhow) * r1_Dt_ice))",
    ),
}


@pytest.mark.parametrize(
    "site",
    [
        "ice_perm_eff",
        "perm_brine_poly",
        "pnd_lev_core",
        *_CORE_SINGLE_LINE_SITES.keys(),
        "ice_thd_pnd",
        "thd_pnd_vanished_flux",
    ],
)
def test_real_model_barrier_groups_have_bit_changing_stripped_plants(site, monkeypatch):
    """Stripping nemo_source_round from each arithmetic site/group changes bits under jit with runtime args."""
    rng = np.random.default_rng(20261026)
    T0 = _ORACLE_RT0

    if site in ("ice_perm_eff", "perm_brine_poly"):
        t_i = jnp.asarray(T0 - rng.uniform(0.15, 1.5, (128, 10)), dtype=jnp.float64)
        sz_i = jnp.asarray(rng.uniform(3.0, 15.0, (128, 10)), dtype=jnp.float64)
        fn = lambda t, s: pnd_mod.si3_ice_perm_eff(
            t, s, constants=NEMO_SI3_CONSTANTS_CONFIG
        )
        if site == "ice_perm_eff":
            stripped_perm = _without_source_rounding(pnd_mod.si3_ice_perm_eff)
        else:
            stripped_perm = _source_mutant(
                pnd_mod.si3_ice_perm_eff,
                "term1 = -sr(c1 * zt1)\n"
                "    term2 = sr(c2 * zt2)\n"
                "    term3 = sr(c3 * zt3)\n"
                "    zs_br = sr(sr(term1 - term2) - term3)",
                "term1 = -(c1 * zt1)\n"
                "    term2 = c2 * (zt1 * zt1)\n"
                "    term3 = c3 * ((zt1 * zt1) * zt1)\n"
                "    zs_br = (term1 - term2) - term3",
            )
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

    if site == "pnd_lev_core" or site in _CORE_SINGLE_LINE_SITES:
        n = 128
        core_arrays = (
            jnp.asarray(rng.uniform(0.3, 0.95, n), dtype=jnp.float64),  # at_i
            jnp.asarray(rng.uniform(-1.0e-5, 1.0e-5, n), dtype=jnp.float64),  # wfx_pnd
            jnp.asarray(rng.uniform(0.15, 0.85, n), dtype=jnp.float64),  # a_i
            jnp.asarray(rng.uniform(0.3, 2.5, n), dtype=jnp.float64),  # h_i
            jnp.asarray(T0 - rng.uniform(0.2, 6.0, n), dtype=jnp.float64),  # t_su
            jnp.asarray(rng.uniform(0.03, 0.25, n), dtype=jnp.float64),  # a_ip
            jnp.asarray(rng.uniform(0.03, 0.20, n), dtype=jnp.float64),  # h_ip
            jnp.asarray(rng.uniform(0.005, 0.04, n), dtype=jnp.float64),  # h_il
            jnp.asarray(-rng.uniform(0.001, 0.05, n), dtype=jnp.float64),  # dh_i_sum
            jnp.asarray(-rng.uniform(0.001, 0.03, n), dtype=jnp.float64),  # dh_s_sum
            jnp.asarray(rng.uniform(1.5, 4.0, (n, 10)), dtype=jnp.float64),  # sz_i
            jnp.asarray(T0 - rng.uniform(1.5, 3.5, (n, 10)), dtype=jnp.float64),  # t_i
            jnp.asarray(RDT, dtype=jnp.float64),  # rDt_ice
            jnp.asarray(R1_RDT, dtype=jnp.float64),  # r1_Dt_ice
            jnp.asarray(0.1, dtype=jnp.float64),  # rn_himin
            jnp.asarray(0.15, dtype=jnp.float64),  # rn_apnd_min
            jnp.asarray(0.80, dtype=jnp.float64),  # rn_apnd_max
            jnp.asarray(0.1, dtype=jnp.float64),  # rn_pnd_flush
            jnp.asarray(RHO_W, dtype=jnp.float64),  # rhow
            jnp.asarray(GRAVITY, dtype=jnp.float64),  # grav
            jnp.asarray(1.0 / RHO_W, dtype=jnp.float64),  # z1_rhow
            jnp.asarray(1.0 / 0.8, dtype=jnp.float64),  # z1_aspect
            jnp.asarray(1.0 / -0.15, dtype=jnp.float64),  # z1_Tp
        )

        def _wrap_core(core_fn):
            def _runner(
                at_i_a,
                wfx_a,
                ai_a,
                hi_a,
                tsu_a,
                aip_a,
                hip_a,
                hil_a,
                dhi_a,
                dhs_a,
                szi_a,
                ti_a,
                rdt_a,
                r1dt_a,
                himin_a,
                apmin_a,
                apmax_a,
                flush_a,
                rhow_a,
                grav_a,
                z1rhow_a,
                z1asp_a,
                z1tp_a,
            ):
                return core_fn(
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
                    rDt_ice=rdt_a,
                    r1_Dt_ice=r1dt_a,
                    rn_himin=himin_a,
                    rn_apnd_min=apmin_a,
                    rn_apnd_max=apmax_a,
                    rn_pnd_flush=flush_a,
                    ln_pnd_lids=True,
                    rhow=rhow_a,
                    grav=grav_a,
                    constants=NEMO_SI3_CONSTANTS_CONFIG,
                    z1_rhow=z1rhow_a,
                    z1_aspect=z1asp_a,
                    z1_Tp=z1tp_a,
                )

            return _runner

        if site == "pnd_lev_core":
            stripped_core = _without_source_rounding(pnd_mod._pnd_lev_single_category)
        else:
            old_line, new_line = _CORE_SINGLE_LINE_SITES[site]
            stripped_core = _source_mutant(
                pnd_mod._pnd_lev_single_category,
                old_line,
                new_line,
            )

        run_protected = _wrap_core(pnd_mod._pnd_lev_single_category)
        run_stripped = _wrap_core(stripped_core)
        with jax.disable_jit():
            eager = run_protected(*core_arrays)
        protected = jax.jit(run_protected)(*core_arrays)
        stripped = jax.jit(run_stripped)(*core_arrays)
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
        if site == "thd_pnd_vanished_flux":
            monkeypatch.setattr(
                pnd_mod,
                "si3_ice_thd_pnd",
                _source_mutant(
                    pnd_mod.si3_ice_thd_pnd,
                    "flux_inc = sr(sr(sr(v_ip_int[..., jl] + v_il_int[..., jl]) * rhow) * r1_dt)\n"
                    "        wfx_int = jnp.where(vanished, sr(wfx_int + flux_inc), wfx_int)",
                    "flux_inc = ((v_ip_int[..., jl] + v_il_int[..., jl]) * rhow) * r1_dt\n"
                    "        wfx_int = jnp.where(vanished, wfx_int + flux_inc, wfx_int)",
                ),
            )
        else:
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
    T0 = _ORACLE_RT0
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
    """Verify active _maxsd/_minsd ±0 tie-survivor bits AND the 456/478 checked masking argument."""
    T0 = _ORACLE_RT0
    pos_zero = jnp.asarray([0.0, -0.0], dtype=jnp.float64)
    neg_zero = jnp.asarray([-0.0, 0.0], dtype=jnp.float64)
    sign_bit = np.uint64(1 << 63)

    # 1. Active signed-zero tie-survivor test on _maxsd and _minsd under jax.jit with runtime args:
    # x86 maxsd/minsd return src (second argument) on a (+0.0, -0.0) or (-0.0, +0.0) tie.
    jit_maxsd = jax.jit(pnd_mod._maxsd)
    jit_minsd = jax.jit(pnd_mod._minsd)
    got_max = _bits(jit_maxsd(pos_zero, neg_zero))
    got_min = _bits(jit_minsd(pos_zero, neg_zero))
    np.testing.assert_array_equal(got_max, np.array([sign_bit, np.uint64(0)], dtype=np.uint64))
    np.testing.assert_array_equal(got_min, np.array([sign_bit, np.uint64(0)], dtype=np.uint64))
    # Planted mutant: swapping dst/src on _maxsd or _minsd flips the output sign bits on ±0 ties!
    swapped_max = _bits(jax.jit(lambda d, s: pnd_mod._maxsd(s, d))(pos_zero, neg_zero))
    swapped_min = _bits(jax.jit(lambda d, s: pnd_mod._minsd(s, d))(pos_zero, neg_zero))
    assert np.all(swapped_max != got_max)
    assert np.all(swapped_min != got_min)

    # 2. Pre-mask intermediate sign-bit verification at icethd_pnd.f90:456 and :478:
    # At :456, when dh_i_sum = +0.0, dh_s_sum = +0.0, zdv_avail = -0.0 so zfr_mlt * zdv_avail = -0.0.
    # Unswapped _maxsd(dst=-0.0, src=+0.0) produces +0.0 (0x0); swapped _maxsd(dst=+0.0, src=-0.0)
    # produces -0.0 (0x8000000000000000) before :465/:472 mask it.
    pre_mask_456 = jax.jit(lambda zfr, avail, z: pnd_mod._maxsd(pnd_mod.nemo_source_round(zfr * avail), z))(
        jnp.asarray([0.5], dtype=jnp.float64),
        jnp.asarray([-0.0], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
    )
    pre_mask_456_swapped = jax.jit(lambda zfr, avail, z: pnd_mod._maxsd(z, pnd_mod.nemo_source_round(zfr * avail)))(
        jnp.asarray([0.5], dtype=jnp.float64),
        jnp.asarray([-0.0], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
    )
    assert int(_bits(pre_mask_456)[0]) == 0
    assert int(_bits(pre_mask_456_swapped)[0]) == int(sign_bit)

    # 3. Full pnd_LEV checked masking verification with runtime array arguments to jax.jit:
    lev_args = (
        jnp.asarray([0.8, 0.8], dtype=jnp.float64),  # at_i
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),  # wfx_pnd
        jnp.asarray([0.8, 0.8], dtype=jnp.float64),  # a_i
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),  # h_i
        jnp.asarray([T0, T0], dtype=jnp.float64),  # t_su
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),  # a_ip
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),  # h_ip
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),  # h_il
        jnp.asarray([0.0, -0.0], dtype=jnp.float64),  # dh_i_sum
        jnp.asarray([0.0, -0.0], dtype=jnp.float64),  # dh_s_sum
        jnp.full((2, 10), 5.0, dtype=jnp.float64),  # sz_i
        jnp.full((2, 10), T0 - 2.0, dtype=jnp.float64),  # t_i
    )

    def _run_lev(at_i_a, wfx_a, ai_a, hi_a, tsu_a, aip_a, hip_a, hil_a, dhi_a, dhs_a, szi_a, ti_a):
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
            rn_himin=0.1,
            rn_apnd_min=0.15,
            rn_apnd_max=0.80,
            rn_pnd_flush=0.1,
            ln_pnd_lids=True,
            rho_w=RHO_W,
            gravity=GRAVITY,
            constants=NEMO_SI3_CONSTANTS_CONFIG,
        )

    out = jax.jit(_run_lev)(*lev_args)
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
    out_flipped = jax.jit(_run_lev)(*lev_args)
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

    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        # Compile single-category O10 once on CPU and reuse across all 5 categories
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


@pytest.mark.parametrize(
    ("const_field", "bad_factor"),
    [
        ("T0", 1.0001),
        ("rho_ocean", 1.001),
        ("rho_ice", 1.001),
        ("rho_snow", 1.001),
        ("latent_fusion", 1.001),
        ("k_ice", 1.001),
    ],
)
def test_independent_oracle_catches_model_constant_defect(const_field, bad_factor):
    """Independent oracle shares no model constants, so a model constant defect fails."""
    fx = _pnd_2d_fixture()
    oracle = _oracle_ice_thd_pnd(
        **fx,
        namelist=DECK_PND_NML,
        rDt_ice=RDT,
        r1_Dt_ice=R1_RDT,
        nn_hls=2,
    )
    bad_constants = NEMO_SI3_CONSTANTS_CONFIG._replace(
        **{const_field: getattr(NEMO_SI3_CONSTANTS_CONFIG, const_field) * bad_factor}
    )
    args = _pnd_array_args()
    arrays = dict(zip(_PND_ARRAY_NAMES, args, strict=True))
    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        mutated = pnd_mod.si3_ice_thd_pnd(
            **arrays,
            namelist=DECK_PND_NML,
            rDt_ice=RDT,
            r1_Dt_ice=R1_RDT,
            rho_w=RHO_W,
            gravity=GRAVITY,
            constants=bad_constants,
            nn_hls=2,
        )
    mismatches = sum(
        np.count_nonzero(_bits(got) != _bits(want))
        for got, want in zip(mutated, oracle, strict=True)
    )
    assert mismatches > 0, f"constant defect in {const_field} was not caught by independent oracle"


def test_production_step_sea_ice_and_step_ponds_wire_si3_lev_and_preserve_cesm(monkeypatch):
    """Verify production step_sea_ice (sea_ice.py:2423) and step_ponds (ponds.py:31) wire si3_lev."""
    namthd_pnd = SI3PondConfig(
        rn_himin=float(DECK_PND_NML["rn_himin"]),
        rn_apnd_min=float(DECK_PND_NML["rn_apnd_min"]),
        rn_apnd_max=float(DECK_PND_NML["rn_apnd_max"]),
        rn_pnd_flush=float(DECK_PND_NML["rn_pnd_flush"]),
        ln_pnd_lids=bool(DECK_PND_NML["ln_pnd_lids"]),
    )
    shape = (2, 2)
    t_init = SeaIceConfig().T_freeze_ocean
    base_state = init_dynamic_ice_state(shape, S_ice_init=2.5, T_ice_init=t_init)
    state = base_state._replace(
        h_ice=base_state.h_ice.replace(data=jnp.full(shape, 1.4, dtype=jnp.float64)),
        T_ice=base_state.T_ice.replace(data=jnp.full(shape, t_init, dtype=jnp.float64)),
        concentration=base_state.concentration.replace(data=jnp.full(shape, 0.80, dtype=jnp.float64)),
        h_snow=base_state.h_snow.replace(data=jnp.full(shape, 0.05, dtype=jnp.float64)),
        S_ice=base_state.S_ice.replace(data=jnp.full(shape, 2.5, dtype=jnp.float64)),
        pond_area=base_state.pond_area.replace(data=jnp.full(shape, 0.15, dtype=jnp.float64)),
        pond_depth=base_state.pond_depth.replace(data=jnp.full(shape, 0.12, dtype=jnp.float64)),
    )
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    forcing = AtmToSurface(
        sw_down=zeros,
        lw_down=zeros,
        precip_total=zeros,
        precip_snow=zeros,
        T_lowest=jnp.full(shape, t_init, dtype=jnp.float64),
        q_lowest=zeros,
        u_lowest=zeros,
        v_lowest=zeros,
        p_lowest=jnp.full(shape, 100000.0, dtype=jnp.float64),
        p_surface=jnp.full(shape, 101325.0, dtype=jnp.float64),
        rho_lowest=jnp.full(shape, 1.25, dtype=jnp.float64),
        cos_zenith=zeros,
        co2_ppmv=jnp.full(shape, 400.0, dtype=jnp.float64),
        has_radiation=zeros,
        has_precipitation=zeros,
        z_lowest=jnp.full(shape, 10.0, dtype=jnp.float64),
    )
    ocean_sst = jnp.full(shape, t_init, dtype=jnp.float64)
    cfg_si3 = SeaIceConfig(
        emissivity_ice=0.0,
        Cd_ice=0.0,
        Ch_ice=0.0,
        Ce_ice=0.0,
        ocean_heat_transfer_coeff=0.0,
        snow=SnowConfig(enabled=True, flooding=False),
        ponds=MeltPondConfig(enabled=True, scheme="si3_lev", namthd_pnd=namthd_pnd),
        ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
    )

    def _run_step(st, frc, sst, u_ocn, v_ocn):
        return sea_ice_mod.step_sea_ice(
            st, frc, sst, u_ocn, v_ocn, cfg_si3, U_min=1.0, dt=RDT
        )

    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        with jax.disable_jit():
            state_eager, resp_eager = _run_step(state, forcing, ocean_sst, zeros, zeros)
        state_jit, resp_jit = jax.jit(_run_step)(state, forcing, ocean_sst, zeros, zeros)

    conc_out = float(state_jit.concentration.data[0, 0])
    oracle_step = _oracle_pnd_lev_category(
        at_i=conc_out,
        a_i=conc_out,
        h_i=float(state_jit.h_ice.data[0, 0]),
        t_su=float(state_jit.T_ice.data[0, 0]),
        a_ip=0.15 * conc_out,
        h_ip=0.12 * (0.80 / conc_out),
        h_il=0.0,
        dh_i_sum=0.0,
        dh_s_sum=0.0,
        sz_i=[2.5],
        t_i=[float(t_init)],
        wfx_pnd=0.0,
        rDt_ice=RDT,
        r1_Dt_ice=R1_RDT,
        rn_himin=namthd_pnd.rn_himin,
        rn_apnd_min=namthd_pnd.rn_apnd_min,
        rn_apnd_max=namthd_pnd.rn_apnd_max,
        rn_pnd_flush=namthd_pnd.rn_pnd_flush,
        ln_pnd_lids=namthd_pnd.ln_pnd_lids,
    )
    want_step_pa = oracle_step[0] / conc_out
    assert oracle_step[2] > 0.0
    np.testing.assert_array_equal(_bits(state_eager.pond_area.data), _bits(np.full(shape, want_step_pa)))
    np.testing.assert_array_equal(_bits(state_jit.pond_area.data), _bits(np.full(shape, want_step_pa)))
    np.testing.assert_array_equal(_bits(state_eager.pond_depth.data), _bits(np.full(shape, oracle_step[1])))
    np.testing.assert_array_equal(_bits(state_jit.pond_depth.data), _bits(np.full(shape, oracle_step[1])))
    np.testing.assert_array_equal(_bits(state_eager.h_il.data), _bits(np.full(shape, oracle_step[2])))
    np.testing.assert_array_equal(_bits(state_jit.h_il.data), _bits(np.full(shape, oracle_step[2])))
    np.testing.assert_array_equal(_bits(resp_eager.freshwater_flux), _bits(resp_jit.freshwater_flux))

    # Direct step_ponds(scheme="si3_lev") vs independent scalar oracle _oracle_pnd_lev_category
    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        pa_new, pd_new, drain_m, refreeze_m, hil_new = jax.jit(
            lambda pa, pd, hil, ai, hi, tsu, dhi, dhs, szi, ti: ponds_mod.step_ponds(
                pa,
                pd,
                melt_water_m=zeros,
                rain_water_m=zeros,
                ice_mask=ai > 0.0,
                h_snow=zeros,
                T_air=tsu,
                dt=RDT,
                drainage_timescale=86400.0,
                refreeze_threshold=_ORACLE_RT0,
                pond_to_ice_max_area=0.6,
                depth_to_area_ratio=0.8,
                scheme="si3_lev",
                at_i=ai,
                a_i=ai,
                h_i=hi,
                t_su=tsu,
                h_il=hil,
                dh_i_sum=dhi,
                dh_s_sum=dhs,
                sz_i=szi,
                t_i=ti,
                namthd_pnd=namthd_pnd,
                rho_w=RHO_W,
                gravity=GRAVITY,
                ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
            )
        )(
            state.pond_area.data,
            state.pond_depth.data,
            jnp.full(shape, 0.008, dtype=jnp.float64),
            state.concentration.data,
            state.h_ice.data,
            jnp.full(shape, _ORACLE_RT0, dtype=jnp.float64),
            jnp.full(shape, -0.012, dtype=jnp.float64),
            jnp.full(shape, -0.004, dtype=jnp.float64),
            state.S_ice.data,
            state.T_ice.data,
        )
    oracle_cat = _oracle_pnd_lev_category(
        at_i=0.80,
        a_i=0.80,
        h_i=1.4,
        t_su=_ORACLE_RT0,
        a_ip=0.15 * 0.80,
        h_ip=0.12,
        h_il=0.008,
        dh_i_sum=-0.012,
        dh_s_sum=-0.004,
        sz_i=[2.5],
        t_i=[float(t_init)],
        wfx_pnd=0.0,
        rDt_ice=RDT,
        r1_Dt_ice=R1_RDT,
        rn_himin=namthd_pnd.rn_himin,
        rn_apnd_min=namthd_pnd.rn_apnd_min,
        rn_apnd_max=namthd_pnd.rn_apnd_max,
        rn_pnd_flush=namthd_pnd.rn_pnd_flush,
        ln_pnd_lids=namthd_pnd.ln_pnd_lids,
    )
    want_pa = oracle_cat[0] / 0.80
    want_pd = oracle_cat[1]
    want_hil = oracle_cat[2]
    want_drain_m = (oracle_cat[5] * RDT / _ORACLE_RHOW) / 0.80
    np.testing.assert_array_equal(_bits(pa_new), _bits(np.full(shape, want_pa)))
    np.testing.assert_array_equal(_bits(pd_new), _bits(np.full(shape, want_pd)))
    np.testing.assert_array_equal(_bits(hil_new), _bits(np.full(shape, want_hil)))
    np.testing.assert_array_equal(_bits(drain_m), _bits(np.full(shape, want_drain_m)))
    np.testing.assert_array_equal(_bits(refreeze_m), _bits(np.zeros(shape)))

    # Planted mutant 1: removing step_ponds call in sea_ice.py:2423 must fail on step_sea_ice!
    monkeypatch.setattr(
        sea_ice_mod,
        "step_ponds",
        lambda pa, pd, **kw: (pa, pd, jnp.zeros_like(pd), jnp.zeros_like(pd), kw["h_il"]),
    )
    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        state_no_call, resp_no_call = jax.jit(
            lambda st, frc, sst, u_ocn, v_ocn: _run_step(st, frc, sst, u_ocn, v_ocn)
        )(state, forcing, ocean_sst, zeros, zeros)
    assert np.count_nonzero(_bits(state_no_call.pond_area.data) != _bits(state_jit.pond_area.data)) > 0
    assert np.count_nonzero(_bits(state_no_call.pond_depth.data) != _bits(state_jit.pond_depth.data)) > 0
    assert np.count_nonzero(_bits(state_no_call.h_il.data) != _bits(state_jit.h_il.data)) > 0
    assert np.count_nonzero(_bits(resp_no_call.freshwater_flux) != _bits(resp_jit.freshwater_flux)) > 0
    monkeypatch.undo()

    # Planted mutant 2: ignoring scheme="si3_lev" and running the CESM pond branch must fail!
    orig_step_ponds = ponds_mod.step_ponds
    monkeypatch.setattr(
        sea_ice_mod,
        "step_ponds",
        lambda *a, **kw: (*orig_step_ponds(*a, **{**kw, "scheme": "cesm", "namthd_pnd": None}), kw["h_il"]),
    )
    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        state_cesm_mut, resp_cesm_mut = jax.jit(
            lambda st, frc, sst, u_ocn, v_ocn: _run_step(st, frc, sst, u_ocn, v_ocn)
        )(state, forcing, ocean_sst, zeros, zeros)
    assert np.count_nonzero(_bits(state_cesm_mut.pond_area.data) != _bits(state_jit.pond_area.data)) > 0
    assert np.count_nonzero(_bits(state_cesm_mut.pond_depth.data) != _bits(state_jit.pond_depth.data)) > 0
    assert np.count_nonzero(_bits(state_cesm_mut.h_il.data) != _bits(state_jit.h_il.data)) > 0
    assert np.count_nonzero(_bits(resp_cesm_mut.freshwater_flux) != _bits(resp_jit.freshwater_flux)) > 0
    monkeypatch.undo()

    # Check that default scheme="cesm" is bit-identical to explicit scheme="cesm"
    cfg_cesm_default = SeaIceConfig(
        snow=SnowConfig(enabled=True, flooding=False),
        ponds=MeltPondConfig(enabled=True),
    )
    cfg_cesm_explicit = SeaIceConfig(
        snow=SnowConfig(enabled=True, flooding=False),
        ponds=MeltPondConfig(enabled=True, scheme="cesm"),
    )
    st_def, rsp_def = sea_ice_mod.step_sea_ice(
        state, forcing, ocean_sst, zeros, zeros, cfg_cesm_default, U_min=1.0, dt=RDT
    )
    st_exp, rsp_exp = sea_ice_mod.step_sea_ice(
        state, forcing, ocean_sst, zeros, zeros, cfg_cesm_explicit, U_min=1.0, dt=RDT
    )
    for got, want in zip(jax.tree.leaves(st_def), jax.tree.leaves(st_exp), strict=True):
        np.testing.assert_array_equal(_bits(got), _bits(want))
    for got, want in zip(jax.tree.leaves(rsp_def), jax.tree.leaves(rsp_exp), strict=True):
        np.testing.assert_array_equal(_bits(got), _bits(want))

    # Validation guards on step_ponds and DynamicSeaIceState.h_il
    with pytest.raises(ValueError, match="requires MeltPondConfig.namthd_pnd"):
        sea_ice_mod.step_sea_ice(
            state,
            forcing,
            ocean_sst,
            zeros,
            zeros,
            SeaIceConfig(ponds=MeltPondConfig(enabled=True, scheme="si3_lev")),
            U_min=1.0,
            dt=RDT,
        )
    with pytest.raises(ValueError, match="does not accept namthd_pnd"):
        sea_ice_mod.step_sea_ice(
            state,
            forcing,
            ocean_sst,
            zeros,
            zeros,
            SeaIceConfig(ponds=MeltPondConfig(enabled=True, scheme="cesm", namthd_pnd=namthd_pnd)),
            U_min=1.0,
            dt=RDT,
        )
    with pytest.raises(ValueError, match="DynamicSeaIceState.h_il must be a Field"):
        sea_ice_mod.step_sea_ice(
            state._replace(h_il=None),
            forcing,
            ocean_sst,
            zeros,
            zeros,
            cfg_si3,
            U_min=1.0,
            dt=RDT,
        )


def test_lid_thickness_h_il_persists_across_steps_and_restart_round_trip(tmp_path, monkeypatch):
    """Verify prognostic h_il is carried in DynamicSeaIceState across steps and run restart."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.restart import load_run_restart, save_run_restart
    from legoesm.ocean.vertical import create_ocean_z_star

    namthd_pnd = SI3PondConfig(
        rn_himin=float(DECK_PND_NML["rn_himin"]),
        rn_apnd_min=float(DECK_PND_NML["rn_apnd_min"]),
        rn_apnd_max=float(DECK_PND_NML["rn_apnd_max"]),
        rn_pnd_flush=float(DECK_PND_NML["rn_pnd_flush"]),
        ln_pnd_lids=bool(DECK_PND_NML["ln_pnd_lids"]),
    )
    shape = (2, 2)
    t_cold = SeaIceConfig().T_freeze_ocean  # 271.35 K (< T0, grows lid via Flocco 2010)
    base_state = init_dynamic_ice_state(shape, S_ice_init=2.5, T_ice_init=t_cold)
    # Four distinct cells:
    # (0, 0): starts at h_il=0.0, grows lid on step 1, evolves from that lid on step 2
    # (0, 1): starts with existing lid h_il=0.015 m, insulates pond on step 1 and step 2
    # (1, 0): starts with thicker lid h_il=0.025 m and pond depth h_ip=0.16 m
    # (1, 1): starts with moderate lid h_il=0.006 m and deeper pond h_ip=0.18 m
    pa_0 = np.array([[0.15, 0.22], [0.20, 0.25]], dtype=np.float64)
    pd_0 = np.array([[0.12, 0.14], [0.16, 0.18]], dtype=np.float64)
    hil_0 = np.array([[0.00, 0.015], [0.025, 0.006]], dtype=np.float64)
    ai_0 = np.array([[0.50, 0.75], [0.85, 0.90]], dtype=np.float64)
    hi_0 = np.array([[1.60, 1.60], [1.80, 2.10]], dtype=np.float64)
    state_0 = base_state._replace(
        h_ice=base_state.h_ice.replace(data=jnp.asarray(hi_0)),
        T_ice=base_state.T_ice.replace(data=jnp.full(shape, t_cold, dtype=jnp.float64)),
        concentration=base_state.concentration.replace(data=jnp.asarray(ai_0)),
        h_snow=base_state.h_snow.replace(data=jnp.full(shape, 0.02, dtype=jnp.float64)),
        S_ice=base_state.S_ice.replace(data=jnp.full(shape, 2.5, dtype=jnp.float64)),
        pond_area=base_state.pond_area.replace(data=jnp.asarray(pa_0)),
        pond_depth=base_state.pond_depth.replace(data=jnp.asarray(pd_0)),
        h_il=base_state.h_il.replace(data=jnp.asarray(hil_0)),
    )
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    forcing = AtmToSurface(
        sw_down=zeros,
        lw_down=zeros,
        precip_total=zeros,
        precip_snow=zeros,
        T_lowest=jnp.full(shape, t_cold, dtype=jnp.float64),
        q_lowest=zeros,
        u_lowest=zeros,
        v_lowest=zeros,
        p_lowest=jnp.full(shape, 100000.0, dtype=jnp.float64),
        p_surface=jnp.full(shape, 101325.0, dtype=jnp.float64),
        rho_lowest=jnp.full(shape, 1.25, dtype=jnp.float64),
        cos_zenith=zeros,
        co2_ppmv=jnp.full(shape, 400.0, dtype=jnp.float64),
        has_radiation=zeros,
        has_precipitation=zeros,
        z_lowest=jnp.full(shape, 10.0, dtype=jnp.float64),
    )
    ocean_sst = jnp.full(shape, t_cold, dtype=jnp.float64)
    cfg_si3 = SeaIceConfig(
        emissivity_ice=0.0,
        Cd_ice=0.0,
        Ch_ice=0.0,
        Ce_ice=0.0,
        ocean_heat_transfer_coeff=0.0,
        snow=SnowConfig(enabled=True, flooding=False),
        ponds=MeltPondConfig(enabled=True, scheme="si3_lev", namthd_pnd=namthd_pnd),
        ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
    )

    @jax.jit
    def _run_step(st, frc, sst, u_ocn, v_ocn):
        return sea_ice_mod.step_sea_ice(
            st, frc, sst, u_ocn, v_ocn, cfg_si3, U_min=1.0, dt=RDT
        )

    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        with jax.disable_jit():
            st1_eager, rsp1_eager = sea_ice_mod.step_sea_ice(
                state_0, forcing, ocean_sst, zeros, zeros, cfg_si3, U_min=1.0, dt=RDT
            )
            st2_eager, rsp2_eager = sea_ice_mod.step_sea_ice(
                st1_eager, forcing, ocean_sst, zeros, zeros, cfg_si3, U_min=1.0, dt=RDT
            )
        st1_jit, rsp1_jit = _run_step(state_0, forcing, ocean_sst, zeros, zeros)
        st2_jit, rsp2_jit = _run_step(st1_jit, forcing, ocean_sst, zeros, zeros)

    # Verify step 1 and step 2 against sequential scalar oracle _oracle_pnd_lev_category
    want_pa1 = np.zeros(shape, dtype=np.float64)
    want_pd1 = np.zeros(shape, dtype=np.float64)
    want_hil1 = np.zeros(shape, dtype=np.float64)
    want_fw1 = np.zeros(shape, dtype=np.float64)
    want_pa2 = np.zeros(shape, dtype=np.float64)
    want_pd2 = np.zeros(shape, dtype=np.float64)
    want_hil2 = np.zeros(shape, dtype=np.float64)
    want_fw2 = np.zeros(shape, dtype=np.float64)
    for j in range(shape[0]):
        for i in range(shape[1]):
            c1 = float(st1_jit.concentration.data[j, i])
            h1 = float(st1_jit.h_ice.data[j, i])
            t1 = float(st1_jit.T_ice.data[j, i])
            dil1 = float(ai_0[j, i]) / c1
            o1 = _oracle_pnd_lev_category(
                at_i=c1,
                a_i=c1,
                h_i=h1,
                t_su=t1,
                a_ip=float(pa_0[j, i]) * c1,
                h_ip=float(pd_0[j, i]) * dil1,
                h_il=float(hil_0[j, i]) * dil1,
                dh_i_sum=0.0,
                dh_s_sum=0.0,
                sz_i=[2.5],
                t_i=[float(t_cold)],
                wfx_pnd=0.0,
                rDt_ice=RDT,
                r1_Dt_ice=R1_RDT,
                rn_himin=namthd_pnd.rn_himin,
                rn_apnd_min=namthd_pnd.rn_apnd_min,
                rn_apnd_max=namthd_pnd.rn_apnd_max,
                rn_pnd_flush=namthd_pnd.rn_pnd_flush,
                ln_pnd_lids=namthd_pnd.ln_pnd_lids,
            )
            want_pa1[j, i] = o1[0] / c1
            want_pd1[j, i] = o1[1]
            want_hil1[j, i] = o1[2]
            want_fw1[j, i] = (((o1[5] * RDT / _ORACLE_RHOW) / c1) * _ORACLE_RHOW / RDT) * c1

            c2 = float(st2_jit.concentration.data[j, i])
            h2 = float(st2_jit.h_ice.data[j, i])
            t2 = float(st2_jit.T_ice.data[j, i])
            dil2 = c1 / c2
            o2 = _oracle_pnd_lev_category(
                at_i=c2,
                a_i=c2,
                h_i=h2,
                t_su=t2,
                a_ip=want_pa1[j, i] * c2,
                h_ip=want_pd1[j, i] * dil2,
                h_il=want_hil1[j, i] * dil2,
                dh_i_sum=0.0,
                dh_s_sum=0.0,
                sz_i=[2.5],
                t_i=[t1],
                wfx_pnd=0.0,
                rDt_ice=RDT,
                r1_Dt_ice=R1_RDT,
                rn_himin=namthd_pnd.rn_himin,
                rn_apnd_min=namthd_pnd.rn_apnd_min,
                rn_apnd_max=namthd_pnd.rn_apnd_max,
                rn_pnd_flush=namthd_pnd.rn_pnd_flush,
                ln_pnd_lids=namthd_pnd.ln_pnd_lids,
            )
            want_pa2[j, i] = o2[0] / c2
            want_pd2[j, i] = o2[1]
            want_hil2[j, i] = o2[2]
            want_fw2[j, i] = (((o2[5] * RDT / _ORACLE_RHOW) / c2) * _ORACLE_RHOW / RDT) * c2

    for st1, st2 in ((st1_eager, st2_eager), (st1_jit, st2_jit)):
        np.testing.assert_array_equal(_bits(st1.pond_area.data), _bits(want_pa1))
        np.testing.assert_array_equal(_bits(st1.pond_depth.data), _bits(want_pd1))
        np.testing.assert_array_equal(_bits(st1.h_il.data), _bits(want_hil1))
        np.testing.assert_array_equal(_bits(st2.pond_area.data), _bits(want_pa2))
        np.testing.assert_array_equal(_bits(st2.pond_depth.data), _bits(want_pd2))
        np.testing.assert_array_equal(_bits(st2.h_il.data), _bits(want_hil2))
    np.testing.assert_array_equal(_bits(rsp1_eager.freshwater_flux), _bits(want_fw1))
    np.testing.assert_array_equal(_bits(rsp2_eager.freshwater_flux), _bits(want_fw2))

    # Restart round-trip between step 1 and step 2 (save_run_restart / load_run_restart)
    grid = create_latlon_grid(n_lat=shape[0], n_lon=shape[1])
    z_coord = create_ocean_z_star(n_levels=2, H_max=100.0)
    ocn_state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=12.0, T_deep=2.0, S_uniform=35.0, H_max=100.0
    )
    rst_path = tmp_path / "lid_restart.npz"
    save_run_restart(
        rst_path,
        ocn_state,
        step=1,
        time_days=RDT / 86400.0,
        grid_type="latlon",
        ice_state=st1_jit,
    )
    _, got_ice, _ = load_run_restart(
        rst_path,
        ocn_state,
        ice_template=init_dynamic_ice_state(shape),
        grid_type="latlon",
    )
    np.testing.assert_array_equal(_bits(got_ice.h_il.data), _bits(st1_jit.h_il.data))
    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        st2_from_rst, rsp2_from_rst = _run_step(got_ice, forcing, ocean_sst, zeros, zeros)
    for got, want in zip(jax.tree.leaves(st2_from_rst), jax.tree.leaves(st2_jit), strict=True):
        np.testing.assert_array_equal(_bits(got), _bits(want))
    for got, want in zip(jax.tree.leaves(rsp2_from_rst), jax.tree.leaves(rsp2_jit), strict=True):
        np.testing.assert_array_equal(_bits(got), _bits(want))

    # Planted mutant (reproducing the pre-fix defect where h_il was reset to zero every step):
    orig_step_ponds = ponds_mod.step_ponds
    monkeypatch.setattr(
        sea_ice_mod,
        "step_ponds",
        lambda pa, pd, **kw: (
            *orig_step_ponds(pa, pd, **{**kw, "h_il": jnp.zeros_like(pd)})[:4],
            jnp.zeros_like(pd),
        ),
    )
    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        st1_reset, _ = jax.jit(
            lambda st, frc, sst, u_ocn, v_ocn: sea_ice_mod.step_sea_ice(
                st, frc, sst, u_ocn, v_ocn, cfg_si3, U_min=1.0, dt=RDT
            )
        )(state_0, forcing, ocean_sst, zeros, zeros)
        st2_reset, rsp2_reset = jax.jit(
            lambda st, frc, sst, u_ocn, v_ocn: sea_ice_mod.step_sea_ice(
                st, frc, sst, u_ocn, v_ocn, cfg_si3, U_min=1.0, dt=RDT
            )
        )(st1_reset, forcing, ocean_sst, zeros, zeros)
    assert np.count_nonzero(_bits(st2_reset.h_il.data) != _bits(st2_jit.h_il.data)) == shape[0] * shape[1]
    assert np.count_nonzero(_bits(st2_reset.pond_depth.data) != _bits(st2_jit.pond_depth.data)) == shape[0] * shape[1]
    assert np.count_nonzero(_bits(st2_reset.pond_area.data) != _bits(st2_jit.pond_area.data)) > 0
    assert np.count_nonzero(_bits(rsp2_reset.freshwater_flux) != _bits(rsp2_jit.freshwater_flux)) == shape[0] * shape[1]


def test_multicategory_jpl5_step_sea_ice_sums_at_i_over_categories(monkeypatch):
    """Verify multi-category (jpl=5) step_sea_ice computes at_i over categories (ice_var_agg), not spatial axis."""
    from legoesm.grids.latlon import create_latlon_grid

    namthd_pnd = SI3PondConfig(
        rn_himin=float(DECK_PND_NML["rn_himin"]),
        rn_apnd_min=float(DECK_PND_NML["rn_apnd_min"]),
        rn_apnd_max=float(DECK_PND_NML["rn_apnd_max"]),
        rn_pnd_flush=float(DECK_PND_NML["rn_pnd_flush"]),
        ln_pnd_lids=bool(DECK_PND_NML["ln_pnd_lids"]),
    )
    spatial = (2, 3)
    n_cat = 5
    shape_mc = (*spatial, n_cat)
    t_melt = SeaIceConfig().T_melt_surface  # 273.15 K
    t_init = SeaIceConfig().T_freeze_ocean  # 271.35 K (< T0 so zperm > 0 without flushing ponds dry)
    base_state = init_dynamic_ice_state(shape_mc, n_categories=n_cat, S_ice_init=2.5, T_ice_init=t_init)

    # Distinct per-category AND spatially varying concentrations conc[j, i, k]
    # with sum_k conc[j, i, k] <= 1.0 in every cell, while spatial sum along axis=1 differs in every cell.
    conc_mc = np.array(
        [
            [
                [0.0625, 0.125, 0.1875, 0.25, 0.125],
                [0.125, 0.0625, 0.125, 0.1875, 0.1875],
                [0.1875, 0.1875, 0.25, 0.125, 0.125],
            ],
            [
                [0.125, 0.125, 0.0625, 0.1875, 0.125],
                [0.1875, 0.125, 0.25, 0.1875, 0.1875],
                [0.0625, 0.1875, 0.1875, 0.125, 0.25],
            ],
        ],
        dtype=np.float64,
    )
    # Distinct per-category thicknesses inside the 5 category bins [0, 0.6, 1.4, 2.4, 3.6, inf)
    h_bins = np.array([0.451, 0.6588, 1.45, 2.4775, 3.65], dtype=np.float64)
    h_mc = np.broadcast_to(h_bins, shape_mc).copy()
    pa_mc = np.stack(
        [np.full(spatial, 0.08 + 0.02 * k, dtype=np.float64) for k in range(n_cat)],
        axis=-1,
    )
    pd_mc = np.stack(
        [np.full(spatial, 0.04 + 0.01 * k, dtype=np.float64) for k in range(n_cat)],
        axis=-1,
    )
    hil_mc = np.stack(
        [np.full(spatial, 0.25 + 0.02 * k, dtype=np.float64) for k in range(n_cat)],
        axis=-1,
    )
    sal_mc = np.stack(
        [np.full(spatial, 1.8 + 0.4 * k, dtype=np.float64) for k in range(n_cat)],
        axis=-1,
    )
    state_mc = base_state._replace(
        h_ice=base_state.h_ice.replace(data=jnp.asarray(h_mc)),
        T_ice=base_state.T_ice.replace(data=jnp.full(shape_mc, t_init, dtype=jnp.float64)),
        concentration=base_state.concentration.replace(data=jnp.asarray(conc_mc)),
        h_snow=base_state.h_snow.replace(data=jnp.full(shape_mc, 0.08, dtype=jnp.float64)),
        S_ice=base_state.S_ice.replace(data=jnp.asarray(sal_mc)),
        pond_area=base_state.pond_area.replace(data=jnp.asarray(pa_mc)),
        pond_depth=base_state.pond_depth.replace(data=jnp.asarray(pd_mc)),
        h_il=base_state.h_il.replace(data=jnp.asarray(hil_mc)),
    )
    zeros_2d = jnp.zeros(spatial, dtype=jnp.float64)
    forcing = AtmToSurface(
        sw_down=jnp.full(spatial, 1800.0, dtype=jnp.float64),
        lw_down=zeros_2d,
        precip_total=zeros_2d,
        precip_snow=zeros_2d,
        T_lowest=jnp.full(spatial, t_melt, dtype=jnp.float64),
        q_lowest=zeros_2d,
        u_lowest=zeros_2d,
        v_lowest=zeros_2d,
        p_lowest=jnp.full(spatial, 100000.0, dtype=jnp.float64),
        p_surface=jnp.full(spatial, 101325.0, dtype=jnp.float64),
        rho_lowest=jnp.full(spatial, 1.25, dtype=jnp.float64),
        cos_zenith=zeros_2d,
        co2_ppmv=jnp.full(spatial, 400.0, dtype=jnp.float64),
        has_radiation=jnp.ones(spatial, dtype=jnp.float64),
        has_precipitation=zeros_2d,
        z_lowest=jnp.full(spatial, 10.0, dtype=jnp.float64),
    )
    ocean_sst = jnp.full(spatial, t_melt, dtype=jnp.float64)
    cfg_mc = SeaIceConfig(
        n_categories=n_cat,
        itd_remap="lipscomb2001",
        T_freeze_ocean=t_melt,
        emissivity_ice=0.0,
        Cd_ice=0.0,
        Ch_ice=0.0,
        Ce_ice=0.0,
        ocean_heat_transfer_coeff=0.0,
        snow=SnowConfig(enabled=True, flooding=False),
        ponds=MeltPondConfig(enabled=True, scheme="si3_lev", namthd_pnd=namthd_pnd),
        ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
    )

    grid_mc = create_latlon_grid(n_lat=spatial[0], n_lon=spatial[1])

    @jax.jit
    def _run_mc(st, frc, sst, u_ocn, v_ocn):
        return sea_ice_mod.step_sea_ice(
            st, frc, sst, u_ocn, v_ocn, cfg_mc, U_min=1.0, dt=RDT, grid=grid_mc
        )

    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        with jax.disable_jit():
            st_eager, rsp_eager = sea_ice_mod.step_sea_ice(
                state_mc, forcing, ocean_sst, zeros_2d, zeros_2d, cfg_mc, U_min=1.0, dt=RDT, grid=grid_mc
            )
        st_jit, rsp_jit = _run_mc(state_mc, forcing, ocean_sst, zeros_2d, zeros_2d)

    # Independent scalar oracle at jpl=5 with Fortran left-to-right category sum at_i
    q_sfc = 1800.0 * (1.0 - cfg_mc.albedo_ice)
    want_pa = np.zeros(shape_mc, dtype=np.float64)
    want_pd = np.zeros(shape_mc, dtype=np.float64)
    want_hil = np.zeros(shape_mc, dtype=np.float64)
    want_fw_cat = np.zeros(shape_mc, dtype=np.float64)
    for j in range(spatial[0]):
        for i in range(spatial[1]):
            at_i_cell = 0.0
            for k in range(n_cat):
                at_i_cell = float(np.float64(at_i_cell) + np.float64(conc_mc[j, i, k]))
            for k in range(n_cat):
                ak = float(conc_mc[j, i, k])
                hk = float(h_mc[j, i, k])
                r_tot = (0.08 / cfg_mc.snow.k_snow) + (hk / cfg_mc.k_ice)
                k_cond = 1.0 / r_tot
                skin_cap = (cfg_mc.rho_ice * cfg_mc.c_ice * hk * 0.5) + (
                    cfg_mc.snow.rho_snow * cfg_mc.snow.c_snow * 0.08
                )
                cap_dt = skin_cap / RDT
                t_trial = ((cap_dt * t_init) + q_sfc + (k_cond * t_melt)) / (cap_dt + k_cond)
                e_pos = ((t_trial - t_melt) * (cap_dt + k_cond)) * RDT
                snow_melt_k = (e_pos / cfg_mc.L_f) / cfg_mc.snow.rho_snow
                o_cat = _oracle_pnd_lev_category(
                    at_i=at_i_cell,
                    a_i=ak,
                    h_i=hk,
                    t_su=float(t_melt),
                    a_ip=float(pa_mc[j, i, k]) * ak,
                    h_ip=float(pd_mc[j, i, k]),
                    h_il=float(hil_mc[j, i, k]),
                    dh_i_sum=0.0,
                    dh_s_sum=-snow_melt_k,
                    sz_i=[float(sal_mc[j, i, k])],
                    t_i=[float(t_init)],
                    wfx_pnd=0.0,
                    rDt_ice=RDT,
                    r1_Dt_ice=R1_RDT,
                    rn_himin=namthd_pnd.rn_himin,
                    rn_apnd_min=namthd_pnd.rn_apnd_min,
                    rn_apnd_max=namthd_pnd.rn_apnd_max,
                    rn_pnd_flush=namthd_pnd.rn_pnd_flush,
                    ln_pnd_lids=namthd_pnd.ln_pnd_lids,
                )
                want_pa[j, i, k] = o_cat[0] / ak
                want_pd[j, i, k] = o_cat[1]
                want_hil[j, i, k] = o_cat[2]
                melt_fw_k = ((snow_melt_k * cfg_mc.snow.rho_snow) * ak) / RDT
                drain_fw_k = (((o_cat[5] * RDT / _ORACLE_RHOW) / ak) * _ORACLE_RHOW / RDT) * ak
                want_fw_cat[j, i, k] = melt_fw_k + drain_fw_k
    want_fw_mc = jnp.sum(jnp.asarray(want_fw_cat), axis=-1)

    for st_out in (st_eager, st_jit):
        np.testing.assert_array_equal(_bits(st_out.pond_area.data), _bits(want_pa))
        np.testing.assert_array_equal(_bits(st_out.pond_depth.data), _bits(want_pd))
        np.testing.assert_array_equal(_bits(st_out.h_il.data), _bits(want_hil))
    np.testing.assert_array_equal(_bits(rsp_eager.freshwater_flux), _bits(want_fw_mc))

    # Planted mutant: reproducing the pre-fix bug where _thermo_v2 summed conc_new along the last spatial axis!
    orig_step_ponds = ponds_mod.step_ponds
    monkeypatch.setattr(
        sea_ice_mod,
        "step_ponds",
        lambda pa, pd, **kw: orig_step_ponds(
            pa,
            pd,
            **{
                **kw,
                "at_i": jnp.broadcast_to(
                    jnp.sum(kw["a_i"], axis=-1, keepdims=True), kw["a_i"].shape
                ),
            },
        ),
    )
    with precision_scope(PrecisionPolicy.fp64(transcendentals="libm")):
        st_spatial_bug, rsp_spatial_bug = jax.jit(
            lambda st, frc, sst, u_ocn, v_ocn: sea_ice_mod.step_sea_ice(
                st, frc, sst, u_ocn, v_ocn, cfg_mc, U_min=1.0, dt=RDT, grid=grid_mc
            )
        )(state_mc, forcing, ocean_sst, zeros_2d, zeros_2d)
    assert np.count_nonzero(_bits(st_spatial_bug.pond_area.data) != _bits(st_jit.pond_area.data)) == math.prod(shape_mc)
    assert np.count_nonzero(_bits(st_spatial_bug.pond_depth.data) != _bits(st_jit.pond_depth.data)) == math.prod(shape_mc)
    assert np.count_nonzero(_bits(st_spatial_bug.h_il.data) != _bits(st_jit.h_il.data)) == np.count_nonzero(
        _bits(st_jit.h_il.data) != 0
    )
    assert np.count_nonzero(_bits(rsp_spatial_bug.freshwater_flux) != _bits(rsp_jit.freshwater_flux)) == math.prod(spatial)


def test_m6_parent_commit_vs_current_across_all_other_callers(monkeypatch):
    """M6 control: compare PARENT (e31236c2f6) vs current HEAD across every output of every other caller."""
    import subprocess
    import types
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ice.config import BrineConfig, RidgingConfig
    from legoesm.ice.state import (
        SeaIceState,
        distribute_dynamic_state_to_categories,
        slab_to_dynamic,
    )

    parent_rev = "e31236c2f6"
    repo_root = pathlib.Path(__file__).resolve().parents[3]

    def _load_parent_module(name: str, relpath: str):
        src = subprocess.check_output(
            ["git", "show", f"{parent_rev}:{relpath}"],
            cwd=repo_root,
            text=True,
        )
        mod = types.ModuleType(name)
        mod.__file__ = f"{parent_rev}:{relpath}"
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        return mod

    parent_state_mod = _load_parent_module("_parent_ice_state", "packages/ice/legoesm/ice/state.py")
    parent_ponds_mod = _load_parent_module("_parent_ice_ponds", "packages/ice/legoesm/ice/ponds.py")
    parent_sea_ice_mod = _load_parent_module("_parent_ice_sea_ice", "packages/ice/legoesm/ice/sea_ice.py")
    # Wire parent sea_ice to parent state and parent ponds so the parent call graph is 100% parent code
    parent_sea_ice_mod.step_ponds = parent_ponds_mod.step_ponds
    parent_sea_ice_mod.DynamicSeaIceState = parent_state_mod.DynamicSeaIceState
    parent_sea_ice_mod.SeaIceState = parent_state_mod.SeaIceState
    parent_sea_ice_mod.init_dynamic_ice_state = parent_state_mod.init_dynamic_ice_state
    parent_sea_ice_mod.dynamic_to_slab = parent_state_mod.dynamic_to_slab
    parent_sea_ice_mod.slab_to_dynamic = parent_state_mod.slab_to_dynamic

    shape_2d = (3, 4)
    rng = np.random.default_rng(20261027)
    pa_in = jnp.asarray(0.05 + 0.25 * rng.random(shape_2d), dtype=jnp.float64)
    pd_in = jnp.asarray(0.02 + 0.15 * rng.random(shape_2d), dtype=jnp.float64)
    melt_in = jnp.asarray(0.005 * rng.random(shape_2d), dtype=jnp.float64)
    rain_in = jnp.asarray(0.002 * rng.random(shape_2d), dtype=jnp.float64)
    mask_in = jnp.asarray([[True, True, False, True], [True, False, True, True], [True, True, True, False]])
    hsnow_in = jnp.asarray(0.04 * rng.random(shape_2d), dtype=jnp.float64)
    tair_in = jnp.asarray(268.0 + 6.0 * rng.random(shape_2d), dtype=jnp.float64)

    total_arrays = 0
    total_elements = 0
    unequal_elements = 0

    def _compare_pair(got_arr, want_arr):
        nonlocal total_arrays, total_elements, unequal_elements
        g = _bits(got_arr)
        w = _bits(want_arr)
        assert g.shape == w.shape
        total_arrays += 1
        total_elements += int(g.size)
        unequal_elements += int(np.count_nonzero(g != w))

    # Caller 1: direct step_ponds (default CESM scheme)
    p_def = MeltPondConfig()
    got_ponds = ponds_mod.step_ponds(
        pa_in,
        pd_in,
        melt_water_m=melt_in,
        rain_water_m=rain_in,
        ice_mask=mask_in,
        h_snow=hsnow_in,
        T_air=tair_in,
        dt=RDT,
        drainage_timescale=p_def.drainage_timescale_s,
        refreeze_threshold=p_def.refreeze_threshold,
        pond_to_ice_max_area=p_def.pond_to_ice_max_area,
        depth_to_area_ratio=p_def.depth_to_area_ratio,
    )
    want_ponds = parent_ponds_mod.step_ponds(
        pa_in,
        pd_in,
        melt_water_m=melt_in,
        rain_water_m=rain_in,
        ice_mask=mask_in,
        h_snow=hsnow_in,
        T_air=tair_in,
        dt=RDT,
        drainage_timescale=p_def.drainage_timescale_s,
        refreeze_threshold=p_def.refreeze_threshold,
        pond_to_ice_max_area=p_def.pond_to_ice_max_area,
        depth_to_area_ratio=p_def.depth_to_area_ratio,
    )
    assert len(got_ponds) == 4 and len(want_ponds) == 4
    for g, w in zip(got_ponds, want_ponds, strict=True):
        _compare_pair(g, w)

    # Shared forcing and ocean inputs for step_sea_ice callers
    forcing = AtmToSurface(
        sw_down=jnp.asarray(120.0 + 40.0 * rng.random(shape_2d), dtype=jnp.float64),
        lw_down=jnp.asarray(240.0 + 30.0 * rng.random(shape_2d), dtype=jnp.float64),
        precip_total=jnp.asarray(2.0e-5 * rng.random(shape_2d), dtype=jnp.float64),
        precip_snow=jnp.asarray(1.0e-5 * rng.random(shape_2d), dtype=jnp.float64),
        T_lowest=tair_in,
        q_lowest=jnp.asarray(0.002 + 0.001 * rng.random(shape_2d), dtype=jnp.float64),
        u_lowest=jnp.asarray(-3.0 + 6.0 * rng.random(shape_2d), dtype=jnp.float64),
        v_lowest=jnp.asarray(-2.0 + 4.0 * rng.random(shape_2d), dtype=jnp.float64),
        p_lowest=jnp.full(shape_2d, 100000.0, dtype=jnp.float64),
        p_surface=jnp.full(shape_2d, 101325.0, dtype=jnp.float64),
        rho_lowest=jnp.full(shape_2d, 1.25, dtype=jnp.float64),
        cos_zenith=jnp.full(shape_2d, 0.5, dtype=jnp.float64),
        co2_ppmv=jnp.full(shape_2d, 400.0, dtype=jnp.float64),
        has_radiation=jnp.ones(shape_2d, dtype=jnp.float64),
        has_precipitation=jnp.ones(shape_2d, dtype=jnp.float64),
        z_lowest=jnp.full(shape_2d, 10.0, dtype=jnp.float64),
    )
    ocean_sst = jnp.asarray(271.2 + 0.4 * rng.random(shape_2d), dtype=jnp.float64)
    ocean_u = jnp.asarray(-0.05 + 0.10 * rng.random(shape_2d), dtype=jnp.float64)
    ocean_v = jnp.asarray(-0.05 + 0.10 * rng.random(shape_2d), dtype=jnp.float64)

    # Caller 2: State builders (init_dynamic_ice_state, slab_to_dynamic, distribute_dynamic_state_to_categories)
    cur_init = init_dynamic_ice_state(shape_2d)
    par_init = parent_state_mod.init_dynamic_ice_state(shape_2d)
    for name in parent_state_mod.DynamicSeaIceState._fields:
        _compare_pair(getattr(cur_init, name).data, getattr(par_init, name).data)

    slab_cur = SeaIceState(
        h_ice=cur_init.h_ice.replace(data=jnp.asarray(0.8 + 1.2 * rng.random(shape_2d), dtype=jnp.float64)),
        T_ice=cur_init.T_ice.replace(data=jnp.asarray(265.0 + 5.0 * rng.random(shape_2d), dtype=jnp.float64)),
        concentration=cur_init.concentration.replace(data=jnp.asarray(0.4 + 0.5 * rng.random(shape_2d), dtype=jnp.float64)),
    )
    slab_par = parent_state_mod.SeaIceState(
        h_ice=slab_cur.h_ice,
        T_ice=slab_cur.T_ice,
        concentration=slab_cur.concentration,
    )
    cur_lifted = slab_to_dynamic(slab_cur)
    par_lifted = parent_state_mod.slab_to_dynamic(slab_par)
    for name in parent_state_mod.DynamicSeaIceState._fields:
        _compare_pair(getattr(cur_lifted, name).data, getattr(par_lifted, name).data)

    cur_seeded = cur_lifted._replace(
        h_snow=cur_lifted.h_snow.replace(data=hsnow_in),
        S_ice=cur_lifted.S_ice.replace(data=jnp.full(shape_2d, 4.0, dtype=jnp.float64)),
        pond_area=cur_lifted.pond_area.replace(data=pa_in),
        pond_depth=cur_lifted.pond_depth.replace(data=pd_in),
    )
    par_seeded = par_lifted._replace(
        h_snow=par_lifted.h_snow.replace(data=hsnow_in),
        S_ice=par_lifted.S_ice.replace(data=jnp.full(shape_2d, 4.0, dtype=jnp.float64)),
        pond_area=par_lifted.pond_area.replace(data=pa_in),
        pond_depth=par_lifted.pond_depth.replace(data=pd_in),
    )
    cur_mc = distribute_dynamic_state_to_categories(cur_seeded, 5)
    par_mc = parent_state_mod.distribute_dynamic_state_to_categories(par_seeded, 5)
    for name in parent_state_mod.DynamicSeaIceState._fields:
        _compare_pair(getattr(cur_mc, name).data, getattr(par_mc, name).data)

    # Caller 3: step_sea_ice slab mode (SeaIceState, default SeaIceConfig)
    cfg_slab = SeaIceConfig()
    got_slab_st, got_slab_rsp = sea_ice_mod.step_sea_ice(
        slab_cur, forcing, ocean_sst, ocean_u, ocean_v, cfg_slab, U_min=1.0, dt=RDT
    )
    want_slab_st, want_slab_rsp = parent_sea_ice_mod.step_sea_ice(
        slab_par, forcing, ocean_sst, ocean_u, ocean_v, cfg_slab, U_min=1.0, dt=RDT
    )
    for g, w in zip(jax.tree.leaves(got_slab_st), jax.tree.leaves(want_slab_st), strict=True):
        _compare_pair(g, w)
    for g, w in zip(jax.tree.leaves(got_slab_rsp), jax.tree.leaves(want_slab_rsp), strict=True):
        _compare_pair(g, w)

    # Caller 4: step_sea_ice legacy dynamic mode (dynamics="free_drift", uses_new_physics=False)
    cfg_leg_dyn = SeaIceConfig(dynamics="free_drift")
    got_leg_st, got_leg_rsp = sea_ice_mod.step_sea_ice(
        cur_seeded, forcing, ocean_sst, ocean_u, ocean_v, cfg_leg_dyn, U_min=1.0, dt=RDT
    )
    want_leg_st, want_leg_rsp = parent_sea_ice_mod.step_sea_ice(
        par_seeded, forcing, ocean_sst, ocean_u, ocean_v, cfg_leg_dyn, U_min=1.0, dt=RDT
    )
    for name in parent_state_mod.DynamicSeaIceState._fields:
        _compare_pair(getattr(got_leg_st, name).data, getattr(want_leg_st, name).data)
    for g, w in zip(jax.tree.leaves(got_leg_rsp), jax.tree.leaves(want_leg_rsp), strict=True):
        _compare_pair(g, w)

    # Caller 5: step_sea_ice single-category new-physics mode (default CESM ponds + snow + brine + delta-Eddington)
    cfg_v2_1cat = SeaIceConfig(
        shortwave_scheme="delta_eddington",
        snow=SnowConfig(enabled=True, flooding=True),
        brine=BrineConfig(enabled=True),
        ponds=MeltPondConfig(enabled=True),
    )
    got_v2_1_st, got_v2_1_rsp = sea_ice_mod.step_sea_ice(
        cur_seeded, forcing, ocean_sst, ocean_u, ocean_v, cfg_v2_1cat, U_min=1.0, dt=RDT
    )
    want_v2_1_st, want_v2_1_rsp = parent_sea_ice_mod.step_sea_ice(
        par_seeded, forcing, ocean_sst, ocean_u, ocean_v, cfg_v2_1cat, U_min=1.0, dt=RDT
    )
    for name in parent_state_mod.DynamicSeaIceState._fields:
        _compare_pair(getattr(got_v2_1_st, name).data, getattr(want_v2_1_st, name).data)
    for g, w in zip(jax.tree.leaves(got_v2_1_rsp), jax.tree.leaves(want_v2_1_rsp), strict=True):
        _compare_pair(g, w)

    # Caller 6: step_sea_ice 5-category new-physics mode (EVP + advect + Lipscomb2001 + ridging + CESM ponds + snow + brine)
    grid = create_latlon_grid(n_lat=shape_2d[0], n_lon=shape_2d[1])
    cfg_v2_5cat = SeaIceConfig(
        n_categories=5,
        dynamics="evp",
        N_evp=4,
        transport="advect",
        itd_remap="lipscomb2001",
        shortwave_scheme="delta_eddington",
        snow=SnowConfig(enabled=True, flooding=True),
        brine=BrineConfig(enabled=True),
        ridging=RidgingConfig(enabled=True),
        ponds=MeltPondConfig(enabled=True),
    )
    got_v2_5_st, got_v2_5_rsp = sea_ice_mod.step_sea_ice(
        cur_mc, forcing, ocean_sst, ocean_u, ocean_v, cfg_v2_5cat, U_min=1.0, dt=RDT, grid=grid
    )
    want_v2_5_st, want_v2_5_rsp = parent_sea_ice_mod.step_sea_ice(
        par_mc, forcing, ocean_sst, ocean_u, ocean_v, cfg_v2_5cat, U_min=1.0, dt=RDT, grid=grid
    )
    for name in parent_state_mod.DynamicSeaIceState._fields:
        _compare_pair(getattr(got_v2_5_st, name).data, getattr(want_v2_5_st, name).data)
    for g, w in zip(jax.tree.leaves(got_v2_5_rsp), jax.tree.leaves(want_v2_5_rsp), strict=True):
        _compare_pair(g, w)

    assert total_arrays == 161
    assert total_elements == 2604
    assert unequal_elements == 0

    # Planted control: a 1-ULP perturbation on current step_ponds must be caught by the parent comparison
    orig_step_ponds = ponds_mod.step_ponds
    monkeypatch.setattr(
        sea_ice_mod,
        "step_ponds",
        lambda *a, **kw: (
            jnp.asarray(np.nextafter(np.asarray(orig_step_ponds(*a, **kw)[0]), np.inf)),
            *orig_step_ponds(*a, **kw)[1:],
        ),
    )
    perturbed_st, _ = sea_ice_mod.step_sea_ice(
        cur_seeded, forcing, ocean_sst, ocean_u, ocean_v, cfg_v2_1cat, U_min=1.0, dt=RDT
    )
    assert np.count_nonzero(_bits(perturbed_st.pond_area.data) != _bits(want_v2_1_st.pond_area.data)) > 0
