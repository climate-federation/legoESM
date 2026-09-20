#!/usr/bin/env python3
"""Ordered kt=2 ORCA2 TKE walk, fail-closed at the first non-bit statement.

The oracle trace is evaluated statement-by-statement on NEMO-owned rank-zero
cells.  Production helpers are JIT compiled; the literal trace is harness-only
and never becomes an alternate model implementation.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card
from legoesm.ocean.physics.vertical_mixing.tke import (
    _nemo_literal_langmuir_operands,
    _nemo_literal_tke_solve,
    _nemo_etau_htau,
    compute_K_from_tke,
    compute_mixing_lengths,
    nemo_etau_injection,
    nemo_langmuir_tke_source,
    nemo_literal_langmuir_tke_update,
    nemo_tke_effective_ice_fraction,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_phase2u_tke_acquisition_gate as admission,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_phase2t_sh2_gate import (
    owned3,
    read_arrays as read_zdf_arrays,
    score,
)

RECORD = admission.RECORD
ZDF_RECORD = "oracle_zdf_sh2_operands_kt00000002.bin"
HALO = 2
DT = 10800.0
RHO0 = 1026.0
GRAV = 9.80665
RN_BSHEAR = 1.0e-20
RI_CRI = 2.0 / (2.0 + 0.7 / 0.1)
RN_EMIN = 1.0e-10
RN_EMIN0 = 1.0e-4
RN_EBB = 67.83

ORDER = (
    "taum_input", "ice_fraction", "zWlc2", "zpelc", "imlc", "zhlc", "zus3",
    "en_post_lc", "production_langmuir", "pdlr", "zdiag_pre_solve",
    "zd_lw_pre_solve", "zd_up_pre_solve", "en_rhs_pre_solve",
    "zdiag_after_forward", "zd_lw_after_forward", "en_post_solve",
    "production_solve", "htau", "en_post_etau", "production_etau", "mxlm",
    "mxld", "production_mixing_lengths", "avm_post", "avt_post",
    "dissl_post", "production_closure",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _read_tke_arrays(path: Path, mesh: Path) -> tuple[dict[str, np.ndarray], dict]:
    metadata = admission.read_tke(path, mesh)
    extents = [tuple(int(v) for v in row) for row in metadata["extent_table"]]
    arrays: dict[str, np.ndarray] = {}
    offset = 16 + 4 * admission.HEADER_INTS + 12 * len(admission.FIELDS)
    with path.open("rb") as handle:
        handle.seek(offset)
        for name, extent in zip(admission.FIELDS, extents, strict=True):
            count = int(np.prod(extent))
            raw = handle.read(8 * count)
            require(len(raw) == 8 * count, f"{name}: truncated")
            arrays[name] = np.frombuffer(raw, np.float64).reshape(extent, order="F")
        require(handle.read(1) == b"", "TKE payload did not end at exact EOF")
    return arrays, metadata


def _owned_tke(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    value = arrays[name].transpose(1, 0, 2)
    return value[HALO:-HALO, HALO:-HALO]


def _owned_tke2(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    return np.squeeze(_owned_tke(arrays, name), axis=2)


def _owned_zdf2(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    value = np.squeeze(arrays[name], axis=2).T
    # ZDF 2-D taum is reduced; fr_i/mbkt are full-domain canonical arrays.
    return value if value.shape == (148, 90) else value[HALO:-HALO, HALO:-HALO]


def _masks(mesh: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from netCDF4 import Dataset

    with Dataset(mesh) as dataset:
        tmask = np.asarray(dataset["tmask"][0], bool).transpose(1, 2, 0)
    # retained mesh is (z,y,x); after transpose above it is (y,x,z)
    require(tmask.shape == (148, 90, 31), f"unexpected tmask {tmask.shape}")
    wmask = np.zeros_like(tmask)
    wmask[..., 0] = tmask[..., 0]
    wmask[..., 1:] = tmask[..., 1:] & tmask[..., :-1]
    surface = tmask[..., 0]
    return tmask, wmask, surface


@jax.jit
def _literal_langmuir_trace(taum, rn2b, gdepw, e3w, ice, mbkt, wmask):
    """zdftke.F90:358,366-393 in Fortran statement order."""
    zcsd = jnp.asarray(0.5 * 0.016 * 0.016 / (1.22 * 1.5e-3), taum.dtype)
    zwlc2 = zcsd * taum
    terms = jnp.maximum(rn2b, 0.0) * gdepw * e3w
    rows = []
    carry = terms[..., 0]
    rows.append(carry)
    for k in range(1, terms.shape[-1]):
        carry = carry + terms[..., k]
        rows.append(carry)
    zpelc = jnp.stack(rows, axis=-1)
    imlc = jnp.asarray(mbkt, jnp.int32)
    for k in range(zpelc.shape[-1] - 2, 0, -1):
        imlc = jnp.where(zpelc[..., k] > zwlc2, k, imlc)
    zhlc = jnp.take_along_axis(gdepw, imlc[..., None], axis=-1)[..., 0]
    zus = jnp.sqrt(2.0 * zwlc2)
    zus3 = jnp.maximum(0.0, 1.0 - ice) * zus * zus * zus * wmask[..., 0]
    return zwlc2, zpelc, imlc, zhlc, zus3


@jax.jit
def _literal_matrix_trace(en_lc, sh2, avm, avt, rn2, rn2b, dissl,
                          e3t, e3w, wmask):
    """zdftke.F90:420-547, including all three source recurrences."""
    pdlr = jnp.zeros_like(en_lc)
    zri = jnp.where(
        rn2b <= 0.0, 0.0,
        rn2b * avm / jnp.where(sh2 + RN_BSHEAR == 0.0, RN_BSHEAR,
                              sh2 + RN_BSHEAR))
    pdlr = pdlr.at[..., 1:-1].set(
        jnp.maximum(0.1, RI_CRI / jnp.maximum(RI_CRI, zri[..., 1:-1])))

    lower = jnp.zeros_like(en_lc)
    upper = jnp.zeros_like(en_lc)
    diag = jnp.zeros_like(en_lc)
    rhs = en_lc
    surface = jnp.maximum(RN_EMIN0, (RN_EBB / RHO0) * jnp.maximum(0.0, 0.0))
    # The actual surface en is already in en_lc; preserve its reciprocal row.
    diag = diag.at[..., 0].set(1.0 / en_lc[..., 0])
    lower = lower.at[..., 0].set(1.0)
    for k in range(1, en_lc.shape[-1] - 1):
        zcof = (-0.5 * DT) * wmask[..., k]
        zu = (zcof * jnp.maximum(avm[..., k + 1] + avm[..., k], 2.0e-5)
              / (e3t[..., k] * e3w[..., k]))
        zl = (zcof * jnp.maximum(avm[..., k] + avm[..., k - 1], 2.0e-5)
              / (e3t[..., k - 1] * e3w[..., k]))
        upper = upper.at[..., k].set(zu)
        lower = lower.at[..., k].set(zl)
        diag = diag.at[..., k].set(
            1.0 - zl - zu + (1.5 * DT * 0.7) * dissl[..., k] * wmask[..., k])
        updated = rhs[..., k] + DT * (
            sh2[..., k] - avt[..., k] * rn2[..., k]
            + (0.5 * 0.7) * dissl[..., k] * rhs[..., k]) * wmask[..., k]
        rhs = rhs.at[..., k].set(updated)

    diag_fwd = diag
    for k in range(1, en_lc.shape[-1] - 1):
        diag_fwd = diag_fwd.at[..., k].set(
            diag_fwd[..., k] - lower[..., k] * upper[..., k - 1]
            / diag_fwd[..., k - 1])
    lower_fwd = lower
    for k in range(1, en_lc.shape[-1] - 1):
        lower_fwd = lower_fwd.at[..., k].set(
            rhs[..., k] - lower_fwd[..., k] / diag_fwd[..., k - 1]
            * lower_fwd[..., k - 1])
    solved = rhs
    last = en_lc.shape[-1] - 2
    solved = solved.at[..., last].set(lower_fwd[..., last] / diag_fwd[..., last])
    for k in range(last - 1, 0, -1):
        solved = solved.at[..., k].set(
            (lower_fwd[..., k] - upper[..., k] * solved[..., k + 1])
            / diag_fwd[..., k])
    solved = solved.at[..., 1:-1].set(
        jnp.maximum(solved[..., 1:-1], RN_EMIN) * wmask[..., 1:-1])
    return pdlr, diag, lower, upper, rhs, diag_fwd, lower_fwd, solved


@jax.jit
def _literal_mxl_closure(en, rn2, e3t, taum, fr_i, hm_i, wmask,
                         pdlr, avtb2d):
    """zdftke.F90:671,684-820 for resolved ln_mxl0/nn_mxlice=2/nn_mxl=3."""
    rmxl_min = jnp.asarray(1.0e-3, en.dtype)
    anchor = ((1.0 - fr_i) * (0.4 * 2.0e5 / (RHO0 * GRAV)) * taum
              + fr_i * hm_i * 2.0) * wmask[..., 0]
    anchor = jnp.maximum(rmxl_min, anchor)
    raw = jnp.full_like(en, rmxl_min)
    raw = raw.at[..., 0].set(anchor)
    raw = raw.at[..., 1:-1].set(jnp.maximum(
        rmxl_min,
        jnp.sqrt((2.0 * en[..., 1:-1]) /
                 jnp.maximum(rn2[..., 1:-1], 0.5 * jnp.finfo(en.dtype).eps))))
    lup = raw
    for k in range(1, en.shape[-1] - 1):
        lup = lup.at[..., k].set(jnp.minimum(
            lup[..., k - 1] + e3t[..., k - 1], raw[..., k]))
    ldn = raw.at[..., -1].set(rmxl_min)
    for k in range(en.shape[-1] - 2, 0, -1):
        ldn = ldn.at[..., k].set(jnp.minimum(
            ldn[..., k + 1] + e3t[..., k + 1], raw[..., k]))
    mxlm = jnp.minimum(lup, ldn)
    mxld = jnp.sqrt(lup * ldn)
    sqen = jnp.sqrt(en)
    zav = 0.1 * mxlm * sqen
    avm = jnp.maximum(zav, 1.2e-4) * wmask
    avt0 = jnp.maximum(zav, avtb2d[..., None] * 1.2e-5) * wmask
    avt = avt0.at[..., 1:-1].set(jnp.maximum(
        pdlr[..., 1:-1] * avt0[..., 1:-1],
        avtb2d[..., None] * 1.2e-5) * wmask[..., 1:-1])
    dissl = sqen / mxld
    return mxlm, mxld, avm, avt, dissl


def _target_plant(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    altered = np.array(target, copy=True)
    first = tuple(int(v) for v in np.argwhere(mask)[0])
    altered[first] = np.nextafter(altered[first], np.inf)
    return altered


def validate(deck: Path, run_a: Path, run_b: Path, baseline: Path,
             identity: Path, mesh: Path, plant: str | None) -> dict:
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu", "CPU-only gate")
    require(not jax.config.jax_disable_jit, "production JIT disabled")

    admitted = admission.validate(run_a, run_b, baseline, identity)
    raw, tke_meta = _read_tke_arrays(run_a / RECORD, mesh)
    zdf, zdf_meta = read_zdf_arrays(run_a / ZDF_RECORD, mesh)
    card = build_orca2_zps_card(deck)
    cfg = card.recipe.model_config.physics.vertical_mixing.tke
    lat_native = getattr(card.recipe.grid, "native_lat_T_deg", None)
    lat = np.asarray(np.degrees(card.recipe.grid.lat_T) if lat_native is None
                     else lat_native, np.float64)
    # Rank zero owns the western 90-column subdomain in the fixed 2x1 layout.
    lat = lat[:, :90]
    tmask, wmask, surface = _masks(mesh)
    matrix_mask = wmask.copy()
    matrix_mask[..., 0] = surface
    matrix_mask[..., -1] = False
    column_mask = np.zeros_like(wmask)
    mbkt = _owned_zdf2(zdf, "mbkt_real").astype(np.int32)
    for j, i in np.argwhere(surface):
        column_mask[j, i, :min(int(mbkt[j, i]) + 1, 31)] = True

    def z(name):
        return np.asarray(owned3(zdf, name), np.float64)

    def t(name):
        return np.asarray(_owned_tke(raw, name), np.float64)

    def t2(name):
        return np.asarray(_owned_tke2(raw, name), np.float64)

    taum, fr_i = _owned_zdf2(zdf, "taum"), _owned_zdf2(zdf, "fr_i")
    # The NCAR bulk boundary remains owned by Lane 3b.  Downstream of that
    # open boundary this walk follows the campaign's operand-substitution
    # rule: the exact post-SBC NEMO taum is injected into the production TKE
    # call.  JIT the hand-off itself so a future cast/reshape cannot hide in
    # the harness; this row certifies the supplied operand boundary, not the
    # unresolved NCAR operator that produced it.
    production_taum = np.asarray(jax.jit(lambda value: value)(
        jnp.asarray(taum, dtype=jnp.float64)))
    rn2, rn2b = z("rn2"), z("rn2b")
    gdepw, e3w, e3t = z("gdepw_Kmm"), z("e3w_Kmm"), z("e3t_Kmm")
    sh2, avm, avt = z("sh2"), z("avm_k_pre"), z("avt_k_pre")
    en_pre, dissl_pre = z("en_pre"), t("dissl_pre")
    ice = np.asarray(jax.jit(
        lambda value: nemo_tke_effective_ice_fraction(value, int(cfg.eice))
    )(jnp.asarray(fr_i)))
    lc = tuple(np.asarray(value) for value in _literal_langmuir_trace(
        jnp.asarray(taum), jnp.asarray(rn2b), jnp.asarray(gdepw),
        jnp.asarray(e3w), jnp.asarray(ice), jnp.asarray(mbkt),
        jnp.asarray(wmask)))
    zwlc2, zpelc, imlc, zhlc, zus3 = lc
    en_seed = np.array(en_pre, copy=True)
    en_seed[..., 0] = np.maximum(RN_EMIN0, (RN_EBB / RHO0) * taum)
    # Bottom drag was certified upstream; isolate this walk by supplying its
    # exact post-boundary slots from the oracle trace.
    oracle_lc = t("en_post_lc")
    bottom = column_mask & ~wmask
    en_seed[bottom] = oracle_lc[bottom]
    literal_cfg = cfg._replace(tke_langmuir_evaluation="nemo_literal")
    literal_lc = np.asarray(jax.jit(
        lambda e, tau, n2, dep, dz, f, bottom, wet:
        nemo_literal_langmuir_tke_update(
            e, DT, tau, n2, dep, dz, literal_cfg, f, bottom, wet)
    )(jnp.asarray(en_seed), jnp.asarray(taum), jnp.asarray(rn2b),
      jnp.asarray(gdepw), jnp.asarray(e3w), jnp.asarray(ice),
      jnp.asarray(mbkt), jnp.asarray(wmask)))
    production_lc = np.asarray(jax.jit(
        lambda e, tau, n2, dep, dz, f: e + DT * nemo_langmuir_tke_source(
            tau, n2, dep, dz, cfg, ice_frac=f,
            bottom_level=jnp.asarray(mbkt), w_active=jnp.asarray(wmask))
    )(jnp.asarray(en_seed), jnp.asarray(taum), jnp.asarray(rn2b),
      jnp.asarray(gdepw), jnp.asarray(e3w), jnp.asarray(ice)))

    matrix = tuple(np.asarray(value) for value in _literal_matrix_trace(
        jnp.asarray(oracle_lc), jnp.asarray(sh2), jnp.asarray(avm),
        jnp.asarray(avt), jnp.asarray(rn2), jnp.asarray(rn2b),
        jnp.asarray(dissl_pre), jnp.asarray(e3t), jnp.asarray(e3w),
        jnp.asarray(wmask)))
    pdlr, diag, lower, upper, rhs, diag_fwd, lower_fwd, solved = matrix

    # Production literal solver consumes the recorded pre-solve operands;
    # this is the same single shared implementation called by TKE production.
    production_solved = np.asarray(jax.jit(
        lambda a, b, c, rhs_, surface_, wet: _nemo_literal_tke_solve(
            a, b, c, rhs_, surface_, wet, RN_EMIN)
    )(jnp.asarray(t("zd_lw_pre_solve")),
      jnp.asarray(t("zdiag_pre_solve")),
      jnp.asarray(t("zd_up_pre_solve")),
      jnp.asarray(t("en_rhs_pre_solve")),
      jnp.asarray(t("en_post_lc")[..., 0]), jnp.asarray(wmask[..., 1:])))
    oracle_solved = t("en_post_solve")
    literal_etau_cfg = cfg._replace(
        etau_mode="below_ml", etau_htau_mode="latitude",
        tke_htau_evaluation="nemo_literal",
        tke_etau_exponential_evaluation="nemo_literal")
    literal_etau = np.asarray(jax.jit(
        lambda e, tau, depth, latitude, f: nemo_etau_injection(
            e, tau, depth, literal_etau_cfg, RHO0, latitude, f)
    )(jnp.asarray(oracle_solved[..., 1:-1]), jnp.asarray(taum),
      jnp.asarray(gdepw[..., 1:-1]), jnp.asarray(lat), jnp.asarray(ice)))
    literal_htau = np.asarray(jax.jit(
        lambda latitude: _nemo_etau_htau(
            latitude, literal_etau_cfg, jnp.dtype(jnp.float64))
    )(jnp.asarray(lat)))
    production_etau = np.asarray(jax.jit(
        lambda e, tau, depth, latitude, f: nemo_etau_injection(
            e, tau, depth, cfg, RHO0, latitude, f)
    )(jnp.asarray(oracle_solved[..., 1:-1]), jnp.asarray(taum),
      jnp.asarray(gdepw[..., 1:-1]), jnp.asarray(lat), jnp.asarray(ice)))

    avtb2d = np.ones_like(taum)
    select = (-15.0 <= lat) & (lat < -5.0)
    avtb2d[select] = 1.0 - 0.09 * (lat[select] + 15.0)
    select = (-5.0 <= lat) & (lat < 5.0)
    avtb2d[select] = 0.1
    select = (5.0 <= lat) & (lat < 15.0)
    avtb2d[select] = 0.1 + 0.09 * (lat[select] - 5.0)
    closure = tuple(np.asarray(value) for value in _literal_mxl_closure(
        jnp.asarray(t("en_post_etau")), jnp.asarray(rn2), jnp.asarray(e3t),
        jnp.asarray(taum), jnp.asarray(fr_i), jnp.asarray(t2("hm_i")),
        jnp.asarray(wmask), jnp.asarray(t("pdlr")), jnp.asarray(avtb2d)))
    mxlm, mxld, avm_post, avt_post, dissl_post = closure

    # Production closure helpers use the card's selected tuple.  Their rows
    # expose selector debt independently of the source-literal oracle trace.
    en_i, rn2_i = t("en_post_etau")[..., 1:-1], rn2[..., 1:-1]
    e3w_i, e3t_all = e3w[..., 1:-1], e3t
    anchor = np.maximum(
        1.0e-3,
        ((1.0 - fr_i) * (0.4 * 2.0e5 / (RHO0 * GRAV)) * taum
         + fr_i * t2("hm_i") * 2.0) * surface)
    prod_lengths = tuple(np.asarray(value) for value in jax.jit(
        lambda e, n, half, cell, a: compute_mixing_lengths(
            e, n, half, cfg, signed_n2=True, dz_cell=cell,
            l_surface_anchor=a)
    )(jnp.asarray(en_i), jnp.asarray(rn2_i), jnp.asarray(e3w_i),
      jnp.asarray(e3t_all), jnp.asarray(anchor)))
    prod_k = tuple(np.asarray(value) for value in jax.jit(
        lambda e, lk, n, s, nb, p, km: compute_K_from_tke(
            e, lk, cfg, N2=n, shear_sq=s, N2_prandtl=nb,
            p_sh2_override=lambda _: p, prandtl_K_M=km)
    )(jnp.asarray(en_i), jnp.asarray(prod_lengths[0]), jnp.asarray(rn2_i),
      jnp.asarray(sh2[..., 1:-1]), jnp.asarray(rn2b[..., 1:-1]),
      jnp.asarray(t("pdlr")[..., 1:-1]), jnp.asarray(avm[..., 1:-1])))

    rowspec = [
        ("taum_input", production_taum, taum, surface,
         "ORACLE_SUPPLIED post-sbc taum -> zdftke.F90:265,332"),
        ("ice_fraction", ice, t2("zice_fra"), surface, "zdftke.F90:253-258"),
        ("zWlc2", zwlc2, t2("zWlc2"), surface, "zdftke.F90:326-333"),
        ("zpelc", zpelc, t("zpelc"), column_mask, "zdftke.F90:339-345"),
        ("imlc", imlc.astype(np.float64), t2("imlc_real"), surface, "zdftke.F90:347-351"),
        ("zhlc", zhlc, t2("zhlc"), surface, "zdftke.F90:352-355"),
        ("zus3", zus3, t2("zus3"), surface, "zdftke.F90:357-360"),
        ("en_post_lc", literal_lc, oracle_lc, column_mask, "zdftke.F90:361-370"),
        ("production_langmuir", production_lc, oracle_lc, column_mask, "tke.py:nemo_langmuir_tke_source"),
        ("pdlr", pdlr, t("pdlr"), matrix_mask, "zdftke.F90:381-400"),
        ("zdiag_pre_solve", diag, t("zdiag_pre_solve"), matrix_mask, "zdftke.F90:403-420"),
        ("zd_lw_pre_solve", lower, t("zd_lw_pre_solve"), matrix_mask, "zdftke.F90:403-420"),
        ("zd_up_pre_solve", upper, t("zd_up_pre_solve"), matrix_mask, "zdftke.F90:403-420"),
        ("en_rhs_pre_solve", rhs, t("en_rhs_pre_solve"), column_mask, "zdftke.F90:416-420"),
        ("zdiag_after_forward", diag_fwd, t("zdiag_after_forward"), matrix_mask, "zdftke.F90:513"),
        ("zd_lw_after_forward", lower_fwd, t("zd_lw_after_forward"), matrix_mask, "zdftke.F90:529"),
        ("en_post_solve", solved, oracle_solved, column_mask, "zdftke.F90:451-469"),
        ("production_solve", production_solved, oracle_solved[..., 1:], wmask[..., 1:], "tke.py:_nemo_literal_tke_solve"),
        ("htau", literal_htau, t2("htau"), surface,
            "zdftke.F90:1057-1062"),
        ("en_post_etau", literal_etau, t("en_post_etau")[..., 1:-1],
            wmask[..., 1:-1], "zdftke.F90:581-582"),
        ("production_etau", production_etau, t("en_post_etau")[..., 1:-1], wmask[..., 1:-1], "tke.py:nemo_etau_injection"),
        ("mxlm", mxlm, t("mxlm"), column_mask, "zdftke.F90:684-800"),
        ("mxld", mxld, t("mxld"), column_mask, "zdftke.F90:684-800"),
        ("production_mixing_lengths", prod_lengths[0], t("mxlm")[..., 1:-1], wmask[..., 1:-1], "tke.py:compute_mixing_lengths"),
        ("avm_post", avm_post, t("avm_post"), column_mask, "zdftke.F90:809-812"),
        ("avt_post", avt_post, t("avt_post"), column_mask, "zdftke.F90:809-820"),
        ("dissl_post", dissl_post, t("dissl_post"), column_mask, "zdftke.F90:813"),
        ("production_closure", prod_k[0], t("avm_post")[..., 1:-1], wmask[..., 1:-1], "tke.py:compute_K_from_tke"),
    ]

    results = []
    first_debt = None
    for name, candidate, target, mask, citation in rowspec:
        if plant == name:
            target = _target_plant(target, mask)
        if np.ndim(candidate) == 2:
            candidate = np.asarray(candidate)[..., None]
            target = np.asarray(target)[..., None]
            mask = np.asarray(mask)[..., None]
        row = score(candidate, target, mask)
        row.update({"name": name, "citation": citation,
                    "plant": plant == name})
        results.append(row)
        if plant == name:
            require(row["status"] == "DEBT", f"plant {plant} did not bind")
            raise GateError(f"binding {plant} plant rejected")
        if plant is None and row["status"] != "AT_BAR":
            first_debt = row
            break
    if plant is not None:
        raise GateError(f"plant {plant} was not present in the row inventory")

    return {
        "status": "AT_BAR" if first_debt is None else "DEBT",
        "execution": {"backend": jax.default_backend(), "jit": "production",
                      "dtype": "float64", "transcendentals": get_policy().transcendentals},
        "admission": {"status": admitted["status"],
                      "twin_raw_exact": admitted["twin_raw_exact"],
                      "inherited_raw_exact": admitted["inherited_raw_exact"]},
        "records": {RECORD: tke_meta["sha256"], ZDF_RECORD: zdf_meta["sha256"]},
        "resolved_card": {
            "eice": int(cfg.eice), "lc": bool(cfg.lc),
            "langmuir_evaluation": cfg.tke_langmuir_evaluation,
            "etau_mode": cfg.etau_mode, "mxl": int(cfg.tke_mxl_choice),
            "mxl_raw": cfg.tke_mxl_raw_evaluation,
            "matrix": cfg.tke_matrix_evaluation,
            "solver": cfg.tke_solver_evaluation,
        },
        "ordered_rows": results,
        "first_non_bit": first_debt,
        "owner_rule": (
            "card selector or ORCA2 forcing operand -> LANE4; shared TKE "
            "arithmetic -> GYRE_OWNER_SHARED_TKE with this reproducer"),
        "oracle_supplied": ["certified SH2", "bottom-drag boundary slots"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--identity-control", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--plant", choices=ORDER)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.deck_root, args.run_a, args.run_b, args.baseline,
                          args.identity_control, args.mesh, args.plant)
    except (GateError, OSError, ValueError, struct.error) as error:
        print(f"FAIL: {error}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
