#!/usr/bin/env python
"""Close the ordered ZDF sweep at rows 28--32 from existing d180 dumps.

Rows 28 and 29 bind previously committed receipts.  Row 30 re-enters the
production ldf_slp implementation and walks its dumped operands.  Rows 31 and
32 apply the preregistered NEMO-state substitutions to the production generic
implicit solver; row 31 additionally evaluates NEMO's literal matrix and
ordered recurrences to discriminate an input error from solver lowering.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bn2_alpha_compare as loaders  # noqa: E402
import dump_lane  # noqa: E402
import ldf_slp_per_element as ldf  # noqa: E402
import zdf_chain_sweep as sweep  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.fidelity.nemo_io import (  # noqa: E402
    read_nemo_restart, read_nemo_restart_before,
)
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (  # noqa: E402
    nemo_iso_a33, nemo_iso_face_masks,
)
from legoesm.ocean.physics.vertical_mixing import (  # noqa: E402
    implicit_vertical_diffusion_ocean,
    implicit_vertical_diffusion_ocean_pair,
)

POINTWISE = 1.0e-15
ACCUMULATING = 1.0e-12
ROW28_SHA256 = "a64a0261e968f2a3467c8beef1a48c3dfc118df426ccf1c9368231a164bf63ff"
ROW28_DUMP_SHA256 = "d3a62643bc8e5ea0370a784a6659cc386410baa516b270f86cacedd0603edad9"
SEOS_ORACLE_SO_SHA256 = "fd831e156b2efed818bc3e96dc8ffaab37b6f46a66fcff9f88b2e40de3a8f68e"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _column_metric(actual, expected, wet, focus, bar):
    return sweep.metrics(np.asarray(actual), np.asarray(expected), wet, focus, bar)


def _literal_nemo_momentum_solve(
    rhs, avm, dz, dzh, wet, bottom, drag_rate, rdt, face_axis,
):
    """dynzdf.F90:199-214,311-315,340-380 in written association."""
    nlev = rhs.shape[-1]
    zdt2 = 0.5 * rdt
    avm_other = np.roll(avm, -1, axis=face_axis)
    a = np.zeros_like(rhs)
    c = np.zeros_like(rhs)
    for k in range(1, nlev):
        a[..., k] = (
            -zdt2 * (avm_other[..., k] + avm[..., k])
            / (dz[..., k] * dzh[..., k - 1]) * wet[..., k] * wet[..., k - 1]
        )
    for k in range(nlev - 1):
        c[..., k] = (
            -zdt2 * (avm_other[..., k + 1] + avm[..., k + 1])
            / (dz[..., k] * dzh[..., k]) * wet[..., k + 1] * wet[..., k]
        )
    b = 1.0 - a - c
    b = b + rdt * drag_rate[..., None] / np.maximum(dz, 1.0e-10) * bottom
    for k in range(1, nlev):
        b[..., k] = b[..., k] - a[..., k] * c[..., k - 1] / b[..., k - 1]
    x = np.array(rhs, copy=True)
    for k in range(1, nlev):
        x[..., k] = x[..., k] - a[..., k] / b[..., k - 1] * x[..., k - 1]
    x[..., -1] = x[..., -1] / b[..., -1]
    for k in range(nlev - 2, -1, -1):
        x[..., k] = (x[..., k] - c[..., k] * x[..., k + 1]) / b[..., k]
    return x * wet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--row28-artifact", type=Path, required=True)
    ap.add_argument("--row28-run-dir", type=Path, required=True)
    ap.add_argument("--tail-artifact", type=Path, required=True)
    ap.add_argument("--mld-maps", type=Path, required=True)
    ap.add_argument("--nemo-source-root", type=Path, required=True)
    ap.add_argument("--repo-sha", required=True)
    ap.add_argument("--seos-oracle-so", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("chain-end receipt requires CPU fp64")
    run = args.run_dir.resolve()
    if Path(dump_lane.RUN_DIR).resolve() != run or dump_lane.LANE != "d180":
        raise SystemExit("chain-end receipt accepts only the registered d180 lane")
    focus = sweep.focus_from_maps(args.mld_maps)
    jpi, jpj, jpk, hls = loaders._read_dims(str(run))
    ni, nj = jpi - 2 * hls, jpj - 2 * hls

    def haloed(name):
        return loaders._load_haloed(str(run / name), jpi, jpj, hls)

    def interior_path(path):
        return loaders._load_interior(str(path), ni, nj)

    row28_path = args.row28_artifact.resolve()
    row28 = json.loads(row28_path.read_text())
    row28_dump = args.row28_run_dir.resolve() / "zdf_dump_hmld_turb.bin"
    if (sha256(row28_path) != ROW28_SHA256
            or sha256(row28_dump) != ROW28_DUMP_SHA256
            or row28.get("disposition") != "VERIFIED"
            or row28.get("metric", {}).get("n_diverged_columns") != 0
            or row28.get("metric", {}).get("n_wet_columns") != 9920
            or not all(row28.get("controls", {}).values())):
        raise SystemExit("row28 receipt is not promotable")
    tail_path = args.tail_artifact.resolve()
    tail = json.loads(tail_path.read_text())
    row29 = tail["rows"]["29"]["targeting_preview_avm_interior"]
    row29_controls = tail["rows"]["29"].get("controls", {})
    if (not row29.get("pass") or row29.get("n_diverged_columns") != 0
            or not row29_controls or not all(row29_controls.values())):
        raise SystemExit("row29 preview is not promotable")

    # Row 30: production call and every dumped j-side operand in source order.
    ls = ldf.build_state()
    out, loc = ldf.capture_locals(ls["recall"], ldf.compute_nemo_native_slopes.__code__)
    cap_mod = sys.modules[ldf.compute_nemo_native_slopes.__module__]
    cap = ldf._JnpCapture(cap_mod.jnp)
    old_jnp = cap_mod.jnp
    cap_mod.jnp = cap
    try:
        out_cap = ls["recall"]()
    finally:
        cap_mod.jnp = old_jnp
    if len(cap.captured) != 4 or not all(np.array_equal(np.asarray(a), np.asarray(b))
                                         for a, b in zip(out, out_cap)):
        raise SystemExit("row30 raw-slope capture is not inert")
    km1 = lambda a: np.concatenate([np.zeros_like(a[..., :1]), a[..., :-1]], axis=-1)
    row30_values = {
        "prd": np.asarray(loc["prd"]),
        "zgrv_iik": np.asarray(loc["zgrv"]),
        "zgrv_iikm1": km1(np.asarray(loc["zgrv"])),
        "zaj": np.asarray(loc["zaj"]),
        "zbw": np.asarray(loc["zbw"]),
        "zbj": np.asarray(loc["zbj"]),
        "zfk": 1.0 - np.asarray(loc["in_ml_w"], dtype=float),
        "zww_raw": cap.captured[3],
        "wslpj": np.asarray(out[3]),
    }
    row30_metrics = {}
    for name, actual in row30_values.items():
        expected = haloed(ldf.CHAIN_DUMPS[name])
        wet = (ls["active"] if name == "prd" else
               ldf.wet_v_mask(ls["active"], ls["v_mask"])
               if name.startswith("zgrv") else ldf.wet_w_mask(ls["active"]))
        score = np.zeros_like(wet, dtype=bool)
        if name == "prd":
            # prd is an INPUT consumed at levels 1:jpkm1, including zero-based
            # k=0.  KLO applies only to the subsequently dumped W-loop
            # intermediates; applying it here would hide a consumed operand.
            score[..., :min(35, score.shape[-1])] = wet[..., :min(35, wet.shape[-1])]
        else:
            score[..., ldf.KLO:ldf.KHI] = wet[..., ldf.KLO:ldf.KHI]
        nk = min(actual.shape[-1], expected.shape[-1], score.shape[-1], 35)
        row30_metrics[name] = _column_metric(
            actual[..., :nk], expected[..., :nk], score[..., :nk], focus,
            POINTWISE)
    for name, dump in ldf.DUMP_META.items():
        actual = np.asarray(ls["lego"][name])
        expected = haloed(dump)
        wet = ldf.MASK_FN[name](ls)
        nk = min(actual.shape[-1], expected.shape[-1], wet.shape[-1])
        row30_metrics[name] = _column_metric(
            actual[..., :nk], expected[..., :nk], wet[..., :nk], focus,
            POINTWISE)
    prd_dump = haloed(ldf.CHAIN_DUMPS["prd"])
    prd_actual = np.asarray(loc["prd"])[..., :35]
    prd_expected = prd_dump[..., :35]
    prd_wet = np.asarray(ls["active"], dtype=bool)[..., :35]
    row30_controls = sweep.planted_controls(
        prd_actual, prd_expected, prd_wet, POINTWISE)
    baseline_exact = prd_wet & (prd_actual != prd_expected)
    baseline_exact_columns = np.any(baseline_exact, axis=-1)
    ulp_candidates = prd_wet & (prd_actual == prd_expected) & (prd_actual != 0.0)
    ulp_index = tuple(np.argwhere(ulp_candidates)[0])
    ulp_prd = prd_actual.copy()
    ulp_prd[ulp_index] = np.nextafter(ulp_prd[ulp_index], np.inf)
    ulp_bad_columns = np.any(prd_wet & (ulp_prd != prd_expected), axis=-1)
    baseline_bad_count = int(baseline_exact_columns.sum())
    ulp_bad_count = int(ulp_bad_columns.sum())
    row30_controls["baseline_exact_unequal_columns"] = baseline_bad_count
    row30_controls["one_ulp_exact_identity_fired"] = (
        ulp_bad_count == baseline_bad_count + 1)
    row30_controls["one_ulp_exact_failing_columns"] = ulp_bad_count
    row30_controls["one_ulp_ji_k"] = [int(x) for x in ulp_index]
    seos_so = args.seos_oracle_so.resolve()
    if sha256(seos_so) != SEOS_ORACLE_SO_SHA256:
        raise SystemExit("row30 Fortran S-EOS discriminator binary SHA changed")
    lib = ctypes.CDLL(str(seos_so))
    seos = lib.row30_seos_prd
    fp = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    seos.argtypes = [ctypes.c_int, fp, fp, fp, fp]
    seos.restype = None
    active_flat = np.asarray(ls["active"], dtype=bool).ravel()
    t_flat = np.ascontiguousarray(np.asarray(ls["T"])[..., :].ravel()[active_flat])
    s_flat = np.ascontiguousarray(np.asarray(ls["S"])[..., :].ravel()[active_flat])
    depth_flat = np.ascontiguousarray(
        np.asarray(loc["_gdept_prd"]).ravel()[active_flat])
    fortran_prd = np.empty_like(t_flat)
    seos(t_flat.size, t_flat, s_flat, depth_flat, fortran_prd)
    jax_prd = np.asarray(loc["prd"]).ravel()[active_flat]
    seos_exact = int(np.count_nonzero(fortran_prd != jax_prd))
    poisoned_prd = fortran_prd.copy()
    poisoned_prd[0] = np.nextafter(poisoned_prd[0], np.inf)
    row30_seos_discriminator = {
        "n_wet_elements": int(t_flat.size),
        "exact_unequal": seos_exact,
        "max_abs_difference": float(np.max(np.abs(fortran_prd - jax_prd))),
        "pass": seos_exact == 0,
        "one_ulp_output_control_fails": bool(
            np.count_nonzero(poisoned_prd != jax_prd) > 0),
        "shared_object": str(seos_so),
        "shared_object_sha256": sha256(seos_so),
        "source": str((Path(__file__).parent / "row30_seos_oracle.f90").resolve()),
        "source_sha256": sha256(Path(__file__).parent / "row30_seos_oracle.f90"),
    }

    # Shared exact NEMO geometry and state for rows 31--32.
    now = read_nemo_restart(str(run / dump_lane.RESTART), nn_hls=0)
    before = read_nemo_restart_before(str(run / dump_lane.RESTART), nn_hls=0)
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        def z3(name):
            return np.moveaxis(np.asarray(ds[name][0]), 0, -1)
        tmask, umask, vmask = (z3("tmask") > 0.5, z3("umask") > 0.5,
                               z3("vmask") > 0.5)
        e3t0, e3w0 = z3("e3t_0"), z3("e3w_0")
        e3u0, e3v0 = z3("e3u_0"), z3("e3v_0")
        e3uw0, e3vw0 = z3("e3uw_0"), z3("e3vw_0")
        e1e2t = np.asarray(ds["e1t"][0]) * np.asarray(ds["e2t"][0])
        e1e2u = np.asarray(ds["e1u"][0]) * np.asarray(ds["e2u"][0])
        e1e2v = np.asarray(ds["e1v"][0]) * np.asarray(ds["e2v"][0])
    rdt = 5400.0
    ht = np.sum(e3t0 * tmask, axis=-1)
    safe_ht = np.where(ht > 0.0, ht, 1.0)
    r3bb = np.asarray(before.ssh) / safe_ht
    r3mm = np.asarray(now.ssh) / safe_ht
    r3aa = haloed("seq_dump_r3t_aaa.bin")[..., 0]
    e3bb = e3t0 * (1.0 + r3bb[..., None] * tmask)
    e3mm = e3t0 * (1.0 + r3mm[..., None] * tmask)
    e3aa = e3t0 * (1.0 + r3aa[..., None] * tmask)
    e3wmm = e3w0 * (1.0 + r3mm[..., None] * tmask)

    hu0, hv0 = np.sum(e3u0 * umask, -1), np.sum(e3v0 * vmask, -1)
    ssh = np.asarray(now.ssh)
    r3u = (0.5 * (e1e2t * ssh + np.roll(e1e2t * ssh, -1, axis=1))
           / np.where(hu0 * e1e2u != 0.0, hu0 * e1e2u, 1.0))
    r3v = (0.5 * (e1e2t * ssh + np.roll(e1e2t * ssh, -1, axis=0))
           / np.where(hv0 * e1e2v != 0.0, hv0 * e1e2v, 1.0))
    r3ua = haloed("seq_dump_r3u_aaa.bin")[..., 0]
    r3va = haloed("seq_dump_r3v_aaa.bin")[..., 0]
    e3uaa = e3u0 * (1.0 + r3ua[..., None] * umask)
    e3vaa = e3v0 * (1.0 + r3va[..., None] * vmask)
    e3uwmm = e3uw0 * (1.0 + r3u[..., None] * umask)
    e3vwmm = e3vw0 * (1.0 + r3v[..., None] * vmask)

    kr_u, kr_v = haloed("stp_dump_07_dynspg_u.bin"), haloed("stp_dump_07_dynspg_v.bin")
    out_u, out_v = haloed("stp_dump_08_dynzdf_u.bin"), haloed("stp_dump_08_dynzdf_v.bin")
    ub = haloed("stp_dump_07_dynspg_ub.bin")[..., 0]
    vb = haloed("stp_dump_07_dynspg_vb.bin")[..., 0]
    wu, wv = umask[..., :35], vmask[..., :35]
    rhs_u = (np.asarray(before.u)[..., :35] + rdt * kr_u - ub[..., None]) * wu
    rhs_v = (np.asarray(before.v)[..., :35] + rdt * kr_v - vb[..., None]) * wv
    rcd = haloed("drg_dump_rCdU_bot.bin")[..., 0]
    drag_u = -0.5 * (rcd + np.roll(rcd, -1, axis=1))
    drag_v = -0.5 * (rcd + np.roll(rcd, -1, axis=0))
    kk = np.arange(35)[None, None, :]
    bot_u = wu & (kk == wu.sum(-1)[..., None] - 1)
    bot_v = wv & (kk == wv.sum(-1)[..., None] - 1)
    rhs_u -= rdt * drag_u[..., None] / np.maximum(e3uaa[..., :35], 1e-10) * bot_u * ub[..., None]
    rhs_v -= rdt * drag_v[..., None] / np.maximum(e3vaa[..., :35], 1e-10) * bot_v * vb[..., None]
    ws_u = (interior_path(run / "zdf_dump_u1_poststress.bin")[..., 0]
            - interior_path(run / "zdf_dump_u1_prestress.bin")[..., 0])
    ws_v = (interior_path(run / "zdf_dump_v1_poststress.bin")[..., 0]
            - interior_path(run / "zdf_dump_v1_prestress.bin")[..., 0])
    rhs_u[..., 0] += ws_u
    rhs_v[..., 0] += ws_v
    avm = haloed("dump_avm.bin")
    avm_u = 0.5 * (avm + np.roll(avm, -1, axis=1))
    avm_v = 0.5 * (avm + np.roll(avm, -1, axis=0))
    Ku = avm_u[..., 1:35] * (wu[..., 1:] & wu[..., :-1])
    Kv = avm_v[..., 1:35] * (wv[..., 1:] & wv[..., :-1])
    diag_u = rdt * drag_u[..., None] / np.maximum(e3uaa[..., :35], 1e-10) * bot_u
    diag_v = rdt * drag_v[..., None] / np.maximum(e3vaa[..., :35], 1e-10) * bot_v
    generic_u = implicit_vertical_diffusion_ocean(
        jnp.asarray(rhs_u), jnp.asarray(Ku), jnp.asarray(e3uaa[..., :35]),
        jnp.asarray(e3uwmm[..., 1:35]), rdt, extra_diag=jnp.asarray(diag_u))
    generic_v = implicit_vertical_diffusion_ocean(
        jnp.asarray(rhs_v), jnp.asarray(Kv), jnp.asarray(e3vaa[..., :35]),
        jnp.asarray(e3vwmm[..., 1:35]), rdt, extra_diag=jnp.asarray(diag_v))
    literal_u = _literal_nemo_momentum_solve(
        rhs_u, avm, e3uaa[..., :35], e3uwmm[..., 1:35], wu, bot_u,
        drag_u, rdt, 1)
    literal_v = _literal_nemo_momentum_solve(
        rhs_v, avm, e3vaa[..., :35], e3vwmm[..., 1:35], wv, bot_v,
        drag_v, rdt, 0)
    row31_generic = {
        "u": _column_metric(generic_u, out_u, wu, focus, ACCUMULATING),
        "v": _column_metric(generic_v, out_v, wv, focus, ACCUMULATING),
    }
    row31_literal = {
        "u": _column_metric(literal_u, out_u, wu, focus, ACCUMULATING),
        "v": _column_metric(literal_v, out_v, wv, focus, ACCUMULATING),
    }
    wrong_u = _literal_nemo_momentum_solve(
        rhs_u, avm, e3uaa[..., :35], e3uwmm[..., 1:35], wu, bot_u,
        drag_u, 2700.0, 1)
    row31_controls = {
        "wrong_rdt_fails": not _column_metric(wrong_u, out_u, wu, focus, ACCUMULATING)["pass"],
        "one_cell_roll_fails": not _column_metric(
            np.roll(literal_u, 1, axis=1), out_u, wu, focus, ACCUMULATING)["pass"],
    }

    # Row 32 targeting-only volume-form substitution (ordered behind row 31).
    kt = haloed("stp_dump_23_after_traldf_tem.bin")
    ks = haloed("stp_dump_23_after_traldf_sal.bin")
    nt = haloed("stp_dump_21_trazdf_tem.bin")
    ns = haloed("stp_dump_21_trazdf_sal.bin")
    tin = ((e3bb[..., :35] * np.asarray(before.T)[..., :35]
            + rdt * e3mm[..., :35] * kt) / e3aa[..., :35])
    sin = ((e3bb[..., :35] * np.asarray(before.S)[..., :35]
            + rdt * e3mm[..., :35] * ks) / e3aa[..., :35])
    wi, wj = haloed("eiv_dump_wslpi.bin"), haloed("eiv_dump_wslpj.bin")
    pad = lambda a: np.concatenate([a, np.zeros_like(a[..., :1])], axis=-1)
    ahtu, ahtv = haloed("ldftra_dump_ahtu.bin"), haloed("ldftra_dump_ahtv.bin")
    um3, vm3, wm3 = nemo_iso_face_masks(
        jnp.asarray(ls["u_mask"]), jnp.asarray(ls["v_mask"]), jnp.asarray(tmask))
    geom = ls["grid"]
    _, akz = nemo_iso_a33(
        jnp.asarray(ahtu), um3, vm3, wm3, jnp.asarray(pad(wi)),
        jnp.asarray(pad(wj)), jnp.asarray(geom.dx_u)[:, 1:],
        jnp.asarray(geom.dy_v)[1:, :], jnp.asarray(e3wmm ** 2), dt=rdt,
        msc=bool(ls["gm_cfg"].msc_stabilize), aht_v=jnp.asarray(ahtv))
    k33 = np.asarray(akz)[..., 1:35]
    avt = haloed("dump_avt.bin")
    ktr = avt[..., 1:35] + k33
    gt, gs = implicit_vertical_diffusion_ocean_pair(
        jnp.asarray(tin), jnp.asarray(sin), jnp.asarray(ktr),
        jnp.asarray(e3aa[..., :35]), jnp.asarray(e3wmm[..., 1:35]), rdt)
    row32_target = {
        "temperature": _column_metric(gt, nt, tmask[..., :35], focus, ACCUMULATING),
        "salinity": _column_metric(gs, ns, tmask[..., :35], focus, ACCUMULATING),
    }
    wrong_tin = ((e3bb[..., :35] * np.asarray(before.T)[..., :35]
                  + rdt * e3mm[..., :35] * kt) / e3mm[..., :35])
    wrong_t, _ = implicit_vertical_diffusion_ocean_pair(
        jnp.asarray(wrong_tin), jnp.asarray(sin), jnp.asarray(ktr),
        jnp.asarray(e3aa[..., :35]), jnp.asarray(e3wmm[..., 1:35]), rdt)
    row32_controls = {
        "wrong_e3t_slot_fails": not _column_metric(
            wrong_t, nt, tmask[..., :35], focus, ACCUMULATING)["pass"],
        "one_cell_roll_fails": not _column_metric(
            np.roll(np.asarray(gt), 1, axis=1), nt, tmask[..., :35], focus,
            ACCUMULATING)["pass"],
    }

    row30_pass = all(item["pass"] for item in row30_metrics.values())
    row31_pass = all(item["pass"] for item in row31_generic.values())
    rows = {
        "28": {"disposition": "VERIFIED", "receipt": str(row28_path),
               "receipt_sha256": sha256(row28_path), "metric": row28["metric"]},
        "29": {"disposition": "VERIFIED", "metric": row29,
               "controls": row29_controls,
               "qualification": (
                   "registered census excludes halos; NEMO LBC is not claimed "
                   "to execute in legoESM")},
        "30": {"disposition": "VERIFIED" if row30_pass else "DIVERGED",
               "operands": row30_metrics, "controls": row30_controls,
               "fortran_seos_discriminator": row30_seos_discriminator,
               "first_available_failing_operand": (
                   None if row30_pass else "uslp"),
               "localization_interval": (None if row30_pass else
                   "after exact prd and before final uslp; zgru is not dumped. "
                   "The v sibling is bounded after exact zgrv. Dump zgru, "
                   "zdzr, zau/zbu/limited_zbu/raw_uslp in source order"),
               "nemo_lines": ["ldfslp.F90:217-285"]},
        "31": {"disposition": ("VERIFIED" if row30_pass and row31_pass else
                                  "TARGETING-BLOCKED-BY-ROW30"),
               "generic_production_solver": row31_generic,
               "nemo_literal_discriminator": row31_literal,
               "controls": row31_controls,
               "owner": (None if row31_pass else
                   "implicit matrix construction / Thomas evaluation order"),
               "nemo_lines": ["dynzdf.F90:199-214", "dynzdf.F90:340-380"]},
        "32": {"disposition": ("VERIFIED" if row30_pass and row31_pass and
                                all(x["pass"] for x in row32_target.values())
                                else "TARGETING-BLOCKED-BY-EARLIER-ROW"),
               "targeting_volume_form": row32_target, "controls": row32_controls},
    }
    paths = [row28_path, row28_dump, tail_path, run / "mesh_mask.nc",
             run / dump_lane.RESTART, args.mld_maps.resolve(), seos_so,
             run / "ocean.output", run / "namelist_cfg", run / "namelist_ref"]
    paths.append((Path(__file__).parent / "row30_seos_oracle.f90").resolve())
    executable_link = run / "nemo"
    paths.extend([executable_link, executable_link.resolve()])
    for name in (
        *ldf.CHAIN_DUMPS.values(), *ldf.DUMP_META.values(), "dump_avm.bin",
        "dump_avt.bin", "ldftra_dump_ahtu.bin", "ldftra_dump_ahtv.bin",
        "stp_dump_07_dynspg_u.bin", "stp_dump_07_dynspg_v.bin",
        "stp_dump_07_dynspg_ub.bin", "stp_dump_07_dynspg_vb.bin",
        "stp_dump_08_dynzdf_u.bin", "stp_dump_08_dynzdf_v.bin",
        "stp_dump_23_after_traldf_tem.bin", "stp_dump_23_after_traldf_sal.bin",
        "stp_dump_21_trazdf_tem.bin", "stp_dump_21_trazdf_sal.bin",
        "seq_dump_r3t_aaa.bin", "seq_dump_r3u_aaa.bin", "seq_dump_r3v_aaa.bin",
        "drg_dump_rCdU_bot.bin", "zdf_dump_u1_prestress.bin",
        "zdf_dump_u1_poststress.bin", "zdf_dump_v1_prestress.bin",
        "zdf_dump_v1_poststress.bin",
    ):
        paths.append(run / name)
    source_root = args.nemo_source_root.resolve()
    sources = {
        "ldfslp.F90": source_root / "cfgs/DINO/MY_SRC/ldfslp.F90",
        "dynzdf.F90": source_root / "cfgs/DINO/MY_SRC/dynzdf.F90",
        "trazdf.F90": source_root / "cfgs/DINO/WORK/trazdf.F90",
    }
    artifact = {
        "schema": "dino-zdf-chain-end-v1",
        "disposition": ("DIVERGED-ROW30" if not row30_pass else
                        "DIVERGED-ROW31" if not row31_pass else "VERIFIED"),
        "first_divergence": (30 if not row30_pass else
                             31 if not row31_pass else None),
        "lane": dump_lane.banner(), "repo_sha": args.repo_sha,
        "bars": {"pointwise": POINTWISE, "accumulating": ACCUMULATING},
        "focus_columns_ji": [list(x) for x in focus], "rows": rows,
        "provenance_sha256": {str(p): sha256(p) for p in sorted(set(paths))},
        "oracle_source_sha256": {
            k: {"path": str(v), "sha256": sha256(v)} for k, v in sources.items()},
        "probe": {"path": str(Path(__file__).resolve()),
                  "sha256": sha256(Path(__file__).resolve())},
    }
    row30_control_bools = [v for v in row30_controls.values()
                           if isinstance(v, (bool, np.bool_))]
    if (not all(row29_controls.values())
            or not row30_control_bools or not all(row30_control_bools)
            or not all(row31_controls.values()) or not all(row32_controls.values())
            or not row30_seos_discriminator["pass"]
            or not row30_seos_discriminator["one_ulp_output_control_fails"]):
        raise SystemExit("a planted row31/32 control did not fire")
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"row28 VERIFIED: {row28['metric']['n_diverged_columns']}/9920")
    print(f"row29 VERIFIED: {row29['n_diverged_columns']}/9920")
    print("row30 " + ("VERIFIED" if row30_pass else "DIVERGED") + ": "
          + ", ".join(f"{k}={v['n_diverged_columns']}/{v['n_wet_columns']}"
                      for k, v in row30_metrics.items()))
    print("row31 generic: " + ", ".join(
        f"{k}={v['n_diverged_columns']}/{v['n_wet_columns']}" for k, v in row31_generic.items()))
    print("row31 literal discriminator: " + ", ".join(
        f"{k}={v['n_diverged_columns']}/{v['n_wet_columns']}" for k, v in row31_literal.items()))
    print("row32 targeting: " + ", ".join(
        f"{k}={v['n_diverged_columns']}/9920" for k, v in row32_target.items()))
    print(f"wrote {args.output}")
    return 0 if row30_pass and row31_pass else (30 if not row30_pass else 31)


if __name__ == "__main__":
    raise SystemExit(main())
