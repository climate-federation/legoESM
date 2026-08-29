#!/usr/bin/env python
"""Qualified ZDF rows 19--32 using only existing day-180 dumps.

Row 18 is deliberately still open.  Results from this probe are therefore
``PROVISIONAL-DOWNSTREAM`` receipts: they identify work that can be cleared or
targeted without a NEMO rebuild, but cannot advance the ordered verified
frontier beyond row 17.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import jax
import jax.numpy as jnp
import netCDF4  # noqa: N813
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bn2_alpha_compare as loaders  # noqa: E402
import dump_lane  # noqa: E402
import ldf_slp_per_element as ldf_probe  # noqa: E402
import zdf_chain_sweep as sweep  # noqa: E402
from kamm_twin_90d import _build_twin_state  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.fidelity.time_levels import time_level_for_dump  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.tke import (  # noqa: E402
    compute_K_from_tke,
    compute_mixing_lengths,
)

POINTWISE = 1.0e-15
ACCUMULATING = 1.0e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def disposition(metric: dict) -> str:
    return "VERIFIED" if metric["pass"] else "DIVERGED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--mld-maps", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row-tail receipt requires CPU and JAX fp64")
    run = args.run_dir.resolve()
    if Path(dump_lane.RUN_DIR).resolve() != run:
        raise SystemExit(
            f"dump_lane resolves to {Path(dump_lane.RUN_DIR).resolve()}, not requested run {run}"
        )
    if dump_lane.LANE != "d180" or dump_lane.KT_DUMP != 5761:
        raise SystemExit("only the registered d180/kt=5761 lane is accepted")

    focus = sweep.focus_from_maps(args.mld_maps)
    jpi, jpj, jpk, hls = loaders._read_dims(str(run))
    ni, nj = jpi - 2 * hls, jpj - 2 * hls

    def interior(name: str) -> np.ndarray:
        time_level_for_dump(name)
        return loaders._load_interior(str(run / name), ni, nj)

    def haloed(name: str) -> np.ndarray:
        time_level_for_dump(name)
        return loaders._load_haloed(str(run / name), jpi, jpj, hls)

    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        tmask = np.moveaxis(np.asarray(ds["tmask"][0]), 0, -1) > 0.5
    wmask = np.zeros_like(tmask)
    wmask[..., 0] = tmask[..., 0]
    wmask[..., 1:] = tmask[..., 1:] & tmask[..., :-1]
    wet_interior = wmask[..., : jpk - 1].copy()
    # zdfphy closure-copy/EVD loops scored here start at Fortran jk=2.
    wet_zdfphy = wet_interior.copy()
    wet_zdfphy[..., 0] = False
    if int(np.any(wet_interior, axis=-1).sum()) != 9920:
        raise AssertionError("whole-domain wet-column census changed")

    en_full = interior("tke_dump_en.bin")
    rn2_full = interior("tke_dump_rn2.bin")
    rn2b_full = interior("tke_dump_rn2b.bin")
    sh2_full = interior("tke_dump_sh2.bin")
    avm_in_full = interior("tke_dump_avm_in.bin")
    pdlr_full = interior("tke_dump_pdlr.bin")
    zmxlm_full = interior("tke_dump_zmxlm.bin")
    zmxld_full = interior("tke_dump_zmxld.bin")
    avm_closure_full = interior("tke_dump_avm_final.bin")
    avt_closure_full = interior("tke_dump_avt_final.bin")
    dissl_full = interior("tke_dump_dissl.bin")

    # The closure/TKE bridge maps its 35 W rows to NEMO jk=1..jpkm1.  This is
    # distinct from the row-17 solve census, which excludes the prescribed
    # surface row and scores jk=2..jpkm1.  Including NEMO's never-written jpk
    # pad here manufactures one floor-sized miss in every column.
    sl = slice(0, jpk - 1)
    en = en_full[..., sl]
    rn2 = rn2_full[..., sl]
    rn2b = rn2b_full[..., sl]
    sh2 = sh2_full[..., sl]
    avm_in = avm_in_full[..., sl]
    pdlr = pdlr_full[..., sl]
    zmxlm = zmxlm_full[..., sl]
    zmxld = zmxld_full[..., sl]
    avm_closure = avm_closure_full[..., sl]
    avt_closure = avt_closure_full[..., sl]
    dissl = dissl_full[..., sl]

    br, cfg, mc, model, forcing, sf, state = _build_twin_state(
        "nemo_dino_kamm_mlf",
        str(run),
        str(run),
        bridge_tke=True,
        bridge_before=True,
        restart_file="DINO_00005760_restart.nc",
        e3t_mode="both",
    )
    _, _, captured = sweep.capture_face_sh2_call(model, state, sf)
    mxl_args, mxl_kwargs, _ = captured["mxl_calls"][-1]
    if len(mxl_args) != 4 or en.shape != np.asarray(mxl_args[0]).shape:
        raise AssertionError("production mixing-length call contract changed")

    # Isolate rows 19+20 by substituting exact NEMO post-row18 en and rn2 into
    # the real production implementation, retaining every production geometry
    # and configuration operand captured at the call site.
    lk, leps = compute_mixing_lengths(
        jnp.asarray(en), jnp.asarray(rn2), mxl_args[2], mxl_args[3], **mxl_kwargs
    )
    row20_lk = sweep.metrics(np.asarray(lk), zmxlm, wet_interior, focus, ACCUMULATING)
    row20_leps = sweep.metrics(np.asarray(leps), zmxld, wet_interior, focus, ACCUMULATING)

    tke_cfg = mc.physics.vertical_mixing.tke

    def p_sh2_override(_):
        return jnp.asarray(sh2)

    km, kh = compute_K_from_tke(
        jnp.asarray(en),
        jnp.asarray(zmxlm),
        tke_cfg,
        N2=jnp.asarray(rn2),
        shear_sq=jnp.ones_like(jnp.asarray(sh2)),
        N2_prandtl=jnp.asarray(rn2b),
        p_sh2_override=p_sh2_override,
        prandtl_K_M=jnp.asarray(avm_in),
    )
    row21_avm = sweep.metrics(np.asarray(km), avm_closure, wet_interior, focus, POINTWISE)
    dissl_lego = np.divide(np.sqrt(en), zmxld, out=np.zeros_like(en), where=zmxld > 0.0)
    row21_dissl = sweep.metrics(dissl_lego, dissl, wet_interior, focus, POINTWISE)
    row22_avt = sweep.metrics(np.asarray(kh), avt_closure, wet_interior, focus, POINTWISE)
    row22_pdlr = sweep.metrics(np.asarray(km) * 0.0 + pdlr, pdlr, wet_interior, focus, POINTWISE)

    # zdfphy rows 23/25/26/29 are completely reconstructible from existing
    # closure, rn2/rn2b, mask, and composed-coefficient dumps.
    avt_composed = haloed("dump_avt.bin")
    avm_composed = haloed("dump_avm.bin")
    closure_avt_z = avt_closure_full[..., : jpk - 1]
    closure_avm_z = avm_closure_full[..., : jpk - 1]
    rn2_z = rn2_full[..., : jpk - 1]
    rn2b_z = rn2b_full[..., : jpk - 1]
    fired = wet_zdfphy & (np.minimum(rn2_z, rn2b_z) <= -1.0e-12)
    stable = wet_zdfphy & ~fired
    predicted_avt = np.where(fired, 100.0 * wet_interior, closure_avt_z)
    predicted_avm = np.where(fired, 100.0 * wet_interior, closure_avm_z)
    row23_avt = sweep.metrics(closure_avt_z, avt_composed, stable, focus, POINTWISE)
    row23_avm = sweep.metrics(closure_avm_z, avm_composed, stable, focus, POINTWISE)
    row25 = sweep.metrics(predicted_avt, avt_composed, fired, focus, POINTWISE)
    row26 = sweep.metrics(predicted_avm, avm_composed, fired, focus, POINTWISE)
    row29 = sweep.metrics(predicted_avm, avm_composed, wet_zdfphy, focus, POINTWISE)

    # Row 30: call the existing production-native slope probe once, capture
    # its own first intermediate (prd), and score all four returned slopes.
    ldf_state = ldf_probe.build_state()
    _, ldf_locals = ldf_probe.capture_locals(
        ldf_state["recall"], ldf_probe.compute_nemo_native_slopes.__code__
    )
    row30_fields = {}
    for name, dump_name in ldf_probe.DUMP_META.items():
        lego = np.asarray(ldf_state["lego"][name])
        nemo = loaders._load_haloed(str(run / dump_name), jpi, jpj, hls)
        nk = min(lego.shape[-1], nemo.shape[-1])
        wet = ldf_probe.MASK_FN[name](ldf_state)[..., :nk]
        row30_fields[name] = sweep.metrics(lego[..., :nk], nemo[..., :nk], wet, focus, POINTWISE)
    prd_n = loaders._load_haloed(str(run / ldf_probe.CHAIN_DUMPS["prd"]), jpi, jpj, hls)
    prd_l = np.asarray(ldf_locals["prd"])
    nk_prd = min(prd_l.shape[-1], prd_n.shape[-1])
    wet_prd = ldf_probe.wet_w_mask(ldf_state["active"])[..., :nk_prd]
    wet_prd[..., 0] = False
    if nk_prd > ldf_probe.KHI:
        wet_prd[..., ldf_probe.KHI :] = False
    row30_prd = sweep.metrics(prd_l[..., :nk_prd], prd_n[..., :nk_prd], wet_prd, focus, POINTWISE)

    numeric_controls = {
        "row20_lk": sweep.planted_controls(zmxlm, zmxlm, wet_interior, ACCUMULATING),
        "row22_avt": sweep.planted_controls(avt_closure, avt_closure, wet_interior, POINTWISE),
        "row25_avt": sweep.planted_controls(avt_composed, avt_composed, fired, POINTWISE),
        "row30_prd": sweep.planted_controls(
            prd_n[..., :nk_prd], prd_n[..., :nk_prd], wet_prd, POINTWISE
        ),
    }

    rows = {
        "19": {
            "operation": "raw buoyancy mixing length before scans",
            "disposition": "UNMEASURED-NEEDS-DUMP",
            "reason": "existing zmxlm/zmxld slots are post-scan; no raw pre-limit slot",
        },
        "20": {
            "operation": "nn_mxl=3 limiting scans (row19+20 composite)",
            "disposition": ("VERIFIED" if row20_lk["pass"] and row20_leps["pass"] else "DIVERGED"),
            "zmxlm": row20_lk,
            "zmxld": row20_leps,
            "qualification": (
                "composite isolation using exact NEMO en/rn2; row19 cannot "
                "be separated without a raw slot"
            ),
        },
        "21": {
            "operation": "base avm/avt/dissl assembly",
            "disposition": "UNMEASURED-NEEDS-DUMP",
            "avm_preview": row21_avm,
            "dissl_preview": row21_dissl,
            "reason": "avm and dissl are directly checkable; base pre-Prandtl avt is not dumped",
        },
        "22": {
            "operation": "inverse-Prandtl avt correction",
            "disposition": disposition(row22_avt),
            "avt": row22_avt,
            "pdlr_input_identity": row22_pdlr,
        },
        "23": {
            "operation": "closure coefficient copy on EVD-stable points",
            "disposition": ("VERIFIED" if row23_avt["pass"] and row23_avm["pass"] else "DIVERGED"),
            "avt": row23_avt,
            "avm": row23_avm,
        },
        "24": {
            "operation": "river-mouth enhancement",
            "disposition": "WAIVED",
            "reason": (
                "ln_rnf=F and reference ln_rnf_mouth=.false.; branch zdfphy.F90:317-321 is inactive"
            ),
        },
        "25": {
            "operation": "EVD tracer overwrite",
            "disposition": disposition(row25),
            "avt": row25,
            "fired_wet_elements": int(fired.sum()),
        },
        "26": {
            "operation": "EVD momentum overwrite",
            "disposition": disposition(row26),
            "avm": row26,
            "fired_wet_elements": int(fired.sum()),
        },
        "27": {
            "operation": "avs copy and optional enhancements",
            "disposition": "UNMEASURED-NEEDS-DUMP",
            "reason": (
                "live avs=avt copy lacks an avs slot; DDM, surface-wave, and "
                "internal-wave branches are WAIVED by resolved false flags"
            ),
        },
        "28": {
            "operation": "composed-avt turbocline scan",
            "disposition": "UNMEASURED-NEEDS-DUMP",
            "reason": "no same-step imld/hmld (mldkz5) slot exists",
        },
        "29": {
            "operation": "avm lateral boundary update, interior census",
            "disposition": disposition(row29),
            "avm_interior": row29,
            "qualification": "halo exchange is outside the registered interior column census",
        },
        "30": {
            "operation": "ldf_slp",
            "disposition": (
                "VERIFIED" if all(x["pass"] for x in row30_fields.values()) else "DIVERGED"
            ),
            "fields": row30_fields,
            "first_failing_operand": {
                "name": "prd argument entering ldf_slp",
                "metric": row30_prd,
                "nemo_line": (
                    "cfgs/DINO/MY_SRC/ldfslp.F90:202-213 (prd input; first consumed operand)"
                ),
                "design": (
                    "separate the EOS-produced prd arithmetic from downstream "
                    "zgrv with the existing prd slot; literal SEOS association "
                    "is the next targeted production option only after row18 closes"
                ),
            },
        },
        "31": {
            "operation": "momentum implicit solve application",
            "disposition": "UNMEASURED-NEEDS-DUMP",
            "reason": (
                "stage-7 Naa aliases Krhs and is a tendency, not the true "
                "pre-solve velocity; no existing dump brackets Krhs after "
                "dyn_spg and before dyn_zdf"
            ),
        },
        "32": {
            "operation": "tracer implicit solve application",
            "disposition": "UNMEASURED-NEEDS-DUMP",
            "reason": (
                "post-tra_zdf Naa exists, but no same-step pre-tra_zdf Naa "
                "state isolates the implicit tracer application"
            ),
        },
    }

    consumed = [
        "mesh_mask.nc",
        "DINO_00005760_restart.nc",
        "ocean.output",
        "tke_dump_en.bin",
        "tke_dump_rn2.bin",
        "tke_dump_rn2b.bin",
        "tke_dump_sh2.bin",
        "tke_dump_avm_in.bin",
        "tke_dump_pdlr.bin",
        "tke_dump_zmxlm.bin",
        "tke_dump_zmxld.bin",
        "tke_dump_avm_final.bin",
        "tke_dump_avt_final.bin",
        "tke_dump_dissl.bin",
        "dump_avm.bin",
        "dump_avt.bin",
        *ldf_probe.DUMP_META.values(),
        ldf_probe.CHAIN_DUMPS["prd"],
    ]
    artifact = {
        "schema": "dino-zdf-chain-tail-existing-v1",
        "status": "PROVISIONAL-DOWNSTREAM; ordered frontier remains row 17 while row 18 is open",
        "repo_sha": git_sha(),
        "lane": dump_lane.banner(),
        "run_dir": str(run),
        "focus_columns_ji": [list(x) for x in focus],
        "wet_columns": 9920,
        "bars": {"pointwise": POINTWISE, "accumulating": ACCUMULATING},
        "rows": rows,
        "controls": numeric_controls,
        "provenance_sha256": {name: sha256(run / name) for name in sorted(set(consumed))},
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": {k: v["disposition"] for k, v in rows.items()},
                "row20_fail_columns": {
                    "zmxlm": row20_lk["n_diverged_columns"],
                    "zmxld": row20_leps["n_diverged_columns"],
                },
                "row22_fail_columns": row22_avt["n_diverged_columns"],
                "row25_fail_columns": row25["n_diverged_columns"],
                "row26_fail_columns": row26["n_diverged_columns"],
                "row29_fail_columns": row29["n_diverged_columns"],
                "row30_fail_columns": {k: v["n_diverged_columns"] for k, v in row30_fields.items()},
                "row30_prd_fail_columns": row30_prd["n_diverged_columns"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
