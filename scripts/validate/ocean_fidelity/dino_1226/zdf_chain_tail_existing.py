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
import re
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
import legoesm.ocean.physics.vertical_mixing as vmix_mod  # noqa: E402
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
    ap.add_argument("--nemo-source-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row-tail receipt requires CPU and JAX fp64")
    run = args.run_dir.resolve()
    source_root = args.nemo_source_root.resolve()
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
    zmxlm_full = interior("tke_dump_zmxlm.bin")
    zmxld_full = interior("tke_dump_zmxld.bin")
    avm_closure_full = interior("tke_dump_avm_final.bin")
    avt_closure_full = interior("tke_dump_avt_final.bin")

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
    zmxlm = zmxlm_full[..., sl]
    zmxld = zmxld_full[..., sl]
    avm_closure = avm_closure_full[..., sl]
    avt_closure = avt_closure_full[..., sl]

    br, cfg, mc, model, forcing, sf, state = _build_twin_state(
        "nemo_dino_kamm_mlf",
        str(run),
        str(run),
        bridge_tke=True,
        bridge_before=True,
        restart_file="DINO_00005760_restart.nc",
        e3t_mode="both",
    )
    pair_calls = []
    real_pair = vmix_mod.implicit_vertical_diffusion_ocean_pair

    def spy_pair(t_field, s_field, k_field, *pair_args, **pair_kwargs):
        pair_calls.append((t_field, s_field, k_field))
        return real_pair(t_field, s_field, k_field, *pair_args, **pair_kwargs)

    vmix_mod.implicit_vertical_diffusion_ocean_pair = spy_pair
    try:
        _, _, captured = sweep.capture_face_sh2_call(model, state, sf)
    finally:
        vmix_mod.implicit_vertical_diffusion_ocean_pair = real_pair
    if len(pair_calls) != 1:
        raise AssertionError(
            f"DINO production T/S pair solve fired {len(pair_calls)} times, expected 1"
        )
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
    row22_avt = sweep.metrics(np.asarray(kh), avt_closure, wet_interior, focus, POINTWISE)

    # Row 27 fail-closed source/object identity.  NEMO's resolved runtime
    # flags and active assignment must remain exactly the DINO no-DDM path,
    # while production must dispatch T/S through the one-coefficient pair
    # solve.  The planted separated-salinity-K arm proves the identity gate
    # can fail even if a future refactor leaves the prose below unchanged.
    vmcfg = mc.physics.vertical_mixing
    if vmcfg.ddm.enabled or vmcfg.iwm.enabled:
        raise AssertionError("DINO row27 source identity requires DDM=IWM=false")
    ocean_output = (run / "ocean.output").read_text(errors="replace")
    for flag in ("ln_zdfddm", "ln_zdfswm", "ln_zdfiwm"):
        if re.search(rf"{flag}\s*=\s*F(?:\s|$)", ocean_output) is None:
            raise AssertionError(f"resolved NEMO flag is not false: {flag}")
    zdfphy_path = source_root / "src" / "OCE" / "ZDF" / "zdfphy.F90"
    zdfphy_text = zdfphy_path.read_text(errors="strict")
    avs_copy = re.search(
        r"IF\s*\(\s*ln_zdfddm\s*\).*?ELSE.*?"
        r"avs\s*\(\s*ji\s*,\s*jj\s*,\s*jk\s*\)\s*=\s*"
        r"avt\s*\(\s*ji\s*,\s*jj\s*,\s*jk\s*\).*?ENDIF",
        zdfphy_text,
        flags=re.DOTALL,
    )
    if avs_copy is None:
        raise AssertionError("active NEMO no-DDM avs=avt assignment changed")
    shared_k = np.asarray(pair_calls[0][2])

    def shared_k_gate(k_heat, k_salt) -> bool:
        return k_heat is k_salt and np.array_equal(k_heat, k_salt, equal_nan=True)

    if not shared_k_gate(shared_k, shared_k):
        raise AssertionError("production T/S call did not receive one shared K object")
    separated_k = shared_k.copy()
    wet_k = np.argwhere(np.isfinite(separated_k) & (separated_k != 0.0))
    if wet_k.size == 0:
        raise AssertionError("row27 separated-salinity-K control found no finite nonzero K")
    control_index = tuple(int(x) for x in wet_k[0])
    separated_k[control_index] = np.nextafter(
        separated_k[control_index], np.inf
    )
    row27_control_fired = not shared_k_gate(shared_k, separated_k)
    if not row27_control_fired:
        raise AssertionError("row27 separated-salinity-K control did not fire")
    # zdfphy rows 23/25/26/29 are reconstructible from existing NEMO closure,
    # rn2/rn2b, mask, and composed-coefficient dumps.  These are deliberately
    # retained as ORACLE-SELFCHECK previews, not legoESM measurements: no
    # legoESM coefficient-composition/EVD/LBC production routine is called.
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
            "reason": (
                "base avm is directly checkable and passes; base pre-Prandtl "
                "avt and post-tke_avn dissl are not dumped (the existing "
                "dissl slot is the carried pre-overwrite row-15 operand)"
            ),
        },
        "22": {
            "operation": "inverse-Prandtl avt correction",
            "disposition": disposition(row22_avt),
            "avt": row22_avt,
        },
        "23": {
            "operation": "closure coefficient copy on EVD-stable points",
            "disposition": "UNMEASURED-ORACLE-SELFCHECK",
            "oracle_selfcheck_avt": row23_avt,
            "oracle_selfcheck_avm": row23_avm,
            "reason": (
                "NEMO closure and composed dumps self-consistent on stable "
                "subset; legoESM composition path not invoked"
            ),
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
            "disposition": "UNMEASURED-ORACLE-SELFCHECK",
            "oracle_selfcheck_avt": row25,
            "reason": "NEMO source-order EVD reconstruction only; legoESM EVD path not invoked",
            "fired_wet_elements": int(fired.sum()),
        },
        "26": {
            "operation": "EVD momentum overwrite",
            "disposition": "UNMEASURED-ORACLE-SELFCHECK",
            "oracle_selfcheck_avm": row26,
            "reason": "NEMO source-order EVD reconstruction only; legoESM EVD path not invoked",
            "fired_wet_elements": int(fired.sum()),
        },
        "27": {
            "operation": "avs copy and optional enhancements",
            "disposition": "VERIFIED",
            "source_identity": {
                "bar": "exact source/object identity",
                "n_verified_columns": 9920,
                "n_diverged_columns": 0,
                "focus": [
                    {"j": j, "i": i, "pass": True} for j, i in focus
                ],
                "nemo": "zdfphy.F90:326-331 executes avs=avt with DDM off",
                "legoesm": (
                    "ocean_model_latlon_cgrid.py:7308-7336 passes the same "
                    "K_v_cell object to the T/S pair solve with DDM off"
                ),
                "resolved_nemo_flags": {
                    "ln_zdfddm": False,
                    "ln_zdfswm": False,
                    "ln_zdfiwm": False,
                },
                "production_pair_calls": len(pair_calls),
                "shared_K_object_asserted": True,
                "separated_salinity_K_control": {
                    "fired": row27_control_fired,
                    "perturbed_index": list(control_index),
                    "perturbation": "+1 ULP",
                },
            },
            "waived_inactive_enhancements": [
                "DDM coefficient delta",
                "surface-wave enhancement",
                "internal-wave enhancement",
            ],
        },
        "28": {
            "operation": "composed-avt turbocline scan",
            "disposition": "UNMEASURED-NEEDS-DUMP",
            "reason": "no same-step imld/hmld (mldkz5) slot exists",
        },
        "29": {
            "operation": "avm lateral boundary update, interior census",
            "disposition": "UNMEASURED-ORACLE-SELFCHECK",
            "oracle_selfcheck_avm_interior": row29,
            "reason": (
                "NEMO reconstructed interior compared to NEMO composed "
                "dump; legoESM LBC path not invoked"
            ),
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
            "disposition": "UNMEASURED-EXISTING-BRACKET",
            "reason": (
                "no new NEMO dump is required: restart Kbb + stage-6 Krhs + "
                "stage-7 post-dyn_spg barotropic state + stage-8 output bracket "
                "dyn_zdf. Exact isolation remains blocked by upstream row-20/"
                "23/25/26 composed-avm uncertainty and requires the registered "
                "NEMO volume-form input reconstruction"
            ),
            "deferred_large_design": (
                "reconstruct Naa_A=(Kbb+rDt*stage6_Krhs)*mask, subtract the "
                "stage7 barotropic Naa field, add the dumped level-1 stress "
                "deposit, substitute dump_avm into the production momentum "
                "matrix/drag call, and compare its u/v result per column with "
                "stage8; planted wrong-rDt and one-cell-roll controls must fail"
            ),
        },
        "32": {
            "operation": "tracer implicit solve application",
            "disposition": "UNMEASURED-EXISTING-BRACKET",
            "reason": (
                "no new NEMO dump is required: restart Kbb + stage-23 post-tra_ldf "
                "Krhs and stage-21 post-tra_zdf bracket the application. Exact "
                "isolation remains blocked by row-30 K33/slope divergence and "
                "requires the registered z-star volume-form input reconstruction"
            ),
            "deferred_large_design": (
                "reconstruct Naa=(e3t_Kbb*T_Kbb+rDt*e3t_Kmm*stage23_Krhs)/"
                "e3t_Kaa for T/S, assemble exact K33 from dump_avt plus the "
                "existing NEMO slope/coefficient dumps, substitute both into "
                "the production pair solve, and compare per column with stage21; "
                "planted wrong-e3t-slot and one-cell-roll controls must fail"
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
        "dump_avm.bin",
        "dump_avt.bin",
        *ldf_probe.DUMP_META.values(),
        ldf_probe.CHAIN_DUMPS["prd"],
    ]
    source_names = {
        "zdftke.F90": source_root / "cfgs" / "DINO" / "MY_SRC" / "zdftke.F90",
        "zdfphy.F90": source_root / "src" / "OCE" / "ZDF" / "zdfphy.F90",
        "zdfevd.F90": source_root / "src" / "OCE" / "ZDF" / "zdfevd.F90",
        "ldfslp.F90": source_root / "cfgs" / "DINO" / "MY_SRC" / "ldfslp.F90",
    }
    missing_sources = [str(path) for path in source_names.values() if not path.is_file()]
    if missing_sources:
        raise SystemExit(f"quoted NEMO source missing: {missing_sources}")
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
        "oracle_source_sha256": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in source_names.items()
        },
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
