#!/usr/bin/env python
"""Qualified ZDF rows 19--32 using existing day-180 dumps.

Row 19 is accepted only through its strict deterministic-writer receipt.  Row
20 is the registered row-19+20 composite.  Row 21 is accepted only through its
strict five-slot receipt.  Rows 22--27 then use the production TKE/profile
composer captures; the ordered frontier stops at row 28's missing direct
turbocline index/depth slot.
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
import legoesm.ocean.physics.vertical_mixing.k_profiles as kprofiles_mod  # noqa: E402
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
ROW19_RECEIPT_SHA256 = "f5e42f1d15cd9e823e81fa3f5b56a2b9eebd504c8ef823718c50dd9b9f3fc29b"
ROW21_RECEIPT_SHA256 = "84885e45ecc149082606c0b44b411271f40942a99497996b0d4b35e051b6a97a"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def tracked_tree_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()


def disposition(metric: dict) -> str:
    return "VERIFIED" if metric["pass"] else "DIVERGED"


def _enforce_ordered_stop(rows: dict[str, dict]) -> str | None:
    """Block every later row after the first promoted divergence."""
    promoted_order = tuple(str(row) for row in range(20, 28))
    first_diverged = next(
        (row for row in promoted_order if rows[row]["disposition"] == "DIVERGED"),
        None,
    )
    if first_diverged is None:
        return None
    stop_index = int(first_diverged)
    for row in range(stop_index + 1, 33):
        key = str(row)
        previous = rows[key]["disposition"]
        rows[key]["targeting_preview_disposition"] = previous
        rows[key]["disposition"] = f"UNMEASURED-BLOCKED-BY-ROW{first_diverged}"
        rows[key]["ordered_block_reason"] = (
            f"ordered promotion stopped at first divergence row {first_diverged}"
        )
    return first_diverged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--row19-run-dir", type=Path, required=True)
    ap.add_argument("--row19-artifact", type=Path, required=True)
    ap.add_argument("--row21-artifact", type=Path, required=True)
    ap.add_argument("--mld-maps", type=Path, required=True)
    ap.add_argument("--nemo-source-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row-tail receipt requires CPU and JAX fp64")
    if not tracked_tree_clean():
        raise SystemExit("row-tail receipt requires a clean tracked tree")
    run = args.run_dir.resolve()
    row19_run_dir = args.row19_run_dir.resolve()
    row19_path = args.row19_artifact.resolve()
    row21_path = args.row21_artifact.resolve()
    source_root = args.nemo_source_root.resolve()
    if Path(dump_lane.RUN_DIR).resolve() != run:
        raise SystemExit(
            f"dump_lane resolves to {Path(dump_lane.RUN_DIR).resolve()}, not requested run {run}"
        )
    if dump_lane.LANE != "d180" or dump_lane.KT_DUMP != 5761:
        raise SystemExit("only the registered d180/kt=5761 lane is accepted")

    row19_receipt = json.loads(row19_path.read_text())
    row19_metric = row19_receipt.get("production_metric", {})
    row19_bracket = row19_receipt.get("bracket", {})
    row19_controls = row19_receipt.get("controls", {})
    row19_provenance = row19_receipt.get("provenance_sha256", {})
    row19_run = str(row19_run_dir / "tke_dump_zmxlm_raw.bin")
    if (
        sha256(row19_path) != ROW19_RECEIPT_SHA256
        or row19_receipt.get("schema") != "dino-zdf-row19-raw-mxl-v1"
        or row19_receipt.get("disposition") != "VERIFIED"
        or row19_metric.get("pass") is not True
        or row19_metric.get("n_diverged_columns") != 0
        or row19_metric.get("n_wet_columns") != 9920
        or row19_receipt.get("exact_unequal_wet_elements") != 0
        or row19_receipt.get("source_formula_unequal_wet_elements") != 0
        or row19_bracket.get("determinism_stream_count") != 198
        or row19_bracket.get("shared_stream_count") != 197
        or row19_bracket.get("strict_byte_identical_stream_count") != 197
        or row19_bracket.get("uninitialized_memory_exclusions") != []
        or row19_provenance.get(row19_run)
        != sha256(row19_run_dir / "tke_dump_zmxlm_raw.bin")
        or not row19_controls
        or not all(row19_controls.values())
        or len(row19_metric.get("focus", [])) != 4
        or not all(item.get("pass") is True for item in row19_metric["focus"])
    ):
        raise SystemExit("row19 deterministic-writer receipt is not promotable")

    row21_receipt = json.loads(row21_path.read_text())
    row21_rows = row21_receipt.get("rows", {})
    row21_bracket = row21_receipt.get("bracket", {})
    row21_controls = row21_receipt.get("controls", {})
    row21_bracket_controls = row21_receipt.get("bracket_controls", {})
    row21_provenance = row21_receipt.get("provenance_sha256", {})
    row21_paths = row21_receipt.get("provenance_paths", {})
    expected_row21_dumps = {
        "zsqen_base", "zav_base", "avm_base", "avt_base", "dissl_postavn"
    }
    if (
        sha256(row21_path) != ROW21_RECEIPT_SHA256
        or row21_receipt.get("schema") != "dino-zdf-row21-coeff-assembly-v1"
        or row21_receipt.get("disposition") != "VERIFIED"
        or row21_receipt.get("first_divergence") is not None
        or set(row21_rows) != expected_row21_dumps
        or not all(item.get("metric", {}).get("pass") is True for item in row21_rows.values())
        or not all(item["metric"].get("n_diverged_columns") == 0 for item in row21_rows.values())
        or not all(
            item["metric"].get("exact_unequal_wet_elements") == 0
            for item in row21_rows.values()
        )
        or row21_bracket.get("on_stream_count") != 202
        or row21_bracket.get("off_stream_count") != 197
        or row21_bracket.get("strict_byte_identical_shared_stream_count") != 197
        or row21_bracket.get("exclusions") != []
        or not row21_controls
        or not all(all(control.values()) for control in row21_controls.values())
        or not all(row21_bracket_controls.values())
    ):
        raise SystemExit("row21 five-slot receipt is not promotable")
    for name in expected_row21_dumps:
        key = f"dump_{name}"
        expected_path = str(run / f"tke_dump_{name}.bin")
        if (
            row21_paths.get(key) != expected_path
            or row21_provenance.get(key) != sha256(Path(expected_path))
        ):
            raise SystemExit(f"row21 receipt does not bind current run dump: {name}")

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
    # Production's TKE bridge drops NEMO's prescribed surface W row:
    # restart/dump ``[..., 1:]`` -> lego interior interfaces ``0..jpk-2``.
    # The former ``:jpk-1`` slice compared the wrong row at every level and
    # manufactured the provisional all-column row-20 divergence.
    wet_interior = wmask[..., 1:jpk].copy()
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
    sl = slice(1, jpk)
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
    apply_calls = []
    profile_calls = []
    closure_calls = []
    evd_calls = []
    real_pair = vmix_mod.implicit_vertical_diffusion_ocean_pair
    real_profiles = vmix_mod.compute_vertical_K_profiles
    real_closure = kprofiles_mod._vmix_K_profiles
    real_evd = kprofiles_mod._enhanced_diffusion_K
    model_cls = type(model)
    real_apply = model_cls._apply_implicit_vertical_mixing

    def spy_pair(t_field, s_field, k_field, *pair_args, **pair_kwargs):
        pair_calls.append((t_field, s_field, k_field))
        return real_pair(t_field, s_field, k_field, *pair_args, **pair_kwargs)

    def spy_apply(self, *apply_args, **apply_kwargs):
        apply_calls.append((apply_args, apply_kwargs))
        return real_apply(self, *apply_args, **apply_kwargs)

    def spy_profiles(*profile_args, **profile_kwargs):
        result = real_profiles(*profile_args, **profile_kwargs)
        profile_calls.append((profile_args, profile_kwargs, result))
        return result

    def spy_closure(*closure_args, **closure_kwargs):
        result = real_closure(*closure_args, **closure_kwargs)
        closure_calls.append((closure_args, closure_kwargs, result))
        return result

    def spy_evd(*evd_args, **evd_kwargs):
        result = real_evd(*evd_args, **evd_kwargs)
        evd_calls.append((evd_args, evd_kwargs, result))
        return result

    vmix_mod.implicit_vertical_diffusion_ocean_pair = spy_pair
    vmix_mod.compute_vertical_K_profiles = spy_profiles
    kprofiles_mod._vmix_K_profiles = spy_closure
    kprofiles_mod._enhanced_diffusion_K = spy_evd
    model_cls._apply_implicit_vertical_mixing = spy_apply
    try:
        _, _, captured = sweep.capture_face_sh2_call(model, state, sf)
    finally:
        vmix_mod.implicit_vertical_diffusion_ocean_pair = real_pair
        vmix_mod.compute_vertical_K_profiles = real_profiles
        kprofiles_mod._vmix_K_profiles = real_closure
        kprofiles_mod._enhanced_diffusion_K = real_evd
        model_cls._apply_implicit_vertical_mixing = real_apply
    if len(pair_calls) != 1:
        raise AssertionError(
            f"DINO production T/S pair solve fired {len(pair_calls)} times, expected 1"
        )
    if len(apply_calls) != 1:
        raise AssertionError(
            f"DINO production implicit call fired {len(apply_calls)} times, expected 1"
        )
    if len(profile_calls) != 1 or len(closure_calls) != 1 or len(evd_calls) != 1:
        raise AssertionError(
            "DINO production profile composer did not fire exactly once: "
            f"profiles={len(profile_calls)} closure={len(closure_calls)} "
            f"evd={len(evd_calls)}"
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

    profile_k = np.asarray(profile_calls[0][2][0])
    profile_a = np.asarray(profile_calls[0][2][1])
    closure_k = np.asarray(closure_calls[0][2][0])
    closure_a = np.asarray(closure_calls[0][2][1])
    evd_k = np.asarray(evd_calls[0][2][0])
    evd_a = np.asarray(evd_calls[0][2][1])
    expected_shape = wet_interior.shape
    for name, field in (
        ("profile avt", profile_k), ("profile avm", profile_a),
        ("closure avt", closure_k), ("closure avm", closure_a),
        ("EVD avt", evd_k), ("EVD avm", evd_a),
    ):
        if field.shape != expected_shape:
            raise AssertionError(
                f"production {name} shape changed: {field.shape} vs {expected_shape}"
            )

    row23_avt = sweep.metrics(
        closure_k, avt_closure, wet_interior, focus, POINTWISE)
    row23_avm = sweep.metrics(
        closure_a, avm_closure, wet_interior, focus, POINTWISE)

    # NEMO's composed coefficient streams contain jk=1..jpkm1.  legoESM's
    # interior W state maps to jk=2..jpk, so drop the prescribed surface row
    # and append the dry jpk terminal.  The terminal is outside the wet census.
    def composed_to_legoesm(name: str) -> np.ndarray:
        nemo = haloed(name)
        if nemo.shape != (nj, ni, jpk - 1):
            raise AssertionError(f"composed coefficient shape changed: {nemo.shape}")
        return np.concatenate(
            [nemo[..., 1:], np.zeros_like(nemo[..., :1])], axis=-1)

    avt_composed_lego = composed_to_legoesm("dump_avt.bin")
    avm_composed_lego = composed_to_legoesm("dump_avm.bin")
    row25 = sweep.metrics(
        profile_k, avt_composed_lego, wet_interior, focus, POINTWISE)
    row26 = sweep.metrics(
        profile_a, avm_composed_lego, wet_interior, focus, POINTWISE)
    # Row 29 is the post-LBC INTERIOR census.  NEMO's lateral-boundary update
    # changes only halos, which are deliberately outside this registered
    # census; the compared interior is therefore the same production-composed
    # avm pair as row 26.  Give the promoted row its own fail-capable controls
    # rather than inheriting row 26's EVD branch-mask control.
    row29_controls = sweep.planted_controls(
        profile_a, avm_composed_lego, wet_interior, POINTWISE)
    nemo_evd_mask = wet_interior & (np.minimum(rn2, rn2b) <= -1.0e-12)
    lego_evd_mask = wet_interior & (evd_k == 100.0) & (evd_a == 100.0)
    evd_xor = wet_interior & (nemo_evd_mask != lego_evd_mask)
    evd_bad_columns = np.any(evd_xor, axis=-1)
    evd_wet_columns = np.any(wet_interior, axis=-1)
    evd_focus = [
        {
            "j": j,
            "i": i,
            "wet": bool(evd_wet_columns[j, i]),
            "xor_levels": int(evd_xor[j, i].sum()),
            "pass": bool(evd_wet_columns[j, i] and not evd_bad_columns[j, i]),
        }
        for j, i in focus
    ]
    row25_26_mask = {
        "bar": "exact branch-mask equality",
        "n_wet_columns": int(evd_wet_columns.sum()),
        "n_diverged_columns": int(evd_bad_columns.sum()),
        "n_verified_columns": int(evd_wet_columns.sum() - evd_bad_columns.sum()),
        "n_xor_wet_elements": int(evd_xor.sum()),
        "n_fired_wet_elements": int(nemo_evd_mask.sum()),
        "focus": evd_focus,
        "pass": bool(not evd_bad_columns.any()),
    }

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
    namelist_ref = (run / "namelist_ref").read_text(errors="strict")
    def row24_inactive(output_text, namelist_text, source_text):
        return (
            re.search(r"ln_rnf\s*=\s*F(?:\s|$)", output_text) is not None
            and re.search(
                r"ln_rnf_mouth\s*=\s*\.false\.", namelist_text,
                flags=re.IGNORECASE) is not None
            and re.search(
                r"IF\s*\(\s*ln_rnf_mouth\s*\)\s*THEN",
                source_text) is not None)

    rnf_disabled = re.search(
        r"ln_rnf\s*=\s*F(?:\s|$)", ocean_output) is not None
    rnf_mouth_disabled = re.search(
        r"ln_rnf_mouth\s*=\s*\.false\.", namelist_ref,
        flags=re.IGNORECASE) is not None
    rnf_source_live = re.search(
        r"IF\s*\(\s*ln_rnf_mouth\s*\)\s*THEN", zdfphy_text) is not None
    if not (rnf_disabled and rnf_mouth_disabled and rnf_source_live):
        raise AssertionError(
            "row24 waiver preconditions changed: resolved ln_rnf, "
            "ln_rnf_mouth, or active zdfphy branch")
    planted_output = re.sub(
        r"ln_rnf\s*=\s*F", "ln_rnf = T", ocean_output, count=1)
    row24_control_fired = not row24_inactive(
        planted_output, namelist_ref, zdfphy_text)
    if not row24_control_fired:
        raise AssertionError("row24 enabled-runoff control did not fail waiver")
    shared_k = np.asarray(pair_calls[0][2])

    wet_k = np.argwhere(np.isfinite(shared_k) & (shared_k != 0.0))
    if wet_k.size == 0:
        raise AssertionError("row27 separated-salinity-K control found no finite nonzero K")
    control_index = tuple(int(x) for x in wet_k[0])

    # Red production arm: enable DDM on the exact captured implicit-call
    # operands, inject a +1-ULP salt-only delta, and require dispatch to switch
    # from the pair solver to two scalar tracer solves whose K operands differ
    # at exactly that point.  This re-enters the production selector; it is not
    # a post-hoc object-identity predicate.
    ddm_enabled = vmcfg.ddm._replace(enabled=True)
    vmcfg_ddm = vmcfg._replace(ddm=ddm_enabled)
    physics_ddm = mc.physics._replace(vertical_mixing=vmcfg_ddm)
    mc_ddm = mc._replace(physics=physics_ddm)
    zero_ddm = jnp.zeros_like(pair_calls[0][2])
    salt_ddm = zero_ddm.at[control_index].set(
        jnp.nextafter(zero_ddm[control_index], jnp.asarray(jnp.inf, zero_ddm.dtype))
    )
    # A +1 ULP from zero can be flushed by later arithmetic; plant one ULP of
    # the nonzero baseline K instead, while preserving a single-point delta.
    salt_ddm = salt_ddm.at[control_index].set(
        jnp.nextafter(
            pair_calls[0][2][control_index],
            jnp.asarray(jnp.inf, pair_calls[0][2].dtype),
        )
        - pair_calls[0][2][control_index]
    )

    def fake_ddm_profile(*ddm_args, **ddm_kwargs):
        del ddm_args, ddm_kwargs
        return zero_ddm, salt_ddm

    control_pair_calls = []
    control_scalar_calls = []
    real_ddm_profile = kprofiles_mod.ddm_K_profile
    real_scalar = vmix_mod.implicit_vertical_diffusion_ocean

    def control_pair(*control_args, **control_kwargs):
        control_pair_calls.append((control_args, control_kwargs))
        return real_pair(*control_args, **control_kwargs)

    def control_scalar(field, k_field, *scalar_args, **scalar_kwargs):
        control_scalar_calls.append((field, k_field))
        return real_scalar(field, k_field, *scalar_args, **scalar_kwargs)

    control_args, control_kwargs = apply_calls[0]
    control_kwargs = dict(control_kwargs)
    control_kwargs["config"] = mc_ddm
    kprofiles_mod.ddm_K_profile = fake_ddm_profile
    vmix_mod.implicit_vertical_diffusion_ocean_pair = control_pair
    vmix_mod.implicit_vertical_diffusion_ocean = control_scalar
    try:
        with jax.disable_jit():
            real_apply(model, *control_args, **control_kwargs)
    finally:
        kprofiles_mod.ddm_K_profile = real_ddm_profile
        vmix_mod.implicit_vertical_diffusion_ocean_pair = real_pair
        vmix_mod.implicit_vertical_diffusion_ocean = real_scalar
    if control_pair_calls:
        raise AssertionError("DDM control incorrectly retained the production T/S pair solve")
    if len(control_scalar_calls) < 2:
        raise AssertionError("DDM control did not dispatch separate T/S scalar solves")
    heat_k = np.asarray(control_scalar_calls[0][1])
    salt_k = np.asarray(control_scalar_calls[1][1])
    k_unequal = ~(heat_k == salt_k)
    row27_control_fired = (
        int(k_unequal.sum()) == 1 and bool(k_unequal[control_index])
    )
    if not row27_control_fired:
        raise AssertionError(
            "row27 production DDM control did not isolate one salt-K coefficient"
        )
    # Row 29's interior LBC no-op preview is numerically the same production
    # composed-avm comparison as row 26.  It cannot be promoted across row 28.
    row29 = row26

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

    evd_mask_plant = lego_evd_mask.copy()
    evd_control_index = tuple(int(x) for x in np.argwhere(wet_interior)[0])
    evd_mask_plant[evd_control_index] = ~evd_mask_plant[evd_control_index]
    evd_mask_control_fired = bool(
        np.any(wet_interior & (evd_mask_plant != nemo_evd_mask)))

    numeric_controls = {
        "row20_lk": sweep.planted_controls(zmxlm, zmxlm, wet_interior, ACCUMULATING),
        "row21_actual_card_avm": sweep.planted_controls(
            avm_closure, avm_closure, wet_interior, POINTWISE),
        "row22_avt": sweep.planted_controls(avt_closure, avt_closure, wet_interior, POINTWISE),
        "row23_avm": sweep.planted_controls(
            avm_closure, avm_closure, wet_interior, POINTWISE),
        "row24_enabled_runoff_waiver_fired": row24_control_fired,
        "row25_avt": sweep.planted_controls(
            avt_composed_lego, avt_composed_lego, wet_interior, POINTWISE),
        "row25_26_mask_flip_fired": evd_mask_control_fired,
        "row30_prd": sweep.planted_controls(
            prd_n[..., :nk_prd], prd_n[..., :nk_prd], wet_prd, POINTWISE
        ),
    }

    rows = {
        "19": {
            "operation": "raw buoyancy mixing length before scans",
            "disposition": "VERIFIED",
            "metric": row19_metric,
            "exact_unequal_wet_elements": 0,
            "deterministic_bracket": {
                "on_streams": row19_bracket["determinism_stream_count"],
                "shared_streams": row19_bracket["shared_stream_count"],
                "strict_byte_identical_shared_streams": row19_bracket[
                    "strict_byte_identical_stream_count"
                ],
                "uninitialized_memory_exclusions": [],
            },
            "receipt": str(row19_path),
            "receipt_sha256": sha256(row19_path),
        },
        "20": {
            "operation": "nn_mxl=3 limiting scans (row19+20 composite)",
            "disposition": (
                "VERIFIED"
                if row20_lk["pass"] and row20_leps["pass"] else "DIVERGED"
            ),
            "zmxlm": row20_lk,
            "zmxld": row20_leps,
            "qualification": (
                "production composite using exact NEMO en/rn2; row19's raw "
                "operand and deterministic writer bracket are independently VERIFIED"
            ),
        },
        "21": {
            "operation": "base avm/avt/dissl assembly",
            "disposition": "VERIFIED" if row21_avm["pass"] else "DIVERGED",
            "receipt": str(row21_path),
            "receipt_sha256": sha256(row21_path),
            "operands": row21_rows,
            "production_actual_card_avm": row21_avm,
            "write_only_bracket": {
                "on_streams": row21_bracket["on_stream_count"],
                "off_streams": row21_bracket["off_stream_count"],
                "strict_byte_identical_shared_streams": row21_bracket[
                    "strict_byte_identical_shared_stream_count"
                ],
                "exclusions": [],
            },
        },
        "22": {
            "operation": "inverse-Prandtl avt correction",
            "disposition": "VERIFIED" if row22_avt["pass"] else "DIVERGED",
            "avt": row22_avt,
        },
        "23": {
            "operation": "closure coefficient copy on EVD-stable points",
            "disposition": (
                "VERIFIED" if row23_avt["pass"] and row23_avm["pass"]
                else "DIVERGED"
            ),
            "production_closure_avt": row23_avt,
            "production_closure_avm": row23_avm,
            "reason": (
                "the production _vmix_K_profiles result is captured before "
                "the production EVD composer and compared to NEMO closure dumps"
            ),
        },
        "24": {
            "operation": "river-mouth enhancement",
            "disposition": "WAIVED",
            "reason": (
                "fail-closed parse confirms resolved ln_rnf=F, namelist_ref "
                "ln_rnf_mouth=.false., and live zdfphy.F90:317-321 branch"
            ),
        },
        "25": {
            "operation": "EVD tracer overwrite",
            "disposition": (
                "VERIFIED" if row25["pass"] and row25_26_mask["pass"]
                else "DIVERGED"
            ),
            "production_composed_avt": row25,
            "branch_mask": row25_26_mask,
            "fired_wet_elements": row25_26_mask["n_fired_wet_elements"],
        },
        "26": {
            "operation": "EVD momentum overwrite",
            "disposition": (
                "VERIFIED" if row26["pass"] and row25_26_mask["pass"]
                else "DIVERGED"
            ),
            "production_composed_avm": row26,
            "branch_mask": row25_26_mask,
            "fired_wet_elements": row25_26_mask["n_fired_wet_elements"],
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
                    "production_pair_calls": len(control_pair_calls),
                    "production_scalar_calls": len(control_scalar_calls),
                    "K_unequal_elements": int(k_unequal.sum()),
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
            "disposition": "UNMEASURED-BLOCKED-BY-ROW28",
            "targeting_preview_avm_interior": row29,
            "controls": row29_controls,
            "reason": (
                "the registered census excludes halos, so NEMO's lateral "
                "boundary update is an interior no-op; the production-composed "
                "avm interior already matches and ordered promotion waits for row 28"
            ),
        },
        "30": {
            "operation": "ldf_slp",
            "disposition": "UNMEASURED-BLOCKED-BY-ROW28",
            "targeting_preview_disposition": (
                "VERIFIED" if all(x["pass"] for x in row30_fields.values())
                else "DIVERGED"),
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
                    "is the next targeted production option only after rows 21--29 close"
                ),
            },
        },
        "31": {
            "operation": "momentum implicit solve application",
            "disposition": "UNMEASURED-BLOCKED-BY-ROW28",
            "reason": (
                "no new NEMO dump is required: restart Kbb + stage-6 Krhs + "
                "stage-7 post-dyn_spg barotropic state + stage-8 output bracket "
                "dyn_zdf. The registered volume-form substitution is ready, "
                "but ordered promotion cannot cross row 28"
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
            "disposition": "UNMEASURED-BLOCKED-BY-ROW28",
            "reason": (
                "no new NEMO dump is required: restart Kbb + stage-23 post-tra_ldf "
                "Krhs and stage-21 post-tra_zdf bracket the application. The "
                "registered z-star volume-form substitution is ready, but ordered "
                "promotion cannot cross row 28"
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

    # Enforce the campaign's ordered stop in the receipt itself. Later
    # calculations are useful targeting previews, but a failed promoted row
    # must never leave downstream VERIFIED/WAIVED labels or a successful exit.
    first_diverged = _enforce_ordered_stop(rows)

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
        "status": (
            "ROWS-19-23,25-27-VERIFIED; ROW-24-WAIVED; "
            "ordered frontier is row 28 turbocline diagnostic"
            if first_diverged is None
            else f"DIVERGED at row {first_diverged}; later rows ordered-blocked"
        ),
        "first_divergence": first_diverged,
        "repo_sha": git_sha(),
        "probe": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "lane": dump_lane.banner(),
        "run_dir": str(run),
        "focus_columns_ji": [list(x) for x in focus],
        "wet_columns": 9920,
        "bars": {"pointwise": POINTWISE, "accumulating": ACCUMULATING},
        "row19_receipt": {
            "path": str(row19_path),
            "sha256": sha256(row19_path),
            "repo_sha": row19_receipt["repo_sha"],
        },
        "row21_receipt": {
            "path": str(row21_path),
            "sha256": sha256(row21_path),
            "repo_sha": row21_receipt["repo_sha"],
        },
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
    return 0 if first_diverged is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
