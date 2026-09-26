#!/usr/bin/env python
"""Score ZDF row 28's literal turbocline scan and depth lookup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import bn2_alpha_compare as loaders
import jax
import jax.numpy as jnp
import legoesm.ocean.physics.vertical_mixing as vmix_mod
import netCDF4  # noqa: N813
import numpy as np
import zdf_chain_sweep as sweep
from kamm_twin_90d import _build_twin_state
from legoesm.core.precision import PrecisionPolicy, set_policy
from zdf_stream_bracket import (
    files_byte_identical,
    manifest_sha256,
    one_bit_file_control,
    stream_manifest,
)

BAR = 1.0e-15
DUMP = "zdf_dump_hmld_turb.bin"
EXPECTED_SIZE = 90_944
EXPECTED_ON_STREAMS = 203
EXPECTED_OFF_STREAMS = 202
RESTART_SHA256 = "33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c"
TAIL_RECEIPT_SHA256 = "2f669e0b85b5902816bff57109555cc65fba4479f62e80c19fb82d48b208a91b"


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


def absolute_depth_metrics(
    lego: np.ndarray,
    nemo: np.ndarray,
    wet: np.ndarray,
    focus: list[tuple[int, int]],
) -> dict:
    nonfinite = wet & (~np.isfinite(lego) | ~np.isfinite(nemo))
    error = np.where(wet & ~nonfinite, np.abs(lego - nemo), 0.0)
    bad = wet & ((error > BAR) | nonfinite)
    focus_rows = [
        {
            "j": j,
            "i": i,
            "wet": bool(wet[j, i]),
            "column_error_m": float(error[j, i]) if wet[j, i] else None,
            "pass": bool(wet[j, i] and not bad[j, i]),
        }
        for j, i in focus
    ]
    return {
        "absolute_column_bar_m": BAR,
        "n_wet_columns": int(wet.sum()),
        "n_diverged_columns": int(bad.sum()),
        "n_verified_columns": int(wet.sum() - bad.sum()),
        "n_nonfinite_wet_columns": int(nonfinite.sum()),
        "max_column_error_m": float(error.max()),
        "exact_unequal_wet_columns": int(np.count_nonzero(wet & ~(lego == nemo))),
        "focus": focus_rows,
        "pass": bool(not bad.any()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--bracket-dir", type=Path, required=True)
    ap.add_argument("--tail-artifact", type=Path, required=True)
    ap.add_argument("--mld-maps", type=Path, required=True)
    ap.add_argument("--nemo-source", type=Path, required=True)
    ap.add_argument("--bracket-nemo-source", type=Path, required=True)
    ap.add_argument("--oracle-zdfmxl-source", type=Path, required=True)
    ap.add_argument("--nemo-binary", type=Path, required=True)
    ap.add_argument("--bracket-nemo-binary", type=Path, required=True)
    ap.add_argument("--build-binary-receipt", type=Path, required=True)
    ap.add_argument("--on-source-manifest", type=Path, required=True)
    ap.add_argument("--off-source-manifest", type=Path, required=True)
    ap.add_argument("--instrument-patch", type=Path, required=True)
    ap.add_argument("--expected-repo-sha", required=True)
    ap.add_argument("--expected-dump-sha", required=True)
    ap.add_argument("--expected-source-sha", required=True)
    ap.add_argument("--expected-bracket-source-sha", required=True)
    ap.add_argument("--expected-oracle-source-sha", required=True)
    ap.add_argument("--expected-binary-sha", required=True)
    ap.add_argument("--expected-bracket-binary-sha", required=True)
    ap.add_argument("--expected-build-receipt-sha", required=True)
    ap.add_argument("--expected-on-manifest-sha", required=True)
    ap.add_argument("--expected-off-manifest-sha", required=True)
    ap.add_argument("--expected-patch-sha", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row-28 receipt requires CPU fp64")
    if git_sha() != args.expected_repo_sha or not tracked_tree_clean():
        raise SystemExit("row-28 receipt requires the registered clean repository HEAD")

    tail = json.loads(args.tail_artifact.read_text())
    if (
        sha256(args.tail_artifact) != TAIL_RECEIPT_SHA256
        or any(tail["rows"][str(row)]["disposition"] != "VERIFIED" for row in range(19, 24))
        or tail["rows"]["24"]["disposition"] != "WAIVED"
        or any(tail["rows"][str(row)]["disposition"] != "VERIFIED" for row in range(25, 28))
    ):
        raise SystemExit("ordered predecessor receipt through row 27 is not promotable")

    run, bracket = args.run_dir.resolve(), args.bracket_dir.resolve()
    expected_paths = {
        args.nemo_source: args.expected_source_sha,
        args.bracket_nemo_source: args.expected_bracket_source_sha,
        args.oracle_zdfmxl_source: args.expected_oracle_source_sha,
        args.nemo_binary: args.expected_binary_sha,
        args.bracket_nemo_binary: args.expected_bracket_binary_sha,
        args.build_binary_receipt: args.expected_build_receipt_sha,
        args.on_source_manifest: args.expected_on_manifest_sha,
        args.off_source_manifest: args.expected_off_manifest_sha,
        args.instrument_patch: args.expected_patch_sha,
    }
    for path, expected in expected_paths.items():
        if sha256(path.resolve()) != expected:
            raise SystemExit(f"SHA receipt changed: {path}")
    if sha256(run / DUMP) != args.expected_dump_sha:
        raise SystemExit("row-28 direct dump SHA changed")
    if (run / DUMP).stat().st_size != EXPECTED_SIZE:
        raise SystemExit("row-28 direct dump size changed")
    if args.nemo_binary.stat().st_mtime_ns < args.nemo_source.stat().st_mtime_ns:
        raise SystemExit("row-28 binary is older than its patched source")

    def receipt_binds(path: Path, expected: str) -> bool:
        fields = path.read_text(errors="strict").split()
        return bool(fields and fields[0] == expected)

    for path, expected in (
        (args.build_binary_receipt, args.expected_binary_sha),
        (run / ".nemo_binary_sha256", args.expected_binary_sha),
        (bracket / ".nemo_binary_sha256", args.expected_bracket_binary_sha),
    ):
        if not receipt_binds(path, expected):
            raise SystemExit(f"binary receipt does not bind the executed binary: {path}")
    if sha256(run / "nemo") != args.expected_binary_sha:
        raise SystemExit("ON executable changed")
    if sha256(bracket / "nemo") != args.expected_bracket_binary_sha:
        raise SystemExit("OFF executable changed")
    restart = "DINO_00005761_restart.nc"
    if sha256(run / restart) != RESTART_SHA256 or sha256(bracket / restart) != RESTART_SHA256:
        raise SystemExit("row-28 behavior-neutral restart bracket failed")
    for directory in (run, bracket):
        if not (directory / "run.attempt1.log").read_text().rstrip().endswith("STOP 0"):
            raise SystemExit(f"NEMO run did not stop cleanly: {directory}")

    on_manifest, off_manifest = stream_manifest(run), stream_manifest(bracket)
    on_names, off_names = set(on_manifest), set(off_manifest)

    def exact_inventory(on: set[str], off: set[str]) -> bool:
        return (
            len(on) == EXPECTED_ON_STREAMS
            and len(off) == EXPECTED_OFF_STREAMS
            and on - off == {DUMP}
            and not off - on
        )

    if not exact_inventory(on_names, off_names):
        raise SystemExit("row-28 stream inventory changed")
    unequal = [
        name for name in sorted(off_names)
        if not files_byte_identical(run / name, bracket / name)
    ]
    if unequal:
        raise SystemExit(f"row-28 shared-stream bracket failed: {unequal}")
    shared = sorted(off_names)[0]
    one_bit_fired = one_bit_file_control(run / shared)
    missing_fired = not exact_inventory(on_names - {shared}, off_names)
    if not (one_bit_fired and missing_fired):
        raise SystemExit("row-28 bracket controls did not fire")

    source = args.oracle_zdfmxl_source.read_text(errors="strict")
    for literal in (
        "imld(:,:) = mbkt(T2D(0)) + 1",
        "IF( avt (ji,jj,jk) < avt_c * wmask(ji,jj,jk) )   imld(ji,jj) = jk",
        "hmld (ji,jj) = gdepw(ji,jj,iik  ,Kmm) * ssmask(ji,jj)",
    ):
        if literal not in source:
            raise SystemExit(f"active row-28 NEMO expression changed: {literal}")

    focus = sweep.focus_from_maps(args.mld_maps)
    jpi, jpj, jpk, hls = loaders._read_dims(str(run))
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        tmask = np.moveaxis(np.asarray(ds["tmask"][0]), 0, -1) > 0.5
    wmask = np.zeros_like(tmask)
    wmask[..., 0] = tmask[..., 0]
    wmask[..., 1:] = tmask[..., 1:] & tmask[..., :-1]
    wet_w = wmask[..., 1:jpk]
    wet_columns = tmask[..., 0]
    if int(wet_columns.sum()) != 9920:
        raise SystemExit("row-28 wet-column census changed")
    bottom_level = tmask.sum(axis=-1).astype(np.int32)

    _, _, _, model, _, surface_forcing, state = _build_twin_state(
        "nemo_dino_kamm_mlf", str(run), str(run), bridge_tke=True,
        bridge_before=True, restart_file="DINO_00005760_restart.nc", e3t_mode="both",
    )
    profile_calls = []
    real_profiles = vmix_mod.compute_vertical_K_profiles

    def spy_profiles(*profile_args, **profile_kwargs):
        result = real_profiles(*profile_args, **profile_kwargs)
        profile_calls.append((profile_args, profile_kwargs, result))
        return result

    vmix_mod.compute_vertical_K_profiles = spy_profiles
    try:
        sweep.capture_face_sh2_call(model, state, surface_forcing)
    finally:
        vmix_mod.compute_vertical_K_profiles = real_profiles
    if len(profile_calls) != 1:
        raise SystemExit("production coefficient composer did not fire exactly once")
    kwargs, result = profile_calls[0][1], profile_calls[0][2]
    avt = jnp.asarray(result[0])
    bundle = kwargs.get("tke_n2_bundle")
    if bundle is None:
        raise SystemExit("row-28 production capture lacks frozen step-entry geometry")
    gdepw = jnp.asarray(bundle.gdepw_Kmm)
    if avt.shape != wet_w.shape or gdepw.shape != wet_w.shape:
        raise SystemExit("row-28 production operand shape changed")

    # NEMO nlb10 is 2 on this mesh.  Scan jk=35..2 in literal order; lego k=0
    # maps to NEMO jk=2 and its final k=34 is the dry jpk terminal.
    if not bool(np.all(np.asarray(gdepw)[..., 0][wet_columns] > 10.0)):
        raise SystemExit("DINO nlb10=2 premise changed")
    initial = jnp.asarray(bottom_level + 1)
    wet_j = jnp.asarray(wet_w)

    def body(scan_i, imld):
        k = jpk - 3 - scan_i
        fires = avt[..., k] < jnp.asarray(5.0e-4, avt.dtype) * wet_j[..., k]
        return jnp.where(fires, k + 2, imld)

    imld = jax.jit(lambda init: jax.lax.fori_loop(0, jpk - 2, body, init))(initial)
    gather = (imld - 2)[..., None]
    hmld = jnp.take_along_axis(gdepw, gather, axis=-1)[..., 0]
    hmld = jnp.where(jnp.asarray(wet_columns), hmld, 0.0)
    lego_hmld = np.asarray(hmld)
    lego_imld = np.asarray(imld)

    nemo_hmld = loaders._load_haloed(str(run / DUMP), jpi, jpj, hls)[..., 0]
    gdepw_np = np.asarray(gdepw)
    depth_matches = gdepw_np == nemo_hmld[..., None]
    match_count = depth_matches.sum(axis=-1)
    if np.any(wet_columns & (match_count != 1)):
        raise SystemExit("direct hmld dump does not invert uniquely to imld")
    nemo_imld = np.argmax(depth_matches, axis=-1) + 2
    index_bad = wet_columns & (lego_imld != nemo_imld)
    metric = absolute_depth_metrics(lego_hmld, nemo_hmld, wet_columns, focus)
    metric["n_index_diverged_columns"] = int(index_bad.sum())
    metric["index_exact_pass"] = bool(not index_bad.any())

    idx = tuple(int(x) for x in np.argwhere(wet_columns)[0])
    value_plant = nemo_hmld.copy()
    value_plant[idx] += max(2.0e-15, 4.0 * np.spacing(value_plant[idx]))
    value_fired = not absolute_depth_metrics(
        value_plant, nemo_hmld, wet_columns, focus)["pass"]
    wet_values = np.where(wet_columns, nemo_hmld, np.nan)
    low = tuple(int(x) for x in np.unravel_index(np.nanargmin(wet_values), wet_values.shape))
    high = tuple(int(x) for x in np.unravel_index(np.nanargmax(wet_values), wet_values.shape))
    if abs(nemo_hmld[high] - nemo_hmld[low]) <= BAR:
        raise SystemExit("row-28 depth field lacks a discriminating layout control")
    layout_plant = nemo_hmld.copy()
    layout_plant[low], layout_plant[high] = nemo_hmld[high], nemo_hmld[low]
    layout_fired = not absolute_depth_metrics(
        layout_plant, nemo_hmld, wet_columns, focus)["pass"]
    nan_plant = nemo_hmld.copy()
    nan_plant[idx] = np.nan
    nonfinite_fired = not absolute_depth_metrics(
        nan_plant, nemo_hmld, wet_columns, focus)["pass"]
    index_plant = nemo_imld.copy()
    index_plant[idx] += 1
    index_fired = int(np.count_nonzero(wet_columns & (index_plant != nemo_imld))) == 1
    controls = {
        "value_plant_fired": value_fired,
        "unequal_column_swap_fired": layout_fired,
        "nonfinite_fired": nonfinite_fired,
        "index_plant_fired": index_fired,
        "one_bit_shared_stream_fired": one_bit_fired,
        "missing_stream_inventory_fired": missing_fired,
    }
    if not all(controls.values()):
        raise SystemExit("one or more row-28 controls did not fire")

    disposition = (
        "VERIFIED" if metric["pass"] and metric["index_exact_pass"] else "DIVERGED"
    )
    paths = {
        "dump": run / DUMP,
        "run_input_restart": run / "DINO_00005760_restart.nc",
        "off_input_restart": bracket / "DINO_00005760_restart.nc",
        "run_mesh_mask": run / "mesh_mask.nc",
        "off_mesh_mask": bracket / "mesh_mask.nc",
        "run_ocean_output": run / "ocean.output",
        "off_ocean_output": bracket / "ocean.output",
        "run_restart": run / restart,
        "off_restart": bracket / restart,
        "run_log": run / "run.attempt1.log",
        "off_run_log": bracket / "run.attempt1.log",
        "run_binary_receipt": run / ".nemo_binary_sha256",
        "off_binary_receipt": bracket / ".nemo_binary_sha256",
        "tail_artifact": args.tail_artifact.resolve(),
        "mld_maps": args.mld_maps.resolve(),
        "nemo_source": args.nemo_source.resolve(),
        "bracket_nemo_source": args.bracket_nemo_source.resolve(),
        "oracle_zdfmxl_source": args.oracle_zdfmxl_source.resolve(),
        "nemo_binary": args.nemo_binary.resolve(),
        "bracket_nemo_binary": args.bracket_nemo_binary.resolve(),
        "build_binary_receipt": args.build_binary_receipt.resolve(),
        "on_source_manifest": args.on_source_manifest.resolve(),
        "off_source_manifest": args.off_source_manifest.resolve(),
        "instrument_patch": args.instrument_patch.resolve(),
        "probe": Path(__file__).resolve(),
    }
    artifact = {
        "schema": "dino-zdf-row28-turbocline-v1",
        "repo_sha": git_sha(),
        "disposition": disposition,
        "metric": metric,
        "controls": controls,
        "focus_columns_ji": [list(x) for x in focus],
        "nemo_lines": "cfgs/DINO/WORK/zdfmxl.F90:145-152",
        "bracket": {
            "on_stream_count": len(on_names),
            "off_stream_count": len(off_names),
            "strict_byte_identical_shared_stream_count": len(off_names),
            "new_stream": DUMP,
            "on_stream_manifest_sha256": manifest_sha256(on_manifest),
            "off_stream_manifest_sha256": manifest_sha256(off_manifest),
            "on_stream_manifest": on_manifest,
            "off_stream_manifest": off_manifest,
            "exclusions": [],
        },
        "provenance_paths": {key: str(path) for key, path in paths.items()},
        "provenance_sha256": {key: sha256(path) for key, path in paths.items()},
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        f"row28 {disposition}: {metric['n_diverged_columns']}/9920 depth "
        f"columns fail {BAR:.1e} m; index_fail={metric['n_index_diverged_columns']}; "
        f"focus_fail={sum(not item['pass'] for item in metric['focus'])}"
    )
    return 0 if disposition == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
