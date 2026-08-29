#!/usr/bin/env python
"""Score the preregistered day-180 ZDF row-19 raw mixing-length slot.

The receipt is fail-closed on the repaired write-only bracket: byte-identical
restarts, byte-identical shared streams, byte-identical repeated executions,
and exactly one new stream are required before a numerical result can be called
VERIFIED. No uninitialized-memory exception is permitted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import bn2_alpha_compare as loaders
import jax
import jax.numpy as jnp
import netCDF4  # noqa: N813
import numpy as np
import zdf_chain_sweep as sweep
from legoesm.ocean.fidelity.time_levels import time_level_for_dump
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import _tke_raw_mixing_length
from zdf_stream_bracket import (
    files_byte_identical,
    manifest_sha256,
    one_bit_file_control,
    sha256,
    stream_manifest,
)

BAR = 1.0e-15
RAW_NAME = "tke_dump_zmxlm_raw.bin"
EXPECTED_SIZE = 2_980_224
EXPECTED_ON_STREAMS = 198
EXPECTED_OFF_STREAMS = 197


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def tracked_tree_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()


def absolute_metrics(
    actual: np.ndarray,
    expected: np.ndarray,
    wet: np.ndarray,
    focus: list[tuple[int, int]],
    bar: float,
) -> dict:
    """Preregistered absolute-metre per-column census for row 19."""
    registered = np.asarray(wet, dtype=bool)
    nonfinite = registered & (~np.isfinite(actual) | ~np.isfinite(expected))
    finite = registered & ~nonfinite
    column = np.where(
        np.any(registered, axis=-1),
        np.max(np.where(finite, np.abs(actual - expected), 0.0), axis=-1),
        np.nan,
    )
    wet_columns = np.any(registered, axis=-1)
    bad = wet_columns & (
        (column > bar) | np.any(nonfinite, axis=-1))
    focus_rows = [
        {
            "j": j,
            "i": i,
            "wet": bool(wet_columns[j, i]),
            "column_error_m": (
                float(column[j, i]) if wet_columns[j, i] else None),
            "pass": bool(wet_columns[j, i] and not bad[j, i]),
        }
        for j, i in focus
    ]
    return {
        "units": "m",
        "absolute_column_bar_m": bar,
        "n_wet_elements": int(registered.sum()),
        "n_nonfinite_wet_elements": int(nonfinite.sum()),
        "n_wet_columns": int(wet_columns.sum()),
        "n_diverged_columns": int(bad.sum()),
        "n_verified_columns": int(wet_columns.sum() - bad.sum()),
        "max_column_error_m": float(np.nanmax(column)),
        "worst_column_ji": [
            int(x) for x in np.unravel_index(
                np.nanargmax(column), column.shape)],
        "focus": focus_rows,
        "pass": bool(not bad.any()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bracket-dir", type=Path, required=True)
    parser.add_argument("--determinism-a-dir", type=Path, required=True)
    parser.add_argument("--determinism-b-dir", type=Path, required=True)
    parser.add_argument("--determinism-a-binary-receipt", type=Path, required=True)
    parser.add_argument("--determinism-b-binary-receipt", type=Path, required=True)
    parser.add_argument("--run-binary-receipt", type=Path, required=True)
    parser.add_argument("--bracket-binary-receipt", type=Path, required=True)
    parser.add_argument("--on-build-binary-receipt", type=Path, required=True)
    parser.add_argument("--off-build-binary-receipt", type=Path, required=True)
    parser.add_argument("--mld-maps", type=Path, required=True)
    parser.add_argument("--nemo-source", type=Path, required=True)
    parser.add_argument("--bracket-nemo-source", type=Path, required=True)
    parser.add_argument("--nemo-binary", type=Path, required=True)
    parser.add_argument("--bracket-nemo-binary", type=Path, required=True)
    parser.add_argument("--writer-patch", type=Path, required=True)
    parser.add_argument("--remove-patch", type=Path, required=True)
    parser.add_argument("--on-source-manifest", type=Path, required=True)
    parser.add_argument("--off-source-manifest", type=Path, required=True)
    parser.add_argument("--donor-restart", type=Path, required=True)
    parser.add_argument("--expected-raw-sha", required=True)
    parser.add_argument("--expected-restart-sha", required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--expected-bracket-source-sha", required=True)
    parser.add_argument("--expected-binary-sha", required=True)
    parser.add_argument("--expected-bracket-binary-sha", required=True)
    parser.add_argument("--expected-writer-patch-sha", required=True)
    parser.add_argument("--expected-remove-patch-sha", required=True)
    parser.add_argument("--expected-on-source-manifest-sha", required=True)
    parser.add_argument("--expected-off-source-manifest-sha", required=True)
    parser.add_argument("--expected-donor-sha", required=True)
    parser.add_argument("--expected-repo-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row-19 receipt requires CPU fp64")
    run = args.run_dir.resolve()
    bracket = args.bracket_dir.resolve()
    det_a = args.determinism_a_dir.resolve()
    det_b = args.determinism_b_dir.resolve()
    if run != det_a:
        raise SystemExit("scored row-19 run must be determinism arm A")
    if git_sha() != args.expected_repo_sha:
        raise SystemExit("repository HEAD does not match the registered receipt")
    if not tracked_tree_clean():
        raise SystemExit("row-19 receipt refuses a dirty tracked worktree")
    raw_path = run / RAW_NAME
    if time_level_for_dump(RAW_NAME) != "now":
        raise SystemExit("raw MXL time-level registry changed")
    if raw_path.stat().st_size != EXPECTED_SIZE:
        raise SystemExit(f"raw dump size changed: {raw_path.stat().st_size}")
    if sha256(raw_path) != args.expected_raw_sha:
        raise SystemExit("raw dump SHA does not match the human-bound receipt")
    if sha256(args.nemo_source) != args.expected_source_sha:
        raise SystemExit("active NEMO source SHA does not match the receipt")
    if sha256(args.bracket_nemo_source) != args.expected_bracket_source_sha:
        raise SystemExit("bracket NEMO source SHA does not match the receipt")
    if sha256(args.nemo_binary) != args.expected_binary_sha:
        raise SystemExit("NEMO executable SHA does not match the receipt")
    if sha256(args.bracket_nemo_binary) != args.expected_bracket_binary_sha:
        raise SystemExit("bracket NEMO executable SHA does not match the receipt")
    if sha256(args.writer_patch) != args.expected_writer_patch_sha:
        raise SystemExit("deterministic-writer patch SHA does not match the receipt")
    if sha256(args.remove_patch) != args.expected_remove_patch_sha:
        raise SystemExit("row-19 removal patch SHA does not match the receipt")
    if sha256(args.on_source_manifest) != args.expected_on_source_manifest_sha:
        raise SystemExit("ON source-manifest SHA does not match the receipt")
    if sha256(args.off_source_manifest) != args.expected_off_source_manifest_sha:
        raise SystemExit("OFF source-manifest SHA does not match the receipt")
    if sha256(args.donor_restart) != args.expected_donor_sha:
        raise SystemExit("donor restart SHA does not match the receipt")
    if args.nemo_binary.stat().st_mtime_ns < args.nemo_source.stat().st_mtime_ns:
        raise SystemExit("NEMO executable is older than the instrumented source")
    if args.bracket_nemo_binary.stat().st_mtime_ns < args.bracket_nemo_source.stat().st_mtime_ns:
        raise SystemExit("bracket NEMO executable is older than the OFF source")
    for receipt in (
        args.run_binary_receipt,
        args.determinism_a_binary_receipt,
        args.determinism_b_binary_receipt,
    ):
        fields = receipt.read_text(errors="strict").split()
        if not fields or fields[0] != args.expected_binary_sha:
            raise SystemExit(
                f"determinism run binary receipt does not bind current binary: {receipt}")
    bracket_fields = args.bracket_binary_receipt.read_text(errors="strict").split()
    if (not bracket_fields
            or bracket_fields[0] != args.expected_bracket_binary_sha):
        raise SystemExit("bracket run binary receipt does not bind the OFF binary")
    on_build_fields = args.on_build_binary_receipt.read_text(errors="strict").split()
    off_build_fields = args.off_build_binary_receipt.read_text(errors="strict").split()
    if not on_build_fields or on_build_fields[0] != args.expected_binary_sha:
        raise SystemExit("ON build receipt does not bind the executed binary")
    if not off_build_fields or off_build_fields[0] != args.expected_bracket_binary_sha:
        raise SystemExit("OFF build receipt does not bind the executed binary")
    donor_name = "DINO_00005760_restart.nc"
    for control_dir in (bracket, det_a, det_b):
        if sha256(control_dir / donor_name) != args.expected_donor_sha:
            raise SystemExit(
                f"control donor restart does not match the receipt: {control_dir}")
    for control_dir in (det_a, det_b):
        if sha256(control_dir / "nemo") != args.expected_binary_sha:
            raise SystemExit(
                f"determinism run executable does not bind current binary: {control_dir}")
    if sha256(run / "nemo") != args.expected_binary_sha:
        raise SystemExit("row-19 run executable does not bind current binary")
    if sha256(bracket / "nemo") != args.expected_bracket_binary_sha:
        raise SystemExit("bracket run executable does not bind the OFF binary")

    restart_name = "DINO_00005761_restart.nc"
    restart_on = run / restart_name
    restart_off = bracket / restart_name
    restart_sha = sha256(restart_on)
    if restart_sha != args.expected_restart_sha or sha256(restart_off) != restart_sha:
        raise SystemExit("write-only bracket restart identity failed")
    if (sha256(det_a / restart_name) != restart_sha
            or sha256(det_b / restart_name) != restart_sha):
        raise SystemExit("identical-binary determinism-control restart identity failed")

    jpi, jpj, jpk, hls = loaders._read_dims(str(run))

    new_manifest = stream_manifest(run)
    old_manifest = stream_manifest(bracket)
    det_b_manifest = stream_manifest(det_b)
    new_bins = set(new_manifest)
    old_bins = set(old_manifest)

    def registered_inventory(on_names: set[str], off_names: set[str]) -> bool:
        return (
            len(on_names) == EXPECTED_ON_STREAMS
            and len(off_names) == EXPECTED_OFF_STREAMS
            and on_names - off_names == {RAW_NAME}
            and not off_names - on_names
        )

    if not registered_inventory(new_bins, old_bins):
        raise SystemExit(
            "row-19 stream inventory changed: "
            f"ON={len(new_bins)}, OFF={len(old_bins)}, "
            f"new={new_bins-old_bins}, missing={old_bins-new_bins}")
    shared = sorted(old_bins)
    unequal_bracket = [
        name for name in shared
        if not files_byte_identical(run / name, bracket / name)]
    if unequal_bracket:
        raise SystemExit(
            f"repaired write-only shared-stream bracket failed: {unequal_bracket}")
    for label, control_dir in (("A", det_a), ("B", det_b)):
        control_bins = {p.name for p in control_dir.glob("*.bin")}
        if len(control_bins) != EXPECTED_ON_STREAMS or control_bins != new_bins:
            raise SystemExit(
                f"determinism-{label} stream inventory changed: "
                f"new={control_bins-new_bins}, missing={new_bins-control_bins}")
    unequal_determinism = [
        name for name in sorted(new_bins)
        if not files_byte_identical(det_a / name, det_b / name)]
    if unequal_determinism:
        raise SystemExit(
            f"repaired identical-binary stream determinism failed: "
            f"{unequal_determinism}")

    # Red controls use the same predicates as the production bracket.
    control_name = shared[0]
    bracket_control_fired = one_bit_file_control(run / control_name)
    missing_control_fired = not registered_inventory(
        new_bins - {control_name}, old_bins)
    if not (bracket_control_fired and missing_control_fired):
        raise SystemExit(
            "exact bracket controls did not fire: "
            f"one_bit={bracket_control_fired}, missing={missing_control_fired}")

    source = args.nemo_source.read_text(errors="strict")
    for literal in (
        "zrn2 = MAX( rn2(ji,jj,jk), rsmall )",
        "SQRT( 2._wp * en(ji,jj,jk) / zrn2 )",
    ):
        if literal not in source:
            raise SystemExit(f"active NEMO source expression changed: {literal}")

    focus = sweep.focus_from_maps(args.mld_maps)
    ni, nj = jpi - 2 * hls, jpj - 2 * hls

    def interior(name: str) -> np.ndarray:
        time_level_for_dump(name)
        return loaders._load_interior(str(run / name), ni, nj)

    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        tmask = np.moveaxis(np.asarray(ds["tmask"][0]), 0, -1) > 0.5
    wmask = np.zeros_like(tmask)
    wmask[..., 0] = tmask[..., 0]
    wmask[..., 1:] = tmask[..., 1:] & tmask[..., :-1]
    wet = wmask[..., 1:jpk]
    if int(np.any(wet, axis=-1).sum()) != 9920:
        raise SystemExit("wet-column census changed")

    en = interior("tke_dump_en.bin")[..., 1:jpk]
    rn2 = interior("tke_dump_rn2.bin")[..., 1:jpk]
    nemo = interior(RAW_NAME)[..., 1:jpk]
    rsmall = np.float64(0.5) * np.finfo(np.float64).eps
    source_formula = np.maximum(
        np.float64(0.01),
        np.sqrt((np.float64(2.0) * en) / np.maximum(rn2, rsmall)),
    )
    cfg = TKEConfig(
        tke_mxl_raw_evaluation="nemo_literal", mxl_min=0.01)
    lego = np.asarray(jax.jit(
        lambda e, n2: _tke_raw_mixing_length(e, n2, cfg)
    )(jnp.asarray(en), jnp.asarray(rn2)))
    source_formula_unequal = int(np.count_nonzero(
        wet & ~(source_formula == nemo)))
    metric = absolute_metrics(lego, nemo, wet, focus, BAR)
    exact_unequal = int(np.count_nonzero(wet & ~(lego == nemo)))

    exact_plant = nemo.copy()
    idx = tuple(int(x) for x in np.argwhere(wet)[0])
    exact_plant[idx] = np.nextafter(exact_plant[idx], np.float64(np.inf))
    ulp_control = int(np.count_nonzero(wet & ~(exact_plant == nemo))) == 1
    roll_control = not absolute_metrics(
        np.roll(nemo, 1, axis=1), nemo, wet, focus, BAR)["pass"]
    value_plant = nemo.copy()
    # Use the preregistered 2e-12 violation, while also ensuring it remains
    # representable if a future state carries an unexpectedly large length.
    value_plant[idx] += max(
        np.float64(2.0e-12),
        np.float64(4.0) * np.spacing(value_plant[idx]),
    )
    value_control = not absolute_metrics(
        value_plant, nemo, wet, focus, BAR)["pass"]
    if not (ulp_control and roll_control and value_control):
        raise SystemExit(
            "one or more planted row-19 controls failed to fire: "
            f"ulp={ulp_control}, roll={roll_control}, value={value_control}")

    artifact = {
        "schema": "dino-zdf-row19-raw-mxl-v1",
        "repo_sha": git_sha(),
        "disposition": "VERIFIED" if metric["pass"] else "DIVERGED",
        "bar_absolute_m": BAR,
        "production_metric": metric,
        "exact_unequal_wet_elements": exact_unequal,
        "source_formula_unequal_wet_elements": source_formula_unequal,
        "focus_columns_ji": [[j, i] for j, i in focus],
        "controls": {
            "plus_one_ulp_exact_census_fired": ulp_control,
            "one_i_roll_fired": roll_control,
            "value_plant_fired": value_control,
            "one_bit_shared_stream_bracket_plant_fired": bracket_control_fired,
            "missing_stream_inventory_plant_fired": missing_control_fired,
        },
        "bracket": {
            "restart_sha256": restart_sha,
            "shared_stream_count": len(shared),
            "strict_byte_identical_stream_count": len(shared),
            "new_stream": RAW_NAME,
            "determinism_stream_count": len(new_bins),
            "on_stream_manifest_sha256": manifest_sha256(new_manifest),
            "determinism_b_stream_manifest_sha256": manifest_sha256(
                det_b_manifest
            ),
            "off_stream_manifest_sha256": manifest_sha256(old_manifest),
            "on_stream_manifest": new_manifest,
            "off_stream_manifest": old_manifest,
            "uninitialized_memory_exclusions": [],
            "mechanism": (
                "all instrumentation capture buffers are initialized before "
                "interior fill; every shared stream and repeated-run stream "
                "is required to be byte-identical without exclusions"
            ),
        },
        "nemo_line": "cfgs/DINO/MY_SRC/zdftke.F90:840-841",
        "provenance_sha256": {
            str(raw_path): sha256(raw_path),
            str(restart_on): restart_sha,
            str(args.mld_maps.resolve()): sha256(args.mld_maps.resolve()),
            str(args.nemo_source.resolve()): sha256(args.nemo_source.resolve()),
            str(args.bracket_nemo_source.resolve()): sha256(
                args.bracket_nemo_source.resolve()),
            str(args.nemo_binary.resolve()): sha256(args.nemo_binary.resolve()),
            str(args.bracket_nemo_binary.resolve()): sha256(
                args.bracket_nemo_binary.resolve()),
            str(args.writer_patch.resolve()): sha256(args.writer_patch.resolve()),
            str(args.remove_patch.resolve()): sha256(args.remove_patch.resolve()),
            str(args.on_source_manifest.resolve()): sha256(
                args.on_source_manifest.resolve()),
            str(args.off_source_manifest.resolve()): sha256(
                args.off_source_manifest.resolve()),
            str(args.donor_restart.resolve()): sha256(args.donor_restart.resolve()),
            str(args.determinism_a_binary_receipt.resolve()): sha256(
                args.determinism_a_binary_receipt.resolve()),
            str(args.determinism_b_binary_receipt.resolve()): sha256(
                args.determinism_b_binary_receipt.resolve()),
            str(args.run_binary_receipt.resolve()): sha256(
                args.run_binary_receipt.resolve()),
            str(args.bracket_binary_receipt.resolve()): sha256(
                args.bracket_binary_receipt.resolve()),
            str(args.on_build_binary_receipt.resolve()): sha256(
                args.on_build_binary_receipt.resolve()
            ),
            str(args.off_build_binary_receipt.resolve()): sha256(
                args.off_build_binary_receipt.resolve()
            ),
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str((Path(__file__).parent / "zdf_stream_bracket.py").resolve()): sha256(
                (Path(__file__).parent / "zdf_stream_bracket.py").resolve()
            ),
            str(restart_off): sha256(restart_off),
            str(det_a / restart_name): sha256(det_a / restart_name),
            str(det_b / restart_name): sha256(det_b / restart_name),
            str(det_a / "run.attempt1.log"): sha256(det_a / "run.attempt1.log"),
            str(det_b / "run.attempt1.log"): sha256(det_b / "run.attempt1.log"),
            str(bracket / "run.attempt1.log"): sha256(
                bracket / "run.attempt1.log"),
        },
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        f"row19 {artifact['disposition']}: {metric['n_diverged_columns']}/"
        f"{metric['n_wet_columns']} columns fail absolute {BAR:.1e} m; "
        f"exact unequal={exact_unequal}; "
        f"focus_fail={sum(not x['pass'] for x in metric['focus'])}")
    return 0 if metric["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
