#!/usr/bin/env python
"""Score the preregistered day-180 ZDF row-19 raw mixing-length slot.

The receipt is fail-closed on the write-only bracket: a byte-identical restart,
identical initialized shared physics slots, the exact registered 13-stream/four-
halo-slot oracle-writer defect signature, and exactly one new stream are required
before a numerical result can be called VERIFIED.
"""

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

import bn2_alpha_compare as loaders
import jax
import jax.numpy as jnp
import netCDF4  # noqa: N813
import numpy as np
import zdf_chain_sweep as sweep
from legoesm.ocean.fidelity.time_levels import time_level_for_dump
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import _tke_raw_mixing_length

BAR = 1.0e-15
RAW_NAME = "tke_dump_zmxlm_raw.bin"
EXPECTED_SIZE = 2_980_224
UNINITIALIZED_HALO_STREAMS = frozenset({
    "cor2d_dump_zu_trd_substep1.bin",
    "cor2d_dump_zv_trd_substep1.bin",
    "eiv_dump_zaeiw.bin",
    "fct_dump_zwx_up.bin",
    "fct_dump_zwy_up.bin",
    "fct_dump_zwz_anti.bin",
    "fct_dump_zwz_anti_sal.bin",
    "fct_dump_zwz_up.bin",
    "fct_dump_zwz_up_sal.bin",
    "sbc_dump_qns.bin",
    "sbc_dump_qsr.bin",
    "sbc_dump_sfx.bin",
    "sbc_dump_utau.bin",
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


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


def differing_element_indices(a_path: Path, b_path: Path) -> np.ndarray:
    """Return flat binary64 slots whose raw bits differ."""
    a = np.fromfile(a_path, dtype="<u8")
    b = np.fromfile(b_path, dtype="<u8")
    if a.shape != b.shape:
        raise SystemExit(f"stream size differs: {a_path} vs {b_path}")
    return np.flatnonzero(a != b)


def halo_only_difference(
    a_path: Path,
    b_path: Path,
    *,
    jpi: int,
    jpj: int,
    hls: int,
) -> bool:
    """True only when every changed binary64 slot is outside the interior."""
    changed = differing_element_indices(a_path, b_path)
    if changed.size == 0:
        return True
    nxy = jpi * jpj
    if a_path.stat().st_size % (8 * nxy):
        raise SystemExit(f"stream is not a haloed 2-D/3-D field: {a_path}")
    nlev = a_path.stat().st_size // (8 * nxy)
    mask = np.zeros((nlev, jpj, jpi), dtype=bool)
    mask.reshape(-1)[changed] = True
    return not bool(mask[:, hls:jpj-hls, hls:jpi-hls].any())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bracket-dir", type=Path, required=True)
    parser.add_argument("--determinism-a-dir", type=Path, required=True)
    parser.add_argument("--determinism-b-dir", type=Path, required=True)
    parser.add_argument("--determinism-a-binary-receipt", type=Path, required=True)
    parser.add_argument("--determinism-b-binary-receipt", type=Path, required=True)
    parser.add_argument("--mld-maps", type=Path, required=True)
    parser.add_argument("--nemo-source", type=Path, required=True)
    parser.add_argument("--nemo-binary", type=Path, required=True)
    parser.add_argument("--donor-restart", type=Path, required=True)
    parser.add_argument("--expected-raw-sha", required=True)
    parser.add_argument("--expected-restart-sha", required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--expected-binary-sha", required=True)
    parser.add_argument("--expected-donor-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row-19 receipt requires CPU fp64")
    run = args.run_dir.resolve()
    bracket = args.bracket_dir.resolve()
    det_a = args.determinism_a_dir.resolve()
    det_b = args.determinism_b_dir.resolve()
    raw_path = run / RAW_NAME
    if time_level_for_dump(RAW_NAME) != "now":
        raise SystemExit("raw MXL time-level registry changed")
    if raw_path.stat().st_size != EXPECTED_SIZE:
        raise SystemExit(f"raw dump size changed: {raw_path.stat().st_size}")
    if sha256(raw_path) != args.expected_raw_sha:
        raise SystemExit("raw dump SHA does not match the human-bound receipt")
    if sha256(args.nemo_source) != args.expected_source_sha:
        raise SystemExit("active NEMO source SHA does not match the receipt")
    if sha256(args.nemo_binary) != args.expected_binary_sha:
        raise SystemExit("NEMO executable SHA does not match the receipt")
    if sha256(args.donor_restart) != args.expected_donor_sha:
        raise SystemExit("donor restart SHA does not match the receipt")
    if args.nemo_binary.stat().st_mtime_ns < args.nemo_source.stat().st_mtime_ns:
        raise SystemExit("NEMO executable is older than the instrumented source")
    for receipt in (
        args.determinism_a_binary_receipt,
        args.determinism_b_binary_receipt,
    ):
        fields = receipt.read_text(errors="strict").split()
        if not fields or fields[0] != args.expected_binary_sha:
            raise SystemExit(
                f"determinism run binary receipt does not bind current binary: {receipt}")
    donor_name = "DINO_00005760_restart.nc"
    for control_dir in (bracket, det_a, det_b):
        if sha256(control_dir / donor_name) != args.expected_donor_sha:
            raise SystemExit(
                f"control donor restart does not match the receipt: {control_dir}")
    for control_dir in (det_a, det_b):
        if sha256(control_dir / "nemo") != args.expected_binary_sha:
            raise SystemExit(
                f"determinism run executable does not bind current binary: {control_dir}")

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

    new_bins = {p.name for p in run.glob("*.bin")}
    old_bins = {p.name for p in bracket.glob("*.bin")}
    if new_bins - old_bins != {RAW_NAME} or old_bins - new_bins:
        raise SystemExit(
            "row-19 stream inventory changed: "
            f"new={new_bins-old_bins}, missing={old_bins-new_bins}")
    shared = sorted(old_bins)
    strict_equal = []
    halo_exclusions = {}
    for name in shared:
        if sha256(run / name) == sha256(bracket / name):
            strict_equal.append(name)
            continue
        if name not in UNINITIALIZED_HALO_STREAMS:
            raise SystemExit(f"write-only shared-stream bracket failed: {name}")
        current_changed = differing_element_indices(run / name, bracket / name)
        control_changed = differing_element_indices(det_a / name, det_b / name)
        if current_changed.size != 4 or control_changed.size != 4:
            raise SystemExit(
                f"registered four-slot halo signature changed for {name}: "
                f"patch={current_changed.size}, control={control_changed.size}")
        if not np.array_equal(current_changed, control_changed):
            raise SystemExit(
                f"patch/control changed-slot signature differs for {name}")
        if not halo_only_difference(
            run / name, bracket / name, jpi=jpi, jpj=jpj, hls=hls
        ) or not halo_only_difference(
            det_a / name, det_b / name, jpi=jpi, jpj=jpj, hls=hls
        ):
            raise SystemExit(f"physical-interior stream difference in {name}")
        halo_exclusions[name] = {
            "changed_binary64_slots": current_changed.tolist(),
            "determinism_a_sha256": sha256(det_a / name),
            "determinism_b_sha256": sha256(det_b / name),
            "patch_sha256": sha256(run / name),
            "bracket_sha256": sha256(bracket / name),
        }
    if set(halo_exclusions) != UNINITIALIZED_HALO_STREAMS:
        raise SystemExit(
            "registered uninitialized-halo inventory changed: "
            f"observed={sorted(halo_exclusions)}")

    # Red bracket control: a changed physical-interior binary64 slot must not
    # be accepted merely because its filename appears in the exclusion list.
    control_name = sorted(UNINITIALIZED_HALO_STREAMS)[0]
    control_data = np.fromfile(run / control_name, dtype="<u8").reshape(
        -1, jpj, jpi)
    control_copy = control_data.copy()
    control_copy[0, hls, hls] ^= np.uint64(1)
    bracket_control_fired = bool(np.any(
        control_data[:, hls:jpj-hls, hls:jpi-hls]
        != control_copy[:, hls:jpj-hls, hls:jpi-hls]))
    if not bracket_control_fired:
        raise SystemExit("physical-interior bracket control did not fire")

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
            "physical_interior_bracket_plant_fired": bracket_control_fired,
        },
        "bracket": {
            "restart_sha256": restart_sha,
            "shared_stream_count": len(shared),
            "strict_byte_identical_stream_count": len(strict_equal),
            "new_stream": RAW_NAME,
            "uninitialized_halo_exclusions": halo_exclusions,
            "mechanism": (
                "the same four out-of-interior binary64 slots differ in two "
                "runs of the identical control binary; every physical "
                "interior slot and both restart containers are exact"
            ),
        },
        "nemo_line": "cfgs/DINO/MY_SRC/zdftke.F90:831-833",
        "provenance_sha256": {
            str(raw_path): sha256(raw_path),
            str(restart_on): restart_sha,
            str(args.mld_maps.resolve()): sha256(args.mld_maps.resolve()),
            str(args.nemo_source.resolve()): sha256(args.nemo_source.resolve()),
            str(args.nemo_binary.resolve()): sha256(args.nemo_binary.resolve()),
            str(args.donor_restart.resolve()): sha256(args.donor_restart.resolve()),
            str(args.determinism_a_binary_receipt.resolve()): sha256(
                args.determinism_a_binary_receipt.resolve()),
            str(args.determinism_b_binary_receipt.resolve()): sha256(
                args.determinism_b_binary_receipt.resolve()),
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
