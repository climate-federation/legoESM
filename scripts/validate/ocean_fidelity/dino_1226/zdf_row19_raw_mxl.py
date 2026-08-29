#!/usr/bin/env python
"""Score the preregistered day-180 ZDF row-19 raw mixing-length slot.

The receipt is fail-closed on the write-only bracket: a byte-identical restart,
identical shared binary physics streams (apart from the registered nondeterministic
``cor2d`` exclusion), and exactly one new stream are required before a numerical
result can be called VERIFIED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import bn2_alpha_compare as loaders
import jax
import netCDF4  # noqa: N813
import numpy as np
import zdf_chain_sweep as sweep
from legoesm.ocean.fidelity.time_levels import time_level_for_dump

BAR = 1.0e-15
RAW_NAME = "tke_dump_zmxlm_raw.bin"
NONDETERMINISTIC_EXCLUSION = "cor2d_dump_zu_trd_substep1.bin"
EXPECTED_SIZE = 2_980_224


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bracket-dir", type=Path, required=True)
    parser.add_argument("--mld-maps", type=Path, required=True)
    parser.add_argument("--nemo-source", type=Path, required=True)
    parser.add_argument("--expected-raw-sha", required=True)
    parser.add_argument("--expected-restart-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row-19 receipt requires CPU fp64")
    run = args.run_dir.resolve()
    bracket = args.bracket_dir.resolve()
    raw_path = run / RAW_NAME
    if time_level_for_dump(RAW_NAME) != "now":
        raise SystemExit("raw MXL time-level registry changed")
    if raw_path.stat().st_size != EXPECTED_SIZE:
        raise SystemExit(f"raw dump size changed: {raw_path.stat().st_size}")
    if sha256(raw_path) != args.expected_raw_sha:
        raise SystemExit("raw dump SHA does not match the human-bound receipt")

    restart_name = "DINO_00005761_restart.nc"
    restart_on = run / restart_name
    restart_off = bracket / restart_name
    restart_sha = sha256(restart_on)
    if restart_sha != args.expected_restart_sha or sha256(restart_off) != restart_sha:
        raise SystemExit("write-only bracket restart identity failed")

    new_bins = {p.name for p in run.glob("*.bin")}
    old_bins = {p.name for p in bracket.glob("*.bin")}
    if new_bins - old_bins != {RAW_NAME} or old_bins - new_bins:
        raise SystemExit(
            "row-19 stream inventory changed: "
            f"new={new_bins-old_bins}, missing={old_bins-new_bins}")
    shared = sorted(old_bins - {NONDETERMINISTIC_EXCLUSION})
    unequal = [name for name in shared if sha256(run / name) != sha256(bracket / name)]
    if unequal:
        raise SystemExit(f"write-only shared-stream bracket failed: {unequal}")

    source = args.nemo_source.read_text(errors="strict")
    for literal in (
        "zrn2 = MAX( rn2(ji,jj,jk), rsmall )",
        "SQRT( 2._wp * en(ji,jj,jk) / zrn2 )",
    ):
        if literal not in source:
            raise SystemExit(f"active NEMO source expression changed: {literal}")

    focus = sweep.focus_from_maps(args.mld_maps)
    jpi, jpj, jpk, hls = loaders._read_dims(str(run))
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
    lego = np.maximum(
        np.float64(0.01),
        np.sqrt((np.float64(2.0) * en) / np.maximum(rn2, rsmall)),
    )
    metric = sweep.metrics(lego, nemo, wet, focus, BAR)
    exact_unequal = int(np.count_nonzero(wet & ~(lego == nemo)))

    exact_plant = lego.copy()
    idx = tuple(int(x) for x in np.argwhere(wet)[0])
    exact_plant[idx] = np.nextafter(exact_plant[idx], np.float64(np.inf))
    ulp_control = int(np.count_nonzero(wet & ~(exact_plant == nemo))) > exact_unequal
    roll_control = not sweep.metrics(np.roll(nemo, 1, axis=1), nemo, wet, focus, BAR)["pass"]
    value_plant = nemo.copy()
    value_plant[idx] += np.float64(2.0e-15) * max(
        np.float64(1.0), np.sqrt(np.mean(nemo[wet] ** 2)))
    value_control = not sweep.metrics(value_plant, nemo, wet, focus, BAR)["pass"]
    if not (ulp_control and roll_control and value_control):
        raise SystemExit("one or more planted row-19 controls failed to fire")

    artifact = {
        "schema": "dino-zdf-row19-raw-mxl-v1",
        "disposition": "VERIFIED" if metric["pass"] else "DIVERGED",
        "bar": BAR,
        "metric": metric,
        "exact_unequal_wet_elements": exact_unequal,
        "focus_columns_ji": [[j, i] for j, i in focus],
        "controls": {
            "plus_one_ulp_exact_census_fired": ulp_control,
            "one_i_roll_fired": roll_control,
            "value_plant_fired": value_control,
        },
        "bracket": {
            "restart_sha256": restart_sha,
            "shared_stream_count": len(shared),
            "new_stream": RAW_NAME,
            "nondeterministic_exclusion": NONDETERMINISTIC_EXCLUSION,
        },
        "nemo_line": "cfgs/DINO/MY_SRC/zdftke.F90:831-833",
        "provenance_sha256": {
            str(raw_path): sha256(raw_path),
            str(restart_on): restart_sha,
            str(args.mld_maps.resolve()): sha256(args.mld_maps.resolve()),
            str(args.nemo_source.resolve()): sha256(args.nemo_source.resolve()),
        },
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        f"row19 {artifact['disposition']}: {metric['n_diverged_columns']}/"
        f"{metric['n_wet_columns']} columns fail; exact unequal={exact_unequal}; "
        f"focus_fail={sum(not x['pass'] for x in metric['focus'])}")
    return 0 if metric["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
