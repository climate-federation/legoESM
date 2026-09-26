#!/usr/bin/env python
"""Close row 30 with existing raw and post-Shapiro U/V dumps."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bn2_alpha_compare as loaders  # noqa: E402
import dump_lane  # noqa: E402
import ldf_slp_per_element as ldf  # noqa: E402
import zdf_chain_sweep as sweep  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402

BAR = 1.0e-15
DUMPS = {
    "uslp_raw": ("u", "eiv_dump_uslp_raw.bin",
                 "866a5fbfbc4c2993a990e441f46670302497848222c83da0a90c7470602275c9"),
    "vslp_raw": ("v", "eiv_dump_vslp_raw.bin",
                 "09848b07dd7e149463fe16b5eee5d7c91402ae536b6e7725847197726c9e2ea2"),
    "uslp_postshapiro": ("u", "eiv_dump_uslp_postshapiro.bin",
                         "b332b6dc0535a1e039222a6dcc557044db4417a9b9af495a3fc8e9ed5a373b4e"),
    "vslp_postshapiro": ("v", "eiv_dump_vslp_postshapiro.bin",
                         "1558e104c21b37ed64914abe3cba0a174599b2b34054b7dc7ea6f2487b510a56"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mld-maps", type=Path, required=True)
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--bracket-receipt", type=Path, required=True)
    parser.add_argument("--nemo-source", type=Path, required=True)
    parser.add_argument("--expected-repo-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row30 composite close requires CPU fp64")
    git = lambda *a: subprocess.check_output(["git", *a], text=True).strip()
    if git("rev-parse", "HEAD") != args.expected_repo_sha or git(
            "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("row30 composite close requires registered clean HEAD")
    run = args.run_dir.resolve()
    if Path(dump_lane.RUN_DIR).resolve() != run:
        raise SystemExit("lane selector does not resolve to registered run")
    bound = {
        args.parent_artifact:
            "8be5ec24beed64967b1a31646df516a40a1b6647f4e5c334d00852cb3e4f435e",
        args.bracket_receipt:
            "6eea10e2b3034ba81999c55c5dd2b37891f6e80cd48d56b43afe2b0cca45afbe",
        args.nemo_source:
            "2d59df4697b3d16f0ee9dc2b38ce929cca707d59ef43442dee3600c8d8b1f7ca",
        args.mld_maps:
            "9fb7344d1e6f92232d211f6a52ff8636022f0d9b0b05acea2b0bae6e6afd9bf0",
        run / "mesh_mask.nc":
            "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
        run / "DINO_00005760_restart.nc":
            "0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e",
        run / "DINO_00005761_restart.nc":
            "33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c",
        **{run / basename: digest
           for _, basename, digest in DUMPS.values()},
    }
    for path, expected in bound.items():
        if sha256(path.resolve()) != expected:
            raise SystemExit(f"SHA receipt changed: {path}")
    source = args.nemo_source.read_text(errors="strict")
    for quote in (
            "zwz(ji,jj) = ( zfi * zau / ( zbu - zeps )",
            "zww(ji,jj) = ( zfj * zav / ( zbv - zeps )",
            "uslp(ji,jj,jk) = z1_16 *",
            "vslp(ji,jj,jk) = z1_16 *"):
        if quote not in source:
            raise SystemExit(f"active ldfslp source quote changed: {quote}")

    state = ldf.build_state()
    uv_code = next(
        item for item in ldf.compute_nemo_native_slopes.__code__.co_consts
        if isinstance(item, types.CodeType) and item.co_name == "_uv_slp")
    traced, calls = ldf.capture_return_locals(state["recall"], uv_code)
    plain = state["recall"]()
    if len(calls) != 2 or not all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(plain, traced)):
        raise SystemExit("row30 frame capture changed production output")
    values = {
        "uslp_raw": np.asarray(calls[0][1]),
        "vslp_raw": np.asarray(calls[1][1]),
        "uslp_postshapiro": np.asarray(traced[0]),
        "vslp_postshapiro": np.asarray(traced[1]),
    }
    masks = {
        "u": ldf.wet_u_mask(state["active"], state["u_mask"]),
        "v": ldf.wet_v_mask(state["active"], state["v_mask"]),
    }
    focus = sweep.focus_from_maps(args.mld_maps)
    jpi, jpj, _, hls = loaders._read_dims(str(run))
    scores = {}
    controls = {}
    first = None
    for stage, (direction, basename, _) in DUMPS.items():
        expected = loaders._load_haloed(str(run / basename), jpi, jpj, hls)
        actual = values[stage]
        nk = min(actual.shape[-1], expected.shape[-1], masks[direction].shape[-1])
        wet = np.zeros_like(masks[direction][..., :nk], dtype=bool)
        wet[..., ldf.KLO:min(ldf.KHI, nk)] = masks[direction][
            ..., ldf.KLO:min(ldf.KHI, nk)]
        expected = expected[..., :nk]
        score = sweep.metrics(actual[..., :nk], expected, wet, focus, BAR)
        scores[stage] = score
        focus_fail = sum(not item["pass"] for item in score["focus"])
        print(f"{stage}: {score['n_diverged_columns']}/{score['n_wet_columns']} "
              f"fail bar {BAR:.1e}; focus_fail={focus_fail}")
        if first is None and not score["pass"]:
            first = stage
            break
        baseline = sweep.metrics(expected, expected, wet, focus, BAR)
        if not baseline["pass"]:
            raise SystemExit(f"control baseline is not exact: {stage}")
        idx = tuple(np.argwhere(wet)[0])
        planted = expected.copy()
        planted[idx] += np.float64(1.0e-8) * max(
            np.float64(1.0), abs(expected[idx]))
        rolled = np.roll(expected, 1, axis=1)
        wet_nan = expected.copy()
        wet_nan[idx] = np.nan
        stage_controls = {
            "bar_scale": not sweep.metrics(
                planted, expected, wet, focus, BAR)["pass"],
            "zonal_roll": not sweep.metrics(
                rolled, expected, wet, focus, BAR)["pass"],
            "wet_nan": not sweep.metrics(
                wet_nan, expected, wet, focus, BAR)["pass"],
        }
        if not all(stage_controls.values()):
            raise SystemExit(f"row30 composite control did not fire: {stage}")
        controls[stage] = stage_controls

    artifact = {
        "schema": "dino-zdf-row30-composite-close-v1",
        "disposition": "VERIFIED" if first is None else "DIVERGED",
        "first_divergence": first,
        "bar": BAR,
        "focus_columns_ji": [list(x) for x in focus],
        "scores": scores,
        "controls": controls,
        "repo_sha": args.expected_repo_sha,
        "provenance_sha256": {
            str(path.resolve()): sha256(path.resolve())
            for path in (*bound.keys(), Path(__file__), Path(ldf.__file__))
        },
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"FIRST DIVERGENT STAGE: {first or 'none'}")
    return 30 if first else 0


if __name__ == "__main__":
    raise SystemExit(main())
