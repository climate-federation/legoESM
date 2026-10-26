#!/usr/bin/env python
"""Own row-30 zdepu from independently dumped operands at day 180."""
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
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bn2_alpha_compare as loaders  # noqa: E402
import dump_lane  # noqa: E402
import ldf_slp_per_element as ldf  # noqa: E402
import zdf_chain_sweep as sweep  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.eos import nemo_r3t_stretch  # noqa: E402

BAR = 1.0e-15
SHAS = {
    "eiv_dump_gdept.bin":
        "0e11bebbee22106dcb09cb50a0a76942408049e3bb7da14f15fc86998c93ad67",
    "eiv_dump_e3u_miku.bin":
        "7ef49fe8286fce72fd46bea6e72abb9bd01843861d66f46420020404bb14f938",
    "eiv_dump_zdepu.bin":
        "3ea7def80a96e51a9020f65ae9b438af40072bd7c7d2d2e04c87c8c765959e69",
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
    parser.add_argument("--dump-manifest", type=Path, required=True)
    parser.add_argument("--nemo-source", type=Path, required=True)
    parser.add_argument("--expected-repo-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row30 zdepu probe requires CPU fp64")
    git = lambda *a: subprocess.check_output(["git", *a], text=True).strip()
    if git("rev-parse", "HEAD") != args.expected_repo_sha or git(
            "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("row30 zdepu probe requires registered clean HEAD")
    run = args.run_dir.resolve()
    if Path(dump_lane.RUN_DIR).resolve() != run:
        raise SystemExit("lane selector does not resolve to registered run")
    bound = {
        args.parent_artifact:
            "5c01587801a5ebc10f1522e33e425e9f81b53c60e465a981ccaf469e1c4f1c74",
        args.bracket_receipt:
            "6eea10e2b3034ba81999c55c5dd2b37891f6e80cd48d56b43afe2b0cca45afbe",
        args.dump_manifest:
            "ba10ac72c4c2045a6953ffa6b0fbd72eea929e0fc4155757b4985a2ae4ef5111",
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
        **{run / name: value for name, value in SHAS.items()},
    }
    for path, expected in bound.items():
        if sha256(path.resolve()) != expected:
            raise SystemExit(f"SHA receipt changed: {path}")
    source = args.nemo_source.read_text(errors="strict")
    quote = "zdepu = 0.5_wp * ( ( gdept(ji,jj,jk,Kmm) + gdept(ji+1,jj,jk,Kmm) )"
    if quote not in source or "- e3u(ji,jj,miku(ji,jj),Kmm)" not in source:
        raise SystemExit("ldfslp.F90 zdepu source quote changed")

    state = ldf.build_state()
    _, outer = ldf.capture_locals(
        state["recall"], ldf.compute_nemo_native_slopes.__code__)
    _, calls = ldf.capture_return_locals(
        state["recall"], next(
            item for item in ldf.compute_nemo_native_slopes.__code__.co_consts
            if hasattr(item, "co_name") and item.co_name == "_uv_slp"))
    u_loc, raw = calls[0]
    jpi, jpj, _, hls = loaders._read_dims(str(run))
    nemo = {
        name: loaders._load_haloed(str(run / name), jpi, jpj, hls)
        for name in SHAS
    }
    nk = min(raw.shape[-1], *(value.shape[-1] for value in nemo.values()))
    wet = np.zeros_like(state["active"][..., :nk], dtype=bool)
    wet_full = ldf.wet_u_mask(
        np.asarray(state["active"]), state["u_mask"])[..., :nk]
    wet[..., ldf.KLO:min(ldf.KHI, nk)] = wet_full[
        ..., ldf.KLO:min(ldf.KHI, nk)]
    focus = sweep.focus_from_maps(args.mld_maps)
    expected = nemo["eiv_dump_zdepu.bin"][..., :nk]
    dumped_gdept = nemo["eiv_dump_gdept.bin"][..., :nk]
    dumped_e3u = nemo["eiv_dump_e3u_miku.bin"][..., :nk]

    # Independent oracle-operands arm: preserve the written parentheses.
    dumped_pair = dumped_gdept + np.roll(dumped_gdept, -1, axis=1)
    dumped_literal = np.float64(0.5) * (dumped_pair - dumped_e3u)

    raw_gdept = jnp.asarray(
        state["z_coord"].nemo_gdept_0, dtype=state["T"].dtype)
    stretch = nemo_r3t_stretch(
        state["z_coord"], state["eta"], state["H_bathy"],
        evaluation="nemo_reciprocal")
    live_gdept = jax.lax.optimization_barrier(
        raw_gdept * stretch[..., None])
    live_pair = jax.lax.optimization_barrier(
        live_gdept + jnp.roll(live_gdept, -1, axis=1))
    live_e3u = jnp.asarray(outer["e3u_k"])[..., :1]
    production_literal = jax.lax.optimization_barrier(
        jnp.asarray(0.5, live_gdept.dtype)
        * jax.lax.optimization_barrier(live_pair - live_e3u))

    def compiled_literal(ssh):
        stretch_jit = nemo_r3t_stretch(
            state["z_coord"], ssh, state["H_bathy"],
            evaluation="nemo_reciprocal")
        gd = jax.lax.optimization_barrier(
            raw_gdept * stretch_jit[..., None])
        pair = jax.lax.optimization_barrier(
            gd + jnp.roll(gd, -1, axis=1))
        return jax.lax.optimization_barrier(
            jnp.asarray(0.5, gd.dtype)
            * jax.lax.optimization_barrier(pair - live_e3u))

    production_jit_literal = jax.jit(compiled_literal)(state["eta"])

    scores = {
        "current_production_zdepu": sweep.metrics(
            np.asarray(u_loc["zdep_face"])[..., :nk], expected, wet, focus, BAR),
        "dumped_operand_literal": sweep.metrics(
            dumped_literal, expected, wet, focus, BAR),
        "production_live_gdept": sweep.metrics(
            np.asarray(live_gdept)[..., :nk], dumped_gdept, wet, focus, BAR),
        "production_literal": sweep.metrics(
            np.asarray(production_literal)[..., :nk], expected, wet, focus, BAR),
        "production_jit_literal": sweep.metrics(
            np.asarray(production_jit_literal)[..., :nk], expected,
            wet, focus, BAR),
    }

    old_association = np.float64(0.5) * dumped_pair - dumped_e3u
    rolled_face = np.float64(0.5) * (
        dumped_pair - np.roll(dumped_e3u, 1, axis=1))
    wet_nan = dumped_literal.copy()
    first_wet = tuple(np.argwhere(wet)[0])
    wet_nan[first_wet] = np.nan
    ulp = dumped_literal.copy()
    ulp[first_wet] = np.nextafter(ulp[first_wet], np.inf)

    def red(planted):
        return not sweep.metrics(planted, expected, wet, focus, BAR)["pass"]

    controls = {
        "subtract_after_half_fails": red(old_association),
        "current_jacobian_depth_fails": not scores[
            "current_production_zdepu"]["pass"],
        "rolled_face_e3u_fails": red(rolled_face),
        "wet_nan_fails": red(wet_nan),
        "one_ulp_exact_identity_fires": not np.array_equal(ulp, dumped_literal),
    }
    if not all(controls.values()):
        raise SystemExit("row30 zdepu control did not fire")
    confirmed = (scores["dumped_operand_literal"]["pass"]
                 and scores["production_live_gdept"]["pass"]
                 and scores["production_literal"]["pass"]
                 and scores["production_jit_literal"]["pass"]
                 and not scores["current_production_zdepu"]["pass"])
    artifact = {
        "schema": "dino-zdf-row30-zdepu-operands-v1",
        "disposition": "DIVERGED-OWNED" if confirmed else "DIVERGED",
        "owner": ("stored-reciprocal live gdept plus literal in-parentheses "
                  "e3u(miku,Kmm) subtraction at ldfslp.F90:298-301"
                  if confirmed else None),
        "bar": BAR,
        "focus_columns_ji": [list(x) for x in focus],
        "scores": scores,
        "controls": controls,
        "repo_sha": args.expected_repo_sha,
        "source_quote": quote,
        "provenance_sha256": {
            str(path.resolve()): sha256(path.resolve())
            for path in (*bound.keys(), Path(__file__), Path(ldf.__file__))
        },
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for name, score in scores.items():
        focus_fail = sum(not item["pass"] for item in score["focus"])
        print(f"{name}: {score['n_diverged_columns']}/{score['n_wet_columns']} "
              f"fail bar {BAR:.1e}; focus_fail={focus_fail}")
    print(f"DISPOSITION: {artifact['disposition']}; owner={artifact['owner']}")
    return 30 if confirmed or not scores["production_literal"]["pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
