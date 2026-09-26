#!/usr/bin/env python
"""Localize row-30's first raw U-slope divergence using existing dumps."""
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
    "dump_nmln.bin": "b1bb902dba4801eb5613ab563589bef388f506a4bb1050caebd84d161dfba111",
    "eiv_dump_gdept.bin": "0e11bebbee22106dcb09cb50a0a76942408049e3bb7da14f15fc86998c93ad67",
    "eiv_dump_zau.bin": "78f5d2159baad245f6c2fbb29f4cfbc24bc2724cfbd8b4077953e5b3576398b1",
    "eiv_dump_zbu_post.bin": "ed2079f378739d502967d13d99f37eda3282811185260cae890985cfef123e9c",
    "eiv_dump_uslp_raw.bin": "866a5fbfbc4c2993a990e441f46670302497848222c83da0a90c7470602275c9",
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
    parser.add_argument("--nemo-source", type=Path, required=True)
    parser.add_argument("--expected-repo-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row30 raw-slope probe requires CPU fp64")
    if os.environ.get("DINO_1226_LANE") != "d180":
        raise SystemExit("row30 raw-slope probe requires the d180 lane")
    git = lambda *a: subprocess.check_output(["git", *a], text=True).strip()
    if git("rev-parse", "HEAD") != args.expected_repo_sha or git(
            "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("row30 raw-slope probe requires registered clean HEAD")
    run = args.run_dir.resolve()
    if Path(dump_lane.RUN_DIR).resolve() != run:
        raise SystemExit("lane selector does not resolve to registered run")
    bound = {
        args.parent_artifact: "365942724667605ebc1de470ffb8b6dd11f07f9ae75272ac43e3b70c559a614b",
        args.nemo_source: "8b4d8cffe35d66241eb77bdc508ef15d6dd90d8ff192fd60201ff83a7d523a29",
        args.mld_maps: "9fb7344d1e6f92232d211f6a52ff8636022f0d9b0b05acea2b0bae6e6afd9bf0",
        **{run / name: value for name, value in SHAS.items()},
    }
    for path, expected in bound.items():
        if sha256(path.resolve()) != expected:
            raise SystemExit(f"SHA receipt changed: {path}")
    source = args.nemo_source.read_text(errors="strict")
    quote = (
        "zdepu = 0.5_wp * ( ( gdept(ji,jj,jk,Kmm) + gdept(ji+1,jj,jk,Kmm) )")
    if quote not in source or "- e3u(ji,jj,miku(ji,jj),Kmm)" not in source:
        raise SystemExit("ldfslp.F90:261-264 quote changed")

    state = ldf.build_state()
    _, outer = ldf.capture_locals(
        state["recall"], ldf.compute_nemo_native_slopes.__code__)
    _, calls = ldf.capture_return_locals(
        state["recall"], next(
            item for item in ldf.compute_nemo_native_slopes.__code__.co_consts
            if hasattr(item, "co_name") and item.co_name == "_uv_slp"))
    u_loc, production_raw = calls[0]
    jpi, jpj, _, hls = loaders._read_dims(str(run))
    nemo_nmln = loaders._load_haloed(
        str(run / "dump_nmln.bin"), jpi, jpj, hls)[..., 0]
    nemo_gdept = loaders._load_haloed(
        str(run / "eiv_dump_gdept.bin"), jpi, jpj, hls)
    nemo_raw = loaders._load_haloed(
        str(run / "eiv_dump_uslp_raw.bin"), jpi, jpj, hls)
    nemo_zau = loaders._load_haloed(
        str(run / "eiv_dump_zau.bin"), jpi, jpj, hls)
    nk = min(production_raw.shape[-1], nemo_raw.shape[-1], nemo_gdept.shape[-1])
    wet = np.zeros_like(state["active"][..., :nk], dtype=bool)
    wet_full = ldf.wet_u_mask(
        np.asarray(state["active"]), state["u_mask"])[..., :nk]
    wet[..., ldf.KLO:min(ldf.KHI, nk)] = wet_full[
        ..., ldf.KLO:min(ldf.KHI, nk)]
    focus = sweep.focus_from_maps(args.mld_maps)

    production_iku = np.asarray(outer["iku"])
    nemo_iku = np.maximum(nemo_nmln, np.roll(nemo_nmln, -1, axis=1))
    wet2 = np.any(wet, axis=-1)
    offsets = {
        str(offset): int(np.count_nonzero(
            wet2 & (production_iku != nemo_iku + offset)))
        for offset in (-1, 0, 1)
    }
    # The canonical gdept dump contains the 35 evaluated NEMO slots; verify
    # those against production, then use that verified production array's
    # finite 36th sentinel so _uv_slp retains its registered full shape.
    gdept_full = jnp.asarray(outer["_gd_col"], dtype=state["T"].dtype)
    gdept_score = sweep.metrics(
        np.asarray(gdept_full)[..., :nk], nemo_gdept[..., :nk], wet, focus, BAR)
    raw_gdept = jnp.asarray(
        state["z_coord"].nemo_gdept_0, dtype=state["T"].dtype)
    reciprocal_stretch = nemo_r3t_stretch(
        state["z_coord"], state["eta"], state["H_bathy"],
        evaluation="nemo_reciprocal")
    reciprocal_gdept = raw_gdept * reciprocal_stretch[..., None]
    reciprocal_gdept_score = sweep.metrics(
        np.asarray(reciprocal_gdept)[..., :nk], nemo_gdept[..., :nk],
        wet, focus, BAR)
    gdept = reciprocal_gdept
    face_sum = jax.lax.optimization_barrier(
        gdept + jnp.roll(gdept, -1, axis=1))
    live_surface_e3u = jnp.asarray(outer["e3u_k"])[..., :1]
    correct_zdepu = jax.lax.optimization_barrier(
        jnp.asarray(0.5, dtype=gdept.dtype)
        * jax.lax.optimization_barrier(face_sum - live_surface_e3u))
    literal_raw = outer["_uv_slp"](
        outer["zgru"], outer["zb_u"], outer["e1u"], outer["e3u_k"],
        outer["iku"], outer["r1_hmlu"], correct_zdepu, outer["umask3"])

    production_score = sweep.metrics(
        np.asarray(production_raw)[..., :nk], nemo_raw[..., :nk], wet, focus, BAR)
    correct_score = sweep.metrics(
        np.asarray(literal_raw)[..., :nk], nemo_raw[..., :nk], wet, focus, BAR)
    current_zdep = np.asarray(u_loc["zdep_face"])[..., :nk]
    correct_zdep_np = np.asarray(correct_zdepu)[..., :nk]
    zdep_delta = sweep.metrics(current_zdep, correct_zdep_np, wet, focus, BAR)
    # Controls attach to the independently verified upstream zau row; the
    # candidate stage is allowed to stay red without disabling its receipt.
    controls = sweep.planted_controls(
        np.asarray(u_loc["zau"])[..., :nk], nemo_zau[..., :nk], wet, BAR)
    controls["old_tpoint_surface_e3_fails"] = not production_score["pass"]
    rolled = jax.lax.optimization_barrier(
        jnp.asarray(0.5, dtype=gdept.dtype)
        * jax.lax.optimization_barrier(
            face_sum - jnp.roll(live_surface_e3u, 1, axis=1)))
    rolled_raw = outer["_uv_slp"](
        outer["zgru"], outer["zb_u"], outer["e1u"], outer["e3u_k"],
        outer["iku"], outer["r1_hmlu"], rolled, outer["umask3"])
    controls["rolled_face_e3_fails"] = not sweep.metrics(
        np.asarray(rolled_raw)[..., :nk], nemo_raw[..., :nk], wet, focus, BAR)["pass"]
    controls["shifted_nmln_fails"] = offsets["0"] != offsets["-1"]
    controls["production_jacobian_gdept_fails"] = not gdept_score["pass"]
    ulp_gdept = np.asarray(reciprocal_gdept).copy()
    first_wet_3d = tuple(np.argwhere(wet)[0])
    ulp_gdept[first_wet_3d] = np.nextafter(
        ulp_gdept[first_wet_3d], np.inf)
    controls["one_ulp_gdept_exact_fired"] = not np.array_equal(
        ulp_gdept, np.asarray(reciprocal_gdept))
    if not all(value for value in controls.values() if isinstance(value, bool)):
        failed = sorted(
            name for name, value in controls.items()
            if isinstance(value, bool) and not value)
        raise SystemExit(
            f"row30 raw-slope planted control did not fire: {failed}")

    owner = (reciprocal_gdept_score["pass"] and correct_score["pass"]
             and not production_score["pass"])
    artifact = {
        "schema": "dino-zdf-row30-uslp-raw-v1",
        "disposition": "DIVERGED-LIVE-GDEPT-AND-ZDEPU" if owner else "DIVERGED",
        "owner": ("stored-reciprocal live gdept plus e3u(miku,Kmm) in zdepu "
                  "at ldfslp.F90:261-264" if owner else None),
        "bar": BAR,
        "focus_columns_ji": [list(x) for x in focus],
        "iku_offset_failures": offsets,
        "production_raw": production_score,
        "live_face_zdepu_raw": correct_score,
        "current_vs_live_face_zdepu": zdep_delta,
        "gdept_existing_dump": gdept_score,
        "reciprocal_gdept_existing_dump": reciprocal_gdept_score,
        "controls": controls,
        "repo_sha": args.expected_repo_sha,
        "source_quote": quote,
        "provenance_sha256": {
            str(path.resolve()): sha256(path.resolve())
            for path in (*bound.keys(), Path(__file__), Path(ldf.__file__))
        },
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for name, score in (("production uslp_raw", production_score),
                        ("live-face zdepu uslp_raw", correct_score)):
        focus_fail = sum(not item["pass"] for item in score["focus"])
        print(f"{name}: {score['n_diverged_columns']}/{score['n_wet_columns']} "
              f"fail bar {BAR:.1e}; focus_fail={focus_fail}")
    print(f"iku offset failures: {offsets}")
    print(f"DISPOSITION: {artifact['disposition']}; owner={artifact['owner']}")
    return 30 if owner or not correct_score["pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
