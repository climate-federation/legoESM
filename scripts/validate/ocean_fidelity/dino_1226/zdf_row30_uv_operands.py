#!/usr/bin/env python
"""Score the preregistered source-ordered row-30 U/V slope operand ladder."""

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
from legoesm.ocean.fidelity.time_levels import time_level_for_dump  # noqa: E402
from zdf_stream_bracket import (  # noqa: E402
    files_byte_identical,
    one_bit_file_control,
    stream_manifest,
)

BAR = 1.0e-15
EXPECTED_SIZE = 3_273_984
EXPECTED_ON_STREAMS = 209
EXPECTED_OFF_STREAMS = 197
RESTART_SHA256 = "33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c"
OFF_BINARY_SHA256 = "4f28b00205658781d7c2a6bbc35362a5a19eb8d57d245bad28a9fbe96ebfafae"

NEW_DUMPS = (
    "eiv_dump_zgru_iik.bin",
    "eiv_dump_zgru_iikm1.bin",
    "eiv_dump_zau.bin",
    "eiv_dump_zav.bin",
    "eiv_dump_zbu_pre.bin",
    "eiv_dump_zbv_pre.bin",
    "eiv_dump_zbu_post.bin",
    "eiv_dump_zbv_post.bin",
    "eiv_dump_uslp_raw.bin",
    "eiv_dump_vslp_raw.bin",
    "eiv_dump_uslp_postshapiro.bin",
    "eiv_dump_vslp_postshapiro.bin",
)
EXISTING_DUMPS = (
    "eiv_dump_zgrv_iik.bin",
    "eiv_dump_zgrv_iikm1.bin",
)

# Registered disposition order. The two rolling slots form one gradient stage
# per direction, so both U receipts precede both V receipts; subsequent pairs
# follow ldfslp.F90's written U-then-V order. Each item is (receipt name,
# direction, dump basename, production value key, source line).
OPERANDS = (
    ("zgru_iik", "u", "eiv_dump_zgru_iik.bin", "zgru_iik", "203,217"),
    ("zgru_iikm1", "u", "eiv_dump_zgru_iikm1.bin", "zgru_iikm1", "203,217"),
    ("zgrv_iik", "v", "eiv_dump_zgrv_iik.bin", "zgrv_iik", "204,218"),
    ("zgrv_iikm1", "v", "eiv_dump_zgrv_iikm1.bin", "zgrv_iikm1", "204,218"),
    ("zau", "u", "eiv_dump_zau.bin", "zau", 242),
    ("zav", "v", "eiv_dump_zav.bin", "zav", 243),
    ("zbu_pre", "u", "eiv_dump_zbu_pre.bin", "zbu_pre", 244),
    ("zbv_pre", "v", "eiv_dump_zbv_pre.bin", "zbv_pre", 245),
    ("zbu_post", "u", "eiv_dump_zbu_post.bin", "zbu_post", 248),
    ("zbv_post", "v", "eiv_dump_zbv_post.bin", "zbv_post", 249),
    ("uslp_raw", "u", "eiv_dump_uslp_raw.bin", "uslp_raw", 269),
    ("vslp_raw", "v", "eiv_dump_vslp_raw.bin", "vslp_raw", 270),
    ("uslp_postshapiro", "u", "eiv_dump_uslp_postshapiro.bin", "uslp_postshapiro", 279),
    ("vslp_postshapiro", "v", "eiv_dump_vslp_postshapiro.bin", "vslp_postshapiro", 286),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def tracked_tree_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()


def _receipt_binds(path: Path, expected: str) -> bool:
    fields = path.read_text(errors="strict").split()
    return bool(fields and fields[0] == expected)


def _inventory_is_exact(on_names: set[str], off_names: set[str]) -> bool:
    return (
        len(on_names) == EXPECTED_ON_STREAMS
        and len(off_names) == EXPECTED_OFF_STREAMS
        and on_names - off_names == set(NEW_DUMPS)
        and not off_names - on_names
    )


def _capture_values(state: dict) -> tuple[dict[str, np.ndarray], dict]:
    plain = state["recall"]()
    uv_code = next(
        item for item in ldf.compute_nemo_native_slopes.__code__.co_consts
        if isinstance(item, types.CodeType) and item.co_name == "_uv_slp"
    )
    traced, calls = ldf.capture_return_locals(state["recall"], uv_code)
    if len(calls) != 2:
        raise SystemExit(f"expected two _uv_slp calls, captured {len(calls)}")
    if not all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(plain, traced)):
        raise SystemExit("local-frame capture changed a production result")
    u_loc, u_raw = calls[0]
    v_loc, v_raw = calls[1]
    def km1(a):
        return np.concatenate(
            [np.zeros_like(a[..., :1]), a[..., :-1]], axis=-1)
    values = {
        "_u_zg": np.asarray(u_loc["zg"]),
        "_v_zg": np.asarray(v_loc["zg"]),
        "zau": np.asarray(u_loc["zau"]),
        "zav": np.asarray(v_loc["zau"]),
        "zbu_pre": np.asarray(u_loc["zb_pair"]),
        "zbv_pre": np.asarray(v_loc["zb_pair"]),
        "zbu_post": np.asarray(u_loc["zbu"]),
        "zbv_post": np.asarray(v_loc["zbu"]),
        "uslp_raw": np.asarray(u_raw),
        "vslp_raw": np.asarray(v_raw),
        "uslp_postshapiro": np.asarray(traced[0]),
        "vslp_postshapiro": np.asarray(traced[1]),
    }
    meta = {"uv_call_order": ["u", "v"], "capture_inert": True, "km1": km1}
    return values, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bracket-dir", type=Path, required=True)
    parser.add_argument("--mld-maps", type=Path, required=True)
    parser.add_argument("--nemo-source", type=Path, required=True)
    parser.add_argument("--bracket-nemo-source", type=Path, required=True)
    parser.add_argument("--nemo-binary", type=Path, required=True)
    parser.add_argument("--bracket-nemo-binary", type=Path, required=True)
    parser.add_argument("--instrument-patch", type=Path, required=True)
    parser.add_argument("--dump-sha-manifest", type=Path, required=True)
    parser.add_argument("--on-source-manifest", type=Path, required=True)
    parser.add_argument("--off-source-manifest", type=Path, required=True)
    parser.add_argument("--expected-repo-sha", required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--expected-bracket-source-sha", required=True)
    parser.add_argument("--expected-binary-sha", required=True)
    parser.add_argument("--expected-patch-sha", required=True)
    parser.add_argument("--expected-on-source-manifest-sha", required=True)
    parser.add_argument("--expected-off-source-manifest-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row-30 receipt requires CPU fp64")
    if git_sha() != args.expected_repo_sha or not tracked_tree_clean():
        raise SystemExit("row-30 receipt requires the registered clean shadow-git HEAD")
    if os.environ.get("DINO_1226_LANE") != "d180":
        raise SystemExit("row-30 receipt accepts only DINO_1226_LANE=d180")

    run, off = args.run_dir.resolve(), args.bracket_dir.resolve()
    if Path(dump_lane.RUN_DIR).resolve() != run:
        raise SystemExit("DINO d180 lane selector does not resolve to --run-dir")
    fixed_receipts = {
        args.nemo_source.resolve(): args.expected_source_sha,
        args.bracket_nemo_source.resolve(): args.expected_bracket_source_sha,
        args.nemo_binary.resolve(): args.expected_binary_sha,
        args.bracket_nemo_binary.resolve(): OFF_BINARY_SHA256,
        args.instrument_patch.resolve(): args.expected_patch_sha,
        args.on_source_manifest.resolve(): args.expected_on_source_manifest_sha,
        args.off_source_manifest.resolve(): args.expected_off_source_manifest_sha,
    }
    for path, expected in fixed_receipts.items():
        if sha256(path) != expected:
            raise SystemExit(f"SHA receipt changed: {path}")
    if args.nemo_binary.stat().st_mtime_ns < args.nemo_source.stat().st_mtime_ns:
        raise SystemExit("instrumented binary is older than patched source")
    if not _receipt_binds(run / ".nemo_binary_sha256", args.expected_binary_sha):
        raise SystemExit("ON run does not bind the measured instrumented binary")
    if not _receipt_binds(off / ".nemo_binary_sha256", OFF_BINARY_SHA256):
        raise SystemExit("OFF run does not bind the certified control binary")
    if (sha256(run / "nemo") != args.expected_binary_sha
            or sha256(off / "nemo") != OFF_BINARY_SHA256):
        raise SystemExit("executed binary SHA changed")
    for directory in (run, off):
        if sha256(directory / "DINO_00005761_restart.nc") != RESTART_SHA256:
            raise SystemExit(f"certified restart changed: {directory}")
        log = (directory / "run.attempt1.log").read_text(errors="strict")
        if not log.rstrip().endswith("STOP 0"):
            raise SystemExit(f"NEMO did not stop cleanly on first attempt: {directory}")
    registered_inputs = (
        "DINO_00005760_restart.nc", "mesh_mask.nc", "namelist_cfg",
        "namelist_ref", "layout.dat", "layout.nc",
    )
    unequal_inputs = [
        name for name in registered_inputs
        if sha256(run / name) != sha256(off / name)
    ]
    if unequal_inputs:
        raise SystemExit(f"ON/OFF registered input manifest differs: {unequal_inputs}")

    on_manifest, off_manifest = stream_manifest(run), stream_manifest(off)
    on_names, off_names = set(on_manifest), set(off_manifest)
    if not _inventory_is_exact(on_names, off_names):
        raise SystemExit(
            f"row30 inventory changed: ON={len(on_names)} OFF={len(off_names)} "
            f"new={sorted(on_names-off_names)} missing={sorted(off_names-on_names)}")
    unequal = [name for name in sorted(off_names)
               if not files_byte_identical(run / name, off / name)]
    if unequal:
        raise SystemExit(f"row30 shared-stream bracket failed: {unequal}")
    shared = sorted(off_names)[0]
    controls = {
        "shared_streams_exact": len(unequal) == 0,
        "one_bit_shared_stream_fails": one_bit_file_control(run / shared),
        "missing_stream_inventory_fails": not _inventory_is_exact(
            on_names - {NEW_DUMPS[0]}, off_names),
    }
    if not all(controls.values()):
        raise SystemExit("row30 bracket control did not fire")

    expected_dump_shas = json.loads(args.dump_sha_manifest.read_text())
    if set(expected_dump_shas) != set(NEW_DUMPS):
        raise SystemExit("measured dump-SHA manifest has the wrong inventory")
    for name, expected in expected_dump_shas.items():
        path = run / name
        if path.stat().st_size != EXPECTED_SIZE or sha256(path) != expected:
            raise SystemExit(f"measured row30 dump receipt changed: {name}")
    for name in NEW_DUMPS + EXISTING_DUMPS:
        if time_level_for_dump(name) != "before":
            raise SystemExit(f"time-level registry changed: {name}")

    focus = sweep.focus_from_maps(args.mld_maps)
    state = ldf.build_state()
    plain_outer, outer = ldf.capture_locals(
        state["recall"], ldf.compute_nemo_native_slopes.__code__)
    values, capture = _capture_values(state)
    if not all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(plain_outer, state["recall"]())):
        raise SystemExit("outer-frame capture changed a production result")
    zgru, zgrv = np.asarray(outer["zgru"]), np.asarray(outer["zgrv"])
    if (not np.array_equal(values.pop("_u_zg"), zgru)
            or not np.array_equal(values.pop("_v_zg"), zgrv)
            or len(capture["uv_call_order"]) != 2):
        raise SystemExit("U/V capture association changed")
    km1 = capture.pop("km1")
    values.update({
        "zgru_iik": zgru,
        "zgrv_iik": zgrv,
        "zgru_iikm1": km1(zgru),
        "zgrv_iikm1": km1(zgrv),
    })
    jpi, jpj, jpk, hls = loaders._read_dims(str(run))
    masks = {
        "u": ldf.wet_u_mask(state["active"], state["u_mask"]),
        "v": ldf.wet_v_mask(state["active"], state["v_mask"]),
    }
    scores = {}
    first_divergence = None
    for name, direction, dump, key, line in OPERANDS:
        nemo = loaders._load_haloed(str(run / dump), jpi, jpj, hls)
        actual = values[key]
        nk = min(actual.shape[-1], nemo.shape[-1], masks[direction].shape[-1])
        wet = np.zeros_like(masks[direction][..., :nk], dtype=bool)
        wet[..., ldf.KLO:min(ldf.KHI, nk)] = masks[direction][..., ldf.KLO:min(ldf.KHI, nk)]
        score = sweep.metrics(actual[..., :nk], nemo[..., :nk], wet, focus, BAR)
        score.update({"direction": direction, "dump": dump,
                      "nemo_source": f"ldfslp.F90:{line}",
                      "localization_kind": (
                          "atomic_operand" if name not in (
                              "uslp_raw", "vslp_raw",
                              "uslp_postshapiro", "vslp_postshapiro")
                          else "composite_stage")})
        scores[name] = score
        if first_divergence is None and not score["pass"]:
            first_divergence = name
            break

    # Science-bar controls use the already verified zgrv(iik) operand. The
    # one-ULP arm is an exact-identity control because one ULP is below BAR.
    control_actual = values["zgrv_iik"][..., :35]
    control_nemo = loaders._load_haloed(
        str(run / "eiv_dump_zgrv_iik.bin"), jpi, jpj, hls)[..., :35]
    control_wet = np.zeros_like(masks["v"][..., :35], dtype=bool)
    control_wet[..., ldf.KLO:ldf.KHI] = masks["v"][..., ldf.KLO:ldf.KHI]
    science_controls = sweep.planted_controls(
        control_actual, control_nemo, control_wet, BAR)
    exact_before = int(np.count_nonzero(control_wet & (control_actual != control_nemo)))
    ulp = control_actual.copy()
    idx = tuple(np.argwhere(control_wet & (control_actual == control_nemo)
                            & (control_actual != 0.0))[0])
    ulp[idx] = np.nextafter(ulp[idx], np.inf)
    exact_after = int(np.count_nonzero(control_wet & (ulp != control_nemo)))
    science_controls["one_ulp_exact_identity_fired"] = exact_after == exact_before + 1
    science_controls["one_ulp_ji_k"] = [int(x) for x in idx]
    if not all(value for value in science_controls.values() if isinstance(value, bool)):
        raise SystemExit("row30 science control did not fire")

    source_text = args.nemo_source.read_text(errors="strict")
    source_quotes = {
        "gradient_u": (
            "zgru(ji,jj,iikm1) = umask(ji,jj,jk-1) * "
            "( prd(ji+1,jj  ,jk-1) - prd(ji,jj,jk-1) )"),
        "gradient_v": (
            "zgrv(ji,jj,iikm1) = vmask(ji,jj,jk-1) * "
            "( prd(ji  ,jj+1,jk-1) - prd(ji,jj,jk-1) )"),
        "metric_u": "zau = zgru(ji,jj,iik) * r1_e1u(ji,jj)",
        "metric_v": "zav = zgrv(ji,jj,iik) * r1_e2v(ji,jj)",
        "pre_bound": "zbu = 0.5_wp * ( zdzr(ji,jj) + zdzr(ji+1,jj  ) )",
        "pre_bound_v": "zbv = 0.5_wp * ( zdzr(ji,jj) + zdzr(ji  ,jj+1) )",
        "bound": (
            "zbu = MIN(  zbu, - z1_slpmax * ABS( zau ) , "
            "-7.e+3_wp/e3u(ji,jj,jk,Kmm)* ABS( zau )  )"),
        "bound_v": (
            "zbv = MIN(  zbv, - z1_slpmax * ABS( zav ) , "
            "-7.e+3_wp/e3v(ji,jj,jk,Kmm)* ABS( zav )  )"),
        "raw": (
            "zwz(ji,jj) = ( zfi * zau / ( zbu - zeps ) + "
            "( 1._wp - zfi ) * zdepu * zuslp_hml(ji,jj) ) * umask(ji,jj,jk)"),
        "raw_v": (
            "zww(ji,jj) = ( zfj * zav / ( zbv - zeps ) + "
            "( 1._wp - zfj ) * zdepv * zvslp_hml(ji,jj) ) * vmask(ji,jj,jk)"),
        "shapiro_u": "uslp(ji,jj,jk) = z1_16 *",
        "shapiro_v": "vslp(ji,jj,jk) = z1_16 *",
    }
    missing_quotes = [name for name, quote in source_quotes.items() if quote not in source_text]
    if missing_quotes:
        raise SystemExit(f"active ldfslp source quote changed: {missing_quotes}")

    artifact = {
        "schema": "dino-zdf-row30-uv-operands-v1",
        "lane": dump_lane.banner(),
        "disposition": "VERIFIED" if first_divergence is None else "DIVERGED",
        "first_divergence": first_divergence,
        "first_divergent_stage": first_divergence,
        "bar": BAR,
        "focus_columns_ji": [list(x) for x in focus],
        "operands_in_source_order": scores,
        "bracket": {
            "on_streams": len(on_names), "off_streams": len(off_names),
            "shared_streams_exact": len(off_names), "controls": controls,
            "registered_inputs_exact": True,
            "registered_input_names": list(registered_inputs),
        },
        "controls": science_controls,
        "capture": capture,
        "source_quotes": source_quotes,
        "repo_sha": args.expected_repo_sha,
        "provenance_sha256": {
            str(path): sha256(path) for path in sorted({
                args.nemo_source.resolve(), args.bracket_nemo_source.resolve(),
                args.nemo_binary.resolve(), args.bracket_nemo_binary.resolve(),
                args.instrument_patch.resolve(), args.dump_sha_manifest.resolve(),
                args.on_source_manifest.resolve(), args.off_source_manifest.resolve(),
                args.mld_maps.resolve(), run / "mesh_mask.nc",
                run / dump_lane.RESTART, run / "DINO_00005761_restart.nc",
                off / "DINO_00005761_restart.nc", Path(__file__).resolve(),
                Path(ldf.__file__).resolve(),
                run / ".nemo_binary_sha256", off / ".nemo_binary_sha256",
                run / "run.attempt1.log", off / "run.attempt1.log",
                run / "ocean.output", off / "ocean.output",
                run / "domain_cfg_out.nc", off / "domain_cfg_out.nc",
                *(run / name for name in registered_inputs),
                *(off / name for name in registered_inputs),
                *(run / name for name in NEW_DUMPS + EXISTING_DUMPS),
            })
        },
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for name, _, _, _, _ in OPERANDS:
        score = scores[name]
        print(f"{name}: {score['n_diverged_columns']}/{score['n_wet_columns']} "
              f"fail bar {BAR:.1e}; focus_fail="
              f"{sum(not item['pass'] for item in score['focus'])}")
        if name == first_divergence:
            print(f"FIRST DIVERGENT STAGE: {name} ({score['nemo_source']})")
            break
    if first_divergence is None:
        print("row30 VERIFIED: complete U/V operand ladder 0 failures")
    print(f"wrote {args.output}")
    return 0 if first_divergence is None else 30


if __name__ == "__main__":
    raise SystemExit(main())
