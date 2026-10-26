#!/usr/bin/env python
"""Score the held row-30 raw-U write-only operand ladder in source order."""
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
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bn2_alpha_compare as loaders  # noqa: E402
import dump_lane  # noqa: E402
import ldf_slp_per_element as ldf  # noqa: E402
import zdf_stream_bracket as bracket_gate  # noqa: E402
import zdf_chain_sweep as sweep  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402

BAR = 1.0e-15
DUMPS = (
    "eiv_dump_iku.bin", "eiv_dump_zfi.bin", "eiv_dump_e3u_miku.bin",
    "eiv_dump_zdepu.bin", "eiv_dump_zuslp_hml_pre.bin",
    "eiv_dump_sint_u.bin", "eiv_dump_mlterm_u.bin",
    "eiv_dump_blend_u.bin",
)
EXPECTED_SIZE = 3_273_984
MESH_SHA = "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622"
INPUT_RESTART_SHA = "0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e"
OUTPUT_RESTART_SHA = "33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c"
NAMELIST_SHA = "55f17d2344e5aaa58f7c6ef23e7d5dfebd9c351888c6499d5d312131b8515355"


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
    parser.add_argument("--dump-sha-manifest", type=Path, required=True)
    parser.add_argument("--bracket-receipt", type=Path, required=True)
    parser.add_argument("--expected-bracket-sha", required=True)
    parser.add_argument("--nemo-source", type=Path, required=True)
    parser.add_argument("--nemo-binary", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--expected-binary-sha", required=True)
    parser.add_argument("--expected-repo-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row30 held scorer requires CPU fp64")
    git = lambda *a: subprocess.check_output(["git", *a], text=True).strip()
    if git("rev-parse", "HEAD") != args.expected_repo_sha or git(
            "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("row30 held scorer requires registered clean HEAD")
    run = args.run_dir.resolve()
    if Path(dump_lane.RUN_DIR).resolve() != run:
        raise SystemExit("lane selector does not resolve to held run")
    manifest = json.loads(args.dump_sha_manifest.read_text(errors="strict"))
    if set(manifest) != set(DUMPS):
        raise SystemExit("dump SHA manifest has missing or extra names")
    if sha256(args.bracket_receipt) != args.expected_bracket_sha:
        raise SystemExit("registered bracket receipt SHA changed")
    bracket = json.loads(args.bracket_receipt.read_text(errors="strict"))
    if bracket.get("schema") != "dino-zdf-row30-uslp-bracket-v1":
        raise SystemExit("wrong bracket receipt schema")
    if Path(bracket.get("on_dir", "")).resolve() != run:
        raise SystemExit("bracket receipt belongs to a different ON run")
    off = Path(bracket.get("off_dir", "")).resolve()
    if bracket.get("shared_count") != 197 or not bracket.get("shared_exact"):
        raise SystemExit("bracket receipt did not certify 197/197 streams")
    controls = bracket.get("controls", {})
    if controls != {"missing_stream": True, "one_bit_file": True}:
        raise SystemExit("bracket controls missing or did not fire")
    on_manifest = bracket_gate.stream_manifest(run)
    off_manifest = bracket_gate.stream_manifest(off)
    shared = {name: on_manifest[name] for name in sorted(off_manifest)}
    if (len(on_manifest) != 217 or len(off_manifest) != 197
            or set(off_manifest) - set(on_manifest)
            or any(on_manifest[name] != off_manifest[name]
                   for name in off_manifest)
            or bracket_gate.manifest_sha256(shared)
            != bracket.get("shared_manifest_sha256")):
        raise SystemExit("current streams do not reproduce bracket receipt")
    for role, directory in (("on", run), ("off", off)):
        recorded = bracket.get(f"{role}_sha256", {})
        required = {
            "nemo", "mesh_mask.nc", "DINO_00005760_restart.nc",
            "DINO_00005761_restart.nc", "namelist_cfg", "run.attempt1.log",
        }
        if set(recorded) != required:
            raise SystemExit(f"{role} bracket provenance incomplete")
        for name, digest in recorded.items():
            if sha256(directory / name) != digest:
                raise SystemExit(f"{role} bracket input changed: {name}")
    if bracket["on_sha256"]["nemo"] != args.expected_binary_sha:
        raise SystemExit("bracket ON binary differs from registered binary")
    bound = {
        args.nemo_source: args.expected_source_sha,
        args.nemo_binary: args.expected_binary_sha,
        run / "nemo": args.expected_binary_sha,
        args.mld_maps: "9fb7344d1e6f92232d211f6a52ff8636022f0d9b0b05acea2b0bae6e6afd9bf0",
        run / "mesh_mask.nc": MESH_SHA,
        run / "DINO_00005760_restart.nc": INPUT_RESTART_SHA,
        run / "DINO_00005761_restart.nc": OUTPUT_RESTART_SHA,
        run / "namelist_cfg": NAMELIST_SHA,
        run / "eiv_dump_zau.bin": "78f5d2159baad245f6c2fbb29f4cfbc24bc2724cfbd8b4077953e5b3576398b1",
        args.bracket_receipt: args.expected_bracket_sha,
        **{run / name: digest for name, digest in manifest.items()},
    }
    for path, expected in bound.items():
        path = path.resolve()
        if sha256(path) != expected:
            raise SystemExit(f"SHA receipt changed: {path}")
    for name in DUMPS:
        if (run / name).stat().st_size != EXPECTED_SIZE:
            raise SystemExit(f"wrong dump size: {name}")
    source = args.nemo_source.read_text(errors="strict")
    for quote in ("iku_dump(ji,jj,jk) = REAL( iku, wp )",
                  "zdepu_dump(ji,jj,jk) = zdepu",
                  "blend_u_dump(ji,jj,jk) = zfi * zau / ( zbu - zeps )"):
        if quote not in source:
            raise SystemExit(f"instrumented source quote changed: {quote}")

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
        for name in DUMPS
    }
    nk = min(raw.shape[-1], *(value.shape[-1] for value in nemo.values()))
    wet = np.zeros_like(state["active"][..., :nk], dtype=bool)
    full = ldf.wet_u_mask(
        np.asarray(state["active"]), state["u_mask"])[..., :nk]
    wet[..., ldf.KLO:min(ldf.KHI, nk)] = full[..., ldf.KLO:min(ldf.KHI, nk)]
    focus = sweep.focus_from_maps(args.mld_maps)
    in_ml = np.asarray(u_loc["in_ml"])[..., :nk]
    anchor = np.asarray(u_loc["anchor"])[..., None]
    current_zdep = np.asarray(u_loc["zdep_face"])[..., :nk]
    sint = np.asarray(u_loc["s_int"])[..., :nk]
    e3_surface = np.broadcast_to(
        np.asarray(outer["e3u_k"])[..., :1], raw.shape)[..., :nk]
    values = (
        ("iku", np.broadcast_to(
            (np.asarray(outer["iku"]) + 1)[..., None], raw.shape)[..., :nk]),
        ("zfi", (~in_ml).astype(np.float64)),
        ("e3u_miku", e3_surface),
        ("zdepu", current_zdep),
        ("zuslp_hml_pre", np.where(in_ml, anchor, 0.0)),
        ("sint_u", sint),
        ("mlterm_u", np.where(in_ml, current_zdep * anchor, 0.0)),
        # Literal ldfslp.F90 association: (zfi*zau)/(zbu-zeps) first,
        # then ((1-zfi)*zdepu)*anchor, then the final add.  Do not replace
        # this by np.where while peeling ULP residuals.
        ("blend_u", None),
    )
    zfi = (~in_ml).astype(np.float64)
    zau = np.asarray(u_loc["zau"])[..., :nk]
    zbu = np.asarray(u_loc["zbu"])[..., :nk]
    zeps = np.float64(1.0e-20)
    first_term = (zfi * zau) / (zbu - zeps)
    second_term = ((np.float64(1.0) - zfi) * current_zdep) * anchor
    literal_blend = first_term + second_term
    values = tuple((stage, literal_blend if stage == "blend_u" else value)
                   for stage, value in values)
    scores = {}
    first = None
    for stage, actual in values:
        expected = nemo[f"eiv_dump_{stage}.bin"][..., :nk]
        score = sweep.metrics(actual, expected, wet, focus, BAR)
        scores[stage] = score
        focus_fail = sum(not item["pass"] for item in score["focus"])
        print(f"{stage}: {score['n_diverged_columns']}/{score['n_wet_columns']} "
              f"fail bar {BAR:.1e}; focus_fail={focus_fail}")
        if not score["pass"]:
            first = stage
            break
    def red(label, planted, stage):
        expected = nemo[f"eiv_dump_{stage}.bin"][..., :nk]
        baseline = sweep.metrics(expected, expected, wet, focus, BAR)
        if not baseline["pass"]:
            raise SystemExit(f"control baseline is not exact: {label}")
        outcome = sweep.metrics(
            planted, expected, wet, focus, BAR)
        if outcome["pass"]:
            raise SystemExit(f"new-ladder control did not fire: {label}")
        return True

    first_wet = tuple(np.argwhere(wet)[0])
    iku_one = np.array(nemo["eiv_dump_iku.bin"][..., :nk], copy=True)
    iku_one[first_wet] += np.float64(1.0)
    zfi_shift = np.roll(nemo["eiv_dump_zfi.bin"][..., :nk], 1, axis=-1)
    e3u_roll = np.roll(
        nemo["eiv_dump_e3u_miku.bin"][..., :nk], 1, axis=1)
    e3u_level = np.broadcast_to(
        np.asarray(outer["e3u_k"])[..., 1:2], raw.shape)[..., :nk]
    zdepu_roll = np.roll(
        nemo["eiv_dump_zdepu.bin"][..., :nk], 1, axis=1)
    wet_nan = np.array(nemo["eiv_dump_iku.bin"][..., :nk], copy=True)
    wet_nan[first_wet] = np.nan
    controls = {
        "iku_plus_one_at_wet": red("iku_plus_one_at_wet", iku_one, "iku"),
        "zfi_vertical_shift": red("zfi_vertical_shift", zfi_shift, "zfi"),
        "e3u_zonal_roll": red("e3u_zonal_roll", e3u_roll, "e3u_miku"),
        "e3u_wrong_level": red("e3u_wrong_level", e3u_level, "e3u_miku"),
        "zdepu_zonal_roll": red("zdepu_zonal_roll", zdepu_roll, "zdepu"),
        "first_stage_wet_nan": red("first_stage_wet_nan", wet_nan, "iku"),
    }
    artifact = {
        "schema": "dino-zdf-row30-uslp-held-operands-v1",
        "disposition": "DIVERGED" if first else "VERIFIED",
        "first_divergence": first,
        "bar": BAR,
        "focus_columns_ji": [list(x) for x in focus],
        "scores": scores,
        "controls": controls,
        "repo_sha": args.expected_repo_sha,
        "provenance_sha256": {
            str(path.resolve()): sha256(path.resolve())
            for path in (*bound.keys(), args.dump_sha_manifest, Path(__file__), Path(ldf.__file__))
        },
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"FIRST DIVERGENT STAGE: {first or 'none'}")
    return 30 if first else 0


if __name__ == "__main__":
    raise SystemExit(main())
