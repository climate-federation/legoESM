#!/usr/bin/env python
"""Score the preregistered day-180 ZDF row-21 coefficient operands.

The receipt is fail-closed on the write-only ON/OFF bracket.  Numerical
comparisons use the real legoESM TKE coefficient implementation with exact
NEMO post-row-20 ``en``/``zmxlm``/``zmxld`` substitutions.  The three scalar
intermediates retain NEMO's literal source association around that production
call; they are not reconstructed from NEMO outputs.
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
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import bn2_alpha_compare as loaders
import jax
import jax.numpy as jnp
import netCDF4  # noqa: N813
import numpy as np
import zdf_chain_sweep as sweep
from kamm_twin_90d import _build_twin_state
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.time_levels import time_level_for_dump
from legoesm.ocean.physics.vertical_mixing.tke import compute_K_from_tke
from zdf_stream_bracket import (
    files_byte_identical,
    manifest_sha256,
    one_bit_file_control,
    stream_manifest,
)

BAR = 1.0e-15
EXPECTED_SIZE = 2_980_224
EXPECTED_ON_STREAMS = 202
EXPECTED_OFF_STREAMS = 197
RESTART_SHA256 = "33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c"
FIELDS = (
    ("zsqen_base", "sqrt(en)", "m s-1"),
    ("zav_base", "rn_ediff*zmxlm*sqrt(en)", "m2 s-1"),
    ("avm_base", "max(zav,avmb)*wmask", "m2 s-1"),
    ("avt_base", "max(zav,avtb)*wmask", "m2 s-1"),
    ("dissl_postavn", "sqrt(en)/zmxld", "s-1"),
)
NEW_STREAMS = {f"tke_dump_{name}.bin" for name, _, _ in FIELDS}


def file_sha256(path: Path) -> str:
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


def absolute_metrics(
    lego: np.ndarray,
    nemo: np.ndarray,
    wet: np.ndarray,
    focus: list[tuple[int, int]],
    units: str,
) -> dict:
    registered = np.asarray(wet, dtype=bool)
    nonfinite = registered & (~np.isfinite(lego) | ~np.isfinite(nemo))
    finite = registered & ~nonfinite
    error = np.where(finite, np.abs(lego - nemo), 0.0)
    column = np.where(np.any(registered, axis=-1), np.max(error, axis=-1), np.nan)
    wet_columns = np.any(registered, axis=-1)
    bad = wet_columns & ((column > BAR) | np.any(nonfinite, axis=-1))
    focus_rows = [
        {
            "j": j,
            "i": i,
            "wet": bool(wet_columns[j, i]),
            "column_error": float(column[j, i]) if wet_columns[j, i] else None,
            "pass": bool(wet_columns[j, i] and not bad[j, i]),
        }
        for j, i in focus
    ]
    return {
        "units": units,
        "absolute_column_bar": BAR,
        "n_wet_elements": int(registered.sum()),
        "n_nonfinite_wet_elements": int(nonfinite.sum()),
        "n_wet_columns": int(wet_columns.sum()),
        "n_diverged_columns": int(bad.sum()),
        "n_verified_columns": int(wet_columns.sum() - bad.sum()),
        "max_column_error": float(np.nanmax(column)),
        "worst_column_ji": [int(x) for x in np.unravel_index(np.nanargmax(column), column.shape)],
        "exact_unequal_wet_elements": int(np.count_nonzero(registered & ~(lego == nemo))),
        "focus": focus_rows,
        "pass": bool(not bad.any()),
    }


def inventory_is_exact(on_names: set[str], off_names: set[str]) -> bool:
    return (
        len(on_names) == EXPECTED_ON_STREAMS
        and len(off_names) == EXPECTED_OFF_STREAMS
        and on_names - off_names == NEW_STREAMS
        and not off_names - on_names
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--bracket-dir", type=Path, required=True)
    ap.add_argument("--mld-maps", type=Path, required=True)
    ap.add_argument("--nemo-source", type=Path, required=True)
    ap.add_argument("--bracket-nemo-source", type=Path, required=True)
    ap.add_argument("--nemo-binary", type=Path, required=True)
    ap.add_argument("--bracket-nemo-binary", type=Path, required=True)
    ap.add_argument("--build-binary-receipt", type=Path, required=True)
    ap.add_argument("--on-source-manifest", type=Path, required=True)
    ap.add_argument("--off-source-manifest", type=Path, required=True)
    ap.add_argument("--instrument-patch", type=Path, required=True)
    ap.add_argument("--expected-repo-sha", required=True)
    ap.add_argument("--expected-source-sha", required=True)
    ap.add_argument("--expected-bracket-source-sha", required=True)
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
        raise SystemExit("row-21 receipt requires CPU fp64")
    if git_sha() != args.expected_repo_sha or not tracked_tree_clean():
        raise SystemExit("row-21 receipt requires the registered clean repository HEAD")

    run = args.run_dir.resolve()
    bracket = args.bracket_dir.resolve()
    expected_paths = {
        args.nemo_source: args.expected_source_sha,
        args.bracket_nemo_source: args.expected_bracket_source_sha,
        args.nemo_binary: args.expected_binary_sha,
        args.bracket_nemo_binary: args.expected_bracket_binary_sha,
        args.build_binary_receipt: args.expected_build_receipt_sha,
        args.on_source_manifest: args.expected_on_manifest_sha,
        args.off_source_manifest: args.expected_off_manifest_sha,
        args.instrument_patch: args.expected_patch_sha,
    }
    for path, expected in expected_paths.items():
        if file_sha256(path.resolve()) != expected:
            raise SystemExit(f"SHA receipt changed: {path}")
    if args.nemo_binary.stat().st_mtime_ns < args.nemo_source.stat().st_mtime_ns:
        raise SystemExit("instrumented binary is older than its source")

    def receipt_binds(path: Path, expected: str) -> bool:
        fields = path.read_text(errors="strict").split()
        return bool(fields and fields[0] == expected)

    if not receipt_binds(args.build_binary_receipt, args.expected_binary_sha):
        raise SystemExit("build receipt does not bind the executed ON binary")
    if not receipt_binds(run / ".nemo_binary_sha256", args.expected_binary_sha):
        raise SystemExit("ON run receipt does not bind the instrumented binary")
    if not receipt_binds(bracket / ".nemo_binary_sha256", args.expected_bracket_binary_sha):
        raise SystemExit("OFF run receipt does not bind the control binary")
    if file_sha256(run / "nemo") != args.expected_binary_sha:
        raise SystemExit("ON run executable changed")
    if file_sha256(bracket / "nemo") != args.expected_bracket_binary_sha:
        raise SystemExit("OFF run executable changed")

    restart = "DINO_00005761_restart.nc"
    if (
        file_sha256(run / restart) != RESTART_SHA256
        or file_sha256(bracket / restart) != RESTART_SHA256
    ):
        raise SystemExit("row-21 behavior-neutral restart bracket failed")
    for directory in (run, bracket):
        log = (directory / "run.attempt1.log").read_text(errors="strict")
        if not log.rstrip().endswith("STOP 0"):
            raise SystemExit(f"NEMO run did not stop cleanly: {directory}")

    on_manifest = stream_manifest(run)
    off_manifest = stream_manifest(bracket)
    on_names, off_names = set(on_manifest), set(off_manifest)
    if not inventory_is_exact(on_names, off_names):
        raise SystemExit(
            f"row-21 inventory changed: ON={len(on_names)} OFF={len(off_names)} "
            f"new={sorted(on_names-off_names)} missing={sorted(off_names-on_names)}"
        )
    unequal = [
        name for name in sorted(off_names)
        if not files_byte_identical(run / name, bracket / name)
    ]
    if unequal:
        raise SystemExit(f"row-21 shared-stream bracket failed: {unequal}")
    shared_name = sorted(off_names)[0]
    one_bit_control = one_bit_file_control(run / shared_name)
    missing_control = not inventory_is_exact(on_names - {shared_name}, off_names)
    if not (one_bit_control and missing_control):
        raise SystemExit("row-21 bracket controls did not fire")

    expected_dump_shas = {
        "tke_dump_zsqen_base.bin": (
            "4b84052d755fe2276982fa6ac76ea468258153d4cf21b8f267089d0cd761dc05"
        ),
        "tke_dump_zav_base.bin": "3ca117b3cfd2760e4b680cb7cffc782e2654dc5e2414b46bbe1be2ab3ffeadff",
        "tke_dump_avm_base.bin": "1240ccb86e674edd1309831e2292a244aaabc3bca2f245b530fd3b88b59c275f",
        "tke_dump_avt_base.bin": "a70d41988125d9097d83575793d1fec2e08d6f7b44cbdb25d4a7baa8b0029ddc",
        "tke_dump_dissl_postavn.bin": (
            "b99936aa66b0a3726b9ed44657488f19215e4e7d049a6091591b71947e7f7e42"
        ),
    }
    for name, expected in expected_dump_shas.items():
        path = run / name
        if path.stat().st_size != EXPECTED_SIZE or file_sha256(path) != expected:
            raise SystemExit(f"registered row-21 dump changed: {name}")
        time_level_for_dump(name)

    source = args.nemo_source.read_text(errors="strict")
    literals = (
        "zsqen = SQRT( en(ji,jj,jk) )",
        "zav   = rn_ediff * zmxlm(ji,jk) * zsqen",
        "p_avm(ji,jj,jk) = MAX( zav,                  avmb(jk) ) * wmask(ji,jj,jk)",
        "p_avt(ji,jj,jk) = MAX( zav, avtb_2d(ji,jj) * avtb(jk) ) * wmask(ji,jj,jk)",
        "dissl(ji,jj,jk) = zsqen / zmxld(ji,jk)",
    )
    if any(literal not in source for literal in literals):
        raise SystemExit("active row-21 NEMO source expressions changed")

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
        raise SystemExit("row-21 wet-column census changed")

    en = interior("tke_dump_en.bin")[..., 1:jpk]
    zmxlm = interior("tke_dump_zmxlm.bin")[..., 1:jpk]
    zmxld = interior("tke_dump_zmxld.bin")[..., 1:jpk]
    rn2 = interior("tke_dump_rn2.bin")[..., 1:jpk]
    rn2b = interior("tke_dump_rn2b.bin")[..., 1:jpk]
    sh2 = interior("tke_dump_sh2.bin")[..., 1:jpk]
    avm_in = interior("tke_dump_avm_in.bin")[..., 1:jpk]

    _, _, mc, model, _, surface_forcing, state = _build_twin_state(
        "nemo_dino_kamm_mlf", str(run), str(run), bridge_tke=True,
        bridge_before=True, restart_file="DINO_00005760_restart.nc", e3t_mode="both",
    )
    _, _, capture = sweep.capture_face_sh2_call(model, state, surface_forcing)
    mxl_args, mxl_kwargs, _ = capture["mxl_calls"][-1]
    cfg = mc.physics.vertical_mixing.tke
    if mxl_args[3] != cfg or tuple(np.asarray(mxl_args[0]).shape) != en.shape:
        raise SystemExit("production row-21 TKE call contract changed")
    if cfg.kappa_convention != "veros_sqrte" or cfg.prandtl_mode != "nemo_ri":
        raise SystemExit("DINO row-21 coefficient card changed")

    # Exact NEMO row-20 substitution into legoESM's real coefficient helper.
    # Pr=1 exposes the base pre-Prandtl tracer coefficient while retaining the
    # production amplitude and independent avm/avt floors.
    base_cfg = cfg._replace(prandtl_mode="constant", Prandtl_tke0=1.0)
    base_avm, base_avt = jax.jit(
        lambda e, lk, n2_, sh2_: compute_K_from_tke(
            e, lk, base_cfg, N2=n2_, shear_sq=sh2_)
    )(jnp.asarray(en), jnp.asarray(zmxlm), jnp.asarray(rn2), jnp.asarray(sh2))
    # The actual card is also called: its avm must be the same base assembly;
    # its final avt is row 22 and is scored by the ordered tail probe.
    final_avm, _ = jax.jit(
        lambda e, lk, n2_, n2b_, sh2_, avm_: compute_K_from_tke(
            e, lk, cfg, N2=n2_, shear_sq=jnp.ones_like(sh2_),
            N2_prandtl=n2b_, p_sh2_override=lambda _: sh2_,
            prandtl_K_M=avm_)
    )(
        jnp.asarray(en), jnp.asarray(zmxlm), jnp.asarray(rn2),
        jnp.asarray(rn2b), jnp.asarray(sh2), jnp.asarray(avm_in),
    )
    if not np.array_equal(np.asarray(final_avm), np.asarray(base_avm)):
        raise SystemExit("actual-card avm differs from the production base assembly")

    # Literal scalar intermediates surrounding the production coefficient
    # call.  These use legoESM production inputs/config, not NEMO outputs.
    zsqen = jax.jit(jnp.sqrt)(jnp.asarray(en))
    zav = jax.jit(lambda lk, sq: cfg.c_k * lk * sq)(jnp.asarray(zmxlm), zsqen)
    dissl = jax.jit(lambda sq, le: sq / le)(zsqen, jnp.asarray(zmxld))
    lego_fields = {
        "zsqen_base": np.asarray(zsqen),
        "zav_base": np.asarray(zav),
        "avm_base": np.asarray(base_avm),
        "avt_base": np.asarray(base_avt),
        "dissl_postavn": np.asarray(dissl),
    }

    rows = {}
    controls = {}
    first_divergence = None
    for name, expression, units in FIELDS:
        nemo = interior(f"tke_dump_{name}.bin")[..., 1:jpk]
        metric = absolute_metrics(lego_fields[name], nemo, wet, focus, units)
        rows[name] = {"expression": expression, "metric": metric}
        planted = nemo.copy()
        idx = tuple(int(x) for x in np.argwhere(wet)[0])
        planted[idx] += max(np.float64(2.0e-15), np.float64(4.0) * np.spacing(planted[idx]))
        value_fired = not absolute_metrics(planted, nemo, wet, focus, units)["pass"]
        roll_fired = not absolute_metrics(np.roll(nemo, 1, axis=1), nemo, wet, focus, units)["pass"]
        nonfinite = nemo.copy()
        nonfinite[idx] = np.nan
        nonfinite_fired = not absolute_metrics(nonfinite, nemo, wet, focus, units)["pass"]
        controls[name] = {
            "value_plant_fired": value_fired,
            "one_i_roll_fired": roll_fired,
            "nonfinite_fired": nonfinite_fired,
        }
        if not all(controls[name].values()):
            raise SystemExit(f"row-21 numeric control did not fire: {name}")
        if first_divergence is None and not metric["pass"]:
            first_divergence = name

    disposition = "VERIFIED" if first_divergence is None else "DIVERGED"
    consumed = {
        "run_log": run / "run.attempt1.log",
        "off_run_log": bracket / "run.attempt1.log",
        "run_restart": run / restart,
        "off_restart": bracket / restart,
        "run_binary_receipt": run / ".nemo_binary_sha256",
        "off_binary_receipt": bracket / ".nemo_binary_sha256",
        "mld_maps": args.mld_maps.resolve(),
        "nemo_source": args.nemo_source.resolve(),
        "bracket_nemo_source": args.bracket_nemo_source.resolve(),
        "nemo_binary": args.nemo_binary.resolve(),
        "bracket_nemo_binary": args.bracket_nemo_binary.resolve(),
        "build_binary_receipt": args.build_binary_receipt.resolve(),
        "on_source_manifest": args.on_source_manifest.resolve(),
        "off_source_manifest": args.off_source_manifest.resolve(),
        "instrument_patch": args.instrument_patch.resolve(),
        "probe": Path(__file__).resolve(),
    }
    for name, _, _ in FIELDS:
        consumed[f"dump_{name}"] = run / f"tke_dump_{name}.bin"
    artifact = {
        "schema": "dino-zdf-row21-coeff-assembly-v1",
        "repo_sha": git_sha(),
        "disposition": disposition,
        "first_divergence": first_divergence,
        "bar_absolute": BAR,
        "focus_columns_ji": [list(x) for x in focus],
        "wet_columns": 9920,
        "rows": rows,
        "controls": controls,
        "bracket_controls": {
            "one_bit_shared_stream_fired": one_bit_control,
            "missing_stream_inventory_fired": missing_control,
        },
        "bracket": {
            "restart_sha256": RESTART_SHA256,
            "on_stream_count": len(on_names),
            "off_stream_count": len(off_names),
            "shared_stream_count": len(off_names),
            "strict_byte_identical_shared_stream_count": len(off_names),
            "new_streams": sorted(NEW_STREAMS),
            "on_stream_manifest_sha256": manifest_sha256(on_manifest),
            "off_stream_manifest_sha256": manifest_sha256(off_manifest),
            "on_stream_manifest": on_manifest,
            "off_stream_manifest": off_manifest,
            "exclusions": [],
        },
        "nemo_lines": {
            "baseline": "cfgs/DINO/MY_SRC/zdftke.F90:913-925",
            "instrumented_expressions": "cfgs/DINO/MY_SRC/zdftke.F90:924-928",
            "instrumented_captures": "cfgs/DINO/MY_SRC/zdftke.F90:930-934",
        },
        "provenance_sha256": {key: file_sha256(path) for key, path in consumed.items()},
        "provenance_paths": {key: str(path) for key, path in consumed.items()},
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    counts = ", ".join(
        f"{name}={rows[name]['metric']['n_diverged_columns']}/9920"
        for name, _, _ in FIELDS
    )
    print(f"row21 {disposition}: {counts}; first_divergence={first_divergence}")
    return 0 if disposition == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
