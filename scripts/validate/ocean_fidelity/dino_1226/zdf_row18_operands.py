#!/usr/bin/env python
"""Row-18 ``-gdepw(Kmm)/htau`` operand and substitution receipt.

This consumes the two write-only streams preregistered in
``PREREG_zdf_chain_sweep_round18_operands.md``.  It captures the production
``nemo_etau_injection`` call, substitutes one NEMO operand at a time, and
scores both the exponent argument and the complete literal row-18 update.
"""

from __future__ import annotations

import argparse
import ctypes
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
import netCDF4  # noqa: N813
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bn2_alpha_compare as loaders  # noqa: E402
import dump_lane  # noqa: E402
import zdf_chain_sweep as sweep  # noqa: E402
from kamm_twin_90d import _build_twin_state  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.fidelity.time_levels import time_level_for_dump  # noqa: E402
from legoesm.ocean.physics.vertical_mixing import tke as tke_mod  # noqa: E402

POINTWISE = 1.0e-15
EXPECTED_RESTART_SHA = "33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c"
EXPECTED_GDEPW_SHA = "fc601f5a4c0a9715245189fa10e9f354f0ad204f87bb5e5c31b5859107a01c3d"
EXPECTED_HTAU_SHA = "3b1e2574a9ea37deb500f9f1a94a56d71b428f9be5b5558dfdf842df1eedfa0d"
EXPECTED_SIZE = 2_980_224


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def first_difference(left: Path, right: Path) -> int | None:
    with left.open("rb") as a, right.open("rb") as b:
        offset = 0
        while True:
            aa = a.read(1024 * 1024)
            bb = b.read(1024 * 1024)
            if aa == bb:
                if not aa:
                    return None
                offset += len(aa)
                continue
            for index, (x, y) in enumerate(zip(aa, bb, strict=False)):
                if x != y:
                    return offset + index + 1  # cmp's one-origin convention
            return offset + min(len(aa), len(bb)) + 1


def exact_control(reference: np.ndarray, wet: np.ndarray) -> dict:
    idx = tuple(int(x) for x in np.argwhere(wet)[0])
    planted = reference.copy()
    planted[idx] = np.nextafter(planted[idx], np.inf)
    unequal = wet & ~(planted == reference)
    if int(unequal.sum()) != 1 or not bool(unequal[idx]):
        raise AssertionError("+1 ULP exact-operand control did not fire once")
    return {
        "fired": True,
        "perturbed_ji_k": list(idx),
        "n_unequal_wet_elements": int(unequal.sum()),
        "perturbation": "+1 ULP",
    }


def exact_census(actual: np.ndarray, expected: np.ndarray,
                 wet: np.ndarray) -> dict:
    """Report an elementwise binary64 census over the registered wet mask."""
    actual_wet = np.asarray(actual, dtype=np.float64)[wet]
    expected_wet = np.asarray(expected, dtype=np.float64)[wet]
    unequal = actual_wet.view(np.uint64) != expected_wet.view(np.uint64)
    return {
        "n_wet_elements": int(actual_wet.size),
        "n_unequal_wet_elements": int(unequal.sum()),
        "max_abs_error": float(np.max(np.abs(actual_wet - expected_wet))),
    }


def negative_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Evaluate ``-numerator/denominator`` without touching dry zero slots."""
    result = np.zeros_like(numerator)
    np.divide(-numerator, denominator, out=result, where=denominator != 0.0)
    return result


def glibc_vector_sin(phase: np.ndarray, library_path: Path) -> np.ndarray:
    """Call the host's two-lane ``_ZGVbN2v_sin`` through its C ABI wrapper."""
    phase = np.ascontiguousarray(phase, dtype=np.float64)
    if phase.ndim != 2 or phase.shape[1] % 2:
        raise ValueError("vector-SIN phase must be 2-D with an even i extent")
    output = np.empty_like(phase)
    library = ctypes.CDLL(str(library_path))
    function = library.zdf_glibc_v2_sin_rows
    array_pointer = np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
    function.argtypes = [array_pointer, array_pointer,
                         ctypes.c_size_t, ctypes.c_size_t]
    function.restype = ctypes.c_int
    status = function(phase, output, phase.shape[0], phase.shape[1])
    if status != 0:
        raise RuntimeError(f"glibc vector-SIN wrapper returned {status}")
    return output


def glibc_version() -> str:
    libc = ctypes.CDLL("libc.so.6")
    libc.gnu_get_libc_version.restype = ctypes.c_char_p
    return libc.gnu_get_libc_version().decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bracket-off-dir", type=Path, required=True)
    parser.add_argument("--determinism-a-dir", type=Path, required=True)
    parser.add_argument("--determinism-b-dir", type=Path, required=True)
    parser.add_argument("--mld-maps", type=Path, required=True)
    parser.add_argument("--nemo-source", type=Path, required=True)
    parser.add_argument("--nemo-binary", type=Path, required=True)
    parser.add_argument("--vector-sin-library", type=Path, required=True)
    parser.add_argument("--vector-sin-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row18 operand receipt requires CPU and JAX fp64")
    run = args.run_dir.resolve()
    off = args.bracket_off_dir.resolve()
    if Path(dump_lane.RUN_DIR).resolve() != run:
        raise SystemExit(
            f"dump_lane resolves to {Path(dump_lane.RUN_DIR).resolve()}, not {run}"
        )
    if dump_lane.LANE != "d180" or dump_lane.KT_DUMP != 5761:
        raise SystemExit("only registered d180 kt=5761 is accepted")

    gdepw_path = run / "tke_dump_etau_gdepw.bin"
    htau_path = run / "tke_dump_etau_htau.bin"
    on_restart = run / "DINO_00005761_restart.nc"
    off_restart = off / "DINO_00005761_restart.nc"
    for path, expected_sha in (
        (gdepw_path, EXPECTED_GDEPW_SHA),
        (htau_path, EXPECTED_HTAU_SHA),
        (on_restart, EXPECTED_RESTART_SHA),
        (off_restart, EXPECTED_RESTART_SHA),
    ):
        if not path.is_file() or sha256(path) != expected_sha:
            raise AssertionError(f"provenance mismatch: {path}")
    if gdepw_path.stat().st_size != EXPECTED_SIZE or htau_path.stat().st_size != EXPECTED_SIZE:
        raise AssertionError("row18 operand dump size changed")
    if on_restart.read_bytes() != off_restart.read_bytes():
        raise AssertionError("row18 on/off restart container bytes differ")

    # The excluded cor2d diagnostic is independently red: two runs of the
    # identical unpatched executable diverge beginning at byte 3 while their
    # restart containers remain byte-identical.  This stream is unrelated to
    # ZDF and is not consumed below.
    cor_name = "cor2d_dump_zu_trd_substep1.bin"
    det_a = args.determinism_a_dir.resolve()
    det_b = args.determinism_b_dir.resolve()
    det_a_restart = det_a / "DINO_00005761_restart.nc"
    det_b_restart = det_b / "DINO_00005761_restart.nc"
    if (sha256(det_a_restart) != EXPECTED_RESTART_SHA
            or sha256(det_b_restart) != EXPECTED_RESTART_SHA):
        raise AssertionError("determinism-control restarts changed")
    cor_first_diff = first_difference(det_a / cor_name, det_b / cor_name)
    if cor_first_diff != 3:
        raise AssertionError(
            f"cor2d determinism control first difference {cor_first_diff}, expected 3"
        )

    # All shared TKE streams are the relevant write-only bracket.  Assert
    # their inventory and bytes; only the two new operands may be on-only.
    new_names = {gdepw_path.name, htau_path.name}
    off_tke = {p.name for p in off.glob("tke_dump_*.bin")}
    on_tke = {p.name for p in run.glob("tke_dump_*.bin")}
    if on_tke - off_tke != new_names or off_tke - on_tke:
        raise AssertionError("shared TKE stream inventory differs unexpectedly")
    shared_tke_differences = [
        name for name in sorted(off_tke)
        if sha256(off / name) != sha256(run / name)
    ]
    if shared_tke_differences:
        raise AssertionError(f"shared TKE streams differ: {shared_tke_differences}")

    focus = sweep.focus_from_maps(args.mld_maps)
    jpi, jpj, jpk, hls = loaders._read_dims(str(run))
    ni, nj = jpi - 2 * hls, jpj - 2 * hls
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        active = np.moveaxis(np.asarray(ds["tmask"][0]), 0, -1)
        nemo_lat_deg = np.asarray(ds["gphit"][0], dtype=np.float64)
    # This canonical mesh_mask is already the halo-free (199,52,36) field;
    # binary streams alone carry the 2-cell halo described by _read_dims.
    if active.shape[:2] != (nj, ni):
        raise AssertionError(
            f"mesh mask interior shape changed: {active.shape[:2]} != {(nj, ni)}"
        )
    active = active > 0.5
    wet = (active[..., :-1] & active[..., 1:])
    if int(np.any(wet, axis=-1).sum()) != 9920:
        raise AssertionError("row18 wet-column census changed")

    br, cfg, mc, model, forcing, sf, state = _build_twin_state(
        "nemo_dino_kamm_mlf", str(run), str(run), bridge_tke=True,
        bridge_before=True, restart_file="DINO_00005760_restart.nc",
        e3t_mode="both",
    )
    del br, cfg, forcing
    _, _, captured = sweep.capture_face_sh2_call(model, state, sf)
    if len(captured["etau_calls"]) != 1:
        raise AssertionError("production etau call count changed")
    etau_args, etau_kwargs, _ = captured["etau_calls"][0]
    if len(etau_args) != 4:
        raise AssertionError("production etau call contract changed")
    pre_l = np.asarray(etau_args[0])
    gdepw_l = np.asarray(etau_args[2])
    netau = min(gdepw_l.shape[-1], wet.shape[-1])
    wet = wet[..., :netau]

    def interior(name: str) -> np.ndarray:
        time_level_for_dump(name)
        return loaders._load_interior(str(run / name), ni, nj)

    gdepw_n = interior(gdepw_path.name)[..., 1:1 + netau]
    htau_n = interior(htau_path.name)[..., 1:1 + netau]
    argument_n = interior("tke_dump_etau_argument.bin")[..., 1:1 + netau]
    exp_n = interior("tke_dump_etau_exp.bin")[..., 1:1 + netau]
    post_n = interior("tke_dump_en.bin")[..., 1:1 + netau]

    # Score the production selector, not a duplicated legacy expression.  The
    # selected DINO card must receive the bridge-carried native gphit degrees.
    htau_l_2d = np.asarray(tke_mod._nemo_etau_htau(
        etau_kwargs["lat_deg"], etau_args[3], pre_l.dtype))
    htau_l = np.broadcast_to(htau_l_2d[..., None], gdepw_l.shape)[..., :netau]
    gdepw_l = gdepw_l[..., :netau]

    source_phase = (
        np.float64(np.pi) / np.float64(180.0)
        * np.asarray(etau_kwargs["lat_deg"], dtype=np.float64)
    )
    if nemo_lat_deg.shape != source_phase.shape:
        raise AssertionError(
            f"NEMO gphit shape {nemo_lat_deg.shape} != {source_phase.shape}")
    legacy_phase = np.asarray(
        jnp.deg2rad(etau_kwargs["lat_deg"]), dtype=np.float64)
    vector_sine = glibc_vector_sin(source_phase, args.vector_sin_library)

    def assemble_htau(sine_value: np.ndarray) -> np.ndarray:
        return np.maximum(
            tke_mod._NEMO_TKE_HTAU_MIN_M,
            np.minimum(
                tke_mod._NEMO_TKE_HTAU_MAX_M,
                tke_mod._NEMO_TKE_HTAU_SLOPE_M * np.abs(sine_value),
            ),
        )

    vector_htau_2d = assemble_htau(vector_sine)
    direct_phase = (
        np.float64(np.pi) / np.float64(180.0) * nemo_lat_deg)
    direct_jax_sine = np.asarray(jnp.sin(jnp.asarray(direct_phase)))
    direct_vector_sine = glibc_vector_sin(
        direct_phase, args.vector_sin_library)
    direct_jax_htau_2d = assemble_htau(direct_jax_sine)
    direct_vector_htau_2d = assemble_htau(direct_vector_sine)
    vector_htau = np.broadcast_to(
        vector_htau_2d[..., None], gdepw_l.shape)[..., :netau]
    direct_jax_htau = np.broadcast_to(
        direct_jax_htau_2d[..., None], gdepw_l.shape)[..., :netau]
    direct_vector_htau = np.broadcast_to(
        direct_vector_htau_2d[..., None], gdepw_l.shape)[..., :netau]
    wet_columns = np.any(wet, axis=-1)

    operands = {
        "gdepw_production_vs_nemo": sweep.metrics(
            gdepw_l, gdepw_n, wet, focus, POINTWISE),
        "htau_production_vs_nemo": sweep.metrics(
            htau_l, htau_n, wet, focus, POINTWISE),
    }
    operands["gdepw_production_vs_nemo"]["exact_census"] = exact_census(
        gdepw_l, gdepw_n, wet)
    operands["htau_production_vs_nemo"]["exact_census"] = exact_census(
        htau_l, htau_n, wet)
    arguments = {
        "production": negative_ratio(gdepw_l, htau_l),
        "substitute_gdepw": negative_ratio(gdepw_n, htau_l),
        "substitute_htau": negative_ratio(gdepw_l, htau_n),
        "substitute_both": negative_ratio(gdepw_n, htau_n),
        "vector_sin_htau": negative_ratio(gdepw_l, vector_htau),
        "direct_gphit_jax_htau": negative_ratio(gdepw_l, direct_jax_htau),
        "direct_gphit_vector_htau": negative_ratio(
            gdepw_l, direct_vector_htau),
    }

    rounded = tke_mod._nemo_binary64_round
    etau_cfg = etau_args[3]
    surface = jnp.maximum(
        tke_mod._NEMO_TKE_EMIN0,
        tke_mod._NEMO_TKE_EBB / etau_kwargs["rho_0"]
        * jnp.maximum(etau_args[1], 0.0),
    )
    ice = etau_kwargs["ice_frac"]

    def literal_row(argument: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        exp_value = tke_mod._nemo_glibc234_vector_exp(jnp.asarray(argument))
        increment = rounded(etau_cfg.etau_frac * surface[..., None])
        increment = rounded(increment * exp_value)
        if ice is not None:
            increment = rounded(
                increment * jnp.maximum(0.0, 1.0 - ice[..., None])
            )
        candidate = rounded(
            jnp.asarray(pre_l[..., :netau]) + increment[..., :netau]
        )
        return np.asarray(exp_value), np.asarray(candidate)

    arms = {}
    for name, argument in arguments.items():
        exp_value, candidate = literal_row(argument)
        arms[name] = {
            "argument": sweep.metrics(
                argument, argument_n, wet, focus, POINTWISE),
            "literal_exp": sweep.metrics(
                exp_value, exp_n, wet, focus, POINTWISE),
            "full_row": sweep.metrics(
                candidate, post_n, wet, focus, POINTWISE),
        }

    vector_htau_metric = sweep.metrics(
        vector_htau, htau_n, wet, focus, POINTWISE)
    vector_htau_exact = exact_census(vector_htau, htau_n, wet)
    direct_jax_exact = exact_census(direct_jax_htau, htau_n, wet)
    direct_vector_exact = exact_census(direct_vector_htau, htau_n, wet)
    vector_exact = vector_htau_exact["n_unequal_wet_elements"] == 0
    direct_jax_is_exact = direct_jax_exact["n_unequal_wet_elements"] == 0
    direct_vector_is_exact = (
        direct_vector_exact["n_unequal_wet_elements"] == 0)
    if direct_jax_is_exact and not vector_exact:
        htau_owner = "LATITUDE"
    elif vector_exact and not direct_jax_is_exact:
        htau_owner = "VECTOR-SIN"
    elif direct_vector_is_exact and not direct_jax_is_exact and not vector_exact:
        htau_owner = "INTERACTION"
    elif direct_vector_is_exact:
        htau_owner = "AMBIGUOUS"
    else:
        htau_owner = "NEXT-ASSEMBLY-OPERAND"

    singles = [
        name for name in ("substitute_gdepw", "substitute_htau")
        if arms[name]["full_row"]["pass"]
    ]
    production_checks = [
        operands["gdepw_production_vs_nemo"]["pass"],
        operands["htau_production_vs_nemo"]["pass"],
        arms["production"]["argument"]["pass"],
        arms["production"]["literal_exp"]["pass"],
        arms["production"]["full_row"]["pass"],
    ]
    if all(production_checks):
        disposition, owner = "VERIFIED", "none (production clears row)"
    elif singles == ["substitute_gdepw"]:
        disposition, owner = "VERIFIED-WITH-SUBSTITUTION", "gdepw(Kmm)"
    elif singles == ["substitute_htau"]:
        disposition, owner = "VERIFIED-WITH-SUBSTITUTION", "htau"
    elif len(singles) == 2:
        disposition, owner = "DIVERGED-AMBIGUOUS", "both single substitutions clear"
    elif arms["substitute_both"]["full_row"]["pass"]:
        disposition, owner = "DIVERGED-INTERACTION", "gdepw/htau interaction"
    else:
        disposition, owner = "DIVERGED-DOWNSTREAM", "neither operand substitution clears"

    roll = np.roll(gdepw_n, 1, axis=1)
    roll_metric = sweep.metrics(roll, gdepw_n, wet, [], POINTWISE)
    if roll_metric["pass"]:
        raise AssertionError("one-cell gdepw roll control did not fail")
    controls = {
        "gdepw_plus_one_ulp": exact_control(gdepw_n, wet),
        "htau_plus_one_ulp": exact_control(htau_n, wet),
        "gdepw_one_i_roll_fired": True,
        "literal_both_arm": sweep.planted_controls(
            np.asarray(literal_row(arguments["substitute_both"])[1]),
            post_n, wet, POINTWISE,
        ),
    }
    all_next_phase = np.nextafter(source_phase, np.inf)
    all_next_sine = glibc_vector_sin(
        all_next_phase, args.vector_sin_library)
    sensitive = (
        wet_columns
        & (all_next_sine.view(np.uint64) != vector_sine.view(np.uint64)))
    if not np.any(sensitive):
        raise AssertionError("no +1 ULP-sensitive wet vector-SIN phase exists")
    phase_planted = source_phase.copy()
    phase_index = tuple(int(x) for x in np.argwhere(sensitive)[0])
    phase_planted[phase_index] = np.nextafter(
        phase_planted[phase_index], np.inf)
    planted_sine = glibc_vector_sin(phase_planted, args.vector_sin_library)
    phase_control_mismatches = int(np.sum(
        wet_columns
        & (planted_sine.view(np.uint64) != vector_sine.view(np.uint64))))
    if phase_control_mismatches < 1:
        raise AssertionError("+1 ULP vector-SIN phase control did not fire")
    rolled_htau = np.roll(vector_htau, 1, axis=0)
    rolled_htau_metric = sweep.metrics(
        rolled_htau, htau_n, wet, [], POINTWISE)
    if rolled_htau_metric["pass"]:
        raise AssertionError("one-j-row vector-htau roll control did not fail")
    controls["vector_sin_phase_plus_one_ulp"] = {
        "fired": True,
        "perturbed_ji": list(phase_index),
        "n_changed_wet_sine_elements": phase_control_mismatches,
    }
    controls["vector_htau_one_j_roll_fired"] = True

    provenance_paths = [
        gdepw_path, htau_path, on_restart, off_restart,
        run / "tke_dump_etau_argument.bin", run / "tke_dump_etau_exp.bin",
        run / "tke_dump_en.bin", run / "tke_dump_en_postsolve.bin",
        run / "tke_dump_en_postlc.bin", run / "mesh_mask.nc",
        run / "ocean.output", args.mld_maps, args.nemo_source,
        args.nemo_binary, det_a / cor_name, det_b / cor_name,
        args.vector_sin_library, args.vector_sin_source,
        det_a_restart, det_b_restart,
    ]
    artifact = {
        "schema": "dino-zdf-row18-operands-v1",
        "repo_sha": git_sha(),
        "lane": dump_lane.banner(),
        "run_dir": str(run),
        "focus_columns_ji": [list(x) for x in focus],
        "wet_columns": 9920,
        "bar": POINTWISE,
        "disposition": disposition,
        "owner": owner,
        "vector_sin_discriminator": {
            "disposition": (
                "CONFIRM-HTAU-OPERANDS" if direct_vector_is_exact
                else "REFUTE-HTAU-OPERANDS"),
            "owner": htau_owner,
            "glibc_version": glibc_version(),
            "captured_latitude_vs_nemo_gphit_exact": exact_census(
                np.asarray(etau_kwargs["lat_deg"])[..., None],
                nemo_lat_deg[..., None], wet_columns[..., None]),
            "source_phase_vs_legacy_exact": exact_census(
                source_phase[..., None], legacy_phase[..., None],
                wet_columns[..., None]),
            "vector_sin_vs_jax_exact": exact_census(
                vector_sine[..., None],
                np.asarray(jnp.sin(jnp.asarray(legacy_phase)))[..., None],
                wet_columns[..., None]),
            "htau": vector_htau_metric,
            "htau_exact": vector_htau_exact,
            "full_row": arms["vector_sin_htau"]["full_row"],
            "direct_gphit_jax": {
                "htau_exact": direct_jax_exact,
                "htau": sweep.metrics(
                    direct_jax_htau, htau_n, wet, focus, POINTWISE),
                "full_row": arms["direct_gphit_jax_htau"]["full_row"],
            },
            "direct_gphit_vector": {
                "htau_exact": direct_vector_exact,
                "htau": sweep.metrics(
                    direct_vector_htau, htau_n, wet, focus, POINTWISE),
                "full_row": arms["direct_gphit_vector_htau"]["full_row"],
            },
        },
        "operands": operands,
        "arms": arms,
        "controls": controls,
        "bracket": {
            "restart_byte_identical": True,
            "restart_sha256": EXPECTED_RESTART_SHA,
            "shared_tke_stream_count": len(off_tke),
            "shared_tke_streams_byte_identical": True,
            "excluded_oracle_diagnostic": {
                "name": cor_name,
                "reason": (
                    "run-to-run nondeterministic under identical unpatched "
                    "binary; likely uninitialized write-only buffer; not ZDF-consumed"
                ),
                "determinism_first_difference_byte": cor_first_diff,
                "determinism_a_sha256": sha256(det_a / cor_name),
                "determinism_b_sha256": sha256(det_b / cor_name),
                "determinism_restarts_byte_identical": True,
            },
        },
        "nemo_line": (
            "cfgs/DINO/MY_SRC/zdftke.F90:660 patched instrument: "
            "etau_arg_dump=-gdepw(Kmm)/htau"
        ),
        "provenance_sha256": {
            str(path): sha256(path) for path in provenance_paths
        },
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "disposition": disposition,
        "owner": owner,
        "operand_fail_columns": {
            name: metric["n_diverged_columns"] for name, metric in operands.items()
        },
        "full_row_fail_columns": {
            name: value["full_row"]["n_diverged_columns"]
            for name, value in arms.items()
        },
    }, indent=2, sort_keys=True))
    return 0 if disposition == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
