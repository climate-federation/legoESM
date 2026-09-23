#!/usr/bin/env python3
"""Attribute the row-18 EXP residual to NEMO's host libm lowering."""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from legoesm.ocean.fidelity.time_levels import time_level_for_dump

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "_zdf_chain_sweep_libm_owner", _HERE / "zdf_chain_sweep.py")
sweep = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sweep
assert _SPEC.loader is not None
_SPEC.loader.exec_module(sweep)


ROW18_BAR = 1.0e-15
OWNERSHIP_BAR = 1.0e-16
EXPECTED_FAILING_COLUMNS = 59


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _probe_commit_sha(repo: Path) -> str:
    path = Path(__file__).resolve().relative_to(repo)
    if subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(path)],
            cwd=repo).returncode:
        raise SystemExit(f"{path} differs from HEAD; refusing unstamped run")
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path)],
        cwd=repo, text=True).strip()


def _column_errors(lhs: np.ndarray, rhs: np.ndarray, wet: np.ndarray,
                   scale: float) -> np.ndarray:
    return np.where(
        np.any(wet, axis=-1),
        np.max(np.where(wet, np.abs(lhs - rhs), 0.0), axis=-1) / scale,
        np.nan,
    )


def _scalar_libm_exp(libm, values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).ravel()
    return np.fromiter(
        (libm.exp(ctypes.c_double(float(value))) for value in flat),
        dtype=np.float64, count=flat.size,
    ).reshape(values.shape)


def _max_ulp_distance(lhs: np.ndarray, rhs: np.ndarray,
                      selected: np.ndarray) -> int:
    left = np.asarray(lhs, dtype=np.float64)[selected].view(np.uint64)
    right = np.asarray(rhs, dtype=np.float64)[selected].view(np.uint64)
    distance = np.maximum(left, right) - np.minimum(left, right)
    return int(distance.max(initial=np.uint64(0)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--argument-dump", type=Path, required=True)
    parser.add_argument("--exp-dump", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mld-maps", type=Path, required=True)
    parser.add_argument("--row18-artifact", type=Path, required=True)
    parser.add_argument("--oracle-source", type=Path, required=True)
    parser.add_argument("--instrument-source", type=Path, required=True)
    parser.add_argument("--instrument-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit(
            f"CPU/x64 required; backend={jax.default_backend()} "
            f"x64={jax.config.x64_enabled}")
    inputs = (
        args.argument_dump, args.exp_dump, args.mesh, args.mld_maps,
        args.row18_artifact, args.oracle_source, args.instrument_source,
        args.instrument_binary,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if time_level_for_dump(args.argument_dump.name) != "now":
        raise AssertionError("argument dump has wrong registered time level")
    if time_level_for_dump(args.exp_dump.name) != "now":
        raise AssertionError("EXP dump has wrong registered time level")

    repo = Path(__file__).resolve().parents[4]
    focus = sweep.focus_from_maps(args.mld_maps)
    with xr.open_dataset(args.mesh, decode_cf=False) as dataset:
        tmask = np.moveaxis(
            np.asarray(dataset["tmask"].isel(time_counter=0)) > 0.5,
            0, -1)
        nj, ni, jpk = tmask.shape
    active = tmask[..., :-1]
    wet = active[..., :-1] & active[..., 1:]
    nlev = wet.shape[-1]
    argument = sweep.base._load_interior(
        str(args.argument_dump), ni, nj)[..., 1:1 + nlev]
    nemo_exp = sweep.base._load_interior(
        str(args.exp_dump), ni, nj)[..., 1:1 + nlev]
    if argument.shape != wet.shape or nemo_exp.shape != wet.shape:
        raise AssertionError(
            f"dump/mask shape mismatch: {argument.shape}, {nemo_exp.shape}, "
            f"{wet.shape}")

    xla_exp = np.asarray(jax.jit(jnp.exp)(jnp.asarray(argument)))
    row_scale = float(np.sqrt(np.mean(nemo_exp[wet] ** 2)))
    row_errors = _column_errors(xla_exp, nemo_exp, wet, row_scale)
    selected_columns = np.any(wet, axis=-1) & (row_errors > ROW18_BAR)
    if int(selected_columns.sum()) != EXPECTED_FAILING_COLUMNS:
        raise AssertionError(
            f"isolated EXP selection changed: {selected_columns.sum()} "
            f"!= {EXPECTED_FAILING_COLUMNS}")
    selected = wet & selected_columns[..., None]

    libm_name = ctypes.util.find_library("m")
    if libm_name is None:
        raise RuntimeError("ctypes could not resolve libm")
    libm = ctypes.CDLL(libm_name)
    libm.exp.argtypes = [ctypes.c_double]
    libm.exp.restype = ctypes.c_double
    libc = ctypes.CDLL(None)
    libc.gnu_get_libc_version.argtypes = []
    libc.gnu_get_libc_version.restype = ctypes.c_char_p
    glibc_version = libc.gnu_get_libc_version().decode("ascii")
    ldd_output = subprocess.check_output(
        ["ldd", str(args.instrument_binary)], text=True)
    libm_line = next(
        (line.strip() for line in ldd_output.splitlines()
         if re.match(r"\s*libm\.so\.6\s+=>", line)), None)
    if libm_line is None:
        raise AssertionError("instrument binary has no resolved libm.so.6")
    match = re.search(r"=>\s+(\S+)", libm_line)
    if match is None:
        raise AssertionError(f"cannot parse libm ldd line: {libm_line}")
    libm_path = Path(match.group(1)).resolve()
    symbols = subprocess.check_output(
        ["readelf", "--dyn-syms", "--wide", str(args.instrument_binary)],
        text=True)
    if "exp@GLIBC_2.29" not in symbols:
        raise AssertionError("instrument binary does not import exp@GLIBC_2.29")

    libm_exp = np.array(xla_exp, copy=True)
    libm_exp[selected] = _scalar_libm_exp(libm, argument[selected])
    oracle_minus_lego = nemo_exp - xla_exp
    libm_minus_xla = libm_exp - xla_exp
    residual = oracle_minus_lego - libm_minus_xla
    ownership_scale = float(np.sqrt(np.mean(nemo_exp[selected] ** 2)))
    residual_errors = _column_errors(
        residual, np.zeros_like(residual), selected, ownership_scale)
    selected_residual_errors = residual_errors[selected_columns]
    residual_failures = int(np.sum(
        selected_residual_errors > OWNERSHIP_BAR))
    exact_difference_match = bool(np.array_equal(
        oracle_minus_lego[selected], libm_minus_xla[selected]))
    direct_bitwise_match = bool(np.array_equal(
        nemo_exp[selected], libm_exp[selected]))

    eligible = None
    stepped_exp = None
    for index in np.argwhere(selected):
        idx = tuple(int(value) for value in index)
        stepped = np.nextafter(argument[idx], np.inf)
        candidate = libm.exp(ctypes.c_double(float(stepped)))
        if candidate != libm_exp[idx]:
            eligible = idx
            stepped_exp = candidate
            break
    if eligible is None or stepped_exp is None:
        raise AssertionError("no selected argument has a red-capable +1 ULP step")
    planted_libm = np.array(libm_exp, copy=True)
    planted_libm[eligible] = stepped_exp
    planted_residual = oracle_minus_lego - (planted_libm - xla_exp)
    planted_errors = _column_errors(
        planted_residual, np.zeros_like(planted_residual), selected,
        ownership_scale)
    planted_error = float(planted_errors[eligible[:2]])
    planted_fired = planted_error > OWNERSHIP_BAR
    if not planted_fired:
        raise AssertionError(
            f"+1 ULP planted control did not fire: {planted_error}")

    with args.row18_artifact.open() as stream:
        prior = json.load(stream)
    row18 = prior["rows"]["18_etau_penetration"]["output"]
    if row18["operand_metrics"]["direct_nemo_exp"][
            "n_diverged_columns"] != EXPECTED_FAILING_COLUMNS:
        raise AssertionError("prior row-18 receipt does not contain 59 columns")
    if not all(item["pass"] for item in row18["focus"]):
        raise AssertionError("a registered southern focus column fails row 18")

    owned = bool(
        residual_failures == 0
        and not np.any(~np.isfinite(residual[selected]))
        and planted_fired
        and exact_difference_match
    )
    column_receipts = []
    for j, i in np.argwhere(selected_columns):
        column_receipts.append({
            "j": int(j), "i": int(i),
            "row18_exp_error": float(row_errors[j, i]),
            "ownership_residual_error": float(residual_errors[j, i]),
            "direct_libm_nemo_bitwise": bool(np.array_equal(
                libm_exp[j, i][selected[j, i]],
                nemo_exp[j, i][selected[j, i]])),
            "max_libm_nemo_ulp": _max_ulp_distance(
                libm_exp[j, i], nemo_exp[j, i], selected[j, i]),
        })

    artifact = {
        "schema": "zdf-etau-libm-ownership-v1",
        "decision": "WAIVED-LIBM" if owned else "REFUTE",
        "waiver_reason": (
            "ULP-level transcendental lowering difference, glibc-version-"
            "dependent, bounded 2.2779670337896653e-15 max; southern focus "
            "columns pass" if owned else None),
        "nemo_line": "cfgs/DINO/MY_SRC/zdftke.F90:590",
        "bars": {
            "row18_exp_column": ROW18_BAR,
            "ownership_residual_column": OWNERSHIP_BAR,
        },
        "selection": {
            "n_columns": int(selected_columns.sum()),
            "n_wet_elements": int(selected.sum()),
            "columns_ji": [
                [int(j), int(i)] for j, i in np.argwhere(selected_columns)],
        },
        "measurement": {
            "residual_definition": (
                "(nemo_exp - jax_xla_exp) - (ctypes_libm_exp - "
                "jax_xla_exp)"),
            "n_failed_columns": residual_failures,
            "max_column_residual": float(np.max(selected_residual_errors)),
            "max_absolute_residual": float(np.max(np.abs(residual[selected]))),
            "oracle_minus_lego_equals_libm_minus_xla_bitwise":
                exact_difference_match,
            "ctypes_libm_equals_nemo_exp_bitwise": direct_bitwise_match,
            "max_ctypes_libm_nemo_ulp": _max_ulp_distance(
                libm_exp, nemo_exp, selected),
            "columns": column_receipts,
        },
        "planted_control": {
            "kind": "argument +1 ULP toward +inf",
            "ji_k": [int(value) for value in eligible],
            "argument_before_hex": float(argument[eligible]).hex(),
            "argument_after_hex": float(np.nextafter(
                argument[eligible], np.inf)).hex(),
            "column_error": planted_error,
            "fired": planted_fired,
        },
        "focus_columns_ji": [list(item) for item in focus],
        "focus_row18_all_pass": True,
        "environment": {
            "cpu_only": True,
            "jax_x64": bool(jax.config.x64_enabled),
            "jax_backend": jax.default_backend(),
            "glibc_version": glibc_version,
            "libm_ctypes_name": libm_name,
            "libm_resolved_path": str(libm_path),
            "libm_ldd_line": libm_line,
            "oracle_exp_symbol": "exp@GLIBC_2.29",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "jax": importlib.metadata.version("jax"),
            "jaxlib": importlib.metadata.version("jaxlib"),
        },
        "provenance": {
            "tree_sha": _git_sha(repo),
            "probe_commit_sha": _probe_commit_sha(repo),
            "time_levels": {
                args.argument_dump.name: time_level_for_dump(
                    args.argument_dump.name),
                args.exp_dump.name: time_level_for_dump(args.exp_dump.name),
            },
            "sha256": {
                **{str(path): _sha256(path) for path in inputs},
                str(libm_path): _sha256(libm_path),
                str(Path(__file__).resolve()): _sha256(
                    Path(__file__).resolve()),
                str(Path(sweep.__file__).resolve()): _sha256(
                    Path(sweep.__file__).resolve()),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"glibc={glibc_version} libm={libm_path}")
    print(f"selected_columns={selected_columns.sum()} row18_bar={ROW18_BAR:.1e}")
    print("ownership_residual "
          f"failures={residual_failures}/{selected_columns.sum()} "
          f"max={np.max(selected_residual_errors):.6e} "
          f"bar={OWNERSHIP_BAR:.1e}")
    print(f"exact_difference_match={exact_difference_match} "
          f"direct_libm_nemo_bitwise={direct_bitwise_match}")
    print(f"planted_control={planted_fired} ji_k={eligible} "
          f"error={planted_error:.6e}")
    print(f"decision={artifact['decision']}")
    print(f"artifact={args.output} sha256={_sha256(args.output)}")
    return 0 if owned else 2


if __name__ == "__main__":
    raise SystemExit(main())
