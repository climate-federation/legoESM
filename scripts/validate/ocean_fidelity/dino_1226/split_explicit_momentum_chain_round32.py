#!/usr/bin/env python3
"""Score the held NEMO/production row-1.4 substep trajectory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import jax.numpy as jnp
import numpy as np

import split_explicit_momentum_chain_round29 as r29
import spg_substep_chain as inherited
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask
from zdf_stream_bracket import files_byte_identical, stream_manifest


ROUND31_SHA = "7ce94a68528e0e1b0fe950eeaa1ca194bcd9bbdfa0c8545459bbd7141888643d"
TRACE_NAMES = (
    "row14_dump_ssh_substeps.bin",
    "row14_dump_u_substeps.bin",
    "row14_dump_v_substeps.bin",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_trace(path: Path, icycle: int, jpj: int, jpi: int, hls: int) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    expected = icycle * jpj * jpi
    if raw.size != expected:
        raise SystemExit(f"{path}: expected {expected} values, got {raw.size}")
    full = raw.reshape((icycle, jpj, jpi))
    return full[:, hls:-hls, hls:-hls]


def _capture_production_trace(output: Path, n_loop: int) -> tuple[np.ndarray, ...]:
    real_fori = jax.lax.fori_loop
    captures: list[tuple[Any, Any, Any]] = []

    def tracing_fori(lower, upper, body, init, *args, **kwargs):
        if (lower == 0 and upper == n_loop
                and isinstance(init, tuple) and len(init) == 14):
            def scan_body(carry, index):
                new_carry = body(index, carry)
                # The NEMO writer is before the current substep update: row jn
                # is sshn_e/un_e/vn_e at substep START. Return the carry-in,
                # while advancing the production recurrence with new_carry.
                return new_carry, (carry[0], carry[1], carry[2])

            final, trace = jax.lax.scan(
                scan_body, init, jnp.arange(lower, upper))
            captures.append(trace)
            return final
        return real_fori(lower, upper, body, init, *args, **kwargs)

    jax.lax.fori_loop = tracing_fori
    try:
        r29._capture_production_pssh(output)
    finally:
        jax.lax.fori_loop = real_fori
    if jax.lax.fori_loop is not real_fori:
        raise SystemExit("jax.lax.fori_loop restoration failed")
    if len(captures) != 2:
        raise SystemExit(
            f"expected forcing-only and forcing+seed outer traces, got {len(captures)}")
    # Round 6 executes forcing_only first. The second capture is its
    # forcing+seed restoration/control arm.
    return tuple(np.asarray(jax.device_get(value)) for value in captures[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--on-run", type=Path, required=True)
    parser.add_argument("--off-run", type=Path, required=True)
    parser.add_argument("--bracket", type=Path, required=True)
    parser.add_argument("--round31", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_policy(PrecisionPolicy.fp64())
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    root = Path(__file__).resolve().parents[4]
    if r29._tracked_status(root):
        raise SystemExit("tracked-clean checkout required")
    if sha256(args.round31) != ROUND31_SHA:
        raise SystemExit("round-31 production replay changed")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")

    on, off = args.on_run.resolve(), args.off_run.resolve()
    bracket = json.loads(args.bracket.read_text())
    on_manifest, off_manifest = stream_manifest(on), stream_manifest(off)
    expected = set(TRACE_NAMES)
    bracket_ok = (
        len(off_manifest) == 223
        and len(on_manifest) == 226
        and set(on_manifest) - set(off_manifest) == expected
        and set(off_manifest).issubset(on_manifest)
        and all(files_byte_identical(on / name, off / name)
                for name in off_manifest)
        and all(on_manifest[name]["size_bytes"] == 6_184_192
                for name in expected)
        and bracket.get("shared_exact") is True
        and bracket.get("new_streams") == sorted(expected)
    )
    if not bracket_ok:
        raise SystemExit("223/223+3 trace bracket failed")

    jpi, jpj, _jpk, hls, icycle, _nn_e = inherited._read_dims(str(on))
    if (jpi, jpj, hls, icycle) != (56, 203, 2, 68):
        raise SystemExit(f"unexpected runtime dimensions {(jpi, jpj, hls, icycle)}")
    mesh = read_nemo_mesh_mask(str(on / "mesh_mask.nc"), nn_hls=0)
    nemo = {
        "ssh": _load_trace(on / TRACE_NAMES[0], icycle, jpj, jpi, hls),
        "u": _load_trace(on / TRACE_NAMES[1], icycle, jpj, jpi, hls),
        "v": _load_trace(on / TRACE_NAMES[2], icycle, jpj, jpi, hls),
    }
    eta_trace, u_trace, v_trace = _capture_production_trace(
        Path("/tmp/dino_split_explicit_momentum_chain_round32_capture.json"),
        icycle)
    production = {
        "ssh": eta_trace,
        "u": u_trace[:, :, 1:],
        "v": v_trace[:, 1:, :],
    }
    expected_shapes = {
        "ssh": (icycle, 199, 52),
        "u": (icycle, 199, 52),
        "v": (icycle, 199, 52),
    }
    for name in production:
        if production[name].shape != expected_shapes[name] or nemo[name].shape != expected_shapes[name]:
            raise SystemExit(
                f"{name} trace shape mismatch production={production[name].shape} "
                f"nemo={nemo[name].shape}")
    masks = {
        "ssh": np.asarray(mesh.tmask[..., 0]) > 0.5,
        "u": np.asarray(mesh.umask[..., 0]) > 0.5,
        "v": np.asarray(mesh.vmask[..., 0]) > 0.5,
    }
    populations = {name: int(mask.sum()) for name, mask in masks.items()}
    if populations != {"ssh": 9920, "u": 9758, "v": 9868}:
        raise SystemExit(f"population drift: {populations}")

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in production}
    first_strict: dict[str, int | None] = {}
    for name in ("ssh", "u", "v"):
        for index in range(icycle):
            rows[name].append(r29._score(
                production[name][index], nemo[name][index], masks[name]))
        first_strict[name] = next((
            index + 1 for index, row in enumerate(rows[name])
            if row["gate_status"] != "AT BAR"), None)

    bound = json.loads(args.round31.read_text())
    bound_entry = {
        row["field"]: row for row in bound["measurements"]
        if row["arm"] == "forcing_only" and row["subrow"] == "1.2"
    }
    mappings = {"ssh": "sshn_e_init", "u": "un_e_init", "v": "vn_e_init"}
    bound_ok = all(np.isclose(
        rows[name][0]["normalized_rms_error"],
        bound_entry[field]["normalized_rms_error"], rtol=0.0, atol=0.0)
        for name, field in mappings.items())
    controls = r29._strict_controls(
        production["ssh"][0], nemo["ssh"][0], masks["ssh"])
    controls.update({
        "bracket_exact": bracket_ok,
        "fori_loop_restored": True,
        "substep1_entry_bound": bound_ok,
        "last_start_finite": all(np.isfinite(value[-1]).all()
                                 for value in production.values()),
    })
    if not all(controls[name] for name in (
            "identity_at_bar", "four_nextafter_fires", "zero_shift_best",
            "bracket_exact", "fori_loop_restored", "substep1_entry_bound",
            "last_start_finite")):
        disposition = "INVALID"
    elif first_strict["ssh"] is None:
        disposition = "NO_STRICT_SSH_FAILURE"
    else:
        first_velocity = min(
            value for value in (first_strict["u"], first_strict["v"])
            if value is not None)
        disposition = (
            "SSH_PROPAGATION_LOCALIZED"
            if first_velocity < first_strict["ssh"]
            else "SSH_FAILS_WITH_VELOCITY")

    paths = {
        "round31": args.round31,
        "bracket": args.bracket,
        "patch": root / "scripts/validate/ocean_fidelity/dino_1226/nemo_spg_row14_substep_trace.patch",
        "script": Path(__file__).resolve(),
        **{name: on / name for name in TRACE_NAMES},
    }
    for hidden in (".nemo_binary_sha256", ".dynspg_source_sha256"):
        if not (on / hidden).is_file():
            raise SystemExit(f"missing producer receipt {on / hidden}")
        paths[hidden] = on / hidden
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round32-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "source": "dynspg_ts.F90 pre-update carry at each jn; prior swap commits next row",
        "shapes": expected_shapes,
        "populations": populations,
        "first_strict_failure_substep": first_strict,
        "rows": rows,
        "controls": controls,
        "bindings": {name: sha256(path) for name, path in paths.items()},
        "disposition": disposition,
    }
    if r29._tracked_status(root):
        raise SystemExit("tracked worktree changed during score")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition} first_strict_failure_substep={first_strict}")
    return 0 if disposition != "INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
