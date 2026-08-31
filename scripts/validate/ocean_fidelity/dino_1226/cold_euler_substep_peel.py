#!/usr/bin/env python
"""CPU-only first-divergence peel of DINO's cold Euler barotropic loop.

Consumes the preregistered ``RUN_KT2/substep_dump.bin`` trajectory and runs
the public standalone construction from the exact common analytic T/S state.
The probe captures every legoESM substep carry without changing the operator,
then reports the first pointwise difference above 1e-15.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import jax
import numpy as np

import hpg_tendency_compare as hpg
import standalone_20y as standalone
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.experiments.dino import apply_dino_lat_lon_surface_forcing
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask


BAR = 1.0e-15
SESSION_ID = "01a053d4-8e9f-7212-bbdb-19ba2d64e140"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diff(name: str, actual, expected, mask) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if actual.shape != expected.shape or mask.shape != expected.shape:
        raise ValueError(
            f"{name}: shape actual={actual.shape} expected={expected.shape} "
            f"mask={mask.shape}")
    delta = np.abs(actual - expected)
    selected = delta[mask]
    if not np.isfinite(selected).all():
        raise ValueError(f"{name}: non-finite values on registered population")
    indices = np.argwhere(mask & (delta > BAR))
    first = None
    if indices.size:
        index = tuple(int(i) for i in indices[0])
        first = {"index": list(index), "actual": float(actual[index]),
                 "expected": float(expected[index]),
                 "abs": float(delta[index])}
    return {
        "name": name,
        "bar": BAR,
        "n": int(selected.size),
        "max_abs": float(selected.max(initial=0.0)),
        "rms": float(np.sqrt(np.mean(selected * selected))),
        "mismatch_count": int(np.count_nonzero(selected > BAR)),
        "status": "PASS" if not indices.size else "OVER_BAR",
        "first_over_bar": first,
    }


def read_trajectory(path: Path) -> tuple[int, int, int, dict]:
    with path.open("rb") as stream:
        jpi, jpj, icycle = (
            int(value) for value in np.fromfile(stream, dtype="<i4", count=3))
        n = jpi * jpj
        names = ("sshn_e", "ssha_e", "zsshp2_e", "un_e", "vn_e",
                 "ua_e", "va_e")
        trajectory = {}
        for _ in range(icycle):
            jn = int(np.fromfile(stream, dtype="<i4", count=1)[0])
            trajectory[jn] = {
                name: np.fromfile(stream, dtype="<f8", count=n).reshape(jpj, jpi)
                for name in names
            }
        if stream.read(1):
            raise ValueError("substep_dump.bin has trailing bytes")
    return jpi, jpj, icycle, trajectory


def capture_cold_step(model, state, forcing, step_forcing, z_coord, cfg):
    import legoesm.ocean.dynamics.barotropic_latlon_cgrid as baro

    captured = {}
    original_run = baro._run_substep_loop
    original_fori = jax.lax.fori_loop
    signature = inspect.signature(original_run)

    def fori_capture(lower, upper, body, init, *args, **kwargs):
        def scan_body(carry, index):
            new = body(index, carry)
            return new, new
        final, stack = jax.lax.scan(
            scan_body, init, jax.numpy.arange(lower, upper))
        captured["carries"] = stack
        return final

    def run_capture(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        captured["loop"] = bound.arguments
        jax.lax.fori_loop = fori_capture
        try:
            return original_run(*args, **kwargs)
        finally:
            jax.lax.fori_loop = original_fori

    common, rate = apply_dino_lat_lon_surface_forcing(
        state, forcing, z_coord, cfg, standalone.DT_SECONDS,
        t_seconds=standalone.DT_SECONDS, return_rate=True)
    baro._run_substep_loop = run_capture
    try:
        with jax.disable_jit():
            model.step(
                common, standalone.DT_SECONDS,
                surface_forcing=step_forcing, external_tracer_rate=rate)
    finally:
        baro._run_substep_loop = original_run
        jax.lax.fori_loop = original_fori
    if "carries" not in captured or "loop" not in captured:
        raise RuntimeError("barotropic substep capture did not fire")
    return captured


def run(args: argparse.Namespace) -> int:
    if os.environ.get("JAX_PLATFORM_NAME") != "cpu":
        raise SystemExit("CPU-only peel requires explicit JAX_PLATFORM_NAME=cpu")
    producer, dirty = standalone.git_provenance()
    if dirty:
        raise SystemExit("REFUSING dirty tracked producer")
    set_policy(PrecisionPolicy.fp64())

    run_dir = args.run_kt2.resolve()
    trajectory_path = run_dir / "substep_dump.bin"
    mesh_path = run_dir / "mesh_mask.nc"
    jpi, jpj, icycle, nemo = read_trajectory(trajectory_path)
    grid = read_nemo_mesh_mask(str(mesh_path), nn_hls=0)
    nemo_t, nemo_s = hpg.nemo_istate_case4(
        grid.gdept_0, grid.gphit, grid.tmask)
    # NEMO construction frame -> standalone physical core.
    nemo_t = np.asarray(nemo_t)[2:-2, 2:-2, :-1]
    nemo_s = np.asarray(nemo_s)[2:-2, 2:-2, :-1]

    (cfg, _geometry, z_coord, state, _model_cfg, model, forcing,
     step_forcing, _perturbation) = standalone.build_standalone(0)
    state = state._replace(
        T=state.T.replace(data=nemo_t), S=state.S.replace(data=nemo_s))
    captured = capture_cold_step(
        model, state, forcing, step_forcing, z_coord, cfg)
    loop = captured["loop"]
    carries = captured["carries"]
    if int(loop["n_loop"]) != icycle:
        raise ValueError(
            f"substep count lego={int(loop['n_loop'])} NEMO={icycle}")
    if (jpj, jpi) != (203, 56):
        raise ValueError(f"unexpected NEMO runtime shape {(jpj, jpi)}")

    # Full runtime frame has the two MPI halos plus the two construction
    # rings excluded by standalone: 203x56 -> 195x48.
    edge = 4
    core = lambda value: np.asarray(value)[edge:-edge, edge:-edge]
    # ``read_nemo_mesh_mask(nn_hls=0)`` has already removed the runtime MPI
    # halo, leaving only the two construction rings; do not strip four twice.
    mesh_core = lambda value: np.asarray(value)[2:-2, 2:-2]
    tmask = (mesh_core(np.asarray(grid.tmask)[..., 0]) > 0.5) & (
        np.asarray(state.land_mask.data) > 0.5)
    umask = (mesh_core(np.asarray(grid.umask)[..., 0]) > 0.5) & (
        np.asarray(state.u_mask.data)[:, 1:] > 0.5)
    vmask = (mesh_core(np.asarray(grid.vmask)[..., 0]) > 0.5) & (
        np.asarray(state.v_mask.data)[1:, :] > 0.5)

    def slow_forcing(name: str) -> np.ndarray:
        value = np.fromfile(run_dir / name, dtype="<f8")
        if value.size != 199 * 52:
            raise ValueError(f"{name}: unexpected value count {value.size}")
        return value.reshape(199, 52)[2:-2, 2:-2]

    rows = [
        diff("slow_forcing_U", np.asarray(loop["F_slow_u"])[:, 1:],
             slow_forcing("spg_dump_zu_frc.bin"), umask),
        diff("slow_forcing_V", np.asarray(loop["F_slow_v"])[1:, :],
             slow_forcing("spg_dump_zv_frc.bin"), vmask),
    ]
    for index in range(icycle):
        jn = index + 1
        rows.extend([
            diff(f"substep_{jn:02d}_SSH_after", carries[0][index],
                 core(nemo[jn]["ssha_e"]), tmask),
            diff(f"substep_{jn:02d}_U_after", carries[1][index][:, 1:],
                 core(nemo[jn]["ua_e"]), umask),
            diff(f"substep_{jn:02d}_V_after", carries[2][index][1:, :],
                 core(nemo[jn]["va_e"]), vmask),
        ])
    first = next((row for row in rows if row["status"] == "OVER_BAR"), None)

    # Live planted trajectory shift: it must move the first-substep velocity
    # check by more than the bar and therefore cannot pass vacuously.
    planted = diff(
        "CONTROL_shift_NEMO_U_one_substep", carries[1][0][:, 1:],
        core(nemo[2]["ua_e"]), umask)
    if planted["status"] != "OVER_BAR":
        raise ValueError("planted one-substep trajectory shift did not fire")

    artifact = {
        "schema": "dino_cold_euler_substep_peel_v1",
        "session_id": SESSION_ID,
        "producer_git_sha": producer,
        "producer_dirty_tracked_files": dirty,
        "bar": BAR,
        "nemo_source": {
            "seed": "dynspg_ts.F90:578-586 (Kbb; cold istate Kbb==Kmm)",
            "ramp": "dynspg_ts.F90:620-628 (first two rows 1,0,0)",
            "continuity": "dynspg_ts.F90:688-714",
            "am4_pgf": "dynspg_ts.F90:748-758",
            "momentum": "dynspg_ts.F90:837-851",
        },
        "controls": {"substep_count": icycle,
                     "shift_plant": planted},
        "first_over_bar": None if first is None else first["name"],
        "rows": rows,
        "loop_receipt": {
            "dt_s": float(np.asarray(loop["dt_s"])),
            "n_loop": int(loop["n_loop"]),
            "seed_eta_max_abs": float(np.max(np.abs(np.asarray(loop["eta"])))),
            "seed_u_max_abs": float(np.max(np.abs(np.asarray(loop["U_bar"])))),
            "seed_v_max_abs": float(np.max(np.abs(np.asarray(loop["V_bar"])))),
        },
        "inputs": {
            str(trajectory_path): {"bytes": trajectory_path.stat().st_size,
                                   "sha256": sha256(trajectory_path)},
            str(mesh_path): {"bytes": mesh_path.stat().st_size,
                             "sha256": sha256(mesh_path)},
        },
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for row in rows:
        if row["status"] == "OVER_BAR" or row["name"].startswith("substep_01"):
            print(f"ROW {row['name']} {row['status']} "
                  f"max={row['max_abs']:.17e} n={row['mismatch_count']}")
        if first is row:
            break
    print(f"FIRST_OVER_BAR={artifact['first_over_bar']}")
    print(f"WROTE={args.output}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-kt2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
