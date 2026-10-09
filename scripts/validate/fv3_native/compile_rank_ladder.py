"""Compile-time rank ladder for the FV3 M6 window step.

The CPU arm runs the 1 / 6 / 24-device ladder in fresh subprocesses with
``--xla_force_host_platform_device_count``.  It measures Python/JAX lowering,
partitioning, and the effect of program shape without any GPU work.  It NEVER
measures or rules out GPU PTX/LLVM code generation, autotuning, or NCCL layout.

The GPU arm starts at one device with collapsed sharding; larger rows use one
process/GPU and ``--distributed``.  Each row times ``lower()`` and ``compile()``
separately for the same M6 window-step construction used by
``tiled_m6_model_gate.py`` and counts optimized HLO instructions.  Cross-rank
min/median/max values are printed only after every rank has reported, so the
last compiling rank, not an earlier waiting rank, determines the row.

Examples::

    python compile_rank_ladder.py --backend cpu
    python compile_rank_ladder.py --backend gpu --devices 1
    srun ... python compile_rank_ladder.py --backend gpu --devices 24 \
        --distributed
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_DEVICE_COUNTS = (1, 6, 24)
_HLO_INSTRUCTION_RE = re.compile(
    r"^\s*(?:ROOT\s+)?%?\S+\s+=\s+\S+\s+[-\w.]+\(", re.MULTILINE)


def partition_for_devices(device_count: int) -> tuple[int, str]:
    """Return the window tile factor and label for a ladder device count."""
    if device_count == 1:
        return 1, "collapsed"
    if device_count == 6:
        return 1, "faces-6x1x1"
    if device_count == 24:
        return 2, "windows-6x2x2"
    raise ValueError(f"device count must be one of {_DEVICE_COUNTS}, got {device_count}")


def count_hlo_instructions(hlo_text: str) -> int:
    """Count instruction assignments in optimized XLA HLO text."""
    return len(_HLO_INSTRUCTION_RE.findall(hlo_text))


def measure_lower_compile(jitted, operands, *, clock=time.perf_counter,
                          report=lambda phase, event, seconds: None):
    """Time the two compilation phases and return the compiled executable."""
    report("lower", "begin", None)
    start = clock()
    lowered = jitted.lower(*operands)
    lower_s = clock() - start
    report("lower", "complete", lower_s)
    report("compile", "begin", None)
    start = clock()
    compiled = lowered.compile()
    compile_s = clock() - start
    report("compile", "complete", compile_s)
    return lower_s, compile_s, compiled


def _replace_fake_device_flag(flags: str, device_count: int) -> str:
    kept = [part for part in flags.split()
            if not part.startswith("--xla_force_host_platform_device_count=")]
    kept.append(f"--xla_force_host_platform_device_count={device_count}")
    return " ".join(kept)


def _worker_command(args, device_count: int) -> list[str]:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--backend", args.backend,
        "--devices", str(device_count),
        "--n", str(args.n),
        "--km", str(args.km),
        "--pad", str(args.pad),
        "--n-split", str(args.n_split),
        "--dt", str(args.dt),
        "--worker",
    ]
    if args.nh:
        cmd.append("--nh")
    if args.distributed:
        cmd.append("--distributed")
    return cmd


def _jitted_step(model):
    for cell in model._step_fn.__closure__ or ():
        if hasattr(cell.cell_contents, "lower"):
            return cell.cell_contents
    raise RuntimeError("M6 step wrapper contains no jitted callable")


def _run_worker(args) -> int:
    import jax
    import numpy as np

    if args.distributed:
        ids = os.environ.get("JAX_LOCAL_DEVICE_IDS")
        jax.distributed.initialize(
            initialization_timeout=900,
            local_device_ids=[int(i) for i in ids.split(",")] if ids else None,
        )
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
    FV3DuoConfig,
    FV3DuoDynamicsModel,
    ORACLE_DAMPING,
)
    from legoesm.grids.factory import create_fv3_duo_grid

    rank = jax.process_index()
    if jax.device_count() != args.devices:
        raise SystemExit(
            f"expected {args.devices} global devices, got {jax.device_count()}")
    kt, partition = partition_for_devices(args.devices)
    grid = create_fv3_duo_grid(args.n)
    cfg = FV3DuoConfig(**ORACLE_DAMPING, 
        km=args.km, hydrostatic=not args.nh, n_split=args.n_split)
    if args.devices == 1:
        model = FV3DuoDynamicsModel(grid, cfg, step_windows=(kt, args.pad))
    else:
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(6, kt, kt),
            ("face", "tile_i", "tile_j"),
        )
        model = FV3DuoDynamicsModel(
            grid, cfg, step_spmd_mesh=mesh, step_windows=(kt, args.pad))
    state = model.dcmip16_initial_state(do_pert=True)
    operands = (
        state["state"], state["press"], state["q"], args.dt,
        state["omga"], state["nh"],
    )
    jitted = _jitted_step(model)
    def report(phase, event, seconds):
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        elapsed = "" if seconds is None else f" seconds={seconds:.6f}"
        print(f"[compile-rank-ladder] {stamp} rank={rank} "
              f"devices={args.devices} {phase} {event}{elapsed}", flush=True)

    lower_s, compile_s, compiled = measure_lower_compile(
        jitted, operands, report=report)
    hlo_count = count_hlo_instructions(compiled.as_text())

    local = np.asarray((lower_s, compile_s, hlo_count), dtype=np.float64)
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        gathered = np.asarray(multihost_utils.process_allgather(local))
    else:
        gathered = local[None, :]
    if rank == 0:
        lower = gathered[:, 0]
        compile_ = gathered[:, 1]
        hlo = gathered[:, 2].astype(np.int64)
        slowest = int(np.argmax(compile_))
        print(
            f"[compile-rank-ladder] ROW backend={args.backend} "
            f"devices={args.devices} partition={partition} "
            f"lower_s={lower.min():.6f}/{np.median(lower):.6f}/{lower.max():.6f} "
            f"compile_s={compile_.min():.6f}/{np.median(compile_):.6f}/"
            f"{compile_.max():.6f} slowest_compile_rank={slowest} "
            f"hlo_instructions={hlo.min()}..{hlo.max()}",
            flush=True,
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--devices", default=None,
                        help="comma-separated counts; CPU default is 1,6,24; "
                             "GPU default is 1")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--km", type=int, default=10, choices=(5, 10))
    parser.add_argument("--nh", action="store_true")
    parser.add_argument("--pad", type=int, default=11)
    parser.add_argument("--n-split", type=int, default=3)
    parser.add_argument("--dt", type=float, default=900.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    counts = tuple(int(v) for v in (
        args.devices or ("1,6,24" if args.backend == "cpu" else "1")
    ).split(","))
    for count in counts:
        partition_for_devices(count)
    if args.worker:
        if len(counts) != 1:
            raise SystemExit("a worker requires exactly one device count")
        args.devices = counts[0]
        return _run_worker(args)

    print("[compile-rank-ladder] TABLE columns: backend devices partition "
          "lower_s=min/median/max compile_s=min/median/max "
          "slowest_compile_rank hlo_instructions=min..max", flush=True)
    rc = 0
    for count in counts:
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu" if args.backend == "cpu" else "cuda"
        env["JAX_LOG_COMPILES"] = "1"
        if args.backend == "cpu":
            env["XLA_FLAGS"] = _replace_fake_device_flag(
                env.get("XLA_FLAGS", ""), count)
        rc = max(rc, subprocess.run(
            _worker_command(args, count), env=env, check=False).returncode)
    return rc


if __name__ == "__main__":
    sys.exit(main())
