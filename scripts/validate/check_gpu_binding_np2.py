"""#1516 GPU-binding validator: prove each MPI rank of a single-node
multi-rank launch computes on its OWN physical GPU.

Why this exists: ``maybe_init_jax_distributed`` returns early on a
single-node multi-rank launch (jax.distributed is not needed there), and
historically NOTHING bound rank -> device, so every rank silently booted on
default GPU 0 — an N-GPU job that was really a 1-GPU job with N-fold memory
pressure, measuring a believable ~1.0x "speedup" (#1516).  The fix pins
``CUDA_VISIBLE_DEVICES`` per rank BEFORE the mpi4py import (a CUDA-aware
MPI initialises the CUDA driver during MPI_Init, and the driver snapshots
CVD at first init — a later pin is silently ignored; measured, Levante
job 26829100 vs 26829180).

Gate design (the non-vacuous instrument):

* NEVER a device LISTING — ``nvidia-smi -L`` / UUID queries ignore
  ``CUDA_VISIBLE_DEVICES`` and pass with the fix removed (the #1516
  vacuous-guard trap).
* Per rank we record what JAX ITSELF resolved (``jax.local_devices()``,
  the ``.sharding`` of a real computed array) AND the PHYSICAL placement:
  every NVML device whose compute-process list contains THIS rank's PID.
  NVML process lists are cgroup-aware and cannot be faked by visibility.
* STRICT success = each rank owns exactly ONE physical GPU and the two
  UUIDs DIFFER.  The unpinned control arm must FAIL this (red = the gate
  can see the defect).

Usage (inside an sbatch on a 2-GPU node; account/partition per site)::

    srun --mpi=pmix --ntasks=2 python scripts/validate/check_gpu_binding_np2.py \
        burn OUT_DIR ARM_NAME
    python scripts/validate/check_gpu_binding_np2.py verdict OUT_DIR ARM_NAME pinned
    # control arm: LEGOESM_NO_LOCAL_GPU_PIN=1 ... verdict ... unpinned

Verified on Levante job 26829180 (node l50187): pinned arm -> ranks on
GPU-9efe... / GPU-9c36... (PASS); unpinned control -> both ranks' compute
on device 0 of a 2-device view (RED AS EXPECTED).
"""
from __future__ import annotations

import json
import os
import sys
import time


def evaluate_binding(records: list[dict], mode: str) -> tuple[bool, str]:
    """Pure verdict logic (unit-tested; no GPUs needed).

    ``records`` — one dict per rank with at least ``rank`` and
    ``physical_uuids`` (the NVML devices holding that rank's PID).
    ``mode`` — ``"pinned"`` (expect strict per-rank binding) or
    ``"unpinned"`` (control: expect the COLLISION, i.e. NOT strict).

    Returns ``(exit_ok, message)``.
    """
    if mode not in ("pinned", "unpinned"):
        raise ValueError(f"unknown mode {mode!r}")
    sets = [set(r["physical_uuids"]) for r in records]
    # STRICT success = each rank owns exactly ONE physical GPU and no GPU
    # is shared.  Anything else (shared GPU, or a rank spread over several
    # — the n_local>1 signature where compute defaults to device 0) is the
    # #1516 collision.
    ok_strict = (
        all(len(s) == 1 for s in sets)
        and len(set.union(*sets)) == len(sets))
    if mode == "pinned":
        return (ok_strict,
                "PASS — each rank on exactly one, DIFFERENT physical GPU"
                if ok_strict else
                f"FAIL — placement {sets} is not one-distinct-GPU-per-rank")
    return ((not ok_strict),
            f"RED AS EXPECTED — placement {sets} collides without the pin "
            "(gate is non-vacuous)" if not ok_strict
            else "UNEXPECTED — ranks distinct without the pin "
                 "(control arm proves nothing)")


def _my_physical_gpus() -> list[str]:
    """UUIDs of physical GPUs where THIS PID has a compute context."""
    import pynvml

    pynvml.nvmlInit()
    me = os.getpid()
    hits = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        except Exception:
            continue
        if any(p.pid == me for p in procs):
            u = pynvml.nvmlDeviceGetUUID(h)
            hits.append(u.decode() if isinstance(u, bytes) else u)
    return hits


def burn(out_dir: str, arm: str) -> None:
    """Rank body: the REAL early-init path, then record what happened."""
    env_keys = ("CUDA_VISIBLE_DEVICES", "SLURM_LOCALID", "SLURM_PROCID",
                "SLURM_STEP_GPUS", "SLURM_STEP_NUM_TASKS",
                "OMPI_COMM_WORLD_LOCAL_RANK")
    pre_env = {k: os.environ.get(k, "UNSET") for k in env_keys}
    rank = pre_env["SLURM_PROCID"]

    # The REAL path under test, exactly as run_amip.py invokes it (before
    # any jax.numpy / legoesm import that would init the XLA backend).
    from legoesm.parallel.early_init import maybe_init_jax_distributed
    maybe_init_jax_distributed()

    post_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET")

    import jax
    import jax.numpy as jnp

    devs = jax.local_devices()
    x = jnp.ones((4096, 4096), jnp.float32)
    x = (x @ x).block_until_ready()  # force a real context + kernel
    rec = {
        "rank": rank, "arm": arm, "pre_env": pre_env, "post_cvd": post_cvd,
        "n_local_devices": len(devs), "devices": [str(d) for d in devs],
        "sharding_devices": [str(d) for d in x.sharding.device_set],
        "physical_uuids": _my_physical_gpus(),
    }
    print(f"[binding-{arm} rank {rank}] post_CVD={post_cvd!r} "
          f"n_local={len(devs)} devices={devs} "
          f"sharding={list(x.sharding.device_set)} "
          f"PHYSICAL={rec['physical_uuids']}", flush=True)
    with open(os.path.join(out_dir, f"rank{rank}_{arm}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    # Hold the context briefly so an external utilisation sampler can see
    # the overlap (secondary evidence only; the gate is the NVML PID scan).
    t0 = time.time()
    while time.time() - t0 < float(os.environ.get("PROBE_BURN_S", "15")):
        x = (x @ x) / 4096.0
        x.block_until_ready()


def verdict(out_dir: str, arm: str, mode: str) -> None:
    records = []
    for r in ("0", "1"):
        p = os.path.join(out_dir, f"rank{r}_{arm}.json")
        if not os.path.exists(p):
            print(f"VERDICT {arm}: FAIL — missing {p}")
            sys.exit(1)
        with open(p) as fh:
            records.append(json.load(fh))
    for r in records:
        print(f"  rank {r['rank']}: post_CVD={r['post_cvd']!r} "
              f"n_local={r['n_local_devices']} "
              f"physical={r['physical_uuids']}")
    ok, msg = evaluate_binding(records, mode)
    print(f"VERDICT {arm} ({mode}): {msg}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "burn":
        burn(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 5 and sys.argv[1] == "verdict":
        verdict(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)
