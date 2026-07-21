"""SELF-SPAWNING route-B multicontroller gate for the ``run_omip`` DRIVER.

Companion to the ocean *bench* selfspawn gate
(``test_latlon_ocean_spmd_multicontroller_selfspawn.py``, which drives
``bench_ocean_latlon_spmd_scaling.py``): THIS variant spawns two worker
processes and drives the FULL PRODUCTION DRIVER
(``scripts/run/run_omip.py --enable-latlon-spmd --multicontroller``)
end-to-end — pinning the driver pieces a Derecho/Levante multi-node NCCL
OMIP run exercises that the bench does NOT:

* the early ``init_multicontroller_distributed`` bootstrap firing BEFORE any
  device work (a module-import device query would make
  ``jax.distributed.initialize`` raise "must be called before backend init");
* the lifted ``jax.process_count() > 1`` refusal + the ALL-global-device mesh;
* the rank-0-gated file I/O (``_save_output(write=...)``, the gather-aware
  ``save_restart`` splitting the collective gather from the rank-0 write, the
  MLD snapshot) while every rank still dispatches the collective gather;
* a consistent per-process exit code (all ranks build ``results``).

Numerics of the sharded step itself live in
``tests/parallel/test_latlon_ocean_spmd_step.py``; SPMD-vs-serial loop parity
in ``tests/unit/test_run_omip_latlon_spmd.py``.  This gate asserts the DRIVER
federates + writes exactly one clean output set without a hang or crash.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_RUN_OMIP = _ROOT / "scripts" / "run" / "run_omip.py"
N_PROC = 2


def _load_day23():
    """Load the JRA55 dispatch harness (synthetic-cache builder) by path."""
    day23 = _ROOT / "tests" / "unit" / "test_run_omip_jra55_dispatch.py"
    spec = importlib.util.spec_from_file_location("_day23_mc", day23)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_day23_mc"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.timeout(600)
def test_run_omip_two_process_selfspawn_multicontroller(tmp_path):
    from multihost_harness import run_federated

    # Build a tiny JRA55-do synthetic cache once in the PARENT; both workers
    # read it (read-only) — this drives the richest route-B path: the
    # gpu-interp block-scan with forcing stacks sharded across processes, the
    # cross-process gather, and the rank-0 restart write.
    _day23 = _load_day23()
    n_lat, n_lon = 16, 32
    cache = _day23._make_synthetic_cache(
        tmp_path, n_lat=n_lat, n_lon=n_lon, n_records=16)

    out_dir = tmp_path / "omip_mc"
    base_env = dict(os.environ)
    base_env["JAX_PLATFORMS"] = "cpu"
    base_env["JAX_ENABLE_X64"] = "1"
    base_env.pop("XLA_FLAGS", None)  # 1 real CPU device per process
    # The worker loads run_omip.py by path; give it repo root + every package
    # on PYTHONPATH so its imports resolve without an editable install.
    _pkgs = os.pathsep.join(
        str(p) for p in sorted((_ROOT / "packages").glob("*")) if p.is_dir())
    base_env["PYTHONPATH"] = os.pathsep.join(
        [_pkgs, str(_ROOT), base_env.get("PYTHONPATH", "")])

    def build_cmd(rank, port):
        cmd = [
            sys.executable, str(_RUN_OMIP),
            "--grid", "latlon", "--resolution", f"{n_lat}x{n_lon}",
            "--nlev", "4",
            "--forcing-mode", "jra55_do_tropical", "--jra55-cache", str(cache),
            "--days", "0.05", "--dt", "600",      # ~7 steps
            "--checkpoint-days", "0.03",          # force a rank-0 restart write
            # Arm the wallclock path (never exhausts at this size) so the
            # cross-process exit-consensus all-gather runs each block without
            # triggering exit — exercises the deadlock-free wallclock check.
            "--max-wallclock-seconds", "3600",
            "--enable-latlon-spmd", "--multicontroller",
            "--coordinator", f"localhost:{port}",
            "--output", str(out_dir),
        ]
        # run_omip reads rank/size from OMPI env when --coordinator is given;
        # run_federated only varies argv, so inject the rank via `env`.
        return [
            "env",
            f"OMPI_COMM_WORLD_SIZE={N_PROC}",
            f"OMPI_COMM_WORLD_RANK={rank}",
        ] + cmd

    rcs, outs = run_federated(build_cmd, N_PROC, base_env, timeout_s=540)
    for rank, (rc, txt) in enumerate(zip(rcs, outs)):
        assert rc == 0, (
            f"rank {rank} exited {rc}\n--- output ---\n{txt[-4000:]}")

    # Rank 0 owns the output; exactly one clean results.json under the
    # grid/resolution tree (non-root ranks return early with write=False).
    res_dir = out_dir / "latlon" / f"{n_lat}x{n_lon}"
    res_path = res_dir / "results.json"
    assert res_path.exists(), (
        f"no results.json at {res_path}\nrank0 tail:\n{outs[0][-3000:]}")
    rec = json.loads(res_path.read_text())
    assert rec["status"] in ("PASS", "FAIL"), rec
    # The federation advanced without blowing up (finite final SST).
    assert rec["final_SST"] is not None and abs(rec["final_SST"]) < 1e4, rec

    # The gather-aware restart write fired on rank 0: its collective gather ran
    # on BOTH ranks (a rank-0-only gather would have hung the federation), then
    # only rank 0 wrote the file.  The restart must carry the full (n_lat+1, ..)
    # staggered v the single-controller gate also checks.
    restarts = sorted((res_dir / "restarts").glob("restart_day*.npz"))
    assert restarts, (
        f"no restart under {res_dir/'restarts'}\nrank0 tail:\n{outs[0][-3000:]}")
    import numpy as np
    with np.load(restarts[-1]) as data:
        v_key = [k for k in data.files if k in ("v", "v_data")]
        assert v_key, f"no v field in restart: {data.files}"
        assert data[v_key[0]].shape[0] == n_lat + 1, (
            f"restart v leading dim {data[v_key[0]].shape[0]} != n_lat+1 "
            f"({n_lat + 1}) — route-B save missed the gather")
