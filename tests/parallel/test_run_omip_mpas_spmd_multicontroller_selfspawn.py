"""Two-process CPU route-B gate for the MPAS ocean SPMD lane
(``run_omip.py --grid mpas --enable-mpas-spmd --multicontroller``).

Single-controller parity (``test_mpas_ocean_spmd_step.py``) cannot see the
multi-process failure class: a process-spanning sharded array captured by
closure in the jitted block scan, or a host fetch of a non-addressable array
in the loop (both killed the first multicontroller GPU smoke, job 27326561).
This spawns two federated worker processes on CPU and drives the FULL driver
end-to-end on ico2 with a synthetic JRA55 cache (host-regrid lane, partial
cells off, prognostic thermo-only sea ice so the sharded ice tile reaches
the restart writer), asserting every rank exits 0 and rank 0 writes results
+ a restart carrying the SPMD provenance.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_RUN_OMIP = _ROOT / "scripts" / "run" / "run_omip.py"
N_PROC = 2
sys.path.insert(0, str(_THIS.parent))


def _load_day23():
    day23 = _ROOT / "tests" / "unit" / "test_run_omip_jra55_dispatch.py"
    spec = importlib.util.spec_from_file_location("_day23_mpas_mc", day23)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_day23_mpas_mc"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.timeout(900)
def test_run_omip_mpas_spmd_two_process_selfspawn(tmp_path):
    from multihost_harness import run_federated
    _day23 = _load_day23()
    cache = _day23._make_synthetic_cache(tmp_path, n_lat=8, n_lon=16, n_records=16)
    out_dir = tmp_path / "omip_mpas_mc"
    base_env = dict(os.environ)
    base_env["JAX_PLATFORMS"] = "cpu"
    base_env["JAX_ENABLE_X64"] = "1"
    base_env.pop("XLA_FLAGS", None)  # 1 real CPU device per process
    _pkgs = os.pathsep.join(
        str(p) for p in sorted((_ROOT / "packages").glob("*")) if p.is_dir())
    base_env["PYTHONPATH"] = os.pathsep.join(
        [_pkgs, str(_ROOT), base_env.get("PYTHONPATH", "")])

    def build_cmd(rank, port):
        cmd = [
            sys.executable, str(_RUN_OMIP),
            "--grid", "mpas", "--resolution", "ico2", "--nlev", "4",
            "--forcing-mode", "jra55_do_tropical", "--jra55-cache", str(cache),
            "--no-gpu-interp",
            "--jra55-sea-ice", "--ice-dynamics", "none", "--ice-categories", "1",
            "--days", "0.05", "--dt", "600",      # ~7 steps
            "--checkpoint-days", "0.03",          # force a rank-0 restart write
            "--max-wallclock-seconds", "3600",
            "--enable-mpas-spmd", "--multicontroller",
            "--coordinator", f"localhost:{port}",
            "--output", str(out_dir),
        ]
        return ["env", f"OMPI_COMM_WORLD_SIZE={N_PROC}",
                f"OMPI_COMM_WORLD_RANK={rank}"] + cmd

    rcs, outs = run_federated(build_cmd, N_PROC, base_env, timeout_s=800)
    for rank, (rc, txt) in enumerate(zip(rcs, outs)):
        assert rc == 0, f"rank {rank} exited {rc}\n--- output ---\n{txt[-4000:]}"
        # Route-B swallows a rank-0 restart write error as a WARNING to keep the
        # federation in lockstep -- so a host fetch of an ungathered shard
        # would otherwise pass rc == 0 with zero restarts written.
        assert "restart write failed" not in txt, f"rank {rank}:\n{txt[-3000:]}"
    res_dir = out_dir / "mpas" / "ico2"
    rec = json.loads((res_dir / "results.json").read_text())
    assert rec["status"] in ("PASS", "FAIL"), rec
    assert rec["final_SST"] is not None and abs(rec["final_SST"]) < 1e4, rec
    restarts = sorted((res_dir / "restarts").glob("restart_day*.npz"))
    assert restarts, f"no restart under {res_dir / 'restarts'}\nrank0 tail:\n{outs[0][-3000:]}"
    with np.load(restarts[-1]) as data:
        assert int(data["mpas_spmd_n_devices"]) == N_PROC
        n_cells = data["T"].shape[0]
        for k in ("ice_h_ice", "ice_T_ice", "ice_concentration"):
            assert data[k].shape[0] == n_cells, (k, data[k].shape, n_cells)
            assert np.all(np.isfinite(data[k])), k


@pytest.mark.timeout(600)
def test_run_omip_mpas_jra55_single_process_control_runs(tmp_path):
    """Single-device control of the gate above (same driver path WITHOUT
    --enable-mpas-spmd / --multicontroller).  The JRA55 block-scan loop used
    to call the lat-lon-only ``prime_step_caches`` hook on the MPAS model
    and abort every single-device MPAS JRA55 run before its first step --
    which also made the SPMD-vs-serial discriminating control unrunnable."""
    import subprocess
    _day23 = _load_day23()
    cache = _day23._make_synthetic_cache(tmp_path, n_lat=8, n_lon=16, n_records=16)
    out_dir = tmp_path / "omip_mpas_serial"
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env.pop("XLA_FLAGS", None)
    _pkgs = os.pathsep.join(
        str(p) for p in sorted((_ROOT / "packages").glob("*")) if p.is_dir())
    env["PYTHONPATH"] = os.pathsep.join([_pkgs, str(_ROOT), env.get("PYTHONPATH", "")])
    cmd = [sys.executable, str(_RUN_OMIP),
           "--grid", "mpas", "--resolution", "ico2", "--nlev", "4",
           "--forcing-mode", "jra55_do_tropical", "--jra55-cache", str(cache),
           "--no-gpu-interp",
           "--jra55-sea-ice", "--ice-dynamics", "none", "--ice-categories", "1",
           "--days", "0.05", "--dt", "600", "--output", str(out_dir)]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=500)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    rec = json.loads((out_dir / "mpas" / "ico2" / "results.json").read_text())
    assert rec["status"] in ("PASS", "FAIL"), rec
    assert rec["final_SST"] is not None and abs(rec["final_SST"]) < 1e4, rec
