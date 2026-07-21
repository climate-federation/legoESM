"""SELF-SPAWNING route-B multicontroller gate for the ``run_amip`` DRIVER.

Companion to the run_omip route-B gate
(``test_run_omip_latlon_spmd_multicontroller_selfspawn.py``): THIS variant
spawns two worker processes and drives the FULL PRODUCTION run_amip DRIVER
(``scripts/run/run_amip.py --enable-latlon-spmd --multicontroller``) end-to-end
through the OPERATOR-SPLIT physics lane — pinning the pieces a Derecho/Levante
multi-node NCCL run_amip run exercises that a single-controller run does NOT:

* the ``main()`` ``init_multicontroller_distributed`` bootstrap firing AFTER
  argparse but BEFORE any device work — and the import-time MPI auto-detect
  correctly SKIPPED for ``--multicontroller`` (it would try to load libmpi
  before argv is parsed and crash);
* the ALL-global-device mesh (``jax.devices()`` spans both processes) with the
  strict-subset guard;
* the cross-process carry gather (``replicate_leaf(multiprocess=True)`` — the
  jit-identity all-gather; a raw ``device_put`` cannot reshard shards on the
  other process's device);
* an all-rank-consistent exit (the finite/NaN guard reads the REPLICATED
  gathered state, so both ranks return the same status in lockstep).

Bitwise SPMD-vs-serial physics parity is
``tests/parallel/test_operator_split_spmd_driver_parity.py`` (the sharded step
is identical whether its devices sit in one process or across two); this gate
asserts the DRIVER federates + integrates to COMPLETED with a finite state and
without a hang/crash.  run_amip's SPMD lane refuses the diagnostics/checkpoint
writers, so completion is asserted from rank-0 stdout, not an output file.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_RUN_AMIP = _ROOT / "scripts" / "run" / "run_amip.py"
N_PROC = 2
N_LAT = 8          # resolution=8 -> n_lat=8 (divisible by N_PROC -> 4 rows/band)


@pytest.mark.timeout(600)
def test_run_amip_two_process_selfspawn_multicontroller(tmp_path):
    from multihost_harness import run_federated

    out_dir = tmp_path / "amip_mc"
    base_env = dict(os.environ)
    base_env["JAX_PLATFORMS"] = "cpu"
    base_env["JAX_ENABLE_X64"] = "1"
    base_env.pop("XLA_FLAGS", None)          # 1 real CPU device per process
    _pkgs = os.pathsep.join(
        str(p) for p in sorted((_ROOT / "packages").glob("*")) if p.is_dir())
    base_env["PYTHONPATH"] = os.pathsep.join(
        [_pkgs, str(_ROOT), base_env.get("PYTHONPATH", "")])

    def build_cmd(rank, port):
        cmd = [
            sys.executable, str(_RUN_AMIP),
            "--grid-type", "latlon", "--resolution", str(N_LAT), "--nlev", "4",
            "--dataset", "analytical",
            "--days", "1", "--dt", "3600",     # 24 steps, CFL-safe on coarse grid
            # The SPMD lane refuses the diag/checkpoint writers (a follow-up);
            # disable them so the route-B run reaches COMPLETED (default diag=5).
            "--diag-days", "0", "--checkpoint-days", "0",
            # gray + TKE only (operator-split lane active); everything else off
            # so the tiny run is fast + stable — the federation plumbing, not the
            # physics, is under test.
            "--radiation", "gray", "--turbulence", "tke",
            "--convection", "none", "--microphysics", "none",
            "--gravity-wave-drag", "none",
            # AMIP normally requires every parameterization active; this is an
            # idealized federation-plumbing run (gray+TKE only), so opt in.
            "--allow-disabled-physics",
            "--fix-mass",
            "--enable-latlon-spmd", "--multicontroller",
            "--coordinator", f"localhost:{port}",
            "--output", str(out_dir),
        ]
        # run_amip reads rank/size from OMPI env when --coordinator is given;
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

    # Rank 0 owns the banners/summary (the lane gates them on process_index==0).
    root = outs[0]
    assert "route-B multicontroller" in root, (
        f"no route-B banner on rank 0 — the multicontroller lane did not "
        f"activate\nrank0 tail:\n{root[-3000:]}")
    m = re.search(
        r"operator-split SPMD run: COMPLETED,.*max\|T\|=([\d.]+) "
        r"max\|u\|=([\d.]+)", root)
    assert m, (
        f"no COMPLETED summary line on rank 0 (a hang/blowup/refusal?)\n"
        f"rank0 tail:\n{root[-3000:]}")
    max_T, max_u = float(m.group(1)), float(m.group(2))
    # The federation integrated to a finite, physical state across processes.
    assert 100.0 < max_T < 1e3, f"max|T|={max_T} unphysical (route-B blowup?)"
    assert 0.0 < max_u < 1e3, f"max|u|={max_u} unphysical (route-B blowup?)"

    # Non-root rank ran the SAME steps + collectives but stays quiet on the
    # rank-0-gated summary (a duplicate write/print would mean I/O wasn't gated).
    assert "operator-split SPMD run: COMPLETED," not in outs[1], (
        f"rank 1 printed the rank-0 summary — I/O not gated to process 0\n"
        f"rank1 tail:\n{outs[1][-2000:]}")
