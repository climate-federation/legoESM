"""The long-run MPI driver runs substepped horizontal acoustics on its
domain-decomposed path, as the serial production runner does.

Three driver runs on a tiny grid: 1 rank (serial ``step``), 2 ranks with
``--use-dd`` (``step_halo``), and 2 ranks with the substep turned off. The
1- and 2-rank runs must agree; the substep must change the answer (so the
flag reaches the halo path; measured 2026-11-20: without it the 2 K bubble
goes non-finite within the 24 steps, with it np1 and np2 agree to 0 ulp of
the float32 snapshots).

Spawns ``mpiexec`` itself; run with plain ``pytest`` (not under mpiexec).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "scripts" / "run" / "run_rce_mpi_long.py"
MPIEXEC = os.environ.get("MPIEXEC") or shutil.which("mpiexec") or shutil.which("mpirun")
DT = 5.0
N_STEPS = 24

pytestmark = pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")


def _run(out: Path, n_ranks: int, *extra: str) -> dict[str, np.ndarray]:
    pytest.importorskip("mpi4py")
    pytest.importorskip("mpi4jax")
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env.pop("LEGOESM_RCE_MPI_FP32", None)
    cmd = [
        MPIEXEC, "-n", str(n_ranks), sys.executable, str(DRIVER),
        "--nx", "12", "--ny", "12", "--nlev", "20",
        "--dx", "2000.0", "--dt", str(DT),
        "--days", repr(N_STEPS * DT / 86400.0),
        "--semi-implicit-acoustic", "--acoustic-off-centering", "0.1",
        "--n-acoustic-substeps", "6", "--advection", "upwind1",
        "--hyperdiff", "5e6", "--bubble-theta-pert", "2.0",
        "--n-physics-substeps", "1", "--no-radiation",
        "--snapshot-3d-hours", repr(N_STEPS * DT / 3600.0),
        "--output", str(out), *extra,
    ]
    if n_ranks > 1:
        cmd.append("--use-dd")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       timeout=900, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, f"{cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    snap = np.load(out / "snapshots_3d" / "snap_hr_0000.npz")
    assert abs(float(snap["t_sim"]) - N_STEPS * DT) < 1e-9
    return {k: np.asarray(snap[k], np.float64) for k in ("T", "qv", "mse")}


def test_dd_substep_horizontal_matches_serial_and_is_active(tmp_path):
    serial = _run(tmp_path / "np1", 1)
    dd = _run(tmp_path / "np2", 2)
    dd_off = _run(tmp_path / "np2_off", 2, "--no-substep-horizontal-acoustic")
    # Snapshots are stored as float32: measure both gaps in float32 ulps of
    # the field's magnitude (the storage quantum).
    for k in ("T", "qv", "mse"):
        assert np.all(np.isfinite(serial[k])) and np.all(np.isfinite(dd[k])), k
        ulp = float(np.spacing(np.float32(np.max(np.abs(serial[k])))))
        gap = np.max(np.abs(dd[k] - serial[k])) / ulp
        # A non-finite OFF run (the expected blow-up) counts as a change.
        effect = (np.inf if not np.all(np.isfinite(dd_off[k]))
                  else np.max(np.abs(dd_off[k] - dd[k])) / ulp)
        print(f"{k}: |np2-np1|={gap:.1f} ulp  |off-on|={effect:.1f} ulp")
        assert gap <= 2.0, (k, gap)
        assert effect >= 100.0, (k, effect)
