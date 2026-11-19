"""The Levante AMIP chain's opt-in GPU-direct mode (AMIP_GPU_DIRECT=1).

Off by default: the default launch stays srun + the cluster OpenMPI with
host-staged halos.  On, the run goes through NVIDIA HPC-X under its own
mpirun with CUDA-aware MPI.  The guards are lifted from the script and run in
a shell, so a guard that stops refusing goes red.
"""
from __future__ import annotations

import pathlib
import subprocess

_REPO = pathlib.Path(__file__).resolve().parents[2]
_LEVANTE = _REPO / "scripts" / "cluster" / "levante" / "amip_mpas_gpu_chain.sbatch"


def _between(start: str, end: str) -> str:
    text = _LEVANTE.read_text()
    i = text.index(start)
    return text[i:text.index(end, i)]


def _run(tmp_path, body: str, **env) -> subprocess.CompletedProcess:
    pre = "".join(f"{k}={v}\n" for k, v in env.items())
    p = tmp_path / "block.sh"
    p.write_text("set -uo pipefail\n" + pre + body + "\nexit 0\n")
    return subprocess.run(["bash", str(p)], capture_output=True, text=True)


def _mode_block() -> str:
    return _between(': "${AMIP_GPU_DIRECT:=0}"', 'PKG="${REPO}/packages"')


def test_default_is_host_staged_with_the_production_venv(tmp_path):
    r = _run(tmp_path, _mode_block() + '\necho "$AMIP_GPU_DIRECT $PY"', REPO="/r")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["0", "/r/.venv/bin/python"]


def test_gpu_direct_runs_on_the_production_venv(tmp_path):
    body = _mode_block() + '\necho "$AMIP_GPU_DIRECT $PY"'
    r = _run(tmp_path, body, REPO="/r", AMIP_GPU_DIRECT=1)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["1", "/r/.venv/bin/python"]


def test_an_unknown_mode_is_refused(tmp_path):
    for bad in ("yes", "2", "true"):
        assert _run(tmp_path, _mode_block(), REPO="/r", AMIP_GPU_DIRECT=bad).returncode == 71


def test_gpu_direct_is_refused_off_a_single_node(tmp_path):
    guard = _between('if [[ -n "${AMIP_STEP_NODE:-}" ]] || ((', "# shellcheck source=/dev/null")
    assert _run(tmp_path, guard, SLURM_JOB_NUM_NODES=2).returncode == 72
    assert _run(tmp_path, guard, AMIP_STEP_NODE="l40001").returncode == 72
    assert _run(tmp_path, guard, SLURM_JOB_NUM_NODES=1).returncode == 0


_MODE_IF = 'if [[ "${AMIP_GPU_DIRECT}" == "0" ]]; then'


def _launch_branches() -> tuple[str, str]:
    """(host-staged, GPU-direct) halves of the launch if/else, anchored on its own `if`."""
    block = _between(_MODE_IF, 'echo "[chain] launching on')
    host, sep, gdr = block.partition("\nelse\n")
    assert sep, "launch if/else not found"
    return host, gdr


def test_default_launch_is_unchanged_srun_host_staged():
    host, _ = _launch_branches()
    assert "export MPI4JAX_USE_CUDA_MPI=0" in host
    assert 'LAUNCH=( srun --ntasks="${AMIP_N_GPUS}"' in host
    assert "hpcx" not in host


def test_gpu_direct_launch_uses_hpcx_mpirun_one_gpu_per_rank():
    _, gdr = _launch_branches()
    assert "LAUNCH=( srun" not in gdr
    assert 'source "${REPO}/scripts/cluster/levante/hpcx_gdr_env.sh"' in gdr
    assert "export MPI4JAX_USE_CUDA_MPI=1" in gdr
    assert 'LAUNCH=( mpirun -np "${AMIP_N_GPUS}"' in gdr
    assert "CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}" in gdr


def test_the_mode_is_carried_into_the_next_link():
    text = _LEVANTE.read_text()
    resubmit = text[text.index("NEXT=$(sbatch"):]
    assert "AMIP_GPU_DIRECT=${AMIP_GPU_DIRECT}" in resubmit
