"""``run_omip.py --multicontroller`` must federate BEFORE any legoESM import.

``packages/land/legoesm/land/canopy/solver.py`` builds ``jnp.array`` constants
at import time, which initialises the XLA backend; ``run_omip.py`` imports
that module transitively at the top of the file.  ``jax.distributed.initialize``
then refuses ("must be called before any JAX calls that might initialise the
XLA backend") — which is exactly how the first ORCA12 route-B smoke died on 16
GPUs (job 27324036).  The driver now sniffs ``--multicontroller`` from argv and
federates before its legoESM imports; this test imports the module in a
SUBPROCESS with a 1-process explicit-coordinator federation and checks that
(a) the import does not raise the backend-order error and (b) the federation is
up after import.  A control import without the flag must NOT federate.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _import_in_subprocess(argv_extra, env_extra):
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("SLURM_") or k.startswith("PMI") or k.startswith("OMPI_"))}
    env.update({"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"})
    env.update(env_extra)
    code = (
        "import sys; sys.argv = ['run_omip.py'] + " + repr(list(argv_extra)) + "; "
        f"sys.path.insert(0, {str(_ROOT)!r}); "
        "import scripts.run.run_omip; import jax; "
        "print('FEDERATED', jax.distributed.is_initialized())"
    )
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.mark.timeout(700)
def test_multicontroller_flag_federates_before_legoesm_imports():
    port = _free_port()
    r = _import_in_subprocess(
        ["--grid", "tripole", "--enable-latlon-spmd", "--multicontroller",
         "--coordinator", f"127.0.0.1:{port}"],
        {"OMPI_COMM_WORLD_SIZE": "1", "OMPI_COMM_WORLD_RANK": "0"})
    assert "must be called before any JAX calls" not in r.stderr, r.stderr[-3000:]
    assert r.returncode == 0, r.stdout[-2000:] + "\n" + r.stderr[-4000:]
    assert "FEDERATED True" in r.stdout, r.stdout[-2000:]


@pytest.mark.timeout(700)
def test_eq_coordinator_form_federates_and_abbreviation_does_not():
    """``--coordinator=host:port`` is honoured; an ABBREVIATED
    ``--multicontrol`` is deliberately not (the full parser would call ``--m``
    ambiguous), so it must not federate at import -- main() then refuses."""
    port = _free_port()
    r = _import_in_subprocess(
        ["--grid", "tripole", "--enable-latlon-spmd", "--multicontroller",
         f"--coordinator=127.0.0.1:{port}"],
        {"OMPI_COMM_WORLD_SIZE": "1", "OMPI_COMM_WORLD_RANK": "0"})
    assert r.returncode == 0, r.stdout[-2000:] + "\n" + r.stderr[-4000:]
    assert "FEDERATED True" in r.stdout, r.stdout[-2000:]
    r = _import_in_subprocess(
        ["--grid", "tripole", "--enable-latlon-spmd", "--multicontrol"],
        {"OMPI_COMM_WORLD_SIZE": "1", "OMPI_COMM_WORLD_RANK": "0"})
    assert r.returncode == 0, r.stderr[-4000:]
    assert "FEDERATED False" in r.stdout, r.stdout[-2000:]


@pytest.mark.timeout(700)
def test_without_flag_import_does_not_federate():
    r = _import_in_subprocess(["--grid", "tripole"], {})
    assert r.returncode == 0, r.stderr[-4000:]
    assert "FEDERATED False" in r.stdout, r.stdout[-2000:]
