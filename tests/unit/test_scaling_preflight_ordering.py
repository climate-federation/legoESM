"""#1361 / PR #1376: the preflight must decide BEFORE JAX is imported.

The PR claimed parse-time ordering for all three benches; codex found it held
only for the cube one, because the atm and ocean benches imported `jax` at
module load. These tests pin the property mechanically instead of by claim:
importing a bench module must not pull in JAX, and a rejected config must exit
without ever importing it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BENCHES = [
    "scripts/bench/bench_atm_latlon_spmd_scaling.py",
    "scripts/bench/bench_ocean_latlon_spmd_scaling.py",
    "scripts/bench/bench_cube_tiled_step_scaling.py",
]


@pytest.mark.parametrize("bench", BENCHES)
def test_importing_the_bench_does_not_import_jax(bench):
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('b', r'{REPO / bench}')\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['b'] = m\n"
        "spec.loader.exec_module(m)\n"
        "assert 'jax' not in sys.modules, sorted(k for k in sys.modules if k.startswith('jax'))\n"
        "print('NO_JAX')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=300, cwd=REPO)
    assert "NO_JAX" in proc.stdout, (
        f"{bench} imported JAX at module load — the #1361 preflight can no "
        f"longer reject a config before the driver is touched.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def test_rejected_config_exits_before_jax_is_imported():
    """A non-divisible atm config must SystemExit with jax still unimported."""
    bench = REPO / BENCHES[0]
    code = (
        "import sys, runpy\n"
        f"sys.argv = ['bench', '--mode', 'strong', '--n-lat', '720', "
        "'--n-lon', '1440', '--n-devices', '64', '--nlev', '10']\n"
        "try:\n"
        f"    runpy.run_path(r'{bench}', run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    print('EXIT', e.code)\n"
        "    print('JAX_IMPORTED', 'jax' in sys.modules)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=300, cwd=REPO)
    assert "not divisible" in (proc.stdout + proc.stderr), proc.stdout + proc.stderr
    assert "JAX_IMPORTED False" in proc.stdout, (
        "the divisibility rejection happened only AFTER jax was imported:\n"
        + proc.stdout + proc.stderr)
