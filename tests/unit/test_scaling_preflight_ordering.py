"""#1361 / PR #1376: the preflight must decide BEFORE JAX is imported.

The PR claimed parse-time ordering for all three benches; codex found it held
only for the cube one, because the atm and ocean benches imported `jax` at
module load. These tests pin the property mechanically instead of by claim:
importing a bench module must not pull in JAX, and a rejected config must exit
without ever importing it.
"""

from __future__ import annotations

import os
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


def test_ocean_rejected_config_exits_before_jax_is_imported():
    """Ocean gets its own rejected-config run, not just import coverage."""
    bench = REPO / BENCHES[1]
    code = (
        "import sys, runpy\n"
        "sys.argv = ['bench', '--mode', 'strong', '--n-lat', '720', "
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
    assert "JAX_IMPORTED False" in proc.stdout, proc.stdout + proc.stderr


def test_cube_rejected_memory_exits_before_jax_is_imported(monkeypatch):
    """Cube path: n_devices is derived (6*kt^2) so the device-count gate can
    never fire from the CLI — the memory gate is the one a user hits. This is
    also the LEGOESM_DEVICE_HBM default path: no --device-hbm flag is passed.
    """
    bench = REPO / BENCHES[2]
    code = (
        "import sys, runpy\n"
        "sys.argv = ['bench', '--resolution', '1152', '--nlev', '60', "
        "'--kt', '3', '--steps', '3']\n"
        "try:\n"
        f"    runpy.run_path(r'{bench}', run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    print('EXIT', e.code)\n"
        "    print('JAX_IMPORTED', 'jax' in sys.modules)\n"
    )
    env = {**os.environ, "LEGOESM_DEVICE_HBM": "a100-40"}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=300, cwd=REPO, env=env)
    out = proc.stdout + proc.stderr
    assert "exceeds" in out, out
    assert "JAX_IMPORTED False" in proc.stdout, out


def test_nearest_divisible_is_actually_nearest():
    """Regression: (190, 64) used to omit 256, which is nearer than 64."""
    from legoesm.scaling_preflight import nearest_divisible
    assert nearest_divisible(190, 64) == [192, 128, 256]
    assert nearest_divisible(720, 64) == [704, 768, 640]
    for v in nearest_divisible(1000, 7):
        assert v % 7 == 0 and v > 0
