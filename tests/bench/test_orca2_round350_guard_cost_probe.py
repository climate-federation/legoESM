"""The guards-off probe must reach every guard and refuse when one escaped the swap."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROBE = Path(__file__).resolve().parents[2] / "scripts/bench/orca2_round350_guard_cost_probe.py"
CALLER = "legoesm.ocean.eos"


def _run(body: str) -> str:
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('probe', {str(PROBE)!r})\n"
        "probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)\n" + body
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_swap_before_import_reaches_callers():
    out = _run(
        "def check():\n"
        "    original, calls = probe.install_passthrough_guard()\n"
        f"    import {CALLER} as caller\n"
        "    import jax.numpy as jnp\n"
        "    x = jnp.asarray(1.5)\n"
        "    assert caller.nemo_source_round(x) is x\n"
        "    import jax\n"
        "    jax.jit(caller.nemo_source_round)(x)\n"
        "    print(calls['eager'], calls['traced'], probe.stale_bindings(original))\n"
        "check()\n"
    )
    assert out == "1 1 []"


def test_caller_imported_before_swap_is_reported():
    out = _run(
        "def check():\n"
        f"    import {CALLER}\n"
        "    original, calls = probe.install_passthrough_guard()\n"
        "    print(probe.stale_bindings(original))\n"
        "check()\n"
    )
    assert CALLER in out


def test_step_count_only_inside_the_compiled_step():
    out = _run(
        "def check():\n"
        "    original, calls = probe.install_passthrough_guard()\n"
        f"    import {CALLER} as caller\n"
        "    import jax, jax.numpy as jnp\n"
        "    x = jnp.asarray(1.5)\n"
        "    jax.jit(caller.nemo_source_round)(x)\n"
        "    def _step_jitted(v):\n"
        "        return caller.nemo_source_round(v)\n"
        "    jax.jit(_step_jitted)(x)\n"
        "    print(calls['traced'], calls['step'])\n"
        "check()\n"
    )
    assert out == "2 1"
