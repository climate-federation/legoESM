"""Route-B sweeps must fan out over PRECISION, not just for the cube.

Before this, ``PRECISION`` was passed ONLY to the cubed-sphere branch
(``run_cpu_mpi_scaling --cs-spmd``); the latlon and icosahedral SPMD benches
never received it, so every latlon/ico row in the route-B campaign came out
float32 with no way to request float64.  Those benches take no ``--precision``
flag at all -- they read ``jax.config.jax_enable_x64`` and RECORD the result --
so the knob for them is the ``JAX_ENABLE_X64`` env var.

These tests EXECUTE the dispatch block extracted from each PBS script rather
than merely grepping for it, so a broken mapping (f64 silently running f32) or
a deleted unknown-value guard fails here.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_LANES = Path(__file__).resolve().parents[2] / "scripts" / "cluster" / "scaling_derecho"
_SWEEPS = ("routeb_gpu_sweep.pbs", "routeb_cpu_sweep.pbs")


def _dispatch_block(script: Path) -> str:
    """The `case "$PREC" in ... esac` block, as runnable shell."""
    src = script.read_text()
    m = re.search(r'(case\s+"\$PREC"\s+in.*?esac)', src, re.S)
    assert m, f"{script.name}: no `case \"$PREC\"` precision dispatch found"
    return m.group(1)


def _run(script: Path, prec: str) -> tuple[int, str]:
    """Run the dispatch for one PREC value -> (returncode, JAX_ENABLE_X64)."""
    block = _dispatch_block(script)
    proc = subprocess.run(
        ["bash", "-c", f'PREC="{prec}"\n{block}\necho "X64=${{JAX_ENABLE_X64:-unset}}"'],
        capture_output=True, text=True,
    )
    val = ""
    for line in proc.stdout.splitlines():
        if line.startswith("X64="):
            val = line.split("=", 1)[1]
    return proc.returncode, val


@pytest.mark.parametrize("script", _SWEEPS)
@pytest.mark.parametrize("prec,expected", [
    ("float32", "0"), ("fp32", "0"),
    ("float64", "1"), ("fp64", "1"),
])
def test_precision_maps_to_x64(script, prec, expected):
    rc, val = _run(_LANES / script, prec)
    assert rc == 0, f"{script}: PREC={prec} exited {rc}"
    assert val == expected, (
        f"{script}: PREC={prec} set JAX_ENABLE_X64={val}, expected {expected} "
        f"-- a wrong mapping runs one precision and labels it the other")


@pytest.mark.parametrize("script", _SWEEPS)
def test_unknown_precision_is_a_hard_error(script):
    """Dispatch hardening: a typo must abort, never silently run float32 and
    get recorded as whatever the user asked for."""
    rc, _ = _run(_LANES / script, "flaot64")
    assert rc != 0, (
        f"{script}: an unknown PRECISION exited 0 -- a typo would silently run "
        f"the default precision under the wrong label")


@pytest.mark.parametrize("script", _SWEEPS)
def test_precision_is_fanned_out_and_tagged(script):
    """PRECISIONS must be a list the loop iterates, and the per-grid output
    paths must carry ${PREC} -- otherwise an f64 rung overwrites the f32 rung
    of the same (res, devices) and the ladder silently loses half its rows."""
    src = (_LANES / script).read_text()
    assert "PRECISIONS=" in src, f"{script}: no PRECISIONS list"
    assert re.search(r"for\s+PREC\s+in\s+\$PRECISIONS", src), \
        f"{script}: PRECISIONS is never iterated"
    # latlon + ico JSONL outputs must be precision-tagged
    outs = re.findall(r'--out\s+"([^"]*\.jsonl)"', src)
    assert outs, f"{script}: no --out jsonl paths found"
    for o in outs:
        assert "${PREC}" in o, f"{script}: output path {o} lacks ${{PREC}} -> f32/f64 collide"
    # the cube driver takes the flag directly; it must use the loop variable
    assert '--precision "$PREC"' in src, \
        f"{script}: cube branch does not pass the loop's $PREC"
    assert '--precision "$PRECISION"' not in src, \
        f"{script}: cube branch still uses the pre-fan-out $PRECISION"


@pytest.mark.parametrize("script", _SWEEPS)
def test_default_precision_preserves_previous_behaviour(script):
    """Default must stay float32: these sweeps are ONE job that loops
    internally, so silently defaulting to two precisions would double the
    walltime and risk losing the whole job to the PBS limit."""
    src = (_LANES / script).read_text()
    m = re.search(r'PRECISIONS="\$\{PRECISIONS:-\$\{PRECISION:-([^}]*)\}\}"', src)
    assert m, f"{script}: PRECISIONS default not in the expected form"
    assert m.group(1).strip() == "float32", \
        f"{script}: default precision is {m.group(1)!r}, expected float32"
