"""Smoke test for ``scripts/bench_plane_dycore.py``.

iter-259: sixth (and last) script in the iter-252..258
untested-bench coverage chain. ``bench_plane_dycore.py`` is the
single-rank plane NH dycore throughput micro-benchmark — JIT
compile time + per-step wall + cells/sec + SYPD (simulated years
per wall day). NO MPI surface.

Pre iter-259 it had ZERO test coverage; iter-259 adds a smoke
covering argparse + JIT compile + bench loop + stdout schema.
"""
from __future__ import annotations

import re

from tests.atmosphere.nonhydrostatic.integration._bench_smoke_helpers import (
    REPO_ROOT, fail_on_nonzero, run_bench,
)


SCRIPT = REPO_ROOT / "scripts" / "bench_plane_dycore.py"


# Stdout snippets the bench emits (literal substrings):
#   "JIT compile time (first step): X.YZ s"
#   "    M.MMM ms/step"
#   "  NNN.NN steps/s"
#   "    K.KK M cells/s"
#   "  E.EE+EE SYPD ..."
_MS_RE = re.compile(r"([\d.]+)\s+ms/step")
_STEPS_PER_S_RE = re.compile(r"([\d.]+)\s+steps/s")
_CELLS_PER_S_RE = re.compile(r"([\d.]+)\s+M cells/s")
_JIT_RE = re.compile(r"JIT compile time \(first step\):\s+([\d.]+)\s+s")


def test_bench_plane_dycore_single_rank_smoke():
    """Single-rank smoke covering argparse + JIT compile + bench
    loop + the throughput-summary stdout schema.
    """
    result = run_bench(SCRIPT, [
        "--nx", "12", "--ny", "12", "--nlev", "8",
        "--dt", "1.0",
        "--n-warmup", "1",
        "--n-bench", "3",
    ])
    fail_on_nonzero(result, "bench_plane_dycore.py")

    stdout = result.stdout
    # Each summary line present + extractable.
    jit_match = _JIT_RE.search(stdout)
    ms_match = _MS_RE.search(stdout)
    sps_match = _STEPS_PER_S_RE.search(stdout)
    cps_match = _CELLS_PER_S_RE.search(stdout)
    for name, mat in [
        ("JIT compile time", jit_match),
        ("ms/step", ms_match),
        ("steps/s", sps_match),
        ("M cells/s", cps_match),
    ]:
        assert mat is not None, (
            f"Stdout summary missing '{name}' line. "
            f"stdout tail:\n{stdout[-1500:]}"
        )
    jit_s = float(jit_match.group(1))
    ms_per_step = float(ms_match.group(1))
    steps_per_s = float(sps_match.group(1))
    m_cells_per_s = float(cps_match.group(1))

    # JIT compile takes 0.5-5 s on M5 Pro for this small mesh.
    # Floor at 0.01 s catches a regression where JIT didn't fire
    # (model.step ran in eager Python). Cap at 60 s catches a
    # pathological compile.
    assert 0.01 < jit_s < 60.0, (
        f"JIT compile time {jit_s} s outside (0.01, 60) range; "
        f"likely a JIT regression or pathological lowering."
    )

    # iter-259 no-op detector (mirrors iter-256 HIGH#1):
    # Real bench on 12x12x8 takes ~1-3 ms/step; an elided
    # model.step would be <1 us = 0.001 ms.
    assert ms_per_step > 1.0e-2, (
        f"ms_per_step={ms_per_step} below 10us floor; "
        f"model.step likely no-op'd."
    )

    # Throughput consistency (steps_per_s ≈ 1000/ms_per_step).
    expected_sps = 1000.0 / ms_per_step
    rel_err = abs(steps_per_s - expected_sps) / expected_sps
    assert rel_err < 0.10, (
        f"steps_per_s={steps_per_s} inconsistent with 1000/ms="
        f"{expected_sps}: rel_err={rel_err:.3f}."
    )

    # cells/s consistency (12*12*8 = 1152 cells per step).
    expected_cells_per_s = 1152.0 * steps_per_s / 1.0e6
    rel_err_cells = abs(m_cells_per_s - expected_cells_per_s) / expected_cells_per_s
    assert rel_err_cells < 0.10, (
        f"M cells/s={m_cells_per_s} inconsistent with "
        f"1152*steps/1e6={expected_cells_per_s}: "
        f"rel_err={rel_err_cells:.3f}."
    )
