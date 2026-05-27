"""Smoke test for ``scripts/run_scm_rce.py``.

iter-300: 10th previously-untested CRM-adjacent script. SCM RCE
is a single-column radiative-convective equilibrium harness
(gray radiation + Louis turbulence + Kessler microphysics +
simple mass-flux convection). Pre iter-300 it had ZERO test
coverage.

This is the canonical single-column sanity test: under
heating/cooling balance, T_sfc should equilibrate near
prescribed initial T_sfc ≈ 300 K, T_top near gray-radiation
equilibrium ≈ 200 K. The script exits non-zero if final state
fails finite + coarse RCE checks. iter-302 (Codex iter-301
round-2 LOW): the actual gates in scripts/run_scm_rce.py are
``240 < T_sfc < 320`` K and ``150 < T_top < 270`` K — the
pre-iter-302 docstring incorrectly stated T_top bounds
[160, 260].

iter-300 smoke runs 0.5 sim-days (~72 outer steps at default
dt=600 s) at default --lat 0 — verifies the harness composes
+ produces sane day-half-day trajectory + emits the documented
``[result]`` + ``[OK] RCE sanity passed`` lines.
"""
from __future__ import annotations

import re

from tests.atmosphere.nonhydrostatic.integration._bench_smoke_helpers import (
    REPO_ROOT, fail_on_nonzero, run_bench,
)


SCRIPT = REPO_ROOT / "scripts" / "run_scm_rce.py"


_T_SFC_RE = re.compile(r"T_sfc_final\s*=\s*([\d.eE+-]+)\s*K")
_T_TOP_RE = re.compile(r"T_top_final\s*=\s*([\d.eE+-]+)\s*K")
_Q_V_SFC_RE = re.compile(r"q_v_sfc_final\s*=\s*([\d.eE+-]+)\s*g/kg")
_P_S_RE = re.compile(r"p_s_final\s*=\s*([\d.eE+-]+)\s*Pa")


def test_run_scm_rce_half_day_smoke():
    """0.5-sim-day SCM RCE smoke. Verifies the full single-column
    stack (radiation + Louis turbulence + Kessler microphysics +
    mass-flux convection) composes + emits the documented result
    schema + passes the script's coarse RCE sanity gates.
    """
    result = run_bench(SCRIPT, ["--days", "0.5"], timeout_s=300)
    fail_on_nonzero(result, "run_scm_rce.py")

    # Stdout schema: [SCM RCE] header, [result] lines, [OK] gate.
    assert "[SCM RCE]" in result.stdout, (
        f"Missing [SCM RCE] header. stdout:\n{result.stdout[-1500:]}"
    )
    assert "[OK] RCE sanity passed" in result.stdout, (
        f"Script's coarse RCE sanity gate did not pass. "
        f"stdout:\n{result.stdout[-1500:]}"
    )

    # Parse the 4 numeric [result] lines.
    matches = {}
    for name, regex in (
        ("T_sfc", _T_SFC_RE),
        ("T_top", _T_TOP_RE),
        ("q_v_sfc", _Q_V_SFC_RE),
        ("p_s", _P_S_RE),
    ):
        m = regex.search(result.stdout)
        assert m is not None, (
            f"[result] line for {name} not found.\n"
            f"stdout:\n{result.stdout[-1500:]}"
        )
        matches[name] = float(m.group(1))

    # T_sfc near IC (300 K) at 0.5 sim-days — the column has just
    # started relaxing. Empirical: 299.38 K. Tolerate ±10 K either
    # way (catches a regression that flips sign of net heating).
    assert 290.0 < matches["T_sfc"] < 310.0, (
        f"T_sfc_final={matches['T_sfc']} outside (290, 310) K. "
        f"Script's RCE sanity gate is [240, 320] K; this tighter "
        f"bound catches a regression while staying loose enough "
        f"for short-window dynamics."
    )

    # T_top at 200 K is the gray-radiation equilibrium target.
    # Empirical at 0.5 sim-days: 200.06 K (already very close).
    # Tolerate ±5 K.
    assert 195.0 < matches["T_top"] < 205.0, (
        f"T_top_final={matches['T_top']} outside (195, 205) K — "
        f"gray-radiation equilibrium target is 200 K."
    )

    # q_v_sfc near IC ≈ 18 g/kg. Tolerate ±3.
    assert 15.0 < matches["q_v_sfc"] < 21.0, (
        f"q_v_sfc_final={matches['q_v_sfc']} g/kg outside "
        f"(15, 21). IC ~18 g/kg; regression?"
    )

    # p_s exactly 100000 Pa (script uses fixed surface pressure).
    assert abs(matches["p_s"] - 100000.0) < 1.0, (
        f"p_s_final={matches['p_s']} Pa != 100000 ± 1 — "
        f"surface pressure should be constant."
    )

    # iter-301 (Codex iter-300 LOW): the prior comment claimed
    # "monotonic-ish" but the assertion only checks the trajectory
    # line is PRESENT in stdout. Reworded for accuracy.
    assert "T_sfc trajectory" in result.stdout, (
        f"[history] T_sfc trajectory line missing.\n"
        f"stdout:\n{result.stdout[-1500:]}"
    )
