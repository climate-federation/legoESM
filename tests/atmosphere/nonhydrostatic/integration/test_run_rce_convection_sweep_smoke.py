"""Smoke test for ``scripts/run_rce_convection_sweep.py``.

iter-273: ``run_rce_convection_sweep.py`` sweeps cumulus convection
schemes (Tiedtke, Zhang-McFarlane, Emanuel, Bechtold, Kuo,
Kain-Fritsch, simple mass-flux) on a fixed-SST lat-lon FV column
with gray radiation + bulk BL + warm-rain microphysics. Pre
iter-273 it had ZERO test coverage.

iter-273 adds a single-scheme single-day smoke covering argparse
+ factory dispatch (one scheme: ``kuo`` — fastest of the 7) +
column physics composition + summary-table stdout + per-scheme
snapshot .npz emission.

9th previously-untested CRM script in the iter-238..273 coverage
chain.
"""
from __future__ import annotations

import re

from tests.atmosphere.nonhydrostatic.integration._bench_smoke_helpers import (
    REPO_ROOT, fail_on_nonzero, run_bench,
)


SCRIPT = REPO_ROOT / "scripts" / "run_rce_convection_sweep.py"


# Summary table line: "  kuo    1.0   300.00   292.27     0.000   73.88     0.60      0.9  OK"
_SUMMARY_RE = re.compile(
    r"^\s*(?P<scheme>\S+)\s+"
    r"(?P<day>[\d.]+)\s+"
    r"(?P<t_sfc>[\d.]+)\s+"
    r"(?P<t_atm>[\d.]+)\s+"
    r"(?P<precip>[\d.eE+-]+)\s+"
    r"(?P<cwv>[\d.]+)\s+"
    r"(?P<max_v>[\d.]+)\s+"
    r"(?P<wall_s>[\d.]+)\s+"
    r"(?P<status>\S+)\s*$"
)


def test_run_rce_convection_sweep_kuo_1day_smoke(tmp_path):
    """1-day kuo-only RCE column smoke at N=6 / nlev=10 / dt=600 s.
    Verifies argparse + factory dispatch + column physics stack +
    summary table emission. kuo is the fastest of the 7 schemes
    (~1 s wall on M5 Pro at this mesh).
    """
    out_dir = tmp_path / "sweep"
    result = run_bench(SCRIPT, [
        "--schemes", "kuo",
        "--days", "1",
        "--N", "6",
        "--nlev", "10",
        "--dt", "600.0",
        "--output", str(out_dir),
    ])
    fail_on_nonzero(result, "run_rce_convection_sweep.py")

    # The snapshot file should land at <output_dir>/snapshot_kuo.npz.
    snapshot = out_dir / "snapshot_kuo.npz"
    assert snapshot.exists(), (
        f"Per-scheme snapshot not written at {snapshot}.\n"
        f"stdout tail:\n{result.stdout[-1500:]}"
    )

    # Find the summary table row for kuo.
    summary_match = None
    in_summary = False
    for line in result.stdout.splitlines():
        if "SUMMARY" in line:
            in_summary = True
            continue
        if in_summary and line.strip().startswith("kuo"):
            summary_match = _SUMMARY_RE.match(line)
            break
    assert summary_match is not None, (
        f"Summary table row for 'kuo' not found or failed regex.\n"
        f"stdout tail:\n{result.stdout[-1500:]}"
    )

    # Schema lock: scheme name + status.
    assert summary_match.group("scheme") == "kuo"
    assert summary_match.group("status") == "OK", (
        f"kuo run did not finish OK: status="
        f"{summary_match.group('status')!r}"
    )

    # Numeric sanity: at fixed SST=300 K + Wing IC + 1 day, the
    # column relaxes only slightly toward equilibrium.
    day = float(summary_match.group("day"))
    t_sfc = float(summary_match.group("t_sfc"))
    t_atm = float(summary_match.group("t_atm"))
    precip = float(summary_match.group("precip"))
    cwv = float(summary_match.group("cwv"))
    max_v = float(summary_match.group("max_v"))
    wall_s = float(summary_match.group("wall_s"))

    assert abs(day - 1.0) < 1e-3, (
        f"day={day} != 1.0 — sweep didn't reach --days target."
    )
    # SST is FIXED to --sst-init=300.0 (script default); should
    # not drift across the run.
    assert abs(t_sfc - 300.0) < 1e-2, (
        f"T_sfc={t_sfc} drifted from --sst-init=300 (fixed by "
        f"design); SST-relaxation regression?"
    )
    # T_atm at 1 day is still well below T_sfc=300 K (the column
    # needs many days to relax). Empirical at this config: 292.27 K.
    assert 280.0 < t_atm < 300.0, (
        f"T_atm={t_atm} outside (280, 300) K range at 1 sim-day."
    )
    # CWV in moist tropical column at SST=300 K: ~50-80 kg/m^2.
    assert 50.0 < cwv < 100.0, (
        f"CWV={cwv} outside (50, 100) kg/m^2 — moist tropical "
        f"column expected ~70 at SST=300 K."
    )
    # Precip non-negative + finite.
    assert precip >= 0.0
    # max|v| from BL + convection: positive, sub-50 m/s.
    assert 0.0 < max_v < 50.0, (
        f"max|v|={max_v} outside (0, 50) m/s — likely a dycore "
        f"or BL regression."
    )
    # Wall positive (no-op floor — kuo on 6x6x10 at 1 sim-day
    # takes ~1 s; floor at 0.01 s catches a fully-elided run).
    assert wall_s > 0.01, (
        f"wall_s={wall_s} below 10 ms floor; the dycore or "
        f"column physics may have no-op'd."
    )
