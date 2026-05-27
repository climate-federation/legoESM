"""Smoke test for ``scripts/run_rce_convection_sweep.py``.

iter-273: ``run_rce_convection_sweep.py`` sweeps 8 cumulus
convection schemes (sbm, tiedtke, zhang_mcfarlane, emanuel,
bechtold, kuo, kain_fritsch, mass_flux) on a fixed-SST lat-lon
FV column with gray radiation + bulk BL + warm-rain microphysics.
Pre iter-273 it had ZERO test coverage.

iter-274 (Codex iter-273 round-1 HIGH#1 + MEDIUM#1) hardening:
the smoke now uses ``--schemes kuo,mass_flux`` (CSV) instead of
``--schemes kuo`` so we hit:
* Two distinct factory branches (kuo's diluted plume + mass_flux's
  shared kernel). A regression in only the mass-flux side would
  have passed the iter-273 single-scheme smoke.
* The argparse CSV-split path at scripts/run_rce_convection_sweep.py
  _parse_schemes — which was advertised in the script's CLI docs
  but never exercised in the iter-273 smoke.

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


def test_run_rce_convection_sweep_csv_two_schemes_1day_smoke(tmp_path):
    """1-day CSV-list RCE column smoke at N=6 / nlev=10 / dt=600 s
    running BOTH kuo + mass_flux schemes. Verifies argparse + CSV
    split + 2 distinct factory branches + column physics stack +
    summary table emission + per-scheme snapshot .npz output.

    kuo is the fastest single-scheme of the 8; mass_flux exercises
    the shared mass-flux kernel (used by tiedtke, zhang_mcfarlane,
    emanuel, bechtold, kain_fritsch too). Together: ~2 s wall on
    M5 Pro at this mesh.
    """
    out_dir = tmp_path / "sweep"
    result = run_bench(SCRIPT, [
        "--schemes", "kuo,mass_flux",
        "--days", "1",
        "--N", "6",
        "--nlev", "10",
        "--dt", "600.0",
        "--output", str(out_dir),
    ])
    fail_on_nonzero(result, "run_rce_convection_sweep.py")

    # Both per-scheme snapshot files should land in <output_dir>.
    for scheme in ("kuo", "mass_flux"):
        snapshot = out_dir / f"snapshot_{scheme}.npz"
        assert snapshot.exists(), (
            f"Per-scheme snapshot not written at {snapshot}.\n"
            f"stdout tail:\n{result.stdout[-1500:]}"
        )

    # Find summary rows for BOTH schemes.
    summary_matches: dict[str, re.Match] = {}
    in_summary = False
    for line in result.stdout.splitlines():
        if "SUMMARY" in line:
            in_summary = True
            continue
        if in_summary:
            stripped = line.strip()
            for scheme in ("kuo", "mass_flux"):
                if stripped.startswith(scheme + " ") or stripped.startswith(scheme + "\t"):
                    m = _SUMMARY_RE.match(line)
                    if m is not None:
                        summary_matches[scheme] = m
    assert set(summary_matches) == {"kuo", "mass_flux"}, (
        f"Expected summary rows for both kuo + mass_flux; got "
        f"{list(summary_matches)!r}.\nstdout tail:\n{result.stdout[-1500:]}"
    )

    for scheme, m in summary_matches.items():
        assert m.group("scheme") == scheme
        assert m.group("status") == "OK", (
            f"{scheme} did not finish OK: status="
            f"{m.group('status')!r}"
        )

        day = float(m.group("day"))
        t_sfc = float(m.group("t_sfc"))
        t_atm = float(m.group("t_atm"))
        precip = float(m.group("precip"))
        cwv = float(m.group("cwv"))
        max_v = float(m.group("max_v"))
        wall_s = float(m.group("wall_s"))

        assert abs(day - 1.0) < 1e-3, (
            f"{scheme}: day={day} != 1.0."
        )
        # SST fixed by design.
        assert abs(t_sfc - 300.0) < 1e-2, (
            f"{scheme}: T_sfc={t_sfc} drifted from --sst-init=300."
        )
        # T_atm at 1 day: column not fully relaxed (~292 K kuo,
        # similar for mass_flux).
        assert 280.0 < t_atm < 300.0, (
            f"{scheme}: T_atm={t_atm} outside (280, 300) K."
        )
        # iter-274 (Codex iter-273 round-1 LOW#1): tightened CWV
        # bound from (50, 100) to (65, 85) — anchored to empirical
        # ~73.88 baseline. Catches deterministic drift that the
        # wide iter-273 bound would have missed.
        assert 65.0 < cwv < 85.0, (
            f"{scheme}: CWV={cwv} outside (65, 85) kg/m^2 — "
            f"empirical baseline ~73.88; >10% drift indicates "
            f"a regression."
        )
        assert precip >= 0.0
        assert 0.0 < max_v < 50.0, (
            f"{scheme}: max|v|={max_v} outside (0, 50) m/s."
        )
        # iter-274 (Codex iter-273 round-1 MEDIUM#2): the script
        # rounds wall_s to 1 decimal in the summary, so the
        # effective gate is "prints at least 0.1 s". Loosened
        # accordingly — 0.1 s catches a fully-elided run while
        # tolerating fast hardware.
        assert wall_s >= 0.1, (
            f"{scheme}: wall_s={wall_s} below 0.1 s floor "
            f"(rounded-to-1-decimal in summary)."
        )
