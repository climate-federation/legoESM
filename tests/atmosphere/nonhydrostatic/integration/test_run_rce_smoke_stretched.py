"""Smoke test for ``scripts/run_rce_smoke_stretched.py``.

iter-271: ``run_rce_smoke_stretched.py`` is the RCE smoke harness
for the full PR #296-#305 RCE stack on the plane CRM
(stretched HeightCoordinate + Wing IC + tracer positivity + CFL
diag + mean-wind removal + moist-mass fixer + RCE surface flux +
RCE diagnostics: CWV, MSE, cloud-fraction, precip proxy).

Pre iter-271 it had ZERO test coverage. iter-271 adds a single-
process smoke covering argparse + the full RCE stack composition
+ the per-step output line schema.

Closes 8th previously-untested scripts/ path in the
iter-238..271 chain.
"""
from __future__ import annotations

import re

from tests.atmosphere.nonhydrostatic.integration._bench_smoke_helpers import (
    REPO_ROOT, fail_on_nonzero, run_bench,
)


SCRIPT = REPO_ROOT / "scripts" / "run_rce_smoke_stretched.py"


# Output line format (whitespace-separated):
# step  t[s]  CWV_mean  MSE_mean  cloud_max  precip_max  C_h_max  C_v_max  C_a_max
_DATA_LINE_RE = re.compile(
    r"^\s*(?P<step>\d+)\s+"
    r"(?P<t_s>[\d.]+)\s+"
    r"(?P<cwv>[\d.eE+-]+)\s+"
    r"(?P<mse>[\d.eE+-]+)\s+"
    r"(?P<cloud_max>[\d.eE+-]+)\s+"
    r"(?P<precip_max>[\d.eE+-]+)\s+"
    r"(?P<ch_max>[\d.eE+-]+)\s+"
    r"(?P<cv_max>[\d.eE+-]+)\s+"
    r"(?P<ca_max>[\d.eE+-]+)\s*$"
)


def test_run_rce_smoke_stretched_3_step_smoke(tmp_path):
    """3-step end-to-end RCE smoke at tiny mesh (6x6x12, dx=4 km,
    dt=1 s + n_acoustic=24 to satisfy the script's per-substep
    acoustic CFL guard). Verifies the full RCE stack composes
    cleanly + emits the documented diagnostic columns.
    """
    out_file = tmp_path / "rce_stretched_smoke.txt"
    result = run_bench(SCRIPT, [
        "--nx", "6", "--ny", "6", "--nlev", "12",
        "--dx", "4000.0", "--dt", "1.0",
        "--n-acoustic-substeps", "24",
        "--steps", "3",
        "--output", str(out_file),
    ])
    fail_on_nonzero(result, "run_rce_smoke_stretched.py")
    assert out_file.exists(), (
        f"Script did not write output at {out_file}.\n"
        f"stdout tail:\n{result.stdout[-500:]}"
    )
    lines = out_file.read_text().splitlines()
    # 2 header (#-prefixed) lines + 3 data lines.
    header_lines = [L for L in lines if L.lstrip().startswith("#")]
    data_lines = [L for L in lines if not L.lstrip().startswith("#") and L.strip()]
    assert len(header_lines) == 2, (
        f"Expected exactly 2 header lines, got {len(header_lines)}.\n"
        f"lines={lines!r}"
    )
    assert len(data_lines) == 3, (
        f"Expected exactly 3 data lines (steps), got "
        f"{len(data_lines)}.\nlines={lines!r}"
    )
    # Header content lock.
    assert "RCE smoke" in header_lines[0]
    assert all(col in header_lines[1] for col in (
        "CWV_mean", "MSE_mean", "cloud_max", "precip_max",
        "C_h_max", "C_v_max", "C_a_max",
    )), (
        f"Header column line missing expected columns: "
        f"{header_lines[1]!r}"
    )
    # Per-step data line parse.
    parsed = []
    for line in data_lines:
        m = _DATA_LINE_RE.match(line)
        assert m is not None, (
            f"Data line failed regex: {line!r}\n"
            f"Regex: {_DATA_LINE_RE.pattern}"
        )
        parsed.append({k: float(v) for k, v in m.groupdict().items()})
    # Step indices 0, 1, 2.
    assert [int(p["step"]) for p in parsed] == [0, 1, 2], (
        f"Step sequence drift: {[int(p['step']) for p in parsed]}"
    )
    # t[s] = step+1 (1.0, 2.0, 3.0 at dt=1.0).
    for i, p in enumerate(parsed):
        assert abs(p["t_s"] - (i + 1) * 1.0) < 1e-6, (
            f"t[s] mismatch at step {i}: got {p['t_s']}, "
            f"expected {(i+1) * 1.0}"
        )
    # CWV_mean must be positive + ~Wing-2018-IC for 6x6x12 at
    # 50m surface dz (script's default vertical grid). Empirical
    # measurement: ~6.15e+01 kg/m^2.
    for p in parsed:
        assert 1.0 < p["cwv"] < 200.0, (
            f"CWV_mean={p['cwv']} outside (1, 200) kg/m^2 range; "
            f"Wing 2018 IC expected ~60 kg/m^2."
        )
    # MSE_mean positive + finite.
    for p in parsed:
        assert 1e9 < p["mse"] < 1e10, (
            f"MSE_mean={p['mse']} outside (1e9, 1e10) J/m^2 range."
        )
    # Acoustic Courant C_a_max < 1.0 (script's --acoustic-cfl-max
    # default). Catches a regression where the per-substep CFL
    # guard mis-fires.
    for p in parsed:
        assert p["ca_max"] < 1.0, (
            f"C_a_max={p['ca_max']} exceeds 1.0 — per-substep "
            f"acoustic CFL would have raised the script's "
            f"RuntimeError guard."
        )
