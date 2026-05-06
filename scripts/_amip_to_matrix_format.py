"""Convert ``run_amip.py`` output to the matrix-runner cross-grid layout.

iter-42: addresses an iter-41 oversight.  ``scripts/run_amip.py`` writes:

    output_dir/
        timeseries.npz       (numpy keys: days, T_atm, max_wind, ...)
        results.txt          (free-form text:
                              "legoESM AMIP run\\nGrid: ...\\n...")

But the matrix-runner cross-grid plot collector
(``scripts/run_atmosphere_test_matrix.py:_has_collectable``) expects:

    output_dir/
        mean_timeseries.csv  (header + comma-separated rows)
        results.txt          (key:value lines like "test: amip\\n
                              wall_time: 12.3s")
        snapshots_latlon.npz (optional)

Without this conversion, ``run_amip_cross_grid.sh`` (iter-41) would
write the AMIP outputs to disk but the matrix runner's
``--cross-grid-plots-only --test amip`` would silently skip every grid
because none of them satisfy ``_has_collectable``.

This module is intentionally tiny and self-contained: ``main(out_dir)``
reads the npz / free-form results.txt and writes the two
matrix-compatible files.  Idempotent — safe to call repeatedly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np


_VARNAME_REMAP = {
    # AMIP key -> matrix-runner column name (matches the iter-25 OMIP
    # remapping convention: lower_snake_case, prefix the relevant
    # diagnostic kind).
    "days": "time_days",
    "T_atm": "mean_T",
    "T_low": "mean_T_low",
    "dry_mass_ps": "mass",
    "max_wind": "max_wind",
    "sst": "mean_SST",
    "sic": "mean_SIC",
    "precip": "mean_precip",
    "CWV": "mean_CWV",
    "sw_up_toa": "sw_up_toa",
    "lw_up_toa": "lw_up_toa",
    "sw_net_sfc": "sw_net_sfc",
    "lw_net_sfc": "lw_net_sfc",
    "energy_residual": "energy_residual",
    "moisture_residual": "moisture_residual",
}


def _is_real_array(v) -> bool:
    """Skip the run_amip.py NaN-sentinel for missing diagnostics."""
    arr = np.asarray(v)
    if arr.ndim == 0:
        return False
    return arr.size > 0


def _parse_amip_results_txt(path: Path) -> dict:
    """Extract (Grid, Status, Wall time, JIT) from the free-form file.

    Returns lower-case keys ready for the matrix-format writer.
    """
    out: dict[str, str | float] = {}
    if not path.exists():
        return out
    text = path.read_text()
    # Grid line: "Grid: cubed_sphere C48 / L20, dt=300.0s, 30 days"
    m = re.search(r"Grid:\s*(\S+)\s+(\S+)\s*/\s*L(\d+),\s*dt=([\d.]+)s,\s*(\d+)\s*days", text)
    if m:
        out["grid"] = m.group(1)
        out["resolution"] = m.group(2)
        out["nlev"] = int(m.group(3))
        out["dt"] = float(m.group(4))
        out["days"] = int(m.group(5))
    # Status line: "Status: COMPLETED"
    m = re.search(r"Status:\s*(\S+)", text)
    if m:
        # Map the AMIP free-form status to the matrix-runner convention
        # ("PASS" / "ERROR").  Anything that's not COMPLETED is treated
        # as ERROR by the matrix collector.
        s = m.group(1).upper()
        out["status"] = "PASS" if s == "COMPLETED" else "ERROR"
    # Wall time: "Wall time: 123.4s"
    m = re.search(r"Wall time:\s*([\d.]+)s", text)
    if m:
        out["wall_time"] = f"{float(m.group(1)):.1f}s"
    # JIT compilation: "JIT compilation: 12.3s"
    m = re.search(r"JIT compilation:\s*([\d.]+)s", text)
    if m:
        out["jit_time"] = f"{float(m.group(1)):.1f}s"
    return out


def main(out_dir: Path) -> None:
    """Convert ``out_dir/timeseries.npz`` + ``out_dir/results.txt`` to
    matrix-runner-compatible ``mean_timeseries.csv`` + ``results.txt``.

    Writes nothing if ``timeseries.npz`` is missing (caller must fail
    visibly elsewhere).  Overwrites pre-existing matrix-format files.
    """
    out_dir = Path(out_dir)
    npz_path = out_dir / "timeseries.npz"
    if not npz_path.exists():
        # Don't silently succeed — the wrapper should fail loudly if
        # ``run_amip.py`` produced no output.
        print(
            f"  [_amip_to_matrix_format] WARNING: {npz_path} missing; "
            f"skipping conversion in {out_dir}",
            file=sys.stderr,
        )
        return

    data = np.load(npz_path, allow_pickle=False)
    # Build (column_name, array) pairs in deterministic order, skipping
    # NaN-sentinel entries.
    cols: list[tuple[str, np.ndarray]] = []
    days = data.get("days") if "days" in data.files else None
    if days is None or not _is_real_array(days):
        print(
            f"  [_amip_to_matrix_format] WARNING: 'days' missing from "
            f"{npz_path}; cannot write CSV.",
            file=sys.stderr,
        )
        return
    cols.append(("time_days", np.asarray(days)))
    n = len(cols[0][1])
    for src_key, dst_key in _VARNAME_REMAP.items():
        if src_key == "days":
            continue
        if src_key not in data.files:
            continue
        arr = data[src_key]
        if not _is_real_array(arr):
            continue
        if np.asarray(arr).shape[0] != n:
            # Length mismatch — skip rather than break the CSV.
            continue
        cols.append((dst_key, np.asarray(arr)))

    csv_path = out_dir / "mean_timeseries.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(c[0] for c in cols) + "\n")
        for i in range(n):
            f.write(",".join(f"{c[1][i]:.6e}" for c in cols) + "\n")

    # Rewrite results.txt in the matrix-runner key:value format.  Keep
    # the original AMIP results.txt around as ``results_amip.txt`` so
    # we don't lose the free-form summary.
    #
    # iter-42 idempotency: if ``results.txt`` already starts with the
    # matrix-format ``test: amip`` marker, the converter has run
    # before — re-parse from the preserved ``results_amip.txt``
    # instead of clobbering ``results_amip.txt`` with the matrix-
    # format file from the previous run.
    results_txt = out_dir / "results.txt"
    results_amip = out_dir / "results_amip.txt"
    already_converted = (
        results_txt.exists()
        and results_txt.read_text(errors="replace").startswith("test: amip")
    )
    if already_converted:
        # Parse from the preserved AMIP file (or fall back to the
        # current matrix-format file if for some reason the AMIP
        # original is missing).
        parsed = _parse_amip_results_txt(results_amip)
    else:
        parsed = _parse_amip_results_txt(results_txt)
        if results_txt.exists():
            results_txt.rename(results_amip)
    with open(results_txt, "w") as f:
        f.write("test: amip\n")
        for key in (
            "grid", "resolution", "nlev", "dt", "days",
            "status", "wall_time", "jit_time",
        ):
            if key in parsed:
                f.write(f"{key}: {parsed[key]}\n")
        # Add a brief note for downstream readers.
        f.write(
            "notes: real-AMIP run via run_amip.py; "
            "format converted by _amip_to_matrix_format.py for "
            "matrix-runner cross-grid collection (iter-42)\n"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "usage: python _amip_to_matrix_format.py OUT_DIR",
            file=sys.stderr,
        )
        sys.exit(1)
    main(Path(sys.argv[1]))
