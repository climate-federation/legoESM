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

iter-42 codex review tightened this module against several silent-
failure modes.  See inline comments tagged ``iter-42 codex``.

This module is intentionally tiny and self-contained: ``main(out_dir)``
reads the npz / free-form results.txt and writes the two
matrix-compatible files.  Idempotent — safe to call repeatedly.
"""

from __future__ import annotations

import os
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


# iter-42 codex MEDIUM: status whitelist.  ``run_amip.py`` writes
# "Status: COMPLETED" on success, but matrix-runner-compatible status
# parsers also exist for legacy run scripts that emit SUCCESS / PASS /
# OK.  Treat any of these as PASS so the cross-grid collector picks up
# successful runs from older / sister scripts; everything else → ERROR.
_PASS_STATUSES = frozenset({
    "COMPLETED", "COMPLETE", "SUCCESS", "PASS", "PASSED", "OK", "DONE",
})


def _is_real_array(v) -> bool:
    """Return True iff ``v`` looks like a usable per-step diagnostic.

    iter-42 codex MEDIUM: also reject 1-D all-NaN arrays.  ``run_amip.py``
    currently uses scalar ``np.float64(np.nan)`` as the missing-
    diagnostic sentinel, but a future change might emit a 1-D NaN array
    with the same length as the time axis.  Either way, we don't want a
    column of NaNs in the CSV.
    """
    arr = np.asarray(v)
    if arr.ndim == 0:
        return False
    if arr.size == 0:
        return False
    # Treat all-NaN arrays as sentinels too.
    if np.all(np.isnan(arr.astype(np.float64, copy=False))):
        return False
    return True


def _parse_amip_results_txt(path: Path) -> dict:
    """Extract (Grid, Status, Wall time, JIT) from the free-form file.

    Returns lower-case keys ready for the matrix-format writer.

    iter-42 codex MEDIUM: relax the regexes against the brittle
    iter-42 v1 patterns:

      * ``dt=``: accept scientific notation (``300.0`` and ``3e2``).
      * ``days``: accept singular / plural / decimal.
      * resolution: allow internal hyphens / colons (``T42``,
        ``90x180``, ``ico6``, ``C48-ext``).
      * nlev: optional — newer ``run_amip.py`` may not include it.
    """
    out: dict[str, str | float] = {}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    # Grid line: "Grid: cubed_sphere C48 / L20, dt=300.0s, 30 days"
    # iter-42 codex MEDIUM: relaxed regex.  The L<n> chunk and ``dt=``
    # chunk are optional; the day count accepts decimals and singular.
    m = re.search(
        r"Grid:\s*(\S+)\s+(\S+)"               # grid + resolution
        r"(?:\s*/\s*L(\d+))?"                  # optional L<n>
        r"(?:\s*,\s*dt=([\d.eE+-]+)\s*s)?"     # optional dt
        r"(?:\s*,\s*([\d.]+)\s*days?)?",        # optional days (sing/plur)
        text,
    )
    if m:
        out["grid"] = m.group(1)
        out["resolution"] = m.group(2)
        if m.group(3) is not None:
            out["nlev"] = int(m.group(3))
        if m.group(4) is not None:
            try:
                out["dt"] = float(m.group(4))
            except ValueError:
                pass
        if m.group(5) is not None:
            try:
                d = float(m.group(5))
                out["days"] = int(d) if d == int(d) else d
            except ValueError:
                pass
    # Status line: "Status: COMPLETED"
    m = re.search(r"Status:\s*(\S+)", text)
    if m:
        s = m.group(1).upper().rstrip(",.")
        out["status"] = "PASS" if s in _PASS_STATUSES else "ERROR"
    # Wall time: "Wall time: 123.4s"
    m = re.search(r"Wall time:\s*([\d.eE+-]+)s", text)
    if m:
        try:
            out["wall_time"] = f"{float(m.group(1)):.1f}s"
        except ValueError:
            pass
    # JIT compilation: "JIT compilation: 12.3s"
    m = re.search(r"JIT compilation:\s*([\d.eE+-]+)s", text)
    if m:
        try:
            out["jit_time"] = f"{float(m.group(1)):.1f}s"
        except ValueError:
            pass
    return out


def _atomic_write_text(path: Path, text: str) -> None:
    """iter-42 codex LOW: write to ``<name>.tmp`` then ``os.replace``
    so a crashed converter run cannot leave a half-written file that
    still satisfies ``_has_collectable``.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _purge_stale_matrix_outputs(out_dir: Path) -> None:
    """Remove matrix-format outputs from a previous (now-stale) run.

    iter-42 codex HIGH: a failed AMIP rerun into an existing OUTDIR
    can be silently collected as old data unless we explicitly remove
    the prior converted files when ``timeseries.npz`` is absent.
    Called only on the missing-npz path.
    """
    for name in ("mean_timeseries.csv", "results.txt", "results_amip.txt"):
        p = out_dir / name
        if p.exists():
            p.unlink()


def main(out_dir: Path) -> int:
    """Convert ``out_dir/timeseries.npz`` + ``out_dir/results.txt`` to
    matrix-runner-compatible ``mean_timeseries.csv`` + ``results.txt``.

    Returns 0 on success.  Returns 2 on missing/empty ``timeseries.npz``
    (caller — the wrapper script — should treat this as a failed
    AMIP run).  Idempotent across repeated calls.

    iter-42 codex HIGH:
      * Missing ``timeseries.npz`` is now a NON-zero exit code AND
        purges any stale matrix-format files in ``out_dir`` so a
        cross-grid collection pass cannot pick up old data.
      * If the AMIP ``results.txt`` is missing or has no parsable
        Status line, the matrix-format ``results.txt`` written by
        this converter explicitly emits ``status: ERROR`` so the
        downstream collector treats the run as failed.
    """
    out_dir = Path(out_dir)
    npz_path = out_dir / "timeseries.npz"
    if not npz_path.exists():
        # iter-42 codex HIGH: failed AMIP → purge stale converted
        # outputs and return non-zero so the wrapper aborts the
        # cross-grid collection step.
        print(
            f"  [_amip_to_matrix_format] ERROR: {npz_path} missing; "
            f"purging stale matrix outputs in {out_dir}",
            file=sys.stderr,
        )
        _purge_stale_matrix_outputs(out_dir)
        return 2

    data = np.load(npz_path, allow_pickle=False)
    # Build (column_name, array) pairs in deterministic order, skipping
    # NaN-sentinel entries.
    cols: list[tuple[str, np.ndarray]] = []
    # iter-70 codex review: distinguish "days array missing" from
    # "days array empty" in the error message — the iter-69 5-day
    # AMIP smoke initially produced empty diagnostics
    # (``--days 1`` shorter than ``--diag-days 5``) and the
    # iter-42 message read "'days' missing" which is misleading.
    if "days" not in data.files:
        print(
            f"  [_amip_to_matrix_format] ERROR: 'days' key not in "
            f"{npz_path}; cannot write CSV.",
            file=sys.stderr,
        )
        _purge_stale_matrix_outputs(out_dir)
        return 2
    days = data["days"]
    if not _is_real_array(days):
        days_arr = np.asarray(days)
        print(
            f"  [_amip_to_matrix_format] ERROR: 'days' array in "
            f"{npz_path} is empty (shape={days_arr.shape}); cannot "
            f"write CSV.  Did the AMIP run shorter than "
            f"``--diag-days``?",
            file=sys.stderr,
        )
        _purge_stale_matrix_outputs(out_dir)
        return 2
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
        arr_np = np.asarray(arr)
        if arr_np.shape[0] != n:
            # iter-42 codex MEDIUM: warn loudly so a length mismatch
            # is not a silent data drop.
            print(
                f"  [_amip_to_matrix_format] WARNING: variable "
                f"'{src_key}' has length {arr_np.shape[0]} but the "
                f"time axis is length {n}; dropping it from the CSV.",
                file=sys.stderr,
            )
            continue
        cols.append((dst_key, arr_np))

    # iter-43 codex LOW: write order matters for crash safety.
    # ``has_collectable_atmosphere_outputs`` requires BOTH
    # ``mean_timeseries.csv`` AND ``results.txt`` to be present, so
    # if we crash between writing the CSV and writing the new
    # results.txt while results.txt is still the AMIP free-form
    # original, the directory would be ``collectable`` but with the
    # wrong format.  Mitigation: rename results.txt → results_amip.txt
    # FIRST so the directory transiently has no results.txt, then
    # write the CSV, then write the new matrix-format results.txt.
    # At every crash point in this sequence,
    # ``has_collectable_atmosphere_outputs`` returns False (because
    # results.txt is missing) until the final atomic write.
    results_txt = out_dir / "results.txt"
    results_amip = out_dir / "results_amip.txt"
    already_converted = (
        results_txt.exists()
        and results_txt.read_text(errors="replace").startswith("test: amip")
    )
    if already_converted:
        # On a re-run, both results_amip.txt and the matrix-format
        # results.txt already exist.  Re-parse from the preserved
        # AMIP file (or fall back to the matrix-format file if the
        # AMIP original was manually removed).
        if results_amip.exists():
            parsed = _parse_amip_results_txt(results_amip)
        else:
            parsed = _parse_matrix_results_txt(results_txt)
    elif not results_txt.exists() and results_amip.exists():
        # iter-44 codex LOW: post-crash recovery path.  A previous
        # converter run crashed between the AMIP→matrix rename and
        # the final results.txt write, so results.txt is missing
        # but results_amip.txt has the AMIP free-form file.  Re-
        # parse from results_amip.txt rather than dropping metadata
        # and emitting a bogus ``status: ERROR``.
        parsed = _parse_amip_results_txt(results_amip)
    else:
        parsed = _parse_amip_results_txt(results_txt)
        if results_txt.exists():
            # Rename FIRST so the directory transiently has no
            # results.txt — _has_collectable returns False during
            # the crash window between this and the final write.
            results_txt.rename(results_amip)

    # iter-42 codex HIGH: if no Status was found, default to ERROR.
    if "status" not in parsed:
        parsed["status"] = "ERROR"

    # Build CSV and matrix-format results.txt content in memory.
    csv_lines = [",".join(c[0] for c in cols)]
    for i in range(n):
        csv_lines.append(",".join(f"{c[1][i]:.6e}" for c in cols))
    out_text_lines = ["test: amip"]
    for key in (
        "grid", "resolution", "nlev", "dt", "days",
        "status", "wall_time", "jit_time",
    ):
        if key in parsed:
            out_text_lines.append(f"{key}: {parsed[key]}")
    out_text_lines.append(
        "notes: real-AMIP run via run_amip.py; "
        "format converted by _amip_to_matrix_format.py for "
        "matrix-runner cross-grid collection (iter-42)"
    )

    # Write CSV first (this is safe at any crash point because the
    # rename above removed results.txt).  Then write the matrix-
    # format results.txt last (the final commit point that flips
    # the directory to collectable).
    _atomic_write_text(
        out_dir / "mean_timeseries.csv", "\n".join(csv_lines) + "\n",
    )
    _atomic_write_text(results_txt, "\n".join(out_text_lines) + "\n")
    return 0


def _parse_matrix_results_txt(path: Path) -> dict:
    """Re-parse a matrix-format ``results.txt`` back into the dict.

    Used as a recovery path when the AMIP original is missing.
    iter-42 codex LOW: makes the idempotency fallback honest.
    """
    out: dict[str, str | float] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            continue
        if k in ("nlev",):
            try:
                out[k] = int(v)
            except ValueError:
                pass
        elif k in ("dt",):
            try:
                out[k] = float(v)
            except ValueError:
                pass
        elif k == "days":
            try:
                d = float(v)
                out[k] = int(d) if d == int(d) else d
            except ValueError:
                pass
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "usage: python _amip_to_matrix_format.py OUT_DIR",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1])))
