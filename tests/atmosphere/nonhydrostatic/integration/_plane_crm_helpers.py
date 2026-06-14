"""Shared helpers for the plane CRM smoke + production-scale tests.

iter-79: extracted ``_parse_rad_call_count`` (iter-40 / iter-54
provenance) from ``test_plane_crm_end_to_end_smoke.py`` into a
dedicated sibling module so the iter-53/54 unit-test file can
import it without triggering pytest collection of the source test
file as an import side effect. Mirrors the iter-78 hydrostatic
helpers extraction.

iter-84: added ``_read_log`` (pure log.txt parser, file-local
until now). All 5 use sites in ``test_plane_crm_end_to_end_smoke.py``
import via this module so the parser is unit-testable. (Codex
iter-85 LOW: previous doc said "4 use sites" — was an undercount
discovered post-commit.)

The driver subprocess helpers (``_run_driver``,
``_run_driver_with_radiation``, ``_run_driver_production_scale``)
stay in ``test_plane_crm_end_to_end_smoke.py`` — they hardcode
config-specific CLI args and have no cross-file consumers.

Underscore prefix marks the module as a non-test helper so
pytest's ``test_*.py`` glob skips it.
"""
from __future__ import annotations

from pathlib import Path


def _read_log(output_dir):
    """Read ``log.txt`` rows as a list of dicts.

    Driver format (``scripts/run/run_rce_mpi_long.py:722-734``):
    * Header lines start with ``# RCE MPI LONG`` + ``# physics:``.
    * Schema line starts with ``# step,`` listing the CSV columns.
    * Subsequent lines are CSV data rows.
    * Returns an empty list when ``log.txt`` is missing (driver
      crashed before writing) — caller asserts ``rows`` non-empty.
    """
    log_path = Path(output_dir) / "log.txt"
    if not log_path.exists():
        return []
    rows = []
    with open(log_path) as fh:
        header = None
        for line in fh:
            line = line.strip()
            if line.startswith("# step,"):
                header = line.lstrip("# ").split(",")
            elif line.startswith("#") or not line:
                continue
            elif header is not None:
                parts = line.split(",")
                if len(parts) == len(header):
                    rows.append(dict(zip(header, parts)))
    return rows


def _parse_rad_call_count(stdout: str) -> int | None:
    """Extract ``rad_calls=N`` from the driver's final ``Done.`` line.

    Driver format (``scripts/run/run_rce_mpi_long.py:914``):
        ``f"rad_calls={rad_call_count}."`` — always integer + trailing
        period (the Done.-line punctuation).

    Regex requirements (anchored to that exact format):
    * line starts with ``Done.`` (multiline + ``\\b`` word-bound
      avoids ``total_rad_calls=5`` substring drift — iter-40 Codex
      LOW#4 fix).
    * ``\\d+\\.[^\\S\\n]*$`` (multiline) requires integer-then-
      period as the LAST non-whitespace token on the Done. line.
      The character class ``[^\\S\\n]*`` is "any whitespace except
      newline" — so trailing spaces / tabs / CR (before \\n) are
      tolerated, but a future schema drift like
      ``rad_calls=5. (cached)`` or ``rad_calls=5.0.`` is rejected
      (iter-54 hardening; closes the iter-53-pinned gap).

    Returns the count when exactly one ``Done.`` line matches, else
    ``None`` (older driver / parser-side regression / unexpected
    multiple matches).
    """
    import re
    matches = re.findall(
        r"(?m)^Done\..*\brad_calls=(\d+)\.[^\S\n]*$", stdout,
    )
    if len(matches) != 1:
        return None
    return int(matches[0])
