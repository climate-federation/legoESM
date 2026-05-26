"""Shared helpers for the plane CRM smoke + production-scale tests.

iter-79: extracted ``_parse_rad_call_count`` (iter-40 / iter-54
provenance) from ``test_plane_crm_end_to_end_smoke.py`` into a
dedicated sibling module so the iter-53/54 unit-test file can
import it without triggering pytest collection of the source test
file as an import side effect. Mirrors the iter-78 hydrostatic
helpers extraction.

The driver subprocess helpers (``_run_driver``, ``_read_log``,
``_run_driver_with_radiation``, ``_run_driver_production_scale``)
stay in ``test_plane_crm_end_to_end_smoke.py`` — they are file-
local with no cross-file consumers.

Underscore prefix marks the module as a non-test helper so
pytest's ``test_*.py`` glob skips it.
"""
from __future__ import annotations


def _parse_rad_call_count(stdout: str) -> int | None:
    """Extract ``rad_calls=N`` from the driver's final ``Done.`` line.

    Driver format (``scripts/run_rce_mpi_long.py:914``):
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
