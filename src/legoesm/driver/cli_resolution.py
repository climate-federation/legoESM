"""Shared ``--resolution N`` per-grid dispatch + validation
for legoESM CLI scripts.

iter-115 (codex iter-114-followup HIGH-1, MEDIUM-2): centralized
the iter-95/102/107/108/110/111/112 dispatch + validation logic
so that ``run_atmosphere_test_matrix.py``,
``run_ocean_test_matrix.py``,
``scripts/ocean_test_matrix/cli.py``, and
``run_omip.py`` (which iter-114 codex review caught was
missing the dispatch entirely) all share a single
implementation.

The dispatch table is defined here once; callers either:
* call ``expand_cli_resolution(cli_res, grid_type)`` for a
  single grid expansion, or
* call ``validate_cli_resolution(cli_res)`` to reject
  obviously-invalid forms (decimal, scientific, NaN,
  infinity, complex, alphabetic) before any grid-specific
  parsing.

Per-grid dispatch table:

* cubed_sphere → ``f"C{N}"``  (iter-95)
* latlon → ``f"{N}x{2*N}"``  (iter-95)
* icosahedral / mpas → ``f"ico{level}"`` where
  ``level = round(log4(2N²/10))`` clipped to [2, 8] (iter-95)
* spectral → ``f"T{N}"``  (iter-95)
* mpas_regional → ``f"{N}km"``  (iter-110)
* latlon_regional → ``f"{N}x{2*N}"``  (iter-110)
* cs_regional → ``f"C{N}"``  (iter-110)

Pre-formatted strings (``"C24"``, ``"ico5"``, ``"36x72"``,
``"T21"``, ``"50km"``) pass through unchanged.

Validation rules:

* Reject ``N <= 0`` (iter-107)
* Reject decimal / scientific / inf / nan numerics
  (iter-111/112) via float() try/except + a positive-format
  whitelist (iter-115 codex MEDIUM-2: 2j and hello slipped
  through pre-iter-115 because float() returned ValueError
  on them but the per-grid parser then crashed later).
"""
from __future__ import annotations

import math
import re
import sys
from typing import Optional

# Recognized per-grid format string patterns.  Any non-numeric
# string passed to ``--resolution`` must match one of these,
# otherwise iter-115 codex MEDIUM-2 says we should reject at
# the boundary.
#
# iter-118 (codex iter-117 followup MEDIUM-1): require POSITIVE
# components — pre-iter-118 patterns like ``\d+`` matched 0
# (e.g., ``C0``, ``0x32``, ``16x0``, ``0km``), which then
# crashed later in the per-grid parsers.  iter-118 uses
# ``[1-9]\d*`` to require a non-zero leading digit.
_PER_GRID_FORMAT_PATTERNS = {
    "C": re.compile(r"^C[1-9]\d*$"),                          # cubed_sphere / cs_regional
    "x": re.compile(r"^[1-9]\d*x[1-9]\d*$"),                  # latlon / latlon_regional
    "ico": re.compile(r"^ico[1-9]\d*$"),                      # icosahedral / mpas
    "T": re.compile(r"^T[1-9]\d*$"),                          # spectral
    "km": re.compile(r"^[1-9]\d*km$"),                        # mpas_regional
}


def _is_known_per_grid_format(s: str) -> bool:
    """True if ``s`` matches any of the known per-grid format
    string patterns.  Used by ``validate_cli_resolution`` to
    reject invalid non-numeric strings."""
    return any(p.match(s) for p in _PER_GRID_FORMAT_PATTERNS.values())


def validate_cli_resolution(
    cli_res: str,
    *,
    error_prefix: str = "--resolution",
    additional_examples: str = "'C24', 'ico3', '36x72', 'T21', '50km'",
) -> Optional[int]:
    """Validate ``cli_res`` and return ``N`` if it's a bare
    positive integer, else ``None`` for valid per-grid format
    strings.

    Calls ``sys.exit(2)`` with a clear error message for:
    * ``N <= 0`` (zero or negative integer)
    * Non-integer numerics (decimal, scientific, inf, nan)
    * Strings that don't match any known per-grid format
      pattern (e.g., ``2j``, ``hello``, ``garbage``).

    Parameters
    ----------
    cli_res
        The raw ``args.resolution`` value.
    error_prefix
        The flag name to use in error messages (default
        ``--resolution``).
    additional_examples
        Comma-separated examples of valid per-grid formats
        for the error message — pass the per-CLI list (atmos
        omits ``50km``, ocean adds it).

    Returns
    -------
    int or None
        ``N`` (the bare integer) if input is a positive
        integer; ``None`` if input is a valid per-grid format
        string.  Never returns for invalid input — exits 2.
    """
    # Try int parse first (the common case).
    try:
        N = int(cli_res)
    except ValueError:
        N = None

    if N is not None:
        if N <= 0:
            print(
                f"error: {error_prefix} must be a positive "
                f"integer (or a per-grid format string like "
                f"{additional_examples}).  Got N={N}.",
                file=sys.stderr,
            )
            sys.exit(2)
        return N

    # Not a bare integer.  Reject non-integer numerics.
    try:
        _ = float(cli_res)
        # Parses as float but not as int → non-integer numeric.
        print(
            f"error: {error_prefix} must be a positive INTEGER "
            f"(or a per-grid format string like "
            f"{additional_examples}).  Got non-integer numeric "
            f"string {cli_res!r}.",
            file=sys.stderr,
        )
        sys.exit(2)
    except ValueError:
        # Not a numeric string at all.  iter-115 codex MEDIUM-2:
        # reject if it doesn't match any known per-grid format
        # pattern — pre-iter-115 strings like ``2j`` or
        # ``hello`` slipped past the float() check and crashed
        # later in the per-grid parser.
        pass

    if not _is_known_per_grid_format(cli_res):
        print(
            f"error: {error_prefix} must match a known per-grid "
            f"format ({additional_examples}) or be a positive "
            f"integer.  Got unrecognized string {cli_res!r}.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Valid per-grid format string — pass through unchanged.
    return None


def expand_cli_resolution(N: int, grid_type: str) -> str:
    """Expand a bare positive integer ``N`` to the per-grid
    format string for ``grid_type``.

    See module docstring for the dispatch table.

    For ``icosahedral`` / ``mpas``, prints a warning when
    ``N`` is large enough to map to a level > 8 (the maximum
    supported per ``voronoi.py:1063``); the level is then
    clipped to 8.

    Parameters
    ----------
    N
        Positive integer (validated by ``validate_cli_resolution``).
    grid_type
        Grid type name (``cubed_sphere``, ``latlon``,
        ``icosahedral``, ``mpas``, ``spectral``,
        ``mpas_regional``, ``latlon_regional``,
        ``cs_regional``).

    Returns
    -------
    str
        Per-grid format string (``f"C{N}"``, ``f"{N}x{2*N}"``, etc.).
    """
    if grid_type == "cubed_sphere":
        return f"C{N}"
    elif grid_type == "latlon":
        return f"{N}x{2 * N}"
    elif grid_type in ("icosahedral", "mpas"):
        raw_level = round(math.log(2 * N * N / 10) / math.log(4))
        level = max(2, min(8, raw_level))
        if raw_level > 8:
            print(
                f"warning: --resolution {N} maps to icosahedral "
                f"level {raw_level} which exceeds the maximum "
                f"supported level (8 = 655,362 cells); clipping "
                f"to ico8.  Use load_mpas_mesh() with a "
                f"pre-built mesh file for higher resolutions.",
            )
        return f"ico{level}"
    elif grid_type == "spectral":
        return f"T{N}"
    elif grid_type == "mpas_regional":
        return f"{N}km"
    elif grid_type == "latlon_regional":
        return f"{N}x{2 * N}"
    elif grid_type == "cs_regional":
        return f"C{N}"
    else:
        # Unknown grid type — pass through the integer as a
        # string for graceful degradation; the per-grid parser
        # will raise a clearer error.
        return str(N)
