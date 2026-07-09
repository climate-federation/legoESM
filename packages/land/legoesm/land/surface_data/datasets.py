"""Registry of supported land-cover datasets (historical reconstructions).

The reconstruction *choice* surfaces here.  Each transient dataset has a producer
that builds a harmonized ``legoesm_surfdata`` file (e.g.
``build_luh2_transient_surfdata`` for LUH2); a driver records which one built the
file it loaded via ``land_cover_dataset``.  The E_LUC land-use-change carbon
bookkeeping keys off :func:`has_gross_transitions`: only LUH2 / LUH3 carry native
**gross** transitions (shifting cultivation + wood harvest); the others provide
states only, so net transitions must be derived from year-to-year differences
(a documented fidelity limitation for those datasets).

Pure host-side metadata; no heavy deps, safe to import anywhere.
"""

from __future__ import annotations

# ``clm5`` is the static single-year CLM5 base (no land-use change); the rest are
# transient historical land-use/land-cover reconstructions.
KNOWN_LAND_COVER_DATASETS = ("clm5", "luh2", "luh3", "hyde", "pongratz", "kk10")

# Datasets providing NATIVE gross land-use transitions (vs states only).
_GROSS_TRANSITION_DATASETS = frozenset({"luh2", "luh3"})


def validate_land_cover_dataset(name: str) -> str:
    """Return ``name`` if it is a known dataset, else raise (no silent default)."""
    if name not in KNOWN_LAND_COVER_DATASETS:
        raise ValueError(
            f"unknown land_cover_dataset {name!r}; expected one of "
            f"{KNOWN_LAND_COVER_DATASETS}")
    return name


def has_gross_transitions(name: str) -> bool:
    """Whether ``name`` carries native gross transitions (faithful E_LUC input).

    Raises on an unknown dataset (validated first).
    """
    return validate_land_cover_dataset(name) in _GROSS_TRANSITION_DATASETS
