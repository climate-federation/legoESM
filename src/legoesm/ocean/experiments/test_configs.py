"""Per-term test configuration factory.

Returns ocean config NamedTuples with all-but-one terms disabled, used by
the per-term validation tests landed under
``tests/ocean/unit/test_per_term_*.py``.

Test configurations are produced through this factory rather than
ad-hoc kwargs in test files so that:

  1. ``test_mode=True`` is set by construction.  This is a *convention*
     marker — production paths do not yet runtime-reject it — but it
     makes test configs visibly identifiable in the audit trail.
     Adding a defensive check in ``LatLonCGridOceanModel`` /
     ``MPASOceanModel`` constructors that refuses ``test_mode=True``
     for production entry points is a tracked follow-up.
  2. Adding a new per-term test requires adding one named constructor
     here, surfacing the design choice in one place.
  3. The factory is the single place that knows which ``disable_*``
     flags map to which test recipe — defined in
     ``docs/ocean/adcroft_followups.md`` §Item 6 / §Plan-of-work 1C.

See also ``docs/ocean/per_term_test_methodology.md`` for the
"convergence rate, not absolute L2" methodology.
"""

from __future__ import annotations

from typing import Literal

from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.mpas_config import MPASOceanConfig


TermLiteral = Literal[
    "coriolis_only",          # Recipe #1: only Coriolis active
    "tracer_advection_only",  # Recipe #2: only tracer advection active
    "gravity_wave",           # Recipe #5: PGF + free surface; Coriolis off
]

GridLiteral = Literal["latlon", "mpas"]


def make_test_config(
    term: TermLiteral,
    grid: GridLiteral = "latlon",
    **overrides,
):
    """Return a config NamedTuple with all but the named term(s) disabled.

    Parameters
    ----------
    term
        Per-term test recipe.
    grid
        ``"latlon"`` returns a ``LatLonCGridOceanConfig``; ``"mpas"``
        returns an ``MPASOceanConfig``.
    **overrides
        Forwarded as ``_replace`` to the produced NamedTuple, e.g.
        ``make_test_config("coriolis_only", n_barotropic_substeps=10)``.

    Returns
    -------
    LatLonCGridOceanConfig | MPASOceanConfig
        With ``test_mode=True`` and the appropriate ``disable_*`` flags
        set.  All non-disabled fields keep their NamedTuple defaults
        (Smagorinsky off, GM/Redi off, drag off, etc.).
    """
    if grid == "latlon":
        ConfigCls = LatLonCGridOceanConfig
    elif grid == "mpas":
        ConfigCls = MPASOceanConfig
    else:
        raise ValueError(f"Unknown grid {grid!r}; expected 'latlon' or 'mpas'.")

    flags = _disable_flags_for_term(term)
    # Baseline safety overrides — production defaults that have been
    # flipped over time and would silently change per-term test
    # behaviour.  Per-term tests run on tiny domains where these
    # production-realistic operators are inappropriate and obscure the
    # scheme under test.  Override here rather than at every call site.
    safety_baseline = dict(
        implicit_vertical_mixing=False,
    )
    base = ConfigCls(test_mode=True, **safety_baseline, **flags)
    if overrides:
        base = base._replace(**overrides)
    return base


def _disable_flags_for_term(term: TermLiteral) -> dict:
    """Map test recipe to the ``disable_*`` flag dictionary."""
    if term == "coriolis_only":
        return dict(
            disable_pgf=True,
            disable_momentum_advection=True,
            disable_tracer_advection=True,
            disable_drag=True,
        )
    if term == "tracer_advection_only":
        return dict(
            disable_coriolis=True,
            disable_pgf=True,
            disable_momentum_advection=True,
            disable_drag=True,
        )
    if term == "gravity_wave":
        return dict(
            disable_coriolis=True,
            disable_momentum_advection=True,
            disable_tracer_advection=True,
            disable_drag=True,
        )
    raise ValueError(f"Unknown term {term!r}.")
