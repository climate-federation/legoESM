"""The legoESM capability taxonomy — one discoverable map of what can be built.

The three "Lego" axes (complexity x extent x grid/core) are each enforced by a
canonical source of truth elsewhere — the grid factory's type lists
(:mod:`legoesm.grids.factory`), the component complexity rungs
(:mod:`legoesm.components.complexity`), and the atmosphere dynamics options
(:mod:`legoesm.atmosphere.dynamics`).  This module does NOT redefine them; it
**imports** those canonical values and adds the cross-axis organisation +
per-grid capability metadata (global / regional / variable-resolution) that lives
nowhere else, so a user (or a config validator, or the docs) can answer "what can
legoESM instantiate?" from one place.

A consistency test (``tests/unit/test_taxonomy.py``) pins this map to the canonical
sources so the taxonomy can never silently drift from the factories that build
the grids and components.
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm.atmosphere.dynamics import DISCRETIZATION_OPTIONS, DYNAMICS_OPTIONS
from legoesm.components.complexity import (
    AtmosphereComplexity,
    IceComplexity,
    LandComplexity,
    ModelComplexity,
    OceanComplexity,
)


class GridCapability(NamedTuple):
    """What a grid family supports across the extent axis."""

    name: str
    global_: bool          # instantiable via create_grid (resolution-based / file)
    regional: bool         # instantiable via create_regional_grid (limited-area)
    variable_resolution: bool  # local refinement (Schmidt stretch / density mesh)
    refinement: str        # how refinement is achieved ("" if none)


#: Per-grid capability across the extent axis.  ``global_``/``regional`` are pinned
#: to the factory lists by the consistency test; ``variable_resolution`` is the
#: extra metadata this taxonomy adds.
GRID_CAPABILITIES: tuple[GridCapability, ...] = (
    GridCapability("cubed_sphere", True, True, True,
                   "Schmidt stretch (create_cubed_sphere stretch_fac/target_*)"),
    GridCapability("gaussian", True, False, False, ""),
    GridCapability("latlon", True, True, False, ""),
    GridCapability("mpas", True, True, True,
                   "density-weighted Lloyd (create_voronoi_mesh density_fn)"),
    GridCapability("mercator", False, True, False, ""),
    GridCapability("tripole", True, False, False, ""),
)

#: The three extent modes (the user's "global / regional / idealized" axis).
EXTENT_MODES: tuple[str, ...] = ("global", "regional", "idealized")

#: The component complexity ladders (the "different complexity" axis), keyed by
#: component.  Each value is the canonical StrEnum of rungs.
COMPLEXITY_LADDERS: dict[str, type] = {
    "atmosphere": AtmosphereComplexity,
    "ocean": OceanComplexity,
    "land": LandComplexity,
    "ice": IceComplexity,
}

#: The model-wide complexity dial (idealized / intermediate / full).
MODEL_COMPLEXITY = ModelComplexity

#: The Earth-system components that can be instanced independently OR coupled.
COMPONENT_KINDS: tuple[str, ...] = ("atmosphere", "ocean", "land", "ice")


def grid_capability(name: str) -> GridCapability:
    """Return the :class:`GridCapability` for *name*, or raise ``ValueError``."""
    for cap in GRID_CAPABILITIES:
        if cap.name == name:
            return cap
    raise ValueError(
        f"unknown grid family {name!r}; known: "
        f"{', '.join(c.name for c in GRID_CAPABILITIES)}"
    )


def capability_report() -> dict:
    """A structured snapshot of the full taxonomy (for docs / introspection)."""
    return {
        "axes": {
            "complexity": {k: [r.value for r in v]
                           for k, v in COMPLEXITY_LADDERS.items()},
            "extent": list(EXTENT_MODES),
            "grids": {c.name: {"global": c.global_, "regional": c.regional,
                               "variable_resolution": c.variable_resolution,
                               "refinement": c.refinement}
                      for c in GRID_CAPABILITIES},
        },
        "model_complexity": [r.value for r in ModelComplexity],
        "components": list(COMPONENT_KINDS),
        "atmosphere_dynamics": {
            "model_types": list(DYNAMICS_OPTIONS),
            "discretizations": list(DISCRETIZATION_OPTIONS),
        },
    }
