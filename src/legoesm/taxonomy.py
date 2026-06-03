"""The legoESM capability taxonomy — one discoverable map of what can be built.

The three "Lego" axes (complexity x extent x grid/core) are each enforced by a
canonical source of truth elsewhere — the grid factory's type lists
(:mod:`legoesm.grids.factory`), the component complexity rungs
(:mod:`legoesm.components.complexity`), and the atmosphere dynamics options
(:mod:`legoesm.atmosphere.dynamics`).  This module does NOT redefine them; it
**imports** those canonical values (grid global/regional membership, complexity
rungs, dynamics options) and adds only the cross-axis organisation + the per-grid
*refinement* metadata that lives nowhere else, so a user (or a config validator,
or the docs) can answer "what can legoESM instantiate?" from one place.

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
    model_complexity_rungs,
)
from legoesm.grids.factory import GLOBAL_GRID_TYPES, REGIONAL_GRID_TYPES


class GridCapability(NamedTuple):
    """What a grid family supports across the extent axis."""

    name: str
    global_: bool          # instantiable via create_grid (resolution-based / file)
    regional: bool         # instantiable via create_regional_grid (limited-area)
    variable_resolution: bool  # local refinement (Schmidt stretch / density mesh)
    refinement: str        # how refinement is achieved ("" if none)


#: The ONLY locally-owned grid metadata: how a refinement-capable family refines.
#: The grid *set* and its global/regional flags are derived from the factory below
#: (no duplication); this dict adds the refinement mechanism, which lives nowhere
#: else.  Keyed only by the refinement-capable families.
_REFINEMENT_METHOD: dict[str, str] = {
    "cubed_sphere": "Schmidt stretch (create_cubed_sphere stretch_fac/target_*)",
    "mpas": "density-weighted Lloyd (create_voronoi_mesh density_fn)",
}


def grid_capabilities() -> tuple[GridCapability, ...]:
    """Derive the per-grid capability table from the canonical factory lists."""
    names = sorted(set(GLOBAL_GRID_TYPES) | set(REGIONAL_GRID_TYPES))
    return tuple(
        GridCapability(
            name=n,
            global_=n in GLOBAL_GRID_TYPES,
            regional=n in REGIONAL_GRID_TYPES,
            variable_resolution=n in _REFINEMENT_METHOD,
            refinement=_REFINEMENT_METHOD.get(n, ""),
        )
        for n in names
    )


#: Per-grid capability across the extent axis (derived from the factory + the local
#: refinement metadata).  Computed once at import.
GRID_CAPABILITIES: tuple[GridCapability, ...] = grid_capabilities()

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


def model_dial() -> dict[str, dict[str, str]]:
    """The model-wide complexity dial resolved to per-component rungs.

    e.g. ``{"full": {"atmosphere": "hydrostatic", "ocean": "full_3d",
    "land": "multilayer", "ice": "dynamic"}, ...}`` — so a consumer can reconstruct
    what each dial setting selects.
    """
    out: dict[str, dict[str, str]] = {}
    for level in ModelComplexity:
        rungs = model_complexity_rungs(level)
        out[level.value] = {
            "atmosphere": str(rungs.atmosphere),
            "ocean": str(rungs.ocean),
            "land": str(rungs.land),
            "ice": str(rungs.ice),
        }
    return out


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
        "model_complexity": model_dial(),
        "components": list(COMPONENT_KINDS),
        "atmosphere_dynamics": {
            "model_types": list(DYNAMICS_OPTIONS),
            "discretizations": list(DISCRETIZATION_OPTIONS),
        },
    }
