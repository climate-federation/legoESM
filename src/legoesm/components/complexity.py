"""Component complexity ladder — the §6 matrix 'physical complexity' rows.

Each Earth-system component runs at a *configurable level of complexity*, from the
cheapest prescribed/slab brick up to the full prognostic model — the same source
spanning the hierarchy of models (Held 2005), selected by a config field rather
than a separate code path.  This module names those levels and maps them onto the
existing factories (it does NOT reimplement them), so the driver/coupler can
resolve ``ocean.complexity = "slab_multilayer"`` to the right brick.

**Ocean** (increasing complexity):

==================  ===========================================================
level               implementation
==================  ===========================================================
``fixed_sst``       prescribed SST            (``make_ocean`` mode ``"fixed"``)
``slab``            single-layer slab mixed   (``make_ocean`` mode ``"slab"``)
``slab_multilayer`` two-layer slab            (``make_ocean`` mode ``"two_layer"``)
``full_3d``         prognostic 3D dynamics    (MPAS / lat-lon C-grid ocean core)
==================  ===========================================================

**Land**:

==============  ===============================================================
level           implementation
==============  ===============================================================
``slab``        slab thermal + bucket hydrology  (``land.slab_land``)
``multilayer``  multi-layer soil column          (``land.multilayer_land``)
==============  ===============================================================

``full_3d`` ocean is a prognostic 3D model (MPAS or lat-lon C-grid core), built by
the driver's ocean component factory
(``driver.component_factory.create_ocean_component``) — NOT a ``make_ocean`` slab
mode.  :func:`ocean_simple_mode` makes that explicit by refusing it, so a driver
routes ``OceanComplexity.FULL_3D`` to that factory and the lower rungs to
``make_ocean``.  (This module stays a pure taxonomy — it names the rungs and does
not import or wrap the driver/ocean factories, which sit above ``components``;
their exact call signatures are owned there, not pinned here.)
"""

from __future__ import annotations

from enum import StrEnum


class OceanComplexity(StrEnum):
    """Ocean complexity rungs, fixed-SST -> slab(1L) -> slab(ML) -> full 3D."""

    FIXED_SST = "fixed_sst"
    SLAB = "slab"
    SLAB_MULTILAYER = "slab_multilayer"
    FULL_3D = "full_3d"


class LandComplexity(StrEnum):
    """Land complexity rungs: slab bucket -> multi-layer soil column."""

    SLAB = "slab"
    MULTILAYER = "multilayer"


# complexity -> the SimpleOceanConfig.mode accepted by ocean.simple_ocean.make_ocean.
_OCEAN_SIMPLE_MODE: dict[OceanComplexity, str] = {
    OceanComplexity.FIXED_SST: "fixed",
    OceanComplexity.SLAB: "slab",
    OceanComplexity.SLAB_MULTILAYER: "two_layer",
}


def ocean_simple_mode(complexity: str | OceanComplexity) -> str:
    """Return the ``make_ocean`` mode string for a *non-3D* ocean complexity.

    ``ocean_simple_mode("slab_multilayer") -> "two_layer"``.  Raises ``ValueError``
    for ``full_3d`` (a prognostic 3D ocean model built by the driver's ocean
    component factory, not a slab mode) and for any unknown level.
    """
    c = OceanComplexity(complexity)
    if c is OceanComplexity.FULL_3D:
        raise ValueError(
            "full_3d ocean is a prognostic 3D model (MPAS / lat-lon C-grid), not a "
            "make_ocean slab mode; build it with the driver's ocean component "
            "factory (driver.component_factory.create_ocean_component)."
        )
    return _OCEAN_SIMPLE_MODE[c]
