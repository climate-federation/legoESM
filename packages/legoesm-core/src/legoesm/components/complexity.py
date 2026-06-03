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

**Atmosphere** (dynamical complexity = the dycore ``model_type``; the numerical
discretization cdgrid/spectral/mpas/... is an orthogonal *method* choice, not a
fidelity rung):

=================  ============================================================
level              implementation
=================  ============================================================
``shallow_water``  barotropic shallow water        (single-layer)
``hydrostatic``    hydrostatic primitive equations (3D)
``nonhydrostatic`` fully compressible nonhydrostatic Euler (3D)
=================  ============================================================

**Sea ice**:

================  ===========================================================
level             implementation
================  ===========================================================
``thermodynamic`` single-category slab thermodynamics, diagnostic free drift
``dynamic``       EVP rheology + 5-category CICE ice-thickness distribution
================  ===========================================================

Sea-ice sub-physics (snow / brine / ridging / ponds) are independent opt-in
gates on ``SeaIceConfig``, NOT complexity rungs, so they are not selected by
:class:`IceComplexity`.

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
from typing import NamedTuple


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


class AtmosphereComplexity(StrEnum):
    """Atmosphere dynamical-complexity rungs: barotropic shallow water ->
    hydrostatic primitive equations -> fully compressible nonhydrostatic.

    This is the *physical* complexity (the dycore ``model_type``), distinct from
    the numerical *discretization* (cdgrid / spectral / mpas / latlon_cgrid /
    finite_volume / sfno), which is an orthogonal choice of method, not of
    fidelity.  The rung values deliberately equal the ``model_type`` strings.
    """

    SHALLOW_WATER = "shallow_water"
    HYDROSTATIC = "hydrostatic"
    NONHYDROSTATIC = "nonhydrostatic"


def atmosphere_model_type(complexity: str | AtmosphereComplexity) -> str:
    """Return the dycore ``model_type`` string for an atmosphere complexity.

    ``atmosphere_model_type("hydrostatic") -> "hydrostatic"``.  The rung values
    equal the ``model_type`` strings (``atmosphere.dynamics.DYNAMICS_OPTIONS``),
    so this is the typed, validated entry to that axis.  Raises ``ValueError``
    for an unknown rung.  (Pure taxonomy: it returns a string and does not import
    the atmosphere package; a unit test grounds the rungs against the live
    ``DYNAMICS_OPTIONS`` so the two cannot drift.)

    Note the deliberate asymmetry with ocean/land/ice: the atmosphere dycore
    already selects its physical complexity through the ``model_type`` *config
    field* (read by ``create_atmosphere_dycore`` from ``ExperimentConfig.dycore``),
    so there is no separate ``atmosphere_complexity`` factory argument — this
    resolver is the typed way to set that field, and it becomes load-bearing in
    the model-wide complexity dial (``ModelComplexity``).
    """
    return AtmosphereComplexity(complexity).value


class IceComplexity(StrEnum):
    """Sea-ice complexity rungs: thermodynamic slab -> dynamic multi-category.

    ``thermodynamic`` is the default single-category slab with diagnostic free
    drift (no momentum rheology); ``dynamic`` adds EVP rheology + a 5-category
    CICE-standard ice-thickness distribution.  Orthogonal sub-physics (snow /
    brine / ridging / ponds) are independent opt-in gates on ``SeaIceConfig``,
    NOT complexity rungs, so they are not selected here.  The rung -> config
    resolution lives in the driver's ice factory (which owns ``SeaIceConfig``),
    not in this pure taxonomy.
    """

    THERMODYNAMIC = "thermodynamic"
    DYNAMIC = "dynamic"


class ComponentComplexities(NamedTuple):
    """The per-component complexity rungs of one model-wide complexity level."""

    atmosphere: AtmosphereComplexity
    ocean: OceanComplexity
    land: LandComplexity
    ice: IceComplexity


class ModelComplexity(StrEnum):
    """A single model-wide complexity dial climbing the Held hierarchy of models.

    One level resolves all four components to a coherent rung at once
    (:func:`model_complexity_rungs`), so a user dials ``complexity="idealized"``
    instead of hand-assembling four enums:

    ===============  =============  ==========  ============  ==============
    level            atmosphere     ocean       land          ice
    ===============  =============  ==========  ============  ==============
    ``idealized``    shallow_water  fixed_sst   slab          thermodynamic
    ``intermediate`` hydrostatic    slab        multilayer    thermodynamic
    ``full``         hydrostatic    full_3d     multilayer    dynamic
    ===============  =============  ==========  ============  ==============

    The ``full`` atmosphere is hydrostatic primitive equations — the standard
    *global-climate* dycore.  ``nonhydrostatic`` is a specialised km-scale
    CRM/LES choice (not "more Earth" at climate resolution), reached directly via
    :class:`AtmosphereComplexity`, so it is intentionally NOT on this Earth dial.
    """

    IDEALIZED = "idealized"
    INTERMEDIATE = "intermediate"
    FULL = "full"


_MODEL_COMPLEXITY: dict[ModelComplexity, ComponentComplexities] = {
    ModelComplexity.IDEALIZED: ComponentComplexities(
        atmosphere=AtmosphereComplexity.SHALLOW_WATER,
        ocean=OceanComplexity.FIXED_SST,
        land=LandComplexity.SLAB,
        ice=IceComplexity.THERMODYNAMIC,
    ),
    ModelComplexity.INTERMEDIATE: ComponentComplexities(
        atmosphere=AtmosphereComplexity.HYDROSTATIC,
        ocean=OceanComplexity.SLAB,
        land=LandComplexity.MULTILAYER,
        ice=IceComplexity.THERMODYNAMIC,
    ),
    ModelComplexity.FULL: ComponentComplexities(
        atmosphere=AtmosphereComplexity.HYDROSTATIC,
        ocean=OceanComplexity.FULL_3D,
        land=LandComplexity.MULTILAYER,
        ice=IceComplexity.DYNAMIC,
    ),
}


def model_complexity_rungs(level: str | ModelComplexity) -> ComponentComplexities:
    """Resolve a model-wide complexity *level* to each component's rung.

    ``model_complexity_rungs("full").ocean is OceanComplexity.FULL_3D``.  Pure
    taxonomy: returns the component-complexity enums (all owned by this module);
    a driver feeds each to its factory.  Raises ``ValueError`` on an unknown
    level; the table covers every :class:`ModelComplexity` member (a unit test
    asserts totality, so a newly added level without a mapping fails loudly).
    """
    return _MODEL_COMPLEXITY[ModelComplexity(level)]
