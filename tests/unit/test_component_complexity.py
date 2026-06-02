"""The ocean/land complexity ladder — grounded against the real factories.

Ocean: fixed_sst -> slab(1L) -> slab_multilayer(2L) -> full_3d.
Land:  slab -> multilayer (column).
"""

from __future__ import annotations

import pytest

from legoesm.components import (
    AtmosphereComplexity,
    IceComplexity,
    LandComplexity,
    ModelComplexity,
    OceanComplexity,
    atmosphere_model_type,
    model_complexity_rungs,
    ocean_simple_mode,
)


def test_ocean_complexity_rungs() -> None:
    assert [c.value for c in OceanComplexity] == [
        "fixed_sst",
        "slab",
        "slab_multilayer",
        "full_3d",
    ]


def test_land_complexity_rungs() -> None:
    assert [c.value for c in LandComplexity] == ["slab", "multilayer"]


@pytest.mark.parametrize(
    "complexity,mode",
    [
        (OceanComplexity.FIXED_SST, "fixed"),
        (OceanComplexity.SLAB, "slab"),
        (OceanComplexity.SLAB_MULTILAYER, "two_layer"),
    ],
)
def test_ocean_simple_mode_is_accepted_by_make_ocean(complexity, mode) -> None:
    """Each non-3D rung maps to a mode make_ocean actually accepts (no drift)."""
    from legoesm.ocean.simple_ocean import SimpleOceanConfig, make_ocean

    assert ocean_simple_mode(complexity) == mode
    # make_ocean returns a step closure for a valid mode and raises on unknown;
    # this grounds the taxonomy against the live factory.
    step = make_ocean(SimpleOceanConfig(mode=mode))
    assert callable(step)


def test_full_3d_is_not_a_slab_mode() -> None:
    with pytest.raises(ValueError, match="full_3d"):
        ocean_simple_mode("full_3d")


def test_full_3d_ocean_factory_has_the_advertised_contract() -> None:
    """The factory full_3d points at is real and takes a grid + an ocean_config."""
    import inspect

    from legoesm.driver.component_factory import create_ocean_component
    from legoesm.ocean.state import OceanConfig

    assert callable(create_ocean_component)
    assert OceanConfig is not None
    params = inspect.signature(create_ocean_component).parameters
    # full_3d is "built by the driver's ocean component factory" — confirm that
    # factory really takes a grid and an ocean_config, so the taxonomy's pointer
    # is not stale.
    assert "grid" in params
    assert "ocean_config" in params


def test_unknown_ocean_complexity_raises() -> None:
    with pytest.raises(ValueError):
        ocean_simple_mode("teleport")


def test_land_complexity_models_exist() -> None:
    """Both land rungs map to a real land model."""
    import legoesm.land as land

    assert hasattr(land, "step_land")  # slab
    assert hasattr(land, "step_multilayer_land")  # multilayer / column


# --- the taxonomy is load-bearing: the ocean factory consumes a rung ---


@pytest.mark.parametrize(
    "complexity,expected_mode",
    [
        (OceanComplexity.FIXED_SST, "fixed"),
        (OceanComplexity.SLAB, "slab"),
        (OceanComplexity.SLAB_MULTILAYER, "two_layer"),
    ],
)
def test_factory_consumes_complexity_rung(complexity, expected_mode) -> None:
    """create_ocean_component accepts an OceanComplexity and builds the mode."""
    from unittest.mock import patch

    from legoesm.driver.component_factory import create_ocean_component

    # make_ocean is imported inside the factory at call time -> patch its source.
    with patch("legoesm.ocean.simple_ocean.make_ocean") as mk:
        mk.return_value = lambda *a, **k: None
        step = create_ocean_component(config=None, grid=None, ocean_config=complexity)
    assert callable(step)
    # The rung was resolved to the right SimpleOceanConfig.mode (no drift).
    built_cfg = mk.call_args.args[0]
    assert built_cfg.mode == expected_mode


def test_factory_rejects_full_3d_rung_pointing_at_oceanconfig() -> None:
    """full_3d is not a slab rung — the factory directs to an explicit OceanConfig."""
    from legoesm.driver.component_factory import create_ocean_component

    with pytest.raises(ValueError, match="OceanConfig"):
        create_ocean_component(
            config=None, grid=None, ocean_config=OceanComplexity.FULL_3D
        )


@pytest.mark.parametrize(
    "complexity,expected_step",
    [
        (LandComplexity.SLAB, "step_land"),
        (LandComplexity.MULTILAYER, "step_multilayer_land"),
    ],
)
def test_land_factory_consumes_complexity_rung(complexity, expected_step) -> None:
    """create_land_component accepts a LandComplexity and returns its model."""
    import legoesm.land as land
    from legoesm.driver.component_factory import create_land_component

    step = create_land_component(config=None, grid=None, land_config=complexity)
    assert step is getattr(land, expected_step)


# --- atmosphere complexity ladder (model_type), grounded vs the live options ---


def test_atmosphere_complexity_rungs() -> None:
    assert [c.value for c in AtmosphereComplexity] == [
        "shallow_water",
        "hydrostatic",
        "nonhydrostatic",
    ]


@pytest.mark.parametrize("rung", list(AtmosphereComplexity))
def test_atmosphere_model_type_grounds_against_dynamics_options(rung) -> None:
    """Each rung resolves to a real dycore model_type — no drift from the source."""
    from legoesm.atmosphere.dynamics import DYNAMICS_OPTIONS

    model_type = atmosphere_model_type(rung)
    assert model_type == rung.value  # rung values ARE the model_type strings
    assert model_type in DYNAMICS_OPTIONS


def test_unknown_atmosphere_complexity_raises() -> None:
    with pytest.raises(ValueError):
        atmosphere_model_type("quasi_geostrophic")


# --- sea-ice complexity ladder: thermodynamic slab -> dynamic multi-category ---


def test_ice_complexity_rungs() -> None:
    assert [c.value for c in IceComplexity] == ["thermodynamic", "dynamic"]


@pytest.mark.parametrize(
    "complexity,expected_dynamics,expected_ncat",
    [
        (IceComplexity.THERMODYNAMIC, "none", 1),
        (IceComplexity.DYNAMIC, "evp", 5),
    ],
)
def test_ice_complexity_config_resolves(
    complexity, expected_dynamics, expected_ncat
) -> None:
    """Each ice rung builds a SeaIceConfig with the right dynamics + ITD."""
    from legoesm.driver.component_factory import ice_complexity_config

    cfg = ice_complexity_config(complexity)
    assert cfg.dynamics == expected_dynamics
    assert cfg.n_categories == expected_ncat


def test_ice_factory_consumes_complexity_rung() -> None:
    """create_ice_component returns the step bound to the resolved rung config."""
    import legoesm.ice as ice
    from legoesm.driver.component_factory import create_ice_component

    component = create_ice_component(
        config=None, grid=None, ice_config=IceComplexity.DYNAMIC
    )
    assert component.step is ice.step_sea_ice
    # The dynamic rung is CAPTURED in the returned config — not silently dropped,
    # so the caller cannot accidentally run thermodynamic defaults downstream.
    assert component.config.dynamics == "evp"
    assert component.config.n_categories == 5


def test_unknown_ice_complexity_raises() -> None:
    from legoesm.driver.component_factory import ice_complexity_config

    with pytest.raises(ValueError):
        ice_complexity_config("pancake")


# --- the model-wide complexity dial: one level -> all four component rungs ---


def test_model_complexity_levels() -> None:
    assert [c.value for c in ModelComplexity] == ["idealized", "intermediate", "full"]


@pytest.mark.parametrize(
    "level,atm,ocn,lnd,ice",
    [
        (
            ModelComplexity.IDEALIZED,
            AtmosphereComplexity.SHALLOW_WATER,
            OceanComplexity.FIXED_SST,
            LandComplexity.SLAB,
            IceComplexity.THERMODYNAMIC,
        ),
        (
            ModelComplexity.INTERMEDIATE,
            AtmosphereComplexity.HYDROSTATIC,
            OceanComplexity.SLAB,
            LandComplexity.MULTILAYER,
            IceComplexity.THERMODYNAMIC,
        ),
        (
            ModelComplexity.FULL,
            AtmosphereComplexity.HYDROSTATIC,
            OceanComplexity.FULL_3D,
            LandComplexity.MULTILAYER,
            IceComplexity.DYNAMIC,
        ),
    ],
)
def test_model_complexity_rungs_per_level(level, atm, ocn, lnd, ice) -> None:
    rungs = model_complexity_rungs(level)
    assert (rungs.atmosphere, rungs.ocean, rungs.land, rungs.ice) == (atm, ocn, lnd, ice)


@pytest.mark.parametrize("level", list(ModelComplexity))
def test_resolve_model_complexity_builds_every_component(level) -> None:
    """The driver resolver's spec is consumable by the real factories at EVERY
    level — full_3d ocean included (it builds a real OceanModel, the rung the
    ocean factory alone rejects). This is the end-to-end proof that the top
    complexity level is load-bearing, not just nameable."""
    from legoesm.atmosphere.dynamics import DYNAMICS_OPTIONS
    from legoesm.driver.component_factory import (
        create_land_component,
        create_ocean_component,
        resolve_model_complexity,
    )
    from legoesm.grids.factory import create_grid
    from legoesm.ocean.vertical import create_ocean_z_star

    spec = resolve_model_complexity(level, grid_type="cubed_sphere")

    # atmosphere: the resolved model_type is a real dycore option
    assert spec.atmosphere_model_type in DYNAMICS_OPTIONS

    # ocean: the spec's config builds a real component for every level — a step
    # closure for the simple rungs, a prognostic OceanModel for full_3d (which
    # needs a real ocean z-star coordinate; the simple rungs ignore it).
    grid = create_grid("cubed_sphere", 8)
    z_coord = create_ocean_z_star(n_levels=4)
    ocean = create_ocean_component(
        config=None, grid=grid, vertical_coord=z_coord, ocean_config=spec.ocean_config
    )
    assert ocean is not None

    # land: the rung builds a real land step fn
    assert callable(
        create_land_component(config=None, grid=None, land_config=spec.land_config)
    )

    # ice: the spec carries a concrete, already-resolved SeaIceConfig
    assert spec.ice_config.dynamics in ("none", "evp")


@pytest.mark.parametrize("grid_type", ["latlon", "mpas"])
def test_full_complexity_off_cubed_sphere_is_scoped(grid_type) -> None:
    """full_3d ocean is wired only for cubed_sphere; other grid families fail
    loudly rather than silently building the wrong (cubed-sphere) ocean class.
    The simple-ocean levels stay grid-agnostic and resolve fine everywhere."""
    from legoesm.driver.component_factory import resolve_model_complexity

    with pytest.raises(ValueError, match="cubed_sphere"):
        resolve_model_complexity(ModelComplexity.FULL, grid_type=grid_type)
    for level in (ModelComplexity.IDEALIZED, ModelComplexity.INTERMEDIATE):
        spec = resolve_model_complexity(level, grid_type=grid_type)
        assert spec.ocean_config is not None


def test_unknown_model_complexity_raises() -> None:
    from legoesm.driver.component_factory import resolve_model_complexity

    with pytest.raises(ValueError):
        model_complexity_rungs("kitchen_sink")
    with pytest.raises(ValueError):
        resolve_model_complexity("kitchen_sink")
