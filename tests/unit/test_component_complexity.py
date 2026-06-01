"""The ocean/land complexity ladder — grounded against the real factories.

Ocean: fixed_sst -> slab(1L) -> slab_multilayer(2L) -> full_3d.
Land:  slab -> multilayer (column).
"""

from __future__ import annotations

import pytest

from legoesm.components import LandComplexity, OceanComplexity, ocean_simple_mode


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
