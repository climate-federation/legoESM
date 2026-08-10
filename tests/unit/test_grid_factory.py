"""Uniform grid instantiation — any global grid, usable by any component."""

from __future__ import annotations

import jax
import pytest

from legoesm.grids.factory import GLOBAL_GRID_TYPES, create_grid

_needs_x64 = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="dycore-on-grid check needs JAX_ENABLE_X64=1",
)


def test_global_grid_types() -> None:
    assert GLOBAL_GRID_TYPES == (
        "cubed_sphere",
        "fesom",
        "gaussian",
        "latlon",
        "mpas",
        "tripole",
    )


@pytest.mark.parametrize(
    "grid_type,resolution,kwargs,expected_cls",
    [
        ("cubed_sphere", 4, {}, "CubedSphereGrid"),
        ("gaussian", 21, {}, "GaussianGrid"),
        ("latlon", 16, {}, "LatLonGrid"),
        ("mpas", 1, {"lloyd_iterations": 2}, "VoronoiMesh"),
    ],
)
def test_create_each_global_grid(grid_type, resolution, kwargs, expected_cls) -> None:
    """Every global grid ocean and atmosphere share instantiates via one entry."""
    grid = create_grid(grid_type, resolution, **kwargs)
    assert grid is not None
    assert type(grid).__name__ == expected_cls


def test_per_grid_kwargs_pass_through() -> None:
    """Per-type options (e.g. latlon n_lon) reach the constructor."""
    grid = create_grid("latlon", 8, n_lon=24)
    # n_lon was honoured rather than defaulting to 2*n_lat.
    assert grid.n_lon == 24 or getattr(grid, "nlon", None) == 24


def test_resolution_grid_requires_resolution() -> None:
    with pytest.raises(ValueError, match="requires a resolution"):
        create_grid("cubed_sphere")


def test_tripole_is_file_backed() -> None:
    """The ocean OMIP tripole grid is covered, but needs a NEMO mesh file."""
    with pytest.raises(ValueError, match="grid_file"):
        create_grid("tripole")  # no grid_file -> clear, actionable error


def test_unknown_grid_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown grid_type"):
        create_grid("flat_earth", 4)


def test_plane_is_directed_elsewhere() -> None:
    with pytest.raises(ValueError, match="create_plane_grid"):
        create_grid("plane", 4)


@_needs_x64
def test_atmosphere_component_builds_on_a_factory_grid() -> None:
    """A factory-created grid feeds a real component (atmosphere dycore)."""
    from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_grid("cubed_sphere", 8)
    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=5),
        dycore=DycoreConfig(
            model_type="shallow_water", discretization="centered", dt=300.0
        ),
        days=1,
    )
    sigma = create_sigma_coordinate(5)
    dycore = create_atmosphere_dycore(config, grid, sigma)
    assert hasattr(dycore, "step")
