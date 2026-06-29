"""Capability matrix + validated grid x extent x operators x nesting factory."""

import pytest

from legoesm.grids.capability import (
    EXTENTS,
    capability_matrix,
    instantiate,
    operator_family,
    supported_extents,
)


# ---------------------------------------------------------------------------
# Matrix data
# ---------------------------------------------------------------------------

def test_capability_matrix_shape():
    m = capability_matrix()
    # Every grid family is present with the three columns.
    for g in ("latlon", "cubed_sphere", "plane", "mpas", "gaussian", "tripole"):
        assert g in m, g
        assert set(m[g]) == {"extents", "operator_family", "nesting_extents"}


def test_extent_support_is_derived_from_the_factory():
    # Every grid supports the 0-D "column" extent; horizontal extents are derived
    # from the factory's grid-type sets.
    assert supported_extents("gaussian") == frozenset({"global", "column"})
    assert supported_extents("mercator") == frozenset({"regional", "column"})
    assert supported_extents("plane") == frozenset({"double_periodic", "column"})
    assert supported_extents("latlon") == frozenset({"global", "regional", "column"})
    assert supported_extents("cubed_sphere") == frozenset({"global", "regional", "column"})


def test_operator_families():
    assert operator_family("latlon") == "grid_operators"
    assert operator_family("cubed_sphere") == "grid_operators"
    assert operator_family("mpas") == "edge_operators"
    assert operator_family("plane") == "plane_operators"
    assert operator_family("gaussian") is None  # spectral — no grid-local ops


def test_icosahedral_is_an_alias_for_mpas():
    assert supported_extents("icosahedral") == supported_extents("mpas")
    assert operator_family("icosahedral") == "edge_operators"
    assert operator_family("xy") == operator_family("plane")


# ---------------------------------------------------------------------------
# Valid instantiation
# ---------------------------------------------------------------------------

def test_global_grids_build():
    for g in ("latlon", "cubed_sphere", "mpas", "gaussian"):
        grid = instantiate(g, extent="global", resolution=4)
        assert grid is not None


def test_double_periodic_plane_builds():
    grid = instantiate(
        "plane", extent="double_periodic", nx=8, ny=8, nlev=4, dx=1e3, dy=1e3,
    )
    assert grid is not None


def test_global_with_operators_returns_pair():
    grid, ops = instantiate("latlon", extent="global", resolution=4, operators=True)
    assert grid is not None and ops is not None
    # the adapter satisfies the GridOperators contract
    from legoesm.grids.operator_protocol import validate_grid_operators
    validate_grid_operators(ops)


def test_mpas_with_operators_is_edge_operators():
    grid, ops = instantiate("mpas", extent="global", resolution=2, operators=True)
    from legoesm.grids.operator_protocol import validate_edge_operators
    validate_edge_operators(ops)


def test_cubed_sphere_operators_are_callable():
    # Regression: instantiate(cubed_sphere, operators=True) used to hand back an
    # adapter wrapping the bare CubedSphereGrid (no C-D metrics), which raised
    # AttributeError 'rdxc' on the FIRST operator call.  The capability path now
    # converts to the cdgrid, so the operators must actually be callable.
    import jax.numpy as jnp
    from legoesm.grids.operator_protocol import validate_grid_operators
    grid, ops = instantiate("cubed_sphere", extent="global", resolution=8,
                            operators=True)
    assert grid is not None and ops is not None
    validate_grid_operators(ops)
    gx, gy = ops.gradient(jnp.ones((6, 8, 8)))  # must not AttributeError
    assert gx.shape[0] == 6 and gy.shape[0] == 6


def test_regional_cubed_sphere_operators_raises_not_implemented():
    # Regional cubed-sphere panels need panel-local C-D metrics (not the global
    # six-face supergrid metrics); operators=True must raise a clear
    # requires-implementation error, not silently discard the panel's CDGrid or
    # build six-face operators for a one-face panel.
    with pytest.raises(ValueError, match="[Ii]mplement"):
        instantiate("cubed_sphere", extent="regional", operators=True, n=8)


# ---------------------------------------------------------------------------
# Error cases — the whole point: flag unavailable combinations
# ---------------------------------------------------------------------------

def test_unknown_grid_raises():
    with pytest.raises(ValueError, match="Unknown grid_type"):
        instantiate("ekman_spiral", extent="global")


def test_unknown_extent_raises():
    with pytest.raises(ValueError, match="Unknown extent"):
        instantiate("latlon", extent="planetary")


def test_regional_spectral_is_rejected():
    # Gaussian/spectral is global-only — a regional spectral grid does not exist.
    with pytest.raises(ValueError, match="does not support extent 'regional'"):
        instantiate("gaussian", extent="regional", resolution=4)


def test_global_mercator_is_rejected():
    # Mercator is a regional construction; there is no global mercator grid.
    with pytest.raises(ValueError, match="does not support extent 'global'"):
        instantiate("mercator", extent="global")


def test_double_periodic_only_for_plane():
    with pytest.raises(ValueError, match="does not support extent 'double_periodic'"):
        instantiate("latlon", extent="double_periodic", resolution=4)


def test_operators_on_spectral_grid_raises():
    # Gaussian/spectral has no grid-local differential operators.
    with pytest.raises(ValueError, match="no grid-local differential operators"):
        instantiate("gaussian", extent="global", resolution=4, operators=True)


def test_plane_operators_live_with_the_component_not_a_substrate_adapter():
    # The doubly-periodic plane's operators are the Cartesian operators in the
    # atmosphere component, not a substrate adapter — instantiate flags this
    # rather than returning a half-built pair.
    with pytest.raises(ValueError, match="plane.*operators are Cartesian"):
        instantiate(
            "plane", extent="double_periodic", operators=True,
            nx=8, ny=8, nlev=4, dx=1e3, dy=1e3,
        )


def test_nesting_unavailable_for_non_latlon_grids():
    # Nesting is implemented only for lat-lon (a one-way parent->child nest);
    # every other grid still raises a precise NotImplementedError.
    with pytest.raises(NotImplementedError, match="nesting is not"):
        instantiate("cubed_sphere", extent="regional", nesting=True, n=12)


def test_latlon_nesting_builds_nested_grid():
    from legoesm.grids.nesting import NestedLatLonGrid

    nest = instantiate(
        "latlon", extent="regional", nesting=True,
        parent_n_lat=24, refinement_ratio=2,
        lat_south_deg=0.0, lat_north_deg=40.0,
        lon_west_deg=0.0, lon_east_deg=80.0,
    )
    assert isinstance(nest, NestedLatLonGrid)
    assert nest.refinement_ratio == 2


def test_latlon_nesting_in_capability_matrix():
    m = capability_matrix()
    assert "regional" in m["latlon"]["nesting_extents"]
    assert m["cubed_sphere"]["nesting_extents"] == []


def test_nesting_with_operators_rejected():
    with pytest.raises(ValueError, match="pair"):
        instantiate(
            "latlon", extent="regional", nesting=True, operators=True,
            parent_n_lat=24, refinement_ratio=2,
            lat_south_deg=0.0, lat_north_deg=40.0,
            lon_west_deg=0.0, lon_east_deg=80.0,
        )


def test_extents_constant():
    assert EXTENTS == ("global", "regional", "double_periodic", "column")


# ---------------------------------------------------------------------------
# Runtime axes: architecture x precision x time integrator
# ---------------------------------------------------------------------------

from legoesm.grids.capability import (
    ARCHITECTURES,
    available_integrators,
    available_precision_modes,
    validate_runtime,
)


def test_runtime_axes_sets():
    assert set(ARCHITECTURES) == {"cpu", "gpu", "tpu", "mps"}
    assert {"fp32", "fp64", "mixed"} <= set(available_precision_modes())
    ints = available_integrators()
    assert "ssp_rk3" in ints and "rk4" in ints


def test_validate_runtime_accepts_valid_combos():
    validate_runtime("cpu", "fp64", "ssp_rk3")
    validate_runtime("gpu", "mixed", "rk4")
    validate_runtime(architecture="mps", precision="fp32")  # fp32 ok on mps
    validate_runtime()  # all None — no-op


def test_unknown_architecture_precision_integrator_raise():
    with pytest.raises(ValueError, match="Unknown architecture"):
        validate_runtime(architecture="quantum")
    with pytest.raises(ValueError, match="Unknown precision"):
        validate_runtime(precision="fp128")
    with pytest.raises(ValueError, match="Unknown time_integrator"):
        validate_runtime(time_integrator="forward_euler")


def test_fp64_on_metal_is_rejected():
    # fp64 storage needs an fp64-capable backend; the Apple GPU (mps) has none.
    with pytest.raises(ValueError, match="no float64 support"):
        validate_runtime(architecture="mps", precision="fp64")
    with pytest.raises(ValueError, match="no float64 support"):
        validate_runtime(architecture="mps", precision="mixed_fp64_storage")


def test_instantiate_validates_runtime_before_building():
    # A bad runtime axis is caught before any grid is built (validate-only).
    with pytest.raises(ValueError, match="no float64 support"):
        instantiate("latlon", extent="global", resolution=4,
                    architecture="mps", precision="fp64")
    with pytest.raises(ValueError, match="Unknown time_integrator"):
        instantiate("cubed_sphere", extent="global", resolution=4,
                    time_integrator="leapfrog_3")


def test_instantiate_with_valid_runtime_builds():
    # Valid runtime axes (validate-only, configure=False) + a real grid.
    grid = instantiate("latlon", extent="global", resolution=4,
                       architecture="cpu", precision="fp32", time_integrator="ssp_rk3")
    assert grid is not None


def test_fp64_without_named_architecture_checks_current_backend():
    # architecture=None falls back to the CURRENT backend: if it has no float64
    # (e.g. Metal), an fp64-storage precision must still be rejected, not slip
    # through to a silent float32 truncation (codex review).
    from unittest import mock
    import legoesm.runtime.backend as backend

    with mock.patch.object(backend, "supports_float64", return_value=False):
        with pytest.raises(ValueError, match="no float64 support"):
            validate_runtime(precision="fp64")
        with pytest.raises(ValueError, match="no float64 support"):
            validate_runtime(precision="mixed_fp64_storage")
        # fp32 / mixed (fp32 storage) stay fine on an fp64-less backend.
        validate_runtime(precision="fp32")
        validate_runtime(precision="mixed")

    # With float64 available, fp64 validates.
    with mock.patch.object(backend, "supports_float64", return_value=True):
        validate_runtime(precision="fp64")


# ---------------------------------------------------------------------------
# 0-D column extent + per-component complexity (slab) axis
# ---------------------------------------------------------------------------

from legoesm.grids.capability import (
    COMPONENTS,
    component_complexities,
    validate_complexity,
)


def test_column_extent_is_universal():
    # Every grid family supports the 0-D single-column extent.
    for g in ("latlon", "cubed_sphere", "mpas", "gaussian", "plane"):
        assert "column" in supported_extents(g)


def test_column_builds_single_column_grid():
    from legoesm.core.grid_adapters import SingleColumnGrid
    col = instantiate("latlon", extent="column")
    assert isinstance(col, SingleColumnGrid)
    # works for any component family — a column ocean / atmosphere / land / ice.
    assert isinstance(instantiate("mpas", extent="column"), SingleColumnGrid)


def test_column_rejects_operators():
    with pytest.raises(ValueError, match="0-D.*no horizontal"):
        instantiate("latlon", extent="column", operators=True)


def test_component_complexity_rungs():
    assert "slab" in component_complexities("ocean")
    assert "slab_multilayer" in component_complexities("ocean")
    assert set(component_complexities("land")) == {"slab", "multilayer"}
    assert "thermodynamic" in component_complexities("ice")
    assert "hydrostatic" in component_complexities("atmosphere")


def test_validate_complexity_rejects_wrong_rung():
    validate_complexity("ocean", "slab")
    validate_complexity("land", "multilayer")
    with pytest.raises(ValueError, match="not a rung of the land ladder"):
        validate_complexity("land", "full_3d")          # ocean rung, not land
    with pytest.raises(ValueError, match="not a rung of the atmosphere ladder"):
        validate_complexity("atmosphere", "slab")        # atm slab = column extent
    with pytest.raises(ValueError, match="Unknown component"):
        validate_complexity("biosphere", "slab")


def test_instantiate_validates_component_complexity():
    # A slab ocean column: valid component+complexity, 0-D.
    from legoesm.core.grid_adapters import SingleColumnGrid
    col = instantiate("mpas", extent="column", component="ocean", complexity="slab")
    assert isinstance(col, SingleColumnGrid)
    # complexity without a component is ambiguous.
    with pytest.raises(ValueError, match="complexity requires a component"):
        instantiate("latlon", extent="global", resolution=4, complexity="slab")
    # an invalid rung is caught before the grid builds.
    with pytest.raises(ValueError, match="not a rung of the land ladder"):
        instantiate("latlon", extent="column", component="land", complexity="full_3d")
