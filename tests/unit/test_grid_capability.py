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
    # global-only spectral grid, regional-only mercator, double-periodic plane.
    assert supported_extents("gaussian") == frozenset({"global"})
    assert supported_extents("mercator") == frozenset({"regional"})
    assert supported_extents("plane") == frozenset({"double_periodic"})
    # lat-lon and cubed-sphere are both global AND regional.
    assert supported_extents("latlon") == frozenset({"global", "regional"})
    assert supported_extents("cubed_sphere") == frozenset({"global", "regional"})


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


def test_nesting_is_flagged_unavailable():
    with pytest.raises(NotImplementedError, match="nesting is not"):
        instantiate("latlon", extent="regional", nesting=True, n_lat=8, n_lon=8)


def test_extents_constant():
    assert EXTENTS == ("global", "regional", "double_periodic")
