"""Discrimination tests for the sorted-RPE (RPE_mov) spurious-mixing metric.

The metric is ``run_ocean_test_matrix._compute_sorted_rpe`` -- the number the
lock-exchange matrix gate ``RPE_rel_mixing_sign`` PASSes/FAILs on. Each test
here fails for a specific broken implementation:

- ``test_homogenising_layers_raises_sorted_rpe``: a densest-at-SURFACE
  packing (the sign inversion this metric replaced) makes homogenisation
  LOWER the value; a constant-returning implementation returns 0 change.
  Level-swap invariance alone cannot catch either (codex 2026-08-09).
- ``test_rearrangement_invariance``: sorted RPE depends only on the
  (rho, vol) multiset, so an adiabatic rearrangement must not move it. The
  rearrangement is a LONGITUDE roll: cell area varies with latitude only and
  the z levels are stretched, so a lon roll is the volume-preserving
  permutation (a level swap is NOT -- it re-pairs rho with different dz).
- ``test_moving_volumes_enter_the_metric``: RPE_mov weights parcels with the
  MOVING z-star volume area*h(eta); a fixed-reference-volume (dz_ref)
  implementation is invariant to eta and fails this test. Moving volumes are
  the 2026-08-10 retraction: on the two bounded-front arms (FESOM FCT,
  tripole implicit_cn) fixed volumes gave a NEGATIVE drift while moving
  volumes gave the physical positive sign.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def matrix_mod():
    path = _REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py"
    sys.path.insert(0, str(_REPO / "scripts" / "matrix"))
    spec = importlib.util.spec_from_file_location("_rm_rpe_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rm_rpe_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def latlon_setup(matrix_mod):
    m = matrix_mod
    tc = SimpleNamespace(grid_type="latlon", resolution="8x16",
                         case="lock_exchange", run_kwargs={})
    grid, z_coord, _cfg, _model, _ck, _lon, _lat = m._create_ocean_setup(
        tc, nlev=6, H_max=20.0)
    state = m._create_rest_state(tc, grid, z_coord, H_max=20.0)
    return m, tc, grid, z_coord, state


def _with_stratified_T(state):
    """Stable stratification: T decreases with depth, uniform horizontally."""
    import jax.numpy as jnp
    T = np.asarray(state.T.data)
    prof = 20.0 - 2.0 * np.arange(T.shape[-1])
    T_new = np.broadcast_to(prof, T.shape).astype(T.dtype)
    return state._replace(T=state.T.replace(data=jnp.asarray(T_new)))


def test_homogenising_layers_raises_sorted_rpe(latlon_setup):
    import jax.numpy as jnp
    m, tc, grid, z_coord, state = latlon_setup
    st = _with_stratified_T(state)
    rpe0 = m._compute_sorted_rpe(st, tc.grid_type, grid, z_coord)

    T = np.asarray(st.T.data).copy()
    # VOLUME-weighted homogenisation (levels are stretched, so an arithmetic
    # mean would not be a pure mixing event -- codex 2026-08-10).
    w = np.asarray(z_coord.dz_ref)[2:4]
    T[..., 2:4] = np.sum(T[..., 2:4] * w, axis=-1, keepdims=True) / w.sum()
    st_mixed = st._replace(T=st.T.replace(data=jnp.asarray(T)))
    rpe1 = m._compute_sorted_rpe(st_mixed, tc.grid_type, grid, z_coord)

    assert rpe1 > rpe0, (
        f"homogenising two stably-stratified layers must RAISE sorted RPE "
        f"(mixing sign); got {rpe1 - rpe0:.3e}")
    assert (rpe1 - rpe0) / abs(rpe0) > 1e-10


def test_rearrangement_invariance(latlon_setup):
    import jax.numpy as jnp
    m, tc, grid, z_coord, state = latlon_setup
    st = _with_stratified_T(state)
    # Lon-dependent front so the roll below is a non-trivial rearrangement.
    T = np.asarray(st.T.data).copy()
    n_lon = T.shape[1]
    T += 5.0 * (np.arange(n_lon) < n_lon // 2)[np.newaxis, :, np.newaxis]
    st = st._replace(T=st.T.replace(data=jnp.asarray(T)))
    rpe0 = m._compute_sorted_rpe(st, tc.grid_type, grid, z_coord)

    # Roll along LONGITUDE: area varies with latitude only, dz with level
    # only, so this permutes parcels among equal-volume slots -- the
    # (rho, vol) multiset is unchanged and sorted RPE must not move.
    st_roll = st._replace(T=st.T.replace(
        data=jnp.asarray(np.roll(T, 3, axis=1))))
    rpe_roll = m._compute_sorted_rpe(st_roll, tc.grid_type, grid, z_coord)

    assert abs(rpe_roll - rpe0) / abs(rpe0) < 1e-12


def test_pack_sorted_rpe_kernel_discrimination():
    """Production kernel (legoesm.ocean.rpe.pack_sorted_rpe): mixing must
    RAISE the value. A densest-at-SURFACE packing (the pre-2026-08-10
    compute_rpe inline block) makes mixing LOWER it and fails here."""
    from legoesm.ocean.rpe import pack_sorted_rpe
    rho = np.array([1027.0, 1022.0])
    vol = np.array([50.0, 50.0])
    r0 = pack_sorted_rpe(rho, vol, total_area=10.0)
    rho_mixed = np.array([1024.5, 1024.5])
    r1 = pack_sorted_rpe(rho_mixed, vol, total_area=10.0)
    assert r1 > r0
    # Two-layer analytic check: dRPE = g*V1*V2*(z_top-z_bot)*(rho1-rho2)/(V1+V2)
    from legoesm import constants
    expected = constants.g * (50.0 * 50.0 / 100.0) * 5.0 * 5.0
    assert abs((r1 - r0) - expected) / expected < 1e-12


def test_moving_volumes_enter_the_metric(latlon_setup):
    import jax.numpy as jnp
    m, tc, grid, z_coord, state = latlon_setup
    st = _with_stratified_T(state)
    # T front in lon + eta bump CORRELATED with it: the warm parcels gain
    # volume, the cold ones lose it, so the (rho, vol) multiset -- and hence
    # RPE_mov -- must change. (An eta bump over horizontally-uniform T is
    # degenerate: equal-and-opposite volume changes within each density
    # class cancel and even the moving-volume metric legitimately stays put.)
    T = np.asarray(st.T.data).copy()
    n_lon = T.shape[1]
    warm = (np.arange(n_lon) < n_lon // 2)
    T += 5.0 * warm[np.newaxis, :, np.newaxis]
    st = st._replace(T=st.T.replace(data=jnp.asarray(T)))
    rpe0 = m._compute_sorted_rpe(st, tc.grid_type, grid, z_coord)

    eta = np.asarray(st.eta.data)
    bump = 0.5 * np.where(warm, 1.0, -1.0)[np.newaxis, :]
    st_eta = st._replace(eta=st.eta.replace(
        data=jnp.asarray(np.broadcast_to(bump, eta.shape).astype(eta.dtype))))
    rpe_eta = m._compute_sorted_rpe(st_eta, tc.grid_type, grid, z_coord)

    assert abs(rpe_eta - rpe0) / abs(rpe0) > 1e-8, (
        "an eta perturbation must move RPE_mov (moving z-star volumes); "
        "invariance means the implementation regressed to fixed dz_ref "
        "volumes")


# ---------------------------------------------------------------------------
# Which attribute the cell area comes from, per grid family
# ---------------------------------------------------------------------------
#
# _rpe_extract used to name "mpas" alone and let EVERY other grid fall
# through to `grid.area`. A VoronoiMesh carries `areaCell` and no `area`,
# so a regional or channel Voronoi arm raised AttributeError -- and
# run_lock_exchange calls this diagnostic before its first step, so such an
# arm would have died before integrating. Found while scoping a resolved
# lock-exchange arm on mpas_regional (codex 2026-08-13).

def _one_attr_grid(which: str, n=4):
    """A grid exposing ONLY one of the two area attributes.

    This is what makes the tests below able to tell them apart. Supplying
    BOTH -- which an earlier revision did -- means the call succeeds
    whichever one the code reads, so the test could not distinguish the
    families it claimed to be testing (codex 2026-08-13).
    """
    return SimpleNamespace(**{which: np.full(n, 2.0)})


def _minimal_state(n=4, nlev=2):
    """Enough state for _rpe_extract to run to completion.

    eta is required: the z-star branch calls compute_layer_thickness on it
    to build the layer depths.
    """
    return SimpleNamespace(
        T=SimpleNamespace(data=np.zeros((n, nlev))),
        S=SimpleNamespace(data=np.zeros((n, nlev))),
        eta=SimpleNamespace(data=np.zeros(n)),
        land_mask=SimpleNamespace(data=np.ones(n)))


def _z_coord(nlev=2):
    return SimpleNamespace(z_interfaces=np.linspace(0.0, -20.0, nlev + 1),
                           dz=np.full(nlev, 10.0),
                           z_centers=np.array([-5.0, -15.0])[:nlev])


def _area_attr_read(m, grid_type, which):
    """Which area attribute does _rpe_extract reach for on this grid_type?

    The grid exposes ONLY ``which``. The call is expected to fail LATER --
    the minimal state has no bathymetry and no real z-coordinate -- so what
    is asserted is narrow and exact: whether it died reaching for the area,
    and if so which name it wanted. Building a full valid state for five
    grid families would test the thickness code, not the dispatch this
    guard is about.

    Returns the missing attribute name, or None if the area lookup
    succeeded and the call got past it.
    """
    try:
        m._rpe_extract(_minimal_state(), grid_type, _one_attr_grid(which),
                       _z_coord())
    except AttributeError as exc:
        for name in ("areaCell", "area"):
            if f"'{name}'" in str(exc) or f"attribute {name}" in str(exc):
                return name
        return None       # died later, past the area lookup
    return None


@pytest.mark.parametrize("grid_type", ["mpas", "mpas_regional",
                                       "mpas_channel"])
def test_voronoi_grids_read_areaCell(matrix_mod, grid_type):
    """Given ONLY areaCell, the area lookup must succeed."""
    assert _area_attr_read(matrix_mod, grid_type, "areaCell") is None


@pytest.mark.parametrize("grid_type", ["latlon", "latlon_regional",
                                       "latlon_channel", "tripole", "fesom"])
def test_structured_grids_read_area(matrix_mod, grid_type):
    """Given ONLY area, the area lookup must succeed."""
    assert _area_attr_read(matrix_mod, grid_type, "area") is None


def test_the_attribute_is_load_bearing(matrix_mod):
    """NON-VACUITY. Hand each family the OTHER family's attribute and the
    lookup must fail, naming the one it wanted -- otherwise the two tests
    above would pass for any dispatch at all."""
    assert _area_attr_read(matrix_mod, "mpas", "area") == "areaCell"
    assert _area_attr_read(matrix_mod, "latlon", "areaCell") == "area"


def test_every_voronoi_grid_reads_area_from_areaCell(matrix_mod):
    """Not just the global one: regional and channel meshes too.

    The grid-family tables live in the shared energy-diagnostics module both
    matrix runners now call; ``matrix_mod`` only delegates to it."""
    spec = importlib.util.spec_from_file_location(
        "_rm_energy_diagnostics_test",
        _REPO / "scripts" / "matrix" / "ocean_test_matrix" / "energy_diagnostics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert "mpas" in m._MPAS_GRID_TYPES
    for g in ("mpas_regional", "mpas_channel"):
        assert g in m._MPAS_GRID_TYPES, (
            f"{g} is a Voronoi mesh and must read areaCell; falling through "
            f"to grid.area raises AttributeError before the first step")
    # And the two families must not overlap, or the branch order decides.
    assert not (set(m._MPAS_GRID_TYPES) & set(m._CELL_AREA_GRID_TYPES))


def test_an_unclassified_grid_raises_instead_of_guessing(matrix_mod):
    """NON-VACUITY for the hardened fall-through.

    The two families keep the cell area under DIFFERENT attribute names, so
    a default branch reads whichever happens to exist. A new grid type must
    be classified, not silently absorbed.
    """
    m = matrix_mod
    state = SimpleNamespace(T=SimpleNamespace(data=np.zeros((4, 2))),
                            S=SimpleNamespace(data=np.zeros((4, 2))),
                            land_mask=SimpleNamespace(data=np.ones(4)))
    grid = SimpleNamespace(area=np.ones(4), areaCell=np.ones(4))
    with pytest.raises(ValueError, match="unknown grid_type"):
        m._rpe_extract(state, "some_new_grid", grid, None)


def test_the_classified_grids_do_not_raise(matrix_mod):
    """The guard must not have swallowed the families it is meant to admit."""
    m = matrix_mod
    for g in ("mpas", "latlon"):
        state = SimpleNamespace(T=SimpleNamespace(data=np.zeros((4, 2))),
                                S=SimpleNamespace(data=np.zeros((4, 2))),
                                land_mask=SimpleNamespace(data=np.ones(4)))
        grid = SimpleNamespace(area=np.ones(4), areaCell=np.ones(4))
        # The minimal fixture has no eta/z_coord, so the call fails LATER
        # on purpose -- what is asserted is only that it got PAST the
        # classification, i.e. the guard admits these families. Any
        # exception is fine except the one that says it did not.
        try:
            m._rpe_extract(state, g, grid, None)
        except Exception as exc:             # noqa: BLE001 - see above
            assert "unknown grid_type" not in str(exc), (
                f"{g} must be classified, not rejected: {exc}")
