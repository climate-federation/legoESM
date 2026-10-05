"""The EEN vertex thickness on a tripolar fold row.

NEMO gives that row no formula of its own.  ``dyn_vor_init`` evaluates the
masked four-cell average over the owned domain
(``ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynvor.f90:913-919``),
completes the field with the ordinary F-point north-fold exchange, sign +1
(``:935``), and only then substitutes the reference thickness for any
remaining zero (``:937``).  Under a T pivot that exchange rewrites the LAST
OWNED row from the row immediately below it at the mirrored longitude: the
compiled row loop runs to ``ipj - ihls`` with source row ``ipj - ihls - 1``,
and the compiled longitude loop pairs ``ii1 + ii2 = ipi + 1``
(``lbcnfd.f90:722-746``).

The pairing is restated here from the compiled loop rather than re-using the
production permutation, so an inverted index shift in the helper fails.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_geometry
from legoesm.grids.tripole import (
    create_synthetic_tripole, create_synthetic_tripole_pivot,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import een_e3f_h_vtx

jax.config.update("jax_enable_x64", True)

N_LAT, N_LON, N_LEV = 12, 24, 3


def _thickness(n_lat: int, n_lon: int) -> jnp.ndarray:
    """A wet-everywhere-but-a-coast thickness field with no symmetry."""
    ladder = jnp.asarray([10.0, 25.0, 60.0], dtype=jnp.float64)
    column = (jnp.arange(n_lat * n_lon, dtype=jnp.float64).reshape(n_lat, n_lon)
              % 7.0 + 1.0)
    thickness = column[:, :, None] * ladder[None, None, :]
    dry = jnp.zeros((1, 1, N_LEV), dtype=jnp.float64)
    return thickness.at[3:5, 6:9].set(dry)


def _dz_ref() -> jnp.ndarray:
    return jnp.asarray([10.0, 25.0, 60.0], dtype=jnp.float64)


def _compiled_pairing(n_lon: int) -> np.ndarray:
    """``ii1 + ii2 = ipi + 1`` in this array's west-shifted vertex columns.

    NEMO's F column ``i`` is vertex column ``i + 1`` here, so the compiled
    zero-based pairing ``i_dst + i_src = Ni0glo - 1`` becomes
    ``c_dst + c_src = Ni0glo + 1`` modulo the longitude count.
    """
    return np.asarray([(n_lon + 1 - c) % n_lon for c in range(n_lon)])


@pytest.mark.parametrize("scheme", ["nemo_avg4", "nemo_avg", "min"])
def test_fold_row_is_defined_for_every_vertex_thickness_rule(scheme):
    """No rule may refuse the fold row; ``nemo_avg4`` used to."""
    grid = create_synthetic_tripole(N_LAT, N_LON)
    h_vtx, _, _ = een_e3f_h_vtx(
        _thickness(N_LAT, N_LON), None, None, grid, scheme, dz_ref=_dz_ref())
    assert h_vtx.shape == (N_LAT + 1, N_LON + 1, N_LEV)
    assert bool(jnp.all(jnp.isfinite(h_vtx)))


def test_nemo_avg4_fold_row_is_the_row_below_at_the_mirrored_longitude():
    """dynvor.f90:935 through lbcnfd.f90:722-746, restated independently."""
    grid = create_synthetic_tripole(N_LAT, N_LON)
    h_vtx, _, _ = een_e3f_h_vtx(
        _thickness(N_LAT, N_LON), None, None, grid, "nemo_avg4",
        dz_ref=_dz_ref())
    core = np.asarray(h_vtx)[:, :N_LON]
    expected = core[-2][_compiled_pairing(N_LON)]
    assert np.array_equal(core[-1].view(np.uint64), expected.view(np.uint64))
    # The appended wrap column stays a copy of column zero.
    assert np.array_equal(np.asarray(h_vtx)[:, N_LON], np.asarray(h_vtx)[:, 0])


def test_the_pairing_assertion_can_fail():
    """Synthetic violation: the T-origin mirror must NOT satisfy it.

    Without this the test above would pass for any self-consistent rule.
    """
    grid = create_synthetic_tripole(N_LAT, N_LON)
    h_vtx, _, _ = een_e3f_h_vtx(
        _thickness(N_LAT, N_LON), None, None, grid, "nemo_avg4",
        dz_ref=_dz_ref())
    core = np.asarray(h_vtx)[:, :N_LON]
    t_origin = core[-2][np.asarray([(-c) % N_LON for c in range(N_LON)])]
    assert not np.array_equal(core[-1], t_origin)
    own_row = core[-1][_compiled_pairing(N_LON)]
    assert not np.array_equal(core[-1], own_row)


def test_only_the_fold_row_moves():
    """Everything below the fold row is the ordinary interior average."""
    grid = create_synthetic_tripole(N_LAT, N_LON)
    thickness = _thickness(N_LAT, N_LON)
    tripolar, _, _ = een_e3f_h_vtx(
        thickness, None, None, grid, "nemo_avg4", dz_ref=_dz_ref())
    flat = create_latlon_geometry(N_LAT, N_LON)
    regular, _, _ = een_e3f_h_vtx(
        thickness, None, None, flat, "nemo_avg4", dz_ref=_dz_ref())
    assert np.array_equal(np.asarray(tripolar)[:-1], np.asarray(regular)[:-1])
    assert not np.array_equal(
        np.asarray(tripolar)[-1], np.asarray(regular)[-1])


def test_a_closed_grid_takes_no_fold_branch_at_all():
    """GYRE's identity: an inactive fold leaves the helper byte-unchanged."""
    flat = create_latlon_geometry(N_LAT, N_LON)
    assert not bool(getattr(flat.fold, "is_active", False))
    built, _, _ = een_e3f_h_vtx(
        _thickness(N_LAT, N_LON), None, None, flat, "nemo_avg4",
        dz_ref=_dz_ref())
    assert bool(jnp.all(jnp.isfinite(built)))


def test_a_stored_pivot_mesh_routes_each_scheme_to_its_own_fold_branch():
    """The 2026-09-25 merge had to choose between two fold-row statements.

    One lane refused ``nemo_avg4`` on every tripolar fold; the other had
    transcribed the row from NEMO.  A third statement arrived with GitHub
    main: a branch for a STORED-PIVOT mesh, which is what the ORCA2 mesh
    classifies as.  This pins the routing that resolution produced, so a
    later edit cannot quietly send one scheme down the other's branch.
    """
    from legoesm.grids.operators_latlon_cgrid import fold_perm_f

    grid = create_synthetic_tripole_pivot(N_LAT, N_LON)
    assert bool(grid.fold.pivot_row_stored), "fixture must be a pivot mesh"
    thickness = _thickness(N_LAT, N_LON)

    avg4 = np.asarray(een_e3f_h_vtx(
        thickness, None, None, grid, "nemo_avg4", dz_ref=_dz_ref())[0])
    avg = np.asarray(een_e3f_h_vtx(
        thickness, None, None, grid, "nemo_avg", dz_ref=_dz_ref())[0])

    # nemo_avg4 keeps NEMO's own F-origin pairing in the shifted vertex
    # columns, on a pivot mesh exactly as on any other.
    assert np.array_equal(avg4[-1, :N_LON], avg4[-2, :N_LON][
        _compiled_pairing(N_LON)])
    # nemo_avg on the same mesh takes main's pivot-row branch instead: the
    # row below permuted with the descriptor's own F map, no column shift.
    assert np.array_equal(avg[-1, :N_LON],
                          avg[-2, :N_LON][np.asarray(fold_perm_f(grid.fold))])
    # The two index maps really are different statements on this mesh, so
    # neither assertion above can be satisfied by the other's branch.
    assert not np.array_equal(_compiled_pairing(N_LON),
                              np.asarray(fold_perm_f(grid.fold)))


def test_nemo_avg4_fold_row_does_not_depend_on_the_storage_layout():
    """Why the merge retires the refusal rather than keying it on the layout.

    ``nemo_t_fold_f_owned`` reads NEMO's F origin directly instead of the
    storage convention detected for generic scalar exchange, so the ORCA2
    transcription is the same statement whether or not the classifier calls
    the mesh stored-pivot.  If that ever stops being true, keying a refusal
    on the layout would become meaningful again and this fails.
    """
    pivot = create_synthetic_tripole_pivot(N_LAT, N_LON)
    assert bool(pivot.fold.pivot_row_stored), "fixture must be a pivot mesh"
    legacy = pivot._replace(fold=pivot.fold._replace(
        pivot_row_stored=False, perm_u=None, perm_f=None))
    thickness = _thickness(N_LAT, N_LON)
    a = een_e3f_h_vtx(thickness, None, None, pivot, "nemo_avg4",
                      dz_ref=_dz_ref())[0]
    b = een_e3f_h_vtx(thickness, None, None, legacy, "nemo_avg4",
                      dz_ref=_dz_ref())[0]
    assert np.array_equal(np.asarray(a).view(np.uint64),
                          np.asarray(b).view(np.uint64))
