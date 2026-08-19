"""Tests for the km-general six-face state container.

Unit 1+2 of the staged 3-D duo port. Nothing here is physics; it is the
layout contract every later unit is checked against, so each assertion is
paired with the synthetic violation that must trip it.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.fv3_native_state_3d import (
    MAX_KM_WITHOUT_REMAP,
    STATE_FIELDS,
    build_state_3d,
    field_shape,
    level_slice,
    require_no_remap_needed,
    state_signature,
    validate_state_3d,
)

N, NG, KM = 6, 3, 3
M_A = N + 2 * NG
M_B = M_A + 1


# --------------------------------------------------------- the remap guard

def test_km_up_to_four_is_allowed_because_the_oracle_does_not_remap():
    """fv_dynamics.F90:568 gates Lagrangian_to_Eulerian on npz > 4."""
    for km in (1, 2, 3, 4):
        require_no_remap_needed(km)


def test_km_above_four_is_refused_with_the_reason():
    with pytest.raises(ValueError, match=r"fv_dynamics.F90:568"):
        require_no_remap_needed(5)
    with pytest.raises(ValueError, match="theta_v"):
        require_no_remap_needed(32)


def test_the_guard_boundary_is_exactly_four():
    require_no_remap_needed(MAX_KM_WITHOUT_REMAP)
    with pytest.raises(ValueError):
        require_no_remap_needed(MAX_KM_WITHOUT_REMAP + 1)


def test_nonsense_km_is_refused():
    for bad in (0, -1, 2.5, "3", None):
        with pytest.raises(ValueError):
            require_no_remap_needed(bad)


def test_build_state_honours_the_guard():
    with pytest.raises(ValueError, match="fv_mapz"):
        build_state_3d(N, NG, 8)


# ------------------------------------------------------- the layout contract

def test_the_staggers_are_distinct_and_not_interchangeable():
    """u and v have DIFFERENT shapes; a port that treats them alike would
    broadcast rather than fail."""
    assert field_shape("u", N, NG, KM) == (M_A, M_B, KM)
    assert field_shape("v", N, NG, KM) == (M_B, M_A, KM)
    assert field_shape("u", N, NG, KM) != field_shape("v", N, NG, KM)
    # C winds are the transpose pair of the D winds
    assert field_shape("uc", N, NG, KM) == field_shape("v", N, NG, KM)
    assert field_shape("vc", N, NG, KM) == field_shape("u", N, NG, KM)


def test_interface_fields_carry_km_plus_one_not_km():
    """Assuming one vertical extent for every field is the obvious way to
    get this wrong."""
    assert field_shape("pk", N, NG, KM)[-1] == KM + 1
    assert field_shape("gz", N, NG, KM)[-1] == KM + 1
    assert field_shape("delp", N, NG, KM)[-1] == KM
    assert field_shape("pkz", N, NG, KM)[-1] == KM


def test_pe_and_peln_keep_the_load_bearing_i_k_j_order():
    """fv3_native_pgrad declares (i,k,j) load-bearing for its bit-exact
    compare. If someone restages to (i,j,k) this must go red."""
    assert field_shape("pe", N, NG, KM) == (N + 2, KM + 1, N + 2)
    assert field_shape("peln", N, NG, KM) == (N, KM + 1, N)
    # the k axis is in the MIDDLE, not last
    assert field_shape("pe", N, NG, KM)[1] == KM + 1
    assert field_shape("peln", N, NG, KM)[1] == KM + 1


def test_unknown_field_is_refused_rather_than_guessed():
    with pytest.raises(KeyError, match="add it to field_shape"):
        field_shape("not_a_field", N, NG, KM)


# ------------------------------------------------------------ build/validate

def test_build_state_shapes_and_dtype():
    st = build_state_3d(N, NG, KM)
    assert len(st) == 6
    for face in st:
        assert set(face) == set(STATE_FIELDS)
        for name, a in face.items():
            assert a.shape == field_shape(name, N, NG, KM)
            assert a.dtype == np.float64
    validate_state_3d(st, N, NG, KM)


def test_faces_are_independent_arrays():
    """Six references to one array would make every face identical and
    every cross-face barrier a no-op."""
    st = build_state_3d(N, NG, KM)
    st[0]["delp"][0, 0, 0] = 1.0
    assert st[1]["delp"][0, 0, 0] == 0.0
    assert all(st[0]["delp"] is not f["delp"] for f in st[1:])


def test_validate_rejects_a_wrong_shape():
    st = build_state_3d(N, NG, KM)
    st[2]["delp"] = np.zeros((M_A, M_A, KM + 1))
    with pytest.raises(ValueError, match=r"face 3 field 'delp'"):
        validate_state_3d(st, N, NG, KM)


def test_validate_rejects_a_swapped_u_v_stagger():
    """The exact slip the shape contract exists to catch."""
    st = build_state_3d(N, NG, KM)
    st[0]["u"], st[0]["v"] = st[0]["v"], st[0]["u"]
    with pytest.raises(ValueError, match=r"face 1 field '[uv]'"):
        validate_state_3d(st, N, NG, KM)


def test_validate_rejects_float32():
    st = build_state_3d(N, NG, KM)
    st[0]["pt"] = st[0]["pt"].astype(np.float32)
    with pytest.raises(TypeError, match="float64"):
        validate_state_3d(st, N, NG, KM)


def test_validate_rejects_nonfinite():
    st = build_state_3d(N, NG, KM)
    st[4]["w"][1, 1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_state_3d(st, N, NG, KM)


def test_validate_rejects_a_missing_field():
    st = build_state_3d(N, NG, KM)
    del st[3]["pt"]
    with pytest.raises(KeyError, match="face 4"):
        validate_state_3d(st, N, NG, KM)


def test_validate_rejects_the_wrong_number_of_faces():
    st = build_state_3d(N, NG, KM)
    with pytest.raises(ValueError, match="6 per-face dicts"):
        validate_state_3d(st[:5], N, NG, KM)


def test_validate_accepts_the_state_it_built_at_every_allowed_km():
    for km in (1, 2, 3, 4):
        validate_state_3d(build_state_3d(N, NG, km), N, NG, km)


# --------------------------------------------------------------- level_slice

def test_level_slice_returns_2d_views_the_kernels_can_take():
    st = build_state_3d(N, NG, KM)
    lev = level_slice(st[0], 1, KM)
    assert lev["delp"].shape == (M_A, M_A)
    assert lev["u"].shape == (M_A, M_B)
    assert lev["v"].shape == (M_B, M_A)
    assert all(a.ndim == 2 for a in lev.values())


def test_level_slice_is_a_view_so_in_place_kernels_update_the_state():
    """The Fortran dummies are intent(inout); a copy here would silently
    discard every stage's output."""
    st = build_state_3d(N, NG, KM)
    lev = level_slice(st[0], 2, KM)
    lev["delp"][3, 4] = 7.0
    assert st[0]["delp"][3, 4, 2] == pytest.approx(7.0)


def test_level_slice_touches_only_its_own_level():
    st = build_state_3d(N, NG, KM)
    level_slice(st[0], 1, KM)["pt"][:] = 5.0
    assert np.all(st[0]["pt"][:, :, 1] == 5.0)
    assert np.all(st[0]["pt"][:, :, 0] == 0.0)
    assert np.all(st[0]["pt"][:, :, 2] == 0.0)


def test_level_slice_rejects_an_out_of_range_level():
    st = build_state_3d(N, NG, KM)
    for bad in (-1, KM, KM + 5):
        with pytest.raises(IndexError):
            level_slice(st[0], bad, KM)


# ----------------------------------------------------------------- signature

def test_signature_is_deterministic_and_change_sensitive():
    st = build_state_3d(N, NG, KM)
    a = state_signature(st)
    assert state_signature(st) == a
    st[2]["pt"][0, 0, 0] += 1e-9
    assert state_signature(st) != a


def test_signature_distinguishes_which_face_moved():
    st = build_state_3d(N, NG, KM)
    a = state_signature(st)
    st[4]["delp"][0, 0, 0] = 1.0
    b = state_signature(st)
    changed = [k for k in a if a[k] != b[k]]
    assert changed == ["t5.delp"]
