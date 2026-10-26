"""The wind unit vectors physics coupling needs (``update_dwinds_phys``).

These four fields — the local east/north pair at cell centres and the
unit tangents the D-grid ``u`` and ``v`` lie along at cell edges — are
the first grid quantities in this port that the dycore itself never
consumes, so nothing else exercises them.

WHAT THESE TESTS DO AND DO NOT ESTABLISH. They pin the geometric
identities the construction must satisfy: unit length, tangency to the
sphere at the point each vector belongs to, the orthogonality of the
east/north pair, the orientation of the edge tangents, and the exact
window upstream leaves unwritten. They do NOT establish agreement with
the oracle's own ``gridstruct%es``/``ew``/``vlon``/``vlat`` — that needs
the dumped runtime gridstruct and is the next step, not this file's job.
An invariant test passing is not a parity certificate, and the docstring
says so here rather than letting a green suite imply otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.grids.fv3_native_metrics import (
    compute_fv3_native_wind_vectors,
    latlon2xyz,
)

N, NG = 8, 3
M = N + 2 * NG


@pytest.fixture(scope="module")
def face():
    """One face's padded corner/centre lon-lat, from the real builder."""
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    gs = ctx["gs6"][0]
    return (gs["grid_lon"], gs["grid_lat"],
            gs["agrid_lon"], gs["agrid_lat"])


def test_centre_pair_is_an_orthonormal_tangent_frame(face):
    """vlon/vlat are the local east/north unit vectors at cell centres."""
    g_lon, g_lat, a_lon, a_lat = face
    out = compute_fv3_native_wind_vectors(g_lon, g_lat, a_lon, a_lat)
    vlon, vlat = out["vlon"], out["vlat"]
    assert vlon.shape == vlat.shape == (M, M, 3)
    assert np.all(np.isfinite(vlon)) and np.all(np.isfinite(vlat))

    one = np.ones((M, M))
    assert np.allclose(np.einsum("...i,...i", vlon, vlon), one,
                       rtol=0, atol=1e-14)
    assert np.allclose(np.einsum("...i,...i", vlat, vlat), one,
                       rtol=0, atol=1e-14)
    # east . north = 0, and both are tangent to the sphere at the centre.
    assert np.allclose(np.einsum("...i,...i", vlon, vlat), 0.0,
                       rtol=0, atol=1e-14)
    rad = latlon2xyz(np.stack([a_lon, a_lat], axis=-1))
    assert np.allclose(np.einsum("...i,...i", vlon, rad), 0.0,
                       rtol=0, atol=1e-14)
    assert np.allclose(np.einsum("...i,...i", vlat, rad), 0.0,
                       rtol=0, atol=1e-14)
    # NORTH, not south: the third component is +cos(lat), which is the
    # right-hand convention upstream keeps and the left-hand variant it
    # leaves commented out. A sign error here flips every meridional
    # wind the physics hands back and still looks plausible.
    assert np.allclose(vlat[..., 2], np.cos(a_lat), rtol=0, atol=1e-15)


def test_edge_tangents_are_unit_tangent_and_correctly_oriented(face):
    """es/ew are unit, tangent at the EDGE midpoint, and point i+ / j+."""
    g_lon, g_lat, a_lon, a_lat = face
    out = compute_fv3_native_wind_vectors(g_lon, g_lat, a_lon, a_lat)
    es1, ew2 = out["es1"], out["ew2"]
    assert es1.shape == (M, M + 1, 3)
    assert ew2.shape == (M + 1, M, 3)

    g3 = latlon2xyz(np.stack([g_lon, g_lat], axis=-1))

    def _mid(a, b):
        e = a + b
        return e / np.linalg.norm(e, axis=-1, keepdims=True)

    # es lives on the i-running edge between corners (i,j) and (i+1,j).
    e_s = es1[:, 1:-1, :]
    p_a, p_b = g3[:-1, 1:-1, :], g3[1:, 1:-1, :]
    assert np.allclose(np.einsum("...i,...i", e_s, e_s), 1.0,
                       rtol=0, atol=1e-14)
    assert np.allclose(np.einsum("...i,...i", e_s, _mid(p_a, p_b)), 0.0,
                       rtol=0, atol=1e-14)
    # ORIENTATION: from corner (i,j) toward (i+1,j), not the reverse.
    assert np.all(np.einsum("...i,...i", e_s, p_b - p_a) > 0.0)

    e_w = ew2[1:-1, :, :]
    q_a, q_b = g3[1:-1, :-1, :], g3[1:-1, 1:, :]
    assert np.allclose(np.einsum("...i,...i", e_w, e_w), 1.0,
                       rtol=0, atol=1e-14)
    assert np.allclose(np.einsum("...i,...i", e_w, _mid(q_a, q_b)), 0.0,
                       rtol=0, atol=1e-14)
    assert np.all(np.einsum("...i,...i", e_w, q_b - q_a) > 0.0)


def test_the_unwritten_window_is_nan_not_zero(face):
    """Upstream's loops start one index in; those slots stay undefined.

    A zero there is a finite, plausible unit vector that projects a wind
    to nothing, so a consumer reaching outside the written window would
    get a silently wrong answer instead of an error.
    """
    g_lon, g_lat, a_lon, a_lat = face
    out = compute_fv3_native_wind_vectors(g_lon, g_lat, a_lon, a_lat)
    es1, ew2 = out["es1"], out["ew2"]
    assert np.isnan(es1[:, 0, :]).all() and np.isnan(es1[:, -1, :]).all()
    assert np.isfinite(es1[:, 1:-1, :]).all()
    assert np.isnan(ew2[0, :, :]).all() and np.isnan(ew2[-1, :, :]).all()
    assert np.isfinite(ew2[1:-1, :, :]).all()
    # The window update_dwinds_phys actually reads (u over j = js..je+1,
    # v over i = is..ie+1) must be entirely inside the written part --
    # otherwise the NaN above would be a blocker, not a guard.
    assert np.isfinite(es1[NG:NG + N, NG:NG + N + 1, :]).all()
    assert np.isfinite(ew2[NG:NG + N + 1, NG:NG + N, :]).all()


def test_shape_mismatches_are_refused(face):
    g_lon, g_lat, a_lon, a_lat = face
    with pytest.raises(ValueError, match="corner"):
        compute_fv3_native_wind_vectors(g_lon, g_lat[:-1], a_lon, a_lat)
    with pytest.raises(ValueError, match="agrid"):
        compute_fv3_native_wind_vectors(g_lon, g_lat, a_lon[:-1, :-1],
                                        a_lat[:-1, :-1])


# --- the parity instrument's transform helpers (scripts/validate) ---
# compare_wind_vectors imports only numpy at module scope (its jax imports are
# inside main), so these pure-geometry helpers are unit-testable offline.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]
                        / "scripts" / "validate" / "fv3_native"))
from compare_wind_vectors import edge_expected_sign, crop_to_oracle  # noqa: E402
from compare_extchain_oracle import op_vector, OPS  # noqa: E402


def test_edge_expected_sign_matches_the_certified_op_vector():
    """edge_expected_sign must equal the orientation sign the certified covariant
    transform (op_vector) applies to the es(u)/ew(v) pair, for every dihedral op.

    op_vector on all-ones inputs returns exactly su/sv in the slot each field
    lands in (index transform of ones is ones), so the slot's constant value IS
    the orientation sign -- the ground truth for edge_expected_sign."""
    m = 5
    u = np.ones((m, m + 1))   # es-like (i-running edge)
    v = np.ones((m + 1, m))   # ew-like (j-running edge)
    for op in OPS:
        sw = op[0]
        uo, vo = op_vector(u, v, op)
        es_slot = vo if sw else uo   # es lands in oracle-v under a swap, else -u
        ew_slot = uo if sw else vo
        assert np.allclose(es_slot, edge_expected_sign("es1", op)), op
        assert np.allclose(ew_slot, edge_expected_sign("ew2", op)), op


def test_crop_to_oracle_trims_one_ring_and_refuses_asymmetry():
    a = np.arange(54 * 54, dtype=float).reshape(54, 54)
    # vlon case: port halo 3 (54) -> oracle halo 2 (52): drop one ring each side
    c = crop_to_oracle(a, (52, 52))
    assert c.shape == (52, 52)
    assert np.array_equal(c, a[1:-1, 1:-1])
    # es case: already matching (no trim)
    b = np.zeros((54, 55))
    assert crop_to_oracle(b, (54, 55)) is b or crop_to_oracle(b, (54, 55)).shape == (54, 55)
    # a non-symmetric / negative gap is refused
    with pytest.raises(SystemExit):
        crop_to_oracle(a, (53, 54))   # odd i-gap
    with pytest.raises(SystemExit):
        crop_to_oracle(a, (56, 56))   # port smaller than oracle
