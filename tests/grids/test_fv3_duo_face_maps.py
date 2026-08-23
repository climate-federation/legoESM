"""The panel-boundary metrics must respond to what their names claim.

Two metrics live in the map script: an in-panel boundary-strip roughness
(a high-pass ratio on the field) and a residual-location number (the same
ratio on |port - oracle|).  Neither gates anything, but both get quoted,
so both are pinned here -- including the exact cells the ring covers,
because the first version of the ring walked one cell further in than its
docstring said and no test could see it.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

_MOD = pathlib.Path(__file__).resolve().parents[2] / (
    "scripts/plot/fv3_duo_face_maps.py")
_spec = importlib.util.spec_from_file_location("fv3_duo_face_maps", _MOD)
fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fm)

N = 24
W = 3


def _smooth_plane():
    x = np.linspace(0.0, 1.0, N)
    return np.outer(np.exp(3.0 * x), np.cos(2.0 * x))


def test_ring_mask_is_the_frame_it_says_it_is():
    m = fm.ring_mask((N, N), W)
    idx = np.indices((N, N))
    depth = np.minimum(np.minimum(idx[0], N - 1 - idx[0]),
                       np.minimum(idx[1], N - 1 - idx[1]))
    assert np.array_equal(m, depth < W)


def test_the_field_ring_is_offset_for_the_high_pass_shrink():
    """The high-pass ring must sit on the ORIGINAL boundary cells.

    ``high_pass`` returns a plane shrunk by one on each side, so its row
    r is centred on original row r+1 and a ring of ``width`` original
    cells is ``width - 1`` cells of the high-pass.  These two index
    assertions are the whole claim, and the second one fails on the
    version this replaced, which used ``width`` there and so reached one
    cell further in than its docstring said.
    """
    hp_shape = (N - 2, N - 2)
    m = fm.ring_mask(hp_shape, W - 1)
    assert m[W - 2, N // 2], "original cell W-1 is not in the ring"
    assert not m[W - 1, N // 2], "original cell W is in the ring"


def test_smooth_field_has_no_edge_excess():
    e, i, r = fm.edge_interior_ratio(_smooth_plane(), W)
    assert e > 0.0 and i > 0.0
    assert r < 3.0, f"a smooth exponential-times-cosine scored {r}"


def _checker(plane, mask, frac=0.05):
    out = plane.copy()
    chk = np.indices(plane.shape).sum(axis=0) % 2
    out[mask] += frac * float(np.abs(plane).max()) * (2 * chk[mask] - 1)
    return out


def test_grid_scale_noise_on_the_ring_is_detected():
    plane = _smooth_plane()
    ring = np.zeros((N, N), dtype=bool)
    ring[:W, :] = ring[-W:, :] = True
    ring[:, :W] = ring[:, -W:] = True
    _, _, r_clean = fm.edge_interior_ratio(plane, W)
    _, _, r_noisy = fm.edge_interior_ratio(_checker(plane, ring), W)
    assert r_noisy > 10.0 * r_clean, (
        f"edge 2dx noise moved the ratio only {r_clean} -> {r_noisy}")


def test_interior_noise_does_not_masquerade_as_an_edge_artifact():
    """Interior noise must RAISE the interior term, not the edge one."""
    plane = _smooth_plane()
    core = np.zeros((N, N), dtype=bool)
    core[N // 2 - 3:N // 2 + 3, N // 2 - 3:N // 2 + 3] = True
    e_c, i_c, r_c = fm.edge_interior_ratio(plane, W)
    e_n, i_n, r_n = fm.edge_interior_ratio(_checker(plane, core), W)
    assert e_n == pytest.approx(e_c), "interior noise moved the EDGE term"
    assert i_n > i_c and r_n < r_c


def test_a_proportional_error_is_invisible_to_the_field_ratio():
    """Pinned as a LIMIT, not a feature.

    port = 1.01 * oracle has exactly the oracle's ratio, so the
    descriptive number reports no excess on a field that is one percent
    wrong everywhere. This is why it does not gate; if a future version
    claims it can gate, this test says what it must first fix.
    """
    plane = _smooth_plane()
    _, _, r_o = fm.edge_interior_ratio(plane, W)
    _, _, r_p = fm.edge_interior_ratio(1.01 * plane, W)
    assert r_p == pytest.approx(r_o)


def test_residual_location_number_finds_a_boundary_residual():
    oracle = _smooth_plane()
    even = oracle + 1e-6
    assert fm.edge_concentration(even, oracle, W) == pytest.approx(1.0)

    edge_only = oracle.copy()
    edge_only[:W, :] += 1e-6
    edge_only[-W:, :] += 1e-6
    edge_only[:, :W] += 1e-6
    edge_only[:, -W:] += 1e-6
    assert fm.edge_concentration(edge_only, oracle, W) == float("inf")

    interior_only = oracle.copy()
    interior_only[N // 2, N // 2] += 1e-6
    assert fm.edge_concentration(interior_only, oracle, W) == 0.0


def test_bitwise_equal_planes_report_evenly_spread_not_a_crash():
    plane = _smooth_plane()
    assert fm.edge_concentration(plane, plane, W) == 1.0


def test_a_smooth_edge_bias_survives_the_residual_number():
    """The residual number must NOT high-pass the error field.

    A constant offset on the boundary strip is a real edge defect and a
    second difference would annihilate it. This is the case that decides
    between the two designs.
    """
    oracle = _smooth_plane()
    biased = oracle.copy()
    biased[:W, :] += 1e-6
    assert fm.edge_concentration(biased, oracle, W) > 1.0


def test_width_one_is_refused_rather_than_silently_empty():
    with pytest.raises(SystemExit):
        fm.edge_interior_ratio(_smooth_plane(), 1)


def test_panel_too_small_refuses_rather_than_overlapping():
    with pytest.raises(SystemExit):
        fm.edge_interior_ratio(np.zeros((2 * W + 2, 2 * W + 2)), W)
