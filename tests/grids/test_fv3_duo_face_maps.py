"""The cube-edge artifact metric must respond to an edge artifact.

A high-pass ratio is only worth printing if it is large exactly when
grid-scale structure sits on the panel boundary and small when the field
is smooth however steep.  Both directions are asserted here, on synthetic
planes whose answer is known by construction, because a metric that only
ever returns "fine" would certify a broken port.
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


def test_smooth_field_has_no_edge_excess():
    e, i, r = fm.edge_interior_ratio(_smooth_plane(), W)
    assert e > 0.0 and i > 0.0
    assert r < 3.0, f"a smooth exponential-times-cosine scored {r}"


def test_grid_scale_noise_on_the_ring_is_detected():
    plane = _smooth_plane()
    noisy = plane.copy()
    checker = np.indices((N, N)).sum(axis=0) % 2
    amp = 0.05 * float(np.abs(plane).max())
    ring = np.zeros((N, N), dtype=bool)
    ring[:W + 1, :] = ring[-W - 1:, :] = True
    ring[:, :W + 1] = ring[:, -W - 1:] = True
    noisy[ring] += amp * (2 * checker[ring] - 1)

    _, _, r_clean = fm.edge_interior_ratio(plane, W)
    _, _, r_noisy = fm.edge_interior_ratio(noisy, W)
    assert r_noisy > 10.0 * r_clean, (
        f"edge 2dx noise moved the ratio only {r_clean} -> {r_noisy}")


def test_interior_noise_does_not_masquerade_as_an_edge_artifact():
    """The ratio must not fire on noise that is NOT at the boundary."""
    plane = _smooth_plane()
    noisy = plane.copy()
    checker = np.indices((N, N)).sum(axis=0) % 2
    amp = 0.05 * float(np.abs(plane).max())
    core = np.zeros((N, N), dtype=bool)
    core[N // 2 - 3:N // 2 + 3, N // 2 - 3:N // 2 + 3] = True
    noisy[core] += amp * (2 * checker[core] - 1)

    _, _, r_clean = fm.edge_interior_ratio(plane, W)
    _, _, r_noisy = fm.edge_interior_ratio(noisy, W)
    assert r_noisy < r_clean, (
        "interior noise raised the EDGE ratio, so the metric is not "
        "localising what its name claims")


def test_panel_too_small_refuses_rather_than_overlapping():
    with pytest.raises(SystemExit):
        fm.edge_interior_ratio(np.zeros((2 * W + 2, 2 * W + 2)), W)
