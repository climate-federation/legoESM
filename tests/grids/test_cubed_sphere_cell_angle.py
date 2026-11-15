"""Cube cell-centre grid angle: FV3 ``get_center_vect`` form, all six faces.

The former lat/lon centred-difference angle was 13.7 deg off within ~4 cells
of each pole at every resolution (non-converging); the Cartesian FV3 form
converges at second order on every face.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.grids.halo import compute_padded_angle, face_gnomonic_to_lonlat  # noqa: E402


def _xyz(lo, la):
    return jnp.stack([jnp.cos(la) * jnp.cos(lo), jnp.cos(la) * jnp.sin(lo),
                      jnp.sin(la)], axis=-1)


def _east_north(lo, la):
    e = jnp.stack([-jnp.sin(lo), jnp.cos(lo), 0.0 * lo], axis=-1)
    nn = jnp.stack([-jnp.sin(la) * jnp.cos(lo), -jnp.sin(la) * jnp.sin(lo),
                    jnp.cos(la)], axis=-1)
    return e, nn


def _ang_err_deg(a, b):
    return np.abs(np.degrees((np.asarray(a) - np.asarray(b) + np.pi)
                             % (2 * np.pi) - np.pi))


def _exact_equiangular_angle(n, halo):
    """Exact i-direction at each padded centre: d(position)/d(alpha_x),
    Cartesian finite difference with h = 1e-7 rad (no lon singularity)."""
    da = jnp.pi / (2 * n)
    a = -jnp.pi / 4 - halo * da + da * (jnp.arange(n + 2 * halo) + 0.5)
    ax, ay = jnp.meshgrid(a, a, indexing="ij")
    h = 1e-7
    out = []
    for f in range(6):
        lo, la = face_gnomonic_to_lonlat(f, ax, ay)
        t = (_xyz(*face_gnomonic_to_lonlat(f, ax + h, ay))
             - _xyz(*face_gnomonic_to_lonlat(f, ax - h, ay)))
        e, nn = _east_north(lo, la)
        out.append(jnp.arctan2(jnp.sum(t * nn, -1), jnp.sum(t * e, -1)))
    return jnp.stack(out)


@pytest.mark.parametrize("halo", [1, 2])
def test_equiangular_angle_converges_on_every_face(halo):
    errs = []
    for n in (12, 24, 48):
        d = _ang_err_deg(compute_padded_angle(n, halo=halo),
                         _exact_equiangular_angle(n, halo))
        errs.append([float(d[f].max()) for f in range(6)])
    errs = np.asarray(errs)
    print("max angle error deg per face:", errs)
    # second order on all six faces (the old form plateaued at 13.7 deg on 4-5)
    assert np.all(errs[0] / errs[1] > 3.5) and np.all(errs[1] / errs[2] > 3.5), errs
    assert errs[2].max() < 0.05, errs


def test_ed_angle_converges_on_every_face():
    """gnomonic_ed: compare with the Cartesian chord direction between the
    i-neighbour centres (independent construction, 2nd-order)."""
    from legoesm.grids.cubed_sphere import (
        compute_padded_angle_ed, gnomonic_ed_padded_centers)
    errs = []
    for n in (12, 24, 48):
        a = compute_padded_angle_ed(n, halo=1)
        lo, la = gnomonic_ed_padded_centers(n, 1)          # (6, n+4, n+4)
        X = _xyz(lo, la)
        t = X[:, 2:, 1:-1] - X[:, :-2, 1:-1]
        e, nn = _east_north(lo[:, 1:-1, 1:-1], la[:, 1:-1, 1:-1])
        ref = jnp.arctan2(jnp.sum(t * nn, -1), jnp.sum(t * e, -1))
        d = _ang_err_deg(a, ref)
        errs.append([float(d[f].max()) for f in range(6)])
    errs = np.asarray(errs)
    print("ed max angle diff deg per face:", errs)
    assert np.all(errs[0] / errs[1] > 3.0) and np.all(errs[1] / errs[2] > 3.0), errs
