"""Controls for the shear squaring-order probe.

The probe answers whether our Richardson number is too large because we
average velocity to cell centres before squaring the vertical difference,
where NEMO squares at each face first. Its whole value is the RATIO it
reports, so it is checked here against cases whose answer is known in
closed form before it is believed on the ocean.

The coastal case is the one that matters: the first version of the probe
averaged the NEMO side over two faces unconditionally while the centred side
divided by the number of wet neighbours, so a column with one dry face
returned exactly 0.5 and violated Jensen. It was caught by the caller's
assertion on the probe's first run, and `test_coastal_one_face_is_unity`
pins it.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.compare_tendencies_nemo import (
    shear_order_squared,
)


def _field(u_by_x, z=2, ny=3, nx=4):
    """Velocity pair with a prescribed vertical difference per x-face.

    Level 0 carries ``u_by_x`` and level 1 is zero, so the vertical
    difference at face i is exactly ``u_by_x[i]``. v is zero everywhere and
    contributes nothing to either order.
    """
    un = np.zeros((z, ny, nx))
    un[0, :, :] = u_by_x
    return un, np.zeros((z, ny, nx))


def test_uniform_shear_is_unity():
    """Equal faces -> the two orders agree exactly; Jensen's equality case."""
    n, o = shear_order_squared(*_field([2.0, 2.0, 2.0, 2.0]))
    r = (n / o)[:, 0]
    assert np.allclose(r, 1.0)


def test_alternating_faces_match_closed_form():
    """a=3, b=1 -> 0.5(a^2+b^2) / ((a+b)/2)^2 = 5/4.

    A probe that merely returned 1.0 everywhere would pass the uniform test,
    so this is the one that shows it measures the Jensen gap at all.
    """
    n, o = shear_order_squared(*_field([1.0, 3.0, 1.0, 3.0]))
    got = (n / o).reshape(3, 4, 1)[1, 1, 0]
    assert got == pytest.approx(1.25, rel=1e-12)


def test_coastal_one_face_is_unity():
    """One dry face -> both orders see the same single face, so ratio is 1.

    Returned 0.5 before the face sets were made to agree. 0.5 is not a
    Jensen gap at all, it is a denominator mismatch, and it would have been
    read as the centred order DOUBLING the shear next to every coast.
    """
    un, vn = _field([1.0, 3.0, 1.0, 3.0])
    un[:, :, 0] = np.nan
    n, o = shear_order_squared(un, vn)
    got = (n / o).reshape(3, 4, 1)[1, 1, 0]
    assert got == pytest.approx(1.0, rel=1e-12)


def test_jensen_floor_holds_on_random_field_with_land():
    """The bound the caller asserts, exercised where the real data lives."""
    rng = np.random.default_rng(0)
    un = rng.normal(size=(5, 8, 9))
    vn = rng.normal(size=(5, 8, 9))
    un[:, 2, 3] = np.nan
    vn[:, 5, 6] = np.nan
    n, o = shear_order_squared(un, vn)
    m = o > 1e-20
    assert m.any()
    assert (n[m] / o[m]).min() >= 1.0 - 1e-9


def test_probe_would_fail_if_face_sets_disagreed():
    """Non-vacuity: reintroducing the original bug must break the floor.

    The NEMO side is re-formed with the unconditional /2 the first version
    used, against the same mask-aware centred side, and the coastal column
    must go back to 0.5. Without this, the tests above only show the fixed
    probe is self-consistent, not that they can detect the defect.
    """
    un, vn = _field([1.0, 3.0, 1.0, 3.0])
    un[:, :, 0] = np.nan
    _, o = shear_order_squared(un, vn)

    v = np.nan_to_num(un, nan=0.0)
    fin = np.isfinite(un).astype(np.float64)
    face = fin[:-1] * fin[1:]
    dv = (v[:-1] - v[1:]) * face
    buggy = 0.5 * (dv ** 2 + np.roll(dv ** 2, 1, axis=-1))   # always /2
    buggy = np.transpose(buggy.reshape(1, 3 * 4), (1, 0))

    got = (buggy / o).reshape(3, 4, 1)[1, 1, 0]
    assert got == pytest.approx(0.5, rel=1e-12)
