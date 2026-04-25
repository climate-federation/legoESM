"""Iter-885: pin `_edge_interpolate4` against an explicit Fortran-
faithful NumPy reference.

`_edge_interpolate4` (`src/legoesm/core/fv3_sw_core.py`) is a port
of Fortran's `edge_interpolate4` (`sw_core.F90:3709-3720`) — a
4-point non-uniform-spacing Lagrange-style interpolation to the
interface between cells 2 and 3 of a 4-cell stencil.  Used at the
FV3 cube-face boundaries by `_d2a2c_vect` to compute transport
velocities in face-boundary regions where the standard 4-cell
uniform stencil isn't applicable.

Fortran reference (`sw_core.F90:3709-3720`):
```
t1 = dxa(1) + dxa(2)
t2 = dxa(3) + dxa(4)
edge_interpolate4 = 0.5*( ((t1+dxa(2))*ua(2) - dxa(2)*ua(1)) / t1 +
                          ((t2+dxa(3))*ua(3) - dxa(3)*ua(4)) / t2 )
```

Index mapping: Fortran 1-based → Python 0-based:
- Fortran ua(1..4) → Python ua4[0..3]
- Fortran dxa(1..4) → Python dxa4[0..3]

This sentinel pins the function with:
1. Hand-built test inputs covering edge cases (uniform spacing,
   non-uniform spacing, scalar input, vectorized input).
2. Cross-Fortran reference comparison at 1e-12 rtol.
3. Random-input cross-check across multiple seeds.

Iter-885 lessons applied (from iter-881/882 chain):
- Use NON-ZERO ua values so corruption of multipliers is visible.
- Use NON-UNIFORM dxa values so spacing-dependent terms are
  exercised (uniform spacing makes (t1+dxa(2))/t1 = 1.5 etc. —
  numerically identical to a different formula).
- Verify behaviour on a deterministic check value derived
  analytically.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.core.fv3_sw_core import _edge_interpolate4


def _edge_interpolate4_fortran_reference(ua4, dxa4):
    """Element-wise NumPy port of Fortran sw_core.F90:3709-3720."""
    ua4 = np.asarray(ua4, dtype=np.float64)
    dxa4 = np.asarray(dxa4, dtype=np.float64)
    t1 = dxa4[..., 0] + dxa4[..., 1]
    t2 = dxa4[..., 2] + dxa4[..., 3]
    return 0.5 * (
        ((t1 + dxa4[..., 1]) * ua4[..., 1] - dxa4[..., 1] * ua4[..., 0]) / t1
        + ((t2 + dxa4[..., 2]) * ua4[..., 2] - dxa4[..., 2] * ua4[..., 3]) / t2
    )


def test_iter885_edge_interpolate4_uniform_spacing():
    """With uniform spacing dxa = [d, d, d, d], the function reduces
    to a specific weighted average of ua[1] and ua[2] adjusted by
    ua[0] and ua[3].

    For dxa = [1, 1, 1, 1]:
      t1 = 2, t2 = 2
      first  = ((2+1)*ua[1] - 1*ua[0]) / 2 = (3*ua[1] - ua[0]) / 2
      second = ((2+1)*ua[2] - 1*ua[3]) / 2 = (3*ua[2] - ua[3]) / 2
      result = 0.5 * (first + second)
             = 0.25 * (3*ua[1] - ua[0] + 3*ua[2] - ua[3])

    For ua = [10, 20, 30, 40]:
      result = 0.25 * (60 - 10 + 90 - 40) = 0.25 * 100 = 25
    """
    ua4 = jnp.array([10.0, 20.0, 30.0, 40.0])
    dxa4 = jnp.array([1.0, 1.0, 1.0, 1.0])

    actual = float(_edge_interpolate4(ua4, dxa4))
    expected = 25.0

    assert abs(actual - expected) < 1e-12, (
        f"Uniform-spacing edge_interpolate4 result {actual} differs "
        f"from analytical expected {expected}.")


def test_iter885_edge_interpolate4_non_uniform_spacing():
    """With non-uniform spacing, the formula's spacing-dependent
    terms are genuinely exercised.

    For dxa = [1, 2, 3, 4], ua = [10, 20, 30, 40]:
      t1 = 1 + 2 = 3
      t2 = 3 + 4 = 7
      first  = ((3+2)*20 - 2*10) / 3 = (100 - 20) / 3 = 80/3 ≈ 26.667
      second = ((7+3)*30 - 3*40) / 7 = (300 - 120) / 7 = 180/7 ≈ 25.714
      result = 0.5 * (26.667 + 25.714) = 0.5 * 52.381 ≈ 26.190

    Exact: 0.5 * (80/3 + 180/7) = 0.5 * ((80*7 + 180*3)/(3*7))
                                = 0.5 * ((560 + 540)/21)
                                = 0.5 * (1100/21)
                                = 550/21
    """
    ua4 = jnp.array([10.0, 20.0, 30.0, 40.0])
    dxa4 = jnp.array([1.0, 2.0, 3.0, 4.0])

    actual = float(_edge_interpolate4(ua4, dxa4))
    expected = 550.0 / 21.0

    assert abs(actual - expected) < 1e-12, (
        f"Non-uniform-spacing edge_interpolate4 result {actual} "
        f"differs from analytical expected {expected:.12f} (= 550/21). "
        f"Audit `_edge_interpolate4` against Fortran sw_core.F90:"
        f"3709-3720 — the formula MUST exercise spacing-dependent "
        f"terms via t1, t2, dxa[1], dxa[2].")


def test_iter885_edge_interpolate4_matches_fortran_reference():
    """Hand-built input + Fortran NumPy reference comparison."""
    # 4 distinct test cases stacked on the leading axis.
    ua4 = jnp.array([
        [1.0, 2.0, 3.0, 4.0],         # monotonic ramp
        [10.0, 20.0, 30.0, 40.0],     # uniform
        [-5.0, 0.0, 5.0, 10.0],       # crosses zero
        [100.0, 99.0, 98.0, 97.0],    # decreasing
    ])
    dxa4 = jnp.array([
        [1.0, 1.0, 1.0, 1.0],         # uniform
        [1.0, 2.0, 3.0, 4.0],         # increasing
        [4.0, 3.0, 2.0, 1.0],         # decreasing
        [2.0, 1.0, 1.0, 2.0],         # symmetric
    ])

    actual = _edge_interpolate4(ua4, dxa4)
    expected = _edge_interpolate4_fortran_reference(ua4, dxa4)

    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=1e-12, atol=1e-12,
        err_msg=(
            "`_edge_interpolate4` output differs from Fortran "
            "reference.  Audit against sw_core.F90:3709-3720."))


@pytest.mark.parametrize("seed", [11, 23, 51])
def test_iter885_edge_interpolate4_random_inputs(seed):
    """Random-input cross-check at 1e-12 rtol across batched inputs.
    The function operates on the LAST axis, so shape (n_samples, 4)
    exercises the broadcasting form."""
    rng = np.random.default_rng(seed)
    n_samples = 32
    ua4 = jnp.asarray(rng.normal(size=(n_samples, 4)))
    # Strictly positive dxa to avoid divide-by-zero in t1/t2.
    dxa4 = jnp.asarray(rng.uniform(0.1, 5.0, size=(n_samples, 4)))

    actual = _edge_interpolate4(ua4, dxa4)
    expected = _edge_interpolate4_fortran_reference(ua4, dxa4)

    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=1e-12, atol=1e-12,
        err_msg=f"_edge_interpolate4 mismatch on seed={seed}")
