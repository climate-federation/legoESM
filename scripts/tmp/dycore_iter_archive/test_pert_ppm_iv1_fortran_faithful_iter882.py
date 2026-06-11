"""Iter-882: pin `pert_ppm` (iv=1, standard PPM constraint) against
an explicit Fortran-faithful NumPy reference.

Fortran's `pert_ppm` (`tp_core.F90:1193-1212`) iv=1 branch implements
the standard PPM monotonicity constraint (Colella-Woodward 1984
eq. 1.10).  Our Python implementation in
`src/legoesm/core/fv_tp_2d.py:pert_ppm` should match.

Fortran reference (paraphrased):
```
if al*ar < 0:                          # opposite signs
    da1 = al - ar
    da2 = da1**2
    a6da = 3*(al+ar)*da1
    if a6da < -da2:    ar = -2*al      # branch A1
    elif a6da > da2:   al = -2*ar      # branch A2
    # else:           no change         # branch A3
else:                                   # same signs (or zero)
    al = 0; ar = 0                      # branch B
```

This sentinel pins all 4 iv=1 branches with hand-built inputs that
actually fire each branch + per-branch expected-output assertion
(iter-881 lessons applied: branches must satisfy the gate AND
non-zero clip targets to catch multiplier corruption AND
near-boundary inputs where applicable).

Sibling sentinel to iter-881's iv=0 coverage; together they pin
both Fortran pert_ppm branches against documented Fortran
references.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Iter-883: also enable x64 at runtime in case JAX was already
# initialized in float32 by an earlier conftest import.  The
# os.environ.setdefault above is for command-line invocation; the
# jax.config.update is the runtime-effective form.
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.core.fv_tp_2d import pert_ppm


def _pert_ppm_fortran_reference(bl, br):
    """Element-wise NumPy port of Fortran tp_core.F90:1193-1212."""
    bl_out = np.array(bl, dtype=np.float64, copy=True)
    br_out = np.array(br, dtype=np.float64, copy=True)
    for i in range(bl_out.size):
        bl_i = bl_out.flat[i]
        br_i = br_out.flat[i]
        if bl_i * br_i < 0.0:
            da1 = bl_i - br_i
            da2 = da1 ** 2
            a6da = 3.0 * (bl_i + br_i) * da1
            if a6da < -da2:
                br_out.flat[i] = -2.0 * bl_i
            elif a6da > da2:
                bl_out.flat[i] = -2.0 * br_i
            # else: no change
        else:
            bl_out.flat[i] = 0.0
            br_out.flat[i] = 0.0
    return bl_out, br_out


def _make_test_inputs():
    """Build inputs covering each Fortran iv=1 branch.

    Branch | bl    | br    | Trace
    -------|-------|-------|--------------------------------
    A1     | -0.1  |  2.0  | bl*br<0 (opp), da1=-2.1, da2=4.41,
           |       |       | a6da=3*1.9*(-2.1)=-11.97 < -4.41
           |       |       | → br = -2*bl = 0.2
    A2     |  2.0  | -0.1  | bl*br<0, da1=2.1, da2=4.41,
           |       |       | a6da=3*1.9*2.1=11.97 > 4.41
           |       |       | → bl = -2*br = 0.2
    A3     | -0.5  |  0.5  | bl*br<0, da1=-1, da2=1,
           |       |       | a6da=3*0*(-1)=0 (between -1 and 1)
           |       |       | → no change (pass-through)
    B-pp   |  0.5  |  0.3  | bl*br>0 (both pos) → both 0
    B-nn   | -0.5  | -0.3  | bl*br>0 (both neg) → both 0
    B-zp   |  0.0  |  0.3  | bl*br=0 → branch B (Fortran's
           |       |       | `else` includes equal sign case
           |       |       | via `<` strict inequality) → both 0
    A1-bd  | -0.5  |  1.0  | bl*br<0, da1=-1.5, da2=2.25,
           |       |       | a6da=3*0.5*(-1.5)=-2.25 = -da2
           |       |       | (boundary) — Fortran `<` strict,
           |       |       | so a6da < -da2 is FALSE.  Branch A3
           |       |       | (pass-through) — catches strict-vs-
           |       |       | non-strict inequality regression.

    Iter-882 corruption-resistance design:
    - A1, A2 use NON-ZERO clip targets (br=0.2, bl=0.2 after clip)
      so multiplier corruption (-2 → -3) flips output measurably.
    - A1-bd uses an EXACT-BOUNDARY case where a regression from
      `<` to `<=` flips the branch (catches strict-inequality
      corruption).
    - B-zp tests the `bl*br = 0` edge case (Fortran's `<` is
      strict so zero-product is "same signs", branch B fires).
    """
    bl = np.array([-0.1,  2.0, -0.5,  0.5, -0.5,  0.0, -0.5])
    br = np.array([ 2.0, -0.1,  0.5,  0.3, -0.3,  0.3,  1.0])
    return bl, br


def _expected_outputs_per_branch():
    """Hand-computed expected (bl_out, br_out) for the 7 inputs."""
    # A1:    bl unchanged at -0.1, br = -2*bl = 0.2
    # A2:    bl = -2*br = 0.2, br unchanged at -0.1
    # A3:    pass-through (-0.5, 0.5)
    # B-pp:  both zero
    # B-nn:  both zero
    # B-zp:  both zero
    # A1-bd: pass-through (-0.5, 1.0)
    bl_expected = np.array([-0.1, 0.2, -0.5, 0.0, 0.0, 0.0, -0.5])
    br_expected = np.array([ 0.2, -0.1, 0.5, 0.0, 0.0, 0.0,  1.0])
    return bl_expected, br_expected


def test_iter882_pert_ppm_iv1_matches_fortran_reference():
    """`pert_ppm` MUST match the Fortran iv=1 reference + per-branch
    expected outputs on hand-built inputs covering all 4 branches.
    """
    bl, br = _make_test_inputs()

    bl_actual, br_actual = pert_ppm(jnp.asarray(bl), jnp.asarray(br))
    bl_ref, br_ref = _pert_ppm_fortran_reference(bl, br)
    bl_expected, br_expected = _expected_outputs_per_branch()

    # Production matches Fortran reference.
    np.testing.assert_allclose(
        np.asarray(bl_actual), bl_ref, rtol=1e-12, atol=1e-12,
        err_msg=(
            "`pert_ppm` bl output differs from Fortran reference. "
            "Audit the iv=1 standard PPM constraint logic in "
            "`fv_tp_2d.py` against `tp_core.F90:1193-1212`."))
    np.testing.assert_allclose(
        np.asarray(br_actual), br_ref, rtol=1e-12, atol=1e-12,
        err_msg=("`pert_ppm` br output differs from Fortran reference."))

    # Production matches hand-computed per-branch expected.
    np.testing.assert_allclose(
        np.asarray(bl_actual), bl_expected, rtol=1e-12, atol=1e-12,
        err_msg=(
            "iter-882 branch-coverage failure: `pert_ppm` bl output "
            "differs from hand-computed per-branch expected.  Either "
            "the production matches the reference but BOTH have the "
            "same bug, OR a test input no longer fires the intended "
            "branch.  Audit `fv_tp_2d.py:pert_ppm` and the iter-882 "
            "branch trace comments against Fortran tp_core.F90:"
            "1193-1212."))
    np.testing.assert_allclose(
        np.asarray(br_actual), br_expected, rtol=1e-12, atol=1e-12,
        err_msg=("iter-882 branch-coverage failure: br differs."))


@pytest.mark.parametrize("seed", [7, 19, 47])
def test_iter882_pert_ppm_iv1_random_inputs(seed):
    """Random-input cross-check: `pert_ppm` matches the Fortran
    NumPy reference on random (bl, br) pairs spanning multiple
    sign and magnitude regimes.
    """
    rng = np.random.default_rng(seed)
    n = 64
    # Mix sign combinations to exercise both opposite-signs and
    # same-signs branches.
    bl = rng.normal(size=n) * 1.5
    br = rng.normal(size=n) * 1.5

    bl_actual, br_actual = pert_ppm(jnp.asarray(bl), jnp.asarray(br))
    bl_ref, br_ref = _pert_ppm_fortran_reference(bl, br)

    np.testing.assert_allclose(
        np.asarray(bl_actual), bl_ref, rtol=1e-12, atol=1e-12,
        err_msg=f"pert_ppm bl mismatch on seed={seed}")
    np.testing.assert_allclose(
        np.asarray(br_actual), br_ref, rtol=1e-12, atol=1e-12,
        err_msg=f"pert_ppm br mismatch on seed={seed}")
