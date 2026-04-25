"""Iter-881: pin `_pert_ppm_iv0` (positive-definite PPM constraint)
against an explicit Fortran-faithful NumPy reference.

Fortran's `pert_ppm` (`tp_core.F90:1169-1192`) iv=0 branch implements
the positive-definite constraint used by hord=9 (FV3 default for
mass, vorticity, momentum transport).  Our Python implementation
in `src/legoesm/core/fv_tp_2d.py:_pert_ppm_iv0` should match.

Fortran reference (paraphrased):
```
if (a0 <= 0):
    al = 0; ar = 0           # zero-out non-positive
else:
    a4 = -3*(ar + al)
    da1 = ar - al
    if abs(da1) < -a4:        # parabola has extremum in [0,1]
        fmin = a0 + 0.25/a4*da1**2 + a4*r12
        if fmin < 0:           # parabola minimum negative
            if ar>0 and al>0: # both perturbations positive
                ar = 0; al = 0
            elif da1 > 0:      # positive slope: clip ar
                ar = -2*al
            else:              # negative slope: clip al
                al = -2*ar
```

This test pins the iv=0 path with a hand-computed reference on a
synthetic input that exercises every branch:
- Non-positive cell: zero-out.
- Positive cell, no extremum in [0,1]: pass through.
- Positive cell, extremum in [0,1], fmin >= 0: pass through.
- Positive cell, extremum in [0,1], fmin < 0, both bl & br > 0: zero.
- Positive cell, extremum in [0,1], fmin < 0, da1 > 0: clip br.
- Positive cell, extremum in [0,1], fmin < 0, da1 < 0: clip bl.

A regression that breaks any of these branches will fail this
test.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.core.fv_tp_2d import _pert_ppm_iv0


def _pert_ppm_iv0_fortran_reference(q, bl, br):
    """Element-wise NumPy port of Fortran tp_core.F90:1169-1192."""
    r12 = 1.0 / 12.0
    bl_out = np.array(bl, dtype=np.float64, copy=True)
    br_out = np.array(br, dtype=np.float64, copy=True)
    q_arr = np.asarray(q, dtype=np.float64)
    for i in range(q_arr.size):
        if q_arr.flat[i] <= 0.0:
            bl_out.flat[i] = 0.0
            br_out.flat[i] = 0.0
        else:
            a4 = -3.0 * (br_out.flat[i] + bl_out.flat[i])
            da1 = br_out.flat[i] - bl_out.flat[i]
            if abs(da1) < -a4:
                fmin = (q_arr.flat[i] + 0.25 / a4 * da1**2 + a4 * r12)
                if fmin < 0.0:
                    if br_out.flat[i] > 0.0 and bl_out.flat[i] > 0.0:
                        br_out.flat[i] = 0.0
                        bl_out.flat[i] = 0.0
                    elif da1 > 0.0:
                        br_out.flat[i] = -2.0 * bl_out.flat[i]
                    else:
                        bl_out.flat[i] = -2.0 * br_out.flat[i]
    return bl_out, br_out


def _make_test_inputs():
    """Build inputs that exercise every Fortran iv=0 branch.

    Iter-881b (Codex iter-881 stop-time fix): the original hand-
    built inputs claimed to cover all 6 branches but actually
    landed branches 4-6 in branch 2 (no-extremum) because the
    `abs(da1) < -a4` gate requires `a4 < 0` (i.e., `bl+br > 0`),
    which the original inputs didn't satisfy.

    The Fortran iv=0 algorithm:
      a4 = -3*(ar + al)         # parabola "a4" coefficient
      da1 = ar - al              # slope
      has_extremum = abs(da1) < -a4    # only true when a4 < 0
                                       # (i.e., bl + br > 0)
      if has_extremum:
          fmin = q + 0.25/a4*da1**2 + a4/12
          if fmin < 0:
              if br>0 and bl>0:  zero
              elif da1 > 0:       clip br = -2*bl
              else:               clip bl = -2*br

    For branches 4, 5, 6 we need `bl + br > 0` (so a4 < 0) AND
    fmin < 0.  For branch 4 specifically we need bl > 0 AND br > 0.
    For branches 5/6 we need bl ≤ 0 OR br ≤ 0, plus the da1 sign.

    Trace verified by hand for the corrected inputs below.
    """
    # Branch  | q       | bl    | br    | Trace
    # --------|---------|-------|-------|--------------------------
    # 1a      | -1.0    | 0.5   | -0.5  | a0 ≤ 0 → both → 0
    # 1b      |  0.0    | 0.5   | -0.5  | a0 ≤ 0 → both → 0
    # 2       |  1.0    | 0.1   | -0.1  | a4=0, abs(0.2)<0 false → pass
    # 3       |  0.6    | 0.5   |  1.0  | a4=-4.5, |0.5|<4.5 ✓,
    #         |         |       |       | fmin=0.6 - 0.0139 - 0.375
    #         |         |       |       |      ≈ 0.211 ≥ 0 → pass
    # 4       |  0.5    | 2.0   |  2.0  | a4=-12, |0|<12 ✓, fmin=-0.5<0,
    #         |         |       |       | both > 0 → zero
    # 5       |  0.5    | -0.5  |  2.0  | a4=-4.5, |2.5|<4.5 ✓, fmin≈-0.22<0,
    #         |         |       |       | NOT both > 0 (bl<0), da1=2.5>0
    #         |         |       |       | → br = -2*bl = 1.0
    # 6       |  0.5    | 2.0   | -0.5  | a4=-4.5, |-2.5|<4.5 ✓, fmin≈-0.22<0,
    #         |         |       |       | NOT both > 0 (br<0), da1=-2.5<0
    #         |         |       |       | → bl = -2*br = 1.0
    #
    # Iter-881b crucial detail: branches 5 and 6 use NON-ZERO bl/br
    # so the clip output (e.g., `-2*bl`) is non-zero and a corruption
    # of the multiplier (e.g., `-2` → `-3`) produces a measurable
    # diff.  The pre-iter-881b inputs used bl=0 / br=0 which made
    # `-2*0 == -3*0 == 0`, masking such corruptions (false-pass).
    #
    # Iter-881c (Codex iter-881b stop-time fix): branch 3 now uses
    # NON-ZERO da1 (bl=0.5, br=1.0, da1=0.5) so the fmin formula's
    # `0.25/a4*da1**2` term is genuinely exercised.  Pre-iter-881c
    # branch 3 used bl=br=0.4 with da1=0, making the fmin formula's
    # coefficient on `da1**2` irrelevant — a corruption like
    # `0.25 → 0.50` would have produced no observable diff (zero
    # contribution either way).  iter-881c's da1=0.5 makes the
    # coefficient corruption produce a measurable fmin shift that
    # could flip pass-through to clip.
    q  = np.array([-1.0, 0.0, 1.0, 0.6, 0.5,  0.5,  0.5])
    bl = np.array([ 0.5, 0.5, 0.1, 0.5, 2.0, -0.5,  2.0])
    br = np.array([-0.5, -0.5, -0.1, 1.0, 2.0, 2.0, -0.5])
    return q, bl, br


def _expected_outputs_per_branch():
    """Hand-computed expected (bl_out, br_out) for the 7 inputs
    above, derived from the Fortran iv=0 logic.  Used to verify
    BOTH that the implementation matches Fortran AND that each
    input actually fires the intended branch (i.e., the output
    structurally matches the branch's expected effect)."""
    # Branch 1a/b: zero both.
    # Branch 2: pass-through (bl, br unchanged).
    # Branch 3: pass-through (iter-881c: bl=0.5 br=1.0).
    # Branch 4: zero both.
    # Branch 5: br = -2*bl = -2*(-0.5) = 1.0 (bl unchanged).
    # Branch 6: bl = -2*br = -2*(-0.5) = 1.0 (br unchanged).
    bl_expected = np.array([0.0, 0.0, 0.1, 0.5, 0.0, -0.5,  1.0])
    br_expected = np.array([0.0, 0.0, -0.1, 1.0, 0.0,  1.0, -0.5])
    return bl_expected, br_expected


def test_iter881_pert_ppm_iv0_matches_fortran_reference():
    """`_pert_ppm_iv0` MUST match the Fortran `pert_ppm(iv=0)`
    reference on inputs that exercise each Fortran branch.

    A regression that breaks any branch (e.g., wrong fmin formula,
    wrong da1 sign convention, wrong "both positive" check) will
    fail this test.

    Iter-881b (Codex iter-881 stop-time fix): also verify against
    hand-computed branch-specific expected outputs.  The Fortran-
    reference comparison alone could pass if BOTH the production
    function AND the reference contained the same bug; pinning
    the per-branch expected outputs ensures the inputs actually
    exercise the intended branches.
    """
    q, bl, br = _make_test_inputs()

    bl_actual, br_actual = _pert_ppm_iv0(
        jnp.asarray(q), jnp.asarray(bl), jnp.asarray(br))
    bl_ref, br_ref = _pert_ppm_iv0_fortran_reference(q, bl, br)
    bl_expected, br_expected = _expected_outputs_per_branch()

    # First check: production matches Fortran reference.
    np.testing.assert_allclose(
        np.asarray(bl_actual), bl_ref, rtol=1e-12, atol=1e-12,
        err_msg=(
            "`_pert_ppm_iv0` bl output differs from Fortran reference. "
            "Audit the iv=0 positive-definite constraint logic in "
            "`fv_tp_2d.py` against `tp_core.F90:1169-1192`."))
    np.testing.assert_allclose(
        np.asarray(br_actual), br_ref, rtol=1e-12, atol=1e-12,
        err_msg=(
            "`_pert_ppm_iv0` br output differs from Fortran reference."))

    # Second check (iter-881b): production matches hand-computed
    # per-branch expected outputs.  This pins that each test input
    # actually exercises the claimed branch, not just that the two
    # implementations agree.
    np.testing.assert_allclose(
        np.asarray(bl_actual), bl_expected, rtol=1e-12, atol=1e-12,
        err_msg=(
            "iter-881b branch-coverage failure: `_pert_ppm_iv0` bl "
            "output differs from hand-computed per-branch expected. "
            "This means either (a) the Python implementation is wrong "
            "in a way that AGREES with the reference (both have same "
            "bug) — check both against Fortran tp_core.F90:1169-1192 "
            "directly, OR (b) one of the test inputs no longer "
            "exercises the intended branch (e.g., a numerical "
            "precision change shifted the gate)."))
    np.testing.assert_allclose(
        np.asarray(br_actual), br_expected, rtol=1e-12, atol=1e-12,
        err_msg=(
            "iter-881b branch-coverage failure: br output differs "
            "from per-branch expected."))


@pytest.mark.parametrize("seed", [3, 17, 42])
def test_iter881_pert_ppm_iv0_random_inputs(seed):
    """Random-input cross-check: `_pert_ppm_iv0` matches the Fortran
    NumPy reference on random (q, bl, br) tuples.  Catches edge
    cases not exercised by the hand-built input above.
    """
    rng = np.random.default_rng(seed)
    n = 64
    # Mix positive and negative q to exercise both the zero-out
    # branch and the constraint-active branches.
    q = rng.normal(size=n) * 2.0  # roughly symmetric around 0
    bl = rng.normal(size=n) * 0.5
    br = rng.normal(size=n) * 0.5

    bl_actual, br_actual = _pert_ppm_iv0(
        jnp.asarray(q), jnp.asarray(bl), jnp.asarray(br))
    bl_ref, br_ref = _pert_ppm_iv0_fortran_reference(q, bl, br)

    np.testing.assert_allclose(
        np.asarray(bl_actual), bl_ref, rtol=1e-12, atol=1e-12,
        err_msg=f"_pert_ppm_iv0 bl mismatch on seed={seed}")
    np.testing.assert_allclose(
        np.asarray(br_actual), br_ref, rtol=1e-12, atol=1e-12,
        err_msg=f"_pert_ppm_iv0 br mismatch on seed={seed}")
