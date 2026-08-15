"""The shared gate helpers must be able to FAIL.

Every guard in ``fv3_gate_helpers`` exists because a gate somewhere in
this campaign passed while proving nothing.  A helper whose guards were
silently broken would re-open all of them at once, and no parity test
would notice -- they would simply go green.  So each guard gets a case
that must raise and a case that must pass.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tests.grids import fv3_gate_helpers as helpers  # noqa: E402


def _f(x):
    return np.asarray(x, dtype=np.float64)


# ------------------------------------------------------------ cmp_fields

def test_cmp_passes_on_equal_finite_fields():
    a = _f([[1.0, 2.0], [3.0, 4.0]])
    rel, n_over = helpers.cmp_fields(a, a.copy(), "equal", 1e-15)
    assert rel == 0.0 and n_over == 0


def test_cmp_reports_a_real_difference():
    a = _f([1.0, 2.0, 3.0])
    b = a.copy()
    b[1] += 1.0e-6
    rel, _ = helpers.cmp_fields(a, b, "close", 1e-3)
    assert 1e-8 < rel < 1e-3


def test_cmp_refuses_a_nan_mask_mismatch():
    a = _f([1.0, np.nan, 3.0])
    b = _f([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError, match="non-finite masks differ"):
        helpers.cmp_fields(a, b, "maskslip", 1e-9)


def test_cmp_refuses_a_drifted_fill():
    """1e30 passes every isfinite guard, so a moved fill is a defect
    wearing a finite disguise."""
    a = _f([1.0, 1.0e30, 3.0])
    b = _f([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError, match="workspace-FILL masks differ"):
        helpers.cmp_fields(a, b, "filldrift", 1e-9)


def test_cmp_refuses_an_all_fill_pair():
    a = _f([1.0e30, 1.0e30])
    with pytest.raises(AssertionError, match="pass vacuously"):
        helpers.cmp_fields(a, a.copy(), "allfill", 1e-9)


def test_cmp_refuses_an_identically_zero_reference():
    """The gate that cannot fail: zero vs zero at any tolerance."""
    z = np.zeros(4)
    with pytest.raises(AssertionError, match="CANNOT FAIL"):
        helpers.cmp_fields(z, z.copy(), "zeroref", 1e-9)


def test_cmp_refuses_a_shape_mismatch():
    with pytest.raises(AssertionError):
        helpers.cmp_fields(np.zeros((2, 3)), np.zeros((3, 2)), "shape", 1e-9)


def test_cmp_scale_floor_stops_one_huge_cell_hiding_the_rest():
    """Without the median floor, a 1e12 cell divides every physical
    discrepancy in the field down to nothing."""
    b = _f([1.0, 1.0, 1.0, 1.0e12])
    a = b.copy()
    a[0] += 0.5                     # a 50 % error on a unit cell
    rel, _ = helpers.cmp_fields(a, b, "hugecell", 1.0)
    assert rel > 0.1, rel


# -------------------------------------------------------- bitwise_equal

def test_bitwise_equal_treats_nan_as_equal_and_catches_one_ulp():
    a = _f([1.0, np.nan])
    assert helpers.bitwise_equal(a, a.copy())
    b = a.copy()
    b[0] = np.nextafter(1.0, 2.0)
    assert not helpers.bitwise_equal(a, b)


# ----------------------------------------------------------- assert_real

def test_assert_real_rejects_all_nan_and_all_zero():
    with pytest.raises(AssertionError, match="entirely non-finite"):
        helpers.assert_real(_f([np.nan, np.nan]), "nan")
    with pytest.raises(AssertionError, match="exactly 0.0"):
        helpers.assert_real(np.zeros(3), "zero")
    helpers.assert_real(_f([0.0, 1.0]), "mixed")


# ------------------------------------------------------------- adjoint

def test_adjoint_identity_holds_for_a_linear_map():
    m = jnp.asarray(np.random.default_rng(0).standard_normal((5, 5)))

    def f(x):
        return m @ x

    x = jnp.asarray(np.random.default_rng(1).standard_normal(5))
    r = helpers.check_adjoint("linear", f, (x,), 1e-12)
    assert r < 1e-12


def test_adjoint_identity_holds_for_a_nonlinear_map():
    def f(x):
        return jnp.tanh(x) * jnp.sum(x ** 2)

    x = jnp.asarray(np.random.default_rng(2).standard_normal(6))
    assert helpers.check_adjoint("nonlinear", f, (x,), 1e-12) < 1e-12


def test_adjoint_gate_refuses_a_trivially_satisfied_identity():
    """A map whose Jacobian is zero satisfies <Jv,w> == <v,J^T w> as
    0 == 0.  That must not read as a pass."""
    def f(x):
        return jnp.zeros_like(x) + jax.lax.stop_gradient(x)

    x = jnp.asarray(np.ones(4))
    with pytest.raises(AssertionError, match="satisfied trivially"):
        helpers.check_adjoint("dead", f, (x,), 1e-12)


def test_adjoint_gate_names_a_nan_in_the_tangent():
    def f(x):
        return jnp.sqrt(x)          # NaN tangent at a negative primal

    x = jnp.asarray([-1.0, 1.0])
    with pytest.raises(AssertionError, match="not finite"):
        helpers.check_adjoint("nanjac", f, (x,), 1e-12)


# ------------------------------------------------------------- counted

def test_counted_preserves_the_signature_jit_needs():
    def g(x, km):
        return x * km

    box, wrapped = helpers.counted(g)
    import inspect
    assert list(inspect.signature(wrapped).parameters) == ["x", "km"]
    jitted = jax.jit(wrapped, static_argnames=("km",))
    jitted(jnp.ones(3), km=2)
    jitted(jnp.zeros(3), km=2)
    assert box["n"] == 1, "a second trace on the same shapes/statics"


# --------------------------------------------------- container helpers

def test_deepcopy_faces_really_decouples():
    src = [{"a": np.ones(3)} for _ in range(6)]
    cp = helpers.deepcopy_faces(src)
    cp[0]["a"][0] = 99.0
    assert src[0]["a"][0] == 1.0


def test_stack_np_puts_the_face_axis_first():
    src = [{"u": np.full((2, 3), float(t))} for t in range(6)]
    out = helpers.stack_np(src)
    assert out["u"].shape == (6, 2, 3)
    assert float(out["u"][4, 0, 0]) == 4.0
