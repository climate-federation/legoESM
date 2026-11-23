"""T28 vertical-split budgets; these do not certify full-model transport."""

import ast
import inspect
from pathlib import Path
import runpy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean import vertical as ov
from legoesm.ocean.dynamics import ocean_pe_cdgrid as cd
from legoesm.ocean.dynamics import spectral_ocean_pe as so

_census = runpy.run_path(str(Path(__file__).resolve().parents[3] /
                            "scripts/validate/diag_hybrid_tracer_smoothing.py"))
paired_column_budget = _census["paired_column_budget"]


@pytest.fixture(autouse=True)
def fp64_policy():
    old = get_policy()
    assert jax.config.x64_enabled, "run JAX_ENABLE_X64=1"
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(old)


@pytest.fixture(params=["shared", "CD-grid", "spectral"])
def operator(request):
    return {"shared": ov.vertical_advection_ocean,
            "CD-grid": cd._vertical_advection_ocean,
            "spectral": so.vertical_advection_ocean}[request.param]


@pytest.mark.parametrize("kind", ["zstar", "partial", "single"])
@pytest.mark.parametrize("tracer", ["T", "S"])
def test_paired_column_conservation(operator, kind, tracer):
    zc, J, h, T, S, w = _census["realistic_ocean_columns"]()
    q = T if tracer == "T" else S
    if kind == "single":
        # Degenerate shape/zero-flux edge case, not nonzero budget evidence.
        zc = ov.create_z_star_from_thicknesses([5.0], allow_single_level=True)
        q, w = q[:, :1], jnp.zeros((3, 2))
        h = zc.dz_ref * J[:, None]
    elif kind == "partial":
        zc = ov.create_partial_cell_coordinate(zc, jnp.asarray([120.0, 800.0, zc.H_max]))
        h = zc.h_partial * J[:, None]
        active = zc.is_active
        # Independent continuity: explicitly close faces at each local seafloor.
        faces = jnp.pad(active[:, :-1] & active[:, 1:], ((0, 0), (1, 1)))
        w = w * faces
        q = jnp.where(active, q, 1e6)
    dq = operator(q, w, zc, J)
    r = paired_column_budget(h, q, dq, jnp.diff(w))
    assert r["max_relative"] <= 1e-12, r
    if kind != "single":
        assert np.all(np.asarray(r["denominator"]) > 0)
        leak = dq.at[:, 0].add(1e-3)
        assert paired_column_budget(h, q, leak, jnp.diff(w))["max_relative"] > 1e-12
    if kind == "partial":
        np.testing.assert_array_equal(np.asarray(dq)[~np.asarray(active)], 0.0)
        # Noise at rock interfaces must not affect active or inactive cells.
        noisy_w = w + jnp.where(faces, 0.0, 1e-10)
        np.testing.assert_array_equal(operator(q, noisy_w, zc, J), dq)


def test_constant_tracer_and_jit(operator):
    zc, J, h, T, S, w = _census["realistic_ocean_columns"]()
    q = jnp.full_like(T, 35.0)
    dq = operator(q, w, zc, J)
    np.testing.assert_allclose(dq, 0.0, rtol=0.0, atol=1e-12)
    compiled = jax.jit(lambda q, w, J: operator(q, w, zc, J))
    np.testing.assert_allclose(compiled(T, w, J), operator(T, w, zc, J), rtol=1e-12, atol=1e-12)
    assert jax.eval_shape(compiled, T, w, J).shape == T.shape


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_upwind_sign(operator, sign):
    zc = ov.create_z_star_from_thicknesses([5.0, 10.0])
    q = jnp.asarray([[2.0, 5.0]])
    w = jnp.asarray([[0.0, sign*0.01, 0.0]])
    expected = [[0.006, 0.0]] if sign > 0 else [[0.0, -0.003]]
    np.testing.assert_allclose(operator(q, w, zc, jnp.ones(1)), expected, rtol=1e-12, atol=1e-12)


def test_fp32_gradients(operator):
    set_policy(PrecisionPolicy.fp32())
    zc, J, h, T, S, w = _census["realistic_ocean_columns"]()
    q, w, J = (x.astype(jnp.float32) for x in (T, w, J))
    assert operator(q, w, zc, J).dtype == jnp.float32
    loss = lambda q, w, J: jnp.sum(operator(q, w, zc, J)**2)
    gradients = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, w, J)
    for grad in gradients:
        assert grad.dtype == jnp.float32
        assert np.isfinite(grad).all()
        assert np.any(np.asarray(grad) != 0.0)


def test_census_known_budgets():
    q = np.asarray([[2.0, 3.0]])
    good = paired_column_budget(1.0, q, [[1.0, -1.0]], 0.0)
    bad = paired_column_budget(1.0, q, [[1.0, 0.0]], 0.0)
    assert good["max_relative"] == 0.0 and good["verdict"] == "CONSERVES"
    assert bad["max_relative"] == 1.0 and bad["verdict"] == "LEAKS"
    assert paired_column_budget(1.0, q, [[0.0, 0.0]], 0.0)["verdict"] == "CONSERVES"
    with pytest.raises(ValueError, match="nonfinite"):
        paired_column_budget(1.0, q, [[np.nan, 0.0]], 0.0)


@pytest.mark.parametrize("module,fn,momentum,tracer", [
    (cd, cd.ocean_baroclinic_tendencies_cdgrid, "advective_vertical_momentum_ocean", "_vertical_advection_ocean"),
    (so, so.spectral_ocean_tendencies, "advective_vertical_momentum_ocean", "vertical_advection_ocean"),
])
def test_production_wiring(module, fn, momentum, tracer):
    # Inspect the actual tendency function, including nested vmap bodies.
    tree = ast.parse(inspect.getsource(fn))
    calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert calls.count(momentum) == (1 if module is cd else 2)
    assert calls.count(tracer) == 1
    # Also refuse any additional caller elsewhere in either backend module.
    module_tree = ast.parse(inspect.getsource(module))
    module_calls = [n.func.id for n in ast.walk(module_tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert module_calls.count(momentum) == calls.count(momentum)
    assert module_calls.count(tracer) == calls.count(tracer)
    assert getattr(module, momentum) is ov.advective_vertical_momentum_ocean
    assert getattr(module, tracer) is ov.vertical_advection_ocean
