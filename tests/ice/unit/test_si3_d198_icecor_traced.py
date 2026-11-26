"""DECISION 198 round 3: the traced O6 ice_cor loops keep bits and differentiate.

Under a JAX transform the ``ice_itd_reb`` rebinning passes (``DO jl``,
iceitd.f90:628-690), the salinity bounds (``DO jl / DO jk``, icecor.f90:114-122)
and the ``ice_var_zapsmall`` flux sums (icevar.f90:633-722) run as static-trip
``fori_loop``s; eager calls keep the Python loops.  On the O6 unit-test ice_cor
fixture, at its own layer counts and at ORCA2's, both paths are pinned to the
parent program's digest (56b768300a), and the compiled reverse pass must match
central finite differences of the same compiled forward.
"""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tests.ice.unit.test_si3_o6_rdgrft_cor as o6
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice.itd import SI3IceCorState, si3_ice_cor_dyn

# sha256 over (layers, path, dtype, shape, bytes) of every SI3IceCorState leaf
# for o6._cor_inputs() at 3/2 and 10/5 ice/snow layers, measured at 56b768300a
# under jax.jit and jax.disable_jit (both gave this digest).
PARENT_DIGEST = "5ce61bd2a20bc34f267ea2a182667d581d2c07ea6e4fb2ed3d65603ba6b0468e"
LAYERS = ((o6.NLAY_I, o6.NLAY_S), (10, 5))


@pytest.fixture(autouse=True)
def _libm_cpu():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    with jax.default_device(jax.devices("cpu")[0]):
        yield
    set_policy(previous)


def _fixture(monkeypatch, nlay_i, nlay_s):
    monkeypatch.setattr(o6, "NLAY_I", nlay_i)
    monkeypatch.setattr(o6, "NLAY_S", nlay_s)
    c, amax, sst, sss = o6._cor_inputs()
    return {k: jnp.asarray(c[k]) for k in SI3IceCorState._fields}, (amax, sst, sss)


def _run(st, forcing):
    amax, sst, sss = forcing
    return si3_ice_cor_dyn(SI3IceCorState(**st), rn_amax_2d=amax, sss_m=sss, sst_m=sst,
                           hi_max=o6.HI_MAX, hi_mean=o6.HI_MEAN, nn_hls=o6.NN_HLS, config=o6.COR_CFG)


def _digest(results) -> str:
    sha = hashlib.sha256()
    for layers, tree in results:
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            leaf = np.ascontiguousarray(np.asarray(leaf))
            sha.update(f"{layers}|{jax.tree_util.keystr(path)}|{leaf.dtype}|{leaf.shape}|".encode())
            sha.update(leaf.tobytes())
    return sha.hexdigest()


def test_traced_ice_cor_keeps_parent_forward_bits(monkeypatch):
    eager, compiled = [], []
    for layers in LAYERS:
        st, forcing = _fixture(monkeypatch, *layers)
        with jax.disable_jit():  # concrete inputs: the Python loops, op by op
            eager.append((layers, _run(st, forcing)))
        compiled.append((layers, jax.jit(lambda y, f=forcing: _run(y, f))(st)))  # the fori_loops
    assert (_digest(eager), _digest(compiled)) == (PARENT_DIGEST, PARENT_DIGEST)


# (state leaf, index, relative step) in cells the rebinning moves ice out of
# (5,2) and (5,5); (5,2) also has every category-2 layer above the salinity
# bound.  Each case is judged by its own relative error.  The quotient is the
# objective's central difference taken element by element, so its large
# unchanged part (temperatures, enthalpies) does not swamp it.
# Measured: 7.9e-10, 6.5e-11, 3.2e-12, 1.0e-8, 9.7e-13, 4.6e-13.
FD_CASES = (
    ("v_i", (5, 2, 1), 1.0e-4),
    ("a_i", (5, 2, 1), 1.0e-4),
    ("v_i", (5, 5, 0), 1.0e-4),
    ("a_i", (5, 5, 0), 1.0e-4),
    ("szv_i", (5, 2, 1, 1), 1.0e-4),
    ("v_s", (5, 5, 0), 1.0e-4),
)


def test_traced_ice_cor_reverse_mode_matches_central_differences(monkeypatch):
    st, forcing = _fixture(monkeypatch, *LAYERS[0])
    forward = jax.jit(lambda y: list(_run(y, forcing)))
    weights = [1.0 / (1.0 + float(np.max(np.abs(np.asarray(o))))) for o in forward(st)]

    def objective(y):
        return sum(jnp.sum(o) * w for o, w in zip(_run(y, forcing), weights))

    value, grad = jax.jit(jax.value_and_grad(objective)).lower(st).compile()(st)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(g)).all() for g in grad.values())
    failures = []
    for name, index, rel in FD_CASES:
        x0 = float(st[name][index])
        g = float(grad[name][index])
        h = rel * abs(x0)
        plus = forward({**st, name: st[name].at[index].set(x0 + h)})
        minus = forward({**st, name: st[name].at[index].set(x0 - h)})
        fd = sum(float(np.sum(np.asarray(p) - np.asarray(m))) * w
                 for p, m, w in zip(plus, minus, weights)) / (2.0 * h)
        relative = abs(g - fd) / max(abs(g), abs(fd))
        if g == 0.0 or not relative <= 1.0e-6:
            failures.append((name, index, g, fd, relative))
    assert failures == []
