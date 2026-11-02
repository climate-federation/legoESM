"""DECISION 198: the traced O6 redistribution donor loop keeps bits and differentiates.

Under a JAX transform the ``rdgrft_shift`` donor-category loop (``DO jl1``,
icedyn_rdgrft.f90:722-939) runs as one static-trip ``fori_loop``; eager calls
keep the Python loop.  On the O6 unit-test fixture both paths are pinned to the
parent program's digest (b17f14b73c), and the compiled reverse pass must match
central finite differences of the same compiled forward on smooth elements.
"""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tests.ice.unit.test_si3_o6_rdgrft_cor as o6
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice.ridging import SI3RdgrftState, si3_ice_dyn_rdgrft

# sha256 over (path, dtype, shape, bytes) of every SI3RdgrftResult leaf for
# o6._inputs(), measured at b17f14b73c under jax.jit and jax.disable_jit
# (both gave this digest).
PARENT_DIGEST = "961587bdc9948e28eadad1b8ea7e226cca630b88a70b61efc5ec234ab51de01e"


@pytest.fixture(autouse=True)
def _libm_cpu():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    with jax.default_device(jax.devices("cpu")[0]):
        yield
    set_policy(previous)


def _fixture():
    s, divu, delta, sst, sss = o6._inputs()
    return ({k: jnp.asarray(s[k]) for k in SI3RdgrftState._fields},
            jnp.asarray(divu), jnp.asarray(delta), jnp.asarray(sst), jnp.asarray(sss))


def _run(x):
    st, divu, delta, sst, sss = x
    return si3_ice_dyn_rdgrft(SI3RdgrftState(**st), divu_i=divu, delta_i=delta, sst_m=sst, sss_m=sss,
                              hi_max=o6.HI_MAX, nn_hls=o6.NN_HLS, config=o6.RDG_CFG)


def _digest(tree) -> str:
    sha = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        leaf = np.ascontiguousarray(np.asarray(leaf))
        sha.update(f"{jax.tree_util.keystr(path)}|{leaf.dtype}|{leaf.shape}|".encode())
        sha.update(leaf.tobytes())
    return sha.hexdigest()


def test_traced_rdgrft_keeps_parent_forward_bits():
    x = _fixture()
    with jax.disable_jit():  # concrete inputs: the eager Python donor loop, op by op
        eager = _run(x)
    compiled = jax.jit(_run)(x)  # traced inputs: the donor fori_loop
    assert int(eager.n_passes) >= 2 and int(np.asarray(eager.selected).sum()) == 19
    assert (_digest(eager), _digest(compiled)) == (PARENT_DIGEST, PARENT_DIGEST)


# (state leaf, index, relative step): area, thickness, snow, enthalpy, pond and
# heat-content elements of ridging cells; central differences at relative
# steps 1e-4 and 1e-5 agree to 1e-6.  Each case is judged by its own relative
# error |AD - FD| / max(|AD|, |FD|).  The quotient's rounding noise,
# eps |f| / (2 h) with |f| ~ 9.2e2, is at most 9.1e-8 of |AD| (the pond case).
# Measured: 1.4e-9, 7.5e-10, 1.3e-10, 2.1e-10, 1.7e-8, 1.3e-10.
FD_CASES = (
    ("a_i", (6, 4, 4), 1.0e-4),
    ("v_i", (6, 4, 4), 1.0e-4),
    ("v_s", (6, 4, 4), 1.0e-4),
    ("e_i", (5, 3, 0, 2), 1.0e-4),
    ("v_ip", (6, 4, 4), 1.0e-4),
    ("oa_i", (5, 3, 2), 1.0e-4),
)


def test_traced_rdgrft_reverse_mode_matches_central_differences():
    x = _fixture()
    weights = [1.0 / (1.0 + float(np.max(np.abs(np.asarray(o))))) for o in jax.jit(lambda y: list(_run(y).state))(x)]

    def objective(y):
        return sum(jnp.sum(o) * w for o, w in zip(_run(y).state, weights))

    value, grad = jax.jit(jax.value_and_grad(objective)).lower(x).compile()(x)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree_util.tree_leaves(grad))
    forward = jax.jit(objective)

    def perturbed(name, index, v):
        st = dict(x[0])
        st[name] = st[name].at[index].set(v)
        return (st,) + x[1:]

    failures = []
    for name, index, rel in FD_CASES:
        x0 = float(x[0][name][index])
        g = float(grad[0][name][index])
        h = rel * abs(x0)
        fd = (float(forward(perturbed(name, index, x0 + h)))
              - float(forward(perturbed(name, index, x0 - h)))) / (2.0 * h)
        relative = abs(g - fd) / max(abs(g), abs(fd))
        if g == 0.0 or not relative <= 1.0e-6:
            failures.append((name, index, g, fd, relative))
    assert failures == []
