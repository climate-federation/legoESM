"""DECISION 194: traced O9--O12 loops (category scan, remap, permeability) keep bits and differentiate.

The five-category O12 composition (live flux carry, as O17 calls it) is
compiled with ``jax.jit``.  Its forward bits are pinned to the Python-loop
program at parent commit 9998276020, its reverse pass must compile and stay
finite, and the AD gradient must match central finite differences of the same
compiled forward on smooth elements (thickness, concentration, enthalpy,
snow, pond and a forcing field).
"""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice import si3_thd_pnd as pnd_mod
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
from tests.ice.unit import test_si3_o9_zdf_dh_temp as o9_fixture
from tests.ice.unit import test_si3_o10_salinity as o10_fixture
from tests.ice.unit.test_si3_o12_pnd import (
    DECK_PND_NML,
    GRAVITY,
    R1_NLAY_I,
    R1_NLAY_S,
    R1_RDT,
    RDT,
    RHO_I,
    RHO_S,
    RHO_W,
)

# sha256 over (path, dtype, shape, bytes) of every output leaf of ``_fixture``'s
# O12 composition, measured at 9998276020 (Python loops) under jax.jit and under
# jax.disable_jit -- both runs gave this digest.
PARENT_DIGEST = "e74fcfa0c610d6c33b80babc5639b4471ab70c4c70eba86b520e226cba8373bf"
JPJ = JPI = 6


@pytest.fixture(autouse=True)
def _libm_cpu():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    with jax.default_device(jax.devices("cpu")[0]):
        yield
    set_policy(previous)


def _fixture():
    namelist = {**o9_fixture._deck(), **o10_fixture.NML, **DECK_PND_NML,
                "ln_icedA": True, "rn_beta": 0.5, "rn_dmin": 8.0}
    categories = o9_fixture._categories(seed=19)
    rng = np.random.default_rng(20261205)
    entries, flux0 = [], None
    for jl in range(5):
        cells = [categories[jl][i % len(categories[jl])] for i in range(JPJ * JPI)]
        e1, f1 = o9_fixture._entry(cells)
        entry = {k: v.reshape((JPJ, JPI) + v.shape[1:]) for k, v in e1.items()}
        entry.update(at_i=jnp.full((JPJ, JPI), 0.75), sst=jnp.full((JPJ, JPI), 0.2),
                     o_i=jnp.full((JPJ, JPI), 10.0), hfx_err_dif=jnp.zeros((JPJ, JPI)),
                     hfx_dif=jnp.zeros((JPJ, JPI)))
        if flux0 is None:
            flux0 = bl.SI3DHFluxes(**{f: getattr(f1, f).reshape(JPJ, JPI) for f in f1._fields})
        entries.append(entry)
    a_ip = jnp.asarray(rng.uniform(0.02, 0.08, (JPJ, JPI, 5)))
    h_ip = jnp.asarray(rng.uniform(0.05, 0.14, (JPJ, JPI, 5)))
    h_il = jnp.asarray(rng.uniform(0.0, 0.02, (JPJ, JPI, 5)))
    a_i = np.stack([np.asarray(e["a_i"]) for e in entries], axis=-1)
    zero = jnp.zeros((JPJ, JPI))
    arrays = dict(
        category_entries=tuple(entries), category_fluxes=flux0, sfx_bri=zero, sfx_lam=zero,
        hfx_thd=zero, wfx_lam=zero, a_ip=a_ip, v_ip=h_ip * a_ip, h_ip=h_ip, v_il=h_il * a_ip,
        h_il=h_il, a_ip_frac=jnp.asarray(np.asarray(a_ip) / np.maximum(a_i, 1.0e-10)), wfx_pnd=zero,
    )
    static = dict(namelist=namelist, rDt_ice=RDT, r1_Dt_ice=R1_RDT, r1_nlay_i=R1_NLAY_I,
                  r1_nlay_s=R1_NLAY_S, rho_i=RHO_I, rho_s=RHO_S, rho_w=RHO_W, gravity=GRAVITY,
                  nn_hls=2, constants=NEMO_SI3_CONSTANTS_CONFIG)
    return arrays, static


def _digest(tree) -> str:
    sha = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        leaf = np.ascontiguousarray(np.asarray(leaf))
        sha.update(f"{jax.tree_util.keystr(path)}|{leaf.dtype}|{leaf.shape}|".encode())
        sha.update(leaf.tobytes())
    return sha.hexdigest()


def test_traced_o12_loops_keep_parent_forward_bits():
    arrays, static = _fixture()
    run = lambda a: pnd_mod.si3_ice_thd_o12_stages(**a, **static)  # noqa: E731
    jaxpr = jax.make_jaxpr(run)(arrays)
    assert "scan" in {eqn.primitive.name for eqn in jaxpr.jaxpr.eqns}
    assert _digest(jax.jit(run)(arrays)) == PARENT_DIGEST


# (category or None, leaf, index, relative step): thickness, concentration,
# enthalpy, snow, pond and forcing elements away from the max/min/where
# thresholds (central differences at relative steps 1e-4 and 1e-5 agree).
# The step is large enough that the objective's rounding (|f| ~ 1.9e4) stays
# far below the difference; measured |AD - FD| / max(|AD|, |FD|):
# 2.6e-9, 2.4e-8, 2.6e-10, 7.1e-9, 9.9e-9, 9.1e-9.
FD_CASES = (
    (2, "h_i", (3, 2), 1.0e-4),
    (0, "a_i", (3, 2), 1.0e-4),
    (2, "e_i", (3, 2, 3), 1.0e-4),
    (1, "h_s", (2, 3), 1.0e-4),
    (None, "a_ip", (2, 3, 0), 1.0e-4),
    (0, "qns_ice", (3, 2), 1.0e-4),
)


def test_traced_o12_loops_reverse_mode_compiles_and_matches_central_differences():
    arrays, static = _fixture()

    def outputs(a):
        out = pnd_mod.si3_ice_thd_o12_stages(**a, **static)
        return [x for x in jax.tree_util.tree_leaves(out) if jnp.issubdtype(x.dtype, jnp.floating)]

    weights = [1.0 / (1.0 + float(np.max(np.abs(x)))) for x in jax.jit(outputs)(arrays)]

    def objective(a):
        return sum(jnp.sum(x) * w for x, w in zip(outputs(a), weights))

    value_and_grad = jax.jit(jax.value_and_grad(objective)).lower(arrays).compile()
    value, grad = value_and_grad(arrays)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree_util.tree_leaves(grad))
    forward = jax.jit(objective)

    def perturbed(jl, leaf, index, x):
        a = dict(arrays)
        if jl is None:
            a[leaf] = a[leaf].at[index].set(x)
        else:
            entries = list(a["category_entries"])
            entries[jl] = {**entries[jl], leaf: entries[jl][leaf].at[index].set(x)}
            a["category_entries"] = tuple(entries)
        return a

    for jl, leaf, index, rel in FD_CASES:
        x0 = float((arrays[leaf] if jl is None else arrays["category_entries"][jl][leaf])[index])
        g = float((grad[leaf] if jl is None else grad["category_entries"][jl][leaf])[index])
        h = rel * abs(x0)
        fd = (float(forward(perturbed(jl, leaf, index, x0 + h)))
              - float(forward(perturbed(jl, leaf, index, x0 - h)))) / (2.0 * h)
        assert g != 0.0, (jl, leaf, index)
        assert abs(g - fd) <= 1.0e-6 * max(abs(g), abs(fd)), (jl, leaf, index, g, fd)
