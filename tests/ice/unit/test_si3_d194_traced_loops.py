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
# O12 composition with the NEMO DH->DA flux carry, measured under jax.jit and
# jax.disable_jit -- both runs gave this digest.
EXPECTED_DIGEST = "96e36c79fb959b1563fcf3bd712752bdb586cb1324cb433ff3020eb320c5b0f0"
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
    with jax.disable_jit():  # concrete inputs: the eager Python-loop path, op by op
        eager = _digest(run(arrays))
    compiled = _digest(jax.jit(run)(arrays))  # traced inputs: the scan/fori_loop path
    assert (eager, compiled) == (EXPECTED_DIGEST, EXPECTED_DIGEST)


def test_o12_da_receives_dh_updated_hfx_thd(monkeypatch):
    """DH's heat-flux update reaches DA instead of the stale category-entry value."""
    arrays, static = _fixture()
    real_da = pnd_mod.si3_ice_thd_da
    da_calls = []

    def record_da(**kwargs):
        result = real_da(**kwargs)
        da_calls.append((kwargs, result))
        return result

    monkeypatch.setattr(pnd_mod, "si3_ice_thd_da", record_da)
    with jax.disable_jit():
        out = pnd_mod.si3_ice_thd_o12_stages(**arrays, **static)

    dh_hfx = out.categories[0].o10.o9.dh.fluxes.hfx_thd
    stale_hfx = arrays["hfx_thd"]
    assert np.any(np.asarray(dh_hfx).view(np.uint64) != np.asarray(stale_hfx).view(np.uint64))
    np.testing.assert_array_equal(
        np.asarray(da_calls[0][0]["hfx_thd"]).view(np.uint64),
        np.asarray(dh_hfx).view(np.uint64),
    )
    stale_da = real_da(**{**da_calls[0][0], "hfx_thd": stale_hfx})
    assert np.any(
        np.asarray(da_calls[0][1].hfx_thd).view(np.uint64)
        != np.asarray(stale_da.hfx_thd).view(np.uint64)
    )


# (category or None, leaf, index, relative step): thickness, concentration,
# enthalpy, snow, pond and forcing elements away from the max/min/where
# thresholds (central differences at relative steps 1e-4 and 1e-5 agree).
# Each case is judged by its own relative error |AD - FD| / max(|AD|, |FD|)
# (no absolute floor).  The difference quotient's rounding noise,
# eps |f| / (2 h) with |f| ~ 1.9e4, is 2.1e-9, 5.1e-9, 1.9e-10, 2.3e-9, 1.2e-7
# and 4.6e-9 of |AD| for the six cases, so a gradient wrong by 1e-6 or more is
# visible in each.  Measured: 2.6e-9, 2.4e-8, 2.6e-10, 7.1e-9, 9.9e-9, 9.1e-9.
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

    failures = []
    for jl, leaf, index, rel in FD_CASES:
        x0 = float((arrays[leaf] if jl is None else arrays["category_entries"][jl][leaf])[index])
        g = float((grad[leaf] if jl is None else grad["category_entries"][jl][leaf])[index])
        h = rel * abs(x0)
        fd = (float(forward(perturbed(jl, leaf, index, x0 + h)))
              - float(forward(perturbed(jl, leaf, index, x0 - h)))) / (2.0 * h)
        relative = abs(g - fd) / max(abs(g), abs(fd))
        if g == 0.0 or not relative <= 1.0e-6:
            failures.append((jl, leaf, index, g, fd, relative))
    assert failures == []


def test_ssl_snow_transmission_is_not_contracted_under_jit():
    """The snow SSL absorption difference must match the eager program bit for bit.

    Without the barrier on the transmitted radiation, XLA CPU fuses that
    product into the absorption difference as an FMA (7754 of 20000 elements
    differed on this input when measured).
    """
    rng = np.random.default_rng(7)
    n = 4000
    args = (jnp.asarray(rng.uniform(0.2, 3.0, n)), jnp.asarray(rng.uniform(0.05, 0.6, n)),
            jnp.asarray(rng.uniform(1.0, 100.0, n)), jnp.full(n, 260.0))
    run = lambda *a: bl._radiation(*a, rn_kappa_i=1.0, rn_kappa_s=10.0, _nlay=(10, 5))  # noqa: E731
    compiled = jax.jit(run)(*args)
    with jax.disable_jit():
        eager = run(*args)
    for got, want in zip(compiled, eager, strict=True):
        np.testing.assert_array_equal(np.asarray(got).view(np.uint64), np.asarray(want).view(np.uint64))
