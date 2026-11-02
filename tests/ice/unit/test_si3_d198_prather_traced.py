"""DECISION 198 round 2: the traced O4 Prather subcycles keep bits and differentiate.

Under a JAX transform ``DO jt = 1, icycle`` (icedyn_adv_pra.f90:156-498) runs
as one three-step ``lax.scan`` whose sweep order is a traced ``lax.cond``;
eager calls keep the unrolled subcycles.  On a 12 x 10 all-ocean tile with
random ice in every cell, two-dimensional velocities and the O17 card's state
configuration (5 categories, 10 ice and 5 snow layers, LEV ponds, salinity
option 4), the delayed CFL selects two and three subcycles at both sweep
parities.  Both paths are pinned to the parent program's digest (743146b590),
and the compiled reverse pass must match central finite differences of the
same compiled forward.
"""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ice.si3_orca2_state import SI3AllPratherMoments, SI3ORCA2AdvectionGrid, si3_dyn_adv_pra_orca2
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_ad_probe import DEFAULT_MASTER, _case

# sha256 over (variant, path, dtype, shape, bytes) of every si3_dyn_adv_pra_orca2
# output leaf for both variants, measured at 743146b590 under jax.jit and under
# jax.disable_jit (both gave this digest).
PARENT_DIGEST = "309b9dbfae17bf9328d8264763bb517ad5b6b3557074a4710cf1c23fa69840c5"
VARIANTS = {"icycle2_kt2": (2, 1.0), "icycle3_kt1": (1, 2.0)}  # (kt, delayed zcfl)
JPI, JPJ = 12, 10


@pytest.fixture(autouse=True)
def _cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _fixture(variant):
    cfg = _case(DEFAULT_MASTER)[2].state  # the O17 card; also sets the libm fp64 policy
    kt, cfl = VARIANTS[variant]
    rng = np.random.default_rng(2026110201)
    s3, s4i, s4s = (JPI, JPJ, cfg.jpl), (JPI, JPJ, cfg.nlay_i, cfg.jpl), (JPI, JPJ, cfg.nlay_s, cfg.jpl)
    one = np.ones((JPI, JPJ))
    a_i = rng.uniform(0.02, 0.18, s3)
    v_i = a_i * rng.uniform(0.4, 3.0, s3)
    v_s = a_i * rng.uniform(0.02, 0.3, s3)
    a_ip = a_i * rng.uniform(0.05, 0.3, s3)
    v_ip = a_ip * rng.uniform(0.02, 0.2, s3)
    arrays = dict(
        u_ice=rng.uniform(-4.0, 4.0, (JPI, JPJ)), v_ice=rng.uniform(-4.0, 4.0, (JPI, JPJ)),
        ato_i=1.0 - a_i.sum(-1), v_i=v_i, v_s=v_s, sv_i=v_i * rng.uniform(2.0, 8.0, s3),
        oa_i=a_i * rng.uniform(1.0, 30.0, s3), a_i=a_i, a_ip=a_ip, v_ip=v_ip,
        v_il=a_ip * rng.uniform(0.0, 0.03, s3), e_s=rng.uniform(1.0e7, 5.0e7, s4s),
        e_i=rng.uniform(1.0e8, 4.0e8, s4i), szv_i=rng.uniform(0.5, 2.5, s4i),
        h_i=v_i / a_i, h_s=v_s / a_i, h_ip=v_ip / a_ip,
        wfx_res=0 * one, hfx_res=0 * one, sfx_res=0 * one,
        diag_adv_mass=0 * one, diag_adv_salt=0 * one, diag_adv_heat=0 * one,
    )
    arrays = {k: jnp.asarray(v) for k, v in arrays.items()}
    arrays["prather"] = SI3AllPratherMoments.zeros(JPI, JPJ, cfg.jpl, cfg.nlay_i, cfg.nlay_s)
    arrays["grid"] = SI3ORCA2AdvectionGrid.from_metrics(
        e1u=200000.0 * one, e2v=200000.0 * one, e2u=195000.0 * one, e1v=205000.0 * one,
        e1e2t=3.9e10 + rng.uniform(-2.0e9, 2.0e9, (JPI, JPJ)), tmask=one, umask=one, vmask=one)
    arrays["cflice_delayed"] = jnp.asarray(cfl)
    return arrays, dict(kt=kt, config=cfg, hi_max_top=99.0)


def _run(arrays, static):
    return si3_dyn_adv_pra_orca2(**arrays, **static)


def _digest(results) -> str:
    sha = hashlib.sha256()
    for variant, tree in results:
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            leaf = np.ascontiguousarray(np.asarray(leaf))
            sha.update(f"{variant}|{jax.tree_util.keystr(path)}|{leaf.dtype}|{leaf.shape}|".encode())
            sha.update(leaf.tobytes())
    return sha.hexdigest()


def test_traced_prather_keeps_parent_forward_bits():
    eager, compiled = [], []
    for variant in VARIANTS:
        arrays, static = _fixture(variant)
        with jax.disable_jit():  # concrete inputs: the unrolled subcycles, op by op
            eager.append((variant, _run(arrays, static)))
        compiled.append((variant, jax.jit(lambda a, s=static: _run(a, s))(arrays)))  # the scan
        assert int(eager[-1][1][-1]) == int(compiled[-1][1][-1]) == int(variant[6])
    assert (_digest(eager), _digest(compiled)) == (PARENT_DIGEST, PARENT_DIGEST)


# (input, index, relative step): thickness, area, snow, enthalpy, pond and both
# velocity components at interior cells.  Each case is judged by its own
# relative error; with |f| ~ 3.9e3 the quotient's rounding noise,
# eps |f| / (2 h |AD|), is at most 2.3e-9.  Measured: 9.7e-10, 3.0e-10, 6.3e-10,
# 9.8e-10, 1.9e-9, 1.5e-9, 7.5e-9.  Cells avoided: a local-maximum cell that
# only exports keeps h = v/a after the sweep, so ``Hbig_pra``'s ``h > hmax``
# test sits on its tie there and the one-sided slopes differ (v_i, a_i at
# (5,4,0)); there reverse mode returns the slope of the branch taken.
FD_CASES = (
    ("v_i", (6, 5, 2), 1.0e-4),
    ("a_i", (6, 5, 2), 1.0e-4),
    ("v_s", (4, 6, 4), 1.0e-4),
    ("e_i", (6, 5, 2, 2), 1.0e-4),
    ("v_ip", (4, 6, 4), 1.0e-4),
    ("u_ice", (4, 6), 1.0e-4),
    ("v_ice", (4, 6), 1.0e-4),
)


def test_traced_prather_reverse_mode_matches_central_differences():
    arrays, static = _fixture("icycle3_kt1")
    names = sorted({name for name, _, _ in FD_CASES})

    def outputs(x):
        out = _run({**arrays, **x}, static)
        return [o for o in jax.tree_util.tree_leaves(out) if jnp.issubdtype(o.dtype, jnp.floating)]

    x0 = {name: arrays[name] for name in names}
    weights = [1.0 / (1.0 + float(np.max(np.abs(np.asarray(b))))) for b in jax.jit(outputs)(x0)]

    def objective(x):
        return sum(jnp.sum(o) * w for o, w in zip(outputs(x), weights))

    value, grad = jax.jit(jax.value_and_grad(objective)).lower(x0).compile()(x0)
    assert np.isfinite(float(value))
    forward = jax.jit(objective)
    failures = []
    for name, index, rel in FD_CASES:
        xv = float(x0[name][index])
        g = float(grad[name][index])
        h = rel * abs(xv)
        fd = (float(forward({**x0, name: x0[name].at[index].set(xv + h)}))
              - float(forward({**x0, name: x0[name].at[index].set(xv - h)}))) / (2.0 * h)
        relative = abs(g - fd) / max(abs(g), abs(fd))
        if g == 0.0 or not relative <= 1.0e-6:
            failures.append((name, index, g, fd, relative))
    assert failures == []
