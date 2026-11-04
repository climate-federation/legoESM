"""DECISION 198 round 5: traced si3_var_agg(kn=2) and si3_o15_step keep bits and differentiate.

Under a JAX transform:
1. ``si3_var_agg(..., kn=2)`` (``packages/ice/legoesm/ice/si3_orca2_state.py:1176-1211``)
   runs the ``DO jl = 1, jpl / DO jk = 1, nlay_i`` (``src/ICE/icevar.F90:181-191``)
   and ``DO jl = 1, jpl / DO jk = 1, nlay_s`` (``src/ICE/icevar.F90:192-200``)
   accumulations as static-trip ``jax.lax.scan`` carries in exact ``(jl, jk)``
   order; eager calls keep the Python loops.
2. ``si3_o15_step``'s ``_ice_itd_reb`` and ``si3_ice_cor``
   (``packages/ice/legoesm/ice/si3_finalize.py:440-819``) route through the
   Round-3 traced helpers (``si3_ice_itd_reb``, ``si3_ice_cor_salinity_bounds``,
   ``si3_var_zapsmall``) in ``packages/ice/legoesm/ice/itd.py``.

Both paths are pinned to the parent program's digests at ``3dd560f7f05f`` across
``(3, 2)`` and ``(10, 5)`` ice/snow layers (including surviving IEEE-754 ``-0.0``
inputs and outputs in ``si3_var_agg(kn=2)``), and the compiled reverse passes
match central finite differences of the same compiled forward programs to
``relative <= 1.0e-6`` per entry with no absolute floor.
"""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tests.ice.unit.test_si3_o6_rdgrft_cor as o6
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice import si3_finalize as finalize_mod
from legoesm.ice import si3_orca2_state as state_mod
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_ad_probe import (
    build_negative_zero_var_agg_fixture,
)

# sha256 over (layers, path, dtype, shape, bytes) of every SI3AggregatedVars leaf
# for _var_agg_fixture() at (3, 2) and (10, 5) ice/snow layers (containing -0.0
# in all 16 input arrays and 18 surviving -0.0 output entries across 12 fields),
# measured at 3dd560f7f05f under jax.disable_jit and jax.jit.
PARENT_VAR_AGG_DIGEST = "587141e538eafe408a2cee07111a5ad3c6c0b96ac28ff571b60a4ff9527ee2ad"

# sha256 over (layers, path, dtype, shape, bytes) of every SI3CorrectionResult
# leaf for _o15_fixture() at (3, 2) and (10, 5) ice/snow layers, measured at
# 3dd560f7f05f under jax.disable_jit and jax.jit.
PARENT_O15_DIGEST = "193e003a7fe31bbf87ab7495e4406f19a032f80925470bf1a5705ced94c6894b"

LAYERS = ((o6.NLAY_I, o6.NLAY_S), (10, 5))

SURVIVING_NEG_ZERO_FIELDS = {
    "at_ip_eff": 1,
    "et_i": 1,
    "et_s": 1,
    "hm_il": 2,
    "hm_ip": 2,
    "om_i": 2,
    "sm_i": 2,
    "st_i": 1,
    "tm_i": 1,
    "tm_s": 1,
    "tm_si": 2,
    "tm_su": 2,
}


@pytest.fixture(autouse=True)
def _libm_cpu():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    with jax.default_device(jax.devices("cpu")[0]):
        yield
    set_policy(previous)


def _var_agg_fixture(nlay_i: int, nlay_s: int):
    cfg = state_mod.SI3ORCA2StateConfig.for_lane3_c1d(
        jpl=5,
        nlay_i=nlay_i,
        nlay_s=nlay_s,
        nn_hls=2,
        nn_icesal=4,
        ln_pnd=True,
        ln_pnd_LEV=True,
        ln_pnd_lids=True,
        ln_pnd_alb=True,
        ln_icethd=True,
        rn_simin=0.1,
        rn_sinew=4.0,
        rDt_ice=10800.0,
    )
    t0 = float(cfg.ice_constants.T0)
    rng = np.random.default_rng(19805 + 100 * nlay_i + nlay_s)
    shape3 = (6, 5, 5)
    shape_i = (6, 5, nlay_i, 5)
    shape_s = (6, 5, nlay_s, 5)
    a_i = rng.uniform(0.05, 0.22, size=shape3)
    v_i = a_i * rng.uniform(0.2, 3.5, size=shape3)
    v_s = a_i * rng.uniform(0.02, 0.35, size=shape3)
    sv_i = v_i * rng.uniform(1.0, 6.0, size=shape3)
    oa_i = a_i * rng.uniform(5.0, 40.0, size=shape3)
    a_ip = a_i * rng.uniform(0.05, 0.25, size=shape3)
    v_ip = a_ip * rng.uniform(0.02, 0.15, size=shape3)
    v_il = a_ip * rng.uniform(0.005, 0.04, size=shape3)
    t_su = t0 - rng.uniform(0.5, 12.0, size=shape3)
    t_si = t0 - rng.uniform(0.2, 8.0, size=shape3)
    a_ip_eff = rng.uniform(0.05, 0.25, size=shape3)
    t_i = t0 - rng.uniform(0.5, 15.0, size=shape_i)
    t_s = t0 - rng.uniform(0.5, 18.0, size=shape_s)
    e_i = v_i[:, :, None, :] * rng.uniform(2.0e8, 3.5e8, size=shape_i) / nlay_i
    e_s = v_s[:, :, None, :] * rng.uniform(0.8e8, 1.4e8, size=shape_s) / nlay_s
    szv_i = v_i[:, :, None, :] * rng.uniform(0.5, 8.0, size=shape_i) / nlay_i
    kw = {
        "v_i": jnp.asarray(v_i),
        "v_s": jnp.asarray(v_s),
        "sv_i": jnp.asarray(sv_i),
        "a_i": jnp.asarray(a_i),
        "oa_i": jnp.asarray(oa_i),
        "a_ip": jnp.asarray(a_ip),
        "v_ip": jnp.asarray(v_ip),
        "v_il": jnp.asarray(v_il),
        "t_su": jnp.asarray(t_su),
        "t_si": jnp.asarray(t_si),
        "e_s": jnp.asarray(e_s),
        "e_i": jnp.asarray(e_i),
        "szv_i": jnp.asarray(szv_i),
        "a_ip_eff": jnp.asarray(a_ip_eff),
        "t_i": jnp.asarray(t_i),
        "t_s": jnp.asarray(t_s),
    }
    return build_negative_zero_var_agg_fixture(kw), cfg


def _o15_fixture(monkeypatch, nlay_i: int, nlay_s: int):
    monkeypatch.setattr(o6, "NLAY_I", nlay_i)
    monkeypatch.setattr(o6, "NLAY_S", nlay_s)
    c, amax, sst, sss = o6._cor_inputs()
    cfg = state_mod.SI3ORCA2StateConfig.for_lane3_c1d(
        jpl=5,
        nlay_i=nlay_i,
        nlay_s=nlay_s,
        nn_hls=o6.NN_HLS,
        nn_icesal=4,
        ln_pnd=True,
        ln_pnd_LEV=True,
        ln_pnd_lids=True,
        ln_pnd_alb=True,
        ln_icethd=True,
        rn_simin=o6.COR_CFG.rn_simin,
        rn_sinew=o6.COR_CFG.rn_sinew,
        rDt_ice=o6.COR_CFG.rdt_ice,
    )
    ni, nj = c["v_i"].shape[:2]
    st0 = state_mod.SI3State.zeros(ni, nj, cfg)
    eqv = st0.equivalent._replace(
        h_i=jnp.asarray(c["h_i"]),
        h_s=jnp.asarray(c["h_s"]),
        h_ip=jnp.asarray(c["h_ip"]),
        h_il=jnp.asarray(c["h_il"]),
        sz_i=jnp.asarray(c["sz_i"]),
        t_i=jnp.asarray(c["t_i"]),
        t_s=jnp.asarray(c["t_s"]),
        a_i=jnp.asarray(c["a_i"]),
        v_s=jnp.asarray(c["v_s"]),
    )
    agg = st0.aggregated._replace(
        at_i=jnp.asarray(c["at_i"]),
        vt_i=jnp.asarray(c["vt_i"]),
        ato_i=jnp.asarray(c["ato_i"]),
    )
    st = st0._replace(
        a_i=jnp.asarray(c["a_i"]),
        v_i=jnp.asarray(c["v_i"]),
        v_s=jnp.asarray(c["v_s"]),
        sv_i=jnp.asarray(c["v_i"]) * 2.0,
        oa_i=jnp.asarray(c["oa_i"]),
        a_ip=jnp.asarray(c["a_ip"]),
        v_ip=jnp.asarray(c["v_ip"]),
        v_il=jnp.asarray(c["v_il"]),
        e_s=jnp.asarray(c["e_s"]),
        e_i=jnp.asarray(c["e_i"]),
        szv_i=jnp.asarray(c["szv_i"]),
        t_su=jnp.asarray(c["t_su"]),
        equivalent=eqv,
        aggregated=agg,
    )
    flx = finalize_mod.SI3Fluxes(
        **{k: jnp.zeros((ni, nj), dtype=jnp.float64) for k in finalize_mod.SI3Fluxes._fields}
    )._replace(
        wfx_res=jnp.asarray(c["wfx_res"]),
        hfx_res=jnp.asarray(c["hfx_res"]),
        sfx_res=jnp.asarray(c["sfx_res"]),
    )
    return st, flx, (jnp.asarray(sss), jnp.asarray(sst), jnp.asarray(amax), o6.COR_CFG.rn_himin, cfg)


def _digest(results) -> str:
    sha = hashlib.sha256()
    for layers, tree in results:
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            leaf = np.ascontiguousarray(np.asarray(leaf))
            sha.update(f"{layers}|{jax.tree_util.keystr(path)}|{leaf.dtype}|{leaf.shape}|".encode())
            sha.update(leaf.tobytes())
    return sha.hexdigest()


def test_traced_var_agg_kn2_keeps_parent_forward_bits_and_negative_zeros():
    neg_zero_bits = np.uint64(0x8000000000000000)
    eager, compiled = [], []
    for layers in LAYERS:
        kw, cfg = _var_agg_fixture(*layers)
        assert all(
            int(np.count_nonzero(np.asarray(v).view(np.uint64) == neg_zero_bits)) > 0
            for v in kw.values()
        )
        with jax.disable_jit():
            out_e = state_mod.si3_var_agg(**kw, kn=2, config=cfg)
        out_j = jax.jit(lambda d, c=cfg: state_mod.si3_var_agg(**d, kn=2, config=c))(kw)
        for out in (out_e, out_j):
            surviving = {
                name: int(
                    np.count_nonzero(np.asarray(getattr(out, name)).view(np.uint64) == neg_zero_bits)
                )
                for name in out._fields
                if int(
                    np.count_nonzero(np.asarray(getattr(out, name)).view(np.uint64) == neg_zero_bits)
                )
                > 0
            }
            assert surviving == SURVIVING_NEG_ZERO_FIELDS
        eager.append((layers, out_e))
        compiled.append((layers, out_j))
    assert (_digest(eager), _digest(compiled)) == (
        PARENT_VAR_AGG_DIGEST,
        PARENT_VAR_AGG_DIGEST,
    )


def test_traced_o15_step_keeps_parent_forward_bits(monkeypatch):
    eager, compiled = [], []
    for layers in LAYERS:
        st, flx, (sss, sst, amax, himin, cfg) = _o15_fixture(monkeypatch, *layers)
        with jax.disable_jit():
            out_e = finalize_mod.si3_o15_step(
                st, flx, sss_m=sss, sst_m=sst, rn_amax_2d=amax, rn_himin=himin, config=cfg
            )
        out_j = jax.jit(
            lambda s, f, ss=sss, st_=sst, am=amax, hm=himin, c=cfg: finalize_mod.si3_o15_step(
                s, f, sss_m=ss, sst_m=st_, rn_amax_2d=am, rn_himin=hm, config=c
            )
        )(st, flx)
        eager.append((layers, out_e))
        compiled.append((layers, out_j))
    assert (_digest(eager), _digest(compiled)) == (PARENT_O15_DIGEST, PARENT_O15_DIGEST)


# Measured on _var_agg_fixture(3, 2) at cell (3, 2):
# rel_err: 4.99e-12, 1.33e-11, 1.25e-8, 4.64e-11, 5.04e-10, 2.87e-12, 7.24e-14, 7.25e-12.
VAR_AGG_FD_CASES = (
    ("t_i", (3, 2, 1, 2), 1.0e-4),
    ("t_s", (3, 2, 0, 1), 1.0e-4),
    ("v_i", (3, 2, 2), 1.0e-4),
    ("v_s", (3, 2, 1), 1.0e-4),
    ("a_i", (3, 2, 0), 1.0e-4),
    ("t_su", (3, 2, 3), 1.0e-4),
    ("szv_i", (3, 2, 2, 4), 1.0e-4),
    ("v_ip", (3, 2, 1), 1.0e-4),
)


def test_traced_var_agg_kn2_reverse_mode_matches_central_differences():
    kw, cfg = _var_agg_fixture(*LAYERS[0])
    forward = jax.jit(lambda d: list(state_mod.si3_var_agg(**d, kn=2, config=cfg)))
    weights = [1.0 / (1.0 + float(np.max(np.abs(np.asarray(o))))) for o in forward(kw)]

    def objective(d):
        return sum(
            jnp.sum(o) * w for o, w in zip(state_mod.si3_var_agg(**d, kn=2, config=cfg), weights)
        )

    value, grad = jax.jit(jax.value_and_grad(objective)).lower(kw).compile()(kw)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(g)).all() for g in grad.values())
    failures = []
    for name, index, rel in VAR_AGG_FD_CASES:
        x0 = float(kw[name][index])
        g = float(grad[name][index])
        h = rel * abs(x0)
        plus = forward({**kw, name: kw[name].at[index].set(x0 + h)})
        minus = forward({**kw, name: kw[name].at[index].set(x0 - h)})
        fd = sum(
            float(np.sum(np.asarray(p) - np.asarray(m))) * w
            for p, m, w in zip(plus, minus, weights)
        ) / (2.0 * h)
        relative = abs(g - fd) / max(abs(g), abs(fd))
        if g == 0.0 or not relative <= 1.0e-6:
            failures.append((name, index, g, fd, relative))
    assert failures == []


O15_FD_CASES = (
    ("v_i", (5, 2, 1), 1.0e-4),
    ("a_i", (5, 2, 1), 1.0e-4),
    ("v_i", (5, 5, 0), 1.0e-4),
    ("a_i", (5, 5, 0), 1.0e-4),
    ("szv_i", (5, 2, 1, 1), 1.0e-4),
    ("v_s", (5, 5, 0), 1.0e-4),
)


def test_traced_o15_step_reverse_mode_matches_central_differences(monkeypatch):
    st, flx, (sss, sst, amax, himin, cfg) = _o15_fixture(monkeypatch, *LAYERS[0])
    diff_keys = ("v_i", "a_i", "szv_i", "v_s")
    st_dict = {k: getattr(st, k) for k in diff_keys}

    def _run_from_dict(d):
        s_in = st._replace(
            v_i=d["v_i"],
            a_i=d["a_i"],
            szv_i=d["szv_i"],
            v_s=d["v_s"],
            equivalent=st.equivalent._replace(a_i=d["a_i"], v_s=d["v_s"]),
        )
        res = finalize_mod.si3_o15_step(
            s_in, flx, sss_m=sss, sst_m=sst, rn_amax_2d=amax, rn_himin=himin, config=cfg
        )
        return [
            res.state.a_i,
            res.state.v_i,
            res.state.v_s,
            res.state.oa_i,
            res.state.a_ip,
            res.state.v_ip,
            res.state.v_il,
            res.state.t_su,
            res.state.e_s,
            res.state.e_i,
            res.state.szv_i,
            res.state.equivalent.h_i,
            res.state.equivalent.h_s,
            res.state.equivalent.h_ip,
            res.state.equivalent.h_il,
            res.state.equivalent.sz_i,
            res.state.equivalent.t_i,
            res.state.equivalent.t_s,
            res.state.aggregated.at_i,
            res.state.aggregated.vt_i,
            res.state.aggregated.ato_i,
            res.fluxes.wfx_res,
            res.fluxes.hfx_res,
            res.fluxes.sfx_res,
        ]

    forward = jax.jit(_run_from_dict)
    weights = [1.0 / (1.0 + float(np.max(np.abs(np.asarray(o))))) for o in forward(st_dict)]

    def objective(d):
        return sum(jnp.sum(o) * w for o, w in zip(_run_from_dict(d), weights))

    value, grad = jax.jit(jax.value_and_grad(objective)).lower(st_dict).compile()(st_dict)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(g)).all() for g in grad.values())
    failures = []
    for name, index, rel in O15_FD_CASES:
        x0 = float(st_dict[name][index])
        g = float(grad[name][index])
        h = rel * abs(x0)
        plus = forward({**st_dict, name: st_dict[name].at[index].set(x0 + h)})
        minus = forward({**st_dict, name: st_dict[name].at[index].set(x0 - h)})
        fd = sum(
            float(np.sum(np.asarray(p) - np.asarray(m))) * w
            for p, m, w in zip(plus, minus, weights)
        ) / (2.0 * h)
        relative = abs(g - fd) / max(abs(g), abs(fd))
        if g == 0.0 or not relative <= 1.0e-6:
            failures.append((name, index, g, fd, relative))
    assert failures == []
