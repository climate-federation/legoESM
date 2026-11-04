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


EXPECTED_INPUT_NEG_ZERO_TOTALS = {
    (3, 2): 148,
    (10, 5): 318,
}


def test_traced_var_agg_kn2_keeps_parent_forward_bits_and_negative_zeros():
    neg_zero_bits = np.uint64(0x8000000000000000)
    eager, compiled = [], []
    for layers in LAYERS:
        kw, cfg = _var_agg_fixture(*layers)
        input_counts = {
            k: int(np.count_nonzero(np.asarray(v).view(np.uint64) == neg_zero_bits))
            for k, v in kw.items()
        }
        assert all(c > 0 for c in input_counts.values())
        assert sum(input_counts.values()) == EXPECTED_INPUT_NEG_ZERO_TOTALS[layers]
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
            assert sum(surviving.values()) == 18
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


# In synthetic negative-zero cell (2, 2), build_negative_zero_var_agg_fixture sets
# category-1 a_i = v_i = a_ip = 1.0e30 (ulp(1.0e30) = 1.407e14) and category-0
# t_su = t_si = oa_i = v_ip = v_il = szv_i[2,2,0,0] = -1.0e-300 so that dividing by
# 1.0e30 underflows below float64 subnormal 4.94e-324 to -0.0. Across the 77 non-zero
# input entries of cell (2, 2), 65 non-zero (out_field, in_name, index) derivatives
# match central FD to <= 1.0e-6 (max 4.236e-9), and these 19 entries have FD == 0.0
# while AD != 0.0 identically in HEAD and parent 3dd560f7f05f:
# - 13 entries absorbed into the 1.0e30 category-1 sum/denominator in forward float64;
# - 6 entries where -1.0e-300 / 1.0e30 intentionally underflows to -0.0 in forward float64.
VAR_AGG_CELL22_EXCLUDED = frozenset(
    {
        ("vt_i", "v_i", (2, 2, 0)),  # x0=1.0, AD=parent_AD=1.0, FD=parent_FD=0.0 (1.0 + 1e30 -> 1e30)
        ("at_i", "a_i", (2, 2, 0)),  # x0=1.0, AD=parent_AD=1.0, FD=parent_FD=0.0 (1.0 + 1e30 -> 1e30)
        ("at_ip", "a_ip", (2, 2, 0)),  # x0=1.0, AD=parent_AD=1.0, FD=parent_FD=0.0 (1.0 + 1e30 -> 1e30)
        ("ato_i", "a_i", (2, 2, 0)),  # x0=1.0, AD=parent_AD=-1.0, FD=parent_FD=0.0 (1.0 - (1.0 + 1e30))
        ("at_ip_eff", "a_i", (2, 2, 0)),  # x0=1.0, AD=parent_AD=1.755678e-1, FD=parent_FD=0.0 (absorbed in 1.5e29)
        ("at_ip_eff", "a_ip_eff", (2, 2, 0)),  # x0=1.756e-1, AD=parent_AD=1.0, FD=parent_FD=0.0 (absorbed in 1.5e29)
        ("tm_su", "t_su", (2, 2, 0)),  # x0=-1e-300, AD=parent_AD=1.0e-30, FD=parent_FD=0.0 (-1e-300/1e30 -> -0.0)
        ("hm_i", "v_i", (2, 2, 0)),  # x0=1.0, AD=parent_AD=1.0e-30, FD=parent_FD=0.0 (1.0 + 1e30 -> 1e30)
        ("hm_i", "a_i", (2, 2, 0)),  # x0=1.0, AD=parent_AD=-1.0e-30, FD=parent_FD=0.0 (1.0 + 1e30 -> 1e30)
        ("hm_s", "a_i", (2, 2, 0)),  # x0=1.0, AD=parent_AD=-8.941731e-62, FD=parent_FD=0.0 (1.0 + 1e30 -> 1e30)
        ("tm_si", "t_si", (2, 2, 0)),  # x0=-1e-300, AD=parent_AD=1.0e-30, FD=parent_FD=0.0 (-1e-300/1e30 -> -0.0)
        ("om_i", "oa_i", (2, 2, 0)),  # x0=-1e-300, AD=parent_AD=1.0e-30, FD=parent_FD=0.0 (-1e-300/1e30 -> -0.0)
        ("sm_i", "szv_i", (2, 2, 0, 0)),  # x0=-1e-300, AD=parent_AD=1.0e-30, FD=parent_FD=0.0 (-1e-300/1e30 -> -0.0)
        ("tm_i", "v_i", (2, 2, 0)),  # x0=1.0, AD=parent_AD=-2.731964e-30, FD=parent_FD=0.0 (absorbed in 2.6e32/1e30)
        ("tm_i", "t_i", (2, 2, 0, 0)),  # x0=265.9, AD=parent_AD=3.333333e-31, FD=parent_FD=0.0 (absorbed in 1e30)
        ("tm_i", "t_i", (2, 2, 1, 0)),  # x0=260.7, AD=parent_AD=3.333333e-31, FD=parent_FD=0.0 (absorbed in 1e30)
        ("tm_i", "t_i", (2, 2, 2, 0)),  # x0=262.2, AD=parent_AD=3.333333e-31, FD=parent_FD=0.0 (absorbed in 1e30)
        ("hm_ip", "v_ip", (2, 2, 0)),  # x0=-1e-300, AD=parent_AD=1.0e-30, FD=parent_FD=0.0 (-1e-300/1e30 -> -0.0)
        ("hm_il", "v_il", (2, 2, 0)),  # x0=-1e-300, AD=parent_AD=1.0e-30, FD=parent_FD=0.0 (-1e-300/1e30 -> -0.0)
    }
)

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
    in_names = tuple(kw.keys())
    out_fields = state_mod.SI3AggregatedVars._fields

    def _run_tuple(d):
        return tuple(state_mod.si3_var_agg(**d, kn=2, config=cfg))

    forward = jax.jit(_run_tuple)
    out0, vjp_fn = jax.vjp(_run_tuple, kw)
    jit_vjp = jax.jit(vjp_fn)

    # 1. Per-output-field, per-entry central FD across all 21 SI3AggregatedVars output
    # fields (including tm_i, tm_s, and all 19 other aggregated fields) and all 16 input
    # arrays across interior cells (3, 2) and (2, 2), each entry compared on its own
    # with no absolute floor. Because si3_var_agg is column-local in (i, j), perturbing
    # each (in_name, sub_idx) slice across (i, j) evaluates every cell's exact central
    # difference for that input entry.
    fd_per_out = {of: {k: np.zeros_like(np.asarray(kw[k])) for k in in_names} for of in out_fields}
    for k in in_names:
        arr = kw[k]
        for sub_idx in np.ndindex(arr.shape[2:]):
            sl = (slice(None), slice(None)) + sub_idx
            x_sl = arr[sl]
            h_sl = 1.0e-4 * jnp.abs(x_sl)
            safe_h = jnp.where(h_sl > 0.0, h_sl, 1.0)
            plus = forward({**kw, k: arr.at[sl].set(x_sl + h_sl)})
            minus = forward({**kw, k: arr.at[sl].set(x_sl - h_sl)})
            for idx_of, of in enumerate(out_fields):
                fd_per_out[of][k][sl] = (np.asarray(plus[idx_of]) - np.asarray(minus[idx_of])) / (
                    2.0 * np.asarray(safe_h)
                )

    cell32_passed = 0
    cell32_tm_i_passed = 0
    cell32_tm_s_passed = 0
    cell22_passed = 0
    seen_cell22_excluded = set()
    per_entry_failures = []

    for idx_of, of in enumerate(out_fields):
        cot = tuple(
            jnp.ones_like(o) if idx == idx_of else jnp.zeros_like(o) for idx, o in enumerate(out0)
        )
        grad_out = jit_vjp(cot)[0]
        assert all(np.isfinite(np.asarray(g)).all() for g in grad_out.values())
        for k in in_names:
            g_arr = np.asarray(grad_out[k])
            x_arr = np.asarray(kw[k])
            f_arr = fd_per_out[of][k]
            for cell in ((3, 2), (2, 2)):
                for sub_idx in np.ndindex(x_arr.shape[2:]):
                    idx = cell + sub_idx
                    x0 = float(x_arr[idx])
                    if x0 == 0.0:
                        continue
                    g = float(g_arr[idx])
                    fd = float(f_arr[idx])
                    if g == 0.0 and fd == 0.0:
                        continue
                    key = (of, k, idx)
                    if key in VAR_AGG_CELL22_EXCLUDED:
                        assert g != 0.0 and fd == 0.0
                        seen_cell22_excluded.add(key)
                        continue
                    relative = abs(g - fd) / max(abs(g), abs(fd))
                    if g == 0.0 or not relative <= 1.0e-6:
                        per_entry_failures.append((of, k, idx, g, fd, relative))
                    elif cell == (3, 2):
                        cell32_passed += 1
                        if of == "tm_i":
                            cell32_tm_i_passed += 1
                        elif of == "tm_s":
                            cell32_tm_s_passed += 1
                    else:
                        cell22_passed += 1

    assert per_entry_failures == []
    assert cell32_passed == 210
    assert cell32_tm_i_passed == 20
    assert cell32_tm_s_passed == 15
    assert cell22_passed == 65
    assert seen_cell22_excluded == VAR_AGG_CELL22_EXCLUDED

    # 2. Weighted-objective per-entry FD check across all 115 active entries in cell (3, 2)
    # and the 8 canonical VAR_AGG_FD_CASES.
    weights = [1.0 / (1.0 + float(np.max(np.abs(np.asarray(o))))) for o in out0]

    def objective(d):
        return sum(
            jnp.sum(o) * w for o, w in zip(state_mod.si3_var_agg(**d, kn=2, config=cfg), weights)
        )

    value, grad = jax.jit(jax.value_and_grad(objective)).lower(kw).compile()(kw)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(g)).all() for g in grad.values())
    failures = []
    weighted_cell32_passed = 0
    for name in in_names:
        for sub_idx in np.ndindex(kw[name].shape[2:]):
            index = (3, 2) + sub_idx
            g = float(grad[name][index])
            fd = sum(float(fd_per_out[of][name][index]) * w for of, w in zip(out_fields, weights))
            if g == 0.0 and fd == 0.0:
                assert name == "sv_i"  # nn_icesal == 4 uses szv_i, so sv_i has exact 0 derivative
                continue
            relative = abs(g - fd) / max(abs(g), abs(fd))
            if g == 0.0 or not relative <= 1.0e-6:
                failures.append((name, index, g, fd, relative))
            else:
                weighted_cell32_passed += 1
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
    assert weighted_cell32_passed == 115
    assert failures == []


O15_FD_CASES = (
    ("v_i", (5, 2, 1), 1.0e-4),
    ("a_i", (5, 2, 1), 1.0e-4),
    ("v_i", (5, 5, 0), 1.0e-4),
    ("a_i", (5, 5, 0), 1.0e-4),
    ("szv_i", (5, 2, 1, 1), 1.0e-4),
    ("v_s", (5, 5, 0), 1.0e-4),
)

# Individual (in_name, in_index, out_name, out_index) pairs for si3_o15_step where each
# output entry is differentiated and compared against central FD on its own (no cross-field
# or cross-category reduction), spanning rebinning, salinity bounds, and zapsmall fluxes.
O15_OUTPUT_ENTRY_FD_CASES = (
    ("v_i", (5, 2, 1), "v_i", (5, 2, 4)),
    ("v_i", (5, 2, 1), "v_s", (5, 2, 4)),
    ("v_i", (5, 2, 1), "v_ip", (5, 2, 4)),
    ("v_i", (5, 2, 1), "v_il", (5, 2, 4)),
    ("v_i", (5, 2, 1), "e_i", (5, 2, 0, 4)),
    ("v_i", (5, 2, 1), "e_s", (5, 2, 0, 4)),
    ("v_i", (5, 2, 1), "szv_i", (5, 2, 0, 4)),
    ("v_i", (5, 2, 1), "h_i", (5, 2, 4)),
    ("v_i", (5, 2, 1), "vt_i", (5, 2)),
    ("v_i", (5, 2, 1), "sfx_res", (5, 2)),
    ("a_i", (5, 2, 1), "a_i", (5, 2, 1)),
    ("a_i", (5, 2, 1), "a_i", (5, 2, 4)),
    ("a_i", (5, 2, 1), "t_su", (5, 2, 2)),
    ("a_i", (5, 2, 1), "at_i", (5, 2)),
    ("v_i", (5, 5, 0), "v_i", (5, 5, 0)),
    ("v_i", (5, 5, 0), "a_i", (5, 5, 0)),
    ("v_i", (5, 5, 0), "a_ip", (5, 5, 0)),
    ("a_i", (5, 5, 0), "a_ip", (5, 5, 0)),
    ("v_s", (5, 5, 0), "v_s", (5, 5, 0)),
    ("oa_i", (5, 2, 1), "oa_i", (5, 2, 4)),
    ("v_ip", (5, 2, 1), "v_ip", (5, 2, 4)),
    ("e_i", (5, 2, 0, 1), "e_i", (5, 2, 0, 4)),
    ("szv_i", (5, 2, 1, 1), "sfx_res", (5, 2)),
    ("szv_i", (5, 2, 1, 1), "szv_i", (5, 2, 1, 4)),
    ("e_i", (3, 3, 0, 0), "hfx_res", (3, 3)),
)

# Across all 1,057 non-zero interior entries of the 7 active inputs
# ("a_i", "v_i", "v_s", "oa_i", "v_ip", "e_i", "szv_i") on _o15_fixture(3, 2):
# - 1,007 entries pass relative <= 1.0e-6 (max 1.145e-7);
# - 11 entries in zapsmalled categories (v_i <= 1.0e-11 < 1.0e-10) have exact zero
#   derivative in both AD and FD (AD == FD == 0.0 in both HEAD and 3dd560f7f05f)
#   because ice_var_zapsmall zeroes a_i and oa_i without routing them into water/heat/salt
#   conservation fluxes and a_ip is zero in those categories;
# - 39 entries in zapsmalled categories (v_i <= 1.0e-11 < 1.0e-10) have inputs x0 <= 1.0e-11
#   where the step h = 1.0e-4 * |x0| <= 1.0e-15 falls below the output fluxes' float64
#   rounding floor identically in HEAD and parent 3dd560f7f05f.
O15_EXACT_ZERO_ENTRIES = frozenset(
    {
        ("a_i", (2, 2, 0)),
        ("a_i", (2, 2, 1)),
        ("a_i", (2, 3, 0)),
        ("a_i", (3, 4, 0)),
        ("oa_i", (2, 2, 0)),
        ("oa_i", (2, 2, 1)),
        ("oa_i", (2, 3, 0)),
        ("oa_i", (2, 4, 0)),
        ("oa_i", (3, 2, 0)),
        ("oa_i", (3, 3, 0)),
        ("oa_i", (3, 4, 0)),
    }
)

O15_ZAPSMALL_EXCLUDED_ENTRIES = frozenset(
    {
        ("a_i", (2, 4, 0)),  # x0=1.000e-12, AD=parent_AD=-1.596913, FD=parent_FD=-1.192845
        ("a_i", (3, 2, 0)),  # x0=1.000e-12, AD=parent_AD=-1.360182, FD=parent_FD=-1.184121
        ("a_i", (3, 3, 0)),  # x0=1.400e-12, AD=parent_AD=-1.611140, FD=parent_FD=-1.344075
        ("v_i", (2, 2, 0)),  # x0=7.498e-13, AD=parent_AD=0.0848655, FD=parent_FD=0.0852745
        ("v_i", (2, 2, 1)),  # x0=1.000e-11, AD=parent_AD=0.0848655, FD=parent_FD=0.0851810
        ("v_i", (2, 3, 0)),  # x0=5.915e-14, AD=parent_AD=0.0848655, FD=parent_FD=0.0847288
        ("v_i", (2, 4, 0)),  # x0=1.225e-12, AD=parent_AD=0.260545, FD=parent_FD=-0.0143443
        ("v_i", (3, 2, 0)),  # x0=9.733e-13, AD=parent_AD=0.234920, FD=parent_FD=0.340024
        ("v_i", (3, 3, 0)),  # x0=6.531e-13, AD=parent_AD=0.235027, FD=parent_FD=1.947051
        ("v_i", (3, 4, 0)),  # x0=1.638e-13, AD=parent_AD=0.0848655, FD=parent_FD=0.0847785
        ("v_s", (2, 2, 0)),  # x0=1.947e-13, AD=parent_AD=0.0305405, FD=parent_FD=0.0306069
        ("v_s", (2, 3, 0)),  # x0=7.155e-15, AD=parent_AD=0.0305405, FD=parent_FD=0.0283996
        ("v_s", (2, 4, 0)),  # x0=1.022e-13, AD=parent_AD=0.0305405, FD=parent_FD=0.0304946
        ("v_s", (3, 2, 0)),  # x0=2.560e-13, AD=parent_AD=0.0305405, FD=parent_FD=0.0305521
        ("v_s", (3, 3, 0)),  # x0=6.411e-14, AD=parent_AD=0.0305405, FD=parent_FD=0.0306365
        ("v_s", (3, 4, 0)),  # x0=1.821e-14, AD=parent_AD=0.0305405, FD=parent_FD=0.0297581
        ("v_ip", (2, 2, 0)),  # x0=6.865e-15, AD=parent_AD=0.0925469, FD=parent_FD=0.157852
        ("v_ip", (2, 3, 0)),  # x0=3.162e-14, AD=parent_AD=0.0925469, FD=parent_FD=0.0921059
        ("v_ip", (2, 4, 0)),  # x0=6.187e-15, AD=parent_AD=0.0925469, FD=parent_FD=0.0985246
        ("v_ip", (3, 2, 0)),  # x0=5.728e-16, AD=parent_AD=0.0925469, FD=parent_FD=0.118235
        ("v_ip", (3, 3, 0)),  # x0=8.808e-14, AD=parent_AD=0.0925469, FD=parent_FD=0.0926606
        ("v_ip", (3, 4, 0)),  # x0=1.380e-14, AD=parent_AD=0.0925469, FD=parent_FD=0.0932204
        ("szv_i", (2, 2, 0, 0)),  # x0=9.204e-13, AD=parent_AD=0.0161453, FD=parent_FD=0.0179203
        ("szv_i", (2, 2, 1, 0)),  # x0=1.751e-12, AD=parent_AD=0.0161453, FD=parent_FD=0.0169547
        ("szv_i", (2, 2, 2, 0)),  # x0=3.237e-13, AD=parent_AD=0.0161453, FD=parent_FD=0.0101917
        ("szv_i", (2, 3, 0, 0)),  # x0=5.700e-14, AD=parent_AD=0.0161453, FD=parent_FD=0.0162749
        ("szv_i", (2, 3, 1, 0)),  # x0=1.169e-13, AD=parent_AD=0.0161453, FD=parent_FD=0.0160957
        ("szv_i", (2, 3, 2, 0)),  # x0=6.650e-14, AD=parent_AD=0.0161453, FD=parent_FD=0.0162757
        ("szv_i", (2, 4, 0, 0)),  # x0=3.157e-12, AD=parent_AD=0.0161453, FD=parent_FD=0.0161452
        ("szv_i", (2, 4, 1, 0)),  # x0=1.048e-12, AD=parent_AD=0.0161453, FD=parent_FD=0.0161445
        ("szv_i", (3, 2, 0, 0)),  # x0=5.271e-13, AD=parent_AD=0.0161453, FD=parent_FD=0.0161332
        ("szv_i", (3, 2, 1, 0)),  # x0=1.672e-12, AD=parent_AD=0.0161453, FD=parent_FD=0.0161490
        ("szv_i", (3, 2, 2, 0)),  # x0=2.112e-12, AD=parent_AD=0.0161453, FD=parent_FD=0.0161454
        ("szv_i", (3, 3, 0, 0)),  # x0=2.748e-13, AD=parent_AD=0.0161453, FD=parent_FD=0.0161313
        ("szv_i", (3, 3, 1, 0)),  # x0=3.516e-13, AD=parent_AD=0.0161453, FD=parent_FD=0.0161269
        ("szv_i", (3, 3, 2, 0)),  # x0=1.051e-12, AD=parent_AD=0.0161453, FD=parent_FD=0.0161572
        ("szv_i", (3, 4, 0, 0)),  # x0=4.212e-13, AD=parent_AD=0.0161453, FD=parent_FD=0.0161525
        ("szv_i", (3, 4, 1, 0)),  # x0=1.189e-13, AD=parent_AD=0.0161453, FD=parent_FD=0.0160334
        ("szv_i", (3, 4, 2, 0)),  # x0=2.812e-13, AD=parent_AD=0.0161453, FD=parent_FD=0.0161299
    }
)


def test_traced_o15_step_reverse_mode_matches_central_differences(monkeypatch):
    st, flx, (sss, sst, amax, himin, cfg) = _o15_fixture(monkeypatch, *LAYERS[0])
    diff_keys = ("a_i", "v_i", "v_s", "oa_i", "v_ip", "e_i", "szv_i")
    st_dict = {k: getattr(st, k) for k in diff_keys}
    out_names = (
        "a_i",
        "v_i",
        "v_s",
        "oa_i",
        "a_ip",
        "v_ip",
        "v_il",
        "t_su",
        "e_s",
        "e_i",
        "szv_i",
        "h_i",
        "h_s",
        "h_ip",
        "h_il",
        "sz_i",
        "t_i",
        "t_s",
        "at_i",
        "vt_i",
        "ato_i",
        "wfx_res",
        "hfx_res",
        "sfx_res",
    )

    def _run_from_dict(d):
        s_in = st._replace(
            a_i=d["a_i"],
            v_i=d["v_i"],
            v_s=d["v_s"],
            oa_i=d["oa_i"],
            v_ip=d["v_ip"],
            e_i=d["e_i"],
            szv_i=d["szv_i"],
            equivalent=st.equivalent._replace(a_i=d["a_i"], v_s=d["v_s"]),
        )
        res = finalize_mod.si3_o15_step(
            s_in, flx, sss_m=sss, sst_m=sst, rn_amax_2d=amax, rn_himin=himin, config=cfg
        )
        return (
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
        )

    forward = jax.jit(_run_from_dict)
    out0, vjp_fn = jax.vjp(_run_from_dict, st_dict)
    jit_vjp = jax.jit(vjp_fn)
    weights = [1.0 / (1.0 + float(np.max(np.abs(np.asarray(o))))) for o in out0]

    def objective(d):
        return sum(jnp.sum(o) * w for o, w in zip(_run_from_dict(d), weights))

    value, grad = jax.jit(jax.value_and_grad(objective)).lower(st_dict).compile()(st_dict)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(g)).all() for g in grad.values())

    # 1. Full interior scan across all 1,057 non-zero interior entries of the 7 inputs,
    # each entry judged by its own relative error with no absolute floor (as Round 3 did).
    # Because si3_o15_step is column-local in (i, j), perturbing each (in_name, sub_idx)
    # slice across (i, j) evaluates every interior cell's exact central difference.
    fd_weighted = {k: np.zeros_like(np.asarray(st_dict[k])) for k in diff_keys}
    out_idx_map = {name: idx for idx, name in enumerate(out_names)}
    fd_output_entries = {}
    for name in diff_keys:
        arr = st_dict[name]
        for sub_idx in np.ndindex(arr.shape[2:]):
            sl = (slice(None), slice(None)) + sub_idx
            x_sl = arr[sl]
            h_sl = 1.0e-4 * jnp.abs(x_sl)
            safe_h = jnp.where(h_sl > 0.0, h_sl, 1.0)
            plus = forward({**st_dict, name: arr.at[sl].set(x_sl + h_sl)})
            minus = forward({**st_dict, name: arr.at[sl].set(x_sl - h_sl)})
            cell_diff = np.zeros(x_sl.shape, dtype=np.float64)
            for p, m, w in zip(plus, minus, weights):
                dp = np.asarray(p) - np.asarray(m)
                dp_cell = np.sum(dp, axis=tuple(range(2, dp.ndim))) if dp.ndim > 2 else dp
                cell_diff += dp_cell * w
            fd_weighted[name][sl] = cell_diff / (2.0 * np.asarray(safe_h))
            for in_name, in_idx, out_name, out_idx in O15_OUTPUT_ENTRY_FD_CASES:
                if in_name == name and in_idx[2:] == sub_idx:
                    target_pos = out_idx_map[out_name]
                    h_entry = float(safe_h[in_idx[0], in_idx[1]])
                    fd_output_entries[(in_name, in_idx, out_name, out_idx)] = (
                        float(plus[target_pos][out_idx] - minus[target_pos][out_idx])
                        / (2.0 * h_entry)
                    )

    v_i_np = np.asarray(st.v_i)
    total_nonzero = 0
    passed_count = 0
    seen_exact_zero = set()
    seen_excluded = set()
    failures = []
    for name in diff_keys:
        arr = np.asarray(st_dict[name])
        g_arr = np.asarray(grad[name])
        f_arr = fd_weighted[name]
        for i in range(2, arr.shape[0] - 2):
            for j in range(2, arr.shape[1] - 2):
                for sub_idx in np.ndindex(arr.shape[2:]):
                    index = (i, j) + sub_idx
                    x0 = float(arr[index])
                    if x0 == 0.0:
                        continue
                    total_nonzero += 1
                    g = float(g_arr[index])
                    fd = float(f_arr[index])
                    key = (name, index)
                    if key in O15_EXACT_ZERO_ENTRIES:
                        assert g == 0.0 and fd == 0.0
                        assert float(v_i_np[i, j, sub_idx[-1]]) <= 1.0e-11
                        seen_exact_zero.add(key)
                        continue
                    if key in O15_ZAPSMALL_EXCLUDED_ENTRIES:
                        assert float(v_i_np[i, j, sub_idx[-1]]) <= 1.0e-11
                        assert abs(x0) <= 1.0e-11
                        seen_excluded.add(key)
                        continue
                    relative = abs(g - fd) / max(abs(g), abs(fd))
                    if g == 0.0 or not relative <= 1.0e-6:
                        failures.append((name, index, g, fd, relative))
                    else:
                        passed_count += 1

    # Also verify the 6 canonical O15_FD_CASES with single-entry perturbations.
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

    # 2. Individual output-entry FD checks (O15_OUTPUT_ENTRY_FD_CASES): each output entry
    # out_field[out_idx] is differentiated on its own and compared against its own central
    # difference with no cross-field or cross-category reduction and no absolute floor.
    grad_cache = {}
    for in_name, in_idx, out_name, out_idx in O15_OUTPUT_ENTRY_FD_CASES:
        if (out_name, out_idx) not in grad_cache:
            target_pos = out_idx_map[out_name]
            cot = tuple(
                jnp.zeros_like(o).at[out_idx].set(1.0) if idx == target_pos else jnp.zeros_like(o)
                for idx, o in enumerate(out0)
            )
            grad_cache[(out_name, out_idx)] = jit_vjp(cot)[0]
        g = float(grad_cache[(out_name, out_idx)][in_name][in_idx])
        fd = fd_output_entries[(in_name, in_idx, out_name, out_idx)]
        relative = abs(g - fd) / max(abs(g), abs(fd))
        if g == 0.0 or not relative <= 1.0e-6:
            failures.append(((in_name, in_idx, out_name, out_idx), g, fd, relative))

    assert failures == []
    assert total_nonzero == 1057
    assert passed_count == 1007
    assert seen_exact_zero == O15_EXACT_ZERO_ENTRIES
    assert seen_excluded == O15_ZAPSMALL_EXCLUDED_ENTRIES
