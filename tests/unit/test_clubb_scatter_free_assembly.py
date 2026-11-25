"""CLUBB's scatter-free banded-system assembly reproduces the scatter version.

``wp23_rhs``, ``wp23_lhs``, ``xm_wpxp_lhs`` and ``xm_wpxp_rhs`` were rewritten
to build their bands by interleaving/stacking instead of ``.at[].set`` into a
zero matrix.  The fixture holds random inputs and the outputs of the scatter
version (clubb.py at main 847d291863), in float64 and float32; the rewrite must
match them bit for bit.

Regenerate (only if the reference itself changes):
    git show 847d291863:packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py \
        > /tmp/clubb_ref.py
    JAX_ENABLE_X64=1 python tests/unit/test_clubb_scatter_free_assembly.py /tmp/clubb_ref.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "clubb_assembly_main_847d291863.npz"
_NCOL, _NZM = 7, 12
_NZT = _NZM - 1
_INVRS_DT = 1.0 / 300.0
_DT = 300.0
_W_TOL_SQD = 4e-4

_WP23_RHS = dict(rhs_pr_turb_wp3="t", rhs_pr_dfsn_wp3="t", rhs_pr_dfsn_wp2="m", rhs_pr1_wp2="m",
                 rhs_bp1_pr2_wp3="t", rhs_pr1_wp3="t", rhs_bp_pr2_wp2="m", rhs_pr3_wp2="m",
                 rhs_dp1_wp2="m", lhs_pr1_wp2="m", lhs_tp_wp3=(2, "t"), lhs_pr1_wp3="t",
                 lhs_dp1_wp2="m", lhs_ta_wp3=(5, "t"), wp2="m", wp3="t")
_WP23_LHS = dict(lhs_ma_zm=(3, "m"), lhs_diff_zm=(3, "m"), lhs_ta_wp2=(2, "m"), lhs_ac_pr2_wp2="m",
                 lhs_dp1_wp2="m", lhs_pr1_wp2="m", lhs_splat_wp2="m", lhs_ma_zt=(3, "t"),
                 lhs_diff_zt=(3, "t"), lhs_tp_wp3=(2, "t"), lhs_ac_pr2_wp3="t", lhs_pr1_wp3="t",
                 lhs_splat_wp3="t", lhs_ta_wp3=(5, "t"))
_XM_LHS = [(3, "m"), (3, "m"), (3, "t"), (3, "m"), (2, "t"), (2, "m"), "m", "m"]
_XM_RHS = ["m", "t", "m", "t", "m", "m", (3, "m"), "m"]


def _shape(kind):
    lead, grid = kind if isinstance(kind, tuple) else ((), kind)
    lead = (lead,) if lead else ()
    return lead + (_NCOL, _NZM if grid == "m" else _NZT)


def _make_inputs(dtype, rng):
    draw = lambda kind: rng.standard_normal(_shape(kind)).astype(dtype)  # noqa: E731
    return {
        "wp23_rhs": {k: draw(v) for k, v in _WP23_RHS.items()},
        "wp23_lhs": {k: draw(v) for k, v in _WP23_LHS.items()},
        "xm_wpxp_lhs": [draw(v) for v in _XM_LHS],
        "xm_wpxp_rhs": [draw(v) for v in _XM_RHS],
    }


def _evaluate(mod, inp):
    import jax.numpy as jnp
    j = lambda d: {k: jnp.asarray(v) for k, v in d.items()}  # noqa: E731
    return {
        "wp23_rhs": mod.wp23_rhs(nzm=_NZM, invrs_dt=_INVRS_DT, w_tol_sqd=_W_TOL_SQD,
                                 **j(inp["wp23_rhs"])),
        "wp23_lhs": mod.wp23_lhs(nzm=_NZM, ndim=2 * _NZM - 1, invrs_dt=_INVRS_DT,
                                 **j(inp["wp23_lhs"])),
        "xm_wpxp_lhs": mod.xm_wpxp_lhs(*map(jnp.asarray, inp["xm_wpxp_lhs"]), _DT),
        "xm_wpxp_rhs": mod.xm_wpxp_rhs(*map(jnp.asarray, inp["xm_wpxp_rhs"]), _DT),
    }


def _flatten(inp, out, tag):
    flat = {}
    for fn, args in inp.items():
        items = args.items() if isinstance(args, dict) else enumerate(args)
        for k, v in items:
            flat[f"{tag}/in/{fn}/{k}"] = v
    for fn, v in out.items():
        flat[f"{tag}/out/{fn}"] = np.asarray(v)
    return flat


def _load_inputs(data, tag):
    inp = {"wp23_rhs": {}, "wp23_lhs": {}, "xm_wpxp_lhs": {}, "xm_wpxp_rhs": {}}
    for key in data.files:
        t, kind, fn, *rest = key.split("/")
        if t == tag and kind == "in":
            inp[fn][rest[0]] = data[key]
    for fn in ("xm_wpxp_lhs", "xm_wpxp_rhs"):
        inp[fn] = [inp[fn][str(i)] for i in range(len(inp[fn]))]
    return inp


@pytest.mark.parametrize("tag", ["float64", "float32"])
def test_scatter_free_assembly_matches_scatter_reference_bitwise(tag):
    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.atmosphere.physics.turbulence import clubb

    data = np.load(_FIXTURE)
    got = _evaluate(clubb, _load_inputs(data, tag))
    for fn, v in got.items():
        ref = data[f"{tag}/out/{fn}"]
        assert v.dtype == ref.dtype, (fn, v.dtype, ref.dtype)
        assert v.shape == ref.shape, (fn, v.shape, ref.shape)
        np.testing.assert_array_equal(np.asarray(v), ref, err_msg=fn)


if __name__ == "__main__":
    import importlib.util
    import sys

    import jax
    jax.config.update("jax_enable_x64", True)
    spec = importlib.util.spec_from_file_location("clubb_ref", sys.argv[1])
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)
    rng = np.random.default_rng(0)
    flat = {}
    for dtype in (np.float64, np.float32):
        inp = _make_inputs(dtype, rng)
        flat.update(_flatten(inp, _evaluate(ref_mod, inp), np.dtype(dtype).name))
    _FIXTURE.parent.mkdir(exist_ok=True)
    np.savez_compressed(_FIXTURE, **flat)
    print(f"wrote {_FIXTURE} ({len(flat)} arrays)")
