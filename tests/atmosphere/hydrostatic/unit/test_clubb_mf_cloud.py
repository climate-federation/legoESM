"""CLUBB-MF plume cloud into the cam6_clubb deep cloud (CAM7 clubb3_run,
do_clubb_mf_rad), combined with ZM's as the larger of the two (user
2026-11-23): RULE 5 hard limit, stored per-case gap, check_grads; the
moist-plume aggregates against CAM's loop.  Re-measure the stored gaps with
``python test_clubb_mf_cloud.py`` when a smooth form or default changes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.turbulence.clubb_mf import (
    HARD,
    MF_CONCLD_MAX,
    integrate_mf,
    mf_concld,
    mf_deep_cloud,
)

jax.config.update("jax_enable_x64", True)

GAPS = Path(__file__).resolve().parents[2] / "oracle_fixtures" / "clubb_mf_cloud_gaps.json"
FIXTURE = GAPS.with_name("clubb_mf.npz")
CFG = CloudConfig(scheme="cam6_clubb", clubb_mf_cloud=True)
HARD_CFG = CFG._replace(mf_deepcu_select_sharpness=math.inf, mf_deepcu_cap_sharpness=math.inf,
                        mf_deepcu_frac_gate_sharpness=math.inf,
                        mf_deepcu_ic_gate_sharpness=math.inf)
# case -> (ZM deepcu, ZM icwmr, plume cloud fraction, plume qc); one column of
# four layers (the bottom one is always zeroed, as CAM's k = 1..pver-1 loop)
CASES = {
    "plume_only": ([0.0, 0.0, 0.0, 0.0], [0.0] * 4, [0.02, 0.05, 0.01, 0.03], [1e-3] * 4),
    "zm_only": ([0.3, 0.2, 0.1, 0.0], [2e-4] * 4, [0.0] * 4, [0.0] * 4),
    "both_near_tie": ([0.3, 0.2, 0.1, 0.0], [2e-4] * 4,
                      [0.0301, 0.0199, 0.0102, 0.02], [1e-3] * 4),
    "frac_gate": ([0.0] * 4, [0.0] * 4, [0.00101, 0.00099, 0.001, 0.02], [1e-3] * 4),
    "ic_gate": ([0.0] * 4, [0.0] * 4, [0.05] * 4, [1.01e-12, 0.99e-12, 1e-12, 1e-3]),
    "cap": ([0.0] * 4, [0.0] * 4, [0.0805, 0.0795, 0.2, 0.08], [1e-3] * 4),
}


def _arr(case):
    return tuple(jnp.asarray([v], jnp.float64) for v in CASES[case])


def _cam(case):
    """CAM7's hard form (clubb3_run :3455-3473) + the user's larger-of-two."""
    zm, zi, cf, qc = (np.asarray(v, float) for v in CASES[case])
    dc = 10.0 * cf
    dc = np.where((dc <= 0.01) | (qc < 1e-12), 0.0, dc)
    dc[-1] = 0.0
    pick = dc > zm
    deep = np.where(pick, dc, zm)
    return deep, np.where(pick, qc, zi), np.minimum(deep, 0.8)


def _ours(case, cfg):
    zm, zi, cf, qc = _arr(case)
    deep, icw = mf_deep_cloud(zm, zi, cf, qc, cfg)
    return np.asarray(deep[0]), np.asarray(icw[0]), np.asarray(mf_concld(deep, cfg)[0])


@pytest.mark.parametrize("case", list(CASES))
def test_hard_limit_is_cams_hookup_with_larger_of_two(case):
    for got, want in zip(_ours(case, HARD_CFG), _cam(case)):
        np.testing.assert_array_equal(got, want, err_msg=case)


def _gaps():
    return {c: [float(np.max(np.abs(g - w))) for g, w in zip(_ours(c, CFG), _cam(c))]
            for c in CASES}


def test_default_gap_per_case_is_the_stored_one():
    """RULE 5 KNOWN DIFFERENCE (deepcu, icwmr, concld) per case at the defaults."""
    stored = json.loads(GAPS.read_text())
    got = _gaps()
    assert set(got) == set(stored)
    for c in got:
        np.testing.assert_allclose(got[c], stored[c], rtol=1e-6, atol=1e-12, err_msg=c)


@pytest.mark.parametrize("case, which", [
    ("both_near_tie", "select"), ("cap", "cap"), ("frac_gate", "frac"), ("ic_gate", "ic")])
def test_check_grads_through_each_switch_at_default(case, which):
    """Gradient w.r.t. the plume cloud on the switch: finite, nonzero, and
    different from the same point with only that switch hard."""
    zm, zi, cf, qc = _arr(case)
    field = {"select": "mf_deepcu_select_sharpness", "cap": "mf_deepcu_cap_sharpness",
             "frac": "mf_deepcu_frac_gate_sharpness", "ic": "mf_deepcu_ic_gate_sharpness"}[which]

    def f(cf_, qc_, cfg=CFG):
        deep, icw = mf_deep_cloud(zm, zi, cf_, qc_, cfg)
        return jnp.sum(mf_concld(deep, cfg)) + jnp.sum(icw)

    # eps well inside each switch's width (centred FD truncation stays < 1e-6)
    eps = {"select": 1e-7, "cap": 1e-7, "frac": 1e-8, "ic": 1e-16}[which]
    check_grads(f, (cf, qc), order=1, modes=("fwd", "rev"), eps=eps)
    g = jax.grad(f, argnums=(0, 1))(cf, qc)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in g)
    assert any(np.any(np.asarray(x) != 0.0) for x in g)
    g_hard = jax.grad(lambda a, b: f(a, b, CFG._replace(**{field: math.inf})),
                      argnums=(0, 1))(cf, qc)
    assert any(np.any(np.asarray(a) != np.asarray(b)) for a, b in zip(g, g_hard))


def test_two_empty_clouds_stay_exactly_empty():
    z = jnp.zeros((1, 4))
    deep, icw = mf_deep_cloud(z, z, z, z, CFG)
    assert float(jnp.max(jnp.abs(deep))) == 0.0 and float(jnp.max(jnp.abs(icw))) == 0.0
    assert float(jnp.max(mf_concld(deep, CFG))) == 0.0
    assert MF_CONCLD_MAX == 0.8


def test_moist_aggregates_follow_cams_loop():
    """moist_a = sum of upa over plumes with upqc > 0; moist_qc = their
    upa-weighted mean upqc (clubb_mf.F90:1628-1676), hard limit, every
    fixture column."""
    fx = dict(np.load(FIXTURE, allow_pickle=True))
    keys = ("zt", "dzt", "rho_zt", "p_zt", "u", "v", "thl", "qt", "thv", "zm", "p_zm",
            "iex_zm", "rho_zm", "wthl_sfc", "wqt_sfc", "ths", "pblh", "lts")
    for i in range(fx["c_dzt"].shape[0]):
        c = {k: jnp.asarray(fx[f"c_{k}"][i]) for k in keys}
        o = integrate_mf(c, jnp.asarray(fx["ora1_enti"][i]),
                         jnp.asarray(fx["ora1_ztopm1_in"][i]), HARD)
        upa, upqc = np.asarray(o["upa"]), np.asarray(o["upqc"])
        moist = upqc > 0.0
        a = np.sum(np.where(moist, upa, 0.0), axis=1)
        q = np.where(a > 0.0, np.sum(np.where(moist, upa * upqc, 0.0), axis=1)
                     / np.where(a > 0.0, a, 1.0), 0.0)
        np.testing.assert_allclose(np.asarray(o["moist_a"]), a, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(np.asarray(o["moist_qc"]), q, rtol=1e-12, atol=1e-300)


if __name__ == "__main__":
    GAPS.write_text(json.dumps(_gaps(), indent=1))
    print("wrote", GAPS, _gaps())
