"""Oracle match: WRF v4.8.0 Tiedtke convection (``tiedtke.tiedtke_wrf``).

Fixture ``oracle_fixtures/tiedtke_wrf.npz``: 21 columns x 2 forcings (the
tier-2 audit's inputs) and the outputs of RUNNING WRF v4.8.0
phys/module_cu_tiedtke.F (SHA 06d4240ae989cc3e50af412bb472df3d9048783c) built
with legoESM constants and the legoesm.thermo saturation curve, water only, fp64
(scripts/validate/oracle/tiedtke/{build.sh,make_fixture.py}).

Error metric (as tests/atmosphere/test_clubb_core_oracle.py): per field,
max|ours - oracle| / max|oracle| over all columns and levels.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.convection.tiedtke import tiedtke_wrf

_FIX = Path(__file__).parent / "oracle_fixtures" / "tiedtke_wrf.npz"
_TOL = 1.0e-10  # user-approved oracle-match tolerance (max relative error, fp64)
_FIELDS = {  # ours -> oracle key
    "dT_dt": "dtdt", "dq_v_dt": "dqdt", "detrained_condensate": "dcdt",
    "du_dt": "dudt", "dv_dt": "dvdt", "mfu": "mfu", "mfd": "mfd",
    "dq_c_dt": "rqccuten", "dq_i_dt": "rqicuten", "precip": "prec",
}
_IN = ("T", "q_v", "u", "v", "p_full", "p_half", "geopotential", "omega",
       "dq_v_dt_dyn", "dq_v_dt_pbl", "surface_moisture_flux", "land", "sigma")


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(_FIX))


def _inputs(fix, fk, dtype=jnp.float64):
    return {k: (jnp.asarray(fix[f"{fk}/in/{k}"]) if k == "land"
                else jnp.asarray(fix[f"{fk}/in/{k}"], dtype)) for k in _IN}


def _run(fix, fk, dtype=jnp.float64):
    dt = json.loads(str(fix["meta"]))["dt"]
    return tiedtke_wrf(**_inputs(fix, fk, dtype), dt=dt)


def _err(v, ref):
    v = np.asarray(v)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(ref))):
        return np.inf
    return float(np.max(np.abs(v - ref)) / max(np.max(np.abs(ref)), 1e-300))


@pytest.mark.parametrize("fk", ["F1", "F2"])
def test_tiedtke_matches_wrf_v480(fix, fk):
    out = _run(fix, fk)
    o = lambda k: fix[f"{fk}/out/{k}"]
    ldcum = o("ldcum") > 0
    assert ldcum.sum() >= 10  # the fixture exercises active convection
    np.testing.assert_array_equal(np.asarray(out.ldcum), ldcum)
    np.testing.assert_array_equal(np.asarray(out.ktype), o("ktype"))
    np.testing.assert_array_equal(np.asarray(out.kcbot)[ldcum] + 1, o("kcbot")[ldcum])
    np.testing.assert_array_equal(np.asarray(out.kctop)[ldcum] + 1, o("kctop")[ldcum])
    errs = {k: _err(getattr(out, k), o(ref)) for k, ref in _FIELDS.items()}
    errs["mfub"] = _err(out.mfub, np.where(ldcum, o("mfub"), 0.0))
    print(fk, "max rel err", {k: f"{v:.2e}" for k, v in errs.items()})
    bad = {k: v for k, v in errs.items() if not v <= _TOL}
    assert not bad, f"{fk}: fields above {_TOL}: {bad}"


def test_fixture_has_each_cloud_type_and_a_downdraft(fix):
    """Deep, shallow and midlevel labels and a non-zero downdraft occur in the
    oracle data.  Coverage limit: the fixture has omega = 0, so midlevel
    clouds carry only the cmfcmin floor mass flux; the omega-driven midlevel
    closure is not exercised with a material mass flux."""
    kt = np.concatenate([fix[f"{fk}/out/ktype"][fix[f"{fk}/out/ldcum"] > 0]
                         for fk in ("F1", "F2")])
    assert {1, 2, 3} <= set(kt.astype(int).tolist())
    assert any(np.min(fix[f"{fk}/out/mfd"]) < 0.0 for fk in ("F1", "F2"))


def test_jit_matches_eager(fix):
    eager = _run(fix, "F2")
    dt = json.loads(str(fix["meta"]))["dt"]
    jitted = jax.jit(tiedtke_wrf, static_argnames="dt")(**_inputs(fix, "F2"), dt=dt)
    for k in _FIELDS:  # same metric and tolerance as the oracle match
        assert _err(getattr(jitted, k), np.asarray(getattr(eager, k))) <= _TOL, k


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_gradient_finite_on_fixture_columns(fix, dtype):
    """jax.grad of the column heating + drying w.r.t. T, q_v, u and the
    moisture forcing is finite on every fixture column (piecewise
    differentiable port; hard switches are jnp.where)."""
    args = _inputs(fix, "F2", dtype)
    dt = json.loads(str(fix["meta"]))["dt"]

    def loss(T, q, u, mc):
        o = tiedtke_wrf(**{**args, "T": T, "q_v": q, "u": u, "dq_v_dt_dyn": mc}, dt=dt)
        return jnp.sum(o.dT_dt) + 1.0e3 * jnp.sum(o.dq_v_dt) + jnp.sum(o.du_dt) + jnp.sum(o.precip)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(args["T"], args["q_v"], args["u"],
                                                 args["dq_v_dt_dyn"])
    for gr in grads:
        assert np.all(np.isfinite(np.asarray(gr)))
    assert np.any(np.asarray(grads[0]) != 0.0)
