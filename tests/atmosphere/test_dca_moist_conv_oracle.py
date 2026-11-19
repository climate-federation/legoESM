"""Manabe DCA vs the RUNNING GFDL moist_conv oracle.

Fixture ``tests/atmosphere/oracle_fixtures/dca_moist_conv.npz`` was generated
by RUNNING NOAA-GFDL/atmos_phys ``2021.02``
(46817a12232ed54d1a5d5a4394a695621f575812) ``moist_conv.F90``, compiled
unmodified (gfortran 8.5.0, fp64) against FMS stubs with legoESM's constants
and saturation curve, via
``scripts/validate/oracle/dca_moist_conv/{build.sh,make_fixture.py}``
(metadata in the npz ``meta``).  23 columns (the 19 tier-2 audit columns and
4 constructed supersaturated unstable ones), dt = 1800 s, coldT = F.

Tolerance (user decision 2026-10-19): max relative error <= 1e-10 in fp64 at
the hard limit of the existing smoothing fields.  The oracle has no CAPE gate;
its hard limit is the gate switched off (``cape_threshold = -1``,
``cape_sharpness = 1e12``).  Error metric: the tier-1 CLUBB oracle test's
field-normalised ``max|ours - oracle| / max|oracle|`` over all columns.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection.config import DCAConfig
from legoesm.atmosphere.physics.convection.dca import dca_convection

jax.config.update("jax_enable_x64", True)

FIXTURE = Path(__file__).resolve().parent / "oracle_fixtures" / "dca_moist_conv.npz"
TOL = 1.0e-10
NO_GATE = DCAConfig(cape_threshold=-1.0, cape_sharpness=1.0e12)


@pytest.fixture(scope="module")
def fx():
    z = dict(np.load(FIXTURE))
    z["meta"] = json.loads(str(z["meta"]))
    return z


def _run(fx, cfg, dtype=jnp.float64, fn=dca_convection):
    args = [jnp.asarray(fx[k], dtype) for k in ("T", "q", "p_full", "p_half")]
    return fn(*args, float(fx["meta"]["dt"]), cfg)


def _rain(out, fx):
    dp = np.diff(fx["p_half"], axis=1)
    return np.sum(np.asarray(out.dq_c_conv_dt) * dp / constants.g, axis=1)


def _err(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return np.inf
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def test_fixture_provenance(fx):
    m = fx["meta"]
    assert m["tag"] == "2021.02"
    assert m["sha"] == "46817a12232ed54d1a5d5a4394a695621f575812"
    assert m["dt"] == 1800.0 and m["coldT"] is False
    adjusted = np.any(fx["ora_dT"] != 0.0, axis=1)
    assert np.any(adjusted) and np.any(~adjusted)
    assert np.all(fx["ora_snow"] == 0.0)
    # coverage: a column adjusted in two separate blocks, and columns that
    # moist_conv re-tested (its Conv flags are reset by the re-test, which only
    # happens when a block reaches the bottom; they come back all-stable)
    hit = (fx["ora_dT"] != 0.0).astype(int)
    runs = [int(np.sum(np.diff(np.concatenate([[0], c, [0]])) == 1)) for c in hit]
    assert max(runs) >= 2
    assert np.any(adjusted & (fx["ora_conv"].sum(axis=1) == 0))
    assert np.any(adjusted & (fx["ora_conv"].sum(axis=1) > 0))


def test_matches_gfdl_moist_conv_with_gate_off(fx):
    out = _run(fx, NO_GATE)
    errs = {"dT_dt": _err(out.dT_dt, fx["ora_dT"]),
            "dq_v_dt": _err(out.dq_v_dt, fx["ora_dq"]),
            "rain": _err(_rain(out, fx), fx["ora_rain"])}
    print("DCA vs moist_conv max rel err", errs)
    assert max(errs.values()) <= TOL, errs
    # columns the oracle leaves alone are untouched exactly
    still = np.all(fx["ora_dT"] == 0.0, axis=1)
    assert np.all(np.asarray(out.dT_dt)[still] == 0.0)
    assert np.all(np.asarray(out.dq_v_dt)[still] == 0.0)


def test_column_budgets_close(fx):
    """Column water closes (rain = column drying on these columns) and every
    block conserves column moist enthalpy, as moist_conv does."""
    out = _run(fx, NO_GATE)
    dp = np.diff(fx["p_half"], axis=1)
    dT, dq, dqc = (np.asarray(x) for x in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt))
    water = np.sum((dq + dqc) * dp, axis=1) / constants.g
    assert np.max(np.abs(water)) <= TOL * np.max(fx["ora_rain"])
    mse = np.sum((constants.c_pd * dT + constants.L_v * dq) * dp, axis=1)
    scale = np.max(np.sum(constants.c_pd * np.abs(dT) * dp, axis=1))
    assert scale > 0.0 and np.max(np.abs(mse)) <= TOL * scale


def test_gradients_finite_fp64_and_fp32(fx):
    for dtype in (jnp.float64, jnp.float32):
        args = [jnp.asarray(fx[k], dtype) for k in ("T", "q", "p_full", "p_half")]

        def loss(T, q):
            o = dca_convection(T, q, args[2], args[3], 1800.0, DCAConfig())
            return jnp.sum(o.dT_dt) + 1.0e3 * jnp.sum(o.dq_c_conv_dt)

        gT, gq = jax.grad(loss, argnums=(0, 1))(args[0], args[1])
        assert bool(jnp.all(jnp.isfinite(gT))) and bool(jnp.all(jnp.isfinite(gq))), dtype
        assert float(jnp.max(jnp.abs(gT))) > 0.0


def test_eager_jit_parity(fx):
    a = _run(fx, DCAConfig())
    b = _run(fx, DCAConfig(), fn=jax.jit(dca_convection, static_argnums=(4, 5)))
    for x, y in zip((a.dT_dt, a.dq_v_dt, a.dq_c_conv_dt, a.cape),
                    (b.dT_dt, b.dq_v_dt, b.dq_c_conv_dt, b.cape)):
        assert _err(y, x) <= TOL


@pytest.mark.parametrize("field, value", [
    ("n_iterations", 5), ("mixing_fraction", 0.5), ("instability_blend_sharpness", 50.0)])
def test_inert_sweep_fields_reject_non_default(fx, field, value):
    with pytest.raises(ValueError, match=field):
        _run(fx, DCAConfig(**{field: value}))
    # a traced config (jit argument) cannot raise: the outputs are NaN instead
    args = [jnp.asarray(fx[k]) for k in ("T", "q", "p_full", "p_half")]
    f = jax.jit(lambda x: dca_convection(*args, 1800.0, DCAConfig(**{field: x})))
    assert bool(jnp.all(jnp.isnan(f(jnp.asarray(value)).dT_dt)))
    assert bool(jnp.all(jnp.isfinite(f(jnp.asarray(DCAConfig._field_defaults[field])).dT_dt)))
