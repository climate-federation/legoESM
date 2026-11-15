"""The LES radiative temperature tendency as a candidate column input."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.les_suite.candidate import column_inputs, load_candidate
from legoesm.atmosphere.les_suite.candidate_eval import rollout
from legoesm.atmosphere.les_suite.scm_runner import scm_physics_config
from tests.atmosphere.les_suite.test_cloudbench_scm import _DT, _build, _turb, case  # noqa: F401

_NSTEPS = 3

_CAND = '''
import jax.numpy as jnp
KIND = "both"
READS_RAD = {flag}
PARAMS = {{"c": (1.0, 0.0, 2.0, "1")}}
def init_aux(inputs):
    return {{}}
def fluxes(params, inputs, aux):
    n = inputs.theta_l.shape[0]
    z = jnp.zeros(n - 1)
    zt = jnp.zeros(1)
    wt = jnp.concatenate([jnp.reshape(inputs.w_theta_s, (1,)), z, zt])
    wq = jnp.concatenate([jnp.reshape(inputs.w_qt_s, (1,)), z, zt])
    uw = jnp.concatenate([jnp.reshape(inputs.uw_s, (1,)), z, zt])
    vw = jnp.concatenate([jnp.reshape(inputs.vw_s, (1,)), z, zt])
    return (wt, wq), (uw, vw), aux
'''


def _cand(tmp_path, flag):
    p = tmp_path / f"cand_{str(flag).lower()}.py"
    p.write_text(_CAND.format(flag=flag))
    return load_candidate(p)


def test_cloudbench_column_carries_the_forcings_radiation(case):  # noqa: F811
    art, _ = case
    scm, _ = _build(case)
    nlev = int(scm.forcing_arg.rad_T_tendency(0.0).shape[0])

    def rad_fn(t):  # varies in time and height: catches a shifted step or no reversal
        return -1e-5 * (1.0 + t / 1000.0) * jnp.linspace(0.2, 1.0, nlev)
    scm.forcing_arg = scm.forcing_arg._replace(rad_T_tendency=rad_fn)
    rec, _ = rollout(scm, scm_physics_config(art, _turb()), _NSTEPS,
                     energy_form_baseline=True)
    got = np.asarray(rec.inputs.rad_T_tendency)              # (steps, n) surface-first
    assert got.shape[0] == _NSTEPS and np.all(np.isfinite(got))
    for k in range(_NSTEPS):
        want = np.asarray(rad_fn(k * _DT))[::-1]               # SCM channel is top-first
        np.testing.assert_allclose(got[k], want, rtol=1e-12, atol=0.0)   # D8 tolerance
    assert np.max(np.abs(got)) > 0.0
    # pin the channel to what the forcing ADDS to dT/dt: with minus without radiation
    from legoesm.atmosphere.forcing.scm.scm_forcing import compute_forcing_tendencies
    t0 = scm.t_seconds
    with_rad = compute_forcing_tendencies(scm.state, scm.sigma_coord, scm.forcing_arg, t0)
    no_rad = compute_forcing_tendencies(scm.state, scm.sigma_coord,
                                        scm.forcing_arg._replace(rad_T_tendency=None), t0)
    added = np.asarray(with_rad.dT_dt.data - no_rad.dT_dt.data)[0, 0, 0][::-1]
    np.testing.assert_allclose(added, np.asarray(rad_fn(t0))[::-1], rtol=1e-12, atol=0.0)


def test_a_column_without_a_radiation_channel_carries_nan():
    one = jnp.ones((2,))
    inp = column_inputs(z_full=one, z_half=jnp.arange(3.0), rho_full=one,
                        rho_half=jnp.ones((3,)), mass=one, exner=one,
                        exner_half=jnp.ones((3,)), p_full=one, p_half=jnp.ones((3,)),
                        theta_l=300 * one, q_t=0 * one, u=one, v=one, w_theta_s=0.0,
                        w_qt_s=0.0, uw_s=0.0, vw_s=0.0, dt=_DT)
    assert inp.rad_T_tendency.shape == (2,) and bool(jnp.all(jnp.isnan(inp.rad_T_tendency)))


def test_a_radiation_reading_candidate_refuses_a_case_without_the_channel(case, tmp_path):  # noqa: F811
    art, _ = case
    scm, _ = _build(case)
    scm.forcing_arg = scm.forcing_arg._replace(rad_T_tendency=None)
    cand = _cand(tmp_path, True)
    with pytest.raises(ValueError, match="READS_RAD"):
        rollout(scm, scm_physics_config(art, _turb()), _NSTEPS, candidate=cand,
                params=cand.default_params(), heat_form="enthalpy", ke_heating=True)


def test_a_candidate_not_reading_radiation_runs_without_the_channel(case, tmp_path):  # noqa: F811
    art, _ = case
    scm, _ = _build(case)
    scm.forcing_arg = scm.forcing_arg._replace(rad_T_tendency=None)
    cand = _cand(tmp_path, False)
    assert cand.reads_rad is False
    rec, _ = rollout(scm, scm_physics_config(art, _turb()), _NSTEPS, candidate=cand,
                     params=cand.default_params(), heat_form="enthalpy", ke_heating=True)
    assert bool(jnp.all(jnp.isnan(rec.inputs.rad_T_tendency)))


def test_read_rad_flag_must_be_a_bool(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text(_CAND.format(flag='"yes"'))
    with pytest.raises(ValueError, match="READS_RAD must be a bool"):
        load_candidate(p)
