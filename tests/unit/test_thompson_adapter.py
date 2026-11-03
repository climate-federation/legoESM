"""Host adapter of the WRF v4.8.0 Thompson port (``thompson.thompson_microphysics``).

The port is certified against the running Fortran in
tests/atmosphere/microphysics/unit/test_thompson_wrf_oracle.py.  These tests
pin the MAPPING the adapter adds (votes H3-H5, J2-J10), on the same oracle
fixture columns rebuilt in the host's representation (top-down, specific
humidity, per-volume N_r / N_c, per-kg-moist N_i):

* with the carried droplet number equal to WRF's Nt_c the adapter reproduces
  the port's updated state and precipitation (the oracle test's metric and
  1e-10 tolerance);
* the carried N_c is read (a different N_c changes the answer) and nt_c=None
  is the oracle's constant bitwise;
* a step that changes nothing returns zero tendencies (number included);
* the temperature tendency is the port's tten (J10);
* missing w / N_c and an out-of-range sub-step cap refuse.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.microphysics import thompson_wrf_tables as W
from legoesm.atmosphere.physics.microphysics.config import ThompsonConfig
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
from legoesm.atmosphere.physics.microphysics.thompson_wrf import (
    thompson_air_density,
    thompson_wrf_step,
)

_FIX = (Path(__file__).parents[1] / "atmosphere" / "microphysics"
        / "oracle_fixtures" / "thompson_wrf.npz")
_FIN = ("T", "p", "dz", "w", "qv", "qc", "qi", "qr", "qs", "qg", "ni", "nr")
_TOL = 1.0e-10           # the oracle test's metric and tolerance
_N_SED = ThompsonConfig().sed_substeps_max


def _rel(a, b):
    """max|a-b| / max|b| (R1 adapter-test metric): the specific-humidity <->
    mixing-ratio round trip leaves ~1e-16 relative noise that a pointwise
    metric with no floor amplifies on near-empty levels."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = np.max(np.abs(b))
    return float(np.max(np.abs(a - b)) / scale) if scale > 0 else float(np.max(np.abs(a)))


@pytest.fixture(scope="module", params=["cols", "snow", "ice"])
def case(request):
    f = dict(np.load(_FIX))
    a = {v: jnp.asarray(f[f"{request.param}_in_{v}"]) for v in _FIN}
    return a


def _host(a, nt_c=W.NT_C):
    """The oracle's bottom-up dry-basis inputs in the host representation."""
    fl = lambda x: x[:, ::-1]                                  # noqa: E731
    r = [a[k] for k in ("qv", "qc", "qi", "qr", "qs", "qg")]
    wet = 1.0 + sum(r)
    q = [fl(x / wet) for x in r]
    rho_p = thompson_air_density(a["T"], a["p"], a["qv"])
    hyd = HydrometeorState(
        q_c=q[1], q_r=q[3], q_i=q[2], q_s=q[4], q_g=q[5],
        N_c=jnp.full(a["T"].shape, nt_c), N_r=fl(a["nr"] * rho_p),
        N_i=fl(a["ni"] / wet))
    return fl(a["T"]), q[0], hyd, fl(a["p"]), fl(a["dz"]), fl(a["w"])


def _adapter(a, dt, nt_c=W.NT_C, **kw):
    T, qv, hyd, p, dz, w = _host(a, nt_c)
    kw.setdefault("w", w)
    return thompson_microphysics(T, qv, hyd, p, None, None, dz, dt, **kw), (T, qv, hyd)


def _port(a, dt, nt_c=None):
    return thompson_wrf_step(*(a[k] for k in _FIN), float(dt), n_sed_max=_N_SED,
                             nt_c=nt_c)


def test_adapter_reproduces_port_state_and_precip(case):
    dt = 300.0
    out, (T, qv, hyd) = _adapter(case, dt)
    ref = _port(case, dt)
    fl = lambda x: np.asarray(x)[:, ::-1]                      # noqa: E731
    wet = 1.0 + sum(getattr(ref, k) for k in ("qv", "qc", "qi", "qr", "qs", "qg"))
    assert _rel(T + dt * out.dT_dt, fl(ref.T)) <= _TOL
    for host_q, rate, k in ((qv, out.dq_v_dt, "qv"), (hyd.q_c, out.dq_c_dt, "qc"),
                            (hyd.q_r, out.dq_r_dt, "qr"), (hyd.q_i, out.dq_i_dt, "qi"),
                            (hyd.q_s, out.dq_s_dt, "qs"), (hyd.q_g, out.dq_g_dt, "qg")):
        assert _rel(host_q + dt * rate, fl(getattr(ref, k) / wet)) <= _TOL, k
    tot = ref.pptrain + ref.pptsnow + ref.pptgraul + ref.pptice
    assert _rel(out.precipitation * dt, tot) <= _TOL
    assert _rel(out.precip_frozen * dt, tot - ref.pptrain) <= _TOL
    np.testing.assert_array_equal(np.asarray(out.dN_c_dt), 0.0)


def test_fixture_precipitates():
    """The precipitation mapping above is not vacuous: the 'cols' set rains."""
    f = dict(np.load(_FIX))
    a = {v: jnp.asarray(f[f"cols_in_{v}"]) for v in _FIN}
    ref = _port(a, 300.0)
    assert float(jnp.max(ref.pptrain)) > 0.0


def test_temperature_tendency_is_tten(case):
    """Votes H4/J10: the updated temperature is T + dt*tten (the host tendency
    is taken from the updated T), compared on the temperature itself."""
    dt = 300.0
    out, (T, _, _) = _adapter(case, dt)
    ref = _port(case, dt)
    assert _rel(T + dt * out.dT_dt, T + dt * np.asarray(ref.tten)[:, ::-1]) <= _TOL


def test_nt_c_none_is_the_oracle_constant_and_carried_nc_is_read(case):
    dt = 300.0
    a, b = _port(case, dt), _port(case, dt, nt_c=jnp.full(case["T"].shape, W.NT_C))
    np.testing.assert_array_equal(np.asarray(a.qc), np.asarray(b.qc))
    np.testing.assert_array_equal(np.asarray(a.qr), np.asarray(b.qr))
    if not np.any(np.asarray(case["qc"]) > W.R1):
        return                       # no cloud water: N_c cannot enter
    o1, _ = _adapter(case, dt)
    o2, _ = _adapter(case, dt, nt_c=4.0 * W.NT_C)
    assert not np.array_equal(np.asarray(o1.dq_c_dt), np.asarray(o2.dq_c_dt))


def test_no_op_column_returns_zero_tendencies():
    """Warm, dry, condensate-free column: nothing happens, and the number
    conversions round-trip so dN_r/dN_i are exactly zero (vote J4)."""
    nz = 6
    T = jnp.linspace(250.0, 295.0, nz)[None, :]                 # top-down
    p = jnp.linspace(3.0e4, 9.5e4, nz)[None, :]
    z = jnp.zeros_like(T)
    hyd = HydrometeorState(q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
                           N_c=jnp.full_like(T, 1e8), N_r=z, N_i=z)
    out = thompson_microphysics(T, 1e-4 * jnp.ones_like(T), hyd, p, None, None,
                                jnp.full_like(T, 500.0), 300.0, w=z)
    for f in ("dT_dt", "dq_c_dt", "dN_r_dt", "dN_i_dt", "precipitation"):
        np.testing.assert_array_equal(np.asarray(getattr(out, f)), 0.0, err_msg=f)
    # q -> r -> q is exact up to round-off of the dry-basis factor
    assert float(jnp.max(jnp.abs(out.dq_v_dt))) * 300.0 <= 1e-15 * 1e-4


def test_refusals(case):
    T, qv, hyd, p, dz, w = _host(case)
    with pytest.raises(ValueError, match="vertical velocity"):
        thompson_microphysics(T, qv, hyd, p, None, None, dz, 300.0)
    with pytest.raises(ValueError, match="N_c"):
        thompson_microphysics(T, qv, hyd._replace(N_c=None), p, None, None, dz,
                              300.0, w=w)
    with pytest.raises(ValueError, match="sed_substeps_max"):
        thompson_microphysics(T, qv, hyd, p, None, None, dz, 300.0,
                              ThompsonConfig(sed_substeps_max=2000), w=w)


def test_lane_kwargs_only_for_thompson():
    from legoesm.atmosphere.physics.microphysics.integration import (
        half_level_w_to_full,
        thompson_lane_kwargs,
    )

    def boom():
        raise AssertionError("w built for a scheme that does not read it")
    assert thompson_lane_kwargs("morrison", boom) == {}
    w = thompson_lane_kwargs("thompson", lambda: jnp.ones((2, 3)))["w"]
    assert w.shape == (2, 3)
    wf = half_level_w_to_full(jnp.arange(8.0).reshape(2, 4), 2, 3)
    np.testing.assert_array_equal(np.asarray(wf), [[0.5, 1.5, 2.5], [4.5, 5.5, 6.5]])


def test_fp32_state_runs_in_fp64_and_returns_state_dtype(case):
    """Vote J11a: an fp32 state (x64 enabled) runs the fp64-certified port as
    an fp64 island (no dtype crash in the sedimentation loop) and returns
    finite fp32 tendencies."""
    T, qv, hyd, p, dz, w = _host(case)
    f32 = lambda x: None if x is None else x.astype(jnp.float32)  # noqa: E731
    out32 = thompson_microphysics(f32(T), f32(qv), HydrometeorState(*map(f32, hyd)),
                                  f32(p), None, None, f32(dz), 300.0, w=f32(w))
    for f in ("dT_dt", "dq_v_dt", "dN_r_dt", "precipitation"):
        x = getattr(out32, f)
        assert x.dtype == jnp.float32, f
        assert np.all(np.isfinite(np.asarray(x))), f
