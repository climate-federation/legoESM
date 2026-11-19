"""Host adapter of the WRF v4.8.0 Thompson port (``thompson.thompson_microphysics``).

The port is certified against the running Fortran in
tests/atmosphere/microphysics/unit/test_thompson_wrf_oracle.py.  These tests
pin the MAPPING the adapter adds (votes H3-H5, J2-J10), on the same oracle
fixture columns rebuilt in the host's representation (top-down, specific
humidity, per-volume N_r, per-kg-moist N_i):

* the adapter reproduces the port's updated state and precipitation (the
  oracle test's metric and 1e-10 tolerance);
* the carried N_c is NOT read: the droplet number is WRF's fixed Nt_c
  (user decision 2026-11-04), so carried zero and carried 4*Nt_c agree
  bitwise;
* a step that changes nothing returns zero tendencies (number included);
* the temperature tendency is the port's tten (J10);
* missing w / N_r and an out-of-range sub-step cap refuse.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
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


def _host(a, n_c=0.0):
    """The oracle's bottom-up dry-basis inputs in the host representation."""
    fl = lambda x: x[:, ::-1]                                  # noqa: E731
    r = [a[k] for k in ("qv", "qc", "qi", "qr", "qs", "qg")]
    wet = 1.0 + sum(r)
    q = [fl(x / wet) for x in r]
    rho_p = thompson_air_density(a["T"], a["p"], a["qv"])
    hyd = HydrometeorState(
        q_c=q[1], q_r=q[3], q_i=q[2], q_s=q[4], q_g=q[5],
        N_c=jnp.full(a["T"].shape, n_c), N_r=fl(a["nr"] * rho_p),
        N_i=fl(a["ni"] / wet))
    # host layer mass whose derived port thickness is the fixture's dz (V3b)
    dp = fl(constants.g * rho_p * wet * a["dz"])
    ph = jnp.concatenate([jnp.zeros_like(dp[:, :1]), jnp.cumsum(dp, axis=-1)], axis=-1)
    return fl(a["T"]), q[0], hyd, fl(a["p"]), ph, fl(a["w"])


def _adapter(a, dt, n_c=0.0, **kw):
    T, qv, hyd, p, ph, w = _host(a, n_c)
    kw.setdefault("w", w)
    return thompson_microphysics(T, qv, hyd, p, ph, None, None, dt, **kw), (T, qv, hyd)


def _port(a, dt, cloud_sed_to_rain=ThompsonConfig().cloud_sedimentation_to_precip):
    return thompson_wrf_step(*(a[k] for k in _FIN), float(dt), n_sed_max=_N_SED,
                             cloud_sed_to_rain=cloud_sed_to_rain)


def test_adapter_reproduces_port_state_and_precip(case):
    dt = 300.0
    out, (T, qv, hyd) = _adapter(case, dt)
    ref = _port(case, dt)
    fl = lambda x: np.asarray(x)[:, ::-1]                      # noqa: E731
    # back on the INPUT dry factor (vote V1b)
    wet = 1.0 + sum(case[k] for k in ("qv", "qc", "qi", "qr", "qs", "qg"))
    assert _rel(T + dt * out.dT_dt, fl(ref.T)) <= _TOL
    for host_q, rate, k in ((qv, out.dq_v_dt, "qv"), (hyd.q_c, out.dq_c_dt, "qc"),
                            (hyd.q_r, out.dq_r_dt, "qr"), (hyd.q_i, out.dq_i_dt, "qi"),
                            (hyd.q_s, out.dq_s_dt, "qs"), (hyd.q_g, out.dq_g_dt, "qg")):
        assert _rel(host_q + dt * rate, fl(getattr(ref, k) / wet)) <= _TOL, k
    tot = ref.pptrain + ref.pptsnow + ref.pptgraul + ref.pptice
    assert _rel(out.precipitation * dt, tot) <= _TOL
    assert _rel(out.precip_frozen * dt, tot - ref.pptrain) <= _TOL
    np.testing.assert_array_equal(np.asarray(out.dN_c_dt), 0.0)
    # numbers back in host units: per m^3 (N_r) and per kg moist air (N_i)
    # with the input factors (vote V2b)
    rho_p = thompson_air_density(case["T"], case["p"], case["qv"])
    assert _rel(hyd.N_r + dt * out.dN_r_dt, fl(ref.nr * rho_p)) <= _TOL
    assert _rel(hyd.N_i + dt * out.dN_i_dt, fl(ref.ni / wet)) <= _TOL


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


def test_carried_nc_is_not_read(case):
    """WRF's fixed Nt_c, not the carried N_c (user decision 2026-11-04)."""
    dt = 300.0
    o1, _ = _adapter(case, dt)
    o2, _ = _adapter(case, dt, n_c=4.0 * W.NT_C)
    for f in o1._fields:
        a, b = getattr(o1, f), getattr(o2, f)
        if a is not None:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=f)


def test_no_op_column_returns_zero_tendencies():
    """Warm, dry, condensate-free column: nothing happens, and the number
    conversions round-trip so dN_r/dN_i are exactly zero (vote J4)."""
    nz = 6
    T = jnp.linspace(250.0, 295.0, nz)[None, :]                 # top-down
    p = jnp.linspace(3.0e4, 9.5e4, nz)[None, :]
    z = jnp.zeros_like(T)
    hyd = HydrometeorState(q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
                           N_c=jnp.full_like(T, 1e8), N_r=z, N_i=z)
    ph = jnp.linspace(2.5e4, 1.0e5, nz + 1)[None, :]
    out = thompson_microphysics(T, 1e-4 * jnp.ones_like(T), hyd, p, ph, None,
                                None, 300.0, w=z)
    for f in ("dT_dt", "dq_c_dt", "dN_r_dt", "dN_i_dt", "precipitation"):
        np.testing.assert_array_equal(np.asarray(getattr(out, f)), 0.0, err_msg=f)
    # q -> r -> q is exact up to round-off of the dry-basis factor
    assert float(jnp.max(jnp.abs(out.dq_v_dt))) * 300.0 <= 1e-15 * 1e-4


def test_refusals(case):
    T, qv, hyd, p, ph, w = _host(case)
    with pytest.raises(ValueError, match="vertical velocity"):
        thompson_microphysics(T, qv, hyd, p, ph, None, None, 300.0)
    with pytest.raises(ValueError, match="p_half"):
        thompson_microphysics(T, qv, hyd, p, None, None, None, 300.0, w=w)
    with pytest.raises(ValueError, match="N_r"):
        thompson_microphysics(T, qv, hyd._replace(N_r=None), p, ph, None, None,
                              300.0, w=w)
    with pytest.raises(ValueError, match="sed_substeps_max"):
        thompson_microphysics(T, qv, hyd, p, ph, None, None, 300.0,
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
    T, qv, hyd, p, ph, w = _host(case)
    f32 = lambda x: None if x is None else x.astype(jnp.float32)  # noqa: E731
    out32 = thompson_microphysics(f32(T), f32(qv), HydrometeorState(*map(f32, hyd)),
                                  f32(p), f32(ph), None, None, 300.0, w=f32(w))
    for f in ("dT_dt", "dq_v_dt", "dN_r_dt", "precipitation"):
        x = getattr(out32, f)
        assert x.dtype == jnp.float32, f
        assert np.all(np.isfinite(np.asarray(x))), f


def test_surface_ice_counts_as_frozen_precip(case, monkeypatch):
    """The fixtures never deposit ice at the surface (pptice = 0), so the
    pptice term of the precipitation mapping is pinned with a port output
    whose surface ice is set to a known value."""
    import legoesm.atmosphere.physics.microphysics.thompson as TA
    real = TA.thompson_wrf_step
    monkeypatch.setattr(TA, "thompson_wrf_step", lambda *a, **k: real(*a, **k)._replace(
        pptice=jnp.full(case["T"].shape[:1], 0.3)))
    dt = 300.0
    out, _ = _adapter(case, dt)
    ref = _port(case, dt)
    np.testing.assert_allclose(np.asarray(out.precip_frozen) * dt,
                               np.asarray(ref.pptsnow + ref.pptgraul) + 0.3, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(out.precipitation) * dt,
                               np.asarray(ref.pptrain + ref.pptsnow + ref.pptgraul) + 0.3,
                               rtol=1e-12)


def test_refuses_without_x64(case):
    """Vote U9a: with x64 off the fp64 island would silently be fp32."""
    T, qv, hyd, p, ph, w = _host(case)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="jax_enable_x64"):
            thompson_microphysics(T, qv, hyd, p, ph, None, None, 300.0, w=w)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_spectral_w_refuses_hybrid_coordinate():
    """Vote U11a: the spectral bridge's pure-sigma omega has no hybrid form."""
    from legoesm.atmosphere.physics.microphysics import integration as MI
    from legoesm.grids.vertical import create_hybrid_coordinate
    nlev = 4
    hyb = create_hybrid_coordinate(nlev, jnp.linspace(0.1, 0.0, nlev + 1),
                                   jnp.linspace(0.0, 1.0, nlev + 1))
    z = jnp.zeros((1, nlev))
    with pytest.raises(ValueError, match="hybrid"):
        MI._spectral_w({"div": z}, jnp.ones(1), hyb, z + 280.0, z + 5e4, z, 1, nlev)


def test_host_water_budget_equals_the_ports_own(case):
    """Votes V1b-V4b: at FIXED host layer mass dp/g the adapter adds no water
    error of its own; the host budget residual equals the port's residual on
    its own masses rho_p*dz (WRF's, not zero) to round-off."""
    dt = 300.0
    out, (T, qv, hyd) = _adapter(case, dt)
    _, _, _, _, ph, _ = _host(case)
    dpg = (ph[:, 1:] - ph[:, :-1]) / constants.g
    sp = ("qv", "qc", "qi", "qr", "qs", "qg")
    q0 = qv + hyd.q_c + hyd.q_i + hyd.q_r + hyd.q_s + hyd.q_g
    dq = (out.dq_v_dt + out.dq_c_dt + out.dq_i_dt + out.dq_r_dt + out.dq_s_dt
          + out.dq_g_dt)
    host = jnp.sum(dt * dq * dpg, axis=-1) + dt * out.precipitation
    ref = _port(case, dt)
    m = thompson_air_density(case["T"], case["p"], case["qv"]) * case["dz"]
    port = (jnp.sum((sum(getattr(ref, k) for k in sp) - sum(case[k] for k in sp)) * m,
                    axis=-1)
            + ref.pptrain + ref.pptsnow + ref.pptgraul + ref.pptice)
    scale = float(jnp.max(jnp.sum(q0 * dpg, axis=-1)))
    assert float(jnp.max(jnp.abs(host - port))) <= 1e-10 * scale   # V4b


def _port_residual(a, dt, cloud_sed_to_rain):
    """Column water change on the port's own masses rho_p*dz plus its surface
    precipitation [kg/m^2]: zero for a closed budget."""
    sp = ("qv", "qc", "qi", "qr", "qs", "qg")
    ref = _port(a, dt, cloud_sed_to_rain)
    m = thompson_air_density(a["T"], a["p"], a["qv"]) * a["dz"]
    return ref, (jnp.sum((sum(getattr(ref, k) for k in sp) - sum(a[k] for k in sp)) * m,
                         axis=-1)
                 + ref.pptrain + ref.pptsnow + ref.pptgraul + ref.pptice)


def _bottom_cloud(qc0=3.0e-4):
    """The 'cols' set with only cloud water, ``qc0`` in the lowest three
    levels, and w = 0 so WRF's low-level cloud settling runs (the oracle sets
    hold no cloud there)."""
    f = dict(np.load(_FIX))
    a = {v: jnp.asarray(f[f"cols_in_{v}"]) for v in _FIN}
    z = jnp.zeros_like(a["qc"])
    a.update(qr=z, qi=z, qs=z, qg=z, ni=z, nr=z, w=z, qc=z.at[:, :3].set(qc0))
    return a


def test_cloud_sedimentation_counts_as_rain():
    """Default (user 2026-11-09): the cloud water WRF lets settle out of the
    lowest level, and books nowhere, is surface rain.  WRF-exact loses it."""
    dt = 300.0
    a = _bottom_cloud()
    ref_on, res_on = _port_residual(a, dt, True)
    ref_off, res_off = _port_residual(a, dt, False)
    added = np.asarray(ref_on.pptrain - ref_off.pptrain)
    s = added > 0.0
    assert s.sum() >= 3                                    # not vacuous
    # measured max |res| / added: 1.7e-4 (other processes, ~1e-7 kg/m^2)
    assert np.all(np.abs(np.asarray(res_on))[s] <= 1.0e-3 * added[s])
    assert np.all(np.abs(np.asarray(res_off)[s] + added[s]) <= 1.0e-3 * added[s])
    # the adapter reaches both settings through its config
    out_on, _ = _adapter(a, dt)
    out_off, _ = _adapter(a, dt, config=ThompsonConfig(cloud_sedimentation_to_precip=False))
    assert _rel(out_on.precipitation * dt, ref_on.pptrain) <= _TOL
    assert _rel(out_off.precipitation * dt, ref_off.pptrain) <= _TOL
    assert ThompsonConfig().cloud_sedimentation_to_precip is True


def test_bridges_refuse_thompson_without_the_full_registry():
    """Vote W1a: a bridge would zero-fill the missing tracers and drop their
    tendencies, silently re-initialising the port each step."""
    from legoesm.atmosphere.physics.microphysics.integration import (
        require_thompson_tracers,
    )
    from legoesm.core.tracers import make_full_moisture_registry
    full = make_full_moisture_registry().names
    require_thompson_tracers("thompson", {n: None for n in full})
    require_thompson_tracers("thompson", len(full))
    require_thompson_tracers("morrison", {})                  # other schemes untouched
    with pytest.raises(ValueError, match="N_i"):
        require_thompson_tracers("thompson", {n: None for n in full if n != "N_i"})
    with pytest.raises(ValueError, match="q_i"):
        require_thompson_tracers("thompson", 3)
    with pytest.raises(ValueError, match="lacks"):
        require_thompson_tracers("thompson", None)


@pytest.mark.parametrize("dt", [60.0, 300.0])
def test_traced_dt_matches_concrete_dt(case, dt):
    """Vote X1a: the coupled pipeline jits dt as an ordinary argument; the
    port's dt > 120 s warm-riming switch then runs as a traced select and
    gives the concrete-dt answer on both sides of the threshold.  Compared
    jit-to-jit: the adapter's jit and eager results differ by round-off that
    can flip a WRF hard switch (measured 1.5e-5 on the snow set at 60 s, the
    same with a concrete dt), which is not what this test is about."""
    T, qv, hyd, p, ph, w = _host(case)
    run = lambda d: thompson_microphysics(T, qv, hyd, p, ph, None, None, d, w=w)  # noqa: E731
    concrete, traced = jax.jit(lambda: run(dt))(), jax.jit(run)(dt)
    for f in ("dT_dt", "dq_c_dt", "dq_r_dt", "dq_s_dt", "dq_g_dt", "precipitation"):
        assert _rel(getattr(traced, f), getattr(concrete, f)) <= _TOL, f
