"""Direct test for the opt-in mixed-precision baroclinic EOS lever
(``LEGOESM_BAROCLINIC_F32``): the EOS ``compute_dtype`` override + the
f32-work/f64-state path in ``iterate_eos_and_pressure_anomaly``, gated by BOTH
the env AND the caller's explicit ``allow_baroclinic_f32`` (so GM/Redi/MLE/N^2/
init callers of the shared helper are NOT affected — codex scope-leak fix).

Runs under the PRODUCTION fp64 policy (``set_policy(PrecisionPolicy.fp64())``,
as ``run_omip_core2`` does), so it exercises the policy conflict codex flagged: a
v1 input-cast was a no-op because ``wright_eos`` re-promotes to the policy f64 —
this test proves the ``compute_dtype`` path genuinely runs the EOS polynomial in
f32 (output differs from f64 by ~f32 ULP) while returning the f64 STATE dtype,
that the anomaly PGF matches f64 within the offline-experiment tolerance (job
8520588), and that the lever fires ONLY when the caller passes the flag.
"""
from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.precision import set_policy, PrecisionPolicy
from legoesm.ocean.eos import make_eos_fn, wright_eos
from legoesm.ocean.dynamics import ocean_tendency_common as otc

# Production-representative: EOS compute = f64 unless the lever overrides it.
set_policy(PrecisionPolicy.fp64())

RHO0 = float(constants.rho_ocean)
G = float(constants.g)


def _state(n_lat=8, n_lon=24, nlev=20):
    z_half = 5000.0 * (np.expm1(np.linspace(0, 1, nlev + 1) * 2.5)
                       / np.expm1(2.5))
    dz_ref = np.diff(z_half)
    z_cen = 0.5 * (z_half[:-1] + z_half[1:])
    x = np.linspace(0.0, 1.0, n_lon)
    T_surf = 20.0 - 5.0 * np.tanh((x - 0.5) * 6.0)
    decay = np.exp(-z_cen / 800.0)
    T = 2.0 + (T_surf[None, :, None] - 2.0) * decay[None, None, :]
    T = np.broadcast_to(T, (n_lat, n_lon, nlev)).copy()
    S = 35.0 + 0.2 * np.tanh((x - 0.5) * 6.0)[None, :, None]
    S = np.broadcast_to(S, (n_lat, n_lon, nlev)).copy()
    return (jnp.asarray(T, jnp.float64), jnp.asarray(S, jnp.float64),
            jnp.asarray(dz_ref, jnp.float64))


def _call(T, S, dz, allow_f32=True):
    eos = make_eos_fn(eos="wright")
    mask = jnp.ones(T.shape[:-1], dtype=T.dtype)
    return otc.iterate_eos_and_pressure_anomaly(
        T, S, mask, lambda a: a, eos, dz, RHO0, G, n_iter=2,
        allow_baroclinic_f32=allow_f32)


def test_gate_helper_env_and_dtype(monkeypatch):
    monkeypatch.setenv("LEGOESM_BAROCLINIC_F32", "1")
    assert otc._baroclinic_f32_enabled(jnp.float64) is True
    assert otc._baroclinic_f32_enabled(jnp.float32) is False   # needs f64 state
    monkeypatch.setenv("LEGOESM_BAROCLINIC_F32", "0")
    assert otc._baroclinic_f32_enabled(jnp.float64) is False
    monkeypatch.delenv("LEGOESM_BAROCLINIC_F32", raising=False)
    assert otc._baroclinic_f32_enabled(jnp.float64) is False


def test_wright_compute_dtype_actually_runs_f32():
    """Under the fp64 policy, wright_eos(compute_dtype=f32) must genuinely run the
    polynomial in f32 — differ from the f64 result by ~f32 rounding (NOT be
    bit-identical, exactly how the v1 input-cast silently failed) — while
    returning the f64 input dtype."""
    T, S, dz = _state()
    rho_f64 = wright_eos(T, S, jnp.zeros_like(T))
    rho_f32 = wright_eos(T, S, jnp.zeros_like(T), compute_dtype=jnp.float32)
    assert rho_f64.dtype == jnp.float64
    assert rho_f32.dtype == jnp.float64
    d = float(np.max(np.abs(np.asarray(rho_f32) - np.asarray(rho_f64))))
    assert d > 1e-6, f"compute_dtype=f32 did NOT change the result (no-op!): {d}"
    assert d < 1e-2, f"f32 EOS too far from f64: {d}"


def test_f32_path_returns_f64_state(monkeypatch):
    T, S, dz = _state()
    monkeypatch.setenv("LEGOESM_BAROCLINIC_F32", "1")
    rho, rho_p, p_p = _call(T, S, dz, allow_f32=True)
    assert rho.dtype == jnp.float64
    assert rho_p.dtype == jnp.float64
    assert p_p.dtype == jnp.float64


def test_f32_matches_f64_within_tol(monkeypatch):
    T, S, dz = _state()
    monkeypatch.delenv("LEGOESM_BAROCLINIC_F32", raising=False)
    rho_ref, rhop_ref, pp_ref = (np.asarray(a, np.float64)
                                 for a in _call(T, S, dz, allow_f32=True))
    monkeypatch.setenv("LEGOESM_BAROCLINIC_F32", "1")
    rho_f, rhop_f, pp_f = (np.asarray(a, np.float64)
                           for a in _call(T, S, dz, allow_f32=True))

    def relrms(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2))
                     / (np.sqrt(np.mean(b ** 2)) + 1e-300))

    assert relrms(rhop_f, rhop_ref) > 0.0           # f32 genuinely ran
    assert relrms(rhop_f, rhop_ref) < 1e-4, relrms(rhop_f, rhop_ref)
    assert relrms(pp_f, pp_ref) < 1e-4, relrms(pp_f, pp_ref)
    gx = lambda p: (p[:, 2:, :] - p[:, :-2, :])
    assert relrms(gx(pp_f), gx(pp_ref)) < 1e-3, relrms(gx(pp_f), gx(pp_ref))


def test_scope_leak_guard_requires_flag(monkeypatch):
    """codex scope-leak fix: even with the env ON, the lever must NOT fire when the
    caller leaves allow_baroclinic_f32=False (the GM/Redi/MLE/init default) — output
    must be bit-identical to the f64 path."""
    T, S, dz = _state()
    monkeypatch.setenv("LEGOESM_BAROCLINIC_F32", "1")
    rho_off, rhop_off, pp_off = (np.asarray(a) for a in _call(T, S, dz, allow_f32=False))
    monkeypatch.delenv("LEGOESM_BAROCLINIC_F32", raising=False)
    rho_ref, rhop_ref, pp_ref = (np.asarray(a) for a in _call(T, S, dz, allow_f32=False))
    np.testing.assert_array_equal(rhop_off, rhop_ref)
    np.testing.assert_array_equal(pp_off, pp_ref)


def test_default_off_byte_identical(monkeypatch):
    T, S, dz = _state()
    monkeypatch.delenv("LEGOESM_BAROCLINIC_F32", raising=False)
    a = [np.asarray(x) for x in _call(T, S, dz, allow_f32=True)]
    b = [np.asarray(x) for x in _call(T, S, dz, allow_f32=True)]
    for x, y in zip(a, b):
        assert x.dtype == np.float64
        np.testing.assert_array_equal(x, y)


def test_make_eos_fn_variants_accept_compute_dtype():
    """Every make_eos_fn variant must ACCEPT compute_dtype (so the lever never
    crashes on a non-wright EOS) and return the f64 input dtype.  (wright + the
    input-dtype variants HONOUR it — proven for wright above; unesco80/veros
    re-promote internally so it is a documented no-op there, which is safe.)"""
    T, S, dz = _state()
    p = jnp.zeros_like(T)
    for scheme in ("wright", "linear", "unesco80"):
        fn = make_eos_fn(eos=scheme)
        r0 = fn(T, S, p)
        r1 = fn(T, S, p, compute_dtype=jnp.float32)
        assert r0.dtype == jnp.float64 and r1.dtype == jnp.float64
