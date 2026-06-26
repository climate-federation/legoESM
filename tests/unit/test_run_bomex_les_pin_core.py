"""Deck-free coverage for the ``--core pin`` (non-spectral) path of the BOMEX
reference driver.

The full driver reads a gSAM deck (skipped where absent), so this exercises the
new core wiring directly: the pin-config dispatch, grid/state construction, and a
real ``pin.step`` + Morrison microphysics adapter on a synthetic sounding —
asserting the non-spectral core advances a moist column to finite state. Also
locks the dispatch-hardening (``weno5_hv`` has no pin analogue → hard exit).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics import pseudo_incompressible_plane as pin
from legoesm.atmosphere.dynamics.spectral_les_moist import (
    make_anelastic_reference,
    make_les_microphysics_fn,
)
from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
    MorrisonConfig,
)

_DRIVER = (Path(__file__).resolve().parents[2]
           / "scripts" / "run" / "run_bomex_les.py")


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_bomex_les", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(**over):
    base = dict(nx=4, ny=4, nz=24, Lx=400.0, Ly=400.0, Lz=1200.0, z0=1e-4,
                cs=0.18, dynamic=True, sgs_model="vreman", nu_floor=0.0,
                n_tracers=9, scalar_advection="van_leer", theta_hyperdiff=0.0,
                core="pin")
    base.update(over)
    return argparse.Namespace(**base)


def test_pin_config_builds_grid_and_dispatch_hardening():
    drv = _load_driver()
    cfg = drv._pin_config(_args())
    assert cfg.moist and cfg.surface == "flux" and cfg.scheme == "van_leer"
    assert cfg.sgs == "lasd"                    # --dynamic ⇒ LASD
    g = pin.make_grid(cfg, dtype=jnp.float64)   # raises if cfg invalid
    assert g.z_c.shape == (cfg.nz,)
    # weno5 maps through; weno5_hv has no pin analogue → hard exit, never a
    # silent reconstruction swap.
    assert drv._pin_config(_args(scalar_advection="weno5")).scheme == "weno5"
    with pytest.raises(SystemExit):
        drv._pin_config(_args(scalar_advection="weno5_hv"))


def test_pin_core_steps_moist_column_to_finite_state():
    """A real pin.step + Morrison adapter advances a synthetic moist column."""
    args = _args()
    dtype = jnp.float64
    cfg = _load_driver()._pin_config(args)
    cfg = cfg._replace(sfc_theta_flux=8.0e-3, sfc_qv_flux=5.0e-5)
    g = pin.make_grid(cfg, dtype=dtype)

    z_c = np.asarray(g.z_c)
    # Slightly stable θ with a moist, near-saturated sub-cloud layer.
    th_prof = 298.0 + 4.0e-3 * z_c
    qv_prof = np.maximum(0.016 - 1.0e-5 * z_c, 2.0e-3)
    ref = make_anelastic_reference(z_c, np.asarray(g.z_f), 1.015e5,
                                   th_prof, qv_prof, dtype=dtype)

    ny, nx, nz = args.ny, args.nx, args.nz
    key = jax.random.PRNGKey(0)
    seed = (jnp.asarray(z_c) < 600.0).astype(dtype)
    th3 = (jnp.broadcast_to(jnp.asarray(th_prof, dtype), (ny, nx, nz))
           + 0.1 * jax.random.normal(key, (ny, nx, nz), dtype) * seed)
    u3 = jnp.zeros((ny, nx, nz), dtype)
    v3 = jnp.zeros((ny, nx, nz), dtype)
    w3 = jnp.zeros((ny, nx, nz + 1), dtype)
    tr = jnp.zeros((ny, nx, nz, args.n_tracers), dtype).at[..., 0].set(
        jnp.asarray(qv_prof, dtype)[None, None, :])
    pi0 = jnp.zeros((ny, nx, nz), dtype)
    u3, v3, w3, _ = pin.project(u3, v3, w3, th3, tr, pi0, 1.0, g)
    st = pin.PseudoIncompressibleState(
        u=u3, v=v3, w=w3, theta=th3, pi_prev=pi0, tracers=tr)

    micro = make_les_microphysics_fn(
        MicrophysicsConfig(scheme="morrison",
                           morrison=MorrisonConfig(morrison_flavor="sam")),
        ref, g.dz, 1.0)

    dt = jnp.asarray(1.0, dtype)
    for _ in range(6):
        st = pin.step(st, g, dt, None)
        dth, dtr, _pr = micro(st.theta, st.tracers)
        tr2 = jnp.maximum(st.tracers + dt * dtr, 0.0)
        st = st._replace(theta=st.theta + dt * dth, tracers=tr2)

    assert np.all(np.isfinite(np.asarray(st.theta)))
    assert np.all(np.isfinite(np.asarray(st.w)))
    assert np.all(np.asarray(st.tracers[..., :6]) >= 0.0)   # water non-negative
    assert float(jnp.max(jnp.abs(st.w))) < 50.0             # not blowing up


def test_promote_micro_f64_runs_inner_in_f64_and_restores_dtype():
    """--mixed-micro wrapper: inner micro sees float64, output matches state f32."""
    drv = _load_driver()
    seen = {}

    def fake_micro(theta, tracers):
        seen["theta"] = theta.dtype
        seen["tracers"] = tracers.dtype
        # tendencies in the (f64) input dtype, as a real adapter would return
        return theta * 0.0, tracers * 0.0, jnp.zeros(theta.shape[:-1], theta.dtype)

    fake_micro.scheme_name = "fake"
    wrapped = drv._promote_micro_f64(fake_micro)
    th = jnp.ones((2, 2, 3), jnp.float32)
    tr = jnp.ones((2, 2, 3, 6), jnp.float32)
    dth, dtr, pr = wrapped(th, tr)

    assert seen["theta"] == jnp.float64 and seen["tracers"] == jnp.float64
    assert dth.dtype == jnp.float32 and dtr.dtype == jnp.float32
    assert pr.dtype == jnp.float32
    assert wrapped.scheme_name == "fake"          # attr preserved for the header


def test_mixed_micro_rejects_f32_combo():
    """--mixed-micro + --f32 is contradictory (x64 disabled) → hard exit."""
    import subprocess
    out = subprocess.run(
        [sys.executable, str(_DRIVER), "--mixed-micro", "--f32",
         "--case-dir", "/nonexistent"],
        capture_output=True, text=True)
    assert out.returncode != 0
    assert "mutually exclusive" in (out.stdout + out.stderr)
