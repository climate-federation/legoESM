"""Conservation of the plane-LES surface-coupling column operator.

The implicit vertical-diffusion column (`_implicit_vertical_diffusion`, the
surface-flux-only coupling that replaced the over-mixing PBL column) must:

  1. CONSERVE the column inventory Σ φ_k dz_k under no surface forcing
     (pure redistribution, no-flux top+bottom);
  2. inject EXACTLY ``flux_sfc · dt`` into the inventory for a prescribed
     surface flux (so the BL heat/moisture budget closes);
  3. remove momentum monotonically under surface drag (a sink, not a source).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
_ROOT = Path(__file__).resolve().parents[2]


def _load_driver():
    path = _ROOT / "scripts" / "run" / "run_les_plane.py"
    spec = importlib.util.spec_from_file_location("run_les_plane", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(n=40):
    dz = np.full(n, 10.0)                       # uniform cell thickness [m]
    dzc = np.full(n - 1, 10.0)                  # interface spacing
    K = np.linspace(0.5, 5.0, n - 1)            # interior diffusivity
    rng = np.random.default_rng(0)
    phi = jnp.asarray(rng.normal(size=n) + 280.0)
    return jnp.asarray(dz), jnp.asarray(dzc), jnp.asarray(K), phi


def test_no_flux_conserves_inventory():
    m = _load_driver()
    dz, dzc, K, phi = _setup()
    new = m._implicit_vertical_diffusion(phi, K, dz, dzc, dt=2.0)
    inv0 = float(jnp.sum(phi * dz))
    inv1 = float(jnp.sum(new * dz))
    assert abs(inv1 - inv0) / abs(inv0) < 1e-12   # pure redistribution


def test_surface_flux_changes_inventory_by_flux_dt():
    m = _load_driver()
    dz, dzc, K, phi = _setup()
    F, dt = 0.3, 5.0
    new = m._implicit_vertical_diffusion(phi, K, dz, dzc, dt=dt, flux_sfc=F)
    d_inv = float(jnp.sum((new - phi) * dz))
    assert abs(d_inv - F * dt) < 1e-9             # exact surface-flux budget


def test_drag_removes_momentum():
    m = _load_driver()
    dz, dzc, K, _ = _setup()
    u = jnp.full(40, 8.0)
    new = m._implicit_vertical_diffusion(u, K, dz, dzc, dt=5.0, drag_sfc=0.05)
    inv0, inv1 = float(jnp.sum(u * dz)), float(jnp.sum(new * dz))
    assert inv1 < inv0                            # drag is a sink
    assert float(new[0]) < float(u[0])           # near-surface deceleration
