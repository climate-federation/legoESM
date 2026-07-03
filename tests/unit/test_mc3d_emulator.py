"""Tests for the differentiable 3D-CNN emulator of the mc3d MC SW transport."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.radiation.mc3d.emulator import (
    EmulatorConfig,
    N_IN_CHANNELS,
    UNet3D,
    pack_inputs,
)


def _fields(nx=16, ny=16, nz=16):
  k_ext = jnp.full((nx, ny, nz), 5e-3)
  return (k_ext, jnp.full_like(k_ext, 0.9), jnp.full_like(k_ext, 0.85),
          jnp.full_like(k_ext, 0.1), jnp.full_like(k_ext, 10.5))


def test_pack_inputs_shape():
  k_ext, ssa, g, rf, reff = _fields()
  x = pack_inputs(k_ext, ssa, g, rf, reff, 0.7, 0.3)
  assert x.shape == (N_IN_CHANNELS, 16, 16, 16)
  assert bool(jnp.all(jnp.isfinite(x)))


def test_unet_forward_nonneg_and_shape():
  net = UNet3D(EmulatorConfig(base_features=8, depth=2), jax.random.PRNGKey(0))
  k_ext, ssa, g, rf, reff = _fields()
  x = pack_inputs(k_ext, ssa, g, rf, reff, 0.7, 0.0)
  y = net(x)
  assert y.shape == (16, 16, 16)            # linear head
  assert float(jnp.min(net.predict_nonneg(x))) >= 0.0   # clipped for physics


def test_unet_differentiable():
  net = UNet3D(EmulatorConfig(base_features=8, depth=2), jax.random.PRNGKey(0))
  k_ext, ssa, g, rf, reff = _fields()
  x = pack_inputs(k_ext, ssa, g, rf, reff, 0.7, 0.0)
  grads = eqx.filter_grad(lambda m: jnp.mean(m(x)))(net)
  leaves = [l for l in jax.tree_util.tree_leaves(grads) if eqx.is_array(l)]
  assert leaves and all(bool(jnp.all(jnp.isfinite(l))) for l in leaves)


def test_emulator_training_reduces_loss():
  """Overfit a few synthetic samples: training MSE must drop substantially."""
  import importlib.util
  import pathlib
  path = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "experiment" / "train_mc3d_emulator.py")
  spec = importlib.util.spec_from_file_location("train_mc3d_emulator", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)

  # Synthetic dataset: target = smooth function of the optical field (learnable).
  rng = np.random.default_rng(0)
  n, nx, ny, nz = 12, 16, 16, 16
  ks, ss, gs, rs, es, mu, az, tg = [], [], [], [], [], [], [], []
  for _ in range(n):
    k = rng.uniform(1e-4, 2e-2, (nx, ny, nz))
    ks.append(k); ss.append(np.full_like(k, 0.9)); gs.append(np.full_like(k, 0.8))
    rs.append(np.full_like(k, 0.1)); es.append(np.full_like(k, 10.0))
    mu.append(0.7); az.append(0.0)
    tg.append(1.0 - np.exp(-k * 1e3 * 0.05))   # smooth, optical-depth-like
  inputs = jnp.stack([pack_inputs(jnp.asarray(ks[i]), jnp.asarray(ss[i]),
                                  jnp.asarray(gs[i]), jnp.asarray(rs[i]),
                                  jnp.asarray(es[i]), mu[i], az[i])
                      for i in range(n)])
  targets = jnp.asarray(np.stack(tg))
  _model, history, _ = mod.train(
      inputs, targets, config=EmulatorConfig(base_features=8, depth=2),
      epochs=40, batch_size=3, lr=1e-3, val_frac=0.25, seed=0)
  # Clear, non-trivial reduction (the surrogate is learning the mapping).
  assert history[-1]["train_mse"] < 0.7 * history[0]["train_mse"]
  assert all(np.isfinite(h["val_mse"]) for h in history)
