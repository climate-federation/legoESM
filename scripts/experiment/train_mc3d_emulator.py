"""Train the differentiable 3D-CNN emulator of the mc3d Monte-Carlo SW transport.

Loads a dataset from ``gen_mc3d_emulator_data.py`` (optical fields + MC
absorbed-flux targets), trains the ``UNet3D`` emulator with the repo-standard
warmup+cosine+clip optimizer (``ml.training.create_optimizer``), and reports
train/val MSE + relative error vs the Monte-Carlo target. The trained emulator
is a smooth, ``jax.grad``-able surrogate for the (non-differentiable) photon MC.

Usage:
    JAX_ENABLE_X64=1 python scripts/experiment/train_mc3d_emulator.py \
        --data scripts/tmp/_mc3d_emu.npz --epochs 50 --out scripts/tmp/_emu.eqx
"""

from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# NN training runs fine in fp32 (the default) and fp32 is required on backends
# without x64 (e.g. Apple MPS / many GPUs). Honor JAX_ENABLE_X64 from the env
# (set it to 1 for an fp64 CPU run); do NOT force x64 here.

from legoesm.atmosphere.physics.radiation.mc3d.emulator import (
    EmulatorConfig,
    UNet3D,
    pack_inputs,
)
from legoesm.ml.training import TrainingConfig, create_optimizer


def load_dataset(path):
  """Load npz -> (inputs (N,C,nx,ny,nz), targets (N,nx,ny,nz))."""
  d = np.load(path)
  n = d["abs_frac"].shape[0]

  def _pack(i):
    return pack_inputs(jnp.asarray(d["k_ext"][i]), jnp.asarray(d["ssa"][i]),
                       jnp.asarray(d["g"][i]), jnp.asarray(d["rayleigh_frac"][i]),
                       jnp.asarray(d["r_eff"][i]), float(d["mu0"][i]),
                       float(d["azimuth"][i]))

  inputs = jnp.stack([_pack(i) for i in range(n)])
  targets = jnp.asarray(d["abs_frac"])
  return inputs, targets


def _batched_apply(model, x_batch):
  """Apply the model over a batch WITHOUT vmap. The Apple MPS plugin (when
  installed) monkeypatches several ops, e.g. jax.nn.gelu, with versions that have
  no batching rule -- so vmap over the model raises NotImplementedError even on
  CPU. The batch is small and its size is static under jit, so this loop unrolls
  cheaply and is correct on every backend."""
  return jnp.stack([model(x_batch[i]) for i in range(x_batch.shape[0])])


def _loss(model, x_batch, y_batch, weight_alpha=0.0):
  """Magnitude-weighted MSE. Absorbed-flux targets are very sparse (most cells
  ~0, few cloud cells high), so plain MSE (weight_alpha=0) collapses to the
  trivial zero predictor. weight = 1 + weight_alpha * y/mean(y) upweights
  high-absorption cells (so clouds must be fit) while keeping background weight
  1 (so false positives are still penalized)."""
  pred = _batched_apply(model, x_batch)
  w = 1.0 + weight_alpha * y_batch / (jnp.mean(y_batch) + 1e-6)
  return jnp.sum(w * (pred - y_batch) ** 2) / jnp.sum(w)


def train(inputs, targets, *, config: EmulatorConfig, epochs: int,
          batch_size: int = 4, lr: float = 5e-4, val_frac: float = 0.2,
          seed: int = 0, weight_alpha: float = 0.0):
  """Train the emulator; return (model, history) with train/val MSE + rel err."""
  key = jax.random.PRNGKey(seed)
  n = inputs.shape[0]
  if n < 2:
    raise ValueError(f"need >= 2 samples for a train/val split; got {n}.")
  if epochs < 1:
    raise ValueError(f"epochs must be >= 1; got {epochs}.")
  # Keep at least one training sample (never let the val split consume all).
  n_val = min(max(1, int(n * val_frac)), n - 1)
  perm = np.random.default_rng(seed).permutation(n)
  vi, ti = perm[:n_val], perm[n_val:]
  x_tr, y_tr = inputs[ti], targets[ti]
  x_val, y_val = inputs[vi], targets[vi]

  key, mkey = jax.random.split(key)
  model = UNet3D(config, mkey)
  steps_per_epoch = max(1, len(ti) // batch_size)
  total_steps = epochs * steps_per_epoch
  opt = create_optimizer(TrainingConfig(
      lr=lr, warmup_steps=max(1, total_steps // 10), total_steps=total_steps,
      grad_clip_norm=1.0))
  opt_state = opt.init(eqx.filter(model, eqx.is_array))

  @eqx.filter_jit
  def step(model, opt_state, xb, yb):
    loss, grads = eqx.filter_value_and_grad(
        lambda m, a, b: _loss(m, a, b, weight_alpha))(model, xb, yb)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

  def _rel_err(model, x, y):
    pred = _batched_apply(model, x)
    return float(jnp.linalg.norm(pred - y) / (jnp.linalg.norm(y) + 1e-30))

  history = []
  for ep in range(epochs):
    order = np.random.default_rng(seed + ep + 1).permutation(len(ti))
    ep_loss = 0.0
    for b in range(steps_per_epoch):
      idx = order[b * batch_size:(b + 1) * batch_size]
      model, opt_state, loss = step(model, opt_state, x_tr[idx], y_tr[idx])
      ep_loss += float(loss)
    val_mse = float(_loss(model, x_val, y_val, weight_alpha))  # same metric as train
    history.append({"epoch": ep, "train_mse": ep_loss / steps_per_epoch,
                    "val_mse": val_mse, "val_rel": _rel_err(model, x_val, y_val)})
  return model, history, (x_val, y_val)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--data", default="scripts/tmp/_mc3d_emu.npz")
  ap.add_argument("--epochs", type=int, default=50)
  ap.add_argument("--base-features", type=int, default=16)
  ap.add_argument("--depth", type=int, default=2)
  ap.add_argument("--batch-size", type=int, default=4)
  ap.add_argument("--weight-alpha", type=float, default=20.0,
                  help="magnitude weighting for the sparse absorbed-flux target "
                       "(0 = plain MSE)")
  ap.add_argument("--out", default="scripts/tmp/_mc3d_emu_model.eqx")
  args = ap.parse_args(argv)

  inputs, targets = load_dataset(args.data)
  print(f"dataset: inputs {inputs.shape}, targets {targets.shape}")
  cfg = EmulatorConfig(base_features=args.base_features, depth=args.depth)
  model, history, _ = train(inputs, targets, config=cfg, epochs=args.epochs,
                            batch_size=args.batch_size,
                            weight_alpha=args.weight_alpha)
  h0, hN = history[0], history[-1]
  print(f"epoch 0  : train {h0['train_mse']:.3e}  val {h0['val_mse']:.3e}  "
        f"rel {h0['val_rel']:.3f}")
  print(f"epoch {hN['epoch']:<3}: train {hN['train_mse']:.3e}  "
        f"val {hN['val_mse']:.3e}  rel {hN['val_rel']:.3f}")
  eqx.tree_serialise_leaves(args.out, model)
  print(f"saved emulator -> {args.out}")


if __name__ == "__main__":
  main()
