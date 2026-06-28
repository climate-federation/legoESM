"""Differentiable 3D-CNN (U-Net) emulator of the mc3d Monte-Carlo shortwave
transport.

Goal: a smooth, fully-differentiable surrogate for the (non-differentiable,
expensive) Monte-Carlo 3D radiative transport. It maps a 3D optical field +
solar geometry to the per-cell absorbed-flux fraction (the MC tracer's
``abs_flux`` normalised by incident), so ``jax.grad`` flows end-to-end for
training / inversion where the photon MC cannot. This is a standalone CNN
emulator (NOT a neural operator in the SFNO sense, and NOT in the core
SFNO+dycore training path); it learns the MC transport operator.

I/O (channels-first for ``eqx.nn.Conv3d``):
  input  (C_in, nx, ny, nz): per-cell [k_ext_norm, ssa, g, rayleigh_frac,
          r_eff_norm] + broadcast solar [mu0, sin_zenith*cos_az, sin_zenith*sin_az]
  output (nx, ny, nz): absorbed-flux fraction per cell. The head is LINEAR (a
  squashing non-negativity like softplus drove a dead zone -> exact-zero
  collapse on the sparse target); clip to >= 0 for physical use via
  ``predict_nonneg``.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

N_OPTICAL_CHANNELS = 5   # k_ext_norm, ssa, g, rayleigh_frac, r_eff_norm
N_SOLAR_CHANNELS = 3     # mu0, sx, sy (beam direction-ish)
N_IN_CHANNELS = N_OPTICAL_CHANNELS + N_SOLAR_CHANNELS


class EmulatorConfig(NamedTuple):
  """3D U-Net emulator hyperparameters."""

  base_features: int = 16   # channels at the finest level
  depth: int = 2            # number of down/up-sampling stages
  kernel: int = 3


def _norm_k_ext(k_ext):
  """Compress the wide extinction range to ~O(1): log1p of optical-depth-ish."""
  return jnp.log1p(jnp.maximum(k_ext, 0.0) * 1.0e3)


def pack_inputs(k_ext, ssa, g, rayleigh_frac, r_eff_um, mu0, azimuth):
  """Build the (C_in, nx, ny, nz) emulator input from optical fields + sun.

  Fields are ``(nx, ny, nz)``; ``mu0``/``azimuth`` are scalars. r_eff is
  normalised to ~[0,1] over the 2.5..21.5 um LUT span.
  """
  shape = k_ext.shape
  sin_z = jnp.sqrt(jnp.maximum(1.0 - mu0 * mu0, 0.0))
  solar = jnp.stack([
      jnp.full(shape, mu0),
      jnp.full(shape, sin_z * jnp.cos(azimuth)),
      jnp.full(shape, sin_z * jnp.sin(azimuth)),
  ])
  optical = jnp.stack([
      _norm_k_ext(k_ext),
      jnp.clip(ssa, 0.0, 1.0),
      jnp.clip(g, -1.0, 1.0),
      jnp.clip(rayleigh_frac, 0.0, 1.0),
      jnp.clip((r_eff_um - 2.5) / 19.0, 0.0, 1.0),  # coeff-ok: microhh Mie LUT r_eff norm [um] (2.5..21.5 span 19)
  ])
  return jnp.concatenate([optical, solar], axis=0)   # (C_in, nx, ny, nz)


class _ConvBlock(eqx.Module):
  c1: eqx.nn.Conv3d
  c2: eqx.nn.Conv3d

  def __init__(self, c_in, c_out, kernel, key):
    k1, k2 = jax.random.split(key)
    pad = kernel // 2
    self.c1 = eqx.nn.Conv3d(c_in, c_out, kernel, padding=pad, key=k1)
    self.c2 = eqx.nn.Conv3d(c_out, c_out, kernel, padding=pad, key=k2)

  def __call__(self, x):
    return jax.nn.gelu(self.c2(jax.nn.gelu(self.c1(x))))


class UNet3D(eqx.Module):
  """Small 3D U-Net: encoder (conv + 2x avgpool) / decoder (trilinear up + skip)
  with a LINEAR head (clip >=0 via predict_nonneg; a squashing head collapsed to
  zero on the sparse target)."""

  inc: _ConvBlock
  downs: list
  ups: list
  head: eqx.nn.Conv3d
  depth: int = eqx.field(static=True)

  def __init__(self, config: EmulatorConfig, key, c_in: int = N_IN_CHANNELS):
    if config.kernel % 2 == 0:
      raise ValueError(
          f"EmulatorConfig.kernel must be ODD for shape-preserving convolutions;"
          f" got {config.kernel}.")
    if config.depth < 0:
      raise ValueError(f"EmulatorConfig.depth must be >= 0; got {config.depth}.")
    # NOTE: each spatial dim must be divisible by 2**depth (the avgpool stages);
    # the caller is responsible for a compatible grid (e.g. 16,16,16 @ depth<=4).
    self.depth = config.depth
    keys = jax.random.split(key, 2 * config.depth + 2)
    f = config.base_features
    self.inc = _ConvBlock(c_in, f, config.kernel, keys[0])
    downs, ups = [], []
    feats = [f]
    for d in range(config.depth):
      downs.append(_ConvBlock(feats[-1], feats[-1] * 2, config.kernel,
                              keys[1 + d]))
      feats.append(feats[-1] * 2)
    for d in range(config.depth):
      # decoder block takes upsampled + skip (concatenated) channels.
      c_up = feats[-1 - d]
      c_skip = feats[-2 - d]
      ups.append(_ConvBlock(c_up + c_skip, c_skip, config.kernel,
                            keys[1 + config.depth + d]))
    self.downs = downs
    self.ups = ups
    self.head = eqx.nn.Conv3d(f, 1, 1, key=keys[-1])

  def __call__(self, x):
    skips = []
    h = self.inc(x)
    for blk in self.downs:
      skips.append(h)
      h = blk(_avgpool2(h))
    for d, blk in enumerate(self.ups):
      h = _upsample2(h, skips[-1 - d].shape[1:])
      h = blk(jnp.concatenate([h, skips[-1 - d]], axis=0))
    return self.head(h)[0]   # (nx,ny,nz) LINEAR (clip >=0 via predict_nonneg)

  def predict_nonneg(self, x):
    """Forward + clip to the physical non-negative absorbed-flux fraction."""
    return jnp.maximum(self(x), 0.0)


def _avgpool2(x):
  """2x average pool over the 3 spatial dims (channels-first)."""
  return eqx.nn.AvgPool3d(kernel_size=2, stride=2)(x)


def _upsample2(x, target_shape):
  """Trilinear-ish upsample (jax.image.resize) to ``target_shape`` (nx,ny,nz)."""
  c = x.shape[0]
  return jax.image.resize(x, (c, *target_shape), method="linear")
