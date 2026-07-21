"""Exponential moving average (EMA) of model weights for training.

Standard SOTA practice for autoregressive weather emulators: train the raw
weights, evaluate/rollout the EMA weights (U-Cast arXiv:2604.09041 uses
decay 0.9999; GenCast-family diffusion forecasters likewise). See
docs/superpowers/specs/2026-07-19-unified-wb-aimip-training-design.md (D3).

Pure pytree math on the inexact-array leaves only — static leaves (configs,
activation choices) are taken from the EMA model unchanged, so the EMA model
stays a drop-in replacement wherever the raw model is used. Safe inside
``jax.jit`` (no Python side effects).
"""

from __future__ import annotations

import equinox as eqx
import jax

# U-Cast / GenCast convention; ~1/(1-decay) = 10k-step averaging horizon.
EMA_DECAY = 0.9999


def init_ema(model):
    """EMA state at training start: an exact copy of the model.

    JAX arrays are immutable, so returning the model itself is a correct
    (aliasing-free) initial EMA.
    """
    return model


def ema_update(ema_model, model, decay: float = EMA_DECAY):
    """One EMA step: ``ema <- decay * ema + (1 - decay) * model``.

    Blends only the inexact-array leaves (trainable weights); every other
    leaf (ints, static config, activation callables) is kept from
    ``ema_model``. ``ema_model`` and ``model`` must share a pytree
    structure (they always do: the EMA starts as a copy).
    """
    if not (0.0 <= decay < 1.0):
        raise ValueError(f"EMA decay must be in [0, 1); got {decay}.")

    def _blend(e, m):
        if e is None:
            return None
        return e * decay + (1.0 - decay) * m

    e_params, e_static = eqx.partition(ema_model, eqx.is_inexact_array)
    m_params = eqx.filter(model, eqx.is_inexact_array)
    blended = jax.tree_util.tree_map(
        _blend, e_params, m_params, is_leaf=lambda x: x is None
    )
    return eqx.combine(blended, e_static)
