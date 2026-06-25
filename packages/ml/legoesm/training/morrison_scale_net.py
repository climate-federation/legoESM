"""State-conditioned warm-rain scaling network for Morrison microphysics.

A small pointwise MLP that maps each grid cell's local thermodynamic state
``(T, q_v, q_c, q_r, rho)`` to per-level multipliers on Morrison's warm-rain
process rates (autoconversion / accretion / rain-evaporation), returned as a
:class:`~legoesm.atmosphere.physics.microphysics.morrison.WarmRainRateScales`.

This is the trainable "local scaling factor" closure: the LES (which resolves
the subgrid variability a single column cannot) is the reference, and the net
learns the per-state correction that makes column-mean Morrison reproduce the
LES-mean microphysics. It is plugged into Morrison via
``MorrisonConfig.warm_rain_scale_fn`` (config-pytree injection inside the loss,
the SegmentForcing doctrine) — production leaves the field ``None``.

Design choices:
  * POINTWISE (no vertical coupling) ⇒ genuinely "local" (depends on local
    conditions, not on level index) and parameter-light.
  * IDENTITY INITIALISATION: the final layer is zero-initialised so an untrained
    net returns exactly 1.0 everywhere ⇒ wrapping baseline Morrison with a fresh
    net is bit-identical to baseline until training moves the weights.
  * BOUNDED OUTPUT: ``scale = exp(log_bound · tanh(raw)) ∈ [1/B, B]`` keeps the
    multiplier strictly positive and within a physical band (default B = 10).
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.microphysics.morrison import WarmRainRateScales

# Fixed input-normalisation (loc/scale) so the MLP sees O(1) features — these
# are conditioning constants for the network input, NOT physical parameters.
# Order: T [K], q_v, q_c, q_r [kg/kg], rho [kg/m^3].
_FEATURE_LOC = jnp.array([280.0, 5.0e-3, 3.0e-4, 1.0e-4, 1.0])
_FEATURE_SCALE = jnp.array([30.0, 5.0e-3, 5.0e-4, 2.0e-4, 0.5])
_N_FEATURES = 5
_N_RATES = 3  # autoconv, accretion, rain_evap


class MorrisonWarmRainScaleNet(eqx.Module):
    """Pointwise MLP ``(T, q_v, q_c, q_r, rho) -> WarmRainRateScales``."""

    mlp: eqx.nn.MLP
    log_bound: float = eqx.field(static=True)

    def __init__(self, key, *, width: int = 16, depth: int = 2,
                 bound: float = 10.0):
        self.log_bound = math.log(bound)
        mlp = eqx.nn.MLP(
            in_size=_N_FEATURES, out_size=_N_RATES, width_size=width,
            depth=depth, activation=jax.nn.tanh, key=key,
        )
        # Identity init: zero the final Linear layer (weight + bias) so the raw
        # output is 0 -> tanh(0)=0 -> exp(0)=1 -> identity scaling at init.
        last = mlp.layers[-1]
        last = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias), last,
            (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)),
        )
        self.mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)

    def __call__(self, T, q_v, q_c, q_r, rho) -> WarmRainRateScales:
        feats = jnp.stack(
            jnp.broadcast_arrays(T, q_v, q_c, q_r, rho), axis=-1,
        )  # (..., 5)
        feats = (feats - _FEATURE_LOC) / _FEATURE_SCALE
        lead = feats.shape[:-1]
        raw = jax.vmap(self.mlp)(feats.reshape(-1, _N_FEATURES))
        raw = raw.reshape(lead + (_N_RATES,))
        scale = jnp.exp(self.log_bound * jnp.tanh(raw))
        return WarmRainRateScales(
            autoconv=scale[..., 0],
            accretion=scale[..., 1],
            rain_evap=scale[..., 2],
        )
