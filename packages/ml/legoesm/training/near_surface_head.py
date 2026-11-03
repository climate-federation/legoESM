"""Learned 2 m temperature / 10 m wind head shared by the three WB arms.

Owner decisions 2026-11-01..03 (item 4/11 of the surface-skill brief): a
per-column MLP, two hidden layers of 64, on the lowest three model levels of
T, u, v, q, the surface pressure, the land fraction, the surface geopotential,
sin/cos latitude, sin/cos local time of the lead's valid time and a learned
per-location embedding (zero-initialised).  It outputs a correction to the
eval's current estimate (``evaluations.wb_forecast.diagnose_headline_fields``):
t2m = lowest-level T + d_t2m, u10/v10 = neutral log-law 10 m wind + d_u/d_v,
and the 10 m speed is recomputed from the corrected components.  Inputs are
standardised and outputs scaled by the 2015-2019 statistics of the scales job
(``scripts/data/build_wb_scale_stats.py``), copied into the deck.  The last
layer starts at zero, so an untrained head reproduces the current estimate
exactly.  It replaces that estimate in the loss and the eval and trains
jointly with each arm (for CAM6 its gradient reaches the physics).
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# Input order (the scales job's ``head_inputs`` keys); the per-level fields
# carry the lowest N_HEAD_LEVELS levels, top to bottom.
N_HEAD_LEVELS = 3
LEVEL_INPUTS = ("T", "u", "v", "q_v")
PLANE_INPUTS = ("p_s", "land_frac", "phis", "sin_lat", "cos_lat",
                "sin_local_time", "cos_local_time")
OUTPUTS = ("t2m", "u10", "v10")
N_INPUTS = N_HEAD_LEVELS * len(LEVEL_INPUTS) + len(PLANE_INPUTS)
HIDDEN = (64, 64)


def _norms(head_inputs, head_outputs):
    """``(mean, std, out_scale)`` float tuples in input order, from the deck's
    copy of the scales job's ``head_inputs`` / ``head_outputs`` blocks."""
    mean, std = [], []
    for k in LEVEL_INPUTS:
        m, s = head_inputs[k]["mean"], head_inputs[k]["std"]
        if len(m) != N_HEAD_LEVELS or len(s) != N_HEAD_LEVELS:
            raise ValueError(f"head input {k}: need {N_HEAD_LEVELS} levels")
        mean += [float(v) for v in m]
        std += [float(v) for v in s]
    for k in PLANE_INPUTS:
        mean.append(float(head_inputs[k]["mean"]))
        std.append(float(head_inputs[k]["std"]))
    out = tuple(float(head_outputs[f"{k}_gap_std"]) for k in OUTPUTS)
    if not all(np.isfinite(v) and v > 0 for v in std + list(out)):
        raise ValueError("near-surface head scales must be finite and > 0")
    return tuple(mean), tuple(std), out


class NearSurfaceHead(eqx.Module):
    """Per-column MLP correction to the 2 m / 10 m estimate (see module)."""

    layers: list
    pos_embed: jax.Array | None     # (n_columns, n_embed); None if n_embed 0
    in_mean: tuple = eqx.field(static=True)
    in_std: tuple = eqx.field(static=True)
    out_scale: tuple = eqx.field(static=True)

    def __init__(self, n_columns, head_inputs, head_outputs, *, key,
                 n_embed=None):
        from legoesm.atmosphere.physics.neural_physics import N_COLUMN_POS_EMBED

        n_embed = N_COLUMN_POS_EMBED if n_embed is None else int(n_embed)
        self.in_mean, self.in_std, self.out_scale = _norms(head_inputs,
                                                           head_outputs)
        dims = [N_INPUTS + n_embed, *HIDDEN, len(OUTPUTS)]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
                       for i in range(len(dims) - 1)]
        last = self.layers[-1]
        self.layers[-1] = eqx.tree_at(
            lambda l: (l.weight, l.bias), last,
            (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)))
        self.pos_embed = (jnp.zeros((int(n_columns), n_embed),
                                    self.layers[0].weight.dtype)
                          if n_embed else None)

    def _column(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)

    def __call__(self, feats):
        """``feats`` (n_columns, N_INPUTS) physical inputs -> (n_columns, 3)
        physical corrections (K, m/s, m/s)."""
        dtype = self.layers[0].weight.dtype
        z = ((feats - jnp.asarray(self.in_mean, feats.dtype))
             / jnp.asarray(self.in_std, feats.dtype)).astype(dtype)
        if self.pos_embed is not None:
            z = jnp.concatenate([z, self.pos_embed], axis=-1)
        return jax.vmap(self._column)(z) * jnp.asarray(self.out_scale, dtype)


def head_features(T, u, v, q, p_s, phis, land_frac, lat, lon, seconds_utc):
    """``(n_lat * n_lon, N_INPUTS)`` head inputs in physical units.

    ``T, u, v, q`` (n_lat, n_lon, nlev) gridded state; ``p_s, phis,
    land_frac`` (n_lat, n_lon); ``lat, lon`` (n_lat,), (n_lon,) radians;
    ``seconds_utc`` the valid time's UTC seconds of day (local time angle =
    2 pi seconds_utc / 86400 + lon, the scales job's convention)."""
    n_lat, n_lon = p_s.shape
    lat2 = jnp.broadcast_to(lat[:, None], (n_lat, n_lon))
    ang = 2.0 * jnp.pi * seconds_utc / 86400.0 + jnp.broadcast_to(
        lon[None, :], (n_lat, n_lon))
    lev = [f[..., -N_HEAD_LEVELS:] for f in (T, u, v, q)]
    planes = [p_s, jnp.reshape(land_frac, (n_lat, n_lon)), phis,
              jnp.sin(lat2), jnp.cos(lat2), jnp.sin(ang), jnp.cos(ang)]
    x = jnp.concatenate(lev + [p[..., None] for p in planes], axis=-1)
    return x.reshape(n_lat * n_lon, N_INPUTS)


class WithHead(eqx.Module):
    """An arm's trainable parameters plus its near-surface head, trained
    jointly: ``core`` is what the arm's rollout reads, ``head`` the diagnosis."""

    core: object
    head: NearSurfaceHead


HEAD_DECK_KEYS = frozenset({"seed", "inputs", "outputs"})


def build_near_surface_head(yml, grid):
    """The deck's head (``near_surface_head`` block: ``seed``, ``inputs`` and
    ``outputs`` copied from the scales job), or None without the block.  The
    learned per-location embedding is on iff the top-level
    ``learned_position_embedding`` is true (it must be stated)."""
    blk = yml.get("near_surface_head")
    if blk is None:
        return None
    if not isinstance(blk, dict):
        raise ValueError("near_surface_head must be a block (seed, inputs, "
                         f"outputs) or absent, got {blk!r}")
    unknown = set(blk) - HEAD_DECK_KEYS
    missing = HEAD_DECK_KEYS - set(blk)
    if unknown or missing:
        raise ValueError(f"near_surface_head block: unknown {sorted(unknown)}, "
                         f"missing {sorted(missing)}")
    if not any(float((yml.get("loss") or {}).get(f"w_{k}", 0.0)) > 0.0
               for k in OUTPUTS):
        raise ValueError("near_surface_head without a positive w_t2m, w_u10 or "
                         "w_v10 would train nothing: every head leaf would be "
                         "inert (mslp does not read the head).")
    lpe = yml.get("learned_position_embedding")
    if not isinstance(lpe, bool):
        raise ValueError("near_surface_head needs learned_position_embedding "
                         f"stated in the deck as true or false, got {lpe!r}")
    from legoesm.atmosphere.physics.neural_physics import N_COLUMN_POS_EMBED

    n_embed = N_COLUMN_POS_EMBED if lpe else 0
    return NearSurfaceHead(int(grid.n_lat) * int(grid.n_lon), blk["inputs"],
                           blk["outputs"], key=jax.random.PRNGKey(int(blk["seed"])),
                           n_embed=n_embed)


# The head starts with a zero last layer, so at the initial parameters every
# other head leaf has an exactly-zero gradient (W_last^T delta = 0).  The
# reachability probe therefore measures the head with this perturbed last
# layer (fixed key, std below): a leaf it still cannot reach is truly inert.
PROBE_LAST_LAYER_STD = 1e-2


def probe_view(params):
    """``params`` with a WithHead's last head layer perturbed (see above);
    any other tree unchanged."""
    if not isinstance(params, WithHead):
        return params
    last = params.head.layers[-1]
    k_w, k_b = jax.random.split(jax.random.PRNGKey(0))
    w = PROBE_LAST_LAYER_STD * jax.random.normal(k_w, last.weight.shape,
                                                 last.weight.dtype)
    b = PROBE_LAST_LAYER_STD * jax.random.normal(k_b, last.bias.shape,
                                                 last.bias.dtype)
    return eqx.tree_at(lambda p: (p.head.layers[-1].weight,
                                  p.head.layers[-1].bias), params,
                       (last.weight + w, last.bias + b))
