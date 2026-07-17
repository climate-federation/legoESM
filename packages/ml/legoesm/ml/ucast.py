"""U-Cast: a convolutional U-Net atmosphere emulator (ADM / DhariwalUNet).

JAX/Equinox port of the network used by **U-Cast** ("A Surprisingly Simple
and Efficient Frontier Probabilistic AI Weather Forecaster", Rose-STL-Lab,
ICML 2026; https://github.com/Rose-STL-Lab/u-cast).  Unlike the spherical
spectral :class:`~legoesm.ml.sfno.SFNO`, U-Cast is a plain convolutional
U-Net in the lineage of Dhariwal & Nichol's ADM / Karras et al.'s EDM
backbone — *no* spherical-harmonic transforms.  Its identity as a
*probabilistic* forecaster comes from **MC-Dropout** ensembling plus a
fair-CRPS fine-tuning stage, both layered on top of this one deterministic
backbone (see :mod:`legoesm.atmosphere.dynamics.neural.ucast_pe`).

This module ports only the network ``DhariwalUNet`` from the reference
``src/models/networks_edm.py``.  The architecture, in order:

1. **Encoder** — a stem 3x3 conv lifts the input channels to
   ``model_channels``; then, per resolution ``level`` in ``channel_mult``,
   a downsampling :class:`UNetBlock` (``level > 0``) followed by
   ``num_blocks`` residual :class:`UNetBlock` s.  Self-attention is enabled
   on levels in ``attn_levels``.
2. **Bottleneck** — two :class:`UNetBlock` s at the coarsest level, the
   first with attention.
3. **Decoder** — per level (coarse → fine) an upsampling block then
   ``num_blocks + 1`` blocks, each concatenating the corresponding encoder
   skip.
4. **Head** — GroupNorm → SiLU → zero-initialised 3x3 conv.

Faithful details preserved from the reference:

* **Geographic padding.**  The reference monkey-patches ``conv2d`` to pad
  *circularly along the periodic axis* and zero-pad the bounded axis
  (``circular_height_only`` in ``src/models/modules/padding.py``).  legoESM
  packs fields as ``(n_lat, n_lon, channels)`` (see
  :mod:`legoesm.ml.channel_packing`), so the periodic axis is **longitude**
  (the last spatial axis).  :class:`GeoConv2d` therefore wraps longitude
  circularly and zero-pads latitude — the physically correct choice and the
  exact analogue of the reference's intent.
* **Weight init.**  The ADM custom initialisation (kaiming-uniform with
  ``init_weight = sqrt(1/3)`` and matching bias scale; zero-init for the
  second conv of every block, the attention projection, and the output
  conv) is reproduced so that, at initialisation, every residual/attention
  branch and the head contribute zero — the network starts as the identity
  map under ``residual_prediction`` (mirroring the SFNO contract).
* **Attention.**  Scaled dot-product multi-head self-attention over the
  spatial grid, ``num_heads = out_channels // channels_per_head`` (0 → no
  attention, matching the reference's integer floor).

Operates per-sample on ``(n_lat, n_lon, in_channels)`` → ``(n_lat, n_lon,
out_channels)``, matching :class:`legoesm.ml.sfno.SFNO`'s call convention so
the two are drop-in interchangeable inside the dycore wrapper.  Ensembling is
obtained by ``jax.vmap`` over per-member dropout keys at the wrapper level.

References
----------
- U-Cast (Rose-STL-Lab, ICML 2026): https://github.com/Rose-STL-Lab/u-cast
- Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis.
  NeurIPS (ADM architecture).
- Karras et al. (2022). Elucidating the Design Space of Diffusion-Based
  Generative Models. NeurIPS (EDM; the reference's ``networks_edm``).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx


# ============================================================================
# Configuration
# ============================================================================

class UCastConfig(NamedTuple):
    """Configuration for the U-Cast (DhariwalUNet) backbone.

    Attributes
    ----------
    in_channels : int
        Number of input channels (state variables packed into the last axis).
    out_channels : int
        Number of predicted output channels.
    model_channels : int
        Base channel width; level ``i`` uses ``model_channels * channel_mult[i]``.
        The paper's ERA5 1.5deg config uses 320.
    channel_mult : tuple[int, ...]
        Per-resolution channel multipliers.  Length = number of U-Net levels
        (each level beyond the first halves the spatial resolution).  Paper: (1, 2, 3, 4).
    num_blocks : int
        Residual blocks per encoder level (decoder uses ``num_blocks + 1``).
        Paper: 4.
    attn_levels : tuple[int, ...]
        Resolution levels (0 = finest) at which self-attention is enabled.
        Paper: (2, 3).  The coarsest bottleneck block always uses attention.
    channels_per_head : int
        Channels per attention head; ``num_heads = out_channels // channels_per_head``
        (0 disables attention for that block, matching the reference floor).
    dropout : float
        Dropout probability.  Acts as a regulariser at training time and as the
        sole stochasticity source for MC-Dropout ensembling at inference.
    residual_prediction : bool
        If True, predict residuals: ``output = input[..., :out_channels] + net(input)``.
        The zero-initialised head then makes the untrained network an identity map
        (mirrors :class:`legoesm.ml.sfno.SFNO`).
    num_conditional_channels : int
        Number of extra conditioning channels concatenated onto the input
        (static + forcing fields).  0 for the bare emulator.
    """
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 3, 4)
    num_blocks: int = 2
    attn_levels: tuple[int, ...] = (2, 3)
    channels_per_head: int = 64
    dropout: float = 0.1
    residual_prediction: bool = True
    num_conditional_channels: int = 0


# ============================================================================
# Weight initialisation (port of networks_edm.weight_init)
# ============================================================================

def _weight_init(
    shape: tuple[int, ...],
    *,
    mode: str,
    fan_in: float,
    fan_out: float,
    key: jax.Array,
) -> jnp.ndarray:
    """Port of the reference ``weight_init`` (kaiming/xavier uniform/normal)."""
    if mode == "xavier_uniform":
        return math.sqrt(6 / (fan_in + fan_out)) * (
            jax.random.uniform(key, shape) * 2 - 1
        )
    if mode == "xavier_normal":
        return math.sqrt(2 / (fan_in + fan_out)) * jax.random.normal(key, shape)
    if mode == "kaiming_uniform":
        return math.sqrt(3 / fan_in) * (jax.random.uniform(key, shape) * 2 - 1)
    if mode == "kaiming_normal":
        return math.sqrt(1 / fan_in) * jax.random.normal(key, shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ============================================================================
# Geographic convolution (circular longitude, zero-padded latitude)
# ============================================================================

class GeoConv2d(eqx.Module):
    """3x3 (or 1x1) conv with optional 2x up/down-sampling and geo-padding.

    Channel-first ``(C, H, W)`` operation, where ``H`` is latitude (bounded,
    zero-padded) and ``W`` is longitude (periodic, circularly wrapped).  A
    ``kernel == 0`` layer carries *no* weights and acts as a pure resampler
    (used by residual skips that only need to change resolution) — faithfully
    reproducing the reference ``Conv2d(kernel=0, up/down=...)`` behaviour.

    The reference applies up/down-sampling *before* the convolution; this is
    reproduced exactly so the stride semantics match.
    """
    weight: jnp.ndarray | None
    bias: jnp.ndarray | None
    up: bool = eqx.field(static=True)
    down: bool = eqx.field(static=True)
    pad: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        *,
        up: bool = False,
        down: bool = False,
        init_mode: str = "kaiming_normal",
        init_weight: float = 1.0,
        init_bias: float = 0.0,
        key: jax.Array,
    ):
        assert not (up and down)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.pad = kernel // 2

        if kernel:
            fan_in = in_channels * kernel * kernel
            fan_out = out_channels * kernel * kernel
            k_w, k_b = jax.random.split(key)
            self.weight = _weight_init(
                (out_channels, in_channels, kernel, kernel),
                mode=init_mode, fan_in=fan_in, fan_out=fan_out, key=k_w,
            ) * init_weight
            self.bias = _weight_init(
                (out_channels,),
                mode=init_mode, fan_in=fan_in, fan_out=fan_out, key=k_b,
            ) * init_bias
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply to a channel-first ``(C, H, W)`` field."""
        if self.up:
            c, h, w = x.shape
            x = jax.image.resize(x, (c, h * 2, w * 2), method="nearest")
        if self.down:
            # avg_pool 2x2 stride 2, VALID (drops a trailing odd row/col,
            # matching torch ``F.avg_pool2d(kernel_size=2)`` floor semantics).
            x = jax.lax.reduce_window(
                x, 0.0, jax.lax.add,
                window_dimensions=(1, 2, 2),
                window_strides=(1, 2, 2),
                padding="VALID",
            ) * 0.25

        if self.weight is not None:
            if self.pad > 0:
                # Latitude (axis -2): zero pad (bounded domain).
                # Longitude (axis -1): circular wrap (periodic).
                x = jnp.pad(
                    x, ((0, 0), (self.pad, self.pad), (0, 0)), mode="constant",
                )
                x = jnp.pad(
                    x, ((0, 0), (0, 0), (self.pad, self.pad)), mode="wrap",
                )
            # Cast weights to the running dtype (the reference does
            # ``w = self.weight.to(x.dtype)``); ``conv_general_dilated``
            # rejects mixed dtypes, which would otherwise bite under x64.
            x = jax.lax.conv_general_dilated(
                x[None],                       # (1, C, H, W)
                self.weight.astype(x.dtype),   # (O, I, kh, kw)
                window_strides=(1, 1),
                padding="VALID",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )[0]
        if self.bias is not None:
            x = x + self.bias.astype(x.dtype)[:, None, None]
        return x


# ============================================================================
# U-Net residual block (port of networks_edm.UNetBlock)
# ============================================================================

class GroupNorm(eqx.Module):
    """Group normalisation (port of ``networks_edm.GroupNorm``).

    Group count follows the reference: ``min(num_groups, C // min_channels_per_group)``
    with ``num_groups=32``, ``min_channels_per_group=4``.  The affine
    parameters are cast to the input dtype at use (the reference does
    ``weight.to(x.dtype)``), keeping the whole network single-dtype even when
    parameters were created under x64.  Operates on channel-first ``(C, H, W)``.
    """
    weight: jnp.ndarray
    bias: jnp.ndarray
    num_groups: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        num_groups: int = 32,
        min_channels_per_group: int = 4,
        eps: float = 1e-5,
    ):
        groups = min(num_groups, channels // min_channels_per_group)
        groups = max(groups, 1)
        if channels % groups != 0:
            raise ValueError(
                f"GroupNorm channel count {channels} is not divisible by the "
                f"derived group count {groups} (= min({num_groups}, "
                f"{channels}//{min_channels_per_group})). Choose model_channels "
                f"/ channel_mult so every block's channel count is divisible."
            )
        self.num_groups = groups
        self.eps = eps
        self.weight = jnp.ones(channels)
        self.bias = jnp.zeros(channels)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        c, h, w = x.shape
        xr = x.reshape(self.num_groups, c // self.num_groups, h, w)
        mean = jnp.mean(xr, axis=(1, 2, 3), keepdims=True)
        var = jnp.var(xr, axis=(1, 2, 3), keepdims=True)
        xr = (xr - mean) * jax.lax.rsqrt(var + self.eps)
        xn = xr.reshape(c, h, w)
        wt = self.weight.astype(x.dtype)[:, None, None]
        bs = self.bias.astype(x.dtype)[:, None, None]
        return xn * wt + bs


class UNetBlock(eqx.Module):
    """Residual block: GN→SiLU→conv→GN→SiLU→dropout→conv (+ skip), optional attention.

    Faithful to the ADM ``UNetBlock``: the second conv and the attention
    projection are zero-initialised, ``skip_scale`` defaults to 1, and
    attention (when enabled) is applied *after* the residual add.
    """
    norm0: GroupNorm
    conv0: GeoConv2d
    norm1: GroupNorm
    conv1: GeoConv2d
    dropout: eqx.nn.Dropout
    skip: GeoConv2d | None
    norm2: GroupNorm | None
    qkv: GeoConv2d | None
    proj: GeoConv2d | None
    num_heads: int = eqx.field(static=True)
    skip_scale: float = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        resample_proj: bool = False,
        key: jax.Array,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale
        self.num_heads = (
            0 if not attention else out_channels // channels_per_head
        )
        # The attention reshape ``(num_heads, c2, 3, HW)`` requires
        # ``out_channels`` to split evenly across heads.  The reference relies
        # on this implicitly; guard it so a bad (out_channels, channels_per_head)
        # pairing fails fast at construction rather than at the reshape.
        if self.num_heads and out_channels % self.num_heads != 0:
            raise ValueError(
                f"Attention requires out_channels ({out_channels}) divisible by "
                f"num_heads ({self.num_heads} = {out_channels}//{channels_per_head}). "
                f"Adjust channels_per_head or the channel widths."
            )

        keys = jax.random.split(key, 5)

        # Main path. conv0 uses the ADM "init"; conv1 is zero-initialised.
        self.norm0 = GroupNorm(in_channels)
        self.conv0 = GeoConv2d(
            in_channels, out_channels, 3, up=up, down=down,
            init_mode="kaiming_uniform", init_weight=math.sqrt(1 / 3),
            init_bias=math.sqrt(1 / 3), key=keys[0],
        )
        self.norm1 = GroupNorm(out_channels)
        self.conv1 = GeoConv2d(
            out_channels, out_channels, 3,
            init_mode="kaiming_uniform", init_weight=0.0, init_bias=0.0,
            key=keys[1],
        )
        self.dropout = eqx.nn.Dropout(p=dropout)

        # Residual skip: a projection/resampler only when shape changes.
        if out_channels != in_channels or up or down:
            kernel = 1 if (resample_proj or out_channels != in_channels) else 0
            self.skip = GeoConv2d(
                in_channels, out_channels, kernel, up=up, down=down,
                init_mode="kaiming_uniform", init_weight=math.sqrt(1 / 3),
                init_bias=math.sqrt(1 / 3), key=keys[2],
            )
        else:
            self.skip = None

        # Self-attention branch (zero-initialised projection).
        if self.num_heads:
            self.norm2 = GroupNorm(out_channels)
            self.qkv = GeoConv2d(
                out_channels, out_channels * 3, 1,
                init_mode="kaiming_uniform", init_weight=math.sqrt(1 / 3),
                init_bias=math.sqrt(1 / 3), key=keys[3],
            )
            self.proj = GeoConv2d(
                out_channels, out_channels, 1,
                init_mode="kaiming_uniform", init_weight=0.0, init_bias=0.0,
                key=keys[4],
            )
        else:
            self.norm2 = None
            self.qkv = None
            self.proj = None

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        key: jax.Array | None = None,
        inference: bool = True,
    ) -> jnp.ndarray:
        """Apply to channel-first ``(C, H, W)``."""
        orig = x
        x = self.conv0(jax.nn.silu(self.norm0(x)))
        x = self.dropout(
            jax.nn.silu(self.norm1(x)), key=key, inference=inference,
        )
        x = self.conv1(x)
        x = x + (self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            c, h, w = x.shape
            c2 = c // self.num_heads
            qkv = self.qkv(self.norm2(x))                  # (3C, H, W)
            qkv = qkv.reshape(self.num_heads, c2, 3, h * w)
            q = qkv[:, :, 0]                               # (nh, c2, HW)
            k = qkv[:, :, 1]
            v = qkv[:, :, 2]
            scale = 1.0 / math.sqrt(c2)
            attn = jnp.einsum("ncq,nck->nqk", q, k * scale)  # (nh, HW, HW)
            attn = jax.nn.softmax(attn, axis=2)
            a = jnp.einsum("nqk,nck->ncq", attn, v)          # (nh, c2, HW)
            a = a.reshape(c, h, w)
            x = x + self.proj(a)
            x = x * self.skip_scale

        return x


# ============================================================================
# Full U-Cast network (port of networks_edm.DhariwalUNet)
# ============================================================================

class UCast(eqx.Module):
    """U-Cast convolutional U-Net atmosphere emulator.

    Parameters
    ----------
    config : UCastConfig
        Model configuration.
    key : jax.random.PRNGKey
        Random key for initialisation.

    Notes
    -----
    The block sequence (``enc``/``dec`` lists) is built statically; the
    decoder consumes encoder skips whenever the running channel count differs
    from a block's declared ``in_channels`` (exactly the reference forward
    rule), with bilinear interpolation reconciling any spatial-size mismatch
    from odd grid dimensions.
    """
    enc: list
    dec: list
    out_norm: GroupNorm
    out_conv: GeoConv2d
    config: UCastConfig = eqx.field(static=True)
    _n_dropout: int = eqx.field(static=True)

    def __init__(self, config: UCastConfig, *, key: jax.Array):
        self.config = config
        in_channels = config.in_channels + config.num_conditional_channels
        out_channels = config.out_channels

        block_kwargs = dict(
            channels_per_head=config.channels_per_head,
            dropout=config.dropout,
        )
        init_kwargs = dict(
            init_mode="kaiming_uniform", init_weight=math.sqrt(1 / 3),
            init_bias=math.sqrt(1 / 3),
        )

        enc: list = []
        dec: list = []

        # --- Encoder ---
        cout = in_channels
        for level, mult in enumerate(config.channel_mult):
            key, k = jax.random.split(key)
            if level == 0:
                cin, cout = cout, config.model_channels * mult
                enc.append(GeoConv2d(cin, cout, 3, key=k, **init_kwargs))
            else:
                enc.append(UNetBlock(cout, cout, down=True, key=k, **block_kwargs))
            for _ in range(config.num_blocks):
                key, k = jax.random.split(key)
                cin, cout = cout, config.model_channels * mult
                enc.append(UNetBlock(
                    cin, cout, attention=(level in config.attn_levels),
                    key=k, **block_kwargs,
                ))
        skips = [block.out_channels for block in enc]

        # --- Decoder ---
        n_levels = len(config.channel_mult)
        for level, mult in reversed(list(enumerate(config.channel_mult))):
            if level == n_levels - 1:
                key, k0 = jax.random.split(key)
                key, k1 = jax.random.split(key)
                dec.append(UNetBlock(cout, cout, attention=True, key=k0, **block_kwargs))
                dec.append(UNetBlock(cout, cout, key=k1, **block_kwargs))
            else:
                key, k = jax.random.split(key)
                dec.append(UNetBlock(cout, cout, up=True, key=k, **block_kwargs))
            for _ in range(config.num_blocks + 1):
                key, k = jax.random.split(key)
                cin, cout = cout + skips.pop(), config.model_channels * mult
                dec.append(UNetBlock(
                    cin, cout, attention=(level in config.attn_levels),
                    key=k, **block_kwargs,
                ))

        self.enc = enc
        self.dec = dec

        key, k = jax.random.split(key)
        self.out_norm = GroupNorm(cout)
        self.out_conv = GeoConv2d(
            cout, out_channels, 3,
            init_mode="kaiming_uniform", init_weight=0.0, init_bias=0.0, key=k,
        )

        # Every UNetBlock owns one dropout; the stem GeoConv2d owns none.
        self._n_dropout = sum(isinstance(b, UNetBlock) for b in (enc + dec))

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        condition: jnp.ndarray | None = None,
        key: jax.Array | None = None,
    ) -> jnp.ndarray:
        """Forward pass.

        Parameters
        ----------
        x : array, shape (n_lat, n_lon, in_channels)
            Input field on the grid (channel-last, matching channel packing).
        condition : array, shape (n_lat, n_lon, num_conditional_channels), optional
            Static / forcing conditioning fields concatenated onto the input
            channels (the reference's ``concat_condition_if_needed``).  Must be
            supplied iff ``config.num_conditional_channels > 0``.
        key : jax.random.PRNGKey, optional
            When ``None`` (default), dropout runs in **inference** mode (no
            stochasticity) — the deterministic forecast.  When a key is
            supplied, dropout is **active** (MC-Dropout): distinct keys yield
            distinct ensemble members.  ``jax.vmap`` over keys gives a batched
            ensemble.

        Returns
        -------
        array, shape (n_lat, n_lon, out_channels)
        """
        x_in = x
        inference = key is None

        # Concatenate conditioning fields when the model was built with them
        # (faithful to the reference ``concat_condition_if_needed``).
        n_cond = self.config.num_conditional_channels
        if x.shape[-1] != self.config.in_channels:
            raise ValueError(
                f"UCast expected {self.config.in_channels} input channels, "
                f"got {x.shape[-1]}."
            )
        if n_cond > 0:
            if condition is None:
                raise ValueError(
                    f"num_conditional_channels={n_cond} but no condition was "
                    f"passed to UCast.__call__."
                )
            if condition.shape[-1] != n_cond:
                raise ValueError(
                    f"condition has {condition.shape[-1]} channels, expected "
                    f"num_conditional_channels={n_cond}."
                )
            if condition.shape[:2] != x.shape[:2]:
                raise ValueError(
                    f"condition spatial shape {condition.shape[:2]} does not "
                    f"match input {x.shape[:2]}."
                )
            x = jnp.concatenate([x, condition], axis=-1)
        elif condition is not None:
            raise ValueError(
                "A condition was passed but num_conditional_channels=0."
            )

        # Channel-last (H, W, C) -> channel-first (C, H, W).
        x = jnp.transpose(x, (2, 0, 1))

        # Per-block dropout keys (only consumed when key is not None).
        if key is None:
            subkeys = [None] * self._n_dropout
        else:
            subkeys = list(jax.random.split(key, self._n_dropout))
        ki = 0

        skips = []
        for block in self.enc:
            if isinstance(block, UNetBlock):
                x = block(x, key=subkeys[ki], inference=inference)
                ki += 1
            else:
                x = block(x)
            skips.append(x)

        for block in self.dec:
            if x.shape[0] != block.in_channels:
                skip = skips.pop()
                if skip.shape[-2:] != x.shape[-2:]:
                    x = jax.image.resize(
                        x, (x.shape[0], skip.shape[-2], skip.shape[-1]),
                        method="bilinear",
                    )
                x = jnp.concatenate([x, skip], axis=0)
            x = block(x, key=subkeys[ki], inference=inference)
            ki += 1

        x = self.out_conv(jax.nn.silu(self.out_norm(x)))

        # Channel-first (C, H, W) -> channel-last (H, W, C).
        x = jnp.transpose(x, (1, 2, 0))

        if self.config.residual_prediction:
            n_out = self.config.out_channels
            x = x + x_in[..., :n_out]

        return x
