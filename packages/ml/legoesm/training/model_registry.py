"""One place that builds the trainable model for a legoESM training variant.

The WeatherBench and AIMIP campaigns train the SAME four model variants; what
differs between the campaigns is the training strategy and the loss, never the
model.  Before this module they each constructed their own, and the
architecture defaults had drifted: the WB lane built an ``sfno_physics`` at
embed 256 / 8 blocks where AIMIP built 128 / 4, and the column network's width
was read from ``neural_gcm.nn_hidden`` on one side and ``nn_hidden_dim`` on the
other.  Both lanes now call :func:`build_variant`, so "same variant" means the
same pytree.

No model code lives here.  This module only selects and constructs the classes
that already exist (``TrainablePhysicsParams``, ``NeuralPhysics``, ``SFNO``),
with one set of defaults and one override vocabulary.

Defaults are AIMIP's, which makes routing ``run_aimip`` through this module a
no-op; every WB config in ``config/wb/`` pins its SFNO size explicitly, so the
choice does not move any existing run either way (verified 2026-08-11).
"""
from __future__ import annotations

from typing import Any

VALID_VARIANTS = ("classical", "column_nn", "sfno_physics", "sfno_full")

# Architecture defaults, per variant, in the ONE vocabulary both campaigns use.
# Provenance: run_aimip.py's `cfg.get(...)` defaults (the active campaign).
# `mlp_expansion` 4 and `dropout` 0.0 additionally match SFNOConfig's own
# class defaults, which is what the WB lane got by not passing them.
VARIANT_DEFAULTS: dict[str, dict[str, Any]] = {
    "classical": {},
    # `residual_scale` is present but NULL: the sentinel means "pass nothing",
    # so `build_column_physics` applies its own value (0.01) — what BOTH lanes
    # get today. Restating the number here would be a second source of truth.
    "column_nn": {"nn_hidden_dim": 256, "n_layers": 4, "residual_scale": None},
    "sfno_physics": {
        "sfno_embed_dim": 128,
        "sfno_n_blocks": 4,
        "sfno_mlp_expansion": 4,
    },
    "sfno_full": {
        "sfno_embed_dim": 128,
        "sfno_n_blocks": 4,
        "sfno_mlp_expansion": 4,
        # MC-Dropout is the U-Cast ensembling source and the CRPS finetune
        # requires it > 0; only the full emulator wires it today.
        "sfno_dropout": 0.0,
        "sfno_history_steps": 0,
    },
}

# Decoder zero-init is a PHYSICS policy, not a tunable, so it is not an
# override key: an untrained ``sfno_physics`` must emit exactly-zero tendencies
# so epoch 0 integrates the pure dycore.  ``sfno_full`` REPLACES the dycore, so
# there is nothing to fall back to and a zeroed decoder would emit a zero state.
_ZERO_INIT_DECODER = {"sfno_physics": True, "sfno_full": False}


def _resolve(variant: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Merge ``overrides`` onto the variant defaults, rejecting unknown keys.

    An unknown key is a hard error rather than a silent no-op: a mistyped
    ``sfno_embed_dm`` that is quietly ignored trains a different-sized model
    than the config says it does.
    """
    resolved = dict(VARIANT_DEFAULTS[variant])
    for key, value in (overrides or {}).items():
        if key not in resolved:
            raise ValueError(
                f"unknown override {key!r} for variant {variant!r}; "
                f"valid keys: {sorted(resolved) or '(none)'}")
        resolved[key] = value
    return resolved


def sfno_arch_config(variant: str, *, nlev: int,
                     overrides: dict[str, Any] | None = None):
    """The ``SFNOConfig`` :func:`build_variant` would use for an SFNO variant.

    Exposed because the full-emulator caller needs the same architecture
    description a second time, to build ``SFNOPrimitiveEquationConfig``.
    Reading it from here keeps the emulator wrapper and the weights it wraps
    from ever disagreeing on channel counts.
    """
    if variant not in ("sfno_physics", "sfno_full"):
        raise ValueError(
            f"sfno_arch_config is for the SFNO variants, got {variant!r}")
    kw = _resolve(variant, overrides)

    from legoesm.ml.channel_packing import PE3DChannelSpec
    from legoesm.ml.sfno import SFNOConfig

    spec = PE3DChannelSpec(nlev=nlev)
    if variant == "sfno_physics":
        # The surface-forcing planes (T_sfc, sic, insolation) ride the INPUT
        # side only; the physics emits tendencies on the state channels.
        # Source of truth for the count is neural_gcm_spectral, imported here
        # rather than copied so the two cannot drift (deferred: that module
        # pulls in the whole spectral stack).
        from legoesm.training.neural_gcm_spectral import (
            N_SFNO_FORCING_CHANNELS,
        )
        in_channels = spec.n_channels + N_SFNO_FORCING_CHANNELS
        extra = {}
    else:  # sfno_full: the emulator sees (1 + history) copies of the state.
        in_channels = spec.n_channels * (1 + int(kw["sfno_history_steps"]))
        extra = {"dropout": float(kw["sfno_dropout"])}

    return SFNOConfig(
        in_channels=in_channels,
        out_channels=spec.n_channels,
        embed_dim=int(kw["sfno_embed_dim"]),
        n_blocks=int(kw["sfno_n_blocks"]),
        mlp_expansion=int(kw["sfno_mlp_expansion"]),
        # Tendencies (or a full next state), NOT residuals on top of the
        # input: SFNOConfig defaults this True, which would add the input
        # state to the output and destroy the rollout in one step.
        residual_prediction=False,
        **extra,
    )


def build_variant(variant: str, *, nlev: int, grid=None, seed: int = 0,
                  overrides: dict[str, Any] | None = None):
    """Build the trainable pytree for one training variant.

    Parameters
    ----------
    variant : str
        One of :data:`VALID_VARIANTS`.  Unknown -> ``ValueError`` (never a
        silent default: a typo must not train a different model).
    nlev : int
        Vertical levels.  Sets the column network's input size and the packed
        channel count of both SFNO variants.
    grid : GaussianGrid, optional
        Spectral grid the SFNO is built on.  Required by ``sfno_physics`` and
        ``sfno_full``; unused by the other two.
    seed : int, default 0
        PRNG seed for weight initialisation.
    overrides : dict, optional
        Architecture overrides from the suite/campaign YAML.  Valid keys per
        variant are exactly the keys of that variant's
        :data:`VARIANT_DEFAULTS` entry.

    Returns
    -------
    equinox.Module
        ``TrainablePhysicsParams`` (classical), ``NeuralPhysics``
        (column_nn), or ``SFNO`` (both sfno variants).
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r}; valid: {VALID_VARIANTS}")
    kw = _resolve(variant, overrides)

    if variant == "classical":
        from legoesm.training.trainable_params import TrainablePhysicsParams
        return TrainablePhysicsParams.from_defaults()

    import jax

    if variant == "column_nn":
        from legoesm.atmosphere.physics.neural_physics import (
            build_column_physics,
        )
        extra_cn = ({} if kw["residual_scale"] is None
                    else {"residual_scale": float(kw["residual_scale"])})
        return build_column_physics(
            nlev=nlev,
            hidden_dim=int(kw["nn_hidden_dim"]),
            n_layers=int(kw["n_layers"]),
            key=jax.random.PRNGKey(seed),
            **extra_cn,
        )

    if grid is None:
        raise ValueError(f"variant {variant!r} needs a spectral `grid`")

    import equinox as eqx
    import jax.numpy as jnp

    from legoesm.ml.sfno import SFNO

    sfno = SFNO(
        sfno_arch_config(variant, nlev=nlev, overrides=overrides),
        grid,
        key=jax.random.PRNGKey(seed),
    )

    if _ZERO_INIT_DECODER[variant]:
        sfno = eqx.tree_at(
            lambda m: (m.decoder.weight, m.decoder.bias), sfno,
            (jnp.zeros_like(sfno.decoder.weight),
             jnp.zeros_like(sfno.decoder.bias)))
    return sfno
