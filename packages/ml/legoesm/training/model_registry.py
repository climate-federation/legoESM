"""One place that builds the trainable model for a legoESM training variant.

The WeatherBench and AIMIP campaigns train the SAME model variants; what
differs between the campaigns is the training strategy and the loss, never the
model.  Before this module they each constructed their own, and the
architecture defaults had drifted (the column network's width was read from
``neural_gcm.nn_hidden`` on one side and ``nn_hidden_dim`` on the other).

Both lanes now call :func:`build_variant`, so "same variant" means the
same pytree.

The SFNO-inside-the-dynamical-core variant was removed (user 2026-11-06): an
SFNO replaces the whole atmosphere (``sfno_full``) or is not used.  The last
commit with it is tagged ``pre-dycore-sfno-removal``.

CLASSICAL_LEGACY_NOTE: before 2026-08-12 the WB lane's "classical" was a
different, smaller model — ``TrainablePhysicsParams``, 6 knobs, gray radiation,
and only two parameterization families (convection + radiation). It still
exists for ``train_physics_params_spectral`` (an idealized-physics trainer),
but it is NOT what "classical" means to a campaign, and its numbers are not a
baseline for the current one.

No model code lives here.  This module only selects and constructs the classes
that already exist (``TrainablePhysicsParams``, ``NeuralPhysics``, ``SFNO``),
with one set of defaults and one override vocabulary.

Defaults are AIMIP's, which makes routing ``run_aimip`` through this module a
no-op.
"""
from __future__ import annotations

from typing import Any

VALID_VARIANTS = ("classical", "column_nn", "sfno_full")

# Architecture defaults, per variant, in the ONE vocabulary both campaigns use.
# Provenance: run_aimip.py's `cfg.get(...)` defaults (the active campaign).
# `mlp_expansion` 4 and `dropout` 0.0 additionally match SFNOConfig's own
# class defaults, which is what the WB lane got by not passing them.
VARIANT_DEFAULTS: dict[str, dict[str, Any]] = {
    "classical": {
        # The trainable 2-D surface field (roughness / albedo / exchange
        # coefficients) is off unless a suite asks for it.
        "spatial_surface": False,
        "spatial_init_std": 0.0,
        "spatial_seed": 0,
        # Knob names that train; every other knob is a plain float, outside
        # the differentiated tree (AIMIPClassicalParams.from_defaults).  None
        # = every knob an array (checkpoint templates, tests).
        "trainable": None,
    },
    # `residual_scale` is present but NULL: the sentinel means "pass nothing",
    # so `build_column_physics` applies its own value (0.01) — what BOTH lanes
    # get today. Restating the number here would be a second source of truth.
    # ``spatial_embedding``: NeuralGCM-style static features + learned
    # per-grid-point embedding (column_nn).  ``era5_surface_fluxes``: the six ERA5 surface-flux planes
    # as extra inputs.  Both OFF here so AIMIP is unchanged; the WB campaign
    # decks set them explicitly.
    # ``learned_position_embedding`` (with spatial_embedding): the learned
    # per-location map on (True, the NeuralGCM recipe) or off (the ablation).
    # Weights the training loss cannot reach are not created (user
    # 2026-11-06).  column_nn ``kept_outputs``: the output groups to create
    # (``losses.column_outputs_the_loss_reads`` of the run's loss config);
    # None = all.
    "column_nn": {"nn_hidden_dim": 256, "n_layers": 4, "residual_scale": None,
                  "spatial_embedding": False, "era5_surface_fluxes": False,
                  "learned_position_embedding": True,
                  "kept_outputs": None},
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
    if variant != "sfno_full":
        raise ValueError(
            f"sfno_arch_config is for the SFNO variant, got {variant!r}")
    kw = _resolve(variant, overrides)

    from legoesm.ml.channel_packing import PE3DChannelSpec
    from legoesm.ml.sfno import SFNOConfig

    spec = PE3DChannelSpec(nlev=nlev)
    # The emulator sees (1 + history) copies of the state.
    in_channels = spec.n_channels * (1 + int(kw["sfno_history_steps"]))
    return SFNOConfig(
        in_channels=in_channels,
        out_channels=spec.n_channels,
        embed_dim=int(kw["sfno_embed_dim"]),
        n_blocks=int(kw["sfno_n_blocks"]),
        mlp_expansion=int(kw["sfno_mlp_expansion"]),
        # A full next state, NOT a residual on top of the input:
        # SFNOConfig defaults this True, which would add the input state to
        # the output and destroy the rollout in one step.
        residual_prediction=False,
        dropout=float(kw["sfno_dropout"]),
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
        channel count of the SFNO.
    grid : GaussianGrid, optional
        Spectral grid the SFNO is built on.  Required by ``sfno_full``;
        ``column_nn`` needs it only with ``spatial_embedding``.
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
        (column_nn), or ``SFNO`` (sfno_full).
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r}; valid: {VALID_VARIANTS}")
    kw = _resolve(variant, overrides)

    if variant == "classical":
        # ONE classical model for both campaigns (owner, 2026-08-12:
        # "classical means the classical physics-based parameterization",
        # matrix {WB, AIMIP} x {classical, column_nn, sfno}). That is the
        # six-family scheme-parameter set, not the 6-knob idealized set the WB
        # lane used to train — see CLASSICAL_LEGACY_NOTE below.
        from legoesm.training.aimip_params import AIMIPClassicalParams
        return AIMIPClassicalParams.from_defaults(
            spatial_surface=bool(kw["spatial_surface"]),
            spatial_init_std=float(kw["spatial_init_std"]),
            spatial_seed=int(kw["spatial_seed"]),
            trainable=kw["trainable"],
        )

    import jax

    if variant == "column_nn":
        from legoesm.atmosphere.physics.neural_physics import (
            build_column_physics,
        )
        extra_cn = ({} if kw["residual_scale"] is None
                    else {"residual_scale": float(kw["residual_scale"])})
        _spatial = bool(kw["spatial_embedding"])
        if _spatial:
            # The learned per-grid-point embedding table is sized by the grid.
            if grid is None:
                raise ValueError(
                    "variant 'column_nn' with spatial_embedding=True needs a "
                    "spectral `grid` (it sizes the learned per-grid-point "
                    "embedding table)")
            extra_cn["n_columns"] = int(grid.n_lat) * int(grid.n_lon)
        return build_column_physics(
            nlev=nlev,
            hidden_dim=int(kw["nn_hidden_dim"]),
            n_layers=int(kw["n_layers"]),
            key=jax.random.PRNGKey(seed),
            spatial_embedding=_spatial,
            era5_surface_fluxes=bool(kw["era5_surface_fluxes"]),
            learned_embedding=bool(kw["learned_position_embedding"]),
            kept_outputs=kw["kept_outputs"],
            **extra_cn,
        )

    if grid is None:
        raise ValueError(f"variant {variant!r} needs a spectral `grid`")

    from legoesm.ml.sfno import SFNO

    return SFNO(
        sfno_arch_config(variant, nlev=nlev, overrides=overrides),
        grid,
        key=jax.random.PRNGKey(seed),
    )
