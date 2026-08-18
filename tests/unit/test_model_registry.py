"""One registry builds every training variant, identically for both campaigns.

The defect this guards: the WeatherBench lane and the AIMIP lane each
constructed their own models and the architecture defaults drifted apart
(``sfno_physics`` at embed 256 / 8 blocks on one side, 128 / 4 on the other),
so "the same variant" silently meant two different networks. The decisive test
is :func:`test_wb_and_aimip_build_the_same_pytree`, which builds each variant
through the WeatherBench entry point and through the registry the AIMIP
trainers call, and compares tree structure and every leaf shape.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

from legoesm.training.model_registry import (  # noqa: E402
    VALID_VARIANTS,
    VARIANT_DEFAULTS,
    build_variant,
    sfno_arch_config,
)

_NLEV = 4
_N_MAX = 8


@pytest.fixture(scope="module")
def grid():
    from legoesm.grids.gaussian import create_gaussian_grid
    return create_gaussian_grid(_N_MAX, dealiasing="quadratic")


def _shapes(tree):
    return [np.shape(x) for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def _dtypes(tree):
    return [np.asarray(x).dtype
            for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


# --------------------------------------------------------------------------
# Dispatch hardening: a typo must never silently train a different model.
# --------------------------------------------------------------------------

def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown variant"):
        build_variant("sfno", nlev=_NLEV)


def test_unknown_override_key_raises(grid):
    # 'sfno_embed_dm' is the realistic typo: silently ignored, it trains a
    # differently sized network than the config claims.
    with pytest.raises(ValueError, match="unknown override"):
        build_variant("sfno_full", nlev=_NLEV, grid=grid,
                      overrides={"sfno_embed_dm": 32})
    # A key valid for one variant is not valid for another.
    with pytest.raises(ValueError, match="unknown override"):
        build_variant("column_nn", nlev=_NLEV,
                      overrides={"sfno_embed_dim": 32})


def test_sfno_arch_config_rejects_non_sfno_variants():
    with pytest.raises(ValueError, match="SFNO variants"):
        sfno_arch_config("column_nn", nlev=_NLEV)


# --------------------------------------------------------------------------
# The requirement: same variant -> same model, whichever campaign asks.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode,variant,over", [
    ("physics", "classical", {}),
    ("neural_gcm", "column_nn", {"nn_hidden_dim": 32, "n_layers": 2}),
    ("sfno", "sfno_physics", {"sfno_embed_dim": 16, "sfno_n_blocks": 2}),
])
def test_wb_and_aimip_build_the_same_pytree(mode, variant, over, grid):
    """The WB builder and the registry (what the AIMIP trainers call) agree.

    Compared on tree structure and every leaf shape — the properties that make
    a checkpoint interchangeable — not on weight values, which legitimately
    depend on the seed.
    """
    from legoesm.training.scale_build import _build_mode_components_spectral

    yml = {
        "n_lat": 2 * _N_MAX, "nlev": _NLEV,
        "spectral": {"n_max": _N_MAX, "dt": 1800.0},
        "loss": {},
        "neural_gcm": {"nn_hidden": over.get("nn_hidden_dim", 256),
                       "nn_layers": over.get("n_layers", 4)},
        "sfno": {k: v for k, v in over.items() if k.startswith("sfno_")},
    }
    cfg = SimpleNamespace(mode=mode, training_core="spectral", smoke=False,
                          multi_step_hours=(6,))
    _model, _grid, _sigma, wb_params, _seg, _loss, _dt = (
        _build_mode_components_spectral(cfg, yml))

    aimip_params = build_variant(variant, nlev=_NLEV, grid=grid,
                                 overrides=over)

    assert (jax.tree_util.tree_structure(wb_params)
            == jax.tree_util.tree_structure(aimip_params))
    assert _shapes(wb_params) == _shapes(aimip_params)
    assert _dtypes(wb_params) == _dtypes(aimip_params)


@pytest.mark.parametrize("mode,variant", [
    ("physics", "classical"),
    ("neural_gcm", "column_nn"),
    ("sfno", "sfno_physics"),
])
def test_the_two_lanes_agree_when_NEITHER_pins_the_architecture(mode, variant,
                                                                grid):
    """The drift this module exists to kill lived in the DEFAULTS.

    The sibling test above pins the same size on both sides, so it would still
    pass if one lane went back to its own hardcoded defaults (codex). This one
    passes no architecture keys at all: it is the assertion that fails the
    moment the WB lane re-grows an ``ov.get("sfno_embed_dim", 256)``.
    """
    from legoesm.training.scale_build import _build_mode_components_spectral

    yml = {"n_lat": 2 * _N_MAX, "nlev": _NLEV,
           "spectral": {"n_max": _N_MAX, "dt": 1800.0}, "loss": {}}
    cfg = SimpleNamespace(mode=mode, training_core="spectral", smoke=False,
                          multi_step_hours=(6,))
    *_, wb_params, _seg, _loss, _dt = _build_mode_components_spectral(cfg, yml)
    registry_params = build_variant(variant, nlev=_NLEV, grid=grid)

    assert _shapes(wb_params) == _shapes(registry_params)
    assert _dtypes(wb_params) == _dtypes(registry_params)


# --------------------------------------------------------------------------
# Per-variant contracts.
# --------------------------------------------------------------------------

def test_sfno_physics_zero_inits_its_decoder(grid):
    """Untrained physics must emit exactly-zero tendencies: epoch 0 is then the
    pure dycore. (The AIMIP lane did NOT do this before the registry; unifying
    on the WB contract is a deliberate change, recorded in the commit.)"""
    sfno = build_variant("sfno_physics", nlev=_NLEV, grid=grid,
                         overrides={"sfno_embed_dim": 16, "sfno_n_blocks": 2})
    assert np.all(np.asarray(sfno.decoder.weight) == 0.0)
    assert np.all(np.asarray(sfno.decoder.bias) == 0.0)


def test_sfno_full_does_not_zero_init_its_decoder(grid):
    """The full emulator replaces the dycore — a zeroed decoder would emit a
    zero state, not a harmless zero tendency."""
    sfno = build_variant("sfno_full", nlev=_NLEV, grid=grid,
                         overrides={"sfno_embed_dim": 16, "sfno_n_blocks": 2})
    assert not np.all(np.asarray(sfno.decoder.weight) == 0.0)


def test_channel_counts_follow_the_variant(grid):
    """sfno_physics carries three forcing planes on the input side; sfno_full
    carries (1 + history) copies of the state and no forcing planes."""
    from legoesm.ml.channel_packing import PE3DChannelSpec
    from legoesm.training.neural_gcm_spectral import N_SFNO_FORCING_CHANNELS

    n_state = PE3DChannelSpec(nlev=_NLEV).n_channels
    assert n_state == 4 * _NLEV + 2

    phys = sfno_arch_config("sfno_physics", nlev=_NLEV)
    assert phys.in_channels == n_state + N_SFNO_FORCING_CHANNELS
    assert phys.out_channels == n_state

    full0 = sfno_arch_config("sfno_full", nlev=_NLEV)
    full2 = sfno_arch_config("sfno_full", nlev=_NLEV,
                             overrides={"sfno_history_steps": 2})
    assert full0.in_channels == n_state
    assert full2.in_channels == 3 * n_state


def test_residual_prediction_is_off_for_both_sfno_variants():
    """SFNOConfig defaults residual_prediction=True, which would add the input
    state to the output and destroy the rollout in one step."""
    for variant in ("sfno_physics", "sfno_full"):
        assert sfno_arch_config(variant, nlev=_NLEV).residual_prediction is False


def test_defaults_are_the_aimip_values_and_are_overridable(grid):
    """Routing run_aimip through the registry must be a no-op, so the defaults
    are AIMIP's. Every shipped WB config pins its SFNO size explicitly, so this
    choice moves no existing run (verified 2026-08-11)."""
    assert VARIANT_DEFAULTS["sfno_physics"]["sfno_embed_dim"] == 128
    assert VARIANT_DEFAULTS["sfno_physics"]["sfno_n_blocks"] == 4
    assert VARIANT_DEFAULTS["column_nn"]["nn_hidden_dim"] == 256

    cfg = sfno_arch_config("sfno_full", nlev=_NLEV,
                           overrides={"sfno_embed_dim": 32})
    assert cfg.embed_dim == 32
    assert cfg.n_blocks == VARIANT_DEFAULTS["sfno_full"]["sfno_n_blocks"]


def test_seed_controls_weights_and_is_reproducible():
    a = build_variant("column_nn", nlev=_NLEV, seed=0,
                      overrides={"nn_hidden_dim": 8, "n_layers": 2})
    b = build_variant("column_nn", nlev=_NLEV, seed=0,
                      overrides={"nn_hidden_dim": 8, "n_layers": 2})
    c = build_variant("column_nn", nlev=_NLEV, seed=1,
                      overrides={"nn_hidden_dim": 8, "n_layers": 2})
    la, lb, lc = (jax.tree.leaves(eqx.filter(x, eqx.is_array)) for x in (a, b, c))
    assert all(np.array_equal(x, y) for x, y in zip(la, lb))
    assert any(not np.array_equal(x, y) for x, y in zip(la, lc))


def test_every_valid_variant_builds(grid):
    for variant in VALID_VARIANTS:
        params = build_variant(
            variant, nlev=_NLEV, grid=grid,
            overrides=({"sfno_embed_dim": 16, "sfno_n_blocks": 2}
                       if variant.startswith("sfno") else None))
        assert jax.tree.leaves(eqx.filter(params, eqx.is_array))


def test_sfno_variants_require_a_grid():
    with pytest.raises(ValueError, match="needs a spectral"):
        build_variant("sfno_full", nlev=_NLEV)
