"""``prune_unused_weights`` (user 2026-11-06): weights no WeatherBench loss can
reach are not created, and the predictions of what survives are unchanged.

Proof by transplant: a full model with TRAINED-looking weights (random, except
that the pruned outputs keep the exact zeros their zero initialisation and
zero gradient leave them at) and the pruned model holding the same surviving
weights give bit-identical outputs.  For the SFNO the full model's m = 0
imaginary spectral weights are random NONZERO: they cannot change the output,
so the pruned model, which does not have them, must agree exactly.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _randomise(tree, key, scale=0.3):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    out = [x + scale * jax.random.normal(k, x.shape, x.dtype)
           if eqx.is_inexact_array(x) else x for x, k in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(treedef, out)


@pytest.mark.parametrize("kept", [("T", "q_v", "u", "v"),
                                  ("T", "q_v", "u", "v", "lw_up_toa", "sw_net_sfc")])
def test_neural_gcm_pruned_outputs_are_not_created_and_predictions_match(kept):
    from legoesm.atmosphere.physics.neural_physics import (
        build_column_physics,
        column_output_index,
    )

    nlev = 8
    full = build_column_physics(nlev=nlev, key=jax.random.PRNGKey(0))
    pruned = build_column_physics(nlev=nlev, key=jax.random.PRNGKey(0),
                                  kept_outputs=kept)
    # The network creates its groups in LAYOUT order, whatever order asked.
    keep = np.concatenate([column_output_index(g, nlev)
                           for g in pruned.kept_outputs])
    assert full.n_output == 6 * nlev + 6 and pruned.n_output == keep.size
    # Initial weights: identical where they survive.
    for a, b in zip(full.layers[:-1], pruned.layers[:-1]):
        assert bool(jnp.all(a.weight == b.weight) & jnp.all(a.bias == b.bias))
    # Trained-looking full model; the pruned rows stay exactly zero.
    full = _randomise(full, jax.random.PRNGKey(1))
    dead = np.setdiff1d(np.arange(full.n_output), keep)
    last = full.layers[-1]
    full = eqx.tree_at(lambda m: (m.layers[-1].weight, m.layers[-1].bias), full,
                       (last.weight.at[dead].set(0.0), last.bias.at[dead].set(0.0)))
    pruned = eqx.tree_at(
        lambda m: m.layers, pruned,
        [*full.layers[:-1], eqx.tree_at(
            lambda l: (l.weight, l.bias), pruned.layers[-1],
            (full.layers[-1].weight[keep], full.layers[-1].bias[keep]))])
    x = jax.random.normal(jax.random.PRNGKey(2), (5, full.n_input))
    y_full = jax.vmap(full)(x)
    y_pruned = jax.vmap(pruned)(x)
    assert y_pruned.shape == y_full.shape
    assert bool(jnp.all(y_full == y_pruned))
    assert bool(jnp.all(y_pruned[:, dead] == 0.0))
    assert float(jnp.min(jnp.max(jnp.abs(y_full[:, keep]), axis=0))) > 0.0


def test_kept_outputs_follow_what_the_loss_scores():
    from legoesm.training.losses import LossConfig, column_outputs_the_loss_reads

    state = ("T", "q_v", "u", "v")
    assert column_outputs_the_loss_reads(LossConfig(), lane_writes_fluxes=True,
                                         lane_prognostics=state) == state
    lc = LossConfig(w_flux_olr=1.0, w_bias_flux_sfc_sw=0.5)
    assert column_outputs_the_loss_reads(lc, lane_writes_fluxes=True,
                                         lane_prognostics=state) == (
        *state, "lw_up_toa", "sw_net_sfc")
    # A lane that drops the network's fluxes (the WB spectral adapter) cannot
    # train them whatever the flux weights say.
    assert column_outputs_the_loss_reads(lc, lane_writes_fluxes=False,
                                         lane_prognostics=state) == state
    # A lane that evolves condensate keeps its tendencies (they reach the
    # scored state through the dynamics); a lane missing a scored prognostic
    # is refused.
    assert column_outputs_the_loss_reads(
        lc, lane_writes_fluxes=True,
        lane_prognostics=("u", "v", "T", "q_v", "q_c", "q_r")) == (
        *state, "q_c", "q_r", "lw_up_toa", "sw_net_sfc")
    with pytest.raises(ValueError, match="lacks the scored state"):
        column_outputs_the_loss_reads(lc, lane_writes_fluxes=True,
                                      lane_prognostics=("T", "u", "v"))


@pytest.mark.parametrize("n_max", [10])
def test_sfno_pruned_weights_are_not_created_and_predictions_match(n_max):
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.ml.sfno import SFNO, SFNOConfig

    grid = create_gaussian_grid(n_max)
    cfg = SFNOConfig(in_channels=7, out_channels=6, embed_dim=8, n_blocks=2,
                     residual_prediction=False)
    full = SFNO(cfg, grid, key=jax.random.PRNGKey(0))
    pruned = SFNO(cfg._replace(out_channels=5, real_zonal_mean_modes=True),
                  grid, key=jax.random.PRNGKey(0))
    m0 = np.nonzero(np.asarray(grid.ms) == 0)[0]
    rest = np.nonzero(np.asarray(grid.ms) != 0)[0]
    for bf, bp in zip(full.blocks, pruned.blocks):
        assert bp.spectral_conv.weight_imag.shape[0] == len(rest)
        # Initial weights: the surviving imaginary rows are bit-identical.
        assert bool(jnp.all(bf.spectral_conv.weight_imag[rest]
                            == bp.spectral_conv.weight_imag))
    # Trained-looking full model: everything random, INCLUDING the m = 0
    # imaginary weights (they must not matter); the pruned output channel
    # (the last) keeps its exact zero decoder row.
    full = _randomise(full, jax.random.PRNGKey(1))
    full = eqx.tree_at(lambda m: (m.decoder.weight, m.decoder.bias), full,
                       (full.decoder.weight.at[-1].set(0.0),
                        full.decoder.bias.at[-1].set(0.0)))
    assert all(float(jnp.max(jnp.abs(b.spectral_conv.weight_imag[m0]))) > 0
               for b in full.blocks)
    # Transplant leaf by leaf (same paths): the pruned model's spectral
    # imaginary weights are the full ones' m != 0 rows; its decoder drops the
    # last output row; every other leaf is copied unchanged.
    full_l = {jax.tree_util.keystr(p): x
              for p, x in jax.tree_util.tree_leaves_with_path(full)}

    def take(path, x):
        k = jax.tree_util.keystr(path)
        src = full_l[k]
        if k.endswith("spectral_conv.weight_imag"):
            src = src[rest]
        elif k.endswith("decoder.weight") or k.endswith("decoder.bias"):
            src = src[:-1]
        assert src.shape == x.shape, k
        return src
    pruned = jax.tree_util.tree_map_with_path(take, pruned)
    x = jax.random.normal(jax.random.PRNGKey(3),
                          (int(grid.n_lat), int(grid.n_lon), 7), jnp.float32)
    y_full = full(x, grid)
    y_pruned = pruned(x, grid)
    assert y_pruned.shape[-1] == 5
    np.testing.assert_array_equal(np.asarray(y_full[..., :-1]), np.asarray(y_pruned))
    assert bool(jnp.all(y_full[..., -1] == 0.0))
    assert float(jnp.max(jnp.abs(y_pruned))) > 0.0


def test_wb_decks_select_pruning_explicitly():
    import yaml

    for arm in ("ngcm", "ngcm_nolearnedembed", "sfno", "sfno_nolearnedembed"):
        y = yaml.safe_load(open(f"config/wb/campaign/spectral_t63_{arm}_v9.yaml"))
        assert y["neural_gcm"]["prune_unused_weights"] is True
        assert y["sfno"]["prune_unused_weights"] is True
    # the Derecho lat-lon deck (neural_gcm; the lat-lon sfno arm has no pruning)
    y = yaml.safe_load(open("config/wb/scale/train_07deg.yaml"))
    assert y["neural_gcm"]["prune_unused_weights"] is True


def test_registry_builds_the_pruned_layouts():
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.ml.channel_packing import PE3DChannelSpec
    from legoesm.training.model_registry import build_variant

    grid = create_gaussian_grid(10)
    cn = build_variant("column_nn", nlev=8, grid=grid,
                       overrides={"kept_outputs": ("T", "q_v", "u", "v")})
    assert cn.n_output == 4 * 8
    sf = build_variant("sfno_physics", nlev=8, grid=grid,
                       overrides={"prune_unused_weights": True})
    assert sf.decoder.weight.shape[0] == PE3DChannelSpec(nlev=8).n_channels - 1
    n_m0 = int(np.sum(np.asarray(grid.ms) == 0))
    assert sf.blocks[0].spectral_conv.weight_imag.shape[0] == int(grid.n_sh) - n_m0
    # Off by default: AIMIP's layout is unchanged.
    assert build_variant("column_nn", nlev=8, grid=grid).n_output == 6 * 8 + 6
