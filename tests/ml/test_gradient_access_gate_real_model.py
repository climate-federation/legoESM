"""Gradient-access gate on the REAL WeatherBench NeuralGCM loss (tiny grid).

The freshly built model (zero-initialised decoder and head output layer, zero
position embeddings) must pass; a planted detachment of the near-surface head,
of its learned position embedding, or of only its land-fraction input weights
must abort, naming the leaf.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from tests.ml.test_scale_build_spectral_core import _cfg, _synthetic_carry, _yml
from tests.unit.test_near_surface_head import _scales

jax.config.update("jax_enable_x64", True)

LAND = 13   # head input column of the land fraction (12 level inputs, p_s, land)


@pytest.fixture(scope="module")
def built():
    from legoesm.training.losses import LossConfig
    from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
    from legoesm.training.scale_build import (
        build_mode_components,
        lead_diagnose_fn,
        multi_lead_loss,
    )

    from evaluations.wb_forecast import diagnose_headline_fields

    i, o = _scales()
    # Scales matched to the synthetic state: the unit scales of the unit
    # test leave p_s at 1e5 standard deviations, which saturates the GELUs to
    # exact zeros and kills half the hidden units by construction.
    lv = {"T": (280.0, 20.0), "u": (10.0, 10.0), "v": (0.0, 1.0),
          "q_v": (3e-3, 2e-3)}
    i = {**i, **{k: {"mean": [m] * 3, "std": [s] * 3} for k, (m, s) in lv.items()},
         "p_s": {"mean": 1.0e5, "std": 1.0e3},
         "phis": {"mean": 0.0, "std": 1.0e3}}
    # Pruned (prune_unused_weights): no output the loss cannot reach.
    yml = {**_yml(), "near_surface_head": {"seed": 0, "inputs": i, "outputs": o},
           "neural_gcm": {"prune_unused_weights": True},
           "learned_position_embedding": True,
           # w_flux_olr: the spectral lane drops the network's fluxes, so a
           # flux weight must not create an output row nothing can train.
           "loss": {"w_t2m": 1.0, "w_u10": 1.0, "w_v10": 1.0, "w_flux_olr": 1.0}}
    _, grid, sigma, params, mk, _, dt = build_mode_components(_cfg("neural_gcm"), yml)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    # Every head output scored: an unscored output row is dead by construction.
    cfg = LossConfig(w_t2m=1.0, w_u10=1.0, w_v10=1.0, t2m_resid_scale=2.0,
                     u10_resid_scale=2.0, v10_resid_scale=2.0)
    land = (jnp.arange(ncol) % 3 == 0).astype(jnp.float64)
    sic = jnp.where(jnp.arange(ncol) < ncol // 8, 0.7, 0.0)
    terrain = jnp.reshape(land * 500.0 * (1.0 + jnp.arange(ncol) % 5),
                          (int(grid.n_lat), int(grid.n_lon)))

    def sample(k):
        ic = _synthetic_carry(grid, 8)
        # Terrain present (a flat zero plane makes its head input dead).
        ic = ic._replace(T=ic.T + 0.5 * k, phis=terrain)
        # Sea ice present somewhere (a zero plane makes its input weights
        # dead by construction, which is what the v9 decks' climatology fixes).
        f = {"T_sfc": jnp.full((ncol,), 300.0 - k), "sic": sic,
             "land_frac": land, "day_of_year": jnp.asarray(80.0 + k),
             "seconds_of_day": jnp.asarray(21600.0 * k)}
        out = mk(params).raw_multi(ic, 1, 1, f)
        sfc = {key: diagnose_headline_fields(carry_to_spectral_state(out[0], grid),
                                             grid, sigma).fields[key] + 1.0 + k
               for key in ("t2m", "u10", "v10", "mslp")}
        return ic, ((out[0], sfc),), f

    arr, static = eqx.partition(params, eqx.is_inexact_array)

    def make_loss(edit):
        def loss(a, s):
            p = edit(eqx.combine(a, static))
            ic, tg, f = s
            return multi_lead_loss(
                mk(p), ic, tg, f, n_segs=1, seg_steps=1,
                sigma_full=jnp.asarray(sigma.sigma_full), grid=grid, sigma=sigma,
                loss_config=cfg, lead_segs=(1,),
                diagnose_fn=lead_diagnose_fn(diagnose_headline_fields, p, f, dt))
        return loss
    return arr, [sample(k) for k in range(6)], make_loss


def _gate(built, edit=lambda p: p):
    from legoesm.ml.training import TrainingConfig, create_optimizer, make_lr_schedule
    from legoesm.training.data_parallel import build_dp_value_and_grad
    from legoesm.training.inert_params import gradient_access_gate

    arr, samples, make_loss = built
    tc = TrainingConfig(lr=3e-3, optimizer="adamw", warmup_steps=4,
                        total_steps=100, weight_decay=1e-5, grad_clip_norm=1.0)
    opt = create_optimizer(tc)
    vg = build_dp_value_and_grad(make_loss(edit))
    return gradient_access_gate([(vg, s) for s in samples], [arr, opt.init(arr)],
                                opt, make_lr_schedule(tc), optimizer_name="adamw")


def test_fresh_neural_gcm_with_head_and_embedding_passes(built):
    rep = _gate(built)
    assert rep["k"] == 4


def test_detached_head_aborts(built):
    from legoesm.training.inert_params import GradientAccessError
    with pytest.raises(GradientAccessError, match=r"\.head\.layers"):
        _gate(built, lambda p: eqx.tree_at(lambda q: q.head, p,
                                           jax.lax.stop_gradient(p.head)))


def test_detached_head_position_embedding_aborts(built):
    from legoesm.training.inert_params import GradientAccessError
    with pytest.raises(GradientAccessError, match=r"\.head\.pos_embed"):
        _gate(built, lambda p: eqx.tree_at(
            lambda q: q.head.pos_embed, p, jax.lax.stop_gradient(p.head.pos_embed)))


def test_detached_land_input_weights_abort(built):
    from legoesm.training.inert_params import GradientAccessError

    def edit(p):
        w = p.head.layers[0].weight
        w = w.at[:, LAND].set(jax.lax.stop_gradient(w[:, LAND]))
        return eqx.tree_at(lambda q: q.head.layers[0].weight, p, w)
    with pytest.raises(GradientAccessError,
                       match=r"\.head\.layers\[0\]\.weight \(\d+ el: no_grad=64 "):
        _gate(built, edit)
