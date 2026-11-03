"""The shared learned 2 m / 10 m head (training.near_surface_head) and its
hand-off into the eval diagnosis, the training loss and the eval scorer."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.training import near_surface_head as nsh

from evaluations.wb_forecast import diagnose_headline_fields
from tests.ml.test_scale_build_spectral_core import _synthetic_carry

NLEV = 8


def _scales(out=(2.0, 1.5, 1.25)):
    inputs = {k: {"mean": [0.0] * nsh.N_HEAD_LEVELS, "std": [1.0] * nsh.N_HEAD_LEVELS}
              for k in nsh.LEVEL_INPUTS}
    inputs.update({k: {"mean": 0.0, "std": 1.0} for k in nsh.PLANE_INPUTS})
    outputs = {f"{k}_gap_std": v for k, v in zip(nsh.OUTPUTS, out)}
    return inputs, outputs


@pytest.fixture(scope="module")
def setup():
    from legoesm.training.neural_gcm_spectral import carry_to_spectral_state

    grid = create_gaussian_grid(10)
    sigma = create_sigma_coordinate(NLEV)
    state = carry_to_spectral_state(_synthetic_carry(grid, NLEV), grid)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    land = jnp.asarray((np.arange(ncol) % 3 == 0).astype(float))
    return grid, sigma, state, ncol, land


def _head(ncol, **kw):
    i, o = _scales()
    return nsh.NearSurfaceHead(ncol, i, o, key=jax.random.PRNGKey(0), **kw)


def test_zero_initialised_head_reproduces_the_proxy(setup):
    grid, sigma, state, ncol, land = setup
    ref = diagnose_headline_fields(state, grid, sigma).fields
    got = diagnose_headline_fields(state, grid, sigma, head=_head(ncol),
                                   land_frac=land, seconds_utc=3600.0).fields
    for k in ("t2m", "u10", "v10", "mslp", "z500"):
        np.testing.assert_array_equal(np.asarray(got[k]), np.asarray(ref[k]))
    np.testing.assert_allclose(np.asarray(got["wind_speed_10m"]),
                               np.asarray(ref["wind_speed_10m"]), rtol=1e-9)
    assert float(jnp.max(ref["u10"])) > 1.0        # the wind is not trivially 0


def test_zero_head_reproduces_the_calm_wind_speed():
    """At u = v = 0 the proxy speed is its regularised floor; the head path
    must return that floor, not a fresh one."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import isothermal_rest_state_spectral

    grid = create_gaussian_grid(10)
    sigma = create_sigma_coordinate(NLEV)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=280.0, p_s_init=1e5)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    ref = diagnose_headline_fields(state, grid, sigma).fields["wind_speed_10m"]
    got = diagnose_headline_fields(state, grid, sigma, head=_head(ncol),
                                   land_frac=jnp.zeros(ncol), seconds_utc=0.0
                                   ).fields["wind_speed_10m"]
    assert float(jnp.max(ref)) < 1e-5
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-9)


def test_correction_is_added_in_physical_units(setup):
    """A last-layer bias b gives t2m = proxy + 2 b0, u10 = log-law + 1.5 b1,
    v10 = log-law + 1.25 b2 (the gap stds), speed from the components."""
    grid, sigma, state, ncol, land = setup
    h = _head(ncol)
    h = eqx.tree_at(lambda m: m.layers[-1].bias, h, jnp.asarray([1.0, -2.0, 4.0]))
    ref = diagnose_headline_fields(state, grid, sigma).fields
    got = diagnose_headline_fields(state, grid, sigma, head=h, land_frac=land,
                                   seconds_utc=0.0).fields
    np.testing.assert_allclose(np.asarray(got["t2m"] - ref["t2m"]), 2.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(got["u10"] - ref["u10"]), -3.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(got["v10"] - ref["v10"]), 5.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(got["wind_speed_10m"]),
                               np.hypot(np.asarray(got["u10"]), np.asarray(got["v10"])),
                               rtol=1e-9)


def test_head_needs_its_context(setup):
    grid, sigma, state, ncol, _ = setup
    with pytest.raises(ValueError, match="land_frac and seconds_utc"):
        diagnose_headline_fields(state, grid, sigma, head=_head(ncol))


def test_features_follow_the_scales_job_conventions():
    """Lowest three levels top to bottom, then the planes; local time angle =
    2 pi utc / 86400 + lon (build_wb_scale_stats.accumulate)."""
    n_lat, n_lon, nl = 2, 3, 5
    T = jnp.broadcast_to(jnp.arange(nl, dtype=float), (n_lat, n_lon, nl))
    z = jnp.zeros((n_lat, n_lon))
    lat, lon = jnp.asarray([-0.5, 0.5]), jnp.asarray([0.0, 2.0, 4.0])
    x = nsh.head_features(T, T, T, T, z + 1e5, z + 9.8, jnp.ones(6), lat, lon, 21600.0)
    assert x.shape == (6, nsh.N_INPUTS)
    np.testing.assert_array_equal(np.asarray(x[0, :3]), [2.0, 3.0, 4.0])
    want = np.sin(2 * np.pi * 21600.0 / 86400.0 + np.asarray(lon))
    np.testing.assert_allclose(np.asarray(x[:3, -2]), want, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(x[3:, 15]), np.sin(0.5), rtol=1e-12)


def test_probe_view_reaches_every_head_leaf(setup):
    """At the zero init only the last layer has a gradient; the probe's
    perturbed last layer reaches every head leaf (incl. the learned map)."""
    *_, ncol, _ = setup
    feats = jax.random.normal(jax.random.PRNGKey(1), (ncol, nsh.N_INPUTS))
    w = nsh.WithHead(core=jnp.ones(()), head=_head(ncol))

    def grads(view):
        g = jax.grad(lambda p: jnp.sum(view(p).head(feats) ** 2 + p.head(feats)))(w)
        return [float(jnp.max(jnp.abs(x))) for x in jax.tree.leaves(g.head)]

    raw = grads(lambda p: p)
    assert sum(v > 0 for v in raw) == 2           # last weight + bias only
    assert all(v > 0 for v in grads(nsh.probe_view))


def test_build_from_deck_block():
    grid = create_gaussian_grid(10)
    i, o = _scales()
    blk = {"seed": 0, "inputs": i, "outputs": o}
    loss = {"loss": {"w_t2m": 1.0}}
    assert nsh.build_near_surface_head({}, grid) is None
    with pytest.raises(ValueError, match="learned_position_embedding"):
        nsh.build_near_surface_head({"near_surface_head": blk, **loss}, grid)
    with pytest.raises(ValueError, match="missing"):
        nsh.build_near_surface_head({"near_surface_head": {"seed": 0}, **loss,
                                     "learned_position_embedding": True}, grid)
    with pytest.raises(ValueError, match="inert"):       # mslp alone
        nsh.build_near_surface_head({"near_surface_head": blk,
                                     "loss": {"w_mslp": 0.1},
                                     "learned_position_embedding": True}, grid)
    on = nsh.build_near_surface_head(
        {"near_surface_head": blk, **loss, "learned_position_embedding": True}, grid)
    off = nsh.build_near_surface_head(
        {"near_surface_head": blk, **loss, "learned_position_embedding": False}, grid)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    assert on.pos_embed.shape == (ncol, 8) and not np.any(np.asarray(on.pos_embed))
    assert off.pos_embed is None
    with pytest.raises(ValueError, match="true or false"):    # quoted YAML
        nsh.build_near_surface_head({"near_surface_head": blk, **loss,
                                     "learned_position_embedding": "false"}, grid)
    with pytest.raises(ValueError, match="must be a block"):
        nsh.build_near_surface_head({"near_surface_head": False}, grid)
    bad = {**o, "t2m_gap_std": 0.0}
    with pytest.raises(ValueError, match="finite and > 0"):
        nsh.NearSurfaceHead(ncol, i, bad, key=jax.random.PRNGKey(0))


def test_orchestrator_reads_the_head_at_each_lead_valid_time(monkeypatch):
    """run_wb_forecast_eval hands the head the case's land fraction and the
    lead's valid time (init seconds of day + lead)."""
    import evaluations.wb_orchestrator as orch

    seen = []

    def fake(mem, grid, sigma, *, resolution_deg=None, **kw):
        seen.append(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(orch, "diagnose_and_regrid", fake)

    class _Case:
        init_state = None
        forcing = {"land_frac": "LAND", "seconds_of_day": 7200.0}

    with pytest.raises(RuntimeError, match="stop"):
        orch.run_wb_forecast_eval(None, None, None, None, 3600.0, [_Case()], [6], {},
                                  rollout_fn=lambda s, *a, **k: s, head="HEAD")
    assert seen == [{"head": "HEAD", "land_frac": "LAND",
                     "seconds_utc": 7200.0 + 6 * 3600.0}]


# ----------------------------------------- learned per-location embeddings

def test_column_net_ablation_keeps_statics_drops_the_learned_map():
    from legoesm.atmosphere.physics.neural_physics import (
        N_COLUMN_POS_EMBED,
        build_column_physics,
    )

    kw = dict(nlev=NLEV, hidden_dim=16, n_layers=2, key=jax.random.PRNGKey(0),
              spatial_embedding=True, n_columns=12)
    on = build_column_physics(**kw, learned_embedding=True)
    off = build_column_physics(**kw, learned_embedding=False)
    assert on.pos_embed.shape == (12, N_COLUMN_POS_EMBED)
    assert off.pos_embed is None
    assert on.n_static == off.n_static == 4
    assert on.n_input - off.n_input == N_COLUMN_POS_EMBED


def test_sfno_learned_map_starts_as_a_no_op_and_is_live():
    from legoesm.ml.sfno import SFNO, SFNOConfig

    grid = create_gaussian_grid(10)
    cfg = SFNOConfig(in_channels=5, out_channels=5, embed_dim=8, n_blocks=1)
    off = SFNO(cfg, grid, key=jax.random.PRNGKey(0))
    on = SFNO(cfg._replace(learned_position_embedding=True), grid,
              key=jax.random.PRNGKey(0))
    assert off.pos_embed is None
    assert on.pos_embed.shape == (int(grid.n_lat), int(grid.n_lon), 8)
    x = jax.random.normal(jax.random.PRNGKey(2), (int(grid.n_lat), int(grid.n_lon), 5))
    np.testing.assert_array_equal(np.asarray(on(x, grid)), np.asarray(off(x, grid)))
    moved = eqx.tree_at(lambda m: m.pos_embed, on, on.pos_embed + 0.1)
    assert float(jnp.max(jnp.abs(moved(x, grid) - on(x, grid)))) > 1e-6


# ------------------------------------------------ the head in the training loss

@pytest.mark.parametrize("mode", ["neural_gcm", "physics"])
def test_head_trains_jointly_through_the_multi_lead_loss(mode):
    """build_mode_components wraps the arm with the deck's head; the loss's
    near-surface term ALONE (every state weight zero) reaches every head leaf
    (probe view) and the arm's own parameters (for CAM6: the physics)."""
    from legoesm.training.losses import LossConfig
    from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
    from legoesm.training.scale_build import (
        build_mode_components,
        lead_diagnose_fn,
        multi_lead_loss,
    )

    from tests.ml.test_scale_build_spectral_core import _cfg, _yml

    i, o = _scales()
    yml = {**_yml(), "near_surface_head": {"seed": 0, "inputs": i, "outputs": o},
           "learned_position_embedding": True, "loss": {"w_t2m": 1.0}}
    _, grid, sigma, params, mk, _, dt = build_mode_components(_cfg(mode), yml)
    assert isinstance(params, nsh.WithHead)
    ic = _synthetic_carry(grid, NLEV)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    f = {"T_sfc": jnp.full((ncol,), 300.0), "sic": jnp.zeros((ncol,)),
         "land_frac": jnp.zeros((ncol,)),
         "day_of_year": jnp.asarray(80.0), "seconds_of_day": jnp.asarray(0.0)}
    if mode == "physics":
        f["land_frac"] = jnp.asarray((np.arange(ncol) % 4 == 0).astype(float))
    outs = mk(params).raw_multi(ic, 1, 1, f)
    sfc = [{k: diagnose_headline_fields(carry_to_spectral_state(x, grid), grid,
                                        sigma).fields[k] + 1.0
            for k in ("t2m", "u10", "v10", "mslp")} for x in outs]
    cfg = LossConfig(w_T=0.0, w_u=0.0, w_v=0.0, w_q=0.0, w_ps=0.0,
                     w_t2m=1.0, t2m_resid_scale=2.0, w_u10=0.1, u10_resid_scale=3.0,
                     w_v10=0.1, v10_resid_scale=3.0)
    assert all(getattr(cfg, k) == 0.0 for k in cfg._fields
               if k.startswith("w_") and not k.endswith(("t2m", "u10", "v10")))
    arr, static = eqx.partition(nsh.probe_view(params), eqx.is_inexact_array)

    def loss(a):
        p = eqx.combine(a, static)
        return multi_lead_loss(
            mk(p), ic, tuple(zip(outs, sfc)), f, n_segs=1, seg_steps=1,
            sigma_full=jnp.asarray(sigma.sigma_full), grid=grid, sigma=sigma,
            loss_config=cfg, lead_segs=(1,),
            diagnose_fn=lead_diagnose_fn(diagnose_headline_fields, p, f, dt))

    g = jax.grad(loss)(arr)
    head_g = [float(jnp.max(jnp.abs(x))) for x in jax.tree.leaves(g.head)]
    assert all(v > 0 for v in head_g)
    core_g = [float(jnp.max(jnp.abs(x))) for x in jax.tree.leaves(g.core)]
    assert max(core_g) > 0


def test_trained_head_survives_the_checkpoint_round_trip(tmp_path):
    """The trainer's on-disk view (checkpoint_view of the whole WithHead)
    loads back into a fresh template with the trained head intact."""
    from legoesm.ml.checkpoint_io import load_checkpoint_or_fail
    from legoesm.ml.training import checkpoint_view

    tmpl = nsh.WithHead(core=jnp.zeros(3), head=_head(12))
    trained = eqx.tree_at(lambda m: m.head.layers[-1].bias, tmpl,
                          jnp.asarray([1.0, 2.0, 3.0]))
    p = tmp_path / "epoch_0001.eqx"
    eqx.tree_serialise_leaves(p, checkpoint_view(trained))
    got = load_checkpoint_or_fail(str(p), tmpl, what="head")
    np.testing.assert_array_equal(np.asarray(got.head.layers[-1].bias), [1.0, 2.0, 3.0])


def test_latlon_core_refuses_the_spectral_only_keys():
    from types import SimpleNamespace

    from legoesm.training.scale_build import build_mode_components

    cfg = SimpleNamespace(mode="neural_gcm", training_core="latlon", smoke=False)
    with pytest.raises(ValueError, match="spectral training core"):
        build_mode_components(cfg, {"learned_position_embedding": False})
