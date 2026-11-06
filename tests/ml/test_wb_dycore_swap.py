"""One WeatherBench training step per dynamical core, through the gradient gate.

For each WB training core: build the trainer's own components
(``build_mode_components``, neural_gcm mode, small grid), take the
trainer's loss (``make_run_seg(p).raw`` -> ``combined_loss``) over a
two-step rollout on a synthetic smooth state, and run
``gradient_access_gate`` -- which rehearses value_and_grad + optimizer
update + apply and aborts unless every trainable element is reached.
"""
import importlib.util
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from tests.ml.test_scale_build_spectral_core import _synthetic_carry

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "train_wb_scale_swap", _ROOT / "scripts" / "run" / "train_weatherbench_scale.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

N_STEPS = 2
# --smoke only unlocks gray radiation; the grid is set here (32x64x8 lat-lon,
# T10 spectral as in test_scale_build_spectral_core).
_YML = {"n_lat": 32, "n_lon": 64, "cube_n": 12, "nlev": 8, "dt": 300.0,
        "radiation": "gray",
        "spectral": {"n_max": 10, "dt": 1800.0}, "loss": {},
        # as the v9 WB decks: drop network outputs the loss cannot reach
        "neural_gcm": {"prune_unused_weights": True}}


def _carry(core, grid, nlev):
    """Smooth synthetic SegmentCarry (ERA5 stand-in) on the core's grid."""
    if core != "cubed_sphere":
        return _synthetic_carry(grid, nlev)
    import numpy as np

    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState
    from legoesm.driver.compiled_segments import pack_carry

    lat = np.asarray(grid.grid_lat)                 # (6, n, n)
    s3, prof = lat.shape + (nlev,), np.linspace(0.0, 1.0, nlev)
    u = jnp.asarray(20.0 * np.cos(lat)[..., None] * (0.4 + 0.6 * prof))
    T = jnp.asarray(np.broadcast_to(250.0 + 40.0 * prof, s3).copy())
    q = jnp.asarray(np.broadcast_to(5e-3 * (0.1 + 0.9 * prof), s3).copy())
    z3, z2 = jnp.zeros(s3), jnp.zeros(lat.shape)
    state = HydrostaticState(
        u=Field(u, name="u"), v=Field(z3, name="v"), T=Field(T, name="T"),
        p_s=Field(jnp.full(lat.shape, 1.0e5), name="p_s"),
        phis=Field(z2, name="phis"))
    return pack_carry(state, q_v=q, q_c=z3, q_r=z3, held_dT_rad=z3,
                      held_sw_net_sfc=z2, held_lw_net_sfc=z2,
                      held_sw_up_toa=z2, held_lw_up_toa=z2,
                      held_sw_down_toa=z2, step_index=0)


def _forcing(core, cfg, yml, grid, k):
    """The forcing the core's ERA5 loader would hand the segment."""
    if core == "spectral":
        ncol = int(grid.n_lat) * int(grid.n_lon)
        # sea ice somewhere: a zero plane makes its input weights dead by
        # construction (as in test_gradient_access_gate_real_model)
        return {"T_sfc": jnp.full((ncol,), 300.0 - k),
                "sic": jnp.where(jnp.arange(ncol) < ncol // 8, 0.7, 0.0),
                "day_of_year": jnp.asarray(80.0 + k),
                "seconds_of_day": jnp.asarray(0.0)}
    from legoesm.driver.compiled_segments import pack_forcing
    from legoesm.training.scale_build import _driver_for_ctx, build_latlon_config

    config = build_latlon_config(cfg, yml)
    driver = _driver_for_ctx(config)
    ctx = driver._prepare_run_context(0, config.start_day, restore_carry=False)
    import numpy as np
    lat = np.asarray(grid.grid_lat)
    sst = jnp.full(lat.shape, 300.0 - k)
    # sea ice poleward of 60 deg (a zero plane = dead input weights)
    return pack_forcing(
        sst=sst, sic=jnp.where(jnp.abs(jnp.asarray(lat)) > np.deg2rad(60.0),
                               0.7, 0.0), day_of_year=jnp.asarray(80.0 + k),
        seconds_of_day=jnp.asarray(0.0), solar_weights=ctx["solar_weights"],
        s_0=ctx["current_s_0"], o3_vmr=ctx["o3_vmr"],
        aerosol_od=ctx["aerosol_od"],
        o3_top_vmr=getattr(driver, "_o3_top_vmr", None))


@pytest.mark.parametrize("core,loss", [
    ("latlon", {}), ("spectral", {}), ("cubed_sphere", {}),
    # lat-lon writes the network's OLR into the carry the flux loss reads, so
    # a scored OLR keeps that output row, and the gate proves it is live.
    ("latlon", {"w_flux_olr": 1.0}),
])
def test_wb_training_step_passes_the_gradient_gate(core, loss):
    from legoesm.ml.training import TrainingConfig, create_optimizer, make_lr_schedule
    from legoesm.training.data_parallel import build_dp_value_and_grad
    from legoesm.training.inert_params import gradient_access_gate
    from legoesm.training.losses import combined_loss
    from legoesm.training.scale_build import build_mode_components

    cfg = _mod.build_scale_config_from_args(
        ["--mode", "neural_gcm", "--smoke", "--training-core", core])
    yml = {**_YML, "loss": dict(loss)}
    _, grid, sigma, params, mk, loss_config, _ = build_mode_components(cfg, yml)
    sigma_full = jnp.asarray(sigma.sigma_full)
    nlev = int(yml["nlev"])
    f0 = _forcing(core, cfg, yml, grid, 0)
    if core != "spectral":
        # sea ice on some columns, not all: a constant plane is a dead input
        frac = float(jnp.mean(f0.sic > 0.0))
        assert 0.0 < frac < 0.5, frac

    def sample(k):
        ic = _carry(core, grid, nlev)
        ic = ic._replace(T=ic.T + 0.5 * k)
        # nonzero OLR target: the zero-initialised network emits 0, and a
        # zero target would make the scored OLR row dead by construction
        target = ic._replace(T=ic.T + 1.0, u=ic.u * 0.9,
                             held_lw_up_toa=ic.held_lw_up_toa + 240.0)
        return ic, target, f0

    arr, static = eqx.partition(params, eqx.is_inexact_array)

    def loss_fn(a, s):
        ic, target, f = s
        pred = mk(eqx.combine(a, static)).raw(ic, N_STEPS, f)
        return combined_loss(pred, target, sigma_full, grid=grid,
                             config=loss_config)

    tc = TrainingConfig(lr=3e-3, optimizer="adamw", warmup_steps=4,
                        total_steps=100, weight_decay=1e-5, grad_clip_norm=1.0)
    opt = create_optimizer(tc)
    vg = build_dp_value_and_grad(loss_fn)
    rep = gradient_access_gate([(vg, sample(k)) for k in range(6)],
                               [arr, opt.init(arr)], opt, make_lr_schedule(tc),
                               optimizer_name="adamw")
    assert rep["k"] >= 1


@pytest.mark.parametrize("mode,loss,match", [
    ("sfno", {}, "not on 'cubed_sphere'"),
    ("neural_gcm", {"spectral_weight": 0.5}, "needs a Gaussian grid"),
])
def test_cube_core_refuses_what_it_cannot_build(mode, loss, match):
    from legoesm.training.scale_build import build_mode_components

    cfg = _mod.build_scale_config_from_args(
        ["--mode", mode, "--smoke", "--training-core", "cubed_sphere"])
    with pytest.raises(ValueError, match=match):
        build_mode_components(cfg, {**_YML, "loss": loss})


@pytest.mark.parametrize("bad", [None, 0, -4, 12.0, True, "24"])
def test_cube_n_is_required_and_a_positive_int(bad):
    from legoesm.training.scale_build import wb_cube_n

    yml = {} if bad is None else {"cube_n": bad}
    with pytest.raises(ValueError, match="cube_n"):
        wb_cube_n(yml)
    assert wb_cube_n({"cube_n": 24}) == 24
