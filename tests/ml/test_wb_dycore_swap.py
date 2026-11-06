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
_YML = {"n_lat": 32, "n_lon": 64, "nlev": 8, "dt": 300.0, "radiation": "gray",
        "spectral": {"n_max": 10, "dt": 1800.0}, "loss": {},
        # as the v9 WB decks: drop network outputs the loss cannot reach
        "neural_gcm": {"prune_unused_weights": True}}


def _forcing(core, cfg, yml, grid, k):
    """The forcing the core's ERA5 loader would hand the segment."""
    ncol = int(grid.n_lat) * int(grid.n_lon)
    if core == "spectral":
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
    sst = jnp.full((int(grid.n_lat), int(grid.n_lon)), 300.0 - k)
    return pack_forcing(
        sst=sst, sic=jnp.where(jnp.arange(sst.shape[0])[:, None] < 4, 0.7, 0.0)
        * jnp.ones_like(sst), day_of_year=jnp.asarray(80.0 + k),
        seconds_of_day=jnp.asarray(0.0), solar_weights=ctx["solar_weights"],
        s_0=ctx["current_s_0"], o3_vmr=ctx["o3_vmr"],
        aerosol_od=ctx["aerosol_od"],
        o3_top_vmr=getattr(driver, "_o3_top_vmr", None))


@pytest.mark.parametrize("core,loss", [
    ("latlon", {}), ("spectral", {}),
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

    def sample(k):
        ic = _synthetic_carry(grid, nlev)
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
