"""Coverage for the WeatherBench scale-training entrypoint wiring fixes (#797).

The heavy end-to-end smoke needs a GPU + ERA5 (cluster); these tests lock the
JAX-free config/plumbing decisions the five wiring fixes turned on:

- ``_clamped_warmup``   : the cosine schedule never sees decay_steps <= 0.
- ``_apply_smoke_overrides`` : --smoke runs gray radiation (item 6) so the
  wiring check compiles in minutes, with a ``smoke_radiation`` opt-out.
- lat/lon 2-D field for radiation (fix 4): the lat-lon grid exposes ``lat2d``
  while the cube does not, so ``getattr(grid, "lat2d", grid.lat)`` is 2-D on
  both.
- ``build_latlon_config`` passes an explicit int ``rad_update_steps`` (fix 1)
  and propagates the radiation scheme.

The module is import-light + JAX-free at top level (heavy imports live in
``main``), so the CLI/helper tests never import JAX.
"""
from __future__ import annotations

import pytest

from scripts.run.train_weatherbench_scale import (
    ScaleConfig,
    _apply_smoke_overrides,
    _clamped_warmup,
    build_scale_config_from_args,
)


def test_cli_parse_round_trip():
    cfg = build_scale_config_from_args(
        ["--mode", "physics", "--smoke", "--epochs", "7", "--lr", "1e-3"])
    assert isinstance(cfg, ScaleConfig)
    assert cfg.mode == "physics" and cfg.smoke is True
    assert cfg.n_epochs == 7 and cfg.lr == 1e-3


def test_unknown_mode_rejected():
    with pytest.raises(SystemExit):
        build_scale_config_from_args(["--mode", "bogus"])


@pytest.mark.parametrize("epochs", ["0", "-3"])
def test_non_positive_epochs_rejected(epochs):
    # total_steps = n_epochs * n_local feeds optax decay_steps (must be > 0).
    with pytest.raises(SystemExit):
        build_scale_config_from_args(["--epochs", epochs])


def test_clamped_warmup_floors_negative_yaml_to_zero():
    # A YAML warmup_steps: -1 must not reach optax as a negative warmup.
    assert _clamped_warmup(100, -1) == 0
    assert _clamped_warmup(100, -1000) == 0


@pytest.mark.parametrize("total_steps", [1, 2, 3, 4, 5, 10, 40, 100, 100_000])
@pytest.mark.parametrize("desired", [0, 1, 1000])
def test_clamped_warmup_keeps_cosine_decay_positive(total_steps, desired):
    """optax.warmup_cosine_decay_schedule needs total_steps - warmup >= 1."""
    warmup = _clamped_warmup(total_steps, desired)
    assert warmup >= 0
    assert warmup <= desired  # never inflates the requested warmup
    assert total_steps - warmup >= 1  # decay_steps strictly positive


def test_clamped_warmup_degenerate_single_step():
    # total_steps == 1 (one epoch over a single local sample) is the case that
    # previously produced decay_steps == 0 and crashed create_optimizer.
    assert _clamped_warmup(1, 1000) == 0


@pytest.mark.parametrize("total_steps,desired", [(100_000, 1000), (4000, 1000)])
def test_clamped_warmup_preserves_configured_warmup(total_steps, desired):
    # A real run must keep its configured warmup verbatim whenever it is valid
    # (< total_steps) — including desired > 10% of the run (4000, 1000 = 25%),
    # which the old 10%-cap heuristic would have silently coerced to 400.
    assert _clamped_warmup(total_steps, desired) == desired


def test_smoke_overrides_use_gray_radiation():
    cfg = ScaleConfig(
        mode="physics", config_path="x.yaml", resolution_deg=0.7, n_epochs=40,
        multi_step_hours=(6, 12), lr=3e-4, optimizer="adamw", grad_accum=1,
        out_dir="out", resume=False, eval_wb2=False, smoke=True,
    )
    yml = {"warmup_steps": 1000, "radiation": "rrtmgp"}
    new_cfg = _apply_smoke_overrides(cfg, yml)
    # gray radiation crushes the rrtmgp compile wall (item 6) ...
    assert yml["radiation"] == "gray"
    # ... and shrinks the grid + trains a single epoch.
    assert (yml["n_lat"], yml["n_lon"], yml["nlev"]) == (32, 64, 8)
    assert new_cfg.n_epochs == 1
    # cfg is otherwise untouched (NamedTuple replace, not in-place).
    assert new_cfg.mode == "physics" and new_cfg.smoke is True


def test_smoke_radiation_opt_out_into_rrtmgp():
    cfg = build_scale_config_from_args(["--smoke"])
    yml = {"smoke_radiation": "rrtmgp"}
    _apply_smoke_overrides(cfg, yml)
    assert yml["radiation"] == "rrtmgp"


def test_latlon_grid_exposes_2d_lat_for_radiation():
    """Fix 4: radiation flattens a 2-D lat/lon; lat-lon stores a 1-D vector plus
    lat2d/lon2d, the cube stores a 2-D lat and no lat2d. ``getattr(grid, 'lat2d',
    grid.lat)`` must therefore be 2-D on both."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.latlon import create_latlon_grid

    ll = create_latlon_grid(n_lat=16, n_lon=32)
    assert ll.lat.ndim == 1 and ll.lon.ndim == 1
    lat_field = getattr(ll, "lat2d", ll.lat)
    lon_field = getattr(ll, "lon2d", ll.lon)
    assert lat_field.shape == (16, 32)
    assert lon_field.shape == (16, 32)

    cube = create_cubed_sphere(8)
    assert not hasattr(cube, "lat2d")  # falls back to the (6,n,n) lat
    cube_lat = getattr(cube, "lat2d", cube.lat)
    assert cube_lat is cube.lat and cube_lat.ndim == 3


def test_build_latlon_config_rad_update_steps_is_int():
    """Fix 1: scale_build skips run_amip's _postprocess_args (auto None->int), so
    it must pass an explicit int rad_update_steps or the driver hits
    ``int(None)``. Default 1, YAML-overridable; radiation propagates."""
    try:
        from legoesm.training.scale_build import build_latlon_config
    except ImportError as exc:  # pragma: no cover - env-gated
        pytest.skip(f"scale_build import needs the full stack ({exc})")

    cfg = build_scale_config_from_args(["--mode", "physics", "--smoke"])
    base = {"n_lat": 32, "n_lon": 64, "nlev": 8, "dt": 300.0, "radiation": "gray"}

    ec = build_latlon_config(cfg, dict(base))
    assert isinstance(ec.rad_update_steps, int) and ec.rad_update_steps == 1
    assert ec.radiation == "gray"

    ec2 = build_latlon_config(cfg, dict(base, rad_update_steps=6))
    assert ec2.rad_update_steps == 6
