"""CRM driver: RRTMGP column chunking (``--radiation-column-chunk``).

The flag threads into ``RRTMGPConfig.column_chunk_size``; the plane radiation
path then solves the columns in equal blocks. Columns are independent, so the
cloudy CRM heating and its gradient must match the single-shot solve.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)

_SRC = (pathlib.Path(__file__).resolve().parents[2]
        / "scripts/run/run_rcemip_plane.py")


def _load():
    if "run_rcemip_plane" in sys.modules:
        return sys.modules["run_rcemip_plane"]
    spec = importlib.util.spec_from_file_location("run_rcemip_plane", _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_rcemip_plane"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_flag_default_off_and_threads_into_rrtmgp_config():
    rcp = _load()
    assert rcp.parse_args([]).radiation_column_chunk == 0
    assert rcp.parse_args(
        ["--radiation-column-chunk", "64"]).radiation_column_chunk == 64
    assert rcp._build_radiation_config("rrtmgp") == rcp._build_radiation_config(
        "rrtmgp", column_chunk_size=0)
    cfg = rcp._build_radiation_config("rrtmgp", column_chunk_size=64)
    assert cfg.rrtmgp.column_chunk_size == 64


class _Captured(Exception):
    pass


def test_main_forwards_flag_to_radiation_builder(monkeypatch):
    rcp = _load()

    def capture(*args, **kwargs):
        raise _Captured(kwargs)

    monkeypatch.setattr(rcp, "_build_radiation_config", capture)
    monkeypatch.setattr(sys, "argv", [
        "run_rcemip_plane.py", "--nx", "4", "--ny", "4", "--nlev", "8",
        "--steps", "1", "--semi-implicit", "--radiation-column-chunk", "8"])
    with pytest.raises(_Captured) as exc:
        rcp.main()
    assert exc.value.args[0]["column_chunk_size"] == 8


@pytest.mark.parametrize("scheme", ["gray", "none"])
def test_chunk_with_non_rrtmgp_scheme_raises(scheme):
    rcp = _load()
    with pytest.raises(ValueError, match="only applies to --radiation rrtmgp"):
        rcp._build_radiation_config(scheme, column_chunk_size=64)


@pytest.fixture(scope="module")
def cloudy_plane():
    """4x4x10 plane (16 columns), distinct cloudy columns, 4 tracer slots."""
    ny = nx = 4
    nlev = 10
    grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=4_000.0, dy=4_000.0,
                             dtype=jnp.float64)
    hc = create_height_coordinate(nlev, H=20_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    rng = np.random.default_rng(0)
    qv = 0.015 * np.exp(-np.asarray(hc.z_full) / 2500.0)
    tracers = np.zeros((ny, nx, nlev, 4))
    tracers[..., 0] = qv[None, None, :] * rng.uniform(0.6, 1.0, (ny, nx, 1))
    tracers[..., 3:6, 1] = rng.uniform(0.0, 4e-4, (ny, nx, 3))
    tracers[..., 6:8, 3] = rng.uniform(0.0, 1e-4, (ny, nx, 2))
    theta_p = rng.uniform(-2.0, 2.0, (ny, nx, nlev))
    state = rest._replace(
        tracers=rest.tracers.replace(data=jnp.asarray(tracers)),
        theta_prime=rest.theta_prime.replace(data=jnp.asarray(theta_p)),
    )
    return grid, hc, tm, state


def _rad_fn(rcp, grid, chunk):
    cfg = rcp._build_radiation_config("rrtmgp", clouds=True,
                                      column_chunk_size=chunk)
    return rcp._plane_radiation_physics_with_sst(cfg, grid, 300.0)


def test_chunked_cloudy_heating_matches_single_shot(cloudy_plane):
    rcp = _load()
    grid, hc, tm, state = cloudy_plane
    ref = _rad_fn(rcp, grid, 0)(state, grid, hc, tm).dtheta_prime_dt.data
    got = _rad_fn(rcp, grid, 4)(state, grid, hc, tm).dtheta_prime_dt.data
    # Non-vacuous: clouds make columns differ, so a block mix-up would show.
    assert float(jnp.max(jnp.std(ref, axis=(0, 1)))) > 1e-7
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref),
                               rtol=0.0, atol=1e-10)


def test_chunked_gradient_matches_single_shot(cloudy_plane):
    rcp = _load()
    grid, hc, tm, state = cloudy_plane

    def loss(fn):
        def f(theta_p):
            s = state._replace(theta_prime=state.theta_prime.replace(data=theta_p))
            return jnp.sum(fn(s, grid, hc, tm).dtheta_prime_dt.data ** 2)
        return f

    x0 = state.theta_prime.data
    g_ref = jax.grad(loss(_rad_fn(rcp, grid, 0)))(x0)
    g_got = jax.grad(loss(_rad_fn(rcp, grid, 4)))(x0)
    scale = float(jnp.max(jnp.abs(g_ref)))
    assert np.isfinite(np.asarray(g_got)).all() and scale > 0.0
    assert float(jnp.max(jnp.abs(g_got - g_ref))) <= 1e-9 * scale


def test_first_step_returns_freed_heap_when_radiation_runs_every_step(
        monkeypatch, tmp_path):
    """--radiation-interval 1 folds radiation into the step, so no gated
    refresh trims the compile heap; the first step must do it (once)."""
    rcp = _load()
    import legoesm.parallel.staggered_init as si
    calls = []
    monkeypatch.setattr(si, "release_freed_heap", lambda: calls.append(1) or True)
    monkeypatch.setattr(sys, "argv", [
        "run_rcemip_plane.py", "--nx", "4", "--ny", "4", "--nlev", "8",
        "--steps", "2", "--semi-implicit", "--radiation", "gray",
        "--radiation-interval", "1", "--print-every", "1",
        "--output", str(tmp_path)])
    rcp.main()
    assert len(calls) == 1
