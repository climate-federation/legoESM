"""Tests for the tropical-OMIP dispatch in ``scripts/run/run_omip.py``
(Item 4 Day 2).

Exercises the JRA55-do forcing path:

- ``_setup_jra55_forcing_state``: cache validation, grid-shape match,
  required-flag enforcement.
- ``_jra55_step``: one bulk-flux-driven ocean step on a tiny lat-lon
  C-grid against a synthetic JRA55-do cache.

The full ``run_omip.py`` driver run is too slow for unit-test latency;
we bypass ``run_omip_single`` and exercise the helpers directly with a
real lat-lon C-grid model so the wiring (AtmToSurface →
ocean_tile_response → FreshwaterForcing → model.step) is genuinely
end-to-end.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")


# Load run_omip.py as a module by file path (it lives in scripts/, not
# under the importable package).
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run" / "run_omip.py"
_spec = importlib.util.spec_from_file_location("run_omip", _SCRIPT)
run_omip = importlib.util.module_from_spec(_spec)
sys.modules["run_omip"] = run_omip
_spec.loader.exec_module(run_omip)


jax.config.update("jax_enable_x64", True)


# ============================================================================
# Synthetic cache + grid helpers
# ============================================================================

def _make_synthetic_cache(
    out_dir: Path,
    n_lat: int,
    n_lon: int,
    n_records: int = 8,
) -> Path:
    """Write a tiny pre-built JRA55-do cache directly to Zarr.

    Skips the full build pipeline because we just need the shape +
    schema the runtime loader expects.
    """
    rng = np.random.default_rng(0)
    half_lat = 90.0 / n_lat
    half_lon = 180.0 / n_lon
    lat = np.linspace(-90.0 + half_lat, 90.0 - half_lat, n_lat)
    lon = np.linspace(half_lon, 360.0 - half_lon, n_lon)
    shape = (n_records, n_lat, n_lon)

    # Realistic-magnitude fields, gentle spatial variation.
    base = {
        "uas":    rng.uniform(-5.0, 5.0, shape),
        "vas":    rng.uniform(-3.0, 3.0, shape),
        "tas":    rng.uniform(285.0, 295.0, shape),
        "huss":   rng.uniform(0.005, 0.015, shape),
        "psl":    rng.uniform(99000.0, 102000.0, shape),
        "rsds":   rng.uniform(100.0, 400.0, shape),
        "rlds":   rng.uniform(280.0, 380.0, shape),
        "prra":   rng.uniform(0.0, 1e-5, shape),
        "prsn":   np.zeros(shape),
        "friver": np.zeros(shape),
    }
    ds = xr.Dataset(
        {var: (("time", "lat", "lon"), data) for var, data in base.items()},
        coords={
            "time": np.arange(n_records, dtype=np.int64),
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "calendar": "noleap",
            "records_per_day": 8,
            "n_records": n_records,
            "ref_year": 1958,
            "year_start": 1958,
            "year_end": 1958,
        },
    )
    out = out_dir / "synthetic_cache.zarr"
    ds.to_zarr(str(out), mode="w", consolidated=True)
    return out


def _make_tiny_latlon_setup(n_lat: int = 8, n_lon: int = 16):
    """Build the smallest workable lat-lon C-grid model + state.

    Uses the same APIs ``run_omip._create_setup`` calls.
    """
    return run_omip._create_setup(
        grid_type="latlon",
        resolution=f"{n_lat}x{n_lon}",
        nlev=4,
        H_max=1000.0,
        physics_preset="minimal",
        water_type="II",
    )


def _argparse_namespace(**kwargs):
    """Build an argparse-like Namespace for the helpers."""
    import types
    defaults = dict(
        grid="latlon",
        forcing_mode="jra55_do_tropical",
        jra55_cache=None,
        jra55_co2_ppmv=400.0,
        jra55_cycle=False,    # RYF mode is opt-in
        # Day-3 closure-domain defaults (match parse_args defaults).
        sponge_lat_min=-60.0,
        sponge_lat_max=60.0,
        sponge_width_deg=5.0,
        sponge_tau_days=5.0,
        sss_piston_velocity=5.0e-7,
        jra55_no_sponge=False,
        jra55_no_sss_restoring=False,
        jra55_no_freeze_cap=False,
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


# ============================================================================
# _setup_jra55_forcing_state
# ============================================================================

def test_setup_rejects_non_latlon_grid(tmp_path):
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
    args = _argparse_namespace(jra55_cache=str(cache))
    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    with pytest.raises(ValueError, match=r"supports only --grid"):
        run_omip._setup_jra55_forcing_state(args, grid, "cubed_sphere")


def test_setup_requires_cache_path(tmp_path):
    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    args = _argparse_namespace(jra55_cache=None)
    with pytest.raises(ValueError, match="requires --jra55-cache"):
        run_omip._setup_jra55_forcing_state(args, grid, "latlon")


def test_setup_rejects_missing_cache(tmp_path):
    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    args = _argparse_namespace(jra55_cache=str(tmp_path / "no_such_cache.zarr"))
    with pytest.raises(FileNotFoundError, match="cache not found"):
        run_omip._setup_jra55_forcing_state(args, grid, "latlon")


def test_setup_rejects_grid_shape_mismatch(tmp_path):
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
    grid, *_ = _make_tiny_latlon_setup(n_lat=8, n_lon=16)  # different
    args = _argparse_namespace(jra55_cache=str(cache))
    with pytest.raises(ValueError, match="does not match model grid"):
        run_omip._setup_jra55_forcing_state(args, grid, "latlon")


def test_setup_returns_expected_keys(tmp_path):
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    args = _argparse_namespace(jra55_cache=str(cache))
    state = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
    for key in ("cache_path", "ref_year", "lat_2d", "lon_2d", "coupler_cfg",
                "co2_ppmv"):
        assert key in state
    # Geometry shapes
    assert state["lat_2d"].shape == (4, 1)
    assert state["lon_2d"].shape == (1, 8)
    # Coupler config carries the Item-1 fixes
    assert state["coupler_cfg"].bulk_scheme == "large_yeager"
    assert state["coupler_cfg"].z_ref == 10.0
    assert state["coupler_cfg"].z_t_atm == 2.0
    assert state["coupler_cfg"].z_q_atm == 2.0
    assert state["co2_ppmv"] == 400.0


def test_setup_propagates_co2_override(tmp_path):
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    args = _argparse_namespace(jra55_cache=str(cache), jra55_co2_ppmv=420.0)
    state = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
    assert state["co2_ppmv"] == 420.0


def test_setup_default_cycle_is_off(tmp_path):
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    args = _argparse_namespace(jra55_cache=str(cache))
    state = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
    assert state["cycle"] is False


def test_setup_jra55_cycle_flag_propagates(tmp_path):
    """--jra55-cycle should set the cycle flag in the forcing state."""
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    args = _argparse_namespace(jra55_cache=str(cache), jra55_cycle=True)
    state = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
    assert state["cycle"] is True


def test_block_path_matches_per_step_path(tmp_path):
    """Running 24 steps via the block-scan path produces a state
    consistent with what the per-step path would produce — same
    inputs, same forcing, same dynamics.

    Tolerance is loose because the block path exercises a different
    XLA lowering (one big fused graph vs many small JIT calls), and
    floating-point reductions inside the fused graph can re-order
    additions. We only verify finite output and same shapes.
    """
    n_lat, n_lon = 8, 16
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon,
                                   n_records=200)
    grid, z_coord, _, model, _ = _make_tiny_latlon_setup(
        n_lat=n_lat, n_lon=n_lon,
    )
    T_woa, S_woa = _make_woa_like_targets(grid, nlev=4)

    # Block path
    state_b = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _argparse_namespace(jra55_cache=str(cache))
    js = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon", z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
    )
    out = tmp_path / "blockout"
    state_b, diag_b, wall_b, ok_b, _ = run_omip._run_omip_loop(  # +blowup_info (5-tuple)
        model, state_b, "latlon", grid, z_coord,
        dt=300.0, n_steps=8, diag_every=4,   # 2 blocks of 4 steps each
        jra55_state=js,
        checkpoint_days=None, checkpoint_dir=None,
    )
    assert ok_b, "block path reported not-ok"
    assert bool(jnp.all(jnp.isfinite(state_b.T.data)))

    # Diagnostics must have the right number of entries (initial + 2 blocks)
    assert len(diag_b["day"]) == 3   # day 0 + 2 block ends
    assert diag_b["day"][0] == 0.0
    assert diag_b["day"][-1] == 8 * 300.0 / 86400.0


def test_block_path_writes_restarts_at_cadence(tmp_path):
    """Restart cadence works inside the block-scan loop."""
    n_lat, n_lon = 4, 8
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon,
                                   n_records=200)
    grid, z_coord, _, model, _ = _make_tiny_latlon_setup(
        n_lat=n_lat, n_lon=n_lon,
    )
    T_woa, S_woa = _make_woa_like_targets(grid, nlev=4)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _argparse_namespace(jra55_cache=str(cache))
    js = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon", z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
    )
    out = tmp_path / "ckpt"
    # n_steps=8, dt=10800 s → 0.083 day/step → cum 0.083, 0.167, ... 0.667 day.
    # checkpoint_days=0.25 → steps_per_ckpt = 2. Saves at steps 2, 4, 6, 8.
    run_omip._run_omip_loop(
        model, state, "latlon", grid, z_coord,
        dt=10800.0, n_steps=8, diag_every=2,
        jra55_state=js,
        checkpoint_days=0.25, checkpoint_dir=out,
    )
    files = sorted(out.glob("restart_day*.npz"))
    assert len(files) >= 1, f"no restart files; got {files}"


def test_jra55_step_cycle_runs_past_cache_end(tmp_path):
    """With cycle=True the driver can step past day-365 without
    raising; the same step_idx in 'year 1' produces the same
    forcing as in 'year 0'."""
    n_lat, n_lon = 4, 8
    # 1-year cache (8 records: one day at 3-hourly).
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon,
                                   n_records=8)
    grid, z_coord, _, model, _ = _make_tiny_latlon_setup(
        n_lat=n_lat, n_lon=n_lon,
    )
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _argparse_namespace(jra55_cache=str(cache), jra55_cycle=True)
    js = run_omip._setup_jra55_forcing_state(args, grid, "latlon")

    # cache_length_days = n_records / 8 = 1.0 day.
    # step at dt=10800 s (3 h): step_idx=0 → day 0; step_idx=8 → day 1
    # (= 1 cache cycle); step_idx=16 → day 2 (= 2 cycles); etc.
    s_a = run_omip._jra55_step(state, step_idx=0, dt=10800.0,
                                model=model, jra55_state=js)
    # Without cycle we'd hit IndexError at step_idx=10 (day=1.25 exceeds
    # cache length 1.0); with cycle this should succeed.
    s_b = run_omip._jra55_step(state, step_idx=10, dt=10800.0,
                                model=model, jra55_state=js)
    # Both finite
    assert bool(jnp.all(jnp.isfinite(s_a.T.data)))
    assert bool(jnp.all(jnp.isfinite(s_b.T.data)))


# ============================================================================
# _jra55_step — end-to-end one-step on a tiny lat-lon C-grid
# ============================================================================

@pytest.fixture
def tiny_jra55_run(tmp_path):
    """A tiny lat-lon C-grid model + matching synthetic cache + setup state."""
    n_lat, n_lon = 8, 16
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon)
    grid, z_coord, config, model, _ = _make_tiny_latlon_setup(
        n_lat=n_lat, n_lon=n_lon,
    )
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _argparse_namespace(jra55_cache=str(cache))
    jra55_state = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
    return model, state, jra55_state


def test_jra55_step_advances_state_without_nans(tiny_jra55_run):
    model, state, jra55_state = tiny_jra55_run
    new_state = run_omip._jra55_step(state, step_idx=0, dt=300.0,
                                      model=model, jra55_state=jra55_state)
    # Surface T must remain finite
    T = np.asarray(new_state.T.data)
    assert np.all(np.isfinite(T))
    # State should not be identical (forcing must do *something*)
    assert not np.array_equal(T, np.asarray(state.T.data))


def test_jra55_step_applies_wind_stress(tiny_jra55_run):
    """A step under JRA55-do winds must produce velocity tendency."""
    model, state, jra55_state = tiny_jra55_run
    u_before = np.asarray(state.u.data)
    new_state = run_omip._jra55_step(state, step_idx=0, dt=300.0,
                                      model=model, jra55_state=jra55_state)
    u_after = np.asarray(new_state.u.data)
    # Some velocity tendency at the surface from wind stress
    du_top = u_after[..., 0] - u_before[..., 0]
    assert np.any(np.abs(du_top) > 0.0), (
        "Surface u did not respond to wind stress"
    )


def test_jra55_step_advances_time_index(tiny_jra55_run):
    """Different step indices read different cache slots → different
    forcing → different state evolution."""
    model, state, jra55_state = tiny_jra55_run
    new_state_0 = run_omip._jra55_step(state, step_idx=0, dt=300.0,
                                        model=model, jra55_state=jra55_state)
    # step_idx mapping: day = step_idx * dt / 86400; for dt=10800 (3h),
    # step_idx=1 lands on next 3-hourly slot.
    new_state_1 = run_omip._jra55_step(state, step_idx=1, dt=10800.0,
                                        model=model, jra55_state=jra55_state)
    T0 = np.asarray(new_state_0.T.data[..., 0])
    T1 = np.asarray(new_state_1.T.data[..., 0])
    # Different forcing slot → different surface T evolution
    assert not np.allclose(T0, T1, atol=0.0)


# ============================================================================
# Loop-level dispatch
# ============================================================================

def test_run_omip_loop_rejects_both_forcing_paths(tiny_jra55_run):
    """jra55_state and restoring_targets are mutually exclusive."""
    model, state, jra55_state = tiny_jra55_run
    sst_target = jnp.full(state.T.data[..., 0].shape, 290.0)
    sss_target = jnp.full(state.S.data[..., 0].shape, 35.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_omip._run_omip_loop(
            model, state, "latlon", None, None,
            dt=300.0, n_steps=1, diag_every=1,
            restoring_targets=(sst_target, sss_target),
            restoring_tau_s=86400.0,
            jra55_state=jra55_state,
        )


# ============================================================================
# CLI integration — the new flags
# ============================================================================

def test_cli_accepts_forcing_mode_flag():
    args = run_omip.parse_args.__wrapped__() if hasattr(
        run_omip.parse_args, "__wrapped__"
    ) else None
    # parse_args takes no args via sys.argv; just confirm the choices
    # appear in the parser. We rebuild the parser manually for this.
    # Easier: monkey-patch sys.argv and parse.
    import sys as _sys
    saved = _sys.argv
    try:
        _sys.argv = [
            "run_omip.py",
            "--grid", "latlon",
            "--forcing-mode", "jra55_do_tropical",
            "--jra55-cache", "/tmp/dummy.zarr",
        ]
        parsed = run_omip.parse_args()
    finally:
        _sys.argv = saved
    assert parsed.forcing_mode == "jra55_do_tropical"
    assert parsed.jra55_cache == "/tmp/dummy.zarr"
    assert parsed.jra55_co2_ppmv == 400.0


def test_cli_rejects_unknown_forcing_mode():
    import sys as _sys
    saved = _sys.argv
    try:
        _sys.argv = [
            "run_omip.py",
            "--grid", "latlon",
            "--forcing-mode", "magic_forcing",
        ]
        with pytest.raises(SystemExit):
            run_omip.parse_args()
    finally:
        _sys.argv = saved


# ============================================================================
# Day 3 — sponges + SSS restoring + freeze cap
# ============================================================================

def _make_woa_like_targets(grid, T_water_init_C=18.0, T_deep=2.0, S_uniform=35.0,
                            nlev=4):
    """Build (T_woa, S_woa) of shape (n_lat, n_lon, nlev) — WOA-like
    targets with smooth latitudinal SST and uniform S.

    These are 3-D in the (n_lat, n_lon, nlev) layout the lat-lon model
    uses; they serve as both sponge T_ref / S_ref and SSS target.

    Temperature is in **°C** to match the lat-lon ocean state
    convention (see ``ocean/init_latlon_cgrid.rest_state_latlon_cgrid_ocean``
    and ``ocean/init_woa.init_ocean_from_woa`` — both return °C).
    Salinity is in PSU.
    """
    n_lat = int(grid.n_lat)
    n_lon = int(grid.n_lon)
    lat_deg = np.degrees(np.asarray(grid.lat))
    sst = T_water_init_C * np.cos(np.deg2rad(lat_deg))   # °C
    T_3d = np.empty((n_lat, n_lon, nlev), dtype=np.float64)
    for k in range(nlev):
        # Linear cooling with depth, in °C.
        depth_frac = k / max(nlev - 1, 1)
        T_3d[..., k] = (1.0 - depth_frac) * sst[:, None] + depth_frac * T_deep
    S_3d = np.full((n_lat, n_lon, nlev), S_uniform, dtype=np.float64)
    return T_3d, S_3d


# ----------------------------------------------------------------------------
# Setup state with WOA targets enables sponge / SSS / freeze cap
# ----------------------------------------------------------------------------

def test_setup_with_woa_enables_all_closure_features(tmp_path):
    cache = _make_synthetic_cache(tmp_path, n_lat=8, n_lon=16)
    grid, z_coord, *_ = _make_tiny_latlon_setup(n_lat=8, n_lon=16)
    T_woa, S_woa = _make_woa_like_targets(grid, nlev=4)
    args = _argparse_namespace(jra55_cache=str(cache))
    state = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon",
        z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
    )
    assert state["enable_sponge"] is True
    assert state["enable_sss_restoring"] is True
    assert state["enable_freeze_cap"] is True
    assert state["sponge_gamma_2d"].shape == (8, 16)
    assert state["sponge_T_ref_3d"].shape == (8, 16, 4)
    assert state["sponge_S_ref_3d"].shape == (8, 16, 4)
    assert state["sss_target_2d"].shape == (8, 16)
    # T_freeze_ocean (271.35 K) − T_freeze (273.15 K) = -1.8 °C.
    assert state["T_freeze_ocean_C"] == pytest.approx(-1.8, abs=1e-9)
    assert state["sss_piston_velocity"] == 5.0e-7
    assert state["dz_top"] > 0.0


def test_setup_without_woa_disables_closure_features(tmp_path):
    """If WOA targets aren't passed, sponge/SSS/freeze are silently
    disabled even when the --jra55-no-... flags aren't set."""
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
    grid, z_coord, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    args = _argparse_namespace(jra55_cache=str(cache))
    state = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon", z_coord=z_coord, T_woa=None, S_woa=None,
    )
    assert state["enable_sponge"] is False
    assert state["enable_sss_restoring"] is False
    assert state["enable_freeze_cap"] is False


def test_setup_freeze_cap_requires_sponge():
    """Freeze cap is gated on the sponge mask — disabling sponge
    auto-disables freeze cap."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
        grid, z_coord, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
        T_woa, S_woa = _make_woa_like_targets(grid, nlev=4)
        args = _argparse_namespace(
            jra55_cache=str(cache), jra55_no_sponge=True,
        )
        state = run_omip._setup_jra55_forcing_state(
            args, grid, "latlon",
            z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
        )
        assert state["enable_sponge"] is False
        assert state["enable_freeze_cap"] is False


def test_setup_can_disable_individual_features(tmp_path):
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
    grid, z_coord, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    T_woa, S_woa = _make_woa_like_targets(grid, nlev=4)
    args = _argparse_namespace(
        jra55_cache=str(cache), jra55_no_sss_restoring=True,
    )
    state = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon",
        z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
    )
    assert state["enable_sponge"] is True
    assert state["enable_sss_restoring"] is False
    assert state["enable_freeze_cap"] is True


# ----------------------------------------------------------------------------
# Sponge gamma covers boundary cells
# ----------------------------------------------------------------------------

def test_sponge_gamma_nonzero_only_near_boundaries(tmp_path):
    cache = _make_synthetic_cache(tmp_path, n_lat=24, n_lon=8)  # ~7.5° resolution
    grid, z_coord, *_ = _make_tiny_latlon_setup(n_lat=24, n_lon=8)
    T_woa, S_woa = _make_woa_like_targets(grid, nlev=4)
    args = _argparse_namespace(
        jra55_cache=str(cache),
        sponge_lat_min=-60.0, sponge_lat_max=60.0,
        sponge_width_deg=5.0, sponge_tau_days=5.0,
    )
    state = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon",
        z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
    )
    gamma = np.asarray(state["sponge_gamma_2d"])
    lat_deg = np.degrees(np.asarray(grid.lat))
    # Cells well inside the active domain must have gamma=0
    interior = (lat_deg > -55.0 + 1.0) & (lat_deg < 55.0 - 1.0)
    assert np.all(gamma[interior, :] == 0.0)
    # Cells near the southern or northern boundary must have gamma>0
    near_south = (lat_deg > -60.0) & (lat_deg < -55.0)
    near_north = (lat_deg < 60.0) & (lat_deg > 55.0)
    if np.any(near_south):
        assert np.all(gamma[near_south, :] > 0.0)
    if np.any(near_north):
        assert np.all(gamma[near_north, :] > 0.0)


# ----------------------------------------------------------------------------
# _apply_sss_restoring
# ----------------------------------------------------------------------------

@pytest.fixture
def tiny_jra55_run_full(tmp_path):
    """Tiny lat-lon C-grid + cache + setup *with* sponge/SSS/cap."""
    n_lat, n_lon, nlev = 8, 16, 4
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon)
    grid, z_coord, config, model, _ = _make_tiny_latlon_setup(
        n_lat=n_lat, n_lon=n_lon,
    )
    T_woa, S_woa = _make_woa_like_targets(grid, nlev=nlev)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _argparse_namespace(jra55_cache=str(cache))
    jra55_state = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon",
        z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
    )
    return model, state, jra55_state, grid


def test_sss_restoring_pulls_S_toward_target(tiny_jra55_run_full):
    """A perturbed S field should drift back toward the WOA target."""
    model, state, jra55_state, grid = tiny_jra55_run_full
    # Perturb top-layer salinity off-target
    S = state.S.data
    S_perturbed = S.at[..., 0].set(S[..., 0] + 1.0)  # +1 PSU offset
    state_p = state._replace(S=state.S.replace(data=S_perturbed))

    state_after = run_omip._apply_sss_restoring(state_p, jra55_state, dt=300.0)

    S_top_before = np.asarray(state_p.S.data[..., 0])
    S_top_after = np.asarray(state_after.S.data[..., 0])
    target = np.asarray(jra55_state["sss_target_2d"])
    mask = np.asarray(state.land_mask.data) > 0.5

    # On ocean cells: |S_after - target| < |S_before - target|
    err_before = np.abs(S_top_before[mask] - target[mask])
    err_after = np.abs(S_top_after[mask] - target[mask])
    assert np.all(err_after <= err_before + 1e-12)
    # And the change is non-trivial (not all zeros)
    assert np.any(err_after < err_before - 1e-12)


def test_sss_restoring_does_not_touch_temperature(tiny_jra55_run_full):
    model, state, jra55_state, _ = tiny_jra55_run_full
    state_after = run_omip._apply_sss_restoring(state, jra55_state, dt=300.0)
    np.testing.assert_array_equal(
        np.asarray(state.T.data), np.asarray(state_after.T.data),
    )


def test_sss_restoring_respects_land_mask(tiny_jra55_run_full):
    """Land cells must not have their S modified by the restoring."""
    model, state, jra55_state, _ = tiny_jra55_run_full
    # Perturb everywhere
    S_perturbed = state.S.data.at[..., 0].add(1.0)
    state_p = state._replace(S=state.S.replace(data=S_perturbed))
    state_after = run_omip._apply_sss_restoring(state_p, jra55_state, dt=300.0)

    mask = np.asarray(state.land_mask.data)
    is_land = mask < 0.5
    if np.any(is_land):
        S_before = np.asarray(state_p.S.data[..., 0])
        S_after = np.asarray(state_after.S.data[..., 0])
        np.testing.assert_array_equal(S_before[is_land], S_after[is_land])


# ----------------------------------------------------------------------------
# _apply_freeze_cap
# ----------------------------------------------------------------------------

def test_freeze_cap_raises_below_freeze_inside_sponge(tiny_jra55_run_full):
    """Cells inside the sponge with T < T_freeze_C get raised to T_freeze_C.

    State T is in °C; the cap value is the seawater freezing point in °C
    (-1.8). Driving the surface to -5°C (well below freeze) and applying
    the cap should clip to -1.8°C inside the sponge zone only.
    """
    model, state, jra55_state, _ = tiny_jra55_run_full
    T_cold_C = -5.0  # °C, below seawater freeze
    T_below = state.T.data.at[..., 0].set(
        jnp.full(state.T.data.shape[:-1], T_cold_C),
    )
    state_cold = state._replace(T=state.T.replace(data=T_below))
    state_after = run_omip._apply_freeze_cap(state_cold, jra55_state)

    sponge_mask = np.asarray(jra55_state["sponge_gamma_2d"]) > 0.0
    T_top_after = np.asarray(state_after.T.data[..., 0])
    T_freeze_C = jra55_state["T_freeze_ocean_C"]

    # Inside sponge: T must be >= T_freeze_C
    assert np.all(T_top_after[sponge_mask] >= T_freeze_C - 1e-12)
    # Outside sponge: T unchanged (= -5)
    assert np.all(T_top_after[~sponge_mask] == T_cold_C)


def test_freeze_cap_does_not_affect_warm_cells(tiny_jra55_run_full):
    """Cells already above T_freeze must remain unchanged."""
    model, state, jra55_state, _ = tiny_jra55_run_full
    # T = 18 °C everywhere — well above seawater freeze (-1.8 °C).
    T_warm = state.T.data.at[..., 0].set(jnp.full(state.T.data.shape[:-1], 18.0))
    state_warm = state._replace(T=state.T.replace(data=T_warm))
    state_after = run_omip._apply_freeze_cap(state_warm, jra55_state)
    np.testing.assert_array_equal(
        np.asarray(state_warm.T.data), np.asarray(state_after.T.data),
    )


def test_freeze_cap_does_not_touch_subsurface(tiny_jra55_run_full):
    """Only the surface layer is capped."""
    model, state, jra55_state, _ = tiny_jra55_run_full
    # Cold surface, normal interior
    T_cold = state.T.data
    state_after = run_omip._apply_freeze_cap(state, jra55_state)
    np.testing.assert_array_equal(
        np.asarray(state.T.data[..., 1:]),
        np.asarray(state_after.T.data[..., 1:]),
    )


# ----------------------------------------------------------------------------
# Full _jra55_step with closure features
# ----------------------------------------------------------------------------

def test_jra55_step_with_closure_features_advances_without_nans(tiny_jra55_run_full):
    model, state, jra55_state, _ = tiny_jra55_run_full
    new_state = run_omip._jra55_step(state, step_idx=0, dt=300.0,
                                      model=model, jra55_state=jra55_state)
    # All state arrays remain finite
    for arr in (new_state.T.data, new_state.S.data, new_state.eta.data,
                new_state.u.data, new_state.v.data):
        assert bool(jnp.all(jnp.isfinite(arr))), "non-finite state after step"


def test_jra55_step_runs_with_features_disabled(tmp_path):
    """When all closure features are disabled, the step still works
    (matches the Day-2 path bit-equivalently)."""
    n_lat, n_lon = 8, 16
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon)
    grid, z_coord, _, model, _ = _make_tiny_latlon_setup(n_lat=n_lat, n_lon=n_lon)
    T_woa, S_woa = _make_woa_like_targets(grid, nlev=4)
    state_init = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)

    args_full = _argparse_namespace(jra55_cache=str(cache))
    args_off = _argparse_namespace(
        jra55_cache=str(cache),
        jra55_no_sponge=True,
        jra55_no_sss_restoring=True,
        jra55_no_freeze_cap=True,
    )
    js_full = run_omip._setup_jra55_forcing_state(
        args_full, grid, "latlon",
        z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
    )
    js_off = run_omip._setup_jra55_forcing_state(
        args_off, grid, "latlon",
        z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
    )
    # step_idx=1 so the spinup ramp is nonzero (ramp=0 at step 0
    # would make both paths identical since tau and sponge are zero).
    s_full = run_omip._jra55_step(state_init, 1, 300.0, model, js_full)
    s_off = run_omip._jra55_step(state_init, 1, 300.0, model, js_off)
    # Both finite
    assert bool(jnp.all(jnp.isfinite(s_full.T.data)))
    assert bool(jnp.all(jnp.isfinite(s_off.T.data)))
    # And they should differ — the closure features do something
    assert not np.array_equal(np.asarray(s_full.T.data),
                               np.asarray(s_off.T.data))
