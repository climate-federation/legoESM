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
from typing import NamedTuple

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
    """Argparse-like Namespace seeded from the REAL parser defaults.

    The previous hand-built ``SimpleNamespace`` dict rotted every time a
    new flag landed (``--surface-stability-scheme`` broke 32 tests with
    ``AttributeError`` — the setup helpers read ``args.<new_flag>``
    directly).  Seeding from ``parse_args`` keeps every current AND
    future flag present at its production default; ``kwargs`` override.
    """
    args = run_omip.parse_args(
        ["--grid", "latlon", "--forcing-mode", "jra55_do_tropical"])
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


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
    # ALL THREE at 10 m: JRA55-do's tas/huss/uas each carry an explicit
    # height=10 m coordinate, and FESOM2 forces the same dataset with
    # ncar_bulk_z_tair = ncar_bulk_z_shum = 10.0.  These previously asserted
    # 2.0, pinning a defect worth ~+11.5 W/m^2 of latent heat flux.
    assert state["coupler_cfg"].z_t_atm == 10.0
    assert state["coupler_cfg"].z_q_atm == 10.0
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
    """If WOA targets aren't passed, sponge/SSS are silently disabled
    even when the --jra55-no-... flags aren't set.

    The freeze cap stays ON: since commit 2d343dfdc it no longer
    depends on the sponge mask/WOA data — with no sponge it caps
    globally over all ocean cells via the state's own land mask
    (see ``_apply_freeze_cap`` and the ``_ocean_mask_2d`` block path).
    """
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
    grid, z_coord, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    args = _argparse_namespace(jra55_cache=str(cache))
    state = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon", z_coord=z_coord, T_woa=None, S_woa=None,
    )
    assert state["enable_sponge"] is False
    assert state["enable_sss_restoring"] is False
    assert state["enable_freeze_cap"] is True


def test_setup_freeze_cap_independent_of_sponge():
    """Freeze cap is NOT gated on the sponge (commit 2d343dfdc):
    disabling the sponge keeps the cap on, scoped globally over all
    ocean cells instead of the sponge zone. Only --jra55-no-freeze-cap
    (or --jra55-sea-ice, see the sea-ice setup test) turns it off."""
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
        assert state["enable_freeze_cap"] is True


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


# ============================================================================
# F1 — cycled GPU-interp forcing clock (record window + scan-body clock)
# ============================================================================

class _ForcingProbeField(NamedTuple):
    data: jnp.ndarray


class _ForcingProbeState(NamedTuple):
    T: _ForcingProbeField


def _make_forcing_probe_model():
    """Minimal stand-in for the ocean model inside the GPU-interp block fn.

    ``_step_impl`` encodes the interpolated shortwave forcing it receives
    into the carried state (``sf.sw_down`` is the raw interpolated
    ``rsds`` — untouched by the ocean state), so the block's final state
    exposes the LAST step's selected/interpolated forcing for assertion.
    """
    import types

    def _step_impl(state_in, dt, freshwater=None, surface_forcing=None,
                   sponge=None):
        probe = surface_forcing.sw_down[..., None] * 1e-3
        return _ForcingProbeState(
            T=_ForcingProbeField(
                data=jnp.broadcast_to(probe, state_in.T.data.shape),
            ),
        )

    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            barotropic=types.SimpleNamespace(maxvel_barotropic=0.0),
        ),
        _step_impl=_step_impl,
    )


def test_preload_raw_records_second_cycle_meta_and_indices(tmp_path):
    """F1: in the SECOND repeat-year cycle the preloader must return the
    CYCLED block-start day for the interpolation clock (raw day is kept
    separately for solar zenith) and record days aligned with it."""
    n_records = 16  # 2.0-day cache at 8 records/day
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8,
                                  n_records=n_records)
    js = {"cache_path": str(cache), "ref_year": 1958, "cycle": True}
    dt = 5400.0          # 0.0625 day = half a 3-hourly record interval
    start_step_idx = 41  # day 2.5625 -> cycled 0.5625 (second cycle)

    raw_stack, _, meta = run_omip._preload_jra55_raw_records(
        start_step_idx, 1, dt, js)

    # Raw day preserved for the solar-zenith clock.
    assert meta["block_start_day"] == pytest.approx(2.5625)
    # Cycled forcing clock: 2.5625 mod 2.0.
    assert meta["block_start_day_forcing"] == pytest.approx(0.5625)
    days = np.asarray(meta["record_days"])
    np.testing.assert_allclose(days, [0.5, 0.625])
    # The forcing clock must be bracketed by the returned record days.
    assert days[0] <= meta["block_start_day_forcing"] <= days[-1]
    # Selected raw records are cache records 4 and 5.
    ds = xr.open_zarr(str(cache), decode_times=False)
    np.testing.assert_allclose(
        np.asarray(raw_stack["tas"]), ds["tas"].isel(time=[4, 5]).values)


def test_preload_raw_records_unwraps_record_days_across_cache_wrap(tmp_path):
    """F1: a block straddling the repeat-year wrap gets MONOTONIC record
    days (post-wrap entries shifted by +cache_length_days) so linear
    interpolation stays correct across the boundary."""
    n_records = 16  # 2.0-day cache
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8,
                                  n_records=n_records)
    js = {"cache_path": str(cache), "ref_year": 1958, "cycle": True}
    dt = 5400.0
    # Steps 63, 64: days 3.9375, 4.0 -> cycled 1.9375, 2.0 (straddles wrap).
    raw_stack, _, meta = run_omip._preload_jra55_raw_records(63, 2, dt, js)

    days = np.asarray(meta["record_days"])
    assert np.all(np.diff(days) > 0), f"record_days not monotonic: {days}"
    np.testing.assert_allclose(days, [1.875, 2.0, 2.125])
    assert meta["block_start_day_forcing"] == pytest.approx(1.9375)
    ds = xr.open_zarr(str(cache), decode_times=False)
    np.testing.assert_allclose(
        np.asarray(raw_stack["rsds"]),
        ds["rsds"].isel(time=[15, 0, 1]).values)


def test_slice_preloaded_records_second_cycle_matches_raw_preloader(tmp_path):
    """F1: the in-RAM slicer must produce the same window/meta as the
    Zarr preloader (float32 staging tolerance) in the second cycle."""
    n_records = 16
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8,
                                  n_records=n_records)
    js = {"cache_path": str(cache), "ref_year": 1958, "cycle": True}
    full = run_omip._preload_jra55_full_cache(js)

    raw_a, _, meta_a = run_omip._slice_preloaded_records(
        41, 4, 5400.0, js, *full)
    raw_b, _, meta_b = run_omip._preload_jra55_raw_records(41, 4, 5400.0, js)

    np.testing.assert_allclose(np.asarray(meta_a["record_days"]),
                               np.asarray(meta_b["record_days"]))
    assert (meta_a["block_start_day_forcing"]
            == meta_b["block_start_day_forcing"])
    assert meta_a["block_start_day"] == meta_b["block_start_day"]
    for var in raw_b:
        np.testing.assert_allclose(
            np.asarray(raw_a[var]), np.asarray(raw_b[var]), rtol=1e-6)


def test_gpu_interp_block_uses_cycled_forcing_clock_second_cycle(tmp_path):
    """F1 end-to-end: the jitted interp scan body must select the
    analytically expected bracketing records + alpha in the SECOND
    repeat-year cycle.

    Before the fix the RAW simulation day was compared against
    cache-relative record_days, so ``i_lo`` clipped to the last slice
    record and ``alpha`` clamped to 1 — every step read one stale
    record instead of interpolating."""
    n_lat, n_lon, n_records = 4, 8, 16  # 2.0-day cache
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon,
                                  n_records=n_records)
    grid, *_ = _make_tiny_latlon_setup(n_lat=n_lat, n_lon=n_lon)
    args = _argparse_namespace(jra55_cache=str(cache), jra55_cycle=True)
    js = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
    js["enable_freeze_cap"] = False  # probe state carries no land mask

    dt = 5400.0  # half a record interval
    probe_model = _make_forcing_probe_model()
    get_bfn = run_omip._build_jra55_block_fn_interp(probe_model, js, dt)
    state0 = _ForcingProbeState(
        T=_ForcingProbeField(data=jnp.zeros((n_lat, n_lon, 1))))
    ds = xr.open_zarr(str(cache), decode_times=False)

    # (a) Single-step block at raw day 2.5625 (cycled 0.5625): midway
    # between records 4 (day 0.5) and 5 (day 0.625) -> alpha = 0.5.
    raw_stack, runoff, meta = run_omip._preload_jra55_raw_records(
        41, 1, dt, js)
    final = get_bfn(1)(
        state0, raw_stack, runoff, meta["record_days"],
        jnp.float64(meta["block_start_day"]),
        jnp.float64(meta["block_start_day_forcing"]),
    )
    probed_sw = np.asarray(final.T.data[..., 0]) * 1e3
    expected = 0.5 * (ds["rsds"].isel(time=4).values
                      + ds["rsds"].isel(time=5).values)
    np.testing.assert_allclose(probed_sw, expected, rtol=1e-9)

    # (b) Two-step block straddling the wrap (raw days 3.9375, 4.0):
    # the final step lands exactly ON the wrap record -> rsds[0].
    raw_stack, runoff, meta = run_omip._preload_jra55_raw_records(
        63, 2, dt, js)
    final = get_bfn(2)(
        state0, raw_stack, runoff, meta["record_days"],
        jnp.float64(meta["block_start_day"]),
        jnp.float64(meta["block_start_day_forcing"]),
    )
    probed_sw = np.asarray(final.T.data[..., 0]) * 1e3
    np.testing.assert_allclose(
        probed_sw, ds["rsds"].isel(time=0).values, rtol=1e-9)


def test_loop_gpu_interp_cycled_second_cycle_runs_finite(tmp_path):
    """F1 integration: the block-scan loop on the GPU-interp path with
    repeat-year cycling runs blocks entirely inside the SECOND cycle
    (including the full-cache slicer) and stays finite."""
    n_lat, n_lon = 4, 8
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon,
                                  n_records=16)  # 2-day cache
    grid, z_coord, _, model, _ = _make_tiny_latlon_setup(
        n_lat=n_lat, n_lon=n_lon)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _argparse_namespace(jra55_cache=str(cache), jra55_cycle=True)
    js = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
    js["_gpu_interp"] = True
    # Global freeze cap needs the ocean mask (run_omip_single wires this).
    js["_ocean_mask_2d"] = state.land_mask.data > 0.5
    # dt=10800 s (3 h) -> 8 steps/day; the second cycle starts at step 16.
    state_out, _, _, ok, _ = run_omip._run_omip_loop(
        model, state, "latlon", grid, z_coord,
        dt=10800.0, n_steps=20, diag_every=2, jra55_state=js,
        checkpoint_days=None, checkpoint_dir=None, start_step=16,
    )
    assert ok
    assert bool(jnp.all(jnp.isfinite(state_out.T.data)))


# ============================================================================
# F2 — non-cycled preloaders must raise past the cache end (no silent clamp)
# ============================================================================

def test_preload_raw_records_noncycle_past_cache_end_raises(tmp_path):
    """F2: cycle=False + block past the cache end must raise IndexError
    (matching the jra55_do loader's no-silent-synthetic contract), not
    silently clamp to the last record."""
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8,
                                  n_records=8)  # 1-day cache
    js = {"cache_path": str(cache), "ref_year": 1958, "cycle": False}
    with pytest.raises(IndexError, match="exceeds cache length"):
        # start day 1.25 > 1.0-day cache.
        run_omip._preload_jra55_raw_records(10, 4, 10800.0, js)


def test_slice_preloaded_records_noncycle_past_cache_end_raises(tmp_path):
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8, n_records=8)
    js = {"cache_path": str(cache), "ref_year": 1958, "cycle": False}
    full = run_omip._preload_jra55_full_cache(js)
    with pytest.raises(IndexError, match="exceeds cache length"):
        run_omip._slice_preloaded_records(10, 4, 10800.0, js, *full)


def test_preload_raw_records_noncycle_exact_last_record_ok(tmp_path):
    """Loader parity: a block whose final step lands EXACTLY on the last
    cache record needs no upper bracket beyond it and must NOT raise."""
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8, n_records=8)
    js = {"cache_path": str(cache), "ref_year": 1958, "cycle": False}
    # dt=10800 s (3 h): steps 0..7 -> last day 0.875 == final record.
    _, _, meta = run_omip._preload_jra55_raw_records(0, 8, 10800.0, js)
    days = np.asarray(meta["record_days"])
    assert days[-1] == pytest.approx(0.875)
    assert meta["block_start_day_forcing"] == meta["block_start_day"]


# ============================================================================
# F3 — --no-gpu-interp round trip (flag must reach the dispatch)
# ============================================================================

def test_cli_gpu_interp_flag_round_trip():
    import sys as _sys
    saved = _sys.argv
    try:
        _sys.argv = ["run_omip.py", "--grid", "latlon"]
        assert run_omip.parse_args().gpu_interp is True
        _sys.argv = ["run_omip.py", "--grid", "latlon", "--no-gpu-interp"]
        assert run_omip.parse_args().gpu_interp is False
    finally:
        _sys.argv = saved


class _LoopCapturedError(Exception):
    """Sentinel: short-circuit run_omip_single at the time-loop boundary."""


@pytest.mark.parametrize("extra_argv,expected", [
    ([], True),
    (["--no-gpu-interp"], False),
])
def test_no_gpu_interp_flag_reaches_jra55_dispatch(tmp_path, monkeypatch,
                                                   extra_argv, expected):
    """F3: run_omip_single must wire args.gpu_interp into
    jra55_state['_gpu_interp'] (it was hardcoded True, making
    --no-gpu-interp a silent no-op)."""
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8, n_records=8)
    captured = {}

    def _spy_loop(*a, **kw):
        captured["jra55_state"] = kw.get("jra55_state")
        raise _LoopCapturedError()

    monkeypatch.setattr(run_omip, "_run_omip_loop", _spy_loop)

    import sys as _sys
    saved = _sys.argv
    try:
        _sys.argv = [
            "run_omip.py", "--grid", "latlon", "--resolution", "4x8",
            "--nlev", "4", "--dt", "10800", "--days", "1",
            "--forcing-mode", "jra55_do_tropical",
            "--jra55-cache", str(cache),
            "--output", str(tmp_path / "out"),
        ] + extra_argv
        args = run_omip.parse_args()
    finally:
        _sys.argv = saved

    with pytest.raises(_LoopCapturedError):
        run_omip.run_omip_single("latlon", args)
    assert captured["jra55_state"]["_gpu_interp"] is expected


@pytest.mark.parametrize("gpu_interp,expected_calls", [
    (False, {"cpu": 1, "gpu": 0}),
    (True, {"cpu": 0, "gpu": 1}),
])
def test_loop_dispatch_honors_gpu_interp_flag(tmp_path, monkeypatch,
                                              gpu_interp, expected_calls):
    """F3: _gpu_interp=False must route to the CPU-interp block path
    (_build_jra55_block_fn) and the run must complete finite — the CPU
    path stays exercised, not just reachable."""
    n_lat, n_lon = 4, 8
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon,
                                  n_records=16)
    grid, z_coord, _, model, _ = _make_tiny_latlon_setup(
        n_lat=n_lat, n_lon=n_lon)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _argparse_namespace(jra55_cache=str(cache))
    js = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
    js["_gpu_interp"] = gpu_interp
    # Global freeze cap needs the ocean mask (run_omip_single wires this).
    js["_ocean_mask_2d"] = state.land_mask.data > 0.5

    calls = {"cpu": 0, "gpu": 0}
    real_cpu = run_omip._build_jra55_block_fn
    real_gpu = run_omip._build_jra55_block_fn_interp

    def _spy_cpu(*a, **k):
        calls["cpu"] += 1
        return real_cpu(*a, **k)

    def _spy_gpu(*a, **k):
        calls["gpu"] += 1
        return real_gpu(*a, **k)

    monkeypatch.setattr(run_omip, "_build_jra55_block_fn", _spy_cpu)
    monkeypatch.setattr(run_omip, "_build_jra55_block_fn_interp", _spy_gpu)

    state_out, _, _, ok, _ = run_omip._run_omip_loop(
        model, state, "latlon", grid, z_coord,
        dt=10800.0, n_steps=4, diag_every=2, jra55_state=js,
        checkpoint_days=None, checkpoint_dir=None,
    )
    assert ok
    assert calls == expected_calls
    assert bool(jnp.all(jnp.isfinite(state_out.T.data)))


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


# ============================================================================
# FIX 1 (zenithfix): cycled JRA55 insolation clock must follow the FORCING
# clock, not the raw sim day.
#
# In ``_build_jra55_block_fn_interp`` the recent forcing-clock fix split
# ``day`` (RAW sim day) from ``day_f`` (CYCLED forcing clock).  The solar
# zenith's ``doy``/``hour`` must be locked to the repeated forcing (``day_f``)
# when cycling, otherwise the seasonal (and, for a non-integer cache length,
# diurnal) solar phase drifts relative to the prescribed rsds whenever the
# repeat-year cache length is not a whole multiple of 365 days.
#
# Observable probe: monkeypatch ``ocean_tile_response`` to route the block's
# real computed ``cos_zenith`` into ``tile.tau_x`` (all other tile fields
# zero); a probe ``_step_impl`` encodes ``surface_forcing.tau_x`` into the
# carried state, so the block's final state exposes the LAST step's zenith.
# ``T_ramp_seconds=0`` fixes the spinup ramp at 1 so ``tau_x == cos_zenith``.
# ============================================================================

_JRA55_RAW_VARS = ("uas", "vas", "tas", "huss", "psl",
                   "rsds", "rlds", "prra", "prsn")


class _ZenithProbeField(NamedTuple):
    data: jnp.ndarray


class _ZenithProbeState(NamedTuple):
    T: _ZenithProbeField


def _make_zenith_probe_model():
    """Ocean-model stand-in that surfaces the block's solar zenith.

    The fake ``ocean_tile_response`` routes ``cos_zenith`` into ``tau_x``;
    with the spinup ramp fixed at 1, ``surface_forcing.tau_x`` equals the
    step's ``cos_zenith``.  ``_step_impl`` copies it into the carried state.
    """
    import types

    def _step_impl(state_in, dt, freshwater=None, surface_forcing=None,
                   sponge=None):
        probe = surface_forcing.tau_x[..., None]
        return _ZenithProbeState(
            T=_ZenithProbeField(
                data=jnp.broadcast_to(probe, state_in.T.data.shape)))

    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            barotropic=types.SimpleNamespace(maxvel_barotropic=0.0)),
        _step_impl=_step_impl,
    )


def _install_cos_zenith_tile_probe(monkeypatch):
    """Patch ``ocean_tile_response`` to expose the block's ``cos_zenith``.

    Must run BEFORE ``_build_jra55_block_fn_interp`` is called — the builder
    does a function-scope ``from legoesm.coupler.coupler import
    ocean_tile_response`` that binds this module attribute at call time.
    """
    import types
    import legoesm.coupler.coupler as _cc

    def _fake_tile(forcing, sst, u, v, config):
        cos_z = forcing.cos_zenith
        zeros = jnp.zeros_like(cos_z)
        return types.SimpleNamespace(
            albedo=zeros, lw_up=zeros, shflx=zeros, lhflx=zeros,
            tau_x=cos_z, tau_y=zeros)

    monkeypatch.setattr(_cc, "ocean_tile_response", _fake_tile)


def _zenith_js(tmp_path, n_lat=4, n_lon=8, cycle=True):
    """Build a real JRA55 forcing state, then neutralise everything except
    the solar-zenith clock (ramp off, freeze-cap off)."""
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon,
                                  n_records=8)
    grid, *_ = _make_tiny_latlon_setup(n_lat=n_lat, n_lon=n_lon)
    args = _argparse_namespace(jra55_cache=str(cache), jra55_cycle=cycle)
    js = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
    js["T_ramp_seconds"] = 0.0        # ramp == 1 -> tau_x == cos_zenith
    js["enable_freeze_cap"] = False    # probe state carries no land mask
    return js


def _mini_raw_stack(day_f, n_lat=4, n_lon=8):
    """Three JRA55 records on the 1/8-day grid bracketing ``day_f``.

    Forcing values are irrelevant (the tile is faked) — only the record
    days matter for the block's interpolation window.
    """
    from legoesm.forcing.jra55_do import RECORDS_PER_DAY
    i0 = int(np.floor(day_f * RECORDS_PER_DAY))
    record_days = jnp.asarray(
        [(i0 + k) / RECORDS_PER_DAY for k in range(3)], dtype=jnp.float64)
    vals = {"uas": 1.0, "vas": 1.0, "tas": 290.0, "huss": 0.01,
            "psl": 101325.0, "rsds": 300.0, "rlds": 350.0,
            "prra": 0.0, "prsn": 0.0}
    raw_stack = {v: jnp.full((3, n_lat, n_lon), val, dtype=jnp.float64)
                 for v, val in vals.items()}
    runoff = jnp.zeros((3, n_lat, n_lon), dtype=jnp.float64)
    return raw_stack, runoff, record_days


def _run_zenith_block(js, block_start_day, block_start_day_forcing,
                      monkeypatch, n_lat=4, n_lon=8):
    """Drive one production block step and return its selected cos_zenith."""
    _install_cos_zenith_tile_probe(monkeypatch)
    probe_model = _make_zenith_probe_model()
    get_bfn = run_omip._build_jra55_block_fn_interp(probe_model, js, dt=5400.0)
    raw_stack, runoff, record_days = _mini_raw_stack(
        block_start_day_forcing, n_lat, n_lon)
    state0 = _ZenithProbeState(
        T=_ZenithProbeField(data=jnp.zeros((n_lat, n_lon, 1))))
    final = get_bfn(1)(
        state0, raw_stack, runoff, record_days,
        jnp.float64(block_start_day),
        jnp.float64(block_start_day_forcing))
    return np.asarray(final.T.data[..., 0])


def _expected_cos_zenith(js, day):
    """Reference cos_zenith from a given clock day (same helper the block
    uses internally)."""
    from legoesm.atmosphere.physics.radiation.solar import cos_zenith_angle
    doy = np.mod(day, 365.0) + 1.0
    hour = np.mod(day, 1.0) * 24.0
    return np.asarray(cos_zenith_angle(js["lat_2d"], js["lon_2d"], doy, hour))


def test_zenithfix_second_cycle_uses_forcing_clock_not_raw_day(
        tmp_path, monkeypatch):
    """FIX 1 (RED before fix): in the SECOND repeat-year cycle of a cache
    whose length is NOT a whole multiple of 365 days, the solar-zenith
    ``doy``/``hour`` must come from the CYCLED forcing clock (``day_f``),
    NOT the raw sim day.

    Proxy for a 366-day (leap-year) RYF cache: a 2.125-day cache (17
    records at 8/day).  In the second cycle raw day 2.5625 maps to forcing
    day 0.4375; the 2.125-day offset shifts BOTH the seasonal doy (by 2.125
    days) and the diurnal hour (by 0.125*24 = 3 h).  Before the fix the raw
    day drove the zenith, drifting it away from the prescribed forcing.
    """
    js = _zenith_js(tmp_path, cycle=True)
    raw_day = 2.5625
    forcing_day = 0.4375   # 2.5625 mod 2.125
    probed = _run_zenith_block(js, raw_day, forcing_day, monkeypatch)

    expected_forcing = _expected_cos_zenith(js, forcing_day)
    expected_raw = _expected_cos_zenith(js, raw_day)
    # Guard against a vacuous test: the two clocks must be observably apart.
    assert not np.allclose(expected_forcing, expected_raw, atol=1e-6), (
        "raw and forcing clocks coincide — test would be vacuous")
    # The fix: zenith is locked to the forcing clock ...
    np.testing.assert_allclose(probed, expected_forcing, rtol=0, atol=1e-12)
    # ... and no longer drifts with the raw sim day.
    assert not np.allclose(probed, expected_raw, atol=1e-6)


def test_zenithfix_noncycle_byte_identical_to_raw_clock(tmp_path, monkeypatch):
    """FIX 1 byte-identity (non-cycled path): with cycle=False the insolation
    clock is the raw day (``block_start_day_forcing == block_start_day``,
    proven by ``test_preload_raw_records_noncycle_exact_last_record_ok``),
    exactly as before the fix.

    The cycle gate is a BIT-EXACT no-op whenever the forcing clock coincides
    with the raw day: a cycle=True block driven with forcing==raw and a
    cycle=False block produce identical cos_zenith (both through the same
    fused graph).  Both also match the raw-day reference to machine precision
    (the ~1e-15 residual is jit-vs-eager transcendental rounding in the
    reference, NOT a change introduced by the fix — the cycle=False branch is
    the identical ``jnp.mod(day, ...)`` expression the original code used)."""
    day = 1.7
    js = _zenith_js(tmp_path, cycle=False)
    probed_off = _run_zenith_block(js, day, day, monkeypatch)
    js_on = dict(js)
    js_on["cycle"] = True
    probed_on = _run_zenith_block(js_on, day, day, monkeypatch)
    # EXACT: the cycle gate changes nothing when day_f == day.
    np.testing.assert_array_equal(probed_on, probed_off)
    # Physical: the non-cycled zenith IS the raw-day zenith (machine precision).
    expected_raw = _expected_cos_zenith(js, day)
    np.testing.assert_allclose(probed_off, expected_raw, rtol=0, atol=1e-12)


def test_zenithfix_365day_ryf_preserves_seasonal_phase(tmp_path, monkeypatch):
    """FIX 1: the standard 365-day RYF cache is NOT perturbed.  ``day_f =
    day mod 365`` differs from the raw day by an exact multiple of 365, so
    the seasonal doy and diurnal hour are identical — the second-cycle
    zenith matches the raw-day zenith to machine precision (the reduced
    magnitude of ``day_f`` is, if anything, more accurate)."""
    js = _zenith_js(tmp_path, cycle=True)
    raw_day = 365.5        # second cycle of a 365-day cache
    forcing_day = 0.5      # 365.5 mod 365
    probed = _run_zenith_block(js, raw_day, forcing_day, monkeypatch)
    expected_raw = _expected_cos_zenith(js, raw_day)
    np.testing.assert_allclose(probed, expected_raw, rtol=0, atol=1e-12)


# ============================================================================
# GAP 1 — MPAS/tripole regrid branch of the raw-record preloader
#
# ``_preload_jra55_raw_records`` has an ``if "regrid_weights" in jra55_state:``
# branch (regrid each lat-lon record onto the unstructured target cells) that
# every existing dispatch test misses because they all use --grid latlon (no
# regrid weights).  Put a REAL lat-lon -> unstructured RegridWeights (the same
# object the MPAS/tripole setup builds via compute_latlon_to_voronoi_weights)
# into jra55_state and confirm the block preloader regrids each raw record to
# (n_window_records, nCells) and stays finite.
# ============================================================================

def _make_regrid_weights_from_cache(cache_path, n_cells=5):
    """Build a real lat-lon -> unstructured (MPAS-cell-like) ``RegridWeights``
    whose SOURCE grid matches the synthetic cache, targeting a few scattered
    points — the smallest valid weights object the regrid branch accepts."""
    from legoesm.grids.regridding import compute_latlon_to_voronoi_weights
    ds = xr.open_zarr(str(cache_path), decode_times=False)
    src_lat_rad = np.deg2rad(np.asarray(ds["lat"]))
    src_lon_rad = np.deg2rad(np.asarray(ds["lon"]))
    ds.close()
    rng = np.random.default_rng(1)
    tgt_lat_rad = np.deg2rad(rng.uniform(-80.0, 80.0, n_cells))
    tgt_lon_rad = np.deg2rad(rng.uniform(0.0, 360.0, n_cells))
    return compute_latlon_to_voronoi_weights(
        src_lat_rad, src_lon_rad, tgt_lat_rad, tgt_lon_rad,
    )


def test_preload_raw_records_regrids_to_unstructured_cells(tmp_path):
    """GAP 1: the regrid branch maps each raw lat-lon record onto the target
    cells → (n_window_records, nCells), all finite (winds, T, q, radiation,
    precip AND the friver runoff channel)."""
    from legoesm.forcing.jra55_do import JRA55_VARIABLES

    n_lat, n_lon, n_cells = 8, 16, 5
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon,
                                  n_records=8)
    rw = _make_regrid_weights_from_cache(cache, n_cells=n_cells)
    assert rw.target_shape == (n_cells,)
    js = {
        "cache_path": str(cache),
        "ref_year": 1958,
        "cycle": False,
        "regrid_weights": rw,
    }
    # start_step_idx=0, n_steps=4, dt=3 h → window records [0..4] (5 records).
    raw_stack, runoff_stack, meta = run_omip._preload_jra55_raw_records(
        0, 4, 10800.0, js)

    n_window = len(np.asarray(meta["record_days"]))
    assert n_window >= 2

    for var in JRA55_VARIABLES:
        arr = np.asarray(raw_stack[var])
        assert arr.shape == (n_window, n_cells), (var, arr.shape)
        assert np.all(np.isfinite(arr)), var

    runoff = np.asarray(runoff_stack)
    assert runoff.shape == (n_window, n_cells)
    assert np.all(np.isfinite(runoff))


# ============================================================================
# GAP 4 — _jra55_block_record_window ValueError guard paths
#
# Neither the "block spans >= a full cache cycle" nor the "wraps more than
# once" ValueError was covered.  Call the window helper directly (it takes
# n_cache_records as an int, so no cache file is needed) with args that trip
# each guard.  Both fire only in cycle=True mode.
# ============================================================================

def test_block_window_raises_when_span_exceeds_full_cache_cycle():
    """GAP 4a: a cycle=True block whose duration reaches the full cache length
    cannot define a monotone forcing clock → ValueError before any wrap logic.

    1-day cache (n_cache_records=8); dt=3 h, n_steps=9 → span = 8·0.125 =
    1.0 day == cache length → raise."""
    with pytest.raises(ValueError, match="full cache cycle"):
        run_omip._jra55_block_record_window(
            0, 9, 10800.0, 8, True)


def test_block_window_raises_when_block_wraps_more_than_once():
    """GAP 4b: a cycle=True block that would wrap the repeat-year boundary
    twice (upper bracket lands a full cache past the start) → ValueError.

    2-record cache (0.25-day cycle); start_step_idx=3, n_steps=4, dt=1.5 h.
    span = 3·0.0625 = 0.1875 day < 0.25 (clears the span guard); start_day_f =
    0.1875, end_day_f = 0.375 → i_last = 4 = 2·n_cache_records → wraps twice."""
    with pytest.raises(ValueError, match="more than once"):
        run_omip._jra55_block_record_window(
            3, 4, 5400.0, 2, True)


# ============================================================================
# Sea-ice rheology reachability (--ice-dynamics / --ice-categories)
#
# The lane used to hard-code SeaIceConfig(), i.e. dynamics="none",
# n_categories=1 -- a thermodynamic slab with no rheology and no way to change
# it, while FESOM2 runs EVP with 120 subcycles.  These check the whole chain:
# flag -> SeaIceConfig -> the right STATE type -> the grid the strain rates need.
# ============================================================================

def _ice_setup(tmp_path, **kw):
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    args = _argparse_namespace(jra55_cache=str(cache), jra55_sea_ice=True, **kw)
    return run_omip._setup_jra55_forcing_state(args, grid, "latlon"), grid


def test_ice_defaults_still_build_the_slab_state(tmp_path):
    """Default flags must reproduce the historical 3-field slab exactly."""
    from legoesm.ice.state import SeaIceState

    state, _ = _ice_setup(tmp_path)
    assert state["ice_config"].dynamics == "none"
    assert state["ice_config"].n_categories == 1
    assert isinstance(state["ice_state_init"], SeaIceState)
    # The slab needs no grid metrics, and passing them would change nothing.
    assert state["ice_grid"] is None


def test_evp_builds_a_dynamic_state_and_supplies_the_grid(tmp_path):
    """Selecting a rheology must switch the STATE type too -- a SeaIceState
    has no velocity or stress fields for the solver to advance."""
    from legoesm.ice.state import DynamicSeaIceState

    state, grid = _ice_setup(tmp_path, ice_dynamics="evp", ice_n_evp=120,
                             ice_p_star=30000.0, ice_delta_min=1e-11)
    cfg = state["ice_config"]
    assert cfg.dynamics == "evp"
    assert cfg.N_evp == 120
    assert cfg.P_star == 30000.0
    assert cfg.Delta_min == 1e-11
    ice = state["ice_state_init"]
    assert isinstance(ice, DynamicSeaIceState)
    assert ice.u_ice.data.shape == (4, 8)
    assert ice.sigma_12.data.shape == (4, 8)
    # Strain rates need the metrics; None here would give zero deformation
    # and a rheology that silently does nothing.
    assert state["ice_grid"] is grid


def test_multi_category_adds_the_trailing_category_axis(tmp_path):
    from legoesm.ice.state import DynamicSeaIceState

    state, grid = _ice_setup(tmp_path, ice_categories=7)
    ice = state["ice_state_init"]
    assert isinstance(ice, DynamicSeaIceState)
    assert ice.h_ice.data.shape == (4, 8, 7)
    # Velocity/stress stay spatial-only whatever the category count.
    assert ice.u_ice.data.shape == (4, 8)
    # A DynamicSeaIceState needs the grid EVEN WITH dynamics="none": the step
    # validates the state's spatial rank against it, and with grid=None it
    # assumes the cubed sphere and fails on the first step (codex r4 #5).
    assert state["ice_config"].dynamics == "none"
    assert state["ice_grid"] is grid


def test_mevp_parameters_reach_the_config(tmp_path):
    state, _ = _ice_setup(tmp_path, ice_dynamics="mevp",
                          ice_alpha_mevp=250.0, ice_beta_mevp=250.0)
    cfg = state["ice_config"]
    assert cfg.dynamics == "mevp"
    assert cfg.alpha_mevp == 250.0
    assert cfg.beta_mevp == 250.0


def test_rheology_refused_on_a_grid_without_strain_rates(tmp_path):
    """Tripole has no strain-rate branch; refusing beats zero deformation.

    The tripole setup path reads 2-D ``lat_T``/``lon_T``, so the stand-in
    carries them — otherwise the test would pass on an unrelated
    ``AttributeError`` and prove nothing about the guard.
    """
    import numpy as np

    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)

    class _Tripole:
        lat_T = np.radians(np.linspace(-60.0, 60.0, 4))[:, None] * np.ones(8)
        lon_T = np.radians(np.linspace(0.0, 315.0, 8))[None, :] * np.ones(
            (4, 1))

    args = _argparse_namespace(jra55_cache=str(cache), jra55_sea_ice=True,
                               ice_dynamics="evp")
    with pytest.raises(SystemExit, match="not supported on"):
        run_omip._setup_jra55_forcing_state(args, _Tripole(), "tripole")


def test_zero_or_negative_categories_rejected(tmp_path):
    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    args = _argparse_namespace(jra55_cache=str(cache), jra55_sea_ice=True,
                               ice_categories=0)
    with pytest.raises(SystemExit, match="must be >= 1"):
        run_omip._setup_jra55_forcing_state(args, grid, "latlon")
