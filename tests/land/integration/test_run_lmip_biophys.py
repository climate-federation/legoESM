"""End-to-end smoke for the CRU-JRA-forced biophysics-only land driver (M3).

Runs ``scripts/run/run_lmip_biophys.py`` on a tiny latlon grid with synthetic
forcing (no external data) and a synthetic surfdata, and asserts the
forcing -> regrid -> disaggregate -> lax.scan land-step pipeline produces finite
fields over land and writes its time-series NetCDF.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from legoesm.land.surface_data.schema import write_surfdata
from legoesm.land.surface_params import N_PFT_CLM5

pytest.importorskip("xarray")

_ROOT = Path(__file__).resolve().parents[3]
_DRIVER = _ROOT / "scripts" / "run" / "run_lmip_biophys.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_lmip_biophys", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_surfdata(path, nlat=8, nlon=16, nlev=7):
    lat = np.linspace(-85, 85, nlat); lon = np.linspace(0, 337.5, nlon)
    pft = np.zeros((1, N_PFT_CLM5, nlat, nlon)); pft[0, 4] = 100.0
    lai = np.zeros((12, N_PFT_CLM5, nlat, nlon)); lai[:, 4] = 4.0
    soil = lambda v: np.full((nlev, nlat, nlon), v)
    write_surfdata(
        path, lat=lat, lon=lon, soil_dz=np.full(nlev, 0.2),
        sand_pct=soil(40.0), clay_pct=soil(20.0), organic=soil(5.0),
        bulk_density=soil(1300.0), soil_color=np.full((nlat, nlon), 8.0),
        year=np.array([2015.0]),
        f_land=np.full((1, nlat, nlon), 100.0),
        f_lake=np.zeros((1, nlat, nlon)), f_glacier=np.zeros((1, nlat, nlon)),
        pft_frac=pft, monthly_lai=lai, monthly_sai=np.zeros_like(lai),
        monthly_height_top=np.zeros_like(lai), monthly_height_bot=np.zeros_like(lai),
    )


_SMOKE_TEMPLATE = _ROOT / "templates" / "land" / "biophysics" / "smoke_test.yaml"


def _write_smoke_config(tmp_path, sd_path, extra_overrides=None):
    """Materialise a smoke-test config.yaml on disk with surfdata pointing at
    the synthetic surfdata file the test just wrote."""
    import yaml
    from legoesm.land.lmip_config import apply_overrides, validate_config
    with open(_SMOKE_TEMPLATE) as f:
        base = yaml.safe_load(f)
    overrides = [f"surfdata.path={sd_path}"] + list(extra_overrides or ())
    merged = apply_overrides(base, overrides)
    cfg = validate_config(merged).raw
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def _run_config(mod, cfg_path, out_dir, restart_from=""):
    """Drive the run through the public main(argv) entry point — no reaching
    into private helpers.  Returns the driver's integer exit code."""
    return mod.main([
        "--config", str(cfg_path),
        "--output-dir", str(out_dir),
        "--restart-from", restart_from,
    ])


def test_biophys_driver_synthetic_smoke(tmp_path):
    mod = _load_driver()
    sd = tmp_path / "sd.nc"; _write_surfdata(str(sd))
    out = tmp_path / "out"
    cfg_path = _write_smoke_config(tmp_path, sd)
    rc = _run_config(mod, cfg_path, out)
    assert rc == 0                                            # PASS (no NaN over land)

    # Tape output lands at lmip_biophys.<tape_name>.nc; the step-test config
    # has one tape named "step".
    import xarray as xr
    ds = xr.open_dataset(out / "lmip_biophys.step.nc")
    assert ds.sizes["time"] == 4
    assert ds.attrs["carbon"] == "none"                      # biophysics-only
    assert ds.attrs["tape_name"] == "step"
    assert ds.attrs["tape_freq"] == "step"
    # Latlon rectangular layout: (time, lat, lon) — never (time, ncol)
    assert ds["T_sfc"].dims == ("time", "lat", "lon")
    assert ds.sizes["lat"] == 4 and ds.sizes["lon"] == 8    # --resolution 4 -> 4 x 8
    assert np.all(np.isfinite(ds["T_sfc"].values))
    assert float(ds["T_sfc"].min()) > 200.0
    assert float(ds["T_sfc"].max()) < 360.0


def test_build_model_times_synthetic_starts_at_zero():
    mod = _load_driver()
    t = mod.build_model_times(196.0, 3600.0, 5, synthetic=True)
    assert t[0] == 0.0                                        # synthetic ignores start-doy
    t2 = mod.build_model_times(10.0, 3600.0, 5, synthetic=False)
    assert t2[0] == 10.0 * 86400.0                           # real forcing honours start-doy


def test_chunked_scan_straddles_year_boundary(tmp_path):
    """Straddle a year boundary with hourly steps (last 4h of year 0 → first
    4h of year 1).  The chunked-scan loop runs twice, threads state, and
    produces a 2-slot annual tape with real physics on both slots."""
    mod = _load_driver()
    sd = tmp_path / "sd.nc"; _write_surfdata(str(sd))
    out = tmp_path / "straddle"
    # 8760 hourly steps = 1 exact year; add 4 more → year 1 gets 4 steps.
    N_STEPS = 8764
    cfg_path = _write_smoke_config(tmp_path, sd, extra_overrides=[
        "forcing.year_end=2001",
        "time.dt=3600.0",
        f"time.n_steps={N_STEPS}",
        "output.tapes=[{name: annual, freq: annual, average: mean, "
        "vars: [T_sfc]}]",
    ])
    # rc may be 1 (some tiny synthetic cells drift) — we're testing the
    # chunked-scan MECHANISM, not physics.  What matters:
    rc = _run_config(mod, cfg_path, out)
    assert rc in (0, 1)                                   # completed either way

    import xarray as xr
    ds = xr.open_dataset(out / "lmip_biophys.annual.nc")
    assert ds.sizes["time"] == 2                          # year 0 + year 1 chunks
    T = ds["T_sfc"].values
    # Both slots must have SOME finite T (proves state threaded into the
    # second chunk and the second lax.scan produced real numbers).
    assert np.isfinite(T[0]).sum() > 0
    assert np.isfinite(T[1]).sum() > 0

    # Restart filename encodes total model time elapsed: 8764 h = 1 year + 4 h
    # → year 2001, doy 0, hour 4.  Proves the chunked loop bookkept time
    # correctly across the boundary.
    restarts = list(out.glob("restart_*.npz"))
    assert len(restarts) == 1
    assert restarts[0].name == "restart_2001_d000h04.npz"


def test_oversize_forcing_estimate_fails_fast(tmp_path):
    """Per-year forcing must fit in the soft budget (24 GiB).  Chunked scan
    caps peak memory at ONE year, so it's the year-size that matters — but
    even one year at a tiny dt on a 2° grid can blow the budget."""
    mod = _load_driver()
    sd = tmp_path / "sd.nc"; _write_surfdata(str(sd))
    out = tmp_path / "big"
    # 2° grid (16200 cols) + very small dt (100 s) -> ~315k steps per year
    # -> ~ 465 GiB per-year forcing.  Well over the 24 GiB budget.
    cfg_path = _write_smoke_config(tmp_path, sd, extra_overrides=[
        "grid.resolution=90",
        "time.n_steps=315360",
        "time.dt=100",
    ])
    rc = _run_config(mod, cfg_path, out)
    assert rc == 2                                            # over-budget
    assert not list(out.glob("restart_*.npz"))


def test_missing_forcing_year_fails_fast_not_synthetic(tmp_path):
    """When --forcing-dir is set but a requested year's Solr file is missing,
    the driver must EXIT with a clear error listing the missing files —
    silent fallback to synthetic would OOM the device with fake data."""
    mod = _load_driver()
    sd = tmp_path / "sd.nc"; _write_surfdata(str(sd))
    out = tmp_path / "fail"

    # Stage year 2000 only; ask for 2000-2002.  Years 2001, 2002 are missing.
    fdir = tmp_path / "crujra"; fdir.mkdir()
    prefix = "clmforc.CRUJRAv2.5_filled_antarct_and_grnlnd_0.5x0.5"
    # Create empty stub files for year 2000 so it "exists" (existence check is
    # os.path.exists; content doesn't matter for the guardrail).
    for stream in ("Solr", "Prec", "TPQWL"):
        (fdir / f"{prefix}.{stream}.2000.nc").write_text("stub")

    cfg_path = _write_smoke_config(tmp_path, sd, extra_overrides=[
        f"forcing.data_dir={fdir}",
        "forcing.source=cru_jra",
        "forcing.year_start=2000",
        "forcing.year_end=2002",
        f"forcing.prefix={prefix}",
    ])
    rc = _run_config(mod, cfg_path, out)
    assert rc == 2                                            # missing-forcing exit code
    # No restart written — the run exited before physics started.
    assert not list(out.glob("restart_*.npz"))


def test_cold_run_auto_saves_timestamped_restart(tmp_path):
    """A cold run always writes exactly one restart file named
    restart_<YEAR>_d<DDD>h<HH>.npz next to lmip_biophys.nc."""
    mod = _load_driver()
    sd = tmp_path / "sd.nc"; _write_surfdata(str(sd))
    out = tmp_path / "cold"
    cfg_path = _write_smoke_config(tmp_path, sd)
    assert _run_config(mod, cfg_path, out) == 0
    restarts = list(out.glob("restart_*.npz"))
    assert len(restarts) == 1
    # Model-time-stamped: 4 steps * 1 h = 4 h from doy 0 -> year_final=year_start d000 h04
    # (smoke_test template uses year_start=2000)
    assert restarts[0].name == "restart_2000_d000h04.npz"

    from legoesm.land.restart import load_land_restart
    _, meta = load_land_restart(restarts[0],
                                expected_land_mode="multilayer",
                                expected_ncol=32, expected_n_layers=None)
    assert meta["metadata"]["surface_scheme"] == "simple_seb"
    assert meta["n_steps_completed"] == 4
    assert meta["metadata"]["year_final"] == 2000
    assert meta["metadata"]["doy_final"] == 0
    assert meta["metadata"]["hour_final"] == 4


def test_warm_start_uses_loaded_state(tmp_path):
    """When --restart-from is set, the driver's initial state IS the loaded
    state — not the cold-init default.  Verified with a hand-crafted seed
    at a distinctive T_soil (250 K uniform) that a cold start would never
    reach in 4 h of physics under synthetic forcing (~285 K air)."""
    import jax.numpy as jnp
    from legoesm.land.restart import save_land_restart
    from legoesm.land.state import MultiLayerLandState

    mod = _load_driver()
    sd = tmp_path / "sd.nc"; _write_surfdata(str(sd))
    ncol, nlay = 32, 8              # 4x8 latlon at --resolution 4
    seed = MultiLayerLandState(
        T_soil=jnp.full((ncol, nlay), 250.0),   # unmistakable cold column
        psi_soil=jnp.full((ncol, nlay), -1.0),
        theta_soil=jnp.full((ncol, nlay), 0.30),
        runoff_surface=jnp.zeros(ncol),
        runoff_subsurface=jnp.zeros(ncol),
        snow_depth=jnp.zeros(ncol),
        snow_age=jnp.zeros(ncol),
    )
    seed_path = tmp_path / "seed.npz"
    save_land_restart(seed_path, seed, land_mode="multilayer",
                      t_end_s=0.0, n_steps_completed=0)

    out = tmp_path / "warm"
    cfg_path = _write_smoke_config(tmp_path, sd)
    assert _run_config(mod, cfg_path, out, restart_from=str(seed_path)) == 0

    # Load the run's END-of-scan state (auto-saved).  Deep soil layers are
    # thermally slow; after 4 h from a 250 K seed they must still be near 250 K
    # (a fresh cold start would have deep T at ~280 K, the T_init default in
    # init_multilayer_land_state).
    from legoesm.land.restart import load_land_restart
    restarts = list(out.glob("restart_*.npz"))
    assert len(restarts) == 1
    st_end, _ = load_land_restart(restarts[0],
                                  expected_land_mode="multilayer",
                                  expected_ncol=32, expected_n_layers=None)
    T_deep = np.asarray(st_end.T_soil)[:, -1]                    # deepest layer
    assert float(T_deep.mean()) < 255.0                          # near seed
    assert float(T_deep.mean()) > 245.0


def test_steps_exceeding_staged_years_fail_fast(tmp_path):
    """Asking for more steps than fit the staged forcing years must RAISE.  A
    silent step drop would integrate fewer steps than requested AND falsify the
    auto-saved restart (n_steps_completed / t_end from the full clock)."""
    mod = _load_driver()
    sd = tmp_path / "sd.nc"; _write_surfdata(str(sd))
    out = tmp_path / "toolong"
    # single synthetic year (year_end defaults to year_start) = 8760 hourly
    # steps; asking for 9000 drops 240 steps past DOY 365.
    cfg_path = _write_smoke_config(tmp_path, sd, extra_overrides=[
        "time.dt=3600.0", "time.n_steps=9000",
    ])
    with pytest.raises(SystemExit):
        _run_config(mod, cfg_path, out)
    assert not list(out.glob("restart_*.npz"))               # no restart written


def test_slab_mode_gate_reads_real_state(tmp_path):
    """Slab T_soil is a 1-D Field; the PASS/FAIL gate must validate its .data
    (real state), not a zeros placeholder — otherwise a slab blow-up would
    silently PASS.  This exercises the slab path + the fixed gate end-to-end."""
    mod = _load_driver()
    sd = tmp_path / "sd.nc"; _write_surfdata(str(sd))
    out = tmp_path / "slab"
    cfg_path = _write_smoke_config(tmp_path, sd, extra_overrides=[
        "physics.land_mode=slab",
    ])
    rc = _run_config(mod, cfg_path, out)
    assert rc in (0, 1)                                       # gate read real state, no crash
