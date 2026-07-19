"""End-to-end for the offline land spin-up driver + the coupled land-IC path (#746).

Runs ``scripts/run/run_land_spinup.py`` on a tiny cubed-sphere grid with a
synthetic surfdata (no external data): a few short "years" of the multilayer
land under the idealised seasonal cycle, then asserts it (a) writes an
equilibrium-metrics JSON with a per-year drift trajectory and a verdict, and
(b) writes a MultiLayerLandState restart that ``load_land_restart`` (and hence
the coupled AMIP ``--land-ic`` path) can consume with matching ncol/n_layers.
The chained ``--restart-from`` resume is exercised too.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from legoesm.land.surface_data.schema import write_surfdata
from legoesm.land.surface_params import N_PFT_CLM5

pytest.importorskip("xarray")

_ROOT = Path(__file__).resolve().parents[3]
_DRIVER = _ROOT / "scripts" / "run" / "run_land_spinup.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_land_spinup", _DRIVER)
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


# A STABLE dt (1800 s — the soil column blows up at multi-hour dt) with a SHORT
# 1-day "year" (48 steps/year) keeps this WIRING test fast while exercising the
# real multi-year outer loop + inner lax.scan + drift metrics + restart I/O.
_YR = 86400.0                        # a 1-day spin-up "year" for the test
_FAST = ["--dt", "1800.0", "--seconds-per-year", str(_YR)]
_N_LAYERS = 8                        # default SoilGridConfig n_layers


def _run(mod, argv):
    return mod.main(argv)


def test_land_spinup_writes_metrics_and_restart(tmp_path):
    mod = _load_driver()
    sd = tmp_path / "sd.nc"; _write_surfdata(str(sd))
    out = tmp_path / "spin"
    rc = _run(mod, [
        "--surfdata", str(sd), "--grid-type", "cubed_sphere", "--resolution", "2",
        "--years", "3", "--output", str(out),
        "--checkpoint-every-years", "0", *_FAST,
    ])
    assert rc == 0

    # (a) equilibrium metrics: a per-year drift trajectory + a verdict.
    metrics = json.loads((out / "spinup_metrics.json").read_text())
    assert metrics["years"] == 3 and len(metrics["per_year"]) == 3
    assert metrics["verdict"] in ("CONVERGED", "NOT_CONVERGED")
    for row in metrics["per_year"]:
        assert np.isfinite(row["drift_T_K_per_yr"])
        assert np.isfinite(row["T_soil_land_mean"])
    # The soil-T drift should SHRINK as the column settles (year 3 <= year 1).
    d1 = abs(metrics["per_year"][0]["drift_T_K_per_yr"])
    d3 = abs(metrics["per_year"][-1]["drift_T_K_per_yr"])
    assert d3 <= d1 + 1e-6

    # (b) a MultiLayerLandState restart the coupled --land-ic path can load.
    from legoesm.land.restart import load_land_restart
    ic = out / "land_ic.npz"
    assert ic.exists()
    grid = mod._load_smoke().make_grid("cubed_sphere", 2)
    ncol = 6 * 2 * 2
    state, meta = load_land_restart(
        str(ic), expected_land_mode="multilayer",
        expected_ncol=ncol, expected_n_layers=_N_LAYERS)
    assert meta["t_end_s"] == pytest.approx(3 * _YR)
    assert np.all(np.isfinite(np.asarray(state.T_soil)))
    assert np.asarray(state.T_soil).shape == (ncol, _N_LAYERS)


def test_land_spinup_voronoi_grid_pipeline():
    """A3: the standalone land drivers support the Voronoi/MPAS mesh.  ``make_grid``
    now builds a VoronoiMesh, and the two grid-facing helpers the land columns
    depend on — ``grid_latlon_rad`` (per-column lat/lon) and the surfdata regrid
    target ``_target_latlon_flat`` — are already grid-agnostic (both read
    ``latCell``/``lonCell``), so the per-column land model runs on ``nCells``
    columns exactly like any gridded columns.  This validates the voronoi grid
    plumbing (the A3 change); the full end-to-end soil/canopy step is exercised
    grid-agnostically by the cubed_sphere spin-up test above."""
    from legoesm.land.global_surface_data import _target_latlon_flat
    from legoesm.grids.voronoi import VoronoiMesh
    smoke = _load_driver()._load_smoke()
    grid = smoke.make_grid("voronoi", 2)              # SCVT level-2 mesh
    assert isinstance(grid, VoronoiMesh)
    lat, lon = smoke.grid_latlon_rad(grid)
    ncol = int(lat.shape[0])
    assert ncol > 0 and bool(np.isfinite(np.asarray(lat)).all())
    assert -np.pi / 2 - 1e-6 <= float(lat.min())
    assert float(lat.max()) <= np.pi / 2 + 1e-6
    tlat, tlon, shp = _target_latlon_flat(grid)       # surfdata regrids to these
    assert tlat.shape[0] == ncol and tuple(shp) == (ncol,)


def test_land_spinup_restart_resume_advances_calendar(tmp_path):
    """--restart-from resumes at the prior t_end_s and advances the year index
    (the chained long-spin-up path)."""
    mod = _load_driver()
    sd = tmp_path / "sd.nc"; _write_surfdata(str(sd))
    out1 = tmp_path / "s1"
    assert _run(mod, [
        "--surfdata", str(sd), "--grid-type", "cubed_sphere", "--resolution", "2",
        "--years", "2", "--output", str(out1),
        "--restart-out", str(out1 / "ic.npz"),
        "--checkpoint-every-years", "0", *_FAST,
    ]) == 0

    out2 = tmp_path / "s2"
    assert _run(mod, [
        "--surfdata", str(sd), "--grid-type", "cubed_sphere", "--resolution", "2",
        "--years", "2", "--output", str(out2),
        "--restart-from", str(out1 / "ic.npz"),
        "--restart-out", str(out2 / "ic.npz"),
        "--checkpoint-every-years", "0", *_FAST,
    ]) == 0
    m2 = json.loads((out2 / "spinup_metrics.json").read_text())
    # Resumed run's years are 3 and 4 (continued from the 2-year restart).
    assert [r["year"] for r in m2["per_year"]] == [3, 4]
    from legoesm.land.restart import load_land_restart
    _s, meta = load_land_restart(
        str(out2 / "ic.npz"), expected_land_mode="multilayer",
        expected_ncol=6 * 2 * 2, expected_n_layers=_N_LAYERS)
    assert meta["t_end_s"] == pytest.approx(4 * _YR)
