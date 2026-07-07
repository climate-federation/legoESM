import numpy as np, numpy.testing as npt, jax.numpy as jnp
from legoesm.land.carbon.climate_features import (
    ClimateFeatures, reduce_climatology_to_features,
)
from legoesm.land.carbon.config import CarbonState
from legoesm.land.carbon.global_init import (
    ArchetypeTable, build_archetypes, equilibrate_archetypes, map_to_grid,
)

def _feats(mat, mapyr, seas, arid, sw):
    a = lambda v: np.asarray(v, float)
    return ClimateFeatures(a(mat), a(mapyr), a(seas), a(arid), a(sw))

def test_two_pft_two_climate_makes_expected_archetypes():
    # 4 cells: PFT0 in warm+cold, PFT1 in warm+cold.
    ncell, npft = 4, 2
    w = np.zeros((ncell, npft)); w[0, 0] = w[1, 0] = 1.0; w[2, 1] = w[3, 1] = 1.0
    feats = _feats([300, 270, 300, 270], [2000]*4, [3, 12, 3, 12], [1.5]*4, [250]*4)
    soil = np.array(["loam"] * ncell)
    tab, cid, cw = build_archetypes(w, feats, soil, np.ones(ncell, bool), k_per_pft=1, w_min=0.05, seed=0)
    # k=1 per PFT -> 2 archetypes (one per PFT), each averaging its climate.
    assert tab.pft_id.shape == (2,)
    assert set(np.unique(tab.pft_id)) == {0, 1}
    # Every occupied (cell,pft) maps to a valid archetype; empties are -1.
    assert (cid[w >= 0.05] >= 0).all() and (cid[w < 0.05] == -1).all()
    # Cover weights reproduce the input cover.
    npt.assert_allclose(cw, w, rtol=0, atol=0)

def test_determinism_same_seed():
    ncell, npft = 20, 3
    rng = np.random.default_rng(1); w = rng.random((ncell, npft)); w /= w.sum(1, keepdims=True)
    feats = _feats(rng.uniform(270, 305, ncell), rng.uniform(200, 3000, ncell),
                   rng.uniform(2, 15, ncell), rng.uniform(0.2, 3, ncell), rng.uniform(120, 300, ncell))
    soil = np.array(["loam"] * ncell)
    a = build_archetypes(w, feats, soil, np.ones(ncell, bool), k_per_pft=3, seed=7)
    b = build_archetypes(w, feats, soil, np.ones(ncell, bool), k_per_pft=3, seed=7)
    npt.assert_array_equal(a[1], b[1]); npt.assert_allclose(a[0].mat_k, b[0].mat_k)

def test_masked_cell_has_zero_weight_and_no_archetype():
    import numpy as np
    from legoesm.land.carbon.climate_features import ClimateFeatures
    from legoesm.land.carbon.global_init import build_archetypes
    a = lambda v: np.asarray(v, float)
    ncell, npft = 2, 1
    w = np.array([[0.9], [0.9]])                       # both cells have the PFT
    feats = ClimateFeatures(a([298, 283]), a([2000, 800]), a([3, 12]), a([2, 1]), a([230, 180]))
    soil = np.array(["loam", "loam"])
    mask = np.array([True, False])                     # cell 1 is NOT land
    tab, cid, cw = build_archetypes(w, feats, soil, mask, k_per_pft=1, w_min=0.05, seed=0)
    assert cid[1, 0] == -1                             # masked cell -> no archetype
    assert cw[1, 0] == 0.0                             # ...and zero weight (the invariant)
    # Invariant holds everywhere: weight is 0 exactly where id is -1.
    assert np.all((cid == -1) == (cw == 0.0))


def _cs(vals):  # vals: (n_arch,) per pool identical for simplicity
    a = lambda: jnp.asarray(vals, float)
    return CarbonState(a(), a(), a(), a(), a(), a())

def test_single_pft_cell_equals_archetype():
    eq = _cs([10.0, 20.0])
    cid = np.array([[0, -1], [1, -1]]); cw = np.array([[1.0, 0.0], [1.0, 0.0]])
    out = map_to_grid(cid, cw, eq)
    npt.assert_allclose(np.asarray(out.C_som), [10.0, 20.0], rtol=1e-9)

def test_mixed_cell_is_cover_weighted_mix():
    eq = _cs([10.0, 30.0])
    cid = np.array([[0, 1]]); cw = np.array([[0.25, 0.75]])
    out = map_to_grid(cid, cw, eq)
    npt.assert_allclose(np.asarray(out.C_som), [0.25 * 10 + 0.75 * 30], rtol=1e-9)

def test_absent_pft_contributes_zero_and_pools_nonneg():
    eq = _cs([10.0, 30.0])
    cid = np.array([[-1, 1]]); cw = np.array([[0.0, 0.5]])
    out = map_to_grid(cid, cw, eq)
    npt.assert_allclose(np.asarray(out.C_som), [0.5 * 30], rtol=1e-9)
    assert (np.asarray(out.C_som) >= 0).all()


# ---------------------------------------------------------------------------
# Integration gate for Tasks 2-5: features -> archetypes -> equilibrate -> map
# on a synthetic world (no data files).  Compute-node scale (JIT-compiles the
# coupled land+carbon model) -- run via the sbatch/srun wrapper, not login node.
# ---------------------------------------------------------------------------
def test_core_pipeline_synthetic_world():
    # 3 cells, 2 PFTs (tropical=4, temperate=7), simple climatology.
    ncell, npft = 3, 17
    w = np.zeros((ncell, npft)); w[0, 4] = 1.0; w[1, 7] = 1.0; w[2, 4] = 0.5; w[2, 7] = 0.5
    t = np.stack([np.full(12, 298.0), np.full(12, 283.0), np.full(12, 290.0)])
    pr = np.stack([np.full(12, 6e-5), np.full(12, 2.5e-5), np.full(12, 4e-5)])
    sw = np.full((ncell, 12), 220.0); nr = np.full((ncell, 12), 90.0)
    feats = reduce_climatology_to_features(t, pr, sw, nr)
    soil = np.array(["clay_loam", "loam", "loam"])
    tab, cid, cw = build_archetypes(w, feats, soil, np.ones(ncell, bool), k_per_pft=1, seed=0)
    eq, qc = equilibrate_archetypes(tab, n_spinup=15, n_verify=5, dt=7200.0, n_layers=6, soil_depth=2.0)
    grid = map_to_grid(cid, cw, eq)
    assert grid.C_som.shape == (ncell,)
    assert np.all(np.isfinite(np.asarray(grid.C_som)))
    # Mixed cell 2 is between the two pure cells for total ecosystem C.
    tot = lambda i: sum(float(np.asarray(getattr(grid, f))[i]) for f in grid._fields)
    assert min(tot(0), tot(1)) - 1.0 <= tot(2) <= max(tot(0), tot(1)) + 1.0


# ---------------------------------------------------------------------------
# The build_global_carbon_ic.py driver's full --dry-run-synthetic main() path:
# fabricate a tiny world -> features -> archetypes -> equilibrate -> map ->
# write both .npz.  Also compute-node scale (JIT) -- run via the srun wrapper.
# ---------------------------------------------------------------------------
def test_driver_dry_run_synthetic_writes_npz(tmp_path):
    import importlib.util
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[3]
    py = repo / "scripts" / "data" / "build_global_carbon_ic.py"
    spec = importlib.util.spec_from_file_location("build_global_carbon_ic", py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = tmp_path / "gcic"
    finidat_path, arch_path = mod.main([
        "--dry-run-synthetic", "--output", str(out),
        "--k-per-pft", "1", "--n-spinup", "6", "--n-verify", "3",
        "--dt", "7200", "--n-layers", "6", "--soil-depth", "2.0",
    ])
    assert finidat_path.exists() and arch_path.exists()
    assert finidat_path.name == "global_carbon_ic.npz"
    assert arch_path.name == "archetypes.npz"

    with np.load(finidat_path, allow_pickle=False) as d:
        # 6-cell, 17-PFT synthetic world.
        for pool in CarbonState._fields:
            assert d[pool].shape == (6,), pool
            assert np.all(np.isfinite(d[pool]))
        assert d["lat"].shape == (6,) and d["lon"].shape == (6,)
        assert d["dominant_pft"].shape == (6,)
        assert d["pft_present"].shape == (6, 17)
        assert d["pft_weights"].shape == (6, 17)

    with np.load(arch_path, allow_pickle=False) as d:
        n_arch = d["pft_id"].shape[0]
        assert n_arch >= 1
        assert d["soil_class"].shape == (n_arch,)
        for pool in CarbonState._fields:
            assert d[f"eq_{pool}"].shape == (n_arch,)
        for key in ("gpp", "npp", "som_kgC", "biomass_kgC", "drift_frac_per_yr"):
            assert d[f"qc_{key}"].shape == (n_arch,)
            assert np.all(np.isfinite(d[f"qc_{key}"]))
        # Task-8h: soil-column geometry is persisted for the drift validator.
        assert int(d["n_layers"]) == 6
        assert float(d["soil_depth"]) == 2.0
        assert float(d["dt"]) == 7200.0


# ===========================================================================
# Task-8h: driver hardening (PFT-axis alignment, geometry persistence,
# zonal-climate mode).  These load the build_global_carbon_ic.py driver.
# ===========================================================================
def _load_driver():
    import importlib.util
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[3]
    py = repo / "scripts" / "data" / "build_global_carbon_ic.py"
    spec = importlib.util.spec_from_file_location("build_global_carbon_ic", py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_align_pft_axis_pads_and_errors():
    from legoesm.land.surface_params import N_PFT_CLM5
    drv = _load_driver()
    ncell = 4
    # 17-PFT source passes through unchanged.
    w17 = np.random.default_rng(0).random((ncell, N_PFT_CLM5))
    npt.assert_array_equal(drv._align_pft_axis(w17, "t"), w17)
    # 15-PFT natural-only source zero-pads the two crop columns (indices 15,16).
    w15 = np.random.default_rng(1).random((ncell, N_PFT_CLM5 - 2))
    out = drv._align_pft_axis(w15, "t")
    assert out.shape == (ncell, N_PFT_CLM5)
    npt.assert_array_equal(out[:, :N_PFT_CLM5 - 2], w15)
    npt.assert_array_equal(out[:, N_PFT_CLM5 - 2:], 0.0)
    # Any other count is a real mismatch -> raise, never silent.
    import pytest
    with pytest.raises(ValueError):
        drv._align_pft_axis(np.zeros((ncell, N_PFT_CLM5 - 1)), "t")


def test_geometry_round_trips_through_archetypes_npz(tmp_path):
    drv = _load_driver()
    n_arch = 2
    table = ArchetypeTable(
        pft_id=np.array([4, 13]),
        mat_k=np.array([298.0, 285.0]), map_yr=np.array([2000.0, 900.0]),
        t_seasonal_amp_k=np.array([3.0, 10.0]), aridity=np.array([2.0, 1.2]),
        sw_mean_w=np.array([230.0, 190.0]),
        soil_class=np.array(["clay_loam", "loam"], dtype=object))
    eq = CarbonState(**{f: jnp.arange(n_arch, dtype=float) + 1.0
                        for f in CarbonState._fields})
    qc = {k: np.zeros(n_arch)
          for k in ("gpp", "npp", "som_kgC", "biomass_kgC", "drift_frac_per_yr")}
    path = tmp_path / "archetypes.npz"
    drv._write_archetypes_npz(
        path, table, eq, qc, ["bare_soil"], n_layers=7, soil_depth=2.5,
        dt=1800.0, res_deg=2.0)
    with np.load(path, allow_pickle=False) as d:
        assert int(d["n_layers"]) == 7
        assert float(d["soil_depth"]) == 2.5
        assert float(d["dt"]) == 1800.0
        assert float(d["resolution_deg"]) == 2.0
        # Table + equilibria still round-trip alongside the geometry.
        npt.assert_array_equal(d["pft_id"], [4, 13])
        for f in CarbonState._fields:
            npt.assert_allclose(d[f"eq_{f}"], np.arange(n_arch) + 1.0)


def test_zonal_climate_varies_with_latitude():
    # A tropical (0 deg) vs polar (80 deg) cell must differ in mean-annual T and
    # seasonal amplitude after reduce_climatology_to_features -- the zonal climate
    # is genuinely latitude-dependent, reusing the lmip_forcing latitude pieces.
    drv = _load_driver()
    lat = np.array([0.0, 80.0])
    t, pr, sw, nr = drv.zonal_monthly_climate(lat)
    assert t.shape == (2, 12) and sw.shape == (2, 12)
    feats = reduce_climatology_to_features(t, pr, sw, nr)
    mat = np.asarray(feats.mat_k)
    seas = np.asarray(feats.t_seasonal_amp_k)
    swm = np.asarray(feats.sw_mean_w)
    assert mat[0] > mat[1] + 5.0            # tropics warmer than the pole
    assert seas[1] > seas[0] + 2.0          # pole has a larger seasonal cycle
    assert swm[0] > swm[1]                   # tropics get more annual-mean SW
    assert np.all(np.isfinite(np.concatenate([t, pr, sw, nr], axis=1)))
