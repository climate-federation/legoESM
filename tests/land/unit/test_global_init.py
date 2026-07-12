import numpy as np, numpy.testing as npt, jax.numpy as jnp
from legoesm.land.carbon.climate_features import (
    ClimateFeatures, reduce_climatology_to_features,
)
from legoesm.land.carbon.config import CarbonState
from legoesm.land.carbon.global_init import (
    ArchetypeTable, build_archetypes, equilibrate_archetypes, map_to_grid,
    map_to_grid_frozen_fraction,
)

def _feats(mat, mapyr, seas, arid, sw):
    a = lambda v: np.asarray(v, float)
    return ClimateFeatures(a(mat), a(mapyr), a(seas), a(arid), a(sw))

def test_two_pft_two_climate_makes_expected_archetypes():
    # 4 cells: PFT1 in warm+cold, PFT2 in warm+cold (col 0 = bare, excluded).
    ncell, npft = 4, 3
    w = np.zeros((ncell, npft)); w[0, 1] = w[1, 1] = 1.0; w[2, 2] = w[3, 2] = 1.0
    feats = _feats([300, 270, 300, 270], [2000]*4, [3, 12, 3, 12], [1.5]*4, [250]*4)
    soil = np.array(["loam"] * ncell)
    tab, cid, cw = build_archetypes(w, feats, soil, np.ones(ncell, bool), k_per_pft=1, w_min=0.05, seed=0)
    # k=1 per PFT -> 2 archetypes (one per non-bare PFT), each averaging climate.
    assert tab.pft_id.shape == (2,)
    assert set(np.unique(tab.pft_id)) == {1, 2}
    # Every occupied (cell,pft) maps to a valid archetype; empties + bare are -1.
    assert (cid[w >= 0.05] >= 0).all() and (cid[w < 0.05] == -1).all()
    # Cover weights reproduce the input cover (bare column is zeroed but w[:,0]=0).
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
    ncell, npft = 2, 2                                 # col 0 = bare (excluded)
    w = np.array([[0.0, 0.9], [0.0, 0.9]])             # both cells have PFT 1
    feats = ClimateFeatures(a([298, 283]), a([2000, 800]), a([3, 12]), a([2, 1]), a([230, 180]))
    soil = np.array(["loam", "loam"])
    mask = np.array([True, False])                     # cell 1 is NOT land
    tab, cid, cw = build_archetypes(w, feats, soil, mask, k_per_pft=1, w_min=0.05, seed=0)
    assert cid[1, 1] == -1                             # masked cell -> no archetype
    assert cw[1, 1] == 0.0                             # ...and zero weight (the invariant)
    assert cid[0, 1] >= 0                              # the land cell DID cluster
    # Invariant holds everywhere (incl. the excluded bare col): weight is 0
    # exactly where id is -1.
    assert np.all((cid == -1) == (cw == 0.0))


def test_bare_cover_yields_no_archetype_and_zero_pools():
    # F2: a cell dominated by bare ground (>= w_min) plus a real PFT.  Bare must
    # NOT produce an archetype and must contribute ZERO carbon in the map.
    ncell, npft = 2, 3
    w = np.zeros((ncell, npft))
    w[0, 0] = 0.7; w[0, 1] = 0.3      # cell 0: 70% bare + 30% PFT 1
    w[1, 1] = 1.0                     # cell 1: all PFT 1
    feats = _feats([295, 290], [1500, 1200], [6, 8], [1.2, 1.0], [220, 210])
    soil = np.array(["loam", "loam"])
    tab, cid, cw = build_archetypes(
        w, feats, soil, np.ones(ncell, bool), k_per_pft=1, w_min=0.05, seed=0)
    # No bare (pft 0) archetype was created.
    assert 0 not in set(np.unique(tab.pft_id).tolist())
    assert set(np.unique(tab.pft_id).tolist()) == {1}
    # Bare column carries no archetype id and no weight anywhere.
    assert (cid[:, 0] == -1).all()
    assert (cw[:, 0] == 0.0).all()
    # map_to_grid: bare contributes zero even though cell 0 is 70% bare.
    n_arch = int(tab.pft_id.shape[0])
    eq = CarbonState(**{f: jnp.asarray([100.0] * n_arch)
                        for f in CarbonState._fields})
    out = map_to_grid(cid, cw, eq)
    # Cell 0's pools come ONLY from PFT 1's 0.3 cover (bare's 0.7 adds nothing).
    npt.assert_allclose(np.asarray(out.C_som_active)[0], 0.3 * 100.0, rtol=1e-9)
    npt.assert_allclose(np.asarray(out.C_som_active)[1], 1.0 * 100.0, rtol=1e-9)


def test_archetype_climate_mean_is_cover_weighted():
    # F4: one PFT, two cells clustered together (k=1) -- a DOMINANT-cover cell
    # and a TRACE-cover cell with very different climate.  The archetype's
    # climate mean must sit near the dominant cell (cover-weighted), NOT at the
    # midpoint an unweighted mean would give.
    ncell, npft = 2, 2                # col 0 bare, PFT at col 1
    w = np.zeros((ncell, npft))
    w[0, 1] = 0.95                    # dominant cover, warm cell
    w[1, 1] = 0.06                    # trace cover (>= w_min), cold cell
    feats = _feats([300.0, 270.0], [2000, 2000], [4, 4], [1.5, 1.5], [250, 250])
    soil = np.array(["loam", "loam"])
    tab, cid, cw = build_archetypes(
        w, feats, soil, np.ones(ncell, bool), k_per_pft=1, w_min=0.05, seed=0)
    assert tab.pft_id.shape == (1,)
    mat = float(tab.mat_k[0])
    unweighted = 0.5 * (300.0 + 270.0)                       # 285.0
    cover_weighted = (0.95 * 300.0 + 0.06 * 270.0) / (0.95 + 0.06)
    npt.assert_allclose(mat, cover_weighted, rtol=1e-9)
    # Much closer to the dominant (warm) cell than the unweighted mean would be.
    assert mat > unweighted + 5.0


def test_dropped_cover_fraction_audit():
    # F3: sub-w_min NON-BARE cover is dropped from the map; the audit surfaces it.
    from legoesm.land.carbon.global_init import dropped_cover_fraction
    ncell, npft = 2, 3
    w = np.zeros((ncell, npft))
    w[0, 1] = 0.5; w[0, 2] = 0.02     # cell 0: PFT1 kept, PFT2 dropped (< w_min)
    w[1, 1] = 0.01; w[1, 2] = 0.03    # cell 1: both dropped
    mean_drop, max_drop = dropped_cover_fraction(
        w, np.ones(ncell, bool), w_min=0.05)
    # dropped (non-bare, 0 < cover < w_min): cell0 -> 0.02, cell1 -> 0.04.
    npt.assert_allclose(max_drop, 0.04, rtol=1e-9)
    npt.assert_allclose(mean_drop, 0.03, rtol=1e-9)
    # No land cells -> (0, 0), never a divide-by-zero.
    assert dropped_cover_fraction(w, np.zeros(ncell, bool)) == (0.0, 0.0)


def _cs(vals):  # vals: (n_arch,) per pool identical for simplicity
    a = lambda: jnp.asarray(vals, float)
    # 8 positional args (6->8 after the SOM split); no CarbonState defaults, so
    # a missed arg fails loudly with TypeError.
    return CarbonState(a(), a(), a(), a(), a(), a(), a(), a())

def test_single_pft_cell_equals_archetype():
    eq = _cs([10.0, 20.0])
    cid = np.array([[0, -1], [1, -1]]); cw = np.array([[1.0, 0.0], [1.0, 0.0]])
    out = map_to_grid(cid, cw, eq)
    npt.assert_allclose(np.asarray(out.C_som_active), [10.0, 20.0], rtol=1e-9)

def test_mixed_cell_is_cover_weighted_mix():
    eq = _cs([10.0, 30.0])
    cid = np.array([[0, 1]]); cw = np.array([[0.25, 0.75]])
    out = map_to_grid(cid, cw, eq)
    npt.assert_allclose(np.asarray(out.C_som_active), [0.25 * 10 + 0.75 * 30], rtol=1e-9)

def test_absent_pft_contributes_zero_and_pools_nonneg():
    eq = _cs([10.0, 30.0])
    cid = np.array([[-1, 1]]); cw = np.array([[0.0, 0.5]])
    out = map_to_grid(cid, cw, eq)
    npt.assert_allclose(np.asarray(out.C_som_active), [0.5 * 30], rtol=1e-9)
    assert (np.asarray(out.C_som_active) >= 0).all()


# ---------------------------------------------------------------------------
# map_to_grid_frozen_fraction: cover-weighted per-cell permafrost index phi that
# the coupled run threads into step_multilayer_land to preserve seeded permafrost
# SOC.  phi is INTENSIVE (cover-weighted MEAN, unlike the extensive pool SUM) and
# BYTE-IDENTICAL to the per-archetype phi the spin-up applied (IC-consistency).
# ---------------------------------------------------------------------------
def _phi_table(mat_k, t_seasonal_amp_k):
    n = len(mat_k)
    return ArchetypeTable(
        pft_id=np.arange(n) % 16 + 1,   # any non-bare ids (phi ignores pft)
        mat_k=np.asarray(mat_k, float),
        map_yr=np.full(n, 1000.0),
        t_seasonal_amp_k=np.asarray(t_seasonal_amp_k, float),
        aridity=np.full(n, 1.0),
        sw_mean_w=np.full(n, 200.0),
        soil_class=np.array(["loam"] * n, dtype=object),
    )


def test_frozen_fraction_byte_identical_to_annual_frozen_fraction():
    """Per-cell phi (single-PFT cells) is BYTE-IDENTICAL to the climate-only
    annual_frozen_fraction at the production-default width -- the SAME phi the
    archetype spin-up applied, so a run protects with the phi that seeded the SOC
    (exact IC-consistency, option (a))."""
    from legoesm.land.carbon.carbon_cycle import annual_frozen_fraction
    from legoesm.land.carbon.config import CarbonConfig
    mat = np.array([250.0, 285.0, 268.0]); amp = np.array([15.0, 8.0, 20.0])
    table = _phi_table(mat, amp)
    cid = np.array([[0, -1], [1, -1], [2, -1]])          # single-PFT cells
    cw = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    phi_cell = np.asarray(map_to_grid_frozen_fraction(table, cid, cw))
    phi_ref = np.asarray(annual_frozen_fraction(
        jnp.asarray(mat), jnp.asarray(amp), CarbonConfig(scheme="differland")))
    npt.assert_array_equal(phi_cell, phi_ref)            # byte-identical
    assert np.all((phi_cell >= 0.0) & (phi_cell <= 1.0))
    assert phi_cell[0] > 0.9 and phi_cell[1] < 0.1        # cold~1, warm~0


def test_frozen_fraction_is_intensive_mean_not_extensive_sum():
    """phi is a cover-weighted MEAN (an intensive climate fraction), NOT the
    extensive pool SUM map_to_grid uses: a half-bare permafrost cell keeps phi~1,
    not phi/2 (a partially-vegetated cold cell has the SAME frozen climate)."""
    table = _phi_table([250.0], [15.0])                  # one cold archetype, phi~1
    phi_half = np.asarray(map_to_grid_frozen_fraction(   # 50% cold PFT + 50% bare
        table, np.array([[0, -1]]), np.array([[0.5, 0.0]])))
    phi_full = np.asarray(map_to_grid_frozen_fraction(   # 100% cold PFT
        table, np.array([[0, -1]]), np.array([[1.0, 0.0]])))
    npt.assert_allclose(phi_half, phi_full, rtol=1e-9)   # NOT halved
    assert phi_half[0] > 0.9


def test_frozen_fraction_mixed_cell_is_normalised_weighted_mean():
    """A cell mixing a cold (phi~1) and a warm (phi~0) PFT gets the normalised
    cover-weighted mean of the two archetype phi values."""
    table = _phi_table([248.0, 290.0], [12.0, 6.0])      # cold, warm
    phi_arch = np.asarray(map_to_grid_frozen_fraction(
        table, np.array([[0, -1], [1, -1]]),
        np.array([[1.0, 0.0], [1.0, 0.0]])))
    phi_cell = np.asarray(map_to_grid_frozen_fraction(
        table, np.array([[0, 1]]), np.array([[0.25, 0.75]])))
    expect = 0.25 * phi_arch[0] + 0.75 * phi_arch[1]     # weights sum to 1 here
    npt.assert_allclose(phi_cell, [expect], rtol=1e-9)
    assert phi_arch[1] < phi_cell[0] < phi_arch[0]       # strictly between


def test_frozen_fraction_no_cover_cell_is_zero():
    """A cell with no vegetated cover -> phi = 0 (no seeded SOC to protect there;
    f_perma(0)~1 leaves any carbon unchanged)."""
    table = _phi_table([250.0, 260.0], [15.0, 15.0])
    phi_cell = np.asarray(map_to_grid_frozen_fraction(
        table, np.array([[-1, -1]]), np.array([[0.0, 0.0]])))
    npt.assert_array_equal(phi_cell, [0.0])


def test_frozen_fraction_empty_table_is_zeros():
    """Degenerate all-bare world (EMPTY archetype table) -> phi = 0 everywhere,
    guarded before the gather (which would index an empty phi_arch)."""
    empty = ArchetypeTable(
        pft_id=np.zeros(0, int), mat_k=np.zeros(0), map_yr=np.zeros(0),
        t_seasonal_amp_k=np.zeros(0), aridity=np.zeros(0), sw_mean_w=np.zeros(0),
        soil_class=np.zeros(0, dtype=object))
    phi = np.asarray(map_to_grid_frozen_fraction(
        empty, np.full((3, 2), -1), np.zeros((3, 2))))
    npt.assert_array_equal(phi, np.zeros(3))


def test_frozen_fraction_honors_config_width():
    """phi uses the PASSED config's som_freeze_width_K -- so a --tuned-params build
    that overrides the freeze width stays byte-identical to its own spin-up phi
    (codex finding: mapping must not silently pin the default width)."""
    from legoesm.land.carbon.carbon_cycle import annual_frozen_fraction
    from legoesm.land.carbon.config import CarbonConfig
    table = _phi_table([270.0], [10.0])            # below-freezing mean
    cid = np.array([[0, -1]]); cw = np.array([[1.0, 0.0]])
    wide = CarbonConfig(scheme="differland", som_freeze_width_K=6.0)
    phi_default = np.asarray(map_to_grid_frozen_fraction(table, cid, cw))
    phi_wide = np.asarray(map_to_grid_frozen_fraction(table, cid, cw, config=wide))
    ref_wide = np.asarray(annual_frozen_fraction(
        jnp.asarray([270.0]), jnp.asarray([10.0]), wide))
    npt.assert_array_equal(phi_wide, ref_wide)     # honored the passed width
    assert abs(float(phi_wide[0]) - float(phi_default[0])) > 1e-3  # width matters


# ---------------------------------------------------------------------------
# Per-PFT phenology grouping (Phase B): evergreen vs deciduous leaf habit is a
# third archetype-group key so tropical/needleleaf-evergreen PFTs equilibrate
# with continuous phenology.
# ---------------------------------------------------------------------------
def test_is_evergreen_classifies_leaf_habit():
    # The classifier is the shared surface_params helper the archetype grouping
    # imports (single source of truth with the per-pixel validator).
    from legoesm.land.surface_params import is_evergreen
    assert is_evergreen("broadleaf_evergreen_tropical") is True
    assert is_evergreen("needleleaf_evergreen_boreal") is True
    assert is_evergreen("broadleaf_deciduous_temperate") is False
    assert is_evergreen("needleleaf_deciduous_boreal") is False
    assert is_evergreen("c3_grass") is False


def test_iter_archetype_batches_splits_evergreen_from_deciduous():
    """Two WOODY archetypes on the SAME soil differing only in leaf habit
    (broadleaf_evergreen_tropical=pft 4 vs broadleaf_deciduous_temperate=pft 7)
    must land in SEPARATE (woody, evergreen, soil) groups, and the evergreen
    group's config must carry evergreen=True.  No archetype is dropped/duped."""
    from legoesm.land.carbon.global_init import iter_archetype_batches
    table = ArchetypeTable(
        pft_id=np.array([4, 7]),
        mat_k=np.array([298.0, 283.0]),
        map_yr=np.array([2000.0, 1000.0]),
        t_seasonal_amp_k=np.array([2.0, 12.0]),
        aridity=np.array([1.0, 1.0]),
        sw_mean_w=np.array([220.0, 200.0]),
        soil_class=np.array(["loam", "loam"], dtype=object),
    )
    batches = iter_archetype_batches(
        table, n_layers=6, soil_depth=2.0, dt=7200.0)
    # Leaf habit splits them into two distinct groups.
    assert len(batches) == 2
    # Every archetype covered exactly once (no drop / duplicate).
    covered = np.concatenate([np.asarray(b.g_idx) for b in batches])
    npt.assert_array_equal(np.sort(covered), np.array([0, 1]))
    # Map each single-member batch to its PFT and check the evergreen flag.
    by_pft = {int(table.pft_id[np.asarray(b.g_idx)[0]]): b for b in batches}
    assert by_pft[4].config.carbon.evergreen is True   # evergreen tropical
    assert by_pft[7].config.carbon.evergreen is False  # deciduous temperate
    # Both are woody, so woodiness alone would NOT have separated them.
    assert by_pft[4].config.carbon.woody is True
    assert by_pft[7].config.carbon.woody is True


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
    assert grid.C_som_active.shape == (ncell,)
    assert np.all(np.isfinite(np.asarray(grid.C_som_active)))
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
        # Per-cell perennial-frost index phi is persisted for the coupled run to
        # thread into step_multilayer_land (preserving seeded permafrost SOC).
        assert d["soil_frozen_fraction"].shape == (6,)
        assert np.all(np.isfinite(d["soil_frozen_fraction"]))
        assert np.all((d["soil_frozen_fraction"] >= 0.0)
                      & (d["soil_frozen_fraction"] <= 1.0))

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
