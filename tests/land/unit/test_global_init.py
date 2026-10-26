import numpy as np, numpy.testing as npt, jax.numpy as jnp
import pytest
from legoesm.land.carbon.climate_features import (
    ClimateFeatures, reduce_climatology_to_features,
)
from legoesm.land.carbon.config import CarbonState
from legoesm.land.carbon.global_init import (
    ArchetypeTable, build_archetypes, equilibrate_archetypes,
    load_finidat_carbon_ic, map_to_grid, map_to_grid_frozen_fraction,
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


def test_iter_archetype_batches_splits_and_flags_cold_deciduous():
    """Two WOODY DECIDUOUS archetypes on the SAME soil differing only in the
    cold-deciduous trait (needleleaf_deciduous_boreal larch vs
    broadleaf_deciduous_temperate) must land in SEPARATE groups; with
    cold_deciduous_dormancy=True the larch group carries cold_deciduous=True and
    the master flag, the temperate group cold_deciduous=False.  Neither is
    evergreen, so the leaf-habit key alone would NOT have separated them."""
    from legoesm.land.surface_params import CLM5_PFT_NAMES
    from legoesm.land.carbon.global_init import iter_archetype_batches
    larch = CLM5_PFT_NAMES.index("needleleaf_deciduous_boreal")
    temp = CLM5_PFT_NAMES.index("broadleaf_deciduous_temperate")
    table = ArchetypeTable(
        pft_id=np.array([larch, temp]),
        mat_k=np.array([270.0, 285.0]),
        map_yr=np.array([400.0, 900.0]),
        t_seasonal_amp_k=np.array([20.0, 12.0]),
        aridity=np.array([1.0, 1.1]),
        sw_mean_w=np.array([150.0, 200.0]),
        soil_class=np.array(["loam", "loam"], dtype=object),
    )
    batches = iter_archetype_batches(
        table, n_layers=6, soil_depth=2.0, dt=7200.0,
        cold_deciduous_dormancy=True)
    assert len(batches) == 2
    by_pft = {int(table.pft_id[np.asarray(b.g_idx)[0]]): b for b in batches}
    assert by_pft[larch].config.carbon.cold_deciduous is True
    assert by_pft[larch].config.carbon.cold_deciduous_dormancy is True
    assert by_pft[temp].config.carbon.cold_deciduous is False
    # master flag on for both; the per-PFT trait scopes the gate.
    assert by_pft[temp].config.carbon.cold_deciduous_dormancy is True
    assert by_pft[larch].config.carbon.evergreen is False
    assert by_pft[temp].config.carbon.evergreen is False
    # Default (no flag) -> master off everywhere (byte-identical).
    for b in iter_archetype_batches(table, n_layers=6, soil_depth=2.0, dt=7200.0):
        assert b.config.carbon.cold_deciduous_dormancy is False


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
    physics = {"stomatal_model": "medlyn", "nsc_gated_respiration": True,
               "cold_deciduous_dormancy": False, "leaf_c_resorption_frac": 0.5}
    assert set(physics) == set(drv.PHYSICS_PROVENANCE_KEYS)
    drv._write_archetypes_npz(
        path, table, eq, qc, ["bare_soil"], n_layers=7, soil_depth=2.5,
        dt=1800.0, res_deg=2.0, physics=physics)
    with np.load(path, allow_pickle=False) as d:
        assert int(d["n_layers"]) == 7
        assert float(d["soil_depth"]) == 2.5
        # physics provenance round-trips with its types (the drift validator
        # rebuilds the model from these; a str "False" would be truthy)
        assert {k: d[k].item() for k in physics} == physics
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


# ---------------------------------------------------------------------------
# load_finidat_carbon_ic: the finidat->run carbon-IC loader (the inverse of
# map_to_grid + map_to_grid_frozen_fraction).  Detection (a non-carbon .npz ->
# None), STRICT grid-match (fail-loud), and phi validation.
# ---------------------------------------------------------------------------
def _write_finidat(path, ncol, *, lat_deg=None, lon_deg=None, phi=None,
                   with_phi=True, drop_pool=None, som_scale=1000.0,
                   land_mask=None, drop_land_mask=False):
    """Write a tiny finidat global_carbon_ic.npz with the 8 pools (+ optional phi).

    Pool values are per-cell and DISTINCT per pool so a round-trip cannot alias
    two fields; SOM is seeded large (``som_scale``) so a seeded-vs-cold test has a
    clear separation.  ``drop_pool`` omits a pool (to exercise the detection gate);
    ``with_phi=False`` omits ``soil_frozen_fraction`` (a legacy finidat).
    ``land_mask`` defaults to all-land (so every existing caller is unchanged);
    ``drop_land_mask`` omits the field entirely.
    """
    fields = CarbonState._fields
    d = {}
    for i, f in enumerate(fields):
        base = (som_scale if f.startswith("C_som") else 10.0 * (i + 1))
        d[f] = np.full(ncol, base + np.arange(ncol), dtype=np.float64)
    if drop_pool is not None:
        d.pop(drop_pool)
    d["lat"] = (np.asarray(lat_deg, float) if lat_deg is not None
                else np.linspace(-80.0, 80.0, ncol))
    d["lon"] = (np.asarray(lon_deg, float) if lon_deg is not None
                else np.linspace(0.0, 350.0, ncol))
    if not drop_land_mask:
        d["land_mask"] = (np.ones(ncol, bool) if land_mask is None
                          else np.asarray(land_mask, bool))
    if with_phi:
        d["soil_frozen_fraction"] = (np.asarray(phi, float) if phi is not None
                                     else np.linspace(0.0, 1.0, ncol))
    np.savez(path, **d)
    return path


def test_load_finidat_carbon_ic_round_trips_pools_and_phi(tmp_path):
    ncol = 12
    p = _write_finidat(tmp_path / "gcic.npz", ncol)
    carbon, phi = load_finidat_carbon_ic(p, expect_ncol=ncol)
    assert isinstance(carbon, CarbonState)
    for f in CarbonState._fields:
        got = np.asarray(getattr(carbon, f))
        assert got.shape == (ncol,)
        base = (1000.0 if f.startswith("C_som")
                else 10.0 * (CarbonState._fields.index(f) + 1))
        npt.assert_allclose(got, base + np.arange(ncol))
    npt.assert_allclose(np.asarray(phi), np.linspace(0.0, 1.0, ncol))


def test_load_finidat_carbon_ic_matches_target_lat_lon(tmp_path):
    # STRICT grid-match passes when the finidat lat/lon equal the run grid columns.
    ncol = 8
    lat = np.linspace(-70.0, 70.0, ncol); lon = np.linspace(10.0, 340.0, ncol)
    p = _write_finidat(tmp_path / "gcic.npz", ncol, lat_deg=lat, lon_deg=lon)
    carbon, phi = load_finidat_carbon_ic(
        p, expect_ncol=ncol, target_lat_deg=lat, target_lon_deg=lon)
    assert carbon is not None and phi is not None


def test_load_finidat_carbon_ic_lon_modulo_360_not_a_false_mismatch(tmp_path):
    # A finidat stored in 0..360 vs a run grid in -180..180 is the SAME grid.
    ncol = 6
    lat = np.linspace(-50.0, 50.0, ncol)
    lon_finidat = np.array([0.0, 60.0, 120.0, 200.0, 280.0, 350.0])   # 0..360
    lon_target = np.where(lon_finidat > 180.0, lon_finidat - 360.0, lon_finidat)
    p = _write_finidat(tmp_path / "gcic.npz", ncol,
                       lat_deg=lat, lon_deg=lon_finidat)
    carbon, _ = load_finidat_carbon_ic(
        p, expect_ncol=ncol, target_lat_deg=lat, target_lon_deg=lon_target)
    assert carbon is not None


def test_load_finidat_carbon_ic_not_a_carbon_finidat_returns_none(tmp_path):
    # A .npz missing a pool field is NOT a carbon finidat -> (None, None) so the
    # caller keeps its cold-start (static feature gate).
    ncol = 5
    p = _write_finidat(tmp_path / "legacy.npz", ncol, drop_pool="C_som_active")
    carbon, phi = load_finidat_carbon_ic(p, expect_ncol=ncol)
    assert carbon is None and phi is None


def test_load_finidat_carbon_ic_ncol_mismatch_raises(tmp_path):
    p = _write_finidat(tmp_path / "gcic.npz", 12)
    with pytest.raises(ValueError):
        load_finidat_carbon_ic(p, expect_ncol=10)


def test_load_finidat_carbon_ic_lat_mismatch_raises(tmp_path):
    ncol = 8
    lat = np.linspace(-70.0, 70.0, ncol); lon = np.linspace(10.0, 340.0, ncol)
    p = _write_finidat(tmp_path / "gcic.npz", ncol, lat_deg=lat, lon_deg=lon)
    with pytest.raises(ValueError):
        # A shifted latitude (different / re-ordered grid) must fail loud.
        load_finidat_carbon_ic(p, expect_ncol=ncol,
                               target_lat_deg=lat + 5.0, target_lon_deg=lon)


def test_load_finidat_carbon_ic_phi_absent_is_none(tmp_path):
    # A legacy finidat with no soil_frozen_fraction -> phi None (coupler
    # unprotected, byte-identical); the pools still load.
    ncol = 7
    p = _write_finidat(tmp_path / "gcic.npz", ncol, with_phi=False)
    carbon, phi = load_finidat_carbon_ic(p, expect_ncol=ncol)
    assert carbon is not None and phi is None


def test_load_finidat_carbon_ic_phi_out_of_range_raises(tmp_path):
    ncol = 6
    bad = np.linspace(0.0, 1.5, ncol)   # > 1: not an annual frozen FRACTION
    p = _write_finidat(tmp_path / "gcic.npz", ncol, phi=bad)
    with pytest.raises(ValueError):
        load_finidat_carbon_ic(p, expect_ncol=ncol)


def test_load_finidat_carbon_ic_latlon_grid_glue(tmp_path):
    # Exercise the EXACT conversion the coupled driver does: build a finidat on a
    # real create_latlon_grid, then load with target lat/lon = rad2deg(grid).ravel()
    # (row-major) -- the driver's grid-match glue, so a rad<->deg / ravel-order bug
    # would fail here cheaply (no atmosphere compile).
    from legoesm.grids.latlon import create_latlon_grid
    grid = create_latlon_grid(n_lat=6, n_lon=8)
    lat2d = np.rad2deg(np.asarray(grid.lat2d)); lon2d = np.rad2deg(np.asarray(grid.lon2d))
    ncol = lat2d.size
    p = _write_finidat(tmp_path / "gcic.npz", ncol,
                       lat_deg=lat2d.ravel(), lon_deg=lon2d.ravel())
    carbon, phi = load_finidat_carbon_ic(
        p, expect_ncol=ncol,
        target_lat_deg=lat2d.ravel(), target_lon_deg=lon2d.ravel())
    assert carbon is not None and np.asarray(carbon.C_som_active).shape == (ncol,)


# ---------------------------------------------------------------------------
# load_finidat_carbon_ic_at_point: seeding a SINGLE-COLUMN run (run_lmip) from
# the nearest land cell of the global finidat.  Covers the nearest-cell pick,
# the land-only restriction, the 0/360 seam, the distance guard, and the
# great-circle metric (which a lat/lon Euclidean metric gets WRONG at high
# latitude -- the case the naive metric fails is asserted explicitly).
# ---------------------------------------------------------------------------
def _pool_value(field, cell, ncol=None, som_scale=1000.0):
    """The value _write_finidat puts in `field` at flat index `cell`."""
    base = (som_scale if field.startswith("C_som")
            else 10.0 * (CarbonState._fields.index(field) + 1))
    return base + cell


def test_point_ic_picks_nearest_cell_and_shapes_one_column(tmp_path):
    from legoesm.land.carbon.global_init import load_finidat_carbon_ic_at_point
    lat = np.array([-40.0, 0.0, 10.0, 55.0])
    lon = np.array([0.0, 0.0, 0.0, 0.0])
    p = _write_finidat(tmp_path / "gcic.npz", 4, lat_deg=lat, lon_deg=lon,
                       phi=np.array([0.0, 0.1, 0.2, 0.9]))

    carbon, phi, match = load_finidat_carbon_ic_at_point(p, 11.0, 0.0)

    assert match.index == 2                      # 10.0 is nearest to 11.0
    assert match.lat_deg == 10.0 and match.is_land
    npt.assert_allclose(match.distance_deg, 1.0, atol=1e-9)
    for f in CarbonState._fields:
        got = np.asarray(getattr(carbon, f))
        assert got.shape == (1,)                 # shaped for a 1-column run
        npt.assert_allclose(got, [_pool_value(f, 2)])
    npt.assert_allclose(np.asarray(phi), [0.2])


def test_point_ic_skips_ocean_cells(tmp_path):
    """The nearest cell overall is ocean -> the nearest LAND cell is used."""
    from legoesm.land.carbon.global_init import load_finidat_carbon_ic_at_point
    lat = np.array([0.0, 10.0, 12.0])
    lon = np.zeros(3)
    p = _write_finidat(tmp_path / "gcic.npz", 3, lat_deg=lat, lon_deg=lon,
                       land_mask=[True, False, True])   # cell 1 is ocean

    _c, _phi, match = load_finidat_carbon_ic_at_point(p, 10.4, 0.0)
    assert match.index == 2                      # NOT 1, which is 0.4 deg away
    npt.assert_allclose(match.distance_deg, 1.6, atol=1e-9)

    # Without the land restriction the ocean cell IS the nearest -- so the
    # restriction, not the geometry, is what moved the answer.
    _c2, _phi2, m2 = load_finidat_carbon_ic_at_point(
        p, 10.4, 0.0, require_land=False)
    assert m2.index == 1


def test_point_ic_longitude_wraps_at_the_seam(tmp_path):
    """A -10 deg request matches a cell stored at 350 deg (same meridian)."""
    from legoesm.land.carbon.global_init import load_finidat_carbon_ic_at_point
    lat = np.array([0.0, 0.0])
    lon = np.array([350.0, 20.0])
    p = _write_finidat(tmp_path / "gcic.npz", 2, lat_deg=lat, lon_deg=lon)

    _c, _phi, match = load_finidat_carbon_ic_at_point(p, 0.0, -10.0)
    assert match.index == 0
    npt.assert_allclose(match.distance_deg, 0.0, atol=1e-9)


def test_point_ic_uses_great_circle_not_latlon_euclidean(tmp_path):
    """At 85 N a 40 deg longitude offset is only ~3.4 deg of arc, i.e. CLOSER
    than a 4 deg latitude offset.  A lat/lon Euclidean metric ranks these the
    other way round, so this case discriminates the two metrics."""
    from legoesm.land.carbon.global_init import load_finidat_carbon_ic_at_point
    lat = np.array([85.0, 81.0])
    lon = np.array([40.0, 0.0])
    p = _write_finidat(tmp_path / "gcic.npz", 2, lat_deg=lat, lon_deg=lon)

    _c, _phi, match = load_finidat_carbon_ic_at_point(
        p, 85.0, 0.0, max_distance_deg=5.0)
    assert match.index == 0                      # great circle: 3.42 < 4.0
    npt.assert_allclose(match.distance_deg, 3.418, atol=1e-2)
    # The metric the code must NOT be using would have chosen cell 1:
    naive = np.hypot(lat - 85.0, lon - 0.0)
    assert int(np.argmin(naive)) == 1


def test_point_ic_too_far_raises(tmp_path):
    from legoesm.land.carbon.global_init import load_finidat_carbon_ic_at_point
    p = _write_finidat(tmp_path / "gcic.npz", 2,
                       lat_deg=np.array([80.0, 85.0]), lon_deg=np.zeros(2))
    with pytest.raises(ValueError, match="beyond max_distance_deg"):
        load_finidat_carbon_ic_at_point(p, 0.0, 0.0)


def test_point_ic_not_a_carbon_finidat_raises(tmp_path):
    """Unlike the auto-detecting strict loader (which returns None), an explicit
    point request for a non-finidat is an ERROR -- the caller named the file."""
    from legoesm.land.carbon.global_init import load_finidat_carbon_ic_at_point
    p = _write_finidat(tmp_path / "gcic.npz", 3, lat_deg=np.zeros(3),
                       lon_deg=np.zeros(3), drop_pool="C_wood")
    with pytest.raises(ValueError, match="not a carbon finidat"):
        load_finidat_carbon_ic_at_point(p, 0.0, 0.0)


def test_point_ic_no_land_mask_requires_opt_out(tmp_path):
    """A finidat with no land_mask cannot honour require_land -> fail loud, and
    require_land=False is the explicit opt-out."""
    from legoesm.land.carbon.global_init import load_finidat_carbon_ic_at_point
    p = _write_finidat(tmp_path / "gcic.npz", 2, lat_deg=np.zeros(2),
                       lon_deg=np.array([0.0, 1.0]), drop_land_mask=True)
    with pytest.raises(ValueError, match="no 'land_mask'"):
        load_finidat_carbon_ic_at_point(p, 0.0, 0.0)
    _c, _phi, match = load_finidat_carbon_ic_at_point(
        p, 0.0, 0.0, require_land=False)
    assert match.index == 0


def test_point_ic_legacy_finidat_without_phi(tmp_path):
    from legoesm.land.carbon.global_init import load_finidat_carbon_ic_at_point
    p = _write_finidat(tmp_path / "gcic.npz", 2, lat_deg=np.zeros(2),
                       lon_deg=np.array([0.0, 1.0]), with_phi=False)
    _c, phi, _m = load_finidat_carbon_ic_at_point(p, 0.0, 0.0)
    assert phi is None


def test_point_ic_reports_the_spinup_soil_column(tmp_path):
    """The match carries the soil column the pools were spun up on, so a caller
    can compare it against its own grid (the pools have no vertical dimension,
    so a difference is a consistency warning, not a load failure)."""
    from legoesm.land.carbon.global_init import load_finidat_carbon_ic_at_point
    p = _write_finidat(tmp_path / "gcic.npz", 2, lat_deg=np.zeros(2),
                       lon_deg=np.array([0.0, 1.0]))
    _c, _phi, match = load_finidat_carbon_ic_at_point(p, 0.0, 0.0)
    # _write_finidat stores no geometry -> None, never a fabricated default.
    assert match.n_layers is None and match.soil_depth_m is None

    import numpy as _np
    with _np.load(p, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    d["n_layers"] = _np.asarray(10)
    d["soil_depth"] = _np.asarray(3.0)
    p2 = tmp_path / "gcic_geom.npz"
    _np.savez(p2, **d)
    _c2, _phi2, m2 = load_finidat_carbon_ic_at_point(p2, 0.0, 0.0)
    assert m2.n_layers == 10
    npt.assert_allclose(m2.soil_depth_m, 3.0)


def test_point_ic_reports_the_cells_dominant_cover(tmp_path):
    """A finidat cell is a PFT MIXTURE; the match surfaces which cover dominates
    it (and by how much) so a single-veg_type point run can see the mismatch."""
    import numpy as _np
    from legoesm.land.carbon.global_init import load_finidat_carbon_ic_at_point
    p = _write_finidat(tmp_path / "gcic.npz", 2, lat_deg=_np.zeros(2),
                       lon_deg=_np.array([0.0, 1.0]))
    with _np.load(p, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    d["dominant_pft"] = _np.array([2, 1])
    d["pft_names"] = _np.array(["bare_soil", "c3_grass", "broadleaf"], dtype="<U40")
    w = _np.zeros((2, 3)); w[0, 2] = 0.6; w[0, 1] = 0.4; w[1, 1] = 1.0
    d["pft_weights"] = w
    p2 = tmp_path / "gcic_pft.npz"
    _np.savez(p2, **d)

    _c, _phi, m = load_finidat_carbon_ic_at_point(p2, 0.0, 0.0)
    assert m.index == 0
    assert m.dominant_pft == "broadleaf"
    npt.assert_allclose(m.dominant_pft_weight, 0.6)
    # A finidat without the cover fields reports None, never a fabricated cover.
    _c2, _phi2, m2 = load_finidat_carbon_ic_at_point(p, 0.0, 0.0)
    assert m2.dominant_pft is None and m2.dominant_pft_weight is None


def test_point_ic_misaligned_selection_array_raises(tmp_path):
    """A finidat whose lat/land_mask length disagrees with the pools would make
    the nearest-cell pick index a DIFFERENT column -- fail loud instead."""
    import numpy as _np
    from legoesm.land.carbon.global_init import load_finidat_carbon_ic_at_point
    p = _write_finidat(tmp_path / "gcic.npz", 4, lat_deg=_np.zeros(4),
                       lon_deg=_np.arange(4.0))
    with _np.load(p, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    d["lat"] = _np.zeros(3)          # 3 coords vs 4 pool cells
    p2 = tmp_path / "gcic_bad.npz"
    _np.savez(p2, **d)
    with pytest.raises(ValueError, match="different cells"):
        load_finidat_carbon_ic_at_point(p2, 0.0, 0.0)


def _one_archetype_table():
    return ArchetypeTable(
        pft_id=np.array([4]),
        mat_k=np.array([298.0]),
        map_yr=np.array([2000.0]),
        t_seasonal_amp_k=np.array([2.0]),
        aridity=np.array([1.0]),
        sw_mean_w=np.array([220.0]),
        soil_class=np.array(["loam"], dtype=object),
    )


def test_iter_archetype_batches_canopy_switch_reaches_config():
    """The stomatal model must land on BOTH the two-leaf surface scheme and the
    mirrored big-leaf StomataConfig, and the surface scheme must be the
    two-leaf canopy, not SimpleSEB."""
    from legoesm.land.carbon.global_init import iter_archetype_batches
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    (b,) = iter_archetype_batches(
        _one_archetype_table(), n_layers=6, soil_depth=2.0, dt=7200.0,
        stomatal_model="medlyn")
    cc = b.config.surface_scheme
    assert isinstance(cc, TwoLeafCanopyConfig)
    assert cc.stomatal_model == "medlyn"
    st = b.config.stomata
    assert st.enabled is True
    assert st.stomata_model == "medlyn"
    (a,) = iter_archetype_batches(
        _one_archetype_table(), n_layers=6, soil_depth=2.0, dt=7200.0)
    assert a.config.surface_scheme.stomatal_model == "ball_berry"


@pytest.mark.parametrize("bad", [
    dict(stomatal_model="jarvis"),
    dict(stomatal_model="leuning"),
])
def test_iter_archetype_batches_rejects_unknown_switches(bad):
    from legoesm.land.carbon.global_init import iter_archetype_batches
    with pytest.raises(ValueError):
        iter_archetype_batches(
            _one_archetype_table(), n_layers=6, soil_depth=2.0, dt=7200.0,
            **bad)


def test_archetype_step_diagnostics_use_the_canopy_gpp():
    """The per-step carbon diagnostics the semi-analytic reset reads must carry
    the SAME GPP the coupled step fed to the carbon update -- the two-leaf
    canopy's own ``SurfaceFluxOutput.gpp`` -- not a big-leaf re-derivation.
    ``diag.gpp`` is a daily rate [gC/m2/day]; the canopy GPP is [gC/m2/s]."""
    from legoesm.land.carbon.carbon_cycle import init_carbon_state
    from legoesm.land.carbon.global_init import (
        _U_MIN, iter_archetype_batches, make_archetype_step_fn)
    from legoesm.land.multilayer_land import (
        init_multilayer_land_state, step_multilayer_land_with_diagnostics)
    dt = 7200.0
    (b,) = iter_archetype_batches(
        _one_archetype_table(), n_layers=6, soil_depth=2.0, dt=dt,
        stomatal_model="medlyn")
    step_fn = make_archetype_step_fn(
        b.config, b.land_params, dt=dt,
        soil_frozen_fraction=b.soil_frozen_fraction)
    state0 = init_multilayer_land_state(1, b.config, T_init=b.t_init)
    carbon0 = init_carbon_state((1,), b.config.carbon)
    forcing = b.forcing_fn(180.0, 12.0)          # a summer noon: GPP > 0
    _st, _cb, diag = step_fn(state0, carbon0, forcing, 180.0)
    _s2, _r, _c2, surface_out = step_multilayer_land_with_diagnostics(
        state0, forcing, b.config, _U_MIN, dt,
        lat=jnp.zeros(1), carbon_state=carbon0, doy=180.0,
        land_params=b.land_params,
        soil_frozen_fraction=b.soil_frozen_fraction)
    gpp_canopy_day = np.asarray(surface_out.gpp) * 86400.0
    assert float(gpp_canopy_day[0]) > 0.0
    npt.assert_allclose(np.asarray(diag.gpp), gpp_canopy_day, rtol=1e-6)
    # ...and the coupled step's OWN carbon update used that same GPP: rebuild
    # it from the step's end-of-step state with the canopy GPP and require the
    # carbon state step_fn returned.  A coupled step that re-derived big-leaf
    # GPP would pass the diagnostics check above and fail here.
    from legoesm.land.carbon.carbon_cycle import step_carbon
    from legoesm.land.multilayer_land import root_zone_beta_soil
    from legoesm.land.soil_grid import make_soil_grid
    grid = make_soil_grid(b.config.soil_grid)
    root_frac = jnp.exp(-grid.z_node[None, :] / b.land_params.root_depth[:, None])
    root_frac = root_frac / jnp.sum(root_frac, axis=-1, keepdims=True)
    beta_soil_new, _ = root_zone_beta_soil(
        _st.theta_soil, root_frac, b.land_params.theta_wp,
        b.land_params.theta_fc, b.config.beta_min, spatial=True)
    carbon_expected, _flux = step_carbon(
        carbon0, forcing.sw_down, _st.T_soil[:, 0], forcing.co2_ppmv,
        beta_soil_new, jnp.zeros(1), 180.0, forcing.precip_total,
        b.config.carbon, dt, gpp_override=surface_out.gpp,
        soil_frozen_fraction=b.soil_frozen_fraction)
    for f in carbon_expected._fields:
        npt.assert_allclose(np.asarray(getattr(_cb, f)),
                            np.asarray(getattr(carbon_expected, f)),
                            rtol=1e-10, err_msg=f)


def test_archetype_spinup_runs_the_drainage_limiter_at_the_calibrated_value():
    """The carbon spin-up must run the land at the calibrated drainage
    limiter (0.5), not the RichardsConfig library default (0.0)."""
    from legoesm.land.surface_params import CLM5_PFT_NAMES
    from legoesm.land.carbon.global_init import iter_archetype_batches
    pft = CLM5_PFT_NAMES.index("broadleaf_deciduous_temperate")
    table = ArchetypeTable(
        pft_id=np.array([pft]), mat_k=np.array([285.0]),
        map_yr=np.array([900.0]), t_seasonal_amp_k=np.array([12.0]),
        aridity=np.array([1.1]), sw_mean_w=np.array([200.0]),
        soil_class=np.array(["loam"], dtype=object))
    batches = iter_archetype_batches(table, n_layers=6, soil_depth=2.0, dt=7200.0)
    assert len(batches) == 1
    assert batches[0].config.richards.fc_drain_saturation == 0.5
