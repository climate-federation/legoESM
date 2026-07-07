import numpy as np, numpy.testing as npt
from legoesm.land.carbon.climate_features import ClimateFeatures
from legoesm.land.carbon.global_init import build_archetypes

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
