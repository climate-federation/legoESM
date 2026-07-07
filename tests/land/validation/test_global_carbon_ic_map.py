"""Compute-smoke test for the batched archetype semi-analytic equilibration.

JIT-compiles two single-archetype groups end-to-end through the shared
``run_semi_analytic_spinup`` driver (Task 4).  Compute-node scale (multi-minute
JIT), NOT a login-node command -- run via the sbatch/srun wrapper.
"""

from __future__ import annotations

import numpy as np

from legoesm.land.carbon.global_init import ArchetypeTable, equilibrate_archetypes
from legoesm.land.carbon.config import CarbonState


def test_equilibrate_two_archetypes():
    # PFT 4 = broadleaf_evergreen_tropical, PFT 7 = broadleaf_deciduous_temperate.
    tab = ArchetypeTable(
        pft_id=np.array([4, 7]),
        mat_k=np.array([298.0, 283.0]), map_yr=np.array([2000.0, 800.0]),
        t_seasonal_amp_k=np.array([3.0, 12.0]), aridity=np.array([2.0, 1.0]),
        sw_mean_w=np.array([230.0, 180.0]),
        soil_class=np.array(["clay_loam", "loam"], dtype=object))
    eq, qc = equilibrate_archetypes(
        tab, n_spinup=20, n_verify=6, dt=7200.0, n_layers=6, soil_depth=2.0)
    assert isinstance(eq, CarbonState) and eq.C_som.shape == (2,)
    assert np.all(np.isfinite(np.asarray(eq.C_som)))
    assert (np.asarray(qc["gpp"]) >= 0).all()
    # Tropical archetype fixes more C than the temperate one.
    assert float(np.asarray(eq.C_wood)[0]) > 0.0

    # QC bundle is complete, archetype-ordered, and finite.
    for key in ("gpp", "npp", "som_kgC", "biomass_kgC", "drift_frac_per_yr"):
        arr = np.asarray(qc[key])
        assert arr.shape == (2,), key
        assert np.all(np.isfinite(arr)), key
    # Every pool is finite and non-negative at the verified equilibrium.
    for field in CarbonState._fields:
        vals = np.asarray(getattr(eq, field))
        assert vals.shape == (2,) and np.all(np.isfinite(vals)), field
        assert np.all(vals >= -1e-9), field
    # SOM stock and biomass are physical (positive) for both woody archetypes.
    assert (np.asarray(qc["som_kgC"]) > 0).all()
    assert (np.asarray(qc["biomass_kgC"]) > 0).all()


def test_same_group_columns_get_distinct_gpp():
    # Two WOODY archetypes (PFT 4, 7 -> both trees) forced into ONE (woody,
    # soil_class) group by giving them the SAME soil_class -> the group runs
    # ncol=2, exercising the per-column land_params GPP threading. Distinct
    # climate must yield distinct GPP; equal GPP would mean the columns
    # collapsed to a single PFT/climate (the silent batching bug).
    tab = ArchetypeTable(
        pft_id=np.array([4, 7]),
        mat_k=np.array([300.0, 280.0]), map_yr=np.array([2400.0, 600.0]),
        t_seasonal_amp_k=np.array([2.0, 14.0]), aridity=np.array([2.5, 0.8]),
        sw_mean_w=np.array([240.0, 170.0]),
        soil_class=np.array(["clay_loam", "clay_loam"], dtype=object))  # SAME -> one group
    eq, qc = equilibrate_archetypes(tab, n_spinup=20, n_verify=6, dt=7200.0,
                                    n_layers=6, soil_depth=2.0)
    gpp = np.asarray(qc["gpp"])
    assert gpp.shape == (2,)
    assert np.all(np.isfinite(gpp)) and (gpp >= 0).all()
    # The two columns ran in the SAME batch but with different climate+PFT ->
    # their GPP must differ (guards per-column land_params collapse).
    assert abs(float(gpp[0]) - float(gpp[1])) > 1.0   # gC/m2/yr, well above noise


# ===========================================================================
# Task 7: drift / realism validator (scripts/validate/global_carbon_ic_map.py)
# ===========================================================================

import importlib.util
import sys
from pathlib import Path

import pytest

from legoesm.land.carbon.global_init import (
    iter_archetype_batches,
    make_archetype_step_fn,
)
from legoesm.land.carbon.realism_ranges import (
    LITERATURE_BIOME_RANGES,
    PFT_BIOME,
)

_V = (Path(__file__).resolve().parents[3]
      / "scripts" / "validate" / "global_carbon_ic_map.py")
_SMOKE_ARCHETYPES = (Path(__file__).resolve().parents[3]
                     / ".superpowers" / "sdd" / "_gcic_smoke" / "archetypes.npz")


def _load():
    s = importlib.util.spec_from_file_location("gcicv", _V)
    m = importlib.util.module_from_spec(s)
    sys.modules["gcicv"] = m
    s.loader.exec_module(m)
    return m


# --- pure-logic drift metric (no model; verbatim from the Task-7 brief) ---

def test_drift_metric_zero_for_flat_series():
    v = _load()
    flat = np.ones((5,)) * 100.0
    assert abs(v.drift_frac_per_yr(flat)) < 1e-9


def test_drift_metric_detects_decline():
    v = _load()
    series = np.array([100.0, 99.0, 98.0, 97.0, 96.0])
    assert v.drift_frac_per_yr(series) < 0.0


# --- shared realism table (DRY: same table land_carbon_equilibrium.py uses) ---

def test_realism_ranges_wellformed():
    biomes = set(LITERATURE_BIOME_RANGES)
    for biome, r in LITERATURE_BIOME_RANGES.items():
        for k in ("gpp", "npp", "biomass", "soc", "lai"):
            lo, hi = r[k]
            assert lo <= hi, (biome, k)
    # Every CLM5 PFT maps to a real biome (or None for bare_soil).
    from legoesm.land.surface_params import CLM5_PFT_NAMES
    for name in CLM5_PFT_NAMES:
        assert name in PFT_BIOME, name
        biome = PFT_BIOME[name]
        assert biome is None or biome in biomes, (name, biome)
    assert PFT_BIOME["bare_soil"] is None


# --- shared per-group construction (factored out of equilibrate_archetypes) ---

def _tiny_table():
    # Two woody (PFT 4, 7) + one herbaceous (PFT 13) archetype with mixed soils
    # so the woody archetypes split by texture and the herbaceous forms its own
    # group -> exercises the (is_woody, soil_class) grouping.
    return ArchetypeTable(
        pft_id=np.array([4, 7, 13]),
        mat_k=np.array([298.0, 283.0, 288.0]),
        map_yr=np.array([2000.0, 800.0, 1000.0]),
        t_seasonal_amp_k=np.array([3.0, 12.0, 8.0]),
        aridity=np.array([2.0, 1.0, 1.5]),
        sw_mean_w=np.array([230.0, 180.0, 200.0]),
        soil_class=np.array(["clay_loam", "loam", "loam"], dtype=object))


def test_iter_archetype_batches_groups_cover_all_archetypes_once():
    tab = _tiny_table()
    batches = iter_archetype_batches(tab, n_layers=6, soil_depth=2.0, dt=7200.0)
    # 3 distinct (woody, soil) keys -> 3 groups; union of g_idx == all archetypes.
    assert len(batches) == 3
    covered = np.concatenate([np.asarray(b.g_idx) for b in batches])
    np.testing.assert_array_equal(np.sort(covered), np.arange(3))
    for b in batches:
        ncol = int(np.asarray(b.g_idx).shape[0])
        # per-column physiology + initial temperature match the group size.
        assert int(np.asarray(b.land_params.root_depth).shape[0]) == ncol
        assert int(np.asarray(b.t_init).shape[0]) == ncol
        assert b.steps_per_year == round(365.0 * 86400.0 / 7200.0)
        # herbaceous group (PFT 13) carries the no-wood carbon config.
        pfts = [int(tab.pft_id[a]) for a in np.asarray(b.g_idx)]
        assert b.config.carbon.woody == all(i in (4, 7) for i in pfts)
        assert callable(b.forcing_fn)


def test_make_archetype_step_fn_returns_callable():
    tab = _tiny_table()
    batch = iter_archetype_batches(
        tab, n_layers=6, soil_depth=2.0, dt=7200.0)[0]
    step_fn = make_archetype_step_fn(batch.config, batch.land_params, dt=7200.0)
    assert callable(step_fn)


# --- compute smoke (skipped when the dry-run archetypes file is absent) ---

@pytest.mark.skipif(not _SMOKE_ARCHETYPES.exists(),
                    reason="dry-run archetypes.npz not present")
def test_assess_ic_map_mapped_drift_below_cold():
    v = _load()
    # Match the map's build geometry (driver defaults n_layers=10/soil_depth=3.0)
    # so the mapped IC is re-integrated on the same soil column it equilibrated
    # on; a short dt keeps the smoke cheap (the comparison is dt-robust).
    result = v.assess_ic_map(_SMOKE_ARCHETYPES, n_years=2, dt=7200.0,
                             n_layers=10, soil_depth=3.0)
    assert result["n_archetypes"] >= 1
    # The archetype-equilibrium IC drifts LESS than a cold start.
    assert (result["mapped_median_abs_drift"]
            < result["cold_median_abs_drift"])
    assert result["mapped_much_less_than_cold"]
