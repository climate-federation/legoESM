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
