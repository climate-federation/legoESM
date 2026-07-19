"""Physics test for the TUNED-knob carbon-IC rebuild mechanism.

`scripts/data/build_global_carbon_ic.py --tuned-params` equilibrates every archetype
WITH calibrated CarbonConfig knobs (via `equilibrate_archetypes(carbon_overrides=...)`,
threaded into `iter_archetype_batches`). This test proves the override reaches the
coupled spin-up + analytic reset and moves equilibrium SOM in the physically-correct
direction -- the mechanism the tuned rebuild relies on (the build's cache-key + JSON-
loader wiring is unit-tested in `test_build_global_carbon_ic.py`).

Compute-node scale: `equilibrate_archetypes` JIT-compiles the coupled land+carbon
spin-up -- run via the sbatch/srun wrapper, NOT the login node. `JAX_ENABLE_X64=1`.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

from legoesm.land.carbon.climate_features import reduce_climatology_to_features
from legoesm.land.carbon.global_init import build_archetypes, equilibrate_archetypes


def test_equilibrate_archetypes_overrides_change_som():
    """carbon_overrides splice TUNED SOM knobs into every group's spin-up config:
    HALVING the passive turnover (slower decomposition -> C = I / k RISES) strictly
    RAISES equilibrium SOM for every archetype.  Proves the override reaches the
    coupled spin-up + analytic reset (not merely the cache key) and moves SOC in the
    physically-correct direction -- the mechanism the tuned-knob rebuild relies on."""
    from legoesm.land.carbon.config import CarbonConfig, som_total

    ncell, npft = 2, 17
    w = np.zeros((ncell, npft)); w[0, 4] = 1.0; w[1, 7] = 1.0
    t = np.stack([np.full(12, 298.0), np.full(12, 283.0)])
    pr = np.stack([np.full(12, 6e-5), np.full(12, 2.5e-5)])
    sw = np.full((ncell, 12), 220.0); nr = np.full((ncell, 12), 90.0)
    feats = reduce_climatology_to_features(t, pr, sw, nr)
    tab, _, _ = build_archetypes(
        w, feats, np.array(["loam", "loam"]), np.ones(ncell, bool), k_per_pft=1, seed=0)
    spin = dict(n_spinup=15, n_verify=5, dt=7200.0, n_layers=6, soil_depth=2.0)
    tor_pass = float(CarbonConfig().tor_som_passive)

    eq_def, _ = equilibrate_archetypes(tab, **spin)
    eq_slow, _ = equilibrate_archetypes(
        tab, carbon_overrides={"tor_som_passive": 0.5 * tor_pass}, **spin)
    som_def = np.asarray(som_total(eq_def))
    som_slow = np.asarray(som_total(eq_slow))
    assert np.all(np.isfinite(som_slow))
    # Slower passive turnover -> more passive SOC -> strictly more total SOM (the
    # passive pool dominates SOC and C_passive = I/k roughly doubles when k halves).
    assert np.all(som_slow > som_def + 1.0), (som_def, som_slow)
