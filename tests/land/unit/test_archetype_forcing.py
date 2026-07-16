"""Unit tests for the shared per-archetype growing-season forcing
(``legoesm.land.carbon.archetype_forcing``).

The ONE forcing-construction reused by the SIF and leaf-delta13C single-step forwards (no
copy-paste): the model's own climatological forcing + per-archetype PFT ``Vc_max25`` at the
representative growing-season sampling point.

Compute-node scale (vmaps ``make_climatological_forcing``); run via the sbatch/srun wrapper,
NOT the login node.  ``JAX_ENABLE_X64=1``; forced CPU here.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import numpy.testing as npt


def _table(pft_id, mat_k, sw, *, t_amp=None, map_yr=None):
    from legoesm.land.carbon.global_init import ArchetypeTable
    pft_id = np.asarray(pft_id, dtype=int)
    n = pft_id.shape[0]
    return ArchetypeTable(
        pft_id=pft_id,
        mat_k=np.asarray(mat_k, dtype=float),
        map_yr=np.asarray(map_yr if map_yr is not None else [1200.0] * n, dtype=float),
        t_seasonal_amp_k=np.asarray(t_amp if t_amp is not None else [6.0] * n, dtype=float),
        aridity=np.asarray([1.0] * n, dtype=float),
        sw_mean_w=np.asarray(sw, dtype=float),
        soil_class=np.asarray(["loam"] * n, dtype=object),
    )


def test_build_archetype_forcing_shape_and_finite():
    """One forcing value per archetype, all finite; Vc_max25 is the per-archetype PFT value."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.archetype_forcing import build_archetype_forcing

    table = _table([1, 4, 14], [283.0, 298.0, 300.0], [170.0, 230.0, 250.0])
    f = build_archetype_forcing(table)
    for arr in (f.T_leaf, f.sw_down, f.co2_ppmv, f.q_air, f.p_surface, f.Vc_max25):
        a = np.asarray(arr)
        assert a.shape == (3,), a.shape
        assert np.all(np.isfinite(a))
    # sw_down and CO2/pressure are positive; Vc_max25 is the per-PFT capacity (positive).
    assert np.all(np.asarray(f.sw_down) >= 0.0)
    assert np.all(np.asarray(f.co2_ppmv) > 0.0)
    assert np.all(np.asarray(f.p_surface) > 0.0)
    assert np.all(np.asarray(f.Vc_max25) > 0.0)


def test_vc_max25_matches_clm5_table_per_pft():
    """The forcing Vc_max25 is exactly the CLM5 table's per-PFT Vc_max25 (no re-derivation)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.carbon.archetype_forcing import build_archetype_forcing
    from legoesm.land.surface_params import (
        PARAM_NAMES,
        array_to_params,
        clm5_pft_table,
    )

    pfts = [1, 4, 7, 14]
    f = build_archetype_forcing(_table(pfts, [290.0] * 4, [200.0] * 4))
    expected = np.asarray(
        array_to_params(np.asarray(clm5_pft_table())[pfts], PARAM_NAMES).Vc_max25)
    npt.assert_allclose(np.asarray(f.Vc_max25), expected, rtol=1e-12)


def test_sif_forward_still_matches_after_refactor():
    """The SIF forward (refactored onto build_archetype_forcing) is unchanged: SIF >= 0,
    finite, and scales linearly with the escape probability (the aggregation wiring)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.canopy.sif import SIFConfig
    from legoesm.land.carbon.sif_forward import simulate_archetype_sif

    table = _table([1, 4, 7], [283.0, 295.0, 300.0], [170.0, 220.0, 250.0])
    full = np.asarray(simulate_archetype_sif(table, SIFConfig(escape_probability=1.0)))
    half = np.asarray(simulate_archetype_sif(table, SIFConfig(escape_probability=0.5)))
    assert full.shape == (3,)
    assert np.all(np.isfinite(full)) and np.all(full >= 0.0)
    npt.assert_allclose(half, 0.5 * full, rtol=1e-10)
