"""The explicit biharmonic vorticity filter must be refused beyond its
stability limit (level-8 blowup, 2026-09-04: K_zeta_bih = 1e14 fixed by the
NEMO-match recipe crossed the forward-Euler limit on the 29 km dual mesh and
nothing checked it).  The gate uses lambda_max ~ 8/dv_min^2: 1.4 on level 7
(ran 180 days), 11 on level 8 (blew up in 10 steps)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(scope="module")
def mesh():
    return create_voronoi_mesh(2)


def _model(mesh, K):
    return MPASOceanModel(mesh, create_ocean_z_star(3, H_max=300.0),
                          MPASOceanConfig(K_zeta_bih=K))


def test_number_matches_the_level_7_and_level_8_measurements(mesh):
    """Scale the coarse test mesh's dv to the production values by choosing K
    and dt so the formula reproduces the measured stability numbers."""
    m = _model(mesh, 1.0e14)
    dv_min = float(np.min(np.where(np.asarray(mesh.dvEdge) > 0, np.asarray(mesh.dvEdge), np.inf)))
    lam = 8.0 / dv_min ** 2
    assert m.vorticity_filter_stability_number(150.0) == pytest.approx(1e14 * 150.0 * lam * lam, rel=1e-12)
    # the same formula on the production MINIMUM dual spacings (Lloyd meshes:
    # 28.9 km on level 7, 14.4 km on level 8; codex 2026-09-04)
    for dv, dt, expect in ((28.9e3, 150.0, 1.4), (14.4e3, 75.0, 11.0)):
        n = 1e14 * dt * (8.0 / dv ** 2) ** 2
        assert n == pytest.approx(expect, rel=0.15)


def test_zero_filter_is_free(mesh):
    assert _model(mesh, 0.0).vorticity_filter_stability_number(1800.0) == 0.0
    _model(mesh, 0.0).check_vorticity_filter_stability(1800.0)   # no raise


def test_unstable_filter_raises_and_marginal_warns(mesh):
    m = _model(mesh, 1.0)
    dv_min = float(np.min(np.where(np.asarray(mesh.dvEdge) > 0, np.asarray(mesh.dvEdge), np.inf)))
    lam = 8.0 / dv_min ** 2
    dt_at = lambda n: n / (1.0 * lam * lam)              # dt giving number n at K=1
    with pytest.raises(ValueError, match="unstable"):
        m.check_vorticity_filter_stability(dt_at(11.0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.check_vorticity_filter_stability(dt_at(1.4))
    assert any("marginal" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.check_vorticity_filter_stability(dt_at(0.5))
    assert not any("marginal" in str(x.message) for x in w)


def test_first_checked_step_runs_the_gate(mesh):
    """The gate lives inside the existing first-step CFL check so a card
    cannot skip it."""
    m = _model(mesh, 1.0)
    dv_min = float(np.min(np.where(np.asarray(mesh.dvEdge) > 0, np.asarray(mesh.dvEdge), np.inf)))
    lam = 8.0 / dv_min ** 2
    with pytest.raises(ValueError, match="vorticity filter"):
        m.check_barotropic_cfl(11.0 / (lam * lam))


def test_production_step_runs_the_gate_too(mesh):
    """codex 2026-09-06: the OMIP runner calls step(), not step_checked()."""
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    z = create_ocean_z_star(3, H_max=300.0)
    m = _model(mesh, 1.0)
    dv_min = float(np.min(np.where(np.asarray(mesh.dvEdge) > 0, np.asarray(mesh.dvEdge), np.inf)))
    lam = 8.0 / dv_min ** 2
    state = rest_state_mpas_ocean(mesh, z)
    with pytest.raises(ValueError, match="vorticity filter"):
        m.step(state, 11.0 / (lam * lam))
