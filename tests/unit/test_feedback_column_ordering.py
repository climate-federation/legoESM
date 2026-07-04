"""Column-ordering contract: feedback field ↔ physics ColumnAdapter.

The LES-informed per-column coefficient is built at the manifest's flat grid
indices (``rank_worst_columns`` uses ``score.reshape(-1)`` — C/row-major) and
applied to the scheme config as an ``(ncol,)`` array (``apply_feedback_to_scheme``
→ ``reshape(-1)``).  The physics consumes that array via the
:class:`~legoesm.core.grid_adapters.ColumnAdapter`, which flattens the grid the
SAME way (``field.reshape(ncol)``).  If the two flattens ever diverged (a
transpose, F-order, a different grid orientation), the LES coefficient for a
worst column would silently land on the WRONG grid cell — corrupting the whole
correction.  These tests LOCK that contract (Codex iter-35 follow-up).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
from legoesm.core.grid_adapters import ColumnAdapter
from legoesm.training.column_manifest import ColumnEnvironment, ColumnRecord
from legoesm.training.feedback_assembly import assemble_feedback_field
from legoesm.training.promotable_params import apply_feedback_to_scheme


class _Eddy:
    """Minimal eddy-diffusivity diagnosis (reduced to a scalar by the assembler)."""

    def __init__(self, k):
        self.K = jnp.array([k, k])
        self.valid = jnp.array([True, True])


def _record(flat, grid_index):
    return ColumnRecord(
        flat_index=flat, grid_index=grid_index, lat_deg=0.0, lon_deg=0.0,
        time_index=0, combined_score=1.0, T_rmse_K=0.0, qv_rmse_kg_kg=0.0,
        wind_rmse_m_s=0.0, precip_err_mm_day=0.0,
        environment=ColumnEnvironment(sst_K=290.0, cape_J_kg=500.0, bulk_shear_m_s=5.0),
    )


@pytest.mark.parametrize("grid_shape", [(4, 5), (6, 4, 4)])
def test_diagnosis_lands_at_physics_column_for_its_grid_cell(grid_shape):
    """A worst column's diagnosed K ends up at exactly the ``(ncol,)`` index the
    physics ColumnAdapter maps that grid cell to — the end-to-end ordering."""
    ncol = math.prod(grid_shape)
    # Two worst columns at distinct interior grid cells, distinct diagnosed K.
    cells = [
        tuple(d // 2 for d in grid_shape),                       # a central cell
        tuple(min(d - 1, d // 2 + 1) for d in grid_shape),       # a neighbour
    ]
    flats = [int(np.ravel_multi_index(c, grid_shape)) for c in cells]
    assert flats[0] != flats[1]
    records = [_record(flats[0], cells[0]), _record(flats[1], cells[1])]
    diagnoses = [_Eddy(10.0), _Eddy(20.0)]

    field = assemble_feedback_field(
        records, diagnoses, grid_shape, method="eddy_diffusivity", background=0.4)
    corrected = apply_feedback_to_scheme(
        CLUBBLiteConfig(), "clubb_lite_C_K", field, expected_ncol=ncol)
    c_k = np.asarray(corrected.C_K)
    assert c_k.shape == (ncol,)

    # The physics column index of grid cell c is ColumnAdapter's row-major
    # ravel(c) — assert the diagnosed K sits there (and nowhere else does).
    adapter = ColumnAdapter(ncol=ncol, shape_2d=grid_shape)
    assert float(c_k[flats[0]]) == pytest.approx(10.0)
    assert float(c_k[flats[1]]) == pytest.approx(20.0)
    # Every other column keeps the background.
    others = [i for i in range(ncol) if i not in flats]
    np.testing.assert_allclose(c_k[others], 0.4)
    # And reshaping the (ncol,) C_K back through the adapter recovers the grid
    # field placement (round-trip identity ⇒ the two orderings agree).
    np.testing.assert_array_equal(
        np.asarray(adapter.unflatten_2d(corrected.C_K)), np.asarray(field))


@pytest.mark.parametrize("grid_shape", [(8, 16), (6, 8, 8)])
def test_feedback_flatten_equals_adapter_flatten(grid_shape):
    """Decisive guard: the ``(ncol,)`` array the feedback path produces from a
    grid field equals the one the physics ColumnAdapter produces — for an
    arbitrary NON-symmetric field, so any transpose/F-order divergence fails."""
    ncol = math.prod(grid_shape)
    # arange ⇒ every cell distinct; a transposed/F-order flatten would reorder it.
    field = jnp.arange(ncol, dtype=jnp.float64).reshape(grid_shape)

    # Feedback side (what reaches the kernel config).
    corrected = apply_feedback_to_scheme(
        CLUBBLiteConfig(), "clubb_lite_C_K", field, expected_ncol=ncol)
    # Physics side (how the dycore maps grid → columns).
    adapter = ColumnAdapter(ncol=ncol, shape_2d=grid_shape)

    np.testing.assert_array_equal(
        np.asarray(corrected.C_K), np.asarray(adapter.flatten_2d(field)))
    # Sanity: it IS the row-major flatten (the manifest/rank_worst convention).
    np.testing.assert_array_equal(
        np.asarray(corrected.C_K), np.arange(ncol))


def _experiment(grid_type, resolution, nlev=5):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type=grid_type, resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite",
    )


@pytest.mark.parametrize("grid_type,res", [("latlon", 8), ("cubed_sphere", 8)])
def test_real_pipeline_adapter_matches_feedback_flatten(grid_type, res):
    """Stronger than the hand-built guard: the REAL physics pipeline's adapter
    (build_physics_pipeline → make_adapter(grid)) has shape_2d == grid.grid_lat
    shape, and its flatten equals the feedback (ncol,) array (Codex iter-36)."""
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    from legoesm.grids.factory import create_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_grid(grid_type, resolution=res)
    pipe = build_physics_pipeline(grid, create_sigma_coordinate(5),
                                  _experiment(grid_type, res))
    shape_2d = tuple(int(d) for d in grid.grid_lat.shape)
    assert tuple(pipe.adapter.shape_2d) == shape_2d

    ncol = math.prod(shape_2d)
    field = jnp.arange(ncol, dtype=jnp.float64).reshape(shape_2d)
    corrected = apply_feedback_to_scheme(
        CLUBBLiteConfig(), "clubb_lite_C_K", field, expected_ncol=ncol)
    np.testing.assert_array_equal(
        np.asarray(corrected.C_K), np.asarray(pipe.adapter.flatten_2d(field)))


@pytest.mark.slow
def test_compare_state_shape_matches_physics_adapter_real_driver():
    """End-to-end (Codex iter-36 Gap 1): on a REAL coupled driver, the compare
    ColumnState column shape (model.T.shape[:-1], from column_state_from_hydrostatic)
    equals the physics adapter's shape_2d — so a global (ncol,) C_K built at the
    manifest's flat grid order maps to the same columns the physics applies it to."""
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.training.compare_reanalysis import column_state_from_hydrostatic

    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=1.0), radiation="gray",
        turbulence="clubb_lite", days=1,
    )
    driver = CoupledESMDriver(atm, PRESETS["aquaplanet"](),
                              ocean_grid=create_latlon_grid(n_lat=8, n_lon=16))
    driver.setup()  # builds the physics pipeline (+ adapter); NO run needed
    adapter = driver._atm.physics.adapter
    model = column_state_from_hydrostatic(
        driver.state, driver.q_v, sst_K=driver.ocean_state.T_sfc.data)
    assert tuple(model.T.shape[:-1]) == tuple(adapter.shape_2d)
    # Elementwise (not just shape): the physics adapter flattens the COMPARE
    # state's surface field in the SAME order as the manifest convention
    # (reshape(-1)) — so a C_K built at manifest flat order lands on the matching
    # physics columns.  Uses the real driver state (non-vacuous: a transpose/
    # F-order in either would make these differ).
    np.testing.assert_array_equal(
        np.asarray(adapter.flatten_2d(model.p_s)),
        np.asarray(model.p_s.reshape(-1)))
