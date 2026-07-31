"""Per-process budget ledger on the MPAS physics path (#1311 attribution port).

``--budget-ledger`` is an FV compiled-segment diagnostic and REFUSES on the
MPAS lean loop, so the production AMIP lane had no per-process attribution at
all.  These tests pin the physics half of the port:

  1. default OFF is byte-identical (the gate is a static Python bool);
  2. the per-scheme rows SUM to the physics total actually applied -- the
     property that makes the table trustworthy, since a row that does not
     close lets a term hide;
  3. rows land on the RIGHT process (a radiation-only config puts everything
     on the radiation row and nothing on convection);
  4. the row map cannot silently drift out of step with the module list.

Run with JAX_ENABLE_X64=1 (conservation/closure assertions).
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics  # noqa: E402
from legoesm.atmosphere.physics.radiation.config import RadiationConfig  # noqa: E402
from legoesm.atmosphere.physics.convection.config import ConvectionConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig  # noqa: E402
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig  # noqa: E402
from legoesm.diagnostics.process_ledger import (  # noqa: E402
    N_LEDGER, ROW_CONVECTION, ROW_RADIATION, ledger_entry_column,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

NCOL, NLEV = 24, 12


def _cfg(**kw):
    base = dict(
        radiation=RadiationConfig(scheme="gray"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
    )
    base.update(kw)
    return PhysicsConfig(**base)


def _state_and_grid():
    """A minimal MPAS-shaped column state the combined physics can run on."""
    from legoesm.core.state import Field, HydrostaticState
    from legoesm.grids.vertical import create_sigma_coordinate

    rng = np.random.default_rng(0)
    sigma = create_sigma_coordinate(NLEV)
    T = 250.0 + 20.0 * rng.standard_normal((NCOL, NLEV))
    p_s = 1.0e5 + 2.0e3 * rng.standard_normal(NCOL)
    u = rng.standard_normal((NCOL, NLEV))
    q_v = np.abs(rng.standard_normal((NCOL, NLEV))) * 1e-3
    st = HydrostaticState(
        u=Field(data=jnp.asarray(u), name="u", dims=("ncol", "nlev")),
        v=None,
        T=Field(data=jnp.asarray(T), name="T", dims=("ncol", "nlev")),
        p_s=Field(data=jnp.asarray(p_s), name="p_s", dims=("ncol",)),
        phis=Field(data=jnp.zeros(NCOL), name="phis", dims=("ncol",)),
        tracers={"q_v": Field(data=jnp.asarray(q_v), name="q_v",
                              dims=("ncol", "nlev"))},
    )
    return st, sigma, _Mesh(NCOL)


class _Mesh:
    """Minimal Voronoi-shaped mesh: the radiation lat/lon helper dispatches on
    ``latCell`` (integration.py:431), which is how the real MPAS mesh is
    recognised."""

    def __init__(self, ncol):
        self.latCell = jnp.asarray(np.linspace(-1.3, 1.3, ncol))
        self.lonCell = jnp.asarray(np.linspace(0.0, 6.2, ncol))
        self.nCells = ncol


def _run(cfg, ledger):
    st, sigma, mesh = _state_and_grid()
    fn = make_physics(cfg, model_type="mpas", dt=75.0, budget_ledger=ledger)
    out = fn(st, mesh, sigma)
    tend = out[0] if isinstance(out, tuple) else out
    return tend, st, sigma


def test_default_off_carries_no_ledger_and_is_byte_identical():
    """OFF must leave ledger_rows None AND leave the tendencies unchanged."""
    cfg = _cfg()
    off, _, _ = _run(cfg, False)
    on, _, _ = _run(cfg, True)

    assert off.ledger_rows is None, (
        "budget_ledger defaults False -- ledger_rows must stay None")
    assert on.ledger_rows is not None, (
        "budget_ledger=True produced no ledger; the gate is not wired")
    # The physics itself must be untouched by the diagnostic.
    np.testing.assert_array_equal(np.asarray(off.dT_dt.data),
                                  np.asarray(on.dT_dt.data))


def test_ledger_shape_and_finiteness():
    tend, _, _ = _run(_cfg(), True)
    led = np.asarray(tend.ledger_rows)
    assert led.shape == (NCOL, N_LEDGER, 2), led.shape
    assert np.isfinite(led).all()


def test_rows_sum_to_the_applied_physics_total():
    """CLOSURE: sum over processes == the ledger row of the COMBINED tendency.

    This is what makes the table trustworthy -- if the rows did not sum to
    what the model actually applies, a term could hide in the gap.
    """
    # A REAL Voronoi mesh: Louis turbulence reconstructs cell velocity from
    # edge normals (voronoi.reconstruct_cell_velocity), so the stub mesh is
    # not enough.  Level 2 = 162 cells, and lloyd_iterations=0 keeps it fast
    # (mesh QUALITY is irrelevant here -- this asserts an algebraic identity
    # between rows and their sum, not a physical result).
    from legoesm.core.state import Field, HydrostaticState
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2, lloyd_iterations=0)
    ncol = int(mesh.nCells)
    nedge = int(np.asarray(mesh.dvEdge).size)
    rng = np.random.default_rng(11)
    sigma = create_sigma_coordinate(NLEV)
    st = HydrostaticState(
        u=Field(data=jnp.asarray(rng.standard_normal((nedge, NLEV))),
                name="u", dims=("nedge", "nlev")),
        v=None,
        T=Field(data=jnp.asarray(250.0 + 20.0 * rng.standard_normal(
            (ncol, NLEV))), name="T", dims=("ncol", "nlev")),
        p_s=Field(data=jnp.asarray(1.0e5 + 2.0e3 * rng.standard_normal(ncol)),
                  name="p_s", dims=("ncol",)),
        phis=Field(data=jnp.zeros(ncol), name="phis", dims=("ncol",)),
        tracers={"q_v": Field(
            data=jnp.asarray(np.abs(rng.standard_normal((ncol, NLEV))) * 1e-3),
            name="q_v", dims=("ncol", "nlev"))},
    )
    cfg = _cfg(convection=ConvectionConfig(scheme="sbm"),
               turbulence=TurbulenceConfig(scheme="louis"))
    fn = make_physics(cfg, model_type="mpas", dt=75.0, budget_ledger=True)
    out = fn(st, mesh, sigma)
    tend = out[0] if isinstance(out, tuple) else out

    rows_sum = np.asarray(tend.ledger_rows).sum(axis=1)      # (ncol, 2)

    dq = None
    if tend.tracer_tendencies:
        for k in ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g"):
            if k in tend.tracer_tendencies:
                d = tend.tracer_tendencies[k].data
                dq = d if dq is None else dq + d
    total = np.asarray(ledger_entry_column(
        dq, tend.dT_dt.data, st.p_s.data, sigma.dsigma))

    np.testing.assert_allclose(rows_sum, total, rtol=1e-9, atol=1e-20)


def test_rows_land_on_the_right_process():
    """A radiation-only config must put its energy on the RADIATION row and
    leave convection at exactly zero -- otherwise the row map is mislabelled
    and every attribution built on it would name the wrong scheme."""
    tend, _, _ = _run(_cfg(), True)
    led = np.asarray(tend.ledger_rows)

    assert np.abs(led[:, ROW_RADIATION, 1]).max() > 0.0, (
        "radiation is active but its energy row is all zero")
    np.testing.assert_array_equal(led[:, ROW_CONVECTION, :],
                                  np.zeros((NCOL, 2)))
    # Radiation moves no water.
    np.testing.assert_array_equal(led[:, ROW_RADIATION, 0], np.zeros(NCOL))


def test_row_map_drift_guard_is_not_vacuous():
    """The factory asserts one ledger row per physics module.  Prove the guard
    exists by reading it back from source -- a silent drift would
    mis-attribute every module after the missing one."""
    import inspect

    from legoesm.atmosphere.physics import combined
    src = inspect.getsource(combined._make_hydrostatic_combined)
    assert "_ledger_row_of" in src
    assert "drifted" in src, (
        "the row-map/module-count drift guard is gone -- a module added "
        "without a row would silently shift every later attribution")


def test_number_tracers_are_excluded_from_the_water_column():
    """N_c/N_i/N_r are [#/kg] or [#/m^3]; adding them to a [kg/kg] water
    budget is a units error.  Pin the canonical species tuple by VALUE -- this
    is the exclusion whose omission caused the 2026-07 conserving-borrow
    defect."""
    from legoesm.diagnostics.process_ledger import LEDGER_WATER_SPECIES

    assert LEDGER_WATER_SPECIES == ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g")
    for bad in ("N_c", "N_i", "N_r", "N_s", "N_g"):
        assert bad not in LEDGER_WATER_SPECIES


def test_grads_survive_the_ledger():
    """The ledger rides inside a differentiable step; enabling it must not
    break AD through the physics."""
    cfg = _cfg()
    st, sigma, mesh = _state_and_grid()
    fn = make_physics(cfg, model_type="mpas", dt=75.0, budget_ledger=True)

    def loss(T):
        s = st._replace(T=st.T.replace(data=T))
        out = fn(s, mesh, sigma)
        tend = out[0] if isinstance(out, tuple) else out
        return jnp.sum(tend.dT_dt.data)

    g = jax.grad(loss)(st.T.data)
    assert np.isfinite(np.asarray(g)).all()
