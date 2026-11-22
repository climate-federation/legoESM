"""Budget ledger on a HYBRID column: rows weighted by the true layer masses.

On a hybrid grid ``dp = dA*p_ref + dB*p_s``, and ``p_s*dsigma`` is wrong by
``dA*(p_s - p_ref)``.  The MPAS lane used to book every ledger row with
``p_s*dsigma``; this pins that the rows now sum to the column store change
measured with the coordinate's own ``layer_thickness_dp``, over uneven surface
pressure (terrain), and that the old weighting would not.

Needs JAX_ENABLE_X64=1: in float32 a one-step store difference of whole
columns is below the state's resolution (measured 0.24 relative mismatch),
so the closure can only be pinned in float64.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics  # noqa: E402
from legoesm.atmosphere.physics.convection.config import ConvectionConfig  # noqa: E402
from legoesm.atmosphere.physics.radiation.config import RadiationConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig  # noqa: E402
from legoesm.core.conservation import conservation_accumulator  # noqa: E402
from legoesm.diagnostics.process_ledger import (  # noqa: E402
    LEDGER_WATER_SPECIES, column_store_snapshot_column,
)

DT = 75.0
NLEV = 8

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="ledger closure on a hybrid grid needs JAX_ENABLE_X64=1")


def _hybrid_model_and_state():
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, MPASPrimitiveEquationModel,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas,
    )
    from legoesm.grids.vertical import create_hybrid_coordinate, create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2, lloyd_iterations=0)
    # 8 levels like the sigma ledger test, with part of every interior level
    # moved into the pressure term: A = 0.2 s (1 - s), B = s - A
    s_half = np.asarray(create_sigma_coordinate(NLEV).sigma_half, dtype=np.float64)
    a_half = 0.2 * s_half * (1.0 - s_half)
    coord = create_hybrid_coordinate(NLEV, jnp.asarray(a_half),
                                     jnp.asarray(s_half - a_half))
    model = MPASPrimitiveEquationModel(
        mesh, coord, MPASPrimitiveEquationConfig(conservative_tracer_clamp=True))
    state = held_suarez_init_mpas(mesh, coord)
    ncol, nlev = state.T.data.shape
    rng = np.random.default_rng(5)
    # uneven surface pressure so dA*(p_s - p_ref) is not zero (+-3 percent;
    # without terrain a larger spread is not a state the dynamics can carry)
    ps = 1.0e5 * (0.97 + 0.06 * rng.random(ncol))
    qv = np.clip(rng.standard_normal((ncol, nlev)) * 2e-4 + 4e-4, 1e-7, None)
    state = state._replace(p_s=state.p_s.replace(data=jnp.asarray(ps)),
                           tracers={"q_v": state.p_s.replace(data=jnp.asarray(qv))})
    return model, state, coord


def _snap(state, coord, true_dp):
    water = [state.tracers[k].data for k in LEDGER_WATER_SPECIES
             if state.tracers is not None and k in state.tracers]
    dp = (coord.layer_thickness_dp(state.p_s.data.astype(conservation_accumulator()))
          if true_dp else None)
    return column_store_snapshot_column(
        state.p_s.data, coord.dsigma, state.T.data, *water, dp=dp)


def test_hybrid_ledger_closes_with_true_layer_masses():
    model, state, coord = _hybrid_model_and_state()
    physics = make_physics(
        # Kuo convection too: a water-moving scheme, and without it Louis
        # alone leaves 5 of 162 columns non-finite on this hybrid grid
        # (a separate issue, also with the ledger off)
        PhysicsConfig(radiation=RadiationConfig(scheme="gray"),
                      turbulence=TurbulenceConfig(scheme="louis"),
                      convection=ConvectionConfig(scheme="kuo")),
        model_type="mpas", dt=DT, budget_ledger=True)
    before_true, before_old = _snap(state, coord, True), _snap(state, coord, False)
    out = model.step(state, DT, physics_fn=physics, phys_state=None)
    assert np.isfinite(np.asarray(out.T.data)).all()
    rows = np.asarray(model._step_ledger).sum(axis=1)
    assert np.isfinite(rows).all()
    actual_true = (np.asarray(_snap(out, coord, True)) - np.asarray(before_true)) / DT
    actual_old = (np.asarray(_snap(out, coord, False)) - np.asarray(before_old)) / DT
    np.testing.assert_allclose(rows, actual_true, rtol=1e-8, atol=1e-16)
    # the old p_s*dsigma weighting measures a different store change here
    assert np.abs(actual_old - actual_true).max() > 1e3 * np.abs(rows - actual_true).max()


_ZM_POS = ConvectionConfig(
    scheme="zhang_mcfarlane",
    zhang_mcfarlane=ConvectionConfig().zhang_mcfarlane._replace(positivity_after_zm=True))


@pytest.mark.parametrize("convection, n_macmic", [
    (ConvectionConfig(scheme="kuo"), 1),
    (_ZM_POS, 2),   # + the positivity-after-ZM correction row
])
def test_physics_module_rows_use_true_layer_masses(monkeypatch, convection, n_macmic):
    """Every per-module physics row is weighted with the column's true layer
    masses: each ledger_entry_column call from the physics carries
    dp == layer_thickness_dp(p_s).  The step-total test above cannot see
    this, because the closure residual goes to other_physics.  Row routing
    is main's and is not changed here."""
    import legoesm.diagnostics.process_ledger as pl
    model, state, coord = _hybrid_model_and_state()
    real = pl.ledger_entry_column
    has_dp, diffs = [], []

    def spy(dq, dT, p_s, dsigma, dp=None, level_weight=None):
        has_dp.append(dp is not None)          # static under jit
        if dp is not None:
            want = coord.layer_thickness_dp(p_s.astype(conservation_accumulator()))
            jax.debug.callback(lambda d: diffs.append(float(d)),
                               jnp.max(jnp.abs(dp - want)))
        return real(dq, dT, p_s, dsigma, dp=dp, level_weight=level_weight)

    monkeypatch.setattr(pl, "ledger_entry_column", spy)
    physics = make_physics(
        PhysicsConfig(radiation=RadiationConfig(scheme="gray"),
                      turbulence=TurbulenceConfig(scheme="louis"),
                      convection=convection),
        model_type="mpas", dt=DT, budget_ledger=True, cld_macmic_num_steps=n_macmic)
    model.step(state, DT, physics_fn=physics, phys_state=None)
    jax.effects_barrier()
    assert len(has_dp) >= 3 and all(has_dp)   # radiation, turbulence, convection
    assert diffs and max(diffs) == 0.0
