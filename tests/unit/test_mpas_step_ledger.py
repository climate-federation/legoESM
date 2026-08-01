"""Step-level budget ledger on the MPAS lane (#1311, dycore half).

The physics half (test_mpas_physics_ledger) pins that the per-scheme rows sum
to the applied physics.  These tests pin the STEP-level property that makes
the whole table trustworthy for the detonation question:

    sum over ALL rows (physics + dynamics + clips + residual)
        == (column_store(final) - column_store(pre-step)) / dt,  PER COLUMN.

If that identity holds, no term can hide: whatever heats/moistens a runaway
column appears in a named row.

Also pinned: ledger OFF leaves the model state BIT-IDENTICAL and the
side-channel None (the diagnostic must not perturb the trajectory).

Closure assertions: x64 REQUIRED and enforced below (under fp32, #1424's
layer-mass op-order change moves answers by ~1e-6 rel — float noise).
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics  # noqa: E402
from legoesm.atmosphere.physics.radiation.config import RadiationConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig  # noqa: E402
from legoesm.diagnostics.process_ledger import (  # noqa: E402
    LEDGER_WATER_SPECIES, N_LEDGER, column_store_snapshot_column,
)

DT = 75.0
NLEV = 8


def _model_and_state(conservative_clamp=False):
    """Small real MPAS model + a moist state with deliberate q undershoots."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, MPASPrimitiveEquationModel,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2, lloyd_iterations=0)
    sigma = create_sigma_coordinate(NLEV)
    cfg = MPASPrimitiveEquationConfig(
        conservative_tracer_clamp=conservative_clamp)
    model = MPASPrimitiveEquationModel(mesh, sigma, cfg)

    ncol = int(mesh.nCells)
    rng = np.random.default_rng(3)
    state = held_suarez_init_mpas(mesh, sigma)
    # q_v with a few NEGATIVE cells so the floors/clips row is non-trivially
    # exercised, not just zero (mean 4e-4, sd 2e-4 => ~2.3% negative).
    qv = rng.standard_normal((ncol, NLEV)) * 2e-4 + 4e-4
    state = state._replace(tracers={
        "q_v": state.p_s.replace(data=jnp.asarray(qv))})
    return model, state, sigma


def _physics(ledger):
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray"),
        turbulence=TurbulenceConfig(scheme="louis"),
    )
    return make_physics(cfg, model_type="mpas", dt=DT, budget_ledger=ledger)


def _snap(state, sigma):
    water = [state.tracers[k].data for k in LEDGER_WATER_SPECIES
             if state.tracers is not None and k in state.tracers]
    return column_store_snapshot_column(
        state.p_s.data, sigma.dsigma, state.T.data, *water)


def test_ledger_off_is_bit_identical_and_channel_none():
    model_a, state, sigma = _model_and_state()
    model_b, _, _ = _model_and_state()

    out_off = model_a.step(state, DT, physics_fn=_physics(False),
                           phys_state=None)
    out_on = model_b.step(state, DT, physics_fn=_physics(True),
                          phys_state=None)

    assert getattr(model_a, "_step_ledger", None) is None
    assert getattr(model_b, "_step_ledger", None) is not None
    # The diagnostic must not perturb the trajectory.  NOT asserted bitwise:
    # adding the ledger outputs to the jitted graph changes XLA fusion and
    # hence ROUNDING of the same math (measured max rel diff 4.6e-12 on this
    # fixture) — the same reason enabling the FV SegmentCarry ledger recompiles
    # its segment.  The repo contract is "default OFF == byte-identical to the
    # pre-change code" (guaranteed structurally: the static gate excludes every
    # ledger op from the OFF graph).  What this pins is that the ledger has no
    # FEEDBACK into the state: a real coupling (an accidental state update from
    # a ledger intermediate) would show at O(tendency*dt), many orders above
    # this tolerance.
    np.testing.assert_allclose(np.asarray(out_off.T.data),
                               np.asarray(out_on.T.data), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(out_off.p_s.data),
                               np.asarray(out_on.p_s.data), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(out_off.tracers["q_v"].data),
                               np.asarray(out_on.tracers["q_v"].data),
                               rtol=1e-9, atol=1e-18)


@pytest.mark.parametrize("clamp", [False, True],
                         ids=["naive-clip", "conserving-borrow"])
def test_full_step_ledger_closes_per_column(clamp):
    """THE property: all rows sum to the actual per-column store change."""
    model, state, sigma = _model_and_state(conservative_clamp=clamp)
    before = _snap(state, sigma)

    out = model.step(state, DT, physics_fn=_physics(True), phys_state=None)
    led = np.asarray(model._step_ledger)
    after = _snap(out, sigma)

    ncol = before.shape[0]
    assert led.shape == (ncol, N_LEDGER, 2)
    assert np.isfinite(led).all()

    actual = (np.asarray(after) - np.asarray(before)) / DT
    rows_sum = led.sum(axis=1)
    # rtol tight (the residual row makes closure exact by construction);
    # atol covers exact-zero water columns.
    np.testing.assert_allclose(rows_sum, actual, rtol=1e-8, atol=1e-16)


def test_clips_row_reports_the_naive_clip_water_invention():
    """With the naive maximum(0) clip and deliberate negative q cells, the
    clips row must show POSITIVE water in the affected columns -- the
    measured 2026-07 invention mechanism.  Non-vacuous: if the clips row
    were dropped, this max would be ~0."""
    from legoesm.diagnostics.process_ledger import ROW_CLIPS

    model, state, sigma = _model_and_state(conservative_clamp=False)
    n_neg = int((np.asarray(state.tracers["q_v"].data) < 0).sum())
    assert n_neg > 0, "fixture no longer has negative q_v cells"

    model.step(state, DT, physics_fn=_physics(True), phys_state=None)
    led = np.asarray(model._step_ledger)
    assert led[:, ROW_CLIPS, 0].max() > 0.0, (
        "negative q_v cells were clipped but the clips row shows no "
        "invented water")
