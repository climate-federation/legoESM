"""Differentiable MPAS step exported from the production run loop (review F9).

``ModelDriver.export_mpas_step`` hands back the closures ``_run_column`` builds;
``mpas_diff_step.mpas_segment`` advances them as a pure function.  Pinned:

* the segment reproduces the eager run loop BITWISE over a window that
  crosses every cadence branch (full physics + radiation, held radiation,
  held physics) with the post-step edits (top sponge, q_v smoother) on;
* ``jax.grad`` through the segment is finite and matches a central
  difference (the reverse pass actually reaches the initial state).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.driver.config import (  # noqa: E402
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402
from legoesm.driver.mpas_diff_step import mpas_cadence, mpas_segment  # noqa: E402

DT, N = 600.0, 4


def _driver(tmp_path, name):
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=2, nlev=8,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(dt=DT, discretization="mpas"),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=(N + 0.5) * DT / 86400.0, start_day=0.0,
        dataset="analytical", radiation="gray",
        convection="sbm", turbulence="tke", microphysics="kessler",
        physics_update_steps=2, rad_update_steps=4,
        mpas_qv_smooth_del2_m2s=1.0e5, sponge_enabled=True,
        precision="fp64", distributed=False,
    )
    d = ModelDriver(cfg, output_dir=str(tmp_path / name))
    d.setup()
    return d


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mpas_diff")
    eager = _driver(tmp, "eager")
    assert eager.run() == "COMPLETED"
    drv = _driver(tmp, "export")
    exp = drv.export_mpas_step()
    exp["driver"] = drv          # test-only handle
    return eager, exp


def test_exported_driver_refuses_a_later_run(runs):
    """export_mpas_step's setup consumes restart side data (land stress), so
    the driver is single-use: a later run() must refuse, not run degraded."""
    _, exp = runs
    with pytest.raises(RuntimeError, match="consumed by export_mpas_step"):
        exp["driver"].run()
    with pytest.raises(RuntimeError, match="consumed by an earlier export"):
        exp["driver"].export_mpas_step()


def test_cadence_covers_every_branch():
    kw = dict(subcycle_rad=True, rad_update_steps=4, hold_phys=True,
              phys_update_steps=2)
    assert [mpas_cadence(s, **kw) for s in range(N)] == [
        (True, True), (False, False), (False, True), (False, False)]


def test_segment_matches_eager_loop_bitwise(runs):
    eager, exp = runs
    assert exp["physics_fn_held"] is not None and exp["physics_fn_norad"] is not None
    s, ps = mpas_segment(exp, exp["state"], exp["phys_state"], N)
    for k in ("u", "T", "p_s"):
        np.testing.assert_array_equal(np.asarray(getattr(eager.state, k).data),
                                      np.asarray(getattr(s, k).data), err_msg=k)
    for k, f in eager.state.tracers.items():
        np.testing.assert_array_equal(np.asarray(f.data),
                                      np.asarray(s.tracers[k].data), err_msg=k)
    # the post-step edits are live in this window (else bitwise is vacuous)
    assert float(jnp.abs(exp["poststep"](s).u.data - s.u.data).max()) > 0.0


def _fields(state):
    """The continuous prognostic fields, incl. the two the post-step edits
    touch (winds: top sponge; q_v: smoother)."""
    return [state.u.data, state.T.data, state.p_s.data,
            state.tracers["q_v"].data]


def segment_loss(exp, seed=0):
    """L(T0): random projection of the end-of-segment u, T, p_s, q_v, each as
    its departure from the unperturbed run over its RMS (so a gradient cut on
    any of them shows and no field's magnitude swamps fp64).  The physics
    carry enters through the held-physics steps (1 and 3 re-apply it).  The
    hydrometeor tracers are left out: switching on from zero they put O(10)
    jumps into an RMS-normalised projection at any step size (measured)."""
    s0, ps0 = exp["state"], exp["phys_state"]
    ref = _fields(mpas_segment(exp, s0, ps0, N)[0])
    rng = np.random.default_rng(seed)
    ws = [jnp.asarray(rng.standard_normal(np.shape(x)))
          / float(jnp.sqrt(jnp.mean(jnp.square(x)))) for x in ref]

    def loss(T):
        out = _fields(mpas_segment(exp, s0._replace(T=s0.T.replace(data=T)),
                                   ps0, N, checkpoint=True)[0])
        return sum(jnp.sum(w * (x - r)) for w, x, r in zip(ws, out, ref))
    return loss


def test_segment_gradient_matches_central_difference(runs):
    _, exp = runs
    s0 = exp["state"]
    rng = np.random.default_rng(0)
    v = jnp.asarray(0.1 * rng.standard_normal(s0.T.data.shape))
    loss = segment_loss(exp)
    g = jax.grad(loss)(s0.T.data)
    assert bool(jnp.all(jnp.isfinite(g)))
    ad = float(jnp.vdot(g, v))
    # Measured (fp64, AD = -34.25): FD/AD - 1 = -1.1e-3, -1.3e-4, 1.7e-4,
    # -8.6e-8, -2.2e-2 at h = 1e-1, 1e-2, 1e-3, 1e-4, 1e-5.  The loss along
    # x + s v is deterministic (repeats bitwise) but carries JUMPS of
    # <= 4.5e-5 at any span down to 1e-5 (threshold branches in the physics
    # flipping), which no gradient sees and which give D(h) = FD*2h - AD*2h
    # independent of h; a wrong adjoint gives D proportional to h instead.  At
    # h = 1e-2 the jumps contribute <= 7e-5 of |AD| and truncation 1.3e-4, so
    # 1e-3 bounds the adjoint error without resting on a lucky h.
    h = 1e-2
    fd = float((loss(s0.T.data + h * v) - loss(s0.T.data - h * v)) / (2 * h))
    assert abs(fd - ad) <= 1e-3 * abs(ad), (ad, fd)


def test_segment_gradient_reaches_the_physics_carry(runs):
    """The INITIAL physics carry is threaded into the reverse pass: the end
    state has a finite, non-zero gradient w.r.t. the carried prognostic TKE
    (exactly zero with the carry detached).  Reachability only: TKE starts at
    its floor, where FD does not resolve the value, and a carry detached only
    between later steps moves the T-gradient by ~3e-4 here, under what FD
    resolves through the physics jumps.  (The production probe's 16-step
    segment, whose 15 held-physics steps re-apply the step-0 carry, matched
    FD to FD/AD - 1 = 1.5e-5.)"""
    _, exp = runs
    s0, ps0 = exp["state"], exp["phys_state"]

    def loss(k):
        s, _ = mpas_segment(exp, s0, ps0._replace(tke=k), N, checkpoint=True)
        return sum(jnp.sum(x) for x in _fields(s))
    g = jax.grad(loss)(ps0.tke)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0.0


def test_qv_smoother_floor_reverse_pass_finite_on_a_dry_level():
    """The post-step smoother's water-mass floor (restore_level_mass) is in
    the differentiated segment: a level whose filtered field is entirely
    non-positive must not NaN the reverse pass (merge review, GLM)."""
    from legoesm.core.conservation import restore_level_mass
    q_new = jnp.asarray([[1e-3, -1e-6], [2e-3, -2e-6], [5e-4, 0.0]])
    dp = jnp.full((3, 2), 5.0e3)
    g = jax.grad(lambda q: jnp.sum(restore_level_mass(
        q, q_new, dp, jnp.ones(3), clip=True)))(q_new)
    assert bool(jnp.all(jnp.isfinite(g)))
