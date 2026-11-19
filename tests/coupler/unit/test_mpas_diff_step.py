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
from legoesm.driver.mpas_diff_step import (  # noqa: E402
    clock_tables,
    mpas_branch_index,
    mpas_cadence,
    mpas_segment,
    pick_physics_fn,
)

DT, N = 600.0, 4
# Scheduled vs unrolled loop form, both relative to the field's max|.|.
# Measured (res2 L8 fp64, 10 steps): the forward differs by 4.9e-13 and the
# gradient by 1.5e-15 -- the same 4.9e-13 separates the unrolled segment
# traced under one jax.jit from its eager per-step form, so the forward gap is
# whole-trace XLA fusion roundoff, not the switch; a single step is bitwise.
FWD_TOL = 1e-11
GRAD_TOL = 1e-12


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


def segment_loss(exp, seed=0, n_steps=N, **segment_kw):
    """L(T0): random projection of the end-of-segment u, T, p_s, q_v, each as
    its departure from the unperturbed run over its RMS (so a gradient cut on
    any of them shows and no field's magnitude swamps fp64).  The physics
    carry enters through the held-physics steps (1 and 3 re-apply it).  The
    hydrometeor tracers are left out: switching on from zero they put O(10)
    jumps into an RMS-normalised projection at any step size (measured).
    ``segment_kw`` selects the loop form (default: unrolled, per-step remat);
    the reference run is always the unrolled segment."""
    s0, ps0 = exp["state"], exp["phys_state"]
    ref = _fields(mpas_segment(exp, s0, ps0, n_steps)[0])
    rng = np.random.default_rng(seed)
    ws = [jnp.asarray(rng.standard_normal(np.shape(x)))
          / float(jnp.sqrt(jnp.mean(jnp.square(x)))) for x in ref]
    segment_kw = segment_kw or {"checkpoint": True}

    def loss(T):
        out = _fields(mpas_segment(exp, s0._replace(T=s0.T.replace(data=T)),
                                   ps0, n_steps, **segment_kw)[0])
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


def _rad_carry_grad(exp, step0, **kw):
    """d(sum of the end state's prognostic fields) / d(the carried cached
    radiative heating) over 3 steps from ``step0``."""
    s0, ps0 = exp["state"], exp["phys_state"]

    def loss(r):
        s, _ = mpas_segment(exp, s0, ps0._replace(rad_heating=r), 3,
                            step0=step0, **kw)
        return sum(jnp.sum(x) for x in _fields(s))
    return jax.grad(loss)(ps0.rad_heating)


def test_segment_gradient_reaches_the_physics_carry(runs):
    """The INITIAL physics carry is threaded into the reverse pass: from step
    1 the held-radiation steps re-apply the carried cached radiative heating,
    so the end state has a finite, non-zero gradient w.r.t. it; from step 0
    the full-radiation step overwrites the carry first and the gradient is
    exactly zero (the control: a detached carry gives 0).  (Previously keyed
    on the carried TKE; since turbulence='tke' routes to the WRF MYJ port the
    end state does not depend on the carried TKE in this fixture -- its
    gradient is exactly 0 even with TKE well above its floor, on main too.)"""
    _, exp = runs
    g = _rad_carry_grad(exp, 1, checkpoint=True)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0.0
    assert float(jnp.abs(_rad_carry_grad(exp, 0, checkpoint=True)).max()) == 0.0


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


# ---------------------------------------------------------------------------
# Scheduled (fixed-length, lax.switch) loop form vs the unrolled segment
# ---------------------------------------------------------------------------

# Two cadence periods (radiation every 4) plus a 2-step tail: every variant
# runs more than once and the window is not a multiple of the period.
N_SCHED = 10
SCHEDULES = [("uniform", None), ("binomial", 2)]   # K=2 forces Treeverse recompute


def test_branch_index_is_pick_physics_fn_on_indices():
    kw = dict(subcycle_rad=True, rad_update_steps=4, hold_phys=True,
              phys_update_steps=2)
    exp = dict(kw, physics_fn="full", physics_fn_norad="norad",
               physics_fn_held="held")
    names = ("full", "norad", "held")
    for s in range(2 * 4):
        assert names[mpas_branch_index(s, exp)] == pick_physics_fn(
            *mpas_cadence(s, **kw), "full", "norad", "held")
    assert [mpas_branch_index(s, exp) for s in range(4)] == [0, 2, 1, 2]


@pytest.mark.parametrize("schedule,checkpoints", SCHEDULES)
@pytest.mark.parametrize("n", [1, N_SCHED])
def test_scheduled_segment_forward_matches_unrolled(runs, schedule,
                                                    checkpoints, n):
    """n=1 uses one variant (no switch is built) and is bitwise; n=N_SCHED
    switches through all three and is compared at the measured whole-trace
    fusion roundoff (FWD_TOL)."""
    _, exp = runs
    s0, ps0 = exp["state"], exp["phys_state"]
    ref, pref = mpas_segment(exp, s0, ps0, n)
    out, pout = jax.jit(lambda s, p: mpas_segment(
        exp, s, p, n, schedule=schedule, checkpoints=checkpoints))(s0, ps0)
    fields = list(zip(_fields(ref), _fields(out), ("u", "T", "p_s", "q_v")))
    fields.append((pref.tke, pout.tke, "tke carry"))
    for a, b, k in fields:
        a, b = np.asarray(a), np.asarray(b)
        if n == 1:
            np.testing.assert_array_equal(a, b, err_msg=k)
        else:
            np.testing.assert_allclose(b, a, rtol=0.0,
                                       atol=FWD_TOL * np.abs(a).max(), err_msg=k)


@pytest.mark.parametrize("schedule,checkpoints", SCHEDULES)
def test_scheduled_segment_gradient_matches_unrolled(runs, schedule,
                                                     checkpoints):
    _, exp = runs
    t0 = exp["state"].T.data
    g_ref = jax.grad(segment_loss(exp, n_steps=N_SCHED, checkpoint=True))(t0)
    g = jax.jit(jax.grad(segment_loss(
        exp, n_steps=N_SCHED, schedule=schedule, checkpoints=checkpoints)))(t0)
    assert bool(jnp.all(jnp.isfinite(g)))
    scale = float(jnp.abs(g_ref).max())
    assert scale > 0.0
    np.testing.assert_allclose(np.asarray(g), np.asarray(g_ref),
                               rtol=0.0, atol=GRAD_TOL * scale)


@pytest.mark.parametrize("schedule,checkpoints", SCHEDULES)
def test_scheduled_segment_gradient_reaches_the_physics_carry(runs, schedule,
                                                              checkpoints):
    """The scheduled loop threads the initial physics carry into the reverse
    pass exactly like the unrolled loop (see the unrolled test for the
    carry, the step-1 start and the step-0 control)."""
    _, exp = runs
    ref = _rad_carry_grad(exp, 1, checkpoint=True)
    g = _rad_carry_grad(exp, 1, schedule=schedule, checkpoints=checkpoints)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0.0
    np.testing.assert_allclose(np.asarray(g), np.asarray(ref), rtol=0.0,
                               atol=GRAD_TOL * float(jnp.abs(ref).max()))
    g0 = _rad_carry_grad(exp, 0, schedule=schedule, checkpoints=checkpoints)
    assert float(jnp.abs(g0).max()) == 0.0


def test_scheduled_segment_refuses_unknown_schedule_and_nested_checkpoint(runs):
    _, exp = runs
    s0, ps0 = exp["state"], exp["phys_state"]
    with pytest.raises(ValueError, match="unknown schedule"):
        mpas_segment(exp, s0, ps0, 1, schedule="sqrt")
    with pytest.raises(ValueError, match="checkpoint=True"):
        mpas_segment(exp, s0, ps0, 1, schedule="uniform", checkpoint=True)


def test_segment_tap_sees_every_step_and_is_identity(runs):
    """The per-step hook receives the step index of every unrolled step, its
    output is what the step consumes (an identity hook leaves the segment
    bitwise unchanged), and the scheduled loop refuses it."""
    _, exp = runs
    s0, ps0 = exp["state"], exp["phys_state"]
    seen = []

    def tap(step, s, p):
        seen.append(step)
        return s, p
    ref = _fields(mpas_segment(exp, s0, ps0, N, step0=2)[0])
    out = _fields(mpas_segment(exp, s0, ps0, N, step0=2, tap=tap)[0])
    assert seen == [2, 3, 4, 5]
    for r, o in zip(ref, out):
        assert bool(jnp.all(o == r))
    # a custom_vjp identity in the hook rides through per-step remat + grad
    hits = jax.new_ref(jnp.zeros(N))

    @jax.custom_vjp
    def stash(step, s, p):
        return s, p

    def _fwd(step, s, p):
        return (s, p), step

    def _bwd(step, g):
        hits[step] = 1.0
        return None, g[0], g[1]
    stash.defvjp(_fwd, _bwd)

    def loss(T):
        s, _ = mpas_segment(exp, s0._replace(T=s0.T.replace(data=T)), ps0, N,
                            checkpoint=True, tap=stash)
        return jnp.sum(s.T.data)
    g = jax.jit(jax.grad(loss))(s0.T.data)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert bool(jnp.all(hits[...] == 1.0))
    with pytest.raises(ValueError, match="tap"):
        mpas_segment(exp, s0, ps0, N, schedule="uniform", tap=tap)


def test_clock_tables_advance_model_seconds_like_the_eager_clock(runs):
    """The scheduled loop's clock tables carry every key the unrolled loop's
    per-step clock advances -- including ``model_seconds`` (the ecrad_ifs
    McICA seed clock) when the forcing carries it -- from the same sources,
    at the same values; and they omit it when the forcing does not."""
    _, exp = runs
    steps = range(3, 9)
    forcing = dict(exp["forcing"] or {}, model_seconds=0.0)
    t = clock_tables(exp, forcing, steps)
    assert set(t) == {"day_of_year", "seconds_of_day", "model_seconds"}
    for k, s in enumerate(steps):
        doy, sod = exp["calendar"](s)
        assert float(t["day_of_year"][k]) == float(jnp.asarray(doy))
        assert float(t["seconds_of_day"][k]) == float(jnp.asarray(sod))
        assert float(t["model_seconds"][k]) == float(jnp.asarray(exp["model_seconds"](s)))
    assert len(set(np.asarray(t["model_seconds"]).tolist())) == len(steps)
    t0 = clock_tables(exp, dict(exp["forcing"] or {}), steps)
    assert "model_seconds" not in t0
    assert clock_tables(exp, None, steps) is None


@pytest.mark.parametrize("schedule,checkpoints", SCHEDULES)
def test_scheduled_segment_from_a_nonzero_step0_matches_unrolled(runs, schedule,
                                                                  checkpoints):
    """step0 = 1, 3 steps: a strict subset of the variants, the variant table
    and the clock indexed from a non-zero start."""
    _, exp = runs
    s0, ps0 = exp["state"], exp["phys_state"]
    ref, _ = mpas_segment(exp, s0, ps0, 3, step0=1)
    out, _ = jax.jit(lambda s, p: mpas_segment(
        exp, s, p, 3, step0=1, schedule=schedule, checkpoints=checkpoints))(s0, ps0)
    for a, b, k in zip(_fields(ref), _fields(out), ("u", "T", "p_s", "q_v")):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(b, a, rtol=0.0, atol=FWD_TOL * np.abs(a).max(),
                                   err_msg=k)


def test_segment_refuses_schedule_arguments_it_would_ignore(runs):
    _, exp = runs
    s0, ps0 = exp["state"], exp["phys_state"]
    with pytest.raises(ValueError, match="checkpoints/storage"):
        mpas_segment(exp, s0, ps0, 1, checkpoints=4)
    with pytest.raises(ValueError, match="checkpoints/storage"):
        mpas_segment(exp, s0, ps0, 1, storage="host")
    with pytest.raises(ValueError, match="binomial budget"):
        mpas_segment(exp, s0, ps0, 1, schedule="uniform", checkpoints=4)
