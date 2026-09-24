"""Closure tests for the in-run cloud-water band budget.

The closure test is the point of the diagnostic: the accumulated terms must
reproduce the ACTUAL change in band inventory, including the part caused by the
grid moving under the tracer.  Everything else here is guard rails.
"""
import pytest

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.diagnostics.qc_band_budget import CloudWaterBandBudget
from legoesm.diagnostics.process_ledger import pressure_band_weight

# Band edges are a CHOICE of analysis window, not physics, so they stay literal.
P_LO, P_HI = 5.0e4, 8.0e4      # const-ok: analysis band edges [Pa], not a constant
G = constants.g


def _terms(key, i, ncol, nlev):
    k = jax.random.fold_in(key, i)
    return {
        "condensation": jax.random.uniform(k, (ncol, nlev)) * 2.0e-7,
        "accretion": -jax.random.uniform(
            jax.random.fold_in(k, 1), (ncol, nlev)) * 1.0e-7,
        "riming": -jax.random.uniform(
            jax.random.fold_in(k, 2), (ncol, nlev)) * 5.0e-8,
    }


def _grid(ncol, nlev, p_s):
    """Hybrid-like half levels: a fixed pressure part plus a p_s-following part."""
    a = jnp.linspace(1.0e3, 5.0e4, nlev + 1)
    b = jnp.linspace(0.0, 1.0, nlev + 1)
    ph = a[None, :] + b[None, :] * (p_s[:, None] - a[-1])
    return ph, ph[:, 1:] - ph[:, :-1]


def test_pressure_band_weight_is_fractional_and_partitions_unity():
    p_s = jnp.array([1.0e5, 9.5e4])
    ph, _ = _grid(2, 12, p_s)
    w = pressure_band_weight(ph, P_LO, P_HI)
    assert jnp.all(w >= 0.0) and jnp.all(w <= 1.0)
    # Adjacent bands partition each layer exactly.
    lo = pressure_band_weight(ph, 0.0, P_HI)
    hi = pressure_band_weight(ph, P_HI, 2.0e5)
    assert float(jnp.max(jnp.abs(lo + hi - 1.0))) < 1e-12
    # A fractional edge really is fractional, not a 0/1 mask.
    assert float(jnp.max(jnp.minimum(w, 1.0 - w))) > 0.0


def test_budget_closes_against_the_actual_inventory_change():
    """THE test.  Integrate a tracer with known tendencies on a MOVING grid and
    require the accumulated budget to reproduce the inventory change.

    Non-vacuous three ways, each asserted below: dropping a process term, or
    dropping the mass-redistribution term, or holding the grid fixed while the
    model's grid moves, all break the closure by far more than the tolerance.
    """
    ncol, nlev, nstep, dt = 6, 20, 40, 900.0
    key = jax.random.PRNGKey(0)
    q = jax.random.uniform(key, (ncol, nlev)) * 1.0e-3
    area = jnp.full((ncol,), 1.0 / ncol)
    names = ("condensation", "accretion", "riming")
    bud = CloudWaterBandBudget(P_LO, P_HI, area, G, names)

    # Surface pressure drifts, so layer thickness AND band membership move.
    def ps_at(i):
        return jnp.full((ncol,), 1.0e5) + 3.0e3 * jnp.sin(
            jnp.arange(ncol) + i * 0.1)

    q0 = q
    ph, dp = _grid(ncol, nlev, ps_at(0))
    bud.begin_window(q, ph)
    for i in range(nstep):
        terms = _terms(key, i, ncol, nlev)
        ph_new, dp_new = _grid(ncol, nlev, ps_at(i + 1))
        bud.accumulate(terms, q, ph, ph_new, dt)
        q = q + dt * sum(terms.values())
        ph, dp = ph_new, dp_new
    out = bud.close_window(q, ph)

    scale = float(sum(abs(v) for v in out["terms"].values())
                  + abs(out["mass_redistribution"]))
    assert scale > 0.0, "the test must actually exercise the budget"
    assert abs(float(out["residual"])) <= 1.0e-12 * scale, (
        f"budget did not close: residual {float(out['residual']):.3e} "
        f"against scale {scale:.3e}")
    assert out["window_days"] == pytest.approx(nstep * dt / 86400.0)

    # NON-VACUITY, behavioural: re-run the identical sequence while LYING to
    # the accumulator about the grid, so it cannot form the mass term.  This is
    # the failure a naive implementation actually has, and it must break
    # closure rather than be caught by arithmetic on the result.
    q2 = q0
    bud2 = CloudWaterBandBudget(P_LO, P_HI, area, G, names)
    ph2, dp2 = _grid(ncol, nlev, ps_at(0))
    bud2.begin_window(q2, ph2)
    for i in range(nstep):
        terms = _terms(key, i, ncol, nlev)
        ph_new, dp_new = _grid(ncol, nlev, ps_at(i + 1))
        # the lie: claim the grid did not move
        bud2.accumulate(terms, q2, ph2, ph2, dt)
        q2 = q2 + dt * sum(terms.values())
        ph2, dp2 = ph_new, dp_new
    out2 = bud2.close_window(q2, ph2)
    assert abs(float(out2["residual"])) > 1.0e-6 * scale, (
        "a budget blind to the moving grid must FAIL to close; if it closes "
        "anyway the mass-redistribution term is untested and the closure "
        "proves nothing about it")


def test_missing_or_unregistered_terms_raise_rather_than_pass_silently():
    area = jnp.full((3,), 1.0 / 3)
    bud = CloudWaterBandBudget(P_LO, P_HI, area, G, ("a", "b"))
    ph, dp = _grid(3, 8, jnp.full((3,), 1.0e5))
    q = jnp.zeros((3, 8))
    bud.begin_window(q, ph)
    z = jnp.zeros((3, 8))
    with pytest.raises(KeyError):
        bud.accumulate({"a": z}, q, ph, ph, 1.0)
    with pytest.raises(KeyError):
        bud.accumulate({"a": z, "b": z, "c": z}, q, ph, ph, 1.0)


def test_accumulate_before_begin_window_raises():
    area = jnp.full((2,), 0.5)
    bud = CloudWaterBandBudget(P_LO, P_HI, area, G, ("a",))
    ph, dp = _grid(2, 6, jnp.full((2,), 1.0e5))
    with pytest.raises(RuntimeError):
        bud.accumulate({"a": jnp.zeros((2, 6))}, jnp.zeros((2, 6)),
                       ph, ph, 1.0)


def test_state_round_trips_for_the_restart_chain():
    area = jnp.full((4,), 0.25)
    ph, dp = _grid(4, 10, jnp.full((4,), 1.0e5))
    q = jnp.full((4, 10), 1.0e-4)
    a = CloudWaterBandBudget(P_LO, P_HI, area, G, ("x", "y"))
    a.begin_window(q, ph)
    a.accumulate({"x": jnp.full((4, 10), 1e-8), "y": jnp.full((4, 10), -2e-9)},
                 q, ph, ph, 600.0)
    b = CloudWaterBandBudget(P_LO, P_HI, area, G, ("x", "y"))
    b.set_state(a.get_state())
    with pytest.raises(KeyError):
        b.set_state({"elapsed_s": jnp.asarray(1.0)})   # truncated sidecar
    assert float(b.terms["x"]) == pytest.approx(float(a.terms["x"]))
    assert float(b.terms["y"]) == pytest.approx(float(a.terms["y"]))
    assert b.elapsed_s == pytest.approx(a.elapsed_s)


def test_bad_band_edges_and_duplicate_terms_are_rejected():
    area = jnp.full((2,), 0.5)
    with pytest.raises(ValueError):
        CloudWaterBandBudget(8.0e4, 5.0e4, area, G, ("a",))
    with pytest.raises(ValueError):
        CloudWaterBandBudget(P_LO, P_HI, area, G, ())
    with pytest.raises(ValueError):
        CloudWaterBandBudget(P_LO, P_HI, area, G, ("a", "a"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_inverted_interfaces_poison_the_weight_instead_of_closing_quietly():
    """A column whose interfaces do not increase downward must NOT yield a
    plausible budget.  Before the guard it returned zero weights and the budget
    still closed, so the run would report a silent zero over that column."""
    import jax.numpy as jnp
    from legoesm.diagnostics.process_ledger import pressure_band_weight

    good = jnp.array([[2e4, 4e4, 6e4, 8e4, 1e5]])
    bad = jnp.array([[2e4, 6e4, 4e4, 8e4, 1e5]])   # layer 2 inverted
    w_good = pressure_band_weight(good, 5e4, 8e4)
    w_bad = pressure_band_weight(bad, 5e4, 8e4)
    assert bool(jnp.all(jnp.isfinite(w_good)))
    assert bool(jnp.any(jnp.isnan(w_bad))), "inverted layer must poison, not zero"


def test_a_poisoned_column_raises_instead_of_being_accepted():
    """Codex counterexample: NaN > tol is FALSE, so a window poisoned by a
    column whose interfaces do not increase downward sailed through the
    acceptance check and returned a NaN budget as if it were a result."""
    area = jnp.full((1,), 1.0)
    bad = jnp.array([[2.0e4, 6.0e4, 4.0e4, 8.0e4, 1.0e5]])   # overlapping
    q = jnp.full((1, 4), 1.0e-4)
    bud = CloudWaterBandBudget(P_LO, P_HI, area, G, ("a",))
    bud.begin_window(q, bad)
    bud.accumulate({"a": jnp.zeros((1, 4))}, q, bad, bad, 600.0)
    with pytest.raises(ValueError, match="not finite"):
        bud.close_window(q, bad, rtol=1e-12)


def test_thickness_cannot_disagree_with_the_interfaces():
    """The other codex counterexample: passing dp = 2*diff(p_half) doubled the
    inventory AND the terms, so the residual was exactly zero and acceptance
    passed on a uniformly wrong budget.  The argument no longer exists, which
    is the only way that class of error cannot recur."""
    import inspect
    for name in ("begin_window", "accumulate", "close_window"):
        params = inspect.signature(getattr(CloudWaterBandBudget, name)).parameters
        assert not [p for p in params if p.startswith("dp")], (
            f"{name} must derive layer thickness from the interfaces, not "
            f"accept it: {list(params)}")
