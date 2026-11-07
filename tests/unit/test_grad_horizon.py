"""Tests for the chaos / long-window adjoint guardrails (grad_horizon.py)."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.training.grad_horizon import (  # noqa: E402
    check_grad_horizon,
    estimate_growth_rate,
    global_grad_norm,
    grad_norm_vs_horizon,
)


def test_global_grad_norm_matches_l2():
    g = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([0.0])}
    assert jnp.allclose(global_grad_norm(g), 5.0)


def test_global_grad_norm_empty():
    assert float(global_grad_norm({})) == 0.0


def test_grad_norm_grows_on_chaotic_map():
    # logistic map at r=3.9 is chaotic => |d x_n / d x_0| ~ exp(lyap * n)
    r = 3.9

    def loss_for_horizon(n, x):
        val = x
        for _ in range(n):
            val = r * val * (1.0 - val)
        return val ** 2

    x0 = jnp.asarray(0.4)
    norms = grad_norm_vs_horizon(loss_for_horizon, [1, 5, 10, 20], x0)
    assert norms[20] > norms[1]
    rate = estimate_growth_rate(norms)
    assert rate > 0.0  # positive Lyapunov exponent => exponential adjoint growth


def test_estimate_growth_rate_needs_two_points():
    import math
    assert math.isnan(estimate_growth_rate({5: 1.0}))
    assert math.isnan(estimate_growth_rate({1: 0.0, 2: 0.0}))  # no positive samples


def test_estimate_growth_rate_pure_exponential():
    # grad_norm = exp(0.3 n)  => recovered slope ~ 0.3
    norms = {n: float(jnp.exp(0.3 * n)) for n in (1, 2, 4, 8, 16)}
    assert abs(estimate_growth_rate(norms) - 0.3) < 1e-6


def test_check_grad_horizon_warn_and_raise():
    assert check_grad_horizon(1.0, 10, 5.0) is True
    assert check_grad_horizon(10.0, 10, 5.0) is False
    assert check_grad_horizon(jnp.inf, 10, 5.0) is False
    with pytest.raises(RuntimeError, match="predictability horizon"):
        check_grad_horizon(10.0, 10, 5.0, raise_on_exceed=True)


# ---------------------------------------------------------------------------
# The sweep must not hide its own finding (2026-09-16).
#
# The earlier ``estimate_growth_rate`` filtered non-finite samples out of the
# log-linear fit.  A horizon sweep exists to find the horizon at which the
# adjoint stops being a number, so dropping exactly those samples returned a
# reassuring slope measured on the healthy prefix with nothing in the return
# value to say a blow-up had been seen at all.
# ---------------------------------------------------------------------------

def test_blowup_horizon_flags_only_non_finite_samples():
    from legoesm.training.grad_horizon import blowup_horizon, unfittable_horizons
    assert blowup_horizon({1: 1.0, 2: 2.0, 4: 4.0}) is None
    assert blowup_horizon({1: 1.0, 2: float("inf"), 4: float("nan")}) == 2
    # A zero adjoint is a perfectly finite answer that merely has no logarithm.
    # Calling it a blow-up would invent a failure AND discard every longer
    # horizon behind it, which is how a severed gradient path could masquerade
    # as an overflow.
    assert blowup_horizon({1: 1.0, 2: 0.0, 4: 4.0}) is None
    assert unfittable_horizons({1: 1.0, 2: 0.0, 4: 4.0}) == [2]


def test_growth_rate_fits_only_below_the_blowup():
    """NON-VACUITY: a post-blow-up survivor must not enter the fit.

    Horizons 1-4 sit on a clean ``exp(0.5 n)`` branch.  Horizon 8 is ``inf``.
    Horizon 16 is finite but tiny — the kind of survivor that appears when a
    later leaf underflows after an earlier one overflowed.  The old
    filter-and-fit-everything rule would have fitted 1, 2, 4 AND 16 together and
    returned a strongly NEGATIVE slope, i.e. "the adjoint is shrinking", from
    data containing an overflow.  The fit must use 1, 2, 4 only.
    """
    import math
    from legoesm.training.grad_horizon import estimate_growth_rate
    clean = {n: math.exp(0.5 * n) for n in (1, 2, 4)}
    with_blowup = {**clean, 8: float("inf"), 16: 1e-8}
    assert estimate_growth_rate(clean) == pytest.approx(0.5, rel=1e-6)
    assert estimate_growth_rate(with_blowup) == pytest.approx(0.5, rel=1e-6)
    # The corrupted fit the old rule produced is NOT 0.5, so the assertion above
    # can actually go red if the filtering rule regresses.
    all_finite = {**clean, 16: 1e-8}
    assert abs(estimate_growth_rate(all_finite) - 0.5) > 0.1


def test_growth_rate_nan_when_too_few_samples_below_blowup():
    import math
    from legoesm.training.grad_horizon import estimate_growth_rate
    assert math.isnan(estimate_growth_rate({1: 1.0, 2: float("nan")}))


def test_max_norm_does_not_overflow_where_an_l2_norm_would():
    """The measurement must survive the magnitudes it exists to measure."""
    from legoesm.training.grad_horizon import (
        global_grad_norm, grad_max_norm, leaf_grad_report,
    )
    big = {"a": jnp.full((4,), 1e30, dtype=jnp.float32)}
    report = leaf_grad_report(big)
    # Every element is finite ...
    assert all(ok for _, ok in report.values())
    # ... and squaring them is not: the L2 norm reports a blow-up that the data
    # does not contain, which is precisely the false positive this replaces.
    assert not bool(jnp.isfinite(global_grad_norm(big)))
    assert grad_max_norm(report) == pytest.approx(1e30)


def test_diagnosis_needs_a_reference_and_uses_amplitude_not_nan_kind():
    """NaN-vs-inf alone cannot name a cause; amplitude relative to a short
    horizon can, and without that reference the verdict must abstain."""
    from legoesm.training.grad_horizon import diagnose_blowup, leaf_grad_report
    clean = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}
    assert diagnose_blowup(leaf_grad_report(clean), 1.0) == "finite"

    # NaN while the rest of the adjoint is still at the reference scale:
    # amplification cannot do that, so it points at a kernel.
    kernel = {"a": jnp.array([2.0]), "b": jnp.array([jnp.nan])}
    assert diagnose_blowup(leaf_grad_report(kernel), 1.0) == "kernel"

    # The SAME NaN, but the surviving leaves have grown a trillionfold: the
    # adjoint was being amplified whatever the NaN means.  A rule that keyed off
    # "NaN and not inf" would call both of these "kernel" and be wrong here.
    growth = {"a": jnp.array([1e12]), "b": jnp.array([jnp.nan])}
    assert diagnose_blowup(leaf_grad_report(growth), 1.0) == "growth"

    # No reference -> abstain rather than guess.
    assert diagnose_blowup(leaf_grad_report(kernel)) == "unclassified"
    assert diagnose_blowup(leaf_grad_report(kernel), 0.0) == "unclassified"

    # EVERY leaf non-finite: no surviving amplitude to compare, so abstain.
    # The old rule fell through to "kernel" here because max() over an empty
    # set defaulted to 0.0 -- a total overflow labelled as a kernel NaN.
    total = {"a": jnp.array([jnp.inf]), "b": jnp.array([jnp.inf])}
    assert diagnose_blowup(leaf_grad_report(total), 1.0) == "unclassified"


def test_leaf_report_separates_nan_from_inf_and_names_the_leaf():
    from legoesm.training.grad_horizon import leaf_grad_report
    rep = leaf_grad_report({"a": jnp.array([1.0, 2.0]),
                            "b": jnp.array([jnp.inf]),
                            "c": jnp.array([jnp.nan])})
    by_name = {}
    for k, (v, ok) in rep.items():
        for name in ("a", "b", "c"):
            if name in k:
                by_name[name] = (v, ok)
    assert by_name["a"] == (pytest.approx(2.0), True)
    assert by_name["b"][1] is False and by_name["b"][0] == float("inf")
    assert by_name["c"][1] is False and by_name["c"][0] != by_name["c"][0]
