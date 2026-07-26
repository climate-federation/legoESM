"""Column-conserving tracer positivity clamp.

Motivation is a MEASUREMENT, not a theory: on the MPAS AMIP century the
floors stage clamped every water tracer with a plain ``maximum(q, 0.0)``
while the tracer transport carries no limiter.  Horizontal advection alone
left enough undershoot that the clamp invented +0.0822 kg/m2/day
(+30 kg/m2/yr) of water — 96% of it from ``q_i``/``q_c`` — which compounded
into column water 23->42 kg/m2, OLR 199->109 W/m2 and +10 K/yr of warming.

The invariant under test: clipping must not create mass.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.conservation import conservative_positive_clip


def _col(n=8, seed=0, base=1.0, under=0.3):
    """A REALISTIC tracer column: positive nearly everywhere with small
    transport undershoots, which is what water tracers actually look like.

    Purely random (mean-zero) data would make whole columns net-negative,
    where the fixer deliberately zeroes rather than conserves — see
    ``TestNetNegativeColumn``.  Testing conservation on that regime would be
    testing a case the fixer does not claim to handle.
    """
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.0, base, size=(5, n))
    idx = rng.integers(0, n, size=5)
    q[np.arange(5), idx] = -under * rng.uniform(0.2, 1.0, size=5)
    return jnp.asarray(q), jnp.asarray(rng.uniform(0.5, 1.5, size=n))


class TestConservation:
    def test_weighted_integral_is_unchanged(self):
        q, w = _col()
        out, _ = conservative_positive_clip(q, w, axis=-1)
        np.testing.assert_allclose(
            np.asarray(jnp.sum(out * w, axis=-1)),
            np.asarray(jnp.sum(q * w, axis=-1)), rtol=1e-12, atol=1e-14)

    def test_naive_clip_would_have_created_mass(self):
        """Non-vacuity: the defect this exists to prevent must be real for
        this input, or the test above proves nothing."""
        q, w = _col()
        naive = jnp.sum(jnp.maximum(q, 0.0) * w, axis=-1)
        exact = jnp.sum(q * w, axis=-1)
        assert float(jnp.max(naive - exact)) > 1e-3, (
            "test input has no undershoot — it cannot detect the defect")

    def test_conserves_across_undershoot_magnitudes(self):
        for under in (1e-6, 1e-3, 0.1, 0.5):
            q, w = _col(seed=7, under=under)
            out, _ = conservative_positive_clip(q, w, axis=-1)
            np.testing.assert_allclose(
                np.asarray(jnp.sum(out * w, axis=-1)),
                np.asarray(jnp.sum(q * w, axis=-1)), rtol=1e-12, atol=1e-14)

    def test_created_diagnostic_matches_the_naive_excess(self):
        q, w = _col()
        _, created = conservative_positive_clip(q, w, axis=-1)
        excess = jnp.sum((jnp.maximum(q, 0.0) - q) * w)
        np.testing.assert_allclose(float(created), float(excess), rtol=1e-12)

    def test_monotone_input_is_a_no_op(self):
        """A field with no negatives must come through untouched (a monotone
        transport scheme must not be perturbed by the fixer)."""
        q = jnp.asarray(np.random.default_rng(1).uniform(0.1, 2.0, (4, 6)))
        w = jnp.ones(6)
        out, created = conservative_positive_clip(q, w, axis=-1)
        np.testing.assert_allclose(np.asarray(out), np.asarray(q), rtol=1e-12)
        assert float(created) == 0.0


class TestOutputSanity:
    def test_output_is_non_negative(self):
        q, w = _col()
        out, _ = conservative_positive_clip(q, w, axis=-1)
        assert float(jnp.min(out)) >= 0.0

    def test_all_negative_column_goes_to_zero_not_nan(self):
        """Nothing to borrow from: the column must zero out cleanly."""
        q = jnp.full((2, 4), -1e-9)
        out, _ = conservative_positive_clip(q, jnp.ones(4), axis=-1)
        assert np.isfinite(np.asarray(out)).all()
        np.testing.assert_allclose(np.asarray(out), 0.0)


class TestNetNegativeColumn:
    """DOCUMENTED LIMITATION, asserted so it cannot change silently.

    When a column's weighted integral is itself NEGATIVE there is nothing to
    borrow from, and the fixer zeroes the column — which does raise the
    integral (0 > negative).  Conservation is therefore claimed only for
    columns with a positive integral, which is every physical water column;
    a net-negative water column is already unphysical.  Making this loud
    matters because the whole point of the fixer is not creating mass.
    """

    def test_net_negative_column_is_zeroed_not_conserved(self):
        q = jnp.asarray([[-1.0, 0.2, -0.5, 0.1]])
        w = jnp.ones(4)
        assert float(jnp.sum(q * w)) < 0.0
        out, _ = conservative_positive_clip(q, w, axis=-1)
        np.testing.assert_allclose(np.asarray(out), 0.0)

    def test_positive_columns_unaffected_by_a_negative_neighbour(self):
        q = jnp.asarray([[-1.0, 0.2, -0.5, 0.1],
                         [1.0, -0.1, 0.5, 0.3]])
        w = jnp.ones(4)
        out, _ = conservative_positive_clip(q, w, axis=-1)
        np.testing.assert_allclose(float(jnp.sum(out[1] * w)),
                                   float(jnp.sum(q[1] * w)), rtol=1e-12)

    def test_all_zero_column_is_stable(self):
        out, created = conservative_positive_clip(
            jnp.zeros((2, 4)), jnp.ones(4), axis=-1)
        assert np.isfinite(np.asarray(out)).all()
        assert float(created) == 0.0

    def test_columns_are_independent(self):
        """Borrowing is LOCAL: perturbing one column must not move another —
        this is what makes the fixer identical serial / sharded / under MPI."""
        q, w = _col()
        out_a, _ = conservative_positive_clip(q, w, axis=-1)
        q2 = q.at[0].set(q[0] * 3.0)
        out_b, _ = conservative_positive_clip(q2, w, axis=-1)
        np.testing.assert_allclose(np.asarray(out_a[1:]),
                                   np.asarray(out_b[1:]), rtol=1e-12)


class TestJaxContracts:
    def test_jit_and_grad_are_finite(self):
        q, w = _col()

        def loss(x):
            return jnp.sum(conservative_positive_clip(x, w, axis=-1)[0] ** 2)

        g = jax.jit(jax.grad(loss))(q)
        assert np.isfinite(np.asarray(g)).all(), "NaN in the clamp gradient"

    def test_grad_finite_on_an_all_negative_column(self):
        """The 0/0 guard: an empty column must not poison the backward pass."""
        q = jnp.full((2, 4), -1e-9)

        def loss(x):
            return jnp.sum(
                conservative_positive_clip(x, jnp.ones(4), axis=-1)[0] ** 2)

        assert np.isfinite(np.asarray(jax.grad(loss)(q))).all()

    def test_dtype_preserved(self):
        q = jnp.ones((2, 3), dtype=jnp.float32) - 2.0
        out, _ = conservative_positive_clip(q, jnp.ones(3, jnp.float64))
        assert out.dtype == q.dtype


def test_repeated_application_does_not_drift():
    """The century failure was COMPOUNDING: 1152 steps/day.  Applying the
    fixer many times must not accumulate mass the way the naive clamp did."""
    q, w = _col(seed=3)
    total0 = float(jnp.sum(q * w))
    x = q
    for _ in range(200):
        x, _ = conservative_positive_clip(x, w, axis=-1)
    assert abs(float(jnp.sum(x * w)) - total0) < 1e-10 * max(abs(total0), 1.0)


def test_mpas_config_flag_defaults_off_and_is_wired():
    """Default must be bit-identical to today; the flag must reach the step."""
    import inspect

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, MPASPrimitiveEquationModel,
    )
    assert MPASPrimitiveEquationConfig().conservative_tracer_clamp is False
    # The floors stage lives in ``_step_jit`` (``step`` delegates to it) — the
    # method actually executed by the MPAS lane.  Asserting against ``step``
    # silently passed nothing, the same wrong-target mistake that once had a
    # clamp blamed in ``_run_per_step`` while the century ran ``_run_mpas``.
    src = inspect.getsource(MPASPrimitiveEquationModel._step_jit)
    assert "conservative_tracer_clamp" in src
    assert "conservative_positive_clip" in src
    assert "jnp.maximum(f.data, 0.0)" in src, (
        "the default (flag off) path must still be the plain clamp so "
        "existing MPAS results stay bit-identical")


def test_cli_round_trip_and_factory_wiring():
    """#691 rule: a new user-tunable field needs a CLI flag AND must reach the
    dycore.  A flag that stops at DycoreConfig is the silently-inert-wiring
    defect this campaign hit four times."""
    from scripts.run.run_amip import (
        _postprocess_args, build_arg_parser, build_config_from_args,
    )
    parser = build_arg_parser()
    off = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert off.dycore.mpas_conservative_tracer_clamp is False

    on = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--mpas-conservative-tracer-clamp"]),
        parser))
    assert on.dycore.mpas_conservative_tracer_clamp is True
    on.validate_strict()


def test_factory_forwards_the_flag_to_the_dycore_config():
    """The DycoreConfig value must land on MPASPrimitiveEquationConfig."""
    import inspect

    from legoesm.driver import component_factory
    src = inspect.getsource(component_factory)
    assert "conservative_tracer_clamp=dc.mpas_conservative_tracer_clamp" in src


class TestWaterMassOnly:
    """Number concentrations must NOT get the conserving borrow.

    Rescaling N_c/N_i/N_r to preserve a column integral is unphysical — number
    is not conserved under transport — and it perturbs the microphysics
    directly (M2005 ice deposition goes as N_i^(2/3)).  The LES reference
    makes the same split; the first MPAS wiring did not, and applied the
    borrow to every tracer in the dict.
    """

    def test_water_mass_tracers_selected(self):
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            _is_water_mass_tracer,
        )
        for k in ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g"):
            assert _is_water_mass_tracer(k), k
            assert _is_water_mass_tracer("trc_" + k), k

    def test_number_and_other_tracers_excluded(self):
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            _is_water_mass_tracer,
        )
        for k in ("N_c", "N_i", "N_r", "trc_N_i", "aerosol_number", "ozone"):
            assert not _is_water_mass_tracer(k), k

    def test_step_applies_the_split(self):
        import inspect

        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel,
        )
        src = inspect.getsource(MPASPrimitiveEquationModel._step_jit)
        assert "_is_water_mass_tracer" in src, (
            "the conserving borrow must be restricted to water mass tracers")


class TestCodexFindings:
    """Regression tests for the codex 2026-07-26 FIX-FIRST findings."""

    def test_tiny_positive_column_is_preserved_not_zeroed(self):
        """Finding 1: a finite positive column below eps must keep its values
        (plain clip), never be zeroed."""
        q = jnp.asarray([[5e-31, 3e-31]])
        out, _ = conservative_positive_clip(q, jnp.ones(2))
        np.testing.assert_allclose(np.asarray(out), np.asarray(q))

    def test_monotone_tiny_column_is_an_exact_no_op(self):
        q = jnp.asarray([[1e-25, 2e-25, 5e-26]])
        out, created = conservative_positive_clip(q, jnp.ones(3))
        np.testing.assert_array_equal(np.asarray(out), np.asarray(q))
        assert float(created) == 0.0

    def test_non_trailing_axis_raises(self):
        """Finding 1: weight broadcasts on the trailing dim; a non-trailing
        axis would silently mis-conserve, so it must refuse."""
        q = jnp.ones((2, 3))
        with pytest.raises(ValueError, match="TRAILING"):
            conservative_positive_clip(q, jnp.ones(2), axis=0)

    def test_float32_gradient_finite_near_threshold(self):
        """Finding 3: with eps=1e-30 raw, a float32 column just above
        threshold has after**2 ~ 1e-60 -> underflow -> Inf in the quotient
        VJP.  The dtype-aware eps_eff = sqrt(tiny) must keep it finite."""
        q32 = jnp.asarray([[2e-30, -5e-31]], dtype=jnp.float32)
        w32 = jnp.ones(2, dtype=jnp.float32)

        def loss(x):
            return jnp.sum(conservative_positive_clip(x, w32)[0] ** 2)

        g = jax.grad(loss)(q32)
        assert np.isfinite(np.asarray(g)).all(), (
            "float32 VJP emitted non-finite gradients near the eps threshold")

    def test_float32_conserves_above_its_effective_threshold(self):
        q32 = jnp.asarray([[1e-3, -2e-4, 5e-4]], dtype=jnp.float32)
        w32 = jnp.ones(3, dtype=jnp.float32)
        out, _ = conservative_positive_clip(q32, w32)
        np.testing.assert_allclose(float(jnp.sum(out * w32)),
                                   float(jnp.sum(q32 * w32)), rtol=2e-6)

    def test_mpi_floors_thread_the_flag(self):
        """Finding 6: distributed MPAS runs ``make_voronoi_mpi_step``'s OWN
        floors block, not ``_step_jit``.  It previously hard-coded the plain
        clamp, silently ignoring ``conservative_tracer_clamp`` — the repo's
        recurring dropped-flag defect.  Both branches must exist there."""
        import inspect

        from legoesm.parallel import voronoi_mpi
        src = inspect.getsource(voronoi_mpi.make_voronoi_mpi_step)
        assert "conservative_tracer_clamp" in src
        assert "conservative_positive_clip" in src
        assert "_is_water_mass_tracer" in src
