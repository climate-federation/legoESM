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


class TestAllTracersBorrowed:
    """EVERY tracer — numbers included — gets the conserving borrow.

    REVERSAL of the 2026-07-26 number exclusion, by measurement: the naive
    clip on N_* INVENTED number at every advection undershoot (28k cells and
    6.8e-4 of the field per step on century3 day 40 = x2.2/day compound
    growth; N_i reached 1e193 and overflowed into NaN at day 803).  The old
    reasoning conflated PROCESS-level number non-conservation (microphysics
    may create/destroy number) with TRANSPORT-level conservation: advection
    conserves every mass-weighted mixing ratio, number included.
    """

    def test_step_borrows_per_mass_tracers(self):
        """Source pin on the REAL hot-loop method: the borrow must run
        through the explicit per-mass eligibility rule (codex 2026-07-28),
        never the old water-only gate that invented N_i x2.2/day."""
        import inspect

        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel,
        )
        src = inspect.getsource(MPASPrimitiveEquationModel._step_jit)
        assert "conservative_positive_clip" in src
        assert "is_borrow_eligible_tracer" in src
        assert "_is_water_mass_tracer" not in src, (
            "the number exclusion reappeared — it invents number x2.2/day")

    def test_eligibility_rule(self):
        """Per-mass fields (mixing ratios + N_i/N_s/N_g) are borrowed;
        per-volume N_c/N_r are not (dsigma weight has no conservation
        meaning for #/m^3 — codex 2026-07-28)."""
        from legoesm.core.conservation import is_borrow_eligible_tracer
        for k in ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g",
                  "N_i", "N_s", "N_g", "trc_N_i"):
            assert is_borrow_eligible_tracer(k), k
        for k in ("N_c", "N_r", "trc_N_r", "aerosol_number", "ozone"):
            assert not is_borrow_eligible_tracer(k), k

    def test_net_negative_number_column_with_positive_ice_mass(self):
        """Codex 2026-07-28: the net-negative fallback zeroes N_i while
        q_i>0 remains — Morrison's LAMI clamp handles N_i=0 without NaN,
        but the PSD coupling is degraded there.  Pin the behaviour so the
        fallback stays LOUD in review rather than drifting silently."""
        n = jnp.asarray([[-2.0e4, 0.5e4, -1.0e4, 0.4e4]])  # net negative
        w = jnp.ones(4)
        assert float(jnp.sum(n * w)) < 0.0
        out, _ = conservative_positive_clip(n, w)
        np.testing.assert_allclose(np.asarray(out), 0.0)  # zeroed, finite

    def test_mpi_lane_borrows_every_tracer(self):
        import inspect

        from legoesm.parallel import voronoi_mpi
        src = inspect.getsource(voronoi_mpi.make_voronoi_mpi_step)
        assert "is_borrow_eligible_tracer" in src
        assert "_is_water_mass_tracer" not in src, (
            "the MPI floors reintroduced the number exclusion")

    def test_naive_clip_invents_number_borrow_does_not(self):
        """The mechanism regression: on a field with advection-style
        undershoot, the naive clip INCREASES the weighted integral (invention)
        while the borrow preserves it exactly."""
        w = jnp.asarray([1.0, 1.0, 1.0, 1.0])
        n = jnp.asarray([[4.0e4, -1.0e4, 3.0e4, 2.0e4]])  # one undershoot
        total_before = float((n * w).sum())
        naive = jnp.maximum(n, 0.0)
        assert float((naive * w).sum()) > total_before  # invention
        borrowed, _ = conservative_positive_clip(n, w)
        np.testing.assert_allclose(
            float((borrowed * w).sum()), total_before, rtol=1e-12)
        assert float(borrowed.min()) >= 0.0


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
        # 2026-07-28 reversal: the borrow now covers EVERY tracer (the
        # number exclusion invented number x2.2/day — see
        # TestAllTracersBorrowed).
        assert "_is_water_mass_tracer" not in src


class TestRound2Findings:
    """Codex round-2 (review-2.md): the two remaining blockers."""

    def test_float16_is_refused(self):
        """sqrt(tiny_f16) = 7.8e-3 is a LARGE mixing ratio: the degenerate
        branch could invent ~10 kg/m2 per column.  Must refuse, not corrupt."""
        q = jnp.asarray([[5e-3, -1e-3]], dtype=jnp.float16)
        with pytest.raises(ValueError, match="too coarse"):
            conservative_positive_clip(q, jnp.ones(2, jnp.float16))

    def test_bfloat16_is_accepted(self):
        """bf16 shares float32's exponent range — sqrt(tiny) ~ 1e-19, fine."""
        q = jnp.asarray([[5e-3, -1e-3]], dtype=jnp.bfloat16)
        out, _ = conservative_positive_clip(q, jnp.ones(2, jnp.bfloat16))
        assert out.dtype == jnp.bfloat16

    def test_mass_fixer_runs_before_the_floors_serial(self):
        """Finding 5: the p_s fix must precede the tracer clamp so end-of-step
        column water sum(q*p_s*dsigma)/g is conserved exactly.  Order in the
        SOURCE of the method that runs (_step_jit)."""
        import inspect

        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel,
        )
        src = inspect.getsource(MPASPrimitiveEquationModel._step_jit)
        assert (src.index("_fix_mass_mpas_hydro")
                < src.index("conservative_tracer_clamp")), (
            "p_s mass fix must run BEFORE the tracer floors")

    def test_mass_fixer_runs_before_the_floors_mpi(self):
        import inspect

        from legoesm.parallel import voronoi_mpi
        src = inspect.getsource(voronoi_mpi.make_voronoi_mpi_step)
        assert src.index("_fix_mass_mpi(") < src.index(
            "conservative_tracer_clamp"), (
            "MPI p_s mass fix must run BEFORE the tracer floors")

    def test_mpi_factory_rejects_mismatched_sigma_coord(self):
        """Finding 6 nit: a direct caller passing a foreign vertical
        coordinate must fail at build, not silently mis-conserve."""
        import inspect

        from legoesm.parallel import voronoi_mpi
        src = inspect.getsource(voronoi_mpi.make_voronoi_mpi_step)
        assert "differs from" in src and "model.sigma_coord.dsigma" in src


def test_hybrid_dp_weight_conserves_the_physical_integral():
    """Codex 2026-07-28 round-2 blocker: with per-column dp weights (the
    hybrid-correct layer mass) the DP-integral is conserved; the old flat
    dsigma weight would conserve the WRONG integral there (non-vacuity:
    the two integrals genuinely differ on this input)."""
    rng = np.random.default_rng(11)
    ncol, nlev = 4, 8
    q = jnp.asarray(rng.uniform(0.0, 1.0, (ncol, nlev)))
    q = q.at[:, 3].set(-0.2)
    # hybrid-like dp: per-column variation (A-part flat + B-part scaled)
    p_s = jnp.asarray(rng.uniform(6.0e4, 1.03e5, (ncol, 1)))
    a_part = jnp.linspace(2000.0, 500.0, nlev)[None, :]
    b_part = jnp.linspace(0.002, 0.2, nlev)[None, :] * p_s
    dp = a_part + b_part
    dsig = jnp.asarray(np.full(nlev, 1.0 / nlev))
    out, _ = conservative_positive_clip(q, dp, axis=-1)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(out * dp, axis=-1)),
        np.asarray(jnp.sum(q * dp, axis=-1)), rtol=1e-12)
    # non-vacuity: the dsigma integral of the SAME output is NOT conserved
    ds_in = np.asarray(jnp.sum(q * dsig, axis=-1))
    ds_out = np.asarray(jnp.sum(out * dsig, axis=-1))
    assert np.max(np.abs(ds_out - ds_in)) > 1e-6


def test_flag_on_step_executes_with_number_tracers():
    """Lane-level regression (codex 2026-07-28 round 3): the flag-ON floors
    must actually EXECUTE — round 3 caught ``pressure_at_half(Field)``
    raising TypeError at the call site while every helper-level test stayed
    green.  One real step with mixed per-mass/per-volume tracers, clamp on."""
    import jax

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, MPASPrimitiveEquationModel,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas,
    )
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate

    mesh = create_voronoi_mesh(2)
    sigma = create_sigma_coordinate(6)
    cfg = MPASPrimitiveEquationConfig(conservative_tracer_clamp=True)
    model = MPASPrimitiveEquationModel(mesh, sigma, cfg)
    state = held_suarez_init_mpas(mesh, sigma)
    ncell, nlev = state.T.data.shape
    rng = np.random.default_rng(5)
    tr = {}
    for k in ("q_v", "q_i", "N_i", "N_r"):
        data = jnp.asarray(rng.uniform(0.0, 1e-3, (ncell, nlev)))
        data = data.at[:, 2].add(-2e-4)  # guarantee undershoot for the clamp
        tr[k] = state.p_s.replace(data=data)
    state = state._replace(tracers=tr)
    out = model.step(state, 75.0)
    for k in ("q_v", "q_i", "N_i", "N_r"):
        arr = np.asarray(out.tracers[k].data)
        assert np.isfinite(arr).all(), k
        assert arr.min() >= 0.0, k


class TestGlobalResidualRedistribution:
    """century4 residual engine (2026-07-28): net-negative COLUMNS are common
    on spiky number fields (868/10242 per step measured) and column-local
    zeroing alone re-created x2.74/day growth.  The _global variant must
    conserve the TOTAL exactly whenever the global integral is non-negative."""

    def _spiky(self, seed=0):
        rng = np.random.default_rng(seed)
        q = rng.uniform(0.0, 1.0, (40, 8))
        q[rng.integers(0, 40, 6), :] = -0.05      # whole columns net-negative
        q[rng.integers(0, 40, 3), 2] = 1e4        # spikes (number-field shape)
        return jnp.asarray(q), jnp.asarray(rng.uniform(0.5, 1.5, 8))

    def test_total_conserved_with_net_negative_columns(self):
        from legoesm.core.conservation import (
            conservative_positive_clip, conservative_positive_clip_global,
        )
        q, w = self._spiky()
        col_only, _ = conservative_positive_clip(q, w)
        out, _ = conservative_positive_clip_global(q, w)
        t_in = float(jnp.sum(q * w))
        # non-vacuity: the column-only fixer INVENTS here
        assert float(jnp.sum(col_only * w)) > t_in * (1 + 1e-12)
        np.testing.assert_allclose(float(jnp.sum(out * w)), t_in, rtol=1e-12)
        assert float(out.min()) >= 0.0

    def test_identical_to_column_variant_when_no_negative_columns(self):
        from legoesm.core.conservation import (
            conservative_positive_clip, conservative_positive_clip_global,
        )
        q, w = _col(seed=2)
        a, _ = conservative_positive_clip(q, w)
        b, _ = conservative_positive_clip_global(q, w)
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12)

    def test_iterated_total_is_stable(self):
        from legoesm.core.conservation import (
            conservative_positive_clip_global,
        )
        q, w = self._spiky(seed=4)
        t0 = float(jnp.sum(q * w))
        x = q
        for _ in range(200):
            x, _ = conservative_positive_clip_global(x, w)
        assert abs(float(jnp.sum(x * w)) - t0) < 1e-9 * max(abs(t0), 1.0)

    def test_grad_finite_with_negative_columns(self):
        from legoesm.core.conservation import (
            conservative_positive_clip_global,
        )
        q, w = self._spiky(seed=6)

        def loss(x):
            return jnp.sum(conservative_positive_clip_global(x, w)[0] ** 2)

        assert np.isfinite(np.asarray(jax.grad(loss)(q))).all()

    def test_floors_use_the_global_variant(self):
        import inspect

        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel,
        )
        from legoesm.parallel import voronoi_mpi
        for src in (
            inspect.getsource(MPASPrimitiveEquationModel._step_jit),
            inspect.getsource(voronoi_mpi.make_voronoi_mpi_step),
        ):
            assert "conservative_positive_clip_global" in src


def test_global_variant_keeps_tiny_positive_field():
    """Codex 2026-07-28: a degenerate-but-positive global total must KEEP the
    column result (float32 q=[[5e-20]] was zeroed by the first cut)."""
    from legoesm.core.conservation import conservative_positive_clip_global
    q = jnp.asarray([[5e-20]], dtype=jnp.float32)
    out, _ = conservative_positive_clip_global(q, jnp.ones(1))
    np.testing.assert_allclose(np.asarray(out), np.asarray(q))


def test_mpi_floor_iterates_tracers_sorted():
    """Collectives inside the tracer loop must pair identically on every
    rank: the iteration must be over sorted keys, never dict order."""
    import inspect

    from legoesm.parallel import voronoi_mpi
    src = inspect.getsource(voronoi_mpi.make_voronoi_mpi_step)
    assert "sorted(state_new.tracers)" in src
