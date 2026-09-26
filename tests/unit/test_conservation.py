"""Unit tests for conservation fixers."""

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.core.conservation import (
    fix_mass_shallow_water,
    fix_energy_shallow_water,
    apply_conservation_fixer,
    compute_conservation_diagnostics,
    energy_consistent_moisture_floor,
    energy_consistent_water_floor,
    apply_water_positivity,
)
from legoesm.core.operators import global_integral
from legoesm.grids.cubed_sphere import create_cubed_sphere


class TestConservationFixers:
    """Tests for mass and energy conservation fixers."""

    @pytest.fixture
    def grid(self):
        return create_cubed_sphere(8)

    @pytest.fixture
    def state(self, grid):
        shape = (6, 8, 8)
        dims = ("face", "x", "y")
        return ShallowWaterState(
            h=Field(data=jnp.ones(shape) * 1e4, name="h", dims=dims, units="m"),
            u=Field(data=jnp.ones(shape) * 10.0, name="u", dims=dims, units="m/s"),
            v=Field(data=jnp.ones(shape) * 5.0, name="v", dims=dims, units="m/s"),
            h_s=Field(data=jnp.zeros(shape), name="h_s", dims=dims, units="m"),
        )

    def test_mass_fixer_exact(self, grid, state):
        """Mass fixer should restore exact mass."""
        # Perturb mass
        key = jax.random.PRNGKey(0)
        perturbation = jax.random.normal(key, state.h.shape) * 10.0
        state_perturbed = state._replace(
            h=state.h.replace(data=state.h.data + perturbation)
        )

        state_fixed = fix_mass_shallow_water(state_perturbed, state, grid)

        mass_original = global_integral(state.h, grid)
        mass_fixed = global_integral(state_fixed.h, grid)
        # float32 limits precision to ~1e-7 relative; mass fixer is exact
        # in infinite precision, but float32 accumulation rounds off
        assert jnp.allclose(mass_fixed, mass_original, rtol=1e-5)

    def test_mass_fixer_preserves_gradients(self, grid, state):
        """Mass fixer should be differentiable."""
        def loss(h_data):
            state_new = state._replace(h=state.h.replace(data=h_data + 1.0))
            state_fixed = fix_mass_shallow_water(state_new, state, grid)
            return jnp.sum(state_fixed.h.data ** 2)

        grads = jax.grad(loss)(state.h.data)
        assert jnp.all(jnp.isfinite(grads))

    def test_energy_fixer_exact(self, grid, state):
        """Energy fixer should restore exact total energy."""
        # Perturb velocities (simulating a time step that drifts energy)
        state_perturbed = state._replace(
            u=state.u.replace(data=state.u.data * 1.05),
            v=state.v.replace(data=state.v.data * 0.97),
        )

        state_fixed = fix_energy_shallow_water(state_perturbed, state, grid)

        def total_energy(s):
            h, u, v, h_s = s.h.data, s.u.data, s.v.data, s.h_s.data
            ke = 0.5 * h * (u**2 + v**2)
            pe = 0.5 * constants.g * (h + h_s)**2  # iter-166: was 9.80616
            return jnp.sum((ke + pe) * grid.area)

        E_old = total_energy(state)
        E_fixed = total_energy(state_fixed)
        assert jnp.allclose(E_fixed, E_old, rtol=1e-5), (
            f"Energy not restored: old={float(E_old):.6e}, fixed={float(E_fixed):.6e}"
        )

    def test_energy_fixer_large_perturbation(self, grid, state):
        """Energy fixer should handle >1% drift (no clamping)."""
        # Large velocity perturbation: ~21% KE increase
        state_perturbed = state._replace(
            u=state.u.replace(data=state.u.data * 1.10),
            v=state.v.replace(data=state.v.data * 1.10),
        )

        state_fixed = fix_energy_shallow_water(state_perturbed, state, grid)

        def total_energy(s):
            h, u, v, h_s = s.h.data, s.u.data, s.v.data, s.h_s.data
            ke = 0.5 * h * (u**2 + v**2)
            pe = 0.5 * constants.g * (h + h_s)**2  # iter-166: was 9.80616
            return jnp.sum((ke + pe) * grid.area)

        E_old = total_energy(state)
        E_fixed = total_energy(state_fixed)
        assert jnp.allclose(E_fixed, E_old, rtol=1e-5), (
            f"Energy not restored for large drift: "
            f"old={float(E_old):.6e}, fixed={float(E_fixed):.6e}"
        )

    def test_energy_fixer_no_nan_when_pe_exceeds_total(self, grid, state):
        """Energy fixer must not produce NaN when PE > E_old."""
        # After mass fixer, h could increase enough that PE_new > E_old.
        # Simulate this by inflating h while keeping velocities small.
        state_high_pe = state._replace(
            h=state.h.replace(data=state.h.data * 2.0),
            u=state.u.replace(data=state.u.data * 0.01),
            v=state.v.replace(data=state.v.data * 0.01),
        )

        state_fixed = fix_energy_shallow_water(state_high_pe, state, grid)

        assert jnp.all(jnp.isfinite(state_fixed.u.data)), "u contains NaN/Inf"
        assert jnp.all(jnp.isfinite(state_fixed.v.data)), "v contains NaN/Inf"

    def test_energy_fixer_preserves_gradients(self, grid, state):
        """Energy fixer should be differentiable."""
        def loss(u_data):
            state_new = state._replace(
                u=state.u.replace(data=u_data * 1.02),
            )
            state_fixed = fix_energy_shallow_water(state_new, state, grid)
            return jnp.sum(state_fixed.u.data ** 2)

        grads = jax.grad(loss)(state.u.data)
        assert jnp.all(jnp.isfinite(grads)), "Gradients through energy fixer are not finite"

    def test_conservation_diagnostics(self, grid, state):
        """Diagnostics should return reasonable values."""
        diag = compute_conservation_diagnostics(state, grid)
        assert 'total_mass' in diag
        assert 'total_energy' in diag
        assert float(diag['total_mass']) > 0
        assert float(diag['total_energy']) > 0


class TestMoistureCorrectionDiagnostic:
    """Iter-87: ``diagnose_moisture_correction`` reports the silent
    correction that ``fix_moisture_hydrostatic`` would apply.  Lets
    callers track moisture-mass injections that would otherwise be
    invisible against advection-clip / saturation-adjustment sources.
    """

    @pytest.fixture
    def grid(self):
        return create_cubed_sphere(8)

    def test_zero_correction_when_target_equals_current(self, grid):
        """When q_v already integrates to target, scale = 1, correction = 0."""
        from legoesm.core.conservation import (
            diagnose_moisture_correction,
            compute_global_moisture,
        )
        nlev = 4
        shape_3d = (6, 8, 8, nlev)
        shape_2d = (6, 8, 8)
        q_v = jnp.full(shape_3d, 0.005)  # uniform 5 g/kg
        p_s = jnp.full(shape_2d, 1.013e5)
        dsigma = jnp.full((nlev,), 0.25)
        # target = current (compute_global_moisture).
        current = compute_global_moisture(q_v, p_s, dsigma, grid)
        diag = diagnose_moisture_correction(q_v, current, p_s, dsigma, grid)
        assert float(diag['correction_mass']) == pytest.approx(0.0, abs=1e-6)
        assert float(diag['scale']) == pytest.approx(1.0, abs=1e-10)

    def test_correction_matches_actual_fix(self, grid):
        """diagnose_moisture_correction's reported scale must equal the
        scale that fix_moisture_hydrostatic actually applies."""
        from legoesm.core.conservation import (
            diagnose_moisture_correction,
            fix_moisture_hydrostatic,
            compute_global_moisture,
        )
        nlev = 4
        shape_3d = (6, 8, 8, nlev)
        shape_2d = (6, 8, 8)
        q_v = jnp.full(shape_3d, 0.005)
        p_s = jnp.full(shape_2d, 1.013e5)
        dsigma = jnp.full((nlev,), 0.25)
        # Set target 10 % below current (typical drift correction)
        current = compute_global_moisture(q_v, p_s, dsigma, grid)
        target = current * 0.9

        diag = diagnose_moisture_correction(q_v, target, p_s, dsigma, grid)
        q_v_fixed = fix_moisture_hydrostatic(q_v, target, p_s, dsigma, grid)

        # The reported scale should equal the actual scale applied
        actual_scale = float(jnp.mean(q_v_fixed / q_v))
        assert float(diag['scale']) == pytest.approx(actual_scale, rel=1e-5)
        # The fixed q_v should integrate to target exactly
        new_mass = compute_global_moisture(q_v_fixed, p_s, dsigma, grid)
        assert float(new_mass) == pytest.approx(float(target), rel=1e-6)
        # correction_mass should equal target − current
        assert float(diag['correction_mass']) == pytest.approx(
            float(target - current), rel=1e-6,
        )


class TestEnergyConsistentMoistureFloor:
    """Issue #323: the opt-in ``energy_consistent_moisture_floor`` must floor
    ``q_v`` at zero WITHOUT injecting spurious latent heat — the failure mode
    behind the kessler+sbm wind blow-up.

    Setup mirrors one physics tracer step under organised convection: a
    strong vapour sink ``dq_v_dt`` plus the condensation latent heating that
    *exactly* matches it (``dT_dt = -(L_v/c_pd) dq_v_dt``), so the raw
    (pre-floor) update conserves moist static energy ``c_pd*T + L_v*q_v`` by
    construction.  Any post-floor change in MSE is therefore the spurious
    injection introduced by the floor itself.
    """

    L_v = constants.L_v
    c_pd = constants.c_pd
    dt = 150.0

    def _firing_case(self):
        """A column where the sink drives q_v < 0 (the clip fires)."""
        q_v = jnp.array([0.012, 0.003, 8.0e-4, 5.0e-3])
        dq_v_dt = jnp.array([-2.0e-4, -5.0e-5, -3.0e-5, -1.0e-5])  # strong drying
        dT_dt = -(self.L_v / self.c_pd) * dq_v_dt                  # condensation warming
        T = jnp.array([295.0, 290.0, 288.0, 292.0])
        q_v_raw = q_v + self.dt * dq_v_dt
        T_upd = T + self.dt * dT_dt
        mse_pre = self.c_pd * T + self.L_v * q_v
        return q_v_raw, T_upd, mse_pre

    def test_clip_actually_fires(self):
        q_v_raw, _, _ = self._firing_case()
        assert bool((q_v_raw < 0).any()), "test case must drive q_v negative"

    def test_naive_floor_injects_energy(self):
        """Baseline (flag OFF): plain max(q_v, 0) gains energy = L_v * deficit."""
        q_v_raw, T_upd, mse_pre = self._firing_case()
        q_v_naive = jnp.maximum(q_v_raw, 0.0)
        mse_naive = self.c_pd * T_upd + self.L_v * q_v_naive
        injected = float(jnp.sum(mse_naive - mse_pre))
        deficit = float(jnp.sum(jnp.maximum(-q_v_raw, 0.0)))
        assert injected > 0.0
        assert injected == pytest.approx(self.L_v * deficit, rel=1e-10)

    def test_energy_consistent_floor_is_mse_neutral(self):
        """Flag ON: the floor conserves moist static energy to roundoff."""
        q_v_raw, T_upd, mse_pre = self._firing_case()
        q_v_out, T_out = energy_consistent_moisture_floor(q_v_raw, T_upd)
        mse_post = self.c_pd * T_out + self.L_v * q_v_out
        assert float(jnp.sum(mse_post - mse_pre)) == pytest.approx(0.0, abs=1e-6)

    def test_positivity_enforced(self):
        q_v_raw, T_upd, _ = self._firing_case()
        q_v_out, _ = energy_consistent_moisture_floor(q_v_raw, T_upd)
        assert bool((q_v_out >= 0.0).all())

    def test_q_v_matches_plain_floor(self):
        """q_v output is exactly max(q_v_raw, 0) — only T differs from naive."""
        q_v_raw, T_upd, _ = self._firing_case()
        q_v_out, _ = energy_consistent_moisture_floor(q_v_raw, T_upd)
        assert bool((q_v_out == jnp.maximum(q_v_raw, 0.0)).all())

    def test_no_op_when_already_nonnegative(self):
        """When the sink does not exhaust the vapour, the floor is a bitwise
        no-op (so the flag is bit-identical to the legacy path on every cell
        that does not clip)."""
        q_v_raw = jnp.array([0.012, 0.003, 1.0e-5, 0.0])  # all >= 0
        T_upd = jnp.array([295.0, 290.0, 288.0, 292.0])
        q_v_out, T_out = energy_consistent_moisture_floor(q_v_raw, T_upd)
        assert bool((q_v_out == q_v_raw).all())
        assert bool((T_out == T_upd).all())

    def test_differentiable_finite(self):
        """jax.grad through the floor stays finite at the q_v_raw = 0 kink."""
        def loss(q_v_raw, T_upd):
            q_v_out, T_out = energy_consistent_moisture_floor(q_v_raw, T_upd)
            return jnp.sum(q_v_out ** 2) + jnp.sum(T_out ** 2)
        q_v_raw, T_upd, _ = self._firing_case()
        g_q, g_T = jax.grad(loss, argnums=(0, 1))(q_v_raw, T_upd)
        assert bool(jnp.isfinite(g_q).all())
        assert bool(jnp.isfinite(g_T).all())


class TestFixTotalWater:
    """``fix_total_water`` scales all water species (q_v + q_c + q_r) by ONE
    uniform factor so the column-integrated total water hits a target — the
    conservation-correct building block for fix-moisture under precipitating
    microphysics (config.py warns fix_moisture rescales only q_v).  Wiring it
    into the AMIP step needs a precip-tracking total-water target (a new
    SegmentCarry field); this covers the pure-function contract meanwhile.
    """

    @pytest.fixture
    def grid(self):
        return create_cubed_sphere(8)

    def _tracers(self):
        # SPATIALLY VARYING fields so an erroneous non-uniform scaling (mean
        # right, elementwise wrong) cannot pass the elementwise checks below.
        shape_3d = (6, 8, 8, 4)
        base = jnp.linspace(0.001, 0.011, 6 * 8 * 8 * 4).reshape(shape_3d)
        return {
            "q_v": base,
            "q_c": 0.3 * base + 0.0002,
            "q_r": 0.1 * base + 0.0001,
        }

    def test_scales_all_species_to_target(self, grid):
        from legoesm.core.conservation import (
            fix_total_water, compute_global_moisture,
        )
        tracers = self._tracers()
        p_s = jnp.full((6, 8, 8), 1.013e5)
        dsigma = jnp.full((4,), 0.25)
        total = tracers["q_v"] + tracers["q_c"] + tracers["q_r"]
        current = compute_global_moisture(total, p_s, dsigma, grid)
        target = current * 0.9                      # 10% drift correction
        out = fix_total_water(tracers, target, p_s, dsigma, grid)
        # Column-integrated total water hits the target exactly.
        new_total = out["q_v"] + out["q_c"] + out["q_r"]
        new_int = compute_global_moisture(new_total, p_s, dsigma, grid)
        assert float(new_int) == pytest.approx(float(target), rel=1e-6)
        # ALL species scaled ELEMENTWISE by the SAME single factor (0.9),
        # preserving spatial structure and inter-species ratios.
        for name in ("q_v", "q_c", "q_r"):
            assert bool(jnp.allclose(out[name], 0.9 * tracers[name], rtol=1e-6))

    def test_identity_when_target_equals_current(self, grid):
        from legoesm.core.conservation import (
            fix_total_water, compute_global_moisture,
        )
        tracers = self._tracers()
        p_s = jnp.full((6, 8, 8), 1.013e5)
        dsigma = jnp.full((4,), 0.25)
        total = tracers["q_v"] + tracers["q_c"] + tracers["q_r"]
        current = compute_global_moisture(total, p_s, dsigma, grid)
        out = fix_total_water(tracers, current, p_s, dsigma, grid)
        for name in ("q_v", "q_c", "q_r"):
            assert float(jnp.mean(out[name] / tracers[name])) == pytest.approx(
                1.0, rel=1e-6)

    def test_only_named_species_scaled(self, grid):
        """A non-water tracer passed in the dict is left untouched."""
        from legoesm.core.conservation import fix_total_water, compute_global_moisture
        tracers = self._tracers()
        tracers["dust"] = jnp.full((6, 8, 8, 4), 3.0)
        p_s = jnp.full((6, 8, 8), 1.013e5)
        dsigma = jnp.full((4,), 0.25)
        total = tracers["q_v"] + tracers["q_c"] + tracers["q_r"]
        target = compute_global_moisture(total, p_s, dsigma, grid) * 0.8
        out = fix_total_water(tracers, target, p_s, dsigma, grid)
        assert bool(jnp.all(out["dust"] == tracers["dust"]))


# =========================================================================
# Cross-grid water positivity: shared borrow + energy-consistent hard floor
# (#1354/#1515 — every dycore routes through apply_water_positivity)
# =========================================================================

def _frozen_mse_cell(tracers, T):
    """Per-cell frozen MSE water+thermal part: c_pd*T + L_v*q_v - L_f*q_frozen."""
    g = lambda k: tracers[k].data if k in tracers else 0.0
    q_frozen = g("q_i") + g("q_s") + g("q_g")
    return constants.c_pd * T + constants.L_v * g("q_v") - constants.L_f * q_frozen


def _mixed_water_state():
    """A 3-level column with negatives in vapour AND ice (transport overshoot)."""
    F = lambda name, arr: Field(data=jnp.asarray(arr, dtype=jnp.float64),
                                name=name, dims=("z",), units="kg/kg")
    tracers = {
        "q_v": F("q_v", [1.0e-2, -3.0e-4, 5.0e-3]),   # negative aloft
        "q_c": F("q_c", [2.0e-4, 1.0e-4, -5.0e-5]),   # liquid, one negative
        "q_i": F("q_i", [-2.0e-4, 4.0e-4, 1.0e-4]),   # ice, negative at surface
        "q_s": F("q_s", [0.0, -1.0e-5, 2.0e-5]),
    }
    T = jnp.asarray([250.0, 240.0, 288.0], dtype=jnp.float64)
    dp = jnp.asarray([3000.0, 3000.0, 4000.0], dtype=jnp.float64)
    return tracers, T, dp


def test_borrow_conserves_every_species_column_integral():
    tracers, T, dp = _mixed_water_state()
    out, T_out = apply_water_positivity(
        tracers, T, dp, conservative=True, energy_consistent=False)
    # T untouched by the borrow (frozen-MSE-neutral, Claim A)
    assert jnp.allclose(T_out, T)
    for k in tracers:
        pre = float(jnp.sum(tracers[k].data * dp))
        post = float(jnp.sum(out[k].data * dp))
        assert jnp.all(out[k].data >= -1e-30), k
        # column integral preserved where it was >= 0 (all these are)
        assert abs(post - pre) <= 1e-9 * max(abs(pre), 1e-30) + 1e-18, (k, pre, post)


def test_hard_floor_energy_consistent_conserves_frozen_mse_incl_ice():
    tracers, T, dp = _mixed_water_state()
    h_pre = _frozen_mse_cell(tracers, T)  # per-cell, BEFORE floor
    out, T_out = apply_water_positivity(
        tracers, T, dp, conservative=False, energy_consistent=True)
    h_post = _frozen_mse_cell(out, T_out)
    for k in out:
        assert jnp.all(out[k].data >= 0.0), k
    # frozen MSE conserved per cell to roundoff (vapour cooled, ice warmed)
    assert jnp.allclose(h_post, h_pre, rtol=0, atol=1e-6), (h_pre, h_post)


def test_plain_floor_breaks_frozen_mse_nonvacuity():
    # Same floor WITHOUT the energy correction must NOT conserve frozen MSE,
    # else the energy-consistent test above proves nothing.
    tracers, T, dp = _mixed_water_state()
    h_pre = _frozen_mse_cell(tracers, T)
    out, T_out = apply_water_positivity(
        tracers, T, dp, conservative=False, energy_consistent=False)
    assert jnp.allclose(T_out, T)  # plain floor leaves T alone
    h_post = _frozen_mse_cell(out, T_out)
    # vapour floor at level 1 injects +L_v*3e-4; ice floor injects -L_f*...:
    # the net is nonzero somewhere -> NOT conserved.
    assert not jnp.allclose(h_post, h_pre, atol=1e-6)


def test_ice_correction_sign_is_warming():
    # An ice-only negative: flooring it up must WARM (opposite to vapour).
    F = lambda n, a: Field(data=jnp.asarray(a, dtype=jnp.float64), name=n,
                           dims=("z",), units="kg/kg")
    tracers = {"q_i": F("q_i", [-1.0e-4, 2.0e-4])}
    T = jnp.asarray([250.0, 250.0], dtype=jnp.float64)
    out, T_out = energy_consistent_water_floor(tracers, T)
    assert float(T_out[0]) > 250.0          # warmed where ice was floored up
    assert jnp.isclose(T_out[0], 250.0 + constants.L_f / constants.c_pd * 1.0e-4)
    assert jnp.isclose(T_out[1], 250.0)     # untouched where ice was positive


def test_water_floor_is_differentiable():
    tracers, T, _ = _mixed_water_state()

    def loss(qv0):
        tt = dict(tracers)
        tt["q_v"] = tt["q_v"].replace(data=tt["q_v"].data.at[1].set(qv0))
        out, T_out = energy_consistent_water_floor(tt, T)
        return jnp.sum(T_out) + jnp.sum(out["q_v"].data)

    g = jax.grad(loss)(-3.0e-4)
    assert jnp.isfinite(g)
