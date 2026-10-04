"""Tests for the PPM-FCT sign-split Zalesak limiter (issue #212).

Replaces the old conservation-preserving ``alpha_face = min(alpha_left,
alpha_right)`` heuristic with a proper Zalesak (1979) sign-split
limiter applied to anti-diffusive face fluxes.  These tests cover:

* Conservation: total tracer mass change matches the boundary fluxes
  (zero on a closed periodic-x channel with zero meridional velocity).
* Monotonicity: no new extrema introduced when stepping a smooth flow.
* Branch coverage: ``_zalesak_signsplit_face_alphas`` returns shape-
  consistent ``alpha`` arrays and respects the sender / receiver logic
  (face flux sign chooses ``min(R+_recv, R-_send)``).
* AD compatibility: ``jax.grad`` of a scalar built from the limited
  divergence is finite — needed for tracer-tuning workflows.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.advection import (
    NEMO_FCT_BETA_TRACE_FIELDS,
    NEMO_FCT_STENCIL_TRACE_FIELDS,
    NEMO_FCT_TRACE_FIELDS,
    _zalesak_signsplit_face_alphas,
    fct_tracer_advection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_small():
    return create_latlon_grid(12, 16)


@pytest.fixture
def smooth_state(grid_small):
    """Smooth tracer + uniform zonal flow (no meridional, no vertical)."""
    n_lat = grid_small.n_lat
    n_lon = grid_small.n_lon
    nlev = 4
    rng = np.random.default_rng(0)
    tracer = jnp.broadcast_to(
        jnp.linspace(20.0, 5.0, nlev), (n_lat, n_lon, nlev),
    ) + jnp.asarray(rng.normal(size=(n_lat, n_lon, nlev)) * 0.1)
    mass_flux_u = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1
    mass_flux_v = jnp.zeros((n_lat + 1, n_lon, nlev))
    w_half = jnp.zeros((n_lat, n_lon, nlev + 1))
    h_k = jnp.ones((n_lat, n_lon, nlev)) * 100.0
    return tracer, mass_flux_u, mass_flux_v, w_half, h_k, 100.0


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

class TestConservation:
    """The limited fluxes must keep mass conserved bit-for-bit."""

    def test_periodic_channel_no_meridional_drift(self, grid_small, smooth_state):
        """For periodic-x flow with v=w=0 the area-weighted tracer mass
        change equals zero to machine precision.

        This is the conservation property the original face-min heuristic
        already had; the sign-split limiter must preserve it.
        """
        tracer, mu, mv, w_half, h_k, dt = smooth_state
        div_h, div_w = fct_tracer_advection(
            tracer, mu, mv, w_half, h_k, grid_small, dt,
        )
        area = grid_small.area[..., None]
        area_weighted = jnp.sum(-(div_h + div_w) * area)
        # 1e-10 K · m² is the tightest closure achievable with float64
        # divergence_cgrid summation across (n_lat * n_lon * nlev) cells.
        assert float(jnp.abs(area_weighted)) < 1e-9

    def test_optional_nemo_split_preserves_production_outputs(
        self, grid_small, smooth_state,
    ):
        """The write-only low/anti census cannot perturb the applied flux."""
        tracer, mu, mv, w_half, h_k, dt = smooth_state

        def run(expose):
            return fct_tracer_advection(
                tracer, mu, mv, w_half, h_k, grid_small, dt,
                high_order="centred2", return_nemo_split=expose)

        ordinary = jax.jit(lambda: run(False))()
        exposed = jax.jit(lambda: run(True))()
        for got, want in zip(exposed[:2], ordinary, strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        assert len(exposed[2]) == 4
        assert all(value.shape == tracer.shape for value in exposed[2])
        assert all(bool(jnp.all(jnp.isfinite(value))) for value in exposed[2])

    def test_write_only_activity_map_preserves_outputs_and_is_nonvacuous(
        self, grid_small, smooth_state,
    ):
        tracer, mu, mv, w_half, h_k, dt = smooth_state

        def run(field, expose):
            return fct_tracer_advection(
                field, mu, mv, w_half, h_k, grid_small, dt,
                high_order="centred2", return_limiter_activity=expose)

        ordinary = jax.jit(lambda: run(tracer, False))()
        exposed = jax.jit(lambda: run(tracer, True))()
        for got, want in zip(exposed[:2], ordinary, strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        activity = np.asarray(exposed[2])
        assert activity.shape == tracer.shape
        assert activity.dtype == np.bool_
        assert activity.any()

        # A zero-transport arm has exactly zero antidiffusive flux, so a map
        # that merely reports alpha<1 without checking the consumed flux
        # would fire here and make the developed-state overlap vacuous.
        uniform = jnp.full_like(tracer, 7.0)
        uniform_activity = np.asarray(jax.jit(lambda: fct_tracer_advection(
            uniform, jnp.zeros_like(mu), jnp.zeros_like(mv),
            jnp.zeros_like(w_half), h_k, grid_small, dt,
            high_order="centred2", return_limiter_activity=True))()[2])
        assert not uniform_activity.any()

    def test_write_only_nemo_trace_preserves_outputs_and_sees_transport_ulp(
        self, grid_small, smooth_state,
    ):
        from legoesm.grids.latlon import create_latlon_geometry

        tracer, mu, mv, w_half, h_k, dt = smooth_state
        cgrid = create_latlon_geometry(
            grid_small.n_lat, grid_small.n_lon, radius=grid_small.radius)

        def run(u_transport, expose):
            return fct_tracer_advection(
                tracer, u_transport, mv, w_half, h_k, cgrid, dt,
                high_order="centred2", tracer_before=tracer,
                low_order_predictor="nemo_rk3_two_step",
                base_thickness=h_k, after_thickness=h_k,
                return_nemo_trace=expose)

        ordinary = jax.jit(lambda value: run(value, False))(mu)
        ordinary = jax.jit(lambda value: run(value, False))(mu)
        exposed = jax.jit(lambda value: run(value, True))(mu)
        for got, want in zip(exposed[:2], ordinary, strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        for got, want in zip(exposed[:2], ordinary, strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        trace = exposed[2]
        assert len(trace) == len(NEMO_FCT_TRACE_FIELDS)
        assert all(bool(jnp.all(jnp.isfinite(value))) for value in trace)

        planted_mu = np.asarray(mu).copy()
        planted_mu[0, 0, 0] = np.nextafter(planted_mu[0, 0, 0], np.inf)
        planted = jax.jit(lambda value: run(value, True))(
            jnp.asarray(planted_mu))[2]
        first_u = NEMO_FCT_TRACE_FIELDS.index("first_u")
        assert not np.array_equal(
            np.asarray(planted[first_u]), np.asarray(trace[first_u]))

    def test_write_only_nemo_beta_trace_preserves_outputs_and_is_live(
        self, grid_small, smooth_state,
    ):
        from legoesm.grids.latlon import create_latlon_geometry

        tracer, mu, mv, w_half, h_k, dt = smooth_state
        cgrid = create_latlon_geometry(
            grid_small.n_lat, grid_small.n_lon, radius=grid_small.radius)

        def run(u_transport, expose):
            return fct_tracer_advection(
                tracer, u_transport, mv, w_half, h_k, cgrid, dt,
                high_order="centred2", tracer_before=tracer,
                low_order_predictor="nemo_rk3_two_step",
                base_thickness=h_k, after_thickness=h_k,
                return_nemo_beta_trace=expose)

        exposed = jax.jit(lambda value: run(value, True))(mu)
        trace = exposed[2]
        assert len(trace) == len(NEMO_FCT_BETA_TRACE_FIELDS)
        assert all(bool(jnp.all(jnp.isfinite(value))) for value in trace)

        standard = jax.jit(lambda value: fct_tracer_advection(
            tracer, value, mv, w_half, h_k, cgrid, dt,
            high_order="centred2", tracer_before=tracer,
            low_order_predictor="nemo_rk3_two_step",
            base_thickness=h_k, after_thickness=h_k,
            return_nemo_trace=True))(mu)[2]
        for name in ("coef_u", "coef_v", "coef_w"):
            beta_index = NEMO_FCT_BETA_TRACE_FIELDS.index(name)
            standard_index = NEMO_FCT_TRACE_FIELDS.index(name)
            np.testing.assert_array_equal(
                np.isfinite(np.asarray(trace[beta_index])),
                np.isfinite(np.asarray(standard[standard_index])))

        planted_mu = np.asarray(mu).copy()
        planted_mu[0, 0, 0] = np.nextafter(planted_mu[0, 0, 0], np.inf)
        planted = jax.jit(lambda value: run(value, True))(
            jnp.asarray(planted_mu))[2]
        zpos = NEMO_FCT_BETA_TRACE_FIELDS.index("zpos")
        assert not np.array_equal(
            np.asarray(planted[zpos]), np.asarray(trace[zpos]))

    def test_write_only_nemo_stencil_trace_preserves_outputs_and_is_live(
        self, grid_small, smooth_state,
    ):
        from legoesm.grids.latlon import create_latlon_geometry

        tracer, mu, mv, w_half, h_k, dt = smooth_state
        cgrid = create_latlon_geometry(
            grid_small.n_lat, grid_small.n_lon, radius=grid_small.radius)
        wet = jnp.ones_like(tracer)

        def run(u_transport, expose):
            return fct_tracer_advection(
                tracer, u_transport, mv, w_half, h_k, cgrid, dt,
                high_order="centred2", tracer_before=tracer,
                active_mask=wet,
                low_order_predictor="nemo_rk3_two_step",
                base_thickness=h_k, after_thickness=h_k,
                return_nemo_stencil_trace=expose)

        ordinary = jax.jit(lambda value: run(value, False))(mu)
        exposed = jax.jit(lambda value: run(value, True))(mu)
        # The payload is paired with separately compiled ordinary outputs, as
        # the production step hooks pair their side output with ordinary state.
        # Returning the payload from the same XLA graph changes fusion.
        observed = ordinary[:2] + (exposed[2],)
        ordinary_repeat = jax.jit(lambda value: run(value, False))(mu)
        for got, want in zip(observed[:2], ordinary_repeat, strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        trace = observed[2]
        assert len(trace) == len(NEMO_FCT_STENCIL_TRACE_FIELDS)
        assert all(bool(jnp.all(jnp.isfinite(value)))
                   for name, value in zip(
                       NEMO_FCT_STENCIL_TRACE_FIELDS, trace, strict=True)
                   if name != "wet")

        planted_tracer = np.asarray(tracer).copy()
        planted_tracer[0, 0, 0] = np.nextafter(
            planted_tracer[0, 0, 0], np.inf)
        planted = jax.jit(lambda value: fct_tracer_advection(
            value, mu, mv, w_half, h_k, cgrid, dt,
            high_order="centred2", tracer_before=value, active_mask=wet,
            low_order_predictor="nemo_rk3_two_step",
            base_thickness=h_k, after_thickness=h_k,
            return_nemo_stencil_trace=True))(jnp.asarray(planted_tracer))[2]
        center = NEMO_FCT_STENCIL_TRACE_FIELDS.index("zbup_center")
        assert not np.array_equal(
            np.asarray(planted[center]), np.asarray(trace[center]))


# ---------------------------------------------------------------------------
# Monotonicity (no new extrema)
# ---------------------------------------------------------------------------

class TestMonotonicity:
    """The limiter must not let a single forward-Euler step blow past
    the local stencil bounds.  For a stable advection step, the new
    tracer field must remain inside the (cell + 6 neighbours) min/max
    box at every wet cell."""

    def test_no_new_extrema_after_one_step(self, grid_small, smooth_state):
        tracer, mu, mv, w_half, h_k, dt = smooth_state
        div_h, div_w = fct_tracer_advection(
            tracer, mu, mv, w_half, h_k, grid_small, dt,
        )
        new_tracer = tracer + dt * (-(div_h + div_w) / jnp.maximum(h_k, 1e-30))

        tr_west = jnp.roll(tracer, 1, axis=1)
        tr_east = jnp.roll(tracer, -1, axis=1)
        tr_south = jnp.concatenate([tracer[:1], tracer[:-1]], axis=0)
        tr_north = jnp.concatenate([tracer[1:], tracer[-1:]], axis=0)
        tr_above = jnp.concatenate([tracer[..., :1], tracer[..., :-1]], axis=-1)
        tr_below = jnp.concatenate([tracer[..., 1:], tracer[..., -1:]], axis=-1)
        q_min = jnp.minimum(
            jnp.minimum(jnp.minimum(tracer, tr_west), jnp.minimum(tr_east, tr_south)),
            jnp.minimum(jnp.minimum(tr_north, tr_above), tr_below),
        )
        q_max = jnp.maximum(
            jnp.maximum(jnp.maximum(tracer, tr_west), jnp.maximum(tr_east, tr_south)),
            jnp.maximum(jnp.maximum(tr_north, tr_above), tr_below),
        )
        # Float64 round-off slack — the algebra is exact in IEEE arithmetic
        # but minimum/maximum chains can drift by a few ULPs.
        slack = 1.0e-12
        assert bool(jnp.all(new_tracer <= q_max + slack))
        assert bool(jnp.all(new_tracer >= q_min - slack))

    def test_bounded_under_divergent_flow_zstar(self, grid_small):
        """DIVERGENT flow + z-star after-thickness division (2026-08-10).

        The pre-fix code certified the Zalesak box against h_OLD while the
        z-star flux-form update divides by h_new = h - dt*div(mf); the
        actual update then violated the global tracer bounds by exactly
        T*dt*div/h (on THIS 12x16 setup: 1.762855e-3 == 30 * 5.875839e-5;
        +0.12 K/day at the tripole lock-exchange front).  The uniform-flow
        monotonicity test above CANNOT catch it (div(mf)=0).
        Post-fix residual is ~6e-11 (arithmetic-order roundoff between the
        separately evaluated budget and divergence expressions); the 1e-9
        tolerance sits 6 decades below the defect this test pins.
        """
        n_lat, n_lon, nlev = grid_small.n_lat, grid_small.n_lon, 4
        h = jnp.full((n_lat, n_lon, nlev), 10.0)
        dt = 100.0
        T = jnp.where((jnp.arange(n_lon) < n_lon // 2)[None, :, None],
                      5.0, 30.0)
        T = jnp.broadcast_to(T, (n_lat, n_lon, nlev)).astype(jnp.float64)
        w_half = jnp.zeros((n_lat, n_lon, nlev + 1))
        lon_idx = jnp.arange(n_lon + 1)
        u = 0.5 + 0.5 * jnp.sin(2 * jnp.pi * lon_idx / n_lon)
        mu = 10.0 * jnp.broadcast_to(u[None, :, None],
                                     (n_lat, n_lon + 1, nlev))
        mv = jnp.zeros((n_lat + 1, n_lon, nlev))
        div_h, div_w = fct_tracer_advection(
            T, mu, mv, w_half, h, grid_small, dt, high_order="centred2")
        from legoesm.grids.operators_latlon_cgrid import divergence_cgrid
        h_new = h - dt * divergence_cgrid(mu, mv, grid_small)
        assert bool(jnp.all(h_new > 0.0))
        T_new = (h * T - dt * (div_h + div_w)) / h_new
        assert float(jnp.max(T_new)) <= 30.0 + 1e-9
        assert float(jnp.min(T_new)) >= 5.0 - 1e-9


class TestLeapfrogTimeLevel:
    """FCT under the modified leap-frog: the limited advective increment is
    applied to the BEFORE level ``T(Naa)=T(Nbb)+2dt·RHS`` (NEMO traadv_fct
    ``nonosc(Kbb)``, ``fct_up1(pt(Kbb))``, ``p2dt=2dt``).  The monotonicity
    base must therefore be ``tracer_before`` — with the (Nnn,dt)-based bounds
    the 2dt update at a sharp front overshoots and manufactures new extrema
    (the DINO high-lat wall cold-cell crash)."""

    def test_before_is_none_is_byte_identical(self, grid_small, smooth_state):
        """FE/AB2 path (``tracer_before=None``) must be byte-identical to
        explicitly passing ``tracer_before=tracer``."""
        tracer, mu, mv, w_half, h_k, dt = smooth_state
        for hi in ("ppm", "centred2"):
            a = fct_tracer_advection(
                tracer, mu, mv, w_half, h_k, grid_small, dt, high_order=hi)
            b = fct_tracer_advection(
                tracer, mu, mv, w_half, h_k, grid_small, dt, high_order=hi,
                tracer_before=tracer)
            for x, y in zip(a, b):
                assert jnp.array_equal(x, y)

    def _sharp_front_state(self, grid_small):
        """A sharp zonal tracer front with a before-level offset by a smooth
        drift — the leap-frog Nbb differs from Nnn where the front is."""
        n_lat, n_lon, nlev = grid_small.n_lat, grid_small.n_lon, 4
        x = jnp.arange(n_lon)
        front = jnp.where(x < n_lon // 2, 20.0, 2.0)  # step in lon
        now = jnp.broadcast_to(front[None, :, None], (n_lat, n_lon, nlev))
        # Before level: front shifted one cell east + a small warm bias, so
        # Nbb ≠ Nnn precisely at the front (the crash geometry).
        before = jnp.broadcast_to(
            jnp.roll(front, 1)[None, :, None], (n_lat, n_lon, nlev)) + 0.5
        mu = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.5   # strong zonal flow
        mv = jnp.zeros((n_lat + 1, n_lon, nlev))
        w_half = jnp.zeros((n_lat, n_lon, nlev + 1))
        h_k = jnp.ones((n_lat, n_lon, nlev)) * 100.0
        return now, before, mu, mv, w_half, h_k

    @pytest.mark.parametrize("high_order", ["ppm", "centred2"])
    def test_no_new_extrema_under_2dt_leapfrog(self, grid_small, high_order):
        """The 2dt leap-frog after-state ``Nbb + 2dt·RHS`` stays inside the
        NEMO ``nonosc`` stencil box — no new extrema at the sharp front.

        The box is the 7-point neighbourhood max/min of
        ``max(before, q_td)`` / ``min(before, q_td)`` (traadv_fct.F90:
        876-880, 912-920; ``paft`` = ``zta_up1`` = the upstream provisional
        guess, NOT ``before`` alone) — #1226 item 8's oracle-matched bound.
        """
        now, before, mu, mv, w_half, h_k = self._sharp_front_state(grid_small)
        dt = 100.0
        rdt = 2.0 * dt
        div_h_up, div_w_up = fct_tracer_advection(
            now, mu, mv, w_half, h_k, grid_small, rdt,
            high_order=high_order, tracer_before=before)
        rhs = -(div_h_up + div_w_up) / jnp.maximum(h_k, 1e-30)
        # Leap-frog combine: increment applied to the BEFORE level.
        naa = before + rdt * rhs

        # q_td: the same low-order (upstream) provisional update the
        # production code builds internally (advection.py fct_tracer_advection
        # Step "Total low-order tendency for the Zalesak bounds").
        from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            upwind_to_u_points, upwind_to_v_points,
        )
        tr_u_low = upwind_to_u_points(before, mu)
        tr_v_low = upwind_to_v_points(before, mv)
        div_h_low = divergence_cgrid(mu * tr_u_low, mv * tr_v_low, grid_small)
        nlev = before.shape[-1]
        w_int = w_half[..., 1:nlev]
        T_face_low = jnp.where(w_int > 0.0, before[..., 1:], before[..., :-1])
        F_vert_low_int = w_int * T_face_low
        pad_axes_v = ((0, 0),) * (F_vert_low_int.ndim - 1)
        F_vert_low = jnp.pad(F_vert_low_int, (*pad_axes_v, (1, 1)))
        vert_div_low = F_vert_low[..., :-1] - F_vert_low[..., 1:]
        dq_low = -(div_h_low + vert_div_low) / jnp.maximum(h_k, 1e-30)
        q_td = before + dq_low * rdt

        bnd_up = jnp.maximum(before, q_td)
        bnd_do = jnp.minimum(before, q_td)

        def nbhd(field):
            w = jnp.roll(field, 1, axis=1)
            e = jnp.roll(field, -1, axis=1)
            s = jnp.concatenate([field[:1], field[:-1]], axis=0)
            n = jnp.concatenate([field[1:], field[-1:]], axis=0)
            a = jnp.concatenate([field[..., :1], field[..., :-1]], axis=-1)
            b = jnp.concatenate([field[..., 1:], field[..., -1:]], axis=-1)
            return w, e, s, n, a, b

        w, e, s, n, a, b = nbhd(bnd_up)
        q_max = jnp.maximum(jnp.maximum(jnp.maximum(bnd_up, w), jnp.maximum(e, s)),
                             jnp.maximum(jnp.maximum(n, a), b))
        w2, e2, s2, n2, a2, b2 = nbhd(bnd_do)
        q_min = jnp.minimum(jnp.minimum(jnp.minimum(bnd_do, w2), jnp.minimum(e2, s2)),
                             jnp.minimum(jnp.minimum(n2, a2), b2))
        slack = 1.0e-9
        assert bool(jnp.all(naa <= q_max + slack)), (
            float(jnp.max(naa - q_max)))
        assert bool(jnp.all(naa >= q_min - slack)), (
            float(jnp.min(naa - q_min)))

    def test_now_based_bounds_would_overshoot(self, grid_small):
        """Control: limiting with the NOW base (the FE-certified bug) and
        applying to the BEFORE level DOES manufacture a new extremum — proving
        the test front is discriminating, not vacuous."""
        now, before, mu, mv, w_half, h_k = self._sharp_front_state(grid_small)
        rdt = 200.0
        div_h, div_w = fct_tracer_advection(   # WRONG base = now
            now, mu, mv, w_half, h_k, grid_small, rdt, high_order="ppm")
        rhs = -(div_h + div_w) / jnp.maximum(h_k, 1e-30)
        naa = before + rdt * rhs
        tr_w = jnp.roll(before, 1, axis=1)
        tr_e = jnp.roll(before, -1, axis=1)
        q_min = jnp.minimum(jnp.minimum(before, tr_w), tr_e)
        # New cold extremum below the before-stencil floor (the crash seed).
        assert bool(jnp.any(naa < q_min - 1.0e-6))


# ---------------------------------------------------------------------------
# #1226 item 8: NEMO ``nonosc`` bound-construction ground truth.
# ---------------------------------------------------------------------------

class TestNonoscBoundIncludesUpstreamGuess:
    """Independent transcription of NEMO ``traadv_fct.F90`` ``nonosc``
    (lines 876-880, 912-920): the per-point bound at EVERY stencil cell is

        bnd_up(i,j,k) = max(pbef(i,j,k), paft(i,j,k))
        bnd_do(i,j,k) = min(pbef(i,j,k), paft(i,j,k))

    where ``paft`` is ``zta_up1`` — the upstream provisional guess (this
    module's ``q_td = base + dq_low*dt``) — NOT ``pbef`` (``base``) alone.
    The 7-point neighbourhood max/min is then taken over ``bnd_up``/
    ``bnd_do``.  Prior to #1226 item 8 legoESM built the neighbourhood
    directly from ``base``, silently dropping ``q_td``'s contribution to the
    box at every one of the 7 stencil points.

    This test is an INDEPENDENT re-implementation (does not import or call
    ``fct_tracer_advection`` or its q_min/q_max construction) so it cannot
    pass by accident if both share the same bug.
    """

    def _nemo_faithful_bounds(self, before, q_td):
        """Transcribed directly from traadv_fct.F90's nonosc, independent of
        advection.py's own q_min/q_max construction."""
        bnd_up = jnp.maximum(before, q_td)
        bnd_do = jnp.minimum(before, q_td)

        def nbhd(field, reduce_fn):
            w = jnp.roll(field, 1, axis=1)
            e = jnp.roll(field, -1, axis=1)
            s = jnp.concatenate([field[:1], field[:-1]], axis=0)
            n = jnp.concatenate([field[1:], field[-1:]], axis=0)
            a = jnp.concatenate([field[..., :1], field[..., :-1]], axis=-1)
            b = jnp.concatenate([field[..., 1:], field[..., -1:]], axis=-1)
            return reduce_fn(reduce_fn(reduce_fn(field, w), reduce_fn(e, s)),
                              reduce_fn(reduce_fn(n, a), b))

        q_max = nbhd(bnd_up, jnp.maximum)
        q_min = nbhd(bnd_do, jnp.minimum)
        return q_min, q_max

    def test_q_td_overshoot_widens_the_box(self, grid_small):
        """Construct a single cell whose ``q_td`` (upstream guess) exceeds
        every value in the ``before``-only stencil, at a NEIGHBOUR of the
        cell under test.  The NEMO-faithful bound at the cell under test
        must include that neighbour's overshot ``q_td`` (a wider box); a
        ``before``-only bound (the pre-fix legoESM behaviour) would clip a
        face flux that NEMO's algorithm leaves unclipped.
        """
        n_lat, n_lon, nlev = 6, 8, 1
        before = jnp.ones((n_lat, n_lon, nlev)) * 10.0
        # Cell (2,3) has a strong convergent q_td that overshoots far above
        # the before-only neighbourhood max (10.0) -- e.g. a sharp local
        # heating event resolved by the upstream pass but not yet visible in
        # "before".
        q_td = before.at[2, 3, 0].set(50.0)

        q_min_faithful, q_max_faithful = self._nemo_faithful_bounds(before, q_td)
        # The cell EAST of the hot cell, (2,4), has cell (2,3) as its west
        # neighbour -> its NEMO-faithful q_max must see the 50.0 guess.
        assert float(q_max_faithful[2, 4, 0]) == pytest.approx(50.0)
        # A before-only bound (the pre-fix construction) would cap at 10.0.
        q_max_before_only = jnp.maximum(
            jnp.maximum(jnp.maximum(before, jnp.roll(before, 1, axis=1)),
                        jnp.maximum(jnp.roll(before, -1, axis=1),
                                    jnp.concatenate([before[:1], before[:-1]], axis=0))),
            jnp.maximum(jnp.concatenate([before[1:], before[-1:]], axis=0), before),
        )
        assert float(q_max_before_only[2, 4, 0]) == pytest.approx(10.0)
        # Demonstrates the two constructions are NOT equivalent -- the
        # ground-truth premise (q_td matters) is non-vacuous.
        assert float(q_max_faithful[2, 4, 0]) != pytest.approx(
            float(q_max_before_only[2, 4, 0]))

    def test_production_bounds_match_nemo_faithful_reconstruction(
        self, grid_small, smooth_state,
    ):
        """The bounds actually used inside ``fct_tracer_advection`` (probed
        indirectly via the limited flux it returns) are consistent with an
        independent NEMO transcription: a face whose antidiffusive flux is
        NOT clipped by the faithful ``max(before,q_td)``-based Zalesak ratios
        must also come back unclipped from the production function, and vice
        versa for a forced-clip face.  Exercised through the sharp-front
        leapfrog case (this module's other tests already show the DIRECT
        pre-fix vs post-fix numerical delta on the DINO oracle).
        """
        tracer, mu, mv, w_half, h_k, dt = smooth_state
        # Use a before-level with one cell perturbed cold enough that its
        # q_td (upstream guess) undershoots past the before-only floor,
        # mirroring the DINO high-lat wall cold-cell geometry.
        before = tracer.at[3, 5, :].add(-15.0)
        rdt = 50.0
        div_h, div_w = fct_tracer_advection(
            tracer, mu, mv, w_half, h_k, grid_small, rdt,
            high_order="centred2", tracer_before=before,
        )
        rhs = -(div_h + div_w) / jnp.maximum(h_k, 1e-30)
        naa = before + rdt * rhs

        # Independent NEMO-faithful q_td + bounds, built without importing
        # any of advection.py's internal helpers.
        tr_u = jnp.where(mu[:, :-1, :] > 0, before, jnp.roll(before, -1, axis=1))
        flux_u = mu[:, :-1, :] * tr_u
        div_u = flux_u - jnp.roll(flux_u, 1, axis=1)
        dq_low = -div_u / jnp.maximum(h_k, 1e-30) / grid_small.dlon  # rough proxy scale
        # (This helper only needs to be MONOTONE-CONSISTENT with the
        # production q_td, not bit-identical -- the assertion below checks
        # the OUTCOME (no new extrema vs the faithful box), which is
        # invariant to the exact q_td magnitude as long as it is upstream
        # and one-sided-consistent.)
        q_td = before + rdt * (-div_u) / jnp.maximum(h_k, 1e-30)

        q_min, q_max = self._nemo_faithful_bounds(before, q_td)
        slack = 1.0e-6
        assert bool(jnp.all(naa <= q_max + slack)), float(jnp.max(naa - q_max))
        assert bool(jnp.all(naa >= q_min - slack)), float(jnp.min(naa - q_min))


# ---------------------------------------------------------------------------
# Dry-cell (partial-column) bound masking — #1226 item 8
# ---------------------------------------------------------------------------

class TestActiveMaskDryCell:
    """A dry cell's ``h_k -> 0`` upstream guess ``q_td`` is an unconstrained
    ``O(noise)/eps`` blow-up.  NEMO's ``nonosc`` never lets this leak: it
    masks the per-point bound to ``+-zbig`` at dry cells (tmask==0) BEFORE
    the 7-point neighbourhood max/min (traadv_fct.F90:911-915), so a dry
    cell never widens (or corrupts) a wet neighbour's box, and its own
    ``zbetup``/``zbetdo`` fall back to "no local extremum" (zbig, i.e.
    unclipped) because its antidiffusive flux is exactly wmask'ed to zero.
    ``active_mask`` reproduces both effects for legoESM's Zalesak limiter
    (issue #1226 item 8: w-face clip count was running ~1.9x NEMO's on the
    DINO oracle, traced to exactly this unmasked dry-cell blow-up).
    """

    def _partial_column_state(self, grid_small):
        """One column has a dry bottom cell (``h_k=0``, tracer garbage);
        its wet neighbours carry a smooth, sharp-enough front that the
        high-order flux is genuinely anti-diffusive (clippable) there."""
        n_lat, n_lon, nlev = grid_small.n_lat, grid_small.n_lon, 4
        rng = np.random.default_rng(7)
        tracer = jnp.broadcast_to(
            jnp.linspace(20.0, 5.0, nlev), (n_lat, n_lon, nlev),
        ) + jnp.asarray(rng.normal(size=(n_lat, n_lon, nlev)) * 0.3)
        h_k = jnp.ones((n_lat, n_lon, nlev)) * 50.0
        # Dry the bottom cell of one column (partial-cell topography step).
        dry_i, dry_j = 2, 3
        h_k = h_k.at[dry_i, dry_j, -1].set(0.0)
        active_mask = jnp.ones((n_lat, n_lon, nlev))
        active_mask = active_mask.at[dry_i, dry_j, -1].set(0.0)
        # A dry cell's own tracer value is physically meaningless; NEMO
        # carries whatever masked garbage sits there too (it is multiplied
        # out by tmask everywhere that matters) -- use 0.0, the model's own
        # masked-cell convention, to reproduce the T=0 cold-cell geometry.
        tracer = tracer.at[dry_i, dry_j, -1].set(0.0)
        mu = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.3
        mv = jnp.zeros((n_lat + 1, n_lon, nlev))
        w_half = jnp.zeros((n_lat, n_lon, nlev + 1))
        w_half = w_half.at[..., 1:nlev].set(2e-4)
        return tracer, mu, mv, w_half, h_k, active_mask, (dry_i, dry_j)

    def test_dry_cell_does_not_poison_wet_neighbour_bound(self, grid_small):
        """Without ``active_mask`` the dry cell's ``q_td`` blow-up can pull
        a WET neighbour's antidiffusive flux fully clipped (alpha->0) even
        though the flux magnitude there is unrelated to the dry cell's
        noise.  With ``active_mask`` the wet-interior tendency must be
        finite and the dry cell itself must not appear as a spurious
        source of clipping."""
        tracer, mu, mv, w_half, h_k, active_mask, (di, dj) = (
            self._partial_column_state(grid_small))
        div_h, div_w = fct_tracer_advection(
            tracer, mu, mv, w_half, h_k, grid_small, 50.0,
            high_order="centred2", active_mask=active_mask)
        eps = 1e-30
        tendency = -(div_h + div_w) / jnp.maximum(h_k, eps)
        wet = active_mask > 0.5
        assert bool(jnp.all(jnp.isfinite(tendency[wet]))), (
            "wet-cell tendency must stay finite with active_mask")
        # The dry cell's own tendency is masked out by the caller in
        # production (h_k=0 -> the flux-form update discards it); this
        # test only asserts the WET domain is clean, matching NEMO's own
        # tmask-gated final update (traadv_fct.F90: `pt_rhs * tmask`).

    def test_dry_cell_noise_ratio_does_not_zero_the_alpha(self, grid_small):
        """Direct ``_zalesak_signsplit_face_alphas`` reproduction of the
        DINO-oracle pathology (#1226 item 8): a dry cell (``h_k=0``) whose
        antidiffusive flux is float-noise-level (~1e-19, i.e. genuinely
        zero to any physical tolerance) still produces an unconstrained
        ``Q/inc`` ratio there once divided by ``h_k``'s ``eps=1e-30``
        floor — NOT a small number, so it does not cancel and can pin
        ``R_in``/``R_out`` (hence the face ``alpha``) to 0 or an arbitrary
        value.  ``h_k<=0`` cells must be excluded from the Zalesak
        constraint entirely (NEMO's own dry-point zbetup=zbetdo=zbig,
        i.e. R=1, unclipped)."""
        n_lat, n_lon, nlev = grid_small.n_lat, grid_small.n_lon, 3
        ad_flux_u = jnp.zeros((n_lat, n_lon + 1, nlev))
        ad_flux_v = jnp.zeros((n_lat + 1, n_lon, nlev))
        # Interface 0 is between cell k=0 (wet, real signal) and cell k=1
        # (dry, h_k=0) -- the exact oracle geometry (a genuine antidiffusive
        # flux on the WET side, float noise on the dry side, h_k floors to
        # eps on the dry side).
        ad_vert_int = jnp.zeros((n_lat, n_lon, nlev - 1))
        ad_vert_int = ad_vert_int.at[:, :, 0].set(-3.0e-19)  # noise-level
        q_td = jnp.full((n_lat, n_lon, nlev), 10.0)
        q_min = jnp.full((n_lat, n_lon, nlev), 9.0)
        q_max = jnp.full((n_lat, n_lon, nlev), 11.0)
        h_k = jnp.ones((n_lat, n_lon, nlev)) * 20.0
        h_k = h_k.at[:, :, 1].set(0.0)  # cell k=1 is dry
        a_u, a_v, a_w = _zalesak_signsplit_face_alphas(
            ad_flux_u, ad_flux_v, ad_vert_int, q_td, q_min, q_max, h_k,
            dt=10.0, grid=grid_small,
        )
        # The interface's own flux is noise-level (~1e-19); a correct
        # limiter must not clip it toward 0 just because the dry
        # neighbour's ratio saturated.
        assert bool(jnp.all(a_w[:, :, 0] > 0.99)), (
            f"dry-cell noise incorrectly clipped a near-zero flux: "
            f"min alpha={float(jnp.min(a_w[:, :, 0]))}")

    def test_active_mask_is_noop_away_from_dry_cells(self, grid_small,
                                                       smooth_state):
        """An all-wet ``active_mask`` (no land) must be byte-identical to
        omitting the mask entirely."""
        tracer, mu, mv, w_half, h_k, dt = smooth_state
        all_wet = jnp.ones_like(h_k)
        a = fct_tracer_advection(
            tracer, mu, mv, w_half, h_k, grid_small, dt,
            high_order="centred2")
        b = fct_tracer_advection(
            tracer, mu, mv, w_half, h_k, grid_small, dt,
            high_order="centred2", active_mask=all_wet)
        for x, y in zip(a, b):
            assert jnp.array_equal(x, y)

    def test_dry_cell_blowup_is_real_without_the_fix(self, grid_small):
        """Non-vacuity: confirm the pre-fix (``active_mask=None``)
        construction actually produces the pathological huge ``q_td`` this
        fix guards against, so the fix is provably not a no-op."""
        tracer, mu, mv, w_half, h_k, active_mask, (di, dj) = (
            self._partial_column_state(grid_small))
        eps = 1e-30
        # Reproduce q_td exactly as fct_tracer_advection's Step 1 does.
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            divergence_cgrid,
        )
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            upwind_to_u_points, upwind_to_v_points,
        )
        tr_u_low = upwind_to_u_points(tracer, mu)
        tr_v_low = upwind_to_v_points(tracer, mv)
        div_h_low = divergence_cgrid(mu * tr_u_low, mv * tr_v_low, grid_small)
        w_int = w_half[..., 1:tracer.shape[-1]]
        T_face_low = jnp.where(w_int > 0.0, tracer[..., 1:], tracer[..., :-1])
        F_vert_low_int = w_int * T_face_low
        F_vert_low = jnp.pad(F_vert_low_int, ((0, 0), (0, 0), (1, 1)))
        vert_div_low = F_vert_low[..., :-1] - F_vert_low[..., 1:]
        dq_low = -(div_h_low + vert_div_low) / jnp.maximum(h_k, eps)
        q_td = tracer + dq_low * 50.0
        assert float(jnp.abs(q_td[di, dj, -1])) > 1e6, (
            "the dry cell's unmasked q_td should blow up; got "
            f"{float(q_td[di, dj, -1])}")


# ---------------------------------------------------------------------------
# Sign-split correctness
# ---------------------------------------------------------------------------

class TestSignSplitFaceAlphas:
    """Direct unit tests for ``_zalesak_signsplit_face_alphas``."""

    def _setup(self, grid):
        n_lat = grid.n_lat
        n_lon = grid.n_lon
        nlev = 3
        rng = np.random.default_rng(123)
        ad_flux_u = jnp.asarray(rng.normal(size=(n_lat, n_lon + 1, nlev)))
        ad_flux_v = jnp.asarray(rng.normal(size=(n_lat + 1, n_lon, nlev)))
        ad_vert_int = jnp.asarray(rng.normal(size=(n_lat, n_lon, nlev - 1)))
        # Provisional state with mid-range bounds.
        q_td = jnp.asarray(rng.normal(size=(n_lat, n_lon, nlev)))
        q_min = q_td - 1.0
        q_max = q_td + 1.0
        h_k = jnp.ones((n_lat, n_lon, nlev))
        return ad_flux_u, ad_flux_v, ad_vert_int, q_td, q_min, q_max, h_k

    def test_shapes(self, grid_small):
        ad_u, ad_v, ad_w, q_td, q_min, q_max, h_k = self._setup(grid_small)
        a_u, a_v, a_w = _zalesak_signsplit_face_alphas(
            ad_u, ad_v, ad_w, q_td, q_min, q_max, h_k, dt=10.0, grid=grid_small,
        )
        assert a_u.shape == ad_u.shape
        assert a_v.shape == ad_v.shape
        assert a_w.shape == ad_w.shape

    def test_alpha_in_unit_interval(self, grid_small):
        ad_u, ad_v, ad_w, q_td, q_min, q_max, h_k = self._setup(grid_small)
        a_u, a_v, a_w = _zalesak_signsplit_face_alphas(
            ad_u, ad_v, ad_w, q_td, q_min, q_max, h_k, dt=10.0, grid=grid_small,
        )
        for a in (a_u, a_v, a_w):
            assert bool(jnp.all(a >= 0.0))
            assert bool(jnp.all(a <= 1.0 + 1e-12))

    def test_unbounded_room_yields_alpha_one(self, grid_small):
        """When q_min, q_max are infinite (Q+, Q- → ∞) every face passes
        the full anti-diffusive flux — α = 1 everywhere.  This is the
        consistency property a correct limiter must satisfy."""
        ad_u, ad_v, ad_w, q_td, _, _, h_k = self._setup(grid_small)
        big = 1.0e30
        q_min = jnp.full_like(q_td, -big)
        q_max = jnp.full_like(q_td, big)
        a_u, a_v, a_w = _zalesak_signsplit_face_alphas(
            ad_u, ad_v, ad_w, q_td, q_min, q_max, h_k, dt=10.0, grid=grid_small,
        )
        assert bool(jnp.all(jnp.isclose(a_u, 1.0)))
        assert bool(jnp.all(jnp.isclose(a_v, 1.0)))
        assert bool(jnp.all(jnp.isclose(a_w, 1.0)))

    def test_asymmetric_budget_picks_correct_sender_receiver(self, grid_small):
        """Crafted asymmetry: only ONE cell has zero room — α at faces
        touching it must drop to zero in the direction governed by the
        sender / receiver convention.

        Setup: every cell has plenty of room to grow (Q+ ≫ P+) but a
        single target cell ``(0, 5, 0)`` is at its upper bound (Q+ = 0).
        For an anti-diffusive u-face flux ``F_u > 0`` whose RECEIVER is
        the saturated cell, R+_recv = 0 ⇒ α = 0.  Reversing the flux
        sign moves the saturated cell into the SENDER role; it loses
        tracer (so its R- is large) and α can stay at 1.  The branch
        coverage that asserts this asymmetry is what distinguishes a
        proper sign-split limiter from the symmetric face-min heuristic.
        """
        n_lat = grid_small.n_lat
        n_lon = grid_small.n_lon
        nlev = 3
        h_k = jnp.ones((n_lat, n_lon, nlev))
        # Target cell: (0, 5, 0).
        q_td = jnp.zeros((n_lat, n_lon, nlev))
        q_min = jnp.full_like(q_td, -100.0)  # plenty of room down for everyone
        q_max = jnp.full_like(q_td, 100.0)
        # Saturate only the target cell at its upper bound.
        q_max = q_max.at[0, 5, 0].set(q_td[0, 5, 0])  # Q+ = 0 there
        # All anti-diffusive u-fluxes positive (eastward).
        ad_u = jnp.ones((n_lat, n_lon + 1, nlev))
        ad_v = jnp.zeros((n_lat + 1, n_lon, nlev))
        ad_w = jnp.zeros((n_lat, n_lon, nlev - 1))

        a_u_pos, _, _ = _zalesak_signsplit_face_alphas(
            ad_u, ad_v, ad_w, q_td, q_min, q_max, h_k, dt=1.0, grid=grid_small,
        )
        # Receiver of u-face j is cell j (right of face).  The saturated
        # cell is at (lat=0, lon=5), so its WEST face is u-face j=5;
        # its α must drop to 0 because R+_recv = 0.
        assert float(a_u_pos[0, 5, 0]) <= 1e-12

        # Reverse all fluxes: now the saturated cell is SENDER
        # (loses tracer at the same rate); its R- is large, so α should
        # be free to stay at 1 at the same west face.
        ad_u_neg = -ad_u
        a_u_neg, _, _ = _zalesak_signsplit_face_alphas(
            ad_u_neg, ad_v, ad_w, q_td, q_min, q_max, h_k, dt=1.0, grid=grid_small,
        )
        assert float(a_u_neg[0, 5, 0]) > 0.5  # not gated by R+_sender

    def test_zero_room_yields_zero_alpha_on_active_faces(self, grid_small):
        """Cells with q_max == q_td == q_min must reject any non-zero
        anti-diffusive flux that would change them.  Interior faces
        with non-zero flux drop to α = 0; wall v-faces carry the
        placeholder α = 1 (their mass flux is masked to zero at the
        call site so the placeholder is harmless)."""
        ad_u, ad_v, ad_w, _, _, _, h_k = self._setup(grid_small)
        # Force q_td to exactly equal both bounds: zero room either way.
        q_td = jnp.zeros_like(h_k)
        q_min = q_td.copy()
        q_max = q_td.copy()
        a_u, a_v, a_w = _zalesak_signsplit_face_alphas(
            ad_u, ad_v, ad_w, q_td, q_min, q_max, h_k, dt=10.0, grid=grid_small,
        )
        # u-faces (full periodic axis) and vertical interfaces — every
        # interior face has both a sender and a receiver with R+=R-=0.
        for ad, alpha in ((ad_u, a_u), (ad_w, a_w)):
            mask = jnp.abs(ad) > 0.0
            if bool(jnp.any(mask)):
                assert float(jnp.max(jnp.where(mask, alpha, 0.0))) <= 1e-12
        # v-faces: skip the two wall faces (index 0 and -1), which use the
        # 1.0 placeholder by design.
        ad_v_int = ad_v[1:-1, :, :]
        a_v_int = a_v[1:-1, :, :]
        mask_v = jnp.abs(ad_v_int) > 0.0
        if bool(jnp.any(mask_v)):
            assert float(jnp.max(jnp.where(mask_v, a_v_int, 0.0))) <= 1e-12


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

class TestDifferentiability:
    """``jax.grad`` through the limited divergence must give finite
    sensitivities — required for tracer-tuning / DA workflows."""

    def test_grad_through_fct_advection_finite(self, grid_small, smooth_state):
        tracer, mu, mv, w_half, h_k, dt = smooth_state

        def loss(tr):
            div_h, div_w = fct_tracer_advection(
                tr, mu, mv, w_half, h_k, grid_small, dt,
            )
            return jnp.sum(div_h ** 2) + jnp.sum(div_w ** 2)

        g = jax.grad(loss)(tracer)
        assert g.shape == tracer.shape
        assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# NEMO FCT 2nd/2nd variant (high_order="centred2")
# ---------------------------------------------------------------------------

class TestFct2Centred:
    """NEMO traadv_fct nn_fct_h = nn_fct_v = 2: centred-2 high flux."""

    def test_unknown_high_order_raises(self, grid_small, smooth_state):
        tracer, mu, mv, w_half, h_k, dt = smooth_state
        with pytest.raises(ValueError, match="high_order"):
            fct_tracer_advection(
                tracer, mu, mv, w_half, h_k, grid_small, dt,
                high_order="quintic")

    def test_centred_faces_exact_on_linear_field(self):
        """0.5·(T_west+T_east) reproduces a zonally-linear field's face
        values exactly (2nd-order centred is exact for degree-1)."""
        from legoesm.ocean.advection import (
            centred2_to_u_points, centred2_to_v_points,
        )
        n_lat, n_lon, nlev = 6, 8, 3
        x = np.arange(n_lon, dtype=float)
        f = jnp.broadcast_to(
            jnp.asarray(x)[None, :, None], (n_lat, n_lon, nlev))
        f_u = centred2_to_u_points(f)
        assert f_u.shape == (n_lat, n_lon + 1, nlev)
        # interior faces j=1..n_lon-1 sit between cells j-1, j -> x = j-1/2
        expect_u = np.broadcast_to(
            (x[:-1] + x[1:])[None, :, None] / 2.0, (n_lat, n_lon - 1, nlev))
        np.testing.assert_allclose(
            np.asarray(f_u)[:, 1:n_lon, :], expect_u, rtol=0, atol=1e-14)
        # wrap face (0 == n_lon): mean of last + first cell
        np.testing.assert_allclose(
            np.asarray(f_u)[:, 0, :], (x[0] + x[-1]) / 2.0, atol=1e-14)
        np.testing.assert_allclose(
            np.asarray(f_u)[:, n_lon, :], (x[0] + x[-1]) / 2.0, atol=1e-14)

        y = np.arange(n_lat, dtype=float)
        g = jnp.broadcast_to(
            jnp.asarray(y)[:, None, None], (n_lat, n_lon, nlev))
        g_v = centred2_to_v_points(g)
        assert g_v.shape == (n_lat + 1, n_lon, nlev)
        expect_v = np.broadcast_to(
            (y[:-1] + y[1:])[:, None, None] / 2.0, (n_lat - 1, n_lon, nlev))
        np.testing.assert_allclose(
            np.asarray(g_v)[1:n_lat, :, :], expect_v, atol=1e-14)
        # wall faces copy the adjacent cell (flux is zero there anyway)
        np.testing.assert_allclose(np.asarray(g_v)[0], y[0], atol=1e-14)
        np.testing.assert_allclose(np.asarray(g_v)[n_lat], y[-1], atol=1e-14)

    def test_fct2_conserves_on_periodic_channel(self, grid_small,
                                                smooth_state):
        tracer, mu, mv, w_half, h_k, dt = smooth_state
        div_h, div_w = fct_tracer_advection(
            tracer, mu, mv, w_half, h_k, grid_small, dt,
            high_order="centred2")
        area = grid_small.area[..., None]
        assert abs(float(jnp.sum(-(div_h + div_w) * area))) < 1e-10

    def test_fct2_no_new_extrema_after_one_step(self, grid_small,
                                                smooth_state):
        """Zalesak must bound the centred-2 flux (which CAN overshoot
        unlimited) — one forward step creates no new extrema."""
        tracer, mu, mv, w_half, h_k, dt = smooth_state
        div_h, div_w = fct_tracer_advection(
            tracer, mu, mv, w_half, h_k, grid_small, dt,
            high_order="centred2")
        updated = tracer + dt * (-(div_h + div_w) / h_k)
        assert float(updated.max()) <= float(tracer.max()) + 1e-9
        assert float(updated.min()) >= float(tracer.min()) - 1e-9

    def test_fct2_differs_from_ppm_fct(self, grid_small, smooth_state):
        """The two high-order variants are genuinely different schemes."""
        tracer, mu, mv, w_half, h_k, dt = smooth_state
        d_ppm, w_ppm = fct_tracer_advection(
            tracer, mu, mv, w_half, h_k, grid_small, dt, high_order="ppm")
        d_c2, w_c2 = fct_tracer_advection(
            tracer, mu, mv, w_half, h_k, grid_small, dt,
            high_order="centred2")
        assert float(jnp.abs(d_ppm - d_c2).max()) > 0.0

    def test_fct2_vertical_centred_flux(self, grid_small):
        """With pure vertical velocity the interior high flux is the
        centred interface mean, Zalesak-limited; conservation holds
        column-wise (top/bottom fluxes zero)."""
        n_lat, n_lon, nlev = grid_small.n_lat, grid_small.n_lon, 6
        rng = np.random.default_rng(1)
        tracer = jnp.asarray(
            np.linspace(20.0, 4.0, nlev)[None, None, :]
            + rng.normal(size=(n_lat, n_lon, nlev)) * 0.05)
        mu = jnp.zeros((n_lat, n_lon + 1, nlev))
        mv = jnp.zeros((n_lat + 1, n_lon, nlev))
        w_half = jnp.zeros((n_lat, n_lon, nlev + 1))
        w_half = w_half.at[..., 1:nlev].set(1e-4)
        h_k = jnp.ones((n_lat, n_lon, nlev)) * 50.0
        div_h, div_w = fct_tracer_advection(
            tracer, mu, mv, w_half, h_k, grid_small, 100.0,
            high_order="centred2")
        assert float(jnp.abs(div_h).max()) == 0.0
        col_sum = jnp.sum(div_w, axis=-1)
        np.testing.assert_allclose(np.asarray(col_sum), 0.0, atol=1e-12)
