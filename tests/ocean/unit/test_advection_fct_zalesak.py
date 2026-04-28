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
