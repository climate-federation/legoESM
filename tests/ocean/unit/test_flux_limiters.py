"""Tests for shared TVD flux limiters and the tracer_advection="superbee" wiring.

The Sweby/superbee limiter and the Van Leer limiter are centralized in
``legoesm.ocean.dynamics._flux_limiters``. Veros's default tracer
advection (``enable_superbee_advection=True``) corresponds to legoESM's
``tracer_advection="superbee"``; ``"tvd"`` continues to mean Van Leer.

These tests cover:

1. Mathematical properties of both limiters (TVD region, key values).
2. ``resolve_tvd_limiter`` dispatch + error raising on unknown names.
3. Parameterized TVD reconstruction functions on lat-lon C-grid and MPAS
   yield identical results for ``"tvd"`` (i.e. the refactor is a no-op)
   and *different* (well-defined) results for ``"superbee"``.
4. Model-level dispatch raises ``ValueError`` on unknown literals
   (closes the previous silent-fall-back to upwind / TVD path).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.dynamics._flux_limiters import (
    resolve_tvd_limiter,
    sweby_limiter,
    van_leer_limiter,
)


# ---------------------------------------------------------------------------
# 1. Limiter math
# ---------------------------------------------------------------------------

class TestVanLeerLimiter:
    """Properties of the Van Leer limiter (the existing default for `"tvd"`)."""

    def test_zero_at_non_positive_r(self):
        r = jnp.array([-2.0, -1.0, -0.5, 0.0])
        assert jnp.all(van_leer_limiter(r) == 0.0)

    def test_one_at_r_one(self):
        assert float(van_leer_limiter(jnp.array(1.0))) == 1.0

    def test_asymptote_two(self):
        # Van Leer: psi(r) = 2r / (1 + r) → 2 from below as r → ∞.
        # Asymptote is slow: psi(1e4) ≈ 1.9998, psi(1e6) ≈ 1.999998.
        r = jnp.array([1e4, 1e6, 1e9])
        result = van_leer_limiter(r)
        assert jnp.all(result < 2.0)
        assert jnp.all(result > 2.0 - 1e-3)

    def test_smooth_differentiable(self):
        r = jnp.linspace(0.01, 5.0, 50)
        grad_fn = jax.vmap(jax.grad(lambda r_: van_leer_limiter(r_)))
        g = grad_fn(r)
        assert jnp.all(jnp.isfinite(g))


class TestSwebyLimiterMath:
    """Sweby/superbee mathematical properties (mirroring the existing
    DST-3 test class but pointing at the shared module).
    """

    def test_zero_at_negative_r(self):
        r = jnp.array([-2.0, -1.0, -0.5, -0.01])
        assert jnp.all(sweby_limiter(r) == 0.0)

    def test_one_at_r_one(self):
        assert float(sweby_limiter(jnp.array(1.0))) == 1.0

    def test_two_at_large_r(self):
        r = jnp.array([3.0, 5.0, 100.0])
        assert jnp.allclose(sweby_limiter(r), 2.0)

    def test_known_values(self):
        # r < 0.5: psi = 2r
        assert jnp.isclose(sweby_limiter(jnp.array(0.25)), 0.5)
        # 0.5 <= r <= 1: psi = 1
        assert jnp.isclose(sweby_limiter(jnp.array(0.75)), 1.0)
        # 1 <= r <= 2: psi = r
        assert jnp.isclose(sweby_limiter(jnp.array(1.5)), 1.5)
        # r >= 2: psi = 2
        assert jnp.isclose(sweby_limiter(jnp.array(2.5)), 2.0)

    def test_tvd_region(self):
        r = jnp.linspace(0.0, 4.0, 100)
        result = sweby_limiter(r)
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= jnp.minimum(2.0 * r, 2.0) + 1e-10)

    def test_aggressive_compared_to_van_leer(self):
        """Sweby >= Van Leer on r in (0, 1), which is why Veros prefers
        it for sharp-front tracer advection."""
        r = jnp.linspace(0.1, 0.9, 20)
        assert jnp.all(sweby_limiter(r) >= van_leer_limiter(r) - 1e-12)


class TestResolveTvdLimiter:
    """``resolve_tvd_limiter`` is the central dispatcher used by the
    lat-lon C-grid and MPAS model dispatch sites."""

    def test_tvd_returns_van_leer(self):
        assert resolve_tvd_limiter("tvd") is van_leer_limiter

    def test_superbee_returns_sweby(self):
        assert resolve_tvd_limiter("superbee") is sweby_limiter

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown TVD literal"):
            resolve_tvd_limiter("minmod")

    def test_unknown_lists_valid_options(self):
        with pytest.raises(ValueError, match="superbee"):
            resolve_tvd_limiter("typo")


# ---------------------------------------------------------------------------
# 2. Parameterized TVD reconstruction on lat-lon C-grid
# ---------------------------------------------------------------------------

class TestLatLonTvdParameterization:
    """``_tvd_to_u_points`` / ``_tvd_to_v_points`` take a ``limiter_fn``
    kwarg; default behavior matches the pre-refactor Van Leer path,
    and Sweby produces a different (more aggressive) reconstruction
    in the presence of a sharp gradient."""

    def _smooth_gradient_state(self):
        # (n_lat, n_lon, nlev) with a smooth zonal gradient that lands
        # in r ∈ (0.5, 1) where Sweby psi=1 but Van Leer psi<1 — the
        # interval where the two limiters disagree.
        # Use f(j) = 1 + j (linear in j) → r = 1 + eps from the formula
        # so we add a curvature so r lands in (0.5, 1).
        n_lat, n_lon, nlev = 6, 12, 4
        x = jnp.arange(n_lon, dtype=jnp.float64)
        # f(j) = j^1.4 gives non-uniform deltas with r in roughly (0.5, 1)
        # depending on the cell index.
        f_2d = x ** 1.4
        f = jnp.broadcast_to(f_2d[None, :, None], (n_lat, n_lon, nlev))
        # Eastward mass flux (positive) at every u-face.
        mass_flux_u = jnp.full((n_lat, n_lon + 1, nlev), 0.5)
        return jnp.asarray(f), mass_flux_u

    def test_default_matches_van_leer(self):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _tvd_to_u_points,
        )
        f, mass_flux_u = self._smooth_gradient_state()
        default = _tvd_to_u_points(f, mass_flux_u)
        van_leer = _tvd_to_u_points(f, mass_flux_u, limiter_fn=van_leer_limiter)
        assert jnp.allclose(default, van_leer)

    def test_sweby_differs_from_van_leer_on_smooth_gradient(self):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _tvd_to_u_points,
        )
        f, mass_flux_u = self._smooth_gradient_state()
        van_leer = _tvd_to_u_points(f, mass_flux_u, limiter_fn=van_leer_limiter)
        sweby = _tvd_to_u_points(f, mass_flux_u, limiter_fn=sweby_limiter)
        # On a smooth gradient with r ∈ (0.5, 1) Sweby gives psi=1 (max
        # anti-diffusion within the TVD region) while Van Leer gives
        # psi < 1 — so Sweby's face values are systematically further
        # from the donor cell value.
        assert float(jnp.max(jnp.abs(sweby - van_leer))) > 1e-3


# ---------------------------------------------------------------------------
# 3. Parameterized TVD reconstruction on MPAS edges
# ---------------------------------------------------------------------------

class TestMpasTvdParameterization:
    """``tvd_tracer_to_edges`` accepts a ``limiter_fn`` kwarg."""

    def _minimal_mesh_state(self):
        # Three cells in a line: 0 -- 1 -- 2.  Two edges, each with one
        # neighbor on either side. Sharp jump between cell 1 and cell 2.
        from types import SimpleNamespace

        nCells, nEdges, nlev = 3, 2, 1
        cellsOnEdge = jnp.asarray([[0, 1], [1, 2]]).T  # shape (2, nEdges)
        # Build a minimal namespace that walks like a VoronoiMesh enough
        # for the function we're calling (which only touches
        # ``mesh.cellsOnEdge``).
        mesh = SimpleNamespace(cellsOnEdge=cellsOnEdge)
        tr = jnp.asarray([[1.0], [1.0], [5.0]])  # (nCells, nlev)
        mass_flux = jnp.asarray([[0.5], [0.5]])  # (nEdges, nlev), positive

        # upup_pos: for positive flow (c1 -> c2), upwind-of-upwind of c1.
        #   edge 0 (c1=0, c2=1): no real upup => point to donor c1=0
        #   edge 1 (c1=1, c2=2): upup is c1=0
        upup_pos = jnp.asarray([0, 0])
        # upup_neg: for negative flow, upup of c2. (No negative flow in
        # this test, but the function still references it.)
        upup_neg = jnp.asarray([1, 2])
        return mesh, tr, mass_flux, upup_pos, upup_neg

    def test_default_matches_van_leer(self):
        from legoesm.ocean.dynamics.advection_mpas import tvd_tracer_to_edges
        mesh, tr, mass_flux, upup_pos, upup_neg = self._minimal_mesh_state()
        default = tvd_tracer_to_edges(tr, mass_flux, mesh, upup_pos, upup_neg)
        van_leer = tvd_tracer_to_edges(
            tr, mass_flux, mesh, upup_pos, upup_neg, limiter_fn=van_leer_limiter
        )
        assert jnp.allclose(default, van_leer)

    def test_sweby_differs_from_van_leer(self):
        from legoesm.ocean.dynamics.advection_mpas import tvd_tracer_to_edges
        mesh, tr, mass_flux, upup_pos, upup_neg = self._minimal_mesh_state()
        van_leer = tvd_tracer_to_edges(
            tr, mass_flux, mesh, upup_pos, upup_neg, limiter_fn=van_leer_limiter
        )
        sweby = tvd_tracer_to_edges(
            tr, mass_flux, mesh, upup_pos, upup_neg, limiter_fn=sweby_limiter
        )
        # Edge 1: r = (tr[1] - tr[0]) / (tr[2] - tr[1]) = (1-1)/(5-1) = 0
        # so the limiter argument is 0 — no anti-diffusion for either
        # limiter at this edge. So we exercise the difference via a
        # smooth-gradient state instead.
        tr_smooth = jnp.asarray([[1.0], [2.0], [3.5]])
        van_leer_s = tvd_tracer_to_edges(
            tr_smooth, mass_flux, mesh, upup_pos, upup_neg,
            limiter_fn=van_leer_limiter,
        )
        sweby_s = tvd_tracer_to_edges(
            tr_smooth, mass_flux, mesh, upup_pos, upup_neg,
            limiter_fn=sweby_limiter,
        )
        # Edge 1: r = (2-1)/(3.5-2) = 0.667, in (0.5, 1] → Sweby psi = 1,
        # Van Leer psi = 0.8 → Sweby reconstruction is more anti-diffusive.
        assert float(jnp.max(jnp.abs(sweby_s - van_leer_s))) > 1e-3


# ---------------------------------------------------------------------------
# 4. Model-level dispatch raises on unknown literal
# ---------------------------------------------------------------------------

class TestDispatchRaisesOnUnknown:
    """Closing the previous silent-fallback to upwind on the lat-lon
    C-grid horizontal-flux dispatch (and to upup=None on MPAS)."""

    def test_latlon_raises_on_unknown(self):
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            _compute_advection_flux_div,
        )
        # Minimal arrays — values do not matter because the dispatch
        # raises before any computation.
        zero = jnp.zeros((2, 2, 1))
        zero_u = jnp.zeros((2, 3, 1))
        zero_v = jnp.zeros((3, 2, 1))
        zero_w = jnp.zeros((2, 2, 2))

        class _FakeGrid:
            lat = jnp.zeros(2)
            lon = jnp.zeros(2)
            dxT = jnp.ones((2, 2))
            dyT = jnp.ones((2, 2))

        with pytest.raises(ValueError, match="Unknown tracer_advection literal"):
            _compute_advection_flux_div(
                tr=zero,
                mass_flux_u=zero_u,
                mass_flux_v=zero_v,
                w_baro=zero_w,
                h_k_old=zero,
                h_u_old=zero_u,
                h_v_old=zero_v,
                grid=_FakeGrid(),
                dt=300.0,
                tracer_advection="not_a_real_scheme",
            )
