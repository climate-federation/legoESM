"""Term-by-term analytic validation for the plane non-hydrostatic compressible
Euler (CRM) dycore.

CRM analogue of ``tests/atmosphere/shallow_water/unit/test_term_by_term_analytic.py``.
Each slow tendency returned by ``plane_compressible_euler_slow_tendencies`` is
isolated by zeroing the other config knobs and compared to a closed-form
analytic answer.  Tendency-only (not time integration): the split-explicit
acoustic substep couples w/θ'/ρ' through the fast pressure mode, so a
time-integrated single-term ODE has no clean closed form — the shallow-water
suite uses the same tendency-only rationale to dodge its IGW feedback.

Three regimes (mapping onto the SW/ocean battery):

1. ``TestCoriolisTendencyCRM``    — Coriolis term in isolation:
                                    ``du/dt=+f v``, ``dv/dt=−f u`` (uniform flow,
                                    f-plane).
2. ``TestBiharmonicHyperdiffCRM`` — isolated horizontal hyperdiffusion.  NOTE:
                                    this dycore's horizontal diffusion is
                                    BIHARMONIC ∇⁴ (not the SW/ocean Laplacian
                                    ∇²), so the single-mode tendency scales as
                                    ``k⁴`` (slope 4), not ``k²``.
3. ``TestAdvectionTendencyCRM``   — scalar advection ``dθ'/dt = −U ∂θ'/∂x``
                                    (uniform wind, centred scheme), the
                                    compressible analogue of Burgers-advection.

Skipped (no clean closed form on this core): horizontal pressure-gradient and
ρ' continuity are fused with buoyancy / the acoustic solve; documented below.

Run::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \\
        tests/unit/test_compressible_euler_plane_analytic.py -v
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler import CompressibleEulerConfig
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric,
    make_rest_state,
    plane_compressible_euler_slow_tendencies as slow_tend,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _cfg(**overrides):
    """Everything dissipative/rotational OFF — the baseline for isolating one
    term.  Centred advection (the only scheme whose tendency equals the exact
    derivative; upwind1/weno carry numerical diffusion)."""
    base = dict(
        sponge_coeff=0.0, hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0, semi_implicit_acoustic=False, use_coriolis=False,
        moist_buoyancy=False, smagorinsky_cs=0.0,
        horizontal_advection_scheme="centered",
        horizontal_momentum_advection_scheme="centered",
        vertical_theta_diffusion=0.0,
    )
    base.update(overrides)
    return CompressibleEulerConfig(**base)


def _build(nx=32, ny=32, nlev=8, dx=1.0e3, dy=1.0e3, H=20.0e3, f0=0.0, **cfgkw):
    # f0>0 ⇒ f-plane (populates grid.f_y); else "none" leaves f_y=0.
    cmode = "f_plane" if f0 != 0.0 else "none"
    grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dy,
                             coriolis_mode=cmode, f0=f0, dtype=jnp.float64)
    hc = create_height_coordinate(grid.nlev, H=H)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = _cfg(**cfgkw)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    return grid, hc, tm, cfg, state


def _set(state, **fields):
    """state._replace with Field.replace(data=...) for each named prognostic."""
    repl = {k: getattr(state, k).replace(data=v) for k, v in fields.items()}
    return state._replace(**repl)


# ---------------------------------------------------------------------------
# 1. Coriolis term in isolation
# ---------------------------------------------------------------------------
class TestCoriolisTendencyCRM:
    """Uniform flow on an f-plane, rest thermodynamics ⇒ the only momentum
    tendency is rotation.  Uniform ⇒ no advection/PG, clean at every cell."""

    F = 1.0e-4

    def _tend(self, U0, V0):
        grid, hc, tm, cfg, st = _build(f0=self.F, use_coriolis=True)
        sh = st.u.data.shape
        st = _set(st, u=jnp.full(sh, U0), v=jnp.full(sh, V0))
        t = slow_tend(st, grid, hc, tm, cfg)
        return np.asarray(t.du_dt.data), np.asarray(t.dv_dt.data)

    def test_du_dt_equals_plus_f_v(self):
        for V0 in (-0.2, -0.05, 0.1, 0.2):
            du, _ = self._tend(0.0, V0)
            assert np.allclose(du, self.F * V0, atol=1e-12)

    def test_dv_dt_equals_minus_f_u(self):
        for U0 in (-0.2, -0.05, 0.1, 0.2):
            _, dv = self._tend(U0, 0.0)
            assert np.allclose(dv, -self.F * U0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Biharmonic horizontal hyperdiffusion (k⁴ law)
# ---------------------------------------------------------------------------
class TestBiharmonicHyperdiffCRM:
    """A transverse shear mode ``u(y)=U₀ cos(k_y y)`` (so self-advection
    ∂u/∂x=0) with only ``hyperdiff_coeff`` active.  The biharmonic tendency is
    ``du/dt = −K₄ ∂⁴u/∂y⁴ = −K₄ k_y⁴ u`` ⇒ peak |du/dt| ∝ k⁴ (slope 4)."""

    K4 = 1.0e8     # [m⁴/s]
    U0 = 1.0

    def _peak(self, m):
        grid, hc, tm, cfg, st = _build(hyperdiff_coeff=self.K4)
        ny, nx, nlev = st.u.data.shape
        ky = 2.0 * np.pi * m / (ny * grid.dy)
        yc = jnp.asarray(grid.yc, jnp.float64)
        u = self.U0 * jnp.cos(ky * yc)[:, None, None] * jnp.ones((ny, nx, nlev))
        st = _set(st, u=u)
        t = slow_tend(st, grid, hc, tm, cfg)
        return ky, float(np.max(np.abs(np.asarray(t.du_dt.data))))

    def test_sign_and_proportionality(self):
        """du/dt = −K₄ k⁴ u: opposite sign to u, damping the mode."""
        grid, hc, tm, cfg, st = _build(hyperdiff_coeff=self.K4)
        ny, nx, nlev = st.u.data.shape
        ky = 2.0 * np.pi * 2 / (ny * grid.dy)
        yc = jnp.asarray(grid.yc, jnp.float64)
        u = self.U0 * jnp.cos(ky * yc)[:, None, None] * jnp.ones((ny, nx, nlev))
        t = slow_tend(_set(st, u=u), grid, hc, tm, cfg)
        du = np.asarray(t.du_dt.data)
        assert np.max(np.abs(du)) > 1e-9                    # non-vacuous (du≠0)
        assert np.all(du * np.asarray(u) <= 1e-30)          # damping (opposite sign)

    def test_k_fourth_scaling(self):
        ks, amps = [], []
        for m in (1, 2, 3, 4):
            ky, a = self._peak(m)
            ks.append(ky); amps.append(a)
        slope = np.polyfit(np.log(ks), np.log(amps), 1)[0]
        # Discrete 5-point biharmonic stencil ⇒ slope slightly below 4 at higher
        # m; low modes keep it within 0.2 of the continuous k⁴ exponent.
        assert abs(slope - 4.0) < 0.2, f"biharmonic exponent {slope:.2f} ≠ 4"


# ---------------------------------------------------------------------------
# 3. Scalar advection tendency (Burgers analogue)
# ---------------------------------------------------------------------------
class TestAdvectionTendencyCRM:
    """Uniform wind ``u=U`` advecting a scalar mode ``θ'(x)=θ₀ sin(k_x x)``;
    centred scheme ⇒ ``dθ'/dt = −U ∂θ'/∂x = −U θ₀ k_x cos(k_x x)`` (up to the
    centred-difference factor ``sin(k_x Δx)/(k_x Δx)``, →1 for a resolved mode)."""

    U = 5.0
    TH0 = 2.0

    def test_minus_u_dtheta_dx(self):
        m = 2
        grid, hc, tm, cfg, st = _build()
        ny, nx, nlev = st.u.data.shape
        kx = 2.0 * np.pi * m / (nx * grid.dx)
        xc = jnp.asarray(grid.xc, jnp.float64)
        theta_p = self.TH0 * jnp.sin(kx * xc)[None, :, None] * jnp.ones((ny, nx, nlev))
        st = _set(st, u=jnp.full((ny, nx, nlev), self.U), theta_prime=theta_p)
        t = slow_tend(st, grid, hc, tm, cfg)
        dthp = np.asarray(t.dtheta_prime_dt.data)
        # Discrete centred 2nd-order x-derivative factor.
        fac = np.sin(kx * grid.dx) / (kx * grid.dx)
        expected = -self.U * self.TH0 * kx * fac * np.cos(kx * np.asarray(xc))
        expected = np.broadcast_to(expected[None, :, None], dthp.shape)
        assert np.allclose(dthp, expected, rtol=2e-2, atol=1e-9)


# ---------------------------------------------------------------------------
# Documented skips — fused terms with no clean closed form
# ---------------------------------------------------------------------------
@pytest.mark.skip(reason="Horizontal pressure gradient is coupled to buoyancy "
                         "and the vertical Exner solve inside the acoustic "
                         "substep; not separable as a standalone tendency.")
def test_pressure_gradient_isolated():
    pass


@pytest.mark.skip(reason="ρ' continuity is the mass-flux divergence of u; with "
                         "u≡0 it is identically zero and with u≠0 it cannot be "
                         "separated from momentum advection. No isolated form.")
def test_continuity_isolated():
    pass
