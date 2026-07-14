"""Term-by-term analytic validation for the pseudo-spectral incompressible
LES dycore (``spectral_les_plane``).

LES analogue of ``tests/atmosphere/shallow_water/unit/test_term_by_term_analytic.py``
and ``tests/ocean/unit/test_term_by_term_analytic.py``: each tendency returned
by ``spectral_les_plane.rhs`` is isolated by zeroing the other knobs and
compared to a closed-form analytic answer, plus a time-integrated decay and the
RK3 temporal convergence order.

Why tendency-only for rotation, and which battery entries are SKIPPED.  This
dycore is a wall-modelled ABL-LES core: the surface vertical momentum flux is
ALWAYS replaced by the Monin–Obukhov wall stress ``τ_w = −C_d⟨|u₁|⟩u₁`` (a
nonlinear drag that cannot be switched off via config).  So:

* the **Coriolis** operator is probed at the TENDENCY level on a uniform flow
  (the wall stress then only contaminates the single surface cell ``k=0``;
  interior cells ``k≥1`` see the clean ``±f`` rotation) — exactly the
  tendency-only rationale the shallow-water suite uses to dodge its IGW
  feedback;
* a clean (undamped) **inertial oscillation** / **damped-inertial** TIME
  integration is NOT available — the nonlinear wall drag is always active and is
  not a linear Rayleigh term, so there is no closed form.  Documented skip; the
  Coriolis tendency test covers the rotation operator itself.
* **Burgers self-advection** vanishes under the incompressible projection for a
  divergence-free mode; the faithful analogue — rotational advection conserves
  resolved KE (``⟨u·C⟩≈0``) — is already in ``test_spectral_les_plane.py``.
  Documented skip here to avoid duplication.

What IS verified analytically here:

1. ``TestCoriolisTendencyLES``      — interior ``du/dt=+fV``, ``dv/dt=−fU``.
2. ``TestViscousTendencyLES``       — constant-ν SGS reduces to ``ν∇²u``;
                                       interior ``du/dt=−ν k² u`` and its k²
                                       scaling (mirrors the SW/ocean Laplacian
                                       k²-scaling panel).
3. ``TestViscousDecayLES``          — time-integrated interior amplitude decays
                                       as ``exp(−ν k² t)``.
4. ``TestRK3TemporalConvergenceLES``— global error of that decay ∝ ``dt³``
                                       (SSP-RK3 order).  Spatial accuracy is
                                       SPECTRAL (exponential, not O(dx²)) — the
                                       single-mode exactness test in
                                       ``test_spectral_les_plane.py`` covers it,
                                       so no dx-convergence sweep here.

Run::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \\
        tests/unit/test_spectral_les_analytic.py -v
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _enable_x64():
    """Analytic comparisons require float64."""
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _inviscid_cfg(nx=32, ny=32, nz=24, Lx=320.0, Ly=320.0, Lz=240.0, **kw):
    """All SGS + filtering knobs OFF (inviscid, unfiltered): the baseline for
    isolating a single tendency.  ``nu_floor`` is the ONLY viscosity, so setting
    it makes ``ν_t`` a known CONSTANT (eddy_viscosity = 0 + nu_floor)."""
    base = dict(
        nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz,
        c_s=0.0, c_vreman=0.0, smagorinsky_dynamic=False, nu_floor=0.0,
        buoyancy=False, wall_damping=False, spectral_filter=False, dealias=True,
        time_scheme="rk3",
    )
    base.update(kw)
    return sl.SpectralLESConfig(**base)


def _state(u, v, w):
    return sl.SpectralLESState(
        u=u, v=v, w=w, theta=None, rhs_theta_prev=None,
        rhs_u_prev=jnp.zeros_like(u), rhs_v_prev=jnp.zeros_like(v),
        rhs_w_prev=jnp.zeros_like(w))


def _y_nodes(g):
    """Physical y node positions (rfft convention: samples at j·dy, j=0..ny-1)."""
    return (jnp.arange(g.cfg.ny) * g.dy).astype(jnp.float64)


# ---------------------------------------------------------------------------
# 1. Coriolis term in isolation
# ---------------------------------------------------------------------------
class TestCoriolisTendencyLES:
    """Uniform flow + f-plane, no SGS, no buoyancy ⇒ the only momentum tendency
    is rotation: ``du/dt=+f v``, ``dv/dt=−f u``.  Checked at interior cells
    (k≥1); the surface cell k=0 carries the always-on wall stress."""

    F = 1.0e-4

    def _tend(self, U0, V0):
        g = sl.make_grid(_inviscid_cfg())
        ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
        u = jnp.full((ny, nx, nz), U0)
        v = jnp.full((ny, nx, nz), V0)
        w = jnp.zeros((ny, nx, nz + 1))
        Ru, Rv, Rw, _us, _rth, _rtr = sl.rhs(u, v, w, g, u_geo=(0.0, 0.0), f_cor=self.F)
        return np.asarray(Ru), np.asarray(Rv)

    def test_du_dt_equals_plus_f_v(self):
        for V0 in (-0.2, -0.05, 0.1, 0.2):
            Ru, _ = self._tend(0.0, V0)
            assert np.allclose(Ru[..., 1:], self.F * V0, atol=1e-12)

    def test_dv_dt_equals_minus_f_u(self):
        for U0 in (-0.2, -0.05, 0.1, 0.2):
            _, Rv = self._tend(U0, 0.0)
            assert np.allclose(Rv[..., 1:], -self.F * U0, atol=1e-12)

    def test_no_self_tendency_on_aligned_component(self):
        """du/dt independent of u (pure cross-term); dv/dt independent of v."""
        Ru_a, _ = self._tend(0.1, 0.0)
        Ru_b, _ = self._tend(0.2, 0.0)
        assert np.allclose(Ru_a[..., 1:], 0.0, atol=1e-12)
        assert np.allclose(Ru_b[..., 1:], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Constant-ν SGS Laplacian tendency
# ---------------------------------------------------------------------------
class TestViscousTendencyLES:
    """A divergence-free, z-independent shear mode ``u=U₀ sin(k_y y)``, ``v=w=0``
    with a CONSTANT eddy viscosity ``ν=nu_floor`` (all other SGS off).  The SGS
    force ``∂_j(2ν S_ij)`` reduces to the horizontal Laplacian ``ν ∂²u/∂y²``;
    advection vanishes (``∂u/∂x=0``); the always-on wall stress touches only
    k=0.  So at interior cells ``du/dt = −ν k_y² u``."""

    NU = 0.5
    U0 = 0.3

    def _setup(self, m):
        g = sl.make_grid(_inviscid_cfg(nu_floor=self.NU))
        ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
        ky = 2.0 * np.pi * m / g.cfg.Ly
        y = _y_nodes(g)
        u = self.U0 * jnp.sin(ky * y)[:, None, None] * jnp.ones((ny, nx, nz))
        v = jnp.zeros((ny, nx, nz))
        w = jnp.zeros((ny, nx, nz + 1))
        Ru, _Rv, _Rw, _us, _rth, _rtr = sl.rhs(u, v, w, g, u_geo=(0.0, 0.0), f_cor=0.0)
        return g, np.asarray(u), np.asarray(Ru), ky

    def test_laplacian_shape(self):
        g, u, Ru, ky = self._setup(m=2)
        kmid = g.cfg.nz // 2
        expected = -self.NU * ky ** 2 * u[..., kmid]
        assert np.allclose(Ru[..., kmid], expected, rtol=1e-9, atol=1e-12)

    def test_k_squared_scaling(self):
        """Peak |du/dt| ∝ k² (Laplacian).  Slope of log|du/dt| vs log k ≈ 2."""
        ks, amps = [], []
        for m in (1, 2, 3, 4):
            g, _u, Ru, ky = self._setup(m)
            kmid = g.cfg.nz // 2
            ks.append(ky)
            amps.append(np.max(np.abs(Ru[..., kmid])))
        slope = np.polyfit(np.log(ks), np.log(amps), 1)[0]
        assert abs(slope - 2.0) < 0.02


# ---------------------------------------------------------------------------
# 3. Time-integrated viscous decay
# ---------------------------------------------------------------------------
class TestViscousDecayLES:
    """Integrate the constant-ν shear mode: an interior cell's amplitude must
    decay as ``U₀ exp(−ν k_y² t)`` (the wall stress only damps k=0)."""

    NU = 8.0     # with m=4 below ⇒ rate·T≈0.5 (≈40 % decay): SUBSTANTIAL, so a
    U0 = 0.2     # wrong diffusion coefficient cannot hide under the tolerance.

    def test_interior_amplitude_decay(self):
        m = 4
        g = sl.make_grid(_inviscid_cfg(nu_floor=self.NU))
        ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
        ky = 2.0 * np.pi * m / g.cfg.Ly
        y = _y_nodes(g)
        shape = jnp.sin(ky * y)[:, None, None] * jnp.ones((ny, nx, nz))
        u = self.U0 * shape
        v = jnp.zeros((ny, nx, nz))
        w = jnp.zeros((ny, nx, nz + 1))
        st = _state(u, v, w)
        dt = 0.05
        n = 200
        rate = self.NU * ky ** 2
        first = True
        # Track the effective decay rate (fit) as well as the endpoint, so the
        # WHOLE exp(−ν k² t) curve is verified, not just one time.
        ts, amps = [], []
        for i in range(n):
            st, _ = sl.step(st, g=g, dt=dt, u_geo=(0.0, 0.0), f_cor=0.0,
                            first=first)
            first = False
            if (i + 1) % 40 == 0:
                kmid = nz // 2
                a = 2.0 * np.mean(np.asarray(st.u)[:, 0, kmid]
                                  * np.asarray(jnp.sin(ky * y)))
                ts.append((i + 1) * dt); amps.append(a)
        ts, amps = np.array(ts), np.array(amps)
        t = n * dt
        amp_ana = self.U0 * np.exp(-rate * t)
        # Endpoint amplitude matches the analytic decay to 0.3 %.
        assert abs(amps[-1] - amp_ana) / amp_ana < 3e-3
        # Fitted decay rate matches ν k² to 0.5 %.
        rate_fit = -np.polyfit(ts, np.log(amps), 1)[0]
        assert abs(rate_fit - rate) / rate < 5e-3


# ---------------------------------------------------------------------------
# 4. SSP-RK3 temporal convergence order
# ---------------------------------------------------------------------------
class TestRK3TemporalConvergenceLES:
    """With SPECTRAL (exponentially accurate) space, the viscous-decay error to a
    fixed time T is dominated by the SSP-RK3 time integrator and must shrink as
    ``dt³``.  Spatial dx-convergence is NOT swept: spectral accuracy is
    exponential (covered by the single-mode exactness test elsewhere)."""

    NU = 8.0     # chosen with k (below) so rate·T≈O(1): the decay is substantial,
    U0 = 0.2     # so the global error sits in the dt³ regime, above the x64 floor.

    def _final_amp_error(self, dt, T, g, ky, y, ny, nx, nz):
        shape = jnp.sin(ky * y)[:, None, None] * jnp.ones((ny, nx, nz))
        st = _state(self.U0 * shape, jnp.zeros((ny, nx, nz)),
                    jnp.zeros((ny, nx, nz + 1)))
        n = int(round(T / dt))
        first = True
        for _ in range(n):
            st, _ = sl.step(st, g=g, dt=dt, u_geo=(0.0, 0.0), f_cor=0.0,
                            first=first)
            first = False
        kmid = nz // 2
        amp = 2.0 * np.mean(np.asarray(st.u)[:, 0, kmid]
                            * np.asarray(jnp.sin(ky * y)))
        return abs(amp - self.U0 * np.exp(-self.NU * ky ** 2 * T))

    def test_third_order_in_time(self):
        m = 6
        g = sl.make_grid(_inviscid_cfg(nu_floor=self.NU))
        ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
        ky = 2.0 * np.pi * m / g.cfg.Ly       # rate = ν k² ≈ 0.11 /s
        y = _y_nodes(g)
        T = 10.0                              # rate·T ≈ 1.1 ⇒ substantial decay
        dts = [0.2, 0.1, 0.05]
        errs = [self._final_amp_error(dt, T, g, ky, y, ny, nx, nz) for dt in dts]
        # Halving dt should cut the error by ~2³=8 (RK3). Use the end ratio;
        # allow generous tolerance (round-off floor at the smallest dt).
        slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        assert slope > 2.5, f"RK3 temporal order {slope:.2f} < 2.5"


# ---------------------------------------------------------------------------
# Documented skips — battery entries with no clean closed form on this core
# ---------------------------------------------------------------------------
@pytest.mark.skip(reason="Wall stress τ_w=−C_d⟨|u₁|⟩u₁ is always active and "
                         "nonlinear (not a linear Rayleigh drag) — no closed "
                         "form for an undamped/linearly-damped inertial "
                         "oscillation. Coriolis rotation is covered at the "
                         "tendency level (TestCoriolisTendencyLES).")
def test_inertial_oscillation_time_integration():
    pass


@pytest.mark.skip(reason="Incompressible projection removes self-advection of a "
                         "divergence-free mode (Burgers steepening). The "
                         "faithful analogue — rotational advection conserves "
                         "resolved KE ⟨u·C⟩≈0 — is in test_spectral_les_plane.py.")
def test_burgers_advection():
    pass
