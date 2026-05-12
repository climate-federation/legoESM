"""Regression tests for the AD-safe ``safe_divide`` helper and its call sites.

The bug being guarded against: ``a / jnp.clip(b, tiny, None)`` is
forward-safe but not AD-safe — the VJP of a divide contains a
``-a / b**2`` term that overflows even at float64 when ``b`` is at the
clip floor.  Under ``jax.value_and_grad`` this propagates NaN/Inf
gradients to every upstream traced parameter.  The double-``where`` /
mask-before-divide idiom in
``legoesm.atmosphere.physics._shared.safe_divide`` removes the bad
branch from the backward pass entirely.

Each test below constructs a state in which the denominator of the
patched divide is at or near zero and asserts
``jnp.all(jnp.isfinite(jax.grad(fn)(x)))``.  Under the previous
``clip + divide`` implementation these would all fail.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics._shared import safe_divide


# ---------------------------------------------------------------------------
# 1. safe_divide unit tests
# ---------------------------------------------------------------------------

class TestSafeDivideHelper:

    def test_forward_matches_naive_divide_above_eps(self):
        """When denominator > eps everywhere, output equals a / b bitwise."""
        a = jnp.array([1.0, -2.5, 0.7, 3.0])
        b = jnp.array([2.0, 4.0, 0.5, 1.0])
        out = safe_divide(a, b, eps=1e-10)
        np.testing.assert_array_equal(out, a / b)

    def test_forward_uses_fill_at_zero_denominator(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([0.0, 0.0])
        out = safe_divide(a, b, eps=1e-10, fill=0.0)
        np.testing.assert_array_equal(out, jnp.zeros(2))

    def test_grad_finite_at_zero_denominator(self):
        """The signature failure mode: grad through a divide at b=0
        gives NaN/Inf under the naive ``/jnp.clip(b, tiny)`` pattern.
        ``safe_divide`` must return a finite gradient for both
        operands."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.zeros(3)

        def loss(b_in):
            return jnp.sum(safe_divide(a, b_in, eps=1e-10) ** 2)

        grad_b = jax.grad(loss)(b)
        assert jnp.all(jnp.isfinite(grad_b))

        def loss_a(a_in):
            return jnp.sum(safe_divide(a_in, b, eps=1e-10) ** 2)

        grad_a = jax.grad(loss_a)(a)
        assert jnp.all(jnp.isfinite(grad_a))

    def test_grad_matches_naive_divide_above_eps(self):
        """When denominator stays comfortably above eps, the gradient
        is identical to the naive divide — the helper is a strict
        improvement, not a numerical change."""
        a = jnp.array([1.0, -2.0, 3.0])
        b = jnp.array([10.0, 20.0, 5.0])

        def loss_safe(b_in):
            return jnp.sum(safe_divide(a, b_in, eps=1e-10))

        def loss_naive(b_in):
            return jnp.sum(a / b_in)

        np.testing.assert_allclose(
            jax.grad(loss_safe)(b), jax.grad(loss_naive)(b), rtol=1e-12,
        )

    def test_scalar_numerator(self):
        """Numerator may be a Python float — the helper must not
        require it to expose ``.dtype``."""
        b = jnp.array([0.0, 0.5, 0.0])
        out = safe_divide(2.0 * math.pi, b, eps=1e-10)
        assert jnp.isfinite(out).all()


# ---------------------------------------------------------------------------
# Helpers for building minimal column states
# ---------------------------------------------------------------------------

def _make_pressure_columns(ncol: int, nlev: int, p_s: float = 1.0e5):
    """Build full- and half-level pressure on a uniform sigma grid."""
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_full = jnp.broadcast_to(p_s * sigma_full[None, :], (ncol, nlev))
    p_half = jnp.broadcast_to(p_s * sigma_half[None, :], (ncol, nlev + 1))
    return p_full, p_half


# ---------------------------------------------------------------------------
# 2. Convection sites
# ---------------------------------------------------------------------------

class TestConvectionColumnRescaleGrad:
    """sbm.py / dca.py / kuo.py share the same column mass-conservation
    rescale ``local_cond * (col_net_drying / col_local_cond)``.  When
    no level condenses (dry stable column), ``col_local_cond = 0`` —
    exactly the regime that NaN'd under the prior ``clip + divide``.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        # Dry stable column: T isothermal, q_v=0 → no condensation
        # candidate at any level → col_local_cond is identically zero.
        self.ncol, self.nlev = 2, 6
        self.T = 280.0 * jnp.ones((self.ncol, self.nlev))
        self.q_v = jnp.zeros((self.ncol, self.nlev))
        self.p_full, self.p_half = _make_pressure_columns(
            self.ncol, self.nlev,
        )
        self.dt = 300.0

    def test_sbm_grad_finite_at_zero_col_cond(self):
        from legoesm.atmosphere.physics.convection.sbm import sbm_convection
        from legoesm.atmosphere.physics.convection.config import SBMConfig

        config = SBMConfig()

        def loss(qv_in):
            out = sbm_convection(
                self.T, qv_in, self.p_full, self.p_half, self.dt, config,
            )
            return jnp.sum(out.dq_c_conv_dt ** 2)

        grad = jax.grad(loss)(self.q_v)
        assert jnp.all(jnp.isfinite(grad)), (
            "SBM gradient at zero col_local_cond — safe_divide guard regressed."
        )

    def test_dca_grad_finite_at_zero_col_cond(self):
        from legoesm.atmosphere.physics.convection.dca import dca_convection
        from legoesm.atmosphere.physics.convection.config import DCAConfig

        config = DCAConfig()

        def loss(qv_in):
            out = dca_convection(
                self.T, qv_in, self.p_full, self.p_half, self.dt, config,
            )
            return jnp.sum(out.dq_c_conv_dt ** 2)

        grad = jax.grad(loss)(self.q_v)
        assert jnp.all(jnp.isfinite(grad)), (
            "DCA gradient at zero col_local_cond — safe_divide guard regressed."
        )

    def test_kuo_grad_finite_at_zero_col_cond(self):
        from legoesm.atmosphere.physics.convection.kuo import kuo_convection
        from legoesm.atmosphere.physics.convection.config import KuoConfig

        config = KuoConfig()

        def loss(qv_in):
            out = kuo_convection(
                self.T, qv_in, self.p_full, self.p_half, self.dt, config,
            )
            return jnp.sum(out.dq_c_conv_dt ** 2)

        grad = jax.grad(loss)(self.q_v)
        assert jnp.all(jnp.isfinite(grad)), (
            "Kuo gradient at zero col_local_cond — safe_divide guard regressed."
        )


# ---------------------------------------------------------------------------
# 3. Microphysics sites
# ---------------------------------------------------------------------------

class TestMicrophysicsAutoconvAggregationGrad:
    """morrison.py / thompson.py / seifert_beheng.py compute
    ``dN_c_dt ~ -dq_c_au * rho / x_c``; ``x_c`` is the mean droplet
    mass which vanishes with ``q_c``.  Morrison/Thompson additionally
    compute ``-aggregation * N_i / q_i`` with ``q_i → 0`` in
    cold-start ice-free columns.  Both are the same ``-a / b**2``
    overflow pattern as the convection sites."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.atmosphere.physics.microphysics.output import (
            HydrometeorState,
        )
        self.ncol, self.nlev = 2, 6
        # Dry stable column with NO cloud field anywhere — drives both
        # x_c → 0 (autoconv) and q_i → 0 (aggregation).
        self.T = 250.0 * jnp.ones((self.ncol, self.nlev))
        self.q_v = jnp.zeros((self.ncol, self.nlev))
        self.p_full, self.p_half = _make_pressure_columns(
            self.ncol, self.nlev,
        )
        self.rho = 1.0 * jnp.ones((self.ncol, self.nlev))
        self.dz = 1000.0 * jnp.ones((self.ncol, self.nlev))
        self.dt = 300.0
        z = jnp.zeros((self.ncol, self.nlev))
        self.hydro = HydrometeorState(
            q_c=z, q_r=z, q_i=z, q_s=z, q_g=z, N_c=z, N_r=z, N_i=z,
        )

    def test_morrison_grad_finite_at_zero_cloud(self):
        from legoesm.atmosphere.physics.microphysics.morrison import (
            morrison_microphysics,
        )
        from legoesm.atmosphere.physics.microphysics.config import (
            MorrisonConfig,
        )
        config = MorrisonConfig()

        def loss(qc_in):
            hydro = self.hydro._replace(q_c=qc_in)
            out = morrison_microphysics(
                self.T, self.q_v, hydro, self.p_full, self.p_half,
                self.rho, self.dz, self.dt, config,
            )
            return jnp.sum(out.dN_c_dt ** 2) + jnp.sum(out.dN_i_dt ** 2)

        grad = jax.grad(loss)(self.hydro.q_c)
        assert jnp.all(jnp.isfinite(grad)), (
            "Morrison dN_c/dN_i gradient at zero cloud — safe_divide regressed."
        )

    def test_thompson_grad_finite_at_zero_cloud(self):
        from legoesm.atmosphere.physics.microphysics.thompson import (
            thompson_microphysics,
        )
        from legoesm.atmosphere.physics.microphysics.config import (
            ThompsonConfig,
        )
        config = ThompsonConfig()

        def loss(qc_in):
            hydro = self.hydro._replace(q_c=qc_in)
            out = thompson_microphysics(
                self.T, self.q_v, hydro, self.p_full, self.p_half,
                self.rho, self.dz, self.dt, config,
            )
            return jnp.sum(out.dN_c_dt ** 2) + jnp.sum(out.dN_i_dt ** 2)

        grad = jax.grad(loss)(self.hydro.q_c)
        assert jnp.all(jnp.isfinite(grad)), (
            "Thompson dN_c/dN_i gradient at zero cloud — safe_divide regressed."
        )

    def test_seifert_beheng_grad_finite_at_zero_cloud(self):
        from legoesm.atmosphere.physics.microphysics.seifert_beheng import (
            seifert_beheng_microphysics,
        )
        from legoesm.atmosphere.physics.microphysics.config import (
            SeifertBehengConfig,
        )
        config = SeifertBehengConfig()

        def loss(qc_in):
            hydro = self.hydro._replace(q_c=qc_in)
            out = seifert_beheng_microphysics(
                self.T, self.q_v, hydro, self.p_full, self.p_half,
                self.rho, self.dz, self.dt, config,
            )
            return jnp.sum(out.dN_c_dt ** 2)

        grad = jax.grad(loss)(self.hydro.q_c)
        assert jnp.all(jnp.isfinite(grad)), (
            "Seifert-Beheng dN_c gradient at zero cloud — safe_divide regressed."
        )


# ---------------------------------------------------------------------------
# 4. Radiation polar singularity
# ---------------------------------------------------------------------------

class TestSolarPolarSingularityGrad:
    """``daily_mean_insolation`` divides by ``cos(lat) * cos(delta)``
    which vanishes at the poles."""

    def test_daily_mean_grad_finite_at_pole(self):
        from legoesm.atmosphere.physics.radiation.solar import (
            daily_mean_insolation,
        )
        # Latitudes spanning equator → north pole → south pole.
        lat = jnp.array([0.0, math.pi / 2, -math.pi / 2, 0.6])

        def loss(lat_in):
            return jnp.sum(daily_mean_insolation(lat_in, day_of_year=80.0))

        grad = jax.grad(loss)(lat)
        assert jnp.all(jnp.isfinite(grad)), (
            "daily_mean_insolation gradient at polar singularity — safe_divide regressed."
        )

    def test_daylight_fraction_grad_finite_at_pole(self):
        from legoesm.atmosphere.physics.radiation.solar import daylight_fraction
        lat = jnp.array([math.pi / 2, -math.pi / 2, 0.3])

        def loss(lat_in):
            return jnp.sum(daylight_fraction(lat_in, day_of_year=80.0))

        grad = jax.grad(loss)(lat)
        assert jnp.all(jnp.isfinite(grad)), (
            "daylight_fraction gradient at polar singularity — safe_divide regressed."
        )


# ---------------------------------------------------------------------------
# 5. Gravity wave drag sites
# ---------------------------------------------------------------------------

def _gwd_column_state(ncol: int, nlev: int):
    """Build a minimal isothermal column for GWD tests."""
    T = 250.0 * jnp.ones((ncol, nlev))
    u = 10.0 * jnp.ones((ncol, nlev))
    v = jnp.zeros((ncol, nlev))
    p_full, p_half = _make_pressure_columns(ncol, nlev)
    # Heights: linearly spaced from 0 (surface, level -1) to ~16 km (top).
    z_full_1d = jnp.linspace(16e3, 1e3, nlev)
    z_half_1d = jnp.linspace(17e3, 0.0, nlev + 1)
    z_full = jnp.broadcast_to(z_full_1d[None, :], (ncol, nlev))
    z_half = jnp.broadcast_to(z_half_1d[None, :], (ncol, nlev + 1))
    rho = 1.0 * jnp.ones((ncol, nlev))
    lat = jnp.zeros(ncol)
    return T, u, v, p_full, p_half, z_full, z_half, rho, lat


class TestLindzenGwdGrad:

    def test_grad_finite_isothermal_column(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import (
            lindzen_gwd,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            LindzenConfig,
        )
        ncol, nlev = 2, 6
        T, u, v, p_full, p_half, z_full, z_half, rho, lat = (
            _gwd_column_state(ncol, nlev)
        )
        config = LindzenConfig()

        def loss(T_in):
            # Isothermal T → dtheta/dz is tiny → N_full is tiny (at
            # the N2 clip floor).  This is the regime in which the
            # prior ``/clip(N_full, 1e-6)`` would emit large cotangents.
            out = lindzen_gwd(
                u, v, T_in, p_full, p_half, z_full, z_half, rho, lat,
                300.0, config,
            )
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad)), (
            "Lindzen gradient through tau_sat divide — safe_divide regressed."
        )


class TestPrognosticSpectralGwdGrad:

    def test_grad_finite_isothermal_column(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
            prognostic_spectral_gwd,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            PrognosticSpectralConfig,
        )
        ncol, nlev = 2, 6
        T, u, v, p_full, p_half, z_full, z_half, rho, lat = (
            _gwd_column_state(ncol, nlev)
        )
        config = PrognosticSpectralConfig()
        spectrum_in = config.launch_flux * jnp.ones(
            (ncol, config.n_azimuths, config.n_wavenumbers),
        )

        def loss(T_in):
            out, _ = prognostic_spectral_gwd(
                u, v, T_in, p_full, p_half, z_full, z_half, rho, lat,
                300.0, config, spectrum_in,
            )
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad)), (
            "Prognostic spectral GWD gradient — safe_divide regressed."
        )
