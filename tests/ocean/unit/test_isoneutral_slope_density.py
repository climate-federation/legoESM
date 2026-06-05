"""Tests for the Veros-faithful NEUTRAL isoneutral-slope-density option.

``GMRediConfig.slope_density`` selects the density gradient used to BUILD the
isoneutral slopes:

- ``"in_situ"`` (default) — finite-difference the in-situ density (carries the
  adiabatic compressibility term in ``∂_z ρ`` → slopes / S² / K_33 too small).
- ``"neutral"`` — the locally-referenced ``∂ρ/∂T·∇T + ∂ρ/∂S·∇S`` form with the
  EOS partials at the local cell pressure (Veros ``get_drhodT``/``get_drhodS``).

Truth-tier gates (these OUTRANK any oracle correlation):

(a) default-off (``"in_situ"``) is BIT-IDENTICAL to the prior single-arg call;
(b) neutral K_33 is O(10×) larger than in-situ on a stratified column (the
    lever the localization identified);
(c) Redi cancellation: for a tracer ``q`` aligned with the neutral surface
    (``q = ρ`` under a linear EOS, where ∂ρ/∂T, ∂ρ/∂S are constant) the neutral
    iso-tendency vanishes to machine precision;
(d) the divergence-form neutral flux globally conserves the tracer;
(e) ``jax.grad`` is finite through the neutral slope build (EOS-derivative
    autodiff + the stable-strat floor).

Run with:
    JAX_ENABLE_X64=1 python -m pytest \
        tests/ocean/unit/test_isoneutral_slope_density.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _neumann_fill_cgrid
from legoesm.ocean.dynamics.ocean_tendency_common import (
    iterate_eos_and_pressure_anomaly,
)
from legoesm.ocean.eos import (
    LinearEOSConfig,
    eos_density_derivatives,
    haline_contraction_coeff,
    make_eos_fn,
    thermal_expansion_coeff,
    wright_eos,
)
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    compute_isoneutral_K33_latlon,
    gm_redi_tracer_tendency_latlon,
    gm_redi_tracer_tendency_triads_latlon_cgrid,
)
from legoesm.ocean.vertical import compute_ocean_jacobian, create_ocean_z_star

RHO_0 = 1024.0
G = 9.81


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _setup(n_lat=10, n_lon=14, nlev=8, dTdy=0.08, dTdz_scale=9.0,
           seed=1, salt_structure=True):
    """Stratified, meridionally tilted T/S channel on a lat-lon C-grid."""
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z = create_ocean_z_star(
        n_levels=nlev, H_max=4000.0, dz_surface=80.0, dz_deep=900.0,
    )
    rng = np.random.default_rng(seed)
    lat = np.linspace(-60, -40, n_lat)
    Tprof = np.linspace(2.0 + dTdz_scale, 2.0, nlev)
    T = (Tprof[None, None, :] + dTdy * lat[:, None, None]
         + 0.03 * rng.standard_normal((n_lat, n_lon, nlev)))
    if salt_structure:
        S = (35.0 + 0.01 * lat[:, None, None]
             + 0.02 * rng.standard_normal((n_lat, n_lon, nlev)))
    else:
        S = 35.0 + 0.0 * T
    T = jnp.asarray(T)
    S = jnp.asarray(S)
    eta = jnp.zeros((n_lat, n_lon))
    H = jnp.full((n_lat, n_lon), 4000.0)
    mask = jnp.ones((n_lat, n_lon))
    um = jnp.ones((n_lat, n_lon + 1))
    vm = jnp.ones((n_lat + 1, n_lon))
    J = compute_ocean_jacobian(eta, H, z)
    return grid, z, T, S, eta, H, mask, um, vm, J


def _rho(T, S, mask, z, eos_fn):
    rho, _, _ = iterate_eos_and_pressure_anomaly(
        T, S, mask, lambda f: _neumann_fill_cgrid(f, mask), eos_fn,
        z.dz_ref, RHO_0, G, n_iter=2,
    )
    return rho


# =====================================================================
# eos.py: drhodT / drhodS helper
# =====================================================================

class TestEOSDensityDerivatives:
    def test_wright_matches_alpha_beta(self):
        """∂ρ/∂T = -ρα and ∂ρ/∂S = +ρβ for the wright EOS (the reuse path)."""
        T = jnp.asarray([2.0, 5.0, 8.0, 1.0])
        S = jnp.asarray([35.0, 34.8, 34.9, 35.1])
        p = jnp.asarray([5e5, 5e6, 2e7, 4e7])
        eos = make_eos_fn("wright")
        dT, dS = eos_density_derivatives(eos, T, S, p)
        rho = wright_eos(T, S, p)
        a = thermal_expansion_coeff(T, S, p)
        b = haline_contraction_coeff(T, S, p)
        assert jnp.allclose(dT, -rho * a, atol=0.0, rtol=0.0)
        assert jnp.allclose(dS, rho * b, atol=0.0, rtol=0.0)

    def test_veros_nonlin2_matches_closed_form(self):
        """The autodiff partials match the Veros nonlin2 closed-form derivatives."""
        from legoesm.ocean.eos import VerosNonlin2Config
        cfg = VerosNonlin2Config()
        dep = jnp.asarray([50.0, 500.0, 2000.0, 4000.0])
        Tc = jnp.asarray([2.0, 5.0, 8.0, 1.0])
        S = jnp.asarray([35.0, 34.8, 34.9, 35.1])
        p = dep * cfg.rho_0 * cfg.grav  # invert depth_m = p/(rho_0 g)
        eos = make_eos_fn("veros_nonlin2")
        dT, dS = eos_density_derivatives(eos, Tc, S, p)
        zz = -dep - cfg.z0
        th = Tc - cfg.theta0_C
        drdT_closed = -(cfg.betaTs * th
                        + cfg.betaT * (1.0 - cfg.gammas * cfg.grav * zz * cfg.rho_0)
                        ) * cfg.rho_0
        drdS_closed = cfg.betaS * cfg.rho_0
        assert jnp.allclose(dT, drdT_closed, atol=1e-6)
        assert jnp.allclose(dS, drdS_closed, atol=1e-9)

    def test_units_and_signs(self):
        """∂ρ/∂T < 0 (warmer = lighter), ∂ρ/∂S > 0 (saltier = denser)."""
        T = jnp.asarray([5.0]); S = jnp.asarray([35.0]); p = jnp.asarray([1e6])
        dT, dS = eos_density_derivatives(make_eos_fn("wright"), T, S, p)
        assert float(dT[0]) < 0.0
        assert float(dS[0]) > 0.0


class TestSlopeDensityDispatch:
    def test_unknown_literal_raises(self):
        """An unrecognised slope_density must fail fast (Dispatch Discipline) —
        no silent fall-through to the in-situ branch."""
        grid, z, T, S, eta, H, mask, um, vm, J = _setup()
        bad = GMRediConfig(
            kappa_GM=1000.0, kappa_Redi=1000.0, S_max=0.01,
            taper_width_frac=0.5, slope_density="insitu",  # typo
        )
        with pytest.raises(ValueError, match="slope_density"):
            gm_redi_tracer_tendency_latlon(
                T, S, eta, H, grid, z, bad, eos="veros_nonlin2",
                mask=mask, u_mask=um, v_mask=vm, rho_0=RHO_0, g=G,
            )


# =====================================================================
# (a) Gate 1: default-off bit-identity
# =====================================================================

class TestDefaultOffBitIdentity:
    def test_triad_in_situ_bit_identical(self):
        """The triad tendency with slope_density='in_situ' (the new default) is
        byte-for-byte identical to the legacy call without the option."""
        grid, z, T, S, eta, H, mask, um, vm, J = _setup()
        eos_fn = make_eos_fn("veros_nonlin2")
        rho = _rho(T, S, mask, z, eos_fn)
        # Legacy positional call (no slope_density / neutral kwargs).
        dq_legacy = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, um, vm, z, J, grid, 1000.0, 1000.0, 0.01, 0.5,
            True, 500.0,
        )
        # Explicit in_situ + neutral kwargs supplied (must be ignored).
        dq_insitu = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, um, vm, z, J, grid, 1000.0, 1000.0, 0.01, 0.5,
            True, 500.0, slope_density="in_situ",
            T_tracer=T, S_tracer=S, eos_fn=eos_fn, rho_0=RHO_0, g=G,
        )
        assert float(jnp.max(jnp.abs(dq_legacy - dq_insitu))) == 0.0

    def test_orchestrator_in_situ_default_unchanged(self):
        """The orchestrator default (config without slope_density set) equals
        the explicit in_situ config, bit-for-bit."""
        grid, z, T, S, eta, H, mask, um, vm, J = _setup()
        cfg_default = GMRediConfig(
            kappa_GM=1000.0, kappa_Redi=1000.0, S_max=0.01,
            taper_width_frac=0.5, implicit_K33=True, K_iso_steep=500.0,
        )
        assert cfg_default.slope_density == "in_situ"
        dT1, dS1 = gm_redi_tracer_tendency_latlon(
            T, S, eta, H, grid, z, cfg_default, eos="veros_nonlin2",
            mask=mask, u_mask=um, v_mask=vm, rho_0=RHO_0, g=G,
        )
        cfg_explicit = cfg_default._replace(slope_density="in_situ")
        dT2, dS2 = gm_redi_tracer_tendency_latlon(
            T, S, eta, H, grid, z, cfg_explicit, eos="veros_nonlin2",
            mask=mask, u_mask=um, v_mask=vm, rho_0=RHO_0, g=G,
        )
        assert float(jnp.max(jnp.abs(dT1 - dT2))) == 0.0
        assert float(jnp.max(jnp.abs(dS1 - dS2))) == 0.0


# =====================================================================
# (b) Gate: neutral K_33 >> in-situ K_33 (the lever)
# =====================================================================

class TestNeutralK33Lever:
    def test_neutral_k33_much_larger(self):
        """On a stratified column the neutral K_33 must be O(10×) larger than
        the compressibility-biased in-situ K_33 (the localized lever)."""
        grid, z, T, S, eta, H, mask, um, vm, J = _setup()
        nlev = T.shape[-1]
        kw = jnp.full((T.shape[0], T.shape[1], nlev - 1), 1000.0)
        cfgN = GMRediConfig(
            kappa_GM=1000.0, kappa_Redi=1000.0, S_max=0.01,
            taper_width_frac=0.5, slope_density="neutral",
        )
        cfgI = cfgN._replace(slope_density="in_situ")
        k33N = compute_isoneutral_K33_latlon(
            T, S, eta, H, grid, z, cfgN, eos="veros_nonlin2", mask=mask,
            rho_0=RHO_0, g=G, kappa_redi_override=kw,
        )
        k33I = compute_isoneutral_K33_latlon(
            T, S, eta, H, grid, z, cfgI, eos="veros_nonlin2", mask=mask,
            rho_0=RHO_0, g=G, kappa_redi_override=kw,
        )
        assert jnp.all(k33N >= 0.0)
        assert jnp.all(k33I >= 0.0)
        ratio = float(jnp.mean(k33N)) / (float(jnp.mean(k33I)) + 1e-30)
        assert ratio > 10.0, (
            f"neutral K_33 not the lever: mean ratio {ratio:.2f} (<10×)"
        )


# =====================================================================
# (c) Gate 2: Redi cancellation for q aligned with the neutral surface
# =====================================================================

class TestNeutralRediCancellation:
    def test_q_eq_rho_linear_eos_cancels(self):
        """Linear EOS ⇒ ∂ρ/∂T, ∂ρ/∂S are CONSTANT, so q = ρ has gradients
        exactly ∂ρ/∂T·∇T + ∂ρ/∂S·∇S — the neutral slope numerator.  The Redi
        tendency for q = ρ must then vanish to machine precision, even with
        pure Redi (kappa_GM = 0 ≠ kappa_Redi) so the off-diagonal does not
        trivially cancel via (kappa_Redi - kappa_GM) = 0."""
        grid, z, T, S, eta, H, mask, um, vm, J = _setup(
            dTdy=0.15, dTdz_scale=10.0, salt_structure=False,
        )
        lin = LinearEOSConfig()
        eos_fn = make_eos_fn("linear", lin)
        rho = _rho(T, S, mask, z, eos_fn)
        # q = rho, pure Redi: kappa_GM = 0, kappa_Redi = 1000.
        dq = gm_redi_tracer_tendency_triads_latlon_cgrid(
            rho, rho, mask, um, vm, z, J, grid, 0.0, 1000.0, 0.05, 0.5,
            False, 0.0, slope_density="neutral",
            T_tracer=T, S_tracer=S, eos_fn=eos_fn, rho_0=RHO_0, g=G,
        )
        # A non-aligned tracer (q = level index) gives a genuine, much larger
        # Redi tendency — the cancellation residual must be far below it.
        nlev = T.shape[-1]
        q_misaligned = jnp.broadcast_to(
            jnp.arange(nlev, dtype=jnp.float64)[None, None, :], T.shape)
        dq_mis = gm_redi_tracer_tendency_triads_latlon_cgrid(
            q_misaligned, rho, mask, um, vm, z, J, grid, 0.0, 1000.0, 0.05,
            0.5, False, 0.0, slope_density="neutral",
            T_tracer=T, S_tracer=S, eos_fn=eos_fn, rho_0=RHO_0, g=G,
        )
        resid = float(jnp.max(jnp.abs(dq)))
        signal = float(jnp.max(jnp.abs(dq_mis)))
        assert signal > 1e-13, "test setup produced no genuine Redi signal"
        assert resid < 1e-6 * signal, (
            f"neutral Redi cancellation broke: residual {resid:.3e} vs "
            f"genuine signal {signal:.3e} (ratio {resid/signal:.2e})"
        )


# =====================================================================
# (d) Gate 3: tracer conservation (divergence form)
# =====================================================================

class TestNeutralConservation:
    def test_volume_integral_conserved(self):
        """The neutral iso-diffusion flux is divergence-form ⇒ the
        volume-weighted sum of the tendency is zero on a closed domain."""
        grid, z, T, S, eta, H, mask, um, vm, J = _setup()
        eos_fn = make_eos_fn("veros_nonlin2")
        rho = _rho(T, S, mask, z, eos_fn)
        dz = z.dz_ref * J[:, :, None]
        try:
            cell_area = np.asarray(grid.area)
        except Exception:
            cell_area = np.ones(mask.shape)
        if cell_area.ndim == 2:
            vol = cell_area[:, :, None] * np.asarray(dz)
        else:
            vol = np.asarray(dz)
        for kg, kr in [(1000.0, 1000.0), (0.0, 1000.0)]:
            dT = gm_redi_tracer_tendency_triads_latlon_cgrid(
                T, rho, mask, um, vm, z, J, grid, kg, kr, 0.01, 0.5,
                False, 0.0, slope_density="neutral",
                T_tracer=T, S_tracer=S, eos_fn=eos_fn, rho_0=RHO_0, g=G,
            )
            integral = float(np.sum(np.asarray(dT) * vol))
            scale = float(np.sum(np.abs(np.asarray(dT)) * vol))
            rel = abs(integral) / (scale + 1e-30)
            assert rel < 1e-10, (
                f"neutral conservation violated (kg={kg}, kr={kr}): "
                f"rel {rel:.3e}"
            )


# =====================================================================
# (e) Gate 4: differentiability
# =====================================================================

class TestNeutralDifferentiability:
    def test_grad_finite_through_neutral_path(self):
        """jax.grad through the full neutral build (EOS-derivative autodiff +
        the stable-strat floor) must be finite, and match a finite-difference
        probe to FD-truncation accuracy."""
        grid, z, T0, S0, eta, H, mask, um, vm, J = _setup(
            n_lat=8, n_lon=10, nlev=6, seed=2,
        )
        eos_fn = make_eos_fn("veros_nonlin2")

        def loss(T):
            rho = _rho(T, S0, mask, z, eos_fn)
            dT = gm_redi_tracer_tendency_triads_latlon_cgrid(
                T, rho, mask, um, vm, z, J, grid, 1000.0, 1000.0, 0.01, 0.5,
                True, 500.0, slope_density="neutral",
                T_tracer=T, S_tracer=S0, eos_fn=eos_fn, rho_0=RHO_0, g=G,
            )
            return jnp.sum(dT ** 2)

        g = jax.grad(loss)(T0)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_matches_finite_difference(self):
        """FD check on one tracer element through the neutral tendency."""
        grid, z, T0, S0, eta, H, mask, um, vm, J = _setup(
            n_lat=8, n_lon=10, nlev=6, seed=2,
        )
        eos_fn = make_eos_fn("veros_nonlin2")

        def loss(T):
            rho = _rho(T, S0, mask, z, eos_fn)
            # Pure Redi on a misaligned tracer for a non-degenerate gradient.
            q = jnp.broadcast_to(
                jnp.arange(T.shape[-1], dtype=jnp.float64)[None, None, :],
                T.shape)
            dq = gm_redi_tracer_tendency_triads_latlon_cgrid(
                q, rho, mask, um, vm, z, J, grid, 0.0, 1000.0, 0.05, 0.5,
                False, 0.0, slope_density="neutral",
                T_tracer=T, S_tracer=S0, eos_fn=eos_fn, rho_0=RHO_0, g=G,
            )
            return jnp.sum(dq ** 2)

        g = jax.grad(loss)(T0)
        assert jnp.all(jnp.isfinite(g))
        idx = (4, 5, 3)
        eps = 1e-4
        fd = (float(loss(T0.at[idx].add(eps)))
              - float(loss(T0.at[idx].add(-eps)))) / (2 * eps)
        ad = float(g[idx])
        relerr = abs(fd - ad) / (abs(fd) + 1e-30)
        assert relerr < 1e-3, f"AD vs FD mismatch: FD={fd:.3e} AD={ad:.3e}"


# =====================================================================
# Free-run-style stability smoke (short forward stepping of T, S)
# =====================================================================

class TestNeutralStability:
    def test_short_stepping_stays_finite(self):
        """A few explicit forward steps of the neutral GM/Redi tendency stay
        finite and bounded (no slope blow-up from the floor / division)."""
        grid, z, T, S, eta, H, mask, um, vm, J = _setup()
        cfg = GMRediConfig(
            kappa_GM=1000.0, kappa_Redi=1000.0, S_max=0.01,
            taper_width_frac=0.5, implicit_K33=False, K_iso_steep=500.0,
            slope_density="neutral",
        )
        dt = 3600.0
        for _ in range(5):
            dT, dS = gm_redi_tracer_tendency_latlon(
                T, S, eta, H, grid, z, cfg, eos="veros_nonlin2",
                mask=mask, u_mask=um, v_mask=vm, rho_0=RHO_0, g=G,
            )
            assert jnp.all(jnp.isfinite(dT))
            assert jnp.all(jnp.isfinite(dS))
            T = T + dt * dT
            S = S + dt * dS
        assert jnp.all(jnp.isfinite(T))
        assert float(jnp.max(jnp.abs(T))) < 1e3
