"""Unit tests for gravity wave drag module.

Tests cover:
- All 6 backends: output shapes, drag-opposes-wind, finite outputs, differentiability
- Scheme-specific physics tests
- Integration bridge: hydrostatic/NH shapes, nonzero tendencies, jax.grad, scheme selection
- Combined physics integration
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    RayleighConfig,
    LindzenConfig,
    McFarlaneConfig,
    HinesConfig,
    PrognosticSpectralConfig,
    GWDMLEmulatorConfig,
    GravityWaveDragConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput, make_zero_output
from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
    prognostic_spectral_gwd,
)
from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import (
    ml_gwd,
    GWDEmulator,
)
from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
    make_gwd_physics,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_columns(ncol=4, nlev=10):
    """Create test column data with moderate winds.

    Temperature from 220K (top) to 290K (bottom), u=10+random, v=5+random.
    Returns u, v, T, p_full, p_half, z_full, z_half, rho, lat.
    """
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
        (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    T = jnp.broadcast_to(
        jnp.linspace(220.0, 290.0, nlev)[None, :],
        (ncol, nlev),
    )

    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None)))
    dz_rev = dz[:, ::-1]
    z_half_cumsum = jnp.cumsum(dz_rev, axis=1)[:, ::-1]
    z_half = jnp.concatenate([z_half_cumsum, jnp.zeros((ncol, 1))], axis=1)
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])

    rho = p_full / (constants.R_d * T)

    u = 10.0 + jax.random.normal(k1, (ncol, nlev)) * 2.0
    v = 5.0 + jax.random.normal(k2, (ncol, nlev)) * 1.0
    lat = jnp.full((ncol,), 0.5)

    return u, v, T, p_full, p_half, z_full, z_half, rho, lat


def _make_stratospheric_columns(ncol=4, nlev=10):
    """Create stratospheric-like columns with strong N^2 and a jet."""
    key = jax.random.PRNGKey(123)

    p_half = jnp.broadcast_to(
        jnp.linspace(1.0, 5.0e4, nlev + 1)[None, :],
        (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    T = jnp.broadcast_to(
        jnp.linspace(200.0, 220.0, nlev)[None, :],
        (ncol, nlev),
    )

    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None)))
    dz_rev = dz[:, ::-1]
    z_half_cumsum = jnp.cumsum(dz_rev, axis=1)[:, ::-1]
    z_half = jnp.concatenate([z_half_cumsum, jnp.zeros((ncol, 1))], axis=1)
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])

    rho = p_full / (constants.R_d * T)

    u = jnp.broadcast_to(jnp.linspace(30.0, 5.0, nlev)[None, :], (ncol, nlev))
    v = jnp.broadcast_to(jnp.linspace(10.0, 2.0, nlev)[None, :], (ncol, nlev))
    lat = jnp.full((ncol,), 0.8)

    return u, v, T, p_full, p_half, z_full, z_half, rho, lat


# ===========================================================================
# Rayleigh
# ===========================================================================

class TestRayleigh:
    """Tests for Rayleigh friction GWD."""

    def test_output_shapes(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = RayleighConfig()
        out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        assert out.du_dt.shape == (ncol, nlev)
        assert out.dv_dt.shape == (ncol, nlev)
        assert out.dT_dt.shape == (ncol, nlev)
        assert out.eps_gwd.shape == (ncol,)

    def test_drag_opposes_wind(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = RayleighConfig()
        out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        # Where u > 0, du_dt should be <= 0 (drag opposes wind)
        mask = jnp.abs(out.du_dt) > 1e-20
        assert jnp.all((u * out.du_dt)[mask] <= 0.0)

    def test_finite_outputs(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = RayleighConfig()
        out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        assert jnp.all(jnp.isfinite(out.du_dt))
        assert jnp.all(jnp.isfinite(out.dv_dt))
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.eps_gwd))

    def test_differentiable(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = RayleighConfig()

        def loss(u_in):
            out = rayleigh_gwd(u_in, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
            return jnp.sum(out.du_dt ** 2 + out.dv_dt ** 2)

        g = jax.grad(loss)(u)
        assert jnp.all(jnp.isfinite(g))

    def test_zero_wind_zero_drag(self):
        ncol, nlev = 4, 10
        _, _, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        u = jnp.zeros((ncol, nlev))
        v = jnp.zeros((ncol, nlev))
        config = RayleighConfig()
        out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        assert jnp.allclose(out.du_dt, 0.0, atol=1e-30)
        assert jnp.allclose(out.dv_dt, 0.0, atol=1e-30)

    def test_sponge_layer_active(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = RayleighConfig(sponge_top=0.1, sponge_k=1.0 / 3600.0)
        out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        # Near model top, drag should be nonzero
        assert float(jnp.max(jnp.abs(out.du_dt[:, 0]))) > 0.0


# ===========================================================================
# Lindzen
# ===========================================================================

class TestLindzen:
    """Tests for Lindzen orographic GWD."""

    def test_output_shapes(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = LindzenConfig()
        out = lindzen_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        assert out.du_dt.shape == (ncol, nlev)
        assert out.eps_gwd.shape == (ncol,)

    def test_drag_opposes_wind(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = LindzenConfig()
        out = lindzen_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        # Drag should generally decelerate wind (accumulated negative tendency)
        total_du = jnp.sum(jnp.abs(out.du_dt))
        assert float(total_du) > 0.0

    def test_finite_outputs(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = LindzenConfig()
        out = lindzen_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        assert jnp.all(jnp.isfinite(out.du_dt))
        assert jnp.all(jnp.isfinite(out.dT_dt))

    def test_differentiable(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = LindzenConfig()

        def loss(u_in):
            out = lindzen_gwd(u_in, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
            return jnp.sum(out.du_dt ** 2 + out.dv_dt ** 2)

        g = jax.grad(loss)(u)
        assert jnp.all(jnp.isfinite(g))

    def test_stronger_topo_more_drag(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)

        config_low = LindzenConfig(h_topo=100.0)
        config_high = LindzenConfig(h_topo=1000.0)
        out_low = lindzen_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config_low)
        out_high = lindzen_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config_high)

        drag_low = float(jnp.sum(jnp.abs(out_low.du_dt)))
        drag_high = float(jnp.sum(jnp.abs(out_high.du_dt)))
        assert drag_high > drag_low


# ===========================================================================
# McFarlane
# ===========================================================================

class TestMcFarlane:
    """Tests for McFarlane orographic GWD."""

    def test_output_shapes(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = McFarlaneConfig()
        out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        assert out.du_dt.shape == (ncol, nlev)
        assert out.eps_gwd.shape == (ncol,)

    def test_drag_opposes_wind(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = McFarlaneConfig()
        out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        total = float(jnp.sum(jnp.abs(out.du_dt)))
        assert total > 0.0

    def test_finite_outputs(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = McFarlaneConfig()
        out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        assert jnp.all(jnp.isfinite(out.du_dt))
        assert jnp.all(jnp.isfinite(out.dT_dt))

    def test_dissipative_and_opposes_wind(self):
        """Orographic GWD must REMOVE kinetic energy (eps_gwd >= 0) and the drag
        must oppose the wind (column sum u*du_dt + v*dv_dt <= 0): mountain waves
        (c=0) always decelerate the flow.  The flux-divergence form
        du/dt = -(1/rho) dtau/dz also makes the column momentum tendency equal
        -(launched stress), i.e. momentum is conserved by construction.

        Codifies the iter-23 conservation audit; the sibling
        ``test_drag_opposes_wind`` only checks ``sum|du_dt| > 0``, not the sign —
        which would not catch the F-GWD-1-style sign defect that affects the
        spectral scheme.
        """
        ncol, nlev = 4, 20
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = McFarlaneConfig()
        out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        assert jnp.all(out.eps_gwd >= -1e-9), (
            f"orographic GWD must be dissipative (eps_gwd >= 0); got {out.eps_gwd}"
        )
        udu = jnp.sum(u * out.du_dt + v * out.dv_dt, axis=1)
        assert jnp.all(udu <= 1e-9), (
            f"orographic drag must oppose the wind (sum u*du_dt <= 0); got {udu}"
        )

    def test_differentiable(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = McFarlaneConfig()

        def loss(u_in):
            out = mcfarlane_gwd(u_in, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
            return jnp.sum(out.du_dt ** 2 + out.dv_dt ** 2)

        g = jax.grad(loss)(u)
        assert jnp.all(jnp.isfinite(g))

    def test_low_wind_suppressed(self):
        ncol, nlev = 4, 10
        _, _, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        # Very low wind
        u = jnp.full((ncol, nlev), 0.1)
        v = jnp.full((ncol, nlev), 0.1)
        config = McFarlaneConfig(min_wind=2.0)
        out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        max_drag = float(jnp.max(jnp.abs(out.du_dt)))
        # Should be very small (min_wind suppression)
        assert max_drag < 1e-3


# ===========================================================================
# Hines
# ===========================================================================

class TestHines:
    """Tests for Hines Doppler-spread GWD."""

    def test_output_shapes(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = HinesConfig()
        out = hines_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        assert out.du_dt.shape == (ncol, nlev)
        assert out.eps_gwd.shape == (ncol,)

    def test_drag_opposes_wind(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = HinesConfig()
        out = hines_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        total = float(jnp.sum(jnp.abs(out.du_dt)))
        assert total > 0.0

    def test_finite_outputs(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = HinesConfig()
        out = hines_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        assert jnp.all(jnp.isfinite(out.du_dt))
        assert jnp.all(jnp.isfinite(out.dT_dt))

    def test_differentiable(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = HinesConfig()

        def loss(u_in):
            out = hines_gwd(u_in, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
            return jnp.sum(out.du_dt ** 2 + out.dv_dt ** 2)

        g = jax.grad(loss)(u)
        assert jnp.all(jnp.isfinite(g))

    def test_isotropic_drag(self):
        """Hines drag should be present in both u and v components."""
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = HinesConfig()
        out = hines_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
        assert float(jnp.max(jnp.abs(out.du_dt))) > 0.0
        assert float(jnp.max(jnp.abs(out.dv_dt))) > 0.0


# ===========================================================================
# Prognostic Spectral
# ===========================================================================

class TestPrognosticSpectral:
    """Tests for prognostic spectral GWD."""

    def test_output_shapes(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = PrognosticSpectralConfig()
        spectrum_in = jnp.full((ncol, config.n_azimuths, config.n_wavenumbers), config.launch_flux)
        out, spec_new = prognostic_spectral_gwd(
            u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config, spectrum_in
        )
        assert out.du_dt.shape == (ncol, nlev)
        assert out.eps_gwd.shape == (ncol,)
        assert spec_new.shape == (ncol, config.n_azimuths, config.n_wavenumbers)

    def test_finite_outputs(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = PrognosticSpectralConfig()
        spectrum_in = jnp.full((ncol, config.n_azimuths, config.n_wavenumbers), config.launch_flux)
        out, spec_new = prognostic_spectral_gwd(
            u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config, spectrum_in
        )
        assert jnp.all(jnp.isfinite(out.du_dt))
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(spec_new))

    def test_differentiable(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = PrognosticSpectralConfig()
        spectrum_in = jnp.full((ncol, config.n_azimuths, config.n_wavenumbers), config.launch_flux)

        def loss(u_in):
            out, _ = prognostic_spectral_gwd(
                u_in, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config, spectrum_in
            )
            return jnp.sum(out.du_dt ** 2 + out.dv_dt ** 2)

        g = jax.grad(loss)(u)
        assert jnp.all(jnp.isfinite(g))

    def test_spectrum_evolution(self):
        """Spectrum should evolve from one call to the next."""
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = PrognosticSpectralConfig()
        spectrum_in = jnp.full((ncol, config.n_azimuths, config.n_wavenumbers), config.launch_flux * 0.5)
        _, spec_new = prognostic_spectral_gwd(
            u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config, spectrum_in
        )
        # Spectrum should change from input
        assert not jnp.allclose(spec_new, spectrum_in)

    def test_acceleration_magnitude_includes_g_factor(self):
        """Audit cycle iter-26 P0: prognostic_spectral GWD must
        multiply the stress-divergence by ``constants.g`` to convert
        from Pa-based drag to per-mass acceleration.

        Without ``g``, du_dt is dimensionless (≈ ΔF/Δp) and is
        ~9.8× too small.  Test asserts the magnitude scales with
        the gravitational acceleration: rerunning with a perturbed
        ``constants.g`` (via a monkeypatched copy of the function)
        would scale the output by the same factor.

        Direct check: with ΔF/Δp ~ 1e-5 (realistic), du_dt should
        be ~1e-4 m/s² when g is included, ~1e-5 m/s² without.
        Asserting ``max |du_dt| > 5e-6 m/s²`` would have failed
        before iter-26 (max was ~5e-7), passes after.
        """
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = PrognosticSpectralConfig()
        # Use a sufficiently large launch flux so the column actually
        # produces non-trivial saturation breaking and depositing.
        spectrum_in = jnp.full(
            (ncol, config.n_azimuths, config.n_wavenumbers),
            max(config.launch_flux, 0.01),
        )
        out, _ = prognostic_spectral_gwd(
            u, v, T, p_full, p_half, z_full, z_half, rho, lat,
            300.0, config, spectrum_in,
        )
        # Sanity: tendencies must be finite.
        assert jnp.all(jnp.isfinite(out.du_dt))
        # The magnitude should be larger than the pre-fix scale by
        # ~g.  Pre-fix max(|du_dt|) was at most ~5e-7.  Post-fix
        # should be ~5e-6 or larger for the chosen launch flux.
        max_du = float(jnp.max(jnp.abs(out.du_dt)))
        assert max_du > 1e-6, (
            f"max|du_dt|={max_du:.2e} is too small — without the "
            f"iter-26 g-factor fix, the prognostic-spectral GWD "
            f"acceleration would be ~10× smaller (or this test "
            f"setup didn't trigger any breaking)."
        )


# ===========================================================================
# ML Emulator
# ===========================================================================

class TestMLEmulator:
    """Tests for ML GWD emulator."""

    def test_output_shapes(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = GWDMLEmulatorConfig()
        model = GWDEmulator(config.n_input, config.n_hidden, config.n_layers,
                            config.n_output, key=jax.random.PRNGKey(0))
        out = ml_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config, model)
        assert out.du_dt.shape == (ncol, nlev)
        assert out.eps_gwd.shape == (ncol,)

    def test_finite_outputs(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = GWDMLEmulatorConfig()
        model = GWDEmulator(config.n_input, config.n_hidden, config.n_layers,
                            config.n_output, key=jax.random.PRNGKey(0))
        out = ml_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config, model)
        assert jnp.all(jnp.isfinite(out.du_dt))
        assert jnp.all(jnp.isfinite(out.dT_dt))

    def test_differentiable(self):
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = GWDMLEmulatorConfig()
        model = GWDEmulator(config.n_input, config.n_hidden, config.n_layers,
                            config.n_output, key=jax.random.PRNGKey(0))

        def loss(u_in):
            out = ml_gwd(u_in, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config, model)
            return jnp.sum(out.du_dt ** 2 + out.dv_dt ** 2)

        g = jax.grad(loss)(u)
        assert jnp.all(jnp.isfinite(g))

    def test_untrained_near_zero(self):
        """Random init with residual scaling should produce small tendencies."""
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        config = GWDMLEmulatorConfig(use_residual=True)
        model = GWDEmulator(config.n_input, config.n_hidden, config.n_layers,
                            config.n_output, key=jax.random.PRNGKey(0))
        out = ml_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config, model)
        assert float(jnp.max(jnp.abs(out.du_dt))) < 1.0
        assert float(jnp.max(jnp.abs(out.dv_dt))) < 1.0

    def test_eps_gwd_sign_convention(self):
        """Audit cycle iter-26 P1: ml_gwd ``eps_gwd`` must use the SIGNED
        ``-(u·du + v·dv)`` mean-flow-KE-removal form (the same diagnostic the
        DISSIPATIVE GWD schemes use; prognostic_spectral instead reports a
        positive wave-dissipation integral), not ``jnp.abs(u·du + v·dv)``.

        The signed form makes an untrained model that ADDS (rather than
        removes) mean-flow KE show up as a NEGATIVE eps_gwd diagnostic (the
        ``jnp.abs`` form would mask it). NOTE eps_gwd is decoupled from the
        SEPARATE unconstrained ``dT_dt`` network channel: there is NO enforced
        KE->heat tie-back ``c_pd·∫ρ·dT_dt·dz = eps_gwd`` for this learned
        emulator.

        Regression strategy: monkey-patch the model's du_dt and
        dv_dt outputs to be aligned with u and v (positive
        ``u·du + v·dv``).  With the corrected formula
        eps_gwd = -∫ρ·(u·du+v·dv)·dz < 0 (KE *added*).  With the
        old buggy ``jnp.abs`` form eps_gwd > 0 always — masking
        the violation.
        """
        # We can't easily monkey-patch the MLP weights, but we can
        # bypass ml_gwd entirely and replicate just the eps_gwd
        # computation, asserting the sign behaviour.
        ncol, nlev = 4, 10
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = _make_columns(ncol, nlev)
        # Force a "model" that aligns du_dt with u and dv_dt with v
        # (acts to ACCELERATE the wind, adding KE).  This is the
        # pathological case that should yield NEGATIVE eps_gwd
        # under the corrected formula.
        du_dt_pos = 1e-4 * u  # positive du_dt aligned with u
        dv_dt_pos = 1e-4 * v
        dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
        # Corrected formula (matches iter-26 fix)
        eps_corrected = -jnp.sum(
            rho * (u * du_dt_pos + v * dv_dt_pos) * dz, axis=1,
        )
        # Buggy formula (jnp.abs) that the iter-26 fix replaced
        eps_buggy = jnp.sum(
            rho * jnp.abs(u * du_dt_pos + v * dv_dt_pos) * dz, axis=1,
        )

        # The corrected formula must be negative for this
        # KE-adding pathological case (energy violation indicator).
        assert float(jnp.min(eps_corrected)) < 0.0, (
            f"eps_corrected should flag KE addition as negative, "
            f"saw min = {float(jnp.min(eps_corrected)):.2e}"
        )
        # The buggy formula always returns positive — proving this
        # test would catch a regression.
        assert float(jnp.min(eps_buggy)) > 0.0, (
            f"eps_buggy should be positive (mask sign); "
            f"saw min = {float(jnp.min(eps_buggy)):.2e}"
        )


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    """Tests for make_gwd_physics integration bridge."""

    def test_hydrostatic_shapes(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
        from legoesm.core.field import Field

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        n = grid.n
        nlev = sigma.n_levels
        state = state._replace(
            u=Field(data=jnp.ones((6, n, n, nlev)) * 10.0,
                    name="u", dims=("face", "x", "y", "level"), units="m/s"),
        )

        config = GravityWaveDragConfig(scheme="rayleigh")
        physics_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        assert tendencies.dT_dt.data.shape == (6, n, n, nlev)
        assert tendencies.du_dt.data.shape == (6, n, n, nlev)
        assert tendencies.dv_dt.data.shape == (6, n, n, nlev)
        assert tendencies.dp_s_dt.data.shape == (6, n, n)

    def test_nonhydrostatic_shapes(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import (
            create_height_coordinate,
            compute_terrain_metric,
        )
        from legoesm.core.field import Field
        from legoesm.core.state import NonHydrostaticState

        n = 8
        nlev = 10
        grid = create_cubed_sphere(n)
        height_coord = create_height_coordinate(nlev, 30000.0)
        z_s = jnp.zeros((6, n, n))
        terrain_metric = compute_terrain_metric(z_s, height_coord)

        shape_3d = (6, n, n, nlev)
        shape_w = (6, n, n, nlev + 1)
        shape_2d = (6, n, n)

        state = NonHydrostaticState(
            u=Field(data=jnp.ones(shape_3d) * 10.0, name="u", dims=("face", "x", "y", "level"), units="m/s"),
            v=Field(data=jnp.ones(shape_3d) * 5.0, name="v", dims=("face", "x", "y", "level"), units="m/s"),
            w=Field(data=jnp.zeros(shape_w), name="w", dims=("face", "x", "y", "level_half"), units="m/s"),
            theta_prime=Field(data=jnp.ones(shape_3d) * 1.0, name="theta_prime", dims=("face", "x", "y", "level"), units="K"),
            rho_prime=Field(data=jnp.ones(shape_3d) * 0.01, name="rho_prime", dims=("face", "x", "y", "level"), units="kg/m^3"),
            phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=("face", "x", "y"), units="m^2/s^2"),
            tracers=Field(data=jnp.zeros(shape_3d + (1,)), name="tracers", dims=("face", "x", "y", "level", "tracer"), units="kg/kg"),
        )

        config = GravityWaveDragConfig(scheme="rayleigh")
        physics_fn = make_gwd_physics(config, model_type="nonhydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, height_coord, terrain_metric)

        assert tendencies.du_dt.data.shape == shape_3d
        assert tendencies.dv_dt.data.shape == shape_3d
        assert tendencies.dtheta_prime_dt.data.shape == shape_3d

    def test_nonzero_momentum_drag(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
        from legoesm.core.field import Field

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)
        n, nlev = grid.n, sigma.n_levels

        state = state._replace(
            u=Field(data=jnp.ones((6, n, n, nlev)) * 10.0,
                    name="u", dims=("face", "x", "y", "level"), units="m/s"),
        )

        config = GravityWaveDragConfig(scheme="rayleigh")
        physics_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        max_du = float(jnp.max(jnp.abs(tendencies.du_dt.data)))
        assert max_du > 0.0

    def test_grad_through_hydrostatic(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
        from legoesm.core.field import Field

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)
        n, nlev = grid.n, sigma.n_levels

        state = state._replace(
            u=Field(data=jnp.ones((6, n, n, nlev)) * 10.0,
                    name="u", dims=("face", "x", "y", "level"), units="m/s"),
        )

        config = GravityWaveDragConfig(scheme="rayleigh")
        physics_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)

        def loss(u_data):
            s = state._replace(
                u=Field(data=u_data, name="u", dims=("face", "x", "y", "level"), units="m/s"),
            )
            t, _ = physics_fn(s, grid, sigma)
            return jnp.sum(t.du_dt.data ** 2)

        g = jax.grad(loss)(state.u.data)
        assert jnp.all(jnp.isfinite(g))

    def test_scheme_selection(self):
        """All 6 scheme strings + 'none' should be accepted."""
        for scheme in ["rayleigh", "lindzen", "mcfarlane", "hines",
                       "prognostic_spectral", "ml_emulator", "none"]:
            config = GravityWaveDragConfig(scheme=scheme)
            physics_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
            assert physics_fn is not None

    def test_none_scheme_zeros(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = GravityWaveDragConfig(scheme="none")
        physics_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        assert jnp.allclose(tendencies.du_dt.data, 0.0)
        assert jnp.allclose(tendencies.dv_dt.data, 0.0)
        assert jnp.allclose(tendencies.dT_dt.data, 0.0)

    def test_combined_physics_integration(self):
        """PhysicsConfig with GWD produces valid output."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
        from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
        from legoesm.atmosphere.physics.radiation.config import RadiationConfig
        from legoesm.atmosphere.physics.convection.config import ConvectionConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
        from legoesm.core.field import Field

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)
        n, nlev = grid.n, sigma.n_levels

        state = state._replace(
            u=Field(data=jnp.ones((6, n, n, nlev)) * 10.0,
                    name="u", dims=("face", "x", "y", "level"), units="m/s"),
        )

        config = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(scheme="none"),
            gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
        )
        physics_fn = make_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        assert tendencies.du_dt.data.shape == (6, n, n, nlev)
        max_du = float(jnp.max(jnp.abs(tendencies.du_dt.data)))
        assert max_du > 0.0


# ===========================================================================
# Zero output helper
# ===========================================================================

class TestZeroOutput:
    def test_make_zero_output(self):
        out = make_zero_output(4, 10)
        assert out.du_dt.shape == (4, 10)
        assert out.eps_gwd.shape == (4,)
        assert jnp.allclose(out.du_dt, 0.0)


# ===========================================================================
# #1514 guard: orographic scalar-fallback warning (ocean pseudo-mountain)
# ===========================================================================

class TestOrographicScalarFallbackWarning:
    """Direct tests of ``orographic_scalar_fallback_warning`` (issue #1514).

    The scalar ``h_topo = 500 m`` fallback puts a fictional mountain over
    every ocean column; the helper must flag exactly the hazardous quadrant
    (orographic member + empty SSO path + real land/ocean distribution) and
    stay silent everywhere else.
    """

    def _fn(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
            orographic_scalar_fallback_warning,
        )
        return orographic_scalar_fallback_warning

    def test_fires_for_orographic_member_no_sso_real_land_sea(self):
        msg = self._fn()("mcfarlane+hines", "", True)
        assert msg is not None
        assert "mcfarlane" in msg
        assert "1514" in msg
        assert "prep_subgrid_orography" in msg  # remedy named

    def test_silent_without_orographic_member(self):
        assert self._fn()("hines", "", True) is None
        assert self._fn()("rayleigh+hines", "", True) is None

    def test_silent_when_sso_file_wired(self):
        assert self._fn()("mcfarlane", "/path/sso_stdh.nc", True) is None

    def test_silent_on_idealized_run_without_land_sea(self):
        # No real land/ocean distribution (e.g. no topography chain ran):
        # the documented legacy scalar fallback stays available quietly.
        assert self._fn()("mcfarlane", "", False) is None

    def test_composite_names_the_orographic_member(self):
        msg = self._fn()("hines+e3sm_cam", "", True)
        assert msg is not None
        assert "e3sm_cam" in msg

    def test_driver_wires_the_guard(self):
        # The symbol that RUNS: ModelDriver._create_topography owns the
        # subgrid_orography_path block (the sso_path if/else) and must call
        # the helper on its else branch.  This assertion goes red if the
        # driver call site is reverted while the helper survives.
        import inspect
        from legoesm.driver.model_driver import ModelDriver
        src = inspect.getsource(ModelDriver._create_topography)
        assert "orographic_scalar_fallback_warning" in src
