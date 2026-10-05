"""Unit and integration tests for the lat-lon C-grid shallow water model."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
    CGridLatLonShallowWaterModel,
    CGridLatLonShallowWaterConfig,
    CGridLatLonShallowWaterState,
    cgrid_latlon_sw_tendencies,
    williamson_test2_cgrid,
    williamson_test2_exact_cgrid,
    williamson_test5_cgrid,
    compute_error_norms_cgrid,
    _kinetic_energy_cgrid,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    interp_cell_to_uface,
    interp_cell_to_vface,
)
from legoesm import constants


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def grid():
    return create_latlon_grid(n_lat=32, radius=constants.R_earth, omega=constants.Omega)


@pytest.fixture(scope="module")
def fine_grid():
    return create_latlon_grid(n_lat=64, radius=constants.R_earth, omega=constants.Omega)


@pytest.fixture(scope="module")
def model(grid):
    config = CGridLatLonShallowWaterConfig(fix_mass=True)
    return CGridLatLonShallowWaterModel(grid, config)


# ==============================================================================
# Shape and basic consistency tests
# ==============================================================================

class TestShapes:

    def test_williamson2_shapes(self, grid):
        state = williamson_test2_cgrid(grid)
        n_lat, n_lon = grid.n_lat, grid.n_lon
        assert state.h.shape == (n_lat, n_lon)
        assert state.u.shape == (n_lat, n_lon + 1)
        assert state.v.shape == (n_lat + 1, n_lon)
        assert state.h_s.shape == (n_lat, n_lon)

    def test_tendencies_shapes(self, grid):
        state = williamson_test2_cgrid(grid)
        dh, du, dv = cgrid_latlon_sw_tendencies(state, grid)
        n_lat, n_lon = grid.n_lat, grid.n_lon
        assert dh.shape == (n_lat, n_lon)
        assert du.shape == (n_lat, n_lon + 1)
        assert dv.shape == (n_lat + 1, n_lon)

    def test_interp_uface_shape(self, grid):
        h = jnp.ones((grid.n_lat, grid.n_lon))
        assert interp_cell_to_uface(h).shape == (grid.n_lat, grid.n_lon + 1)

    def test_interp_vface_shape(self, grid):
        h = jnp.ones((grid.n_lat, grid.n_lon))
        assert interp_cell_to_vface(h).shape == (grid.n_lat + 1, grid.n_lon)


# ==============================================================================
# Rest state tests
# ==============================================================================

class TestRestState:

    def test_rest_state_zero_tendencies(self, grid):
        n_lat, n_lon = grid.n_lat, grid.n_lon
        state = CGridLatLonShallowWaterState(
            h=jnp.full((n_lat, n_lon), 1000.0),
            u=jnp.zeros((n_lat, n_lon + 1)),
            v=jnp.zeros((n_lat + 1, n_lon)),
            h_s=jnp.zeros((n_lat, n_lon)),
        )
        dh, du, dv = cgrid_latlon_sw_tendencies(state, grid)
        assert jnp.allclose(dh, 0.0, atol=1e-20)
        assert jnp.allclose(du, 0.0, atol=1e-20)
        assert jnp.allclose(dv, 0.0, atol=1e-20)

    def test_rest_state_with_topography(self, grid):
        """Flat free surface: h + h_s = const → zero tendency."""
        n_lat, n_lon = grid.n_lat, grid.n_lon
        h_s = 100.0 * jnp.sin(grid.lat2d) ** 2
        state = CGridLatLonShallowWaterState(
            h=jnp.full((n_lat, n_lon), 5000.0) - h_s,
            u=jnp.zeros((n_lat, n_lon + 1)),
            v=jnp.zeros((n_lat + 1, n_lon)),
            h_s=h_s,
        )
        dh, du, dv = cgrid_latlon_sw_tendencies(state, grid)
        assert jnp.allclose(dh, 0.0, atol=1e-20)
        assert jnp.allclose(du, 0.0, atol=1e-10)
        assert jnp.allclose(dv, 0.0, atol=1e-10)


# ==============================================================================
# Geostrophic balance (Williamson Test 2) — tightened to 1e-6
# ==============================================================================

class TestGeostrophicBalance:

    def test_tendencies_finite(self, grid):
        state = williamson_test2_cgrid(grid)
        dh, du, dv = cgrid_latlon_sw_tendencies(state, grid)
        assert jnp.all(jnp.isfinite(dh))
        assert jnp.all(jnp.isfinite(du))
        assert jnp.all(jnp.isfinite(dv))

    def test_height_tendency_small(self, grid):
        state = williamson_test2_cgrid(grid)
        dh, _, _ = cgrid_latlon_sw_tendencies(state, grid)
        assert jnp.max(jnp.abs(dh)) < 1e-10

    def test_v_tendency_small(self, grid):
        """dv/dt should be near zero (geostrophic balance at C-grid accuracy).

        The residual is O(dx^2) truncation error from the discrete Coriolis–
        pressure-gradient balance.  At C32 (~600 km) this is ~O(1e-4).
        """
        state = williamson_test2_cgrid(grid)
        _, _, dv = cgrid_latlon_sw_tendencies(state, grid)
        max_dv = float(jnp.max(jnp.abs(dv)))
        assert max_dv < 5e-4, f"max |dv/dt| = {max_dv} (expected < 5e-4)"


# ==============================================================================
# Pole consistency
# ==============================================================================

class TestPoleConsistency:

    def test_v_zero_at_poles(self, grid):
        model = CGridLatLonShallowWaterModel(grid)
        state = williamson_test2_cgrid(grid)
        state_new = model.step(state, dt=60.0)
        assert jnp.allclose(state_new.v[0, :], 0.0)
        assert jnp.allclose(state_new.v[-1, :], 0.0)

    def test_gradient_y_zero_at_poles(self, grid):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import gradient_y_cgrid
        h = jnp.ones((grid.n_lat, grid.n_lon))
        grad_y = gradient_y_cgrid(h, grid)
        assert jnp.allclose(grad_y[0, :], 0.0)
        assert jnp.allclose(grad_y[-1, :], 0.0)


# ==============================================================================
# Mass conservation — uses explicit target_mass API
# ==============================================================================

class TestMassConservation:

    def test_mass_conserved_100_steps(self, grid):
        config = CGridLatLonShallowWaterConfig(fix_mass=True)
        model = CGridLatLonShallowWaterModel(grid, config)
        state = williamson_test5_cgrid(grid)
        target_mass = model.compute_mass(state)

        area64 = grid.area.astype(jnp.float64)
        mass_init = float(jnp.sum(state.h.astype(jnp.float64) * area64))
        dt = 60.0
        for _ in range(100):
            state = model.step(state, dt, target_mass=target_mass)

        mass_final = float(jnp.sum(state.h.astype(jnp.float64) * area64))
        # iter-164: centralized helper.
        from legoesm.diagnostics import compute_relative_drift
        rel_err = compute_relative_drift([mass_init, mass_final])
        assert rel_err < 1e-6, f"Mass conservation error: {rel_err}"


# ==============================================================================
# Stability and error bounds — tightened L2 to 0.01
# ==============================================================================

class TestStability:

    def test_williamson2_6hr_stable(self, grid):
        config = CGridLatLonShallowWaterConfig(fix_mass=True)
        model = CGridLatLonShallowWaterModel(grid, config)
        state = williamson_test2_cgrid(grid)
        target_mass = model.compute_mass(state)

        dt = 60.0
        n_steps = int(6 * 3600 / dt)
        for _ in range(n_steps):
            state = model.step(state, dt, target_mass=target_mass)

        assert jnp.all(jnp.isfinite(state.h))
        assert jnp.all(jnp.isfinite(state.u))
        assert jnp.all(jnp.isfinite(state.v))

    def test_williamson2_error_bounded(self, grid):
        """After 6h, L2 error < 0.01 (was 0.1 — tightened)."""
        config = CGridLatLonShallowWaterConfig(fix_mass=True)
        model = CGridLatLonShallowWaterModel(grid, config)
        state = williamson_test2_cgrid(grid)
        target_mass = model.compute_mass(state)

        dt = 60.0
        for _ in range(int(6 * 3600 / dt)):
            state = model.step(state, dt, target_mass=target_mass)

        ref = williamson_test2_exact_cgrid(grid, t=6 * 3600)
        norms = compute_error_norms_cgrid(state, ref, grid)
        assert norms["l2"] < 0.01, f"L2 error = {norms['l2']}"
        assert norms["linf"] < 0.05, f"Linf error = {norms['linf']}"

    def test_williamson5_6hr_stable(self, grid):
        config = CGridLatLonShallowWaterConfig(fix_mass=True)
        model = CGridLatLonShallowWaterModel(grid, config)
        state = williamson_test5_cgrid(grid)
        target_mass = model.compute_mass(state)

        dt = 60.0
        for _ in range(int(6 * 3600 / dt)):
            state = model.step(state, dt, target_mass=target_mass)

        assert jnp.all(jnp.isfinite(state.h))
        assert float(jnp.min(state.h)) > 0.0


# ==============================================================================
# KE diagnostic
# ==============================================================================

class TestKineticEnergy:

    def test_ke_zero_for_rest_state(self, grid):
        n_lat, n_lon = grid.n_lat, grid.n_lon
        ke = _kinetic_energy_cgrid(
            jnp.zeros((n_lat, n_lon + 1)), jnp.zeros((n_lat + 1, n_lon)),
        )
        assert jnp.allclose(ke, 0.0)

    def test_ke_uniform_zonal(self, grid):
        n_lat, n_lon = grid.n_lat, grid.n_lon
        u0 = 10.0
        ke = _kinetic_energy_cgrid(
            jnp.full((n_lat, n_lon + 1), u0), jnp.zeros((n_lat + 1, n_lon)),
        )
        assert jnp.allclose(ke, 0.5 * u0**2, rtol=1e-10)


# ==============================================================================
# Total energy conservation
# ==============================================================================

class TestEnergyConservation:

    def _total_energy(self, state, grid, g):
        """KE + PE, area-weighted."""
        u_c = 0.5 * (state.u[:, :-1] + state.u[:, 1:])
        v_c = 0.5 * (state.v[:-1, :] + state.v[1:, :])
        ke = 0.5 * state.h * (u_c**2 + v_c**2)
        pe = 0.5 * g * (state.h + state.h_s) ** 2
        return float(jnp.sum((ke + pe) * grid.area))

    def test_energy_drift_bounded(self, grid):
        """Total energy drift should be < 1% over 100 steps."""
        g = constants.g
        config = CGridLatLonShallowWaterConfig(fix_mass=True, A_h=0.0)
        model = CGridLatLonShallowWaterModel(grid, config)
        state = williamson_test2_cgrid(grid)
        target_mass = model.compute_mass(state)

        E0 = self._total_energy(state, grid, g)
        dt = 60.0
        for _ in range(100):
            state = model.step(state, dt, target_mass=target_mass)
        E1 = self._total_energy(state, grid, g)

        rel_drift = abs(E1 - E0) / abs(E0)
        assert rel_drift < 0.01, f"Energy drift = {rel_drift:.4e}"


# ==============================================================================
# Convergence rate (Williamson 2: n=32 vs n=64 → ~4x L2 reduction)
# ==============================================================================

class TestConvergence:

    def test_tendency_convergence(self, grid, fine_grid):
        """Area-weighted L2 tendency residual should decrease from C32 → C64.

        Pure spatial convergence test: single tendency evaluation on the
        Williamson 2 steady state (analytic tendency = 0).  The
        area-weighted L2 norm avoids pole-point sensitivity of max norm.
        """
        state_32 = williamson_test2_cgrid(grid)
        dh_32, _, _ = cgrid_latlon_sw_tendencies(state_32, grid)
        l2_32 = float(jnp.sqrt(jnp.sum(dh_32**2 * grid.area) / grid.total_area))

        state_64 = williamson_test2_cgrid(fine_grid)
        dh_64, _, _ = cgrid_latlon_sw_tendencies(state_64, fine_grid)
        l2_64 = float(jnp.sqrt(jnp.sum(dh_64**2 * fine_grid.area) / fine_grid.total_area))

        # Both should be near zero; fine grid should be smaller
        assert l2_32 < 1e-8, f"dh/dt L2 at C32 not near zero: {l2_32:.4e}"
        assert l2_64 < 1e-8, f"dh/dt L2 at C64 not near zero: {l2_64:.4e}"
        assert l2_64 <= l2_32, (
            f"Fine grid L2 ({l2_64:.4e}) not ≤ coarse ({l2_32:.4e})"
        )


# ==============================================================================
# Fallback transport (use_ppm_transport=False)
# ==============================================================================

class TestFallbackTransport:

    def test_no_ppm_williamson2_stable(self, grid):
        """The 2nd-order fallback transport should be stable for 100 steps."""
        config = CGridLatLonShallowWaterConfig(
            fix_mass=True, use_ppm_transport=False,
        )
        model = CGridLatLonShallowWaterModel(grid, config)
        state = williamson_test2_cgrid(grid)
        target_mass = model.compute_mass(state)

        dt = 60.0
        for _ in range(100):
            state = model.step(state, dt, target_mass=target_mass)

        assert jnp.all(jnp.isfinite(state.h))
        assert jnp.all(jnp.isfinite(state.u))
        assert jnp.all(jnp.isfinite(state.v))


# ==============================================================================
# Polar filter
# ==============================================================================

def _polar_perturbed_state(grid):
    """Williamson 2 + an UNBALANCED polar perturbation.

    * v: grid-scale (Nyquist zonal wavenumber) noise on the v-face rows
      just inside each pole — exactly the modes the polar zonal CFL
      forbids at the relaxed (equatorial-CFL) dt.
    * u: a uniform (k=0) zonal wind on the polar u rows.  k=0 passes any
      Fourier mask, and it strengthens the (zeta+f)*u_at_v coupling that
      feeds the high-k v noise back into dv/dt (Williamson 2 alone has
      u ~ u0*cos(lat) ~ 0 near the poles, which hides an unfiltered dv).

    The balanced-W2 case used by the original polar-filter test has
    dv ~ 0 everywhere, which is exactly why the missing dv filtering
    went unnoticed.
    """
    state = williamson_test2_cgrid(grid)
    n_lon = grid.n_lon
    pert = 5.0 * jnp.where(jnp.arange(n_lon) % 2 == 0, 1.0, -1.0)
    v = state.v.at[1, :].add(pert).at[2, :].add(pert)
    v = v.at[-2, :].add(pert).at[-3, :].add(pert)
    u = state.u.at[1, :].add(20.0).at[2, :].add(20.0)
    u = u.at[-2, :].add(20.0).at[-3, :].add(20.0)
    return state._replace(v=v, u=u)


class TestPolarFilter:

    def test_polar_filter_stable_at_larger_dt(self, grid):
        dt = 300.0
        config = CGridLatLonShallowWaterConfig(
            fix_mass=True, use_polar_filter=True,
            polar_filter_cutoff_deg=60.0, polar_filter_max_wave_speed=300.0,
        )
        model = CGridLatLonShallowWaterModel(grid, config, dt=dt)
        state = williamson_test2_cgrid(grid)
        target_mass = model.compute_mass(state)

        for _ in range(int(6 * 3600 / dt)):
            state = model.step(state, dt, target_mass=target_mass)

        assert jnp.all(jnp.isfinite(state.h))

    def test_polar_filter_builds_v_face_mask(self, grid):
        """Filter ON builds BOTH masks (cell rows for dh/du, v-face rows
        for dv); filter OFF builds neither."""
        dt = 300.0
        cfg_on = CGridLatLonShallowWaterConfig(use_polar_filter=True)
        model_on = CGridLatLonShallowWaterModel(grid, cfg_on, dt=dt)
        n_freq = grid.n_lon // 2 + 1
        assert model_on._polar_mask is not None
        assert model_on._polar_mask.shape == (grid.n_lat, n_freq)
        assert model_on._polar_mask_v is not None
        assert model_on._polar_mask_v.shape == (grid.n_lat + 1, n_freq)
        # The v-face mask must actually truncate at the polar rows.
        assert float(jnp.sum(model_on._polar_mask_v[1] == 0.0)) > 0

        model_off = CGridLatLonShallowWaterModel(
            grid, CGridLatLonShallowWaterConfig(use_polar_filter=False), dt=dt)
        assert model_off._polar_mask is None
        assert model_off._polar_mask_v is None

    def test_polar_filter_damps_v_like_u(self, grid):
        """The per-step v increment must contain NO energy in the zonal
        modes the v-face mask forbids at the polar rows — dv is filtered
        exactly like du/dh.

        Non-vacuous: before the dv filtering fix (dh/du filtered, dv
        not), this same case leaves ~0.16 of spectral amplitude in the
        forbidden band (measured by re-running with an all-pass v mask);
        with the fix it is 0 to float noise.
        """
        dt = 300.0
        config = CGridLatLonShallowWaterConfig(
            fix_mass=True, use_polar_filter=True,
            polar_filter_cutoff_deg=60.0, polar_filter_max_wave_speed=300.0,
        )
        model = CGridLatLonShallowWaterModel(grid, config, dt=dt)
        state = _polar_perturbed_state(grid)
        target_mass = model.compute_mass(state)
        new = model.step(state, dt, target_mass=target_mass)

        rows = (1, 2, -3, -2)
        # v increment: forbidden-band energy per the v-face mask.
        spec_v = jnp.abs(jnp.fft.rfft(new.v - state.v, axis=-1))
        mask_v = model._polar_mask_v
        e_v = sum(
            float(jnp.sum(spec_v[r] * (mask_v[r] == 0.0))) for r in rows
        )
        assert e_v < 1e-8, (
            f"dv increment leaks {e_v:.3e} spectral amplitude into the "
            f"polar-forbidden band — v is not being polar-filtered"
        )
        # u increment (periodic interior columns): same property with the
        # cell-row mask — v is damped LIKE u, not differently.
        spec_u = jnp.abs(jnp.fft.rfft((new.u - state.u)[:, :-1], axis=-1))
        mask_u = model._polar_mask
        e_u = sum(
            float(jnp.sum(spec_u[r] * (mask_u[r] == 0.0))) for r in (1, 2)
        )
        assert e_u < 1e-8

    def test_polar_filter_unbalanced_perturbation_stable(self, grid):
        """6 h at dt=300 (far above the ~80 s pole CFL) from the UNBALANCED
        polar state stays finite and bounded with the filter on."""
        dt = 300.0
        config = CGridLatLonShallowWaterConfig(
            fix_mass=True, use_polar_filter=True,
            polar_filter_cutoff_deg=60.0, polar_filter_max_wave_speed=300.0,
        )
        model = CGridLatLonShallowWaterModel(grid, config, dt=dt)
        state = _polar_perturbed_state(grid)
        target_mass = model.compute_mass(state)
        for _ in range(int(6 * 3600 / dt)):
            state = model.step(state, dt, target_mass=target_mass)
        assert jnp.all(jnp.isfinite(state.h))
        assert jnp.all(jnp.isfinite(state.u))
        assert jnp.all(jnp.isfinite(state.v))
        assert float(jnp.max(jnp.abs(state.v))) < 100.0


# ==============================================================================
# Time integrator flexibility
# ==============================================================================

class TestTimeIntegrators:

    def test_rk4_stable(self, grid):
        config = CGridLatLonShallowWaterConfig(fix_mass=True, time_integrator="rk4")
        model = CGridLatLonShallowWaterModel(grid, config)
        state = williamson_test2_cgrid(grid)
        for _ in range(10):
            state = model.step(state, dt=60.0)
        assert jnp.all(jnp.isfinite(state.h))

    def test_ssp_rk54_stable(self, grid):
        config = CGridLatLonShallowWaterConfig(fix_mass=True, time_integrator="ssp_rk54")
        model = CGridLatLonShallowWaterModel(grid, config)
        state = williamson_test2_cgrid(grid)
        for _ in range(10):
            state = model.step(state, dt=60.0)
        assert jnp.all(jnp.isfinite(state.h))


# ==============================================================================
# C-grid PPM transport
# ==============================================================================

class TestPPMTransport:

    def test_cgrid_ppm_uniform_field(self, grid):
        from legoesm.core.operators_fv_latlon import cgrid_fv_flux_divergence_latlon
        n_lat, n_lon = grid.n_lat, grid.n_lon
        q = jnp.ones((n_lat, n_lon))
        u = jnp.ones((n_lat, n_lon + 1)) * 10.0
        v = jnp.zeros((n_lat + 1, n_lon))
        tend = cgrid_fv_flux_divergence_latlon(q, u, v, grid)
        assert jnp.max(jnp.abs(tend)) < 1e-10


# ==============================================================================
# Biharmonic (del-4) viscosity
# ==============================================================================

class TestBiharmonicViscosity:
    """The nu_del4 term must be scale-selective (unlike A_h), damping
    (sign), pole-capped (stability), and must fail loud without dt."""

    NU4 = 1.0e16  # interior coefficient sized for n_lat=32 (dy ~ 625 km)

    def _isolated_biharmonic_tendency(self, grid, u, v, dt=300.0):
        """Return the biharmonic contribution alone: tend(nu4>0) - tend(nu4=0).

        Both calls share every other term (PGF, Coriolis, KE gradient),
        so the difference isolates -nu4*lap^2(u,v) exactly.
        """
        h = jnp.full((grid.n_lat, grid.n_lon), 5000.0)
        h_s = jnp.zeros_like(h)
        state = CGridLatLonShallowWaterState(h=h, u=u, v=v, h_s=h_s)
        cfg_on = CGridLatLonShallowWaterConfig(nu_del4=self.NU4)
        cfg_off = CGridLatLonShallowWaterConfig(nu_del4=0.0)
        _, du_on, dv_on = cgrid_latlon_sw_tendencies(state, grid, cfg_on, dt)
        _, du_off, dv_off = cgrid_latlon_sw_tendencies(state, grid, cfg_off, dt)
        return du_on - du_off, dv_on - dv_off

    def test_scale_selective(self, grid):
        """Per-unit-amplitude damping of a 2*dx checkerboard must exceed
        that of a zonal-wavenumber-2 mode by orders of magnitude (k^4)."""
        lon_f = jnp.concatenate([grid.lon - 0.5 * grid.dlon,
                                 (grid.lon - 0.5 * grid.dlon)[0:1] + 2 * jnp.pi])
        amp = 1.0e-3  # keep the quadratic KE-gradient term negligible
        u_large = amp * jnp.cos(2.0 * lon_f)[None, :] * jnp.ones((grid.n_lat, 1))
        i = jnp.arange(grid.n_lon + 1)
        j = jnp.arange(grid.n_lat)
        u_noise = amp * ((-1.0) ** (i[None, :] + j[:, None]))
        v0 = jnp.zeros((grid.n_lat + 1, grid.n_lon))

        du_l, _ = self._isolated_biharmonic_tendency(grid, u_large, v0)
        du_n, _ = self._isolated_biharmonic_tendency(grid, u_noise, v0)
        # Compare damping rates on interior rows (pole rows are capped).
        sl = slice(8, 24)
        rate_large = float(jnp.max(jnp.abs(du_l[sl]))) / amp
        rate_noise = float(jnp.max(jnp.abs(du_n[sl]))) / amp
        assert rate_noise > 50.0 * rate_large, (rate_noise, rate_large)

    def test_sign_damps(self, grid):
        """-nu4*lap^2(u) must anticorrelate with u (energy sink)."""
        i = jnp.arange(grid.n_lon + 1)
        j = jnp.arange(grid.n_lat)
        u_noise = 1.0e-3 * ((-1.0) ** (i[None, :] + j[:, None]))
        v0 = jnp.zeros((grid.n_lat + 1, grid.n_lon))
        du, _ = self._isolated_biharmonic_tendency(grid, u_noise, v0)
        assert float(jnp.sum(u_noise * du)) < 0.0

    def test_requires_dt(self, grid):
        state = williamson_test2_cgrid(grid)
        cfg = CGridLatLonShallowWaterConfig(nu_del4=self.NU4)
        with pytest.raises(ValueError, match="requires the"):
            cgrid_latlon_sw_tendencies(state, grid, cfg, None)

    def test_pole_cap_profile(self, grid):
        """Interior rows keep nu_del4; pole-adjacent rows are reduced."""
        from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
            _nu_del4_row_profiles)
        big = 1.0e18  # deliberately above every row's stability cap
        cfg = CGridLatLonShallowWaterConfig(nu_del4=big)
        nu_u, nu_v = _nu_del4_row_profiles(grid, cfg, dt=300.0)
        assert nu_u.shape == (grid.n_lat, 1)
        assert nu_v.shape == (grid.n_lat + 1, 1)
        # Every row obeys the cap on BOTH staggerings:
        # nu * dt * lam^2 <= cfl_frac.
        dy = grid.radius * grid.dlat
        dx_u = grid.radius * grid.dlon * grid.cos_lat
        dx_v = grid.radius * grid.dlon * grid.cos_lat_v
        lam_u = 4.0 / dx_u ** 2 + 4.0 / dy ** 2
        lam_v = 4.0 / dx_v ** 2 + 4.0 / dy ** 2
        # fp32-safe tolerance: capped rows sit EXACTLY at the bound, and
        # the grid/profile arrays are stored fp32 under the default
        # precision policy, so the recomputed product can exceed the
        # bound by ~1e-7 relative rounding.
        assert jnp.all(nu_u[:, 0] * 300.0 * lam_u ** 2
                       <= cfg.nu_del4_cfl_frac * (1 + 1e-5))
        assert jnp.all(nu_v[:, 0] * 300.0 * lam_v ** 2
                       <= cfg.nu_del4_cfl_frac * (1 + 1e-5))
        # Pole-adjacent rows are strictly below the equatorial rows, and
        # the exact-pole v rows (cos clamped to 1e-10) are ~zero.
        assert float(nu_u[0, 0]) < float(nu_u[grid.n_lat // 2, 0])
        assert float(nu_v[0, 0]) < 1e-6 * float(nu_v[grid.n_lat // 2, 0])
        assert float(nu_v[-1, 0]) < 1e-6 * float(nu_v[grid.n_lat // 2, 0])
        # Every value is finite (no inf leaking from the pole clamps).
        assert jnp.all(jnp.isfinite(nu_u)) and jnp.all(jnp.isfinite(nu_v))

    def test_checkerboard_decay_rate_matches_discrete_eigenvalue(self, grid):
        """Absolute-rate check (codex review): for the 2-D checkerboard,
        the isolated biharmonic tendency must equal -nu*lam^2*u with
        lam = 4/dx^2 + 4/dy^2 on near-equator rows (dx ~ dy there and
        the metric is locally uniform)."""
        i = jnp.arange(grid.n_lon + 1)
        j = jnp.arange(grid.n_lat)
        amp = 1.0e-3
        u_noise = amp * ((-1.0) ** (i[None, :] + j[:, None]))
        v0 = jnp.zeros((grid.n_lat + 1, grid.n_lon))
        du, _ = self._isolated_biharmonic_tendency(grid, u_noise, v0)
        # Predicted rate on the equator-adjacent rows.
        dy = float(grid.radius * grid.dlat)
        row = grid.n_lat // 2
        dx = float(grid.radius * grid.dlon * grid.cos_lat[row])
        lam = 4.0 / dx ** 2 + 4.0 / dy ** 2
        predicted = self.NU4 * lam ** 2
        measured = float(jnp.abs(du[row, grid.n_lon // 2])) / amp
        assert 0.5 * predicted < measured < 1.5 * predicted, (
            measured, predicted)

    def test_tendencies_noarg_falls_back_to_constructor_dt(self, grid):
        """model.tendencies(state) without dt must
        work with nu_del4 > 0 via the constructor-dt fallback (codex
        iter-3)."""
        cfg = CGridLatLonShallowWaterConfig(nu_del4=self.NU4)
        model = CGridLatLonShallowWaterModel(grid, cfg, dt=300.0)
        state = williamson_test2_cgrid(grid)
        dh, du, dv = model.tendencies(state)
        assert jnp.all(jnp.isfinite(dh))
        assert jnp.all(jnp.isfinite(du))
        assert jnp.all(jnp.isfinite(dv))

    def test_step_stable_with_polar_grid_noise(self, grid):
        """Seeded grid-scale noise in the pole-adjacent rows must stay
        finite under an above-cap coefficient: without the row cap the
        polar rows violate the del-4 CFL within a few steps."""
        cfg = CGridLatLonShallowWaterConfig(nu_del4=1.0e18, fix_mass=True)
        model = CGridLatLonShallowWaterModel(grid, cfg)
        state = williamson_test2_cgrid(grid)
        i = jnp.arange(grid.n_lon + 1)
        j = jnp.arange(grid.n_lat)
        noise = 1.0 * ((-1.0) ** (i[None, :] + j[:, None]))
        band = ((j < 3) | (j >= grid.n_lat - 3)).astype(noise.dtype)
        state = state._replace(u=state.u + noise * band[:, None])
        for _ in range(10):
            state = model.step(state, dt=60.0)
        assert jnp.all(jnp.isfinite(state.h))
        assert jnp.all(jnp.isfinite(state.u))
        assert jnp.all(jnp.isfinite(state.v))

    def test_step_stable_with_biharmonic(self, grid):
        """W2 steps finitely with an above-cap coefficient (pole rows
        would blow up within a few steps without the row cap)."""
        cfg = CGridLatLonShallowWaterConfig(
            nu_del4=1.0e18, fix_mass=True)
        model = CGridLatLonShallowWaterModel(grid, cfg)
        state = williamson_test2_cgrid(grid)
        for _ in range(10):
            state = model.step(state, dt=60.0)
        assert jnp.all(jnp.isfinite(state.h))
        assert jnp.all(jnp.isfinite(state.u))
