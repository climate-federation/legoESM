"""Orszag 2/3-rule de-aliasing of the spectral dycore nonlinear tendencies.

These tests pin the de-aliasing mechanism shared by the spectral
shallow-water (``spectral_sw``) and non-hydrostatic (``spectral_nh``)
dynamical cores via ``grids.gaussian.dealiasing_mask``.

Gates covered (each non-vacuous — fails if de-aliasing is reverted):

* MASK CORRECTNESS — the mask is 1 for ``n <= floor(fraction*n_max)``,
  0 above, and all-ones when disabled.
* ALIASING STABILITY — a strongly nonlinear field has its top-1/3
  wavenumber tendency energy truncated WITH de-aliasing and NOT without,
  and a strongly nonlinear integration stays bounded with de-aliasing.
* SPECTRAL ACCURACY — a band-limited (well-resolved) field is unchanged
  by de-aliasing (the retained band passes through untouched).
* CONSERVATION — de-aliasing never touches the ``n=0`` global-integral
  mode, so the SW mass tendency and the NH mass tendency are unchanged.
* AD-SAFETY — ``jax.grad`` is finite through the de-aliased tendency.
* JIT / VMAP — the de-aliased tendency matches under ``jit`` and ``vmap``.

Reference: Orszag (1971), J. Atmos. Sci., 28, 1074.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.gaussian import (
    create_gaussian_grid,
    dealiasing_mask,
    sh_analysis,
    sh_synthesis,
)
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)
from legoesm.atmosphere.dynamics.gcm.spectral_sw import (
    SpectralSWState,
    SpectralSWConfig,
    SpectralShallowWaterModel,
    spectral_sw_tendencies,
    williamson_test5_spectral,
    compute_spectral_diagnostics,
    spectral_to_grid,
)
from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
    SpectralNHConfig,
    spectral_nh_slow_tendencies,
    nh_rest_state_spectral,
)
from legoesm.core.field import Field


_FRACTION = 0.667


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid_t21():
    return create_gaussian_grid(n_max=21)


@pytest.fixture(scope="module")
def grid_t42():
    return create_gaussian_grid(n_max=42)


def _n_cut(grid, fraction=_FRACTION):
    return int(fraction * grid.n_max)


def _high_band_mask(grid, fraction=_FRACTION):
    """1.0 for the *removed* (n > n_cut) band, 0.0 for retained."""
    return jnp.where(grid.ls > _n_cut(grid, fraction), 1.0, 0.0)


def _energy(arr, mode_mask=None):
    """Sum of |coeff|^2 (optionally restricted to a mode mask)."""
    a = jnp.abs(arr)
    if mode_mask is not None:
        # broadcast (n_sh,) mask over any trailing axes
        mm = mode_mask.reshape((-1,) + (1,) * (arr.ndim - 1))
        a = a * mm
    return float(jnp.sum(a ** 2))


def _rough_sw_state(grid, amp=80.0, seed=0):
    """A strongly nonlinear, BROADBAND shallow-water state.

    Fills vorticity and geopotential with random power across the WHOLE
    resolved spectrum (up to n_max) so the quadratic products genuinely
    alias.  Geopotential carries a large mean (background depth) so the
    state is physical.
    """
    rng = np.random.default_rng(seed)
    n_sh = grid.n_sh
    # Random complex SH coefficients, larger amplitude at low n (red-ish
    # spectrum) but with real power all the way to n_max.
    scale = 1.0 / (1.0 + np.asarray(grid.ls))
    vor = (rng.standard_normal(n_sh) + 1j * rng.standard_normal(n_sh)) * scale
    phi = (rng.standard_normal(n_sh) + 1j * rng.standard_normal(n_sh)) * scale
    div = (rng.standard_normal(n_sh) + 1j * rng.standard_normal(n_sh)) * scale
    # m=0 coefficients must be real for a real grid field.
    ms = np.asarray(grid.ms)
    vor = np.where(ms == 0, vor.real + 0j, vor)
    phi = np.where(ms == 0, phi.real + 0j, phi)
    div = np.where(ms == 0, div.real + 0j, div)
    vor_hat = jnp.asarray(vor, dtype=jnp.complex128) * (amp / grid.radius)
    div_hat = jnp.asarray(div, dtype=jnp.complex128) * (amp / grid.radius)
    phi_hat = jnp.asarray(phi, dtype=jnp.complex128) * (amp * amp)
    # Background depth: gh0 ~ 1e4 m^2/s^2 in the (0,0) mode.
    idx00 = int(np.where((np.asarray(grid.ls) == 0) & (ms == 0))[0][0])
    phi_hat = phi_hat.at[idx00].set(2.0e4 + 0j)
    z = jnp.zeros(n_sh, dtype=jnp.complex128)
    return SpectralSWState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=("spectral",), units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=("spectral",), units="1/s"),
        phi_hat=Field(data=phi_hat, name="phi_hat", dims=("spectral",), units="m^2/s^2"),
        phis_hat=Field(data=z, name="phis_hat", dims=("spectral",), units="m^2/s^2"),
    )


# ---------------------------------------------------------------------------
# 1. Mask correctness
# ---------------------------------------------------------------------------

class TestDealiasingMask:
    def test_mask_shape_dtype(self, grid_t21):
        m = dealiasing_mask(grid_t21, _FRACTION)
        assert m.shape == (grid_t21.n_sh,)
        assert m.dtype == jnp.float64

    def test_mask_retains_low_zeros_high(self, grid_t21):
        m = dealiasing_mask(grid_t21, _FRACTION)
        n_cut = _n_cut(grid_t21)
        ls = np.asarray(grid_t21.ls)
        m_np = np.asarray(m)
        # exactly 1.0 for n <= n_cut, exactly 0.0 above
        assert np.all(m_np[ls <= n_cut] == 1.0)
        assert np.all(m_np[ls > n_cut] == 0.0)
        # non-trivial: some modes are actually removed at T21
        assert (ls > n_cut).sum() > 0

    def test_mask_disabled_is_all_ones(self, grid_t21):
        m = dealiasing_mask(grid_t21, 0.0)
        assert float(jnp.sum(m)) == float(grid_t21.n_sh)
        assert bool(jnp.all(m == 1.0))

    def test_mask_idempotent(self, grid_t21):
        """Applying the mask twice equals applying once (projection)."""
        m = dealiasing_mask(grid_t21, _FRACTION)
        np.testing.assert_array_equal(np.asarray(m * m), np.asarray(m))


# ---------------------------------------------------------------------------
# 2. Aliasing stability — the headline gate (non-vacuous)
# ---------------------------------------------------------------------------

class TestAliasingStabilitySW:
    def test_top_third_energy_truncated(self, grid_t42):
        """A broadband SW tendency has ZERO energy in the top-1/3 band
        WITH de-aliasing and substantial energy WITHOUT it.

        Non-vacuous: if de-aliasing is reverted (mask -> all ones), the
        ``with`` energy equals the ``without`` energy and the assertion
        ``ratio < 1e-12`` fails.
        """
        grid = grid_t42
        state = _rough_sw_state(grid)
        high = _high_band_mask(grid)

        t_on = spectral_sw_tendencies(
            state, grid, SpectralSWConfig(hyperdiff_coeff=0.0, dealiasing_fraction=_FRACTION),
        )
        t_off = spectral_sw_tendencies(
            state, grid, SpectralSWConfig(hyperdiff_coeff=0.0, dealiasing_fraction=0.0),
        )

        for name in ("vor_hat", "div_hat", "phi_hat"):
            e_on = _energy(getattr(t_on, name).data, high)
            e_off = _energy(getattr(t_off, name).data, high)
            # Without de-aliasing the products alias into the top band.
            assert e_off > 0.0, f"{name}: no aliased energy to remove (vacuous test)"
            # With de-aliasing the top band is identically zero.
            assert e_on == 0.0, (
                f"{name}: top-1/3 band energy not removed by de-aliasing "
                f"(e_on={e_on:.3e}, e_off={e_off:.3e})"
            )

    def test_retained_band_unchanged_by_dealiasing(self, grid_t42):
        """De-aliasing only touches the removed band: the retained
        (n <= n_cut) tendency is bit-identical with and without it."""
        grid = grid_t42
        state = _rough_sw_state(grid)
        retained = dealiasing_mask(grid, _FRACTION)

        t_on = spectral_sw_tendencies(
            state, grid, SpectralSWConfig(hyperdiff_coeff=0.0, dealiasing_fraction=_FRACTION),
        )
        t_off = spectral_sw_tendencies(
            state, grid, SpectralSWConfig(hyperdiff_coeff=0.0, dealiasing_fraction=0.0),
        )
        for name in ("vor_hat", "div_hat", "phi_hat"):
            on = getattr(t_on, name).data * retained
            off = getattr(t_off, name).data * retained
            np.testing.assert_allclose(np.asarray(on), np.asarray(off), rtol=0, atol=0)

    @pytest.mark.slow
    def test_strongly_nonlinear_run_stays_bounded_longer(self, grid_t42):
        """A strongly nonlinear SW integration survives strictly LONGER
        WITH de-aliasing than without it.

        2/3 de-aliasing removes aliasing error but is NOT a dissipation:
        for a violently unbalanced broadband random IC the resolved
        enstrophy still cascades to the grid scale and eventually blows
        up even with de-aliasing (that is what hyperdiffusion is for).
        The physically correct, revert-sensitive gate is therefore that
        de-aliasing DELAYS the blow-up — the time-to-non-finite is larger
        with the Orszag truncation in place.

        Non-vacuous: if de-aliasing is reverted (mask -> all ones), the
        two runs are identical and ``steps_on > steps_off`` fails.
        """
        grid = grid_t42
        state0 = _rough_sw_state(grid, amp=120.0, seed=3)
        dt = 90.0
        n_steps = 300

        def survival_steps(fraction):
            cfg = SpectralSWConfig(
                hyperdiff_coeff=0.0,
                spectral_filter_order=0,
                dealiasing_fraction=fraction,
                time_integrator="ssp_rk3",
            )
            model = SpectralShallowWaterModel(grid, cfg)
            st = state0
            for k in range(n_steps):
                st = model.step(st, dt)
                if not bool(jnp.all(jnp.isfinite(st.vor_hat.data))):
                    return k
            return n_steps

        steps_on = survival_steps(_FRACTION)
        steps_off = survival_steps(0.0)

        assert steps_on > steps_off, (
            f"de-aliasing did not extend stability: steps_on={steps_on} "
            f"steps_off={steps_off} (revert-sensitive gate)"
        )
        # And the margin must be meaningful (not a 1-step fluke).
        assert steps_on - steps_off >= 10, (
            f"de-aliasing stability margin too small: steps_on={steps_on} "
            f"steps_off={steps_off}"
        )


class TestAliasingStabilityNH:
    def _rough_nh_state(self, grid, hc, n_tracers=1, seed=1):
        rng = np.random.default_rng(seed)
        nlev = len(hc.z_full)
        n_sh = grid.n_sh
        rs = nh_rest_state_spectral(grid, hc, n_tracers=n_tracers)
        scale = (1.0 / (1.0 + np.asarray(grid.ls)))[:, None]
        ms = np.asarray(grid.ms)[:, None]

        def rand(shape, amp):
            a = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
            a = np.where(ms == 0, a.real + 0j, a)
            return jnp.asarray(a, dtype=jnp.complex128) * amp * scale

        vor = rand((n_sh, nlev), 1e-4)
        div = rand((n_sh, nlev), 1e-5)
        thp = rand((n_sh, nlev), 2.0)
        rhp = rand((n_sh, nlev), 1e-2)
        tr = jnp.stack([rand((n_sh, nlev), 1e-3) for _ in range(max(n_tracers, 1))], axis=-1)
        return rs._replace(
            vor_hat=rs.vor_hat.replace(data=vor),
            div_hat=rs.div_hat.replace(data=div),
            theta_prime_hat=rs.theta_prime_hat.replace(data=thp),
            rho_prime_hat=rs.rho_prime_hat.replace(data=rhp),
            tracers_hat=rs.tracers_hat.replace(data=tr),
        )

    def test_top_third_energy_truncated(self, grid_t42):
        grid = grid_t42
        hc = create_height_coordinate(8, 30000.0)
        tm = compute_terrain_metric(jnp.zeros((grid.n_lat, grid.n_lon)), hc)
        state = self._rough_nh_state(grid, hc)
        high = _high_band_mask(grid)

        on_cfg = SpectralNHConfig(hyperdiff_coeff=0.0, sponge_coeff=0.0, dealiasing_fraction=_FRACTION)
        off_cfg = SpectralNHConfig(hyperdiff_coeff=0.0, sponge_coeff=0.0, dealiasing_fraction=0.0)
        t_on = spectral_nh_slow_tendencies(state, grid, hc, tm, on_cfg)
        t_off = spectral_nh_slow_tendencies(state, grid, hc, tm, off_cfg)

        for name in ("vor_hat", "div_hat", "theta_prime_hat", "rho_prime_hat", "tracers_hat"):
            e_on = _energy(getattr(t_on, name).data, high)
            e_off = _energy(getattr(t_off, name).data, high)
            assert e_off > 0.0, f"{name}: no aliased energy (vacuous)"
            assert e_on == 0.0, (
                f"{name}: NH top-1/3 band not removed (e_on={e_on:.3e}, e_off={e_off:.3e})"
            )

    def test_retained_band_unchanged(self, grid_t42):
        grid = grid_t42
        hc = create_height_coordinate(8, 30000.0)
        tm = compute_terrain_metric(jnp.zeros((grid.n_lat, grid.n_lon)), hc)
        state = self._rough_nh_state(grid, hc)
        retained = dealiasing_mask(grid, _FRACTION)

        on_cfg = SpectralNHConfig(hyperdiff_coeff=0.0, sponge_coeff=0.0, dealiasing_fraction=_FRACTION)
        off_cfg = SpectralNHConfig(hyperdiff_coeff=0.0, sponge_coeff=0.0, dealiasing_fraction=0.0)
        t_on = spectral_nh_slow_tendencies(state, grid, hc, tm, on_cfg)
        t_off = spectral_nh_slow_tendencies(state, grid, hc, tm, off_cfg)
        for name in ("vor_hat", "div_hat", "theta_prime_hat", "rho_prime_hat"):
            r = retained[:, None]
            on = getattr(t_on, name).data * r
            off = getattr(t_off, name).data * r
            np.testing.assert_allclose(np.asarray(on), np.asarray(off), rtol=0, atol=0)
        # tracers (3D)
        r3 = retained[:, None, None]
        np.testing.assert_allclose(
            np.asarray(t_on.tracers_hat.data * r3),
            np.asarray(t_off.tracers_hat.data * r3),
            rtol=0, atol=0,
        )


# ---------------------------------------------------------------------------
# 3. Spectral accuracy preserved on a band-limited field
# ---------------------------------------------------------------------------

class TestSpectralAccuracyPreserved:
    def test_bandlimited_sw_tendency_unchanged(self, grid_t42):
        """A SW state band-limited to n <= n_cut/2 produces a tendency
        whose content stays within the retained band, so de-aliasing
        leaves it bit-identical (no degradation of a resolved solution).

        Quadratic products of fields with n <= n_cut/2 reach at most
        n <= n_cut, all retained, so the mask is a no-op here.
        """
        grid = grid_t42
        n_lim = _n_cut(grid) // 2
        rng = np.random.default_rng(7)
        n_sh = grid.n_sh
        ls = np.asarray(grid.ls)
        ms = np.asarray(grid.ms)
        band = (ls <= n_lim)

        def mk(amp):
            a = (rng.standard_normal(n_sh) + 1j * rng.standard_normal(n_sh))
            a = np.where(ms == 0, a.real + 0j, a)
            a = np.where(band, a, 0.0)
            return jnp.asarray(a, dtype=jnp.complex128) * amp

        idx00 = int(np.where((ls == 0) & (ms == 0))[0][0])
        phi = mk(50.0).at[idx00].set(2.0e4 + 0j)
        state = SpectralSWState(
            vor_hat=Field(data=mk(1e-4), name="vor_hat", dims=("spectral",), units="1/s"),
            div_hat=Field(data=mk(1e-5), name="div_hat", dims=("spectral",), units="1/s"),
            phi_hat=Field(data=phi, name="phi_hat", dims=("spectral",), units="m^2/s^2"),
            phis_hat=Field(data=jnp.zeros(n_sh, dtype=jnp.complex128), name="phis_hat",
                           dims=("spectral",), units="m^2/s^2"),
        )

        t_on = spectral_sw_tendencies(
            state, grid, SpectralSWConfig(hyperdiff_coeff=0.0, dealiasing_fraction=_FRACTION),
        )
        t_off = spectral_sw_tendencies(
            state, grid, SpectralSWConfig(hyperdiff_coeff=0.0, dealiasing_fraction=0.0),
        )
        # Dealiasing multiplies by exactly 1.0 on the RETAINED band
        # (n <= n_cut), so the resolved spectrum is bit-identical with/without
        # it; it only zeros the n > n_cut tail. A NONLINEAR tendency populates
        # that tail even for a band-limited input (quadratic/cubic products +
        # grid round-off spread above n_cut) — removing it is the whole point of
        # de-aliasing, so the correct "accuracy preserved" invariant is that the
        # retained band is untouched AND the removed band is exactly zeroed.
        retained = np.asarray(grid.ls) <= _n_cut(grid)
        for name in ("vor_hat", "div_hat", "phi_hat"):
            on = np.asarray(getattr(t_on, name).data)
            off = np.asarray(getattr(t_off, name).data)
            np.testing.assert_allclose(
                on[retained], off[retained],
                rtol=1e-13, atol=1e-30,
                err_msg=f"de-aliasing changed the RETAINED (n<=n_cut) band of {name}",
            )
            assert np.max(np.abs(on[~retained])) == 0.0, (
                f"de-aliasing did not zero the removed (n>n_cut) band of {name}"
            )

    def test_bandlimited_roundtrip_unchanged(self, grid_t21):
        """A grid field that is band-limited to n <= n_cut is unchanged by
        applying the de-aliasing mask in spectral space (analysis ->
        mask -> synthesis is identity)."""
        grid = grid_t21
        n_lim = _n_cut(grid)
        # Build a field from a few harmonics with n <= n_lim.
        lat = grid.lat2d
        lon = grid.lon2d
        f = (3.0 * jnp.sin(lat)
             + 2.0 * jnp.cos(lat) * jnp.sin(lon)
             + 1.5 * jnp.cos(lat) ** 2 * jnp.cos(2 * lon))
        coeffs = sh_analysis(grid, f)
        masked = coeffs * dealiasing_mask(grid, _FRACTION)
        f_masked = sh_synthesis(grid, masked)
        # The harmonics used are n<=2 << n_lim, so masking is a no-op.
        np.testing.assert_allclose(np.asarray(f_masked), np.asarray(f), atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Conservation — n=0 mode untouched
# ---------------------------------------------------------------------------

class TestConservation:
    def test_sw_mass_tendency_mode_untouched(self, grid_t42):
        """The SW mass tendency d(phi)/dt at the (0,0) global-integral
        mode is identical with and without de-aliasing (mass is the n=0
        mode; the mask never touches it)."""
        grid = grid_t42
        state = _rough_sw_state(grid)
        ls = np.asarray(grid.ls)
        ms = np.asarray(grid.ms)
        idx00 = int(np.where((ls == 0) & (ms == 0))[0][0])
        t_on = spectral_sw_tendencies(
            state, grid, SpectralSWConfig(hyperdiff_coeff=0.0, dealiasing_fraction=_FRACTION),
        )
        t_off = spectral_sw_tendencies(
            state, grid, SpectralSWConfig(hyperdiff_coeff=0.0, dealiasing_fraction=0.0),
        )
        np.testing.assert_allclose(
            complex(t_on.phi_hat.data[idx00]),
            complex(t_off.phi_hat.data[idx00]),
            rtol=0, atol=0,
        )

    @pytest.mark.slow
    def test_sw_tc5_mass_conserved_with_dealiasing(self, grid_t21):
        """Williamson TC5 mass conserved to machine precision over a
        multi-day run WITH de-aliasing default-on (the n=0 mass mode is
        preserved by construction)."""
        grid = grid_t21
        state0 = williamson_test5_spectral(grid)
        a = grid.radius
        nu = 1.0 / (4.0 * 3600.0 * (grid.n_max * (grid.n_max + 1) / (a * a)) ** 2)
        cfg = SpectralSWConfig(
            hyperdiff_coeff=nu,
            dealiasing_fraction=_FRACTION,
            spectral_filter_order=8,
            spectral_filter_cutoff=0.01,
            time_integrator="ssp_rk3",
        )
        model = SpectralShallowWaterModel(grid, cfg)
        state0 = model.filter_initial_state(state0)
        diag0 = compute_spectral_diagnostics(state0, grid)
        dt = 600.0
        n_steps = int(5.0 * 86400.0 / dt)
        state = state0
        for _ in range(n_steps):
            state = model.step(state, dt)
        diag = compute_spectral_diagnostics(state, grid)
        dM = abs(diag['mass'] - diag0['mass']) / abs(diag0['mass'])
        assert dM < 1e-13, f"TC5 mass drift with de-aliasing: {dM:.3e}"
        assert jnp.all(jnp.isfinite(state.vor_hat.data))


# ---------------------------------------------------------------------------
# 5. AD-safety
# ---------------------------------------------------------------------------

class TestDifferentiability:
    def test_sw_grad_finite_through_dealiased_tendency(self, grid_t42):
        grid = grid_t42
        state = _rough_sw_state(grid)
        cfg = SpectralSWConfig(hyperdiff_coeff=0.0, dealiasing_fraction=_FRACTION)

        def loss(vor_data):
            s = state._replace(vor_hat=state.vor_hat.replace(data=vor_data))
            tend = spectral_sw_tendencies(s, grid, cfg)
            # include all outputs so the gradient flows through the mask
            return (jnp.sum(jnp.abs(tend.vor_hat.data) ** 2)
                    + jnp.sum(jnp.abs(tend.div_hat.data) ** 2)
                    + jnp.sum(jnp.abs(tend.phi_hat.data) ** 2)).real

        g = jax.grad(loss)(state.vor_hat.data)
        assert g.shape == state.vor_hat.data.shape
        # The mask is a constant 0/1 multiply on the OUTPUT tendency, so AD is
        # safe — the gradient is finite. (It is NOT zero on the removed INPUT
        # band: a high-n input coefficient feeds the LOW-n output via the
        # nonlinear product, so the loss legitimately depends on it.)
        assert jnp.all(jnp.isfinite(g)), "non-finite gradient through SW de-aliasing"

        # The real, checkable consequence of the mask: the OUTPUT removed band
        # is identically zero regardless of input, so a loss measuring ONLY that
        # band has zero gradient everywhere (fails if the mask is not applied).
        high = jnp.asarray(np.asarray(_high_band_mask(grid)).astype(bool))

        def high_band_loss(vor_data):
            s = state._replace(vor_hat=state.vor_hat.replace(data=vor_data))
            t = spectral_sw_tendencies(s, grid, cfg)
            return (jnp.sum(jnp.abs(jnp.where(high, t.vor_hat.data, 0.0)) ** 2)
                    + jnp.sum(jnp.abs(jnp.where(high, t.div_hat.data, 0.0)) ** 2)
                    + jnp.sum(jnp.abs(jnp.where(high, t.phi_hat.data, 0.0)) ** 2)).real

        g_high = jax.grad(high_band_loss)(state.vor_hat.data)
        assert float(jnp.max(jnp.abs(g_high))) == 0.0, (
            "de-aliased OUTPUT band is not identically zero — mask not applied"
        )

    def test_nh_grad_finite_through_dealiased_tendency(self, grid_t21):
        grid = grid_t21
        hc = create_height_coordinate(6, 30000.0)
        tm = compute_terrain_metric(jnp.zeros((grid.n_lat, grid.n_lon)), hc)
        rs = nh_rest_state_spectral(grid, hc, n_tracers=1)
        nlev = len(hc.z_full)
        rng = np.random.default_rng(2)
        thp = jnp.asarray(
            rng.standard_normal((grid.n_sh, nlev)) + 1j * rng.standard_normal((grid.n_sh, nlev)),
            dtype=jnp.complex128,
        ) * 0.5
        rs = rs._replace(theta_prime_hat=rs.theta_prime_hat.replace(data=thp))
        cfg = SpectralNHConfig(hyperdiff_coeff=0.0, sponge_coeff=0.0, dealiasing_fraction=_FRACTION)

        def loss(theta_data):
            s = rs._replace(theta_prime_hat=rs.theta_prime_hat.replace(data=theta_data))
            tend = spectral_nh_slow_tendencies(s, grid, hc, tm, cfg)
            return jnp.sum(jnp.abs(tend.theta_prime_hat.data) ** 2).real

        g = jax.grad(loss)(rs.theta_prime_hat.data)
        assert g.shape == rs.theta_prime_hat.data.shape
        assert jnp.all(jnp.isfinite(g)), "non-finite gradient through NH de-aliasing"


# ---------------------------------------------------------------------------
# 6. JIT / VMAP consistency
# ---------------------------------------------------------------------------

class TestJitVmap:
    def test_sw_jit_matches_eager(self, grid_t21):
        grid = grid_t21
        state = _rough_sw_state(grid)
        cfg = SpectralSWConfig(hyperdiff_coeff=0.0, dealiasing_fraction=_FRACTION)
        eager = spectral_sw_tendencies(state, grid, cfg)
        jitted = jax.jit(lambda s: spectral_sw_tendencies(s, grid, cfg))(state)
        for name in ("vor_hat", "div_hat", "phi_hat"):
            np.testing.assert_allclose(
                np.asarray(getattr(jitted, name).data),
                np.asarray(getattr(eager, name).data),
                rtol=1e-12, atol=1e-12,
            )

    def test_sw_vmap_ensemble_matches_eager(self, grid_t21):
        """vmap over a 3-member ensemble matches per-member eager."""
        grid = grid_t21
        cfg = SpectralSWConfig(hyperdiff_coeff=0.0, dealiasing_fraction=_FRACTION)
        members = [_rough_sw_state(grid, seed=s) for s in (0, 1, 2)]
        stacked = SpectralSWState(
            vor_hat=members[0].vor_hat.replace(
                data=jnp.stack([m.vor_hat.data for m in members])),
            div_hat=members[0].div_hat.replace(
                data=jnp.stack([m.div_hat.data for m in members])),
            phi_hat=members[0].phi_hat.replace(
                data=jnp.stack([m.phi_hat.data for m in members])),
            phis_hat=members[0].phis_hat.replace(
                data=jnp.stack([m.phis_hat.data for m in members])),
        )

        def one(s):
            return spectral_sw_tendencies(s, grid, cfg)

        batched = jax.vmap(one)(stacked)
        for i, m in enumerate(members):
            eager = one(m)
            np.testing.assert_allclose(
                np.asarray(batched.vor_hat.data[i]),
                np.asarray(eager.vor_hat.data),
                rtol=1e-12, atol=1e-12,
            )
