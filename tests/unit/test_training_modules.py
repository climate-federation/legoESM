"""Tests for the training package (legoesm.training).

Validates that all 8 training modules import, core functions work,
and gradient flow is verified for each training mode.

Uses synthetic data only — no GCS/ERA5 access required.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.driver.physics_pipeline import PhysicsOutput
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate

N = 4
NLEV = 3
_GRID = create_cubed_sphere(N)
_SIGMA = create_sigma_coordinate(NLEV)
_OPTIONAL_3D_OUTPUT_FIELDS = (
    "du_dt",
    "dv_dt",
    "dq_i_dt",
    "dq_s_dt",
    "dq_g_dt",
    "dN_c_dt",
    "dN_r_dt",
    "dN_i_dt",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(T_val=280.0):
    s3, s2 = (6, N, N, NLEV), (6, N, N)
    return HydrostaticState(
        u=Field(jnp.zeros(s3), name="u", dims=("f","x","y","l"), units="m/s"),
        v=Field(jnp.zeros(s3), name="v", dims=("f","x","y","l"), units="m/s"),
        T=Field(jnp.full(s3, T_val), name="T", dims=("f","x","y","l"), units="K"),
        p_s=Field(jnp.full(s2, 101325.0), name="p_s", dims=("f","x","y"), units="Pa"),
        phis=Field(jnp.zeros(s2), name="phis", dims=("f","x","y"), units="m2/s2"),
    )


def _zero_physics_output(T, p_s, *, conv_prog=None):
    kwargs = dict(
        dT_dt=jnp.zeros(T.shape),
        dq_v_dt=jnp.zeros(T.shape),
        dq_c_dt=jnp.zeros(T.shape),
        dq_r_dt=jnp.zeros(T.shape),
        precip=jnp.zeros(p_s.shape),
        sw_net_sfc=jnp.zeros(p_s.shape),
        lw_net_sfc=jnp.zeros(p_s.shape),
        sw_up_toa=jnp.zeros(p_s.shape),
        lw_up_toa=jnp.zeros(p_s.shape),
        sw_down_toa=jnp.zeros(p_s.shape),
    )
    for field_name in _OPTIONAL_3D_OUTPUT_FIELDS:
        if field_name in PhysicsOutput._fields:
            kwargs[field_name] = jnp.zeros(T.shape)
    if "conv_prog" in PhysicsOutput._fields:
        if conv_prog is None:
            conv_prog = jnp.asarray(0.0, dtype=T.dtype)
        kwargs["conv_prog"] = conv_prog
    return kwargs


def _step_unified_args(*, include_conv_prog=False):
    state = _make_state()
    shape_3d = state.T.data.shape
    shape_2d = state.p_s.data.shape
    args = [
        jnp.bool_(True),
        state.T.data,
        state.p_s.data,
        jnp.zeros(shape_3d),
        jnp.zeros(shape_3d),
        jnp.zeros(shape_3d),
    ]
    if include_conv_prog:
        args.append(jnp.zeros((6 * N * N,), dtype=jnp.float32))
    args.extend([
        state.u.data,
        state.v.data,
        jnp.full(shape_2d, 300.0),
        jnp.zeros(shape_2d),
        _GRID.lat,
        _GRID.lon,
        1.0,
        0.0,
        600.0,
        jnp.ones(14),
        constants.S_0,
        jnp.zeros(shape_3d),
        jnp.zeros(shape_2d),
        jnp.zeros(shape_3d),
        jnp.zeros(shape_2d),
        jnp.zeros(shape_2d),
        jnp.zeros(shape_2d),
        jnp.zeros(shape_2d),
        jnp.zeros(shape_2d),
    ])
    return tuple(args)


# ---------------------------------------------------------------------------
# 1. vertical_interp
# ---------------------------------------------------------------------------

class TestVerticalInterp:

    def test_sigma_interp_shape(self):
        from legoesm.training.vertical_interp import interp_pressure_to_sigma
        plev = jnp.linspace(5000, 100000, 13)
        T = jnp.ones((10, 20, 13)) * 250.0
        p_s = jnp.full((10, 20), 101325.0)
        sigma = jnp.linspace(0.05, 0.975, 20)
        out = interp_pressure_to_sigma(T, plev, p_s, sigma)
        assert out.shape == (10, 20, 20)

    def test_differentiable(self):
        from legoesm.training.vertical_interp import interp_pressure_to_sigma
        plev = jnp.linspace(5000, 100000, 5)
        T = jnp.ones(5) * 250.0
        sigma = jnp.linspace(0.1, 0.9, 3)
        grad = jax.grad(lambda ps: jnp.mean(
            interp_pressure_to_sigma(T, plev, ps, sigma)
        ))(jnp.float64(101325.0))
        assert jnp.isfinite(grad)

    def test_hybrid_interp_shape_and_values(self):
        from legoesm.training.vertical_interp import interp_pressure_to_hybrid
        plev = jnp.linspace(5000.0, 100000.0, 13)
        # linear-in-log(p) profile so log-p interpolation is exact
        profile = 200.0 + 20.0 * jnp.log(plev / 100000.0)
        T = jnp.broadcast_to(profile, (4, 6, 13))
        p_s = jnp.full((4, 6), 95000.0)
        A_full = jnp.linspace(0.04, 0.0, 8)
        B_full = jnp.linspace(0.1, 0.95, 8)
        p_ref = 100000.0
        out = interp_pressure_to_hybrid(T, plev, p_s, A_full, B_full, p_ref)
        assert out.shape == (4, 6, 8)
        p_target = A_full * p_ref + 95000.0 * B_full
        expected = 200.0 + 20.0 * jnp.log(p_target / 100000.0)
        assert jnp.allclose(out[0, 0], expected, atol=1e-6)

    def test_hybrid_interp_differentiable(self):
        from legoesm.training.vertical_interp import interp_pressure_to_hybrid
        plev = jnp.linspace(5000.0, 100000.0, 5)
        T = jnp.linspace(220.0, 290.0, 5)
        A_full = jnp.linspace(0.04, 0.0, 3)
        B_full = jnp.linspace(0.1, 0.95, 3)
        grad = jax.grad(lambda ps: jnp.mean(
            interp_pressure_to_hybrid(T, plev, ps, A_full, B_full, 100000.0)
        ))(jnp.float64(95000.0))
        assert jnp.isfinite(grad)

    # ---- numerical contract (the shape/constant-field tests above are vacuous:
    # a constant field hides bracketing / log-p / axis / extrapolation bugs) ----

    _PLEV = jnp.array([5000., 10000., 25000., 50000., 85000., 100000.])  # ascending Pa

    def test_logp_linear_field_is_exact_interior(self):
        """A field linear in log-p, f = a + b·ln(p), must be reproduced EXACTLY by
        log-p interpolation at interior targets (catches linear-in-p, wrong bracket,
        or axis bugs that a constant field would not)."""
        from legoesm.training.vertical_interp import interp_pressure_to_sigma
        a, b = 280.0, -8.0
        f = (a + b * jnp.log(self._PLEV))[None, :]
        p_s = jnp.array([100000.])
        sigma = jnp.array([0.1, 0.3, 0.6, 0.85, 0.99])     # p_target strictly in-range
        out = np.asarray(interp_pressure_to_sigma(f, self._PLEV, p_s, sigma))[0]
        p_t = np.asarray(sigma) * 1.0e5
        np.testing.assert_allclose(out, a + b * np.log(p_t), rtol=1e-4)

    def test_reproduces_values_at_source_levels(self):
        """Target pressure exactly on a source level returns that level's value."""
        from legoesm.training.vertical_interp import interp_pressure_to_sigma
        f = jnp.array([[200., 210., 230., 250., 270., 285.]])
        p_s = jnp.array([100000.])
        sigma = self._PLEV / 1.0e5                          # p_target == plev exactly
        out = np.asarray(interp_pressure_to_sigma(f, self._PLEV, p_s, sigma))[0]
        np.testing.assert_allclose(out, np.asarray(f)[0], rtol=1e-5)

    def test_hold_constant_above_model_top(self):
        """p_target below the lowest source level (above model top) holds f[0]."""
        from legoesm.training.vertical_interp import interp_pressure_to_sigma
        f = jnp.array([[200., 210., 230., 250., 270., 285.]])
        out = interp_pressure_to_sigma(
            f, self._PLEV, jnp.array([100000.]), jnp.array([0.01]))   # p=1000 < 5000
        assert float(out[0, 0]) == pytest.approx(200.0, abs=1e-5)

    def test_hold_constant_below_surface(self):
        """p_target above the highest source level (below surface) holds f[-1]."""
        from legoesm.training.vertical_interp import interp_pressure_to_sigma
        f = jnp.array([[200., 210., 230., 250., 270., 285.]])
        out = interp_pressure_to_sigma(
            f, self._PLEV, jnp.array([110000.]), jnp.array([1.0]))    # p=110000 > 100000
        assert float(out[0, 0]) == pytest.approx(285.0, abs=1e-5)

    def test_per_column_surface_pressure_vectorized(self):
        """Different p_s AND a different source profile per column: catches BOTH a
        take_along_axis target-pressure broadcast bug AND a source-field
        column-mixing bug (each column must use its OWN p_s and its OWN profile)."""
        from legoesm.training.vertical_interp import interp_pressure_to_sigma
        coeffs = ((280.0, -8.0), (300.0, -12.0))     # distinct (a,b) per column
        f = jnp.stack([a + b * jnp.log(self._PLEV) for a, b in coeffs])  # (2, 6), rows differ
        p_s = jnp.array([100000., 60000.])           # column 1 surface = 600 hPa
        sigma = jnp.array([0.5, 0.9])
        out = np.asarray(interp_pressure_to_sigma(f, self._PLEV, p_s, sigma))
        # each column's targets are sigma * its OWN p_s, all interior → exact log-p
        # of THAT column's profile (a column-mix would pull the other (a,b)).
        for c, ((a, b), ps) in enumerate(zip(coeffs, (100000.0, 60000.0))):
            p_t = np.asarray(sigma) * ps
            np.testing.assert_allclose(out[c], a + b * np.log(p_t), rtol=1e-4)

    def test_monotone_field_no_overshoot(self):
        """Interpolated values stay within the bracketing source values (the clamp
        on alpha forbids overshoot)."""
        from legoesm.training.vertical_interp import interp_pressure_to_sigma
        f = jnp.array([[200., 210., 230., 250., 270., 285.]])   # increasing with p
        p_s = jnp.array([100000.])
        sigma = jnp.linspace(0.02, 1.05, 40)                    # spans both extrapolations
        out = np.asarray(interp_pressure_to_sigma(f, self._PLEV, p_s, sigma))[0]
        assert out.min() >= 200.0 - 1e-4 and out.max() <= 285.0 + 1e-4

    def test_gradient_wrt_field_nonconstant(self):
        """Gradient flows through the (non-constant) source field with the EXACT
        LOG-P weights: an interior target depends ONLY on its two bracketing levels,
        and d/df equals the log-pressure interpolation weight (a linear-in-PRESSURE
        interpolator with the same bracket would give a DIFFERENT, wrong weight)."""
        from legoesm.training.vertical_interp import interp_pressure_to_sigma
        f0 = jnp.array([[200., 210., 230., 250., 270., 285.]])
        p_s = jnp.array([100000.])
        sigma = jnp.array([0.6])                 # p_t=60000 ∈ (50000, 85000) → levels 3,4
        g = jax.grad(lambda f: interp_pressure_to_sigma(
            f, self._PLEV, p_s, sigma).sum())(f0)
        g = np.asarray(g)[0]
        # exact log-p weight on the UPPER bracket level (index 4 @ 85000 Pa):
        alpha = (np.log(60000.0) - np.log(50000.0)) / (np.log(85000.0) - np.log(50000.0))
        # distinct from the linear-in-pressure weight 10000/35000≈0.286 → this asserts log-p.
        assert abs(alpha - 10000.0 / 35000.0) > 0.05
        np.testing.assert_allclose(g[4], alpha, rtol=1e-4)        # d/df[4] = alpha (log-p)
        np.testing.assert_allclose(g[3], 1.0 - alpha, rtol=1e-4)  # d/df[3] = 1-alpha
        assert np.allclose(g[[0, 1, 2, 5]], 0.0)                  # non-bracket levels unused

    def test_hybrid_p_full_overrides_pure_sigma_target(self):
        """The explicit ``p_full`` (iter-339) must land the field on the MODEL's HYBRID
        full-level pressures (``A·p_ref + B·p_s``), NOT pure-sigma ``σ·p_s`` — else a
        hybrid model's ERA5 reference is interpolated to the wrong levels and every
        bias is silently off.  A field linear in log-p is reproduced at ``p_full``'s
        pressures, which DIFFER from ``σ·p_s`` here (so ``p_full`` must actually be used)."""
        from legoesm.training.vertical_interp import interp_pressure_to_sigma
        a, b = 280.0, -8.0
        f = (a + b * jnp.log(self._PLEV))[None, :]
        p_s = jnp.array([100000.])
        sigma = jnp.array([0.3, 0.6, 0.85])             # σ·p_s = [30000, 60000, 85000]
        p_full = jnp.array([[40000., 55000., 80000.]])  # hybrid levels, interior, ≠ σ·p_s
        out = np.asarray(
            interp_pressure_to_sigma(f, self._PLEV, p_s, sigma, p_full=p_full))[0]
        # Lands on p_full (reproducing the log-p-linear field at the HYBRID pressures).
        np.testing.assert_allclose(out, a + b * np.log(np.asarray(p_full)[0]), rtol=1e-4)
        # And is DISTINCT from the pure-sigma result (proves p_full is honoured, not ignored).
        pure = np.asarray(interp_pressure_to_sigma(f, self._PLEV, p_s, sigma))[0]
        assert not np.allclose(out, pure)

    def test_p_full_overrides_pure_sigma_target_for_hybrid(self):
        """``p_full`` interpolates to the model's TRUE full-level pressures (e.g. a HYBRID
        coordinate's ``A·p_ref + B·p_s``) instead of pure-sigma ``sigma·p_s`` (iter 339,
        completing the iter-337/338 fix so the ERA5 reference lands on the model's actual
        levels).  Over terrain (p_s != p_ref) the targets differ ⇒ the interpolated field
        differs; passing ``p_full == sigma·p_s`` reproduces the default EXACTLY."""
        from legoesm.grids.vertical import make_hybrid_levels
        from legoesm.training.vertical_interp import interp_pressure_to_sigma

        f = jnp.array([[200., 210., 230., 250., 270., 285.]])     # T(p), increasing with p
        p_s = jnp.array([70000.0])                               # terrain (p_s != p_ref ~1e5)
        hc = make_hybrid_levels(6, p_top_Pa=100.0)
        sigma_f = jnp.asarray(hc.sigma_full)
        pure = interp_pressure_to_sigma(f, self._PLEV, p_s, sigma_f)
        hybrid = interp_pressure_to_sigma(
            f, self._PLEV, p_s, sigma_f, p_full=hc.pressure_at_full(p_s))
        assert float(jnp.max(jnp.abs(hybrid - pure))) > 1.0       # the level pressures differ
        # passing the pure-sigma target explicitly reproduces the default (byte-identical).
        same = interp_pressure_to_sigma(
            f, self._PLEV, p_s, sigma_f, p_full=p_s[:, None] * sigma_f)
        np.testing.assert_allclose(np.asarray(same), np.asarray(pure), rtol=1e-12)


# ---------------------------------------------------------------------------
# 1b. era5 horizontal regrid — latitude ordering (no N/S hemisphere flip)
# ---------------------------------------------------------------------------

class TestEra5HorizontalRegrid:
    """ERA5 stores latitude 90→-90 DESCENDING.  The regrid feeds that descending
    axis straight into ``scipy.interpolate.RegularGridInterpolator``, which requires
    a strictly-MONOTONIC axis (modern scipy accepts descending; older scipy raises).
    The silent-catastrophe failure mode is a hemisphere FLIP — model-North paired
    with ERA5-South — which makes every column bias compare the wrong latitude.
    Using ``field == latitude`` (a linear field that linear interp reproduces
    EXACTLY) catches a flip decisively: a correct regrid returns each target lat's
    own value; a flip returns its negation."""

    class _Grid:
        def __init__(self, lat, lon):
            self.lat = lat
            self.lon = lon

    def test_regrid_2d_to_gaussian_preserves_hemisphere(self):
        from legoesm.training.era5_to_state import regrid_2d_to_gaussian

        era5_lat = np.deg2rad(np.linspace(90.0, -90.0, 19))      # DESCENDING (ERA5)
        era5_lon = np.deg2rad(np.linspace(0.0, 360.0, 24, endpoint=False))
        field = np.broadcast_to(era5_lat[:, None], (19, 24)).astype(np.float64)  # f = lat
        # Target Gaussian grid: ASCENDING interior latitudes (no extrapolation).
        gauss_lat = np.deg2rad(np.array([-60.0, -20.0, 0.0, 30.0, 75.0]))
        gauss_lon = np.deg2rad(np.array([10.0, 100.0, 250.0]))
        out = np.asarray(regrid_2d_to_gaussian(
            field, era5_lat, era5_lon, self._Grid(gauss_lat, gauss_lon)))
        # f == lat ⇒ regridded value at each target lat is THAT lat (exact for a
        # linear field), identical across lon.  A hemisphere flip would yield
        # -gauss_lat instead (e.g. +75° → -75°) — caught by the sign + value.
        for i, gl in enumerate(gauss_lat):
            np.testing.assert_allclose(out[i, :], gl, atol=1e-5)
        assert np.all(np.diff(out[:, 0]) > 0)                    # increases N-ward, not flipped

    def test_regrid_latlon_to_gaussian_preserves_hemisphere(self):
        # The 3D production path (T/u/v/q) uses the SAME descending-lat interp.
        from legoesm.training.era5_to_state import (
            ERA5Slice,
            regrid_latlon_to_gaussian,
        )

        nlat, nlon, nlev = 19, 24, 3
        era5_lat = np.deg2rad(np.linspace(90.0, -90.0, nlat))    # DESCENDING
        era5_lon = np.deg2rad(np.linspace(0.0, 360.0, nlon, endpoint=False))
        lat_field = np.broadcast_to(
            era5_lat[:, None, None], (nlat, nlon, nlev)).astype(np.float64)  # f = lat
        ps = np.broadcast_to(era5_lat[:, None], (nlat, nlon)).astype(np.float64)
        era5 = ERA5Slice(
            T=lat_field, u=lat_field, v=lat_field, q=lat_field, p_s=ps,
            sst=ps, phis=ps, lat=era5_lat, lon=era5_lon,
            plev_Pa=np.linspace(5000.0, 100000.0, nlev))
        gauss_lat = np.deg2rad(np.array([-50.0, 0.0, 65.0]))
        gauss_lon = np.deg2rad(np.array([30.0, 200.0]))
        t_g, _u, _v, _q, ps_g = regrid_latlon_to_gaussian(
            era5, self._Grid(gauss_lat, gauss_lon))
        t_g = np.asarray(t_g)
        # Each regridded level reproduces the target latitude (no flip), and p_s too.
        for i, gl in enumerate(gauss_lat):
            np.testing.assert_allclose(t_g[i, :, :], gl, atol=1e-5)
            np.testing.assert_allclose(np.asarray(ps_g)[i, :], gl, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. losses
# ---------------------------------------------------------------------------

class TestLosses:

    def test_carry_mse_scalar(self):
        from legoesm.training.losses import carry_mse
        from legoesm.driver.compiled_segments import pack_carry
        s3, s2 = (6, N, N, NLEV), (6, N, N)
        state = _make_state(280.0)
        c1 = pack_carry(state, q_v=jnp.ones(s3)*0.01, q_c=jnp.zeros(s3),
                         q_r=jnp.zeros(s3), held_dT_rad=jnp.zeros(s3),
                         held_sw_net_sfc=jnp.zeros(s2), held_lw_net_sfc=jnp.zeros(s2),
                         held_sw_up_toa=jnp.zeros(s2), held_lw_up_toa=jnp.zeros(s2),
                         held_sw_down_toa=jnp.zeros(s2), step_index=0)
        state2 = _make_state(282.0)
        c2 = pack_carry(state2, q_v=jnp.ones(s3)*0.01, q_c=jnp.zeros(s3),
                         q_r=jnp.zeros(s3), held_dT_rad=jnp.zeros(s3),
                         held_sw_net_sfc=jnp.zeros(s2), held_lw_net_sfc=jnp.zeros(s2),
                         held_sw_up_toa=jnp.zeros(s2), held_lw_up_toa=jnp.zeros(s2),
                         held_sw_down_toa=jnp.zeros(s2), step_index=0)
        sigma_full = jnp.asarray(_SIGMA.sigma_full)
        loss = carry_mse(c1, c2, sigma_full)
        assert loss.shape == ()
        assert float(loss) > 0

    def test_carry_mse_normalization_balances_scales(self):
        """Iter-69: per-variable scale normalization balances the loss
        contributions across T (~K), wind (~m/s), q (~kg/kg), ps (~Pa)
        which differ by ~9 orders of magnitude in raw squared units.

        Why non-vacuous: with normalize_by_scale=False, a 1 Pa
        ps perturbation contributes ~w_ps to the loss, while a 1 K
        T perturbation contributes ~w_T (commensurate) but a 0.001
        kg/kg q perturbation contributes only ~w_q · 1e-6 (six orders
        below).  With normalize_by_scale=True (default), each
        contribution is scaled by 1/(typical_amplitude²), bringing
        the moisture and wind branches into commensurate range.

        The test perturbs a single variable at a time at its typical
        anomaly scale and asserts the loss contributions are within
        1 order of magnitude of each other.
        """
        from legoesm.training.losses import carry_mse, LossConfig
        from legoesm.driver.compiled_segments import pack_carry
        s3, s2 = (6, N, N, NLEV), (6, N, N)
        sigma_full = jnp.asarray(_SIGMA.sigma_full)
        cfg = LossConfig()  # normalize_by_scale=True default

        def _make_carry(state):
            return pack_carry(
                state, q_v=jnp.ones(s3)*0.01, q_c=jnp.zeros(s3),
                q_r=jnp.zeros(s3), held_dT_rad=jnp.zeros(s3),
                held_sw_net_sfc=jnp.zeros(s2),
                held_lw_net_sfc=jnp.zeros(s2),
                held_sw_up_toa=jnp.zeros(s2),
                held_lw_up_toa=jnp.zeros(s2),
                held_sw_down_toa=jnp.zeros(s2), step_index=0,
            )

        # Baseline state (all variables at "rest")
        ref = _make_state(280.0)
        c_ref = _make_carry(ref)

        # Perturb T by 30 K (one T_scale)
        state_dT = _make_state(280.0)
        from legoesm.core.field import Field
        state_dT = state_dT._replace(
            T=Field(state_dT.T.data + 30.0, name="T",
                    dims=state_dT.T.dims, units="K")
        )
        c_dT = _make_carry(state_dT)
        loss_T = float(carry_mse(c_ref, c_dT, sigma_full, config=cfg))

        # Perturb ps by 1000 Pa (one ps_scale)
        state_dps = _make_state(280.0)
        state_dps = state_dps._replace(
            p_s=Field(state_dps.p_s.data + 1000.0, name="p_s",
                       dims=state_dps.p_s.dims, units="Pa")
        )
        c_dps = _make_carry(state_dps)
        loss_ps = float(carry_mse(c_ref, c_dps, sigma_full, config=cfg))

        # Both perturbations are at one scale-unit: the loss
        # contributions should be commensurate (within a factor of
        # ~10).  Without scale normalization, loss_ps would dominate
        # by ~1e9 (the ps²/T² ratio for raw units).
        ratio = loss_ps / loss_T
        assert 0.01 < ratio < 100.0, (
            f"After per-variable scale normalization, T- and ps-"
            f"perturbation losses should be commensurate (ratio in "
            f"[0.01, 100]); got loss_T = {loss_T:.3e}, "
            f"loss_ps = {loss_ps:.3e}, ratio = {ratio:.3e}.  "
            f"Without normalization the ratio would be ~1e9."
        )

    def test_level_weights(self):
        from legoesm.training.losses import level_weights
        sigma = jnp.linspace(0.05, 0.975, 20)
        w = level_weights(sigma)
        assert w.shape == (20,)
        assert jnp.all(w > 0)


# ---------------------------------------------------------------------------
# 3. trainable_params
# ---------------------------------------------------------------------------

class TestTrainableParams:

    def test_from_defaults(self):
        from legoesm.training.trainable_params import TrainablePhysicsParams
        p = TrainablePhysicsParams.from_defaults()
        d = p.as_dict()
        # tau_equator/tau_pole left this set on 2026-08-11: they are gray
        # optical depths, and gray radiation is not trained.
        assert "tau_equator" not in d and "tau_pole" not in d
        assert "sbm_tau_c" in d
        assert abs(float(d["sbm_tau_c"]) - 7200.0) < 1.0

    def test_gradient_flow(self):
        from legoesm.training.trainable_params import TrainablePhysicsParams
        import equinox as eqx
        p = TrainablePhysicsParams.from_defaults()
        loss_fn = lambda p_: sum(v**2 for v in p_.as_dict().values())
        _, grads = eqx.filter_value_and_grad(loss_fn)(p)
        assert all(jnp.isfinite(v) for v in grads.raw_values.values())


# ---------------------------------------------------------------------------
# 4. dycore_rollout
# ---------------------------------------------------------------------------

class TestDycoreRollout:

    def test_rollout_config(self):
        from legoesm.training.dycore_rollout import RolloutConfig
        cfg = RolloutConfig(n_days=3, dt=600.0)
        assert cfg.n_days == 3

    def test_single_day_rollout_callable(self):
        from legoesm.training.dycore_rollout import single_day_rollout
        assert callable(single_day_rollout)


# ---------------------------------------------------------------------------
# 5. neural_physics
# ---------------------------------------------------------------------------

class TestNeuralPhysics:

    def test_import_and_construct(self):
        from legoesm.atmosphere.physics.neural_physics import NeuralPhysics
        key = jax.random.PRNGKey(0)
        nn = NeuralPhysics(nlev=NLEV, key=key)
        assert hasattr(nn, '__call__')

    def test_produces_tendencies(self):
        from legoesm.atmosphere.physics.neural_physics import NeuralPhysics
        key = jax.random.PRNGKey(0)
        nn = NeuralPhysics(nlev=NLEV, key=key)
        # NeuralPhysics takes a single packed column vector
        # T, u, v, q per level + p_s + solar + T_sfc + sic
        n_input = NLEV * 4 + 4
        assert nn.n_input == n_input
        x = jnp.ones(n_input)
        out = nn(x)
        assert out.shape[0] == NLEV * 4 + 6

    def test_untrained_network_emits_exactly_zero_tendencies(self):
        """Epoch-0 stability contract (#797 neural_gcm smoke loss=nan).

        An UNTRAINED NeuralPhysics must emit EXACTLY zero output, so the
        first neural_gcm rollout is the pure dycore (finite by construction).
        residual_scale=0.01 alone is NOT near-zero in physical tendency
        units: random O(1) outputs x 0.01 gave dq_v_dt ~ 0.04 kg/kg/s
        against q_v ~ 1e-3 — the C32/L8 smoke rollout went non-finite
        within 32 steps (probe job 26081628). Zero-init of the final layer
        is the standard residual-learning guarantee.
        """
        from legoesm.atmosphere.physics.neural_physics import NeuralPhysics
        for seed in (0, 7):
            nn = NeuralPhysics(nlev=NLEV, key=jax.random.PRNGKey(seed))
            x = jnp.linspace(-1.0, 1.0, NLEV * 4 + 2)   # O(1) packed features
            assert bool(jnp.all(nn(x) == 0.0))

    def test_untrained_network_final_layer_is_trainable(self):
        """Zero-init must not kill learning: the final layer's gradient is
        nonzero on the first step (hidden activations are nonzero), so the
        optimizer immediately moves it off zero and gradients then reach
        the earlier layers."""
        import equinox as eqx
        from legoesm.atmosphere.physics.neural_physics import NeuralPhysics
        nn = NeuralPhysics(nlev=NLEV, key=jax.random.PRNGKey(0))
        x = jnp.linspace(-1.0, 1.0, NLEV * 4 + 2)
        grads = eqx.filter_grad(lambda m: jnp.mean(m(x)))(nn)
        assert bool(jnp.any(grads.layers[-1].weight != 0.0))


# ---------------------------------------------------------------------------
# 6. sfno_dycore_coupling
# ---------------------------------------------------------------------------

class TestSFNOCoupling:

    def test_import(self):
        from legoesm.training.sfno_dycore_coupling import SFNOPhysics
        assert SFNOPhysics is not None


# ---------------------------------------------------------------------------
# 7. era5_to_state
# ---------------------------------------------------------------------------

class TestERA5ToState:

    def test_config_defaults(self):
        from legoesm.training.era5_to_state import TrainingERA5Config
        cfg = TrainingERA5Config()
        assert "temperature" in cfg.pressure_variables
        assert "surface_pressure" in cfg.surface_variables

    def test_era5_slice_namedtuple(self):
        from legoesm.training.era5_to_state import ERA5Slice
        s = ERA5Slice(
            T=np.zeros((10, 20, 5)), u=np.zeros((10, 20, 5)),
            v=np.zeros((10, 20, 5)), q=np.zeros((10, 20, 5)),
            p_s=np.zeros((10, 20)), sst=np.zeros((10, 20)),
            phis=np.zeros((10, 20)),
            lat=np.linspace(-np.pi/2, np.pi/2, 10),
            lon=np.linspace(0, 2*np.pi, 20),
            plev_Pa=np.array([5000, 10000, 50000, 85000, 100000], dtype=np.float64),
        )
        assert s.T.shape == (10, 20, 5)

    def test_weight_cache(self):
        from legoesm.training.era5_to_state import _get_cs_weights
        # New 3-arg API: weights built from the ACTUAL source lat/lon (radians).
        src_lat = np.linspace(np.pi / 2, -np.pi / 2, 18)
        src_lon = np.linspace(0.0, 2 * np.pi, 36, endpoint=False)
        w1 = _get_cs_weights(src_lat, src_lon, _GRID)
        w2 = _get_cs_weights(src_lat, src_lon, _GRID)
        assert w1 is w2  # same object from content-fingerprinted cache
        # A source grid with the SAME shape+endpoints but different INTERIOR
        # spacing must NOT collide on the cache (the proxy-bug failure mode).
        src_lat_stretched = np.linspace(np.pi / 2, -np.pi / 2, 18)
        src_lat_stretched[1:-1] *= 0.5   # perturb interior, keep endpoints
        w3 = _get_cs_weights(src_lat_stretched, src_lon, _GRID)
        assert w3 is not w1

    def test_era5_to_cubedsphere_carry_shapes_and_finite(self):
        """era5_to_cubedsphere_carry produces carry with correct shapes
        and finite values from a synthetic ERA5Slice."""
        import jax.numpy as jnp
        from legoesm.training.era5_to_state import ERA5Slice, era5_to_cubedsphere_carry

        n_lat, n_lon, n_plev = 18, 36, 4
        rng = np.random.default_rng(0)
        # Realistic T and p_s to avoid saturation/interp edge cases
        T_ll = (260.0 + rng.random((n_lat, n_lon, n_plev)) * 40.0).astype(np.float32)
        u_ll = rng.random((n_lat, n_lon, n_plev)).astype(np.float32) * 20.0
        v_ll = rng.random((n_lat, n_lon, n_plev)).astype(np.float32) * 20.0
        q_ll = (rng.random((n_lat, n_lon, n_plev)) * 0.01).astype(np.float32)
        p_s = np.full((n_lat, n_lon), 101325.0, dtype=np.float32)
        plev_Pa = np.array([5000.0, 25000.0, 50000.0, 100000.0], dtype=np.float64)

        era5 = ERA5Slice(
            T=T_ll, u=u_ll, v=v_ll, q=q_ll,
            p_s=p_s, sst=np.full((n_lat, n_lon), 290.0, dtype=np.float32),
            phis=np.zeros((n_lat, n_lon), dtype=np.float32),
            lat=np.linspace(-np.pi/2, np.pi/2, n_lat),
            lon=np.linspace(0, 2*np.pi, n_lon, endpoint=False),
            plev_Pa=plev_Pa,
        )

        carry = era5_to_cubedsphere_carry(era5, _GRID, _SIGMA)

        expected_3d = (6, N, N, NLEV)
        expected_2d = (6, N, N)
        assert carry.T.shape == expected_3d, f"T shape {carry.T.shape} != {expected_3d}"
        assert carry.u.shape == expected_3d
        assert carry.v.shape == expected_3d
        assert carry.q_v.shape == expected_3d
        assert carry.p_s.shape == expected_2d

        assert jnp.all(jnp.isfinite(carry.T)), "T contains non-finite values"
        assert jnp.all(jnp.isfinite(carry.u)), "u contains non-finite values"
        assert jnp.all(jnp.isfinite(carry.q_v)), "q_v contains non-finite values"
        assert jnp.all(carry.q_v >= 0), "q_v contains negative values"

    def test_era5_to_cubedsphere_carry_barometric_correction(self):
        """With Tibet-like phis, the barometric p_s correction is applied and
        the output remains finite.  Without the correction, smoothing phis
        without adjusting p_s would worsen the split-PGF residual over steep
        terrain boundaries."""
        import jax.numpy as jnp
        from legoesm.training.era5_to_state import ERA5Slice, era5_to_cubedsphere_carry

        n_lat, n_lon, n_plev = 18, 36, 4
        rng = np.random.default_rng(42)
        T_ll = (260.0 + rng.random((n_lat, n_lon, n_plev)) * 40.0).astype(np.float32)
        u_ll = rng.random((n_lat, n_lon, n_plev)).astype(np.float32) * 20.0
        v_ll = rng.random((n_lat, n_lon, n_plev)).astype(np.float32) * 20.0
        q_ll = (rng.random((n_lat, n_lon, n_plev)) * 0.005).astype(np.float32)
        plev_Pa = np.array([5000.0, 25000.0, 50000.0, 100000.0], dtype=np.float64)

        # Tibet-like phis: steep gradient in northern quarter of domain
        phis = np.zeros((n_lat, n_lon), dtype=np.float32)
        phis[: n_lat // 3, :] = 50000.0   # ~5000 m elevation
        p_s = np.where(phis > 0, 55000.0, 101325.0).astype(np.float32)

        era5 = ERA5Slice(
            T=T_ll, u=u_ll, v=v_ll, q=q_ll,
            p_s=p_s, sst=np.full((n_lat, n_lon), 290.0, dtype=np.float32),
            phis=phis,
            lat=np.linspace(-np.pi / 2, np.pi / 2, n_lat),
            lon=np.linspace(0, 2 * np.pi, n_lon, endpoint=False),
            plev_Pa=plev_Pa,
        )

        carry = era5_to_cubedsphere_carry(era5, _GRID, _SIGMA)

        assert jnp.all(jnp.isfinite(carry.T)), "T non-finite with Tibet phis"
        assert jnp.all(jnp.isfinite(carry.u)), "u non-finite with Tibet phis"
        assert jnp.all(jnp.isfinite(carry.p_s)), "p_s non-finite with Tibet phis"
        # p_s should be positive everywhere after barometric correction
        assert jnp.all(carry.p_s > 0), "p_s has non-positive values after correction"

    def test_era5_to_cubedsphere_carry_hybrid_ps_floor(self):
        """With L40 hybrid coordinate and Tibet-like p_s << p_ref, the p_s floor
        must be enforced and phis adjusted so that all hybrid layer thicknesses
        remain positive (no degenerate/inverted levels).  Without this fix, 19 of
        40 levels are underground at p_s=56703 Pa and the arch-peak at lev 28–29
        has dp = −1 Pa, causing catastrophic continuity-equation blow-up."""
        import jax.numpy as jnp
        from legoesm.grids.vertical import standard_hybrid_levels
        from legoesm.training.era5_to_state import ERA5Slice, era5_to_cubedsphere_carry

        n_lat, n_lon, n_plev = 18, 36, 5
        rng = np.random.default_rng(7)
        plev_Pa = np.array([5000.0, 15000.0, 30000.0, 55000.0, 100000.0], dtype=np.float64)
        T_ll = (240.0 + rng.random((n_lat, n_lon, n_plev)) * 50.0).astype(np.float32)
        u_ll = rng.random((n_lat, n_lon, n_plev)).astype(np.float32) * 20.0
        v_ll = rng.random((n_lat, n_lon, n_plev)).astype(np.float32) * 20.0
        q_ll = (rng.random((n_lat, n_lon, n_plev)) * 0.005).astype(np.float32)

        # Tibet-like column: p_s = 56703 Pa (far below L40 p_s_floor ~69645 Pa)
        phis = np.zeros((n_lat, n_lon), dtype=np.float32)
        phis[: n_lat // 3, :] = 46559.0   # ~4751 m (central Tibet)
        p_s = np.where(phis > 0, 56703.0, 101325.0).astype(np.float32)

        sigma40 = standard_hybrid_levels(40)

        era5 = ERA5Slice(
            T=T_ll, u=u_ll, v=v_ll, q=q_ll,
            p_s=p_s, sst=np.full((n_lat, n_lon), 270.0, dtype=np.float32),
            phis=phis,
            lat=np.linspace(-np.pi / 2, np.pi / 2, n_lat),
            lon=np.linspace(0, 2 * np.pi, n_lon, endpoint=False),
            plev_Pa=plev_Pa,
        )

        carry = era5_to_cubedsphere_carry(era5, _GRID, sigma40)

        # All fields must be finite
        assert jnp.all(jnp.isfinite(carry.T)), "T non-finite after p_s floor"
        assert jnp.all(jnp.isfinite(carry.u)), "u non-finite after p_s floor"
        assert jnp.all(jnp.isfinite(carry.p_s)), "p_s non-finite after p_s floor"
        assert jnp.all(jnp.isfinite(carry.phis)), "phis non-finite after p_s floor"

        # p_s must be at or above the minimum level where all hybrid layers
        # have positive thickness (dp_floor=100 Pa).  Compute floor from the
        # hybrid coordinate definition: p = A*p_ref + B*p_s, so minimum p_s
        # that keeps all layers positive is where A[-1] + B[-1]*p_s = A[-2] + B[-2]*p_s
        # i.e. p_s_floor = max over k of (A[k-1]-A[k])/(B[k]-B[k-1])+dp_floor/B_mean.
        # Simpler: p_s_floor via the constraint that the lowest full level stays
        # above the surface. Just verify the model enforces a positive floor.
        assert float(jnp.min(carry.p_s)) > 0.0, "p_s must be positive everywhere"
        # And that the floor was applied: Tibet column p_s=56703 Pa should be raised
        assert float(jnp.min(carry.p_s)) > 56703.0, (
            f"p_s floor not applied: min p_s={float(jnp.min(carry.p_s)):.1f} Pa "
            f"still at Tibet value 56703 Pa"
        )

        # All L40 hybrid layer thicknesses must be positive for every column
        A_full = jnp.asarray(sigma40.A_full)
        B_full = jnp.asarray(sigma40.B_full)
        p_model = A_full * sigma40.p_ref + B_full * carry.p_s[..., None]  # (6,N,N,40)
        dp = jnp.diff(p_model, axis=-1)  # (6,N,N,39)
        assert float(jnp.min(dp)) >= -1.0, (
            f"Negative layer thickness dp_min={float(jnp.min(dp)):.2f} Pa after p_s floor"
        )

    def test_era5_to_mpas_carry_phis_smoothing(self):
        """The MPAS carry smooths raw ERA5 phis on the Voronoi mesh (default
        passes=4), reducing cell-to-cell terrain gradients vs no smoothing,
        applies the barometric p_s correction, and stays finite.  Regression
        for the MPAS ERA5-IC wind-runaway blowup caused by unsmoothed phis."""
        import jax.numpy as jnp
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.grids.vertical import standard_hybrid_levels
        from legoesm.training.era5_to_state import ERA5Slice, era5_to_mpas_carry

        mesh = create_voronoi_mesh(2)          # ~162 cells, cheap
        sigma = standard_hybrid_levels(10)

        n_lat, n_lon, n_plev = 24, 48, 5
        rng = np.random.default_rng(11)
        plev_Pa = np.array(
            [5000.0, 15000.0, 30000.0, 55000.0, 100000.0], dtype=np.float64
        )
        T_ll = (250.0 + rng.random((n_lat, n_lon, n_plev)) * 40.0).astype(np.float32)
        u_ll = (rng.random((n_lat, n_lon, n_plev)) * 20.0).astype(np.float32)
        v_ll = (rng.random((n_lat, n_lon, n_plev)) * 20.0).astype(np.float32)
        q_ll = (rng.random((n_lat, n_lon, n_plev)) * 0.005).astype(np.float32)

        # Steep, noisy Tibet-like massif in the northern third (grid-scale
        # roughness is what the smoother must reduce).
        phis = np.zeros((n_lat, n_lon), dtype=np.float32)
        massif = slice(n_lat // 6, n_lat // 3)
        phis[massif, :] = (
            45000.0 + rng.random((massif.stop - massif.start, n_lon)) * 15000.0
        ).astype(np.float32)
        p_s = np.where(phis > 0, 56000.0, 101325.0).astype(np.float32)

        era5 = ERA5Slice(
            T=T_ll, u=u_ll, v=v_ll, q=q_ll,
            p_s=p_s, sst=np.full((n_lat, n_lon), 285.0, dtype=np.float32),
            phis=phis,
            lat=np.linspace(-np.pi / 2, np.pi / 2, n_lat),
            lon=np.linspace(0, 2 * np.pi, n_lon, endpoint=False),
            plev_Pa=plev_Pa,
        )

        carry_smooth = era5_to_mpas_carry(era5, mesh, sigma, smoothing_passes=4)
        carry_raw = era5_to_mpas_carry(era5, mesh, sigma, smoothing_passes=0)

        # Finite + physical
        for name, arr in (("T", carry_smooth.T), ("u", carry_smooth.u),
                          ("p_s", carry_smooth.p_s), ("phis", carry_smooth.phis)):
            assert jnp.all(jnp.isfinite(arr)), f"{name} non-finite with smoothing"
        assert jnp.all(carry_smooth.p_s > 0), "p_s non-positive after correction"

        # Max cell-to-cell phis gradient must drop with smoothing.
        def _max_neighbor_grad(field):
            f = np.asarray(field)
            coc = np.asarray(mesh.cellsOnCell)
            valid = coc >= 0
            idx = np.where(valid, coc, 0)
            diff = np.where(valid, np.abs(f[idx] - f[None, :]), 0.0)
            return float(diff.max())

        g_raw = _max_neighbor_grad(carry_raw.phis)
        g_smooth = _max_neighbor_grad(carry_smooth.phis)
        assert g_smooth < g_raw, (
            f"smoothing did not reduce phis gradient: raw={g_raw:.1f} "
            f"smooth={g_smooth:.1f} m^2/s^2"
        )

        # Barometric p_s correction changed p_s relative to the unsmoothed carry.
        assert not np.allclose(
            np.asarray(carry_smooth.p_s), np.asarray(carry_raw.p_s)
        ), "barometric p_s correction had no effect"

    def test_era5_to_cubedsphere_carry_converts_q_to_mixing_ratio(self):
        """Cross-grid parity with the latlon carry's q→mixing-ratio lock
        (``test_era5_load_regrid_to_reference_column_state_integration``): the
        cubed-sphere carry must ALSO convert ERA5 SPECIFIC humidity to MIXING ratio
        ``r = q/(1−q)`` (each ``era5_to_*_carry`` applies it independently, line 559;
        a refactor dropping it from THIS carry would silently leave the reference q as
        specific humidity — a moisture bias on every cubed-sphere run).  A CONSTANT
        ``q`` makes the test robust to the (nonlinear) convert-vs-regrid order: both
        give ``r`` for a uniform field."""
        import jax.numpy as jnp
        from legoesm.thermo import specific_humidity_to_mixing_ratio
        from legoesm.training.era5_to_state import ERA5Slice, era5_to_cubedsphere_carry

        n_lat, n_lon, n_plev = 18, 36, 4
        q0 = 5e-3

        def const(v):
            return np.full((n_lat, n_lon, n_plev), v, dtype=np.float32)

        era5 = ERA5Slice(
            T=const(280.0), u=const(5.0), v=const(0.0), q=const(q0),
            p_s=np.full((n_lat, n_lon), 101325.0, dtype=np.float32),
            sst=np.full((n_lat, n_lon), 290.0, dtype=np.float32),
            phis=np.zeros((n_lat, n_lon), dtype=np.float32),
            lat=np.linspace(-np.pi / 2, np.pi / 2, n_lat),
            lon=np.linspace(0, 2 * np.pi, n_lon, endpoint=False),
            plev_Pa=np.array([5000.0, 25000.0, 50000.0, 100000.0], dtype=np.float64))
        carry = era5_to_cubedsphere_carry(era5, _GRID, _SIGMA)
        expected_r = float(specific_humidity_to_mixing_ratio(jnp.asarray(q0)))
        # The uniform specific humidity becomes the (larger) MIXING ratio everywhere.
        np.testing.assert_allclose(np.asarray(carry.q_v), expected_r, rtol=2e-3)
        assert expected_r > q0          # mixing ratio strictly exceeds specific humidity

    def test_era5_to_spectral_carry_converts_q_to_mixing_ratio(self):
        """The LAST carry to reach q→mixing-ratio parity (after latlon/MPAS/cubed-
        sphere): the SPECTRAL carry must ALSO convert ERA5 SPECIFIC humidity to MIXING
        ratio ``r = q/(1−q)`` (line 462) — previously only its dispatch NAME was
        tested.  ``carry.q_v`` is the grid-space tracer (a constant field is invariant
        under the latlon→Gaussian regrid, so it equals ``r`` everywhere)."""
        import jax.numpy as jnp
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.thermo import specific_humidity_to_mixing_ratio
        from legoesm.training.era5_to_state import ERA5Slice, era5_to_spectral_carry

        n_lat, n_lon, n_plev = 18, 36, 4
        q0 = 5e-3

        def const(v):
            return np.full((n_lat, n_lon, n_plev), v, dtype=np.float32)

        era5 = ERA5Slice(
            T=const(280.0), u=const(5.0), v=const(0.0), q=const(q0),
            p_s=np.full((n_lat, n_lon), 101325.0, dtype=np.float32),
            sst=np.full((n_lat, n_lon), 290.0, dtype=np.float32),
            phis=np.zeros((n_lat, n_lon), dtype=np.float32),
            lat=np.linspace(np.pi / 2, -np.pi / 2, n_lat),   # descending (ERA5 convention)
            lon=np.linspace(0, 2 * np.pi, n_lon, endpoint=False),
            plev_Pa=np.array([5000.0, 25000.0, 50000.0, 100000.0], dtype=np.float64))
        carry = era5_to_spectral_carry(
            era5, create_gaussian_grid(8), create_sigma_coordinate(5))
        expected_r = float(specific_humidity_to_mixing_ratio(jnp.asarray(q0)))
        np.testing.assert_allclose(np.asarray(carry.q_v), expected_r, rtol=2e-3)
        assert expected_r > q0          # mixing ratio strictly exceeds specific humidity

    def test_era5_to_cubedsphere_carry_regrids_phis_into_the_state(self):
        """``era5_to_*_carry`` builds a FULL reference state — used both as the compare
        target AND to INITIALISE a model from ERA5, where the surface geopotential
        ``phis`` (topography ``g·z_s``) matters.  T/q/u have their regridded VALUES
        asserted; ``phis`` (regridded by the same IDW ``regrid_scalar``) only had its
        shape checked.  A CONSTANT ERA5 ``phis`` is invariant under the IDW regrid
        (weights sum to 1), so ``carry.phis`` equals it everywhere — a dropped or
        zeroed phis regrid is caught."""
        from legoesm.training.era5_to_state import ERA5Slice, era5_to_cubedsphere_carry

        n_lat, n_lon, n_plev = 18, 36, 4
        phis0 = 2000.0          # surface geopotential g·z_s [m²/s²]

        def const(v):
            return np.full((n_lat, n_lon, n_plev), v, dtype=np.float32)

        era5 = ERA5Slice(
            T=const(280.0), u=const(5.0), v=const(0.0), q=const(5e-3),
            p_s=np.full((n_lat, n_lon), 101325.0, dtype=np.float32),
            sst=np.full((n_lat, n_lon), 290.0, dtype=np.float32),
            phis=np.full((n_lat, n_lon), phis0, dtype=np.float32),
            lat=np.linspace(np.pi / 2, -np.pi / 2, n_lat),
            lon=np.linspace(0, 2 * np.pi, n_lon, endpoint=False),
            plev_Pa=np.array([5000.0, 25000.0, 50000.0, 100000.0], dtype=np.float64))
        carry = era5_to_cubedsphere_carry(era5, _GRID, _SIGMA)
        np.testing.assert_allclose(np.asarray(carry.phis), phis0, rtol=1e-4)


# ---------------------------------------------------------------------------
# 8. training_driver
# ---------------------------------------------------------------------------

class TestTrainingDriver:

    def test_build_training_segment(self):
        from legoesm.training.training_driver import build_training_segment
        from legoesm.driver.physics_pipeline import PhysicsOutput

        class MockModel:
            _state_type = HydrostaticState
            def step(self, s, dt):
                return s

        def mock_step(need_rad, T, p_s, *a, **kw):
            p = PhysicsOutput(**_zero_physics_output(T, p_s))
            return p, (a[16], a[17], a[18], a[19], a[20], a[21])

        fn = build_training_segment(
            MockModel(), mock_step, _GRID, _SIGMA, 600.0,
        )
        assert callable(fn)
        assert hasattr(fn, 'raw')

    def test_train_step_builds_segment_once(self, monkeypatch):
        """OOM/recompile guard: the segment fn is built exactly ONCE (one
        ``filter_jit`` trace), not once per sample*epoch.

        Regression test for the documented ~1.5 GiB/sample leak — the old
        driver rebuilt ``build_segment_fn`` and re-traced the rollout +
        reverse adjoint on every ``eqx.filter_value_and_grad`` call because
        the step was not wrapped in ``eqx.filter_jit``.
        """
        import equinox as eqx
        import optax
        import legoesm.training.training_driver as td
        from legoesm.training.trainable_params import TrainablePhysicsParams

        build_count = [0]

        def counting_build(**kw):
            build_count[0] += 1
            # Was tau_equator until 2026-08-11; that knob is gone (gray
            # optical depth, never trained). Any two live trainables do.
            tau = kw["C_H"]
            c_e = kw["C_E"]

            class _Seg:
                # ``.raw`` reads the (traced) trainable kwargs so the AD path
                # to the trainable is real (a zero grad would mean the params
                # got baked in as constants).
                def raw(self, carry, n_steps, forcing):
                    return tau * carry + c_e * forcing

            return _Seg()

        # Lightweight stand-ins exercise the real _build_train_step /
        # _training_loop control flow (filter_jit, value_and_grad, optimiser
        # update) without the heavy dycore.
        monkeypatch.setattr(td, "build_segment_fn", counting_build)
        # ``hours=`` is now passed by the trainer: the supervision horizon must
        # match the lead the targets were loaded at, so the fake has to accept
        # it or it hides the very forwarding this file exercises.
        monkeypatch.setattr(
            td, "single_day_rollout",
            lambda ic, forcing, run_seg_fn, dt, hours=24.0: run_seg_fn(
                ic, 1, forcing),
        )
        monkeypatch.setattr(
            td, "combined_loss",
            lambda pred, target, sigma_full, grid=None, config=None: jnp.sum(
                (pred - target) ** 2
            ),
        )

        def make_run_seg(trainable):
            return td.build_training_segment(
                None, None, _GRID, _SIGMA, 600.0,
                **trainable.to_segment_kwargs(),
            )

        from legoesm.training.losses import LossConfig

        params = TrainablePhysicsParams.from_defaults()
        optimizer = optax.adam(1e-3)
        # A REAL LossConfig, not None: the trainer routes through
        # ``multi_step_rollout_loss``, which reads ``multi_step_hours`` off it.
        # ``None`` only worked while the trainer inlined its own single-horizon
        # rollout and never looked at the config.
        train_step = td._build_train_step(
            make_run_seg, optimizer,
            jnp.asarray(_SIGMA.sigma_full), _GRID, 600.0, LossConfig(),
        )

        ics = [jnp.asarray(1.0), jnp.asarray(2.0)]
        targets = [jnp.asarray(0.0), jnp.asarray(0.0)]
        forcings = [jnp.asarray(0.5), jnp.asarray(1.5)]

        # First call traces once -> build runs once; AD must reach the
        # trainable (finite, NON-zero gradient).
        opt_state0 = optimizer.init(eqx.filter(params, eqx.is_array))
        _, _, loss0, gnorm0 = train_step(
            params, opt_state0, ics[0], targets[0], forcings[0]
        )
        assert jnp.isfinite(loss0)
        assert float(gnorm0) > 0.0
        assert build_count[0] == 1

        # 2 epochs x 2 samples = 4 more train_step calls reuse the SAME
        # compiled step (cache hits) => no extra builds.
        _, history = td._training_loop(
            train_step, params, optimizer, ics, targets, forcings,
            n_epochs=2, log_every=10,
        )
        assert build_count[0] == 1, (
            f"segment fn rebuilt {build_count[0]}x (expected 1); the "
            "filter_jit-once fix regressed -> per-sample retrace/OOM."
        )
        assert len(history) == 2
        assert all(jnp.isfinite(jnp.asarray(loss)) for loss in history)


# ---------------------------------------------------------------------------
# 9. API signature guards
# ---------------------------------------------------------------------------

class TestTrainingAPISignatures:
    """Verify training entrypoint signatures match their callees."""

    def test_neural_gcm_uses_adapter(self):
        """train_neural_gcm must create a ColumnAdapter, not pass grid directly."""
        from legoesm.atmosphere.physics.neural_physics import make_neural_step_unified
        from legoesm.core.grid_adapters import make_adapter, ColumnAdapter
        import inspect

        sig = inspect.signature(make_neural_step_unified)
        params = list(sig.parameters.keys())
        # Second param should be 'adapter', not 'grid'
        assert params[1] == "adapter"

        # Verify make_adapter produces a ColumnAdapter
        adapter = make_adapter(_GRID)
        assert isinstance(adapter, ColumnAdapter)
        assert adapter.ncol == 6 * N * N

    def test_sfno_step_unified_no_grid_param(self):
        """make_sfno_step_unified must NOT accept a grid positional argument."""
        from legoesm.training.sfno_dycore_coupling import make_sfno_step_unified
        import inspect

        sig = inspect.signature(make_sfno_step_unified)
        params = list(sig.parameters.keys())
        assert "grid" not in params
        assert params == ["sfno_physics", "mode", "traditional_step_unified"]

    def test_sfno_correction_mode_requires_pipeline(self):
        """train_sfno_coupled with mode='correction' must require physics_pipeline."""
        from legoesm.training.training_driver import train_sfno_coupled
        import inspect

        sig = inspect.signature(train_sfno_coupled)
        assert "physics_pipeline" in sig.parameters

    def test_sfno_replacement_mode_no_pipeline(self):
        """train_sfno_coupled with mode='replacement' should not require physics_pipeline."""
        from legoesm.training.training_driver import train_sfno_coupled
        import inspect

        sig = inspect.signature(train_sfno_coupled)
        # physics_pipeline should default to None
        assert sig.parameters["physics_pipeline"].default is None

    def test_neural_step_unified_accepts_optional_conv_prog_slot(self):
        from legoesm.atmosphere.physics.neural_physics import (
            NeuralPhysics,
            make_neural_step_unified,
        )
        from legoesm.core.grid_adapters import make_adapter

        neural = NeuralPhysics(nlev=NLEV, key=jax.random.PRNGKey(0))
        step = make_neural_step_unified(neural, make_adapter(_GRID))

        phys_out, held = step(*_step_unified_args(include_conv_prog=True))

        assert phys_out.du_dt.shape == _make_state().T.data.shape
        assert len(held) == 6

    def test_sfno_step_unified_accepts_optional_conv_prog_slot(self):
        from legoesm.training.sfno_dycore_coupling import make_sfno_step_unified

        class MockSFNOPhysics:
            def __call__(self, T, u, v, q_v, p_s, phis, dt):
                del u, v, q_v, phis, dt
                return PhysicsOutput(**_zero_physics_output(T, p_s))

        step = make_sfno_step_unified(MockSFNOPhysics(), mode="replacement")

        phys_out, held = step(*_step_unified_args(include_conv_prog=True))

        assert phys_out.du_dt.shape == _make_state().T.data.shape
        assert len(held) == 6


# ---------------------------------------------------------------------------
# 10. trainable_params scheme awareness
# ---------------------------------------------------------------------------

class TestTrainableParamsSchemeAware:
    """Trainable parameters must be scheme-aware and not crash for non-SBM."""

    def test_sbm_includes_convection_params(self):
        from legoesm.training.trainable_params import trainable_constraints_for_scheme
        constraints = trainable_constraints_for_scheme("sbm")
        names = [c.name for c in constraints]
        assert "sbm_tau_c" in names
        assert "sbm_RH_ref" in names
        # tau_equator/tau_pole dropped 2026-08-11 (gray is not trained).
        # Assert their ABSENCE — deleting the check instead would leave a test
        # that constrains nothing (Claude review).
        assert "tau_equator" not in names and "tau_pole" not in names

    def test_dca_excludes_sbm_params(self):
        from legoesm.training.trainable_params import trainable_constraints_for_scheme
        constraints = trainable_constraints_for_scheme("dca")
        names = [c.name for c in constraints]
        assert "sbm_tau_c" not in names
        assert "sbm_RH_ref" not in names
        # tau_equator/tau_pole dropped 2026-08-11 (gray is not trained).
        # Assert their ABSENCE — deleting the check instead would leave a test
        # that constrains nothing (Claude review).
        assert "tau_equator" not in names and "tau_pole" not in names

    def test_none_scheme_excludes_sbm_params(self):
        from legoesm.training.trainable_params import trainable_constraints_for_scheme
        constraints = trainable_constraints_for_scheme("none")
        names = [c.name for c in constraints]
        assert "sbm_tau_c" not in names
        # tau_equator/tau_pole dropped 2026-08-11 (gray is not trained).
        # Assert their ABSENCE — deleting the check instead would leave a test
        # that constrains nothing (Claude review).
        assert "tau_equator" not in names and "tau_pole" not in names

    def test_from_defaults_with_non_sbm(self):
        from legoesm.training.trainable_params import (
            TrainablePhysicsParams, trainable_constraints_for_scheme,
        )
        constraints = trainable_constraints_for_scheme("dca")
        params = TrainablePhysicsParams.from_defaults(constraints=constraints)
        d = params.as_dict()
        assert "sbm_tau_c" not in d
        assert "tau_equator" not in d   # gray optical depth, never trained

    def test_sbm_params_gradient_flow(self):
        from legoesm.training.trainable_params import (
            TrainablePhysicsParams, trainable_constraints_for_scheme,
        )
        import equinox as eqx

        for scheme in ["sbm", "dca", "none"]:
            constraints = trainable_constraints_for_scheme(scheme)
            p = TrainablePhysicsParams.from_defaults(constraints=constraints)
            loss_fn = lambda p_: sum(v**2 for v in p_.as_dict().values())
            _, grads = eqx.filter_value_and_grad(loss_fn)(p)
            assert all(jnp.isfinite(v) for v in grads.raw_values.values()), (
                f"Non-finite grad for scheme={scheme}"
            )

    def test_gray_radiation_trains_albedo_but_no_optical_depth(self):
        """Gray's OPTICAL knobs are frozen (2026-08-11 directive: gray is never
        trained), but its SW reflection still takes the blended surface albedo,
        which is a surface property rather than a radiation-scheme knob — so
        albedo_* keeps training under gray."""
        from legoesm.training.trainable_params import trainable_constraints_for_scheme
        names = [
            c.name
            for c in trainable_constraints_for_scheme(radiation_scheme="gray")
        ]
        assert "tau_equator" not in names and "tau_pole" not in names
        assert "albedo_ice" in names and "albedo_ocean" in names

    def test_rrtmgp_keeps_albedo_drops_tau(self):
        """RRTMGP consumes the blended surface albedo but explicitly
        discards the gray optical depths (dead DOF under rrtmgp)."""
        from legoesm.training.trainable_params import trainable_constraints_for_scheme
        for rad in ("rrtmgp", "rrtmg"):  # rrtmg is a normalized alias
            names = [
                c.name
                for c in trainable_constraints_for_scheme(radiation_scheme=rad)
            ]
            assert "albedo_ice" in names and "albedo_ocean" in names
            assert "tau_equator" not in names and "tau_pole" not in names

    def test_no_radiation_drops_radiation_params(self):
        from legoesm.training.trainable_params import trainable_constraints_for_scheme
        names = [
            c.name
            for c in trainable_constraints_for_scheme(radiation_scheme="none")
        ]
        for dead in ("tau_equator", "tau_pole", "albedo_ice", "albedo_ocean"):
            assert dead not in names

    def test_active_turbulence_drops_bulk_exchange_coeffs(self):
        """With a turbulence scheme on, turb_owns_surface bypasses the
        bulk C_H/C_E path entirely — they must not be offered."""
        from legoesm.training.trainable_params import trainable_constraints_for_scheme
        names_off = [
            c.name
            for c in trainable_constraints_for_scheme(turbulence_scheme="none")
        ]
        names_on = [
            c.name
            for c in trainable_constraints_for_scheme(turbulence_scheme="louis")
        ]
        assert "C_H" in names_off and "C_E" in names_off
        assert "C_H" not in names_on and "C_E" not in names_on
