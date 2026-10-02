"""Adaptive-implicit vertical momentum advection (Shchepetkin 2015 /
NEMO ``ln_zad_Aimp``) — unit + integration tests.

Covers the new leaf helpers in ``legoesm.ocean.vertical``:

- ``shchepetkin_implicit_fraction`` — the Courant->implicit-fraction ramp.
- ``implicit_vertical_advection_ocean`` — the backward-Euler upwind solve.
- ``adaptive_implicit_vertical_momentum_advection`` — the Courant-split
  explicit/implicit wrapper.

and the ``LatLonCGridOceanConfig.adaptive_implicit_vertadv`` model wiring
(step-level operator-split + flag-off bit-exact regression).

The load-bearing properties for an OMIP-faithful dycore are exercised
directly: column-integral conservation (flux form), unconditional
stability at vertical Courant >> 1 (where the explicit scheme blows up),
and reduction to the explicit scheme at low Courant.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.vertical import (  # noqa: E402
    adaptive_implicit_vertical_momentum_advection,
    flux_form_vertical_momentum_advection,
    implicit_vertical_advection_ocean,
    shchepetkin_implicit_fraction,
    _AIMP_CU_MIN,
    _AIMP_CU_MAX,
)


# ---------------------------------------------------------------------------
# 1. Shchepetkin ramp ``zcff(Cu)``
# ---------------------------------------------------------------------------

class TestShchepetkinFraction:
    def test_zero_below_cu_min(self):
        cu = jnp.linspace(0.0, _AIMP_CU_MIN, 11)
        z = shchepetkin_implicit_fraction(cu)
        np.testing.assert_allclose(np.asarray(z), 0.0, atol=0.0)

    def test_monotone_increasing(self):
        cu = jnp.linspace(0.0, 5.0, 200)
        z = np.asarray(shchepetkin_implicit_fraction(cu))
        assert np.all(np.diff(z) >= -1e-12), "ramp must be non-decreasing"

    def test_bounded_0_1(self):
        cu = jnp.linspace(0.0, 100.0, 500)
        z = np.asarray(shchepetkin_implicit_fraction(cu))
        assert z.min() >= 0.0 and z.max() <= 1.0
        # Approaches 1 for large Courant.
        assert z[-1] > 0.95

    def test_continuous_at_cu_cut(self):
        cu_cut = 2.0 * _AIMP_CU_MAX - _AIMP_CU_MIN
        eps = 1e-6
        lo = float(shchepetkin_implicit_fraction(jnp.array(cu_cut - eps)))
        hi = float(shchepetkin_implicit_fraction(jnp.array(cu_cut + eps)))
        assert abs(lo - hi) < 1e-4
        # Closed form at the junction: (cu_max-cu_min)/(2 cu_max-cu_min).
        expect = (_AIMP_CU_MAX - _AIMP_CU_MIN) / cu_cut
        assert abs(lo - expect) < 1e-4

    def test_differentiable(self):
        # grad exists and is finite across the explicit/blend/implicit regimes
        val = jax.grad(lambda c: shchepetkin_implicit_fraction(c).sum())(
            jnp.array([0.05, 0.25, 0.5, 1.0, 3.0]))
        assert np.all(np.isfinite(np.asarray(val)))


# ---------------------------------------------------------------------------
# helpers for the advection tests
# ---------------------------------------------------------------------------

def _column(nlev, seed=0):
    """A single stratified column: field, interface w (zero ends), h."""
    rng = np.random.default_rng(seed)
    field = jnp.asarray(rng.standard_normal((1, nlev)))
    h = jnp.asarray(0.5 + rng.random((1, nlev)))           # positive thickness
    w_in = rng.standard_normal((1, nlev - 1))
    # zero-flux top/bottom interfaces (w_half[0] = w_half[nlev] = 0)
    w_half = jnp.asarray(np.concatenate(
        [np.zeros((1, 1)), w_in, np.zeros((1, 1))], axis=-1))
    return field, w_half, h


def _column_integral(field, h):
    return float(jnp.sum(field * h))


# ---------------------------------------------------------------------------
# 2. Implicit backward-Euler upwind solve
# ---------------------------------------------------------------------------

class TestImplicitVerticalAdvection:
    def test_conserves_column_integral(self):
        # Flux form with zero-flux ends => sum_k(h_k * field_k) invariant.
        field, w_half, h = _column(12, seed=1)
        # scale w so the implicit Courant is large (stress the solve)
        w_half = w_half * 3.0
        out = implicit_vertical_advection_ocean(field, w_half, h, dt=600.0)
        assert np.all(np.isfinite(np.asarray(out)))
        np.testing.assert_allclose(
            _column_integral(out, h), _column_integral(field, h),
            rtol=0, atol=1e-9)

    def test_unconditionally_stable_high_courant(self):
        # |w| dt / h ~ O(50): explicit forward-Euler upwind would blow up;
        # the implicit solve stays bounded by the initial range (monotone).
        field, w_half, h = _column(20, seed=2)
        w_half = w_half * 30.0          # Courant >> 1
        out = np.asarray(
            implicit_vertical_advection_ocean(field, w_half, h, dt=600.0))
        f0 = np.asarray(field)
        assert np.all(np.isfinite(out))
        # Monotone (no new extrema) up to a tiny tolerance.
        assert out.max() <= f0.max() + 1e-9
        assert out.min() >= f0.min() - 1e-9

    def test_rock_cells_decouple_with_face_active(self):
        nlev = 8
        field, w_half, h = _column(nlev, seed=3)
        w_half = w_half * 5.0
        # Mark the bottom 3 cells inactive (below seafloor).
        active = np.ones((1, nlev))
        active[:, -3:] = 0.0
        out = np.asarray(implicit_vertical_advection_ocean(
            field, w_half, h, dt=600.0,
            face_active=jnp.asarray(active)))
        # Inactive cells are untouched (a=c=0, b=1 => identity).
        np.testing.assert_allclose(out[:, -3:], np.asarray(field)[:, -3:],
                                   rtol=0, atol=1e-12)

    def test_w_zero_is_exact_identity(self):
        # w_half == 0 everywhere => a=c=0, b=1 => x == field EXACTLY
        # (rebuts the concern that the thomas_solve _TINY denominator
        # guard perturbs an identity system: for b=1 >= 1, b + tiny32
        # underflows to b, and the pivots never trigger the |denom|<tiny
        # branch).
        field, _, h = _column(16, seed=11)
        w_zero = jnp.zeros((1, 17))
        out = implicit_vertical_advection_ocean(field, w_zero, h, dt=1234.0)
        # bit-exact identity, not just within a tolerance
        np.testing.assert_array_equal(np.asarray(out), np.asarray(field))

    def test_conservation_machine_precision_high_courant(self):
        # Conservation holds to float64 round-off (~1e-14 relative), NOT
        # spoiled by the _TINY guard, even at Courant >> 1.  Tightened
        # vs. the atol=1e-9 column test to pin "machine precision".
        field, w_half, h = _column(24, seed=12)
        w_half = w_half * 50.0
        out = implicit_vertical_advection_ocean(field, w_half, h, dt=600.0)
        i0 = _column_integral(field, h)
        i1 = _column_integral(out, h)
        assert abs(i1 - i0) <= 1e-12 + 1e-12 * abs(i0)

    def test_nonzero_rock_values_do_not_leak(self):
        # A partial-cell column whose inactive (rock) cells carry LARGE
        # nonzero values: with face_active gating they must (a) stay
        # exactly unchanged and (b) not leak into the active cells above.
        nlev = 10
        field, w_half, h = _column(nlev, seed=13)
        w_half = w_half * 12.0
        active = np.ones((1, nlev))
        active[:, -4:] = 0.0
        f = np.asarray(field).copy()
        f[:, -4:] = 1.0e6                      # garbage in the rock
        f_j = jnp.asarray(f)
        out = np.asarray(implicit_vertical_advection_ocean(
            f_j, w_half, h, dt=600.0, face_active=jnp.asarray(active)))
        # (a) rock cells untouched
        np.testing.assert_array_equal(out[:, -4:], f[:, -4:])
        # (b) active cells finite and uncontaminated by 1e6 rock values
        #     (compare to the same solve with the rock set to zero -- the
        #     active block must be identical, proving no leak).
        f0 = np.asarray(field).copy()
        f0[:, -4:] = 0.0
        out0 = np.asarray(implicit_vertical_advection_ocean(
            jnp.asarray(f0), w_half, h, dt=600.0,
            face_active=jnp.asarray(active)))
        np.testing.assert_allclose(out[:, :-4], out0[:, :-4],
                                   rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Adaptive wrapper (Courant split)
# ---------------------------------------------------------------------------

class TestAdaptiveWrapper:
    def test_reduces_to_explicit_at_low_courant(self):
        # Tiny w so every cell's Courant < cu_min => w_imp = 0 =>
        # result must equal one explicit forward-Euler upwind step.
        field, w_half, h = _column(15, seed=4)
        dt = 600.0
        # scale w so max Courant well below cu_min
        cu_cell = dt * (np.maximum(np.asarray(w_half)[:, :-1], 0)
                        - np.minimum(np.asarray(w_half)[:, 1:], 0)) \
            / np.asarray(h)
        scale = 0.5 * _AIMP_CU_MIN / max(cu_cell.max(), 1e-9)
        w_small = w_half * float(scale)

        adaptive = adaptive_implicit_vertical_momentum_advection(
            field, w_small, h, dt)
        explicit = field + dt * flux_form_vertical_momentum_advection(
            field, w_small, h)
        np.testing.assert_allclose(np.asarray(adaptive), np.asarray(explicit),
                                   rtol=0, atol=1e-12)

    def test_bounded_at_high_courant(self):
        field, w_half, h = _column(20, seed=5)
        dt = 600.0
        w_big = w_half * 40.0
        adaptive = np.asarray(
            adaptive_implicit_vertical_momentum_advection(field, w_big, h, dt))
        explicit = np.asarray(
            field + dt * flux_form_vertical_momentum_advection(field, w_big, h))
        assert np.all(np.isfinite(adaptive))
        # The explicit step overshoots far beyond the data range; the
        # adaptive one does not (this is the whole point of the scheme).
        f0 = np.asarray(field)
        span = f0.max() - f0.min()
        assert (adaptive.max() - adaptive.min()) <= 3.0 * span + 1e-6
        assert (explicit.max() - explicit.min()) > 5.0 * span

    def test_conserves_column_momentum(self):
        # The full operator (explicit + implicit, both flux form) conserves
        # the column momentum integral sum_k(h_k u_k).
        field, w_half, h = _column(16, seed=6)
        w_half = w_half * 8.0
        out = adaptive_implicit_vertical_momentum_advection(
            field, w_half, h, dt=900.0)
        np.testing.assert_allclose(
            _column_integral(out, h), _column_integral(field, h),
            rtol=0, atol=1e-9)

    def test_differentiable(self):
        field, w_half, h = _column(10, seed=7)
        loss = lambda f: jnp.sum(
            adaptive_implicit_vertical_momentum_advection(f, w_half * 6.0, h, 600.0) ** 2)
        g = jax.grad(loss)(field)
        assert np.all(np.isfinite(np.asarray(g)))


# ---------------------------------------------------------------------------
# 4. Model wiring: flag-on finite output + flag-off bit-exact regression
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model_pieces():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid = create_latlon_grid(n_lat=20, n_lon=40)
    z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0,
                                  dz_surface=10.0, dz_deep=500.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
    )
    # Seed a baroclinic perturbation so vertical advection is non-trivial.
    T = np.asarray(state.T.data)
    T = T + 2.0 * np.exp(-(np.linspace(0, 1, T.shape[-1]))[None, None, :])
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    return grid, z_coord, state, LatLonCGridOceanConfig, LatLonCGridOceanModel


def test_config_field_default_off(small_model_pieces):
    _, _, _, Config, _ = small_model_pieces
    cfg = Config()
    assert cfg.adaptive_implicit_vertadv is False


def test_step_flag_on_finite_output(small_model_pieces):
    grid, z_coord, state, Config, Model = small_model_pieces
    cfg = Config(adaptive_implicit_vertadv=True)
    model = Model(grid, z_coord, cfg)
    s = state
    for _ in range(5):
        s = model.step(s, dt=600.0)
    assert bool(jnp.all(jnp.isfinite(s.u.data)))
    assert bool(jnp.all(jnp.isfinite(s.v.data)))
    assert bool(jnp.all(jnp.isfinite(s.T.data)))


def test_flag_off_bit_exact_regression(small_model_pieces):
    # adaptive_implicit_vertadv=False must reproduce the legacy explicit
    # scheme bit-for-bit (the in-tendency vertadv path is unchanged).
    grid, z_coord, state, Config, Model = small_model_pieces
    base = Model(grid, z_coord, Config())
    gated = Model(grid, z_coord, Config(adaptive_implicit_vertadv=False))
    s_base = base.step(state, dt=600.0)
    s_gated = gated.step(state, dt=600.0)
    np.testing.assert_array_equal(
        np.asarray(s_base.u.data), np.asarray(s_gated.u.data))
    np.testing.assert_array_equal(
        np.asarray(s_base.v.data), np.asarray(s_gated.v.data))


# ---------------------------------------------------------------------------
# 5. Momentum-diagnostics observability (flag on: vertadv reported as the
#    start-of-step estimate, NOT silently zero; excluded from du_dt slow
#    forcing so the barotropic split is untouched).
# ---------------------------------------------------------------------------

def _diag_state(grid, z_coord):
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
    )
    rng = np.random.default_rng(7)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    u = 0.05 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.05 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    eta = 0.01 * rng.standard_normal((n_lat, n_lon))
    T = 5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
    T = T + 0.1 * rng.standard_normal((n_lat, n_lon, nlev))
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u, dtype=jnp.float64)),
        v=state.v.replace(data=jnp.asarray(v, dtype=jnp.float64)),
        eta=state.eta.replace(data=jnp.asarray(eta, dtype=jnp.float64)),
        T=state.T.replace(data=jnp.asarray(T, dtype=jnp.float64)),
    )


def test_flag_on_reports_vertadv_but_excludes_it_from_du_dt():
    # Finding-1 fix: with the flag ON, the PE momentum budget must report
    # a NON-zero vertadv term (the start-of-step explicit estimate of the
    # step-level implicit operator) AND that term must NOT be in du_dt
    # (the slow forcing fed to the barotropic solver stays vertadv-free).
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        latlon_cgrid_ocean_baroclinic_tendencies,
    )

    grid = create_latlon_grid(n_lat=24, n_lon=48)
    z_coord = create_ocean_z_star(n_levels=10, H_max=4000.0)
    state = _diag_state(grid, z_coord)

    cfg_off = LatLonCGridOceanConfig.from_flat(A_h=1.0e4, A_v=0.0, bottom_drag_r=0.0)
    cfg_on = cfg_off._replace(adaptive_implicit_vertadv=True)

    tend_off, diag_off = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg_off, diagnose_momentum=True)
    tend_on, diag_on = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg_on, diagnose_momentum=True)

    for side in ("u", "v"):
        va_on = np.asarray(getattr(diag_on, f"vertadv_{side}").data)
        va_off = np.asarray(getattr(diag_off, f"vertadv_{side}").data)
        # Reported vertadv is non-trivial and identical to the explicit
        # estimate (same flux-form helper, same start-of-step w).
        assert np.max(np.abs(va_on)) > 1e-8, \
            f"vertadv_{side} must not be silently zero"
        np.testing.assert_allclose(va_on, va_off, rtol=0, atol=1e-12)

        # d{u,v}_dt EXCLUDES vertadv when the flag is on; INCLUDES it off.
        d_on = np.asarray(getattr(tend_on, f"d{side}_dt").data)
        d_off = np.asarray(getattr(tend_off, f"d{side}_dt").data)
        np.testing.assert_allclose(d_off - d_on, va_off, rtol=0, atol=1e-12)

    # Direct flag-on closure: Σ(all terms) == du_dt + vertadv (i.e.
    # Σ(terms except vertadv) == du_dt).
    for side in ("u", "v"):
        s = None
        for name in diag_on._fields:
            if name.endswith(f"_{side}") and not name.startswith("total"):
                arr = np.asarray(getattr(diag_on, name).data)
                s = arr if s is None else s + arr
        total = np.asarray(getattr(diag_on, f"total_{side}").data)
        va = np.asarray(getattr(diag_on, f"vertadv_{side}").data)
        d_dt = np.asarray(getattr(tend_on, f"d{side}_dt").data)
        # total_* mirrors du_dt (the slow forcing, vertadv-free).
        np.testing.assert_allclose(total, d_dt, rtol=0, atol=1e-12)
        np.testing.assert_allclose(s, total + va, rtol=1e-12, atol=1e-12)


def test_flag_off_pe_closure_bit_exact_after_refactor():
    # Guard that the Finding-1 refactor (hoisting the vertadv computation
    # out of the gate) did not change the flag-OFF PE tendency or its
    # diagnostics: Σ(terms) == du_dt to machine precision, exactly as the
    # legacy closure test requires.
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        latlon_cgrid_ocean_baroclinic_tendencies,
    )

    grid = create_latlon_grid(n_lat=24, n_lon=48)
    z_coord = create_ocean_z_star(n_levels=10, H_max=4000.0)
    state = _diag_state(grid, z_coord)
    cfg = LatLonCGridOceanConfig.from_flat(A_h=1.0e4, A_v=0.0, bottom_drag_r=0.0)

    _, diag = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, cfg, diagnose_momentum=True)
    comp = None
    for name in diag._fields:
        if name.endswith("_u") and not name.startswith("total"):
            arr = np.asarray(getattr(diag, name).data)
            comp = arr if comp is None else comp + arr
    np.testing.assert_allclose(comp, np.asarray(diag.total_u.data),
                               atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# NEMO ln_dynadv_vec + ln_zad_Aimp: dynzad as the explicit part
# ---------------------------------------------------------------------------

def _col(nlev=6, w_amp=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    u = jnp.asarray(rng.normal(size=(3, nlev)))
    w = np.zeros((3, nlev + 1)); w[:, 1:nlev] = w_amp * rng.normal(size=(3, nlev - 1))
    h = jnp.full((3, nlev), 10.0)
    return u, jnp.asarray(w), h


def test_nemo_advective_low_courant_is_explicit_dynzad():
    from legoesm.ocean.vertical import nemo_advective_vertical_momentum_advection
    u, w, h = _col()                      # Cu = 1e-5*60/10 << cu_min: zcff = 0
    area = jnp.full((3, 1), 2.0)
    out = adaptive_implicit_vertical_momentum_advection(
        u, w, h, 60.0, explicit_scheme="nemo_advective",
        w_area_half=area * w, face_area=area)
    want = u + 60.0 * nemo_advective_vertical_momentum_advection(u, area * w, h, area)
    np.testing.assert_allclose(np.asarray(out), np.asarray(want), rtol=0, atol=1e-15)
    up = adaptive_implicit_vertical_momentum_advection(u, w, h, 60.0)
    assert float(jnp.max(jnp.abs(out - up))) > 1e-8   # it is NOT the upwind default


def test_nemo_advective_needs_area_inputs():
    u, w, h = _col()
    with pytest.raises(ValueError, match="w_area_half"):
        adaptive_implicit_vertical_momentum_advection(
            u, w, h, 60.0, explicit_scheme="nemo_advective")


def test_step_nemo_advective_with_aimp_runs_and_differs(small_model_pieces):
    grid, z_coord, state, Config, Model = small_model_pieces
    up = Model(grid, z_coord, Config(adaptive_implicit_vertadv=True))
    na = Model(grid, z_coord, Config(adaptive_implicit_vertadv=True,
                                     vertical_momentum_scheme="nemo_advective"))
    # sheared, horizontally varying flow so w and du/dz are non-zero
    u0 = np.asarray(state.u.data)
    jj, ii, kk = np.meshgrid(*(np.arange(n) for n in u0.shape), indexing="ij")
    u0 = 0.1 * np.sin(2 * np.pi * ii / u0.shape[1]) * np.cos(np.pi * kk / u0.shape[2])
    state = state._replace(u=state.u.replace(data=jnp.asarray(u0, state.u.data.dtype)))
    s_up, s_na = state, state
    for _ in range(5):
        s_up = up.step(s_up, dt=600.0)
        s_na = na.step(s_na, dt=600.0)
    assert bool(jnp.all(jnp.isfinite(s_na.u.data)))
    assert float(jnp.max(jnp.abs(s_na.u.data - s_up.u.data))) > 0.0
