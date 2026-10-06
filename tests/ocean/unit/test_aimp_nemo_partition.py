"""NEMO wAimp_RK3_t adaptive-implicit partition (``aimp_partition='nemo_rk3_t'``).

Kernel (sshwzv.F90:826-843), tracer share fused into the shared implicit
tracer matrix (trazdf.F90:207-215, flux form), momentum share in NEMO's
vector-invariant advective form (dynzdf.F90:235-250), the tripole wiring,
and the read-only census of both trigger rules.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.physics.vertical_mixing.implicit_solver import (  # noqa: E402
    implicit_vertical_diffusion_ocean,
)
from legoesm.ocean.vertical import (  # noqa: E402
    adaptive_implicit_vertical_momentum_advection,
    implicit_vertical_advection_ocean_advective,
    nemo_aimp_fraction,
    shchepetkin_implicit_fraction,
)


def _frac(cu_v, cu_h=0.0, w=1.0):
    cu_v = jnp.asarray([cu_v], dtype=jnp.float64)
    cu_h = jnp.full((2,), cu_h, dtype=jnp.float64)
    return float(nemo_aimp_fraction(cu_v, cu_h, jnp.asarray([w]))[0])


# --------------------------------------------------------------------------
# 1. kernel = NEMO's formula
# --------------------------------------------------------------------------
def test_kernel_matches_hand_formula():
    # cu_h = 0: thresholds 0.8 / 1.1, Fcu = 4*1.1*0.3, cut = 1.4
    assert _frac(0.5) == 0.0
    assert _frac(0.95) == pytest.approx(0.15**2 / (1.32 + 0.15**2), rel=1e-14)
    assert _frac(2.0) == pytest.approx((2.0 - 1.1) / 2.0, rel=1e-14)
    # cu_h = 0.55 halves both thresholds (1 - 0.55/1.1 = 0.5): 0.4 / 0.55
    assert _frac(0.5, cu_h=0.55) == pytest.approx(
        0.1**2 / (4 * 0.55 * 0.15 + 0.1**2), rel=1e-14)
    # the legacy 0.15/0.30 rule fires at Cu=0.5; NEMO's must not
    assert float(shchepetkin_implicit_fraction(jnp.asarray(0.5))) > 0.0


def test_kernel_agrees_with_legacy_ramp_at_nemo_thresholds():
    cu = jnp.linspace(0.0, 5.0, 501, dtype=jnp.float64)
    got = nemo_aimp_fraction(cu, jnp.zeros(502), jnp.ones(501))
    ref = shchepetkin_implicit_fraction(cu, 0.8, 1.1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-13, atol=1e-15)


def test_kernel_gradient_finite_at_branch_edges():
    # Cu = 0, Cu_min exactly, inside the ramp, at the cut, and implicit
    cu = jnp.asarray([0.0, 0.8, 0.95, 1.4, 2.0], dtype=jnp.float64)

    def total(c, h):
        return jnp.sum(nemo_aimp_fraction(c, h, jnp.ones(5)))
    g_cu, g_h = jax.grad(total, argnums=(0, 1))(cu, jnp.zeros(6))
    assert np.all(np.isfinite(np.asarray(g_cu))) and np.all(np.isfinite(np.asarray(g_h)))
    assert float(g_cu[2]) > 0.0 and float(g_cu[0]) == 0.0


def test_kernel_gradient_finite_fp32_at_horizontal_limit():
    # codex r3 case: Cu_h at the 1.1 limit, tiny Cu_v inside the ramp
    cu = jnp.asarray([9.5403195e-8], dtype=jnp.float32)
    ch = jnp.asarray([1.0999999, 1.0999999], dtype=jnp.float32)
    g = jax.grad(lambda c, h: jnp.sum(nemo_aimp_fraction(c, h, jnp.ones(1, jnp.float32))),
                 argnums=(0, 1))(cu, ch)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in g)


def test_partition_helper_gradient_finite_with_zero_w():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import nemo_aimp_implicit_w
    grid = SimpleNamespace(area_T=jnp.full((1, 2), 1.0e8),
                           dy_u=jnp.full((1, 3), 1.0e4), dx_v=jnp.full((2, 2), 1.0e4))
    h = jnp.full((1, 2, 3), 10.0)

    def total(scale):
        w = scale * jnp.asarray([[[0.0, 0.0, -0.02, 0.0], [0.0, 1e-5, 0.0, 0.0]]])
        return jnp.sum(nemo_aimp_implicit_w(jnp.zeros((1, 3, 3)), jnp.zeros((2, 2, 3)),
                                            w, h, h, jnp.ones((1, 2, 3)), grid, 1000.0))
    g = jax.grad(total)(1.0)
    assert np.isfinite(float(g)) and float(g) != 0.0


def test_kernel_reads_upstream_cell_horizontal_courant():
    cu_v = jnp.asarray([0.9], dtype=jnp.float64)
    cu_h = jnp.asarray([0.0, 0.6])            # [above, below] the interface
    up = float(nemo_aimp_fraction(cu_v, cu_h, jnp.asarray([1.0]))[0])    # from below
    down = float(nemo_aimp_fraction(cu_v, cu_h, jnp.asarray([-1.0]))[0])  # from above
    assert up > 0.0 and down == pytest.approx(_frac(0.9), rel=1e-14)
    assert up != down


# --------------------------------------------------------------------------
# 2. tracer share: flux form in the shared implicit matrix
# --------------------------------------------------------------------------
def _col(seed, nlev=6):
    r = np.random.default_rng(seed)
    dz = jnp.asarray(r.uniform(5.0, 50.0, nlev))
    dzh = 0.5 * (dz[:-1] + dz[1:])
    K = jnp.asarray(r.uniform(1e-5, 1e-2, nlev - 1))
    return r, dz, dzh, K


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_tracer_share_conserves_column_content(sign):
    r, dz, dzh, K = _col(0)
    wi = np.zeros(7); wi[1:-1] = sign * r.uniform(1e-4, 5e-3, 5)
    T = jnp.asarray(r.normal(10.0, 3.0, 6))
    Tn = implicit_vertical_diffusion_ocean(T, K, dz, dzh, 3600.0,
                                           implicit_w=jnp.asarray(wi))
    assert float(jnp.sum(dz * Tn)) == pytest.approx(float(jnp.sum(dz * T)), rel=1e-13)
    assert float(jnp.max(jnp.abs(Tn - implicit_vertical_diffusion_ocean(
        T, K, dz, dzh, 3600.0)))) > 1e-6          # the share did something


def _constancy_residual(sign, flip=False):
    """Uniform T0 through the explicit remainder + implicit share must come
    back T0: T* = T0 (1 + dt (wi_top - wi_bot)/h_new) is what the explicit
    stage leaves when it advects with w - wi under full-w continuity."""
    r, dz, dzh, K = _col(1)
    dt = 3600.0
    wi = np.zeros(7); wi[1:-1] = sign * r.uniform(1e-4, 5e-3, 5)
    T0 = 7.3
    Tstar = T0 * (1.0 + dt * (wi[:-1] - wi[1:]) / np.asarray(dz))
    used = -wi if flip else wi
    Tn = implicit_vertical_diffusion_ocean(jnp.asarray(Tstar), K, dz, dzh, dt,
                                           implicit_w=jnp.asarray(used))
    return float(jnp.max(jnp.abs(Tn - T0)))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_tracer_share_preserves_constancy(sign):
    assert _constancy_residual(sign) < 1e-12


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_tracer_share_planted_sign_flip_breaks_constancy(sign):
    assert _constancy_residual(sign, flip=True) > 1e-3


def test_tracer_share_known_downward_w_two_levels():
    dz = jnp.asarray([10.0, 20.0]); dzh = jnp.asarray([15.0])
    dt, w = 100.0, -0.02
    T = jnp.asarray([3.0, 1.0])
    Tn = implicit_vertical_diffusion_ocean(T, jnp.zeros(1), dz, dzh, dt,
                                           implicit_w=jnp.asarray([0.0, w, 0.0]))
    t0 = 3.0 / (1.0 + dt * abs(w) / 10.0)
    t1 = 1.0 + dt * abs(w) / 20.0 * t0
    np.testing.assert_allclose(np.asarray(Tn), [t0, t1], rtol=1e-14)


def test_tracer_share_zero_is_bitwise_identity():
    r, dz, dzh, K = _col(2)
    T = jnp.asarray(r.normal(10.0, 3.0, 6))
    np.testing.assert_array_equal(
        np.asarray(implicit_vertical_diffusion_ocean(T, K, dz, dzh, 600.0,
                                                     implicit_w=jnp.zeros(7))),
        np.asarray(implicit_vertical_diffusion_ocean(T, K, dz, dzh, 600.0)))


# --------------------------------------------------------------------------
# 3. momentum share: NEMO dynzdf vector-invariant (advective) form
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_advective_share_keeps_uniform_field(sign):
    r = np.random.default_rng(3)
    h = jnp.asarray(r.uniform(5.0, 60.0, 8))
    w = np.zeros(9); w[1:-1] = sign * r.uniform(1e-3, 3e-2, 7)
    w[-2] = min(w[-2], 0.0)       # bottom cell: no upward share (see next test)
    u = jnp.full(8, 0.37)
    un = implicit_vertical_advection_ocean_advective(u, jnp.asarray(w), h, 900.0)
    assert float(jnp.max(jnp.abs(un - 0.37))) < 1e-14


def test_advective_share_known_w_both_signs():
    h0, h1, dt, wv = 10.0, 30.0, 200.0, 0.01
    u = jnp.asarray([1.0, 0.4]); h = jnp.asarray([h0, h1])
    # downward: upwind from above into level 1; level 0 untouched
    dn = implicit_vertical_advection_ocean_advective(
        u, jnp.asarray([0.0, -wv, 0.0]), h, dt)
    wn = -dt * 0.5 * wv / (0.5 * (h0 + h1))
    np.testing.assert_allclose(np.asarray(dn), [1.0, (0.4 - wn * 1.0) / (1.0 - wn)],
                               rtol=1e-14)
    # upward: level 0 from level 1; the bottom level couples to the masked
    # zero below it (NEMO dynzdf.F90:237-243 loops to jpkm1), so it decays
    up = implicit_vertical_advection_ocean_advective(
        u, jnp.asarray([0.0, wv, 0.0]), h, dt)
    wp0 = dt * 0.5 * wv / (0.5 * (h0 + h1))
    wp1 = dt * 0.5 * wv / (0.5 * h1)
    u1 = 0.4 / (1.0 + wp1)
    np.testing.assert_allclose(np.asarray(up), [(1.0 + wp0 * u1) / (1.0 + wp0), u1],
                               rtol=1e-14)


def test_advective_share_inactive_rows_untouched_and_masked_below():
    h = jnp.asarray([10.0, 10.0, 10.0]); act = jnp.asarray([1.0, 1.0, 0.0])
    u = jnp.asarray([1.0, 1.0, 5.0])          # garbage below the seafloor
    un = implicit_vertical_advection_ocean_advective(
        u, jnp.asarray([0.0, 0.01, 0.0, 0.0]), h, 100.0, face_active=act)
    assert float(un[2]) == 5.0
    # level 1 relaxes toward ZERO below, never toward the garbage 5.0
    assert float(un[1]) < 1.0


def test_momentum_zero_share_equals_legacy_when_both_rules_idle():
    r = np.random.default_rng(4)
    u = jnp.asarray(r.normal(0.0, 0.2, (3, 6)))
    h = jnp.asarray(r.uniform(50.0, 300.0, (3, 6)))
    w = np.zeros((3, 7)); w[:, 1:-1] = r.uniform(-1e-6, 1e-6, (3, 5))
    w = jnp.asarray(w)
    area = jnp.full((3, 1), 1.0e8)
    kw = dict(explicit_scheme="nemo_advective", w_area_half=w * area,
              face_area=area)
    legacy = adaptive_implicit_vertical_momentum_advection(u, w, h, 600.0, **kw)
    new = adaptive_implicit_vertical_momentum_advection(
        u, w, h, 600.0, w_imp_half=jnp.zeros_like(w),
        w_imp_area_half=jnp.zeros_like(w), implicit_form="advective", **kw)
    np.testing.assert_array_equal(np.asarray(new), np.asarray(legacy))


def test_momentum_unknown_implicit_form_raises():
    u = jnp.zeros((1, 3)); w = jnp.zeros((1, 4)); h = jnp.ones((1, 3))
    with pytest.raises(ValueError, match="implicit_form"):
        adaptive_implicit_vertical_momentum_advection(u, w, h, 1.0,
                                                      implicit_form="vector")


# --------------------------------------------------------------------------
# 4. tripole wiring: partition helper, validation, census
# --------------------------------------------------------------------------
def test_partition_helper_masks_and_scales():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import nemo_aimp_implicit_w
    ny, nx, nl = 2, 3, 4
    grid = SimpleNamespace(area_T=jnp.full((ny, nx), 1.0e8),
                           dy_u=jnp.full((ny, nx + 1), 1.0e4),
                           dx_v=jnp.full((ny + 1, nx), 1.0e4))
    h = jnp.full((ny, nx, nl), 10.0)
    act = jnp.ones((ny, nx, nl)).at[0, 0, 3].set(0.0)
    w = jnp.zeros((ny, nx, nl + 1)).at[..., 1:nl].set(-0.02)   # Cu_v = 2 at dt=1000
    wi = nemo_aimp_implicit_w(jnp.zeros((ny, nx + 1, nl)), jnp.zeros((ny + 1, nx, nl)),
                              w, h, h, act, grid, 1000.0)
    f = float(wi[1, 1, 2] / w[1, 1, 2])
    assert f == pytest.approx((2.0 - 1.1) / 2.0, rel=1e-12)
    assert float(wi[0, 0, 3]) == 0.0                 # interface onto a dry cell
    assert float(jnp.max(jnp.abs(wi[..., 0]))) == 0.0
    assert float(jnp.max(jnp.abs(wi[..., -1]))) == 0.0


_N_LAT, _N_LON, _DT = 12, 16, 1800.0


@pytest.fixture
def fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _basin(scheme="nemo_advective", **cfg_kw):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    grid = create_latlon_grid(_N_LAT, _N_LON)
    z = create_ocean_z_star(n_levels=6, H_max=4000.0)
    lm = np.ones((_N_LAT, _N_LON)); lm[:2] = 0.0; lm[-2:] = 0.0
    state = rest_state_latlon_cgrid_ocean(
        grid, z, land_mask_override=jnp.asarray(lm),
        H_bathy_override=jnp.full((_N_LAT, _N_LON), 4000.0))
    lat = np.degrees(np.asarray(grid.lat))
    T = np.asarray(state.T.data) + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
        enable_runtime_checks=False, barotropic_solver="implicit_cn",
        vertical_momentum_scheme=scheme,
        adaptive_implicit_vertadv=True, **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z, cfg)


@pytest.mark.parametrize("bad, match", [
    (dict(aimp_partition="nemo"), "aimp_partition must be one of"),
    (dict(aimp_partition="nemo_rk3_t", adaptive_implicit_vertadv=False),
     "needs adaptive_implicit_vertadv"),
    (dict(aimp_partition="nemo_rk3_t", vertical_momentum_scheme="centered_full"),
     "is wired for"),
    (dict(aimp_partition="nemo_rk3_t", implicit_vertical_mixing=False),
     "not wired for implicit_vertical_mixing"),
    (dict(aimp_partition="nemo_rk3_t", outer_integrator="ab2"),
     "not wired for outer_integrator"),
    (dict(aimp_partition="nemo_rk3_t", tracer_time_integrator="rk3"),
     "not wired for tracer_time_integrator"),
    (dict(aimp_partition="nemo_rk3_t", tracer_advection="fct2"),
     "not wired for tracer_advection"),
])
def test_validation_raises(bad, match):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    kw = dict(barotropic_solver="implicit_cn", vertical_momentum_scheme="nemo_advective",
              adaptive_implicit_vertadv=True)
    kw.update(bad)
    with pytest.raises(ValueError, match=match):
        LatLonCGridOceanModel(create_latlon_grid(_N_LAT, _N_LON),
                              create_ocean_z_star(n_levels=4, H_max=4000.0),
                              LatLonCGridOceanConfig.from_flat(**kw))


@pytest.mark.parametrize("scheme", ["upwind_perturbation", "nemo_advective"])
def test_idle_rules_give_bitwise_identical_trajectory_and_census_counts(scheme, fp64):
    """Weak flow: neither rule fires, so nemo_rk3_t must reproduce the legacy
    trajectory bit for bit (every added term is an exact zero), and the
    census must see wet interfaces but no activation."""
    s0, m0 = _basin(scheme)
    s1, m1 = _basin(scheme, aimp_partition="nemo_rk3_t")
    seen = []
    m1._aimp_census_callback = lambda c, a, b, cmx: seen.append(np.asarray(c))
    f0, _ = m0.integrate_scan(s0, n_steps=4, dt=_DT)
    f1, _ = m1.integrate_scan(s1, n_steps=4, dt=_DT)
    jax.effects_barrier()
    for a, b in ((f0.T.data, f1.T.data), (f0.S.data, f1.S.data),
                 (f0.u.data, f1.u.data), (f0.v.data, f1.v.data),
                 (f0.eta.data, f1.eta.data)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert len(seen) == 4 and all(c.shape == (5,) for c in seen)
    assert all(c[4] > 0 and c[0] == c[1] == c[2] == c[3] == 0 for c in seen)


def test_cli_aimp_flags_parse_and_default_off():
    from scripts.run import run_omip_core2 as runner
    ap = runner._build_arg_parser()
    a = ap.parse_args([])
    assert a.aimp_partition is None and a.aimp_census is False
    a = ap.parse_args(["--aimp-partition", "nemo_rk3_t", "--aimp-census"])
    assert a.aimp_partition == "nemo_rk3_t" and a.aimp_census is True
    with pytest.raises(SystemExit):
        ap.parse_args(["--aimp-partition", "nemo"])


def _forced_firing_run(monkeypatch, partition, flip_tracer_share=False):
    """Uniform T, S gradient drives the flow; the NEMO fraction is FORCED to
    0.5 on every interface so the share is large in-model.  Returns
    (T spread after 4 steps, relative change of sum(area*h*S))."""
    import legoesm.ocean.vertical as V
    import legoesm.ocean.physics.vertical_mixing.implicit_solver as IS
    from legoesm.ocean.vertical import compute_layer_thickness
    monkeypatch.setattr(V, "nemo_aimp_fraction",
                        lambda cu_v, cu_h, w: jnp.full_like(cu_v, 0.5))
    if flip_tracer_share:
        orig = IS._build_implicit_tridiag

        def flipped(*a, implicit_w=None, **k):
            return orig(*a, implicit_w=None if implicit_w is None else -implicit_w, **k)
        monkeypatch.setattr(IS, "_build_implicit_tridiag", flipped)
    kw = {} if partition is None else dict(aimp_partition=partition)
    state, model = _basin("upwind_perturbation", **kw)
    lat = np.degrees(np.asarray(model.grid.lat))
    wet = np.asarray(state.land_mask.data)[..., None] > 0.5
    S = 35.0 + 0.5 * np.tanh(lat / 15.0)[:, None, None] * wet * np.ones(state.S.data.shape)
    state = state._replace(T=state.T.replace(data=jnp.where(wet, 10.0, state.T.data)),
                           S=state.S.replace(data=jnp.asarray(S)))
    z = model.z_coord if hasattr(model, "z_coord") else None

    def content(st):
        h = compute_layer_thickness(st.eta.data, st.H_bathy.data, z)
        a = jnp.asarray(getattr(model.grid.area_T, "data", model.grid.area_T))[..., None]
        return float(jnp.sum(a * h * st.S.data * wet))
    f, _ = model.integrate_scan(state, n_steps=4, dt=_DT)
    T = np.asarray(f.T.data)[np.broadcast_to(wet, f.T.data.shape)]
    return float(T.max() - T.min()), abs(content(f) / content(state) - 1.0), f


def test_forced_share_keeps_constancy_and_conserves(monkeypatch, fp64):
    spread_l, drift_l, f_l = _forced_firing_run(monkeypatch, None)
    spread_n, drift_n, f_n = _forced_firing_run(monkeypatch, "nemo_rk3_t")
    assert spread_l < 1e-10                       # the harness resolves constancy
    assert spread_n < 1e-10
    assert drift_n <= 10.0 * drift_l + 1e-13
    assert float(jnp.max(jnp.abs(f_n.S.data - f_l.S.data))) > 1e-13  # share is live
    du = float(jnp.max(jnp.abs(f_n.u.data - f_l.u.data)))
    assert du > 1e3 * np.finfo(np.float64).eps * float(jnp.max(jnp.abs(f_l.u.data)))


def test_forced_share_planted_sign_flip_breaks_constancy(monkeypatch, fp64):
    spread, _, _ = _forced_firing_run(monkeypatch, "nemo_rk3_t", flip_tracer_share=True)
    assert spread > 1e-8


def test_census_accumulator_window_max_and_drain():
    from scripts.run import run_omip_core2 as runner
    acc = runner._AimpCensusAccumulator()
    acc(np.array([1, 2, 3, 0, 9]), np.array([0.1, 0.0]), np.array([0.0, 0.2]),
        np.array([0.5, 0.9]))
    acc(np.array([0, 0, 0, 1, 9]), np.array([0.0, 0.3]), np.array([0.0, 0.1]),
        np.array([1.2, 0.1]))
    out = acc.drain()
    assert out["aimp_counts"].shape == (2, 5)
    np.testing.assert_array_equal(out["aimp_max_legacy"], [0.1, 0.3])
    np.testing.assert_array_equal(out["aimp_max_nemo"], [0.0, 0.2])
    np.testing.assert_array_equal(out["aimp_cmx_v"], [1.2, 0.9])
    assert acc.drain() == {}
