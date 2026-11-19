"""KPP-Langmuir wave-enhanced surface mixing validation.

Covers: default-off byte-identity + param inertness, enhancement >= classical
KPP, the published VR12-AL factor, Stokes-drift La_t dependence, and
differentiability.
"""

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import wright_eos
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.physics.vertical_mixing.kpp import (
    kpp_vertical_mixing, _kpp_langmuir_enhancement,
)

jax.config.update("jax_enable_x64", True)


def _state(n_levels=12, H_max=400.0):
    z = create_ocean_z_star(n_levels=n_levels, H_max=H_max)
    shape = (6, 2, 2, n_levels)
    T_prof = jnp.concatenate([jnp.full((4,), 18.0), jnp.linspace(18.0, 4.0, n_levels - 4)])
    T = jnp.broadcast_to(T_prof, shape).astype(jnp.float64)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    rho = wright_eos(T, S, jnp.zeros_like(T))
    u = jnp.broadcast_to(jnp.linspace(0.2, 0.0, n_levels), shape).astype(jnp.float64)
    v = jnp.zeros(shape, dtype=jnp.float64)
    eta = jnp.zeros((6, 2, 2), dtype=jnp.float64)
    J = jnp.ones((6, 2, 2), dtype=jnp.float64)
    return u, v, T, S, rho, eta, z, J


def _run(cfg, u_stokes=None):
    u, v, T, S, rho, eta, z, J = _state()
    tau_x = jnp.full((6, 2, 2), 0.1, dtype=jnp.float64)
    tau_y = jnp.zeros((6, 2, 2), dtype=jnp.float64)
    B_f = jnp.full((6, 2, 2), 1e-7, dtype=jnp.float64)
    return kpp_vertical_mixing(u, v, T, S, rho, eta, z, J, cfg,
                               tau_x=tau_x, tau_y=tau_y, B_f=B_f,
                               apply_diffusion=False, u_stokes=u_stokes)


def test_default_off_byte_identical():
    """enable_langmuir=False is bit-identical and inert to the Langmuir params."""
    off = _run(KPPConfig(enable_langmuir=False))
    off2 = _run(KPPConfig(enable_langmuir=False, langmuir_c1=4.0,
                          langmuir_number_default=0.2))
    assert jnp.array_equal(off.K_v, off2.K_v)
    assert jnp.array_equal(off.A_v, off2.A_v)


def test_enhances_boundary_layer_mixing():
    """Langmuir enhances (never reduces) BL diffusivity/viscosity."""
    off = _run(KPPConfig(enable_langmuir=False))
    on = _run(KPPConfig(enable_langmuir=True))
    assert jnp.all(on.K_v >= off.K_v - 1e-12)
    assert jnp.all(on.A_v >= off.A_v - 1e-12)
    # Strictly larger somewhere in the boundary layer.
    assert jnp.max(on.K_v - off.K_v) > 1e-6


def test_enhancement_factor_matches_formula():
    """Independent La-form reference: Li et al. 2016 Table 2 / Eq. 12."""
    import numpy as np
    cfg = KPPConfig(enable_langmuir=True)
    la = np.array([0.2, 0.3, 0.5, 1.0, 3.0])
    expected = np.sqrt(1 + (3.1 * la)**-2 + (5.4 * la)**-4)
    got = _kpp_langmuir_enhancement(jnp.full(la.shape, 0.01),
                                   jnp.asarray(0.01 / la**2), cfg)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    assert cfg.langmuir_c1 == 3.1 and cfg.langmuir_c2 == 5.4


def test_zero_stokes_is_exact_no_wave_limit():
    off = _run(KPPConfig(enable_langmuir=False))
    on = _run(KPPConfig(enable_langmuir=True), u_stokes=jnp.zeros((6, 2, 2)))
    assert jnp.array_equal(off.K_v, on.K_v)
    assert jnp.array_equal(off.A_v, on.A_v)


def test_langmuir_reaches_unresolved_shear_boundary_depth(monkeypatch):
    """Observe the production V_t2 input, including its production call chain.

    A coarse saturated Ri crossing can leave h unchanged even when V_t2
    changes, so a depth-only assertion would be a vacuous wiring test.
    """
    import legoesm.ocean.physics.vertical_mixing.kpp as impl
    seen = []
    original = impl._kpp_unresolved_shear_variance

    def capture(N, d, w_s, cfg, eps):
        seen.append(w_s)
        return original(N, d, w_s, cfg, eps)

    monkeypatch.setattr(impl, "_kpp_unresolved_shear_variance", capture)
    _run(KPPConfig(enable_langmuir=False))
    _run(KPPConfig(enable_langmuir=True), u_stokes=jnp.full((6, 2, 2), 0.02))
    # Independent reference in the La_t form, at this fixture's exact stress.
    from legoesm.ocean.eos import rho_0
    ustar = ((0.1**2 + impl._EPS)**0.5 / rho_0)**0.5
    la = (ustar / 0.02)**0.5
    expected = (1 + (3.1 * la)**-2 + (5.4 * la)**-4)**0.5
    assert len(seen) == 2
    assert jnp.allclose(seen[1] / seen[0], expected, rtol=1e-12)
    assert expected > 1.0


def test_langmuir_redistributes_heat_salt_and_momentum_conservatively():
    u, v, T, S, _, eta, z, J = _state()
    S = S + jnp.linspace(0.0, 1.0, S.shape[-1])
    rho = wright_eos(T, S, jnp.zeros_like(T))
    out = kpp_vertical_mixing(
        u, v, T, S, rho, eta, z, J, KPPConfig(enable_langmuir=True),
        tau_x=jnp.full(eta.shape, 0.1), tau_y=jnp.zeros_like(eta),
        B_f=jnp.full(eta.shape, 1e-9),
        Q_sfc_T=jnp.full(eta.shape, -1e-6),
        Q_sfc_S=jnp.full(eta.shape, 1e-8),
        u_stokes=jnp.full(eta.shape, 0.2))
    for tendency in (out.du_dt, out.dT_dt, out.dS_dt):
        content = tendency * z.dz_ref
        scale = jnp.sum(jnp.abs(content), axis=-1)
        assert jnp.all(scale > 0.0)
        assert jnp.all(jnp.abs(jnp.sum(content, axis=-1)) <= 1e-10 * scale)
        # Planted uncompensated boundary source must violate the same budget.
        corrupted = content.at[..., 0].add(scale)
        assert jnp.all(jnp.abs(jnp.sum(corrupted, axis=-1)) > 1e-10 * scale)


def test_stokes_drift_controls_langmuir_number():
    """Larger surface Stokes drift -> smaller La_t -> stronger enhancement
    than the default fully-developed-sea fallback."""
    default = _run(KPPConfig(enable_langmuir=True))            # La_t = 0.3 fallback
    # u* = sqrt(0.1/rho_0) ~ 0.0099; u_stokes=0.25 -> La_t ~ 0.20 < 0.3.
    strong = _run(KPPConfig(enable_langmuir=True),
                  u_stokes=jnp.full((6, 2, 2), 0.25, dtype=jnp.float64))
    assert jnp.max(strong.K_v) > jnp.max(default.K_v)


def test_differentiable():
    """grad flows through the Langmuir enhancement (w.r.t. Stokes drift)."""
    def loss(us):
        out = _run(KPPConfig(enable_langmuir=True),
                   u_stokes=jnp.full((6, 2, 2), us, dtype=jnp.float64))
        return jnp.mean(out.K_v)
    g = jax.grad(loss)(0.2)
    assert jnp.isfinite(g)
    assert g > 0.0
    fd = (loss(0.20001) - loss(0.19999)) / 0.00002
    assert jnp.allclose(g, fd, rtol=1e-4, atol=1e-10)


def test_implicit_k_profile_path_reads_u_stokes_from_surface_forcing():
    """The K-profile composer (the coupled driver's implicit path) forwards
    ``OceanSurfaceForcing.u_stokes`` into KPP; dropping the pass-through makes
    the two runs identical."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        compute_vertical_K_profiles,
    )
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=8, n_lon=12)
    z = create_ocean_z_star(n_levels=5, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)
    # This composer path expects cell-centre velocities (C-grid faces here).
    u_c = 0.5 * (state.u.data[:, :-1, :] + state.u.data[:, 1:, :])
    v_c = 0.5 * (state.v.data[:-1, :, :] + state.v.data[1:, :, :])
    state = state._replace(
        u=state.u.replace(data=u_c), v=state.v.replace(data=v_c))

    defaults = OceanPhysicsConfig()
    cfg = defaults._replace(
        vertical_mixing=VerticalMixingConfig(
            scheme="kpp", kpp=KPPConfig(K_conv=1.0, enable_langmuir=True)),
        lateral_mixing=defaults.lateral_mixing._replace(scheme="none"),
        surface_forcing=defaults.surface_forcing._replace(scheme="none"),
        convection=defaults.convection._replace(scheme="none"),
        shortwave_penetration=None,
    )
    shape = (8, 12)
    tau = jnp.full(shape, 0.1, dtype=jnp.float64)

    def run(u_stokes):
        sf = OceanSurfaceForcing(
            tau_x=tau, tau_y=jnp.zeros_like(tau), u_stokes=u_stokes)
        return compute_vertical_K_profiles(state, z, sf, cfg)

    K_default, A_default = run(None)
    K_stokes, A_stokes = run(jnp.full(shape, 0.01, dtype=jnp.float64))
    assert jnp.all(jnp.isfinite(K_stokes))
    assert jnp.max(jnp.abs(K_stokes - K_default)) > 1e-6
    assert jnp.max(jnp.abs(A_stokes - A_default)) > 1e-6
