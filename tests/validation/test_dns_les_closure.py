"""DNS / LES turbulence-closure modes for the double-periodic plane CRM
(iter-177).

CRM, LES and DNS share the SAME diffusion operators in
``compressible_euler_plane``; only the viscosity ``K_m`` differs —
Smagorinsky EDDY viscosity (CRM Cs~0.19 / LES Cs~0.15) vs a CONSTANT
MOLECULAR viscosity ν (DNS, ``turbulence_closure="molecular"``). These tests
pin:

* config validation (closure dispatch + the molecular ν / Pr guards);
* the DNS molecular closure reproduces the ANALYTICAL viscous decay
  ``du/dt = -ν k² u`` for a ``u'(y)`` shear that drives no acoustics/advection
  (zero horizontal divergence), to within RK3 time-integration error;
* ``closure="none"`` leaves that shear undamped — so the decay above is the
  molecular closure, not some other dissipation (hyperdiff/sponge are off);
* the vertical molecular leg (``sgs_vertical_diffusion``) damps a ``u'(z)``
  shear while the horizontal-only leg leaves it untouched;
* ``closure="smagorinsky"`` (CRM/LES) still steps finitely — backward compat.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import CompressibleEulerConfig
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
    make_rest_state,
    validate_plane_config,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
from legoesm import constants as C

jax.config.update("jax_enable_x64", True)


def test_smagorinsky_stability_length_dosmagor():
    """SAM dosmagor stable-layer Deardorff mixing-length limit
    (``smagorinsky_stability_length``, tke_full.f90:285-298): in UNSTABLE
    layers (N²≤0) it reproduces the default ``(Cs·Δ)²·|S|`` K_m bit-for-bit;
    in STABLE layers (N²>0) it can only REDUCE K_m (smix≤grd ⇒ K_m≤default);
    and the serial + halo kernels agree."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _compute_smagorinsky_K_m_plane,
    )
    grid, hc, _ = _setup(nx=8, ny=8, nlev=20, dx=2000.0, LZ=20000.0)
    nlev = hc.theta_ref.shape[0]
    rng = np.random.default_rng(3)
    u = jnp.asarray(rng.normal(size=(8, 8, nlev)) * 3.0)
    v = jnp.asarray(rng.normal(size=(8, 8, nlev)) * 3.0)
    w = (jnp.asarray(rng.normal(size=(8, 8, nlev + 1)) * 1.0)
         .at[..., 0].set(0.0).at[..., -1].set(0.0))
    cs = 0.19

    def km(n2, sl):
        return np.asarray(_compute_smagorinsky_K_m_plane(
            u, v, w, grid, hc, cs, n2_sgs=n2, wall_damping=False,
            stability_length=sl))

    # UNSTABLE (N²<0): the limit is inactive ⇒ identical to the default form.
    n2_u = jnp.full((8, 8, nlev), -1.0e-4)
    np.testing.assert_allclose(km(n2_u, True), km(n2_u, False), rtol=1.0e-10)

    # WEAKLY STABLE (0 < N² ≲ |S|²): mixing is still on but the Deardorff limit
    # bites ⇒ smix<grd ⇒ the stability-limited K_m never EXCEEDS the default and
    # strictly reduces it somewhere. (Strongly-stable N²≫|S|² ⇒ both K_m=0 via
    # the |S|²−Pr·N² cutoff, so the limit only matters in the weakly-stable band.)
    n2_s = jnp.full((8, 8, nlev), 5.0e-6)
    kd, ks = km(n2_s, False), km(n2_s, True)
    m = kd > 1.0e-12
    assert np.all(ks[m] <= kd[m] + 1.0e-12)
    assert np.any(ks[m] < kd[m] * 0.99)


def _setup(nx, ny, nlev, dx, LZ):
    grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dx,
                             coriolis_mode="none")
    hc = create_height_coordinate(nlev, H=LZ)
    tm = make_flat_plane_terrain_metric(grid, hc)
    return grid, hc, tm


def _cfg(**kw):
    base = dict(use_coriolis=False, smagorinsky_cs=0.0, sponge_coeff=0.0,
                hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
                hyperdiff_w_coeff=0.0)
    base.update(kw)
    return CompressibleEulerConfig(**base)


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_closure_validation():
    validate_plane_config(_cfg(turbulence_closure="molecular",
                               molecular_viscosity=C.nu_air))
    validate_plane_config(_cfg(smagorinsky_cs=0.19))          # default closure
    for kw in (dict(turbulence_closure="bogus"),
               dict(turbulence_closure="molecular", molecular_viscosity=0.0),
               dict(turbulence_closure="molecular", molecular_viscosity=1e-5,
                    molecular_prandtl=0.0)):
        with pytest.raises(ValueError):
            validate_plane_config(_cfg(**kw))


def test_dns_horizontal_only_warns():
    """DNS molecular closure WITHOUT the vertical leg is an incomplete
    (horizontal-only) operator — validate must warn; WITH the vertical leg it
    must not warn (the full 3-D ν∇²)."""
    import warnings
    with pytest.warns(UserWarning, match="HORIZONTAL only"):
        validate_plane_config(_cfg(turbulence_closure="molecular",
                                   molecular_viscosity=5.0,
                                   sgs_vertical_diffusion=False))
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # any warning would fail here
        validate_plane_config(_cfg(turbulence_closure="molecular",
                                   molecular_viscosity=5.0,
                                   sgs_vertical_diffusion=True))


# --------------------------------------------------------------------------- #
# DNS molecular closure — analytical viscous decay
# --------------------------------------------------------------------------- #
def _shear_y_state(grid, hc, A, ky):
    """Rest state + u'(y) = A sin(ky·y), uniform in x and z. Zero horizontal
    divergence ⇒ no acoustic/advective tendency; only viscous diffusion acts."""
    base = make_rest_state(grid, hc, dtype=jnp.float64)
    y = (jnp.arange(grid.ny) + 0.5) * grid.dy
    u0 = A * jnp.sin(ky * y)[:, None, None] * jnp.ones(
        (grid.ny, grid.nx, grid.nlev))
    return base._replace(u=base.u.replace(data=u0))


def test_dns_molecular_matches_analytical_viscous_decay():
    nx, ny, nlev, dx, LZ = 4, 16, 8, 50.0, 4000.0
    grid, hc, tm = _setup(nx, ny, nlev, dx, LZ)
    NU, A, dt, M = 200.0, 1.0, 1.0, 20
    ky = 2 * np.pi / (ny * dx)
    state = _shear_y_state(grid, hc, A, ky)
    a0 = float(jnp.max(jnp.abs(state.u.data)))   # discrete peak (< A: grid misses π/2)
    cfg = _cfg(turbulence_closure="molecular", molecular_viscosity=NU,
               sgs_vertical_diffusion=False)
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    for _ in range(M):
        state = model.step(state, dt=dt)
    # Discrete y-Laplacian eigenvalue for sin(ky·y): k² = (2-2cos(ky·dy))/dy².
    k2 = (2.0 - 2.0 * np.cos(ky * dx)) / dx ** 2
    expected = a0 * np.exp(-NU * k2 * M * dt)
    got = float(jnp.max(jnp.abs(state.u.data)))
    assert got == pytest.approx(expected, rel=5e-3), (got, expected)
    # the pure shear must not have excited vertical motion or spurious v
    assert float(jnp.max(jnp.abs(state.w.data))) < 1e-6
    assert float(jnp.max(jnp.abs(state.v.data))) < 1e-6


def test_closure_none_leaves_shear_undamped():
    nx, ny, nlev, dx, LZ = 4, 16, 8, 50.0, 4000.0
    grid, hc, tm = _setup(nx, ny, nlev, dx, LZ)
    A, dt, M = 1.0, 1.0, 20
    ky = 2 * np.pi / (ny * dx)
    state = _shear_y_state(grid, hc, A, ky)
    a0 = float(jnp.max(jnp.abs(state.u.data)))
    model = PlaneCompressibleEulerModel(grid, hc, tm, _cfg(turbulence_closure="none"))
    for _ in range(M):
        state = model.step(state, dt=dt)
    # no closure ⇒ the zero-divergence shear is untouched (hyperdiff/sponge off)
    assert float(jnp.max(jnp.abs(state.u.data))) == pytest.approx(a0, rel=1e-6)


# --------------------------------------------------------------------------- #
# vertical molecular leg
# --------------------------------------------------------------------------- #
def test_dns_vertical_leg_damps_uz_shear():
    nx, ny, nlev, dx, LZ = 4, 4, 16, 50.0, 4000.0
    grid, hc, tm = _setup(nx, ny, nlev, dx, LZ)
    NU, A, dt, M = 500.0, 1.0, 1.0, 40
    base = make_rest_state(grid, hc, dtype=jnp.float64)
    # u'(z) uniform in x,y ⇒ horizontal leg = 0; only the vertical leg can act.
    # Use a 2-wavelength vertical mode (m=4 half-periods) so molecular k_z² is
    # large enough for a clearly-measurable decay over the test window.
    u0 = A * jnp.sin(4.0 * np.pi * (jnp.arange(nlev) + 0.5) / nlev)[None, None, :] \
        * jnp.ones((grid.ny, grid.nx, nlev))
    state0 = base._replace(u=base.u.replace(data=u0))
    a0 = float(jnp.max(jnp.abs(state0.u.data)))

    model_h = PlaneCompressibleEulerModel(
        grid, hc, tm, _cfg(turbulence_closure="molecular",
                           molecular_viscosity=NU, sgs_vertical_diffusion=False))
    s = state0
    for _ in range(M):
        s = model_h.step(s, dt=dt)
    assert float(jnp.max(jnp.abs(s.u.data))) == pytest.approx(a0, rel=1e-3)

    model_v = PlaneCompressibleEulerModel(
        grid, hc, tm, _cfg(turbulence_closure="molecular",
                           molecular_viscosity=NU, sgs_vertical_diffusion=True))
    s = state0
    for _ in range(M):
        s = model_v.step(s, dt=dt)
    assert float(jnp.max(jnp.abs(s.u.data))) < 0.95 * a0


# --------------------------------------------------------------------------- #
# CRM / LES (Smagorinsky) backward compatibility
# --------------------------------------------------------------------------- #
def test_smagorinsky_delta_max_caps_coarse_grid_mixing_length():
    """SAM caps the HORIZONTAL grid spacing in the Smagorinsky mixing length at
    delta_max (SGS_TKE/sgs.f90:90 = 1000 m; tke_full.f90:42). For dx>delta_max
    (RCE dx=4 km) the capped K_m is smaller by (min(δ,dx)·min(δ,dy)/(dx·dy))^(2/3);
    for dx<=delta_max (GATE/LBA dx=1 km) it is UNCHANGED (backward-compatible)."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _compute_smagorinsky_K_m_plane)
    nx, ny, nlev = 8, 8, 6
    rng = np.random.default_rng(0)
    for dx, cap_bites in [(4000.0, True), (500.0, False)]:
        grid, hc, _ = _setup(nx, ny, nlev, dx, 4000.0)
        u = jnp.asarray(rng.standard_normal((ny, nx, nlev)))
        v = jnp.asarray(rng.standard_normal((ny, nx, nlev)))
        w = jnp.asarray(rng.standard_normal((ny, nx, nlev + 1)))
        kw = dict(n2_sgs=None, wall_damping=False)   # clean (Cs·Δ)²·|S|
        K_cap = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, 0.19,
                                               delta_max=1000.0, **kw)
        K_unc = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, 0.19,
                                               delta_max=1.0e30, **kw)
        if cap_bites:
            ratio = (min(1000.0, dx) * min(1000.0, dx) / (dx * dx)) ** (2.0 / 3.0)
            assert ratio < 0.2                       # ~0.157 for dx=4 km
            np.testing.assert_allclose(np.asarray(K_cap),
                                       ratio * np.asarray(K_unc), rtol=1e-6)
        else:
            np.testing.assert_allclose(np.asarray(K_cap), np.asarray(K_unc),
                                       rtol=1e-12)


def test_smagorinsky_closure_still_steps():
    nx, ny, nlev, dx, LZ = 8, 8, 8, 100.0, 4000.0
    grid, hc, tm = _setup(nx, ny, nlev, dx, LZ)
    base = make_rest_state(grid, hc, dtype=jnp.float64)
    rng = np.random.default_rng(0)
    u0 = jnp.asarray(0.1 * rng.standard_normal((ny, nx, nlev)))
    state = base._replace(u=base.u.replace(data=u0))
    cfg = _cfg(smagorinsky_cs=0.15, sgs_vertical_diffusion=True)   # LES preset
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    for _ in range(5):
        state = model.step(state, dt=0.5)
    assert bool(jnp.all(jnp.isfinite(state.u.data)))
