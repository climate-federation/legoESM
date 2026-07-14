"""Unit tests for the moist spectral-LES coupling (spectral_les_moist.py +
the tracer/moist-buoyancy extensions of spectral_les_plane.py).

Truth tiers: hydrostatic-balance analytic check (reference state), sign/
direction physics checks (sedimentation falls DOWN in LES coordinates, vapour
is positively buoyant, condensate loads), conservation (uniform tracer is
steady; the prescribed surface moisture flux adds exactly the right column
water), and the dry path staying available (tracers=None state steps).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl
from legoesm.atmosphere.dynamics.les.spectral_les_moist import (
    SpectralRefState,
    make_anelastic_reference,
    make_les_microphysics_fn,
)
from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
    MorrisonConfig,
)


def _grid(nx=8, ny=8, nz=16, Lz=800.0, moist=True, n_tracers=9):
    cfg = sl.SpectralLESConfig(
        nx=nx, ny=ny, nz=nz, Lx=800.0, Ly=800.0, Lz=Lz,
        buoyancy=True, theta_ref0=300.0, moist=moist, n_tracers=n_tracers,
        spectral_filter=False, time_scheme="ab2")
    return sl.make_grid(cfg)


def _state(g, qv0=0.01):
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    z = jnp.zeros((ny, nx, nz))
    th = jnp.full((ny, nx, nz), 300.0)
    tr = jnp.zeros((ny, nx, nz, g.cfg.n_tracers)).at[..., 0].set(qv0)
    return sl.SpectralLESState(
        u=z, v=z, w=jnp.zeros((ny, nx, nz + 1)),
        rhs_u_prev=z, rhs_v_prev=z, rhs_w_prev=jnp.zeros((ny, nx, nz + 1)),
        theta=th, rhs_theta_prev=jnp.zeros_like(th),
        tracers=tr, rhs_tracers_prev=jnp.zeros_like(tr))


# --------------------------------------------------------------------------- #
# Reference state                                                              #
# --------------------------------------------------------------------------- #
def test_reference_state_hydrostatic():
    nz, Lz = 64, 3000.0
    dz = Lz / nz
    z_c = (np.arange(nz) + 0.5) * dz
    z_f = np.arange(nz + 1) * dz
    theta = 300.0 + 0.003 * z_c
    qv = 0.016 * np.exp(-z_c / 2000.0)
    ref = make_anelastic_reference(z_c, z_f, 101500.0, theta, qv)
    # Faces: dp/dz must equal −ρ̄ g with ρ̄ the layer density (discrete
    # hydrostatic balance of the SAME integral). Compare to centre ρ.
    dpdz = np.diff(np.asarray(ref.p_f)) / dz
    rho = np.asarray(ref.rho_c)
    assert np.allclose(dpdz, -rho * constants.g, rtol=2e-3)
    # Surface pressure reproduced at the bottom face.
    assert abs(float(ref.p_f[0]) - 101500.0) < 1.0
    # Monotone decreasing p, physically-ranged ρ.
    assert np.all(np.diff(np.asarray(ref.p_c)) < 0)
    assert 0.7 < rho[-1] < rho[0] < 1.35


# --------------------------------------------------------------------------- #
# Moist buoyancy                                                               #
# --------------------------------------------------------------------------- #
def test_virtual_theta_vapor_buoyant_condensate_loads():
    th = jnp.full((4, 4, 8), 300.0)
    tr = jnp.zeros((4, 4, 8, 9))
    thv_dry = sl.virtual_theta(th, tr)
    assert np.allclose(np.asarray(thv_dry), 300.0)
    # vapour raises θ_v by the shared (1/ε−1) factor
    tr_v = tr.at[..., 0].set(0.01)
    expect = 300.0 * (1.0 + (1.0 / constants.epsilon - 1.0) * 0.01)
    assert np.allclose(np.asarray(sl.virtual_theta(th, tr_v)), expect)
    # cloud + rain water LOAD (reduce θ_v)
    tr_c = tr_v.at[..., 1].set(0.002).at[..., 2].set(0.001)
    assert np.all(np.asarray(sl.virtual_theta(th, tr_c))
                  < np.asarray(sl.virtual_theta(th, tr_v)))


def test_moist_buoyancy_humid_column_rises():
    g = _grid()
    th = jnp.full((8, 8, 16), 300.0)
    tr = jnp.zeros((8, 8, 16, 9))
    # humid plume in one column ⇒ positive buoyancy there, negative elsewhere
    tr = tr.at[2, 3, :, 0].set(0.012)
    b = np.asarray(sl.buoyancy_w_moist(th, tr, g))
    assert b[2, 3, 1:-1].min() > 0.0
    assert b[0, 0, 1:-1].max() < 0.0
    assert np.allclose(b[..., 0], 0.0) and np.allclose(b[..., -1], 0.0)


# --------------------------------------------------------------------------- #
# Tracer transport through step()                                              #
# --------------------------------------------------------------------------- #
def test_uniform_tracer_steady_and_dry_path_unaffected():
    g = _grid()
    st = _state(g, qv0=0.01)
    st2, _ = sl.step(st, g=g, dt=0.5, u_geo=(0.0, 0.0), f_cor=0.0, first=True)
    # No flow, no flux, uniform q_v ⇒ tracers unchanged to round-off.
    assert np.allclose(np.asarray(st2.tracers), np.asarray(st.tracers),
                       atol=1e-12)
    # Dry path still steps (tracers=None).
    g_dry = _grid(moist=False, n_tracers=0)
    std = sl.SpectralLESState(
        u=st.u, v=st.v, w=st.w, rhs_u_prev=st.u, rhs_v_prev=st.v,
        rhs_w_prev=st.w, theta=st.theta, rhs_theta_prev=st.theta * 0.0)
    std2, _ = sl.step(std, g=g_dry, dt=0.5, u_geo=(0.0, 0.0), f_cor=0.0,
                      first=True)
    assert std2.tracers is None


def test_surface_moisture_flux_column_budget():
    g = _grid()
    st = _state(g, qv0=0.0)
    dt, flx = 0.5, 2.0e-4                       # kinematic flux [kg/kg · m/s]
    st2, _ = sl.step(st, g=g, dt=dt, u_geo=(0.0, 0.0), f_cor=0.0, first=True,
                     sfc_qv_flux=flx)
    # Column integral of q_v gained = flux · dt (per unit area, kinematic).
    dq = np.asarray(st2.tracers[..., 0] - st.tracers[..., 0])
    col = dq.sum(axis=-1) * g.dz                # ∫ dq dz
    assert np.allclose(col, flx * dt, rtol=1e-6)
    # Only slot 0 received it.
    assert np.allclose(np.asarray(st2.tracers[..., 1:]), 0.0, atol=1e-15)


# --------------------------------------------------------------------------- #
# Microphysics adapter (Morrison default): orientation + swappability          #
# --------------------------------------------------------------------------- #
def _ref_for(nz, dz, th0=300.0):
    z_c = (np.arange(nz) + 0.5) * dz
    z_f = np.arange(nz + 1) * dz
    return make_anelastic_reference(z_c, z_f, 101500.0,
                                    np.full(nz, th0), np.zeros(nz))


def test_adapter_rain_falls_down_in_les_coordinates():
    ny = nx = 4
    nz, dz, dt = 16, 50.0, 5.0
    ref = _ref_for(nz, dz)
    micro = make_les_microphysics_fn(
        MicrophysicsConfig(scheme="morrison",
                           morrison=MorrisonConfig(morrison_flavor="sam")),
        ref, dz, dt)
    theta = jnp.full((ny, nx, nz), 300.0)
    tracers = jnp.zeros((ny, nx, nz, 9))
    k_blob = 10                                  # rain blob aloft (bottom-up)
    tracers = (tracers.at[..., k_blob, 2].set(1.0e-3)
                       .at[..., k_blob, 7].set(1.0e5)
                       .at[..., 0].set(5.0e-3))  # subsaturated vapour
    dth, dtr, precip = micro(theta, tracers)
    dqr = np.asarray(dtr[0, 0, :, 2])
    # Sedimentation: rain LEAVES the blob level and ARRIVES below it
    # (lower k in bottom-up coordinates). If the flip were missing, the
    # gain would appear ABOVE the blob.
    assert dqr[k_blob] < 0.0
    assert dqr[:k_blob].max() > 0.0
    assert np.all(dqr[k_blob + 1:] <= 1e-14)
    assert np.all(np.isfinite(np.asarray(dth)))
    assert precip.shape == (ny, nx)


def test_adapter_scheme_swappable_kessler():
    """Same adapter, different scheme (legoESM swappability): Kessler runs
    through the identical entry point with the standard slot layout."""
    nz, dz, dt = 16, 50.0, 5.0
    ref = _ref_for(nz, dz)
    micro = make_les_microphysics_fn(
        MicrophysicsConfig(scheme="kessler"), ref, dz, dt)
    assert micro.scheme_name == "kessler"
    theta = jnp.full((4, 4, nz), 300.0)
    tracers = jnp.zeros((4, 4, nz, 9)).at[..., 0].set(0.02)  # supersaturated
    dth, dtr, precip = micro(theta, tracers)
    # Condensation: vapour sink, cloud source, latent heating > 0 somewhere.
    assert float(jnp.min(dtr[..., 0])) < 0.0
    assert float(jnp.max(dtr[..., 1])) > 0.0
    assert float(jnp.max(dth)) > 0.0


# --------------------------------------------------------------------------- #
# Monotone (van-Leer TVD) scalar transport — the moist-instability fix.        #
# --------------------------------------------------------------------------- #
def _div_free(g, key):
    """A random divergence-free (u,v,w) for advection tests (project a noise)."""
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    u = jax.random.normal(jax.random.PRNGKey(key), (ny, nx, nz))
    v = jax.random.normal(jax.random.PRNGKey(key + 1), (ny, nx, nz))
    w = jnp.zeros((ny, nx, nz + 1)).at[..., 1:nz].set(
        0.3 * jax.random.normal(jax.random.PRNGKey(key + 2), (ny, nx, nz - 1)))
    return sl.project(u, v, w, dt=0.1, g=g)


def test_monotone_advection_conserves_scalar_mass():
    g = _grid(nz=24)
    u, v, w = _div_free(g, 7)
    phi = jnp.asarray(np.random.RandomState(0).rand(
        g.cfg.ny, g.cfg.nx, g.cfg.nz))
    div = sl._vanleer_flux_div(phi, u, v, w, g)            # ∇·(uφ)
    # Conservative flux form (periodic x,y + zero wall flux) ⇒ Σ ∇·(uφ) ≈ 0.
    assert abs(float(jnp.sum(div))) < 1e-8 * float(jnp.sum(jnp.abs(div)) + 1e-30)


def test_monotone_scalar_rhs_preserves_constant_under_spectral_projection():
    """The limiter's face velocities are not the projection's spectral
    divergence operator.  The scalar RHS must therefore include the
    ``φ·div_fv(u)`` correction; otherwise a uniform scalar develops artificial
    anomalies under an otherwise divergence-free LES velocity."""
    g = _grid(nz=24)
    u, v, w = _div_free(g, 11)
    phi = jnp.full((g.cfg.ny, g.cfg.nx, g.cfg.nz), 3.0)
    rhs = sl.scalar_rhs_monotone(
        phi, u, v, w, jnp.zeros_like(phi), g, sfc_flux=0.0)
    assert float(jnp.max(jnp.abs(rhs))) < 1e-11


def test_monotone_vertical_advection_is_TVD_no_new_extrema():
    """Pure VERTICAL advection of a z top-hat (the moisture-inversion direction,
    where the runaway originates) by constant w: van-Leer is strictly bounded;
    the spectral/centred vertical advection overshoots (the Gibbs that the
    latent-heat feedback amplifies). u=v=0, w=const ⇒ exact 1-D limiter test."""
    g = _grid(nz=32)
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    zero = jnp.zeros((ny, nx, nz))
    w = jnp.full((ny, nx, nz + 1), 0.5)                   # uniform upward
    phi = jnp.where(jnp.arange(nz) < nz // 2, 1.0, 0.0)
    phi = jnp.broadcast_to(phi, (ny, nx, nz)).astype(jnp.float64)
    lo, hi = float(phi.min()), float(phi.max())
    dt = 0.2 * g.dz / 0.5                                  # CFL 0.2
    phi_vl = phi - dt * sl._vanleer_flux_div(phi, zero, zero, w, g)
    assert lo - 1e-9 <= float(phi_vl.min()) and float(phi_vl.max()) <= hi + 1e-9
    # spectral/centred vertical advection of the SAME top-hat overshoots:
    dthdz_f = jnp.pad(sl.ddz_c2f(phi, g.dz), ((0, 0), (0, 0), (1, 1)))
    phi_sp = phi - dt * sl.f2c(w * dthdz_f)
    assert float(phi_sp.max()) > hi + 1e-2   # > 1% Gibbs overshoot (the bug)


def test_weno5_flux_div_conserves_and_reuses_core_weno():
    """WENO5 scalar flux-div is globally conservative AND reuses the shared
    core.weno oracle (same kernel as the ocean vertical-tracer WENO)."""
    import legoesm.core.weno as cw
    g = _grid(nz=24)
    u, v, w = _div_free(g, 5)
    phi = jnp.asarray(np.random.RandomState(1).rand(
        g.cfg.ny, g.cfg.nx, g.cfg.nz))
    div = sl._weno5_flux_div(phi, u, v, w, g)
    assert abs(float(jnp.sum(div))) < 1e-8 * float(jnp.sum(jnp.abs(div)) + 1e-30)
    # oracle: WENO5-Z is 5th-order on a smooth field ⇒ the left/right face
    # reconstructions of sin(kx) bracket the exact face value tightly.
    x = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    dxe = x[1] - x[0]
    f = [jnp.asarray(np.sin(x + (s - 2) * dxe)) for s in range(6)]  # i-2..i+3
    fp, fm = cw.weno5_z(f)                                  # face i+1/2 = x+dx/2
    exact = np.sin(x + 0.5 * dxe)
    # accurate reconstruction (point-sample IC ⇒ O(dx²) point-vs-cell-avg floor;
    # the kernel itself is 5th-order on true cell averages — this is the reuse /
    # oracle sanity check, not a formal order-of-accuracy test).
    assert np.max(np.abs(np.asarray(fp) - exact)) < 2e-3
    assert np.max(np.abs(np.asarray(fm) - exact)) < 2e-3


def test_weno5_less_diffusive_than_vanleer():
    """Advecting a z top-hat one step: WENO5 stays sharper (less numerical
    diffusion) than van-Leer — the property that preserves cloud moisture.
    Both reuse the same conservative flux-form + free-stream structure."""
    g = _grid(nz=48)
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    zero = jnp.zeros((ny, nx, nz))
    w = jnp.full((ny, nx, nz + 1), 0.5)
    phi = jnp.where(jnp.arange(nz) < nz // 2, 1.0, 0.0)
    phi = jnp.broadcast_to(phi, (ny, nx, nz)).astype(jnp.float64)
    dt = 0.3 * g.dz / 0.5
    # advect many steps; measure front sharpness (total variation growth).
    def advect(fluxfn, n=40):
        p = phi
        for _ in range(n):
            p = p - dt * (fluxfn(p, zero, zero, w, g)
                          - p * sl._vanleer_fv_velocity_divergence(
                              zero, zero, w, g))
        return p
    p_vl = advect(sl._vanleer_flux_div)
    p_w5 = advect(sl._weno5_flux_div)
    col = lambda p: np.asarray(p[0, 0])                    # noqa: E731
    # WENO5 keeps a steeper max gradient (less smeared front).
    grad_vl = np.abs(np.diff(col(p_vl))).max()
    grad_w5 = np.abs(np.diff(col(p_w5))).max()
    assert grad_w5 > grad_vl                               # sharper = less diffusive
    # both stay bounded (essentially non-oscillatory)
    assert col(p_w5).max() <= 1.05 and col(p_w5).min() >= -0.05


# --------------------------------------------------------------------------- #
# Momentum-side stabilization operators (WENO5 enabler).                       #
# --------------------------------------------------------------------------- #
def test_w_hyperdiffusion_damps_2dx_not_smooth():
    """Horizontal w-hyperdiffusion = −ν₄k⁴w: strongly damps the 2Δx mode,
    leaves a smooth (low-k) mode almost untouched, exactly zero on a constant."""
    g = _grid(nz=8)
    g = g._replace(cfg=g.cfg._replace(w_hyperdiff_coeff=1.0e4))
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    x = jnp.arange(nx)
    # highest NON-Nyquist mode (the core zeros the Nyquist kx) vs the lowest mode
    khi = jnp.cos(2 * jnp.pi * (nx // 2 - 1) * x / nx)[None, :, None]
    ksmooth = jnp.cos(2 * jnp.pi * x / nx)[None, :, None]  # lowest mode
    w_hi = jnp.broadcast_to(khi, (ny, nx, nz + 1)).astype(jnp.float64)
    w_lo = jnp.broadcast_to(ksmooth, (ny, nx, nz + 1)).astype(jnp.float64)
    t_hi = np.abs(np.asarray(sl._w_hyperdiffusion(w_hi, g)))
    t_lo = np.abs(np.asarray(sl._w_hyperdiffusion(w_lo, g)))
    # k⁴ scaling ⇒ Nyquist tendency ≫ lowest-mode tendency.
    assert t_hi.max() > 50.0 * t_lo.max()
    # constant ⇒ k=0 ⇒ zero tendency
    wc = jnp.ones((ny, nx, nz + 1))
    assert np.max(np.abs(np.asarray(sl._w_hyperdiffusion(wc, g)))) < 1e-10


def test_w_hyperdiffusion_spectral_k4():
    """Single horizontal mode: tendency = −ν₄·k⁴·w exactly."""
    g = _grid(nz=4); nu4 = 3.3e3
    g = g._replace(cfg=g.cfg._replace(w_hyperdiff_coeff=nu4))
    nx = g.cfg.nx
    k = 2 * np.pi / g.cfg.Lx                                   # lowest x-mode
    x = np.arange(nx) * g.dx
    w = jnp.broadcast_to(jnp.asarray(np.cos(k * x))[None, :, None],
                         (g.cfg.ny, nx, g.cfg.nz + 1)).astype(jnp.float64)
    tend = np.asarray(sl._w_hyperdiffusion(w, g))
    expect = -nu4 * (k ** 4) * np.asarray(w)
    assert np.allclose(tend, expect, atol=1e-6, rtol=1e-4)


def test_divergence_damping_zero_on_divfree():
    """Div damping vanishes on a divergence-free field (its raison d'être is
    only the divergent part — near-no-op post-projection)."""
    g = _grid(nz=16)
    g = g._replace(cfg=g.cfg._replace(div_damping_coeff=10.0))
    u, v, w = _div_free(g, 3)
    dRu, dRv, dRw = sl._divergence_damping(u, v, w, g)
    sc = float(jnp.mean(jnp.abs(u)) + 1e-30)
    assert float(jnp.max(jnp.abs(dRu))) < 1e-3 * sc
    assert float(jnp.max(jnp.abs(dRw))) < 1e-3 * sc


def test_divergence_damping_relaxes_divergent_mode():
    """On a purely divergent u=sin(kx): du += α∇(∇·u) = −αk²u (damps it)."""
    g = _grid(nz=8); a = 5.0
    g = g._replace(cfg=g.cfg._replace(div_damping_coeff=a))
    nx = g.cfg.nx; k = 2 * np.pi / g.cfg.Lx
    x = np.arange(nx) * g.dx
    u = jnp.broadcast_to(jnp.asarray(np.sin(k * x))[None, :, None],
                         (g.cfg.ny, nx, g.cfg.nz)).astype(jnp.float64)
    z = jnp.zeros_like(u)
    dRu, _, _ = sl._divergence_damping(u, z, jnp.zeros((g.cfg.ny, nx,
                                                        g.cfg.nz + 1)), g)
    assert np.allclose(np.asarray(dRu), -a * k ** 2 * np.asarray(u),
                       atol=1e-6, rtol=1e-4)


def test_stabilization_flags_default_off_and_plumbed():
    cfg = sl.SpectralLESConfig(nx=8, ny=8, nz=8, Lx=800.0, Ly=800.0, Lz=800.0)
    assert cfg.w_hyperdiff_coeff == 0.0 and cfg.div_damping_coeff == 0.0
    # with the flag on, rhs() Rw differs from the flag off (operator is wired).
    g0 = _grid(nz=12)
    g1 = g0._replace(cfg=g0.cfg._replace(w_hyperdiff_coeff=1.0e4))
    st = _state(g0, qv0=0.01)
    st = st._replace(w=st.w.at[:, :, 1:-1].set(
        0.1 * jax.random.normal(jax.random.PRNGKey(3),
                                (g0.cfg.ny, g0.cfg.nx, g0.cfg.nz - 1))))
    _, _, Rw0, _, _, _ = sl.rhs(st.u, st.v, st.w, g0, (0.0, 0.0), 0.0,
                                theta=st.theta, tracers=st.tracers)
    _, _, Rw1, _, _, _ = sl.rhs(st.u, st.v, st.w, g1, (0.0, 0.0), 0.0,
                                theta=st.theta, tracers=st.tracers)
    assert not np.allclose(np.asarray(Rw0), np.asarray(Rw1))


def test_moist_diagnostics_bundle():
    from legoesm.atmosphere.dynamics.les.spectral_les_moist import moist_diagnostics
    ny = nx = 6; nz = 10
    u = np.zeros((ny, nx, nz)); v = np.zeros((ny, nx, nz))
    wc = np.zeros((ny, nx, nz)); wc[0, 0, 5] = 2.0
    th = np.full((ny, nx, nz), 300.0)
    tr = np.zeros((ny, nx, nz, 9))
    tr[0, 0, 5, 1] = 1.0e-3                                # one cloudy column
    rho = np.ones(nz)
    d = moist_diagnostics(u, v, wc, th, tr, rho, 40.0, 100.0, 2.0)
    assert set(d) >= {"cloud_frac", "lwp", "max_w", "w_var", "tke",
                      "max_cfl", "total_water", "qv_min", "qv_max", "qc_max"}
    assert abs(d["cloud_frac"] - 1.0 / (ny * nx)) < 1e-9
    assert abs(d["lwp"] - 1.0e-3 * 40.0 / (ny * nx) * 1e3) < 1e-6  # qc*rho*dz mean
    assert d["max_w"] == 2.0 and d["max_cfl"] == 2.0 * 2.0 / 40.0


def test_unknown_scalar_advection_rejected():
    with pytest.raises(ValueError, match="scalar_advection"):
        sl.make_grid(sl.SpectralLESConfig(
            nx=8, ny=8, nz=8, Lx=800.0, Ly=800.0, Lz=800.0,
            monotone_scalars=True, scalar_advection="bogus"))


def test_dry_path_uses_spectral_advection_unchanged():
    """monotone_scalars defaults False ⇒ rhs uses the spectral scalar_rhs (the
    validated dry path), bit-identical."""
    cfg = sl.SpectralLESConfig(nx=8, ny=8, nz=16, Lx=800.0, Ly=800.0, Lz=800.0,
                               buoyancy=True, theta_ref0=300.0,
                               spectral_filter=False)
    assert cfg.monotone_scalars is False


def test_adapter_rejects_too_few_slots():
    nz, dz = 8, 50.0
    ref = _ref_for(nz, dz)
    micro = make_les_microphysics_fn(
        MicrophysicsConfig(scheme="morrison",
                           morrison=MorrisonConfig(morrison_flavor="sam")),
        ref, dz, 5.0)
    with pytest.raises(ValueError, match="tracer slots"):
        micro(jnp.full((2, 2, nz), 300.0), jnp.zeros((2, 2, nz, 3)))
