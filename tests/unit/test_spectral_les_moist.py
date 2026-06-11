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
from legoesm.atmosphere.dynamics import spectral_les_plane as sl
from legoesm.atmosphere.dynamics.spectral_les_moist import (
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


def test_adapter_rejects_too_few_slots():
    nz, dz = 8, 50.0
    ref = _ref_for(nz, dz)
    micro = make_les_microphysics_fn(
        MicrophysicsConfig(scheme="morrison",
                           morrison=MorrisonConfig(morrison_flavor="sam")),
        ref, dz, 5.0)
    with pytest.raises(ValueError, match="tracer slots"):
        micro(jnp.full((2, 2, nz), 300.0), jnp.zeros((2, 2, nz, 3)))
