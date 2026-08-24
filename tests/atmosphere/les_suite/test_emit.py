"""Tests for the LES→artifact emission diagnostics (les_suite/emit.py).

The SGS-flux sign convention is the load-bearing physics here, so it gets explicit
analytic checks (CLAUDE.md sign-convention gate).
"""
from __future__ import annotations

import numpy as np
import pytest
from legoesm.atmosphere.les_suite.bridge import LESReferenceArtifact
from legoesm.atmosphere.les_suite.emit import (
    build_reference_artifact,
    horizontal_mean,
    resolved_vertical_flux,
    sgs_vertical_scalar_flux_mean,
)

NY, NX, NZ = 4, 5, 8


def _z(nz=NZ):
    return np.linspace(10.0, 1600.0, nz)


def test_horizontal_mean_reduces_to_nz():
    f = np.random.default_rng(0).normal(size=(NY, NX, NZ))
    m = horizontal_mean(f)
    assert m.shape == (NZ,)
    assert np.allclose(m, f.mean(axis=(0, 1)))


def test_horizontal_mean_rejects_non_3d():
    with pytest.raises(ValueError):
        horizontal_mean(np.zeros((NY, NX)))


def test_resolved_flux_zero_for_uniform_w():
    # no vertical-velocity fluctuation ⇒ zero resolved flux
    w = np.ones((NY, NX, NZ))
    phi = np.random.default_rng(1).normal(size=(NY, NX, NZ))
    flux = resolved_vertical_flux(w, phi)
    assert np.allclose(flux, 0.0)


def test_resolved_flux_positive_correlation_upward():
    # w' and phi' positively correlated ⇒ positive (upward) flux
    rng = np.random.default_rng(2)
    base = rng.normal(size=(NY, NX, NZ))
    w = base.copy()
    phi = base.copy()  # perfectly correlated
    flux = resolved_vertical_flux(w, phi)
    assert np.all(flux >= -1e-12)
    assert np.any(flux > 0)


def test_resolved_flux_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        resolved_vertical_flux(np.zeros((NY, NX, NZ)), np.zeros((NY, NX, NZ + 1)))


# --- SGS flux sign convention (analytic) --------------------------------------
DZ = float(_z()[1] - _z()[0])  # uniform spacing of the test grid


def test_sgs_flux_stable_layer_is_downward():
    # ∂θ/∂z > 0 (stable) with K_h > 0 ⇒ interior SGS heat flux NEGATIVE (downward).
    z = _z()
    theta_col = 300.0 + 0.01 * z            # increasing with height
    theta = np.broadcast_to(theta_col, (NY, NX, NZ)).copy()
    nu_t = np.full((NY, NX, NZ), 5.0)
    flux = sgs_vertical_scalar_flux_mean(theta, nu_t, DZ, pr_sgs=1.0)
    assert np.all(flux < 0.0)
    # interior magnitude = K_h * dθ/dz = 5 * 0.01 = 0.05 (endpoints blend with BCs)
    assert np.allclose(flux[1:-1], -0.05, atol=1e-6)


def test_sgs_flux_superadiabatic_surface_is_upward():
    # ∂θ/∂z < 0 (super-adiabatic) ⇒ SGS heat flux POSITIVE (upward) — the
    # near-surface flux the resolved field misses.
    z = _z()
    theta_col = 300.0 - 0.02 * z            # decreasing with height
    theta = np.broadcast_to(theta_col, (NY, NX, NZ)).copy()
    nu_t = np.full((NY, NX, NZ), 3.0)
    flux = sgs_vertical_scalar_flux_mean(theta, nu_t, DZ, pr_sgs=1.0)
    assert np.all(flux > 0.0)


def test_sgs_flux_surface_face_carries_prescribed_flux():
    # a NEUTRAL column (no interior gradient) with a prescribed surface flux Q0:
    # the surface face carries exactly Q0, so the lowest centre = Q0/2 (f2c of
    # [Q0, 0]) and it decays to 0 aloft.
    theta = np.broadcast_to(np.full(NZ, 300.0), (NY, NX, NZ)).copy()
    nu_t = np.full((NY, NX, NZ), 2.0)
    Q0 = 0.06
    flux = sgs_vertical_scalar_flux_mean(theta, nu_t, DZ, pr_sgs=1.0, surface_flux=Q0)
    assert flux[0] == pytest.approx(0.5 * Q0)   # f2c([Q0, 0]) at the lowest centre
    assert np.allclose(flux[1:], 0.0, atol=1e-9)


def test_sgs_flux_prandtl_scaling():
    z = _z()
    theta = np.broadcast_to(300.0 + 0.01 * z, (NY, NX, NZ)).copy()
    nu_t = np.full((NY, NX, NZ), 4.0)
    f1 = sgs_vertical_scalar_flux_mean(theta, nu_t, DZ, pr_sgs=1.0)
    f2 = sgs_vertical_scalar_flux_mean(theta, nu_t, DZ, pr_sgs=2.0)
    # K_h halves ⇒ flux magnitude halves (at every level, incl. endpoints)
    assert np.allclose(f2, 0.5 * f1, atol=1e-9)


def test_sgs_flux_bad_prandtl_rejected():
    theta = np.zeros((NY, NX, NZ))
    with pytest.raises(ValueError):
        sgs_vertical_scalar_flux_mean(theta, np.ones((NY, NX, NZ)), DZ, pr_sgs=0.0)


def test_sgs_flux_matches_core_face_operators():
    # emit must reproduce the SAME vertical SGS flux the spectral core integrates.
    # Build the flux from the core's own c2f/ddz_c2f/f2c ops and compare.
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl

    rng = np.random.default_rng(7)
    z = _z()
    dz = float(z[1] - z[0])
    theta = (300.0 + 0.005 * z + rng.normal(0, 0.2, size=(NY, NX, NZ))).astype(np.float64)
    nu_t = np.abs(rng.normal(3.0, 0.5, size=(NY, NX, NZ)))
    pr, Q0 = 0.7, 0.06

    emit_flux = sgs_vertical_scalar_flux_mean(theta, nu_t, dz, pr_sgs=pr, surface_flux=Q0)

    # reference via the core operators: turbulent flux = -J, J face = [-Q0, K_f*ddz, 0]
    th = jnp.asarray(theta)
    kh = jnp.asarray(nu_t) / pr
    j_int = sl.c2f(kh) * sl.ddz_c2f(th, dz)
    zc = jnp.zeros_like(th[..., :1])
    j_face = jnp.concatenate([-Q0 + zc, j_int, zc], axis=-1)   # core's flux_full
    turb_center = sl.f2c(-j_face)                              # <w'θ'> = -J, f2c
    ref = np.asarray(turb_center).mean(axis=(0, 1))
    assert np.allclose(emit_flux, ref, atol=1e-10)


# --- artifact assembly --------------------------------------------------------
def test_build_reference_artifact_dry():
    nt = 3
    z = _z()
    art = build_reference_artifact(
        case_name="cbl_emit",
        sgs="lasd",
        z=z,
        times_s=np.linspace(0.0, 3600.0, nt),
        theta=np.full((nt, NZ), 300.0),
        u=np.zeros((nt, NZ)),
        v=np.zeros((nt, NZ)),
        wtheta_resolved=np.full((nt, NZ), 0.05),
        wtheta_sgs=np.full((nt, NZ), 0.01),
        subsidence_w=np.zeros(NZ),
        prescribe="fluxes",
        w_theta_s=np.full(nt, 0.06),
    )
    assert isinstance(art, LESReferenceArtifact)
    assert not art.is_moist
    assert art.nt == nt and art.nz == NZ


def test_build_reference_artifact_moist_roundtrips_validation():
    nt = 2
    z = _z()
    art = build_reference_artifact(
        case_name="bomex_emit",
        sgs="lasd",
        z=z,
        times_s=np.array([0.0, 3600.0]),
        theta=np.full((nt, NZ), 300.0),
        u=np.full((nt, NZ), -8.0),
        v=np.zeros((nt, NZ)),
        wtheta_resolved=np.full((nt, NZ), 0.02),
        wtheta_sgs=np.full((nt, NZ), 0.005),
        qt=np.full((nt, NZ), 0.016),
        wqt_resolved=np.full((nt, NZ), 1e-4),
        wqt_sgs=np.full((nt, NZ), 2e-5),
        u_geo=np.full(NZ, -8.75),
        prescribe="fluxes",
        w_theta_s=np.full(nt, 0.02),
        w_qv_s=np.full(nt, 5e-5),
    )
    assert art.is_moist


def test_build_reference_artifact_invalid_raises():
    # dry artifact carrying a moisture flux must be rejected by validate()
    nt = 2
    z = _z()
    with pytest.raises(Exception):
        build_reference_artifact(
            case_name="bad",
            sgs="lasd",
            z=z,
            times_s=np.array([0.0, 3600.0]),
            theta=np.full((nt, NZ), 300.0),
            u=np.zeros((nt, NZ)),
            v=np.zeros((nt, NZ)),
            wtheta_resolved=np.full((nt, NZ), 0.05),
            wtheta_sgs=None,
            wqt_resolved=np.full((nt, NZ), 1e-4),  # moisture on a dry artifact
            prescribe="fluxes",
            w_theta_s=np.full(nt, 0.06),
        )
