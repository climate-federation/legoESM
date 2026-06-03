"""Validation cases for the ocean single-column model.

Physical benchmarks that exercise the column physics end-to-end:

* **Convective adjustment** — a statically unstable column is mixed back
  toward neutral stratification while conserving heat.
* **Surface heating** — a prescribed surface heat flux warms the column
  with an exact energy budget.
* **Wind-driven Ekman layer** — Crank-Nicolson Coriolis keeps the column
  bounded over many inertial periods (forward Euler would blow up), the
  surface current veers to the right of the wind (northern hemisphere),
  and the inertial-period-averaged transport matches −τ/(ρf).

Run with:
    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/validation/test_scm_ocean.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.tier1  # research: single-column ocean (convection/Ekman)

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.scm import OceanColumnModel
from legoesm.ocean.scm_forcing import OceanSCMForcing
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# ---------------------------------------------------------------------------
# Convective adjustment
# ---------------------------------------------------------------------------


def test_convective_adjustment_mixes_and_conserves_heat():
    nlev = 20
    # Statically unstable (constant S): cold/dense surface over warm/light
    # deep water -> T increases downward.
    T0 = jnp.linspace(8.0, 16.0, nlev)
    cfg = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
    )
    scm = OceanColumnModel.create(
        physics_config=cfg, nlev=nlev, dt=600.0, T_profile=T0,
        H_max=200.0, dz_surface=5.0, dz_deep=15.0,
    )
    dz = np.asarray(scm.z_coord.dz_ref)

    def hc(s):
        return float(np.sum(np.asarray(s.T.data[0, 0, 0]) * dz))

    hc0 = hc(scm.state)
    scm.run(nsteps=600, save_every=10**9)
    Tf = np.asarray(scm.state.T.data[0, 0, 0])

    assert np.all(np.isfinite(Tf))
    # heat content conserved (convective diffusion redistributes only)
    assert abs(hc(scm.state) - hc0) / abs(hc0) < 1e-10
    # stratification strongly reduced toward neutral
    assert Tf.std() < 0.5 * float(np.std(np.asarray(T0)))
    spread0 = float(np.max(np.asarray(T0)) - np.min(np.asarray(T0)))
    assert (Tf.max() - Tf.min()) < 0.5 * spread0
    # mean temperature preserved
    Tmean = float(np.sum(np.asarray(T0) * dz) / np.sum(dz))
    assert float(np.sum(Tf * dz) / np.sum(dz)) == pytest.approx(Tmean, rel=1e-10)


def test_stable_column_not_mixed_by_convection():
    """A statically stable column (warm over cold) is left essentially
    untouched by the convection scheme."""
    nlev = 20
    T0 = jnp.linspace(16.0, 8.0, nlev)  # warm surface over cold deep = stable
    cfg = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
    )
    scm = OceanColumnModel.create(
        physics_config=cfg, nlev=nlev, dt=600.0, T_profile=T0,
        H_max=200.0, dz_surface=5.0, dz_deep=15.0,
        K_v_background=0.0, A_v_background=0.0,
    )
    scm.run(nsteps=200, save_every=10**9)
    Tf = np.asarray(scm.state.T.data[0, 0, 0])
    # A stable column triggers no convective mixing: only the scheme's small
    # background floor (EnhancedDiffusionConfig.K_bg = 1e-5 m^2/s) acts, so
    # the stratification is preserved — unlike the unstable case it is NOT
    # homogenized (which would collapse the standard deviation toward 0).
    assert Tf.std() > 0.95 * float(np.std(np.asarray(T0)))
    assert np.max(np.abs(Tf - np.asarray(T0))) < 0.1


# ---------------------------------------------------------------------------
# Surface heating
# ---------------------------------------------------------------------------


def test_surface_heating_warms_column_with_exact_budget():
    nlev = 30
    q = 200.0  # W/m^2 into ocean
    scm = OceanColumnModel.create(
        physics_config=OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="constant"),
            convection=OceanConvectionConfig(scheme="none"),
        ),
        nlev=nlev, dt=1800.0, T_profile=jnp.full((nlev,), 10.0),
        H_max=1000.0, forcing=OceanSCMForcing(q_net=lambda t: q),
    )
    dz = np.asarray(scm.z_coord.dz_ref)

    def hc(s):
        return (float(np.sum(np.asarray(s.T.data[0, 0, 0]) * dz))
                * constants.rho_ocean * constants.c_sw)

    hc0 = hc(scm.state)
    Tmean0 = float(np.mean(np.asarray(scm.state.T.data)))
    nsteps = 120
    scm.run(nsteps=nsteps, save_every=10**9)

    assert float(np.mean(np.asarray(scm.state.T.data))) > Tmean0  # warmed
    dHC = hc(scm.state) - hc0
    expected = q * nsteps * scm.dt
    assert abs(dHC - expected) / abs(expected) < 1e-6


# ---------------------------------------------------------------------------
# Wind-driven Ekman layer
# ---------------------------------------------------------------------------


def _run_ekman(nlev=40, f=1.0e-4, dt=1200.0, tau=0.1, H_max=800.0,
               A_v=1.0e-2, spinup_periods=4):
    scm = OceanColumnModel.create(
        physics_config=OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="constant"),
            convection=OceanConvectionConfig(scheme="none"),
        ),
        nlev=nlev, dt=dt, T_profile=jnp.full((nlev,), 15.0),
        H_max=H_max, latitude_deg=43.0,
        forcing=OceanSCMForcing(f_c=f, tau_x=lambda t: tau),
        A_v_background=A_v,
    )
    Tin = 2 * np.pi / f
    scm.run(nsteps=int(spinup_periods * Tin / dt), save_every=10**9)
    dz = np.asarray(scm.z_coord.dz_ref)
    nper = int(Tin / dt)
    Us, Vs, us, vs, max_speed = [], [], [], [], 0.0
    for _ in range(nper):
        scm.step()
        u = np.asarray(scm.state.u.data[0, 0, 0])
        v = np.asarray(scm.state.v.data[0, 0, 0])
        Us.append(np.sum(u * dz))
        Vs.append(np.sum(v * dz))
        us.append(u[0])
        vs.append(v[0])
        max_speed = max(max_speed, float(np.max(np.abs(u))),
                        float(np.max(np.abs(v))))
    return {
        "Mx": float(np.mean(Us)), "My": float(np.mean(Vs)),
        "u_surf": float(np.mean(us)), "v_surf": float(np.mean(vs)),
        "max_speed": max_speed, "tau": tau, "f": f,
    }


def test_ekman_is_stable_over_many_inertial_periods():
    """Crank-Nicolson Coriolis stays bounded; forward Euler would amplify
    inertial energy by ~exp((f·dt)²·N/2) and blow up."""
    r = _run_ekman(spinup_periods=8)
    assert np.isfinite(r["max_speed"])
    # bounded by an Ekman-scale velocity, not runaway growth
    assert r["max_speed"] < 1.0


def test_ekman_surface_current_veers_right_of_wind():
    """Northern-hemisphere wind in +x drives a surface current deflected to
    the right (positive u, negative v)."""
    r = _run_ekman()
    assert r["u_surf"] > 0.0
    assert r["v_surf"] < 0.0
    angle = np.degrees(np.arctan2(r["v_surf"], r["u_surf"]))
    assert -90.0 < angle < 0.0  # clockwise from +x = right of wind


def test_ekman_meridional_transport_matches_theory():
    """Inertial-period-averaged transport: ∫v dz ≈ −τ/(ρf), ∫u dz ≈ 0."""
    r = _run_ekman(spinup_periods=4)
    My_th = -r["tau"] / (constants.rho_ocean * r["f"])
    assert r["My"] == pytest.approx(My_th, rel=0.25)
    # zonal transport small relative to the meridional Ekman transport
    assert abs(r["Mx"]) < 0.3 * abs(My_th)
