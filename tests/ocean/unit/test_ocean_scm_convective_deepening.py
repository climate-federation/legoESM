"""Physics-fidelity test for the ocean SCM: free-convective mixed-layer
deepening against an analytic oracle (issue #335).

ORACLE — non-penetrative convective encroachment into a linear stratification
=============================================================================
A resting, linearly stratified column ``T(z) = T_s0 - Gamma * z`` (z = depth,
Gamma = dT/dz > 0) is cooled at the surface with a constant heat loss
``Q0`` [W/m^2].  With a heat-conserving convective-adjustment scheme the
surface-cooled water convects and homogenises a deepening mixed layer.  Three
scheme-independent consequences are tested:

1. **Energy conservation.** The column heat-content change equals the heat
   removed at the surface::

       rho_0 * c_sw * integral(T_init - T) dz  ==  Q0 * t        (exact)

2. **sqrt(t) encroachment law.** Removing ``Q0 * t`` from a *linear* profile
   homogenises a slab of depth ``h_slab(t) = sqrt(2 Q0 t / (rho_0 c_sw Gamma))``.
   The MLD therefore deepens like ``sqrt(t)`` — the diagnostic signature of
   convective deepening.  We test the **scaling** ``h(t2)/h(t1) ~ sqrt(t2/t1)``
   (robust to the MLD-threshold constant).

3. **Amplitude.** The threshold MLD must track the analytic slab depth to a
   bounded, time-independent shape factor (``MLD / h_slab`` within a fixed
   bracket).  Without this, a scheme that removes the right heat but entrains
   it into a layer that is a constant factor too shallow/deep would still
   pass (1) and (2); the amplitude guard closes that gap.

This is the lightweight, data-free idealized oracle of #335; the
``scripts/validate/validate_ocean_scm.py`` harness runs the same case across
several times and is the hook for plugging in LES / GOTM oracles later.

Run with::
    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/ocean/unit/test_ocean_scm_convective_deepening.py -v
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.scm import OceanColumnModel
from legoesm.ocean.scm_forcing import OceanSCMForcing
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.eos import rho_0, c_sw


@pytest.fixture(autouse=True, scope="module")
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# Idealized convective-deepening setup (kept small for test runtime).
_NLEV = 24
_H_MAX = 300.0
_DZ_SURF = 6.0
_DZ_DEEP = 16.0
_GAMMA = 0.01          # dT/dz [K/m] — stable stratification
_T_S0 = 20.0           # surface temperature [degC]
_Q0 = 250.0            # surface heat loss [W/m^2]
_DT = 3600.0           # time step [s]
_MLD_THRESHOLD = 0.1   # K from surface
# Diffusive (non-slab) adjustment makes the threshold MLD a fixed multiple of
# the analytic slab depth; measured ~1.45 and time-independent across 2-12 d.
_AMP_LO, _AMP_HI = 1.2, 1.8


def _config():
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="constant"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
    )


def _depths():
    probe = OceanColumnModel.create(
        nlev=_NLEV, dt=_DT, T_profile=jnp.full((_NLEV,), _T_S0), S_profile=35.0,
        physics_config=_config(), H_max=_H_MAX, dz_surface=_DZ_SURF,
        dz_deep=_DZ_DEEP,
    )
    dz = np.asarray(probe.z_coord.dz_ref)
    return dz, np.cumsum(dz) - 0.5 * dz


def _run_days(days, forcing):
    dz, zcen = _depths()
    T_init = jnp.asarray(_T_S0 - _GAMMA * zcen)
    model = OceanColumnModel.create(
        nlev=_NLEV, dt=_DT, T_profile=T_init, S_profile=35.0,
        physics_config=_config(), H_max=_H_MAX, dz_surface=_DZ_SURF,
        dz_deep=_DZ_DEEP, forcing=forcing, implicit_vertical_mixing=True,
    )
    nsteps = int(round(days * 86400.0 / _DT))
    final, _ = model.run(nsteps, save_every=nsteps)
    T = np.asarray(final.T.data).reshape(-1)
    return T, dz, zcen, np.asarray(T_init), nsteps * _DT


def _mld(T, zcen, threshold=_MLD_THRESHOLD):
    return float(zcen[np.abs(T - T[0]) < threshold].max())


def _h_slab(t):
    return math.sqrt(2.0 * _Q0 * t / (rho_0 * c_sw * _GAMMA))


@pytest.fixture(scope="module")
def cooled_runs():
    """Cooled-column runs at two times (shared across tests to avoid recompute)."""
    forcing = OceanSCMForcing(q_net=lambda t: -_Q0)
    t1_days, t2_days = 2.0, 8.0
    r1 = _run_days(t1_days, forcing)
    r2 = _run_days(t2_days, forcing)
    return {"t1_days": t1_days, "t2_days": t2_days, "r1": r1, "r2": r2}


def test_surface_cooling_conserves_column_heat(cooled_runs):
    """Heat removed at the surface == column heat-content deficit (exact)."""
    T, dz, zcen, T_init, t = cooled_runs["r2"]
    heat_removed = rho_0 * c_sw * float(np.sum(dz * (T_init - T)))
    assert heat_removed == pytest.approx(_Q0 * t, rel=5e-3), (
        f"column heat deficit {heat_removed:.4e} J/m^2 != Q0*t {_Q0 * t:.4e}"
    )


def test_mixed_layer_deepens_like_sqrt_t(cooled_runs):
    """MLD(t2)/MLD(t1) follows the sqrt(t2/t1) encroachment scaling."""
    _, _, zcen, _, _ = cooled_runs["r1"]
    h1 = _mld(cooled_runs["r1"][0], zcen)
    h2 = _mld(cooled_runs["r2"][0], zcen)
    assert h1 > _DZ_SURF and h2 > h1, (
        f"mixed layer did not deepen: h1={h1}, h2={h2}"
    )
    ratio = h2 / h1
    expected = math.sqrt(cooled_runs["t2_days"] / cooled_runs["t1_days"])  # 2.0
    assert ratio == pytest.approx(expected, rel=0.12), (
        f"deepening ratio {ratio:.3f} not ~ sqrt(t2/t1)={expected:.3f} "
        f"(h1={h1:.1f} m, h2={h2:.1f} m)"
    )


def test_mld_amplitude_tracks_slab_oracle(cooled_runs):
    """MLD / h_slab stays within a fixed, time-independent bracket.

    Guards against a scheme that conserves heat and scales as sqrt(t) but
    entrains into a layer a constant factor too shallow/deep.
    """
    _, _, zcen, _, _ = cooled_runs["r1"]
    amps = []
    for key in ("r1", "r2"):
        T, _, _, _, t = cooled_runs[key]
        amps.append(_mld(T, zcen) / _h_slab(t))
    for a in amps:
        assert _AMP_LO < a < _AMP_HI, (
            f"MLD/h_slab={a:.3f} outside [{_AMP_LO}, {_AMP_HI}] — entrainment "
            f"depth materially off the analytic slab oracle"
        )
    # Shape factor must be ~time-independent (encroachment self-similarity).
    assert abs(amps[1] - amps[0]) < 0.15, (
        f"MLD/h_slab not time-independent: {amps}"
    )


def test_no_cooling_no_deepening():
    """Control: with zero surface flux the resting column stays put."""
    T, dz, zcen, T_init, _ = _run_days(4.0, OceanSCMForcing())
    assert _mld(T, zcen) < 3.0 * _DZ_SURF
