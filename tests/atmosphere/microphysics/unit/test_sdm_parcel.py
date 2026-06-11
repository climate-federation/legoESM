"""Adiabatic-parcel validation of the SDM condensation + coupling chain.

A closed parcel rising at constant updraft must: conserve total water
(q_v + q_l) to round-off; develop a supersaturation that peaks then relaxes as
droplets grow; grow its droplets and liquid water content; and cool along the
moist adiabat (warmer than the dry adiabat because of latent heating). This is
the classic warm-cloud activation parcel and exercises condensation.py +
coupling.py + box_model.py together.

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/microphysics/unit/test_sdm_parcel.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.microphysics.sdm import (
    ParcelState,
    SDMConfig,
    SuperDropletState,
    liquid_mixing_ratio,
    run_parcel,
    saturation_ratio,
)


def _initial_parcel(T0=283.15, p0=9.0e4, rh0=0.98, n_sd=200, r0=1.0e-6,
                    N_per_m3=5.0e7):
    """Saturated-ish parcel seeded with monodisperse cloud droplets."""
    q_sat0 = float(saturation_mixing_ratio(jnp.asarray(T0), jnp.asarray(p0)))
    q_v0 = rh0 * q_sat0
    rho_air = p0 / (constants.R_d * T0)
    xi_per_kg = N_per_m3 / rho_air / n_sd   # real droplets per kg air per super-droplet
    ones = jnp.ones((n_sd,))
    droplets = SuperDropletState(
        multiplicity=ones * xi_per_kg,
        radius=ones * r0,
        solute_mass=ones * 0.0,
        active=ones,
    )
    return ParcelState(droplets=droplets, T=jnp.asarray(T0), p=jnp.asarray(p0),
                       q_v=jnp.asarray(q_v0), z=jnp.asarray(0.0))


def test_parcel_conserves_total_water():
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    n_substeps_condensation=2)
    parcel0 = _initial_parcel()
    q_t0 = float(parcel0.q_v + liquid_mixing_ratio(parcel0.droplets))
    final, hist = run_parcel(parcel0, w=3.0, dt=0.05, n_steps=1000, cfg=cfg)
    q_t = np.asarray(hist["q_t"])
    assert np.max(np.abs(q_t - q_t0)) / q_t0 < 1e-8   # total water conserved


def test_parcel_supersaturation_peaks_then_relaxes():
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    n_substeps_condensation=2)
    parcel0 = _initial_parcel()
    final, hist = run_parcel(parcel0, w=3.0, dt=0.05, n_steps=1000, cfg=cfg)
    S = np.asarray(hist["S"])
    S_max = S.max()
    assert S_max > 1.0                     # supersaturation develops
    i_max = int(S.argmax())
    assert i_max < len(S) - 1              # peak is interior
    assert S[-1] < S_max                   # relaxes after the peak


def test_parcel_droplets_and_lwc_grow():
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    n_substeps_condensation=2)
    parcel0 = _initial_parcel()
    r0 = float(parcel0.droplets.radius[0])
    q_l0 = float(liquid_mixing_ratio(parcel0.droplets))
    final, hist = run_parcel(parcel0, w=3.0, dt=0.05, n_steps=1000, cfg=cfg)
    assert float(final.droplets.radius[0]) > r0          # droplets grew
    assert float(liquid_mixing_ratio(final.droplets)) > q_l0  # LWC increased


def test_parcel_follows_moist_adiabat():
    """Final T is below the start (ascent cools) but above the dry-adiabatic
    value (latent heating warms the parcel)."""
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    n_substeps_condensation=2)
    parcel0 = _initial_parcel()
    T0 = float(parcel0.T)
    final, hist = run_parcel(parcel0, w=3.0, dt=0.05, n_steps=1000, cfg=cfg)
    z = float(final.z)
    T_dry = T0 - constants.g / constants.c_pd * z   # dry-adiabatic descent of T
    assert float(final.T) < T0
    assert float(final.T) > T_dry


def test_parcel_jit():
    cfg = SDMConfig(include_curvature=False, include_solute=False)
    parcel0 = _initial_parcel()
    run = jax.jit(lambda pc: run_parcel(pc, 2.0, 0.05, 200, cfg)[0])
    final = run(parcel0)
    assert jnp.isfinite(final.T) and jnp.isfinite(final.q_v)
