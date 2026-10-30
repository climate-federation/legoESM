"""ePBL single column vs MOM6 0e9a561 (issue #2181; the PR #2122 case).

The fixture ``fixtures/mom6_epbl_scm/`` holds the MOM6 inputs, the MOM6
parameter record and the MOM6 temperature after 8 h for six cases
(Q = -200, 0, +200 W/m^2 x dt = 1200, 300 s); ``provenance.json`` says how it
was built.  legoESM runs the same column with the MOM6 step order
(``EPBLConfig.k_after_surface_flux=True``) and the exact (unsmoothed) ePBL:
every RULE 5 smoothing sharpness is pinned to ``math.inf``, so this file is
the exact-limit test of each smoothed switch (issue #2180; the smooth-default
gaps and gradients are in ``test_epbl_smooth_switches.py``).

Each case's measured max |T_legoESM - T_MOM6| at 8 h is stored below and must
be reproduced to 1e-6 relative (1e-12 degC floor): any change to the port's
answer, better or worse, turns this red and must re-measure the gaps.
The column runs under legoESM's fp64 precision policy (pinned per test, as the
MOM6 oracle is double precision); the default fp32 policy gives different gaps.
"""
from __future__ import annotations

import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    EPBLConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.scm import OceanColumnModel
from legoesm.ocean.scm_forcing import OceanSCMForcing
from legoesm.ocean.vertical import create_ocean_z_star

FIXTURE = (pathlib.Path(__file__).parent / "fixtures" / "mom6_epbl_scm"
           / "mom6_epbl_scm_0e9a561.npz")

# Measured max |T_legoESM - T_MOM6| over the column at 8 h [degC], fp64 policy
# (SLURM job 28522629; JAX_ENABLE_X64=1).  Re-measured when the fp64 policy
# pin was added: the earlier values were taken under the fp32 default policy.
MEASURED_GAP = {
    ("q-200", 1200): 0.0002513095684157918,
    ("q0", 1200): 2.5934809855243657e-13,
    ("q200", 1200): 0.0011014506588296058,
    ("q-200", 300): 2.586951265470816e-05,
    ("q0", 300): 1.0196288258157438e-12,
    ("q200", 300): 0.00013344868003528632,
}
_Q = {"q-200": -200.0, "q0": 0.0, "q200": 200.0}

#: Every RULE 5 smoothing field of EPBLConfig at its exact (MOM6) limit.
EXACT_SWITCHES = {f: math.inf for f in EPBLConfig._fields
                  if f.startswith("smooth_")}


def _epbl_case_config(dt: float, smooth=None) -> OceanPhysicsConfig:
    """The PR #2122 case options (docs/ocean/specs/epbl_energetic_pbl.md)."""
    epbl = EPBLConfig(
        dt_s=dt, k_after_surface_flux=True,
        mstar_scheme="reichl_h18", wt_scheme="reichl_h18", nstar=0.06,
        omega_frac=0.001, tke_decay=0.001, mstar_convect_coef=0.66,
        rh18_mstar_cn3=-6.0, wstar_ustar_coef=15.0, fixed_mstar=0.0,
        mixlen_exponent=1.0, translay_scale=0.01, vstar_scale_fac=0.5477,
        vstar_surf_fac=1.8258, use_mld_iteration=True, mld_bisection=True,
        mld_tol=0.1, eqdisc_shape=True, eqdisc_v0h=True,
        K_bg=1.0e-5, A_bg=1.0e-4,
        **(EXACT_SWITCHES if smooth is None else smooth))
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="epbl", epbl=epbl),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None)


def _legoesm_T_8h(q: float, dt: float, T0, S0, smooth=None) -> np.ndarray:
    zc = create_ocean_z_star(n_levels=50, H_max=50.0, dz_surface=1.0,
                             dz_deep=1.0)
    f40 = 2.0 * constants.Omega * np.sin(np.deg2rad(40.0))
    forcing = OceanSCMForcing(
        f_c=f40, tau_x=lambda t: 0.2, tau_y=lambda t: 0.0,
        q_net=lambda t: q, e_minus_p=lambda t: 0.0)
    scm = OceanColumnModel.create(
        nlev=50, dt=dt, T_profile=jnp.asarray(T0), S_profile=jnp.asarray(S0),
        physics_config=_epbl_case_config(dt, smooth), z_coord=zc, latitude_deg=40.0,
        forcing=forcing, implicit_vertical_mixing=True,
        K_v_background=0.0, A_v_background=0.0)
    final, _ = scm.run(nsteps=int(round(8 * 3600 / dt)), save_every=10**6)
    return np.asarray(final.T.data[0, 0, 0])


@pytest.fixture(autouse=True)
def _fp64():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


@pytest.mark.parametrize("case,dt", sorted(MEASURED_GAP))
def test_epbl_scm_reproduces_its_measured_gap_to_mom6(case, dt):
    key = f"{case}_dt{dt}"
    with np.load(FIXTURE) as fx:
        T_mom6, T0, S0 = (fx[f"{key}_T_8h"], fx[f"{key}_T_init"],
                          fx[f"{key}_S_init"])
    T_lego = _legoesm_T_8h(_Q[case], float(dt), T0, S0)
    gap = float(np.max(np.abs(T_lego - T_mom6)))
    np.testing.assert_allclose(gap, MEASURED_GAP[(case, dt)],
                               rtol=1e-6, atol=1e-12)
