"""TurbulenceOutput.Kh_iface — interface heat diffusivity export.

The five schemes that compute their heat diffusivity on interfaces (Smagorinsky,
Louis, YSU, Holtslag-Boville, MYNN 2.5) export those interior values on all
``nlev+1`` interfaces with zero at the model top and the surface (CAM6's non-CLUBB
``kvh`` convention).  The interior must be the very array the full-level ``Kh``
diagnostic is averaged from, i.e. the one the scheme's own diffusion solve used.
Schemes that compute diffusivity on full levels only (TKE, EDMF, CLUBB-lite) leave it
None; prognostic CLUBB is not exported yet either (its interface diffusivity is a separate stage).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402
from legoesm.atmosphere.physics._shared import half_to_full  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import (  # noqa: E402
    CLUBBLiteConfig, HoltslagBovilleConfig, LouisConfig, MYNN25Config, SmagorinskyConfig,
    TKEConfig, TurbulentEDMFConfig, YSUConfig,
)
from legoesm.atmosphere.physics.turbulence.clubb_lite import clubb_lite_turbulence  # noqa: E402
from legoesm.atmosphere.physics.turbulence.edmf import edmf_turbulence  # noqa: E402
from legoesm.atmosphere.physics.turbulence.holtslag_boville import (  # noqa: E402
    holtslag_boville_turbulence,
)
from legoesm.atmosphere.physics.turbulence.louis import louis_turbulence  # noqa: E402
from legoesm.atmosphere.physics.turbulence.mynn25 import mynn25_turbulence  # noqa: E402
from legoesm.atmosphere.physics.turbulence.smagorinsky import (  # noqa: E402
    smagorinsky_turbulence,
)
from legoesm.atmosphere.physics.turbulence.tke import tke_turbulence  # noqa: E402
from legoesm.atmosphere.physics.turbulence.ysu import ysu_turbulence  # noqa: E402

from tests.atmosphere.hydrostatic.unit.test_turbulence import _make_column_data  # noqa: E402

NCOL, NLEV = 4, 10


def _inputs():
    u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(NCOL, NLEV)
    # Fixture only: a slightly unstable dry-adiabatic column (theta = 300 K) so every
    # scheme, including the Richardson-limited Smagorinsky closure, mixes somewhere.
    # Heights stay those of the helper's column; only the export contract is tested here.
    T = 300.0 * (p_full / constants.p_ref) ** constants.kappa
    rho = p_full / (constants.R_d * T)
    q_v = 0.5 * saturation_mixing_ratio(T, p_full)
    T_sfc = T[:, -1] + 5.0
    q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
    return u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho


def _run(name):
    u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho = _inputs()
    common = (p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho)
    if name == "smagorinsky":
        return smagorinsky_turbulence(u, v, T, q_v, *common, dt=300.0, config=SmagorinskyConfig())
    if name == "louis":
        return louis_turbulence(u, v, T, q_v, *common, dt=300.0, config=LouisConfig())
    if name == "ysu":
        return ysu_turbulence(u, v, T, q_v, *common, dt=300.0, config=YSUConfig())
    if name == "holtslag_boville":
        return holtslag_boville_turbulence(u, v, T, q_v, *common, dt=300.0,
                                           config=HoltslagBovilleConfig())
    tke0 = jnp.full((NCOL, NLEV), 0.5)
    if name == "mynn25":
        return mynn25_turbulence(u, v, T, q_v, tke0, *common, dt=300.0, config=MYNN25Config())[0]
    if name == "tke":
        return tke_turbulence(u, v, T, q_v, tke0, *common, dt=300.0, config=TKEConfig())[0]
    if name == "edmf":
        return edmf_turbulence(u, v, T, q_v, tke0, *common, dt=300.0,
                               config=TurbulentEDMFConfig())[0]
    if name == "clubb_lite":
        return clubb_lite_turbulence(u, v, T, q_v, tke0, *common, dt=300.0,
                                     config=CLUBBLiteConfig())[0]
    raise ValueError(name)


@pytest.mark.parametrize("name", ["smagorinsky", "louis", "ysu", "holtslag_boville", "mynn25"])
def test_interface_native_schemes_export_kh_iface(name):
    out = _run(name)
    k = np.asarray(out.Kh_iface)
    assert k.shape == (NCOL, NLEV + 1)
    np.testing.assert_array_equal(k[:, 0], 0.0)    # model-top interface
    np.testing.assert_array_equal(k[:, -1], 0.0)   # surface interface
    assert np.all(np.isfinite(k)) and np.all(k >= 0.0)
    assert np.max(k[:, 1:-1]) > 0.0, "interior must carry the scheme's diffusivity"
    # Interior is exactly the interface array the full-level Kh is averaged from.
    np.testing.assert_array_equal(np.asarray(half_to_full(out.Kh_iface[:, 1:-1])),
                                  np.asarray(out.Kh))


@pytest.mark.parametrize("name", ["tke", "edmf", "clubb_lite"])
def test_full_level_schemes_leave_kh_iface_unset(name):
    assert _run(name).Kh_iface is None
