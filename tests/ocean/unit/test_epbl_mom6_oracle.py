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
The four shortwave cases (#2178, ``mom6_epbl_scm_sw_0e9a561.npz``) store the
per-level gap vector instead, to the same tolerance.
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
from legoesm.ocean.physics.shortwave_penetration import (
    ShortwavePenetrationConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import (
    EPBLConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.scm import OceanColumnModel
from legoesm.ocean.scm_forcing import OceanSCMForcing
from legoesm.ocean.vertical import create_ocean_z_star

FIXTURE = (pathlib.Path(__file__).parent / "fixtures" / "mom6_epbl_scm"
           / "mom6_epbl_scm_0e9a561.npz")
FIXTURE_SW = FIXTURE.with_name("mom6_epbl_scm_sw_0e9a561.npz")

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


def _epbl_case_config(dt: float, smooth=None,
                      shortwave=None) -> OceanPhysicsConfig:
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
        shortwave_penetration=shortwave)


def _legoesm_T_8h(q: float, dt: float, T0, S0, smooth=None,
                  sw_down=None) -> np.ndarray:
    zc = create_ocean_z_star(n_levels=50, H_max=50.0, dz_surface=1.0,
                             dz_deep=1.0)
    f40 = 2.0 * constants.Omega * np.sin(np.deg2rad(40.0))
    forcing = OceanSCMForcing(
        f_c=f40, tau_x=lambda t: 0.2, tau_y=lambda t: 0.0,
        q_net=lambda t: q, e_minus_p=lambda t: 0.0,
        sw_down=None if sw_down is None else (lambda t: sw_down))
    # The MOM6 shortwave cases' optics: DOUBLE_EXP with Jerlov type I bands.
    shortwave = (None if sw_down is None else ShortwavePenetrationConfig(
        scheme="jerlov_2band", water_type="I"))
    scm = OceanColumnModel.create(
        nlev=50, dt=dt, T_profile=jnp.asarray(T0), S_profile=jnp.asarray(S0),
        physics_config=_epbl_case_config(dt, smooth, shortwave), z_coord=zc,
        latitude_deg=40.0,
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


# Shortwave cases (#2178): 200 W/m^2 of penetrating shortwave, sensible Q =
# -200 / 0 W/m^2.  The column maximum of the gap sits in the bottom layer (the
# seabed-deposit KNOWN DIFFERENCE, epbl.py docstring) and does not see ePBL's
# shortwave terms, so the stored gap is the per-level |T_legoESM - T_MOM6|
# vector at 8 h [degC], level 0 = surface (fp64 policy, exact switches).
# Measured in SLURM job 28601202 (JAX_ENABLE_X64=1).
MEASURED_GAP_SW = {
    ('q-200', 1200): [
        0.0013586146237898333, 0.0013586595132757395, 0.0013587208520284833,
        0.0013588124007952729, 0.0013589470278816407, 0.0013591471721703385,
        0.0013594574003370496, 0.0013599754166939704, 0.0013609551038022971,
        0.001363272409179217, 0.0011984476476314398, 0.0013352847972711857,
        0.0013438143120403367, 0.0013441505691531574, 0.0013441610620184008,
        0.0013441613461786517, 0.001344161353198814, 0.0013441613533622387,
        0.0013441613533622387, 0.0013441613533480279, 0.001344161353333817,
        0.0013441613533551333, 0.0013441613533515806, 0.0013441613533515806,
        0.0013441613533551333, 0.0013441613533409225, 0.001344161353333817,
        0.0013441613533515806, 0.001344161353358686, 0.0013441613533480279,
        0.0013441613533373697, 0.0013441613533480279, 0.0013441613533409225,
        0.0013441613533480279, 0.0013441613533444752, 0.0013441613533693442,
        0.0013441613533622387, 0.0013441613533480279, 0.0013441613533480279,
        0.0013441613533302643, 0.0013441613527138685, 0.0013441613357887405,
        0.001344160926567639, 0.001344151839752783, 0.0013439689691274737,
        0.0013406941219642476, 0.0012897922091390512, 0.0006270268132588797,
        0.006221659827804871, 0.05752291187546277,
    ],
    ('q-200', 300): [
        0.0013479940293557036, 0.0013480184055438826, 0.001348051252481497,
        0.0013480995380810157, 0.0013481695087804724, 0.001348271707851012,
        0.0013484268143422184, 0.0013486793503716399, 0.0013491422438747236,
        0.0013501896419469972, 0.0013031390059623504, 0.001341835287576032,
        0.001344082794741297, 0.0013441594055443318, 0.001344161313728165,
        0.0013441613523816898, 0.0013441613530567054, 0.0013441613530496,
        0.0013441613530531527, 0.0013441613530460472, 0.001344161353074469,
        0.0013441613530531527, 0.0013441613530531527, 0.001344161353060258,
        0.0013441613530424945, 0.0013441613530638108, 0.0013441613530531527,
        0.0013441613530673635, 0.0013441613530567054, 0.0013441613530567054,
        0.0013441613531028906, 0.0013441613530709162, 0.001344161353060258,
        0.0013441613530531527, 0.0013441613530531527, 0.0013441613530957852,
        0.0013441613531028906, 0.001344161353109996, 0.0013441613531135488,
        0.0013441613530815744, 0.0013441613529288077, 0.0013441613456901536,
        0.0013441611341136195, 0.001344155574539485, 0.001344026164963097,
        0.001341408931011756, 0.0012965934131123902, 0.0006701328893292668,
        0.006097439651822967, 0.05769781529144957,
    ],
    ('q0', 1200): [
        0.0012947447810205404, 0.0012948001242776286, 0.0012948734257776096,
        0.0012949962202100096, 0.0012951909659442151, 0.0012954864756729023,
        0.0012959266873906472, 0.0012965693158939473, 0.0012974434758206144,
        0.0012982371931293812, 0.001295931207330625, 0.0012597759027528355,
        0.0019423800039781725, 0.0013614807178932153, 0.0013445743372280106,
        0.001344170610625639, 0.001344161557906176, 0.0013441613578599743,
        0.001344161353458162, 0.0013441613533444752, 0.001344161353333817,
        0.0013441613533551333, 0.0013441613533515806, 0.0013441613533515806,
        0.0013441613533551333, 0.0013441613533409225, 0.001344161353333817,
        0.0013441613533515806, 0.001344161353358686, 0.0013441613533480279,
        0.0013441613533373697, 0.0013441613533480279, 0.0013441613533409225,
        0.0013441613533480279, 0.0013441613533444752, 0.0013441613533693442,
        0.0013441613533622387, 0.0013441613533480279, 0.0013441613533480279,
        0.0013441613533302643, 0.0013441613527138685, 0.0013441613357887405,
        0.001344160926567639, 0.001344151839752783, 0.0013439689691274737,
        0.0013406941219642476, 0.0012897922091390512, 0.0006270268132588797,
        0.006221659827804871, 0.05752291187546277,
    ],
    ('q0', 300): [
        0.0013367547011995384, 0.001336779830680257, 0.0013368076657087613,
        0.0013368591120688222, 0.0013369476436899674, 0.0013370879232681432,
        0.00133730274617605, 0.0013376238546882746, 0.0013380786124912447,
        0.0013385835443529004, 0.0013381797892257907, 0.0013280641085877676,
        0.001433303223688398, 0.001345859745637057, 0.0013441873884332267,
        0.0013441617564389219, 0.0013441613596931745, 0.0013441613531561813,
        0.0013441613530460472, 0.0013441613530460472, 0.001344161353074469,
        0.0013441613530531527, 0.0013441613530531527, 0.001344161353060258,
        0.0013441613530424945, 0.0013441613530638108, 0.0013441613530531527,
        0.0013441613530673635, 0.0013441613530567054, 0.0013441613530567054,
        0.0013441613531028906, 0.0013441613530709162, 0.001344161353060258,
        0.0013441613530531527, 0.0013441613530531527, 0.0013441613530957852,
        0.0013441613531028906, 0.001344161353109996, 0.0013441613531135488,
        0.0013441613530815744, 0.0013441613529288077, 0.0013441613456901536,
        0.0013441611341136195, 0.001344155574539485, 0.001344026164963097,
        0.001341408931011756, 0.0012965934131123902, 0.0006701328893292668,
        0.006097439651822967, 0.05769781529144957,
    ],
}
_SW_DOWN = 200.0


@pytest.mark.parametrize("case,dt", sorted(MEASURED_GAP_SW))
def test_epbl_scm_shortwave_reproduces_its_measured_gap_to_mom6(case, dt):
    key = f"sw200_{case}_dt{dt}"
    with np.load(FIXTURE_SW) as fx:
        T_mom6, T0, S0 = (fx[f"{key}_T_8h"], fx[f"{key}_T_init"],
                          fx[f"{key}_S_init"])
    T_lego = _legoesm_T_8h(_Q[case], float(dt), T0, S0, sw_down=_SW_DOWN)
    np.testing.assert_allclose(np.abs(T_lego - T_mom6),
                               MEASURED_GAP_SW[(case, dt)],
                               rtol=1e-6, atol=1e-12)
