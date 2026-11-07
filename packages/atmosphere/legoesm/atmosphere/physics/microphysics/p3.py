"""P3 (Predicted Particle Properties) microphysics -- faithful port of WRF P3.

Port of ``p3_main`` of WRF v4.8.0 ``phys/module_mp_p3.F`` (P3 v4.5.2, Morrison &
Milbrandt 2015; https://github.com/wrf-model/WRF tag v4.8.0, SHA
06d4240ae989cc3e50af412bb472df3d9048783c) with its lookup table
``run/p3_lookupTable_1.dat-v5.4_2momI``, in the configuration WRF runs for
``mp_physics = 50`` (``P3_1CATEGORY``, ``mp_p3_wrapper_wrf``): ONE free ice
category with 2-moment ice (prognostic total mass, number, rime mass and rime
volume), SPECIFIED cloud-droplet number (``log_predictNc=.false.``; ``nccnst`` =
``P3Config.Nc_0``), Khairoutdinov-Kogan (2000) warm rain (``iparam = 3``), Cooper
(1986) deposition / condensation-freezing nucleation, ``max_total_Ni`` = 2000 /L,
no sub-grid cloud fraction (``scpf_on=.false.``), calibration factors
``clbfact_dep = clbfact_sub = 1``.

``tests/atmosphere/microphysics/unit/test_p3_oracle.py`` compares this module with
the RUNNING oracle (``scripts/validate/oracle/p3/``) at max relative error
<= 1e-10 in fp64.

What is ported, in the oracle's order (line numbers are module_mp_p3.F):

1. entry clipping of sub-``qsmall`` / evaporating condensate (2316-2414);
2. per level: cloud and rain size distributions with their lambda limiters
   (``get_cloud_dsd2`` / ``get_rain_dsd2``), the total-Ni cap, the ice lookup
   table (``find_lookupTable_indices_1a/1b``, ``access_lookup_table[_coll]``,
   bulk rime density, ice number limiter), collection of cloud and rain by ice
   (shedding above freezing), ice self-collection, melting, wet growth and
   shedding, rime density (Cober & List), immersion freezing of cloud and rain,
   semi-analytic condensation / evaporation / deposition / sublimation with the
   rain ventilation table, Cooper nucleation, specified-Nc droplet activation,
   KK2000 autoconversion / accretion, rain self-collection and breakup, the
   saturation-adjustment limits, the cloud / rain / ice / vapour conservation
   limiters, the state update with rime densification on melting, and the
   end-of-level clipping (2449-3925);
3. cloud, rain and ice sedimentation with the oracle's adaptive (CFL) sub-steps
   (3972-4496), homogeneous freezing below 233.15 K (4528-4610) and the final
   consistency checks (4722-4895).

Not ported (not reachable for ``mp_physics = 50``): predicted Nc and aerosol
activation (``mp_physics = 51``), triple-moment ice (53), multi-category ice
(52: lookup table 2, ice-ice collection, Hallett-Mossop), SB2001 / Beheng /
Kogan warm rain (``iparam`` 1, 2, 4), the sub-grid cloud / precipitation
fraction (SCPF), predicted supersaturation, the first-time-step (``it <= 1``)
saturation adjustment, and the diagnostic outputs (reflectivity, effective
radii, visibility, precipitation types).  Predicted Nc (``mp_physics = 51``)
and the carried ``th_old`` / ``qv_old`` state are deferred until a committed
config selects ``p3`` (tier-2 decision C9).

Interface glue (the oracle-fixture harness's, which the oracle match
certifies): levels are top-down here (surface at ``[:, -1]``), bottom-up in the
oracle; ``N_c`` and ``N_r`` are per m^3 here and per kg in the oracle (converted
with the INPUT ``rho``; P3 itself uses its own dry density ``p/(R_d T)``, as WRF
P3 does), ``N_i`` per kg in both; potential temperature uses P3's ``rd/cp``; the
beginning-of-step state equals the input state (``th_old = th``,
``qv_old = qv``: this operator-split interface carries no previous-step state),
and the step is not the first (``it = 2``).  KNOWN DEVIATION in a coupled run:
WRF carries ``th_old`` / ``qv_old`` as state from the previous microphysics
call, so its supersaturation-forcing term (``aaa``) includes the dynamics
tendency since then; here that part is zero and only its Bergeron part
remains.  The oracle match certifies the port under the same zero forcing,
not this term.  The thermodynamic constants
(``cp``, ``g``, ``rd``, ``rv``, the latent heats) and the saturation curves are
legoESM's (``legoesm.constants``, ``legoesm.thermo`` specific humidity), fed to
the oracle by the fixture harness; ``e0`` (melting, wet growth) stays the
oracle's own ``polysvp1`` at the freezing point.

Slot reuse in HydrometeorState / MicrophysicsOutput::

    q_s -> q_rim  [kg/kg]       rime mass mixing ratio
    q_g -> B_rim  [m^3/kg_air]  rime volume mixing ratio

Differentiability: every oracle branch is a ``jnp.where`` on the oracle's own
condition (piecewise differentiable, exactly the oracle).  The three gates that
already had a smoothing field keep it and reduce to the oracle at their hard
limit: the nucleation temperature gate (``ice_sigmoid_sharpness``), the
nucleation ice-supersaturation gate (``cooper_supi_sharpness``) and the melting
temperature gate (``melt_sharpness``).  At exactly a gate's threshold a
sigmoid is 0.5 at every sharpness while WRF's strict comparison is 0 or 1, so
the hard-limit match holds off the thresholds (measure-zero inputs).  The oracle's data-dependent
sedimentation sub-step loop runs on the Morrison scheme's fixed-maximum
machinery (``MorrisonConfig`` default ``sed_cfl_substeps_max``, chunked
``fori_loop``, the strict path's ``equinox.error_if`` + NaN poison): a column
that needs more sub-steps than the maximum is an error, never a clamp.

References
----------
- Morrison, H. & Milbrandt, J. A. (2015). J. Atmos. Sci., 72, 287-311.
- Cooper, W. A. (1986). Meteor. Monogr., 21, 29-32.
- Khairoutdinov, M. & Kogan, Y. (2000). Mon. Wea. Rev., 128, 229-243.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gamma as _gamma
from jax.scipy.special import gammaln as _gammaln

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity as _qsat_liq
from legoesm.thermo import saturation_specific_humidity_ice as _qsat_ice
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    accretion_kk2000,
    autoconversion_kk2000,
    hard_saturation_blend,
)
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig, P3Config
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
)

__physics_contract__ = {
    "summary": (
        "P3 (Predicted Particle Properties; Morrison & Milbrandt 2015): faithful "
        "port of WRF v4.8.0 module_mp_p3.F p3_main (P3 v4.5.2) with lookup table "
        "v5.4_2momI for mp_physics=50 -- one free 2-moment ice category with "
        "prognostic rime mass and volume, specified droplet number, KK2000 warm "
        "rain, Cooper nucleation, adaptive sedimentation; oracle-tested to 1e-10."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "hydrometeors.q_c": "kg/kg",
        "hydrometeors.q_r": "kg/kg", "hydrometeors.q_i": "kg/kg",
        "hydrometeors.q_s": "kg/kg (P3 rime mass q_rim)",
        "hydrometeors.q_g": "m^3/kg_air (P3 rime volume B_rim)",
        "hydrometeors.N_c": "1/m^3 (not read: specified Nc)",
        "hydrometeors.N_r": "1/m^3", "hydrometeors.N_i": "1/kg",
        "p_full": "Pa", "rho": "kg/m^3 (N_c/N_r unit conversion only)",
        "dz": "m", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "dq_i_dt": "kg/kg/s",
        "dq_s_dt": "kg/kg/s (rime mass q_rim)",
        "dq_g_dt": "m^3/kg_air/s (rime volume B_rim)",
        "dN_c_dt": "1/(m^3 s)", "dN_r_dt": "1/(m^3 s)", "dN_i_dt": "1/(kg s)",
        "precipitation": "kg/m^2/s", "precip_frozen": "kg/m^2/s",
    },
    "sign_convention": (
        "z up; surface at [:, -1].  Tendencies are (state after the P3 step - "
        "input state)/dt.  Latent heating uses L_v (vapour-liquid), L_v+L_f "
        "(vapour-ice), L_f (liquid-ice), each divided by c_pd, as WRF P3 with "
        "legoESM's latent heats.  Precipitation >= 0 into the surface removes "
        "water from the column."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Morrison & Milbrandt (2015), J. Atmos. Sci. 72, 287-311; WRF v4.8.0 module_mp_p3.F",
    "idealized_test": (
        "tests/atmosphere/microphysics/unit/test_p3_oracle.py -- whole-call "
        "match to the running WRF P3 oracle (fixture p3.npz) on six columns "
        "and 226 nucleation points at dt 20 s and 300 s."
    ),
}

# --- P3 mathematical constants (p3_init, module_mp_p3.F:179-183) ---
# P3 truncates pi; the oracle match needs P3's own value (math.pi leaves
# 1.1e-9 relative in mi0, cons3 and every PSD moment).
_P3_PI = 3.14159265
_THRD = 1.0 / 3.0
_SXTH = 1.0 / 6.0
_PIOV3 = _P3_PI * _THRD
_PIOV6 = _P3_PI * _SXTH

# --- P3 physical parameters (p3_init, module_mp_p3.F:186-284) ---
_MAX_TOTAL_NI = 2000.0e3          # max_total_Ni [1/m^3]
_EP_2 = 0.622                     # const-ok: P3 qsat0 prefactor (melting/wet growth), not R_d/R_v
_RHOSUR_P = 100000.0              # rhosur = 100000/(rd*T_freeze) numerator [Pa]
_RHOSUI_P = 60000.0               # rhosui = 60000/(rd*253.15) numerator [Pa]
_RHOSUI_T = 253.15                # rhosui temperature [K]
_F1R = 0.78                       # rain ventilation coefficients
_F2R = 0.32
_D_R_MAX = 0.002                  # inv_Drmax = 1/0.002: max rain mean diameter [m]
_QSMALL = 1.0e-14                 # minimum prognostic mass [kg/kg]
_NSMALL = 1.0e-16                 # minimum prognostic number [1/kg]
_BIMM = 2.0                       # Barklie & Gokhale (1959) immersion freezing
_AIMM = 0.65
_RHO_ICE_NUC = 900.0              # mi0 density [kg/m^3]
_ICE_NUC_RADIUS = 1.0e-6          # mi0 radius [m]
_MI0 = 4.0 * _PIOV3 * _RHO_ICE_NUC * _ICE_NUC_RADIUS ** 3
_ECI = 0.5                        # cloud-ice collection efficiency
_ERI = 1.0                        # rain-ice collection efficiency
_BCN = 2.0                        # Stokes fall-speed exponent
_NMLTRATIO = 1.0                  # rain number from melting / ice number lost
_E0 = 6.11239921 * 100.0          # e0 = polysvp1 at the freezing point [Pa] (its liquid a0)
_MU_R = 0.0                       # mu_r_constant
_DSD_DROP_SIZE = 25.0e-6          # cons3 drop radius [m]
_ACT_DROP_SIZE = 1.0e-6           # cons7 drop radius [m]
_RHO_RIME_DEFAULT = 400.0         # rhorime_c without Cober-List riming [kg/m^3] (p3_main:2290)
_CLBFACT_DEP = 1.0                # mp_p3_wrapper_wrf clbfact_dep / clbfact_sub
_CLBFACT_SUB = 1.0

# --- P3 rate-law coefficients (p3_main) ---
_MU_AIR_A = 1.496e-6              # air viscosity mu = A T^1.5/(T+120)
_MU_AIR_T = 120.0
_DV_A = 8.794e-5                  # vapour diffusivity dv = A T^1.81/p
_DV_EXP = 1.81
_KAP_A = 1.414e3                  # thermal conductivity kap = A mu
_EII_T_LO = 253.15                # aggregation efficiency ramp (2548-2554)
_EII_LO = 0.001
_EII_HI = 0.3
_EII_DT = 20.0
_EIIF_FR_LO = 0.6                 # Eii_fact rime-fraction ramp (2665-2677)
_EIIF_FR_HI = 0.9
_EIIF_FR_WIDTH = 0.3
_SHED_DROP_INV_MASS = 1.923e6     # 1/(mass of a 1 mm drop) [1/kg]
_WETGROWTH_QMIN = 1.0e-6          # qc+qr threshold for wet growth
_SHED_MIN = 1.0e-10               # shedding thresholds
_RIME_ITC_MAX = -0.001            # Cober-List iTc cap
_RIME_RI_A = 0.5e6                # Ri = -(0.5e6 D_c) V_impact / Tc
_RIME_RI_MIN = 1.0
_RIME_RI_MAX = 12.0
_RIME_RI_SPLIT = 8.0
_RIME_C0 = 0.051                  # rho_rime = (0.051+0.114 Ri-0.0055 Ri^2) 1000
_RIME_C1 = 0.114
_RIME_C2 = 0.0055
_RIME_SCALE = 1000.0
_RIME_HI0 = 611.0                 # rho_rime = 611+72.25 (Ri-8)
_RIME_HI1 = 72.25
_T_IMMERSION = 269.15             # immersion freezing at or below -4 C
_T_HOMOG = 233.15                 # homogeneous freezing below -40 C (4540, 4578)
_EVAP_QMIN = 1.0e-12              # instant evaporation/sublimation
_EVAP_SUPMIN = -0.001
_XX_MIN = 1.0e-20
_NUC_MIN = 1.0e-20                # N_nuc threshold
_ACT_SUPMIN = 1.0e-6              # specified-Nc activation
_AUTO_QCMIN = 1.0e-8              # autoconversion qc threshold
_RAIN_D_BREAKUP = 280.0e-6        # breakup threshold
_BREAKUP_EXP = 2300.0
_KR_RAIN_SC = 5.78                # rain self-collection, iparam 3
_CLIP_DRY_QMAX = 1.0e-8           # entry clipping
_CLIP_DRY_SUP = -0.1
_NUC_POSSIBLE_SUP = -0.05         # nucleation-possible gate
_SAT_TINY = 1.0e-20               # conservation-limiter floors
_SED_DT_MIN = 1.0e-4              # sedimentation sub-step loop
_NI_MIN_SUM = 1.0e-20             # impose_max_total_Ni
_RHO_RIME_BMIN = 1.0e-15          # calc_bulkRhoRime birim threshold
# cloud DSD (get_cloud_dsd2)
_MUC_A = 0.0005714
_MUC_B = 0.2714
_MUC_MIN = 2.0
_MUC_MAX = 15.0
_LAMC_MIN = 2.5e4
_LAMC_MAX = 1.0e6
# rain DSD (get_rain_dsd2)
_LAMR_MAX = 1.0e5
_LAMR_MIN = 1.0 / _D_R_MAX
# ice table indices (find_lookupTable_indices_1a/1b)
_ISIZE = 50
_RIMSIZE = 4
_DENSIZE = 5
_RCOLLSIZE = 30
_QI_IDX_OFF = 18.0
_QI_IDX_SCALE = 3.444606
_QI_IDX_SHIFT = 10.0
_RHOP_SPLIT = 650.0
_RHOP_LO0 = 50.0
_RHOP_LO_SCALE = 0.005
_RHOP_HI_SCALE = 0.004
_RHOP_HI_IDX = 4.0
_QR_IDX_OFF = 5.0
_QR_IDX_SCALE = 10.70415
# rain table indices (find_lookupTable_indices_3)
_RTAB_SPLIT = 195.0e-6
_RTAB_LO_OFF = 5.0
_RTAB_INV_LO = 0.1
_RTAB_SPLIT_UM = 195.0
_RTAB_HI_BASE = 20.0
_RTAB_LO_MAX = 20.0
_RTAB_HI_MAX = 300.0
_RTAB_JJ_MAX = 10.0
# rain fall-speed table generation (p3_init 640-686)
_RT_N = 300
_RT_NBIN = 10000
_RT_DD = 2.0                      # bin width [um]
_RT_RHO_DROP = 997.0              # [kg/m^3]
_RT_D1 = 134.43                   # fall-speed regime limits [um]
_RT_D2 = 1511.64
_RT_D3 = 3477.84
_RT_A1 = 4.5795e3
_RT_A2 = 4.962e1
_RT_A3 = 1.732e1
_RT_VMAX = 9.17
_RT_DM_SMALL = (10.0, 5.0)        # dm = (jj*10-5) um for jj <= 20
_RT_DM_LARGE = (30.0, 195.0)      # dm = ((jj-20)*30+195) um
_RT_FLOOR = 1.0e-30
_RT_N_SMALL = 20                  # rain-table bins: 20 small (10 um), then 280 large
_RT_N_MAX = 299                   # last interpolation index (300 bins)
_UM = 1.0e-6                      # micrometre [m]
_M_TO_UM = 1.0e6
_KG_TO_G = 1000.0
_RTAB_HI_SCALE = _THRD * 0.1       # rain-table index step above 195 um
_ITAB_NCOL, _COLL_NCOL = 17, 5    # columns per itab / itabcoll row in the v5.4_2momI file
_ITAB_NVAL = 14                   # tabsize
_P0 = 1.0e5                       # theta reference pressure [Pa]
_INV_P0 = 1.0e-5
_RHOFAC_EXP = 0.54                # air-density fall-speed correction exponent
_STOKES = 18.0                    # Stokes drag: acn = g*rhow/(18*mu)
_HET_GAMMA_A = 7.0                # immersion freezing gamma(7+mu_c)
_CM3_PER_M3 = 1.0e-6

# P3 thermodynamic constants (legoESM values, as fed to the oracle).
_CP = constants.c_pd
_INV_CP = 1.0 / _CP
_RD = constants.R_d
_RV = constants.R_v
_G = constants.g
_XXLV = constants.L_v
_XXLS = _XXLV + constants.L_f
_XLF = _XXLS - _XXLV
_RHOW = constants.rho_water
_INV_RHOW = 1.0 / _RHOW
_CPW = constants.c_pw
_RHO_ICE_BULK = constants.rho_ice  # rime densification target on melting (917 in P3)
_T0 = constants.T_freeze
_RHOSUR = _RHOSUR_P / (_RD * _T0)
_RHOSUI = _RHOSUI_P / (_RD * _RHOSUI_T)
_CONS1 = _PIOV6 * _RHOW
_CONS2 = 4.0 * _PIOV3 * _RHOW
_CONS3 = 1.0 / (_CONS2 * _DSD_DROP_SIZE ** 3)
_CONS5 = _PIOV6 * _BIMM
_CONS6 = _PIOV6 ** 2 * _RHOW * _BIMM
_CONS7 = 4.0 * _PIOV3 * _RHOW * _ACT_DROP_SIZE ** 3
# gamma at the constant rain shape mu_r = 0
_G_MUR1 = math.gamma(_MU_R + 1.0)
_G_MUR2 = math.gamma(_MU_R + 2.0)
_G_MUR4 = math.gamma(_MU_R + 4.0)
_G_MUR7 = math.gamma(7.0 + _MU_R)

# Sedimentation sub-step loop: the Morrison scheme's fixed maximum (its
# ``sed_cfl_substeps_max`` default) and chunk length (``output._SED_CHUNK``;
# cost only, the values are identical for any chunk length).
_SED_SUBSTEPS_MAX = MorrisonConfig._field_defaults["sed_cfl_substeps_max"]
_SED_CHUNK = 8

P3_LOOKUP_TABLE_VERSION = "5.4_2momI"
P3_LOOKUP_TABLE_NAME = "p3_lookupTable_1.dat-v5.4_2momI"
# Not vendored: fetched on first use and cached (the repo's P3 table policy).
# The URL is the oracle's own file, WRF v4.8.0 run/ (sha256 below, also pinned
# in the oracle fixture's meta).
P3_LOOKUP_TABLE_URL = ("https://raw.githubusercontent.com/wrf-model/WRF/v4.8.0/run/"
                       + P3_LOOKUP_TABLE_NAME)
P3_LOOKUP_TABLE_SHA256 = "be1ab6fb03481e376e47c6c79d808af5d8ab069f2b242931e9c54801bad4ae84"
_FETCH_TIMEOUT_S = 60.0


def p3_lookup_table_path() -> Path:
    """Local path of the P3 lookup table, downloading it on first use.

    Cache: ``$LEGOESM_CACHE_DIR/p3/`` or ``~/.cache/legoesm/p3/``.  The file is
    sha256-checked every call; a mismatch or a failed download raises.
    """
    import hashlib
    import os
    import shutil
    import tempfile
    import urllib.request

    root = os.environ.get("LEGOESM_CACHE_DIR") or str(Path.home() / ".cache" / "legoesm")
    path = Path(root).expanduser() / "p3" / P3_LOOKUP_TABLE_NAME

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    if path.exists():
        if sha(path) != P3_LOOKUP_TABLE_SHA256:
            raise RuntimeError(f"P3 lookup table {path} has the wrong sha256; delete "
                               f"it to re-download from {P3_LOOKUP_TABLE_URL}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # unique temp name: ranks on different nodes sharing the cache may share a PID
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as f:
        part = Path(f.name)
    try:
        with urllib.request.urlopen(P3_LOOKUP_TABLE_URL, timeout=_FETCH_TIMEOUT_S) as r, \
                open(part, "wb") as f:
            shutil.copyfileobj(r, f)
        if sha(part) != P3_LOOKUP_TABLE_SHA256:
            raise RuntimeError(f"downloaded file has sha256 {sha(part)}, "
                               f"expected {P3_LOOKUP_TABLE_SHA256}")
        part.replace(path)
    except Exception as exc:
        # another rank sharing the cache may have installed the table meanwhile.
        # ponytail: one re-check, no lock; a rank failing while a peer is still
        # downloading raises.  Multi-rank jobs should start with a warm cache.
        if path.exists() and sha(path) == P3_LOOKUP_TABLE_SHA256:
            return path
        raise RuntimeError(
            f"P3 lookup table not cached at {path} and the download from "
            f"{P3_LOOKUP_TABLE_URL} failed ({exc!r}). Download it on a machine with "
            f"network access and copy it to {path} (sha256 {P3_LOOKUP_TABLE_SHA256}), "
            f"or set LEGOESM_CACHE_DIR to a cache that holds p3/{P3_LOOKUP_TABLE_NAME}."
        ) from exc
    finally:
        part.unlink(missing_ok=True)
    return path


class P3Tables(NamedTuple):
    """P3 lookup tables (numpy, fp64).

    itab     : (densize=5, rimsize=4, isize=50, 14)  ice table (p3_init 361-368)
    itabcoll : (5, 4, 50, rcollsize=30, 2)  ice-rain collection table, stored
               as log10 in the file and read as is (p3_init 370-376)
    vn, vm, revap : (300,)  rain number- and mass-weighted fall speed and
               ventilation tables at mu_r = 0 (p3_init 640-686; the oracle's
               ten mu_r columns are identical because mu_r_constant = 0)
    """
    itab: np.ndarray
    itabcoll: np.ndarray
    vn: np.ndarray
    vm: np.ndarray
    revap: np.ndarray


def read_p3_lookup_table_1(path=None):
    """Read ``p3_lookupTable_1.dat-v5.4_2momI`` as p3_init does (module_mp_p3.F:343-376).

    Raises ``ValueError`` on a version other than 5.4_2momI (p3_init stops
    there for model='WRF') or a malformed size.  Returns ``(itab, itabcoll)``.
    ``path=None`` reads the cached table (:func:`p3_lookup_table_path`).
    """
    if path is None:
        path = p3_lookup_table_path()
    with open(path) as f:
        head = f.readline().split()
        if len(head) < 2 or head[1] != P3_LOOKUP_TABLE_VERSION:
            raise ValueError(
                f"P3 lookup table {path} has version {head[1:]!r}; P3 v4.5.2 "
                f"needs v{P3_LOOKUP_TABLE_VERSION}")
        vals = np.array(f.read().split(), dtype=np.float64)
    blk = _ISIZE * _ITAB_NCOL + _ISIZE * _RCOLLSIZE * _COLL_NCOL
    if vals.size != _DENSIZE * _RIMSIZE * blk:
        raise ValueError(f"P3 lookup table {path}: {vals.size} values, expected "
                         f"{_DENSIZE * _RIMSIZE * blk}")
    vals = vals.reshape(_DENSIZE, _RIMSIZE, blk)
    rows = vals[:, :, :_ISIZE * _ITAB_NCOL].reshape(_DENSIZE, _RIMSIZE, _ISIZE, _ITAB_NCOL)
    itab = rows[..., 3:3 + _ITAB_NVAL].copy()
    coll = vals[:, :, _ISIZE * _ITAB_NCOL:].reshape(_DENSIZE, _RIMSIZE, _ISIZE, _RCOLLSIZE,
                                                     _COLL_NCOL)
    itabcoll = coll[..., 3:5].copy()
    return itab, itabcoll


def _rain_tables():
    """vn/vm/revap tables at mu_r = mu_r_constant = 0 (p3_init 640-686),
    sequential sums like the oracle's kk loop."""
    mu_r = _MU_R
    jj = np.arange(1, _RT_N + 1, dtype=np.float64)
    dm = np.where(jj <= _RT_N_SMALL, (jj * _RT_DM_SMALL[0] - _RT_DM_SMALL[1]) * _UM,
                  ((jj - _RT_N_SMALL) * _RT_DM_LARGE[0] + _RT_DM_LARGE[1]) * _UM)
    lamr = (mu_r + 1) / dm
    dd = _RT_DD
    kk = np.arange(1, _RT_NBIN + 1, dtype=np.float64)
    dia = (kk * dd - dd * 0.5) * _UM
    amg = _PIOV6 * _RT_RHO_DROP * dia ** 3
    amg = amg * _KG_TO_G
    d_um = dia * _M_TO_UM
    vt = np.where(d_um <= _RT_D1, _RT_A1 * amg ** (2.0 * _THRD),
                  np.where(d_um < _RT_D2, _RT_A2 * amg ** _THRD,
                           np.where(d_um < _RT_D3, _RT_A3 * amg ** _SXTH, _RT_VMAX)))
    e = np.exp(-lamr[:, None] * dia[None, :])
    p0 = 10.0 ** (mu_r * np.log10(dia) + 4.0 * mu_r)
    p3 = 10.0 ** ((mu_r + 3.0) * np.log10(dia) + 4.0 * mu_r)
    p1 = 10.0 ** ((mu_r + 1.0) * np.log10(dia) + 3.0 * mu_r)

    def seq(a):          # sequential sum over the PSD bins, as the kk loop
        return np.cumsum(a, axis=1)[:, -1]

    dum1 = seq(vt * p0 * e * dd * _UM)
    dum2 = seq(p0 * e * dd * _UM)
    dum3 = seq(vt * p3 * e * dd * _UM)
    dum4 = seq(p3 * e * dd * _UM)
    dum5 = seq((vt * dia) ** 0.5 * p1 * e * dd * _UM)
    dum2 = np.maximum(dum2, _RT_FLOOR)
    dum4 = np.maximum(dum4, _RT_FLOOR)
    dum5 = np.maximum(dum5, _RT_FLOOR)
    vn = dum1 / dum2
    vm = dum3 / dum4
    revap = 10.0 ** (np.log10(dum5) + (mu_r + 1.0) * np.log10(lamr) - (3.0 * mu_r))
    return vn, vm, revap


@functools.lru_cache(maxsize=None)
def p3_lookup_tables(path: str | None = None) -> P3Tables:
    """All P3 tables (read and generated once per process)."""
    itab, itabcoll = read_p3_lookup_table_1(path)
    vn, vm, revap = _rain_tables()
    return P3Tables(itab=itab, itabcoll=itabcoll, vn=vn, vm=vm, revap=revap)


# ---------------------------------------------------------------- helpers
def _floor(x, lo):
    """``max(x, lo)`` for a positive floor ``lo`` that the oracle divides by.

    Same values as ``jnp.maximum``; written as a select so the floored branch
    gets no gradient (``jnp.maximum`` multiplies the cotangent by a 0/1 mask,
    and ``0 * (x / lo**2)`` is NaN once ``lo**2`` flushes to zero in fp32).
    """
    return jnp.where(x > lo, x, lo)


@jax.custom_jvp
def _div(x, y):
    """``x / y`` (same value) whose derivative is ``(dx - (x/y) dy) / y``.

    JAX's own rule forms ``x * y**-2``: for the oracle's tiny limiter
    denominators ``y**2`` underflows (always so under fp32 flush-to-zero) and a
    zero cotangent becomes ``0 * inf = NaN``.  Same derivative, never squares y.
    """
    return x / y


@_div.defjvp
def _div_jvp(primals, tangents):
    x, y = primals
    dx, dy = tangents
    q = x / y
    return q, (dx - q * dy) / y


def _ftrunc(x):
    """Fortran int(): truncation toward zero."""
    return jnp.trunc(x).astype(jnp.int32)


def _ice_indices(qitot, nitot, qirim, rhop):
    """find_lookupTable_indices_1a (module_mp_p3.F:6316-6370)."""
    dum1 = (jnp.log10(qitot / nitot) + _QI_IDX_OFF) * _QI_IDX_SCALE - _QI_IDX_SHIFT
    dumi = _ftrunc(dum1)
    dum1 = jnp.maximum(jnp.minimum(dum1, float(_ISIZE)), 1.0)
    dumi = jnp.minimum(jnp.maximum(1, dumi), _ISIZE - 1)
    dum4 = (qirim / qitot) * 3.0 + 1.0
    dumii = _ftrunc(dum4)
    dum4 = jnp.maximum(jnp.minimum(dum4, float(_RIMSIZE)), 1.0)
    dumii = jnp.minimum(jnp.maximum(1, dumii), _RIMSIZE - 1)
    dum5 = jnp.where(rhop <= _RHOP_SPLIT, (rhop - _RHOP_LO0) * _RHOP_LO_SCALE + 1.0,
                     (rhop - _RHOP_SPLIT) * _RHOP_HI_SCALE + _RHOP_HI_IDX)
    dumjj = _ftrunc(dum5)
    dum5 = jnp.maximum(jnp.minimum(dum5, float(_DENSIZE)), 1.0)
    dumjj = jnp.minimum(jnp.maximum(1, dumjj), _DENSIZE - 1)
    return dumjj, dumii, dumi, dum1, dum4, dum5


def _rain_coll_index(qr, nr):
    """find_lookupTable_indices_1b (6374-6410)."""
    ok = (qr >= _QSMALL) & (nr > 0.0)
    qs = jnp.where(ok, qr, 1.0)
    ns = jnp.where(ok, nr, 1.0)
    dumlr = (qs / (_P3_PI * _RHOW * ns)) ** _THRD
    dum3 = (jnp.log10(1.0 * dumlr) + _QR_IDX_OFF) * _QR_IDX_SCALE
    dumj = _ftrunc(dum3)
    dum3 = jnp.maximum(jnp.minimum(dum3, float(_RCOLLSIZE)), 1.0)
    dumj = jnp.minimum(jnp.maximum(1, dumj), _RCOLLSIZE - 1)
    return jnp.where(ok, dumj, 1), jnp.where(ok, dum3, 1.0)


def _access(tab, idx):
    """access_lookup_table (5205-5243); tab (5,4,50) for one table index."""
    jj, ii, i, d1, d4, d5 = idx

    def a(J, I2, I3):
        return tab[J - 1, I2 - 1, I3 - 1]

    w1 = d1 - i.astype(d1.dtype)
    w4 = d4 - ii.astype(d4.dtype)
    iproc1 = a(jj, ii, i) + w1 * (a(jj, ii, i + 1) - a(jj, ii, i))
    gproc1 = a(jj, ii + 1, i) + w1 * (a(jj, ii + 1, i + 1) - a(jj, ii + 1, i))
    tmp1 = iproc1 + w4 * (gproc1 - iproc1)
    iproc1 = a(jj + 1, ii, i) + w1 * (a(jj + 1, ii, i + 1) - a(jj + 1, ii, i))
    gproc1 = a(jj + 1, ii + 1, i) + w1 * (a(jj + 1, ii + 1, i + 1) - a(jj + 1, ii + 1, i))
    tmp2 = iproc1 + w4 * (gproc1 - iproc1)
    return tmp1 + (d5 - jj.astype(d5.dtype)) * (tmp2 - tmp1)


def _access_coll(tab, idx, dumj, dum3):
    """access_lookup_table_coll (5246-5314); tab (5,4,50,30) for one index."""
    jj, ii, i, d1, d4, d5 = idx

    def a(J, I2, I3, J4):
        return tab[J - 1, I2 - 1, I3 - 1, J4 - 1]

    w1 = d1 - i.astype(d1.dtype)
    w3 = dum3 - dumj.astype(dum3.dtype)
    w4 = d4 - ii.astype(d4.dtype)

    def at(J, I2):
        dproc1 = a(J, I2, i, dumj) + w1 * (a(J, I2, i + 1, dumj) - a(J, I2, i, dumj))
        dproc2 = a(J, I2, i, dumj + 1) + w1 * (a(J, I2, i + 1, dumj + 1) - a(J, I2, i, dumj + 1))
        return dproc1 + w3 * (dproc2 - dproc1)

    iproc1 = at(jj, ii)
    gproc1 = at(jj, ii + 1)
    tmp1 = iproc1 + w4 * (gproc1 - iproc1)
    iproc1 = at(jj + 1, ii)
    gproc1 = at(jj + 1, ii + 1)
    tmp2 = iproc1 + w4 * (gproc1 - iproc1)
    return tmp1 + (d5 - jj.astype(d5.dtype)) * (tmp2 - tmp1)


def _rain_table_interp(tab, mu_r, lamr):
    """find_lookupTable_indices_3 (6589-6632) + the caller's interpolation;
    the mu_r axis is fixed (mu_r = 0, both columns equal, so the mu_r
    interpolation returns the first column exactly)."""
    dum1 = (mu_r + 1.0) / lamr
    lo = dum1 <= _RTAB_SPLIT
    r_lo = jnp.minimum(jnp.maximum((dum1 * _M_TO_UM + _RTAB_LO_OFF) * _RTAB_INV_LO, 1.0),
                       _RTAB_LO_MAX)
    r_hi = jnp.minimum(jnp.maximum((dum1 * _M_TO_UM - _RTAB_SPLIT_UM) * _RTAB_HI_SCALE
                                   + _RTAB_HI_BASE, _RTAB_HI_BASE), _RTAB_HI_MAX)
    rdumii = jnp.where(lo, r_lo, r_hi)
    dumii = jnp.where(lo, jnp.minimum(jnp.maximum(_ftrunc(r_lo), 1), _RT_N_SMALL),
                      jnp.minimum(jnp.maximum(_ftrunc(r_hi), _RT_N_SMALL), _RT_N_MAX))
    rdumjj = min(max(mu_r + 1.0, 1.0), _RTAB_JJ_MAX)
    dumjj = min(max(int(rdumjj), 1), 9)
    w = rdumii - dumii.astype(rdumii.dtype)
    v1 = tab[dumii - 1] + w * (tab[dumii] - tab[dumii - 1])
    v2 = v1                                    # the mu_r+1 column (identical)
    return v1 + (rdumjj - float(dumjj)) * (v2 - v1)


def _cloud_dsd(qc, nc, rho, mask):
    """get_cloud_dsd2 (6635-6701), iSCF = 1, iparam = 3.  Returns
    (nc, mu_c, lamc, cdist, cdist1); outside ``mask`` the oracle's values (nc
    unchanged, mu_c = lamc = cdist = cdist1 = 0)."""
    qcs = jnp.where(mask, qc, 1.0)
    ncl = jnp.maximum(jnp.where(mask, nc, 1.0) * 1.0, _NSMALL)
    mu_c = _MUC_A * (ncl * _CM3_PER_M3 * rho) + _MUC_B
    mu_c = 1.0 / (mu_c ** 2) - 1.0
    mu_c = jnp.minimum(jnp.maximum(mu_c, _MUC_MIN), _MUC_MAX)
    lamc = (_CONS1 * ncl * (mu_c + 3.0) * (mu_c + 2.0) * (mu_c + 1.0) / qcs) ** _THRD
    lammin = (mu_c + 1.0) * _LAMC_MIN
    lammax = (mu_c + 1.0) * _LAMC_MAX
    lo = lamc < lammin
    hi = lamc > lammax
    lamc = jnp.where(lo, lammin, jnp.where(hi, lammax, lamc))
    nc_lim = 6.0 * lamc ** 3 * qcs / (_P3_PI * _RHOW * (mu_c + 3.0) * (mu_c + 2.0) * (mu_c + 1.0))
    ncl = jnp.where(lo | hi, nc_lim, ncl)
    cdist = ncl * (mu_c + 1.0) / lamc
    cdist1 = ncl / _gamma(mu_c + 1.0)
    nc_new = ncl / 1.0
    z = jnp.zeros_like(qc)
    return (jnp.where(mask, nc_new, nc), jnp.where(mask, mu_c, z),
            jnp.where(mask, lamc, z), jnp.where(mask, cdist, z), jnp.where(mask, cdist1, z))


def _rain_dsd(qr, nr, mask):
    """get_rain_dsd2 (6705-6780), iSPF = 1, mu_r = 0.  Returns
    (nr, lamr, cdistr, logn0r) with the oracle's zeros outside ``mask``."""
    qrs = jnp.where(mask, qr, 1.0)
    nrl = jnp.maximum(jnp.where(mask, nr, 1.0) * 1.0, _NSMALL)
    mu_r = _MU_R
    lamr = (_CONS1 * nrl * (mu_r + 3.0) * (mu_r + 2) * (mu_r + 1.0) / qrs) ** _THRD
    lammax = (mu_r + 1.0) * _LAMR_MAX
    lammin = (mu_r + 1.0) * _LAMR_MIN
    lo = lamr < lammin
    hi = lamr > lammax
    lamr = jnp.where(lo, lammin, jnp.where(hi, lammax, lamr))
    nr_lim = jnp.exp(3.0 * jnp.log(lamr) + jnp.log(qrs) + math.log(_G_MUR1)
                     - math.log(_G_MUR4)) / _CONS1
    nrl = jnp.where(lo | hi, nr_lim, nrl)
    logn0r = jnp.log10(nrl) + (mu_r + 1.0) * jnp.log10(lamr) - math.log10(_G_MUR1)
    cdistr = nrl / _G_MUR1
    nr_new = nrl / 1.0
    z = jnp.zeros_like(qr)
    return (jnp.where(mask, nr_new, nr), jnp.where(mask, lamr, z),
            jnp.where(mask, cdistr, z), jnp.where(mask, logn0r, z))


def _bulk_rho_rime(qi_tot, qi_rim, bi_rim, mask, rho_min, rho_max):
    """calc_bulkRhoRime (6784-6830); returns (qi_rim, bi_rim, rho_rime),
    qi_rim/bi_rim unchanged outside ``mask``."""
    has_b = bi_rim >= _RHO_RIME_BMIN
    bs = jnp.where(has_b, bi_rim, 1.0)
    rr = qi_rim / bs
    lo = rr < rho_min
    hi = rr > rho_max
    rr = jnp.where(lo, rho_min, jnp.where(hi, rho_max, rr))
    b1 = jnp.where(lo | hi, qi_rim / rr, bi_rim)
    q1 = jnp.where(has_b, qi_rim, 0.0)
    b1 = jnp.where(has_b, b1, 0.0)
    rho_rime = jnp.where(has_b, rr, 0.0)
    cap = (q1 > qi_tot) & (rho_rime > 0.0)
    q2 = jnp.where(cap, qi_tot, q1)
    b2 = jnp.where(cap, q2 / jnp.where(cap, rho_rime, 1.0), b1)
    small = q2 < _QSMALL
    q3 = jnp.where(small, 0.0, q2)
    b3 = jnp.where(small, 0.0, b2)
    return (jnp.where(mask, q3, qi_rim), jnp.where(mask, b3, bi_rim),
            jnp.where(mask, rho_rime, 0.0))


def _impose_max_total_ni(nitot, inv_rho, mask):
    """impose_max_total_Ni (6833-6855), nCat = 1."""
    m = mask & (nitot >= _NI_MIN_SUM)
    dum = _div(_MAX_TOTAL_NI * inv_rho, jnp.where(m, nitot, 1.0))
    return jnp.where(m, nitot * jnp.minimum(dum, 1.0), nitot)


def _ice_table_lookup(itab, qitot, nitot, qirim, birim, mask, cfg, idx_list):
    """nitot >= nsmall, bulk rime density, table indices and the ``idx_list``
    (0-based) table entries on the ``mask`` levels."""
    nitot = jnp.where(mask, jnp.maximum(nitot, _NSMALL), nitot)
    qirim, birim, rhop = _bulk_rho_rime(qitot, qirim, birim, mask,
                                        cfg.rho_rim_min, cfg.rho_rim_max)
    qs = jnp.where(mask, qitot, 1.0)
    ns = jnp.where(mask, nitot, 1.0)
    idx = _ice_indices(qs, ns, jnp.where(mask, qirim, 0.0), jnp.where(mask, rhop, _RHOP_LO0))
    vals = [_access(itab[..., k], idx) for k in idx_list]
    return nitot, qirim, birim, idx, vals


def _ice_number_limiter(nitot, qitot, f09, f10, mask):
    """nitot = min(nitot, f1pr09*qitot); nitot = max(nitot, f1pr10*qitot)."""
    nitot = jnp.where(mask, jnp.minimum(nitot, f09 * qitot), nitot)
    return jnp.where(mask, jnp.maximum(nitot, f10 * qitot), nitot)


def _sediment(fields0, rho, inv_rho, inv_dzq, dt, present, speeds):
    """The oracle's sedimentation sub-step loop (3995-4469) for one species,
    all columns at once, on the Morrison scheme's fixed-maximum machinery.

    ``fields0`` : tuple of (ncol, nlev) arrays, the first is the mass.
    ``speeds(fields) -> (fields, V_list, V_q)`` recomputes the fall speeds and
    applies the per-sub-step state adjustments (lambda limiters etc.);
    ``V_list[m]`` multiplies ``fields[m]`` in the flux; ``V_q`` sets the CFL.
    Whole-column fluxes equal the oracle's moving-bottom-pointer loop: outside
    the oracle's [k_qxbot-1, k_qxtop] window every flux is exactly zero (no
    mass there, so no fall speed).  At most ``_SED_SUBSTEPS_MAX`` sub-steps
    run (in chunks of ``_SED_CHUNK``, skipping chunks once every column is
    done); returns (fields, accumulated surface mass flux x dt [kg/m^2],
    overflow) where ``overflow`` is True if any column still had time left.
    """
    dt_left0 = jnp.where(present, dt, 0.0).astype(rho.dtype)
    prt0 = jnp.zeros(rho.shape[0], rho.dtype)

    def body(c):
        fields, dt_left, prt = c
        active = (dt_left > _SED_DT_MIN)[:, None]
        new_fields, vlist, vq = speeds(fields)
        fields = tuple(jnp.where(active, n, o) for n, o in zip(new_fields, fields))
        co_max = jnp.maximum(jnp.max(vq * dt_left[:, None] * inv_dzq, axis=1), 0.0)
        nsub = jnp.floor(co_max + 1.0)
        dt_sub = jnp.minimum(dt_left, dt_left / nsub)
        out = []
        for m, f in enumerate(fields):
            flux = vlist[m] * f * rho
            if m == 0:
                prt = jnp.where(active[:, 0], prt + flux[:, -1] * dt_sub, prt)
            flux_above = jnp.pad(flux[:, :-1], ((0, 0), (1, 0)))
            fluxdiv = (flux_above - flux) * inv_dzq
            out.append(jnp.where(active, f + fluxdiv * dt_sub[:, None] * inv_rho, f))
        dt_left = jnp.where(active[:, 0], dt_left - dt_sub, dt_left)
        return tuple(out), dt_left, prt

    def chunk(_, c):
        return jax.lax.cond(
            jnp.any(c[1] > _SED_DT_MIN),
            lambda cc: jax.lax.fori_loop(0, _SED_CHUNK, lambda j, x: body(x), cc),
            lambda cc: cc, c)

    fields, dt_left, prt = jax.lax.fori_loop(
        0, -(-_SED_SUBSTEPS_MAX // _SED_CHUNK), chunk, (tuple(fields0), dt_left0, prt0))
    return fields, prt, jnp.any(dt_left > _SED_DT_MIN)


# ---------------------------------------------------------------- P3 step
def p3_step(T, q_v, hydrometeors: HydrometeorState, p_full, rho, dz, dt,
            config: P3Config = P3Config(), tables: P3Tables | None = None):
    """One WRF P3 step on (ncol, nlev) top-down columns.

    ``rho`` (the caller's air density) converts N_c / N_r between per m^3 and
    per kg only; P3's processes use its own ``p/(R_d T)``.  Returns
    ``(state, prt_liq, prt_sol)``: ``state`` a dict of the post-step T, q_v,
    q_c, N_c [1/m^3], q_r, N_r [1/m^3], q_i, q_rim, N_i [1/kg], B_rim;
    ``prt_liq`` / ``prt_sol`` the oracle's surface precipitation rates [m/s
    liquid-equivalent].  Raises (``equinox.error_if``; NaN outputs and
    gradients) if a column needs more sedimentation sub-steps than the maximum.
    """
    tb = tables if tables is not None else p3_lookup_tables()
    dtype = T.dtype
    itab = jnp.asarray(tb.itab, dtype)
    itabcoll = jnp.asarray(tb.itabcoll, dtype)
    vn_tab = jnp.asarray(tb.vn, dtype)
    vm_tab = jnp.asarray(tb.vm, dtype)
    revap_tab = jnp.asarray(tb.revap, dtype)

    pres = p_full
    odt = 1.0 / dt
    inv_dzq = 1.0 / dz
    nccnst = config.Nc_0
    inv_rho_rime_max = 1.0 / config.rho_rim_max
    bsmall = _QSMALL * inv_rho_rime_max

    # --- interface glue (driver) + p3_main set-up (2287-2297) ---
    th = T * (_P0 / pres) ** (_RD / _CP)
    tmparr1 = (pres * _INV_P0) ** (_RD * _INV_CP)
    invexn = 1.0 / tmparr1
    t = th * tmparr1
    t_old = t
    qv_old = q_v
    qv = jnp.maximum(q_v, 0.0)
    qc = hydrometeors.q_c
    qr = hydrometeors.q_r
    nr = hydrometeors.N_r / rho
    qitot = hydrometeors.q_i
    qirim = hydrometeors.q_s
    nitot = hydrometeors.N_i
    birim = hydrometeors.q_g

    # --- k_loop_1 (2323-2411) ---
    rho_p = pres / (_RD * t)
    inv_rho = 1.0 / rho_p
    qvs = _qsat_liq(jnp.maximum(t_old, 1.0), pres)
    qvi = _qsat_ice(jnp.maximum(t_old, 1.0), pres)
    ssat = qv_old - qvs
    sup = qv_old / qvs - 1.0
    supi = qv_old / qvi - 1.0
    rhofacr = (_RHOSUR * inv_rho) ** _RHOFAC_EXP
    rhofaci = (_RHOSUI * inv_rho) ** _RHOFAC_EXP
    mu_air = _MU_AIR_A * t ** 1.5 / (t + _MU_AIR_T)
    acn = _G * _RHOW / (_STOKES * mu_air)
    nc = nccnst * inv_rho
    c1 = (qc < _QSMALL) | ((qc < _CLIP_DRY_QMAX) & (sup < _CLIP_DRY_SUP))
    qv = jnp.where(c1, qv + qc, qv)
    th = jnp.where(c1, th - invexn * qc * _XXLV * _INV_CP, th)
    qc = jnp.where(c1, 0.0, qc)
    nc = jnp.where(c1, 0.0, nc)
    c2 = (qr < _QSMALL) | ((qr < _CLIP_DRY_QMAX) & (sup < _CLIP_DRY_SUP))
    qv = jnp.where(c2, qv + qr, qv)
    th = jnp.where(c2, th - invexn * qr * _XXLV * _INV_CP, th)
    qr = jnp.where(c2, 0.0, qr)
    nr = jnp.where(c2, 0.0, nr)
    c3 = (qitot < _QSMALL) | ((qitot < _CLIP_DRY_QMAX) & (supi < _CLIP_DRY_SUP))
    qv = jnp.where(c3, qv + qitot, qv)
    th = jnp.where(c3, th - invexn * qitot * _XXLS * _INV_CP, th)
    qitot, nitot, qirim, birim = (jnp.where(c3, 0.0, x) for x in (qitot, nitot, qirim, birim))
    c4 = (qitot >= _QSMALL) & (qitot < _CLIP_DRY_QMAX) & (t >= _T0)
    qr = jnp.where(c4, qr + qitot, qr)
    nr = jnp.where(c4, nr + nitot, nr)
    th = jnp.where(c4, th - invexn * qitot * _XLF * _INV_CP, th)
    qitot, nitot, qirim, birim = (jnp.where(c4, 0.0, x) for x in (qitot, nitot, qirim, birim))
    qv_cld = qv          # compute_SCPF with scpf off: Qv_cld = Qv

    # --- level classes (2449-2528) ---
    exitlevel = ~((qc >= _QSMALL) | (qr >= _QSMALL) | (qitot >= _QSMALL))
    dry = ((t < _T0) & (supi < _NUC_POSSIBLE_SUP)) | ((t >= _T0) & (sup < _NUC_POSSIBLE_SUP))
    proc = ~(exitlevel & dry)        # levels that reach label 444
    full = ~exitlevel                # levels that run every process

    # --- time/space varying physical variables (2531-2554) ---
    mu = _MU_AIR_A * t ** 1.5 / (t + _MU_AIR_T)
    dv = _DV_A * t ** _DV_EXP / pres
    sc = mu / (rho_p * dv)
    dum = 1.0 / (_RV * t ** 2)
    dqsdt = _XXLV * qvs * dum
    dqsidt = _XXLS * qvi * dum
    ab = 1.0 + dqsdt * _XXLV * _INV_CP
    abi = 1.0 + dqsidt * _XXLS * _INV_CP
    kap = _KAP_A * mu
    eii = jnp.where(t < _EII_T_LO, _EII_LO,
                    jnp.where(t < _T0, _EII_LO + (t - _EII_T_LO) * (_EII_HI - _EII_LO) / _EII_DT,
                              _EII_HI))

    # --- size distributions and the total-Ni cap (2556-2566) ---
    has_qc = qc >= _QSMALL
    has_qr = qr >= _QSMALL
    cmask = full & has_qc
    nc, mu_c, lamc, cdist, cdist1 = _cloud_dsd(qc, nc, rho_p, cmask)
    rmask = full & has_qr
    nr, lamr, cdistr, logn0r = _rain_dsd(qr, nr, rmask)
    nitot = _impose_max_total_ni(nitot, inv_rho, full)

    # --- ice lookup table (2570-2677) ---
    imask = full & (qitot >= _QSMALL)
    nr = jnp.where(imask, jnp.maximum(nr, _NSMALL), nr)
    nitot, qirim, birim, idx, (f1pr02, f1pr03, f1pr04, f1pr05, f1pr09, f1pr10, f1pr14) = \
        _ice_table_lookup(itab, qitot, nitot, qirim, birim, imask, config, (1, 2, 3, 4, 6, 7, 9))
    dumj, dum3 = _rain_coll_index(qr, nr)
    f1pr07 = _access_coll(itabcoll[..., 0], idx, dumj, dum3)
    f1pr08 = _access_coll(itabcoll[..., 1], idx, dumj, dum3)
    nitot = _ice_number_limiter(nitot, qitot, f1pr09, f1pr10, imask)
    qis = jnp.where(imask, qitot, 1.0)
    frim = qirim / qis
    eii_fact = jnp.where(
        imask & (qirim > 0.0),
        jnp.where(frim < _EIIF_FR_LO, 1.0,
                  jnp.where(frim < _EIIF_FR_HI, 1.0 - (frim - _EIIF_FR_LO) / _EIIF_FR_WIDTH, 0.0)),
        1.0)

    # --- collection of droplets and rain by ice (2697-2748) ---
    col_c = rhofaci * f1pr04 * qc * _ECI * rho_p * nitot * 1.0
    qccol = jnp.where(imask & has_qc & (t <= _T0), col_c, 0.0)
    qcshd = jnp.where(imask & has_qc & (t > _T0), col_c, 0.0)
    ncshdc = qcshd * _SHED_DROP_INV_MASS
    rcol = imask & has_qr
    qrcol = jnp.where(rcol & (t <= _T0),
                      10.0 ** (f1pr08 + logn0r) * rho_p * rhofaci * _ERI * nitot * 1.0 * (1.0 - 0.0),
                      0.0)
    nrcol = jnp.where(rcol,
                      10.0 ** (f1pr07 + logn0r) * rho_p * rhofaci * _ERI * nitot * 1.0 * (1.0 - 0.0),
                      0.0)

    # --- self-collection of ice (2840-2842) ---
    nislf = jnp.where(imask, f1pr03 * rho_p * eii * eii_fact * rhofaci * nitot * nitot * 1.0, 0.0)

    # --- melting (2851-2865); T gate smoothed by melt_sharpness ---
    qsat0 = _EP_2 * _E0 / (pres - _E0)
    vent = f1pr05 + f1pr14 * sc ** _THRD * (rhofaci * rho_p / mu) ** 0.5
    melt_gate = jax.nn.sigmoid(config.melt_sharpness * (t - _T0))
    qimlt = ((vent * ((t - _T0) * kap - rho_p * _XXLV * dv * (qsat0 - qv_cld)) * 2.0 * _P3_PI / _XLF
              + 0.0) * nitot)
    qimlt = jnp.where(imask, jnp.maximum(qimlt, 0.0), 0.0) * melt_gate
    nimlt = qimlt * (nitot / qis)

    # --- wet growth and shedding (2873-2894) ---
    wet = imask & (qc + qr >= _WETGROWTH_QMIN) & (t < _T0)
    # Unit denominator outside the oracle's branch: L_f + c_pw (T - T0) is 0
    # near 194 K, and an unselected 0/0 would still poison reverse-mode AD.
    den_w = jnp.where(wet, _XLF + _CPW * (t - _T0), 1.0)
    qwgrth = ((vent * 2.0 * _P3_PI * (rho_p * _XXLV * dv * (qsat0 - qv_cld) - (t - _T0) * kap)
               / den_w) * nitot)
    qwgrth = jnp.where(wet, jnp.maximum(qwgrth, 0.0), 0.0)
    dumw = jnp.maximum(0.0, (qccol + qrcol) - qwgrth)
    shed = wet & (dumw >= _SHED_MIN)
    nrshdr = jnp.where(shed, 0.0 + dumw * _SHED_DROP_INV_MASS, 0.0)
    tot = qccol + qrcol
    shed2 = shed & (tot >= _SHED_MIN)
    dum1w = 1.0 / jnp.where(shed2, tot, 1.0)
    qcshd = jnp.where(shed2, qcshd + dumw * qccol * dum1w, qcshd)
    qccol, qrcol = (jnp.where(shed2, qccol - dumw * qccol * dum1w, qccol),
                    jnp.where(shed2, qrcol - dumw * qrcol * dum1w, qrcol))
    log_wetgrowth = shed

    # --- ice relaxation timescale (2900-2906) ---
    epsi = jnp.where(imask & (t < _T0), (vent * 2.0 * _P3_PI * rho_p * dv) * nitot, 0.0)
    epsi_tot = 0.0 + epsi

    # --- rime density, Cober & List (2925-2969) ---
    # (the oracle's "keep the previous level's rhorime_c" case needs qccol >=
    # qsmall with qc < qsmall, which cannot happen: qccol > 0 needs qc >= qsmall)
    rime = (qccol >= _QSMALL) & (t < _T0) & has_qc
    vtrmi1 = f1pr02 * rhofaci
    itc = 1.0 / jnp.minimum(_RIME_ITC_MAX, t - _T0)
    lamc_s = jnp.where(cmask, lamc, 1.0)
    vt_qc = acn * _gamma(4.0 + _BCN + mu_c) / (lamc_s ** _BCN * _gamma(mu_c + 4.0))
    d_c = (mu_c + 4.0) / lamc_s
    v_impact = jnp.abs(vtrmi1 - vt_qc)
    ri = -(_RIME_RI_A * d_c) * v_impact * itc
    ri = jnp.maximum(_RIME_RI_MIN, jnp.minimum(ri, _RIME_RI_MAX))
    rho_cl = jnp.where(ri <= _RIME_RI_SPLIT,
                       (_RIME_C0 + _RIME_C1 * ri - _RIME_C2 * ri ** 2) * _RIME_SCALE,
                       _RIME_HI0 + _RIME_HI1 * (ri - _RIME_RI_SPLIT))
    rhorime_c = jnp.where(rime, rho_cl, _RHO_RIME_DEFAULT)

    # --- immersion freezing of cloud (2984-3015) and rain (3021-3043) ---
    het_c = cmask & (t <= _T_IMMERSION)
    tmpdbl1 = jnp.exp(_AIMM * (_T0 - t))
    # WRF's product cdist1*exp(aimm*(T0-T))*gamma(7+mu_c)*dumh**2 is formed in
    # log space (tier-2 decision C8): the same value to round-off in fp64, no
    # fp32 overflow of the intermediate product for cloud at or below ~190 K.
    # Masked lanes take 1.0 inside the logs and exponent 0, so neither the
    # value nor the gradient there is inf/NaN (fp32 at very low T).
    log_c1 = jnp.log(jnp.where(het_c, cdist1, 1.0))
    log_lc = jnp.log(jnp.where(het_c, lamc_s, 1.0))       # log(dumh) = -3*log_lc
    log_h = log_c1 + _AIMM * (_T0 - t)
    ex_q = jnp.where(het_c, _gammaln(_HET_GAMMA_A + mu_c) + log_h - 6.0 * log_lc, 0.0)
    ex_n = jnp.where(het_c, _gammaln(mu_c + 4.0) + log_h - 3.0 * log_lc, 0.0)
    qcheti = jnp.where(het_c, _CONS6 * jnp.exp(ex_q), 0.0)
    ncheti = jnp.where(het_c, _CONS5 * jnp.exp(ex_n), 0.0)
    het_r = rmask & (t <= _T_IMMERSION)
    cds = jnp.where(het_r, cdistr, 1.0)
    lrs = jnp.where(het_r, lamr, 1.0)
    tq1 = jnp.exp(jnp.log(cds) + math.log(_G_MUR7) - 6.0 * jnp.log(lrs))
    tq2 = jnp.exp(jnp.log(cds) + math.log(_G_MUR4) - 3.0 * jnp.log(lrs))
    qrheti = jnp.where(het_r, _CONS6 * (tq1 * tmpdbl1) * 1.0, 0.0)
    nrheti = jnp.where(het_r, _CONS5 * (tq2 * tmpdbl1) * 1.0, 0.0)

    # --- condensation/evaporation, deposition/sublimation (3114-3255) ---
    lr_s = jnp.where(rmask, lamr, 1.0)
    vdum = _rain_table_interp(revap_tab, _MU_R, lr_s)
    epsr = jnp.where(rmask, 2.0 * _P3_PI * cdistr * rho_p * dv
                     * (_F1R * _G_MUR2 / lr_s + _F2R * (rho_p / mu) ** 0.5 * sc ** _THRD * vdum), 0.0)
    epsc = jnp.where(cmask, 2.0 * _P3_PI * rho_p * dv * cdist, 0.0)
    cold = t < _T0
    oabi = 1.0 / abi
    xx = jnp.where(cold, epsc + epsr + epsi_tot * (1.0 + _XXLS * _INV_CP * dqsdt) * oabi,
                   epsc + epsr)
    dumqvi = qvi
    dumw2 = -_CP / _G * (t - t_old) * odt
    aaa = jnp.where(
        cold,
        (qv - qv_old) * odt - dqsdt * (-dumw2 * _G * _INV_CP)
        - (qvs - dumqvi) * (1.0 + _XXLS * _INV_CP * dqsdt) * oabi * epsi_tot,
        (qv - qv_old) * odt - dqsdt * (-dumw2 * _G * _INV_CP))
    xx = _floor(xx, _XX_MIN)
    oxx = _div(1.0, xx)
    ssat_cld = ssat
    ssat_r = ssat
    expf = 1.0 - jnp.exp(-(xx * dt))
    qccon = jnp.where(cmask,
                      (aaa * epsc * oxx + (ssat_cld * 1.0 - aaa * oxx) * odt * epsc * oxx * expf) / ab,
                      0.0)
    qrcon = jnp.where(rmask,
                      (aaa * epsr * oxx + (ssat_r * 1.0 - aaa * oxx) * odt * epsr * oxx * expf) / ab,
                      0.0)
    qccon = jnp.where(full & (sup < _EVAP_SUPMIN) & (qc < _EVAP_QMIN), -qc * odt, qccon)
    qrcon = jnp.where(full & (sup < _EVAP_SUPMIN) & (qr < _EVAP_QMIN), -qr * odt, qrcon)
    evc = qccon < 0.0
    qcevp = jnp.where(evc, -qccon, 0.0)
    qccon = jnp.where(evc, 0.0, jnp.minimum(qccon, qv * odt))
    evr = qrcon < 0.0
    qrevp = jnp.where(evr, -qrcon, 0.0)
    nrevp = jnp.where(evr, qrevp * (nr / jnp.where(evr, qr, 1.0)), 0.0)
    qrcon = jnp.where(evr, 0.0, jnp.minimum(qrcon, qv * odt))
    qidep = jnp.where(imask & cold,
                      (aaa * epsi * oxx + (ssat_cld * 1.0 - aaa * oxx) * odt * epsi * oxx * expf) * oabi
                      + (qvs - dumqvi) * epsi * oabi, 0.0)
    qidep = jnp.where(full & (supi < _EVAP_SUPMIN) & (qitot < _EVAP_QMIN), -qitot * odt, qidep)
    sub = qidep < 0.0
    qisub = jnp.where(sub, jnp.minimum(-qidep * _CLBFACT_SUB, qitot * odt), 0.0)
    nisub = jnp.where(sub, qisub * (nitot / jnp.where(sub, qitot, 1.0)), 0.0)
    qidep = jnp.where(sub, 0.0, jnp.minimum(qidep * _CLBFACT_DEP, qv * odt))
    # levels that skip to label 444 run none of the above
    qccon, qcevp, qrcon, qrevp, nrevp, qidep, qisub, nisub = (
        jnp.where(full, x, 0.0) for x in (qccon, qcevp, qrcon, qrevp, nrevp, qidep, qisub, nisub))

    # ==================== label 444: every level in ``proc`` ====================
    # --- Cooper (1986) deposition/condensation-freezing nucleation (3268-3289);
    #     T and supi gates smoothed by ice_sigmoid_sharpness / cooper_supi_sharpness ---
    nuc_gate = (jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_nuc - t))
                * jax.nn.sigmoid(config.cooper_supi_sharpness * (supi - config.cooper_supi_min)))
    dumn = config.N_i0 * jnp.exp(config.cooper_a * (_T0 - t)) * inv_rho
    dumn = jnp.minimum(dumn, config.N_i_nuc_max * inv_rho * 1.0)
    n_nuc = jnp.maximum(0.0, (dumn - nitot) * odt)
    q_nuc = jnp.maximum(0.0, (dumn - nitot) * _MI0 * odt)
    nuc_on = proc & (n_nuc >= _NUC_MIN)
    ninuc = jnp.where(nuc_on, n_nuc, 0.0) * nuc_gate
    qinuc = jnp.where(nuc_on, q_nuc, 0.0) * nuc_gate

    # --- droplet activation, specified Nc, it > 1 (3299-3307) ---
    duma = nccnst * inv_rho * _CONS7 - qc
    duma = jnp.maximum(0.0, duma * 1.0)
    dumqvs = _qsat_liq(t, pres)
    dqsdt_a = _XXLV * dumqvs / (_RV * t * t)
    ab_a = 1.0 + dqsdt_a * _XXLV * _INV_CP
    duma = jnp.maximum(0.0, jnp.minimum(duma, (qv_cld - dumqvs) / ab_a))
    qcnuc = jnp.where(proc & (sup > _ACT_SUPMIN), duma * odt * 1.0, 0.0)

    # --- KK2000 autoconversion (3358-3412) and accretion (3436-3468) ---
    au = proc & (qc * 1.0 >= _AUTO_QCMIN)
    qcs = jnp.where(au, qc, 1.0)
    ncs = jnp.where(au, nc, 1.0)
    prc, _, _ = autoconversion_kk2000(qcs * 1.0, ncs * 1.0 * rho_p, rho_p, dt)
    qcaut = jnp.where(au, prc * 1.0, 0.0)
    ncautr = qcaut * _CONS3
    ncautc = jnp.where(au, qcaut * ncs / qcs, 0.0)
    ncautc = jnp.where(qcaut == 0.0, 0.0, ncautc)
    qcaut = jnp.where(ncautc == 0.0, 0.0, qcaut)
    ac = proc & has_qr & has_qc
    qcacc = jnp.where(ac, accretion_kk2000(qc * 1.0, qr * 1.0) * (1.0 - 0.0), 0.0)
    ncacc = jnp.where(ac, qcacc * nc / jnp.where(ac, qc, 1.0), 0.0)
    ncacc = jnp.where(qcacc == 0.0, 0.0, ncacc)
    qcacc = jnp.where(ncacc == 0.0, 0.0, qcacc)

    # --- rain self-collection and breakup (3474-3500) ---
    sr = proc & has_qr
    nr = jnp.where(sr, jnp.maximum(nr, _NSMALL), nr)
    nrs = jnp.where(sr, nr, 1.0)
    qrs = jnp.where(sr, qr, 1.0)
    # inactive cells -> dum2r = 0 (no breakup branch; nrslf is 0 there anyway):
    # keeps the unselected exp finite for the reverse pass (fp32 overflow)
    dum2r = jnp.where(sr, (qrs / (_P3_PI * _RHOW * nrs)) ** _THRD, 0.0)
    brk = jnp.where(dum2r < _RAIN_D_BREAKUP, 1.0,
                    2.0 - jnp.exp(_BREAKUP_EXP * (dum2r - _RAIN_D_BREAKUP)))
    nrslf = jnp.where(sr, brk * _KR_RAIN_SC * nrs * 1.0 * qrs * 1.0 * rho_p * 1.0, 0.0)

    # --- saturation-adjustment limits (3511-3563) ---
    qcon_satadj = (qv_cld - dumqvs) / (1.0 + _XXLV ** 2 * dumqvs / (_CP * _RV * t ** 2)) * odt * 1.0
    tmp1 = qccon + qrcon + qcnuc
    zero_c = (tmp1 > 0.0) & (qcon_satadj < 0.0)
    lim_c = ~zero_c & (tmp1 > 0.0) & (tmp1 > qcon_satadj)
    ratio = jnp.minimum(1.0, _div(jnp.maximum(0.0, qcon_satadj), jnp.where(lim_c, tmp1, 1.0)))
    qccon, qrcon, qcnuc = (jnp.where(zero_c, 0.0, jnp.where(lim_c, x * ratio, x))
                           for x in (qccon, qrcon, qcnuc))
    evs = qcevp + qrevp
    lim_e = ~zero_c & ~lim_c & (evs > 0.0)
    ratio = jnp.minimum(1.0, _div(jnp.maximum(0.0, -qcon_satadj), jnp.where(lim_e, evs, 1.0)))
    qcevp, qrevp, nrevp = (jnp.where(lim_e, x * ratio, x) for x in (qcevp, qrevp, nrevp))

    qv_tmp = qv_cld + (-qcnuc - qccon - qrcon + qcevp + qrevp) * dt
    t_tmp = t + (qcnuc + qccon + qrcon - qcevp - qrevp) * _XXLV * _INV_CP * dt
    dumqvi2 = _qsat_ice(t_tmp, pres)
    qdep_satadj = (qv_tmp - dumqvi2) / (1.0 + _XXLS ** 2 * dumqvi2 / (_CP * _RV * t_tmp ** 2)) \
        * odt * 1.0
    tmp1 = qidep + qinuc
    zero_d = (tmp1 > 0.0) & (qdep_satadj < 0.0)
    lim_d = ~zero_d & (tmp1 > 0.0) & (tmp1 > qdep_satadj)
    ratio = jnp.minimum(1.0, _div(jnp.maximum(0.0, qdep_satadj), jnp.where(lim_d, tmp1, 1.0)))
    qidep, qinuc, ninuc = (jnp.where(zero_d, 0.0, jnp.where(lim_d, x * ratio, x))
                           for x in (qidep, qinuc, ninuc))
    dums = _floor(qisub, _SAT_TINY)
    qisub_l = qisub * jnp.minimum(1.0, _div(jnp.maximum(0.0, -qdep_satadj), _floor(qisub, _SAT_TINY)))
    nisub_l = nisub * jnp.minimum(1.0, _div(qisub_l, dums))
    qisub = jnp.where(zero_d, qisub, qisub_l)
    nisub = jnp.where(zero_d, nisub, nisub_l)

    # --- mass-conservation limiters (3567-3640) ---
    sinks = (qcaut + qcacc + qccol + qcevp + 0.0 + qcheti + qcshd) * dt
    sources = qc + (qccon + qcnuc) * dt
    lim = (sinks > sources) & (sinks >= _SAT_TINY)
    ratio = _div(sources, jnp.where(lim, sinks, 1.0))
    qcaut, qcacc, qcevp, qccol, qcheti, qcshd, ncautc, ncacc, ncheti = (
        jnp.where(lim, x * ratio, x)
        for x in (qcaut, qcacc, qcevp, qccol, qcheti, qcshd, ncautc, ncacc, ncheti))
    sinks = (qrevp + qrcol + 0.0 + qrheti + 0.0) * dt
    sources = qr + (qrcon + qcaut + qcacc + qimlt + qcshd) * dt
    lim = (sinks > sources) & (sinks >= _SAT_TINY)
    ratio = _div(sources, jnp.where(lim, sinks, 1.0))
    qrevp, qrcol, qrheti, nrevp, nrcol, nrheti = (
        jnp.where(lim, x * ratio, x) for x in (qrevp, qrcol, qrheti, nrevp, nrcol, nrheti))
    sinks = (qisub + qimlt) * dt
    sources = qitot + (qidep + qinuc + qrcol + qccol + 0.0 + qrheti + 0.0 + qcheti + 0.0) * dt
    lim = (sinks > sources) & (sinks >= _SAT_TINY)
    ratio = _div(sources, jnp.where(lim, sinks, 1.0))
    qisub, qimlt, nisub, nimlt = (
        jnp.where(lim, x * ratio, x) for x in (qisub, qimlt, nisub, nimlt))
    sinks = (qccon + qrcon + qcnuc + qidep + qinuc) * dt
    sources = qv + (qcevp + qrevp + qisub) * dt
    lim = (sinks > sources) & (sinks >= _SAT_TINY)
    ratio = _div(sources, jnp.where(lim, sinks, 1.0))
    qccon, qrcon, qcnuc, qidep, qinuc, ninuc = (
        jnp.where(lim, x * ratio, x) for x in (qccon, qrcon, qcnuc, qidep, qinuc, ninuc))

    # --- state update and end-of-level clipping (3750-3925), ``proc`` levels ---
    has_i = qitot >= _QSMALL
    qis2 = jnp.where(has_i, qitot, 1.0)
    rimevolume = jnp.where(has_i, birim / qis2, 0.0)
    rimefraction = jnp.where(has_i, qirim / qis2, 0.0)
    qc_u = qc + (-0.0 - qcheti - qccol - qcshd) * dt
    qr_u = qr + (-qrcol + qimlt - 0.0 - qrheti + qcshd - 0.0) * dt
    nr_u = nr + (-nrcol - 0.0 - nrheti + _NMLTRATIO * nimlt + nrshdr + ncshdc) * dt
    birim_u = birim - ((qisub + qimlt) * dt * rimevolume)
    qirim_u = qirim - ((qisub + qimlt) * dt * rimefraction)
    qitot_u = qitot - (qisub + qimlt) * dt
    dumi = (qrcol + qccol + 0.0 + qrheti + 0.0 + qcheti + 0.0) * dt
    qitot_u = qitot_u + (qidep + qinuc) * dt + dumi
    qirim_u = qirim_u + dumi
    birim_u = birim_u + (qrcol * inv_rho_rime_max + qccol / rhorime_c
                         + (0.0 + qrheti + 0.0 + qcheti + 0.0) * inv_rho_rime_max) * dt
    nitot_u = nitot + (ninuc - nimlt - nisub - nislf + 0.0 + nrheti + 0.0 + ncheti + 0.0) * dt
    neg = qirim_u < 0.0
    qirim_u = jnp.where(neg, 0.0, qirim_u)
    birim_u = jnp.where(neg, 0.0, birim_u)
    qirim_u = jnp.where(log_wetgrowth, qitot_u, qirim_u)
    birim_u = jnp.where(log_wetgrowth, qirim_u * inv_rho_rime_max, birim_u)
    # densification of rime mass on melting (3837-3841)
    dens = (qitot_u >= _QSMALL) & (birim_u >= bsmall) & (qimlt > 0.0)
    bs_ = jnp.where(dens, birim_u, 1.0)
    rho_b = qirim_u / bs_
    q_bm = jnp.where(dens, qitot_u + qimlt * dt, 1.0)
    # unselected levels get a unit denominator (their 0/0 would NaN the reverse pass)
    den_b = jnp.where(dens, rho_b + (_RHO_ICE_BULK - rho_b) * qimlt * dt / q_bm, 1.0)
    birim_u = jnp.where(dens, qirim_u / den_b, birim_u)
    qv_u = qv + (-qidep + qisub - qinuc) * dt
    th_u = th + invexn * ((qidep - qisub + qinuc) * _XXLS * _INV_CP
                          + (qrcol + qccol + 0.0 + qcheti + 0.0 + qrheti + 0.0 - qimlt)
                          * _XLF * _INV_CP) * dt
    qc_u = qc_u + (-qcacc - qcaut + qcnuc + qccon - qcevp) * dt
    qr_u = qr_u + (qcacc + qcaut + qrcon - qrevp) * dt
    nc_u = nccnst * inv_rho
    nr_u = nr_u + (ncautr - nrslf - nrevp) * dt
    qv_u = qv_u + (-qcnuc - qccon - qrcon + qcevp + qrevp) * dt
    th_u = th_u + invexn * ((qcnuc + qccon + qrcon - qcevp - qrevp) * _XXLV * _INV_CP) * dt
    k1 = qc_u < _QSMALL
    qv_u = jnp.where(k1, qv_u + qc_u, qv_u)
    th_u = jnp.where(k1, th_u - invexn * qc_u * _XXLV * _INV_CP, th_u)
    qc_u = jnp.where(k1, 0.0, qc_u)
    nc_u = jnp.where(k1, 0.0, nc_u)
    k2 = qr_u < _QSMALL
    qv_u = jnp.where(k2, qv_u + qr_u, qv_u)
    th_u = jnp.where(k2, th_u - invexn * qr_u * _XXLV * _INV_CP, th_u)
    qr_u = jnp.where(k2, 0.0, qr_u)
    nr_u = jnp.where(k2, 0.0, nr_u)
    k3 = qitot_u < _QSMALL
    qv_u = jnp.where(k3, qv_u + qitot_u, qv_u)
    th_u = jnp.where(k3, th_u - invexn * qitot_u * _XXLS * _INV_CP, th_u)
    qitot_u, nitot_u, qirim_u, birim_u = (jnp.where(k3, 0.0, x)
                                          for x in (qitot_u, nitot_u, qirim_u, birim_u))
    qv_u = jnp.maximum(0.0, qv_u)
    nitot_u = _impose_max_total_ni(nitot_u, inv_rho, proc)
    # log_hydrometeorsPresent after the process loop (3897-3914, 3970)
    present = jnp.any(proc & ~(k1 & k2 & k3), axis=1, keepdims=True)

    def upd(new, old):
        return jnp.where(proc, new, old)

    qc, qr, nc, nr = upd(qc_u, qc), upd(qr_u, qr), upd(nc_u, nc), upd(nr_u, nr)
    qitot, qirim = upd(qitot_u, qitot), upd(qirim_u, qirim)
    nitot, birim = upd(nitot_u, nitot), upd(birim_u, birim)
    qv, th = upd(qv_u, qv), upd(th_u, th)

    # ==================== sedimentation (3984-4495) ====================
    def cloud_speeds(f):
        qc_, nc_ = f
        m = qc_ * 1.0 > _QSMALL
        nc_, mu_c_, lamc_, _, _ = _cloud_dsd(qc_, nc_, rho_p, m)
        lam = jnp.where(m, lamc_, 1.0)
        v = jnp.where(m, acn * _gamma(4.0 + _BCN + mu_c_) * (1.0 / lam ** _BCN)
                      / _gamma(mu_c_ + 4.0), 0.0)
        return (qc_, nc_), (v, jnp.zeros_like(v)), v

    def rain_speeds(f):
        qr_, nr_ = f
        m = qr_ * 1.0 > _QSMALL
        nr_ = jnp.where(m, jnp.maximum(nr_, _NSMALL), nr_)
        nr_, lamr_, _, _ = _rain_dsd(qr_, nr_, m)
        lam = jnp.where(m, lamr_, 1.0)
        vq = jnp.where(m, _rain_table_interp(vm_tab, _MU_R, lam) * rhofacr, 0.0)
        vnr = jnp.where(m, _rain_table_interp(vn_tab, _MU_R, lam) * rhofacr, 0.0)
        return (qr_, nr_), (vq, vnr), vq

    def ice_speeds(f):
        qi_, ni_, qir_, bir_ = f
        m = qi_ >= _QSMALL
        ni_, qir_, bir_, _, (f01, f02, f09, f10) = _ice_table_lookup(
            itab, qi_, ni_, qir_, bir_, m, config, (0, 1, 6, 7))
        ni_ = _ice_number_limiter(ni_, qi_, f09, f10, m)
        vq = jnp.where(m, f02 * rhofaci, 0.0)
        vn = jnp.where(m, f01 * rhofaci, 0.0)
        return (qi_, ni_, qir_, bir_), (vq, vn, vq, vq), vq

    pcol = present[:, 0]
    (qc, nc), acc_c, over_c = _sediment((qc, nc), rho_p, inv_rho, inv_dzq, dt,
                                        pcol & jnp.any(qc * 1.0 >= _QSMALL, axis=1),
                                        cloud_speeds)
    (qr, nr), acc_r, over_r = _sediment((qr, nr), rho_p, inv_rho, inv_dzq, dt,
                                        pcol & jnp.any(qr * 1.0 >= _QSMALL, axis=1),
                                        rain_speeds)
    (qitot, nitot, qirim, birim), acc_i, over_i = _sediment(
        (qitot, nitot, qirim, birim), rho_p, inv_rho, inv_dzq, dt,
        pcol & jnp.any(qitot >= _QSMALL, axis=1), ice_speeds)
    prt_liq = acc_c * _INV_RHOW * odt
    prt_liq = prt_liq + acc_r * _INV_RHOW * odt
    prt_sol = 0.0 + acc_i * _INV_RHOW * odt

    # ==================== homogeneous freezing (4528-4610) ====================
    for is_cloud in (True, False):
        qx, nx = (qc, nc) if is_cloud else (qr, nr)
        hf = present & (qx >= _QSMALL) & (t < _T_HOMOG)
        nn = jnp.maximum(nx, _NSMALL)
        qirim = jnp.where(hf, qirim + qx, qirim)
        qitot = jnp.where(hf, qitot + qx, qitot)
        birim = jnp.where(hf, birim + qx * inv_rho_rime_max, birim)
        nitot = jnp.where(hf, nitot + nn, nitot)
        th = jnp.where(hf, th + invexn * qx * _XLF * _INV_CP, th)
        if is_cloud:
            qc, nc = jnp.where(hf, 0.0, qc), jnp.where(hf, 0.0, nc)
        else:
            qr, nr = jnp.where(hf, 0.0, qr), jnp.where(hf, 0.0, nr)

    # ==================== final checks (4722-4895) ====================
    fc = present & (qc * 1.0 >= _QSMALL)
    nc_f, _, _, _, _ = _cloud_dsd(qc, nc, rho_p, fc)
    gone = present & ~fc
    qv = jnp.where(gone, qv + qc, qv)
    th = jnp.where(gone, th - invexn * qc * _XXLV * _INV_CP, th)
    nc = jnp.where(fc, nc_f, jnp.where(gone, 0.0, nc))
    qc = jnp.where(gone, 0.0, qc)
    fr = present & (qr >= _QSMALL)
    nr_f, _, _, _ = _rain_dsd(qr, nr, fr)
    gone = present & ~fr
    qv = jnp.where(gone, qv + qr, qv)
    th = jnp.where(gone, th - invexn * qr * _XXLV * _INV_CP, th)
    nr = jnp.where(fr, nr_f, jnp.where(gone, 0.0, nr))
    qr = jnp.where(gone, 0.0, qr)
    nitot = _impose_max_total_ni(nitot, inv_rho, jnp.broadcast_to(present, nitot.shape))
    fi = present & (qitot >= _QSMALL)
    nr = jnp.where(fi, jnp.maximum(nr, _NSMALL), nr)
    nitot, qirim, birim, _, (g09, g10) = _ice_table_lookup(
        itab, qitot, nitot, qirim, birim, fi, config, (6, 7))
    nitot = _ice_number_limiter(nitot, qitot, g09, g10, fi)
    rsmall = fi & (qirim < _QSMALL)
    qirim = jnp.where(rsmall, 0.0, qirim)
    birim = jnp.where(rsmall, 0.0, birim)
    gone = present & ~fi
    qv = jnp.where(gone, qv + qitot, qv)
    th = jnp.where(gone, th - invexn * qitot * _XXLS * _INV_CP, th)
    qitot, nitot, qirim, birim = (jnp.where(gone, 0.0, x) for x in (qitot, nitot, qirim, birim))
    nr = jnp.where(present & (qr < _QSMALL), 0.0, nr)

    T_new = th * (pres * _INV_P0) ** (_RD / _CP)
    state = dict(T=T_new, q_v=qv, q_c=qc, N_c=nc * rho, q_r=qr, N_r=nr * rho,
                 q_i=qitot, q_rim=qirim, N_i=nitot, B_rim=birim)
    # A column that needs more sedimentation sub-steps than the fixed maximum
    # is an error (the Morrison strict path): equinox error_if on every output,
    # plus a NaN poison so jit(grad(...)) and outputs that do not descend from
    # sedimentation cannot drop the check.
    overflow = over_c | over_r | over_i
    leaves = (state, prt_liq, prt_sol)
    leaves = eqx.error_if(
        leaves, overflow,
        "P3 sedimentation: a column needs more CFL sub-steps than the fixed "
        f"maximum {_SED_SUBSTEPS_MAX} (MorrisonConfig default "
        "sed_cfl_substeps_max); shorten the physics step")
    poison = jnp.where(overflow, jnp.nan, 1.0).astype(dtype)
    state, prt_liq, prt_sol = jax.tree_util.tree_map(lambda x: x * poison, leaves)
    return state, prt_liq, prt_sol


def p3_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: P3Config = P3Config(),
) -> MicrophysicsOutput:
    """WRF P3 tendencies, ``(state after the P3 step - input state) / dt``.

    Parameters
    ----------
    T, q_v : (ncol, nlev) temperature [K], specific humidity [kg/kg]
    hydrometeors : HydrometeorState (q_s = q_rim [kg/kg], q_g = B_rim
        [m^3/kg], N_c and N_r per m^3, N_i per kg); the input N_c is not
        read (specified droplet number) and dN_c_dt resets it.
    p_full, p_half : (ncol, nlev), (ncol, nlev+1) pressure [Pa] (p_half unused)
    rho : (ncol, nlev) air density [kg/m^3] (N_c / N_r unit conversion only)
    dz : (ncol, nlev) layer thickness [m]
    dt : physics time step [s]
    config : P3Config

    Returns
    -------
    MicrophysicsOutput (dq_s_dt carries dq_rim/dt, dq_g_dt dB_rim/dt;
    precipitation and precip_frozen in kg/m^2/s).
    """
    del p_half
    s, prt_liq, prt_sol = p3_step(T, q_v, hydrometeors, p_full, rho, dz, dt, config)
    odt = 1.0 / dt
    dT_dt = (s["T"] - T) * odt
    dq_v_dt = (s["q_v"] - q_v) * odt
    dq_c_dt = (s["q_c"] - hydrometeors.q_c) * odt
    # Opt-in repo-wide super-saturation guard (not part of WRF P3; default
    # OFF => not traced, oracle path unchanged).  Same shared implementation
    # and post-step placement as ml_emulator: drain what the P3 step leaves
    # above hard_sat_adjust_threshold * q_sat onto the liquid curve,
    # q_v -= rate*dt, q_c += rate*dt, T += (L_v/c_pd)*rate*dt (rate >= 0 =
    # condensation; conserves total water and c_pd*T + L_v*q_v).
    if config.hard_saturation_adjustment:
        if jax.config.jax_enable_x64:
            q_sat_post = _qsat_liq(s["T"].astype(jnp.float64),
                                   p_full.astype(jnp.float64)).astype(s["q_v"].dtype)
        else:
            q_sat_post = _qsat_liq(s["T"], p_full)
        rate = hard_saturation_blend(
            jnp.zeros_like(s["q_v"]), s["T"], s["q_v"], p_full, dt, q_sat_post,
            config.hard_sat_adjust_threshold, config.hard_sat_max_heating_K)
        dq_v_dt = dq_v_dt - rate
        dq_c_dt = dq_c_dt + rate
        dT_dt = dT_dt + (constants.L_v / constants.c_pd) * rate
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=(s["q_r"] - hydrometeors.q_r) * odt,
        dq_i_dt=(s["q_i"] - hydrometeors.q_i) * odt,
        dq_s_dt=(s["q_rim"] - hydrometeors.q_s) * odt,
        dq_g_dt=(s["B_rim"] - hydrometeors.q_g) * odt,
        dN_c_dt=(s["N_c"] - hydrometeors.N_c) * odt,
        dN_r_dt=(s["N_r"] - hydrometeors.N_r) * odt,
        dN_i_dt=(s["N_i"] - hydrometeors.N_i) * odt,
        precipitation=(prt_liq + prt_sol) * _RHOW,
        precip_frozen=prt_sol * _RHOW,
    )
