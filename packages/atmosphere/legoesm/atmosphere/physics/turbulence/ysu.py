"""YSU (Yonsei University) PBL turbulence scheme.

Two code paths share :func:`ysu_turbulence`:

* **WRF YSU (faithful)** — taken when the caller passes ``surface_layer``
  (a :class:`YSUSurfaceLayer`: the surface-layer state WRF's YSU reads from
  its surface-layer scheme).  :func:`bl_ysu_run` is a JAX port of WRF v4.8.0
  ``phys/physics_mmm/bl_ysu.F90::bl_ysu_run`` (WRF 06d4240ae989cc3e50af412bb472df3d9048783c,
  physics_mmm 550b5b494758ad2756394e93cb08b5709b8feb83) on its clear-sky path
  (``ysu_topdown_pblmix=.false.``, no cloud water/ice, no BEP urban canopy,
  ``ctopo`` absent, no ocean current), oracle-tested against the running
  Fortran (``tests/atmosphere/hydrostatic/unit/test_ysu_oracle.py``).  The
  existing sharpness fields smooth the oracle's three hard switches and reduce
  to it at their hard limit; every other switch is ported as ``jnp.where``.
  **NOT RULE 5 COMPLIANT** (user decision 2026-10-28): the PBL-top level and
  the switches derived from it are hard at the default.  Opt-in kernel only;
  these switches must be smoothed before any lane uses this path (#2231).
* **Legacy surrogate (NOT-YET-FAITHFUL)** — taken when ``surface_layer`` is
  ``None``, which is what every lane does today: no lane builds the WRF
  surface-layer state (bulk Richardson number, integrated stability functions,
  roughness, 10 m wind, land/water flag).  This path is NOT WRF YSU: the
  tier-2 oracle audit measured PBL height 40 % (median) and diffusivities
  ~90 % (median) away from the oracle.  It is kept unchanged, as documented
  below, until it is decided how lanes supply the surface-layer state.

The rest of this docstring describes the LEGACY SURROGATE path.

Nonlocal K-profile with an explicit entrainment flux at the PBL top.
The K-profile follows the same structure as Holtslag-Boville but adds a
free-convective-LIMIT surrogate for the Hong et al. (2006) PBL-top
entrainment: a prescribed heat flux ``(w'θ')_h = −e_ratio·(w'θ')_0`` (a fixed
fraction of the SURFACE flux, dropping Hong06's shear-dependent ``w_m³`` term),
applied as a flux-matched diffusivity localized at the inversion (see the
entrainment block in :func:`ysu_turbulence` and the Faithfulness section below).

.. note::

   **Implementation note vs published YSU.** Published YSU (Hong et al.
   2006) prescribes the PBL-top entrainment as an explicit,
   gradient-independent inversion-flux source term (not a boundary
   condition), scaled by the entrainment velocity ``w_m³ = w*³ + 5·u*³`` (so it
   carries a mechanical ``u*``/shear contribution).  This module makes
   TWO simplifications: (i) the entrainment flux is taken as a fixed
   fraction of the SURFACE flux, ``w'θ'_h ≈ −0.15·w'θ'_0`` — the
   FREE-CONVECTIVE limit, dropping the shear term; and (ii) because the
   module builds K profiles and integrates them with implicit vertical
   diffusion, that prescribed flux enters as a FLUX-MATCHED eddy
   diffusivity
   ``K_h,ent = e_ratio·(w'θ')_0 / max(∂θ_v/∂z, floor)`` under a
   Gaussian envelope centred on ``h_pbl`` (width
   ``config.entrainment_width_frac``): the resulting down-gradient flux
   ``−K_ent·∂θ_v/∂z`` at the inversion reproduces the prescribed
   ``(w'θ')_h`` exactly where the inversion gradient exceeds the floor,
   and saturates (stability-capped) in weakly stratified interfaces.
   Chosen as a smooth, fully differentiable closure (no prescribed-
   inversion-flux branch) at GCM-typical vertical resolution.

Faithfulness to Hong et al. (2006) / Troen-Mahrt (1986)
-------------------------------------------------------
FAITHFUL (published forms + constants):
  * **Nonlocal K-profile** ``K_m = κ·w_s·z·(1 − z/h)²`` (Troen-Mahrt 1986 /
    Hong06), whose shape ``(z/h)(1 − z/h)²`` peaks at exactly ``4/27`` at
    ``z = h/3`` (``_KPROFILE_PEAK_FRAC``, used as the entrainment-K stability cap).
  * **Mixed-layer velocity scale** ``w_s = (u*³ + c·κ·w*³·z/h)^{1/3}`` blending
    mechanical and convective scaling (Hong06), with the convective velocity
    ``w* = ((g/θ)·h·(w'θ')_0)^{1/3}``.
  * **Bulk-Richardson PBL height** with the Troen-Mahrt thermal-excess parcel
    ``θ_T = b·(w'θ')_0/w_s`` — an unstable surface heat flux deepens ``h_pbl``.
  * **Local-Ri free-atmosphere mixing** above the PBL (a genuine YSU component) and
    the entrainment COEFFICIENT ``e_ratio = 0.15`` (the Hong06 free-convective
    entrainment value).  Sign: unstable ``(w'θ')_0 > 0`` ⇒ downward entrainment flux
    (``K_ent ≥ 0``).
DEPARTURES / SURROGATES:
  * **Fixed surface-flux-ratio entrainment**: the implemented
    ``(w'θ')_h = −e_ratio·(w'θ')_0`` is only the FREE-CONVECTIVE LIMIT of Hong06.
    The published scheme scales the entrainment velocity by ``w_m³ = w*³ + 5·u*³``,
    so its true PBL-top flux carries a mechanical (``u*``/shear) contribution that
    this shear-independent fixed ratio DROPS.  Only the 0.15 coefficient is the
    published value; the fixed-fraction-of-surface-flux LAW is a surrogate.
  * **Flux-matched K realization** of that prescribed flux (see the note above):
    ``K_ent = e_ratio·(w'θ')_0/max(∂θ_v/∂z, floor)`` under a Gaussian envelope at
    ``h_pbl`` — a smooth, fully differentiable stand-in for the prescribed
    inversion flux, NOT that inversion-flux source term itself.
  * **Entrainment stability cap** ``K_ent ≤ (4/27)·κ·w_s(h)·h`` and a gradient floor
    ``∂θ_v/∂z ≥ 1e-4`` — guards (not tunables) that break the ``e_ratio`` linearity
    once the flux-matched K would exceed the peak mixed-layer K.
  * **Local-K coefficients + smooth blend**: the above-PBL local-Ri mixing is
    faithful YSU, but its specific Louis (1982) coefficients (shared
    ``_shared.louis_stability_functions``) and the sigmoid PBL/local blend are a
    re-tuned, smoothed closure.
  * **Smooth switches**: sigmoid PBL/local blend and a smooth first-crossing PBL-top
    detection replace hard index searches, for differentiability.
Non-behavioral pins: ``tests/atmosphere/hydrostatic/unit/test_ysu_faithful.py``.

References
----------
- Hong, S.-Y., Noh, Y., & Dudhia, J. (2006). A new vertical diffusion
  package with an explicit treatment of entrainment processes. Mon.
  Wea. Rev., 134, 2318-2341.
- Troen, I., & Mahrt, L. (1986). A simple model of the atmospheric
  boundary layer; sensitivity to surface evaporation. Boundary-Layer
  Meteorol., 37, 129-148.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.timestepping.tridiagonal import thomas_solve_batched
from legoesm.atmosphere.physics._shared import (
    virtual_temperature,
    mixing_length,
    buoyancy_coefficient,
    exner_function,
    louis_stability_functions,
    half_to_iface_zero_ends,
)
from legoesm.atmosphere.physics.turbulence.config import YSUConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import (
    first_crossing_height,
)
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    latent_enthalpy_correction,
    surface_moisture_flux,
    compute_surface_fluxes,
    surface_fluxes_at_lowest_level,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    diagnostic_heat_flux_full,
    implicit_vertical_diffusion,
)
from legoesm.atmosphere.physics.turbulence.holtslag_boville import (
    diffuse_theta_with_countergradient,
)


__physics_contract__ = {
    "summary": (
        "YSU (Yonsei University) PBL scheme. With surface_layer given: WRF "
        "v4.8.0 bl_ysu_run (clear-sky path), oracle-tested. Without it (every "
        "lane today): the legacy surrogate, NOT WRF YSU -- a convective "
        "mixed-layer K-profile with a countergradient nonlocal heat flux and "
        "a free-convective-limit PBL-top entrainment flux (w'th')_h = "
        "-e_ratio*(w'th')_0 (a surrogate for Hong06's shear-dependent w_m "
        "entrainment) applied as a flux-matched K at the inversion, "
        "all applied by implicit vertical diffusion with surface-flux BCs."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "T_sfc": "K", "q_sfc": "kg/kg", "rho": "kg/m^3", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m",
    },
    "sign_convention": (
        "Nonlocal down-gradient mixing: heat flux F = -Kh (dtheta/dz - "
        "gamma_c) with countergradient gamma_c >= 0 (upward heat transport in "
        "the convective BL, gated to unstable surface forcing). Km, Kh >= 0; "
        "an unstable surface heat flux (shflx > 0 upward) deepens h_pbl via a "
        "thermal-excess parcel. The prescribed (free-convective-limit) PBL-top "
        "entrainment flux (w'th')_h = -e_ratio*(w'th')_0 is NEGATIVE (downward) "
        "for unstable "
        "surface forcing — warm air entrained down across the inversion, "
        "warming the PBL top region — and is realized down-gradient via a "
        "flux-matched K_ent >= 0 at the inversion. shflx, lhflx positive "
        "UPWARD and injected as the lower boundary condition (a source), so "
        "the column is NOT closed. z increases upward; level index 0 is the "
        "top, -1 the surface."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Hong, Noh & Dudhia (2006), Mon. Wea. Rev. 134, 2318-2341; "
        "Troen & Mahrt (1986), Boundary-Layer Meteorol. 37, 129-148"
    ),
    "idealized_test": (
        "WRF path: tests/atmosphere/hydrostatic/unit/test_ysu_oracle.py "
        "(running-oracle match, column water/heat budgets). Legacy path: "
        "tests/atmosphere/hydrostatic/unit/test_atm_turb_gwd_validation.py "
        "and test_turbulence.py: an unstable surface heat flux deepens h_pbl "
        "(two-pass thermal excess); Km, Kh >= 0 with entrainment K localized "
        "at the PBL top; the PBL top is found by smooth first-crossing "
        "interpolation (differentiable, no hard index search)."
    ),
}


# --- PBL-top entrainment guards (Hong et al. 2006 flux-matched K) ----------
# Floor [K/m] on the inversion theta_v gradient in
# K_h,ent = e_ratio*(w'th')_0 / max(dtheta_v/dz, floor): below it (a well-mixed
# or unstably-stratified interface at GCM vertical resolution) the flux-matched
# K saturates and the cap below takes over.  Numerics guard, not a tunable.
_ENTRAIN_DTHDZ_FLOOR = 1.0e-4
# Cap fraction for the entrainment diffusivity: the interior K-profile shape
# (z/h)*(1 - z/h)^2 peaks at exactly 4/27 (at z = h/3, exact math), so
# kappa*w_s(h)*h*(4/27) is the peak mixed-layer diffusivity magnitude —
# the entrainment K may not exceed it (stability guard, not a tunable).
_KPROFILE_PEAK_FRAC = 4.0 / 27.0


# ===========================================================================
# WRF YSU: JAX port of WRF v4.8.0 phys/physics_mmm/bl_ysu.F90::bl_ysu_run
# (WRF 06d4240ae989cc3e50af412bb472df3d9048783c, physics_mmm submodule
# 550b5b494758ad2756394e93cb08b5709b8feb83).  Line numbers below refer to
# that bl_ysu.F90.  The oracle builds with RWORDSIZE=8 (-r8), so every literal
# is fp64 as written (h1 = 0.33333335, not 1/3).
# ===========================================================================
# --- bl_ysu_run parameters (l.141-155) ---
_XKZMINM = 0.1          # background momentum diffusivity xkzom [m2/s]
_XKZMINH = 0.01         # background heat/moisture diffusivity xkzoh [m2/s]
_XKZMAX = 1000.0        # diffusivity cap [m2/s]
_RIMIN = -100.0         # lower bound on z/L and on the local Ri
_RLAM = 30.0            # minimum asymptotic mixing length [m]
_PRMIN = 0.25           # Prandtl-number bounds
_PRMAX = 4.0
_BRCR_UB = 0.0          # critical bulk Ri, unstable/neutral passes
_BRCR_SB = 0.25         # critical bulk Ri, stable surface over land
_CORI = 1.0e-4          # Coriolis parameter in the surface Rossby number [1/s]
_AFAC = 6.8             # thermal-excess coefficients
_BFAC = 6.8
_PFAC = 2.0             # K-profile exponents (momentum/heat and moisture)
_PFAC_Q = 2.0
_PHIFAC = 8.0           # convective coefficient in the velocity scale
_SFCFRAC = 0.1          # surface-layer fraction of the PBL depth
_D1 = 0.02              # entrainment-layer thickness coefficients
_D2 = 0.05
_D3 = 0.001
_H1 = 0.33333335        # cube-root exponent as written in the oracle
_H2 = 0.6666667
_ZFMIN = 1.0e-8         # floors on z/L and on the K-profile factor
_APHI5 = 5.0            # Businger-Dyer stability-function coefficients
_APHI16 = 16.0
_TMIN = 1.0e-2          # minimum PBL-top jumps (theta, theta_v, wind)
_GAMCRT = 3.0           # caps on the heat / moisture countergradient [K], [kg/kg]
_GAMCRQ = 2.0e-3
# --- literals inside bl_ysu_run ---
_SPDK2_MIN = 1.0        # max(u^2+v^2, 1.) in the bulk Ri (l.629)
_WSPD1_EPS = 1.0e-9     # added to the lowest-level wind speed (l.551)
_ENT_BFX_COEFF = 0.15   # bfxpbl = -0.15*thv1/g*wm3/h (l.837)
_WM3_UST_COEFF = 5.0    # wm3 = wstar3 + 5*ust^3 (l.835)
_HGAMU_COEFF = 15.9     # momentum countergradient (l.694) and prfac2 (l.949)
_PRFAC2_COEFF = 4.0     # 1 + 4*karman*wstar3/ust3 (l.949)
_PRNUMFAC_COEFF = 3.0   # prnumfac = -3*(...)^2/h^2 (l.950)
_PRPBL = 1.0            # momentum/heat entrainment ratio prpbl (l.910)
_DELTA_MAX = 100.0      # entrainment-layer thickness cap [m] (l.924)
_ROSS_A = 0.16          # brcr over water: min(0.16*(1e-7*Ro)^-0.18, 0.3) (l.787)
_ROSS_SCALE = 1.0e-7
_ROSS_EXP = -0.18
_BRCR_WATER_MAX = 0.3
_ENTFAC_UNSET = 1.0e30  # entfac outside the entrainment layer (l.933)
_ENTFAC_MAX = 4.6       # entrainment layer: entfac < 4.6 (l.1059)
_SS_EPS = 1.0e-9        # added to the squared shear (l.989)
_RLAMDZ_FRAC = 0.1      # rlamdz = min(max(0.1*dz, rlam), 300, dz) (l.1004)
_RLAMDZ_MAX = 300.0
_KF_UNST = 8.0          # free-atmosphere unstable K (l.1012-1013)
_KF_UNST_M = 1.746
_KF_UNST_H = 1.286
_KF_STAB = 5.0          # free-atmosphere stable K and Prandtl number (l.1016-1017)
_KF_PR_STAB = 2.1
_WSCALEK_MIN = 0.001    # stable-surface velocity-scale floor (l.957)
_THX_OFFSET = 300.0     # theta offset of the heat solve (l.1045, l.1103)
_XLAND_WATER = 1.5      # water when xland - 1.5 >= 0 (l.783)


class YSUSurfaceLayer(NamedTuple):
    """Surface-layer state ``bl_ysu_run`` reads from WRF's surface-layer scheme.

    Every field has shape (ncol,).  Names and meanings are the oracle's
    argument names: ``psim``/``psih`` are what WRF passes under those names
    since 2012, the INTEGRATED stability functions ``ln(z1/z0) - psi_m`` and
    ``ln(z1/z0) - psi_h`` (sfclay's ``fm``/``fh``); YSU only uses them through
    ``z/L = br*psim^2/psih``.

    Fields
    ------
    br : surface-layer bulk Richardson number (sign sets stable/unstable).
    psim, psih : integrated stability functions (see above).
    wspd : surface-layer wind speed [m/s].
    znt : roughness length [m].
    u10, v10 : 10 m wind [m/s].
    xland : land/water flag (1 land, 2 water; water when ``xland >= 1.5``).
    """
    br: jax.Array
    psim: jax.Array
    psih: jax.Array
    wspd: jax.Array
    znt: jax.Array
    u10: jax.Array
    v10: jax.Array
    xland: jax.Array


class YSUOutput(NamedTuple):
    """Outputs of :func:`bl_ysu_run`, level index 0 = top (repo convention).

    ``exch_h``/``exch_m`` are the oracle's ``exch_hx``/``exch_mx``: entry k
    holds the diffusivity of the interface BELOW level k (the oracle's
    ``exch(k+1) = xkz(k)``), so the surface entry (index -1) is 0.
    ``kpbl`` is the oracle's 1-based level index counted from the surface.
    """
    du_dt: jax.Array
    dv_dt: jax.Array
    dT_dt: jax.Array
    dq_v_dt: jax.Array
    hpbl: jax.Array
    kpbl: jax.Array
    exch_h: jax.Array
    exch_m: jax.Array
    exch_q: jax.Array
    wstar: jax.Array
    delta: jax.Array
    pblflg: jax.Array


def _take(a, idx):
    """a[:, idx] per column."""
    return jnp.take_along_axis(a, idx[:, None], axis=1)[:, 0]


def bl_ysu_run(u, v, T, q_v, p_full, p_half, exner, dz, ust, hfx, qfx,
               sfc: YSUSurfaceLayer, dt, *, pbl_sharpness, region_sharpness,
               ri_sharpness) -> YSUOutput:
    """WRF v4.8.0 ``bl_ysu_run`` (clear-sky path), batched over columns.

    Ported path: ``ysu_topdown_pblmix=.false.``, ``f_qc=f_qi=.false.`` (so the
    cloud-top entrainment and in-cloud Ri branches, which need condensate, are
    not reached), ``nmix=0``, no BEP, ``ctopo``/``ctopo2``/``uox``/``vox``
    absent.  Constants are ``legoesm.constants`` (cp=c_pd, g, rd=R_d,
    ep1=R_v/R_d-1, karman=kappa_von_karman).

    Arrays are (ncol, nlev) with level 0 = TOP (``p_half`` is (ncol, nlev+1));
    the port flips to the oracle's bottom-up order internally.  ``dz`` is the
    layer thickness (the oracle's ``dz8w``), ``exner`` its ``pi2d``, ``ust`` /
    ``hfx`` [W/m2] / ``qfx`` [kg/m2/s] the surface friction velocity and
    fluxes, ``q_v`` is passed where the oracle reads ``qvx``.

    Smoothing (each reduces to the oracle's hard switch as its sharpness
    goes to infinity; tested at 1e300):

    * ``pbl_sharpness``: the first-crossing selection of the bulk-Ri scans
      (oracle: first level with ``brup > brcr``; an exact tie is not a
      crossing at any sharpness); the in-pair interpolation
      is the oracle's ``brint`` exactly.
    * ``region_sharpness``: the in-PBL (``k < kpbl``) vs above-PBL selection of
      the diffusivities and of the in-PBL nonlocal fluxes, as
      ``1 - sigmoid(s*(za(k)/hpbl - 1))`` (``k < kpbl`` iff ``za(k) < hpbl``
      when ``kpbl > 1``, except at equality, where ``k < kpbl`` itself is used
      at any sharpness).
    * ``ri_sharpness``: the ``ri < 0`` unstable/stable selection of the
      free-atmosphere diffusivity, as ``sigmoid(s*ri)``; each branch is
      evaluated at ``min(ri,0)`` / ``max(ri,0)``, its value at ``ri = 0``,
      where both branches equal ``dk``.

    Every other oracle switch is ported as a hard ``jnp.where``; operands of
    unselected branches are replaced by benign values only so their gradients
    stay finite (selected values are untouched).

    NOT RULE 5 COMPLIANT (opt-in kernel; smooth before any lane uses it,
    #2231): see the unsmoothed switches below.

    KNOWN DIFFERENCE (RULE 5, measured 2026-10-28 on the 23-column oracle
    fixture plus its 8-level upper-endpoint column, fp64): the oracle uses the
    hard switches; at the ``YSUConfig`` defaults (``pbl_smooth_sharpness`` 20,
    ``blend_pbl_sharpness`` 10, ``blend_ri_sharpness`` 100) the port differs
    from it per column by up to 48 % in PBL height (median 5 %), up to 95 % in
    the diffusivities (medians 6-9 %), and up to 89 % / 35 % / 44 % in the
    u / v / temperature tendencies (medians 6 % / 12 % / 4 %); one column's moisture
    tendency changes sign (200 %), and one column's ``kpbl`` differs.  Setting
    all three fields to 1e300 recovers the oracle to 1e-10.  The column water
    and heat budgets still close (residual < 1e-13 of the surface flux).  At an
    exact bulk-Ri tie (``brup == brcr``, e.g. a constant-theta_v layer) the
    default is DISCONTINUOUS: the tie gets weight 0 while any perturbation off
    it gets ~1/2.  The hard switches listed above as ``jnp.where`` (the
    PBL-top level ``kpbl`` and the flags and level picks derived from it) are
    not smoothed: the derivative ignores which level is the PBL top, and the
    output jumps where one of them flips.  Per-case gaps are stored in
    ``tests/atmosphere/hydrostatic/unit/test_ysu_oracle.py::DEFAULT_GAP``.
    """
    flip = lambda a: a[:, ::-1]  # noqa: E731  top-first -> oracle bottom-up
    ux, vx, tx, qx = flip(u), flip(v), flip(T), flip(q_v)
    p2d, pi2d, dz8w, p2di = flip(p_full), flip(exner), flip(dz), flip(p_half)
    g, cp, rd, karman = constants.g, constants.c_pd, constants.R_d, constants.kappa_von_karman
    ep1 = constants.R_v / constants.R_d - 1.0
    ncol, n = tx.shape
    dt2 = 2.0 * dt                       # l.560: implicit step over 2*dt
    rdt = 1.0 / dt2
    cont = cp / g

    # --- preliminaries (l.418-555) ---
    thx = tx / pi2d
    thvx = thx * (1.0 + ep1 * qx)
    rhox = p2di[:, 0] / (rd * tx[:, 0] * (1.0 + ep1 * qx[:, 0]))
    govrth = g / thx[:, 0]
    zq = jnp.concatenate([jnp.zeros_like(dz8w[:, :1]), jnp.cumsum(dz8w, axis=1)], axis=1)
    za = 0.5 * (zq[:, :-1] + zq[:, 1:])
    dele = p2di[:, :-1] - p2di[:, 1:]
    dza = jnp.concatenate([za[:, :1], za[:, 1:] - za[:, :-1]], axis=1)
    zl1 = za[:, 0]
    wspd1 = jnp.sqrt(ux[:, 0] * ux[:, 0] + vx[:, 0] * vx[:, 0]) + _WSPD1_EPS
    sflux = hfx / rhox / cp + qfx / rhox * ep1 * thx[:, 0]
    sfcflg = ~(sfc.br > 0.0)
    thermal1 = thvx[:, 0]

    spdk2 = jnp.maximum(ux[:, 1:] ** 2 + vx[:, 1:] ** 2, _SPDK2_MIN)
    gza = g * za[:, 1:] / thvx[:, :1]

    def scan(thermal, brcr):
        """One bulk-Ri pass (l.624-648): first level k >= 2 with brup > brcr,
        hpbl interpolated with the oracle's brint; kpbl = n if none."""
        brcr = jnp.broadcast_to(brcr, (ncol,))[:, None]
        brup = (thvx[:, 1:] - thermal[:, None]) * gza / spdk2
        crossed = brup > brcr
        kpbl = jnp.where(jnp.any(crossed, axis=1),
                         jnp.argmax(crossed, axis=1) + 2, n)
        vals = jnp.concatenate([sfc.br[:, None], brup], axis=1)
        lo, hi = vals[:, :-1], vals[:, 1:]
        mid = (lo < brcr) & (hi > brcr)
        brint = jnp.where(lo >= brcr, 0.0, jnp.where(
            hi <= brcr, 1.0, (brcr - lo) / jnp.where(mid, hi - lo, 1.0)))
        h_pair = za[:, :-1] + brint * (za[:, 1:] - za[:, :-1])
        # an exact tie (brup == brcr, e.g. a layer of constant theta_v against
        # brcr = 0) is NOT a crossing, as the oracle's strict '>' has it; the
        # sigmoid alone would give it weight 1/2 at any sharpness
        w_c = jnp.where(brup == brcr, 0.0, jax.nn.sigmoid(pbl_sharpness * (brup - brcr)))
        not_yet = jnp.cumprod(jnp.concatenate(
            [jnp.ones((ncol, 1), tx.dtype), 1.0 - w_c[:, :-1]], axis=1), axis=1)
        # no crossing: the oracle interpolates in the top pair (kpbl = n)
        h = (jnp.sum(not_yet * w_c * h_pair, axis=1)
             + jnp.prod(1.0 - w_c, axis=1) * h_pair[:, -1])
        return h, jnp.where(h < zq[:, 1], 1, kpbl)

    # --- first guess (l.617-648) ---
    h_guess, kpbl = scan(thermal1, _BRCR_UB)
    pblflg = kpbl > 1

    # --- surface-layer scales from the first guess (l.650-678) ---
    zol1 = jnp.maximum(sfc.br * sfc.psim * sfc.psim / sfc.psih, _RIMIN)
    zol1 = jnp.where(sfcflg, jnp.minimum(zol1, -_ZFMIN), jnp.maximum(zol1, _ZFMIN))
    hol1 = zol1 * h_guess / zl1 * _SFCFRAC
    base_u = jnp.where(sfcflg, 1.0 - _APHI16 * hol1, 1.0)
    phim = jnp.where(sfcflg, base_u ** (-0.25), 1.0 + _APHI5 * hol1)
    phih = jnp.where(sfcflg, base_u ** (-0.5), 1.0 + _APHI5 * hol1)
    wstar3 = jnp.where(sfcflg, govrth * jnp.maximum(sflux, 0.0) * h_guess, 0.0)
    wstar = jnp.where(wstar3 > 0.0, jnp.where(wstar3 > 0.0, wstar3, 1.0) ** _H1, 0.0)
    ust3 = ust ** 3.0
    wscale = (ust3 + _PHIFAC * karman * wstar3 * 0.5) ** _H1
    wscale = jnp.maximum(jnp.minimum(wscale, ust * _APHI16), ust / _APHI5)

    # --- unstable surface: thermal excess, countergradients (l.683-700) ---
    unst = sfcflg & (sflux > 0.0)
    gamfac = _BFAC / rhox / wscale
    hgamt = jnp.minimum(gamfac * hfx / cp, _GAMCRT)
    hgamq = jnp.minimum(gamfac * qfx, _GAMCRQ)
    vpert = (hgamt + ep1 * thx[:, 0] * hgamq) / _BFAC * _AFAC
    thermal2 = thermal1 + jnp.maximum(vpert, 0.0) * jnp.minimum(
        za[:, 0] / (_SFCFRAC * h_guess), 1.0)
    hgamt = jnp.where(unst, jnp.maximum(hgamt, 0.0), 0.0)
    brint_u = (-_HGAMU_COEFF * ust * ust / jnp.where(unst, sfc.wspd, 1.0)
               * wstar3 / (wscale ** 4.0))
    hgamu = jnp.where(unst, brint_u * ux[:, 0], 0.0)
    hgamv = jnp.where(unst, brint_u * vx[:, 0], 0.0)
    pblflg = pblflg & unst

    # --- second pass with the thermal (l.704-768, topdown off) ---
    h2, k2 = scan(thermal2, _BRCR_UB)
    hpbl = jnp.where(pblflg, h2, h_guess)
    kpbl = jnp.where(pblflg, k2, kpbl)
    pblflg = pblflg & (k2 > 1)

    # --- stable boundary layer (l.770-822) ---
    stab = (~sfcflg) & (hpbl < zq[:, 1])
    water = (sfc.xland - _XLAND_WATER) >= 0.0
    s10 = sfc.u10 * sfc.u10 + sfc.v10 * sfc.v10
    ok = stab & water & (s10 > 0.0)
    ross = jnp.sqrt(jnp.where(ok, s10, 1.0)) / (_CORI * sfc.znt)
    # zero 10 m wind: (1e-7*ross)**(-0.18) is +inf and min() returns 0.3
    brcr_sbro = jnp.minimum(jnp.where(
        ok, _ROSS_A * (_ROSS_SCALE * jnp.where(ok, ross, 1.0)) ** _ROSS_EXP, jnp.inf),
        _BRCR_WATER_MAX)
    h3, k3 = scan(thermal1, jnp.where(water, brcr_sbro, _BRCR_SB))
    hpbl = jnp.where(stab, h3, hpbl)
    kpbl = jnp.where(stab, k3, kpbl)
    pblflg = pblflg & ~(stab & (k3 <= 1))

    # --- entrainment parameters, clear-sky branch (l.827-926) ---
    kb = jnp.clip(kpbl - 2, 0, n - 2)          # 0-based level k = kpbl-1
    wm3 = wstar3 + _WM3_UST_COEFF * ust3
    wm2 = wm3 ** _H2
    bfxpbl = -_ENT_BFX_COEFF * thvx[:, 0] / g * wm3 / hpbl
    dthvx = jnp.maximum(_take(thvx, kb + 1) - _take(thvx, kb), _TMIN)
    we = jnp.maximum(bfxpbl / dthvx, -jnp.sqrt(wm2))
    dthx = jnp.maximum(_take(thx, kb + 1) - _take(thx, kb), _TMIN)
    dqx = jnp.minimum(_take(qx, kb + 1) - _take(qx, kb), 0.0)
    hfxpbl = we * dthx
    qfxpbl = we * dqx

    def mom_ent_flux(d):
        return jnp.where(d > _TMIN, jnp.maximum(_PRPBL * we * d, -ust * ust),
                         jnp.where(d < -_TMIN, jnp.minimum(_PRPBL * we * d, ust * ust),
                                   0.0))
    ufxpbl = mom_ent_flux(_take(ux, kb + 1) - _take(ux, kb))
    vfxpbl = mom_ent_flux(_take(vx, kb + 1) - _take(vx, kb))
    delb = govrth * _D3 * hpbl
    delta = jnp.where(pblflg, jnp.minimum(_D1 * hpbl + _D2 * wm2 / delb, _DELTA_MAX), 0.0)

    # interface k (1..n-1, Fortran) between levels k and k+1 sits at zq(k+1)
    zqi = zq[:, 1:n]
    kk = jnp.arange(1, n)[None, :]
    above = kk >= kpbl[:, None]
    entfac = jnp.where(pblflg[:, None] & above, ((zqi - hpbl[:, None]) / jnp.where(
        pblflg, delta, 1.0)[:, None]) ** 2.0, _ENTFAC_UNSET)

    # --- diffusivities below the PBL top, k < kpbl (l.937-980) ---
    sf = sfcflg[:, None]
    zfac = jnp.clip(1.0 - (zqi - zl1[:, None]) / jnp.where(
        kpbl > 1, hpbl - zl1, 1.0)[:, None], _ZFMIN, 1.0)
    zfacent = (1.0 - zfac) ** 3.0
    wscalek = (ust3[:, None] + _PHIFAC * karman * wstar3[:, None] * (1.0 - zfac)) ** _H1
    # wscalek2 = (phifac*karman*wstar3_2*zfac)**h1 is 0: wstar3_2 is set only on
    # the cloud-top branch, which needs condensate (not ported).
    prfac = jnp.where(sfcflg, _BFAC * karman * _SFCFRAC, 0.0)[:, None]
    prfac2 = jnp.where(sfcflg, _HGAMU_COEFF * wstar3 / ust3 / (
        1.0 + _PRFAC2_COEFF * karman * wstar3 / ust3), 0.0)[:, None]
    prnumfac = jnp.where(sf, -_PRNUMFAC_COEFF * jnp.maximum(
        zqi - _SFCFRAC * hpbl[:, None], 0.0) ** 2.0 / hpbl[:, None] ** 2.0, 0.0)
    phim8z = jnp.where(sf, 1.0, 1.0 + _APHI5 * zol1[:, None] * zqi / zl1[:, None])
    wscalek = jnp.where(sf, wscalek, jnp.maximum(ust[:, None] / phim8z, _WSCALEK_MIN))
    prnum0 = jnp.clip((phih / phim)[:, None] + prfac, _PRMIN, _PRMAX)
    km_p = wscalek * karman * zqi * zfac ** _PFAC
    kq_p = km_p / (1.0 + (prnum0 - 1.0) * jnp.exp(prnumfac)) * zfac ** (_PFAC_Q - _PFAC)
    prnum0 = prnum0 / (1.0 + prfac2 * karman * _SFCFRAC)
    kh_p = km_p / (1.0 + (prnum0 - 1.0) * jnp.exp(prnumfac))
    km_p = jnp.minimum(km_p + _XKZMINM, _XKZMAX)
    kh_p = jnp.minimum(kh_p + _XKZMINH, _XKZMAX)
    kq_p = jnp.minimum(kq_p + _XKZMINH, _XKZMAX)

    # --- free atmosphere, k >= kpbl (l.984-1028; dry ri, imvdif in-cloud
    #     branch needs condensate) ---
    du = ux[:, 1:] - ux[:, :-1]
    dv = vx[:, 1:] - vx[:, :-1]
    dzi = dza[:, 1:]
    ss = (du * du + dv * dv) / (dzi * dzi) + _SS_EPS
    govrthv = g / (0.5 * (thvx[:, 1:] + thvx[:, :-1]))
    ri = govrthv * (thvx[:, 1:] - thvx[:, :-1]) / (ss * dzi)
    zk = karman * zqi
    rlamdz = jnp.minimum(dzi, jnp.minimum(jnp.maximum(_RLAMDZ_FRAC * dzi, _RLAM), _RLAMDZ_MAX))
    dk = (zk * rlamdz / (rlamdz + zk)) ** 2 * jnp.sqrt(ss)
    neg = ri < 0.0
    ri_u = jnp.where(neg, jnp.maximum(ri, _RIMIN), -1.0)
    sri = jnp.sqrt(-ri_u)
    km_u = jnp.where(neg, dk * (1.0 + _KF_UNST * (-ri_u) / (1.0 + _KF_UNST_M * sri)), dk)
    kh_u = jnp.where(neg, dk * (1.0 + _KF_UNST * (-ri_u) / (1.0 + _KF_UNST_H * sri)), dk)
    ri_s = jnp.maximum(ri, 0.0)
    kh_s = dk / (1.0 + _KF_STAB * ri_s) ** 2
    km_s = kh_s * jnp.minimum(1.0 + _KF_PR_STAB * ri_s, _PRMAX)
    w_st = jax.nn.sigmoid(ri_sharpness * ri)
    km_f = jnp.minimum((1.0 - w_st) * km_u + w_st * km_s + _XKZMINM, _XKZMAX)
    kh_f = jnp.minimum((1.0 - w_st) * kh_u + w_st * kh_s + _XKZMINH, _XKZMAX)

    # --- entrainment layer above the PBL top (l.1059-1063, l.1339-1342) ---
    ent = pblflg[:, None] & above & (entfac < _ENTFAC_MAX)
    arg = -we[:, None] * _take(dza, jnp.clip(kpbl - 1, 0, n - 1))[:, None] * jnp.exp(-entfac)
    kh_e = jnp.clip(jnp.sqrt(jnp.where(ent, arg * kh_f, 1.0)), _XKZMINH, _XKZMAX)
    km_e = jnp.clip(jnp.sqrt(_PRPBL * kh_e * km_f), _XKZMINM, _XKZMAX)

    # --- PBL / above-PBL selection ---
    # za(k) == hpbl (hpbl exactly on level kpbl-1 after an exact tie, or on
    # level kpbl when brint rounds to 1): the sigmoid would give 1/2 at any
    # sharpness, so equality takes the oracle's k < kpbl
    zr = za[:, :-1] / hpbl[:, None] - 1.0
    wp = jnp.where(kpbl > 1, 1.0, 0.0)[:, None] * jnp.where(
        zr == 0.0, jnp.where(kk < kpbl[:, None], 1.0, 0.0),
        1.0 - jax.nn.sigmoid(region_sharpness * zr))
    kh_a = jnp.where(ent, kh_e, kh_f)
    Kh = wp * kh_p + (1.0 - wp) * kh_a
    Kq = wp * kq_p + (1.0 - wp) * kh_a
    Km = wp * km_p + (1.0 - wp) * jnp.where(ent, km_e, km_f)
    fw = jnp.where(pblflg, 1.0, 0.0)[:, None] * wp     # in-PBL nonlocal fluxes

    # --- implicit solves over 2*dt (l.1031-1405) ---
    dsig = p2d[:, :-1] - p2d[:, 1:]
    rdz = 1.0 / dzi
    dtodsd = dt2 / dele[:, :-1]
    dtodsu = dt2 / dele[:, 1:]
    zero = jnp.zeros((ncol, 1), tx.dtype)
    h_ = hpbl[:, None]

    def solve(base, K, dsdz, extra0):
        """Rows: f(k) += dtodsd*dsdz(k), f(k+1) -= dtodsu*dsdz(k); matrix
        au = -dtodsd*dsig*K*rdz^2, al = -dtodsu*(...), ad = 1-al(k-1)-au(k)."""
        rhs = base + jnp.concatenate([dtodsd * dsdz, zero], axis=1) \
            - jnp.concatenate([zero, dtodsu * dsdz], axis=1)
        dsdz2 = dsig * K * rdz * rdz
        au = jnp.concatenate([-dtodsd * dsdz2, zero], axis=1)
        al = jnp.concatenate([zero, -dtodsu * dsdz2], axis=1)
        diag = (1.0 - al - au).at[:, 0].add(extra0)
        return thomas_solve_batched(al, diag, au, rhs)

    th_base = (thx - _THX_OFFSET).at[:, 0].add(hfx / cont / dele[:, 0] * dt2)
    dsdzt = fw * dsig * Kh * rdz * (-hgamt[:, None] / h_ - hfxpbl[:, None] * zfacent / Kh)
    th_new = solve(th_base, Kh, dsdzt, 0.0)
    ttend = (th_new - thx + _THX_OFFSET) * rdt * pi2d

    q_base = qx.at[:, 0].add(qfx * g / dele[:, 0] * dt2)
    dsdzq = fw * dsig * Kq * rdz * (-qfxpbl[:, None] * zfacent / Kq)
    qtend = (solve(q_base, Kq, dsdzq, 0.0) - qx) * rdt

    fric = ust ** 2 / wspd1 * rhox * g / dele[:, 0] * dt2 * (wspd1 / sfc.wspd) ** 2
    dsdzu = fw * dsig * Km * rdz * (-hgamu[:, None] / h_ - ufxpbl[:, None] * zfacent / Km)
    dsdzv = fw * dsig * Km * rdz * (-hgamv[:, None] / h_ - vfxpbl[:, None] * zfacent / Km)
    utend = (solve(ux, Km, dsdzu, fric) - ux) * rdt
    vtend = (solve(vx, Km, dsdzv, fric) - vx) * rdt

    exch = lambda K: flip(jnp.concatenate([zero, K], axis=1))  # noqa: E731
    return YSUOutput(
        du_dt=flip(utend), dv_dt=flip(vtend), dT_dt=flip(ttend), dq_v_dt=flip(qtend),
        hpbl=hpbl, kpbl=kpbl, exch_h=exch(Kh), exch_m=exch(Km), exch_q=exch(Kq),
        wstar=wstar, delta=delta, pblflg=pblflg)


# Legacy-surrogate config fields with no counterpart in WRF YSU.  On the WRF
# path they are inert, so a non-default value is refused rather than ignored.
_SURROGATE_ONLY_FIELDS = (
    "l_mix_max", "Pr_t", "entrainment_ratio", "entrainment_width_frac",
    "Ri_crit", "louis_b", "louis_c", "louis_d", "countergrad_coeff",
    "ws_conv_coeff", "sfc_excess_zfrac",
)


def _refuse_surrogate_fields(config: YSUConfig) -> None:
    defaults = YSUConfig._field_defaults
    bad = [f for f in _SURROGATE_ONLY_FIELDS
           if not (isinstance(getattr(config, f), (int, float))
                   and getattr(config, f) == defaults[f])]
    if bad:
        raise ValueError(
            f"YSUConfig fields {bad} belong to the legacy YSU surrogate and have "
            "no counterpart in WRF YSU; they are inert when surface_layer is "
            "given, so they must stay at their defaults (concrete Python numbers).")


def _ysu_wrf_turbulence(u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc,
                        q_sfc, rho, dt, config, surface_layer):
    """WRF YSU path of :func:`ysu_turbulence`.

    The surface friction velocity and fluxes come from the scheme's existing
    surface-layer call (unchanged from the legacy path, including its 1e-4 m/s
    friction-velocity floor); ``hfx`` is the sensible heat flux and ``qfx`` the
    surface water flux, as the oracle reads them.  The oracle's own surface
    drag (implicit, from ``ust``) replaces the explicit stress, and its layer
    geometry comes from ``z_half`` (full-level heights are the oracle's layer
    midpoints, so ``z_full`` is not read).
    """
    _refuse_surrogate_fields(config)
    _tx, _ty, shflx, lhflx, ustar = surface_fluxes_at_lowest_level(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface, z_full[:, -1] - z_half[:, -1])
    ustar = jnp.clip(ustar, 1e-4, None)  # coeff-ok: u* floor [m/s] (legacy path's)
    evap = surface_moisture_flux(config.surface, lhflx, T_sfc)
    o = bl_ysu_run(
        u, v, T, q_v, p_full, p_half, exner_function(p_full),
        z_half[:, :-1] - z_half[:, 1:], ustar, shflx, evap, surface_layer, dt,
        pbl_sharpness=config.pbl_smooth_sharpness,
        region_sharpness=config.blend_pbl_sharpness,
        ri_sharpness=config.blend_ri_sharpness)
    return TurbulenceOutput(
        du_dt=o.du_dt, dv_dt=o.dv_dt, dT_dt=o.dT_dt, dq_v_dt=o.dq_v_dt,
        Km=o.exch_m, Kh=o.exch_h, shflx=shflx, lhflx=lhflx, evap_sfc=evap,
        ustar=ustar, h_pbl=o.hpbl,
        Kh_iface=half_to_iface_zero_ends(o.exch_h[:, :-1]))


def ysu_turbulence(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: YSUConfig,
    surface_layer: YSUSurfaceLayer | None = None,
) -> TurbulenceOutput:
    """Compute turbulence tendencies using YSU PBL scheme.

    With ``surface_layer`` (a :class:`YSUSurfaceLayer`) this runs WRF YSU
    (:func:`bl_ysu_run`); without it (every lane today) the legacy surrogate,
    which is NOT WRF YSU (see the module docstring).

    Parameters
    ----------
    u, v : jax.Array
        Wind components at full levels [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    z_full : jax.Array
        Height at full levels [m], shape (ncol, nlev).
    z_half : jax.Array
        Height at half levels [m], shape (ncol, nlev+1).
    T_sfc : jax.Array
        Surface temperature [K], shape (ncol,).
    q_sfc : jax.Array
        Surface saturation mixing ratio [kg/kg], shape (ncol,).
    rho : jax.Array
        Air density at full levels [kg/m^3], shape (ncol, nlev).
    dt : float
        Time step [s].
    config : YSUConfig
    surface_layer : YSUSurfaceLayer or None
        WRF surface-layer state; selects the WRF YSU path.

    Returns
    -------
    TurbulenceOutput
    """
    if surface_layer is not None:
        return _ysu_wrf_turbulence(u, v, T, q_v, p_full, p_half, z_full, z_half,
                                   T_sfc, q_sfc, rho, dt, config, surface_layer)
    ncol, nlev = T.shape

    # ----- Half-level gradients (same as Louis/HB) -----
    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    dz_half = jnp.clip(dz_half, 1.0, None)

    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2 = du_dz ** 2 + dv_dz ** 2 + 1e-10
    S = jnp.sqrt(S2)

    # Virtual potential temperature via the canonical inverse-Exner helper
    # (exner_pref = 1/Π = (p_ref/p)^κ; the helper applies the same 1 Pa floor).
    exner_pref = 1.0 / exner_function(p_full)
    theta_v = virtual_temperature(T, q_v) * exner_pref

    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    # N² = (g/θ_v)·∂θ_v/∂z (Brunt-Väisälä) via the shared buoyancy coefficient.
    N2 = buoyancy_coefficient(jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz
    Ri = N2 / S2  # (ncol, nlev-1)

    # ----- Surface fluxes -----
    tau_x, tau_y, shflx, lhflx, ustar = surface_fluxes_at_lowest_level(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface, z_full[:, -1] - z_half[:, -1])
    ustar = jnp.clip(ustar, 1e-4, None)  # coeff-ok: u* floor [m/s]

    # ----- PBL height via smooth bulk-Ri (Troen-Mahrt 1986 / Hong et al. 2006)
    # The bulk-Richardson PBL height carries an UNSTABLE surface-excess parcel
    # temperature θ_T = b·(w'θ')_0 / w_s so a convective surface heat flux
    # deepens the diagnosed PBL.  Without θ_T the numerator is purely the
    # resolved θ_v gradient, so a well-mixed (neutral) column pins h_pbl at the
    # floor REGARDLESS of surface heating — the K-profile then normalizes z by
    # the stale h_pbl (z_norm = z/h_pbl, (1-z_norm)² suppression) and stays
    # shallow even as w* grows (codex atm-turb-gwd finding A; Hong et al. 2006
    # eq. for h via Ri_b with the thermal excess).  The excess needs w_s, which
    # needs h_pbl, so we use the standard two-pass: first guess h_pbl with no
    # excess, then refine with θ_T(h_pbl_guess).
    theta_v_sfc = theta_v[:, -1]
    z_sfc = z_full[:, -1:]
    u_sfc = u[:, -1:]
    v_sfc = v[:, -1:]
    dz_from_sfc = jnp.abs(z_full - z_sfc) + 1.0
    dtheta_v_bulk = theta_v - theta_v_sfc[:, None]
    # Wind shear from surface (not absolute wind)
    dV2 = (u - u_sfc) ** 2 + (v - v_sfc) ** 2 + 1e-4  # coeff-ok: wind-shear floor [m^2/s^2]

    # Kinematic surface heat flux and the well-mixed θ_v scale (shared by both
    # passes and the convective velocity scale below).
    wtheta_sfc = shflx / (rho[:, -1] * constants.c_pd)  # kinematic (ncol,)
    theta_bar = jnp.mean(theta_v, axis=1)  # (ncol,)
    g_over_thbar = buoyancy_coefficient(jnp.clip(theta_bar, 1.0, None))

    def _bulk_pbl_height(excess_theta):
        # excess_theta : (ncol,) surface-parcel virtual-θ excess [K], >=0.
        # h_pbl = LOWEST height where the bulk Richardson number first reaches
        # Ri_crit, found by smooth first-crossing interpolation (surface-up).
        # The earlier ``sigma*(1-sigma)`` centre-of-mass weighting is fragile:
        # at a sharp capping inversion the crossing falls BETWEEN two grid
        # levels, so EVERY level saturates the sigmoid (w~0) and the weighted
        # mean collapses to the simple domain-mean height (codex round-2: the
        # excess-parcel fix exposed this, yielding a 7 km PBL above a 2 km
        # inversion).  A first-crossing interpolation is robust to an
        # arbitrarily sharp inversion and is the standard bulk-Ri PBL diagnosis
        # (the shared ``first_crossing_height`` kernel, also used by
        # Holtslag-Boville).  The 100 m floor is this scheme's h_pbl floor.
        Ri_b = g_over_thbar[:, None] * (
            (dtheta_v_bulk - excess_theta[:, None]) * dz_from_sfc / dV2
        )
        return jnp.clip(
            first_crossing_height(
                Ri_b, z_full, config.Ri_crit, config.pbl_smooth_sharpness,
            ),
            100.0, None,
        )

    # Pass 1: no excess.
    h_pbl_guess = _bulk_pbl_height(jnp.zeros_like(theta_v_sfc))

    # Mixed-layer velocity scale w_s evaluated at the surface layer (z=0.1·h),
    # the standard Troen-Mahrt level for the excess parcel.  w* from the
    # first-guess depth; w_s = (u*³ + c·κ·w*³·0.1)^{1/3}.
    # (g/θ_bar)·(w'θ')_0·h via the shared buoyancy coefficient.
    buoy_guess = (
        buoyancy_coefficient(jnp.clip(theta_bar, 1.0, None))
        * jnp.maximum(wtheta_sfc, 0.0) * h_pbl_guess
    )
    w_star_guess = jnp.cbrt(jnp.maximum(buoy_guess, 1e-20))
    w_s_sfc = jnp.cbrt(
        ustar ** 3 + config.ws_conv_coeff * constants.kappa_von_karman
        * w_star_guess ** 3 * config.sfc_excess_zfrac
    )
    # Thermal excess θ_T = b·(w'θ')_0 / w_s, only for unstable (kbfs>0).
    excess_theta = config.countergrad_coeff * jnp.maximum(wtheta_sfc, 0.0) / jnp.clip(
        w_s_sfc, 1e-6, None
    )

    # Pass 2: refined PBL height with the unstable surface excess.
    h_pbl = _bulk_pbl_height(excess_theta)

    # ----- Convective velocity scale w* -----
    # w* = (g · h · (w'θ')_sfc / θ_bar)^{1/3}, defined only for unstable
    # (positive surface buoyancy flux).  Used by the mixed-layer velocity
    # scale w_s (below), the PBL-top entrainment flux, and the nonlocal
    # countergradient.  Recomputed with the refined (deeper) h_pbl.
    # w* = ((g/θ_bar)·h·(w'θ')_sfc)^{1/3} via the shared buoyancy coefficient.
    buoyancy_flux = (
        buoyancy_coefficient(jnp.clip(theta_bar, 1.0, None))
        * jnp.maximum(wtheta_sfc, 0.0) * h_pbl
    )
    w_star = jnp.cbrt(jnp.maximum(buoyancy_flux, 1e-20))  # (ncol,)

    # ----- K-profile -----
    z_half_inner = 0.5 * (z_full[:, :-1] + z_full[:, 1:])
    z_norm = z_half_inner / h_pbl[:, None]
    z_norm_clip = jnp.clip(z_norm, 0.0, 1.0)
    # Hong et al. (2006) mixed-layer velocity scale: blends mechanical (u*)
    # and convective (w*) scaling, w_s = (u*³ + c·κ·w*³·z/h)^{1/3}.  The
    # previous code used bare u*, so the convective mixed layer carried NO
    # convective enhancement and was systematically under-mixed.
    w_s = jnp.cbrt(
        ustar[:, None] ** 3
        + config.ws_conv_coeff * constants.kappa_von_karman
        * w_star[:, None] ** 3 * z_norm_clip
    )
    Km_profile = (
        constants.kappa_von_karman * w_s * z_half_inner * (1.0 - z_norm_clip) ** 2
    )

    # Local Ri-based Km above PBL: shared Blackadar mixing-length helper
    # (same expression as Louis/TKE/CLUBB-lite/EDMF/Smagorinsky).
    l_mix = mixing_length(z_half_inner, config.l_mix_max)
    # Louis (1982) stability functions via the SHARED helper
    # (physics._shared.louis_stability_functions — identical b/c/d coefficient
    # split and sigmoid stable/unstable blend as louis.py; YSU consumes only
    # the momentum function f_m).  Coefficients config.louis_b /
    # config.louis_c / config.louis_d and config.blend_ri_sharpness.
    f_m, _ = louis_stability_functions(
        Ri, l_mix, dz_half,
        config.louis_b, config.louis_c, config.louis_d,
        config.blend_ri_sharpness,
    )
    Km_local = l_mix ** 2 * S * f_m

    # Smooth blend from K-profile to local
    blend_pbl = jax.nn.sigmoid(config.blend_pbl_sharpness * (z_norm - 1.0))

    # ----- Entrainment flux at PBL top (Hong et al. 2006, explicit closure) --
    # YSU's defining innovation over Holtslag-Boville: a PRESCRIBED entrainment
    # heat flux at the inversion, tied to the surface flux
    #     (w'θ')_h = −e_ratio · (w'θ')_0,   e_ratio = config.entrainment_ratio.
    # Sign convention (flux positive-UP, z up): unstable surface forcing
    # (w'θ')_0 > 0 gives (w'θ')_h < 0 — the entrainment flux at h is DOWNWARD
    # (warm free-tropospheric air entrained down across the inversion, warming
    # the PBL-top region).  This module builds K profiles and then runs
    # implicit vertical diffusion, so the prescribed flux enters as a
    # FLUX-MATCHED diffusivity at the PBL-top interfaces:
    #     K_h,ent = −(w'θ')_h / max(∂θ_v/∂z, floor)
    #             =  e_ratio·(w'θ')_0 / max(∂θ_v/∂z, floor)  ≥ 0,
    # so the down-gradient flux −K_h,ent·∂θ_v/∂z at the inversion (where
    # ∂θ_v/∂z ≫ floor) equals the prescribed (w'θ')_h EXACTLY — pinned to the
    # surface flux, independent of the resolved gradient (the previous Gaussian
    # K_ent = c·w*·h blob produced an ordinary down-gradient flux whose
    # magnitude scaled with the local gradient, NOT Hong06's ratio closure).
    # A Gaussian envelope centered at h (width config.entrainment_width_frac·h)
    # localizes the entrainment K to the inversion; moisture entrains
    # down-gradient with the same K (Hong06 applies w_e to all scalars).
    width = config.entrainment_width_frac * h_pbl[:, None]  # (ncol, 1)
    envelope = jnp.exp(
        -((z_half_inner - h_pbl[:, None]) / jnp.clip(width, 1.0, None)) ** 2
    )
    # Inversion θ_v gradient, floored (see _ENTRAIN_DTHDZ_FLOOR): below the
    # floor the flux-matched K saturates and the stability cap takes over.
    dthdz_inv = jnp.maximum(dtheta_v_dz, _ENTRAIN_DTHDZ_FLOOR)
    K_ent_h = (
        config.entrainment_ratio
        * jnp.maximum(wtheta_sfc, 0.0)[:, None] / dthdz_inv
    )
    # Stability cap: entrainment K may not exceed the peak mixed-layer
    # K-profile magnitude κ·w_s(h)·h·(4/27) (see _KPROFILE_PEAK_FRAC), with
    # w_s(h) the mixed-layer velocity scale evaluated at z = h.
    w_s_h = jnp.cbrt(
        ustar ** 3 + config.ws_conv_coeff * constants.kappa_von_karman * w_star ** 3
    )
    K_ent_cap = (
        _KPROFILE_PEAK_FRAC * constants.kappa_von_karman * w_s_h * h_pbl
    )[:, None]
    K_ent_h = jnp.minimum(K_ent_h, K_ent_cap) * envelope

    # Combined: profile inside PBL, local above, entrainment added everywhere
    # (the flux-matched entrainment K is self-localizing around the PBL top).
    # Momentum analog per Hong06 (Prandtl number at h, here the scheme Pr_t):
    # Km carries Pr_t·K_ent_h so heat receives exactly the flux-matched
    # K_ent_h after the Kh = Km/Pr_t division.
    Km_half = (
        (1.0 - blend_pbl) * Km_profile + blend_pbl * Km_local
        + config.Pr_t * K_ent_h
    )
    Kh_half = Km_half / config.Pr_t

    # Interpolate to full levels for diagnostics — single concat per
    # field instead of the previous ``zeros + 3 .at[].set`` triple
    # scatter (lowers to one HLO op).
    Km_interior = 0.5 * (Km_half[:, :-1] + Km_half[:, 1:])
    Km_full = jnp.concatenate(
        [Km_half[:, :1], Km_interior, Km_half[:, -1:]], axis=1,
    )
    Kh_interior = 0.5 * (Kh_half[:, :-1] + Kh_half[:, 1:])
    Kh_full = jnp.concatenate(
        [Kh_half[:, :1], Kh_interior, Kh_half[:, -1:]], axis=1,
    )

    # Layer thicknesses
    dz_layer = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz_layer = jnp.clip(dz_layer, 1.0, None)

    sflx_u = tau_x
    sflx_v = tau_y
    sflx_q = surface_moisture_flux(config.surface, lhflx, T_sfc)
    # Heat BC carries the latent enthalpy correction (water at L(T) vs L_v).
    sflx_T = (shflx + latent_enthalpy_correction(lhflx, sflx_q)) / constants.c_pd

    # ----- Nonlocal countergradient (Troen-Mahrt 1986 / Hong et al. 2006) -----
    # γ_c = b·(w'θ')_0 / (w_s0·h)  [K/m] (Troen & Mahrt 1986; Hong et al.
    # 2006), YSU's defining nonlocal upward heat transport in the convective
    # BL.  w_s0 is the MIXED-LAYER velocity scale evaluated at the top of the
    # surface layer, w_s0 = w_s(0.1h) = (u*³ + c·κ·w*³·0.1)^{1/3} — the SAME
    # ``w_s_sfc`` already used for the surface thermal-excess parcel
    # θ_T = b·(w'θ')_0/w_s0 above, so γ_c = θ_T/h BY CONSTRUCTION (the papers
    # tie both to w_s0; ``excess_theta`` carries the same clip floor).  The
    # previous code divided by the PURE convective w*, which overestimates γ_c
    # in windy, weakly-convective columns (u* ≫ w*, where w_s0 → u* but
    # w* → 0) and breaks the γ_c = θ_T/h identity.  Gated to unstable surface
    # forcing via max(w'θ', 0) inside θ_T (zero for neutral/stable).  The
    # previous YSU lumped the entire countergradient into the lowest-level
    # surface BC (an enhanced surface flux with a single column-mean Kh),
    # depositing all the nonlocal heating in the surface-adjacent layer
    # instead of distributing it through the mixed layer.  Per Troen-Mahrt/
    # Hong the nonlocal term enters the heat equation as the modified flux
    # F = −Kh·(∂θ/∂z − γ_c) whose vertical divergence redistributes heat
    # across the BL — apply it that way via the shared HB countergradient
    # diffusion, with γ_c as a half-level field gated to inside the PBL
    # (1 − blend_pbl ≈ 1 below h_pbl, ≈ 0 above).
    counter_grad = excess_theta / h_pbl  # (ncol,) [K/m] = b·(w'θ')_0/(w_s0·h)
    gamma_theta_half = counter_grad[:, None] * (1.0 - blend_pbl)  # (ncol, nlev-1) [K/m]

    # Implicit vertical diffusion.  Heat in θ-space (dry-adiabat neutral),
    # with the nonlocal countergradient applied as a distributed flux
    # divergence (shared with Holtslag-Boville).
    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = diffuse_theta_with_countergradient(
        T, Kh_half, rho, dz_layer, dz_half, p_full, dt, sflx_T, gamma_theta_half,
    )
    q_new = implicit_vertical_diffusion(q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q)

    # Q1 diagnostic (LES-suite): NONLOCAL heat flux F = -Kh·(∂θ/∂z − γ) with the
    # scheme's own countergradient γ (gamma_theta_half), θ = T·exner_pref [=T/Π].
    # Pure diagnostic — does NOT feed tendencies.
    wtheta_flux = diagnostic_heat_flux_full(
        T * exner_pref, dz_half, Kh_half, gamma_theta_half,
    )

    return TurbulenceOutput(
        du_dt=(u_new - u) / dt,
        dv_dt=(v_new - v) / dt,
        dT_dt=(T_new - T) / dt,
        dq_v_dt=(q_new - q_v) / dt,
        Km=Km_full,
        Kh=Kh_full,
        shflx=shflx,
        lhflx=lhflx, evap_sfc=sflx_q,
        ustar=ustar,
        h_pbl=h_pbl,
        wtheta_flux=wtheta_flux,
        Kh_iface=half_to_iface_zero_ends(Kh_half),
    )
