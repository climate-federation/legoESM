"""SCM columns for the ANALYTIC dry-PBL LES cases (GABLS1, CBL, Ekman).

The gSAM-deck cases get their column from a file; these three are defined by a
handful of numbers in their LES driver, so the case definition lives here. Each
entry is transcribed from the driver that produced the stored reference, NOT
from the same-named case in ``scripts/matrix/scm/`` -- those do not match:

* ``scripts/matrix/scm/wangara.py`` is Wangara Day 33 (theta = 277 K uniform,
  a diurnal cosine surface flux, f_c = -8.26e-5, an easterly geostrophic wind,
  moist) while ``run_spectral_cbl.py`` runs Nieuwstadt CBL_N91 (theta = 300 K
  under an 800 m inversion, a CONSTANT flux, f_c = 0, no geostrophic wind,
  dry). They share only a ``--case-label``.
* ``scripts/matrix/scm/gabls1.py`` drives a time-varying ``T_s``; the stored
  GABLS1 reference took the prescribed-FLUX branch, because ``--cooling-rate``
  defaults to 0. Verified independently from the archive: its column heat
  budget over 9 h is -162.000 K m exactly, i.e. a constant -0.005 K m/s.

Every case here prescribes a kinematic surface heat FLUX rather than a surface
temperature, which keeps ``SCMForcing.T_s`` out of the picture entirely -- a
time-varying prescribed surface temperature cannot be traced (its validator
materialises with NumPy), so a ``T_s``-driven case could not be gradient-tuned
at all without changing that validator first.

That flux is CONSTANT for every case except Wangara Day 33, whose defining
feature is its diurnal cycle. This file previously said the flux was constant
everywhere and built it that way, so Wangara's SCM arm ran its 09:00 value for
all 8 h -- through the 13:00 maximum and the afternoon decay -- and took a
uniform -5.5 m/s geostrophic wind where the LES uses a sheared profile. The
LES driver reads both from ``legoesm.atmosphere.forcing.wangara_day33``; the
spec now names the same functions, so the two sides really cannot be driven
differently. A case declares non-steady forcing with
``surface_theta_flux_fn`` / ``geostrophic_u_fn``.

Two heights, deliberately distinct. The SCM column spans the LES DOMAIN
(``les_lz_m``) so its lid matches the LES lid; SCORING stops at
``les_domain_top_m``, which is the sponge base where the LES sponges
(``run_spectral_sbl.py`` and ``run_spectral_les.py``, top 25 %) and the lid
itself where it does not (``run_spectral_cbl.py`` has no sponge). Conflating
them put a lid 400 m below CBL's 1600 m domain and truncated its entrainment
layer -- a scoring cutoff must never become a physical boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.forcing.scm.sam_case_scm import (
    heights_from_pressure,
    hydrostatic_pressure_from_theta,
)
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing
from legoesm.atmosphere.forcing import wangara_day33
from legoesm.atmosphere.physics._shared import exner_function
from legoesm.grids.vertical import create_sigma_coordinate

__all__ = [
    "AnalyticSCMCase",
    "AnalyticSCMCaseSpec",
    "ANALYTIC_SCM_CASES",
    "load_analytic_scm_case",
]

_P_S_PA = 1.0e5          # the analytic drivers are Boussinesq; 1000 hPa column


@dataclass(frozen=True)
class AnalyticSCMCaseSpec:
    """Everything needed to rebuild the LES case's column, from its driver."""
    les_driver: str
    theta0_K: float
    inversion_z_m: float | None      # None = no inversion (neutral)
    lapse_above_K_m: float
    inversion_width_m: float         # tanh smoothing; 0 = sharp step
    sfc_theta_flux_K_m_s: float      # constant prescribed kinematic flux
    u_geo_m_s: float
    v_geo_m_s: float
    f_c: float
    les_z0_m: float
    # The LES domain depth: the SCM column spans THIS, so the physical lid
    # matches the LES lid.
    les_lz_m: float
    # Where SCORING stops. Equal to les_lz_m when the LES has no sponge; the
    # sponge base otherwise. Conflating the two put a lid 400 m below CBL's
    # 1600 m domain and truncated its entrainment layer.
    les_domain_top_m: float
    default_dt_s: float
    scored: tuple[str, ...]
    note: str

    # --- non-steady forcing, for the cases that have it ---------------------
    # A case whose surface flux varies in TIME, or whose geostrophic wind
    # varies with HEIGHT, declares the function here and the loader uses it in
    # place of the scalar above. Wangara Day 33 has both, and without them its
    # SCM arm is a constant-flux, uniform-wind CBL that merely starts from the
    # same 277 K sounding -- which is what it was: the LES driver pulls the
    # diurnal cycle and the sheared geostrophic wind from
    # `legoesm.atmosphere.forcing.wangara_day33` while this loader hard-coded
    # `w_th_s = lambda _t: spec.sfc_theta_flux_K_m_s`, so the two sides WERE
    # driven differently despite the claim that they could not be.
    #
    # `surface_theta_flux_fn(t_seconds) -> K m/s` is called with the SCM's own
    # clock, which starts at zero; `t_start_s` is added first, so a case that
    # keys its forcing to local time gets the right hour.
    surface_theta_flux_fn: Any = None
    geostrophic_u_fn: Any = None            # (z_m) -> m/s
    t_start_s: float = 0.0

    # WHERE THE WIND STARTS, which the LES driver decides and this column must
    # copy. It is declared per case rather than inferred because the two live
    # in different files and only the driver knows.
    #
    #   "geostrophic" — u = the geostrophic wind, v = 0. `run_spectral_les.py`
    #     --ekman initialises `u = broadcast_to(u_tar)` with v at noise level,
    #     so its SCM twin starts balanced too.
    #   "rest" — u = v = 0. `run_spectral_cbl.py` (cbl AND wangara) builds
    #     `u = zeros, v = zeros` and lets Coriolis and friction spin the wind
    #     up from nothing.
    #
    # This is not a detail that averages out. Wangara's inertial period is
    # 2*pi/|f| = 21.1 h and the benchmark is 8 h, so an initial-wind mismatch
    # does NOT decay within the run -- it rotates. Starting the column at a
    # uniform -5.5 m/s against an LES starting from rest measured u +2.4 and
    # v -4.2 m/s of bias at the analysis window, against an LES profile whose
    # own spread is 0.27 and 0.23 m/s: normalised errors of 9 and 18, IDENTICAL
    # across all nine closures, which is the signature of the arm rather than
    # the closures. The surface-flux boundary condition was a separate defect
    # in the same arm and fixing it did not move these two terms at all.
    initial_wind: str = "geostrophic"

    # The tuner reads these off the deck spec; analytic cases prescribe their
    # surface flux, so there is no bulk exchange coefficient to carry.
    @property
    def bulk_ch(self) -> None:
        return None

    @property
    def bulk_ce(self) -> None:
        return None

    @property
    def les_n_c_m3(self) -> None:
        """Analytic cases are dry, so no droplet concentration applies."""
        return None


ANALYTIC_SCM_CASES: dict[str, AnalyticSCMCaseSpec] = {
    "gabls1": AnalyticSCMCaseSpec(
        les_driver="run_spectral_sbl.py",
        theta0_K=265.0, inversion_z_m=100.0, lapse_above_K_m=0.01,
        inversion_width_m=25.0,
        sfc_theta_flux_K_m_s=-0.005,       # stable: cooling the surface layer
        u_geo_m_s=8.0, v_geo_m_s=0.0, f_c=1.39e-4,
        les_z0_m=0.1, les_lz_m=400.0, les_domain_top_m=300.0,  # sponge > 300 m
        default_dt_s=10.0,
        scored=("theta", "u", "v"),        # stable BL + low-level jet
        note="GABLS1 stable boundary layer; constant -0.005 K m/s surface "
             "flux (the archived reference's branch), 9 h.",
    ),
    "cbl": AnalyticSCMCaseSpec(
        les_driver="run_spectral_cbl.py",
        theta0_K=300.0, inversion_z_m=800.0, lapse_above_K_m=0.008,
        inversion_width_m=0.0,             # driver uses a sharp jnp.where
        sfc_theta_flux_K_m_s=0.06,
        # run_spectral_cbl.py builds u = v = 0. Numerically identical to
        # "geostrophic" here (u_geo = v_geo = 0), declared so the driver's
        # choice is recorded rather than coincidental.
        initial_wind="rest",
        u_geo_m_s=0.0, v_geo_m_s=0.0, f_c=0.0,
        # run_spectral_cbl.py has NO sponge, so scoring runs to the lid.
        les_z0_m=0.1, les_lz_m=1600.0, les_domain_top_m=1600.0,
        default_dt_s=10.0,
        # No rotation and no geostrophic wind, so u and v stay ~0 and their
        # spread is ~0: scoring them would divide by the floor.
        scored=("theta",),
        note="Nieuwstadt CBL_N91 dry convective boundary layer; constant "
             "+0.06 K m/s surface flux, no Coriolis, 4 h. NOT Wangara Day 33 "
             "despite the LES driver's --case-label.",
    ),
    "wangara": AnalyticSCMCaseSpec(
        les_driver="run_spectral_cbl.py --case wangara",
        # The capping inversion is `run_spectral_cbl.py`'s DEFAULT zi0/gamma:
        # the wangara branch overrides only theta0, f_cor, t_start_s and Q0, so
        # `th = where(z > 800, 277 + 0.008*(z-800), 277)` is what it built.
        # MEASURED off frame 0 of the stored reference rather than read off the
        # argparse defaults: theta = 277.000 +- 0.001 K below 800 m and a
        # 0.00800 K/m lapse fitted above 900 m. Declaring no inversion here
        # left the column uniform at 277 K, ~1.8 K colder than the LES in the
        # domain mean, which is the whole of the -2.1 K theta bias that
        # survived the surface-flux and initial-wind fixes.
        theta0_K=277.0, inversion_z_m=800.0, lapse_above_K_m=0.008,
        inversion_width_m=0.0,   # a sharp `where`, no tanh: matches the driver
        # DIURNAL in the real case: the constant here is the flux at the 09:00
        # start, and the SCM arm overrides it with the shared time-dependent
        # forcing (see wangara_day33). It is carried so the spec stays
        # comparable with its siblings, not because the case is steady.
        sfc_theta_flux_K_m_s=0.0897,
        u_geo_m_s=-5.5, v_geo_m_s=0.0, f_c=-8.2634e-5,
        les_z0_m=0.01, les_lz_m=2000.0, les_domain_top_m=1700.0,
        default_dt_s=10.0,
        # Dry: the LES driver carries no moisture, so scoring q_v would compare
        # a moist SCM against a dry reference. theta plus the Ekman-like wind
        # structure under the height-dependent geostrophic forcing is what this
        # case constrains.
        scored=("theta", "u", "v"),
        # The SAME module the LES driver reads, so the two sides really cannot
        # be driven differently. Wired as functions rather than as the scalars
        # above because both quantities genuinely vary.
        surface_theta_flux_fn=wangara_day33.surface_theta_flux,
        geostrophic_u_fn=wangara_day33.geostrophic_u,
        t_start_s=wangara_day33.T_START_S,
        # THE SAME run_spectral_cbl.py build: the LES starts from REST and
        # spins the wind up under Coriolis. Starting this column at a uniform
        # -5.5 m/s instead left u +2.4 and v -4.2 m/s biased at 8 h, 9x and
        # 18x the LES profile's own spread, on every closure alike.
        initial_wind="rest",
        note="Wangara Day 33 convective boundary layer (Clarke et al. 1971), "
             "DRY: diurnal surface heat flux peaking at 13:00 local, "
             "southern-hemisphere Coriolis, height-dependent easterly "
             "geostrophic wind, 09:00 start. NOT the Nieuwstadt CBL_N91 case, "
             "which is the separate 'cbl' entry.",
    ),
    "ekman": AnalyticSCMCaseSpec(
        les_driver="run_spectral_les.py --ekman",
        theta0_K=290.0, inversion_z_m=None, lapse_above_K_m=0.0,
        inversion_width_m=0.0,
        sfc_theta_flux_K_m_s=0.0,          # neutral
        u_geo_m_s=10.0, v_geo_m_s=0.0, f_c=1.0e-4,
        les_z0_m=0.1, les_lz_m=1000.0, les_domain_top_m=750.0,  # sponge > 750 m
        default_dt_s=10.0,
        # Neutral by construction: theta is constant, so its spread is ~0 and
        # a normalized theta score is meaningless. The Ekman spiral IS the
        # physics here.
        scored=("u", "v"),
        note="Neutral Ekman layer; geostrophic 10 m/s, f = 1e-4 (inertial "
             "period 17.5 h), no surface heat flux.",
    ),
}


def _theta_profile(spec: AnalyticSCMCaseSpec, z: np.ndarray) -> np.ndarray:
    """theta(z), transcribed from the case's LES driver."""
    z = np.asarray(z, dtype=np.float64)
    if spec.inversion_z_m is None:
        return np.full_like(z, spec.theta0_K)
    zi, gam, di = spec.inversion_z_m, spec.lapse_above_K_m, spec.inversion_width_m
    if di > 0.0:
        # run_spectral_sbl.py's tanh-smoothed inversion, written in the
        # log-cosh form the driver uses so the two are the same curve.
        return spec.theta0_K + 0.5 * gam * (
            (z - zi) + di * np.log(np.cosh((z - zi) / di)) + di * np.log(2.0)
        )
    return np.where(z > zi, spec.theta0_K + gam * (z - zi), spec.theta0_K)


@dataclass(frozen=True)
class AnalyticSCMCase:
    """Same surface as ``SAMSCMCase`` so the tuner consumes either."""
    name: str
    spec: AnalyticSCMCaseSpec
    nlev: int
    dt: float
    sigma_top: float
    p_s: float
    T_profile: np.ndarray
    q_v_profile: np.ndarray
    u_profile: np.ndarray
    v_profile: np.ndarray
    z_full: np.ndarray
    p_full: np.ndarray
    forcing: SCMForcing
    rho_sfc: float
    case_dir: str = "<analytic>"

    @property
    def latitude_deg(self) -> float:
        """Latitude whose Coriolis parameter is the case's f_c."""
        from legoesm import constants
        s = float(np.clip(self.spec.f_c / (2.0 * constants.Omega), -1.0, 1.0))
        return float(np.rad2deg(np.arcsin(s)))

    @property
    def les_domain_top_m(self) -> float:
        return self.spec.les_domain_top_m

    def les_mask(self) -> np.ndarray:
        return np.asarray(self.z_full) <= self.spec.les_domain_top_m

    def scm_kwargs(self) -> dict[str, Any]:
        return dict(
            nlev=self.nlev, dt=self.dt,
            T_profile=jnp.asarray(self.T_profile),
            q_v_profile=jnp.asarray(self.q_v_profile),
            u=jnp.asarray(self.u_profile), v=jnp.asarray(self.v_profile),
            p_s=self.p_s, latitude_deg=self.latitude_deg,
            sigma_top=self.sigma_top, forcing=self.forcing,
        )

    def create_scm(self, *, physics_config, **overrides) -> SingleColumnModel:
        kwargs = self.scm_kwargs()
        kwargs.update(overrides)
        return SingleColumnModel.create(physics_config=physics_config, **kwargs)


def load_analytic_scm_case(case: str, *, nlev: int = 48,
                           dt: float | None = None) -> AnalyticSCMCase:
    """Build the SCM column for an analytic dry-PBL LES case."""
    if case not in ANALYTIC_SCM_CASES:
        raise ValueError(
            f"Unknown analytic SCM case {case!r}; choose from "
            f"{sorted(ANALYTIC_SCM_CASES)}."
        )
    spec = ANALYTIC_SCM_CASES[case]
    if nlev < 2:
        raise ValueError(f"nlev must be >= 2, got {nlev}")
    dt = float(spec.default_dt_s if dt is None else dt)

    # Hydrostatic p(z) from the analytic theta, via the SHARED mapping the deck
    # bridge uses. Dry cases, so theta_v == theta.
    z_top = spec.les_lz_m
    z_ref = np.linspace(0.0, z_top, 512)
    z_aux, p_aux = hydrostatic_pressure_from_theta(
        z_ref, _theta_profile(spec, z_ref), _P_S_PA,
    )
    sigma_top = float(
        np.interp(spec.les_lz_m, z_aux[::-1], p_aux[::-1]) / _P_S_PA
    )
    if not 0.0 < sigma_top < 1.0:
        raise ValueError(f"derived sigma_top={sigma_top} outside (0, 1)")

    sigma = create_sigma_coordinate(nlev, sigma_top=sigma_top,
                                    dtype=jnp.float64)
    p_full = np.asarray(sigma.sigma_full, dtype=np.float64) * _P_S_PA
    z_full = heights_from_pressure(z_aux, p_aux, p_full)

    exner = np.asarray(exner_function(jnp.asarray(p_full)), dtype=np.float64)
    theta = _theta_profile(spec, z_full)
    T_profile = theta * exner
    q_v_profile = np.zeros(nlev)                 # every analytic case is dry
    # Initial wind: copy the LES driver's, which is per case (see the
    # `initial_wind` field). Dispatch raises on an unknown value rather than
    # falling through to a default -- a typo here is a silently different
    # experiment, not an error.
    if spec.initial_wind == "rest":
        u_profile = np.zeros(nlev)
        v_profile = np.zeros(nlev)
    elif spec.initial_wind == "geostrophic":
        # The PROFILE when the case declares one. Using the scalar here was
        # the same defect the forcing had: a case with a sheared geostrophic
        # wind would start uniform and be relaxed toward a sheared target.
        if spec.geostrophic_u_fn is not None:
            u_profile = np.asarray(spec.geostrophic_u_fn(z_full),
                                   dtype=np.float64)
        else:
            u_profile = np.full(nlev, spec.u_geo_m_s)
        v_profile = np.full(nlev, spec.v_geo_m_s)
    else:
        raise ValueError(
            f"case {case!r}: initial_wind={spec.initial_wind!r} is not one of "
            "'geostrophic' / 'rest'."
        )

    from legoesm import constants
    rho_sfc = _P_S_PA / (constants.R_d * float(T_profile[-1]))

    # Geostrophic wind: the scalar unless the case declares a PROFILE. Wangara
    # Day 33's easterly jet is sheared (-5.5 m/s at the surface, kinking at
    # 1 km), and a uniform -5.5 gives the SCM a different momentum forcing from
    # the LES it is scored against.
    if spec.geostrophic_u_fn is not None:
        u_geo = jnp.asarray(spec.geostrophic_u_fn(z_full), dtype=jnp.float64)
    else:
        u_geo = jnp.full(nlev, spec.u_geo_m_s)
    v_geo = jnp.full(nlev, spec.v_geo_m_s)

    # Surface heat flux: the scalar unless the case declares a TIME FUNCTION.
    # The scalar is evaluated at the run's first instant for a diurnal case, so
    # using it there holds the 09:00 flux for the whole 8 h -- straight through
    # the 13:00 maximum and the afternoon decay.
    if spec.surface_theta_flux_fn is not None:
        flux_fn = spec.surface_theta_flux_fn
        t0 = float(spec.t_start_s)

        def _w_th_s(t):
            return jnp.asarray(flux_fn(jnp.asarray(t) + t0),
                               dtype=jnp.float64)
    else:
        flux = jnp.asarray(spec.sfc_theta_flux_K_m_s)

        def _w_th_s(_t):
            return flux                       # CONSTANT, stays concrete

    forcing = SCMForcing(
        f_c=float(spec.f_c),
        u_geo=(lambda _t: u_geo) if spec.f_c != 0.0 else None,
        v_geo=(lambda _t: v_geo) if spec.f_c != 0.0 else None,
        prescribe="fluxes",
        w_th_s=_w_th_s,
    )

    return AnalyticSCMCase(
        name=case, spec=spec, nlev=nlev, dt=dt, sigma_top=sigma_top,
        p_s=_P_S_PA, T_profile=T_profile, q_v_profile=q_v_profile,
        u_profile=u_profile, v_profile=v_profile, z_full=z_full,
        p_full=p_full, forcing=forcing, rho_sfc=rho_sfc,
    )
