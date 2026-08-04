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

Every case here therefore uses a CONSTANT prescribed kinematic surface heat
flux, which also keeps ``SCMForcing.T_s`` out of the picture entirely -- a
time-varying prescribed surface temperature cannot be traced (its validator
materialises with NumPy), so a ``T_s``-driven case could not be gradient-tuned
at all without changing that validator first.

The comparable region stops at 0.75*Lz wherever the LES applies its Rayleigh
sponge (``run_spectral_sbl.py`` and ``run_spectral_les.py`` both do); above
that the LES is relaxed toward a reference and represents nothing.
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
from legoesm.atmosphere.physics._shared import exner_function
from legoesm.grids.vertical import create_sigma_coordinate

__all__ = [
    "AnalyticSCMCase",
    "AnalyticSCMCaseSpec",
    "ANALYTIC_SCM_CASES",
    "load_analytic_scm_case",
]

_P_S_PA = 1.0e5          # the analytic drivers are Boussinesq; 1000 hPa column
_Z_TOP_PAD = 1.25        # build the column a little above the scored region


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
    les_domain_top_m: float          # 0.75*Lz where the LES sponges
    default_dt_s: float
    scored: tuple[str, ...]
    note: str

    # The tuner reads these off the deck spec; analytic cases prescribe their
    # surface flux, so there is no bulk exchange coefficient to carry.
    @property
    def bulk_ch(self) -> None:
        return None

    @property
    def bulk_ce(self) -> None:
        return None


ANALYTIC_SCM_CASES: dict[str, AnalyticSCMCaseSpec] = {
    "gabls1": AnalyticSCMCaseSpec(
        les_driver="run_spectral_sbl.py",
        theta0_K=265.0, inversion_z_m=100.0, lapse_above_K_m=0.01,
        inversion_width_m=25.0,
        sfc_theta_flux_K_m_s=-0.005,       # stable: cooling the surface layer
        u_geo_m_s=8.0, v_geo_m_s=0.0, f_c=1.39e-4,
        les_z0_m=0.1, les_domain_top_m=300.0,     # 0.75 * 400 m (sponge above)
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
        u_geo_m_s=0.0, v_geo_m_s=0.0, f_c=0.0,
        les_z0_m=0.1, les_domain_top_m=1200.0,    # 0.75 * 1600 m
        default_dt_s=10.0,
        # No rotation and no geostrophic wind, so u and v stay ~0 and their
        # spread is ~0: scoring them would divide by the floor.
        scored=("theta",),
        note="Nieuwstadt CBL_N91 dry convective boundary layer; constant "
             "+0.06 K m/s surface flux, no Coriolis, 4 h. NOT Wangara Day 33 "
             "despite the LES driver's --case-label.",
    ),
    "ekman": AnalyticSCMCaseSpec(
        les_driver="run_spectral_les.py --ekman",
        theta0_K=290.0, inversion_z_m=None, lapse_above_K_m=0.0,
        inversion_width_m=0.0,
        sfc_theta_flux_K_m_s=0.0,          # neutral
        u_geo_m_s=10.0, v_geo_m_s=0.0, f_c=1.0e-4,
        les_z0_m=0.1, les_domain_top_m=750.0,     # 0.75 * 1000 m
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
    z_top = spec.les_domain_top_m * _Z_TOP_PAD
    z_ref = np.linspace(0.0, z_top, 512)
    z_aux, p_aux = hydrostatic_pressure_from_theta(
        z_ref, _theta_profile(spec, z_ref), _P_S_PA,
    )
    sigma_top = float(
        np.interp(spec.les_domain_top_m, z_aux[::-1], p_aux[::-1]) / _P_S_PA
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
    u_profile = np.full(nlev, spec.u_geo_m_s)
    v_profile = np.full(nlev, spec.v_geo_m_s)

    from legoesm import constants
    rho_sfc = _P_S_PA / (constants.R_d * float(T_profile[-1]))

    u_geo = jnp.full(nlev, spec.u_geo_m_s)
    v_geo = jnp.full(nlev, spec.v_geo_m_s)
    flux = jnp.asarray(spec.sfc_theta_flux_K_m_s)
    forcing = SCMForcing(
        f_c=float(spec.f_c),
        u_geo=(lambda _t: u_geo) if spec.f_c != 0.0 else None,
        v_geo=(lambda _t: v_geo) if spec.f_c != 0.0 else None,
        prescribe="fluxes",
        # CONSTANT, so it stays concrete under tracing.
        w_th_s=lambda _t: flux,
    )

    return AnalyticSCMCase(
        name=case, spec=spec, nlev=nlev, dt=dt, sigma_top=sigma_top,
        p_s=_P_S_PA, T_profile=T_profile, q_v_profile=q_v_profile,
        u_profile=u_profile, v_profile=v_profile, z_full=z_full,
        p_full=p_full, forcing=forcing, rho_sfc=rho_sfc,
    )
