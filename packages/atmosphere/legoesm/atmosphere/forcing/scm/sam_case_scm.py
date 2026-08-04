"""Build an SCM column from the SAM/gSAM case decks the LES cases already read.

The LES drivers (``run_bomex_les.py``, ``run_rico_les.py``,
``run_dycoms_les.py``) initialise from a gSAM deck via
:mod:`legoesm.atmosphere.forcing.sam_case_forcing`.  This module maps the SAME
deck onto a :class:`~legoesm.atmosphere.forcing.scm.scm.SingleColumnModel`, so
an SCM run and its LES reference share one source of truth for the initial
sounding, the large-scale forcing and the surface boundary condition.  That is
what makes an LES-vs-SCM comparison a controlled one, and what makes the
comparison across turbulence closures controlled: every scheme is handed a
byte-identical column and forcing, and only ``PhysicsConfig.turbulence`` differs.

The DEPHY loader (:mod:`legoesm.atmosphere.forcing.scm.dephy_scm`) plays the
same role for DEPHY-format NetCDF cases; this is its gSAM-deck sibling, and it
deliberately mirrors that module's ``load_* -> case.create_scm(...)`` shape.

Two conversions in here are easy to get backwards, so both are stated
explicitly and both are covered by tests:

**Absolute-T vs potential-T tendencies.**  The deck's ``tls`` column is an
ABSOLUTE-temperature tendency ``dT/dt`` (``sam_case_forcing.SAMForcing.T_ls``
says so), while :attr:`SCMForcing.theta_adv` is a genuine POTENTIAL-temperature
tendency — ``compute_forcing_tendencies`` multiplies it by the Exner function
before adding it to ``dT/dt``.  So the deck value is divided by Exner here,
exactly as ``plane_large_scale_forcing.make_plane_ls_forcing_from_sam_case``
does for the LES side.

**The surface heat flux is NOT a theta flux on the SCM side.**  ``w_th_s`` is
added straight into ``dT_dt`` by ``compute_forcing_tendencies`` with no Exner
factor (``scm_forcing.py``, the ``prescribe="fluxes"`` branch), unlike
``theta_adv`` and the subsidence term which are both scaled by Exner.  It is
therefore a kinematic ABSOLUTE-TEMPERATURE flux, and the deck's SHF converts as
``SHF / (rho_sfc * c_pd)`` with NO Exner divide.  The LES wants the same
physical flux expressed as a POTENTIAL-temperature flux, so its conversion
(``run_bomex_les.py``) carries the extra ``/ Exner_sfc``.  Copying the LES
number into ``w_th_s`` would bias the SCM surface heat flux by a factor of
Exner.  Both models receive the same W/m^2; only the variable each one
prognoses differs.

Vertical layout: the SCM is a sigma-coordinate column indexed TOP-TO-BOTTOM,
while every deck array is bottom-to-top in height.  Everything returned by this
module is already in SCM (top-to-bottom) order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.forcing.sam_case_forcing import (
    interp_forcing_to_levels,
    interp_sounding_to_levels,
    read_sam_lsf,
    read_sam_sfc,
    read_sam_snd,
    resolve_sam_case_dir,
    surface_at_day,
)
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing
from legoesm.atmosphere.physics._shared import (
    exner_function,
    exner_to_pressure,
    virtual_temperature,
)
from legoesm.grids.vertical import (
    create_sigma_coordinate,
    create_stretched_height_coordinate,
)

from legoesm import constants

__all__ = [
    "SAMSCMCase",
    "SAM_SCM_CASES",
    "load_sam_scm_case",
    "surface_kinematic_temperature_flux",
    "surface_kinematic_moisture_flux",
]

# Levels of the auxiliary height coordinate used to reconstruct the deck's
# missing pressure column. Fine enough that the SCM's p->z inversion does not
# depend on it (12-35 m over these 1.5-4.5 km cases); it is thrown away
# immediately afterwards.
#
# MUST stay well under ~308: create_stretched_height_coordinate brackets its
# bisection at r_hi = 10.0 and evaluates r_hi**n_levels, so any n_levels above
# ~308 raises OverflowError from 10**n exceeding float64 range, before the
# grid is ever built.
_AUX_LEVELS = 128


@dataclass(frozen=True)
class SAMSCMCaseSpec:
    """Per-case metadata that is NOT in the deck itself."""
    gsam_dir: str
    latitude_deg: float
    les_domain_top_m: float
    default_dt_s: float
    # Surface mode MUST mirror what the case's LES driver actually does, not
    # what the deck happens to contain. RICO's sfc file carries H=15 and
    # LE=115 W/m2 but run_rico_les.py IGNORES them and computes interactive
    # bulk fluxes over the fixed SST (SFC_FLX_FXD=.false.), so a "nonzero flux
    # in the deck => prescribe it" heuristic gives RICO the wrong boundary.
    surface_mode: str
    # Bulk exchange coefficients the case's LES actually uses, for
    # surface_mode="T_s". None means the SCM default stands. RICO's LES uses
    # the van Zanten (2011) C_H=0.001094 / C_Q=0.001133 while the SCM default
    # is a single Ch=0.0015, a 32-37% error that the closures would be tuned
    # to compensate for.
    bulk_ch: float | None
    bulk_ce: float | None
    # Aerodynamic roughness the case's LES wall model uses (its --z0). The SCM
    # derives its neutral drag from this so both models see the same log law.
    les_z0_m: float
    note: str


# Only cases with a legoESM LES driver are registered: the whole point of this
# module is a matched LES reference, so a case with no LES is not admissible
# here even though its deck may be cached on disk.
SAM_SCM_CASES: dict[str, SAMSCMCaseSpec] = {
    "bomex": SAMSCMCaseSpec(
        gsam_dir="BOMEX", latitude_deg=15.0, les_domain_top_m=3000.0,
        default_dt_s=60.0, surface_mode="fluxes",
        bulk_ch=None, bulk_ce=None, les_z0_m=1.0e-4,
        note="Siebesma et al. 2003 shallow non-precipitating trade cumulus; "
             "prescribed surface fluxes.",
    ),
    "rico": SAMSCMCaseSpec(
        gsam_dir="RICO", latitude_deg=18.0, les_domain_top_m=4000.0,
        default_dt_s=60.0, surface_mode="T_s",
        # run_rico_les.py _C_H / _C_Q (van Zanten et al. 2011, at 20 m).
        bulk_ch=0.001094, bulk_ce=0.001133, les_z0_m=1.0e-4,
        note="van Zanten et al. 2011 precipitating trade cumulus; interactive "
             "bulk fluxes over a fixed SST.",
    ),
    "dycoms": SAMSCMCaseSpec(
        gsam_dir="DYCOMS_RF01", latitude_deg=31.5, les_domain_top_m=1500.0,
        default_dt_s=30.0, surface_mode="fluxes",
        bulk_ch=None, bulk_ce=None, les_z0_m=1.0e-4,
        note="Stevens et al. 2005 RF01 nocturnal stratocumulus; prescribed "
             "surface fluxes.",
    ),
}


def surface_kinematic_temperature_flux(shf_w_m2: float, rho_sfc: float) -> float:
    """Deck SHF [W/m^2] -> kinematic ABSOLUTE-temperature flux [K m/s].

    This is the form ``SCMForcing.w_th_s`` consumes: ``compute_forcing_tendencies``
    adds it to ``dT_dt`` directly, with no Exner factor. See the module
    docstring — the LES equivalent carries an extra ``/ Exner_sfc`` because a
    height-coordinate LES prognoses theta.

    Sign convention: positive = UPWARD (surface heating the atmosphere), which
    is the deck's convention and the convention of the lowest-cell tendency
    ``+w_th_s / dz`` in ``compute_forcing_tendencies``.
    """
    if not np.isfinite(rho_sfc) or rho_sfc <= 0.0:
        raise ValueError(f"rho_sfc must be finite and positive, got {rho_sfc!r}")
    return float(shf_w_m2) / (float(rho_sfc) * constants.c_pd)


def surface_kinematic_moisture_flux(lhf_w_m2: float, rho_sfc: float) -> float:
    """Deck LHF [W/m^2] -> kinematic moisture flux [(kg/kg) m/s].

    Positive = UPWARD (surface moistening the atmosphere).
    """
    if not np.isfinite(rho_sfc) or rho_sfc <= 0.0:
        raise ValueError(f"rho_sfc must be finite and positive, got {rho_sfc!r}")
    return float(lhf_w_m2) / (float(rho_sfc) * constants.L_v)


def _const_profile_fn(values: np.ndarray):
    """Time-independent (nlev,) profile callable, closed over a device array."""
    arr = jnp.asarray(values)
    return lambda _t: arr


def _interp_profile_fn(days_s: np.ndarray, per_time: np.ndarray):
    """Time-interpolating (nlev,) profile callable, pure JAX.

    ``per_time`` is ``(n_time, nlev)``. Kept in JAX (``jnp.interp``, which
    clamps outside the range) rather than re-reading the deck per call: the
    callable is evaluated inside the traced physics step, where a NumPy read
    would be a host callback.
    """
    t_arr = jnp.asarray(days_s)
    p_arr = jnp.asarray(per_time)

    def fn(t_seconds):
        t = jnp.asarray(t_seconds, dtype=p_arr.dtype)
        return jax.vmap(lambda col: jnp.interp(t, t_arr, col), in_axes=1)(p_arr)

    return fn


def _interp_scalar_fn(days_s: np.ndarray, values: np.ndarray):
    """Time-interpolating scalar callable, pure JAX.

    A CONSTANT series collapses to a constant rather than an interpolation.
    Besides being cheaper, this keeps a prescribed surface temperature off the
    traced path: inject_prescribed_T_sfc_into_phys_state validates T_s by
    materialising it with numpy.asarray, on the stated assumption that the SCM
    driver is not traced -- which is false inside this campaign's scanned,
    differentiated rollout. Every deck used here (BOMEX, RICO, DYCOMS) stores a
    time-invariant surface series, so the constant path is the one taken; a
    genuinely time-varying T_s case would still hit that validator and is
    rejected below rather than failing deep inside the first rollout.
    """
    v_np = np.asarray(values, dtype=np.float64)
    if v_np.size == 1 or np.allclose(v_np, v_np.flat[0], rtol=0.0, atol=0.0):
        scalar = jnp.asarray(v_np.flat[0])
        return lambda _t: scalar
    t_arr = jnp.asarray(days_s)
    v_arr = jnp.asarray(v_np)
    return lambda t: jnp.interp(jnp.asarray(t, dtype=v_arr.dtype), t_arr, v_arr)


def hydrostatic_pressure_from_theta(z_m, theta_K, p_s_pa: float, *,
                                    n_aux: int = _AUX_LEVELS):
    """``(z, p)`` top-to-bottom for an arbitrary theta(z), integrated
    hydrostatically from ``p_s_pa``.

    Shared by the deck bridge and the analytic-case bridge so the two cannot
    grow different vertical mappings. The integration itself is NOT written
    here: an auxiliary :class:`HeightCoordinate` is built with ``theta_ref_fn``
    set to the supplied profile -- the same integrator
    ``build_sam_case_height_coord`` gives the LES -- and its Exner reference is
    inverted with the shared :func:`exner_to_pressure`.

    ``theta_K`` should be the VIRTUAL potential temperature where moisture is
    present; for a dry case theta_v == theta.
    """
    z_arr = np.asarray(z_m, dtype=np.float64)
    th_arr = np.asarray(theta_K, dtype=np.float64)
    z_top = float(np.max(z_arr))
    z_j = jnp.asarray(z_arr)
    th_j = jnp.asarray(th_arr)

    def theta_ref_fn(z):
        return jnp.interp(z, z_j, th_j)

    hc = create_stretched_height_coordinate(
        n_aux, H=z_top, dz_sfc=0.5 * z_top / n_aux,
        theta_ref_fn=theta_ref_fn, p_sfc=float(p_s_pa),
    )
    z_aux = np.asarray(hc.z_full, dtype=np.float64)
    p_aux = np.asarray(exner_to_pressure(hc.exner_ref), dtype=np.float64)
    if np.any(np.diff(z_aux) >= 0.0):
        raise ValueError("auxiliary height coordinate is not top-to-bottom.")
    if np.any(np.diff(p_aux) <= 0.0):
        raise ValueError(
            "hydrostatic pressure must increase downward; got a non-monotonic "
            "p(z) from the supplied theta profile."
        )
    return z_aux, p_aux


def heights_from_pressure(z_aux, p_aux, p_full_pa: np.ndarray) -> np.ndarray:
    """Public alias of the p->z inversion, for the analytic-case bridge."""
    return _heights_from_pressure(z_aux, p_aux, p_full_pa)


def _deck_pressure_profile(snd, p_s_pa: float, *, n_aux: int = _AUX_LEVELS):
    """Hydrostatic ``p(z)`` for a deck whose ``p`` column is the ``-999`` sentinel.

    Every gSAM deck used here (BOMEX, RICO, DYCOMS_RF01) stores ``p = -999`` at
    every level and carries only ``pres0``; pressure is meant to be integrated
    from the sounding's potential temperature. That integration is NOT redone
    here — an auxiliary :class:`HeightCoordinate` is built over the sounding's
    own depth with ``theta_ref_fn`` set to the deck sounding, which is the same
    integrator ``build_sam_case_height_coord`` gives the LES, and its Exner
    reference is inverted with the shared :func:`exner_to_pressure`.

    Returns ``(z_aux, p_aux)``, both descending in height (top-to-bottom).
    """
    z_top = float(np.max(np.asarray(snd.z)))
    z_snd = jnp.asarray(np.asarray(snd.z, dtype=np.float64))
    # VIRTUAL potential temperature, not dry theta. The hydrostatic balance is
    # set by density, so a dry mapping displaces heights by ~epsilon*q_v: in
    # BOMEX's ~17 g/kg boundary layer that is ~1%, and the matched spectral LES
    # builds its own reference state from theta_v (make_anelastic_reference).
    # A dry mapping here would place the SCM profiles and forcing at heights
    # ~1% off the LES they are compared against.
    # theta_v via the SHARED helper the global model and the CRM use, not an
    # inline (1 + 0.608 q) -- it is a multiplicative factor, so applying it to
    # theta gives theta_v exactly as applying it to T gives T_v.
    theta_v_snd = jnp.asarray(virtual_temperature(
        np.asarray(snd.theta, dtype=np.float64),
        np.asarray(snd.q_v, dtype=np.float64),
    ))

    def theta_ref_fn(z):
        return jnp.interp(z, z_snd, theta_v_snd)

    # dz_sfc must leave room for a stretch ratio > 1 (exactly H/n_aux is
    # rejected as "uniform layers would exceed H"), so use half of it: the
    # solved ratio stays near 1 (dz_top/dz_sfc ~ 4 at n_aux=512) and the extra
    # near-surface resolution is where the hydrostatic integral matters most.
    hc = create_stretched_height_coordinate(
        n_aux, H=z_top, dz_sfc=0.5 * z_top / n_aux,
        theta_ref_fn=theta_ref_fn, p_sfc=float(p_s_pa),
    )
    z_aux = np.asarray(hc.z_full, dtype=np.float64)
    p_aux = np.asarray(exner_to_pressure(hc.exner_ref), dtype=np.float64)
    if np.any(np.diff(z_aux) >= 0.0):
        raise ValueError("auxiliary height coordinate is not top-to-bottom.")
    if np.any(np.diff(p_aux) <= 0.0):
        raise ValueError(
            "hydrostatic pressure must increase downward; got a non-monotonic "
            "p(z) from the deck sounding."
        )
    return z_aux, p_aux


def _heights_from_pressure(z_aux, p_aux, p_full_pa: np.ndarray) -> np.ndarray:
    """Height of each SCM pressure level, by inverting ``p(z)``.

    Interpolation is linear in ``ln p``, which is where ``z`` is closest to
    linear.
    """
    # Both inputs are already top-to-bottom, so p_aux ASCENDS (small at the
    # model top, p_s at the surface) exactly as np.interp needs its xp, and
    # z_aux descends alongside it. Reversing either one silently feeds np.interp
    # a descending xp and returns garbage.
    z_full = np.interp(np.log(np.asarray(p_full_pa, dtype=np.float64)),
                       np.log(p_aux), z_aux)
    if np.any(np.diff(z_full) >= 0.0):
        raise ValueError(
            "SCM heights must decrease monotonically from level 0 (model top) "
            "to the surface; got a non-monotonic z_full."
        )
    return z_full


@dataclass(frozen=True)
class SAMSCMCase:
    """An SCM column + forcing built from a gSAM case deck.

    All profile arrays are ``(nlev,)`` indexed TOP-TO-BOTTOM, matching
    ``make_column_state``.
    """
    name: str
    spec: SAMSCMCaseSpec
    nlev: int
    dt: float
    sigma_top: float
    p_s: float                      # [Pa]
    T_profile: np.ndarray           # [K]
    q_v_profile: np.ndarray         # [kg/kg]
    u_profile: np.ndarray           # [m/s]
    v_profile: np.ndarray           # [m/s]
    z_full: np.ndarray              # [m] above surface
    p_full: np.ndarray              # [Pa]
    forcing: SCMForcing
    surface: dict[str, float]       # raw deck surface values at ``day``
    rho_sfc: float                  # [kg/m^3]
    case_dir: str

    @property
    def latitude_deg(self) -> float:
        return self.spec.latitude_deg

    @property
    def les_domain_top_m(self) -> float:
        return self.spec.les_domain_top_m

    def les_mask(self) -> np.ndarray:
        """Boolean ``(nlev,)``: SCM levels inside the LES domain.

        Scoring an SCM against an LES reference is only defined where the LES
        actually has a domain; above its top the LES has a sponge and a lid and
        represents nothing.
        """
        return np.asarray(self.z_full) <= self.les_domain_top_m

    def scm_kwargs(self) -> dict[str, Any]:
        return dict(
            nlev=self.nlev,
            dt=self.dt,
            T_profile=jnp.asarray(self.T_profile),
            q_v_profile=jnp.asarray(self.q_v_profile),
            u=jnp.asarray(self.u_profile),
            v=jnp.asarray(self.v_profile),
            p_s=self.p_s,
            latitude_deg=self.spec.latitude_deg,
            sigma_top=self.sigma_top,
            forcing=self.forcing,
        )

    def create_scm(self, *, physics_config, **overrides) -> SingleColumnModel:
        """Build the SCM. ``overrides`` win over the case defaults."""
        kwargs = self.scm_kwargs()
        kwargs.update(overrides)
        return SingleColumnModel.create(physics_config=physics_config, **kwargs)


def load_sam_scm_case(
    case: str,
    *,
    nlev: int = 64,
    dt: float | None = None,
    sigma_top: float | None = None,
    day: float = 0.0,
    case_dir: str | None = None,
    coriolis: bool = True,
    geostrophic: bool = True,
) -> SAMSCMCase:
    """Load a gSAM case deck as an SCM column + :class:`SCMForcing`.

    Parameters
    ----------
    case
        Key of :data:`SAM_SCM_CASES` (``"bomex"``, ``"rico"``, ``"dycoms"``).
    nlev
        Number of SCM levels spanning the case column.
    sigma_top
        Top of the SCM column, as a fraction of surface pressure. ``None``
        (default) puts the column top at the DECK SOUNDING TOP, which is the
        depth the case actually specifies (1.5-4.5 km for these cases) and the
        depth the matching LES runs. Extrapolating a stratosphere the case
        never defined would invent the profile the closure is scored against,
        so the default deliberately does not do it.
    dt
        Physics timestep [s]; defaults to the case's ``default_dt_s``.
    day
        Deck time [days] at which the initial surface state is sampled. The
        forcing callables interpolate in time from ``day`` onwards.
    coriolis, geostrophic
        Disable to run without rotation / without geostrophic relaxation.
    """
    if case not in SAM_SCM_CASES:
        raise ValueError(
            f"Unknown SAM SCM case {case!r}; choose from "
            f"{sorted(SAM_SCM_CASES)}."
        )
    spec = SAM_SCM_CASES[case]
    if nlev < 2:
        raise ValueError(f"nlev must be >= 2, got {nlev}")
    if sigma_top is not None and not 0.0 < sigma_top < 1.0:
        raise ValueError(f"sigma_top must be in (0, 1), got {sigma_top}")
    resolved_dir = case_dir or resolve_sam_case_dir(spec.gsam_dir)
    dt = float(spec.default_dt_s if dt is None else dt)

    snd = read_sam_snd(f"{resolved_dir}/snd")
    lsf = read_sam_lsf(f"{resolved_dir}/lsf")
    sfc = read_sam_sfc(f"{resolved_dir}/sfc")

    p_s = float(snd.pres0) * 100.0                      # mb -> Pa

    # These decks store p = -999 at every level, so reconstruct p(z)
    # hydrostatically from the sounding's own theta before anything else.
    z_aux, p_aux = _deck_pressure_profile(snd, p_s)
    z_snd_top = float(np.max(np.asarray(snd.z)))
    if sigma_top is None:
        # Column top = the LES DOMAIN top, not the sounding top. Masking
        # out-of-domain levels only removes them from the SCORE; they would
        # still set the SCM's upper boundary and change the gradients and
        # entrainment feeding the levels that ARE scored. BOMEX's sounding
        # reaches 4 km while its LES lid is at 3 km, so defaulting to the
        # sounding top would evolve a materially different column.
        top_m = min(float(spec.les_domain_top_m), z_snd_top)
        sigma_top = float(np.interp(top_m, z_aux[::-1], p_aux[::-1]) / p_s)
    if not 0.0 < sigma_top < 1.0:
        raise ValueError(
            f"derived sigma_top={sigma_top} outside (0, 1); deck p_s={p_s} Pa."
        )

    sigma = create_sigma_coordinate(nlev, sigma_top=sigma_top, dtype=jnp.float64)
    sigma_full = np.asarray(sigma.sigma_full, dtype=np.float64)
    p_full = sigma_full * p_s                            # [Pa], top-to-bottom
    if p_full.min() < p_aux.min() - 1.0:
        raise ValueError(
            f"SCM column top ({p_full.min():.1f} Pa) is above the deck sounding "
            f"top ({p_aux.min():.1f} Pa, z={z_snd_top:.0f} m); the sounding "
            "does not define the profile there. Lower nlev/sigma_top or pick a "
            "case with a deeper sounding."
        )
    z_full = _heights_from_pressure(z_aux, p_aux, p_full)

    exner_full = np.asarray(exner_function(jnp.asarray(p_full)), dtype=np.float64)

    ic = interp_sounding_to_levels(snd, z_full)
    T_profile = np.asarray(ic["theta"], dtype=np.float64) * exner_full
    q_v_profile = np.asarray(ic["q_v"], dtype=np.float64)
    u_profile = np.asarray(ic["u"], dtype=np.float64)
    v_profile = np.asarray(ic["v"], dtype=np.float64)

    # --- large-scale forcing, interpolated in height then carried in time ---
    days = np.asarray(lsf.days, dtype=np.float64)
    n_time = days.shape[0]
    per_time = {k: [] for k in ("w_ls", "T_adv", "qv_adv", "u_ls", "v_ls")}
    for t in range(n_time):
        f_t = interp_forcing_to_levels(lsf, z_full, day=float(days[t]))
        for k in per_time:
            per_time[k].append(np.asarray(f_t[k], dtype=np.float64))
    stacked = {k: np.stack(v, axis=0) for k, v in per_time.items()}

    # Deck tls is dT_abs/dt; SCMForcing.theta_adv is a POTENTIAL-temperature
    # tendency (compute_forcing_tendencies multiplies it by Exner). Convert.
    stacked["theta_adv"] = stacked["T_adv"] / exner_full[None, :]

    days_s = (days - float(day)) * 86400.0

    def _profile(key):
        arr = stacked[key]
        if n_time == 1:
            return _const_profile_fn(arr[0])
        return _interp_profile_fn(days_s, arr)

    # --- surface boundary ---------------------------------------------------
    sfc0 = surface_at_day(sfc, day)
    # Surface air density from the DECK sounding at the surface, using the
    # virtual temperature. Deliberately NOT the lowest SCM level: that would
    # make the W/m^2 -> kinematic flux conversion depend on nlev, so changing
    # the SCM resolution would silently change the surface forcing even though
    # the LES deck is unchanged. The LES uses its own lowest cell-centre
    # density from the same deck, which is likewise resolution-independent
    # for a fixed LES grid.
    theta_sfc = float(np.interp(0.0, np.asarray(snd.z, dtype=np.float64),
                                np.asarray(snd.theta, dtype=np.float64)))
    q_v_sfc = float(np.interp(0.0, np.asarray(snd.z, dtype=np.float64),
                              np.asarray(snd.q_v, dtype=np.float64)))
    exner_sfc = float(np.asarray(exner_function(jnp.asarray(p_s))))
    T_v_sfc = float(virtual_temperature(theta_sfc * exner_sfc, q_v_sfc))
    rho_sfc = p_s / (constants.R_d * T_v_sfc)

    if spec.surface_mode not in ("fluxes", "T_s"):
        raise ValueError(
            f"{case!r}: surface_mode={spec.surface_mode!r} must be 'fluxes' "
            "or 'T_s'."
        )
    if spec.surface_mode == "T_s":
        prescribe = "T_s"
        sst_series = np.asarray(sfc.sst, dtype=np.float64)
        if sst_series.size > 1 and not np.allclose(sst_series, sst_series[0],
                                                   rtol=0.0, atol=0.0):
            # inject_prescribed_T_sfc_into_phys_state validates T_s by
            # materialising it with numpy.asarray, on the stated assumption
            # that the SCM driver is not traced. That assumption is false
            # inside a scanned, differentiated rollout, so a time-varying T_s
            # would raise TracerArrayConversionError deep in the first step.
            # Refuse here, where the message is actionable.
            raise NotImplementedError(
                f"{case!r} has a time-varying SST "
                f"({sst_series.min():.2f}-{sst_series.max():.2f} K). "
                "SCMForcing.T_s is validated on the host, so it cannot be "
                "traced; a time-varying prescribed surface temperature needs "
                "that validator made trace-safe first."
            )
        surface_kwargs = dict(
            T_s=_interp_scalar_fn((np.asarray(sfc.days) - day) * 86400.0,
                                  np.asarray(sfc.sst, dtype=np.float64)),
        )
    else:
        prescribe = "fluxes"
        sfc_days_s = (np.asarray(sfc.days) - day) * 86400.0
        w_T = np.array([surface_kinematic_temperature_flux(float(s), rho_sfc)
                        for s in np.asarray(sfc.shf)], dtype=np.float64)
        w_q = np.array([surface_kinematic_moisture_flux(float(s), rho_sfc)
                        for s in np.asarray(sfc.lhf)], dtype=np.float64)
        surface_kwargs = dict(
            w_th_s=_interp_scalar_fn(sfc_days_s, w_T),
            w_qv_s=_interp_scalar_fn(sfc_days_s, w_q),
        )

    f_c = (2.0 * constants.Omega * float(np.sin(np.deg2rad(spec.latitude_deg)))
           if coriolis else 0.0)

    forcing = SCMForcing(
        f_c=f_c,
        u_geo=_profile("u_ls") if geostrophic else None,
        v_geo=_profile("v_ls") if geostrophic else None,
        subsidence_w=_profile("w_ls"),      # deck w_ls is positive UP, as is
                                            # SCMForcing.subsidence_w
        theta_adv=_profile("theta_adv"),
        qv_adv=_profile("qv_adv"),
        prescribe=prescribe,
        **surface_kwargs,
    )

    return SAMSCMCase(
        name=case, spec=spec, nlev=nlev, dt=dt, sigma_top=sigma_top, p_s=p_s,
        T_profile=T_profile, q_v_profile=q_v_profile,
        u_profile=u_profile, v_profile=v_profile,
        z_full=z_full, p_full=p_full, forcing=forcing,
        surface=sfc0, rho_sfc=rho_sfc, case_dir=resolved_dir,
    )
