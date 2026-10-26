"""Loader for NCAR-SCCM (ARM SGP) single-column forcing + observation files.

The ARM Single-Column-Community-Model (SCCM) IOP files — e.g. the ARM SGP
July-1997 variational-analysis product ``arm9707.nc`` (Khairoutdinov, CSU,
May 2000) — are the observational analogue of a DEPHY case: an obs-CONSTRAINED
forcing dataset (time-varying surface fluxes + large-scale advective tendencies
+ subsidence, all derived from the ARM sounding array and the constrained
variational analysis) that ALSO carries the observed targets a column model is
scored against (GOES cloud amount, MWR liquid-water path, precipitation,
sounding temperature/humidity).

This loader maps that file onto the SAME objects the idealized suite uses — an
:class:`SCMForcing` + initial profiles for :class:`SingleColumnModel` — and, in
addition, extracts an :class:`ARMObsReference` of the observed fields so an SCM
run driven by the obs-derived forcing can be scored against real observations
(not only LES truth).  It reuses the shared format-agnostic conversions in
:mod:`scm_forcing_io`; only the SCCM variable-name mapping lives here.

SCCM variable conventions honoured here (see CLAUDE.md sign/unit rule):

* ``lev`` — fixed pressure levels [Pa]; profiles are ``(time, lev)`` after the
  singleton ``lat``/``lon`` axes are squeezed.
* ``omega`` — vertical PRESSURE velocity [Pa/s], positive DOWNWARD; converted to
  the SCM's upward ``subsidence_w`` [m/s] via hydrostatic ``-omega/(rho g)``.
* ``divT`` — HORIZONTAL temperature advective tendency [K/s]; converted to a
  potential-temperature tendency ``theta_adv = divT / Pi``.  The VERTICAL
  advective tendency (``vertdivT``) is deliberately NOT added: the SCM
  reconstructs vertical advection itself from the prescribed ``subsidence_w``,
  so adding it would double-count (standard "revealed forcing" decomposition).
* ``divq`` — HORIZONTAL water-vapour (mixing-ratio) advective tendency
  [(kg/kg)/s]; used directly as ``qv_adv`` (``q`` is a mixing ratio).
* ``shflx``/``lhflx`` — surface sensible/latent heat flux [W/m2], positive
  UPWARD; converted to kinematic surface fluxes ``w_th_s = shflx/(rho_s c_pd)``
  [K m/s, a temperature flux] and ``w_qv_s = lhflx/(rho_s L_v)`` [(kg/kg) m/s],
  matching the ``SCMForcing`` prescribe="fluxes" convention (positive upward).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing
from legoesm.atmosphere.forcing.scm.scm_forcing_io import (
    forcing_cadence,
    interp_profile_to_pressure,
    omega_to_w,
    profile_time_fn,
    scalar_time_fn,
)
from legoesm.atmosphere.physics._shared import compute_rho, exner_function
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.thermo import mixing_ratio_to_specific_humidity

from legoesm import constants

# GOES/MWR cloud-amount fields are stored as a percentage; scale to a fraction.
_PERCENT_TO_FRACTION = 0.01
_SECONDS_PER_DAY = 86400.0
_MM_PER_M = 1000.0
# The ARM forcing cadence is ~3 h; that is FAR too large as an SCM integration
# step.  The forcing callables interpolate between frames, so the model steps at
# a small stable dt and samples interpolated forcing.  Default the integration
# step to min(cadence, this cap).
_DEFAULT_MAX_DT_S = 300.0


@dataclass(frozen=True)
class ARMObsReference:
    """Observed fields from an ARM SCCM IOP file, for scoring an SCM run.

    Host-side numpy arrays (never traced).  ``time_seconds`` are seconds after
    0Z on ``base_date`` (a ``yymmdd`` integer).  Profiles are ``(ntime, nlev)``
    on ``pressure_pa`` [Pa].  Cloud amounts are fractions (GOES % / 100); LWP is
    ``kg/m^2`` (MWR liquid depth [m] x rho_water).  ``None`` marks an absent
    variable.
    """

    base_date: int
    latitude_deg: float
    longitude_deg: float
    time_seconds: np.ndarray
    pressure_pa: np.ndarray
    # sounding profiles (observed / analysed state)
    T: np.ndarray | None
    q: np.ndarray | None
    u: np.ndarray | None
    v: np.ndarray | None
    # surface scalars
    surface_pressure_pa: np.ndarray | None
    surface_air_temperature_k: np.ndarray | None
    surface_rh_percent: np.ndarray | None
    surface_wind_speed: np.ndarray | None
    sensible_heat_flux_wm2: np.ndarray | None
    latent_heat_flux_wm2: np.ndarray | None
    precip_m_s: np.ndarray | None
    # cloud observations
    total_cloud_fraction: np.ndarray | None
    low_cloud_fraction: np.ndarray | None
    mid_cloud_fraction: np.ndarray | None
    high_cloud_fraction: np.ndarray | None
    liquid_water_path_kg_m2: np.ndarray | None
    cloud_top_height_m: np.ndarray | None
    cloud_thickness_m: np.ndarray | None
    # radiation observations
    surface_net_down_radiation_wm2: np.ndarray | None
    toa_lw_up_wm2: np.ndarray | None
    toa_sw_dn_wm2: np.ndarray | None

    @property
    def precip_mm_day(self) -> np.ndarray | None:
        """Precipitation rate [mm/day] (``precip_m_s`` x 1000 x 86400)."""
        if self.precip_m_s is None:
            return None
        return self.precip_m_s * (_MM_PER_M * _SECONDS_PER_DAY)


@dataclass(frozen=True)
class ARMSCMCase:
    """An ARM SCCM case mapped to legoESM SCM inputs plus its obs reference."""

    path: Path
    case_id: str
    latitude_deg: float
    longitude_deg: float
    base_date: int
    sigma_top: float
    dt: float
    duration_seconds: float
    time_seconds: np.ndarray
    pressure_full: jax.Array
    T_profile: jax.Array
    q_v_profile: jax.Array | None
    u_profile: jax.Array
    v_profile: jax.Array
    p_s: float
    phis: float
    forcing: SCMForcing
    obs: ARMObsReference
    notes: tuple[str, ...] = ()

    @property
    def nlev(self) -> int:
        return int(self.T_profile.shape[0])

    @property
    def recommended_nsteps(self) -> int:
        return int(np.ceil(self.duration_seconds / self.dt))

    def scm_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :meth:`SingleColumnModel.create`."""
        return {
            "nlev": self.nlev,
            "dt": self.dt,
            "T_profile": self.T_profile,
            "q_v_profile": self.q_v_profile,
            "u": self.u_profile,
            "v": self.v_profile,
            "p_s": self.p_s,
            "phis": self.phis,
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "sigma_top": self.sigma_top,
            "forcing": self.forcing,
            "t0_seconds": float(self.time_seconds[0]),
        }

    def create_scm(self, *, physics_config, **overrides) -> SingleColumnModel:
        """Create a :class:`SingleColumnModel` for this case.

        ``physics_config`` is explicit so the caller decides which schemes are
        active.  Unlike a DEPHY ``radiation='tend'`` case, the SCCM forcing here
        carries ONLY the horizontal advective tendencies (``divT``/``divq``) and
        subsidence — NOT a prescribed radiative tendency (the file's ``colRH`` is
        a column-integrated diagnostic, not a level tendency).  Radiation is
        therefore NOT double-counted by an active radiation scheme; run
        interactive radiation for a realistic column (or off to isolate the
        turbulence closure), but do not disable it to "avoid double-counting".
        """
        kwargs = self.scm_kwargs()
        kwargs.update(overrides)
        return SingleColumnModel.create(physics_config=physics_config, **kwargs)


def load_arm_sccm_case(
    path: str | Path,
    *,
    nlev: int = 64,
    sigma_top: float | None = None,
    dt: float | None = None,
    dtype=None,
    include_advective_tendencies: bool = True,
    include_vertical_velocity: bool = True,
    include_surface_fluxes: bool = True,
) -> ARMSCMCase:
    """Load an NCAR-SCCM (ARM) IOP NetCDF file.

    Parameters
    ----------
    path
        Path to an ARM SCCM file (e.g. ``arm9707.nc``).
    nlev
        Number of legoESM sigma full levels to initialise.
    sigma_top
        Sigma at the model top; ``None`` infers a top that stays inside the
        SCCM pressure range.
    dt
        SCM integration step [s]; ``None`` defaults to ``min(forcing cadence,
        300 s)`` — the ~3 h ARM cadence is too large to integrate at, and the
        forcing is interpolated between frames.
    """
    if dtype is None:
        dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

    path = Path(path)
    ds = _open_dataset(path)
    try:
        time = _dim_var(ds, "tsec")
        lev = _dim_var(ds, "lev")  # pressure [Pa], fixed in time
        latitude = _first(_squeeze(ds, "lat")) if _has(ds, "lat") else 0.0
        longitude = _first(_squeeze(ds, "lon")) if _has(ds, "lon") else 0.0
        base_date = int(_first(_squeeze(ds, "bdate"))) if _has(ds, "bdate") else 0
        # Surface geopotential [m^2/s^2] (ARM SGP sits at ~318 m elevation); the
        # SCCM stores it directly, so do not silently drop it to zero.
        phis = _first(_squeeze(ds, "phis")) if _has(ds, "phis") else 0.0

        ps_series = _series(ds, "Ps", time)
        p_s = float(ps_series[0]) if ps_series is not None else constants.p_ref

        if sigma_top is None:
            sigma_top = max(0.01, float(np.nanmin(lev)) / p_s)
        sigma_top = float(sigma_top)
        sigma_coord = create_sigma_coordinate(nlev, sigma_top=sigma_top, dtype=dtype)
        p_target = np.asarray(
            sigma_coord.pressure_at_full(jnp.asarray(p_s, dtype=dtype)),
            dtype=np.float64,
        )
        exner_target = np.asarray(exner_function(jnp.asarray(p_target)), dtype=np.float64)

        # --- initial profiles (t=0 sounding interpolated to the sigma grid) ---
        T_src = _profile(ds, "T", time)  # (ntime, nlev_src)
        q_src = _profile(ds, "q", time)
        u_src = _profile(ds, "u", time)
        v_src = _profile(ds, "v", time)
        if T_src is None:
            raise ValueError("ARM SCCM file must contain temperature 'T'")
        T_profile = interp_profile_to_pressure(p_target, lev, T_src[0])
        q_profile = (
            np.maximum(interp_profile_to_pressure(p_target, lev, q_src[0]), 0.0)
            if q_src is not None else None
        )
        u_profile = (
            interp_profile_to_pressure(p_target, lev, u_src[0])
            if u_src is not None else np.zeros_like(p_target)
        )
        v_profile = (
            interp_profile_to_pressure(p_target, lev, v_src[0])
            if v_src is not None else np.zeros_like(p_target)
        )

        notes: list[str] = []

        # Time-varying sounding on the model grid, reused for the hydrostatic
        # density in omega->w AND the near-surface density in the surface-flux
        # conversion.  ``compute_rho``/``omega_to_w`` want SPECIFIC humidity, so
        # convert the SCCM mixing ratio q_grid_r -> q_grid_qv here.
        T_grid = _interp_time_profiles(p_target, lev, T_src)
        q_grid_qv = None
        if q_src is not None:
            q_grid_r = np.maximum(_interp_time_profiles(p_target, lev, q_src), 0.0)
            q_grid_qv = np.asarray(
                mixing_ratio_to_specific_humidity(jnp.asarray(q_grid_r)), dtype=np.float64
            )

        # --- large-scale advective tendencies (HORIZONTAL only; vertical comes
        #     from subsidence_w to avoid double-counting) ---
        theta_adv_values = None
        qv_adv_values = None
        if include_advective_tendencies:
            divT = _profile(ds, "divT", time)
            if divT is not None:
                # divT is a TEMPERATURE tendency [K/s]; theta = T/Pi -> divide.
                divT_grid = _interp_time_profiles(p_target, lev, divT)
                theta_adv_values = divT_grid / exner_target.reshape(1, -1)
                notes.append("ARM divT (horizontal T tendency) converted to theta_adv via Exner.")
            divq = _profile(ds, "divq", time)
            if divq is not None:
                # divq is already a mixing-ratio tendency [(kg/kg)/s].
                qv_adv_values = _interp_time_profiles(p_target, lev, divq)

        # --- large-scale subsidence (omega -> upward w) ---
        subsidence_values = None
        if include_vertical_velocity:
            omega = _profile(ds, "omega", time)
            if omega is not None:
                omega_grid = _interp_time_profiles(p_target, lev, omega)
                subsidence_values = omega_to_w(omega_grid, T_grid, p_target, q_grid_qv)
                notes.append("ARM omega (Pa/s, +down) converted to upward subsidence_w.")

        # --- surface fluxes -> kinematic (positive upward) ---
        # TIME-VARYING near-surface (lowest model level) air density, so the
        # kinematic flux tracks the diurnal density cycle over the multi-day IOP.
        qv_sfc_series = None if q_grid_qv is None else q_grid_qv[:, -1]
        rho_sfc_series = np.asarray(
            compute_rho(
                jnp.asarray(T_grid[:, -1]),
                jnp.asarray(np.full(time.size, p_target[-1])),
                None if qv_sfc_series is None else jnp.asarray(qv_sfc_series),
            ),
            dtype=np.float64,
        )
        surface_kwargs: dict[str, Any] = {}
        shflx_series = _series(ds, "shflx", time)
        lhflx_series = _series(ds, "lhflx", time)
        if include_surface_fluxes and (shflx_series is not None or lhflx_series is not None):
            surface_kwargs["prescribe"] = "fluxes"
            if shflx_series is not None:
                w_th = shflx_series / (rho_sfc_series * constants.c_pd)  # [K m/s], T-flux
                surface_kwargs["w_th_s"] = scalar_time_fn(time, w_th)
            if lhflx_series is not None:
                w_qv = lhflx_series / (rho_sfc_series * constants.L_v)  # [(kg/kg) m/s]
                surface_kwargs["w_qv_s"] = scalar_time_fn(time, w_qv)

        f_c = float(2.0 * constants.Omega * np.sin(np.deg2rad(latitude)))
        if u_src is not None:
            notes.append(
                "ARM SCCM has observed winds but no geostrophic wind; winds evolve "
                "under surface drag from the initial sounding (no wind forcing/nudging)."
            )

        forcing = SCMForcing(
            f_c=f_c,
            u_geo=None,
            v_geo=None,
            subsidence_w=(
                profile_time_fn(time, subsidence_values, dtype)
                if subsidence_values is not None else None
            ),
            theta_adv=(
                profile_time_fn(time, theta_adv_values, dtype)
                if theta_adv_values is not None else None
            ),
            qv_adv=(
                profile_time_fn(time, qv_adv_values, dtype)
                if qv_adv_values is not None else None
            ),
            **surface_kwargs,
        )

        obs = _build_obs_reference(
            ds, time, lev, base_date, float(latitude), float(longitude),
            T_src, q_src, u_src, v_src, ps_series, shflx_series, lhflx_series,
        )

        dt_out = (
            float(dt) if dt is not None
            else min(forcing_cadence(time), _DEFAULT_MAX_DT_S)
        )
        duration = float(time[-1] - time[0]) if time.size > 1 else 0.0

        return ARMSCMCase(
            path=path,
            case_id=str(getattr(ds, "attrs", {}).get("title", path.stem)),
            latitude_deg=float(latitude),
            longitude_deg=float(longitude),
            base_date=base_date,
            sigma_top=sigma_top,
            dt=dt_out,
            duration_seconds=duration,
            time_seconds=np.asarray(time, dtype=np.float64),
            pressure_full=jnp.asarray(p_target, dtype=dtype),
            T_profile=jnp.asarray(T_profile, dtype=dtype),
            q_v_profile=(jnp.asarray(q_profile, dtype=dtype) if q_profile is not None else None),
            u_profile=jnp.asarray(u_profile, dtype=dtype),
            v_profile=jnp.asarray(v_profile, dtype=dtype),
            p_s=p_s,
            phis=float(phis),
            forcing=forcing,
            obs=obs,
            notes=tuple(dict.fromkeys(notes)),
        )
    finally:
        close = getattr(ds, "close", None)
        if callable(close):
            close()


def _build_obs_reference(
    ds, time, lev, base_date, latitude, longitude,
    T_src, q_src, u_src, v_src, ps_series, shflx_series, lhflx_series,
) -> ARMObsReference:
    def frac(name: str) -> np.ndarray | None:
        s = _series(ds, name, time)
        return None if s is None else s * _PERCENT_TO_FRACTION

    cldliq = _series(ds, "cldliq", time)  # MWR liquid depth [m]
    lwp = None if cldliq is None else cldliq * constants.rho_water  # -> kg/m^2
    return ARMObsReference(
        base_date=base_date,
        latitude_deg=latitude,
        longitude_deg=longitude,
        time_seconds=np.asarray(time, dtype=np.float64),
        pressure_pa=np.asarray(lev, dtype=np.float64),
        T=T_src,
        q=q_src,
        u=u_src,
        v=v_src,
        surface_pressure_pa=ps_series,
        surface_air_temperature_k=_series(ds, "Tsair", time),
        surface_rh_percent=_series(ds, "RH", time),
        surface_wind_speed=_series(ds, "windsrf", time),
        sensible_heat_flux_wm2=shflx_series,
        latent_heat_flux_wm2=lhflx_series,
        precip_m_s=_series(ds, "Prec", time),
        total_cloud_fraction=frac("totcld"),
        low_cloud_fraction=frac("lowcld"),
        mid_cloud_fraction=frac("midcld"),
        high_cloud_fraction=frac("hghcld"),
        liquid_water_path_kg_m2=lwp,
        cloud_top_height_m=_series(ds, "cldht", time),
        cloud_thickness_m=_series(ds, "cldthk", time),
        surface_net_down_radiation_wm2=_series(ds, "NDRsrf", time),
        toa_lw_up_wm2=_series(ds, "TOA_LWup", time),
        toa_sw_dn_wm2=_series(ds, "TOA_SWdn", time),
    )


# --- host-side NetCDF access helpers (I/O only; squeeze the singleton
#     lat/lon axes SCCM carries on every field) -------------------------------
def _open_dataset(path: Path):
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - dependency in pyproject
        raise ImportError("load_arm_sccm_case requires xarray to read NetCDF files.") from exc
    return xr.open_dataset(path, decode_times=False).load()


def _has(ds, name: str) -> bool:
    return name in ds.variables


def _squeeze(ds, name: str) -> np.ndarray:
    arr = np.squeeze(np.asarray(ds[name].values, dtype=np.float64))
    return arr.reshape(1) if arr.ndim == 0 else arr


def _first(arr: np.ndarray) -> float:
    return float(np.asarray(arr, dtype=np.float64).flat[0])


def _dim_var(ds, name: str) -> np.ndarray:
    if not _has(ds, name):
        raise ValueError(f"ARM SCCM file missing required variable {name!r}")
    arr = _squeeze(ds, name)
    if arr.ndim != 1:
        raise ValueError(f"ARM SCCM {name!r} must be 1-D, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def _series(ds, name: str, time: np.ndarray) -> np.ndarray | None:
    """A ``(ntime,)`` surface/scalar time series, or ``None`` if absent."""
    if not _has(ds, name):
        return None
    arr = _squeeze(ds, name)
    if arr.ndim == 1 and arr.size == time.size:
        return np.asarray(arr, dtype=np.float64)
    if arr.size == time.size:
        return np.asarray(arr, dtype=np.float64).reshape(time.size)
    if arr.size == 1:
        return np.full(time.size, float(arr.flat[0]), dtype=np.float64)
    raise ValueError(f"ARM SCCM {name!r} shape {arr.shape} does not match time={time.size}")


def _profile(ds, name: str, time: np.ndarray) -> np.ndarray | None:
    """A ``(ntime, nlev)`` profile time series, or ``None`` if absent."""
    if not _has(ds, name):
        return None
    arr = _squeeze(ds, name)
    if arr.ndim != 2:
        raise ValueError(f"ARM SCCM profile {name!r} must be 2-D (time, lev), got {arr.shape}")
    if arr.shape[0] == time.size:
        return np.asarray(arr, dtype=np.float64)
    if arr.shape[1] == time.size:
        return np.asarray(arr.T, dtype=np.float64)
    raise ValueError(f"ARM SCCM profile {name!r} shape {arr.shape} does not match time={time.size}")


def _interp_time_profiles(
    p_target: np.ndarray, p_source: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Interpolate each time row of ``values`` (nt, nz_src) onto ``p_target``."""
    return np.stack(
        [
            interp_profile_to_pressure(p_target, p_source, values[it])
            for it in range(values.shape[0])
        ],
        axis=0,
    )
