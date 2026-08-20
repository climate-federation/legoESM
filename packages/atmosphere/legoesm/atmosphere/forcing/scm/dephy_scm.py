"""Loader for DEPHY-SCM atmospheric single-column cases.

DEPHY-SCM cases are atmospheric SCM/LES forcing files in a common NetCDF
format.  This module maps the SCM-format files onto legoESM's atmospheric
single-column model by constructing initial profiles and an :class:`SCMForcing`
object.  It is intentionally host-side I/O: the returned forcing callables are
static Python callables, matching :mod:`legoesm.atmosphere.forcing.scm.scm_forcing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics._shared import compute_rho, exner_function
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.thermo import (
    mixing_ratio_to_specific_humidity,
    specific_humidity_tendency_to_mixing_ratio_tendency,
    specific_humidity_to_mixing_ratio,
)


_UNSUPPORTED_PROFILE_FORCINGS = {
    "tnua_adv": "eastward wind advection tendency is not represented in SCMForcing",
    "tnva_adv": "northward wind advection tendency is not represented in SCMForcing",
    "ua_nud": "wind nudging profiles are not represented in SCMForcing",
    "va_nud": "wind nudging profiles are not represented in SCMForcing",
    "ta_nud": "temperature nudging profiles are not represented in SCMForcing",
    "theta_nud": "potential-temperature nudging profiles are not represented in SCMForcing",
    "thetal_nud": "liquid-potential-temperature nudging profiles are not represented in SCMForcing",
    "qv_nud": "moisture nudging profiles are not represented in SCMForcing",
    "qt_nud": "total-water nudging profiles are not represented in SCMForcing",
    "rv_nud": "mixing-ratio nudging profiles are not represented in SCMForcing",
    "rt_nud": "total-water mixing-ratio nudging profiles are not represented in SCMForcing",
}


@dataclass(frozen=True)
class DEPHYSCMCase:
    """A DEPHY-SCM case mapped to legoESM's atmospheric SCM inputs."""

    path: Path
    case_id: str
    title: str
    surface_type: str
    radiation: str
    start_date: str
    end_date: str
    latitude_deg: float
    longitude_deg: float
    orography_m: float
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
    notes: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()

    @property
    def nlev(self) -> int:
        return int(self.T_profile.shape[0])

    @property
    def recommended_nsteps(self) -> int:
        return int(np.ceil(self.duration_seconds / self.dt))

    def scm_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for :meth:`SingleColumnModel.create`."""
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

        ``physics_config`` is explicit so callers decide which physics schemes
        are active.  For DEPHY files with ``radiation='tend'`` or advection that
        already includes radiative heating, use a radiation-off config to avoid
        double counting.
        """
        kwargs = self.scm_kwargs()
        kwargs.update(overrides)
        return SingleColumnModel.create(
            physics_config=physics_config,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Named cases: the published field-campaign forcings this repo can run.
# ---------------------------------------------------------------------------
#
# These files are NOT vendored. Each entry records where the community
# distributes the case, and ``scripts/data/fetch_dephy_cases.py`` downloads it
# into the same ``data/les_cases/`` cache the gSAM decks use. Downloading
# rather than copying keeps provenance with the people who published the case
# and keeps a 36 MB file out of the history.
#
# ``regime``/``campaign`` tie each entry back to the field-campaign programme
# in docs/atmosphere/les_scm_case_programme.md, so a case cannot be added here
# without saying which row of that programme it is -- or that it is a PROXY for
# one, which is a different claim and is spelled out in ``note``.


@dataclass(frozen=True)
class DEPHYCaseSpec:
    """Where a named DEPHY-format case comes from, and what it is."""

    url: str
    filename: str
    reference: str
    regime: str
    campaign: str
    note: str
    # SHA-256 of the file this repo was tested against.
    sha256: str = ""
    # Variable renames for files that carry the DEPHY structure under other
    # names. Empty for anything already in DEPHY v1 naming.
    rename: tuple[tuple[str, str], ...] = ()

    @property
    def rename_map(self) -> dict[str, str]:
        return dict(self.rename)


# PINNED to a commit, not a branch. A branch URL silently serves a different
# file after any upstream edit, and a forcing file that changed under a case
# name is the definition of an uncontrolled comparison. Each entry also carries
# the SHA-256 of the bytes this repo was tested against, so a changed upstream
# file fails loudly instead of quietly becoming a different experiment.
_DEPHY_COMMIT = "661b0328acd253f345715534986e65724c8919db"
_DEPHY_RAW = f"https://raw.githubusercontent.com/GdR-DEPHY/DEPHY-SCM/{_DEPHY_COMMIT}"
_COMBLE_COMMIT = "873d1cb5d838a53cbb58bef21570890150cd4d2c"

# COMBLE ships its forcing under the intercomparison's own variable names while
# declaring "DEPHY SCM format version 2.0". Each rename below was checked
# against the file's units and dimensions, not assumed from the name:
#   pressure [Pa] (t0, lev, lat, lon)  -> pa    same quantity, same units
#   temp     [K]  (t0, lev, lat, lon)  -> ta    absolute temperature
#   u, v     [m/s]                     -> ua, va
# qv, ug, vg, ts, ps and theta already use the DEPHY names. The `*_nudging`
# profiles are deliberately NOT renamed to `*_nud`: the file's own attributes
# set every `nudging_*` switch to 0, so they are reference data rather than an
# active forcing, and renaming them would make the loader report unsupported
# channels that the case does not actually use.
#   ts       [K]  (time,)              -> ts_forc  the surface-temperature
#            SERIES. This one is load-bearing rather than cosmetic: the loader
#            maps a prescribed skin temperature only when it finds `ts_forc`,
#            so without the rename COMBLE came back with NO surface forcing at
#            all -- a 20 h cold-air outbreak whose entire physics is heat and
#            moisture picked up from a warming sea surface, run with the sea
#            surface switched off. It still integrated and still looked
#            finite, which is exactly why this is checked by a test rather
#            than by reading the file. DEPHY v1 files carry the initial value
#            in `ts` and the series in `ts_forc`; COMBLE has only the series.
_COMBLE_RENAME = (
    ("pressure", "pa"),
    ("temp", "ta"),
    ("u", "ua"),
    ("v", "va"),
    ("ts", "ts_forc"),
)

DEPHY_SCM_CASES: dict[str, DEPHYCaseSpec] = {
    "sandu_ref": DEPHYCaseSpec(
        sha256="1803a15d956e9020ca1e79ec8eb782ad343d6b8ebd4b4fcd6a47a3f19cb6840b",
        url=f"{_DEPHY_RAW}/SANDU/REF/SANDU_REF_SCM_driver.nc",
        filename="SANDU_REF_SCM_driver.nc",
        reference="Sandu and Stevens (2011, JAS)",
        regime="stratocumulus-to-cumulus transition (Lagrangian)",
        campaign="SCT composite",
        note="The reference composite transition, 3 days following the "
             "boundary layer along its trajectory over warming water. KNOWN LIMITATION: the composite nudges temperature and moisture toward the trajectory analysis, which SCMForcing cannot represent, so the free troposphere above the boundary layer is unconstrained over the three days. The surface temperature, subsidence and geostrophic wind ARE applied.",
    ),
    "sandu_fast": DEPHYCaseSpec(
        sha256="834f07d284aa318386dfa07f5c62acfd73d4e8620c7cff09a403e89836ce064d",
        url=f"{_DEPHY_RAW}/SANDU/FAST/SANDU_FAST_SCM_driver.nc",
        filename="SANDU_FAST_SCM_driver.nc",
        reference="Sandu and Stevens (2011, JAS)",
        regime="stratocumulus-to-cumulus transition (Lagrangian)",
        campaign="SCT composite",
        note="The FAST composite: the same transition over a faster-warming "
             "sea surface, which breaks the deck up sooner. KNOWN LIMITATION: the composite nudges temperature and moisture toward the trajectory analysis, which SCMForcing cannot represent, so the free troposphere above the boundary layer is unconstrained over the three days. The surface temperature, subsidence and geostrophic wind ARE applied.",
    ),
    "sandu_slow": DEPHYCaseSpec(
        sha256="48904b75fd5fad473b2559d77c63e47e85fb0edab8d7b66f419537929f9eb800",
        url=f"{_DEPHY_RAW}/SANDU/SLOW/SANDU_SLOW_SCM_driver.nc",
        filename="SANDU_SLOW_SCM_driver.nc",
        reference="Sandu and Stevens (2011, JAS)",
        regime="stratocumulus-to-cumulus transition (Lagrangian)",
        campaign="SCT composite",
        note="The SLOW composite, the weak end of the transition spread. KNOWN LIMITATION: the composite nudges temperature and moisture toward the trajectory analysis, which SCMForcing cannot represent, so the free troposphere above the boundary layer is unconstrained over the three days. The surface temperature, subsidence and geostrophic wind ARE applied.",
    ),
    "mpace": DEPHYCaseSpec(
        sha256="f2a1bca710f79996fb1bbf40006fefc90cb81c34e95c86a4b2574ac08915fee0",
        url=f"{_DEPHY_RAW}/MPACE/REF/MPACE_REF_SCM_driver.nc",
        filename="MPACE_REF_SCM_driver.nc",
        reference="Klein et al. (2009, QJRMS)",
        regime="Arctic mixed-phase stratus",
        campaign="M-PACE",
        note="12 h from 17 UTC 9 October 2004 over a 274.01 K ocean surface, "
             "prescribed surface fluxes and advective tendencies. KNOWN "
             "LIMITATION: the case forces its wind by NUDGING toward the "
             "initial profile, and no relaxation channel is wired into "
             "SCMForcing, so the loaded column has no wind forcing -- and "
             "therefore no Coriolis "
             "either, since rotation with nothing to rotate toward would spin "
             "the wind down to rest. Its thermodynamics are forced as "
             "specified; its wind is not.",
    ),
    "armcu": DEPHYCaseSpec(
        sha256="c26ffa08ff04a259da8fd211809f66eb78037548005f58b8e3dcd611e6023d1e",
        url=f"{_DEPHY_RAW}/ARMCU/REF/ARMCU_REF_SCM_driver.nc",
        filename="ARMCU_REF_SCM_driver.nc",
        reference="Brown et al. (2002, QJRMS)",
        regime="continental cumulus",
        campaign="ARM SGP, 21 June 1997",
        note="PROXY, not RACORO. The continental-cumulus programme row is the "
             "RACORO IOP (Vogelmann et al. 2015); this is the older ARM SGP "
             "diurnal-cycle case over the same site in a different year. Use "
             "it for the regime, never as RACORO.",
    ),
    "comble": DEPHYCaseSpec(
        sha256="9dabe7606799f5203d3c8b869f2840303e39a06c142f1d06537b6dbe1d065425",
        url=("https://raw.githubusercontent.com/ARM-Development/comble-mip/"
             f"{_COMBLE_COMMIT}/notebooks/forcing/"
             "COMBLE_INTERCOMPARISON_FORCING_V2.4.nc"),
        filename="COMBLE_INTERCOMPARISON_FORCING_V2.4.nc",
        reference="COMBLE model-observation intercomparison "
                  "(Juliano, Tornow and Fridlind; "
                  "https://arm-development.github.io/comble-mip/)",
        regime="high-latitude cold-air outbreak (Lagrangian)",
        campaign="COMBLE, 12-13 March 2020",
        note="20 h down a marine cold-air-outbreak trajectory from the ice "
             "edge, forced by a time- and height-varying geostrophic wind and "
             "a prescribed skin temperature that starts at 247 K over ice and "
             "warms over open water. COMBLE-FORCED, not the COMBLE case: the "
             "intercomparison also specifies three prognostic aerosol modes, "
             "a wind-speed-dependent sea-spray source and an "
             "ice-nucleating-particle closure, and in a cold-air outbreak "
             "those set droplet number, warm-rain rate and the mixed-phase "
             "transition. This runs the thermodynamic and dynamic forcing "
             "only, so it is not comparable to that intercomparison's target "
             "quantities.",
        rename=_COMBLE_RENAME,
    ),
}


def resolve_dephy_case_path(name: str) -> str:
    """Local path of a named DEPHY case file, whether or not it is present.

    The file lives in the same cache as the gSAM decks
    (``data/les_cases/dephy/``, overridable with ``$LEGOESM_LES_FORCING``), so
    one environment variable moves every forcing file this repo reads. The path
    is returned even when absent so a caller's existence check can name the
    fetch command rather than failing on a missing file deep inside xarray.
    """
    if name not in DEPHY_SCM_CASES:
        raise ValueError(
            f"Unknown DEPHY case {name!r}; choose from "
            f"{sorted(DEPHY_SCM_CASES)}."
        )
    from legoesm.atmosphere.forcing.sam_case_forcing import resolve_sam_case_dir
    return str(Path(resolve_sam_case_dir("dephy")) / DEPHY_SCM_CASES[name].filename)


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dephy_case(name: str, *, verify: bool = True, **kwargs) -> DEPHYSCMCase:
    """Load a named DEPHY case from the local cache.

    ``kwargs`` are forwarded to :func:`load_dephy_scm_case`; the case's own
    variable renames are applied unless the caller overrides them.

    ``verify`` checks the cached file against the registry's SHA-256 before
    reading it. The fetcher checks on download, but a cache is a directory on
    disk that anything can write to, and a case file replaced by a different
    valid NetCDF is a different experiment running under the same name. Pass
    ``verify=False`` only when deliberately working with a modified file.
    """
    spec = DEPHY_SCM_CASES[name] if name in DEPHY_SCM_CASES else None
    if spec is None:
        raise ValueError(
            f"Unknown DEPHY case {name!r}; choose from "
            f"{sorted(DEPHY_SCM_CASES)}."
        )
    path = Path(resolve_dephy_case_path(name))
    if not path.is_file():
        raise FileNotFoundError(
            f"DEPHY case {name!r} is not in the local cache: {path}\n"
            f"  Fetch it from {spec.url} with:\n"
            f"    python scripts/data/fetch_dephy_cases.py --only {name}"
        )
    if verify and spec.sha256:
        got = _file_sha256(path)
        if got != spec.sha256:
            raise ValueError(
                f"DEPHY case {name!r}: the cached file does not match the "
                f"registry checksum.\n  path     {path}\n  expected "
                f"{spec.sha256}\n  found    {got}\nRe-fetch it with "
                f"`python scripts/data/fetch_dephy_cases.py --only {name} "
                "--force`, or pass verify=False if the change is deliberate."
            )
    kwargs.setdefault("rename", spec.rename_map)
    return load_dephy_scm_case(path, **kwargs)


def load_dephy_scm_case(
    path: str | Path,
    *,
    nlev: int = 64,
    sigma_top: float | None = None,
    dt: float | None = None,
    dtype=None,
    include_geostrophic: bool = True,
    include_vertical_velocity: bool = True,
    include_advective_tendencies: bool = True,
    include_surface_fluxes: bool = True,
    include_prescribed_surface_temperature: bool = True,
    strict: bool = False,
    rename: dict[str, str] | None = None,
) -> DEPHYSCMCase:
    """Load a DEPHY-SCM SCM-format NetCDF file.

    Parameters
    ----------
    path
        Path to a ``*_SCM_driver.nc`` file.
    nlev
        Number of legoESM sigma full levels to initialize.
    sigma_top
        Sigma value at the model top.  ``None`` infers a top that stays within
        the DEPHY pressure profile when possible.
    dt
        SCM time step [s].  ``None`` uses the DEPHY forcing cadence.
    strict
        Raise if unsupported nonzero DEPHY forcing channels are present.
    rename
        Variable renames applied to the dataset before anything reads it, for
        files that carry the DEPHY structure under different names. STRICT: a
        declared source the file does not contain, or a destination that would
        overwrite an existing variable, raises. Skipping an absent source would
        let a renamed file quietly read through the loader's defaults and still
        produce a column. This is format glue for one file, NOT a general
        format shim: a rename that changed what a variable MEANS (units,
        staggering, sign) would be a silent mis-forcing, so every entry has to
        be justified where it is declared.
    """
    if dtype is None:
        dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

    path = Path(path)
    ds = _open_loaded_dataset(path)
    if rename:
        missing = sorted(k for k in rename if k not in ds.variables)
        if missing:
            # Silently skipping an absent source is how a rename stops
            # happening without anyone noticing: the file then reads through
            # the loader's DEFAULTS -- no surface forcing, no pressure profile
            # -- and still produces a column that looks plausible. Fail here,
            # where the message names the file.
            raise ValueError(
                f"{path.name}: rename declares source variable(s) {missing} "
                f"that the file does not contain (it has "
                f"{sorted(ds.variables)[:12]}...). The upstream file has "
                "changed, or the registry entry is for a different file."
            )
        collide = sorted(v for v in rename.values() if v in ds.variables)
        if collide:
            raise ValueError(
                f"{path.name}: rename would overwrite existing variable(s) "
                f"{collide}."
            )
        ds = ds.rename(dict(rename))
    try:
        time = _time_axis(ds)
        p_s = float(
            _scalar_var(
                ds,
                "ps",
                fallback=_scalar_var(ds, "ps_forc", constants.p_ref),
            )
        )
        pa_init = _profile_var(ds, "pa")

        if sigma_top is None:
            sigma_top = max(0.01, float(np.nanmin(pa_init) / p_s))
        sigma_top = float(sigma_top)

        sigma_coord = create_sigma_coordinate(nlev, sigma_top=sigma_top, dtype=dtype)
        p_target = np.asarray(
            sigma_coord.pressure_at_full(jnp.asarray(p_s, dtype=dtype)),
            dtype=np.float64,
        )

        T_profile_np = _initial_temperature(ds, p_target)
        q_profile_np, q_source, q_specific_np = _initial_moisture(ds, p_target)
        u_profile_np = _initial_profile_or_zero(ds, "ua", p_target)
        v_profile_np = _initial_profile_or_zero(ds, "va", p_target)

        pa_forc = _forcing_pressure(ds, time, pa_init)
        notes: list[str] = []
        unsupported = _detect_unsupported(ds, p_s)

        theta_adv_values = None
        if include_advective_tendencies:
            theta_adv_values = _theta_tendency_values(
                ds, time, p_target, pa_forc, notes, strict=strict,
            )

        qv_adv_values = None
        if include_advective_tendencies:
            qv_adv_values = _moisture_tendency_values(
                ds, time, p_target, pa_forc, q_specific_np, notes, strict=strict,
            )

        subsidence_values = None
        if include_vertical_velocity:
            subsidence_values = _vertical_velocity_values(
                ds, time, p_target, pa_forc, T_profile_np, q_profile_np,
                notes,
            )

        u_geo_values = None
        v_geo_values = None
        if include_geostrophic and _has_var(ds, "ug") and _has_var(ds, "vg"):
            u_geo_values = _interp_forcing(ds, "ug", time, p_target, pa_forc)
            v_geo_values = _interp_forcing(ds, "vg", time, p_target, pa_forc)

        surface_kwargs = {}
        if include_surface_fluxes or include_prescribed_surface_temperature:
            surface_kwargs = _surface_forcing_kwargs(
                ds,
                time,
                p_target,
                T_profile_np,
                q_profile_np,
                q_specific_np,
                include_surface_fluxes=include_surface_fluxes,
                include_prescribed_surface_temperature=(
                    include_prescribed_surface_temperature
                ),
                notes=notes,
            )

        if strict and unsupported:
            raise ValueError(
                "DEPHY case contains unsupported forcing channels: "
                + "; ".join(unsupported)
            )

        latitude = _scalar_var(ds, "lat", 0.0)
        longitude = _scalar_var(ds, "lon", 0.0)
        f_c = 0.0
        if include_geostrophic and u_geo_values is not None:
            f_c = float(2.0 * constants.Omega * np.sin(np.deg2rad(latitude)))

        forcing = SCMForcing(
            f_c=f_c,
            u_geo=(
                _profile_time_fn(time, u_geo_values, dtype)
                if u_geo_values is not None else None
            ),
            v_geo=(
                _profile_time_fn(time, v_geo_values, dtype)
                if v_geo_values is not None else None
            ),
            subsidence_w=(
                _profile_time_fn(time, subsidence_values, dtype)
                if subsidence_values is not None else None
            ),
            theta_adv=(
                _profile_time_fn(time, theta_adv_values, dtype)
                if theta_adv_values is not None else None
            ),
            qv_adv=(
                _profile_time_fn(time, qv_adv_values, dtype)
                if qv_adv_values is not None else None
            ),
            **surface_kwargs,
        )

        if surface_kwargs.get("prescribe") == "fluxes":
            wind_mode = str(ds.attrs.get("surface_forcing_wind", "none"))
            if wind_mode not in ("none", ""):
                notes.append(
                    "DEPHY surface wind forcing metadata "
                    f"{wind_mode!r} is not a momentum channel in SCMForcing; "
                    "heat and moisture surface fluxes were mapped."
                )

        q_profile = None
        if q_profile_np is not None:
            q_profile = jnp.asarray(q_profile_np, dtype=dtype)

        orog = _scalar_var(ds, "orog", 0.0)
        if q_source == "qv":
            notes.append("DEPHY qv specific humidity was converted to mixing ratio.")
        elif q_source == "qt":
            notes.append(
                "DEPHY qt total-water specific humidity was converted to "
                "mixing ratio and used as q_v."
            )
        elif q_source == "rt":
            notes.append("DEPHY rt total-water mixing ratio was used as q_v.")

        dt_out = float(dt) if dt is not None else _forcing_cadence(time)
        duration = float(time[-1] - time[0]) if time.size > 1 else 0.0

        return DEPHYSCMCase(
            path=path,
            case_id=str(ds.attrs.get("case", path.stem)),
            title=str(ds.attrs.get("title", "")),
            surface_type=str(ds.attrs.get("surface_type", "")),
            radiation=str(ds.attrs.get("radiation", "")),
            start_date=str(ds.attrs.get("start_date", "")),
            end_date=str(ds.attrs.get("end_date", "")),
            latitude_deg=float(latitude),
            longitude_deg=float(longitude),
            orography_m=float(orog),
            sigma_top=sigma_top,
            dt=dt_out,
            duration_seconds=duration,
            time_seconds=np.asarray(time, dtype=np.float64),
            pressure_full=jnp.asarray(p_target, dtype=dtype),
            T_profile=jnp.asarray(T_profile_np, dtype=dtype),
            q_v_profile=q_profile,
            u_profile=jnp.asarray(u_profile_np, dtype=dtype),
            v_profile=jnp.asarray(v_profile_np, dtype=dtype),
            p_s=p_s,
            phis=float(constants.g * orog),
            forcing=forcing,
            notes=tuple(dict.fromkeys(notes)),
            unsupported=tuple(dict.fromkeys(unsupported)),
        )
    finally:
        close = getattr(ds, "close", None)
        if callable(close):
            close()


def _open_loaded_dataset(path: Path):
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - dependency in pyproject
        raise ImportError(
            "load_dephy_scm_case requires xarray to read DEPHY NetCDF files."
        ) from exc
    ds = xr.open_dataset(path, decode_times=False)
    return ds.load()


def _has_var(ds, name: str) -> bool:
    return name in ds.variables


def _array_var(ds, name: str) -> np.ndarray:
    return np.asarray(ds[name].values, dtype=np.float64)


def _squeezed_var(ds, name: str) -> np.ndarray:
    arr = np.asarray(_array_var(ds, name), dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def _time_axis(ds) -> np.ndarray:
    if _has_var(ds, "time"):
        time = _squeezed_var(ds, "time")
    else:
        time = np.asarray([0.0], dtype=np.float64)
    if time.ndim != 1:
        raise ValueError(f"DEPHY time axis must be 1-D, got shape {time.shape}")
    return np.asarray(time, dtype=np.float64)


def _forcing_cadence(time: np.ndarray) -> float:
    if time.size < 2:
        return 300.0
    diffs = np.diff(time)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 300.0
    return float(np.nanmin(diffs))


def _scalar_var(ds, name: str, fallback=None) -> float:
    if not _has_var(ds, name):
        if fallback is None:
            raise KeyError(name)
        return float(fallback)
    arr = _squeezed_var(ds, name)
    return float(arr.flat[0])


def _profile_var(ds, name: str) -> np.ndarray:
    arr = _squeezed_var(ds, name)
    if arr.ndim == 1:
        return np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return np.asarray(arr[0], dtype=np.float64)
    raise ValueError(f"DEPHY variable {name!r} is not a single profile: {arr.shape}")


def _profile_time_var(ds, name: str, ntime: int) -> np.ndarray:
    arr = _squeezed_var(ds, name)
    if arr.ndim == 1:
        return np.broadcast_to(arr.reshape(1, -1), (ntime, arr.size)).copy()
    if arr.ndim != 2:
        raise ValueError(f"DEPHY variable {name!r} must be 1-D or 2-D")
    if arr.shape[0] == ntime:
        return np.asarray(arr, dtype=np.float64)
    if arr.shape[1] == ntime:
        return np.asarray(arr.T, dtype=np.float64)
    if arr.shape[0] == 1:
        return np.broadcast_to(arr, (ntime, arr.shape[1])).copy()
    raise ValueError(
        f"DEPHY variable {name!r} shape {arr.shape} does not match time={ntime}"
    )


def _interp_profile_to_pressure(
    p_target: np.ndarray,
    p_source: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    p = np.asarray(p_source, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if p.size != v.size:
        raise ValueError(
            f"pressure/profile size mismatch: pressure={p.size}, values={v.size}"
        )
    mask = np.isfinite(p) & np.isfinite(v)
    if np.count_nonzero(mask) < 2:
        raise ValueError("at least two finite pressure/profile samples are required")
    p = p[mask]
    v = v[mask]
    order = np.argsort(p)
    p = p[order]
    v = v[order]
    p_unique, idx = np.unique(p, return_index=True)
    v_unique = v[idx]
    return np.interp(p_target, p_unique, v_unique)


def _interp_initial(ds, name: str, p_target: np.ndarray) -> np.ndarray:
    return _interp_profile_to_pressure(p_target, _profile_var(ds, "pa"), _profile_var(ds, name))


def _initial_temperature(ds, p_target: np.ndarray) -> np.ndarray:
    if _has_var(ds, "ta"):
        return _interp_initial(ds, "ta", p_target)
    if _has_var(ds, "theta"):
        theta = _interp_initial(ds, "theta", p_target)
        exner = np.asarray(exner_function(jnp.asarray(p_target)), dtype=np.float64)
        return theta * exner
    if _has_var(ds, "thetal"):
        theta = _interp_initial(ds, "thetal", p_target)
        exner = np.asarray(exner_function(jnp.asarray(p_target)), dtype=np.float64)
        return theta * exner
    raise ValueError("DEPHY SCM file must contain ta, theta, or thetal")


def _initial_moisture(
    ds,
    p_target: np.ndarray,
) -> tuple[np.ndarray | None, str, np.ndarray | None]:
    if _has_var(ds, "rv"):
        r = np.maximum(_interp_initial(ds, "rv", p_target), 0.0)
        q = np.asarray(mixing_ratio_to_specific_humidity(jnp.asarray(r)))
        return r, "rv", q
    if _has_var(ds, "rt"):
        r = np.maximum(_interp_initial(ds, "rt", p_target), 0.0)
        q = np.asarray(mixing_ratio_to_specific_humidity(jnp.asarray(r)))
        return r, "rt", q
    if _has_var(ds, "qv"):
        q = np.maximum(_interp_initial(ds, "qv", p_target), 0.0)
        r = np.asarray(specific_humidity_to_mixing_ratio(jnp.asarray(q)))
        return r, "qv", q
    if _has_var(ds, "qt"):
        q = np.maximum(_interp_initial(ds, "qt", p_target), 0.0)
        r = np.asarray(specific_humidity_to_mixing_ratio(jnp.asarray(q)))
        return r, "qt", q
    return None, "", None


def _initial_profile_or_zero(ds, name: str, p_target: np.ndarray) -> np.ndarray:
    if not _has_var(ds, name):
        return np.zeros_like(p_target, dtype=np.float64)
    return _interp_initial(ds, name, p_target)


def _forcing_pressure(ds, time: np.ndarray, pa_init: np.ndarray) -> np.ndarray:
    if _has_var(ds, "pa_forc"):
        return _profile_time_var(ds, "pa_forc", time.size)
    return np.broadcast_to(pa_init.reshape(1, -1), (time.size, pa_init.size)).copy()


def _interp_forcing(
    ds,
    name: str,
    time: np.ndarray,
    p_target: np.ndarray,
    pa_forc: np.ndarray,
) -> np.ndarray:
    data = _profile_time_var(ds, name, time.size)
    return np.stack(
        [
            _interp_profile_to_pressure(p_target, pa_forc[it], data[it])
            for it in range(time.size)
        ],
        axis=0,
    )


def _condensate_present(ds) -> bool:
    for name in ("ql", "qi", "rl", "ri"):
        if _has_var(ds, name) and _nonzero_array(_squeezed_var(ds, name)):
            return True
    return False


def _theta_like_tendency(
    ds,
    candidates: tuple[str, ...],
    time: np.ndarray,
    p_target: np.ndarray,
    pa_forc: np.ndarray,
    notes: list[str],
    *,
    strict: bool,
) -> np.ndarray | None:
    for name in candidates:
        if not _has_var(ds, name):
            continue
        values = _interp_forcing(ds, name, time, p_target, pa_forc)
        if name.startswith("tnta_"):
            exner = np.asarray(exner_function(jnp.asarray(p_target)), dtype=np.float64)
            values = values / exner.reshape(1, -1)
            notes.append(f"DEPHY {name} temperature tendency was converted to theta.")
        elif "thetal" in name and _condensate_present(ds):
            msg = (
                f"DEPHY {name} is a liquid-potential-temperature tendency; "
                "SCMForcing has only theta_adv."
            )
            if strict:
                raise ValueError(msg)
            notes.append(msg + " It was mapped as theta because no condensate channel exists.")
        return values
    return None


def _theta_tendency_values(
    ds,
    time: np.ndarray,
    p_target: np.ndarray,
    pa_forc: np.ndarray,
    notes: list[str],
    *,
    strict: bool,
) -> np.ndarray | None:
    pieces = []
    adv = _theta_like_tendency(
        ds,
        ("tntheta_adv", "tnthetal_adv", "tnta_adv"),
        time,
        p_target,
        pa_forc,
        notes,
        strict=strict,
    )
    if adv is not None:
        pieces.append(adv)

    radiation_mode = str(ds.attrs.get("radiation", "")).lower()
    if radiation_mode == "tend":
        rad = _theta_like_tendency(
            ds,
            ("tntheta_rad", "tnthetal_rad", "tnta_rad"),
            time,
            p_target,
            pa_forc,
            notes,
            strict=strict,
        )
        if rad is not None:
            pieces.append(rad)
            notes.append(
                "DEPHY prescribed radiative tendency was added to theta_adv; "
                "disable active radiation to avoid double-counting."
            )

    if not pieces:
        return None
    return np.sum(np.stack(pieces, axis=0), axis=0)


def _moisture_tendency_values(
    ds,
    time: np.ndarray,
    p_target: np.ndarray,
    pa_forc: np.ndarray,
    q_specific: np.ndarray | None,
    notes: list[str],
    *,
    strict: bool,
) -> np.ndarray | None:
    if _has_var(ds, "tnrv_adv"):
        return _interp_forcing(ds, "tnrv_adv", time, p_target, pa_forc)
    if _has_var(ds, "tnrt_adv"):
        if _condensate_present(ds):
            msg = (
                "DEPHY tnrt_adv is a total-water mixing-ratio tendency; "
                "SCMForcing has only qv_adv."
            )
            if strict:
                raise ValueError(msg)
            notes.append(msg + " It was mapped as qv_adv.")
        return _interp_forcing(ds, "tnrt_adv", time, p_target, pa_forc)

    for name in ("tnqv_adv", "tnqt_adv"):
        if not _has_var(ds, name):
            continue
        if q_specific is None:
            raise ValueError(f"{name} requires qv/qt initial profile for conversion")
        values_q = _interp_forcing(ds, name, time, p_target, pa_forc)
        values_r = np.asarray(
            specific_humidity_tendency_to_mixing_ratio_tendency(
                jnp.asarray(q_specific).reshape(1, -1),
                jnp.asarray(values_q),
            ),
            dtype=np.float64,
        )
        notes.append(f"DEPHY {name} specific-humidity tendency was converted to mixing ratio.")
        return values_r
    return None


def _vertical_velocity_values(
    ds,
    time: np.ndarray,
    p_target: np.ndarray,
    pa_forc: np.ndarray,
    T_profile: np.ndarray,
    q_profile: np.ndarray | None,
    notes: list[str],
) -> np.ndarray | None:
    if _has_var(ds, "wa"):
        return _interp_forcing(ds, "wa", time, p_target, pa_forc)
    if not _has_var(ds, "wap"):
        return None
    omega = _interp_forcing(ds, "wap", time, p_target, pa_forc)
    q = np.zeros_like(T_profile) if q_profile is None else q_profile
    rho = np.asarray(
        compute_rho(
            jnp.asarray(T_profile).reshape(1, -1),
            jnp.asarray(p_target).reshape(1, -1),
            jnp.asarray(q).reshape(1, -1),
        )[0],
        dtype=np.float64,
    )
    notes.append("DEPHY omega (wap) was converted to upward w using hydrostatic density.")
    return -omega / (rho.reshape(1, -1) * constants.g)


def _surface_forcing_kwargs(
    ds,
    time: np.ndarray,
    p_target: np.ndarray,
    T_profile: np.ndarray,
    q_profile: np.ndarray | None,
    q_specific: np.ndarray | None,
    *,
    include_surface_fluxes: bool,
    include_prescribed_surface_temperature: bool,
    notes: list[str],
) -> dict[str, Any]:
    temp_mode = str(ds.attrs.get("surface_forcing_temp", "none")).lower()
    moist_mode = str(ds.attrs.get("surface_forcing_moisture", "none")).lower()
    q = np.zeros_like(T_profile) if q_profile is None else q_profile
    rho_sfc = float(
        np.asarray(
            compute_rho(
                jnp.asarray(T_profile[-1]),
                jnp.asarray(p_target[-1]),
                jnp.asarray(q[-1]),
            )
        )
    )

    if include_surface_fluxes and (
        temp_mode == "surface_flux" or moist_mode == "surface_flux"
    ):
        w_th = None
        w_qv = None
        if _has_var(ds, "wpthetap_s"):
            exner_sfc = float(np.asarray(exner_function(jnp.asarray(p_target[-1]))))
            vals = _scalar_series(ds, "wpthetap_s", time) * exner_sfc
            w_th = _scalar_time_fn(time, vals)
            notes.append(
                "DEPHY wpthetap_s potential-temperature flux was converted "
                "to temperature flux for SCMForcing."
            )
        elif _has_var(ds, "hfss"):
            vals = _scalar_series(ds, "hfss", time) / (rho_sfc * constants.c_pd)
            w_th = _scalar_time_fn(time, vals)

        if _has_var(ds, "wprvp_s"):
            w_qv = _scalar_time_fn(time, _scalar_series(ds, "wprvp_s", time))
        elif _has_var(ds, "wpqvp_s"):
            vals = _scalar_series(ds, "wpqvp_s", time)
            if q_specific is not None:
                denom = max((1.0 - float(q_specific[-1])) ** 2, 1.0e-12)
                vals = vals / denom
            notes.append("DEPHY wpqvp_s specific-humidity flux was converted to mixing-ratio flux.")
            w_qv = _scalar_time_fn(time, vals)
        elif _has_var(ds, "hfls"):
            vals = _scalar_series(ds, "hfls", time) / (rho_sfc * constants.L_v)
            w_qv = _scalar_time_fn(time, vals)

        kwargs: dict[str, Any] = {"prescribe": "fluxes"}
        if w_th is not None:
            kwargs["w_th_s"] = w_th
        if w_qv is not None:
            kwargs["w_qv_s"] = w_qv
        if len(kwargs) > 1:
            return kwargs

    if include_prescribed_surface_temperature:
        if temp_mode == "ts" and _has_var(ds, "ts_forc"):
            return {
                "prescribe": "T_s",
                "T_s": _scalar_time_fn(time, _scalar_series(ds, "ts_forc", time)),
            }
        if temp_mode == "thetas" and _has_var(ds, "thetas_forc"):
            exner_sfc = float(np.asarray(exner_function(jnp.asarray(p_target[-1]))))
            vals = _scalar_series(ds, "thetas_forc", time) * exner_sfc
            return {"prescribe": "T_s", "T_s": _scalar_time_fn(time, vals)}
        if temp_mode not in ("surface_flux", "none", "") and _has_var(ds, "tskin"):
            notes.append(
                f"DEPHY surface_forcing_temp={temp_mode!r} has no direct "
                "mapper; tskin was used as prescribed T_s."
            )
            return {
                "prescribe": "T_s",
                "T_s": _scalar_time_fn(time, _scalar_series(ds, "tskin", time)),
            }

    return {}


def _scalar_series(ds, name: str, time: np.ndarray) -> np.ndarray:
    arr = _squeezed_var(ds, name)
    if arr.ndim != 1:
        raise ValueError(f"DEPHY scalar forcing {name!r} has shape {arr.shape}")
    if arr.size == time.size:
        return np.asarray(arr, dtype=np.float64)
    if arr.size == 1:
        return np.full(time.size, float(arr[0]), dtype=np.float64)
    raise ValueError(
        f"DEPHY scalar forcing {name!r} length {arr.size} does not match time={time.size}"
    )


def _profile_time_fn(time: np.ndarray, values: np.ndarray, dtype):
    time_np = np.asarray(time, dtype=np.float64)
    values_np = np.asarray(values, dtype=np.float64)

    def fn(t_seconds: float):
        t = float(t_seconds)
        out = np.empty(values_np.shape[1], dtype=np.float64)
        for k in range(values_np.shape[1]):
            out[k] = np.interp(t, time_np, values_np[:, k])
        return jnp.asarray(out, dtype=dtype)

    return fn


def _scalar_time_fn(time: np.ndarray, values: np.ndarray):
    time_np = np.asarray(time, dtype=np.float64)
    values_np = np.asarray(values, dtype=np.float64)

    def fn(t_seconds: float):
        return jnp.asarray(np.interp(float(t_seconds), time_np, values_np))

    return fn


def _nonzero_array(arr: np.ndarray) -> bool:
    if arr.size == 0:
        return False
    finite = np.asarray(arr, dtype=np.float64)
    if not np.any(np.isfinite(finite)):
        return False
    return bool(np.nanmax(np.abs(finite)) > 0.0)


def _detect_unsupported(ds, p_s: float) -> tuple[str, ...]:
    unsupported: list[str] = []
    for name, reason in _UNSUPPORTED_PROFILE_FORCINGS.items():
        if _has_var(ds, name) and _nonzero_array(_squeezed_var(ds, name)):
            unsupported.append(f"{name}: {reason}")
    if _has_var(ds, "ps_forc"):
        ps_forc = _squeezed_var(ds, "ps_forc")
        if ps_forc.size > 1 and np.nanmax(np.abs(ps_forc - p_s)) > 1.0e-6:
            unsupported.append(
                "ps_forc: time-varying surface-pressure forcing is not "
                "represented in the current atmospheric SCM"
            )
    return tuple(unsupported)
