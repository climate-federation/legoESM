"""Drive legoESM's CLM-ML-JAX adapter offline at a FLUXNET pilot site.

Purpose (E3 / E6a of docs/MLC_experiment_plan/experiment_plan.md).
-----------------------------------------------------------------

Reads the slim ONEFlux FLUXNET FULLSET CSV at
``data/fluxnet/<SITE>/*_slim.csv`` (already filtered to the target window
by :mod:`legoesm.land.boundary_data.fluxnet_forcing`), loops over the
half-hourly (or hourly) FLUXNET timesteps calling
:func:`legoesm.land.canopy.clm_ml_interface.compute_clm_ml_canopy_fluxes`
with adapter overrides matching the CHATS7 Fortran namelist
(``met_type=3``, ``runge_kutta_type=41``, ``num_ml_steps=30`` -> dtime_ml=60 s),
and writes:

* ``<TAG>_flux.out``, ``<TAG>_fsun.out``, ``<TAG>_aux.out``: same schema
  as the CHATS7 runner + Fortran offline driver so
  ``scripts/validate/validate_clm_ml_canopy.py`` can diff them.
* ``obs_targets.csv``: per-timestep observed LE/H/USTAR/NETRAD for E3/E6a
  validation + training targets.  Missing values are NaN.

Site parameters live in :data:`legoesm.land.boundary_data.fluxnet_forcing.SITES`
(US-MMS DBF, FI-Hyy ENF, US-Ton SAV).

Usage
-----
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run/run_fluxnet_offline.py \
        --site US-MMS --output-dir results/us_mms_2012_full \
        [--start-date 2012-06-14] [--end-date 2012-07-14] \
        [--spinup-days 90] \
        [--met-type 3] [--num-ml-steps 30] [--runge-kutta-type 41]

Notes
-----
* Time-varying LAI is NOT wired here yet -- we use ``lai_peak`` from the
  site metadata for every step, matching Phase 3 §9 rule 3 (Xu-Ri LAI
  climatology / static peak).  Full time-varying LAI is a TODO for E6a
  Phase 6.
* FI-Hyy: Nov-Apr is excluded from the analysis by default (Phase 3 §9
  scope-out: "no snow-covered canopy").  Pass ``--allow-snow`` to keep
  the full window.
* US-Ton: the site mixes C3 (oak) + C4 (grass); we run C3 physics only
  (``pft_clm=7`` from site metadata) and label results "C3-only" per
  Phase 3 §9 rule 4.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time as time_mod
from types import SimpleNamespace
from typing import IO

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_pkg_root = REPO_ROOT / "packages"
_paths = [REPO_ROOT / "src"]
_paths += [p for p in sorted(_pkg_root.iterdir()) if p.is_dir() and (p / "legoesm").exists()]
_paths.append(REPO_ROOT / "clm-ml-jax" / "src")
for _p in _paths:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# --------------------------------------------------------------------------
# netCDF4 shim: on Apple Silicon + Python 3.14 the pre-built netCDF4 wheel is
# x86_64 and refuses to load. clm-ml-jax's ``LookupPsihatINI`` uses only
# ``nc.Dataset(path, "r").dimensions[...]`` and ``.variables[...][:]``,
# which scipy.io.netcdf covers exactly.  Install the shim once, at import
# time, BEFORE any ``from clm_ml_jax...`` import triggers netCDF4 loading.
# Copied VERBATIM from ``scripts/run/run_chats7_offline.py`` (per CLAUDE.md
# doctrine: no re-derivation of shared load-side scaffolding).
try:
    import netCDF4 as _nc  # noqa: F401
except ImportError:
    import scipy.io as _sio
    import types as _types

    class _NcDim:
        def __init__(self, length: int):
            self._length = int(length)

        def __len__(self):
            return self._length

    class _NcDataset:
        def __init__(self, path: str, mode: str = "r"):
            self._f = _sio.netcdf_file(str(path), mode)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._f.close()

        @property
        def dimensions(self):
            return {k: _NcDim(v) for k, v in self._f.dimensions.items()}

        @property
        def variables(self):
            return {k: v for k, v in self._f.variables.items()}

    _shim = _types.ModuleType("netCDF4")
    _shim.Dataset = _NcDataset
    sys.modules["netCDF4"] = _shim

# NOTE: these imports MUST come after the netCDF4 shim installation above.
from legoesm.land.boundary_data.fluxnet_forcing import (  # noqa: E402
    SITES,
    FluxnetForcing,
    FluxnetSite,
    build_atm2sfc_at,
    load_fluxnet_forcing,
)

# Reuse writer helpers from the CHATS7 runner verbatim -- do not duplicate
# (CLAUDE.md: "No duplicate numerics across dycores/physics/grids/tests.").
# ``run_chats7_offline`` lives in the same ``scripts/run`` directory; the
# sys.path bootstrap above does not add scripts/, so import via a
# module-file loader.
import importlib.util  # noqa: E402

_CHATS7_PATH = pathlib.Path(__file__).with_name("run_chats7_offline.py")
_spec = importlib.util.spec_from_file_location("_run_chats7_offline", _CHATS7_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load writers from {_CHATS7_PATH}")
_chats7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chats7)
_write_flux_row = _chats7._write_flux_row
_write_fsun_row = _chats7._write_fsun_row
_write_aux_row = _chats7._write_aux_row


# ---------------------------------------------------------------------------
# Site-specific scope filters
# ---------------------------------------------------------------------------

# FI-Hyy snow-season months to exclude by default (Phase 3 §9 scope-out
# "no snow-covered canopy"; ``--allow-snow`` overrides).  Northern-hemisphere
# November through April.
_FIHYY_SNOW_MONTHS = frozenset({11, 12, 1, 2, 3, 4})


# ---------------------------------------------------------------------------
# Config assembly (mirrors CHATS7 runner)
# ---------------------------------------------------------------------------


def _build_land_params(
    site: FluxnetSite,
    lai_override: float | None = None,
    sai_override: float | None = None,
    htop_override: float | None = None,
):
    """Minimal ``LandSurfaceParams``-like object (LAI + SAI + htop).

    Uses the site's peak LAI (``lai_peak``) for every step -- time-varying
    LAI is a TODO for E6a Phase 6.

    The ``*_override`` kwargs let a caller substitute conservative canopy
    geometry (e.g. CHATS7-safe LAI=2, SAI=0.7, htop=10) at sites whose
    default PFT beta-distribution parameters produce zero-PAI layers in
    ``initVerticalStructure``.  Physics-fidelity to the site is reduced
    but the pipeline is unblocked for a POC.
    """
    lai = float(site.lai_peak if lai_override is None else lai_override)
    sai = float(site.sai if sai_override is None else sai_override)
    htop = float(site.htop_m if htop_override is None else htop_override)
    return SimpleNamespace(
        LAI=jnp.full((1,), lai),
        SAI=jnp.full((1,), sai),
        htop=jnp.full((1,), htop),
    )


def _build_land_config(
    site: FluxnetSite,
    met_type: int,
    runge_kutta_type: int,
    num_ml_steps: int,
):
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.soil_grid import SoilGridConfig

    canopy = CLMMLCanopyConfig(
        pft_clm=int(site.pft_clm),
        met_type=int(met_type),
        runge_kutta_type=int(runge_kutta_type),
        num_ml_steps=int(num_ml_steps),
    )
    land = MultiLayerLandConfig(
        surface_scheme=canopy,
        soil_grid=SoilGridConfig(),
        z_ref=float(site.z_ref_m),
    )
    return land


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _time_to_doy_frac(ts: np.datetime64) -> float:
    """0-based fractional day-of-year in UTC (0.0 = Jan 1 00:00).

    Same convention as :func:`run_chats7_offline._time_to_doy_frac`.
    """
    ts_dt = ts.astype("datetime64[s]").astype(object)
    year_start = np.datetime64(f"{ts_dt.year}-01-01T00:00:00", "s")
    seconds = (ts.astype("datetime64[s]") - year_start).astype("int64").item()
    return seconds / 86400.0  # coeff-ok: seconds-per-day (24*3600)


def _slice_window(
    forcing: FluxnetForcing,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> np.ndarray:
    """Return boolean mask of rows within ``[start_date, end_date]``.

    Dates are compared in the forcing's LOCAL-standard-time index (that
    is the ``FluxnetForcing.time`` convention).  ``end_date`` is
    INCLUSIVE at midnight-start of that day + 24h (so
    ``--end-date 2012-07-14`` picks up all rows on 2012-07-14).
    """
    n = len(forcing.time)
    mask = np.ones(n, dtype=bool)
    # ``forcing.time`` may be a pandas.DatetimeIndex or a numpy.ndarray of
    # datetime64[ns].  Coerce to DatetimeIndex once so ``>=`` / ``<`` return
    # numpy bool arrays (avoiding the ``.to_numpy()`` accessor).
    t = pd.DatetimeIndex(forcing.time)
    if start_date is not None:
        mask &= np.asarray(t >= start_date)
    if end_date is not None:
        end_excl = end_date + pd.Timedelta(days=1)
        mask &= np.asarray(t < end_excl)
    return mask


def _apply_snow_filter(
    forcing: FluxnetForcing, mask: np.ndarray, allow_snow: bool,
) -> tuple[np.ndarray, int]:
    """Drop FI-Hyy Nov-Apr rows unless ``allow_snow`` is set.

    Returns updated mask + count of rows dropped.
    """
    if allow_snow or forcing.site.code != "FI-Hyy":
        return mask, 0
    months = np.asarray(pd.DatetimeIndex(forcing.time).month)
    snow_month = np.isin(months, list(_FIHYY_SNOW_MONTHS))
    dropped = int((mask & snow_month).sum())
    mask = mask & ~snow_month
    return mask, dropped


# ---------------------------------------------------------------------------
# obs_targets.csv writer
# ---------------------------------------------------------------------------


def _open_obs_targets(path: pathlib.Path) -> tuple[IO, csv.writer]:
    """Open ``obs_targets.csv`` with the E3/E6a schema and write header."""
    handle = open(path, "w", newline="")
    writer = csv.writer(handle)
    writer.writerow(["time", "LE_obs", "H_obs", "USTAR_obs", "NETRAD_obs"])
    return handle, writer


def _fmt_obs(v: float) -> str:
    """Format an observation value, propagating NaN as ``NaN`` (not blank)."""
    fv = float(v)
    if not np.isfinite(fv):
        return "NaN"
    return f"{fv:.6g}"


def _write_obs_row(
    writer: csv.writer, ts: pd.Timestamp,
    LE: float, H: float, ustar: float, NETRAD: float,
) -> None:
    writer.writerow([
        ts.isoformat(),
        _fmt_obs(LE),
        _fmt_obs(H),
        _fmt_obs(ustar),
        _fmt_obs(NETRAD),
    ])


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _default_csv_for_site(site_code: str) -> pathlib.Path:
    """Return the in-tree slim FLUXNET CSV path for ``site_code``.

    Filenames follow the ONEFlux release naming (``{tag}_slim.csv``); we
    resolve them by globbing ``data/fluxnet/<site>/*_slim.csv`` so the
    runner keeps working across ONEFlux re-releases.
    """
    site_dir = REPO_ROOT / "data" / "fluxnet" / site_code
    matches = sorted(site_dir.glob("*_slim.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No slim FLUXNET CSV found for site {site_code!r} at {site_dir}. "
            "Expected data/fluxnet/<SITE>/*_slim.csv."
        )
    return matches[0]


def run_fluxnet(
    site_code: str,
    output_dir: pathlib.Path,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    spinup_days: int = 0,
    met_type: int = 3,
    runge_kutta_type: int = 41,
    num_ml_steps: int = 30,
    allow_snow: bool = False,
    csv_path: pathlib.Path | None = None,
    lai_override: float | None = None,
    sai_override: float | None = None,
    htop_override: float | None = None,
    relaxed_qc: bool = False,
) -> None:
    """Run the FLUXNET offline driver and write .out + obs_targets.csv."""
    from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
    from legoesm.land.canopy.state import CanopyState
    from legoesm.land.soil_hydraulics import psi_from_theta

    if site_code not in SITES:
        raise ValueError(
            f"Unknown FLUXNET site_code={site_code!r}; "
            f"expected one of {sorted(SITES)}"
        )
    site = SITES[site_code]
    csv_path = csv_path or _default_csv_for_site(site_code)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fluxnet:{site_code}] output -> {output_dir}", flush=True)
    print(f"[fluxnet:{site_code}] loading {csv_path.name}", flush=True)

    forcing = load_fluxnet_forcing(csv_path, site_code)
    if relaxed_qc:
        # POC only: treat every finite-forcing row as valid, regardless of the
        # gap-fill QC flag.  Some sites (e.g. US-Ton) have mostly gap-filled met
        # in certain years, so the strict QC=0 mask marks everything invalid.
        relaxed = (
            np.isfinite(forcing.T_bot_K)
            & np.isfinite(forcing.sw_down)
            & np.isfinite(forcing.lw_down)
            & np.isfinite(forcing.ws)
            & np.isfinite(forcing.p_surface_Pa)
            & np.isfinite(forcing.precip_kg_m2_s)
            & np.isfinite(forcing.co2_ppmv)
        )
        forcing = forcing._replace(valid=relaxed) if hasattr(forcing, "_replace") \
            else type(forcing)(**{**forcing.__dict__, "valid": relaxed})
    dt = float(forcing.dt_s)
    n_total = len(forcing.time)

    # 1) Window filter.
    window_mask = _slice_window(forcing, start_date, end_date)
    # 2) FI-Hyy snow-season filter (Phase 3 §9 scope-out).
    window_mask, n_snow_dropped = _apply_snow_filter(
        forcing, window_mask, allow_snow=allow_snow,
    )
    if n_snow_dropped > 0:
        print(
            f"[fluxnet:{site_code}] dropped {n_snow_dropped} rows in "
            "Nov-Apr snow-covered window (pass --allow-snow to keep)",
            flush=True,
        )
    # 3) Validity filter.
    analysis_mask = window_mask & forcing.valid
    n_analysis = int(analysis_mask.sum())
    n_invalid = int((window_mask & ~forcing.valid).sum())
    print(
        f"[fluxnet:{site_code}] total_rows={n_total}, in_window={int(window_mask.sum())}, "
        f"valid_in_window={n_analysis}, invalid_dropped={n_invalid}, dt={dt}s",
        flush=True,
    )
    if n_analysis == 0:
        raise SystemExit(
            f"[fluxnet:{site_code}] no valid rows in the requested window "
            f"[{start_date}, {end_date}] -- nothing to do."
        )

    # US-Ton C3-only label (Phase 3 §9 rule 4): report the run tag prefix
    # so downstream tooling knows this is C3-physics only.
    is_c3_only = site_code == "US-Ton"
    tag_suffix = "_C3only" if is_c3_only else ""
    run_tag = f"{site_code}{tag_suffix}"
    if is_c3_only:
        print(
            f"[fluxnet:{site_code}] labeling result 'C3-only' (Phase 3 §9 rule 4: "
            "site mixes C3 oak + C4 grass; we run C3 physics via pft_clm=7)",
            flush=True,
        )

    # 4) Spinup selection: last ``spinup_days`` of valid rows BEFORE the
    # analysis window (fall back to warm-restart-free if there aren't
    # enough).  When ``start_date`` is None the analysis starts at the
    # first valid row, so spinup = the first row's forcing repeated is
    # meaningless -- we skip spinup in that case with a warning.
    n_spinup_target = int(spinup_days * (86400.0 / dt))
    analysis_indices = np.flatnonzero(analysis_mask)
    if n_spinup_target > 0:
        first_analysis_i = int(analysis_indices[0])
        # Rows strictly BEFORE the first analysis index, still valid.
        pre_mask = np.zeros_like(analysis_mask)
        pre_mask[:first_analysis_i] = True
        pre_mask &= forcing.valid
        # For FI-Hyy also drop snow months from spinup unless allowed.
        pre_mask, _ = _apply_snow_filter(forcing, pre_mask, allow_snow=allow_snow)
        pre_indices = np.flatnonzero(pre_mask)
        if len(pre_indices) < n_spinup_target:
            print(
                f"[fluxnet:{site_code}] spinup: requested {n_spinup_target} steps "
                f"but only {len(pre_indices)} valid pre-analysis rows exist -- "
                "using all available",
                flush=True,
            )
        spinup_indices = pre_indices[-n_spinup_target:] if len(pre_indices) else pre_indices
    else:
        spinup_indices = np.empty(0, dtype=np.int64)

    # ---------------- Land config / initial soil state ---------------------
    land_config = _build_land_config(
        site=site,
        met_type=met_type,
        runge_kutta_type=runge_kutta_type,
        num_ml_steps=num_ml_steps,
    )
    canopy_config = land_config.surface_scheme
    land_params = _build_land_params(
        site,
        lai_override=lai_override,
        sai_override=sai_override,
        htop_override=htop_override,
    )

    # Bug fix (US-MMS PAI-zero): CLM-ML's ``dpai_min`` (default 0.01) zeroes
    # any canopy layer whose PAI is smaller.  With ``nlevmlcan=100`` module
    # default and a tall canopy (htop=27 m at US-MMS), each layer is only
    # ~0.27 m thick and its natural PAI (from the beta distribution) can
    # fall below 0.01, triggering ``initVerticalStructure: canopy layer has
    # zero plant area index``.  Scale ``dpai_min`` by (10 m / htop) so the
    # threshold matches layer-size / total-PAI at the canopy top height.
    # ``_ensure_clm_initialized`` runs on the first call to the adapter, so
    # setting this AFTER the first call would be too late — we set it now.
    _htop_used = float(htop_override) if htop_override is not None else float(site.htop_m)
    _lai_used = float(lai_override) if lai_override is not None else float(site.lai_peak)
    _sai_used = float(sai_override) if sai_override is not None else float(site.sai)
    _pai = _lai_used + _sai_used
    import multilayer_canopy.MLclm_varctl as _ml_ctl_init
    from multilayer_canopy.MLclm_varpar import nlevmlcan as _nlev_ml
    # Two heuristics for the minimum layer PAI threshold; take the smaller.
    # (a) Geometric: layer thickness ~ htop / nlevmlcan.  Tall canopies with a
    #     fixed nlevmlcan=100 have thin layers that fall below dpai_min=0.01 in
    #     the beta-distribution tails -> zero-PAI error.
    # (b) Total-PAI: mean layer PAI ~ PAI / nlevmlcan.  Sparse canopies (low
    #     total PAI) also produce sub-threshold tails.
    _dpai_geom = 0.01 * min(1.0, 10.0 / max(_htop_used, 1.0))  # coeff-ok: 10 m reference matches CHATS default
    _dpai_pai = _pai / (3.0 * _nlev_ml)  # coeff-ok: 1/3 keeps beta-tail layers alive
    _ml_ctl_init.dpai_min = float(min(_dpai_geom, _dpai_pai, 0.01))
    if _ml_ctl_init.dpai_min < 0.01 - 1e-9:
        print(
            f"[fluxnet:{site_code}] PAI={_pai:.2f} htop={_htop_used:.1f} m: "
            f"lowered dpai_min to {_ml_ctl_init.dpai_min:.4g} "
            f"(default 0.01 would zero beta-tail layers with {_nlev_ml} nlevmlcan).",
            flush=True,
        )

    grid_n_layers = 10
    T_soil0 = jnp.full((1, grid_n_layers), 290.0)
    theta_soil0 = jnp.full((1, grid_n_layers), 0.25)
    psi_soil0 = psi_from_theta(theta_soil0, land_config.hydraulics)

    canopy_state: CanopyState | None = None

    # LST -> UTC hour offset for this site (FLUXNET convention is LST, not UTC).
    # Applies to ``doy`` and to the timestamps written to obs_targets.csv so
    # observations and modelled fluxes align in the diurnal-composite plotter.
    utc_hour_offset = site.lon_deg / 15.0  # coeff-ok: exact 360°/24h = 15°/h
    utc_offset_td = np.timedelta64(int(round(utc_hour_offset * 3600e9)), "ns")

    def _lst_to_utc_doy(ts_lst: np.datetime64) -> float:
        ts_utc = ts_lst - utc_offset_td
        return _time_to_doy_frac(ts_utc)

    def _step(i: int) -> None:
        nonlocal canopy_state
        atm = build_atm2sfc_at(forcing, i)
        ts_lst = forcing.time[i].to_numpy()
        doy_frac_utc = _lst_to_utc_doy(ts_lst)
        _, canopy_state = compute_clm_ml_canopy_fluxes(
            T_soil_top=T_soil0[:, 0],
            forcing=atm,
            canopy_config=canopy_config,
            land_config=land_config,
            land_params=land_params,
            w_frac_rz=jnp.full((1,), 0.6),
            wind_speed=jnp.full((1,), max(float(forcing.ws[i]), 1.0)),
            canopy_state=canopy_state,
            dt=dt,
            T_soil=T_soil0,
            psi_soil=psi_soil0,
            theta_soil=theta_soil0,
            lat=jnp.array([float(site.lat_deg)]),
            doy=doy_frac_utc,
        )

    # ---------------- Spinup pass (no output) ------------------------------
    t0 = time_mod.perf_counter()
    if len(spinup_indices) > 0:
        for k, i in enumerate(spinup_indices):
            _step(int(i))
            if (k + 1) % 48 == 0:
                elapsed = time_mod.perf_counter() - t0
                print(
                    f"[fluxnet:{site_code}] spinup step {k + 1}/{len(spinup_indices)} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )

    # ---------------- Analysis pass (write outputs) ------------------------
    fpath = output_dir / f"{run_tag}_flux.out"
    xpath = output_dir / f"{run_tag}_fsun.out"
    apath = output_dir / f"{run_tag}_aux.out"
    opath = output_dir / "obs_targets.csv"

    # obs_targets.csv is written row-for-row with the .out files: exactly one
    # obs row per MODELED (valid) step, in order.  Both FLUXNET plot loaders
    # pair obs_targets.csv and the .out files by position, so emitting an obs
    # row for a skipped (invalid) step would shift every later comparison onto
    # the wrong observation/timestamp.  (Missing obs targets on a modeled step
    # are still written as NaN.)
    obs_indices_all = np.flatnonzero(window_mask)

    obs_handle, obs_writer = _open_obs_targets(opath)
    with (
        open(fpath, "w") as nout1,
        open(xpath, "w") as nout4,
        open(apath, "w") as nout2,
        obs_handle,
    ):
        # Iterate the window in order; write BOTH a .out row and its matching
        # obs row only for modeled (valid) rows, keeping the files aligned.
        analysis_step_counter = 0
        analysis_valid_set = set(int(x) for x in np.flatnonzero(analysis_mask))
        for i in obs_indices_all:
            i = int(i)
            if i not in analysis_valid_set:
                continue
            ts_lst = forcing.time[i]
            # obs_targets in UTC so the plotter's diurnal composite lines up
            # with the adapter's UTC-based curr_calday.
            ts_utc = pd.Timestamp(ts_lst) - pd.Timedelta(seconds=utc_hour_offset * 3600.0)
            _step(i)
            curr_calday = _lst_to_utc_doy(forcing.time[i].to_numpy()) + 1.0
            mlcan = canopy_state.mlcanopy
            _write_flux_row(nout1, curr_calday, mlcan, dt, met_type=met_type)
            _write_fsun_row(nout4, mlcan)
            _write_aux_row(nout2, mlcan)
            _write_obs_row(
                obs_writer, ts_utc,
                LE=forcing.LE_obs[i],
                H=forcing.H_obs[i],
                ustar=forcing.ustar_obs[i],
                NETRAD=forcing.NETRAD_obs[i],
            )
            nout1.flush(); nout4.flush(); nout2.flush(); obs_handle.flush()
            analysis_step_counter += 1
            if analysis_step_counter % 24 == 0:
                elapsed = time_mod.perf_counter() - t0
                print(
                    f"[fluxnet:{site_code}] analysis step "
                    f"{analysis_step_counter}/{n_analysis} "
                    f"({elapsed:.1f}s wall; SH={float(mlcan.shflx_canopy[1]):.1f} "
                    f"LH={float(mlcan.lhflx_canopy[1]):.1f} W/m²)",
                    flush=True,
                )

    print(
        f"[fluxnet:{site_code}] wrote {fpath.name}, {xpath.name}, {apath.name}, "
        f"{opath.name}",
        flush=True,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CLM-ML-JAX adapter offline at a FLUXNET pilot site.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--site", type=str, required=True, choices=sorted(SITES),
        help="FLUXNET site code (US-MMS, FI-Hyy, US-Ton).",
    )
    p.add_argument(
        "--output-dir", type=pathlib.Path, required=True,
        help="Directory to write flux.out/fsun.out/aux.out/obs_targets.csv.",
    )
    p.add_argument(
        "--csv-path", type=pathlib.Path, default=None,
        help="Override slim CSV path (default: data/fluxnet/<site>/*_slim.csv).",
    )
    p.add_argument(
        "--start-date", type=str, default=None,
        help="Analysis window start (YYYY-MM-DD, LST). Default: first row.",
    )
    p.add_argument(
        "--end-date", type=str, default=None,
        help="Analysis window end (YYYY-MM-DD, LST; INCLUSIVE). Default: last row.",
    )
    p.add_argument(
        "--spinup-days", type=int, default=0,
        help="Number of pre-analysis days of spinup (valid rows, no output).",
    )
    p.add_argument(
        "--met-type", type=int, default=3,
        help="CLM-ML met_type (3 = centred, matches CHATS7 nl.CHATS7.1day).",
    )
    p.add_argument(
        "--num-ml-steps", type=int, default=30,
        help="CLM-ML sub-steps per FLUXNET timestep (30 -> dtime_ml=60s for HH).",
    )
    p.add_argument(
        "--runge-kutta-type", type=int, default=41,
        help="Runge-Kutta scheme id (41 = RK4, standalone default).",
    )
    p.add_argument(
        "--allow-snow", action="store_true",
        help="FI-Hyy: keep Nov-Apr rows (default drops them per Phase 3 §9).",
    )
    p.add_argument(
        "--lai", type=float, default=None,
        help="Override site LAI (POC: use CHATS7-safe 2.0 if the site default "
             "trips 'zero PAI' in initVerticalStructure).",
    )
    p.add_argument(
        "--sai", type=float, default=None,
        help="Override site SAI.",
    )
    p.add_argument(
        "--htop", type=float, default=None,
        help="Override site canopy height in m.",
    )
    p.add_argument(
        "--relaxed-qc", action="store_true",
        help="Only require FINITE forcing inputs — skip the QC=0 "
             "(observed-only, not gap-filled) constraint.  Recommended for "
             "any site-year where gap-filled met is common (e.g. US-Ton in "
             "2012 has ~all rows QC>=1); the tower observations against which "
             "we validate are still the observed values.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    start = pd.Timestamp(args.start_date) if args.start_date else None
    end = pd.Timestamp(args.end_date) if args.end_date else None
    run_fluxnet(
        site_code=args.site,
        output_dir=args.output_dir,
        start_date=start,
        end_date=end,
        spinup_days=int(args.spinup_days),
        met_type=int(args.met_type),
        runge_kutta_type=int(args.runge_kutta_type),
        num_ml_steps=int(args.num_ml_steps),
        allow_snow=bool(args.allow_snow),
        csv_path=args.csv_path,
        lai_override=args.lai,
        sai_override=args.sai,
        htop_override=args.htop,
        relaxed_qc=bool(args.relaxed_qc),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
