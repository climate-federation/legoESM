"""CMIP-style experiment templates for legoESM.

Provides predefined experiment configurations following CMIP6 protocol
conventions.  Each template encodes the forcing type (fixed vs transient),
time span, parent experiment, and baseline greenhouse-gas concentrations.

Built-in GHG concentration time series for historical, SSP2-4.5 and
SSP5-8.5 scenarios allow the model to run without external data files.

Public API
----------
- ``ExperimentTemplate`` : experiment metadata NamedTuple
- ``EXPERIMENT_TEMPLATES`` : dict of built-in templates
- ``ghg_at_year(experiment_name, year)`` : interpolated (co2, ch4, n2o)
- ``get_ghg_for_experiment(name, year)`` : same, returned as a dict
- ``create_experiment_config(name, **overrides)`` : build an
  ``ExperimentConfig`` from a template (canonical runtime config)
- ``create_amip_experiment_config(name, **overrides)`` : legacy wrapper
  returning ``AMIPExperimentConfig`` for backward compatibility
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

from legoesm.forcing.amip_config import AMIPExperimentConfig

logger = logging.getLogger(__name__)

# Set of (table_name, year) pairs that have already triggered the
# out-of-range GHG warning in this process.  Without de-duplication a
# 365-step AMIP run would log the same warning 365 times.
_GHG_OUT_OF_RANGE_WARNED: set[tuple[str, int]] = set()


# ======================================================================
# ExperimentTemplate
# ======================================================================

class ExperimentTemplate(NamedTuple):
    """Metadata for a CMIP-style experiment.

    Parameters
    ----------
    name : str
        Short experiment identifier (e.g. ``"piControl"``).
    description : str
        One-line human-readable description.
    start_year : int
        First simulation year.
    end_year : int
        Last simulation year (inclusive).
    parent_experiment : str
        Name of the parent experiment (empty string if none).
    forcing_type : str
        ``"fixed"`` for time-invariant forcing, ``"transient"`` for
        time-varying GHG concentrations.
    variant_label : str
        CMIP6 variant label (e.g. ``"r1i1p1f1"``).
    base_co2_ppmv : float
        CO2 concentration [ppmv].  For fixed experiments this is the
        constant value; for transient experiments the starting value.
    base_ch4_ppbv : float
        CH4 concentration [ppbv] (same convention as CO2).
    base_n2o_ppbv : float
        N2O concentration [ppbv] (same convention as CO2).
    """

    name: str
    description: str
    start_year: int
    end_year: int
    parent_experiment: str
    forcing_type: str
    variant_label: str
    base_co2_ppmv: float
    base_ch4_ppbv: float
    base_n2o_ppbv: float


# ======================================================================
# Built-in GHG concentration time series (CMIP6 key years)
# ======================================================================

# Each dict maps year -> (co2_ppmv, ch4_ppbv, n2o_ppbv).

_GHG_HISTORICAL: dict[int, tuple[float, float, float]] = {
    1850: (284.3, 808.2, 273.0),
    1900: (295.7, 911.0, 275.7),
    1950: (310.7, 1147.0, 289.0),
    1979: (336.78, 1550.0, 301.0),  # AMIP start year (NOAA Mauna Loa + AGGI)
    1980: (338.7, 1547.0, 301.0),
    1990: (354.39, 1714.0, 308.0),
    2000: (369.5, 1773.0, 316.0),
    2010: (389.85, 1798.0, 323.0),
    2014: (397.5, 1834.0, 327.0),  # AMIP CMIP6 end year
    2021: (414.72, 1895.0, 334.5),  # CMIP7 AMIP extension
}

_GHG_SSP245: dict[int, tuple[float, float, float]] = {
    2015: (401.0, 1877.0, 328.9),
    2030: (437.0, 1803.0, 338.0),
    2050: (502.0, 1555.0, 348.0),
    2070: (554.0, 1350.0, 355.0),
    2100: (603.0, 1122.0, 360.0),
}

_GHG_SSP585: dict[int, tuple[float, float, float]] = {
    2015: (401.0, 1877.0, 328.9),
    2030: (472.0, 2030.0, 345.0),
    2050: (601.0, 2200.0, 373.0),
    2070: (798.0, 2200.0, 405.0),
    2100: (1135.0, 2000.0, 435.0),
}

# Mapping from experiment name to its GHG table (only for experiments
# that have transient or explicitly tabulated concentrations).
_GHG_TABLES: dict[str, dict[int, tuple[float, float, float]]] = {
    "historical": _GHG_HISTORICAL,
    "ssp245": _GHG_SSP245,
    "ssp585": _GHG_SSP585,
    # AMIP shares the historical GHG trajectory: the CMIP6 AMIP protocol
    # mandates the same time-varying CO2/CH4/N2O as the historical run
    # (only the SST/SIC are observed rather than coupled).
    "amip": _GHG_HISTORICAL,
}


# ======================================================================
# GHG interpolation
# ======================================================================

def _interp_ghg_table(
    table: dict[int, tuple[float, float, float]],
    year: float,
    table_name: str = "ghg",
) -> tuple[float, float, float]:
    """Linearly interpolate a GHG table, clamped at endpoints.

    A one-shot warning is emitted when ``year`` falls more than 0.5
    years outside the table's anchor range (e.g. ``ghg_at_year("amip",
    2026)`` when the historical table stops at 2021).  ``np.interp``
    silently clamps in this case, so the silent extrapolation /
    constant-tail behaviour is otherwise invisible to the user — this
    is exactly the kind of forcing bias the iter-3/4 codex review
    flagged as a high-leverage failure mode for production CMIP6 AMIP
    runs that extend past 2021 (or past 2014 for the strict CMIP6
    protocol).

    Parameters
    ----------
    table : dict[int, (co2, ch4, n2o)]
        Sparse year-to-concentration mapping.
    year : float
        Target year (may be fractional).
    table_name : str, optional
        Used in the out-of-range warning so users can identify which
        scenario / experiment is producing the clamped value.

    Returns
    -------
    (co2_ppmv, ch4_ppbv, n2o_ppbv) : tuple of float
    """
    years = sorted(table.keys())
    co2_vals = np.array([table[y][0] for y in years])
    ch4_vals = np.array([table[y][1] for y in years])
    n2o_vals = np.array([table[y][2] for y in years])
    years_arr = np.array(years, dtype=np.float64)

    # One-shot out-of-range warning: triggered once per (table,
    # integer-year) bin so a multi-year run doesn't flood the log.
    # Truncate (``int(...)``) rather than round-half-to-even so that
    # 2025.0 and 2025.5 share the same warning slot.
    y_min, y_max = float(years_arr[0]), float(years_arr[-1])
    if year < y_min - 0.5 or year > y_max + 0.5:
        key = (table_name, int(year))
        if key not in _GHG_OUT_OF_RANGE_WARNED:
            _GHG_OUT_OF_RANGE_WARNED.add(key)
            logger.warning(
                f"[ghg_table] year={year:.2f} is outside the {table_name!r} "
                f"GHG anchor range [{y_min:.0f}, {y_max:.0f}] — "
                f"np.interp will clamp to the endpoint value, producing "
                f"a flat tail in CO2/CH4/N2O.  For accurate forcing "
                f"past {y_max:.0f}, supply an external GHG file via "
                f"--ghg-forcing external --ghg-file <path> or extend "
                f"the anchor table in src/legoesm/forcing/experiments.py."
            )

    co2 = float(np.interp(year, years_arr, co2_vals))
    ch4 = float(np.interp(year, years_arr, ch4_vals))
    n2o = float(np.interp(year, years_arr, n2o_vals))
    return co2, ch4, n2o


def ghg_at_year(
    experiment_name: str,
    year: float,
) -> tuple[float, float, float]:
    """Return GHG concentrations for *experiment_name* at *year*.

    Linear interpolation between benchmark years; clamped at the edges
    of each scenario's time series.

    For ``"piControl"`` the concentrations are constant (the template's
    base values).  ``"amip"`` shares the ``"historical"`` GHG table so
    CO2/CH4/N2O follow the CMIP6 historical trajectory (1979-2014).
    For ``"1pctCO2"`` the CO2 grows at 1 % per year from 284.3 ppmv
    while CH4 and N2O stay at pre-industrial levels.

    Parameters
    ----------
    experiment_name : str
        One of the keys in ``EXPERIMENT_TEMPLATES``.
    year : float
        Calendar year (may be fractional, e.g. 1990.5).

    Returns
    -------
    (co2_ppmv, ch4_ppbv, n2o_ppbv) : tuple of float

    Raises
    ------
    ValueError
        If *experiment_name* is not recognised.
    """
    if experiment_name not in EXPERIMENT_TEMPLATES:
        raise ValueError(
            f"Unknown experiment {experiment_name!r}. "
            f"Available: {sorted(EXPERIMENT_TEMPLATES)}"
        )

    tmpl = EXPERIMENT_TEMPLATES[experiment_name]

    # Experiments with a dedicated GHG table.
    if experiment_name in _GHG_TABLES:
        return _interp_ghg_table(
            _GHG_TABLES[experiment_name], year,
            table_name=experiment_name,
        )

    # 1pctCO2: 1 % per year compound increase from pre-industrial CO2.
    if experiment_name == "1pctCO2":
        years_elapsed = max(year - tmpl.start_year, 0.0)
        co2 = tmpl.base_co2_ppmv * (1.01 ** years_elapsed)
        return co2, tmpl.base_ch4_ppbv, tmpl.base_n2o_ppbv

    # Fixed-forcing experiments (piControl, amip, ...).
    return tmpl.base_co2_ppmv, tmpl.base_ch4_ppbv, tmpl.base_n2o_ppbv


def get_ghg_for_experiment(
    name: str,
    year: float,
) -> dict[str, float]:
    """Return GHG concentrations as a dict.

    Convenience wrapper around :func:`ghg_at_year`.

    Parameters
    ----------
    name : str
        Experiment name (key in ``EXPERIMENT_TEMPLATES``).
    year : float
        Calendar year.

    Returns
    -------
    dict
        ``{"co2_ppmv": ..., "ch4_ppbv": ..., "n2o_ppbv": ...}``
    """
    co2, ch4, n2o = ghg_at_year(name, year)
    return {"co2_ppmv": co2, "ch4_ppbv": ch4, "n2o_ppbv": n2o}


# ======================================================================
# Experiment templates
# ======================================================================

EXPERIMENT_TEMPLATES: dict[str, ExperimentTemplate] = {
    "piControl": ExperimentTemplate(
        name="piControl",
        description="Pre-industrial control with fixed 1850 forcing",
        start_year=1850,
        end_year=2350,
        parent_experiment="",
        forcing_type="fixed",
        variant_label="r1i1p1f1",
        base_co2_ppmv=284.3,
        base_ch4_ppbv=808.2,
        base_n2o_ppbv=273.0,
    ),
    "historical": ExperimentTemplate(
        name="historical",
        description="Historical simulation with observed transient forcing (1850-2014)",
        start_year=1850,
        end_year=2014,
        parent_experiment="piControl",
        forcing_type="transient",
        variant_label="r1i1p1f1",
        base_co2_ppmv=284.3,
        base_ch4_ppbv=808.2,
        base_n2o_ppbv=273.0,
    ),
    "ssp245": ExperimentTemplate(
        name="ssp245",
        description="SSP2-4.5 future scenario (medium forcing pathway)",
        start_year=2015,
        end_year=2100,
        parent_experiment="historical",
        forcing_type="transient",
        variant_label="r1i1p1f1",
        base_co2_ppmv=401.0,
        base_ch4_ppbv=1877.0,
        base_n2o_ppbv=328.9,
    ),
    "ssp585": ExperimentTemplate(
        name="ssp585",
        description="SSP5-8.5 future scenario (high forcing pathway)",
        start_year=2015,
        end_year=2100,
        parent_experiment="historical",
        forcing_type="transient",
        variant_label="r1i1p1f1",
        base_co2_ppmv=401.0,
        base_ch4_ppbv=1877.0,
        base_n2o_ppbv=328.9,
    ),
    "amip": ExperimentTemplate(
        name="amip",
        description=(
            "AMIP simulation with prescribed SST and sea-ice (1979-2014). "
            "GHGs follow the CMIP6 historical trajectory (transient)."
        ),
        start_year=1979,
        end_year=2014,
        parent_experiment="",
        forcing_type="transient",
        variant_label="r1i1p1f1",
        base_co2_ppmv=336.78,
        base_ch4_ppbv=1550.0,
        base_n2o_ppbv=301.0,
    ),
    "1pctCO2": ExperimentTemplate(
        name="1pctCO2",
        description="1% per year CO2 increase from pre-industrial (idealized)",
        start_year=1850,
        end_year=2000,
        parent_experiment="piControl",
        forcing_type="transient",
        variant_label="r1i1p1f1",
        base_co2_ppmv=284.3,
        base_ch4_ppbv=808.2,
        base_n2o_ppbv=273.0,
    ),
}


# ======================================================================
# Factory: template -> AMIPExperimentConfig
# ======================================================================

_DAYS_PER_YEAR = 365


def _year_to_day(year: int, ref_year: int) -> float:
    """Convert a calendar year to a model day relative to *ref_year*."""
    return float((year - ref_year) * _DAYS_PER_YEAR)


def create_experiment_config(
    name: str,
    **overrides,
):
    """Create an :class:`ExperimentConfig` from a template.

    The factory translates year-based experiment metadata into
    model-day-based configuration fields, sets GHG concentrations
    from the template's base values, and applies any caller-supplied
    overrides on top.

    Returns the **canonical** ``ExperimentConfig``.  For the legacy
    ``AMIPExperimentConfig`` use ``create_amip_experiment_config``.

    Parameters
    ----------
    name : str
        Experiment name (must be a key in ``EXPERIMENT_TEMPLATES``).
    **overrides
        Any ``ExperimentConfig`` field name (or sub-config field name
        like ``resolution``, ``nlev``, ``dt``) with the desired value.
        Legacy AMIPExperimentConfig flat field names are accepted for
        backward compatibility and mapped to the correct sub-config.

    Returns
    -------
    ExperimentConfig

    Raises
    ------
    ValueError
        If *name* is not a recognised experiment.
    TypeError
        If an override key is not a valid field.

    Examples
    --------
    >>> cfg = create_experiment_config("piControl", resolution=48)
    >>> cfg.days
    182500
    >>> cfg.co2_ppmv
    284.3
    """
    # Lazy import to avoid circular dependency chain.
    from legoesm.driver.config import (
        ExperimentConfig,
        GridConfig,
        DycoreConfig,
        OutputConfig,
    )

    if name not in EXPERIMENT_TEMPLATES:
        raise ValueError(
            f"Unknown experiment {name!r}. "
            f"Available: {sorted(EXPERIMENT_TEMPLATES)}"
        )

    tmpl = EXPERIMENT_TEMPLATES[name]

    # Total integration length in model days.
    total_days = (tmpl.end_year - tmpl.start_year) * _DAYS_PER_YEAR

    # GHG concentrations at the start of the experiment.
    co2, ch4, n2o = ghg_at_year(name, tmpl.start_year)

    # Partition overrides into sub-config fields vs top-level fields.
    # Legacy flat field names (resolution, nlev, dt, etc.) are mapped
    # to the appropriate sub-config.
    _GRID_FIELDS = set(GridConfig._fields)
    _DYCORE_FIELDS = set(DycoreConfig._fields)
    _OUTPUT_FIELDS = set(OutputConfig._fields)
    _TOP_FIELDS = set(ExperimentConfig._fields)

    grid_ov: dict = {}
    dycore_ov: dict = {}
    output_ov: dict = {}
    top_ov: dict = {}

    for k, v in overrides.items():
        if k in _TOP_FIELDS:
            top_ov[k] = v
        elif k in _GRID_FIELDS:
            grid_ov[k] = v
        elif k in _DYCORE_FIELDS:
            dycore_ov[k] = v
        elif k in _OUTPUT_FIELDS:
            output_ov[k] = v
        # Legacy flat field names from AMIPExperimentConfig
        elif k == "resolution":
            grid_ov["resolution"] = v
        elif k == "nlev":
            grid_ov["nlev"] = v
        elif k == "dt":
            dycore_ov["dt"] = v
        elif k == "diag_days":
            output_ov["diag_days"] = v
        elif k == "checkpoint_days":
            output_ov["checkpoint_days"] = v
        elif k == "hyperdiff_scale":
            dycore_ov["hyperdiff_scale"] = v
        else:
            raise TypeError(
                f"Invalid ExperimentConfig field: {k!r}"
            )

    grid = GridConfig(**{**GridConfig()._asdict(), **grid_ov})

    # Auto-select a valid discretization for the chosen grid_type
    # when the user hasn't explicitly overridden it.
    _GRID_DEFAULT_DISCRETIZATION = {
        "cubed_sphere": "cdgrid",
        "latlon": "latlon_cgrid",
        "gaussian": "spectral",
        "voronoi": "mpas",
        "mpas": "mpas",
    }
    _GRID_DEFAULT_MODEL = {
        "cubed_sphere": "hydrostatic",
        "latlon": "hydrostatic",
        "gaussian": "spectral_pe",
        "voronoi": "hydrostatic",
        "mpas": "hydrostatic",
    }
    if "discretization" not in dycore_ov and "grid_type" in grid_ov:
        gt = grid.grid_type
        if gt in _GRID_DEFAULT_DISCRETIZATION:
            dycore_ov["discretization"] = _GRID_DEFAULT_DISCRETIZATION[gt]
        else:
            raise ValueError(
                f"Unsupported grid_type={gt!r} for auto-discretization. "
                f"Supported: {sorted(_GRID_DEFAULT_DISCRETIZATION)}"
            )
    if "model_type" not in dycore_ov and "grid_type" in grid_ov:
        gt = grid.grid_type
        if gt in _GRID_DEFAULT_MODEL:
            dycore_ov["model_type"] = _GRID_DEFAULT_MODEL[gt]

    dycore = DycoreConfig(**{**DycoreConfig()._asdict(), **dycore_ov})
    output = OutputConfig(**{**OutputConfig()._asdict(), **output_ov})

    # Production-appropriate defaults for CMIP experiments.
    # These can be overridden via **overrides.
    radiation = "gray"   # safe default for idealized tests
    dataset = "analytical"

    if name in ("amip",):
        # AMIP should use prescribed SST; analytical is a fallback.
        # Set radiation to rrtmgp when available for production.
        radiation = "rrtmgp"
    elif name in ("historical", "ssp245", "ssp585", "1pctCO2"):
        radiation = "rrtmgp"
    # piControl keeps gray as default — it's an idealized control run.

    # Allow explicit overrides to win.
    if "radiation" in top_ov:
        radiation = top_ov.pop("radiation")
    if "dataset" in top_ov:
        dataset = top_ov.pop("dataset")

    # Build the top-level config.
    cfg_dict: dict = {
        "grid": grid,
        "dycore": dycore,
        "output": output,
        "days": total_days,
        "start_day": 0.0,
        "co2_ppmv": co2,
        "ch4_ppbv": ch4,
        "n2o_ppbv": n2o,
        "experiment": name,
        "start_year": tmpl.start_year,
        "radiation": radiation,
        "dataset": dataset,
    }

    # Apply remaining top-level overrides.
    cfg_dict.update(top_ov)

    cfg = ExperimentConfig(**{
        **ExperimentConfig()._asdict(),
        **cfg_dict,
    })

    # Validation warnings for CMIP-inappropriate configs.
    import warnings
    if name == "amip" and cfg.dataset == "analytical":
        warnings.warn(
            f"CMIP experiment '{name}' is using analytical SST/SIC. "
            f"Set dataset='hadisst' or a real SST file for production runs.",
            UserWarning,
            stacklevel=2,
        )
    if (tmpl.forcing_type == "transient"
            and cfg.radiation == "gray"
            and "radiation" not in overrides):
        warnings.warn(
            f"CMIP transient experiment '{name}' is using gray radiation. "
            f"Transient GHG trajectories require rrtmgp/rrtmg radiation "
            f"to affect the simulation.",
            UserWarning,
            stacklevel=2,
        )

    return cfg


def create_amip_experiment_config(
    name: str,
    **overrides,
) -> AMIPExperimentConfig:
    """Create an :class:`AMIPExperimentConfig` from a template.

    .. deprecated::
        Use ``create_experiment_config`` which returns the canonical
        ``ExperimentConfig``.  This legacy factory is retained only
        for backward compatibility with old code paths.
    """
    import warnings
    warnings.warn(
        "create_amip_experiment_config is deprecated; "
        "use create_experiment_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if name not in EXPERIMENT_TEMPLATES:
        raise ValueError(
            f"Unknown experiment {name!r}. "
            f"Available: {sorted(EXPERIMENT_TEMPLATES)}"
        )

    # Validate override keys.
    valid_fields = set(AMIPExperimentConfig._fields)
    bad = set(overrides) - valid_fields
    if bad:
        raise TypeError(
            f"Invalid AMIPExperimentConfig field(s): {sorted(bad)}"
        )

    tmpl = EXPERIMENT_TEMPLATES[name]

    # Total integration length in model days.
    total_days = (tmpl.end_year - tmpl.start_year) * _DAYS_PER_YEAR

    # GHG concentrations at the start of the experiment.
    co2, ch4, n2o = ghg_at_year(name, tmpl.start_year)

    # Build the config dict, starting from defaults.
    cfg_dict: dict = {
        "days": total_days,
        "start_day": 0.0,
        "co2_ppmv": co2,
        "ch4_ppbv": ch4,
        "n2o_ppbv": n2o,
    }

    # Apply caller overrides.
    cfg_dict.update(overrides)

    return AMIPExperimentConfig(**{
        **AMIPExperimentConfig()._asdict(),
        **cfg_dict,
    })
