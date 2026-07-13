"""LMIP experiment configuration — the single source of truth for run_lmip_biophys.

An experiment is fully described by a YAML file with this schema::

    grid:
      type: latlon | gaussian | cubed_sphere
      resolution: int

    physics:
      land_mode: multilayer | slab
      surface_scheme: two_leaf_canopy | simple_seb
      bulk_scheme: most | constant
      enable_freeze_thaw: bool         # soil-water latent zero-curtain (default false)

    forcing:
      source: cru_jra | synthetic
      data_dir: <path or "">           # "" -> synthetic
      prefix: <cru-jra filename prefix>
      suffix: <optional after-year suffix>
      year_start: int
      year_end: int                    # >= year_start; == year_start for single-year
      k_neighbors: int                 # forcing regrid IDW neighbours (default 4)

    surfdata:
      path: <path to a surfdata NetCDF (static CLM5 or transient LUH2/HYDE/...)>
      land_cover_dataset: clm5 | luh2 | luh3 | hyde | pongratz | kk10  # default clm5

    time:
      dt: float                        # seconds
      n_steps: int
      start_doy: float

    restart:
      from: <path to a .npz or "">     # "" -> cold start

    land_frac_min: float                # surfdata land threshold (default 0.0 = any land)
    land_mask_file: <optional CMIP6 sftlf path>

    output:
      tapes: [ {name, freq, average, vars}, ... ]

Templates live in ``templates/land/*.yaml``; ``init_experiment.py`` deep-merges
overrides into a template and writes the resolved config to
``<expt_dir>/config.yaml``.  ``run_lmip_biophys.py --config <path>`` consumes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, NamedTuple

# --- Schema keys, validated at load time ---
_GRID_TYPES = ("latlon", "gaussian", "cubed_sphere")
_LAND_MODES = ("multilayer", "slab")
_SURFACE_SCHEMES = ("two_leaf_canopy", "simple_seb")
_BULK_SCHEMES = ("most", "constant")
_STOMATA_MODELS = ("ball_berry", "medlyn")
_SNOW_SCHEMES = ("single", "multilayer")
_FORCING_SOURCES = ("cru_jra", "synthetic")

_DEFAULT_PREFIX = "clmforc.CRUJRAv2.5_filled_antarct_and_grnlnd_0.5x0.5"


class LMIPConfig(NamedTuple):
    """Resolved LMIP run configuration (validated).  Direct dict-access on the
    nested sub-dicts intentionally — YAML shape ↔ dict is 1-to-1."""
    grid: dict
    physics: dict
    forcing: dict
    surfdata: dict
    time: dict
    restart: dict
    output: dict
    land_frac_min: float
    land_mask_file: str
    raw: dict                          # the un-merged dict, for provenance


def load_config(path: str | Path) -> LMIPConfig:
    """Load and validate a resolved config.yaml."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return validate_config(data)


def validate_config(data: dict) -> LMIPConfig:
    """Fail-fast schema check.  Missing sub-dicts get sensible defaults where
    the meaning is unambiguous (e.g. ``restart.from = ""`` = cold start)."""

    def req(section: str, keys: Iterable[str]) -> None:
        d = data.get(section) or {}
        for k in keys:
            if k not in d:
                raise ValueError(f"config: missing {section}.{k}")

    req("grid", ("type", "resolution"))
    grid = data["grid"]
    if grid["type"] not in _GRID_TYPES:
        raise ValueError(f"grid.type={grid['type']!r} not in {_GRID_TYPES}")

    req("physics", ("land_mode", "surface_scheme", "bulk_scheme"))
    physics = data["physics"]
    if physics["land_mode"] not in _LAND_MODES:
        raise ValueError(f"physics.land_mode={physics['land_mode']!r} not in {_LAND_MODES}")
    if physics["surface_scheme"] not in _SURFACE_SCHEMES:
        raise ValueError(f"physics.surface_scheme={physics['surface_scheme']!r} not in {_SURFACE_SCHEMES}")
    if physics["bulk_scheme"] not in _BULK_SCHEMES:
        raise ValueError(f"physics.bulk_scheme={physics['bulk_scheme']!r} not in {_BULK_SCHEMES}")
    # NOTE: simple_seb + MOST is now SUPPORTED — land/stable fixed SimpleSEB
    # cold-start stability and amip_sota runs exactly this pairing, so the prior
    # blanket rejection has been removed.

    # --- Composable physics knobs (optional; defaults = the AMIP-consistent
    #     multilayer two-leaf canopy).  None means "use the land default / the
    #     per-PFT surfdata value". ------------------------------------------------
    physics.setdefault("stomatal_model", "ball_berry")
    if physics["stomatal_model"] not in _STOMATA_MODELS:
        raise ValueError(
            f"physics.stomatal_model={physics['stomatal_model']!r} not in {_STOMATA_MODELS}")
    physics.setdefault("stomata_enabled", False)
    physics.setdefault("snow_albedo_feedback", True)
    # Soil-water freeze/thaw (apparent-heat-capacity zero-curtain, off by
    # default = bit-identical sensible-only soil heat).  Enabling it stabilises
    # boreal/Arctic winter columns whose energy budget otherwise diverges once
    # the top layer crosses the freezing point.  Multilayer-only (the slab land
    # has no soil column); harmlessly ignored for land_mode='slab'.
    physics.setdefault("enable_freeze_thaw", False)
    if not isinstance(physics["enable_freeze_thaw"], bool):
        raise ValueError(
            f"physics.enable_freeze_thaw must be a bool "
            f"(got {physics['enable_freeze_thaw']!r})")
    # Snow thermal scheme (Phase 2b): "single" (default, single-node bulk SWE) or
    # "multilayer" (CLM-faithful prognostic snow column that insulates the soil).
    # Multilayer is two_leaf_canopy-only (step_multilayer_land raises otherwise).
    physics.setdefault("snow_scheme", "single")
    if physics["snow_scheme"] not in _SNOW_SCHEMES:
        raise ValueError(
            f"physics.snow_scheme={physics['snow_scheme']!r} not in {_SNOW_SCHEMES}")
    if physics["snow_scheme"] == "multilayer" and physics["surface_scheme"] != "two_leaf_canopy":
        raise ValueError(
            "physics.snow_scheme='multilayer' requires surface_scheme='two_leaf_canopy' "
            f"(got {physics['surface_scheme']!r}).")
    # Stomatal calibration scalars (None = land default / per-PFT): sanity bounds
    # (StomataConfig.__param_spec__ enforces tighter physical ranges downstream).
    for _k, _lo, _hi in (("vc_max25", 10.0, 200.0), ("g1", 0.5, 30.0), ("gs_max", 0.05, 1.5)):
        _v = physics.get(_k)
        if _v is not None and not (_lo <= float(_v) <= _hi):
            raise ValueError(f"physics.{_k}={_v} out of sane range [{_lo}, {_hi}]")

    req("forcing", ("source", "year_start", "year_end"))
    forcing = data["forcing"]
    if forcing["source"] not in _FORCING_SOURCES:
        raise ValueError(f"forcing.source={forcing['source']!r} not in {_FORCING_SOURCES}")
    if forcing["year_end"] < forcing["year_start"]:
        raise ValueError(
            f"forcing.year_end ({forcing['year_end']}) "
            f"< forcing.year_start ({forcing['year_start']})")
    forcing.setdefault("data_dir", "")
    forcing.setdefault("prefix", _DEFAULT_PREFIX)
    forcing.setdefault("suffix", "")
    forcing.setdefault("k_neighbors", 4)
    # source must agree with data_dir: the driver decides real-vs-synthetic from
    # data_dir non-emptiness, so a mismatch would silently run the OTHER forcing
    # than declared (source='cru_jra' + empty data_dir -> silent synthetic).
    _data_dir = str(forcing.get("data_dir") or "")
    if forcing["source"] == "cru_jra" and not _data_dir:
        raise ValueError(
            "forcing.source='cru_jra' requires a non-empty forcing.data_dir "
            "(the staged CRU-JRA directory); use source='synthetic' for a "
            "no-data smoke run.")
    if forcing["source"] == "synthetic" and _data_dir:
        raise ValueError(
            f"forcing.source='synthetic' must not set forcing.data_dir "
            f"(got {_data_dir!r}); use source='cru_jra' to run real forcing.")

    req("surfdata", ("path",))
    surfdata = data["surfdata"]
    # Which reconstruction built the surfdata: provenance, and (Phase 2) selects
    # E_LUC's gross-vs-net transition handling.  Default = static CLM5 base.
    from legoesm.land.surface_data.datasets import validate_land_cover_dataset
    surfdata.setdefault("land_cover_dataset", "clm5")
    validate_land_cover_dataset(surfdata["land_cover_dataset"])

    # E_LUC land-use-change bookkeeping (optional block; scheme fail-fast here,
    # the numeric knobs are validated by validate_luc_config at run entry).
    luc = data.get("land_use_change") or {}
    from legoesm.land.land_use_change import LUC_SCHEMES, LandUseChangeConfig
    if luc.get("scheme", "none") not in LUC_SCHEMES:
        raise ValueError(
            f"land_use_change.scheme={luc.get('scheme')!r} not in {LUC_SCHEMES}")
    # A typo'd knob (e.g. clear_brun_frac) would otherwise be silently dropped and
    # the bookkeeping would run with defaults — a hard error instead.
    _luc_unknown = set(luc) - set(LandUseChangeConfig._fields)
    if _luc_unknown:
        raise ValueError(
            f"land_use_change: unknown key(s) {sorted(_luc_unknown)}; "
            f"valid keys: {sorted(LandUseChangeConfig._fields)}")

    req("time", ("dt", "n_steps", "start_doy"))
    time = data["time"]
    if time["n_steps"] <= 0:
        raise ValueError(f"time.n_steps must be > 0 (got {time['n_steps']})")
    if time["dt"] <= 0:
        raise ValueError(f"time.dt must be > 0 (got {time['dt']})")

    restart = data.get("restart") or {}
    restart.setdefault("from", "")

    output = data.get("output") or {}
    if "tapes" not in output or not output["tapes"]:
        raise ValueError("output.tapes: at least one tape must be declared")

    return LMIPConfig(
        grid=grid,
        physics=physics,
        forcing=forcing,
        surfdata=surfdata,
        time=time,
        restart=restart,
        output=output,
        land_frac_min=float(data.get("land_frac_min", 0.0)),
        land_mask_file=str(data.get("land_mask_file", "")),
        raw=data,
    )


def apply_overrides(base: dict, overrides: Iterable[str]) -> dict:
    """Apply ``-o dot.notation=value`` overrides in place on a deep-copied
    ``base``.  Values are YAML-parsed so ``x=3`` -> int, ``x=1.5`` -> float,
    ``x=true`` -> bool, ``x=[a,b]`` -> list, ``x=hello`` -> str.
    """
    import copy
    import yaml
    out = copy.deepcopy(base)
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"override {spec!r}: expected KEY.PATH=VALUE")
        key, raw_val = spec.split("=", 1)
        try:
            val = yaml.safe_load(raw_val)
        except yaml.YAMLError:
            val = raw_val
        _set_dotted(out, key.strip().split("."), val)
    return out


def _set_dotted(d: dict, path: list[str], value: Any) -> None:
    for p in path[:-1]:
        if p not in d or not isinstance(d[p], dict):
            d[p] = {}
        d = d[p]
    d[path[-1]] = value


__all__ = [
    "LMIPConfig",
    "apply_overrides",
    "load_config",
    "validate_config",
]
