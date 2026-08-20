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
      albedo: {field: value}           # LandAlbedoConfig field values (absent = default)
      glacier_albedo_vis: float        # ice-sheet base albedo pair (both or neither;
      glacier_albedo_nir: float        #   absent = uncalibrated module default)
      soil_n_layers: int               # Richards soil layers (default 8)
      soil_depth_m: float              # total soil column depth [m] (0 = geometric default ~6.375)
      soil_growth_factor: float        # layer thickness ratio (2.0 = default geometric)
      root_depth_per_pft: [17 floats]  # per-PFT root-zone uptake, CLM5 PFT order
      theta_wp_per_pft: [17 floats]    #   (all three together; absent = the scalar
      theta_fc_per_pft: [17 floats]    #   MultiLayerLandConfig values)

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

import math
from pathlib import Path
from typing import Any, Iterable, NamedTuple

from legoesm.land.soil_grid import SoilGridConfig

# --- Schema keys, validated at load time ---
# The SCVT Voronoi aliases are here because the DRIVER already builds that mesh
# (``run_lmip_biophys.make_grid`` branches on it) and only this list stood in
# the way of asking for one. That gap mattered: a coupled AMIP run on the
# Voronoi mesh needs its land state spun up on the SAME mesh, because the
# restart loader compares column counts and refuses a mismatch rather than
# interpolating. Without the aliases the only offline spin-up reachable for
# such a run was a different grid whose output it would then reject.
_GRID_TYPES = ("latlon", "gaussian", "cubed_sphere",
               "voronoi", "icosahedral", "ico", "mpas", "mpas_voronoi")
_LAND_MODES = ("multilayer", "slab")
_SURFACE_SCHEMES = ("two_leaf_canopy", "simple_seb")
_BULK_SCHEMES = ("most", "constant")
_STOMATA_MODELS = ("ball_berry", "medlyn")
# Snow thermal scheme.  Only the single-node bulk SWE budget exists on this
# lane today; the key is validated so a config naming the (future) multilayer
# snow column fails loudly here instead of silently running different physics.
_SNOW_SCHEMES = ("single",)
# Per-PFT root tables are indexed by the CLM5 17-PFT axis (bare + 16 veg).
_N_PFT_CLM5 = 17
# The three per-PFT root-zone lists travel together (the builders gather all
# three at the dominant PFT); partial specification is a config error.
_PFT_ROOT_KEYS = ("root_depth_per_pft", "theta_wp_per_pft", "theta_fc_per_pft")
# Retired 2026-08 selector keys: values now go in the config directly
# (physics.albedo / glacier_albedo_vis+nir / the *_per_pft lists).  Rejected
# loudly so an old config cannot silently run uncalibrated.
_RETIRED_SELECTOR_KEYS = ("albedo_calibration", "root_calibration")
# Vertical soil grid defaults = SoilGridConfig() (8 geometric layers from
# dz_top 0.025 m, growth 2 -> 6.375 m total).  Kept as named module constants so
# the schema default and the SoilGridConfig default cannot silently diverge.
_SOIL_N_LAYERS_DEFAULT = SoilGridConfig().n_layers
_SOIL_DEPTH_M_DEFAULT = SoilGridConfig().total_depth
_SOIL_GROWTH_FACTOR_DEFAULT = SoilGridConfig().growth_factor
_FORCING_SOURCES = ("cru_jra", "synthetic")

# --- Schema SANITY bounds for user-supplied stomatal overrides (name, lo, hi) ---
# NOT physics coefficients: these only bracket obvious typos at config-load time.
# The authoritative physical ranges are the scheme's StomataConfig.__param_spec__,
# which is enforced downstream; these are deliberately wider.
_STOMATA_SANITY_BOUNDS = (
    ("vc_max25", 10.0, 200.0),     # umol m-2 s-1
    ("g1", 0.5, 30.0),             # ball_berry slope / medlyn g1
    ("gs_max", 0.05, 1.5),         # mol m-2 s-1
)

# The public CESM inputdata naming, which is what `download_lmip_data.sh`
# stages off-site and what the reader itself defaults to
# (`cru_jra.CRUJRA_FILE_PREFIX`).  It used to be the glade-only
# `_filled_antarct_and_grnlnd_` variant, so the two defaults in this repo
# disagreed and every template asked for files no download could produce.
_DEFAULT_PREFIX = "clmforc.CRUJRAv2.5_0.5x0.5"


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
    # Snow thermal scheme: "single" (single-node bulk SWE) is the only scheme on
    # this lane; the key exists so configs are explicit and a future scheme name
    # fails here rather than silently running different physics.
    physics.setdefault("snow_scheme", "single")
    if physics["snow_scheme"] not in _SNOW_SCHEMES:
        raise ValueError(
            f"physics.snow_scheme={physics['snow_scheme']!r} not in {_SNOW_SCHEMES}")
    # Retired selector keys fail loudly: silently ignoring them would run an
    # old calibrated config with the uncalibrated defaults.
    for _rk in _RETIRED_SELECTOR_KEYS:
        if _rk in physics:
            raise ValueError(
                f"physics.{_rk} was retired: give the parameter VALUES instead "
                "(physics.albedo: {LandAlbedoConfig field: value}, "
                "physics.glacier_albedo_vis/_nir, and the three "
                "physics.*_per_pft lists).  See templates/land/biophysics/"
                "lmip_biophys_2deg.yaml for the calibrated values.")
    # --- Surface albedo: generic field values, generic names -----------------
    # ``physics.albedo`` is an optional {field: value} block over
    # LandAlbedoConfig's own field names; an absent field keeps that field's
    # default, and an absent/empty block is bit-identical to every pre-existing
    # LMIP run.  Bounds live with the scheme (__param_spec__); here we only
    # reject unknown names and non-finite values.
    physics.setdefault("albedo", {})
    _alb = physics["albedo"]
    if not isinstance(_alb, dict):
        raise ValueError(f"physics.albedo must be a mapping (got {_alb!r})")
    if _alb:
        from legoesm.surface_albedo import LandAlbedoConfig
        _bad = set(_alb) - set(LandAlbedoConfig._fields)
        if _bad:
            raise ValueError(
                f"physics.albedo: unknown field(s) {sorted(_bad)}; valid fields "
                f"are {sorted(LandAlbedoConfig._fields)}")
        for _k, _v in _alb.items():
            if not isinstance(_v, (int, float)) or isinstance(_v, bool) \
                    or not math.isfinite(float(_v)):
                raise ValueError(
                    f"physics.albedo.{_k} must be a finite number (got {_v!r})")
    # Ice-sheet base albedo pair: both or neither, each a fraction in [0, 1].
    _gv = physics.setdefault("glacier_albedo_vis", None)
    _gn = physics.setdefault("glacier_albedo_nir", None)
    if (_gv is None) != (_gn is None):
        raise ValueError(
            "physics.glacier_albedo_vis and glacier_albedo_nir must be given "
            "together (the canopy consumes a (vis, nir) pair).")
    for _k, _v in (("glacier_albedo_vis", _gv), ("glacier_albedo_nir", _gn)):
        if _v is not None and (not isinstance(_v, (int, float))
                               or isinstance(_v, bool)
                               or not (0.0 <= float(_v) <= 1.0)):
            raise ValueError(
                f"physics.{_k} must be an albedo fraction in [0, 1] (got {_v!r})")
    # Vertical soil discretisation.  Defaults reproduce SoilGridConfig() (8
    # geometric layers, dz_top 0.025 m -> ~6.375 m); AMIP parity is 10 / 3.0 m.
    # The surfdata soil profile is remapped onto THIS grid (init_land_surface_data
    # passes it to the loader), so the two can never desync.
    physics.setdefault("soil_n_layers", _SOIL_N_LAYERS_DEFAULT)
    physics.setdefault("soil_depth_m", _SOIL_DEPTH_M_DEFAULT)
    _nl = physics["soil_n_layers"]
    if not isinstance(_nl, int) or isinstance(_nl, bool) or not (1 <= _nl <= 50):  # coeff-ok: schema sanity bound on a layer COUNT, not a physics coefficient
        raise ValueError(
            f"physics.soil_n_layers must be an int in [1, 50] (got {_nl!r})")
    _sd = physics["soil_depth_m"]
    if not isinstance(_sd, (int, float)) or isinstance(_sd, bool) or _sd < 0.0:
        raise ValueError(
            f"physics.soil_depth_m must be a non-negative float "
            f"(0 = geometric default; got {_sd!r})")
    if 0.0 < float(_sd) < 0.1:  # coeff-ok: schema sanity floor on column depth [m]; 0 means "geometric default"
        raise ValueError(
            f"physics.soil_depth_m={_sd} is implausibly shallow for a land column "
            "(< 0.1 m); use 0 for the geometric default.")
    # --- Per-PFT root-zone water uptake (optional; all three lists together) --
    # Length-17 (CLM5 PFT order) values for the root e-folding depth [m] and
    # the PLANT wilting-point / field-capacity thresholds [m3/m3], applied per
    # column at the dominant PFT.  Absent -> the scalar MultiLayerLandConfig
    # values (bit-identical to the pre-existing behaviour).
    _given = [k for k in _PFT_ROOT_KEYS if physics.setdefault(k, None) is not None]
    if _given and len(_given) != len(_PFT_ROOT_KEYS):
        raise ValueError(
            f"physics.{_PFT_ROOT_KEYS} must be given together (got only "
            f"{_given}); the builders gather all three at the dominant PFT.")
    if _given:
        for _k in _PFT_ROOT_KEYS:
            _v = physics[_k]
            if (not isinstance(_v, (list, tuple)) or len(_v) != _N_PFT_CLM5
                    or not all(isinstance(x, (int, float))
                               and not isinstance(x, bool)
                               and math.isfinite(float(x)) for x in _v)):
                raise ValueError(
                    f"physics.{_k} must be a list of {_N_PFT_CLM5} finite "
                    f"numbers (CLM5 PFT order)")
            physics[_k] = [float(x) for x in _v]
        for _x in physics["root_depth_per_pft"]:
            if not (0.01 <= _x <= 10.0):  # coeff-ok: schema sanity bound on a root depth [m]
                raise ValueError(
                    f"physics.root_depth_per_pft entry {_x} outside sane "
                    "range [0.01, 10] m")
        for _w, _f in zip(physics["theta_wp_per_pft"],
                          physics["theta_fc_per_pft"]):
            if not (0.0 < _w < _f < 1.0):
                raise ValueError(
                    f"per-PFT theta_wp={_w} / theta_fc={_f} must satisfy "
                    "0 < wp < fc < 1")
    physics.setdefault("soil_growth_factor", _SOIL_GROWTH_FACTOR_DEFAULT)
    _gf = physics["soil_growth_factor"]
    if not isinstance(_gf, (int, float)) or isinstance(_gf, bool) or not (1.0 <= _gf <= 4.0):  # coeff-ok: schema sanity bound on a layer-thickness RATIO
        raise ValueError(
            f"physics.soil_growth_factor must be a float in [1.0, 4.0] (got {_gf!r})")
    physics["soil_growth_factor"] = float(_gf)
    physics["soil_depth_m"] = float(_sd)
    # Stomatal calibration scalars (None = land default / per-PFT): sanity bounds
    # (StomataConfig.__param_spec__ enforces tighter physical ranges downstream).
    for _k, _lo, _hi in _STOMATA_SANITY_BOUNDS:
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

    # Run the multilayer land under the PHYSICS the baked per-PFT tables were
    # calibrated under (legoesm.land.config.calibrated_multilayer_setup): MOST
    # exchange, big-leaf SEB, Farquhar stomata on a prescribed leaf carbon, and
    # the calibration soil column.  A land INITIAL CONDITION is only meaningful
    # for the model it was equilibrated under, so a spin-up feeding a calibrated
    # coupled run has to be on this too.  Off (default) = unchanged.
    #
    # NAMED "_physics" DELIBERATELY, and it is the whole promise: this driver
    # builds its per-plant-type PARAMETERS from its own surface dataset, NOT
    # from the tuned tables the coupled run uses, so the schemes and the soil
    # column match while the parameter values do not.  Closing that gap means
    # changing which surface dataset this driver reads — a separate decision.
    physics.setdefault("calibrated_land_physics", False)
    if not isinstance(physics["calibrated_land_physics"], bool):
        raise ValueError(
            f"physics.calibrated_land_physics must be a bool "
            f"(got {physics['calibrated_land_physics']!r})")
    if physics["calibrated_land_physics"]:
        if physics["land_mode"] != "multilayer":
            raise ValueError(
                "physics.calibrated_land_physics=true requires physics.land_mode="
                "'multilayer' — the calibrated tables are a multilayer bake with "
                "no slab equivalent."
            )
        # EVERY field the calibrated setup replaces, plus freeze/thaw, which
        # this driver applies AFTER the setup and would otherwise survive into
        # the run.  Checked, never silently overridden, so a config can never
        # read as one land model while running another.
        from legoesm.land.config import calibrated_multilayer_setup
        _cal = calibrated_multilayer_setup()
        for _key, _exp in (
                ("surface_scheme", "simple_seb"),
                ("bulk_scheme", _cal["bulk_scheme"]),
                ("stomata_enabled", _cal["stomata"].enabled),
                ("stomatal_model", _cal["stomata"].stomata_model),
                ("snow_albedo_feedback", _cal["snow_albedo_feedback"]),
                ("enable_freeze_thaw", False),
        ):
            if physics[_key] != _exp:
                raise ValueError(
                    f"physics.calibrated_land_physics=true requires "
                    f"physics.{_key}={_exp!r} (the value the baked tables were "
                    f"fitted under); got {physics[_key]!r}. Set it or drop "
                    "calibrated_land_physics — these are not silently overridden."
                )
        for _stom_key in ("gs_max", "vc_max25", "g1"):
            if physics.get(_stom_key) is not None:
                raise ValueError(
                    f"physics.calibrated_land_physics=true cannot be combined "
                    f"with physics.{_stom_key}: the calibrated setup replaces the "
                    "whole stomatal configuration, so the value would vanish "
                    "without a word. Drop one of the two.")

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
