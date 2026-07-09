#!/usr/bin/env python
"""Run a fully coupled Earth System Model simulation.

Supports multiple configuration presets:
  aquaplanet    — Slab ocean everywhere, no land, no carbon
  slab_simple   — Slab ocean + slab bucket land
  slab_pft      — Slab ocean + PFT-weighted land parameters
  slab_richards — Slab ocean + Richards' equation soil hydrology
  slab_carbon   — + DifferLand carbon + atmospheric CO2 tracer
  full_coupled  — + ocean biogeochemistry CO2

Example usage::

    JAX_ENABLE_X64=1 python scripts/run_coupled.py \\
        --preset aquaplanet --days 365 --resolution 16

    JAX_ENABLE_X64=1 python scripts/run_coupled.py \\
        --preset slab_carbon --days 730 --resolution 16 --nlev 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import NamedTuple

# Ensure the project root is on sys.path for test_cases imports
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import jax

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_coupled")

_LAND_SCHEMES = ("slab", "multilayer")

def _check_params_clobber(params: dict, land_params: str) -> None:
    """Refuse --params overrides that ``CoupledESMDriver.setup()`` would clobber.

    The coupled driver REBUILDS several component sub-configs during setup from
    external maps / scheme-forcing, discarding a ``--params`` override on them:

    * the **ocean** config is rebuilt from the mesh/preset — calibrate the ocean
      via ``run_omip.py --params`` (0 ocean params are coupled-only);
    * the **land carbon** config is rebuilt regardless of the land path (carbon-
      active + differland, and the multilayer diurnal-surface force-upgrade to
      differland) — calibrate via ``run_lmip.py --params``;
    * under ``--land-params clm`` (the default) the CLM reference surfdata maps
      and the per-PFT parameter provider override essentially ALL config-level
      land parameters (drag, soil hydraulics/thermal, surface + material fields),
      so no ``land.*`` override is effective — pass ``--land-params analytical``
      (or use ``run_lmip.py``) to calibrate land via ``--params``.

    Refuse each loudly (dispatch-hardening), pointing at the effective path, so a
    ``--params`` override is never silently dropped.  Everything run_coupled owns
    and passes through unchanged — atmosphere scalars, sea-ice, coupler, and the
    non-carbon land config under ``--land-params analytical`` — still applies.
    """
    if not params:
        return
    keys = set(params)

    ocean = sorted(k for k in keys if k.split(".")[0] == "ocean")
    if ocean:
        raise SystemExit(
            f"run_coupled --params: ocean parameter(s) {ocean} are not settable "
            "here — the coupled driver rebuilds the ocean config from the "
            "mesh/preset during setup.  Calibrate the ocean via "
            "`run_omip.py --params`."
        )

    carbon = sorted(k for k in keys if k.startswith("land.carbon."))
    if carbon:
        raise SystemExit(
            f"run_coupled --params: land carbon parameter(s) {carbon} are not "
            "settable here — the coupled driver rebuilds the carbon config "
            "during setup (carbon-active / differland, and the multilayer "
            "diurnal-surface force-upgrade).  Calibrate land carbon via "
            "`run_lmip.py --params`."
        )

    if land_params == "clm":
        land = sorted(k for k in keys if k.split(".")[0] == "land")
        if land:
            raise SystemExit(
                f"run_coupled --params: land parameter(s) {land} are not settable "
                "under --land-params clm (the default) — the CLM reference "
                "surfdata maps and the per-PFT parameter provider override the "
                "config-level land parameters during setup.  Pass --land-params "
                "analytical to calibrate land via --params, or use "
                "`run_lmip.py --params`."
            )


class _CoupledParamsBundle(NamedTuple):
    """--params routing bundle: every non-atmosphere component config
    run_coupled passes explicitly to ``CoupledESMDriver``.  A single bundle
    application keeps the loader's absent/ambiguous detection exact."""

    coupled: object
    coupler: object
    ice: object
    lake: object


def build_params_bundle(coupled_cfg, coupler_config=None) -> _CoupledParamsBundle:
    """The exact --params bundle ``apply_coupled_params`` routes into: the
    coupled config plus the coupler / sea-ice / lake configs (defaults when
    None — ``CoupledESMDriver`` builds the identical defaults, so a no-params
    run is unchanged).  The reachability audit
    (tests/unit/test_params_reachability_audit.py) walks THIS bundle, so the
    audited routing surface cannot drift from what main() applies."""
    from legoesm.coupler.config import CouplerConfig
    from legoesm.coupler.lake.config import LakeConfig
    from legoesm.ice.config import SeaIceConfig

    return _CoupledParamsBundle(
        coupled=coupled_cfg,
        coupler=coupler_config or CouplerConfig(),
        ice=SeaIceConfig(),
        lake=LakeConfig(),
    )


def apply_coupled_params(params_path, land_params, atm_config, coupled_cfg,
                         coupler_config):
    """Apply the --params calibration layer (issue #691) across EVERY component
    config this driver builds: atmosphere params route to the flattened
    ExperimentConfig scalars (scalar map); land params route into the
    coupled_cfg's nested land_config; coupler/ice/lake params route into the
    coupler/sea-ice/lake configs (built here with their defaults and passed
    explicitly — CoupledESMDriver builds the identical defaults when they are
    None, so a no-params run is unchanged).

    Returns ``(atm_config, coupled_cfg, coupler_config, ice_config,
    lake_config)``; ice/lake stay ``None`` when no non-atmosphere params are
    given.  Single source of truth for the split-and-bundle application:
    ``main()`` calls this, and the unit tests exercise it directly
    (tests/unit/test_run_coupled_config_yaml.py)."""
    from legoesm.driver.run_config_yaml import (
        apply_params_to_config,
        build_atm_scalar_param_map,
        load_params_config,
    )

    ice_config = None
    lake_config = None
    params = load_params_config(params_path)
    _check_params_clobber(params, land_params)
    amap = build_atm_scalar_param_map()
    atm_params = {k: v for k, v in params.items() if k in amap}
    rest_params = {k: v for k, v in params.items() if k not in amap}
    if atm_params:
        atm_config = apply_params_to_config(
            atm_config, atm_params, driver="run_coupled",
            scalar_param_map=amap)
    if rest_params:
        bundle = apply_params_to_config(
            build_params_bundle(coupled_cfg, coupler_config),
            rest_params, driver="run_coupled")
        coupled_cfg = bundle.coupled
        coupler_config = bundle.coupler
        ice_config = bundle.ice
        lake_config = bundle.lake
    return atm_config, coupled_cfg, coupler_config, ice_config, lake_config


def land_scheme_overrides(land_scheme: str) -> dict:
    """CoupledConfig overrides selecting the land surface model.

    The coupler dispatches on the land-config TYPE, so the (land_mode,
    land_config) pair must agree: ``MultiLayerLandConfig`` -> 8-layer soil
    thermal + Richards soil-moisture column tile; ``LandConfig`` -> 1-layer slab.
    Raises on an unknown scheme (dispatch hardening)."""
    from legoesm.land.config import LandConfig, MultiLayerLandConfig
    if land_scheme == "multilayer":
        return {"land_mode": "multilayer", "land_config": MultiLayerLandConfig()}
    if land_scheme == "slab":
        return {"land_mode": "slab", "land_config": LandConfig()}
    raise ValueError(
        f"land_scheme must be one of {_LAND_SCHEMES}, got {land_scheme!r}.")


#: Max atm-ocean coupling interval [days] for the dynamic 3D ocean.  diag_days
#: sets the integration segment length and the coupler fires once per segment,
#: so diag_days IS the coupling interval; looser than this overheats the atm on
#: stale SST and goes unstable (see clamp_coupling_diag_days).
_MAX_COUPLED_DIAG_DAYS = 10


def clamp_coupling_diag_days(ocean: str, diag_days: int) -> int:
    """Clamp diag_days to a tight coupling cadence for the dynamic 3D ocean.

    Returns ``min(diag_days, _MAX_COUPLED_DIAG_DAYS)`` when ``ocean == "dynamic"``
    (diag_days is the atm-ocean coupling interval there), else ``diag_days``
    unchanged.  Pure + side-effect-free so it is unit-testable.
    """
    if ocean == "dynamic" and diag_days > _MAX_COUPLED_DIAG_DAYS:
        return _MAX_COUPLED_DIAG_DAYS
    return diag_days


# The --config YAML loader is the shared single source of truth
# (legoesm.driver.run_config_yaml) reused by run_amip.py — see main(), which
# imports it deferred.  run_coupled-specific example dests for the unknown-key
# error hint:
_COUPLED_EXAMPLE_KEYS = (
    "'surface_bulk_scheme', 'ocean', 'surface_gustiness_zi', "
    "'bulk_thermo_convention', 'cloud_q_c_diagnostic', "
    "'ocean_restore_sst_tau_days'"
)


def _find_latest_checkpoint(output_dir):
    """Return ``(atm_checkpoint_path, day_token)`` for the highest-day
    ``checkpoint_day_NNNN.npz`` in ``output_dir``, or ``(None, None)`` if none
    exists.  Used by ``--resume`` to chain multi-segment equilibration jobs.
    Pure + side-effect-free so it is unit-testable."""
    import glob
    import os
    import re
    best, best_day = None, None
    for p in glob.glob(os.path.join(str(output_dir), "checkpoint_day_*.npz")):
        m = re.search(r"checkpoint_day_(\d+)\.npz$", os.path.basename(p))
        if m is None:
            continue
        d = int(m.group(1))
        if best_day is None or d > best_day:
            best, best_day = p, d
    return best, best_day


def build_parser():
    """Build the run_coupled argument parser (exposed for CLI round-trip tests)."""
    # Central microphysics literal set — keep the CLI allowlist in sync with
    # ExperimentConfig.validate_strict (no drift / no dropped advertised scheme).
    from legoesm.driver.config import VALID_MICROPHYSICS

    parser = argparse.ArgumentParser(
        description="Run a fully coupled ESM simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Preset
    parser.add_argument(
        "--preset", default="aquaplanet",
        choices=["aquaplanet", "slab_simple", "slab_pft",
                 "slab_richards", "slab_carbon", "full_coupled"],
        help="Configuration preset (default: aquaplanet)",
    )

    # Atmosphere
    parser.add_argument("--resolution", "-n", type=int, default=16,
                        help="Cubed-sphere resolution C{N} (default: 16)")
    parser.add_argument("--nlev", type=int, default=20,
                        help="Number of vertical levels (default: 20)")
    parser.add_argument("--dt", type=float, default=450.0,
                        help="Atmosphere time step [s] (default: 450)")
    parser.add_argument("--days", type=int, default=30,
                        help="Simulation duration [days] (default: 30)")
    parser.add_argument("--radiation", default="rrtmgp",
                        choices=["gray", "rrtmg", "rrtmgp"],
                        help="Radiation scheme (default: rrtmgp — true CMIP6 "
                             "GHG/cloud-radiative transfer; use --minimal-physics "
                             "or --radiation gray for a cheap idealized run)")
    parser.add_argument(
        "--rad-update-steps", type=int, default=4,
        help="Call radiation every N physics steps (Issue #316: N>1 dispatches "
             "the other steps to a no-radiation segment variant, cutting the "
             "rrtmgp compiled-segment compile from O(hours) to O(minutes); "
             "physically fine since radiation evolves slowly). Default 4 "
             "(amortizes the rrtmgp default).",
    )
    parser.add_argument(
        "--unfused-radiation", action=argparse.BooleanOptionalAction, default=True,
        help="Run radiation as a SEPARATE host-level jit (not fused into the "
             "lax.scan), so rrtmgp and the dynamics scan compile as two small "
             "executables instead of one ~3h module. Requires --rad-update-steps>1. "
             "Default ON (needed to make the rrtmgp default compile in minutes; "
             "~1e-6 phase-shift vs the fused path). Pass --no-unfused-radiation "
             "for the byte-identical legacy fused path.",
    )
    parser.add_argument(
        "--orbital-insolation", action="store_true", default=False,
        dest="orbital_insolation",
        help="Use realistic (Berger 1978) orbital insolation: present-day "
             "orbital declination + Earth-Sun distance factor (a/r)^2 "
             "eccentricity asymmetry (~+/-3.4%%). Default off = circular "
             "orbit. Recommended for CMIP historical/abrupt-4xCO2/1pctCO2.",
    )
    # RRTMGP g-point compile/runtime tuning (forward CMIP runs only — these are
    # ANSWER-IDENTITY for a non-AD forward integration).  ``--rrtmgp-gpoint-
    # batch-size N>0`` processes the two-stream g-points in vmap blocks of N
    # (~6x faster radiation on GPU; FORWARD/inference only — NOT AD-safe).
    # ``--no-rrtmgp-gpoint-checkpoint`` selects a plain ``lax.scan`` over
    # g-points (one reused scan-body kernel) instead of the
    # ``jax.checkpoint(prevent_cse=True)`` path, shrinking the compiled-code
    # footprint (relieves the XLA-CPU LLVM-JIT code-region pressure / cuts the
    # rrtmgp cold-compile).  The checkpoint is ONLY needed for reverse-mode AD
    # memory, which a forward coupled run never uses.  Defaults match the
    # global config (0 / True = current behaviour); recommended for a forward
    # rrtmgp CMIP run: ``--rrtmgp-gpoint-batch-size 16 --no-rrtmgp-gpoint-checkpoint``.
    parser.add_argument("--rrtmgp-gpoint-batch-size", type=int, default=0,
                        help="RRTMGP g-point vmap block size (0=checkpointed "
                             "scan; >0=forward-only ~6x faster radiation)")
    parser.add_argument("--rrtmgp-gpoint-checkpoint",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Checkpoint the per-g-point two-stream scan "
                             "(default on = AD-safe; --no-... = smaller/faster "
                             "compile for forward-only runs)")
    parser.add_argument("--radiation-column-chunk", type=int, default=0,
                        help="RRTMGP column-chunk block size (0 = off). >0 maps "
                             "the rrtmgp solve over fixed-size column blocks so "
                             "the per-block XLA graph compiles ONCE at this size "
                             "— caps the super-linear rrtmgp compile time at "
                             "higher horizontal resolution. Numerically exact "
                             "(columns are independent); must divide the column "
                             "count.")
    # Atmosphere physics suite.  DEFAULT = full realistic CMIP6 atmosphere:
    # convection=sbm, turbulence=holtslag_boville, gravity-wave-drag=hines,
    # clouds=sundqvist, microphysics=kessler (+ rrtmgp radiation above).  This
    # suite is empirically stable coupled at coarse res (C18/L20, dt<=450s).
    # Pass --minimal-physics (or the individual --<scheme> none flags) for a
    # cheap idealized run.
    parser.add_argument("--convection", default="sbm",
                        choices=["sbm", "dca", "kuo", "mass_flux", "edmf", "none"],
                        help="Convection scheme (default: sbm)")
    parser.add_argument("--turbulence", default="holtslag_boville",
                        choices=["smagorinsky", "louis", "tke", "holtslag_boville",
                                 "mynn25", "clubb", "edmf", "none"],
                        help="Boundary-layer turbulence scheme "
                             "(default: holtslag_boville)")
    # NOTE: "most" is deliberately NOT offered here although the coupler
    # ocean tile accepts it: the atmosphere surface layer's
    # compute_surface_fluxes treats "most" as constant (fixed-roughness LAND
    # scheme), so offering it would silently split the interface (ocean tile
    # MOST vs atmosphere constant) — the exact inconsistency validate() below
    # rejects for turbulence="none".
    parser.add_argument("--surface-bulk-scheme", default="constant",
                        choices=["constant", "most", "coare3", "large_yeager"],
                        help="AIR-SEA surface bulk-flux algorithm for the "
                             "atmosphere surface layer + the coupler OCEAN tile "
                             "(the 3D-ocean air-sea flux). 'coare3' is the "
                             "ocean-appropriate choice (Charnock roughness + "
                             "convective-gustiness w*). The LAND and SLAB-ocean "
                             "tiles use their OWN scheme (--land-bulk-scheme / "
                             "--slab-bulk-scheme, default 'most'). 'constant' "
                             "(default, byte-identical) = neutral coefficients; "
                             "'most' = generic iterative MOST (fixed roughness); "
                             "'coare3'/'large_yeager' = ocean stability-dependent "
                             "MOST. Requires a turbulence scheme (not "
                             "--turbulence none).")
    parser.add_argument("--land-bulk-scheme", default="most",
                        choices=["constant", "most", "coare3", "large_yeager"],
                        help="Surface bulk-flux scheme for the LAND tile "
                             "(slab/multilayer land). Default 'most' = "
                             "Monin-Obukhov similarity with a land roughness "
                             "(no ocean Charnock); 'constant' = neutral "
                             "coefficients (legacy). Land already carries dynamic "
                             "water pools + dryness-limited ET (bucket / Richards "
                             "soil moisture), so MOST + soil-moisture stress gives "
                             "a physical land-atmosphere flux.")
    parser.add_argument("--slab-bulk-scheme", default="most",
                        choices=["constant", "most", "coare3", "large_yeager"],
                        help="Surface bulk-flux scheme for the SLAB / two-layer "
                             "ocean heat budget (--ocean slab|two_layer). Default "
                             "'most' (generic MOST). The prognostic 3D ocean "
                             "(--ocean dynamic) instead uses --surface-bulk-scheme "
                             "on the coupler ocean tile (coare3 for air-sea).")
    parser.add_argument("--gustiness-zi", dest="surface_gustiness_zi",
                        type=float, default=None,
                        help="COARE 3.0 convective-gustiness boundary-layer depth "
                             "z_i [m] for the MOST surface fluxes (needs "
                             "--surface-bulk-scheme coare3/large_yeager). Unset = "
                             "scheme-native (coare3: 600 m per AeroBulk/Fairall "
                             "2003; large_yeager/constant: off). 0 = force off. "
                             "The w* gust lets a calm warm ocean evaporate "
                             "(fixes the persistent tropical hfls<<Earth / R_TOA "
                             "imbalance). Applied to the atmosphere surface layer "
                             "AND the slab ocean heat budget (kept consistent).")
    parser.add_argument("--bulk-thermo-convention", dest="bulk_thermo_convention",
                        type=str, default="legoesm",
                        choices=["legoesm", "aerobulk"],
                        help="Thermodynamic constants set for the MOST bulk "
                             "fluxes (coare3/large_yeager): 'legoesm' (default) "
                             "= constant L_v / dry c_pd; 'aerobulk' = "
                             "NEMO/AeroBulk/COARE parity (SST-dependent L_vap, "
                             "moist cp_air). Applied CONSISTENTLY to the "
                             "atmosphere surface layer, the slab ocean heat "
                             "budget, and the coupler ocean tile (air-sea only; "
                             "land/ice/lake keep the default).")
    parser.add_argument("--surface-stability-scheme", default="dyer1974",
                        choices=["dyer1974", "beljaars_holtslag1991",
                                 "grachev2007_sheba", "gryanik2020"],
                        help="Stable-regime (zeta>0) Monin-Obukhov similarity "
                             "functions for the MOST-family surface bulk "
                             "schemes (coare3/large_yeager). Applied "
                             "CONSISTENTLY to BOTH the atmosphere surface "
                             "layer (SurfaceLayerConfig) and the coupler "
                             "ocean tile (CouplerConfig) so the interface "
                             "cannot split. 'dyer1974' (default) = the "
                             "historical linear -5*zeta, byte-identical; "
                             "'grachev2007_sheba'/'gryanik2020' = SHEBA-based "
                             "Arctic/strong-stable forms; "
                             "'beljaars_holtslag1991' avoids the stable flux "
                             "collapse. Unstable branch stays Businger-Dyer.")
    parser.add_argument("--gravity-wave-drag", default="hines",
                        choices=["rayleigh", "lindzen", "mcfarlane", "hines",
                                 "prognostic_spectral", "e3sm_cam", "ml_emulator",
                                 "none"],
                        help="Gravity-wave-drag scheme (default: hines)")
    parser.add_argument("--clouds", default="sundqvist",
                        choices=["none", "sundqvist", "xu_randall", "resolved"],
                        help="Cloud-fraction scheme (default: sundqvist)")
    parser.add_argument("--convective-cloud", dest="convective_cloud",
                        action="store_true", default=False,
                        help="Add a bounded Slingo(1987) convective cumulus "
                             "cloud-fraction source driven by the (lagged) "
                             "convective precip — restores the tropical "
                             "cloud-radiative effect the adjustment convection "
                             "scheme (sbm) + RH-based cloud miss (the ~4.5 K "
                             "coupled cold-bias fix).  Default off.")
    parser.add_argument("--rh-crit", dest="cloud_rh_crit", type=float,
                        default=None,
                        help="Override Sundqvist critical RH (CloudConfig."
                             "rh_crit). HIGHER => less stratiform cloud => LOWER "
                             "planetary albedo. Range [0.5, 0.99]. Default: "
                             "CloudConfig default (byte-identical). The SW knob "
                             "for the coare3 moisture-driven albedo overshoot.")
    parser.add_argument("--q-c-diagnostic", dest="cloud_q_c_diagnostic",
                        type=float, default=None,
                        help="Override diagnostic in-cloud condensate [kg/kg] "
                             "(CloudConfig.q_c_diagnostic). LOWER => optically "
                             "THINNER cloud => lower albedo, still LW-active. "
                             "Range [5e-5, 1e-3]. Default: CloudConfig default.")
    parser.add_argument("--conv-cloud-max", dest="cloud_conv_cloud_max",
                        type=float, default=None,
                        help="Override convective (Slingo) cloud-cover cap "
                             "(CloudConfig.conv_cloud_max). Range [0.1, 1.0]. "
                             "Default: CloudConfig default.")
    parser.add_argument("--conv-cloud-condensate",
                        dest="cloud_conv_cloud_condensate",
                        type=float, default=None,
                        help="Override in-cloud condensate [kg/kg] of the "
                             "convective anvil deck "
                             "(CloudConfig.conv_cloud_condensate). LOWER => "
                             "optically THINNER / more realistic anvil. Range "
                             "[1e-5, 1e-3]. Default: CloudConfig default.")
    parser.add_argument("--microphysics", default="morrison",
                        choices=list(VALID_MICROPHYSICS),
                        help="Microphysics scheme (default: morrison — the "
                             "ice-capable double-moment scheme; warm-rain-only "
                             "kessler leaves SUPERCOOLED LIQUID high cloud aloft "
                             "(no freeze->snow->precip sink), which drives the TOA "
                             "cold drift in coupled CMIP runs. Use --microphysics "
                             "kessler for the cheap warm-rain path; 'none' with "
                             "active convection gives pr=0 and a cloud-water trap)")
    parser.add_argument(
        "--minimal-physics", action="store_true",
        help="Override the full-physics defaults to a cheap idealized "
             "atmosphere: gray radiation, SBM convection only "
             "(no turbulence / GWD / clouds / microphysics). For fast "
             "aquaplanet / dynamical-core sanity runs.",
    )
    parser.add_argument("--diag-days", type=int, default=5,
                        help="Diagnostic interval [days] (default: 5)")

    # Ocean.  The coupled driver runs a thermodynamic SLAB ocean (no 3D
    # dynamics — that lives in the standalone OceanModel and is not yet wired
    # into the coupler).  DEFAULT = two_layer: a mixed layer + deep layer with
    # bulk vertical mixing and deep-layer restoring (a cold deep reservoir that
    # damps SST drift), the most ocean physics the coupled slab supports today.
    parser.add_argument("--ocean", default="two_layer",
                        choices=["fixed", "slab", "two_layer", "dynamic"],
                        help="Coupled ocean mode (default: two_layer slab). "
                             "'dynamic' = the prognostic 3D LatLonCGridOceanModel "
                             "stepped by the coupler on a SHARED lat-lon grid "
                             "(requires --grid latlon); slab/two_layer/fixed = "
                             "thermodynamic slab")
    parser.add_argument("--ocean-h-mix", type=float, default=50.0,
                        help="Slab ocean mixed-layer depth [m]")
    parser.add_argument("--grid", default="cubed_sphere",
                        choices=["cubed_sphere", "latlon"],
                        help="Atmosphere grid (default cubed_sphere); 'latlon' "
                             "is required for --ocean dynamic (shared grid)")
    parser.add_argument("--couple-surface-radiation",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Feed the coupler's tile-blended (land+ocean) skin "
                             "T/albedo back to atmosphere radiation (--ocean-ic "
                             "woa). Default on; the slab-land skin feedback is "
                             "stiff — turn off (--no-couple-surface-radiation) "
                             "to trade land-radiation realism for stability.")
    parser.add_argument("--land-scheme", choices=_LAND_SCHEMES,
                        default=None,
                        help="Override the preset's land surface model. 'slab' = "
                             "1-layer bucket; 'multilayer' = 8-layer soil thermal + "
                             "Richards soil moisture (column land). Both route through "
                             "the coupler land tile. Default (unset): keep the preset's "
                             "land (e.g. full_coupled=multilayer). --ocean-ic woa "
                             "defaults to slab when unset.")
    parser.add_argument("--land-params", choices=("analytical", "clm"),
                        default="clm",
                        help="Spatial land parameters when land is active. 'clm' "
                             "(default) = CLM reference surfdata: real global PFT "
                             "classification + reference soil map (downloaded + "
                             "cached on first use). 'analytical' = latitude-band "
                             "PFT fractions, no soil map.")
    parser.add_argument("--transient-land-cover", dest="transient_land_cover",
                        action="store_true", default=False,
                        help="Re-weight the CLM land vegetation params each segment "
                             "from --land-cover-surfdata's transient pft_frac(year,...) "
                             "(LUH2/HYDE/...); soil frozen.  Requires --land-params clm.")
    parser.add_argument("--land-cover-surfdata", dest="land_cover_surfdata", default="",
                        help="Transient legoesm_surfdata NetCDF (multi-year "
                             "pft_frac) for --transient-land-cover.")
    parser.add_argument("--land-diurnal-surface", dest="land_diurnal_surface",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Coupled diurnal surface model for multilayer land (ON "
                             "by default): physical Monin-Obukhov (MOST) surface "
                             "exchange + Farquhar photosynthesis-stomata coupling.  "
                             "--no-land-diurnal-surface reverts to a constant bulk "
                             "coefficient + soil-only beta.")
    parser.add_argument("--elev-bands", dest="land_elev_bands",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Sub-grid elevation-band snow for multilayer+CLM land "
                             "(OFF by default): re-partition precip phase / melt over "
                             "sub-grid elevation bands (CLM STD_ELEV) so warm cells "
                             "keep bright snow on cold high fractions -- fixes the "
                             "high-elevation / perennial-snow warm-albedo bias.")
    parser.add_argument("--snow-albedo-feedback", dest="snow_albedo_feedback",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Enable the land snow-albedo feedback + latitude-"
                             "varying vegetation albedo (surface_albedo: veg "
                             "0.15 tropics / 0.20 midlat / 0.25 highlat, with "
                             "snow-covered land brightening toward ~0.6-0.8).  "
                             "OFF (default) leaves land at a constant 0.2 — too "
                             "DARK over snow-covered high-latitude land (should "
                             "be bright snow).  Recommended ON for realistic "
                             "land/cryosphere surface albedo.")
    parser.add_argument("--warm-start-soil", dest="warm_start_soil",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Warm-start the land soil at the atmosphere's "
                             "lat-structured near-surface air temperature (t=0) "
                             "instead of the uniform 280 K default.  The uniform "
                             "default starts tropical land soil ~18 K too cold, "
                             "and the slow multilayer soil takes months to spin "
                             "up — dragging global near-surface air T down. "
                             "Default off (byte-identical); recommended ON for a "
                             "faster, more realistic land spin-up.")
    parser.add_argument("--stomata", dest="stomata",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="STOMATAL CONDUCTANCE control of land "
                             "evapotranspiration (StomataConfig.enabled).  DEFAULT "
                             "ON for realistic land ET; --no-stomata => land ET is "
                             "limited only by the bucket/Richards soil-moisture "
                             "beta (no physiological control).  ON => the "
                             "effective beta is further "
                             "limited by stomatal conductance: with the DifferLand "
                             "carbon scheme (--preset slab_carbon) it is the "
                             "CO2-COUPLED Farquhar + Ball-Berry/Medlyn solver "
                             "(transpiration responds to atmospheric CO2 via "
                             "forcing.co2_ppmv); without carbon it is the Jarvis "
                             "multiplicative model (no CO2).  Soil moisture still "
                             "depletes dynamically (bucket dW/dt=P-E / Richards "
                             "theta).  Recommended ON for realistic land ET.")
    parser.add_argument("--polar-filter", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Fourier polar filter for the lat-lon C-grid "
                             "(default on for --grid latlon; ignored on cube). "
                             "Truncates the high-wavenumber lon modes that "
                             "violate the pole-cell CFL, so dt is set by the "
                             "EQUATORIAL CFL (~60x larger dt at 2deg) instead of "
                             "being clamped to ~5s. Without it a 2deg lat-lon "
                             "run is ~80x more steps and infeasible.")
    parser.add_argument("--ocean-nlev", type=int, default=20,
                        help="3D ocean vertical levels (--ocean dynamic)")
    parser.add_argument("--ocean-dt", type=float, default=300.0,
                        help="3D ocean SUBSTEP dt [s] (--ocean dynamic); the "
                             "coupler substeps the ocean at this dt within each "
                             "coupling_dt (never step the 3D ocean at 3600 s)")
    parser.add_argument("--ocean-H-max", type=float, default=5500.0,
                        help="Max ocean depth [m] (--ocean dynamic)")
    parser.add_argument("--ocean-ic", default="rest",
                        choices=["rest", "woa"],
                        help="3D ocean initial condition (--ocean dynamic): "
                             "'rest' = idealized aquaplanet rest state; 'woa' = "
                             "WOA18 reanalysis T/S + WOA-derived continents "
                             "(realistic cold start; f_land co-derived from the "
                             "same ocean mask, land tile enabled)")
    parser.add_argument("--woa-t-path",
                        default="data/woa18/woa18_decav_t00_01.nc",
                        help="WOA18 temperature file (--ocean-ic woa)")
    parser.add_argument("--woa-s-path",
                        default="data/woa18/woa18_decav_s00_01.nc",
                        help="WOA18 salinity file (--ocean-ic woa)")
    parser.add_argument("--ocean-restore-sst-tau-days", type=float, default=0.0,
                        help="3D ocean (--ocean dynamic --ocean-ic woa): Newtonian "
                             "relaxation timescale [days] for the surface "
                             "temperature toward the WOA initial state. 0 = off. "
                             "Anchors the surface against the cold-start drift "
                             "during the coupled spin-up (the gustiness fix alone "
                             "is insufficient; ~30 d is a moderate start).")
    parser.add_argument("--ocean-restore-sss-tau-days", type=float, default=0.0,
                        help="3D ocean (--ocean dynamic --ocean-ic woa): Newtonian "
                             "relaxation timescale [days] for the surface salinity "
                             "toward the WOA initial state. 0 = off.")
    parser.add_argument("--tripole-mesh", default=None,
                        help="NEMO eORCA mesh_mask file (e.g. "
                             "data/grids/eORCA1.2_mesh_mask.nc).  When set with "
                             "--ocean dynamic, the 3D ocean runs on the TRIPOLE "
                             "grid (a DIFFERENT grid from the lat-lon atmosphere) "
                             "coupled via the Phase-2 cross-grid conservative "
                             "remap, using the OMIP-validated cold-start recipe. "
                             "The land mask + bathymetry come from this file.")
    parser.add_argument(
        "--fold-convention", default="auto",
        choices=["auto", "n_lon-1-i", "(n_lon-i)%n_lon"],
        help="Tripole T-fold seam origin (default auto). 'auto' raises on a "
             "genuinely ambiguous (near-constant) fold row; pass the convention "
             "explicitly for such a mesh ('n_lon-1-i' halo-inclusive eORCA1.2, "
             "'(n_lon-i)%%n_lon' de-haloed eORCA025).")

    # Atmosphere initial condition.  CRITICAL for realism: the bare
    # ExperimentConfig default ic="default" is a UNIFORM T_init (~isothermal
    # ~300 K) scaffold — convectively dead-stable (no lapse rate => convection
    # never triggers), warm everywhere (huge q_sat => CWV ~80 kg/m2, ~3x
    # Earth), and warm aloft (OLR ~390 W/m2).  A coupled run from this IC spends
    # many days spinning up before it precipitates.  ic="standard" overlays a
    # realistic constant-lapse-rate troposphere + cold isothermal stratosphere +
    # equator-pole gradient (CWV ~15-30 kg/m2, convectively active, OLR ~240).
    # NOTE: ic="standard" is currently wired for --grid latlon ONLY (the cube
    # path needs the balanced-jet component rotation); the default below is
    # grid-aware so a cube run still works.
    parser.add_argument("--ic", default=None,
                        choices=["default", "standard", "era5"],
                        help="Atmosphere initial condition (default: 'standard' "
                             "on --grid latlon, 'default' on cube). 'standard' = "
                             "realistic lapse-rate troposphere + cold "
                             "stratosphere (Earth-like CWV/OLR, fast spin-up to "
                             "a precipitating state); 'default' = uniform T_init "
                             "scaffold; 'era5' = ERA5 reanalysis (needs --ic-path)")
    parser.add_argument("--ic-path", default="",
                        help="ERA5 Zarr path when --ic era5")

    # Carbon
    parser.add_argument("--co2-init", type=float, default=415.0,
                        help="Initial CO2 concentration [ppmv]")

    # CMIP6 experiment / output
    parser.add_argument(
        "--experiment", default="",
        help="CMIP6 experiment id (e.g. historical, ssp585, piControl, "
             "1pctCO2, abrupt-4xCO2). Selects the transient external-forcing "
             "trajectory (GHG/ozone/aerosol/solar). Empty = idealized/constant "
             "(default).",
    )
    parser.add_argument(
        "--start-year", type=int, default=1979,
        help="Calendar start year used to index CMIP6 forcing "
             "(e.g. 1850 for historical) (default: 1979)",
    )
    parser.add_argument(
        "--cmip-output", action="store_true",
        help="Write CMOR-style monthly NetCDF output (tas, pr, tos, siconc, ...)",
    )
    parser.add_argument(
        "--cmip-resolution-deg", type=float, default=5.0,
        help="Lat-lon grid spacing for CMIP output [deg] (default: 5.0)",
    )

    # Devices
    parser.add_argument(
        "--n-devices", type=int, default=None, metavar="N",
        help="Number of GPUs to use (default: auto-select largest valid count)",
    )

    # Output
    parser.add_argument("--output", "-o", default="results/coupled",
                        help="Output directory")
    parser.add_argument("--config", default=None,
                        help="YAML run-config file (e.g. config/cmip/"
                             "cmip_ocean_slab.yaml): its keys set argument "
                             "DEFAULTS, so any explicit CLI flag still overrides "
                             "it. Keys are run_coupled argument dests; an unknown "
                             "key is a hard error (no silent typo'd override).")
    parser.add_argument("--params", default=None,
                        help="YAML calibration file of tuned parameters keyed by "
                             "param_collector qualified name 'scheme_key.field' "
                             "(e.g. atm.clouds.CloudConfig.q_c_diagnostic); "
                             "validated against __param_spec__ bounds and applied "
                             "to the flattened atmosphere ExperimentConfig scalar "
                             "fields. Applied after --config/CLI. (issue #691)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in --output "
                             "(atm checkpoint_day_NNNN.npz + coupled "
                             "coupled_day_NNNN.npz), continuing the integration "
                             "from that day instead of the initial condition. "
                             "Enables job-chained multi-month equilibration of "
                             "the dynamic 3D ocean (ckpt v2). No checkpoint "
                             "present => starts fresh.")
    parser.add_argument("--checkpoint-days", type=int, default=0,
                        help="Write a full coupled checkpoint every N sim-days "
                             "(0=off). Needed for --resume job-chaining; a "
                             "small N bounds the work lost to an abrupt cancel.")
    parser.add_argument("--max-wallclock-hours", type=float, default=0.0,
                        help="Wallclock budget (hours): checkpoint and exit "
                             "cleanly before this elapsed time so SLURM does "
                             "not kill the job mid-step (0=off). Set it just "
                             "under the SLURM --time so the next --resume link "
                             "picks up the exact end state.")

    return parser


def main():
    parser = build_parser()

    # Two-pass parse so a --config file supplies defaults that explicit CLI
    # flags still override (precedence: CLI > config file > parser default).
    pre, _ = parser.parse_known_args()
    if pre.config is not None:
        from legoesm.driver.run_config_yaml import load_yaml_config
        parser.set_defaults(**load_yaml_config(
            pre.config, parser, example_keys=_COUPLED_EXAMPLE_KEYS))

    args = parser.parse_args()

    # --minimal-physics: collapse the full-physics defaults to a cheap
    # idealized atmosphere (gray radiation + SBM convection only).  Applied
    # AFTER parsing so it cleanly overrides whatever the per-scheme defaults
    # are, without fighting argparse precedence.
    if args.minimal_physics:
        args.radiation = "gray"
        args.turbulence = "none"
        args.gravity_wave_drag = "none"
        args.clouds = "none"
        args.microphysics = "none"
        args.unfused_radiation = False
        args.rad_update_steps = 1
        args.ocean = "slab"          # cheap single-layer slab for idealized runs

    # Atmosphere IC default is GRID-AWARE: ic="standard" (the realistic
    # lapse-rate troposphere + cold stratosphere) is wired for BOTH --grid
    # latlon AND cubed_sphere (validated: CWV ~17/30 kg/m2, precip ~1.5/3.3
    # mm/day, stable; vs the uniform ic="default" scaffold's CWV ~84, precip
    # ~0).  spectral/mpas have no grid-space T Field, so they keep the uniform
    # default.  An explicit --ic standard on an unsupported grid still raises
    # loudly in ExperimentConfig.validate_strict (no silent degrade).
    if args.ic is None:
        args.ic = ("standard" if args.grid in ("latlon", "cubed_sphere")
                   else "default")

    # Coupling-interval guard (dynamic 3D ocean): diag_days sets the integration
    # SEGMENT length, and the atm<->ocean coupler (_segment_hook) — which steps
    # the ocean and refreshes the SST the atmosphere sees — fires ONCE PER
    # SEGMENT.  So diag_days IS the coupling interval.  A large diag_days lets
    # the atmosphere integrate many days on a FIXED (stale) SST, which overheats
    # the column and goes unstable (measured: --diag-days 20 -> column-T 258.9K,
    # max_v 28.6, NaN by ~day 40; --diag-days 10 stays at 252.7K, stable).  Clamp
    # to a tight coupling cadence for the dynamic ocean so infrequent-output runs
    # don't silently loosen the coupling.  (Proper fix: a coupling interval
    # independent of the output interval in model_driver's segment-length calc.)
    _clamped = clamp_coupling_diag_days(args.ocean, args.diag_days)
    if _clamped != args.diag_days:
        logger.warning(
            "--diag-days %d is too loose for --ocean dynamic (diag_days sets the "
            "atm-ocean coupling interval); clamping to %d to keep the coupling "
            "tight and avoid the stale-SST overheating instability.",
            args.diag_days, _clamped)
        args.diag_days = _clamped

    # Unfused radiation only engages when rad_update_steps > 1 (the host-loop
    # dispatch in _run_compiled requires it).  Make the no-op EXPLICIT rather
    # than silently falling back to the fused path.
    if args.unfused_radiation and args.rad_update_steps <= 1:
        logger.warning(
            "--unfused-radiation requires --rad-update-steps>1; got %d. "
            "Disabling unfused radiation (would silently no-op).",
            args.rad_update_steps,
        )
        args.unfused_radiation = False

    logger.info("=" * 60)
    logger.info("  legoESM Coupled ESM")
    logger.info("=" * 60)
    logger.info(f"  Preset:     {args.preset}")
    logger.info(f"  Resolution: C{args.resolution}/L{args.nlev}")
    logger.info(f"  Days:       {args.days}")
    logger.info(f"  Radiation:  {args.radiation}"
                + (" (unfused)" if args.unfused_radiation else ""))
    _full_suite = (
        args.radiation == "rrtmgp" and args.convection != "none"
        and args.turbulence != "none" and args.gravity_wave_drag != "none"
        and args.clouds != "none" and args.microphysics != "none"
    )
    _suite_tag = (
        "  [full CMIP6 suite]" if _full_suite
        else "  [MINIMAL]" if args.minimal_physics else "  [custom]"
    )
    logger.info(
        "  Physics:    conv=%s turb=%s gwd=%s clouds=%s micro=%s%s"
        % (args.convection, args.turbulence, args.gravity_wave_drag,
           args.clouds, args.microphysics, _suite_tag)
    )
    logger.info(f"  Ocean:      slab/{args.ocean}"
                + ("  (deep restoring)" if args.ocean == "two_layer" else ""))
    logger.info(f"  Experiment: {args.experiment or '(idealized/constant)'}"
                f"  start_year={args.start_year}")
    logger.info(f"  CMIP out:   {args.cmip_output}"
                + (f" @ {args.cmip_resolution_deg}deg" if args.cmip_output else ""))
    logger.info(f"  Devices:    {args.n_devices if args.n_devices is not None else 'auto'}")
    logger.info(f"  Backend:    {jax.default_backend()}")
    logger.info(f"  X64:        {jax.config.jax_enable_x64}")
    logger.info("=" * 60)

    # Build configs
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.ocean.simple_ocean import SimpleOceanConfig

    atm_config = ExperimentConfig(
        grid=GridConfig(
            grid_type=args.grid,
            resolution=args.resolution,
            nlev=args.nlev,
        ),
        dycore=DycoreConfig(
            dt=args.dt, model_type="hydrostatic",
            # On lat-lon, use the Arakawa-C-grid hydrostatic dycore so the atm
            # co-locates with the C-grid 3D ocean (--ocean dynamic); cdgrid is
            # cube-only.  Cube keeps the default cdgrid.
            discretization=("latlon_cgrid" if args.grid == "latlon"
                            else "cdgrid"),
            # Fourier polar filter (lat-lon only): lets the factory/CFL clamp dt
            # by the equatorial CFL instead of the ~60x-smaller pole-cell dx, so
            # a 2deg run uses dt~450s (5760 steps/30d) not dt~5.6s (460k steps).
            use_polar_filter=(args.grid == "latlon" and args.polar_filter),
        ),
        output=OutputConfig(
            diag_days=args.diag_days,
            cmip_output=args.cmip_output,
            cmip_resolution_deg=args.cmip_resolution_deg,
            # Periodic checkpoint cadence (days) + a wallclock budget that
            # triggers a clean checkpoint+exit before SLURM kills the job — both
            # needed so a long dynamic-3D-ocean equilibration survives an abrupt
            # cancel / walltime and resumes via --resume (ckpt v2).
            checkpoint_days=args.checkpoint_days,
            max_wallclock_seconds=(args.max_wallclock_hours * 3600.0
                                   if args.max_wallclock_hours else 0.0),
        ),
        radiation=args.radiation,
        rad_update_steps=args.rad_update_steps,
        unfused_radiation=args.unfused_radiation,
        orbital_insolation=args.orbital_insolation,
        rrtmgp_gpoint_batch_size=args.rrtmgp_gpoint_batch_size,
        rrtmgp_gpoint_checkpoint=args.rrtmgp_gpoint_checkpoint,
        rrtmgp_column_chunk_size=args.radiation_column_chunk,
        ic=args.ic,
        ic_path=args.ic_path,
        convection=args.convection,
        turbulence=args.turbulence,
        surface_bulk_scheme=args.surface_bulk_scheme,
        surface_gustiness_zi=args.surface_gustiness_zi,
        surface_thermo_convention=args.bulk_thermo_convention,
        surface_stability_scheme=args.surface_stability_scheme,
        gravity_wave_drag=args.gravity_wave_drag,
        cloud_scheme=args.clouds,
        convective_cloud=args.convective_cloud,
        cloud_rh_crit=args.cloud_rh_crit,
        cloud_q_c_diagnostic=args.cloud_q_c_diagnostic,
        cloud_conv_cloud_max=args.cloud_conv_cloud_max,
        cloud_conv_cloud_condensate=args.cloud_conv_cloud_condensate,
        microphysics=args.microphysics,
        days=args.days,
        experiment=args.experiment,
        start_year=args.start_year,
        n_devices=args.n_devices if args.n_devices is not None else "auto",
    )
    # (--params is applied AFTER coupled_cfg / coupler_config are built, just
    # before driver construction, so it can reach every component's config —
    # see the calibration block below; issue #691.)

    # Build coupled config from preset with overrides.  The ocean_config is
    # ALWAYS overridden from --ocean so the coupled default is the two_layer
    # slab (the presets all set a single-layer mode="slab"); two_layer enables
    # deep-layer restoring (a cold reservoir that damps SST drift) — the most
    # ocean physics the coupled slab supports.  Every preset holds a
    # SimpleOceanConfig, so replacing it is type-safe.
    overrides = {}
    ocean_grid_obj = None   # None => ocean co-located on the atm grid (no remap)
    if args.ocean == "dynamic":
        # Prognostic 3D LatLonCGridOceanModel.  Two grid configurations:
        #   * SHARED lat-lon (default): the ocean lives on the atmosphere's
        #     lat-lon grid (no remap).  Requires --grid latlon.
        #   * TRIPOLE (--tripole-mesh): the ocean runs on the eORCA tripole grid
        #     (a DIFFERENT grid from the lat-lon atm), coupled via the Phase-2
        #     cross-grid conservative remap (coupler.grid_remap).
        from legoesm.ocean.state import LatLonCGridOceanConfig
        if args.grid != "latlon":
            raise SystemExit(
                "--ocean dynamic requires --grid latlon (the atmosphere is "
                "lat-lon; the 3D ocean is either co-located lat-lon or, with "
                "--tripole-mesh, the tripole grid coupled by the cross-grid "
                "remap).  Cube-atm + tripole-ocean needs the deferred "
                "cross-family remap.")
        overrides["ocean_mode"] = "dynamic"
        overrides["ocean_config"] = LatLonCGridOceanConfig()
        overrides["ocean_nlev"] = args.ocean_nlev
        overrides["ocean_dt_s"] = args.ocean_dt
        overrides["ocean_H_max_m"] = args.ocean_H_max
        overrides["ocean_ic"] = args.ocean_ic
        # WOA surface restoring (coupled spin-up anchor); 0 => off => unchanged.
        overrides["ocean_restore_sst_tau_days"] = args.ocean_restore_sst_tau_days
        overrides["ocean_restore_sss_tau_days"] = args.ocean_restore_sss_tau_days
        if (args.ocean_restore_sst_tau_days > 0.0
                or args.ocean_restore_sss_tau_days > 0.0):
            if args.ocean_ic != "woa":
                raise SystemExit(
                    "--ocean-restore-*-tau-days requires --ocean-ic woa "
                    "(the restoring target is the WOA climatology).")
            logger.info("  3D-ocean WOA restoring: SST tau=%.1f d, SSS tau=%.1f d",
                        args.ocean_restore_sst_tau_days,
                        args.ocean_restore_sss_tau_days)
        if args.tripole_mesh:
            # Build the tripole geometry from the NEMO mesh and pass it as a
            # DISTINCT ocean grid (make_grid_remapper builds the atm<->tripole
            # cross-grid remap; _init_tripole_dynamic_ocean clones the OMIP
            # cold-start recipe and reads mask+bathy from this same mesh).
            from legoesm.grids.tripole import create_tripole_grid
            ocean_grid_obj = create_tripole_grid(
                args.tripole_mesh, fold_convention=args.fold_convention)
            overrides["tripole_mesh_path"] = args.tripole_mesh
            logger.info(f"  Ocean grid: TRIPOLE from {args.tripole_mesh} "
                        f"({ocean_grid_obj.n_lat}x{ocean_grid_obj.n_lon}); "
                        f"atm lat-lon -> tripole cross-grid remap")
        if args.ocean_ic == "woa":
            # Realistic WOA cold start: observed T/S + WOA-derived continents.
            overrides["woa_t_path"] = args.woa_t_path
            overrides["woa_s_path"] = args.woa_s_path
            # Co-derive the atmosphere land fraction from the SAME ocean mask
            # and enable a land tile over the continents (f_land>0 with
            # land_mode='none' would try to run an unused land model).
            overrides["f_land_mode"] = "from_ocean"
            # Select the land surface model (coupler dispatches on the config
            # type: MultiLayerLandConfig -> Richards column tile, else slab).
            # woa defaults to slab when --land-scheme is unset.
            overrides.update(land_scheme_overrides(args.land_scheme or "slab"))
            # With real continents the atmospheric radiative surface boundary
            # SHOULD be the tile-blended (land+ocean) skin T / albedo, not the
            # ocean SST everywhere (else land cells radiate at the dynamic-ocean
            # SST; codex MED).  But the slab-land skin temperature is a stiff
            # radiative feedback that can destabilise the coarse coupled run, so
            # it is gated by --couple-surface-radiation (default on; turn off to
            # trade land-radiation realism for stability).
            overrides["couple_surface_radiation"] = args.couple_surface_radiation
    elif args.ocean == "two_layer":
        overrides["ocean_config"] = SimpleOceanConfig(
            mode="two_layer", h_mix=args.ocean_h_mix, restore_deep=True,
            # The SLAB/two-layer ocean uses its own bulk scheme (default MOST);
            # the prognostic 3D ocean is what gets COARE on the coupler tile.
            # thermo_convention keeps the slab heat-budget turbulent fluxes
            # constant-set-consistent with the atmosphere surface layer.
            bulk_scheme=args.slab_bulk_scheme,
            gustiness_w_zi=(args.surface_gustiness_zi or 0.0),
            thermo_convention=args.bulk_thermo_convention,
        )
        overrides["ocean_mode"] = "two_layer"
    else:
        overrides["ocean_config"] = SimpleOceanConfig(
            mode=args.ocean, h_mix=args.ocean_h_mix,
            bulk_scheme=args.slab_bulk_scheme,
            gustiness_w_zi=(args.surface_gustiness_zi or 0.0),
            thermo_convention=args.bulk_thermo_convention,
        )
        # ocean_mode log label (fixed/slab -> "slab").
        overrides["ocean_mode"] = "slab"
    if args.co2_init != 415.0:
        overrides["co2_ppmv_init"] = args.co2_init

    # Spatial land parameters: CLM reference map (real PFT + soil) by default.
    overrides["land_param_source"] = args.land_params
    if args.land_params == "clm":
        overrides["use_pft"] = True
    overrides["land_diurnal_surface"] = args.land_diurnal_surface
    overrides["land_elev_bands"] = args.land_elev_bands
    overrides["transient_land_cover"] = args.transient_land_cover
    overrides["land_cover_surfdata"] = args.land_cover_surfdata

    # Explicit --land-scheme overrides the preset's land model for ANY ocean mode
    # (the woa branch already applied its own default above; re-applying the same
    # explicit value is idempotent). Unset -> keep the preset's land choice.
    if args.land_scheme is not None:
        overrides.update(land_scheme_overrides(args.land_scheme))

    coupled_cfg = PRESETS[args.preset](**overrides)

    # Land snow-albedo feedback + lat-varying vegetation albedo (opt-in): the
    # presets build the land config with snow_albedo_feedback=False, which holds
    # land at a constant 0.2 — too dark over snow-covered high-latitude land.
    # Enabling it activates the surface_albedo veg-by-latitude + snow-brightening
    # scheme (both slab and multilayer land read config.snow_albedo_feedback).
    if (getattr(args, "snow_albedo_feedback", False)
            and coupled_cfg.land_mode != "none"
            and coupled_cfg.land_config is not None):
        coupled_cfg = coupled_cfg._replace(
            land_config=coupled_cfg.land_config._replace(
                snow_albedo_feedback=True))
        logger.info("  Land albedo: snow-albedo feedback + lat-varying "
                    "vegetation albedo ENABLED")

    # Land surface bulk-flux scheme (default MOST): wire --land-bulk-scheme onto
    # the LandConfig / MultiLayerLandConfig so the land tile computes its
    # turbulent fluxes with Monin-Obukhov similarity (land roughness, no ocean
    # Charnock) instead of the legacy constant coefficients.  The land model
    # already carries dynamic water pools + dryness-limited ET (bucket / Richards
    # soil moisture), so MOST + the soil-moisture stress gives a physical
    # land-atmosphere flux.  The config-level default stays "constant" (other
    # callers / tests byte-identical); only run_coupled opts into MOST.
    if (coupled_cfg.land_mode != "none"
            and coupled_cfg.land_config is not None
            and hasattr(coupled_cfg.land_config, "bulk_scheme")):
        coupled_cfg = coupled_cfg._replace(
            land_config=coupled_cfg.land_config._replace(
                bulk_scheme=args.land_bulk_scheme))
        logger.info("  Land surface flux scheme: %s (land tile; dynamic water "
                    "pools + dryness-limited ET active)", args.land_bulk_scheme)

    # Stomatal conductance (opt-in): enable physiological control of land ET so
    # transpiration is NOT just soil-moisture-limited.  compute_effective_beta
    # then routes through the CO2-coupled Farquhar+Ball-Berry/Medlyn solver when
    # the DifferLand carbon scheme is active (CO2 via forcing.co2_ppmv), else the
    # Jarvis model.  Default config keeps stomata.enabled=False (byte-identical).
    if (getattr(args, "stomata", False)
            and coupled_cfg.land_mode != "none"
            and coupled_cfg.land_config is not None
            and hasattr(coupled_cfg.land_config, "stomata")):
        coupled_cfg = coupled_cfg._replace(
            land_config=coupled_cfg.land_config._replace(
                stomata=coupled_cfg.land_config.stomata._replace(enabled=True)))
        _co2_coupled = (coupled_cfg.carbon_active
                        and coupled_cfg.carbon_land == "differland")
        logger.info("  Stomatal conductance ENABLED (%s); land ET physiologically "
                    "limited, soil moisture depletes dynamically",
                    "CO2-coupled Farquhar/Ball-Berry" if _co2_coupled
                    else "Jarvis (no CO2 — use --preset slab_carbon for "
                         "CO2-coupling)")
        if not _co2_coupled:
            logger.warning(
                "  --stomata WITHOUT the DifferLand carbon scheme: stomata use "
                "the Jarvis model (no CO2 response).  For CO2-coupled stomatal "
                "conductance use --preset slab_carbon (carbon_active + differland "
                "+ co2_tracer).")

    # Soil warm-start (opt-in): init soil at the atmosphere's lat-structured
    # near-surface air T (t=0) instead of a uniform 280 K cold start.
    if getattr(args, "warm_start_soil", False) and coupled_cfg.land_mode != "none":
        coupled_cfg = coupled_cfg._replace(warm_start_soil=True)
        logger.info("  Soil warm-start ENABLED (atm near-surface air T at t=0)")

    # Create and run driver
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    # Keep the coupler ocean-tile bulk-flux scheme consistent with the
    # atmosphere surface layer (interface energy balance: the flux leaving the
    # slab must match the flux entering the atmosphere).  Only override when the
    # user opts out of "constant" so the default run stays byte-identical (the
    # driver builds the default CouplerConfig when coupler_config is None).
    coupler_config = None
    if (args.surface_bulk_scheme != "constant"
            or args.surface_stability_scheme != "dyer1974"):
        from legoesm.coupler.config import CouplerConfig
        # Thread the SAME convective-gustiness BL depth onto the coupler ocean
        # tile that the atmosphere surface layer uses (--gustiness-zi), so the
        # air-sea interface flux is energy-consistent: the latent heat the
        # ocean loses == the moisture flux the atmosphere gains.  Without this
        # the 3D-ocean q_net used the non-gusty tile flux -> weak evaporation
        # -> dry atmosphere -> cold collapse (cmip_air_sea_decoupling).
        coupler_config = CouplerConfig(
            gustiness_w_zi=(args.surface_gustiness_zi or 0.0),
            thermo_convention=args.bulk_thermo_convention,
            stability_scheme=args.surface_stability_scheme,
        )
        logger.info("  Surface bulk-flux scheme: %s (thermo: %s; stability: %s; "
                    "atmosphere + coupler ocean tile); convective "
                    "gustiness z_i=%.0f m",
                    args.surface_bulk_scheme, args.bulk_thermo_convention,
                    args.surface_stability_scheme,
                    (args.surface_gustiness_zi or 0.0))

    # Apply the --params calibration layer (issue #691) across EVERY component
    # config this driver builds — see apply_coupled_params (the single source
    # of truth for the atm-scalar-map / coupled-bundle split, exercised
    # directly by the unit tests and the reachability audit).
    ice_config = None
    lake_config = None
    if getattr(args, "params", None):
        (atm_config, coupled_cfg, coupler_config, ice_config,
         lake_config) = apply_coupled_params(
            args.params, args.land_params, atm_config, coupled_cfg,
            coupler_config)

    driver = CoupledESMDriver(
        atm_config, coupled_cfg, coupler_config=coupler_config,
        ice_config=ice_config, lake_config=lake_config,
        ocean_grid=ocean_grid_obj, output_dir=args.output,
    )

    t0 = time.time()
    driver.setup()
    t_setup = time.time() - t0
    logger.info(f"Setup completed in {t_setup:.1f}s")

    # --resume: continue the integration from the latest checkpoint in --output
    # (atm + coupled written together, same day token).  setup() has already
    # rebuilt the IC + ocean geometry + WOA restoring targets deterministically;
    # the load overwrites the prognostic state with the saved day-N values.
    start_step, start_day = 0, None
    if getattr(args, "resume", False):
        import os
        ckpt, _tok = _find_latest_checkpoint(args.output)
        if ckpt is None:
            logger.info("  --resume: no checkpoint in %s; starting fresh",
                        args.output)
        else:
            step, day = driver._atm.load_checkpoint(ckpt)
            elapsed = day - getattr(atm_config, "start_day", 0.0)
            driver.load_coupled_checkpoint(float(elapsed),
                                           checkpoint_dir=args.output)
            start_step, start_day = step, day
            logger.info("  RESUME from %s: step=%d, day=%.1f (elapsed %.1f) -> "
                        "integrating to day %d", os.path.basename(ckpt),
                        step, day, elapsed, args.days)

    t0 = time.time()
    status = driver.run(start_step=start_step, start_day=start_day)
    t_run = time.time() - t0

    # Summary
    logger.info("=" * 60)
    logger.info(f"  Status: {status}")
    logger.info(f"  Wall time: {t_run:.1f}s ({t_run/60:.1f} min)")
    logger.info(f"  Per sim-day: {t_run / max(args.days, 1):.1f}s")

    # Final SST: slab stores T_sfc [K]; the dynamic 3D ocean stores top-level T
    # [degC] -> convert.  Use the driver's grid-agnostic accessor.  For the
    # dynamic ocean with a realistic land mask, reduce over OCEAN cells only
    # (land cells carry an inert abyssal-fill T that would cold-bias the mean).
    import numpy as _np
    sst = _np.asarray(driver._ocean_surface_KuvC()[0])
    _omask = getattr(driver, "_ocean_land_mask", None)  # 1=ocean, 0=land
    if _omask is not None:
        _wet = _np.asarray(_omask) > 0.5
        sst_red = sst[_wet] if _wet.any() else sst
    else:
        sst_red = sst
    logger.info(f"  SST final (ocean): mean={float(_np.nanmean(sst_red)):.1f}K, "
                f"range=[{float(_np.nanmin(sst_red)):.1f}, "
                f"{float(_np.nanmax(sst_red)):.1f}]K")

    if driver.coupled_diagnostics:
        d0 = driver.coupled_diagnostics[0]
        df = driver.coupled_diagnostics[-1]
        drift = df["sst_mean"] - d0["sst_mean"]
        logger.info(f"  SST drift: {drift:.2f}K over {args.days} days "
                    f"({drift / max(args.days, 1) * 365:.1f} K/yr)")
        if "co2_ppmv_mean" in df:
            logger.info(f"  CO2: {df['co2_ppmv_mean']:.1f} ppmv")

    # Ocean-circulation snapshot (dynamic 3D ocean only): the prognostic
    # currents are not in the atmosphere DiagnosticCollector, so dump the final
    # ocean state's C-grid velocities (centred to cell centres), the
    # depth-integrated transport (barotropic-streamfunction proxy), SST, and the
    # wet mask + grid to ocean_circulation.npz for the circulation figure.
    # Defensive: a diagnostic dump must never fail the run.
    if args.ocean == "dynamic" and getattr(driver, "ocean_state", None) is not None:
        try:
            os_ = driver.ocean_state
            u = _np.asarray(os_.u.data)          # (nlat, nlon+1, nlev) [m/s]
            v = _np.asarray(os_.v.data)          # (nlat+1, nlon, nlev) [m/s]
            uc = 0.5 * (u[:, :-1, :] + u[:, 1:, :])   # -> cell centres
            vc = 0.5 * (v[:-1, :, :] + v[1:, :, :])
            T = _np.asarray(os_.T.data)          # (nlat, nlon, nlev) [degC]
            z = driver._ocean_z_coord
            dz = _np.asarray(getattr(z, "dz_ref", _np.ones(u.shape[-1])))
            dz = dz.reshape(1, 1, -1) if dz.ndim == 1 else dz
            Utr = _np.sum(uc * dz, axis=-1)      # depth-integrated zonal transport
            Vtr = _np.sum(vc * dz, axis=-1)
            g = driver._ocean_grid
            lat = _np.degrees(_np.asarray(getattr(g, "lat")))
            lon = _np.degrees(_np.asarray(getattr(g, "lon")))
            _np.savez(
                Path(args.output) / "ocean_circulation.npz",
                u_sfc=uc[..., 0], v_sfc=vc[..., 0],
                speed_sfc=_np.hypot(uc[..., 0], vc[..., 0]),
                U_transport=Utr, V_transport=Vtr,
                T_sfc=T[..., 0], land_mask=_np.asarray(driver._ocean_land_mask),
                lat=lat, lon=lon,
            )
            logger.info("  Ocean circulation saved: ocean_circulation.npz "
                        f"(max sfc current {float(_np.nanmax(_np.hypot(uc[...,0], vc[...,0]))):.2f} m/s)")
        except Exception as _e:   # pragma: no cover - diagnostic only
            logger.warning(f"  Ocean-circulation dump skipped: {_e}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
