#!/usr/bin/env python
"""Reference OMIP simulation — ocean-only forced integration on all grids.

Runs a realistic ocean simulation using WOA18-based initialization (or
analytical fallback), full physics stack (KPP, GM/Redi, SW penetration,
convection, bottom drag), and restoring surface forcing.  Supports all
four ocean grids: cubed-sphere, lat-lon, MPAS, and spectral.

Usage:
    # Quick 30-day smoke test on cubed-sphere:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py --quick --grid cubed_sphere

    # All grids, 1 year:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py --days 365 --grid all

    # Full physics with WOA18 data:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py \\
        --days 365 --grid cubed_sphere --woa-t woa18_t.nc --woa-s woa18_s.nc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # annotation-only; the runtime import stays function-scoped
    from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
        IWMConfig,
    )

sys.stdout.reconfigure(line_buffering=True)

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# Precision policy is applied from --precision in main() via
# legoesm.runtime.precision.apply_precision (default fp64 == prior behavior:
# OMIP ran unconditional fp64).  x64 is enabled at import (above) so the
# fp64/mixed accumulate+control roles stay exact regardless of mode;
# apply_precision installs the per-module mixed overrides when requested.
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig,
    VisbeckConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import (
    KPPConfig,
    VerticalMixingConfig,
)

# ===========================================================================
# Grid types and default resolutions / timesteps
# ===========================================================================

GRID_TYPES = ["cubed_sphere", "latlon", "mpas", "spectral", "tripole"]

GRID_DEFAULTS: dict[str, dict] = {
    # cubed_sphere now uses the FC-Gram spectral baroclinic-tendency
    # backend (see _create_setup) which eliminates the face-edge PGF
    # halo amplification that previously blew up the cd-grid path
    # at ~4 days under WOA SST restoring.  With FC + the existing
    # tuned defaults (A_h, K_h, nbaro, restoring tau + ramp) the
    # 30-day smoke run completes with physical max_speed ≈ 3e-5 m/s.
    # dt=60 retained out of caution; FC step is ~1.6× slower per
    # iteration than cdgrid because of the per-face Fourier
    # continuation cost.
    "cubed_sphere": {"resolution": "C24", "dt": 60.0},
    "latlon":       {"resolution": "36x72", "dt": 300.0},
    "mpas":         {"resolution": "ico3", "dt": 300.0},
    "spectral":     {"resolution": "T21", "dt": 300.0},
    # Tripolar eORCA1 (332×362 mesh). Matches the validated 20-yr
    # idealized production runner (run_tripole_20yr.py): dt=600s,
    # 20 levels, ETOPO bathymetry + 80°N cap removal.
    "tripole":      {"resolution": "eorca1", "dt": 600.0},
}

ALL_RESULTS: list[dict] = []

_VALID_VERTICAL_MIXING_SCHEMES = ("kpp", "tke", "catke", "richardson", "constant", "none")
_DEFAULT_KPP_CONFIG = KPPConfig()
from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (  # noqa: E402
    IWMConfig as _IWMConfig,
)
_DEFAULT_IWM_CONFIG = _IWMConfig()

# Production GM/Redi config for the realistic-bathymetry (ETOPO) lat-lon path,
# hoisted from _create_setup so a --params calibration file can override its
# tunables (kappa_GM/kappa_Redi/S_max, the Visbeck adaptive-kappa knobs, and
# the Treguier-1997 adaptive-kappa cap aei0 — #691/#724 reachability audit).
# Values are unchanged from the inline construction (byte-identical default).
_DEFAULT_BATHY_GM_REDI = GMRediConfig(
    kappa_GM=800.0,
    kappa_Redi=800.0,
    S_max=0.005,
    visbeck=VisbeckConfig(
        enabled=True,
        alpha=0.015,
        kappa_min=200.0,
        kappa_max=2000.0,
    ),
)


class OMIPRunConfig(NamedTuple):
    """CLI-resolved run controls that are not a single ocean model config."""

    max_wallclock_seconds: float
    restart_buffer_seconds: float
    seed: int
    vertical_mixing: VerticalMixingConfig
    precision: str = "fp64"
    # GM/Redi bundle threaded into _create_setup's realistic-bathymetry
    # lat-lon path (inert on flat-bottom / other-grid runs; --no-gm-redi
    # still disables it entirely).  Carried here so --params can reach
    # GMRediConfig / VisbeckConfig / TreguierConfig (#691/#724).
    gm_redi: GMRediConfig = _DEFAULT_BATHY_GM_REDI
    # Lat-band SPMD (single-controller multi-GPU) for the lat-lon restoring
    # lane: wraps the loop's dynamics step in ``make_sharded_ocean_step``
    # (the #751/#758-validated lane-D step).  0 devices = all local.
    enable_latlon_spmd: bool = False
    spmd_n_devices: int = 0
    # Route-B multicontroller (jax.distributed cross-process NCCL): promote the
    # lat-band lane to span ALL global devices across processes (multi-node).
    multicontroller: bool = False
    coordinator: str | None = None


from legoesm.driver.checkpoint import wallclock_exhausted as _wallclock_exhausted


def build_vertical_mixing_config_from_args(
    args,
    *,
    default_scheme: str = "kpp",
) -> VerticalMixingConfig:
    """Resolve the OMIP vertical-mixing CLI flags into the physics config."""
    scheme = args.vertical_mixing_scheme or default_scheme
    if scheme == "catke":
        # CATKE (Wagner 2025) uses its own VerticalMixingConfig.catke defaults
        # (calibrated); the kpp-tuning CLI flags don't apply.  Implicit-only.
        #
        # iwm/ddm DO apply: both are ADDITIVE onto avt/avs/avm INSIDE
        # compute_vertical_K_profiles, AFTER the primary closure (NEMO's zdfphy
        # ordering), so they are independent of which closure ran.  This branch
        # previously omitted them, silently DROPPING `--iwm` whenever the user
        # also passed `--vertical-mixing-scheme catke` -- the flag parsed, and
        # the run just had no internal-wave mixing.  Only the KPP-tuning flags
        # are legitimately inapplicable here.
        from legoesm.ocean.physics.vertical_mixing.config import CATKEConfig
        return VerticalMixingConfig(
            scheme="catke", catke=CATKEConfig(),
            iwm=build_iwm_config_from_args(args),
            ddm=build_ddm_config_from_args(args),
        )
    return VerticalMixingConfig(
        scheme=scheme,
        kpp=KPPConfig(
            Ri_crit=args.kpp_ri_crit,
            K_max=args.kpp_k_max,
            K_bg=args.kpp_k_bg,
            K_conv=args.kpp_k_conv,
            A_bg=args.kpp_a_bg,
            enable_langmuir=args.langmuir,
            langmuir_coeff=args.langmuir_coeff,
            langmuir_number_default=args.langmuir_number_default,
        ),
        iwm=build_iwm_config_from_args(args),
        ddm=build_ddm_config_from_args(args),
    )


def build_ddm_config_from_args(args):
    """Resolve the zdfddm (Merryfield 1999 / Large-CVMix) CLI flag into config.

    Only ``enabled`` is threaded: the float knobs (rn_avts, rn_hsbfr) carry a
    ``__param_spec__`` and so are reachable via ``--params ocean.vm.ddm.*``,
    which is the repo's convention for spec'd float tunables.
    """
    from legoesm.ocean.physics.vertical_mixing.double_diffusion import (
        DoubleDiffusionConfig,
    )
    return DoubleDiffusionConfig(enabled=bool(getattr(args, "ddm", False)))


def build_iwm_config_from_args(args) -> "IWMConfig":
    """Resolve the zdfiwm (de Lavergne 2020) CLI flags into IWMConfig."""
    from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
        IWMConfig,
    )
    return IWMConfig(
        enabled=bool(getattr(args, "iwm", False)),
        mevar=bool(getattr(args, "iwm_mevar", False)),
        tsdiff=bool(getattr(args, "iwm_tsdiff", False)),
        power_bot_wm2=args.iwm_power_bot,
        power_cri_wm2=args.iwm_power_cri,
        power_nsq_wm2=args.iwm_power_nsq,
        power_sho_wm2=args.iwm_power_sho,
        scale_bot_m=args.iwm_scale_bot,
        scale_cri_m=args.iwm_scale_cri,
    )


def build_config_from_args(args) -> OMIPRunConfig:
    """Build the CLI-resolved OMIP config fragments used by the driver."""
    return OMIPRunConfig(
        max_wallclock_seconds=args.max_wallclock_seconds,
        restart_buffer_seconds=args.restart_buffer_seconds,
        seed=args.seed,
        vertical_mixing=build_vertical_mixing_config_from_args(args),
        precision=args.precision,
        enable_latlon_spmd=getattr(args, "enable_latlon_spmd", False),
        spmd_n_devices=getattr(args, "spmd_n_devices", 0),
        multicontroller=getattr(args, "multicontroller", False),
        coordinator=getattr(args, "coordinator", None),
    )


def _apply_drag_iwm_overrides(args, grid_type, grid, z_coord, config, model):
    """Post-``_create_setup`` application of the NEMO zdfdrg drag-law flags
    and the zdfiwm forcing maps (mirrors the run_omip_core2 replace-flat +
    model-rebuild pattern; a plain config edit needs the model rebuilt so
    the jitted step captures it).

    Returns ``(config, model)`` — unchanged (bit-identical objects) when
    no NEMO drag scheme / IWM flag is active.
    """
    drag_flat = {}
    if args.bottom_drag_scheme != "legacy":
        drag_flat = dict(
            bottom_drag_scheme=args.bottom_drag_scheme,
            bottom_drag_cd0=args.bottom_drag_cd0,
            bottom_drag_cdmax=args.bottom_drag_cdmax,
            bottom_drag_z0=args.bottom_drag_z0,
            bottom_drag_ke0=args.bottom_drag_ke0,
        )
    # Wide-halo split-explicit barotropic (lat-lon band scaling lever):
    # nested BarotropicConfig fields, reachable via the flat-name mapping.
    want_wide_halo = bool(getattr(args, "barotropic_wide_halo", False))
    if want_wide_halo:
        drag_flat = dict(
            drag_flat,
            barotropic_wide_halo=True,
            barotropic_wide_halo_chunk=getattr(
                args, "barotropic_wide_halo_chunk", 0),
            # The wide path's per-substep clamp is LOCAL by construction;
            # the config validator REQUIRES the local-clamp scheme to be
            # explicit, so the flag sets it (documented in --help).
            barotropic_local_subcycle_clamp=True,
        )
    want_iwm = bool(getattr(args, "iwm", False))
    # --ddm must be in this guard too.  The block below is the ONLY thing that
    # puts an additive mixing rider onto config.physics, and the flat-bottom
    # lat-lon path ships physics=None -- so an early return here makes the flag
    # a silent no-op.  That is exactly what the codex r2 #1 comment below
    # records for --iwm; --ddm was added later and walked into the same trap.
    want_ddm = bool(getattr(args, "ddm", False))
    if not drag_flat and not want_iwm and not want_ddm:
        return config, model

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    if isinstance(model, LatLonCGridOceanModel):
        if drag_flat:
            config = config.replace_flat(**drag_flat)
        iwm_forcing = None
        if want_iwm or want_ddm:
            # Make sure the IWM/DDM config ACTUALLY reaches the implicit
            # solve (codex r2 #1: the flat-bottom lat-lon path
            # ships config.physics=None, so without this --iwm would be a
            # silent no-op — k_profiles never sees vertical_mixing.iwm).
            # DDM rides here for the same reason, though it lands in a
            # different place: IWM is added onto the tracer AND momentum
            # profiles in k_profiles, whereas DDM contributes heat/salt-only
            # diffusivities (avm untouched) applied later in
            # ocean_model_latlon_cgrid's implicit salinity solve.
            _iwm_cfg = build_iwm_config_from_args(args)
            _ddm_cfg = build_ddm_config_from_args(args)
            _phys = config.physics
            if _phys is None:
                from legoesm.ocean.physics.combined import OceanPhysicsConfig
                from legoesm.ocean.physics.vertical_mixing.config import (
                    VerticalMixingConfig,
                )
                from legoesm.ocean.physics.lateral_mixing.config import (
                    LateralMixingConfig,
                )
                from legoesm.ocean.physics.surface_forcing.config import (
                    SurfaceForcingConfig,
                )
                from legoesm.ocean.physics.convection.config import (
                    OceanConvectionConfig,
                )
                # Minimal pipeline: every module inert except the IWM/DDM
                # riders (the flat path's diffusion stays config-based).
                _phys = OceanPhysicsConfig(
                    vertical_mixing=VerticalMixingConfig(
                        scheme="none", iwm=_iwm_cfg, ddm=_ddm_cfg),
                    lateral_mixing=LateralMixingConfig(scheme="none"),
                    surface_forcing=SurfaceForcingConfig(scheme="none"),
                    convection=OceanConvectionConfig(scheme="none"),
                    shortwave_penetration=None,
                )
            else:
                _phys = _phys._replace(
                    vertical_mixing=_phys.vertical_mixing._replace(
                        iwm=_iwm_cfg, ddm=_ddm_cfg))
            config = config._replace(physics=_phys)
            # Both riders contribute only through the IMPLICIT solve, and both
            # RAISE if enabled on an explicit path -- so force it rather than
            # let the run die on a guard the user cannot see from the flag.
            if not getattr(config, "implicit_vertical_mixing", False):
                config = config.replace_flat(implicit_vertical_mixing=True)
                _who = "zdfiwm" if want_iwm else "zdfddm"
                print(f"[setup] {_who}: implicit_vertical_mixing forced ON "
                      "(the additive avt/avs/avm enter the backward-Euler "
                      "solve)")
            # NEMO zdfiwm_init FORCES the model backgrounds to molecular
            # values (avmb = rnu = 1.4e-6 m²/s, avtb = 1e-10 m²/s): the
            # wave field IS the interior background.  Mirror that so the
            # OMIP A_v/K_v floors don't double-count (codex r1 #2).
            # zdfiwm ONLY: this is a zdfiwm_init convention (the wave field IS
            # the interior background), not a property of additive mixing in
            # general.  Applying it for a bare --ddm would silently strip the
            # user's A_v/K_v backgrounds.
            if want_iwm:
                from legoesm import constants as _const
                config = config.replace_flat(
                    A_v=_const.nu_ocean_molecular, K_v=1.0e-10)
                print("[setup] zdfiwm: model backgrounds forced to molecular "
                      f"(A_v={_const.nu_ocean_molecular:g}, K_v=1e-10) per "
                      "zdfiwm_init")
        if want_iwm and args.iwm_forcing_file:
            import numpy as _np
            from legoesm.ocean.iwm_forcing import load_iwm_forcing
            lat_T = getattr(grid, "lat_T", None)
            if lat_T is not None:
                # tripole: lat_T/lon_T are stored in RADIANS (2-D)
                lat_T = _np.degrees(_np.asarray(lat_T))
                lon_T = _np.degrees(_np.asarray(grid.lon_T))
            else:                                     # regular lat-lon (radians)
                lat_T = _np.degrees(_np.asarray(grid.lat))
                lon_T = _np.degrees(_np.asarray(grid.lon))
            iwm_forcing = load_iwm_forcing(
                args.iwm_forcing_file, lat_T, lon_T)
            print(f"[setup] zdfiwm forcing maps loaded from "
                  f"{args.iwm_forcing_file}")
        model = LatLonCGridOceanModel(
            grid, z_coord, config, iwm_forcing=iwm_forcing)
        return config, model

    if want_iwm:
        raise SystemExit(
            f"--iwm is supported on the lat-lon / tripole grids only "
            f"(the {grid_type} vertical-mixing bridge does not consume "
            f"IWM yet)")
    if want_ddm:
        # Same rule, same reason: only the lat-lon C-grid path routes the
        # salinity solve through the DDM avs.  Without this, --ddm was silently
        # INERT on cubed_sphere (setup ships physics=None and this branch never
        # wired it) -- a flag that parses and does nothing is the failure this
        # branch exists to remove, so reject it loudly instead (codex).
        raise SystemExit(
            f"--ddm is supported on the lat-lon / tripole grids only "
            f"(the {grid_type} vertical-mixing bridge does not consume "
            f"double-diffusive avs yet)")
    if want_wide_halo:
        raise SystemExit(
            f"--barotropic-wide-halo is supported on the lat-lon / tripole "
            f"C-grid ocean only (the wide-halo subcycle is a lat-band "
            f"path); the {grid_type} grid has no wide-halo barotropic")
    # Non-latlon models with flat drag fields (MPAS Voronoi, cubed-sphere):
    # replace the flat NamedTuple fields and rebuild the same model class.
    if not hasattr(config, "bottom_drag_scheme"):
        raise SystemExit(
            f"--bottom-drag-scheme={args.bottom_drag_scheme} is not "
            f"supported on the {grid_type} grid (its ocean config has no "
            f"bottom-drag law fields)")
    config = config._replace(**drag_flat)
    model = type(model)(grid, z_coord, config)
    return config, model


def apply_run_precision(args) -> None:
    """Apply the ``--precision`` policy globally (idempotent).

    Called from BOTH ``main()`` and ``run_omip_single()`` so that a direct
    in-process ``run_omip_single()`` caller cannot silently run at the wrong
    (default) precision: the unconditional module-import ``set_policy(fp64)``
    was removed, so the policy is now applied at every model-building entry
    point. The canonical bridge sets the global policy + x64 and installs the
    mixed-mode per-module overrides; unknown modes raise (dispatch-hardening).
    """
    from legoesm.runtime.precision import apply_precision
    apply_precision(args.precision)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Reference OMIP simulation on all ocean grids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file supplying argument defaults (keys are "
                        "argument dests; explicit CLI flags still override). "
                        "See config/omip/*.yaml. (issue #691)")
    p.add_argument("--require-config", action="store_true",
                   help="Strict mode: fail unless a --config file is given, so "
                        "the run is fully specified by a committed config (no "
                        "hidden parser defaults). (issue #691)")
    p.add_argument("--params", type=str, default=None,
                   help="YAML calibration file of tuned parameters keyed by "
                        "param_collector qualified name 'scheme_key.field', "
                        "validated against each scheme's __param_spec__ bounds "
                        "and spliced into the nested *Config (here the vertical-"
                        "mixing configs, e.g. KPPConfig). (issue #691)")
    p.add_argument("--surface-stability-scheme", default="dyer1974",
                   choices=["dyer1974", "beljaars_holtslag1991",
                            "grachev2007_sheba", "gryanik2020"],
                   help="Stable-regime (zeta>0) MOST similarity functions for "
                        "the air-sea (CouplerConfig, LY09 bulk) AND air-ice "
                        "(SeaIceConfig) turbulent fluxes. 'dyer1974' (default) "
                        "= historical -5*zeta, byte-identical; "
                        "'grachev2007_sheba' = SHEBA Arctic stable functions "
                        "(the sea-ice reference); 'gryanik2020' = modified "
                        "SHEBA; 'beljaars_holtslag1991' avoids stable flux "
                        "collapse. Unstable branch stays Businger-Dyer.")
    p.add_argument("--grid", type=str, default="all",
                   choices=GRID_TYPES + ["all"])
    p.add_argument("--resolution", type=str, default=None,
                   help="Grid resolution (e.g. C24, 36x72, ico3, T21)")
    p.add_argument("--nlev", type=int, default=40,
                   help="Ocean vertical levels (default 40). SOTA OMIP ocean "
                        "models use ~60-75 levels (NEMO ORCA1 L75, MOM6, POP2); "
                        "40 is the climate-usable minimum. Pass --nlev 20 for a "
                        "faster dev/matrix run.")
    p.add_argument("--H-max", type=float, default=5500.0)
    p.add_argument("--dt", type=float, default=None,
                   help="Timestep [s] (default: grid-specific)")
    p.add_argument("--days", type=float, default=365.0)
    p.add_argument("--quick", action="store_true",
                   help="Short 30-day run for CI")
    p.add_argument("--output", type=str, default="results/omip")
    p.add_argument("--checkpoint-days", type=float, default=30.0)
    p.add_argument("--max-wallclock-seconds", type=float, default=0.0,
                   help="Wallclock budget [s] for clean checkpoint+exit")
    p.add_argument("--restart-buffer-seconds", type=float, default=600.0,
                   help="Wallclock buffer [s] reserved for restart writes")
    p.add_argument("--seed", type=int, default=0,
                   help="Master RNG seed for reproducibility metadata")
    p.add_argument("--precision", type=str, default="fp64",
                   choices=["fp32", "fp64", "mixed"],
                   help=(
                       "Precision policy (default fp64 = prior OMIP behavior): "
                       "fp32 (storage+compute float32), fp64 (all float64), or "
                       "mixed (fp32 storage/compute, fp64 accumulate/control + "
                       "fp64 overrides on the precision-sensitive ocean kernels "
                       "barotropic_solver/pressure_gradient/equation_of_state/"
                       "coriolis). NOTE: mixed is NOT yet validated for "
                       "century-scale OMIP drift — see scripts/validate/."))
    p.add_argument("--woa-t", type=str, default=None,
                   help="WOA18 temperature NetCDF path")
    p.add_argument("--woa-s", type=str, default=None,
                   help="WOA18 salinity NetCDF path")
    p.add_argument("--woa-init", action="store_true",
                   help="Initialize T/S from WOA18 instead of rest state. "
                        "Requires --woa-t and --woa-s.")
    p.add_argument("--nudge-woa-tau", type=float, default=0.0,
                   help="Nudge T toward WOA18 with this restoring timescale [days]. "
                        "Applied after each block step. 0=disabled. "
                        "Typical 90-180 days for gentle spinup.")
    p.add_argument("--bathymetry", type=str, default=None,
                   help=(
                       "Path to ETOPO/GEBCO NetCDF bathymetry file. "
                       "When provided, loads realistic topography instead "
                       "of a flat-bottom domain."
                   ))
    p.add_argument("--H-min", type=float, default=10.0,
                   help="Minimum ocean depth [m]; shallower cells become land (default 10).")
    p.add_argument("--smoothing-passes", type=int, default=2,
                   help="Laplacian smoothing passes for bathymetry (default 2).")
    p.add_argument("--r-factor-max", type=float, default=0.2,
                   help="Maximum bathymetric slope r-factor for partial cells (default 0.2).")
    p.add_argument("--north-cap-lat", type=float, default=90.0,
                   help="Latitude [°N] above which all cells become land (default 80).")
    p.add_argument("--south-cap-lat", type=float, default=-80.0,
                   help=(
                       "Latitude [°S] below which all cells become land. "
                       "Set to -90 to disable the southern cap and let "
                       "ETOPO define Antarctica naturally (default -80)."
                   ))
    p.add_argument("--A-h", type=float, default=None,
                   help="Override Laplacian viscosity A_h [m²/s] (default: grid-dependent).")
    p.add_argument("--B-h", type=float, default=None,
                   help="Override biharmonic viscosity B_h [m⁴/s] (default: 5e9 for bathymetry).")
    p.add_argument("--K-h", type=float, default=None,
                   help="Override horizontal tracer diffusivity K_h [m²/s] (default: 1e3 with bathy).")
    p.add_argument("--no-lat-scaling", action="store_true",
                   help=(
                       "Disable cos²(lat) scaling of A_h. By default A_h is "
                       "scaled by cos²(lat) to keep the viscous CFL latitude-"
                       "independent.  This flag uses a constant A_h everywhere, "
                       "useful for diagnosing whether cos² scaling drives "
                       "high-latitude instability."
                   ))
    p.add_argument("--A-h-eq-boost", type=float, default=1.0,
                   help=(
                       "Equatorial A_h boost (>=1). Multiplies A_h by "
                       "1 + (boost-1)*exp(-(lat/sigma)^2) so eq momentum gets "
                       "extra dissipation. K_h (tracers) untouched. "
                       "Typical 3-10. 1=disabled."
                   ))
    p.add_argument("--A-h-eq-sigma", type=float, default=5.0,
                   help="Eq A_h boost Gaussian half-width [degrees]. Typical 3-7.")
    p.add_argument("--C-smag", type=float, default=None,
                   help="Smagorinsky biharmonic coefficient (dimensionless, OM4 uses 0.06).")
    p.add_argument("--C-smag-lap", type=float, default=0.15,
                   help="Laplacian Smagorinsky coefficient (dimensionless, default 0.15).")
    p.add_argument("--A-h-floor", type=float, default=2000.0,
                   help="Minimum effective A_h after latitude scaling [m²/s] (default 2000).")
    p.add_argument("--C-leith", type=float, default=None,
                   help="Leith biharmonic coefficient (dimensionless, typical 1.0-2.0).")
    p.add_argument("--pgf-scheme", type=str, default=None,
                   choices=["adcroft", "smc03"],
                   help="PGF scheme override (default smc03 with bathymetry).")
    p.add_argument("--slope-foot-alpha", type=float, default=0.0,
                   help=(
                       "Slope-foot viscosity enhancement (MOM6 OM4 KH_BG_2D analog). "
                       "Multiplies horizontal viscosity in bottom-N levels by "
                       "1 + alpha*tanh(|grad H|/H/0.1). 0=disabled, 3.0=production. "
                       "Targets African shelf, ITF, equatorial trench instabilities."
                   ))
    p.add_argument("--min-passage-width", type=int, default=0,
                   help=(
                       "Minimum passage width in grid cells. Passages narrower "
                       "than this are filled (become land). Set to 2 to eliminate "
                       "all 1-cell-wide straits that bottleneck WBCs (default 0=disabled)."
                   ))
    p.add_argument("--close-arctic-lat", type=float, default=None,
                   help=(
                       "Close off the Arctic completely above this latitude. "
                       "Unlike --north-cap-lat which just caps land, this makes "
                       "ALL cells above the latitude into land, creating a solid "
                       "wall. E.g. 65.0 closes off the entire Arctic basin."
                   ))
    p.add_argument("--sw-down", type=float, default=200.0,
                   help="Constant downwelling SW [W/m²]")
    p.add_argument("--physics", type=str, default="full",
                   choices=["full", "minimal", "none"])
    p.add_argument("--water-type", type=str, default="II",
                   choices=["I", "IA", "IB", "II", "III"])
    p.add_argument("--vertical-mixing-scheme", type=str, default=None,
                   choices=_VALID_VERTICAL_MIXING_SCHEMES,
                   help="Override vertical mixing scheme")
    p.add_argument("--kpp-ri-crit", type=float,
                   default=_DEFAULT_KPP_CONFIG.Ri_crit,
                   help="KPP critical bulk Richardson number")
    p.add_argument("--langmuir", action="store_true",
                   help="Enable KPP-Langmuir wave-enhanced surface mixing "
                        "(eps_L = sqrt(1 + C_L/La_t^2) on the KPP velocity scales).")
    p.add_argument("--langmuir-coeff", type=float,
                   default=_DEFAULT_KPP_CONFIG.langmuir_coeff,
                   help="KPP-Langmuir enhancement coefficient C_L")
    p.add_argument("--langmuir-number-default", type=float,
                   default=_DEFAULT_KPP_CONFIG.langmuir_number_default,
                   help="Fallback turbulent Langmuir number (no Stokes-drift input)")
    p.add_argument("--kpp-k-max", type=float,
                   default=_DEFAULT_KPP_CONFIG.K_max,
                   help="KPP maximum diffusivity [m^2/s]")
    p.add_argument("--kpp-k-conv", type=float,
                   default=_DEFAULT_KPP_CONFIG.K_conv,
                   help="KPP convective diffusivity [m^2/s]")
    p.add_argument("--kpp-k-bg", type=float,
                   default=_DEFAULT_KPP_CONFIG.K_bg,
                   help="KPP background diffusivity [m^2/s]")
    p.add_argument("--kpp-a-bg", type=float,
                   default=_DEFAULT_KPP_CONFIG.A_bg,
                   help="KPP background viscosity [m^2/s]")
    # --- wide-halo split-explicit barotropic (scaling-audit item 3) ---
    p.add_argument("--barotropic-wide-halo", action="store_true",
                   dest="barotropic_wide_halo",
                   help="Opt-in wide-halo split-explicit barotropic: one "
                        "fused wide lat-halo exchange per chunk of substeps "
                        "instead of ~4 halo pads per substep (lat-lon band "
                        "MPI/SPMD latency lever at >=16 ranks; serial "
                        "value-identical). Regular lat-lon C-grid only "
                        "(tripole fold refused at construction); requires "
                        "barotropic_solver=explicit_substep and ALSO SETS "
                        "barotropic_local_subcycle_clamp=True (the wide "
                        "path's per-substep clamp is local; global mass is "
                        "restored once per step).")
    p.add_argument("--barotropic-wide-halo-chunk", type=int, default=0,
                   dest="barotropic_wide_halo_chunk",
                   help="Substeps per wide exchange (0 = auto from the local "
                        "band height). With uneven --wet-balance bands set "
                        "it so chunk x stencil-reach <= min band height.")
    # --- internal wave-driven mixing (NEMO zdfiwm, de Lavergne 2020) ---
    _IWM_DEF = _DEFAULT_IWM_CONFIG
    # Double-diffusive mixing (NEMO zdfddm / Large-CVMix). Additive after the
    # primary closure, hence scheme-independent -- but NOT in the same place as
    # --iwm: IWM adds onto the tracer AND momentum profiles inside
    # compute_vertical_K_profiles, while DDM contributes heat/salt-only
    # diffusivities (avm untouched) applied later in the lat-lon model's
    # implicit salinity solve (codex corrected an earlier claim here).
    # ``DoubleDiffusionConfig.enabled`` is a BOOL, and only ``:float`` fields are
    # __param_spec__-eligible -- so --params can reach ddm's rn_avts/rn_hsbfr but
    # can NEVER reach this gate. Without this flag the whole scheme was
    # unreachable: implemented, oracle-pinned (PR #1074), and impossible to turn
    # on from any driver.
    p.add_argument("--ddm", action="store_true",
                   help="Enable double-diffusive mixing (salt fingering + "
                        "diffusive convection; additive avt/avs). Requires "
                        "implicit vertical mixing. Default off => bit-exact "
                        "legacy. Tune via --params ocean.vm.ddm.rn_avts=...")
    p.add_argument("--iwm", action="store_true",
                   help="Enable internal wave-driven mixing (NEMO zdfiwm; "
                        "additive avt/avm through the implicit vertical "
                        "solve; requires implicit vertical mixing).")
    p.add_argument("--iwm-mevar", action="store_true",
                   help="zdfiwm ln_mevar: variable mixing efficiency "
                        "(ORCA1 oracle: off).")
    p.add_argument("--iwm-tsdiff", action="store_true",
                   help="zdfiwm ln_tsdiff: differential T/S mixing "
                        "(unsupported on the shared-K solve; raises).")
    p.add_argument("--iwm-forcing-file", type=str, default=None,
                   help="NetCDF de Lavergne power/decay maps "
                        "(zdfiwm_forcing_TRA.nc layout).  Omit for the "
                        "uniform constant-power fallback.")
    p.add_argument("--iwm-power-bot", type=float,
                   default=_IWM_DEF.power_bot_wm2,
                   help="Uniform-fallback abyssal-hill power [W/m^2]")
    p.add_argument("--iwm-power-cri", type=float,
                   default=_IWM_DEF.power_cri_wm2,
                   help="Uniform-fallback critical-slope power [W/m^2]")
    p.add_argument("--iwm-power-nsq", type=float,
                   default=_IWM_DEF.power_nsq_wm2,
                   help="Uniform-fallback N^2-scaled power [W/m^2]")
    p.add_argument("--iwm-power-sho", type=float,
                   default=_IWM_DEF.power_sho_wm2,
                   help="Uniform-fallback shoaling power [W/m^2]")
    p.add_argument("--iwm-scale-bot", type=float,
                   default=_IWM_DEF.scale_bot_m,
                   help="Uniform-fallback abyssal-hill decay scale [m]")
    p.add_argument("--iwm-scale-cri", type=float,
                   default=_IWM_DEF.scale_cri_m,
                   help="Uniform-fallback critical-slope decay scale [m]")
    # --- NEMO zdfdrg bottom-drag laws ---
    p.add_argument("--bottom-drag-scheme", type=str, default="legacy",
                   choices=["legacy", "nemo_quadratic", "nemo_loglayer"],
                   help="Bottom-drag law: 'legacy' = historical MOM6-style "
                        "r/DRAG_BG_VEL path; 'nemo_quadratic' = zdfdrg "
                        "np_non_lin (the ORCA1 namelist); 'nemo_loglayer' "
                        "= zdfdrg np_loglayer.")
    p.add_argument("--bottom-drag-cd0", type=float, default=1.0e-3,
                   help="NEMO rn_Cd0 [-] (loglayer: Cd minimum)")
    p.add_argument("--bottom-drag-cdmax", type=float, default=0.1,
                   help="NEMO rn_Cdmax [-] (loglayer Cd cap)")
    p.add_argument("--bottom-drag-z0", type=float, default=3.0e-3,
                   help="NEMO rn_z0 bottom roughness [m]")
    p.add_argument("--bottom-drag-ke0", type=float, default=2.5e-3,
                   help="NEMO rn_ke0 background bottom KE [m^2/s^2]")
    p.add_argument("--no-conservation-fixer", action="store_true")
    p.add_argument("--restoring-timescale", type=float, default=None,
                   help=(
                       "SST/SSS restoring timescale [days].  Omitting this "
                       "uses 1095 days for latlon/mpas/spectral and 3650 "
                       "days for cubed_sphere (the slower default delays "
                       "the cubed-sphere face-edge PGF instability).  "
                       "Passing an explicit value overrides the default "
                       "for every grid, including cubed_sphere."
                   ))
    p.add_argument("--restoring-ramp-days", type=float, default=None,
                   help=(
                       "Ramp restoring strength linearly from 0 to full over "
                       "this many days at the start of the integration.  "
                       "Omitting this uses 0 days (full strength from step "
                       "1) for latlon/mpas/spectral and 14 days for "
                       "cubed_sphere.  Pass --restoring-ramp-days 0 "
                       "explicitly to disable the cubed-sphere ramp."
                   ))
    p.add_argument("--no-restoring", action="store_true",
                   help="Disable SST/SSS restoring")
    p.add_argument("--diag-every", type=int, default=None,
                   help="Diagnostic interval in steps (default: ~1 day)")
    # Tropical-OMIP forcing (Item 4 of tropical_omip_plan.md).
    p.add_argument("--forcing-mode", type=str, default="restoring",
                   choices=["restoring", "jra55_do_tropical"],
                   help=(
                       "Surface forcing source. 'restoring' (default) uses "
                       "Haney SST/SSS restoring toward WOA. 'jra55_do_tropical' "
                       "uses LY09 bulk fluxes from a pre-built JRA55-do cache "
                       "(see scripts/data/prepare_omip_forcing.py). Currently "
                       "supports only --grid latlon."
                   ))
    p.add_argument("--enable-latlon-spmd", action="store_true", default=False,
                   help=(
                       "Run the lat-lon lane's dynamics step lat-band-SPMD "
                       "across the local devices (make_sharded_ocean_step — "
                       "the validated multi-GPU ocean lane). Supports the "
                       "restoring lane AND the JRA55 block-scan lanes "
                       "(forcing stacks are lat-band-sharded; the in-scan "
                       "bulk fluxes stay shard-local). Single-controller "
                       "only (one process; multi-node scaling lives in "
                       "bench_ocean_latlon_spmd_scaling --multicontroller); "
                       "requires --grid latlon and n_lat divisible by the "
                       "device count. Unsupported: --jra55-sea-ice, the "
                       "JRA55 single-step fallback."
                   ))
    p.add_argument("--spmd-n-devices", type=int, default=0,
                   help=(
                       "Device count for --enable-latlon-spmd "
                       "(0 = all local devices)."
                   ))
    p.add_argument("--multicontroller", action="store_true", default=False,
                   help=(
                       "Promote --enable-latlon-spmd to ROUTE-B "
                       "(jax.distributed, cross-process NCCL): the lat-band "
                       "ocean mesh spans ALL global devices, one band per "
                       "device across every process — the multi-node OMIP "
                       "lane. Single-controller (one process, local devices) "
                       "is the default when this is off. Requires --grid "
                       "latlon. Launch under SLURM/mpiexec with one process "
                       "per GPU; only rank 0 writes restarts/output."
                   ))
    p.add_argument("--coordinator", type=str, default=None,
                   help=(
                       "jax.distributed coordinator address (host:port) for "
                       "--multicontroller under mpiexec (Open MPI OMPI_* / "
                       "Cray PALS PMI_* launcher env). Omit under SLURM/OMPI "
                       "for auto-detection."
                   ))
    p.add_argument("--jra55-cache", type=str, default=None,
                   help=(
                       "Path to JRA55-do Zarr cache. Required when "
                       "--forcing-mode=jra55_do_tropical."
                   ))
    p.add_argument("--jra55-co2-ppmv", type=float, default=400.0,
                   help=(
                       "Static atmospheric CO2 [ppmv] for JRA55-do mode "
                       "(default 400 — OMIP-2 protocol holds CO2 constant)."
                   ))
    p.add_argument("--jra55-cycle", action="store_true",
                   help=(
                       "Cycle the JRA55-do cache modulo its length. "
                       "Use this with a 1-year RYF cache (Stewart 2020) "
                       "for multi-year repeat-year-forcing runs. Default "
                       "off (cache must cover the requested run length, "
                       "e.g. for IAF mode)."
                   ))
    # Tropical-OMIP sponge / SSS restoring / freeze-cap (Day 3 of Item 4).
    p.add_argument("--sponge-lat-min", type=float, default=-60.0,
                   help="Southern boundary of tropical-OMIP active domain [°].")
    p.add_argument("--sponge-lat-max", type=float, default=60.0,
                   help="Northern boundary of tropical-OMIP active domain [°].")
    p.add_argument("--sponge-width-deg", type=float, default=5.0,
                   help="Sponge-zone width inside the active domain [°].")
    p.add_argument("--sponge-tau-days", type=float, default=5.0,
                   help="Sponge relaxation timescale at the boundary [days].")
    p.add_argument("--sss-piston-velocity", type=float, default=5.0e-7,
                   help=(
                       "SSS restoring piston velocity [m/s] "
                       "(default 5e-7 ≈ 200-day timescale at 10 m, NEMO/ORCA standard)."
                   ))
    p.add_argument("--T-ramp-days", type=float, default=1.0,
                   help=(
                       "Wind-stress spinup ramp timescale [days]. "
                       "tau is multiplied by min(1, t/T_ramp) to avoid "
                       "violent geostrophic adjustment from rest (default 1)."
                   ))
    p.add_argument("--jra55-no-sponge", action="store_true",
                   help="Disable the polar sponge layer.")
    p.add_argument("--jra55-no-sss-restoring", action="store_true",
                   help="Disable global SSS restoring.")
    p.add_argument("--jra55-no-freeze-cap", action="store_true",
                   help="Disable the T_freeze cap inside the sponge zone.")
    p.add_argument("--jra55-sea-ice", action="store_true", default=False,
                   dest="jra55_sea_ice",
                   help="Prognostic slab (thermodynamic) sea-ice tile coupled "
                        "to the ocean, replacing the freeze-cap SST stand-in: "
                        "open-ocean bulk fluxes scale by f_ocean=(1-A) and the "
                        "ice tile feeds basal heat / melt-freeze freshwater / "
                        "brine salt / ice-ocean stress to the ocean. Forces the "
                        "freeze cap OFF (no double-capping).")
    p.add_argument("--restart", type=str, default=None,
                   help=(
                       "Path to a restart_dayXXXXXX.npz file from a previous "
                       "run. When provided, the state is loaded from the "
                       "restart instead of initializing from rest. The time "
                       "loop starts from the restart day."
                   ))
    p.add_argument("--gpu-interp", action="store_true", default=True,
                   help=(
                       "Move JRA55 forcing interpolation from CPU to GPU "
                       "(DEFAULT, ~38%% faster). Loads only native 3-hourly "
                       "records and interpolates inside the lax.scan body, "
                       "reducing host-side I/O from ~288 to ~9 calls per "
                       "day-block. Use --no-gpu-interp to disable."
                   ))
    p.add_argument("--no-gpu-interp", action="store_false", dest="gpu_interp",
                   help="Disable GPU-side forcing interpolation (use CPU path).")
    p.add_argument("--no-gm-redi", action="store_true",
                   help="Disable GM/Redi isopycnal mixing (for diagnostic experiments).")
    p.add_argument("--implicit-vertical-mixing", action="store_true",
                   dest="implicit_vertical_mixing",
                   help=("Use backward-Euler implicit vertical viscosity and "
                         "diffusivity (issue #204).  Removes the explicit-CFL "
                         "limit dt < dz²/(2K) that becomes binding when "
                         "K_conv=1 m²/s convection fires with surface dz<30 m "
                         "or when vertical resolution is increased.  KPP non-"
                         "local fluxes remain explicit."))
    # Two-pass parse so a --config file supplies defaults that explicit CLI
    # flags still override (precedence: CLI > config file > parser default).
    # Shared loader (single source of truth) — same mechanism as run_amip /
    # run_coupled (issue #691).
    pre, _ = p.parse_known_args(argv)
    if pre.config is not None:
        from legoesm.driver.run_config_yaml import load_yaml_config
        p.set_defaults(**load_yaml_config(
            pre.config, p,
            example_keys="'grid', 'nlev', 'dt', 'days', "
                         "'vertical_mixing_scheme', 'kpp_ri_crit'"))
    args = p.parse_args(argv)
    if getattr(args, "require_config", False):
        from legoesm.driver.run_config_yaml import require_config
        require_config(args.config, driver="run_omip")
    # Resolve the additive-mixing riders' implicit-solve requirement HERE, at
    # parse time, so every downstream path sees it.
    #
    # Both riders contribute ONLY through the implicit solve and the
    # LatLonCGridOceanModel constructor RAISES if they are enabled without it.
    # _apply_drag_iwm_overrides forces it too, but that hook runs AFTER the
    # bathymetry path has already built its config+model from `args`, so
    # `--ddm --bathymetry ...` died in the constructor with an error the user
    # could not connect to the flag they passed (codex). Forcing it at the
    # source fixes the flat, bathymetry and tripole paths in one place; the
    # later force becomes a harmless backstop.
    if getattr(args, "ddm", False) or getattr(args, "iwm", False):
        if not getattr(args, "implicit_vertical_mixing", False):
            args.implicit_vertical_mixing = True
            _who = "zdfddm" if getattr(args, "ddm", False) else "zdfiwm"
            print(f"[setup] {_who}: --implicit-vertical-mixing forced ON "
                  "(the additive avt/avs/avm only enter the backward-Euler "
                  "solve; the model refuses the explicit path)")
    return args


# ===========================================================================
# Resolution parsing
# ===========================================================================

def _parse_resolution(grid_type: str, resolution: str) -> dict:
    if grid_type == "cubed_sphere":
        return {"n": int(resolution.lstrip("Cc"))}
    elif grid_type == "latlon":
        parts = resolution.split("x")
        return {"n_lat": int(parts[0]), "n_lon": int(parts[1])}
    elif grid_type == "mpas":
        return {"level": int(resolution.replace("ico", ""))}
    elif grid_type == "spectral":
        return {"truncation": int(resolution.lstrip("Tt"))}
    elif grid_type == "tripole":
        # Resolution maps to a mesh-file + its T-fold index convention. eORCA1
        # (1 deg, 332x362, halo-inclusive -> n_lon-1-i) and eORCA025 (1/4 deg,
        # 1207x1442, de-haloed -> (n_lon-i)%n_lon) -- both NEMO tripole
        # mesh_mask files, read identically by create_tripole_grid
        # (glamt/e1t.../tmask + fold). Passing the validated convention
        # explicitly avoids relying on _detect_fold's auto tie-break.
        meshes = {
            "eorca1": ("data/grids/eORCA1.2_mesh_mask.nc", "n_lon-1-i"),
            "eorca025": ("data/grids/eORCA025_mesh_mask.nc", "(n_lon-i)%n_lon"),
            # eORCA05 (1/2 deg) is 2x-decimated from eORCA025 (de-haloed -> same
            # (n_lon-i)%n_lon fold convention, fold-verified at build time).
            "eorca05": ("data/grids/eORCA05_mesh_mask.nc", "(n_lon-i)%n_lon"),
        }
        key = resolution.lower()
        if key not in meshes:
            raise ValueError(
                f"Unknown tripole resolution {resolution!r}. "
                f"Available: {sorted(meshes)}."
            )
        mesh_path, fold_convention = meshes[key]
        return {"mesh_path": mesh_path, "fold_convention": fold_convention}
    raise ValueError(f"Unknown grid type: {grid_type}")


# ===========================================================================
# Physics config presets
# ===========================================================================

def _build_physics_config(
    preset: str,
    water_type: str,
    vertical_mixing: VerticalMixingConfig | None = None,
):
    """Build OceanPhysicsConfig from a preset name."""
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.shortwave_penetration import ShortwavePenetrationConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    if preset == "none":
        return None

    if preset == "minimal":
        return OceanPhysicsConfig(
            vertical_mixing=(
                vertical_mixing or VerticalMixingConfig(scheme="constant")
            ),
            lateral_mixing=LateralMixingConfig(scheme="harmonic"),
            surface_forcing=SurfaceForcingConfig(scheme="restoring"),
            bottom_drag=BottomDragConfig(scheme="linear"),
            convection=OceanConvectionConfig(scheme="none"),
            shortwave_penetration=ShortwavePenetrationConfig(water_type=water_type),
        )

    # "full" preset
    return OceanPhysicsConfig(
        vertical_mixing=vertical_mixing or VerticalMixingConfig(scheme="kpp"),
        lateral_mixing=LateralMixingConfig(scheme="gm_redi"),
        surface_forcing=SurfaceForcingConfig(scheme="restoring"),
        bottom_drag=BottomDragConfig(scheme="quadratic"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
        shortwave_penetration=ShortwavePenetrationConfig(water_type=water_type),
    )


# ===========================================================================
# Grid + model creation
# ===========================================================================

def _create_setup(grid_type: str, resolution: str, nlev: int, H_max: float,
                  physics_preset: str, water_type: str,
                  use_bathymetry: bool = False,
                  A_h_override: float = None,
                  B_h_override: float = None,
                  K_h_override: float = None,
                  A_h_eq_boost: float = 1.0,
                  A_h_eq_sigma_deg: float = 5.0,
                  C_smag: float = None,
                  C_smag_lap: float = 0.15,
                  A_h_floor: float = 2000.0,
                  C_leith: float = None,
                  pgf_scheme: str = None,
                  slope_foot_alpha: float = 0.0,
                  no_lat_scaling: bool = False,
                  no_gm_redi: bool = False,
                  gm_redi: GMRediConfig | None = None,
                  implicit_vertical_mixing: bool = False,
                  vertical_mixing: VerticalMixingConfig | None = None,
                  forcing_mode: str = "restoring",
                  use_conservation_fixer: bool = True,
                  dz_ref_override=None):
    """Create grid, z_coord, config, model for any grid type.

    All grids use the SAME config-based diffusion (A_h, K_h, A_v, K_v)
    via ``physics=None`` (built-in tendencies) so that the 4 grids are
    physically equivalent.  The ``physics_preset`` only affects which
    OceanPhysicsConfig modules are enabled on cubed-sphere grids where
    the modular pipeline is supported.

    ``dz_ref_override`` (1-D thicknesses [m], e.g. NEMO ``e3t_1d``) replaces the
    stretched z* profile with an EXACT external vertical grid -- its length must
    equal ``nlev`` (the IC / rest-state shapes are built at ``nlev``).

    Returns (grid, z_coord, config, model, coord_kind).
    """
    from legoesm.ocean.vertical import create_ocean_z_star
    if dz_ref_override is not None:
        from legoesm.ocean.vertical import create_z_star_from_thicknesses
        if len(dz_ref_override) != nlev:
            raise ValueError(
                f"dz_ref_override has {len(dz_ref_override)} levels but nlev="
                f"{nlev}; pass --nlev {len(dz_ref_override)} to match.")
        z_coord = create_z_star_from_thicknesses(dz_ref_override)
    elif use_bathymetry:
        # Partial cells with ETOPO: use the same vertical stretching
        # as the global-overturning production scripts (dz_surface=20,
        # dz_deep=500) to avoid degenerate thin layers.
        z_coord = create_ocean_z_star(
            n_levels=nlev, H_max=H_max, dz_surface=20.0, dz_deep=500.0,
        )
    else:
        z_coord = create_ocean_z_star(n_levels=nlev, H_max=H_max)
    params = _parse_resolution(grid_type, resolution)
    vertical_mixing = vertical_mixing or VerticalMixingConfig(scheme="kpp")

    # Mixing coefficients tuned per grid for equivalent effective diffusion
    # at ~5° resolution.  FV grids (cubed-sphere, latlon, MPAS) need higher
    # explicit K_h because the discrete Laplacian has truncation error;
    # spectral grids are spectrally accurate and rely on hyperdiffusion.
    A_v = 1.0e-3  # vertical viscosity [m²/s] (all grids)
    K_v = 1.0e-4  # vertical tracer diffusivity [m²/s] (all grids)
    if grid_type == "spectral":
        A_h = 1.0e4   # spectral Laplacian is exact: less needed
        K_h = 1.0e3
    elif grid_type == "mpas":
        A_h = 1.0e4   # MPAS ico3 is very coarse (~900 km); lower K_h stable
        K_h = 1.0e3
    else:
        A_h = 1.0e5   # cubed-sphere/latlon need more dissipation at ~5°
        K_h = 1.0e5

    if grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        from legoesm.ocean.state import OceanConfig
        # Register the FV3 shallow-water barotropic core provider (fv3sw/fv3edge)
        # so the cube ocean can use the FV3-faithful C-D barotropic solver below
        # instead of the forbidden a_grid solver (never-A-grid directive).
        import legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid  # noqa: F401

        grid = create_cubed_sphere(params["n"])
        # Cubed-sphere OMIP-stability tuning.
        #
        # The cubed-sphere ``OceanModel`` exhibits a slow exponential
        # instability when the horizontal density field develops
        # gradients of any magnitude — the artifact lives at face
        # boundaries and shows up as accelerating velocity / SSH growth
        # at the cells just inside the cube edges (the diagnostic loop
        # locates the first NaN at face-edge cells).  With the
        # latlon-equivalent defaults (A_h=1e5, K_h=1e5,
        # n_barotropic_substeps=30, dt=300s) the rest-state + WOA
        # restoring smoke test reaches NaN at ≈ 2.2 days.  The values
        # below extend the failure to ≈ 4-5 days under the 30-day
        # quick CLI defaults — enough to exercise the cubed-sphere
        # dycore but NOT enough for the 30-day quick matrix run to
        # complete.  ``--grid all`` therefore excludes cubed_sphere
        # in ``main`` so CI smoke tests are not permanently red; users
        # who explicitly opt into ``--grid cubed_sphere`` get the
        # warning printed at startup.  The structural fix (SMC03-style
        # density-Jacobian PGF + duogrid halo on T, S) is tracked in
        # docs/ocean/experiments/cubed_sphere_pgf_stability.md.
        A_h_cs = max(A_h, 5.0e5)
        K_h_cs = max(K_h, 5.0e6)
        if A_h_cs > A_h:
            print(
                f"  WARNING: cubed_sphere A_h raised from {A_h:g} to "
                f"{A_h_cs:g} for face-edge stability"
            )
        if K_h_cs > K_h:
            print(
                f"  WARNING: cubed_sphere K_h raised from {K_h:g} to "
                f"{K_h_cs:g} for face-edge stability"
            )
        config = OceanConfig(
            A_h=A_h_cs, K_h=K_h_cs, A_v=A_v, K_v=K_v,
            n_barotropic_substeps=60,
            barotropic_diffusion_alpha=0.3,
            use_conservation_fixer=use_conservation_fixer,
            physics=None,
            # FV3-faithful C-D barotropic (vector-invariant absolute-vorticity
            # flux + RK3 + div-damp/hyperdiff); replaces the a_grid solver whose
            # computational pressure mode grows a ~40% non-zonal eta artifact
            # (a_grid forbidden by the never-A-grid / FV3-faithfulness directive).
            barotropic_staggering="fv3sw",
        )
        # FV3 C-D grid baroclinic backend (the only cube ocean backend; the
        # deprecated FC-Gram A-grid spectral backend was removed).  The
        # cd-grid A-L corner stencil's face-edge halo amplification under
        # horizontal density gradients is the documented cube cold-start gate;
        # the structural fix (partial cells + SMC03 density-Jacobian PGF on the
        # C-D grid) is tracked in docs/dev-notes/ocean_faithfulness_nemo.md.
        model = OceanModel(grid, z_coord, config)
        return grid, z_coord, config, model, "cube"

    elif grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
        from legoesm.ocean.state import LatLonCGridOceanConfig

        grid = create_latlon_grid(params["n_lat"], params["n_lon"])
        if use_bathymetry:
            # Production config for real ETOPO bathymetry, matching the
            # global-overturning realistic-geometry scripts that run
            # stable 50+ year integrations.  Key ingredients:
            #   - implicit CN barotropic solver (no checkerboard mode)
            #   - SMC03 density-Jacobian PGF (accurate on partial cells)
            #   - biharmonic viscosity (damps topographic comp. modes)
            #   - linear bottom drag with BBL
            #   - convective adjustment (enhanced diffusion)
            #   - GM/Redi isopycnal mixing (Visbeck adaptive)
            #   - K_h=0 (GM/Redi replaces horizontal tracer diffusion)
            from legoesm.ocean.physics.combined import OceanPhysicsConfig
            from legoesm.ocean.physics.convection.config import (
                OceanConvectionConfig, EnhancedDiffusionConfig,
            )
            from legoesm.ocean.physics.lateral_mixing.config import (
                LateralMixingConfig,
            )
            bathy_physics = OceanPhysicsConfig(
                vertical_mixing=vertical_mixing,
                convection=OceanConvectionConfig(
                    scheme="enhanced_diffusion",
                    enhanced_diffusion=EnhancedDiffusionConfig(
                        K_conv=1.0, K_bg=1e-5,
                    ),
                ),
                # Disable the physics pipeline's lateral mixing — the
                # C-grid model applies its own A_h/B_h/K_h viscosity
                # and GM/Redi is wired via the gm_redi config field.
                lateral_mixing=LateralMixingConfig(scheme="none"),
                # Disable shortwave penetration — q_net already includes
                # SW, so the physics SW module would double-count.
                shortwave_penetration=None,
            )
            # Hoisted to _DEFAULT_BATHY_GM_REDI (module level) so the --params
            # calibration layer can override its tunables via OMIPRunConfig
            # (#691/#724); values unchanged.
            bathy_gm_redi = (
                gm_redi if gm_redi is not None else _DEFAULT_BATHY_GM_REDI
            )
            _A_h = A_h_override if A_h_override is not None else 2.0e5
            _B_h = B_h_override if B_h_override is not None else 5.0e9
            _K_h = K_h_override if K_h_override is not None else 1e3
            _C_smag = C_smag if C_smag is not None else 0.0
            _C_leith = C_leith if C_leith is not None else 0.0
            config = LatLonCGridOceanConfig.from_flat(
                A_h=_A_h, A_h_lat_scaling=(not no_lat_scaling),
                A_h_floor=A_h_floor,
                A_h_eq_boost=A_h_eq_boost,
                A_h_eq_sigma_deg=A_h_eq_sigma_deg,
                K_h=_K_h, A_v=A_v, K_v=K_v,
                B_h=_B_h,
                # OMIP global freshwater correction: conserve global salt under
                # an unbalanced P-E+R (matches the MPAS config). Volume is already
                # conserved via fix_eta_drift; this adds the salt normalization.
                normalize_freshwater=True,
                C_smag=_C_smag,
                C_smag_lap=C_smag_lap,
                C_leith=_C_leith,
                C_leith_modified=(_C_leith > 0),
                slope_foot_alpha=slope_foot_alpha,
                # A2: biharmonic hyperviscosity on the DEPTH-MEAN
                # (U_bar, V_bar) only.  Surgically damps the barotropic
                # standing mode at deep cells next to steep slopes
                # (Rhines 1969 bottom-trapped wave with f≈0) without
                # touching baroclinic geostrophy.  HIM/MOM6
                # BIHARMONIC_BAROTROPIC analog at 1° global resolution.
                # Scaled per Griffies-Hallberg 2000: ν₄ ≈ Δx³·U/8.
                # At 1° (Δx≈111 km, U≈1 m/s) that's 1.7e14 m⁴/s.
                B_h_barotropic=1.0e14,
                bottom_drag_r=2.5e-3,
                bottom_drag_bbl_thickness=100.0,
                # MOM6 OM4 DRAG_BG_VEL — quadratic-with-floor drag.
                # At standing-mode amplitudes (~0.05 m/s) this gives ~3×
                # more drag than pure linear, which is the cheapest
                # production fix for the deep-cell barotropic mode at
                # steep slopes.  Recovers linear drag (bit-exact) at
                # |u|→0; scales as Cd·|u| for |u|≫u_bg.
                bottom_drag_bg_velocity=0.1,
                n_barotropic_substeps=30,
                use_conservation_fixer=use_conservation_fixer,
                physics=bathy_physics,
                gm_redi=None if no_gm_redi else bathy_gm_redi,
                barotropic_solver="implicit_cn",
                pgf_scheme=pgf_scheme if pgf_scheme is not None else "smc03",
                # Fourier polar filter ON by default for the GLOBAL regular
                # lat-lon bathy path.  A global lat-lon ocean has converging
                # meridians: dx = R*dlon*cos(lat) -> 0 at the N-pole, so the
                # explicit advection/metric terms violate CFL poleward and the
                # WOA cold-start blows up (~day 0.25) regardless of the time
                # integrator (implicit_cn barotropic / implicit vmix do NOT
                # cure it -- see PolarFilterConfig docstring, #939).  Force it
                # on structurally, mirroring implicit_vertical_mixing=True
                # above, so a recipe that omits --polar-filter can't silently
                # reintroduce the pole blowup.  60.0 is the cutoff LATITUDE
                # [deg] (a filter config value, not a physical constant) and
                # matches the documented stable latlon config.  Safe for the
                # tripole: it is built via the separate _create_setup("tripole")
                # branch (dlon>0), so this default never reaches a dlon==0 grid
                # (which _apply_polar_filter rejects).
                use_polar_filter=True,
                polar_filter_cutoff_lat_deg=60.0,
                # MOM6 MAXVEL: clip barotropic velocities to prevent
                # blowup from WBC intensification at coarse resolution.
                # MOM6 default is 6.0 m/s; we use 3.0 since realistic
                # currents at 1° shouldn't exceed ~2 m/s.
                maxvel_barotropic=0.0,  # disabled — let physics handle it
                implicit_vertical_mixing=implicit_vertical_mixing,
                # NOTE: ke_gradient_scheme left at the "centered" config
                # default.  The Hollingsworth-Kållberg KE gradient was
                # A/B-tested on the realistic WOA cold-start (job 8106193)
                # and made NO difference — both centered and hollingsworth
                # go non-finite by day 0.5 on latlon-bathy AND tripole.  So
                # the KE-gradient scheme is NOT the OMIP cold-start blocker.
                # A/B it via ``run_omip_core2.py --ke-gradient-scheme``
                # (that flag is NOT plumbed into run_omip.py); do not change
                # this default without a case where it demonstrably helps +
                # an ocean test-matrix regression run.
            )
        else:
            config = LatLonCGridOceanConfig.from_flat(
                A_h=A_h, K_h=K_h, A_v=A_v, K_v=K_v,
                n_barotropic_substeps=30,
                use_conservation_fixer=use_conservation_fixer,
                physics=None,
                implicit_vertical_mixing=implicit_vertical_mixing,
            )
        model = LatLonCGridOceanModel(grid, z_coord, config)
        return grid, z_coord, config, model, "latlon"

    elif grid_type == "mpas":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.fidelity.nemo_match_recipe import (
            nemo_match_mpas_model_config,
        )
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.surface_forcing.config import (
            SurfaceForcingConfig, PrescribedForcingConfig, RestoringConfig,
        )
        from legoesm.ocean.physics.convection.config import (
            OceanConvectionConfig, EnhancedDiffusionConfig,
        )
        from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
        from legoesm.ocean.physics.lateral_mixing.config import (
            LateralMixingConfig,
        )

        mesh = create_voronoi_mesh(params["level"])

        # For JRA55 forcing mode, use scheme="none" so that external
        # tau/q_net from the bulk-flux solver are applied via the
        # surface_forcing argument to model.step().  For restoring mode,
        # use the same "combined" config as the comparison scripts.
        # forcing_mode is passed from run_omip_single() via the parameter.
        if forcing_mode == "jra55_do_tropical":
            sf_config = SurfaceForcingConfig(scheme="none")
        else:
            sf_config = SurfaceForcingConfig(
                scheme="combined",
                prescribed=PrescribedForcingConfig(
                    wind_profile="global_wind", tau_max=0.1,
                    tropical_wind_scale=0.5,
                    tropical_wind_lat_deg=15.0,
                ),
                restoring=RestoringConfig(
                    tau_T=2592000.0, tau_S=2592000.0,
                    T_star_eq=25.0, T_star_pole=0.0,
                    S_star=35.0, T_profile="cosine",
                ),
            )

        # SETUP (run-dependent): surface_forcing depends on forcing_mode and the
        # per-run vertical_mixing is passed in; the rest is the proven dycore.
        physics = OceanPhysicsConfig(
            surface_forcing=sf_config,
            vertical_mixing=vertical_mixing,
            lateral_mixing=LateralMixingConfig(scheme="none"),
            bottom_drag=BottomDragConfig(scheme="none"),
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0),
            ),
            shortwave_penetration=None,
        )

        # Build the PROVEN OMIP NEMO-match MPAS dycore from the shared factory
        # (single source of truth, locked to the catalog recipe
        # ``omip_nemo_match_mpas_v1`` by tests/ocean/unit/test_recipes.py), then
        # overlay only the run-dependent SETUP physics above.
        config = nemo_match_mpas_model_config(physics=physics)
        # Enforce global surface-freshwater balance, exactly as the lat-lon/tripole
        # config does (LatLonCGridOceanConfig.from_flat(normalize_freshwater=True) above).
        # The CORE-II P-E+R integral is a net ~+0.65 Sv freshwater input (a true
        # forcing imbalance, identical on every grid); without this the MPAS ocean
        # accumulates it as a ~-0.5 PSU global-mean fresh drift in 90 days, while
        # the tripole/lat-lon path (which sets the flag) stays balanced.  The MPAS
        # step already reads config.normalize_freshwater (ocean_pe_mpas) — the only
        # gap was the flag defaulting False on MPASOceanConfig.
        config = config._replace(normalize_freshwater=True)
        model = MPASOceanModel(mesh, z_coord, config)
        return mesh, z_coord, config, model, "mpas"

    elif grid_type == "spectral":
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.ocean.dynamics.spectral_ocean_pe import SpectralOceanModel
        from legoesm.ocean.state import SpectralOceanConfig

        grid = create_gaussian_grid(params["truncation"])
        # Keep eta_hyperdiff for barotropic stability (required by unsplit
        # SSP-RK3), but reduce 3D hyperdiffusion to be more consistent
        # with the explicit A_h/K_h on the other grids.
        config = SpectralOceanConfig(
            A_h=A_h, K_h=K_h, A_v=A_v, K_v=K_v,
            # Keep default hyperdiffusion coefficients — they are tuned
            # for stability of the unsplit SSP-RK3 spectral solver.
        )
        model = SpectralOceanModel(grid, z_coord, config)
        return grid, z_coord, config, model, "gaussian"

    elif grid_type == "tripole":
        # eORCA1 tripolar via LatLonCGridOceanModel.  Mirrors the
        # MPAS JRA55 path: SurfaceForcingConfig(scheme="none") when
        # forcing_mode=="jra55_do_tropical" so the dynamics-core
        # external-tau block (ocean_pe_latlon_cgrid.py:1838+) is the
        # sole consumer of tau / q_net / sw_down from
        # OceanSurfaceForcing.  Bathymetry + dissipation defaults are
        # lifted from run_tripole_20yr.py (validated by the 20-yr
        # idealized production run).
        from legoesm.grids.tripole import create_tripole_grid
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.fidelity.nemo_match_recipe import (
            nemo_match_tripole_model_config,
        )
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.surface_forcing.config import (
            SurfaceForcingConfig, RestoringConfig,
        )
        from legoesm.ocean.physics.convection.config import (
            OceanConvectionConfig, EnhancedDiffusionConfig,
        )
        from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
        from legoesm.ocean.physics.lateral_mixing.config import (
            LateralMixingConfig,
        )

        geom = create_tripole_grid(
            params["mesh_path"],
            fold_convention=params.get("fold_convention", "auto"))

        if forcing_mode == "jra55_do_tropical":
            sf_config = SurfaceForcingConfig(scheme="none")
        else:
            sf_config = SurfaceForcingConfig(
                scheme="restoring",
                restoring=RestoringConfig(
                    tau_T=2592000.0, tau_S=2592000.0,
                    T_star_eq=25.0, T_star_pole=0.0,
                    S_star=35.0, T_profile="cosine",
                ),
            )

        # SETUP (run-dependent): surface_forcing depends on forcing_mode and the
        # per-run vertical_mixing is passed in; the rest is the proven dycore.
        physics = OceanPhysicsConfig(
            surface_forcing=sf_config,
            vertical_mixing=vertical_mixing,
            lateral_mixing=LateralMixingConfig(scheme="none"),
            bottom_drag=BottomDragConfig(scheme="none"),
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0),
            ),
            shortwave_penetration=None,
        )

        # Build the PROVEN OMIP NEMO-match tripole eORCA025 dycore from the
        # shared factory (single source of truth, locked to the catalog recipe
        # ``omip_nemo_match_tripole_v1`` by tests/ocean/unit/test_recipes.py),
        # then overlay only the run-dependent SETUP physics above.  NOTE:
        # ke_gradient_scheme is intentionally left at the "centered" config
        # default for the TRIPOLE — the Hollingsworth KE stencil
        # (ocean_pe_latlon_cgrid.py) widens to j±1 with edge-replication wall
        # halos and does NOT yet use the tripole north-fold permutation/sign, so
        # defaulting it on would compute KE gradients across the wrong topology
        # at the bipolar cap.  A/B-test it explicitly via ``run_omip_core2.py
        # --ke-gradient-scheme hollingsworth`` (job 8106193 showed it does not
        # fix the equatorial cold-start blowup anyway); a fold-aware KE halo +
        # regression test is the prerequisite to ever making it the default.
        config = nemo_match_tripole_model_config(physics=physics)
        model = LatLonCGridOceanModel(geom, z_coord, config)
        return geom, z_coord, config, model, "tripole"

    raise ValueError(f"Unknown grid type: {grid_type}")


# ===========================================================================
# State initialization with WOA T/S
# ===========================================================================

def _init_rest_state(grid_type, grid, z_coord, H_max,
                     H_bathy=None, land_mask=None,
                     bathy_cfg=None,
                     use_etopo_postinit=False):
    """Create rest-state initial condition (zero velocity, exponential T, uniform S).

    Uses each grid's standard rest_state function, which provides a
    horizontally uniform stratified profile.  This avoids creating strong
    pressure gradients from horizontal T/S contrasts.

    Parameters
    ----------
    H_bathy, land_mask : array or None
        When provided, override the default flat-bottom / idealized
        bathymetry with realistic topography (e.g. from ETOPO).
    """
    # land_lat_threshold=80 → land at high latitudes (standard for ocean
    # test cases).  The spectral model requires land boundaries to constrain
    # the barotropic mode (eta_hyperdiff is tuned for this case).
    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        return rest_state_ocean(grid, z_coord, H_max=H_max)
    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        return rest_state_latlon_cgrid_ocean(
            grid, z_coord, H_max=H_max,
            H_bathy_override=H_bathy,
            land_mask_override=land_mask,
        )
    elif grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        # When ETOPO bathymetry will be applied post-init, use
        # land_lat_threshold=90 so the initial state has ocean
        # everywhere (including the Arctic). The ETOPO block will
        # set the real land_mask and H_bathy afterwards.
        lat_thresh = 90.0 if use_etopo_postinit else 80.0
        return rest_state_mpas_ocean(
            grid, z_coord, H_max=H_max,
            bathymetry=bathy_cfg,
            land_lat_threshold=lat_thresh,
        )
    elif grid_type == "spectral":
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
        return rest_state_spectral_ocean(grid, z_coord, H_max=H_max)
    elif grid_type == "tripole":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        # Reuse the lat-lon C-grid rest-state (tripole shares the same
        # geometry struct + state layout). H_bathy / land_mask come
        # from ETOPO post-init, same as MPAS.
        return rest_state_latlon_cgrid_ocean(
            grid, z_coord, H_max=H_max,
            H_bathy_override=H_bathy,
            land_mask_override=land_mask,
        )
    raise ValueError(f"Unknown grid type: {grid_type}")


# ===========================================================================
# Surface forcing construction
# ===========================================================================

def _build_surface_forcing(grid_type, grid, sw_down_value):
    """Create OceanSurfaceForcing with constant SW for the given grid."""
    from legoesm.ocean.state import OceanSurfaceForcing

    if grid_type == "cubed_sphere":
        shape = (6, grid.n, grid.n)
    elif grid_type == "latlon":
        shape = (grid.lat.shape[0], grid.lon.shape[0])
    elif grid_type == "mpas":
        shape = (grid.nCells,)
    elif grid_type == "spectral":
        # Spectral model doesn't use surface_forcing pipeline
        return None
    else:
        return None

    sw = jnp.full(shape, sw_down_value)
    return OceanSurfaceForcing(sw_down=sw)


# ===========================================================================
# Grid-agnostic SST/SSS restoring
# ===========================================================================

def _apply_restoring(state, grid_type, grid, T_target, S_target, dt, tau_s,
                     ramp_scale: float = 1.0):
    """Apply SST/SSS restoring toward WOA climatology.

    This is the OMIP-standard Haney (1971) surface flux restoring:
    dT/dt|surface = -(T_surface - T*) / tau, applied to the top layer only.

    Works identically across all grid types by operating on the state
    arrays directly.

    Parameters
    ----------
    state : ocean state (any grid type)
    grid_type : str
    T_target, S_target : jnp.ndarray
        Target surface T/S from WOA, shape matching the surface layer.
    dt : float
        Timestep [s].
    tau_s : float
        Restoring timescale [s].
    ramp_scale : float
        Scale factor in [0, 1] applied to the per-step restoring
        coefficient.  Used for a linear spin-up ramp on cubed-sphere
        OMIP where the rest-state ↔ WOA shock excites the face-edge
        PGF instability.
    """
    # Restoring coefficient: fraction toward target per step (with optional ramp)
    alpha = dt / tau_s * ramp_scale

    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_analysis
        # Apply restoring directly in spectral space to avoid aliasing
        # from repeated synthesis→modify→analysis cycles.
        # T_target and S_target are grid-space 2D fields that were
        # pre-transformed to spectral coefficients and stored alongside
        # the restoring targets (see run_omip_single).
        T_hat = state.T_hat.data
        S_hat = state.S_hat.data
        # T_target / S_target are spectral coefficients of the WOA
        # surface field (pre-computed once).
        T_hat_new = T_hat.at[:, 0].set(
            T_hat[:, 0] - alpha * (T_hat[:, 0] - T_target),
        )
        S_hat_new = S_hat.at[:, 0].set(
            S_hat[:, 0] - alpha * (S_hat[:, 0] - S_target),
        )
        return state._replace(
            T_hat=state.T_hat.replace(data=T_hat_new),
            S_hat=state.S_hat.replace(data=S_hat_new),
        )

    # FV grids: cubed-sphere (6,n,n,nlev), latlon (nlat,nlon,nlev),
    #           MPAS (nCells, nlev)
    T = state.T.data
    S = state.S.data
    mask = state.land_mask.data

    # Restore surface layer only, ocean cells only
    if T.ndim == 4:
        # cubed-sphere: mask shape (6,n,n), target shape (6,n,n)
        mask_sfc = mask
    elif T.ndim == 3:
        # latlon: mask shape (nlat,nlon), target shape (nlat,nlon)
        mask_sfc = mask
    else:
        # MPAS: mask shape (nCells,), target shape (nCells,)
        mask_sfc = mask

    # Canonical Haney surface relaxation kernel (shared with the coupled
    # 3D-ocean spin-up path; no duplicated relaxation math).
    from legoesm.ocean.forcing.surface_relaxation import relax_surface_tracers
    T_top_new, S_top_new = relax_surface_tracers(
        T[..., 0], S[..., 0], T_target, S_target, alpha, alpha, mask_sfc,
    )
    T_new = T.at[..., 0].set(T_top_new)
    S_new = S.at[..., 0].set(S_top_new)

    return state._replace(
        T=state.T.replace(data=T_new),
        S=state.S.replace(data=S_new),
    )


# ===========================================================================
# Tropical OMIP — JRA55-do forcing path (Item 4 of tropical_omip_plan.md)
# ===========================================================================

def _setup_jra55_forcing_state(args, grid, grid_type,
                               z_coord=None, T_woa=None, S_woa=None):
    """Open the JRA55-do cache and pre-compute per-step constants.

    Returns a dict with everything ``_jra55_step`` needs:
    cache path, ref_year, 2-D lat/lon arrays in radians, ``CouplerConfig``
    with the Item-1 LY09 / 0.98 q_sat / z_t/z_q fixes, plus optional
    sponge / SSS-restoring / T_freeze-cap state when ``z_coord`` and
    the WOA targets are provided.

    Validates that the cache target grid matches the model grid;
    mismatched cache must be rebuilt with the right resolution.

    Parameters
    ----------
    z_coord : OceanZStarCoordinate or None
        Vertical coordinate of the model. Required for SSS restoring
        (uses the top-layer thickness for the piston-velocity step).
    T_woa, S_woa : ndarray or None, shape (n_lat, n_lon, nlev)
        WOA monthly climatology targets. Required for the sponge T,S
        references and the SSS-restoring target. If either is None,
        the corresponding feature is disabled regardless of the
        ``--jra55-no-...`` flags.
    """
    _supported_jra55_grids = ("latlon", "mpas", "tripole")
    if grid_type not in _supported_jra55_grids:
        raise ValueError(
            "--forcing-mode jra55_do_tropical currently supports only "
            f"--grid {_supported_jra55_grids} (got {grid_type!r})."
        )
    if args.jra55_cache is None:
        raise ValueError(
            "--forcing-mode jra55_do_tropical requires --jra55-cache PATH."
        )
    cache_path = Path(args.jra55_cache)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"JRA55-do cache not found at {cache_path}. Build one with "
            "scripts/data/prepare_omip_forcing.py."
        )

    import xarray as xr
    ds = xr.open_zarr(str(cache_path), decode_times=False)
    cache_n_lat = int(ds.sizes["lat"])
    cache_n_lon = int(ds.sizes["lon"])

    # For MPAS / tripole we need regrid weights; for lat-lon we validate
    # grid match (cache is pre-built at lat-lon resolution).
    regrid_weights = None
    if grid_type == "mpas":
        from legoesm.grids.regridding import compute_latlon_to_voronoi_weights
        src_lat_rad = np.deg2rad(np.asarray(ds["lat"]))
        src_lon_rad = np.deg2rad(np.asarray(ds["lon"]))
        tgt_lat_rad = np.asarray(grid.latCell)
        tgt_lon_rad = np.asarray(grid.lonCell)
        regrid_weights = compute_latlon_to_voronoi_weights(
            src_lat_rad, src_lon_rad, tgt_lat_rad, tgt_lon_rad,
        )
        print(f"  JRA55 regrid: {cache_n_lat}×{cache_n_lon} lat-lon → "
              f"{grid.nCells} MPAS cells (k=4 IDW)")
        # lat/lon for zenith angle (1-D, radians, on MPAS cells)
        lat_2d = jnp.asarray(tgt_lat_rad)
        lon_2d = jnp.asarray(tgt_lon_rad)
    elif grid_type == "tripole":
        # Same KDTree IDW path as MPAS, but the target is a 2-D
        # curvilinear array (lat_T, lon_T).  Flatten for the regrid
        # weight builder, then reshape the output in _jra55_step (the
        # latlon-cgrid model consumes 2-D surface fields).
        from legoesm.grids.regridding import compute_latlon_to_voronoi_weights
        src_lat_rad = np.deg2rad(np.asarray(ds["lat"]))
        src_lon_rad = np.deg2rad(np.asarray(ds["lon"]))
        tgt_lat_rad_2d = np.asarray(grid.lat_T)
        tgt_lon_rad_2d = np.asarray(grid.lon_T)
        n_lat = int(tgt_lat_rad_2d.shape[0])
        n_lon = int(tgt_lat_rad_2d.shape[1])
        regrid_weights = compute_latlon_to_voronoi_weights(
            src_lat_rad, src_lon_rad,
            tgt_lat_rad_2d.ravel(), tgt_lon_rad_2d.ravel(),
        )
        # Mark the regrid output as 2-D so _jra55_step can reshape.
        regrid_weights = regrid_weights._replace(
            target_shape=(n_lat, n_lon),
        )
        print(f"  JRA55 regrid: {cache_n_lat}×{cache_n_lon} lat-lon → "
              f"{n_lat}×{n_lon} tripole T-points (k=4 IDW)")
        lat_2d = jnp.asarray(tgt_lat_rad_2d)
        lon_2d = jnp.asarray(tgt_lon_rad_2d)
    else:
        # lat-lon: validate cache matches model grid
        model_n_lat = int(grid.lat.shape[0])
        model_n_lon = int(grid.lon.shape[0])
        if (cache_n_lat, cache_n_lon) != (model_n_lat, model_n_lon):
            raise ValueError(
                f"JRA55-do cache grid ({cache_n_lat}×{cache_n_lon}) does not "
                f"match model grid ({model_n_lat}×{model_n_lon}). Rebuild the "
                "cache with prepare_omip_forcing.py at the matching resolution."
            )
        # Build 2-D lat/lon (in radians) for cos_zenith / atm_to_surface.
        lat_2d = jnp.asarray(np.deg2rad(np.asarray(grid.lat))[:, None])
        lon_2d = jnp.asarray(np.deg2rad(np.asarray(grid.lon))[None, :])

    # Coupler config: LY09 bulk flux at 10 m winds, 2 m T/q (the JRA55-do
    # convention). The Item 1 fixes (LY09 U^6 term, 0.98 q_sat, separate
    # reference heights) are wired through CouplerConfig.
    from legoesm.coupler.config import CouplerConfig
    coupler_cfg = CouplerConfig(
        bulk_scheme="large_yeager",
        z_ref=10.0,
        z_t_atm=2.0,
        z_q_atm=2.0,
        stability_scheme=args.surface_stability_scheme,
    )

    state: dict = {
        "cache_path": str(cache_path),
        "ref_year": int(ds.attrs.get("ref_year", 1958)),
        "lat_2d": lat_2d,
        "lon_2d": lon_2d,
        "coupler_cfg": coupler_cfg,
        "co2_ppmv": float(args.jra55_co2_ppmv),
        "grid_type": grid_type,
        # Cycle the cache modulo its length when --jra55-cycle is set.
        # This is the Stewart 2020 RYF path: a single-year cache drives
        # a multi-year run by replaying the same 12 months.
        "cycle": bool(getattr(args, "jra55_cycle", False)),
        # Wind-stress spinup ramp: tau *= min(1, t / T_ramp).
        # Stored in seconds for direct use in the step functions.
        "T_ramp_seconds": float(getattr(args, "T_ramp_days", 1.0)) * 86400.0,
    }
    if regrid_weights is not None:
        state["regrid_weights"] = regrid_weights

    # Sponge layer at 60°S/60°N — uses the existing
    # legoesm.ocean.sponge.compute_sponge_gamma_latlon utility. Active
    # only when WOA targets are available (the sponge needs T_ref, S_ref).
    sponge_enabled = (
        not args.jra55_no_sponge
        and T_woa is not None
        and S_woa is not None
    )
    if sponge_enabled:
        if grid_type == "mpas":
            from legoesm.ocean.sponge import compute_sponge_gamma_mpas
            sponge_gamma = compute_sponge_gamma_mpas(
                grid,
                lat_south=args.sponge_lat_min,
                lat_north=args.sponge_lat_max,
                width_deg=args.sponge_width_deg,
                timescale_days=args.sponge_tau_days,
            )
        else:
            from legoesm.ocean.sponge import compute_sponge_gamma_latlon
            sponge_gamma = compute_sponge_gamma_latlon(
                grid,
                lat_south=args.sponge_lat_min,
                lat_north=args.sponge_lat_max,
                width_deg=args.sponge_width_deg,
                timescale_days=args.sponge_tau_days,
            )
        state["sponge_gamma_2d"] = jnp.asarray(sponge_gamma)
        state["sponge_T_ref_3d"] = jnp.asarray(T_woa)
        state["sponge_S_ref_3d"] = jnp.asarray(S_woa)
    state["enable_sponge"] = sponge_enabled

    # SSS restoring — Haney piston-velocity formulation, applied
    # globally (i.e. on every ocean cell) after the dynamics step.
    sss_restoring_enabled = (
        not args.jra55_no_sss_restoring
        and S_woa is not None
        and z_coord is not None
    )
    if sss_restoring_enabled:
        state["sss_target_2d"] = jnp.asarray(S_woa[..., 0])
        state["sss_piston_velocity"] = float(args.sss_piston_velocity)
        state["dz_top"] = float(np.asarray(z_coord.dz_ref)[0])
    state["enable_sss_restoring"] = sss_restoring_enabled

    # T_freeze cap — stand-in for the missing sea-ice model.
    # When a sponge is active, cap only inside the sponge zone.
    # When no sponge, cap globally over all ocean cells.
    freeze_cap_enabled = not args.jra55_no_freeze_cap
    state["enable_freeze_cap"] = freeze_cap_enabled
    from legoesm import constants as _consts
    # State temperature is stored in °C per the lat-lon C-grid ocean
    # convention (init_latlon_cgrid + init_woa both use °C). The cap
    # threshold therefore lives in °C: T_freeze_ocean (271.35 K) minus
    # T_freeze (273.15 K) = -1.8 °C, the seawater freezing point.
    state["T_freeze_ocean_C"] = float(_consts.T_freeze_ocean - _consts.T_freeze)

    # Prognostic slab sea ice (opt-in --jra55-sea-ice): build the slab config +
    # a zero initial ice state on the forcing grid, and force the freeze-cap
    # stand-in OFF so SST is not double-capped (the ice tile now provides the
    # freezing-point physics).  Off (default) ⇒ the freeze cap above is kept and
    # the block scan is byte-identical.
    if getattr(args, "jra55_sea_ice", False):
        from legoesm.core.field import Field
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState
        state["enable_sea_ice"] = True
        state["enable_freeze_cap"] = False
        # Slab ice: dynamics="none", n_cat=1.  stability_scheme only takes
        # effect if bulk_scheme is switched to a MOST-family scheme.
        state["ice_config"] = SeaIceConfig(
            stability_scheme=args.surface_stability_scheme)
        # Ice state lives on the full ocean-surface 2-D grid: lat_2d/lon_2d are
        # broadcast factors ((n_lat,1) x (1,n_lon) for lat-lon; (nCells,) for
        # MPAS), so the surface shape is their broadcast — matching SST
        # (state.T.data[..., 0]) that step_sea_ice broadcasts against.
        _ice_shape = tuple(np.broadcast_shapes(
            np.asarray(state["lat_2d"]).shape,
            np.asarray(state["lon_2d"]).shape,
        ))
        _ice_dims = tuple(f"dim{i}" for i in range(len(_ice_shape)))
        _zeros = jnp.zeros(_ice_shape)
        state["ice_state_init"] = SeaIceState(
            h_ice=Field(_zeros, name="h_ice", dims=_ice_dims, units="m"),
            T_ice=Field(jnp.full(_ice_shape, float(_consts.T_freeze_ocean)),
                        name="T_ice", dims=_ice_dims, units="K"),
            concentration=Field(_zeros, name="concentration",
                                 dims=_ice_dims, units="1"),
        )
    else:
        state["enable_sea_ice"] = False

    return state


def _build_sponge_forcing(jra55_state):
    """Build a ``SpongeForcing`` from the precomputed tropical-OMIP state."""
    from legoesm.ocean.sponge import SpongeForcing
    return SpongeForcing(
        gamma=jra55_state["sponge_gamma_2d"],
        T_ref=jra55_state["sponge_T_ref_3d"],
        S_ref=jra55_state["sponge_S_ref_3d"],
    )


def _apply_sss_restoring(state, jra55_state, dt):
    """Haney SSS restoring with a piston-velocity formulation.

    Applied as a post-step operation on the surface salinity layer
    of the lat-lon C-grid state. Restricted to ocean cells via the
    state's land mask.

    The discretised step is::

        alpha = piston_velocity * dt / dz_top
        S_top_new = S_top - alpha * (S_top - S_target)

    For ``piston_velocity = 5e-7 m/s`` and ``dz_top = 10 m``, the
    e-folding timescale is ~230 days — light enough to let the
    bulk-flux freshwater dominate but stiff enough to damp drift.
    """
    alpha = (
        jra55_state["sss_piston_velocity"]
        * dt
        / max(jra55_state["dz_top"], 1e-6)
    )
    S = state.S.data
    mask = state.land_mask.data  # 2-D
    # Cast target + alpha to S's dtype so scatter doesn't trip the
    # JAX dtype-promotion FutureWarning when the state runs in float32.
    S_target = jnp.asarray(jra55_state["sss_target_2d"], dtype=S.dtype)
    alpha = jnp.asarray(alpha, dtype=S.dtype)
    mask = jnp.asarray(mask, dtype=S.dtype)
    S_top_new = S[..., 0] - alpha * (S[..., 0] - S_target) * mask
    S_new = S.at[..., 0].set(S_top_new)
    return state._replace(S=state.S.replace(data=S_new))


def _apply_freeze_cap(state, jra55_state):
    """Cap surface T from below at ``T_freeze_ocean`` globally.

    Stand-in for the missing sea-ice model: prevent the surface layer
    from cooling below seawater's freezing point (-1.8°C). Without
    this, JRA55-do bulk-flux heat loss over polar regions (where the
    atmosphere is very cold) drives SST below freezing and produces
    unphysical densities.

    When a sponge is active, the cap is scoped to the sponge zone only
    (backward-compatible). When no sponge, the cap applies to all
    ocean cells.
    """
    sponge_gamma = jra55_state.get("sponge_gamma_2d", None)
    if sponge_gamma is not None and jnp.any(sponge_gamma > 0):
        # Sponge active: cap only inside sponge zone
        freeze_mask = sponge_gamma > 0.0
    else:
        # No sponge: cap globally over all ocean cells
        freeze_mask = state.land_mask.data > 0.5
    T = state.T.data
    T_top = T[..., 0]
    T_freeze_C = jnp.asarray(jra55_state["T_freeze_ocean_C"], dtype=T.dtype)
    T_top_capped = jnp.where(
        freeze_mask,
        jnp.maximum(T_top, T_freeze_C),
        T_top,
    )
    T_new = T.at[..., 0].set(T_top_capped)
    return state._replace(T=state.T.replace(data=T_new))


def _jra55_step(state, step_idx, dt, model, jra55_state):
    """One forced-ocean step under JRA55-do bulk-flux forcing.

    Per-step pipeline:
      1. Map step index to fractional simulation day (noleap, since
         ``ref_year-01-01``).
      2. Load JRA55Slice from the cache (linear-in-time interp between
         the bracketing 3-hourly records).
      3. Build AtmToSurface from the slice + model lat/lon + day.
      4. Call ``ocean_tile_response`` with surface SST and zero ocean
         current (forced runs at 1° treat |u_o| << |u_a|).
      5. Compute net heat flux into ocean from SW/LW/SH/LH + LW up.
      6. Build FreshwaterForcing with E from the latent heat flux.
      7. Step the ocean model with ``freshwater=``, ``surface_forcing=``,
         and (if enabled) ``sponge=``.
      8. Apply post-step SSS restoring and the T_freeze cap.

    Steps 1–7 are the Day-2 path; steps 7-sponge and 8 are Day 3.
    The ``enable_sponge``, ``enable_sss_restoring``, ``enable_freeze_cap``
    flags in ``jra55_state`` gate each piece independently.
    """
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.forcing.jra55_do import (
        jra55_to_atm_surface,
        jra55_to_freshwater,
        load_jra55_slice,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    day = float(step_idx) * dt / 86400.0
    slc = load_jra55_slice(
        jra55_state["cache_path"], day,
        ref_year=jra55_state["ref_year"],
        cycle=jra55_state.get("cycle", False),
    )
    # Regrid from lat-lon cache to MPAS cell centres if needed.
    if "regrid_weights" in jra55_state:
        from legoesm.forcing.jra55_do import regrid_jra55_slice
        slc = regrid_jra55_slice(slc, jra55_state["regrid_weights"])
    atm = jra55_to_atm_surface(
        slc,
        jra55_state["lat_2d"],
        jra55_state["lon_2d"],
        day,
        ref_year=jra55_state["ref_year"],
        co2_ppmv=jra55_state["co2_ppmv"],
    )

    # State T is stored in °C; the bulk-flux solver and q_sat lookup
    # need K. This matches the conversion in
    # ocean/physics/surface_forcing/bulk_formulas.py:48.
    from legoesm import constants as _consts
    sst_K = state.T.data[..., 0] + _consts.T_freeze
    # Forced-ocean approximation: at 1° non-eddying resolution
    # |u_ocean| ~ 0.1 m/s << |u_atm| ~ 10 m/s, so we drop the
    # surface-current correction in the bulk-flux solver.
    u_o = jnp.zeros_like(sst_K)
    v_o = jnp.zeros_like(sst_K)

    tile_resp = ocean_tile_response(
        atm, sst_K, u_o, v_o, jra55_state["coupler_cfg"],
    )

    # Wind-stress spinup ramp: scale tau by min(1, t/T_ramp) so that
    # a rest-state ocean accelerates gently under the applied forcing,
    # avoiding violent geostrophic adjustment on the first few steps.
    T_ramp = jra55_state.get("T_ramp_seconds", 86400.0)
    t_sim = float(step_idx) * dt
    ramp = min(1.0, t_sim / T_ramp) if T_ramp > 0 else 1.0

    # Net heat into ocean (positive into ocean):
    #   Q_net = SW_absorbed + LW_down − LW_up − SH − LH
    # where SH and LH are positive upward (out of ocean) per the
    # TileResponse contract.
    sw_net = atm.sw_down * (1.0 - tile_resp.albedo)
    q_net = (
        sw_net
        + atm.lw_down
        - tile_resp.lw_up
        - tile_resp.shflx
        - tile_resp.lhflx
    )

    fw = jra55_to_freshwater(slc, tile_resp.lhflx)
    sf = OceanSurfaceForcing(
        sw_down=atm.sw_down,
        q_net=q_net,
        tau_x=tile_resp.tau_x * ramp,
        tau_y=tile_resp.tau_y * ramp,
        freshwater=None,  # using the structured FreshwaterForcing path
    )

    if jra55_state.get("enable_sponge", False):
        sponge = _build_sponge_forcing(jra55_state)
        # Ramp sponge strength alongside wind stress so the relaxation
        # doesn't create violent pressure gradients from a standing start.
        sponge = sponge._replace(gamma=sponge.gamma * ramp)
    else:
        sponge = None

    state = model.step(
        state, dt,
        freshwater=fw,
        surface_forcing=sf,
        sponge=sponge,
    )

    # Post-step closure-domain operations.
    if jra55_state.get("enable_sss_restoring", False):
        state = _apply_sss_restoring(state, jra55_state, dt)
    if jra55_state.get("enable_freeze_cap", False):
        state = _apply_freeze_cap(state, jra55_state)

    return state


# ===========================================================================
# JRA55-do scan-block path (Item 4 follow-up — multi-core utilisation)
# ===========================================================================
#
# The plain Python time loop calling ``model.step`` once per step is
# correct but single-core-bound: XLA can't see across the loop, so the
# Eigen / BLAS threadpools don't get a useful work item per call at 1°
# resolution. The realistic-geometry GO continuation scripts wrap N
# steps in ``jax.lax.scan`` inside a single ``@jax.jit``; that gives
# XLA one big computation graph and 5–10× speedup at 1° on multi-core
# CPU.
#
# We can't put ``load_jra55_slice`` inside ``lax.scan`` (Zarr I/O is
# not a JAX op). Instead the outer Python layer pre-loads N steps of
# forcing into stacked JAX arrays once, then calls a JIT-compiled
# block function that scans through them.

def _preload_jra55_forcing_block(start_step_idx, n_steps, dt, jra55_state):
    """Pre-load N steps of JRA55-do forcing into stacked JAX arrays.

    Returns a tuple ``(atm_stack, runoff_stack)`` where ``atm_stack``
    is a dict of stacked AtmToSurface fields with shape
    ``(n_steps, n_lat, n_lon)`` and ``runoff_stack`` is the per-step
    friver field with the same shape.

    Pure host-side I/O — runs once per block, then the JIT-compiled
    block_fn consumes the result.
    """
    from legoesm.forcing.jra55_do import (
        jra55_to_atm_surface,
        load_jra55_block,
    )

    cache_path = jra55_state["cache_path"]
    ref_year = jra55_state["ref_year"]
    cycle = jra55_state.get("cycle", False)
    lat_2d = jra55_state["lat_2d"]
    lon_2d = jra55_state["lon_2d"]
    co2_ppmv = jra55_state["co2_ppmv"]

    # Bulk-read all N steps in one Zarr open + contiguous slab read.
    start_day = start_step_idx * dt / 86400.0
    slices = load_jra55_block(
        cache_path, start_day, n_steps, dt,
        ref_year=ref_year, cycle=cycle,
    )

    # Names match AtmToSurface field set; collected per-step then stacked.
    fields = (
        "sw_down", "lw_down", "precip_total", "precip_snow",
        "T_lowest", "q_lowest", "u_lowest", "v_lowest",
        "p_lowest", "p_surface", "rho_lowest", "cos_zenith",
    )
    accum: dict[str, list] = {f: [] for f in fields}
    runoffs: list = []

    for k, slc in enumerate(slices):
        day = (start_step_idx + k) * dt / 86400.0
        atm = jra55_to_atm_surface(
            slc, lat_2d, lon_2d, day,
            ref_year=ref_year, co2_ppmv=co2_ppmv,
        )
        for f in fields:
            accum[f].append(getattr(atm, f))
        runoffs.append(slc.friver)

    atm_stack = {f: jnp.stack(accum[f]) for f in fields}
    runoff_stack = jnp.stack(runoffs)
    return atm_stack, runoff_stack


def _preload_jra55_full_cache(jra55_state):
    """Pre-load and regrid the entire JRA55 cache into RAM.

    For repeat-year forcing (``--jra55-cycle``), the same 2920 records
    are read over and over.  Loading everything once at startup and
    regridding to the target grid eliminates all per-block Zarr I/O.

    Returns ``(all_records, record_days, cache_length_days)`` where:
    - ``all_records``: dict of ``(n_total_records, n_spatial)`` arrays
    - ``record_days``: ``(n_total_records,)`` fractional days
    - ``cache_length_days``: float, total cache duration
    """
    import xarray as xr
    from legoesm.forcing.jra55_do import JRA55_VARIABLES, RECORDS_PER_DAY

    cache_path = jra55_state["cache_path"]
    ds = xr.open_zarr(str(cache_path), decode_times=False)
    n_cache_records = int(ds.attrs["n_records"])
    cache_length_days = n_cache_records / RECORDS_PER_DAY

    t0 = time.time()
    all_records = {}
    # float32 cache: JRA55 fields are atmospheric forcing accurate to
    # ~0.1%, so f32 (~7 sig figs) is well within the precision budget.
    # Halves the staged memory footprint (28 GB -> 14 GB) so the cache
    # fits on a 32 GB V100S alongside a 120k-cell tripole state.
    # The model promotes back to float64 at the per-step consumer.
    for var in JRA55_VARIABLES:
        all_records[var] = jnp.asarray(ds[var].values, dtype=jnp.float32)
    ds.close()

    # Regrid from lat-lon to MPAS cells if needed.
    if "regrid_weights" in jra55_state:
        from legoesm.grids.regridding import regrid_scalar
        rw = jra55_state["regrid_weights"]
        for var in all_records:
            arr = all_records[var]  # (n_records, n_lat, n_lon)
            all_records[var] = jnp.stack([
                regrid_scalar(arr[i], rw)
                for i in range(arr.shape[0])
            ])

    record_days = jnp.asarray(
        np.arange(n_cache_records, dtype=np.float64) / RECORDS_PER_DAY,
    )

    elapsed = time.time() - t0
    nbytes = sum(a.nbytes for a in all_records.values())
    print(f"  Pre-loaded full JRA55 cache: {n_cache_records} records, "
          f"{nbytes / 1e9:.1f} GB, {elapsed:.1f}s")

    return all_records, record_days, cache_length_days


def _jra55_block_record_window(start_step_idx, n_steps, dt,
                               n_cache_records, cycle):
    """Compute the cache-record window bracketing one scan block.

    Shared by ``_slice_preloaded_records`` and
    ``_preload_jra55_raw_records`` so the wrap/clock arithmetic exists
    exactly once.

    Returns ``(indices, record_days, start_day, start_day_forcing)``:

    - ``indices``: list of cache record indices (wrap-aware in cycle
      mode: a block straddling the repeat-year boundary reads the tail
      of the cache followed by the head of the next cycle).
    - ``record_days``: ``(n,)`` fractional days of each selected record
      on the *forcing clock*.  Monotonic: records read from the front of
      the cache after a repeat-year wrap get ``+ cache_length_days`` so
      linear interpolation stays correct across the wrap boundary.
    - ``start_day``: RAW simulation day of the block's first step — the
      solar-zenith / spinup-ramp clock.
    - ``start_day_forcing``: block-start day on the forcing clock —
      equal to ``start_day`` when not cycling, else ``start_day mod
      cache_length_days`` (aligned with ``record_days``).

    Raises ``IndexError`` when ``cycle=False`` and the block needs a
    record past the cache end — same contract/message style as
    ``legoesm.forcing.jra55_do.load_jra55_slice`` (no silent synthetic
    forcing from a clamped stale record).
    """
    from legoesm.forcing.jra55_do import RECORDS_PER_DAY

    cache_length_days = n_cache_records / RECORDS_PER_DAY
    start_day = start_step_idx * dt / 86400.0
    end_day = (start_step_idx + n_steps - 1) * dt / 86400.0

    if cycle:
        start_day_f = start_day % cache_length_days
        # Continuous forcing clock within the block: do NOT re-mod the
        # end — a block straddling the wrap keeps increasing past
        # cache_length_days and reads unwrapped record_days.
        end_day_f = start_day_f + (end_day - start_day)
        if end_day - start_day >= cache_length_days:
            raise ValueError(
                f"forcing block spans {end_day - start_day:.3f} days >= "
                f"the full cache cycle ({cache_length_days:.3f} days); "
                "reduce the block size (diag_every)."
            )
    else:
        start_day_f = start_day
        end_day_f = end_day

    i_first = int(np.floor(start_day_f * RECORDS_PER_DAY))
    i_last = int(np.floor(end_day_f * RECORDS_PER_DAY)) + 1  # upper bracket

    if not cycle:
        # The upper bracket is genuinely needed only when the final step
        # falls strictly inside an inter-record interval (matches
        # _floor_indices_and_alpha's alpha==0 shortcut in jra55_do).
        end_pos = end_day_f * RECORDS_PER_DAY
        i_hi_needed = int(np.floor(end_pos))
        if end_pos > i_hi_needed:
            i_hi_needed += 1
        if i_hi_needed >= n_cache_records:
            raise IndexError(
                f"day={end_day_f} (cache slot {i_hi_needed}) exceeds "
                f"cache length {n_cache_records}"
            )
        i_last = min(i_last, n_cache_records - 1)
        indices = list(range(i_first, i_last + 1))
        wrap_at = None
    elif i_last >= n_cache_records:
        # Repeat-year wrap: tail of the cache + head of the next cycle.
        i_last_wrapped = i_last - n_cache_records
        if i_last_wrapped >= n_cache_records:
            raise ValueError(
                f"forcing block wraps the {cache_length_days:.3f}-day "
                "cache more than once; reduce the block size "
                "(diag_every)."
            )
        indices = (list(range(i_first, n_cache_records))
                   + list(range(0, i_last_wrapped + 1)))
        wrap_at = n_cache_records - i_first
    else:
        indices = list(range(i_first, i_last + 1))
        wrap_at = None

    record_days_np = np.asarray(indices, dtype=np.float64) / RECORDS_PER_DAY
    if wrap_at is not None:
        record_days_np[wrap_at:] += cache_length_days
    record_days = jnp.asarray(record_days_np)
    return indices, record_days, start_day, start_day_f


def _slice_preloaded_records(start_step_idx, n_steps, dt, jra55_state,
                             all_records, all_record_days, cache_length_days):
    """Slice bracketing records from the pre-loaded cache for one block.

    Same interface as ``_preload_jra55_raw_records`` but reads from
    in-memory arrays instead of Zarr.
    """
    cycle = jra55_state.get("cycle", False)
    n_cache_records = int(all_record_days.shape[0])

    indices, record_days, start_day, start_day_f = (
        _jra55_block_record_window(
            start_step_idx, n_steps, dt, n_cache_records, cycle))

    raw_stack = {var: all_records[var][jnp.array(indices)] for var in all_records}
    runoff_stack = raw_stack["friver"]

    record_meta = {
        "record_days": record_days,
        "block_start_day": float(start_day),
        "block_start_day_forcing": float(start_day_f),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "cache_length_days": float(cache_length_days),
        "cycle": cycle,
    }
    return raw_stack, runoff_stack, record_meta


def _preload_jra55_raw_records(start_step_idx, n_steps, dt, jra55_state):
    """Pre-load only the native 3-hourly JRA55 records that bracket a block.

    **GPU-interp path (default, --gpu-interp).**

    The JRA55 cache has 8 records/day (3-hourly).  At dt=300s, each
    simulated day has 288 timesteps.  The old CPU path called
    ``jra55_to_atm_surface`` 288 times per day in Python, spending
    ~1.9s/day on host-side I/O.  This path loads only the ~9 native
    records that bracket the block (~0.2s/day) and defers the linear
    interpolation + solar zenith to the JIT-compiled ``lax.scan`` body
    on GPU.  Result: 38% overall speedup (9.5x I/O reduction).

    Returns ``(raw_stack, runoff_stack, record_meta)`` where:
    - ``raw_stack``: dict of ``(n_records, n_lat, n_lon)`` arrays for
      each JRA55 raw variable (uas, vas, tas, huss, psl, rsds, rlds,
      prra, prsn)
    - ``runoff_stack``: ``(n_records, n_lat, n_lon)`` friver
    - ``record_meta``: dict with ``record_days`` (fractional day of each
      record on the forcing clock, unwrapped across the repeat-year
      boundary), ``block_start_day`` (RAW simulation day — solar-zenith
      clock), ``block_start_day_forcing`` (cycled forcing clock aligned
      with ``record_days``), ``dt``, ``n_steps`` — enough for the scan
      body to compute interpolation weights
    """
    import xarray as xr
    from legoesm.forcing.jra55_do import (
        JRA55_VARIABLES, RECORDS_PER_DAY,
    )

    cache_path = jra55_state["cache_path"]
    ref_year = jra55_state["ref_year"]
    cycle = jra55_state.get("cycle", False)

    ds = xr.open_zarr(str(cache_path), decode_times=False)
    n_cache_records = int(ds.attrs["n_records"])
    cache_length_days = n_cache_records / RECORDS_PER_DAY

    # Find the range of 3-hourly record indices needed (wrap-aware).
    indices, record_days, start_day, start_day_f = (
        _jra55_block_record_window(
            start_step_idx, n_steps, dt, n_cache_records, cycle))

    # Bulk-read each variable
    var_data = {}
    for var in JRA55_VARIABLES:
        slab = ds[var].isel(time=indices).values
        var_data[var] = jnp.asarray(slab, dtype=jnp.float64)

    ds.close()

    raw_stack = {var: var_data[var] for var in JRA55_VARIABLES}

    # Regrid from lat-lon cache to MPAS cell centres if needed.
    # Each variable is (n_records, n_lat, n_lon) → (n_records, nCells).
    if "regrid_weights" in jra55_state:
        from legoesm.grids.regridding import regrid_scalar
        rw = jra55_state["regrid_weights"]
        for var in raw_stack:
            raw_stack[var] = jnp.stack([
                regrid_scalar(raw_stack[var][i], rw)
                for i in range(raw_stack[var].shape[0])
            ])

    runoff_stack = raw_stack["friver"]

    record_meta = {
        "record_days": record_days,          # (n_records,) fractional days
        "block_start_day": float(start_day),
        "block_start_day_forcing": float(start_day_f),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "cache_length_days": float(cache_length_days),
        "cycle": cycle,
    }

    return raw_stack, runoff_stack, record_meta


def _seed_mass_flux_for_scan(model, state):
    """Seed ``store_mass_flux``'s state slots before a ``lax.scan`` (#1442).

    The block scans below carry the ocean state as a ``lax.scan`` CARRY, and
    ``_step_impl`` turns ``mass_flux_u``/``_v``/``_w`` from ``None`` into
    ``Field``s when the flag is on -- a carry-structure mismatch that aborts
    the scan on the first iteration (codex round-6 RED 1).  Seeding here, at
    the scan-driver boundary, is the fix; seeding inside the SPMD step is too
    late because this scan wraps it.

    Grid-agnostic: keyed off ``getattr(model.config, "store_mass_flux", False)``
    so an MPAS / cube model (whose config has no such field, and whose state
    has no such slots) returns unchanged, and so does any lat-lon run with the
    flag off.  Called for its structure, never for its values.
    """
    _mass = getattr(model.config, "store_mass_flux", False)
    _salt = getattr(model.config, "store_salt_flux", False)
    if not (_mass or _salt):
        return state
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        seed_mass_flux_carry,
        seed_salt_flux_carry,
    )
    if _mass:
        state = seed_mass_flux_carry(state, True)
    if _salt:
        state = seed_salt_flux_carry(state, True)
    return state


def _build_jra55_block_fn(model, jra55_state, dt, spmd_step=None):
    """Return a JIT-compiled block function that runs N steps via lax.scan.

    Captures everything that's static across the block (sponge, SSS
    target, freeze-cap mask, coupler config, dt) in the closure so
    the scan body has a clean ``(state, idx) → (state', None)`` signature.
    Re-using the returned function across blocks reuses the JIT cache.
    """
    from legoesm import constants as _const
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.coupler.ocean_forcing import omip_sea_ice_surface_forcing
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    coupler_cfg = jra55_state["coupler_cfg"]
    co2_ppmv = float(jra55_state["co2_ppmv"])
    T_ramp_seconds = float(jra55_state.get("T_ramp_seconds", 86400.0))
    enable_ramp = T_ramp_seconds > 0
    enable_sponge = bool(jra55_state.get("enable_sponge", False))
    enable_sss = bool(jra55_state.get("enable_sss_restoring", False))
    enable_freeze = bool(jra55_state.get("enable_freeze_cap", False))
    # Prognostic slab sea ice (opt-in --jra55-sea-ice): replaces the freeze-cap
    # SST stand-in.  Static gate ⇒ ice-off blocks are bit-identical.  Setup
    # forces enable_freeze_cap=False when ice is on (no double-capping).
    enable_sea_ice = bool(jra55_state.get("enable_sea_ice", False))
    ice_cfg = jra55_state.get("ice_config")

    sponge = _build_sponge_forcing(jra55_state) if enable_sponge else None

    if enable_sss:
        sss_pv = float(jra55_state["sss_piston_velocity"])
        sss_dz = float(jra55_state["dz_top"])
        sss_alpha_static = sss_pv * dt / max(sss_dz, 1e-6)
        sss_target_static = jra55_state["sss_target_2d"]
    else:
        sss_alpha_static = 0.0
        sss_target_static = None

    if enable_freeze:
        sponge_gamma = jra55_state.get("sponge_gamma_2d", None)
        if sponge_gamma is not None and np.any(np.asarray(sponge_gamma) > 0):
            freeze_mask_static = sponge_gamma > 0.0
        else:
            # No sponge: cap globally over all ocean cells.
            # Use the land_mask from the initial state (captured below).
            freeze_mask_static = jra55_state.get("_ocean_mask_2d", None)
        T_freeze_C_static = float(jra55_state["T_freeze_ocean_C"])
    else:
        freeze_mask_static = None
        T_freeze_C_static = -1.8

    # 3D velocity clip — caps ALL velocity components (barotropic +
    # baroclinic) after each step.  The barotropic-only MAXVEL inside
    # the split-explicit solver doesn't prevent baroclinic blowup.
    _maxvel_3d = model.config.barotropic.maxvel_barotropic
    enable_maxvel = _maxvel_3d > 0.0

    # Lat-band SPMD (--enable-latlon-spmd): the scan body's dynamics step
    # runs through the sharded wrapper (same forcing kwargs as _step_impl;
    # the wrapper's cache/arm-restore Python runs ONCE at block trace).
    # Sea ice is refused upstream (the ice tile is not SPMD-audited yet).
    if spmd_step is not None and enable_sea_ice:
        raise ValueError(
            "spmd_step + prognostic sea ice is unsupported "
            "(run_omip_single refuses --jra55-sea-ice with "
            "--enable-latlon-spmd).")
    # aux threading (codex r18 P1): the SPMD step's sharded geometry
    # stacks must cross THIS jit boundary as an ARGUMENT — captured in the
    # closure they become outer-trace constants whose value jax cannot
    # fetch for non-addressable arrays on multicontroller (see
    # make_sharded_ocean_step's aux note).
    if spmd_step is not None:
        def _dyn_step(st, d, aux=None, **kw):
            return spmd_step(st, d, aux=aux, **kw)
    else:
        def _dyn_step(st, d, aux=None, **kw):
            return model._step_impl(st, d, **kw)

    @jax.jit
    def block_fn(state, atm_stack, runoff_stack, block_start_step,
                 ice_state=None, aux=None):
        def step_body(carry, idx):
            if enable_sea_ice:
                state_in, ice_in = carry
            else:
                state_in = carry
            atm = AtmToSurface(
                sw_down=atm_stack["sw_down"][idx],
                lw_down=atm_stack["lw_down"][idx],
                precip_total=atm_stack["precip_total"][idx],
                precip_snow=atm_stack["precip_snow"][idx],
                T_lowest=atm_stack["T_lowest"][idx],
                q_lowest=atm_stack["q_lowest"][idx],
                u_lowest=atm_stack["u_lowest"][idx],
                v_lowest=atm_stack["v_lowest"][idx],
                p_lowest=atm_stack["p_lowest"][idx],
                p_surface=atm_stack["p_surface"][idx],
                rho_lowest=atm_stack["rho_lowest"][idx],
                cos_zenith=atm_stack["cos_zenith"][idx],
                co2_ppmv=jnp.asarray(co2_ppmv, dtype=atm_stack["T_lowest"].dtype),
                has_radiation=jnp.asarray(1.0, dtype=atm_stack["T_lowest"].dtype),
                has_precipitation=jnp.asarray(1.0, dtype=atm_stack["T_lowest"].dtype),
            )

            sst_K = state_in.T.data[..., 0] + _const.T_freeze
            u_o = jnp.zeros_like(sst_K)
            v_o = jnp.zeros_like(sst_K)
            tile = ocean_tile_response(atm, sst_K, u_o, v_o, coupler_cfg)

            # Wind-stress spinup ramp (gated at compile time).
            if enable_ramp:
                abs_step = block_start_step + idx
                t_sim = abs_step.astype(jnp.float64) * dt
                ramp = jnp.minimum(1.0, t_sim / T_ramp_seconds)
                tau_x = tile.tau_x * ramp
                tau_y = tile.tau_y * ramp
            else:
                tau_x = tile.tau_x
                tau_y = tile.tau_y

            sw_net = atm.sw_down * (1.0 - tile.albedo)
            q_net = (sw_net + atm.lw_down
                     - tile.lw_up - tile.shflx - tile.lhflx)

            evap = tile.lhflx / _const.L_v
            fw = FreshwaterForcing(
                precip=atm.precip_total,
                evap=evap,
                runoff=runoff_stack[idx],
                ice_fw=jnp.zeros_like(runoff_stack[idx]),
            )
            sf = OceanSurfaceForcing(
                sw_down=atm.sw_down,
                q_net=q_net,
                tau_x=tau_x,
                tau_y=tau_y,
                freshwater=None,
            )
            # Prognostic slab sea ice: advance the ice tile and partition the
            # surface forcing (open-ocean fluxes x f_ocean=(1-A) + the ice
            # tile's basal heat / melt-freeze freshwater / brine salt / stress).
            # ocean_mask: land cells receive no ice->ocean forcing (mask-aware
            # blend contract; land_mask is scan-carry state, traced-safe).
            if enable_sea_ice:
                new_ice, fw, sf = omip_sea_ice_surface_forcing(
                    ice_state=ice_in, ice_config=ice_cfg, atm=atm,
                    ocean_sst_K=sst_K, open_ocean_sf=sf, open_ocean_fw=fw,
                    dt=dt, grid=None,
                    ocean_mask=state_in.land_mask.data,
                )
            # Ramp sponge strength alongside wind stress.
            if enable_ramp and enable_sponge:
                sponge_step = sponge._replace(gamma=sponge.gamma * ramp)
            else:
                sponge_step = sponge

            new_state = _dyn_step(
                state_in, dt, aux=aux,
                freshwater=fw, surface_forcing=sf, sponge=sponge_step,
            )

            # SSS restoring (gated at compile time via Python `if`).
            if enable_sss:
                S = new_state.S.data
                target = jnp.asarray(sss_target_static, dtype=S.dtype)
                alpha = jnp.asarray(sss_alpha_static, dtype=S.dtype)
                mask = jnp.asarray(new_state.land_mask.data, dtype=S.dtype)
                S_top_new = (
                    S[..., 0] - alpha * (S[..., 0] - target) * mask
                )
                new_state = new_state._replace(
                    S=new_state.S.replace(data=S.at[..., 0].set(S_top_new)),
                )

            # T_freeze cap inside sponge.
            if enable_freeze:
                T = new_state.T.data
                T_freeze_C = jnp.asarray(T_freeze_C_static, dtype=T.dtype)
                T_top = T[..., 0]
                T_top_capped = jnp.where(
                    freeze_mask_static,
                    jnp.maximum(T_top, T_freeze_C),
                    T_top,
                )
                new_state = new_state._replace(
                    T=new_state.T.replace(data=T.at[..., 0].set(T_top_capped)),
                )

            # 3D velocity clip (MOM6 MAXVEL analog for full field).
            if enable_maxvel:
                u_clipped = jnp.clip(new_state.u.data, -_maxvel_3d, _maxvel_3d)
                v_clipped = jnp.clip(new_state.v.data, -_maxvel_3d, _maxvel_3d)
                new_state = new_state._replace(
                    u=new_state.u.replace(data=u_clipped),
                    v=new_state.v.replace(data=v_clipped),
                )

            if enable_sea_ice:
                return (new_state, new_ice), None
            return new_state, None

        n = atm_stack["sw_down"].shape[0]
        state = _seed_mass_flux_for_scan(model, state)
        init = (state, ice_state) if enable_sea_ice else state
        final, _ = jax.lax.scan(
            step_body, init, jnp.arange(n, dtype=jnp.int32),
        )
        return final  # (state, ice_state) when sea-ice on, else state

    return block_fn


def _build_jra55_block_fn_interp(model, jra55_state, dt, spmd_step=None):
    """JIT-compiled block function with GPU-side forcing interpolation.

    Like ``_build_jra55_block_fn``, but instead of receiving pre-
    interpolated per-step forcing, receives the native 3-hourly records
    and computes the linear interpolation + solar zenith inside the
    ``lax.scan`` body on GPU.  This reduces host-side I/O from N calls
    to ``jra55_to_atm_surface`` (N=288 for 1 day) down to ~9 Zarr reads
    per block.
    """
    from legoesm import constants as _const
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.coupler.ocean_forcing import omip_sea_ice_surface_forcing
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.atmosphere.physics.radiation.solar import (
        cos_zenith_angle, solar_declination,
    )
    from legoesm.forcing.jra55_do import RECORDS_PER_DAY

    coupler_cfg = jra55_state["coupler_cfg"]
    co2_ppmv = float(jra55_state["co2_ppmv"])
    T_ramp_seconds = float(jra55_state.get("T_ramp_seconds", 86400.0))
    enable_ramp = T_ramp_seconds > 0
    enable_sponge = bool(jra55_state.get("enable_sponge", False))
    enable_sss = bool(jra55_state.get("enable_sss_restoring", False))
    enable_freeze = bool(jra55_state.get("enable_freeze_cap", False))

    sponge = _build_sponge_forcing(jra55_state) if enable_sponge else None

    if enable_sss:
        sss_pv = float(jra55_state["sss_piston_velocity"])
        sss_dz = float(jra55_state["dz_top"])
        sss_alpha_static = sss_pv * dt / max(sss_dz, 1e-6)
        sss_target_static = jra55_state["sss_target_2d"]
    else:
        sss_alpha_static = 0.0
        sss_target_static = None

    if enable_freeze:
        sponge_gamma = jra55_state.get("sponge_gamma_2d", None)
        if sponge_gamma is not None and np.any(np.asarray(sponge_gamma) > 0):
            freeze_mask_static = sponge_gamma > 0.0
        else:
            freeze_mask_static = jra55_state.get("_ocean_mask_2d", None)
        T_freeze_C_static = float(jra55_state["T_freeze_ocean_C"])
    else:
        freeze_mask_static = None
        T_freeze_C_static = -1.8

    # Prognostic slab sea ice (opt-in) — see _build_jra55_block_fn.
    enable_sea_ice = bool(jra55_state.get("enable_sea_ice", False))
    ice_cfg = jra55_state.get("ice_config")

    _maxvel_3d = model.config.barotropic.maxvel_barotropic
    enable_maxvel = _maxvel_3d > 0.0

    # Lat-band SPMD: see _build_jra55_block_fn.
    if spmd_step is not None and enable_sea_ice:
        raise ValueError(
            "spmd_step + prognostic sea ice is unsupported "
            "(run_omip_single refuses --jra55-sea-ice with "
            "--enable-latlon-spmd).")
    if spmd_step is not None:
        def _dyn_step(st, d, aux=None, **kw):
            return spmd_step(st, d, aux=aux, **kw)
    else:
        def _dyn_step(st, d, aux=None, **kw):
            return model._step_impl(st, d, **kw)

    lat_2d = jra55_state["lat_2d"]
    lon_2d = jra55_state["lon_2d"]
    _rpd = float(RECORDS_PER_DAY)
    # Static (compile-time) repeat-year-forcing flag.  Selects which clock
    # drives the solar-zenith insolation geometry (see the scan body).
    cycle = bool(jra55_state.get("cycle", False))

    def _make_block_fn(n_steps_block):
        """Create a JIT-compiled block function for a fixed block size."""
        @jax.jit
        def block_fn(state, raw_stack, runoff_records, record_days,
                     block_start_day, block_start_day_forcing,
                     ice_state=None, aux=None):
            dt_days = dt / 86400.0

            def step_body(carry, idx):
                if enable_sea_ice:
                    state_in, ice_in = carry
                else:
                    state_in = carry
                # Two clocks (see the insolation + ramp notes below):
                # - ``day``: RAW simulation day (elapsed run time).  Always
                #   drives the spinup ramp; drives the solar-zenith clock only
                #   when NOT cycling (cycle=False ⇒ day_f == day).
                # - ``day_f``: forcing clock — the CYCLED block-start day
                #   (aligned with ``record_days``, which the preloaders
                #   unwrap across the repeat-year cache boundary).  Drives the
                #   JRA55 record interpolation, and — when cycle=True — the
                #   solar-zenith insolation clock, so the prescribed rsds and
                #   the computed zenith stay phase-locked.  Using the raw day
                #   for interpolation broke every cycle after the first:
                #   ``day - record_days[0]`` was off by k*cache_length,
                #   i_lo clipped to the last slice record, and each step
                #   read one stale record.
                day = block_start_day + idx * dt_days
                day_f = block_start_day_forcing + idx * dt_days

                # Find bracketing records: record_days is sorted,
                # find floor position relative to the first record.
                local_pos = day_f * _rpd - record_days[0] * _rpd
                i_lo = jnp.clip(
                    jnp.floor(local_pos).astype(jnp.int32),
                    0, record_days.shape[0] - 2,
                )
                i_hi = i_lo + 1
                day_lo = record_days[i_lo]
                day_hi = record_days[i_hi]
                alpha = jnp.clip(
                    jnp.where(day_hi > day_lo,
                              (day_f - day_lo) / (day_hi - day_lo), 0.0),
                    0.0, 1.0,
                )

                def _interp(arr):
                    return (1.0 - alpha) * arr[i_lo] + alpha * arr[i_hi]

                rsds = _interp(raw_stack["rsds"])
                rlds = _interp(raw_stack["rlds"])
                tas = _interp(raw_stack["tas"])
                huss = _interp(raw_stack["huss"])
                uas = _interp(raw_stack["uas"])
                vas = _interp(raw_stack["vas"])
                psl = _interp(raw_stack["psl"])
                prra = _interp(raw_stack["prra"])
                prsn = _interp(raw_stack["prsn"])
                friver = _interp(runoff_records)

                # Derived: virtual-T density + solar zenith. Canonical coefficient
                # 1/epsilon - 1 (~0.608), not the rounded 0.61 (~0.4% drift) — must
                # match jra55_to_atm_surface / _shared.virtual_temperature.
                T_v = tas * (1.0 + (1.0 / _const.epsilon - 1.0) * huss)
                rho_a = psl / (_const.R_d * T_v)
                # Insolation clock (day-of-year + diurnal hour), which sets the
                # solar zenith and hence zenith-dependent surface albedo:
                #   - cycle=True (repeat-year forcing): use the CYCLED forcing
                #     clock ``day_f`` so the solar geometry stays phase-locked
                #     to the repeated rsds/rlds records.  Using the RAW ``day``
                #     drifts the seasonal doy (and, for a non-integer cache
                #     length, the diurnal hour) whenever the cache length is
                #     not a whole multiple of 365 days (e.g. a 366-day
                #     leap-year RYF cache), biasing the surface albedo.
                #   - cycle=False: ``day_f == day`` (no wrap), so this is
                #     byte-identical to the raw-day clock — the common
                #     non-cycled path is unchanged.
                # The SPINUP RAMP (below) intentionally stays on the RAW
                # ``day``: it is a function of elapsed run time, not forcing
                # time.  ``cycle`` is a static Python bool (compile-time
                # feature gate), so this branch is resolved at trace time.
                day_insol = day_f if cycle else day
                doy = jnp.mod(day_insol, 365.0) + 1.0
                hour = jnp.mod(day_insol, 1.0) * 24.0
                cos_z = cos_zenith_angle(lat_2d, lon_2d, doy, hour)

                atm = AtmToSurface(
                    sw_down=rsds, lw_down=rlds,
                    precip_total=prra + prsn, precip_snow=prsn,
                    T_lowest=tas, q_lowest=huss,
                    u_lowest=uas, v_lowest=vas,
                    p_lowest=psl, p_surface=psl,
                    rho_lowest=rho_a, cos_zenith=cos_z,
                    co2_ppmv=jnp.asarray(co2_ppmv, dtype=tas.dtype),
                    has_radiation=jnp.asarray(1.0, dtype=tas.dtype),
                    has_precipitation=jnp.asarray(1.0, dtype=tas.dtype),
                )

                sst_K = state_in.T.data[..., 0] + _const.T_freeze
                u_o = jnp.zeros_like(sst_K)
                v_o = jnp.zeros_like(sst_K)
                tile = ocean_tile_response(atm, sst_K, u_o, v_o, coupler_cfg)

                if enable_ramp:
                    t_sim = day * 86400.0
                    ramp = jnp.minimum(1.0, t_sim / T_ramp_seconds)
                else:
                    ramp = 1.0

                sw_net = atm.sw_down * (1.0 - tile.albedo)
                q_net = (sw_net + atm.lw_down - tile.lw_up
                         - tile.shflx - tile.lhflx)

                _dtype = state_in.T.data.dtype
                E_rate = tile.lhflx / jnp.asarray(_const.L_v, dtype=_dtype)
                fw = FreshwaterForcing(
                    precip=jnp.asarray(prra + prsn, dtype=_dtype),
                    evap=jnp.asarray(E_rate, dtype=_dtype),
                    runoff=jnp.asarray(friver, dtype=_dtype),
                    ice_fw=jnp.zeros_like(sst_K, dtype=_dtype),
                )
                sf = OceanSurfaceForcing(
                    sw_down=atm.sw_down, q_net=q_net,
                    tau_x=tile.tau_x * ramp, tau_y=tile.tau_y * ramp,
                    freshwater=None,
                )

                # Prognostic slab sea ice: partition surface forcing between
                # open ocean (f_ocean=1-A) and the ice tile.  ocean_mask: land
                # cells receive no ice->ocean forcing (mask-aware blend
                # contract; land_mask is scan-carry state, traced-safe).
                if enable_sea_ice:
                    new_ice, fw, sf = omip_sea_ice_surface_forcing(
                        ice_state=ice_in, ice_config=ice_cfg, atm=atm,
                        ocean_sst_K=sst_K, open_ocean_sf=sf, open_ocean_fw=fw,
                        dt=dt, grid=None,
                        ocean_mask=state_in.land_mask.data,
                    )

                sponge_k = (sponge._replace(gamma=sponge.gamma * ramp)
                            if enable_sponge else None)
                new_state = _dyn_step(
                    state_in, dt, aux=aux, freshwater=fw,
                    surface_forcing=sf, sponge=sponge_k,
                )

                if enable_sss:
                    S = new_state.S.data
                    _sss_mask = new_state.land_mask.data
                    S_new = S.at[..., 0].set(
                        S[..., 0] - sss_alpha_static * (
                            S[..., 0] - sss_target_static) * _sss_mask)
                    new_state = new_state._replace(
                        S=new_state.S.replace(data=S_new))
                if enable_freeze:
                    T = new_state.T.data
                    T_top = jnp.where(
                        freeze_mask_static,
                        jnp.maximum(T[..., 0], T_freeze_C_static),
                        T[..., 0],
                    )
                    new_state = new_state._replace(
                        T=new_state.T.replace(
                            data=T.at[..., 0].set(T_top)))
                if enable_maxvel:
                    new_state = new_state._replace(
                        u=new_state.u.replace(
                            data=jnp.clip(new_state.u.data,
                                          -_maxvel_3d, _maxvel_3d)),
                        v=new_state.v.replace(
                            data=jnp.clip(new_state.v.data,
                                          -_maxvel_3d, _maxvel_3d)))

                if enable_sea_ice:
                    return (new_state, new_ice), None
                return new_state, None

            state = _seed_mass_flux_for_scan(model, state)
            init = (state, ice_state) if enable_sea_ice else state
            final, _ = jax.lax.scan(
                step_body, init,
                jnp.arange(n_steps_block, dtype=jnp.int32),
            )
            return final  # (state, ice_state) when sea-ice on, else state

        return block_fn

    # Cache block functions by size to avoid recompilation.
    _block_fn_cache = {}

    def _get_block_fn(n):
        if n not in _block_fn_cache:
            _block_fn_cache[n] = _make_block_fn(n)
        return _block_fn_cache[n]

    return _get_block_fn


# ===========================================================================
# Diagnostics
# ===========================================================================

def _extract_scalars(state, grid_type, grid, z_coord):
    """Compute scalar diagnostics from ocean state."""
    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis_3d, sh_synthesis
        T_grid = sh_synthesis_3d(grid, state.T_hat.data).real
        S_grid = sh_synthesis_3d(grid, state.S_hat.data).real
        eta_grid = sh_synthesis(grid, state.eta_hat.data).real
        mask = np.asarray(state.land_mask_grid.data)
        sst = float(np.nanmean(np.where(mask > 0.5, np.asarray(T_grid[..., 0]), np.nan)))
        sss = float(np.nanmean(np.where(mask > 0.5, np.asarray(S_grid[..., 0]), np.nan)))
        ssh = float(np.nanmean(np.where(mask > 0.5, np.asarray(eta_grid), np.nan)))
        return {"SST": sst, "SSS": sss, "SSH": ssh}

    T = np.asarray(state.T.data)
    S = np.asarray(state.S.data)
    eta = np.asarray(state.eta.data)
    mask = np.asarray(state.land_mask.data)

    if grid_type == "mpas":
        # MPAS: (nCells, nlev), mask: (nCells,)
        wet = mask > 0.5
        sst = float(np.mean(T[wet, 0])) if wet.any() else 0.0
        sss = float(np.mean(S[wet, 0])) if wet.any() else 0.0
        ssh = float(np.mean(eta[wet])) if wet.any() else 0.0
        u = np.asarray(state.u.data)
        max_u = float(np.max(np.abs(u)))
    else:
        # Cubed-sphere (6,n,n,nlev) or latlon (nlat,nlon,nlev)
        if T.ndim == 4:
            mask_3d = mask[..., np.newaxis]
        else:
            mask_3d = mask[..., np.newaxis]
        wet = mask > 0.5
        wet_3d = mask_3d > 0.5
        sst = float(np.mean(T[..., 0][wet]))
        sss = float(np.mean(S[..., 0][wet]))
        ssh = float(np.mean(eta[wet]))
        u_raw = np.asarray(state.u.data)
        v_raw = np.asarray(state.v.data) if hasattr(state, 'v') else np.zeros_like(u_raw)
        # C-grid lat-lon: u is (nlat, nlon+1, nlev), v is (nlat+1, nlon, nlev).
        # Interpolate staggered velocities to cell centers before computing speed.
        if grid_type in ("latlon", "tripole") and u_raw.shape[1] != T.shape[1]:
            u_c = 0.5 * (u_raw[:, :-1] + u_raw[:, 1:])
            v_c = 0.5 * (v_raw[:-1, :] + v_raw[1:, :])
        else:
            u_c = u_raw
            v_c = v_raw
        speed_3d = np.sqrt(u_c**2 + v_c**2)
        max_u = float(np.max(speed_3d))

    # ---- B2 standing-mode purity diagnostic P_bt ----
    # P_bt = ⟨|U_bar|²⟩ / ⟨|u_3d|²⟩ — fraction of KE in the depth-mean
    # (barotropic) component.  At a healthy spinup P_bt ≈ 0.05–0.15
    # depending on the regime; a barotropic standing mode locked onto
    # a single column drives P_bt → 1 there.  Uses simple unweighted
    # depth-mean (partial-cell thickness ignored — proxy good enough
    # for monitoring; it overweights deep columns slightly which is
    # exactly where the failure lives).  Also reports max\|u\| location.
    pbt = 0.0
    j_max = i_max = -1
    if grid_type != "spectral" and grid_type != "mpas":
        nlev_state = u_c.shape[-1]
        U_bar = np.mean(u_c, axis=-1)
        V_bar = np.mean(v_c, axis=-1)
        ke_baro = 0.5 * (U_bar**2 + V_bar**2)
        ke_3d = 0.5 * speed_3d**2
        wet_3d_b = np.broadcast_to(wet[..., np.newaxis], ke_3d.shape)
        ke_baro_total = float(np.sum(np.where(wet, ke_baro, 0.0))) * nlev_state
        ke_3d_total = float(np.sum(np.where(wet_3d_b, ke_3d, 0.0)))
        pbt = ke_baro_total / max(ke_3d_total, 1e-30)
        speed_masked = np.where(wet[..., np.newaxis], speed_3d, -1.0)
        idx = np.unravel_index(np.argmax(speed_masked), speed_3d.shape)
        j_max, i_max = int(idx[0]), int(idx[1])

    return {
        "SST": sst, "SSS": sss, "SSH": ssh,
        "max_speed": max_u if grid_type != "spectral" else 0.0,
        "P_bt": float(pbt),
        "j_maxu": j_max,
        "i_maxu": i_max,
    }


def _check_finite(state, grid_type):
    """Check if state contains finite and physically sensible values."""
    if grid_type == "spectral":
        ok_finite = bool(jnp.all(jnp.isfinite(state.T_hat.data)))
        if not ok_finite:
            return False
        T0_mag = float(jnp.abs(state.T_hat.data[0, 0]))
        return T0_mag < 1000.0

    T = state.T.data
    eta = state.eta.data
    mask = state.land_mask.data

    # Mask to ocean cells only (land cells may have uncontrolled values)
    if grid_type == "mpas":
        mask_3d = mask[:, jnp.newaxis]
        mask_2d = mask
    else:
        mask_3d = mask[..., jnp.newaxis]
        mask_2d = mask

    T_ocean = jnp.where(mask_3d > 0.5, T, 0.0)
    eta_ocean = jnp.where(mask_2d > 0.5, eta, 0.0)

    ok_finite = bool(
        jnp.all(jnp.isfinite(T_ocean))
        & jnp.all(jnp.isfinite(eta_ocean))
    )
    if not ok_finite:
        return False

    # Bound checks: ocean SSH variations are < 10 m even with
    # tsunamis (Mariana Trench depth ~11 km but η is the surface
    # elevation, not depth).  Use 1000 m as the sanity threshold —
    # well above any realistic dynamic range, but catches the
    # iter-71 cube C24 OMIP BLOWUP (eta_max=2677 m at step 500).
    # iter-79 added the η bound; previously only T was bounded
    # (< 100 °C), so an η-only blowup could in principle escape
    # detection (T might still be reasonable while η diverged).
    # The iter-71 BLOWUP was caught via the T bound at step 500
    # but the η bound is defensive.
    #
    # iter-81 codex LOW: state must remain below the threshold
    # (strict ``<``).  Exactly 1000 m would trigger a BLOWUP —
    # acceptable since 1000 m is already absurd for SSH.
    if not bool(jnp.max(jnp.abs(T_ocean)) < 100.0):
        return False
    return bool(jnp.max(jnp.abs(eta_ocean)) < 1000.0)


# ===========================================================================
# Restart I/O — minimum-viable npz format compatible with the
# global-overturning progress plotter.  We dump every field of the
# state that has a ``.data`` attribute, plus the simulation day and
# step index.  The plotter
# (``scripts/run/global_overturning/plot_realistic_geometry_progress.py``
# and the JRA55 sibling) reads these to compute MOC, BSF, snapshots.
# ===========================================================================


# Single in-flight background restart writer (see _save_restart).  The lock
# protects the dict slots (two savers in one process must serialize); the
# writer thread itself only takes the lock to store its error.
_RESTART_WRITER: dict = {
    "thread": None, "error": None, "lock": threading.Lock(),
}


def _join_restart_writer():
    """Block until the in-flight background restart write (if any) completes.

    RE-RAISES the writer's exception (ENOSPC / NFS error / failed
    ``os.replace``) into the caller — an async checkpoint failure must not
    be silently lost (codex HIGH).  Used by consumers that must READ the
    restart file right after :func:`_save_restart` returns (snapshot
    plotter, tests) and by :func:`_save_restart` itself before each new
    write."""
    with _RESTART_WRITER["lock"]:
        t = _RESTART_WRITER["thread"]
    if t is not None:
        t.join()
    with _RESTART_WRITER["lock"]:
        err = _RESTART_WRITER["error"]
        _RESTART_WRITER["error"] = None
    if err is not None:
        raise RuntimeError(
            "background restart write failed (the checkpoint file was NOT "
            "produced; the atomic .tmp+rename guarantees no truncated .npz "
            "exists)") from err


# State slots this restart format deliberately does NOT persist: pure
# DIAGNOSTICS the next step rewrites unconditionally from the prognostic state.
#
# ``_load_restart`` reconstructs from a FRESH template whose optional slots are
# ``None``, and it skips any slot the template leaves ``None`` -- so a slot
# written on save is SILENTLY DROPPED on load.  For a diagnostic that asymmetry
# is harmless in the trajectory but it (a) wastes checkpoint bytes (the #1442
# pair is ~145 MB uncompressed at eORCA1 L75) and (b) reads, to anyone
# inspecting the npz, as a persisted quantity that is in fact ignored.  Not
# writing them makes save and load agree by construction (codex YELLOW 10).
#
# Same classification the run_omip_core2 restart applies through its explicit
# ``_SLOT_POLICY`` (PR #1444, ``mass_flux_u``/``mass_flux_v`` -> DIAGNOSTIC);
# this is the older npz lane, which has no such policy table.
_RESTART_DIAGNOSTIC_SLOTS = ("mass_flux_u", "mass_flux_v", "mass_flux_w",
                             "salt_flux_u_int", "salt_flux_v_int")


def _save_restart(state, day, step, output_dir, ice_state=None,
                  grid_type="latlon"):
    """Save a state restart in the global-overturning npz format.

    The write is ASYNCHRONOUS (background thread, atomic tmp+rename) —
    the returned path may not exist for a few seconds; the previous
    write is always joined (errors re-raised) before a new one starts,
    and an ordinary interpreter exit joins the final write.  Call from
    ONE thread only (the host step loop): consecutive saves serialize
    through the module-level writer slot, concurrent savers are not
    supported.

    Mirrors ``scripts/run/global_overturning/run_global_overturning_*``
    so the same plotting helpers consume both runs without
    discrimination.

    Parameters
    ----------
    state : ocean state
    day : float
        Simulation day (filename uses ``int(round(day))``).
    step : int
        Step index (stored in npz for provenance only).
    output_dir : Path
    ice_state : SeaIceState | DynamicSeaIceState | None
        Prognostic sea-ice state (``--jra55-sea-ice``).  When supplied its
        fields are persisted under ``ice_<field>`` keys so a checkpoint/resume
        does NOT silently reset the ice pack to the zero cold start.  ``None``
        (default, no sea ice) writes the legacy ocean-only restart unchanged.
    grid_type : str
        The run's grid selection (one of ``GRID_TYPES``), stored in the npz
        for provenance AND validated on load: ``_load_restart`` compares it
        against the resuming run's grid_type and refuses a cross-grid restart
        (an MPAS checkpoint reconstructed from a latlon template can pass
        shape checks by coincidence yet be physically meaningless).
        Historically this was hardcoded to ``"latlon"``, mislabeling
        MPAS/tripole checkpoints; ``_run_omip_loop`` now threads the actual
        grid type.
    """
    if grid_type not in GRID_TYPES:
        raise ValueError(
            f"Unknown grid_type {grid_type!r} for restart provenance; "
            f"expected one of {GRID_TYPES}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Join the PREVIOUS in-flight write BEFORE pulling the new payload to
    # host: peak host memory stays ONE payload (pulling first would hold
    # payload A in the still-running writer plus payload B here — an OOM
    # exactly when the checkpoint cadence catches up to the compression
    # time, codex HIGH), and a FAILED previous write re-raises into the
    # loop here (parity with the old synchronous error behavior, one
    # checkpoint late).
    _join_restart_writer()
    payload = {
        "step": int(step),
        "time_days": float(day),
        "grid_type": grid_type,
    }
    for f in state._fields:
        if f in _RESTART_DIAGNOSTIC_SLOTS:
            continue
        obj = getattr(state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        payload[f] = np.asarray(obj.data)
    if ice_state is not None:
        for f in ice_state._fields:
            obj = getattr(ice_state, f)
            if obj is None or not hasattr(obj, "data"):
                continue
            payload[f"ice_{f}"] = np.asarray(obj.data)
    fname = output_dir / f"restart_day{int(round(day)):06d}.npz"
    # Async + atomic write.  The device->host pulls above are synchronous
    # (they snapshot the state), but the gzip+disk write (seconds to
    # minutes at eORCA-class sizes) runs on a background thread so the
    # step loop resumes immediately.  Atomic: write ``<name>.npz.tmp``
    # then ``os.replace`` — a kill mid-write can never leave a truncated
    # file that resume/chain launchers (glob ``restart_day*.npz``) would
    # mistake for a valid restart.  Non-daemon thread: an ORDINARY
    # interpreter exit (incl. the wallclock sys.exit(0) path) joins the
    # final write; a SIGKILL/scheduler hard kill can still lose the
    # in-flight checkpoint, leaving only the harmless ``.tmp``.

    def _write(fname=fname, payload=payload):
        try:
            tmp = fname.with_name(fname.name + ".tmp")
            # Pass an OPEN file handle: np.savez_compressed APPENDS ".npz"
            # to a path that does not already end in it, which would write
            # "<name>.npz.tmp.npz" and break the atomic rename.
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **payload)
            os.replace(tmp, fname)
        except BaseException as e:  # surfaced via _join_restart_writer
            with _RESTART_WRITER["lock"]:
                _RESTART_WRITER["error"] = e

    with _RESTART_WRITER["lock"]:
        t = threading.Thread(target=_write, name="omip-restart-writer",
                             daemon=False)
        t.start()
        _RESTART_WRITER["thread"] = t
    return fname


def _load_ice_restart(restart_path, ice_template):
    """Restore the prognostic sea-ice state from a restart npz (``ice_*`` keys).

    ``ice_template`` (the zero cold-start ``SeaIceState`` from the JRA55 setup)
    supplies the pytree structure + Field metadata; only the data arrays are
    overwritten.  Returns ``None`` if the restart predates sea-ice persistence
    (no ``ice_*`` keys), so an old ocean-only checkpoint resumes on the
    cold-start ice without error.
    """
    data = np.load(restart_path)
    replacements = {}
    for f in ice_template._fields:
        key = f"ice_{f}"
        if key not in data:
            continue
        obj = getattr(ice_template, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        replacements[f] = obj.replace(
            data=jnp.asarray(data[key], dtype=obj.data.dtype))
    if not replacements:
        return None
    return ice_template._replace(**replacements)


def _load_restart(restart_path, template_state, grid_type=None):
    """Load a restart npz and populate the state from a template.

    The template state (from ``_init_rest_state``) provides the pytree
    structure, Field metadata (name, dims, units), and masks.  Only the
    prognostic data arrays (u, v, T, S, eta, and optional SOM/AB2 carry
    fields) are overwritten from the restart file.

    Parameters
    ----------
    restart_path : str or Path
        Path to a ``restart_dayXXXXXX.npz`` file.
    template_state : ocean state
        A freshly initialized state with correct grid, masks, and
        z-coordinate.
    grid_type : str or None
        The resuming run's grid selection.  When supplied (not ``None``) and
        the restart npz carries a ``grid_type`` key, a mismatch is a hard
        error — reconstructing an MPAS restart from a latlon template (or
        vice-versa) can pass per-field shape checks by coincidence yet be
        physically meaningless.  Legacy restarts written before the
        ``grid_type`` key existed lack it and keep the prior best-effort
        behavior (no check).  ``None`` (a bare positional call) skips the
        guard entirely.

    Returns
    -------
    state : same type as template_state
        State with prognostic fields loaded from the restart.
    restart_day : float
        Simulation day at which the restart was saved.
    restart_step : int
        Step index at which the restart was saved.
    """
    data = np.load(restart_path)
    restart_day = float(data["time_days"])
    restart_step = int(data["step"])

    # Provenance guard: refuse a cross-grid restart.  Only enforced when the
    # caller supplies the run's grid_type AND the npz records one (legacy
    # restarts predate the key and fall through unchanged).
    if grid_type is not None and "grid_type" in data:
        saved_grid_type = str(data["grid_type"])
        if saved_grid_type != grid_type:
            raise ValueError(
                f"Restart grid_type {saved_grid_type!r} does not match the "
                f"run's grid_type {grid_type!r} ({restart_path}); loading a "
                "restart across grids reconstructs the pytree from the wrong "
                "template.  Re-run on the matching grid or regenerate the "
                "restart."
            )

    replacements = {}
    for f in template_state._fields:
        if f not in data:
            continue
        obj = getattr(template_state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        arr = jnp.asarray(data[f], dtype=obj.data.dtype)
        replacements[f] = obj.replace(data=arr)

    state = template_state._replace(**replacements)
    return state, restart_day, restart_step


# ===========================================================================
# Time loop
# ===========================================================================

def _run_omip_loop(model, state, grid_type, grid, z_coord, dt, n_steps,
                   diag_every, label="",
                   restoring_targets=None, restoring_tau_s=None,
                   restoring_ramp_days: float = 0.0,
                   jra55_state=None,
                   checkpoint_days=None, checkpoint_dir=None,
                   max_wallclock_seconds: float = 0.0,
                   restart_buffer_seconds: float = 600.0,
                   start_step=0,
                   nudge_woa_tau=0.0, T_woa_3d=None, S_woa_3d=None,
                   snapshot_fn=None, spmd_step=None, spmd_gather=None,
                   spmd_shard_stack=None):
    """Run time loop with diagnostics.

    Two forcing paths, mutually exclusive:

    * **restoring** (default): plain ``model.step(state, dt)`` followed
      by Haney SST/SSS restoring when ``restoring_targets`` is set.
      With ``spmd_step`` set (``--enable-latlon-spmd``), the dynamics
      step runs through that lat-band-SPMD callable instead — the state
      arrives sharded and every downstream op (restoring, finite checks,
      diagnostics, restart saves) works on the sharded global arrays
      transparently under the single-controller GSPMD runtime.
    * **jra55_do_tropical**: ``_jra55_step(...)`` per step using a
      pre-built JRA55-do cache (set ``jra55_state``); the model
      receives bulk-flux fields and structured freshwater forcing.

    Parameters
    ----------
    restoring_targets : tuple(T_target, S_target) or None
        Surface T/S targets for SST/SSS restoring (restoring path).
    restoring_tau_s : float or None
        Restoring timescale [seconds] (restoring path).
    jra55_state : dict or None
        Output of :func:`_setup_jra55_forcing_state`.  When provided,
        replaces the restoring path with JRA55-do bulk-flux forcing.
    checkpoint_days : float or None
        If set, save a ``restart_dayXXXXXX.npz`` snapshot every
        ``checkpoint_days`` simulated days (and at the final step).
        Format mirrors the global-overturning runs so the existing
        progress plotters consume it directly.
    checkpoint_dir : Path or None
        Output directory for restarts. Required when
        ``checkpoint_days`` is set.

    Returns (final_state, diagnostics, wall_time, ok).
    """
    if jra55_state is not None and restoring_targets is not None:
        raise ValueError(
            "_run_omip_loop: jra55_state and restoring_targets are mutually "
            "exclusive — choose one forcing path."
        )
    if spmd_step is not None and jra55_state is not None:
        # The block-scan lanes now thread spmd_step; the two unsupported
        # JRA sub-modes still refuse loudly.
        if jra55_state.get("_use_single_step", False):
            raise ValueError(
                "spmd_step + the JRA55 single-step fallback is unsupported "
                "(_jra55_step calls model.step directly); use the "
                "block-scan path (default).")
        if jra55_state.get("enable_sea_ice", False):
            raise ValueError(
                "spmd_step + prognostic sea ice is unsupported "
                "(--jra55-sea-ice; the ice tile is not SPMD-audited).")
    if checkpoint_days is not None and checkpoint_dir is None:
        raise ValueError(
            "_run_omip_loop: checkpoint_days requires checkpoint_dir."
        )
    if max_wallclock_seconds > 0.0 and checkpoint_dir is None:
        raise ValueError(
            "_run_omip_loop: max_wallclock_seconds requires checkpoint_dir."
        )
    _snapshot_fn = snapshot_fn
    diag: dict[str, list] = {"day": [], "step": []}
    snapshots: dict[int, dict] = {}
    snap_steps = {0, n_steps}
    for i in range(1, min(10, n_steps)):
        snap_steps.add(max(1, int(i * n_steps / 10)))

    # iter-97: capture BLOWUP details so ``results.txt`` can
    # surface them rather than just reporting the last *clean*
    # diagnostic (which masks BLOWUPs as "PASS-shaped FAIL").
    # The iter-96 cube OMIP smoke saw SST=19.76 in
    # results.txt, and only by re-running with verbose output
    # was it visible that max|T|=8.3M K and η=2678 m had
    # actually blown up.  The BLOWUP info now lives in
    # ``blowup_info`` and is emitted in results.txt.
    blowup_info: dict | None = None

    # Initial diagnostics (SPMD: gather the v_lower-carrying sharded state
    # — _extract_scalars centers v and needs the full n_lat+1 rows)
    scalars = _extract_scalars(
        spmd_gather(state) if spmd_gather is not None else state,
        grid_type, grid, z_coord)
    for k, v in scalars.items():
        diag.setdefault(k, []).append(v)
    diag["day"].append(0.0)
    diag["step"].append(0)

    t0 = time.time()
    last_print = t0
    blown_up = False
    # Route-B multicontroller: only rank 0 writes restart/snapshot files, but
    # EVERY rank must still dispatch the collective ``spmd_gather`` (a rank-0
    # gather would hang the others). ``jax.process_index()`` is 0 in serial /
    # single-controller runs, so this is a no-op there (never-regress).
    _io_rank = jax.process_index() == 0
    # Prognostic sea-ice carry (--jra55-sea-ice); None when ice is off.  Set in
    # the scan-blocks branch below and threaded across blocks.  Declared here so
    # the restart helpers (incl. the wallclock-exit closure) persist it — a
    # checkpoint/resume must not reset the ice pack to the cold start.
    ice_state = None

    # SPMD (--enable-latlon-spmd): restart files must carry the FULL
    # (n_lat+1) staggered v/v_mask, not the sharded v_lower layout — every
    # save choke point goes through this gather-aware wrapper.  The gather is
    # a COLLECTIVE (all ranks dispatch it); only rank 0 writes the file and
    # returns the path — non-root gets ``None`` so its callers skip the
    # snapshot/print that would deref a missing filename.
    _multiproc = jax.process_count() > 1
    if spmd_gather is not None:
        def save_restart(st, *a, **kw):
            gathered = spmd_gather(st)          # collective — ALL ranks
            if not _io_rank:
                return None
            if _multiproc:
                # Route-B: a rank-0 write error (e.g. ENOSPC) must NOT raise —
                # the collective gather already ran on every rank, so an
                # exception here would unwind rank 0 while the others advance
                # to the next collective and the federation would split.  Log
                # loudly and continue in lockstep; the next cadence retries.
                try:
                    return _save_restart(gathered, *a, **kw)
                except Exception as e:
                    print(f"    WARNING: rank-0 restart write failed "
                          f"(continuing to keep the federation in lockstep): "
                          f"{type(e).__name__}: {e}", flush=True)
                    return None
            return _save_restart(gathered, *a, **kw)   # single-proc: raise
    else:
        save_restart = _save_restart

    def _maybe_wallclock_exit(state, step: int, day: float) -> None:
        if max_wallclock_seconds <= 0:
            return                     # wallclock exit disabled — no collective
        local_exhausted = _wallclock_exhausted(
            time.time() - t0,
            max_wallclock_seconds,
            restart_buffer_seconds,
        )
        if _multiproc:
            # Route-B: the exit decision MUST be an all-rank consensus. Ranks
            # cross the wallclock threshold at slightly different times (I/O
            # jitter); if one exits (dispatching the save-gather collective +
            # sys.exit) while another runs the next step, the collective order
            # diverges and the federation hangs.  Any rank exhausted -> all
            # exit, in lockstep.
            from jax.experimental import multihost_utils
            exhausted = bool(np.any(np.asarray(
                multihost_utils.process_allgather(
                    np.asarray([local_exhausted])))))
        else:
            exhausted = local_exhausted
        if not exhausted:
            return
        fname = save_restart(state, day, step, checkpoint_dir,
                              ice_state=ice_state, grid_type=grid_type)
        # This is the LAST checkpoint before exit: join the async writer and
        # re-raise a failed write NOW.  The writer thread otherwise stores the
        # exception and the job exits 0 with a missing restart — the chain
        # launcher would then resume from an older (or no) checkpoint (codex
        # audit HIGH, 2026-07-02).
        _join_restart_writer()
        # fname is None on route-B non-root ranks (gather ran, no write).
        if fname is not None:
            if _snapshot_fn is not None:
                try:
                    _snapshot_fn(fname)
                except Exception as e:
                    print(f"    Snapshot failed: {e}", flush=True)
            print(
                f"  Wallclock budget {max_wallclock_seconds:.0f}s nearly "
                f"reached at day {day:.2f}; restart saved: {fname.name}.",
                flush=True,
            )
        sys.exit(0)

    # ---- B2 standing-mode time diagnostic χ ----
    # χ(t) = ||η^n - ½(η^{n−1} + η^{n+1})||² / ||η^n||²  (Williams 2009).
    # Tracks the 2-Δt-block computational-mode amplitude in η: a clean
    # integration sits at χ~1e-6, a growing computational mode shows χ
    # rising exponentially 5–10 days BEFORE max|u| spikes.  Buffer
    # holds the last 3 end-of-block η snapshots; we compute χ on the
    # middle once we have 3.
    eta_history: list[np.ndarray] = []

    # ----- JRA55-do block-scan path (multi-core friendly) ----------------
    # Wraps N ocean steps in lax.scan inside @jax.jit for ~30x GPU
    # speedup.  The scan body calls model._step_impl() (no inner JIT)
    # to avoid nested JIT boundaries that caused divergence with
    # partial-cell coordinates.
    use_scan_blocks = (jra55_state is not None
                       and not jra55_state.get("_use_single_step", False))
    use_gpu_interp = (jra55_state is not None
                      and jra55_state.get("_gpu_interp", False))
    # Prognostic sea ice is wired ONLY into the block-scan path.  The
    # single-step fallback (_jra55_step) does not advance the ice tile, and
    # setup has already disabled the freeze-cap SST stand-in for --jra55-sea-ice
    # — so a fallback run would get NEITHER ice NOR the freezing-point floor.
    # Fail loudly rather than silently run a degraded polar surface closure.
    if (jra55_state is not None
            and jra55_state.get("enable_sea_ice", False)
            and not use_scan_blocks):
        raise SystemExit(
            "--jra55-sea-ice requires the block-scan path, but this run set "
            "_use_single_step (single-step fallback). The single-step path "
            "does not advance prognostic sea ice and the freeze-cap stand-in "
            "is disabled under sea ice, so polar SST would be unconstrained. "
            "Use the block-scan path (default) or drop --jra55-sea-ice."
        )
    if use_scan_blocks:
        # Scan blocks trace _step_impl directly (bypassing the public
        # step shim) — prime the build-once caches from the CONCRETE
        # initial state so the traced body captures the vertex mask as
        # a constant (codex review MAJOR; census 8474554).
        # SPMD: the caller (run_omip_single) already primed the caches from
        # the UNSHARDED state before sharding; re-priming here would run
        # np.asarray on the sharded ``state`` — a hard non-addressable error
        # under route-B (shards span processes).  Skip it when spmd_step is set.
        if spmd_step is None:
            model.prime_step_caches(state)
        # Prognostic slab sea ice (--jra55-sea-ice): the block scan carries
        # (ocean_state, ice_state); thread the ice state across blocks.
        _ice_on = bool(jra55_state.get("enable_sea_ice", False))
        ice_state = jra55_state.get("ice_state_init") if _ice_on else None
        # SPMD aux (codex r18 P1): pass the step's sharded geometry stacks
        # into every block_fn call as an ARGUMENT (see the builders' note).
        _spmd_aux = getattr(spmd_step, "aux", None)
        if use_gpu_interp:
            _get_block_fn_interp = _build_jra55_block_fn_interp(
                model, jra55_state, dt, spmd_step=spmd_step)
            print("  GPU-interp mode: forcing interpolation on GPU")
            # Pre-load the full JRA55 cache for repeat-year runs to
            # eliminate per-block Zarr I/O (~0.3s/block → ~0s/block).
            _full_cache = None
            if jra55_state.get("cycle", False):
                _fc_all, _fc_days, _fc_len = _preload_jra55_full_cache(
                    jra55_state)
                _full_cache = (_fc_all, _fc_days, _fc_len)
        else:
            block_fn = _build_jra55_block_fn(model, jra55_state, dt,
                                             spmd_step=spmd_step)
            _full_cache = None
        block_size = max(1, diag_every)
        if checkpoint_days is not None:
            steps_per_ckpt = max(1, int(round(checkpoint_days * 86400.0 / dt)))
        else:
            steps_per_ckpt = None

        block_start = start_step
        while block_start < n_steps and not blown_up:
            actual = min(block_size, n_steps - block_start)
            t_io_start = time.time()

            if use_gpu_interp and _full_cache is not None:
                raw_stack, runoff_records, record_meta = (
                    _slice_preloaded_records(
                        block_start, actual, dt, jra55_state,
                        *_full_cache))
            elif use_gpu_interp:
                raw_stack, runoff_records, record_meta = (
                    _preload_jra55_raw_records(
                        block_start, actual, dt, jra55_state))
            else:
                atm_stack, runoff_stack = _preload_jra55_forcing_block(
                    block_start, actual, dt, jra55_state,
                )
            if spmd_shard_stack is not None:
                # Lay the per-block forcing stacks out lat-band-sharded so
                # the in-scan interpolation / bulk fluxes stay shard-local
                # (an unsharded stack commits to device 0 and serializes
                # every forcing op there).
                if use_gpu_interp:
                    raw_stack = spmd_shard_stack(raw_stack)
                    runoff_records = spmd_shard_stack(runoff_records)
                else:
                    atm_stack = spmd_shard_stack(atm_stack)
                    runoff_stack = spmd_shard_stack(runoff_stack)
            io_dt = time.time() - t_io_start

            t_compute_start = time.time()
            if use_gpu_interp:
                bfn = _get_block_fn_interp(actual)
                if _ice_on:
                    state, ice_state = bfn(
                        state, raw_stack, runoff_records,
                        record_meta["record_days"],
                        jnp.float64(record_meta["block_start_day"]),
                        jnp.float64(record_meta["block_start_day_forcing"]),
                        ice_state, aux=_spmd_aux,
                    )
                else:
                    state = bfn(
                        state, raw_stack, runoff_records,
                        record_meta["record_days"],
                        jnp.float64(record_meta["block_start_day"]),
                        jnp.float64(record_meta["block_start_day_forcing"]),
                        aux=_spmd_aux,
                    )
            else:
                if _ice_on:
                    state, ice_state = block_fn(
                        state, atm_stack, runoff_stack,
                        jnp.int32(block_start), ice_state, aux=_spmd_aux,
                    )
                else:
                    state = block_fn(
                        state, atm_stack, runoff_stack,
                        jnp.int32(block_start), aux=_spmd_aux,
                    )
            jax.block_until_ready(state.T.data)
            compute_dt = time.time() - t_compute_start

            block_start += actual
            step = block_start
            day = step * dt / 86400.0

            if not _check_finite(state, grid_type):
                print(f"  BLOWUP at step {step}")
                blown_up = True
                break

            # WOA T/S nudging: dX/dt += (X_woa - X) / tau
            # After nudging, correct global-mean eta to prevent steric
            # volume drift from the density change.
            if nudge_woa_tau > 0 and T_woa_3d is not None:
                nudge_per_step = dt / (nudge_woa_tau * 86400.0)
                daily_frac = 1.0 - (1.0 - nudge_per_step) ** actual
                mask_3d = state.land_mask.data[..., jnp.newaxis]
                mask_2d = state.land_mask.data

                # Save pre-nudge eta mean for volume correction
                area = grid.areaCell if hasattr(grid, 'areaCell') else None
                if area is not None:
                    eta_mean_before = jnp.sum(
                        state.eta.data * mask_2d * area
                    ) / jnp.maximum(jnp.sum(mask_2d * area), 1e-10)

                T_nudged = state.T.data + daily_frac * (
                    T_woa_3d - state.T.data) * mask_3d
                state = state._replace(
                    T=state.T.replace(data=T_nudged.astype(state.T.data.dtype)))
                if S_woa_3d is not None:
                    S_nudged = state.S.data + daily_frac * (
                        S_woa_3d - state.S.data) * mask_3d
                    state = state._replace(
                        S=state.S.replace(data=S_nudged.astype(state.S.data.dtype)))

                # Restore global-mean eta to pre-nudge value so the
                # nudge doesn't inject/remove volume stericly.
                if area is not None:
                    eta_mean_after = jnp.sum(
                        state.eta.data * mask_2d * area
                    ) / jnp.maximum(jnp.sum(mask_2d * area), 1e-10)
                    eta_correction = eta_mean_before - eta_mean_after
                    eta_corrected = state.eta.data + eta_correction * mask_2d
                    state = state._replace(
                        eta=state.eta.replace(data=eta_corrected))

            _st_diag = (spmd_gather(state) if spmd_gather is not None
                        else state)
            scalars = _extract_scalars(_st_diag, grid_type, grid, z_coord)

            # B2: chi diagnostic from last 3 eta snapshots.  Read the GATHERED
            # state (_st_diag): under route-B the raw ``state.eta`` is sharded
            # across PROCESSES, so np.asarray on it would fail on the
            # non-addressable remote shards.  _st_diag is replicated.
            chi = 0.0
            if grid_type == "latlon":
                eta_now = np.asarray(_st_diag.eta.data)
                eta_history.append(eta_now)
                if len(eta_history) > 3:
                    eta_history.pop(0)
                if len(eta_history) == 3:
                    eta_m2, eta_m1, eta_0 = eta_history
                    mask_eta = np.asarray(_st_diag.land_mask.data) > 0.5
                    diff = (eta_m1 - 0.5 * (eta_0 + eta_m2)) * mask_eta
                    den = eta_m1 * mask_eta
                    num_sq = float(np.sum(diff * diff))
                    den_sq = float(np.sum(den * den))
                    chi = num_sq / max(den_sq, 1e-30)
            scalars["chi"] = chi

            # Convert max|u| location indices → lat/lon for readability
            if grid_type == "latlon" and "j_maxu" in scalars and scalars["j_maxu"] >= 0:
                lat_v = np.asarray(grid.lat) if hasattr(grid, "lat") else None
                lon_v = np.asarray(grid.lon) if hasattr(grid, "lon") else None
                if lat_v is not None and lon_v is not None:
                    j = scalars["j_maxu"]; i = scalars["i_maxu"]
                    if 0 <= j < len(lat_v) and 0 <= i < len(lon_v):
                        scalars["lat_maxu"] = float(lat_v[j])
                        scalars["lon_maxu"] = float(lon_v[i])

            diag["day"].append(day)
            diag["step"].append(step)
            for k, v in scalars.items():
                diag.setdefault(k, []).append(v)

            elapsed_total = time.time() - t0
            total_days = n_steps * dt / 86400.0
            # Custom summary string includes the new diagnostics
            scalar_summary = (
                f"SST={scalars['SST']:.3g} "
                f"max|u|={scalars['max_speed']:.3g} "
                f"P_bt={scalars['P_bt']:.3g} "
                f"χ={chi:.2e}"
            )
            if "lat_maxu" in scalars:
                scalar_summary += (
                    f" @({scalars['lat_maxu']:.0f},"
                    f"{scalars['lon_maxu']:.0f})"
                )
            summary = scalar_summary
            if _io_rank:
                print(
                    f"    [{label}] Day {day:7.2f}/{total_days:.0f} | {summary} "
                    f"| io={io_dt:.1f}s compute={compute_dt:.1f}s "
                    f"({compute_dt/actual:.2f} s/step) | {elapsed_total:.0f}s total",
                    flush=True,
                )

            if (steps_per_ckpt is not None and
                    (step % steps_per_ckpt == 0 or step == n_steps)):
                fname = save_restart(state, day, step, checkpoint_dir,
                                      ice_state=ice_state, grid_type=grid_type)
                # fname is None on route-B non-root ranks (gather ran, no write).
                if fname is not None:
                    if _snapshot_fn is not None:
                        try:
                            _snapshot_fn(fname)
                        except Exception as e:
                            print(f"    Snapshot failed: {e}", flush=True)
                    print(f"    Restart saved: {fname.name}", flush=True)
            _maybe_wallclock_exit(state, step, day)

        # After the block loop, jump to the post-loop tally below.
        if grid_type == "spectral":
            jax.block_until_ready(state.T_hat.data)
        else:
            jax.block_until_ready(state.T.data)
        wall = time.time() - t0
        ok = not blown_up and _check_finite(state, grid_type)
        return state, diag, wall, ok, blowup_info
    # --------------------------------------------------------------------

    # ----- JRA55-do single-step path (partial-cell fallback) -----------
    if jra55_state is not None and not use_scan_blocks:
        if checkpoint_days is not None:
            steps_per_ckpt = max(1, int(round(checkpoint_days * 86400.0 / dt)))
        else:
            steps_per_ckpt = None

        for i in range(start_step, n_steps):
            state = _jra55_step(state, i, dt, model, jra55_state)

            step = i + 1
            if step % diag_every == 0 or step == n_steps:
                day = step * dt / 86400.0
                if not _check_finite(state, grid_type):
                    print(f"  BLOWUP at step {step}")
                    blown_up = True
                    break
                scalars = _extract_scalars(state, grid_type, grid, z_coord)
                diag["day"].append(day)
                diag["step"].append(step)
                for k, v in scalars.items():
                    diag.setdefault(k, []).append(v)
                elapsed_total = time.time() - t0
                total_days = n_steps * dt / 86400.0
                summary = " | ".join(
                    f"{k}={v:.4g}" for k, v in list(scalars.items())[:4]
                )
                print(
                    f"    [{label}] Day {day:7.2f}/{total_days:.0f} | {summary} "
                    f"| {elapsed_total:.0f}s total",
                    flush=True,
                )
                if (steps_per_ckpt is not None and
                        (step % steps_per_ckpt == 0 or step == n_steps)):
                    fname = save_restart(state, day, step, checkpoint_dir,
                                          ice_state=ice_state,
                                          grid_type=grid_type)
                    if _snapshot_fn is not None:
                        try:
                            _snapshot_fn(fname)
                        except Exception as e:
                            print(f"    Snapshot failed: {e}", flush=True)
                    print(f"    Restart saved: {fname.name}", flush=True)
                _maybe_wallclock_exit(state, step, day)

        jax.block_until_ready(state.T.data)
        wall = time.time() - t0
        ok = not blown_up and _check_finite(state, grid_type)
        return state, diag, wall, ok, blowup_info
    # --------------------------------------------------------------------

    # Restoring ramp: scale restoring strength linearly from 0 to 1 over
    # the first ``restoring_ramp_steps`` steps.  See ``--restoring-ramp-
    # days``; cubed_sphere defaults to 14 days at the call-site to
    # delay the face-edge PGF instability onset, other grids default
    # to 0.
    ramp_days = float(restoring_ramp_days)
    restoring_ramp_steps = max(1, int(ramp_days * 86400.0 / dt)) if ramp_days > 0 else 1

    _dyn_step = spmd_step if spmd_step is not None else model.step

    for i in range(start_step, n_steps):
        state = _dyn_step(state, dt)

        # Apply SST/SSS restoring (grid-agnostic, after dynamics step)
        if restoring_targets is not None:
            T_tgt, S_tgt = restoring_targets
            ramp_scale = min(1.0, (i + 1) / restoring_ramp_steps)
            state = _apply_restoring(
                state, grid_type, grid, T_tgt, S_tgt,
                dt, restoring_tau_s, ramp_scale=ramp_scale,
            )

        step = i + 1

        if step % 100 == 0:
            if not _check_finite(state, grid_type):
                # Debug: identify what failed
                if grid_type != "spectral":
                    mask = state.land_mask.data
                    m3 = mask[:, jnp.newaxis] if grid_type == "mpas" else mask[..., jnp.newaxis]
                    T_oc = jnp.where(m3 > 0.5, state.T.data, 0.0)
                    eta_max = float(jnp.max(jnp.abs(state.eta.data)))
                    eta_finite = bool(jnp.all(jnp.isfinite(state.eta.data)))
                    T_max = float(jnp.max(jnp.abs(T_oc)))
                    T_finite = bool(jnp.all(jnp.isfinite(T_oc)))
                    print(
                        f"  BLOWUP step {step}: "
                        f"max|T|={T_max:.1f} "
                        f"T_finite={T_finite} "
                        f"eta_max={eta_max:.2f} "
                        f"eta_finite={eta_finite}"
                    )
                    # iter-81 codex LOW: report Reason for BOTH T
                    # and η triggers (iter-79 reported only η).
                    # Multiple conditions can fire simultaneously
                    # (e.g., a NaN cascade hits both T and η).
                    reasons = []
                    if not T_finite:
                        msg = "T contains NaN/Inf"
                        print(f"    Reason: {msg}")
                        reasons.append(msg)
                    elif T_max >= 100.0:
                        msg = (f"|T| reached {T_max:.1f} °C "
                               f"(sanity threshold 100 °C)")
                        print(f"    Reason: {msg}")
                        reasons.append(msg)
                    if not eta_finite:
                        msg = "η contains NaN/Inf"
                        print(f"    Reason: {msg}")
                        reasons.append(msg)
                    elif eta_max >= 1000.0:
                        msg = (f"|η| reached {eta_max:.0f} m "
                               f"(iter-79 sanity threshold 1000 m)")
                        print(f"    Reason: {msg}")
                        reasons.append(msg)
                    # iter-97: persist BLOWUP info so the report
                    # can surface it in results.txt.
                    blowup_info = {
                        "step": step,
                        "day": step * dt / 86400.0,
                        "T_max": T_max,
                        "T_finite": T_finite,
                        "eta_max": eta_max,
                        "eta_finite": eta_finite,
                        "reasons": reasons,
                    }
                else:
                    print(f"  BLOWUP at step {step}")
                    blowup_info = {
                        "step": step,
                        "day": step * dt / 86400.0,
                        "reasons": ["spectral state non-finite"],
                    }
                blown_up = True
                break

        if step % diag_every == 0 or step == n_steps:
            day = step * dt / 86400.0
            # SPMD: _extract_scalars centers v (needs the full n_lat+1
            # staggered rows) — gather the v_lower-carrying sharded state
            # at the diag cadence only.
            _st_diag = spmd_gather(state) if spmd_gather is not None else state
            scalars = _extract_scalars(_st_diag, grid_type, grid, z_coord)
            diag["day"].append(day)
            diag["step"].append(step)
            for k, v in scalars.items():
                diag.setdefault(k, []).append(v)

            now = time.time()
            if now - last_print > 15:
                summary = " | ".join(
                    f"{k}={v:.4g}" for k, v in list(scalars.items())[:4])
                elapsed = now - t0
                total_days = n_steps * dt / 86400.0
                if _io_rank:
                    print(f"    [{label}] Day {day:7.1f}/{total_days:.0f} | "
                          f"{summary} | {elapsed:.0f}s elapsed")
                last_print = now

        # Restart-checkpoint cadence (independent of the diag cadence
        # so checkpoints land on round-number simulation days).
        if checkpoint_days is not None:
            day_now = step * dt / 86400.0
            steps_per_ckpt = max(1, int(round(checkpoint_days * 86400.0 / dt)))
            if step % steps_per_ckpt == 0 or step == n_steps:
                fname = save_restart(state, day_now, step, checkpoint_dir,
                                      ice_state=ice_state, grid_type=grid_type)
                # fname is None on route-B non-root ranks (gather ran, no write).
                if fname is not None:
                    # Auto-generate snapshot plot alongside the restart.
                    if _snapshot_fn is not None:
                        try:
                            _snapshot_fn(fname)
                        except Exception as e:
                            print(f"    Snapshot failed: {e}", flush=True)
                    # Friendly progress; gated on the same 15-s cadence as
                    # the diag print so we don't spam.
                    if time.time() - last_print < 1.0:
                        print(f"    Restart saved: {fname.name}", flush=True)
        _maybe_wallclock_exit(state, step, step * dt / 86400.0)

    if grid_type == "spectral":
        jax.block_until_ready(state.T_hat.data)
    else:
        jax.block_until_ready(state.T.data)

    wall = time.time() - t0
    ok = not blown_up and _check_finite(state, grid_type)
    return state, diag, wall, ok, blowup_info


# ===========================================================================
# Output
# ===========================================================================

def _save_output(output_dir: Path, diag, args, grid_type, wall_time, ok,
                 blowup_info: dict | None = None, write: bool = True):
    """Save diagnostics and metadata.

    iter-97: ``blowup_info`` (added kwarg) carries the BLOWUP
    step / max|T| / max|η| / reasons captured at the time the
    state went non-finite, so ``results.txt`` can clearly mark
    BLOWUP runs as such instead of silently reporting the last
    *clean* SST/SSS/SSH (which led to a false-improvement claim
    in iter-96).

    ``write=False`` (route-B non-root ranks) skips every file write but still
    builds and returns the ``results`` dict, so ALL_RESULTS / the exit code
    stay consistent across the federation without N processes clobbering the
    same output files.
    """
    import csv
    keys = list(diag.keys())

    # Results dict — built for EVERY rank (ALL_RESULTS / exit-code
    # consistency); route-B non-root ranks return it here WITHOUT writing any
    # file, so N processes never clobber the same output paths.
    results = {
        "grid_type": grid_type,
        "resolution": args.resolution or GRID_DEFAULTS[grid_type]["resolution"],
        "nlev": args.nlev,
        "days": args.days,
        "dt": args.dt or GRID_DEFAULTS[grid_type]["dt"],
        "physics": args.physics,
        "water_type": args.water_type,
        "sw_down": args.sw_down,
        "wall_time_s": wall_time,
        "status": "PASS" if ok else "FAIL",
        "final_SST": diag["SST"][-1] if diag["SST"] else None,
        "final_SSS": diag["SSS"][-1] if diag["SSS"] else None,
        "final_SSH": diag["SSH"][-1] if diag["SSH"] else None,
        "blowup_info": blowup_info,
        "cli_args": vars(args),
    }
    if not write:
        return results

    output_dir.mkdir(parents=True, exist_ok=True)

    # Timeseries CSV
    csv_path = output_dir / "timeseries.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        n_rows = len(diag[keys[0]])
        for i in range(n_rows):
            w.writerow([diag[k][i] if i < len(diag[k]) else "" for k in keys])

    # iter-25: also write a matrix-compatible ``mean_timeseries.csv``
    # so ``run_ocean_test_matrix.py --replot`` (and the cross-grid
    # plotter generally) can pick this up.  The ocean-matrix
    # plotter expects ``time_days`` column + a set of mean_*
    # diagnostics; rename columns appropriately and write a
    # parallel CSV.  Don't replace ``timeseries.csv`` since that
    # filename + ``results.json`` is the existing OMIP output
    # contract.
    mean_csv_path = output_dir / "mean_timeseries.csv"
    column_renames = {
        "day": "time_days",
        "SST": "mean_SST",
        "SSS": "mean_SSS",
        "SSH": "mean_eta",
        "max_speed": "max_speed",
        "step": "step",
    }
    out_keys = [column_renames.get(k, k) for k in keys]
    with open(mean_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(out_keys)
        n_rows = len(diag[keys[0]])
        for i in range(n_rows):
            w.writerow([diag[k][i] if i < len(diag[k]) else "" for k in keys])

    # Results JSON — the dict was built above (before the write gate).
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # iter-25: also write a matrix-compatible ``results.txt`` next to
    # the existing ``results.json`` so the ocean cross-grid plotter
    # (which parses ``key: value`` lines from results.txt) can pick
    # this up.
    final_sst = diag["SST"][-1] if diag.get("SST") else None
    final_sss = diag["SSS"][-1] if diag.get("SSS") else None
    final_ssh = diag["SSH"][-1] if diag.get("SSH") else None
    notes_parts = []
    # iter-97: lead with BLOWUP marker when the run failed
    # because a BLOWUP was detected.  This is unambiguous —
    # readers no longer mistake "SST=19.76 (last clean diag)"
    # for a healthy run.
    if blowup_info is not None:
        notes_parts.append(
            f"BLOWUP at step {blowup_info['step']} "
            f"(day {blowup_info.get('day', 0):.2f})"
        )
        if "T_max" in blowup_info:
            notes_parts.append(f"max|T|={blowup_info['T_max']:.3e} °C")
        if "eta_max" in blowup_info:
            notes_parts.append(f"max|η|={blowup_info['eta_max']:.0f} m")
        for reason in blowup_info.get("reasons", []):
            notes_parts.append(f"reason: {reason}")
        # Also keep the last clean diagnostic so a reader can
        # see what the system looked like at the last sane
        # state — but mark it as such.
        if final_sst is not None:
            notes_parts.append(f"last clean SST={final_sst:.3f}")
    else:
        if final_sst is not None:
            notes_parts.append(f"SST={final_sst:.3f}")
        if final_sss is not None:
            notes_parts.append(f"SSS={final_sss:.3f}")
        if final_ssh is not None:
            notes_parts.append(f"SSH={final_ssh:.3e}")
    notes_str = ", ".join(notes_parts) if notes_parts else "OMIP complete"
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"test: omip\n")
        f.write(f"grid: {grid_type}\n")
        f.write(f"resolution: {results['resolution']}\n")
        f.write(f"days: {results['days']}\n")
        f.write(f"dt: {results['dt']}\n")
        f.write(f"levels: {results['nlev']}\n")
        f.write(f"physics: {results['physics']}\n")
        f.write(f"status: {results['status']}\n")
        f.write(f"notes: {notes_str}\n")
        f.write(f"wall_time: {wall_time:.1f}s\n")

    # Plot timeseries
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        days = diag["day"]

        for ax, key in zip(axes.flat, ["SST", "SSS", "SSH", "max_speed"]):
            if key in diag:
                ax.plot(days, diag[key])
                ax.set_xlabel("Day")
                ax.set_ylabel(key)
                ax.set_title(key)
                ax.grid(True, alpha=0.3)

        fig.suptitle(f"OMIP {grid_type} — {args.physics} physics", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / "timeseries.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass  # plotting is optional

    return results


# ===========================================================================
# Single-grid runner
# ===========================================================================

def run_omip_single(grid_type: str, args) -> dict:
    """Run OMIP simulation on a single grid type."""
    # Apply precision BEFORE any model state is built. Idempotent + safe for
    # both main() (which also calls it) and direct in-process callers that
    # invoke run_omip_single() without going through main() — the module no
    # longer force-applies fp64 at import (codex 2026-06-21).
    apply_run_precision(args)
    run_config = build_config_from_args(args)
    # --multicontroller is ONLY the route-B transport for the lat-band ocean
    # SPMD lane — it does nothing on its own.  Without --enable-latlon-spmd the
    # SPMD block below is skipped, so spmd_gather stays None and EVERY rank runs
    # the full serial model AND writes the SAME output/restart paths, corrupting
    # them.  Hard-fail (all ranks raise identically) BEFORE any model/device
    # work rather than silently clobber (codex r2 #2).
    if run_config.multicontroller and not run_config.enable_latlon_spmd:
        raise SystemExit(
            "--multicontroller requires --enable-latlon-spmd (it is the "
            "route-B transport for the lat-band ocean SPMD step). Without the "
            "SPMD lane every rank would run the full model and clobber the "
            "same output files.")
    # Apply the --params calibration layer (tuned scheme parameters) into the
    # built config's nested *Config NamedTuples (issue #691).
    if getattr(args, "params", None):
        from legoesm.driver.run_config_yaml import (
            apply_params_to_config,
            load_params_config,
        )
        run_config = apply_params_to_config(
            run_config, load_params_config(args.params), driver="run_omip")
    # iter-115 codex iter-114-followup HIGH-1: pre-iter-115,
    # ``--resolution 16`` was applied verbatim to every grid
    # type.  Cube/spectral parsed it (silently wrong: cube
    # ``int("16"[1:]) = 6``); latlon errored; mpas
    # interpreted as level 16 (4.29e+10 cells).  Now use the
    # iter-115 shared dispatch helper.
    if args.resolution is not None:
        from legoesm.driver.cli_resolution import (
            expand_cli_resolution, validate_cli_resolution,
        )
        N = validate_cli_resolution(
            args.resolution,
            additional_examples="'C24', 'ico3', '36x72', 'T21', '50km'",
        )
        if N is not None:
            resolution = expand_cli_resolution(N, grid_type)
        else:
            # Pre-formatted per-grid string — pass through.
            resolution = args.resolution
    else:
        resolution = GRID_DEFAULTS[grid_type]["resolution"]
    dt = args.dt or GRID_DEFAULTS[grid_type]["dt"]
    days = 30.0 if args.quick else args.days
    n_steps = int(days * 86400.0 / dt)
    diag_every = args.diag_every or max(1, int(86400.0 / dt))  # ~daily

    print(f"\n{'='*70}")
    print(f"  OMIP: {grid_type} | {resolution} | {args.nlev} levels | "
          f"dt={dt:.0f}s | {days:.0f} days ({n_steps} steps)")
    print(f"  Physics: {args.physics} | SW: {args.sw_down} W/m² | "
          f"Water type: {args.water_type}")
    print(f"{'='*70}")

    t_setup = time.time()

    # Create grid + model (all grids use identical config-based diffusion
    # for cross-grid consistency; physics pipeline disabled).
    grid, z_coord, config, model, coord_kind = _create_setup(
        grid_type, resolution, args.nlev, args.H_max,
        args.physics, args.water_type,
        use_bathymetry=(args.bathymetry is not None),
        A_h_override=args.A_h,
        B_h_override=args.B_h,
        K_h_override=args.K_h,
        A_h_eq_boost=args.A_h_eq_boost,
        A_h_eq_sigma_deg=args.A_h_eq_sigma,
        C_smag=args.C_smag,
        C_smag_lap=args.C_smag_lap,
        A_h_floor=args.A_h_floor,
        C_leith=args.C_leith,
        pgf_scheme=args.pgf_scheme,
        slope_foot_alpha=args.slope_foot_alpha,
        no_lat_scaling=args.no_lat_scaling,
        no_gm_redi=getattr(args, "no_gm_redi", False),
        gm_redi=run_config.gm_redi,
        implicit_vertical_mixing=getattr(
            args, "implicit_vertical_mixing", False),
        vertical_mixing=run_config.vertical_mixing,
        forcing_mode=getattr(args, "forcing_mode", "restoring"),
        use_conservation_fixer=not args.no_conservation_fixer,
    )
    # NEMO zdfdrg drag-law + zdfiwm forcing-map overrides (no-op when the
    # flags are at their legacy defaults; rebuilds the model so the jitted
    # step captures the new config / maps).
    config, model = _apply_drag_iwm_overrides(
        args, grid_type, grid, z_coord, config, model)

    # --- Initialization strategy ---
    # Start from rest state with uniform T/S, then restore toward WOA
    # climatology.  Starting from full WOA T/S creates extreme pressure
    # gradients that trigger violent geostrophic adjustment (200+ m/s
    # currents, SSS >70 PSU).  Uniform start + restoring is the standard
    # OMIP spin-up approach: the model gradually builds up the
    # climatological circulation from rest.
    from legoesm.ocean.init_woa import init_ocean_from_woa
    T_woa, S_woa = init_ocean_from_woa(grid, z_coord, args.woa_t, args.woa_s)

    # Bathymetry: realistic (ETOPO) or flat-bottom.
    H_bathy_init = None
    land_mask_init = None
    bathy_cfg = None
    if args.bathymetry is not None and grid_type not in ("mpas", "tripole"):
        # MPAS bathymetry is handled after _init_rest_state via the
        # PR 261 recipe (load_bathymetry_mpas + north cap + snap +
        # partial cells). Tripole follows the same post-init pattern,
        # with init_ocean_bathymetry on the curvilinear geom. The
        # generic path here has lat-lon-specific post-processing
        # (equatorial smoothing, passage widening) that doesn't apply
        # to MPAS or tripole.
        from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
        bathy_cfg = BathymetryConfig(
            source="file",
            path=args.bathymetry,
            H_max=args.H_max,
            H_min=args.H_min,
            smoothing_passes=args.smoothing_passes,
            enforce_straits=True,
            fill_isolated_basins=True,
            depth_is_negative=True,
            r_factor_max=args.r_factor_max,
            north_cap_lat=args.north_cap_lat,
            south_cap_lat=args.south_cap_lat,
        )
        H_bathy_init, land_mask_init = init_ocean_bathymetry(grid, bathy_cfg)
        H_bathy_init = jnp.asarray(H_bathy_init, dtype=jnp.float64)
        land_mask_init = jnp.asarray(land_mask_init, dtype=jnp.float64)
        n_ocean = int(np.sum(np.asarray(land_mask_init) > 0.5))
        n_total = int(np.prod(np.asarray(land_mask_init).shape))
        print(f"  Bathymetry: {Path(args.bathymetry).name} "
              f"({n_ocean}/{n_total} ocean cells, "
              f"H_min={args.H_min}m, {args.smoothing_passes} smoothing passes, "
              f"r_max={args.r_factor_max})")

        # Equatorial-only extra smoothing.
        # The 30-day spinup diagnosed a barotropic standing-mode
        # instability at deep ocean cells in the equatorial belt
        # (Java Trench off Sumatra; deep Atlantic; deep Pacific east of
        # S. America), with growth factors of 500-1200× in 20 days.
        # Common features: |lat|<13°, H>4000 m, large ∂H/∂x near coast.
        # f≈0 there cannot damp PGF errors driven by steep H gradients.
        # The standard fix is more aggressive bathymetry smoothing in
        # the equatorial belt: the loss of "true" bathymetry detail at
        # ±10° is acceptable for a 1° model that can't resolve the EUC
        # anyway.  Apply a Laplacian smoother with cosine taper from
        # full strength at the equator to zero at the band edge, and
        # average only over wet neighbours (don't pull from land).
        eq_band = 15.0          # smooth within ±15° of equator
        eq_passes = 30           # extra Laplacian passes at the equator
        H_np = np.asarray(H_bathy_init)
        ocean_mask = np.asarray(land_mask_init) > 0.5
        # Cell-center latitudes (axis 0)
        from legoesm.grids.latlon import create_latlon_grid as _clg
        # we already have grid; pull lat
        lat_c = np.asarray(grid.lat) if hasattr(grid, 'lat') else None
        if lat_c is not None:
            # taper alpha(lat): quadratic, 1 at eq, 0 at ±eq_band
            # grid.lat is in radians; convert eq_band to radians
            eq_band_rad = np.deg2rad(eq_band)
            x = np.clip(np.abs(lat_c) / eq_band_rad, 0.0, 1.0)
            taper = (1.0 - x**2)                # (n_lat,)
            taper2d = np.broadcast_to(taper[:, None], H_np.shape)
            n_lat_g, n_lon_g = H_np.shape
            for _p in range(eq_passes):
                # neighbour sum over wet cells only, with periodic lon
                up    = np.roll(H_np, -1, axis=0); up_m    = np.roll(ocean_mask, -1, axis=0)
                down  = np.roll(H_np,  1, axis=0); down_m  = np.roll(ocean_mask,  1, axis=0)
                left  = np.roll(H_np,  1, axis=1); left_m  = np.roll(ocean_mask,  1, axis=1)
                right = np.roll(H_np, -1, axis=1); right_m = np.roll(ocean_mask, -1, axis=1)
                # No wrap in lat: zero the off-grid neighbour mask
                up_m[-1, :] = False; down_m[0, :] = False
                neigh_sum = (up * up_m + down * down_m
                             + left * left_m + right * right_m)
                neigh_cnt = up_m.astype(np.float64) + down_m + left_m + right_m
                avg = np.where(neigh_cnt > 0,
                               neigh_sum / np.maximum(neigh_cnt, 1.0),
                               H_np)
                # alpha = 0.5 * taper(lat) → mixes self with neighbour avg
                alpha = 0.5 * taper2d
                new_H = (1.0 - alpha) * H_np + alpha * avg
                # Keep land cells fixed
                H_np = np.where(ocean_mask, new_H, H_np)
            # Floor at H_min so the smoother doesn't accidentally
            # create cells shallower than the physical floor.
            H_np = np.where(ocean_mask, np.maximum(H_np, args.H_min), H_np)
            H_bathy_init = jnp.asarray(H_np, dtype=jnp.float64)
            print(f"  Equatorial smoothing: {eq_passes} extra Laplacian "
                  f"passes within ±{eq_band:.0f}° (cosine taper, wet-only)")

        # --- Close Arctic completely (solid wall) ---
        if args.close_arctic_lat is not None:
            ocean_mask = np.array(land_mask_init, copy=True) > 0.5
            lat_c = np.asarray(grid.lat) if hasattr(grid, 'lat') else None
            if lat_c is not None:
                # grid.lat is in radians; convert threshold to radians
                arctic_rows = lat_c > np.deg2rad(args.close_arctic_lat)
                n_closed = int(np.sum(ocean_mask[arctic_rows, :]))
                ocean_mask[arctic_rows, :] = False
                # Keep H_bathy unchanged — land cells retain depth values
                # but are masked out (setting H=0 confuses partial-cell coord).
                land_mask_init = jnp.asarray(ocean_mask.astype(np.float64))
                print(f"  Arctic closure: {n_closed} cells → land above "
                      f"{args.close_arctic_lat:.1f}°N")

        # --- Widen narrow passages ---
        if args.min_passage_width >= 2:
            H_np = np.array(H_bathy_init, copy=True)
            ocean_mask = np.array(land_mask_init, copy=True) > 0.5
            n_lat_g, n_lon_g = ocean_mask.shape
            fill_cells = np.zeros_like(ocean_mask)
            min_w = args.min_passage_width

            # Find cells that are part of passages narrower than min_w
            # in the zonal direction (land on both sides within min_w-1)
            for j in range(n_lat_g):
                for i in range(n_lon_g):
                    if not ocean_mask[j, i]:
                        continue
                    # Check zonal width: how many consecutive ocean cells
                    # in the east-west direction including this cell?
                    width = 1
                    # count east
                    for di in range(1, min_w):
                        ii = (i + di) % n_lon_g
                        if ocean_mask[j, ii]:
                            width += 1
                        else:
                            break
                    # count west
                    for di in range(1, min_w):
                        ii = (i - di) % n_lon_g
                        if ocean_mask[j, ii]:
                            width += 1
                        else:
                            break
                    if width < min_w:
                        fill_cells[j, i] = True

            # Same for meridional direction
            for j in range(n_lat_g):
                for i in range(n_lon_g):
                    if not ocean_mask[j, i]:
                        continue
                    width = 1
                    for dj in range(1, min_w):
                        jj = j + dj
                        if jj < n_lat_g and ocean_mask[jj, i]:
                            width += 1
                        else:
                            break
                    for dj in range(1, min_w):
                        jj = j - dj
                        if jj >= 0 and ocean_mask[jj, i]:
                            width += 1
                        else:
                            break
                    if width < min_w:
                        # Only fill if ALSO narrow zonally (avoid filling
                        # long coastlines). A true narrow passage is narrow
                        # in at least one direction.
                        fill_cells[j, i] = True

            # Actually we want cells that are narrow in BOTH directions
            # to be filled... No — a 1-cell-wide strait running N-S is
            # narrow zonally but wide meridionally. We want to fill cells
            # narrow in ANY direction. But let's be more careful:
            # Fill cells that are zonally narrow (land within min_w on both sides)
            fill_zonal = np.zeros_like(ocean_mask)
            fill_merid = np.zeros_like(ocean_mask)
            for j in range(n_lat_g):
                for i in range(n_lon_g):
                    if not ocean_mask[j, i]:
                        continue
                    # Zonal: find distance to land on each side
                    dist_e = 0
                    for di in range(1, min_w + 1):
                        ii = (i + di) % n_lon_g
                        if ocean_mask[j, ii]:
                            dist_e += 1
                        else:
                            break
                    dist_w = 0
                    for di in range(1, min_w + 1):
                        ii = (i - di) % n_lon_g
                        if ocean_mask[j, ii]:
                            dist_w += 1
                        else:
                            break
                    # Total passage width = dist_w + 1 + dist_e
                    if (dist_w + 1 + dist_e) < min_w:
                        fill_zonal[j, i] = True

                    # Meridional
                    dist_n = 0
                    for dj in range(1, min_w + 1):
                        jj = j + dj
                        if jj < n_lat_g and ocean_mask[jj, i]:
                            dist_n += 1
                        else:
                            break
                    dist_s = 0
                    for dj in range(1, min_w + 1):
                        jj = j - dj
                        if jj >= 0 and ocean_mask[jj, i]:
                            dist_s += 1
                        else:
                            break
                    if (dist_s + 1 + dist_n) < min_w:
                        fill_merid[j, i] = True

            # A cell in a narrow passage is one that's narrow in at least
            # one direction. But we only want to close actual straits, not
            # peninsulas. A narrow strait is narrow zonally OR meridionally.
            fill_cells = fill_zonal | fill_merid
            n_filled = int(np.sum(fill_cells))
            if n_filled > 0:
                ocean_mask[fill_cells] = False
                # Keep H_np unchanged — land cells retain their depth value
                # but are masked out. Setting H=0 confuses partial-cell coord.
                land_mask_init = jnp.asarray(ocean_mask.astype(np.float64))
                H_bathy_init = jnp.asarray(H_np, dtype=jnp.float64)
            print(f"  Narrow passage fill (min_width={min_w}): "
                  f"{n_filled} cells → land")

        # --- Remove small enclosed basins ---
        # After closing narrow passages and polar caps, some small bays
        # may remain connected to the open ocean only through 1-2 cells.
        # These drain over multi-year runs (no sea ice to buffer).
        # Fix: flood-fill from the largest connected ocean basin, then
        # remove any disconnected basins smaller than min_basin_size.
        from collections import deque
        H_np = np.array(H_bathy_init, copy=True)
        ocean_mask = np.array(land_mask_init, copy=True) > 0.5
        n_lat_g, n_lon_g = ocean_mask.shape
        labeled = np.zeros(ocean_mask.shape, dtype=np.int32)
        basin_id = 0
        basin_sizes = {}
        for j in range(n_lat_g):
            for i in range(n_lon_g):
                if ocean_mask[j, i] and labeled[j, i] == 0:
                    basin_id += 1
                    q = deque()
                    q.append((j, i))
                    labeled[j, i] = basin_id
                    count = 0
                    while q:
                        cj, ci = q.popleft()
                        count += 1
                        for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nj = cj + dj
                            ni = (ci + di) % n_lon_g
                            if 0 <= nj < n_lat_g and ocean_mask[nj, ni] and labeled[nj, ni] == 0:
                                labeled[nj, ni] = basin_id
                                q.append((nj, ni))
                    basin_sizes[basin_id] = count
        if basin_sizes:
            main_basin = max(basin_sizes, key=basin_sizes.get)
            n_removed = 0
            for bid, bsize in basin_sizes.items():
                if bid != main_basin:
                    small_cells = labeled == bid
                    ocean_mask[small_cells] = False
                    n_removed += int(np.sum(small_cells))
            if n_removed > 0:
                land_mask_init = jnp.asarray(ocean_mask.astype(np.float64))
                H_bathy_init = jnp.asarray(H_np, dtype=jnp.float64)
            n_basins_removed = len(basin_sizes) - 1
            print(f"  Small basin removal: {n_removed} cells in "
                  f"{n_basins_removed} disconnected basins → land "
                  f"(main basin: {basin_sizes[main_basin]} cells)")

        # Snap H_bathy to layer interfaces when the resulting partial
        # cell would be too thin.  Thin partial cells (<30% of full
        # dz_ref) at the bottom of deep equatorial columns drove the
        # day-13 PGF instability we diagnosed in the 30-day spinup —
        # f≈0 there, so geostrophy can't damp pressure-gradient errors
        # quickly, and a 67 m partial cell adjacent to a 470 m full
        # cell amplified SMC03 PGF errors enough to blow up.  The snap
        # is the standard MOM6/MITgcm fix: round H_bathy DOWN to the
        # nearest interface above (i.e., bottom moves up by one level)
        # whenever the partial cell would be thinner than the cutoff,
        # so that every column ends with a full bottom cell or a
        # "thick enough" partial cell.
        from legoesm.ocean.vertical import create_partial_cell_coordinate
        H_np = np.asarray(H_bathy_init)
        z_half_np = np.asarray(z_coord.z_half_ref)        # negative
        dz_ref_np = np.asarray(z_coord.dz_ref)            # positive
        abs_z_half = np.abs(z_half_np)                    # positive
        H_snapped = H_np.copy()
        n_snapped = 0
        thin_threshold = 0.3
        for k in range(z_coord.n_levels):
            top = abs_z_half[k]
            bot = abs_z_half[k + 1]
            in_layer = (H_np > top) & (H_np <= bot)
            partial_h = H_np - top
            too_thin = in_layer & (partial_h < thin_threshold * dz_ref_np[k])
            H_snapped = np.where(too_thin, top, H_snapped)
            n_snapped += int(np.sum(too_thin))
        # Cells where the new H_bathy is at the surface (k=0 case
        # snapped down to 0) become land.  Update land_mask consistently.
        new_land = (H_snapped <= 0.0) & (np.asarray(land_mask_init) > 0.5)
        n_new_land = int(np.sum(new_land))
        if n_new_land > 0:
            land_mask_init = jnp.where(
                jnp.asarray(new_land), 0.0, land_mask_init,
            )
        H_bathy_init = jnp.asarray(H_snapped, dtype=jnp.float64)
        print(f"  Partial-cell snap (cutoff {thin_threshold*100:.0f}%): "
              f"{n_snapped} cells snapped, {n_new_land} → land")
        z_coord_partial = create_partial_cell_coordinate(z_coord, H_bathy_init)
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
        model = LatLonCGridOceanModel(
            grid, z_coord_partial, config,
            iwm_forcing=getattr(model, "_iwm_forcing", None))
        # The scan body calls model._step_impl() (no inner JIT) so
        # partial-cell + lax.scan now works correctly.

    # Initial state: from restart, WOA, or rest.
    start_step = 0
    state = _init_rest_state(
        grid_type, grid, z_coord, args.H_max,
        H_bathy=H_bathy_init, land_mask=land_mask_init,
        bathy_cfg=bathy_cfg,
        use_etopo_postinit=(grid_type == "mpas" and args.bathymetry is not None),
    )
    # --- MPAS ETOPO post-processing (matches run_comparison_mpas.py) ---
    # The generic init_ocean_bathymetry path doesn't do north-cap masking,
    # partial-cell snapping, or partial-cell coordinate creation for MPAS.
    # Apply the proven PR 261 recipe here.
    if grid_type == "mpas" and args.bathymetry is not None:
        from legoesm.ocean.bathymetry import BathymetryConfig, load_bathymetry_mpas
        from legoesm.ocean.vertical import create_partial_cell_coordinate

        mpas_bathy_cfg = BathymetryConfig(
            source="file", path=args.bathymetry,
            H_max=args.H_max, H_min=args.H_min,
            smoothing_passes=args.smoothing_passes,
            r_factor_max=args.r_factor_max,
            depth_is_negative=True,
            fill_isolated_basins=True,
        )
        H_bathy_raw, ocean_mask = load_bathymetry_mpas(grid, mpas_bathy_cfg)

        # Optional north cap (default: 90° = full globe, no cap).
        # The comparison scripts used 80°N for parity with lat-lon;
        # for production OMIP runs the full Arctic is desired.
        north_cap_lat = getattr(args, "north_cap_lat", 90.0)
        if north_cap_lat < 90.0:
            lat_cell_deg = np.degrees(np.asarray(grid.latCell))
            cap_mask = jnp.asarray(lat_cell_deg <= north_cap_lat,
                                   dtype=H_bathy_raw.dtype)
            H_bathy_raw = H_bathy_raw * cap_mask
            ocean_mask = ocean_mask * cap_mask

        # Snap thin partial cells to nearest interface (30% threshold)
        snap_frac = 0.30
        abs_z_half = jnp.abs(z_coord.z_half_ref)
        nlev = z_coord.n_levels
        n_above = jnp.sum(
            abs_z_half[None, :] < H_bathy_raw[:, None], axis=1)
        bottom_level = jnp.clip(n_above - 1, 0, nlev - 1)
        dz_at_bottom = z_coord.dz_ref[bottom_level]
        partial_thick = H_bathy_raw - abs_z_half[bottom_level]
        frac = partial_thick / jnp.maximum(dz_at_bottom, 1e-10)
        z_upper = abs_z_half[bottom_level]
        z_lower = abs_z_half[jnp.minimum(bottom_level + 1, nlev)]
        H_snapped = jnp.where(
            H_bathy_raw - z_upper < z_lower - H_bathy_raw,
            z_upper, z_lower)
        needs_snap = (frac < snap_frac) & (frac > 0) & (H_bathy_raw > 0)
        H_bathy_final = jnp.where(needs_snap, H_snapped, H_bathy_raw)
        H_bathy_final = jnp.where(H_bathy_final <= 0, 0.0, H_bathy_final)
        ocean_mask = jnp.where(H_bathy_final > 0, ocean_mask, 0.0)

        # Isolated basin removal is handled inside load_bathymetry_mpas
        # via fill_isolated_basins=True in the BathymetryConfig.

        # Create partial cell coordinate and rebuild model
        pc_coord = create_partial_cell_coordinate(z_coord, H_bathy_final)
        z_coord = pc_coord

        from legoesm.core.field import Field
        state = state._replace(
            H_bathy=Field(data=H_bathy_final, name="H_bathy",
                          dims=("nCells",), units="m"),
            land_mask=Field(data=ocean_mask, name="land_mask",
                            dims=("nCells",), units="1"),
        )
        # Rebuild model with partial cell coordinate
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        model = MPASOceanModel(grid, pc_coord, config)

        n_ocean = int(jnp.sum(ocean_mask > 0.5))
        cap_str = f"cap={north_cap_lat}°N" if north_cap_lat < 90.0 else "no cap"
        print(f"  MPAS ETOPO: {n_ocean}/{grid.nCells} ocean cells "
              f"({cap_str}, snap={snap_frac}, "
              f"smooth={args.smoothing_passes})")

    # Tripole post-init for ETOPO bathymetry — mirrors the validated
    # run_tripole_20yr.py recipe (H_min=200, south_cap=-75, snap+partial
    # cells). Mandatory whenever JRA55 forcing is used on tripole: the
    # bipolar-fold cells have dx_v floor=1000m and would otherwise be
    # treated as ocean, violating CFL within ~3 steps under JRA55 wind.
    if grid_type == "tripole" and args.bathymetry is not None:
        from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
        from legoesm.ocean.vertical import create_partial_cell_coordinate
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )

        tri_bathy_cfg = BathymetryConfig(
            source="file", path=args.bathymetry,
            H_max=args.H_max, H_min=args.H_min,
            smoothing_passes=args.smoothing_passes,
            r_factor_max=args.r_factor_max,
            depth_is_negative=True,
            north_cap_lat=None,
            south_cap_lat=args.south_cap_lat,
        )
        H_raw, mask_raw = init_ocean_bathymetry(grid, tri_bathy_cfg)
        H_raw = jnp.asarray(H_raw, dtype=jnp.float64)
        mask_raw = jnp.asarray(mask_raw, dtype=jnp.float64)

        # Snap thin partial cells to the nearest interface (30% threshold).
        snap_frac = 0.30
        abs_z_half = jnp.abs(z_coord.z_half_ref)
        nlev = z_coord.n_levels
        Hf = H_raw.ravel()
        n_above = jnp.sum(abs_z_half[None, :] < Hf[:, None], axis=1)
        bottom_level = jnp.clip(n_above - 1, 0, nlev - 1)
        dzb = z_coord.dz_ref[bottom_level]
        partial_thick = Hf - abs_z_half[bottom_level]
        frac = partial_thick / jnp.maximum(dzb, 1e-10)
        z_upper = abs_z_half[bottom_level]
        z_lower = abs_z_half[jnp.minimum(bottom_level + 1, nlev)]
        Hs = jnp.where(Hf - z_upper < z_lower - Hf, z_upper, z_lower)
        needs_snap = (frac < snap_frac) & (frac > 0) & (Hf > 0)
        H_snap = jnp.where(needs_snap, Hs, Hf)
        H_snap = jnp.where(H_snap <= 0, 0.0, H_snap).reshape(H_raw.shape)
        mask_snap = jnp.where(H_snap > 0, mask_raw, 0.0)

        # Partial-cell coordinate + model rebuild (the eta-stretched
        # operators expect this z_coord to match the bathymetry).
        pc_coord = create_partial_cell_coordinate(z_coord, H_snap)
        z_coord = pc_coord
        model = LatLonCGridOceanModel(
            grid, z_coord, config,
            iwm_forcing=getattr(model, "_iwm_forcing", None))

        # Rebuild the rest state with the real bathymetry + mask.
        # rest_state_latlon_cgrid_ocean enforces u_mask/v_mask
        # consistency with land_mask atomically.
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, H_max=args.H_max,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H_snap,
            land_mask_override=mask_snap,
        )

        n_ocean = int(jnp.sum(mask_snap > 0.5))
        n_total = int(np.prod(np.asarray(mask_snap).shape))
        print(f"  Tripole ETOPO: {n_ocean}/{n_total} ocean cells "
              f"(H_min={args.H_min}m, south_cap={args.south_cap_lat}°, "
              f"snap={snap_frac}, smooth={args.smoothing_passes})")

    if args.woa_init and T_woa is not None and S_woa is not None:
        # Replace rest-state T/S with WOA18 climatology.
        # Keep zero velocity, zero eta — let the model adjust.
        T_woa_masked = T_woa * state.land_mask.data[..., jnp.newaxis]
        S_woa_masked = S_woa * state.land_mask.data[..., jnp.newaxis]
        state = state._replace(
            T=state.T.replace(data=T_woa_masked.astype(state.T.data.dtype)),
            S=state.S.replace(data=S_woa_masked.astype(state.S.data.dtype)),
        )
        print(f"  WOA18 initialization: T=[{float(T_woa_masked[state.land_mask.data > 0.5].min()):.1f}, "
              f"{float(T_woa_masked[state.land_mask.data > 0.5].max()):.1f}]°C, "
              f"S=[{float(S_woa_masked[state.land_mask.data > 0.5].min()):.1f}, "
              f"{float(S_woa_masked[state.land_mask.data > 0.5].max()):.1f}] PSU")
    if args.restart is not None:
        state, restart_day, restart_step = _load_restart(
            args.restart, state, grid_type=grid_type,
        )
        start_step = restart_step
        print(f"  Restart: loaded day {restart_day:.1f} (step {restart_step}) "
              f"from {Path(args.restart).name}")

    # --enable-latlon-spmd preconditions that are knowable from ARGS: refuse
    # BEFORE the JRA55 forcing setup below builds caches/state (codex r1 #1)
    # and before any device work; a negative device count would otherwise
    # silently no-op through the `_nd or len(devices)` resolution (r1 #3).
    if run_config.enable_latlon_spmd:
        if getattr(args, "jra55_sea_ice", False):
            raise SystemExit(
                "--enable-latlon-spmd does not support --jra55-sea-ice "
                "yet (the prognostic ice tile is not SPMD-audited).")
        if run_config.spmd_n_devices < 0:
            raise SystemExit(
                f"--spmd-n-devices must be >= 0 "
                f"(got {run_config.spmd_n_devices}).")

    # Forcing dispatch: 'restoring' (default) vs JRA55-do bulk fluxes.
    restoring_targets = None
    restoring_tau_s = None
    jra55_state = None

    if args.forcing_mode == "jra55_do_tropical":
        # Tropical OMIP: bulk fluxes from JRA55-do cache. The setup
        # also builds the polar sponge layer (60°S/60°N), the SSS-
        # restoring target, and the T_freeze cap region from the WOA
        # climatology and z-coordinate.
        jra55_state = _setup_jra55_forcing_state(
            args, grid, grid_type,
            z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
        )
        # Provide the ocean mask for global freeze-cap when no sponge.
        jra55_state["_ocean_mask_2d"] = state.land_mask.data > 0.5
        # GPU-interp path (default): interpolation inside the lax.scan
        # block for all grids (MPAS regridding is handled in
        # _preload_jra55_raw_records).  --no-gpu-interp routes to the
        # CPU-interp block path (_build_jra55_block_fn).
        jra55_state["_gpu_interp"] = bool(args.gpu_interp)
        # Restart: restore the prognostic sea-ice state so a checkpointed
        # --jra55-sea-ice run does NOT resume on the zero cold-start ice.  An
        # old ocean-only restart (no ice_* keys) returns None -> cold start
        # kept (no error).
        if (args.restart is not None
                and jra55_state.get("enable_sea_ice", False)):
            _ice_restored = _load_ice_restart(
                args.restart, jra55_state["ice_state_init"])
            if _ice_restored is not None:
                jra55_state["ice_state_init"] = _ice_restored
                print("  Restart: restored prognostic sea-ice state.")
        flags = []
        if jra55_state.get("enable_sponge"):
            flags.append("sponge")
        if jra55_state.get("enable_sss_restoring"):
            flags.append("SSS-restoring")
        if jra55_state.get("enable_freeze_cap"):
            flags.append("freeze-cap")
        flag_str = ", ".join(flags) or "no closure-domain features"
        print(
            f"  Forcing: jra55_do_tropical "
            f"(cache={Path(args.jra55_cache).name}, "
            f"ref_year={jra55_state['ref_year']}, {flag_str})"
        )
    elif not args.no_restoring and grid_type != "spectral":
        # SST/SSS restoring toward WOA climatology for FV grids.
        # Spectral model: restoring is unstable at coarse resolution
        # (large-scale gradients trigger exponential growth that
        # hyperdiffusion cannot suppress); rely on dynamics + diffusion.
        restoring_targets = (T_woa[..., 0], S_woa[..., 0])
        # Cubed-sphere uses a slower default restoring (3650 d
        # vs 1095 d for latlon/mpas) because the rest-state → WOA
        # gradient excites the face-edge PGF instability.  Slower
        # restoring + 14d ramp delays the blow-up but does not
        # eliminate it; production multi-year OMIP requires a different
        # grid until the cubed-sphere PGF rework lands.
        # ``args.restoring_timescale is None`` distinguishes omitted
        # from explicit (the previous ``<= 1095`` check silently
        # ignored any explicit value below 1095 on cubed_sphere,
        # which made parameter sweeps impossible).
        if args.restoring_timescale is None:
            restoring_tau_days = (
                3650.0 if grid_type == "cubed_sphere" else 1095.0
            )
        else:
            restoring_tau_days = args.restoring_timescale
        restoring_tau_s = restoring_tau_days * 86400.0
        print(f"  Restoring: tau={restoring_tau_days:.0f} days")
        print(
            "  WARNING: SST/SSS restoring toward WOA is NOT the OMIP "
            "bulk-forced protocol (Griffies et al. 2016). It is a "
            "robustness/spin-up mode whose surface relaxation overrides the "
            "air-sea flux SST adjustment OMIP is designed to evaluate. For "
            "OMIP-faithful runs use --forcing-mode jra55_do_tropical "
            "(OMIP-2 / JRA55-do) or scripts/run/run_omip_core2.py "
            "(OMIP-1 / CORE-II NCAR bulk).",
            flush=True,
        )
    elif grid_type == "spectral":
        print(f"  Restoring: disabled (spectral stability)")

    setup_time = time.time() - t_setup
    print(f"  Setup: {setup_time:.1f}s")

    # --- Save full run configuration ---
    # Every parameter that the model actually uses is recorded here,
    # including defaults.  This is the authoritative record of what
    # ran — not the CLI args (which may differ from effective values
    # due to grid-specific overrides in _create_setup).
    config_dir = Path(args.output) / grid_type / resolution
    config_dir.mkdir(parents=True, exist_ok=True)

    def _namedtuple_to_dict(obj):
        """Recursively convert NamedTuples to dicts with field names."""
        if obj is None:
            return None
        if hasattr(obj, "_fields"):
            return {
                f: _namedtuple_to_dict(getattr(obj, f))
                for f in obj._fields
            }
        if isinstance(obj, (list, tuple)):
            return [_namedtuple_to_dict(x) for x in obj]
        if callable(obj):
            return f"<callable: {getattr(obj, '__name__', str(obj))}>"
        # JAX arrays → Python scalars for JSON
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except (ValueError, AttributeError):
                return str(obj)
        return obj

    run_config_json = {
        "grid_type": grid_type,
        "resolution": resolution,
        "n_levels": int(z_coord.n_levels),
        "dt_seconds": float(dt),
        "days": float(args.days),
        "seed": int(run_config.seed),
        "forcing_mode": getattr(args, "forcing_mode", "restoring"),
        "initial_condition": "woa18" if args.woa_init else "rest_state",
        "ocean_config": _namedtuple_to_dict(config),
        "cli_args": vars(args),
    }
    config_path = config_dir / "run_config.json"
    try:
        with open(config_path, "w") as f:
            json.dump(run_config_json, f, indent=2, default=str)
        print(f"  Config saved: {config_path}")
    except Exception as e:
        print(f"  Warning: could not save config: {e}")

    # Restart cadence — only wired for the JRA55 path for now (the
    # restoring path is fast enough that re-running from scratch is
    # cheaper than maintaining restarts; revisit if needed).
    checkpoint_dir = None
    checkpoint_days = None
    if (
        (jra55_state is not None and args.checkpoint_days > 0.0)
        or run_config.max_wallclock_seconds > 0.0
    ):
        checkpoint_dir = Path(args.output) / grid_type / resolution / "restarts"
        if jra55_state is not None and args.checkpoint_days > 0.0:
            checkpoint_days = float(args.checkpoint_days)
            print(
                f"  Restart cadence: every {checkpoint_days:g} simulated days "
                f"→ {checkpoint_dir}"
            )

    # Build snapshot function for auto-plotting with each restart save.
    # Only for MPAS with the tripcolor/cartopy plotter; other grids use
    # the end-of-run timeseries plot only.
    _snapshot_fn = None
    if grid_type == "mpas":
        try:
            from scripts.plot.plot_mpas_omip_snapshot import plot_snapshot as _plot_snap
            import threading
            _snap_mesh = grid
            _snap_z = z_coord
            _snap_lock = threading.Lock()

            def _snapshot_fn(restart_path):
                """Plot snapshot in a background thread so the GPU isn't blocked.

                Uses a lock to serialize matplotlib calls (not thread-safe).
                """
                def _render():
                    # The restart write is itself async (atomic tmp+rename,
                    # see _save_restart): join the in-flight writer so the
                    # npz exists before the plotter reads it.
                    _join_restart_writer()
                    with _snap_lock:
                        try:
                            _plot_snap(restart_path, _snap_mesh, _snap_z)
                        except Exception as e:
                            print(f"    Snapshot failed: {e}", flush=True)
                t = threading.Thread(target=_render, daemon=True)
                t.start()
        except ImportError:
            pass

    # Restoring ramp: cubed_sphere defaults to 14d to delay the
    # face-edge PGF instability; other grids default to 0d.  Sentinel
    # ``None`` from argparse means "use grid default" — an explicit
    # ``--restoring-ramp-days 0`` honours the user's choice.
    if args.restoring_ramp_days is not None:
        ramp_days_eff = args.restoring_ramp_days
    elif grid_type == "cubed_sphere":
        ramp_days_eff = 14.0
    else:
        ramp_days_eff = 0.0

    # --- Lat-band SPMD (--enable-latlon-spmd): wrap the dynamics step in
    # the validated multi-GPU sharded ocean step and shard the state.
    # Single-controller only; restoring lane only (the JRA55 block
    # functions call model._step_impl directly — follow-up).  Runs AFTER
    # the restart load so a resumed state is sharded too.
    spmd_step = None
    spmd_gather = None
    spmd_shard_stack = None
    if run_config.enable_latlon_spmd:
        if grid_type != "latlon":
            raise SystemExit(
                f"--enable-latlon-spmd requires --grid latlon "
                f"(got {grid_type}).")
        if (jra55_state is not None
                and jra55_state.get("_use_single_step", False)):
            raise SystemExit(
                "--enable-latlon-spmd requires the JRA55 block-scan path, "
                "but this run selected the single-step fallback "
                "(_jra55_step calls model.step directly).")
        _multi = run_config.multicontroller
        if jax.process_count() > 1 and not _multi:
            raise SystemExit(
                "--enable-latlon-spmd is single-controller only unless "
                "--multicontroller is set; multi-node ocean scaling needs "
                "the route-B lane (jax.distributed cross-process NCCL).")
        if _multi:
            # Route-B: the mesh spans ALL global devices (one band per device
            # across every process). A strict subset would leave some
            # processes' devices out of the program (non-addressable
            # participation hazard — matches the ocean bench's guard).
            _nd = len(jax.devices())
            if run_config.spmd_n_devices and run_config.spmd_n_devices != _nd:
                raise SystemExit(
                    f"--multicontroller uses ALL global devices ({_nd} across "
                    f"{jax.process_count()} processes); --spmd-n-devices "
                    f"({run_config.spmd_n_devices}) must be 0 (auto) or {_nd}.")
        else:
            _nd = run_config.spmd_n_devices or len(jax.devices())
        if _nd > 1:
            if grid.n_lat % _nd != 0:
                raise SystemExit(
                    f"--enable-latlon-spmd: n_lat ({grid.n_lat}) not "
                    f"divisible by the device count ({_nd}); pick "
                    f"--spmd-n-devices dividing n_lat.")
            from functools import partial

            from legoesm.ocean.dynamics.sharded_ocean_step import (
                gather_state_latlon,
                make_sharded_ocean_step,
                shard_forcing_stack_latlon,
                shard_state_latlon,
            )
            from legoesm.parallel.mesh import create_latlon_mesh
            # Prime the build-once vertex-mask cache from the CONCRETE
            # state so the wrapper can build per-band masks host-side.
            model.prime_step_caches(state)
            _dev = create_latlon_mesh(n_devices=_nd)
            spmd_step = make_sharded_ocean_step(model, _dev.mesh)
            spmd_gather = partial(gather_state_latlon, mesh=_dev.mesh)
            state = shard_state_latlon(state, _dev.mesh)
            # Lay per-block forcing stacks out lat-band-sharded so the
            # in-scan interpolation / bulk fluxes stay shard-local (shared
            # layout helper — see shard_forcing_stack_latlon).
            spmd_shard_stack = partial(
                shard_forcing_stack_latlon, mesh=_dev.mesh)
            if jax.process_index() == 0:
                _lane = "route-B multicontroller" if _multi else "single-controller"
                print(f"  SPMD ({_lane}): lat-band sharded dynamics step over "
                      f"{_nd} devices across {jax.process_count()} process(es) "
                      f"({jax.default_backend()}).")
        elif jax.process_index() == 0:
            print("  SPMD: single device visible — flag is a no-op.")

    # Run time loop
    state, diag, wall_time, ok, blowup_info = _run_omip_loop(
        model, state, grid_type, grid, z_coord,
        dt, n_steps, diag_every,
        label=f"{grid_type}/{resolution}",
        restoring_targets=restoring_targets,
        restoring_tau_s=restoring_tau_s,
        restoring_ramp_days=ramp_days_eff,
        jra55_state=jra55_state,
        checkpoint_days=checkpoint_days,
        checkpoint_dir=checkpoint_dir,
        max_wallclock_seconds=run_config.max_wallclock_seconds,
        restart_buffer_seconds=run_config.restart_buffer_seconds,
        start_step=start_step,
        nudge_woa_tau=args.nudge_woa_tau,
        T_woa_3d=(T_woa * state.land_mask.data[..., jnp.newaxis]).astype(
            state.T.data.dtype) if args.nudge_woa_tau > 0 and T_woa is not None else None,
        S_woa_3d=(S_woa * state.land_mask.data[..., jnp.newaxis]).astype(
            state.S.data.dtype) if args.nudge_woa_tau > 0 and S_woa is not None else None,
        snapshot_fn=_snapshot_fn,
        spmd_step=spmd_step,
        spmd_gather=spmd_gather,
        spmd_shard_stack=spmd_shard_stack,
    )
    if spmd_gather is not None:
        # Downstream report/plot/save paths expect the full (n_lat+1)
        # staggered v layout, not the sharded v_lower carry.  Every rank
        # dispatches this gather (it is a collective); only rank 0 writes.
        state = spmd_gather(state)

    # Surface a failed FINAL async restart write while the run can still
    # report it (the writer thread swallows exceptions; _save_restart only
    # re-raises them one checkpoint later — there is no later checkpoint
    # for the last one; codex audit HIGH, 2026-07-02).  Placed AFTER the
    # spmd_gather collective above: under route-B multiproc only rank 0
    # spawns a writer thread, so a rank-0-only re-raise here must not be able
    # to skip that collective and hang the federation.
    _join_restart_writer()

    # Route-B: only rank 0 writes output files (concurrent writes to the same
    # path corrupt them); every rank still builds ``results`` so the exit code
    # is consistent across the federation.
    _io_rank = jax.process_index() == 0
    _multiproc = jax.process_count() > 1

    status = "PASS" if ok else "FAIL"
    icon = "  " if ok else "**"
    sst_str = f"SST={diag['SST'][-1]:.2f}" if diag["SST"] else ""
    # iter-97: when the run blew up, ``diag['SST'][-1]`` is the
    # last *clean* diagnostic from BEFORE the BLOWUP, which can
    # mislead the reader into thinking the run is healthy.
    # Show the BLOWUP marker explicitly.
    if blowup_info is not None:
        sst_str = f"BLOWUP at step {blowup_info['step']}"
    if _io_rank:
        print(f"\n  {icon} {status} | {grid_type}/{resolution} | "
              f"{wall_time:.1f}s | {sst_str}")

    # Save output.  Route-B: a rank-0 write failure must NOT raise past this
    # point — the loop's collectives are done, but a bare exception would give
    # rank 0 a different ALL_RESULTS / exit code than the non-root ranks (which
    # never write).  Under multiprocess, log and rebuild the results dict with
    # write=False so every rank returns the SAME record (codex r2 caveat B).
    output_dir = Path(args.output) / grid_type / resolution
    try:
        results = _save_output(
            output_dir, diag, args, grid_type, wall_time, ok,
            blowup_info=blowup_info, write=_io_rank,
        )
    except Exception as e:
        if not _multiproc:
            raise
        print(f"  WARNING: rank-0 output write failed (continuing for a "
              f"consistent federation exit code): {type(e).__name__}: {e}",
              flush=True)
        results = _save_output(
            output_dir, diag, args, grid_type, wall_time, ok,
            blowup_info=blowup_info, write=False,
        )

    # Final MLD-diagnostic snapshot (de Boyer Montegut / Treguier 2023): the
    # shared writer emits the T/S + geometry contract that
    # scripts/validate/compare_mld_dbm.py and compare_omip_nemo.py consume so
    # a finished run can be scored offline (e.g. CATKE-vs-KPP MLD).  Purely
    # additive output; a diagnostic must never abort the run.  Rank-0 only
    # (writes a file); ``state`` is already gathered/addressable on every rank.
    if _io_rank:
        try:
            from legoesm.ocean.restart import (
                save_mld_snapshot, grid_lat2d_lon2d_deg,
            )
            # Grid coords are radians; the scorers consume degrees -> convert
            # via the shared per-grid extractor (latlon/tripole/cube/mpas).
            lat2d, lon2d = grid_lat2d_lon2d_deg(grid, grid_type)
            snap = save_mld_snapshot(
                state, output_dir / "snapshot_final.npz", z_coord=z_coord,
                lat2d=lat2d, lon2d=lon2d,
                time_s=float(args.days) * 86400.0, step=int(n_steps),
            )
            print(f"  MLD snapshot: {snap}")
        except Exception as e:  # diagnostic snapshot must never crash the run
            print(f"  Warning: MLD snapshot skipped: {type(e).__name__}: {e}")

    ALL_RESULTS.append(results)
    return results


# ===========================================================================
# Summary
# ===========================================================================

def print_summary():
    """Print summary table of all runs."""
    if not ALL_RESULTS:
        return

    print(f"\n{'='*70}")
    print("  OMIP SIMULATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Grid':<15s} {'Resolution':<10s} {'Status':<8s} "
          f"{'Time (s)':<10s} {'Final SST':<10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")

    for r in ALL_RESULTS:
        # iter-97: when the run blew up, show "BLOWUP@N" instead
        # of the last-clean SST (which misled the iter-96 audit
        # into a false-improvement claim).
        blowup = r.get("blowup_info")
        if blowup is not None:
            sst = f"BLOWUP@{blowup['step']}"
        elif r["final_SST"] is not None:
            sst = f"{r['final_SST']:.2f}"
        else:
            sst = "N/A"
        print(f"  {r['grid_type']:<15s} {r['resolution']:<10s} "
              f"{r['status']:<8s} {r['wall_time_s']:<10.1f} {sst:<10s}")

    n_pass = sum(1 for r in ALL_RESULTS if r["status"] == "PASS")
    n_total = len(ALL_RESULTS)
    print(f"\n  {n_pass}/{n_total} passed")
    print(f"{'='*70}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    # Route-B multicontroller (jax.distributed cross-process NCCL): initialize
    # the federation BEFORE any device work (model build / device query), or
    # jax.distributed.initialize() would raise "must be called before any JAX
    # calls that initialise the XLA backend".  No-op unless --multicontroller.
    if getattr(args, "multicontroller", False):
        from legoesm.parallel.early_init import init_multicontroller_distributed
        init_multicontroller_distributed(getattr(args, "coordinator", None))

    # Apply the precision policy before any model state is built. Default
    # fp64 reproduces the prior unconditional behavior exactly. run_omip_single
    # re-applies it idempotently so direct callers are also covered.
    apply_run_precision(args)

    # ``--grid all`` runs every grid.  cubed_sphere is now stable on
    # the FC-Gram spectral baroclinic backend (see _create_setup) so
    # it is included in the default matrix.
    grids = GRID_TYPES if args.grid == "all" else [args.grid]

    # Route-B: every process runs main() to a consistent exit code, but only
    # rank 0 prints the banner/summary (others would duplicate the log).
    _root = jax.process_index() == 0
    if _root:
        print(f"legoESM OMIP Reference Simulation")
        print(f"  Grids: {', '.join(grids)}")
        print(f"  Days: {'30 (quick)' if args.quick else args.days}")
        print(f"  Physics: {args.physics}")
        print(f"  Precision: {args.precision}")

    for grid_type in grids:
        try:
            run_omip_single(grid_type, args)
        except Exception:
            print(f"\n  !! ERROR running {grid_type}:")
            traceback.print_exc()
            ALL_RESULTS.append({
                "grid_type": grid_type,
                "resolution": args.resolution or GRID_DEFAULTS[grid_type]["resolution"],
                "status": "ERROR",
                "wall_time_s": 0.0,
                "final_SST": None,
                "final_SSS": None,
                "final_SSH": None,
            })

    if _root:
        print_summary()

    # Exit with error if any failed (ALL ranks — the launcher needs a
    # consistent per-process exit code, so this is NOT rank-0-gated).
    if any(r["status"] != "PASS" for r in ALL_RESULTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
