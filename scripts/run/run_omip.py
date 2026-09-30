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

# Route-B multicontroller (jax.distributed): the federation MUST be up before
# any JAX call that initialises the XLA backend, and the legoESM imports below
# do exactly that at import time (legoesm.land.canopy.solver builds jnp.array
# constants when imported) -- so the ``--multicontroller`` sniff happens HERE,
# before them, not in main() (that later call is an idempotent no-op).  Same
# argv-sniff pattern as run_omip_core2's ``--fp32``.  First ORCA12 route-B
# smoke (16 GPUs) died on exactly this ordering.
def _preparse_multicontroller(argv):
    """argparse-consistent pre-parse of ONLY --multicontroller / --coordinator
    (abbreviations and ``--coordinator=host:port`` included), so the early
    federation sees exactly what ``parse_args`` will see later."""
    import argparse as _ap
    # allow_abbrev=False: the full parser also has --max-wallclock-seconds /
    # --min-passage-width, so an abbreviation such as ``--m`` would federate
    # here and then be rejected as ambiguous by parse_args.  Only the exact
    # flags federate early; an abbreviation reaches main(), which refuses
    # with the actionable message below instead of failing inside jax.
    pre = _ap.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--multicontroller", action="store_true", default=False)
    pre.add_argument("--coordinator", type=str, default=None)
    known, _ = pre.parse_known_args(argv)
    return bool(known.multicontroller), known.coordinator


_EARLY_FEDERATED = False
_pre_multi, _pre_coord = _preparse_multicontroller(sys.argv[1:])
if _pre_multi:
    from legoesm.parallel.early_init import init_multicontroller_distributed

    init_multicontroller_distributed(_pre_coord)
    _EARLY_FEDERATED = True

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
    TKEConfig,
    VerticalMixingConfig,
    tke_fesom2_card,
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
    # Voronoi/MPAS SPMD (``voronoi_spmd_ocean``): the MPAS twin of the lat-band
    # lane — reordered + padded mesh, owned-block sharding, ppermute halos.
    enable_mpas_spmd: bool = False
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
    tke_card = getattr(args, "tke_card", "default")
    if tke_card == "default":
        tke_cfg = TKEConfig()
    elif tke_card == "fesom2":
        tke_cfg = tke_fesom2_card()
    else:
        raise ValueError(f"unknown --tke-card {tke_card!r}")
    return VerticalMixingConfig(
        scheme=scheme,
        tke=tke_cfg,
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
        enable_mpas_spmd=getattr(args, "enable_mpas_spmd", False),
        spmd_n_devices=getattr(args, "spmd_n_devices", 0),
        multicontroller=getattr(args, "multicontroller", False),
        coordinator=getattr(args, "coordinator", None),
    )


def with_kpp_cfl_dt(run_config: OMIPRunConfig, dt: float) -> OMIPRunConfig:
    """Tie the KPP explicit-diffusion CFL cap to the run's timestep.

    ``KPPConfig.cfl_cap_dt_s`` must equal the ocean dynamics dt (its default
    300 s matched only the MPAS default dt), so the driver passes the value it
    knows instead of leaving a default that disagrees with ``--dt``.  A value
    already moved off the default (set explicitly upstream) that disagrees
    with ``dt`` raises instead of being silently overwritten.
    """
    vm = run_config.vertical_mixing
    current = float(vm.kpp.cfl_cap_dt_s)
    default = float(KPPConfig._field_defaults["cfl_cap_dt_s"])
    if current != default and current != float(dt):
        raise ValueError(
            f"KPPConfig.cfl_cap_dt_s was set to {current} s but the run "
            f"timestep is {float(dt)} s; the KPP CFL cap must use the run dt")
    return run_config._replace(vertical_mixing=vm._replace(
        kpp=vm.kpp._replace(cfl_cap_dt_s=float(dt))))


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
    p.add_argument("--dz-ref-file", type=str, default=None,
                   help=(
                       "Path to a 1-D list of layer thicknesses [m] "
                       "(whitespace/newline separated text, or .npy) that "
                       "REPLACES the tanh-stretched z* profile with an EXACT "
                       "external vertical grid — e.g. FESOM's CORE2 47 levels "
                       "or NEMO's e3t_1d. Its length must equal --nlev. "
                       "Without it the level COUNT can be matched but not the "
                       "level PLACEMENT."
                   ))
    p.add_argument("--H-max", type=float, default=5500.0)
    p.add_argument("--dt", type=float, default=None,
                   help="Timestep [s] (default: grid-specific)")
    p.add_argument("--days", type=float, default=365.0)
    p.add_argument("--quick", action="store_true",
                   help="Short 30-day run for CI")
    p.add_argument("--output", type=str, default="results/omip")
    p.add_argument("--checkpoint-days", type=float, default=30.0)
    p.add_argument("--no-final-snapshot", action="store_true",
                   help="Skip the end-of-run MLD snapshot (snapshot_final.npz). "
                        "For probe arms at ORCA12 the 14 GB compressed write "
                        "outlasts the walltime; with --checkpoint-days 0 the "
                        "arm then writes only its csv/json diagnostics.")
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
    p.add_argument("--ic-from-fesom-mesh", type=str, default=None,
                   help=("Directory of a FESOM2-JAX mesh with a cached initial "
                         "field (T_ic.npy/S_ic.npy, geo_coord_nod2D.npy, Z.npy, "
                         "nlevels_nod2D.npy). Replaces the WOA/PHC files as the "
                         "source of --woa-init (and of the sponge / restoring "
                         "targets): every lane then starts from the reference "
                         "run's own field (init_woa.init_ocean_from_fesom_mesh)."))
    p.add_argument("--ic-cache-dir", type=str, default=None,
                   help=("Absolute directory on SHARED storage where the field built by "
                         "--ic-from-fesom-mesh is cached (rank 0 builds once, every rank "
                         "loads the same bytes). Default: $LEGOESM_MESH_CACHE_DIR."))
    p.add_argument("--woa-void-fill", action="store_true",
                   help=("Harmonic-fill source OCEAN cells the observed T/S "
                         "never sampled at a depth (nearest donor farther than "
                         "one source grid step) instead of stitching them from "
                         "the nearest observed cells, which builds density "
                         "walls (init_woa._fill_source_levels_nearest_valid). "
                         "Shapes the same WOA arrays the SSS restoring, "
                         "--nudge-woa-tau and the sponge targets read, with or "
                         "without --woa-init. Default: nearest-donor stitch."))
    p.add_argument("--nudge-woa-tau", type=float, default=0.0,
                   help="Nudge T toward WOA18 with this restoring timescale [days]. "
                        "Applied after each block step. 0=disabled. "
                        "Typical 90-180 days for gentle spinup.")
    p.add_argument("--tripole-mesh", type=str, default=None,
                   help=(
                       "Override the --grid tripole mesh file(s) chosen by "
                       "--resolution: one NEMO mesh_mask/domain_cfg path, or "
                       "os.pathsep-joined paths of a split mesh "
                       "(mesh_hgr.nc:mesh_zgr.nc:mask.nc). The land mask and "
                       "bathymetry of the tripole lane ALWAYS come from the "
                       "mesh (tmaskutil, sum_k e3t_0*tmask)."))
    p.add_argument("--tripole-fold-convention", type=str, default="auto",
                   choices=["auto", "n_lon-1-i", "(n_lon-i)%n_lon"],
                   help=(
                       "North-fold index convention for a --tripole-mesh "
                       "override (the registry meshes carry their validated "
                       "convention). 'auto' picks the self-symmetric one and "
                       "REFUSES a tie; pass it explicitly for a pivot row whose "
                       "latitude is too flat to disambiguate (ORCA12 stripped: "
                       "'(n_lon-i)%%n_lon')."))
    p.add_argument("--no-tripole-partial-cells", action="store_false",
                   dest="tripole_partial_cells", default=True,
                   help=("Keep the legacy z* bottom on the --grid tripole lane. "
                         "DEFAULT is partial cells: z* stretches the WHOLE "
                         "reference column into every water column, so a NEMO "
                         "75-level grid gives the shallowest shelf column a "
                         "4 mm surface layer and one step of surface heating "
                         "sends it past 1000 C (measured, eORCA1 72.2N 73.6E). "
                         "Partial cells keep the reference layer thicknesses "
                         "and cut only the bottom cell, as NEMO does."))
    p.add_argument("--tripole-closed-seas", type=str, default=None,
                   help=("Comma-separated enclosed seas to mask as land on the "
                         "--grid tripole lane (names in "
                         "legoesm.ocean.init_tripole.CLOSED_SEAS, e.g. "
                         "marmara,black_sea): basins the observed IC cannot "
                         "fill.  Default: none masked."))
    p.add_argument("--tripole-strip-north-rows", type=int, default=0,
                   help=(
                       "Drop this many DEAD halo rows from the north end of "
                       "the tripole mesh before building the grid (NEMO "
                       "jperio=4 T-pivot meshes such as ORCA0083/ORCA12 end "
                       "with one halo row above the self-dual pivot row; the "
                       "operators fold the LAST row onto itself). Default 0 = "
                       "mesh as stored (eORCA1.2 / eORCA025)."))
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
                   help=(
                       "Latitude [°N] above which all cells become land. "
                       "DEFAULT 90 = NO CAP: the North Pole stays open. On a "
                       "global regular lat-lon grid the meridians converge "
                       "there (dx = R·dlon·cos(lat) ≈ 970 m at 89.5°N), so an "
                       "uncapped pole is a CFL trap — a rest-state OMIP run at "
                       "dt=2400 s reached |u| = 19 m/s and advective CFL 47 at "
                       "89.5°N within 3 steps. Capping at 80 removed it "
                       "(|u|max 0.014 m/s). Cap unless you know the polar "
                       "dynamics are handled. (The help here previously "
                       "claimed 'default 80', which was wrong.)"
                   ))
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
    p.add_argument("--B-h-gamma0", type=float, default=None, dest="B_h_gamma0",
                   help=("Tripole lane: FESOM2's resolution-scaled biharmonic "
                         "B = gamma0*h^3 on the local face size (fesom_jax gamma0 = "
                         "0.003); requires --A-h 0 --C-smag-lap 0 and replaces "
                         "--B-h (2-D metrics only)."))
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
    p.add_argument("--C-smag-lap", type=float, default=None,
                   help="Laplacian Smagorinsky coefficient (dimensionless). Unset "
                        "keeps each lane's own value: 0.15 on the lat-lon "
                        "bathymetry branch, 0.33 in the MPAS / tripole NEMO-match "
                        "recipes.")
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
    p.add_argument("--sw-penetration", type=str, default="auto",
                   choices=["auto", "jerlov_2band", "sweeney_2band"],
                   help=("Column shortwave scheme on the JRA55 MPAS/tripole lanes: "
                         "'auto' = the lanes' historical two-band type-II with the "
                         "0.94 skin split (byte-identical); 'jerlov_2band' = two-band "
                         "with --water-type; 'sweeney_2band' = FESOM2's chlorophyll "
                         "two-band (needs --chl-clim; 0.54 of the NET shortwave "
                         "penetrates). Explicit schemes hand the forcing the "
                         "post-albedo shortwave, as FESOM2 does; sweeney_2band "
                         "also deposits the surface part on the live top thickness."))
    p.add_argument("--chl-clim", type=str, default=None,
                   help=("Monthly surface chlorophyll climatology (Sweeney 2005 "
                         "NetCDF, variable 'chl', 12 x 180 x 360 on the forcing-cache "
                         "grid) for --sw-penetration sweeney_2band."))
    p.add_argument("--ocean-albedo", type=float, default=0.06,
                   help=("Open-water shortwave albedo of the bulk-flux coupler "
                         "(default 0.06; FESOM2/CORE2 uses 0.1)."))
    p.add_argument("--tke-card", type=str, default="default",
                   choices=("default", "fesom2"),
                   help="TKE closure constant set: 'default' (Veros/legoESM "
                        "defaults, diagnostic TKE) or 'fesom2' (prognostic TKE "
                        "carried on the state, with the FESOM2-JAX FORCA20 "
                        "constants: Pr=clamp(6.6 Ri,1,10), no Bryan-Lewis "
                        "floor, Av/Kv 1e-4/1e-5, surface flux coeff 3.75).")
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
    p.add_argument("--mpas-k-zeta-bih", type=float, default=None,
                   dest="mpas_k_zeta_bih",
                   help="Pin the MPAS biharmonic vorticity damping coefficient "
                        "[m^4/s] (0 = term off). DEFAULT (flag absent): derived "
                        "from the mesh's own mean cell spacing as "
                        "K_ref*(dx/dx_ref)^3, anchored on the ico6 mesh the "
                        "OMIP NEMO-match value (1e14) was tuned on -- so ico6 "
                        "is unchanged and finer meshes get the scaled value.")
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
    p.add_argument("--mpas-lloyd", type=int, default=50,
                   help=(
                       "Lloyd (SCVT centroidal relaxation) iterations for the "
                       "--grid mpas mesh (default 50 = the production SCVT, "
                       "unchanged). The value is part of the mesh-cache key, "
                       "so 0 selects the scaling campaign's cached lloyd=0 "
                       "icosahedral meshes (cell quality slightly lower)."
                   ))
    p.add_argument("--enable-mpas-spmd", action="store_true", default=False,
                   help=(
                       "Run the MPAS lane multi-device SPMD "
                       "(legoesm.parallel.voronoi_spmd_ocean): the mesh is "
                       "reordered + padded for the device count, the state is "
                       "sharded in owned cell/edge blocks and every step "
                       "fills owned+halo local buffers with coloured "
                       "ppermute rounds, running MPASOceanModel._step_impl on "
                       "each device's local mesh with in-step halo refreshes "
                       "and psum reductions. Requires --grid mpas. Supports "
                       "the restoring lane and both JRA55 block-scan lanes "
                       "(host regrid and GPU-interp); --jra55-sea-ice only with "
                       "--ice-dynamics none and --ice-categories 1 (the "
                       "thermodynamic tile is pointwise). normalize_freshwater "
                       "is supported (owned-masked psum means). Refused: "
                       "ice rheology, "
                       "use_baroclinic_rho_ref, runoff_depth_spread_map. "
                       "Restarts carry the device count and are refused "
                       "across a different one (cell order differs)."
                   ))
    p.add_argument("--spmd-n-devices", type=int, default=0,
                   help=(
                       "Device count for --enable-latlon-spmd / "
                       "--enable-mpas-spmd (0 = all local devices)."
                   ))
    p.add_argument("--multicontroller", action="store_true", default=False,
                   help=(
                       "Promote --enable-latlon-spmd / --enable-mpas-spmd to "
                       "ROUTE-B (jax.distributed, cross-process NCCL): the "
                       "ocean device mesh spans ALL global devices, one band "
                       "(lat-lon) or one owned cell block (MPAS) per device "
                       "across every process — the multi-node OMIP lane. "
                       "Single-controller (one process, local devices) "
                       "is the default when this is off. Requires --grid "
                       "latlon or mpas. Launch under SLURM/mpiexec with one process "
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
    # River-runoff routing.  Default "none" reproduces the historical
    # behaviour bit-for-bit: runoff is delivered where the product puts it and
    # the model's dry-cell masking DISCARDS whatever landed on land — ~71 % of
    # the JRA55-do global total at 0.5625 deg.  FESOM2's counterpart is
    # use_runoff_mapper / runoff_radius (500 km).
    p.add_argument("--runoff-routing", type=str, default="none",
                   choices=["none", "nearest", "spread"],
                   help=(
                       "Route river runoff off dry cells onto wet ones, "
                       "conserving total discharge. 'none' (default) = "
                       "historical behaviour, land-cell runoff is lost. "
                       "'nearest' = each dry cell's discharge goes to its "
                       "closest wet cell within --runoff-radius-km. "
                       "'spread' = divided over every wet cell in range "
                       "(area-weighted), closer to FESOM's mapper and avoids "
                       "a salinity crater under a big river."
                   ))
    p.add_argument("--runoff-source", type=str, default="jra55_friver",
                   choices=["jra55_friver", "core2_climatology"],
                   help=("River runoff: the JRA55-do 'friver' records (default) or "
                         "FESOM2's constant CORE2 climatology (--runoff-file), "
                         "which replaces every record on the model grid."))
    p.add_argument("--runoff-file", type=str, default=None,
                   help="CORE2_runoff.nc for --runoff-source core2_climatology.")
    p.add_argument("--runoff-radius-km", type=float, default=500.0,
                   help=("Search radius [km] for --runoff-routing "
                         "(default 500, matching FESOM2's runoff_radius)."))
    # Sea-ice rheology for the --jra55-sea-ice tile.  Before these existed the
    # lane hard-coded SeaIceConfig(), i.e. dynamics="none" / n_categories=1 --
    # a thermodynamic slab with diagnostic free drift and NO rheology, while
    # every OMIP-2 reference model (FESOM2: EVP, 120 subcycles, P*=30000)
    # runs one.  Defaults below reproduce the old slab bit-for-bit.
    p.add_argument("--ice-dynamics", type=str, default="none",
                   choices=["none", "free_drift", "evp", "mevp"],
                   help=(
                       "Sea-ice momentum solver for --jra55-sea-ice. "
                       "'none' (default) = thermodynamic slab with diagnostic "
                       "free drift, the historical behaviour. 'evp' = "
                       "Hunke-Dukowicz 1997 (FESOM2's whichEVP=0), 'mevp' = "
                       "Bouillon 2013 / Kimmritz 2015."
                   ))
    p.add_argument("--ice-categories", type=int, default=1,
                   help=(
                       "Sea-ice thickness categories for --jra55-sea-ice "
                       "(default 1). CICE-standard bounds exist for 1/3/5/7."
                   ))
    p.add_argument("--ice-n-evp", type=int, default=None,
                   help="EVP subcycles per ice step (SeaIceConfig.N_evp, "
                        "default 120; FESOM2 evp_rheol_steps = 120).")
    p.add_argument("--ice-p-star", type=float, default=None,
                   help="Ice strength parameter P* [N/m^2] "
                        "(default 2.75e4; FESOM2 Pstar = 3.0e4).")
    p.add_argument("--ice-e-yield", type=float, default=None,
                   help="Yield-curve eccentricity (default 2.0).")
    p.add_argument("--ice-c-strength", type=float, default=None,
                   help="Strength decay constant C (default 20.0).")
    p.add_argument("--ice-delta-min", type=float, default=None,
                   help="Minimum deformation rate [1/s] (default 2.0e-9; "
                        "FESOM2 delta_min = 1.0e-11).")
    p.add_argument("--ice-alpha-mevp", type=float, default=None,
                   help="mEVP stress relaxation alpha (default 500; "
                        "FESOM2 alpha_evp = 250).")
    p.add_argument("--ice-beta-mevp", type=float, default=None,
                   help="mEVP velocity relaxation beta (default 500; "
                        "FESOM2 beta_evp = 250).")
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
    p.add_argument("--sss-restoring-target", type=str, default="woa_winter",
                   choices=("woa_winter", "phc2_monthly"),
                   help="Target field of the Haney SSS restoring: the PHC3 "
                        "winter surface slice of the initial condition "
                        "(default) or the monthly PHC2 climatology FESOM2 "
                        "restores to, the month picked by the model calendar day.")
    p.add_argument("--sss-target-file", type=str,
                   default="/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/PHC2_salx.nc",
                   help="PHC2 monthly SSS file (--sss-restoring-target phc2_monthly).")
    p.add_argument("--sss-restoring-remove-mean", action="store_true",
                   help="Subtract the area-weighted global mean of the restoring "
                        "tendency each step, as FESOM2 does, so restoring adds no net salt.")
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
    p.add_argument("--jra55-zero-surface-fluxes", action="store_true",
                   help="PROBE: zero the surface forcing handed to the dynamics "
                        "step (stress, heat, freshwater) after the bulk fluxes / "
                        "ice partition; sponge, SSS restoring, freeze cap and "
                        "the ice tile itself are untouched. Isolates a blowup "
                        "from the forcing.")
    p.add_argument("--frazil", action=argparse.BooleanOptionalAction, default=False,
                   help="Frazil-ice closure on every supercooled ocean level after "
                        "the dynamics step (ocean.physics.frazil): liquid mass, salt "
                        "and cp*T enthalpy conserved, the ice exported into the slab "
                        "sea-ice tile. Requires --jra55-sea-ice (single-category slab) "
                        "and the virtual-salt-flux freshwater closure. Default off.")
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

def _build_runoff_map_for_run(args, grid, grid_type, ocean_mask):
    """Assemble the static river-runoff routing plan, or ``None``.

    Runoff products place discharge at river mouths resolved on THEIR grid; a
    coarser ocean calls many of those cells land and its dry-cell masking then
    discards the water instead of delivering it (measured at ~71 % of the
    JRA55-do global total).  FESOM2 solves this with ``use_runoff_mapper`` /
    ``runoff_radius``; this is the legoESM counterpart.

    Returns ``None`` for ``--runoff-routing none`` (the default), which keeps
    the historical behaviour bit-identical.
    """
    scheme = getattr(args, "runoff_routing", "none")
    if scheme == "none":
        return None

    from legoesm.ocean.forcing.runoff_mapper import build_runoff_map

    mask = np.asarray(ocean_mask)
    if grid_type == "latlon":
        # The grid's OWN metric, not a re-derived one, so the routing budget
        # uses the same areas every other diagnostic does.
        area = np.asarray(grid.area)
        lat_c = np.asarray(grid.lat2d)
        lon_c = np.asarray(grid.lon2d)
    elif grid_type == "mpas":
        # The MPAS lane regrids friver with k=4 inverse-distance weighting
        # (compute_latlon_to_voronoi_weights), which does NOT conserve
        # sum(F*A) for a flux density.  Routing on top of that would conserve
        # an already-wrong discharge and hand back a budget that looks closed.
        # Refuse until friver gets a conservative lat-lon -> MPAS remap.
        raise SystemExit(
            "--runoff-routing is not available on --grid mpas: friver reaches "
            "the MPAS grid through a k=4 IDW scalar interpolation, which does "
            "not conserve total discharge, so routing would conserve the "
            "wrong number. Needs a conservative remap for friver first "
            "(conservative_regrid_unstructured). Use --runoff-routing none, "
            "or --grid latlon."
        )
    else:
        raise SystemExit(
            f"--runoff-routing {scheme!r} is not wired for --grid "
            f"{grid_type!r} (needs cell areas + centres); use latlon or mpas, "
            "or --runoff-routing none."
        )
    if area.shape != mask.shape:
        raise SystemExit(
            f"--runoff-routing: cell-area shape {area.shape} does not match "
            f"the ocean mask {mask.shape}."
        )

    # Build with a REAL record so ``unrouted_fraction`` is the discharge this
    # run actually fails to place, not the 0.0 a missing reference returns.
    ref = _mean_runoff_record(args)
    rmap = build_runoff_map(
        mask, area, lat_c, lon_c,
        radius_m=float(args.runoff_radius_km) * 1.0e3,
        scheme=scheme, reference_runoff=ref,
    )
    n_moved = int(np.asarray(rmap.src_idx).size)
    print(f"  Runoff routing: scheme={scheme}, "
          f"radius={args.runoff_radius_km:g} km, {n_moved} donor-recipient "
          f"pairs over {int((~mask).sum())} dry cells")
    if ref is not None:
        total = float(np.sum(ref * area))
        if total > 0.0:
            on_land = float(np.sum(ref[~mask] * area[~mask]))
            print(f"  Runoff budget (cache time-mean): {total:.4e} kg/s "
                  f"total, {100.0 * on_land / total:.2f} % on dry cells "
                  f"before routing, {100.0 * rmap.unrouted_fraction:.2f} % "
                  f"still unrouted after (no wet cell within "
                  f"{args.runoff_radius_km:g} km)")
        else:
            print("  Runoff budget: cache time-mean discharge is zero; "
                  "nothing to route.")
    return rmap


def _mean_runoff_record(args):
    """TIME-MEAN ``friver`` over the whole JRA55-do cache, or ``None``.

    Used only to report how much discharge the routing radius fails to place;
    never fed to the model.  The mean rather than a single record because the
    loss is seasonal -- 1 January 1958 integrates to 9.35e8 kg/s against an
    annual mean of 1.35e9, so a January reference understates the run's loss
    by a third.  Streamed in chunks so a multi-decade cache does not have to
    fit in memory.

    A cache that exists but cannot be read is a hard error: silently skipping
    the diagnostic is how a malformed cache reaches the integration unnoticed.
    """
    cache = getattr(args, "jra55_cache", None)
    if cache is None or not Path(cache).exists():
        return None
    import xarray as xr
    ds = xr.open_zarr(str(cache), decode_times=False)
    try:
        n_rec = int(ds.sizes["time"])
        acc = np.zeros(
            np.asarray(ds["friver"].isel(time=0).values).shape,
            dtype=np.float64)
        chunk = 200
        for i0 in range(0, n_rec, chunk):
            acc += np.asarray(
                ds["friver"].isel(time=slice(i0, min(i0 + chunk, n_rec)
                                             )).values,
                dtype=np.float64).sum(axis=0)
    finally:
        ds.close()
    return acc / n_rec


def _load_core2_runoff_static(path, regrid_weights, cache_lat, cache_lon) -> np.ndarray:
    """FESOM2's CORE2 climatological runoff regridded to the model grid.

    Reads ``Foxx_o_roff`` (shape (1, nlat, nlon), units (kg/s)/m^2, the same
    kg m-2 s-1 as the JRA55 ``friver``; missing_value 1e30 over land).  Masked
    / NaN / missing entries are ZERO runoff (land), the field is regridded on
    the host (deterministic across processes) and the single record is
    returned with shape ``tuple(regrid_weights.target_shape)`` as float64.
    """
    import netCDF4

    cache_lat = np.asarray(cache_lat, dtype=np.float64)
    cache_lon = np.asarray(cache_lon, dtype=np.float64)
    with netCDF4.Dataset(path) as ds:
        file_lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        file_lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        if file_lat.shape != cache_lat.shape or not np.allclose(file_lat, cache_lat):
            raise ValueError(
                f"{path}: CORE2 runoff 'lat' axis does not match the forcing-cache "
                f"grid (file {file_lat.shape} vs cache {cache_lat.shape})")
        if file_lon.shape != cache_lon.shape or not np.allclose(file_lon, cache_lon):
            raise ValueError(
                f"{path}: CORE2 runoff 'lon' axis does not match the forcing-cache "
                f"grid (file {file_lon.shape} vs cache {cache_lon.shape})")
        raw = ds.variables["Foxx_o_roff"][:]
    data = np.asarray(np.ma.filled(np.ma.masked_invalid(raw), 0.0), dtype=np.float64)
    # unmasked sentinels (1e30, incl. float32 round-trips) -> 0.0
    data = np.where(np.isnan(data) | np.isclose(data, 1.0e30, rtol=1e-6), 0.0, data)
    records = data.reshape(1, file_lat.size, file_lon.size)
    regridded = np.asarray(_regrid_records_host(records, regrid_weights))
    return np.asarray(regridded[0], dtype=np.float64).reshape(
        tuple(regrid_weights.target_shape))


def _route_runoff_stack(runoff_stack, rmap, static=None):
    """Apply the routing plan to a stacked runoff field, or pass it through.

    Routing is LINEAR in the runoff field and the plan is static, so routing
    the RECORDS once here is identical to routing every interpolated step
    inside the scan — and far cheaper.  When ``static`` is not None (a field
    already on the model grid, kg/m^2/s: FESOM2's CORE2 climatological
    runoff) the JRA55 per-record runoff is REPLACED by its per-record
    broadcast before the plan is applied.
    """
    if static is not None:
        if tuple(static.shape) != tuple(runoff_stack.shape[1:]):
            raise ValueError(
                f"static runoff field shape {tuple(static.shape)} does not match "
                f"the runoff stack record shape {tuple(runoff_stack.shape[1:])}")
        runoff_stack = jnp.broadcast_to(static, runoff_stack.shape)
    if rmap is None:
        return runoff_stack
    from legoesm.ocean.forcing.runoff_mapper import apply_runoff_map
    return jax.vmap(lambda r: apply_runoff_map(r, rmap))(runoff_stack)


def load_dz_ref_file(path: str | None):
    """Load a 1-D layer-thickness profile [m] for ``--dz-ref-file``.

    Accepts a ``.npy`` array or any whitespace/newline-separated text file
    (``#`` comments allowed).  Returns ``None`` for ``path is None`` so the
    caller keeps the tanh-stretched default.

    Validated here rather than deep in ``_create_setup`` so a bad file fails
    before any device work: thicknesses must be finite and strictly positive,
    or the z* coordinate they build is not monotonic and every depth-indexed
    diagnostic downstream is quietly wrong.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--dz-ref-file not found: {p}")
    if p.suffix == ".npy":
        dz = np.load(p)
    else:
        dz = np.loadtxt(p, comments="#")
    dz = np.asarray(dz, dtype=np.float64)
    if dz.ndim != 1:
        raise SystemExit(
            f"--dz-ref-file {p} must hold a 1-D list of thicknesses; got "
            f"shape {dz.shape}. Ravelling a 2-D file would silently invent a "
            "vertical grid from whatever order it happened to be stored in."
        )
    if dz.size == 0:
        raise SystemExit(f"--dz-ref-file {p} is empty.")
    if not np.all(np.isfinite(dz)):
        raise SystemExit(f"--dz-ref-file {p} has non-finite thicknesses.")
    if not np.all(dz > 0.0):
        raise SystemExit(
            f"--dz-ref-file {p} has non-positive thicknesses "
            f"(min {dz.min():.6g}); layer thicknesses must be > 0."
        )
    return dz


def _validate_dz_ref_against_setup(dz, nlev: int, H_max: float) -> None:
    """Refuse an external vertical grid that disagrees with ``--nlev`` /
    ``--H-max``.

    ``create_z_star_from_thicknesses`` makes the column exactly ``sum(dz)``
    deep, but the BATHYMETRY and rest state are still built from
    ``args.H_max``.  If the two disagree, every full-depth column is silently
    rescaled and the grid is no longer the "exact" external one that was
    asked for -- a 6000 m level list against the 5500 m default would look
    fine and be wrong.  Checking here costs nothing and the failure is
    otherwise invisible.
    """
    if dz is None:
        return
    if dz.size != nlev:
        raise SystemExit(
            f"--dz-ref-file has {dz.size} levels but --nlev is {nlev}; "
            f"pass --nlev {dz.size}."
        )
    total = float(dz.sum())
    if abs(total - float(H_max)) > 1e-6 * max(1.0, abs(float(H_max))):
        raise SystemExit(
            f"--dz-ref-file sums to {total:.6f} m but --H-max is "
            f"{float(H_max):.6f} m. The bathymetry and rest state are built "
            f"from --H-max while the vertical coordinate is built from the "
            f"file, so a mismatch silently rescales every full-depth column. "
            f"Pass --H-max {total:.6f}."
        )


_ACTIVE_BUILD_CTX = None


def _exit_build_ctx() -> None:
    """Leave the #1370 host-side build context if one is active.  Called at the
    normal end of the setup section AND from main()'s per-grid handlers for
    BOTH Exception and SystemExit, so neither a caught setup error nor a
    refused configuration can leave the CPU as the default device for the next
    grid case.  Idempotent.

    A caller that invokes ``run_omip_single`` directly (not through ``main``)
    owns the same duty: the setup section raises SystemExit on a refused
    configuration from INSIDE the context, so such a caller should call this
    in its own finally."""
    global _ACTIVE_BUILD_CTX
    ctx = _ACTIVE_BUILD_CTX
    _ACTIVE_BUILD_CTX = None
    if ctx is not None:
        ctx.__exit__(None, None, None)


def _memprobe(tag: str) -> None:
    """PROBE (env LEGOESM_OMIP_MEMPROBE=1): per local device, bytes in use /
    peak (includes XLA's compile + scratch workspace) and the five largest
    live arrays by per-device bytes.  Prints on every rank."""
    if not os.environ.get("LEGOESM_OMIP_MEMPROBE"):
        return
    rank = jax.process_index()
    for d in jax.local_devices():
        ms = d.memory_stats() or {}
        use = ms.get("bytes_in_use", 0) / 2**30
        peak = ms.get("peak_bytes_in_use", 0) / 2**30
        live = []
        for a in jax.live_arrays():
            try:
                nb = sum(sh.data.nbytes for sh in a.addressable_shards
                         if sh.device == d)
            except Exception:  # deleted / non-addressable while iterating
                continue
            if nb:
                live.append((nb, tuple(a.shape), str(a.dtype),
                             "sharded" if not a.sharding.is_fully_replicated
                             else "replicated"))
        live.sort(reverse=True)
        top = "; ".join(f"{nb/2**30:.2f}GiB {shp} {dt} {kind}"
                        for nb, shp, dt, kind in live[:5])
        print(f"  [rank {rank}] MEMPROBE {tag} {d}: in_use={use:.2f}GiB "
              f"peak={peak:.2f}GiB live_total="
              f"{sum(x[0] for x in live)/2**30:.2f}GiB top5: {top}", flush=True)


def _report_nonfinite(state, grid_type, grid, spmd_gather=None) -> None:
    """On BLOWUP: say WHERE (row/col/level, lat/lon, lat band, depth) the first
    non-finite ocean cell sits, how many rows carry them, and the T/eta ranges
    the bound check saw.  Rank 0 prints; the gather is a collective."""
    if grid_type == "spectral":
        return
    st = spmd_gather(state) if spmd_gather is not None else state
    T = np.asarray(st.T.data)
    eta = np.asarray(st.eta.data)
    mask = np.asarray(st.land_mask.data) > 0.5
    if T.ndim == 3 and mask.ndim == 2:
        T = np.where(mask[..., None], T, 0.0)
        eta = np.where(mask, eta, 0.0)
    if jax.process_index() != 0:
        return
    bad = np.argwhere(~np.isfinite(T))
    n_bands = jax.device_count() if spmd_gather is not None else 1
    n_rows = T.shape[0]
    msg = [f"  BLOWUP detail: nonfinite T cells={len(bad)} "
           f"nonfinite eta cells={int((~np.isfinite(eta)).sum())}"]
    if len(bad):
        rows = np.unique(bad[:, 0])
        j, i = int(bad[0][0]), int(bad[0][1])
        k = int(bad[0][2]) if bad.shape[1] > 2 else -1
        band = j * n_bands // n_rows
        loc = f"first (row={j}, col={i}, lev={k}) band {band}/{n_bands}"
        lat = getattr(grid, "lat_T", None)
        lon = getattr(grid, "lon_T", None)
        if lat is not None and np.ndim(lat) == 2:
            la, lo = float(np.asarray(lat)[j, i]), float(np.asarray(lon)[j, i])
            if abs(la) <= np.pi + 1e-6 and np.abs(np.asarray(lat)).max() <= np.pi + 1e-6:
                la, lo = np.degrees(la), np.degrees(lo)
            loc += f" lat={la:.2f} lon={lo:.2f}"
        H = getattr(st, "H_bathy", None)
        if H is not None:
            loc += f" depth={float(np.asarray(H.data)[j, i]):.1f}m"
        msg.append(f"    {loc}; rows with nonfinite: {rows.min()}..{rows.max()} "
                   f"(n={len(rows)} of {n_rows}); nonfinite per band: "
                   + str(np.bincount(bad[:, 0] * n_bands // n_rows,
                                     minlength=n_bands).tolist()))
    lat = getattr(grid, "lat_T", None)
    lon = getattr(grid, "lon_T", None)
    H = getattr(st, "H_bathy", None)

    def _where(j, i):
        out = f"row={j} col={i} band {j * n_bands // n_rows}/{n_bands}"
        if lat is not None and np.ndim(lat) == 2:
            la = np.asarray(lat)[j, i]
            lo = np.asarray(lon)[j, i]
            if np.abs(np.asarray(lat)).max() <= np.pi + 1e-6:
                la, lo = np.degrees(la), np.degrees(lo)
            out += f" lat={float(la):.2f} lon={float(lo):.2f}"
        if H is not None:
            out += f" depth={float(np.asarray(H.data)[j, i]):.1f}m"
        return out

    finT = np.where(np.isfinite(T), T, 0.0)
    for name, idx in (("T max", np.argmax(finT)), ("T min", np.argmin(finT))):
        c = np.unravel_index(idx, T.shape)
        msg.append(f"    {name}={float(T[c]):.4g} at {_where(int(c[0]), int(c[1]))} "
                   f"lev={int(c[2]) if len(c) > 2 else -1}")
    Tf = T[np.isfinite(T)]
    ef = eta[np.isfinite(eta)]
    if Tf.size and ef.size:
        jm = np.unravel_index(np.argmax(np.where(np.isfinite(eta), np.abs(eta), 0)),
                              eta.shape)
        msg.append(f"    finite ranges: T [{Tf.min():.3g}, {Tf.max():.3g}] "
                   f"eta [{ef.min():.3g}, {ef.max():.3g}] max|eta| at row={jm[0]} "
                   f"col={jm[1]} band {int(jm[0]) * n_bands // n_rows}")
    print("\n".join(msg), flush=True)


def _report_forcing_extremes(stack, tag: str) -> None:
    """On BLOWUP: per rank, the min/max of every forcing leaf's LOCAL shard.
    A regridded field carrying a fill value or an out-of-range value shows up
    here as the leaf and the band it sits in, before any physics is blamed."""
    if not isinstance(stack, dict):
        return
    rank = jax.process_index()
    parts = []
    for k in sorted(stack):
        v = stack[k]
        try:
            loc = np.concatenate([np.asarray(sh.data).ravel()
                                  for sh in v.addressable_shards])
        except Exception:
            loc = np.asarray(v).ravel()
        if loc.size == 0:
            continue
        n_bad = int((~np.isfinite(loc)).sum())
        fin = loc[np.isfinite(loc)]
        parts.append(f"{k}[{fin.min():.4g},{fin.max():.4g}]"
                     + (f" nonfinite={n_bad}" if n_bad else ""))
    print(f"  [rank {rank}] forcing extremes ({tag}, local shard): "
          + " ".join(parts), flush=True)


def _spmd_device_count(run_config) -> int:
    """Device count the lat-band SPMD lane will shard over (1 = off).

    Multicontroller uses ALL global devices; single-controller uses
    ``--spmd-n-devices`` or every local device.  Shared by the tripole
    south-pad (needs it BEFORE the state is built) and the SPMD wiring.
    """
    if not run_config.enable_latlon_spmd:
        return 1
    if run_config.multicontroller:
        _nd = len(jax.devices())
        if run_config.spmd_n_devices and run_config.spmd_n_devices != _nd:
            raise SystemExit(
                f"--multicontroller uses ALL global devices ({_nd} across "
                f"{jax.process_count()} processes); --spmd-n-devices "
                f"({run_config.spmd_n_devices}) must be 0 (auto) or {_nd}.")
        return _nd
    return run_config.spmd_n_devices or len(jax.devices())


def _spmd_sea_ice_guard(ice_dynamics: str, ice_categories: int) -> None:
    """Refuse ``--enable-latlon-spmd --jra55-sea-ice`` combinations the lane
    cannot carry.  The thermodynamic SLAB tile (``--ice-dynamics none``,
    ``--ice-categories 1``) is elementwise, so it shards on the ocean's lat
    bands with no halo; EVP/mEVP/free-drift dynamics and the ITD transport
    reach ``pad_halo_latlon`` OUTSIDE the ocean's shard_map body, where the
    armed SPMD halo backend has no band axis to ppermute over.
    """
    if ice_dynamics != "none" or int(ice_categories) != 1:
        raise SystemExit(
            "--enable-latlon-spmd supports --jra55-sea-ice only as the slab "
            "thermodynamic tile (--ice-dynamics none --ice-categories 1); got "
            f"--ice-dynamics {ice_dynamics!r} --ice-categories "
            f"{int(ice_categories)}. Ice dynamics/ITD run their halo pads "
            "outside the sharded ocean body and are not SPMD-wired.")


def _create_setup(grid_type: str, resolution: str, nlev: int, H_max: float,
                  physics_preset: str, water_type: str,
                  use_bathymetry: bool = False,
                  tripole_mesh: str | None = None,
                  tripole_strip_north_rows: int = 0,
                  tripole_fold_convention: str = "auto",
                  A_h_override: float = None,
                  B_h_override: float = None,
                  K_h_override: float = None,
                  A_h_eq_boost: float = 1.0,
                  A_h_eq_sigma_deg: float = 5.0,
                  C_smag: float = None,
                  C_smag_lap: float | None = None,
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
                  dz_ref_override=None, t_depth_ref_override=None,
                  spmd_n_devices: int = 0,
                  mpas_lloyd: int = 50,
                  mpas_k_zeta_bih: float | None = None,
                  sw_scheme: str = "auto",
                  B_h_gamma0: float | None = None):
    """Create grid, z_coord, config, model for any grid type.

    ``spmd_n_devices > 1`` (MPAS only, ``--enable-mpas-spmd``) reorders + pads
    the Voronoi mesh for that device count BEFORE the model / IC / forcing are
    built on it, so every downstream per-cell array shares the sharded order.

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
        # A thickness-only override has no raw NEMO mesh e3w operand.  This
        # generic MPAS/lat-lon route therefore opts into the documented legacy
        # construction explicitly; NEMO state bridges pass raw e3w_0 and keep
        # the faithful mesh-reference default.
        # t_depth_ref_override carries NEMO's OWN gdept_1d when the caller has
        # it. Thicknesses alone do not determine those depths, and the
        # fidelity PGF (pgf_scheme="nemo_sco") telescopes against that exact
        # ladder; None keeps the arithmetic-midpoint construction, which is
        # the behaviour every existing run gets.
        z_coord = create_z_star_from_thicknesses(
            dz_ref_override, t_depth_ref_override,
            nemo_e3w_source="depth_difference")
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

    # --A-h / --K-h reached ONLY the realistic-bathymetry lat-lon branch
    # below, so on every other lane (the tripole NEMO-mesh lane included) the
    # flags parsed and then did nothing.  Apply them where the per-grid
    # defaults are set, i.e. once, for every branch.  Default is None on both,
    # so a run that does not pass them is unchanged.
    if A_h_override is not None:
        A_h = A_h_override
    if K_h_override is not None:
        K_h = K_h_override

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
                C_smag_lap=(C_smag_lap if C_smag_lap is not None else 0.15),
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
            # --pgf-scheme was read only by the realistic-bathymetry branch
            # above, so on this lane (the tripole NEMO mesh included) it
            # parsed and did nothing.  Unset keeps the config default.
            _flat = dict(
                A_h=A_h, K_h=K_h, A_v=A_v, K_v=K_v,
                n_barotropic_substeps=30,
                use_conservation_fixer=use_conservation_fixer,
                physics=None,
                implicit_vertical_mixing=implicit_vertical_mixing,
            )
            if pgf_scheme is not None:
                _flat["pgf_scheme"] = pgf_scheme
            config = LatLonCGridOceanConfig.from_flat(**_flat)
        model = LatLonCGridOceanModel(grid, z_coord, config)
        return grid, z_coord, config, model, "latlon"

    elif grid_type == "mpas":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.fidelity.nemo_match_recipe import (
            NEMOMatchMPASRecipeConfig,
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

        mesh = create_voronoi_mesh(params["level"], lloyd_iterations=mpas_lloyd)
        if spmd_n_devices > 1:
            from legoesm.parallel.voronoi_partition import (
                reorder_voronoi_for_sharding,
            )
            _n0 = mesh.nCells
            mesh = reorder_voronoi_for_sharding(mesh, spmd_n_devices, edge_order="owner")
            print(f"  MPAS SPMD mesh: reordered for {spmd_n_devices} devices, "
                  f"{_n0} -> {mesh.nCells} cells ({mesh.nCells - _n0} padded "
                  f"ghosts, land)")

        # For JRA55 forcing mode, use scheme="none" so that external
        # tau/q_net from the bulk-flux solver are applied via the
        # surface_forcing argument to model.step().  For restoring mode,
        # use the same "combined" config as the comparison scripts.
        # forcing_mode is passed from run_omip_single() via the parameter.
        if forcing_mode == "jra55_do_tropical":
            sf_config = SurfaceForcingConfig(
                scheme="none", shortwave_scheme=sw_scheme,
                shortwave_water_type=water_type)
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
        _mpas_over = {}
        if A_h_override is not None:
            _mpas_over["A_h"] = A_h_override
        if B_h_override is not None:
            _mpas_over["B_h"] = B_h_override
        if C_smag is not None:
            _mpas_over["C_smag"] = C_smag
        if C_smag_lap is not None:
            _mpas_over["C_smag_lap"] = C_smag_lap
        if C_leith is not None:
            _mpas_over["C_leith"] = C_leith
        if no_gm_redi:
            _mpas_over["gm_redi"] = False
        if _mpas_over:
            print(f"  MPAS recipe overrides: {_mpas_over}")
        config = nemo_match_mpas_model_config(
            NEMOMatchMPASRecipeConfig(**_mpas_over) if _mpas_over else None,
            physics=physics,
        )
        # Enforce global surface-freshwater balance, exactly as the lat-lon/tripole
        # config does (LatLonCGridOceanConfig.from_flat(normalize_freshwater=True) above).
        # The CORE-II P-E+R integral is a net ~+0.65 Sv freshwater input (a true
        # forcing imbalance, identical on every grid); without this the MPAS ocean
        # accumulates it as a ~-0.5 PSU global-mean fresh drift in 90 days, while
        # the tripole/lat-lon path (which sets the flag) stays balanced.  The MPAS
        # step already reads config.normalize_freshwater (ocean_pe_mpas) — the only
        # gap was the flag defaulting False on MPASOceanConfig.
        config = config._replace(normalize_freshwater=True)
        if mpas_k_zeta_bih is not None:
            # Explicit pin (--mpas-k-zeta-bih); otherwise the model derives it
            # from this mesh's spacing (resolution_scaled_k_zeta_bih).
            config = config._replace(K_zeta_bih=mpas_k_zeta_bih)
        model = MPASOceanModel(mesh, z_coord, config)
        # The MODEL's config is the record: it carries the vorticity damping
        # actually in force (derived from this mesh unless pinned), so the
        # run's saved configuration shows the resolved value, not ``None``.
        return mesh, z_coord, model.config, model, "mpas"

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
            NEMOMatchTripoleRecipeConfig,
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

        from legoesm.grids.tripole import mesh_file_list
        _mesh_files = mesh_file_list(tripole_mesh or params["mesh_path"])
        geom = create_tripole_grid(
            _mesh_files,
            fold_convention=(
                tripole_fold_convention if tripole_mesh
                else params.get("fold_convention", "auto")),
            strip_north_rows=int(tripole_strip_north_rows))

        if forcing_mode == "jra55_do_tropical":
            sf_config = SurfaceForcingConfig(
                scheme="none", shortwave_scheme=sw_scheme,
                shortwave_water_type=water_type)
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
        # The recipe is the proven NEMO-match bundle; --A-h / --K-h /
        # --pgf-scheme reach it as recipe fields, or the flags are inert on
        # this lane (a probe that lowered A_h by 33x reproduced the baseline
        # to three digits, because nothing read it).  Unset leaves the recipe
        # byte-identical.
        _recipe_over = {}
        if A_h_override is not None:
            _recipe_over["A_h"] = A_h_override
        if K_h_override is not None:
            # The NEMO-match recipe has no horizontal tracer diffusivity field
            # (tracer mixing on this lane is GM/Redi, kappa_Redi).  Say so
            # rather than let the flag look applied.
            print("  WARNING: --K-h does not apply to the tripole NEMO-match "
                  "recipe (tracer mixing is GM/Redi, kappa_Redi); ignored.")
        if pgf_scheme is not None:
            _recipe_over["pgf_scheme"] = pgf_scheme
        if B_h_override is not None:
            _recipe_over["B_h"] = B_h_override
        if B_h_gamma0 is not None:
            _recipe_over["B_h_gamma0"] = float(B_h_gamma0)
        if C_smag is not None:
            _recipe_over["C_smag"] = C_smag
        if C_smag_lap is not None:
            _recipe_over["C_smag_lap"] = C_smag_lap
        if C_leith is not None:
            _recipe_over["C_leith"] = C_leith
        if no_gm_redi:
            _recipe_over["gm_redi"] = False
        # --B-h / --C-smag / --C-leith > 0 are refused by the recipe factory
        # itself (the operators are not tripole-safe); nothing to guard here.
        _recipe_cfg = (NEMOMatchTripoleRecipeConfig(**_recipe_over)
                       if _recipe_over else None)
        if _recipe_over:
            print(f"  Tripole recipe overrides: {_recipe_over}")
        config = nemo_match_tripole_model_config(_recipe_cfg, physics=physics)

        # Resolution-scaled lateral viscosity.  Resolved HERE: this is where a
        # mesh and its land mask meet the configuration, and the mask is
        # required because the grid builder clamps degenerate cells to a fixed
        # floor, so the minimum over ALL cells is that floor on every mesh.
        # The later south-padded rebuild adds LAND rows only, so the narrowest
        # wet cell -- and therefore this number -- is the same there.
        if config.lateral_viscosity.A_h is None:
            from legoesm.ocean.init_tripole import read_mesh_mask_bathy
            from legoesm.ocean.state import (
                resolution_scaled_lateral_viscosity,
                wet_min_spacing,
            )
            _lm, _ = read_mesh_mask_bathy(
                _mesh_files, strip_north_rows=int(tripole_strip_north_rows))
            _lv = config.lateral_viscosity
            from legoesm.grids.tripole import DEFAULT_MIN_DX_M
            _dx_min = wet_min_spacing(geom, _lm,
                                      clamp_floor_m=DEFAULT_MIN_DX_M)
            _A_h = resolution_scaled_lateral_viscosity(_dx_min, _lv)
            config = config._replace(lateral_viscosity=_lv._replace(
                A_h=_A_h, A_h_dx_m=_dx_min))
            print(f"  Lateral viscosity from the mesh: A_h={_A_h:.6g} m2/s "
                  f"(narrowest wet cell {_dx_min:.1f} m; anchor "
                  f"{_lv.A_h_ref:.6g} m2/s at {_lv.A_h_ref_dx_m:.1f} m, "
                  f"squared law)")

        model = LatLonCGridOceanModel(geom, z_coord, config)
        # the MODEL's config, so a saved configuration records what ran
        return geom, z_coord, model.config, model, "tripole"

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

    # Coupler config: LY09 bulk flux with ALL THREE reference heights at 10 m
    # -- the JRA55-do convention.  ``tas``, ``huss`` and ``uas`` in the
    # JRA55-do v1.4.0 distribution each carry an explicit ``height = 10.0 m``
    # coordinate (the CF ``comment`` string "usually, 2 meter" is boilerplate
    # from the CMOR table and contradicts the file's own coordinate), and
    # FESOM2 forces the same dataset with
    # ``ncar_bulk_z_wind = ncar_bulk_z_tair = ncar_bulk_z_shum = 10.0``.
    # This previously read 2.0 m for T and q, which tells the MOST solver the
    # 10 m state sits at 2 m and inflates the air-sea gradients: measured on
    # one day of 1958 JRA55-do over PHC ocean points, with everything else
    # held fixed, +2.13 W/m^2 sensible (+7.8%) and +11.53 W/m^2 latent
    # (+10.9%).  See docs/ocean/fidelity/fesom2_gap_analysis.md.
    # The Item 1 fixes (LY09 U^6 term, 0.98 q_sat, separate reference
    # heights) are wired through CouplerConfig.
    from legoesm.coupler.config import CouplerConfig
    coupler_cfg = CouplerConfig(
        bulk_scheme="large_yeager",
        z_ref=10.0,
        z_t_atm=10.0,
        z_q_atm=10.0,
        stability_scheme=args.surface_stability_scheme,
        ocean_albedo=float(getattr(args, "ocean_albedo", 0.06)),
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
    state["zero_surface_fluxes"] = bool(
        getattr(args, "jra55_zero_surface_fluxes", False))

    # SSS restoring — Haney piston-velocity formulation, applied
    # globally (i.e. on every ocean cell) after the dynamics step.
    sss_restoring_enabled = (
        not args.jra55_no_sss_restoring
        and S_woa is not None
        and z_coord is not None
    )
    if getattr(args, "sw_penetration", "auto") == "sweeney_2band":
        if not getattr(args, "chl_clim", None):
            raise ValueError("--sw-penetration sweeney_2band needs --chl-clim <Sweeney_2005.nc>")
        if regrid_weights is None:
            raise ValueError("--sw-penetration sweeney_2band needs the JRA55 regrid "
                             "weights (MPAS / tripole lanes).")
        state["chl_monthly"] = jnp.asarray(_load_monthly_clim_target(
            args.chl_clim, "chl", regrid_weights,
            np.asarray(ds["lat"]), np.asarray(ds["lon"])))
    state["sw_net_to_forcing"] = getattr(args, "sw_penetration", "auto") != "auto"
    if getattr(args, "runoff_source", "jra55_friver") == "core2_climatology":
        if not getattr(args, "runoff_file", None):
            raise ValueError("--runoff-source core2_climatology needs --runoff-file <CORE2_runoff.nc>")
        if regrid_weights is None:
            raise ValueError("--runoff-source core2_climatology needs the JRA55 regrid "
                             "weights (MPAS / tripole lanes).")
        state["runoff_static"] = jnp.asarray(_load_core2_runoff_static(
            args.runoff_file, regrid_weights,
            np.asarray(ds["lat"]), np.asarray(ds["lon"])))
    if sss_restoring_enabled:
        state["sss_target_2d"] = jnp.asarray(S_woa[..., 0])
        state["sss_piston_velocity"] = float(args.sss_piston_velocity)
        state["dz_top"] = float(np.asarray(z_coord.dz_ref)[0])
        _sss_kind = getattr(args, "sss_restoring_target", "woa_winter")
        if _sss_kind == "woa_winter":
            pass
        elif _sss_kind == "phc2_monthly":
            if regrid_weights is None:
                raise ValueError("--sss-restoring-target phc2_monthly needs the "
                                 "JRA55 regrid weights (MPAS / tripole lanes).")
            _monthly = _load_phc2_monthly_sss_target(
                args.sss_target_file, regrid_weights,
                np.asarray(ds["lat"]), np.asarray(ds["lon"]))
            state["sss_target_monthly"] = jnp.asarray(_monthly)
            state["sss_target_2d"] = state["sss_target_monthly"][0]
        else:
            raise ValueError(f"unknown --sss-restoring-target {_sss_kind!r}")
        state["sss_remove_mean"] = bool(getattr(args, "sss_restoring_remove_mean", False))
        if state["sss_remove_mean"]:
            _area = (getattr(grid, "areaCell", None) if grid_type == "mpas"
                     else getattr(grid, "area", None))
            if _area is None:
                raise ValueError("--sss-restoring-remove-mean: this grid exposes "
                                 "no cell area (areaCell / area).")
            state["sss_area"] = jnp.asarray(np.asarray(_area))
        print(f"  SSS restoring: target={_sss_kind} "
              f"piston={float(args.sss_piston_velocity):g} m/s "
              f"remove_mean={state['sss_remove_mean']}")
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
        # Rheology + ITD are CLI-selectable (--ice-dynamics / --ice-categories
        # and the --ice-* parameter overrides).  The defaults reproduce the
        # historical slab (dynamics="none", n_categories=1) bit-for-bit, so an
        # existing command line is unaffected; a run that wants FESOM-like ice
        # asks for it explicitly.  stability_scheme only takes effect if
        # bulk_scheme is switched to a MOST-family scheme.
        _ice_dynamics = str(args.ice_dynamics)
        _ice_ncat = int(args.ice_categories)
        if _ice_ncat < 1:
            raise SystemExit(
                f"--ice-categories must be >= 1; got {_ice_ncat}.")
        # Rheology needs grid metrics for the strain rates.  ``rheology.py``
        # implements them for lat-lon and Voronoi (and the cubed sphere, which
        # this lane does not reach); the tripole geometry has no strain-rate
        # branch, so refuse rather than silently return zero deformation.
        if _ice_dynamics != "none" and grid_type not in ("latlon", "mpas"):
            raise SystemExit(
                f"--ice-dynamics {_ice_dynamics!r} is not supported on "
                f"--grid {grid_type!r} (rheology.strain_rates has no branch "
                "for it); use --grid latlon or mpas, or --ice-dynamics none."
            )
        _ice_kw = dict(
            stability_scheme=args.surface_stability_scheme,
            dynamics=_ice_dynamics,
            n_categories=_ice_ncat,
        )
        for _flag, _field in (
            ("ice_n_evp", "N_evp"),
            ("ice_p_star", "P_star"),
            ("ice_e_yield", "e_yield"),
            ("ice_c_strength", "C_strength"),
            ("ice_delta_min", "Delta_min"),
            ("ice_alpha_mevp", "alpha_mevp"),
            ("ice_beta_mevp", "beta_mevp"),
        ):
            _v = getattr(args, _flag, None)
            if _v is not None:
                _ice_kw[_field] = int(_v) if _field == "N_evp" else float(_v)
        state["ice_config"] = SeaIceConfig(**_ice_kw)
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
        if _ice_dynamics == "none" and _ice_ncat == 1:
            # Historical slab: a 3-field SeaIceState.  Kept byte-identical so
            # an existing --jra55-sea-ice command line is unaffected.
            state["ice_state_init"] = SeaIceState(
                h_ice=Field(_zeros, name="h_ice", dims=_ice_dims, units="m"),
                T_ice=Field(jnp.full(_ice_shape,
                                     float(_consts.T_freeze_ocean)),
                            name="T_ice", dims=_ice_dims, units="K"),
                concentration=Field(_zeros, name="concentration",
                                    dims=_ice_dims, units="1"),
            )
        else:
            # Rheology and/or ITD need the 12-field DynamicSeaIceState
            # (velocity + stress components, and a trailing category axis when
            # n_categories > 1).  ``init_dynamic_ice_state`` owns the shape and
            # dim-name contract per grid rank, so build through it rather than
            # assembling Fields here.
            from legoesm.ice.state import init_dynamic_ice_state
            _dyn_shape = (_ice_shape + (_ice_ncat,) if _ice_ncat > 1
                          else _ice_shape)
            state["ice_state_init"] = init_dynamic_ice_state(
                _dyn_shape, n_categories=_ice_ncat,
                T_ice_init=float(_consts.T_freeze_ocean),
            )
        # The grid is needed by ANY DynamicSeaIceState path, not just a
        # rheology: step_sea_ice validates the state's spatial rank against
        # the grid, and with grid=None it assumes the cubed sphere and fails
        # on the first step for a lat-lon/MPAS multi-category state.  So the
        # condition must match the one that chose the dynamic state above.
        state["ice_grid"] = (
            grid if (_ice_dynamics != "none" or _ice_ncat > 1) else None)
    else:
        state["enable_sea_ice"] = False

    # Frazil closure (opt-in --frazil): converts supercooling on every active
    # level into ice mass handed to the slab tile.  Off (default) ⇒ the block
    # scan is byte-identical.  The ice tile is the recipient, so the slab must
    # be on; the slab exchanges fresh water, so the frazil ice is fresh (see
    # below); the lane's freshwater closure must be the virtual salt
    # flux (the module's salt rejection IS that closure; the liquid thickness
    # is then not a prognostic to update).
    state["enable_frazil"] = False
    if getattr(args, "frazil", False):
        from legoesm.ocean.physics.frazil import FrazilConfig
        if not state["enable_sea_ice"]:
            raise SystemExit("--frazil requires --jra55-sea-ice (the slab tile "
                             "receives the frazil ice).")
        if state["ice_config"].dynamics != "none" or int(args.ice_categories) != 1:
            raise SystemExit("--frazil supports the single-category slab only "
                             "(--ice-dynamics none --ice-categories 1).")
        if z_coord is None:
            raise SystemExit("--frazil needs the run's vertical coordinate.")
        _ice = state["ice_config"]
        state["enable_frazil"] = True
        # The single-category slab exchanges FRESH water with the ocean
        # (sea_ice.py: ``salt_flux=zeros`` on the slab path; melt returns
        # freshwater), so the frazil ice must be fresh too or the salt budget
        # would not close on melt.  ``constant`` = the slab's own T_freeze_ocean.
        state["frazil_config"] = FrazilConfig(
            enabled=True, freezing_scheme="constant",
            ice_salinity_psu=0.0, L_f_j_kg=float(_ice.L_f))
        state["frazil_z_coord"] = z_coord

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

    fw = jra55_to_freshwater(slc, tile_resp.lhflx, evap=tile_resp.surface_mass_flux)
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
        regrid_jra55_slice,
    )

    cache_path = jra55_state["cache_path"]
    ref_year = jra55_state["ref_year"]
    cycle = jra55_state.get("cycle", False)
    lat_2d = jra55_state["lat_2d"]
    lon_2d = jra55_state["lon_2d"]
    co2_ppmv = jra55_state["co2_ppmv"]
    # MPAS / tripole: the cache is lat-lon, the model is not -- regrid each
    # slice with the SAME helper the single-step lane uses (the GPU-interp
    # lane regrids its raw records the same way).  Without this the lat-lon
    # slice met the per-cell lat/lon in jra55_to_atm_surface (first MPAS SPMD
    # smoke, job 27326306: broadcast [655364] vs [180,360]).
    regrid_weights = jra55_state.get("regrid_weights")

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
        if regrid_weights is not None:
            slc = regrid_jra55_slice(slc, regrid_weights)
        atm = jra55_to_atm_surface(
            slc, lat_2d, lon_2d, day,
            ref_year=ref_year, co2_ppmv=co2_ppmv,
        )
        for f in fields:
            accum[f].append(getattr(atm, f))
        runoffs.append(slc.friver)

    atm_stack = {f: jnp.stack(accum[f]) for f in fields}
    # Route runoff off dry cells ONCE on the record stack: the routing
    # is linear in the field and its plan is static, so this is
    # identical to routing every interpolated step inside the scan.
    runoff_stack = _route_runoff_stack(
        jnp.stack(runoffs), jra55_state.get("runoff_map"),
        static=jra55_state.get("runoff_static"))
    return atm_stack, runoff_stack


def _load_phc2_monthly_sss_target(path, regrid_weights, cache_lat, cache_lon) -> np.ndarray:
    """Monthly PHC2 SSS climatology (``SALT``, missing -99.0) on the model grid."""
    return _load_monthly_clim_target(path, "SALT", regrid_weights, cache_lat, cache_lon,
                                     missing_value=-99.0)


def _load_monthly_clim_target(path, var, regrid_weights, cache_lat, cache_lon,
                              missing_value=None) -> np.ndarray:
    """Load a 12-month (12, 180, 360) climatology and regrid it to the model grid.

    Entries equal to ``missing_value`` (and any masked / NaN entries) are
    treated as land, gap-filled by nearest valid neighbour on the unit sphere,
    then each month is regridded on the host.  Returns an array of shape
    ``(12,) + tuple(regrid_weights.target_shape)``.  Used for the PHC2 SSS
    restoring target and the Sweeney chlorophyll climatology.
    """
    import netCDF4

    from legoesm.grids.regridding import fill_missing_nearest_valid

    with netCDF4.Dataset(path) as ds:
        file_lat = np.asarray(ds["lat"][:])
        file_lon = np.asarray(ds["lon"][:])
        lat_ok = np.shape(file_lat) == np.shape(cache_lat) and np.allclose(file_lat, cache_lat)
        lon_ok = np.shape(file_lon) == np.shape(cache_lon) and np.allclose(file_lon, cache_lon)
        if not (lat_ok and lon_ok):
            raise ValueError(
                f"{path}: PHC2 lat/lon axes disagree with the forcing cache grid -- "
                f"lat: match={lat_ok} (file shape {file_lat.shape} vs cache "
                f"{np.shape(cache_lat)}); lon: match={lon_ok} (file shape "
                f"{file_lon.shape} vs cache {np.shape(cache_lon)})")
        salt = ds.variables[var][:]

    data = np.ma.asanyarray(salt)
    if missing_value is not None:
        data = np.ma.masked_equal(data, missing_value)
    data = np.ma.masked_invalid(data)
    data = np.ma.filled(data.astype(np.float64), np.nan)
    nlat, nlon = file_lat.size, file_lon.size
    if data.shape != (12, nlat, nlon):
        raise ValueError(f"{path}: expected {var} with shape (12, {nlat}, {nlon}), "
                         f"got {tuple(data.shape)}")

    # Unit-sphere Cartesian coordinates of the (lat, lon) meshgrid, flattened
    # C-order to match data.reshape(12, -1).
    lon2d, lat2d = np.meshgrid(file_lon, file_lat)
    la = np.deg2rad(lat2d.ravel())
    lo = np.deg2rad(lon2d.ravel())
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)
    filled = np.asarray(fill_missing_nearest_valid(data.reshape(12, -1), xyz)).reshape(12, nlat, nlon)
    # Regrid on the HOST (deterministic NumPy), not on each process's GPU:
    # under the route-B multicontroller the GPU regrid gave byte-different
    # results across processes (the defect _regrid_records_host exists for).
    target = np.asarray(_regrid_records_host(filled, regrid_weights))
    return target.reshape((12,) + tuple(regrid_weights.target_shape))


def _sss_month_index(day: float) -> int:
    """Month index (0-11) containing ``day`` on a noleap calendar (day 0 = 1 Jan)."""
    lengths = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    return int(np.searchsorted(np.cumsum(lengths), float(day) % 365.0, side="right"))


def _restore_sss_top(S, target, alpha, mask, area=None, remove_mean=False):
    """Haney piston restoring of the top layer.

    The tendency ``alpha * (target - S_top) * mask`` is applied to the top
    layer only; with ``remove_mean`` False the result is byte-identical to
    ``S_top - alpha * (S_top - target) * mask``.  ``remove_mean`` subtracts
    the area-weighted wet-cell mean of the tendency (FESOM2
    ``sss_runoff_fluxes`` relax_salt balance) so the restoring adds no net
    salt; it requires ``area`` (global reductions: fine on GSPMD arrays).
    """
    if remove_mean and area is None:
        raise ValueError("remove_mean=True requires 'area' (cell areas).")
    S_top = S[..., 0]
    tend = alpha * (target - S_top) * mask
    if remove_mean:
        tend = (tend - jnp.sum(tend * area) / jnp.sum(area * mask)) * mask
    return S.at[..., 0].set(S_top + tend)


def _refs_for_block(base_refs, jra55_state, day0, shard_fn=None):
    """Per-block reference dict for the SSS restoring.

    Selects the current month's slice of the PHC2 climatology as
    ``sss_target`` (no temporal interpolation -- FESOM reads the current
    month) and, when mean removal is on, the ``sss_area`` for it.  Under
    lat-band SPMD each month and the area go through ``shard_fn`` once and
    are cached on ``jra55_state``.  The month is fixed for the whole block
    (one forcing block = the diagnostic cadence, 2880 s on the ico9
    launcher), so a block straddling a month boundary keeps the old target
    for at most that long, once a month.
    """
    refs = dict(base_refs or {})
    m = _sss_month_index(day0)
    if "chl_monthly" in jra55_state:
        chl_monthly = jra55_state["chl_monthly"]
        if shard_fn is None:
            refs["chl"] = chl_monthly[m]
        else:
            if "_chl_monthly_sharded" not in jra55_state:
                jra55_state["_chl_monthly_sharded"] = [shard_fn(chl_monthly[k]) for k in range(12)]
            refs["chl"] = jra55_state["_chl_monthly_sharded"][m]
    if "sss_target_monthly" not in jra55_state:
        return refs if "chl" in refs else base_refs
    monthly = jra55_state["sss_target_monthly"]
    if shard_fn is None:
        refs["sss_target"] = monthly[m]
    else:
        if "_sss_monthly_sharded" not in jra55_state:
            jra55_state["_sss_monthly_sharded"] = [shard_fn(monthly[k]) for k in range(12)]
        refs["sss_target"] = jra55_state["_sss_monthly_sharded"][m]
    if jra55_state.get("sss_remove_mean", False):
        area = jra55_state["sss_area"]
        if shard_fn is None:
            refs["sss_area"] = area
        else:
            if "_sss_area_sharded" not in jra55_state:
                jra55_state["_sss_area_sharded"] = shard_fn(area)
            refs["sss_area"] = jra55_state["_sss_area_sharded"]
    return refs


def _regrid_records_host(recs, rw):
    """Deterministic HOST (NumPy) k-neighbour regrid of stacked records.

    Route-B multicontroller: every process builds the forcing stack itself and
    ``shard_forcing_stack_latlon`` requires the bytes to agree across processes.
    In the 16-GPU ORCA12 smoke (job 27324638) the GPU ``regrid_scalar`` results
    differed across processes (all leaves) while an identical standalone build
    agreed across CPU processes, GPUs and nodes; the divergence is not located,
    so under multi-process the regrid is done here in NumPy -- byte-identical
    by construction.  One record at a time (a (n_target, k) intermediate, not
    (n_rec, n_target, k)); trailing dims after the source (lat, lon) are kept,
    matching ``regrid_scalar``'s contract.  ``recs``: (n_rec, n_lat, n_lon[, ...]).
    """
    idx = np.asarray(rw.src_indices)
    w = np.asarray(rw.weights, dtype=np.float64)
    recs = np.asarray(recs, dtype=np.float64)
    n_rec = int(recs.shape[0])
    spatial = int(rw.src_flat_size)
    n_extra = int(recs[0].size // spatial)
    if n_extra * spatial != recs[0].size:
        raise ValueError(
            f"record size {recs[0].size} is not a multiple of the source grid "
            f"size {spatial}")
    trailing = tuple(recs.shape[3:]) if n_extra > 1 else ()
    out = np.empty((n_rec, idx.shape[0], n_extra), dtype=np.float64)
    for r in range(n_rec):
        flat = recs[r].reshape(spatial, n_extra)
        out[r] = (flat[idx] * w[..., None]).sum(axis=1)
    return jnp.asarray(out.reshape((n_rec,) + tuple(rw.target_shape) + trailing))


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
            if jax.process_count() > 1:
                # same byte-identical host path as _preload_jra55_raw_records
                all_records[var] = _regrid_records_host(arr, rw).astype(
                    arr.dtype)
            else:
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
    runoff_stack = _route_runoff_stack(
        raw_stack["friver"], jra55_state.get("runoff_map"),
        static=jra55_state.get("runoff_static"))

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
        # Route-B multicontroller: every process builds this stack itself and
        # shard_forcing_stack_latlon then REQUIRES the bytes to agree across
        # processes.  The k=4 IDW regrid is a gather + tiny float reduction;
        # in the 16-GPU ORCA12 smoke (job 27324638) its GPU results differed
        # across processes (all 10 leaves), while an identical standalone build
        # agreed across CPU processes, GPUs and nodes -- the divergence is not
        # located yet, so under multi-process the regrid is done in NumPy on
        # the host (deterministic by construction, no device transfers); the
        # single-process path keeps the GPU regrid byte-unchanged.
        if jax.process_count() > 1:
            for var in raw_stack:
                raw_stack[var] = _regrid_records_host(raw_stack[var], rw)
        else:
            for var in raw_stack:
                raw_stack[var] = jnp.stack([
                    regrid_scalar(raw_stack[var][i], rw)
                    for i in range(raw_stack[var].shape[0])
                ])
    runoff_stack = _route_runoff_stack(
        raw_stack["friver"], jra55_state.get("runoff_map"),
        static=jra55_state.get("runoff_static"))

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
    sw_net_to_forcing = bool(jra55_state.get("sw_net_to_forcing", False))
    use_chl = "chl_monthly" in jra55_state
    sss_remove_mean = bool(jra55_state.get("sss_remove_mean", False))
    zero_fluxes = bool(jra55_state.get("zero_surface_fluxes", False))
    enable_freeze = bool(jra55_state.get("enable_freeze_cap", False))
    # Prognostic slab sea ice (opt-in --jra55-sea-ice): replaces the freeze-cap
    # SST stand-in.  Static gate ⇒ ice-off blocks are bit-identical.  Setup
    # forces enable_freeze_cap=False when ice is on (no double-capping).
    enable_sea_ice = bool(jra55_state.get("enable_sea_ice", False))
    ice_cfg = jra55_state.get("ice_config")
    # None unless --ice-dynamics selects a rheology; the strain
    # rates need grid metrics, the slab path does not.
    ice_grid = jra55_state.get("ice_grid")
    # Frazil (opt-in --frazil): static gate ⇒ frazil-off blocks are bit-identical.
    enable_frazil = bool(jra55_state.get("enable_frazil", False))
    if enable_frazil and not enable_sea_ice:
        raise ValueError("frazil needs the slab sea-ice tile as its recipient")
    if enable_frazil:
        from legoesm.ocean.physics.frazil import apply_frazil
        from legoesm.coupler.ocean_forcing import add_frazil_ice
        from legoesm.ocean.vertical import compute_layer_thickness
        frazil_cfg = jra55_state["frazil_config"]
        # The coordinate the model integrates on (the lat-lon lane builds a
        # partial-cell coordinate from the driver's reference one).
        frazil_zc = getattr(model, "z_coord", None) or jra55_state["frazil_z_coord"]
        frazil_rho0 = float(getattr(model.config, "rho_0", _const.rho_ocean))
        if getattr(model.config, "freshwater_closure", "virtual_salt_flux") != "virtual_salt_flux":
            raise ValueError("--frazil requires freshwater_closure='virtual_salt_flux' "
                             f"(got {model.config.freshwater_closure!r}).")
        # Reference hydrostatic sea pressure of each level centre [Pa]: under
        # the "constant" liquidus pressure only enters the potential/in-situ
        # conversion, and the z-star compression (eta/H ~ 1e-4) and the
        # partial bottom cell (never supercooled) shift that by < 1e-3 K.
        _dz_ref = np.asarray(frazil_zc.dz_ref, dtype=np.float64)
        frazil_p_pa = jnp.asarray(_const.rho_ocean * _const.g
                                  * (np.cumsum(_dz_ref) - 0.5 * _dz_ref))

        def _frazil(new_state, new_ice):
            T, S = new_state.T.data, new_state.S.data
            h = compute_layer_thickness(new_state.eta.data, new_state.H_bathy.data, frazil_zc)
            active = (h > 0.0) & (new_state.land_mask.data > 0.5)[..., None]
            # The closure promotes to its widest input; keep the scan carry dtype.
            res = apply_frazil(T, S, h.astype(T.dtype), active,
                               frazil_p_pa.astype(T.dtype), frazil_cfg)
            ice_mass = res.ice_mass_per_area_kg_m2
            # The frozen liquid leaves the column the way the lane's freshwater
            # fluxes do (freshwater_eta_tendency moves eta under the virtual-
            # salt closure too): the z-star Jacobian carries the thickness
            # loss the contract asks for, and the module's salinity update is
            # the closure's virtual salt.  Same convention as the slab tile's
            # own lead freezing (ice_fw < 0), so a later melt returns exactly
            # this water.
            # ponytail: the z-star rescale spreads the thickness loss over the
            # whole column while the closure removed it from the frozen level,
            # a per-layer inventory error of order x*dz_k/H (mm of water over
            # a km column) -- the same approximation every surface freshwater
            # flux makes on this coordinate; exact per-layer thickness needs
            # the real_freshwater closure with a prognostic thickness.
            eta = new_state.eta.data - (ice_mass / frazil_rho0).astype(new_state.eta.data.dtype)
            new_state = new_state._replace(T=new_state.T.replace(data=res.T_C.astype(T.dtype)),
                                           S=new_state.S.replace(data=res.S_psu.astype(S.dtype)),
                                           eta=new_state.eta.replace(data=eta))
            return new_state, add_frazil_ice(new_ice, ice_cfg, ice_mass.astype(new_ice.h_ice.data.dtype))

    sponge = _build_sponge_forcing(jra55_state) if enable_sponge else None

    if enable_sss:
        sss_pv = float(jra55_state["sss_piston_velocity"])
        sss_dz = float(jra55_state["dz_top"])
        sss_alpha_static = sss_pv * dt / max(sss_dz, 1e-6)
        sss_target_static = jra55_state["sss_target_2d"]
        sss_area_static = jra55_state.get("sss_area")
    else:
        sss_alpha_static = 0.0
        sss_target_static = None
        sss_area_static = None

    freeze_from_gamma = False
    if enable_freeze:
        sponge_gamma = jra55_state.get("sponge_gamma_2d", None)
        if sponge_gamma is not None and np.any(np.asarray(sponge_gamma) > 0):
            freeze_mask_static = sponge_gamma > 0.0
            freeze_from_gamma = True
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
    # lat-lon nests the barotropic knobs (config.barotropic.*); MPASOceanConfig
    # carries them flat — read whichever the model has.
    _maxvel_3d = getattr(model.config, "barotropic", model.config).maxvel_barotropic
    enable_maxvel = _maxvel_3d > 0.0

    # Lat-band SPMD (--enable-latlon-spmd): the scan body's dynamics step
    # runs through the sharded wrapper (same forcing kwargs as _step_impl;
    # the wrapper's cache/arm-restore Python runs ONCE at block trace).
    # Sea ice is refused upstream (the ice tile is not SPMD-audited yet).
    if (spmd_step is not None and enable_sea_ice
            and not jra55_state.get("spmd_ice_ok", False)):
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
                 ice_state=None, aux=None, refs=None):
        # Reference fields: closure constants by default (single-process,
        # byte-unchanged); under lat-band SPMD the driver passes them as the
        # ``refs`` ARGUMENT, band-sharded like the forcing stack, so they are
        # not replicated whole on every GPU (ORCA12: ~16 GB of 3-D sponge
        # references per device) and route-B never closes over a
        # non-addressable array.
        _sponge, _sss_target, _freeze_mask = (
            sponge, sss_target_static, freeze_mask_static)
        _sss_area = sss_area_static
        _chl = None
        if refs is not None:
            from legoesm.ocean.sponge import SpongeForcing
            # refs may carry ONLY the per-block SSS entries (monthly target
            # on a non-sharded lane): every key is optional.
            if enable_sponge and "sponge_gamma" in refs:
                _sponge = SpongeForcing(
                    gamma=refs["sponge_gamma"], T_ref=refs["sponge_T_ref"],
                    S_ref=refs["sponge_S_ref"])
            if enable_sss and "sss_target" in refs:
                _sss_target = refs["sss_target"]
            if enable_sss and "sss_area" in refs:
                _sss_area = refs["sss_area"]
            _chl = refs["chl"] if use_chl and "chl" in refs else None
            if freeze_from_gamma and "sponge_gamma" in refs:
                _freeze_mask = refs["sponge_gamma"] > 0.0
            elif freeze_mask_static is not None and "ocean_mask" in refs:
                _freeze_mask = refs["ocean_mask"]

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
                # Explicit --sw-penetration schemes take the post-albedo
                # shortwave (FESOM2's 0.54*(1-albw)*SW); "auto" keeps the
                # historical raw sw_down.
                sw_down=sw_net if sw_net_to_forcing else atm.sw_down,
                q_net=q_net,
                tau_x=tau_x,
                tau_y=tau_y,
                freshwater=None,
                chl=_chl,
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
                    dt=dt, grid=ice_grid,
                    ocean_mask=state_in.land_mask.data,
                )
            if zero_fluxes:
                fw = jax.tree_util.tree_map(jnp.zeros_like, fw)
                sf = jax.tree_util.tree_map(jnp.zeros_like, sf)
            # Ramp sponge strength alongside wind stress.
            if enable_ramp and enable_sponge:
                sponge_step = _sponge._replace(gamma=_sponge.gamma * ramp)
            else:
                sponge_step = _sponge

            new_state = _dyn_step(
                state_in, dt, aux=aux,
                freshwater=fw, surface_forcing=sf, sponge=sponge_step,
            )

            # SSS restoring (gated at compile time via Python `if`).
            if enable_sss:
                S = new_state.S.data
                new_state = new_state._replace(S=new_state.S.replace(
                    data=_restore_sss_top(
                        S, jnp.asarray(_sss_target, dtype=S.dtype),
                        jnp.asarray(sss_alpha_static, dtype=S.dtype),
                        jnp.asarray(new_state.land_mask.data, dtype=S.dtype),
                        area=(jnp.asarray(_sss_area, dtype=S.dtype)
                              if sss_remove_mean else None),
                        remove_mean=sss_remove_mean)))

            # T_freeze cap inside sponge.
            if enable_freeze:
                T = new_state.T.data
                T_freeze_C = jnp.asarray(T_freeze_C_static, dtype=T.dtype)
                T_top = T[..., 0]
                T_top_capped = jnp.where(
                    _freeze_mask,
                    jnp.maximum(T_top, T_freeze_C),
                    T_top,
                )
                new_state = new_state._replace(
                    T=new_state.T.replace(data=T.at[..., 0].set(T_top_capped)),
                )

            if enable_frazil:
                new_state, new_ice = _frazil(new_state, new_ice)

            # 3D velocity clip (MOM6 MAXVEL analog for full field).
            if enable_maxvel:
                # MPAS carries the full velocity as edge-normal u (no v field).
                u_clipped = jnp.clip(new_state.u.data, -_maxvel_3d, _maxvel_3d)
                new_state = new_state._replace(u=new_state.u.replace(data=u_clipped))
                if getattr(new_state, "v", None) is not None:
                    v_clipped = jnp.clip(new_state.v.data, -_maxvel_3d, _maxvel_3d)
                    new_state = new_state._replace(v=new_state.v.replace(data=v_clipped))

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
    sw_net_to_forcing = bool(jra55_state.get("sw_net_to_forcing", False))
    use_chl = "chl_monthly" in jra55_state
    sss_remove_mean = bool(jra55_state.get("sss_remove_mean", False))
    zero_fluxes = bool(jra55_state.get("zero_surface_fluxes", False))
    enable_freeze = bool(jra55_state.get("enable_freeze_cap", False))

    sponge = _build_sponge_forcing(jra55_state) if enable_sponge else None

    if enable_sss:
        sss_pv = float(jra55_state["sss_piston_velocity"])
        sss_dz = float(jra55_state["dz_top"])
        sss_alpha_static = sss_pv * dt / max(sss_dz, 1e-6)
        sss_target_static = jra55_state["sss_target_2d"]
        sss_area_static = jra55_state.get("sss_area")
    else:
        sss_alpha_static = 0.0
        sss_target_static = None
        sss_area_static = None

    freeze_from_gamma = False
    if enable_freeze:
        sponge_gamma = jra55_state.get("sponge_gamma_2d", None)
        if sponge_gamma is not None and np.any(np.asarray(sponge_gamma) > 0):
            freeze_mask_static = sponge_gamma > 0.0
            freeze_from_gamma = True
        else:
            freeze_mask_static = jra55_state.get("_ocean_mask_2d", None)
        T_freeze_C_static = float(jra55_state["T_freeze_ocean_C"])
    else:
        freeze_mask_static = None
        T_freeze_C_static = -1.8

    # Prognostic slab sea ice (opt-in) — see _build_jra55_block_fn.
    enable_sea_ice = bool(jra55_state.get("enable_sea_ice", False))
    ice_cfg = jra55_state.get("ice_config")
    # None unless --ice-dynamics selects a rheology; the strain
    # rates need grid metrics, the slab path does not.
    ice_grid = jra55_state.get("ice_grid")
    # Frazil (opt-in --frazil): static gate ⇒ frazil-off blocks are bit-identical.
    enable_frazil = bool(jra55_state.get("enable_frazil", False))
    if enable_frazil and not enable_sea_ice:
        raise ValueError("frazil needs the slab sea-ice tile as its recipient")
    if enable_frazil:
        from legoesm.ocean.physics.frazil import apply_frazil
        from legoesm.coupler.ocean_forcing import add_frazil_ice
        from legoesm.ocean.vertical import compute_layer_thickness
        frazil_cfg = jra55_state["frazil_config"]
        # The coordinate the model integrates on (the lat-lon lane builds a
        # partial-cell coordinate from the driver's reference one).
        frazil_zc = getattr(model, "z_coord", None) or jra55_state["frazil_z_coord"]
        frazil_rho0 = float(getattr(model.config, "rho_0", _const.rho_ocean))
        if getattr(model.config, "freshwater_closure", "virtual_salt_flux") != "virtual_salt_flux":
            raise ValueError("--frazil requires freshwater_closure='virtual_salt_flux' "
                             f"(got {model.config.freshwater_closure!r}).")
        # Reference hydrostatic sea pressure of each level centre [Pa]: under
        # the "constant" liquidus pressure only enters the potential/in-situ
        # conversion, and the z-star compression (eta/H ~ 1e-4) and the
        # partial bottom cell (never supercooled) shift that by < 1e-3 K.
        _dz_ref = np.asarray(frazil_zc.dz_ref, dtype=np.float64)
        frazil_p_pa = jnp.asarray(_const.rho_ocean * _const.g
                                  * (np.cumsum(_dz_ref) - 0.5 * _dz_ref))

        def _frazil(new_state, new_ice):
            T, S = new_state.T.data, new_state.S.data
            h = compute_layer_thickness(new_state.eta.data, new_state.H_bathy.data, frazil_zc)
            active = (h > 0.0) & (new_state.land_mask.data > 0.5)[..., None]
            # The closure promotes to its widest input; keep the scan carry dtype.
            res = apply_frazil(T, S, h.astype(T.dtype), active,
                               frazil_p_pa.astype(T.dtype), frazil_cfg)
            ice_mass = res.ice_mass_per_area_kg_m2
            # The frozen liquid leaves the column the way the lane's freshwater
            # fluxes do (freshwater_eta_tendency moves eta under the virtual-
            # salt closure too): the z-star Jacobian carries the thickness
            # loss the contract asks for, and the module's salinity update is
            # the closure's virtual salt.  Same convention as the slab tile's
            # own lead freezing (ice_fw < 0), so a later melt returns exactly
            # this water.
            # ponytail: the z-star rescale spreads the thickness loss over the
            # whole column while the closure removed it from the frozen level,
            # a per-layer inventory error of order x*dz_k/H (mm of water over
            # a km column) -- the same approximation every surface freshwater
            # flux makes on this coordinate; exact per-layer thickness needs
            # the real_freshwater closure with a prognostic thickness.
            eta = new_state.eta.data - (ice_mass / frazil_rho0).astype(new_state.eta.data.dtype)
            new_state = new_state._replace(T=new_state.T.replace(data=res.T_C.astype(T.dtype)),
                                           S=new_state.S.replace(data=res.S_psu.astype(S.dtype)),
                                           eta=new_state.eta.replace(data=eta))
            return new_state, add_frazil_ice(new_ice, ice_cfg, ice_mass.astype(new_ice.h_ice.data.dtype))

    # lat-lon nests the barotropic knobs (config.barotropic.*); MPASOceanConfig
    # carries them flat — read whichever the model has.
    _maxvel_3d = getattr(model.config, "barotropic", model.config).maxvel_barotropic
    enable_maxvel = _maxvel_3d > 0.0

    # Lat-band SPMD: see _build_jra55_block_fn.
    if (spmd_step is not None and enable_sea_ice
            and not jra55_state.get("spmd_ice_ok", False)):
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
                     ice_state=None, aux=None, refs=None):
            # Reference fields: closure constants by default (single-process,
            # byte-unchanged); under lat-band SPMD the driver passes them as the
            # ``refs`` ARGUMENT, band-sharded like the forcing stack, so they are
            # not replicated whole on every GPU (ORCA12: ~16 GB of 3-D sponge
            # references per device) and route-B never closes over a
            # non-addressable array.
            _sponge, _sss_target, _freeze_mask = (
                sponge, sss_target_static, freeze_mask_static)
            _sss_area = sss_area_static
            _chl = None
            if refs is not None:
                from legoesm.ocean.sponge import SpongeForcing
                # refs may carry ONLY the per-block SSS entries (monthly target
                # on a non-sharded lane): every key is optional.
                if enable_sponge and "sponge_gamma" in refs:
                    _sponge = SpongeForcing(
                        gamma=refs["sponge_gamma"], T_ref=refs["sponge_T_ref"],
                        S_ref=refs["sponge_S_ref"])
                if enable_sss and "sss_target" in refs:
                    _sss_target = refs["sss_target"]
                if enable_sss and "sss_area" in refs:
                    _sss_area = refs["sss_area"]
                _chl = refs["chl"] if use_chl and "chl" in refs else None
                if freeze_from_gamma and "sponge_gamma" in refs:
                    _freeze_mask = refs["sponge_gamma"] > 0.0
                elif freeze_mask_static is not None and "ocean_mask" in refs:
                    _freeze_mask = refs["ocean_mask"]

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
                    sw_down=sw_net if sw_net_to_forcing else atm.sw_down, q_net=q_net,
                    tau_x=tile.tau_x * ramp, tau_y=tile.tau_y * ramp,
                    freshwater=None,
                    chl=_chl,
                )

                # Prognostic slab sea ice: partition surface forcing between
                # open ocean (f_ocean=1-A) and the ice tile.  ocean_mask: land
                # cells receive no ice->ocean forcing (mask-aware blend
                # contract; land_mask is scan-carry state, traced-safe).
                if enable_sea_ice:
                    new_ice, fw, sf = omip_sea_ice_surface_forcing(
                        ice_state=ice_in, ice_config=ice_cfg, atm=atm,
                        ocean_sst_K=sst_K, open_ocean_sf=sf, open_ocean_fw=fw,
                        dt=dt, grid=ice_grid,
                        ocean_mask=state_in.land_mask.data,
                    )
                if zero_fluxes:
                    fw = jax.tree_util.tree_map(jnp.zeros_like, fw)
                    sf = jax.tree_util.tree_map(jnp.zeros_like, sf)

                sponge_k = (_sponge._replace(gamma=_sponge.gamma * ramp)
                            if enable_sponge else None)
                new_state = _dyn_step(
                    state_in, dt, aux=aux, freshwater=fw,
                    surface_forcing=sf, sponge=sponge_k,
                )

                if enable_sss:
                    S = new_state.S.data
                    new_state = new_state._replace(S=new_state.S.replace(
                        data=_restore_sss_top(
                            S, jnp.asarray(_sss_target, dtype=S.dtype),
                            jnp.asarray(sss_alpha_static, dtype=S.dtype),
                            jnp.asarray(new_state.land_mask.data, dtype=S.dtype),
                            area=(jnp.asarray(_sss_area, dtype=S.dtype)
                                  if sss_remove_mean else None),
                            remove_mean=sss_remove_mean)))
                if enable_freeze:
                    T = new_state.T.data
                    T_top = jnp.where(
                        _freeze_mask,
                        jnp.maximum(T[..., 0], T_freeze_C_static),
                        T[..., 0],
                    )
                    new_state = new_state._replace(
                        T=new_state.T.replace(
                            data=T.at[..., 0].set(T_top)))
                if enable_frazil:
                    new_state, new_ice = _frazil(new_state, new_ice)
                if enable_maxvel:
                    new_state = new_state._replace(
                        u=new_state.u.replace(
                            data=jnp.clip(new_state.u.data,
                                          -_maxvel_3d, _maxvel_3d)))
                    if getattr(new_state, "v", None) is not None:  # MPAS: no v
                        new_state = new_state._replace(
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

    if grid_type == "mpas":
        # Mirrors the lat-lon branch: under the route-B multicontroller the
        # state fields are jax Arrays sharded across PROCESSES, so a host
        # gather (np.asarray) of a field spanning devices is an error; the
        # masked jnp reductions run in place on the shards and each returns
        # a replicated scalar that float() can read.
        T = state.T.data
        S = state.S.data
        eta = state.eta.data
        mask = state.land_mask.data
        # MPAS: (nCells, nlev), mask: (nCells,)
        wet = mask > 0.5
        n_wet = jnp.sum(wet)
        # All-land mesh (n_wet == 0) now yields NaN, not the old 0.0: that
        # plausible-looking 0.0 hid a broken land_mask; NaN makes it visible.
        sst = float(jnp.sum(jnp.where(wet, T[..., 0], 0.0)) / n_wet)
        sss = float(jnp.sum(jnp.where(wet, S[..., 0], 0.0)) / n_wet)
        ssh = float(jnp.sum(jnp.where(wet, eta, 0.0)) / n_wet)
        # u: (nEdges, nlev) edge array; padded ghost edges carry 0, so no wet
        # mask is needed -- identical to the old np.max(np.abs(u)).
        max_u = float(jnp.max(jnp.abs(state.u.data)))
    else:
        # Cubed-sphere (6,n,n,nlev) or latlon/tripole (nlat,nlon,nlev).
        # Reductions, not host arrays: under lat-band SPMD the state is
        # sharded across PROCESSES, and gathering it to compute a handful of
        # scalars costs a full global copy on one device every diagnostic
        # sample (7.4 GiB per field at ORCA12 -- it exhausted one arm's memory
        # and hung another).  jnp reductions run in place on each band and
        # return a replicated scalar, so this works on a sharded state and on
        # a plain host array alike.  Masked means replace boolean indexing,
        # whose output shape would be data-dependent.
        T = state.T.data
        S = state.S.data
        eta = state.eta.data
        mask = state.land_mask.data
        wet = mask > 0.5
        # No clamp on the denominator: an all-land domain gives NaN, the way
        # the mean over an empty selection did.  A plausible 0.0 would hide a
        # broken mask behind a finite number.
        n_wet = jnp.sum(wet)
        sst = float(jnp.sum(jnp.where(wet, T[..., 0], 0.0)) / n_wet)
        sss = float(jnp.sum(jnp.where(wet, S[..., 0], 0.0)) / n_wet)
        ssh = float(jnp.sum(jnp.where(wet, eta, 0.0)) / n_wet)
        u_raw = state.u.data
        v_raw = state.v.data if hasattr(state, 'v') else jnp.zeros_like(u_raw)
        # C-grid lat-lon: u is (nlat, nlon+1, nlev), v is (nlat+1, nlon, nlev).
        # Interpolate staggered velocities to cell centers before computing speed.
        if grid_type in ("latlon", "tripole") and u_raw.shape[1] != T.shape[1]:
            u_c = 0.5 * (u_raw[:, :-1] + u_raw[:, 1:])
            if v_raw.shape[0] == T.shape[0]:
                # Lat-band sharded layout: the state carries v as its lower
                # n_lat rows (the top pole-face row is dead by the carrier
                # contract, so it is dropped).  The face north of each cell is
                # the next row, which for a band's last row lives on the next
                # band: a one-row shift, which the partitioner turns into an
                # exchange with the neighbouring band -- not a gather of the
                # field.  The global last row takes the dead pole face, 0.
                v_next = jnp.roll(v_raw, -1, axis=0)
                v_next = v_next.at[-1].set(0.0)
                v_c = 0.5 * (v_raw + v_next)
            else:
                # Serial/gathered layout.  The top v-face row is dead by the
                # same carrier contract, but only its MASK is checked when a
                # state is sharded, so a dead face carrying non-zero data would
                # make this disagree with the sharded branch above (which does
                # not have that row at all).  Mask it here rather than assume.
                v_top = v_raw[-1:, :]
                vm = getattr(state, "v_mask", None)
                if vm is not None:
                    m_top = vm.data[-1:, :] > 0.5
                    if m_top.ndim < v_top.ndim:
                        m_top = m_top[..., jnp.newaxis]
                    v_top = jnp.where(m_top, v_top, 0.0)
                else:
                    v_top = jnp.zeros_like(v_top)
                v_c = 0.5 * (v_raw[:-1, :]
                             + jnp.concatenate([v_raw[1:-1, :], v_top], axis=0))
        else:
            u_c = u_raw
            v_c = v_raw
        speed_3d = jnp.sqrt(u_c**2 + v_c**2)
        max_u = float(jnp.max(speed_3d))

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
        U_bar = jnp.mean(u_c, axis=-1)
        V_bar = jnp.mean(v_c, axis=-1)
        ke_baro = 0.5 * (U_bar**2 + V_bar**2)
        ke_3d = 0.5 * speed_3d**2
        ke_baro_total = float(jnp.sum(jnp.where(wet, ke_baro, 0.0))) * nlev_state
        ke_3d_total = float(
            jnp.sum(jnp.where(wet[..., jnp.newaxis], ke_3d, 0.0)))
        pbt = ke_baro_total / max(ke_3d_total, 1e-30)
        speed_masked = jnp.where(wet[..., jnp.newaxis], speed_3d, -1.0)
        idx = np.unravel_index(int(jnp.argmax(speed_masked)), speed_3d.shape)
        j_max, i_max = int(idx[0]), int(idx[1])

    return {
        "SST": sst, "SSS": sss, "SSH": ssh,
        "max_speed": max_u if grid_type != "spectral" else 0.0,
        "P_bt": float(pbt),
        "j_maxu": j_max,
        "i_maxu": i_max,
    }


# ===========================================================================
# Partial-cell bottom geometry (shared by the realistic-bathymetry lat-lon
# lane and the tripole NEMO-mesh lane)
# ===========================================================================

_PARTIAL_CELL_THIN_THRESHOLD = 0.3


def _snap_thin_partial_cells(H_bathy, land_mask, z_coord,
                             thin_threshold: float = _PARTIAL_CELL_THIN_THRESHOLD):
    """Round bathymetry up to the interface above where the partial cell would
    be thinner than ``thin_threshold`` of the reference layer, and turn any
    column that loses its last level into land.

    Thin partial cells at the bottom of deep equatorial columns drove the
    day-13 pressure-gradient instability diagnosed in the 30-day spinup (f is
    near zero there, so geostrophy cannot damp pressure-gradient errors
    quickly).  The snap is the standard MOM6/MITgcm fix, so every column ends
    with either a full bottom cell or a thick-enough partial one.

    Returns ``(H_bathy, land_mask)`` as device arrays and prints what changed.
    """
    H_np = np.asarray(H_bathy, dtype=np.float64)
    abs_z_half = np.abs(np.asarray(z_coord.z_half_ref, dtype=np.float64))
    dz_ref_np = np.asarray(z_coord.dz_ref, dtype=np.float64)   # positive
    # A column deeper than the deepest reference interface matches no layer
    # below, so it would keep a bottom index past the last level.  Clamp it:
    # the reference grid is the deepest ocean this vertical coordinate has.
    n_too_deep = int(np.sum(H_np > abs_z_half[-1]))
    if n_too_deep:
        print(f"  Bathymetry deeper than the reference column "
              f"({abs_z_half[-1]:.1f} m): {n_too_deep} cells clamped")
        H_np = np.minimum(H_np, abs_z_half[-1])
    H_snapped = H_np.copy()
    n_snapped = 0
    for k in range(z_coord.n_levels):
        top = abs_z_half[k]
        bot = abs_z_half[k + 1]
        in_layer = (H_np > top) & (H_np <= bot)
        too_thin = in_layer & ((H_np - top) < thin_threshold * dz_ref_np[k])
        H_snapped = np.where(too_thin, top, H_snapped)
        n_snapped += int(np.sum(too_thin))
    new_land = (H_snapped <= 0.0) & (np.asarray(land_mask) > 0.5)
    n_new_land = int(np.sum(new_land))
    if n_new_land > 0:
        land_mask = jnp.where(jnp.asarray(new_land), 0.0,
                              jnp.asarray(land_mask))
    print(f"  Partial-cell snap (cutoff {thin_threshold * 100:.0f}%): "
          f"{n_snapped} cells snapped, {n_new_land} → land")
    return jnp.asarray(H_snapped, dtype=jnp.float64), jnp.asarray(land_mask)


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


def _large_tripole_io(state, grid_type):
    """Coordinate multi-GB tripole output, leaving small/serial I/O alone.

    A 256 MiB global temperature array implies several GiB of checkpoint
    fields.  Use global shape/dtype metadata only: no gather or host copy.
    """
    return (grid_type == "tripole" and jax.process_count() > 1
            and state.T.data.nbytes >= 256 * 1024**2)


def _collective_root_io(operation):
    """Run root-local I/O while all hosts await its completion together.

    Call on EVERY rank, after any state gathers.  ``operation`` must not
    issue collectives.  A worker keeps serialization off the coordinating
    host thread; a one-second status broadcast keeps peers out of shutdown
    and the next model collective until the write has finished.  Errors are
    raised on every rank.  JAX heartbeat/shutdown deadlines are unchanged.

    This does not implement a filesystem-stall watchdog: a live worker stuck
    in I/O still requires the scheduler's walltime limit or cancellation.
    """
    from jax.experimental import multihost_utils

    root = jax.process_index() == 0
    done = threading.Event()
    result = None
    error = None

    def write():
        nonlocal result, error
        try:
            result = operation()
        except BaseException as exc:
            error = exc
        finally:
            done.set()

    worker = None
    if root:
        worker = threading.Thread(target=write, name="omip-collective-io")
        try:
            worker.start()
        except BaseException as exc:   # thread exhaustion: publish it, do not skip the collective
            error, worker = exc, None
            done.set()
    while True:
        # Only root waits; other ranks enter the same small collective.
        # There is never a collective spanning the entire file write.
        if root:
            done.wait(timeout=1.0)
        status = (2 if error is not None else 1) if done.is_set() else 0
        status = int(np.asarray(multihost_utils.broadcast_one_to_all(
            np.asarray(status, dtype=np.int32), is_source=root)))
        if status:
            if worker is not None:
                worker.join()
            if status == 2:
                raise RuntimeError("rank-0 state write failed") from error
            return result


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


# Restart provenance for the MPAS SPMD lane (module slot: the loop's restart
# writer has no view of the SPMD layout).  [0] = serial order.
_MPAS_SPMD_N_DEVICES = [0]


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
        # 0 = serial cell order; >1 = MPAS SPMD reordered+padded order for
        # that device count (set once by run_omip_single's SPMD wiring).
        "mpas_spmd_n_devices": int(_MPAS_SPMD_N_DEVICES[0]),
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
        if tuple(data[key].shape) != tuple(obj.data.shape):
            raise ValueError(
                f"Restart ice field {key!r} shape {tuple(data[key].shape)} "
                f"!= run's {tuple(obj.data.shape)} ({restart_path}); a "
                "checkpoint written on a different (e.g. un-padded) grid "
                "cannot resume this run.")
        replacements[f] = obj.replace(
            data=jnp.asarray(data[key], dtype=obj.data.dtype))
    if not replacements:
        return None
    return ice_template._replace(**replacements)


def _assert_restart_geometry_matches(state_in, state_built, *, what: str):
    """Refuse a restart whose bottom geometry differs from the one just built.

    A checkpoint written before the tripole lane moved to partial cells carries
    the OLD bathymetry and land mask.  Restoring them leaves the model stepping
    on a vertical coordinate that was frozen at build time from the NEW snapped
    bathymetry, so every column's bottom cell is silently inconsistent.  Shapes
    match, so nothing else catches it.
    """
    for name in ("H_bathy", "land_mask"):
        a = np.asarray(getattr(state_in, name).data)
        b = np.asarray(getattr(state_built, name).data)
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-9, atol=1e-9):
            n_diff = int(np.sum(~np.isclose(a, b, rtol=1e-9, atol=1e-9))) \
                if a.shape == b.shape else -1
            raise SystemExit(
                f"restart {what}: its {name} differs from the geometry this "
                f"run built ({n_diff} cells differ). The vertical coordinate "
                f"is frozen at build time, so resuming would step on a "
                f"mismatched bottom geometry. Re-run without the restart, or "
                f"reproduce the geometry the restart was written with.")


def _load_restart(restart_path, template_state, grid_type=None,
                  mpas_spmd_n_devices: int = 0):
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
    # MPAS SPMD restarts are in the REORDERED + PADDED cell order of their
    # device count; any other order (serial, or another count) is a
    # silently-scrambled ocean — refuse in both directions.
    _saved_nd = int(data["mpas_spmd_n_devices"]) if "mpas_spmd_n_devices" in data else 0
    if _saved_nd != int(mpas_spmd_n_devices):
        raise ValueError(
            f"Restart {restart_path} was written by an MPAS SPMD run over "
            f"{_saved_nd} device(s) (0 = serial order) but this run uses "
            f"{int(mpas_spmd_n_devices)}; the cell order differs.")
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
        if tuple(data[f].shape) != tuple(obj.data.shape):
            raise ValueError(
                f"Restart field {f!r} shape {tuple(data[f].shape)} != run's "
                f"{tuple(obj.data.shape)} ({restart_path}); a checkpoint "
                "written on a different grid layout (resolution or SPMD "
                "south-padding) cannot resume this run.")
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
                   spmd_shard_stack=None, spmd_gather_ice=None,
                   spmd_shard_ref=None):
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
        if (jra55_state.get("enable_sea_ice", False)
                and not jra55_state.get("spmd_ice_ok", False)):
            raise ValueError(
                "spmd_step + prognostic sea ice needs spmd_gather_ice (the "
                "restart writer gathers the sharded ice tile); run_omip_single "
                "wires it next to spmd_gather.")
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

    # Initial diagnostics — straight off the sharded state (_extract_scalars
    # reduces in place and handles the v_lower carrier), so the run never
    # pays for a global copy of the state to report five scalars.
    scalars = _extract_scalars(state, grid_type, grid, z_coord)
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
    _coordinate_io = _large_tripole_io(state, grid_type)
    if spmd_gather is not None:
        def save_restart(st, *a, **kw):
            gathered = spmd_gather(st)          # collective — ALL ranks
            if spmd_gather_ice is not None and kw.get("ice_state") is not None:
                # The MPAS lane shards the sea-ice tile too; the restart
                # writer's np.asarray would host-fetch remote shards.
                kw = {**kw, "ice_state": spmd_gather_ice(kw["ice_state"])}
            if _coordinate_io:
                def write_and_join():
                    path = _save_restart(gathered, *a, **kw)
                    _join_restart_writer()
                    return path
                return _collective_root_io(write_and_join)
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
        # Only the lat-lon C-grid model has build-once caches to prime; the MPAS
        # model has no such hook (calling it unconditionally aborted every
        # single-device MPAS JRA55 run before its first step).
        if spmd_step is None and hasattr(model, "prime_step_caches"):
            model.prime_step_caches(state)
        # Prognostic slab sea ice (--jra55-sea-ice): the block scan carries
        # (ocean_state, ice_state); thread the ice state across blocks.
        _ice_on = bool(jra55_state.get("enable_sea_ice", False))
        ice_state = jra55_state.get("ice_state_init") if _ice_on else None
        # SPMD aux (codex r18 P1): pass the step's sharded geometry stacks
        # into every block_fn call as an ARGUMENT (see the builders' note).
        _spmd_aux = getattr(spmd_step, "aux", None)
        _spmd_refs = jra55_state.get("_spmd_refs") if spmd_step is not None else None
        # Monthly SSS target / area for the mean removal: selected per block by
        # the block's model day (band-sharded once under lat-band SPMD).
        _sss_shard_fn = spmd_shard_ref if spmd_step is not None else None
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
                if jax.process_count() > 1 and block_start == start_step:
                    # first block: name the forcing leaf bytes per rank so a
                    # gate failure below is attributable (see SPMD build digests)
                    from legoesm.parallel.geometry_consistency import leaf_digest48
                    _lead = raw_stack if use_gpu_interp else atm_stack
                    _k = next(iter(_lead))
                    print(f"  [rank {jax.process_index()}] forcing leaf "
                          f"{_k!r} digest={int(leaf_digest48(_lead[_k]))}",
                          flush=True)
                if use_gpu_interp:
                    raw_stack = spmd_shard_stack(raw_stack)
                    runoff_records = spmd_shard_stack(runoff_records)
                else:
                    atm_stack = spmd_shard_stack(atm_stack)
                    runoff_stack = spmd_shard_stack(runoff_stack)
            io_dt = time.time() - t_io_start
            if block_start == start_step:
                _memprobe("after preload")

            t_compute_start = time.time()
            if use_gpu_interp:
                bfn = _get_block_fn_interp(actual)
                _refs_blk = _refs_for_block(
                    _spmd_refs, jra55_state,
                    float(record_meta["block_start_day"]), shard_fn=_sss_shard_fn)
                if _ice_on:
                    state, ice_state = bfn(
                        state, raw_stack, runoff_records,
                        record_meta["record_days"],
                        jnp.float64(record_meta["block_start_day"]),
                        jnp.float64(record_meta["block_start_day_forcing"]),
                        ice_state, aux=_spmd_aux, refs=_refs_blk,
                    )
                else:
                    state = bfn(
                        state, raw_stack, runoff_records,
                        record_meta["record_days"],
                        jnp.float64(record_meta["block_start_day"]),
                        jnp.float64(record_meta["block_start_day_forcing"]),
                        aux=_spmd_aux, refs=_refs_blk,
                    )
            else:
                _refs_blk = _refs_for_block(
                    _spmd_refs, jra55_state,
                    float(block_start) * dt / 86400.0, shard_fn=_sss_shard_fn)
                if _ice_on:
                    state, ice_state = block_fn(
                        state, atm_stack, runoff_stack,
                        jnp.int32(block_start), ice_state, aux=_spmd_aux,
                        refs=_refs_blk,
                    )
                else:
                    state = block_fn(
                        state, atm_stack, runoff_stack,
                        jnp.int32(block_start), aux=_spmd_aux,
                        refs=_refs_blk,
                    )
            jax.block_until_ready(state.T.data)
            compute_dt = time.time() - t_compute_start
            if block_start == start_step:
                _memprobe("after block 1")

            block_start += actual
            step = block_start
            day = step * dt / 86400.0

            if not _check_finite(state, grid_type):
                print(f"  BLOWUP at step {step}")
                _report_forcing_extremes(
                    raw_stack if use_gpu_interp else atm_stack, "block forcing")
                _report_nonfinite(state, grid_type, grid, spmd_gather)
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

            # The scalars come off the sharded state directly; only the chi
            # diagnostic below (regular lat-lon lane) needs a host-side eta,
            # so that is the only case that still pays for a gather.
            _st_diag = (spmd_gather(state)
                        if spmd_gather is not None and grid_type == "latlon"
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
            if scalars.get("j_maxu", -1) >= 0:
                scalar_summary += f" ij=({scalars['j_maxu']},{scalars['i_maxu']})"
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
            scalars = _extract_scalars(state, grid_type, grid, z_coord)
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
    if run_config.multicontroller and not (run_config.enable_latlon_spmd
                                           or run_config.enable_mpas_spmd):
        raise SystemExit(
            "--multicontroller requires --enable-latlon-spmd or "
            "--enable-mpas-spmd (it is the "
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
    run_config = with_kpp_cfl_dt(run_config, dt)
    days = 30.0 if args.quick else args.days
    n_steps = int(days * 86400.0 / dt)
    diag_every = args.diag_every or max(1, int(86400.0 / dt))  # ~daily

    print(f"\n{'='*70}")
    print(f"  OMIP: {grid_type} | {resolution} | {args.nlev} levels | "
          f"dt={dt:.0f}s | {days:.0f} days ({n_steps} steps)")
    print(f"  Physics: {args.physics} | SW: {args.sw_down} W/m² | "
          f"Water type: {args.water_type}")
    print(f"  KPP CFL cap timestep (from dt): "
          f"{run_config.vertical_mixing.kpp.cfl_cap_dt_s:.0f} s")
    print(f"{'='*70}")

    t_setup = time.time()

    # Create grid + model (all grids use identical config-based diffusion
    # for cross-grid consistency; physics pipeline disabled).
    _dz_ref = load_dz_ref_file(args.dz_ref_file)
    _validate_dz_ref_against_setup(_dz_ref, args.nlev, args.H_max)
    # #1370 host-side build (the lat-lon SPMD bench's mechanism, reused): under
    # lat-band SPMD every process builds the GLOBAL grid/model/state/WOA/
    # forcing setup itself and only its own band shards may reach the GPU.
    # Built on the default GPU device, ORCA12 (13.3M x 75) put every GPU at
    # 66-75 GB before the first step (job 27325458) and OOMed at the forcing
    # shard. So the whole setup below runs with the CPU backend as jax's
    # default device (host RAM); shard_state_latlon / shard_cell_pytree_latlon
    # / checked_shard_put then hand each device exactly its band slabs
    # (make_array_from_callback, no global device copy). Exited right after
    # the sharding block so the jitted block scan runs on the GPUs.
    import contextlib
    _build_ctx = contextlib.nullcontext()
    if _spmd_device_count(run_config) > 1:
        try:
            _build_ctx = jax.default_device(
                jax.local_devices(backend="cpu")[0])
        except RuntimeError:
            print("[#1370] WARNING: no cpu backend — the global setup will "
                  "materialise on the accelerator (set JAX_PLATFORMS=cuda,cpu "
                  "to enable the host-side build)", flush=True)
    _build_ctx.__enter__()
    global _ACTIVE_BUILD_CTX
    _ACTIVE_BUILD_CTX = _build_ctx
    _mpas_spmd_nd = 0
    if run_config.enable_mpas_spmd:
        if grid_type != "mpas":
            raise SystemExit(
                f"--enable-mpas-spmd requires --grid mpas (got {grid_type}).")
        if run_config.spmd_n_devices < 0:
            raise SystemExit(
                f"--spmd-n-devices must be >= 0 (got {run_config.spmd_n_devices}).")
        if run_config.multicontroller:
            _mpas_spmd_nd = len(jax.devices())
        else:
            _mpas_spmd_nd = run_config.spmd_n_devices or len(jax.devices())
    # Restart provenance for THIS run (0 = serial cell order): set for every
    # run so an earlier SPMD run in the same interpreter cannot leak its count.
    _MPAS_SPMD_N_DEVICES[0] = int(_mpas_spmd_nd) if _mpas_spmd_nd > 1 else 0
    if getattr(args, "B_h_gamma0", None) is not None and grid_type != "tripole":
        raise SystemExit(
            f"--B-h-gamma0 is wired for --grid tripole only (2-D face metrics); "
            f"on --grid {grid_type} it would be silently inert. Use --B-h.")
    grid, z_coord, config, model, coord_kind = _create_setup(
        grid_type, resolution, args.nlev, args.H_max,
        args.physics, args.water_type,
        sw_scheme=getattr(args, "sw_penetration", "auto"),
        B_h_gamma0=getattr(args, "B_h_gamma0", None),
        spmd_n_devices=_mpas_spmd_nd,
        mpas_lloyd=int(getattr(args, "mpas_lloyd", 50)),
        mpas_k_zeta_bih=getattr(args, "mpas_k_zeta_bih", None),
        use_bathymetry=(args.bathymetry is not None),
        tripole_mesh=getattr(args, "tripole_mesh", None),
        tripole_strip_north_rows=int(
            getattr(args, "tripole_strip_north_rows", 0) or 0),
        tripole_fold_convention=getattr(
            args, "tripole_fold_convention", "auto"),
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
        dz_ref_override=_dz_ref,
    )
    # NEMO zdfdrg drag-law + zdfiwm forcing-map overrides (no-op when the
    # flags are at their legacy defaults; rebuilds the model so the jitted
    # step captures the new config / maps).
    _tripole_land_mask = None
    _tripole_H_bathy = None
    if grid_type == "tripole":
        # The tripole lane takes land mask + bathymetry from NEMO's own mesh
        # (the same reader the validated run_omip_core2 lane uses); before
        # this the JRA55 tripole lane ran FLAT-BOTTOM ALL-OCEAN because
        # --bathymetry is refused here and nothing else set them.
        from legoesm.grids.tripole import (
            mesh_file_list, pad_mask_bathy_south, pad_tripole_grid_south,
            south_pad_rows,
        )
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.init_tripole import read_mesh_mask_bathy
        _mesh_files = mesh_file_list(
            getattr(args, "tripole_mesh", None)
            or _parse_resolution(grid_type, resolution)["mesh_path"])
        _closed = tuple(
            n for n in (getattr(args, "tripole_closed_seas", None) or "").split(",") if n)
        _lm, _hb = read_mesh_mask_bathy(
            _mesh_files,
            strip_north_rows=int(
                getattr(args, "tripole_strip_north_rows", 0) or 0),
            closed_seas=_closed)
        _n_lat, _n_lon = int(grid.lat_T.shape[0]), int(grid.lat_T.shape[1])
        if _lm.shape != (_n_lat, _n_lon):
            raise SystemExit(
                f"tripole mesh mask shape {_lm.shape} != grid "
                f"{(_n_lat, _n_lon)} ({_mesh_files})")
        # Lat-band SPMD needs n_lat % n_devices == 0: prepend LAND rows at
        # the south (the fold is north-relative) and rebuild the model on
        # the padded grid -- the run_omip_core2 --n-gpus recipe.
        _n_pad = south_pad_rows(_n_lat, _spmd_device_count(run_config))
        if _n_pad > 0:
            grid = pad_tripole_grid_south(grid, _n_pad)
            _lm, _hb = pad_mask_bathy_south(_lm, _hb, _n_pad)
            # Plain rebuild on the padded grid; the drag / IWM overrides run
            # AFTER this block so their grid-sized maps see the padded grid.
            model = LatLonCGridOceanModel(grid, z_coord, config)
            print(f"  SPMD south-pad: +{_n_pad} LAND rows -> n_lat="
                  f"{int(grid.n_lat)} (fold still north at "
                  f"j={int(grid.fold.fold_j)})")
        _tripole_land_mask = jnp.asarray(_lm, dtype=jnp.float64)
        _tripole_H_bathy = jnp.asarray(_hb, dtype=jnp.float64)
        _n_wet = int(np.sum(_lm > 0.5))
        print(f"  Tripole mesh: {[Path(f).name for f in _mesh_files]} "
              f"({_n_wet}/{_lm.size} ocean cells, "
              f"H_max={float(np.max(_hb)):.0f} m, closed seas masked: "
              f"{list(_closed) or 'none'})")
        if getattr(args, "tripole_partial_cells", True) and args.bathymetry is None:
            # True depth levels with a partial bottom cell, as NEMO's zgr_zps
            # builds them.  The reference thicknesses and T-point depths come
            # from the mesh itself (never reconstructed: the T-point depths
            # decide which level is the bottom one on a stretched grid), so the
            # model's vertical grid is NEMO's own.
            from legoesm.ocean.init_tripole import read_mesh_vertical_1d
            from legoesm.ocean.vertical import (
                create_partial_cell_coordinate, create_z_star_from_thicknesses,
            )
            _e3t_1d, _gdept_1d, _ = read_mesh_vertical_1d(_mesh_files)
            if _e3t_1d.size != int(z_coord.n_levels):
                raise SystemExit(
                    f"--grid tripole partial cells: mesh has {_e3t_1d.size} "
                    f"levels but the run was built with {int(z_coord.n_levels)}"
                    f"; pass --nlev {_e3t_1d.size} (or "
                    f"--no-tripole-partial-cells)")
            # The mesh becomes the source of truth for the vertical grid, so a
            # --dz-ref-file that disagrees with it must not be silently
            # discarded: it would leave the run's provenance describing a
            # vertical grid the model never used.
            # Micron tolerance: the point is to catch a DIFFERENT vertical
            # grid, not the rounding of a text file written from this same
            # mesh (measured 5e-7 m for the ORCA12 dz file).
            if _dz_ref is not None and not np.allclose(
                    np.asarray(_dz_ref, dtype=np.float64), _e3t_1d,
                    rtol=1e-6, atol=1e-6):
                raise SystemExit(
                    "--dz-ref-file disagrees with the tripole mesh's own "
                    "e3t_1d; drop the flag (the mesh supplies the reference "
                    "thicknesses) or pass --no-tripole-partial-cells")
            _mesh_depth = float(np.sum(_e3t_1d))
            if abs(_mesh_depth - float(args.H_max)) > 1e-6 * _mesh_depth:
                raise SystemExit(
                    f"--H-max {float(args.H_max):.6f} m disagrees with the "
                    f"tripole mesh's reference depth {_mesh_depth:.6f} m "
                    f"(sum of e3t_1d); pass --H-max {_mesh_depth:.6f}")
            z_coord = create_z_star_from_thicknesses(
                _e3t_1d, _gdept_1d, nemo_e3w_source="depth_difference")
            _tripole_H_bathy, _tripole_land_mask = _snap_thin_partial_cells(
                _tripole_H_bathy, _tripole_land_mask, z_coord)
            # Reassign z_coord, do not keep a second coordinate alive: the
            # eta-stretched operators, the climatology initialisation, the rest
            # state, the restart writer and the multi-GPU band slicing all read
            # this name, and they must see the same bottom geometry as the
            # bathymetry (the tripole ETOPO path below does the same).
            _z_star_for_snap = z_coord
            z_coord = create_partial_cell_coordinate(
                _z_star_for_snap, _tripole_H_bathy,
                bottom_index_rule="nemo_tpoint")
            model = LatLonCGridOceanModel(
                grid, z_coord, config,
                iwm_forcing=getattr(model, "_iwm_forcing", None))
            _thin = float(np.min(np.asarray(z_coord.dz_ref)))
            print(f"  Tripole partial cells (NEMO zgr_zps T-point rule): "
                  f"{int(z_coord.n_levels)} reference levels, top "
                  f"{float(np.asarray(z_coord.dz_ref)[0]):.3f} m, thinnest "
                  f"reference layer {_thin:.3f} m")
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
    # DEFERRED: the observed profiles are placed on the columns only after the
    # bathymetry (and therefore the partial-cell geometry) is final -- see the
    # init_ocean_from_woa call further down, after the model is built.
    # Sampling them here, on the reference z* levels, put a cut bottom cell's
    # water at the wrong depth.
    _woa_paths = (args.woa_t, args.woa_s)
    T_woa = S_woa = None

    # Bathymetry: realistic (ETOPO) or flat-bottom.
    H_bathy_init = _tripole_H_bathy
    land_mask_init = _tripole_land_mask
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
        _ncap = float(getattr(args, "north_cap_lat", 90.0))
        _cap_str = (f"N-cap={_ncap:g}°N" if _ncap < 90.0
                    else "N-cap=NONE (pole open)")
        print(f"  Bathymetry: {Path(args.bathymetry).name} "
              f"({n_ocean}/{n_total} ocean cells, "
              f"H_min={args.H_min}m, {args.smoothing_passes} smoothing passes, "
              f"r_max={args.r_factor_max}, {_cap_str})")
        # An uncapped pole on a GLOBAL regular lat-lon grid is a CFL trap:
        # dx = R*dlon*cos(lat) collapses to ~970 m at 89.5 N, so a modest
        # velocity there is wildly supercritical (measured: |u| = 19 m/s ->
        # advective CFL 47 within 3 steps of a rest state at dt = 2400 s;
        # capping at 80 N left |u|max = 0.014 m/s).  The tripole path already
        # reported its cap; this lane did not, so an open pole was SILENT.
        if _ncap >= 90.0 and grid_type == "latlon":
            print("  WARNING: North Pole is UNCAPPED on a regular lat-lon "
                  "grid — dx ~ 970 m at 89.5°N. Pass --north-cap-lat (e.g. 80) "
                  "unless the polar dynamics are known to be handled.")

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
        H_bathy_init, land_mask_init = _snap_thin_partial_cells(
            H_bathy_init, land_mask_init, z_coord)
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
    if _mpas_spmd_nd > 1:
        from legoesm.parallel.voronoi_spmd_ocean import (
            mask_padded_cells, n_real_cells,
        )
        state = mask_padded_cells(state, n_real_cells(grid))
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
        if _mpas_spmd_nd > 1:
            # padded ghost cells sit at lat/lon 0 and would read ETOPO depth
            from legoesm.parallel.voronoi_spmd_ocean import n_real_cells
            _nr = n_real_cells(grid)
            H_bathy_raw = H_bathy_raw.at[_nr:].set(0.0)
            ocean_mask = ocean_mask.at[_nr:].set(0.0)

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

    # Place the observed profiles at each cell's TRUE centre depth.  This runs
    # UNCONDITIONALLY, exactly as the original load did: the restoring targets
    # (the DEFAULT forcing mode), the JRA55 sponge / salinity-restoring targets
    # and --nudge-woa-tau all read these fields whether or not --woa-init is
    # set.  ``model.z_coord`` is the coordinate the run integrates on (partial
    # cells included) and ``compute_centroid_depth`` is the package's own helper
    # for that depth -- the same one the seamount rest test uses.
    from legoesm.ocean.vertical import OceanPartialCellCoordinate
    _zc_final = getattr(model, "z_coord", z_coord)
    _cell_depths = None
    if isinstance(_zc_final, OceanPartialCellCoordinate):
        from legoesm.ocean.vertical import compute_centroid_depth
        _H = jnp.asarray(state.H_bathy.data, dtype=jnp.float64)
        # eta = 0 at initialisation; land columns are clamped so the helper's
        # (eta + H)/H factor stays finite -- their values are masked out below.
        _cell_depths = np.asarray(compute_centroid_depth(
            jnp.zeros_like(_H), jnp.maximum(_H, 1e-3), _zc_final))
        # Report the offset over ACTIVE wet cells only: cells below the
        # seafloor inherit the bottom depth and would otherwise dominate.
        _zref = np.abs(np.asarray(_zc_final.z_full_ref))
        # ``[..., None]`` and not ``[:, None]``: the horizontal layout is
        # (nCells,) on the icosahedral mesh but (n_lat, n_lon) on the lat-lon
        # and tripole lanes, where a leading-axis insert cannot broadcast
        # against (n_lat, n_lon, nlev) and aborted the run outright.
        _live = (np.asarray(state.land_mask.data) > 0.5)[..., None] & (
            np.asarray(_zc_final.is_active) > 0.5)
        _off = np.abs(_cell_depths - _zref[None, :])[_live]
        print("  WOA placement: per-cell centroid depths (partial cells); "
              f"sampling depth moved vs the reference centres by "
              f"{float(np.median(_off)):.1f} m median, "
              f"{float(np.max(_off)):.1f} m max")
        del _live, _off, _zref, _H
    if getattr(args, "ic_from_fesom_mesh", None):
        from legoesm.ocean.init_woa import init_ocean_from_fesom_mesh
        T_woa, S_woa = init_ocean_from_fesom_mesh(
            grid, _zc_final, args.ic_from_fesom_mesh,
            cell_center_depths=_cell_depths,
            wet_mask=np.asarray(state.land_mask.data) > 0.5,
            log=lambda m: print("  " + m, flush=True),
            cache_dir=getattr(args, "ic_cache_dir", None))
    else:
        T_woa, S_woa = init_ocean_from_woa(
            grid, _zc_final, _woa_paths[0], _woa_paths[1],
            cell_center_depths=_cell_depths,
            void_fill=bool(getattr(args, "woa_void_fill", False)))
    del _cell_depths   # a full-global (nj, ni, nlev) f64 array per rank at ORCA12

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
    # Prognostic-TKE scan carry: the None -> Field promotion happens HERE,
    # once -- BEFORE the restart load (the loader restores only the fields
    # the template carries, so an unseeded template would drop a saved tke)
    # and before SPMD sharding / the first scanned step, so the carry pytree
    # is stable.  Both seeders are no-ops unless the prognostic TKE closure
    # is active.
    if grid_type == "mpas":
        state = model.seed_tke(state)
    elif getattr(model, "_tke_prognostic_active", None) is not None \
            and model._tke_prognostic_active():
        state = model.seed_scan_carry(state, dt)
    if args.restart is not None:
        _state_built = state
        state, restart_day, restart_step = _load_restart(
            args.restart, state, grid_type=grid_type,
            mpas_spmd_n_devices=_MPAS_SPMD_N_DEVICES[0],
        )
        if grid_type == "tripole":
            _assert_restart_geometry_matches(
                state, _state_built, what=str(Path(args.restart).name))
        start_step = restart_step
        print(f"  Restart: loaded day {restart_day:.1f} (step {restart_step}) "
              f"from {Path(args.restart).name}")

    # --enable-latlon-spmd preconditions that are knowable from ARGS: refuse
    # BEFORE the JRA55 forcing setup below builds caches/state (codex r1 #1)
    # and before any device work; a negative device count would otherwise
    # silently no-op through the `_nd or len(devices)` resolution (r1 #3).
    if run_config.enable_mpas_spmd and getattr(args, "jra55_sea_ice", False):
        if str(args.ice_dynamics) != "none" or int(args.ice_categories) != 1:
            raise SystemExit(
                "--enable-mpas-spmd supports --jra55-sea-ice only as the "
                "pointwise thermodynamic tile (--ice-dynamics none, "
                "--ice-categories 1); ice rheology needs per-subcycle halo "
                f"refreshes that are not built (got --ice-dynamics "
                f"{args.ice_dynamics} --ice-categories {args.ice_categories}).")
    if run_config.enable_latlon_spmd:
        if getattr(args, "jra55_sea_ice", False):
            _spmd_sea_ice_guard(str(args.ice_dynamics),
                                int(args.ice_categories))
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
        if (run_config.enable_latlon_spmd
                and getattr(args, "jra55_sea_ice", False)):
            # _spmd_sea_ice_guard above is this lane's admission test: what
            # survives it is the elementwise thermodynamic slab, which the
            # lat-band lane DOES carry.  Mark it so the shared refusals (added
            # on main for the MPAS lane, which gates its own tile the same
            # way) do not reject a combination this lane supports.
            jra55_state["spmd_ice_ok"] = True
        # River-runoff routing needs the model's OWN land mask, which only
        # exists once the state is built, so the map is assembled here rather
        # than in _setup_jra55_forcing_state.
        jra55_state["runoff_map"] = _build_runoff_map_for_run(
            args, grid, grid_type, jra55_state["_ocean_mask_2d"],
        )
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
        "kpp_cfl_cap_dt_s": float(run_config.vertical_mixing.kpp.cfl_cap_dt_s),
        "days": float(args.days),
        "seed": int(run_config.seed),
        "forcing_mode": getattr(args, "forcing_mode", "restoring"),
        "initial_condition": (("fesom_mesh:" + str(args.ic_from_fesom_mesh))
                              if (args.woa_init and getattr(args, "ic_from_fesom_mesh", None))
                              else "woa18" if args.woa_init else "rest_state"),
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
    spmd_gather_ice = None
    spmd_shard_stack = None
    spmd_shard_ref = None
    if run_config.enable_latlon_spmd:
        if grid_type not in ("latlon", "tripole"):
            raise SystemExit(
                f"--enable-latlon-spmd requires --grid latlon or tripole "
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
        _nd = _spmd_device_count(run_config)
        if _nd > 1:
            if grid.n_lat % _nd != 0:
                raise SystemExit(
                    f"--enable-latlon-spmd: n_lat ({grid.n_lat}) not "
                    f"divisible by the device count ({_nd}); pick "
                    f"--spmd-n-devices dividing n_lat.")
            from functools import partial

            from legoesm.ocean.dynamics.sharded_ocean_step import (
                gather_cell_pytree_latlon,
                gather_state_latlon,
                make_sharded_ocean_step,
                shard_cell_pytree_latlon,
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
            if (jra55_state is not None
                    and jra55_state.get("enable_sea_ice", False)):
                # The slab ice tile (elementwise; guarded above) rides next to
                # the ocean state in the block scan: lay it out on the same
                # lat bands so the jitted scan sees ONE consistent sharding
                # (a replicated/host tile next to process-spanning shards
                # is an error under route-B).
                jra55_state["ice_state_init"] = shard_cell_pytree_latlon(
                    jra55_state["ice_state_init"], _dev.mesh)
                spmd_gather_ice = partial(
                    gather_cell_pytree_latlon, mesh=_dev.mesh)
            if jra55_state is not None:
                # The block scan's reference fields (sponge gamma / T_ref /
                # S_ref, SSS target) go in as the ``refs`` ARGUMENT, lat-band
                # sharded like the forcing stack -- not as closure constants
                # replicated whole on every GPU (ORCA12: ~16 GB per device).
                from legoesm.ocean.dynamics.sharded_ocean_step import (
                    shard_forcing_latlon,
                )
                _refs = {}
                for _key, _name in (("sponge_gamma_2d", "sponge_gamma"),
                                    ("sponge_T_ref_3d", "sponge_T_ref"),
                                    ("sponge_S_ref_3d", "sponge_S_ref"),
                                    ("sss_target_2d", "sss_target"),
                                    ("_ocean_mask_2d", "ocean_mask")):
                    if jra55_state.get(_key) is not None:
                        _refs[_name] = shard_forcing_latlon(
                            jnp.asarray(jra55_state[_key]), _dev.mesh)
                jra55_state["_spmd_refs"] = _refs or None
                spmd_shard_ref = partial(shard_forcing_latlon, mesh=_dev.mesh)
            # Lay per-block forcing stacks out lat-band-sharded so the
            # in-scan interpolation / bulk fluxes stay shard-local (shared
            # layout helper — see shard_forcing_stack_latlon).
            spmd_shard_stack = partial(
                shard_forcing_stack_latlon, mesh=_dev.mesh)
            if _multi:
                # Per-rank digests of the per-process build inputs the forcing
                # gate (shard_forcing_stack_latlon) later requires to agree:
                # when it fires, these lines say WHICH upstream object differed.
                from legoesm.parallel.geometry_consistency import leaf_digest48
                _glat, _glon = ((grid.lat_T, grid.lon_T) if grid_type == "tripole"
                                else (grid.lat, grid.lon))
                _d = {"lat_T": leaf_digest48(_glat),
                      "lon_T": leaf_digest48(_glon),
                      "land_mask": leaf_digest48(state.land_mask.data)
                      if jax.process_count() == 1 else float("nan")}
                _rw = (jra55_state or {}).get("regrid_weights")
                if _rw is not None:
                    _d["regrid_idx"] = leaf_digest48(_rw.src_indices)
                    _d["regrid_w"] = leaf_digest48(_rw.weights)
                print(f"  [rank {jax.process_index()}] SPMD build digests: "
                      + " ".join(f"{k}={int(v) if v == v else 'sharded'}"
                                 for k, v in _d.items()), flush=True)
            if jax.process_index() == 0:
                _lane = "route-B multicontroller" if _multi else "single-controller"
                print(f"  SPMD ({_lane}): lat-band sharded dynamics step over "
                      f"{_nd} devices across {jax.process_count()} process(es) "
                      f"({jax.default_backend()}).")
        elif jax.process_index() == 0:
            print("  SPMD: single device visible — flag is a no-op.")
    _exit_build_ctx()                          # end of the #1370 host-side build
    _memprobe("after setup")

    # Run time loop
    if run_config.enable_mpas_spmd:
        _nd = _mpas_spmd_nd
        if jax.process_count() > 1 and not run_config.multicontroller:
            raise SystemExit(
                "--enable-mpas-spmd is single-controller only unless "
                "--multicontroller is set (route-B jax.distributed).")
        if (run_config.multicontroller and run_config.spmd_n_devices
                and run_config.spmd_n_devices != _nd):
            raise SystemExit(
                f"--multicontroller uses ALL global devices ({_nd}); "
                f"--spmd-n-devices ({run_config.spmd_n_devices}) must be 0 or {_nd}.")
        if jra55_state is not None:
            if jra55_state.get("_use_single_step", False):
                raise SystemExit(
                    "--enable-mpas-spmd requires the JRA55 block-scan path "
                    "(this run selected the single-step fallback).")
            if jra55_state.get("enable_sea_ice", False):
                jra55_state["spmd_ice_ok"] = True   # gated above: thermo-only tile
        if _nd > 1:
            from functools import partial
            from legoesm.parallel.voronoi_spmd_ocean import (
                build_mpas_ocean_spmd_layout,
                gather_state_mpas_ocean_spmd,
                make_sharded_mpas_ocean_step,
                n_real_cells,
                shard_cell_stack_spmd,
                shard_state_mpas_ocean_spmd,
            )
            _layout = build_mpas_ocean_spmd_layout(
                grid, _nd, n_cells_real=n_real_cells(grid),
                tracer_advection=str(model.config.tracer_advection),
                nlev=int(args.nlev))
            spmd_step = make_sharded_mpas_ocean_step(model, _layout)
            spmd_gather = partial(gather_state_mpas_ocean_spmd, layout=_layout)
            spmd_gather_ice = spmd_gather      # generic pytree gather
            spmd_shard_stack = partial(shard_cell_stack_spmd, layout=_layout)
            # The per-block SSS references (monthly target (nCells,), area)
            # must be block-sharded like the forcing stacks under MPAS SPMD,
            # not handed to the jitted block as global arrays;
            # shard_cell_stack_spmd handles 1-D per-cell leaves.
            spmd_shard_ref = spmd_shard_stack
            state = shard_state_mpas_ocean_spmd(state, _layout)
            if jra55_state is not None and jra55_state.get("ice_state_init") is not None:
                jra55_state["ice_state_init"] = shard_cell_stack_spmd(
                    jra55_state["ice_state_init"], _layout)
            if jax.process_index() == 0:
                _lane = ("route-B multicontroller" if run_config.multicontroller
                         else "single-controller")
                print(f"  SPMD ({_lane}): MPAS owned-block sharded step over "
                      f"{_nd} devices across {jax.process_count()} process(es) "
                      f"({jax.default_backend()}); cells/device={_layout.cells_per}, "
                      f"local incl. halo={_layout.max_lc}, ppermute rounds="
                      f"{len(_layout.ppermute_perms)}.")
        elif jax.process_index() == 0:
            print("  SPMD (mpas): single device visible — flag is a no-op.")
    # Disarm the process-global SPMD backend on EVERY exit of the loop (a
    # direct run_omip_single() caller has no main()-loop finally).
    try:
        state, diag, wall_time, ok, blowup_info = _run_omip_loop(
            model, state, grid_type, grid, z_coord,
            dt, n_steps, diag_every,
            label=f"{grid_type}/{resolution}",
            restoring_targets=restoring_targets,
            restoring_tau_s=restoring_tau_s,
            restoring_ramp_days=ramp_days_eff,
            jra55_state=jra55_state,
            spmd_shard_ref=spmd_shard_ref,
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
            spmd_gather_ice=spmd_gather_ice,
        )
        if spmd_gather is not None:
            # Downstream report/plot/save paths expect the full (n_lat+1)
            # staggered v layout, not the sharded v_lower carry.  Every rank
            # dispatches this gather (it is a collective); only rank 0 writes.
            state = spmd_gather(state)
    finally:
        if run_config.enable_mpas_spmd and spmd_step is not None:
            from legoesm.parallel.voronoi_spmd_ocean import disarm_mpas_ocean_spmd
            disarm_mpas_ocean_spmd()

    # Surface a failed FINAL async restart write while the run can still
    # report it (the writer thread swallows exceptions; _save_restart only
    # re-raises them one checkpoint later — there is no later checkpoint
    # for the last one; codex audit HIGH, 2026-07-02).  Placed AFTER the
    # spmd_gather collective above: under route-B multiproc only rank 0
    # spawns a writer thread, so a rank-0-only re-raise here must not be able
    # to skip that collective and hang the federation.
    _coordinate_io = _large_tripole_io(state, grid_type)
    if _coordinate_io:
        _collective_root_io(_join_restart_writer)
    else:
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
    def write_final_snapshot():
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

    if args.no_final_snapshot:
        if _io_rank:
            print("  MLD snapshot skipped (--no-final-snapshot)")
    elif _coordinate_io:
        _collective_root_io(write_final_snapshot)
    elif _io_rank:
        write_final_snapshot()

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
        if not _EARLY_FEDERATED:
            # multicontroller selected by a --config file, an abbreviated flag,
            # or any path the argv pre-parse cannot see: the legoESM imports
            # above have already initialised the XLA backend, so
            # jax.distributed would refuse.
            raise SystemExit(
                "--multicontroller must be given ON THE COMMAND LINE, spelled "
                "out in full (it is acted on before the legoESM imports; a "
                "config-file `multicontroller: true` or an abbreviation is "
                "seen too late to federate).")
        if (getattr(args, "coordinator", None) or None) != (_pre_coord or None):
            raise SystemExit(
                f"--coordinator resolved to {args.coordinator!r} after parsing "
                f"but the early federation used {_pre_coord!r}: pass "
                "--coordinator on the command line (a config-file value is "
                "seen too late).")
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
        except SystemExit:
            # A refused configuration (SystemExit) can be raised INSIDE the
            # host-side build context, and SystemExit is not an Exception, so
            # the handler below would not run: the CPU would stay the default
            # device for the next grid and for anything else in this process.
            _exit_build_ctx()
            raise
        except Exception:
            _exit_build_ctx()      # never leave the CPU as default for the next grid
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
        finally:
            if getattr(args, "enable_mpas_spmd", False):
                # The MPAS SPMD layout arms a process-global "spmd" halo backend
                # + device mesh; a following grid / serial run in this process
                # must not inherit it (psum outside shard_map).
                from legoesm.parallel.voronoi_spmd_ocean import disarm_mpas_ocean_spmd
                disarm_mpas_ocean_spmd()

    if _root:
        print_summary()

    # Exit with error if any failed (ALL ranks — the launcher needs a
    # consistent per-process exit code, so this is NOT rank-0-gated).
    if any(r["status"] != "PASS" for r in ALL_RESULTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
