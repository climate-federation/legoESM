"""Documented tuning guide as executable code.

Provides a registry of key tunable parameters with valid ranges,
sensitivities, and descriptions, plus helpers that validate an
experiment configuration and suggest resolution-appropriate defaults.
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm.forcing.amip_config import AMIPExperimentConfig


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

class TuningParameter(NamedTuple):
    """Metadata for a single tunable model parameter."""

    name: str
    default: float
    min_val: float
    max_val: float
    units: str
    description: str
    category: str       # "dynamics", "radiation", "convection", "diffusion", "surface"
    sensitivity: str    # "high", "medium", "low"
    notes: str


# ---------------------------------------------------------------------------
# Parameter registry
# ---------------------------------------------------------------------------

TUNING_PARAMETERS: dict[str, TuningParameter] = {
    # -- Dynamics --------------------------------------------------------
    "dt": TuningParameter(
        name="dt",
        default=600.0,
        min_val=300.0,
        max_val=1200.0,
        units="s",
        description="Integration time step",
        category="dynamics",
        sensitivity="high",
        notes=(
            "Must satisfy CFL: dt < 0.8 * dx_min / (u_max + sqrt(gH)). "
            "Halve when doubling resolution."
        ),
    ),
    "hyperdiff_scale": TuningParameter(
        name="hyperdiff_scale",
        default=5e16,
        min_val=1e15,
        max_val=1e18,
        units="m^4/s",
        description="Hyper-diffusion coefficient scale",
        category="dynamics",
        sensitivity="high",
        notes=(
            "Controls small-scale noise removal. Use physical e-folding "
            "time: nu4 = dx^4 / tau_efold. Too large causes overdamping; "
            "too small lets grid-scale noise grow."
        ),
    ),

    # -- Radiation -------------------------------------------------------
    "tau_equator": TuningParameter(
        name="tau_equator",
        default=7.2,
        min_val=5.0,
        max_val=10.0,
        units="1",
        description="Gray radiation optical depth at equator",
        category="radiation",
        sensitivity="medium",
        notes="Controls tropical longwave cooling. Higher = warmer tropics.",
    ),
    "tau_pole": TuningParameter(
        name="tau_pole",
        default=1.8,
        min_val=1.0,
        max_val=3.0,
        units="1",
        description="Gray radiation optical depth at poles",
        category="radiation",
        sensitivity="medium",
        notes="Controls polar longwave cooling. Lower = colder poles.",
    ),
    "S_0": TuningParameter(
        name="S_0",
        default=1360.0,
        min_val=1340.0,
        max_val=1380.0,
        units="W/m^2",
        description="Solar constant",
        category="radiation",
        sensitivity="low",
        notes="Total solar irradiance. Standard value ~1361 W/m^2.",
    ),
    "rad_update_steps": TuningParameter(
        name="rad_update_steps",
        default=1,
        min_val=1.0,
        max_val=12.0,
        units="steps",
        description="Radiation call cadence (every N time steps)",
        category="radiation",
        sensitivity="medium",
        notes=(
            "Radiation is expensive; calling less often saves compute "
            "but may miss fast-evolving cloud forcing."
        ),
    ),

    # -- Convection ------------------------------------------------------
    "sbm_tau_c": TuningParameter(
        name="sbm_tau_c",
        default=7200.0,
        min_val=3600.0,
        max_val=14400.0,
        units="s",
        description="Convective relaxation timescale (SBM)",
        category="convection",
        sensitivity="high",
        notes=(
            "Shorter = more aggressive convective adjustment. "
            "Typical range 2-4 hours."
        ),
    ),
    "sbm_RH_ref": TuningParameter(
        name="sbm_RH_ref",
        default=0.7,
        min_val=0.6,
        max_val=0.9,
        units="1",
        description="Reference relative humidity for SBM convection",
        category="convection",
        sensitivity="high",
        notes="Column moistened toward this RH. Higher = wetter atmosphere.",
    ),

    # -- Diffusion / turbulence -----------------------------------------
    "k_free_per_day": TuningParameter(
        name="k_free_per_day",
        default=0.1,
        min_val=0.01,
        max_val=1.0,
        units="1/day",
        description="Free-atmosphere vertical mixing rate",
        category="diffusion",
        sensitivity="medium",
        notes="Background mixing above the BL. Too large smears out jets.",
    ),
    "k_BL_max_per_day": TuningParameter(
        name="k_BL_max_per_day",
        default=1.0,
        min_val=0.5,
        max_val=5.0,
        units="1/day",
        description="Maximum boundary layer mixing rate",
        category="diffusion",
        sensitivity="high",
        notes=(
            "Peak mixing in the BL. Controls depth and intensity of "
            "surface coupling."
        ),
    ),
    "sigma_b": TuningParameter(
        name="sigma_b",
        default=0.7,
        min_val=0.6,
        max_val=0.85,
        units="1",
        description="Boundary layer top (sigma level)",
        category="diffusion",
        sensitivity="medium",
        notes="Sigma level above which BL mixing tapers off.",
    ),

    # -- Surface ---------------------------------------------------------
    "C_H": TuningParameter(
        name="C_H",
        default=1.5e-3,
        min_val=1e-3,
        max_val=5e-3,
        units="1",
        description="Sensible heat bulk transfer coefficient",
        category="surface",
        sensitivity="high",
        notes="Larger = stronger surface sensible heat flux.",
    ),
    "C_E": TuningParameter(
        name="C_E",
        default=1.5e-3,
        min_val=1e-3,
        max_val=5e-3,
        units="1",
        description="Latent heat bulk transfer coefficient",
        category="surface",
        sensitivity="high",
        notes="Larger = stronger surface evaporation.",
    ),
    "albedo_land": TuningParameter(
        name="albedo_land",
        default=0.3,
        min_val=0.1,
        max_val=0.4,
        units="1",
        description="Land surface albedo",
        category="surface",
        sensitivity="medium",
        notes="Affects net surface SW absorption over land.",
    ),
    "albedo_ice": TuningParameter(
        name="albedo_ice",
        default=0.65,
        min_val=0.4,
        max_val=0.8,
        units="1",
        description="Sea ice albedo",
        category="surface",
        sensitivity="high",
        notes=(
            "Strong ice-albedo feedback. Higher = more SW reflected "
            "by ice, colder poles."
        ),
    ),
    "albedo_ocean": TuningParameter(
        name="albedo_ocean",
        default=0.06,
        min_val=0.03,
        max_val=0.10,
        units="1",
        description="Ocean surface albedo",
        category="surface",
        sensitivity="low",
        notes="Open-ocean reflectance. Typically 0.06 for diffuse light.",
    ),

    # -- Turbulence (Louis stability-dependent PBL) ----------------------
    # iter-252 (AIMIP Phase 2.2): exposed Ri_crit as a tunable so the
    # AIMIP training loop can adjust the Louis stable-PBL cutoff against
    # ERA5.  Default 0.25 follows Holtslag & De Bruin (1988); range
    # 0.2-0.5 spans common literature values (Mahrt 1998, Sukoriansky
    # 2005 push as high as 1.0 for very-stable PBL, but 0.5 is a safer
    # upper bound for current Louis stability functions).
    "Ri_crit": TuningParameter(
        name="Ri_crit",
        default=0.25,
        min_val=0.20,
        max_val=0.50,
        units="1",
        description="Critical Richardson number (Louis stability cutoff)",
        category="turbulence",
        sensitivity="medium",
        notes=(
            "Above Ri_crit Louis returns zero diffusivity (stable PBL "
            "shutdown).  Higher = more vigorous nocturnal/polar PBL "
            "mixing; lower = sharper PBL decoupling.  Only consulted "
            "when --turbulence louis."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _extract_tuning_fields(config):
    """Extract tuning-relevant fields from either config type.

    Returns a namespace-like object with attributes: resolution, dt,
    hyperdiff_scale, co2_ppmv, sbm_tau_c, sbm_RH_ref, rad_update_steps,
    radiation, cloud_scheme, microphysics, C_H, C_E, albedo_ice,
    albedo_ocean, nlev.

    Accepts ``ExperimentConfig`` (nested sub-configs) or
    ``AMIPExperimentConfig`` (flat fields).
    """
    from legoesm.driver.config import ExperimentConfig

    if isinstance(config, ExperimentConfig):
        class _Ns:
            pass
        ns = _Ns()
        ns.resolution = config.grid.resolution
        ns.dt = config.dycore.dt
        ns.hyperdiff_scale = config.dycore.hyperdiff_scale
        ns.co2_ppmv = config.co2_ppmv
        ns.sbm_tau_c = config.sbm_tau_c
        ns.sbm_RH_ref = config.sbm_RH_ref
        ns.rad_update_steps = config.rad_update_steps
        ns.radiation = config.radiation
        ns.cloud_scheme = config.cloud_scheme
        ns.microphysics = config.microphysics
        ns.C_H = config.C_H
        ns.C_E = config.C_E
        ns.albedo_ice = config.albedo_ice
        ns.albedo_ocean = config.albedo_ocean
        ns.nlev = config.grid.nlev
        return ns
    # Legacy AMIPExperimentConfig — fields are flat
    return config


def validate_tuning(config) -> list[str]:
    """Return a list of warning messages for potentially problematic settings.

    Accepts either ``ExperimentConfig`` (canonical) or
    ``AMIPExperimentConfig`` (legacy).

    Checks parameter ranges, CFL heuristics, and conflicting options.
    """
    cfg = _extract_tuning_fields(config)
    warnings_list: list[str] = []

    # -- dt vs. resolution heuristic ------------------------------------
    _dt_limits = {16: 900.0, 32: 600.0, 48: 600.0, 64: 300.0, 96: 150.0}
    dt_max = _dt_limits.get(cfg.resolution)
    if dt_max is not None and cfg.dt > dt_max:
        warnings_list.append(
            f"dt={cfg.dt:.0f}s may violate CFL at C{cfg.resolution} "
            f"resolution (recommend dt<={dt_max:.0f}s)"
        )

    # -- Hyper-diffusion vs. resolution ---------------------------------
    _hd_ranges = {
        16: (1e16, 1e18),
        48: (1e15, 5e16),
        96: (1e14, 1e16),
    }
    hd_range = _hd_ranges.get(cfg.resolution)
    if hd_range is not None:
        lo, hi = hd_range
        if cfg.hyperdiff_scale < lo:
            warnings_list.append(
                f"hyperdiff_scale={cfg.hyperdiff_scale:.1e} may be too "
                f"weak for C{cfg.resolution} (recommended >= {lo:.0e})"
            )
        if cfg.hyperdiff_scale > hi:
            warnings_list.append(
                f"hyperdiff_scale={cfg.hyperdiff_scale:.1e} may be too "
                f"strong for C{cfg.resolution} (recommended <= {hi:.0e})"
            )

    # -- Gas concentrations ---------------------------------------------
    if cfg.co2_ppmv <= 0.0:
        warnings_list.append(
            f"co2_ppmv={cfg.co2_ppmv} — zero or negative CO2 will "
            "cause radiation issues"
        )
    if cfg.co2_ppmv > 2000.0:
        warnings_list.append(
            f"co2_ppmv={cfg.co2_ppmv} — unusually high CO2 "
            "(pre-industrial ~280, current ~420)"
        )

    # -- SBM parameters -------------------------------------------------
    if cfg.sbm_tau_c < 1800.0:
        warnings_list.append(
            f"sbm_tau_c={cfg.sbm_tau_c:.0f}s is very short — may "
            "cause excessive convective adjustment"
        )
    if cfg.sbm_RH_ref > 0.95:
        warnings_list.append(
            f"sbm_RH_ref={cfg.sbm_RH_ref:.2f} is very high — may "
            "cause near-saturated atmosphere everywhere"
        )
    if cfg.sbm_RH_ref < 0.4:
        warnings_list.append(
            f"sbm_RH_ref={cfg.sbm_RH_ref:.2f} is very low — may "
            "produce an unrealistically dry atmosphere"
        )

    # -- Radiation cadence ----------------------------------------------
    if cfg.rad_update_steps > 12:
        warnings_list.append(
            f"rad_update_steps={cfg.rad_update_steps} is very large — "
            "radiation will be very stale between calls"
        )

    # -- Conflicting options -------------------------------------------
    if cfg.radiation == "gray" and cfg.cloud_scheme != "none":
        warnings_list.append(
            "Cloud scheme is active but radiation='gray' — clouds will "
            "have no effect on radiation"
        )
    if cfg.microphysics != "none" and cfg.cloud_scheme == "none":
        warnings_list.append(
            "Microphysics is active but cloud_scheme='none' — cloud "
            "water will not affect radiation"
        )

    # -- Surface bulk coefficients --------------------------------------
    for name in ("C_H", "C_E"):
        val = getattr(cfg, name)
        if val > 0.01:
            warnings_list.append(
                f"{name}={val:.4f} is unrealistically large "
                "(typical range 1e-3 to 5e-3)"
            )
        if val < 1e-4:
            warnings_list.append(
                f"{name}={val:.1e} is very small — surface fluxes will "
                "be negligible"
            )

    # -- Albedo range checks -------------------------------------------
    if not (0.0 <= cfg.albedo_ice <= 1.0):
        warnings_list.append(
            f"albedo_ice={cfg.albedo_ice} is outside [0, 1]"
        )
    if not (0.0 <= cfg.albedo_ocean <= 1.0):
        warnings_list.append(
            f"albedo_ocean={cfg.albedo_ocean} is outside [0, 1]"
        )

    # -- Vertical levels ------------------------------------------------
    if cfg.nlev < 10:
        warnings_list.append(
            f"nlev={cfg.nlev} — very few vertical levels, may not "
            "resolve the boundary layer or tropopause"
        )

    return warnings_list


# ---------------------------------------------------------------------------
# Recommended parameters
# ---------------------------------------------------------------------------

def recommended_params(resolution: int, nlev: int) -> dict:
    """Return recommended parameter values for a given resolution and nlev.

    Parameters
    ----------
    resolution : int
        Cubed-sphere face resolution (e.g. 16, 48, 96).
    nlev : int
        Number of vertical levels (informational; may influence dt slightly
        in the future).

    Returns
    -------
    dict
        Suggested parameter values keyed by parameter name.
    """
    presets = {
        16: dict(dt=600.0, hyperdiff_scale=5e16, rad_update_steps=1),
        48: dict(dt=300.0, hyperdiff_scale=5e15, rad_update_steps=3),
        96: dict(dt=150.0, hyperdiff_scale=5e14, rad_update_steps=6),
    }

    if resolution in presets:
        return presets[resolution]

    # Interpolate / extrapolate for arbitrary resolutions
    # dt scales roughly as 1/resolution (CFL), hyperdiff as 1/resolution^4
    ref_res = 48
    ref = presets[ref_res]
    scale = ref_res / resolution
    return dict(
        dt=round(ref["dt"] * scale / 10) * 10,      # round to nearest 10s
        hyperdiff_scale=ref["hyperdiff_scale"] * scale**4,
        rad_update_steps=max(1, round(ref["rad_update_steps"] / scale)),
    )


# ---------------------------------------------------------------------------
# Pretty-printed guide
# ---------------------------------------------------------------------------

def print_tuning_guide() -> None:
    """Print a formatted table of all tuning parameters."""
    categories = sorted({p.category for p in TUNING_PARAMETERS.values()})

    header = (
        f"{'Name':<22s} {'Default':>10s}  {'Range':>22s}  "
        f"{'Units':<8s} {'Sens.':<6s} Description"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  legoESM Tuning Guide")
    print("=" * len(header))

    for cat in categories:
        params = [
            p for p in TUNING_PARAMETERS.values() if p.category == cat
        ]
        if not params:
            continue

        print()
        print(f"  [{cat.upper()}]")
        print(sep)
        print(header)
        print(sep)

        for p in params:
            # Format default and range compactly
            def _fmt(v: float) -> str:
                if v >= 1e6 or (0 < v < 1e-2):
                    return f"{v:.0e}"
                if v == int(v):
                    return f"{int(v)}"
                return f"{v:g}"

            default_s = _fmt(p.default)
            range_s = f"[{_fmt(p.min_val)}, {_fmt(p.max_val)}]"

            print(
                f"  {p.name:<20s} {default_s:>10s}  {range_s:>22s}  "
                f"{p.units:<8s} {p.sensitivity:<6s} {p.description}"
            )
            if p.notes:
                # Wrap notes below the entry
                print(f"{'':>24s}  -> {p.notes}")

        print(sep)

    print()
