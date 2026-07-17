"""Configuration for the canopy energy balance + photosynthesis biophysics.

Implements the DifferBESS-style two-leaf canopy model as a legoESM land
surface scheme.  ``CanopyConfig`` holds the solver-level scalars
(max_iters, LE_module, stomatal_model, ...) and is re-exported from
``legoesm.land.surface_scheme`` as ``TwoLeafCanopyConfig`` — both names
refer to the same NamedTuple type.

PFT Vcmax25 values from Jiang & Ryu (2016) Table A1.
Aerodynamic parameters from Ryu et al. (2011) / DifferBESS defaults.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.land.canopy.sif import SIFConfig


# ---------------------------------------------------------------------------
# PFT Vcmax25 lookup table [μmol m-2 s-1], columns [tropical, temperate, boreal]
# (a.k.a. warm / temperate / cold).
#
# Authoritative CLM 4.5 Tech Note Table 8.1 entries blended with Jiang & Ryu
# (2016) BESS v1 Table A1 where CLM has no direct PFT mapping (Mixed Forest,
# savanna, warm-climate shrubs).  Mirrors the DifferBESS fallback table
# (util/io.py ``_VCMAX25_C3_TABLE`` / ``_VCMAX25_C4_TABLE``).  IGBP->PFT-name
# mapping: CSH+OSH -> SHR; WSA+SAV -> SAV; CRO+CNM -> CRO.  C4 grass uses the
# CLM4.5 C4-grass value (51.6); C4 crop uses Jiang & Ryu (37.0) because the
# CLM4.5 Corn value (100.7) is too aggressive for the Collatz C4 pathway here.
# ---------------------------------------------------------------------------
PFT_VCMAX25_C3: dict[str, list[float]] = {
    "ENF":    [ 62.5,  62.5,  62.6],  # [CLM45] NET Temperate/Boreal
    "EBF":    [ 55.0,  61.5,  61.5],  # [CLM45] BET Tropical/Temperate
    "DNF":    [ 39.1,  39.1,  39.1],  # [CLM45] NDT Boreal (only DNF entry)
    "DBF":    [ 41.0,  57.7,  57.7],  # [CLM45] BDT Tropical/Temperate/Boreal
    "MF":     [ 54.0,  62.0,  63.0],  # [JR] Mixed forest (no CLM MF PFT)
    "SHR":    [ 54.0,  54.0,  54.0],  # [CLM45] BDS / [JR] shrub (OSH+CSH)
    "SAV":    [ 90.0, 120.0, 120.0],  # [JR] savanna (WSA+SAV; no CLM PFT)
    "GRA":    [ 78.2,  78.2,  78.2],  # [CLM45] C3 grass
    "CRO":    [100.7, 100.7, 100.7],  # [CLM45] Crop (C3 unmanaged; CRO+CNM)
    "WET":    [ 78.2,  78.2,  78.2],  # no DifferBESS entry — use C3 grass
}

PFT_VCMAX25_C4: dict[str, list[float]] = {
    "GRA":    [51.6, 51.6, 51.6],  # [CLM45] C4 grass
    "SAV":    [51.6, 51.6, 51.6],  # [CLM45] C4 grass (WSA+SAV)
    "CRO":    [37.0, 37.0, 37.0],  # [JR] C4 crop (CLM4.5 Corn 100.7 too high)
}

# Climate-column indices for the PFT Vcmax25 tables above.
_CLIM_TROPICAL, _CLIM_TEMPERATE, _CLIM_BOREAL = 0, 1, 2
_CLIMATE_INDEX = {
    "tropical": _CLIM_TROPICAL,
    "temperate": _CLIM_TEMPERATE,
    "boreal": _CLIM_BOREAL,
}

# No-PFT fallback Vcmax25 [μmol m-2 s-1] (used when no PFT is assigned):
#   C3 -> DBF-temperate; C4 -> mean of the C4-grass and C4-crop temperate
# values.  Derived from the tables so they stay in sync (not magic literals).
VCMAX25_C3_DEFAULT: float = PFT_VCMAX25_C3["DBF"][_CLIM_TEMPERATE]            # 57.7
VCMAX25_C4_DEFAULT: float = 0.5 * (PFT_VCMAX25_C4["GRA"][_CLIM_TEMPERATE]
                                   + PFT_VCMAX25_C4["CRO"][_CLIM_TEMPERATE])  # 44.3


def lookup_vcmax25(pft: str, climate: str = "temperate", c4: bool = False) -> float:
    """Leaf Vcmax25 [μmol m-2 s-1] for a (PFT, climate), with a no-PFT default.

    Parameters
    ----------
    pft     : PFT name key (ENF/EBF/DNF/DBF/MF/SHR/SAV/GRA/CRO/WET).  Unknown
              PFTs return the no-PFT default (DBF-temperate for C3; mean C4
              grass/crop for C4).
    climate : "tropical" | "temperate" | "boreal".  Unknown -> ValueError.
    c4      : select the C4 table instead of C3.
    """
    if climate not in _CLIMATE_INDEX:
        raise ValueError(
            f"unknown climate {climate!r}; expected one of {sorted(_CLIMATE_INDEX)}")
    idx = _CLIMATE_INDEX[climate]
    table = PFT_VCMAX25_C4 if c4 else PFT_VCMAX25_C3
    default = VCMAX25_C4_DEFAULT if c4 else VCMAX25_C3_DEFAULT
    entry = table.get(pft)
    return float(entry[idx]) if entry is not None else float(default)

# ---------------------------------------------------------------------------
# PFT aerodynamic parameters (DifferBESS / Ryu et al. 2011)
# rz0m = z0m / hc ratio
# rd   = displacement height / hc ratio
# ---------------------------------------------------------------------------
PFT_AERO_PARAMS: dict[str, dict[str, float]] = {
    "ENF": {"rz0m": 0.055, "rd": 0.67},
    "EBF": {"rz0m": 0.075, "rd": 0.67},
    "DNF": {"rz0m": 0.055, "rd": 0.67},
    "DBF": {"rz0m": 0.055, "rd": 0.67},
    "MF":  {"rz0m": 0.055, "rd": 0.67},
    "SHR": {"rz0m": 0.12,  "rd": 0.68},
    "SAV": {"rz0m": 0.12,  "rd": 0.68},
    "GRA": {"rz0m": 0.12,  "rd": 0.68},
    "CRO": {"rz0m": 0.12,  "rd": 0.68},
    "WET": {"rz0m": 0.12,  "rd": 0.68},
}

# Characteristic leaf width per PFT [m] (Schuepp 1993 midrange; DifferBESS
# aa6e8b9).  Drives the leaf boundary-layer resistance rb = 1/(cv*sqrt(uav/
# d_leaf)).  Small needles vs broad leaves; default 0.025 when unspecified.
PFT_LEAF_WIDTH: dict[str, float] = {
    "ENF": 0.01,
    "EBF": 0.04,
    "DNF": 0.01,
    "DBF": 0.025,
    "MF":  0.025,
    "SHR": 0.02,   # OSH + CSH
    "SAV": 0.025,  # WSA + SAV
    "GRA": 0.02,
    "CRO": 0.025,
    "WET": 0.02,
}

# Default canopy heights per PFT [m]  (used when gridded hc not provided)
PFT_CANOPY_HEIGHT: dict[str, float] = {
    "ENF": 15.0,
    "EBF": 20.0,
    "DNF": 12.0,
    "DBF": 15.0,
    "MF":  12.0,
    "SHR": 1.5,
    "SAV": 3.0,
    "GRA": 0.5,
    "CRO": 0.8,
    "WET": 0.8,
}


# ---------------------------------------------------------------------------
# CanopyConfig — scalar physics settings (static Python values, not traced)
# ---------------------------------------------------------------------------
# Valid values for the static leaf-gas-exchange dispatch field.  Kept next to
# the config so the fail-early validator and the config default cannot drift.
_VALID_STOMATAL_MODELS = ("ball_berry", "medlyn")


class CanopyConfig(NamedTuple):
    """Physics settings for the canopy energy balance solver."""

    # Newton-Raphson solver.  With the scalar-clamp damping and the
    # outer Picard canopy↔thermal loop in canopy_land.py, 50 iters is
    # normally ample.
    max_iters: int = 50
    tol: float = 1e-2
    # Only the DifferBESS FULLY_COUPLED scheme is implemented (leaves and
    # soil share the canopy air space Tc, q_c via clumping-weighted
    # below-canopy resistance).  The VEG_ONLY / LEAVES_ATMO variants were
    # removed to keep the Newton residual minimal — re-introduce them via
    # a new static config string if a multi-scheme comparison is needed.
    LE_module: str = "BT"                   # "BT" (Bulk Transfer, default) | "PM" (Penman-Monteith)
    # Stomatal conductance model used inside the leaf energy balance
    # closure.  "ball_berry" interprets ``m``/``b0`` as Ball-Berry slope
    # and intercept; "medlyn" interprets ``m`` as the Medlyn g1 slope
    # [kPa^0.5] and ``b0`` as g0 [mol/m2/s].  Captured as a static
    # Python string via functools.partial — never traced.
    stomatal_model: str = "ball_berry"      # "ball_berry" | "medlyn"
    use_ta_for_photosynthesis: bool = False  # use Ta (True) or Tf (False) for photosynthesis
    # Energy-balance latent-heat cap that keeps the leaf-temperature Newton
    # solve from diverging (NaN leaf T -> NaN fluxes) under hot/dry/high-VPD
    # forcing.  "soft" (default) is a smooth softplus bound that still admits
    # dew and a modest LE>Rn excess; "off" reproduces the pre-cap behaviour;
    # "hard" is the legacy clip(LE, 0, max(Rn,0)).  Resolved at trace time
    # (static, like LE_module/stomatal_model).  See energy_balance.apply_le_cap.
    le_cap_mode: str = "soft"               # "soft" (default) | "hard" | "off"
    # Prognostic LAI feedback (Phase 6 / Stage 2b).  When True and the
    # carbon cycle is active with ``scheme="differland"``, the canopy's
    # LAI is recomputed each step from ``C_fol / LCMA``, bypassing any
    # prescribed ``CanopyLandParams.LAI``.  **Defaults to False** until
    # the reverse-mode ``jax.grad`` NaN through the ``C_fol → LAI →
    # canopy Newton`` feedback loop is resolved (see ``monin_obukhov_
    # stability`` custom-VJP follow-up).  Forward pass and non-feedback
    # gradient paths are unaffected by this default — enable explicitly
    # for coupled carbon ↔ canopy runs that do not require ``jax.grad``
    # through the feedback loop.
    use_prognostic_lai: bool = False

    # NOTE: The former ``G_alpha`` tunable (G = G_alpha · Rn_soil) has been
    # removed.  Ground heat flux is now diagnosed as the surface energy
    # budget residual ``G = Rn_soil - LE_soil - H_soil`` using
    # T_soil[:, 0] as the prescribed skin temperature, then fed as the
    # top BC to ``solve_soil_thermal`` — same pattern as multilayer_land.

    # Emissivities
    epsf: float = 0.97   # leaf emissivity
    epss: float = 0.96   # soil emissivity

    # Leaf boundary-layer forced-convection transfer coefficient
    # [m^-0.5 s^0.5] in rb = 1 / (cv * sqrt(uav / d_leaf)).  CLM5-aligned
    # default (Campbell & Norman 1998; Bonan 2019) replacing the older BESS
    # value 0.01 — see DifferBESS aa6e8b9.  Paired with kB^-1 = 0 in MOST.
    cv: float = 0.0135

    # Soil moisture stress thresholds (when no Richards state available)
    wilting_point: float = 0.15   # theta_wp [m3/m3]
    field_capacity: float = 0.30  # theta_fc [m3/m3]
    n_root_layers: int = 5        # number of layers to integrate for root-zone stress

    # Optional solar-induced fluorescence (SIF) diagnostic.  ``None`` (default)
    # disables it; a ``SIFConfig`` enables the passive top-of-canopy SIF output
    # (sunlit+shaded sum) on ``SurfaceFluxOutput.sif``.  Static config leaf —
    # never traced, so the Python ``is not None`` gate does not double-trace.
    sif: SIFConfig | None = None
    # Whether the soil-moisture stress factor down-regulates the Ball-Berry
    # INTERCEPT b0 (cuticular / residual minimum conductance) as well as the slope
    # m.  True = legacy (both stressed).  False keeps b0 unstressed: the leaf
    # cuticle keeps leaking under drought, so a baseline dry-season transpiration
    # persists (raises LE at drought-adapted / phreatophytic sites), while adding
    # negligible CO2 uptake — light/LAI-limited GPP is essentially unchanged
    # (identically unchanged only at full Vcmax stress, where An -> 0 regardless
    # of gs; at partial drought a slightly higher gs raises Ci and can nudge An
    # up marginally).  Also floors gs at b0>0, avoiding the gs->0 Newton
    # degeneracy.  Static Python bool.  Appended (not inserted mid-tuple) so a
    # positional / tuple reconstruction of a pre-field CanopyConfig stays aligned
    # and defaults this to the legacy True.
    stress_b0: bool = True

    def validate(self) -> "CanopyConfig":
        """Fail-early check of the static string-dispatch fields.

        Called at the non-jitted two-leaf entry (``compute_two_leaf_canopy_
        fluxes``) so a typo'd ``stomatal_model`` aborts at land-component setup
        with a clear message, instead of dying deep inside the JAX leaf kernel
        ``energy_balance._compute_gs_and_ci`` on the first solve/trace (which
        keeps its own raise as a backstop).  Returns ``self`` for chaining.
        """
        if self.stomatal_model not in _VALID_STOMATAL_MODELS:
            raise ValueError(
                f"unknown stomatal_model {self.stomatal_model!r}; the stomatal "
                f"conductance scheme must be one of {_VALID_STOMATAL_MODELS}")
        return self


# Machine-readable tunable/fixed classification for every ``: float`` field of
# CanopyConfig (see tests/test_param_specs.py). Values live on the NamedTuple;
# this spec adds only units/bounds/tier/transform/category/reference.
__param_spec__ = {
    "CanopyConfig": {
        "scheme_key": "land.two_leaf_canopy",
        "excluded": {
            "tol": "numerics: Newton-Raphson convergence tolerance",
        },
        "params": {
            "epsf": {
                "units": "1", "bounds": (0.90, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "radiation",
                "reference": "leaf longwave emissivity (Ryu et al. 2011 / CLM5)",
                "shape": None,
            },
            "epss": {
                "units": "1", "bounds": (0.90, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "radiation",
                "reference": "soil longwave emissivity (Ryu et al. 2011 / CLM5)",
                "shape": None,
            },
            "cv": {
                "units": "m^-0.5 s^0.5", "bounds": (0.005, 0.03), "tunable_tier": 2,
                "transform": "sigmoid", "category": "aerodynamics",
                "reference": "leaf boundary-layer forced-convection coefficient "
                             "(Campbell & Norman 1998 / CLM5)",
                "shape": None,
            },
            "wilting_point": {
                "units": "m^3/m^3", "bounds": (0.05, 0.25), "tunable_tier": 2,
                "transform": "sigmoid", "category": "hydrology",
                "reference": "soil-moisture-stress wilting point theta_wp "
                             "(CLM5 / DifferBESS fallback)",
                "shape": None,
            },
            "field_capacity": {
                "units": "m^3/m^3", "bounds": (0.20, 0.50), "tunable_tier": 2,
                "transform": "sigmoid", "category": "hydrology",
                "reference": "soil-moisture-stress field capacity theta_fc "
                             "(CLM5 / DifferBESS fallback)",
                "shape": None,
            },
        },
    },
    "CLMMLCanopyConfig": {
        "scheme_key": "land.canopy.clm_ml",
        "excluded": {},
        "params": {
            "o2ref": {
                "units": "mmol/mol",
                "bounds": (180.0, 230.0),
                "tunable_tier": 0,
                "transform": "none",
                "category": "atmospheric",
                "reference": "standard atmosphere O2 = 209 mmol/mol",
                "shape": None,
            },
            "f_vis": {
                "units": "1",
                "bounds": (0.40, 0.55),
                "tunable_tier": 2,
                "transform": "sigmoid",
                "category": "radiation",
                "reference": "Weiss & Norman (1985); observation mean ~0.46",
                "shape": None,
            },
            "f_dir": {
                "units": "1 (or -1 for auto)",
                "bounds": (-1.0, 1.0),
                "tunable_tier": 0,
                "transform": "none",
                "category": "radiation",
                "reference": "Erbs et al. (1982) clearness-index estimate; -1=auto",
                "shape": None,
            },
            "smp_default_mm": {
                "units": "mm",
                "bounds": (-200000.0, -1000.0),
                "tunable_tier": 0,
                "transform": "none",
                "category": "soil",
                "reference": "CLM4.5 standalone; moderate stress -50000 mm",
                "shape": None,
            },
            "hk_default_mm_s": {
                "units": "mm/s",
                "bounds": (1.0e-7, 1.0e-2),
                "tunable_tier": 0,
                "transform": "none",
                "category": "soil",
                "reference": "silty clay loam at moderate dryness",
                "shape": None,
            },
            "soilresis_default_s_m": {
                "units": "s/m",
                "bounds": (50.0, 10000.0),
                "tunable_tier": 0,
                "transform": "none",
                "category": "soil",
                "reference": "Sellers-Lockwood formula; ~2000 s/m at Se=0.15",
                "shape": None,
            },
            "root_biomass_default_g_m2": {
                "units": "g/m²",
                "bounds": (50.0, 1000.0),
                "tunable_tier": 2,
                "transform": "softplus",
                "category": "vegetation",
                "reference": "Jackson et al. (1997) Global Ecol. Biogeogr.; 150–500 g/m²",
                "shape": None,
            },
            "albgrd_vis_default": {
                "units": "1",
                "bounds": (0.04, 0.30),
                "tunable_tier": 1,
                "transform": "sigmoid",
                "category": "radiation",
                "reference": "CLM4.5 loam soil lookup; moist ~0.10, dry ~0.17",
                "shape": None,
            },
            "albgrd_nir_default": {
                "units": "1",
                "bounds": (0.08, 0.50),
                "tunable_tier": 1,
                "transform": "sigmoid",
                "category": "radiation",
                "reference": "CLM4.5 loam soil lookup; moist ~0.20, dry ~0.34",
                "shape": None,
            },
            "hbot_frac": {
                "units": "1",
                "bounds": (0.02, 0.30),
                "tunable_tier": 0,
                "transform": "sigmoid",
                "category": "vegetation",
                "reference": "DifferBESS / Bonan et al. (2021) GMD; hbot = 0.1 * htop",
                "shape": None,
            },
            "thk_soil_default_W_m_K": {
                "units": "W/m/K",
                "bounds": (0.1, 3.0),
                "tunable_tier": 0,
                "transform": "softplus",
                "category": "soil",
                "reference": "CLM4.5 Table 3.3 moist loam; 0.9–1.5 for wetter soils",
                "shape": None,
            },
            "f_dir_noclearness_fallback": {
                "units": "1",
                "bounds": (0.10, 0.50),
                "tunable_tier": 0,
                "transform": "sigmoid",
                "category": "radiation",
                "reference": "Erbs et al. (1982) overcast-sky limit; low clearness index",
                "shape": None,
            },
        },
    },
}


# ---------------------------------------------------------------------------
# CanopyLandParams — spatially varying prescribed per-column fields
# All array fields have shape (ncol,) and are provided externally
# (MODIS seasonal cycle, satellite LAI, gridded canopy height maps, etc.)
# ---------------------------------------------------------------------------
class CanopyLandParams(NamedTuple):
    """Per-column (ncol,) prescribed land surface parameters for the canopy model.

    Fields that vary over PFTs (e.g. rz0m, rd, Vcmax25) should be pre-assigned
    from PFT_VCMAX25_C3 / PFT_AERO_PARAMS by the driver before passing here.
    """

    # ---- Vegetation structure ----
    LAI: jax.Array              # Leaf area index [m2/m2] — seasonally prescribed
    hc: jax.Array               # Canopy height [m] — gridded map OR replicated PFT default
    fC4: jax.Array              # C4 fraction [0–1] — from land-cover map
    FNonVeg: jax.Array          # Non-vegetated fraction [0–1]
    CI: jax.Array               # Clumping index [-] — gridded or PFT default
    kn: jax.Array               # Nitrogen extinction coefficient [-] (typically 0.3–0.6)

    # ---- Vcmax25 at leaf scale [μmol m-2 s-1] from lookup table ----
    Vcmax25_C3_leaf: jax.Array  # C3 maximum carboxylation rate at 25°C
    Vcmax25_C4_leaf: jax.Array  # C4 maximum carboxylation rate at 25°C

    # ---- Ball-Berry stomatal parameters ----
    m_C3: jax.Array             # Stomatal slope C3 [mol m-2 s-1 / RH] (typical: 9)
    m_C4: jax.Array             # Stomatal slope C4 (typical: 4)
    b0_C3: jax.Array            # Stomatal intercept C3 [mol m-2 s-1] (typical: 0.01)
    b0_C4: jax.Array            # Stomatal intercept C4 (typical: 0.04)

    # ---- Photosynthesis ----
    # alf is the quantum yield for electron transport [mol CO2 / mol photons].
    # It drives the RuBP-regeneration-limited rate Wj in C3 and the
    # light-limited rate in C4; it is independent of Vcmax25.
    alf: jax.Array              # Quantum yield (typical: 0.3 for C3, 0.067 for C4)
    TgC: jax.Array              # 30-day mean growth temperature [°C] for Vcmax acclimation

    # ---- Radiative ----
    ALB_VIS: jax.Array          # Visible-band surface albedo (from MODIS)
    ALB_NIR: jax.Array          # NIR-band surface albedo (from MODIS)
    emissivity: jax.Array       # Broadband surface emissivity

    # ---- Aerodynamics ----
    # rz0m and rd are PFT-specific ratios from PFT_AERO_PARAMS, replicated to (ncol,)
    rz0m: jax.Array             # z0m / hc ratio
    rd: jax.Array               # Displacement height / hc ratio
    # Characteristic leaf width [m] from PFT_LEAF_WIDTH (Schuepp 1993).
    # Trailing optional field — None falls back to the 0.025 m midrange so
    # existing CanopyLandParams constructors need not be updated.
    d_leaf: jax.Array | None = None
    # Persistent STRUCTURAL leaf area index [m2/m2] driving the forest-floor litter
    # cover in the soil-evaporation resistance (a slowly-varying / seasonal-maximum
    # LAI, so a deciduous forest floor keeps its litter through the leaf-off
    # season).  Trailing optional field — None falls back to the live ``LAI``.
    litter_LAI: jax.Array | None = None


# NOTE: ``CanopyLandConfig`` has been removed.  Canopy is now a surface
# scheme of ``MultiLayerLandConfig`` (and, in Phase 3b, ``LandConfig``):
#
#     cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(...))
#
# Dispatch happens inside ``step_multilayer_land`` / ``step_land`` via
# ``isinstance`` on the ``surface_scheme`` field.


# ---------------------------------------------------------------------------
# CLMMLCanopyConfig — configuration for the CLM-ML-JAX multilayer canopy
# ---------------------------------------------------------------------------

class CLMMLCanopyConfig(NamedTuple):
    """Configuration for the CLM-ML-JAX multilayer canopy surface scheme.

    Used as ``MultiLayerLandConfig(surface_scheme=CLMMLCanopyConfig())``.
    Dispatch inside ``step_multilayer_land`` detects this type via
    ``isinstance`` and routes to the CLM-ML-JAX interface in
    ``legoesm.land.canopy.clm_ml_interface``.

    All fields are static (Python scalars captured in the closure at
    JIT compile time — never traced).  Per-column spatial parameters
    (LAI, SAI, htop, hbot) are provided via ``LandSurfaceParams`` at
    each timestep.
    """

    # Canopy vertical discretisation
    nlevmlcan: int = 9          # Number of canopy layers (MLclm_varpar.nlevmlcan)

    # Sub-cycling / Runge-Kutta integration
    # 10 → Euler (nrk_steps = 0); 2x → RK with x stages
    runge_kutta_type: int = 10
    num_ml_steps: int = 1       # CLM sub-steps per legoESM timestep

    # Reference O2 concentration [mmol/mol]
    o2ref: float = 209.0

    # Atmospheric forcing interpolation mode:
    #   0 → no interpolation (single forcing value per CLM step)
    #   3 → 3-point centred interpolation (bef / cur / next)
    met_type: int = 0

    # CLM PFT index (1-based, 0=bare).  Controls Vcmax25, plant hydraulic
    # parameters, beta-distribution PAD shape, and canopy height defaults
    # from the MLpftcon lookup table.
    #   7  = broadleaf deciduous temperate tree (BDT) — default for forests
    #   13 = C3 non-arctic grass — CLM default grass PFT
    # When in doubt, choose the PFT whose Vcmax25 and htop match the site.
    pft_clm: int = 7

    # SW band partitioning.
    # f_vis: fraction of total SW in the visible (PAR) band [0.4–0.7 µm].
    #   Observation-based climatological mean is ~0.46 (not 0.5).
    # f_dir: direct-beam fraction of total SW.
    #   -1.0 → estimate from solar zenith angle and clearness index (default).
    #   0.0–1.0 → fixed override (use only when the coupler guarantees a
    #              constant sky condition, e.g. idealised aquaplanet runs).
    f_vis: float = 0.46
    f_dir: float = -1.0         # -1 → auto-estimated from zenith + clearness

    # Default soil matric potential [mm] when psi_soil is not provided.
    # -50 000 mm = -0.49 MPa — moderate stress, mid-range of plant-available water
    # (FC ≈ -33 kPa = -3 400 mm; permanent wilting point ≈ -1.5 MPa = -153 000 mm).
    # Previous value (-3 000 mm = -0.029 MPa) was at field capacity and kept
    # btran≈1 (no plant stress) for all May timesteps in the CHATS7 default run.
    # The Fortran standalone uses -10 000 to -40 000 mm for a California walnut.
    smp_default_mm: float = -50_000.0

    # Default unsaturated hydraulic conductivity [mm/s].
    # 1e-5 mm/s ≈ 0.86 mm/day, representative of silty clay loam at
    # moderate dryness.  The prior default (1e-4) was 10× too high.
    hk_default_mm_s: float = 1.0e-5

    # Default soil evaporative resistance [s/m] when soil texture is not known.
    # Sellers-Lockwood formula with silty clay loam parameters and Se≈0.15
    # (corresponding to smp_default_mm = -50000 mm) gives rs ≈ 1927 s/m.
    # Using 2000 s/m as a round number.  Previous default (100 s/m) matched
    # saturated conditions only and was ~20× too low for the new dry default.
    soilresis_default_s_m: float = 2000.0

    # Default fine root biomass per unit ground area [g/m²].
    # Used by CLM-ML SoilResistance to compute root length density and soil
    # hydraulic conductance.  The default mlcanopy_type initializes this to
    # spval=1e36, which produces anomalous root conductance.
    # Literature range for temperate deciduous trees: 150–500 g/m²
    # (Jackson et al. 1997 Global Ecol. Biogeogr.); using 300 g/m² as default.
    root_biomass_default_g_m2: float = 300.0

    # Sub-canopy soil (ground) spectral albedo for the CLM-ML two-stream RT.
    # These are the SOIL bottom-boundary albedos used by MLSolarRadiationMod,
    # NOT the vegetation broadband albedo.  CLM4.5 loam soil lookup values:
    #   VIS (0.4–0.7 µm):  ~0.10  (moist loam; dry loam ~0.17)
    #   NIR (0.7–5.0 µm):  ~0.20  (moist loam; dry loam ~0.34)
    # These are lower than the vegetation albedo (0.15–0.20 broadband) and
    # must NOT be derived from albedo_veg.
    albgrd_vis_default: float = 0.10
    albgrd_nir_default: float = 0.20

    # Bottom-of-canopy height as fraction of canopy top height.
    # CLM-ML expects hbot < htop; 0.1 * htop is the DifferBESS default
    # (Bonan et al. 2021 GMD) for the beta-distribution PAD lower boundary.
    hbot_frac: float = 0.1

    # Leaf-area-index source (mirrors ``CanopyConfig.use_prognostic_lai``):
    #   False (default) → PRESCRIBED LAI: the climatology in
    #     ``LandSurfaceParams.LAI`` (surfdata monthly, PFT-weighted) or the
    #     scalar fallback in ``clm_ml_interface``.
    #   True → PROGNOSTIC LAI: ``LAI = C_fol / LCMA`` from the DifferLand
    #     carbon pool (``compute_prognostic_lai``), so leaf area responds to the
    #     coupled carbon dynamics.  Requires ``land_config.carbon.scheme ==
    #     "differland"`` and a non-None ``carbon_state``; otherwise the call
    #     falls back to the prescribed LAI.  Canopy STRUCTURE (SAI, htop, hbot)
    #     stays prescribed either way — the carbon cycle produces no allometric
    #     height/stem mapping.  FORWARD-ONLY: CLM-ML is eager/non-jit, so the
    #     carbon→LAI feedback is a prognostic forward coupling, not a
    #     differentiable one (do not ``jax.grad`` through the CLM-ML interface).
    use_prognostic_lai: bool = False

    # Soil thermal conductivity [W/m/K] used for the soil-to-canopy heat flux
    # linearization in MLSoilTemperatureMod.  CLM4.5 Table 3.3 moist loam
    # default; 0.9–1.5 W/m/K for wetter/sandier soils.
    thk_soil_default_W_m_K: float = 0.5

    # Direct-beam fraction fallback [0–1] when f_dir < 0 (auto-estimate) but
    # cos_zen is not available.  Corresponds to an overcast sky condition;
    # Erbs et al. (1982) gives f_dir ≈ 0.20–0.35 for low clearness index.
    f_dir_noclearness_fallback: float = 0.30

    # Optional solar-induced fluorescence (SIF) diagnostic.  ``None`` (default)
    # disables it; a ``SIFConfig`` enables the passive top-of-canopy SIF output
    # on ``SurfaceFluxOutput.sif`` — a leaf-area-weighted sum over the CLM-ML
    # canopy layers × sunlit/shaded leaves, sharing the same fluorescence core
    # as the two-leaf / big-leaf paths.  Static config leaf, never traced.
    sif: SIFConfig | None = None

    # Differentiable-mode gate (static Python bool, resolved at trace time —
    # never a traced leaf).  ``False`` (default) keeps PRODUCTION forward-only:
    # the CLM-ML driver runs its Python for-loop / host-syncing checks and NO
    # ``jax.grad`` tape is built (fast, no reverse-mode memory).  ``True`` opts
    # a TRAINING run into the JAX-native diff path: the interface passes a
    # ``GridInfo`` (``grid=``) and sets ``MLclm_varctl.DIFFERENTIABLE_MODE`` so
    # ``MLCanopyFluxes`` runs ``lax.scan`` + ``jax.checkpoint`` and the forcing→
    # flux map is fully on the ``jax.grad`` tape.  Diff mode is single-column
    # (``ncol == 1``) — the diff path reads one concrete ``(ncan, ntop, nbot)``
    # from the warm-start template.  Multi-column diff is NOT vmap-able (the
    # interface mutates CLM module globals host-side); train multiple columns by
    # looping OUTSIDE ``jax.grad`` and accumulating per-column gradients.  See
    # ``docs/land/clm_ml_differentiable_integration_scope.md``.
    differentiable: bool = False
