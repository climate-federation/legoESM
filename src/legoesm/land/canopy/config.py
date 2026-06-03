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


# ---------------------------------------------------------------------------
# PFT Vcmax25 lookup table (Jiang & Ryu 2016, Table A1)
# Columns: [warm, temperate, cold]  [μmol m-2 s-1]
# Missing climate zones filled by replicating the closest available value.
# ---------------------------------------------------------------------------
PFT_VCMAX25_C3: dict[str, list[float]] = {
    "ENF":    [63.0,  63.0,  63.0 ],  # only warm published; replicated
    "EBF":    [41.0,  62.0,  62.0 ],
    "DNF":    [57.0,  57.0,  57.0 ],  # only warm published; replicated
    "DBF":    [66.0,  62.0,  96.0 ],
    "MF":     [54.0,  62.0,  63.0 ],
    "SHR":    [62.0,  54.0,  54.0 ],  # OSH + CSH merged
    "SAV":    [90.0, 120.0, 120.0 ],  # WSA + SAV merged
    "GRA":    [78.0,  78.0, 142.0 ],  # C3 grassland
    "CRO":    [101.0, 101.0, 101.0],  # C3 cropland; only warm published
    "WET":    [78.0,  78.0, 142.0 ],
}

PFT_VCMAX25_C4: dict[str, list[float]] = {
    "GRA":    [40.0,  40.0,  40.0 ],  # C4 grassland; only temperate published
    "CRO":    [37.0,  37.0,  37.0 ],  # C4 cropland; only warm published
    "SAV":    [40.0,  40.0,  40.0 ],  # Savannas often mixed C3/C4; use GRA_C4
}

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

    # Soil moisture stress thresholds (when no Richards state available)
    wilting_point: float = 0.15   # theta_wp [m3/m3]
    field_capacity: float = 0.30  # theta_fc [m3/m3]
    n_root_layers: int = 5        # number of layers to integrate for root-zone stress


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
