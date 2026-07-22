"""
JAX translation of MLclm_varctl Fortran module.

Run control variables for the multilayer canopy (CLMml) model.
Contains integer and real flags governing stomatal conductance,
photosynthesis, turbulence, radiative transfer, time-stepping,
and canopy layer discretization.

Original Fortran module: MLclm_varctl
Fortran lines 1-65
"""

from typing import NamedTuple

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import ispval  # noqa: F401

# ---------------------------------------------------------------------------
# Differentiable-mode infrastructure — not part of original Fortran
# ---------------------------------------------------------------------------


class GridInfo(NamedTuple):
    """Structural grid constants extracted *before* JAX tracing.

    Under ``jax.grad``, NamedTuple fields of ``mlcanopy_inst`` become
    abstract tracers.  Calling ``int()`` on a tracer raises
    ``ConcretizationTypeError``.  ``GridInfo`` bundles those integers so
    they can be extracted once (as concrete Python ints) and threaded
    through all physics functions via a closure or explicit parameter.
    """

    p: int  # patch index (always 1 in single-site mode)
    ncan: int  # number of canopy layers
    ntop: int  # top foliage layer
    nbot: int  # bottom foliage layer


DIFFERENTIABLE_MODE: bool = False
"""When *True*, skip ``endrun`` error checks, diagnostic file I/O, and
any operations that break the JAX gradient tape.  Set to *True* inside
the ``make_clm_ml_forward`` factory before calling physics routines."""


# ---------------------------------------------------------------------------
# Stomatal conductance and photosynthesis — Fortran lines 16-21
# ---------------------------------------------------------------------------

gs_type: int = 2  # Stomatal conductance: Medlyn (0), Ball-Berry (1), WUE optimization (2)
gspot_type: int = 1  # Use potential conductance (0) or water-stressed conductance (1)
gs_solver: int = 2  # WUE solver: Brent (1) or bisection (2)
colim_type: int = 1  # Photosynthesis: minimum rate (0) or co-limited rate (1)
acclim_type: int = 1  # Temperature acclimation: off (0) or on (1)
kn_val: float = -999.0  # Canopy nitrogen profile: user-specified Kn (>0); -999 = not set


# ---------------------------------------------------------------------------
# Canopy turbulence — Fortran lines 23-28
# ---------------------------------------------------------------------------

turb_type: int = 1  # Turbulence parameterization: H&F roughness sublayer (1)
sparse_canopy_type: int = 1  # H&F roughness sublayer: sparse canopy off (0) or on (1)
HF_extension_type: int = 2  # Aerodynamic conductance at ground: extend H&F (1) or log profile (2)
flux_profile_type: int = 1  # Flux-profile solution: dataset (-1), well-mixed (0), implicit (1)
gb_type: int = 3  # Boundary layer conductance: laminar (1), +turbulent (2), +free convection (3)


# ---------------------------------------------------------------------------
# Radiative transfer — Fortran lines 30-32
# ---------------------------------------------------------------------------

light_type: int = 2  # Solar radiative transfer: Norman (1) or two-stream (2)
leaf_optics_type: int = 0  # Leaf optical properties: constant with height (0) or varying (1)
longwave_type: int = 1  # Longwave radiative transfer: Norman (1)


# ---------------------------------------------------------------------------
# Multilayer canopy time-stepping — Fortran lines 34-35
# num_ml_steps = dtime_clm / dtime_ml must be an integer
# ---------------------------------------------------------------------------

dtime_ml: float = 300.0  # Multilayer canopy time step (s)


# ---------------------------------------------------------------------------
# Runge-Kutta method — Fortran lines 37-41
# ---------------------------------------------------------------------------

runge_kutta_type: int = 41  # Euler (10)
# 2nd-order (21:trapezoidal, 22:midpoint, 23:Ralston)
# 3rd-order (31:Huen, 32:Ralston, 33:Kutta)
# 4th-order (41:Kutta)

nrk: int = runge_kutta_type // 10  # Number of Runge-Kutta steps — Fortran: runge_kutta_type/10


# ---------------------------------------------------------------------------
# Canopy layer discretization — Fortran lines 43-50
# Use dz_tall / dz_short, OR set nlayer_above / nlayer_within > 0 directly.
# ---------------------------------------------------------------------------

dz_tall: float = 0.5  # Height increment for tall canopies (htop > dz_param) (m)
dz_short: float = 0.1  # Height increment for short canopies (htop <= dz_param) (m)
dz_param: float = 2.0  # Height threshold separating tall from short canopies (m)

nlayer_above: int = 0  # Number of above-canopy layers (0 = use dz_tall/dz_short)
nlayer_within: int = 0  # Number of within-canopy layers (0 = use dz_tall/dz_short)


# ---------------------------------------------------------------------------
# Miscellaneous — Fortran lines 52-54
# ---------------------------------------------------------------------------

mlcan_to_clm: int = 0  # Pass MLcanopy fluxes to CLM/CAM: no (0), yes (1)
ml_vert_init: int = ispval  # Flag: initialize multilayer canopy vertical structure and profiles


# ---------------------------------------------------------------------------
# Namelist-overridable defaults — Fortran lines 56-63
# These can also be set via the clmML_inparm namelist (controlMod.control()).
# Tower-specific values override these defaults.
# ---------------------------------------------------------------------------

met_type: int = 0  # Meteorological forcing for MLcanopy timestep:
#   0 = no interpolation (standard CLM calendar)
#   2 = 2-point interpolation (not supported)
#   3 = 3-point interpolation (time centred, as in CHATS)

dpai_min: float = 0.01  # Minimum plant area index to treat as a vegetation layer (m2/m2)

pftcon_val: int = 0  # PFT parameters: default values (0) or CHATS overrides (1)
