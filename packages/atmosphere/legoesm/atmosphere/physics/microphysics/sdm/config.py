"""Configuration for the Super-Droplet Method (SDM) microphysics.

All *tunable* scheme parameters live here (audit rule: no magic numbers in JAX
hot loops). Physical *constants* (vapor diffusivity, air thermal conductivity,
water surface tension, gas constants, latent heat, water density) are pulled
from :mod:`legoesm.constants` inside the physics modules, never duplicated here.

The field set grows by iteration as processes are added (collision kernels,
terminal velocity, initialization spectra). Only fields consumed by the current
code are declared, so no dead config fields accumulate.
"""

from __future__ import annotations

from typing import NamedTuple


class SDMConfig(NamedTuple):
    """Super-Droplet Method configuration.

    Fields
    ------
    n_substeps_condensation : int
        Number of equal sub-steps used to integrate the diffusional-growth ODE
        across one physics step. Small droplets near activation are stiff;
        increase this for accuracy. Must be >= 1.
    condensation_integrator : str
        ODE integrator for droplet growth: ``"rk4"`` (4th-order Runge-Kutta,
        default), ``"euler"`` (forward Euler) — both fixed-substep and
        reverse-mode differentiable — or ``"rk4_adaptive"`` (the ERF
        stiffness-based integrator: per-droplet ``dt = cfl/|τ|``,
        stage-positivity step-halving, unconverged/steady exits; NOT
        reverse-mode differentiable). Unknown values raise.
    adaptive_cfl : float
        Stiffness CFL of the adaptive integrator: ``dt = adaptive_cfl/|τ|``
        with ``τ`` the growth-ODE Jacobian (ERF ``mass_change_cfl``).
    adaptive_stol : float
        Steady-state exit tolerance on the relative R² update per step
        (ERF's ``stol = 1e-6``).
    adaptive_max_steps : int
        Cap on ACCEPTED steps of the adaptive loop (ERF ``max_steps = 100``;
        halvings do not consume the budget — they are bounded by the
        too-small exit). Hitting the cap returns the partially integrated
        radius, exactly as ERF does.
    include_curvature : bool
        Include the Kelvin curvature term (raises equilibrium vapor pressure
        over a curved surface). True is the physically complete Köhler growth.
    include_solute : bool
        Include the Raoult solute term (dissolved aerosol lowers equilibrium
        vapor pressure). True is the physically complete Köhler growth.
    solute_ionization : float
        Van't Hoff ionization factor ``i`` of the dissolved aerosol
        (NaCl -> 2). Used to convert solute mass to effective solute moles.
    solute_molar_mass : float
        Molar mass of the dissolved aerosol [kg/mol] (NaCl -> 0.05844).
    collision_kernel : str
        Collision-coalescence kernel: ``"golovin"`` (analytic test kernel,
        default), ``"sedimentation"`` (geometric sweep-out), ``"long"``
        (Long 1974 polynomial efficiency), or ``"hall"`` (Hall 1980 tabulated
        efficiency, bilinear interpolation). Unknown values raise.
    terminal_velocity : str
        Droplet terminal-velocity law: ``"rogers_yau"`` (Stokes R²,
        default), ``"atlas_ulbrich"`` (rain power law), or
        ``"cloud_rain_shima"`` (SCALE-SDM piecewise). Unknown values raise.
    golovin_b : float
        Golovin kernel coefficient ``b`` [1/s] (K = b(X_i+X_j); Shima 2009
        Golovin box test uses b = 1.5e3).
    include_brownian : bool
        Add the Brownian (Seinfeld-Pandis) coagulation coefficient on top of
        the selected collision kernel (ERF ``include_brownian_coalescence``;
        additive, matters only for sub-micron droplets/haze).
    r_rain : float
        Radius threshold [m] separating cloud water from rain when depositing
        super-droplet liquid to grid mixing ratios (ERF default 40 um).
    cdnc : float
        Prescribed cloud-droplet number concentration [1/m^3] used by the
        stateless column operator to reconstruct a mean cloud droplet from the
        grid ``q_c`` (1e8 = maritime). The full Lagrangian model carries
        per-droplet multiplicities instead; this is only the single-step
        column-condensation adapter's closure.
    r_min_reconstruct : float
        Minimum radius [m] of the column operator's reconstructed mean cloud
        droplet (1 um default). For thin cloud the closure reduces the
        *effective droplet number* (``N_eff = min(cdnc, q_c·ρ/m(r_min))``)
        instead of letting the fixed-cdnc inversion produce nm-scale droplets
        — those would be Kelvin-barrier artifacts of the closure. The column
        tendency is then exactly continuous in ``q_c`` (∝ q_c for thin cloud,
        identically zero in clear air) with no cloudy/clear threshold.
    """

    n_substeps_condensation: int = 1
    condensation_integrator: str = "rk4"
    adaptive_cfl: float = 1.0             # [-] dt = cfl/|tau| (ERF mass_change_cfl)
    adaptive_stol: float = 1.0e-6         # [-] steady-state exit (ERF stol)
    adaptive_max_steps: int = 100         # [-] accepted-step cap (ERF max_steps)
    include_curvature: bool = True
    include_solute: bool = True
    solute_ionization: float = 2.0        # van't Hoff i for NaCl
    solute_molar_mass: float = 0.05844    # [kg/mol] NaCl
    collision_kernel: str = "golovin"
    terminal_velocity: str = "rogers_yau"
    golovin_b: float = 1.5e3              # [1/s] Golovin kernel coefficient
    include_brownian: bool = False        # add Brownian coagulation to the kernel
    r_rain: float = 4.0e-5               # [m] cloud/rain radius threshold (40 um)
    cdnc: float = 1.0e8                  # [1/m^3] prescribed cloud-droplet number
    r_min_reconstruct: float = 1.0e-6   # [m] min reconstructed mean-droplet radius
