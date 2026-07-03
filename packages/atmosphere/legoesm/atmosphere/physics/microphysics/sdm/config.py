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


__param_spec__ = {
    "SDMConfig": {
        "scheme_key": "atm.sdm.SDMConfig",
        "excluded": {
            "adaptive_cfl": "numerics: adaptive-substep CFL (dt = cfl/|tau|), solver control",
            "adaptive_stol": "numerics: ERF steady-state exit tolerance (solver convergence)",
            "newton_atol": "numerics: Newton absolute-residual exit tolerance",
            "newton_rtol": "numerics: Newton relative-residual exit tolerance",
            "newton_stol": "numerics: Newton step-size exit tolerance (solver convergence)",
            "column_n_rain_floor": "numerics: fallback rain number when reconstructed Nr<=0 (degenerate-case floor, not a closure)",
            "r_cloud": "diagnostic: activated-cloud size cutoff for the Eulerian q_c output of the advected Lagrangian path (measurement convention, not a physics closure; total water is conserved regardless)",
        },
        "params": {
            "cdnc": {"units": "1/m^3", "bounds": (33000000.0, 300000000.0), "tunable_tier": 1, "transform": "sigmoid", "category": "droplet_number", "reference": "Shima et al. (2009) prescribed cloud-droplet number", "shape": None},
            "golovin_b": {"units": "1/s", "bounds": (495.0, 4500.0), "tunable_tier": 1, "transform": "sigmoid", "category": "collision_coalescence", "reference": "Golovin (1963) additive coalescence kernel coefficient", "shape": None},
            "r_rain": {"units": "m", "bounds": (1.32e-05, 0.00012), "tunable_tier": 2, "transform": "sigmoid", "category": "size_threshold", "reference": "cloud/rain droplet-radius partition (40 um)", "shape": None},
            "r_min_reconstruct": {"units": "m", "bounds": (3.3e-07, 3e-06), "tunable_tier": 3, "transform": "sigmoid", "category": "size_reconstruction", "reference": "minimum reconstructed mean-droplet radius", "shape": None},
            "solute_ionization": {"units": "1", "bounds": (0.66, 6.0), "tunable_tier": 3, "transform": "sigmoid", "category": "kohler", "reference": "van 't Hoff factor i (NaCl=2)", "shape": None},
            "solute_molar_mass": {"units": "kg/mol", "bounds": (0.0192852, 0.17532), "tunable_tier": 3, "transform": "sigmoid", "category": "kohler", "reference": "aerosol solute molar mass (NaCl=0.05844)", "shape": None},
        },
    },
}


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
        reverse-mode differentiable — ``"rk4_adaptive"`` (the ERF
        stiffness-based explicit integrator: per-droplet ``dt = cfl/|τ|``,
        stage-positivity step-halving, unconverged/steady exits), or the
        implicit Newton family sharing the same adaptive outer loop:
        ``"be"`` (backward Euler), ``"cn"`` (Crank-Nicolson), ``"dirk2"``
        (ERF dirk212, 2-stage DIRK) — unconditionally stable for the stiff
        Köhler terms. The adaptive integrators are NOT reverse-mode
        differentiable. Unknown values raise. This completes the ERF
        ``SDMassChangeTIMethod`` enum (RK3BS's role is covered by the
        stiffness-adaptive rk4).
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
    newton_rtol, newton_atol, newton_stol : float
        Newton-solver tolerances for the implicit ``"be"`` integrator
        (relative/absolute residual, step-size exit — ERF
        ``m_newton_{rtol,atol,stol}``).
    newton_maxits : int
        Newton iteration cap (ERF ``m_newton_maxits``).
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
    collision_mode : str
        Collision update mode. ``"stochastic"`` (default) is the Shima
        Monte-Carlo integer-collision algorithm with random pairing/rounding.
        ``"deterministic"`` is an opt-in mean-field update for the persistent
        Lagrangian path: candidate pairs use the expected coalescence increment
        with a smooth cap at the available multiplicity, no accept/reject draw,
        and no PRNG consumption. It is intended for reverse-mode sensitivity
        tests and deterministic optimization experiments.
    r_rain : float
        Radius threshold [m] separating cloud water from rain when depositing
        super-droplet liquid to grid mixing ratios (ERF default 40 um).
    r_cloud : float
        Minimum activated-droplet radius [m] counted as diagnostic Eulerian
        cloud water by the advected Lagrangian SDM path. Smaller wet aerosol
        and haze still carry liquid mass, exchange vapor/heat, collide, and
        conserve total water, but are not written to ``q_c`` where they would
        make clear air appear cloudy.
    lagrangian_diagnostic_assignment : str
        Particle-to-mesh assignment used when the advected Lagrangian LES path
        bins particles back to diagnostic Eulerian ``q_c/q_r``. ``"cic"`` is
        conservative cell-centred cloud-in-cell deposition and avoids
        nearest-cell Monte-Carlo speckle; ``"nearest"`` preserves the original
        cell-bin diagnostic for tests and debugging.
    column_do_coalescence : bool
        Opt-in stateless Eulerian column adapter mode. False (default) keeps
        the legacy condensation-only mean-droplet closure. True reconstructs a
        per-cell super-droplet population from the Eulerian bulk liquid fields,
        advances one ``box_step`` with condensation + Shima coalescence, and
        projects back to bulk cloud/rain mass and number tendencies. This is a
        per-step reconstructed well-mixed box, NOT faithful advected
        Lagrangian SDM; the population and PRNG stream are reset every column
        call.
    column_n_sd : int
        Number of super-droplet slots reconstructed per Eulerian cell in
        ``column_do_coalescence`` mode. Half are initialized from the cloud
        bulk mode and half from the rain mode. Must be >= 4 when coalescence is
        enabled.
    column_seed : int
        Fixed PRNG seed for stateless column reconstruction/coalescence. Because
        the microphysics dispatch is keyless and has no step counter, this is
        deterministic for a given cell index and call. That makes the adapter
        reproducible but can bias long integrations; faithful SDM needs a
        threaded PRNG key with persistent particles.
    column_n_rain_floor : float
        Rain number concentration [1/m^3] used to reconstruct rain mass when
        ``q_r > 0`` but the Eulerian ``N_r`` slot is zero or absent.
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
    newton_rtol: float = 1.0e-8           # [-] Newton relative-residual exit
    newton_atol: float = 1.0e-40          # [m^2/s] Newton absolute-residual exit
    newton_stol: float = 1.0e-10          # [-] Newton step-size exit
    newton_maxits: int = 30               # [-] Newton iteration cap
    include_curvature: bool = True
    include_solute: bool = True
    solute_ionization: float = 2.0        # van't Hoff i for NaCl
    solute_molar_mass: float = 0.05844    # [kg/mol] NaCl
    collision_kernel: str = "golovin"
    terminal_velocity: str = "rogers_yau"
    golovin_b: float = 1.5e3              # [1/s] Golovin kernel coefficient
    include_brownian: bool = False        # add Brownian coagulation to the kernel
    collision_mode: str = "stochastic"    # stochastic | deterministic
    r_rain: float = 4.0e-5               # [m] cloud/rain radius threshold (40 um)
    column_do_coalescence: bool = False  # opt-in reconstructed-box coalescence
    column_n_sd: int = 64                # [-] super-droplets per cell in column box
    column_seed: int = 0                 # [-] fixed key seed for keyless dispatch
    column_n_rain_floor: float = 1.0e6   # [1/m^3] fallback rain number
    cdnc: float = 1.0e8                  # [1/m^3] prescribed cloud-droplet number
    r_min_reconstruct: float = 1.0e-6   # [m] min reconstructed mean-droplet radius
    r_cloud: float = 2.0e-6             # [m] activated-cloud diagnostic threshold
    lagrangian_diagnostic_assignment: str = "cic"
