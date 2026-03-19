"""Spectral Shallow Water Model using spherical harmonic transforms.

Solves the rotating shallow water equations on the sphere using the
vorticity-divergence formulation with pseudospectral (transform) method.

Prognostic variables (in spectral space):
    vor_hat : Relative vorticity SH coefficients
    div_hat : Divergence SH coefficients
    phi_hat : Geopotential (g*h) SH coefficients

Equations:
    d(vor)/dt = -div((vor+f)*v)
    d(div)/dt = curl((vor+f)*v) - laplacian(E + phi + phi_s)
    d(phi)/dt = -div(phi*v)

References
----------
- Hack, J. J. & Jakob, R. (1992). Description of a Global Shallow Water Model
  Based on the Spectral Transform Method. NCAR TN-343+STR.
- Williamson, D. L. et al. (1992). A standard test set for shallow water
  equations in spherical geometry. J. Comput. Phys., 102, 211-224.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.gaussian import (
    GaussianGrid,
    sh_analysis,
    sh_synthesis,
    sh_analysis_oc2,
    sh_analysis_dmu,
    uv_from_vordiv,
    spectral_hyperdiffusion,
)
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm import constants


# =============================================================================
# State and config
# =============================================================================

class SpectralSWState(NamedTuple):
    """State for the spectral shallow water model.

    All fields contain complex SH coefficients of shape (n_sh,).
    """
    vor_hat: Field   # Spectral relative vorticity [1/s]
    div_hat: Field   # Spectral divergence [1/s]
    phi_hat: Field   # Spectral geopotential [m^2/s^2]
    phis_hat: Field  # Spectral surface geopotential [m^2/s^2] (static)


class SpectralSWConfig(NamedTuple):
    """Configuration for the spectral shallow water model."""
    g: float = constants.g
    mean_depth: float = 5960.0        # H_0 for linearized mass equation [m]
    hyperdiff_coeff: float = 2.338e15 # Spectral diffusion coefficient
    hyperdiff_order: int = 2          # Diffusion order (2 = nabla^4)
    spectral_filter_order: int = 0    # Exponential filter order (0 = off)
                                      # Recommended: 8 for runs with topography
    spectral_filter_cutoff: float = 0.65  # Filter value at n_max


# =============================================================================
# Tendency computation
# =============================================================================

def spectral_sw_tendencies(
    state: SpectralSWState,
    grid: GaussianGrid,
    config: SpectralSWConfig,
) -> SpectralSWState:
    """Compute spectral tendencies for the shallow water equations.

    Uses the pseudospectral transform method:
    1. Transform prognostic fields to grid space
    2. Compute nonlinear products on grid
    3. Transform products to spectral space
    4. Assemble tendencies using spectral operators

    Returns tendencies in the same pytree structure as state (for SSP-RK3).
    """
    a = grid.radius

    # --- 1. Transform to grid space ---
    vor = sh_synthesis(grid, state.vor_hat.data)     # (n_lat, n_lon)
    phi = sh_synthesis(grid, state.phi_hat.data)
    phis = sh_synthesis(grid, state.phis_hat.data)

    # --- 2. Compute velocities ---
    u_cos, v_cos = uv_from_vordiv(grid, state.vor_hat.data, state.div_hat.data)
    # u_cos = u * cos(lat), v_cos = v * cos(lat)
    cos_lat_2d = grid.cos_lat[:, None]
    u = u_cos / cos_lat_2d
    v = v_cos / cos_lat_2d

    # --- 3. Nonlinear products on grid ---
    abs_vor = vor + grid.f                # Absolute vorticity (zeta + f)
    kinetic_energy = 0.5 * (u * u + v * v)

    # --- 4. Transform nonlinear products to spectral space ---
    # Spectral div/curl on sphere (Hack & Jakob 1992, Bourke 1972):
    #   div_hat = (im/a)*sh_analysis_oc2(U) - (1/a)*sh_analysis_dmu(V)
    #   curl_hat = (im/a)*sh_analysis_oc2(V) + (1/a)*sh_analysis_dmu(U)
    # where U = flux_lon*cosφ, V = flux_mer*cosφ.
    # The 1/cos²φ weighting is baked into Pnm_oc2 and Dnm matrices (pole-safe).

    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a

    # Vorticity fluxes: U = (ζ+f)*u*cosφ, V = (ζ+f)*v*cosφ
    A_vor = abs_vor * u_cos                     # (ζ+f)*u*cosφ
    B_vor = abs_vor * v_cos                     # (ζ+f)*v*cosφ

    # Vorticity equation: dζ/dt = -div((ζ+f)*v)
    flux_vor_div = (im_over_a * sh_analysis_oc2(grid, A_vor)
                    - one_over_a * sh_analysis_dmu(grid, B_vor))

    # Divergence equation: dδ/dt = curl((ζ+f)*v) - lap*(E+Φ+Φs)
    flux_vor_curl = (im_over_a * sh_analysis_oc2(grid, B_vor)
                     + one_over_a * sh_analysis_dmu(grid, A_vor))

    # Mass fluxes: U = Φ*u*cosφ, V = Φ*v*cosφ
    A_mass = phi * u_cos
    B_mass = phi * v_cos

    # Mass equation: dΦ/dt = -div(Φ*v)
    flux_mass_div = (im_over_a * sh_analysis_oc2(grid, A_mass)
                     - one_over_a * sh_analysis_dmu(grid, B_mass))

    # Kinetic energy + geopotential + surface geopotential -> Laplacian term
    E_phi = kinetic_energy + phi + phis
    E_phi_hat = sh_analysis(grid, E_phi)

    # --- 5. Assemble tendencies ---
    # d(vor_hat)/dt = -div((zeta+f)*v)
    dvor_hat = -flux_vor_div

    # d(div_hat)/dt = curl((zeta+f)*v) - laplacian(E + phi + phis)
    ddiv_hat = flux_vor_curl - grid.lap * E_phi_hat

    # d(phi_hat)/dt = -div(phi*v)
    dphi_hat = -flux_mass_div

    # --- 6. Spectral hyperdiffusion (vorticity & divergence only) ---
    # Hyperdiffusion is NOT applied to phi (geopotential / mass) because:
    #   - It would violate mass distribution conservation
    #   - It causes spurious energy drift
    #   - Standard practice (Hack & Jakob 1992) diffuses only vor and div
    if config.hyperdiff_coeff > 0:
        dvor_hat = dvor_hat + spectral_hyperdiffusion(
            grid, state.vor_hat.data, config.hyperdiff_coeff, config.hyperdiff_order)
        ddiv_hat = ddiv_hat + spectral_hyperdiffusion(
            grid, state.div_hat.data, config.hyperdiff_coeff, config.hyperdiff_order)

    # Return as same pytree structure (for SSP-RK3 tree_map)
    return SpectralSWState(
        vor_hat=state.vor_hat.replace(data=dvor_hat),
        div_hat=state.div_hat.replace(data=ddiv_hat),
        phi_hat=state.phi_hat.replace(data=dphi_hat),
        phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
    )


# =============================================================================
# Model class
# =============================================================================

class SpectralShallowWaterModel:
    """Spectral shallow water model on the sphere.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with precomputed SH transform matrices.
    config : SpectralSWConfig, optional
        Model configuration.
    """

    def __init__(
        self,
        grid: GaussianGrid,
        config: SpectralSWConfig | None = None,
        *,
        allow_unsupported_backend: bool = False,
        legoesm_config=None,
    ):
        self.config = config or SpectralSWConfig()
        self._use_cpu_for_spectral = False
        self._cpu_device = None
        self._default_device = None

        # Precompute exponential spectral filter if enabled.
        # The filter damps high-wavenumber spectral coefficients to
        # suppress Gibbs ringing from non-smooth fields (e.g. conical
        # topography in Williamson TC5).  Applied after each time step.
        if self.config.spectral_filter_order > 0:
            alpha = -jnp.log(jnp.float64(self.config.spectral_filter_cutoff))
            ratio = grid.ls.astype(jnp.float64) / grid.n_max
            self._spectral_filter = jnp.exp(
                -alpha * ratio ** self.config.spectral_filter_order
            )
        else:
            self._spectral_filter = None

        # Extract allow_unsupported from global config if provided
        if legoesm_config is not None:
            allow_unsupported_backend = bool(
                legoesm_config.get(
                    "atmosphere.spectral.allow_unsupported", False
                )
            )

        # Detect Metal backend: auto-route spectral to CPU.
        from legoesm.core.hardware import get_backend
        backend = get_backend()
        if backend == "METAL":
            self._use_cpu_for_spectral = True
            self._cpu_device = jax.devices("cpu")[0]
            self._default_device = jax.devices()[0]
            # Transfer grid to CPU so spectral transforms happen there.
            self.grid = jax.device_put(grid, self._cpu_device)
        else:
            self.grid = grid
            # Guard: verify backend supports float64/complex128
            from legoesm.core.hardware import check_spectral_backend
            check_spectral_backend(
                allow_unsupported=allow_unsupported_backend
            )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: SpectralSWState, dt: float) -> SpectralSWState:
        """Advance one time step using SSP-RK3.

        On Metal, transfers state to CPU for computation, then back.
        If a spectral filter is enabled, it is applied after each step
        to suppress Gibbs ringing from topography or other discontinuities.
        """
        def tendency_fn(s):
            return spectral_sw_tendencies(s, self.grid, self.config)

        if self._use_cpu_for_spectral:
            state_cpu = jax.device_put(state, self._cpu_device)
            result_cpu = ssp_rk3_step(state_cpu, tendency_fn, dt)
            result_cpu = self._apply_filter(result_cpu)
            return jax.device_put(result_cpu, self._default_device)

        result = ssp_rk3_step(state, tendency_fn, dt)
        return self._apply_filter(result)

    def _apply_filter(self, state: SpectralSWState) -> SpectralSWState:
        """Apply exponential spectral filter to all prognostic fields."""
        if self._spectral_filter is None:
            return state
        sf = self._spectral_filter
        return SpectralSWState(
            vor_hat=state.vor_hat.replace(data=state.vor_hat.data * sf),
            div_hat=state.div_hat.replace(data=state.div_hat.data * sf),
            phi_hat=state.phi_hat.replace(data=state.phi_hat.data * sf),
            phis_hat=state.phis_hat,  # topography is static — never filter
        )

    def filter_initial_state(self, state: SpectralSWState) -> SpectralSWState:
        """Filter all initial spectral fields including topography.

        Call this once before time integration when the initial conditions
        contain non-smooth fields (e.g. conical/step topography in TC5).
        The spectral filter removes Gibbs oscillations that would otherwise
        cause nonlinear instability.

        If no spectral filter is configured, returns the state unchanged.
        """
        if self._spectral_filter is None:
            return state
        sf = self._spectral_filter
        return SpectralSWState(
            vor_hat=state.vor_hat.replace(data=state.vor_hat.data * sf),
            div_hat=state.div_hat.replace(data=state.div_hat.data * sf),
            phi_hat=state.phi_hat.replace(data=state.phi_hat.data * sf),
            phis_hat=state.phis_hat.replace(data=state.phis_hat.data * sf),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step_on_cpu(self, state: SpectralSWState, dt: float) -> SpectralSWState:
        """Step without device transfers (for batched CPU integration on Metal)."""
        def tendency_fn(s):
            return spectral_sw_tendencies(s, self.grid, self.config)
        return ssp_rk3_step(state, tendency_fn, dt)

    def integrate(
        self,
        state: SpectralSWState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ) -> tuple[SpectralSWState, list]:
        """Integrate forward for a given duration (Python loop).

        On Metal, batches CPU transfers: transfer state to CPU once,
        run all steps on CPU, then transfer results back to Metal.
        This avoids per-step CPU↔Metal round-trips.
        """
        n_steps = int(duration / dt)

        if self._use_cpu_for_spectral:
            return self._integrate_on_cpu(state, n_steps, dt, save_every)

        trajectory = [state]
        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
        return state, trajectory

    def _integrate_on_cpu(self, state, n_steps, dt, save_every):
        """Batch integration on CPU: transfer once, not per step."""
        state_cpu = jax.device_put(state, self._cpu_device)
        trajectory_cpu = [state_cpu]

        for i in range(n_steps):
            state_cpu = self._step_on_cpu(state_cpu, dt)
            if (i + 1) % save_every == 0:
                trajectory_cpu.append(state_cpu)

        # Transfer back to Metal
        state_out = jax.device_put(state_cpu, self._default_device)
        trajectory_out = [
            jax.device_put(s, self._default_device) for s in trajectory_cpu
        ]
        return state_out, trajectory_out


# =============================================================================
# Test case initialization
# =============================================================================

def williamson_test2_spectral(grid: GaussianGrid) -> SpectralSWState:
    """Williamson Test Case 2 in spectral space: steady geostrophic flow.

    Solid body rotation u = u_0 * cos(lat), v = 0, with height in
    geostrophic balance.
    """
    R = grid.radius
    Omega = constants.Omega
    g = constants.g

    u_0 = 2.0 * jnp.pi * R / (12.0 * 86400.0)
    gh_0 = 2.94e4

    lat2d = grid.lat2d
    sin_lat_2d = jnp.sin(lat2d)

    # Height field (geostrophic balance)
    h = (gh_0 - (R * Omega * u_0 + u_0**2 / 2.0) * sin_lat_2d**2) / g
    phi = g * h  # geopotential

    # Vorticity and divergence on grid
    # For solid body rotation: zeta = +2*u_0/R * sin(lat) (relative vorticity)
    # Divergence = 0
    vor = 2.0 * u_0 / R * sin_lat_2d
    div_field = jnp.zeros_like(vor)

    # No topography
    phis = jnp.zeros_like(phi)

    # Transform to spectral space
    dims = ("spectral",)
    vor_hat = sh_analysis(grid, vor)
    div_hat = sh_analysis(grid, div_field)
    phi_hat = sh_analysis(grid, phi)
    phis_hat = sh_analysis(grid, phis)

    return SpectralSWState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims, units="1/s"),
        phi_hat=Field(data=phi_hat, name="phi_hat", dims=dims, units="m^2/s^2"),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims, units="m^2/s^2"),
    )


def williamson_test5_spectral(grid: GaussianGrid) -> SpectralSWState:
    """Williamson Test Case 5 in spectral space: zonal flow over mountain.

    Same flow as Test 2 but with u_0 = 20 m/s and an isolated mountain.
    """
    R = grid.radius
    Omega = constants.Omega
    g = constants.g

    u_0 = 20.0
    gh_0 = 5960.0 * g

    lat2d = grid.lat2d
    lon2d = grid.lon2d
    cos_lat_2d = jnp.cos(lat2d)
    sin_lat_2d = jnp.sin(lat2d)

    # Vorticity and divergence
    vor = 2.0 * u_0 / R * sin_lat_2d
    div_field = jnp.zeros_like(vor)

    # Mountain topography
    lon_c = 3.0 * jnp.pi / 2.0   # 270E
    lat_c = jnp.pi / 6.0          # 30N
    R_m = jnp.pi / 9.0            # 20 degrees radius
    h_s0 = 2000.0                  # peak height [m]

    r = jnp.arccos(jnp.clip(
        jnp.sin(lat_c) * sin_lat_2d +
        jnp.cos(lat_c) * cos_lat_2d * jnp.cos(lon2d - lon_c),
        -1.0, 1.0,
    ))
    h_s = jnp.where(r < R_m, h_s0 * (1.0 - r / R_m), 0.0)
    phis = g * h_s

    # Height field: free surface from geostrophic balance, then subtract mountain
    # phi = g * fluid_depth, where fluid_depth = free_surface_height - h_s
    h_free = (gh_0 - (R * Omega * u_0 + u_0**2 / 2.0) * sin_lat_2d**2) / g
    phi = g * (h_free - h_s)

    # Transform to spectral space
    dims = ("spectral",)
    vor_hat = sh_analysis(grid, vor)
    div_hat = sh_analysis(grid, div_field)
    phi_hat = sh_analysis(grid, phi)
    phis_hat = sh_analysis(grid, phis)

    return SpectralSWState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims, units="1/s"),
        phi_hat=Field(data=phi_hat, name="phi_hat", dims=dims, units="m^2/s^2"),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims, units="m^2/s^2"),
    )


# =============================================================================
# Diagnostic utilities
# =============================================================================

def spectral_to_grid(
    state: SpectralSWState,
    grid: GaussianGrid,
) -> dict[str, jax.Array]:
    """Convert spectral state to grid-point fields for diagnostics.

    Returns
    -------
    dict with keys: 'h', 'u', 'v', 'vor', 'div', 'phi', 'phis'
        All arrays have shape (n_lat, n_lon).
    """
    g = constants.g
    vor = sh_synthesis(grid, state.vor_hat.data)
    div = sh_synthesis(grid, state.div_hat.data)
    phi = sh_synthesis(grid, state.phi_hat.data)
    phis = sh_synthesis(grid, state.phis_hat.data)

    h = phi / g
    h_s = phis / g

    u_cos, v_cos = uv_from_vordiv(grid, state.vor_hat.data, state.div_hat.data)
    cos_lat_2d = grid.cos_lat[:, None]
    u = u_cos / cos_lat_2d
    v = v_cos / cos_lat_2d

    return {
        'h': h, 'u': u, 'v': v,
        'vor': vor, 'div': div,
        'phi': phi, 'phis': phis,
        'h_s': h_s,
    }


def compute_spectral_diagnostics(
    state: SpectralSWState,
    grid: GaussianGrid,
) -> dict[str, float]:
    """Compute conservation diagnostics for the spectral model.

    Returns mass, energy, and enstrophy integrals.
    """
    g = constants.g
    fields = spectral_to_grid(state, grid)
    h = fields['h']
    u = fields['u']
    v = fields['v']
    vor = fields['vor']

    w = grid.weights[:, None]  # (n_lat, 1)
    dlon = 2.0 * jnp.pi / grid.n_lon
    a2 = grid.radius * grid.radius

    # Integration weight: w * dlon * a^2
    # Gaussian weights integrate over μ=sin(lat), so dA = a² dμ dλ = a² w dlon
    dA = w * dlon * a2

    # Total mass: integral of h
    mass = jnp.sum(h * dA)

    # Total energy: integral of h*(u^2+v^2)/2 + g*h^2/2
    energy = jnp.sum((0.5 * h * (u**2 + v**2) + 0.5 * g * h**2) * dA)

    # Potential enstrophy: integral of (vor+f)^2 / (2*h)
    abs_vor = vor + grid.f
    enstrophy = jnp.sum(abs_vor**2 / (2.0 * h) * dA)

    return {
        'mass': float(mass),
        'energy': float(energy),
        'enstrophy': float(enstrophy),
    }
