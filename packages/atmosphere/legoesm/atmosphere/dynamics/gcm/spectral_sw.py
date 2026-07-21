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
from legoesm.parallel.metal import place_spectral_grid
from legoesm.grids.gaussian import (
    GaussianGrid,
    sh_analysis,
    sh_synthesis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
    uv_from_vordiv,
    spectral_hyperdiffusion_3d,
    dealiasing_mask,
)
from legoesm.timestepping.dispatch import dispatch_integrator
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
    """Configuration for the spectral shallow water model.

    The background depth H_0 is not a config knob: the mass equation is
    fully nonlinear in the prognostic ``phi_hat``, so the mean depth is
    carried by the initial geopotential field.
    """
    g: float = constants.g
    hyperdiff_coeff: float = 2.338e15 # Spectral diffusion coefficient
    hyperdiff_order: int = 2          # Diffusion order (2 = nabla^4)
    spectral_filter_order: int = 0    # Exponential filter order (0 = off)
                                      # Recommended: 8 for runs with topography
    spectral_filter_cutoff: float = 0.65  # Filter value at n_max
    time_integrator: str = "ssp_rk3"  # Any integrator from dispatch
    # Orszag 2/3-rule de-aliasing of the quadratic nonlinear tendencies.
    # The shallow-water vorticity/divergence/mass tendencies are
    # quadratic in the prognostic fields ((zeta+f)*v, Phi*v, |v|^2), so
    # without this truncation the products fold aliased power back into
    # the resolved spectrum and drive grid-scale instability.  Default
    # 2/3 (Orszag 1971) is standard practice; 0.0 disables (e.g. for
    # exact rest-state / single-mode tendency tests).  Only the upper
    # 1 - fraction of wavenumbers (spurious aliasing band) is removed,
    # so well-resolved fields are unchanged.  See
    # ``grids.gaussian.dealiasing_mask``.
    dealiasing_fraction: float = 0.667
    # --- New fields APPENDED (codex review: inserting before existing
    # fields breaks positional NamedTuple construction for callers) ---
    # Apply the exponential filter after EVERY step (True, historical
    # default) or only on demand via ``filter_initial_state`` (False).
    # The per-step application compounds: at T21 with dt=600 s the
    # order-8 / cutoff-0.65 filter multiplies n=10 by 0.99886 per step,
    # i.e. e^-16 over a 100-day run — it silently annihilates every
    # scale above n~7.  Free-evolution cases without persistent Gibbs
    # sources (e.g. colliding modons: smooth ICs, no topography) should
    # set False and rely on hyperdiffusion + de-aliasing; forced cases
    # with non-smooth stationary topography (Williamson 5) keep True.
    spectral_filter_every_step: bool = True


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
    # Stack {vor, phi, phis} along a trailing axis so a single
    # ``sh_synthesis_3d`` (one segment_sum + one IRFFT) replaces three
    # sequential ``sh_synthesis`` calls.  The 3D variant treats the
    # trailing axis as a passive batch — for SW (no level dim) the
    # trailing-axis-of-3 plays the role of nlev=3.  3 SH-syntheses → 1.
    state.vor_hat.data.shape[0]
    _vpp_stack = jnp.stack(
        [state.vor_hat.data, state.phi_hat.data, state.phis_hat.data],
        axis=-1,
    )  # (n_sh, 3)
    _vpp_grid = sh_synthesis_3d(grid, _vpp_stack)  # (n_lat, n_lon, 3)
    vor = _vpp_grid[..., 0]
    phi = _vpp_grid[..., 1]
    phis = _vpp_grid[..., 2]

    # --- 2. Compute cos-lat-weighted velocities (pole-safe) ---
    u_cos, v_cos = uv_from_vordiv(grid, state.vor_hat.data, state.div_hat.data)
    # u_cos = u * cos(lat), v_cos = v * cos(lat)
    # Physical u, v are NOT computed here to avoid 1/cos(lat) singularity
    # at the poles.  Kinetic energy uses sh_analysis_oc2 instead (see below).

    # --- 3. Nonlinear products on grid ---
    abs_vor = vor + grid.f                # Absolute vorticity (zeta + f)

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

    # Mass fluxes: U = Φ*u*cosφ, V = Φ*v*cosφ
    A_mass = phi * u_cos
    B_mass = phi * v_cos

    # Kinetic energy on the grid (KE·cos²φ; 1/cos²φ baked into Pnm_oc2).
    KE_cos2 = 0.5 * (u_cos * u_cos + v_cos * v_cos)  # KE·cos²φ

    # Batch the SH analyses: oc2 needs {A_vor, B_vor, A_mass, KE_cos2}
    # (4 calls), dmu needs {A_vor, B_vor, B_mass} (3 calls), plain
    # sh_analysis needs {phi+phis} (1 call).  Each variant treats the
    # trailing axis as passive batch, so stack along trailing axis and
    # call once on a thicker (n_lat, n_lon, K) tensor.  8 SH-analyses
    # collapse to 3 (one batched per variant).
    _phi_total = phi + phis
    _oc2_stack = jnp.stack([A_vor, B_vor, A_mass, KE_cos2], axis=-1)  # (..., 4)
    _dmu_stack = jnp.stack([A_vor, B_vor, B_mass], axis=-1)            # (..., 3)
    A_vor_oc2, B_vor_oc2, A_mass_oc2, KE_oc2 = jnp.moveaxis(
        sh_analysis_oc2_3d(grid, _oc2_stack), -1, 0,
    )
    A_vor_dmu, B_vor_dmu, B_mass_dmu = jnp.moveaxis(
        sh_analysis_dmu_3d(grid, _dmu_stack), -1, 0,
    )
    phi_total_hat = sh_analysis(grid, _phi_total)

    # Vorticity equation: dζ/dt = -div((ζ+f)*v)
    flux_vor_div = im_over_a * A_vor_oc2 - one_over_a * B_vor_dmu

    # Divergence equation: dδ/dt = curl((ζ+f)*v) - lap*(E+Φ+Φs)
    flux_vor_curl = im_over_a * B_vor_oc2 + one_over_a * A_vor_dmu

    # Mass equation: dΦ/dt = -div(Φ*v)
    flux_mass_div = im_over_a * A_mass_oc2 - one_over_a * B_mass_dmu

    # Kinetic energy + geopotential + surface geopotential -> Laplacian term
    E_phi_hat = KE_oc2 + phi_total_hat

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
        # Batch the two pointwise hyperdiffusions (vor, div) into one
        # call by stacking along a trailing axis.  ``spectral_hyperdiffusion_3d``
        # treats the trailing axis as a passive batch (the operator is
        # purely ``damping * coeffs``), so SW's 2D (n_sh,) inputs work
        # natively as (n_sh, 2).  2 kernel launches → 1.  Same exploit
        # as Loops 120/121 for spectral PE/NH.
        _vd_hat = jnp.stack(
            [state.vor_hat.data, state.div_hat.data], axis=-1,
        )  # (n_sh, 2)
        _hd_pair = spectral_hyperdiffusion_3d(
            grid, _vd_hat, config.hyperdiff_coeff, config.hyperdiff_order,
        )
        dvor_hat = dvor_hat + _hd_pair[..., 0]
        ddiv_hat = ddiv_hat + _hd_pair[..., 1]

    # --- 7. Orszag 2/3-rule de-aliasing of the nonlinear tendencies ---
    # Zero every spectral coefficient with total wavenumber n above
    # ``floor(dealiasing_fraction * n_max)``.  The vor/div/phi tendencies
    # above were built from quadratic products ((zeta+f)*v, Phi*v, |v|^2)
    # whose transform spreads aliased power across the whole spectrum; the
    # mask removes the upper (spurious) band so it cannot fold back and
    # destabilise the run.  Applied LAST (after hyperdiffusion) so the
    # masked modes are held identically at zero — matching spectral_pe.
    # The n=0 mass mode (n << n_cut) is always retained, so the global
    # mass integral d/dt(phi_00) is untouched (mass conservation safe).
    _dealias = dealiasing_mask(grid, config.dealiasing_fraction)
    dvor_hat = dvor_hat * _dealias
    ddiv_hat = ddiv_hat * _dealias
    dphi_hat = dphi_hat * _dealias

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

        # Extract allow_unsupported from global config if provided
        if legoesm_config is not None:
            allow_unsupported_backend = bool(
                legoesm_config.get(
                    "atmosphere.spectral.allow_unsupported", False
                )
            )

        # --- Metal detection MUST happen before any float64 computation ---
        placement = place_spectral_grid(
            grid, allow_unsupported=allow_unsupported_backend
        )
        self.grid = placement.grid
        self._use_cpu_for_spectral = placement.use_cpu_for_spectral
        self._cpu_device = placement.cpu_device
        self._default_device = placement.default_device

        # Precompute exponential spectral filter if enabled.
        # The filter damps high-wavenumber spectral coefficients to
        # suppress Gibbs ringing from non-smooth fields (e.g. conical
        # topography in Williamson TC5).  Applied after each time step.
        # Uses self.grid which is now on CPU when Metal is active.
        if self.config.spectral_filter_order > 0:
            alpha = -jnp.log(jnp.float64(self.config.spectral_filter_cutoff))
            ratio = self.grid.ls.astype(jnp.float64) / self.grid.n_max
            self._spectral_filter = jnp.exp(
                -alpha * ratio ** self.config.spectral_filter_order
            )
        else:
            self._spectral_filter = None

        # State-truncation mask for the Orszag 2/3 rule.  The TENDENCY is
        # masked inside ``spectral_sw_tendencies``, which holds masked
        # modes constant — it cannot remove upper-third power already in
        # the state (e.g. Williamson-5's conical mountain imprinted on
        # the prognostic phi via phi = g(h - h_s), or arbitrary user
        # ICs).  Truncating the STATE after each step enforces the
        # band-limit the Orszag rule assumes; for band-limited states
        # this multiply by a 0/1 mask is an exact no-op.
        if self.config.dealiasing_fraction > 0.0:
            self._dealias_state = dealiasing_mask(
                self.grid, self.config.dealiasing_fraction,
            )
        else:
            self._dealias_state = None

        # Warn if de-aliasing is off — the SW nonlinear tendencies are
        # quadratic and alias without the Orszag 2/3 truncation.
        if self.config.dealiasing_fraction == 0.0:
            import warnings
            warnings.warn(
                "dealiasing_fraction=0.0: spectral aliasing from the "
                "quadratic shallow-water nonlinearities is not suppressed. "
                "Set dealiasing_fraction=0.667 for production runs.",
                stacklevel=2,
            )

    def compute_mass(self, state: SpectralSWState) -> jax.Array:
        """Global ``∫ h dA = (1/g) ∫ phi dA`` (fp64).

        iter-35: API parity with the other SW models (cube FV3Edge/FV3FB
        iter-21, lat-lon SW iter-18, MPAS SW iter-18).  Spectral SW has
        no in-step mass fixer because the SSP-RK3 + spectral-filter
        path is already bit-clean (matrix runner measures ≤ 1.91e-16
        across W2/W5/W6), but exposing the helper keeps the public
        anchor API uniform across all four SW grids.
        """
        from legoesm.grids.gaussian import sh_synthesis
        phi_grid = sh_synthesis(self.grid, state.phi_hat.data)
        h_grid = phi_grid / constants.g
        return jnp.sum(
            h_grid.astype(jnp.float64)
            * self.grid.grid_area.astype(jnp.float64),
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

        integrator = self.config.time_integrator

        if self._use_cpu_for_spectral:
            state_cpu = jax.device_put(state, self._cpu_device)
            result_cpu = dispatch_integrator(state_cpu, tendency_fn, dt, integrator)
            result_cpu = self._apply_filter(result_cpu)
            return jax.device_put(result_cpu, self._default_device)

        result = dispatch_integrator(state, tendency_fn, dt, integrator)
        return self._apply_filter(result)

    def _apply_filter(self, state: SpectralSWState) -> SpectralSWState:
        """Apply the 2/3-rule state truncation + optional exponential filter.

        The truncation keeps the prognostic state inside the de-aliased
        band (masked tendencies alone would FREEZE, not damp, any
        pre-existing upper-third modes).  Topography ``phis_hat`` is
        static forcing and is never touched here.

        The exponential filter participates only when
        ``config.spectral_filter_every_step`` is True — otherwise the
        filter exists solely for ``filter_initial_state`` and this
        method applies the 2/3-rule truncation alone.
        """
        sf = (self._spectral_filter
              if self.config.spectral_filter_every_step else None)
        if self._dealias_state is not None:
            sf = (self._dealias_state if sf is None
                  else sf * self._dealias_state)
        if sf is None:
            return state
        return SpectralSWState(
            vor_hat=state.vor_hat.replace(data=state.vor_hat.data * sf),
            div_hat=state.div_hat.replace(data=state.div_hat.data * sf),
            phi_hat=state.phi_hat.replace(data=state.phi_hat.data * sf),
            phis_hat=state.phis_hat,  # topography is static — never filter
        )

    def filter_initial_state(
        self, state: SpectralSWState, *, include_winds: bool = False,
    ) -> SpectralSWState:
        """Filter topography and geopotential in the initial state.

        By default only filters phi_hat (geopotential) and phis_hat
        (topography) to suppress Gibbs oscillations from non-smooth
        topography (e.g. the conical mountain in TC5).  Vorticity and
        divergence are left unchanged so that the prescribed initial
        wind field is exact (e.g. v=0 for Williamson 5).

        ``include_winds=True`` additionally filters vor_hat/div_hat —
        for wind-defined ICs whose truncation ringing lives in the
        vorticity field (e.g. the near-grid-scale colliding-modon jets
        at T21), used together with
        ``spectral_filter_every_step=False`` so the one-time IC cleanup
        does not become a per-step dissipation.

        If no spectral filter is configured, returns the state unchanged.
        """
        if self._spectral_filter is None:
            return state
        sf = self._spectral_filter
        sf_prog = sf
        if include_winds and self._dealias_state is not None:
            # Also enforce the 2/3-rule band limit on the prognostics:
            # without it, upper-third IC modes (only ATTENUATED by the
            # exponential filter) contaminate every stage of the first
            # nonlinear step before the post-step truncation runs
            # (codex review).  Scoped to include_winds=True (the
            # wind-defined-IC path) to keep the historical phi-only
            # semantics byte-identical for W5/W6.
            sf_prog = sf * self._dealias_state
        vor_hat = state.vor_hat
        div_hat = state.div_hat
        if include_winds:
            vor_hat = vor_hat.replace(data=vor_hat.data * sf_prog)
            div_hat = div_hat.replace(data=div_hat.data * sf_prog)
        return SpectralSWState(
            vor_hat=vor_hat,
            div_hat=div_hat,
            phi_hat=state.phi_hat.replace(data=state.phi_hat.data * sf_prog),
            phis_hat=state.phis_hat.replace(data=state.phis_hat.data * sf),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step_on_cpu(self, state: SpectralSWState, dt: float) -> SpectralSWState:
        """Step without device transfers (for batched CPU integration on Metal).

        Applies the same post-step filter/truncation as :meth:`step` —
        this path previously skipped ``_apply_filter`` entirely, so the
        batched-Metal integration silently ran unfiltered.
        """
        def tendency_fn(s):
            return spectral_sw_tendencies(s, self.grid, self.config)
        result = dispatch_integrator(state, tendency_fn, dt, self.config.time_integrator)
        return self._apply_filter(result)

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
    # Batch the four 2D SH syntheses (vor, div, phi, phis) along a
    # trailing axis — same trailing-axis-passive-batch exploit as the
    # SW tendency block (Loops 95/96 / 101).  4 SH syntheses → 1.
    _vdpp_pair_diag = jnp.stack(
        [
            state.vor_hat.data,
            state.div_hat.data,
            state.phi_hat.data,
            state.phis_hat.data,
        ],
        axis=-1,
    )  # (n_sh, 4)
    _vdpp_grid_diag = sh_synthesis_3d(grid, _vdpp_pair_diag)  # (n_lat, n_lon, 4)
    vor = _vdpp_grid_diag[..., 0]
    div = _vdpp_grid_diag[..., 1]
    phi = _vdpp_grid_diag[..., 2]
    phis = _vdpp_grid_diag[..., 3]

    h = phi / g
    h_s = phis / g

    u_cos, v_cos = uv_from_vordiv(grid, state.vor_hat.data, state.div_hat.data)
    cos_lat_2d = jnp.clip(grid.cos_lat[:, None], 0.01, None)
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
    h_s = fields['h_s']

    w = grid.weights[:, None]  # (n_lat, 1)
    dlon = 2.0 * jnp.pi / grid.n_lon
    a2 = grid.radius * grid.radius

    # Integration weight: w * dlon * a^2
    # Gaussian weights integrate over μ=sin(lat), so dA = a² dμ dλ = a² w dlon
    dA = w * dlon * a2

    # Total mechanical energy density per unit horizontal area:
    #   E = 0.5·h·|v|² + ∫_{h_s}^{h_s+h} g·z dz
    #     = 0.5·h·|v|² + 0.5·g·h² + g·h·h_s
    # The ``g·h·h_s`` topography PE term is essential for any test with
    # non-zero h_s (e.g. Williamson Test 5 isolated mountain) — without
    # it the diagnostic shows spurious "energy non-conservation" even
    # when the prognostic equations conserve total energy exactly.
    abs_vor = vor + grid.f
    _intg = jnp.stack(
        [
            h,
            0.5 * h * (u**2 + v**2) + 0.5 * g * h**2 + g * h * h_s,
            abs_vor**2 / (2.0 * h),
        ],
        axis=-1,
    ) * dA[..., None]
    _diag = jnp.sum(_intg, axis=tuple(range(dA.ndim)))
    mass, energy, enstrophy = _diag[0], _diag[1], _diag[2]

    # One device→host transfer instead of three separate ``float(...)``
    # casts — this diagnostic is called every save_every steps in
    # validation/test loops.
    _h = jax.device_get(jnp.stack([mass, energy, enstrophy]))
    return {
        'mass': float(_h[0]),
        'energy': float(_h[1]),
        'enstrophy': float(_h[2]),
    }
