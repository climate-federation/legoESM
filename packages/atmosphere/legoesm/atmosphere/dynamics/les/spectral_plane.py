"""Spectral-state / filter wrapper around the FD plane dycore.

This is NOT a native spectral dycore. Dynamics still run in physical
space inside the inner FD plane model; the wrapper only adds a
spectral representation at the I/O boundary, an unconditionally
stable spectral biharmonic FILTER applied after each step, and a
2/3-rule dealias mask.

Wraps the finite-difference plane dycore
(:class:`legoesm.atmosphere.dynamics.les.compressible_euler_plane.PlaneCompressibleEulerModel`)
with a 2D-horizontal-Fourier state representation. Dynamics still
run in physical space (the FD plane's Arakawa-C C-grid is already
the discrete adjoint of itself on a periodic uniform mesh, so the
PG/divergence pair is exact at the Fourier-mode level — `grad_x` via
``jnp.roll`` is a discrete shift = FFT phase factor). The spectral
wrapper adds three things the FD path cannot offer:

1. Spectral-state I/O — callers performing ML losses, spectral
   filtering, or scale-decomposition work on the rfft2 coefficients
   directly without per-call transform overhead.
2. Exact biharmonic hyperdiffusion via spectral multiplication by
   ``k⁴`` (the FD composed-Laplacian biharmonic has discrete-stencil
   null-space modes that grow over long integrations).
3. 2/3-rule dealiasing of nonlinear advection products (applied as
   a spectral mask after every step) — suppresses aliased high-``k``
   energy that the FD upwind path damps via stencil dissipation.

Design (mirrors :mod:`legoesm.atmosphere.dynamics.gcm.spectral_pe`)
---------------------------------------------------------------
:class:`SpectralPlaneCompressibleEulerModel.step(spec_state, dt)`:

  1. ``spec → phys`` via ``to_phys`` on every prognostic field.
  2. Run the FD plane model's ``.step(phys_state, dt, physics_fn)``.
  3. ``phys → spec`` via ``to_spec``.
  4. Optionally apply spectral biharmonic hyperdiffusion (exact,
     ``-K · k⁴ · dt``) to ``u_hat, v_hat, w_hat, theta'_hat,
     rho'_hat`` — replacing the FD biharmonic if the caller sets
     ``use_spectral_hyperdiff=True`` in the config (FD biharmonic
     is then turned OFF internally to avoid double-damping).
  5. Apply the 2/3 dealias mask to every prognostic so any
     aliasing introduced by the FD upwind nonlinear products is
     filtered before the next step.

The wrapper is a strict superset of the FD plane dycore at the
"identical config" boundary: setting ``use_spectral_hyperdiff=False``
+ ``apply_dealias=False`` makes the spectral step reproduce the FD
step to round-off (``tests/unit/test_spectral_plane_wrapper.py``
pins this).

State convention
----------------
:class:`legoesm.core.state.SpectralPlanePhysicsState` stores complex
``rfft2`` coefficients of shape ``(ny, nx_r, nlev)`` for cell-centred
fields, ``(ny, nx_r, nlev+1)`` for half-level ``w``, and
``(ny, nx_r, nlev, n_tracers)`` for tracers. ``phis`` (surface
geopotential) is kept REAL because it never participates in the
spectral dynamics; round-tripping it through rfft2/irfftn would just
re-introduce float-cast noise.
"""

from __future__ import annotations

from typing import NamedTuple

import jax

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
)
from legoesm.core.field import Field
from legoesm.core.spectral_plane_ops import (
    SpectralPlaneAxis,
    apply_dealias,
    make_spectral_axis,
    to_phys,
    to_spec,
)
from legoesm.core.state import (
    PlaneNonHydrostaticState,
    SpectralPlanePhysicsState,
)
from legoesm.grids.plane import PlaneGrid
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric


class SpectralPlaneConfig(NamedTuple):
    """Spectral-wrapper extras on top of :class:`CompressibleEulerConfig`.

    Attributes
    ----------
    use_spectral_hyperdiff : bool
        If True, the FD biharmonic hyperdiffusion (set via
        ``CompressibleEulerConfig.hyperdiff_*_coeff``) is DISABLED
        and an exact spectral biharmonic ``-K_h · k⁴`` term is
        applied to every prognostic field instead. ``False`` keeps
        the FD biharmonic.
    apply_dealias : bool
        If True, apply the 2/3-rule dealias mask to every prognostic
        after each step. Suppresses aliasing from the FD upwind
        nonlinear products.
    spectral_hyperdiff_coeff : float
        Coefficient for the spectral biharmonic when
        ``use_spectral_hyperdiff=True``. Same units / role as
        ``CompressibleEulerConfig.hyperdiff_coeff`` (m⁴/s).
    """
    use_spectral_hyperdiff: bool = False
    apply_dealias: bool = True
    spectral_hyperdiff_coeff: float = 1.0e6


class SpectralPlaneCompressibleEulerModel:
    """Spectral-state wrapper around :class:`PlaneCompressibleEulerModel`.

    Holds the FD plane model internally; the only state visible at
    the wrapper's public surface is the spectral representation.
    """

    def __init__(
        self,
        grid: PlaneGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        config: CompressibleEulerConfig,
        spectral_config: SpectralPlaneConfig = SpectralPlaneConfig(),
    ):
        self.grid = grid
        self.height_coord = height_coord
        self.terrain_metric = terrain_metric
        self.spectral_config = spectral_config
        # When the spectral biharmonic replaces the FD one, force the
        # FD coefficients to zero so the same dissipation isn't
        # applied twice.
        if spectral_config.use_spectral_hyperdiff:
            self.config = config._replace(
                hyperdiff_coeff=0.0,
                hyperdiff_rho_coeff=0.0,
                hyperdiff_w_coeff=0.0,
            )
        else:
            self.config = config
        self._fd_model = PlaneCompressibleEulerModel(
            grid, height_coord, terrain_metric, self.config,
        )
        # Build the wavenumber cache once. Dtype matches the grid's
        # area_T (the canonical "what does this plane think x64 is"
        # signal).
        self._axis: SpectralPlaneAxis = make_spectral_axis(
            grid, dealias=spectral_config.apply_dealias,
            dtype=grid.area_T.dtype,
        )

    # ------------------------------------------------------------------
    # Public API: step + transforms
    # ------------------------------------------------------------------

    def step(
        self,
        state: SpectralPlanePhysicsState,
        dt: float,
        physics_fn=None,
    ) -> SpectralPlanePhysicsState:
        """Advance one outer SSP-RK3 step on the spectral state.

        Transform to physical, run the FD dycore step (which already
        handles slow tendency + acoustic substep + Smag + sponge),
        transform back, apply spectral hyperdiff and dealias.
        """
        phys_state = self.to_physical(state)
        phys_state_new = self._fd_model.step(
            phys_state, dt=dt, physics_fn=physics_fn,
        )
        spec_state_new = self.to_spectral(phys_state_new)
        if self.spectral_config.use_spectral_hyperdiff:
            spec_state_new = self._apply_spectral_hyperdiff(
                spec_state_new, dt,
            )
        if self.spectral_config.apply_dealias:
            spec_state_new = self._apply_dealias(spec_state_new)
        return spec_state_new

    def compute_dry_mass(self, state: SpectralPlanePhysicsState) -> jax.Array:
        """Dry-air mass via the FD path (round-trip to physical)."""
        from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
            compute_dry_mass_plane,
        )
        return compute_dry_mass_plane(
            self.to_physical(state),
            self.grid, self.height_coord, self.terrain_metric,
        )

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def to_physical(
        self, state: SpectralPlanePhysicsState,
    ) -> PlaneNonHydrostaticState:
        """Spectral → physical state (rfft2 inverse over horizontal axes)."""
        nx = self.grid.nx
        ny = self.grid.ny
        u = to_phys(state.u_hat.data, nx, ny)
        v = to_phys(state.v_hat.data, nx, ny)
        w = to_phys(state.w_hat.data, nx, ny)
        theta_p = to_phys(state.theta_prime_hat.data, nx, ny)
        rho_p = to_phys(state.rho_prime_hat.data, nx, ny)
        # Tracers carry a trailing tracer axis after the vertical;
        # the FFT axes are still (0, 1) = (y, x), unchanged.
        tracers = to_phys(state.tracers_hat.data, nx, ny)
        return PlaneNonHydrostaticState(
            u=Field(data=u, name="u", dims=("y", "x", "z"), units="m/s",
                    staggering="edge"),
            v=Field(data=v, name="v", dims=("y", "x", "z"), units="m/s",
                    staggering="edge"),
            w=Field(data=w, name="w", dims=("y", "x", "z_half"),
                    units="m/s", staggering="cell"),
            theta_prime=Field(
                data=theta_p, name="theta_prime",
                dims=("y", "x", "z"), units="K", staggering="cell",
            ),
            rho_prime=Field(
                data=rho_p, name="rho_prime",
                dims=("y", "x", "z"), units="kg/m^3", staggering="cell",
            ),
            phis=state.phis,
            tracers=Field(
                data=tracers, name="tracers",
                dims=("y", "x", "z", "tracer"), units="kg/kg",
                staggering="cell",
            ),
        )

    def to_spectral(
        self, state: PlaneNonHydrostaticState,
    ) -> SpectralPlanePhysicsState:
        """Physical → spectral state (rfft2 forward over horizontal axes)."""
        u_hat = to_spec(state.u.data)
        v_hat = to_spec(state.v.data)
        w_hat = to_spec(state.w.data)
        theta_p_hat = to_spec(state.theta_prime.data)
        rho_p_hat = to_spec(state.rho_prime.data)
        tracers_hat = to_spec(state.tracers.data)
        return SpectralPlanePhysicsState(
            u_hat=Field(
                data=u_hat, name="u_hat",
                dims=("ky", "kx", "z"), units="m/s", staggering="spectral",
            ),
            v_hat=Field(
                data=v_hat, name="v_hat",
                dims=("ky", "kx", "z"), units="m/s", staggering="spectral",
            ),
            w_hat=Field(
                data=w_hat, name="w_hat",
                dims=("ky", "kx", "z_half"), units="m/s",
                staggering="spectral",
            ),
            theta_prime_hat=Field(
                data=theta_p_hat, name="theta_prime_hat",
                dims=("ky", "kx", "z"), units="K", staggering="spectral",
            ),
            rho_prime_hat=Field(
                data=rho_p_hat, name="rho_prime_hat",
                dims=("ky", "kx", "z"), units="kg/m^3",
                staggering="spectral",
            ),
            phis=state.phis,
            tracers_hat=Field(
                data=tracers_hat, name="tracers_hat",
                dims=("ky", "kx", "z", "tracer"), units="kg/kg",
                staggering="spectral",
            ),
        )

    # ------------------------------------------------------------------
    # Spectral hyperdiff + dealias
    # ------------------------------------------------------------------

    def _apply_spectral_hyperdiff(
        self,
        state: SpectralPlanePhysicsState,
        dt: float,
    ) -> SpectralPlanePhysicsState:
        """Unconditionally stable spectral biharmonic filter.

        Per-prognostic update is the IMPLICIT-Euler damping
        ``φ_hat_new = φ_hat / (1 + dt · K · k⁴)`` per Fourier mode.
        The amplification factor ``1 / (1 + dt·K·k⁴)`` lies in
        ``(0, 1]`` for any positive ``dt``, ``K``, ``k`` — no CFL
        bound (the explicit form ``φ -= dt·K·k⁴·φ`` would be
        oscillatory at ``dt·K·k_max⁴ > 1`` and unstable at
        ``> 2``). Codex review 2026-05-24 flagged the previous
        explicit-Euler form as unsafe; this implicit filter is
        functionally equivalent in the long-time damping limit and
        removes the CFL guard the explicit form would have required.

        The k=0 mean mode is unaffected (denominator = 1 there), so
        global integrals — dry mass, tracer mean — are preserved
        bit-exactly under the filter.
        """
        K = self.spectral_config.spectral_hyperdiff_coeff
        axis = self._axis
        # Cache the implicit denominator once per call (broadcasts
        # over the trailing vertical / tracer axes via `_bcast` inside
        # `biharmonic_spec`-style broadcasting).
        k4 = axis.k2 ** 2

        def _damp(arr_hat):
            denom_shape = k4.shape + (1,) * (arr_hat.ndim - k4.ndim)
            denom = 1.0 + dt * K * k4.reshape(denom_shape)
            return arr_hat / denom

        return SpectralPlanePhysicsState(
            u_hat=state.u_hat.replace(data=_damp(state.u_hat.data)),
            v_hat=state.v_hat.replace(data=_damp(state.v_hat.data)),
            w_hat=state.w_hat.replace(data=_damp(state.w_hat.data)),
            theta_prime_hat=state.theta_prime_hat.replace(
                data=_damp(state.theta_prime_hat.data),
            ),
            rho_prime_hat=state.rho_prime_hat.replace(
                data=_damp(state.rho_prime_hat.data),
            ),
            phis=state.phis,
            tracers_hat=state.tracers_hat.replace(
                data=_damp(state.tracers_hat.data),
            ),
        )

    def _apply_dealias(
        self, state: SpectralPlanePhysicsState,
    ) -> SpectralPlanePhysicsState:
        """Zero spectral coefficients above the 2/3 cutoff."""
        axis = self._axis
        return SpectralPlanePhysicsState(
            u_hat=state.u_hat.replace(
                data=apply_dealias(state.u_hat.data, axis),
            ),
            v_hat=state.v_hat.replace(
                data=apply_dealias(state.v_hat.data, axis),
            ),
            w_hat=state.w_hat.replace(
                data=apply_dealias(state.w_hat.data, axis),
            ),
            theta_prime_hat=state.theta_prime_hat.replace(
                data=apply_dealias(state.theta_prime_hat.data, axis),
            ),
            rho_prime_hat=state.rho_prime_hat.replace(
                data=apply_dealias(state.rho_prime_hat.data, axis),
            ),
            phis=state.phis,
            tracers_hat=state.tracers_hat.replace(
                data=apply_dealias(state.tracers_hat.data, axis),
            ),
        )


# --------------------------------------------------------------------- #
# Convenience: build a spectral state from a physical state             #
# --------------------------------------------------------------------- #


def spec_state_from_physical(
    phys_state: PlaneNonHydrostaticState,
) -> SpectralPlanePhysicsState:
    """One-shot constructor: take a physical-space plane state and
    return its spectral counterpart via rfft2 on every prognostic."""
    u_hat = to_spec(phys_state.u.data)
    v_hat = to_spec(phys_state.v.data)
    w_hat = to_spec(phys_state.w.data)
    theta_p_hat = to_spec(phys_state.theta_prime.data)
    rho_p_hat = to_spec(phys_state.rho_prime.data)
    tracers_hat = to_spec(phys_state.tracers.data)
    return SpectralPlanePhysicsState(
        u_hat=Field(
            data=u_hat, name="u_hat",
            dims=("ky", "kx", "z"), units="m/s", staggering="spectral",
        ),
        v_hat=Field(
            data=v_hat, name="v_hat",
            dims=("ky", "kx", "z"), units="m/s", staggering="spectral",
        ),
        w_hat=Field(
            data=w_hat, name="w_hat",
            dims=("ky", "kx", "z_half"), units="m/s",
            staggering="spectral",
        ),
        theta_prime_hat=Field(
            data=theta_p_hat, name="theta_prime_hat",
            dims=("ky", "kx", "z"), units="K", staggering="spectral",
        ),
        rho_prime_hat=Field(
            data=rho_p_hat, name="rho_prime_hat",
            dims=("ky", "kx", "z"), units="kg/m^3",
            staggering="spectral",
        ),
        phis=phys_state.phis,
        tracers_hat=Field(
            data=tracers_hat, name="tracers_hat",
            dims=("ky", "kx", "z", "tracer"), units="kg/kg",
            staggering="spectral",
        ),
    )
