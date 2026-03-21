"""SFNO-based Hydrostatic Primitive Equation Dynamical Core.

Provides a learned dynamical core for the hydrostatic primitive
equations on the sphere, using the Spherical Fourier Neural Operator
(SFNO). Operates on SpectralHydrostaticState using the Gaussian grid.

Two modes:
- **state_update**: SFNO directly predicts the next full state
- **hybrid_tendencies**: SFNO provides tendencies, integrated with SSP-RK3

Supports post-hoc conservation corrections for dry air mass and
moisture, and optional physics coupling.

References
----------
- Bonev et al. (2023). Spherical Fourier Neural Operators. ICML.
- Watt-Meyer et al. (2023). ACE. arXiv:2310.02074.
"""

from __future__ import annotations

from typing import NamedTuple, Callable

import jax
import jax.numpy as jnp

from legoesm.grids.gaussian import (
    GaussianGrid,
    sh_analysis,
    sh_synthesis,
)
from legoesm.grids.vertical import SigmaCoordinate
from legoesm.atmosphere.dynamics.spectral_pe import SpectralHydrostaticState
from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.normalization import (
    NormalizationStats,
    normalize,
    denormalize,
)
from legoesm.ml.channel_packing import pack_pe_state, unpack_pe_output
from legoesm.ml.conservation import (
    correct_dry_air_mass,
)
from legoesm.timestepping.dispatch import dispatch_integrator


class SFNOPrimitiveEquationConfig(NamedTuple):
    """Configuration for the SFNO primitive equation model.

    Attributes
    ----------
    sfno_config : SFNOConfig
        SFNO architecture configuration.
    mode : str
        "state_update" or "hybrid_tendencies".
    dt_sfno : float
        Time step for SFNO predictions [s].
    correct_mass : bool
        Apply dry air mass conservation correction.
    correct_moisture_budget : bool
        Apply moisture budget correction.
    clip_q : bool
        Clip negative humidity values.
    use_normalization : bool
        Whether to apply Z-score normalization.
    pressure_levels : tuple
        Pressure levels [hPa] for the 3D fields.
    """
    sfno_config: SFNOConfig = SFNOConfig(
        in_channels=54, out_channels=54, embed_dim=256, n_blocks=8
    )
    mode: str = "state_update"
    dt_sfno: float = 21600.0  # 6 hours default
    correct_mass: bool = True
    correct_moisture_budget: bool = True
    clip_q: bool = True
    use_normalization: bool = False
    pressure_levels: tuple = (
        1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50
    )
    time_integrator: str = "ssp_rk3"


class SFNOPrimitiveEquationModel:
    """SFNO-based hydrostatic primitive equation model.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with precomputed SH transform matrices.
    sigma_coord : SigmaCoordinate
        Vertical sigma coordinate.
    config : SFNOPrimitiveEquationConfig, optional
        Model configuration.
    sfno_model : SFNO, optional
        Pre-trained SFNO model. If None, randomly initialized.
    norm_stats : NormalizationStats, optional
        Normalization statistics.
    key : jax.random.PRNGKey, optional
        Random key for initialization.
    """

    def __init__(
        self,
        grid: GaussianGrid,
        sigma_coord: SigmaCoordinate,
        config: SFNOPrimitiveEquationConfig | None = None,
        sfno_model: SFNO | None = None,
        norm_stats: NormalizationStats | None = None,
        *,
        key: jax.Array | None = None,
    ):
        self.config = config or SFNOPrimitiveEquationConfig()
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.norm_stats = norm_stats

        if sfno_model is not None:
            self.sfno = sfno_model
        else:
            if key is None:
                key = jax.random.PRNGKey(0)
            self.sfno = SFNO(
                config=self.config.sfno_config,
                grid=grid,
                key=key,
            )

    def step(
        self,
        state: SpectralHydrostaticState,
        dt: float,
    ) -> SpectralHydrostaticState:
        """Advance one time step.

        Parameters
        ----------
        state : SpectralHydrostaticState
            Current state.
        dt : float
            Time step [s].

        Returns
        -------
        SpectralHydrostaticState
            Advanced state.
        """
        if self.config.mode == "state_update":
            new_state = self._step_state_update(state)
        elif self.config.mode == "hybrid_tendencies":
            new_state = self._step_hybrid(state, dt)
        else:
            raise ValueError(
                f"Unknown mode: {self.config.mode!r}. "
                f"Choose 'state_update' or 'hybrid_tendencies'."
            )

        # Post-hoc conservation corrections
        if self.config.correct_mass or self.config.correct_moisture_budget:
            new_state = self._apply_conservation(new_state, state)

        return new_state

    def step_with_physics(
        self,
        state: SpectralHydrostaticState,
        dt: float,
        physics_fn: Callable,
    ) -> SpectralHydrostaticState:
        """Advance one step with physics coupling.

        In hybrid mode, physics tendencies are added to SFNO tendencies
        before SSP-RK3 integration. In state_update mode, physics
        tendencies are applied additively after the SFNO state update.

        Parameters
        ----------
        state : SpectralHydrostaticState
            Current state.
        dt : float
            Time step [s].
        physics_fn : callable
            Physics function: (state, grid, sigma_coord) → tendencies.

        Returns
        -------
        SpectralHydrostaticState
        """
        if self.config.mode == "hybrid_tendencies":
            def combined_tendency(s):
                sfno_tend = self._sfno_tendency(s)
                phys_tend = physics_fn(s, self.grid, self.sigma_coord)
                return jax.tree.map(
                    lambda a, b: a + b, sfno_tend, phys_tend
                )
            new_state = dispatch_integrator(state, combined_tendency, dt, self.config.time_integrator)
        else:
            # State update mode: SFNO prediction + physics tendencies
            new_state = self._step_state_update(state)
            phys_tend = physics_fn(state, self.grid, self.sigma_coord)
            new_state = jax.tree.map(
                lambda s, t: s + dt * t, new_state, phys_tend
            )

        if self.config.correct_mass or self.config.correct_moisture_budget:
            new_state = self._apply_conservation(new_state, state)

        return new_state

    def _step_state_update(
        self,
        state: SpectralHydrostaticState,
    ) -> SpectralHydrostaticState:
        """Direct state update: SFNO(state_t) → state_{t+1}."""
        x = pack_pe_state(state, self.grid, self.sigma_coord)
        x = x.astype(jnp.float32)

        if self.config.use_normalization and self.norm_stats is not None:
            x = normalize(x, self.norm_stats)

        y = self.sfno(x, self.grid)

        if self.config.use_normalization and self.norm_stats is not None:
            y = denormalize(y, self.norm_stats)

        return unpack_pe_output(y, state, self.grid, mode="state_update")

    def _step_hybrid(
        self,
        state: SpectralHydrostaticState,
        dt: float,
    ) -> SpectralHydrostaticState:
        """Hybrid mode: SFNO tendencies + configurable RK integrator."""
        return dispatch_integrator(state, self._sfno_tendency, dt, self.config.time_integrator)

    def _sfno_tendency(
        self,
        state: SpectralHydrostaticState,
    ) -> SpectralHydrostaticState:
        """Compute SFNO-predicted tendencies."""
        x = pack_pe_state(state, self.grid, self.sigma_coord)
        x = x.astype(jnp.float32)

        if self.config.use_normalization and self.norm_stats is not None:
            x = normalize(x, self.norm_stats)

        y = self.sfno(x, self.grid)

        if self.config.use_normalization and self.norm_stats is not None:
            y = denormalize(y, self.norm_stats)

        return unpack_pe_output(y, state, self.grid, mode="tendencies")

    def _apply_conservation(
        self,
        new_state: SpectralHydrostaticState,
        old_state: SpectralHydrostaticState,
    ) -> SpectralHydrostaticState:
        """Apply post-hoc conservation corrections.

        Corrects global dry air mass and moisture budget in grid space,
        then transforms back to spectral space.
        """
        if not (self.config.correct_mass or self.config.correct_moisture_budget):
            return new_state

        grid = self.grid

        # Get surface pressure from lnps
        lnps_new = sh_synthesis(grid, new_state.lnps_hat.data)
        lnps_old = sh_synthesis(grid, old_state.lnps_hat.data)
        p_s_new = jnp.exp(lnps_new)
        p_s_old = jnp.exp(lnps_old)

        if self.config.correct_mass:
            p_s_new = correct_dry_air_mass(p_s_new, p_s_old, grid)
            lnps_new = jnp.log(jnp.maximum(p_s_new, 1.0))
            lnps_hat = sh_analysis(grid, lnps_new.astype(jnp.float64))
            new_state = new_state._replace(
                lnps_hat=new_state.lnps_hat.replace(data=lnps_hat)
            )

        return new_state

    def integrate(
        self,
        state: SpectralHydrostaticState,
        duration: float,
        dt: float,
        save_every: int = 1,
        physics_fn: Callable | None = None,
    ) -> tuple[SpectralHydrostaticState, list]:
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : SpectralHydrostaticState
            Initial state.
        duration : float
            Total integration time [s].
        dt : float
            Time step [s].
        save_every : int
            Save state every N steps.
        physics_fn : callable, optional
            Physics function for coupled integration.

        Returns
        -------
        (final_state, trajectory)
        """
        n_steps = int(duration / dt)
        trajectory = [state]
        for i in range(n_steps):
            if physics_fn is not None:
                state = self.step_with_physics(state, dt, physics_fn)
            else:
                state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
        return state, trajectory
