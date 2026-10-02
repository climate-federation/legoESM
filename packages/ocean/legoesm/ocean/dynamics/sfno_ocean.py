"""SFNO-based Ocean Dynamical Core.

Provides a learned dynamical core for the Boussinesq hydrostatic
ocean primitive equations on the sphere, using the Spherical Fourier
Neural Operator (SFNO). Operates natively on the Gaussian grid,
with SpectralOceanState as the prognostic state.

Two modes of operation:
- **state_update**: SFNO directly predicts the next state
- **hybrid_tendencies**: SFNO provides tendencies, integrated with SSP-RK3

Post-hoc conservation corrections are applied for:
- Global ocean volume (eta correction)
- Global heat content (uniform T correction)
- Global salt content (uniform S correction)

References
----------
- Bonev et al. (2023). Spherical Fourier Neural Operators. ICML.
- Watt-Meyer et al. (2023). ACE. arXiv:2310.02074.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.gaussian import (
    GaussianGrid,
    sh_analysis,
    sh_analysis_3d,
    sh_synthesis,
    sh_synthesis_3d,
)
from legoesm.ocean.state import SpectralOceanState
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
)
from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.normalization import (
    NormalizationStats,
    normalize,
    denormalize,
)
from legoesm.ocean.dynamics.channel_packing import (
    pack_ocean_state,
    unpack_ocean_output,
)
from legoesm.ml.conservation import (
    correct_ocean_volume,
    correct_ocean_tracer,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin


class SFNOOceanConfig(NamedTuple):
    """Configuration for the SFNO ocean model.

    Attributes
    ----------
    sfno_config : SFNOConfig
        SFNO architecture configuration.
    mode : str
        "state_update" or "hybrid_tendencies".
    dt_sfno : float
        Time step for SFNO predictions [s].
    g : float
        Gravitational acceleration [m/s^2].
    rho_0 : float
        Reference ocean density [kg/m^3].
    correct_volume : bool
        Apply global ocean volume conservation correction.
    correct_heat : bool
        Apply global heat content conservation correction.
    correct_salt : bool
        Apply global salt content conservation correction.
    use_normalization : bool
        Whether to apply Z-score normalization to SFNO inputs/outputs.
    min_water_column_m : float
        Minimum water column thickness [m] for stability.
    """
    sfno_config: SFNOConfig = SFNOConfig(
        in_channels=42, out_channels=42, embed_dim=256, n_blocks=8
    )
    mode: str = "state_update"
    dt_sfno: float = 3600.0
    g: float = constants.g
    rho_0: float = constants.rho_ocean
    correct_volume: bool = True
    correct_heat: bool = True
    correct_salt: bool = True
    use_normalization: bool = False
    min_water_column_m: float = 0.5
    time_integrator: str = "ssp_rk3"


class SFNOOceanModel(IntegrationMixin):
    """SFNO-based ocean model on the sphere.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with precomputed SH transform matrices.
    z_coord : OceanZStarCoordinate
        Ocean vertical coordinate.
    config : SFNOOceanConfig, optional
        Model configuration.
    sfno_model : SFNO, optional
        Pre-trained SFNO model. If None, randomly initialized.
    norm_stats : NormalizationStats, optional
        Normalization statistics for Z-scoring.
    key : jax.random.PRNGKey, optional
        Random key for initialization.
    """

    def __init__(
        self,
        grid: GaussianGrid,
        z_coord: OceanZStarCoordinate,
        config: SFNOOceanConfig | None = None,
        sfno_model: SFNO | None = None,
        norm_stats: NormalizationStats | None = None,
        *,
        key: jax.Array | None = None,
    ):
        self.config = config or SFNOOceanConfig()
        self.grid = grid
        self.z_coord = z_coord
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
        state: SpectralOceanState,
        dt: float,
    ) -> SpectralOceanState:
        """Advance one time step.

        Parameters
        ----------
        state : SpectralOceanState
            Current spectral ocean state.
        dt : float
            Time step [s].

        Returns
        -------
        SpectralOceanState
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
        if (self.config.correct_volume or self.config.correct_heat
                or self.config.correct_salt):
            new_state = self._apply_conservation(new_state, state)

        return new_state

    def _step_state_update(
        self,
        state: SpectralOceanState,
    ) -> SpectralOceanState:
        """Direct state update: SFNO(state_t) → state_{t+1}."""
        x = pack_ocean_state(state, self.grid).astype(jnp.float32)

        if self.config.use_normalization and self.norm_stats is not None:
            x = normalize(x, self.norm_stats)

        y = self.sfno(x, self.grid)

        if self.config.use_normalization and self.norm_stats is not None:
            y = denormalize(y, self.norm_stats)

        return unpack_ocean_output(y, state, self.grid, mode="state_update")

    def _step_hybrid(
        self,
        state: SpectralOceanState,
        dt: float,
    ) -> SpectralOceanState:
        """Hybrid mode: SFNO tendencies + configurable RK integrator."""
        return dispatch_integrator(state, self._sfno_tendency, dt, self.config.time_integrator)

    def _sfno_tendency(
        self,
        state: SpectralOceanState,
    ) -> SpectralOceanState:
        """Compute SFNO-predicted tendencies."""
        x = pack_ocean_state(state, self.grid).astype(jnp.float32)

        if self.config.use_normalization and self.norm_stats is not None:
            x = normalize(x, self.norm_stats)

        y = self.sfno(x, self.grid)

        if self.config.use_normalization and self.norm_stats is not None:
            y = denormalize(y, self.norm_stats)

        return unpack_ocean_output(y, state, self.grid, mode="tendencies")

    def _apply_conservation(
        self,
        new_state: SpectralOceanState,
        old_state: SpectralOceanState,
    ) -> SpectralOceanState:
        """Apply post-hoc conservation corrections.

        Corrects global ocean volume, heat, and salt content in grid
        space, then transforms back to spectral space.
        """
        grid = self.grid
        z_coord = self.z_coord
        mask = old_state.land_mask_grid.data

        # Get eta in grid space
        eta_old = sh_synthesis(grid, old_state.eta_hat.data).real * mask
        eta_new = sh_synthesis(grid, new_state.eta_hat.data).real * mask

        # Get H_bathy for layer thickness computation
        H_bathy = sh_synthesis(grid, old_state.H_bathy_hat.data).real
        H_bathy = jnp.maximum(H_bathy, 1.0) * mask + 1.0 * (1.0 - mask)

        if self.config.correct_volume:
            eta_new = correct_ocean_volume(eta_new, eta_old, grid, mask)
            eta_hat = sh_analysis(grid, eta_new.astype(jnp.float64))
            new_state = new_state._replace(
                eta_hat=new_state.eta_hat.replace(data=eta_hat)
            )

        if self.config.correct_heat or self.config.correct_salt:
            # Compute layer thicknesses for volume-weighted corrections
            min_col = self.config.min_water_column_m
            h_k_old = compute_layer_thickness(
                eta_old, H_bathy, z_coord, min_water_column_m=min_col,
            )
            h_k_new = compute_layer_thickness(
                eta_new, H_bathy, z_coord, min_water_column_m=min_col,
            )

            T_old = sh_synthesis_3d(grid, old_state.T_hat.data).real
            T_new = sh_synthesis_3d(grid, new_state.T_hat.data).real
            S_old = sh_synthesis_3d(grid, old_state.S_hat.data).real
            S_new = sh_synthesis_3d(grid, new_state.S_hat.data).real

            if self.config.correct_heat:
                T_new = correct_ocean_tracer(
                    T_new, T_old, h_k_new, h_k_old, grid, mask,
                )
                T_hat = sh_analysis_3d(grid, T_new.astype(jnp.float64))
                new_state = new_state._replace(
                    T_hat=new_state.T_hat.replace(data=T_hat)
                )

            if self.config.correct_salt:
                S_new = correct_ocean_tracer(
                    S_new, S_old, h_k_new, h_k_old, grid, mask,
                )
                S_hat = sh_analysis_3d(grid, S_new.astype(jnp.float64))
                new_state = new_state._replace(
                    S_hat=new_state.S_hat.replace(data=S_hat)
                )

        return new_state
