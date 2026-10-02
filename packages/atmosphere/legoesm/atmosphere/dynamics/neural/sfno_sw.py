"""SFNO-based Shallow Water Dynamical Core.

Provides a learned dynamical core for the rotating shallow water
equations on the sphere, using the Spherical Fourier Neural Operator
(SFNO). Operates natively on the Gaussian grid, reusing the existing
SHT infrastructure.

Two modes of operation:
- **state_update**: SFNO directly predicts the next state
- **hybrid_tendencies**: SFNO provides tendencies, integrated with SSP-RK3

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
from legoesm.grids.gaussian import GaussianGrid
from legoesm.atmosphere.dynamics.gcm.spectral_sw import (
    SpectralSWState,
)
from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.normalization import (
    NormalizationStats,
    normalize,
    denormalize,
)
from legoesm.ml.channel_packing import pack_sw_state, unpack_sw_output
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm.atmosphere.dynamics.neural.sfno_pe import check_state_update_dt


class SFNOShallowWaterConfig(NamedTuple):
    """Configuration for the SFNO shallow water model.

    Attributes
    ----------
    sfno_config : SFNOConfig
        SFNO architecture configuration.
    mode : str
        "state_update" or "hybrid_tendencies".
    dt_sfno : float
        Time step for SFNO predictions [s]. In state_update mode,
        this is the interval between SFNO calls, and ``step`` refuses any
        other ``dt``. Unused in hybrid mode (the integrator uses ``dt``).
    g : float
        Gravitational acceleration [m/s^2].
    use_normalization : bool
        Whether to apply Z-score normalization to SFNO inputs/outputs.
    """
    sfno_config: SFNOConfig = SFNOConfig(
        in_channels=4, out_channels=4, embed_dim=128, n_blocks=4
    )
    mode: str = "state_update"
    dt_sfno: float = 3600.0
    g: float = constants.g  # = 9.80616
    use_normalization: bool = False
    time_integrator: str = "ssp_rk3"


class SFNOShallowWaterModel(IntegrationMixin):
    """SFNO-based shallow water model on the sphere.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with precomputed SH transform matrices.
    config : SFNOShallowWaterConfig, optional
        Model configuration.
    sfno_model : SFNO, optional
        Pre-trained SFNO model. If None, a randomly initialized
        model is created.
    norm_stats : NormalizationStats, optional
        Normalization statistics for input/output Z-scoring.
    key : jax.random.PRNGKey, optional
        Random key for SFNO initialization (used if sfno_model is None).
    """

    def __init__(
        self,
        grid: GaussianGrid,
        config: SFNOShallowWaterConfig | None = None,
        sfno_model: SFNO | None = None,
        norm_stats: NormalizationStats | None = None,
        *,
        key: jax.Array | None = None,
    ):
        self.config = config or SFNOShallowWaterConfig()
        self.grid = grid
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
        state: SpectralSWState,
        dt: float,
    ) -> SpectralSWState:
        """Advance one time step.

        Parameters
        ----------
        state : SpectralSWState
            Current spectral shallow water state.
        dt : float
            Time step [s].

        Returns
        -------
        SpectralSWState
            Advanced state.
        """
        if self.config.mode == "state_update":
            check_state_update_dt(dt, self.config.dt_sfno)
            return self._step_state_update(state)
        elif self.config.mode == "hybrid_tendencies":
            return self._step_hybrid(state, dt)
        else:
            raise ValueError(
                f"Unknown mode: {self.config.mode!r}. "
                f"Choose 'state_update' or 'hybrid_tendencies'."
            )

    def _step_state_update(
        self,
        state: SpectralSWState,
    ) -> SpectralSWState:
        """Direct state update: SFNO(state_t) → state_{t+1}.

        The SFNO predicts the full next state directly (with residual
        prediction if configured).
        """
        # Pack state → (n_lat, n_lon, 4)
        x = pack_sw_state(state, self.grid).astype(jnp.float32)

        # Normalize
        if self.config.use_normalization and self.norm_stats is not None:
            x = normalize(x, self.norm_stats)

        # SFNO forward pass
        y = self.sfno(x, self.grid)

        # Denormalize
        if self.config.use_normalization and self.norm_stats is not None:
            y = denormalize(y, self.norm_stats)

        # Unpack → SpectralSWState
        return unpack_sw_output(y, state, self.grid, mode="state_update")

    def _step_hybrid(
        self,
        state: SpectralSWState,
        dt: float,
    ) -> SpectralSWState:
        """Hybrid mode: SFNO provides tendencies, integrated with SSP-RK3.

        The SFNO predicts tendencies (time derivatives) which are used
        as the RHS for SSP-RK3 time integration.
        """
        def tendency_fn(s):
            x = pack_sw_state(s, self.grid).astype(jnp.float32)
            if self.config.use_normalization and self.norm_stats is not None:
                x = normalize(x, self.norm_stats)
            y = self.sfno(x, self.grid)
            if self.config.use_normalization and self.norm_stats is not None:
                y = denormalize(y, self.norm_stats)
            return unpack_sw_output(y, s, self.grid, mode="tendencies")

        return dispatch_integrator(state, tendency_fn, dt, self.config.time_integrator)
