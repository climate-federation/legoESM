"""U-Cast hydrostatic primitive-equation emulator (convolutional U-Net).

Learned, *pure-ML* dynamical core for the hydrostatic primitive equations,
using the U-Cast convolutional U-Net (:class:`legoesm.ml.ucast.UCast`) as a
drop-in alternative to the spectral :class:`~legoesm.atmosphere.dynamics.neural.sfno_pe.SFNOPrimitiveEquationModel`.

It shares the SFNO emulator's plumbing verbatim — channel packing
(:mod:`legoesm.ml.channel_packing`), Z-score normalisation
(:mod:`legoesm.ml.normalization`), post-hoc dry-air-mass conservation
(:mod:`legoesm.ml.conservation`) and the SSP-RK time integrator dispatch —
so the only thing that changes between the two emulators is the learned
network.  This is the "modular" contract: the dynamical-core wrapper is
network-agnostic; SFNO and U-Cast differ only in the ``ml`` model they hold.

Two deterministic modes (identical to the SFNO wrapper):
- **state_update**: U-Cast directly predicts the next full state.
- **hybrid_tendencies**: U-Cast provides tendencies, integrated with SSP-RK3.

Plus the **probabilistic** capability that defines U-Cast:
- :meth:`UCastPrimitiveEquationModel.ensemble_step` runs ``n_members``
  MC-Dropout forward passes (distinct dropout keys, batched with
  ``jax.vmap``) and returns the ensemble of next states.  Pair with the
  fair-CRPS scores in :mod:`legoesm.ml.loss` (``almost_fair_crps`` /
  ``area_weighted_afcrps``) for probabilistic training/evaluation.

References
----------
- U-Cast (Rose-STL-Lab, ICML 2026): https://github.com/Rose-STL-Lab/u-cast
- Watt-Meyer et al. (2023). ACE. arXiv:2310.02074 (conservation correctors).
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
from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralHydrostaticState
from legoesm.ml.ucast import UCast, UCastConfig
from legoesm.ml.normalization import (
    NormalizationStats,
    normalize,
    denormalize,
)
from legoesm.ml.channel_packing import (
    pack_pe_state,
    unpack_pe_output,
    PE3DChannelSpec,
)
from legoesm.ml.conservation import (
    correct_dry_air_mass,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import (
    refuse_unthreaded_stateful_physics,
)


class UCastPrimitiveEquationConfig(NamedTuple):
    """Configuration for the U-Cast primitive-equation emulator.

    Attributes
    ----------
    ucast_config : UCastConfig
        U-Cast U-Net architecture configuration.  ``in_channels`` /
        ``out_channels`` must equal the PE channel count (``4*nlev + 2``).
    mode : str
        "state_update" or "hybrid_tendencies".
    dt_ucast : float
        Time step the network was trained for [s] (6 h default).
    correct_mass : bool
        Apply post-hoc dry-air-mass conservation correction.
    correct_moisture_budget : bool
        Moisture-budget correction.  NOT YET WIRED — like the SFNO wrapper,
        ``correct_moisture`` exists in :mod:`legoesm.ml.conservation` but is
        not called here (moisture is a spectral tracer needing synthesis,
        correction and re-analysis).  Defaults to ``False`` so the config does
        not advertise a correction it does not perform.
    clip_q : bool
        Clip negative humidity.  NOT YET WIRED (see above); defaults ``False``.
    use_normalization : bool
        Apply Z-score normalisation around the network.
    time_integrator : str
        Integrator name for ``hybrid_tendencies`` mode.

    Note: channels are packed on the MODEL SIGMA LEVELS directly (see
    ``legoesm.ml.channel_packing.pack_pe_state``); no sigma→pressure
    interpolation is performed.  A former ``pressure_levels`` field
    advertised WeatherBench2 levels that were never used — removed.
    """
    ucast_config: UCastConfig = UCastConfig(
        in_channels=54, out_channels=54, model_channels=128,
        channel_mult=(1, 2, 3, 4), num_blocks=2, attn_levels=(2, 3),
    )
    mode: str = "state_update"
    dt_ucast: float = 21600.0  # 6 hours default
    correct_mass: bool = True
    correct_moisture_budget: bool = False  # not yet wired in _apply_conservation
    clip_q: bool = False  # not yet wired in _apply_conservation
    use_normalization: bool = False
    time_integrator: str = "ssp_rk3"


class UCastPrimitiveEquationModel:
    """U-Cast-based hydrostatic primitive-equation emulator.

    Parameters
    ----------
    grid : GaussianGrid
        Grid with precomputed SH transform matrices (used by channel packing;
        the U-Net itself is grid-agnostic and applies no SH transforms).
    sigma_coord : SigmaCoordinate
        Vertical sigma coordinate.
    config : UCastPrimitiveEquationConfig, optional
        Model configuration.
    ucast_model : UCast, optional
        Pre-trained U-Cast network.  If None, randomly initialised.
    norm_stats : NormalizationStats, optional
        Normalisation statistics.
    key : jax.random.PRNGKey, optional
        Random key for initialisation.
    """

    def __init__(
        self,
        grid: GaussianGrid,
        sigma_coord: SigmaCoordinate,
        config: UCastPrimitiveEquationConfig | None = None,
        ucast_model: UCast | None = None,
        norm_stats: NormalizationStats | None = None,
        *,
        key: jax.Array | None = None,
    ):
        self.config = config or UCastPrimitiveEquationConfig()
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.norm_stats = norm_stats

        # The network channels must match the PE packing (4*nlev + 2): the
        # input is ``pack_pe_state`` and the output feeds ``unpack_pe_output``,
        # which reads fixed channel slices (u/v/T/q + lnps at 4*nlev).  A
        # too-small out_channels would let those slice indices be silently
        # clamped by JAX gather semantics and corrupt the decoded state.
        _n_expected = PE3DChannelSpec(nlev=sigma_coord.n_levels).n_channels
        _uc = self.config.ucast_config
        if _uc.in_channels != _n_expected or _uc.out_channels != _n_expected:
            raise ValueError(
                f"ucast_config in/out channels ({_uc.in_channels}/"
                f"{_uc.out_channels}) must equal the PE channel count "
                f"4*nlev+2 = {_n_expected} for nlev={sigma_coord.n_levels}."
            )

        # Fail fast on config flags this bridge does not yet honour, rather
        # than silently no-op'ing them (CLAUDE.md: no advertising what we do
        # not do).  ``correct_moisture_budget`` / ``clip_q`` need spectral-tracer
        # synthesis/re-analysis (see ``_apply_conservation``); the PE bridge
        # supplies no static/forcing conditioning channels yet.
        if self.config.correct_moisture_budget or self.config.clip_q:
            raise NotImplementedError(
                "correct_moisture_budget / clip_q are not yet wired in the "
                "U-Cast PE bridge (moisture is a spectral tracer needing "
                "synthesis, correction and re-analysis). Leave both False."
            )
        if self.config.ucast_config.num_conditional_channels != 0:
            raise NotImplementedError(
                "The U-Cast PE bridge does not yet supply conditioning fields; "
                "set ucast_config.num_conditional_channels=0 (the network "
                "supports conditions, but pack_pe_state emits none)."
            )
        # In hybrid mode the network output is read as a *tendency*.  With
        # ``residual_prediction=True`` the network instead emits
        # ``packed_state + delta`` (the right contract for state_update, where
        # the next state ≈ current + change).  Feeding that as a tendency would
        # add the entire state magnitude as d/dt — physically nonsensical — so
        # require a non-residual network for hybrid tendencies.
        if (self.config.mode == "hybrid_tendencies"
                and self.config.ucast_config.residual_prediction):
            raise ValueError(
                "mode='hybrid_tendencies' requires "
                "ucast_config.residual_prediction=False: the network must emit "
                "raw tendencies, not state+delta. Use residual_prediction=True "
                "only with mode='state_update'."
            )
        # Z-score denormalisation (``y*std + mean``) inverts a *state*
        # normalisation, so it is only meaningful when the network output is a
        # state.  In tendency mode it would add the state mean to a tendency
        # (a zero output would denormalise to ``mean`` as a physical d/dt),
        # which needs *separate* tendency-output statistics that are not yet
        # wired.  Reject the combination rather than silently corrupt the
        # tendency.  (``use_normalization`` defaults False.)
        if self.config.mode == "hybrid_tendencies" and self.config.use_normalization:
            raise NotImplementedError(
                "use_normalization=True is not supported with "
                "mode='hybrid_tendencies': denormalising the network output "
                "with state-level NormalizationStats would add the state mean "
                "to a tendency. Use mode='state_update', or supply dedicated "
                "tendency-output stats (not yet wired)."
            )
        # ``use_normalization`` without stats would silently skip (de)normalisation
        # (see ``_forward``), so a normalised checkpoint could run on raw PE
        # channels and emit wrongly-scaled states.  Require the stats up front.
        if self.config.use_normalization and self.norm_stats is None:
            raise ValueError(
                "use_normalization=True requires norm_stats; got None. Pass "
                "NormalizationStats or set use_normalization=False."
            )

        if ucast_model is not None:
            # The config-derived guards above (residual/channels/conditioning)
            # only protect the model if the supplied network actually matches
            # ``config.ucast_config``.  Validate it so e.g. a residual network
            # cannot be slipped under a non-residual hybrid config (which would
            # silently feed state+delta as a tendency).
            if ucast_model.config != self.config.ucast_config:
                raise ValueError(
                    "Supplied ucast_model.config does not match "
                    "config.ucast_config; the architecture/contract guards "
                    "(residual_prediction, channels, conditioning) would not "
                    f"apply to the actual network.\n  model:  {ucast_model.config}\n"
                    f"  config: {self.config.ucast_config}"
                )
            self.ucast = ucast_model
        else:
            if key is None:
                key = jax.random.PRNGKey(0)
            self.ucast = UCast(config=self.config.ucast_config, key=key)

    # ------------------------------------------------------------------
    # Deterministic stepping (mirrors the SFNO wrapper exactly)
    # ------------------------------------------------------------------

    def step(
        self,
        state: SpectralHydrostaticState,
        dt: float,
    ) -> SpectralHydrostaticState:
        """Advance one deterministic time step (MC-Dropout disabled).

        Timestep semantics (identical to the SFNO emulator):
        - In ``state_update`` mode the network *is* the time-stepping operator;
          it emits the next state for its **trained** macro-step
          (``config.dt_ucast``, 6 h by default).  ``dt`` is therefore **not**
          used in this mode — calling ``step(state, dt=3600)`` does not produce
          a 1-hour step, it produces one trained macro-step.  Drive multi-step
          rollouts by repeated calls, not by varying ``dt``.
        - In ``hybrid_tendencies`` mode the network emits tendencies and ``dt``
          is the genuine RK integration step.
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

        if self.config.correct_mass or self.config.correct_moisture_budget:
            new_state = self._apply_conservation(new_state, state)

        return new_state

    def step_with_physics(
        self,
        state: SpectralHydrostaticState,
        dt: float,
        physics_fn: Callable,
    ) -> SpectralHydrostaticState:
        """Advance one deterministic step with physics coupling.

        In hybrid mode, physics tendencies are added to U-Cast tendencies
        before RK integration.  In state_update mode, physics tendencies are
        applied additively after the U-Cast state update.
        """
        # U-Cast PE has no PhysicsState carry slot — refuse a stateful
        # ``make_physics`` fn (tagged _requires_phys_state) rather than
        # silently reseed its prognostic fields every step (#405/#413).
        refuse_unthreaded_stateful_physics(
            physics_fn, None, where="U-Cast PE step_with_physics()")
        if self.config.mode == "hybrid_tendencies":
            def combined_tendency(s):
                ucast_tend = self._ucast_tendency(s)
                _phys_result = physics_fn(s, self.grid, self.sigma_coord)
                phys_tend = _phys_result[0] if type(_phys_result) is tuple else _phys_result
                return jax.tree.map(
                    lambda a, b: a + b, ucast_tend, phys_tend
                )
            new_state = dispatch_integrator(
                state, combined_tendency, dt, self.config.time_integrator,
            )
        elif self.config.mode == "state_update":
            new_state = self._step_state_update(state)
            _phys_result = physics_fn(state, self.grid, self.sigma_coord)
            phys_tend = _phys_result[0] if type(_phys_result) is tuple else _phys_result
            new_state = jax.tree.map(
                lambda s, t: s + dt * t, new_state, phys_tend
            )
        else:
            raise ValueError(
                f"Unknown mode: {self.config.mode!r}. "
                f"Choose 'state_update' or 'hybrid_tendencies'."
            )

        if self.config.correct_mass or self.config.correct_moisture_budget:
            new_state = self._apply_conservation(new_state, state)

        return new_state

    # ------------------------------------------------------------------
    # Probabilistic stepping (MC-Dropout ensemble — the U-Cast headline)
    # ------------------------------------------------------------------

    def ensemble_step(
        self,
        state: SpectralHydrostaticState,
        key: jax.Array,
        n_members: int,
    ) -> SpectralHydrostaticState:
        """Advance one step as an MC-Dropout ensemble.

        Runs ``n_members`` forward passes with **active** dropout (distinct
        per-member keys), batched with ``jax.vmap``.  Returns a
        :class:`SpectralHydrostaticState` whose leaves carry a leading
        ``n_members`` axis.  Only ``state_update`` mode is supported for
        ensembling (the probabilistic forecast the paper trains with CRPS).

        Parameters
        ----------
        state : SpectralHydrostaticState
            Current state.
        key : jax.random.PRNGKey
            Base random key; split into ``n_members`` member keys.
        n_members : int
            Ensemble size.

        Returns
        -------
        SpectralHydrostaticState
            Ensemble of next states (leaves have shape ``(n_members, ...)``).
        """
        if self.config.mode != "state_update":
            raise ValueError(
                f"ensemble_step requires mode='state_update', got "
                f"{self.config.mode!r}."
            )
        member_keys = jax.random.split(key, n_members)

        def one_member(member_key):
            new_state = self._step_state_update(state, key=member_key)
            if self.config.correct_mass or self.config.correct_moisture_budget:
                new_state = self._apply_conservation(new_state, state)
            return new_state

        return jax.vmap(one_member)(member_keys)

    # ------------------------------------------------------------------
    # Internal forward helpers
    # ------------------------------------------------------------------

    def _step_state_update(
        self,
        state: SpectralHydrostaticState,
        key: jax.Array | None = None,
    ) -> SpectralHydrostaticState:
        """Direct state update: ``UCast(state_t) -> state_{t+1}``.

        ``key`` is forwarded to the network's dropout: ``None`` → deterministic
        (inference dropout); a key → an MC-Dropout sample.
        """
        y = self._forward(state, key=key)
        return unpack_pe_output(y, state, self.grid, mode="state_update")

    def _step_hybrid(
        self,
        state: SpectralHydrostaticState,
        dt: float,
    ) -> SpectralHydrostaticState:
        """Hybrid mode: U-Cast tendencies + configurable RK integrator."""
        return dispatch_integrator(
            state, self._ucast_tendency, dt, self.config.time_integrator,
        )

    def _ucast_tendency(
        self,
        state: SpectralHydrostaticState,
    ) -> SpectralHydrostaticState:
        """Compute U-Cast-predicted tendencies (deterministic)."""
        y = self._forward(state, key=None)
        return unpack_pe_output(y, state, self.grid, mode="tendencies")

    def _forward(
        self,
        state: SpectralHydrostaticState,
        key: jax.Array | None,
    ) -> jnp.ndarray:
        """Pack → (normalise) → U-Cast → (denormalise).  Returns grid tensor."""
        x = pack_pe_state(state, self.grid, self.sigma_coord)
        x = x.astype(jnp.float32)

        # ``norm_stats`` is guaranteed present when use_normalization=True
        # (validated in __init__), so this never silently skips.
        if self.config.use_normalization:
            x = normalize(x, self.norm_stats)

        y = self.ucast(x, key=key)

        if self.config.use_normalization:
            y = denormalize(y, self.norm_stats)
        return y

    def _apply_conservation(
        self,
        new_state: SpectralHydrostaticState,
        old_state: SpectralHydrostaticState,
    ) -> SpectralHydrostaticState:
        """Apply post-hoc conservation corrections (dry air mass only).

        Identical to the SFNO wrapper: corrects global dry air mass in grid
        space then transforms back to spectral.  Moisture-budget correction
        and humidity clipping are not yet wired (the corresponding config
        flags default to ``False`` so the configuration is truthful).
        """
        if not (self.config.correct_mass or self.config.correct_moisture_budget):
            return new_state

        grid = self.grid

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
        """Integrate forward deterministically for a given duration."""
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
