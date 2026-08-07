"""SFNO-based Hydrostatic Primitive Equation Dynamical Core.

Provides a learned dynamical core for the hydrostatic primitive
equations on the sphere, using the Spherical Fourier Neural Operator
(SFNO). Operates on SpectralHydrostaticState using the Gaussian grid.

Two modes:
- **state_update**: SFNO directly predicts the next full state
- **hybrid_tendencies**: the SFNO's next-state prediction (a residual
  net: ``output = input + delta``) is converted to a finite-difference
  tendency ``(output - input) / dt_sfno`` and integrated with the
  configured RK scheme (requires ``residual_prediction=True``)

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
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralHydrostaticState,
    apply_spectral_filter_to_state,
    compute_spectral_filter,
)
from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.normalization import (
    NormalizationStats,
    normalize,
    denormalize,
)
from legoesm.ml.channel_packing import pack_pe_state, unpack_pe_output
from legoesm.ml.conservation import (
    clip_humidity,
    correct_dry_air_mass,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import (
    refuse_unthreaded_stateful_physics,
)


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
        Apply moisture budget correction.  NOT YET WIRED — ``correct_moisture``
        exists in ``legoesm.ml.conservation`` but ``_apply_conservation`` does
        not call it (moisture is a spectral tracer needing synthesis,
        correction and re-analysis).  Defaults to ``False`` so the config does
        not advertise a correction it does not perform.
    clip_q : bool
        Clamp negative specific humidity to zero after each state update
        (``legoesm.ml.conservation.clip_humidity``), mirroring ACE2's
        corrector step 1 ("moisture, precipitation rate and radiative fluxes
        are all made to be positive by setting any negative values to zero",
        arXiv:2411.11268 §4.3). Acts on the grid-space ``q_v`` tracer, before
        the loss, so the optimizer never sees a state reachable only through
        negative moisture. No-op when the state carries no ``q_v``.

        NOT moisture-conserving, and deliberately so: ``max(q, 0)`` ADDS water
        wherever the prediction was negative, and nothing downstream removes it
        (``correct_moisture_budget`` is unimplemented, so ACE2's compensating
        global precipitation rescale — its corrector steps 3-4 — is absent).
        It also has zero derivative in the clamped cells, so the loss provides
        no direct gradient pushing them back positive; the pressure to avoid
        negative q comes only from the surrounding unclamped cells. Accepted
        because unbounded negative q corrupts virtual temperature and hence the
        whole column, which is strictly worse than a small moisture source.
    spectral_filter_strength : float
        Exponential-filter value at the truncation limit ``n_max``, applied to
        the prognostic spectral fields once per macro step
        (``spectral_pe.compute_spectral_filter`` +
        ``apply_spectral_filter_to_state`` — the SAME pair the dycore rollout
        uses every step). ``0.0`` disables it.

        WHY THIS EXISTS: in ``state_update`` mode the network emits grid-space
        ``u, v`` and ``unpack_pe_output`` rebuilds vorticity/divergence from
        them by spectral differentiation, which multiplies each coefficient by
        ~n. With no damping anywhere in ``step()`` the small-scale end of that
        cascade grows without bound. Measured on a trained T63 checkpoint
        (2026-07-28, ``scripts/tmp/diag_sfno_full_rollout.py``): |u850|max
        33 -> 158 m/s by macro step 3 (18 h) -> 969 m/s at step 4 -> 1.3e4 at
        step 7, |vor|max rising ~x2.5/step from step 3, with T, q and z500
        leaving physical range only at steps 6-7 and global-mean p_s pinned
        exactly by the mass corrector until step 9 — i.e. the wind cascade is
        the source and everything else is downstream of it. First NaN at step
        12 (72 h). The dycore path never showed this because
        ``spectral_rollout`` filters every step.
    spectral_filter_order : int
        Order of that exponential filter (higher = sharper cutoff, less
        damping of resolved scales).
    use_normalization : bool
        Whether to apply Z-score normalization.

    Note: channels are packed on the MODEL SIGMA LEVELS directly (see
    ``legoesm.ml.channel_packing.pack_pe_state``); no sigma→pressure
    interpolation is performed.  A former ``pressure_levels`` field
    advertised WeatherBench2 levels that were never used — removed.
    """
    sfno_config: SFNOConfig = SFNOConfig(
        in_channels=54, out_channels=54, embed_dim=256, n_blocks=8
    )
    mode: str = "state_update"
    dt_sfno: float = 21600.0  # 6 hours default
    correct_mass: bool = True
    correct_moisture_budget: bool = False  # not yet wired in _apply_conservation
    clip_q: bool = False
    use_normalization: bool = False
    time_integrator: str = "ssp_rk3"
    # 0.0 = legacy (no damping). Callers that roll autoregressively MUST set
    # this; see the class docstring for the measured divergence it prevents.
    spectral_filter_strength: float = 0.0
    spectral_filter_order: int = 8


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

        # Precompute the post-step exponential filter (pure function of the
        # static grid) so ``step`` is a cheap pointwise multiply rather than a
        # per-step filter build. Once per WRAPPER construction — training
        # rebuilds the wrapper inside the JIT trace, so this is once per trace,
        # not once per scan iteration. ``None`` = disabled.
        if not 0.0 <= self.config.spectral_filter_strength < 1.0:
            raise ValueError(
                "spectral_filter_strength is the filter VALUE at n_max and "
                "must lie in [0, 1); 0 disables the filter. Got "
                f"{self.config.spectral_filter_strength!r}."
            )
        if self.config.spectral_filter_order < 1:
            raise ValueError(
                "spectral_filter_order is the exponent in exp(-a (n/n_max)^p) "
                f"and must be >= 1; got {self.config.spectral_filter_order!r}."
            )
        self._spectral_filter = (
            compute_spectral_filter(
                grid.ls, grid.n_max,
                order=self.config.spectral_filter_order,
                cutoff_fraction=self.config.spectral_filter_strength,
            )
            if self.config.spectral_filter_strength > 0.0
            else None
        )

        # Z-score denormalisation (``y*std + mean``) inverts a *state*
        # normalisation, so it is only meaningful when the network output is a
        # state.  In hybrid_tendencies mode the output is read as a tendency;
        # denormalising it with state-level stats would add the state mean to a
        # d/dt (a zero output would map to ``mean``).  That needs separate
        # tendency-output statistics which are not wired here, so reject the
        # combination rather than silently corrupt the tendency.
        if self.config.mode == "hybrid_tendencies" and self.config.use_normalization:
            raise NotImplementedError(
                "use_normalization=True is not supported with "
                "mode='hybrid_tendencies': denormalising the network output "
                "with state-level NormalizationStats would add the state mean "
                "to a tendency. Use mode='state_update', or supply dedicated "
                "tendency-output stats (not yet wired)."
            )

        # hybrid_tendencies reads the SFNO as a next-state predictor and
        # forms the finite-difference tendency (output - input)/dt_sfno
        # (see ``_sfno_tendency``).  That is only well-defined for a
        # residual net (``output = input + delta`` by construction).  With
        # ``residual_prediction=False`` the raw decoder output has no
        # defined scale/semantics (state vs tendency depends entirely on
        # the training loss), so refuse loudly rather than integrate
        # garbage.
        if (self.config.mode == "hybrid_tendencies"
                and not self.config.sfno_config.residual_prediction):
            raise ValueError(
                "mode='hybrid_tendencies' requires "
                "sfno_config.residual_prediction=True: the hybrid path "
                "converts the residual net's next-state prediction to a "
                "tendency via (output - input)/dt_sfno. With "
                "residual_prediction=False the raw decoder output has no "
                "defined scale. Use mode='state_update' for non-residual "
                "state-predicting networks."
            )

        # Mirror the U-Cast PE bridge guard: refuse loudly rather than silently
        # ignore a requested correction that is not wired.
        #
        # ``clip_q`` IS wired now (see ``step``). The old guard also refused it
        # on the grounds that "moisture is a spectral tracer needing synthesis,
        # correction and re-analysis" — that reason does not hold for the
        # clamp: ``SpectralHydrostaticState.tracers`` are GRID-space fields
        # (n_lat, n_lon, nlev), so ``clip_humidity`` is a pointwise maximum with
        # no SH round-trip and therefore no Gibbs ringing. It does still hold
        # for ``correct_moisture_budget``, which needs a column-integrated
        # E - P budget this state carries no precipitation/evaporation channel
        # for; that one stays refused.
        if self.config.correct_moisture_budget:
            raise NotImplementedError(
                "correct_moisture_budget is not yet wired in the SFNO PE "
                "bridge: it needs the column-integrated E - P budget "
                "(ACE2 arXiv:2411.11268 eq. 2), and SpectralHydrostaticState "
                "carries no precipitation or evaporation channel. Leave it "
                "False; correct_mass, clip_q and the post-step spectral "
                "filter are applied."
            )
        # ``use_normalization`` without stats would silently skip (de)normalisation
        # (see ``_step_state_update`` / ``_sfno_tendency``), so a normalised
        # checkpoint could run on raw PE channels and emit wrongly-scaled states.
        # Require the stats up front.
        if self.config.use_normalization and self.norm_stats is None:
            raise ValueError(
                "use_normalization=True requires norm_stats; got None. Pass "
                "NormalizationStats or set use_normalization=False."
            )

        # dt_sfno is the network's trained macro step: state_update jumps
        # by exactly this, hybrid divides by it — zero/negative is
        # meaningless in both modes and would divide-by-zero in hybrid.
        if not self.config.dt_sfno > 0.0:
            raise ValueError(
                f"config.dt_sfno must be positive (trained macro step in "
                f"seconds); got {self.config.dt_sfno!r}."
            )

        if sfno_model is not None:
            # The config-derived guards above (hybrid residual contract,
            # normalization) only protect the model if the supplied network
            # actually matches ``config.sfno_config``.  Validate it so e.g.
            # a NON-residual network cannot be slipped under a residual
            # hybrid config — ``(y - x)/dt_sfno`` would then divide raw
            # decoder output with undefined scale (mirrors the U-Cast
            # bridge guard).
            net_cfg = getattr(sfno_model, "config", None)
            if net_cfg is None:
                raise ValueError(
                    "Supplied sfno_model exposes no .config attribute; the "
                    "SFNO PE bridge needs it to validate the architecture "
                    "contract (residual_prediction, channels). Pass a "
                    "legoesm.ml.sfno.SFNO (or a module carrying an "
                    "SFNOConfig as .config)."
                )
            if net_cfg != self.config.sfno_config:
                raise ValueError(
                    "Supplied sfno_model.config does not match "
                    "config.sfno_config; the architecture/contract guards "
                    "(residual_prediction, channels) would not apply to "
                    f"the actual network.\n  model:  {net_cfg}\n"
                    f"  config: {self.config.sfno_config}"
                )
            # Equal configs do NOT prove grid compatibility: the spectral
            # conv weights are sized (n_sh, ...) by the CONSTRUCTION grid.
            # A T5-trained net under a T8 wrapper passes the config check
            # and then fails deep in sh_analysis — check n_sh up front.
            _blocks = getattr(sfno_model, "blocks", None)
            if _blocks:
                _net_n_sh = getattr(
                    getattr(_blocks[0], "spectral_conv", None), "n_sh", None,
                )
                if _net_n_sh is not None and _net_n_sh != grid.n_sh:
                    raise ValueError(
                        f"Supplied sfno_model was built for n_sh="
                        f"{_net_n_sh} but this wrapper's grid has n_sh="
                        f"{grid.n_sh} (different spectral truncation). "
                        f"Rebuild or load the network on the same grid."
                    )
            self.sfno = sfno_model
        else:
            if key is None:
                key = jax.random.PRNGKey(0)
            self.sfno = SFNO(
                config=self.config.sfno_config,
                grid=grid,
                key=key,
            )

    def _check_state_update_dt(self, dt: float) -> None:
        """Refuse a caller ``dt`` that disagrees with ``dt_sfno``.

        In state_update mode the network advances the state by exactly
        ``config.dt_sfno`` regardless of ``dt``; silently accepting a
        different ``dt`` desynchronises the caller's clock from the
        model state (and, in ``step_with_physics``, applies physics over
        ``dt`` while the dynamics jumped ``dt_sfno``).

        ``dt`` must be a STATIC Python number on this path (it selects a
        fixed macro step; it is not integrable).  A traced ``dt`` (e.g.
        ``jax.jit`` over ``step`` with a jnp-scalar dt) is converted to a
        clear contract error here instead of an opaque
        ``TracerBoolConversionError`` downstream.
        """
        try:
            dt_val = float(dt)
        except TypeError as e:
            raise ValueError(
                "mode='state_update' requires a STATIC Python-float dt "
                "(the network advances the fixed macro step dt_sfno; dt "
                "cannot be traced). Mark dt static under jit, or call "
                "step() outside jit."
            ) from e
        if abs(dt_val - self.config.dt_sfno) > 1e-6 * self.config.dt_sfno:
            raise ValueError(
                f"mode='state_update' advances the state by exactly "
                f"dt_sfno={self.config.dt_sfno} s per call; got dt={dt_val}. "
                f"Pass dt=dt_sfno (or set config.dt_sfno to the desired "
                f"macro step)."
            )

    def step(
        self,
        state: SpectralHydrostaticState,
        dt: float,
        *,
        key: jax.Array | None = None,
    ) -> SpectralHydrostaticState:
        """Advance one time step.

        Parameters
        ----------
        state : SpectralHydrostaticState
            Current state.
        dt : float
            Time step [s].  In ``state_update`` mode this MUST equal
            ``config.dt_sfno`` (the network's trained macro step).
        key : jax.random.PRNGKey, optional
            MC-Dropout key for ``state_update`` mode.  ``None`` (default) is
            the deterministic step.  A stochastic ROLLOUT wants a FRESH key per
            macro step — reusing one key freezes the same dropout mask for the
            whole trajectory, which under-disperses the ensemble.  Rejected in
            ``hybrid_tendencies`` mode, where the network is used as a
            deterministic tendency operator inside an RK integrator (a
            per-stage random mask would make the integrator inconsistent).

        Returns
        -------
        SpectralHydrostaticState
            Advanced state.
        """
        if self.config.mode == "state_update":
            self._check_state_update_dt(dt)
            new_state = self._step_state_update(state, key=key)
        elif self.config.mode == "hybrid_tendencies":
            if key is not None:
                raise ValueError(
                    "step(key=...) is only supported in mode='state_update'; "
                    "the hybrid_tendencies path integrates the network as a "
                    "deterministic tendency operator."
                )
            new_state = self._step_hybrid(state, dt)
        else:
            raise ValueError(
                f"Unknown mode: {self.config.mode!r}. "
                f"Choose 'state_update' or 'hybrid_tendencies'."
            )

        return self._apply_postprocess(new_state, state)

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
        # SFNO PE has no PhysicsState carry slot — refuse a stateful
        # ``make_physics`` fn (tagged _requires_phys_state) rather than
        # silently reseed its prognostic fields every step (#405/#413).
        refuse_unthreaded_stateful_physics(
            physics_fn, None, where="SFNO PE step_with_physics()")
        if self.config.mode == "hybrid_tendencies":
            def combined_tendency(s):
                sfno_tend = self._sfno_tendency(s)
                _phys_result = physics_fn(s, self.grid, self.sigma_coord)
                phys_tend = _phys_result[0] if type(_phys_result) is tuple else _phys_result
                return jax.tree.map(
                    lambda a, b: a + b, sfno_tend, phys_tend
                )
            new_state = dispatch_integrator(state, combined_tendency, dt, self.config.time_integrator)
        elif self.config.mode == "state_update":
            # State update mode: SFNO prediction + physics tendencies
            # (same dt contract as step(); guard AFTER the carry-contract
            # refusal above so stateful-physics misuse stays the first,
            # more specific error).
            self._check_state_update_dt(dt)
            new_state = self._step_state_update(state)
            _phys_result = physics_fn(state, self.grid, self.sigma_coord)
            phys_tend = _phys_result[0] if type(_phys_result) is tuple else _phys_result
            new_state = jax.tree.map(
                lambda s, t: s + dt * t, new_state, phys_tend
            )
        else:
            # Mirror step(): unknown mode raises, never silently runs
            # state_update (dispatch hardening).
            raise ValueError(
                f"Unknown mode: {self.config.mode!r}. "
                f"Choose 'state_update' or 'hybrid_tendencies'."
            )

        # SAME postprocessing as the uncoupled step: an unfiltered coupled
        # rollout diverges exactly like the uncoupled one did.
        return self._apply_postprocess(new_state, state)

    def _apply_postprocess(
        self,
        new_state: SpectralHydrostaticState,
        old_state: SpectralHydrostaticState,
    ) -> SpectralHydrostaticState:
        """Damping + conservation + positivity, in the ONE correct order.

        Shared by :meth:`step` and :meth:`step_with_physics` so a coupled route
        cannot silently keep the divergent legacy behaviour — that split is how
        the physics path was left unguarded when the filter was first added
        (codex review, 2026-07-28).

        ORDER IS LOAD-BEARING — filter, THEN correct_mass, THEN clip:

        * The filter must not be the last thing to touch ``lnps_hat``. It leaves
          the n=0 coefficient at 1.0, so it preserves the global mean of
          ``ln(p_s)`` — but the conserved quantity is the mean of
          ``p_s = exp(lnps)``, and those are not the same functional. Changing
          the spread of ``lnps`` therefore moves ``<exp(lnps)>`` and re-breaks
          the correction (measured on the T8 unit fixture: ``<p_s>`` 1.0e5 ->
          1.79e6 Pa in one step with the order reversed). Note this is a
          "generally will", not an identity: for spatially uniform ``lnps`` the
          filter is a no-op and the mean is untouched.
        * The moisture clamp acts only on grid-space tracers, so it commutes
          with both, and is placed last so nothing can reintroduce a negative.
        """
        if self._spectral_filter is not None:
            new_state = apply_spectral_filter_to_state(
                new_state, self._spectral_filter)

        if self.config.correct_mass:
            new_state = self._apply_conservation(new_state, old_state)

        if self.config.clip_q and new_state.tracers:
            new_state = new_state._replace(
                tracers={
                    name: (
                        t.replace(data=clip_humidity(t.data))
                        if hasattr(t, "replace") else clip_humidity(t)
                    )
                    if name.startswith("q") else t
                    for name, t in new_state.tracers.items()
                }
            )
        return new_state

    def ensemble_step(
        self,
        state: SpectralHydrostaticState,
        key: jax.Array,
        n_members: int,
    ) -> SpectralHydrostaticState:
        """Advance one macro step as an MC-Dropout ensemble.

        Runs ``n_members`` forward passes with **active** dropout (distinct
        per-member keys), batched with ``jax.vmap``, and returns a state whose
        leaves carry a leading ``n_members`` axis.  This is the U-Cast
        stochasticity mechanism (arXiv:2604.09041) on the SFNO backbone: one
        trained network, an unlimited number of members, no diffusion sampler
        and no extra training runs.  Pair with ``area_weighted_afcrps`` for
        probabilistic training/scoring.

        Requires ``sfno_config.dropout > 0`` — with p=0 every member is the
        same deterministic forecast and CRPS collapses to the MAE, which is a
        silently-useless ensemble rather than an error, so it is refused.

        Each member is post-processed through the SAME
        :meth:`_apply_postprocess` chain (filter → mass → clip) as
        :meth:`step`; skipping it would let the members diverge exactly the
        way the unfiltered deterministic rollout did.
        """
        if self.config.mode != "state_update":
            raise ValueError(
                f"ensemble_step requires mode='state_update', got "
                f"{self.config.mode!r}."
            )
        if float(self.config.sfno_config.dropout) <= 0.0:
            raise ValueError(
                "ensemble_step needs sfno_config.dropout > 0 — MC-Dropout is "
                "the only stochasticity source, so at p=0 every member is "
                "identical and the CRPS degenerates to the MAE."
            )
        member_keys = jax.random.split(key, n_members)

        def one_member(member_key):
            new_state = self._step_state_update(state, key=member_key)
            return self._apply_postprocess(new_state, state)

        return jax.vmap(one_member)(member_keys)

    def _step_state_update(
        self,
        state: SpectralHydrostaticState,
        key: jax.Array | None = None,
    ) -> SpectralHydrostaticState:
        """Direct state update: SFNO(state_t) → state_{t+1}.

        ``key`` is forwarded to the network's dropout: ``None`` → deterministic
        (inference dropout); a key → one MC-Dropout sample.
        """
        x = pack_pe_state(state, self.grid, self.sigma_coord)
        x = x.astype(jnp.float32)

        # ``norm_stats`` is guaranteed present when use_normalization=True
        # (validated in __init__), so this never silently skips.
        if self.config.use_normalization:
            x = normalize(x, self.norm_stats)

        y = self.sfno(x, self.grid, key=key)

        if self.config.use_normalization:
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
        """Compute SFNO-derived tendencies.

        The SFNO is a next-state predictor (``residual_prediction=True``
        enforced in ``__init__``: ``y = x + delta``), NOT a per-second
        tendency net, so the tendency is the finite difference
        ``(y - x) / dt_sfno`` in packed-channel space.  Reading ``y``
        directly as a tendency (the previous behaviour) integrated an
        O(state)-sized field as a d/dt — wrong by a factor ~dt_sfno.
        Pure array ops; differentiable through both ``y`` and ``x``.

        Normalization is rejected for this mode in ``__init__``, so
        ``x`` here is exactly the tensor the network consumed.
        """
        x = pack_pe_state(state, self.grid, self.sigma_coord)
        x = x.astype(jnp.float32)

        y = self.sfno(x, self.grid)

        tend = (y - x) / self.config.dt_sfno

        return unpack_pe_output(tend, state, self.grid, mode="tendencies")

    def _apply_conservation(
        self,
        new_state: SpectralHydrostaticState,
        old_state: SpectralHydrostaticState,
    ) -> SpectralHydrostaticState:
        """Global dry-air-mass correction (grid space, then back to spectral).

        Scope note (kept accurate — an earlier version of this docstring said
        humidity clipping was unwired and that moisture is a spectral tracer;
        both were wrong): ``clip_q`` IS wired, in :meth:`_apply_postprocess`,
        and tracers are GRID-space fields so the clamp needs no SH round-trip.
        ``correct_moisture_budget`` remains unimplemented — it needs the
        column-integrated E - P budget and this state carries no
        precipitation/evaporation channel — and the constructor refuses it.

        CAVEAT on the ``max(p_s, 1.0)`` floor below: it is lossy. Once the
        prediction drives p_s negative somewhere, clamping breaks the very
        conservation this function just imposed. It only fires in the diverged
        regime the post-step spectral filter exists to prevent, so it is left
        as a loud-but-unrepaired edge rather than silently rescaled.
        """
        if not self.config.correct_mass:
            return new_state

        grid = self.grid

        # Get surface pressure from lnps
        lnps_new = sh_synthesis(grid, new_state.lnps_hat.data)
        lnps_old = sh_synthesis(grid, old_state.lnps_hat.data)
        p_s_new = jnp.exp(lnps_new)
        p_s_old = jnp.exp(lnps_old)

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
