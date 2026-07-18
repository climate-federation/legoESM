"""Shared integration methods for LegoESM model classes."""

import functools

import jax


def physics_requires_phys_state(physics_fn) -> bool:
    """True if *physics_fn* (or a partial / ``functools.wraps`` wrapper of
    it) is a ``combined.make_physics`` output tagged
    ``_requires_phys_state`` (a prognostic, carry-bearing scheme).

    The tag lives on the raw ``make_physics`` callable.  A
    ``functools.partial`` (e.g. binding ``forcing=``) or an
    ``@functools.wraps`` wrapper strips the direct attribute, so a plain
    ``getattr`` would report a wrapped stateful physics as diagnostic and
    let it bypass the carry contract (codex #413 review).  Follow the
    ``functools.partial.func`` / ``__wrapped__`` chain before trusting a
    ``False``.  Bare hand-written closures expose no such link and so MUST
    propagate the tag explicitly — the in-repo radiation/forcing wrappers
    set ``wrapper._requires_phys_state = physics_requires_phys_state(inner)``.
    """
    seen = 0
    fn = physics_fn
    while fn is not None and seen < 16:
        if getattr(fn, "_requires_phys_state", False):
            return True
        if isinstance(fn, functools.partial):
            fn = fn.func
        elif getattr(fn, "__wrapped__", None) is not None:
            fn = fn.__wrapped__
        else:
            return False
        seen += 1
    return False


def refuse_unthreaded_stateful_physics(physics_fn, phys_state, *, where):
    """Refuse a stateful physics_fn handed to a per-step entry without
    its ``PhysicsState`` carry.

    Issue #405/#413: ``combined.make_physics`` tags its output with
    ``_requires_phys_state`` when the configured physics is prognostic
    (TKE-family / MYNN-2.5 turbulence, mass_flux / edmf / bechtold /
    profile convection, prognostic-spectral GWD).  Such a physics_fn
    carries state across steps; calling a per-step ``step`` /
    ``step_with_physics`` / ``step_cell_centre`` with ``phys_state=None``
    silently reseeds that state every step — the exact failure class the
    carry threading was built to remove.  A non-stateful (untagged)
    physics_fn is unaffected, so the legacy ``phys_state=None`` call
    stays byte-identical.

    Centralised here (already imported by every dycore via
    :class:`IntegrationMixin`) so the loud-contract guard is identical
    across ``integrate()`` and the public per-step APIs of the CDGrid /
    lat-lon / MPAS primitive-equation models.

    Parameters
    ----------
    physics_fn : callable or None
    phys_state : PhysicsState or None
    where : str
        Caller label for the error message (e.g. ``"CDGrid step()"``).
    """
    if (physics_fn is not None
            and phys_state is None
            and physics_requires_phys_state(physics_fn)):
        raise NotImplementedError(
            f"{where} received a stateful physics_fn but no phys_state "
            "carry — the prognostic physics (TKE / convection / GWD) "
            "would silently reseed every step (issue #405/#413).  Seed "
            "one with init_physics_state and thread it back via "
            "phys_state=."
        )


class IntegrationMixin:
    """Mixin providing integrate() and integrate_scan() for any model with a .step() method.

    If the model also defines step_with_physics(state, dt, physics_fn),
    integrate() will delegate to it when a physics_fn is supplied.
    """

    def integrate(self, state, duration, dt, save_every=1, physics_fn=None,
                  phys_state=None):
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : pytree
        duration : float — total integration time [seconds]
        dt : float — time step [seconds]
        save_every : int — save state every N steps
        physics_fn : callable, optional
            Physics forcing function. If provided and the model has
            a ``step_with_physics`` method, that method is used;
            otherwise falls back to ``self.step``.
        phys_state : PhysicsState, optional
            Stateful-physics carry (issue #405/#413).  Required when
            *physics_fn* is tagged ``_requires_phys_state`` (stateful
            schemes) — seed it with
            ``legoesm.atmosphere.physics.physics_state.init_physics_state``.
            The loop feeds the model's updated ``_phys_state`` back
            each step so the prognostic physics keeps its memory.

        Returns
        -------
        final_state, trajectory : (pytree, list of pytree)
        """
        n_steps = int(duration / dt)
        trajectory = [state]
        use_physics = physics_fn is not None and hasattr(self, "step_with_physics")
        # Models without step_with_physics (e.g. the MPAS PE) may still
        # accept physics through step(..., physics_fn=...).  Detect
        # that ONCE; silently running dynamics-only when the caller
        # supplied physics is the same silent-wrong class as a dropped
        # carry (codex round 3).
        _step_takes_physics = False
        if physics_fn is not None and not use_physics:
            import inspect
            _step_takes_physics = (
                "physics_fn" in inspect.signature(self.step).parameters
            )
            if not _step_takes_physics:
                raise NotImplementedError(
                    f"{type(self).__name__} has neither step_with_physics "
                    "nor a step(physics_fn=...) parameter — integrate() "
                    "cannot apply the supplied physics_fn (it would be "
                    "silently dropped)."
                )
        # Issue #405/#413: looping a stateful physics_fn WITHOUT a carry
        # silently reseeds its prognostic fields every step.  Refuse
        # loudly; with a carry supplied, thread it through the models'
        # step/step_with_physics ``phys_state`` contract.
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="integrate()")
        for i in range(n_steps):
            if use_physics:
                if phys_state is not None:
                    state = self.step_with_physics(
                        state, dt, physics_fn, phys_state=phys_state)
                    phys_state = getattr(self, "_phys_state", phys_state)
                else:
                    state = self.step_with_physics(state, dt, physics_fn)
            elif _step_takes_physics:
                if phys_state is not None:
                    state = self.step(state, dt, physics_fn=physics_fn,
                                      phys_state=phys_state)
                    phys_state = getattr(self, "_phys_state", phys_state)
                else:
                    state = self.step(state, dt, physics_fn=physics_fn)
            else:
                state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
        return state, trajectory

    def integrate_scan(self, state, n_steps, dt):
        """Integrate using jax.lax.scan (differentiable, JIT-friendly).

        Note: does not support physics_fn (use integrate for that).

        Parameters
        ----------
        state : pytree
        n_steps : int
        dt : float

        Returns
        -------
        final_state, trajectory : (pytree, pytree with leading n_steps axis)
        """
        def scan_fn(s, _):
            new_s = self.step(s, dt)
            return new_s, new_s
        return jax.lax.scan(scan_fn, state, xs=None, length=n_steps)

    # Anchored dry-mass target: shared by every integrator model (the
    # conservation fixer anchors ``fix_mass_*`` to this instead of the
    # previous step, preventing slow drift).  Subclasses initialise
    # ``self._target_mass = None`` in ``__init__``; these setters were
    # copy-pasted identically across ~12 dycore models before this mixin.
    def reset_target_mass(self) -> None:
        """Clear the anchored mass target (recompute on the next step)."""
        self._target_mass = None

    def set_target_mass(self, target_mass) -> None:
        """Explicitly set the anchored mass target (e.g. the t=0 dry mass)."""
        self._target_mass = target_mass
