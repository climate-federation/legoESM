"""Shared integration methods for LegoESM model classes."""

import jax
import jax.numpy as jnp


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
        if (physics_fn is not None
                and getattr(physics_fn, "_requires_phys_state", False)
                and phys_state is None):
            raise NotImplementedError(
                "integrate() received a stateful physics_fn but no "
                "phys_state carry — the prognostic physics would "
                "silently reseed every step (issue #405/#413).  Seed "
                "one with init_physics_state and pass phys_state=."
            )
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
        return jax.lax.scan(scan_fn, state, jnp.arange(n_steps))
