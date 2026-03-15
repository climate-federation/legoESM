"""Shared integration methods for LegoESM model classes."""

import jax
import jax.numpy as jnp


class IntegrationMixin:
    """Mixin providing integrate() and integrate_scan() for any model with a .step() method.

    If the model also defines step_with_physics(state, dt, physics_fn),
    integrate() will delegate to it when a physics_fn is supplied.
    """

    def integrate(self, state, duration, dt, save_every=1, physics_fn=None):
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

        Returns
        -------
        final_state, trajectory : (pytree, list of pytree)
        """
        n_steps = int(duration / dt)
        trajectory = [state]
        use_physics = physics_fn is not None and hasattr(self, "step_with_physics")
        for i in range(n_steps):
            if use_physics:
                state = self.step_with_physics(state, dt, physics_fn)
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
