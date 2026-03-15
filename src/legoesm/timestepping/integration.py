"""Shared integration methods for LegoESM model classes."""

import jax
import jax.numpy as jnp


class IntegrationMixin:
    """Mixin providing integrate() and integrate_scan() for any model with a .step() method."""

    def integrate(self, state, duration, dt, save_every=1):
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : pytree
        duration : float — total integration time [seconds]
        dt : float — time step [seconds]
        save_every : int — save state every N steps

        Returns
        -------
        final_state, trajectory : (pytree, list of pytree)
        """
        n_steps = int(duration / dt)
        trajectory = [state]
        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
        return state, trajectory

    def integrate_scan(self, state, n_steps, dt):
        """Integrate using jax.lax.scan (differentiable, JIT-friendly).

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
