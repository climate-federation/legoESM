"""Ensemble parallelization for legoESM.

Provides utilities to run many independent model simulations simultaneously
using JAX's vectorization (vmap), multi-device sharding, or both.

Strategies
----------
1. **vmap** (single device):
   Vectorize the model step over an ensemble (batch) dimension.
   XLA fuses kernels across ensemble members for high throughput.
   Best for 10--50 members on one GPU/TPU.

2. **sharded** (multi-device):
   Distribute ensemble members across devices using JAX named sharding.
   Each device runs n_members / n_devices members.  No inter-device
   communication needed (embarrassingly parallel).

3. **hybrid** (multi-device + vmap):
   Shard the ensemble across devices; within each device, vmap over
   the local batch.  Best for large ensembles (50--100+) on multi-GPU
   or TPU-pod systems.

Architecture notes
~~~~~~~~~~~~~~~~~~
* The ensemble dimension is always the **leading** axis of every leaf
  array.  A batched ``HydrostaticState`` has field shapes
  ``(n_members, 6, n, n, nlev)``.
* Grid, vertical coordinate, and configuration are **shared** across
  ensemble members (not vmapped).  Because ``model.step`` captures
  them via the model instance (``self``), no special treatment is
  needed: only the ``state`` argument is vmapped.
* Time integration uses ``jax.lax.scan`` **outside** vmap.  The scan
  body calls the vmapped step function once, and XLA fuses the
  vectorised work within each time step.  This compiles a single
  scan loop (not one per member), giving optimal memory usage and
  compilation time.
* All operations remain fully differentiable.  ``jax.grad`` through
  ``ensemble_integrate`` propagates through scan + vmap with correct
  AD semantics.

Usage
-----
::

    from legoesm.parallel.ensemble import (
        EnsembleConfig,
        perturb_initial_conditions,
        make_ensemble_step,
        ensemble_integrate,
        ensemble_mean,
        ensemble_std,
    )

    # 1. Create ensemble of perturbed initial conditions
    batched_state = perturb_initial_conditions(
        state, jax.random.PRNGKey(0), n_members=32, scale=0.01,
    )

    # 2. Wrap the model step with vmap
    ensemble_step = make_ensemble_step(model.step, in_axes=(0, None))

    # 3. Integrate forward (scan + vmap)
    final_states = ensemble_integrate(
        ensemble_step, batched_state, n_steps=1000, dt=600.0,
    )

    # 4. Analyse spread
    mean_state = ensemble_mean(final_states)
    spread = ensemble_spread(final_states)


Multi-device example
~~~~~~~~~~~~~~~~~~~~
::

    mesh = create_ensemble_mesh(n_members=64)
    sharded = shard_ensemble(batched_state, mesh)
    # jit-compiled step automatically distributes across devices
    final = ensemble_integrate(ensemble_step, sharded, ...)
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


# ============================================================================
# State batching
# ============================================================================

def stack_states(states: Sequence) -> Any:
    """Stack individual states into a batched state with leading ensemble dim.

    Works with any JAX pytree (NamedTuples, nested dicts, Field objects).
    Each leaf array gains a leading dimension of size ``len(states)``.

    Parameters
    ----------
    states : sequence of pytrees
        Individual model states with identical pytree structure.

    Returns
    -------
    pytree
        Batched state where each leaf has shape ``(n_members, ...)``.
    """
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)


def unstack_states(batched_state: Any) -> list:
    """Split a batched state into individual states.

    Parameters
    ----------
    batched_state : pytree
        Batched state with leading ensemble dimension.

    Returns
    -------
    list of pytrees
        Individual model states.
    """
    leaves, treedef = jax.tree.flatten(batched_state)
    n = leaves[0].shape[0]
    return [
        treedef.unflatten([leaf[i] for leaf in leaves])
        for i in range(n)
    ]


# ============================================================================
# Initial-condition perturbation
# ============================================================================

def perturb_initial_conditions(
    state: Any,
    key: jax.Array,
    n_members: int,
    scale: float = 0.01,
    fields: Sequence[str] | None = None,
    multiplicative: bool = True,
) -> Any:
    """Create an ensemble of perturbed initial conditions.

    Replicates *state* ``n_members`` times and adds small random
    perturbations to selected fields.

    Parameters
    ----------
    state : pytree (typically a NamedTuple of Fields)
        Single model state.
    key : jax.Array
        PRNG key for generating perturbations.
    n_members : int
        Number of ensemble members to create.
    scale : float
        Perturbation magnitude.  Relative (``x * (1 + scale*ε)``) when
        *multiplicative* is True; absolute (``x + scale*ε``) otherwise.
    fields : sequence of str, optional
        Names of NamedTuple fields to perturb.  If ``None``, perturb all
        prognostic fields (anything except those whose name contains
        ``"phis"``).
    multiplicative : bool
        Whether to use multiplicative (True) or additive (False) noise.

    Returns
    -------
    pytree
        Batched state with each leaf having shape ``(n_members, ...)``.
    """
    # Tile state: each leaf (S,...) -> (n_members, S,...)
    batched = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None], (n_members,) + x.shape),
        state,
    )

    if not hasattr(state, '_fields'):
        # Generic pytree: perturb every leaf
        flat, treedef = jax.tree.flatten(batched)
        perturbed = []
        for i, leaf in enumerate(flat):
            subkey = jax.random.fold_in(key, i)
            noise = jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype)
            if multiplicative:
                perturbed.append(leaf * (1.0 + scale * noise))
            else:
                perturbed.append(leaf + scale * noise)
        return treedef.unflatten(perturbed)

    # NamedTuple with named fields
    if fields is None:
        fields = [
            f for f in state._fields
            if 'phis' not in f.lower()
        ]

    updates = {}
    for i, fname in enumerate(state._fields):
        if fname not in fields:
            continue
        subkey = jax.random.fold_in(key, i)
        batched_val = getattr(batched, fname)

        # Skip None fields (e.g., optional tracers)
        if batched_val is None:
            continue

        # Handle Field objects (data is the JAX leaf) vs raw arrays
        if hasattr(batched_val, 'data') and hasattr(batched_val, 'replace'):
            data = batched_val.data
            noise = jax.random.normal(subkey, data.shape, dtype=data.dtype)
            if multiplicative:
                new_data = data * (1.0 + scale * noise)
            else:
                new_data = data + scale * noise
            updates[fname] = batched_val.replace(data=new_data)
        elif hasattr(batched_val, 'shape'):
            noise = jax.random.normal(
                subkey, batched_val.shape, dtype=batched_val.dtype,
            )
            if multiplicative:
                updates[fname] = batched_val * (1.0 + scale * noise)
            else:
                updates[fname] = batched_val + scale * noise

    return batched._replace(**updates)


def perturb_parameters(
    params: Any,
    key: jax.Array,
    n_members: int,
    scale: float = 0.01,
    fields: Sequence[str] | None = None,
) -> Any:
    """Create ensemble of perturbed model parameters.

    Identical interface to :func:`perturb_initial_conditions` but
    intended for parameter arrays or config-like pytrees.
    """
    return perturb_initial_conditions(
        params, key, n_members, scale=scale,
        fields=fields, multiplicative=True,
    )


# ============================================================================
# Ensemble step wrappers
# ============================================================================

def make_ensemble_step(
    step_fn: Callable,
    in_axes: int | tuple = 0,
    out_axes: int = 0,
) -> Callable:
    """Create a vmapped step function for ensemble execution.

    Wraps *step_fn* with :func:`jax.vmap` to vectorize over the
    ensemble dimension.

    Common patterns::

        # Dynamics only: step(state, dt) -> state
        ensemble_step = make_ensemble_step(model.step, in_axes=(0, None))

        # With extra batched args: step(state, q_v, dt) -> (state, q_v)
        ensemble_step = make_ensemble_step(full_step, in_axes=(0, 0, None))

        # Stochastic forcing per member: step(state, forcing, dt)
        ensemble_step = make_ensemble_step(step, in_axes=(0, 0, None))

    Parameters
    ----------
    step_fn : callable
        Model step function.
    in_axes : int or tuple
        Axes specification for vmap.  ``0`` = batched,
        ``None`` = shared across ensemble members.
    out_axes : int
        Output axis for batched results.

    Returns
    -------
    callable
        Vmapped step function.
    """
    return jax.vmap(step_fn, in_axes=in_axes, out_axes=out_axes)


def make_ensemble_step_jit(
    step_fn: Callable,
    in_axes: int | tuple = 0,
    out_axes: int = 0,
    donate_argnums: tuple[int, ...] = (),
) -> Callable:
    """Create a vmapped + JIT-compiled step function.

    Combines vmap and jit for maximum single-step performance.
    Use *donate_argnums* to recycle input buffers and reduce
    peak memory during time-stepping loops.

    Parameters
    ----------
    step_fn : callable
        Model step function.
    in_axes, out_axes
        As in :func:`make_ensemble_step`.
    donate_argnums : tuple of int
        Argument indices whose buffers can be donated to the output.
        Typically ``(0,)`` for the state argument.

    Returns
    -------
    callable
        Vmapped + JIT-compiled step function.
    """
    vmapped = jax.vmap(step_fn, in_axes=in_axes, out_axes=out_axes)
    return jax.jit(vmapped, donate_argnums=donate_argnums)


# ============================================================================
# Time integration
# ============================================================================

def ensemble_integrate(
    step_fn: Callable,
    init_states: Any,
    n_steps: int,
    dt: float,
    checkpoint: bool = False,
    save_trajectory: bool = False,
) -> Any | tuple[Any, Any]:
    """Integrate an ensemble forward in time using ``jax.lax.scan``.

    The scan body calls *step_fn* (which should already be vmapped) once
    per time step.  XLA fuses the vectorized work within each step.

    Parameters
    ----------
    step_fn : callable
        Vmapped step function: ``(batched_state, dt) -> batched_state``.
        Use :func:`make_ensemble_step` to create this from a single-member
        step function.
    init_states : pytree
        Batched initial states with leading ensemble dimension.
    n_steps : int
        Number of time steps to integrate.
    dt : float
        Time step size [seconds].
    checkpoint : bool
        If True, apply ``jax.checkpoint`` to the scan body.  Reduces
        memory from O(n_steps) to O(√n_steps) for reverse-mode AD,
        at the cost of recomputing the forward pass during backward.
    save_trajectory : bool
        If True, return ``(final_state, trajectory)`` where each
        trajectory leaf has shape ``(n_steps, n_members, ...)``.

    Returns
    -------
    final_states : pytree
        Final batched state after *n_steps* iterations.
    trajectory : pytree, optional
        Full trajectory if *save_trajectory* is True.
    """
    _step = step_fn
    if checkpoint:
        _step = jax.checkpoint(step_fn, prevent_cse=False)

    def scan_fn(states, _):
        new_states = _step(states, dt)
        return new_states, (new_states if save_trajectory else None)

    final, trajectory = jax.lax.scan(
        scan_fn, init_states, None, length=n_steps,
    )

    if save_trajectory:
        return final, trajectory
    return final


def ensemble_integrate_with_forcing(
    step_fn: Callable,
    init_states: Any,
    forcings: Any,
    dt: float,
    checkpoint: bool = False,
) -> Any:
    """Integrate with time-varying forcing arrays.

    Like :func:`ensemble_integrate`, but passes a per-step forcing
    slice as the second argument to *step_fn*.  ``forcings`` is a
    pytree whose leaves have time as the first axis.

    Parameters
    ----------
    step_fn : callable
        Vmapped step: ``(batched_state, forcing_t, dt) -> batched_state``.
    init_states : pytree
        Batched initial states.
    forcings : pytree
        Per-step forcing.  Each leaf has shape ``(n_steps, ...)``.
        If a leaf should also be batched over ensemble members, include
        that dimension: ``(n_steps, n_members, ...)``.
    dt : float
        Time step size [seconds].
    checkpoint : bool
        Apply gradient checkpointing.

    Returns
    -------
    final_states : pytree
        Final batched state.
    """
    _step = step_fn
    if checkpoint:
        _step = jax.checkpoint(step_fn, prevent_cse=False)

    def scan_fn(states, forcing_t):
        new_states = _step(states, forcing_t, dt)
        return new_states, None

    final, _ = jax.lax.scan(scan_fn, init_states, forcings)
    return final


# ============================================================================
# Multi-device sharding
# ============================================================================

def create_ensemble_mesh(
    n_members: int,
    devices: Sequence | None = None,
) -> Mesh:
    """Create a JAX device mesh for ensemble sharding.

    Maps the ensemble dimension to available devices.  Each device
    handles ``n_members / n_devices`` ensemble members.

    Parameters
    ----------
    n_members : int
        Total number of ensemble members.  Must be divisible by the
        number of devices.
    devices : sequence of jax.Device, optional
        Devices to use.  Default: all available.

    Returns
    -------
    jax.sharding.Mesh
        Device mesh with ``'ensemble'`` axis.
    """
    if devices is None:
        devices = jax.devices()
    n_devices = len(devices)

    if n_members % n_devices != 0:
        raise ValueError(
            f"n_members ({n_members}) must be divisible by "
            f"n_devices ({n_devices})"
        )

    return Mesh(np.array(devices), axis_names=('ensemble',))


def shard_ensemble(
    batched_state: Any,
    mesh: Mesh,
) -> Any:
    """Distribute batched ensemble state across devices.

    Shards the leading (ensemble) dimension across the ``'ensemble'``
    axis of the device mesh.  Scalars are replicated.

    Parameters
    ----------
    batched_state : pytree
        Batched state with leading ensemble dimension.
    mesh : jax.sharding.Mesh
        Device mesh from :func:`create_ensemble_mesh`.

    Returns
    -------
    pytree
        State with arrays sharded across devices.
    """
    sharding = NamedSharding(mesh, PartitionSpec('ensemble'))
    replicated = NamedSharding(mesh, PartitionSpec())

    def shard_leaf(x):
        if x.ndim == 0:
            return jax.device_put(x, replicated)
        return jax.device_put(x, sharding)

    return jax.tree.map(shard_leaf, batched_state)


def gather_ensemble(sharded_state: Any) -> Any:
    """Gather sharded ensemble state to a single device.

    Parameters
    ----------
    sharded_state : pytree
        Sharded state distributed across devices.

    Returns
    -------
    pytree
        State gathered to device 0.
    """
    # Hoist ``jax.devices()`` out of the tree.map closure so the
    # device list isn't walked once per pytree leaf.  Atmospheric
    # state has 30+ leaves; the lookup is cheap individually but adds
    # measurable overhead at each gather call.
    dev0 = jax.devices()[0]
    return jax.tree.map(lambda x: jax.device_put(x, dev0), sharded_state)


# ============================================================================
# Ensemble statistics
# ============================================================================

def ensemble_mean(batched_state: Any) -> Any:
    """Compute ensemble mean (average over leading dimension).

    Parameters
    ----------
    batched_state : pytree
        Batched state with leading ensemble dimension.

    Returns
    -------
    pytree
        Mean state (ensemble dimension removed).
    """
    return jax.tree.map(lambda x: jnp.mean(x, axis=0), batched_state)


def ensemble_std(batched_state: Any) -> Any:
    """Compute ensemble standard deviation.

    Parameters
    ----------
    batched_state : pytree
        Batched state with leading ensemble dimension.

    Returns
    -------
    pytree
        Standard deviation (ensemble dimension removed).
    """
    return jax.tree.map(lambda x: jnp.std(x, axis=0), batched_state)


def ensemble_percentile(batched_state: Any, q: float) -> Any:
    """Compute ensemble percentile.

    Parameters
    ----------
    batched_state : pytree
        Batched state with leading ensemble dimension.
    q : float
        Percentile in [0, 100].

    Returns
    -------
    pytree
        q-th percentile (ensemble dimension removed).
    """
    return jax.tree.map(
        lambda x: jnp.percentile(x, q, axis=0), batched_state,
    )


def ensemble_spread(batched_state: Any) -> dict[str, float]:
    """Compute RMS spread for each named field.

    Parameters
    ----------
    batched_state : pytree (NamedTuple)
        Batched state.

    Returns
    -------
    dict
        Field name -> RMS spread (root-mean-square of std dev).
    """
    std_state = ensemble_std(batched_state)
    if not hasattr(batched_state, '_fields'):
        return {}
    # Stack the per-field RMS into one ``jnp.stack`` and pull host
    # in a single transfer — replaces per-field
    # ``float(jnp.sqrt(jnp.mean(val ** 2)))`` chain (one device→host
    # sync per field).  Each ensemble component (T, u, v, p_s, q_v,
    # q_c, q_r, …) was previously its own GPU stall.
    names: list[str] = []
    rms_terms: list[jax.Array] = []
    for fname in batched_state._fields:
        val = getattr(std_state, fname)
        if val is None:
            continue
        if hasattr(val, 'data'):
            val = val.data
        names.append(fname)
        rms_terms.append(jnp.sqrt(jnp.mean(val ** 2)))
    if not names:
        return {}
    host = np.asarray(jnp.stack(rms_terms))
    return {fname: float(host[i]) for i, fname in enumerate(names)}


def ensemble_crps(
    ensemble_vals: jax.Array,
    observation: jax.Array,
) -> jax.Array:
    """Continuous Ranked Probability Score (CRPS) for an ensemble forecast.

    CRPS = E|X - y| - 0.5 * E|X - X'|

    where X, X' are independent ensemble members and y is the observation.
    Lower is better; CRPS = 0 for a perfect deterministic forecast.

    Parameters
    ----------
    ensemble_vals : array, shape (n_members, ...)
        Ensemble forecast values.
    observation : array, shape (...)
        Observed / analysis values.

    Returns
    -------
    scalar — spatially averaged CRPS.
    """
    n = ensemble_vals.shape[0]
    # E|X - y|: mean absolute error across members
    mae = jnp.mean(jnp.abs(ensemble_vals - observation[None]), axis=0)
    # 0.5 * E|X - X'| over all n^2 ordered member pairs
    #   = 0.5 * 2 * sum_{i<j} |x_i - x_j| / n^2 = sum_{i<j} |x_i - x_j| / n^2
    # Use sorted ensemble for O(n log n) instead of O(n^2)
    sorted_ens = jnp.sort(ensemble_vals, axis=0)
    # For sorted values: sum_{i<j}(x_j - x_i) = sum_k (2k - n + 1) * x_k
    weights = 2.0 * jnp.arange(n).astype(ensemble_vals.dtype) - n + 1.0
    # Reshape weights for broadcasting
    w_shape = (n,) + (1,) * (ensemble_vals.ndim - 1)
    half_spread = jnp.sum(weights.reshape(w_shape) * sorted_ens, axis=0) / (n * n)

    crps = mae - half_spread
    return jnp.mean(crps)


def ensemble_rank_histogram(
    ensemble_vals: jax.Array,
    observation: jax.Array,
) -> jax.Array:
    """Compute rank histogram (Talagrand diagram) bin counts.

    For each grid point, find the rank of the observation within the
    sorted ensemble. A flat histogram indicates a well-calibrated ensemble.

    Parameters
    ----------
    ensemble_vals : array, shape (n_members, ...)
        Ensemble forecast values.
    observation : array, shape (...)
        Observed / analysis values.

    Returns
    -------
    array, shape (n_members + 1,) — histogram bin counts.
    """
    n = ensemble_vals.shape[0]
    # Rank = number of ensemble members below the observation
    ranks = jnp.sum(ensemble_vals < observation[None], axis=0)  # (...)
    # Flatten and bin
    flat_ranks = ranks.ravel()
    bins = jnp.arange(n + 2)
    hist = jnp.histogram(flat_ranks, bins=bins)[0]
    return hist
