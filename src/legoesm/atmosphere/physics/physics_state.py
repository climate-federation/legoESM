"""Explicit prognostic state for all atmospheric physics modules.

Physics state is carried through the time loop alongside the dynamical
state.  It is checkpointable, vmappable, and visible to JAX tracing.

This replaces the previous pattern of mutable ``nonlocal`` variables
captured in factory closures, which prevented checkpoint/restart of
physics state, ensemble ``vmap``, and clean JIT tracing.

Every physics module that carries prognostic variables (TKE, convective
mass flux, GWD wave action spectrum) reads from and writes to fields
in this state.  Modules that are disabled (``scheme="none"``) or
stateless simply ignore the corresponding fields.

Example
-------
>>> from legoesm.atmosphere.physics.physics_state import (
...     PhysicsState, init_physics_state,
... )
>>> ps = init_physics_state(ncol=1536, nlev=40, physics_config=config)
>>> # Thread through time loop:
>>> tendencies, ps = physics_fn(state, grid, coord, ps)
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class PhysicsState(NamedTuple):
    """Prognostic physics state — passed through time loop, checkpointable.

    All fields are concrete ``jax.Array`` instances (never ``None``).
    Unused fields are zero-filled and ignored by their modules.

    Fields
    ------
    tke : jax.Array, shape (ncol, nlev)
        Turbulent kinetic energy for TKE / CLUBB-lite / EDMF schemes.
        Initialized to ``tke_min`` when active, zero otherwise.
    conv_prog : jax.Array, shape (ncol,)
        Convection prognostic variable: mass flux ``M_c`` for mass_flux
        scheme, updraft area fraction ``a_u`` for EDMF scheme.
        Initialized to scheme default when active, zero otherwise.
    gwd_spectrum : jax.Array, shape (ncol, n_azimuths, n_wavenumbers)
        Gravity wave drag wave action spectrum for the prognostic
        spectral scheme.  Shape is ``(ncol, 1, 1)`` when inactive
        (minimal allocation).
    """
    tke: jnp.ndarray
    conv_prog: jnp.ndarray
    gwd_spectrum: jnp.ndarray


def init_physics_state(
    ncol: int,
    nlev: int,
    physics_config,
    *,
    dtype=None,
) -> PhysicsState:
    """Create initial physics state with correct shapes and defaults.

    Parameters
    ----------
    ncol : int
        Number of atmospheric columns (e.g. 6*n*n for cubed-sphere).
    nlev : int
        Number of vertical levels.
    physics_config : PhysicsConfig
        Unified physics configuration that determines which modules
        need prognostic state and what their initial values are.
    dtype : jnp.dtype or None
        Storage dtype for the persistent prognostic arrays.  When
        ``None`` (default) JAX picks its default float dtype (float64
        under x64).  Pass an explicit dtype (typically the precision
        policy's storage dtype) when running in a precision policy
        that holds the model state at f32 — defaulting otherwise
        forces every physics step to read/write the persistent state
        at f64 even though the column physics path runs at f32, which
        doubles the GPU bandwidth on the prognostic carry.

    Returns
    -------
    PhysicsState
        Initialized state ready for the time loop.

    Notes
    -----
    The non-prognostic branches (``conv_scheme != "mass_flux"|"edmf"``,
    ``gwd_scheme != "prognostic_spectral"``) still allocate
    zero-element placeholders — there's nothing to keep at any
    particular precision in those cases.  The ``dtype`` kwarg only
    matters for the configurations that produce a non-trivial
    persistent prognostic array.
    """
    # --- Turbulence TKE ---
    turb_cfg = physics_config.turbulence
    tke_schemes = ("tke", "clubb_lite", "edmf")
    if turb_cfg.scheme in tke_schemes:
        scheme_sub = getattr(turb_cfg, turb_cfg.scheme)
        tke_min = getattr(scheme_sub, "tke_min", 1e-6)
        tke = jnp.full((ncol, nlev), tke_min, dtype=dtype)
    else:
        tke = jnp.zeros((ncol, nlev), dtype=dtype)

    # --- Convection prognostic ---
    conv_cfg = physics_config.convection
    if conv_cfg.scheme == "mass_flux":
        conv_prog = jnp.full((ncol,), conv_cfg.mass_flux.M_c_init, dtype=dtype)
    elif conv_cfg.scheme == "edmf":
        conv_prog = jnp.full((ncol,), conv_cfg.edmf.a_u_init, dtype=dtype)
    else:
        conv_prog = jnp.zeros((ncol,), dtype=dtype)

    # --- GWD wave action spectrum ---
    gwd_cfg = physics_config.gravity_wave_drag
    if gwd_cfg.scheme == "prognostic_spectral":
        sc = gwd_cfg.prognostic_spectral
        gwd_spectrum = jnp.full(
            (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux,
            dtype=dtype,
        )
    else:
        # Minimal allocation — zero-element trailing dims avoid wasted memory.
        gwd_spectrum = jnp.zeros((ncol, 1, 1), dtype=dtype)

    return PhysicsState(
        tke=tke,
        conv_prog=conv_prog,
        gwd_spectrum=gwd_spectrum,
    )


def update_physics_state(phys_state, updates):
    """Rebuild a ``PhysicsState`` from per-field updates returned by sub-physics.

    Co-located with ``PhysicsState`` and ``init_physics_state`` so that adding
    a new prognostic field only requires editing this one module.

    Parameters
    ----------
    phys_state : PhysicsState or None
        Input physics state.  When ``None``, returns ``None`` (the orchestrator
        is running without a prognostic carry).
    updates : dict
        Mapping from field name (``'tke'``, ``'conv_prog'``, ``'gwd_spectrum'``)
        to the updated ``jax.Array`` returned by the sub-physics function.
        Fields absent from the dict are carried over unchanged.

    Returns
    -------
    PhysicsState or None
    """
    if phys_state is None:
        return None
    return PhysicsState(
        tke=updates.get("tke", phys_state.tke),
        conv_prog=updates.get("conv_prog", phys_state.conv_prog),
        gwd_spectrum=updates.get("gwd_spectrum", phys_state.gwd_spectrum),
    )
