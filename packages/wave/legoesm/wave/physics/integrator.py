"""Operator-splitting integrator for the spectral wave field.

Port of WADEN 46aadc7 ``src/waden/integrator.py``.  Per substep the split
order is::

    theta half -> wavenumber half -> spatial (X/Y, alternating) ->
    optional GSE -> theta half -> wavenumber half -> source term

The whole loop is a ``jax.lax.scan`` over a fixed substep count, so it is
jit-able and differentiable.  ``checkpoint_level`` selects how much of
the step is rematerialised for the backward pass.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from legoesm.wave.config import WaveConfig
from legoesm.wave.grid import WaveEnv, wave_cell_area
from legoesm.wave.physics.boundary import (
    apply_boundary_condition,
    identity_boundary_update,
    update_absorb_boundary,
)
from legoesm.wave.physics.tendencies import (
    k_tendency,
    theta_tendency,
    x_tendency,
    y_tendency,
)

__physics_contract__ = {
    "summary": (
        "Operator-split spectral transport with alternating X/Y order, fixed "
        "substep count from the gcd of the component time steps, optional GSE "
        "spatial averaging, and an explicit source-term update."
    ),
    "inputs": {
        "spectra": "wave spectral density, (B,Nlon,Nlat,Ndir,Nfreq)",
        "stot": "source/sink tendency, same shape",
        "depth": "water depth [m], (B,Nlon,Nlat,1,1)",
        "mask": "True where sea",
        "boundary_condition": "edge spectra; zeros = absorbing",
    },
    "outputs": {"spectra": "updated spectrum after one global step"},
    "sign_convention": "Updates are added to the spectrum; clipped at zero.",
    "conserves": ["none"],
    "differentiable": True,
    "reference": "WADEN 46aadc7 src/waden/integrator.py; WW3 manual v6.07 §3.3.",
    "idealized_test": (
        "tests/wave/unit/test_integrator.py::test_simultaneous_spatial_events_count_once"),
}

TendencyFn = Callable[..., jax.Array]
StotAdvanceFn = Callable[..., jax.Array]
GSEAlleviationFn = Callable[..., jax.Array]

CHECKPOINT_NONE = "none"
CHECKPOINT_PER_STEP = "per_step"
CHECKPOINT_PER_PHASE = "per_phase"
_VALID_CHECKPOINT_LEVELS = frozenset({CHECKPOINT_NONE, CHECKPOINT_PER_STEP, CHECKPOINT_PER_PHASE})

# The component steps are exact integers; the alignment product is evaluated
# in the state dtype, so compare against zero with a small tolerance.
_STEP_ALIGN_TOL = 1e-5


def _as_grid_field(value, dtype) -> jax.Array:
    array = jnp.asarray(value, dtype=dtype)
    while array.ndim < 5:
        array = jnp.expand_dims(array, axis=-1)
    if array.shape[-2] != 1 or array.shape[-1] != 1:
        array = array[..., :1, :1]
    return array


def compute_depth_gradients(depth: jax.Array, env: WaveEnv) -> tuple[jax.Array, jax.Array]:
    """Spherical depth gradients, zeroed on land."""
    depth_3d = jnp.squeeze(jnp.squeeze(depth, axis=-1), axis=-1)

    depthx = jnp.pad(depth_3d, ((0, 0), (1, 1), (0, 0)), mode="wrap")
    depthy = jnp.pad(depth_3d, ((0, 0), (0, 0), (1, 1)), mode="edge")

    dlon = _as_grid_field(env.dlon, depth.dtype)
    dlat = _as_grid_field(env.dlat, depth.dtype)

    dd_dx = 0.5 * (
        (depthx[:, 2:, :] - depthx[:, 1:-1, :])
        + (depthx[:, 1:-1, :] - depthx[:, :-2, :]))
    dd_dy = 0.5 * (
        (depthy[:, :, 2:] - depthy[:, :, 1:-1])
        + (depthy[:, :, 1:-1] - depthy[:, :, :-2]))

    dd_dx = jnp.expand_dims(jnp.expand_dims(dd_dx, axis=-1), axis=-1) / dlon
    dd_dy = jnp.expand_dims(jnp.expand_dims(dd_dy, axis=-1), axis=-1) / dlat

    dd_dx = dd_dx / env.x_metric
    dd_dy = dd_dy / env.y_metric

    land_mask = depth > 0
    return dd_dx * land_mask, dd_dy * land_mask


def _sanitize_physics_fields(env: WaveEnv, land_mask: jax.Array) -> WaveEnv:
    mask = jnp.asarray(land_mask, dtype=bool)
    while mask.ndim < 5:
        mask = mask[..., None]
    expanded = jnp.broadcast_to(mask, env.wavenum.shape)
    return env._replace(
        wavenum=jnp.where(expanded, env.wavenum, 0.0),
        group_velocity=jnp.where(expanded, env.group_velocity, 0.0),
    )


def _compute_should_mask(num_steps: int, dt_min: int, dt: float, dtype) -> jax.Array:
    if dt <= 0:
        return jnp.zeros((num_steps,), dtype=bool)
    step_indices = jnp.arange(1, num_steps + 1, dtype=dtype)
    return jnp.mod(
        step_indices * jnp.asarray(dt_min, dtype=dtype),
        jnp.asarray(dt, dtype=dtype)) < _STEP_ALIGN_TOL


def _update_interior(spectra: jax.Array, update: jax.Array, project):
    if update.shape == spectra.shape:
        updated = spectra + update
    else:
        updated = spectra.at[:, 1:-1, 1:-1, :, :].add(update)
    return project(updated)


def simple_stot_integrator(
    spectra: jax.Array,
    stot: jax.Array,
    dt: float,
    mask: jax.Array,
) -> jax.Array:
    """Explicit SI source update: ``spectra + stot * dt``.

    The 0.01 unit bridge from the NeuralSWM-trained source belongs to the
    model adapter, not to this kernel (WADEN integrator.py:143).
    """
    update_term = stot * jnp.asarray(dt, dtype=spectra.dtype)
    updated_spectra = spectra + update_term
    mask_expanded = jnp.expand_dims(jnp.expand_dims(mask.astype(bool), axis=-1), axis=-1)
    mask_expanded = jnp.broadcast_to(mask_expanded, spectra.shape)
    return jnp.where(mask_expanded, updated_spectra, spectra)


def _phase_spectral_half_step(
    spectra: jax.Array,
    should_integrate_th: jax.Array,
    should_integrate_k: jax.Array,
    dt_th: float,
    dt_k: float,
    th_tendency_fn: TendencyFn,
    k_tendency_fn: TendencyFn,
    env: WaveEnv,
    depth: jax.Array,
    dd_dx: jax.Array,
    dd_dy: jax.Array,
    physics,
    boundary_condition, mask, apply_boundary_func, project,
) -> jax.Array:
    spectra = apply_boundary_func(spectra, boundary_condition, mask)
    th_tend = th_tendency_fn(
        spectra, dt=dt_th, env=env, depth=depth, dd_dx=dd_dx, dd_dy=dd_dy,
        physics=physics,
    )
    update_th = jnp.where(should_integrate_th, 0.5 * dt_th * th_tend, jnp.zeros_like(th_tend))
    spectra, clip_th = _update_interior(spectra, update_th, project)

    k_tend = k_tendency_fn(
        spectra, dt=dt_k, env=env, depth=depth, dd_dx=dd_dx, dd_dy=dd_dy,
        physics=physics,
    )
    update_k = jnp.where(should_integrate_k, 0.5 * dt_k * k_tend, jnp.zeros_like(k_tend))
    spectra, clip_k = _update_interior(spectra, update_k, project)
    return spectra, clip_th+clip_k


def _phase_spatial_single_direction(
    spectra: jax.Array,
    should_integrate_first: jax.Array,
    should_integrate_second: jax.Array,
    dt_first: float,
    dt_second: float,
    first_tendency_fn: TendencyFn,
    second_tendency_fn: TendencyFn,
    env: WaveEnv,
    physics,
    land_mask: jax.Array,
    boundary_condition: jax.Array,
    mask: jax.Array,
    boundary_update_func, apply_boundary_func, project,
) -> jax.Array:
    spectra = apply_boundary_func(spectra, boundary_condition, mask)
    tend_1 = first_tendency_fn(spectra, dt=dt_first, env=env, land_mask=land_mask, physics=physics)
    update_1 = jnp.where(should_integrate_first, dt_first * tend_1, jnp.zeros_like(tend_1))
    spectra, clip_first = project(spectra+update_1)
    spectra = boundary_update_func(spectra)

    spectra = apply_boundary_func(spectra, boundary_condition, mask)
    tend_2 = second_tendency_fn(
        spectra, dt=dt_second, env=env, land_mask=land_mask, physics=physics)
    update_2 = jnp.where(should_integrate_second, dt_second * tend_2, jnp.zeros_like(tend_2))
    spectra, clip_second = project(spectra+update_2)
    return boundary_update_func(spectra), clip_first+clip_second


def _phase_spatial_propagation(
    spectra: jax.Array,
    yfirst_current: jax.Array,
    should_integrate_x: jax.Array,
    should_integrate_y: jax.Array,
    dt_x: float,
    dt_y: float,
    x_tendency_fn: TendencyFn,
    y_tendency_fn: TendencyFn,
    env: WaveEnv,
    physics,
    land_mask: jax.Array,
    boundary_condition: jax.Array,
    mask: jax.Array,
    boundary_update_func, apply_boundary_func, project,
) -> jax.Array:
    common = dict(
        env=env,
        physics=physics,
        land_mask=land_mask,
        boundary_condition=boundary_condition,
        mask=mask, boundary_update_func=boundary_update_func,
        apply_boundary_func=apply_boundary_func, project=project,
    )
    spectra_y_first = _phase_spatial_single_direction(
        spectra,
        should_integrate_first=should_integrate_y,
        should_integrate_second=should_integrate_x,
        dt_first=dt_y, dt_second=dt_x,
        first_tendency_fn=y_tendency_fn,
        second_tendency_fn=x_tendency_fn,
        **common,
    )
    spectra_x_first = _phase_spatial_single_direction(
        spectra,
        should_integrate_first=should_integrate_x,
        should_integrate_second=should_integrate_y,
        dt_first=dt_x, dt_second=dt_y,
        first_tendency_fn=x_tendency_fn,
        second_tendency_fn=y_tendency_fn,
        **common,
    )
    return jax.tree.map(lambda y,x: jnp.where(yfirst_current,y,x), spectra_y_first, spectra_x_first)


def _phase_source_integration(
    spectra: jax.Array,
    should_integrate_stot: jax.Array,
    stot: jax.Array,
    dt_stot: float,
    mask: jax.Array,
    boundary_condition: jax.Array,
    stot_advance: StotAdvanceFn,
    boundary_update_func, apply_boundary_func, project,
) -> jax.Array:
    def _integrate(s: jax.Array) -> jax.Array:
        s = boundary_update_func(s)
        s = apply_boundary_func(s, boundary_condition, mask)
        s = stot_advance(s, stot, dt=dt_stot, mask=mask)
        return project(s)

    return jax.lax.cond(
        should_integrate_stot, _integrate,
        lambda s: (s, project(jnp.zeros_like(s))[1]), spectra)


def _phase_gse_averaging(
    spectra: jax.Array,
    should_integrate_gse: jax.Array,
    dt_gse: float,
    env: WaveEnv,
    mask: jax.Array,
    boundary_condition: jax.Array,
    gse_alleviation: GSEAlleviationFn,
    boundary_update_func, apply_boundary_func, project,
) -> jax.Array:
    def _integrate(s: jax.Array) -> jax.Array:
        s = gse_alleviation(s, env=env, mask=mask, dt=dt_gse)
        s, correction = project(s)
        s = boundary_update_func(s)
        return apply_boundary_func(s, boundary_condition, mask), correction

    return jax.lax.cond(
        should_integrate_gse, _integrate,
        lambda s: (s, project(jnp.zeros_like(s))[1]), spectra)


def prepare_integration_schedule(
    dt_global: int,
    dt_x: int,
    dt_y: int,
    dt_th: int,
    dt_k: int,
    dt_stot: int,
    dtype,
    yfirst_initial: bool,
    dt_gse: int = 0,
    spatial_phase=0,
) -> dict:
    """Substep count and per-component execution masks."""
    component_dts = (dt_x, dt_y, dt_th, dt_k, dt_stot, dt_gse)
    if dt_global <= 0 or any(
            dt < 0 or int(dt) != dt or (dt and dt_global % dt)
            for dt in component_dts):
        raise ValueError("component integer time steps must divide the positive global interval")
    active_dts = [int(dt) for dt in (dt_x, dt_y, dt_th, dt_k, dt_stot, dt_gse) if dt > 0]
    if not active_dts:
        return {"num_steps": 0}

    dt_min = active_dts[0]
    for dt_val in active_dts[1:]:
        dt_min = math.gcd(dt_min, dt_val)
    num_steps = int(dt_global // dt_min)

    should_x = _compute_should_mask(num_steps, dt_min, dt_x, dtype)
    should_y = _compute_should_mask(num_steps, dt_min, dt_y, dtype)
    should_th = _compute_should_mask(num_steps, dt_min, dt_th, dtype)
    should_k = _compute_should_mask(num_steps, dt_min, dt_k, dtype)
    should_stot = _compute_should_mask(num_steps, dt_min, dt_stot, dtype)
    should_gse = _compute_should_mask(num_steps, dt_min, dt_gse, dtype)

    spatial_events = should_x | should_y
    prior_events = jnp.concatenate([
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.cumsum(spatial_events[:-1].astype(jnp.int32)),
    ])
    yfirst_per_step = ((prior_events + spatial_phase) % 2 == 0) == yfirst_initial


    return {
        "num_steps": num_steps,
        "should_integrate_x": should_x,
        "should_integrate_y": should_y,
        "should_integrate_th": should_th,
        "should_integrate_k": should_k,
        "should_integrate_stot": should_stot,
        "should_integrate_gse": should_gse,
        "yfirst_per_step": yfirst_per_step,
    }


def model_advance(
    spectra: jax.Array,
    stot: jax.Array,
    depth: jax.Array,
    env: WaveEnv,
    config: WaveConfig,
    mask: jax.Array,
    boundary_condition: jax.Array,
    x_tendency_fn: TendencyFn = x_tendency,
    y_tendency_fn: TendencyFn = y_tendency,
    th_tendency_fn: TendencyFn = theta_tendency,
    k_tendency_fn: TendencyFn = k_tendency,
    stot_advance: StotAdvanceFn = simple_stot_integrator,
    gse_alleviation: Optional[GSEAlleviationFn] = None,
    boundary_mask: Optional[jax.Array] = None,
    spatial_phase=0,
    audit_clipping: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Integrate the spectrum one global step.

    The caller is responsible for passing an ``env`` whose dispersion
    fields match ``depth`` (see :func:`legoesm.wave.grid.with_dispersion`).
    When GSE is enabled, its footprint preflight
    (:func:`legoesm.wave.physics.gse.GSESpatialAveraging.validate_displacement`)
    must have run outside JIT before this call; it needs concrete arrays and
    is not repeated here.

    ``spatial_phase`` must be the phase returned by the previous call (or 0
    for a fresh state); it selects which of X/Y runs first.  With
    ``audit_clipping=True`` the return is ``(spectra, clipping_mass)``;
    otherwise it is the spectrum alone.
    """
    config.validate()
    checkpoint_level = config.physics.checkpoint_level
    if checkpoint_level not in _VALID_CHECKPOINT_LEVELS:
        raise ValueError(
            f"checkpoint_level must be one of {sorted(_VALID_CHECKPOINT_LEVELS)}, "
            f"got {checkpoint_level!r}")

    time = config.time
    dt_global = time.dt_global_s
    dt_x = time.dt_x_s
    dt_y = time.dt_y_s
    dt_th = time.dt_th_s
    dt_k = time.dt_k_s
    dt_stot = time.dt_stot_s
    dt_gse = time.dt_gse_s

    gse_enabled = (
        gse_alleviation is not None
        and getattr(gse_alleviation, "enabled", False)
        and dt_gse > 0
    )
    active_dt_gse = dt_gse if gse_enabled else 0

    if boundary_mask is None:
        boundary_mask = jnp.sum(boundary_condition, axis=(-2,-1)) > 0
    land_mask = (depth > 0) & (mask | boundary_mask)[..., None, None]
    mask = mask & ~boundary_mask
    apply_boundary_func = partial(apply_boundary_condition, boundary_mask=boundary_mask)
    boundary_update_func = (
        identity_boundary_update
        if config.grid.boundary_mode == "physical" else update_absorb_boundary)
    audit_dtype = jnp.float64 if audit_clipping and jax.config.x64_enabled else spectra.dtype
    zero_clipping = jnp.zeros((spectra.shape[0],), dtype=audit_dtype)
    # WADEN's clipping wet mask includes depth > 0; the caller's mask alone can
    # disagree with it.
    clipping_mask = mask & (depth[..., 0, 0] > 0)

    def project(value):
        value = apply_boundary_func(value, boundary_condition, mask)
        if audit_clipping:
            weight = (wave_cell_area(env)*env.quadrature_df*env.dth).astype(audit_dtype)
            negative = jnp.where(
                clipping_mask[..., None, None],
                jnp.maximum(-value, 0), 0).astype(audit_dtype)
            correction = jnp.sum(negative*weight, axis=(1,2,3,4))
        else:
            correction = zero_clipping
        return jnp.maximum(value,0), correction

    x_tendency_fn = partial(
        x_tendency_fn, stencil_mode=config.grid.x_stencil,
        boundary_mask=boundary_mask)
    y_tendency_fn = partial(
        y_tendency_fn, stencil_mode=config.grid.y_stencil,
        boundary_mask=boundary_mask)
    env = _sanitize_physics_fields(env, land_mask)
    dd_dx, dd_dy = compute_depth_gradients(depth, env)

    schedule = prepare_integration_schedule(
        dt_global, dt_x, dt_y, dt_th, dt_k, dt_stot,
        spectra.dtype, config.physics.yfirst, dt_gse=active_dt_gse, spatial_phase=spatial_phase,
    )
    num_steps = schedule["num_steps"]
    if num_steps == 0:
        return (spectra, zero_clipping) if audit_clipping else spectra

    def _spectral_half(spectra_in: jax.Array, step_idx: jax.Array) -> jax.Array:
        return _phase_spectral_half_step(
            spectra_in,
            should_integrate_th=schedule["should_integrate_th"][step_idx],
            should_integrate_k=schedule["should_integrate_k"][step_idx],
            dt_th=dt_th, dt_k=dt_k,
            th_tendency_fn=th_tendency_fn, k_tendency_fn=k_tendency_fn,
            env=env, depth=depth, dd_dx=dd_dx, dd_dy=dd_dy,
            physics=config.physics, boundary_condition=boundary_condition,
            mask=mask, apply_boundary_func=apply_boundary_func, project=project,
        )

    def _spatial(spectra_in: jax.Array, step_idx: jax.Array) -> jax.Array:
        return _phase_spatial_propagation(
            spectra_in,
            yfirst_current=schedule["yfirst_per_step"][step_idx],
            should_integrate_x=schedule["should_integrate_x"][step_idx],
            should_integrate_y=schedule["should_integrate_y"][step_idx],
            dt_x=dt_x, dt_y=dt_y,
            x_tendency_fn=x_tendency_fn, y_tendency_fn=y_tendency_fn,
            env=env, physics=config.physics, land_mask=land_mask,
            boundary_condition=boundary_condition, mask=mask,
            boundary_update_func=boundary_update_func,
            apply_boundary_func=apply_boundary_func, project=project,
        )

    def _source(spectra_in: jax.Array, step_idx: jax.Array) -> jax.Array:
        return _phase_source_integration(
            spectra_in,
            should_integrate_stot=schedule["should_integrate_stot"][step_idx],
            stot=stot, dt_stot=dt_stot, mask=mask,
            boundary_condition=boundary_condition, stot_advance=stot_advance,
            boundary_update_func=boundary_update_func,
            apply_boundary_func=apply_boundary_func, project=project,
        )

    if gse_enabled:
        def _gse(spectra_in: jax.Array, step_idx: jax.Array) -> jax.Array:
            return _phase_gse_averaging(
                spectra_in,
                should_integrate_gse=schedule["should_integrate_gse"][step_idx],
                dt_gse=dt_gse, env=env, mask=mask,
                boundary_condition=boundary_condition,
                gse_alleviation=gse_alleviation,
                boundary_update_func=boundary_update_func,
                apply_boundary_func=apply_boundary_func, project=project,
            )
    else:
        def _gse(spectra_in: jax.Array, step_idx: jax.Array) -> jax.Array:
            del step_idx
            return spectra_in, zero_clipping

    if checkpoint_level == CHECKPOINT_PER_PHASE:
        _spectral_half = jax.checkpoint(_spectral_half)
        _spatial = jax.checkpoint(_spatial)
        _gse = jax.checkpoint(_gse)
        _source = jax.checkpoint(_source)

    def _step_body(carry, step_idx):
        s, clipped = carry
        s, c1 = _spectral_half(s, step_idx)
        s, c2 = _spatial(s, step_idx)
        s, c3 = _gse(s, step_idx)
        s, c4 = _spectral_half(s, step_idx)
        s, c5 = _source(s, step_idx)
        return (s,clipped+c1+c2+c3+c4+c5), None

    if checkpoint_level == CHECKPOINT_PER_STEP:
        _step_body = jax.checkpoint(_step_body)

    step_indices = jnp.arange(num_steps, dtype=jnp.int32)
    (spectra, clipped), _ = jax.lax.scan(_step_body, (spectra,zero_clipping), step_indices)

    spectra = boundary_update_func(spectra)
    spectra = apply_boundary_func(spectra, boundary_condition, mask)
    spectra, final_clip = project(spectra)
    return (spectra, clipped+final_clip) if audit_clipping else spectra


__all__ = [
    "CHECKPOINT_NONE",
    "CHECKPOINT_PER_PHASE",
    "CHECKPOINT_PER_STEP",
    "compute_depth_gradients",
    "model_advance",
    "prepare_integration_schedule",
    "simple_stot_integrator",
    "spatial_event_count",
]


def spatial_event_count(time) -> int:
    """WADEN count of distinct x-or-y events in one global interval."""
    x, y, duration = time.dt_x_s, time.dt_y_s, time.dt_global_s
    both = duration // math.lcm(x,y) if x and y else 0
    return (duration//x if x else 0)+(duration//y if y else 0)-both
