"""Advection and refraction tendencies for the spectral wave field.

Adapted from WADEN 46aadc7, ``src/waden/tendencies.py``, for spherical
and Cartesian physical-cell grids. Spectral density is E(f,theta); ray
wavenumber motion is not treated as physical frequency shift in still water.

Axis order is ``(batch, longitude, latitude, direction, frequency)``.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from legoesm.wave.config import WavePhysicsConfig
from legoesm.wave.grid import WaveEnv
from legoesm.wave.physics.dispersion import calculate_depth_dispersion_derivative
from legoesm.wave.physics.flux import get_stencils, quickest_flux

__physics_contract__ = {
    "summary": (
        "Spectral transport: longitude/latitude propagation with QUICKEST, "
        "direction and wavenumber refraction, and the learned replacements for "
        "those two refraction tendencies. Spherical metric terms (r cos(lat), "
        "r) and the WW3 depth-refraction term follow the pinned WADEN reference."
    ),
    "inputs": {
        "spectra": "wave spectral density [m2 s rad-1], (B,Nlon,Nlat,Ndir,Nfreq)",
        "env": "WaveEnv with dispersion fields",
        "depth": "water depth [m]",
        "DdDx, DdDy": "depth gradient [m/m]",
    },
    "outputs": {"tendency": "shape matches spectra on all physical cells"},
    "sign_convention": (
        "Tendency is added to the spectrum; the flux divergence is "
        "FLN_minus - FLN_plus over the grid spacing."),
    "conserves": ["tracer"],
    "differentiable": True,
    "reference": "WADEN 46aadc7 src/waden/tendencies.py; WW3 manual v6.07 §3.3.",
    "idealized_test": (
        "tests/wave/unit/test_waden_contracts.py::"
        "test_nonuniform_frequency_budget_and_positivity"),
}

ArrayFn = Callable[..., jax.Array]
FluxFn = Callable[..., tuple[jax.Array, jax.Array]]

# Sampled network evaluation floor (NeuralSWM sampling variant).  A count,
# not a tunable: it bounds how few columns may contribute to a sampled
# gradient.
_MIN_SAMPLES = 1024

# WADEN's zero-dt fallback in the first-order tendency.  A disabled component
# is skipped by the scheduler; this keeps the arithmetic finite in the
# unselected branch (WADEN 46aadc7 tendencies.py:455).
_REFERENCE_DT_S = 3600.0


def _as_array(value, dtype=None):
    array = jnp.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def _broadcast_like(value, reference: jax.Array) -> jax.Array:
    return jnp.broadcast_to(_as_array(value, reference.dtype), reference.shape)


def _safe_divisor(value, dtype, eps: float) -> jax.Array:
    return jnp.clip(_as_array(value, dtype=dtype), min=eps)


def _broadcast_env_fields(spectra: jax.Array, env: WaveEnv):
    return {
        "cg": _broadcast_like(env.group_velocity, spectra),
        "wavenum": _broadcast_like(env.wavenum, spectra),
        "sin_theta": _broadcast_like(env.sin_theta, spectra),
        "cos_theta": _broadcast_like(env.cos_theta, spectra),
    }


def x_tendency(
    spectra: jax.Array,
    dt: float,
    env: WaveEnv,
    physics: WavePhysicsConfig,
    land_mask: Optional[jax.Array] = None,
    flux_fn: FluxFn = quickest_flux,
    eps: float = 1e-9,
    stencil_mode: Optional[str] = None,
    boundary_mask: Optional[jax.Array] = None,
) -> jax.Array:
    """Longitude propagation tendency."""
    fields = _broadcast_env_fields(spectra, env)
    spec = _broadcast_like(physics.fachfa, spectra) * spectra

    dxgrid = _as_array(env.dlon * env.x_metric, dtype=spectra.dtype)

    lamda = _broadcast_like(physics.facvx, spectra) * fields["cg"] * fields["sin_theta"]
    fln_negx, fln_posx = flux_fn(
        N=spec, Cg=lamda, land_mask=land_mask, Dgrid=dxgrid,
        target_dim=1, dt=dt,
        **({"mode": stencil_mode} if stencil_mode is not None else {}),
        **({"boundary_mask": boundary_mask} if flux_fn is quickest_flux else {}),
    )
    return (fln_negx - fln_posx) / _safe_divisor(dxgrid, spectra.dtype, eps)


def y_tendency(
    spectra: jax.Array,
    dt: float,
    env: WaveEnv,
    physics: WavePhysicsConfig,
    land_mask: Optional[jax.Array] = None,
    flux_fn: FluxFn = quickest_flux,
    eps: float = 1e-9,
    stencil_mode: Optional[str] = None,
    boundary_mask: Optional[jax.Array] = None,
) -> jax.Array:
    """Latitude propagation tendency."""
    fields = _broadcast_env_fields(spectra, env)
    spec = _broadcast_like(physics.fachfa, spectra) * spectra
    dygrid = _as_array(env.dlat * env.y_metric * env.cos_lat, dtype=spectra.dtype)
    phi = (
        _broadcast_like(physics.facvy, spectra)
        * fields["cg"]
        * fields["cos_theta"]
        * _broadcast_like(env.cos_lat, spectra)
    )
    fln_negy, fln_posy = flux_fn(
        N=spec, Cg=phi, land_mask=land_mask, Dgrid=dygrid,
        target_dim=2, dt=dt,
        **({"mode": stencil_mode} if stencil_mode is not None else {}),
        **({"boundary_mask": boundary_mask} if flux_fn is quickest_flux else {}),
    )
    return (fln_negy - fln_posy) / _safe_divisor(dygrid, spectra.dtype, eps)


def theta_transport_velocity(
    spectra: jax.Array,
    env: WaveEnv,
    physics: WavePhysicsConfig,
    depth: jax.Array,
    dd_dx: jax.Array,
    dd_dy: jax.Array,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Directional transport velocity ``VCTH``."""
    del spectra
    cg = _as_array(env.group_velocity)
    wavenum = _as_array(env.wavenum)
    sin_theta = _as_array(env.sin_theta)
    cos_theta = _as_array(env.cos_theta)
    dd_dx_full = _broadcast_like(dd_dx, cg)
    dd_dy_full = _broadcast_like(dd_dy, cg)

    dsdd = dsdd_fn(cg, wavenum, env.freq, depth)
    safe_wavenum = jnp.clip(wavenum, min=eps)
    vcth = -_as_array(physics.facth, dtype=cg.dtype) * (1.0 / safe_wavenum) * (
        dsdd * (cos_theta * dd_dx_full - sin_theta * dd_dy_full)
    )
    return vcth + physics.fdg * cg * sin_theta * env.curvature



def theta_tendency(
    spectra: jax.Array,
    dt: float,
    env: WaveEnv,
    physics: WavePhysicsConfig,
    depth: jax.Array,
    dd_dx: jax.Array,
    dd_dy: jax.Array,
    flux_fn: FluxFn = quickest_flux,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Directional refraction tendency (interior cells)."""
    vcth = theta_transport_velocity(
        spectra, env, physics, depth, dd_dx, dd_dy, dsdd_fn=dsdd_fn, eps=eps)
    fln_negth, fln_posth = flux_fn(
        N=spectra, Cg=env.theta_axis_sign*vcth, Dgrid=env.dth, target_dim=3, dt=dt,
    )
    theta_pro = (fln_negth - fln_posth) / _safe_divisor(env.dth, spectra.dtype, eps)
    return theta_pro


def k_tendency(
    spectra: jax.Array,
    dt: float,
    env: WaveEnv,
    physics: WavePhysicsConfig,
    depth: jax.Array,
    dd_dx: jax.Array,
    dd_dy: jax.Array,
) -> jax.Array:
    """Zero physical frequency tendency for E(f,theta) in stationary water.

    E(f,theta) has no physical frequency shift for stationary water/depth.
    Ray k-dot is a separate diagnostic; it does not transport E in f bins.
    """
    return jnp.zeros_like(spectra)



def _neural_inputs(
    spectra: jax.Array,
    env: WaveEnv,
    depth: jax.Array,
    dd_dx: jax.Array,
    dd_dy: jax.Array,
    dsdd_fn: ArrayFn,
    eps: float,
):
    del spectra
    wavenum = _as_array(env.wavenum)
    cg = _as_array(env.group_velocity)
    dd_dx_full = _broadcast_like(dd_dx, wavenum)
    dd_dy_full = _broadcast_like(dd_dy, wavenum)
    dsdd = dsdd_fn(cg, wavenum, env.freq, depth)
    safe_wavenum = jnp.clip(wavenum, min=eps)
    b, h, w, d, f = safe_wavenum.shape
    inputs = jnp.concatenate(
        [
            (safe_wavenum * dd_dx_full).reshape(b * h * w, d * f),
            (safe_wavenum * dd_dy_full).reshape(b * h * w, d * f),
            dsdd.reshape(b * h * w, d * f),
        ],
        axis=-1,
    )
    return inputs, {
        "wavenum": wavenum,
        "cg": cg,
        "sin_theta": _as_array(env.sin_theta),
        "cos_theta": _as_array(env.cos_theta),
    }, (b, h, w, d, f)


def _run_network_blocked(
    network_fn: ArrayFn,
    inputs: jax.Array,
    num_intra_batch_block: Optional[int],
) -> jax.Array:
    if num_intra_batch_block is None or num_intra_batch_block <= 1:
        return jnp.asarray(network_fn(inputs))

    total_points = inputs.shape[0]
    blocks = int(num_intra_batch_block)
    block_size = (total_points + blocks - 1) // blocks
    outputs = []
    for start in range(0, total_points, block_size):
        end = min(start + block_size, total_points)
        outputs.append(jnp.asarray(network_fn(inputs[start:end])))
    return jnp.concatenate(outputs, axis=0)


def _build_sample_mask(
    valid_mask: jax.Array,
    rng_key,
    sample_ratio: float,
    min_samples: int,
) -> jax.Array:
    total_points = valid_mask.shape[0]
    valid_count = jnp.sum(valid_mask.astype(jnp.int32))
    sample_count = jnp.maximum(
        jnp.asarray(min_samples, dtype=jnp.int32),
        jnp.floor(valid_count.astype(jnp.float32) * sample_ratio).astype(jnp.int32),
    )
    sample_count = jnp.minimum(sample_count, valid_count)

    perm = jax.random.permutation(rng_key, total_points)
    valid_perm = valid_mask[perm]
    sampled_perm = valid_perm & (jnp.cumsum(valid_perm.astype(jnp.int32)) <= sample_count)
    return jnp.zeros_like(valid_mask).at[perm].set(sampled_perm)


def _sampled_network_output(
    inputs: jax.Array,
    depth: jax.Array,
    network_fn: ArrayFn,
    sample_ratio: Optional[float],
    min_samples: int,
    num_intra_batch_block: Optional[int],
    *,
    rng_key: jax.Array | None,
    default_rng_seed: int,
) -> jax.Array:
    full_output = _run_network_blocked(network_fn, inputs, num_intra_batch_block)
    if sample_ratio is None or sample_ratio <= 0:
        return full_output

    if rng_key is None:
        rng_key = jax.random.key(default_rng_seed)

    valid_mask = jnp.reshape(depth > 0, (-1,))
    sample_mask = _build_sample_mask(valid_mask, rng_key, sample_ratio, min_samples)
    sample_mask = jnp.reshape(sample_mask, (-1, 1))
    return jnp.where(sample_mask, full_output, jax.lax.stop_gradient(full_output))


def neural_theta_transport_velocity(
    spectra: jax.Array,
    env: WaveEnv,
    depth: jax.Array,
    dd_dx: jax.Array,
    dd_dy: jax.Array,
    network_fn: ArrayFn,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Learned directional transport velocity."""
    inputs, fields, shape = _neural_inputs(spectra, env, depth, dd_dx, dd_dy, dsdd_fn, eps)
    b, h, w, d, f = shape
    hat_a = jnp.asarray(network_fn(inputs)).reshape(b, h, w, d, f, 2)
    vcth = hat_a[..., 0] * fields["cos_theta"] - hat_a[..., 1] * fields["sin_theta"]
    return jnp.tanh(vcth)


def neural_k_transport_velocity(
    spectra: jax.Array,
    env: WaveEnv,
    depth: jax.Array,
    dd_dx: jax.Array,
    dd_dy: jax.Array,
    network_fn: ArrayFn,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Learned wavenumber transport velocity."""
    inputs, fields, shape = _neural_inputs(spectra, env, depth, dd_dx, dd_dy, dsdd_fn, eps)
    b, h, w, d, f = shape
    hat_a = jnp.asarray(network_fn(inputs)).reshape(b, h, w, d, f, 2)
    vcwn = hat_a[..., 0] * fields["cos_theta"] + hat_a[..., 1] * fields["sin_theta"]
    return jnp.tanh(vcwn)


def sampling_neural_theta_transport_velocity(
    spectra: jax.Array,
    env: WaveEnv,
    depth: jax.Array,
    dd_dx: jax.Array,
    dd_dy: jax.Array,
    network_fn: ArrayFn,
    sample_ratio: Optional[float] = None,
    min_samples: int = _MIN_SAMPLES,
    num_intra_batch_block: Optional[int] = None,
    rng_key: jax.Array | None = None,
    default_rng_seed: int = 0,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Sampled learned directional transport velocity (stop-gradient off-sample)."""
    inputs, fields, shape = _neural_inputs(spectra, env, depth, dd_dx, dd_dy, dsdd_fn, eps)
    b, h, w, d, f = shape
    hat_a_full = _sampled_network_output(
        inputs, depth, network_fn, sample_ratio, min_samples,
        num_intra_batch_block, rng_key=rng_key, default_rng_seed=default_rng_seed,
    )
    hat_a = hat_a_full.reshape(b, h, w, d, f, 2)
    vcth = hat_a[..., 0] * fields["cos_theta"] - hat_a[..., 1] * fields["sin_theta"]
    return jnp.tanh(vcth)


def sampling_neural_k_transport_velocity(
    spectra: jax.Array,
    env: WaveEnv,
    depth: jax.Array,
    dd_dx: jax.Array,
    dd_dy: jax.Array,
    network_fn: ArrayFn,
    sample_ratio: Optional[float] = None,
    min_samples: int = _MIN_SAMPLES,
    num_intra_batch_block: Optional[int] = None,
    rng_key: jax.Array | None = None,
    default_rng_seed: int = 0,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
) -> jax.Array:
    """Sampled learned wavenumber transport velocity (stop-gradient off-sample)."""
    inputs, fields, shape = _neural_inputs(spectra, env, depth, dd_dx, dd_dy, dsdd_fn, eps)
    b, h, w, d, f = shape
    hat_a_full = _sampled_network_output(
        inputs, depth, network_fn, sample_ratio, min_samples,
        num_intra_batch_block, rng_key=rng_key, default_rng_seed=default_rng_seed,
    )
    hat_a = hat_a_full.reshape(b, h, w, d, f, 2)
    vcwn = hat_a[..., 0] * fields["cos_theta"] + hat_a[..., 1] * fields["sin_theta"]
    return jnp.tanh(vcwn)


def _first_order_tendency(
    spectra: jax.Array,
    transport_velocity: jax.Array,
    dt: float,
    target_dim: int,
    mode: str,
) -> jax.Array:
    N_im1, N_i = get_stencils(spectra, target_dim=target_dim, pad_left=1, pad_right=0, mode=mode)
    is_positive = transport_velocity >= 0
    N_upwind = jnp.where(is_positive, N_im1, N_i)
    dt_array = jnp.asarray(dt, dtype=spectra.dtype)
    safe_dt = jnp.where(
        dt_array == 0,
        jnp.asarray(_REFERENCE_DT_S, dtype=spectra.dtype),
        dt_array)
    fln_neg = transport_velocity * N_upwind / safe_dt
    fln_pos = get_stencils(fln_neg, target_dim=target_dim, pad_left=0, pad_right=1, mode=mode)[1]
    return fln_neg - fln_pos


def frequency_flux_tendency(spectra, coefficient, dt, widths):
    """WADEN shared E*df face fluxes with closed frequency endpoints."""
    mass = spectra * widths
    speed = coefficient[..., 1:]
    donor = jnp.where(speed >= 0, mass[..., :-1], mass[..., 1:])
    reference_dt = jnp.asarray(dt, dtype=spectra.dtype)
    reference_dt = jnp.where(
        reference_dt == 0,
        jnp.asarray(_REFERENCE_DT_S, dtype=spectra.dtype),
        reference_dt)
    flux = speed * donor / reference_dt
    padding = [(0, 0)] * (spectra.ndim-1)
    left = jnp.pad(flux, padding+[(1, 0)])
    right = jnp.pad(flux, padding+[(0, 1)])
    return (left-right)/widths


def neural_theta_tendency(
    spectra: jax.Array,
    dt: float,
    env: WaveEnv,
    depth: jax.Array,
    dd_dx: jax.Array,
    dd_dy: jax.Array,
    network_fn: ArrayFn,
    physics=None,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
    neural_dt: Optional[float] = None,
) -> jax.Array:
    """Learned directional tendency, first-order upwind.

    ``neural_dt`` is WADEN's ``neural_dt_th``: the full configured component
    step, NOT the substep the analytic operator receives.  A scheduler that
    divides the analytic time steps by its substep count must still pass the
    full step here, or the learned transfer comes out one substep too strong.
    When omitted, ``dt`` is used.
    """
    del eps, physics
    effective_dt = dt if neural_dt is None else neural_dt
    velocity = neural_theta_transport_velocity(
        spectra, env, depth, dd_dx, dd_dy, network_fn=network_fn, dsdd_fn=dsdd_fn,
    )
    theta_pro = _first_order_tendency(
        spectra, velocity, dt=effective_dt, target_dim=3, mode="circular")
    return theta_pro


def neural_k_tendency(
    spectra: jax.Array,
    dt: float,
    env: WaveEnv,
    depth: jax.Array,
    dd_dx: jax.Array,
    dd_dy: jax.Array,
    network_fn: ArrayFn,
    physics=None,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
    neural_dt: Optional[float] = None,
) -> jax.Array:
    """Learned wavenumber tendency, first-order upwind.

    ``neural_dt`` is WADEN's ``neural_dt_k``: the full configured component
    step, NOT the analytical substep.  When omitted, ``dt`` is used.
    """
    del eps, physics
    effective_dt = dt if neural_dt is None else neural_dt
    velocity = neural_k_transport_velocity(
        spectra, env, depth, dd_dx, dd_dy, network_fn=network_fn, dsdd_fn=dsdd_fn,
    )
    return frequency_flux_tendency(spectra, velocity, effective_dt, env.quadrature_df)


def sampling_neural_theta_tendency(
    spectra: jax.Array,
    dt: float,
    env: WaveEnv,
    depth: jax.Array,
    dd_dx: jax.Array,
    dd_dy: jax.Array,
    network_fn: ArrayFn,
    sample_ratio: Optional[float] = None,
    min_samples: int = _MIN_SAMPLES,
    num_intra_batch_block: Optional[int] = None,
    rng_key: jax.Array | None = None,
    default_rng_seed: int = 0,
    physics=None,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
    neural_dt: Optional[float] = None,
) -> jax.Array:
    """Sampled learned directional tendency, first-order upwind.

    ``neural_dt`` carries WADEN's full component step; see
    :func:`neural_theta_tendency`.
    """
    del eps, physics
    effective_dt = dt if neural_dt is None else neural_dt
    velocity = sampling_neural_theta_transport_velocity(
        spectra, env, depth, dd_dx, dd_dy, network_fn=network_fn,
        sample_ratio=sample_ratio, min_samples=min_samples,
        num_intra_batch_block=num_intra_batch_block, rng_key=rng_key,
        default_rng_seed=default_rng_seed, dsdd_fn=dsdd_fn,
    )
    theta_pro = _first_order_tendency(
        spectra, velocity, dt=effective_dt, target_dim=3, mode="circular")
    return theta_pro


def sampling_neural_k_tendency(
    spectra: jax.Array,
    dt: float,
    env: WaveEnv,
    depth: jax.Array,
    dd_dx: jax.Array,
    dd_dy: jax.Array,
    network_fn: ArrayFn,
    sample_ratio: Optional[float] = None,
    min_samples: int = _MIN_SAMPLES,
    num_intra_batch_block: Optional[int] = None,
    rng_key: jax.Array | None = None,
    default_rng_seed: int = 0,
    physics=None,
    dsdd_fn: ArrayFn = calculate_depth_dispersion_derivative,
    eps: float = 1e-9,
    neural_dt: Optional[float] = None,
) -> jax.Array:
    """Sampled learned wavenumber tendency, first-order upwind.

    ``neural_dt`` carries WADEN's full component step; see
    :func:`neural_k_tendency`.
    """
    del eps, physics
    effective_dt = dt if neural_dt is None else neural_dt
    velocity = sampling_neural_k_transport_velocity(
        spectra, env, depth, dd_dx, dd_dy, network_fn=network_fn,
        sample_ratio=sample_ratio, min_samples=min_samples,
        num_intra_batch_block=num_intra_batch_block, rng_key=rng_key,
        default_rng_seed=default_rng_seed, dsdd_fn=dsdd_fn,
    )
    return frequency_flux_tendency(spectra, velocity, effective_dt, env.quadrature_df)


__all__ = [
    "frequency_flux_tendency",
    "k_tendency",
    "neural_k_tendency",
    "neural_k_transport_velocity",
    "neural_theta_tendency",
    "neural_theta_transport_velocity",
    "sampling_neural_k_tendency",
    "sampling_neural_k_transport_velocity",
    "sampling_neural_theta_tendency",
    "sampling_neural_theta_transport_velocity",
    "theta_tendency",
    "theta_transport_velocity",
    "x_tendency",
    "y_tendency",
]
