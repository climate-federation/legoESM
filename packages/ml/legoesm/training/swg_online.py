"""Differentiable online-learning pieces for the NEMO-faithful SWG card.

This module owns the experimental training seam, not the ocean card.  A model
produces one F-point volume-transport tendency potential; the certified SWG
stage owns its matched rotated gradient.  The deterministic and stochastic
losses share one ensemble-shaped balanced-field interface.
"""
from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from legoesm.grids.operators_latlon_cgrid import compute_vertex_mask
from legoesm.ml.training import TrainingConfig, create_optimizer
from legoesm.ocean.diagnostics import relative_vorticity_cell_centre
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
from legoesm.ocean.dynamics.swe_rk3 import swe_rk3_step
from legoesm.ocean.fidelity.nemo_testcase_recipe import swg_surface_wind
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.training.inert_params import assert_no_inert

SECONDS_PER_HOUR = 3_600
ADMITTED_WINDOWS_DAYS = (1, 2, 4, 8)


class SWGNetworkSpec(NamedTuple):
    """Manifest-required CNN choices; no scientific choice is implicit."""

    hidden_channels: int
    hidden_layers: int
    kernel_size: int
    ssh_scale_m: float
    velocity_scale_m_s: float
    potential_scale_m3_s2: float


class SWGClosureGeometry(eqx.Module):
    """Non-trainable dg50 geometry consumed by the closure and balanced loss."""

    tmask: jax.Array
    umask: jax.Array
    vmask: jax.Array
    vertex_interior: jax.Array
    area_t_m2: jax.Array
    metric_u_m: jax.Array
    metric_v_m: jax.Array
    sine_y: jax.Array
    sine_x: jax.Array
    poisson_eigenvalue_s_m2: jax.Array
    grid: object
    wet_bounds: tuple[int, int, int, int] = eqx.field(static=True)
    spacing_m: float = eqx.field(static=True)


class BalancedTrajectory(NamedTuple):
    """Hourly balanced fields, all on wet T points."""

    relative_vorticity_s1: jax.Array
    rotational_u_m_s: jax.Array
    rotational_v_m_s: jax.Array


class BalancedNormalizers(NamedTuple):
    """Training-split variances, content-pinned by the run manifest."""

    relative_vorticity_variance_s2: float
    rotational_u_variance_m2_s2: float
    rotational_v_variance_m2_s2: float


def build_closure_geometry(card) -> SWGClosureGeometry:
    """Build immutable closure geometry from one resolved SWG card."""
    tmask = np.asarray(card.recipe.land_mask, dtype=bool)
    u_full, v_full = compute_face_masks(card.recipe.land_mask, card.recipe.grid)
    umask = np.asarray(u_full[:, 1:], dtype=bool)
    vmask = np.asarray(v_full[1:, :], dtype=bool)
    vertex_full = np.asarray(
        compute_vertex_mask(card.recipe.land_mask, grid=card.recipe.grid),
        dtype=bool)
    interior = vertex_full[1:, 1:]
    jj, ii = np.where(tmask)
    bounds = int(jj.min()), int(jj.max()), int(ii.min()), int(ii.max())
    rectangle = np.zeros_like(tmask)
    rectangle[bounds[0]:bounds[1] + 1, bounds[2]:bounds[3] + 1] = True
    if not np.array_equal(rectangle, tmask):
        raise ValueError("SWG online closure requires one rectangular wet basin")
    ny = bounds[1] - bounds[0] + 1
    nx = bounds[3] - bounds[2] + 1
    y = np.arange(1, ny + 1, dtype=np.float64)
    x = np.arange(1, nx + 1, dtype=np.float64)
    sine_y = np.sqrt(2.0 / (ny + 1)) * np.sin(
        np.pi * y[:, None] * y[None, :] / (ny + 1))
    sine_x = np.sqrt(2.0 / (nx + 1)) * np.sin(
        np.pi * x[:, None] * x[None, :] / (nx + 1))
    spacing = float(np.sqrt(np.asarray(
        card.recipe.z_coord.nemo_e1e2t, dtype=np.float64)[tmask][0]))
    ky = np.arange(1, ny + 1, dtype=np.float64)
    kx = np.arange(1, nx + 1, dtype=np.float64)
    eigen = (-4.0 / spacing**2) * (
        np.sin(np.pi * ky[:, None] / (2 * (ny + 1))) ** 2
        + np.sin(np.pi * kx[None, :] / (2 * (nx + 1))) ** 2)
    return SWGClosureGeometry(
        tmask=jnp.asarray(tmask), umask=jnp.asarray(umask),
        vmask=jnp.asarray(vmask), vertex_interior=jnp.asarray(interior),
        area_t_m2=jnp.asarray(card.recipe.z_coord.nemo_e1e2t),
        metric_u_m=jnp.asarray(card.recipe.z_coord.nemo_e2u),
        metric_v_m=jnp.asarray(card.recipe.z_coord.nemo_e1v),
        sine_y=jnp.asarray(sine_y), sine_x=jnp.asarray(sine_x),
        poisson_eigenvalue_s_m2=jnp.asarray(eigen), grid=card.recipe.grid,
        wet_bounds=bounds, spacing_m=spacing)


class SWGScalarPotentialCNN(eqx.Module):
    """Small residual-free CNN producing one boundary-zero F potential.

    Four input channels are SSH plus the explicit noise input, T-centred U/V,
    and wet mask. Metrics remain non-trainable geometry in the conservative
    potential-to-face transform rather than redundant uniform CNN channels.
    Noise has no separate
    trainable projection: it enters the SSH channel, so deterministic mode
    cannot leave a noise-only parameter leaf inert.  The final convolution has
    no bias because a spatially constant potential is annihilated by the
    rotated gradient and would therefore be an inert trainable leaf.
    """

    layers: tuple
    spec: SWGNetworkSpec = eqx.field(static=True)

    def __init__(self, spec: SWGNetworkSpec, *, key):
        if spec.hidden_channels <= 0 or spec.hidden_layers < 1:
            raise ValueError("SWG CNN requires positive width and hidden depth")
        if spec.kernel_size <= 0 or spec.kernel_size % 2 != 1:
            raise ValueError("SWG CNN kernel_size must be a positive odd integer")
        if min(spec.ssh_scale_m, spec.velocity_scale_m_s,
               spec.potential_scale_m3_s2) <= 0.0:
            raise ValueError("SWG CNN physical scales must be positive")
        keys = jax.random.split(key, spec.hidden_layers + 1)
        padding = spec.kernel_size // 2
        layers = [eqx.nn.Conv2d(
            4, spec.hidden_channels, spec.kernel_size, padding=padding,
            key=keys[0], dtype=jnp.float64)]
        for layer_key in keys[1:-1]:
            layers.append(eqx.nn.Conv2d(
                spec.hidden_channels, spec.hidden_channels,
                spec.kernel_size, padding=padding, key=layer_key,
                dtype=jnp.float64))
        layers.append(eqx.nn.Conv2d(
            spec.hidden_channels, 1, spec.kernel_size, padding=padding,
            use_bias=False, key=keys[-1], dtype=jnp.float64))
        self.layers = tuple(layers)
        self.spec = spec

    def __call__(self, entry, geometry: SWGClosureGeometry, *, key, noise,
                 noise_scale):
        noise = jnp.asarray(noise, dtype=jnp.asarray(entry.ssh_k).dtype)
        if noise.shape != jnp.asarray(entry.ssh_k).shape:
            raise ValueError("closure noise must have the native F/T shape")
        u = jnp.asarray(entry.u_k)
        v = jnp.asarray(entry.v_k)
        u_west = jnp.concatenate([jnp.zeros_like(u[:, :1]), u[:, :-1]], axis=1)
        v_south = jnp.concatenate([jnp.zeros_like(v[:1]), v[:-1]], axis=0)
        u_t = 0.5 * (u + u_west)
        v_t = 0.5 * (v + v_south)
        # The key is structurally live from phase 3 onward.  A caller-provided
        # noise realization remains reparameterizable and reproducible.
        keyed_noise = noise + noise_scale * jax.random.normal(
            key, noise.shape, noise.dtype)
        channels = jnp.stack((
            jnp.asarray(entry.ssh_k) / self.spec.ssh_scale_m + keyed_noise,
            u_t / self.spec.velocity_scale_m_s,
            v_t / self.spec.velocity_scale_m_s,
            geometry.tmask.astype(u.dtype)), axis=0)
        value = channels
        for layer in self.layers[:-1]:
            value = jax.nn.elu(layer(value))
        potential = self.layers[-1](value)[0]
        return (self.spec.potential_scale_m3_s2 * potential
                * geometry.vertex_interior)


def network_parameter_count(network: SWGScalarPotentialCNN) -> int:
    """Count inexact scalar leaves in the trainable Equinox module."""
    leaves = jax.tree.leaves(eqx.filter(network, eqx.is_inexact_array))
    return int(sum(np.prod(leaf.shape, dtype=np.int64) for leaf in leaves))


def _gaussian_smooth_wet(field, geometry: SWGClosureGeometry):
    """Apply the registered sigma-one-cell Gaussian on the wet rectangle."""
    j0, j1, i0, i1 = geometry.wet_bounds
    wet = field[j0:j1 + 1, i0:i1 + 1]
    offsets = jnp.arange(-4, 5, dtype=field.dtype)
    kernel = jnp.exp(-0.5 * offsets**2)
    kernel = kernel / jnp.sum(kernel)
    padded = jnp.pad(wet, ((4, 4), (4, 4)), mode="reflect")
    kernel2 = (kernel[:, None] * kernel[None, :])[None, None]
    filtered = jax.lax.conv_general_dilated(
        padded[None, None], kernel2, window_strides=(1, 1),
        padding="VALID", dimension_numbers=("NCHW", "OIHW", "NCHW"))[0, 0]
    return jnp.zeros_like(field).at[j0:j1 + 1, i0:i1 + 1].set(filtered)


def balanced_fields_from_native(u_native, v_native,
                                geometry: SWGClosureGeometry):
    """Diagnose smoothed vorticity and Dirichlet rotational velocity."""
    u_full = jnp.concatenate([jnp.zeros_like(u_native[:, :1]), u_native], axis=1)
    v_full = jnp.concatenate([jnp.zeros_like(v_native[:1]), v_native], axis=0)
    zeta = relative_vorticity_cell_centre(u_full, v_full, geometry.grid)
    zeta = _gaussian_smooth_wet(zeta * geometry.tmask, geometry)
    j0, j1, i0, i1 = geometry.wet_bounds
    wet_zeta = zeta[j0:j1 + 1, i0:i1 + 1]
    spectral = geometry.sine_y.T @ wet_zeta @ geometry.sine_x
    streamfunction = (geometry.sine_y
                      @ (spectral / geometry.poisson_eigenvalue_s_m2)
                      @ geometry.sine_x.T)
    stream_full = jnp.zeros_like(zeta).at[
        j0:j1 + 1, i0:i1 + 1].set(streamfunction)
    padded = jnp.pad(stream_full, 1)
    rot_u = -(padded[2:, 1:-1] - padded[:-2, 1:-1]) / (
        2.0 * geometry.spacing_m)
    rot_v = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (
        2.0 * geometry.spacing_m)
    return BalancedTrajectory(
        zeta * geometry.tmask, rot_u * geometry.tmask, rot_v * geometry.tmask)


def balanced_trajectory_loss(predicted: BalancedTrajectory,
                             target: BalancedTrajectory,
                             geometry: SWGClosureGeometry,
                             normalizers: BalancedNormalizers, *,
                             kind: str):
    """Area-weighted point loss or multivariate energy score.

    Every predicted array has shape ``(members, hours, y, x)``; targets omit
    the member axis.  The energy score is the proper multivariate score over
    all three normalized, area-weighted balanced fields and requires S >= 2.
    """
    pred = jnp.stack((
        predicted.relative_vorticity_s1
        / jnp.sqrt(normalizers.relative_vorticity_variance_s2),
        predicted.rotational_u_m_s
        / jnp.sqrt(normalizers.rotational_u_variance_m2_s2),
        predicted.rotational_v_m_s
        / jnp.sqrt(normalizers.rotational_v_variance_m2_s2)), axis=-1)
    obs = jnp.stack((
        target.relative_vorticity_s1
        / jnp.sqrt(normalizers.relative_vorticity_variance_s2),
        target.rotational_u_m_s
        / jnp.sqrt(normalizers.rotational_u_variance_m2_s2),
        target.rotational_v_m_s
        / jnp.sqrt(normalizers.rotational_v_variance_m2_s2)), axis=-1)
    weight = geometry.area_t_m2 * geometry.tmask
    weight = weight / jnp.sum(weight)

    def member_distance(left, right):
        return jnp.sqrt(jnp.mean(jnp.sum(
            (left - right) ** 2 * weight[None, :, :, None], axis=(-3, -2, -1))))

    if kind == "point":
        squared = (pred[0] - obs) ** 2 * weight[None, :, :, None]
        return jnp.mean(jnp.sum(squared, axis=(-3, -2)))
    if kind != "energy_score":
        raise ValueError(f"unknown SWG balanced loss {kind!r}")
    members = pred.shape[0]
    if members < 2:
        raise ValueError("energy_score requires at least two ensemble members")
    observation_term = jnp.mean(jax.vmap(lambda member: member_distance(
        member, obs))(pred))
    pairwise = jax.vmap(lambda left: jax.vmap(
        lambda right: member_distance(left, right))(pred))(pred)
    return observation_term - 0.5 * jnp.mean(pairwise)


def build_segment_fn(card, geometry: SWGClosureGeometry, *,
                     window_days: int, checkpoint: bool = True):
    """Build one fixed-window hourly training segment with closed steady wind."""
    if window_days not in ADMITTED_WINDOWS_DAYS:
        raise ValueError(
            f"window_days must be one of {ADMITTED_WINDOWS_DAYS}, got {window_days}")
    steps_per_hour = int(SECONDS_PER_HOUR / card.dt_s)
    if steps_per_hour * card.dt_s != SECONDS_PER_HOUR:
        raise ValueError("SWG timestep must divide one hour exactly")
    tau_u, tau_v = swg_surface_wind(card.resolved_namelist)
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau_u, dtype=jnp.float64),
        tau_j_native=jnp.asarray(tau_v, dtype=jnp.float64))
    n_hours = 24 * window_days
    n_steps = n_hours * steps_per_hour

    def one_step(network, noise_scale, carry, inputs):
        step_key, stage_noise = inputs

        if network is None:
            state = swe_rk3_step(
                carry, card.dt_s, forcing, card.recipe.grid,
                card.recipe.z_coord, card.recipe.model_config,
                stage_potential_fn=None)
            return state, None

        def potential(entry, stage):
            return network(
                entry, geometry, key=jax.random.fold_in(step_key, stage),
                noise=stage_noise[stage - 1], noise_scale=noise_scale)

        state = swe_rk3_step(
            carry, card.dt_s, forcing, card.recipe.grid, card.recipe.z_coord,
            card.recipe.model_config, stage_potential_fn=potential)
        return state, None

    scanned_step = jax.checkpoint(one_step) if checkpoint else one_step

    def raw(network, initial_state, noise, key, noise_scale):
        expected = (n_steps, 3) + tuple(geometry.tmask.shape)
        if noise.shape != expected:
            raise ValueError(f"segment noise shape {noise.shape}, expected {expected}")
        if initial_state.tau_x_prev is None and initial_state.tau_y_prev is None:
            initial_state = initial_state._replace(
                tau_x_prev=forcing.tau_i_native,
                tau_y_prev=forcing.tau_j_native)
        elif initial_state.tau_x_prev is None or initial_state.tau_y_prev is None:
            raise ValueError("segment initial state has incomplete wind-stress carry")
        keys = jax.random.split(key, n_steps)

        def one_hour(state, hour):
            start = hour * steps_per_hour
            hour_keys = jax.lax.dynamic_slice_in_dim(
                keys, start, steps_per_hour, axis=0)
            hour_noise = jax.lax.dynamic_slice_in_dim(
                noise, start, steps_per_hour, axis=0)
            state, _ = jax.lax.scan(
                lambda carry, values: scanned_step(
                    network, noise_scale, carry, values),
                state, (hour_keys, hour_noise))
            native = (state.eta.data, state.u.data[:, 1:, 0],
                      state.v.data[1:, :, 0])
            return state, native

        return jax.lax.scan(one_hour, initial_state, jnp.arange(n_hours))

    return raw, eqx.filter_jit(raw)


def create_swg_optimizer(config: TrainingConfig):
    """Construct the optimizer only through the repository-owned factory."""
    return create_optimizer(config)


def assert_network_gradients_live(grads) -> None:
    """Apply the repository no-inert gate to every trainable CNN leaf."""
    named = {
        jax.tree_util.keystr(path): value
        for path, value in jax.tree.leaves_with_path(
            eqx.filter(grads, eqx.is_inexact_array))
    }
    assert_no_inert(named)
