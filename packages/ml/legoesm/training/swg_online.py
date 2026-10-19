"""Differentiable online-learning pieces for the NEMO-faithful SWG card.

This module owns the experimental training seam, not the ocean card.  A model
produces one F-point volume-transport tendency potential; the certified SWG
stage owns its matched rotated gradient.  The deterministic and stochastic
losses share one ensemble-shaped balanced-field interface.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from legoesm.grids.operators_latlon_cgrid import (
    compute_vertex_mask,
    curl_vertex_cgrid,
)
from legoesm.ml.training import TrainingConfig, create_optimizer
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
from legoesm.ocean.dynamics.swe_rk3 import (
    SWE_SURFACE_MODES,
    _entry_from_state,
    swe_rigid_lid_geometry,
    swe_rigid_lid_projection,
    swe_rigid_lid_stage1_phi,
    swe_rk3_step,
    swe_rotated_potential_forcing,
    wet_rectangle_bounds,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_MADDISON_NAMELIST,
    SWG_MADDISON_RL_NAMELIST,
    swg_surface_wind,
)
from legoesm.ocean.fidelity.swg_coarse_graining import (
    build_jax_filtered_restrictor,
    cgrid_geometry,
    shifted_weighted_sum,
)
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.training.inert_params import assert_no_inert
from scipy.fft import dctn, idctn

SECONDS_PER_HOUR = 3_600
HOURS_PER_DAY = 24
ADMITTED_WINDOWS_DAYS = (1, 2, 4, 8)
OBJECTIVE_OPERATORS = ("symmetric", "template")
# Eq. 7 weights every interior F point.  Whether the first interior ring next
# to the walls is weighted is an owner decision (round 6): no default exists.
WALL_RING_OPTIONS = ("weighted", "excluded")
DATASET_GAP_HOURS = 30 * 24
TEMPLATE_BATCH_SIZE = 10


def window_days_admitted(window_days) -> bool:
    """Whole days: a length of ``ADMITTED_WINDOWS_DAYS`` or longer than all of
    them (a curriculum extension; whether it fits the splits is
    ``require_curriculum_windows``)."""
    return type(window_days) is int and (
        window_days in ADMITTED_WINDOWS_DAYS
        or window_days > ADMITTED_WINDOWS_DAYS[-1])


def require_curriculum_windows(windows, splits_days) -> None:
    """Refuse a curriculum other than strictly increasing admitted lengths in
    which every length beyond ``ADMITTED_WINDOWS_DAYS`` follows all of them
    and fits ``splits_days`` (``{name: [first_day, last_day]}``): at least
    one batch of ``TEMPLATE_BATCH_SIZE`` windows in the training split for
    every starting offset of the training lattice, and in the validation
    split (the held-out lattice starts at its first hour)."""
    if not isinstance(windows, list) or not windows \
            or not all(window_days_admitted(window) for window in windows) \
            or windows != sorted(set(windows)):
        raise ValueError(
            f"window_days must increase through {ADMITTED_WINDOWS_DAYS} and "
            "may then continue with longer whole-day lengths")
    longer = [window for window in windows if window > ADMITTED_WINDOWS_DAYS[-1]]
    if longer and tuple(windows[:len(ADMITTED_WINDOWS_DAYS)]) \
            != ADMITTED_WINDOWS_DAYS:
        raise ValueError(
            f"window_days {windows} continue past {ADMITTED_WINDOWS_DAYS[-1]} "
            f"days without starting with {ADMITTED_WINDOWS_DAYS}")
    for window in longer:
        window_hours = HOURS_PER_DAY * window
        for split, offset in (("train", window_hours - 1), ("validation", 0)):
            first_day, last_day = splits_days[split]
            span = HOURS_PER_DAY * (last_day - first_day)
            count = len(range(offset, span - window_hours + 1, window_hours))
            if count < TEMPLATE_BATCH_SIZE:
                raise ValueError(
                    f"the {split} split holds {count} {window}-day windows "
                    f"(worst starting offset), fewer than one batch of "
                    f"{TEMPLATE_BATCH_SIZE}")
# Sea-level-only start balance, the rule registered with the operator's
# 2026-10-13 ten-start measurement: stop when the closed-wall divergent part
# of the first-step acceleration is at most this fraction of its rotational
# part (T-centred rms); an iteration COUNT is a module constant, never config.
START_REBALANCE_TOLERANCE = 1.0e-3
START_REBALANCE_MAX_ITERATIONS = 20
_V2_SCHEMA = "legoesm-swg-double-gyre-hourly-target-v2"
_HOURLY_FIELDS = (("ssh", "m"), ("u", "m_s"), ("v", "m_s"))
# The fine-deck namelist keys that change with the run length alone.
_RUN_LENGTH_KEYS = frozenset({"nn_itend", "nn_write", "nn_stock"})
# Rigid-lid streams only (user decision 2026-10-18): the hourly-target program
# differs from a plain restart by round-off.  Must equal the driver's
# ``LID_RESTART_TOLERANCE`` (scripts/run/run_swg_double_gyre.py).
RIGID_LID_RESTART_TOLERANCE = {
    "kind": "max_abs_relative_to_field_max", "value": 1.0e-12,
    "applies_to": "rigid_lid"}
# The registered fine namelist of each surface a stream may be generated on.
_FINE_NAMELIST_OF_SURFACE = {
    "explicit": SWG_MADDISON_NAMELIST, "rigid_lid": SWG_MADDISON_RL_NAMELIST}
# Two evaluations of one fine state (restriction inside the compiled scan and
# eagerly) differ by at most 5.1e-13 m SSH and 8.9e-16 m/s on CPU and CPU/GPU
# (round-5 review); 1e-10 of the field's largest magnitude is 300 times that
# and 1e8 times below a one-hour shift (1.3e-2 m).
_ADJACENT_HOUR_RELATIVE_ATOL = 1.0e-10
# ponytail: per-day cache, ~4 MB per day file for one operator; raise it only
# if a measured epoch shows evictions dominate the batch time.
_DAY_FILE_CACHE = 128


class SWGNetworkSpec(NamedTuple):
    """Manifest-required CNN choices; no scientific choice is implicit."""

    hidden_channels: int
    hidden_layers: int
    kernel_size: int
    ssh_scale_m: float
    velocity_scale_m_s: float
    potential_scale_m3_s2: float
    wall_margin_rings: int


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
    """Hourly balanced fields on their native points.

    Relative vorticity is on interior F points (zero on the closed-wall
    vertices); the rotational velocity of its streamfunction is on the native
    U and V faces (zero on wall faces).
    """

    relative_vorticity_s1: jax.Array
    rotational_u_m_s: jax.Array
    rotational_v_m_s: jax.Array


class BalancedNormalizers(NamedTuple):
    """Caller-supplied variances for the dormant energy score only.

    Eq. 7 (experiment one) takes no data normalizers; a split-safe source for
    these must be built with the stochastic phase that consumes them.
    """

    relative_vorticity_variance_s2: float
    rotational_u_variance_m2_s2: float
    rotational_v_variance_m2_s2: float


class HourlyStateRecord(NamedTuple):
    """One content-pinned coarse hourly state stream."""

    ssh_m: np.ndarray
    u_m_s: np.ndarray
    v_m_s: np.ndarray
    absolute_hour: np.ndarray
    operator: str
    source_root: str
    manifest_sha256: str
    includes_initial_state: bool


class WindowBatch(NamedTuple):
    """A batch of sea-level-rebalanced T0 states and hourly targets.

    ``initial`` is the stored state at each start hour with its sea level
    rebalanced (velocities are the stored words); ``target`` holds the stored
    right endpoints 1..24w hours later, from the same operator.
    """

    initial: tuple[np.ndarray, np.ndarray, np.ndarray]
    target: tuple[np.ndarray, np.ndarray, np.ndarray]
    start_absolute_hour: np.ndarray
    split: str
    window_days: int


class PinnedHourlySource(NamedTuple):
    """Immutable membership and time coverage of one hourly chunk."""

    root: str
    manifest_sha256: str
    first_absolute_hour: int
    last_absolute_hour: int
    daily_sha256: tuple[str, ...]
    truth_run: str
    surface: str


class ShiftedConv2d(eqx.Module):
    """Zero-padded 2-D cross-correlation without a convolution primitive.

    Parameter shapes and initialization reproduce ``equinox.nn.Conv2d`` for
    stride-one, dilation-one, single-group kernels.  The installed Equinox
    0.13 implementation splits the supplied key into weight/bias keys and
    samples both uniformly from ``[-1/sqrt(Cin*K*K), +...]``; retaining that
    sequence keeps existing seeds and checkpoints numerically unchanged.
    """

    weight: jax.Array
    bias: jax.Array | None
    kernel_size: int = eqx.field(static=True)
    padding: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 *, padding: int, use_bias: bool = True, key,
                 dtype=jnp.float64):
        if min(in_channels, out_channels) <= 0:
            raise ValueError("shifted convolution channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError("shifted convolution kernel must be positive odd")
        if padding < 0:
            raise ValueError("shifted convolution padding cannot be negative")
        wkey, bkey = jax.random.split(key, 2)
        limit = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = jax.random.uniform(
            wkey, (out_channels, in_channels, kernel_size, kernel_size),
            dtype=dtype, minval=-limit, maxval=limit)
        self.bias = (jax.random.uniform(
            bkey, (out_channels, 1, 1), dtype=dtype,
            minval=-limit, maxval=limit) if use_bias else None)
        self.kernel_size = kernel_size
        self.padding = padding
        self.in_channels = in_channels

    def __call__(self, value):
        """Contract all shifted slices with the unchanged OIHW weights."""
        if value.ndim != 3 or value.shape[0] != self.in_channels:
            raise ValueError(
                "shifted convolution input must have shape (channels,y,x)")
        padded = jnp.pad(
            value, ((0, 0), (self.padding, self.padding),
                    (self.padding, self.padding)))
        height = padded.shape[1] - self.kernel_size + 1
        width = padded.shape[2] - self.kernel_size + 1
        shifted = jnp.stack([
            padded[:, j:j + height, i:i + width]
            for j in range(self.kernel_size)
            for i in range(self.kernel_size)
        ], axis=1)
        weights = self.weight.reshape(
            self.weight.shape[0], self.in_channels,
            self.kernel_size * self.kernel_size)
        output = jnp.einsum("oik,ikhw->ohw", weights, shifted)
        return output if self.bias is None else output + self.bias


def assert_no_convolution_operations(lowered, *, label: str) -> None:
    """Fail if a JAX/Equinox lowered program contains a convolution op."""
    as_text = getattr(lowered, "as_text", None)
    if not callable(as_text):
        raise TypeError("convolution guard requires a lowered JAX program")
    text = as_text()
    markers = ("stablehlo.convolution", "mhlo.convolution", " convolution(")
    present = tuple(marker for marker in markers if marker in text)
    if present:
        raise RuntimeError(
            f"{label} lowers a forbidden convolution operation: {present}")


def build_closure_geometry(card) -> SWGClosureGeometry:
    """Build immutable closure geometry from one resolved SWG card.

    The card must have been built under the fp64 precision policy: a float32
    card would roll out a mixed-precision model that is not the certified one.
    """
    if {np.asarray(card.recipe.z_coord.nemo_e1e2t).dtype,
            np.asarray(card.recipe.initial_state.eta.data).dtype} \
            != {np.dtype(np.float64)}:
        raise ValueError(
            "SWG online closure requires a card built under the fp64 "
            "precision policy")
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
    # F point (j, i) is the north-east corner of T (j, i): the interior F
    # points are [j0, j1) x [i0, i1) and the closed walls are the vertex
    # rows/columns j0-1, j1 and i0-1, i1, where the streamfunction is zero.
    f_rectangle = np.zeros_like(tmask)
    f_rectangle[bounds[0]:bounds[1], bounds[2]:bounds[3]] = True
    if not np.array_equal(interior, f_rectangle):
        raise ValueError("SWG interior F points are not the wet-rectangle corners")
    ny = bounds[1] - bounds[0]
    nx = bounds[3] - bounds[2]
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


def wall_margin_mask(vertex_interior, rings: int):
    """``vertex_interior`` eroded ``rings`` times with the 4-neighbour
    stencil; array edges count as 0 and the dtype is the input's."""
    mask = jnp.asarray(vertex_interior)
    for _ in range(rings):
        padded = jnp.pad(mask, 1)
        mask = jnp.stack((
            padded[1:-1, 1:-1], padded[:-2, 1:-1], padded[2:, 1:-1],
            padded[1:-1, :-2], padded[1:-1, 2:])).min(axis=0)
    return mask


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
        if spec.wall_margin_rings < 0:
            raise ValueError("SWG CNN wall_margin_rings must be >= 0")
        keys = jax.random.split(key, spec.hidden_layers + 1)
        padding = spec.kernel_size // 2
        layers = [ShiftedConv2d(
            4, spec.hidden_channels, spec.kernel_size, padding=padding,
            key=keys[0], dtype=jnp.float64)]
        for layer_key in keys[1:-1]:
            layers.append(ShiftedConv2d(
                spec.hidden_channels, spec.hidden_channels,
                spec.kernel_size, padding=padding, key=layer_key,
                dtype=jnp.float64))
        layers.append(ShiftedConv2d(
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
        potential = (self.spec.potential_scale_m3_s2 * potential
                     * geometry.vertex_interior)
        if self.spec.wall_margin_rings > 0:
            potential = potential * wall_margin_mask(
                geometry.vertex_interior, self.spec.wall_margin_rings)
        return potential


def network_parameter_count(network: SWGScalarPotentialCNN) -> int:
    """Count inexact scalar leaves in the trainable Equinox module."""
    leaves = jax.tree.leaves(eqx.filter(network, eqx.is_inexact_array))
    return int(sum(np.prod(leaf.shape, dtype=np.int64) for leaf in leaves))


class Eq6NetworkScales(NamedTuple):
    """Maddison Eq. 6 scales and this network's shallow-water analogue."""

    vorticity_scale_s1: float
    vorticity_tendency_scale_s2: float
    grid_spacing_m: float
    depth_m: float
    ssh_scale_m: float
    velocity_scale_m_s: float
    potential_scale_m3_s2: float


def maddison_eq6_network_scales(
        card, geometry: SWGClosureGeometry) -> Eq6NetworkScales:
    """Shallow-water analogue of Maddison Eq. 6, every factor from the deck.

    Eq. 6 divides the template network's input vorticity by ``|beta| L``
    and multiplies its output by the wind-forcing scale of the vorticity
    equation, ``alpha_out = |tau0| pi / (rho0 D L)``, ``L`` half the basin
    side (Table 1, "Domain size 2L").  This network reads grid-point SSH and
    velocity and returns an F-point volume-transport potential ``Phi``, so
    each of its scales is the field whose one-spacing (``dx``) differences
    carry the template's scale:

    * velocity ``U = |beta| L dx``: a one-cell difference of ``u/U`` is the
      template's input ``zeta/(|beta| L)`` with ``zeta ~ du/dx``;
    * SSH ``eta_s = |f0| |beta| L dx**2 / g``: the five-point Laplacian of
      ``eta/eta_s`` is the geostrophic ``(g lap(eta)/f0)/(|beta| L)``;
    * potential ``Phi_s = alpha_out D dx**2``: the closure force
      ``(d_y Phi, -d_x Phi)/h`` has curl ``-lap(Phi)/h``, so for ``h = D``
      minus the five-point Laplacian of ``Phi/Phi_s`` is the template's
      output ``S/alpha_out``.

    ``beta``, ``f0``, ``tau0`` and ``2L`` come from the resolved namelist,
    ``rho0`` and ``g`` from the card's constants (the values the model uses
    to turn wind stress into acceleration), ``D`` from the card's resting
    layer thickness, which must be one value over the wet basin, and ``dx``
    from the closure geometry.
    """
    namelist = card.resolved_namelist
    constants = card.recipe.model_config.constants
    wet = np.asarray(geometry.tmask, dtype=bool)
    thickness = np.asarray(card.recipe.z_coord.nemo_e3t_0, dtype=np.float64)
    depths = np.unique(thickness.reshape(wet.shape + (-1,))[wet])
    if depths.size != 1 or not np.isfinite(depths[0]) or depths[0] <= 0.0:
        raise ValueError(
            "Eq. 6 analogue needs one positive resting depth over the basin, "
            f"got {depths[:4]}")
    half_width = maddison_half_width_m(namelist.rn_domsiz_m, geometry)
    beta = abs(float(namelist.rn_beta_m_s))
    f0 = abs(float(namelist.rn_f0_s))
    tau0 = abs(float(namelist.rn_tau_pa))
    rho0 = float(constants.rho_0)
    gravity = float(constants.g)
    depth = float(depths[0])
    spacing = float(geometry.spacing_m)
    if min(beta, f0, tau0, rho0, gravity, spacing) <= 0.0:
        raise ValueError("Eq. 6 analogue needs positive deck parameters")
    vorticity = beta * half_width
    tendency = tau0 * math.pi / (rho0 * depth * half_width)
    return Eq6NetworkScales(
        vorticity_scale_s1=vorticity,
        vorticity_tendency_scale_s2=tendency,
        grid_spacing_m=spacing,
        depth_m=depth,
        ssh_scale_m=f0 * vorticity * spacing**2 / gravity,
        velocity_scale_m_s=vorticity * spacing,
        potential_scale_m3_s2=tendency * depth * spacing**2)


def eq6_scale_report(network: SWGScalarPotentialCNN, states: tuple,
                     card, geometry: SWGClosureGeometry,
                     scales: Eq6NetworkScales, *, key) -> dict:
    """Dimensionless network inputs and initial output on given states.

    Reports, per state and over all of them, the rms and maximum magnitude of
    the scaled SSH and T-centred velocities on wet T points (what the network
    reads), of the template's own input ``zeta/(|beta| L)`` on interior F
    points (for comparison), and of the scaled output ``Phi/Phi_s`` and the
    five-point Laplacian of it (the template's output in grid units).
    """
    if network.spec.ssh_scale_m != scales.ssh_scale_m \
            or network.spec.velocity_scale_m_s != scales.velocity_scale_m_s \
            or network.spec.potential_scale_m3_s2 \
            != scales.potential_scale_m3_s2:
        raise ValueError("network scales differ from the Eq. 6 analogue")
    tau = swg_surface_wind(card.resolved_namelist)
    wet = np.asarray(geometry.tmask, dtype=bool)
    interior = np.asarray(geometry.vertex_interior, dtype=bool)
    fields = {name: [] for name in (
        "ssh", "u_t", "v_t", "template_input_zeta", "output_potential",
        "output_laplacian")}
    for index, state in enumerate(states):
        entry = _network_entry(_entry_from_state(
            state, tau, tau, card.recipe.grid, card.recipe.z_coord,
            card.recipe.model_config), state, card)
        potential = np.asarray(network(
            entry, geometry, key=jax.random.fold_in(key, index),
            noise=jnp.zeros_like(entry.ssh_k), noise_scale=jnp.asarray(0.0)),
            dtype=np.float64) / scales.potential_scale_m3_s2
        u = np.asarray(entry.u_k, dtype=np.float64)
        v = np.asarray(entry.v_k, dtype=np.float64)
        u_t = 0.5 * (u + np.concatenate([np.zeros_like(u[:, :1]), u[:, :-1]], 1))
        v_t = 0.5 * (v + np.concatenate([np.zeros_like(v[:1]), v[:-1]], 0))
        zeta = np.asarray(balanced_fields_from_native(
            u, v, geometry, objective="template").relative_vorticity_s1)
        padded = np.pad(potential, 1)
        laplacian = (padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:]
                     + padded[1:-1, :-2] - 4.0 * potential)
        fields["ssh"].append(
            np.asarray(entry.ssh_k, dtype=np.float64)[wet] / scales.ssh_scale_m)
        fields["u_t"].append(u_t[wet] / scales.velocity_scale_m_s)
        fields["v_t"].append(v_t[wet] / scales.velocity_scale_m_s)
        fields["template_input_zeta"].append(
            zeta[interior] / scales.vorticity_scale_s1)
        fields["output_potential"].append(potential[interior])
        fields["output_laplacian"].append(laplacian[interior])

    def summary(values):
        return {"per_state_rms": [float(np.sqrt(np.mean(x**2))) for x in values],
                "per_state_max_abs": [float(np.max(np.abs(x))) for x in values],
                "rms": float(np.sqrt(np.mean(np.concatenate(values) ** 2))),
                "max_abs": float(np.max(np.abs(np.concatenate(values))))}

    return {"scales": scales._asdict(), "states": len(states),
            **{name: summary(values) for name, values in fields.items()}}


def _gaussian_smooth_vorticity(zeta, geometry: SWGClosureGeometry):
    """Registered sigma-one-cell coarse Gaussian on the closed F rectangle.

    The stencil spans the wall-to-wall vertex rectangle and extends it with
    ODD parity about the wall vertices, identically at all four walls.  Free
    slip (``rn_shlat=0``) makes wall vorticity zero and vorticity odd about a
    wall, which is also the parity the template's fine filter gives it (even
    tangential, odd normal transport).  Walls are re-zeroed after round-off.
    """
    j0, j1, i0, i1 = geometry.wet_bounds
    block = zeta[j0 - 1:j1 + 1, i0 - 1:i1 + 1]
    offsets = jnp.arange(-4, 5, dtype=zeta.dtype)
    kernel = jnp.exp(-0.5 * offsets**2)
    kernel = kernel / jnp.sum(kernel)
    block = shifted_weighted_sum(block, kernel, axis=0, extension="odd")
    block = shifted_weighted_sum(block, kernel, axis=1, extension="odd")
    return (jnp.zeros_like(zeta).at[j0 - 1:j1 + 1, i0 - 1:i1 + 1].set(block)
            * geometry.vertex_interior)


def balanced_fields_from_native(u_native, v_native,
                                geometry: SWGClosureGeometry, *,
                                objective: str):
    """Diagnose objective-specific vorticity and rotational velocity.

    ``symmetric`` applies the coarse-grid Gaussian to the model and target.
    ``template`` leaves the model diagnostic unfiltered; its target state was
    already filtered on the fine grid before restriction.
    """
    if objective not in OBJECTIVE_OPERATORS:
        raise ValueError(
            f"objective must be explicitly selected from {OBJECTIVE_OPERATORS}")
    u_full = jnp.concatenate([jnp.zeros_like(u_native[:, :1]), u_native], axis=1)
    v_full = jnp.concatenate([jnp.zeros_like(v_native[:1]), v_native], axis=0)
    # The SWG vorticity degree of freedom lives at F points.  Do not average
    # four vertices to T here: that average annihilates the native face-grid
    # Nyquist mode which the template objective is specifically meant to see.
    zeta = curl_vertex_cgrid(u_full, v_full, geometry.grid)[1:, 1:]
    zeta = zeta * geometry.vertex_interior
    if objective == "symmetric":
        zeta = _gaussian_smooth_vorticity(zeta, geometry)
    j0, j1, i0, i1 = geometry.wet_bounds
    spectral = geometry.sine_y.T @ zeta[j0:j1, i0:i1] @ geometry.sine_x
    streamfunction = (geometry.sine_y
                      @ (spectral / geometry.poisson_eigenvalue_s_m2)
                      @ geometry.sine_x.T)
    psi = jnp.zeros_like(zeta).at[j0:j1, i0:i1].set(streamfunction)
    # U face (j, i) runs from F (j-1, i) to F (j, i); V face (j, i) from
    # F (j, i-1) to F (j, i).  u = -dpsi/dy and v = +dpsi/dx, so the discrete
    # curl of (u, v) at an interior F point is the five-point Laplacian of psi.
    psi_south = jnp.concatenate([jnp.zeros_like(psi[:1]), psi[:-1]], axis=0)
    psi_west = jnp.concatenate([jnp.zeros_like(psi[:, :1]), psi[:, :-1]], axis=1)
    rot_u = -(psi - psi_south) / geometry.spacing_m
    rot_v = (psi - psi_west) / geometry.spacing_m
    return BalancedTrajectory(
        zeta, rot_u * geometry.umask, rot_v * geometry.vmask)


def diagnose_native_trajectory(hourly_native,
                               geometry: SWGClosureGeometry, *,
                               objective: str):
    """Apply the selected shared diagnostic to model or truth native arrays."""
    _, u, v = hourly_native
    return jax.vmap(
        lambda u_hour, v_hour: balanced_fields_from_native(
            u_hour, v_hour, geometry, objective=objective))(u, v)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@functools.lru_cache(maxsize=_DAY_FILE_CACHE)
def _load_pinned_day(path: str, expected_sha256: str, prefix: str,
                     first_hour: int, with_initial: bool):
    """Load one digest-pinned day file of 24 hourly right endpoints.

    The file must hold exactly 24 records of one grid for every field, with
    absolute hours ``first_hour .. first_hour + 23``.  Results are cached
    (each day file is hashed and decoded once while cached) and returned
    read-only because the cache shares them.
    """
    day_path = Path(path)
    if _sha256(day_path) != expected_sha256:
        raise ValueError(f"hourly target digest differs: {day_path.name}")
    with np.load(day_path, allow_pickle=False) as saved:
        hours = np.asarray(saved["absolute_hour"], dtype=np.int64)
        if not np.array_equal(hours, np.arange(
                first_hour, first_hour + HOURS_PER_DAY, dtype=np.int64)):
            raise ValueError("hourly target has wrong absolute-time lineage")
        hourly = tuple(
            np.asarray(saved[f"{prefix}{name}_hourly_{unit}"], dtype=np.float64)
            for name, unit in _HOURLY_FIELDS)
        initial = tuple(
            np.asarray(saved[f"{prefix}{name}_initial_{unit}"], dtype=np.float64)
            for name, unit in _HOURLY_FIELDS) if with_initial else None
    grid = hourly[0].shape[1:]
    if len(grid) != 2 or any(value.shape != (HOURS_PER_DAY,) + grid
                             for value in hourly) \
            or (initial is not None
                and any(value.shape != grid for value in initial)):
        raise ValueError(
            f"{day_path.name} does not hold 24 hourly records of one grid")
    arrays = hourly + (initial or ())
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("hourly target contains a non-finite state")
    for value in arrays:
        value.setflags(write=False)
    return hourly, initial


def _admitted_hourly_manifest(root: Path, expected_manifest_sha256: str) -> dict:
    """Read one complete hourly manifest pinned by digest and day membership."""
    manifest_path = root / "manifest.json"
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("hourly target manifest differs from registered split")
    manifest = json.loads(manifest_path.read_text())
    days = manifest.get("days")
    if manifest.get("status") != "complete" \
            or manifest.get("completed_days") != days \
            or type(days) is not int or days <= 0:
        raise ValueError("hourly target segment is not a complete run")
    if set(manifest.get("daily_sha256", {})) \
            != {f"day{day:04d}.npz" for day in range(1, days + 1)}:
        raise ValueError("hourly manifest does not pin the exact daily membership")
    return manifest


def _stream_surface(manifest: dict) -> str:
    """The surface a v2 stream was generated on; streams made before the
    rigid-lid surface existed record none and are explicit."""
    surface = manifest.get("surface", "explicit")
    if surface not in SWE_SURFACE_MODES:
        raise ValueError(
            f"hourly stream surface must be one of {SWE_SURFACE_MODES}; "
            f"got {surface!r}")
    return surface


def _truth_run_of(manifest: dict) -> str:
    """Refuse a v2 stream that is not unperturbed, verified fp64 truth.

    Returns the fine truth run (directory of its restart manifest), so that a
    contract can refuse sources drawn from two different truth runs.
    """
    restart = manifest.get("restart") or {}
    if manifest.get("sampling") != "hourly_dual_coarse_target" \
            or manifest.get("hourly_samples_per_day") != HOURS_PER_DAY \
            or manifest.get("hourly_target_deck") != "dg50":
        raise ValueError("hourly stream is not dual-operator dg50 hourly sampling")
    if manifest.get("precision_policy") != "fp64":
        raise ValueError("hourly stream was not generated under the fp64 policy")
    if restart.get("perturb_amplitude_m") is not None \
            or "balanced_perturbation" in restart:
        raise ValueError("hourly stream starts from a perturbed restart, not truth")
    surface = _stream_surface(manifest)
    verification = manifest.get("restart_next_verification") or {}
    result = verification.get("result") or {}
    if result.get("all_equal") is not True and not (
            surface == "rigid_lid" and result.get("within_tolerance") is True
            and verification.get("tolerance") == RIGID_LID_RESTART_TOLERANCE):
        raise ValueError(
            "hourly stream lacks a passed next-day restart verification")
    if (manifest.get("numerical_health") or {}).get("all_finite") is not True:
        raise ValueError("hourly stream did not record finite numerical health")
    registered = json.loads(json.dumps(
        _FINE_NAMELIST_OF_SURFACE[surface]._asdict()))
    namelist = manifest.get("namelist") or {}
    if {key: value for key, value in namelist.items()
            if key not in _RUN_LENGTH_KEYS} \
            != {key: value for key, value in registered.items()
                if key not in _RUN_LENGTH_KEYS}:
        raise ValueError(
            "hourly stream was not generated from the registered fine deck")
    admission = restart.get("manifest_admission") or {}
    if not admission.get("manifest_path"):
        raise ValueError("hourly stream does not name its fine truth run")
    if type(restart.get("source_day")) is not int:
        raise ValueError("hourly stream does not name its restart day")
    snapshot = f"day{restart['source_day']:04d}.npz"
    if admission.get("manifest_status") not in ("complete", "pending") \
            or not restart.get("sha256") \
            or admission.get("snapshot_sha256") != restart["sha256"] \
            or Path(restart.get("path") or "").name != snapshot \
            or _truth_listing(admission["manifest_path"], snapshot) \
            != restart["sha256"]:
        raise ValueError(
            "hourly stream's restart is not a snapshot its truth run lists")
    return str(Path(admission["manifest_path"]).parent)


def _truth_listing(manifest_path: str, snapshot: str) -> str | None:
    """Digest the fine truth run lists today for one daily snapshot, or None.

    A run's pending manifest is renamed ``manifest.json`` when the run
    completes, so the recorded name and both standard names are tried.
    """
    named = Path(manifest_path)
    for candidate in (named, named.parent / "manifest.json",
                      named.parent / "manifest.pending.json"):
        if candidate.is_file():
            listed = json.loads(candidate.read_text()).get(
                "daily_sha256", {}).get(snapshot)
            if listed is not None:
                return listed
    return None


def load_content_pinned_hourly_segment(
        root: Path, *, expected_manifest_sha256: str, operator: str,
) -> HourlyStateRecord:
    """Load one complete v1/v2 hourly stream with exact content membership.

    v2 streams must also carry truth provenance (see ``_truth_run_of``);
    v1 streams are box-only legacy diagnostics, refused by the contract.
    """
    if operator not in OBJECTIVE_OPERATORS:
        raise ValueError(
            f"operator must be explicitly selected from {OBJECTIVE_OPERATORS}")
    manifest = _admitted_hourly_manifest(root, expected_manifest_sha256)
    schema = manifest.get("schema")
    if schema == "legoesm-swg-double-gyre-hourly-target-v1":
        if operator != "symmetric":
            raise ValueError("v1 hourly targets contain only the box operator")
        prefix = ""
        includes_initial = False
    elif schema == _V2_SCHEMA:
        _truth_run_of(manifest)
        prefix = "box_" if operator == "symmetric" else "filtered_"
        includes_initial = True
    else:
        raise ValueError(f"unsupported hourly target schema {schema!r}")
    source_day = int(manifest["restart"]["source_day"])
    days = manifest["days"]
    loaded = [_load_pinned_day(
        str(root / f"day{day:04d}.npz"),
        manifest["daily_sha256"][f"day{day:04d}.npz"], prefix,
        (source_day + day - 1) * HOURS_PER_DAY + 1,
        includes_initial and day == 1) for day in range(1, days + 1)]
    fields = tuple(np.concatenate([hourly[index] for hourly, _ in loaded])
                   for index in range(3))
    hours = np.arange(source_day * HOURS_PER_DAY + 1,
                      (source_day + days) * HOURS_PER_DAY + 1, dtype=np.int64)
    if includes_initial:
        fields = tuple(np.concatenate((value[None], field), axis=0)
                       for value, field in zip(loaded[0][1], fields,
                                               strict=True))
        hours = np.concatenate(
            (np.asarray([source_day * HOURS_PER_DAY], dtype=np.int64), hours))
    return HourlyStateRecord(
        *fields, hours, operator, str(root.resolve()),
        expected_manifest_sha256, includes_initial)


def _pinned_source(root: Path, manifest_sha256: str) -> PinnedHourlySource:
    """Admit one complete, content-pinned v2 chunk with truth provenance."""
    manifest = _admitted_hourly_manifest(root, manifest_sha256)
    if manifest.get("schema") != _V2_SCHEMA:
        raise ValueError(
            "training dataset sources require v2 stored initial states")
    truth_run = _truth_run_of(manifest)
    surface = _stream_surface(manifest)
    days = manifest["days"]
    source_day = int(manifest["restart"]["source_day"])
    return PinnedHourlySource(
        str(root.resolve()), manifest_sha256, source_day * HOURS_PER_DAY,
        (source_day + days) * HOURS_PER_DAY,
        tuple(manifest["daily_sha256"][f"day{day:04d}.npz"]
              for day in range(1, days + 1)), truth_run, surface)


def _pinned_day(source: PinnedHourlySource, objective: str, day_index: int):
    return _load_pinned_day(
        str(Path(source.root) / f"day{day_index + 1:04d}.npz"),
        source.daily_sha256[day_index],
        "box_" if objective == "symmetric" else "filtered_",
        source.first_absolute_hour + HOURS_PER_DAY * day_index + 1,
        day_index == 0)


def _record_at(source: PinnedHourlySource, objective: str, hour: int):
    day_index, index = divmod(
        hour - source.first_absolute_hour - 1, HOURS_PER_DAY)
    return tuple(value[index]
                 for value in _pinned_day(source, objective, day_index)[0])


def stored_state_at(sources, objective: str, hour: int):
    """Stored (ssh, u, v) of ``objective``'s operator at one absolute hour,
    and the pinned source that served it.

    An hour is served by the chunk holding it as a right endpoint (the
    in-scan restriction, like every target).  A chunk's stored t0 serves
    only an hour on which no chunk ends, i.e. the first hour of a record; at
    a shared restart hour it is checked by the contract, not used.
    """
    if objective not in OBJECTIVE_OPERATORS:
        raise ValueError(
            f"objective must be explicitly selected from {OBJECTIVE_OPERATORS}")
    for source in sources:
        if source.first_absolute_hour < hour <= source.last_absolute_hour:
            return _record_at(source, objective, hour), source
    for source in sources:
        if source.first_absolute_hour == hour:
            return _pinned_day(source, objective, 0)[1], source
    raise ValueError(f"dataset source gap at absolute hour {hour}")


def pinned_sources(chunk_roots) -> tuple[PinnedHourlySource, ...]:
    """Admit complete v2 chunks by their on-disk manifest digests."""
    return tuple(
        _pinned_source(Path(root), _sha256(Path(root) / "manifest.json"))
        for root in chunk_roots)


def start_rebalancer(card):
    """The registered window-start rule: one rebalancer for every start."""
    return build_start_rebalancer(
        card, tolerance=START_REBALANCE_TOLERANCE,
        max_iterations=START_REBALANCE_MAX_ITERATIONS)


def record_start_state(card, chunk_roots, *, objective: str,
                       absolute_hour: int):
    """A free-run start built exactly as a training window start.

    The stored state of ``objective``'s operator at ``absolute_hour`` in the
    pinned chunks, its sea level rebalanced by the registered rule
    (``start_rebalancer``); returns ``((ssh, u, v), provenance)`` with the
    serving chunk, its manifest digest and the rebalancing report.
    """
    if type(absolute_hour) is not int or absolute_hour < 0:
        raise ValueError("record start hour must be a non-negative integer")
    (ssh, u, v), source = stored_state_at(
        pinned_sources(chunk_roots), objective, absolute_hour)
    ssh, report = start_rebalancer(card)(ssh, u, v)
    return (ssh, u, v), {
        "absolute_hour": absolute_hour, "objective": objective,
        "operator": "box" if objective == "symmetric" else "filtered",
        "chunk_root": source.root,
        "chunk_manifest_sha256": source.manifest_sha256,
        "rebalance": {
            "rule": "stored state at the start hour, sea level rebalanced "
                    "(build_start_rebalancer), as every training window start",
            "tolerance": START_REBALANCE_TOLERANCE,
            "max_iterations": START_REBALANCE_MAX_ITERATIONS,
            "report": report}}


# The recorder's template filter is a Gaussian of one coarse cell (its sigma
# is the target spacing over the source spacing, in source cells).
STORED_TARGET_FILTER_SIGMA_CELLS = 1.0


def require_target_filter_sigma_cells(sigma_cells, objective: str) -> float:
    """Validate ``dataset.target_filter_sigma_cells``: a number >= 1.

    Only the ``template`` objective may widen the stored sigma-one filter.
    """
    if type(sigma_cells) not in (int, float) or not math.isfinite(sigma_cells) \
            or sigma_cells < STORED_TARGET_FILTER_SIGMA_CELLS:
        raise ValueError(
            "target_filter_sigma_cells must be a finite number >= "
            f"{STORED_TARGET_FILTER_SIGMA_CELLS}, got {sigma_cells!r}")
    if sigma_cells > STORED_TARGET_FILTER_SIGMA_CELLS \
            and objective != "template":
        raise ValueError(
            "target_filter_sigma_cells > 1 is defined for the template "
            f"objective only, not {objective!r}")
    return float(sigma_cells)


def build_target_filter(card, *, sigma_cells: float):
    """Widen the stored sigma-one-cell template state to ``sigma_cells``.

    Gaussians compose in variance, so the stored state (sigma one coarse
    cell) gets one more pass of ``sqrt(sigma_cells**2 - 1)`` coarse cells.
    The pass is the recorder's own filter (the in-scan filtered restrictor
    of ``swg_coarse_graining``: volume and face-transport extensives, even
    scalar/tangential and odd normal wall parity, 6-sigma kernel) run with
    the coarse grid as both source and target (nesting factor one), so no
    wall rule is re-implemented.  Returns ``filter(ssh, u, v)`` on native
    float64 arrays; under the rigid lid sea level is returned exactly zero,
    as the recorder returns it.
    """
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError(
            "the target filter requires JAX_ENABLE_X64=1; refusing a silent "
            "float64 downcast")
    if not math.isfinite(sigma_cells) \
            or sigma_cells <= STORED_TARGET_FILTER_SIGMA_CELLS:
        raise ValueError("the extra target filter needs sigma_cells > 1")
    extra = math.sqrt(sigma_cells**2 - STORED_TARGET_FILTER_SIGMA_CELLS**2)
    coarse = cgrid_geometry(card.resolved_namelist)
    restrict = jax.jit(build_jax_filtered_restrictor(
        coarse, coarse, sigma_source_cells=extra))
    lid = _is_rigid_lid(card)

    def widen(ssh, u, v):
        ssh_f, u_f, v_f = (np.asarray(value, dtype=np.float64) for value in
                           restrict(jnp.asarray(ssh), jnp.asarray(u),
                                    jnp.asarray(v)))
        return (np.zeros_like(ssh_f) if lid else ssh_f), u_f, v_f

    return widen


def build_lid_start_projection(card):
    """Non-divergent projection of native start velocities (rigid lid).

    The stepper's own ``swe_rigid_lid_projection``; ``project(u, v)``.
    """
    lid = swe_rigid_lid_geometry(card.recipe.grid, card.recipe.z_coord)
    project = jax.jit(lambda u, v: swe_rigid_lid_projection(u, v, lid)[:2])

    def apply(u, v):
        return tuple(np.asarray(value, dtype=np.float64)
                     for value in project(jnp.asarray(u), jnp.asarray(v)))

    return apply


@dataclass(frozen=True)
class SWGDatasetContract:
    """Validated, split-safe view of content-pinned hourly state records.

    The constructor, ``from_json`` and ``dataclasses.replace`` all run the
    same validation (``__post_init__``).  Every window starts from the stored
    state at its start hour with its sea level rebalanced by the registered
    rule; its targets are the stored right endpoints of the same operator.

    ``target_filter_sigma_cells`` (>= 1) widens the template state: 1 serves
    the stored sigma-one-cell state unchanged (a contract file with no key
    means 1); a larger value filters every served hour once more on the
    coarse grid (``build_target_filter``, once at ``preload_memory`` when
    cached, else per read) and, under the rigid lid, projects window-start
    velocities non-divergent.  Only the template objective may widen.
    """

    objective: str
    splits: tuple[tuple[str, int, int], ...]
    sources: tuple[PinnedHourlySource, ...]
    contract_path: str
    contract_sha256: str
    batch_size: int
    card: object = field(repr=False, compare=False)
    target_filter_sigma_cells: float
    _rebalance: object = field(
        init=False, repr=False, compare=False, default=None)
    _target_filter: object = field(
        init=False, repr=False, compare=False, default=None)
    _start_projection: object = field(
        init=False, repr=False, compare=False, default=None)
    _issued: dict = field(
        init=False, repr=False, compare=False, default_factory=dict)
    _memory: dict | None = field(
        init=False, repr=False, compare=False, default=None)

    def __post_init__(self):
        if self.objective not in OBJECTIVE_OPERATORS:
            raise ValueError("dataset objective must be symmetric or template")
        require_target_filter_sigma_cells(
            self.target_filter_sigma_cells, self.objective)
        if self.batch_size != TEMPLATE_BATCH_SIZE:
            raise ValueError("dataset contract must use batch size 10")
        if tuple(name for name, _, _ in self.splits) \
                != ("train", "validation", "test"):
            raise ValueError(
                "dataset requires train, validation, and test splits")
        for name, start, end in self.splits:
            if type(start) is not int or type(end) is not int or start >= end:
                raise ValueError(f"invalid {name} split bounds")
        for (_, _, left_end), (_, right_start, _) in zip(
                self.splits[:-1], self.splits[1:], strict=True):
            if right_start - left_end < DATASET_GAP_HOURS:
                raise ValueError(
                    "dataset splits require a gap of at least 30 days")
        if not self.sources:
            raise ValueError("dataset contract has no content-pinned sources")
        if len({(source.root, source.manifest_sha256)
                for source in self.sources}) != len(self.sources):
            raise ValueError("dataset source membership is duplicated")
        for source in self.sources:
            if _pinned_source(Path(source.root), source.manifest_sha256) \
                    != source:
                raise ValueError("dataset source differs from its pinned manifest")
        if len({source.truth_run for source in self.sources}) != 1:
            raise ValueError("dataset sources come from more than one truth run")
        if len({source.surface for source in self.sources}) != 1:
            raise ValueError("dataset sources come from more than one surface")
        if self.sources[0].surface != self.card.recipe.model_config.swe_surface:
            raise ValueError(
                f"dataset surface {self.sources[0].surface!r} differs from "
                "the card's surface "
                f"{self.card.recipe.model_config.swe_surface!r}")
        self._validate_coverage()
        grid = tuple(self.card.recipe.initial_state.eta.data.shape)
        if self._stored_state(self.splits[0][1])[0].shape != grid:
            raise ValueError("dataset grid differs from the coarse card")
        object.__setattr__(self, "_rebalance", start_rebalancer(self.card))
        if self.target_filter_sigma_cells > STORED_TARGET_FILTER_SIGMA_CELLS:
            object.__setattr__(self, "_target_filter", build_target_filter(
                self.card, sigma_cells=self.target_filter_sigma_cells))
            if _is_rigid_lid(self.card):
                object.__setattr__(
                    self, "_start_projection",
                    build_lid_start_projection(self.card))

    @staticmethod
    def from_json(path: Path, *, card) -> SWGDatasetContract:
        payload = json.loads(path.read_text())
        if payload.get("schema") != "legoesm-swg-online-dataset-v1":
            raise ValueError("unsupported SWG dataset contract schema")
        if tuple(payload.get("window_days", ())) != ADMITTED_WINDOWS_DAYS:
            raise ValueError("dataset contract must register 1/2/4/8-day windows")
        raw_splits = payload.get("splits", {})
        if set(raw_splits) != {"train", "validation", "test"}:
            raise ValueError("dataset requires train, validation, and test splits")
        splits = []
        for name in ("train", "validation", "test"):
            bounds = raw_splits[name]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"invalid {name} split bounds")
            splits.append((name, bounds[0], bounds[1]))
        sources = payload.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("dataset contract has no content-pinned sources")
        pins = []
        for source in sources:
            if not isinstance(source, dict) \
                    or set(source) != {"root", "manifest_sha256"}:
                raise ValueError("dataset source membership is ambiguous")
            pins.append(_pinned_source(
                Path(source["root"]), source["manifest_sha256"]))
        return SWGDatasetContract(
            objective=payload.get("objective"), splits=tuple(splits),
            sources=tuple(pins), contract_path=str(path.resolve()),
            contract_sha256=_sha256(path),
            batch_size=payload.get("batch_size"), card=card,
            # Contracts written before the widened filter carry no key; their
            # targets are the stored sigma-one-cell state.
            target_filter_sigma_cells=payload.get(
                "target_filter_sigma_cells", STORED_TARGET_FILTER_SIGMA_CELLS))

    def _validate_coverage(self) -> None:
        ordered = sorted(self.sources, key=lambda source: source.first_absolute_hour)
        for left, right in zip(ordered[:-1], ordered[1:], strict=True):
            if right.first_absolute_hour < left.last_absolute_hour:
                raise ValueError("content-pinned hourly sources overlap ambiguously")
            if right.first_absolute_hour == left.last_absolute_hour:
                # The shared restart hour is stored twice: left's last
                # in-scan record and right's eagerly restricted t0.  Both
                # restrict one fine state, so they agree to round-off; more
                # than that means the chunks are not one adjacent record.
                for end_value, start_value in zip(
                        self._record(left, left.last_absolute_hour),
                        self._initial(right), strict=True):
                    scale = max(float(np.max(np.abs(end_value))),
                                float(np.max(np.abs(start_value))))
                    if np.max(np.abs(end_value - start_value)) \
                            > _ADJACENT_HOUR_RELATIVE_ATOL * scale:
                        raise ValueError(
                            "adjacent hourly chunks disagree at restart hour")
        for name, start, end in self.splits:
            covered_through = start
            for source in ordered:
                if source.last_absolute_hour < covered_through:
                    continue
                if source.first_absolute_hour > covered_through:
                    break
                covered_through = max(
                    covered_through, source.last_absolute_hour)
                if covered_through >= end:
                    break
            if covered_through < end:
                raise ValueError(
                    f"{name} split is not a contiguous pinned hourly record")

    def _record(self, source: PinnedHourlySource, hour: int):
        return _record_at(source, self.objective, hour)

    def _initial(self, source: PinnedHourlySource):
        return _pinned_day(source, self.objective, 0)[1]

    def _served_state(self, hour: int):
        """The stored state at ``hour``, widened when the contract says so."""
        state = stored_state_at(self.sources, self.objective, hour)[0]
        if self._target_filter is None:
            return state
        return self._target_filter(*state)

    def _stored_state(self, hour: int):
        if self._memory is None:
            return self._served_state(hour)
        if hour not in self._memory:
            raise ValueError(
                f"absolute hour {hour} was not preloaded into memory; "
                "preload_memory covers only the registered windows and "
                "target interval")
        return self._memory[hour]

    def preload_memory(self, *, window_days, target_interval_hours: int) -> dict:
        """Hold every stored hour that ``batches`` can read in host memory.

        Each hour is read once through ``stored_state_at`` (the digest-pinned
        day files, so every admission check still runs) and copied, so
        ``batches`` then serves identical float64 arrays without file reads.
        The hours are the union, over the given window lengths, of the window
        starts (training: every offset; held-out: the fixed lattice) and
        their targets ``target_interval_hours, 2 * ..., 24 * w`` later.  A
        batch needing any other hour raises instead of reading the files.
        Returns ``{"hours": count, "bytes": array bytes held}``.
        """
        window_days = tuple(window_days)
        if not window_days or any(
                window not in ADMITTED_WINDOWS_DAYS for window in window_days):
            raise ValueError("preload windows must be from 1, 2, 4, or 8 days")
        interval = require_target_interval_hours(target_interval_hours)
        wanted = []
        for name, start_bound, end_bound in self.splits:
            for window in window_days:
                window_hours = HOURS_PER_DAY * window
                leads = np.arange(interval, window_hours + 1, interval)
                for offset in range(window_hours if name == "train" else 1):
                    starts = np.arange(start_bound + offset,
                                       end_bound - window_hours + 1,
                                       window_hours, dtype=np.int64)
                    wanted += [starts, (starts[:, None] + leads).ravel()]
        memory = {}
        for hour in np.unique(np.concatenate(wanted)):
            state = tuple(np.array(value) for value in
                          self._served_state(int(hour)))
            for value in state:
                value.flags.writeable = False
            memory[int(hour)] = state
        object.__setattr__(self, "_memory", memory)
        return {"hours": len(memory), "bytes": sum(
            value.nbytes for state in memory.values() for value in state)}

    def start_state(self, absolute_hour: int):
        """One window start: the stored state at ``absolute_hour`` with its
        sea level rebalanced by the registered rule; ``((ssh, u, v), report)``."""
        ssh, u, v = self._stored_state(absolute_hour)
        if self._start_projection is not None:
            u, v = self._start_projection(u, v)
        ssh, report = self._rebalance(ssh, u, v)
        return (ssh, u, v), report

    def _split_bounds(self, split: str) -> tuple[int, int]:
        for name, start, end in self.splits:
            if name == split:
                return start, end
        raise ValueError(f"unknown or leaked dataset split {split!r}")

    def batches(self, *, split: str, window_days: int,
                target_interval_hours: int,
                epoch: int | None = None, seed: int | None = None):
        """Yield batches of ten non-overlapping windows inside one split.

        The record is hourly; the targets are every ``target_interval_hours``-th
        stored hour after the window start (hours ``k*interval`` after the
        start, ``k = 1..24w/interval``), so 1 keeps every hour.

        Training draws one starting offset per (seed, window length, epoch)
        and shuffles the windows, as the template redraws both every epoch.
        Held-out splits use one fixed chronological lattice from the split's
        first hour, so ``epoch`` and ``seed`` must be None.  An incomplete
        final batch is dropped; a split without one full batch raises.
        """
        start_bound, end_bound = self._split_bounds(split)
        if not window_days_admitted(window_days):
            raise ValueError(
                f"window must be one of {ADMITTED_WINDOWS_DAYS} days or a "
                f"longer whole number of days, got {window_days!r}")
        require_target_interval_hours(target_interval_hours)
        window_hours = HOURS_PER_DAY * window_days
        if split == "train":
            if type(epoch) is not int or epoch < 0 \
                    or type(seed) is not int or seed < 0:
                raise ValueError(
                    "training batches need non-negative integer epoch and seed")
            rng = np.random.default_rng(
                np.random.SeedSequence((seed, window_days, epoch)))
            offset = int(rng.integers(0, window_hours))
        elif epoch is not None or seed is not None:
            raise ValueError(
                "held-out splits use one fixed lattice; epoch and seed must be None")
        else:
            rng, offset = None, 0
        starts = np.arange(
            start_bound + offset, end_bound - window_hours + 1,
            window_hours, dtype=np.int64)
        if rng is not None:
            rng.shuffle(starts)
        usable = (starts.size // self.batch_size) * self.batch_size
        if usable == 0:
            raise ValueError(
                f"{split} split holds {starts.size} {window_days}-day windows, "
                f"fewer than one batch of {self.batch_size}")
        for first in range(0, usable, self.batch_size):
            batch_starts = starts[first:first + self.batch_size]
            start_states = [self.start_state(int(start))[0]
                            for start in batch_starts]
            initial = tuple(np.stack([state[index] for state in start_states])
                            for index in range(3))
            hourly = [[self._stored_state(int(start) + lead)
                       for lead in range(
                           target_interval_hours, window_hours + 1,
                           target_interval_hours)]
                      for start in batch_starts]
            target = tuple(np.stack([np.stack([state[index] for state in window])
                                     for window in hourly]) for index in range(3))
            batch = WindowBatch(initial, target, batch_starts, split, window_days)
            self._check_bounds(batch)
            self._issued[self._digest(batch)] = (split, window_days)
            yield batch

    def _check_bounds(self, batch: WindowBatch) -> None:
        start_bound, end_bound = self._split_bounds(batch.split)
        if not window_days_admitted(batch.window_days):
            raise ValueError(
                "dataset batch window is not one of "
                f"{ADMITTED_WINDOWS_DAYS} days or a longer whole number of days")
        window_hours = HOURS_PER_DAY * batch.window_days
        starts = np.asarray(batch.start_absolute_hour, dtype=np.int64)
        if starts.shape != (self.batch_size,):
            raise ValueError("dataset batch does not contain exactly ten windows")
        if np.any(starts < start_bound) or np.any(starts + window_hours > end_bound):
            raise ValueError("dataset batch contains a leaked or crossing hour")

    def _digest(self, batch: WindowBatch) -> str:
        digest = hashlib.blake2b(digest_size=32)
        digest.update(self.contract_sha256.encode())
        digest.update(np.asarray([batch.window_days], dtype=np.int64).tobytes())
        for value in (batch.start_absolute_hour,) + tuple(batch.initial) \
                + tuple(batch.target):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def validate_batch_membership(self, batch: WindowBatch) -> None:
        """Refuse any batch this contract did not issue under its split label.

        Start hours are checked against the split bounds; the arrays, start
        hours and window length are bound by a content digest to the split
        they were issued for, so held-out arrays under training labels, a
        relabelled batch, an edited array and a hand-built batch all raise.
        """
        self._check_bounds(batch)
        if self._issued.get(self._digest(batch)) \
                != (batch.split, batch.window_days):
            raise ValueError(
                "dataset batch was not issued by this contract for its split")


def coarse_state_from_native(state, ssh, u, v):
    """Replace one SWG state's prognostics by native T/U/V arrays (traceable)."""
    u_full = jnp.zeros(state.u.data.shape, dtype=jnp.float64).at[
        :, 1:, 0].set(u)
    v_full = jnp.zeros(state.v.data.shape, dtype=jnp.float64).at[
        1:, :, 0].set(v)
    return state._replace(
        eta=state.eta.replace(data=jnp.asarray(ssh, dtype=jnp.float64)),
        u=state.u.replace(data=u_full), v=state.v.replace(data=v_full))


def _is_rigid_lid(card) -> bool:
    return card.recipe.model_config.swe_surface == "rigid_lid"


def require_rigid_lid_zero_ssh(card, ssh) -> None:
    """Refuse a stored state with non-zero sea level on a rigid-lid card.

    Eager (concrete arrays only).  A no-op on an explicit card.
    """
    if _is_rigid_lid(card) and np.any(np.asarray(ssh) != 0.0):
        raise ValueError(
            "a rigid-lid stored state must have identically zero ssh")


def rigid_lid_start_state(card, state):
    """``state`` with ``rigid_lid_phi`` set to what the stepper would compute.

    Under the lid a stored (ssh, u, v) carries no pressure potential, which
    the closure reads as ``phi/g`` ([m], gravity ``constants.g``) in place of
    sea level.  The stored state's potential is the stage-1 projection
    potential of the certified stage at that state
    (``swe_rigid_lid_stage1_phi``: the stepper's own stage-1 computation).
    Traceable; explicit cards are returned unchanged.
    """
    if not _is_rigid_lid(card):
        return state
    tau_u, tau_v = swg_surface_wind(card.resolved_namelist)
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau_u), tau_j_native=jnp.asarray(tau_v))
    phi = swe_rigid_lid_stage1_phi(
        state, card.dt_s, forcing, card.recipe.grid, card.recipe.z_coord,
        card.recipe.model_config)
    return state._replace(rigid_lid_phi=phi)


def start_state_from_native(card, ssh, u, v):
    """The card's state from stored native (ssh, u, v), eager.

    Under the lid: refuses non-zero ssh and carries the derived potential
    (``rigid_lid_start_state``); explicit cards get ``coarse_state_from_native``
    of the card's initial state exactly.
    """
    require_rigid_lid_zero_ssh(card, ssh)
    return rigid_lid_start_state(card, coarse_state_from_native(
        card.recipe.initial_state, ssh, u, v))


def _network_entry(entry, state, card):
    """The closure's stage-entry view of ``state``'s offline entry.

    Explicit: the entry itself (network input = ssh).  Rigid lid: ssh is
    replaced by the equivalent sea level ``rigid_lid_phi / g``, exactly what
    ``swe_rk3_step`` shows the closure at stage 1.
    """
    if not _is_rigid_lid(card):
        return entry
    if state.rigid_lid_phi is None:
        raise ValueError("a rigid-lid state needs rigid_lid_phi "
                         "(rigid_lid_start_state)")
    require_rigid_lid_zero_ssh(card, state.eta.data)
    return entry._replace(ssh_k=jnp.asarray(state.rigid_lid_phi)
                          / card.recipe.model_config.constants.g)


def load_content_pinned_initial_state(
        card, path: Path, *, manifest_path: Path,
        expected_manifest_sha256: str):
    """Restore the conservative state at t0 from its pinned snapshot stream."""
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("conservative-state manifest differs from registered split")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" \
            or manifest.get("sampling") != "daily_snapshot":
        raise ValueError("conservative initial-state stream is incomplete")
    if _sha256(path) != manifest.get("output_daily_sha256", {}).get(path.name):
        raise ValueError("conservative initial-state digest differs")
    with np.load(path, allow_pickle=False) as saved:
        ssh = np.asarray(saved["ssh_m"], dtype=np.float64)
        u_native = np.asarray(saved["u_m_s"], dtype=np.float64)
        v_native = np.asarray(saved["v_m_s"], dtype=np.float64)
        if int(saved["day"]) != int(path.stem[3:]):
            raise ValueError("conservative t0 day differs from its file name")
    state = card.recipe.initial_state
    if ssh.shape != tuple(state.eta.data.shape):
        raise ValueError("conservative t0 shape differs from coarse card")
    return coarse_state_from_native(state, ssh, u_native, v_native)


def closed_wall_divergent_part(
        u: np.ndarray, v: np.ndarray, mask: np.ndarray, spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-wall gradient part of native U/V and its potential (offline).

    Solves the Neumann problem ``lap(phi) = div(u, v)`` on the wet rectangle
    with a DCT-II (zero-mean ``phi``) and returns ``(phi, u_div, v_div)``:
    ``phi`` on wet T points (zero on land) and its discrete gradient on the
    native faces, with zero wall-normal flux, so ``(u - u_div, v - v_div)``
    is discretely divergence-free.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    wet = np.asarray(mask, dtype=bool)
    if u.shape != wet.shape or v.shape != wet.shape:
        raise ValueError("native U/V and T mask shapes differ")
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("projection spacing must be positive finite")
    j0, j1, i0, i1 = wet_rectangle_bounds(wet)
    divergence = (
        u - np.roll(u, 1, axis=1)
        + v - np.roll(v, 1, axis=0)
    ) / spacing_m
    rhs = divergence[j0:j1 + 1, i0:i1 + 1]
    rhs = rhs - np.mean(rhs, dtype=np.float64)
    ny, nx = rhs.shape
    ky = np.arange(ny, dtype=np.float64)
    kx = np.arange(nx, dtype=np.float64)
    eigenvalue = (
        2.0 * (np.cos(np.pi * ky / ny) - 1.0)[:, None]
        + 2.0 * (np.cos(np.pi * kx / nx) - 1.0)[None, :]
    ) / spacing_m ** 2
    rhs_hat = dctn(rhs, type=2, norm="ortho")
    phi_hat = np.zeros_like(rhs_hat)
    phi_hat[1:, :] = rhs_hat[1:, :] / eigenvalue[1:, :]
    phi_hat[0, 1:] = rhs_hat[0, 1:] / eigenvalue[0, 1:]
    phi = idctn(phi_hat, type=2, norm="ortho")

    phi_full = np.zeros_like(u)
    phi_full[j0:j1 + 1, i0:i1 + 1] = phi
    u_div = np.zeros_like(u)
    v_div = np.zeros_like(v)
    u_div[j0:j1 + 1, i0:i1] = np.diff(phi, axis=1) / spacing_m
    v_div[j0:j1, i0:i1 + 1] = np.diff(phi, axis=0) / spacing_m
    return phi_full, u_div, v_div


def _t_centred_rms(u: np.ndarray, v: np.ndarray, wet: np.ndarray) -> float:
    """RMS over wet T points of the T-centred native face velocity."""
    u_centre = 0.5 * (u + np.roll(u, 1, axis=1))
    v_centre = 0.5 * (v + np.roll(v, 1, axis=0))
    return float(np.sqrt(np.mean(
        u_centre[wet] ** 2 + v_centre[wet] ** 2, dtype=np.float64)))


def build_start_rebalancer(card, *, tolerance: float, max_iterations: int):
    """Sea-level-only balance of a coarse start: offline data preparation.

    Returns ``rebalance(ssh, u, v) -> (ssh_balanced, report)``.  Velocities
    are never changed and land SSH is untouched.  Each iteration takes the
    card's own first-step velocity increment with no closure and the steady
    wind, ``a = (u(dt) - u(0)) / dt``, splits off its closed-wall divergent
    part ``grad(phi)``, and stops when that part's T-centred rms is at most
    ``tolerance`` times the remainder's; otherwise it adds
    ``(phi - <phi>) / g`` to wet SSH, ``<.>`` the area mean, so total volume
    is unchanged.  No convergence after ``max_iterations`` updates raises.
    Nothing here is differentiated: the start state is data.
    """
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rebalancing tolerance must be positive finite")
    if type(max_iterations) is not int or max_iterations < 0:
        raise ValueError("rebalancing iteration cap must be a non-negative integer")
    geometry = build_closure_geometry(card)
    wet = np.asarray(geometry.tmask, dtype=bool)
    area = np.asarray(geometry.area_t_m2, dtype=np.float64)[wet]
    gravity = float(card.recipe.model_config.constants.g)
    dt_s = float(card.dt_s)
    tau_u, tau_v = swg_surface_wind(card.resolved_namelist)
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau_u, dtype=jnp.float64),
        tau_j_native=jnp.asarray(tau_v, dtype=jnp.float64))
    template = card.recipe.initial_state._replace(
        tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
    grid = tuple(template.eta.data.shape)

    @jax.jit
    def first_step_velocity(ssh, u, v):
        stepped = swg_online_step(
            None, coarse_state_from_native(template, ssh, u, v), None, None,
            None, card=card, geometry=geometry, forcing=forcing)
        return stepped.u.data[:, 1:, 0], stepped.v.data[1:, :, 0]

    def rebalance(ssh, u, v):
        ssh = np.array(ssh, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if any(value.shape != grid for value in (ssh, u, v)) \
                or not all(np.all(np.isfinite(value)) for value in (ssh, u, v)):
            raise ValueError(f"start state must be three finite {grid} arrays")
        start = ssh.copy()
        history = []
        for iteration in range(max_iterations + 1):
            u_next, v_next = (np.asarray(value, dtype=np.float64)
                              for value in first_step_velocity(ssh, u, v))
            accel_u = (u_next - u) / dt_s
            accel_v = (v_next - v) / dt_s
            phi, div_u, div_v = closed_wall_divergent_part(
                accel_u, accel_v, wet, geometry.spacing_m)
            divergent = _t_centred_rms(div_u, div_v, wet)
            rotational = _t_centred_rms(accel_u - div_u, accel_v - div_v, wet)
            history.append([divergent, rotational])
            if divergent <= tolerance * rotational:   # 0 <= 0 at exact rest
                return ssh, {
                    "iterations": iteration,
                    "divergent_to_rotational_rms": (
                        divergent / rotational if rotational > 0.0 else 0.0),
                    "divergent_rotational_rms_history_m_s2": history,
                    "volume_change_m3": float(np.sum((ssh - start)[wet] * area)),
                }
            if iteration == max_iterations:
                break
            correction = phi[wet] / gravity
            ssh[wet] += correction - np.sum(correction * area) / np.sum(area)
        raise ValueError(
            f"start-state sea-level rebalancing did not converge in "
            f"{max_iterations} updates: divergent {divergent:.3e} > "
            f"{tolerance:.1e} x rotational {rotational:.3e} m s-2 (rms)")

    return rebalance


def initial_closure_acceleration_ratio(
        network: SWGScalarPotentialCNN, states: tuple,
        card, geometry: SWGClosureGeometry, *, key) -> dict:
    """Measure initialized closure acceleration against ``tau0/(rho0 H)``."""
    from legoesm.ocean.fidelity.swg_coarse_graining import (
        cgrid_geometry,
        live_face_thickness,
    )

    native_geometry = cgrid_geometry(card.resolved_namelist)
    tau = swg_surface_wind(card.resolved_namelist)
    ratios = []
    accelerations = []
    for index, state in enumerate(states):
        entry = _entry_from_state(
            state, tau, tau, card.recipe.grid, card.recipe.z_coord,
            card.recipe.model_config)
        net_entry = _network_entry(entry, state, card)
        potential = network(
            net_entry, geometry, key=jax.random.fold_in(key, index),
            noise=jnp.zeros_like(entry.ssh_k), noise_scale=jnp.asarray(0.0))
        eta = np.asarray(entry.ssh_k, dtype=np.float64)
        h_u = jnp.asarray(live_face_thickness(eta, native_geometry, "U"))
        h_v = jnp.asarray(live_face_thickness(eta, native_geometry, "V"))
        force_u, force_v = swe_rotated_potential_forcing(
            potential, geometry.metric_u_m, geometry.metric_v_m,
            h_u, h_v, geometry.umask, geometry.vmask,
            geometry.vertex_interior)
        acceleration = float(max(
            jnp.max(jnp.abs(force_u)), jnp.max(jnp.abs(force_v))))
        reference_depth = float(np.median(
            np.asarray(card.recipe.z_coord.nemo_e3t_0)[
                np.asarray(card.recipe.land_mask, dtype=bool)]))
        wind_acceleration = (
            card.resolved_namelist.rn_tau_pa
            / card.recipe.model_config.constants.rho_0 / reference_depth)
        ratios.append(acceleration / wind_acceleration)
        accelerations.append(acceleration)
    return {
        "per_state_ratio": ratios,
        "per_state_max_acceleration_m_s2": accelerations,
        "maximum_ratio": max(ratios),
        "bar": 0.01,
        "passes": max(ratios) <= 0.01,
    }


def eq7_point_weights(geometry: SWGClosureGeometry, *, wall_ring: str):
    """Eq. 7's diagonal ``W``: ``dx_C^2`` at weighted interior F points, else 0.

    ``wall_ring="weighted"`` is Eq. 7 as written: every interior F point,
    the first ring next to the walls included, and zero on the closed-wall
    vertices and dry points.  ``wall_ring="excluded"`` also zeroes the first
    interior ring (the owner's pending decision on the filtered target's
    wall imbalance).  There is no default; any other value raises.
    """
    if wall_ring not in WALL_RING_OPTIONS:
        raise ValueError(
            f"wall_ring must be explicitly selected from {WALL_RING_OPTIONS}")
    mask = geometry.vertex_interior
    if wall_ring == "excluded":
        j0, j1, i0, i1 = geometry.wet_bounds
        mask = mask & jnp.zeros_like(mask).at[j0 + 1:j1 - 1, i0 + 1:i1 - 1].set(
            True)
    return geometry.spacing_m**2 * mask


def maddison_half_width_m(basin_length_m: float,
                          geometry: SWGClosureGeometry) -> float:
    """Eq. 7's ``L``: half the deck's square-basin side ``rn_domsiz_m = 2L``.

    Maddison's Table 1 gives "Domain size 2L 4,000 km" with the basin
    ``[-L, L]^2``, so ``1/(4 L^2)`` is one over the basin area.  The argument
    must be the full side the namelist registers; it is checked against the
    wet rectangle of the card's own grid, so the half-width or any other
    literal raises instead of rescaling the loss.
    """
    j0, j1, i0, i1 = geometry.wet_bounds
    sides = ((j1 - j0 + 1) * geometry.spacing_m,
             (i1 - i0 + 1) * geometry.spacing_m)
    if not np.isfinite(basin_length_m) or not all(
            math.isclose(basin_length_m, side, rel_tol=1.0e-12, abs_tol=0.0)
            for side in sides):
        raise ValueError(
            "basin_length_m must be the deck's full square-basin side "
            f"rn_domsiz_m (= 2L); the card's wet rectangle is {sides} m")
    return 0.5 * float(basin_length_m)


def balanced_trajectory_loss(predicted: BalancedTrajectory,
                             target: BalancedTrajectory,
                             geometry: SWGClosureGeometry,
                             normalizers: BalancedNormalizers | None, *,
                             kind: str,
                             beta_m1_s1: float | None = None,
                             basin_length_m: float | None = None,
                             wall_ring: str | None = None):
    """Maddison Eq. 7 point loss or Brolly per-lead energy score.

    ``point`` is Eq. 7 exactly, ``1e4/(beta^2 L^2) /(4 L^2)`` times the mean
    over windows and hours of ``(zeta - zeta_C)^T W (zeta - zeta_C)``; it
    takes no data normalizers (pass None).  For the energy score every
    predicted array has shape ``(members, hours, y, x)`` and targets omit the
    member axis; it is the proper multivariate score over all three
    normalized, area-weighted balanced fields, requires S >= 2, and its pair
    term uses the unbiased ``2 S (S-1)`` denominator.
    """
    if kind == "point":
        if normalizers is not None:
            raise ValueError("Eq. 7 point loss takes no data normalizers")
        if beta_m1_s1 is None or basin_length_m is None:
            raise ValueError("Maddison point loss requires beta and basin length")
        if not np.isfinite(beta_m1_s1) or beta_m1_s1 <= 0.0:
            raise ValueError("Maddison beta must be positive finite")
        half_width = maddison_half_width_m(basin_length_m, geometry)
        weights = eq7_point_weights(geometry, wall_ring=wall_ring)
        if predicted.relative_vorticity_s1.shape \
                != target.relative_vorticity_s1.shape \
                or predicted.relative_vorticity_s1.ndim != 4:
            raise ValueError(
                "point loss requires matching (windows,hours,y,x) arrays")
        zeta_error = (predicted.relative_vorticity_s1
                      - target.relative_vorticity_s1)
        weighted_sum = jnp.sum(zeta_error**2 * weights, axis=(-2, -1))
        scale = (1.0e4 / (beta_m1_s1**2 * half_width**2)
                 / (4.0 * half_width**2))
        return scale * jnp.mean(weighted_sum)
    if kind != "energy_score":
        raise ValueError(f"unknown SWG balanced loss {kind!r}")
    if normalizers is None:
        raise ValueError("energy_score requires explicit normalizers")
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
    members = pred.shape[0]
    if members < 2:
        raise ValueError("energy_score requires at least two ensemble members")
    def lead_distance(left, right):
        return jnp.sqrt(jnp.sum(
            (left - right) ** 2 * weight[:, :, None], axis=(-3, -2, -1)))

    observation_distance = jax.vmap(
        lambda member: jax.vmap(lead_distance)(member, obs))(pred)
    pairwise = jax.vmap(lambda left: jax.vmap(
        lambda right: jax.vmap(lead_distance)(left, right))(pred))(pred)
    score_by_lead = (jnp.mean(observation_distance, axis=0)
                     - jnp.sum(pairwise, axis=(0, 1))
                     / (2.0 * members * (members - 1)))
    return jnp.mean(score_by_lead)


def deterministic_trajectory_loss(
        predicted_native, target_native, geometry: SWGClosureGeometry, *,
        objective: str, beta_m1_s1: float, basin_length_m: float,
        wall_ring: str):
    """Exact Eq. 7 batch/hour reduction for one explicitly selected objective.

    Model and target native states go through ONE diagnostic (the selected
    objective's), so the loss is zero when the model equals the target.
    ``basin_length_m`` is the deck's full side ``rn_domsiz_m``; ``wall_ring``
    has no default (see ``eq7_point_weights``).
    """
    if objective not in OBJECTIVE_OPERATORS:
        raise ValueError(
            f"objective must be explicitly selected from {OBJECTIVE_OPERATORS}")
    if len(predicted_native) != 3 or len(target_native) != 3:
        raise ValueError("deterministic loss requires native T/U/V tuples")
    if predicted_native[0].ndim != 4 \
            or any(value.shape != predicted_native[0].shape
                   for value in predicted_native) \
            or any(value.shape != predicted_native[0].shape
                   for value in target_native):
        raise ValueError(
            "deterministic native arrays must share (windows,hours,y,x)")

    def diagnose_one(ssh, u, v):
        return diagnose_native_trajectory(
            (ssh, u, v), geometry, objective=objective)

    predicted = jax.vmap(diagnose_one)(*predicted_native)
    target = jax.vmap(diagnose_one)(*target_native)
    return balanced_trajectory_loss(
        predicted, target, geometry, None, kind="point",
        beta_m1_s1=beta_m1_s1, basin_length_m=basin_length_m,
        wall_ring=wall_ring)


def swg_online_step(network, state, stage_noise, key, noise_scale, *,
                    card, geometry: SWGClosureGeometry, forcing):
    """Run one certified RK3 step with an optional traced closure term."""
    if network is None:
        return swe_rk3_step(
            state, card.dt_s, forcing, card.recipe.grid,
            card.recipe.z_coord, card.recipe.model_config,
            stage_potential_fn=None)
    expected = (3,) + tuple(geometry.tmask.shape)
    if stage_noise.shape != expected:
        raise ValueError(
            f"stage noise shape {stage_noise.shape}, expected {expected}")

    def potential(entry, stage):
        return network(
            entry, geometry, key=jax.random.fold_in(key, stage),
            noise=stage_noise[stage - 1], noise_scale=noise_scale)

    return swe_rk3_step(
        state, card.dt_s, forcing, card.recipe.grid, card.recipe.z_coord,
        card.recipe.model_config, stage_potential_fn=potential)


def require_target_interval_hours(value) -> int:
    """The spacing in hours of the stored states a window is scored on."""
    if type(value) is not int or value < 1 or HOURS_PER_DAY % value:
        raise ValueError(
            "target_interval_hours must be an integer >= 1 that divides 24, "
            f"got {value!r}")
    return value


def steps_per_target_interval(dt_s: float, target_interval_hours: int) -> int:
    """Model steps between two recorded states; refuses a step that does not
    divide the target interval exactly."""
    interval_s = require_target_interval_hours(
        target_interval_hours) * SECONDS_PER_HOUR
    steps = int(interval_s / dt_s)
    if steps < 1 or steps * dt_s != interval_s:
        raise ValueError(
            f"SWG timestep {dt_s} s must divide the "
            f"{target_interval_hours}-hour target interval exactly")
    return steps


def window_steps(card, window_days: int, target_interval_hours: int) -> int:
    """Model steps of one ``window_days`` training window."""
    return (HOURS_PER_DAY * window_days // target_interval_hours
            * steps_per_target_interval(card.dt_s, target_interval_hours))


def build_segment_fn(card, geometry: SWGClosureGeometry, *,
                     window_days: int, target_interval_hours: int,
                     checkpoint: bool = True):
    """Build one fixed-window training segment with closed steady wind.

    The segment records the model state every ``target_interval_hours`` hours
    (1 = hourly), ``24 * window_days / target_interval_hours`` records.
    """
    if not window_days_admitted(window_days):
        raise ValueError(
            f"window_days must be one of {ADMITTED_WINDOWS_DAYS} or a longer "
            f"whole number of days, got {window_days}")
    steps_per_record = steps_per_target_interval(
        card.dt_s, target_interval_hours)
    tau_u, tau_v = swg_surface_wind(card.resolved_namelist)
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau_u, dtype=jnp.float64),
        tau_j_native=jnp.asarray(tau_v, dtype=jnp.float64))
    n_records = HOURS_PER_DAY * window_days // target_interval_hours
    n_steps = n_records * steps_per_record

    def one_step(network, noise_scale, carry, inputs):
        step_key, stage_noise = inputs
        state = swg_online_step(
            network, carry, stage_noise, step_key, noise_scale,
            card=card, geometry=geometry, forcing=forcing)
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

        def one_record(state, record):
            start = record * steps_per_record
            record_keys = jax.lax.dynamic_slice_in_dim(
                keys, start, steps_per_record, axis=0)
            record_noise = jax.lax.dynamic_slice_in_dim(
                noise, start, steps_per_record, axis=0)
            state, _ = jax.lax.scan(
                lambda carry, values: scanned_step(
                    network, noise_scale, carry, values),
                state, (record_keys, record_noise))
            native = (state.eta.data, state.u.data[:, 1:, 0],
                      state.v.data[1:, :, 0])
            return state, native

        return jax.lax.scan(one_record, initial_state, jnp.arange(n_records))

    return raw, eqx.filter_jit(raw)


class SWGOptimizerSettings(NamedTuple):
    """Every optimiser choice of one SWG online run; no field has a default.

    The shared factory's defaults (warm-up, cosine decay, global-norm
    clipping, Optax epsilon 1e-8) are the historical SFNO recipe, not the
    template's, so each choice is named here and the strict wrapper below
    admits none of the shared ones by omission.
    """

    optimizer: str
    learning_rate: float
    schedule: str
    weight_decay: float
    adam_b1: float
    adam_b2: float
    adam_epsilon: float
    adam_epsilon_convention: str
    gradient_clipping: str


def create_swg_optimizer(settings: SWGOptimizerSettings):
    """AdamW at a constant rate through the shared factory, strictly.

    Only ``optimizer="adamw"``, ``schedule="constant"`` (no warm-up, no
    decay) and ``gradient_clipping="none"`` are admitted, and the epsilon
    convention must be named (``keras_epsilon_hat`` is the template's Keras
    AdamW).  A ``TrainingConfig``, whose unset fields silently carry the
    factory defaults, is refused, and a settings tuple with a missing field
    cannot be built.  The factory reads neither its warm-up/total step counts
    nor its clip norm on this path (constant schedule, clipping off).
    """
    if type(settings) is not SWGOptimizerSettings:
        raise TypeError(
            "SWG optimiser requires SWGOptimizerSettings with every field "
            f"stated, not {type(settings).__name__}")
    if settings.optimizer != "adamw":
        raise ValueError("SWG optimiser must be 'adamw'")
    if settings.schedule != "constant":
        raise ValueError(
            "SWG optimiser schedule must be 'constant' (no warm-up or decay)")
    if settings.gradient_clipping != "none":
        raise ValueError("SWG optimiser gradient_clipping must be 'none'")
    if settings.adam_epsilon_convention not in ("keras_epsilon_hat", "optax"):
        raise ValueError(
            "SWG optimiser epsilon convention must be 'keras_epsilon_hat' or "
            "'optax'")
    for name in ("learning_rate", "weight_decay", "adam_b1", "adam_b2",
                 "adam_epsilon"):
        value = getattr(settings, name)
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"SWG optimiser {name} must be a finite float")
    if settings.learning_rate <= 0.0 or settings.weight_decay < 0.0:
        raise ValueError(
            "SWG optimiser needs a positive rate and non-negative decay")
    return create_optimizer(TrainingConfig(
        lr=settings.learning_rate, warmup_steps=0,
        weight_decay=settings.weight_decay, optimizer="adamw",
        schedule="constant", clip_gradients=False,
        adam_epsilon=settings.adam_epsilon, adam_b1=settings.adam_b1,
        adam_b2=settings.adam_b2,
        adam_epsilon_convention=settings.adam_epsilon_convention))


def assert_network_gradients_live(grads) -> None:
    """Apply the repository no-inert gate to every trainable CNN leaf."""
    named = {
        jax.tree_util.keystr(path): value
        for path, value in jax.tree.leaves_with_path(
            eqx.filter(grads, eqx.is_inexact_array))
    }
    poisoned = sorted(
        name for name, value in named.items()
        if not bool(jnp.all(jnp.isfinite(value))))
    if poisoned:
        raise ValueError(
            "SWG network gradient contains non-finite elements: "
            f"{poisoned}")
    assert_no_inert(named)


# ---------------------------------------------------- trained-closure free run
# ONE schema for a trained closure's free run: the training driver
# (scripts/run/run_swg_online_training.py) writes it and the held-out scorer
# (nemo_testcase_l1_swg_online_evaluation.py) reads it, both from here.
# Every day file is a dual file of the double-gyre driver's daily-mean format
# (run_swg_double_gyre.py ``run_long``): the end-of-day snapshot beside the
# arithmetic mean of every post-step state of that model day, under these
# per-file labels; the scorer shows the sampling from the data.
DUAL_SCHEMA = "legoesm-swg-double-gyre-daily-v2"
SNAPSHOT_LABEL = "end-of-day instantaneous state"
DAILY_MEAN_LABEL = "arithmetic mean of every post-step state in model day"
SNAPSHOT_FIELDS = ("ssh_m", "u_m_s", "v_m_s")
DAILY_MEAN_FIELDS = ("ssh_daily_mean_m", "u_daily_mean_m_s", "v_daily_mean_m_s")
FREE_RUN_MANIFEST_SCHEMA = "legoesm-swg-online-free-run-v3"
FREE_RUN_EXECUTIONS = (("cpu", "libm"), ("cuda", "native"))
CUDA_AUTOTUNE_FLAG = "--xla_gpu_autotune_level=0"
FREE_RUN_CLOSURE_KEYS = (
    "kind", "run_id", "experiment", "config_sha256", "dataset_contract_sha256",
    "network_seed", "window_days", "objective", "wall_ring", "network_spec",
    "checkpoint_path", "checkpoint_sha256")
FREE_RUN_CONTROL_KEYS = ("kind", "control")


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and set(value) <= set("0123456789abcdef"))


def free_run_closure(closure) -> dict:
    """Validate the ``closure`` block of a trained-closure free-run manifest.

    The writer builds the block through this function and the reader admits
    it through ``admit_free_run_closure``.  Exactly ``FREE_RUN_CLOSURE_KEYS``;
    ``checkpoint_path`` must name an existing file whose SHA-256 is
    ``checkpoint_sha256``, so a manifest is tied to the network file it names.
    """
    if isinstance(closure, dict) and closure.get("kind") == "none":
        if set(closure) != set(FREE_RUN_CONTROL_KEYS):
            raise ValueError("no-closure free-run closure must hold exactly "
                             f"the keys {sorted(FREE_RUN_CONTROL_KEYS)}")
        if closure["control"] not in ("nu10", "nu80"):
            raise ValueError("no-closure free-run control must be 'nu10' or 'nu80'")
        return dict(closure)
    if not isinstance(closure, dict) \
            or set(closure) != set(FREE_RUN_CLOSURE_KEYS):
        found = (sorted(closure) if isinstance(closure, dict)
                 else type(closure).__name__)
        raise ValueError("free-run closure must hold exactly the keys "
                         f"{sorted(FREE_RUN_CLOSURE_KEYS)}, found {found}")
    if closure["kind"] != "trained":
        raise ValueError("trained free-run closure kind must be 'trained'")
    for key in ("run_id", "experiment"):
        if not isinstance(closure[key], str) or not closure[key]:
            raise ValueError(f"free-run closure {key} must be a non-empty string")
    for key in ("config_sha256", "dataset_contract_sha256", "checkpoint_sha256"):
        if not _is_sha256(closure[key]):
            raise ValueError(f"free-run closure {key} is not a SHA-256")
    seed = closure["network_seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("free-run closure network_seed must be a "
                         "non-negative int")
    window = closure["window_days"]
    if not window_days_admitted(window):
        raise ValueError("free-run closure window_days must be one of "
                         f"{ADMITTED_WINDOWS_DAYS} or a longer whole number "
                         "of days")
    if closure["objective"] not in OBJECTIVE_OPERATORS:
        raise ValueError(f"free-run closure objective must be one of "
                         f"{OBJECTIVE_OPERATORS}")
    if closure["wall_ring"] not in WALL_RING_OPTIONS:
        raise ValueError(f"free-run closure wall_ring must be one of "
                         f"{WALL_RING_OPTIONS}")
    if not isinstance(closure["network_spec"], dict) \
            or set(closure["network_spec"]) != set(SWGNetworkSpec._fields):
        raise ValueError("free-run closure network_spec must hold exactly the "
                         f"keys {sorted(SWGNetworkSpec._fields)}")
    path = closure["checkpoint_path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("free-run closure checkpoint_path must be absolute")
    if not Path(path).is_file():
        raise ValueError(f"free-run checkpoint file is missing: {path}")
    if _sha256(Path(path)) != closure["checkpoint_sha256"]:
        raise ValueError(
            f"free-run checkpoint file {path} differs from checkpoint_sha256")
    return dict(closure)


def admit_free_run_closure(manifest) -> dict:
    """The validated closure of one trained-closure free-run manifest.

    Refuses any other manifest schema and every closure ``free_run_closure``
    refuses (missing, extra or mistyped key; missing or changed checkpoint).
    """
    schema = manifest.get("schema") if isinstance(manifest, dict) else None
    if schema != FREE_RUN_MANIFEST_SCHEMA:
        raise ValueError(f"free-run manifest schema must be "
                         f"{FREE_RUN_MANIFEST_SCHEMA!r}, found {schema!r}")
    return free_run_closure(manifest.get("closure"))


def admit_free_run_execution(manifest) -> dict:
    """Validate and return one free run's explicit execution provenance."""
    if not isinstance(manifest, dict):
        raise ValueError("free-run manifest must be a mapping")
    execution = {key: manifest.get(key) for key in (
        "device", "backend", "math_library", "xla_flags", "jax_version",
        "gpu_name")}
    pair = (execution["device"], execution["math_library"])
    if pair not in FREE_RUN_EXECUTIONS:
        raise ValueError(
            "free-run manifest execution must be cpu+libm or cuda+native")
    expected_backend = "gpu" if execution["device"] == "cuda" else "cpu"
    if execution["backend"] != expected_backend:
        raise ValueError(
            f"free-run manifest backend must be {expected_backend!r}")
    if not isinstance(execution["xla_flags"], str):
        raise ValueError("free-run manifest xla_flags must be a string")
    if not isinstance(execution["jax_version"], str) \
            or not execution["jax_version"]:
        raise ValueError("free-run manifest jax_version must be a non-empty string")
    if execution["device"] == "cuda":
        if CUDA_AUTOTUNE_FLAG not in execution["xla_flags"].split():
            raise ValueError(
                f"CUDA free run lacks required XLA flag {CUDA_AUTOTUNE_FLAG}")
        if not isinstance(execution["gpu_name"], str) \
                or not execution["gpu_name"]:
            raise ValueError("CUDA free run must record a GPU name")
    elif execution["gpu_name"] is not None:
        raise ValueError("CPU free run must record gpu_name as null")
    return execution
