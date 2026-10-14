"""Differentiable online-learning pieces for the NEMO-faithful SWG card.

This module owns the experimental training seam, not the ocean card.  A model
produces one F-point volume-transport tendency potential; the certified SWG
stage owns its matched rotated gradient.  The deterministic and stochastic
losses share one ensemble-shaped balanced-field interface.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
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
    _entry_from_state,
    swe_rk3_step,
    swe_rotated_potential_forcing,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import swg_surface_wind
from legoesm.ocean.fidelity.swg_coarse_graining import shifted_weighted_sum
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.training.inert_params import assert_no_inert

SECONDS_PER_HOUR = 3_600
ADMITTED_WINDOWS_DAYS = (1, 2, 4, 8)
OBJECTIVE_OPERATORS = ("symmetric", "template")
DATASET_GAP_HOURS = 30 * 24
TEMPLATE_BATCH_SIZE = 10


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
    """A batch of T0 states and hourly right-endpoint targets."""

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
    filtered = shifted_weighted_sum(
        wet, kernel, axis=0, extension="reflect")
    filtered = shifted_weighted_sum(
        filtered, kernel, axis=1, extension="reflect")
    return jnp.zeros_like(field).at[j0:j1 + 1, i0:i1 + 1].set(filtered)


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
        zeta = _gaussian_smooth_wet(zeta, geometry)
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


def load_content_pinned_hourly_segment(
        root: Path, *, expected_manifest_sha256: str, operator: str,
) -> HourlyStateRecord:
    """Load one complete v1/v2 hourly stream with exact content membership."""
    if operator not in OBJECTIVE_OPERATORS:
        raise ValueError(
            f"operator must be explicitly selected from {OBJECTIVE_OPERATORS}")
    manifest_path = root / "manifest.json"
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("hourly target manifest differs from registered split")
    manifest = json.loads(manifest_path.read_text())
    days = manifest.get("days")
    if manifest.get("status") != "complete" \
            or manifest.get("completed_days") != days \
            or not isinstance(days, int) or days <= 0:
        raise ValueError("hourly target segment is not a complete run")
    schema = manifest.get("schema")
    if schema == "legoesm-swg-double-gyre-hourly-target-v1":
        if operator != "symmetric":
            raise ValueError("v1 hourly targets contain only the box operator")
        prefix = ""
        includes_initial = False
    elif schema == "legoesm-swg-double-gyre-hourly-target-v2":
        prefix = "box_" if operator == "symmetric" else "filtered_"
        includes_initial = True
    else:
        raise ValueError(f"unsupported hourly target schema {schema!r}")
    keys = tuple(f"{prefix}{name}" for name in (
        "ssh_hourly_m", "u_hourly_m_s", "v_hourly_m_s"))
    arrays = {name: [] for name in keys}
    absolute_hours = []
    initial = None
    pins = manifest.get("daily_sha256", {})
    if set(pins) != {f"day{day:04d}.npz" for day in range(1, days + 1)}:
        raise ValueError("hourly manifest does not pin the exact daily membership")
    for day in range(1, days + 1):
        path = root / f"day{day:04d}.npz"
        if _sha256(path) != pins.get(path.name):
            raise ValueError(f"hourly target digest differs: {path.name}")
        with np.load(path, allow_pickle=False) as saved:
            hours = np.asarray(saved["absolute_hour"], dtype=np.int64)
            expected_first = (
                int(manifest["restart"]["source_day"]) + day - 1) * 24 + 1
            if hours.shape != (24,) or int(hours[0]) != expected_first \
                    or not np.array_equal(hours, np.arange(
                        expected_first, expected_first + 24, dtype=np.int64)):
                raise ValueError("hourly target has wrong absolute-time lineage")
            absolute_hours.append(hours)
            for name in arrays:
                arrays[name].append(np.asarray(saved[name], dtype=np.float64))
            if day == 1 and includes_initial:
                initial_keys = tuple(f"{prefix}{name}" for name in (
                    "ssh_initial_m", "u_initial_m_s", "v_initial_m_s"))
                initial = tuple(np.asarray(saved[name], dtype=np.float64)
                                for name in initial_keys)
    fields = tuple(np.concatenate(arrays[name], axis=0) for name in keys)
    hours = np.concatenate(absolute_hours)
    if includes_initial:
        if initial is None:
            raise ValueError("v2 hourly stream is missing its initial state")
        fields = tuple(np.concatenate((value[None], field), axis=0)
                       for value, field in zip(initial, fields, strict=True))
        hours = np.concatenate((np.asarray([hours[0] - 1]), hours))
    if not all(np.all(np.isfinite(value)) for value in fields):
        raise ValueError("hourly target contains a non-finite state")
    return HourlyStateRecord(
        *fields, hours, operator, str(root.resolve()),
        expected_manifest_sha256, includes_initial)


@dataclass(frozen=True)
class SWGDatasetContract:
    """Validated, split-safe view of content-pinned hourly state records."""

    objective: str
    splits: dict[str, tuple[int, int]]
    sources: tuple[PinnedHourlySource, ...]
    contract_path: str
    contract_sha256: str
    batch_size: int

    @staticmethod
    def from_json(path: Path) -> SWGDatasetContract:
        payload = json.loads(path.read_text())
        if payload.get("schema") != "legoesm-swg-online-dataset-v1":
            raise ValueError("unsupported SWG dataset contract schema")
        objective = payload.get("objective")
        if objective not in OBJECTIVE_OPERATORS:
            raise ValueError("dataset objective must be symmetric or template")
        if payload.get("batch_size") != TEMPLATE_BATCH_SIZE:
            raise ValueError("dataset contract must use batch size 10")
        if tuple(payload.get("window_days", ())) != ADMITTED_WINDOWS_DAYS:
            raise ValueError("dataset contract must register 1/2/4/8-day windows")
        raw_splits = payload.get("splits", {})
        if set(raw_splits) != {"train", "validation", "test"}:
            raise ValueError("dataset requires train, validation, and test splits")
        splits = {}
        for name, bounds in raw_splits.items():
            if not isinstance(bounds, list) or len(bounds) != 2 \
                    or not all(isinstance(value, int) for value in bounds) \
                    or bounds[0] >= bounds[1]:
                raise ValueError(f"invalid {name} split bounds")
            splits[name] = (bounds[0], bounds[1])
        ordered = [splits[name] for name in ("train", "validation", "test")]
        for left, right in zip(ordered[:-1], ordered[1:], strict=True):
            if right[0] - left[1] < DATASET_GAP_HOURS:
                raise ValueError("dataset splits require a gap of at least 30 days")
        sources = payload.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("dataset contract has no content-pinned sources")
        pins = []
        memberships = set()
        for source in sources:
            if set(source) != {"root", "manifest_sha256"}:
                raise ValueError("dataset source membership is ambiguous")
            membership = (source["root"], source["manifest_sha256"])
            if membership in memberships:
                raise ValueError("dataset source membership is duplicated")
            memberships.add(membership)
            root = Path(source["root"])
            manifest_path = root / "manifest.json"
            if _sha256(manifest_path) != source["manifest_sha256"]:
                raise ValueError("dataset source manifest differs from contract")
            manifest = json.loads(manifest_path.read_text())
            days = manifest.get("days")
            if manifest.get("schema") \
                    != "legoesm-swg-double-gyre-hourly-target-v2" \
                    or manifest.get("status") != "complete" \
                    or manifest.get("completed_days") != days:
                raise ValueError(
                    "training dataset sources require v2 stored initial states")
            source_day = int(manifest["restart"]["source_day"])
            pins.append(PinnedHourlySource(
                str(root.resolve()), source["manifest_sha256"],
                source_day * 24, (source_day + int(days)) * 24))
        contract = SWGDatasetContract(
            objective=objective, splits=splits, sources=tuple(pins),
            contract_path=str(path.resolve()), contract_sha256=_sha256(path),
            batch_size=TEMPLATE_BATCH_SIZE)
        contract._validate_coverage()
        return contract

    def _validate_coverage(self) -> None:
        ordered = sorted(self.sources, key=lambda source: source.first_absolute_hour)
        for left, right in zip(ordered[:-1], ordered[1:], strict=True):
            if right.first_absolute_hour < left.last_absolute_hour:
                raise ValueError("content-pinned hourly sources overlap ambiguously")
            if right.first_absolute_hour == left.last_absolute_hour:
                left_state = self._endpoint_state(left, final=True)
                right_state = self._endpoint_state(right, final=False)
                if not all(np.array_equal(a, b) for a, b in zip(
                        left_state, right_state, strict=True)):
                    raise ValueError(
                        "adjacent hourly chunks disagree at restart hour")
        for split, (start, end) in self.splits.items():
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
                    f"{split} split is not a contiguous pinned hourly record")

    def _endpoint_state(self, source: PinnedHourlySource, *, final: bool):
        root = Path(source.root)
        manifest = json.loads((root / "manifest.json").read_text())
        day = int(manifest["days"]) if final else 1
        path = root / f"day{day:04d}.npz"
        if _sha256(path) != manifest["daily_sha256"].get(path.name):
            raise ValueError("hourly chunk endpoint differs from its manifest")
        prefix = "box_" if self.objective == "symmetric" else "filtered_"
        with np.load(path, allow_pickle=False) as saved:
            if final:
                return tuple(np.asarray(saved[f"{prefix}{name}_hourly_{unit}"])[-1]
                             for name, unit in (("ssh", "m"), ("u", "m_s"),
                                                ("v", "m_s")))
            return tuple(np.asarray(saved[f"{prefix}{name}_initial_{unit}"])
                         for name, unit in (("ssh", "m"), ("u", "m_s"),
                                            ("v", "m_s")))

    def _load_source(self, source: PinnedHourlySource) -> HourlyStateRecord:
        return load_content_pinned_hourly_segment(
            Path(source.root), expected_manifest_sha256=source.manifest_sha256,
            operator=self.objective)

    def _source_for_hour(self, hour: int) -> PinnedHourlySource:
        matches = [source for source in self.sources
                   if source.first_absolute_hour <= hour
                   <= source.last_absolute_hour]
        if not matches:
            raise ValueError(f"dataset source gap at absolute hour {hour}")
        if len(matches) > 1:
            # Adjacent chunks share only the restart/end-point hour.  Either
            # byte-identical copy is admissible; the per-batch loader checks it.
            return sorted(matches, key=lambda source: source.root)[0]
        return matches[0]

    def _require_split(self, split: str) -> tuple[int, int]:
        if split not in self.splits:
            raise ValueError(f"unknown or leaked dataset split {split!r}")
        return self.splits[split]

    def batches(self, *, split: str, window_days: int,
                epoch: int, seed: int):
        """Build seeded, shuffled, non-overlapping template batches."""
        start_bound, end_bound = self._require_split(split)
        if window_days not in ADMITTED_WINDOWS_DAYS:
            raise ValueError("window must be one of 1, 2, 4, or 8 days")
        if type(epoch) is not int or epoch < 0 or type(seed) is not int:
            raise ValueError("dataset epoch and seed must be integers")
        window_hours = 24 * window_days
        rng = np.random.default_rng(np.random.SeedSequence((seed, epoch)))
        offset = int(rng.integers(0, window_hours))
        starts = np.arange(
            start_bound + offset, end_bound - window_hours + 1,
            window_hours, dtype=np.int64)
        rng.shuffle(starts)
        usable = (starts.size // self.batch_size) * self.batch_size
        starts = starts[:usable]
        for first in range(0, starts.size, self.batch_size):
            batch_starts = starts[first:first + self.batch_size]
            cache = {}

            def state(hour):
                source = self._source_for_hour(int(hour))
                if source not in cache:
                    cache[source] = self._load_source(source)
                record = cache[source]
                indices = np.flatnonzero(record.absolute_hour == hour)
                if indices.size != 1:
                    raise ValueError(
                        f"hour {hour} is not unique in its content-pinned source")
                index = int(indices[0])
                return record.ssh_m[index], record.u_m_s[index], record.v_m_s[index]

            initial = tuple(np.stack([
                state(int(start))[field] for start in batch_starts])
                for field in range(3))
            target = tuple(np.stack([
                np.stack([state(int(start) + lead)[field]
                          for lead in range(1, window_hours + 1)])
                for start in batch_starts]) for field in range(3))
            batch = WindowBatch(
                initial, target, batch_starts, split, window_days)
            self.validate_batch_membership(batch)
            yield batch

    def training_normalizers(
            self, geometry: SWGClosureGeometry) -> tuple[BalancedNormalizers, dict]:
        """Compute normalizers structurally from train hours and provenance."""
        start, end = self.splits["train"]
        weights = np.asarray(
            geometry.area_t_m2 * geometry.tmask, dtype=np.float64)
        weights /= np.sum(weights, dtype=np.float64)
        sums = np.zeros(3, dtype=np.float64)
        squares = np.zeros(3, dtype=np.float64)
        count = 0
        for source in sorted(
                self.sources, key=lambda item: item.first_absolute_hour):
            if source.last_absolute_hour < start \
                    or source.first_absolute_hour > end:
                continue
            record = self._load_source(source)
            selected = ((record.absolute_hour >= start)
                        & (record.absolute_hour <= end))
            if not np.any(selected):
                continue
            native = tuple(jnp.asarray(value[selected]) for value in (
                record.ssh_m, record.u_m_s, record.v_m_s))
            diagnosed = diagnose_native_trajectory(
                native, geometry, objective=self.objective)
            for index, value in enumerate(diagnosed):
                array = np.asarray(value, dtype=np.float64)
                sums[index] += np.sum(array * weights, dtype=np.float64)
                squares[index] += np.sum(array * array * weights,
                                         dtype=np.float64)
            count += int(np.count_nonzero(selected))
        if count != end - start + 1:
            raise ValueError("training split is not a contiguous pinned record")
        variances = squares / count - (sums / count) ** 2
        if np.any(~np.isfinite(variances)) or np.any(variances <= 0.0):
            raise ValueError("training split variance is not positive finite")
        normalizers = BalancedNormalizers(*map(float, variances))
        provenance = {
            "contract_path": self.contract_path,
            "contract_sha256": self.contract_sha256,
            "training_absolute_hours": [int(start), int(end)],
            "source_manifests": {
                source.root: source.manifest_sha256
                for source in self.sources},
        }
        return normalizers, provenance

    def normalizers_for_split(
            self, split: str, geometry: SWGClosureGeometry):
        """Fail closed if a caller tries to normalize from held-out data."""
        if split != "train":
            raise ValueError("normalizers may consume the training split only")
        return self.training_normalizers(geometry)

    def validate_batch_membership(self, batch: WindowBatch) -> None:
        """Refuse a held-out hour or split-crossing window in a batch."""
        start_bound, end_bound = self._require_split(batch.split)
        window_hours = 24 * batch.window_days
        starts = np.asarray(batch.start_absolute_hour, dtype=np.int64)
        if starts.shape != (self.batch_size,):
            raise ValueError("dataset batch does not contain exactly ten windows")
        if np.any(starts < start_bound) or np.any(starts + window_hours > end_bound):
            raise ValueError("dataset batch contains a leaked or crossing hour")


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
    u = np.zeros(state.u.data.shape, dtype=np.float64)
    v = np.zeros(state.v.data.shape, dtype=np.float64)
    u[:, 1:, 0] = u_native
    v[1:, :, 0] = v_native
    return state._replace(
        eta=state.eta.replace(data=jnp.asarray(ssh)),
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)))


def balanced_normalizers_from_training(
        trajectories: tuple[BalancedTrajectory, ...],
        geometry: SWGClosureGeometry) -> BalancedNormalizers:
    """Compute area-weighted variances from training trajectories only."""
    if not trajectories:
        raise ValueError("normalizers require at least one training trajectory")
    weights = np.asarray(
        geometry.area_t_m2 * geometry.tmask, dtype=np.float64).copy()
    weights /= np.sum(weights, dtype=np.float64)

    def variance(name):
        values = np.concatenate([
            np.asarray(getattr(item, name), dtype=np.float64)
            for item in trajectories], axis=0)
        mean = np.sum(values * weights, axis=(-2, -1), dtype=np.float64).mean()
        result = np.mean(np.sum(
            (values - mean) ** 2 * weights, axis=(-2, -1), dtype=np.float64))
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"training variance for {name} is not positive")
        return float(result)

    return BalancedNormalizers(
        variance("relative_vorticity_s1"),
        variance("rotational_u_m_s"), variance("rotational_v_m_s"))


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
        potential = network(
            entry, geometry, key=jax.random.fold_in(key, index),
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


def balanced_trajectory_loss(predicted: BalancedTrajectory,
                             target: BalancedTrajectory,
                             geometry: SWGClosureGeometry,
                             normalizers: BalancedNormalizers, *,
                             kind: str,
                             beta_m1_s1: float | None = None,
                             basin_length_m: float | None = None,
                             rotational_velocity_weight: float = 0.0):
    """Maddison point loss or Brolly per-lead energy score.

    Every predicted array has shape ``(members, hours, y, x)``; targets omit
    the member axis.  The energy score is the proper multivariate score over
    all three normalized, area-weighted balanced fields and requires S >= 2.
    Its pair term uses the unbiased ``2 S (S-1)`` denominator.
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

    if kind == "point":
        if beta_m1_s1 is None or basin_length_m is None:
            raise ValueError("Maddison point loss requires beta and basin length")
        if not np.isfinite(beta_m1_s1) or beta_m1_s1 <= 0.0 \
                or not np.isfinite(basin_length_m) or basin_length_m <= 0.0:
            raise ValueError("Maddison beta and basin length must be positive")
        if rotational_velocity_weight < 0.0:
            raise ValueError("rotational velocity weight cannot be negative")
        j0, j1, i0, i1 = geometry.wet_bounds
        interior = jnp.zeros_like(geometry.tmask, dtype=bool).at[
            j0 + 1:j1, i0 + 1:i1].set(True)
        if predicted.relative_vorticity_s1.shape \
                != target.relative_vorticity_s1.shape \
                or predicted.relative_vorticity_s1.ndim != 4:
            raise ValueError(
                "point loss requires matching (windows,hours,y,x) arrays")
        zeta_error = (predicted.relative_vorticity_s1
                      - target.relative_vorticity_s1)
        weighted_sum = jnp.sum(
            zeta_error**2 * geometry.area_t_m2 * interior, axis=(-2, -1))
        scale = (1.0e4 / (beta_m1_s1**2 * basin_length_m**2)
                 / (4.0 * basin_length_m**2))
        loss = scale * jnp.mean(weighted_sum)
        if rotational_velocity_weight > 0.0:
            rot_error = (
                (predicted.rotational_u_m_s - target.rotational_u_m_s)**2
                / normalizers.rotational_u_variance_m2_s2
                + (predicted.rotational_v_m_s - target.rotational_v_m_s)**2
                / normalizers.rotational_v_variance_m2_s2)
            loss = loss + rotational_velocity_weight * jnp.mean(jnp.sum(
                rot_error * weight, axis=(-2, -1)))
        return loss
    if kind != "energy_score":
        raise ValueError(f"unknown SWG balanced loss {kind!r}")
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
        normalizers: BalancedNormalizers,
        rotational_velocity_weight: float = 0.0):
    """Exact Eq. 7 batch/hour reduction for one explicitly selected objective."""
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
        predicted, target, geometry, normalizers, kind="point",
        beta_m1_s1=beta_m1_s1, basin_length_m=basin_length_m,
        rotational_velocity_weight=rotational_velocity_weight)


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
