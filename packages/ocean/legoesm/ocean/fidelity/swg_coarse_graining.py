"""Conservative fine-to-coarse restriction for the SWG native C-grid.

The training target is a finite-volume restriction: T-cell volumes and the
native U/V face volume transports are summed over each nested coarse control
volume.  That construction commutes with the discrete continuity divergence.
The round-1 independent Gaussian/area-average operator remains below under
explicit ``diagnostic_*`` names; it is not a closure-ready state target.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import jax
import numpy as np
from scipy.ndimage import gaussian_filter

FILTER_TRUNCATE = 6.0
CONTINUITY_BAR_M_S = 1.0e-15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weighted_mean(field: np.ndarray, mask: np.ndarray,
                   area: np.ndarray) -> float:
    weights = np.where(mask, area, 0.0)
    return float(np.sum(field * weights, dtype=np.float64)
                 / np.sum(weights, dtype=np.float64))


def _restore_mean(field: np.ndarray, mask: np.ndarray, area: np.ndarray,
                  target_mean: float) -> np.ndarray:
    out = np.where(mask, field, 0.0)
    out[mask] += target_mean - _weighted_mean(out, mask, area)
    # Put the final reduction-roundoff residual in one wet cell.  This makes
    # basin-integral preservation explicit instead of relying on summation
    # order after adding a uniform correction.
    weights = np.where(mask, area, 0.0)
    target_sum = target_mean * np.sum(weights, dtype=np.float64)
    residual = target_sum - np.sum(out * weights, dtype=np.float64)
    at = tuple(np.argwhere(mask)[0])
    out[at] += residual / area[at]
    return out


def masked_gaussian_filter(field: np.ndarray, mask: np.ndarray, *,
                           sigma_cells: float, area: np.ndarray | None = None,
                           truncate: float = FILTER_TRUNCATE) -> np.ndarray:
    """Filter a wet field, normalize at walls, and preserve its wet mean."""
    field = np.asarray(field, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if field.ndim != 2 or field.shape != mask.shape:
        raise ValueError("field and mask must be equal-shape 2-D arrays")
    if not np.all(np.isfinite(field)) or not np.any(mask):
        raise ValueError("field must be finite and mask non-empty")
    if not np.isfinite(sigma_cells) or sigma_cells <= 0.0:
        raise ValueError("sigma_cells must be positive finite")
    if area is None:
        area = np.ones(field.shape, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    if area.shape != field.shape or not np.all(np.isfinite(area)) \
            or np.any(area <= 0.0):
        raise ValueError("area must be positive finite with field shape")
    boundary_mode = "wrap" if np.all(mask) else "constant"
    numerator = gaussian_filter(
        np.where(mask, field, 0.0), sigma_cells, mode=boundary_mode, cval=0.0,
        truncate=truncate)
    denominator = gaussian_filter(
        mask.astype(np.float64), sigma_cells, mode=boundary_mode, cval=0.0,
        truncate=truncate)
    filtered = np.zeros_like(field)
    filtered[mask] = numerator[mask] / denominator[mask]
    return _restore_mean(
        filtered, mask, area, _weighted_mean(field, mask, area))


def overlap_weights(source_centres: np.ndarray, target_centres: np.ndarray,
                    source_spacing_m: float,
                    target_spacing_m: float) -> np.ndarray:
    """Return exact one-dimensional cell-overlap lengths."""
    source = np.asarray(source_centres, dtype=np.float64)
    target = np.asarray(target_centres, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("overlap coordinates must be one-dimensional")
    if source.size < 2 or target.size < 2:
        raise ValueError("overlap coordinates need at least two points")
    if not np.all(np.diff(source) > 0.0) or not np.all(np.diff(target) > 0.0):
        raise ValueError("overlap coordinates must increase strictly")
    source_lo = source - 0.5 * source_spacing_m
    source_hi = source + 0.5 * source_spacing_m
    target_lo = target - 0.5 * target_spacing_m
    target_hi = target + 0.5 * target_spacing_m
    return np.maximum(
        0.0,
        np.minimum(target_hi[:, None], source_hi[None, :])
        - np.maximum(target_lo[:, None], source_lo[None, :]),
    )


def area_average(field: np.ndarray, source_mask: np.ndarray,
                 target_mask: np.ndarray, *, source_x_m: np.ndarray,
                 source_y_m: np.ndarray, target_x_m: np.ndarray,
                 target_y_m: np.ndarray, source_spacing_m: float,
                 target_spacing_m: float) -> np.ndarray:
    """Area-average a native stagger from one uniform C-grid to another."""
    field = np.asarray(field, dtype=np.float64)
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    if field.shape != source_mask.shape:
        raise ValueError("source field/mask shapes differ")
    if np.asarray(source_x_m).shape != field.shape \
            or np.asarray(source_y_m).shape != field.shape:
        raise ValueError("source coordinate shapes differ")
    if np.asarray(target_x_m).shape != target_mask.shape \
            or np.asarray(target_y_m).shape != target_mask.shape:
        raise ValueError("target coordinate shapes differ")
    wx = overlap_weights(source_x_m[0], target_x_m[0],
                         source_spacing_m, target_spacing_m)
    wy = overlap_weights(source_y_m[:, 0], target_y_m[:, 0],
                         source_spacing_m, target_spacing_m)
    weights = wy @ source_mask.astype(np.float64) @ wx.T
    numerator = wy @ np.where(source_mask, field, 0.0) @ wx.T
    if np.any(target_mask & (weights <= 0.0)):
        raise ValueError("a wet target cell has no wet source overlap")
    first = field[source_mask][0]
    if np.all(field[source_mask] == first):
        return np.where(target_mask, first, 0.0)
    out = np.zeros(target_mask.shape, dtype=np.float64)
    out[target_mask] = numerator[target_mask] / weights[target_mask]
    source_area = np.full(field.shape, source_spacing_m ** 2)
    target_area = np.full(target_mask.shape, target_spacing_m ** 2)
    return _restore_mean(
        out, target_mask, target_area,
        _weighted_mean(field, source_mask, source_area))


def cgrid_geometry(namelist) -> dict[str, dict[str, np.ndarray] | float]:
    """Instantiate native T/U/V masks and coordinates from a resolved deck."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_swg_card,
        swg_horizontal_coordinates,
    )

    card = build_swg_card(namelist)
    coordinates = swg_horizontal_coordinates(namelist)
    u_full, v_full = compute_face_masks(card.recipe.land_mask, card.recipe.grid)
    masks = {
        "T": np.asarray(card.recipe.land_mask, dtype=bool),
        "U": np.asarray(u_full[:, 1:], dtype=bool),
        "V": np.asarray(v_full[1:, :], dtype=bool),
    }
    z_coord = card.recipe.z_coord
    barotropic = z_coord.nemo_een_barotropic
    return {
        "spacing_m": float(coordinates["spacing_m"]),
        "grid": card.recipe.grid,
        "mask": masks,
        "x": {"T": np.asarray(coordinates["glamt"]),
              "U": np.asarray(coordinates["glamu"]),
              "V": np.asarray(coordinates["glamv"])},
        "y": {"T": np.asarray(coordinates["gphit"]),
              "U": np.asarray(coordinates["gphiu"]),
              "V": np.asarray(coordinates["gphiv"])},
        "area": {
            "T": np.asarray(z_coord.nemo_e1e2t, dtype=np.float64),
            "U": np.asarray(z_coord.nemo_e1e2u, dtype=np.float64),
            "V": np.asarray(z_coord.nemo_e1e2v, dtype=np.float64),
        },
        "face_length": {
            "U": np.asarray(z_coord.nemo_e2u, dtype=np.float64),
            "V": np.asarray(z_coord.nemo_e1v, dtype=np.float64),
        },
        "reference_thickness": {
            "T": np.asarray(z_coord.nemo_e3t_0[..., 0], dtype=np.float64),
            "U": np.asarray(barotropic.e3u_0[..., 0], dtype=np.float64),
            "V": np.asarray(barotropic.e3v_0[..., 0], dtype=np.float64),
        },
        "reference_transport_depth": {
            "U": np.asarray(z_coord.nemo_hu_0, dtype=np.float64),
            "V": np.asarray(z_coord.nemo_hv_0, dtype=np.float64),
        },
    }


def _wet_rectangle(mask: np.ndarray, *, label: str) -> tuple[int, int, int, int]:
    """Return inclusive wet bounds, refusing a non-rectangular SWG basin."""
    wet = np.asarray(mask, dtype=bool)
    if wet.ndim != 2 or not np.any(wet):
        raise ValueError(f"{label} mask must be a nonempty 2-D array")
    jj, ii = np.where(wet)
    bounds = int(jj.min()), int(jj.max()), int(ii.min()), int(ii.max())
    expected = np.zeros_like(wet)
    expected[bounds[0]:bounds[1] + 1, bounds[2]:bounds[3] + 1] = True
    if not np.array_equal(wet, expected):
        raise ValueError(f"{label} wet mask is not one closed rectangle")
    return bounds


def _restriction_layout(source: dict, target: dict) -> dict[str, int]:
    """Validate the integer nesting and return the two wet rectangles."""
    source_bounds = _wet_rectangle(source["mask"]["T"], label="source T")
    target_bounds = _wet_rectangle(target["mask"]["T"], label="target T")
    source_ny = source_bounds[1] - source_bounds[0] + 1
    source_nx = source_bounds[3] - source_bounds[2] + 1
    target_ny = target_bounds[1] - target_bounds[0] + 1
    target_nx = target_bounds[3] - target_bounds[2] + 1
    ratio = float(target["spacing_m"]) / float(source["spacing_m"])
    factor = int(round(ratio))
    if factor <= 0 or ratio != float(factor):
        raise ValueError("source and target spacings must have an integer ratio")
    if (source_ny, source_nx) != (factor * target_ny, factor * target_nx):
        raise ValueError("source and target wet rectangles do not tile exactly")
    return {
        "factor": factor,
        "source_j0": source_bounds[0], "source_j1": source_bounds[1],
        "source_i0": source_bounds[2], "source_i1": source_bounds[3],
        "target_j0": target_bounds[0], "target_j1": target_bounds[1],
        "target_i0": target_bounds[2], "target_i1": target_bounds[3],
        "target_ny": target_ny, "target_nx": target_nx,
    }


def _live_face_thickness(eta: np.ndarray, geometry: dict,
                         kind: str) -> np.ndarray:
    """NEMO QCO live U/V thickness for this uniform one-layer card."""
    if kind not in ("U", "V"):
        raise ValueError(f"face kind must be U or V, got {kind!r}")
    eta = np.asarray(eta, dtype=np.float64)
    tmask = np.asarray(geometry["mask"]["T"], dtype=bool)
    if eta.shape != tmask.shape or not np.all(np.isfinite(eta)):
        raise ValueError("SSH must be finite and have the T-mask shape")
    area_t = np.asarray(geometry["area"]["T"], dtype=np.float64)
    weighted_eta = area_t * np.where(tmask, eta, 0.0)
    axis = 1 if kind == "U" else 0
    numerator = 0.5 * (weighted_eta + np.roll(weighted_eta, -1, axis=axis))
    reference_depth = np.asarray(
        geometry["reference_transport_depth"][kind], dtype=np.float64)
    area_face = np.asarray(geometry["area"][kind], dtype=np.float64)
    reference_thickness = np.asarray(
        geometry["reference_thickness"][kind], dtype=np.float64)
    face_mask = np.asarray(geometry["mask"][kind], dtype=bool)
    ratio = np.zeros_like(eta)
    ratio[face_mask] = (
        numerator[face_mask]
        / reference_depth[face_mask]
        / area_face[face_mask]
    )
    return reference_thickness * (1.0 + ratio * face_mask)


def live_face_thickness(eta: np.ndarray, geometry: dict,
                        kind: str) -> np.ndarray:
    """Public QCO live thickness used by conservative perturbation tools."""
    if kind not in ("U", "V"):
        raise ValueError("live face thickness kind must be U or V")
    return _live_face_thickness(eta, geometry, kind)


def native_face_transports(fields: dict[str, np.ndarray], geometry: dict,
                           ) -> dict[str, np.ndarray]:
    """Return native east/north volume transports [m3/s]."""
    if set(fields) != {"T", "U", "V"}:
        raise ValueError("fields must be exactly T, U, and V")
    eta = np.asarray(fields["T"], dtype=np.float64)
    transports = {}
    for kind in ("U", "V"):
        velocity = np.asarray(fields[kind], dtype=np.float64)
        mask = np.asarray(geometry["mask"][kind], dtype=bool)
        if velocity.shape != mask.shape or not np.all(np.isfinite(velocity)):
            raise ValueError(f"{kind} must be finite and have its mask shape")
        transports[kind] = np.where(
            mask,
            np.asarray(geometry["face_length"][kind], dtype=np.float64)
            * _live_face_thickness(eta, geometry, kind) * velocity,
            0.0,
        )
    return transports


def build_jax_native_transport_diagnostic(namelist):
    """Return a JAX diagnostic for instantaneous native QCO face transports."""
    import jax.numpy as jnp

    geometry = cgrid_geometry(namelist)
    tmask = jnp.asarray(geometry["mask"]["T"])
    area_t = jnp.asarray(geometry["area"]["T"])
    constants = {
        kind: tuple(jnp.asarray(geometry[name][kind]) for name in (
            "mask", "reference_transport_depth", "area",
            "reference_thickness", "face_length"))
        for kind in ("U", "V")
    }

    def diagnostic(eta, u, v):
        weighted_eta = area_t * jnp.where(tmask, eta, 0.0)
        transports = []
        for kind, velocity, axis in (("U", u, 1), ("V", v, 0)):
            mask, depth, area, reference_h, face_length = constants[kind]
            numerator = 0.5 * (
                weighted_eta + jnp.roll(weighted_eta, -1, axis=axis))
            ratio = jnp.where(mask, numerator / depth / area, 0.0)
            thickness = reference_h * (1.0 + ratio * mask)
            transports.append(jnp.where(
                mask, velocity * thickness * face_length, 0.0))
        return tuple(transports)

    return diagnostic


def _restrict_t_cell_sum(values: np.ndarray, source: dict, target: dict,
                         layout: dict[str, int]) -> np.ndarray:
    """Sum fine T-cell extensive values into the nested coarse cells."""
    f = layout["factor"]
    sj0, sj1 = layout["source_j0"], layout["source_j1"]
    si0, si1 = layout["source_i0"], layout["source_i1"]
    tj0, tj1 = layout["target_j0"], layout["target_j1"]
    ti0, ti1 = layout["target_i0"], layout["target_i1"]
    block = np.asarray(values, dtype=np.float64)[sj0:sj1 + 1, si0:si1 + 1]
    summed = block.reshape(
        layout["target_ny"], f, layout["target_nx"], f).sum(axis=(1, 3))
    out = np.zeros(np.asarray(target["mask"]["T"]).shape, dtype=np.float64)
    out[tj0:tj1 + 1, ti0:ti1 + 1] = summed
    return out


def restrict_native_transports(transports: dict[str, np.ndarray], source: dict,
                               target: dict, layout: dict[str, int] | None = None,
                               ) -> dict[str, np.ndarray]:
    """Sum the fine faces that tile every wet coarse U/V face."""
    if set(transports) != {"U", "V"}:
        raise ValueError("transports must be exactly U and V")
    layout = _restriction_layout(source, target) if layout is None else layout
    f = layout["factor"]
    sj0, sj1 = layout["source_j0"], layout["source_j1"]
    si0, si1 = layout["source_i0"], layout["source_i1"]
    tj0, tj1 = layout["target_j0"], layout["target_j1"]
    ti0, ti1 = layout["target_i0"], layout["target_i1"]
    ny, nx = layout["target_ny"], layout["target_nx"]

    fine_u = np.asarray(transports["U"], dtype=np.float64)
    source_u_columns = si0 + f * np.arange(1, nx, dtype=int) - 1
    restricted_u = fine_u[sj0:sj1 + 1, source_u_columns]
    restricted_u = restricted_u.reshape(ny, f, nx - 1).sum(axis=1)
    coarse_u = np.zeros(np.asarray(target["mask"]["U"]).shape, dtype=np.float64)
    coarse_u[tj0:tj1 + 1, ti0:ti1] = restricted_u

    fine_v = np.asarray(transports["V"], dtype=np.float64)
    source_v_rows = sj0 + f * np.arange(1, ny, dtype=int) - 1
    restricted_v = fine_v[source_v_rows, si0:si1 + 1]
    restricted_v = restricted_v.reshape(ny - 1, nx, f).sum(axis=2)
    coarse_v = np.zeros(np.asarray(target["mask"]["V"]).shape, dtype=np.float64)
    coarse_v[tj0:tj1, ti0:ti1 + 1] = restricted_v
    return {"U": coarse_u, "V": coarse_v}


def conservative_restrict_cgrid(fields: dict[str, np.ndarray], source: dict,
                                target: dict) -> dict[str, np.ndarray]:
    """Restrict cell volume and face transport without mean restoration."""
    if set(fields) != {"T", "U", "V"}:
        raise ValueError("fields must be exactly T, U, and V")
    layout = _restriction_layout(source, target)
    eta = np.asarray(fields["T"], dtype=np.float64)
    source_tmask = np.asarray(source["mask"]["T"], dtype=bool)
    if eta.shape != source_tmask.shape or not np.all(np.isfinite(eta)):
        raise ValueError("T must be finite and have the source T-mask shape")
    source_volume = (
        np.asarray(source["area"]["T"], dtype=np.float64)
        * (np.asarray(source["reference_thickness"]["T"], dtype=np.float64)
           + np.where(source_tmask, eta, 0.0))
        * source_tmask
    )
    coarse_volume = _restrict_t_cell_sum(source_volume, source, target, layout)
    target_tmask = np.asarray(target["mask"]["T"], dtype=bool)
    coarse_eta = np.zeros_like(coarse_volume)
    coarse_eta[target_tmask] = (
        coarse_volume[target_tmask]
        / np.asarray(target["area"]["T"], dtype=np.float64)[target_tmask]
        - np.asarray(target["reference_thickness"]["T"],
                     dtype=np.float64)[target_tmask]
    )

    fine_transports = native_face_transports(fields, source)
    coarse_transports = restrict_native_transports(
        fine_transports, source, target, layout)
    output = {"T": coarse_eta}
    for kind in ("U", "V"):
        mask = np.asarray(target["mask"][kind], dtype=bool)
        denominator = (
            np.asarray(target["face_length"][kind], dtype=np.float64)
            * _live_face_thickness(coarse_eta, target, kind)
        )
        velocity = np.zeros_like(coarse_eta)
        velocity[mask] = coarse_transports[kind][mask] / denominator[mask]
        output[kind] = velocity
    return output


def gaussian_kernel_1d(*, sigma_cells: float,
                       truncate: float = FILTER_TRUNCATE) -> np.ndarray:
    """Return the normalized discrete Gaussian used by filtered restriction."""
    if not np.isfinite(sigma_cells) or sigma_cells <= 0.0:
        raise ValueError("Gaussian sigma must be positive finite")
    if not np.isfinite(truncate) or truncate <= 0.0:
        raise ValueError("Gaussian truncate must be positive finite")
    radius = int(truncate * sigma_cells + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_cells) ** 2)
    return kernel / np.sum(kernel, dtype=np.float64)


def _pad_even_cell_centres(values: np.ndarray, radius: int,
                           axis: int) -> np.ndarray:
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    return np.pad(values, padding, mode="symmetric")


def _pad_odd_boundary_faces(values: np.ndarray, radius: int,
                            axis: int) -> np.ndarray:
    """Odd-extend a face field about its two zero-valued wall faces."""
    if values.shape[axis] <= radius:
        raise ValueError("Gaussian support exceeds the closed-basin face axis")
    if np.any(np.take(values, (0, -1), axis=axis) != 0.0):
        raise ValueError("normal transport on a closed wall must be exactly zero")
    interior = np.arange(1, radius + 1)
    left = -np.take(values, interior[::-1], axis=axis)
    right = -np.take(values, values.shape[axis] - 1 - interior, axis=axis)
    return np.concatenate((left, values, right), axis=axis)


def _convolve_valid_axis(values: np.ndarray, kernel: np.ndarray,
                         axis: int) -> np.ndarray:
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), axis, values)


def shifted_weighted_sum(values, kernel, *, axis: int, extension: str):
    """Apply one 1-D stencil as weighted shifted slices, never convolution.

    ``extension`` is one of the three closed-basin conventions used by this
    lane: ``symmetric`` about cell centres, ``reflect`` about the edge cell,
    or ``odd`` about explicitly zero normal wall faces.  The implementation is
    deliberately expressed as padding, slicing, multiplication, and addition;
    it lowers identically on CPU and GPU without a convolution custom call.
    """
    import jax.numpy as jnp

    array = jnp.asarray(values)
    weights = jnp.asarray(kernel, dtype=array.dtype)
    if array.ndim < 1 or weights.ndim != 1 or weights.shape[0] % 2 != 1:
        raise ValueError("shifted stencil requires an odd one-dimensional kernel")
    if not 0 <= axis < array.ndim:
        raise ValueError("shifted stencil axis is out of range")
    radius = int(weights.shape[0] // 2)
    if extension == "odd":
        if array.shape[axis] <= radius:
            raise ValueError("shifted stencil support exceeds the odd face axis")
        indices = jnp.arange(1, radius + 1)
        left = -jnp.take(array, indices[::-1], axis=axis)
        right = -jnp.take(
            array, array.shape[axis] - 1 - indices, axis=axis)
        padded = jnp.concatenate((left, array, right), axis=axis)
    elif extension in ("symmetric", "reflect"):
        padding = [(0, 0)] * array.ndim
        padding[axis] = (radius, radius)
        padded = jnp.pad(array, padding, mode=extension)
    else:
        raise ValueError(
            "shifted stencil extension must be symmetric, reflect, or odd")

    result = jnp.zeros_like(array)
    width = array.shape[axis]
    for offset in range(int(weights.shape[0])):
        selection = [slice(None)] * array.ndim
        selection[axis] = slice(offset, offset + width)
        result = result + weights[offset] * padded[tuple(selection)]
    return result


@jax.jit
def filter_closed_cgrid_extensives_array(
        cell_volume, u_transport, v_transport, *, kernel):
    """Array-backend implementation shared by offline and in-scan filtering."""
    volume = shifted_weighted_sum(
        cell_volume, kernel, axis=0, extension="symmetric")
    volume = shifted_weighted_sum(
        volume, kernel, axis=1, extension="symmetric")
    u = shifted_weighted_sum(
        u_transport, kernel, axis=0, extension="symmetric")
    u = shifted_weighted_sum(u, kernel, axis=1, extension="odd")
    v = shifted_weighted_sum(v_transport, kernel, axis=0, extension="odd")
    v = shifted_weighted_sum(v, kernel, axis=1, extension="symmetric")
    # Odd parity fixes both normal wall faces analytically. Re-impose their
    # exact floating-point representation after accumulation roundoff.
    u = u.at[:, (0, -1)].set(0.0)
    v = v.at[(0, -1), :].set(0.0)
    return volume, u, v


def filter_closed_cgrid_extensives(
        cell_volume: np.ndarray, u_transport: np.ndarray,
        v_transport: np.ndarray, *, sigma_cells: float,
        truncate: float = FILTER_TRUNCATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter closed-basin cell volumes and staggered face transports.

    Cell volumes are even about every wall.  A transport is odd about the wall
    normal to its own direction and even about walls tangent to it.  These
    parities make the extended-grid finite-volume divergence commute with the
    same separable Gaussian while retaining exactly zero normal wall transport.
    Arrays use the local closed-grid shapes ``T=(ny,nx)``,
    ``U=(ny,nx+1)``, and ``V=(ny+1,nx)``.
    """
    volume = np.asarray(cell_volume, dtype=np.float64)
    u = np.asarray(u_transport, dtype=np.float64)
    v = np.asarray(v_transport, dtype=np.float64)
    if volume.ndim != 2 or u.shape != (volume.shape[0], volume.shape[1] + 1) \
            or v.shape != (volume.shape[0] + 1, volume.shape[1]):
        raise ValueError("filtered C-grid extensives have incompatible shapes")
    if not all(np.all(np.isfinite(value)) for value in (volume, u, v)):
        raise ValueError("filtered C-grid extensives must be finite")
    kernel = gaussian_kernel_1d(
        sigma_cells=sigma_cells, truncate=truncate)
    radius = kernel.size // 2
    if u.shape[1] <= radius or v.shape[0] <= radius:
        raise ValueError("Gaussian support exceeds the closed-basin face axis")
    if np.any(u[:, (0, -1)] != 0.0) or np.any(v[(0, -1), :] != 0.0):
        raise ValueError("normal transport on a closed wall must be exactly zero")
    return tuple(np.asarray(value) for value in
                 filter_closed_cgrid_extensives_array(
                     volume, u, v, kernel=kernel))


def _local_closed_extensives(fields: dict[str, np.ndarray], geometry: dict,
                              layout: dict[str, int]):
    """Extract one rectangular SWG basin in closed T/U/V local convention."""
    eta = np.asarray(fields["T"], dtype=np.float64)
    sj0, sj1 = layout["source_j0"], layout["source_j1"]
    si0, si1 = layout["source_i0"], layout["source_i1"]
    mask = np.asarray(geometry["mask"]["T"], dtype=bool)
    volume = (np.asarray(geometry["area"]["T"], dtype=np.float64)
              * (np.asarray(geometry["reference_thickness"]["T"],
                            dtype=np.float64)
                 + np.where(mask, eta, 0.0)) * mask)
    transports = native_face_transports(fields, geometry)
    local_volume = volume[sj0:sj1 + 1, si0:si1 + 1]
    local_u = np.zeros((local_volume.shape[0], local_volume.shape[1] + 1),
                       dtype=np.float64)
    local_v = np.zeros((local_volume.shape[0] + 1, local_volume.shape[1]),
                       dtype=np.float64)
    local_u[:, 1:] = transports["U"][sj0:sj1 + 1, si0:si1 + 1]
    local_v[1:, :] = transports["V"][sj0:sj1 + 1, si0:si1 + 1]
    return local_volume, local_u, local_v


def _state_from_restricted_extensives(
        coarse_volume: np.ndarray, coarse_transports: dict[str, np.ndarray],
        target: dict) -> dict[str, np.ndarray]:
    target_mask = np.asarray(target["mask"]["T"], dtype=bool)
    eta = np.zeros_like(coarse_volume)
    eta[target_mask] = (
        coarse_volume[target_mask]
        / np.asarray(target["area"]["T"], dtype=np.float64)[target_mask]
        - np.asarray(target["reference_thickness"]["T"],
                     dtype=np.float64)[target_mask])
    output = {"T": eta}
    for kind in ("U", "V"):
        mask = np.asarray(target["mask"][kind], dtype=bool)
        cross_section = (
            np.asarray(target["face_length"][kind], dtype=np.float64)
            * _live_face_thickness(eta, target, kind))
        velocity = np.zeros_like(eta)
        velocity[mask] = coarse_transports[kind][mask] / cross_section[mask]
        output[kind] = velocity
    return output


def filtered_restrict_cgrid(
        fields: dict[str, np.ndarray], source: dict, target: dict, *,
        sigma_source_cells: float, truncate: float = FILTER_TRUNCATE,
) -> dict[str, np.ndarray]:
    """Gaussian-filter fine extensives, then restrict them conservatively."""
    if set(fields) != {"T", "U", "V"}:
        raise ValueError("fields must be exactly T, U, and V")
    layout = _restriction_layout(source, target)
    local = _local_closed_extensives(fields, source, layout)
    volume, u_faces, v_faces = filter_closed_cgrid_extensives(
        *local, sigma_cells=sigma_source_cells, truncate=truncate)

    filtered_volume = np.zeros_like(np.asarray(source["area"]["T"]))
    sj0, sj1 = layout["source_j0"], layout["source_j1"]
    si0, si1 = layout["source_i0"], layout["source_i1"]
    filtered_volume[sj0:sj1 + 1, si0:si1 + 1] = volume
    filtered_transports = {
        "U": np.zeros_like(filtered_volume), "V": np.zeros_like(filtered_volume)}
    filtered_transports["U"][sj0:sj1 + 1, si0:si1 + 1] = u_faces[:, 1:]
    filtered_transports["V"][sj0:sj1 + 1, si0:si1 + 1] = v_faces[1:, :]
    coarse_volume = _restrict_t_cell_sum(
        filtered_volume, source, target, layout)
    coarse_transports = restrict_native_transports(
        filtered_transports, source, target, layout)
    return _state_from_restricted_extensives(
        coarse_volume, coarse_transports, target)


def filtered_restrict_transports(
        transports: dict[str, np.ndarray], source: dict, target: dict, *,
        sigma_source_cells: float, truncate: float = FILTER_TRUNCATE,
) -> dict[str, np.ndarray]:
    """Filter and restrict native face transports for temporal budgets."""
    if set(transports) != {"U", "V"}:
        raise ValueError("transports must be exactly U and V")
    layout = _restriction_layout(source, target)
    sj0, sj1 = layout["source_j0"], layout["source_j1"]
    si0, si1 = layout["source_i0"], layout["source_i1"]
    ny = sj1 - sj0 + 1
    nx = si1 - si0 + 1
    local_u = np.zeros((ny, nx + 1), dtype=np.float64)
    local_v = np.zeros((ny + 1, nx), dtype=np.float64)
    local_u[:, 1:] = np.asarray(
        transports["U"], dtype=np.float64)[sj0:sj1 + 1, si0:si1 + 1]
    local_v[1:, :] = np.asarray(
        transports["V"], dtype=np.float64)[sj0:sj1 + 1, si0:si1 + 1]
    _, filtered_u, filtered_v = filter_closed_cgrid_extensives(
        np.zeros((ny, nx), dtype=np.float64), local_u, local_v,
        sigma_cells=sigma_source_cells, truncate=truncate)
    embedded = {"U": np.zeros_like(np.asarray(source["area"]["T"])),
                "V": np.zeros_like(np.asarray(source["area"]["T"]))}
    embedded["U"][sj0:sj1 + 1, si0:si1 + 1] = filtered_u[:, 1:]
    embedded["V"][sj0:sj1 + 1, si0:si1 + 1] = filtered_v[1:, :]
    return restrict_native_transports(embedded, source, target, layout)


def conservative_restrict_daily_mean(
        mean_eta: np.ndarray, mean_transports: dict[str, np.ndarray],
        source: dict, target: dict) -> dict[str, np.ndarray]:
    """Restrict ``mean(h*u)`` and ``mean(h*v)``, never products of means."""
    if set(mean_transports) != {"U", "V"}:
        raise ValueError("mean transports must be exactly U and V")
    layout = _restriction_layout(source, target)
    eta = np.asarray(mean_eta, dtype=np.float64)
    source_mask = np.asarray(source["mask"]["T"], dtype=bool)
    if eta.shape != source_mask.shape or not np.all(np.isfinite(eta)):
        raise ValueError("mean SSH must be finite and have the source shape")
    volume = (np.asarray(source["area"]["T"], dtype=np.float64)
              * (np.asarray(source["reference_thickness"]["T"])
                 + np.where(source_mask, eta, 0.0)) * source_mask)
    coarse_volume = _restrict_t_cell_sum(volume, source, target, layout)
    target_mask = np.asarray(target["mask"]["T"], dtype=bool)
    coarse_eta = np.zeros_like(coarse_volume)
    coarse_eta[target_mask] = (
        coarse_volume[target_mask] / np.asarray(target["area"]["T"])[target_mask]
        - np.asarray(target["reference_thickness"]["T"])[target_mask])
    coarse_transports = restrict_native_transports(
        mean_transports, source, target, layout)
    output = {"T": coarse_eta, "transport_U": coarse_transports["U"],
              "transport_V": coarse_transports["V"]}
    for kind in ("U", "V"):
        mask = np.asarray(target["mask"][kind], dtype=bool)
        denominator = (np.asarray(target["face_length"][kind])
                       * _live_face_thickness(coarse_eta, target, kind))
        velocity = np.zeros_like(coarse_eta)
        velocity[mask] = coarse_transports[kind][mask] / denominator[mask]
        output[kind] = velocity
    return output


def temporal_continuity_residual(
        eta_start: np.ndarray, eta_end: np.ndarray,
        mean_transports: dict[str, np.ndarray], geometry: dict,
        interval_s: float) -> np.ndarray:
    """Return ``d eta/dt + div(mean transport)`` on the native T grid."""
    if not np.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("continuity interval must be positive finite")
    start = np.asarray(eta_start, dtype=np.float64)
    end = np.asarray(eta_end, dtype=np.float64)
    mask = np.asarray(geometry["mask"]["T"], dtype=bool)
    if start.shape != mask.shape or end.shape != mask.shape:
        raise ValueError("continuity SSH shapes differ from the T grid")
    residual = (end - start) / interval_s + _transport_divergence(
        mean_transports, geometry)
    return np.where(mask, residual, 0.0)


def build_jax_conservative_restrictor(source: dict, target: dict):
    """Build the same conservative restriction for an in-scan JAX target."""
    import jax.numpy as jnp

    layout = _restriction_layout(source, target)
    f = layout["factor"]
    sj0, sj1 = layout["source_j0"], layout["source_j1"]
    si0, si1 = layout["source_i0"], layout["source_i1"]
    tj0, tj1 = layout["target_j0"], layout["target_j1"]
    ti0, ti1 = layout["target_i0"], layout["target_i1"]
    ny, nx = layout["target_ny"], layout["target_nx"]
    source_u_columns = jnp.asarray(
        si0 + f * np.arange(1, nx, dtype=int) - 1)
    source_v_rows = jnp.asarray(
        sj0 + f * np.arange(1, ny, dtype=int) - 1)

    source_tmask = jnp.asarray(source["mask"]["T"], dtype=jnp.float64)
    target_masks = {
        kind: jnp.asarray(target["mask"][kind], dtype=jnp.float64)
        for kind in ("T", "U", "V")
    }
    source_area_t = jnp.asarray(source["area"]["T"], dtype=jnp.float64)
    target_area_t = jnp.asarray(target["area"]["T"], dtype=jnp.float64)
    source_reference_t = jnp.asarray(
        source["reference_thickness"]["T"], dtype=jnp.float64)
    target_reference_t = jnp.asarray(
        target["reference_thickness"]["T"], dtype=jnp.float64)

    def live_face_thickness(eta, geometry, kind, face_mask):
        area_t = jnp.asarray(geometry["area"]["T"], dtype=jnp.float64)
        weighted_eta = area_t * eta * jnp.asarray(
            geometry["mask"]["T"], dtype=jnp.float64)
        axis = 1 if kind == "U" else 0
        numerator = 0.5 * (weighted_eta + jnp.roll(
            weighted_eta, -1, axis=axis))
        reference_depth = jnp.asarray(
            geometry["reference_transport_depth"][kind], dtype=jnp.float64)
        area_face = jnp.asarray(
            geometry["area"][kind], dtype=jnp.float64)
        reference_thickness = jnp.asarray(
            geometry["reference_thickness"][kind], dtype=jnp.float64)
        safe = jnp.where(face_mask > 0.0, reference_depth * area_face, 1.0)
        ratio = numerator / safe * face_mask
        return reference_thickness * (1.0 + ratio)

    def restrict(eta, u_native, v_native):
        source_volume = source_area_t * (
            source_reference_t + eta * source_tmask) * source_tmask
        volume_block = source_volume[sj0:sj1 + 1, si0:si1 + 1]
        coarse_volume = volume_block.reshape(ny, f, nx, f).sum(axis=(1, 3))
        coarse_eta_block = (
            coarse_volume / target_area_t[tj0:tj1 + 1, ti0:ti1 + 1]
            - target_reference_t[tj0:tj1 + 1, ti0:ti1 + 1]
        )
        coarse_eta = jnp.zeros_like(target_area_t).at[
            tj0:tj1 + 1, ti0:ti1 + 1].set(coarse_eta_block)

        source_umask = jnp.asarray(source["mask"]["U"], dtype=jnp.float64)
        source_vmask = jnp.asarray(source["mask"]["V"], dtype=jnp.float64)
        fine_u_transport = (
            jnp.asarray(source["face_length"]["U"], dtype=jnp.float64)
            * live_face_thickness(eta, source, "U", source_umask)
            * u_native * source_umask
        )
        fine_v_transport = (
            jnp.asarray(source["face_length"]["V"], dtype=jnp.float64)
            * live_face_thickness(eta, source, "V", source_vmask)
            * v_native * source_vmask
        )
        u_block = fine_u_transport[
            sj0:sj1 + 1, source_u_columns].reshape(ny, f, nx - 1).sum(axis=1)
        v_block = fine_v_transport[
            source_v_rows, si0:si1 + 1].reshape(ny - 1, nx, f).sum(axis=2)
        coarse_u_transport = jnp.zeros_like(target_area_t).at[
            tj0:tj1 + 1, ti0:ti1].set(u_block)
        coarse_v_transport = jnp.zeros_like(target_area_t).at[
            tj0:tj1, ti0:ti1 + 1].set(v_block)

        target_umask = target_masks["U"]
        target_vmask = target_masks["V"]
        u_denominator = (
            jnp.asarray(target["face_length"]["U"], dtype=jnp.float64)
            * live_face_thickness(coarse_eta, target, "U", target_umask)
        )
        v_denominator = (
            jnp.asarray(target["face_length"]["V"], dtype=jnp.float64)
            * live_face_thickness(coarse_eta, target, "V", target_vmask)
        )
        coarse_u = coarse_u_transport / jnp.where(
            target_umask > 0.0, u_denominator, 1.0) * target_umask
        coarse_v = coarse_v_transport / jnp.where(
            target_vmask > 0.0, v_denominator, 1.0) * target_vmask
        return coarse_eta * target_masks["T"], coarse_u, coarse_v

    return restrict


def build_jax_filtered_restrictor(
        source: dict, target: dict, *, sigma_source_cells: float,
        truncate: float = FILTER_TRUNCATE):
    """Build the closed-wall filtered restriction for an in-scan sampler."""
    import jax.numpy as jnp

    layout = _restriction_layout(source, target)
    f = layout["factor"]
    sj0, sj1 = layout["source_j0"], layout["source_j1"]
    si0, si1 = layout["source_i0"], layout["source_i1"]
    tj0, tj1 = layout["target_j0"], layout["target_j1"]
    ti0, ti1 = layout["target_i0"], layout["target_i1"]
    ny, nx = layout["target_ny"], layout["target_nx"]
    fine_ny, fine_nx = f * ny, f * nx
    kernel = jnp.asarray(gaussian_kernel_1d(
        sigma_cells=sigma_source_cells, truncate=truncate))

    source_tmask = jnp.asarray(source["mask"]["T"], dtype=jnp.float64)
    source_umask = jnp.asarray(source["mask"]["U"], dtype=jnp.float64)
    source_vmask = jnp.asarray(source["mask"]["V"], dtype=jnp.float64)
    source_area_t = jnp.asarray(source["area"]["T"], dtype=jnp.float64)
    source_reference_t = jnp.asarray(
        source["reference_thickness"]["T"], dtype=jnp.float64)
    target_area_t = jnp.asarray(target["area"]["T"], dtype=jnp.float64)
    target_reference_t = jnp.asarray(
        target["reference_thickness"]["T"], dtype=jnp.float64)
    target_masks = {
        kind: jnp.asarray(target["mask"][kind], dtype=jnp.float64)
        for kind in ("T", "U", "V")}

    def live_h(eta, geometry, kind, mask):
        area_t = jnp.asarray(geometry["area"]["T"], dtype=jnp.float64)
        weighted = area_t * eta * jnp.asarray(
            geometry["mask"]["T"], dtype=jnp.float64)
        axis = 1 if kind == "U" else 0
        numerator = 0.5 * (weighted + jnp.roll(weighted, -1, axis=axis))
        denominator = (jnp.asarray(
            geometry["reference_transport_depth"][kind], dtype=jnp.float64)
            * jnp.asarray(geometry["area"][kind], dtype=jnp.float64))
        ratio = numerator / jnp.where(mask > 0.0, denominator, 1.0) * mask
        return jnp.asarray(
            geometry["reference_thickness"][kind], dtype=jnp.float64) * (
                1.0 + ratio)

    def restrict(eta, u_native, v_native):
        volume = source_area_t * (
            source_reference_t + eta * source_tmask) * source_tmask
        transport_u = (jnp.asarray(source["face_length"]["U"])
                       * live_h(eta, source, "U", source_umask)
                       * u_native * source_umask)
        transport_v = (jnp.asarray(source["face_length"]["V"])
                       * live_h(eta, source, "V", source_vmask)
                       * v_native * source_vmask)
        local_volume = volume[sj0:sj1 + 1, si0:si1 + 1]
        local_u = jnp.zeros((fine_ny, fine_nx + 1), dtype=eta.dtype).at[
            :, 1:].set(transport_u[sj0:sj1 + 1, si0:si1 + 1])
        local_v = jnp.zeros((fine_ny + 1, fine_nx), dtype=eta.dtype).at[
            1:, :].set(transport_v[sj0:sj1 + 1, si0:si1 + 1])
        filtered_volume, filtered_u, filtered_v = (
            filter_closed_cgrid_extensives_array(
                local_volume, local_u, local_v, kernel=kernel))

        coarse_volume_block = filtered_volume.reshape(
            ny, f, nx, f).sum(axis=(1, 3))
        coarse_eta_block = (
            coarse_volume_block / target_area_t[tj0:tj1 + 1, ti0:ti1 + 1]
            - target_reference_t[tj0:tj1 + 1, ti0:ti1 + 1])
        coarse_eta = jnp.zeros_like(target_area_t).at[
            tj0:tj1 + 1, ti0:ti1 + 1].set(coarse_eta_block)

        selected_u = filtered_u[:, f * jnp.arange(1, nx)].reshape(
            ny, f, nx - 1).sum(axis=1)
        selected_v = filtered_v[f * jnp.arange(1, ny), :].reshape(
            ny - 1, nx, f).sum(axis=2)
        coarse_q_u = jnp.zeros_like(target_area_t).at[
            tj0:tj1 + 1, ti0:ti1].set(selected_u)
        coarse_q_v = jnp.zeros_like(target_area_t).at[
            tj0:tj1, ti0:ti1 + 1].set(selected_v)
        target_umask, target_vmask = target_masks["U"], target_masks["V"]
        cross_u = (jnp.asarray(target["face_length"]["U"])
                   * live_h(coarse_eta, target, "U", target_umask))
        cross_v = (jnp.asarray(target["face_length"]["V"])
                   * live_h(coarse_eta, target, "V", target_vmask))
        coarse_u = coarse_q_u / jnp.where(
            target_umask > 0.0, cross_u, 1.0) * target_umask
        coarse_v = coarse_q_v / jnp.where(
            target_vmask > 0.0, cross_v, 1.0) * target_vmask
        return coarse_eta * target_masks["T"], coarse_u, coarse_v

    return restrict


def _transport_divergence(transports: dict[str, np.ndarray], geometry: dict,
                          ) -> np.ndarray:
    """Finite-volume divergence from native east/north transport records."""
    u = np.asarray(transports["U"], dtype=np.float64)
    v = np.asarray(transports["V"], dtype=np.float64)
    net = u - np.roll(u, 1, axis=1) + v - np.roll(v, 1, axis=0)
    mask = np.asarray(geometry["mask"]["T"], dtype=bool)
    out = np.zeros_like(net)
    out[mask] = net[mask] / np.asarray(
        geometry["area"]["T"], dtype=np.float64)[mask]
    return out


def transport_divergence(transports: dict[str, np.ndarray], geometry: dict,
                         ) -> np.ndarray:
    """Public finite-volume divergence for conservative diagnostics."""
    return _transport_divergence(transports, geometry)


def continuity_commutator(fields: dict[str, np.ndarray], source: dict,
                           target: dict, *, coarse_fields=None) -> dict:
    """Compare restricted fine and coarse finite-volume continuity [m/s]."""
    layout = _restriction_layout(source, target)
    fine_transports = native_face_transports(fields, source)
    fine_divergence = _transport_divergence(fine_transports, source)
    restricted_fine_divergence = _restrict_t_cell_sum(
        fine_divergence * np.asarray(source["area"]["T"], dtype=np.float64),
        source, target, layout,
    ) / np.asarray(target["area"]["T"], dtype=np.float64)
    if coarse_fields is None:
        coarse_fields = conservative_restrict_cgrid(fields, source, target)
    coarse_divergence = _transport_divergence(
        native_face_transports(coarse_fields, target), target)
    residual = coarse_divergence - restricted_fine_divergence
    wet = np.asarray(target["mask"]["T"], dtype=bool)
    tj0, tj1 = layout["target_j0"], layout["target_j1"]
    ti0, ti1 = layout["target_i0"], layout["target_i1"]
    wall = np.zeros_like(wet)
    width = min(4, layout["target_ny"] // 2, layout["target_nx"] // 2)
    wall[tj0:tj0 + width, ti0:ti1 + 1] = True
    wall[tj1 - width + 1:tj1 + 1, ti0:ti1 + 1] = True
    wall[tj0:tj1 + 1, ti0:ti0 + width] = True
    wall[tj0:tj1 + 1, ti1 - width + 1:ti1 + 1] = True
    wall &= wet
    interior = wet & ~wall

    def summary(mask):
        values = residual[mask]
        return {
            "count": int(values.size),
            "rms_m_s": float(np.sqrt(np.mean(values * values))),
            "max_abs_m_s": float(np.max(np.abs(values))),
        }

    return {
        "residual_m_s": residual,
        "restricted_fine_divergence_m_s": restricted_fine_divergence,
        "coarse_divergence_m_s": coarse_divergence,
        "wet": summary(wet),
        "interior": summary(interior),
        "outer_four_cells": summary(wall),
    }


def require_continuity_commutes(fields: dict[str, np.ndarray], source: dict,
                                target: dict, *, atol_m_s: float,
                                coarse_fields=None) -> dict:
    """Fail closed when interior or wall continuity exceeds the declared bar."""
    if not np.isfinite(atol_m_s) or atol_m_s <= 0.0:
        raise ValueError("continuity tolerance must be positive finite")
    result = continuity_commutator(
        fields, source, target, coarse_fields=coarse_fields)
    for region in ("interior", "outer_four_cells"):
        if result[region]["max_abs_m_s"] > atol_m_s:
            raise ValueError(
                f"non-conservative restriction in {region}: "
                f"{result[region]['max_abs_m_s']:.17g} m/s exceeds "
                f"{atol_m_s:.17g} m/s")
    return result


def gaussian_smooth_restricted_scalar(field: np.ndarray, mask: np.ndarray, *,
                                      sigma_cells: float,
                                      truncate: float = FILTER_TRUNCATE,
                                      ) -> np.ndarray:
    """Gaussian-smooth one restricted scalar with reflecting closed walls.

    This is for derived relative-vorticity targets, never for the conservative
    T/U/V state.  Reflection makes the rectangular closed wall a no-flux
    boundary; there is no mask normalization or a-posteriori mean correction.
    ``sigma_cells`` is the Gaussian standard deviation in coarse cells.
    """
    values = np.asarray(field, dtype=np.float64)
    wet = np.asarray(mask, dtype=bool)
    if values.shape != wet.shape or not np.all(np.isfinite(values)):
        raise ValueError("Gaussian scalar must be finite and match its mask")
    if not np.isfinite(sigma_cells) or sigma_cells <= 0.0:
        raise ValueError("sigma_cells must be positive finite")
    j0, j1, i0, i1 = _wet_rectangle(wet, label="Gaussian")
    filtered = gaussian_filter(
        values[j0:j1 + 1, i0:i1 + 1], sigma_cells,
        mode="reflect", truncate=truncate)
    out = np.zeros_like(values)
    out[j0:j1 + 1, i0:i1 + 1] = filtered
    return out


def restricted_relative_vorticity(fields: dict[str, np.ndarray], geometry: dict,
                                   *, gaussian_sigma_cells: float | None = None,
                                   ) -> np.ndarray:
    """Shared C-grid relative vorticity, optionally Gaussian-smoothed."""
    import jax.numpy as jnp
    from legoesm.ocean.diagnostics import relative_vorticity_cell_centre

    shape = np.asarray(geometry["mask"]["T"]).shape
    u = np.zeros((shape[0], shape[1] + 1), dtype=np.float64)
    v = np.zeros((shape[0] + 1, shape[1]), dtype=np.float64)
    u[:, 1:] = np.asarray(fields["U"], dtype=np.float64)
    v[1:, :] = np.asarray(fields["V"], dtype=np.float64)
    zeta = np.asarray(relative_vorticity_cell_centre(
        jnp.asarray(u), jnp.asarray(v), geometry["grid"]), dtype=np.float64)
    wet = np.asarray(geometry["mask"]["T"], dtype=bool)
    zeta = np.where(wet, zeta, 0.0)
    if gaussian_sigma_cells is not None:
        zeta = gaussian_smooth_restricted_scalar(
            zeta, wet, sigma_cells=gaussian_sigma_cells)
    return zeta


def audit_snapshot(path: Path, source_manifest_path: Path, *, commit: str,
                   plant_round1: bool = False) -> dict:
    """Content-pin and audit one real fine snapshot against the continuity bar."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        SWG_MADDISON_NAMELIST,
    )

    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("commit must be a forty-character lowercase Git SHA")
    manifest = json.loads(source_manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("source manifest must be complete")
    expected = manifest.get("daily_sha256", {}).get(path.name)
    digest = sha256(path)
    if expected != digest:
        raise ValueError(f"source manifest does not admit {path.name}")
    with np.load(path, allow_pickle=False) as saved:
        fields = {
            "T": np.asarray(saved["ssh_m"], dtype=np.float64),
            "U": np.asarray(saved["u_m_s"], dtype=np.float64),
            "V": np.asarray(saved["v_m_s"], dtype=np.float64),
        }
        source_day = int(saved["day"])
        source_step = int(saved["step"])
    source = cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = cgrid_geometry(SWG_DG50_NAMELIST)
    coarse_fields = None
    if plant_round1:
        coarse_fields = coarse_grain_cgrid(
            fields, source, target,
            gaussian_length_m=float(target["spacing_m"]))
    result = require_continuity_commutes(
        fields, source, target, atol_m_s=CONTINUITY_BAR_M_S,
        coarse_fields=coarse_fields)
    return {
        "schema": "legoesm-swg-conservative-continuity-audit-v1",
        "status": "PASS",
        "legoesm_git_sha": commit,
        "source_path": str(path.resolve()),
        "source_sha256": digest,
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_day": source_day,
        "source_step": source_step,
        "plant_round1": plant_round1,
        "continuity_bar_m_s": CONTINUITY_BAR_M_S,
        "regions": {
            name: result[name]
            for name in ("wet", "interior", "outer_four_cells")
        },
        "convention": (
            "sum fine T-cell volumes and native e2*h*u/e1*h*v face transports; "
            "no state Gaussian and no a-posteriori mean restoration"),
    }


def coarse_grain_cgrid(fields: dict[str, np.ndarray], source: dict,
                       target: dict, *, gaussian_length_m: float) -> dict[str, np.ndarray]:
    """Round-1 non-conservative Gaussian diagnostic; not a training target."""
    expected = {"T", "U", "V"}
    if set(fields) != expected:
        raise ValueError(f"fields must be exactly {sorted(expected)}")
    sigma_cells = gaussian_length_m / float(source["spacing_m"])
    output = {}
    for kind in ("T", "U", "V"):
        filtered = masked_gaussian_filter(
            fields[kind], source["mask"][kind], sigma_cells=sigma_cells)
        output[kind] = area_average(
            filtered, source["mask"][kind], target["mask"][kind],
            source_x_m=source["x"][kind], source_y_m=source["y"][kind],
            target_x_m=target["x"][kind], target_y_m=target["y"][kind],
            source_spacing_m=float(source["spacing_m"]),
            target_spacing_m=float(target["spacing_m"]))
    return output


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def truth_source_namelist(source_deck: str):
    """Resolve one pre-registered truth resolution for dg50 restriction."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_MADDISON_NAMELIST
    from legoesm.ocean.fidelity.swg_online_decks import SWG_DG3125_NAMELIST

    decks = {"dg625": SWG_MADDISON_NAMELIST, "dg3125": SWG_DG3125_NAMELIST}
    if source_deck not in decks:
        raise ValueError(f"unsupported truth source deck {source_deck!r}")
    return decks[source_deck]


def filter_truth_directory(input_root: Path, output_root: Path, *,
                           commit: str, source_deck: str = "dg625") -> dict:
    """Write conservatively restricted snapshots and diagnostic daily means."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
    )

    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("commit must be a forty-character lowercase Git SHA")
    manifest_path = input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" \
            or manifest.get("sampling") != "daily_mean" \
            or manifest.get("completed_days") != manifest.get("days"):
        raise ValueError("input must be a complete daily-mean fine run")
    resolved_source = truth_source_namelist(source_deck)
    expected_source = json.loads(json.dumps(resolved_source._asdict()))
    source_namelist = manifest.get("namelist", {})
    for field in expected_source.keys() - {"nn_itend", "nn_stock"}:
        if source_namelist.get(field) != expected_source[field]:
            raise ValueError(f"input fine deck differs at {field}")
    source = cgrid_geometry(resolved_source)
    target = cgrid_geometry(SWG_DG50_NAMELIST)
    gaussian_length_m = float(target["spacing_m"])
    output_root.mkdir(parents=True)
    roots = {name: output_root / name for name in ("snapshots", "daily_means")}
    for root in roots.values():
        root.mkdir()
    source_hashes = manifest.get("daily_sha256", {})
    output_hashes = {name: {} for name in roots}
    days = int(manifest["days"])
    base = {
        "schema": "legoesm-swg-conservative-truth-v2",
        "status": "pending",
        "completed_days": 0,
        "days": days,
        "legoesm_git_sha": commit,
        "source_root": str(input_root.resolve()),
        "source_manifest_sha256": sha256(manifest_path),
        "source_daily_sha256": source_hashes,
        "source_namelist": source_namelist,
        "source_deck": source_deck,
        "target_namelist": SWG_DG50_NAMELIST._asdict(),
        "filter": {
            "name": "exact nested T-volume and native face-transport restriction",
            "gaussian_length_m": gaussian_length_m,
            "sigma_fine_cells": gaussian_length_m / float(source["spacing_m"]),
            "truncate_sigma": FILTER_TRUNCATE,
            "state_gaussian": "none",
            "derived_vorticity_gaussian": (
                "optional after restriction; reflecting closed walls; sigma is "
                "a standard deviation in coarse cells"),
            "mean_conservation": "finite-volume construction; no mean restoration",
            "daily_mean_transport": (
                "exact arithmetic mean of instantaneous h*u and h*v, then "
                "face-transport restriction"),
            "staggering": {"ssh_m": "T", "u_m_s": "U", "v_m_s": "V"},
        },
    }
    for day in range(1, days + 1):
        name = f"day{day:04d}.npz"
        path = input_root / name
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if source_hashes.get(name) != digest:
            raise ValueError(f"input digest mismatch: {name}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            snapshot = conservative_restrict_cgrid(
                {"T": saved["ssh_m"], "U": saved["u_m_s"],
                 "V": saved["v_m_s"]}, source, target)
            required_mean = {
                "u_daily_mean_transport_m3_s",
                "v_daily_mean_transport_m3_s",
            }
            if not required_mean <= set(saved.files):
                raise ValueError(
                    "daily mean restriction requires archived mean(h*u) and "
                    "mean(h*v); product-of-means input is refused")
            daily_mean = conservative_restrict_daily_mean(
                saved["ssh_daily_mean_m"],
                {"U": saved["u_daily_mean_transport_m3_s"],
                 "V": saved["v_daily_mean_transport_m3_s"]}, source, target)
            metadata = {
                "schema": np.str_("legoesm-swg-conservative-truth-v2"),
                "source_day": np.int64(saved["day"]),
                "source_step": np.int64(saved["step"]),
                "source_dt_s": np.float64(saved["dt_s"]),
                "day": np.int64(day),
                "step": np.int64(day * int(86_400 / SWG_DG50_NAMELIST.rn_Dt_s)),
                "dt_s": np.float64(SWG_DG50_NAMELIST.rn_Dt_s),
            }
        records = {
            "snapshots": metadata | {"ssh_m": snapshot["T"],
                                      "u_m_s": snapshot["U"],
                                      "v_m_s": snapshot["V"]},
            "daily_means": metadata | {
                "daily_mean_sampling": np.str_(
                    "restricted exact fine arithmetic-mean volume and face transport"),
                "daily_mean_sample_count": np.int64(
                    86_400 / resolved_source.rn_Dt_s),
                "ssh_daily_mean_m": daily_mean["T"],
                "u_daily_mean_m_s": daily_mean["U"],
                "v_daily_mean_m_s": daily_mean["V"],
                "u_daily_mean_transport_m3_s": daily_mean["transport_U"],
                "v_daily_mean_transport_m3_s": daily_mean["transport_V"]},
        }
        for sampling, arrays in records.items():
            temporary = roots[sampling] / f".{name}.tmp"
            with temporary.open("wb") as stream:
                np.savez(stream, **arrays)
            temporary.replace(roots[sampling] / name)
            output_hashes[sampling][name] = sha256(roots[sampling] / name)
        for sampling, root in roots.items():
            _write_json_atomic(root / "manifest.pending.json", base | {
                "sampling": ("daily_snapshot" if sampling == "snapshots"
                             else "daily_mean"),
                "completed_days": day,
                "output_daily_sha256": output_hashes[sampling],
            })
    manifests = {}
    for sampling, root in roots.items():
        pending = root / "manifest.pending.json"
        complete = json.loads(pending.read_text()) | {"status": "complete"}
        _write_json_atomic(pending, complete)
        pending.replace(root / "manifest.json")
        manifests[sampling] = complete
    return manifests


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-snapshot", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-deck", choices=("dg625", "dg3125"),
                        default="dg625")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--plant-round1", action="store_true")
    args = parser.parse_args(argv)
    try:
        from legoesm.ocean.fidelity.provenance import git_sha

        commit = git_sha()
        if args.audit_snapshot is not None:
            if args.source_manifest is None or args.json is None:
                raise ValueError("audit requires --source-manifest and --json")
            if args.input is not None or args.output is not None:
                raise ValueError("audit excludes --input and --output")
            if args.source_deck != "dg625":
                raise ValueError("audit currently admits only source deck dg625")
            report = audit_snapshot(
                args.audit_snapshot, args.source_manifest, commit=commit,
                plant_round1=args.plant_round1)
            _write_json_atomic(args.json, report)
            print(
                "PASS continuity "
                f"interior_max={report['regions']['interior']['max_abs_m_s']:.3e} "
                "wall_max="
                f"{report['regions']['outer_four_cells']['max_abs_m_s']:.3e}")
            return 0
        if args.input is None or args.output is None:
            raise ValueError("conversion requires --input and --output")
        if args.source_manifest is not None or args.json is not None \
                or args.plant_round1:
            raise ValueError("conversion excludes audit-only arguments")
        manifests = filter_truth_directory(
            args.input, args.output, commit=commit,
            source_deck=args.source_deck)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 2
    print(f"COMPLETED days={manifests['snapshots']['days']} output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
