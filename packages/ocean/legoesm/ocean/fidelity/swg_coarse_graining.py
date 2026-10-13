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


def filter_truth_directory(input_root: Path, output_root: Path, *,
                           commit: str) -> dict:
    """Write conservatively restricted snapshots and diagnostic daily means."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        SWG_MADDISON_NAMELIST,
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
    expected_source = json.loads(json.dumps(SWG_MADDISON_NAMELIST._asdict()))
    source_namelist = manifest.get("namelist", {})
    for field in expected_source.keys() - {"nn_itend", "nn_stock"}:
        if source_namelist.get(field) != expected_source[field]:
            raise ValueError(f"input fine deck differs at {field}")
    source = cgrid_geometry(SWG_MADDISON_NAMELIST)
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
            "daily_mean_caveat": (
                "restriction of archived mean SSH/U/V; exact mean face transport "
                "is unavailable because mean thickness-times-velocity was not archived"),
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
            daily_mean = conservative_restrict_cgrid(
                {"T": saved["ssh_daily_mean_m"],
                 "U": saved["u_daily_mean_m_s"],
                 "V": saved["v_daily_mean_m_s"]}, source, target)
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
                    "restricted archived fine arithmetic-mean state; not mean transport"),
                "daily_mean_sample_count": np.int64(1_728),
                "ssh_daily_mean_m": daily_mean["T"],
                "u_daily_mean_m_s": daily_mean["U"],
                "v_daily_mean_m_s": daily_mean["V"]},
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
            args.input, args.output, commit=commit)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 2
    print(f"COMPLETED days={manifests['snapshots']['days']} output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
