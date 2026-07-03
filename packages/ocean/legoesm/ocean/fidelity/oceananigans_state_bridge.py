"""Oceananigans snapshot -> legoESM lat-lon C-grid state bridge.

The Oceananigans counterpart of :mod:`mitgcm_state_bridge`. Oceananigans is an
Arakawa **C-grid** model like MITgcm, so the face placement transfers almost
unchanged. The conventions this bridge reconciles (all "mimicry-only glue" — it
lives HERE, never in the model, per ``oracle_recipe_strategy.md``):

* **Vertical order — the one real difference from MITgcm.** Oceananigans stores
  ``z`` increasing UPWARD with ``k=1`` at the BOTTOM (``z`` runs from ``-H`` to
  ``0``); legoESM uses ``k=0`` at the SURFACE, increasing downward. So 3-D fields
  are **z-reversed** (like the Veros bridge), unlike the MITgcm bridge which is
  already top-down.
* **Axis order.** The NetCDF reader returns ``(time, z, y, x)`` for 3-D fields
  and ``(time, y, x)`` for the 2-D free surface. After selecting a time index a
  3-D field is ``(z, y, x)``; legoESM wants ``(lat=ny, lon=nx, lev=nz)`` — a
  ``moveaxis(0, -1)`` then the z-reverse.
* **C-grid faces.** Oceananigans ``u`` sits at ``(Face, Center, Center)`` = the
  WEST face, ``v`` at ``(Center, Face, Center)`` = the SOUTH face — identical to
  legoESM ``u_face``/``v_face``. Oceananigans writes ``N+1`` face points in a
  ``Bounded`` direction and ``N`` in a ``Periodic`` one; legoESM always carries
  ``nx+1``/``ny+1``. The face helpers therefore ADAPT: append the boundary face
  (zonal wrap if ``cyclic_x`` else a zero wall) only when the oracle array is one
  short, and pass through when it already includes the boundary face.

Field-name translation uses the cross-oracle concept registry
(``oracle_to_canonical("oceananigans")``): ``u/v/w`` -> same, ``eta``/``η`` ->
``eta``. Buoyancy ``b`` -> ``T`` is a *physics* conversion (linear EOS), handled
in the recipe/setup for the stratified cases, NOT here. Fields absent from the
snapshot (e.g. tracers for the homogeneous barotropic gyre) stay at ``base_state``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from legoesm.ocean.fidelity.concept_registry import oracle_to_canonical
from legoesm.ocean.fidelity.oceananigans_runner import OceananigansResult
from legoesm.ocean.state import LatLonCGridOceanState

# Canonical state fields this bridge places. (b->T is a physics conversion done
# upstream, so it is not in this set; the bridge only places already-canonical
# state fields.)
_PLACEABLE = frozenset({"u", "v", "w", "T", "S", "eta"})


class OceananigansStateBridgeOutput(NamedTuple):
    """Result of :func:`oceananigans_snapshot_to_legoesm_state`."""

    state: LatLonCGridOceanState
    info: dict


def _to_latlon_lev_zrev(arr: np.ndarray) -> np.ndarray:
    """``(nz, ny, nx)`` -> ``(ny, nx, nz)`` WITH the vertical reversed;
    ``(ny, nx)`` passes through unchanged.

    Oceananigans 3-D fields come off disk vertical-first AND bottom-up; legoESM
    wants vertical-last AND surface-first, so reverse the (now last) z axis.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        return np.moveaxis(arr, 0, -1)[:, :, ::-1]
    if arr.ndim == 2:
        return arr
    raise ValueError(
        f"expected a 2-D or 3-D Oceananigans field, got ndim={arr.ndim}"
    )


def _conform_to(arr: np.ndarray, target_shape: tuple[int, ...], *, what: str) -> np.ndarray:
    """Reconcile a placed field with the legoESM target shape (single-level
    2-D<->3-D), mirroring the MITgcm bridge. Any other mismatch raises."""
    if arr.shape == target_shape:
        return arr
    if arr.ndim == 2 and len(target_shape) == 3 and target_shape[2] == 1 \
            and arr.shape == target_shape[:2]:
        return arr[:, :, None]
    if arr.ndim == 3 and arr.shape[2] == 1 and len(target_shape) == 2 \
            and arr.shape[:2] == target_shape:
        return arr[:, :, 0]
    raise ValueError(
        f"{what}: Oceananigans field shape {arr.shape} cannot be conformed to "
        f"legoESM target shape {target_shape}"
    )


def _u_cell_to_face(u_west: np.ndarray, target_nx: int, *, cyclic_x: bool) -> np.ndarray:
    """West-face ``u`` ``(ny, nx_o[, nz])`` -> legoESM ``(ny, target_nx[, nz])``.

    Adapts to Oceananigans' Bounded(N+1)/Periodic(N) face count: append the east
    boundary face (zonal wrap if ``cyclic_x`` else a zero wall) only when one
    short; pass through when the oracle already includes it.
    """
    nx_o = u_west.shape[1]
    if nx_o == target_nx:
        return u_west
    if nx_o == target_nx - 1:
        extra = u_west[:, :1, ...] if cyclic_x else np.zeros_like(u_west[:, :1, ...])
        return np.concatenate([u_west, extra], axis=1)
    raise ValueError(
        f"u: oracle x-face count {nx_o} is neither target_nx={target_nx} nor "
        f"target_nx-1={target_nx - 1}"
    )


def _v_cell_to_face(v_south: np.ndarray, target_ny: int) -> np.ndarray:
    """South-face ``v`` ``(ny_o, nx[, nz])`` -> legoESM ``(target_ny, nx[, nz])``.

    Meridional boundaries are closed (Bounded) for the cases we mirror; append a
    zero north-wall row only when one short, pass through otherwise.
    """
    ny_o = v_south.shape[0]
    if ny_o == target_ny:
        return v_south
    if ny_o == target_ny - 1:
        zero = np.zeros_like(v_south[:1, ...])
        return np.concatenate([v_south, zero], axis=0)
    raise ValueError(
        f"v: oracle y-face count {ny_o} is neither target_ny={target_ny} nor "
        f"target_ny-1={target_ny - 1}"
    )


def oceananigans_snapshot_to_legoesm_state(
    result: OceananigansResult,
    base_state: LatLonCGridOceanState,
    *,
    time_index: int = -1,
    cyclic_x: bool | None = None,
) -> OceananigansStateBridgeOutput:
    """Initialise a legoESM lat-lon C-grid state from an Oceananigans snapshot.

    Parameters
    ----------
    result
        Output of :func:`oceananigans_runner.load_oceananigans_reference`.
    base_state
        Rest legoESM state at the matching grid; provides grid / mask templates
        and the values for any field the snapshot omits.
    time_index
        Which snapshot to extract (default last).
    cyclic_x
        Override the zonal topology; defaults to ``result.grid_metadata['cyclic_x']``
        (the case spec). Wrap ``u`` if periodic, zero east wall if bounded.

    Returns
    -------
    OceananigansStateBridgeOutput
    """
    name_map = oracle_to_canonical("oceananigans")
    if cyclic_x is None:
        cyclic_x = bool(result.grid_metadata.get("cyclic_x", False))
    has_time_axis = int(np.asarray(result.times_s).size) > 1

    placed: dict[str, np.ndarray] = {}
    placed_from: dict[str, str] = {}
    skipped: list[str] = []
    for raw_name, raw_arr in result.variables.items():
        canon = name_map.get(raw_name)
        if canon is None or canon not in _PLACEABLE:
            skipped.append(raw_name)
            continue
        if canon in placed:
            raise ValueError(
                f"two Oceananigans fields map to the same legoESM state field "
                f"{canon!r}: {placed_from[canon]!r} and {raw_name!r}."
            )
        arr = np.asarray(raw_arr, dtype=np.float64)
        if has_time_axis:
            arr = arr[time_index]
        placed[canon] = _to_latlon_lev_zrev(arr)
        placed_from[canon] = raw_name

    replacements: dict[str, jnp.ndarray] = {}
    info_shapes: dict[str, tuple[int, ...]] = {}

    if "u" in placed:
        target = base_state.u.data.shape  # (ny, nx+1[, nz])
        u_face = _u_cell_to_face(placed["u"], target[1], cyclic_x=cyclic_x)
        u_face = _conform_to(u_face, target, what="u")
        replacements["u"] = jnp.asarray(u_face)
        info_shapes["u"] = u_face.shape
    if "v" in placed:
        target = base_state.v.data.shape  # (ny+1, nx[, nz])
        v_face = _v_cell_to_face(placed["v"], target[0])
        v_face = _conform_to(v_face, target, what="v")
        replacements["v"] = jnp.asarray(v_face)
        info_shapes["v"] = v_face.shape
    for canon in ("T", "S", "eta", "w"):
        if canon in placed:
            target = getattr(base_state, canon).data.shape
            arr = _conform_to(placed[canon], target, what=canon)
            replacements[canon] = jnp.asarray(arr)
            info_shapes[canon] = arr.shape

    state = base_state._replace(
        **{
            canon: getattr(base_state, canon).replace(data=data)
            for canon, data in replacements.items()
        }
    )

    info = {
        "case_name": result.case_name,
        "time_index": time_index if has_time_axis else 0,
        "time_s": float(np.asarray(result.times_s).reshape(-1)[
            time_index if has_time_axis else 0]) if np.asarray(result.times_s).size else None,
        "cyclic_x": cyclic_x,
        "placed_fields": sorted(replacements),
        "left_at_base": sorted(f for f in _PLACEABLE if f not in replacements),
        "skipped_nonstate": sorted(skipped),
        "placed_shapes": info_shapes,
    }
    return OceananigansStateBridgeOutput(state=state, info=info)
