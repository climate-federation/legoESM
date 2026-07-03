"""MITgcm snapshot -> legoESM lat-lon C-grid state bridge.

The MITgcm counterpart of :mod:`veros_state_bridge`. Because MITgcm is already an
Arakawa **C-grid** model, this bridge is markedly simpler than the Veros (B-grid)
one — there is no corner->face interpolation. The only conventions to reconcile:

* **No halo.** MITgcm ``mdsio`` global dumps contain the physical domain only
  (no ``OLx/OLy`` overlap), so nothing is stripped.
* **Vertical order.** MITgcm ``k=1`` is the surface with ``k`` increasing
  downward — the SAME top-down order legoESM uses — so there is NO z-reversal
  (the Veros bridge reverses because Veros stores ``k=0`` at the bottom).
* **Axis order.** :func:`mitgcm_io.read_mds` returns ``(nz, ny, nx)`` (x fastest
  on the last axis); legoESM fields are ``(lat=ny, lon=nx, lev=nz)`` — a single
  ``moveaxis(0, -1)``.
* **C-grid faces.** MITgcm ``U`` sits on the WEST face of its tracer cell and
  ``V`` on the SOUTH face — identical to legoESM ``u_face``/``v_face``. legoESM
  carries one EXTRA face per row/column (the east / north boundary face); for a
  closed basin that face is the wall (zero), for a zonally cyclic domain ``u``
  wraps. ``cyclic_x`` selects which.

Field-name translation uses the cross-oracle concept registry
(``oracle_to_canonical("mitgcm")``) rather than a private table, so ``U``/``uVel``
-> ``u``, ``Eta``/``ETAN`` -> ``eta``, ``Theta`` -> ``T`` etc. stay in one place.
Fields absent from the snapshot (e.g. ``T``/``S`` for the homogeneous barotropic
gyre, where ``tempStepping=.FALSE.``) are left at the ``base_state`` values.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from legoesm.ocean.fidelity.concept_registry import oracle_to_canonical
from legoesm.ocean.fidelity.mitgcm_runner import MitgcmResult
from legoesm.ocean.state import LatLonCGridOceanState

# Canonical state-field names this bridge knows how to place. (Other canonical
# names returned by the registry — A_h, rho_0 … — are parameters, not state.)
_PLACEABLE = frozenset({"u", "v", "w", "T", "S", "eta"})


class MitgcmStateBridgeOutput(NamedTuple):
    """Result of :func:`mitgcm_snapshot_to_legoesm_state`."""

    state: LatLonCGridOceanState
    info: dict  # provenance / shapes / which fields were placed vs left at base


def _to_latlon_lev(arr: np.ndarray) -> np.ndarray:
    """``(nz, ny, nx)`` -> ``(ny, nx, nz)``; ``(ny, nx)`` passes through.

    MITgcm 3-D fields come off disk with the vertical on the FIRST axis; legoESM
    wants it LAST. 2-D (single-level / surface) fields are already ``(ny, nx)``.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        return np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"expected a 2-D or 3-D MITgcm field, got ndim={arr.ndim}")


def _conform_to(arr: np.ndarray, target_shape: tuple[int, ...], *, what: str) -> np.ndarray:
    """Reconcile a placed field with the legoESM target field shape.

    Handles the single-level case: a 2-D ``(ny, nx)`` MITgcm field maps to a
    3-D ``(ny, nx, 1)`` legoESM field (and vice-versa). Any other mismatch is a
    real error (raised, never silently broadcast).
    """
    if arr.shape == target_shape:
        return arr
    # 2-D field -> (ny, nx, 1) target.
    if arr.ndim == 2 and len(target_shape) == 3 and target_shape[2] == 1 \
            and arr.shape == target_shape[:2]:
        return arr[:, :, None]
    # (ny, nx, 1) field -> 2-D target.
    if arr.ndim == 3 and arr.shape[2] == 1 and len(target_shape) == 2 \
            and arr.shape[:2] == target_shape:
        return arr[:, :, 0]
    raise ValueError(
        f"{what}: MITgcm field shape {arr.shape} cannot be conformed to "
        f"legoESM target shape {target_shape}"
    )


def _u_cell_to_face(u_cell: np.ndarray, *, cyclic_x: bool) -> np.ndarray:
    """``(ny, nx[, nz])`` west-face velocities -> legoESM ``(ny, nx+1[, nz])``.

    MITgcm ``U[:, i]`` is the west face of tracer cell ``i`` — the same as
    legoESM ``u_face[:, i]``. legoESM additionally carries the east face of the
    last column: append the wrap of column 0 (``cyclic_x``) or a zero wall.
    """
    if cyclic_x:
        extra = u_cell[:, :1, ...]
    else:
        extra = np.zeros_like(u_cell[:, :1, ...])
    return np.concatenate([u_cell, extra], axis=1)


def _v_cell_to_face(v_cell: np.ndarray) -> np.ndarray:
    """``(ny, nx[, nz])`` south-face velocities -> legoESM ``(ny+1, nx[, nz])``.

    MITgcm ``V[j, :]`` is the south face of tracer cell ``j`` = legoESM
    ``v_face[j, :]``; append a zero north-wall row (meridional boundaries are
    always closed for the rectilinear ocean configs we mirror).
    """
    zero = np.zeros_like(v_cell[:1, ...])
    return np.concatenate([v_cell, zero], axis=0)


def mitgcm_snapshot_to_legoesm_state(
    result: MitgcmResult,
    base_state: LatLonCGridOceanState,
    *,
    time_index: int = -1,
    cyclic_x: bool = False,
) -> MitgcmStateBridgeOutput:
    """Initialise a legoESM lat-lon C-grid state from a MITgcm snapshot.

    Parameters
    ----------
    result : MitgcmResult
        Output of :func:`legoesm.ocean.fidelity.mitgcm_runner.load_mitgcm_reference`.
    base_state : LatLonCGridOceanState
        Empty/rest legoESM state at the matching grid; provides grid / land-mask
        / face-mask templates and the values for any field the snapshot omits.
    time_index : int, default -1
        Which loaded iteration to extract when several were stacked (default
        last). Ignored for a single-iteration result.
    cyclic_x : bool, default False
        Whether the zonal boundary is periodic (wrap ``u``) or a closed wall
        (zero east face). The tutorial barotropic gyre is a closed basin.

    Returns
    -------
    MitgcmStateBridgeOutput
        ``state`` with ``u``/``v``/``T``/``S``/``eta`` overwritten where the
        snapshot provides them; ``info`` records which fields were placed.
    """
    name_map = oracle_to_canonical("mitgcm")
    has_time_axis = int(np.asarray(result.iters).size) > 1
    # Index into iters for provenance: ignore time_index for a single-iteration
    # result (the data path below does the same), so passing an out-of-range
    # time_index does not IndexError when there is no time axis.
    iter_index = time_index if has_time_axis else 0

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
                f"two MITgcm fields map to the same legoESM state field "
                f"{canon!r}: {placed_from[canon]!r} and {raw_name!r}. A snapshot "
                f"must carry only one alias per field."
            )
        arr = np.asarray(raw_arr, dtype=np.float64)
        if has_time_axis:
            arr = arr[time_index]
        placed[canon] = _to_latlon_lev(arr)
        placed_from[canon] = raw_name

    replacements: dict[str, jnp.ndarray] = {}
    info_shapes: dict[str, tuple[int, ...]] = {}

    if "u" in placed:
        u_face = _u_cell_to_face(placed["u"], cyclic_x=cyclic_x)
        u_face = _conform_to(u_face, base_state.u.data.shape, what="u")
        replacements["u"] = jnp.asarray(u_face)
        info_shapes["u"] = u_face.shape
    if "v" in placed:
        v_face = _v_cell_to_face(placed["v"])
        v_face = _conform_to(v_face, base_state.v.data.shape, what="v")
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
        "time_index": iter_index,
        "iteration": int(np.asarray(result.iters).reshape(-1)[iter_index]),
        "cyclic_x": cyclic_x,
        "placed_fields": sorted(replacements),
        "left_at_base": sorted(
            f for f in _PLACEABLE if f not in replacements
        ),
        "skipped_nonstate": sorted(skipped),
        "placed_shapes": info_shapes,
    }
    return MitgcmStateBridgeOutput(state=state, info=info)
