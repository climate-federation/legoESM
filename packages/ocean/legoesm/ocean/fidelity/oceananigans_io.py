"""NetCDF reader for Oceananigans (Julia) oracle output.

Counterpart to :mod:`legoesm.ocean.fidelity.mitgcm_io`, but Oceananigans writes
**NetCDF** (via ``NetCDFOutputWriter``) rather than the ``mdsio`` binary format,
so this is a thin :mod:`xarray` wrapper rather than a hand-rolled binary parser.

This reader returns the **raw** oracle arrays + coordinate vectors. It does NOT
reconcile any conventions: the Oceananigans staggering, the bottom-up vertical
order (``k=1`` is the BOTTOM, ``z`` increases upward), the ``(time, z, y, x)``
axis order, and the buoyancy/velocity naming are all handled by the *state
bridge* (``oceananigans_state_bridge.py``) — "would a user with a different goal
ever select this glue? no -> harness", per ``oracle_recipe_strategy.md``.

Oceananigans NetCDF conventions this reader merely *exposes*:

- **Dimensions** (a subset, depending on which fields were dumped): ``time``,
  ``xC``/``xF`` (cell-centre / face longitude), ``yC``/``yF`` (latitude),
  ``zC``/``zF`` (depth). A field carries the staggered coords of its location:
  ``u`` is ``(Face, Center, Center)``, ``v`` is ``(Center, Face, Center)``,
  ``w`` is ``(Center, Center, Face)``, a tracer (``b``/``T``/``S``) is
  ``(Center, Center, Center)``, and the free surface ``η`` is 2-D
  ``(Center, Center)``.
- **Coordinate variables**: ``xC, xF, yC, yF, zC, zF`` (1-D node positions).

The reader is dependency-light: :mod:`xarray` is imported lazily inside the
function (the legoESM deferred-heavy-import convention), so importing this module
never pulls NetCDF stacks into unrelated code paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class OceananigansNetCDF(NamedTuple):
    """Raw contents of an Oceananigans NetCDF output file.

    Attributes
    ----------
    variables : dict[str, np.ndarray]
        Data variables keyed by their Oceananigans name (``u``, ``v``, ``w``,
        ``b``/``T``/``S``, ``η``/``eta`` ...), as numpy arrays with their
        on-disk axis order preserved (typically ``(time, z, y, x)`` for 3-D
        fields, ``(time, y, x)`` for the 2-D free surface).
    coords : dict[str, np.ndarray]
        1-D coordinate vectors keyed by name (``xC, xF, yC, yF, zC, zF``) — the
        ones present in the file.
    times : np.ndarray
        The ``time`` coordinate values (seconds), shape ``(n_time,)``; an empty
        array if the file has no time dimension.
    sizes : dict[str, int]
        Length of every dimension found (``time, xC, yC, zC, ...``).
    attrs : dict
        Global file attributes (provenance: Oceananigans version, etc.).
    """

    variables: dict
    coords: dict
    times: np.ndarray
    sizes: dict
    attrs: dict


# Oceananigans names its staggered coordinate vectors by location code, and the
# scheme has changed across versions:
#   - older: ``xC/xF, yC/yF, zC/zF``
#   - v0.110.x (LatitudeLongitudeGrid): ``λ_caa/λ_faa`` (lon Center/Face),
#     ``φ_aca/φ_afa`` (lat Center/Face), ``z_aac/z_aaf`` (depth Center/Face).
# Rather than hard-code a list (which silently drops coords on a version bump),
# the reader captures EVERY 1-D coordinate/variable generically (see below).


def read_oceananigans_netcdf(
    path: str | Path,
    *,
    variables: tuple[str, ...] | None = None,
) -> OceananigansNetCDF:
    """Read an Oceananigans NetCDF output file into raw numpy arrays + coords.

    Parameters
    ----------
    path
        Path to the ``.nc`` file produced by an Oceananigans
        ``NetCDFOutputWriter``.
    variables
        Optional whitelist of data-variable names to load. ``None`` loads every
        non-coordinate data variable. An explicitly requested name that is
        absent from the file raises ``KeyError`` (no silent drop) — the dispatch
        rule, mirrored from the runner's case guard.

    Returns
    -------
    OceananigansNetCDF
        Raw arrays (on-disk axis order preserved), coordinate vectors, time
        values, dimension sizes, and global attributes.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If a name in *variables* is not a data variable in the file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Oceananigans NetCDF file not found: {p}")

    import xarray as xr  # deferred heavy import

    with xr.open_dataset(p, decode_times=False) as ds:
        available = tuple(ds.data_vars)
        if variables is None:
            selected = available
        else:
            missing = [v for v in variables if v not in ds.data_vars]
            if missing:
                raise KeyError(
                    f"Variable(s) {missing} not in {p.name}; "
                    f"available data variables: {available}"
                )
            selected = tuple(variables)

        var_arrays = {
            name: np.asarray(ds[name].values) for name in selected
        }
        # Capture every 1-D coordinate vector generically (version-agnostic):
        # both xarray coords and any 1-D data variable whose name matches its
        # single dimension (Oceananigans writes coord vectors as such).
        coords: dict = {}
        for name in ds.coords:
            arr = np.asarray(ds[name].values)
            if arr.ndim == 1:
                coords[str(name)] = arr
        for name in ds.variables:
            v = ds[name]
            if (str(name) not in coords and v.ndim == 1
                    and v.dims[0] == name):
                coords[str(name)] = np.asarray(v.values)
        times = (
            np.asarray(ds["time"].values)
            if "time" in ds.variables or "time" in ds.coords
            else np.asarray([], dtype=float)
        )
        sizes = {str(k): int(v) for k, v in ds.sizes.items()}
        attrs = dict(ds.attrs)

    return OceananigansNetCDF(
        variables=var_arrays,
        coords=coords,
        times=times,
        sizes=sizes,
        attrs=attrs,
    )
