"""Dependency-free MITgcm ``mdsio`` (``.meta`` / ``.data``) reader + writer.

MITgcm is a Fortran oracle: unlike Veros it cannot be stepped in-process, so the
MITgcm fidelity layer is an **offline-reference** harness — it loads diagnostics
that a separate MITgcm run dumped to disk (see :mod:`mitgcm_runner`). This module
is the lowest layer: parsing the ``mdsio`` binary format.

The ``mdsio`` format (one field per file pair):

* ``<prefix>.meta`` — small ASCII namelist::

       nDims = [   2 ];
       dimList = [
          62,    1,   62,
          62,    1,   62
       ];
       dataprec = [ 'float32' ];
       nrecords = [     1 ];
       timeStepNumber = [          10 ];
       fldList = { 'THETA   ' };          # optional (multi-field diag files)

  Each ``dimList`` row is ``[ globalSize, startIndex, endIndex ]`` for one
  dimension, ordered fastest-varying first (x, then y, then z).

* ``<prefix>.data`` — raw binary, **big-endian**, Fortran column-major order
  (x fastest), ``nrecords`` records of ``prod(globalSizes)`` elements each.

Scope. This reader handles **single global files** (``globalFiles=.TRUE.`` or
post-``gluemncbig`` / single-tile output) — exactly what the small
``verification/`` experiments we use as oracles produce. For multi-tile output
(``U.<iter>.001.001.data`` …) install ``MITgcmutils`` and use
:func:`legoesm.ocean.fidelity.mitgcm_runner` with ``prefer_mitgcmutils=True``,
which delegates tile-gluing to the canonical reader. This module deliberately
carries no MITgcm dependency so the bridge/tests run anywhere.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

# mdsio ``dataprec`` token -> big-endian numpy dtype. MITgcm writes
# big-endian ('b' machineformat) by default; the reader/writer pin that so a
# little-endian host does not silently byte-swap the reference.
_DATAPREC_TO_DTYPE: dict[str, str] = {
    "float32": ">f4",
    "float64": ">f8",
}
_DTYPE_TO_DATAPREC: dict[str, str] = {
    np.dtype(">f4").str: "float32",
    np.dtype("<f4").str: "float32",
    np.dtype(">f8").str: "float64",
    np.dtype("<f8").str: "float64",
}


class MdsMeta(NamedTuple):
    """Parsed contents of an mdsio ``.meta`` file.

    Attributes
    ----------
    dims : tuple[int, ...]
        Global sizes, fastest-varying first (x, y, z) — the on-disk Fortran
        order, exactly as ``dimList`` lists them.
    nrecords : int
        Number of stacked records in the ``.data`` file.
    dataprec : str
        ``"float32"`` or ``"float64"``.
    fld_list : tuple[str, ...] | None
        Field names for a multi-field diagnostics file, else ``None``.
    time_step : int | None
        ``timeStepNumber`` if present, else ``None``.
    """

    dims: tuple[int, ...]
    nrecords: int
    dataprec: str
    fld_list: tuple[str, ...] | None
    time_step: int | None


def _resolve_pair(prefix: str | Path, iteration: int | None) -> tuple[Path, Path]:
    """Return ``(meta_path, data_path)`` for ``prefix`` (+ optional iteration).

    MITgcm names iteration dumps ``<prefix>.<iter:010d>.{meta,data}``.
    """
    prefix = Path(prefix)
    if iteration is not None:
        stem = f"{prefix.name}.{int(iteration):010d}"
        base = prefix.with_name(stem)
    else:
        base = prefix
    return base.with_name(base.name + ".meta"), base.with_name(base.name + ".data")


def parse_meta(text: str) -> MdsMeta:
    """Parse the text of an mdsio ``.meta`` file into :class:`MdsMeta`."""
    # Collapse to a single line so the [...] / {...} blocks parse regardless of
    # the embedded newlines MITgcm writes inside dimList.
    flat = " ".join(text.split())
    blocks = dict(re.findall(r"(\w+)\s*=\s*(\[.*?\]|\{.*?\})\s*;", flat))

    if "nDims" not in blocks or "dimList" not in blocks or "dataprec" not in blocks:
        raise ValueError(
            "mdsio .meta missing required nDims/dimList/dataprec; got keys "
            f"{sorted(blocks)}"
        )

    n_dims = int(re.search(r"-?\d+", blocks["nDims"]).group())
    dim_ints = [int(x) for x in re.findall(r"-?\d+", blocks["dimList"])]
    if len(dim_ints) != 3 * n_dims:
        raise ValueError(
            f"dimList has {len(dim_ints)} ints, expected {3 * n_dims} "
            f"(3 per dim x nDims={n_dims})"
        )
    # Each dim is [globalSize, start, end]; take the global size (every 3rd).
    dims = tuple(dim_ints[0::3])

    prec_match = re.search(r"'(\w+)'", blocks["dataprec"])
    if prec_match is None or prec_match.group(1) not in _DATAPREC_TO_DTYPE:
        raise ValueError(f"unsupported/parse-failed dataprec block {blocks['dataprec']!r}")
    dataprec = prec_match.group(1)

    nrecords = int(re.search(r"-?\d+", blocks["nrecords"]).group()) if "nrecords" in blocks else 1

    fld_list: tuple[str, ...] | None = None
    if "fldList" in blocks:
        fld_list = tuple(s.strip() for s in re.findall(r"'([^']*)'", blocks["fldList"]))

    time_step = None
    if "timeStepNumber" in blocks:
        time_step = int(re.search(r"-?\d+", blocks["timeStepNumber"]).group())

    return MdsMeta(
        dims=dims,
        nrecords=nrecords,
        dataprec=dataprec,
        fld_list=fld_list,
        time_step=time_step,
    )


def read_mds(
    prefix: str | Path,
    *,
    iteration: int | None = None,
    squeeze: bool = True,
) -> tuple[np.ndarray, MdsMeta]:
    """Read one mdsio field pair into a numpy array + its :class:`MdsMeta`.

    The returned array has shape ``(nrecords, *reversed(dims))`` — i.e. the last
    axis is x (fastest on disk), matching natural ``arr[..., k, j, i]`` indexing.
    With ``squeeze=True`` (default) ONLY the leading single-record axis is
    dropped (``nrecords == 1`` -> ``arr[0]``). Unlike ``np.squeeze`` / ``rdmds``
    this deliberately PRESERVES a physical dimension of extent 1 (a single
    vertical level, a zonal slice ``nx==1`` …) so the returned rank is
    unambiguous for the state bridge.
    """
    meta_path, data_path = _resolve_pair(prefix, iteration)
    if not meta_path.exists() or not data_path.exists():
        raise FileNotFoundError(
            f"mdsio pair not found: expected {meta_path} and {data_path}"
        )
    meta = parse_meta(meta_path.read_text())

    dtype = np.dtype(_DATAPREC_TO_DTYPE[meta.dataprec])
    flat = np.fromfile(data_path, dtype=dtype)
    expected = meta.nrecords * int(np.prod(meta.dims))
    if flat.size != expected:
        raise ValueError(
            f"{data_path.name}: read {flat.size} elements, meta implies "
            f"{expected} (nrecords={meta.nrecords}, dims={meta.dims})"
        )
    # A multi-field diagnostics file stacks one record per field: nrecords must
    # be a whole multiple of the field count, else the meta/data are corrupt or
    # the file is something other than what fldList claims.
    if meta.fld_list and meta.nrecords % len(meta.fld_list) != 0:
        raise ValueError(
            f"{data_path.name}: nrecords={meta.nrecords} is not a multiple of "
            f"fldList length {len(meta.fld_list)} ({meta.fld_list})"
        )
    # On disk: x fastest. A C-order reshape to (nrecords, *reversed(dims))
    # places x on the last axis with the correct strides (no copy/transpose).
    arr = flat.reshape((meta.nrecords, *meta.dims[::-1]))
    # Return host-native byte order so downstream numpy/JAX ops don't choke on
    # a big-endian dtype, but keep float precision.
    arr = arr.astype(np.float64 if meta.dataprec == "float64" else np.float32)
    if squeeze and meta.nrecords == 1:
        # Drop ONLY the single record axis — never a physical (x/y/z) dimension
        # that legitimately has extent 1, which np.squeeze would collapse and
        # leave the bridge unable to tell which axis is which.
        arr = arr[0]
    return arr, meta


def write_mds(
    prefix: str | Path,
    array: np.ndarray,
    *,
    iteration: int | None = None,
    fld_list: tuple[str, ...] | None = None,
    time_step: int | None = None,
    dataprec: str | None = None,
) -> tuple[Path, Path]:
    """Write a numpy array as an mdsio ``.meta`` / ``.data`` pair.

    Inverse of :func:`read_mds` for the single-record, single-global-file case
    (the format the Slice-2 reference generator emits and the tests round-trip).
    ``array`` is interpreted in C order with the LAST axis fastest-varying (x),
    consistent with :func:`read_mds`'s output. The whole array is ONE record
    (``nrecords=1``); a MITgcm field is at most 3-D ``(nz, ny, nx)``. ``dataprec``
    defaults to the array's float width.

    Returns the written ``(meta_path, data_path)``.
    """
    array = np.asarray(array)
    if array.ndim > 3:
        raise ValueError(
            f"write_mds writes a single 2-D or 3-D field (nrecords=1); got "
            f"ndim={array.ndim}. Multi-record output is not supported — write "
            f"one file per record/field."
        )
    if fld_list is not None and len(fld_list) != 1:
        raise ValueError(
            f"write_mds writes one field per file; fld_list must name exactly "
            f"one field, got {fld_list!r}."
        )
    if dataprec is None:
        dataprec = _DTYPE_TO_DATAPREC.get(array.dtype.str)
        if dataprec is None:
            dataprec = "float64" if array.dtype == np.float64 else "float32"
    if dataprec not in _DATAPREC_TO_DTYPE:
        raise ValueError(f"unsupported dataprec {dataprec!r}")

    # dimList is fastest-first (x, y, z); our array is C-order x-last, so the
    # on-disk dim order is the reverse of array.shape.
    dims = tuple(int(n) for n in array.shape[::-1])
    n_dims = len(dims)

    dtype = np.dtype(_DATAPREC_TO_DTYPE[dataprec])
    meta_path, data_path = _resolve_pair(prefix, iteration)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    array.astype(dtype).tofile(data_path)

    dim_rows = "\n".join(f" {n:5d},    1, {n:5d}," for n in dims)
    lines = [
        f" nDims = [ {n_dims:3d} ];",
        " dimList = [",
        dim_rows,
        " ];",
        f" dataprec = [ '{dataprec}' ];",
        " nrecords = [     1 ];",
    ]
    if time_step is not None:
        lines.append(f" timeStepNumber = [ {int(time_step):10d} ];")
    if fld_list is not None:
        flds = " ".join(f"'{name}'" for name in fld_list)
        lines.append(f" fldList = {{ {flds} }};")
    meta_path.write_text("\n".join(lines) + "\n")
    return meta_path, data_path
