"""ENVI/ESRI **BIL** raster reader for surface-data sources (pure NumPy, host-side).

Several primary surface datasets ship as a regular lat-lon raster in ESRI/ENVI
*band-interleaved-by-line* (BIL) format with a plain-text ``.hdr`` sidecar — most
relevantly FAO **HWSD v2.0**'s 30 arc-second global soil mapping-unit grid
(``HWSD2.bil``, 21600x43200 uint16).  A single-band BIL is just a row-major
binary array fully described by its header, so we read it with ``numpy.memmap``
and need **no GDAL / rasterio** dependency.

Coordinate convention: ESRI ``.hdr`` ``ULXMAP`` / ``ULYMAP`` give the map
coordinate of the **centre of the upper-left pixel** (not its corner); rows run
north→south.  :func:`cell_centers` returns the per-pixel centre lat/lon arrays
accordingly.

Host-side only — this module is part of the surface-data *producer* and is never
imported into the traced model.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np


class EnviBilHeader(NamedTuple):
    """Parsed ESRI/ENVI ``.hdr`` for a regular lat-lon BIL raster."""

    nrows: int
    ncols: int
    nbands: int
    dtype: np.dtype            # resolved numpy dtype incl. byte order
    nodata: float | None       # NODATA sentinel in raw units, or None
    # Geotransform (ESRI convention: map coord of the UL pixel *centre*).
    ulxmap: float              # longitude of UL pixel centre [deg]
    ulymap: float              # latitude of UL pixel centre [deg]
    xdim: float                # pixel size in x (lon) [deg]
    ydim: float                # pixel size in y (lat) [deg]
    layout: str                # "BIL" (only layout supported here)


def _resolve_dtype(nbits: int, pixeltype: str, byteorder: str) -> np.dtype:
    """Map ENVI ``NBITS``/``PIXELTYPE``/``BYTEORDER`` to a numpy dtype.

    ``BYTEORDER`` ``I`` = little-endian (Intel), ``M`` = big-endian (Motorola).
    ``PIXELTYPE`` defaults to unsigned integer when absent (ESRI BIL norm).
    """
    endian = "<" if byteorder.upper().startswith("I") else ">"
    pt = (pixeltype or "UNSIGNEDINT").upper()
    if pt in ("UNSIGNEDINT", "UINT", "U"):
        base = {8: "u1", 16: "u2", 32: "u4", 64: "u8"}
    elif pt in ("SIGNEDINT", "INT", "S"):
        base = {8: "i1", 16: "i2", 32: "i4", 64: "i8"}
    elif pt in ("FLOAT", "F"):
        base = {32: "f4", 64: "f8"}
    else:
        raise ValueError(f"Unsupported ENVI PIXELTYPE {pixeltype!r}.")
    if nbits not in base:
        raise ValueError(f"Unsupported NBITS={nbits} for PIXELTYPE {pt!r}.")
    # 1-byte types carry no endianness; numpy rejects the prefix there.
    code = base[nbits]
    return np.dtype(code if nbits == 8 else endian + code)


def read_envi_hdr(hdr_path: str) -> EnviBilHeader:
    """Parse an ESRI/ENVI ``.hdr`` file into an :class:`EnviBilHeader`.

    Keys are case-insensitive whitespace-separated ``KEY VALUE`` pairs.  Only the
    fields needed to memory-map a regular lat-lon BIL are interpreted; unknown
    keys are ignored.
    """
    fields: dict[str, str] = {}
    with open(hdr_path, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                fields[parts[0].upper()] = parts[1]

    layout = fields.get("LAYOUT", "BIL").upper()
    if layout != "BIL":
        raise ValueError(
            f"raster.read_envi_hdr supports LAYOUT=BIL only, got {layout!r}."
        )
    nbits = int(fields.get("NBITS", "0"))
    if nbits == 0:
        raise ValueError(f"{hdr_path}: missing/zero NBITS.")
    dtype = _resolve_dtype(
        nbits, fields.get("PIXELTYPE", ""), fields.get("BYTEORDER", "I")
    )
    nodata = float(fields["NODATA"]) if "NODATA" in fields else None

    return EnviBilHeader(
        nrows=int(fields["NROWS"]),
        ncols=int(fields["NCOLS"]),
        nbands=int(fields.get("NBANDS", "1")),
        dtype=dtype,
        nodata=nodata,
        ulxmap=float(fields["ULXMAP"]),
        ulymap=float(fields["ULYMAP"]),
        xdim=float(fields["XDIM"]),
        ydim=float(fields["YDIM"]),
        layout=layout,
    )


def _hdr_path_for(bil_path: str) -> str:
    """Return the sibling ``.hdr`` path for a ``.bil`` file."""
    root, _ = os.path.splitext(bil_path)
    return root + ".hdr"


def open_bil_memmap(
    bil_path: str, header: EnviBilHeader | None = None
) -> tuple[np.memmap, EnviBilHeader]:
    """Memory-map a BIL raster as a read-only NumPy array.

    Returns ``(array, header)``.  For a single-band raster ``array`` is shaped
    ``(nrows, ncols)``; for multi-band BIL it is ``(nrows, nbands, ncols)``
    (band-interleaved-by-line).  The header is read from the sibling ``.hdr``
    when not supplied.

    The mapping is lazy: no pixel data is read until indexed, so a 1.9 GB raster
    costs nothing to open.  Aggregation reads it in row-blocks (see
    :mod:`~legoesm.land.surface_data.aggregate`).
    """
    if header is None:
        header = read_envi_hdr(_hdr_path_for(bil_path))
    if header.nbands == 1:
        shape: tuple[int, ...] = (header.nrows, header.ncols)
    else:
        shape = (header.nrows, header.nbands, header.ncols)
    mm = np.memmap(bil_path, dtype=header.dtype, mode="r", shape=shape)
    return mm, header


def cell_centers(header: EnviBilHeader) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel centre coordinates ``(lat, lon)`` in degrees.

    ``lat`` is length ``nrows`` and runs north→south (descending); ``lon`` is
    length ``ncols`` ascending.  Both follow the ESRI centre-of-UL-pixel
    convention encoded in the header.
    """
    lon = header.ulxmap + np.arange(header.ncols, dtype=np.float64) * header.xdim
    lat = header.ulymap - np.arange(header.nrows, dtype=np.float64) * header.ydim
    return lat, lon


def write_envi_bil(bil_path: str, data: np.ndarray, header: EnviBilHeader) -> None:
    """Write a 2-D ``(nrows, ncols)`` array + ``.hdr`` in BIL form (single band).

    Provided so the reader is round-trip testable without shipping a large
    fixture raster; production sources only ever *read* BIL.
    """
    if data.ndim != 2:
        raise ValueError("write_envi_bil supports single-band (2-D) data only.")
    arr = np.ascontiguousarray(data.astype(header.dtype))
    arr.tofile(bil_path)
    byteorder = "M" if header.dtype.byteorder == ">" else "I"
    pix = "FLOAT" if header.dtype.kind == "f" else (
        "SIGNEDINT" if header.dtype.kind == "i" else "UNSIGNEDINT"
    )
    lines = [
        "BYTEORDER      " + byteorder,
        "LAYOUT         BIL",
        f"NROWS          {header.nrows}",
        f"NCOLS          {header.ncols}",
        "NBANDS         1",
        f"NBITS          {header.dtype.itemsize * 8}",
        f"PIXELTYPE      {pix}",
        f"ULXMAP         {header.ulxmap}",
        f"ULYMAP         {header.ulymap}",
        f"XDIM           {header.xdim}",
        f"YDIM           {header.ydim}",
    ]
    if header.nodata is not None:
        lines.append(f"NODATA         {header.nodata}")
    with open(_hdr_path_for(bil_path), "w") as fh:
        fh.write("\n".join(lines) + "\n")
