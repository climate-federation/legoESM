"""Raw fp64 binary I/O matching the Fortran driver's self-describing format.

Format (see ``cfgs/DINO/UNIT_HARNESS/unit_harness_io.F90`` docstring for the
authoritative Fortran-side definition -- this module is the Python mirror,
not a second definition):

    int64  ndim                (little-endian, native byte order on this
                                 machine -- x86_64 Linux is little-endian,
                                 matching gfortran's default STREAM I/O)
    ndim x int64  shape
    prod(shape) x float64  data, FORTRAN (column-major) order

``ndim == 0`` is the scalar-int convention (one more int64 payload, no shape
array) used for time-level indices (Kmm/Kbb/Krhs) and ``neos``.

Every reader asserts the file's shape against the caller's expected shape
before touching the payload -- mirrors the Fortran side's mandatory
assertion; this module must never silently reshape/truncate either.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "write_array",
    "read_array",
    "write_scalar_int",
    "read_scalar_int",
]


def write_array(path: str, arr: np.ndarray) -> None:
    """Write an ndarray (2-D/3-D/4-D) in the harness's self-describing format."""
    arr64 = np.asarray(arr, dtype=np.float64)
    shape = np.array(arr64.shape, dtype=np.int64)
    with open(path, "wb") as f:
        f.write(np.array([arr64.ndim], dtype=np.int64).tobytes())
        f.write(shape.tobytes())
        f.write(np.asfortranarray(arr64).tobytes(order="F"))


def read_array(path: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    """Read an ndarray, asserting its on-disk shape equals ``expected_shape``."""
    with open(path, "rb") as f:
        raw = f.read()
    ndim = int(np.frombuffer(raw, dtype=np.int64, count=1)[0])
    if ndim != len(expected_shape):
        raise ValueError(
            f"{path}: ndim mismatch: file has ndim={ndim}, "
            f"caller expects {len(expected_shape)} (shape={expected_shape})"
        )
    offset = 8
    shape = tuple(int(x) for x in np.frombuffer(raw, dtype=np.int64, count=ndim, offset=offset))
    if shape != tuple(expected_shape):
        raise ValueError(f"{path}: shape mismatch: file has {shape}, caller expects {expected_shape}")
    offset += 8 * ndim
    n = 1
    for s in shape:
        n *= s
    data = np.frombuffer(raw, dtype=np.float64, count=n, offset=offset)
    return data.reshape(shape, order="F").copy()


def write_scalar_int(path: str, value: int) -> None:
    with open(path, "wb") as f:
        f.write(np.array([0], dtype=np.int64).tobytes())
        f.write(np.array([int(value)], dtype=np.int64).tobytes())


def read_scalar_int(path: str) -> int:
    with open(path, "rb") as f:
        raw = f.read()
    ndim = int(np.frombuffer(raw, dtype=np.int64, count=1)[0])
    if ndim != 0:
        raise ValueError(f"{path}: expected scalar-int file (ndim=0), got ndim={ndim}")
    value = int(np.frombuffer(raw, dtype=np.int64, count=1, offset=8)[0])
    return value
