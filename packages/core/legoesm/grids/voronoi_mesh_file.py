"""Saved global SCVT meshes: build once, load everywhere, verified by hash.

Why this exists
---------------
``create_voronoi_mesh`` regenerates the mesh from scratch (icosahedral
bisection, Lloyd relaxation, scipy ConvexHull).  The generated cells agree
across platforms to about 4e-5 chord, but the hull and the floating point
path through the Lloyd loop are not bitwise portable, so the cell ORDER can
differ: measured on level 4 (2,562 cells), 1,973 cells sat at other indices on
Snellius than on a Mac.  Any per-cell input built on one machine (soil IC,
masks, regridding weights) then lands on the wrong cells on the other.

The fix is to build each mesh once, write every field of the ``VoronoiMesh``
to an ``.npz`` and make runs load that file.  The file carries a SHA-256 of a
canonical serialisation of its contents, so a run can pin the exact mesh it
expects and refuse anything else.

File layout (``.npz``, no pickled objects)
------------------------------------------
* one entry per ``VoronoiMesh`` field that is not ``None``: arrays under their
  field name, scalars (``nCells`` ... ``radius``) as 0-d arrays (int64 or
  float64);
* ``_format``: the format tag (:data:`MESH_FILE_FORMAT`);
* ``_sha256``: the content hash (:func:`mesh_arrays_sha256`);
* ``_metadata``: a JSON string (resolution, lloyd_iterations, radius, omega,
  generator commit, platform, numpy/scipy/jax versions, the Python type of
  every scalar field, the creation time).  Metadata is NOT hashed: the hash
  names the mesh, not the machine that wrote it.

Canonical hash
--------------
SHA-256 over the format tag, then for each mesh entry in sorted key order:
the key, the dtype forced to little-endian (``numpy.dtype.str``), the shape,
the byte length, then the C-ordered little-endian bytes.  It depends only on
names, dtypes, shapes and values, never on zip metadata, compression or write
order.

Dtypes
------
:func:`save_voronoi_mesh` stores arrays in the dtype they have in memory.
``scripts/grids/save_mpas_mesh.py`` builds under the float64 storage policy,
so its files hold the full float64 geometry.  :func:`load_voronoi_mesh`
returns floats in the CURRENT precision policy's storage dtype, exactly as
``_build_mesh_from_generators`` does (it computes in float64 NumPy and casts),
and integers as int32.  A float32 file loaded under a float64 policy is
refused: the lost precision cannot be recovered.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from typing import Any

import numpy as np

from legoesm.grids.voronoi import VoronoiMesh

MESH_FILE_FORMAT = "legoesm-voronoi-mesh-v1"

# The dimension/scalar fields of VoronoiMesh (everything else is an array or
# an optional array that may be None).
_SCALAR_FIELDS = ("nCells", "nEdges", "nVertices", "maxEdges",
                  "vertexDegree", "radius")
_RESERVED = ("_format", "_sha256", "_metadata")


class MeshFileError(ValueError):
    """A saved mesh file is malformed, tampered with or not the one expected."""


# ----------------------------------------------------------------------------
# Canonical serialisation and hash
# ----------------------------------------------------------------------------

def _canonical(arr) -> np.ndarray:
    """``arr`` as a C-contiguous little-endian array (same values, same kind)."""
    a = np.asarray(arr)
    if a.dtype.kind not in "biuf":
        raise MeshFileError(
            f"mesh entries must be bool/int/float, got dtype {a.dtype}")
    return np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<"),
                                         copy=False))


def mesh_arrays_sha256(entries: dict[str, np.ndarray]) -> str:
    """SHA-256 (hex) of a canonical serialisation of ``entries``.

    ``entries`` maps mesh field names to arrays (scalars as 0-d arrays).
    Sorted key order; per entry the key, little-endian dtype string, shape and
    byte count are hashed before the raw C-order little-endian bytes, so no two
    different meshes can share a byte stream by shifting a boundary.
    """
    h = hashlib.sha256()
    h.update(MESH_FILE_FORMAT.encode("ascii") + b"\n")
    for key in sorted(entries):
        a = _canonical(entries[key])
        data = a.tobytes(order="C")
        shape = tuple(int(n) for n in a.shape)
        header = f"{key}|{a.dtype.str}|{shape}|{len(data)}\n"
        h.update(header.encode("ascii"))
        h.update(data)
    return h.hexdigest()


def _scalar_entry(name: str, value) -> tuple[np.ndarray, str]:
    """A scalar field as a 0-d array plus the name of its Python type."""
    if isinstance(value, (bool, np.bool_)):
        raise MeshFileError(f"scalar field {name} is a bool, not a number")
    if isinstance(value, (int, np.integer)):
        return np.asarray(int(value), dtype=np.int64), "int"
    if isinstance(value, (float, np.floating)):
        return np.asarray(float(value), dtype=np.float64), "float"
    raise MeshFileError(
        f"scalar field {name} has unsupported type {type(value).__name__}")


def _mesh_entries(mesh: VoronoiMesh):
    """Every non-None field of ``mesh`` as NumPy arrays, plus scalar types."""
    if not isinstance(mesh, VoronoiMesh):
        raise TypeError(f"expected a VoronoiMesh, got {type(mesh).__name__}")
    entries: dict[str, np.ndarray] = {}
    scalar_types: dict[str, str] = {}
    for name in VoronoiMesh._fields:
        value = getattr(mesh, name)
        if value is None:
            continue
        if name in _SCALAR_FIELDS:
            entries[name], scalar_types[name] = _scalar_entry(name, value)
        else:
            entries[name] = _canonical(np.asarray(value))
    return entries, scalar_types


def voronoi_mesh_sha256(mesh: VoronoiMesh) -> str:
    """Content hash of an in-memory mesh, in the dtypes it holds.

    Equals the hash stored by :func:`save_voronoi_mesh` for the same object.
    A mesh loaded under a narrower storage policy than its file (float64 file,
    float32 run) hashes differently from the file, by design.
    """
    return mesh_arrays_sha256(_mesh_entries(mesh)[0])


# ----------------------------------------------------------------------------
# Metadata helpers
# ----------------------------------------------------------------------------

def _generator_commit() -> str:
    """Git commit of the code that wrote the file ('+dirty' if modified)."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
            text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here, capture_output=True, text=True, timeout=10,
            check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _versions() -> dict[str, str]:
    out = {"numpy": np.__version__}
    try:
        import scipy
        out["scipy"] = scipy.__version__
    except Exception:
        out["scipy"] = "unknown"
    try:
        import jax
        out["jax"] = jax.__version__
    except Exception:
        out["jax"] = "unknown"
    return out


# ----------------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------------

def save_voronoi_mesh(
    mesh: VoronoiMesh,
    path,
    *,
    resolution: int | None = None,
    lloyd_iterations: int | None = None,
    omega: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Write every field of ``mesh`` to ``path`` (.npz) and return its SHA-256.

    ``resolution`` (subdivision level), ``lloyd_iterations`` and ``omega`` are
    recorded as metadata; the model driver pins resolution and
    lloyd_iterations, so pass them whenever the mesh came from
    ``create_voronoi_mesh``.  The write is atomic (temporary file, then
    ``os.replace``).
    """
    entries, scalar_types = _mesh_entries(mesh)
    digest = mesh_arrays_sha256(entries)
    meta: dict[str, Any] = {
        "format": MESH_FILE_FORMAT,
        "resolution": None if resolution is None else int(resolution),
        "lloyd_iterations": (None if lloyd_iterations is None
                             else int(lloyd_iterations)),
        "radius": float(mesh.radius),
        "omega": None if omega is None else float(omega),
        "nCells": int(mesh.nCells),
        "fields": sorted(entries),
        "scalar_types": scalar_types,
        "dtypes": {k: v.dtype.str for k, v in sorted(entries.items())},
        "generator_commit": _generator_commit(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "versions": _versions(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra_metadata:
        clash = set(extra_metadata) & set(meta)
        if clash:
            raise ValueError(f"extra_metadata may not override {sorted(clash)}")
        meta.update(extra_metadata)
    payload = dict(entries)
    payload["_format"] = np.asarray(MESH_FILE_FORMAT)
    payload["_sha256"] = np.asarray(digest)
    payload["_metadata"] = np.asarray(json.dumps(meta, sort_keys=True))

    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return digest


def read_voronoi_mesh_metadata(path) -> dict[str, Any]:
    """The metadata dict (plus the stored ``sha256``) of a mesh file.

    Reads only the small metadata entries; nothing is verified here.
    """
    with np.load(os.fspath(path), allow_pickle=False) as data:
        if "_metadata" not in data.files or "_sha256" not in data.files:
            raise MeshFileError(f"{path}: not a saved Voronoi mesh file")
        meta = json.loads(str(data["_metadata"][()]))
        meta["sha256"] = str(data["_sha256"][()])
    return meta


def _storage_float_dtype():
    """The float dtype ``_build_mesh_from_generators`` would produce now."""
    import jax
    import jax.numpy as jnp
    try:
        from legoesm.core.precision import get_policy
        return get_policy().storage
    except Exception:
        return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def load_voronoi_mesh(path, expected_sha256: str | None = None) -> VoronoiMesh:
    """Load a mesh written by :func:`save_voronoi_mesh`, verifying its hash.

    Refuses (``MeshFileError``) when the file's format tag is unknown, its key
    set does not match the ``VoronoiMesh`` schema, the hash recomputed from its
    arrays differs from the stored one, or ``expected_sha256`` is given and
    differs from it.  Returns the same object ``create_voronoi_mesh`` returns:
    a ``VoronoiMesh`` with Python ``int``/``float`` scalars, int32 JAX
    connectivity arrays and float arrays in the current storage dtype.
    """
    import jax.numpy as jnp

    path = os.fspath(path)
    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        missing_reserved = [k for k in _RESERVED if k not in files]
        if missing_reserved:
            raise MeshFileError(
                f"{path}: not a saved Voronoi mesh file (missing "
                f"{missing_reserved})")
        fmt = str(data["_format"][()])
        if fmt != MESH_FILE_FORMAT:
            raise MeshFileError(
                f"{path}: format {fmt!r}, expected {MESH_FILE_FORMAT!r}")
        stored = str(data["_sha256"][()]).lower()
        meta = json.loads(str(data["_metadata"][()]))
        names = files - set(_RESERVED)
        optional = set(VoronoiMesh._field_defaults)
        required = set(VoronoiMesh._fields) - optional
        unknown = names - set(VoronoiMesh._fields)
        absent = required - names
        if unknown or absent:
            raise MeshFileError(
                f"{path}: fields do not match VoronoiMesh (unknown "
                f"{sorted(unknown)}, missing {sorted(absent)})")
        entries = {k: np.array(data[k]) for k in names}

    actual = mesh_arrays_sha256(entries)
    if actual != stored:
        raise MeshFileError(
            f"{path}: content hash {actual} does not match the stored hash "
            f"{stored}; the file was altered or corrupted. Refusing to use it.")
    if expected_sha256 is not None:
        want = str(expected_sha256).strip().lower()
        if want != stored:
            raise MeshFileError(
                f"{path}: mesh hash {stored} differs from the expected "
                f"{want}. Refusing to use a different mesh.")

    fdtype = jnp.dtype(_storage_float_dtype())
    fields: dict[str, Any] = {}
    for name in VoronoiMesh._fields:
        if name not in entries:
            continue                      # optional field, stays None
        a = entries[name]
        if name in _SCALAR_FIELDS:
            # type from the hashed dtype (int64 / float64), never from the
            # unhashed metadata
            fields[name] = int(a[()]) if a.dtype.kind in "iu" else float(a[()])
        elif a.dtype.kind == "f":
            if a.dtype.itemsize < fdtype.itemsize:
                raise MeshFileError(
                    f"{path}: field {name} is stored as {a.dtype} but the "
                    f"run's storage dtype is {fdtype}; the precision cannot "
                    "be recovered. Rebuild the file with "
                    "scripts/grids/save_mpas_mesh.py (float64).")
            fields[name] = jnp.array(a, dtype=fdtype)
        elif a.dtype.kind in "iu":
            i32 = np.iinfo(np.int32)
            if a.size and (a.min() < i32.min or a.max() > i32.max):
                raise MeshFileError(
                    f"{path}: field {name} holds values outside int32")
            fields[name] = jnp.array(a, dtype=jnp.int32)
        else:
            raise MeshFileError(
                f"{path}: field {name} has unsupported dtype {a.dtype}")
    return VoronoiMesh(**fields)


def load_voronoi_mesh_for_run(
    path,
    *,
    resolution: int,
    expected_sha256: str,
    lloyd_iterations: int | None = None,
    radius: float | None = None,
    variable: bool = False,
) -> VoronoiMesh:
    """Load a pinned global mesh for a model run, or refuse.

    Refuses when ``expected_sha256`` is empty, when the file's recorded
    resolution (subdivision level) differs from ``resolution``, when its cell
    count is not ``10 * 4**resolution + 2``, when ``lloyd_iterations`` or
    ``radius`` are given and differ from the file's, and on any hash problem
    (see :func:`load_voronoi_mesh`).

    ``variable=True`` loads a variable-resolution (non-icosahedral) mesh: the
    file must be marked ``variable_mesh: true`` in its metadata (saved with
    ``extra_metadata={"variable_mesh": True, ...}``), its cell count is checked
    against the count recorded in that metadata instead of the icosahedral
    ``10 * 4**r + 2``, and ``resolution`` and ``lloyd_iterations`` are not
    compared (a variable mesh has no subdivision level).  A file marked
    variable is refused when ``variable`` is False, so a variable mesh is
    never taken for a uniform one.
    """
    path = os.fspath(path)
    if not expected_sha256 or not str(expected_sha256).strip():
        raise MeshFileError(
            f"mesh file {path} given without its SHA-256; set "
            "mpas_mesh_sha256 (printed by scripts/grids/save_mpas_mesh.py). "
            "A run never loads an unpinned mesh.")
    if not os.path.isfile(path):
        raise MeshFileError(f"mesh file {path} does not exist")
    meta = read_voronoi_mesh_metadata(path)
    file_variable = bool(meta.get("variable_mesh", False))
    if file_variable != bool(variable):
        raise MeshFileError(
            f"mesh file {path} is {'a variable' if file_variable else 'an icosahedral'}"
            f" mesh but the run asks for {'a variable' if variable else 'an icosahedral'}"
            " one (mpas_mesh_variable)")
    if variable:
        mesh = load_voronoi_mesh(path, expected_sha256=expected_sha256)
        if mesh.nCells != int(meta.get("nCells", -1)):
            raise MeshFileError(
                f"mesh file {path} has {mesh.nCells} cells, its metadata "
                f"records {meta.get('nCells')}")
        if radius is not None and float(mesh.radius) != float(radius):
            raise MeshFileError(
                f"mesh file {path} has radius {mesh.radius}, the run uses {radius}")
        return mesh
    file_res = meta.get("resolution")
    if file_res is None or int(file_res) != int(resolution):
        raise MeshFileError(
            f"mesh file {path} was built at resolution {file_res}, the run "
            f"asks for {resolution}")
    if (lloyd_iterations is not None
            and meta.get("lloyd_iterations") != int(lloyd_iterations)):
        raise MeshFileError(
            f"mesh file {path} has lloyd_iterations="
            f"{meta.get('lloyd_iterations')}, the run builds with "
            f"{lloyd_iterations}")
    mesh = load_voronoi_mesh(path, expected_sha256=expected_sha256)
    n_expected = 10 * 4 ** int(resolution) + 2
    if mesh.nCells != n_expected:
        raise MeshFileError(
            f"mesh file {path} has {mesh.nCells} cells, resolution "
            f"{resolution} needs {n_expected}")
    if radius is not None and float(mesh.radius) != float(radius):
        raise MeshFileError(
            f"mesh file {path} has radius {mesh.radius}, the run uses {radius}")
    return mesh
