"""WeatherBench training: ERA5 states already regridded to the model grid, cached.

A WB sample is built from ERA5 analyses on the 0.25 degree grid (143 MB per
analysis on disk).  Training needs them only AFTER the ERA5 -> model-grid
conversion, so this cache stores, per analysis time, exactly what the sample
builder makes of it -- the model-grid carry, the near-surface target fields and
the prescribed-forcing planes -- as float32 (owner decision 2026-11-01), about
2 % of the raw volume.

Layout: one zarr store per configuration tag, time axis = the ERA5 indices the
configuration needs (sorted, then any added by a later extension appended), one array per pytree leaf (``c###`` carry, ``s_<key>``
near-surface targets, ``f###`` forcing), each chunked one time per chunk so
SLURM array tasks write disjoint regions.  ``done`` marks every time a task has
finished; the reader refuses a time that is not marked.  The pytree structure
is stored as text and compared with the reader's own template, and the training
loader compares its first sample, built from the raw store and rounded to
float32, with the cached one EXACTLY (``assert_cache_matches``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

STORE_FORMAT = 1


def cache_tag(parts) -> str:
    """16-hex tag of everything that changes what a cached record holds."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str)
                          .encode()).hexdigest()[:16]


def store_path(root, tag) -> Path:
    return Path(root) / f"wb_states_{tag}.zarr"


def _flat(tree):
    import jax
    leaves, treedef = jax.tree.flatten(tree)
    return [np.asarray(x) for x in leaves], str(treedef)


def record_arrays(carry, sfc, forcing) -> tuple[dict, dict]:
    """``({array name: float32 values}, {structure text})`` of one time."""
    c, c_def = _flat(carry)
    f, f_def = _flat(forcing)
    out = {f"c{k:03d}": v.astype(np.float32) for k, v in enumerate(c)}
    out.update({f"f{k:03d}": v.astype(np.float32) for k, v in enumerate(f)})
    out.update({f"s_{k}": np.asarray(v).astype(np.float32)
                for k, v in sorted(sfc.items())})
    return out, {"carry": c_def, "forcing": f_def, "sfc": sorted(sfc)}


def create_store(path, era5_indices, example, structure, meta) -> None:
    """Empty store sized for ``era5_indices``.  An existing store with the same
    structure and metadata is EXTENDED: indices it lacks are appended (unwritten,
    not done) and every time it holds is kept; a different structure is a hard
    error."""
    import zarr

    path = Path(path)
    idx = np.asarray(sorted(int(i) for i in era5_indices), dtype=np.int64)
    attrs = {"format": STORE_FORMAT, "structure": structure, "meta": meta}
    if path.exists():
        g = zarr.open_group(str(path), mode="r+")
        have = np.asarray(g["era5_index"])
        if dict(g.attrs) != json.loads(json.dumps(attrs)):
            raise RuntimeError(f"state cache {path} exists with a different "
                               "structure; delete it and rebuild.")
        new = idx[~np.isin(idx, have)]
        if len(new):
            n = len(have) + len(new)
            for name in ["done", *example]:
                g[name].resize((n, *g[name].shape[1:]))
            g["era5_index"].resize((n,))
            g["era5_index"][len(have):] = new
        return
    g = zarr.open_group(str(path), mode="w")
    g.attrs.update(attrs)
    g.create_array("era5_index", data=idx)
    g.create_array("done", data=np.zeros(len(idx), dtype=bool), chunks=(1,))
    for name, v in example.items():
        g.create_array(name, shape=(len(idx), *v.shape), dtype=np.float32,
                       chunks=(1, *v.shape), fill_value=np.nan)


def write_time(path, era5_index, arrays) -> None:
    """Write one time's record, then mark it done (last, so a killed task
    leaves an unmarked, refused time)."""
    import zarr

    g = zarr.open_group(str(path), mode="r+")
    pos = _position(np.asarray(g["era5_index"]), era5_index)
    for name, v in arrays.items():
        g[name][pos] = v
    g["done"][pos] = True


def _position(store_index, era5_index) -> int:
    # Not sorted once a store has been extended.
    hit = np.flatnonzero(store_index == int(era5_index))
    if not len(hit):
        raise KeyError(f"ERA5 index {era5_index} is not in this state cache")
    return int(hit[0])


class StateCache:
    """Read side.  ``template`` = ``(carry, sfc, forcing)`` built by the caller
    the normal way, giving the pytree structure and the dtypes to cast to."""

    def __init__(self, path, template):
        import zarr

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"state cache {self.path} does not exist; build it with "
                "scripts/data/build_wb_state_cache.py (SLURM).")
        self._g = zarr.open_group(str(self.path), mode="r")
        _, structure = record_arrays(*template)
        if dict(self._g.attrs).get("structure") != structure:
            raise RuntimeError(f"state cache {self.path} holds a different "
                               "sample structure than this run builds.")
        self._done = np.asarray(self._g["done"])
        self._idx = np.asarray(self._g["era5_index"])
        import jax
        self._c_def = jax.tree.structure(template[0])
        self._f_def = jax.tree.structure(template[2])
        self._c_dt = [np.asarray(x).dtype for x in jax.tree.leaves(template[0])]
        self._f_dt = [np.asarray(x).dtype for x in jax.tree.leaves(template[2])]
        self._sfc = sorted(template[1])

    def missing(self, era5_indices) -> list:
        have = set(self._idx[self._done].tolist())
        return sorted(int(i) for i in era5_indices if int(i) not in have)

    def read(self, era5_index):
        """``(carry, sfc, forcing)`` for one ERA5 index, host numpy leaves."""
        import jax

        pos = _position(self._idx, era5_index)
        if not self._done[pos]:
            raise RuntimeError(f"state cache {self.path}: ERA5 index "
                               f"{era5_index} was never written (task killed?).")
        g = self._g
        carry = jax.tree.unflatten(self._c_def, [
            np.asarray(g[f"c{k:03d}"][pos]).astype(dt)
            for k, dt in enumerate(self._c_dt)])
        forcing = jax.tree.unflatten(self._f_def, [
            np.asarray(g[f"f{k:03d}"][pos]).astype(dt)
            for k, dt in enumerate(self._f_dt)])
        sfc = {k: np.asarray(g[f"s_{k}"][pos]).astype(np.float64)
               for k in self._sfc}
        return carry, sfc, forcing


# A GPU rebuild is not bit-reproducible: measured 2026-11-04, one of 589824
# cloud-water values moved by 3.4e-21 (field max 5.4e-4) in 5 of 6 rebuilds.
# A stale cache moves fields by far more than GATE_EPS_MULT float32 machine
# epsilons of the field's own largest finite |value|.  (Voted 3-0: Claude,
# GLM, codex.)
GATE_EPS_MULT = 4


def assert_cache_matches(fresh, cached) -> None:
    """Hard error unless the cached record equals the freshly built one
    rounded to float32, leaf by leaf: NaN/inf exactly (same places, same
    values), every finite value to within ``GATE_EPS_MULT`` * float32 eps *
    that array's largest finite |value|."""
    a, _ = record_arrays(*fresh)
    b, _ = record_arrays(*cached)
    if sorted(a) != sorted(b):
        raise RuntimeError("state cache gate: record fields differ.")
    eps = float(np.finfo(np.float32).eps)
    for name in sorted(a):
        x, y = a[name], b[name]
        fin = np.isfinite(y)
        scale = float(np.max(np.abs(y[fin]), initial=0.0))
        if not (x.shape == y.shape
                and np.array_equal(np.isfinite(x), fin)
                and np.array_equal(x[~fin], y[~fin], equal_nan=True)
                and np.all(np.abs(x[fin].astype(np.float64)
                                  - y[fin].astype(np.float64))
                           <= GATE_EPS_MULT * eps * scale)):
            raise RuntimeError(
                f"state cache gate: {name} differs between the cache and a "
                "fresh build from ERA5; the cache is stale. Rebuild it.")
