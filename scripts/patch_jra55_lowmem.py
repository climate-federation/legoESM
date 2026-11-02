"""Low-memory regridding in legoesm/forcing/jra55_do.py::_regrid_and_align_variable.

Replaces the whole-year load (np.asarray(da.values) + float64 JAX copy) with a
loop over blocks of 240 source records, and stores the result as float32.
Peak memory per variable drops from ~5x a float64 year to ~1x a float32 year
(0.5 deg: ~3 GB); the finished cache is also half the size.

Usage:  python patch_jra55_lowmem.py src/legoesm/forcing/jra55_do.py
Edits only the block between the two marker lines; aborts if either is missing.
"""
import sys

path = sys.argv[1]
src = open(path).read()

start = "    src_data = np.asarray(da.values)  # shape (n_time_src, n_src_lat, n_src_lon)\n"
end = "    out[cache_index[keep_mask]] = regridded\n"

new = '''    time_dim = next(d for d in da.dims if d not in (lat_name, lon_name))
    da = da.transpose(time_dim, lat_name, lon_name)
    n_src_lat = int(da.sizes[lat_name])
    n_src_lon = int(da.sizes[lon_name])
    pad_lon = weights.src_shape[1] == n_src_lon + 1   # periodic wrap column, see build_jra55_cache
    if (n_src_lat, n_src_lon + int(pad_lon)) != tuple(weights.src_shape):
        raise ValueError(
            f"{var}: source shape {(n_src_lat, n_src_lon)} does not "
            f"match weights.src_shape {weights.src_shape}"
        )

    # Regrid in blocks of source records so only one block is in memory at a
    # time, and store float32 (forcing precision; halves memory and cache size).
    keep_idx = np.nonzero(keep_mask)[0]
    dst_idx = cache_index[keep_mask]
    out = np.full((n_records, n_dst_lat, n_dst_lon), np.nan, dtype=np.float32)
    block = 240
    for s in range(0, keep_idx.size, block):
        blk = np.asarray(da.isel({time_dim: keep_idx[s:s + block]}).values)
        if pad_lon:
            blk = np.concatenate([blk, blk[..., :1]], axis=-1)
        reg = apply_conservative_regrid(jnp.asarray(blk, dtype=jnp.float64), weights)
        out[dst_idx[s:s + block]] = np.asarray(reg, dtype=np.float32)
'''

if src.count(start) != 1 or src.count(end) != 1:
    sys.exit("marker lines not found exactly once; nothing changed")
i, j = src.index(start), src.index(end) + len(end)
if i >= j:
    sys.exit("markers out of order; nothing changed")
open(path, "w").write(src[:i] + new + src[j:])
print("patched", path)
