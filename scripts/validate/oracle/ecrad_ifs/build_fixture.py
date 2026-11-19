"""Build the ecRad 49r1 oracle fixture npz from one harness run (vote V38).

usage: python -I build_fixture.py RUN_DIR INPUT.nc OUT.npz [--all-columns]
RUN_DIR holds out.nc, out_radiative_properties.nc, rng.txt and dump/*.txt
(stage49.sbatch). By default only every 4th column (0-based 0,4,...,28) is
kept; --all-columns keeps all (used by the harness-side full match).
Keys: "in:<var>" input-file variables, "dump:<name>" stage dumps (Fortran
shape, column dimension sliced), "lb:dump:<name>" their Fortran lower bounds,
"props:<var>" radiative properties, "flux:<var>" outputs, "rng:values",
"rng:seeds", "rng:block_of_call", "rng:call_sizes" for the McICA stream,
"columns" the kept 0-based column indices.
"""
import glob
import os
import sys

import netCDF4
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpio  # noqa: E402

NCOL = 32


def nc_vars(path, cols):
    out = {}
    with netCDF4.Dataset(path) as f:
        f.set_auto_mask(False)
        for v in f.variables:
            a = np.asarray(f[v][:])
            if a.ndim and a.shape[0] == NCOL:
                a = a[cols]
            out[v] = a
    return out


def read_rng(path):
    seeds, sizes, block, vals = [], [], [], []
    with open(path) as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines):
        t = lines[i].split()
        if not t:
            i += 1
            continue
        if t[0] == "SEED":
            seeds.append(int(t[1]))
        elif t[0] == "U":
            n = int(t[1])
            sizes.append(n)
            block.append(len(seeds) - 1)
            vals.extend(float(x) for x in lines[i + 1:i + 1 + n])
            i += n
        else:
            raise ValueError(f"bad rng line {i}: {lines[i]!r}")
        i += 1
    return (np.array(vals), np.array(seeds, np.int64), np.array(block, np.int64),
            np.array(sizes, np.int64))


def main(run, inp, out, *flags):
    cols = np.arange(NCOL) if "--all-columns" in flags else np.arange(0, NCOL, 4)
    z = {"columns": cols}
    for k, v in nc_vars(inp, cols).items():
        z["in:" + k] = v
    for k, v in nc_vars(f"{run}/out_radiative_properties.nc", cols).items():
        z["props:" + k] = v
    for k, v in nc_vars(f"{run}/out.nc", cols).items():
        z["flux:" + k] = v
    for p in sorted(glob.glob(f"{run}/dump/*.txt")):
        for name, (a, lb) in dumpio.read_arrays(p).items():
            dims = [d for d, s in enumerate(np.shape(a)) if s == NCOL and lb[d] == 1]
            if len(dims) > 1:
                raise SystemExit(f"{name}: ambiguous column dimension {dims}")
            if dims:
                a = np.take(a, cols, axis=dims[0])
            z["dump:" + name] = a
            z["lb:dump:" + name] = np.asarray(lb, np.int64)
    vals, seeds, block, sizes = read_rng(f"{run}/rng.txt")
    # keep only the generator streams of kept columns (LW seed iseed+997,
    # radiation_mcica_lw.F90:189; SW seed iseed, radiation_mcica_sw.F90:194)
    iseed = z["in:iseed"].astype(np.int64)
    keep_seed = np.isin(seeds, np.concatenate([iseed, iseed + 997]))
    start = np.concatenate([[0], np.cumsum(sizes)])
    keep_call = keep_seed[block]
    vals = np.concatenate([vals[start[c]:start[c + 1]] for c in np.flatnonzero(keep_call)] or [np.zeros(0)])
    new_block = np.cumsum(keep_seed) - 1
    block, sizes, seeds = new_block[block[keep_call]], sizes[keep_call], seeds[keep_seed]
    z.update({"rng:values": vals, "rng:seeds": seeds, "rng:block_of_call": block, "rng:call_sizes": sizes})
    # props:planck_hl is the same array as the dump lw_planck_planck_hl: keep one
    if np.array_equal(np.transpose(z["dump:lw_planck_planck_hl"], (2, 1, 0)), z["props:planck_hl"]):
        del z["props:planck_hl"]
    else:
        print("props:planck_hl differs from the dump; both kept")
    np.savez_compressed(out, **z)
    print(f"{out}: {len(z)} arrays, columns {cols.tolist()}, {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
