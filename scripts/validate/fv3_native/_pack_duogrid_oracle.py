import sys
import numpy as np

workdir, out_path = sys.argv[1], sys.argv[2]
arrays = {}
for res in (12, 24):
    # k2e tables: rec stag n i j loc coef(1:nord). Tile-symmetric (verified
    # at pack time); store tile 1 only, keyed by stagger.
    per_stag = {}
    nord = None
    for line in open(f"{workdir}/duogrid_k2e_c{res}.txt"):
        if line.startswith("#"):
            if "k2e_nord =" in line:
                nord = int(line.split()[-1])
            continue
        p = line.split()
        stag, tile, i, j, loc = p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4])
        coef = np.array([float(x) for x in p[5:]])
        per_stag.setdefault(stag, {}).setdefault(tile, {})[(i, j)] = (loc, coef)
    assert nord is not None
    for stag, tiles in per_stag.items():
        t1 = tiles[1]
        for t, recs in tiles.items():
            for key, (loc, coef) in recs.items():
                l1, c1 = t1[key]
                assert l1 == loc and np.abs(c1 - coef).max() < 1e-13, (
                    f"tile asymmetry {stag} {t} {key}")
        keys = np.array(sorted(t1), dtype=np.int64)
        locs = np.array([t1[tuple(k)][0] for k in keys], dtype=np.int64)
        coefs = np.stack([t1[tuple(k)][1] for k in keys])
        arrays[f"k2e_{stag}_ij_c{res}"] = keys
        arrays[f"k2e_{stag}_loc_c{res}"] = locs
        arrays[f"k2e_{stag}_coef_c{res}"] = coefs
    arrays[f"k2e_nord_c{res}"] = np.array(nord)

    # DEFINED remap-coord records (kind n pos layer value); tile symmetry
    # verified at pack time, tile 1 stored as keyed record arrays.
    recs = {}
    for line in open(f"{workdir}/duogrid_coords_c{res}.txt"):
        if line.startswith("#"):
            continue
        p = line.split()
        kind, tile, pos, layer, v = (p[0], int(p[1]), int(p[2]),
                                     int(p[3]), float(p[4]))
        recs.setdefault(kind, {}).setdefault(tile, {})[(pos, layer)] = v
    for kind, tiles in recs.items():
        t1 = tiles[1]
        for t, r in tiles.items():
            for key, v in r.items():
                assert abs(t1[key] - v) < 1e-13, (
                    f"coords tile asymmetry {kind} {t} {key}")
        keys = np.array(sorted(t1), dtype=np.int64)
        vals = np.array([t1[tuple(k)] for k in keys])
        arrays[f"coords_{kind}_key_c{res}"] = keys
        arrays[f"coords_{kind}_val_c{res}"] = vals
np.savez_compressed(out_path, **arrays)
print(f"wrote {out_path}")
