"""Collect the McICA cloud-generator stage dumps into one npz.

usage: python -I build_gen_fixture.py DUMP_DIR OUT.npz [--all-columns]
Keys per spectrum s in (lw, sw): "gen:<s>_od_scaling" (ncol_kept, ng, nlev),
"gen:<s>_total_cloud_cover" (ncol_kept,), "gen:<s>_present" (ncol_kept,) bool:
False where the oracle never called the generator for that column (SW night
columns); those rows are NaN. od_scaling is only meaningful where
total_cloud_cover >= cloud_fraction_threshold (the oracle leaves it unset
otherwise). "columns": kept 0-based columns (every 4th unless --all-columns).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpio  # noqa: E402

NCOL = 32


def main(d, out, *flags):
    cols = np.arange(NCOL) if "--all-columns" in flags else np.arange(0, NCOL, 4)
    z = {"columns": cols}
    for s, ng in (("lw", 140), ("sw", 112)):
        sc = np.full((len(cols), ng, 137), np.nan)
        tcc = np.full(len(cols), np.nan)
        present = np.zeros(len(cols), bool)
        for k, c in enumerate(cols):
            p = f"{d}/{s}_gen_od_scaling_c{c + 1}.txt"
            if not os.path.exists(p):
                continue
            present[k] = True
            sc[k] = dumpio.read_arrays(p)[f"{s}_gen_od_scaling_c{c + 1}"][0]
            tcc[k] = dumpio.read_arrays(f"{d}/{s}_gen_total_cloud_cover_c{c + 1}.txt")[
                f"{s}_gen_total_cloud_cover_c{c + 1}"][0]
        z.update({f"gen:{s}_od_scaling": sc, f"gen:{s}_total_cloud_cover": tcc, f"gen:{s}_present": present})
    np.savez_compressed(out, **z)
    print(out, {k: np.shape(v) for k, v in z.items()}, f"{os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
