"""Write a copy of a model checkpoint with seeded white noise added to T.

Used to start a second realisation of an existing arm (a lagged/perturbed
member) without touching any other field: every other array is copied
byte-for-byte.  Same seed + same shape => identical perturbation, so two arms
perturbed with one seed differ only by their own state.
"""
import argparse

import numpy as np


def perturb(arrays: dict, sigma_k: float, seed: int) -> dict:
    out = dict(arrays)
    rng = np.random.default_rng(seed)
    out["T"] = arrays["T"] + sigma_k * rng.standard_normal(arrays["T"].shape)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--sigma-k", type=float, required=True, help="noise std dev [K]")
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    with np.load(a.src, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    np.savez(a.dst, **perturb(arrays, a.sigma_k, a.seed))
    print(f"wrote {a.dst}: T += N(0, {a.sigma_k} K), seed {a.seed}")


if __name__ == "__main__":
    main()
