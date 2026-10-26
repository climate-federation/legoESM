"""Search entry cells for signed-zero/NaN ties at each O9 MIN/MAX site of the independent oracle.

The oracle is ``tests/ice/unit/test_si3_o9_zdf_dh_temp.py`` (a CPython transcription
of the record build whose ``MINSD``/``MAXSD`` carry their ppsrc site tag).  For each
candidate cell -- the test's nine cell kinds plus zero-heavy mutations (fields and
layers set to +0/-0, surfaces at the melting point) -- it records which sites see a
tie whose operands differ in bits, and, per site, whether swapping that one site's
emitted operand roles changes any output bit.  Writes JSON: the first cell per site
that ties (``first_tie``), the first whose swap changes the output (``found``) with
the changed leaves, and per-site tie counts.  Cells are stored as hex floats.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/ice/unit"))
import test_si3_o9_zdf_dh_temp as o9  # noqa: E402

ZERO_FIELDS = ("evap_ice", "qprec_ice", "sprecip", "qsr_ice", "qtr_ice_top", "qsb_ice_bot", "fhld", "qns_ice",
               "h_s", "hfx_err_dif")
SCALARS = ("h_i", "h_s", "evap_ice", "qprec_ice", "sprecip", "qsr_ice", "qtr_ice_top", "qsb_ice_bot", "fhld",
           "qns_ice", "dqns_ice", "hfx_err_dif", "hfx_dif", "sst", "frq_m")


def candidates(n_random: int, seed: int = 16):
    """Deterministic candidate cells: structured zero mutations, then ``n_random`` random ones."""
    for s in range(6):
        r = np.random.default_rng(s)
        for kind in o9.KINDS:
            base = o9._cell(r, kind)
            yield base
            for f in ZERO_FIELDS:
                for z in (0.0, -0.0):
                    c = copy.deepcopy(base)
                    c[f] = z
                    yield c
            for f1, f2 in itertools.combinations(ZERO_FIELDS, 2):
                c = copy.deepcopy(base)
                c[f1], c[f2] = -0.0, 0.0
                yield c
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        kind = o9.KINDS[int(rng.integers(len(o9.KINDS)))]
        c = o9._cell(np.random.default_rng(int(rng.integers(1 << 30))), kind)
        for f in SCALARS:
            if rng.random() < 0.3:
                c[f] = float(rng.choice([0.0, -0.0]))
        for f in ("e_s", "e_i", "sz_i"):
            if rng.random() < 0.2:
                c[f] = [float(rng.choice([0.0, -0.0])) if rng.random() < 0.5 else v for v in c[f]]
        for f in ("h_i", "h_s"):
            if rng.random() < 0.3:
                c[f] = float(rng.choice([0.0, -0.0]))
        if rng.random() < 0.3:
            c["t_su"] = o9.RT0
        if rng.random() < 0.2:
            c["t_s"] = [o9.RT0] * len(c["t_s"])
        yield c


def oracle_bits(cell, nml, swap=()):
    """(leaf -> bytes, site -> tie count) for one cell, ``swap`` sites with flipped roles; None on a trap."""
    o9.SWAP.clear()
    o9.SWAP.update(swap)
    o9.TIES.clear()
    try:
        rows = o9._oracle_rows([cell], nml)
    except (ZeroDivisionError, ValueError, OverflowError):
        return None, {}
    finally:
        o9.SWAP.clear()
    leaves = o9._oracle_leaves(rows)
    return {k: o9._bits(v).tobytes() for k, v in leaves.items()}, dict(o9.TIES)


def to_hex(cell):
    return {k: ([float.hex(float(x)) for x in v] if isinstance(v, list) else float.hex(float(v)))
            for k, v in cell.items()}


def from_hex(cell):
    return {k: ([float.fromhex(x) for x in v] if isinstance(v, list) else float.fromhex(v))
            for k, v in cell.items()}


def search(n_random: int, seed: int = 16) -> dict:
    nml = o9._deck()
    found, tied, first_tie, n, traps = {}, {}, {}, 0, 0
    for i, c in enumerate(candidates(n_random, seed)):
        n += 1
        ref, ties = oracle_bits(c, nml)
        if ref is None:
            traps += 1
            continue
        for site in ties:
            tied[site] = tied.get(site, 0) + 1
            first_tie.setdefault(site, dict(index=i, cell=to_hex(c)))
            if site in found:
                continue
            alt, _ = oracle_bits(c, nml, {site})
            if alt is not None and alt != ref:
                found[site] = dict(index=i, cell=to_hex(c), leaves=sorted(k for k in ref if ref[k] != alt[k]))
    return dict(candidates=n, traps=traps, seed=seed, tied=dict(sorted(tied.items())),
                first_tie=dict(sorted(first_tie.items())), found=dict(sorted(found.items())))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--random", type=int, default=300000)
    p.add_argument("--seed", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    rep = search(args.random, args.seed)
    args.output.write_text(json.dumps(rep, indent=1))
    print("TIE_SEARCH candidates", rep["candidates"], "traps", rep["traps"], "tied", sorted(rep["tied"]),
          "found", sorted(rep["found"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
