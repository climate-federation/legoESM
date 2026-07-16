#!/usr/bin/env python
"""Gen the duo d2a2c_vect -> divergence_corner_duo CHAIN oracle, or pack it.

Certifies divergence_corner_duo on the REAL duo ua/va (from d2a2c_vect's
dg-initialized branch), closing the scope caveat that its own oracle used
plain-c_sw ua/va.  Same canonical-writer + staging + no-regen-on-pack
discipline as the sibling gen scripts.

  gen_dchain_oracle.py <W>          build inputs + staging + smoke the chain.
  gen_dchain_oracle.py <W> --pack   write the committed fixture npz.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

RES, NG = 12, 3

# union of the d2a2c_vect (cosa/rsin + dxa/dya) and divergence_corner_duo
# (rarea_c/dxc/dyc) metric sets, plus sin_sg/cos_sg (shared) and u/v.
INPUT_FIELDS = (
    ("COSA_S", "cosa_s"), ("RSIN2", "rsin2"),
    ("COSA_U", "cosa_u"), ("RSIN_U", "rsin_u"),
    ("COSA_V", "cosa_v"), ("RSIN_V", "rsin_v"),
    ("DXA", "dxa"), ("DYA", "dya"),
    ("RAREA_C", "rarea_c"), ("DXC", "dxc"), ("DYC", "dyc"),
    ("SIN_SG", "sin_sg"), ("COS_SG", "cos_sg"),
    ("U", "u"), ("V", "v"),
)
FIXTURE_INPUTS = tuple(key for _name, key in INPUT_FIELDS)


def _dump_field(name: str, a: np.ndarray, lo: int) -> str:
    ni, nj = a.shape[:2]
    out = []
    for i in range(ni):
        for j in range(nj):
            if a.ndim == 2:
                out.append(f"{name} {i + lo} {j + lo} {a[i, j]:.17e}\n")
            else:
                for k in range(a.shape[2]):
                    out.append(f"{name} {i + lo} {j + lo} {k + 1} "
                               f"{a[i, j, k]:.17e}\n")
    return "".join(out)


def serialize_dchain_inputs(fields: dict, res: int = RES,
                            ng: int = NG) -> bytes:
    lo = 1 - ng
    s = f"# res {res}\n# ng {ng}\n"
    for name, key in INPUT_FIELDS:
        s += _dump_field(name, np.asarray(fields[key]), lo)
    return s.encode()


def _fix_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "tests", "grids", "fixtures", "dchain_oracle_c12.npz")


def _gen(work: str) -> None:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                    "..", "..", "..", "packages", "core"))
    from legoesm.core.fv3_native_duo_sw_core import (
        d2a2c_vect_duo,
        divergence_corner_duo,
    )
    from legoesm.core.fv3_native_sw_core import Bounds
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        FV3_RADIUS_M,
        analytic_swcore_state,
        build_fv3_native_gridstruct,
    )

    gs = build_fv3_native_gridstruct(RES, NG, radius=FV3_RADIUS_M,
                                     omega=FV3_OMEGA)
    st = analytic_swcore_state(gs)
    bd = Bounds.single_tile(RES, NG)
    fields = {
        "u": np.nan_to_num(np.asarray(st["u"], float), nan=0.0),
        "v": np.nan_to_num(np.asarray(st["v"], float), nan=0.0),
    }
    for k in ("cosa_s", "rsin2", "cosa_u", "rsin_u", "cosa_v", "rsin_v",
              "dxa", "dya", "rarea_c", "dxc", "dyc", "sin_sg", "cos_sg"):
        fields[k] = np.asarray(gs[k])

    with open(f"{work}/dchain_input.txt", "wb") as f:
        f.write(serialize_dchain_inputs(fields))
    np.savez_compressed(f"{work}/staging.npz", **fields)
    # smoke: the composed chain runs
    out = d2a2c_vect_duo(fields["u"], fields["v"], gs, bd, RES + 1, RES + 1,
                         dord4=True, grid_type=0)
    divergence_corner_duo(fields["u"], fields["v"],
                          np.nan_to_num(out["ua"], nan=0.0),
                          np.nan_to_num(out["va"], nan=0.0),
                          gs, bd, RES + 1, RES + 1, grid_type=0)
    print("gen: inputs + staging written")


def _pack(work: str) -> None:
    stag = dict(np.load(f"{work}/staging.npz"))
    canon = serialize_dchain_inputs(stag)
    on_disk = open(f"{work}/dchain_input.txt", "rb").read()
    if canon != on_disk:
        raise SystemExit("dchain_input.txt disagrees with staging.npz — "
                         "regenerate; refusing to pack a drifted fixture")
    inp_hash = hashlib.sha256(canon).hexdigest()

    lo = 1 - NG
    m_b = RES + 2 * NG + 1
    fd = np.full((m_b, m_b), np.nan)
    for line in open(f"{work}/dchain_output.txt"):
        pp = line.split()
        fd[int(pp[1]) - lo, int(pp[2]) - lo] = float(pp[3])

    np.savez_compressed(
        _fix_dir(), divg_d=fd,
        **{k: stag[k] for k in FIXTURE_INPUTS},
        res=RES, ng=NG, input_sha256=inp_hash,
        input_lineage="analytic-swcore-state D-winds; CHAIN duo d2a2c_vect "
        "(dg%is_initialized) ua/va -> divergence_corner_duo -> divg_d.  "
        "Certifies divergence_corner_duo on the REAL duo ua/va (its own "
        "oracle used plain-c_sw ua/va) — closes that scope end-to-end.")
    print("fixture packed; input_sha256", inp_hash)


if __name__ == "__main__":
    import sys

    work = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--pack":
        _pack(work)
    else:
        _gen(work)
