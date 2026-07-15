#!/usr/bin/env python
"""Build the phase-4b d_sw oracle npz from the Fortran driver's text dump.

Reads ``dswcore_output.txt`` (Fortran-index records emitted by
``fv3_dswcore_oracle_driver``) and writes ``dswcore_oracle_c12.npz`` with
each field stored on ITS PYTHON-RETURN ARRAY's Fortran origin — so the
phase-4b test is a plain same-shape masked diff.  Slots the driver does
not dump stay NaN (the source-defined region == the finite mask).

Usage: build_dsw_fixture.py <output.txt> <oracle.npz> [res] [ng]
"""

from __future__ import annotations

import sys

import numpy as np


def build(txt_path: str, npz_path: str, res: int = 12, ng: int = 3) -> None:
    lo = 1 - ng
    m_a, m_b = res + 2 * ng, res + 2 * ng + 1
    # oracle record name -> (python key, python shape, i-origin, j-origin)
    spec = {
        "DELPC": ("delpc", (m_a, m_a), lo, lo),
        "UOUT": ("u", (m_a, m_b), lo, lo),
        "VCOUT": ("vc", (m_a, m_b), lo, lo),
        "VOUT": ("v", (m_b, m_a), lo, lo),
        "UCOUT": ("uc", (m_b, m_a), lo, lo),
        "UAOUT": ("ua", (m_a, m_a), lo, lo),
        "VAOUT": ("va", (m_a, m_a), lo, lo),
        "DIVGD": ("divg_d", (m_b, m_b), lo, lo),
        "CRX": ("crx_adv", (res + 1, m_a), 1, lo),
        "XFX": ("xfx_adv", (res + 1, m_a), 1, lo),
        "CX": ("cx", (res + 1, m_a), 1, lo),
        "CRY": ("cry_adv", (m_a, res + 1), lo, 1),
        "YFX": ("yfx_adv", (m_a, res + 1), lo, 1),
        "CY": ("cy", (m_a, res + 1), lo, 1),
        "XFLUX": ("xflux", (res + 1, res), 1, 1),
        "YFLUX": ("yflux", (res, res + 1), 1, 1),
        "HEAT": ("heat_source", (res, res), 1, 1),
        "DISS": ("diss_est", (res, res), 1, 1),
    }
    out = {pk: np.full(sh, np.nan) for (pk, sh, _, _) in spec.values()}
    origin = {nm: (sp[0], sp[2], sp[3]) for nm, sp in spec.items()}
    for line in open(txt_path):
        if line.startswith("#"):
            continue
        p = line.split()
        nm = p[0]
        if nm not in origin:
            continue
        pk, io, jo = origin[nm]
        i, j, val = int(p[1]), int(p[2]), float(p[3])
        a = out[pk]
        gi, gj = i - io, j - jo
        if 0 <= gi < a.shape[0] and 0 <= gj < a.shape[1]:
            a[gi, gj] = val
    np.savez_compressed(npz_path, **out, res=res, ng=ng, dt=225.0)
    print(f"wrote {npz_path} ({len(out)} fields)")


if __name__ == "__main__":
    txt, npz = sys.argv[1], sys.argv[2]
    r = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    build(txt, npz, r, n)
