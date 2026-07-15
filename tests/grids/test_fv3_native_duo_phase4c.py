"""Phase 4c — divergence_corner_duo ROUTINE-TRANSLATION gate.

The DUO-GRID FV3 (Mouallem/Xi-Chen, Zenodo 8327578,
``atmos_cubed_sphere-symmetryclean``) calls ``divergence_corner_duo``
(nest interior + panel-seam zeroing/quartering) under
``flagstruct%duogrid`` — the branch legoESM's production solver runs.

SCOPE (codex p4c r1): this gate certifies that the python
``fv3_native_duo_sw_core.divergence_corner_duo`` is a FAITHFUL
TRANSLATION of the authoritative Fortran — bit-exact on IDENTICAL inputs
over the 289 written B-nodes.  The extracted Fortran routine is
byte-identical to sw_core.F90:2345-2447 (SHA-256 verified).  It does NOT
yet certify the FULL DUO PIPELINE: the ua/va here come from the plain
phase-4a ``c_sw``, whereas the real duo pipeline feeds ua/va from
``d2a2c_vect``'s dg-initialized cross-face path (different edge/corner
values).  Certifying that end-to-end needs the duo ``d2a2c_vect`` port —
the next phase-4c brick.  The fixture carries its input SHA + a lineage
note so this caveat travels with it.

Fixture ``divduo_oracle_c12.npz`` (re)generated reproducibly by
``scripts/cluster/divduo_oracle.sbatch`` (gen_divduo_oracle.py builds the
inputs + runs the port; fv3_divduo_driver.F90 runs the authoritative
Fortran; the ``--pack`` pass writes the committed npz from that output).
"""

import os

import numpy as np
import pytest

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="module")
def oracle():
    return np.load(os.path.join(FIX, "divduo_oracle_c12.npz"))


def _port_output():
    from legoesm.core.fv3_native_duo_sw_core import divergence_corner_duo
    from legoesm.core.fv3_native_sw_core import Bounds, c_sw
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        FV3_RADIUS_M,
        analytic_swcore_state,
        build_fv3_native_gridstruct,
    )

    res, ng = 12, 3
    gs = build_fv3_native_gridstruct(res, ng, radius=FV3_RADIUS_M,
                                     omega=FV3_OMEGA)
    st = analytic_swcore_state(gs)
    bd = Bounds.single_tile(res, ng)
    csw = c_sw(delp=st["delp"], pt=st["pt"], w=np.zeros_like(st["delp"]),
               u=st["u"], v=st["v"], gs=gs, bd=bd, npx=res + 1, npy=res + 1,
               dt2=112.5, nord=1, hydrostatic=True, dord4=True, grid_type=0)
    ua = np.nan_to_num(np.asarray(csw["ua"], float), nan=0.0)
    va = np.nan_to_num(np.asarray(csw["va"], float), nan=0.0)
    u = np.nan_to_num(np.asarray(st["u"], float), nan=0.0)
    v = np.nan_to_num(np.asarray(st["v"], float), nan=0.0)
    return divergence_corner_duo(u, v, ua, va, gs, bd, res + 1, res + 1,
                                 grid_type=0)


def test_divergence_corner_duo_matches_authoritative(oracle):
    """Bit-exact vs the authoritative Fortran over the 289 WRITTEN B-nodes.

    The divg_d array is init to 1e25 everywhere (361 slots); the compute
    loops write only i,j in isd+1..ied, jsd+1..jed (17x17 = 289).  The 72
    outer-ring slots keep the 1e25 sentinel on BOTH sides and would match
    trivially — certify only the active mask (codex p4c r1 P2)."""
    want = np.asarray(oracle["divg_d"], dtype=np.float64)
    got = np.asarray(_port_output(), dtype=np.float64)
    assert got.shape == want.shape
    active = np.isfinite(want) & (want != 1.0e25)
    assert int(active.sum()) == 289, int(active.sum())
    err = (np.abs(got[active] - want[active])
           / np.maximum(np.abs(want[active]), 1.0)).max()
    assert err == 0.0, f"divergence_corner_duo: max rel err {err:.3e}"


def test_fixture_input_lineage_documented(oracle):
    """The fixture carries its input provenance + the plain-vs-duo caveat
    so the scope limit travels with the data (codex p4c r1 P1)."""
    assert "input_sha256" in oracle.files
    assert "input_lineage" in oracle.files
    assert b"DUO" in bytes(str(oracle["input_lineage"]), "utf8") or \
        "DUO" in str(oracle["input_lineage"])


def test_duo_panel_edge_zeroing_and_quartering():
    """The duo-specific seam handling must fire, distinguishing
    divergence_corner_duo from the nest variant (codex p4c r1 P2):
    (a) the seam B-nodes (i/j == 1, npx) are ZEROED; (b) the next-to-seam
    B-nodes (i/j == 2, npx-1=ie) are QUARTERED — verified by comparing the
    duo output to the un-quartered nest-form divergence at those nodes."""
    from legoesm.core.fv3_native_duo_sw_core import divergence_corner_duo
    from legoesm.core.fv3_native_sw_core import Bounds, c_sw
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        FV3_RADIUS_M,
        analytic_swcore_state,
        build_fv3_native_gridstruct,
    )

    res, ng = 12, 3
    lo = 1 - ng
    npx = res + 1
    gs = build_fv3_native_gridstruct(res, ng, radius=FV3_RADIUS_M,
                                     omega=FV3_OMEGA)
    st = analytic_swcore_state(gs)
    bd = Bounds.single_tile(res, ng)
    csw = c_sw(delp=st["delp"], pt=st["pt"], w=np.zeros_like(st["delp"]),
               u=st["u"], v=st["v"], gs=gs, bd=bd, npx=npx, npy=npx,
               dt2=112.5, nord=1, hydrostatic=True, dord4=True, grid_type=0)
    ua = np.nan_to_num(np.asarray(csw["ua"], float), nan=0.0)
    va = np.nan_to_num(np.asarray(csw["va"], float), nan=0.0)
    u = np.nan_to_num(np.asarray(st["u"], float), nan=0.0)
    v = np.nan_to_num(np.asarray(st["v"], float), nan=0.0)
    out = np.asarray(
        divergence_corner_duo(u, v, ua, va, gs, bd, npx, npx, grid_type=0),
        dtype=np.float64)

    def at(a, i, j):
        return a[i - lo, j - lo]

    # (a) seam B-nodes zeroed
    for j in range(2, npx):
        assert at(out, 1, j) == 0.0
        assert at(out, npx, j) == 0.0
    for i in range(2, npx):
        assert at(out, i, 1) == 0.0
        assert at(out, i, npx) == 0.0

    # (b) next-to-seam quartering: recompute the raw (un-seam-adjusted)
    # divergence at a next-to-seam node and confirm the duo output is 1/4.
    from legoesm.grids.fv3_native_gridstruct import fort
    SIN = fort(gs["sin_sg"], bd.isd, bd.jsd)
    COS = fort(gs["cos_sg"], bd.isd, bd.jsd)
    DXC = fort(gs["dxc"], bd.isd, bd.jsd)
    DYC = fort(gs["dyc"], bd.isd, bd.jsd)
    RC = fort(gs["rarea_c"], bd.isd, bd.jsd)
    U = fort(u, bd.isd, bd.jsd)
    V = fort(v, bd.isd, bd.jsd)
    UA = fort(ua, bd.isd, bd.jsd)
    VA = fort(va, bd.isd, bd.jsd)

    def uf(i, j):
        return (U[i, j] - 0.25 * (VA[i, j - 1] + VA[i, j])
                * (COS[i, j - 1, 3] + COS[i, j, 1])) * DYC[i, j] * 0.5 \
            * (SIN[i, j - 1, 3] + SIN[i, j, 1])

    def vf(i, j):
        return (V[i, j] - 0.25 * (UA[i - 1, j] + UA[i, j])
                * (COS[i - 1, j, 2] + COS[i, j, 0])) * DXC[i, j] * 0.5 \
            * (SIN[i - 1, j, 2] + SIN[i, j, 0])

    i, j = 2, 6   # next-to-west-seam, interior j
    raw = (vf(i, j - 1) - vf(i, j) + uf(i - 1, j) - uf(i, j)) * RC[i, j]
    assert abs(at(out, i, j) - 0.25 * raw) <= 1e-12 * max(abs(raw), 1e-30)
    assert at(out, i, j) != 0.0
