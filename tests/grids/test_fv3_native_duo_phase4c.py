"""Phase 4c — divergence_corner_duo ROUTINE-TRANSLATION gate.

The DUO-GRID FV3 (Mouallem/Xi-Chen, Zenodo 8327578,
``atmos_cubed_sphere-symmetryclean``) calls ``divergence_corner_duo``
(nest interior + panel-seam zeroing/quartering) under
``flagstruct%duogrid`` — the branch legoESM's production solver runs.

SCOPE (codex p4c r1/r2): this gate certifies that the python
``fv3_native_duo_sw_core.divergence_corner_duo`` is a FAITHFUL
TRANSLATION of the authoritative Fortran — the port run on the fixture's
STORED input bytes reproduces the authoritative Fortran divg_d
BIT-for-BIT (uint64 word compare) over the 289 written B-nodes.  The
extracted Fortran routine is byte-identical to sw_core.F90:2345-2447
(SHA-256).  It does NOT yet certify the FULL DUO PIPELINE: the stored
ua/va come from the plain phase-4a ``c_sw``, whereas the real duo
pipeline feeds ua/va from ``d2a2c_vect``'s dg-initialized cross-face path
(different edge/corner values) — the next phase-4c brick.  The fixture
carries the input SHA-256 + a lineage note so the caveat travels with it.

Fixture ``divduo_oracle_c12.npz`` (re)generated reproducibly by
``scripts/cluster/divduo_oracle.sbatch``: gen_divduo_oracle.py builds the
inputs + a staging snapshot, fv3_divduo_driver.F90 runs the authoritative
Fortran on THOSE inputs, and ``--pack`` writes the committed npz from that
one generation (stored inputs + output + full input hash — no
regeneration drift).
"""

import os

import numpy as np
import pytest

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
RES, NG = 12, 3
LO = 1 - NG
M_B = RES + 2 * NG + 1


@pytest.fixture(scope="module")
def oracle():
    return np.load(os.path.join(FIX, "divduo_oracle_c12.npz"))


def _gs_from(oracle):
    return {k: np.asarray(oracle[k]) for k in
            ("rarea_c", "dxc", "dyc", "sin_sg", "cos_sg")}


def _active_mask():
    """Geometry-derived 289-node active mask: divergence_corner_duo's
    compute loops write Fortran i,j in isd+1..ied, jsd+1..jed (17x17)."""
    from legoesm.core.fv3_native_sw_core import Bounds
    bd = Bounds.single_tile(RES, NG)
    m = np.zeros((M_B, M_B), dtype=bool)
    i0, i1 = bd.isd + 1 - LO, bd.ied - LO            # np index range
    j0, j1 = bd.jsd + 1 - LO, bd.jed - LO
    m[i0:i1 + 1, j0:j1 + 1] = True
    return m


def test_divergence_corner_duo_bit_exact_on_stored_inputs(oracle):
    """Run the port on the fixture's STORED inputs; the result must equal
    the authoritative Fortran divg_d BIT-for-BIT (uint64 words) over the
    289 geometry-active B-nodes — so 'bit-exact on identical inputs' is
    genuinely enforced (codex p4c r2 P1/P2)."""
    from legoesm.core.fv3_native_duo_sw_core import divergence_corner_duo
    from legoesm.core.fv3_native_sw_core import Bounds

    bd = Bounds.single_tile(RES, NG)
    gs = _gs_from(oracle)
    got = np.asarray(
        divergence_corner_duo(
            np.asarray(oracle["u"]), np.asarray(oracle["v"]),
            np.asarray(oracle["ua"]), np.asarray(oracle["va"]),
            gs, bd, RES + 1, RES + 1, grid_type=0),
        dtype=np.float64)
    want = np.asarray(oracle["divg_d"], dtype=np.float64)
    assert got.shape == want.shape == (M_B, M_B)

    active = _active_mask()
    assert int(active.sum()) == 289
    # every active node is finite (not the 1e25 sentinel) on both sides
    assert np.isfinite(want[active]).all()
    # bit-for-bit: identical IEEE-754 words (rules out +0.0/-0.0 slips)
    gw = got[active].view(np.uint64)
    ww = want[active].view(np.uint64)
    n_bit_diff = int((gw != ww).sum())
    assert n_bit_diff == 0, f"{n_bit_diff} active nodes differ bitwise"


def test_fixture_input_provenance(oracle):
    """Fixture carries a FULL SHA-256 of the exact input bytes + the
    plain-vs-duo lineage caveat (codex p4c r2 P1)."""
    sha = str(oracle["input_sha256"])
    assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
    assert "d2a2c_vect" in str(oracle["input_lineage"])


def test_duo_seam_zeroing_and_all_quartering_branches(oracle):
    """Distinguish divergence_corner_duo from the nest variant: (a) the
    seam B-nodes (i/j == 1, npx) are ZEROED; (b) ALL FOUR next-to-seam
    branches (i==2, i==ie, j==2, j==je) are EXACTLY 1/4 of the raw
    divergence (exact scalar equality, not tolerance) — codex p4c r2 P2."""
    from legoesm.core.fv3_native_duo_sw_core import divergence_corner_duo
    from legoesm.core.fv3_native_sw_core import Bounds
    from legoesm.grids.fv3_native_gridstruct import fort

    npx = RES + 1
    ie = RES
    bd = Bounds.single_tile(RES, NG)
    gs = _gs_from(oracle)
    u, v = np.asarray(oracle["u"]), np.asarray(oracle["v"])
    ua, va = np.asarray(oracle["ua"]), np.asarray(oracle["va"])
    out = np.asarray(
        divergence_corner_duo(u, v, ua, va, gs, bd, npx, npx, grid_type=0),
        dtype=np.float64)

    def at(a, i, j):
        return a[i - LO, j - LO]

    for j in range(2, npx):
        assert at(out, 1, j) == 0.0
        assert at(out, npx, j) == 0.0
    for i in range(2, npx):
        assert at(out, i, 1) == 0.0
        assert at(out, i, npx) == 0.0

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

    def raw(i, j):
        return (vf(i, j - 1) - vf(i, j) + uf(i - 1, j) - uf(i, j)) * RC[i, j]

    # all four next-to-seam branches (i==is+1=2, i==ie, j==2, j==je),
    # each at an interior partner index; EXACT 1/4 (scalar equality)
    for (i, j) in [(2, 6), (ie, 6), (6, 2), (6, RES)]:
        assert at(out, i, j) == 0.25 * raw(i, j)
        assert at(out, i, j) != 0.0
