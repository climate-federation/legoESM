"""NEMO F-point-pivot north fold (eORCA1, jperio=6) — our helpers vs NEMO.

The reference is an INDEPENDENT line-by-line transcription of the F-pivot
branch of ``lbc_nfd_generic.h90`` (NEMO 5.0.1, ``c_NFtype == 'F'``), written
from the Fortran with 1-based indices — it does not call any model helper.
Arrays are NEMO-shaped ``(ipj, ipi)`` with ``ihls`` = 1 cyclic halo column on
each side (the eORCA1.2 mesh layout).  Our stored fields are the SAME arrays
with the duplicated fold-halo row stripped, mapped through this model's
staggering:

* T (j, i)      = NEMO T(i, j)
* U face (j, k) = NEMO u(k-1, j)   (west face; face 0 and face n wrap)
* v (j+1, i)    = NEMO v(i, j)     (prepended south row)
* q (j+1, k)    = NEMO F(k-1, j)

The legacy (T-pivot) descriptor on the same layout is run through the same
comparison as a STRICT xfail: it must keep failing (non-vacuity), because its
U/F maps fall back to ``n-1-i``.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from legoesm.grids.latlon import FoldDescriptor
from legoesm.grids.operators_latlon_cgrid import (
    apply_north_fold,
    fold_ghost_source_T,
    fold_perm_f,
    fold_perm_u,
    fold_row,
    fpivot_fold_line,
    fpivot_ghost_rows,
    pad_ns_scalar,
    pad_ns_vector_v,
)
from legoesm.grids.tripole import create_synthetic_tripole_fpivot

jax.config.update("jax_enable_x64", True)

IPI = 14          # NEMO columns incl. the 2 cyclic halo columns (even)
IPJ = 7           # NEMO rows incl. the north fold-halo row
IHLS = 1


# ---------------------------------------------------------------------------
# Independent transcription of lbc_nfd_generic.h90, F-point pivot branch.
# p is (ipi, ipj) Fortran-ordered via a 1-based accessor; psgn scalar.
# ---------------------------------------------------------------------------
def nemo_lbc_nfd_fpivot(tab_ji_jj, cd_nat, psgn, ihls=IHLS):
    """Return a copy of ``tab`` (indexed [ji-1, jj-1]) after lbc_nfd 'F'."""
    p = np.array(tab_ji_jj, dtype=float, copy=True)
    ipi, ipj = p.shape
    ni0glo = ipi - 2 * ihls

    def g(i, j):
        return p[i - 1, j - 1]

    def s(i, j, val):
        p[i - 1, j - 1] = val

    if cd_nat in ("T", "W"):
        for jj in range(1, ihls + 1):
            ij1 = ipj + 1 - jj
            ij2 = ipj - 2 * ihls + jj
            for ji in range(1, ihls + 1):
                s(ji, ij1, psgn * g(2 * ihls + 1 - ji, ij2))
            for ji in range(1, ni0glo + 1):
                s(ihls + ji, ij1, psgn * g(ipi - ihls - ji + 1, ij2))
            for ji in range(1, ihls + 1):
                s(ipi - ihls + ji, ij1, psgn * g(ipi - ihls - ji + 1, ij2))
    elif cd_nat == "U":
        for jj in range(1, ihls + 1):
            ij1 = ipj + 1 - jj
            ij2 = ipj - 2 * ihls + jj
            for ji in range(1, ihls):
                s(ji, ij1, psgn * g(2 * ihls - ji, ij2))
            for ji in range(1, 2):
                ii1 = ihls + ji - 1
                s(ii1, ij1, psgn * g(ipi - ii1, ij2))
            for ji in range(1, ni0glo):
                s(ihls + ji, ij1, psgn * g(ipi - ihls - ji, ij2))
            for ji in range(1, 2):
                ii1 = ipi - ihls + ji - 1
                s(ii1, ij1, psgn * g(ii1, ij2))
            for ji in range(1, ihls + 1):
                s(ipi - ihls + ji, ij1, psgn * g(ipi - ihls - ji, ij2))
    elif cd_nat == "V":
        for jj in range(1, ihls + 1):
            ij1 = ipj - jj + 1
            ij2 = ipj - 2 * ihls + jj - 1
            for ji in range(1, ihls + 1):
                s(ji, ij1, psgn * g(2 * ihls + 1 - ji, ij2))
            for ji in range(1, ni0glo + 1):
                s(ihls + ji, ij1, psgn * g(ipi - ihls - ji + 1, ij2))
            for ji in range(1, ihls + 1):
                s(ipi - ihls + ji, ij1, psgn * g(ipi - ihls - ji + 1, ij2))
        ij1 = ipj - ihls
        for ji in range(1, ni0glo // 2 + 1):
            s(ipi // 2 + ji, ij1, psgn * g(ipi // 2 - ji + 1, ij1))
        for ji in range(1, ihls + 1):
            s(ji, ij1, psgn * g(2 * ihls + 1 - ji, ij1))
    elif cd_nat == "F":
        for jj in range(1, ihls + 1):
            ij1 = ipj - jj + 1
            ij2 = ipj - 2 * ihls + jj - 1
            for ji in range(1, ihls):
                s(ji, ij1, psgn * g(2 * ihls - ji, ij2))
            for ji in range(1, 2):
                ii1 = ihls + ji - 1
                s(ii1, ij1, psgn * g(ipi - ii1, ij2))
            for ji in range(1, ni0glo):
                s(ihls + ji, ij1, psgn * g(ipi - ihls - ji, ij2))
            for ji in range(1, 2):
                ii1 = ipi - ihls + ji - 1
                s(ii1, ij1, psgn * g(ii1, ij2))
            for ji in range(1, ihls + 1):
                s(ipi - ihls + ji, ij1, psgn * g(ipi - ihls - ji, ij2))
        ij1 = ipj - ihls
        for ji in range(1, ni0glo // 2):
            s(ipi // 2 + ji, ij1, psgn * g(ipi // 2 - ji, ij1))
        for ji in range(1, ihls):
            s(ji, ij1, psgn * g(2 * ihls - ji, ij1))
    else:
        raise ValueError(cd_nat)
    return p


def nemo_apply(nemo_jj_ji, cd_nat, psgn):
    """(ipj, ipi) row-major wrapper around the (ji, jj) transcription."""
    return nemo_lbc_nfd_fpivot(nemo_jj_ji.T, cd_nat, psgn).T


def ew_cyclic(a):
    """NEMO E-W cyclic halo (ihls=1): col 0 <- col n-2, col n-1 <- col 1."""
    a = a.copy()
    a[:, 0] = a[:, -2]
    a[:, -1] = a[:, 1]
    return a


# ---- our layouts --------------------------------------------------------
def to_west_face(nemo):          # (rows, n) -> (rows, n+1): face k = col k-1
    return np.concatenate([nemo[:, -1:], nemo], axis=1)


def from_west_face(ours):        # (rows, n+1) -> (rows, n)
    return ours[:, 1:]


def to_vrows(nemo):              # NEMO V/F rows 0..J -> our rows 1..J+1
    return np.concatenate([np.zeros_like(nemo[:1]), nemo], axis=0)


# ---- descriptors ---------------------------------------------------------
def _fpivot_fold():
    # NEMO-shaped arrays carry the two cyclic halo columns.
    return create_synthetic_tripole_fpivot(IPJ - 1, IPI, ew_halo=True).fold


def _legacy_fold():
    idx = jnp.arange(IPI, dtype=jnp.int32)
    p = IPI - 1 - idx
    return FoldDescriptor(is_active=True, fold_j=IPJ - 2, cap_j=IPJ - 2,
                          perm_T=p, perm_v=p, vector_sign_u=-1.0,
                          vector_sign_v=-1.0)


FOLDS = [
    pytest.param(_fpivot_fold, id="fpivot"),
    pytest.param(_legacy_fold, id="legacy",
                 marks=pytest.mark.xfail(strict=True, reason=(
                     "legacy T-pivot maps are wrong on the F-pivot layout"))),
]


def _rand(seed, shape):
    return np.random.default_rng(seed).standard_normal(shape)


FOLDS_T = [pytest.param(_fpivot_fold, id="fpivot"),
           pytest.param(_legacy_fold, id="legacy")]   # T map is shared


@pytest.mark.parametrize("cyclic", [False, True], ids=["raw", "ewcyclic"])
class TestGhostsVsNemo:
    """Every ghost / fold-line row, bitwise, vs the Fortran transcription.

    ``raw`` = random arrays, compared on the interior columns 1..ipi-2 (NEMO
    fills the two halo columns from specific interior columns; E-W lbc makes
    them consistent afterwards).  ``ewcyclic`` = E-W-consistent random arrays,
    compared on EVERY column.
    """

    @staticmethod
    def _cols(cyclic, n):
        return slice(0, n) if cyclic else slice(1, n - 1)

    @pytest.mark.parametrize("make_fold", FOLDS_T)
    @pytest.mark.parametrize("psgn", [1.0, -1.0])
    def test_T(self, make_fold, cyclic, psgn):
        fold = make_fold()
        nemo = _rand(1, (IPJ, IPI))
        if cyclic:
            nemo = ew_cyclic(nemo)
        ref = nemo_apply(nemo, "T", psgn)
        ours = jnp.asarray(nemo[:-1])                       # stripped
        ghost = fpivot_ghost_rows(ours, fold, point="T", sign=psgn)
        # legacy composition used by the operators (T ghost source + perm_T)
        ghost2 = fold_row(fold_ghost_source_T(ours, fold), fold.perm_T, psgn, IPI)
        c = self._cols(cyclic, IPI)
        np.testing.assert_array_equal(np.asarray(ghost)[0, c], ref[-1, c])
        np.testing.assert_array_equal(np.asarray(ghost2)[0, c], ref[-1, c])
        np.testing.assert_array_equal(ref[:-1], nemo[:-1])   # nothing else moved

    @pytest.mark.parametrize("make_fold", FOLDS)
    def test_U(self, make_fold, cyclic):
        fold = make_fold()
        nemo = _rand(2, (IPJ, IPI))
        if cyclic:
            nemo = ew_cyclic(nemo)
        ref = nemo_apply(nemo, "U", -1.0)
        ours = jnp.asarray(to_west_face(nemo[:-1]))         # (n_lat, n+1)
        ghost = fpivot_ghost_rows(ours, fold, point="U", sign=-1.0)
        # composition at the Hollingsworth KE / Fu_fold_row sites
        ghost2 = fold_row(fold_ghost_source_T(ours, fold), fold_perm_u(fold),
                          fold.vector_sign_u, IPI)
        c = self._cols(cyclic, IPI)
        for gh in (ghost, ghost2):
            np.testing.assert_array_equal(
                from_west_face(np.asarray(gh))[0, c], ref[-1, c])

    @pytest.mark.parametrize("make_fold", FOLDS)
    @pytest.mark.parametrize("psgn", [1.0, -1.0])
    def test_V(self, make_fold, cyclic, psgn):
        fold = make_fold()
        nemo = _rand(3, (IPJ, IPI))
        if cyclic:
            nemo = ew_cyclic(nemo)
        ref = nemo_apply(nemo, "V", psgn)
        ours = jnp.asarray(to_vrows(nemo[:-1]))             # (n_lat+1, n)
        if make_fold is _legacy_fold:
            # The legacy descriptor has no fold-line identity: its only V
            # formula (pad_ns_vector_v) writes a GHOST of the row below into
            # the fold-line slot.
            line = np.asarray(pad_ns_vector_v(ours[1:-1], _grid(fold)))[-1]
            ghost = line
        else:
            line = np.asarray(fpivot_fold_line(ours, fold, point="V", sign=psgn))[-1]
            ghost = np.asarray(fpivot_ghost_rows(ours, fold, point="V", sign=psgn))[0]
        c = self._cols(cyclic, IPI)
        np.testing.assert_array_equal(line[c], ref[-2, c])
        np.testing.assert_array_equal(ghost[c], ref[-1, c])

    @pytest.mark.parametrize("make_fold", FOLDS)
    @pytest.mark.parametrize("psgn", [1.0, -1.0])
    def test_F(self, make_fold, cyclic, psgn):
        fold = make_fold()
        nemo = _rand(4, (IPJ, IPI))
        if cyclic:
            nemo = ew_cyclic(nemo)
        ref = nemo_apply(nemo, "F", psgn)
        ours = jnp.asarray(to_vrows(to_west_face(nemo[:-1])))   # (n_lat+1, n+1)
        if make_fold is _legacy_fold:
            line = np.asarray(pad_ns_scalar(ours[1:-1], _grid(fold)))[-1]
            ghost = line
        else:
            line = np.asarray(fpivot_fold_line(ours, fold, point="F", sign=psgn))[-1]
            ghost = np.asarray(fpivot_ghost_rows(ours, fold, point="F", sign=psgn))[0]
            # closure column (vertex n == vertex 0) is kept consistent
            assert line[-1] == line[0] and ghost[-1] == ghost[0]
        c = self._cols(cyclic, IPI)
        np.testing.assert_array_equal(line[1:][c], ref[-2, c])
        np.testing.assert_array_equal(ghost[1:][c], ref[-1, c])


def _grid(fold):
    g = create_synthetic_tripole_fpivot(IPJ - 1, IPI, ew_halo=True)
    return g._replace(fold=fold)


def test_measured_maps_on_descriptor():
    """perm_T = perm_v = n-1-i, perm_u = perm_f = (n-i)%n (stage-1 probe)."""
    fold = _fpivot_fold()
    i = np.arange(IPI)
    pure = create_synthetic_tripole_fpivot(IPJ - 1, IPI).fold
    np.testing.assert_array_equal(np.asarray(pure.perm_T), IPI - 1 - i)
    np.testing.assert_array_equal(np.asarray(pure.perm_v), IPI - 1 - i)
    np.testing.assert_array_equal(np.asarray(fold_perm_u(pure)), (IPI - i) % IPI)
    np.testing.assert_array_equal(np.asarray(fold_perm_f(pure)), (IPI - i) % IPI)
    assert pure.fpivot and not pure.pivot_row_stored and not pure.ew_halo
    # halo layout: identical on interior columns, NEMO's sources on the halo
    pt, pu = np.asarray(fold.perm_T), np.asarray(fold_perm_u(fold))
    np.testing.assert_array_equal(pt[1:-1], (IPI - 1 - i)[1:-1])
    np.testing.assert_array_equal(pu[1:-1], ((IPI - i) % IPI)[1:-1])
    assert (pt[0], pt[-1], pu[0], pu[-1]) == (1, IPI - 2, IPI - 2, IPI - 1)


def test_tpivot_helpers_refuse_fpivot():
    """The T-pivot ghost formulas must not run on the F-pivot layout."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        pad_ns_vector_pair, pad_ns_vector_u,
    )
    g = _grid(_fpivot_fold())
    inner = jnp.zeros((IPJ - 2, IPI))
    for fn in (pad_ns_scalar, pad_ns_vector_v, pad_ns_vector_u):
        with pytest.raises(NotImplementedError, match="F-pivot"):
            fn(inner, g)
    with pytest.raises(NotImplementedError, match="F-pivot"):
        pad_ns_vector_pair(inner, inner, g)


def test_fold_vface_row_is_nemo_T_ghost():
    """fold_vface_row = the cell across the fold line = NEMO T ghost (psgn +1)."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import fold_vface_row
    g = _grid(_fpivot_fold())
    nemo = ew_cyclic(_rand(5, (IPJ, IPI)))
    ref = nemo_apply(nemo, "T", 1.0)
    np.testing.assert_array_equal(
        np.asarray(fold_vface_row(jnp.asarray(nemo[:-1]), g))[0], ref[-1])


def test_apply_north_fold_writes_row():
    g = _grid(_fpivot_fold())
    padded = jnp.zeros((IPJ, IPI))
    row = jnp.arange(IPI, dtype=float)[None]
    np.testing.assert_array_equal(np.asarray(apply_north_fold(padded, row, g))[-1],
                                  np.arange(IPI))


def test_hollingsworth_top_row_uses_nemo_u_ghost():
    """NEMO nkeg_HW at the top T row reads u(j+1) = the lbc'd U ghost."""
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        hollingsworth_kinetic_energy,
    )
    g = _grid(_fpivot_fold())
    nk = 2
    nemo_u = np.stack([ew_cyclic(_rand(10 + k, (IPJ, IPI))) for k in range(nk)], -1)
    nemo_v = np.stack([ew_cyclic(_rand(20 + k, (IPJ, IPI))) for k in range(nk)], -1)
    ref_u = np.stack([nemo_apply(nemo_u[..., k], "U", -1.0) for k in range(nk)], -1)
    ref_v = np.stack([nemo_apply(nemo_v[..., k], "V", -1.0) for k in range(nk)], -1)
    # our stored state = stripped, fold line already lbc'd (as after lbc_lnk)
    u = np.concatenate([ref_u[:-1, -1:], ref_u[:-1]], axis=1)
    v = np.concatenate([np.zeros_like(ref_v[:1]), ref_v[:-1]], axis=0)
    ke = np.asarray(hollingsworth_kinetic_energy(jnp.asarray(u), jnp.asarray(v), g))
    # independent NEMO nkeg_HW formula at the top interior T row jj = ipj-2
    # (0-based), columns 1..ipi-2, using the lbc'd arrays incl. the ghost row.
    jj = IPJ - 2
    U, V = ref_u, ref_v
    for ji in range(1, IPI - 1):
        zu = (8.0 * (U[jj, ji - 1] ** 2 + U[jj, ji] ** 2)
              + (U[jj - 1, ji - 1] + U[jj + 1, ji - 1]) ** 2
              + (U[jj - 1, ji] + U[jj + 1, ji]) ** 2)
        zv = (8.0 * (V[jj - 1, ji] ** 2 + V[jj, ji] ** 2)
              + (V[jj - 1, ji - 1] + V[jj - 1, ji + 1]) ** 2
              + (V[jj, ji - 1] + V[jj, ji + 1]) ** 2)
        np.testing.assert_allclose(ke[jj, ji], (zu + zv) / 48.0, rtol=1e-14,
                                   atol=0)


def test_real_mesh_fpivot_detection():
    """eORCA1.2 stripped by one row is detected as F-pivot; unstripped refuses."""
    import os
    mesh = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "data", "grids", "eORCA1.2_mesh_mask.nc")
    if not os.path.exists(mesh):
        pytest.skip("eORCA1.2 mesh not present")
    from legoesm.grids.tripole import create_tripole_grid
    g = create_tripole_grid(mesh, strip_north_rows=1, fold_pivot="F",
                            dtype=jnp.float64)
    assert g.n_lat == 331 and g.n_lon == 362 and g.fold.fpivot
    with pytest.raises(ValueError, match="not an F-pivot fold line"):
        create_tripole_grid(mesh, strip_north_rows=0, fold_pivot="F",
                            dtype=jnp.float64)


def test_real_mesh_fold_line_rotation_is_antisymmetric():
    """The fold-line face normal seen from column c is the opposite of the one
    seen from perm_v[c]: both rotation components flip sign (wet columns of
    the real mesh, halo columns excluded)."""
    import os
    mesh = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "data", "grids", "eORCA1.2_mesh_mask.nc")
    if not os.path.exists(mesh):
        pytest.skip("eORCA1.2 mesh not present")
    from legoesm.grids.tripole import create_tripole_grid
    g = create_tripole_grid(mesh, strip_north_rows=1, fold_pivot="F",
                            dtype=jnp.float64)
    leg = create_tripole_grid(mesh, strip_north_rows=1, dtype=jnp.float64)
    P = np.asarray(g.fold.perm_v)
    c, s = np.asarray(g.cos_alpha_v)[-1], np.asarray(g.sin_alpha_v)[-1]
    k = np.arange(1, g.n_lon - 1)
    k = k[(P[k] >= 1) & (P[k] <= g.n_lon - 2) & (P[k] != k)]
    np.testing.assert_allclose(c[k], -c[P[k]], atol=1e-12)
    np.testing.assert_allclose(s[k], -s[P[k]], atol=1e-12)
    np.testing.assert_allclose(c ** 2 + s ** 2, 1.0, atol=1e-12)
    # rows below the fold are the generic builder's, unchanged
    np.testing.assert_array_equal(np.asarray(g.cos_alpha_v)[:-1],
                                  np.asarray(leg.cos_alpha_v)[:-1])
    cl, sl = np.asarray(leg.cos_alpha_v)[-1], np.asarray(leg.sin_alpha_v)[-1]
    print("legacy top-row antisym defect", np.abs(cl[k] + cl[P[k]]).max(),
          "| new vs legacy max |d cos|", np.abs(c - cl)[k].max(),
          "median", np.median(np.abs(c - cl)[k]))


def test_transcription_reproduces_nemo_mesh_halo():
    """Validate the TRANSCRIPTION itself on data NEMO produced: applying it to
    the eORCA1.2 coordinates (psgn +1) must reproduce the mesh's own stored
    halo row and fold line.  Latitude everywhere; longitude away from the one
    stored placeholder (halo row glam = 73.0 exactly at column 360)."""
    import os
    import netCDF4 as nc
    mesh = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "data", "grids", "eORCA1.2_mesh_mask.nc")
    if not os.path.exists(mesh):
        pytest.skip("eORCA1.2 mesh not present")
    ds = nc.Dataset(mesh)
    for nat, (lo, la) in {"T": ("glamt", "gphit"), "U": ("glamu", "gphiu"),
                          "V": ("glamv", "gphiv"), "F": ("glamf", "gphif")}.items():
        for name in (lo, la):
            a = np.asarray(ds[name][0], dtype=float)
            ref = nemo_apply(a, nat, 1.0)
            rows = [-1, -2] if nat in ("V", "F") else [-1]
            for r in rows:
                d = np.abs(ref[r, 1:-1] - a[r, 1:-1])
                if name.startswith("glam"):
                    d = np.minimum(d % 360.0, 360.0 - d % 360.0)
                    d[359] = 0.0 if nat in ("T", "V") and r == -1 else d[359]  # column 360
                assert d.max() < 1e-6, (nat, name, r, d.max(), int(d.argmax()))
