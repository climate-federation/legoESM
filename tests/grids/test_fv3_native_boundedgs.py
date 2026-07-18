"""Gate: python bounded-conventions gridstruct == the certified
fv3_boundedgs oracle fixture (verbatim symmetryclean grid_utils_init +
fv_grid_tools bounded arms on the extended lattice), plus the direct
C48 cross-check against the numbers PRINTED by the Zenodo duo run's own
fms.out (da_max/da_min & friends) — the strongest oracle tie available:
python vs the actual paper run, no re-derivation in between.

Fixture: tests/grids/fixtures/fv3_boundedgs_oracle.npz
   (gen_boundedgs_oracle.py --n 12 --ng 3 --halo-mode extended)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "fv3_boundedgs_oracle.npz"

# Zenodo C48.sw.case2.alpha0.duo.hord6/rundir/fms.out prints
ZENODO_C48_DA_RATIO = 2.26548304435260
ZENODO_C48_DA_MAX_C = 53362939642.3731
ZENODO_C48_DA_MIN_C = 23543093086.1030


@pytest.fixture(scope="module")
def oracle():
    z = np.load(FIXTURE)
    assert str(z["halo_mode"]) == "extended"
    return z


@pytest.fixture(scope="module")
def gs(oracle):
    from legoesm.grids.fv3_native_gridstruct import (
        build_fv3_native_gridstruct_bounded,
    )

    n = int(oracle["n"])
    ng = int(oracle["ng"])
    return build_fv3_native_gridstruct_bounded(n, ng,
                                               tile=int(oracle["tile"]))


def _compare(records, arr, ilo, jlo, *, rtol, atol=1e-12, skip=None,
             name=""):
    """records = (N, 3) [i_fort, j_fort, val]; arr indexed [i-ilo, j-jlo].

    ``atol`` floors the check for exact-zero slots (face symmetries
    produce signed zeros whose relative error is meaningless)."""
    count = 0
    for i_f, j_f, val in records:
        i_f, j_f = int(i_f), int(j_f)
        if skip is not None and skip(i_f, j_f):
            continue
        got = arr[i_f - ilo, j_f - jlo]
        err = abs(got - val)
        rel = err / max(abs(val), abs(got), 1e-30)
        count += 1
        assert err <= atol or rel <= rtol, (
            f"{name}[{i_f},{j_f}]: oracle {val!r} vs python {got!r} "
            f"(rel {rel:.3e} > {rtol:.1e})")
    assert count > 0, f"{name}: no records compared"


# (record, gs key, axis kinds, dtype tier).  Records carry ABSOLUTE
# Fortran indices, and every python array spans the full lattice with
# row 0 == isd — so the array lo is always isd (axis kinds kept for
# documentation of the staggering only).
_F64 = 1e-12   # f64 dumps: formula-identical, libm-ulp headroom
_F32 = 3e-6    # oracle gridstruct stores these fields in real(4)-class
#                declarations promoted to r8 by -fdefault-real-8; keep
#                the loose tier for the div/mult-chain fields

FIELDS = [
    ("dx", "dx", ("a", "b"), _F64),
    ("dy", "dy", ("b", "a"), _F64),
    ("dxa", "dxa", ("a", "a"), _F64),
    ("dya", "dya", ("a", "a"), _F64),
    ("dxc", "dxc", ("b", "a"), _F64),
    ("dyc", "dyc", ("a", "b"), _F64),
    ("area", "area", ("a", "a"), _F64),
    ("arc", "area_c", ("b", "b"), _F64),
    ("sina", "sina", ("b", "b"), _F64),
    ("cosa", "cosa", ("b", "b"), _F64),
    ("rsna", "rsina", ("b", "b"), _F32),
    ("rsn2", "rsin2", ("a", "a"), _F32),
    ("rsnu", "rsin_u", ("b", "a"), _F32),
    ("rsnv", "rsin_v", ("a", "b"), _F32),
    ("csau", "cosa_u", ("b", "a"), _F32),
    ("csav", "cosa_v", ("a", "b"), _F32),
    ("csas", "cosa_s", ("a", "a"), _F32),
    ("snau", "sina_u", ("b", "a"), _F32),
    ("snav", "sina_v", ("a", "b"), _F32),
    ("dvgu", "divg_u", ("a", "b"), _F32),
    ("dvgv", "divg_v", ("b", "a"), _F32),
    ("dl6u", "del6_u", ("a", "b"), _F32),
    ("dl6v", "del6_v", ("b", "a"), _F32),
    ("agx", "agrid_lon", ("a", "a"), _F64),
    ("agy", "agrid_lat", ("a", "a"), _F64),
    ("rdx", "rdx", ("a", "b"), _F32),
    ("rdy", "rdy", ("b", "a"), _F32),
    ("rdxc", "rdxc", ("b", "a"), _F32),
    ("rdyc", "rdyc", ("a", "b"), _F32),
    ("rara", "rarea", ("a", "a"), _F32),
    ("rdxa", "rdxa", ("a", "a"), _F32),
    ("rdya", "rdya", ("a", "a"), _F32),
    ("rarc", "rarea_c", ("b", "b"), _F32),
]


@pytest.mark.parametrize("rec,key,axes,rtol",
                         FIELDS, ids=[f[0] for f in FIELDS])
def test_field_matches_oracle(oracle, gs, rec, key, axes, rtol):
    n, ng = int(oracle["n"]), int(oracle["ng"])
    isd = 1 - ng
    lo = {"a": isd, "b": isd}
    skip = None
    _compare(oracle[rec], gs[key], lo[axes[0]], lo[axes[1]],
             rtol=rtol, skip=skip, name=rec)


def test_sg_slots_match_oracle(oracle, gs):
    n, ng = int(oracle["n"]), int(oracle["ng"])
    isd = 1 - ng
    for k in range(1, 10):
        _compare(oracle[f"ssg_{k}"], gs["sin_sg"][:, :, k - 1],
                 isd, isd, rtol=_F32, name=f"ssg_{k}")
        _compare(oracle[f"csg_{k}"], gs["cos_sg"][:, :, k - 1],
                 isd, isd, rtol=_F32, name=f"csg_{k}")


def test_da_min_matches_oracle(oracle, gs):
    assert gs["da_min"] == pytest.approx(oracle["da_min"][0, 2], rel=1e-13)
    assert gs["da_min_c"] == pytest.approx(oracle["da_min_c"][0, 2],
                                           rel=1e-13)


def test_vertex_closed_form(gs):
    """The extended lattice renders the cube vertex a regular 120-degree
    kink: cosa=-1/2, sina=sqrt(3)/2, rsina=4/3 EXACTLY (to roundoff) —
    the structural reason duo has no vertex imprint."""
    ng = gs["ng"]
    v = (ng, ng)                      # Fortran B node (1, 1)
    assert gs["cosa"][v] == pytest.approx(-0.5, abs=1e-12)
    assert gs["sina"][v] == pytest.approx(np.sqrt(3.0) / 2.0, abs=1e-12)
    assert gs["rsina"][v] == pytest.approx(4.0 / 3.0, abs=1e-12)


def test_no_poison_in_consumed_regions(gs):
    """No big_number/NaN anywhere the duo operator lanes read: compute-
    domain rsina/cosa/sina B, full-domain sg, dxc/dyc, areas."""
    n, ng = gs["n"], gs["ng"]
    sl_b = slice(ng, ng + n + 1)
    assert np.all(np.abs(gs["rsina"][sl_b, sl_b]) < 1e3)
    assert np.all(np.abs(gs["cosa"][sl_b, sl_b]) <= 1.0)
    assert np.all(np.isfinite(gs["sin_sg"]))
    assert np.all(gs["sin_sg"] > 0.5)          # smooth lattice: no
    assert np.all(np.isfinite(gs["dxc"]))       # degenerate angles
    assert np.all(gs["area"] > 0.0)
    # frame replicated by the bounded outermost-ends block: real and
    # finite over the FULL node domain (rarea_c consumers see no poison)
    assert np.all(gs["area_c"] > 0.0)
    assert np.all(gs["area_c"] < 1.0e29)
    assert np.all(np.isfinite(gs["rarea_c"]))


@pytest.fixture(scope="module")
def duo_ctx():
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )

    return build_six_face_duo_context(12, 3, oracle_conventions=True)


def test_ctx_oracle_conventions_flags(duo_ctx):
    """oracle_conventions ctx: bounded metrics on every face, guards see
    bounded_domain=True, corner flags off, vertex rsina=4/3."""
    ctx = duo_ctx
    ng = 3
    for gs in ctx["gs6"]:
        assert gs["bounded_domain"] is True
        assert not any(gs[k] for k in ("sw_corner", "se_corner",
                                       "ne_corner", "nw_corner"))
        assert gs["rsina"][ng, ng] == pytest.approx(4.0 / 3.0, abs=1e-12)


def test_d_sw1_edge_block_gated_on_bounded(duo_ctx):
    """The auth-656 sin_sg edge ut/vt replacement fires on the plain
    lane and is skipped on the bounded lane; interior slots agree."""
    from legoesm.core.fv3_native_duo_sw_core import d_sw1_duo

    n, ng = 12, 3
    gs = duo_ctx["gs6"][0]
    bd = duo_ctx["bd"]
    m_a = n + 2 * ng
    rng = np.random.default_rng(7)
    shp_uc = (m_a + 1, m_a)
    shp_vc = (m_a, m_a + 1)
    uc = 10.0 * rng.standard_normal(shp_uc)
    vc = 10.0 * rng.standard_normal(shp_vc)
    delp = np.full((m_a, m_a), 3.0e4)
    pt = np.ones((m_a, m_a))
    zeros_x = np.zeros((n + 1, n))
    zeros_y = np.zeros((n, n + 1))

    def run(gs_in):
        return d_sw1_duo(delp, pt, None, uc, vc,
                         zeros_x.copy(), zeros_y.copy(),
                         np.zeros((n + 1, m_a)), np.zeros((m_a, n + 1)),
                         gs_in, bd, n + 1, n + 1, dt=450.0)

    out_b = run(gs)
    gs_plain = dict(gs)
    gs_plain["bounded_domain"] = False
    out_p = run(gs_plain)

    ut_b, ut_p = out_b["ut"], out_p["ut"]
    # Fortran ut row i==1 is the west-edge replacement target
    edge_row_b = ut_b[ng, :]
    edge_row_p = ut_p[ng, :]
    assert not np.allclose(edge_row_b, edge_row_p), (
        "edge gate is dead: bounded and plain lanes produced identical "
        "west-edge ut")
    # interior far from edges/corners must be lane-independent
    mid = slice(ng + 4, ng + n - 4)
    assert np.array_equal(ut_b[mid, mid], ut_p[mid, mid])


def test_c48_matches_zenodo_run_log():
    """Direct python-vs-paper-run gate: the C48 bounded gridstruct
    reproduces the numbers the Zenodo duo run printed in fms.out —
    da ratio to print precision, absolute area_c extrema to ~1e-12 rel
    (da_min_c differs ~0.036 m^2 from the printed tail: the log value
    is a 144-rank reduction; tile symmetry makes the six per-tile
    extrema equal, asserted below).  Radius is pinned by the absolute
    targets; omega is pinned by its own log print (separate test)."""
    from legoesm.grids.fv3_native_gridstruct import (
        build_fv3_native_gridstruct_bounded,
    )

    gs6 = [build_fv3_native_gridstruct_bounded(48, 3, tile=t)
           for t in range(1, 7)]
    # explicit six-face reduction (the log values are global reductions)
    da_min = min(g["da_min"] for g in gs6)
    da_max = max(g["da_max"] for g in gs6)
    da_min_c = min(g["da_min_c"] for g in gs6)
    da_max_c = max(g["da_max_c"] for g in gs6)
    assert da_max / da_min == pytest.approx(ZENODO_C48_DA_RATIO, rel=1e-14)
    assert da_max_c == pytest.approx(ZENODO_C48_DA_MAX_C, rel=1e-12)
    assert da_min_c == pytest.approx(ZENODO_C48_DA_MIN_C, rel=1e-11)
    # per-tile extrema agree across the cube (ED symmetry up to
    # per-face trig roundoff): the global reduction is not hiding a
    # divergent face
    for g in gs6[1:]:
        assert g["da_min_c"] == pytest.approx(gs6[0]["da_min_c"],
                                              rel=1e-9)


def test_omega_pinned_to_zenodo_log():
    """fms.out prints 'omega is 7.29210000000000E-005' — FV3_OMEGA must
    be that exact value (it differs from legoESM's constants.Omega),
    and the Coriolis fields must be built from it (omega=0 would zero
    fC everywhere)."""
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        build_fv3_native_gridstruct_bounded,
    )

    assert FV3_OMEGA == 7.2921e-5
    gs = build_fv3_native_gridstruct_bounded(12, 3)
    ng = 3
    lat = gs["grid_lat"][ng, ng]
    assert gs["fC"][ng, ng] == pytest.approx(
        2.0 * 7.2921e-5 * np.sin(lat), rel=1e-14)


def test_fixture_extract_sha_current():
    """The fixture's recorded extract SHA must match the CURRENT
    extract source — a drifted extract invalidates the certification
    (codex bounded-r2 finding 5)."""
    import hashlib

    z = np.load(FIXTURE)
    src = (Path(__file__).parents[2] / "scripts" / "validate"
           / "fv3_native" / "fv3_boundedgs_extract.F90")
    cur = hashlib.sha256(src.read_bytes()).hexdigest()
    assert str(z["extract_sha256"]) == cur, (
        "fv3_boundedgs_extract.F90 changed since the fixture was "
        "generated — regenerate tests/grids/fixtures/"
        "fv3_boundedgs_oracle.npz (gen_boundedgs_oracle.py)")
