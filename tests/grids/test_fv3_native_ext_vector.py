"""Tests for the faithful ext_scalar/ext_vector six-face port.

Truth tiers (no Fortran available in unit scope; the dg-level oracle
rides the phase-4 sbatch harness):

- exactness invariants: constant-field preservation (Lagrange partition
  of unity), ring-4 A-table availability + partition of unity;
- analytic accuracy: c2l on balanced solid-body W2 winds reproduces the
  geographic field to 2nd order; ext_vector halo strips match the
  analytic field PROJECTED THROUGH THE SAME EXT BASES (layout- and
  convention-independent truth, the SB5b decisive-validation lesson);
- system gate: the assembled stepper with use_ext_bundle=True stays
  mass-exact and bounded over two steps, and its W2 edge error is far
  below the interim index-copy plateau (~14 m/s at C12).
"""

from pathlib import Path

import numpy as np
import pytest
from legoesm.core.fv3_native_duo_stepper import (
    build_six_face_duo_context,
    run_duo_sw,
    w2_six_face_state,
)
from legoesm.grids.fv3_native_ext_vector import (
    _NG_P1,
    _a2d_project,
    _pack_p1,
    build_ext_context,
    c2l_ord2_face,
    ext_scalar_sixface,
    ext_vector_dgrid_sixface,
)
from legoesm.grids.fv3_native_gridstruct import (
    build_fv3_native_gridstruct,
)

N, NG = 12, 3
U0 = 38.61068276698372


@pytest.fixture(scope="module")
def ctx():
    return build_six_face_duo_context(N, NG, use_ext_bundle=True)


@pytest.fixture(scope="module")
def kinked_gs6():
    return [build_fv3_native_gridstruct(N, NG, tile=t)
            for t in range(1, 7)]


@pytest.fixture(scope="module")
def ectx(kinked_gs6):
    return build_ext_context(N, NG, kinked_gs6)


def _analytic_geo(lon, lat, u0=U0):
    """Solid-body W2 (alpha=0): zonal wind u0*cos(lat)."""
    return u0 * np.cos(lat), np.zeros_like(lat)


def test_ring4_a_tables_partition_of_unity():
    from legoesm.grids.fv3_native_halos import compute_fv3_native_k2e

    tab = compute_fv3_native_k2e(N, remap_ng=4, k2e_nord=4)
    ij = tab["A_ij"]
    # ring-4 records exist on all four sides
    assert (ij[:, 1] == -3).any() and (ij[:, 1] == N + 4).any()
    assert (ij[:, 0] == -3).any() and (ij[:, 0] == N + 4).any()
    assert np.abs(tab["A_coef"].sum(axis=1) - 1.0).max() < 1e-10


def test_corner_lagrange_constant_preserved(ectx):
    f = np.full((N + 2 * NG, N + 2 * NG), 7.25)
    ectx["corner_a3"][0].fill(f)
    assert np.abs(f - 7.25).max() < 1e-12


def test_c2l_solid_body_second_order(ctx, kinked_gs6, ectx):
    states = w2_six_face_state(ctx)
    for t in range(6):
        gs = kinked_gs6[t]
        ua, va = c2l_ord2_face(states[t]["u"], states[t]["v"],
                               ectx["dx6"][t], ectx["dy6"][t],
                               ectx["amat6"][t], N, NG)
        sl = slice(NG, NG + N)          # interior cells
        ua_t, va_t = _analytic_geo(gs["agrid_lon"][sl, sl],
                                   gs["agrid_lat"][sl, sl])
        # covariant->geographic on the C12 lattice: 2nd-order average
        assert np.nanmax(np.abs(ua[sl, sl] - ua_t)) < 0.7
        assert np.nanmax(np.abs(va[sl, sl] - va_t)) < 0.7


def test_ext_vector_halo_matches_analytic_projection(ctx, ectx):
    """Halo strips == the analytic wind pushed through the SAME
    ng=4 geographic lattice + a2d ext projection (truth by construction:
    identical bases, no exchange/remap/c2l error)."""
    from legoesm.grids.fv3_native_ext_vector import ext_parity_lonlat_ref

    states = w2_six_face_state(ctx)
    u6 = [np.array(s["u"], copy=True) for s in states]
    v6 = [np.array(s["v"], copy=True) for s in states]
    ext_vector_dgrid_sixface(u6, v6, ectx)

    a_lon4, a_lat4 = ext_parity_lonlat_ref(N, _NG_P1, "A")
    worst_u = 0.0
    for t in range(6):
        ug_t, vg_t = _analytic_geo(a_lon4[t], a_lat4[t])
        ud_t, vd_t = _a2d_project(ug_t, vg_t, t, ectx)
        ngp = _NG_P1
        # compare the halo side strips (rings 1..NG), excluding the
        # corner wedges (own Lagrange convention, checked separately)
        for j_f in list(range(1 - NG, 1)) + list(range(N + 2, N + NG + 2)):
            for i_f in range(1, N + 1):
                got = u6[t][i_f - 1 + NG, j_f - 1 + NG]
                want = ud_t[i_f - 1 + ngp, j_f - 2 + ngp]
                worst_u = max(worst_u, abs(got - want))
    # error budget: c2l 2nd-order is the leading term (measured 0.152
    # at C12); the remap + projection add O(1e-2).  The interim
    # index-copy halos sat at ~14 m/s equivalent inconsistency, the
    # rejected position-only remap at ~22.
    assert worst_u < 0.5, worst_u


def test_ext_scalar_b_smooth_field_accuracy(ectx, kinked_gs6):
    """B-scalar ext halos land near the smooth field at EXT B nodes."""
    from legoesm.grids.fv3_native_ext_vector import ext_parity_lonlat_ref
    from legoesm.grids.fv3_native_gridstruct import (
        build_kinked_corner_lonlat,
    )

    b_lon_e, b_lat_e = ext_parity_lonlat_ref(N, NG, "B")

    def f(lon, lat):
        return np.sin(lat) + 0.3 * np.cos(lon) * np.cos(lat)

    f6 = []
    for t in range(1, 7):
        lon_k, lat_k = build_kinked_corner_lonlat(N, NG, tile=t)
        with np.errstate(invalid="ignore"):
            fk = f(lon_k, lat_k)
        f6.append(fk)
    def _worst(ectx_used, fields):
        ext_scalar_sixface(fields, "B", ectx_used)
        w = 0.0
        for t in range(6):
            fe = f(b_lon_e[t], b_lat_e[t])
            for j_f in (list(range(1 - NG, 1))
                        + list(range(N + 2, N + NG + 2))):
                for i_f in range(1, N + 2):
                    got = fields[t][i_f - 1 + NG, j_f - 1 + NG]
                    w = max(w, abs(got - fe[i_f - 1 + NG, j_f - 1 + NG]))
        return w

    # RESOLVED runtime order (k2e_nord=2, the default context): measured
    # 2.23e-3 at C12 -- and the extchain oracle certificate proves the
    # nord=2 chain matches the oracle, so this accuracy is upstream's
    # own.  The kinked-copy value differs from the ext value by the kink
    # O(0.4*field-gradient); the remap still lands well below that.
    worst2 = _worst(ectx, [np.array(a, copy=True) for a in f6])
    assert worst2 < 5e-3, worst2         # measured 2.23e-3 at nord=2

    # 4th-order calibration kept explicitly (codex nstep review #2: the
    # old wording claimed 4th order while the shared fixture silently
    # moved to 2 -- a ~310x weaker check).  The order gap doubles as a
    # mutation control for a context that ignored k2e_nord.
    from legoesm.grids.fv3_native_ext_vector import build_ext_context
    ectx4 = build_ext_context(N, NG, kinked_gs6, k2e_nord=4)
    worst4 = _worst(ectx4, f6)
    assert worst4 < 5e-5, worst4         # measured 7.2e-6 at nord=4
    assert worst2 > 10.0 * worst4        # the order gap is real


def test_stepper_two_steps_bundle_mass_exact_and_bounded(ctx):
    states = w2_six_face_state(ctx)
    area6 = [np.array(g["area"]) for g in ctx["gs6"]]
    sl = slice(NG, NG + N)

    def mass(ss):
        return sum(float((ss[t]["delp"][sl, sl] * area6[t][sl, sl]).sum())
                   for t in range(6))

    m0 = mass(states)
    out = run_duo_sw(ctx, states, dt=600.0, nsteps=2)
    m2 = mass(out)
    assert abs(m2 - m0) / m0 < 1e-12
    for t in range(6):
        u = out[t]["u"][sl, :]
        assert np.all(np.isfinite(u))
        assert np.abs(u).max() < 8.0 * U0          # SB4 bound, post-norm


def test_ext_scalar_unknown_stagger_raises(ectx):
    with pytest.raises(ValueError):
        ext_scalar_sixface([np.zeros((2, 2))] * 6, "CX", ectx)


def test_pack_p1_layout():
    ua = np.arange((N + 2 * NG) ** 2, dtype=float).reshape(
        N + 2 * NG, N + 2 * NG)
    p1 = _pack_p1(ua, N, NG)
    d = _NG_P1 - NG
    assert p1.shape == (N + 2 * _NG_P1, N + 2 * _NG_P1)
    assert np.isnan(p1[0, :]).all()
    assert (p1[d:d + N + 2 * NG, d:d + N + 2 * NG] == ua).all()


# ---------------------------------------------------------------------------
# independent Fortran projection oracle (codex ext r1 P1-1)
# ---------------------------------------------------------------------------

_FIXTURE = (Path(__file__).parent / "fixtures"
            / "fv3_extproj_oracle.npz")


def _load_extproj():
    if not _FIXTURE.exists():
        pytest.skip("extproj oracle fixture not generated "
                    "(scripts/cluster/fv3_native/extproj_oracle.sbatch)")
    return np.load(_FIXTURE, allow_pickle=False)


def _bases_for(n, ng, tile):
    from legoesm.grids.fv3_native_ext_vector import ext_parity_lonlat_ref
    from legoesm.grids.fv3_native_halos import _compute_ext_vectors_native

    a_lon, a_lat = ext_parity_lonlat_ref(n, ng, "A")
    b_lon, b_lat = ext_parity_lonlat_ref(n, ng, "B")
    vlon, vlat, ew, es = _compute_ext_vectors_native(
        a_lon, a_lat, b_lon, b_lat)
    return {"vlon4": vlon, "vlat4": vlat, "ew4": ew, "es4": es,
            "a_lonlat": (a_lon, a_lat), "b_lonlat": (b_lon, b_lat)}


def test_extproj_fixture_input_sha_enforced():
    d = _load_extproj()
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_extproj_oracle",
        Path(__file__).parents[2] / "scripts/validate/fv3_native"
        / "gen_extproj_oracle.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    n, ng, tile = int(d["n"]), int(d["ng"]), int(d["tile"])
    b = _bases_for(n, ng, tile)
    a_lon, a_lat = b["a_lonlat"]
    b_lon, b_lat = b["b_lonlat"]
    import hashlib

    text = gen.serialize_inputs(n, ng, a_lon[tile - 1], a_lat[tile - 1],
                                b_lon[tile - 1], b_lat[tile - 1],
                                d["ug"], d["vg"])
    assert hashlib.sha256(text.encode()).hexdigest() == str(d["input_sha256"])


def test_extproj_oracle_bases_match():
    """vlon/vlat/ew/es vs the INDEPENDENT Fortran a2stag_metrics.

    Tolerance 1e-13 rel: upstream computes the trig in
    selected_real_kind(20) (unit_vect_latlon_ext) and normalizes in
    r8; the python bases are float64 throughout.
    """
    d = _load_extproj()
    n, ng, tile = int(d["n"]), int(d["ng"]), int(d["tile"])
    b = _bases_for(n, ng, tile)
    t = tile - 1
    lo = 1 - ng
    for rec, arr, idx in (
            ("vlon", b["vlon4"][t], 3), ("vlat", b["vlat4"][t], 3)):
        r = d[rec]
        for i_f, j_f, k, val in r:
            got = arr[int(i_f) - lo, int(j_f) - lo, int(k) - 1]
            assert abs(got - val) <= 1e-13 * max(1.0, abs(val)), \
                (rec, i_f, j_f, k, got, val)
    for rec, arr in (("ew", b["ew4"][t]), ("es", b["es4"][t])):
        r = d[rec]
        for k, i_f, j_f, m, val in r:
            got = arr[int(i_f) - lo, int(j_f) - lo, int(k) - 1, int(m) - 1]
            assert abs(got - val) <= 1e-13 * max(1.0, abs(val)), \
                (rec, k, i_f, j_f, m, got, val)


def test_extproj_oracle_projections_match_and_discriminate():
    """UD/VD/UC/VC vs the verbatim Fortran projections — full arrays
    (all sides, both components, both staggerings), plus the basis-
    variant mutation ("es[...,0] <-> [...,1]") must now FAIL against
    the independent Fortran values (the codex r1 shared-oracle hole).
    """
    from legoesm.grids.fv3_native_ext_vector import (
        _a2c_project,
        _a2d_project,
    )

    d = _load_extproj()
    n, ng, tile = int(d["n"]), int(d["ng"]), int(d["tile"])
    b = _bases_for(n, ng, tile)
    t = tile - 1
    ectx = {"vlon4": b["vlon4"], "vlat4": b["vlat4"],
            "ew4": b["ew4"], "es4": b["es4"]}
    ud, vd = _a2d_project(d["ug"], d["vg"], t, ectx)
    uc, vc = _a2c_project(d["ug"], d["vg"], t, ectx)
    lo = 1 - ng

    def check(rec, arr, col_off=0, row_off=0):
        worst = 0.0
        for i_f, j_f, val in d[rec]:
            got = arr[int(i_f) - lo - row_off, int(j_f) - lo - col_off]
            worst = max(worst, abs(got - val))
        return worst

    # index maps: ud rows = centers, cols = nodes j (fort j -> col j-1-lo)
    assert check("ud", ud, col_off=1) < 1e-12
    assert check("vd", vd, row_off=1) < 1e-12
    assert check("uc", uc, row_off=1) < 1e-12
    assert check("vc", vc, col_off=1) < 1e-12

    # mutation teeth: swap the D es/ew variants -> large mismatch
    mut = {"vlon4": b["vlon4"], "vlat4": b["vlat4"],
           "ew4": b["ew4"][..., ::-1], "es4": b["es4"][..., ::-1]}
    ud_m, vd_m = _a2d_project(d["ug"], d["vg"], t, mut)
    assert check("ud", ud_m, col_off=1) > 1.0
    assert check("vd", vd_m, row_off=1) > 1.0


# ---------------------------------------------------------------------------
# nonconstant staggered corner-fill accuracy (codex ext r1 P2-6)
# ---------------------------------------------------------------------------

def _smooth(lon, lat):
    return np.sin(lat) * 40.0 + 12.0 * np.cos(lon) * np.cos(lat)


def _smooth_a_halo_worst(ectx_used, kinked_gs6):
    from legoesm.grids.fv3_native_ext_vector import ext_parity_lonlat_ref

    a_lon_e, a_lat_e = ext_parity_lonlat_ref(N, NG, "A")
    f6 = [_smooth(gs["agrid_lon"], gs["agrid_lat"]).copy()
          for gs in kinked_gs6]
    ext_scalar_sixface(f6, "A", ectx_used)
    worst = 0.0
    for t in range(6):
        fe = _smooth(a_lon_e[t], a_lat_e[t])
        for i_f in range(1 - NG, N + NG + 1):
            for j_f in range(1 - NG, N + NG + 1):
                if not ((i_f < 1 or i_f > N) or (j_f < 1 or j_f > N)):
                    continue
                worst = max(worst, abs(
                    f6[t][i_f - 1 + NG, j_f - 1 + NG]
                    - fe[i_f - 1 + NG, j_f - 1 + NG]))
    return worst


def test_corner_lagrange_smooth_scalar_a(ectx, kinked_gs6):
    """Smooth-field halo accuracy at BOTH interpolation orders.

    The default context now carries the RESOLVED runtime k2e_nord=2
    (the pinned tree's own duogrid_init; the extchain oracle
    certificate proves the nord=2 chain matches the oracle to <=2e-10),
    so its accuracy on this n=12 kinked harness is the ORACLE's own:
    measured 5.814 on a field ~41.  The historic 0.15 bound was a
    nord=4 calibration -- kept below on an explicit nord=4 context, so
    the order sensitivity itself stays pinned (2-vs-4 must differ by
    ~90x here; a context that silently ignored k2e_nord would fail
    one of the two).
    """
    worst2 = _smooth_a_halo_worst(ectx, kinked_gs6)
    assert worst2 < 8.0, worst2          # measured 5.814 at nord=2

    from legoesm.grids.fv3_native_ext_vector import build_ext_context
    ectx4 = build_ext_context(N, NG, kinked_gs6, k2e_nord=4)
    worst4 = _smooth_a_halo_worst(ectx4, kinked_gs6)
    assert worst4 < 0.15, worst4         # measured 0.064 at nord=4
    assert worst2 > 10.0 * worst4        # the order gap is real


def test_corner_lagrange_smooth_scalar_dstag(ectx):
    """The (0,1) staggered operator on a smooth scalar sampled at the
    EXT D-u lattice: wedge extrapolation error stays under 0.8
    (measured 0.57; the stagger half-shift abscissae are upstream's
    own approximation, compute_lagrange_coeff)."""
    from legoesm.grids.fv3_native_halos import _ED_CARTS, _ed_line

    line = _ed_line(N, 2 * (NG + 2))
    iv = np.array([line[2 * i] for i in range(1 - NG, N + NG + 1)])
    jv = np.array([line[2 * j - 1] for j in range(1 - NG, N + 1 + NG + 1)])
    xg, yg = np.meshgrid(iv, jv, indexing="ij")
    worst = 0.0
    for t in range(6):
        cx, cy, cz = _ED_CARTS[t](xg, yg)
        r = np.sqrt(cx**2 + cy**2 + cz**2)
        lon = np.mod(np.arctan2(cy, cx), 2 * np.pi)
        lat = np.arcsin(cz / r)
        fd = _smooth(lon, lat)
        g = fd.copy()
        for i_f in range(1 - NG, N + NG + 1):
            for j_f in range(1 - NG, N + 1 + NG + 1):
                if (i_f < 1 or i_f > N) and (j_f < 1 or j_f > N + 1):
                    g[i_f - 1 + NG, j_f - 1 + NG] = np.nan
        ectx["corner_du3"][t].fill(g)
        assert np.isfinite(g).all()
        worst = max(worst, float(np.nanmax(np.abs(g - fd))))
    assert worst < 0.8, worst


def test_flags_mutually_exclusive(ctx):
    bad = dict(ctx)
    bad["use_k2e_scalars"] = True
    states = w2_six_face_state(ctx)
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_duo_sw(bad, states, dt=600.0, nsteps=1)


def test_vector_corner_variant_wiring(kinked_gs6):
    """a2d = measurement variant: unknown mode raises; a2d skips the
    final covariant corner overwrite (wedges differ from lagrange)."""
    with pytest.raises(ValueError, match="vector_corner"):
        build_ext_context(N, NG, kinked_gs6, vector_corner="bogus")
    e_l = build_ext_context(N, NG, kinked_gs6, vector_corner="lagrange")
    e_a = build_ext_context(N, NG, kinked_gs6, vector_corner="a2d")
    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True)
    states = w2_six_face_state(ctx)
    outs = {}
    for key, ectx_v in (("lagrange", e_l), ("a2d", e_a)):
        u6 = [np.array(s["u"], copy=True) for s in states]
        v6 = [np.array(s["v"], copy=True) for s in states]
        ext_vector_dgrid_sixface(u6, v6, ectx_v)
        outs[key] = u6
    wl = outs["lagrange"][0][:NG, -NG:]      # a wedge block
    wa = outs["a2d"][0][:NG, -NG:]
    assert np.abs(wl - wa).max() > 0.01


# ---------------------------------------------------------------------------
# corner-Lagrange oracle (oracle-closeness directive, 2026-07-17)
# ---------------------------------------------------------------------------

_CL_FIXTURE = (Path(__file__).parent / "fixtures"
               / "fv3_cornerlag_oracle.npz")


def test_cornerlag_oracle_all_staggers():
    """_CornerLagrange.fill vs the verbatim Fortran
    compute_lagrange_coeff + fill_corner_region_2d, all four
    staggerings, full arrays — certifies the signed-arc Lagrange-weight
    equivalence and the nine-slot sequence against the oracle."""
    if not _CL_FIXTURE.exists():
        pytest.skip("cornerlag oracle fixture not generated "
                    "(scripts/cluster/fv3_native/cornerlag_oracle.sbatch)")
    from legoesm.grids.fv3_native_ext_vector import (
        _CornerLagrange,
        ext_parity_lonlat_ref,
    )

    d = np.load(_CL_FIXTURE, allow_pickle=False)
    n, ng, tile = int(d["n"]), int(d["ng"]), int(d["tile"])
    a_lon4, a_lat4 = ext_parity_lonlat_ref(n, ng + 1, "A")
    worst = {}
    for istag, jstag in ((0, 0), (1, 1), (0, 1), (1, 0)):
        f = np.array(d[f"fld_{istag}{jstag}"], copy=True)
        op = _CornerLagrange(a_lon4[tile - 1], a_lat4[tile - 1],
                             n, ng, istag, jstag)
        op.fill(f)
        rec = d[f"out_{istag}{jstag}"]
        w = 0.0
        for i_f, j_f, val in rec:
            got = f[int(i_f) - 1 + ng, int(j_f) - 1 + ng]
            w = max(w, abs(got - val))
        worst[(istag, jstag)] = w
        # r8 distance-ratio products vs float64 signed-arc products:
        # mathematically identical on great circles; float error
        # amplified by extrapolation weights (|w| up to ~35)
        assert w < 5e-11, (istag, jstag, w)
    # the fill must have CHANGED corner slots (non-vacuity)
    f0 = np.array(d["fld_00"], copy=True)
    rec0 = {(int(i), int(j)): v for i, j, v in d["out_00"]}
    changed = sum(
        1 for (i_f, j_f), v in rec0.items()
        if abs(v - f0[i_f - 1 + ng, j_f - 1 + ng]) > 1e-9)
    assert changed >= 4 * 9, changed


def test_k2e_tables_mirror_vs_authoritative():
    """Phase-3 provenance closure: the k2e tables pinned against the
    luanfs MIRROR generator must match the AUTHORITATIVE modular
    global_grid_gen_k2e.F90 (Zenodo symmetryclean) — all six stagger
    families, loc + coef, C12 and C24."""
    mirror = Path(__file__).parent / "fixtures" / "fv3_duogrid_oracle.npz"
    auth = Path(__file__).parent / "fixtures" / "fv3_duogrid_oracle_auth.npz"
    if not auth.exists():
        pytest.skip("auth fixture not generated "
                    "(scripts/validate/fv3_native/gen_duogrid_oracle_auth.sh)")
    dm = np.load(mirror, allow_pickle=False)
    da = np.load(auth, allow_pickle=False)
    checked = 0
    for k in dm.files:
        if not (k.startswith("k2e_") or "coef" in k or "loc" in k):
            continue
        assert k in da.files, k
        a, b = np.asarray(dm[k]), np.asarray(da[k])
        assert a.shape == b.shape, (k, a.shape, b.shape)
        if a.dtype.kind in "fc":
            worst = float(np.max(np.abs(a - b))) if a.size else 0.0
            assert worst == 0.0, (k, worst)
        else:
            assert np.array_equal(a, b), k
        checked += 1
    assert checked >= 12, checked      # 6 families x (loc+coef) minimum
