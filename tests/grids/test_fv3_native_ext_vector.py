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
    from legoesm.grids.fv3_native_halos import _ed_ext_agrid_lonlat

    states = w2_six_face_state(ctx)
    u6 = [np.array(s["u"], copy=True) for s in states]
    v6 = [np.array(s["v"], copy=True) for s in states]
    ext_vector_dgrid_sixface(u6, v6, ectx)

    a_lon4, a_lat4 = _ed_ext_agrid_lonlat(N, _NG_P1)
    m_a = N + 2 * NG
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
    # error budget: c2l 2nd-order (~0.5) is the leading term; the
    # remap + projection add O(1e-2).  The interim index-copy halos sat
    # at ~14 m/s equivalent inconsistency at this resolution.
    assert worst_u < 1.5, worst_u


def test_ext_scalar_b_smooth_field_accuracy(ectx):
    """B-scalar ext halos land near the smooth field at EXT B nodes."""
    from legoesm.grids.fv3_native_halos import _ed_ext_stagger_lonlat
    from legoesm.grids.fv3_native_gridstruct import (
        build_kinked_corner_lonlat,
    )

    b_lon_e, b_lat_e = _ed_ext_stagger_lonlat(N, NG, "B")

    def f(lon, lat):
        return np.sin(lat) + 0.3 * np.cos(lon) * np.cos(lat)

    f6 = []
    for t in range(1, 7):
        lon_k, lat_k = build_kinked_corner_lonlat(N, NG, tile=t)
        with np.errstate(invalid="ignore"):
            fk = f(lon_k, lat_k)
        f6.append(fk)
    ext_scalar_sixface(f6, "B", ectx)
    worst = 0.0
    for t in range(6):
        fe = f(b_lon_e[t], b_lat_e[t])
        for j_f in list(range(1 - NG, 1)) + list(range(N + 2, N + NG + 2)):
            for i_f in range(1, N + 2):
                got = f6[t][i_f - 1 + NG, j_f - 1 + NG]
                worst = max(worst, abs(got - fe[i_f - 1 + NG, j_f - 1 + NG]))
    # k2e Lagrange (4th order) at C12; the kinked-copy value differs
    # from the ext value by the kink O(0.4·field-gradient) — the remap
    # must land ~two orders below that.
    assert worst < 5e-3, worst


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
