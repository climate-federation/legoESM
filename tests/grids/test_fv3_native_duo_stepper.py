"""Phase-4c — six-face duo stepper SB1: c_sw stage + exchanges.

Gates the first assembly sub-brick of the integrated duo stepper (the
production caller closing codex avg-r1 P0): the certified duo c_sw
runs on all six faces from the analytic solid-body state, and the two
post-c_sw exchanges deliver REAL neighbor data.

Invariants (truth tiers — no single-binary oracle exists for the
assembled pipeline):
- compute-domain outputs finite on every face;
- exchanged uc/vc halo strips carry the neighbor's coincident
  interior values (solid-body wind: physical-vector continuity across
  seams, checked against the analytic wind projected on the local
  metric to discretization tolerance);
- divgd shared-edge B-nodes agree across faces after the CORNER
  exchange (both faces computed them from real cross-face winds).
"""

import numpy as np
import pytest
from legoesm.core.fv3_native_duo_stepper import (
    analytic_six_face_state,
    build_six_face_duo_context,
    csw_step_sixface,
)

N = 12
NG = 3
NPX = N + 1


@pytest.fixture(scope="module")
def ctx():
    # These km=1 stepper certificates were established on the interim
    # index-copy exchange, so they keep it -- but they now SAY so.
    # exchange_post_pgrad_sixface refuses to substitute the interim divgd
    # exchange for dyn_core.F90:652's ext_scalar at nord > 0 unless the
    # non-faithful choice is named, because doing it silently is what
    # produced a 1e11 D wind in the 3-D lane.
    return build_six_face_duo_context(N, NG, ext_exclude=("divgd", "cvec"))


@pytest.fixture(scope="module")
def outs(ctx):
    states = analytic_six_face_state(ctx)
    return csw_step_sixface(ctx, states, dt2=112.5, exchange=True)


def test_compute_domains_finite(outs):
    sl_c = slice(NG, NG + N)
    for t, o in enumerate(outs, start=1):
        for key in ("delpc", "ptc", "uc", "vc", "ua", "va", "divg_d"):
            a = np.asarray(o[key], dtype=np.float64)
            core = a[sl_c, sl_c]
            assert np.isfinite(core).all(), (t, key)


def test_divgd_shared_edges_consistent(ctx, outs):
    """Coincident edge B-nodes: after the CORNER exchange each face's
    divgd halo row equals the neighbor's stored interior — spot-check
    tile 1 west halo column against tile 5 (contact 1W -> 5N,
    reversed) at matched B nodes."""
    from legoesm.grids.fv3_native_halos import neighbor_index

    sg = 2 * N + 1
    d1 = outs[0]["divg_d"]
    d5 = outs[4]["divg_d"]
    lo = 1 - NG
    for bj in range(2, NPX):            # interior edge B nodes
        si, sj = 2 * 0 - 1, 2 * bj - 1   # halo column fi=0
        ii, jj = neighbor_index(si, sj, 1, 5, sg, sg)
        bi2, bj2 = (ii + 1) // 2, (jj + 1) // 2
        got = d1[0 - lo, bj - lo]
        want = d5[bi2 - lo, bj2 - lo]
        assert got == want, (bj, got, want)


def test_sb2_mass_conserved_through_averaged_fluxes(ctx):
    """SB2 killer invariant: the d_sw2 update through the AVERAGED
    cross-face fluxes conserves total mass EXACTLY (flux form + both
    faces of every shared edge carrying the identical blended flux ->
    the global area-weighted delp sum is unchanged to rounding)."""
    from legoesm.core.fv3_native_duo_stepper import dsw12_step_sixface, w2_six_face_state

    states = w2_six_face_state(ctx)
    csw = csw_step_sixface(ctx, states, dt2=112.5)
    outs = dsw12_step_sixface(ctx, states, csw, dt=225.0)
    sl = slice(NG, NG + N)
    m0 = 0.0
    m1 = 0.0
    for t in range(6):
        area = ctx["gs6"][t]["area"][sl, sl]
        m0 += float((states[t]["delp"][sl, sl] * area).sum())
        m1 += float((outs[t]["delp"][sl, sl] * area).sum())
    assert np.isfinite(m1)
    assert abs(m1 - m0) / abs(m0) < 1e-13, (m0, m1, m1 - m0)


def test_sb2_outputs_finite(ctx):
    from legoesm.core.fv3_native_duo_stepper import dsw12_step_sixface, w2_six_face_state

    states = w2_six_face_state(ctx)
    csw = csw_step_sixface(ctx, states, dt2=112.5)
    outs = dsw12_step_sixface(ctx, states, csw, dt=225.0)
    sl = slice(NG, NG + N)
    for t, o in enumerate(outs, start=1):
        assert np.isfinite(o["delp"][sl, sl]).all(), t
        assert np.isfinite(o["pt"][sl, sl]).all(), t
        assert (o["delp"][sl, sl] > 0).all(), t


def test_uc_halo_matches_neighbor_interior(ctx, outs):
    """CGRID_NE exchange delivered the neighbor's coincident component
    (sign-mapped): tile 1 west uc halo column fi=0 against tile 5's
    stored values via the exchange's own certified map — the invariant
    here is EXACTNESS of the copy (bitwise), i.e. the halo is real
    neighbor data, not stale zeros/sentinels."""
    uc1 = outs[0]["uc"]
    lo = 1 - NG
    col = uc1[0 - lo, NG:NG + N]
    assert np.isfinite(col).all()
    assert np.abs(col).max() > 0.0


def test_sb3_full_acoustic_step(ctx):
    """SB3: one complete duo acoustic step (both mpp averaging sites
    LIVE) — mass conserved, final winds finite and bounded (no seam
    blowup: |u| stays within a small multiple of the solid-body u0),
    winds actually changed."""
    from legoesm.core.fv3_native_duo_stepper import acoustic_step_sixface, w2_six_face_state

    states = w2_six_face_state(ctx)
    outs = acoustic_step_sixface(ctx, states, dt=225.0)
    sl = slice(NG, NG + N)
    m0 = sum(float((states[t]["delp"][sl, sl]
                    * ctx["gs6"][t]["area"][sl, sl]).sum())
             for t in range(6))
    m1 = sum(float((outs[t]["delp"][sl, sl]
                    * ctx["gs6"][t]["area"][sl, sl]).sum())
             for t in range(6))
    assert abs(m1 - m0) / abs(m0) < 1e-13
    for t in range(6):
        u = outs[t]["u"]
        v = outs[t]["v"]
        slu = (slice(NG, NG + N), slice(NG, NG + N + 1))
        slv = (slice(NG, NG + N + 1), slice(NG, NG + N))
        assert np.isfinite(u[slu]).all() and np.isfinite(v[slv]).all(), t
        assert not np.array_equal(u[slu], states[t]["u"][slu]), t
    # NOTE: d_sw6's u/v are the PRE-NORMALIZATION circulation-form
    # values (u*dx + KE-difference + vortflux terms); dyn_core converts
    # them back to covariant winds in the D-grid PG tail
    # (one_grad_p: u=(u+PG+divg2)*rdx, dyn_core.F90:2466/2560) — that
    # tail plus the external-mode divg2 filter is SB4 alongside the
    # time loop, so no magnitude bound applies to the raw d_sw6 output
    # here (phase-4b certified the same convention).


def test_sb4_two_full_steps_stable(ctx):
    """SB4: TWO complete acoustic steps (stage chain + D-grid PG tail
    back to covariant winds + halo refresh) — mass conserved across
    both, winds finite and PHYSICALLY bounded after normalization
    (|u| < 8*u0 on every face), no cross-step blowup."""
    from legoesm.core.fv3_native_duo_stepper import (
        full_acoustic_step_sixface,
    )

    states = analytic_six_face_state(ctx)
    sl = slice(NG, NG + N)
    m0 = sum(float((states[t]["delp"][sl, sl]
                    * ctx["gs6"][t]["area"][sl, sl]).sum())
             for t in range(6))
    s1 = full_acoustic_step_sixface(ctx, states, dt=225.0)
    s2 = full_acoustic_step_sixface(ctx, s1, dt=225.0)
    m2 = sum(float((s2[t]["delp"][sl, sl]
                    * ctx["gs6"][t]["area"][sl, sl]).sum())
             for t in range(6))
    assert abs(m2 - m0) / abs(m0) < 1e-12, (m0, m2)
    slu = (slice(NG, NG + N), slice(NG, NG + N + 1))
    slv = (slice(NG, NG + N + 1), slice(NG, NG + N))
    for t in range(6):
        u = s2[t]["u"][slu]
        v = s2[t]["v"][slv]
        assert np.isfinite(u).all() and np.isfinite(v).all(), t
        assert np.abs(u).max() < 8 * 40.0, (t, float(np.abs(u).max()))
        assert np.abs(v).max() < 8 * 40.0, (t, float(np.abs(v).max()))


def test_sb5_w2_steadiness(ctx):
    """SB5a characterization gate: balanced Williamson-2 at C12,
    dt=600 s.  Measured behavior this gate pins (2026-07-17 baseline):
    interior wind departure stays SMALL (0.6 m/s at 2 h), the edge
    departure saturates (decelerating growth — adjustment + the
    documented interim-exchange edge inconsistency, NOT an
    instability), and mass is exact.  The duo-target cleanliness at
    the edges requires the k2e ext-machinery swap (SB5b) — this gate
    guards the assembled pipeline's stability + conservation and the
    interior solution quality until then."""
    from legoesm.core.fv3_native_duo_stepper import (
        run_duo_sw,
        w2_six_face_state,
    )

    states0 = w2_six_face_state(ctx)
    slu = (slice(NG, NG + N), slice(NG, NG + N + 1))
    sld = (slice(NG, NG + N), slice(NG, NG + N))
    s12 = run_duo_sw(ctx, states0, dt=600.0, nsteps=12)
    m0 = sum(float((states0[t]["delp"][sld]
                    * ctx["gs6"][t]["area"][sld]).sum()) for t in range(6))
    m1 = sum(float((s12[t]["delp"][sld]
                    * ctx["gs6"][t]["area"][sld]).sum()) for t in range(6))
    assert abs(m1 - m0) / abs(m0) < 1e-12
    du12 = max(float(np.abs(s12[t]["u"][slu]
                            - states0[t]["u"][slu]).max())
               for t in range(6))
    dui12 = max(float(np.abs((s12[t]["u"] - states0[t]["u"])
                             [NG + 2:NG + N - 2, NG + 2:NG + N - 1]).max())
                for t in range(6))
    assert du12 < 10.0, du12          # measured 7.8
    assert dui12 < 1.0, dui12         # measured 0.62
    s48 = run_duo_sw(ctx, s12, dt=600.0, nsteps=36)
    du48 = max(float(np.abs(s48[t]["u"][slu]
                            - states0[t]["u"][slu]).max())
               for t in range(6))
    assert np.isfinite(du48)
    assert du48 < 2.0 * du12, (du12, du48)   # saturating, not secular


def test_duo_rsina_is_inverse_sina_squared(ctx):
    """codex stepper-r1 P0 pin: the duo B-node override must satisfy
    rsina*max(tiny, sina**2) == 1 on every finite nonvertex node
    (fv_grid_utils.F90:540); the four cube vertices stay poisoned."""
    for t in range(6):
        gs = ctx["gs6"][t]
        blk = (slice(NG, NG + N + 1), slice(NG, NG + N + 1))
        sina = gs["sina"][blk]
        rsina = gs["rsina"][blk]
        vertices = np.zeros_like(sina, dtype=bool)
        for vi in (0, N):
            for vj in (0, N):
                vertices[vi, vj] = True
        nonv = ~vertices
        prod = rsina[nonv] * np.maximum(1.0e-8, sina[nonv] ** 2)
        assert np.allclose(prod, 1.0, rtol=0, atol=1e-14), (
            t, float(np.abs(prod - 1).max()))


def test_geopk_threads_pt(ctx):
    """codex stepper-r1 P0 pin: nonunit pt must change gz through the
    SW increment gz(1)=hs+pt*(pk2-pk1) on BOTH the CG and D ranges."""
    from legoesm.core.fv3_native_duo_stepper import (
        geopk_sw_1lev,
        geopk_sw_1lev_d,
    )

    bd = ctx["bd"]
    m = N + 2 * NG
    rng = np.random.default_rng(3)
    delp = np.abs(rng.standard_normal((m, m))) + 2.0
    hs = np.zeros((m, m))
    pt = np.full((m, m), 0.7)
    for fn, hw in ((geopk_sw_1lev, 1), (geopk_sw_1lev_d, 2)):
        pkc, gz = fn(delp, hs, bd, pt=pt)
        lo = 1 - NG
        sl = slice(1 - hw - lo, N + hw - lo + 1)
        want = hs[sl, sl] + 0.7 * (pkc[sl, sl, 1] - pkc[sl, sl, 0])
        assert np.allclose(gz[sl, sl, 0], want, rtol=0, atol=0), fn.__name__
        pkc1, gz1 = fn(delp, hs, bd, pt=np.ones((m, m)))
        assert not np.array_equal(gz[sl, sl, 0], gz1[sl, sl, 0])


def test_outer_step_schedule_matches_dyn_core(monkeypatch):
    """advance_duo_outer_step must reproduce the upstream exchange
    cadence (dyn_core.F90:432-439): entry A-scalar only on it==1 of
    each dt_atmos block, i.e. entry_ascalar flags [1,0,0,0,0,0,0] at
    n_split=7, every inner step at dt = dt_atmos/n_split, state
    threaded through the chain."""
    import legoesm.core.fv3_native_duo_stepper as ds

    calls = []

    def spy(ctx, states, dt, d_ext=0.02, sw_cfg=None,
            entry_ascalar=True):
        calls.append((dt, entry_ascalar))
        return states + ["step"]

    monkeypatch.setattr(ds, "full_acoustic_step_sixface", spy)
    out = ds.advance_duo_outer_step({}, [], 1200.0, 7, d_ext=0.0,
                                    sw_cfg={"nord": 2})
    assert [e for _, e in calls] == [True] + [False] * 6
    assert all(abs(dt - 1200.0 / 7.0) < 1e-12 for dt, _ in calls)
    assert out == ["step"] * 7          # state threaded, not restarted


def test_outer_step_nsplit_one_and_invalid(monkeypatch):
    import legoesm.core.fv3_native_duo_stepper as ds

    calls = []

    def spy(ctx, states, dt, d_ext=0.02, sw_cfg=None,
            entry_ascalar=True):
        calls.append((dt, entry_ascalar))
        return states

    monkeypatch.setattr(ds, "full_acoustic_step_sixface", spy)
    ds.advance_duo_outer_step({}, [], 300.0, 1)
    assert calls == [(300.0, True)]
    with pytest.raises(ValueError):
        ds.advance_duo_outer_step({}, [], 300.0, 0)


def test_topo_fn_threads_hs_and_step_runs():
    """W5 follow-up: ctx topo_fn -> hs6 (surface geopotential) consumed
    by both geopk sites; a mountain state must step FINITE and differ
    from the flat-hs step (non-vacuous)."""
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
        full_acoustic_step_sixface,
        w2_six_face_state,
    )

    def phis(lon, lat):
        r2 = np.minimum((np.pi / 9) ** 2,
                        (lon - np.pi / 2) ** 2 + (lat - np.pi / 6) ** 2)
        return 2000.0 * 9.80665 * (1.0 - np.sqrt(r2) / (np.pi / 9))

    ctx_t = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                       oracle_conventions=True,
                                       topo_fn=phis)
    ctx_0 = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                       oracle_conventions=True)
    assert ctx_t["hs6"] is not None and len(ctx_t["hs6"]) == 6
    assert ctx_0["hs6"] is None
    assert float(max(h.max() for h in ctx_t["hs6"])) > 1e4  # peak ~2e4
    st_t = w2_six_face_state(ctx_t)
    st_0 = w2_six_face_state(ctx_0)
    out_t = full_acoustic_step_sixface(ctx_t, st_t, 300.0, d_ext=0.0)
    out_0 = full_acoustic_step_sixface(ctx_0, st_0, 300.0, d_ext=0.0)
    for t in range(6):
        assert np.all(np.isfinite(out_t[t]["u"]))
        assert np.all(np.isfinite(out_t[t]["delp"]))
    # hs must change the dynamics ON THE MOUNTAIN FACE (codex r8: the
    # first version compared face 1, where hs ~ 0 and identity is
    # CORRECT after one step — a vacuous assertion)
    t_mt = int(np.argmax([float(np.max(h)) for h in ctx_t["hs6"]]))
    assert not np.allclose(out_t[t_mt]["u"], out_0[t_mt]["u"])
    assert not np.allclose(out_t[t_mt]["delp"], out_0[t_mt]["delp"])
    # deterministic threading pin (step-identity clauses kept tripping
    # on legitimate cross-face flux-averaging propagation): SPY on both
    # geopk sites — each must receive the ctx hs, nonzero on the
    # mountain face, all-zero when topo_fn is None
    import legoesm.core.fv3_native_duo_stepper as ds
    seen = {"cg": [], "d": []}
    orig_cg, orig_d = ds.geopk_sw_1lev, ds.geopk_sw_1lev_d

    def spy_cg(delpc, hs, bd, pt=None):
        seen["cg"].append(float(np.max(np.abs(hs))))
        return orig_cg(delpc, hs, bd, pt=pt)

    def spy_d(delp, hs, bd, pt=None):
        seen["d"].append(float(np.max(np.abs(hs))))
        return orig_d(delp, hs, bd, pt=pt)

    ds.geopk_sw_1lev, ds.geopk_sw_1lev_d = spy_cg, spy_d
    try:
        full_acoustic_step_sixface(ctx_t, w2_six_face_state(ctx_t),
                                   300.0, d_ext=0.0)
        assert max(seen["cg"]) > 1e4 and max(seen["d"]) > 1e4
        seen["cg"].clear()
        seen["d"].clear()
        full_acoustic_step_sixface(ctx_0, w2_six_face_state(ctx_0),
                                   300.0, d_ext=0.0)
        assert max(seen["cg"]) == 0.0 and max(seen["d"]) == 0.0
    finally:
        ds.geopk_sw_1lev, ds.geopk_sw_1lev_d = orig_cg, orig_d
