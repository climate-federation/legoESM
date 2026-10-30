"""CAM6 CLUBB -> MG2 relative cloud-water variance (relvar) vs the RUNNING oracles.

Fixture ``oracle_fixtures/clubb_mg2_relvar.npz``, generator
``scripts/validate/oracle/clubb_mg2_relvar/run.sh`` (metadata in the npz):

* Part A: on the Fortran CLUBB-core trajectories of ``clubb_core_cam.npz``
  (regenerated bit for bit), CLUBB's own ``qclvar_out`` captured from the
  Fortran core on each state s_n, and ``relvar`` from it and the Fortran
  ``rcm`` by CAM clubb_intr.F90 cam_cesm2_1_rel_60 :1998, :2416-2425
  (deep_scheme = 'ZM') in ``relvar_oracle.F90``.
* Part B: the six tier-1 warm columns through CAM cam_cesm2_1_rel_60
  ``micro_mg_tend`` (compiled unmodified) with a per-level relvar profile.

fp64.  qclvar: CLUBB core test's field-normalised metric <= 1e-10 (per-element
worst printed).  End-to-end relvar: per-element ``|a-b|/max(|a|,|b|)`` <= 1e-9
(user decision 2026-10-20).  Relvar formula and MG2 outputs: per-element
<= 1e-10 (the MG2 test's metric).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence import clubb as C  # noqa: E402, N812

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests/atmosphere/oracle_fixtures/clubb_mg2_relvar.npz"
TOL = 1.0e-10
TOL_ELEM = 1.0e-9          # end-to-end relvar, per element
CASES = ["bomex", "dycoms2_rf01", "arm", "wangara", "mpace_b", "bomex_dz100",
         "arm_dz160", "bomex_dryut", "arm_dryut"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


core = _load(ROOT / "tests/atmosphere/test_clubb_core_oracle.py", "clubb_core_oracle")
mg = _load(ROOT / "tests/atmosphere/microphysics/unit/test_mg2_warm_rain_oracle.py",
           "mg2_oracle")


@pytest.fixture(scope="module")
def rv():
    return dict(np.load(FIX))


@pytest.fixture(scope="module")
def cfx():
    return dict(np.load(core._FIX))


def _closure(st, Lscale, env, gr):
    """The core's PDF closure on state ``st`` (the call advance_clubb_core
    makes before its advances)."""
    brunt = C.calc_brunt_vaisala_freq_sqd(
        st.thlm, env["exner_zt"], st.rtm, jnp.zeros_like(st.thlm),
        env["p_in_Pa_zt"], jnp.zeros_like(st.thlm), core._CFG.params.bv_efold,
        core._CFG.T0, gr)[0]
    d = C.compute_clubb_diagnostics(
        st.wp2, st.wp3, st.up2, st.vp2, st.thlp2, st.rtp2, st.wpthlp, st.wprtp,
        Lscale, brunt, gr, core._CFG)
    return C.compute_pdf_closure(
        d, st.wp2, st.wp3, st.rtp2, st.thlp2, st.rtpthlp, st.up2, st.vp2,
        st.wprtp, st.wpthlp, st.upwp, st.vpwp, env["wm_zt"], env["wm_zm"],
        st.rtm, st.thlm, st.um, st.vm, env["exner_zt"], env["p_in_Pa_zt"],
        env["p_sfc"], env["thv_ds_zt"], env["thv_ds_zm"], gr, core._CFG)


def _ours(cfx, case):
    """n -> our step LEAVING oracle state s_n (n = 1..N): the qclvar the core
    hands out (pre-advance rcp2), its rcm_out (CAM's, clipped pre-advance rcm), and the
    post-advance closure's rcm/rcp2 (the liquid our host receives)."""
    gr, forcing, env = core._inputs(cfx, case)
    out = {}
    for n in range(1, int(cfx["meta/n_steps"]) + 1):
        st, rcm = core._state(cfx, case, n)
        o = core._step(st, rcm, forcing, env, gr, core._CFG, extra=("rcp2_zt",),
                       diag_extra=("rcp2_zt", "rcm_out"))
        pre = _closure(st, o["Lscale"], env, gr)
        out[n] = {"qclvar": np.asarray(o["diag/rcp2_zt"]),
                  "pre_rcp2": np.asarray(pre["rcp2_zt"]),
                  "rcm_out": np.asarray(o["diag/rcm_out"]),
                  "post_rcp2": np.asarray(o["rcp2_zt"]),
                  "post_rcm": np.asarray(o["rcm"])}
        for k, v in out[n].items():
            assert np.all(np.isfinite(v)), (case, n, k)
    return out


def _errs(ours, ref):
    """(field-normalised, worst per-element |a-b|/max(|a|,|b|)) errors."""
    d = np.abs(ours - ref)
    field = float(d.max() / max(np.abs(ref).max(), 1e-300))
    den = np.maximum(np.abs(ours), np.abs(ref))
    elem = float(np.max(np.where(den > 0, d / np.where(den > 0, den, 1.0), 0.0)))
    return field, elem


def test_fixture_provenance(rv):
    meta = json.loads(str(rv["meta"]))
    assert meta["cam_sha"] == "a03b84b7c4e34f965b115686f22a043b85739e56"
    assert "54403a04390fc4a0d4865357556e77c1ed6ad513" in meta["clubb_oracle"]
    assert [str(c) for c in rv["meta/cases"]] == CASES


@pytest.mark.parametrize("case", CASES)
def test_qclvar_is_the_pre_advance_closure_of_fortrans_qclvar_out(rv, cfx, case):
    """The qclvar our core hands to the host for the step leaving s_n equals
    CLUBB's own captured qclvar_out on s_n.  Field-normalised <= 1e-10 (the
    CLUBB core metric); the per-element worst is printed for the record."""
    field, elem = 0.0, 0.0
    for n, o in _ours(cfx, case).items():
        np.testing.assert_array_equal(o["qclvar"], o["pre_rcp2"])
        f, e = _errs(o["qclvar"][0], rv[f"{case}/s{n}/qclvar_out"])
        field, elem = max(field, f), max(elem, e)
    print(f"QCLVAR {case}: field-norm {field:.3e} per-element {elem:.3e}")
    assert field <= TOL, (case, field)


@pytest.mark.parametrize("case", CASES)
def test_post_advance_pairing_misses_fortrans_qclvar_out(rv, cfx, case):
    """Non-vacuity of the timing: the post-advance closure of the same step
    (the previous pairing) is NOT CLUBB's qclvar_out for that step."""
    if not any(np.any(rv[f"{case}/s{n}/qclvar_out"]) for n in
               range(1, int(rv["meta/n_steps"]) + 1)):
        pytest.skip("clear column: qclvar_out = 0 under either timing")
    field = 0.0
    for n, o in _ours(cfx, case).items():
        field = max(field, _errs(o["post_rcp2"][0], rv[f"{case}/s{n}/qclvar_out"])[0])
    print(f"QCLVAR_POST {case}: field-norm {field:.3e}")
    assert field > 1.0e3 * TOL, (case, field)


def test_qclvar_fixture_is_not_trivial(rv):
    """Cloudy and clear levels both present."""
    pos = [np.count_nonzero(rv[f"{c}/s1/qclvar_out"]) for c in CASES]
    zero = [np.count_nonzero(rv[f"{c}/s1/qclvar_out"] == 0) for c in CASES]
    assert sum(pos) > 50 and sum(zero) > 50, (pos, zero)


def _rel(a, b):
    return mg._rel(a, b)


def test_relvar_formula_matches_oracle(rv, cfx):
    """cam6_relvar on the oracle's own rcm/qclvar_out = clubb_intr's relvar,
    incl. every branch (default, both clips, qclvar cap, rcm == 0, rcp2 == 0)."""
    rcm, rcp2, ref = [], [], []
    for c in CASES:
        for n in range(1, int(rv["meta/n_steps"]) + 1):
            rcm.append(cfx[f"{c}/s{n}/rcm"][0])
            rcp2.append(rv[f"{c}/s{n}/qclvar_out"])
            ref.append(rv[f"{c}/s{n}/relvar"])
    e = rv["edge/in"]
    rcm = np.concatenate(rcm + [e[:, -1]])
    rcp2 = np.concatenate(rcp2 + [rv["edge/rcp2"]])
    ref = np.concatenate(ref + [rv["edge/relvar"]])
    ours = np.asarray(C.cam6_relvar(jnp.asarray(rcm), jnp.asarray(rcp2)))
    assert float(_rel(ours, ref).max()) <= TOL
    edge = rv["edge/relvar"]
    assert np.sum(edge == 10.0) >= 3 and np.any(edge == 1.0e-3)
    assert np.any((edge > 1.0e-3) & (edge < 10.0))
    assert np.any(rv["edge/rcp2"] > 1.0)          # the qclvar cap binds


@pytest.mark.parametrize("case", CASES)
def test_relvar_end_to_end_cam_timing(rv, cfx, case):
    """relvar from our step's pre-advance rcm and qclvar (CAM's rcm_out and
    qclvar_out, clubb_intr.F90:2424) vs the oracle's: max per-element
    ``|a-b|/max(|a|,|b|)`` <= 1e-9 (user decision 2026-10-20).
    qclvar and rcm_out are the core's own diagnostics, the buffers production
    feeds to relvar (user decision 2026-10-21: CAM's rcm_out, not the
    post-advance host liquid; that gap is printed).
    Cells both sides clip to the same bound contribute 0.
    Worst cell, the trace-cloud outlier: wangara step 3, zt level 26, qclvar
    2.5e-16 against a column max of 1.3e-12 (rcp2 cancels ~4 orders there):
    5.2e-12 measured 2026-10-20.  Under the earlier post-advance pairing that
    cell was 5.6e-10, inside this gate; that pairing fails here elsewhere
    (0.72 in bomex), and the qclvar tests above carry the timing check."""
    worst, prod = (0.0, None), 0.0
    for n, o in _ours(cfx, case).items():
        ref = rv[f"{case}/s{n}/relvar"]
        ours = np.asarray(C.cam6_relvar(jnp.asarray(o["rcm_out"]),
                                        jnp.asarray(o["qclvar"])))[0]
        assert np.all(np.isfinite(ours)), (case, n)
        rel = np.asarray(_rel(ours, ref))
        k = int(np.argmax(rel))
        if rel[k] > worst[0]:
            worst = (float(rel[k]), (n, k, float(o["qclvar"][0, k]),
                                     float(np.abs(rv[f"{case}/s{n}/qclvar_out"]).max())))
        # the rejected pairing: the POST-advance rcm the host receives
        host = np.asarray(C.cam6_relvar(jnp.asarray(o["post_rcm"]),
                                        jnp.asarray(o["qclvar"])))[0]
        prod = max(prod, _errs(host, ref)[0])
    print(f"RELVAR {case}: per-element {worst[0]:.3e} at (step, level, qclvar, "
          f"column max qclvar) = {worst[1]} | host-rcm pairing field-norm {prod:.3e}")
    assert worst[0] <= TOL_ELEM, (case, worst)


def _mg2_cfg():
    cfg = mg._production_morrison()
    assert cfg.warm_rain_scheme == "kk2000_cam6"      # the deck's selection
    return cfg._replace(**mg.HARD_LIMIT)


def _mg2_errors(rv, relvar):
    mfx = mg._fx()
    out = mg._run_columns(mfx, _mg2_cfg(), relvar=relvar)
    qb = out.qc_budget
    evap = np.asarray(out.dq_v_dt) + np.asarray(qb["condensation"])
    errs = {
        "prc": _rel(-np.asarray(qb["autoconversion"]), rv["mg2/prctot"]),
        "pra": _rel(-np.asarray(qb["accretion"]), rv["mg2/pratot"]),
        "evap": _rel(evap, rv["mg2/nevapr"]),
        "dq_r": _rel(out.dq_r_dt, rv["mg2/qrtend"]),
        "dN_r": _rel(np.asarray(out.dN_r_dt) / mfx["col_rho"], rv["mg2/nrtend"]),
        "precip": _rel(out.precipitation, rv["mg2/rain_sfc_flux"]),
    }
    return {k: float(v.max()) for k, v in errs.items()}


def test_mg2_with_per_level_relvar_matches_micro_mg_tend(rv):
    mfx = mg._fx()
    relvar = jnp.broadcast_to(jnp.asarray(rv["mg2/relvar"]), mfx["col_T"].shape)
    worst = _mg2_errors(rv, relvar)
    assert max(worst.values()) <= TOL, worst


def test_mg2_per_level_relvar_is_not_the_constant(rv):
    """Non-vacuity: the deck's constant fallback (no per-cell relvar) misses
    the oracle on the same columns."""
    worst = _mg2_errors(rv, None)
    assert max(worst.values()) > 1.0e3 * TOL, worst


def test_gradients_finite():
    rcm = jnp.asarray([0.0, 1e-6, 2e-4, 3e-3, -4e-5])
    rcp2 = jnp.asarray([1e-8, 1e-6, 2e-8, 1e-8, 1e-8])
    g = jax.grad(lambda a, b: jnp.sum(C.cam6_relvar(a, b)), argnums=(0, 1))(rcm, rcp2)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in g)
    assert float(jnp.max(jnp.abs(g[0]))) > 0.0
    mfx = mg._fx()
    cfg = _mg2_cfg()

    def loss(r):
        out = mg._run_columns(mfx, cfg, relvar=jnp.broadcast_to(r, mfx["col_T"].shape))
        return jnp.sum(out.precipitation)

    gr = jax.grad(loss)(jnp.full(mfx["col_T"].shape[1], 2.0))
    assert bool(jnp.all(jnp.isfinite(gr))) and float(jnp.max(jnp.abs(gr))) > 0.0


def test_relvar_fp32_gradients_finite_at_tiny_qclvar():
    """fp32, qclvar = 1e-30: clip-high, in-range, clip-low and rcm = 0 cells
    have finite gradients and the fp64 forward value."""
    rcm64 = np.array([1.0e-6, 3.0e-16, 1.0e-17, 0.0])
    q64 = np.full(4, 1.0e-30)
    f32 = lambda x: jnp.asarray(x, jnp.float32)  # noqa: E731
    g = jax.grad(lambda a, b: jnp.sum(C.cam6_relvar(a, b)), argnums=(0, 1))(
        f32(rcm64), f32(q64))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in g), g
    ours = np.asarray(C.cam6_relvar(f32(rcm64), f32(q64)), np.float64)
    ref = np.asarray(C.cam6_relvar(jnp.asarray(rcm64), jnp.asarray(q64)))
    assert ref[0] == 10.0 and 1.0e-3 < ref[1] < 10.0 and ref[2] == 1.0e-3
    np.testing.assert_allclose(ours, ref, rtol=1e-6)


def test_relvar_forward_matches_the_fortran_formula_on_edges():
    """Every branch incl. negative qclvar and exact clip edges, against
    clubb_intr.F90:2424-2425 evaluated literally."""
    rcm = np.array([0.0, 1e-4, 1e-4, 1e-4, -1e-4, 1e-4, 1e-4, 1e-4])
    q = np.array([1e-8, 0.0, -1e-8, 1e-9, 1e-9, 1e-12, 1e-5, 2.0])
    want = np.where((rcm != 0) & (np.minimum(1.0, q) != 0),
                    np.minimum(10.0, np.maximum(
                        1e-3, rcm ** 2 / np.where(q != 0, np.minimum(1.0, q), 1.0))),
                    10.0)
    np.testing.assert_allclose(np.asarray(C.cam6_relvar(jnp.asarray(rcm), jnp.asarray(q))),
                               want, rtol=1e-15)


def test_relvar_check_grads_in_range():
    from jax.test_util import check_grads
    rcm = jnp.asarray([1.0, -0.3, 0.2])
    rcp2 = jnp.asarray([0.5, 0.2, 0.2])
    assert bool(jnp.all((C.cam6_relvar(rcm, rcp2) > 1e-3)
                        & (C.cam6_relvar(rcm, rcp2) < 10.0)))
    check_grads(C.cam6_relvar, (rcm, rcp2), order=2, modes=("fwd", "rev"), eps=1e-6)


def test_deck_routes_clubb_relvar_to_mg2():
    """The production deck selects the per-cell path: kk2000_cam6 with the
    CLUBB liquid partition (the condition make_physics keys the hand-off on)."""
    from scripts.run.run_amip import build_arg_parser, build_config_from_args

    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = build_arg_parser()
    keys = load_yaml_config(mg.DECK, parser, example_keys="'microphysics'")
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    cfg = build_config_from_args(parser.parse_args(["--config", str(mg.DECK)]))
    assert cfg.morrison_warm_rain_scheme == "kk2000_cam6"
    assert cfg._liquid_partition_resolved()
    assert cfg.cld_macmic_num_steps >= 2


@pytest.mark.parametrize("incloud", [True, False])
def test_microphysics_reads_clubbs_relvar_from_the_same_substep(monkeypatch, incloud):
    """MPAS lane, CAM order: the relvar Morrison receives in macmic sub-step i
    is the one prognostic CLUBB returned in sub-step i."""
    import sys
    sys.path.insert(0, str(ROOT / "tests/unit"))
    from test_physics_macmic import _moist_setup

    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.convection import ConvectionConfig
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.microphysics import integration as mi
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.radiation import RadiationConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence import integration as ti
    written, read = [], []
    real_upd = ti._carry_update_with_cloud_fraction
    real_micro = mi.morrison_microphysics

    def upd(carry_field, carry_val, turb_out):
        written.append(np.asarray(turb_out.relvar))
        return real_upd(carry_field, carry_val, turb_out)

    def micro(*a, relvar=None, **k):
        read.append(None if relvar is None else np.asarray(relvar))
        return real_micro(*a, relvar=relvar, **k)
    monkeypatch.setattr(ti, "_carry_update_with_cloud_fraction", upd)
    monkeypatch.setattr(mi, "morrison_microphysics", micro)
    mesh, sigma, state = _moist_setup()
    # drop the setup's warm bump so the lower column is supersaturated and
    # CLUBB makes liquid (else relvar is relvarmax everywhere)
    sig = jnp.asarray(sigma.sigma_full)[None, :]
    state = state._replace(T=state.T.replace(data=state.T.data - 20.0 * sig ** 8))
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none",
                                  cloud_config=CloudConfig(scheme="cam6_clubb")),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="clubb", liquid_partition=True,
                                    clubb=C.CLUBBConfig(prognostic=True)),
        microphysics=MicrophysicsConfig(
            scheme="morrison",
            morrison=MorrisonConfig(warm_rain_scheme="kk2000_cam6",
                                    warm_rain_incloud=incloud)))
    fn = make_physics(cfg, model_type="mpas", dt=600.0, cld_macmic_num_steps=2)
    ps = init_physics_state(*state.T.data.shape, cfg)
    fn(state, mesh, sigma, phys_state=ps)
    assert len(written) == len(read) == 2
    for w, r in zip(written, read):
        assert r is not None
        np.testing.assert_array_equal(w.reshape(r.shape), r)
    # discriminating: per-cell values, not the pbuf seed or the constant
    assert np.any((read[0] > 1.0e-3) & (read[0] < 10.0)), np.unique(read[0])
    assert not np.all(read[0] == 2.0)
    # sub-steps differ, so equality per sub-step pins the ordering
    assert not np.array_equal(written[0], written[1])
