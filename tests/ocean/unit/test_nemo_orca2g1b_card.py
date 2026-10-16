"""The ORCA2G1b card (grid ladder rung G1b): G1 plus ORCA2's vector form.

* FILE-FREE: the committed G1b deck is G1's with exactly the &namdyn_adv
  lines ORCA2 states; the reader refuses each moved or misplaced switch.
* ON G1'S ADMITTED DOMAIN FILE (skipped off this machine): the G1b card and
  the G1 card differ in the momentum-advection selectors and the three
  flux-form-only options NEMO's vector deck does not execute, and in
  nothing else; each card refuses the other rung's deck.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

ROOT = Path(__file__).parents[3]
CASES = ROOT / "scripts/validate/ocean_fidelity/testcases"
G1B = CASES / "nemo_testcase_g1b_orca2vec/namelist_cfg"
G1 = CASES / "nemo_testcase_g1_orca2grid/namelist_cfg"
ORCA2 = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg")
DOMAIN = Path(os.environ.get(
    "LEGOESM_ORCA2G1_DOMAIN",
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round4/"
    "oracle_g1_r4/domaincfg/domain_cfg.nc"))


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(before)


def _deck(tmp_path: Path, source: Path, old: str | None = None, new: str = "") -> Path:
    text = source.read_text()
    if old is not None:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    (tmp_path / "namelist_cfg").write_text(text)
    return tmp_path


def test_g1b_deck_is_g1_plus_the_namdyn_adv_lines():
    a = recipe._orca2g1_namelist_values(G1)
    b = recipe._orca2g1_namelist_values(G1B)
    moved = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert moved == {"namdyn_adv.ln_dynadv_off", "namdyn_adv.ln_dynadv_vec",
                     "namdyn_adv.nn_dynkeg"}
    assert (b["namdyn_adv.ln_dynadv_off"], b["namdyn_adv.ln_dynadv_vec"],
            b["namdyn_adv.nn_dynkeg"]) == (False, True, 0)


@pytest.mark.skipif(not ORCA2.is_file(), reason="ORCA2's deck is not on this machine")
def test_g1b_namdyn_adv_is_orca2s():
    orca2 = recipe._orca2g1_namelist_values(ORCA2)
    b = recipe._orca2g1_namelist_values(G1B)
    assert orca2["namdyn_adv.ln_dynadv_vec"] is True and orca2["namdyn_adv.nn_dynkeg"] == 0
    assert {k: v for k, v in orca2.items() if k.startswith("namdyn_adv.")} == {
        "namdyn_adv.ln_dynadv_vec": b["namdyn_adv.ln_dynadv_vec"],
        "namdyn_adv.nn_dynkeg": b["namdyn_adv.nn_dynkeg"]}
    # the other namelist_ref-defaulted choices ORCA2 leaves alone, G1b states
    assert "namzdf.ln_zad_aimp" not in orca2 and b["namzdf.ln_zad_aimp"] is False
    assert orca2["namdyn_vor.ln_dynvor_een"] is True and b["namdyn_vor.ln_dynvor_een"] is True


def test_g1b_deck_reads(tmp_path):
    deck = recipe.read_orca2g1_deck(_deck(tmp_path, G1B))
    assert deck.rn_Dt_s == 10800.0 and deck.nn_itend == 100


@pytest.mark.parametrize("old,new,match", [
    ("   ln_dynadv_OFF  = .false.", "   ln_dynadv_OFF  = .true.", "ln_dynadv_off"),
    ("      nn_dynkeg   = 0 ", "      nn_dynkeg   = 1 ", "nn_dynkeg"),
    ("      nn_dynkeg   = 0 ", "      !", "does not state namdyn_adv.nn_dynkeg"),
])
def test_g1b_reader_refuses_moved_switches(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        recipe.read_orca2g1_deck(_deck(tmp_path, G1B, old, new))


def test_nn_dynkeg_without_the_vector_form_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nn_dynkeg without the vector form"):
        recipe.read_orca2g1_deck(_deck(
            tmp_path, G1, "   ln_dynadv_vec  = .false.\n",
            "   ln_dynadv_vec  = .false.\n   nn_dynkeg = 0\n"))


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    pytest.importorskip("netCDF4")
    if not DOMAIN.is_file():
        pytest.skip(f"G1's admitted domain file is not on this machine: {DOMAIN}")
    out = {}
    for name, deck in (("g1", G1), ("g1b", G1B)):
        root = tmp_path_factory.mktemp(name)
        (root / "namelist_cfg").write_text(deck.read_text())
        (root / "domain_cfg.nc").symlink_to(DOMAIN)
        out[name] = root
    return out


def test_g1b_card_differs_from_g1_only_by_nemos_vector_branch(roots):
    a = recipe.build_nemo_testcase_card("ORCA2G1-flat", deck_root=roots["g1"])
    b = recipe.build_nemo_testcase_card("ORCA2G1b-vec", deck_root=roots["g1b"])
    ca, cb = a.recipe.model_config, b.recipe.model_config
    moved = {f: (getattr(ca, f), getattr(cb, f)) for f in ca._fields
             if repr(getattr(ca, f)) != repr(getattr(cb, f)) and f != "barotropic"}
    assert moved == {
        "momentum_advection": ("flux_form", "vector_invariant"),
        "vorticity_scheme": ("een_planetary", "een_total"),
        "momentum_flux_scheme": ("none", "upwind"),
        "vertical_momentum_scheme": ("none", "nemo_advective"),
        "rk3_stage1_rhs_nemo_literal": (True, False),   # flux-form only (state.py)
        "een_nemo_relative_vorticity": (False, True),   # dynvor.f90:744-748
        "rk3_vector_rhs_nemo_order": (False, True),     # stp2d.f90:138, :144, :161
    }
    ba, bb = ca.barotropic, cb.barotropic
    moved_bt = {f for f in ba._fields if repr(getattr(ba, f)) != repr(getattr(bb, f))}
    assert moved_bt == {"barotropic_flux_update_nemo_literal",
                        "barotropic_entry_depth_nemo_literal",
                        "barotropic_literal_fold_unit_vector_form"}
    assert bb.barotropic_literal_fold_unit_vector_form is True
    assert (bb.barotropic_flux_update_nemo_literal,
            bb.barotropic_entry_depth_nemo_literal) == (False, False)
    assert cb.ke_gradient_scheme == "c2" and not cb.adaptive_implicit_vertadv
    assert b.unmeasured_features == recipe.ORCA2G1B_UNMEASURED
    with pytest.raises(ValueError, match="not execution-ready"):
        recipe.validate_orca2g1_flat_card_for_execution(b, roots["g1b"])
    # recipe state, geometry and time step: the same objects' values
    assert (a.dt_s, a.n_steps) == (b.dt_s, b.n_steps)
    assert repr(a.recipe.initial_state.eta.data.tolist()) == repr(
        b.recipe.initial_state.eta.data.tolist())


@pytest.mark.parametrize("case,deck", [("ORCA2G1b-vec", "g1"), ("ORCA2G1-flat", "g1b")])
def test_each_rung_refuses_the_other_deck(roots, case, deck):
    with pytest.raises(ValueError, match="ln_dynadv_vec"):
        recipe.build_nemo_testcase_card(case, deck_root=roots[deck])


def test_g1b_selector_plants_are_refused(roots):
    card = recipe.build_nemo_testcase_card("ORCA2G1b-vec", deck_root=roots["g1b"])
    cfg = card.recipe.model_config
    for change in ({"vorticity_scheme": "een_planetary"},
                   {"momentum_advection": "flux_form"},
                   {"rk3_stage1_rhs_nemo_literal": True},
                   {"barotropic": cfg.barotropic._replace(
                       barotropic_flux_update_nemo_literal=True)},
                   {"een_nemo_relative_vorticity": False},
                   {"rk3_vector_rhs_nemo_order": False},
                   {"barotropic": cfg.barotropic._replace(
                       barotropic_literal_fold_unit_vector_form=False)}):
        planted = card._replace(recipe=card.recipe._replace(
            model_config=cfg._replace(**change)))
        with pytest.raises(ValueError, match="disagrees with its deck"):
            recipe.validate_orca2g1_flat_card(planted, roots["g1b"])


def test_vector_rhs_nemo_order_refuses_flux_form(roots):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    card = recipe.build_nemo_testcase_card(
        "ORCA2G1b-vec", deck_root=roots["g1b"])
    cfg = card.recipe.model_config._replace(momentum_advection="flux_form")
    with pytest.raises(ValueError, match=(
            "rk3_vector_rhs_nemo_order=True requires vector-form")):
        LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)


def test_g1b_fold_unit_predicate_admits_vector_form_only_when_selected(roots):
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        nemo_literal_external_mode_active)
    cfg = recipe.build_nemo_testcase_card(
        "ORCA2G1b-vec", deck_root=roots["g1b"]).recipe.model_config
    assert nemo_literal_external_mode_active(cfg)
    off = cfg._replace(barotropic=cfg.barotropic._replace(
        barotropic_literal_fold_unit_vector_form=False))
    assert not nemo_literal_external_mode_active(off)     # plant: the old gate
    g1 = recipe.build_nemo_testcase_card(
        "ORCA2G1-flat", deck_root=roots["g1"]).recipe.model_config
    assert nemo_literal_external_mode_active(g1)          # flux form unchanged


def test_nemo_f_vertex_area_is_e1f_times_e2f(roots):
    import numpy as np
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _nemo_f_vertex_area
    card = recipe.build_nemo_testcase_card("ORCA2G1b-vec", deck_root=roots["g1b"])
    grid, z = card.recipe.grid, card.recipe.z_coord
    raw = z.nemo_een_barotropic
    area = np.asarray(_nemo_f_vertex_area(grid, z))
    e1e2f = np.asarray(raw.e1f) * np.asarray(raw.e2f)
    assert np.array_equal(area[1:, 1:], e1e2f)
    assert np.array_equal(area[1:, 0], e1e2f[:, -1])       # cyclic west column
    assert np.array_equal(area[0], np.asarray(grid.area_q)[0])
    # the grid's own F area is not NEMO's on this mesh (the defect it fixes)
    assert not np.array_equal(np.asarray(grid.area_q)[1:, 1:], e1e2f)
    with pytest.raises(ValueError, match="e1f and e2f"):
        _nemo_f_vertex_area(grid, z._replace(nemo_een_barotropic=None))


def test_g1b_step_routes_the_vector_rhs_order_into_every_stage(roots, monkeypatch):
    """stp2d.f90:138/:144/:161 and stprk3_stg.f90 stages 2-3: the card's
    option reaches every stage's tendencies call; off, none of them."""
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as om
    card = recipe.build_nemo_testcase_card("ORCA2G1b-vec", deck_root=roots["g1b"])
    seen = []
    real = om.LatLonCGridOceanModel.tendencies

    def spy(self, *a, **kw):
        seen.append(kw.get("nemo_operator_association", False))
        return real(self, *a, **kw)

    monkeypatch.setattr(om.LatLonCGridOceanModel, "tendencies", spy)
    cfg = card.recipe.model_config
    for on in (True, False):
        seen.clear()
        m = om.LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                     cfg._replace(rk3_vector_rhs_nemo_order=on))
        m.step(card.recipe.initial_state, dt=card.dt_s)
        assert len(seen) >= 3
        assert all(bool(s) is on for s in seen), seen


def test_g6_raw_vmask_consumers_are_live_under_traced_jit(roots):
    """G6's raw V mask reaches both Krhs and the vector stage update.

    The one-level G1b card is the fast instance of the same vector/fold
    program; only G6 selects ``rk3_vector_raw_vmask``.  The perturbed fold-row
    velocities enter ``model.step`` as dynamic inputs to its production JIT.
    Exact binary64 words pin the completed stage-2 RHS and its raw Kaa update,
    so removing either production consumer changes the asserted last bits.
    """
    import jax.numpy as jnp
    import numpy as np

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    card = recipe.build_nemo_testcase_card(
        "ORCA2G1b-vec", deck_root=roots["g1b"])
    cfg = card.recipe.model_config._replace(rk3_vector_raw_vmask=True)
    state = card.recipe.initial_state
    u = state.u.data.at[-1, :, 0].set(
        jnp.linspace(-2.0e-5, 3.0e-5, state.u.data.shape[1]))
    v = state.v.data.at[-2, :, 0].set(
        jnp.linspace(4.0e-5, -1.0e-5, state.v.data.shape[1]))
    state = state._replace(u=state.u.replace(data=u), v=state.v.replace(data=v))

    def exposed(name):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**{name: True}))
        return np.asarray(model.step(state, dt=card.dt_s).v.data)[-1, 29:33, 0]

    rhs_words = exposed("expose_stage2_momentum_rhs").view(np.uint64)
    raw_words = exposed("expose_stage2_raw_momentum").view(np.uint64)
    assert np.array_equal(rhs_words, np.array([
        0xbdff8bd7c5fbf193, 0xbe08b82c5abcff03,
        0xbe115ba09e6de23e, 0xbe193713aaab926c,
    ], dtype=np.uint64))
    assert np.array_equal(raw_words, np.array([
        0xbec4cb6e7bc1d37e, 0xbed04b673cd01519,
        0xbed6e24c40ddddc5, 0xbee09f0e36c098c5,
    ], dtype=np.uint64))


def test_g6_stage_state_hands_vor_een_the_folded_pivot_v(roots, monkeypatch):
    """dynvor.f90:762-766: np_CRV's pivot-row F vorticity reads vv(Kmm) on
    the pivot V row, which the stage-end lbc_lnk fills with the fold image of
    the row below (lbcnfd.f90 'V').  Under G6's raw V mask the stage state
    handed to the momentum tendency keeps that row; the compact mask zeroes
    it.  Fails if the stage state goes back to the compact mask."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as om

    card = recipe.build_nemo_testcase_card(
        "ORCA2G1b-vec", deck_root=roots["g1b"])
    state = card.recipe.initial_state
    v = state.v.data.at[-2, :, 0].set(
        jnp.linspace(4.0e-5, -1.0e-5, state.v.data.shape[1]))
    state = state._replace(v=state.v.replace(data=v))
    seen = []
    real = om.LatLonCGridOceanModel.tendencies

    def spy(self, st, *a, **kw):
        jax.debug.callback(lambda x: seen.append(np.asarray(x)),
                           st.v.data[..., 0], ordered=True)
        return real(self, st, *a, **kw)

    monkeypatch.setattr(om.LatLonCGridOceanModel, "tendencies", spy)
    fold = card.recipe.grid.fold
    pivots = {}
    for on in (True, False):
        seen.clear()
        cfg = card.recipe.model_config._replace(rk3_vector_raw_vmask=on)
        om.LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                 cfg).step(state, dt=card.dt_s)
        assert len(seen) >= 3
        pivots[on] = [s[-1] for s in seen[1:]]
        if on:
            for s in seen[1:]:
                image = fold.vector_sign_v * s[-2][np.asarray(fold.perm_v)]
                wet = s[-1] != 0.0
                assert wet.any()
                assert np.array_equal(s[-1][wet], image[wet])
    assert not any(p.any() for p in pivots[False])


def test_g6_dyn_zad_v_face_mask_keeps_the_pivot_row(roots, monkeypatch):
    """dynzad.f90:125-126: dyn_zad updates the pivot V row, whose raw vmask is
    wet.  Under G6's raw V mask the vector-form vertical momentum advection
    gets that mask for its V faces; the compact mask is dry there and zeroed
    the increment.  Fails if the ZAD call goes back to the compact mask."""
    import jax
    import numpy as np

    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as om
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe

    card = recipe.build_nemo_testcase_card(
        "ORCA2G1b-vec", deck_root=roots["g1b"])
    seen = []
    real = pe._bc_vertical_momentum_advection

    def spy(du, dv, up, vp, w, hu, hv, um, vm, *a, **kw):
        jax.debug.callback(lambda x: seen.append(np.asarray(x)),
                           vm[-1, :, 0], ordered=True)
        return real(du, dv, up, vp, w, hu, hv, um, vm, *a, **kw)

    monkeypatch.setattr(pe, "_bc_vertical_momentum_advection", spy)
    pivot = {}
    for on in (True, False):
        seen.clear()
        cfg = card.recipe.model_config._replace(rk3_vector_raw_vmask=on)
        om.LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                 cfg).step(card.recipe.initial_state,
                                           dt=card.dt_s)
        assert len(seen) >= 3
        pivot[on] = [float(s.sum()) for s in seen[1:]]
    assert all(p > 0 for p in pivot[True])
    assert not any(pivot[False])


def test_g6_pivot_v_ratio_is_nemos_fold_image(roots, monkeypatch):
    """stprk3_stg.f90:167-169: NEMO passes the stage ratios r3u/r3v through
    lbc_lnk (sign +1), so the pivot V row is the mirrored row below, and both
    div_hor (divhor.f90:125-130 -> ww) and dyn_zad's divisor
    (dynzad.f90:125-126) read that image.  The card's stencil wraps the
    pivot V row's northern neighbour to row 0.  Under G6's raw V mask the
    ZAD divisor's pivot row is the fold image (fails if the stage faces drop
    the exchange), and the wzv builder called directly with the exchange moves
    only the pivot-row ww (fails if the builder drops it).  Whether each wzv
    CALL SITE passes the exchange is pinned by the next test."""
    import jax
    import numpy as np
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as om
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe

    card = recipe.build_nemo_testcase_card(
        "ORCA2G1b-vec", deck_root=roots["g1b"])
    s0 = card.recipe.initial_state
    tm = np.asarray(card.recipe.z_coord.is_active)[..., 0]
    eta = np.random.default_rng(0).uniform(-0.05, 0.05, tm.shape) * tm
    s0 = s0._replace(eta=s0.eta.replace(data=jax.numpy.asarray(eta)))
    perm = np.asarray(card.recipe.grid.fold.perm_v)
    seen = []
    real = pe._bc_vertical_momentum_advection

    def spy(du, dv, up, vp, w, hu, hv, um, vm, *a, **kw):
        jax.debug.callback(lambda *x: seen.append([np.asarray(y) for y in x]),
                           hv[-1, :, 0], hv[-2, :, 0], vm[-1, :, 0], w[-1, :, 1],
                           ordered=True)
        return real(du, dv, up, vp, w, hu, hv, um, vm, *a, **kw)

    monkeypatch.setattr(pe, "_bc_vertical_momentum_advection", spy)
    got = {}
    for on in (True, False):
        seen.clear()
        cfg = card.recipe.model_config._replace(rk3_vector_raw_vmask=on)
        om.LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                 cfg).step(s0, dt=card.dt_s)
        assert len(seen) >= 3
        got[on] = seen[1:]
    for (top, below, wet, _), (top_off, _, _, _) in zip(got[True], got[False]):
        wet = wet > 0
        assert wet.sum() > 0
        np.testing.assert_array_equal(top[wet], below[perm][wet])
        assert (top_off[wet] != below[perm][wet]).any()
    # the wzv call: same operands, only the exchange toggled
    rng = np.random.default_rng(1)
    st = card.recipe.initial_state
    u = jax.numpy.asarray(rng.uniform(-0.1, 0.1, np.shape(st.u.data)))
    v = jax.numpy.asarray(rng.uniform(-0.1, 0.1, np.shape(st.v.data)))
    act = jax.numpy.asarray(card.recipe.z_coord.is_active, dtype=u.dtype)
    out = {}
    for on in (True, False):
        out[on] = pe.nemo_qco_wzv_operands(
            jax.numpy.asarray(eta), jax.numpy.asarray(eta), u, v,
            card.recipe.grid, card.recipe.z_coord,
            jax.numpy.asarray(st.u_mask.data)[..., None] * jax.numpy.ones_like(u),
            jax.numpy.asarray(st.v_mask.data)[..., None] * jax.numpy.ones_like(v),
            act, card.dt_s, eta_after_override=jax.numpy.asarray(eta) * 1.5,
            fold_exchange_ratios=on)
    lv_on, lv_off = np.asarray(out[True][2]), np.asarray(out[False][2])
    w_on, w_off = np.asarray(out[True][0]), np.asarray(out[False][0])
    np.testing.assert_array_equal(lv_on[-1][wet], lv_on[-2][perm][wet])
    assert (lv_off[-1][wet] != lv_on[-1][wet]).any()
    assert (w_on[-1] != w_off[-1]).any()
    np.testing.assert_array_equal(w_on[:-1], w_off[:-1])


def test_g6_every_wzv_call_site_takes_the_fold_exchange(roots, monkeypatch):
    """The three G6 wzv calls -- the tracer (stprk3_stg.f90:297) and momentum
    (:360) solves in the stage transport and the tendency's first call -- each
    carry the pivot-row ww fix (sshwzv.f90 via divhor.f90:125-130).  One step
    of G6 records, per call site, whether the fold exchange reached it; fails
    if any one site drops it."""
    import sys

    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as om
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe

    card = recipe.build_nemo_testcase_card(
        "ORCA2G1b-vec", deck_root=roots["g1b"])
    seen = []
    real = pe.nemo_qco_wzv_operands

    def spy(*a, **kw):
        f = sys._getframe(1)
        if f.f_code.co_name == "_step_impl":
            # ponytail: the step-body call only builds the stored diagnostic w
            # on G6 (no prognostic moves over two steps), so it is not pinned.
            return real(*a, **kw)
        seen.append(((f.f_code.co_name, f.f_lineno),
                     kw.get("fold_exchange_ratios", False)))
        return real(*a, **kw)

    monkeypatch.setattr(om, "nemo_qco_wzv_operands", spy)
    monkeypatch.setattr(pe, "_nemo_qco_zad_operands", spy)
    for on in (True, False):
        seen.clear()
        cfg = card.recipe.model_config._replace(rk3_vector_raw_vmask=on)
        om.LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                 cfg).step(card.recipe.initial_state, dt=card.dt_s)
        sites = {s for s, _ in seen}
        names = sorted(n for n, _ in sites)
        assert names == ["_nemo_ws_stage_transport", "_nemo_ws_stage_transport",
                         "latlon_cgrid_ocean_baroclinic_tendencies"], sites
        assert {flag for _, flag in seen} == {on}, seen
