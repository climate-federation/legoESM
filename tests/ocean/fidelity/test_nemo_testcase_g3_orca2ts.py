"""Grid ladder rung G3 (DECISION 117): the certified G2 card and deck with ONE
switch, the initial state (ORCA2's T/S climatology read by &namtsd).

Deck and acquisition-script contracts need no data.  The card tests read
ORCA2's shipped domain and climatology files and skip when absent.  The one
comparison with NEMO (round 2) reads round 1's smoke restart and skips when
absent.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "scripts/validate/ocean_fidelity/testcases"
G3_DIR = CASES / "nemo_testcase_g3_orca2ts"
G2_DIR = CASES / "nemo_testcase_g2_orca2bathy"
INPUTS = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
FILES = {
    "ORCA_R2_zps_domcfg.nc": "7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839",
    "data_1m_potential_temperature_nomask.nc":
        "ca00905c27078305e80130ea6dd07fc575de2ae3257fad38003b329458cf7f7a",
    "data_1m_salinity_nomask.nc":
        "ad648d972f0631bde7e1b598d98470a979b7f2649271fc9cf5ccfca158e31d0c",
}
ORCA2_DECK = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/EXP00/"
                  "namelist_cfg")
needs_inputs = pytest.mark.skipif(
    not all((INPUTS / name).is_file() for name in FILES), reason="no ORCA2 input files")
PRIOR = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round14/"
             "oracle_g3_r14")
TSD = ("namtsd.ln_tsd_dmp", "namtsd.ln_tsd_init", "namtsd.cn_dir", "namtsd.sn_tem", "namtsd.sn_sal")


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _deck(tmp_path, text, name="deck"):
    root = tmp_path / name
    root.mkdir()
    (root / "namelist_cfg").write_text(text)
    (root / "domain_cfg.nc").symlink_to(INPUTS / "ORCA_R2_zps_domcfg.nc")
    for data in list(FILES)[1:]:
        (root / data).symlink_to(INPUTS / data)
    return root


def test_g3_deck_is_g2_plus_orca2s_namtsd_lines():
    g2 = recipe._orca2g1_namelist_values(G2_DIR / "namelist_cfg")
    g3 = recipe._orca2g1_namelist_values(G3_DIR / "namelist_cfg")
    assert {k: v for k, v in g3.items() if k not in TSD} == {
        k: v for k, v in g2.items() if k not in TSD}
    assert (g2["namtsd.ln_tsd_init"], g3["namtsd.ln_tsd_init"]) == (False, True)
    # Decision 120: hand edits on (dtatsd.f90:219), restoring off
    assert (g2["namtsd.ln_tsd_dmp"], g3["namtsd.ln_tsd_dmp"]) == (False, True)
    assert g3["namtra_dmp.ln_tradmp"] is False
    assert sorted(set(g3) - set(g2)) == ["namtsd.cn_dir", "namtsd.sn_sal", "namtsd.sn_tem"]
    assert recipe.read_orca2g1_deck(G3_DIR) == recipe.read_orca2g1_deck(G2_DIR)
    if ORCA2_DECK.is_file():        # the data lines are ORCA2's own
        orca2 = recipe._orca2g1_namelist_values(ORCA2_DECK)
        for name in TSD:   # D120: ln_tsd_dmp is ORCA2's too
            assert "".join(str(g3[name]).split()) == "".join(str(orca2[name]).split())


def test_run_sh_reuses_g2s_binaries_and_pins_the_climatology():
    text = (G3_DIR / "run.sh").read_text()
    assert subprocess.run(["bash", "-n", str(G3_DIR / "run.sh")]).returncode == 0
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r"makenemo|maketools|\bmake\b", code)      # no build
    g2 = (G2_DIR / "run.sh").read_text()
    for pin in ("REF_SHA", "RUN_SHA", "SHIPPED_DOMCFG_SHA"):
        want = re.search(rf"readonly {pin}=([0-9a-f]{{64}})", g2).group(1)
        assert re.search(rf"readonly {pin}=([0-9a-f]{{64}})", text).group(1) == want
    assert FILES["data_1m_potential_temperature_nomask.nc"] in text
    assert FILES["data_1m_salinity_nomask.nc"] in text
    assert "'3,5,7,9,0,0,0,0,0,0'" in code and "'2,4,6,8,10,0,0,0,0,0'" in code
    # after its last entry nitrst is 0: twin c's last restart is step 9, not 10
    assert 'run_one "$RUN_BIN" "$TWIN_C" 10 "$L_ODD" 9' in code
    # admission renders each run's whole deck, restart list included
    assert 'render_deck "$dry/admit_deck" "$steps" "$list"' in code
    assert "gridladder_rounds/round14/oracle_g3_r14" in code
    assert "does not state the T/S read with damping off" in code
    for name, sha in FILES.items():
        if (INPUTS / name).is_file():
            assert hashlib.sha256((INPUTS / name).read_bytes()).hexdigest() == sha


@pytest.fixture(scope="module")
def g3_root(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in FILES):
        pytest.skip("no ORCA2 input files")
    return _deck(tmp_path_factory.mktemp("g3"), (G3_DIR / "namelist_cfg").read_text())


@pytest.fixture(scope="module")
def g3_card(g3_root):
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    return recipe.build_nemo_testcase_card("ORCA2G3-ts", deck_root=g3_root)


@needs_inputs
def test_g3_card_is_the_g2_card_started_from_orca2s_climatology(g3_card, g3_root, tmp_path):
    g2_root = _deck(tmp_path, (G2_DIR / "namelist_cfg").read_text(), "g2")
    g2 = recipe.build_nemo_testcase_card("ORCA2G2-zps", deck_root=g2_root)
    a, b = g3_card.recipe.model_config, g2.recipe.model_config
    # round 3: NEMO's stored r3u(Kmm) in dyn_zdf, selected by G3's card only
    assert [f for f in a._fields if repr(getattr(a, f)) != repr(getattr(b, f))] == [
        "zdf_momentum_stage_r3u_nemo_literal"]
    assert (a.zdf_momentum_stage_r3u_nemo_literal, b.zdf_momentum_stage_r3u_nemo_literal) == (
        True, False)
    assert recipe._orca2g1_selector_failures(g3_card) == []
    assert g3_card.unmeasured_features == recipe.ORCA2G3_UNMEASURED
    with pytest.raises(ValueError, match="not execution-ready"):
        recipe.validate_orca2g1_flat_card_for_execution(g3_card, g3_root)
    s3, s2 = g3_card.recipe.initial_state, g2.recipe.initial_state
    assert np.array_equal(np.asarray(s3.eta.data), np.asarray(s2.eta.data))
    assert np.array_equal(np.asarray(g3_card.recipe.land_mask), np.asarray(g2.recipe.land_mask))
    wet = np.asarray(g3_card.recipe.z_coord.is_active).astype(bool)
    t3 = np.asarray(s3.T.data)
    assert np.all(t3[~wet] == 0.0) and not np.any(t3[wet] == 20.0)
    # Decision 120: ln_tsd_dmp on, so NEMO's dtatsd.f90:219 applies the
    # ORCA_R2 hand alterations: the card is the altered climatology, and the
    # unaltered one differs on exactly the ORCA2 lane's 1,283 T / 720 S cells.
    t_alt, s_alt = recipe.build_orca2_initial_ts(
        INPUTS / "data_1m_potential_temperature_nomask.nc",
        INPUTS / "data_1m_salinity_nomask.nc", wet)
    assert np.array_equal(t3, t_alt) and np.array_equal(np.asarray(s3.S.data), s_alt)
    t_off, s_off = recipe.build_orca2_initial_ts(
        INPUTS / "data_1m_potential_temperature_nomask.nc",
        INPUTS / "data_1m_salinity_nomask.nc", wet, apply_hand_alterations=False)
    assert int((t3 != t_off).sum()) == 1283
    assert int((np.asarray(s3.S.data) != s_off).sum()) == 720


@needs_inputs
def test_g3_refuses_planted_disagreements(g3_card, g3_root, tmp_path):
    state = g3_card.recipe.initial_state
    deep = np.asarray(state.T.data).copy()
    deep[73, 50, 20] += 1.0e-12
    with pytest.raises(ValueError, match="initial_T"):
        recipe.validate_orca2g1_flat_card(g3_card._replace(recipe=g3_card.recipe._replace(
            initial_state=state._replace(T=state.T.replace(data=deep)))), g3_root)
    text = (G3_DIR / "namelist_cfg").read_text()
    plants = {
        "dmp_off": (text.replace("ln_tsd_dmp  = .true.", "ln_tsd_dmp  = .false."), "ln_tsd_dmp"),
        "file": (text.replace("'data_1m_salinity_nomask'", "'sali_ref_clim_monthly'"), "sn_sal"),
        "dt": (text.replace("rn_Dt       =  10800.", "rn_Dt       =   5400."), "rn_dt"),
        "date": (text.replace("nn_date0    =  010101", "nn_date0    =  010115"), "nn_date0"),
        "init_off": (text.replace("ln_tsd_init = .true.", "ln_tsd_init = .false."),
                     "ln_tsd_init"),
    }
    for name, (planted, match) in plants.items():
        assert planted != text, name
        with pytest.raises(ValueError, match=match):
            recipe.orca2g_initial_ts(_deck(tmp_path, planted, name), "ORCA2G3-ts",
                                     np.ones((148, 180, 30), dtype=bool))
    # the G2 rung refuses a deck that reads T/S, and names no data files
    with pytest.raises(ValueError, match="ln_tsd_init"):
        recipe.orca2g_initial_ts(g3_root, "ORCA2G2-zps", np.ones((148, 180, 30), dtype=bool))
    with pytest.raises(ValueError, match="not a grid-ladder rung"):
        recipe.orca2g_initial_ts(g3_root, "ORCA2-zps", np.ones((148, 180, 30), dtype=bool))


def test_run_sh_records_to_nemos_kt11_boundary_and_admits_round_one():
    """Round 2: NEMO stops the 100-step G3 run at kt = 11 (stp_ctl, |V| > 10
    m/s).  The new acquisition runs nothing past kt = 10, admits round 1's
    smoke and stopped run in place before any run, and refuses any stop."""
    code = "\n".join(line for line in (G3_DIR / "run.sh").read_text().splitlines()
                     if not line.lstrip().startswith("#"))
    runs = re.findall(r'^run_one "\$(\w+)" "\$(\w+)" (\d+) "\$(\w+)" (\d+)$', code, re.M)
    assert runs == [("RUN_BIN", "TWIN_A", "10", "L_TEN", "10"),
                    ("RUN_BIN", "TWIN_B", "10", "L_EVEN", "10"),
                    ("RUN_BIN", "TWIN_C", "10", "L_ODD", "9")]
    assert code.index("\nadmit_prior\n") < code.index('\nrun_one "$RUN_BIN"')
    assert "readonly STEPS=10 RECORD_SPGTS=1,2" in code
    assert code.count('--stage-steps "$STEPS"') == 2
    assert "gridladder_rounds/round15/oracle_g3_r15" in code
    # the boundary: kt 11, |V| alone over its limit, the step-10 restart only
    assert '"$(tr -d \' \\n\' <"$REF/time.step")" == 11' in code
    assert "s+0 <= 20 && u+0 <= 10 && v+0 > 10 && n+0 > 0 && x+0 < 100" in code
    assert '[[ -f "$REF/output.abort.nc" ]]' in code and '"$REF/output.abort.nc" )' in code
    # a stop in a new run is refused, never trapped as an unhandled failure
    assert "|| rc=$?" in code and "not the registered kt = 11 boundary" in code


@pytest.fixture(scope="module")
def ladder():
    cases = str(CASES)
    if cases not in sys.path:
        sys.path.insert(0, cases)
    spec = importlib.util.spec_from_file_location(
        "g2_ladder_for_g3", CASES / "nemo_testcase_g2_orca2bathy_ladder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_abort_witness_is_read_only_at_the_step_nemo_stopped(ladder, tmp_path):
    run = tmp_path / "stopped"
    run.mkdir()
    (run / "time.step").write_text("      11\n")
    assert ladder.abort_witness((tmp_path, run), 11) is None     # no abort file yet
    (run / "output.abort.nc").write_bytes(b"")
    assert ladder.abort_witness((tmp_path, run), 11) == run / "output.abort.nc"
    assert ladder.abort_witness((tmp_path, run), 10) is None


@pytest.mark.skip(reason="Decision 120: the as-built (ln_tsd_dmp = F) record is superseded; "
                  "the card refuses it (test_card_refuses_the_as_built_records); G3 is "
                  "re-certified on the round-21 record")
@pytest.mark.skipif(not (PRIOR / "ref_100step/output.abort.nc").is_file(),
                    reason="the operator's G3 round-1 record is not on this machine")
@pytest.mark.slow
def test_g3_chain_against_nemos_steps_2_10_and_its_kt11_stop(ladder):
    """The card's own chain against NEMO's G3 record (round 1): steps 2 (the
    smoke's restart) and 10 (the stopped run's restart) bit for bit on every
    field, level and barotropic history; step 11 (NEMO's output.abort.nc,
    past the record) stops on the same faces with the same bits."""
    out = ladder.run(PRIOR / "smoke_2step", arm="restart_independent", kt_to=11,
                     restart_roots=(PRIOR / "ref_100step",), case="ORCA2G3-ts",
                     allow_dirty=True)
    kt2, kt10, kt11 = out["per_kt"]
    assert (kt2["kt"], kt10["kt"], kt11["kt"]) == (2, 10, 11)
    rows = kt2["rows"] + kt2["history_rows"]
    assert len(rows) == 13 and all(r["n_unequal"] == 0 for r in rows)
    for f, t in kt2["all_levels"].items():
        s = t["summary"]
        assert s["full"]["n_unequal"] == s["partial"]["n_unequal"] == s["dry_n_unequal"] == 0
        assert s["full"]["n_cells"] > 390000 and s["partial"]["n_cells"] > 15000
    # Decision 120's hand-edit cells, counted on the card; none unequal at kt 10
    assert kt2["all_levels"]["T"]["hand_edit_cells"] == 1283
    assert kt2["all_levels"]["S"]["hand_edit_cells"] == 720
    assert all(t["hand_edit_unequal"] == 0 for t in kt10["all_levels"].values())
    rows = kt10["rows"] + kt10["history_rows"]
    assert len(rows) == 13 and all(r["n_unequal"] == 0 for r in rows)
    for t in kt10["all_levels"].values():
        s = t["summary"]
        assert s["full"]["n_unequal"] == s["partial"]["n_unequal"] == s["dry_n_unequal"] == 0
    stp = kt11["stp_ctl_maxima"]
    assert stp["nemo"] == stp["card"]
    assert stp["card"]["|V|"] == {"max": 10.244435336848499, "ji_jj_jk": [22, 84, 27]}
    assert stp["card"]["|ssh|"]["ji_jj_jk"] == [9, 90]


R15 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round15/"
           "oracle_g3_r15")


@pytest.mark.skip(reason="Decision 120: the as-built (ln_tsd_dmp = F) record is superseded; "
                  "the card refuses it (test_card_refuses_the_as_built_records); G3 is "
                  "re-certified on the round-21 record")
@pytest.mark.skipif(not (R15 / "p3_a_10step/oracle_g1step_kt00000003.bin").is_file(),
                    reason="the operator's G3 round-2 record is not on this machine")
@pytest.mark.slow
def test_g3_step3_needs_nemos_stored_r3u_in_dyn_zdf(ladder):
    """Round 3's statement, non-vacuous: deselected, the card's step 3 is
    unequal in exactly one u cell (level 13, full class, last place); selected,
    steps 1-3 are bit-identical on every field and level."""
    kw = dict(arm="independent", kt_to=3, case="ORCA2G3-ts", allow_dirty=True,
              restart_roots=(R15 / "p3_b_10step", R15 / "p3_c_10step"))
    for without in (False, True):
        out = ladder.run(R15 / "p3_a_10step", without_zdf_stage_r3u=without, **kw)
        assert out["zdf_momentum_stage_r3u_nemo_literal"] is (not without)
        kt3 = out["per_kt"][2]
        assert all(r["n_unequal"] == 0 for p in out["per_kt"] for r in p["rows"])
        u = kt3["all_levels"]["u"]
        assert [(r["level"], r["full"]["n_unequal"], r["partial"]["n_unequal"])
                for r in u["levels"] if r["full"]["n_unequal"] or r["partial"]["n_unequal"]] == (
            [(13, 1, 0)] if without else [])
        assert u["summary"]["partial"]["n_unequal"] == u["summary"]["dry_n_unequal"] == 0
        for f in ("v", "T", "S"):
            s = kt3["all_levels"][f]["summary"]
            assert s["full"]["n_unequal"] == s["partial"]["n_unequal"] == s["dry_n_unequal"] == 0


@pytest.mark.skipif(not (R15 / "p3_a_10step/namelist_cfg").is_file(),
                    reason="the operator's G3 round-2 record is not on this machine")
def test_card_refuses_the_as_built_records():
    """Decision 120: the round-14/15 records ran with ln_tsd_dmp = F (no hand
    edits); the G3 card now transcribes only ln_tsd_dmp = T."""
    for root in (R15 / "p3_a_10step", PRIOR / "smoke_2step"):
        if (root / "namelist_cfg").is_file():
            with pytest.raises(ValueError, match="ln_tsd_dmp = True"):
                recipe.orca2g_initial_ts(root, "ORCA2G3-ts", np.ones((148, 180, 30), bool))


@needs_inputs
def test_stored_r3u_refuses_a_card_where_it_would_do_nothing(g3_card):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    r = g3_card.recipe
    cfg = r.model_config._replace(zdf_momentum_nemo_face_thickness=False)
    with pytest.raises(ValueError, match="zdf_momentum_stage_r3u_nemo_literal requires"):
        LatLonCGridOceanModel(r.grid, r.z_coord, cfg).step(r.initial_state, dt=g3_card.dt_s)


D120_NEMO = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/"
                 "acquisition/orca1ice_surface_entry_every_step_a_np2")


@needs_inputs
@pytest.mark.skipif(not (D120_NEMO / "oracle_step_entry_kt00000001.bin").is_file(),
                    reason="the ORCA2 lane's kt = 1 entry record is not on this machine")
def test_d120_initial_state_is_nemos_on_every_wet_cell(g3_root, tmp_path):
    """Decision 120, offline: the G3 card's initial T/S equals NEMO's (the
    ORCA2 lane's kt = 1 entry, hand edits on) on every wet cell; the plant
    turns it red and the no-edits control differs on exactly the edits."""
    import importlib.util
    path = G3_DIR / "d120_initial_state.py"
    spec = importlib.util.spec_from_file_location("d120_initial_state", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    row = mod.run({"ORCA2G3-ts": g3_root})["cases"]["ORCA2G3-ts"]
    assert (row["unequal_T"], row["unequal_S"]) == (0, 0) and row["wet_cells"] == 430552
    assert (row["control_no_edits_unequal_T"], row["control_no_edits_unequal_S"]) == (1283, 720)
    assert row["control_mask_is_edits_T"] and row["control_mask_is_edits_S"]
    assert (row["as_built_g3_level1_unequal_T"], row["as_built_g3_level1_unequal_S"]) == (0, 0)
    assert mod.run({"ORCA2G3-ts": g3_root}, plant=True)["cases"]["ORCA2G3-ts"]["unequal_T"] == 1
