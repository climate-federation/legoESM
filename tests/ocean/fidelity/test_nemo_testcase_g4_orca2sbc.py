"""Grid ladder rung G4 (DECISION 117): the certified G3 card and deck with ONE
switch, ORCA2's surface boundary condition (no sea ice).

Deck and acquisition-script contracts need no data; the ORCA2-deck and
namelist_ref comparisons skip when the NEMO tree is absent, the card tests
when ORCA2's input files are absent.
"""

from __future__ import annotations

import hashlib
import re
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe


def _g_rungs(table):
    """The G rungs of a shared rung table (the V lane appends its own rows)."""
    return {k: v for k, v in table.items() if k.startswith("ORCA2G")}

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "scripts/validate/ocean_fidelity/testcases"
G4_DIR = CASES / "nemo_testcase_g4_orca2sbc"
G3_DIR = CASES / "nemo_testcase_g3_orca2ts"
INPUTS = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs")
ORCA2_DECK = NEMO / "ORCA2_OMIP_L4/EXP00/namelist_cfg"
NAMELIST_REF = NEMO / "SHARED/namelist_ref"
GROUPS = ("namsbc.", "namsbc_blk.", "namtra_qsr.", "namsbc_ssr.", "namsbc_rnf.")
CARD_FILES = ("ORCA_R2_zps_domcfg.nc", "data_1m_potential_temperature_nomask.nc",
              "data_1m_salinity_nomask.nc")
needs_inputs = pytest.mark.skipif(
    not all((INPUTS / name).is_file() for name in CARD_FILES), reason="no ORCA2 input files")
needs_nemo = pytest.mark.skipif(
    not (ORCA2_DECK.is_file() and NAMELIST_REF.is_file()), reason="no NEMO tree")


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


AS_BUILT = "ad1ba285e37a"      # the decks rounds 14-20 acquired (before Decision 120)


def _as_built(path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{AS_BUILT}:{path.relative_to(REPO)}"],
                          check=True, capture_output=True).stdout


def _values(path):
    return recipe._orca2g1_namelist_values(path)


def _canon(value):
    return "".join(value.split()) if isinstance(value, str) else value


def _deck(tmp_path, text, name="deck"):
    root = tmp_path / name
    root.mkdir()
    (root / "namelist_cfg").write_text(text)
    (root / "domain_cfg.nc").symlink_to(INPUTS / "ORCA_R2_zps_domcfg.nc")
    for data in CARD_FILES[1:]:
        (root / data).symlink_to(INPUTS / data)
    return root


def test_g4_deck_is_g3_plus_the_forcing_groups():
    g3, g4 = _values(G3_DIR / "namelist_cfg"), _values(G4_DIR / "namelist_cfg")
    assert {k: v for k, v in g4.items() if not k.startswith(GROUPS)} == {
        k: v for k, v in g3.items() if not k.startswith(GROUPS)}
    moved = sorted(k for k in g4 if k.startswith("namsbc.") and g4[k] != g3[k])
    assert moved == sorted(recipe._ORCA2G_SBC_SWITCHES)
    assert g4["namsbc.nn_ice"] == 0 and g4["namtsd.ln_tsd_dmp"] is True
    assert recipe.read_orca2g1_deck(G4_DIR) == recipe.read_orca2g1_deck(G3_DIR)
    assert recipe.orca2g_forcing_failures(G4_DIR, "ORCA2G4-sbc") == []
    assert recipe.orca2g_forcing_failures(G3_DIR, "ORCA2G3-ts") == []


@needs_nemo
def test_g4_forcing_resolves_as_orca2s_except_ice():
    """Resolved = namelist_ref overwritten by the deck, group by group."""
    ref, orca2, g4 = _values(NAMELIST_REF), _values(ORCA2_DECK), _values(G4_DIR / "namelist_cfg")
    for group in GROUPS:
        keys = {k for k in ref if k.startswith(group)}
        want = {k: _canon(orca2.get(k, ref[k])) for k in keys}
        got = {k: _canon(g4.get(k, ref[k])) for k in keys}
        if group == "namsbc.":
            assert (want.pop("namsbc.nn_ice"), got.pop("namsbc.nn_ice")) == (2, 0)
        assert got == want, group
        # nothing in the group is stated that namelist_ref does not declare
        assert {k for k in g4 if k.startswith(group)} <= keys
    stated = {k: _canon(v) for k, v in orca2.items()
              if k.startswith(GROUPS[1:])}
    assert stated == recipe._ORCA2G4_SBC_FIELDS
    # groups ORCA2 does not state resolve to namelist_ref in both decks
    for group in ("namsbc_apr.", "namsbc_fwb."):
        assert not any(k.startswith(group) for k in orca2)
        assert not any(k.startswith(group) for k in g4)
    assert ref["namsbc.ln_apr_dyn"] is False and ref["namsbc_fwb.rn_fwb0"] == 0.0


def test_forcing_check_refuses_every_planted_line():
    text = (G4_DIR / "namelist_cfg").read_text()
    plants = {
        "ln_blk": ("ln_blk      = .true.", "ln_blk      = .false."),
        "ln_usr": ("ln_usr      = .false.", "ln_usr      = .true."),
        "ln_traqsr": ("ln_traqsr   = .true.", "ln_traqsr   = .false."),
        "ln_ssr": ("ln_ssr      = .true.", "ln_ssr      = .false."),
        "ln_rnf": ("ln_rnf      = .true.", "ln_rnf      = .false."),
        "nn_fwb": ("nn_fwb      = 2", "nn_fwb      = 0"),
        "orca2_forcing_groups": ("rn_deds     =  -166.67", "rn_deds     =  -100."),
    }
    for name, (old, new) in plants.items():
        assert text.count(old) == 1, name
        planted = text.replace(old, new)
        failed = _failures_for(planted, "ORCA2G4-sbc")
        assert any(name in f for f in failed), (name, failed)
    weights = text.replace("'weights_core2_orca2_bicub'  , 'Uwnd'", "''  , 'Uwnd'")
    assert weights != text and "orca2_forcing_groups" in _failures_for(weights, "ORCA2G4-sbc")
    # G3 reads no forcing data; G4's deck under the G3 rung fails every switch
    failed = _failures_for(text, "ORCA2G3-ts")
    assert set(recipe._ORCA2G_SBC_SWITCHES) <= set(failed)
    assert "namsbc_ssr.nn_sssr" in failed
    assert _failures_for(text, "ORCA2-zps") == ["case"]
    # nn_ice and the calendar of the light/forcing are the reader's own refusals
    for old, new, match in (("nn_ice      = 0", "nn_ice      = 2", "nn_ice"),
                            ("ln_dm2dc    = .false.", "ln_dm2dc    = .true.", "ln_dm2dc")):
        with pytest.raises(ValueError, match=match):
            recipe.read_orca2g1_deck(_tmp_deck(text.replace(old, new)))


def _tmp_deck(text):
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="g4deck_"))
    (root / "namelist_cfg").write_text(text)
    return root


def _failures_for(text, case):
    return recipe.orca2g_forcing_failures(_tmp_deck(text), case)


def _code(path):
    return "\n".join(line for line in path.read_text().splitlines()
                     if not line.lstrip().startswith("#"))


def test_run_sh_reuses_g1_binaries_and_pins_every_forcing_file():
    run = G4_DIR / "run.sh"
    assert subprocess.run(["bash", "-n", str(run)]).returncode == 0
    code = _code(run)
    assert not re.search(r"makenemo|maketools|\bmake\b", code)      # no build
    g3 = (G3_DIR / "run.sh").read_text()
    for pin in ("REF_SHA", "RUN_SHA", "SHIPPED_DOMCFG_SHA"):
        want = re.search(rf"readonly {pin}=([0-9a-f]{{64}})", g3).group(1)
        assert re.search(rf"readonly {pin}=([0-9a-f]{{64}})", code).group(1) == want
    # round 17 acquired the as-built G3 deck (before Decision 120)
    assert re.search(r"readonly G3_DECK_SHA=([0-9a-f]{64})", code).group(1) == hashlib.sha256(
        _as_built(G3_DIR / "namelist_cfg")).hexdigest()
    pinned = dict(line.strip().split(":") for line in re.search(
        r"readonly PINNED_FILES=\(\n(.*?)\n\)", code, re.S).group(1).splitlines())
    # every file the deck names (forcing, weights, chlorophyll, runoff), the
    # T/S climatology, and namelist_ref's sss_data; never the unread 'runoffs'
    named = set(re.findall(r"'([\w.]+)'\s*,\s*-?\d", (G4_DIR / "namelist_cfg").read_text()))
    named |= set(re.findall(r"'(weights_[\w]+)'", (G4_DIR / "namelist_cfg").read_text()))
    assert {n + ".nc" for n in named} - {"runoffs.nc"} | {"sss_data.nc"} == set(pinned)
    for name, sha in pinned.items():
        if (INPUTS / name).is_file():
            assert hashlib.sha256((INPUTS / name).read_bytes()).hexdigest() == sha, name
    want = re.search(r"readonly WANT_NAMSBC='([^']+)'", code).group(1)
    block, inside = [], False
    for raw in (G4_DIR / "namelist_cfg").read_text().splitlines():
        line = raw.split("!", 1)[0].strip()
        inside = inside or line.split(" ")[0] == "&namsbc"
        if inside and line:
            block.append(line.replace(" ", ""))
        if inside and line == "/":
            break
    assert "|".join(block) == want


def test_run_sh_takes_the_record_length_from_nemos_stop():
    code = _code(G4_DIR / "run.sh")
    # smoke, then the reference (NEMO stops it), then three twins at its length
    runs = re.findall(r'^run_one "\$(\w+)" "\$(\w+)" (\S+) "\$\(?(\w+)', code, re.M)
    assert runs == [("REF_BIN", "SMOKE", "2", "L_SMOKE"), ("REF_BIN", "REF", "100", "L_REF"),
                    ("RUN_BIN", "TWIN_A", '"$steps"', "list_a"),
                    ("RUN_BIN", "TWIN_B", '"$steps"', "list_b"),
                    ("RUN_BIN", "TWIN_C", '"$steps"', "list_c")]
    assert "s=$(( c < 10 ? c : 10 )); s=$(( s - s % 2 ))" in code
    assert "stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s" in code
    assert "readonly L_REF='2,4,6,8,10,12,14,16,18,20'" in code
    assert "gridladder_rounds/round17/oracle_g4_r17" in code
    lists = subprocess.run(
        ["bash", "-c", code[code.index("list_a()"):code.index("render_deck \"$dry/full\"")]
         + "\nfor s in 2 4 10; do list_a $s; echo; list_b $s; echo; list_c $s; echo; done"],
        capture_output=True, text=True, check=True).stdout.split()
    assert lists == ["2,0,0,0,0,0,0,0,0,0", "2,0,0,0,0,0,0,0,0,0", "0,0,0,0,0,0,0,0,0,0",
                     "4,0,0,0,0,0,0,0,0,0", "2,4,0,0,0,0,0,0,0,0", "3,0,0,0,0,0,0,0,0,0",
                     "10,0,0,0,0,0,0,0,0,0", "2,4,6,8,10,0,0,0,0,0", "3,5,7,9,0,0,0,0,0,0"]


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in CARD_FILES):
        pytest.skip("no ORCA2 input files")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    tmp = tmp_path_factory.mktemp("g4")
    g4_root = _deck(tmp, (G4_DIR / "namelist_cfg").read_text(), "g4")
    g3_root = _deck(tmp, (G3_DIR / "namelist_cfg").read_text(), "g3")
    return (recipe.build_nemo_testcase_card("ORCA2G4-sbc", deck_root=g4_root), g4_root,
            recipe.build_nemo_testcase_card("ORCA2G3-ts", deck_root=g3_root))


@needs_inputs
def test_g4_card_is_the_g3_card_under_orca2s_forcing(cards):
    g4, root, g3 = cards
    a, b = g4.recipe.model_config, g3.recipe.model_config
    # rounds 23/25: NEMO's surface-stress and tracer-update statements,
    # G4's card only
    stress = ["barotropic_wind_nemo_stored_depth", "surface_stress_nemo_fold_vector_sign",
              "zdf_surface_stress_nemo_face_thickness"]
    tracer_update = "rk3_stage_tracer_written_order"
    stage3_heat = "rk3_stage3_heat_nemo_written"     # round 28
    stage3_kmm = "rk3_stage3_tracer_nemo_stored_thickness"     # round 30
    assert sorted(f for f in a._fields if repr(getattr(a, f)) != repr(getattr(b, f))) == sorted(
        ["physics", tracer_update, stage3_heat, stage3_kmm, *stress])
    assert all(getattr(a, f) is True and getattr(b, f) is False for f in stress)
    assert getattr(a, tracer_update) is True and getattr(b, tracer_update) is False
    assert getattr(a, stage3_heat) is True and getattr(b, stage3_heat) is False
    assert getattr(a, stage3_kmm) is True and getattr(b, stage3_kmm) is False
    assert [c for c, on in _g_rungs(recipe._ORCA2G_WIND_STORED_DEPTH).items() if on] == [
        "ORCA2G4-sbc", "ORCA2G5-ice", "ORCA2G6-vec", "ORCA2G7-drg", "ORCA2G8-ldf"]
    assert [c for c, on in _g_rungs(recipe._ORCA2G_TRA_STAGE_ORDER).items() if on] == [
        "ORCA2G4-sbc", "ORCA2G5-ice", "ORCA2G6-vec", "ORCA2G7-drg", "ORCA2G8-ldf"]
    from legoesm.ocean.state import LatLonCGridOceanConfig
    assert all(LatLonCGridOceanConfig._field_defaults[f] is False
               for f in [*stress, tracer_update, stage3_heat, stage3_kmm])
    assert [f for f in a.physics._fields
            if repr(getattr(a.physics, f)) != repr(getattr(b.physics, f))] == [
        "shortwave_penetration"]
    assert b.physics.shortwave_penetration is None
    qsr = a.physics.shortwave_penetration
    assert (qsr.scheme, qsr.rgb_ir_fraction, qsr.rgb_ir_extinction_m, qsr.rgb_chl_profile,
            qsr.nemo_time_step_s) == ("nemo_qsr_rgb", 0.58, 0.35, "morel_berthon", 10800.0)
    assert (g4.surface_boundary_condition, g4.surface_input_operator) == (
        "ncar_core_sbcblk", "nemo_fld_read")
    assert (g3.surface_boundary_condition, g3.surface_input_operator) == ("none", "none")
    assert recipe._orca2g1_selector_failures(g4) == []
    assert g4.unmeasured_features == recipe.ORCA2G4_UNMEASURED
    with pytest.raises(ValueError, match="not execution-ready"):
        recipe.validate_orca2g1_flat_card_for_execution(g4, root)
    for name in ("T", "S", "eta"):
        assert np.array_equal(np.asarray(getattr(g4.recipe.initial_state, name).data),
                              np.asarray(getattr(g3.recipe.initial_state, name).data))
    assert (g4.dt_s, g4.n_steps, g4.icebergs_enabled) == (g3.dt_s, g3.n_steps, False)


@needs_inputs
def test_g4_card_refuses_a_g3_selection(cards):
    g4, root, g3 = cards
    with pytest.raises(ValueError, match="surface_forcing"):
        recipe.validate_orca2g1_flat_card(g4._replace(
            surface_boundary_condition="none", surface_input_operator="none"), root)
    no_light = g4.recipe.model_config._replace(physics=g4.recipe.model_config.physics._replace(
        shortwave_penetration=None))
    with pytest.raises(ValueError, match="surface_forcing"):
        recipe.validate_orca2g1_flat_card(
            g4._replace(recipe=g4.recipe._replace(model_config=no_light)), root)
    with pytest.raises(ValueError, match="ln_blk"):
        recipe.validate_orca2g1_flat_card(g4._replace(case="ORCA2G3-ts"), root)


@pytest.mark.parametrize("deck_dir", [G3_DIR, G4_DIR], ids=["g3", "g4"])
@pytest.mark.parametrize("name", sorted(recipe._ORCA2G_SBC_SWITCHES))
def test_reader_refuses_a_mixed_forcing_switch_set(deck_dir, name):
    key = name.split(".", 1)[1]
    text = (deck_dir / "namelist_cfg").read_text()
    lines = [line for line in text.splitlines() if line.split("=")[0].strip() == key]
    assert len(lines) == 1
    value = lines[0].split("=", 1)[1].split("!")[0].strip()
    flipped = {".true.": ".false.", ".false.": ".true.", "0": "2", "2": "0"}[value]
    planted = text.replace(lines[0], lines[0].replace(value, flipped, 1))
    with pytest.raises(ValueError, match="namsbc"):
        recipe.read_orca2g1_deck(_tmp_deck(planted))


def test_kt11_witness_reads_the_bulk_surface_mean_nemo_reads(tmp_path):
    """sst_m at kt = 11 = (restart-10 partial sum + entering level-1 T) * 0.5
    (sbcssm.f90, nn_fsbc = 2); any other restart is refused."""
    import importlib.util

    import netCDF4
    path = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
            "testcases/nemo_testcase_g4_orca2sbc/kt11_forcing_witness.py")
    spec = importlib.util.spec_from_file_location("g4_kt11_witness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.RECORD_KT11["wndi"] == 5 and mod.RECORD_KT11["qsr_down"] == 1

    def restart(kt, fsbc):
        name = tmp_path / f"r{kt}_{fsbc}.nc"
        with netCDF4.Dataset(name, "w") as ds:
            ds.createDimension("t", 1), ds.createDimension("z", 2)
            ds.createDimension("y", 1), ds.createDimension("x", 2)
            ds.createVariable("kt", "f8")[:] = kt
            ds.createVariable("nn_fsbc", "f8")[:] = fsbc
            ds.createVariable("sst_m", "f8", ("t", "y", "x"))[:] = [[[1.0, 3.0]]]
            ds.createVariable("tn", "f8", ("t", "z", "y", "x"))[:] = [[[[2.0, 0.25]], [[9.0, 9.0]]]]
        return name

    assert mod.sst_m_kt11(restart(10, 2)).tolist() == [[1.5, 1.625]]
    for kt, fsbc in ((9, 2), (10, 1)):
        with pytest.raises(SystemExit) as refused:
            mod.sst_m_kt11(restart(kt, fsbc))
        assert refused.value.code == 73          # never a fired plant's exit 1


# ---- round 3 (gridladder_r19): the per-step surface forcing acquisition -----
R19_PATCHES = ("stprk3", "dynspg_ts", "sbcblk", "sbcblk_algo_ncar")
NEMO_ROOT = NEMO.parent
G1_DIR = CASES / "nemo_testcase_g1_orca2grid"
OLD = {"stprk3": (NEMO_ROOT / "src/OCE/stprk3.F90", G1_DIR / "stprk3_g1_step_record_stage50.patch"),
       "dynspg_ts": (NEMO_ROOT / "src/OCE/DYN/dynspg_ts.F90", G1_DIR / "dynspg_ts_g1_substep_record.patch"),
       "sbcblk": (NEMO_ROOT / "src/OCE/SBC/sbcblk.F90", None),
       "sbcblk_algo_ncar": (NEMO_ROOT / "src/OCE/SBC/sbcblk_algo_ncar.F90", None)}
WRITER_LINE = re.compile(r"^\+ *(USE g1_step_record|USE sbcssr, ONLY : erp, qrp|!|"
                         r"CALL g1r_frame\(|CALL g1r_w2\( '(\w+)', )")


def _added(name):
    text = (G4_DIR / f"{name}_g4_forcing_record.patch").read_text().splitlines()
    return [l for l in text if l.startswith("+") and not l.startswith("+++")], \
           [l for l in text if l.startswith("-") and not l.startswith("---")]


def test_r19_patches_add_only_writer_calls_with_unique_short_names():
    checker = _load_checker()
    frames_seen = set()
    for name in R19_PATCHES:
        added, removed = _added(name)
        assert removed == [] and added
        frame, names = None, {}
        for line in added:
            assert WRITER_LINE.match(line), line
            hit = re.search(r"CALL g1r_frame\( '(\w)' \)", line)
            if hit or "ACHAR( IACHAR('A') + jit - 1 )" in line:
                frame = hit.group(1) if hit else "iteration"
                frames_seen.add(frame)
                names.setdefault(frame, [])
            w = re.search(r"CALL g1r_w2\( '(\w+)'", line)
            if w:
                assert frame is not None and len(w.group(1)) <= 14     # CHARACTER(16) 'X_name'
                names[frame].append(w.group(1))
        for listed in names.values():
            assert len(listed) == len(set(listed))
        for frame, listed in names.items():      # the checker's schema is what is written
            key = "A" if frame == "iteration" else frame
            assert tuple(listed) == checker.FORCING_FIELDS[key], frame
    assert frames_seen == set(checker.FORCING_FRAMES) | set(checker.BULK_FRAMES) | {"iteration"}


def _load_checker():
    import importlib.util
    spec = importlib.util.spec_from_file_location("g1_check_records_r19", G1_DIR / "check_records.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_r19_run_sh_iterations_and_frequency_are_the_resolved_deck():
    text = (G4_DIR / "run_r19.sh").read_text()
    assert re.search(r"readonly BULK_ITERATIONS=5 FSBC=2", text)
    assert _values(G4_DIR / "namelist_cfg")["namsbc.nn_fsbc"] == 2
    assert "namsbc_blk.nn_iter_algo" not in _values(G4_DIR / "namelist_cfg")
    if NAMELIST_REF.is_file():
        assert _values(NAMELIST_REF)["namsbc_blk.nn_iter_algo"] == 5
    # the deck pinned in run_r19.sh is the as-built G4 deck (before Decision 120)
    pinned = re.search(r"G4_DECK_SHA=(\w+)", text).group(1)
    assert hashlib.sha256(_as_built(G4_DIR / "namelist_cfg")).hexdigest() == pinned


@needs_nemo
def test_r19_patches_apply_unfuzzed_on_the_round4_sources(tmp_path):
    for name in R19_PATCHES:
        shipped, old = OLD[name]
        target = tmp_path / f"{name}.F90"
        target.write_bytes(shipped.read_bytes())
        if old is not None:
            subprocess.run(["patch", "-s", "--fuzz=0", str(target)], stdin=old.open(), check=True)
            p5 = NEMO / "ORCA2G1_OMIP_L1_P5/MY_SRC" / f"{name}.F90"
            if p5.is_file():
                assert target.read_bytes() == p5.read_bytes()
        before = target.read_text().splitlines()
        patch = G4_DIR / f"{name}_g4_forcing_record.patch"
        subprocess.run(["patch", "-s", "--fuzz=0", str(target)], stdin=patch.open(), check=True)
        after = target.read_text().splitlines()
        assert len(after) == len(before) + len(_added(name)[0])
        it = iter(after)
        assert all(any(line == other for other in it) for line in before)   # order kept


def test_r19_plant_list_names_every_checker_forcing_plant():
    text = (G4_DIR / "run_r19.sh").read_text()
    for plant in _load_checker().FORCING_PLANTS:
        assert plant in text


# ---- round 4 (gridladder_r20): per-step forcing identity and the NCAR walk ----
R19_TWIN_A = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
                  "round19/oracle_g4_r19/p6_a")
P5_PPSRC = NEMO / "ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo"
# each walk row's cited line must hold the statement whose result it scores
WALK_STATEMENT_TEXT = {
    "ubzu": "Ubzu = MAX( 0.5_wp , U_zu )", "dtv": "ztmp0 = virt_temp(t_zt, q_zt)",
    "stab": "ztmp1 = 0.5_wp + SIGN(0.5_wp,ztmp0)", "cdn": "zCdN = cd_n10_ncar( Ubzu )",
    "ce": "Ce = ce_n10_ncar", "ch": "Ch = ch_n10_ncar", "t_zu": "t_zu = MAX( t_zt",
    "q_zu": "q_zu = MAX( q_zt", "one_on_l": "ztmp0 = One_on_L(", "zeta_u": "zeta_u   = zu*ztmp0",
    "psi_m": "ztmp2 = psi_m_ncar(zeta_u)", "un10": "ztmp0 = MAX( 0.25_wp , UN10_from_CD",
    "it_cdn": "zCdN = cd_n10_ncar(ztmp0)", "sqrt_cdn": "zsqrt_CdN = sqrt(zCdN)",
    "cd": "ztmp1  = 1._wp + zsqrt_CdN/vkarmn", "sqrt_cd": "zsqrt_Cd = SQRT( Cd )",
    "psi_h_term": "psi_h_ncar(zeta_u) ) / vkarmn / zsqrt_CdN", "cd_ratio": "ztmp2 = zsqrt_Cd / zsqrt_CdN",
    "it_stab": "ztmp1 = 0.5_wp + SIGN(0.5_wp,zeta_u)", "chn": "zChN  = 1.e-3_wp",
    "cen": "zCeN  = 1.e-3_wp", "it_ch": "Ch    = MAX( zChN", "it_ce": "Ce    = MAX( zCeN",
    "chain_cd": "Ubzu = MAX( 0.5_wp , U_zu )", "chain_ch": "Ubzu = MAX( 0.5_wp , U_zu )",
    "chain_ce": "Ubzu = MAX( 0.5_wp , U_zu )",
}


def _steps_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("g4_forcing_steps", G4_DIR / "g4_forcing_steps.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not P5_PPSRC.is_dir(), reason="no P5 build")
def test_walk_rows_cite_the_ncar_statement_they_score():
    mod = _steps_module()
    assert set(mod.STATEMENTS) == set(WALK_STATEMENT_TEXT)
    lines = (P5_PPSRC / "sbcblk_algo_ncar.f90").read_text().splitlines()
    for name, cite in mod.STATEMENTS.items():
        first = int(cite.rsplit(":", 1)[1].split("-")[0])
        assert WALK_STATEMENT_TEXT[name] in lines[first - 1], (name, cite, lines[first - 1])


def test_forcing_steps_helpers(tmp_path):
    mod = _steps_module()
    header = dict(zip(mod.cr.HEADER_NAMES, (
        1, 4, 1, 2, 3, 4, 7, 6, 2, 2, 3, 5, 4, 64)))
    full = np.arange(6 * 7, dtype=float).reshape(6, 7)
    assert mod.inner(full, header).shape == (2, 4)
    assert mod.inner(full[2:4, 1:5], header).shape == (2, 4)

    # Header plants: the reader must refuse a filename/header step mismatch
    # and impossible inner bounds before any field can be consumed.
    path = mod.cr.step_path(tmp_path, 4)
    blob = mod.cr.MAGIC + struct.pack("<14i", *header.values())
    path.write_bytes(blob)
    read_header, fields = mod.frames(tmp_path, 4)
    assert read_header == header and fields == {}
    planted_step = bytearray(blob)
    struct.pack_into("<i", planted_step, 20, 5)
    path.write_bytes(planted_step)
    with pytest.raises(SystemExit) as refused:
        mod.frames(tmp_path, 4)
    assert refused.value.code == 73
    planted_bounds = bytearray(blob)
    struct.pack_into("<i", planted_bounds, 52, 0)
    path.write_bytes(planted_bounds)
    with pytest.raises(SystemExit) as refused:
        mod.frames(tmp_path, 4)
    assert refused.value.code == 73

    log = "\n".join(f"   fld_read: var {v} kt =        3 (0.3 days), record:   0002 (x)"
                    for _, _, v, _ in mod.o1.FIELD_SPECS)
    assert set(mod.records(log, 3).values()) == {1}
    with pytest.raises(SystemExit) as refused:
        mod.records(log, 5)
    assert refused.value.code == 73
    wet = np.ones((1, 3), bool)
    nemo, card = np.array([[1.0, 2.0, 4.0]]), np.array([[1.0, 2.0, 5.0]])
    out = mod.split(card, nemo, wet, np.array([[0.0, 10.0, 30.0]]))
    assert [b["unequal"] for b in out.values()] == [0, 0, 1] and out["20..infK"]["max_rel"] == 0.25
    # sbcmod.f90:557: 0.5 (tau_i + tau_i+1)(2 - umask) max(tmask_i, tmask_i+1)
    tau, t = np.array([[2.0, 4.0, 0.0]]), np.array([[1.0, 1.0, 0.0]])
    u = np.array([[1.0, 0.0, 0.0]])
    assert mod.t_to_uv(tau, t, u, u, True)[0, :2].tolist() == [3.0, 4.0]


@pytest.mark.skipif(not (R19_TWIN_A / "oracle_g1step_kt00000001.bin").is_file(),
                    reason="no round-19 record")
def test_walk_on_the_record_names_the_neutral_drag_first():
    out = _steps_module().walk(R19_TWIN_A, 1)["steps"][1]
    assert out["first_unequal"] == "cdn"
    assert out["rows"]["ubzu"]["unequal_cells"] == 0 and out["rows"]["dtv"]["unequal_cells"] == 0


def test_forcing_steps_uv_faces_and_program_checks(tmp_path):
    mod = _steps_module()
    # dry T here, wet T east, umask 0: NEMO's face carries the wet cell's full stress
    tau, t, u = np.array([[0.0, 4.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]), np.zeros((1, 3))
    assert mod.t_to_uv(tau, t, u, u, True)[0, 0] == 4.0
    log = ("halo width (applies to both rows and columns)       nn_hls =            2\n"
           "frequency update of sbc (and ice)             nn_fsbc       =            2\n"
           "Air temperature and humidity reference height (m)   rn_zqt       =    10.000000000000000     \n"
           "Wind vector reference height (m)                    rn_zu        =    10.000000000000000     \n")
    (tmp_path / "namelist_cfg").write_text("&namsbc_blk\n/\n")
    (tmp_path / "namelist_ref").write_text("   ln_cdgw     = .false.\n")
    fields = {f"{c}_cd": 0 for c in mod.ITER} | {"n_cd": 0, "s_utau": 0}
    (tmp_path / "ocean.output").write_text(log)
    mod.require_program(tmp_path, fields)
    for bad_log, bad_fields in ((log.replace("=            2\nAir", "=            3\nAir"), fields),
                                (log.replace("rn_zu        =    10.", "rn_zu        =    2."), fields),
                                (log, {k: v for k, v in fields.items() if k != "E_cd"})):
        (tmp_path / "ocean.output").write_text(bad_log)
        with pytest.raises(SystemExit) as refused:
            mod.require_program(tmp_path, bad_fields)
        assert refused.value.code == 73


def test_r21_run_sh_reacquires_g3_and_g4_on_the_decision120_decks():
    """Round 21: one script, three admitted binaries (no build), the decks
    = the as-built decks with ln_tsd_dmp moved, the round-19 input list."""
    code = (G4_DIR / "run_r21.sh").read_text()
    commands = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"makenemo|maketools|\bmake\b", commands)
    g3, r19 = (G3_DIR / "run.sh").read_text(), (G4_DIR / "run_r19.sh").read_text()
    for mine, theirs, src in (("REF_SHA", "REF_SHA", g3), ("P5_SHA", "RUN_SHA", g3),
                              ("G4_AS_BUILT_SHA", "G4_DECK_SHA", r19)):
        want = re.search(rf"readonly {theirs}=([0-9a-f]{{64}})", src).group(1)
        assert re.search(rf"readonly {mine}=([0-9a-f]{{64}})", code).group(1) == want
    files = lambda text, name: re.search(rf"readonly {name}=\((.*?)\)", text, re.S).group(1).split()
    assert files(code, "G4_FILES") == files(r19, "INPUT_FILES")
    for deck in (G3_DIR, G4_DIR):
        old, new = _values_text(_as_built(deck / "namelist_cfg")), _values(deck / "namelist_cfg")
        assert {k for k in new if new[k] != old.get(k)} | {k for k in old if k not in new} == {
            "namtsd.ln_tsd_dmp"}
        assert (old["namtsd.ln_tsd_dmp"], new["namtsd.ln_tsd_dmp"]) == (False, True)
        assert new["namtra_dmp.ln_tradmp"] is False
    assert re.search(r"readonly AS_BUILT=(\w+)", code).group(1) == AS_BUILT
    # NEMO's own facts the script relies on: no step-1 restart on this deck
    assert "sbcmod.f90:341-349" in code and "restart.f90:119" in code
    assert subprocess.run(["bash", "-n", str(G4_DIR / "run_r21.sh")]).returncode == 0


def _values_text(data: bytes):
    import tempfile
    with tempfile.NamedTemporaryFile("wb", suffix="namelist_cfg") as handle:
        handle.write(data)
        handle.flush()
        return _values(Path(handle.name))


R21_P6_A = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/"
                "oracle_g4_r21/p6_a")


@needs_inputs
@pytest.mark.slow
@pytest.mark.skipif(not (R21_P6_A / "oracle_g1step_kt00000001.bin").is_file(),
                    reason="no round-21 record")
def test_nemo_supplied_forcing_stage_one_and_three(tmp_path):
    """Decision 123 on the Decision-120 record (round 21), kt = 1, NEMO's own
    surface fields through the ORCA2 lane's route.  With NEMO's three
    surface-stress statements (round 23) and source-ordered tracer update
    (round 25), every value at stage 1 is NEMO's; u differs only by signed
    zeros.  Stage 3's stress response is gone.  Each deselected arm restores
    its registered gap."""
    root = tmp_path / "g4"
    root.mkdir()
    (root / "namelist_cfg").write_text((G4_DIR / "namelist_cfg").read_text())
    for f in R21_P6_A.glob("*.nc"):
        if not re.search(r"restart|300h|mesh_mask|layout", f.name):
            (root / f.name).symlink_to(f)
    mod = _steps_module()
    fed = mod.route(R21_P6_A, root, stages="13")["stages"]
    one, three = fed["1"], fed["3"]
    assert [one[n]["unequal"] for n in ("ssh", "uu_b", "vv_b", "v")] == [0, 0, 0, 0]
    assert one["u"]["unequal_values"] == 0
    assert all(one[n]["unequal"] == 0 for n in ("T", "S"))
    assert three["ssh"]["unequal"] == 0 and max(three[n]["max_abs"] for n in ("u", "v")) < 2e-15
    assert max(three[n]["max_abs"] for n in ("T", "S")) < 1e-14   # one last place (T ~ 20 C)
    before = mod.route(R21_P6_A, root, stages="1", without_stress_options=True)["stages"]["1"]
    assert before["ssh"]["unequal"] > 2000 and before["uu_b"]["max_abs"] > 1e-10
    tracer_before = mod.route(
        R21_P6_A, root, stages="1", without_tracer_update_option=True)["stages"]["1"]
    assert (tracer_before["T"]["unequal"], tracer_before["S"]["unequal"]) == (58, 45)
    # round 28: NEMO's written stage-3 heat Krhs and solve RHS take stage-3 T
    # from 489 to 200 unequal cells; deselected, the 489 return (the control).
    # round 30: NEMO's stored stage-3 Kmm thickness through the whole solve
    # takes stage-3 T from 200 to 0; deselected, the 200 return (the control).
    assert all(three[n]["unequal_values"] == 0 for n in three)
    kmm_before = mod.route(R21_P6_A, root, stages="3", without_stage3_kmm_option=True)["stages"]["3"]
    assert kmm_before["T"]["unequal_values"] == 200 and kmm_before["S"]["unequal"] == 0
    # round 28's control: heat (and with it the Kmm option) deselected
    heat_before = mod.route(R21_P6_A, root, stages="3", without_stage3_heat_option=True)["stages"]["3"]
    assert heat_before["T"]["unequal_values"] == 489


@needs_inputs
@pytest.mark.slow
@pytest.mark.skipif(not (R19_TWIN_A / "oracle_g1step_kt00000001.bin").is_file(),
                    reason="no round-19 record")
def test_nemo_supplied_forcing_is_read_at_nemos_stage_one(tmp_path):
    """Decision 123: the round-19 record's kt = 1 surface fields through the
    ORCA2 lane's route.  Zeroed, every wet column's stage-1 state moves (the
    route is consumed); NEMO's own fields leave stage 1 unequal only in the
    sea-level bump's patch, at <= 1e-9 (round 21: P1 REFUTED, open)."""
    root = tmp_path / "g4"
    root.mkdir()
    (root / "namelist_cfg").write_text((G4_DIR / "namelist_cfg").read_text())
    for f in R19_TWIN_A.glob("*.nc"):
        if not re.search(r"restart|300h|mesh_mask|layout", f.name):
            (root / f.name).symlink_to(f)
    mod = _steps_module()
    zero = mod.route(R19_TWIN_A, root, zero=True, stages="1")["stages"]["1"]
    # every wet column (16,433) moves in sea level, T and S; nearly every face in u, v
    assert [zero[n]["unequal"] for n in ("ssh", "T", "S")] == [16433] * 3
    assert all(r["unequal"] > 15000 for r in zero.values())
    # round 21's measurement, the card without round 23's stress statements
    fed = mod.route(R19_TWIN_A, root, stages="1", without_stress_options=True)["stages"]["1"]
    assert all(0 < r["unequal"] < 2500 and r["max_abs"] < 1e-9 for r in fed.values())
