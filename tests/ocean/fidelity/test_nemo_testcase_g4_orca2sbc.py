"""Grid ladder rung G4 (DECISION 117): the certified G3 card and deck with ONE
switch, ORCA2's surface boundary condition (no sea ice).

Deck and acquisition-script contracts need no data; the ORCA2-deck and
namelist_ref comparisons skip when the NEMO tree is absent, the card tests
when ORCA2's input files are absent.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

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
    assert g4["namsbc.nn_ice"] == 0 and g4["namtsd.ln_tsd_dmp"] is False
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
    assert re.search(r"readonly G3_DECK_SHA=([0-9a-f]{64})", code).group(1) == hashlib.sha256(
        (G3_DIR / "namelist_cfg").read_bytes()).hexdigest()
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
    assert [f for f in a._fields if repr(getattr(a, f)) != repr(getattr(b, f))] == ["physics"]
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
    # the deck pinned in run_r19.sh is the committed G4 deck
    pinned = re.search(r"G4_DECK_SHA=(\w+)", text).group(1)
    assert hashlib.sha256((G4_DIR / "namelist_cfg").read_bytes()).hexdigest() == pinned


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
