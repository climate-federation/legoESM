import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires"


def _bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script], check=False, text=True,
        capture_output=True, cwd=ROOT,
    )


def test_deck_selector_derives_names_and_evidence_from_deck():
    deck = TOOL / "deck.sh"
    for name, run_name, suffix in (
        ("shipped", "GYRE_OMIP_L2_P3", "oracle_gyre_r1_stage"),
        ("r4", "GYRE_OMIP_L2_P3_R4", "oracle_gyre_r4_stage"),
        ("r4_smag", "GYRE_OMIP_L2_P3_R4_SMAG",
         "oracle_gyre_r4_smag_stage"),
    ):
        result = _bash(
            f"source {deck}; gyre_hires_select_deck {name} {TOOL}; "
            'printf "%s|%s\\n" "$RUN_NAME" "$DEFAULT_EVIDENCE"'
        )
        assert result.returncode == 0, result.stderr
        round_name = "round2" if name == "r4_smag" else "round1"
        assert result.stdout.strip().split("|") == [
            run_name,
            "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
            f"gyrehr_rounds/{round_name}/{suffix}",
        ]


def test_r4_patch_resolves_exact_operator_physics_and_stage_clock(tmp_path):
    target = tmp_path / "namelist_cfg"
    shutil.copy(TOOL.parent / "configs/gyre_omip_l2_phase3_namelist_cfg", target)
    with (TOOL / "namelist_cfg_r4_kt10.patch").open("rb") as patch:
        result = subprocess.run(
            ["patch", "-s", "--fuzz=0", str(target)], stdin=patch,
            check=False, capture_output=True,
        )
    assert result.returncode == 0, result.stderr.decode()
    checked = _bash(
        f"source {TOOL / 'deck.sh'}; gyre_hires_verify_deck r4 {target}"
    )
    assert checked.returncode == 0, checked.stderr


def test_r4_smag_patch_is_exact_operator_launcher_block(tmp_path):
    target = tmp_path / "namelist_cfg"
    shutil.copy(TOOL.parent / "configs/gyre_omip_l2_phase3_namelist_cfg", target)
    with (TOOL / "namelist_cfg_r4_smag_kt10.patch").open("rb") as patch:
        result = subprocess.run(
            ["patch", "-s", "--fuzz=0", str(target)], stdin=patch,
            check=False, capture_output=True,
        )
    assert result.returncode == 0, result.stderr.decode()
    checked = _bash(
        f"source {TOOL / 'deck.sh'}; gyre_hires_verify_deck r4_smag {target}"
    )
    assert checked.returncode == 0, checked.stderr
    text = target.read_text()
    block = text[text.index("&namdyn_ldf"):text.index("/", text.index("&namdyn_ldf"))]
    assert "rn_Uv" not in block and "rn_Lv" not in block


def test_selector_refuses_unknown_deck():
    result = _bash(
        f"source {TOOL / 'deck.sh'}; gyre_hires_select_deck unknown {TOOL}"
    )
    assert result.returncode == 64
    assert "REFUSE_DECK" in result.stderr


def test_ahm_writer_patch_only_changes_the_two_p3_steppers(tmp_path):
    source = Path(
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
        "cfgs/GYRE_OMIP_L2_P3/MY_SRC"
    )
    if not source.is_dir():
        import pytest
        pytest.skip("local NEMO P3 writer source is unavailable")
    for name in ("stprk3.F90", "stprk3_stg.F90"):
        shutil.copy(source / name, tmp_path / name)
    with (TOOL / "gyre_p3_ahm_writer.patch").open("rb") as patch:
        result = subprocess.run(
            ["patch", "-s", "--fuzz=0", "-d", str(tmp_path), "-p1"], stdin=patch,
            check=False, capture_output=True,
        )
    assert result.returncode == 0, result.stderr.decode()
    assert "NEMO_L4_AHMTF__1" in (tmp_path / "stprk3.F90").read_text()
    stage = (tmp_path / "stprk3_stg.F90").read_text()
    assert "NEMO_L4_DYNLDF_1" in stage
    assert stage.index("l4_uu_before(:,:,:) =") < stage.index("CALL dyn_ldf")
    assert stage.index("CALL dyn_ldf") < stage.index("WRITE(l4_unit)")


def test_ahm_builder_read_only_preflight_applies_pinned_patch():
    result = subprocess.run(
        ["bash", str(TOOL / "build_ahm_writer.sh")],
        check=False, text=True, capture_output=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "PREFLIGHT_OK" in result.stdout
    assert "GYRE_OMIP_L2_P3_AHM" in result.stdout


def test_ahm_operator_recipes_pin_components_registration_inputs_and_one_rank():
    build = (TOOL / "build_ahm_writer.sh").read_text()
    run = (TOOL / "run_ahm.sh").read_text()
    assert '-d "OCE"' in build
    assert "work_cfgs.txt" in build and "printf '%s OCE" in build
    assert "rg -q" not in build and "rg -q" not in run
    assert "GYRE_OMIP_L2_P3/EXP00" in run
    assert "RANKS=${RANKS:-1}" in run
    result = _bash(f"RANKS=8 bash {TOOL / 'run_ahm.sh'}")
    assert result.returncode == 64
    assert "REFUSE_RANKS" in result.stderr


def test_daily_writer_patch_selects_exact_day_boundaries(tmp_path):
    source = Path(
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
        "cfgs/GYRE_OMIP_L2_P3/MY_SRC/stprk3.F90"
    )
    if not source.is_file():
        import pytest
        pytest.skip("local NEMO P3 writer source is unavailable")
    shutil.copy(source, tmp_path / "stprk3.F90")
    with (TOOL / "gyre_p3_daily_writer.patch").open("rb") as patch:
        result = subprocess.run(
            ["patch", "-s", "--fuzz=0", "-d", str(tmp_path), "-p1"],
            stdin=patch, check=False, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    text = (tmp_path / "stprk3.F90").read_text()
    assert "kstp <= nit000 + 720" in text
    assert "MOD( kstp - nit000, 24 ) == 0" in text
    assert "LANE1_STEP_ENTRY_DUMP" in text


def test_daily_duration_patch_is_exact_r4_smag_extension(tmp_path):
    target = tmp_path / "namelist_cfg"
    shutil.copy(TOOL.parent / "configs/gyre_omip_l2_phase3_namelist_cfg", target)
    for patch_name in (
        "namelist_cfg_r4_smag_kt10.patch",
        "namelist_cfg_r4_smag_30d.patch",
    ):
        with (TOOL / patch_name).open("rb") as patch:
            result = subprocess.run(
                ["patch", "-s", "--fuzz=0", str(target)], stdin=patch,
                check=False, capture_output=True)
        assert result.returncode == 0, result.stderr.decode()
    text = target.read_text()
    assert 'cn_exp      =  "GYRE_OMIP_L2_P3_R4_SMAG_30D"' in text
    assert "nn_itend    =     721" in text
    assert "nn_stock    =     721" in text
    assert "nn_write    =      24" in text
    assert "nn_GYRE     =     4" in text
    assert "nn_ahm_ijk_t  = 32" in text


def test_daily_operator_recipes_pin_components_registration_and_one_rank():
    build = (TOOL / "build_daily_writer.sh").read_text()
    run = (TOOL / "run_daily.sh").read_text()
    assert '-d "OCE"' in build
    assert "work_cfgs.txt" in build and "printf '%s OCE" in build
    assert "GYRE_OMIP_L2_P3/EXP00" in run
    assert "RANKS=${RANKS:-1}" in run
    assert "RANKS must be 1" in run
    assert "manifest/oracle_records.sha256" in run
    assert "sha256sum oracle_step_entry_kt*.bin" in run
    assert "REFUSE_RECORD_MANIFEST" in run
    assert "rg " not in build and "rg " not in run
    assert 'CREATE_REFERENCE=GYRE_OMIP_L2_P3' in build
    assert './makenemo -r "$CREATE_REFERENCE"' in build
    assert 'cp -r "$SOURCE_CFG/MY_SRC/." "$TARGET_CFG/MY_SRC/"' in build


def test_daily_writer_builder_read_only_preflight_applies_pinned_patch():
    result = subprocess.run(
        ["bash", str(TOOL / "build_daily_writer.sh")],
        check=False, text=True, capture_output=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "GYRE_OMIP_L2_P3_DAILY" in result.stdout
    assert "PREFLIGHT_OK" in result.stdout


def test_scalar_daily_writer_preflight_uses_scalar_parent_and_arch():
    result = subprocess.run(
        ["bash", str(TOOL / "build_daily_writer.sh"),
         "--math-flavour", "scalar"],
        check=False, text=True, capture_output=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "source=GYRE_OMIP_L2_P3_SMW" in result.stdout
    assert "target=GYRE_OMIP_L2_P3_SMW_DAILY" in result.stdout
    assert "arch=conda-scalarmath" in result.stdout
    assert "PREFLIGHT_OK" in result.stdout


def test_daily_operator_recipe_has_explicit_math_flavour_and_scalar_evidence():
    build = (TOOL / "build_daily_writer.sh").read_text()
    run = (TOOL / "run_daily.sh").read_text()
    assert "--math-flavour vector|scalar" in build
    assert "--math-flavour vector|scalar" in run
    assert "GYRE_OMIP_L2_P3_SMW_DAILY" in build
    assert "GYRE_OMIP_L2_P3_SMW_DAILY" in run
    assert "round7/oracle_gyre_r4_smag_daily_stage_scalar" in run


def test_scalar_step2_writer_widens_only_instrumented_boundaries(tmp_path):
    source = Path(
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
        "cfgs/GYRE_OMIP_L2_P3_SMW/MY_SRC"
    )
    if not source.is_dir():
        import pytest
        pytest.skip("local scalar P3 writer source is unavailable")
    for name in ("stprk3.F90", "stprk3_stg.F90"):
        shutil.copy(source / name, tmp_path / name)
    with (TOOL / "gyre_p3_scalar_step2_writer.patch").open("rb") as patch:
        result = subprocess.run(
            ["patch", "-s", "--fuzz=0", "-d", str(tmp_path), "-p1"],
            stdin=patch, check=False, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    step = (tmp_path / "stprk3.F90").read_text()
    stage = (tmp_path / "stprk3_stg.F90").read_text()
    assert step.count("CALL l1_dump_stage") == 3
    assert step.count("kstp <= nit000 + 1 )   CALL l1_dump_stage") == 3
    assert 'oracle_zdf_entry_kt",I8.8' in step
    assert 'oracle_qsr_stage3_kt",I8.8' in stage
    assert 'oracle_rktracer_stage3_kt",I8.8' in stage
    assert 'oracle_tracer_transport_kt",I8.8' in stage


def test_scalar_step2_acquisition_preflight_pins_one_rank_and_scalar_math():
    script = TOOL / "acquire_scalar_step2.sh"
    result = subprocess.run(
        ["bash", str(script)], check=False, text=True,
        capture_output=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "arch=conda-scalarmath ranks=1" in result.stdout
    assert "PREFLIGHT_OK" in result.stdout
    text = script.read_text()
    assert '-d "OCE"' in text
    assert "work_cfgs.txt" in text
    assert "mpirun -np 1" in text
    assert "rg " not in text
    assert "SOURCE_ZDF_SHA=" in text and "PATCHED_ZDF_SHA=" in text
    assert "trazdf_round35.patch" in text
    assert "trazdf_round38_kt2.patch" in text
    assert "oracle_trazdf_matrix_kt00000001.bin" in text
    assert "oracle_trazdf_matrix_kt00000002.bin" in text
    assert "--finalize-existing" in text
    assert "--writer-identity-only" in text
    assert "--plant-writer-active-word" in text
    assert 'cmp "$SCALAR_TWIN' not in text
