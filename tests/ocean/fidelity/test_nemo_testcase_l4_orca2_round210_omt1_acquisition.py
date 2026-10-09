from __future__ import annotations

from pathlib import Path


RUNNER = Path(
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round209_omt1_frames_acquisition/run.sh"
)


def test_record_build_bootstraps_from_registered_reference() -> None:
    text = RUNNER.read_text()

    assert "reference_cfg=ORCA2_OMIP_L4\n" in text
    assert './makenemo -r "$reference_cfg" -n "$target_cfg"' in text
    assert './makenemo -r "$source_cfg" -n "$target_cfg"' not in text
    assert 'grep -q "^${candidate} " "$nemo_root/cfgs/ref_cfgs.txt"' in text


def test_pinned_instrumented_tree_is_copied_before_rebuild() -> None:
    text = RUNNER.read_text()
    create = text.index('./makenemo -r "$reference_cfg"')
    copy_exp = text.index('find "$source_root/EXP00"', create)
    copy_src = text.index('find "$source_root/MY_SRC"', copy_exp)
    copy_cpp = text.index('cp "$source_root/cpp_$source_cfg.fcm"', copy_src)
    verify = text.index('sha256sum -c "$source_manifest"', copy_cpp)
    rebuild = text.index('./makenemo -n "$target_cfg"', verify)

    assert create < copy_exp < copy_src < copy_cpp < verify < rebuild


def test_repair_uses_fresh_round210_targets_and_two_plants() -> None:
    text = RUNNER.read_text()

    assert "target_cfg=ORCA2_OMIP_L4_R210OMT1_P3" in text
    assert "/orca2_rounds/round210/acquisition" in text
    assert "bootstrap_work_cfg_plant.log" in text
    assert "source_inventory_plant.log" in text

