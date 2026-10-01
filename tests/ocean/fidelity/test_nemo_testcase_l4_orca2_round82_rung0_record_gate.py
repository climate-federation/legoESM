from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round82_rung0_record_gate as gate,
)


def test_preflight_inventory_is_two_rank_and_self_describing():
    report = gate.preflight()
    assert report["status"] == "PASS_RUNG0_RECORD_PREFLIGHT"
    assert len(report["ten_step_restart_names"]) == 20
    assert report["month_restart_names"] == [
        "ORCA2_00000240_restart_0000.nc",
        "ORCA2_00000240_restart_0001.nc",
    ]
    assert report["fields"] == list(gate.FIELDS)
    assert set(report["plants"]) == set(gate.PLANTS[1:])


def test_preflight_names_cover_every_step_and_rank():
    names = set(gate.preflight()["ten_step_restart_names"])
    assert names == {
        f"ORCA2_{step:08d}_restart_{rank:04d}.nc"
        for step in range(1, 11) for rank in (0, 1)
    }
