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
    assert report["ten_step_restart_mode"] == "explicit-list"
    assert report["ten_step_restart_steps"] == list(range(1, 11))
    assert report["month_restart_mode"] == "periodic-terminal"
    assert report["fields"] == list(gate.FIELDS)
    assert set(report["plants"]) == set(gate.PLANTS[1:])


def test_preflight_names_cover_every_step_and_rank():
    names = set(gate.preflight()["ten_step_restart_names"])
    assert names == {
        f"ORCA2_{step:08d}_restart_{rank:04d}.nc"
        for step in range(1, 11) for rank in (0, 1)
    }


def _canonical_controls():
    return {
        "namrun.nn_itend": "240",
        "namrun.nn_stock": "240",
        "namsbc.nn_ice": "0",
    }


def test_restart_list_is_the_only_ten_step_run_control_delta():
    canonical = _canonical_controls()
    values = {
        **canonical,
        "namrun.nn_itend": "10",
        "namrun.nn_stock": "1",
        "namrun.ln_rst_list": ".true.",
        "namrun.nn_stocklist": ", ".join(str(step) for step in range(1, 11)),
    }
    assert gate._validate_run_controls(
        values, canonical, 10, 1, tuple(range(1, 11)),
    ) == {"restart_mode": "list", "restart_steps": list(range(1, 11))}


def test_rendered_ten_step_deck_passes_restart_control_gate():
    canonical_text = """&namrun
   nn_itend = 240
   nn_stock = 240
/
&namsbc
   nn_ice = 0
/
"""
    rendered = gate.render_run_deck(canonical_text, 10, 1, True)
    assert rendered.count("ln_rst_list") == 1
    assert rendered.count("nn_stocklist") == 1
    assert "nn_stocklist = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10" in rendered


def test_rendered_month_has_no_restart_list_delta():
    canonical_text = """&namrun
   nn_itend = 10
   nn_stock = 1
/
"""
    rendered = gate.render_run_deck(canonical_text, 240, 240, False)
    assert "nn_itend = 240" in rendered
    assert "nn_stock = 240" in rendered
    assert "ln_rst_list" not in rendered


def test_restart_list_control_rejects_missing_step():
    canonical = _canonical_controls()
    values = {
        **canonical,
        "namrun.nn_itend": "10",
        "namrun.nn_stock": "1",
        "namrun.ln_rst_list": ".true.",
        "namrun.nn_stocklist": ", ".join(str(step) for step in range(1, 10)),
    }
    try:
        gate._validate_run_controls(
            values, canonical, 10, 1, tuple(range(1, 11)),
        )
    except gate.GateError as error:
        assert "restart list" in str(error)
    else:
        raise AssertionError("incomplete restart list was accepted")


def test_month_control_refuses_restart_list():
    canonical = _canonical_controls()
    values = {
        **canonical,
        "namrun.ln_rst_list": ".true.",
        "namrun.nn_stocklist": "240",
    }
    try:
        gate._validate_run_controls(values, canonical, 240, 240, None)
    except gate.GateError as error:
        assert "assignment inventory" in str(error)
    else:
        raise AssertionError("restart-list month was accepted")
