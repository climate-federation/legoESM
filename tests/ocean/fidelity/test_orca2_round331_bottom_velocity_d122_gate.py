"""Direct structural tests for the round-331 cross-tree D122 gate."""

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round331_bottom_velocity_d122_gate as gate,
)


def test_cross_tree_gate_pins_the_pre_and_post_statement_commits():
    assert gate.BASELINE_COMMIT == "0f463433e04e46878cc22f408a8985ecc202fb62"
    assert gate.CANDIDATE_COMMIT == "0034a5f2a46962e2d4407ea0296d769e1c785fea"
    assert gate.BASELINE_COMMIT != gate.CANDIDATE_COMMIT


def test_cross_tree_gate_names_only_the_cited_statement():
    assert gate.CANDIDATE_ARM == "bottom_tke_uses_stored_kbb_velocity"
    source = gate.Path(gate.__file__).read_text()
    assert "bottom TKE boundary reads stored Kbb velocity" in source
    assert "classify_later_first" in source
    assert '"exact-loss"' in source
