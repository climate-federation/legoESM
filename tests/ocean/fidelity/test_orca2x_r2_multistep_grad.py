"""ORCA2X round 2 probe: static/dynamic input split and the CLI contract.

The rung-10 measurement itself needs the round-5 record and ~100 GiB; its
in-run control is the production-loss bit comparison recorded per N.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2x_r2_multistep_grad as m,
)


def test_split_static_keeps_step_numbers_on_host_and_round_trips():
    tree = [{"a": np.ones(3), "kt": 2, "flag": True, "x": 1.5}]
    dyn, rebuild = m.split_static(tree)
    assert len(dyn) == 2  # the array and the float are traced; kt and flag are not
    back = rebuild([d * 2 for d in dyn])
    assert back[0]["kt"] == 2 and back[0]["flag"] is True
    np.testing.assert_array_equal(back[0]["a"], 2 * np.ones(3))
    assert back[0]["x"] == 3.0


@pytest.mark.parametrize("argv", [["--output", "x.json"], ["--n", "11", "--output", "x.json"],
                                  ["--n", "0", "--output", "x.json"], ["--n", "2"]])
def test_cli_refuses_missing_or_out_of_record_n(argv):
    with pytest.raises(SystemExit) as e:
        m.main(argv)
    assert e.value.code == 2


def test_summary_reads_rows_and_compares_no_remat_with_remat(tmp_path):
    import json

    row = {"nonfinite_cotangent_total": 0, "raw_finite": True, "dT0_l2": 1.0, "deta0_l2": 1.0,
           "state_cotangent_l2": 1.0, "dL_dln_rn_ud": -1.0, "loss_rel_vs_production": 0.0,
           "run_s": 1.0, "run_peak_rss_gib": 1.0, "run_rss_growth_gib": 1.0}
    for mode in ("remat", "noremat"):
        (tmp_path / f"{mode}_n2.json").write_text(json.dumps(
            {"rows": {"T": row, "ssh": row}, "program_A": {"compile_s": 1.0}}))
        (tmp_path / f"{mode}_n2.done").write_text("0 10s peak=5GiB stop=none abc\n")
        for name in ("T", "ssh"):
            np.save(tmp_path / f"{mode}_n2_dT0_{name}.npy", np.array([1.0, 2.0]))
    np.save(tmp_path / "noremat_n2_dT0_T.npy", np.array([1.0, 2.5]))
    rows = {(r["mode"], r["loss"]): r for r in m.summary(tmp_path)}
    assert len(rows) == 4 and rows[("remat", "T")]["done"].startswith("0 10s")
    assert rows[("noremat", "T")]["dT0_vs_remat_max_rel"] == 0.25
    assert rows[("noremat", "ssh")]["dT0_vs_remat_max_rel"] == 0.0
