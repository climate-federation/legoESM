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
