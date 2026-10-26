"""The category-sum caller probe captures jit outputs, finds planted bit changes and counts all -0 sums."""

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_catsum_caller_census.py"

DRIVER = """
import sys
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from legoesm.ice.si3_orca2_state import fortran_sum_categories
plant = len(sys.argv) > 1
a = jnp.array([[-0.0, -0.0, -0.0], [1.0, 2.0, 3.0], [-0.0, 1.0, -0.0]])
f = jax.jit(lambda x: (fortran_sum_categories(x, axis=-1), x * (3.0 if plant else 2.0)))
f(a)
"""


def _run(tmp_path, mode, out, *extra):
    drv = tmp_path / "driver.py"
    drv.write_text(DRIVER)
    subprocess.run([sys.executable, str(PROBE), "run", "--mode", mode, "--out", str(out), "--", str(drv), *extra],
                   check=True, env={**os.environ, "JAX_ENABLE_X64": "1"})


def test_capture_compare_finds_planted_change_and_nothing_else(tmp_path):
    _run(tmp_path, "capture", tmp_path / "a")
    _run(tmp_path, "capture", tmp_path / "b")
    _run(tmp_path, "capture", tmp_path / "p", "plant")
    sys.path.insert(0, str(PROBE.parent))
    import nemo_si3_catsum_caller_census as probe

    same = probe.compare(tmp_path / "a", tmp_path / "b")
    assert same["calls"] == 1 and same["leaves_compared"] == 2 and same["leaves_changed"] == 0
    planted = probe.compare(tmp_path / "a", tmp_path / "p")
    assert planted["leaves_changed"] == 1 and planted["changed"][0]["leaf"] == "[1]"
    assert planted["changed"][0]["cells_changed"] == 4  # the four nonzeros of x*2 vs x*3


def test_census_counts_only_all_negative_zero_cells(tmp_path):
    _run(tmp_path, "census", tmp_path / "c")
    census = json.loads((tmp_path / "c" / "census.json").read_text())
    (site,) = census
    assert census[site] == [1, 3, 1], census  # one call, three cells, one all -0 row
    assert np.isfinite(census[site]).all()
