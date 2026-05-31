"""Regression guard: device_config / runtime cold import must not cycle.

``parallel.device_config`` re-exports XLA-flag helpers from
``runtime.backend`` and defines ``MixedPrecisionPolicy``, while
``runtime.devices`` re-exports ``MixedPrecisionPolicy`` back from
``device_config``.  A top-level ``from legoesm.runtime.backend import ...`` in
device_config closed that cycle, so a *cold* import (fresh interpreter) crashed
with "cannot import name 'MixedPrecisionPolicy' from partially initialized
module".  That broke the MPI reduction wrappers and the first benchmarked
resolution of every grid in the scaling harness.

These tests import the relevant modules in a *subprocess* (so the cycle is
exercised cold, not masked by an already-warmed import graph in the pytest
process).
"""

import subprocess
import sys

import pytest

_COLD_IMPORTS = [
    "import legoesm.parallel.device_config as d; assert d.MixedPrecisionPolicy",
    "import legoesm.parallel.reductions",            # MPI reduction wrappers
    "import legoesm.runtime.devices",                # the back-import site
]


@pytest.mark.parametrize("stmt", _COLD_IMPORTS)
def test_cold_import_no_circular(stmt):
    r = subprocess.run(
        [sys.executable, "-c", stmt],
        capture_output=True, text=True, env={"JAX_PLATFORMS": "cpu", "PATH": ""},
        timeout=120,
    )
    assert r.returncode == 0, (
        f"cold import failed:\n{stmt}\n--- stderr ---\n{r.stderr[-1500:]}"
    )
    assert "circular import" not in r.stderr.lower()
