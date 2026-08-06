"""#1311: --budget-ledger on the MPAS lean loop must FAIL LOUDLY.

The per-process ledger accumulates inside the FV compiled segments
(SegmentCarry); the MPAS lean loop never wired it, so the flag used to be
SILENTLY inert — the 2026-07-23 1-yr chains passed ``--budget-ledger`` and
got no ledger (and no conv-pairing gate) with no trace.  Dispatch-hardening
doctrine: a requested diagnostic that cannot run is an error, not a no-op.
"""

from __future__ import annotations

import pytest

from legoesm.driver.config import (
    ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver


def test_budget_ledger_on_mpas_refused(tmp_path):
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=8,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=300.0),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0,
                            budget_ledger=True),
        days=300.0 / 86400.0, dataset="analytical", radiation="gray",
        convection="none", turbulence="none", precision="fp64",
        distributed=False,
    )
    d = ModelDriver(cfg, output_dir=str(tmp_path))
    d.setup()
    with pytest.raises(ValueError, match="budget-ledger.*MPAS|MPAS.*ledger"):
        d.run()
