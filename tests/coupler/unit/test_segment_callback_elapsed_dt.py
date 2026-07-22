"""F4 regression: the coupler segment callback must receive the TRUE elapsed
time since the previous coupling call, and must fire on EVERY segment -- not
only on diagnostic-interval boundaries.

With a checkpoint cadence finer than the diagnostic interval,
``compute_segment_length`` = GCD(diag_interval, checkpoint_interval) is a
PROPER divisor of the diag interval, so the atmosphere takes several segments
per diag boundary.  Pre-fix the callback was nested inside the diagnostics
block (model_driver.py:9233-9235) and fired once per diag interval, reporting
a single segment's duration and silently dropping the intervening segments'
coupling.  Post-fix it is hoisted out of the diag block and fires per segment
with ``dt_seg = seg_steps * dt``.

CPU-only tiny compiled run (res=8, nlev=5, 2 days, gray radiation) -- the same
class of unit-test model run already used by
tests/unit/test_transient_forcing_resample.py.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def test_segment_callback_fires_every_segment_with_elapsed_dt(tmp_path):
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.driver.model_driver import ModelDriver

    dt = 600.0
    # diag every 2 days (288 steps), checkpoint every 1 day (144 steps)
    # -> segment_length = GCD(288, 144) = 144 = 1 day, a PROPER divisor of
    # the diag interval, so 2 full segments span the 2-day run.
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=dt, model_type="hydrostatic"),
        output=OutputConfig(diag_days=2, checkpoint_days=1),
        radiation="gray",
        days=2,
    )
    driver = ModelDriver(cfg, output_dir=str(tmp_path))
    driver.setup()

    calls = []
    status = driver.run(
        segment_callback=lambda drv, day, dt_seg: calls.append((day, dt_seg))
    )
    assert status == "COMPLETED", status

    seg_seconds = 144 * dt  # one segment == one day

    # One call per segment (2), NOT a single call at the day-2 diag boundary.
    # Pre-fix: len(calls) == 1.
    assert len(calls) == 2, (
        f"segment callback fired {len(calls)} time(s); expected one per "
        f"segment (2). It is still gated inside the diagnostics block."
    )

    # Each call reports exactly its own segment's elapsed duration.
    for _day, dt_seg in calls:
        assert dt_seg == pytest.approx(seg_seconds)

    # Total elapsed time delivered to the coupler == the full run duration:
    # no segments dropped.  Pre-fix the sum was 86400 (1 day) for a 2-day run.
    total = sum(dt_seg for _day, dt_seg in calls)
    assert total == pytest.approx(cfg.days * 86400.0)
