"""Regression test: the compiled-segment path re-samples external forcing
at every segment boundary.

CMIP6 transient experiments (historical / SSP) require GHG, ozone, aerosol and
solar forcing to track the calendar (year-to-year, and ozone/aerosol within the
seasonal cycle).  The compiled-segment run loop (``_run_compiled``) previously
gated forcing re-sampling behind ``RAD_UPDATE_STEPS > 1`` — a radiation
*sub-step cadence* knob that defaults to 1, so the branch was dead and ALL
external forcing froze at the ``START_DAY`` precompute (a 1850-2014 historical
run would see 1850 CO2 / January ozone for all 165 years).

This test pins the fix: the compiled loop now calls
``_precompute_external_forcing`` at each segment boundary (``seg_idx > 0``),
independent of the radiation sub-step cadence, so forcing follows the calendar.
It is radiation-scheme agnostic (gray radiation still re-samples — the cost is a
cheap host-side numpy interpolation, not a recompile, because the SegmentForcing
leaves are traced arrays of fixed shape).
"""

import sys
import unittest
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _make_atm_driver(days=2, diag_days=1, resolution=8, nlev=5, dt=600.0):
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.driver.model_driver import ModelDriver

    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=dt, model_type="hydrostatic"),
        output=OutputConfig(diag_days=diag_days),
        radiation="gray",
        days=days,
    )
    return ModelDriver(cfg)


class TestTransientForcingResample(unittest.TestCase):
    """The compiled path must re-sample external forcing per segment."""

    def test_compiled_path_resamples_each_segment(self):
        driver = _make_atm_driver(days=2, diag_days=1)
        driver.setup()

        # Spy on the (host-side) external-forcing precompute, recording the
        # calendar day it is sampled at.  Wrapping the bound method on the
        # instance shadows it for the ``self._precompute_external_forcing``
        # call sites inside the run loop.
        sampled_days = []
        orig = driver._precompute_external_forcing

        def _spy(day, p_s, lat, **kw):
            sampled_days.append(float(day))
            return orig(day, p_s, lat, **kw)

        driver._precompute_external_forcing = _spy
        status = driver.run()
        self.assertEqual(status, "COMPLETED")

        # >= 2 samples: the START_DAY precompute + at least one segment-boundary
        # re-sample.  Pre-fix this was exactly 1 (the loop branch was dead).
        self.assertGreaterEqual(
            len(sampled_days), 2,
            f"external forcing sampled only at days {sampled_days} — the "
            f"compiled loop is not re-sampling per segment",
        )
        # The sampling day advances through the run (forcing follows the
        # calendar rather than freezing at START_DAY).
        self.assertGreater(
            max(sampled_days), min(sampled_days),
            f"forcing re-sampled but always at the same day: {sampled_days}",
        )


if __name__ == "__main__":
    unittest.main()
