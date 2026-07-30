"""The forcing loader must honour the requested TIME WINDOW, not just its length.

``stage_forcing`` loads only the 6-hourly slices bracketing the requested model
times and the disaggregator then interpolates on ``LandForcing.time_s``.  A
loader that returns ``n_time`` slices always starting at index 0 of the year is
therefore silently wrong for ANY window that does not begin at Jan 1 -- a run
with ``start_doy > 0``, or any sub-year forcing chunk, would be handed January.
"""
import numpy as np
import pytest

from legoesm.land.forcing import cru_jra


def test_synthetic_honours_the_requested_start_index():
    """The absolute slice index must set the time origin.

    Two clocks: ``time_s_solar`` is START-stamped, ``time_s`` is the MIDPOINT
    (= solar + half an interval), matching the CRU-JRA convention.
    """
    step_s = 6 * 3600.0
    for t0 in (0, 40, 1080):
        f = cru_jra.synthetic_land_forcing(2000, n_time=4, t_index0=t0)
        np.testing.assert_allclose(np.asarray(f.time_s_solar),
                                   (t0 + np.arange(4)) * step_s, rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(f.time_s),
                                   (t0 + np.arange(4)) * step_s + 0.5 * step_s,
                                   rtol=0, atol=1e-6)


def test_synthetic_default_is_unchanged():
    """t_index0 defaults to 0 -> every pre-existing caller is bit-identical."""
    a = cru_jra.synthetic_land_forcing(2000, n_time=4)
    b = cru_jra.synthetic_land_forcing(2000, n_time=4, t_index0=0)
    np.testing.assert_array_equal(np.asarray(a.time_s), np.asarray(b.time_s))
    np.testing.assert_array_equal(np.asarray(a.tbot), np.asarray(b.tbot))


def test_load_cru_jra_forwards_the_window_to_synthetic():
    """load_cru_jra passes only a COUNT unless it forwards time_indices[0];
    this is the regression that made sub-year forcing chunks wrong."""
    idx = list(range(40, 44))
    f = cru_jra.load_cru_jra(2000, data_dir=None, time_indices=idx,
                             allow_synthetic=True)
    step_s = 6 * 3600.0
    np.testing.assert_allclose(np.asarray(f.time_s_solar),
                               np.asarray(idx) * step_s, rtol=0, atol=1e-6)


def test_a_mid_year_window_carries_a_different_clock():
    """Two different windows must not land on the same clock.

    NOTE: the synthetic climatology's STATE fields (tbot/qbot/wind/...) are
    time-invariant by construction -- it has no seasonal cycle -- so the window
    offset shows up only in the CLOCKS, and through them in the solar-zenith
    disaggregation (cos_zenith / FSDS).  That is exactly the channel by which an
    unhonoured window corrupted sub-year forcing chunks.
    """
    jan = cru_jra.load_cru_jra(2000, data_dir=None, time_indices=list(range(0, 4)),
                               allow_synthetic=True)
    jul = cru_jra.load_cru_jra(2000, data_dir=None,
                               time_indices=list(range(1080, 1084)),
                               allow_synthetic=True)
    assert not np.allclose(np.asarray(jan.time_s), np.asarray(jul.time_s))
    assert not np.allclose(np.asarray(jan.time_s_solar),
                           np.asarray(jul.time_s_solar))
    # ~6 months apart on the solar clock
    dt_days = (float(np.asarray(jul.time_s_solar)[0])
               - float(np.asarray(jan.time_s_solar)[0])) / 86400.0
    assert 260.0 < dt_days < 280.0, dt_days


@pytest.mark.parametrize("dt_s,chunk,ok", [
    (3600.0, 6, True),      # exactly one 6-h interval
    (3600.0, 30, True),     # 5 intervals
    (3600.0, 744, True),    # a 31-day month at hourly dt
    (3600.0, 720, True),    # a 30-day month
    (3600.0, 31, False),    # splits an interval -> would alter the shortwave
    (3600.0, 1, False),
    (1800.0, 12, True),     # 6 h at 30-min dt
    (1800.0, 10, False),
])
def test_chunk_alignment_rule(dt_s, chunk, ok):
    """A forcing chunk must tile whole 6-hourly intervals.

    The SW disaggregation conserves each interval mean over exactly the substeps
    tiling it, so a boundary inside an interval changes the reconstructed SW.
    Measured: chunk=30 is bit-identical to whole-year staging; chunk=31 diverges
    to 6.7e4 W/m2 in lhflx.  The driver enforces this rule; this pins the
    arithmetic it enforces.
    """
    steps_per_interval = int(cru_jra.CRUJRA_FREQ_HOURS * 3600.0 / dt_s)
    assert (chunk % steps_per_interval == 0) is ok
