"""Seasonal insolation offset (iter 449).

The radiation insolation derives ``day_of_year`` from the model day via
``ModelDriver._calendar_for_radiation``. ``config.insolation_start_doy`` lets an
AMIP run started from a non-January ERA5 date run the matching solar season by
mapping model day 0 to that noleap day-of-year -- decoupled from the
relative-indexed SST forcing (which keeps day 0 = forcing start). These tests
pin the offset math and, crucially, that the DEFAULT (``None``) is byte-identical
to the legacy ``day_to_calendar(day)`` at every insolation site.
"""
import tempfile

import pytest
from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver
from legoesm.forcing.time_utils import day_to_calendar, noleap_day_of_year


def _driver(insolation_start_doy=None) -> ModelDriver:
    """Cheapest real driver: __init__ sets _insolation_day_offset; the seam is
    pure, so no .setup()/run is needed."""
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="gaussian", resolution=8, nlev=10),
        dycore=DycoreConfig(discretization="spectral", dt=600.0),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=1, dataset="analytical", radiation="gray",
        convection="none", turbulence="none", distributed=False,
        insolation_start_doy=insolation_start_doy,
    )
    with tempfile.TemporaryDirectory() as td:
        return ModelDriver(cfg, output_dir=td)


def test_default_offset_is_zero_and_byte_identical():
    d = _driver(None)
    assert d._insolation_day_offset == 0.0
    for day in [0.0, 0.5, 3.25, 244.0, 1000.7, -2.5]:
        assert d._calendar_for_radiation(day) == day_to_calendar(day)


def test_day0_maps_to_start_doy():
    for sd in [1, 60, 152, 244, 365]:
        d = _driver(sd)
        assert d._insolation_day_offset == pytest.approx(float(sd) - 1.0)
        doy, _sod = d._calendar_for_radiation(0.0)
        assert doy == pytest.approx(float(sd))


def test_insolation_day_shifts_both_paths_consistently():
    """``_insolation_day`` is the single offset source feeding BOTH the calendar
    (rrtmgp day_of_year) and the gray ``daily_mean_insolation`` declination, so
    they shift by the SAME amount. Default => identity (gray path unchanged)."""
    d0 = _driver(None)
    for day in [0.0, 0.5, 200.0]:
        assert d0._insolation_day(day) == day            # gray path untouched at default
    d = _driver(244)                                     # Sep 1
    for day in [0.0, 0.5, 200.0]:
        assert d._insolation_day(day) == pytest.approx(day + 243.0)
        # the calendar seam consumes the SAME shifted day
        assert d._calendar_for_radiation(day) == day_to_calendar(day + 243.0)


def test_integer_offset_preserves_diurnal_phase():
    """An integer-day offset shifts day_of_year but NOT seconds_of_day."""
    d = _driver(noleap_day_of_year(9, 1))  # Sep 1 -> doy 244
    for day in [0.5, 3.25, 10.75]:
        doy_off, sod_off = d._calendar_for_radiation(day)
        doy_base, sod_base = day_to_calendar(day)
        assert sod_off == pytest.approx(sod_base)        # diurnal phase kept
        assert doy_off != pytest.approx(doy_base)        # season shifted


def test_start_doy_one_equals_default():
    """Jan 1 (doy 1) is exactly the legacy default."""
    d1 = _driver(1)
    d0 = _driver(None)
    assert d1._insolation_day_offset == d0._insolation_day_offset == 0.0
    for day in [0.0, 5.5, 200.0]:
        assert d1._calendar_for_radiation(day) == d0._calendar_for_radiation(day)


def test_invalid_start_doy_rejected_at_validate_strict():
    for bad in [0, 366, 400, -5]:
        with pytest.raises(ValueError, match="insolation_start_doy"):
            ExperimentConfig(insolation_start_doy=bad).validate_strict()


def test_every_radiation_scheme_consumes_forcing_day_of_year():
    """The insolation seam sets the forcing ``day_of_year``/``seconds_of_day``; this LOCKS that
    EVERY registered radiation scheme's step fn actually CONSUMES them — else
    ``insolation_start_doy`` would be a SILENT no-op for that scheme. Verified (iter 457) that
    rrtmgp builds ``cos_zenith`` from these args exactly like gray (the realistic empirical path
    uses rrtmgp, which the gray-only iter-449 empirical test did not cover). A refactor that
    drops ``day_of_year`` from a builder's signature fails here.

    Introspects the private ``_RADIATION_BUILDERS`` dispatch table on purpose (a contract test
    of the scheme registry; not a production cross-module private import)."""
    import inspect

    from legoesm.driver import physics_pipeline as pp

    schemes = pp._RADIATION_BUILDERS
    assert {"none", "gray", "rrtmgp", "rrtmg"} <= set(schemes)   # the validate_strict set
    checked = 0
    for scheme, builder in schemes.items():
        try:
            rad_fn = builder(ExperimentConfig())
        except (FileNotFoundError, OSError) as exc:               # rrtmgp optics data absent
            pytest.skip(f"{scheme} radiation data unavailable: {exc}")
        params = set(inspect.signature(rad_fn).parameters)
        assert {"day_of_year", "seconds_of_day"} <= params, (
            f"radiation scheme {scheme!r} step fn must consume day_of_year + seconds_of_day "
            f"(the forcing fields the insolation seam sets); got {sorted(params)}")
        checked += 1
    assert checked >= 4                                          # none/gray/rrtmgp/rrtmg
