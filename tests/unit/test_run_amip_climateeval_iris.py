"""Executable tests of the ``--month`` preprocessor on real iris cubes.

Runs ONLY under the ClimateEval environment (iris is not a legoESM
dependency); skipped elsewhere.  Invoke as::

    <ClimateEval>/.pixi/envs/default/bin/python -m pytest --noconftest \\
        --rootdir=tests/unit -c /dev/null tests/unit/test_run_amip_climateeval_iris.py

(``--noconftest`` because the repo conftest imports JAX, which that env lacks.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

iris = pytest.importorskip("iris")
from cf_units import Unit  # noqa: E402
from iris.coords import DimCoord  # noqa: E402
from iris.cube import Cube  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "validate"))
from run_amip_climateeval import extract_months_keep_time  # noqa: E402


def _monthly_cube(n_months: int, start_month: int = 1) -> Cube:
    # mid-month points, days since 1979-01-01, 360-day calendar keeps it exact
    pts = np.array([(start_month - 1 + i) * 30 + 15 for i in range(n_months)], float)
    time = DimCoord(pts, standard_name="time",
                    units=Unit("days since 1979-01-01", calendar="360_day"))
    lat = DimCoord(np.array([-45.0, 45.0]), standard_name="latitude", units="degrees")
    data = np.arange(n_months * 2, dtype=float).reshape(n_months, 2)
    return Cube(data, dim_coords_and_dims=[(time, 0), (lat, 1)])


def test_one_month_cube_keeps_its_time_dimension():
    out = extract_months_keep_time(_monthly_cube(1), [1])
    assert out.ndim == 2 and out.shape[0] == 1
    assert out.coord_dims("time") == (0,)


def test_selects_only_requested_months_from_a_long_record():
    out = extract_months_keep_time(_monthly_cube(24), [1])
    assert out.shape[0] == 2
    assert set(out.coord("month_number").points) == {1}


def test_missing_requested_month_is_a_hard_error():
    with pytest.raises(ValueError, match="months \\[2\\] absent"):
        extract_months_keep_time(_monthly_cube(1), [1, 2])


def test_month_outside_1_12_is_refused():
    with pytest.raises(ValueError, match="1..12"):
        extract_months_keep_time(_monthly_cube(12), [1, 13])


def test_stale_month_number_is_rebuilt_from_time():
    from iris.coords import AuxCoord
    cube = _monthly_cube(3)  # Jan Feb Mar
    cube.add_aux_coord(AuxCoord(np.array([7, 8, 9]), long_name="month_number"), 0)
    out = extract_months_keep_time(cube, [2])
    assert out.shape[0] == 1 and float(out.data[0, 0]) == 2.0


def test_lazy_data_stays_lazy():
    import dask.array as da
    cube = _monthly_cube(12)
    cube.data = da.from_array(cube.data)
    out = extract_months_keep_time(cube, [6])
    assert out.has_lazy_data() and out.shape == (1, 2)


def test_fixed_field_without_time_passes_through():
    lat = DimCoord(np.array([-45.0, 45.0]), standard_name="latitude", units="degrees")
    fx = Cube(np.array([0.3, 0.7]), dim_coords_and_dims=[(lat, 0)])
    assert extract_months_keep_time(fx, [1]) is fx


def test_scalar_time_is_refused_not_silently_passed():
    cube = _monthly_cube(1)[0]  # iris collapses time to a scalar coord
    assert not cube.coords("time", dim_coords=True)
    with pytest.raises(ValueError, match="not a dimension coordinate"):
        extract_months_keep_time(cube, [1])
