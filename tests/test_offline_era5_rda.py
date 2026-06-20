"""Direct (fast, non-slow) unit test for the shared ERA5 archive writer.

The slow AMIP integration tests exercise ``write_forcing_archive`` only indirectly (and are
excluded from default CI), so this locks the shared helper directly per CLAUDE.md ("every
new .py ≥1 direct unit test") — the d633006 filenames, K/[0,1] units, descending latitude,
the lon endpoint, the SST gradient, and the ``sst_add`` bump.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._offline_era5_rda import write_forcing_archive

_SSTK = "e5.oper.an.sfc.128_034_sstk.ll025sc.2020010100_2020013123.nc"
_CI = "e5.oper.an.sfc.128_031_ci.ll025sc.2020010100_2020013123.nc"


def test_write_forcing_archive_default_grid(tmp_path):
    import xarray as xr

    write_forcing_archive(tmp_path)             # default 8x16, nt=48, sst_add=0
    with xr.open_dataset(tmp_path / _SSTK) as sst:
        assert tuple(sst["SSTK"].shape) == (48, 8, 16)
        assert sst["SSTK"].attrs["units"] == "K"
        assert float(sst["latitude"][0]) == pytest.approx(90.0)    # descending (ERA5 order)
        assert float(sst["latitude"][-1]) == pytest.approx(-90.0)
        assert float(sst["longitude"][-1]) == pytest.approx(337.5)  # 360*15/16
        vals = np.asarray(sst["SSTK"])
        assert float(vals.min()) == pytest.approx(275.0)            # the 90deg pole
        assert float(vals.max() - vals.min()) > 10.0               # a real gradient
    with xr.open_dataset(tmp_path / _CI) as ci:
        assert ci["CI"].attrs["units"] == "(0-1)"
        assert float(np.asarray(ci["CI"]).max()) == 0.0            # zero sea-ice


def test_write_forcing_archive_16x32_and_sst_add(tmp_path):
    import xarray as xr

    write_forcing_archive(tmp_path, nlat=16, nlon=32, sst_add=15.0)
    with xr.open_dataset(tmp_path / _SSTK) as sst:
        assert tuple(sst["SSTK"].shape) == (48, 16, 32)
        assert float(sst["longitude"][-1]) == pytest.approx(348.75)  # 360*31/32
        # The uniform +15 K bump shifts the whole field: the 90deg pole is 275 + 15.
        assert float(np.asarray(sst["SSTK"]).min()) == pytest.approx(290.0)
