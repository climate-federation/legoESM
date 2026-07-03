"""Direct (fast, non-slow) unit test for the shared ERA5 archive writer.

The slow AMIP integration tests exercise ``write_forcing_archive`` only indirectly (and are
excluded from default CI), so this locks the shared helper directly per CLAUDE.md ("every
new .py ≥1 direct unit test") — the d633006 filenames, K/[0,1] units, descending latitude,
the lon endpoint, the SST gradient, and the ``sst_add`` bump.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._offline_era5_rda import write_forcing_archive, write_full_archive

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


def test_write_full_archive_has_pl_and_sfc(tmp_path):
    """The full archive writes pl T/U/V/Q at the 13 WB2 levels + sfc SP/SSTK/CI, so the
    compare (open_local_era5_dataset) AND the forcing (build_era5_amip_forcing) both read it."""
    import xarray as xr
    from legoesm.training.era5_to_state import WB2_PRESSURE_LEVELS

    write_full_archive(tmp_path)                # default 6x8
    # pl: T/U/V/Q daily chunks at exactly the 13 WB2 levels.
    with xr.open_dataset(
            tmp_path / "e5.oper.an.pl.128_130_t.ll025sc.2020010100_2020010123.nc") as t:
        assert tuple(t["T"].shape) == (4, len(WB2_PRESSURE_LEVELS), 6, 8)
        assert float(t["T"].mean()) == pytest.approx(250.0)
    for code in ("128_131_u", "128_132_v", "128_133_q"):
        assert (tmp_path / f"e5.oper.an.pl.{code}.ll025sc.2020010100_2020010123.nc").exists()
    # sfc: SP/SSTK(K)/CI([0,1]) monthly chunks.
    with xr.open_dataset(tmp_path / _SSTK) as sst:
        assert sst["SSTK"].attrs["units"] == "K"
        assert float(sst["SSTK"].mean()) == pytest.approx(290.0)   # constant (the compare)
    with xr.open_dataset(tmp_path / _CI) as ci:
        assert ci["CI"].attrs["units"] == "(0-1)"
    assert (tmp_path / "e5.oper.an.sfc.128_134_sp.ll025sc.2020010100_2020013123.nc").exists()
