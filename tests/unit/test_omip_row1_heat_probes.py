"""Direct tests of the row-1 heat probes (omip_row1_heat_terms / _diag).

Each probe's pure helper is exercised on synthetic data with a known answer,
and the run paths are REQUIRED arguments (no hard-coded run directories).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "validate" / "ocean_fidelity"))

import omip_row1_heat_diag as hd  # noqa: E402
import omip_row1_heat_terms as htm  # noqa: E402


def test_thickness_heat_split_is_exact():
    rng = np.random.default_rng(0)
    Ta, Tb = rng.normal(size=(3, 4, 2)), rng.normal(size=(3, 4, 2))
    ha, hb = rng.uniform(1, 2, (3, 4, 2)), rng.uniform(1, 2, (3, 4, 2))
    area = rng.uniform(1, 3, (3, 4))
    k, dt = 4.0e6, 10.0
    H_dh, H_dT = htm.thickness_heat_split(Ta, Tb, ha, hb, area, k, dt)
    want = k * ((Tb * hb - Ta * ha) * area[..., None]).sum() / dt
    assert H_dh + H_dT == pytest.approx(want, rel=1e-13)


def test_parse_fwbudget():
    lines = ["noise\n",
             "[fwbudget] step=576 day=1.00 tripole: P=+1.2000 E=-0.5000 "
             "R=+0.9770 ice=+0.0100 net(P-E+R+ice)=+2.6870 Sv (raw ...)\n"]
    m = htm.parse_fwbudget(lines)
    assert len(m) == 1 and int(m[0].group(1)) == 576
    assert float(m[0].group(5)) == pytest.approx(0.977)
    assert float(m[0].group(7)) == pytest.approx(2.687)


def test_col_heat_and_band_sums():
    from legoesm.ocean.vertical import create_ocean_z_star
    zc = create_ocean_z_star(n_levels=2, H_max=1000.0)
    s = {"T": np.full((2, 3, 2), 2.0), "eta": np.zeros((2, 3)),
         "H_bathy": np.full((2, 3), 1000.0),
         "land_mask": np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
         "cell_area": np.full((2, 3), 5.0)}
    col = hd.col_heat(s, zc, 1000.0, 4000.0)
    assert col[0, 2] == 0.0
    assert col[1, 0] == pytest.approx(1000.0 * 4000.0 * 2.0 * 1000.0 * 5.0)
    lat = np.array([[-10.0] * 3, [10.0] * 3])
    sums = hd.band_sums(col, lat, [(-90, 0), (0, 90)])
    assert sums == pytest.approx([col[0].sum(), col[1].sum()])


@pytest.mark.parametrize("mod, missing", [
    (htm, ["--row-dir", "x", "--row-log-dir", "x", "--base-dir", "x",
           "--base-log-dir", "x"]),
    (hd, ["--row-dir", "x", "--base-dir", "x"]),
])
def test_run_paths_are_required(mod, missing):
    with pytest.raises(SystemExit):
        mod.parse_args(missing)          # --domain-cfg absent
    full = mod.parse_args([*missing, "--domain-cfg", "d.nc"])
    assert full.domain_cfg == "d.nc"
