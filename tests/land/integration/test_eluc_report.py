"""The run_lmip_biophys post-run E_LUC bookkeeping hook (_report_eluc)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp

from legoesm.land.surface_params import N_PFT_CLM5
from scripts.run.run_lmip_biophys import _report_eluc

_FOREST = 1


def _gsd(pft_years):
    nyear = pft_years.shape[0]
    return SimpleNamespace(
        pft_frac=jnp.asarray(pft_years),
        years=jnp.asarray(np.arange(1850.0, 1850.0 + nyear)),
        cell_area=jnp.asarray([1.0e10]),
    )


def test_report_eluc_writes_annual_series(tmp_path):
    # 2-year transient cover: forest -> bare on the single cell (deforestation).
    pft = np.zeros((2, 1, N_PFT_CLM5)); pft[0, 0, _FOREST] = 1.0; pft[1, 0, 0] = 1.0
    args = SimpleNamespace(_cfg_luc={"scheme": "bookkeeping"}, output=str(tmp_path / "run"))
    _report_eluc(args, _gsd(pft))

    out = tmp_path / "run.eluc_annual.txt"
    assert out.exists()
    data = np.loadtxt(out)
    assert data.shape == (2, 2)
    assert data[0, 0] == 1850 and data[1, 0] == 1851       # year column
    assert data[0, 1] == 0.0                               # no prior-year transition
    assert data[1, 1] > 0.0                                # deforestation = source


def test_report_eluc_scheme_none_is_noop(tmp_path):
    pft = np.zeros((2, 1, N_PFT_CLM5)); pft[0, 0, _FOREST] = 1.0; pft[1, 0, 0] = 1.0
    args = SimpleNamespace(_cfg_luc={"scheme": "none"}, output=str(tmp_path / "run"))
    _report_eluc(args, _gsd(pft))
    assert not (tmp_path / "run.eluc_annual.txt").exists()


def test_report_eluc_single_year_no_file(tmp_path, capsys):
    pft = np.zeros((1, 1, N_PFT_CLM5)); pft[0, 0, _FOREST] = 1.0
    args = SimpleNamespace(_cfg_luc={"scheme": "bookkeeping"}, output=str(tmp_path / "run"))
    _report_eluc(args, _gsd(pft))
    assert not (tmp_path / "run.eluc_annual.txt").exists()
    assert "single-year" in capsys.readouterr().out


def test_report_eluc_gross_transition_note(tmp_path, capsys):
    # A gross-transition dataset (LUH2) gets the net-vs-gross understatement note;
    # an anthropogenic dataset (HYDE, net-only) does not.
    pft = np.zeros((2, 1, N_PFT_CLM5)); pft[0, 0, _FOREST] = 1.0; pft[1, 0, 0] = 1.0
    luh2 = SimpleNamespace(_cfg_luc={"scheme": "bookkeeping"},
                           _cfg_land_cover_dataset="luh2", output=str(tmp_path / "l"))
    _report_eluc(luh2, _gsd(pft))
    assert "gross-transition emissions are understated" in capsys.readouterr().out

    hyde = SimpleNamespace(_cfg_luc={"scheme": "bookkeeping"},
                           _cfg_land_cover_dataset="hyde", output=str(tmp_path / "h"))
    _report_eluc(hyde, _gsd(pft))
    assert "understated" not in capsys.readouterr().out
