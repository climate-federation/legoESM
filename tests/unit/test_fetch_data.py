"""Unit tests for ``scripts/experiment/fetch_data.py`` (check/fetch)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXP_DIR = _REPO_ROOT / "scripts" / "experiment"


@pytest.fixture(scope="module")
def fd():
    if str(_EXP_DIR) not in sys.path:
        sys.path.insert(0, str(_EXP_DIR))
    return importlib.import_module("fetch_data")


def test_idealized_template_needs_no_data(fd, capsys):
    rc = fd.main(["check", "2d/williamson2_sw", "--data-root", "/tmp/whatever"])
    assert rc == 0
    assert "no external data required" in capsys.readouterr().out


def test_data_gated_template_missing_is_nonzero(fd, tmp_path):
    rc = fd.main(["check", "coupled/amip", "--data-root", str(tmp_path)])
    assert rc == 1  # amip_sst_sic absent


def test_data_gated_template_present_is_zero(fd, tmp_path):
    target = tmp_path / "amip" / "sst_sic.nc"
    target.parent.mkdir(parents=True)
    target.write_text("")
    rc = fd.main(["check", "coupled/amip", "--data-root", str(tmp_path)])
    assert rc == 0


def test_unknown_template_raises(fd):
    with pytest.raises(SystemExit):
        fd.main(["check", "nope/missing", "--data-root", "/tmp"])


def test_fetch_no_url_reports_manual(fd, tmp_path, capsys):
    # amip_sst_sic has no automatable url -> fetch must report manual + nonzero.
    rc = fd.main(["fetch", "coupled/amip", "--data-root", str(tmp_path)])
    assert rc == 1
    assert "obtain manually" in capsys.readouterr().out
