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
    rc = fd.main(["check", "3d_idealized/hydrostatic_gray_1yr", "--data-root", str(tmp_path)])
    assert rc == 1  # amip_sst_sic absent


def test_data_gated_template_present_is_zero(fd, tmp_path):
    target = tmp_path / "amip" / "sst_sic.nc"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\0" * (1024 * 1024 + 1))   # exceed the 1 MiB integrity floor
    rc = fd.main(["check", "3d_idealized/hydrostatic_gray_1yr", "--data-root", str(tmp_path)])
    assert rc == 0


def test_empty_or_truncated_file_rejected_by_integrity(fd, tmp_path):
    # codex MEDIUM: a 0-byte / truncated placeholder must NOT be accepted as
    # present (amip_sst_sic carries a min_bytes floor).
    target = tmp_path / "amip" / "sst_sic.nc"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\0" * 16)                  # below the floor
    rc = fd.main(["check", "3d_idealized/hydrostatic_gray_1yr", "--data-root", str(tmp_path)])
    assert rc == 1


def test_verify_helper(fd, tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"x" * 2048)
    assert fd._verify({"min_bytes": 1024}, f)[0] is True
    assert fd._verify({"min_bytes": 4096}, f)[0] is False
    assert fd._verify({}, tmp_path / "absent")[0] is False
    d = tmp_path / "store"
    d.mkdir()
    # directory datasets REQUIRE a completeness marker (codex followup):
    assert fd._verify({}, d)[0] is False            # no marker declared -> reject
    (d / "x").write_text("y")
    assert fd._verify({}, d)[0] is False            # non-empty but still no marker
    assert fd._verify({"marker": ".complete"}, d)[0] is False   # marker absent -> incomplete
    (d / ".complete").write_text("")
    assert fd._verify({"marker": ".complete"}, d)[0] is True    # marker present -> ok


def test_retired_amip_template_points_to_production_deck(fd):
    with pytest.raises(SystemExit, match="config/amip/amip_production.yaml"):
        fd.main(["check", "coupled/amip", "--data-root", "/tmp"])


def test_unknown_template_raises(fd):
    with pytest.raises(SystemExit):
        fd.main(["check", "nope/missing", "--data-root", "/tmp"])


def test_fetch_no_url_reports_manual(fd, tmp_path, capsys):
    # amip_sst_sic has no automatable url -> fetch must report manual + nonzero.
    rc = fd.main(["fetch", "3d_idealized/hydrostatic_gray_1yr", "--data-root", str(tmp_path)])
    assert rc == 1
    assert "obtain manually" in capsys.readouterr().out
