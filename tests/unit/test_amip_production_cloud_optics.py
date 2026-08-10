"""The production AMIP config must carry the cloud-optics corrections (#1519).

Without them the radiation solver treats every partly-cloudy column as a single
homogeneous overcast slab at the grid-mean water path, which is always brighter
than the independent-column answer.  Measured cost of the omission on two
otherwise-identical 365-day AMIP runs: rsut +59.33 -> +36.60 W/m2, i.e. 22.7 W/m2
of spurious reflected shortwave, with rlut and prw improving together.

This file previously set NEITHER key, so it fell through to the code defaults
("none" / "constant") and anyone running the production configuration as written
reproduced the uncorrected treatment.  The completed campaign years carry the
corrections only because they were passed on the command line.
"""
from __future__ import annotations

import pathlib

import yaml

_CONFIG = (pathlib.Path(__file__).resolve().parents[2]
           / "config" / "amip" / "amip_production.yaml")


def _load() -> dict:
    return yaml.safe_load(_CONFIG.read_text())


def test_partial_coverage_optics_is_two_column() -> None:
    assert _load()["cloud_partial_coverage_optics"] == "two_column"


def test_inhomogeneity_is_two_region() -> None:
    assert _load()["cloud_optics_inhomogeneity"] == "two_region"


def test_fsd_is_set_and_in_range() -> None:
    # cloud_fsd is the width the two_region optic consumes; it is INERT under
    # "constant", so it is only meaningful alongside the key above.
    fsd = _load()["cloud_fsd"]
    assert isinstance(fsd, float)
    assert 0.0 <= fsd <= 1.0


def test_vertical_overlap_is_not_set_alongside_two_column() -> None:
    # config.py:1799 refuses the pair: max_random and two_column are two
    # COMPLETE alternative treatments of the same degree of freedom, not
    # composable corrections.  Setting both makes the config unloadable, so a
    # future edit that adds one must remove the other.
    cfg = _load()
    if cfg.get("cloud_partial_coverage_optics", "none") != "none":
        assert cfg.get("cloud_vertical_overlap_optics", "none") == "none"


def test_the_three_keys_travel_together() -> None:
    # They are one lever.  Half-applying it (e.g. two_column without
    # two_region) is a configuration nobody validated and silently differs from
    # the runs this campaign scored.
    cfg = _load()
    on = {
        "cloud_partial_coverage_optics": cfg.get(
            "cloud_partial_coverage_optics", "none") != "none",
        "cloud_optics_inhomogeneity": cfg.get(
            "cloud_optics_inhomogeneity", "constant") != "constant",
        "cloud_fsd": cfg.get("cloud_fsd") is not None,
    }
    assert len(set(on.values())) == 1, (
        f"cloud-optics keys half-applied: {on}")
