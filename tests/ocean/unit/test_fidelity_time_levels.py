"""Tests for the NEMO dump time-level registry (#1226).

Non-vacuity: the registry must fire on the ACTUAL mistake that was made twice
in one day — comparing a before-level dump against now-level T/S.
"""
from __future__ import annotations

import pytest

from legoesm.ocean.fidelity.time_levels import (
    register_dump,
    select_ts,
    time_level_for_dump,
)


def test_eos_rab_and_rn2b_dumps_are_before_level():
    """The exact dumps that were mis-compared. stpmlf.F90:184."""
    for name in ("dump_alpha_b.bin", "dump_beta_b.bin", "tke_dump_rn2b.bin",
                 "dump_nmln.bin", "dump_hmlp.bin"):
        assert time_level_for_dump(name) == "before", name


def test_atf_after_dumps_are_after_level():
    assert time_level_for_dump("atf_dump_tem_after.bin") == "after"
    assert time_level_for_dump("atf_dump_tem_before.bin") == "before"


def test_full_paths_are_accepted():
    assert time_level_for_dump("/a/b/RUN_GDB/dump_alpha_b.bin") == "before"


def test_unregistered_dump_raises_and_never_defaults():
    """Fail-closed: a silent 'now' default is the original bug."""
    with pytest.raises(ValueError, match="no registered NEMO time level"):
        time_level_for_dump("dump_something_new.bin")


def test_select_ts_picks_before_for_an_eos_rab_dump():
    now, bef = ("T_now", "S_now"), ("T_bef", "S_bef")
    assert select_ts("dump_alpha_b.bin", now=now, before=bef) == bef


def test_select_ts_raises_when_the_needed_level_is_missing():
    with pytest.raises(ValueError, match="no 'after' state was supplied|needs the 'after'"):
        select_ts("atf_dump_tem_after.bin", now=("a", "b"), before=("c", "d"))


def test_register_dump_requires_a_source_citation():
    with pytest.raises(ValueError, match="needs a NEMO source citation"):
        register_dump("dump_fake.bin", "now", "   ")


def test_register_dump_rejects_a_bogus_level():
    with pytest.raises(ValueError, match="unknown time level"):
        register_dump("dump_fake.bin", "midpoint", "somefile.F90:1")


def test_register_dump_round_trips():
    register_dump("dump_unit_test.bin", "now", "unit_test.F90:1")
    assert time_level_for_dump("dump_unit_test.bin") == "now"
