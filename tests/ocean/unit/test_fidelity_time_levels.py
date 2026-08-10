"""Tests for the NEMO dump time-level registry (#1226).

Non-vacuity: the registry must fire on the ACTUAL mistake that was made twice
in one day — comparing a before-level dump against now-level T/S.
"""
from __future__ import annotations

import pytest

from legoesm.ocean.fidelity.time_levels import (
    _DUMP_TIME_LEVEL,   # white-box: the citation string is part of the contract
    register_dump,
    select_ts,
    time_level_for_dump,
)


def test_eos_rab_and_rn2b_dumps_are_before_level():
    """The exact dumps that were mis-compared. stpmlf.F90:184."""
    for name in ("dump_alpha_b.bin", "dump_beta_b.bin", "tke_dump_rn2b.bin",
                 "dump_nmln.bin", "dump_hmlp.bin"):
        assert time_level_for_dump(name) == "before", name


def test_ttrd_ldf_trend_is_before_level():
    """#1455 A5: NEMO's isoneutral(Redi) tracer trend reads Kbb tracer
    gradients (traldf_iso_scheme.h90:26-30) -- before level. Pins the label
    so a wrong "now" cannot be introduced silently."""
    assert time_level_for_dump("ttrd_ldf") == "before"
    assert time_level_for_dump("strd_ldf") == "before"


def test_raw_pre_shapiro_slope_dumps_are_registered():
    """The ldf_slp internals split (formula vs smoother) needs these."""
    assert time_level_for_dump("eiv_dump_zwz_raw.bin") == "before"
    assert time_level_for_dump("eiv_dump_zww_raw.bin") == "before"


def test_ldfslp_intermediate_chain_is_registered():
    """The full j-direction walk: prd -> zgrv -> zaj -> zbw/zbj -> zfk -> zww."""
    for name in ("eiv_dump_prd_arg.bin", "eiv_dump_zgrv_iik.bin",
                 "eiv_dump_zgrv_iikm1.bin", "eiv_dump_zaj.bin",
                 "eiv_dump_zbw.bin", "eiv_dump_zbj.bin", "eiv_dump_zfk.bin"):
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


def test_cor2d_substep1_dumps_are_before_level():
    """The four dyn_cor_2D substep-1 dumps (dynspg_ts.F90:794-806) are
    BEFORE-level: with DINO's ln_bt_fw=.false. + nn_bt_flt=2 the substep loop
    is seeded from puu_b(Kbb) and the jn=1 AB3 extrapolation is the identity
    (za1=1, za2=za3=0), so ua_e == un_e == puu_b(Kbb).

    Before this registration ``time_level_for_dump`` RAISED on all four (the
    registry fails closed), so any harness comparing them was picking a level
    by hand -- exactly the Rule-1d failure mode.
    """
    for name in ("cor2d_dump_ua_e_in_substep1.bin",
                 "cor2d_dump_va_e_in_substep1.bin",
                 "cor2d_dump_zu_trd_substep1.bin",
                 "cor2d_dump_zv_trd_substep1.bin"):
        assert time_level_for_dump(name) == "before", name
        # Rule 1d: an unsourced entry is a guess -- the citation must name the
        # dumping WRITE and the proof chain, not just the file.
        src = _DUMP_TIME_LEVEL[name][1]
        assert "dynspg_ts.F90:" in src and "Kbb" in src, src
