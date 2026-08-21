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


def test_zu_frc_assembly_dumps_are_levelled_by_their_governing_input():
    """The #1455 JOB-2 forcing-assembly brackets (dynspg_ts.F90) are levelled
    INDIVIDUALLY, by the velocity/tracer that governs each one -- not as a
    block.

    The trap this pins: all six dumps are written inside the same routine and
    within ~140 lines of each other, which makes "they are all the same level"
    an easy and wrong assumption (an earlier draft of the registration made
    exactly that claim). Under DINO's ``ln_bt_fw=.false.``:

      * the drag INCREMENT is the BEFORE-level bottom baroclinic residual
        ``puu(ikbu,Kbb) - puu_b(Kbb)`` (dynspg_ts.F90:1819-1822) -- only its
        r1_hu depth is at Kmm.  Same governing level, same switch and same
        CENTRED branch as the cor2d substep-1 dumps above.
      * ``rCdU_bot`` is NOT that residual: zdf_drg_nonlin reads uu(:,:,:,Kmm)
        (zdfdrg.F90:172-189), so it is NOW even though it is dumped in the very
        same bracket as the BEFORE increment.
      * the assembled ``zu_frc``/``zv_frc`` come from this step's puu(Krhs),
        and ``ssh_frc`` from this step's emp/emp_b -- both NOW.
    """
    for name in ("drg_dump_zu_frc_inc.bin", "drg_dump_zv_frc_inc.bin"):
        assert time_level_for_dump(name) == "before", name
        src = _DUMP_TIME_LEVEL[name][1]
        # the citation must name the governing BEFORE input, not just the file
        assert "dynspg_ts.F90:" in src and "Kbb" in src, src
    for name in ("drg_dump_rCdU_bot.bin", "spg_dump_zu_frc.bin",
                 "spg_dump_zv_frc.bin", "spg_dump_ssh_frc.bin",
                 "sbc_dump_utau.bin"):
        assert time_level_for_dump(name) == "now", name
        assert _DUMP_TIME_LEVEL[name][1].strip(), name

    # The two dumps written in the SAME bracket must disagree on level -- that
    # disagreement is the whole point, so assert it directly rather than
    # trusting the two loops above to have covered it.
    assert (time_level_for_dump("drg_dump_zu_frc_inc.bin")
            != time_level_for_dump("drg_dump_rCdU_bot.bin"))


def test_zu_frc_assembly_citations_record_the_shape_split():
    """Three of these dumps are FULL haloed (jpi x jpj) and three are the
    no-halo interior (A2D(0)) -- and the split does NOT follow the WRITE block:
    ``spg_dump_ssh_frc`` is haloed while ``spg_dump_zu_frc``/``zv_frc``, in the
    SAME WRITE block three lines away, are interior.

    A probe that reshapes an interior dump as haloed (or vice versa) gets a
    plausible array of the wrong size or a silent transpose, so the shape has
    to travel WITH the level. Verified against the real dump sizes:
    56*203*8 = 90944 B haloed, 52*199*8 = 82784 B interior.
    """
    for name in ("drg_dump_rCdU_bot.bin", "spg_dump_ssh_frc.bin",
                 "sbc_dump_utau.bin"):
        src = _DUMP_TIME_LEVEL[name][1]
        assert "haloed" in src, (name, src)
    for name in ("drg_dump_zu_frc_inc.bin", "spg_dump_zu_frc.bin"):
        src = _DUMP_TIME_LEVEL[name][1]
        assert ("Interior" in src or "interior" in src), (name, src)
