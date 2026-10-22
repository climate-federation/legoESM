"""Direct tests of the round-314 rung-10 namelist audit."""

from __future__ import annotations

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round314_namelist_audit as audit,
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_parser_reads_continuations_repeats_structures_and_quoted_bangs(tmp_path):
    src = _write(tmp_path, "nl", (
        "&namx   ! group comment\n"
        "   ln_a = .true.   ! trailing ! comment\n"
        "   rn_b = 1.d-3\n"
        "   sn_c = 'file!name', -12., 'var', .false.\n"
        "   cn_d(2) = 'two'\n"
        "/\n"))
    dump = _write(tmp_path, "dump", (
        "&NAMX\n LN_A=T,\n RN_B=  1.0000000000000000E-003,\n"
        " RN_ARR= 2*1.5000000000000000     ,\n   3.0000000000000000     ,\n"
        ' SN_C%CLNAME="file!name   ",\n SN_C%FREQH= -12.000000000000000     ,\n'
        ' SN_C%CLVAR="var",\n SN_C%LN_TINT=F,\n'
        ' CN_D= "one", "two", "three",\n /\n'))
    s = audit.parse_namelist(src, "oce")
    d = audit.parse_namelist(dump, "oce")
    assert s[("oce", "namx", "ln_a")] == [True]
    assert s[("oce", "namx", "rn_b")] == [1.0e-3]
    assert d[("oce", "namx", "rn_arr")] == [1.5, 1.5, 3.0]
    res = audit.resolve_against_dump(d, s)
    assert res[("oce", "namx", "sn_c%clname")] == ["file!name"]
    assert res[("oce", "namx", "sn_c%freqh")] == [-12.0]
    assert res[("oce", "namx", "sn_c%ln_tint")] == [False]
    assert res[("oce", "namx", "cn_d")] == ["one", "two", "three"]
    assert res[("oce", "namx", "rn_arr")] is None
    for key, dv in d.items():
        if res[key] is not None:
            assert audit.same_list(res[key], dv)[0], key


def test_same_list_explains_only_truncation_and_partial_arrays():
    assert audit.same_list(["DUMMY"], ["DUMM"]) == (True, "dump truncates string to declared length")
    assert audit.same_list([1.0], [1.0, 0.0]) == (True, "partial array")
    assert not audit.same_list([1.0], [2.0])[0]
    assert not audit.same_list([True], [1.0])[0]


class _Ctx:
    def __init__(self, **flat):
        self.flat = flat


def _nemo(**kv):
    return {("oce", *k.split(".")): [v] for k, v in kv.items()}


def test_classifier_fails_closed():
    nemo = _nemo(**{"namdom.rn_dt": 10800.0, "namzz.ln_new": True,
                    "nambdy.ln_bdy": True, "nambdy.nn_rimwidth": 10.0})
    with pytest.raises(audit.AuditError, match="UNCLASSIFIED"):
        audit.classify(("oce", "namzz", "ln_new"), nemo, _Ctx(), "")
    with pytest.raises(audit.AuditError, match="DIFFERS without"):
        audit.classify(("oce", "namdom", "rn_dt"), nemo, _Ctx(**{"card.dt_s": 3600.0}), "")
    with pytest.raises(audit.AuditError, match="INERT switch"):
        audit.classify(("oce", "nambdy", "nn_rimwidth"), nemo, _Ctx(), "")
    with pytest.raises(audit.AuditError, match="'off' rule"):
        audit.classify(("oce", "nambdy", "ln_bdy"), nemo, _Ctx(), "")
    row = audit.classify(("oce", "namdom", "rn_dt"), nemo, _Ctx(**{"card.dt_s": 10800.0}), "")
    assert row["status"] == "MATCH"


def test_compile_key_switch_reads_the_cpp_file():
    nemo = _nemo(**{"namdom.rn_atfp": 0.1})
    row = audit.classify(("oce", "namdom", "rn_atfp"), nemo, _Ctx(), "bld::tool::fppkeys key_RK3")
    assert row["status"] == "INERT"
    with pytest.raises(audit.AuditError):
        audit.classify(("oce", "namdom", "rn_atfp"), nemo, _Ctx(), "bld::tool::fppkeys key_qco")


def test_file_read_settings_compare_the_card_value_not_a_label():
    # The 24 interval / interpolation / climatology rows read the card's value:
    # planting a card value that disagrees with NEMO's turns the row into an
    # unclassed DIFFERS (fail-closed).  The pre-fix rule compared a fixed label
    # with itself and returned MATCH here.
    cases = (("namtsd", "sn_tem%ln_clim", True, "tem.ln_clim", False),
             ("namtsd", "sn_sal%freqh", -1.0, "sal.freqh", -12.0),
             ("namzdf_iwm", "sn_mpb%freqh", -12.0, "iwm.freqh", -1.0),
             ("namzdf_iwm", "sn_dsc%ln_tint", False, "iwm.ln_tint", True))
    for group, key, nemo_v, flat_key, wrong in cases:
        nemo = {("oce", group, key): [nemo_v]}
        assert audit.classify(("oce", group, key), nemo, _Ctx(**{flat_key: nemo_v}), "")["status"] == "MATCH"
        with pytest.raises(audit.AuditError, match="DIFFERS without"):
            audit.classify(("oce", group, key), nemo, _Ctx(**{flat_key: wrong}), "")
    row = audit.classify(("oce", "namtsd", "sn_tem%wname"), {("oce", "namtsd", "sn_tem%wname"): [""]}, _Ctx(), "")
    assert row["status"] == "UNCOMPARED"


def test_off_switch_without_card_value_is_not_a_match():
    row = audit.classify(("oce", "namisf", "ln_isf"), _nemo(**{"namisf.ln_isf": False}), _Ctx(), "")
    assert (row["status"], row["card"]) == ("NOT-MODELLED", "-")


def test_tke_minimum_energy_follows_nemo_wave_mixing_switch():
    key = ("oce", "namzdf_tke", "rn_emin")
    card = _Ctx(**{"physics.vertical_mixing.tke.tke_background": 1.0e-10})
    on = _nemo(**{"namzdf_tke.rn_emin": 1.0e-6, "namzdf.ln_zdfiwm": True})
    off = _nemo(**{"namzdf_tke.rn_emin": 1.0e-6, "namzdf.ln_zdfiwm": False})
    assert audit.classify(key, on, card, "")["status"] == "MATCH"
    with pytest.raises(audit.AuditError, match="DIFFERS without"):
        audit.classify(key, off, card, "")


EXPECTED_DIFFERS = {
    "oce:namrun.nn_itend": "c",
    "oce:namzdf_tke.nn_mxlice": "b:D135",
    "oce:namdrg.ln_drgice_imp": "b:D133",
    "oce:namtra_eiv.ln_ldfeiv": "c",
}
EXPECTED_DIFFERS_EXP00 = {**EXPECTED_DIFFERS, "oce:namberg.ln_icebergs": "b:D7"}


@pytest.fixture(scope="module")
def report():
    for path in (audit.RECORD / "output.namelist.dyn", audit.EXP00 / "namelist_cfg"):
        if not path.is_file():
            pytest.skip(f"{path} is unavailable")
    return audit.audit(audit.RECORD, audit.EXP00, audit.NEMO / "cfgs/SHARED")


def test_rung10_audit_pins_the_round314_table(report):
    assert report["n_keys"] == 1108
    # Round 323: geothermal, damping and MLE rows MATCH; river mouths INERT.
    assert report["counts"] == {"MATCH": 176, "DIFFERS": 4, "NOT-MODELLED": 665, "INERT": 231,
                                "UNCOMPARED": 32}
    assert report["counts_exp00"] == {"MATCH": 175, "DIFFERS": 5, "NOT-MODELLED": 665, "INERT": 231,
                                      "UNCOMPARED": 32}
    assert not [r["key"] for r in report["rows"] if r["status"] == "MATCH" and r["card"] in ("-", "not built (off)")]
    assert report["selfcheck"]["unexplained"] == []
    differs = {f"{r['ns']}:{r['group']}.{r['key']}": r["class"]
               for r in report["rows"] if r["status"] == "DIFFERS"}
    assert differs == EXPECTED_DIFFERS
    differs_exp00 = {f"{r['ns']}:{r['group']}.{r['key']}": r["class_exp00"]
                     for r in report["rows"] if r["status_exp00"] == "DIFFERS"}
    assert differs_exp00 == EXPECTED_DIFFERS_EXP00
    unc = {(r["group"], r["key"].split("%")[1]) for r in report["rows"] if r["status"] == "UNCOMPARED"}
    assert unc == {(g, f) for g in ("namtsd", "namzdf_iwm") for f in ("clftyp", "wname", "vcomp", "lname")}
    assert not any(r["card"] in ("two-record fldread weights", "read once") for r in report["rows"])
    oo_disagree = {r["key"] for r in report["rows"] if r["ocean_output"] == "disagree"}
    assert oo_disagree == {"jpni", "jpnj", "nn_e"}


def test_card_side_is_read_from_the_built_card(report):
    # Plant: the same audit with the card's isoneutral MSC flag flipped moves
    # exactly that row (to DIFFERS since round 320), so the column is read.
    oo = (audit.RECORD / "ocean.output").read_text(errors="replace")
    ctx = audit.build_context(audit.RECORD, oo)
    assert ctx.card.case == "ORCA2-zps"
    planted = ctx._replace(flat={**ctx.flat, "gm_redi.msc_stabilize": False})
    again = audit.audit(audit.RECORD, audit.EXP00, audit.NEMO / "cfgs/SHARED", ctx=planted)
    moved = {r["key"] for r, s in zip(again["rows"], report["rows"]) if r["status"] != s["status"]}
    assert moved == {"ln_traldf_msc"}
