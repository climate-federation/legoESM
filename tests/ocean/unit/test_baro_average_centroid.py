"""Known-answer + synthetic-violation tests for ``baro_average_centroid``.

The probe's whole job is to decide whether two averaging kernels place their
time-average at the same instant, so the tests that matter are the ones that
would go RED if the port drifted from NEMO's ``ts_wgt`` or if the shipped
legoESM kernel stopped agreeing with it.
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

_PROBE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "scripts", "validate", "ocean_fidelity", "dino_1226",
    "baro_average_centroid.py")


def _load():
    spec = importlib.util.spec_from_file_location("baro_average_centroid", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bac = _load()

# The DINO twin's RESOLVED barotropic settings (NEMO ln_bt_auto computes
# nn_e=23; the namelist's 30 is overridden and must never be used here).
DINO_NN_E = 23


def test_dino_centred_boxcar_window_and_centroid():
    """nn_e=23, nn_bt_flt=2, ln_bt_fw=F -> boxcar 24..68 centred on jic=46."""
    w1, w2, icycle = bac.nemo_ts_wgt(DINO_NN_E, 2, False)
    assert icycle == 68
    nz = np.nonzero(w1)[0] + 1
    assert (nz[0], nz[-1]) == (24, 68)
    assert np.all(w1[nz - 1] == 1.0)
    # symmetric about jic = 2*nn_e = 46, so the centroid IS the new-time point
    assert bac.centroid(w1) == pytest.approx(46.0, abs=1e-12)
    # secondary = tail sums: 45 for jn<=24 then a linear ramp 45..1
    assert w2[0] == pytest.approx(45.0)
    assert w2[-1] == pytest.approx(1.0)
    assert bac.centroid(w2) == pytest.approx(76.0 / 3.0, abs=1e-12)


def test_forward_and_narrow_and_none_variants():
    """The other three ts_wgt branches, each with its own known answer."""
    # ll_fw=T moves the centre to jic = nn_e
    w1, _, icycle = bac.nemo_ts_wgt(DINO_NN_E, 2, True)
    assert icycle == 45
    assert bac.centroid(w1) == pytest.approx(23.0, abs=1e-12)
    # nn_bt_flt=1 is the HALF-width boxcar: |jn-46|/23 < 0.5 -> 35..57
    w1, _, icycle = bac.nemo_ts_wgt(DINO_NN_E, 1, False)
    nz = np.nonzero(w1)[0] + 1
    assert (nz[0], nz[-1]) == (35, 57)
    assert icycle == 57
    assert bac.centroid(w1) == pytest.approx(46.0, abs=1e-12)
    # nn_bt_flt=0 is a single spike at jic -- an average of one sample
    w1, _, icycle = bac.nemo_ts_wgt(DINO_NN_E, 0, False)
    assert icycle == 46 and w1.sum() == 1.0
    assert bac.centroid(w1) == pytest.approx(46.0, abs=1e-12)


def test_invalid_filter_and_degenerate_weights_raise():
    with pytest.raises(ValueError, match="nn_bt_flt"):
        bac.nemo_ts_wgt(DINO_NN_E, 7, False)
    # nn_bt_flt=3 forces ll_bt_av=.FALSE.; asking for averaging is a caller bug
    with pytest.raises(ValueError, match="ll_bt_av"):
        bac.nemo_ts_wgt(DINO_NN_E, 3, False, ll_bt_av=True)
    with pytest.raises(ValueError, match="no centroid"):
        bac.centroid(np.zeros(5))


def test_shipped_lego_kernel_matches_nemo_ts_wgt_exactly():
    """THE LOAD-BEARING ONE.

    At the shipped card (leap-frog => substep_scale=2, n_substeps=46) the
    legoESM eta-averaging kernel must be the SAME vector as NEMO's normalised
    wgtbtp1.  If this goes red the probe's "predicted offset 0.000" is void.
    """
    card = bac.lego_card_settings()
    assert card["outer_integrator"] == "leapfrog"
    assert card["substep_scale"] == 2
    assert card["n_barotropic_substeps"] == DINO_NN_E
    assert card["barotropic_time_filter"] in ("nemo_boxcar_ab3",
                                              "nemo_boxcar_centred")
    n_sub = card["n_barotropic_substeps"] * card["substep_scale"]
    lw, ltr, n_loop = bac.lego_weights(n_sub, card["substep_scale"])
    w1, w2, icycle = bac.nemo_ts_wgt(DINO_NN_E, 2, False)
    assert n_loop == icycle == 68
    np.testing.assert_allclose(lw, w1 / w1.sum(), rtol=0, atol=1e-15)
    # ...and therefore the same first moment, which is the probe's verdict
    assert bac.centroid(lw) == pytest.approx(bac.centroid(w1), abs=1e-9)
    # the secondary/transport kernel agrees too (tail sums, up to NEMO's
    # unnormalised convention)
    assert bac.centroid(ltr) == pytest.approx(bac.centroid(w2), abs=1e-9)


def test_wrong_substep_scale_breaks_the_kernel_identity():
    """SYNTHETIC VIOLATION -- proves the test above can fail.

    ``substep_scale`` is what ties legoESM's window to NEMO's ``jic=2*nn_e``.
    Built on the forward-Euler scale the kernel is a DIFFERENT object: 45
    substeps spanning index 1..45 instead of 68 spanning 24..68.

    NOTE, because the obvious reading of this is wrong and the first version of
    this docstring got it wrong: the forward-Euler centroid is NOT "a whole step
    early".  Substep 23 on a clock that starts at ``t`` is the SAME INSTANT as
    substep 46 on a clock that starts at ``t-dt`` -- both are ``t+dt``.  The
    violation is in the index-space kernel identity, and the time-space offset
    is invariant, which is exactly why the probe gates its verdict on the
    kernel comparison rather than on the centroid difference alone.
    """
    lw_fe, _, n_loop_fe = bac.lego_weights(DINO_NN_E, 1)
    w1, _, icycle = bac.nemo_ts_wgt(DINO_NN_E, 2, False)
    assert n_loop_fe == 45 and icycle == 68        # different objects
    assert len(lw_fe) != len(w1)
    assert bac.centroid(lw_fe) == pytest.approx(23.0, abs=1e-6)
    # ...and the two centroids nonetheless describe the SAME instant, t+dt,
    # once each is placed on its own clock.  This is the trap the probe's gate
    # exists to catch.
    rn_dt, nn_e = 2700.0, DINO_NN_E
    t_fe = 0.0 + bac.centroid(lw_fe) * (rn_dt / nn_e)
    t_lf = -rn_dt + bac.centroid(w1) * (rn_dt / nn_e)
    assert t_fe == pytest.approx(rn_dt, abs=1e-9)
    assert t_lf == pytest.approx(rn_dt, abs=1e-9)


def test_ocean_output_parser_reads_the_resolved_values(tmp_path):
    """The parser must take the RESOLVED nn_e, not a namelist literal."""
    log = tmp_path / "ocean.output"
    log.write_text(
        "      Barotropic time filter => nn_bt_flt =            2\n"
        "         Boxcar: width = 2*nn_e\n"
        "                               in iterations nn_e =           23\n"
        "         ln_bt_fw=F => Centred integration of barotropic variables \n"
        "      Barotropic time steps => in seconds         =    117.39130434\n"
        "   ===>>>   Modified Leap-Frog (MLF) :   rDt =    5400.0000\n"
        "      rn_Dt      = 2700.\n")
    got = bac.read_ocean_output(str(log))
    assert got["nn_e"] == 23 and got["nn_bt_flt"] == 2
    assert got["ln_bt_fw"] is False
    assert got["rDt_e_s"] == pytest.approx(117.39130434)
    assert got["rn_Dt_s"] == pytest.approx(2700.0)
    assert got["ln_1st_euler"] is False


def test_ocean_output_parser_refuses_an_incomplete_log(tmp_path):
    """A missing value must ABORT, never fall back to a default."""
    log = tmp_path / "ocean.output"
    log.write_text("      Barotropic time filter => nn_bt_flt =            2\n")
    with pytest.raises(SystemExit, match="nn_e"):
        bac.read_ocean_output(str(log))
    log.write_text(
        "      Barotropic time filter => nn_bt_flt =            2\n"
        "                               in iterations nn_e =           23\n"
        "      Barotropic time steps => in seconds         =    117.4\n")
    with pytest.raises(SystemExit, match="ln_bt_fw"):
        bac.read_ocean_output(str(log))


def _dino_log(tmp_path):
    log = tmp_path / "ocean.output"
    log.write_text(
        "      Barotropic time filter => nn_bt_flt =            2\n"
        "                               in iterations nn_e =           23\n"
        "         ln_bt_fw=F => Centred integration of barotropic variables \n"
        "      Barotropic time steps => in seconds         =    117.39130434782609\n"
        "   ===>>>   Modified Leap-Frog (MLF) :   rDt =    5400.0000000000000\n"
        "      rn_Dt      = 2700.\n")
    return str(log)


def test_main_end_to_end_predicts_a_zero_offset(tmp_path):
    """MAIN's own arithmetic -- the number the write-up quotes.

    The centroid->time conversion (two clocks, two origins) lives only in
    ``main`` and was previously untested: a sign error in either origin would
    have passed the whole suite.
    """
    out = tmp_path / "r.json"
    assert bac.main(["--ocean-output", _dino_log(tmp_path),
                     "--json-out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["rn_Dt_s"] == pytest.approx(2700.0)
    assert res["nemo"]["nn_e"] == 23
    assert res["nemo"]["icycle"] == 68
    assert res["legoesm"]["n_loop"] == 68
    assert res["max_abs_weight_diff"] == 0.0
    # both committed free surfaces sit at the nominal AFTER time, t + 1 step
    assert res["nemo"]["primary_centroid_steps_rel_t"] == pytest.approx(1.0, abs=1e-12)
    assert res["legoesm"]["filter_centroid_steps_rel_t"] == pytest.approx(1.0, abs=1e-12)
    assert res["predicted_offset_steps"] == pytest.approx(0.0, abs=1e-12)


def test_main_refuses_when_the_two_compositions_are_not_the_same_object(
        tmp_path, monkeypatch):
    """THE GATE. A centroid match is necessary, NOT sufficient.

    Both the leap-frog window (24..68 from t-dt) and the forward-Euler one
    (1..45 from t) put their centroid at t+dt, so before this gate existed the
    probe printed the SAME '-0.000000' for a demonstrably wrong configuration
    and had no control that could make its headline nonzero.
    """
    real = bac.lego_card_settings
    monkeypatch.setattr(bac, "lego_card_settings",
                        lambda card=bac.CARD: {**real(card), "substep_scale": 1})
    with pytest.raises(SystemExit, match="NOT the same object"):
        bac.main(["--ocean-output", _dino_log(tmp_path)])


def test_main_refuses_an_inconsistent_clock(tmp_path):
    """rDt_e * nn_e must equal rn_Dt, or the log is not the run we think."""
    log = tmp_path / "bad.output"
    log.write_text(
        "      Barotropic time filter => nn_bt_flt =            2\n"
        "                               in iterations nn_e =           23\n"
        "         ln_bt_fw=F => Centred integration of barotropic variables \n"
        "      Barotropic time steps => in seconds         =    200.0\n"
        "   ===>>>   Modified Leap-Frog (MLF) :   rDt =    5400.0000000000000\n"
        "      rn_Dt      = 2700.\n")
    with pytest.raises(SystemExit, match="rDt_e"):
        bac.read_ocean_output(str(log))


# NEMO announces a forced Euler start in three different ways
# (src/OCE/DOM/domain.F90:399, :408, :416).  A guard that catches only the
# chattiest one lets the other two through silently, which is why all three
# are asserted here.
@pytest.mark.parametrize("marker", [
    "           an Euler initial time step is used : l_1st_euler is forced to .true. ",
    "      ==>>> ssh(Kbb) /= ssh(Kmm), l_1st_euler forced to .true. and ssh(Kbb) = ssh(Kmm)",
    "      ==>>>   forced euler first time-step",
    "      ln_1st_euler = T",
])
def test_main_refuses_every_euler_first_step_log(tmp_path, marker):
    """ts_wgt is called with ll_fw_start, which l_1st_euler forces TRUE."""
    log = tmp_path / "euler.output"
    log.write_text(
        "      Barotropic time filter => nn_bt_flt =            2\n"
        "                               in iterations nn_e =           23\n"
        "         ln_bt_fw=F => Centred integration of barotropic variables \n"
        "      Barotropic time steps => in seconds         =    117.39130434782609\n"
        "      rn_Dt      = 2700.\n"
        + marker + "\n")
    with pytest.raises(SystemExit, match="Euler first step"):
        bac.read_ocean_output(str(log))


def test_the_twins_own_log_is_accepted(tmp_path):
    """CONTROL: the guard must not fire on the log the result rests on."""
    got = bac.read_ocean_output(_dino_log(tmp_path))
    assert got["ln_1st_euler"] is False


def test_card_with_a_foreign_filter_is_refused(monkeypatch):
    """Printing boxcar centroids under a 'cosine' label is a wrong answer."""
    from legoesm.ocean.experiments.dino import DINO_RECIPES
    monkeypatch.setitem(DINO_RECIPES, "_fake",
                        {**DINO_RECIPES[bac.CARD],
                         "barotropic_time_filter": "cosine"})
    with pytest.raises(SystemExit, match="not the NEMO centred"):
        bac.lego_card_settings("_fake")
