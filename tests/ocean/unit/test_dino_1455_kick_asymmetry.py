"""#1455 kick-asymmetry discriminator: the arithmetic and the two perturbation
tools the verdict rests on.

The probe owns three pieces of arithmetic that are not imported from a
recorded harness -- the resolvability bar for the collapse factor, the
legoESM/NEMO spread ratio, and the registered three-way outcome rule.  The two
perturbation tools own one property each: exactly the intended fields move,
and (for the two-level convention) BOTH time levels get the SAME draw.

Every check here is written so it FAILS when the feature it covers is removed;
the same-draw checks in particular are run against a deliberately WRONG
implementation to prove they are not vacuous.
"""
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = REPO_ROOT / "scripts" / "validate" / "ocean_fidelity" / "dino_1226"
EPS = 1e-14


@pytest.fixture(scope="module")
def K():
    sys.path.insert(0, str(PROBE_DIR))
    try:
        import kick_asymmetry
        return kick_asymmetry
    finally:
        try:
            sys.path.remove(str(PROBE_DIR))
        except ValueError:
            pass


@pytest.fixture(scope="module")
def PN():
    sys.path.insert(0, str(PROBE_DIR))
    try:
        import perturb_nemo_tn_90d
        return perturb_nemo_tn_90d
    finally:
        try:
            sys.path.remove(str(PROBE_DIR))
        except ValueError:
            pass


# ------------------------------------------------------- the probe's own math ---
def test_probe_self_check_passes(K):
    assert K._self_check() == 0


def test_resolvability_bar_is_the_n4_estimator_error_not_a_round_number(K):
    """F is a ratio of two ratios of four independent spread estimates, so its
    log carries sqrt(4)/sqrt(2(n-1)).  A hardcoded 5.0 would pass a sloppy
    test; the exact value is 4.955."""
    assert K.N_MEM == 4
    expected = float(np.exp(1.96 * np.sqrt(4.0) / np.sqrt(2 * (K.N_MEM - 1))))
    assert abs(K.F_RESOLVABLE - expected) < 1e-12
    assert abs(K.F_RESOLVABLE - 4.9547) < 1e-3


def test_ratio_is_lego_over_nemo_and_nan_on_a_zero_denominator(K):
    rows = {"lego": {i: {90: {"x": 2.0 * i}} for i in range(4)},
            "nemo": {i: {90: {"x": 1.0 * i}} for i in range(4)}}
    assert abs(K.ratio(rows, "x", 90) - 2.0) < 1e-12
    rows["nemo"] = {i: {90: {"x": 5.0}} for i in range(4)}
    assert np.isnan(K.ratio(rows, "x", 90))     # not inf, not a big float


@pytest.mark.parametrize("f, r2, r1, want", [
    (100.0, 3.0, 300.0, "CONFIRMS H1"),
    (100.0, 30.0, 3000.0, "PARTIAL"),
    # Below the resolvability bar the outcome depends on how big a collapse H1
    # needs: if the 95% upper limit on F could still deliver it, the run said
    # NOTHING. The pre-amendment rule called both of these REFUTES.
    (1.0, 30.0, 30.0, "NOT RESOLVED"),      # needs 3.0x, upper limit 4.95x
    (1.0, 100.0, 100.0, "REFUTES H1"),      # needs 10x, upper limit 4.95x
    (float("nan"), 3.0, 300.0, "INDETERMINATE"),
    (100.0, float("nan"), 300.0, "INDETERMINATE"),
    (100.0, 3.0, float("nan"), "INDETERMINATE"),
])
def test_registered_outcome_rule(K, f, r2, r1, want):
    assert K.verdict(f, r2, r1) == want


def test_a_below_bar_result_is_never_silently_a_refutation(K):
    """The amendment that matters (code review B2): the resolvability bar is
    the width of the null band, so everything under it is NO INFORMATION.
    Turning that into REFUTES was a positive claim the statistic cannot carry,
    and REFUTES is the branch the pre-registration attaches a consequence to."""
    # a genuine 4x collapse -- real physics, under the n=4 bar
    assert K.verdict(4.0, 25.0, 100.0) == "NOT RESOLVED"


def test_outcome_rule_boundary_is_inclusive_as_registered(K):
    assert K.verdict(K.F_RESOLVABLE, K.R2_TWO_SIDED, 100.0) == "CONFIRMS H1"
    assert K.verdict(K.F_RESOLVABLE, K.R2_TWO_SIDED * 1.0001, 100.0) == "PARTIAL"


# ------------------------------------------------- the leapfrog closed form ---
def test_predicted_arm_ratio_matches_the_asselin_eigen_decomposition(K):
    """A one-level kick (0, eps) keeps (1-2g)/(2(1-g)) of itself on the
    physical eigenvector; a two-level kick (eps, eps) keeps all of it. The arm
    effect is the reciprocal of the first, per model."""
    for g in (0.0, 0.05, 0.1, 0.2):
        one_level_survives = (1 - 2 * g) / (2 * (1 - g))
        assert abs(K.predicted_arm_ratio(g) - 1.0 / one_level_survives) < 1e-12
    assert abs(K.predicted_arm_ratio(0.1) - 2.25) < 1e-12
    with pytest.raises(SystemExit):
        K.predicted_arm_ratio(0.5)      # computational mode undamped


def test_the_card_and_the_closed_form_agree_on_gamma(K):
    assert abs(K.asselin_gamma_lego() - 0.1) < 1e-12


def test_collapse_factor_is_the_ratio_of_the_two_arm_ratios(K):
    """F = R1/R2 and F = arm(nemo)/arm(lego) are the same number. The two
    tables are two views of one measurement, so a discrepancy is a bug."""
    d = K.SCORE_DAY
    r1 = {"lego": {i: {d: {"x": 1.0 * i}} for i in range(4)},
          "nemo": {i: {d: {"x": 2.0 * i}} for i in range(4)}}
    r2 = {"lego": {i: {d: {"x": 7.0 * i}} for i in range(4)},
          "nemo": {i: {d: {"x": 3.0 * i}} for i in range(4)}}
    direct = K.ratio(r1, "x", d) / K.ratio(r2, "x", d)
    via_arms = (K.arm_ratio(r1, r2, "nemo", "x")
                / K.arm_ratio(r1, r2, "lego", "x"))
    assert abs(direct - via_arms) < 1e-12


def test_median_propagates_nan_but_honours_the_quantum_exclusion(K):
    assert np.isnan(K.median_over({"a": 1.0, "b": float("nan")}, set()))
    assert abs(K.median_over({"a": 1.0, "b": float("nan"), "c": 3.0}, {"b"}) - 2.0) < 1e-12
    with pytest.raises(SystemExit):
        K.median_over({"a": 1.0}, {"a"})


def test_only_the_perturbed_members_get_the_two_level_flag(K, tmp_path):
    """The control must NOT carry --perturb-both-levels (it carries no seed
    either), and every perturbed member must carry BOTH flags -- a seed
    without the flag would silently run the OLD convention and the whole arm
    would measure nothing."""
    m0 = K.lego_cmd(str(tmp_path), 0)
    assert "--perturb-seed" not in m0 and "--perturb-both-levels" not in m0
    assert "--bridge-before" in m0        # no before-level T without it
    for i in (1, 2, 3):
        cmd = K.lego_cmd(str(tmp_path), i)
        assert "--perturb-both-levels" in cmd
        assert cmd[cmd.index("--perturb-seed") + 1] == str(K.SEEDS[i])
        assert "--bridge-before" in cmd
        # the ONE-level arm runs the same command WITHOUT that flag, and the
        # two command lines must otherwise be identical -- the arms differ in
        # exactly one token.
        one = K.lego_cmd(str(tmp_path), i, both_levels=False)
        assert "--perturb-both-levels" not in one
        assert [t for t in cmd if t != "--perturb-both-levels"] == one
    assert K.SEEDS == (None, 1, 2, 3)     # the recorded seed convention


def test_the_two_arms_are_scored_on_different_directories(K, tmp_path):
    one, two = K.arms(str(tmp_path / "one"), str(tmp_path / "two"))
    assert one.lego_dir != two.lego_dir
    assert one.nemo_dir_fn(1) != two.nemo_dir_fn(1)
    # Both arms are now 90-day legoESM runs at the SAME source revision (code
    # review B1: reusing the recorded 360-day members compared two model
    # revisions). Only the NEMO side of the one-level arm is reused.
    assert one.lego_nsteps == two.lego_nsteps == 32 * K.N_DAYS
    assert one.nemo_log_glob != two.nemo_log_glob


# ---------------------------------------------- the NEMO perturbation tool ---
def _tiny_restart(path, extra_var=True):
    """A 3-variable stand-in for the DINO restart: tn, tb (both with land
    zeros) and one variable that must never move."""
    nc = pytest.importorskip("netCDF4")
    d = nc.Dataset(path, "w")
    d.createDimension("z", 3)
    d.createDimension("y", 4)
    d.createDimension("x", 5)
    rng = np.random.default_rng(0)
    base = 5.0 + rng.random((3, 4, 5))
    base[:, 0, :] = 0.0                    # "land": exactly zero, must stay zero
    for name, off in (("tn", 0.0), ("tb", 1e-3)):
        v = d.createVariable(name, "f8", ("z", "y", "x"))
        v[:] = np.where(base == 0.0, 0.0, base + off)
        v.units = "degC"
    if extra_var:
        u = d.createVariable("un", "f8", ("z", "y", "x"))
        u[:] = rng.random((3, 4, 5))
    d.title = "tiny"
    d.close()


def test_nemo_one_level_moves_tn_only(PN, tmp_path):
    src = tmp_path / "src.nc"
    _tiny_restart(src)
    rep = PN.perturb(1, tmp_path / "one", src=src)
    assert rep["targets"] == ("tn",)
    assert 0 < rep["max_rel_tn_diff"] < 1e-12
    assert 0 < rep["n_tn_changed"] < rep["n_tn_total"]     # land cells stay put
    assert rep["other_vars_bit_identical"] and rep["attrs_bit_identical"]
    assert rep["max_other_var_diff"] == 0.0
    nc = pytest.importorskip("netCDF4")
    a = np.asarray(nc.Dataset(src)["tb"][:])
    b = np.asarray(nc.Dataset(tmp_path / "one" / "src.nc")["tb"][:])
    assert np.array_equal(a, b), "the one-level kick moved tb"


def test_nemo_two_level_moves_both_with_the_same_relative_draw(PN, tmp_path):
    src = tmp_path / "src.nc"
    _tiny_restart(src)
    rep = PN.perturb(1, tmp_path / "two", src=src, both_levels=True)
    assert rep["targets"] == ("tn", "tb")
    for name in ("tn", "tb"):
        pt = rep["per_target"][name]
        assert 0 < pt["max_rel_diff"] < 1e-12
        assert 0 < pt["n_changed"] < pt["n_total"]
    assert rep["other_vars_bit_identical"] and rep["attrs_bit_identical"]
    assert _same_draw_deviation(PN, src, tmp_path / "two" / "src.nc") < 0.05


def test_same_seed_gives_the_same_tn_in_both_conventions(PN, tmp_path):
    """The two arms must differ ONLY in whether tb moved.  If the two-level
    path consumed the generator differently, tn would differ too and the arms
    would differ in more than one variable."""
    src = tmp_path / "src.nc"
    _tiny_restart(src)
    PN.perturb(1, tmp_path / "one", src=src)
    PN.perturb(1, tmp_path / "two", src=src, both_levels=True)
    nc = pytest.importorskip("netCDF4")
    a = np.asarray(nc.Dataset(tmp_path / "one" / "src.nc")["tn"][:])
    b = np.asarray(nc.Dataset(tmp_path / "two" / "src.nc")["tn"][:])
    assert np.array_equal(a, b)


def _same_draw_deviation(PN, src, dst):
    """max |rel(tn) - rel(tb)| / max |rel(tn)| -- the statistic the tool's own
    self-check asserts on."""
    nc = pytest.importorskip("netCDF4")
    s, r = nc.Dataset(src), nc.Dataset(dst)
    rel = {}
    for name in ("tn", "tb"):
        a = np.asarray(s[name][:], dtype=np.float64)
        b = np.asarray(r[name][:], dtype=np.float64)
        rel[name] = np.where(a != 0, (b - a) / np.where(a != 0, a, 1.0), np.nan)
    both = np.isfinite(rel["tn"]) & np.isfinite(rel["tb"])
    return float(np.max(np.abs(rel["tn"][both] - rel["tb"][both]))
                 / np.nanmax(np.abs(rel["tn"])))


def test_the_same_draw_check_is_not_vacuous(PN, tmp_path):
    """PROOF that the statistic above can fail: build a restart perturbed with
    two INDEPENDENT draws -- the defect the reused-draw code exists to avoid --
    and confirm the deviation blows past the bar.  Without this the same-draw
    assertion could be a tautology that no implementation could fail."""
    nc = pytest.importorskip("netCDF4")
    src = tmp_path / "src.nc"
    _tiny_restart(src)
    import shutil
    bad = tmp_path / "bad.nc"
    shutil.copy2(src, bad)
    rng = np.random.default_rng(1)
    d = nc.Dataset(bad, "a")
    for name in ("tn", "tb"):
        v = np.asarray(d[name][:], dtype=np.float64)
        d[name][:] = v * (1.0 + EPS * rng.standard_normal(v.shape))  # SECOND draw
    d.close()
    assert _same_draw_deviation(PN, src, bad) > 0.5


def test_a_touched_bystander_variable_raises(PN, tmp_path, monkeypatch):
    """The receipt must catch a perturbation that leaks outside its targets."""
    nc = pytest.importorskip("netCDF4")
    src = tmp_path / "src.nc"
    _tiny_restart(src)
    real = PN.perturb

    def leaky(seed, dest_dir, src=src, both_levels=False):
        out = real(seed, dest_dir, src=src, both_levels=both_levels)
        return out

    # Perturb legitimately, then move a bystander behind the tool's back and
    # re-verify: _verify is what the receipt rests on, so it is what is tested.
    rep_dir = tmp_path / "leak"
    leaky(1, rep_dir)
    d = nc.Dataset(rep_dir / "src.nc", "a")
    d["un"][0, 0, 0] = float(np.asarray(d["un"][0, 0, 0])) + 1.0
    d.close()
    rep = PN._verify(src, rep_dir / "src.nc", ("tn",))
    assert not rep["other_vars_bit_identical"]
    assert "un" in rep["changed_vars_besides_targets"]


def test_a_changed_attribute_raises(PN, tmp_path):
    nc = pytest.importorskip("netCDF4")
    src = tmp_path / "src.nc"
    _tiny_restart(src)
    PN.perturb(1, tmp_path / "attr", src=src)
    d = nc.Dataset(tmp_path / "attr" / "src.nc", "a")
    d.title = "tampered"
    d.close()
    assert not PN._verify(src, tmp_path / "attr" / "src.nc", ("tn",))["attrs_bit_identical"]


# ------------------------------------------- the legoESM perturbation receipt ---
from legoesm.core.field import Field  # the REAL state field type: a

# registered JAX pytree, so tree_leaves(field) is [data].  A hand-rolled
# stand-in is NOT registered, reads as one opaque leaf, and would make the
# receipt look broken when it is fine -- the receipt is tested against the
# type it actually runs on.


class _State(NamedTuple):
    T: object
    T_before: object
    S: object


def _receipt():
    sys.path.insert(0, str(PROBE_DIR))
    try:
        import kamm_twin_90d
        return kamm_twin_90d._perturb_receipt
    finally:
        try:
            sys.path.remove(str(PROBE_DIR))
        except ValueError:
            pass


@pytest.fixture(scope="module")
def receipt():
    return _receipt()


def _st(t, tb, s):
    return _State(T=Field(np.asarray(t)), T_before=Field(np.asarray(tb)),
                  S=Field(np.asarray(s)))


def test_receipt_passes_when_exactly_the_named_fields_moved(receipt):
    pre = _st([1.0, 2.0], [1.0, 2.0], [30.0])
    post = _st([1.0, 2.0 + 1e-14], [1.0, 2.0], [30.0])
    assert receipt(pre, post, ["T"]) == 2          # T_before and S verified untouched
    post2 = _st([1.0, 2.0 + 1e-14], [1.0 + 1e-14, 2.0], [30.0])
    assert receipt(pre, post2, ["T", "T_before"]) == 1


def test_receipt_raises_when_a_bystander_moved(receipt):
    pre = _st([1.0, 2.0], [1.0, 2.0], [30.0])
    post = _st([1.0, 2.0 + 1e-14], [1.0, 2.0], [30.0 + 1e-13])
    with pytest.raises(SystemExit, match="S"):
        receipt(pre, post, ["T"])


def test_receipt_raises_when_a_target_did_not_move(receipt):
    """A kick that silently did nothing gives an ensemble spread of exactly
    zero -- the most flattering possible artifact -- so this must be fatal,
    not a warning."""
    pre = _st([1.0, 2.0], [1.0, 2.0], [30.0])
    post = _st([1.0, 2.0 + 1e-14], [1.0, 2.0], [30.0])
    with pytest.raises(SystemExit, match="T_before"):
        receipt(pre, post, ["T", "T_before"])


def test_receipt_handles_an_absent_before_level(receipt):
    """Without --bridge-before the state's before-level fields are None on both
    sides. tree_leaves(None) is empty, which must read as 'unchanged' rather
    than as a spurious difference -- but it must NOT be counted as a field the
    receipt verified, because nothing could have touched it (review N3)."""
    pre = _State(T=Field(np.asarray([1.0])), T_before=None,
                 S=Field(np.asarray([30.0])))
    post = _State(T=Field(np.asarray([1.0 + 1e-14])), T_before=None,
                  S=Field(np.asarray([30.0])))
    assert receipt(pre, post, ["T"]) == 1      # S only; the None is not counted


# ------------------------------------------- the legoESM kick's own properties ---
@pytest.fixture(scope="module")
def kick():
    sys.path.insert(0, str(PROBE_DIR))
    try:
        import kamm_twin_90d
        return kamm_twin_90d
    finally:
        try:
            sys.path.remove(str(PROBE_DIR))
        except ValueError:
            pass


def _kick_state(kick):
    rng = np.random.default_rng(7)
    t = 5.0 + rng.random((4, 5, 3))
    return _State(T=Field(np.asarray(t)),
                  T_before=Field(np.asarray(t + 1e-3)),
                  S=Field(np.asarray(35.0 + rng.random((4, 5, 3)))))


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (b - a) / a


def test_lego_two_level_kick_uses_the_same_draw_on_both_levels(kick):
    """The property the whole discriminator rests on, and until now proven
    only by reading the source (review N4). Two independent draws would still
    move both levels and still pass a naive 'both changed' test."""
    pre = _kick_state(kick)
    post, stamp = kick.apply_temperature_kick(pre, 11, True)
    rt = _rel(pre.T.data, post.T.data)
    rb = _rel(pre.T_before.data, post.T_before.data)
    scale = float(np.max(np.abs(rt)))
    dev = float(np.max(np.abs(rt - rb)))
    # round-off on the reconstructed relative change is ~1 ulp / draw ~ 5e-3;
    # independent draws would put this at ~1.
    assert dev / scale < 0.05, (dev, scale)
    assert "levels=now+before" in stamp


def test_the_lego_same_draw_check_is_not_vacuous(kick):
    """PROOF the statistic above can fail: two independent draws of the same
    size blow past the bar."""
    pre = _kick_state(kick)
    rng = np.random.default_rng(11)
    shape = np.asarray(pre.T.data).shape
    a = np.asarray(pre.T.data, dtype=np.float64) * (
        1 + kick.PERTURB_EPS * rng.standard_normal(shape))
    b = np.asarray(pre.T_before.data, dtype=np.float64) * (
        1 + kick.PERTURB_EPS * rng.standard_normal(shape))   # SECOND draw
    rt, rb = _rel(pre.T.data, a), _rel(pre.T_before.data, b)
    assert float(np.max(np.abs(rt - rb))) / float(np.max(np.abs(rt))) > 0.5


def test_same_seed_gives_the_same_now_level_in_both_conventions(kick):
    """The arms must differ ONLY in whether the before level moved. If the
    two-level path consumed the generator differently, the now level would
    differ too and the arms would differ in more than one variable."""
    one, _ = kick.apply_temperature_kick(_kick_state(kick), 11, False)
    two, _ = kick.apply_temperature_kick(_kick_state(kick), 11, True)
    assert np.array_equal(np.asarray(one.T.data), np.asarray(two.T.data))


def test_one_level_kick_leaves_the_before_level_bit_identical(kick):
    pre = _kick_state(kick)
    post, stamp = kick.apply_temperature_kick(pre, 11, False)
    assert np.array_equal(np.asarray(pre.T_before.data),
                          np.asarray(post.T_before.data))
    assert "levels=now" in stamp and "levels=now+before" not in stamp


def test_two_level_kick_without_a_before_level_raises(kick):
    pre = _State(T=Field(np.asarray([[1.0, 2.0]])), T_before=None,
                 S=Field(np.asarray([[35.0]])))
    with pytest.raises(SystemExit, match="bridge-before"):
        kick.apply_temperature_kick(pre, 11, True)


def test_both_models_kick_at_the_same_magnitude(kick, PN):
    """The two ensembles are only comparable if the kick is the same size on
    both sides; nothing else ties the two constants together (review N7)."""
    assert kick.PERTURB_EPS == PN.EPS == 1e-14
