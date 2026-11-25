"""Round 302: per-region stage-history scorer."""

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round302_region_history as gate,
)


def _fields(seed=302):
    rng = np.random.default_rng(seed)
    out = {f: rng.normal(1.0, 0.5, (6, 4, 3)) for f in ("u", "v", "T", "S")}
    out["ssh"] = rng.normal(0.0, 0.2, (6, 4))
    return out


def _masks():
    masks = {f: np.ones((6, 4, 3), bool) for f in ("u", "v", "T", "S")}
    masks["ssh"] = np.ones((6, 4), bool)
    masks["u"][-1, 0, 0] = False
    return masks


def test_fold_band_is_last_three_rows_and_relative_rms_is_against_nemo_signal():
    oracle, masks = _fields(), _masks()
    cand = {k: v.copy() for k, v in oracle.items()}
    cand["u"][-2, 1, 1] += 3.0
    cand["u"][-1, 0, 0] += 100.0  # dry in NEMO's mask: never scored
    rows = gate.score_stage(cand, oracle, masks)
    fold, interior = rows["u"]["fold_band"], rows["u"]["interior"]
    assert (fold["unequal"], interior["unequal"]) == (1, 0)
    assert fold["support"] == 3 * 4 * 3 - 1
    signal = oracle["u"][-3:][masks["u"][-3:]]
    expected = np.sqrt(9.0 / signal.size) / np.sqrt(np.mean(signal ** 2))
    assert np.isclose(fold["relative_rms"], expected, rtol=1e-12)
    assert fold["over_bar"] and not interior["over_bar"]
    assert not any(rows[f][r]["over_bar"] for f in ("v", "ssh", "T", "S")
                   for r in ("fold_band", "interior"))


def test_one_ulp_is_not_over_the_bar_but_counts_unequal():
    oracle, masks = _fields(), _masks()
    cand = {k: v.copy() for k, v in oracle.items()}
    cand["T"][0, 0, 0] = np.nextafter(cand["T"][0, 0, 0], np.inf)
    row = gate.score_stage(cand, oracle, masks)["T"]["interior"]
    assert row["unequal"] == 1 and not row["over_bar"]


def test_first_over_bar_is_earliest_stage_per_field_and_region():
    oracle, masks = _fields(), _masks()
    bad = {k: v.copy() for k, v in oracle.items()}
    bad["v"][0, 0, 0] += 1.0
    history = [{"kt": 1, "stage": s, "fields": gate.score_stage(
        bad if s >= 2 else oracle, oracle, masks)} for s in range(4)]
    first = gate.first_over_bar(history)
    assert first["v"]["interior"] == {"kt": 1, "stage": 2}
    assert first["v"]["fold_band"] is None and first["u"]["interior"] is None
