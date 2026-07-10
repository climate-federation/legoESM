"""Unit tests for the CLM-style output-tape harness."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.output_tapes import (
    TapeSpec,
    accumulate_tape_step,
    build_slot_indices,
    default_config_path,
    finalize_tape,
    init_tape_accumulator,
    load_output_config,
)

jax.config.update("jax_enable_x64", True)

_SEC_PER_DAY = 86400.0
_SEC_PER_HOUR = 3600.0


def test_default_config_ships_and_loads():
    # The default file exists in the repo and parses into >=2 tapes with
    # non-empty variable lists — sanity, not physics.
    p = default_config_path()
    assert p.exists(), p
    tapes = load_output_config(None)
    assert len(tapes) >= 2
    for t in tapes:
        assert isinstance(t, TapeSpec)
        assert t.vars
        assert t.freq in ("step", "hourly", "daily", "monthly", "annual")
        assert t.average in ("mean", "inst")


def test_load_output_config_rejects_bad_freq(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("tapes:\n  - name: h0\n    freq: fortnightly\n    vars: [T_sfc]\n")
    with pytest.raises(ValueError, match="freq"):
        load_output_config(str(cfg))


def test_load_output_config_rejects_bad_average(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("tapes:\n  - name: h0\n    freq: monthly\n    average: median\n"
                   "    vars: [T_sfc]\n")
    with pytest.raises(ValueError, match="average"):
        load_output_config(str(cfg))


def test_slot_indices_step():
    t = np.arange(6, dtype=np.float64) * _SEC_PER_HOUR
    idx, n, times = build_slot_indices(t, "step")
    assert n == 6
    assert idx.tolist() == list(range(6))
    np.testing.assert_allclose(times, t)


def test_slot_indices_hourly_with_sub_hour_dt():
    # dt = 30 min: 4 steps span 2 hours -> 2 slots, 2 steps each
    t = np.arange(4, dtype=np.float64) * 1800.0
    idx, n, times = build_slot_indices(t, "hourly")
    assert n == 2
    assert idx.tolist() == [0, 0, 1, 1]
    np.testing.assert_allclose(times, np.array([0.5, 1.5]) * _SEC_PER_HOUR)


def test_slot_indices_monthly_starts_at_zero_and_rolls_over():
    # Steps at day 5 (Jan), day 35 (Feb), day 400 (Feb next year).
    t = np.array([5.0, 35.0, 400.0]) * _SEC_PER_DAY
    idx, n, _ = build_slot_indices(t, "monthly")
    # slot = year_offset * 12 + month
    # Jan of year 0 = 0; Feb of year 0 = 1; day-400 -> year 1 doy 35 -> Feb of year 1 = 13
    assert idx.tolist() == [0, 1, 13]
    assert n == 14


def test_accumulate_mean_and_finalize():
    tape = TapeSpec(name="h", freq="hourly", average="mean", vars=("v",))
    n_slots, ncol = 3, 4
    carry = init_tape_accumulator(tape, n_slots, ncol)
    # 3 steps all falling in slot 1, with values 10, 20, 30 per column
    for val in (10.0, 20.0, 30.0):
        carry = accumulate_tape_step(
            carry, tape, jnp.int32(1),
            {"v": jnp.full(ncol, val)})
    out = finalize_tape(carry, tape)
    assert out["v"].shape == (n_slots, ncol)
    np.testing.assert_allclose(out["v"][1], 20.0)                  # mean of 10,20,30
    np.testing.assert_allclose(out["v"][0], 0.0)                   # untouched slots stay zero
    np.testing.assert_allclose(out["v"][2], 0.0)


def test_accumulate_inst_takes_last_value():
    tape = TapeSpec(name="h", freq="hourly", average="inst", vars=("v",))
    n_slots, ncol = 2, 3
    carry = init_tape_accumulator(tape, n_slots, ncol)
    for val in (10.0, 20.0, 30.0):
        carry = accumulate_tape_step(
            carry, tape, jnp.int32(0),
            {"v": jnp.full(ncol, val)})
    out = finalize_tape(carry, tape)
    np.testing.assert_allclose(out["v"][0], 30.0)                  # LAST value survives


def test_slot_indices_no_phantom_leading_slots_at_nonzero_start():
    # A run starting at DOY 196 must NOT emit ~196 leading zero-filled daily
    # records: slots are rebased to the first occupied slot, and slot_times stay
    # in the true year_start frame (first slot centred on DOY 196.5).
    day = 86400.0
    t = 196.0 * day + np.arange(0, 3 * day, 3600.0)     # 3 days hourly from DOY 196
    slot_idx, n_slots, slot_times = build_slot_indices(t, "daily")
    assert n_slots == 3                                  # 3 real days, not 199
    assert int(np.asarray(slot_idx).min()) == 0          # rebased to slot 0
    assert slot_times[0] / day == pytest.approx(196.5)   # true first-day time preserved


def test_slot_indices_start_at_zero_unchanged():
    # Backward-compat: a run starting at slot 0 is byte-identical to before.
    day = 86400.0
    t = np.arange(0, 2 * day, 3600.0)
    slot_idx, n_slots, slot_times = build_slot_indices(t, "daily")
    assert n_slots == 2
    assert int(np.asarray(slot_idx).min()) == 0
    assert slot_times[0] / day == pytest.approx(0.5)


def test_slot_indices_monthly_nonzero_start_rebased():
    # Monthly tape starting mid-year: first slot is the start month, not January.
    day = 86400.0
    t = 196.0 * day + np.arange(0, 40 * day, day)        # ~40 days from mid-July
    slot_idx, n_slots, slot_times = build_slot_indices(t, "monthly")
    assert int(np.asarray(slot_idx).min()) == 0
    assert n_slots <= 3                                   # spans ~2 months, no Jan..Jun phantoms
    assert slot_times[0] / day > 180.0                   # first slot is mid-year, not Jan
