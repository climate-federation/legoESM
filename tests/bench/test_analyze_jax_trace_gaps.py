"""The trace analyzer must REFUSE rather than pair the wrong events.

This instrument decides what the campaign believes about where a step goes,
and its dangerous failure is not a crash but a confident wrong number: an
event sequence that does not match the lane's step, indexed anyway with the
wrong period, yields skew figures that look fine and mean nothing.  Two
reviewers found exactly that, so both refusals are pinned here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parents[2] / "scripts" / "bench"
sys.path.insert(0, str(_BENCH_DIR))
_spec = importlib.util.spec_from_file_location(
    "analyze_jax_trace_gaps", _BENCH_DIR / "analyze_jax_trace_gaps.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _name(i, period):
    """Pairwise exchange everywhere but the trailing slot of each step."""
    return "nccl_allreduce" if (i % period) == period - 1 else "nccl_sendrecv"


def _summaries(events_per_rank, period=2, offsets=None):
    """Two ranks whose collectives are perfectly aligned but for a constant
    clock offset, which is the case the analyzer is supposed to handle."""
    out = {}
    for rank, n in events_per_rank.items():
        base = 1000.0 if rank == "rank1" else 0.0
        out[rank] = {"collectives": [
            (_name(i, period),
             base + (offsets[i] if offsets and rank == "rank1" else 0.0) + i * 100.0,
             base + (offsets[i] if offsets and rank == "rank1" else 0.0) + i * 100.0 + 40.0)
            for i in range(n)]}
    return out


# One fill, one partner each way, so the model predicts steps * (1 * 1 + 1).
_PM = {"n_rounds": 1, "ranks": {"0": [[0, 1]], "1": [[0, 0]]}}


def test_a_matching_sequence_is_quoted(capsys):
    n = 4 * (1 * 1 + 1)
    ok = mod.partner_aware_spread(_summaries({"rank0": n, "rank1": n}), _PM,
                                  steps=4, fills=1)
    text = capsys.readouterr().out
    assert ok is True
    assert "arrival skew  median" in text
    assert "NOT quotable" not in text


def test_a_longer_sequence_is_refused_not_indexed(capsys):
    """The ocean's step is longer than the atmosphere's.  Having ENOUGH
    events is how an incompatible sequence used to be accepted silently."""
    n = 4 * (1 * 1 + 1) + 17
    ok = mod.partner_aware_spread(_summaries({"rank0": n, "rank1": n}), _PM,
                                  steps=4, fills=1)
    text = capsys.readouterr().out
    assert ok is False
    assert "NOT quotable" in text
    assert "arrival skew  median" not in text   # the numbers, not the refusal


def test_unaligned_clocks_withhold_the_skew(capsys):
    """Matched participants finish together.  When they do not, the clocks
    are not aligned and every skew number from them is meaningless."""
    n = 4 * (1 * 1 + 1)
    s = _summaries({"rank0": n, "rank1": n})
    # Stretch rank1's kernels so the end residual cannot collapse.
    s["rank1"]["collectives"] = [(nm, a, b + i * 60.0)
                                 for i, (nm, a, b)
                                 in enumerate(s["rank1"]["collectives"])]
    ok = mod.partner_aware_spread(s, _PM, steps=4, fills=1)
    text = capsys.readouterr().out
    assert ok is False
    assert "CALIBRATION FAILED" in text
    assert "arrival skew  median" not in text   # the numbers, not the refusal


def test_a_clock_fault_in_a_minority_of_pairs_is_caught(capsys):
    """The counterexample a reviewer built against the first gate: three
    pairs aligned and one off by ten milliseconds.  A median-based gate sees
    a zero residual and quotes a ten-millisecond skew with a straight face."""
    n = 4 * (1 * 1 + 1)
    offs = [0.0] * n
    offs[-2] = 10000.0          # one pair's clock jumps
    s = _summaries({"rank0": n, "rank1": n}, offsets=offs)
    ok = mod.partner_aware_spread(s, _PM, steps=4, fills=1)
    text = capsys.readouterr().out
    assert ok is False
    assert "CALIBRATION FAILED" in text
    assert "arrival skew  median" not in text


def test_a_sequence_of_the_right_length_but_wrong_types_is_refused(capsys):
    """Count equality is necessary, not sufficient.  A lane whose step has a
    different shape can produce the same number of events."""
    n = 4 * (1 * 1 + 1)
    s = _summaries({"rank0": n, "rank1": n})
    # Every slot a pairwise exchange: no trailing reduction anywhere.
    for rank in s:
        s[rank]["collectives"] = [("nccl_sendrecv", a, b)
                                  for _, a, b in s[rank]["collectives"]]
    ok = mod.partner_aware_spread(s, _PM, steps=4, fills=1)
    text = capsys.readouterr().out
    assert ok is False
    assert "not a reduction" in text
    assert "arrival skew  median" not in text


def test_the_census_reports_a_period_only_when_one_exists():
    """The census exists to LEARN a lane's step, so it must never invent a
    period. An index model built on a period the sequence does not have pairs
    unrelated events, which is the defect this whole file is about."""
    step = ["nccl_sendrecv", "nccl_sendrecv", "nccl_allreduce"]
    assert mod.exact_period(step * 4) == 3
    assert mod.exact_period(["a"] * 6) == 1
    assert mod.exact_period([]) is None
    # No repetition at all means no period. Half a sequence is not one,
    # because a period has to describe the WHOLE sequence.
    assert mod.exact_period(["a", "b", "c", "d"]) is None
    # A truncated final repetition still HAS the period, and reporting it is
    # right: the capture was cut mid-step. The separate exact-count check is
    # what refuses to index such a trace, not this one.
    assert mod.exact_period(step * 4 + step[:1]) == 3


# --- per-family timing: the number that decides where a step goes -------
# A family total that double counts is worse than no total, because the
# share it prints looks authoritative.  The two ways it can double count
# are pinned here: the profiler's zero-width "end: <op>" twin, and
# overlapping events summed instead of unioned.

def _ev(name, ts, dur):
    return {"name": name, "ts": float(ts), "dur": float(dur)}


def test_the_end_marker_is_not_a_second_instruction():
    assert mod.collective_family("all-to-all.18") == "all-to-all"
    assert mod.collective_family("end: all-to-all.18") is None
    assert mod.collective_family("ppermute.1295") == "permute"
    assert mod.collective_family("end: ppermute.1295") is None
    assert mod.collective_family("all-gather.7") == "all-gather"
    assert mod.collective_family("fusion.42") is None

    evs = [_ev("all-to-all.18", 0, 100), _ev("end: all-to-all.18", 100, 0)]
    out = mod.time_by_family(evs, steps=1)
    assert out["families"]["all-to-all"]["instructions"] == 1


def test_overlapping_events_are_unioned_not_summed():
    # Three all-gathers on three device threads, all running 0-100 us.
    # Summing gives 300 us; the honest answer is 100 us of wall time.
    evs = [_ev(f"all-gather.{i}", 0, 100) for i in range(3)]
    out = mod.time_by_family(evs, steps=1)
    fam = out["families"]["all-gather"]
    assert fam["instructions"] == 3
    assert abs(fam["ms_per_step"] - 0.1) < 1e-9, fam


def test_disjoint_events_add_and_steps_divide():
    evs = [_ev("all-gather.0", 0, 100), _ev("all-gather.1", 500, 100)]
    out = mod.time_by_family(evs, steps=2)
    assert abs(out["families"]["all-gather"]["ms_per_step"] - 0.1) < 1e-9


def test_a_capture_with_no_named_collectives_reports_none():
    out = mod.time_by_family([_ev("fusion.1", 0, 100)], steps=1)
    assert out["families"] == {}


def test_a_nonpositive_step_count_is_refused():
    import pytest
    with pytest.raises(ValueError):
        mod.time_by_family([_ev("all-gather.0", 0, 100)], steps=0)


def test_a_fusion_named_after_an_opcode_is_not_that_family():
    # codex: a substring match swallows any op whose NAME contains an
    # opcode, so a fusion would be counted as a gather and its (large)
    # duration attributed to communication.
    assert mod.collective_family("fusion.all-gather.18") is None
    assert mod.collective_family("all-gather.18") == "all-gather"
    assert mod.collective_family("custom-call.3") is None


def test_a_renamed_family_is_refused_not_reported_as_zero():
    # The dangerous case: the program contains gathers, the profiler emits
    # them under some other name, and the mode prints a confident zero.
    evs = [_ev("all-to-all.1", 0, 10)]
    out = mod.time_by_family(evs, steps=1,
                             expect={"all-to-all": 1, "all-gather": 2})
    assert out["census_ok"] is False
    assert any("all-gather" in p for p in out["census_problems"]), out


def test_the_census_multiplies_by_devices_and_steps():
    evs = [_ev(f"all-gather.{i}", 10 * i, 5) for i in range(12)]
    ok = mod.time_by_family(evs, steps=3, expect={"all-gather": 2},
                            devices_per_rank=2)
    assert ok["census_ok"] is True, ok["census_problems"]
    bad = mod.time_by_family(evs, steps=3, expect={"all-gather": 2},
                             devices_per_rank=1)
    assert bad["census_ok"] is False


def test_an_unexpected_family_in_the_capture_is_flagged():
    evs = [_ev("all-gather.1", 0, 10), _ev("all-reduce.1", 20, 10)]
    out = mod.time_by_family(evs, steps=1, expect={"all-gather": 1})
    assert out["census_ok"] is False
    assert any("all-reduce" in p for p in out["census_problems"])


def test_a_zero_width_capture_prints_instead_of_crashing(capsys, tmp_path, monkeypatch):
    # share_of_span is None when the span is zero; the print path used to
    # multiply it by 100 and take the whole run down with a TypeError.
    out = mod.time_by_family([_ev("all-gather.1", 5, 0)], steps=1)
    assert out["families"]["all-gather"]["share_of_span"] is None

    rank = tmp_path / "rank0"
    rank.mkdir()
    monkeypatch.setattr(mod, "load_trace", lambda rd: [])
    monkeypatch.setattr(mod, "device_events",
                        lambda evs: [_ev("all-gather.1", 5, 0)])
    monkeypatch.setattr(
        sys, "argv",
        ["analyze", "--trace-root", str(tmp_path), "--steps", "1",
         "--time-by-family"])
    assert mod.main() == 0
    assert "n/a % of span" in capsys.readouterr().out


def test_the_printing_path_refuses_on_a_census_mismatch(capsys, tmp_path, monkeypatch):
    rank = tmp_path / "rank0"
    rank.mkdir()
    monkeypatch.setattr(mod, "load_trace", lambda rd: [])
    monkeypatch.setattr(mod, "device_events",
                        lambda evs: [_ev("all-gather.1", 0, 10)])
    monkeypatch.setattr(
        sys, "argv",
        ["analyze", "--trace-root", str(tmp_path), "--steps", "1",
         "--time-by-family", "--expect-census", '{"all-gather": 2}'])
    assert mod.main() == 3
    assert "REFUSED" in capsys.readouterr().out


def test_a_capture_missing_ranks_is_refused(capsys, tmp_path, monkeypatch):
    # codex: with only rank0 present the mode exited 0, so a family that
    # lives on the missing ranks read as absent.
    (tmp_path / "rank0").mkdir()
    monkeypatch.setattr(mod, "load_trace", lambda rd: [])
    monkeypatch.setattr(mod, "device_events",
                        lambda evs: [_ev("all-gather.1", 0, 10)])
    monkeypatch.setattr(
        sys, "argv",
        ["analyze", "--trace-root", str(tmp_path), "--steps", "1",
         "--time-by-family", "--expect-ranks", "4"])
    assert mod.main() == 3
    assert "REFUSED" in capsys.readouterr().out

    monkeypatch.setattr(
        sys, "argv",
        ["analyze", "--trace-root", str(tmp_path), "--steps", "1",
         "--time-by-family", "--expect-ranks", "1"])
    assert mod.main() == 0


def test_gate_flags_without_their_mode_are_refused(tmp_path, monkeypatch):
    # GLM: a flag that silently does nothing is a gate the caller believes
    # they armed.
    (tmp_path / "rank0").mkdir()
    for flag, value in (("--expect-ranks", "4"),
                        ("--expect-census", '{"all-gather": 1}'),
                        ("--devices-per-rank", "24")):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze", "--trace-root", str(tmp_path), flag, value])
        try:
            mod.main()
        except SystemExit as exc:
            assert flag in str(exc), (flag, exc)
        else:
            raise AssertionError(f"{flag} was silently ignored")
