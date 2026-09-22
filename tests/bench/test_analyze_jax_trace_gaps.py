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


def _summaries(events_per_rank):
    """Two ranks whose collectives are perfectly aligned but for a constant
    clock offset, which is the case the analyzer is supposed to handle."""
    out = {}
    for rank, n in events_per_rank.items():
        offset = 1000.0 if rank == "rank1" else 0.0
        out[rank] = {"collectives": [(f"nccl_{i}", offset + i * 100.0,
                                      offset + i * 100.0 + 40.0)
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
