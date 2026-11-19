"""Concurrent-STREAM window aggregate: hand-computed answer and refusals."""
import importlib.util
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "microbench" / "stream_sync.py"
_spec = importlib.util.spec_from_file_location("stream_sync", _P)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

# Two copies, 24 GB per rep (1000 M elements): copy 0 runs [100,101],[101,102],
# copy 1 [100.5,101.5],[101.5,102.5]; common window [100.5, 102] = 1.5 s.
# Prorated: 12+24+24+12 = 72 GB -> 48 GB/s.  Wholly inside: 2 reps -> 32 GB/s.
# Medians: 24 GB/s per copy -> 48 GB/s.
LOG = ["rank=0 Mel=1000 REP 0 100.0 101.0", "rank=0 Mel=1000 REP 1 101.0 102.0",
       "rank=1 Mel=1000 REP 0 100.5 101.5", "rank=1 Mel=1000 REP 1 101.5 102.5",
       "rank=1 Mel=1000 STREAM_TRIAD_GBs 24.0 threads 8"]


def test_known_answer():
    a = mod.aggregate(LOG, 1000, 2, 2)
    assert a["window_s"] == pytest.approx(1.5)
    assert a["start_spread_s"] == pytest.approx(0.5)
    assert a["prorated_GBs"] == pytest.approx(48.0)
    assert a["contained_GBs"] == pytest.approx(32.0)
    assert a["median_sum_GBs"] == pytest.approx(48.0)


@pytest.mark.parametrize("log,ncop,nrep", [
    (LOG[:2], 2, 2),                          # a copy is missing
    (LOG[:3] + LOG[2:3], 2, 2),               # duplicate rep
    (LOG, 2, 3),                              # a rep is missing
    (["rank=0 Mel=1 REP 0 1.0 3.0", "rank=0 Mel=1 REP 1 2.0 4.0"], 1, 2),  # overlap
])
def test_incomplete_or_inconsistent_logs_are_refused(log, ncop, nrep):
    with pytest.raises(AssertionError):
        mod.aggregate(log, 1000, ncop, nrep)
