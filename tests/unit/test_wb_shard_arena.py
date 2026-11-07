"""The sharding probe's HLO censuses must see what they claim to see.

Both assertions below are regressions against a real defect: the first
version of this probe counted only the synchronous collective spellings and
reported "no communication" on programs that overlap their collectives, and
its width census was quoted as if it measured memory when it only ever
counted leading axes.
"""
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "wb_shard_arena.py"


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("wb_shard_arena", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ASYNC_HLO = """
  %start = f32[8,4]{1,0} all-reduce-start(%a), replica_groups={}
  %done = f32[8,4]{1,0} all-reduce-done(%start)
  %g = (f32[8,4]{1,0}) all-gather-start(%b), dimensions={0}
  %t = f32[8,4]{1,0} all-to-all(%c), dimensions={0}
"""


def test_the_collective_census_counts_overlapped_communication(probe):
    """An async collective is communication; counting only the sync spelling
    turns a program full of it into a reassuring "none"."""
    counts = probe.census_collectives(ASYNC_HLO)
    assert counts["all-reduce-start"] == 1
    assert counts["all-reduce-done"] == 1
    assert counts["all-gather-start"] == 1
    assert counts["all-to-all"] == 1
    assert sum(counts.values()) == 4


def test_the_collective_census_is_silent_on_a_program_without_any(probe):
    assert probe.census_collectives("%x = f32[4]{0} add(%a, %b)\n") == {}


def test_the_width_census_reads_the_leading_axis_only(probe):
    """The blind spot is the point: a full column count on an inner axis --
    how the radiation arrays hid -- must NOT register as partitioned."""
    hlo = "%r = f64[60,8,4608,1,10]{4,3,2,1,0} fusion(%a)\n"
    lead = probe.census_leading_widths(hlo)
    assert lead[60] == 1
    assert 4608 not in lead, "an inner-axis column count must not look partitioned"
