"""The census summariser must divide by steps and must not invent families."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "scripts/bench/_summarise_cpu_census.py"
_spec = importlib.util.spec_from_file_location("summ", _SRC)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _write(tmp_path, tag, steps, families):
    (tmp_path / f"census_{tag}.json").write_text(
        json.dumps({"steps": steps, "families": families}))


def test_counts_are_per_step_and_missing_families_are_zero(tmp_path, capsys):
    # 120 events over 6 steps is 20 per step -- the ocean solver's iteration
    # count, which a static module count would have reported as 1.
    _write(tmp_path, "oce_nd8", 6, {"all-reduce": {"instructions": 120,
                                                   "ms_per_step": 3.5}})
    # The atmosphere arm has no reduction at all; it must print 0, not
    # inherit the ocean's column.
    _write(tmp_path, "atm_nd8", 6, {"permute": {"instructions": 18,
                                                "ms_per_step": 1.25}})
    rows = mod.load(str(tmp_path))
    assert [(r[0], r[1]) for r in rows] == [("atm", 8), ("oce", 8)]

    mod.main.__globals__["sys"].argv = ["x", str(tmp_path)]
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "20.0" in out, "120 events over 6 steps must read as 20 per step"
    assert "3.0" in out, "18 events over 6 steps must read as 3 per step"
    assert "CPU_CENSUS_DONE" in out


def test_no_census_files_is_an_error_not_an_empty_table(tmp_path):
    # An empty directory must not print a clean table of nothing: a sweep
    # that produced no counts has to be visible as a failure.
    mod.main.__globals__["sys"].argv = ["x", str(tmp_path)]
    assert mod.main() == 1
