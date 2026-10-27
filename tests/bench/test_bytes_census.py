"""bytes_census: a tiny dump + trace whose bytes are counted by hand."""
import gzip
import importlib.util
import json
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "scripts" / "bench"
sys.path.insert(0, str(_BENCH))
_spec = importlib.util.spec_from_file_location("bytes_census", _BENCH / "bytes_census.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

HLO = """HloModule jit_f, entry_computation_layout={(f64[1000]{0}, f64[1000]{0})->f64[1000]{0}}

ENTRY %main.5 (Arg_0.1: f64[1000], Arg_1.2: f64[1000]) -> (f64[1000]) {
  %Arg_0.1 = f64[1000]{0} parameter(0)
  %Arg_1.2 = f64[1000]{0} parameter(1)
  %fusion = f64[1000]{0} fusion(%Arg_0.1, %Arg_1.2), kind=kLoop, calls=%fused_computation
  %sendrecv_mpi.1 = f32[10]{0} custom-call(%fusion), custom_call_target="mpi_sendrecv_ffi"
  ROOT %tuple.3 = (f64[1000]{0}) tuple(%fusion)
}
"""


def _ev(op, ts, dur, tid=1):
    return {"ph": "X", "ts": ts, "dur": dur, "tid": tid,
            "args": {"hlo_op": op, "hlo_module": "jit_f"}}


# Two steps: fusion twice (write 8000 B, read 2 x 8000 B each), one mpi call
# (payload f32[10] = 40 B), a parameter and a tuple (skipped), one unknown op.
EVENTS = [_ev("Arg_0.1", 0, 1), _ev("fusion", 0, 10), _ev("fusion", 5, 10, tid=2),
          _ev("sendrecv_mpi.1", 20, 3), _ev("sendrecv_mpi.1", 30, 7),
          _ev("tuple.3", 40, 1), _ev("nope", 50, 1)]


def test_parse_module_sizes():
    name, ops = mod.parse_module(HLO)
    assert name == "jit_f"
    assert ops["fusion"][:3] == ("fusion", 8000, 16000)
    assert ops["sendrecv_mpi.1"][1] == 40 and ops["sendrecv_mpi.1"][3] == "mpi_sendrecv_ffi"


def test_census_window_known_answer():
    mods = {"jit_f": [mod.parse_module(HLO)[1]]}
    w = mod.census_window(mods, [e for e in EVENTS if e["args"]["hlo_op"] != "sendrecv_mpi.1"]
                          + EVENTS[3:4])
    assert w["W"]["jit_f"] == 16000 and w["R"]["jit_f"] == 32000
    assert w["mpi_n"]["mpi_sendrecv_ffi"] == 1 and w["mpi_b"]["mpi_sendrecv_ffi"] == 40
    assert w["miss"]["jit_f"] == 1 and w["ties"] == []


def test_two_equally_covering_variants_are_flagged():
    ops = mod.parse_module(HLO)[1]
    w = mod.census_window({"jit_f": [ops, dict(ops)]}, EVENTS)
    assert w["ties"] == ["jit_f"]


def test_call_stats_union_and_mpi_quantiles():
    st = mod.call_stats(EVENTS, "jit_f")
    # intervals [0,1] [0,10] [5,15] [20,23] [30,37] [40,41] [50,51] -> union 15+3+7+1+1
    assert st["union_us"] == 27 and st["summed_us"] == 33 and st["span_us"] == 51
    assert st["sendrecv_mpi"]["n"] == 2 and st["sendrecv_mpi"]["min"] == 3


def test_span_runs_to_the_latest_end_not_the_last_start():
    st = mod.call_stats([_ev("fusion", 0, 100), _ev("fusion", 50, 1)], "jit_f")
    assert st["span_us"] == 100 and st["union_us"] == 100


def test_main_end_to_end(tmp_path, capsys):
    (tmp_path / "hlo").mkdir()
    (tmp_path / "hlo" / "module_0001.jit_f.cpu_after_optimizations.txt").write_text(HLO)
    w = tmp_path / "tr" / "w00" / "plugins"
    w.mkdir(parents=True)
    with gzip.open(w / "h.trace.json.gz", "wt") as f:
        json.dump({"traceEvents": EVENTS}, f)
    assert mod.main([str(tmp_path / "hlo"), str(tmp_path / "tr"), "2"]) == 0
    out = capsys.readouterr().out
    assert "mpi_sendrecv_ffi             calls     1.0" in out
    assert "TOTAL" in out and "unmatched 1" in out
