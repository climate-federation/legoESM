"""bytes_census: a tiny dump + trace whose bytes are counted by hand."""
import collections
import gzip
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

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


def _xspace(lines: dict) -> bytes:
    """Raw profile with one plane and one line per key of ``lines``; each
    event is (hlo_op or None, start_ns, duration_ns); None = no hlo_op stat."""
    from jax.profiler import ProfileData

    names = sorted({op for evs in lines.values() for op, _, _ in evs if op})
    meta = {op: i + 2 for i, op in enumerate(names)}
    txt = 'planes { id: 1 name: "/host:CPU" '
    for i, (ln, evs) in enumerate(lines.items()):
        txt += f'lines {{ id: {i + 1} name: "{ln}" timestamp_ns: 0 '
        for op, t0, dur in evs:
            if op is None:
                txt += f"events {{ metadata_id: 1 offset_ps: {t0 * 1000} duration_ps: {dur * 1000} }} "
            else:
                txt += (f"events {{ metadata_id: {meta[op]} offset_ps: {t0 * 1000} "
                        f"duration_ps: {dur * 1000} stats {{ metadata_id: 1 str_value: \"{op}\" }} "
                        f"stats {{ metadata_id: 2 str_value: \"jit_f\" }} }} ")
        txt += "} "
    txt += 'event_metadata { key: 1 value { id: 1 name: "ThreadpoolListener::StartRegion" } } '
    for op, k in meta.items():
        txt += f'event_metadata {{ key: {k} value {{ id: {k} name: "{op}" }} }} '
    txt += ('stat_metadata { key: 1 value { id: 1 name: "hlo_op" } } '
            'stat_metadata { key: 2 value { id: 2 name: "hlo_module" } } }')
    return ProfileData.text_proto_to_serialized_xspace(txt)


# EVENTS as raw-profile lines: ns = 1000 x the us above, tid 1 -> line "a", 2 -> "b",
# plus one event without an hlo_op stat (not an op event).
LINES = {"a": [(e["args"]["hlo_op"], e["ts"] * 1000, e["dur"] * 1000) for e in EVENTS
               if e["tid"] == 1] + [(None, 60_000, 5_000)],
         "b": [(e["args"]["hlo_op"], e["ts"] * 1000, e["dur"] * 1000) for e in EVENTS
               if e["tid"] == 2]}


def _run(tmp_path, windows: dict, n_steps="2", extra=()):
    """Dump with HLO, windows {name: raw-profile bytes | None (folder without one)}."""
    (tmp_path / "hlo").mkdir(exist_ok=True)
    (tmp_path / "hlo" / "module_0001.jit_f.cpu_after_optimizations.txt").write_text(HLO)
    for w, data in windows.items():
        d = tmp_path / "tr" / w / "plugins" / "profile" / "1"
        d.mkdir(parents=True)
        if data is not None:
            (d / "h.xplane.pb").write_bytes(data)
    return mod.main([str(tmp_path / "hlo"), str(tmp_path / "tr"), n_steps, *extra])


def test_ns_to_us():
    assert mod.ns_to_us(2500) == 2.5


def test_op_events_reads_every_line_in_us(tmp_path):
    f = tmp_path / "h.xplane.pb"
    f.write_bytes(_xspace(LINES))
    ev = mod.op_events(str(f))
    assert len(ev) == len(EVENTS)          # the event without hlo_op is not an op event
    assert sorted((e["args"]["hlo_op"], e["ts"], e["dur"]) for e in ev) == sorted(
        (e["args"]["hlo_op"], e["ts"], e["dur"]) for e in EVENTS)
    assert {e["tid"] for e in ev} == {"/host:CPU/a", "/host:CPU/b"}
    w = mod.census_window({"jit_f": [mod.parse_module(HLO)[1]]}, ev)
    assert w["W"]["jit_f"] == 16000 and w["R"]["jit_f"] == 32000 and w["mpi_n"]["mpi_sendrecv_ffi"] == 2


def test_main_end_to_end(tmp_path, capsys):
    assert _run(tmp_path, {"w00": _xspace(LINES)}) == 0
    out = capsys.readouterr().out
    assert "mpi_sendrecv_ffi             calls     1.0" in out
    assert "TOTAL" in out and "unmatched 1" in out
    assert "w00: 7 op events on 2 line(s): /host:CPU/a 6, /host:CPU/b 1" in out
    assert "window w00: 1 raw profile file(s)" in out


@pytest.mark.parametrize("data, msg", [
    (None, "window w01: no raw profile"),
    (b"", r"window w01: raw profile .* is empty \(0 bytes\)"),
    (b"not a profile", "window w01: unreadable raw profile"),
    ("no-ops", "window w01: raw profile .* has no op events"),
])
def test_uncountable_window_is_an_error_naming_it(tmp_path, data, msg):
    if data == "no-ops":
        data = _xspace({"a": [(None, 0, 10)]})
    with pytest.raises(SystemExit, match=msg):
        _run(tmp_path, {"w00": _xspace(LINES), "w01": data})


def test_excluded_window_is_printed_and_not_counted(tmp_path, capsys):
    assert _run(tmp_path, {"w00": _xspace(LINES), "w01": b""}, "2",
                ["--exclude-window", "w01"]) == 0
    out = capsys.readouterr().out
    assert "EXCLUDED window w01" in out and "1 raw profile(s)" in out


def test_excluding_a_window_that_does_not_exist_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match=r"names no window .*\['w07'\]"):
        _run(tmp_path, {"w00": _xspace(LINES)}, "2", ["--exclude-window", "w07"])


# A traced loop of N triads per device, warmed up and synced on both sides of
# the trace; x64, so the loop counter is s64.
GEN = """
import sys, jax, jax.numpy as jnp
out, n_iter, n_dev = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
a = jnp.ones(1000 * n_dev)
if n_dev > 1:
    from jax.sharding import NamedSharding, PartitionSpec as P
    a = jax.device_put(a, NamedSharding(jax.make_mesh((n_dev,), ("x",)), P("x")))
def f(a, b):
    return jax.lax.fori_loop(0, n_iter, lambda i, c: c * 0.5 + b, a)
g = jax.jit(f)
g(a, a).block_until_ready()
with jax.profiler.trace(out):
    g(a, a).block_until_ready()
"""
# 2,000,003 op events per device: past the json conversion's 1,000,000-event
# cap and past the largest production window seen (1,821,773).
N_ITER = 1_000_000


def _trace(tmp_path, n_dev):
    out = tmp_path / f"d{n_dev}"
    env = {**os.environ, "JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": f"--xla_dump_to={out}/hlo --xla_dump_hlo_as_text "
                        f"--xla_force_host_platform_device_count={n_dev}"}
    subprocess.run([sys.executable, "-c", GEN, str(out / "tr" / "w00"), str(N_ITER),
                    str(n_dev)], env=env, check=True)
    return out


# slow: measured 2 min for both cases together, 9 GB peak resident memory
@pytest.mark.slow
@pytest.mark.parametrize("n_dev", [1, 2])
def test_raw_profile_counts_every_op_past_the_json_cap(tmp_path, n_dev):
    out = _trace(tmp_path, n_dev)
    mods = mod.load_dump(str(out / "hlo"))
    assert len(mods["jit_f"]) == 1
    ops = mods["jit_f"][0]
    # read off the optimized HLO: per iteration the triad fusion writes
    # f64[1000] and the counter add s64[]; once, two entry copies (the
    # f64[1000] carry and the s64[] counter) and the while (a container)
    assert ops["multiply_add_fusion"][1] == 8000 and ops["wrapped_add"][1] == 8
    copies = sorted(v[1] for k, v in ops.items() if v[0] == "copy")
    assert copies == [8, 8000]
    per_line_n = 2 * N_ITER + 3
    per_line_w = N_ITER * (8000 + 8) + 8000 + 8

    ev = mod.window_events(next((out / "tr").rglob("*.xplane.pb")), "w00")
    by_line = collections.defaultdict(list)
    for e in ev:
        by_line[e["tid"]].append(e)
    assert len(by_line) == n_dev
    for evs in by_line.values():
        w = mod.census_window(mods, evs)
        assert len(evs) == per_line_n
        assert w["W"]["jit_f"] == per_line_w and not w["miss"] and not w["ties"]
    assert mod.census_window(mods, ev)["W"]["jit_f"] == n_dev * per_line_w
    # one positive-byte event fewer is caught by the exact count
    drop = next(i for i, e in enumerate(ev) if e["args"]["hlo_op"] == "multiply_add_fusion")
    assert mod.census_window(mods, ev[:drop] + ev[drop + 1:])["W"]["jit_f"] != n_dev * per_line_w

    # the json beside it, which the census used to read, is capped
    js = next((out / "tr").rglob("*.trace.json.gz"))
    x = [e for e in json.load(gzip.open(js))["traceEvents"] if e.get("ph") == "X"]
    jop = [e for e in x if "hlo_op" in e.get("args", {})]
    assert len(x) == 1_000_000
    assert len(jop) < n_dev * per_line_n
    assert mod.census_window(mods, jop)["W"]["jit_f"] < n_dev * per_line_w
