#!/usr/bin/env python
"""Bytes written/read and MPI calls per traced step, from a profiler trace
matched to the after-optimizations HLO dump of the same run.

Events come from each window's raw profile (``*.xplane.pb``), never from the
``*.trace.json.gz`` beside it: that conversion keeps only the earliest
1,000,000 events of a window (measured: 20-50 % of the op events of the duo
C24/C96 windows).  A window folder with no raw profile, or one that is empty
(the profiler writes 0 bytes when the profile exceeds the 2 GB protobuf
limit), unreadable or without op events is an error naming the window;
``--exclude-window`` drops named windows explicitly, and ``n_steps`` must
then count only the kept windows' steps.

For every executed XLA op event (stats ``hlo_op`` / ``hlo_module``) the
op's output and operand shapes are looked up in the dump and summed:
  W = output bytes   (every result written once; logical bytes, not DRAM)
  R = operand bytes  (every operand read in FULL: an upper bound, it
                      overstates fusions that only slice a big operand)
Containers (while/conditional/call/tuple/gte/parameter/bitcast/constant) are
skipped; their bodies' ops are separate events.  mpi4jax custom calls
(target ``mpi*``) are counted apart: calls, payload, time.  Per window
(trace file), the dumped variant of each module covering the most traced op
names is used; a tie between variants is printed as TIE, and events with no
match are counted as unmatched (both bias W, so read them before quoting).

USE (trace with LEGOESM_TRACE_STEPS + LEGOESM_TRACE_WINDOWS, dump with
--xla_dump_to=<dir> --xla_dump_hlo_as_text on the traced rank):
  python scripts/bench/bytes_census.py <hlo_dump_dir> <rank_trace_dir> <n_steps>
      [--exclude-window w00 ...]
"""
from __future__ import annotations

import argparse
import collections
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_jax_trace_gaps import union_us  # noqa: E402

DT = {"f64": 8, "f32": 4, "s32": 4, "u32": 4, "pred": 1, "s64": 8, "u8": 1,
      "s8": 1, "f16": 2, "bf16": 2, "u64": 8, "c64": 8, "c128": 16}
SHAPE = re.compile(r"\b(" + "|".join(DT) + r")\[([\d,]*)\]")
CUT = re.compile(r", (?:kind|calls|custom_call_target|metadata|backend_config|"
                 r"to_apply|condition|body|slice|dimensions|index|frontend_attributes)=")
INSTR = re.compile(r"^\s*(?:ROOT )?%?([\w.\-]+) = (.+?) ([\w\-]+)\((.*)$")
SKIP = {"while", "conditional", "call", "tuple", "get-tuple-element",
        "parameter", "bitcast", "constant"}


def nbytes(shape: str) -> int:
    tot = 0
    for dt, dims in SHAPE.findall(shape):
        n = 1
        for d in dims.split(","):
            if d:
                n *= int(d)
        tot += n * DT[dt]
    return tot


def parse_module(text: str) -> tuple[str, dict]:
    """(module name, {op: (opcode, out_bytes, operand_bytes, cc_target)})."""
    head = re.match(r"HloModule ([\w.\-]+)", text)
    if not head:
        raise ValueError("not an HLO module")
    ops = {}
    for line in text.split("\n"):
        m = INSTR.match(line)
        if not m:
            continue
        name, shape, op, rest = m.groups()
        tgt = re.search(r'custom_call_target="([^"]+)"', rest)
        ops[name] = (op, nbytes(shape),
                     re.findall(r"%([\w.\-]+)", CUT.split(rest, 1)[0]),
                     tgt.group(1) if tgt else None)
    # operands are printed as bare %names: resolve each to its producer's size
    return head.group(1), {k: (o, wb, sum(ops[a][1] for a in an if a in ops), tg)
                           for k, (o, wb, an, tg) in ops.items()}


def load_dump(dump_dir: str) -> dict:
    mods = collections.defaultdict(list)
    for f in sorted(glob.glob(dump_dir + "/*after_optimizations.txt")):
        try:
            name, ops = parse_module(Path(f).read_text())
        except ValueError:
            continue
        mods[name].append(ops)
    return mods


def ns_to_us(ns: float) -> float:
    return ns / 1e3


def op_events(xplane_file: str) -> list[dict]:
    """Op events of every plane and line of a raw profile, in the shape the
    census reads (ts/dur in us, tid = line name)."""
    from jax.profiler import ProfileData

    ev = []
    for plane in ProfileData.from_file(xplane_file).planes:
        for line in plane.lines:
            for e in line.events:
                st = dict(e.stats)
                if "hlo_op" in st:
                    ev.append({"ts": ns_to_us(e.start_ns), "dur": ns_to_us(e.duration_ns),
                               "tid": f"{plane.name}/{line.name}",
                               "args": {"hlo_op": st["hlo_op"],
                                        "hlo_module": st.get("hlo_module", "?")}})
    return ev


def window_files(trace_dir: str, exclude: list[str]) -> list[Path]:
    """One raw profile per file under each window folder of ``trace_dir``;
    raises SystemExit naming any window that cannot be counted."""
    root = Path(trace_dir)
    wins = sorted(d.name for d in root.iterdir() if d.is_dir())
    unknown = sorted(set(exclude) - set(wins))
    if unknown:
        raise SystemExit(f"--exclude-window names no window under {trace_dir}: {unknown}")
    files = []
    for w in wins:
        if w in exclude:
            print(f"  EXCLUDED window {w}")
            continue
        found = sorted((root / w).rglob("*.xplane.pb"))
        if not found:
            raise SystemExit(f"window {w}: no raw profile (*.xplane.pb)")
        for f in found:
            if f.stat().st_size == 0:
                raise SystemExit(f"window {w}: raw profile {f} is empty (0 bytes)")
        print(f"  window {w}: {len(found)} raw profile file(s)")
        files += found
    if not files:
        raise SystemExit(f"no window to count under {trace_dir}")
    return files


def window_events(f: Path, window: str) -> list[dict]:
    try:
        ev = op_events(str(f))
    except Exception as exc:  # any parse failure of one window ends the census
        raise SystemExit(f"window {window}: unreadable raw profile {f}: {exc}") from exc
    if not ev:
        raise SystemExit(f"window {window}: raw profile {f} has no op events")
    return ev


def census_window(mods: dict, ev: list[dict]) -> dict:
    """Bytes and MPI calls of one trace window."""
    seen = collections.defaultdict(set)
    for e in ev:
        seen[e["args"].get("hlo_module", "?")].add(e["args"]["hlo_op"])
    pick, ties = {}, []
    for m, names in seen.items():
        if m not in mods:
            continue
        cov = [len(names & o.keys()) for o in mods[m]]
        pick[m] = mods[m][cov.index(max(cov))]
        if cov.count(max(cov)) > 1:
            ties.append(m)
    out = dict(W=collections.Counter(), R=collections.Counter(),
               K=collections.Counter(), miss=collections.Counter(),
               mpi_n=collections.Counter(), mpi_b=collections.Counter(),
               mpi_t=collections.Counter(), ties=ties)
    for e in ev:
        op, mod = e["args"]["hlo_op"], e["args"].get("hlo_module", "?")
        table = pick.get(mod)
        if table is None or op not in table:
            out["miss"][mod] += 1
            continue
        kind, wb, rb, tgt = table[op]
        if kind in SKIP:
            continue
        if tgt and tgt.startswith("mpi"):
            out["mpi_n"][tgt] += 1
            out["mpi_b"][tgt] += wb
            out["mpi_t"][tgt] += e["dur"]
            continue
        out["W"][mod] += wb
        out["R"][mod] += rb
        out["K"][mod] += e["dur"]
    return out


def call_stats(ev: list[dict], module: str) -> dict:
    """Kernel overlap of one module, and per-call mpi* durations [us]."""
    step = [e for e in ev if e["args"].get("hlo_module") == module]
    if not step:
        return {}
    res = {"summed_us": sum(e["dur"] for e in step), "union_us": union_us(step),
           "span_us": max(e["ts"] + e["dur"] for e in step) - min(e["ts"] for e in step),
           "threads": len({e["tid"] for e in step})}
    d = collections.defaultdict(list)
    for e in step:
        if "mpi" in e["args"]["hlo_op"]:
            d[e["args"]["hlo_op"].split(".")[0]].append(e["dur"])
    for k, v in d.items():
        v = sorted(v)
        res[k] = {"n": len(v), "min": v[0], "p10": v[len(v) // 10],
                  "median": v[len(v) // 2], "max": v[-1]}
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dump_dir")
    p.add_argument("trace_dir", help="one rank's trace dir (all windows under it)")
    p.add_argument("n_steps", type=int, help="steps covered by all windows")
    p.add_argument("--module", default="jit__step",
                   help="module for the kernel-overlap / MPI-call statistics")
    p.add_argument("--exclude-window", action="append", default=[],
                   help="window folder to leave out (repeatable)")
    a = p.parse_args(argv)
    mods = load_dump(a.dump_dir)
    trs = window_files(a.trace_dir, a.exclude_window)
    tot = collections.defaultdict(collections.Counter)
    for f in trs:
        win = Path(f).relative_to(a.trace_dir).parts[0]
        ev = window_events(f, win)
        lines = collections.Counter(e["tid"] for e in ev)
        print(f"  {win}: {len(ev)} op events on {len(lines)} line(s): "
              + ", ".join(f"{k} {v}" for k, v in sorted(lines.items())))
        w = census_window(mods, ev)
        for k in ("W", "R", "K", "miss", "mpi_n", "mpi_b", "mpi_t"):
            tot[k].update(w[k])
        for m in w["ties"]:
            print(f"  TIE {m}: dumped variants share the top op-name coverage")
        print(f"  {win}: write "
              f"{sum(w['W'].values())/1e9:7.3f} GB, mpi calls "
              f"{sum(w['mpi_n'].values())}, unmatched {sum(w['miss'].values())}")
    n = a.n_steps
    print(f"{len(trs)} raw profile(s); modules in dump: {len(mods)}; "
          f"unmatched events by module: {dict(tot['miss'])}")
    print(f"per step (/{n}):")
    for mod in sorted(tot["K"], key=tot["K"].get, reverse=True)[:12]:
        print(f"  {mod[:60]:60s} write {tot['W'][mod]/n/1e9:7.3f} GB  read<= "
              f"{tot['R'][mod]/n/1e9:7.3f} GB  kernel {tot['K'][mod]/n/1e3:8.1f} ms")
    print(f"  {'TOTAL':60s} write {sum(tot['W'].values())/n/1e9:7.3f} GB  read<= "
          f"{sum(tot['R'].values())/n/1e9:7.3f} GB  kernel "
          f"{sum(tot['K'].values())/n/1e3:8.1f} ms")
    for t in tot["mpi_n"]:
        print(f"  {t:28s} calls {tot['mpi_n'][t]/n:7.1f}  payload "
              f"{tot['mpi_b'][t]/n/1e6:8.3f} MB  time {tot['mpi_t'][t]/n/1e3:7.1f} ms")
    st = call_stats(ev, a.module)  # the last raw profile's events
    if st:
        print(f"{a.module} kernels in the last window: summed {st['summed_us']/1e3:.1f} "
              f"ms, union {st['union_us']/1e3:.1f} ms, span {st['span_us']/1e3:.1f} "
              f"ms, threads {st['threads']}")
        for k, v in st.items():
            if isinstance(v, dict):
                print(f"  {k}: n={v['n']} min {v['min']:.1f} us  p10 {v['p10']:.1f}  "
                      f"median {v['median']:.1f}  max {v['max']:.1f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
