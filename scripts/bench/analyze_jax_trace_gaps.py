#!/usr/bin/env python
"""Per-thunk device-timeline attribution from jax.profiler chrome traces.

Replaces the nsys route for the MPAS wait-vs-payload question: nsys
silently omitted ncclDevKernel rows on this lane (campaign 2026-08-07),
while the XLA profiler records what the runtime itself executed. Input =
the ``trace.json.gz`` files ``bench_mpas_spmd_scaling.py --profile-dir``
writes for ranks 0-3 (one node; shared clock).

Per rank, on the busiest GPU device track:
  * total span vs summed event durations -> GAP share (launch/idle),
  * duration histogram by op class (nccl* collectives vs fusions),
  * per-occurrence collective (start, end) lists in trace order.
Cross-rank (ranks on ONE node only):
  * per-occurrence collective START spread across ranks — the skew
    discriminator (GLM consult 2026-08-10): spread ~= duration - wire
    means arrival skew IS the collective's cost; spread << duration
    means the transport itself is slow.
  * calibration: matched collective END times must nearly coincide
    (participants finish together); a large END spread means the
    cross-rank clock alignment is invalid and the spread numbers must
    not be quoted.

USE
  python analyze_jax_trace_gaps.py --trace-root <dir>  # holding rank0..rank3
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from pathlib import Path


def load_trace(rank_dir: Path) -> list[dict]:
    hits = sorted(rank_dir.rglob("*.trace.json.gz"))
    if not hits:
        raise SystemExit(f"no trace.json.gz under {rank_dir}")
    with gzip.open(hits[-1], "rt") as f:
        return json.load(f)["traceEvents"]


def device_events(events: list[dict]) -> list[dict]:
    """Complete ('X') events on the busiest GPU device track."""
    # pid -> name via process_name metadata
    pid_name = {e.get("pid"): e.get("args", {}).get("name", "")
                for e in events if e.get("name") == "process_name"}
    gpu_pids = {p for p, n in pid_name.items()
                if "GPU" in n or "gpu" in n}
    xs = [e for e in events
          if e.get("ph") == "X" and e.get("pid") in gpu_pids
          and e.get("dur", 0) > 0]
    if not xs:
        # fallback: any pid whose events look like kernels
        xs = [e for e in events if e.get("ph") == "X"
              and e.get("dur", 0) > 0]
    # busiest pid/tid pair by total duration
    from collections import defaultdict
    tot = defaultdict(float)
    for e in xs:
        tot[(e["pid"], e.get("tid"))] += e["dur"]
    if not tot:
        raise SystemExit("no complete events in trace")
    # keep ALL tids of the busiest pid (kernels spread across streams)
    busiest_pid = max(tot, key=tot.get)[0]
    return [e for e in xs if e["pid"] == busiest_pid]


def is_collective(name: str) -> bool:
    n = name.lower()
    return ("nccl" in n or "collective-permute" in n
            or "all-reduce" in n or "allreduce" in n)


def is_reduction(name: str) -> bool:
    n = name.lower()
    return "all-reduce" in n or "allreduce" in n or "reducescatter" in n


def is_pairwise(name: str) -> bool:
    n = name.lower()
    return ("sendrecv" in n or "send_recv" in n
            or "collective-permute" in n or "permute" in n)


def exact_period(seq: list[str]) -> int | None:
    """The shortest period that describes the WHOLE sequence, or None.

    Used to learn a lane's step from a capture instead of assuming it. It
    returns None rather than a nearest fit, because an index model built on a
    period the sequence does not actually have pairs unrelated events.
    """
    n = len(seq)
    if n == 0:
        return None
    for period in range(1, n // 2 + 1):
        if all(seq[i] == seq[i % period] for i in range(n)):
            return period
    return None


def _union_ms(evs: list[dict]) -> float:
    """Total device-BUSY time as the union of event intervals — events
    on different streams overlap, so a plain duration sum overstates
    busy (the CPU smoke showed busy 33 s inside a 1.5 s span)."""
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in evs)
    total, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


# Two naming schemes reach this table. GPU captures carry HLO instruction
# names ("all-reduce", "collective-permute"); CPU captures carry the JAX-level
# primitive names instead ("psum", "all_gather"), verified by tracing each one
# on this stack. Both are listed because a name that is missing here does not
# raise -- it reads as a confident ZERO for that family, and for a while the
# CPU lanes reported no reductions at all for exactly that reason.
_FAMILY_TOKENS = {
    "all-gather": "all-gather",
    "allgather": "all-gather",
    "all_gather": "all-gather",
    "all-to-all": "all-to-all",
    "alltoall": "all-to-all",
    "all_to_all": "all-to-all",
    "collective-permute": "permute",
    "ppermute": "permute",
    "all-reduce": "all-reduce",
    "allreduce": "all-reduce",
    # psum/pmax/pmin are all all-reduces on the wire; they differ in the
    # combiner, not in what they cost. pmax/pmin are diagnostics-only in this
    # repo (they carry no gradient) but they still occupy the fabric.
    "psum": "all-reduce",
    "pmax": "all-reduce",
    "pmin": "all-reduce",
    "reduce-scatter": "reduce-scatter",
    "reducescatter": "reduce-scatter",
    "reduce_scatter": "reduce-scatter",
}

# An HLO instruction name is the opcode plus optional numeric suffixes:
# "all-to-all.18", "ppermute.1295", bare "collective-permute". Anchoring on
# that shape instead of a substring search is what keeps a FUSION whose name
# merely CONTAINS an opcode -- "fusion.all-gather.18" -- out of the gather
# family, which codex demonstrated a substring match would swallow.
# Underscores are part of the name on CPU captures (all_gather.7);
# a pattern without them silently rejects every such event.
_INSTR_RE = re.compile(r"^([a-z][a-z_-]*)(?:\.\d+)*$")


def collective_family(name: str) -> str | None:
    """Which collective family an event name belongs to, or None.

    Matched on the HLO instruction name the profiler emits, anchored whole,
    so a capture whose names do not look like these returns nothing rather
    than guessing. Callers must pair this with a census check: a family the
    compiled program contains but this function never sees would otherwise
    read as a confident zero.
    """
    n = name.strip().lower()
    if n.startswith("end: "):
        # The profiler emits a zero-width "end: <op>" marker beside each
        # collective. Counting it would double the instruction count and
        # contribute no duration, so it is not a separate event.
        return None
    m = _INSTR_RE.match(n)
    if m is None:
        return None
    return _FAMILY_TOKENS.get(m.group(1))


def time_by_family(evs: list[dict], steps: int, *,
                   expect: dict[str, int] | None = None,
                   devices_per_rank: int = 1) -> dict:
    """Wall time per step in which each collective family was OCCUPIED.

    WHAT THIS NUMBER IS. Per family, the UNION of that family's event
    intervals over the capture, divided by the traced step count. The union
    is used because events overlap -- summing durations double counts, which
    is how a 1.5 s span once reported 33 s of "busy".

    WHAT IT IS NOT. It is OBSERVED OCCUPANCY on the tracks this capture
    holds. Naming the reference quantity each time, because "bound" on its
    own gets the direction wrong (codex caught exactly that here):

    * against the family's WIRE cost on the traced ranks it OVERSTATES,
      because a collective's duration includes the time it spends waiting
      for late participants, so a family that is cheap on the wire can
      occupy most of a step;
    * against the SUMMED per-device work it UNDERSTATES, because concurrent
      events collapse into one interval -- twenty-four devices each running
      10 ms of exchange report 10 ms, not 240;
    * against the STEP'S CRITICAL PATH it is neither, because the capture
      is rank-local and a collective on untraced ranks can delay a later
      operation that IS traced.

    Only a controlled intervention turns this into a cause.

    ``expect`` is the compiled program's per-family instruction count. When
    given, every family must appear with exactly
    ``count * devices_per_rank * steps`` events or the result carries
    ``census_ok=False`` and the reason -- a family renamed or fused away
    would otherwise read as a confident zero.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if devices_per_rank <= 0:
        raise ValueError(
            f"devices_per_rank must be positive, got {devices_per_rank}")
    buckets: dict[str, list[dict]] = {}
    for e in evs:
        fam = collective_family(e.get("name", ""))
        if fam is not None:
            buckets.setdefault(fam, []).append(e)
    span_us = (max(e["ts"] + e["dur"] for e in evs)
               - min(e["ts"] for e in evs)) if evs else 0.0
    out = {"steps": steps, "devices_per_rank": devices_per_rank,
           "span_ms_per_step": span_us / 1e3 / steps, "families": {}}
    for fam, fevs in sorted(buckets.items()):
        union_us = _union_ms(fevs)
        out["families"][fam] = {
            "instructions": len(fevs),
            "ms_per_step": union_us / 1e3 / steps,
            "share_of_span": (union_us / span_us) if span_us else None,
        }
    if expect is not None:
        problems = []
        for fam, hlo_count in sorted(expect.items()):
            want = hlo_count * devices_per_rank * steps
            got = out["families"].get(fam, {}).get("instructions", 0)
            if got != want:
                problems.append(
                    f"{fam}: capture has {got} events, the compiled program "
                    f"says {hlo_count} instructions x {devices_per_rank} "
                    f"device(s) x {steps} steps = {want}")
        for fam in sorted(out["families"]):
            if fam not in expect:
                problems.append(
                    f"{fam}: present in the capture but absent from the "
                    f"compiled census given")
        out["census_ok"] = not problems
        out["census_problems"] = problems
    return out


def per_rank_summary(evs: list[dict]) -> dict:
    evs = sorted(evs, key=lambda e: e["ts"])
    t0, t1 = evs[0]["ts"], max(e["ts"] + e["dur"] for e in evs)
    span = t1 - t0
    busy = _union_ms(evs)
    coll = [e for e in evs if is_collective(e["name"])]
    comp = [e for e in evs if not is_collective(e["name"])]
    return {
        "span_ms": span / 1e3,
        "busy_ms": busy / 1e3,
        "gap_share": 1.0 - min(busy / span, 1.0),
        "n_events": len(evs),
        "n_collectives": len(coll),
        "coll_ms": sum(e["dur"] for e in coll) / 1e3,
        "comp_ms": sum(e["dur"] for e in comp) / 1e3,
        "coll_dur_us_median": (statistics.median(e["dur"] for e in coll)
                               if coll else None),
        "collectives": [(e["name"], e["ts"], e["ts"] + e["dur"])
                        for e in coll],
    }


def partner_aware_spread(summaries: dict, pm: dict, steps: int = 4,
                         fills: int = 3, end_tol_us: float = 5.0) -> bool:
    """Cross-rank arrival spread using the SCHEDULE, not overlap.

    ``pm`` = partner map: rank -> ordered [(round, partner), ...] it
    participates in per fill. Per rank the traced event sequence is
    (fills x participations + 1 reduction) per step, so the k-th
    participation of fill f of step s sits at a KNOWN index — no clock
    needed for identification. For a pair (a, b) that are partners in
    round r, the two paired SendRecv kernels END together (the transfer
    completes on both sides), so per-pair end deltas estimate the
    constant per-rank clock offset; the residual start delta after
    removing it is the genuine arrival skew of that pair.

    THE SEQUENCE MODEL IS AN ASSUMPTION ABOUT THE LANE, NOT A FACT.
    The defaults describe the atmosphere step: four traced steps, three
    halo fills, one reduction. A lane whose step has a different shape —
    an ocean step, for instance, whose free-surface solver contributes two
    exchanges and two reductions per iteration — produces a longer
    sequence, and indexing into it with the wrong period pairs events that
    never belonged together. So the count must match the model EXACTLY.
    A trace that merely has enough events is refused, because "enough" is
    how an incompatible sequence gets silently accepted.

    Returns True when at least one pair was quoted, False when everything
    was refused.
    """
    ranks = sorted(summaries)
    quoted = 0
    part = {rk: pm["ranks"][rk.replace("rank", "")] for rk in ranks
            if rk.replace("rank", "") in pm["ranks"]}
    missing = [rk for rk in ranks if rk not in part]
    if missing:
        print(f"  no schedule entry for {', '.join(missing)} — those ranks "
              f"contribute nothing, so this is a coverage hole, not a result")

    def idx(rank, step, fill, k):
        p = len(part[rank])
        return step * (fills * p + 1) + fill * p + k

    for a_i in range(len(ranks)):
        for b_i in range(a_i + 1, len(ranks)):
            a, b = ranks[a_i], ranks[b_i]
            if a not in part or b not in part:
                continue
            common = [(ka, kb, r)
                      for ka, (r, pa) in enumerate(part[a])
                      if pa == int(b.replace("rank", ""))
                      for kb, (r2, pb) in enumerate(part[b]) if r2 == r]
            if not common:
                continue
            ca, cb = summaries[a]["collectives"], summaries[b]["collectives"]
            # EXACT, not "at least": a longer sequence means the lane's step
            # is not the one this model describes, and indexing into it
            # would pair unrelated events with a straight face.
            want_a = steps * (fills * len(part[a]) + 1)
            want_b = steps * (fills * len(part[b]) + 1)
            if len(ca) != want_a or len(cb) != want_b:
                print(f"{a}-{b}: event count disagrees with the sequence "
                      f"model (a {len(ca)} vs {want_a}, b {len(cb)} vs "
                      f"{want_b}, from steps={steps} fills={fills}) — the "
                      f"model does not describe this lane, NOT quotable")
                continue
            # A matching COUNT is not a matching SEQUENCE. Check that the
            # events the index model lands on are actually the pairwise
            # exchanges it thinks they are, and that each step's trailing
            # slot really is a reduction. Without this an incompatible
            # sequence of the same length still passes.
            bad = None
            for rk, pt in ((a, part[a]), (b, part[b])):
                seq = summaries[rk]["collectives"]
                per = fills * len(pt) + 1
                for s in range(steps):
                    if not is_reduction(seq[s * per + per - 1][0]):
                        bad = (rk, "trailing slot is not a reduction",
                               seq[s * per + per - 1][0]); break
                    for f in range(fills):
                        for k in range(len(pt)):
                            nm = seq[s * per + f * len(pt) + k][0]
                            if not is_pairwise(nm):
                                bad = (rk, "fill slot is not a pairwise "
                                       "exchange", nm); break
                        if bad: break
                    if bad: break
                if bad: break
            if bad:
                print(f"{a}-{b}: {bad[1]} at a computed index on {bad[0]} "
                      f"(saw '{bad[2]}') — the sequence model does not "
                      f"describe this lane, NOT quotable")
                continue

            end_d, start_d, durs = [], [], []
            for s in range(steps):
                for f in range(fills):
                    for ka, kb, r in common:
                        _, sa, ea = ca[idx(a, s, f, ka)]
                        _, sb, eb = cb[idx(b, s, f, kb)]
                        end_d.append(eb - ea)
                        start_d.append(sb - sa)
                        durs.append(statistics.median([ea - sa, eb - sb]))
            off = statistics.median(end_d)
            resid_start = [abs(d - off) for d in start_d]
            resid_end = [abs(d - off) for d in end_d]
            kernel = statistics.median(durs)
            end_resid = statistics.median(resid_end)
            print(f"{a}-{b}: {len(start_d)} partner pairs "
                  f"(rounds {[r for _, _, r in common]}), clock offset "
                  f"{off:+.1f} us")
            # Matched participants finish together, so once the constant
            # offset is removed the END residual must collapse. If it does
            # not, the two clocks are not aligned and every skew number
            # derived from them is meaningless — so withhold them rather
            # than print them next to a caveat nobody will read.
            # The gate is ABSOLUTE and reads the TAIL, for two reasons both
            # reviewers demonstrated with counterexamples. Scaling it to the
            # kernel let a long-kernel lane hide a large clock fault, and
            # reading the median let a fault that corrupts a minority of
            # pairs pass while the skew it produced was quoted to one
            # decimal. What the residual is really bounded by is timestamp
            # jitter, which does not grow with kernel length. The WORST
            # residual is the right statistic, not a quantile: a quantile of
            # a handful of comparisons can step right over the one bad pair,
            # and one bad pair is enough to invalidate the set.
            end_worst = max(resid_end)
            if end_worst > end_tol_us:
                print(f"    CALIBRATION FAILED: worst end residual "
                      f"{end_worst:.1f} us exceeds the {end_tol_us:.1f} us "
                      f"tolerance (median {end_resid:.1f}, kernel "
                      f"{kernel:.1f}). Matched participants finish together, "
                      f"so this pair's timestamps cannot be compared; "
                      f"arrival skew is WITHHELD.")
                continue
            quoted += 1
            print(f"    arrival skew  median {statistics.median(resid_start):7.1f}"
                  f"  p90 {sorted(resid_start)[int(0.9 * len(resid_start))]:7.1f} us"
                  f"   | end residual median {end_resid:5.1f} us"
                  f"   | kernel median {kernel:7.1f} us")
    return quoted > 0

def top_ops(events: list[dict], steps: int, n: int = 25) -> list[dict]:
    """Per-op-name totals over ALL threads of the busiest process: thread
    time per step, executions per step, and the mean number of threads
    busy on that op while any thread is (thread time / union span). A
    parallel op shows a mean near the pool size; a serial one shows ~1.
    Uses every tid, unlike device_events' single-track view, because on
    CPU one op's work is spread over the worker threads."""
    from collections import defaultdict
    xs = [e for e in events if e.get("ph") == "X" and e.get("dur", 0) > 0]
    tot = defaultdict(float)
    for e in xs:
        tot[e["pid"]] += e["dur"]
    if not tot:
        raise SystemExit("no complete events in trace")
    pid = max(tot, key=tot.get)
    by = defaultdict(list)
    for e in xs:
        # XLA ops only: the CPU host track also carries Python frames
        # ("$file:line fn") and the executor's own bookkeeping spans.
        if e["pid"] == pid and not e["name"].startswith("$") \
                and not e["name"].startswith("ThunkExecutor"):
            by[e["name"]].append(e)
    rows = []
    for name, evs in by.items():
        thread_ms = sum(e["dur"] for e in evs) / 1e3
        iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in evs)
        span_us, cur_s, cur_e = 0.0, iv[0][0], iv[0][1]
        for a, b in iv[1:]:
            if a > cur_e:
                span_us += cur_e - cur_s
                cur_s, cur_e = a, b
            else:
                cur_e = max(cur_e, b)
        span_ms = (span_us + cur_e - cur_s) / 1e3
        rows.append({"name": name, "thread_ms_per_step": thread_ms / steps,
                     "span_ms_per_step": span_ms / steps,
                     "count_per_step": len(evs) / steps,
                     "mean_threads": thread_ms / span_ms if span_ms else 0.0})
    rows.sort(key=lambda r: -r["span_ms_per_step"])
    return rows[:n]



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", required=True, type=Path)
    ap.add_argument("--partner-map", type=Path, default=None,
                    help="JSON from the padding-audit job: schedule "
                         "round->partner per rank. Enables partner-aware "
                         "cross-rank arrival skew (the only quotable "
                         "form — overlap matching pairs non-partners).")
    ap.add_argument("--steps", type=int, default=4,
                    help="traced steps in the capture (default 4, which is "
                         "what the atmosphere bench traces)")
    ap.add_argument("--fills", type=int, default=3,
                    help="halo fills per step in the lane under capture "
                         "(default 3, the atmosphere step). Together with "
                         "--steps this IS the sequence model; a trace whose "
                         "event count disagrees is refused rather than "
                         "indexed with the wrong period.")
    ap.add_argument("--end-tol-us", type=float, default=5.0,
                    help="reject a pair whose WORST END-residual exceeds this "
                         "many microseconds. Matched participants finish "
                         "together, so the residual is bounded by timestamp "
                         "jitter, which is absolute and does not grow with "
                         "kernel length. The WORST residual is gated, not a "
                         "quantile, because a quantile of a handful of "
                         "comparisons steps right over the one bad pair.")
    ap.add_argument("--name-census", type=int, default=0, metavar="N",
                    help="print the first N collective names per rank, in "
                         "trace order, with their repeat structure. This is "
                         "how a lane's step is LEARNED instead of guessed: "
                         "the event count alone is degenerate, since two "
                         "fills with three reductions per step totals the "
                         "same as four fills with one.")
    ap.add_argument("--time-by-family", action="store_true",
                    help="report wall time per step in which each collective "
                         "family was OCCUPIED, as the UNION of its intervals "
                         "(overlapping events would otherwise double count). "
                         "Observed occupancy, neither an upper nor a lower "
                         "bound on cost -- see time_by_family.")
    ap.add_argument("--expect-census", default=None, metavar="JSON",
                    help="the compiled program's per-family instruction "
                         "counts, e.g. '{\"permute\": 396, \"all-to-all\": "
                         "12, \"all-gather\": 0}'. Every family must appear "
                         "with count x devices x steps events or the run "
                         "EXITS 3 and no number may be quoted: a family the "
                         "profiler renamed reads as a confident zero.")
    ap.add_argument("--expect-ranks", type=int, default=None, metavar="N",
                    help="how many rank*/ directories the capture must "
                         "hold. A capture missing ranks would otherwise be "
                         "analysed and exit 0, and a family that lives only "
                         "on the missing ranks would read as absent.")
    ap.add_argument("--devices-per-rank", type=int, default=1,
                    help="devices each traced process drives (1 under the "
                         "multi-process launch, 24 in a virtual-CPU smoke)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-ops", type=int, default=0, metavar="N",
                    help="Print the N op names with the largest wall span "
                         "per step on rank 0 with their thread time, count "
                         "and mean threads busy (CPU parallel-efficiency "
                         "view; span is per op name so it does not add up "
                         "across names). Then exit.")
    args = ap.parse_args()
    if args.top_ops:
        rank0 = sorted(args.trace_root.glob("rank*"))[0]
        evs = load_trace(rank0)
        rows = top_ops(evs, args.steps, args.top_ops)
        print(f"{'op':44s} {'span ms/step':>12s} {'thr ms/step':>12s} "
              f"{'n/step':>7s} {'threads':>8s}")
        for r in rows:
            print(f"{r['name'][:44]:44s} {r['span_ms_per_step']:12.1f} "
                  f"{r['thread_ms_per_step']:12.1f} {r['count_per_step']:7.1f} "
                  f"{r['mean_threads']:8.1f}")
        return 0

    # A flag that silently does nothing is a gate the caller thinks they
    # armed (GLM). These two only act inside --time-by-family.
    if not args.time_by_family:
        unused = [n for n, v in (("--expect-ranks", args.expect_ranks),
                                 ("--expect-census", args.expect_census),
                                 ("--devices-per-rank",
                                  args.devices_per_rank if
                                  args.devices_per_rank != 1 else None))
                  if v is not None]
        if unused:
            raise SystemExit(
                f"{', '.join(unused)} only take effect with "
                f"--time-by-family, and would be silently ignored here")

    ranks = sorted(d for d in args.trace_root.iterdir()
                   if d.is_dir() and d.name.startswith("rank"))
    if not ranks:
        raise SystemExit(f"no rank*/ dirs under {args.trace_root}")

    if args.time_by_family:
        # Descriptive mode: no skew model, no refusal gates — it reports
        # per-family wall time and stops. Runs alone so its numbers are
        # never read as a step budget.
        expect = (json.loads(args.expect_census)
                  if args.expect_census else None)
        if args.expect_ranks is not None and len(ranks) != args.expect_ranks:
            print(f"REFUSED: capture holds {len(ranks)} rank dir(s) "
                  f"({', '.join(r.name for r in ranks)}), the launch was "
                  f"told to expect {args.expect_ranks} — a family living "
                  f"only on a missing rank would read as absent")
            return 3
        fam_out, bad = {}, []
        for rd in ranks:
            evs = device_events(load_trace(rd))
            f = time_by_family(evs, args.steps, expect=expect,
                               devices_per_rank=args.devices_per_rank)
            fam_out[rd.name] = f
            print(f"{rd.name}: span {f['span_ms_per_step']:.2f} ms/step")
            if not f["families"]:
                print("    NO COLLECTIVES NAMED in this capture — the "
                      "profiler did not emit HLO instruction names, so "
                      "nothing may be attributed from it")
            for fam, d in f["families"].items():
                # share_of_span is None for a zero-width capture; printing
                # it as a percentage would crash a run whose numbers are
                # otherwise fine (GLM).
                share = ("  n/a % of span" if d["share_of_span"] is None
                         else f"  {100 * d['share_of_span']:5.1f}% of span")
                print(f"    {fam:<15} {d['instructions']:>6} instr  "
                      f"{d['ms_per_step']:8.2f} ms/step{share}")
            for pr in f.get("census_problems", []):
                print(f"    CENSUS MISMATCH {pr}")
                bad.append(f"{rd.name}: {pr}")
        print("NOTE: union of intervals — OBSERVED OCCUPANCY on the traced "
              "ranks only. It OVERSTATES the family's wire cost (duration "
              "includes waiting for late participants), UNDERSTATES the "
              "summed per-device work (concurrent events collapse), and "
              "says nothing about the step's critical path (untraced ranks "
              "are invisible and can delay traced ones). Only a controlled "
              "intervention makes it a cause.")
        if args.out:
            Path(args.out).write_text(json.dumps(fam_out, indent=2))
            print(f"wrote {args.out}")
        if bad:
            print(f"REFUSED: {len(bad)} census mismatch(es) — the capture is "
                  "not the compiled program, so nothing above may be quoted")
            return 3
        if expect is None:
            print("NO CENSUS GIVEN (--expect-census): a family the profiler "
                  "renamed would read as zero and nothing here would notice")
        return 0

    summaries = {}
    for rd in ranks:
        s = per_rank_summary(device_events(load_trace(rd)))
        summaries[rd.name] = s
        print(f"{rd.name}: span {s['span_ms']:.1f} ms, busy "
              f"{s['busy_ms']:.1f} ms (gap {100 * s['gap_share']:.0f}%), "
              f"collectives {s['n_collectives']} x median "
              f"{s['coll_dur_us_median'] and round(s['coll_dur_us_median'])} us "
              f"= {s['coll_ms']:.1f} ms, compute {s['comp_ms']:.1f} ms")

    # Cross-rank matching by TIME OVERLAP (participation differs per rank
    # — a coloured round runs a kernel only on its participants, so
    # occurrence-index matching is invalid; job 26854741 showed 88 vs
    # 136 events across one node's ranks).  Two collectives on two ranks
    # that overlap in wall-clock (shared node clock) and overlap by
    # >50% of the shorter one are treated as the same logical exchange.
    rank_names = sorted(summaries)
    if args.partner_map is not None:
        # Overlap matching pairs ranks that were never partners, so once a
        # schedule is available its numbers are the only ones that may
        # appear at all. Printing both invites the wrong one being quoted.
        print("cross-rank: overlap matching SUPPRESSED — a partner map was "
              "given, and schedule-matched numbers are the quotable ones")
    elif len(rank_names) < 2:
        print("cross-rank: <2 ranks — spread not computed")
    else:
        base = summaries[rank_names[0]]["collectives"]
        spreads, end_spreads, durs, pairs = [], [], [], 0
        for other in rank_names[1:]:
            oc = summaries[other]["collectives"]
            j = 0
            for (nm, s0, e0) in base:
                while j < len(oc) and oc[j][2] < s0:
                    j += 1
                if j >= len(oc):
                    break
                nm1, s1, e1 = oc[j]
                ov = min(e0, e1) - max(s0, s1)
                if ov <= 0 or ov < 0.5 * min(e0 - s0, e1 - s1):
                    continue
                spreads.append(abs(s1 - s0))
                end_spreads.append(abs(e1 - e0))
                durs.append(statistics.median([e0 - s0, e1 - s1]))
                pairs += 1
        if not pairs:
            print("cross-rank: no overlapping collective pairs matched")
        else:
            cal = statistics.median(end_spreads)
            md = statistics.median(durs)
            print(f"cross-rank ({len(rank_names)} ranks, {pairs} "
                  f"overlap-matched pairs), RAW (uncorrected clocks):")
            print(f"  median START spread : {statistics.median(spreads):8.1f} us")
            print(f"  p90    START spread : {sorted(spreads)[int(0.9 * len(spreads))]:8.1f} us")
            print(f"  median duration     : {md:8.1f} us")
            print(f"  median END spread   : {cal:8.1f} us")

            # jax trace timestamps are RELATIVE to each process's own
            # start_trace() call (first_ts ~1.4 ms on every rank of job
            # 26854741) — the raw spreads above are contaminated by the
            # per-process trace-start offset, which is itself a
            # skew-class quantity. Correct it by anchoring on the
            # physics: matched collective ENDS coincide (participants
            # finish together), so each rank's constant clock offset =
            # median(end_k - end_ref) over its matched pairs with the
            # reference rank. After correction the END spread MUST
            # collapse (self-consistency); the start spread is then the
            # real arrival skew.
            ref = summaries[rank_names[0]]["collectives"]
            offsets = {rank_names[0]: 0.0}
            for other in rank_names[1:]:
                oc = summaries[other]["collectives"]
                deltas, j = [], 0
                for (nm, s0, e0) in ref:
                    while j < len(oc) and oc[j][2] < s0:
                        j += 1
                    if j >= len(oc):
                        break
                    nm1, s1, e1 = oc[j]
                    ov = min(e0, e1) - max(s0, s1)
                    if ov > 0 and ov >= 0.5 * min(e0 - s0, e1 - s1):
                        deltas.append(e1 - e0)
                offsets[other] = (statistics.median(deltas)
                                  if deltas else 0.0)
            c_spreads, c_ends = [], []
            for other in rank_names[1:]:
                oc = summaries[other]["collectives"]
                off = offsets[other]
                j = 0
                for (nm, s0, e0) in ref:
                    while j < len(oc) and oc[j][2] - off < s0:
                        j += 1
                    if j >= len(oc):
                        break
                    s1, e1 = oc[j][1] - off, oc[j][2] - off
                    ov = min(e0, e1) - max(s0, s1)
                    if ov > 0 and ov >= 0.5 * min(e0 - s0, e1 - s1):
                        c_spreads.append(abs(s1 - s0))
                        c_ends.append(abs(e1 - e0))
            if c_spreads:
                print(f"  OFFSET-CORRECTED (per-rank offsets "
                      f"{ {k: round(v, 1) for k, v in offsets.items()} }):")
                print(f"  median START spread : {statistics.median(c_spreads):8.1f} us")
                print(f"  p90    START spread : {sorted(c_spreads)[int(0.9 * len(c_spreads))]:8.1f} us")
                print(f"  median END spread   : {statistics.median(c_ends):8.1f} us "
                      f"(must be ~0 by construction — self-consistency)")

    if args.name_census:
        print(f"=== collective name census (first {args.name_census} per rank) ===")
        for rk in sorted(summaries):
            seq = [nm for nm, _, _ in summaries[rk]["collectives"]]
            print(f"--- {rk}: {len(seq)} collectives")
            for i, nm in enumerate(seq[:args.name_census]):
                kind = ("reduction" if is_reduction(nm)
                        else "pairwise" if is_pairwise(nm) else "OTHER")
                print(f"  {i:4d}  {kind:9s}  {nm}")
            # The shortest repeating prefix, if the sequence has one, is the
            # period the index model would need. Reported, never assumed.
            period = exact_period(seq)
            if period is not None:
                print(f"  repeats with period {period} "
                      f"({len(seq) // period} whole repetitions)")
            else:
                print("  no exact repeating period — the sequence is not "
                      "periodic, so no fixed index model can describe it")

    skew_ok = None
    if args.partner_map is not None:
        pm = json.load(open(args.partner_map))
        print("=== partner-aware arrival skew (schedule-matched) ===")
        skew_ok = partner_aware_spread(summaries, pm, steps=args.steps,
                                       fills=args.fills,
                                       end_tol_us=args.end_tol_us)
        if not skew_ok:
            print("NO PAIR SURVIVED: arrival skew is not available from this "
                  "capture. The kernel, collective and gap shares above still "
                  "stand; the skew numbers do not exist and must not be "
                  "inferred from the shares.")

    if args.out:
        payload = {k: {kk: vv for kk, vv in v.items() if kk != "collectives"}
                   for k, v in summaries.items()}
        payload["_status"] = {
            "arrival_skew_quotable": skew_ok,
            "partner_map": str(args.partner_map) if args.partner_map else None,
            "steps": args.steps, "fills": args.fills,
            "end_tol_us": args.end_tol_us,
            # The shares above are CAPTURE statistics. The span includes the
            # harness barriers that sit inside the trace window, and the
            # collective and compute totals sum durations across streams, so
            # concurrent events are counted twice. They do not yet budget the
            # receipt's step time and must not be quoted as if they did.
            "shares_are_capture_statistics_not_a_step_budget": True,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
    # "ran and refused everything" must not look like success to a launcher.
    return 0 if skew_ok is not False else 3


if __name__ == "__main__":
    raise SystemExit(main())
