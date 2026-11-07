"""G1(c) DYNAMIC poison gate -- the REAL acoustic loop, per firing (M3).

Scope (duo_tiled_port_scope.md G1c, GLM): the static table-equality gates
(G1a/b) cannot see a MISTIMED or STALE firing inside the n_split acoustic
loop.  This probe drives the FULL certified ``fv_dynamics_step`` through
the M3 tile arm (kt=1 bridge mesh) with an instrumented ``tile_comm`` that

  1. COUNTS every dispatcher firing per arm name at trace time (the
     enumerated firing census is printed -- scope P3's checklist at run
     level), then
  2. for EVERY (arm name, firing index), rebuilds the step with that ONE
     firing POISONED: the real exchange runs, then every slot the arm's
     split WRITES is replaced by NaN (the write-set is derived from the
     built split tables, not guessed).  A poisoned firing whose writes
     feed any downstream read must drive the INTERIOR of the final state
     non-finite -- the skeleton probe's mechanism on the real step.

VERDICT RULE = MISS-SET PARITY against the certified RING arm (codex M3
close-out MAJOR).  A firing whose poison is not detected is a MISS; the
step's own batched schedule already contains firings whose write-set is
overwritten by a later same-name firing (or gated off) before any
interior read, so "any miss = fail" would fail the certified ring path
too.  The gate therefore runs the IDENTICAL harness on both arms
(``--comm ring`` = ``tab.ring_comm``, the pre-M3 six-device
configuration; ``--comm tile`` = ``tab.tile_comm``) and PASSES iff the
two MISS SETS (name + firing index) are EQUAL -- tiling preserved the
choreography.  ANY set difference is a tiling-changed liveness and FAILS.
Each arm's dump (``--dump-misses``) is JSON with PROVENANCE -- arm, deck
mode, the EXECUTED probe sha256 (hashed by every process over the file
it is running, at import; one value asserted identical over all
per-variant records), the traced census and the result of EVERY
variant -- and is written only for a COMPLETE sweep; ``--assemble-dump``
builds it from split-process sweep dirs of per-process JSON records
(``--record-dir``).  The verdict (``--ref-misses`` / ``--verdict-only``)
REFUSES (exit 4) unless both dumps are complete sweeps of the same
census with identical executed hashes and the expected firing count
(``--expect-census``): empty-vs-empty is a refusal, never a pass, and
``--verdict-selftest`` proves a truncated and an empty dump refuse.
The probe prints whether a missed variant's final state moved at all vs
control (halo-only consumption vs a dead firing) for the per-miss
classification.

CONTROL CONTRACT (codex MINOR, doc fixed): the control arm (no poison)
must stay finite AND sit inside the pre-registered G0 TRIAGE envelope --
per field, ``|control - certified single-device| <= 1e-13 + 4 *
|plain GSPMD face-sharded - certified|``, the accepted sharded-lowering
class -- NOT bit-identity (the kt=1 bridge measured ~7e-11 abs, within
4x the plain face-sharded step's own deviation; bit-level attribution
is the parity job's).  A broken instrument is O(field) off and still
fails this instantly.

EXCHANGE-PURITY MODE (``--purity``, GLM close-out): with a RECORDING
comm the full step runs once on ring and once on tile (kt from ``--kt``)
and every firing's (inputs, outputs) are captured via
``jax.debug.callback``.  GATE RULE: for EVERY recorded input (both
arms' inputs, every execution) the STANDALONE-jitted TILE exchange must
equal the certified single-device impl BITWISE, and no mismatch of any
row may touch a weight-one COPY target cell or a cell outside the
exchange write-set.  The exchange is index copies plus bit-copied-weight
ORDERED stencils (k2e / corner Lagrange / projection); XLA CPU contracts
mul+add into FMA by default, so bit identity of the weighted cells is
only defined under ``XLA_FLAGS=--xla_cpu_max_isa=AVX`` (no FMA3) -- the
gate run pins it (measured: with the pin every scalar eager-vs-compiled
deviation vanishes; job 9600021), and a default-lowering run is kept as
the production record where every deviation must still be confined to
the weighted-stencil TARGET cells.  Also reported, never gated: the
same-input ring-jit vs tile-jit identity and the ring-jit vs certified
impl agreement (the pre-M3 ring lane's own property -- measured to
deviate <= 3e-14 rel in vector projection cells under jit), the
in-graph recorded outputs vs their own standalone jit, and the first
firing where the arms' in-graph INPUTS diverge (the kernels' lowering
class, upstream of the exchange -- what proves the residual step diff
is kernel/reduction-class, never exchange-born).  kt=2 exercises the
per-firing flat<->blocked converters on live interior tile edges (kt=1
has none).

Each poison variant is a fresh model build (fresh context + fresh jit
cache): the tables hash by identity as a static jit arg, so mutating a
traced context's comm would leave a stale cache -- rebuilt instead,
never mutated.
"""
from __future__ import annotations

import argparse
import functools
import time

import numpy as np


# ---------------------------------------------------------------------------
# write-set masks, derived from the built splits (never guessed)
# ---------------------------------------------------------------------------

def _split_write_masks(split, masks=None, kinds=None):
    """Per-buffer bool masks over the BLOCKED global arrays marking every
    slot ANY device's op writes (padded scratch rows excluded).  Reads
    only public split attributes.  ``kinds``: restrict to op kinds
    (``stencil``/``stencil_pair`` = weighted arithmetic targets,
    ``scatter`` = weight-one copies)."""
    kt = split.kt
    if masks is None:
        masks = [np.zeros((6, kt * t0, kt * t1), bool)
                 for (t0, t1) in split.tiles]
    offs = np.asarray(split.offs)
    for ph in split.phases:
        for op in ph.ops:
            if kinds is not None and op.kind not in kinds:
                continue
            dl = np.asarray(op.dst_loc)
            for dev in range(split.ndev):
                f = dev // (kt * kt)
                ti, tj = (dev // kt) % kt, dev % kt
                for s in np.unique(dl[dev]):
                    if s >= split.scr:
                        continue            # padded no-op slot
                    b = int(np.searchsorted(offs, s, side="right") - 1)
                    if b >= len(masks):     # write into a dropped buffer
                        continue
                    t0, t1 = split.tiles[b]
                    i, j = divmod(int(s) - int(offs[b]), t1)
                    masks[b][f, ti * t0 + i, tj * t1 + j] = True
    return masks


def _vector_write_masks(bundle, ex_kinds=None):
    """(u_mask, v_mask) for a composed C/D bundle: union over the three
    sub-splits that write the u/v OUTPUT buffers (ex, wr, cn).  The wr
    split carries 4 buffers (u, v, proj_u, proj_v); only its first two
    land in the output.  ``ex_kinds`` restricts the ex split's ops; the
    wr (projection-derived) and cn (corner Lagrange) writes are always
    FP-arithmetic-derived and always included."""
    m = _split_write_masks(bundle.split_ex, kinds=ex_kinds)
    wr = _split_write_masks(bundle.split_wr)
    m[0] |= wr[0]
    m[1] |= wr[1]
    if bundle.split_cn is not None:
        cn = _split_write_masks(bundle.split_cn)
        m[0] |= cn[0]
        m[1] |= cn[1]
    return m[0], m[1]


_ARITH_KINDS = ("stencil", "stencil_pair")


def _arith_masks(splits):
    """Per arm name: masks of the WEIGHTED-ARITHMETIC target cells (k2e
    ring stencils, corner-Lagrange, projection-derived vector writes);
    the complement within the write-set is weight-one copies."""
    masks = {}
    sa = _split_write_masks(splits["scalar_A"], kinds=_ARITH_KINDS)[0]
    sb = _split_write_masks(splits["scalar_B"], kinds=_ARITH_KINDS)[0]
    du, dv = _vector_write_masks(splits["dgrid"], ex_kinds=_ARITH_KINDS)
    cu, cv = _vector_write_masks(splits["cgrid"], ex_kinds=_ARITH_KINDS)
    masks["scalar_A"] = masks["scalar_A_allk"] = (sa,)
    masks["scalar_B"] = masks["scalar_B_allk"] = (sb,)
    masks["dgrid"] = masks["dgrid_allk"] = (du, dv)
    masks["cgrid"] = masks["cgrid_allk"] = (cu, cv)
    return masks


def _cell_stats(o, e, wmask, amask):
    """Classify the mismatching cells of ``o`` vs ``e`` over the flat
    (6, m0, m1[, K]) array: inside weighted-arithmetic targets, inside
    weight-one copy targets, or OUTSIDE the exchange write-set (a copy
    or outside cell mismatching contradicts the fusion/contraction
    class).  Returns dict(text, rel, copy, outside)."""
    bad = ~np.isclose(o, e, rtol=0, atol=0, equal_nan=True)
    w = np.asarray(wmask)
    a = np.asarray(amask)
    while w.ndim < bad.ndim:
        w, a = w[..., None], a[..., None]
    w, a = np.broadcast_to(w, bad.shape), np.broadcast_to(a, bad.shape)
    n_bad = int(bad.sum())
    n_ar = int((bad & a).sum())
    n_cp = int((bad & w & ~a).sum())
    n_out = int((bad & ~w).sum())
    d = np.where(bad, np.abs(o - e), 0.0)
    dmax = float(np.nanmax(d)) if n_bad else 0.0
    scale = float(np.nanmax(np.abs(e))) or 1.0
    return {"text": (f"nbad={n_bad} arith={n_ar} copy={n_cp} "
                     f"outside_writeset={n_out} dmax={dmax:.3e} "
                     f"rel={dmax / scale:.2e}"),
            "rel": dmax / scale, "copy": n_cp, "outside": n_out}


def _apply_poison(x, mask):
    import jax.numpy as jnp

    m = jnp.asarray(mask)
    while m.ndim < x.ndim:
        m = m[..., None]
    return jnp.where(m, jnp.nan, x)


class InstrumentedTileComm:
    """Duck-types :class:`DuoTileComm` / :class:`DuoRingComm` (same
    method surface, so ONE instrument serves both arms).  Counts firings
    per arm name at trace time; with ``poison=(name, k)`` the k-th
    firing of that arm runs the REAL exchange then NaNs its split's
    write-set; with ``record`` (a dict) every firing's (inputs, outputs)
    are captured at RUN time via ``jax.debug.callback`` (lowered once on
    the full logical value under the jit's ShardingContext -- the
    dispatchers fire outside the comm's own shard_map)."""

    def __init__(self, real, masks, poison=None, record=None):
        self._real = real
        self._masks = masks
        self._poison = poison
        self._record = record
        self.counts = {}

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def _rec(self, name, k, n_in, *arrs):
        arrs = [np.asarray(a) for a in arrs]
        self._record.setdefault((name, k), []).append(
            (arrs[:n_in], arrs[n_in:]))

    def _fire(self, name, ins, outs):
        k = self.counts.get(name, 0)
        self.counts[name] = k + 1
        if self._poison == (name, k):
            outs = tuple(_apply_poison(o, m)
                         for o, m in zip(outs, self._masks[name]))
        if self._record is not None:
            import jax

            jax.debug.callback(
                functools.partial(self._rec, name, k, len(ins)),
                *ins, *outs)
        return outs

    def ext_scalar(self, f6, stag):
        (out,) = self._fire(f"scalar_{stag}", (f6,),
                            (self._real.ext_scalar(f6, stag),))
        return out

    def ext_scalar_allk(self, f6k, stag):
        (out,) = self._fire(f"scalar_{stag}_allk", (f6k,),
                            (self._real.ext_scalar_allk(f6k, stag),))
        return out

    def ext_vector_dgrid(self, u6, v6):
        return self._fire("dgrid", (u6, v6),
                          self._real.ext_vector_dgrid(u6, v6))

    def ext_vector_dgrid_allk(self, u6k, v6k):
        return self._fire("dgrid_allk", (u6k, v6k),
                          self._real.ext_vector_dgrid_allk(u6k, v6k))

    def ext_vector_cgrid(self, uc6, vc6):
        return self._fire("cgrid", (uc6, vc6),
                          self._real.ext_vector_cgrid(uc6, vc6))

    def ext_vector_cgrid_allk(self, uc6k, vc6k):
        return self._fire("cgrid_allk", (uc6k, vc6k),
                          self._real.ext_vector_cgrid_allk(uc6k, vc6k))


def _certified_replay(name, tab0, ins):
    """The certified single-device exchange (no comm attached on
    ``tab0``) for one recorded firing's inputs, by arm name."""
    import jax.numpy as jnp

    from legoesm.grids import fv3_duo_halos as H

    ins = [jnp.asarray(a) for a in ins]
    if name.startswith("scalar_"):
        stag = name.split("_")[1]
        fn = (H.ext_scalar_sixface_allk if name.endswith("_allk")
              else H.ext_scalar_sixface)
        return (np.asarray(fn(ins[0], tab0, stag)),)
    fn = {"dgrid": H.ext_vector_dgrid_sixface,
          "dgrid_allk": H.ext_vector_dgrid_sixface_allk,
          "cgrid": H.ext_vector_cgrid_sixface,
          "cgrid_allk": H.ext_vector_cgrid_sixface_allk}[name]
    return tuple(np.asarray(x) for x in fn(ins[0], ins[1], tab0))


def _probe_sha256(path=None):
    import hashlib

    with open(path or __file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# EXECUTED-BYTES provenance (codex gate-1 re-review MAJOR): every process
# hashes the probe file it is EXECUTING, at import, and writes that hash
# into every record it emits (census + each variant).  The verdict
# requires all executed hashes across BOTH arms to be identical -- a
# stamp copied from a job log proves nothing about the bytes that ran.
#
# THREAT MODEL, stated because the boundary is real (codex r2, NOT fixed
# and not fixable here): this chain defends against STALE and MIXED-UP
# artifacts -- a dump from an older probe, a sweep glued from two probe
# versions, an edited census, a substituted control, a partial sweep.  It
# does NOT defend against a deliberate FORGER with write access to the
# artifact directory: every field is JSON written by our own processes,
# so an adversary who copies the current hashes produces records this
# code cannot distinguish from real ones.  Making provenance unforgeable
# needs a signature key the probe does not hold (or artifacts written to
# storage the sweep cannot rewrite); that is out of scope for a gate
# whose failure mode is accident, not sabotage.
_EXEC_SHA256 = _probe_sha256()


class DumpRefused(Exception):
    """A miss-set dump that cannot carry a verdict (codex close-out
    MAJOR: an empty/truncated/unlabelled dump must REFUSE, never parse
    as an empty set)."""


def _require_unique_census(census, what):
    """codex gate-1 r2 MAJOR: a census with a REPEATED variant id plus
    one result per distinct id would satisfy completeness and the
    expected count while hiding a partial sweep -- refuse duplicates,
    so every count below is a count of DISTINCT variants."""
    dup = sorted({v for v in census if census.count(v) > 1})
    if dup:
        raise DumpRefused(f"{what}: census repeats variant(s) {dup[:5]} "
                          f"({len(census)} entries, {len(set(census))} "
                          f"distinct)")


def _require_executed_here(sha, what):
    """Provenance BINDING (codex gate-1 r2 MAJOR): a hash field is only
    evidence when it equals the hash of the bytes THIS process is
    executing -- recomputed here, never trusted from JSON."""
    if sha != _EXEC_SHA256:
        raise DumpRefused(f"{what}: executed probe {str(sha)[:12]} != the "
                          f"probe bytes running this check "
                          f"{_EXEC_SHA256[:12]} -- the sweep and its "
                          f"assembly/verdict must execute ONE probe")


def _write_dump(path, *, arm, mode, census, results, variant_sha):
    """Arm dump = JSON with PROVENANCE: arm, deck mode, the executed
    probe sha256 (one value, asserted identical over every per-variant
    record), the full census (variant ids), the result of EVERY variant
    run ('miss'/'detected') and each variant's own executed hash.
    Written only for a COMPLETE sweep."""
    import json

    census = [f"{n} {k}" for n, k in census]
    _require_unique_census(census, f"{arm} dump (write)")
    missing = sorted(set(census) - set(results))
    extra = sorted(set(results) - set(census))
    if missing or extra or not census:
        raise DumpRefused(
            f"refusing to write {arm} dump: sweep incomplete -- census "
            f"{len(census)}, ran {len(results)}, missing {missing[:5]}, "
            f"extra {extra[:5]}")
    hashes = {variant_sha.get(v) for v in census}
    if len(hashes) != 1 or None in hashes:
        raise DumpRefused(
            f"refusing to write {arm} dump: executed probe hash not "
            f"unique over the sweep ({len(hashes)} distinct, "
            f"{sum(v not in variant_sha for v in census)} variants "
            f"without one)")
    with open(path, "w") as fh:
        json.dump({"arm": arm, "mode": mode,
                   "executed_sha256": hashes.pop(),
                   "census": census, "results": results,
                   "variant_sha256": {v: variant_sha[v] for v in census}},
                  fh, indent=1)


def _read_dump(path, expect_arm, expect_mode=None, expect_census=None):
    """Load + VALIDATE one arm dump; raises DumpRefused on any gap.
    Returns (dump, miss_set)."""
    import json
    import os

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise DumpRefused(f"{expect_arm} dump {path}: missing or empty")
    try:
        with open(path) as fh:
            d = json.load(fh)
    except ValueError as e:
        raise DumpRefused(f"{expect_arm} dump {path}: not JSON ({e})")
    for key in ("arm", "mode", "executed_sha256", "census", "results",
                "variant_sha256"):
        if key not in d:
            raise DumpRefused(f"{expect_arm} dump {path}: no '{key}' field")
    _require_unique_census(list(d["census"]), f"{expect_arm} dump")
    _require_executed_here(d["executed_sha256"], f"{expect_arm} dump")
    vs = d["variant_sha256"]
    wrong = sorted(v for v in d["census"]
                   if vs.get(v) != d["executed_sha256"])
    if wrong:
        raise DumpRefused(f"{expect_arm} dump: {len(wrong)} variants "
                          f"executed a different probe than the dump "
                          f"declares {wrong[:5]}")
    if d["arm"] != expect_arm:
        raise DumpRefused(f"dump {path} is arm {d['arm']!r}, expected "
                          f"{expect_arm!r}")
    if expect_mode is not None and d["mode"] != expect_mode:
        raise DumpRefused(f"{expect_arm} dump mode {d['mode']!r} != "
                          f"{expect_mode!r}")
    census = list(d["census"])
    if not census:
        raise DumpRefused(f"{expect_arm} dump: EMPTY census")
    if expect_census is not None and len(census) != expect_census:
        raise DumpRefused(f"{expect_arm} dump: census has {len(census)} "
                          f"firings, expected {expect_census} (the "
                          f"site-map census for this deck)")
    ran = d["results"]
    missing = sorted(set(census) - set(ran))
    extra = sorted(set(ran) - set(census))
    if missing or extra:
        raise DumpRefused(f"{expect_arm} dump: sweep INCOMPLETE -- "
                          f"{len(missing)} census variants not run "
                          f"{missing[:5]}, {len(extra)} unknown {extra[:5]}")
    bad = {v: r for v, r in ran.items() if r not in ("miss", "detected")}
    if bad:
        raise DumpRefused(f"{expect_arm} dump: non-terminal results {bad}")
    return d, {v for v, r in ran.items() if r == "miss"}


def _write_record(record_dir, name, payload):
    import json
    import os

    payload = dict(payload, exec_sha256=_EXEC_SHA256)
    with open(os.path.join(record_dir, name), "w") as fh:
        json.dump(payload, fh, indent=1)


def _assemble_dump(dirs, *, arm, mode, out):
    """Build one arm dump from split-process sweep directories, each
    holding the census PROCESS's ``census.json`` and one
    ``variant_<name>_<k>.json`` per variant PROCESS -- every record
    carries the sha256 of the probe bytes THAT process executed.
    Refuses on: census/arm/mode differing between dirs, any record
    whose arm/mode/executed hash disagrees with the census record, a
    variant seen twice with different results, or a malformed record."""
    import glob
    import json
    import os

    census = None
    exec_sha = None
    results, vsha = {}, {}
    for d in dirs:
        cpath = os.path.join(d, "census.json")
        try:
            csha = _probe_sha256(cpath)          # bytes ON DISK, now
            with open(cpath) as fh:
                c = json.load(fh)
        except (OSError, ValueError) as e:
            raise DumpRefused(f"{d}: no readable census.json ({e})")
        if c.get("arm") != arm or c.get("mode") != mode:
            raise DumpRefused(f"{d}/census.json is arm {c.get('arm')!r} "
                              f"mode {c.get('mode')!r}, expected "
                              f"{arm!r}/{mode!r}")
        # BINDING: the census process must have executed the very bytes
        # this assembler executes (recomputed, not read from JSON)
        _require_executed_here(c.get("exec_sha256"), f"{d}/census.json")
        _require_unique_census(list(c["census"]), f"{d}/census.json")
        # codex gate-1 r2 MAJOR (real hole, fixed): a MISSING control.npz
        # used to yield None and SKIP its binding -- deleting the file
        # disabled the check.  The control every variant compared against
        # must still be present and unchanged.
        cpath_ctrl = os.path.join(d, "control.npz")
        if not os.path.exists(cpath_ctrl):
            raise DumpRefused(f"{d}: control.npz is missing -- the control "
                              f"every variant was scored against must be "
                              f"present for its binding to mean anything")
        _load_control(cpath_ctrl)               # refuses an empty archive
        ctrl_sha = _probe_sha256(cpath_ctrl)
        if census is None:
            census, exec_sha = list(c["census"]), c["exec_sha256"]
            ctrl_sha0 = ctrl_sha
        elif list(c["census"]) != census or c["exec_sha256"] != exec_sha:
            raise DumpRefused(f"census or executed hash differs between "
                              f"sweep dirs at {d}")
        elif ctrl_sha != ctrl_sha0:
            # codex gate-1 r3 MAJOR: each dir bound its records to ITS
            # control, so two partial sweeps scored against different
            # controls were internally consistent and glued silently.
            raise DumpRefused(f"control.npz differs between sweep dirs at "
                              f"{d} ({ctrl_sha0[:12]} vs {ctrl_sha[:12]}) "
                              f"-- one assembled arm needs ONE control")
        n_rec = 0
        for rec_path in sorted(glob.glob(os.path.join(d, "variant_*.json"))):
            try:
                with open(rec_path) as fh:
                    r = json.load(fh)
                v, res, sha = r["variant"], r["result"], r["exec_sha256"]
                rc_sha, rk_sha = r["census_sha256"], r["control_sha256"]
            except (OSError, ValueError, KeyError) as e:
                raise DumpRefused(f"malformed variant record {rec_path}: "
                                  f"{e}")
            if r.get("arm") != arm or r.get("mode") != mode:
                raise DumpRefused(f"{rec_path}: arm/mode mismatch")
            _require_executed_here(sha, rec_path)
            # BINDING to the census/control artifacts this variant
            # consumed: their bytes on disk NOW must hash to what the
            # variant process hashed THEN (an edited census.json or a
            # swapped control refuses)
            if rc_sha != csha:
                raise DumpRefused(f"{rec_path}: census.json bytes changed "
                                  f"since this variant ran ({rc_sha[:12]} "
                                  f"then, {csha[:12]} now)")
            if rk_sha != ctrl_sha:
                raise DumpRefused(f"{rec_path}: control.npz bytes changed "
                                  f"since this variant ran ({rk_sha[:12]} "
                                  f"then, {ctrl_sha[:12]} now)")
            if results.get(v, res) != res:
                raise DumpRefused(f"variant {v} recorded as both "
                                  f"{results[v]} and {res}")
            results[v], vsha[v] = res, sha
            n_rec += 1
        if n_rec == 0:
            raise DumpRefused(f"{d}: no variant records")
    if census is None:
        raise DumpRefused("no sweep dirs given")
    _write_dump(out, arm=arm, mode=mode,
                census=[tuple(v.split()) for v in census], results=results,
                variant_sha=vsha)
    print(f"[assemble {arm}] {len(results)}/{len(census)} variant records "
          f"from {len(dirs)} dir(s), executed probe {exec_sha[:12]} -> "
          f"{out}")


def _miss_set_verdict(tile_path, ring_path, label, *, expect_mode=None,
                      expect_census=None):
    """THE GATE RULE.  Both dumps must be COMPLETE sweeps of the SAME
    census under the SAME probe (arm/mode/sha metadata agree) -- any
    gap REFUSES (exit 4), empty-vs-empty included.  Then PASS iff the
    tile miss set == the ring miss set."""
    try:
        td, tile = _read_dump(tile_path, "tile", expect_mode, expect_census)
        rd, ref = _read_dump(ring_path, "ring", expect_mode, expect_census)
        if td["executed_sha256"] != rd["executed_sha256"]:
            raise DumpRefused(f"executed probe differs between arms: tile "
                              f"{td['executed_sha256'][:12]} vs ring "
                              f"{rd['executed_sha256'][:12]}")
        if td["mode"] != rd["mode"]:
            raise DumpRefused(f"deck mode differs: tile {td['mode']} vs "
                              f"ring {rd['mode']}")
        if list(td["census"]) != list(rd["census"]):
            raise DumpRefused("census differs between arms")
    except DumpRefused as e:
        print(f"DYNAMIC POISON GATE ({label}): REFUSED -- {e}")
        return 4
    census = list(td["census"])
    key = (lambda v: (v.split()[0], int(v.split()[1])))
    ex = td["executed_sha256"]
    n_rec = len(td["variant_sha256"]) + len(rd["variant_sha256"])
    print(f"[miss-set {label}] census={len(census)} distinct firings; "
          f"executed probe {ex[:12]} identical over all {n_rec} variant "
          f"records of both arms AND equal to the bytes running this "
          f"verdict ({_EXEC_SHA256[:12]}; refused otherwise); tile "
          f"misses={len(tile)} ring misses={len(ref)}")
    print("  firing            ring-miss  tile-miss")
    for v in sorted(tile | ref, key=key):
        n, k = v.split()
        print(f"  {n}#{k:<10} {str(v in ref):<10} {str(v in tile)}")
    only_t, only_r = tile - ref, ref - tile
    if only_t or only_r:
        for v in sorted(only_t, key=key):
            print(f"TILING-CHANGED LIVENESS: {v.replace(' ', '#')} missed "
                  f"on tile, detected on ring")
        for v in sorted(only_r, key=key):
            print(f"TILING-CHANGED LIVENESS: {v.replace(' ', '#')} "
                  f"detected on tile, missed on ring")
        print(f"DYNAMIC POISON GATE ({label}): FAIL -- miss sets differ")
        return 1
    fmt = (lambda s: "{" + ", ".join(v.replace(" ", "#") for v in
                                     sorted(s, key=key)) + "}")
    print(f"DYNAMIC POISON GATE ({label}): PASS -- census {len(census)}/"
          f"{len(census)} variants run on BOTH arms; tile miss set "
          f"{fmt(tile)} == ring miss set {fmt(ref)} ({len(tile)} "
          f"inherited misses, 0 tiling-changed)")
    return 0


def _verdict_selftest(tile_path, ring_path, label, **kw):
    """Non-vacuity of the verdict: a TRUNCATED tile dump (one census
    variant dropped) and an EMPTY tile dump must both REFUSE (rc 4)."""
    import json
    import os
    import tempfile

    with open(tile_path) as fh:
        d = json.load(fh)
    tmp = tempfile.mkdtemp(prefix="missparity_selftest_")
    trunc = os.path.join(tmp, "tile_truncated.json")
    d2 = dict(d)
    d2["results"] = dict(d["results"])
    dropped = sorted(d2["results"])[0]
    del d2["results"][dropped]
    with open(trunc, "w") as fh:
        json.dump(d2, fh)
    empty = os.path.join(tmp, "tile_empty.json")
    open(empty, "w").close()
    rc_t = _miss_set_verdict(trunc, ring_path, f"{label} SELFTEST-truncated",
                             **kw)
    rc_e = _miss_set_verdict(empty, ring_path, f"{label} SELFTEST-empty",
                             **kw)
    # duplicated-variant census (codex r2): two partial sweeps glued
    # under a repeated id keep the expected count -- must REFUSE
    dup = os.path.join(tmp, "tile_dupcensus.json")
    d3 = dict(d)
    d3["census"] = list(d["census"]) + [list(d["census"])[0]]
    with open(dup, "w") as fh:
        json.dump(d3, fh)
    rc_d = _miss_set_verdict(dup, ring_path, f"{label} SELFTEST-dupcensus",
                             **kw)
    # self-asserted hash (codex r2): a dump whose executed hash was
    # edited to another value must REFUSE against the running bytes
    forged = os.path.join(tmp, "tile_forgedhash.json")
    d4 = dict(d)
    d4["executed_sha256"] = "0" * 64
    d4["variant_sha256"] = {v: "0" * 64 for v in d["variant_sha256"]}
    with open(forged, "w") as fh:
        json.dump(d4, fh)
    rc_f = _miss_set_verdict(forged, ring_path,
                             f"{label} SELFTEST-forgedhash", **kw)
    ok = rc_t == 4 and rc_e == 4 and rc_d == 4 and rc_f == 4
    tag = (lambda rc: f"rc {rc} {'REFUSED' if rc == 4 else 'NOT REFUSED'}")
    print(f"VERDICT SELFTEST ({label}): truncated dump (dropped "
          f"{dropped.replace(' ', '#')}) -> {tag(rc_t)}; empty dump -> "
          f"{tag(rc_e)}; duplicated-variant census -> {tag(rc_d)}; forged "
          f"executed hash -> {tag(rc_f)}; "
          f"{'PASS' if ok else 'FAIL -- verdict is vacuous'}")
    return 0 if ok else 1


def _load_control(path):
    """Load a control dump and REFUSE one with no arrays: ``np.savez``
    with no arguments writes a valid, empty archive that ``dict(np.load)``
    turns into ``{}``, which every downstream check would accept (codex
    gate-1 r3 MAJOR)."""
    import numpy as np

    try:
        ctrl = dict(np.load(path))
    except (OSError, ValueError) as e:
        raise DumpRefused(f"control {path}: unreadable ({e})")
    if not ctrl:
        raise DumpRefused(f"control {path}: archive holds no arrays")
    return ctrl


def _assemble_selftest(dirs, *, arm, mode):
    """Non-vacuity of the provenance binding: on a COPY of the first
    sweep dir, (a) one byte of census.json flipped and (b) one variant
    record's executed hash edited must both make assembly REFUSE."""
    import glob
    import json
    import os
    import shutil
    import tempfile

    src = dirs[0]
    tmp = tempfile.mkdtemp(prefix="missparity_assemble_selftest_")
    out = os.path.join(tmp, "out.json")
    rcs = {}
    import numpy as np

    def _rebind_control(d, **arrays):
        # write a DIFFERENT control and re-sign every record to it, so
        # the dir is internally consistent (the forgery codex described)
        p = os.path.join(d, "control.npz")
        np.savez(p, **arrays)
        sha = _probe_sha256(p)
        for rp in glob.glob(os.path.join(d, "variant_*.json")):
            r = json.load(open(rp))
            r["control_sha256"] = sha
            json.dump(r, open(rp, "w"))

    cases = ("census-byte", "record-hash", "control-deleted",
             "control-empty", "control-glued")
    for case in cases:
        d = os.path.join(tmp, case)
        os.makedirs(d)
        for f in ("census.json", "control.npz"):
            if os.path.exists(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), d)
        for f in glob.glob(os.path.join(src, "variant_*.json")):
            shutil.copy(f, d)
        glue = [d]
        if case == "control-empty":
            _rebind_control(d)                  # valid archive, no arrays
        elif case == "control-glued":
            # a SECOND internally-consistent dir bound to another control
            d2 = os.path.join(tmp, "control-glued-2")
            shutil.copytree(d, d2)
            _rebind_control(d2, sentinel=np.zeros(1))
            glue = [d, d2]
        elif case == "census-byte":
            p = os.path.join(d, "census.json")
            b = bytearray(open(p, "rb").read())
            i = b.index(b"\n") if b"\n" in b else 0
            b[i:i + 1] = b" "          # whitespace flip: still valid JSON
            open(p, "wb").write(bytes(b))
        elif case == "control-deleted":
            os.remove(os.path.join(d, "control.npz"))
        else:
            p = sorted(glob.glob(os.path.join(d, "variant_*.json")))[0]
            r = json.load(open(p))
            r["exec_sha256"] = "0" * 64
            json.dump(r, open(p, "w"))
        try:
            _assemble_dump(glue, arm=arm, mode=mode, out=out)
            rcs[case] = 0
        except DumpRefused as e:
            print(f"ASSEMBLE SELFTEST ({case}): REFUSED -- {e}")
            rcs[case] = 4
    ok = all(rc == 4 for rc in rcs.values())
    verdict = "PASS" if ok else "FAIL -- binding is vacuous"
    print(f"ASSEMBLE SELFTEST ({arm}): "
          + "; ".join(f"{c} -> rc {rcs[c]}" for c in cases)
          + f"; {verdict}")
    return 0 if ok else 1


def _interior(a, n, ng):
    """Interior compute-domain view of one face-stacked array: halo'd
    axes (extent n+2ng cell / n+2ng+1 node) sliced to [ng, L-ng); other
    axes kept whole."""
    sl = [slice(None)]
    for ax in (1, 2):
        if a.ndim <= ax:
            break
        L = a.shape[ax]
        sl.append(slice(ng, L - ng) if L in (n + 2 * ng, n + 2 * ng + 1)
                  else slice(None))
    return a[tuple(sl)]


def _interior_ok(flat, n, ng):
    """True if every field's INTERIOR is finite."""
    for name, a in flat.items():
        if not np.isfinite(_interior(np.asarray(a), n, ng)).all():
            return False, name
    return True, ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=int, default=24)
    ap.add_argument("--km", type=int, default=5, choices=(5, 10))
    ap.add_argument("--n-steps", type=int, default=2)
    ap.add_argument("--dt", type=float, default=120.0)
    ap.add_argument("--nh", action="store_true")
    ap.add_argument("--comm", choices=("tile", "ring"), default="tile",
                    help="which attached comm bundle the IDENTICAL "
                         "harness instruments: 'tile' (tab.tile_comm, "
                         "the (6,kt,kt) tile arm) or 'ring' "
                         "(tab.ring_comm, the certified pre-M3 face "
                         "ring = the miss-set REFERENCE arm)")
    ap.add_argument("--kt", type=int, default=1,
                    help="tile mesh kt (tile arm; needs 6*kt*kt devices)")
    ap.add_argument("--max-variants", type=int, default=None,
                    help="cap the poison sweep (progress prints keep "
                         "partial runs citable); default = ALL firings")
    ap.add_argument("--dump-misses", type=str, default=None,
                    help="write this arm's dump (JSON: arm, mode, probe "
                         "sha256, census, per-variant result); refuses "
                         "unless the sweep is COMPLETE")
    ap.add_argument("--ref-misses", type=str, default=None,
                    help="the ring arm's dump; the verdict is miss-set "
                         "EQUALITY against it, after both dumps validate")
    ap.add_argument("--verdict-only", action="store_true",
                    help="no model: verdict on --misses vs --ref-misses")
    ap.add_argument("--misses", type=str, default=None,
                    help="tile arm dump for --verdict-only")
    ap.add_argument("--expect-census", type=int, default=None,
                    help="the deck's firing census (site-map: 23 hydro "
                         "km5, 36 NH km5); a dump with another count "
                         "REFUSES")
    ap.add_argument("--verdict-selftest", action="store_true",
                    help="with --verdict-only: also prove a truncated and "
                         "an empty tile dump REFUSE")
    ap.add_argument("--assemble-dump", nargs="+", default=None,
                    metavar="DIR",
                    help="no model: build --dump-misses for --comm from "
                         "split-process sweep dir(s) holding census.json "
                         "+ variant_*.json records (--record-dir)")
    ap.add_argument("--assemble-selftest", action="store_true",
                    help="with --assemble-dump: also prove a flipped "
                         "census.json byte and a forged record hash "
                         "REFUSE assembly (on a copy)")
    ap.add_argument("--record-dir", type=str, default=None,
                    help="split mode: write census.json (--census-only) "
                         "or variant_<name>_<k>.json (--poison), each "
                         "stamped with THIS process's executed probe "
                         "sha256")
    ap.add_argument("--purity", action="store_true",
                    help="exchange-purity bitwise gate (see module doc); "
                         "ignores --comm, runs ring AND tile once")
    ap.add_argument("--dump-record", type=str, default=None,
                    help="purity: npz of the FIRST violating firing's "
                         "recorded inputs/output + certified replay")
    # split mode (NH: ~40 step compiles in one process exhaust the LLVM
    # JIT section allocator -- jobs 9591218/9589438): run the census in
    # one process, each poison variant in its own, npz handshake.
    ap.add_argument("--census-only", action="store_true",
                    help="control + firing census, then exit (writes "
                         "--dump-control / --dump-census if given)")
    ap.add_argument("--dump-control", type=str, default=None)
    ap.add_argument("--dump-census", type=str, default=None)
    ap.add_argument("--poison", type=str, default=None, metavar="NAME:K",
                    help="run ONLY this poison variant against the "
                         "control in --control-npz; prints detected/"
                         "MISS and exits")
    ap.add_argument("--control-npz", type=str, default=None)
    args = ap.parse_args(argv)
    mode = "nh" if args.nh else "hydro"
    label = ("NH" if args.nh else "hydro") + f" km{args.km}"

    if args.assemble_dump is not None:
        if not args.dump_misses:
            raise SystemExit("--assemble-dump needs --dump-misses")
        try:
            _assemble_dump(args.assemble_dump, arm=args.comm, mode=mode,
                           out=args.dump_misses)
        except DumpRefused as e:
            print(f"ASSEMBLE ({args.comm}): REFUSED -- {e}")
            return 4
        if args.assemble_selftest:
            return _assemble_selftest(args.assemble_dump, arm=args.comm,
                                      mode=mode)
        return 0

    if args.verdict_only:
        if not (args.misses and args.ref_misses):
            raise SystemExit("--verdict-only needs --misses and "
                             "--ref-misses")
        kw = dict(expect_mode=mode, expect_census=args.expect_census)
        if args.verdict_selftest:
            rc = _verdict_selftest(args.misses, args.ref_misses, label, **kw)
            if rc:
                return rc
        return _miss_set_verdict(args.misses, args.ref_misses, label, **kw)

    import jax

    jax.config.update("jax_enable_x64", True)
    devs = jax.devices()
    kt = int(args.kt)
    n_tile = 6 * kt * kt
    if len(devs) < n_tile:
        raise SystemExit(f"need {n_tile} devices "
                         f"(XLA_FLAGS=--xla_force_host_platform_device_"
                         f"count={n_tile} before jax imports)")

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
    FV3DuoConfig,
    FV3DuoDynamicsModel,
    ORACLE_DAMPING,
)
    from legoesm.grids.factory import create_fv3_duo_grid
    from legoesm.grids.fv3_duo_spmd import (
        build_tiled_ext_scalar_a_split, build_tiled_ext_scalar_b_split,
        build_tiled_ext_vector_splits)

    # the parity probe's flatten, reused (one bundle walk, not two)
    from spmd_face_shard_parity import _flatten_bundle, _shard_bundle

    tmesh = Mesh(np.array(devs[:n_tile]).reshape(6, kt, kt),
                 ("face", "tile_i", "tile_j"))
    tshard = NamedSharding(tmesh, P("face"))
    pmesh = Mesh(np.array(devs[:6]), ("face",))
    pshard = NamedSharding(pmesh, P("face"))
    cfg = FV3DuoConfig(**ORACLE_DAMPING, km=args.km, hydrostatic=not args.nh)
    bundle_grid = create_fv3_duo_grid(args.resolution)
    n, ng = bundle_grid.n, bundle_grid.ng

    def _masks(splits):
        masks = {}
        sa = _split_write_masks(splits["scalar_A"])[0]
        sb = _split_write_masks(splits["scalar_B"])[0]
        du, dv = _vector_write_masks(splits["dgrid"])
        cu, cv = _vector_write_masks(splits["cgrid"])
        masks["scalar_A"] = masks["scalar_A_allk"] = (sa,)
        masks["scalar_B"] = masks["scalar_B_allk"] = (sb,)
        masks["dgrid"] = masks["dgrid_allk"] = (du, dv)
        masks["cgrid"] = masks["cgrid_allk"] = (cu, cv)
        return masks

    def build_model(poison=None, comm=None, record=None):
        """One instrumented model on the ``comm`` arm ('tile' | 'ring').
        SAME instrument, SAME write-set masks (derived from the kt=1
        split tables -- the arm-independent halo write-set) on both
        arms; only the wrapped bundle differs."""
        comm = args.comm if comm is None else comm
        t0 = time.time()
        if comm == "tile":
            m = FV3DuoDynamicsModel(bundle_grid, cfg,
                                    step_out_shardings=tshard,
                                    step_spmd_mesh=tmesh)
            tab = m._ctx_jax.tab
            real = tab.tile_comm
            splits = real.splits if kt == 1 else {
                "scalar_A": build_tiled_ext_scalar_a_split(tab, 1),
                "scalar_B": build_tiled_ext_scalar_b_split(tab, 1),
                "dgrid": build_tiled_ext_vector_splits(tab, 1, "D"),
                "cgrid": build_tiled_ext_vector_splits(tab, 1, "C")}
            inst = InstrumentedTileComm(real, _masks(splits), poison,
                                        record)
            # attach BEFORE the first step call: the jit fn is fresh per
            # model, so this trace is the only one its cache ever holds
            tab.tile_comm = inst
        else:
            m = FV3DuoDynamicsModel(bundle_grid, cfg,
                                    step_out_shardings=pshard,
                                    step_spmd_mesh=pmesh)
            tab = m._ctx_jax.tab
            real = tab.ring_comm
            splits = {
                "scalar_A": build_tiled_ext_scalar_a_split(tab, 1),
                "scalar_B": build_tiled_ext_scalar_b_split(tab, 1),
                "dgrid": build_tiled_ext_vector_splits(tab, 1, "D"),
                "cgrid": build_tiled_ext_vector_splits(tab, 1, "C")}
            inst = InstrumentedTileComm(real, _masks(splits), poison,
                                        record)
            tab.ring_comm = inst
        return m, inst, time.time() - t0

    def run(model, n_steps=None):
        # the arm's own sharding: ring models carry ring_comm (the
        # instrument wrapper), tile models carry tile_comm
        sh = pshard if model._ctx_jax.tab.ring_comm is not None else tshard
        b = _shard_bundle(model.dcmip16_initial_state(do_pert=True), sh)
        for _ in range(args.n_steps if n_steps is None else n_steps):
            b = model.step(b, args.dt)
        jax.block_until_ready(b["state"]["pt"])
        return _flatten_bundle(b)

    # ---- exchange-purity mode ------------------------------------------
    if args.purity:
        single = FV3DuoDynamicsModel(bundle_grid, cfg)
        tab0 = single._ctx_jax.tab
        recs, reals, shards = {}, {}, {}
        for arm in ("ring", "tile"):
            rec = {}
            m_a, inst_a, tb = build_model(comm=arm, record=rec)
            t0 = time.time()
            run(m_a, n_steps=1)
            recs[arm] = rec
            reals[arm] = inst_a._real
            shards[arm] = pshard if arm == "ring" else tshard
            print(f"[purity {arm}] build {tb:.1f}s, 1 step + compile "
                  f"{time.time() - t0:.1f}s, {len(rec)} firings recorded")
        if set(recs["ring"]) != set(recs["tile"]):
            raise SystemExit(f"PURITY: firing census differs ring "
                             f"{sorted(recs['ring'])} vs tile "
                             f"{sorted(recs['tile'])}")
        # cell classes over the FLAT arrays (kt=1 split == flat layout):
        # weighted-stencil targets (k2e / corner Lagrange / projection-
        # derived writes) vs weight-one copy targets
        splits0 = {
            "scalar_A": build_tiled_ext_scalar_a_split(tab0, 1),
            "scalar_B": build_tiled_ext_scalar_b_split(tab0, 1),
            "dgrid": build_tiled_ext_vector_splits(tab0, 1, "D"),
            "cgrid": build_tiled_ext_vector_splits(tab0, 1, "C")}
        write_m = _masks(splits0)
        arith_m = _arith_masks(splits0)
        jit_cache = {}

        def jitted(arm, name):
            """STANDALONE-jitted REAL exchange of ``arm`` (no instrument,
            its own XLA program) on host arrays -> host arrays."""
            if (arm, name) in jit_cache:
                return jit_cache[(arm, name)]
            real, sh = reals[arm], shards[arm]
            if name.startswith("scalar_"):
                stag = name.split("_")[1]
                base = (real.ext_scalar_allk if name.endswith("_allk")
                        else real.ext_scalar)
                f = jax.jit(lambda x: base(x, stag))

                def call(ins):
                    return (np.asarray(f(jax.device_put(ins[0], sh))),)
            else:
                base = {"dgrid": real.ext_vector_dgrid,
                        "dgrid_allk": real.ext_vector_dgrid_allk,
                        "cgrid": real.ext_vector_cgrid,
                        "cgrid_allk": real.ext_vector_cgrid_allk}[name]
                f = jax.jit(lambda u, v: base(u, v))

                def call(ins):
                    return tuple(np.asarray(o) for o in f(
                        *[jax.device_put(a, sh) for a in ins]))
            jit_cache[(arm, name)] = call
            return call

        keys = sorted(recs["ring"], key=lambda t: (t[0], t[1]))
        primary, secondary, direct_bad = [], [], []
        first_div = None
        n_direct = n_exec = n_inputs = 0
        dumped = False
        for key in keys:
            name = key[0]
            rl, tl = recs["ring"][key], recs["tile"][key]
            # a firing traced inside a lax.scan body (acoustic middle
            # sub-steps, tracer subcycle) EXECUTES once per trip: one
            # record per execution, compared execution-by-execution
            if len(rl) != len(tl):
                raise SystemExit(f"PURITY: {name}#{key[1]} executed "
                                 f"{len(rl)}x on ring, {len(tl)}x on tile")
            for x, ((ri, ro), (ti, to)) in enumerate(zip(rl, tl)):
                n_exec += 1
                tag_x = f"{name}#{key[1]}" + (f"[exec {x}]"
                                              if len(rl) > 1 else "")
                for src, ins, outs in (("ring", ri, ro), ("tile", ti, to)):
                    n_inputs += 1
                    # PRIMARY: same input, ring-jitted vs tile-jitted
                    pr = jitted("ring", name)(ins)
                    pt = jitted("tile", name)(ins)
                    for j, (a, b) in enumerate(zip(pr, pt)):
                        if not np.array_equal(a, b, equal_nan=True):
                            primary.append((f"{src}-input", tag_x, j,
                                            _cell_stats(a, b, write_m[name][j],
                                                        arith_m[name][j])))
                    # SECONDARY: in-graph recorded output of the arm that
                    # produced this input vs its own standalone jit, and
                    # vs the eager certified impl (fusion-class diagnostic)
                    own = pr if src == "ring" else pt
                    eag = _certified_replay(name, tab0, ins)
                    # ARM-vs-CERTIFIED (the gate's load-bearing rows):
                    # each arm's standalone jit vs the certified eager
                    # impl on THIS input, whichever arm recorded it
                    for lab, o_arm, ref in (
                            (f"{src} in-graph vs standalone-jit", outs, own),
                            (f"{src} in-graph vs eager impl", outs, eag),
                            ("tile-jit vs certified impl", pt, eag),
                            ("ring-jit vs certified impl", pr, eag)):
                        for j, (o, e) in enumerate(zip(o_arm, ref)):
                            if not np.array_equal(o, e, equal_nan=True):
                                secondary.append((lab, tag_x, j, _cell_stats(
                                    o, e, write_m[name][j], arith_m[name][j])))
                                if args.dump_record and not dumped:
                                    np.savez(args.dump_record, out=o, exp=e,
                                             **{f"in{i}": a for i, a
                                                in enumerate(ins)})
                                    dumped = True
                in_eq = all(np.array_equal(a, b, equal_nan=True)
                            for a, b in zip(ri, ti))
                if in_eq:
                    n_direct += 1
                    for j, (a, b) in enumerate(zip(ro, to)):
                        if not np.array_equal(a, b, equal_nan=True):
                            direct_bad.append(("ring-vs-tile in-graph", tag_x, j,
                                               _cell_stats(a, b, write_m[name][j],
                                                           arith_m[name][j])))
                elif first_div is None:
                    dmax = max(float(np.abs(a - b).max())
                               for a, b in zip(ri, ti))
                    first_div = (tag_x, dmax)
        print(f"[purity] {len(keys)} firings, {n_exec} executions, "
              f"{n_inputs} recorded inputs; ring-vs-tile in-graph inputs "
              f"bit-equal at {n_direct}/{n_exec} executions")
        if first_div is not None:
            print(f"[purity] first ring-vs-tile in-graph INPUT divergence at "
                  f"{first_div[0]} (max abs {first_div[1]:.2e}) -- kernel "
                  f"lowering class, upstream of the exchange")
        else:
            print("[purity] ring and tile in-graph inputs bit-equal at "
                  "EVERY firing")
        for lab, tag_x, j, st in primary:
            print(f"EXCHANGE IDENTITY VIOLATION ({lab}): {tag_x} output[{j}] "
                  f"ring-jit != tile-jit on the SAME input: {st}")
        for lab, tag_x, j, st in direct_bad:
            print(f"IN-GRAPH DIRECT MISMATCH ({lab}): {tag_x} output[{j}]: {st}")
        worst_rel, n_copy_bad, n_out_bad = 0.0, 0, 0
        per_lab = {}
        for lab, tag_x, j, st in secondary:
            print(f"secondary ({lab}): {tag_x} output[{j}]: {st['text']}")
            worst_rel = max(worst_rel, st["rel"])
            n_copy_bad += st["copy"]
            n_out_bad += st["outside"]
            per_lab[lab] = per_lab.get(lab, 0) + 1
        for lab, tag_x, j, st in primary + direct_bad:
            n_copy_bad += st["copy"]
            n_out_bad += st["outside"]
        n_tile_vs_cert = per_lab.get("tile-jit vs certified impl", 0)
        n_ring_vs_cert = per_lab.get("ring-jit vs certified impl", 0)
        print(f"[purity] same-input ring-jit vs tile-jit violations="
              f"{len(primary)}; in-graph direct mismatches at equal "
              f"inputs={len(direct_bad)}; TILE-jit vs certified impl "
              f"mismatches={n_tile_vs_cert}/{n_inputs} inputs; RING-jit vs "
              f"certified impl mismatches={n_ring_vs_cert}/{n_inputs}; "
              f"all mismatch rows={len(secondary)} -- cells outside the "
              f"exchange write-set={n_out_bad}, in weight-one COPY "
              f"targets={n_copy_bad}, worst rel={worst_rel:.2e}")
        tag = f"{label} kt={kt}"
        # GATE: the TILE arm's exchange, standalone-jitted, must equal the
        # certified single-device impl BITWISE on every recorded input
        # (both arms' inputs), and no mismatch anywhere may touch a
        # weight-one copy cell or a cell outside the write-set.  The ring
        # arm's own agreement with the certified impl is REPORTED (it is
        # the pre-M3 lane's property, not the port's); ring-vs-tile
        # identity follows from both equalling the certified impl.
        if n_tile_vs_cert or n_copy_bad or n_out_bad:
            print(f"EXCHANGE PURITY GATE ({tag}): FAIL")
            return 1
        print(f"EXCHANGE PURITY GATE ({tag}): PASS -- tile-jit == certified "
              f"impl BITWISE on all {n_inputs} recorded inputs; ring-jit "
              f"vs certified impl mismatches={n_ring_vs_cert} (ring lane "
              f"finding, arith cells only); no copy/outside-writeset cell "
              f"ever differs (worst rel {worst_rel:.2e})")
        return 0

    # ---- single-variant mode (split sweep) ----------------------------
    if args.poison is not None:
        name, kk = args.poison.rsplit(":", 1)
        ctrl = _load_control(args.control_npz)
        census_bind = None
        if args.record_dir:
            # verify the census PROCESS ran these same probe bytes and
            # this arm/mode, and hash its census.json bytes for binding
            import json
            import os

            cpath = os.path.join(args.record_dir, "census.json")
            try:
                census_bind = _probe_sha256(cpath)
                with open(cpath) as fh:
                    c = json.load(fh)
                _require_executed_here(c.get("exec_sha256"), cpath)
                if c.get("arm") != args.comm or c.get("mode") != mode:
                    raise DumpRefused(f"{cpath}: arm/mode mismatch with "
                                      f"this variant process")
                if f"{name} {kk}" not in c.get("census", []):
                    raise DumpRefused(f"{cpath}: variant {name}#{kk} not "
                                      f"in the census")
            except (OSError, ValueError, DumpRefused) as e:
                print(f"[poison] {name}#{kk}: REFUSED -- {e}")
                return 4
        m_p, _, _ = build_model(poison=(name, int(kk)))
        flat = run(m_p)
        okp, _bad = _interior_ok(flat, n, ng)
        moved = None
        if okp:
            moved = any(not np.array_equal(np.asarray(flat[f]),
                                           ctrl[f], equal_nan=True)
                        for f in ctrl)
            print(f"[poison] {name}#{kk}: MISS (interior finite; state "
                  f"{'moved' if moved else 'IDENTICAL to control'})")
        else:
            print(f"[poison] {name}#{kk}: detected (interior non-finite in "
                  f"{_bad})")
        if args.record_dir:
            # THIS process's record, stamped with the probe bytes it ran
            # and BOUND to the census/control artifacts it consumed
            _write_record(args.record_dir, f"variant_{name}_{kk}.json",
                          {"variant": f"{name} {kk}",
                           "result": "miss" if okp else "detected",
                           "moved": moved, "arm": args.comm, "mode": mode,
                           "census_sha256": census_bind,
                           "control_sha256": _probe_sha256(args.control_npz)})
        return 2 if okp else 0

    # ---- pass 1: control + firing census -------------------------------
    model, inst, t_build = build_model()
    t0 = time.time()
    control = run(model)
    print(f"[control {args.comm}] build {t_build:.1f}s, {args.n_steps} "
          f"steps + compile {time.time() - t0:.1f}s")
    ok, bad = _interior_ok(control, n, ng)
    if not ok:
        raise SystemExit(f"CONTROL non-finite in {bad} -- instrument or "
                         f"step broken, poison sweep meaningless")
    # instrument-sanity control: the G0 TRIAGE contract (module doc).
    # Compare the instrumented control against BOTH the certified
    # single-device step (d_ctrl) and the PLAIN GSPMD face-sharded step
    # (d_plain, the lane's accepted lowering class; an absolute envelope
    # tripped on zero-scale w: 4.1e-12 on a 4e-3-scale field IS that
    # class, job 9589438), and require per-field
    # d_ctrl <= 1e-13 + 4 * d_plain.  Gross exchange breakage is
    # O(field)-sized and still fails this instantly.
    single = FV3DuoDynamicsModel(bundle_grid, cfg)
    bs = single.dcmip16_initial_state(do_pert=True)
    for _ in range(args.n_steps):
        bs = single.step(bs, args.dt)
    ref = _flatten_bundle(bs)
    plain_m = FV3DuoDynamicsModel(bundle_grid, cfg,
                                  step_out_shardings=pshard)
    bp = _shard_bundle(plain_m.dcmip16_initial_state(do_pert=True), pshard)
    for _ in range(args.n_steps):
        bp = plain_m.step(bp, args.dt)
    plain = _flatten_bundle(bp)
    worst = ("", 0.0, 0.0)
    for k in sorted(ref):
        a = np.asarray(ref[k], float)
        d_ctrl = float(np.abs(a - np.asarray(control[k], float)).max())
        d_plain = float(np.abs(a - np.asarray(plain[k], float)).max())
        if d_ctrl > 1e-13 + 4.0 * d_plain:
            raise SystemExit(
                f"CONTROL triage at {k}: d_ctrl={d_ctrl:.3e} > 1e-13 + "
                f"4 x d_plain={d_plain:.3e} -- outside the accepted "
                f"sharded-lowering class, poison sweep meaningless")
        if d_ctrl > worst[1]:
            worst = (k, d_ctrl, d_plain)
    print(f"[control {args.comm}] interior finite; triage vs plain GSPMD "
          f"reference passed (worst {worst[0]}: d_ctrl={worst[1]:.2e}, "
          f"d_plain={worst[2]:.2e})")
    # jit traces ONCE regardless of n_steps -- counts are per traced step
    census = dict(inst.counts)
    if args.dump_control:
        np.savez(args.dump_control,
                 **{k: np.asarray(v) for k, v in control.items()})
    if args.dump_census:
        with open(args.dump_census, "w") as fh:
            for name in sorted(census):
                for k in range(census[name]):
                    fh.write(f"{name} {k}\n")
    if args.record_dir:
        _write_record(args.record_dir, "census.json",
                      {"arm": args.comm, "mode": mode,
                       "census": [f"{name} {k}" for name in sorted(census)
                                  for k in range(census[name])]})
    if args.census_only:
        print(f"[census] written; exiting (census-only)")
        return 0
    print(f"[census {args.comm}] firings per traced step: {census} "
          f"(total {sum(census.values())})")

    # ---- pass 2: poison sweep ------------------------------------------
    variants = [(name, k) for name in sorted(census)
                for k in range(census[name])]
    if args.max_variants is not None:
        variants = variants[:args.max_variants]
    misses = []
    for i, (name, k) in enumerate(variants):
        t0 = time.time()
        m_p, _, _ = build_model(poison=(name, k))
        flat = run(m_p)
        okp, _bad = _interior_ok(flat, n, ng)
        if okp:
            # distinguish vacuous-firing from outside-window consumption
            moved = any(
                not np.array_equal(np.asarray(flat[f]),
                                   np.asarray(control[f]), equal_nan=True)
                for f in control)
            misses.append((name, k, moved))
            print(f"[poison {i + 1}/{len(variants)}] {name}#{k}: MISS "
                  f"(interior finite; state "
                  f"{'moved' if moved else 'IDENTICAL to control'}) "
                  f"[{time.time() - t0:.0f}s]")
        else:
            print(f"[poison {i + 1}/{len(variants)}] {name}#{k}: "
                  f"detected (interior non-finite in {_bad}) "
                  f"[{time.time() - t0:.0f}s]")
    print(f"[poison {args.comm}] {len(variants) - len(misses)}/"
          f"{len(variants)} firings detected")
    for name, k, moved in misses:
        print(f"MISS ({args.comm}): {name}#{k} "
              f"({'writes unread in window' if moved else 'VACUOUS?'})")
    if args.dump_misses:
        # a REFERENCE (or gated) dump is only ever a COMPLETE sweep of
        # the traced census: a capped (--max-variants) or short sweep
        # refuses here rather than recording a partial miss set
        miss_ids = {f"{nm} {k}" for nm, k, _ in misses}
        results = {f"{nm} {k}": ("miss" if f"{nm} {k}" in miss_ids
                                 else "detected") for nm, k in variants}
        try:
            if args.max_variants is not None:
                raise DumpRefused("--max-variants caps the sweep; a capped "
                                  "sweep cannot be a miss-set arm")
            _write_dump(args.dump_misses, arm=args.comm, mode=mode,
                        census=[(nm, k) for nm in sorted(census)
                                for k in range(census[nm])],
                        results=results,
                        variant_sha={v: _EXEC_SHA256 for v in results})
        except DumpRefused as e:
            print(f"DYNAMIC POISON GATE ({label}): REFUSED -- {e}")
            return 4
        print(f"[dump {args.comm}] complete sweep {len(results)}/"
              f"{sum(census.values())} -> {args.dump_misses}")
    if args.ref_misses is None:
        if args.comm == "ring":
            print(f"REFERENCE ARM ({label}): ring miss set recorded "
                  f"({len(misses)} misses); no verdict on the reference")
            return 0
        print(f"DYNAMIC POISON GATE ({label}): NO VERDICT -- the miss-set "
              f"rule needs the ring reference (--ref-misses)")
        return 3
    if not args.dump_misses:
        print(f"DYNAMIC POISON GATE ({label}): NO VERDICT -- the tile arm "
              f"must also --dump-misses (verdict reads validated dumps)")
        return 3
    return _miss_set_verdict(args.dump_misses, args.ref_misses, label,
                             expect_mode=mode,
                             expect_census=args.expect_census)


if __name__ == "__main__":
    raise SystemExit(main())
