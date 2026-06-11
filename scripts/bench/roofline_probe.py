#!/usr/bin/env python
"""Measured roofline + collective-microbenchmark harness (KEEPER).

Codex strategy review ranked this the #1 highest-leverage scaling move:
it turns "we are near the hardware limit" into a *falsifiable* target by
measuring the four ceilings separately instead of inferring them from
efficiency ratios.  Pure measurement — this harness changes NO model
numerics; it imports the same collective primitives the model uses
(``legoesm.parallel`` ppermute / sendrecv / batch_allreduce) and times
them in isolation.

The four ceilings (each measured independently)
-----------------------------------------------
1. **Sustained device bandwidth** (``measure_sustained_bandwidth``):
   a jit'd streaming triad ``y = a*x + y`` over a sweep of array sizes
   spanning L2→HBM.  Reports achieved GB/s and %-of-spec-peak.  The
   large-size asymptote is the HBM (or DRAM) ceiling; the small-size
   numbers expose cache effects (and are NOT the HBM ceiling — see the
   `hits_hbm` flag and the working-set-vs-cache documentation below).

2. **Collective latency + bandwidth by payload**
   (``measure_ppermute_collective`` + ``measure_sendrecv_collective``):
   two transports over a 2^10..2^22 element payload sweep —
     (a) GPU/SPMD ``jax.lax.ppermute`` PRIMITIVE inside a ``shard_map``
         over a 2-device ``("face",)`` mesh — the building block of the
         cubed-sphere halo (``cubesphere_exchange._make_exchange_ppermute``
         uses the same primitive), measured as the per-round collective
         FLOOR, NOT the full multi-round halo-exchange cost;
     (b) CPU ``mpi4jax`` sendrecv via the model's AD-safe
         ``halo_exchange.get_sendrecv_vjp`` wrapper, under ``mpirun``.
   Reports the latency floor (smallest payload) and the asymptotic GB/s
   (largest payload).

3. **Allreduce latency by scalar-batch size**
   (``measure_psum_batch_sweep`` + ``measure_mpi_allreduce_batch_sweep``):
   batches {1,2,4,8,16} via both ``jax.lax.psum`` on an SPMD mesh and
   ``mpi4jax``-backed ``reductions.batch_allreduce_mpi``.
   This is the weak-scaling Amdahl term (the barotropic PCG pays two of
   these per iteration); reports per-call latency vs batch size.

4. **Kernel-launch / scan-dispatch floor** (``measure_dispatch_floor``):
   a jit'd near-zero-work identity on a tiny array, timed both as a bare
   ``jit`` call and inside a ``lax.scan``.  Reports the dispatch overhead
   per step (the lower bound on ms/step no model can beat).

Per-step roofline reporter
---------------------------
``report_step_roofline`` is callable on any measured ms/step + grid +
n_levels (e.g. a ``run_levante_gpu_scaling.TimingResult``).  Given the
sustained-BW ceiling (#1) and the collective floors (#2/#3) it reports
useful cell-levels/s, *inferred* bytes-moved/cell-level, achieved GB/s,
%-of-sustained-BW, the HLO collective census (reusing
``cubesphere_exchange``/``run_levante`` helpers), and the collective
time-floor.  For PCG it evaluates the Amdahl model
``T = local_stencil + M*(2*allreduce_lat + halo + stencil)`` and reports
the nonlocal fraction.

Inferred-bytes-moved assumption (IT IS AN ESTIMATE — labelled everywhere)
-------------------------------------------------------------------------
A memory-bound finite-volume step moves, per prognostic cell-level, at
minimum one READ + one WRITE of every prognostic field, plus halo/stencil
re-reads of neighbour cells that the cache cannot hold.  We model

    bytes_per_cell_level ≈ n_state_fields * dtype_bytes
                           * (rw_factor + stencil_reuse_factor)

with documented, overridable factors (see ``BytesModel``):
  * ``rw_factor = 2.0``    — one read + one write of each field (exact
    for a single pass; a lower bound, since a real step has several
    sub-passes: advection, PGF, mixing).
  * ``stencil_reuse_factor`` — extra neighbour re-reads NOT served from
    cache.  Default ``1.0`` (≈ one extra full re-read of the volume),
    justified below.  This is the single most uncertain term and is
    reported as an ESTIMATE with an explicit lower/upper bracket on the
    inferred bytes moved.  Achieved GB/s = (inferred bytes) / time, so
    ``rw_factor`` alone (fewest bytes) → the LOWER achieved-GB/s bound and
    ``rw + 2*stencil`` (most bytes) → the UPPER achieved-GB/s bound — the
    %-of-peak is therefore always a RANGE, never a false-precision point.

Why ``stencil_reuse_factor = 1.0`` is the DEFAULT SCENARIO (an assumption,
NOT a validated measurement — codex MAJOR): a 2nd-order FV horizontal
stencil touches the 4 face-neighbours; a 3rd-/4th-order or TVD-limited
flux touches 2 cells each side (the model's TVD pad is width 2).  The
*plausibility argument* is that with cache-friendly XLA tiling a centre
cell's read is reused across its neighbours' stencils, so the incremental
DRAM traffic beyond the single-pass read is of order the halo ring plus
cross-pass evictions — O(1)× the volume rather than O(stencil-width)×.
But this harness does NOT measure HLO memory traffic, hardware byte
counters, or tile shapes, so ``1.0`` is a documented modelling CHOICE,
not a proven fact.  Accordingly we report the BRACKET [rw_only,
rw + 2*stencil] as the headline and treat the point value (rw +
1*stencil) as a labelled *central scenario*, never as a validated
number.  The honest output is "achieved BW is X–Y% of sustained,
depending on the stencil-reuse assumption" — the falsifiable target codex
asked for, with the assumption surfaced.  To replace the assumption with
a measured number, pass ``--stencil-reuse-factor`` derived from a
profiler (e.g. nsys/ncu DRAM counters) for the real step.

Output
------
``--output-dir`` gets ``roofline_<host>_<tag>.json`` (all raw
measurements + assumptions) and, if matplotlib is importable,
``roofline_<host>_<tag>.png`` with four panels: achieved GB/s vs
%-peak per device; collective latency vs payload; allreduce latency vs
batch size; the PCG Amdahl breakdown.

Usage
-----
GPU (1 or 2 devices; ppermute needs >=2, gracefully skipped on 1)::

    JAX_PLATFORMS=cuda python scripts/bench/roofline_probe.py \
        --output-dir results/scaling_ginsburg/roofline

CPU MPI (sendrecv + mpi4jax allreduce sweeps)::

    JAX_PLATFORMS=cpu mpirun -np 2 python scripts/bench/roofline_probe.py \
        --output-dir results/scaling_ginsburg/roofline

Single CPU process (BW + dispatch only; collectives need >=2 ranks)::

    JAX_PLATFORMS=cpu python scripts/bench/roofline_probe.py \
        --output-dir results/scaling_ginsburg/roofline
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# JAX_PLATFORMS is honored by the caller (sbatch sets cuda or cpu); we do
# not force it here so the same script runs on either backend.
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Device spec-peak bandwidth table (GB/s).  Used ONLY to report achieved
# vs spec %-peak; the achieved number is the measurement of record.  Keyed
# by a normalized device-kind substring.  Add entries as new hardware is
# benchmarked — an unknown device reports peak=None (achieved-only).
# ---------------------------------------------------------------------------
SPEC_PEAK_GBPS: dict[str, float] = {
    # NVIDIA Quadro RTX 8000 (Ginsburg): GDDR6, 384-bit, ~624 GB/s.
    "rtx 8000": 624.0,
    "quadro rtx 8000": 624.0,
    # Common Ginsburg/Levante accelerators (handy if the harness is reused).
    "a100": 1555.0,         # A100 80GB SXM HBM2e
    "v100": 900.0,          # V100 SXM2 HBM2
    "rtx a6000": 768.0,
}

# A CPU rank's "peak" is its share of socket DRAM bandwidth; there is no
# single right number without lscpu/STREAM, so CPU peak is reported as
# None (achieved-only) unless the user passes --cpu-peak-gbps.


def _normalize_device_name(name: str) -> str:
    return " ".join(name.lower().replace("nvidia", "").split())


def _lookup_spec_peak(device_name: str) -> float | None:
    norm = _normalize_device_name(device_name)
    # Longest-key-first so "quadro rtx 8000" wins over "rtx 8000".
    for key in sorted(SPEC_PEAK_GBPS, key=len, reverse=True):
        if key in norm:
            return SPEC_PEAK_GBPS[key]
    return None


# ---------------------------------------------------------------------------
# Bytes-moved model (DOCUMENTED ESTIMATE — see module docstring)
# ---------------------------------------------------------------------------
@dataclass
class BytesModel:
    """Inferred bytes-moved-per-cell-level model.  EVERY field is a
    documented assumption; the achieved-BW output is a BRACKET, not a
    point estimate, precisely because ``stencil_reuse_factor`` is the
    uncertain term.

    Defaults ground out on the dry non-hydrostatic cubed-sphere state
    (``core.state.NonHydrostaticState``): 5 volumetric prognostic fields
    u, v, w, theta', rho' (phis is 2D/static; tracers add to the count
    in moist runs and the caller should raise ``n_state_fields``).
    """

    n_state_fields: int = 5
    dtype_bytes: int = 8  # float64 default; pass 4 for float32 runs.
    rw_factor: float = 2.0          # 1 read + 1 write per field per pass.
    stencil_reuse_factor: float = 1.0  # DEFAULT SCENARIO, not measured.
    # Bracket multipliers on the stencil term.  achieved GB/s=bytes/time,
    # so fewest bytes -> LOWER achieved GB/s and most bytes -> UPPER.
    stencil_lo_mult: float = 0.0    # rw only -> fewest bytes -> lower GB/s
    stencil_hi_mult: float = 2.0    # rw + 2*stencil -> most bytes -> upper GB/s

    def _bytes(self, stencil_mult: float) -> float:
        return (
            self.n_state_fields
            * self.dtype_bytes
            * (self.rw_factor + stencil_mult * self.stencil_reuse_factor)
        )

    def bytes_per_cell_level_central(self) -> float:
        return self._bytes(1.0)

    def bytes_per_cell_level_bracket(self) -> tuple[float, float]:
        """(lower-bytes, upper-bytes); via achieved=bytes/time these map
        to (lower-GB/s, upper-GB/s) with the SAME ordering."""
        return self._bytes(self.stencil_lo_mult), self._bytes(self.stencil_hi_mult)

    def describe(self) -> dict:
        lo_b, hi_b = self.bytes_per_cell_level_bracket()
        return {
            "n_state_fields": self.n_state_fields,
            "dtype_bytes": self.dtype_bytes,
            "rw_factor": self.rw_factor,
            "stencil_reuse_factor": self.stencil_reuse_factor,
            "bytes_per_cell_level_central": self.bytes_per_cell_level_central(),
            "bytes_per_cell_level_lower": lo_b,
            "bytes_per_cell_level_upper": hi_b,
            "ESTIMATE_note": (
                "bytes/cell-level is an INFERRED ESTIMATE, not a measured "
                "value: n_state_fields*dtype_bytes*(rw_factor + "
                "stencil_mult*stencil_reuse_factor). rw=2 (1R+1W/field) is a "
                "single-pass lower bound; stencil_reuse_factor=1.0 is a "
                "DEFAULT SCENARIO (not profiled). Headline is the RANGE over "
                "bracket=[0,2]; the central value is a labelled scenario. "
                "Pass a profiler-derived --stencil-reuse-factor to replace it."
            ),
        }


# ===========================================================================
# Timing primitive: warmup + block_until_ready boundary
# ===========================================================================
def _time_callable(fn, *, n_warmup: int, n_timing: int, mode: str = "latency"):
    """Time a no-arg JAX callable.  The deliberate ``block_until_ready``
    boundaries are the ONLY host syncs in the timed region (CLAUDE.md
    JAX rule).  Warmup runs (compile + caches) are blocked-on and
    EXCLUDED.  Returns (mean_ms_per_call, raw_seconds_total).

    ``mode`` selects the timing semantics (codex BLOCKER: a single
    pipelined loop measures *throughput*, not per-call *latency*):

    * ``"latency"`` — ``block_until_ready`` after EVERY call, so the host
      cannot let XLA overlap/queue successive launches.  Each call's full
      dispatch→execute→complete round-trip is on the clock; the mean is
      the true per-call latency floor.  This is the right mode for the
      small-payload collective/dispatch points.
    * ``"throughput"`` — dispatch all ``n_timing`` calls, block once on
      the last (XLA may pipeline them).  Reports steady-state
      back-to-back throughput, the right mode for large-payload
      bandwidth points where pipelining is the realistic regime.

    DCE note: every retained ``out`` is blocked-on, and the kernels here
    have data-dependent outputs (triad, ppermute, sendrecv), so XLA
    cannot elide a call whose result is observed.
    """
    if mode not in ("latency", "throughput"):
        raise ValueError(f"unknown timing mode {mode!r}")

    out = None
    for _ in range(n_warmup):
        out = fn()
    if out is not None:
        jax.block_until_ready(out)

    if mode == "latency":
        t0 = time.perf_counter()
        for _ in range(n_timing):
            out = fn()
            jax.block_until_ready(out)
        t1 = time.perf_counter()
    else:  # throughput
        t0 = time.perf_counter()
        out = None
        for _ in range(n_timing):
            out = fn()
        if out is not None:
            jax.block_until_ready(out)
        t1 = time.perf_counter()
    total = t1 - t0
    return (total / n_timing) * 1e3, total


# ===========================================================================
# Ceiling 1: sustained device bandwidth (streaming triad)
# ===========================================================================
@dataclass
class BandwidthPoint:
    n_elements: int
    bytes_moved: int       # 2 reads (x, y) + 1 write (y) of the array.
    working_set_bytes: int  # x + y resident = 2 * array bytes.
    dtype: str
    time_per_call_ms: float
    achieved_gbps: float
    hits_hbm: bool          # working set > est. cache → DRAM-bound.


@dataclass
class BandwidthResult:
    device_kind: str
    device_platform: str
    spec_peak_gbps: float | None
    est_cache_bytes: int
    points: list[BandwidthPoint] = field(default_factory=list)
    # Sustained = MEDIAN over the largest DRAM-bound points (the plateau),
    # NOT max (codex MAJOR: a single threshold-crossing or TLB-favoured
    # point can overstate the HBM ceiling).  ``peak_gbps`` keeps the raw
    # max as a separate diagnostic.
    sustained_gbps: float = 0.0
    peak_gbps: float = 0.0            # raw max over DRAM-bound pts (diag).
    sustained_pct_peak: float | None = None
    n_plateau_points: int = 0         # how many largest pts the median used.


def _estimate_cache_bytes(platform_name: str) -> int:
    """Rough last-level-cache size used ONLY to flag which BW points are
    DRAM-bound (``hits_hbm``).  GPU L2 is small (RTX 8000 ≈ 6 MB; A100 ≈
    40 MB) relative to HBM working sets, so the large-array points are
    safely HBM-bound.  CPU LLC is bigger; we use a conservative 32 MB so
    only genuinely large arrays are tagged DRAM-bound.  Override with
    ``--cache-bytes``.  This is a HEURISTIC for the flag, not a measured
    value, and does not affect the achieved-GB/s numbers."""
    if platform_name == "gpu":
        return 8 * 1024 * 1024     # ~RTX 8000-class L2 upper bound.
    return 32 * 1024 * 1024        # conservative CPU LLC.


def measure_sustained_bandwidth(
    *,
    sizes_mb: list[float],
    dtype=jnp.float32,
    n_warmup: int = 5,
    n_timing: int = 50,
    spec_peak_gbps: float | None = None,
    cache_bytes: int | None = None,
) -> BandwidthResult:
    """Measure sustained streaming bandwidth via a jit'd triad
    ``y = a*x + y`` over a sweep of array sizes.

    The triad moves ``3 * array_bytes`` per call (read x, read y, write y)
    and has arithmetic intensity 2 FLOP / 12 (or 24) bytes — deeply
    memory-bound, so the achieved GB/s is the memory ceiling, not a FLOP
    ceiling.  Sizes span L2→HBM: the small-array points sit in cache (NOT
    the HBM ceiling, flagged ``hits_hbm=False``); the large-array
    asymptote IS the HBM/DRAM ceiling.

    dtype default float32 (the bandwidth ceiling is dtype-agnostic in
    bytes/s; float32 lets us reach large element counts without OOM on a
    48 GB card and avoids the x64 flag).  The reported number is GB/s, so
    it transfers directly to f64 runs.
    """
    devs = jax.devices()
    dev0 = devs[0]
    platform_name = str(getattr(dev0, "platform", "cpu"))
    device_kind = str(getattr(dev0, "device_kind", getattr(dev0, "platform", "cpu")))
    if spec_peak_gbps is None:
        spec_peak_gbps = _lookup_spec_peak(device_kind)
    if cache_bytes is None:
        cache_bytes = _estimate_cache_bytes(platform_name)

    itemsize = jnp.dtype(dtype).itemsize
    a = jnp.asarray(2.0, dtype=dtype)

    @jax.jit
    def _triad(x, y, a):
        return a * x + y

    res = BandwidthResult(
        device_kind=device_kind,
        device_platform=platform_name,
        spec_peak_gbps=spec_peak_gbps,
        est_cache_bytes=cache_bytes,
    )

    for size_mb in sizes_mb:
        n = int(size_mb * 1024 * 1024 / itemsize)
        if n < 1024:
            n = 1024
        # Deterministic, cheap to materialize on device (no host transfer
        # in the timed region — built once here, before timing).
        x = jnp.ones((n,), dtype=dtype)
        y = jnp.zeros((n,), dtype=dtype)
        # Place on device 0 explicitly and block so allocation/H2D is done
        # before timing.
        x = jax.device_put(x, dev0)
        y = jax.device_put(y, dev0)
        jax.block_until_ready((x, y))

        # Latency mode (block each call): guarantees every triad actually
        # streams 3*array_bytes — pipelined/CSE'd repeats of an identical
        # (x,y) input could otherwise be coalesced by XLA.  At the large
        # DRAM-bound sizes the per-call block overhead (~tens of us) is
        # negligible vs the kernel time (a 512 MB triad at ~600 GB/s is
        # ~2.5 ms), so this reports the true streaming bandwidth.
        ms, _ = _time_callable(
            lambda: _triad(x, y, a), n_warmup=n_warmup, n_timing=n_timing,
            mode="latency",
        )
        array_bytes = n * itemsize
        bytes_moved = 3 * array_bytes        # read x, read y, write y.
        working_set = 2 * array_bytes        # x + y resident.
        gbps = bytes_moved / (ms * 1e-3) / 1e9
        res.points.append(BandwidthPoint(
            n_elements=n,
            bytes_moved=bytes_moved,
            working_set_bytes=working_set,
            dtype=str(jnp.dtype(dtype)),
            time_per_call_ms=ms,
            achieved_gbps=gbps,
            hits_hbm=working_set > cache_bytes,
        ))

    # Sustained = MEDIAN over the largest DRAM-bound points — the plateau,
    # not a single max (codex MAJOR: a threshold-crossing or TLB-favoured
    # point can overstate the HBM ceiling).  We take the median of the
    # top-``k`` largest-working-set DRAM-bound points (k = min(4, #pts)).
    # If none cross the cache threshold the caller should widen
    # --bw-sizes-mb; we then fall back to the largest points overall.
    hbm_pts = [p for p in res.points if p.hits_hbm]
    pool = hbm_pts if hbm_pts else res.points
    if pool:
        # Largest-working-set first.
        pool_sorted = sorted(
            pool, key=lambda p: p.working_set_bytes, reverse=True)
        k = min(4, len(pool_sorted))
        plateau = [p.achieved_gbps for p in pool_sorted[:k]]
        res.n_plateau_points = k
        res.peak_gbps = max(p.achieved_gbps for p in pool)
        res.sustained_gbps = float(np.median(plateau))
        if spec_peak_gbps:
            res.sustained_pct_peak = 100.0 * res.sustained_gbps / spec_peak_gbps
    return res


# ===========================================================================
# Ceiling 2: collective latency + bandwidth by payload
# ===========================================================================
@dataclass
class CollectivePoint:
    n_elements: int
    payload_bytes: int       # one direction, per call.
    dtype: str
    time_per_call_ms: float
    achieved_gbps: float     # payload_bytes / time (one-directional).


@dataclass
class CollectiveResult:
    transport: str           # "ppermute_spmd" | "mpi4jax_sendrecv".
    n_participants: int
    dtype: str
    points: list[CollectivePoint] = field(default_factory=list)
    latency_floor_ms: float = 0.0     # smallest-payload time.
    asymptotic_gbps: float = 0.0      # largest-payload achieved GB/s.


def _payload_sweep(log2_lo: int, log2_hi: int) -> list[int]:
    return [1 << k for k in range(log2_lo, log2_hi + 1)]


def measure_ppermute_collective(
    *,
    log2_lo: int = 10,
    log2_hi: int = 22,
    dtype=jnp.float32,
    n_warmup: int = 5,
    n_timing: int = 100,
) -> CollectiveResult | None:
    """Time the ``jax.lax.ppermute`` PRIMITIVE inside a ``shard_map`` over
    a 2-device ``("face",)`` mesh.  Returns ``None`` when fewer than 2
    devices are visible (single-GPU / single CPU process) — ppermute
    requires a partner shard.

    Scope (codex MAJOR — do NOT over-claim): this measures the *ppermute
    primitive's* latency + bandwidth for a clean single-round 2-device
    swap.  It is the building block of the cubed-sphere halo
    (``cubesphere_exchange._make_exchange_ppermute`` uses ``lax.ppermute``
    on the same ``("face",)`` mesh), but it is NOT the full halo-exchange
    cost: the real path adds a multi-round colour schedule, strip
    gather/scatter, edge reversal, optional 3-point interpolation, corner
    fill, and multi-face-per-shard handling.  Use this number as the
    per-round collective FLOOR the halo cannot beat, not as the halo cost.
    To benchmark the full exchange, call the
    ``cubesphere_exchange`` factory on representative halo shapes (future
    extension).

    Each device sends its whole local payload to the other and receives
    the partner's.  ``_time_callable(mode="latency")`` blocks on the
    returned (sharded) array EVERY call, so the measured time is the
    collective round-trip — not a half-hidden async send.  The output
    DEPENDS on the ppermute result, so XLA cannot DCE the collective.
    """
    from jax import shard_map
    from jax.sharding import Mesh, PartitionSpec as P

    devs = jax.devices()
    if len(devs) < 2:
        return None
    mesh_devs = np.asarray(devs[:2])
    mesh = Mesh(mesh_devs, axis_names=("face",))
    perm = [(0, 1), (1, 0)]  # swap: each device's send goes to the other.

    res = CollectiveResult(
        transport="ppermute_spmd",
        n_participants=2,
        dtype=str(jnp.dtype(dtype)),
    )
    itemsize = jnp.dtype(dtype).itemsize

    for n in _payload_sweep(log2_lo, log2_hi):
        # Global array shape (2, n): axis 0 sharded over the 2-device face
        # mesh so each device owns a (1, n) shard → a length-n payload.
        x_global = jnp.arange(2 * n, dtype=dtype).reshape(2, n)
        x_global = jax.device_put(
            x_global, jax.sharding.NamedSharding(mesh, P("face")))
        jax.block_until_ready(x_global)

        @jax.jit
        def _run(xg):
            @partial_shard_map(shard_map, mesh, P("face"), P("face"))
            def _ppm(local):
                # local: (1, n) on this device → swap with the partner.
                got = jax.lax.ppermute(local, "face", perm)
                # Touch the result so it cannot be DCE'd; return it.
                return got
            return _ppm(xg)

        # Latency mode: block every call so the per-call time is the true
        # collective round-trip, not an async-queue throughput number
        # (codex BLOCKER).  block_until_ready waits on the sharded result
        # whose value DEPENDS on the ppermute, so the collective cannot be
        # hidden/overlapped past the clock.
        ms, _ = _time_callable(
            lambda: _run(x_global), n_warmup=n_warmup, n_timing=n_timing,
            mode="latency",
        )
        payload_bytes = n * itemsize
        gbps = payload_bytes / (ms * 1e-3) / 1e9
        res.points.append(CollectivePoint(
            n_elements=n,
            payload_bytes=payload_bytes,
            dtype=str(jnp.dtype(dtype)),
            time_per_call_ms=ms,
            achieved_gbps=gbps,
        ))

    if res.points:
        res.latency_floor_ms = res.points[0].time_per_call_ms
        res.asymptotic_gbps = res.points[-1].achieved_gbps
    return res


def partial_shard_map(shard_map, mesh, in_spec, out_spec):
    """Tiny adapter so ``measure_ppermute_collective`` reads cleanly:
    returns a decorator binding ``shard_map`` with our specs and
    ``check_vma=False`` (matching the model's ppermute kernel, which also
    sets it — the manual collective makes the per-shard VMA check
    irrelevant)."""
    from functools import partial
    return partial(shard_map, mesh=mesh, in_specs=in_spec, out_specs=out_spec,
                   check_vma=False)


def measure_sendrecv_collective(
    *,
    log2_lo: int = 10,
    log2_hi: int = 22,
    dtype=jnp.float32,
    n_warmup: int = 5,
    n_timing: int = 100,
) -> CollectiveResult | None:
    """Time the model's AD-safe ``mpi4jax`` sendrecv
    (``halo_exchange.get_sendrecv_vjp``) over a payload sweep, under
    ``mpirun``.  Returns ``None`` when not run under MPI with >=2 ranks.

    Rank r exchanges with its ring partner (r±1).  We time a jit'd call
    that sends the local buffer and receives the partner's; block on the
    received buffer (which DEPENDS on the collective completing) EVERY
    iteration so the clock captures the full per-call sendrecv, not a
    posted-but-unfinished request.  An MPI barrier brackets the timed
    region (so a fast rank cannot start its clock during a slow rank's
    warmup), and the recorded time is the MAX local-elapsed across ranks
    (``comm.allreduce(..., MPI.MAX)``) — the true collective wall time,
    not rank 0's local view.  Only rank 0 WRITES the result file.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    if nproc < 2:
        return None
    rank = comm.Get_rank()
    src = (rank - 1) % nproc
    dst = (rank + 1) % nproc

    import mpi4jax
    from legoesm.parallel.halo_exchange import get_sendrecv_vjp
    sendrecv = get_sendrecv_vjp(mpi4jax)

    res = CollectiveResult(
        transport="mpi4jax_sendrecv",
        n_participants=nproc,
        dtype=str(jnp.dtype(dtype)),
    )
    itemsize = jnp.dtype(dtype).itemsize

    @jax.jit
    def _run(send_buf, recv_template):
        return sendrecv(send_buf, recv_template, src, dst, 0, 0, comm)

    for n in _payload_sweep(log2_lo, log2_hi):
        send_buf = (jnp.arange(n, dtype=dtype) + rank)
        recv_template = jnp.zeros((n,), dtype=dtype)
        jax.block_until_ready((send_buf, recv_template))

        # Warmup (compile + prime), blocked + barriered.
        for _ in range(n_warmup):
            r = _run(send_buf, recv_template)
        jax.block_until_ready(r)
        comm.Barrier()

        # Latency mode: block on the received buffer EVERY iteration
        # (codex BLOCKER) so each sendrecv's full round-trip is on the
        # clock — a single trailing block would let mpi4jax overlap the
        # posted requests and understate the per-call cost.  The barrier
        # before t0 lines up every rank's start.
        comm.Barrier()
        t0 = time.perf_counter()
        for _ in range(n_timing):
            r = _run(send_buf, recv_template)
            jax.block_until_ready(r)
        t1 = time.perf_counter()
        local_elapsed = t1 - t0
        # MAJOR (codex): rank-0's local elapsed is not the collective wall
        # time — a fast rank can stop its clock while a slow rank is still
        # exchanging.  Report the MAX local elapsed across ranks (the true
        # wall time); this allreduce is a host-side MPI op OUTSIDE the
        # timed region.
        max_elapsed = comm.allreduce(local_elapsed, op=MPI.MAX)
        ms = max_elapsed / n_timing * 1e3

        payload_bytes = n * itemsize
        gbps = payload_bytes / (ms * 1e-3) / 1e9
        res.points.append(CollectivePoint(
            n_elements=n,
            payload_bytes=payload_bytes,
            dtype=str(jnp.dtype(dtype)),
            time_per_call_ms=ms,
            achieved_gbps=gbps,
        ))

    if res.points:
        res.latency_floor_ms = res.points[0].time_per_call_ms
        res.asymptotic_gbps = res.points[-1].achieved_gbps
    return res


# ===========================================================================
# Ceiling 3: allreduce latency by scalar-batch size
# ===========================================================================
@dataclass
class AllreducePoint:
    batch_size: int
    time_per_call_ms: float


@dataclass
class AllreduceResult:
    transport: str           # "psum_spmd" | "mpi4jax_batch".
    n_participants: int
    points: list[AllreducePoint] = field(default_factory=list)


def measure_psum_batch_sweep(
    *,
    batch_sizes: list[int],
    n_warmup: int = 5,
    n_timing: int = 200,
    dtype=jnp.float32,
) -> AllreduceResult | None:
    """``jax.lax.psum`` over a 2-device ``("r",)`` mesh, for scalar
    batches {1,2,4,...}.  Returns ``None`` on <2 devices.  This is the
    SPMD counterpart of the MPI weak-scaling Amdahl term: each PCG
    iteration pays ~2 of these latencies regardless of payload, so we
    measure how that latency grows (or doesn't) with the number of
    scalars reduced together."""
    from jax import shard_map
    from jax.sharding import Mesh, PartitionSpec as P

    devs = jax.devices()
    if len(devs) < 2:
        return None
    mesh = Mesh(np.asarray(devs[:2]), axis_names=("r",))
    res = AllreduceResult(transport="psum_spmd", n_participants=2)

    for b in batch_sizes:
        x_global = jnp.ones((2, b), dtype=dtype)
        x_global = jax.device_put(
            x_global, jax.sharding.NamedSharding(mesh, P("r")))
        jax.block_until_ready(x_global)

        @jax.jit
        def _run(xg):
            @partial_shard_map(shard_map, mesh, P("r"), P(None))
            def _ar(local):
                # local: (1, b); reduce the scalar-batch across devices.
                return jax.lax.psum(local[0], "r")  # (b,)
            return _ar(xg)

        # Latency mode (block each call): the Amdahl term is the per-call
        # allreduce LATENCY, so we must not let XLA pipeline the repeats
        # (codex BLOCKER).
        ms, _ = _time_callable(
            lambda: _run(x_global), n_warmup=n_warmup, n_timing=n_timing,
            mode="latency",
        )
        res.points.append(AllreducePoint(batch_size=b, time_per_call_ms=ms))
    return res


def measure_mpi_allreduce_batch_sweep(
    *,
    batch_sizes: list[int],
    n_warmup: int = 5,
    n_timing: int = 200,
    dtype=jnp.float32,
) -> AllreduceResult | None:
    """Model ``reductions.batch_allreduce_mpi`` for scalar batches
    {1,2,4,...} under ``mpirun``.  Returns ``None`` outside MPI/>=2 ranks.
    Times the SAME helper the barotropic solver uses, so the measured
    per-call latency is directly the PCG Amdahl term."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        return None
    from legoesm.parallel.reductions import batch_allreduce_mpi

    res = AllreduceResult(transport="mpi4jax_batch",
                          n_participants=comm.Get_size())

    for b in batch_sizes:
        vals = [jnp.asarray(float(i + 1), dtype=dtype) for i in range(b)]

        @jax.jit
        def _run(vs):
            return batch_allreduce_mpi(list(vs), op="sum")

        for _ in range(n_warmup):
            r = _run(vals)
        jax.block_until_ready(r)
        comm.Barrier()
        # Latency mode + MAX-across-ranks wall time — same fix as the
        # sendrecv path (codex round-2 NEW-BUG): the Amdahl term is the
        # per-call allreduce LATENCY, so block every iteration (no
        # pipelining) and report the slow-rank max, not rank-0 local.
        comm.Barrier()
        t0 = time.perf_counter()
        for _ in range(n_timing):
            r = _run(vals)
            jax.block_until_ready(r)
        t1 = time.perf_counter()
        local_elapsed = t1 - t0
        max_elapsed = comm.allreduce(local_elapsed, op=MPI.MAX)
        ms = max_elapsed / n_timing * 1e3
        res.points.append(AllreducePoint(batch_size=b, time_per_call_ms=ms))
    return res


# ===========================================================================
# Ceiling 4: kernel-launch / scan-dispatch floor
# ===========================================================================
@dataclass
class DispatchResult:
    bare_jit_ms_per_call: float
    scan_ms_per_step: float
    scan_length: int
    note: str = (
        "Lower bound on ms/step: a near-zero-work identity. bare_jit is "
        "per dispatched call (host launch overhead); scan_ms_per_step is "
        "the amortized per-iteration cost inside lax.scan (the model's "
        "time-integration construct) — the dispatch floor no step can beat."
    )


def measure_dispatch_floor(
    *,
    n_warmup: int = 10,
    n_timing: int = 500,
    scan_length: int = 2000,
    dtype=jnp.float32,
) -> DispatchResult:
    """Time a near-zero-work jit (identity-ish ``x + 1`` on a length-8
    array) as a bare call and inside a ``lax.scan`` of ``scan_length``.

    bare_jit: dispatch + launch overhead per call.  scan: one compiled
    program runs the whole loop on-device, so dividing its wall time by
    ``scan_length`` gives the amortized per-iteration dispatch floor with
    NO per-step host round-trip — the relevant floor for the model's
    scanned time loop."""
    x = jnp.ones((8,), dtype=dtype)
    x = jax.device_put(x)
    jax.block_until_ready(x)

    @jax.jit
    def _tiny(z):
        return z + jnp.asarray(1.0, dtype=dtype)

    # Latency mode: per-call dispatch + launch overhead is exactly a
    # blocked round-trip (codex BLOCKER — a pipelined loop would hide the
    # launch latency we are trying to measure).
    bare_ms, _ = _time_callable(
        lambda: _tiny(x), n_warmup=n_warmup, n_timing=n_timing,
        mode="latency",
    )

    @jax.jit
    def _scan(z0):
        def body(carry, _):
            return _tiny(carry), None
        out, _ = jax.lax.scan(body, z0, None, length=scan_length)
        return out

    # Warmup the scan (compile is large for big scan_length; block on it).
    r = _scan(x)
    jax.block_until_ready(r)
    r = _scan(x)
    jax.block_until_ready(r)
    t0 = time.perf_counter()
    r = _scan(x)
    jax.block_until_ready(r)
    t1 = time.perf_counter()
    scan_ms_per_step = (t1 - t0) / scan_length * 1e3

    return DispatchResult(
        bare_jit_ms_per_call=bare_ms,
        scan_ms_per_step=scan_ms_per_step,
        scan_length=scan_length,
    )


# ===========================================================================
# Per-step roofline reporter  (callable on any TimingResult-like object)
# ===========================================================================
@dataclass
class StepRoofline:
    grid_type: str
    n_grid: int
    n_levels: int
    n_gpus: int
    time_per_step_ms: float
    # Throughput.
    total_cell_levels: int
    cell_levels_per_s: float
    cell_levels_per_s_per_device: float
    # Memory roofline (BRACKETED estimate — see BytesModel).
    bytes_model: dict
    achieved_gbps_central: float
    achieved_gbps_lower: float    # fewest assumed bytes (lower-bytes).
    achieved_gbps_upper: float    # most assumed bytes (upper-bytes).
    sustained_gbps: float | None
    pct_sustained_central: float | None
    pct_sustained_range: tuple[float | None, float | None]
    # Collective census + floor.
    hlo_collective_permute: int
    hlo_collective_permute_start: int
    hlo_collective_permute_done: int
    collective_time_floor_ms: float | None
    collective_floor_note: str
    # PCG Amdahl (None for non-PCG steps).
    pcg_amdahl: dict | None = None


def _cells_for_grid(grid_type: str, n_grid: int, n_levels: int) -> int:
    """Total prognostic cell-levels (matches run_levante_gpu_scaling's
    total_cells convention: cube = 6*n*n*nlev, latlon = n*2n*nlev)."""
    g = grid_type.lower()
    if g in ("cubed-sphere", "cubed_sphere", "cube", "cs"):
        return 6 * n_grid * n_grid * n_levels
    if g in ("latlon", "lat-lon", "lat_lon"):
        return n_grid * (2 * n_grid) * n_levels
    # icosahedral / generic: caller passes total cells in n_grid, nlev=1.
    return n_grid * n_levels


def report_step_roofline(
    *,
    grid_type: str,
    n_grid: int,
    n_levels: int,
    n_gpus: int,
    time_per_step_ms: float,
    bytes_model: BytesModel,
    sustained_gbps: float | None,
    hlo_collective_permute: int = -1,
    hlo_collective_permute_start: int = -1,
    hlo_collective_permute_done: int = -1,
    collective_latency_ms: float | None = None,
    allreduce_latency_ms: float | None = None,
    pcg_iterations: int | None = None,
    pcg_local_stencil_ms: float | None = None,
    pcg_halo_ms: float | None = None,
) -> StepRoofline:
    """Compute the per-step roofline for one measured step.

    ``bytes_model`` carries the documented bytes/cell-level ESTIMATE; the
    achieved-GB/s output is therefore a BRACKET (central + lo/hi).

    Collective census: pass the HLO counts straight from a
    ``run_levante_gpu_scaling.TimingResult`` (``hlo_collective_permute*``)
    — this reporter does NOT re-derive the census, it consumes the one the
    timed-executable HLO guard already produced.

    Collective time-floor: ``collective_latency_ms`` × (number of
    collective ops in the step).  We count each sync collective-permute
    plus each async start as one round (done ops are the completion half
    of a start and are not double-counted).

    PCG Amdahl: if ``pcg_iterations`` (M) is given, evaluate
    ``T = local_stencil + M*(2*allreduce_lat + halo + stencil)`` and report
    the nonlocal (communication) fraction."""
    total_cl = _cells_for_grid(grid_type, n_grid, n_levels)
    t_s = time_per_step_ms * 1e-3
    cl_per_s = total_cl / t_s if t_s > 0 else 0.0
    cl_per_s_dev = cl_per_s / max(n_gpus, 1)
    # Per-device cell-levels actually moved per step.
    cl_per_device = total_cl / max(n_gpus, 1)

    central_b = bytes_model.bytes_per_cell_level_central()
    lo_b, hi_b = bytes_model.bytes_per_cell_level_bracket()

    def _gbps(bytes_per_cl: float) -> float:
        moved = cl_per_device * bytes_per_cl
        return moved / t_s / 1e9 if t_s > 0 else 0.0

    g_central = _gbps(central_b)
    # achieved GB/s = (inferred bytes moved) / time, so MORE assumed bytes
    # → HIGHER inferred achieved bandwidth.  The bracket on bytes
    # [lo_b, hi_b] therefore maps to a bracket on achieved GB/s
    # [g_lower, g_upper] with the SAME ordering (lo_b → g_lower).
    g_lower = _gbps(lo_b)   # fewest bytes assumed → lowest achieved GB/s.
    g_upper = _gbps(hi_b)   # most bytes assumed  → highest achieved GB/s.

    def _pct(g: float) -> float | None:
        return 100.0 * g / sustained_gbps if sustained_gbps else None

    # Collective time-floor.
    floor_ms = None
    floor_note = "no collective-latency input provided"
    if collective_latency_ms is not None and hlo_collective_permute >= 0:
        n_rounds = max(hlo_collective_permute, 0) + max(
            hlo_collective_permute_start, 0)
        floor_ms = n_rounds * collective_latency_ms
        floor_note = (
            f"{n_rounds} collective round(s) "
            f"(sync={max(hlo_collective_permute,0)} + "
            f"async-start={max(hlo_collective_permute_start,0)}) × "
            f"{collective_latency_ms:.4f} ms latency floor"
        )

    pcg = None
    if pcg_iterations is not None:
        M = pcg_iterations
        ar = allreduce_latency_ms or 0.0
        halo = pcg_halo_ms or 0.0
        stencil = pcg_local_stencil_ms or 0.0
        per_iter_nonlocal = 2.0 * ar + halo
        per_iter_total = per_iter_nonlocal + stencil
        T = stencil + M * per_iter_total
        nonlocal_time = M * per_iter_nonlocal
        pcg = {
            "iterations_M": M,
            "allreduce_latency_ms": ar,
            "halo_ms": halo,
            "stencil_ms": stencil,
            "model": "T = local_stencil + M*(2*allreduce_lat + halo + stencil)",
            "T_predicted_ms": T,
            "nonlocal_time_ms": nonlocal_time,
            "nonlocal_fraction": (nonlocal_time / T) if T > 0 else None,
            "note": (
                "Amdahl weak-scaling term: the 2*M allreduce latencies are "
                "payload-independent, so nonlocal_fraction -> 1 as ranks grow "
                "(latency rises) or resolution/rank shrinks (stencil falls)."
            ),
        }

    return StepRoofline(
        grid_type=grid_type,
        n_grid=n_grid,
        n_levels=n_levels,
        n_gpus=n_gpus,
        time_per_step_ms=time_per_step_ms,
        total_cell_levels=total_cl,
        cell_levels_per_s=cl_per_s,
        cell_levels_per_s_per_device=cl_per_s_dev,
        bytes_model=bytes_model.describe(),
        achieved_gbps_central=g_central,
        achieved_gbps_lower=g_lower,
        achieved_gbps_upper=g_upper,
        sustained_gbps=sustained_gbps,
        pct_sustained_central=_pct(g_central),
        pct_sustained_range=(_pct(g_lower), _pct(g_upper)),
        hlo_collective_permute=hlo_collective_permute,
        hlo_collective_permute_start=hlo_collective_permute_start,
        hlo_collective_permute_done=hlo_collective_permute_done,
        collective_time_floor_ms=floor_ms,
        collective_floor_note=floor_note,
        pcg_amdahl=pcg,
    )


# ===========================================================================
# Plotting (matplotlib Agg, campaign conventions)
# ===========================================================================
def plot_roofline(payload: dict, out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping PNG")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_bw, ax_coll, ax_ar, ax_pcg = axes.ravel()

    # --- Panel 1: sustained BW vs array size, %-peak annotated -----------
    bw = payload.get("bandwidth")
    if bw and bw.get("points"):
        sizes = [p["working_set_bytes"] / 1e6 for p in bw["points"]]
        gbps = [p["achieved_gbps"] for p in bw["points"]]
        hbm = [p["hits_hbm"] for p in bw["points"]]
        ax_bw.plot(sizes, gbps, "-", color="#1f77b4", linewidth=1.5, zorder=1)
        ax_bw.scatter(
            [s for s, h in zip(sizes, hbm) if h],
            [g for g, h in zip(gbps, hbm) if h],
            color="#1f77b4", marker="o", s=45, label="DRAM-bound", zorder=2)
        ax_bw.scatter(
            [s for s, h in zip(sizes, hbm) if not h],
            [g for g, h in zip(gbps, hbm) if not h],
            facecolors="none", edgecolors="#1f77b4", marker="o", s=45,
            label="cache-resident", zorder=2)
        peak = bw.get("spec_peak_gbps")
        if peak:
            ax_bw.axhline(peak, color="#d62728", linestyle="--",
                          label=f"spec peak {peak:.0f} GB/s")
        sus = bw.get("sustained_gbps")
        if sus:
            pct = bw.get("sustained_pct_peak")
            lbl = f"sustained {sus:.0f} GB/s"
            if pct:
                lbl += f" ({pct:.0f}% peak)"
            ax_bw.axhline(sus, color="#2ca02c", linestyle=":", label=lbl)
        ax_bw.set_xscale("log", base=2)
        ax_bw.set_xlabel("working set (MB)")
        ax_bw.set_ylabel("achieved GB/s")
        ax_bw.set_title(f"Sustained bandwidth — {bw.get('device_kind','?')}")
        ax_bw.legend(fontsize=8)
        ax_bw.grid(True, alpha=0.3)
    else:
        ax_bw.set_title("Sustained bandwidth (no data)")

    # --- Panel 2: collective latency vs payload --------------------------
    plotted = False
    for coll in payload.get("collectives", []):
        if not coll.get("points"):
            continue
        pb = [p["payload_bytes"] for p in coll["points"]]
        ms = [p["time_per_call_ms"] for p in coll["points"]]
        ax_coll.plot(pb, ms, "o-", markersize=5, linewidth=1.5,
                     label=f"{coll['transport']} (np={coll['n_participants']})")
        plotted = True
    if plotted:
        ax_coll.set_xscale("log", base=2)
        ax_coll.set_yscale("log")
        ax_coll.set_xlabel("payload (bytes, one direction)")
        ax_coll.set_ylabel("time per call (ms)")
        ax_coll.set_title("Collective latency + BW by payload")
        ax_coll.legend(fontsize=8)
        ax_coll.grid(True, alpha=0.3, which="both")
    else:
        ax_coll.set_title("Collectives (no data — single device/rank)")

    # --- Panel 3: allreduce latency vs batch size ------------------------
    plotted = False
    for ar in payload.get("allreduce", []):
        if not ar.get("points"):
            continue
        bs = [p["batch_size"] for p in ar["points"]]
        ms = [p["time_per_call_ms"] for p in ar["points"]]
        ax_ar.plot(bs, ms, "s-", markersize=6, linewidth=1.5,
                   label=f"{ar['transport']} (np={ar['n_participants']})")
        plotted = True
    if plotted:
        ax_ar.set_xscale("log", base=2)
        ax_ar.set_xlabel("scalar batch size")
        ax_ar.set_ylabel("time per call (ms)")
        ax_ar.set_title("Allreduce latency vs batch (PCG Amdahl term)")
        ax_ar.legend(fontsize=8)
        ax_ar.grid(True, alpha=0.3, which="both")
    else:
        ax_ar.set_title("Allreduce (no data — single device/rank)")

    # --- Panel 4: PCG Amdahl breakdown -----------------------------------
    pcg = payload.get("pcg_amdahl_example")
    if pcg:
        labels = ["local\nstencil", "M·stencil", "M·halo", "M·2·allreduce"]
        M = pcg["iterations_M"]
        vals = [
            pcg["stencil_ms"],
            M * pcg["stencil_ms"],
            M * pcg["halo_ms"],
            M * 2 * pcg["allreduce_latency_ms"],
        ]
        colors = ["#2ca02c", "#2ca02c", "#ff7f0e", "#d62728"]
        ax_pcg.bar(labels, vals, color=colors)
        ax_pcg.set_ylabel("time (ms)")
        nf = pcg.get("nonlocal_fraction")
        nf_s = f"{nf:.0%}" if nf is not None else "n/a"
        ax_pcg.set_title(
            f"PCG Amdahl (M={M}): nonlocal fraction = {nf_s}")
        ax_pcg.grid(True, alpha=0.3, axis="y")
    else:
        ax_pcg.set_title("PCG Amdahl (no example provided)")

    fig.suptitle(
        f"Measured roofline — {payload.get('host','?')} "
        f"[{payload.get('jax_platform','?')}]",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_png}")


# ===========================================================================
# CLI / orchestration
# ===========================================================================
def _is_rank0() -> bool:
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        return True


def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measured roofline + collective microbenchmarks")
    ap.add_argument("--output-dir", type=str,
                    default="results/scaling_ginsburg/roofline")
    ap.add_argument("--tag", type=str, default=None,
                    help="filename tag; default = jax platform + nproc")
    # Bandwidth sweep (working set = 2x array; sizes span L2->HBM).
    ap.add_argument("--bw-sizes-mb", type=str,
                    default="0.25,0.5,1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--bw-dtype", choices=["float32", "float64"],
                    default="float32")
    ap.add_argument("--spec-peak-gbps", type=float, default=None,
                    help="override device spec-peak BW (GB/s)")
    ap.add_argument("--cpu-peak-gbps", type=float, default=None,
                    help="CPU DRAM spec-peak per rank (GB/s), reporting only")
    ap.add_argument("--cache-bytes", type=int, default=None,
                    help="LLC estimate for the hits_hbm flag (bytes)")
    # Collective + allreduce sweeps.
    ap.add_argument("--coll-log2-lo", type=int, default=10)
    ap.add_argument("--coll-log2-hi", type=int, default=22)
    ap.add_argument("--allreduce-batches", type=str, default="1,2,4,8,16")
    # Iteration counts.
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-timing", type=int, default=100)
    ap.add_argument("--scan-length", type=int, default=2000)
    # PCG Amdahl example (uses measured allreduce latency + a passed M;
    # halo/stencil default to small placeholders if not supplied — the
    # point is the structure + nonlocal fraction, clearly labelled).
    ap.add_argument("--pcg-iterations", type=int, default=20,
                    help="M for the PCG Amdahl example breakdown")
    ap.add_argument("--pcg-halo-ms", type=float, default=None)
    ap.add_argument("--pcg-stencil-ms", type=float, default=None)
    # Bytes model overrides (the documented estimate).
    ap.add_argument("--n-state-fields", type=int, default=5,
                    help="prognostic volumetric field count (dry NH = 5)")
    ap.add_argument("--stencil-reuse-factor", type=float, default=1.0,
                    help="bytes-model stencil-reuse factor (DEFAULT 1.0 is a "
                         "scenario, NOT measured; pass a profiler-derived "
                         "value to replace the central assumption)")
    ap.add_argument("--skip-bandwidth", action="store_true")
    ap.add_argument("--skip-collectives", action="store_true")
    ap.add_argument("--skip-allreduce", action="store_true")
    ap.add_argument("--skip-dispatch", action="store_true")
    args = ap.parse_args()

    # Validate the bytes-model factor (codex round-2 NEW-BUG): a negative
    # or non-finite stencil-reuse factor would yield negative/inverted
    # byte and GB/s estimates.
    import math
    if not math.isfinite(args.stencil_reuse_factor) \
            or args.stencil_reuse_factor < 0:
        ap.error("--stencil-reuse-factor must be finite and >= 0 "
                 f"(got {args.stencil_reuse_factor})")
    if args.n_state_fields < 1:
        ap.error("--n-state-fields must be >= 1 "
                 f"(got {args.n_state_fields})")

    bw_dtype = jnp.float64 if args.bw_dtype == "float64" else jnp.float32
    if bw_dtype == jnp.float64:
        jax.config.update("jax_enable_x64", True)

    rank0 = _is_rank0()
    devs = jax.devices()
    platform_name = str(getattr(devs[0], "platform", "cpu"))
    nproc = 1
    try:
        from mpi4py import MPI
        nproc = MPI.COMM_WORLD.Get_size()
    except ImportError:
        pass

    tag = args.tag or f"{platform_name}_np{nproc}"
    host = socket.gethostname()

    if rank0:
        print("=" * 70)
        print(f"  Roofline probe — host={host} platform={platform_name} "
              f"nproc={nproc} ndev={len(devs)}")
        print(f"  device[0]={getattr(devs[0],'device_kind',platform_name)}")
        print("=" * 70)

    payload: dict = {
        "host": host,
        "jax_platform": platform_name,
        "jax_version": jax.__version__,
        "n_processes": nproc,
        "n_local_devices": len(devs),
        "python": platform.python_version(),
        "args": vars(args),
    }

    # --- Ceiling 1: bandwidth (every rank measures its own device) -------
    bw_res = None
    if not args.skip_bandwidth:
        bw_res = measure_sustained_bandwidth(
            sizes_mb=_parse_float_list(args.bw_sizes_mb),
            dtype=bw_dtype,
            n_warmup=args.n_warmup,
            n_timing=args.n_timing,
            spec_peak_gbps=args.spec_peak_gbps,
            cache_bytes=args.cache_bytes,
        )
        if platform_name == "cpu" and bw_res.spec_peak_gbps is None \
                and args.cpu_peak_gbps:
            bw_res.spec_peak_gbps = args.cpu_peak_gbps
            if bw_res.sustained_gbps:
                bw_res.sustained_pct_peak = (
                    100.0 * bw_res.sustained_gbps / args.cpu_peak_gbps)
        if rank0:
            print(f"\n[1] Sustained BW: {bw_res.sustained_gbps:.1f} GB/s"
                  + (f" ({bw_res.sustained_pct_peak:.0f}% of "
                     f"{bw_res.spec_peak_gbps:.0f} peak)"
                     if bw_res.sustained_pct_peak else " (peak unknown)"))
        payload["bandwidth"] = asdict(bw_res)

    # --- Ceiling 2: collectives ------------------------------------------
    collectives: list[dict] = []
    if not args.skip_collectives:
        ppm = measure_ppermute_collective(
            log2_lo=args.coll_log2_lo, log2_hi=args.coll_log2_hi,
            dtype=bw_dtype, n_warmup=args.n_warmup, n_timing=args.n_timing)
        if ppm is not None:
            collectives.append(asdict(ppm))
            if rank0:
                print(f"[2a] ppermute SPMD: latency floor "
                      f"{ppm.latency_floor_ms*1e3:.1f} us, "
                      f"asymptotic {ppm.asymptotic_gbps:.1f} GB/s")
        srv = measure_sendrecv_collective(
            log2_lo=args.coll_log2_lo, log2_hi=args.coll_log2_hi,
            dtype=bw_dtype, n_warmup=args.n_warmup, n_timing=args.n_timing)
        if srv is not None:
            collectives.append(asdict(srv))
            if rank0:
                print(f"[2b] mpi4jax sendrecv: latency floor "
                      f"{srv.latency_floor_ms*1e3:.1f} us, "
                      f"asymptotic {srv.asymptotic_gbps:.1f} GB/s")
    payload["collectives"] = collectives

    # --- Ceiling 3: allreduce --------------------------------------------
    allreduce: list[dict] = []
    measured_ar_latency_ms = None
    if not args.skip_allreduce:
        batches = _parse_int_list(args.allreduce_batches)
        psum = measure_psum_batch_sweep(
            batch_sizes=batches, n_warmup=args.n_warmup,
            n_timing=max(args.n_timing, 200), dtype=bw_dtype)
        if psum is not None:
            allreduce.append(asdict(psum))
            measured_ar_latency_ms = psum.points[0].time_per_call_ms
            if rank0:
                print(f"[3a] psum batch{{1}}: "
                      f"{psum.points[0].time_per_call_ms*1e3:.1f} us/call")
        mpiar = measure_mpi_allreduce_batch_sweep(
            batch_sizes=batches, n_warmup=args.n_warmup,
            n_timing=max(args.n_timing, 200), dtype=bw_dtype)
        if mpiar is not None:
            allreduce.append(asdict(mpiar))
            measured_ar_latency_ms = mpiar.points[0].time_per_call_ms
            if rank0:
                print(f"[3b] mpi batch_allreduce{{1}}: "
                      f"{mpiar.points[0].time_per_call_ms*1e3:.1f} us/call")
    payload["allreduce"] = allreduce

    # --- Ceiling 4: dispatch floor ---------------------------------------
    if not args.skip_dispatch:
        disp = measure_dispatch_floor(
            n_warmup=max(args.n_warmup, 10),
            n_timing=max(args.n_timing, 500),
            scan_length=args.scan_length, dtype=bw_dtype)
        payload["dispatch"] = asdict(disp)
        if rank0:
            print(f"[4] Dispatch floor: bare jit "
                  f"{disp.bare_jit_ms_per_call*1e3:.1f} us/call, "
                  f"scan {disp.scan_ms_per_step*1e3:.2f} us/step")

    # --- PCG Amdahl worked example (uses measured allreduce latency) -----
    #   Demonstrates report_step_roofline's PCG branch with a real
    #   allreduce-latency input. halo/stencil are placeholders unless the
    #   user supplies them (CLEARLY labelled in the JSON note).
    if measured_ar_latency_ms is not None:
        bm = BytesModel(n_state_fields=args.n_state_fields,
                        dtype_bytes=jnp.dtype(bw_dtype).itemsize,
                        stencil_reuse_factor=args.stencil_reuse_factor)
        example = report_step_roofline(
            grid_type="latlon", n_grid=128, n_levels=1, n_gpus=nproc,
            time_per_step_ms=1.0,  # placeholder; PCG block is the point.
            bytes_model=bm, sustained_gbps=(bw_res.sustained_gbps
                                            if bw_res else None),
            allreduce_latency_ms=measured_ar_latency_ms,
            pcg_iterations=args.pcg_iterations,
            pcg_halo_ms=(args.pcg_halo_ms
                         if args.pcg_halo_ms is not None
                         else measured_ar_latency_ms),  # halo ~ 1 sendrecv
            pcg_local_stencil_ms=(args.pcg_stencil_ms
                                  if args.pcg_stencil_ms is not None
                                  else 5.0 * measured_ar_latency_ms),
        )
        payload["pcg_amdahl_example"] = example.pcg_amdahl
        payload["pcg_amdahl_example"]["inputs_note"] = (
            "allreduce_latency MEASURED (ceiling 3); halo_ms defaulted to "
            "1 allreduce latency and stencil_ms to 5x (PLACEHOLDERS unless "
            "--pcg-halo-ms/--pcg-stencil-ms passed). The nonlocal fraction "
            "is exact for the supplied terms; supply measured halo/stencil "
            "from a real ocean step for a production number."
        )
        if rank0 and example.pcg_amdahl:
            nf = example.pcg_amdahl["nonlocal_fraction"]
            print(f"[PCG] Amdahl example M={args.pcg_iterations}: "
                  f"nonlocal fraction = "
                  f"{nf:.1%}" if nf is not None else "n/a")

    payload["bytes_model_default"] = BytesModel(
        n_state_fields=args.n_state_fields,
        dtype_bytes=jnp.dtype(bw_dtype).itemsize,
        stencil_reuse_factor=args.stencil_reuse_factor).describe()

    # --- Write outputs (rank 0 only) -------------------------------------
    if rank0:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"roofline_{host}_{tag}.json"
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\n  wrote {out_json}")
        plot_roofline(payload, out_dir / f"roofline_{host}_{tag}.png")
        print("=" * 70)
        print("  DONE")
        print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
