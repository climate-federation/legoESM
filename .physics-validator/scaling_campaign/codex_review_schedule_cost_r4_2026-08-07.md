Reading additional input from stdin...
OpenAI Codex v0.146.1
--------
workdir: /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
model: gpt-5.6-terra
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019fdbe4-8804-7261-b937-687e11c6e69e
--------
user
ADVERSARIAL REVIEW, ROUND 4. Rounds 1-3 found defects; round 3 said NOT READY on 4 items. Verify the round-3 fixes and hunt for new defects. Refute, do not summarize. State plainly SHIP or NOT READY at the end.

Files: `git diff` on scripts/bench/bench_voronoi_partition_methods.py and tests/bench/test_bench_voronoi_partition_methods.py; untracked scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch and tests/bench/test_mpas_schedule_cost_scan_sbatch.py.

ROUND-3 FINDINGS AND CLAIMED FIXES:
1. HIGH 'whitespace-only --expect-rounds bypasses the gate' -> parse_expect_rounds now guards on `if spec and not out` (the raw string), not `spec.strip()`, so '   ' raises while the default '' still means no gate. Test now covers '   ', '\t', ' , , '. VERIFY, and find any remaining bypass.
2. HIGH 'negative --subdivision and --halo-depth are false provenance' -> main() now raises SystemExit for negative lloyd, subdivision AND halo_depth, with a parametrized test. VERIFY all three, and check whether --rank-counts or any other input has a remaining hole of the same class.
3. MEDIUM 'arm-3 exit propagation fixed but unprotected; a shell-level regression test is required before ship' -> I made BENCH overridable via LEGOESM_SCHEDULE_COST_BENCH and added tests/bench/test_mpas_schedule_cost_scan_sbatch.py, which drives the real sbatch under bash with a stub bench and asserts three paths: validation-fail exits non-zero AND does not run arm 3; arm-3 fail propagates its exact status and prints SCAN_FAILED_ARM3 without SCAN_DONE; all-success exits 0 having run arms 8,9,10. VERIFY these tests really exercise the script (not a copy), that the stub's per-call exit-code sequencing is sound, and that they would FAIL if the exit logic regressed. Is making BENCH overridable a hazard in production (could a stray env var silently redirect a real scan)?
4. MEDIUM 'unavailable-metis assertion too weak' -> test now also asserts the explicit {'method':'metis','available':False} row. VERIFY.

ALSO:
- Re-check anything you previously called correct, in case a round-3 edit broke it.
- The scan is RUNNING and its first production row reproduced the reference census exactly (s8 geometric@64: rounds=16, max_degree=16, gap=0, matching the docstring's 'geometric 16/21'). Does that constitute a real instrument validation, or is there a way it could match for the wrong reason?
- Any NEW vacuity in the 6 tests added since round 2.

Cite file:line.
codex
I’ll inspect the actual diff and untracked scan/test files, then trace each claimed fix back to executable behavior and look for ways the new tests could pass vacuously.
exec
/usr/bin/bash -lc 'rg -n "expect-rounds|rank-counts|subdivision|halo-depth|metis|mpas_schedule_cost_scan|SCAN_(DONE|FAILED)|LEGOESM_SCHEDULE_COST_BENCH" scripts tests docs .github 2>/dev/null' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc 'nl -ba scripts/bench/bench_voronoi_partition_methods.py && nl -ba tests/bench/test_bench_voronoi_partition_methods.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc 'git diff -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py && git status --short && nl -ba scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && nl -ba tests/bench/test_mpas_schedule_cost_scan_sbatch.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 160ms:
     1	"""Benchmark MPAS/Voronoi partition methods: RCB vs Hilbert-SFC vs METIS.
     2	
     3	Scaling-audit item 8.  Two independent measurement layers:
     4	
     5	1. **Offline partition quality** (no MPI, exact, every rank enumerated
     6	   serially): edge cut, halo cells (max/mean, halo/owned ratio), load
     7	   balance (cells/rank min/max, imbalance max/mean), neighbor-rank fan-out —
     8	   for each method x rank-count on the real icosahedral mesh.  These are
     9	   the numbers ``resolve_partition_method``'s ``auto`` policy must be
    10	   justified by.
    11	1b. **SPMD halo-schedule depth** (``--schedule-cost``, opt-in because it is
    12	   the expensive layer): ``n_rounds`` — the number of SEQUENTIAL ppermute
    13	   rounds one halo fill costs — and ``max_degree``, the communication
    14	   graph's lower bound on it.  This is the term that binds MPAS GPU strong
    15	   scaling above ~64 devices, and layer 1 CANNOT stand in for it: the
    16	   neighbor fan-out above is a 1-ring proxy that reported 8 rounds for
    17	   every method and rank count while the real depth-3-plus-closure
    18	   schedule reported 12-14.  Scored by the production
    19	   ``spmd_schedule_cost`` (which calls the production builders), never a
    20	   re-derived lookalike.
    21	
    22	   The DECISIVE column is ``coloring_gap = n_rounds - max_degree``, read
    23	   through VIZING'S THEOREM, which bounds what recolouring could ever buy.
    24	   The schedule is a proper EDGE colouring of the device communication
    25	   graph (one colour = one ppermute round; ``_build_ppermute_schedule``
    26	   asserts properness), and ``max_degree`` is that same graph's maximum
    27	   vertex degree.  So the chromatic index obeys ``Delta <= chi' <=
    28	   Delta + 1``: the gap is a bound on recolouring headroom, NOT a
    29	   yes/no flag.
    30	
    31	   * ``gap == 0`` -> ``n_rounds == Delta``, and no proper edge colouring
    32	     can beat ``Delta``.  The colouring is PROVABLY OPTIMAL; recolouring
    33	     headroom is exactly ZERO.
    34	   * ``gap == 1`` -> INCONCLUSIVE.  A Class 2 graph genuinely needs
    35	     ``Delta + 1``, and deciding Class 1 vs Class 2 is NP-complete, so
    36	     this neither establishes nor excludes a one-round win.
    37	   * ``gap >= 2`` -> recolouring is guaranteed to remove AT LEAST
    38	     ``gap - 1`` rounds (the optimum is at worst ``Delta + 1``) and at
    39	     most ``gap``.
    40	
    41	   SCOPE, and it is not a formality: ``Delta`` bounds only a proper
    42	   UNDIRECTED edge colouring of THIS graph.  ``_build_ppermute_schedule``
    43	   enters a device pair into ``comm_pairs`` when EITHER direction has a
    44	   halo dependency and then emits BOTH ppermute directions, even where one
    45	   send map is empty.  A redesigned DIRECTED schedule that exploits
    46	   one-way exchanges is therefore not bounded by ``Delta`` at all, so
    47	   ``gap == 0`` must never be reported as "only ownership can help" — it
    48	   rules out a better undirected edge colouring of this graph, and
    49	   nothing more.
    50	
    51	   NOTE the halo depth differs by lane and is NOT ``--halo-depth``: that
    52	   flag is the MPI lane's (default 2), while the schedule is scored at the
    53	   SPMD production depth ``SPMD_HALO_DEPTH`` (3).  Both are recorded.
    54	   ``n_rounds`` is per HALO FILL, not per step — multiply by the tendency
    55	   evaluations of the integrator actually run.  When
    56	   ``production_strategy`` is ``"allgather"`` (auto-selected below the
    57	   cells/device threshold) there is no ppermute schedule in production and
    58	   the round count is COUNTERFACTUAL; the row says so.
    59	2. **Step time** (optional pointer, NOT run here): drive the existing
    60	   MPI lane with ``bench_ocean_mpas_scaling.py --partition-method <m>``
    61	   (ocean) or ``bench_mpas_spmd_scaling.py --partition-method <m>``
    62	   (atmosphere SPMD) — one method per launch, same case otherwise
    63	   (controlled comparison).
    64	
    65	Guards: methods that are unavailable (``metis`` without ``pymetis``) are
    66	reported as ``"unavailable"`` — never silently substituted, so a table
    67	column can never claim METIS numbers that actually came from the RCB
    68	fallback.  Partition CORRECTNESS is asserted per row (every cell owned by
    69	exactly one rank; owner range valid) before any metric is recorded.
    70	
    71	Run:
    72	  python scripts/bench/bench_voronoi_partition_methods.py \
    73	      --subdivision 6 --rank-counts 2,4,8,16 --out results/partition_quality.json
    74	
    75	  # + the SPMD schedule depth (minutes to hours at subdiv>=8 — batch it):
    76	  python scripts/bench/bench_voronoi_partition_methods.py \
    77	      --subdivision 9 --rank-counts 64,128 --schedule-cost \
    78	      --out results/a1/schedule_cost_s9.json
    79	"""
    80	from __future__ import annotations
    81	
    82	import argparse
    83	import json
    84	import os
    85	import sys
    86	from pathlib import Path
    87	
    88	import numpy as np
    89	
    90	sys.path.insert(0, str(Path(__file__).resolve().parent))
    91	
    92	from metadata import annotate_incomplete, scaling_metadata  # noqa: E402
    93	
    94	METHODS = ("geometric", "sfc", "metis")
    95	
    96	
    97	def method_available(method: str) -> bool:
    98	    if method != "metis":
    99	        return True
   100	    try:
   101	        import pymetis  # noqa: F401
   102	
   103	        return True
   104	    except Exception:
   105	        return False
   106	
   107	
   108	def partition_quality(mesh, cell_owner: np.ndarray, n_ranks: int,
   109	                      halo_depth: int = 2) -> dict:
   110	    """Exact partition-quality metrics from a global owner array.
   111	
   112	    Serial enumeration of every rank (no MPI): the same halo construction
   113	    the runtime uses (``compute_halo_cells``), so the reported halo sizes
   114	    are the runtime's, not an estimate.
   115	    """
   116	    from legoesm.parallel.voronoi_partition import compute_halo_cells
   117	
   118	    n_cells = int(mesh.nCells)
   119	    # Correctness: every cell owned exactly once, owners in range, no
   120	    # empty rank (an empty rank would silently deflate the halo/owned
   121	    # ratio through the max(counts, 1) guard).
   122	    if n_cells == 0 or cell_owner.size == 0:
   123	        raise AssertionError("empty mesh / owner array")
   124	    if cell_owner.shape != (n_cells,):
   125	        raise AssertionError(f"owner shape {cell_owner.shape} != ({n_cells},)")
   126	    if cell_owner.min() < 0 or cell_owner.max() >= n_ranks:
   127	        raise AssertionError("owner out of range")
   128	    counts = np.bincount(cell_owner, minlength=n_ranks).astype(float)
   129	    if int(counts.sum()) != n_cells:
   130	        raise AssertionError("ownership does not cover the mesh")
   131	    if counts.min() <= 0:
   132	        raise AssertionError(
   133	            f"empty rank in partition (counts.min()={counts.min():.0f}) — "
   134	            f"a skipped rank corrupts every per-rank metric")
   135	
   136	    # Edge cut: edges whose two cells have different owners.
   137	    c1, c2 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
   138	    valid = (c1 >= 0) & (c2 >= 0)
   139	    edge_cut = int((cell_owner[c1[valid]] != cell_owner[c2[valid]]).sum())
   140	
   141	    halo_sizes = []
   142	    neighbor_counts = []
   143	    cells_on_cell = np.asarray(mesh.cellsOnCell)
   144	    max_edges = int(cells_on_cell.shape[0]) if cells_on_cell.ndim == 2 else 0
   145	    for r in range(n_ranks):
   146	        halo = compute_halo_cells(
   147	            cell_owner, mesh.cellsOnCell, mesh.maxEdges, r, halo_depth)
   148	        halo_sizes.append(len(halo))
   149	        neighbor_counts.append(
   150	            len(set(int(cell_owner[c]) for c in halo) - {r}))
   151	    _ = max_edges
   152	    halo_sizes = np.array(halo_sizes, dtype=float)
   153	    return {
   154	        "cells_per_rank_min": int(counts.min()),
   155	        "cells_per_rank_max": int(counts.max()),
   156	        "load_imbalance_max_over_mean": float(counts.max() / counts.mean()),
   157	        "edge_cut": edge_cut,
   158	        "edge_cut_fraction": float(edge_cut / max(int(valid.sum()), 1)),
   159	        "halo_cells_max": int(halo_sizes.max()),
   160	        "halo_cells_mean": float(halo_sizes.mean()),
   161	        "halo_owned_ratio_max": float(
   162	            (halo_sizes / np.maximum(counts, 1.0)).max()),
   163	        "neighbor_ranks_max": int(max(neighbor_counts)),
   164	    }
   165	
   166	
   167	def owner_for(mesh, method: str, n_ranks: int) -> np.ndarray:
   168	    from legoesm.parallel.voronoi_partition import (
   169	        partition_cells_geometric,
   170	        partition_cells_metis,
   171	        partition_cells_sfc,
   172	    )
   173	
   174	    if method == "geometric":
   175	        return np.asarray(partition_cells_geometric(mesh, n_ranks))
   176	    if method == "sfc":
   177	        return np.asarray(partition_cells_sfc(mesh, n_ranks))
   178	    if method == "metis":
   179	        return np.asarray(partition_cells_metis(mesh, n_ranks))
   180	    raise ValueError(f"unknown partition method {method!r}; "
   181	                     f"expected one of {METHODS}")
   182	
   183	
   184	def parse_expect_rounds(spec: str) -> dict:
   185	    """Parse ``'sfc:64=12,metis:128=19'`` into ``{("sfc", 64): 12}``.
   186	
   187	    Raises on anything malformed rather than skipping it — a typo'd
   188	    expectation that is silently dropped turns the gate into a no-op, which
   189	    is exactly the failure this flag exists to prevent.  Three ways that
   190	    could happen, all rejected here:
   191	
   192	    * a malformed item (``geometric:2``, ``geometric=2``);
   193	    * an unknown method;
   194	    * a DUPLICATE key — ``geometric:2=999,geometric:2=13`` would otherwise
   195	      let the second silently overwrite the first and pass;
   196	    * a non-empty spec that parses to NOTHING (``",,,"``), which would make
   197	      the caller skip the check while believing it ran.
   198	    """
   199	    out: dict[tuple[str, int], int] = {}
   200	    for item in (s.strip() for s in spec.split(",")):
   201	        if not item:
   202	            continue
   203	        try:
   204	            lhs, rounds = item.split("=")
   205	            method, n_ranks = lhs.split(":")
   206	            key = (method.strip(), int(n_ranks))
   207	            value = int(rounds)
   208	        except ValueError as exc:
   209	            raise ValueError(
   210	                f"--expect-rounds: cannot parse {item!r}; expected "
   211	                f"'<method>:<n_ranks>=<n_rounds>'") from exc
   212	        if key[0] not in METHODS:
   213	            raise ValueError(
   214	                f"--expect-rounds: unknown method {key[0]!r} in {item!r}; "
   215	                f"expected one of {METHODS}")
   216	        if key in out:
   217	            raise ValueError(
   218	                f"--expect-rounds: duplicate expectation for "
   219	                f"{key[0]}:{key[1]} ({out[key]} then {value}); the later one "
   220	                f"would silently overwrite the earlier and the gate would "
   221	                f"pass while discarding a listed expectation.")
   222	        out[key] = value
   223	    # Guard on `spec`, NOT `spec.strip()`: a whitespace-only value is a value
   224	    # the caller PASSED, and silently reading it as "no gate requested" is
   225	    # the same bypass as ",,," (codex round 3).  Only the default empty
   226	    # string means "no gate".
   227	    if spec and not out:
   228	        raise ValueError(
   229	            f"--expect-rounds={spec!r} parses to NO expectations; the gate "
   230	            f"would be skipped while looking like it ran.")
   231	    return out
   232	
   233	
   234	def check_expected_rounds(rows: list, expect: dict) -> list:
   235	    """Compare scored rounds against *expect*; return failure strings.
   236	
   237	    A listed pair that was never scored is a FAILURE, not a skip: otherwise
   238	    a sweep that silently dropped a method (unavailable ``pymetis``) or a
   239	    rank count would still report a clean gate.
   240	    """
   241	    scored = {
   242	        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
   243	        for r in rows
   244	        if r.get("available") and "schedule" in r and "n_ranks" in r
   245	    }
   246	    failures = []
   247	    for (method, n_ranks), want in sorted(expect.items()):
   248	        got = scored.get((method, n_ranks))
   249	        if got is None:
   250	            failures.append(
   251	                f"{method}:{n_ranks} expected rounds={want} but the pair was "
   252	                f"NOT SCORED (method unavailable, or not in this sweep)")
   253	        elif got != want:
   254	            failures.append(
   255	                f"{method}:{n_ranks} expected rounds={want}, got {got}")
   256	    return failures
   257	
   258	
   259	def schedule_cost_row(mesh, method: str, n_ranks: int) -> dict:
   260	    """SPMD halo-schedule depth for one (method, n_ranks) candidate.
   261	
   262	    Thin wrapper over the production
   263	    :func:`legoesm.parallel.sharded_dynamics.spmd_schedule_cost` — it reorders
   264	    the RAW mesh for ``n_ranks`` with ``method`` and colours the real
   265	    depth-``SPMD_HALO_DEPTH`` communication graph, so the number is the one
   266	    production pays, not a 1-ring lookalike.  ``halo_depth`` is deliberately
   267	    LEFT AT THE SCORER'S DEFAULT: this script's ``--halo-depth`` is the MPI
   268	    lane's (2), and scoring the SPMD schedule at 2 would colour a different
   269	    graph.
   270	
   271	    Adds ``coloring_gap = n_rounds - max_degree`` and the Vizing reading of
   272	    it (see the module docstring).  The headroom is reported as an INTERVAL,
   273	    because Vizing pins the optimum only to ``{Delta, Delta + 1}``:
   274	
   275	    * ``coloring_headroom_rounds_min = max(0, gap - 1)`` — rounds a perfect
   276	      recolouring is GUARANTEED to remove (it beats the ``Delta + 1`` case).
   277	    * ``coloring_headroom_rounds_max = gap`` — the best case, realized only
   278	      if the graph is Class 1.
   279	
   280	    It is deliberately NOT a "recolour vs ownership" verdict: at
   281	    ``gap == 1`` the min is 0 and the max is 1, i.e. genuinely inconclusive.
   282	
   283	    Errors are NOT caught.  The scorer's one refusal — a mesh padded for a
   284	    different reorder target, which would mis-slice the owned blocks — is
   285	    unreachable from here: this passes the raw mesh with the scorer's default
   286	    ``reorder_target = n_ranks``, and ``reorder_voronoi_for_sharding`` pads
   287	    ``nCells``/``nEdges`` to be divisible by exactly that target.  Wrapping
   288	    the call would therefore only swallow *unforeseen* failures into a row
   289	    that reads like an orderly skip, which is how a missing number turns into
   290	    a silently wrong table.  Rows already print as the sweep goes, so a raise
   291	    keeps the completed rungs in the log.
   292	    """
   293	    import time
   294	
   295	    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
   296	
   297	    t0 = time.perf_counter()
   298	    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
   299	    gap = int(cost["n_rounds"]) - int(cost["max_degree"])
   300	    return {
   301	        "n_rounds": int(cost["n_rounds"]),
   302	        "n_rounds_greedy": int(cost["n_rounds_greedy"]),
   303	        "max_degree": int(cost["max_degree"]),
   304	        # The decisive column, read through Vizing (see module docstring).
   305	        "coloring_gap": gap,
   306	        # Rounds a perfect recolouring could remove, as an INTERVAL: Vizing
   307	        # pins the optimum to {Delta, Delta+1}, so gap-1 is guaranteed and
   308	        # gap is the best case.  A gap of 1 spans [0, 1] = inconclusive.
   309	        "coloring_headroom_rounds_min": max(0, gap - 1),
   310	        "coloring_headroom_rounds_max": gap,
   311	        "coloring_optimal_proven": gap == 0,
   312	        # Scorer provenance, carried per row: a copied/flattened row must be
   313	        # able to show it scored a raw mesh partitioned for THIS device
   314	        # count, not one reordered for a different target.
   315	        "reorder_target": cost["reorder_target"],
   316	        "already_reordered": cost["already_reordered"],
   317	        "coloring_method": cost["coloring_method"],
   318	        "resolved_method": cost["resolved_method"],
   319	        "schedule_halo_depth": int(cost["halo_depth"]),
   320	        "cells_per_device": int(cost["cells_per_device"]),
   321	        # "allgather" => production runs no ppermute schedule here, so the
   322	        # round count above is COUNTERFACTUAL, not a cost production pays.
   323	        "production_strategy": cost["production_strategy"],
   324	        "score_seconds": round(time.perf_counter() - t0, 2),
   325	    }
   326	
   327	
   328	def main() -> int:
   329	    p = argparse.ArgumentParser(
   330	        description=__doc__,
   331	        formatter_class=argparse.RawDescriptionHelpFormatter)
   332	    p.add_argument("--subdivision", type=int, default=5,
   333	                   help="Icosahedral level (L5=10,242 cells; L6=40,962).")
   334	    p.add_argument("--rank-counts", type=str, default="2,4,8,16")
   335	    p.add_argument("--halo-depth", type=int, default=2,
   336	                   help="Halo layers (runtime default 2, del4 support).")
   337	    p.add_argument("--methods", type=str, default=",".join(METHODS))
   338	    p.add_argument("--schedule-cost", action="store_true",
   339	                   help="Also score the SPMD ppermute halo-schedule depth "
   340	                        "(n_rounds vs max_degree) per method x rank count. "
   341	                        "Uses the production SPMD halo depth, NOT "
   342	                        "--halo-depth. Expensive: minutes per candidate at "
   343	                        "subdiv>=8 — run it under batch.")
   344	    p.add_argument("--lloyd", type=int, default=50,
   345	                   help="Lloyd relaxation iterations for the mesh. 50 = the "
   346	                        "production SCVT key; 0 = the LABELLED synthetic "
   347	                        "scaling mesh. Recorded so a lloyd=0 mesh can never "
   348	                        "masquerade as a production receipt, and it must "
   349	                        "match the prewarmed cache key at subdiv>=9. "
   350	                        "Must be >= 0: the builder relaxes only when this is "
   351	                        "> 0, so a negative behaves exactly like 0 while "
   352	                        "being recorded (and cached) under a different key — "
   353	                        "false provenance.")
   354	    p.add_argument("--expect-rounds", type=str, default="",
   355	                   help="Instrument check, MECHANICAL. Comma-separated "
   356	                        "'<method>:<n_ranks>=<n_rounds>' expectations (e.g. "
   357	                        "'sfc:64=12,sfc:128=14'). Every listed pair must be "
   358	                        "scored and match, or main() returns 1 — so a caller "
   359	                        "that reproduces a known census can GATE on it "
   360	                        "instead of asserting agreement in a comment. "
   361	                        "Requires --schedule-cost.")
   362	    p.add_argument("--out", type=str,
   363	                   default="results/a1/voronoi_partition_quality.json")
   364	    args = p.parse_args()
   365	
   366	    rank_counts = [int(x) for x in args.rank_counts.split(",") if x]
   367	    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
   368	    for m in methods:
   369	        if m not in METHODS:
   370	            raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
   371	    if not rank_counts or any(n < 2 for n in rank_counts):
   372	        raise SystemExit("--rank-counts needs integers >= 2")
   373	    # Negative values in these three are all the SAME false-provenance bug:
   374	    # the underlying code treats them exactly like 0 (no relaxation, no
   375	    # bisection, `range(-1)` is empty), but the run is recorded under the
   376	    # negative value, so a level-0 mesh gets filed as "L-1".
   377	    if args.lloyd < 0:
   378	        raise SystemExit(
   379	            f"--lloyd must be >= 0, got {args.lloyd}: the mesh builder "
   380	            f"relaxes only for > 0, so a negative is indistinguishable from "
   381	            f"0 in the mesh but is recorded and cached under its own key.")
   382	    if args.subdivision < 0:
   383	        raise SystemExit(
   384	            f"--subdivision must be >= 0, got {args.subdivision}: the mesh "
   385	            f"builder bisects only for > 0, so a negative silently yields "
   386	            f"the level-0 base mesh while being recorded as "
   387	            f"L{args.subdivision}.")
   388	    if args.halo_depth < 0:
   389	        raise SystemExit(
   390	            f"--halo-depth must be >= 0, got {args.halo_depth}: the halo "
   391	            f"loop is `range(depth)`, so a negative behaves exactly like 0 "
   392	            f"while being recorded as {args.halo_depth}.")
   393	    expect = parse_expect_rounds(args.expect_rounds)
   394	    if expect and not args.schedule_cost:
   395	        raise SystemExit(
   396	            "--expect-rounds compares scored round counts, so it needs "
   397	            "--schedule-cost; without it nothing is scored and the gate "
   398	            "would pass vacuously.")
   399	
   400	
   401	    from legoesm.grids.voronoi import create_voronoi_mesh
   402	    from legoesm.parallel.voronoi_partition import resolve_partition_method
   403	
   404	    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
   405	                               lloyd_iterations=args.lloyd)
   406	    if max(rank_counts) > int(mesh.nCells):
   407	        raise SystemExit(
   408	            f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
   409	            f"{int(mesh.nCells)} cells (empty ranks are meaningless).")
   410	    print(f"mesh L{args.subdivision}: {int(mesh.nCells)} cells, "
   411	          f"{int(mesh.nEdges)} edges; auto -> "
   412	          f"{resolve_partition_method('auto')!r}")
   413	
   414	    rows = []
   415	    for method in methods:
   416	        if not method_available(method):
   417	            print(f"  {method:9s}: UNAVAILABLE (pymetis not importable) — "
   418	                  f"column omitted, never substituted")
   419	            rows.append({"method": method, "available": False})
   420	            continue
   421	        for n_ranks in rank_counts:
   422	            q = partition_quality(
   423	                mesh, owner_for(mesh, method, n_ranks), n_ranks,
   424	                halo_depth=args.halo_depth)
   425	            row = {"method": method, "available": True,
   426	                   "n_ranks": n_ranks, **q}
   427	            rows.append(row)
   428	            print(f"  {method:9s} np={n_ranks:3d} | "
   429	                  f"imbalance={q['load_imbalance_max_over_mean']:.3f} | "
   430	                  f"edge_cut={q['edge_cut']:6d} "
   431	                  f"({100 * q['edge_cut_fraction']:.2f}%) | "
   432	                  f"halo max={q['halo_cells_max']:5d} "
   433	                  f"mean={q['halo_cells_mean']:8.1f} | "
   434	                  f"halo/owned max={q['halo_owned_ratio_max']:.3f} | "
   435	                  f"nbrs max={q['neighbor_ranks_max']}")
   436	            if args.schedule_cost:
   437	                sc = schedule_cost_row(mesh, method, n_ranks)
   438	                row["schedule"] = sc
   439	                note = ("  [COUNTERFACTUAL: production auto-selects "
   440	                        "allgather here, no ppermute schedule]"
   441	                        if sc["production_strategy"] == "allgather" else "")
   442	                verdict = ("colouring PROVABLY OPTIMAL (Vizing floor)"
   443	                           if sc["coloring_optimal_proven"] else
   444	                           f"recolour headroom "
   445	                           f"{sc['coloring_headroom_rounds_min']}-"
   446	                           f"{sc['coloring_headroom_rounds_max']} round(s)")
   447	                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
   448	                      f"rounds={sc['n_rounds']:3d} "
   449	                      f"max_degree={sc['max_degree']:3d} "
   450	                      f"gap={sc['coloring_gap']:+d} "
   451	                      f"-> {verdict} "
   452	                      f"({sc['score_seconds']:.1f}s){note}", flush=True)
   453	
   454	    payload = {
   455	        "rows": rows,
   456	        "auto_resolves_to": resolve_partition_method("auto"),
   457	        "step_time_pointer": (
   458	            "step-time per method: bench_ocean_mpas_scaling.py / "
   459	            "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
   460	            "per launch, same case otherwise)"),
   461	        "metadata": annotate_incomplete(scaling_metadata(
   462	            grid="voronoi",
   463	            component="partitioning",
   464	            resolution=f"L{args.subdivision}",
   465	            n_levels=0,
   466	            precision="n/a",
   467	            decomposition="cell_partition",
   468	            solver_variant="n/a",
   469	            scaling_kind="partition-quality",
   470	            transport="none",
   471	            extra={"rank_counts": rank_counts, "methods": methods,
   472	                   "halo_depth": args.halo_depth,
   473	                   "schedule_cost": bool(args.schedule_cost),
   474	                   "lloyd_iterations": args.lloyd},
   475	        )),
   476	    }
   477	    outdir = os.path.dirname(args.out)
   478	    if outdir:
   479	        os.makedirs(outdir, exist_ok=True)
   480	    with open(args.out, "w") as f:
   481	        json.dump(payload, f, indent=2)
   482	    print(f"JSON: {args.out}")
   483	
   484	    if expect:
   485	        failures = check_expected_rounds(rows, expect)
   486	        payload["expected_rounds_check"] = {
   487	            "expected": {f"{m}:{n}": v for (m, n), v in sorted(expect.items())},
   488	            "failures": failures,
   489	            "passed": not failures,
   490	        }
   491	        with open(args.out, "w") as f:
   492	            json.dump(payload, f, indent=2)
   493	        if failures:
   494	            print(f"INSTRUMENT CHECK FAILED ({len(failures)}):")
   495	            for line in failures:
   496	                print(f"  {line}")
   497	            print("The scorer did NOT reproduce the known census — treat every "
   498	                  "unknown row in this run as UNTRUSTED.")
   499	            return 1
   500	        print(f"INSTRUMENT CHECK PASSED ({len(expect)} expectations)")
   501	    return 0
   502	
   503	
   504	if __name__ == "__main__":
   505	    raise SystemExit(main())
     1	"""Direct tests for the Voronoi partition-method quality bench (item 8)."""
     2	from __future__ import annotations
     3	
     4	import importlib.util
     5	import json
     6	import sys
     7	from pathlib import Path
     8	
     9	import numpy as np
    10	import pytest
    11	
    12	_BENCH = (Path(__file__).resolve().parents[2]
    13	          / "scripts" / "bench" / "bench_voronoi_partition_methods.py")
    14	_spec = importlib.util.spec_from_file_location("bench_vor_part", _BENCH)
    15	mod = importlib.util.module_from_spec(_spec)
    16	_spec.loader.exec_module(mod)
    17	
    18	
    19	def _mesh(level=2):
    20	    from legoesm.grids.voronoi import create_voronoi_mesh
    21	
    22	    return create_voronoi_mesh(subdivision_level=level)
    23	
    24	
    25	def test_partition_quality_metrics_shape_and_sanity():
    26	    mesh = _mesh()
    27	    owner = mod.owner_for(mesh, "geometric", 4)
    28	    q = mod.partition_quality(mesh, owner, 4)
    29	    assert q["cells_per_rank_min"] >= 1
    30	    assert q["cells_per_rank_max"] >= q["cells_per_rank_min"]
    31	    assert q["load_imbalance_max_over_mean"] >= 1.0
    32	    assert 0 < q["edge_cut"] < int(mesh.nEdges)
    33	    assert 0.0 < q["edge_cut_fraction"] < 1.0
    34	    assert q["halo_cells_max"] >= q["halo_cells_mean"] > 0
    35	    assert 1 <= q["neighbor_ranks_max"] < 4
    36	
    37	
    38	def test_partition_quality_rejects_bad_owner():
    39	    mesh = _mesh()
    40	    n = int(mesh.nCells)
    41	    with pytest.raises(AssertionError, match="out of range"):
    42	        mod.partition_quality(mesh, np.full(n, 7), 4)
    43	    with pytest.raises(AssertionError, match="shape"):
    44	        mod.partition_quality(mesh, np.zeros(n - 1, dtype=int), 4)
    45	
    46	
    47	def test_owner_for_unknown_method_raises():
    48	    mesh = _mesh()
    49	    with pytest.raises(ValueError, match="unknown partition method"):
    50	        mod.owner_for(mesh, "voodoo", 4)
    51	
    52	
    53	def test_method_available_never_substitutes():
    54	    # geometric/sfc are dependency-free; metis truthfully reports.
    55	    assert mod.method_available("geometric") is True
    56	    assert mod.method_available("sfc") is True
    57	    try:
    58	        import pymetis  # noqa: F401
    59	
    60	        assert mod.method_available("metis") is True
    61	    except Exception:
    62	        assert mod.method_available("metis") is False
    63	
    64	
    65	def test_main_writes_quality_table(tmp_path, monkeypatch):
    66	    out = tmp_path / "q.json"
    67	    monkeypatch.setattr(sys, "argv", [
    68	        "bench", "--subdivision", "2", "--rank-counts", "2,4",
    69	        "--methods", "geometric,sfc", "--out", str(out)])
    70	    assert mod.main() == 0
    71	    payload = json.loads(out.read_text())
    72	    rows = [r for r in payload["rows"] if r.get("available")]
    73	    assert {(r["method"], r["n_ranks"]) for r in rows} == {
    74	        ("geometric", 2), ("geometric", 4), ("sfc", 2), ("sfc", 4)}
    75	    assert payload["auto_resolves_to"] in ("metis", "geometric")
    76	    md = payload["metadata"]
    77	    assert md["transport"] == "none"
    78	    assert "_incomplete" not in md
    79	
    80	
    81	def test_main_rejects_bad_args(monkeypatch):
    82	    monkeypatch.setattr(sys, "argv", ["bench", "--methods", "voodoo"])
    83	    with pytest.raises(SystemExit):
    84	        mod.main()
    85	    monkeypatch.setattr(sys, "argv", ["bench", "--rank-counts", "1"])
    86	    with pytest.raises(SystemExit):
    87	        mod.main()
    88	
    89	
    90	class _SyntheticMesh:
    91	    """4-cell ring: cells 0-1-2-3 cyclic (each cell has 2 neighbors).
    92	
    93	    Edges: (0,1) (1,2) (2,3) (3,0) + one INVALID edge (-1,-1) to lock the
    94	    valid-edge masking in the edge-cut denominator.
    95	    """
    96	    nCells = 4
    97	    nEdges = 5
    98	    maxEdges = 2
    99	    cellsOnEdge = np.array([[0, 1, 2, 3, -1],
   100	                            [1, 2, 3, 0, -1]])
   101	    cellsOnCell = np.array([[1, 2, 3, 0],    # neighbor k=0
   102	                            [3, 0, 1, 2]])   # neighbor k=1
   103	
   104	
   105	def test_metric_definitions_locked_on_synthetic_mesh():
   106	    """Exact edge cut / halo / neighbor values on a hand-built ring —
   107	    a denominator or halo-construction drift fails HERE, not in a range
   108	    check (codex finding 3)."""
   109	    mesh = _SyntheticMesh()
   110	    owner = np.array([0, 0, 1, 1])  # cells 0,1 -> rank0; 2,3 -> rank1
   111	    q = mod.partition_quality(mesh, owner, 2, halo_depth=1)
   112	    # Cut edges: (1,2) and (3,0) -> 2 of 4 VALID edges (invalid edge
   113	    # excluded from the denominator).
   114	    assert q["edge_cut"] == 2
   115	    assert q["edge_cut_fraction"] == pytest.approx(0.5)
   116	    # halo_depth=1: each rank's halo = the 2 cells of the other rank that
   117	    # touch it (ring: both of them).
   118	    assert q["halo_cells_max"] == 2
   119	    assert q["halo_cells_mean"] == pytest.approx(2.0)
   120	    assert q["halo_owned_ratio_max"] == pytest.approx(1.0)
   121	    assert q["neighbor_ranks_max"] == 1
   122	    assert q["cells_per_rank_min"] == q["cells_per_rank_max"] == 2
   123	    assert q["load_imbalance_max_over_mean"] == pytest.approx(1.0)
   124	
   125	
   126	def test_empty_rank_rejected():
   127	    mesh = _SyntheticMesh()
   128	    owner = np.array([0, 0, 0, 0])  # rank 1 skipped
   129	    with pytest.raises(AssertionError, match="empty rank"):
   130	        mod.partition_quality(mesh, owner, 2)
   131	
   132	
   133	def test_halo_matches_runtime_partition():
   134	    """Real-mesh lock: the bench's halo size equals the RUNTIME partition's
   135	    (n_local - n_owned) for the same owner array — the bench reports the
   136	    runtime's halos, not an estimate (codex finding 3)."""
   137	    from legoesm.parallel.voronoi_partition import partition_voronoi_mesh
   138	
   139	    mesh = _mesh()
   140	    owner = mod.owner_for(mesh, "geometric", 4)
   141	    q = mod.partition_quality(mesh, owner, 4, halo_depth=2)
   142	    halos = []
   143	    for r in range(4):
   144	        part = partition_voronoi_mesh(
   145	            mesh, 4, r, method="geometric", halo_depth=2, cell_owner=owner)
   146	        halos.append(int(part.n_local_cells) - int(part.n_owned_cells))
   147	    assert q["halo_cells_max"] == max(halos)
   148	    assert q["halo_cells_mean"] == pytest.approx(float(np.mean(halos)))
   149	
   150	
   151	# --- SPMD halo-schedule depth (--schedule-cost) ---------------------------
   152	
   153	
   154	def test_schedule_cost_row_reports_rounds_against_their_lower_bound():
   155	    """``schedule_cost_row`` returns the REAL schedule depth and the bound it
   156	    must be read against."""
   157	    mesh = _mesh()
   158	    sc = mod.schedule_cost_row(mesh, "geometric", 2)
   159	    # max_degree is the graph's own lower bound on a proper edge colouring,
   160	    # so a schedule can never beat it.  -1 is the scorer's "not reported"
   161	    # sentinel and would make the gap meaningless.
   162	    assert sc["max_degree"] >= 1
   163	    assert sc["n_rounds"] >= sc["max_degree"]
   164	    assert sc["coloring_gap"] == sc["n_rounds"] - sc["max_degree"]
   165	    assert sc["n_rounds_greedy"] >= sc["n_rounds"]
   166	    assert sc["score_seconds"] >= 0.0
   167	
   168	
   169	@pytest.mark.parametrize(
   170	    "n_rounds, max_degree, gap, hmin, hmax, proven",
   171	    [
   172	        # Provably optimal: no proper edge colouring beats max_degree.
   173	        (12, 12, 0, 0, 0, True),
   174	        # Vizing allows the true optimum to BE max_degree+1, so a gap of 1
   175	        # spans [0, 1] — inconclusive.  This is the case that a naive
   176	        # "gap > 0 means recolour" rule would over-claim.
   177	        (13, 12, 1, 0, 1, False),
   178	        # Guaranteed to remove at least gap-1 = 3, at most gap = 4.
   179	        (14, 10, 4, 3, 4, False),
   180	    ])
   181	def test_coloring_gap_and_headroom_are_derived_not_assumed(
   182	        monkeypatch, n_rounds, max_degree, gap, hmin, hmax, proven):
   183	    """Gap and the Vizing-bounded headroom must be COMPUTED, not assumed.
   184	
   185	    Non-vacuity, the hard way: on every mesh small enough to test quickly the
   186	    real gap is 0 (measured L2/L3/L4 x {geometric,sfc} x nd 2-16, and s6
   187	    lloyd=0 at np8/np16 — the colourer lands exactly on ``max_degree`` every
   188	    time), so a real-mesh assertion cannot tell a correct subtraction from a
   189	    hardcoded ``0``; that exact mutation passed the first version of this
   190	    test.  Stubbing the production scorer with KNOWN values is what makes
   191	    the assertion able to fail.
   192	
   193	    The ``gap == 1`` row is the one that matters: the schedule is a proper
   194	    EDGE colouring and ``max_degree`` is that graph's max vertex degree, so
   195	    Vizing gives ``Delta <= chi' <= Delta + 1``.  A gap of 1 is therefore
   196	    indistinguishable from optimal (Class 2), and claiming recolouring
   197	    headroom there would be an over-claim.
   198	    """
   199	    stub = {
   200	        "n_rounds": n_rounds, "max_degree": max_degree,
   201	        "n_rounds_greedy": n_rounds + 1, "coloring_method": "stub",
   202	        "resolved_method": "geometric", "halo_depth": 3,
   203	        "cells_per_device": 99_999, "production_strategy": "ppermute",
   204	        "reorder_target": 8, "already_reordered": False,
   205	    }
   206	    import legoesm.parallel.sharded_dynamics as sd
   207	    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
   208	
   209	    sc = mod.schedule_cost_row(object(), "geometric", 8)
   210	    assert sc["coloring_gap"] == gap
   211	    # The headroom is an INTERVAL: Vizing pins the optimum to
   212	    # {Delta, Delta+1}, so gap-1 is guaranteed and gap is the best case.
   213	    assert sc["coloring_headroom_rounds_min"] == hmin
   214	    assert sc["coloring_headroom_rounds_max"] == hmax
   215	    assert sc["coloring_optimal_proven"] is proven
   216	
   217	
   218	def test_schedule_rounds_never_beat_the_vizing_floor_on_a_real_mesh():
   219	    """Real-mesh sanity on the bound itself: a proper edge colouring can
   220	    never use fewer rounds than the graph's max degree, and the multi-start
   221	    search should not overshoot Vizing's ``Delta + 1`` either.  If this ever
   222	    fires, ``max_degree`` is not the degree of the graph being coloured and
   223	    every gap-based conclusion built on it is void."""
   224	    for method in ("geometric", "sfc"):
   225	        for n_ranks in (2, 4, 8):
   226	            sc = mod.schedule_cost_row(_mesh(level=3), method, n_ranks)
   227	            assert sc["max_degree"] <= sc["n_rounds"] <= sc["max_degree"] + 1, (
   228	                f"{method} np={n_ranks}: rounds={sc['n_rounds']} "
   229	                f"max_degree={sc['max_degree']}")
   230	
   231	
   232	def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
   233	    """The schedule is scored at the SPMD production halo depth, NOT this
   234	    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
   235	    a different graph and quietly report the wrong lane's cost."""
   236	    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH
   237	
   238	    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
   239	    assert sc["schedule_halo_depth"] == SPMD_HALO_DEPTH
   240	
   241	
   242	def test_schedule_cost_matches_the_production_scorer_exactly():
   243	    """Lock: the wrapper reports what the production scorer returns — it is
   244	    a passthrough, not a re-derivation (the whole point: a 1-ring lookalike
   245	    reports 8 rounds where the real depth-3 graph reports 12-14)."""
   246	    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
   247	
   248	    mesh = _mesh()
   249	    ref = spmd_schedule_cost(mesh, 2, method="sfc")
   250	    sc = mod.schedule_cost_row(mesh, "sfc", 2)
   251	    for key in ("n_rounds", "max_degree", "n_rounds_greedy",
   252	                "coloring_method", "resolved_method", "cells_per_device",
   253	                "production_strategy"):
   254	        assert sc[key] == ref[key], key
   255	
   256	
   257	def test_schedule_cost_flag_is_opt_in_and_lands_in_the_json(
   258	        tmp_path, monkeypatch):
   259	    """Off by default (it is the expensive layer); on, every scored row
   260	    carries the schedule block and the run records that it ran."""
   261	    out = tmp_path / "off.json"
   262	    monkeypatch.setattr(sys, "argv", [
   263	        "bench", "--subdivision", "2", "--rank-counts", "2",
   264	        "--methods", "geometric", "--out", str(out)])
   265	    assert mod.main() == 0
   266	    payload = json.loads(out.read_text())
   267	    assert "schedule" not in payload["rows"][0]
   268	    assert payload["metadata"]["extra"]["schedule_cost"] is False
   269	
   270	    out2 = tmp_path / "on.json"
   271	    monkeypatch.setattr(sys, "argv", [
   272	        "bench", "--subdivision", "2", "--rank-counts", "2",
   273	        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
   274	    assert mod.main() == 0
   275	    payload2 = json.loads(out2.read_text())
   276	    row = payload2["rows"][0]
   277	    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
   278	    assert payload2["metadata"]["extra"]["schedule_cost"] is True
   279	    # Scorer provenance per row: a copied row must show it scored a mesh
   280	    # partitioned for THIS device count, not one reordered for another.
   281	    assert row["schedule"]["reorder_target"] == row["n_ranks"]
   282	    assert row["schedule"]["already_reordered"] is False
   283	
   284	
   285	def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
   286	    """``--lloyd`` must actually select the mesh, not just be recorded.
   287	
   288	    Non-vacuity: asserting only the recorded default (50) passes even if the
   289	    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
   290	    that.  This spies on the builder, so dropping ``lloyd_iterations=
   291	    args.lloyd`` fails here, and it checks a NON-default value so the
   292	    assertion cannot be satisfied by the default.
   293	    """
   294	    import legoesm.grids.voronoi as vor
   295	
   296	    seen = {}
   297	    real = vor.create_voronoi_mesh
   298	
   299	    def spy(*a, **k):
   300	        seen.update(k)
   301	        # lloyd=0 is cheap and is what the scaling meshes actually use.
   302	        return real(*a, **k)
   303	
   304	    monkeypatch.setattr(vor, "create_voronoi_mesh", spy)
   305	    out = tmp_path / "lloyd0.json"
   306	    monkeypatch.setattr(sys, "argv", [
   307	        "bench", "--subdivision", "2", "--rank-counts", "2",
   308	        "--methods", "geometric", "--lloyd", "0", "--out", str(out)])
   309	    assert mod.main() == 0
   310	    assert seen.get("lloyd_iterations") == 0, seen
   311	    # And it is recorded, so a synthetic mesh cannot be read back as a
   312	    # production SCVT receipt.
   313	    payload = json.loads(out.read_text())
   314	    assert payload["metadata"]["extra"]["lloyd_iterations"] == 0
   315	
   316	
   317	def test_expect_rounds_gate_fails_loudly_and_never_vacuously(
   318	        tmp_path, monkeypatch):
   319	    """The instrument check must FAIL on a wrong expectation and on a pair
   320	    that was never scored — a gate that can only pass is not a gate."""
   321	    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
   322	            "--methods", "geometric", "--schedule-cost"]
   323	
   324	    # Truth first: read what this configuration really scores.
   325	    out = tmp_path / "truth.json"
   326	    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
   327	    assert mod.main() == 0
   328	    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
   329	
   330	    # Matching expectation -> pass, and the check is recorded.
   331	    ok = tmp_path / "ok.json"
   332	    monkeypatch.setattr(sys, "argv", base + [
   333	        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
   334	    assert mod.main() == 0
   335	    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
   336	
   337	    # Wrong expectation -> non-zero exit.
   338	    bad = tmp_path / "bad.json"
   339	    monkeypatch.setattr(sys, "argv", base + [
   340	        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
   341	    assert mod.main() == 1
   342	    assert json.loads(bad.read_text())["expected_rounds_check"]["failures"]
   343	
   344	    # A pair that was never scored is a FAILURE, not a silent skip —
   345	    # otherwise a sweep missing a method still reports a clean gate.
   346	    missing = tmp_path / "missing.json"
   347	    monkeypatch.setattr(sys, "argv", base + [
   348	        "--expect-rounds", "sfc:2=3", "--out", str(missing)])
   349	    assert mod.main() == 1
   350	    fails = json.loads(missing.read_text())["expected_rounds_check"]["failures"]
   351	    assert any("NOT SCORED" in f for f in fails), fails
   352	
   353	
   354	def test_expect_rounds_refuses_to_pass_vacuously_without_scoring(monkeypatch):
   355	    """Without --schedule-cost nothing is scored, so the gate would pass on
   356	    an empty comparison. It must refuse instead."""
   357	    monkeypatch.setattr(sys, "argv", [
   358	        "bench", "--subdivision", "2", "--rank-counts", "2",
   359	        "--methods", "geometric", "--expect-rounds", "geometric:2=1",
   360	        "--out", "/dev/null"])
   361	    with pytest.raises(SystemExit, match="needs --schedule-cost"):
   362	        mod.main()
   363	
   364	
   365	@pytest.mark.parametrize("spec", ["geometric:2", "geometric=2", "voodoo:2=3"])
   366	def test_expect_rounds_rejects_malformed_specs(spec):
   367	    """A typo'd expectation must raise, never be dropped — a silently
   368	    skipped expectation turns the gate into a no-op."""
   369	    with pytest.raises(ValueError, match="expect-rounds"):
   370	        mod.parse_expect_rounds(spec)
   371	
   372	
   373	def test_expect_rounds_rejects_duplicate_and_empty_specs():
   374	    """Two more ways the gate could be silently bypassed (codex round 2).
   375	
   376	    A duplicate key would let the later expectation overwrite the earlier,
   377	    so a listed-but-wrong expectation is discarded and the gate passes. A
   378	    non-empty spec that parses to nothing (``",,,"``) would make main() skip
   379	    the check entirely while the caller believes it ran.
   380	    """
   381	    with pytest.raises(ValueError, match="duplicate expectation"):
   382	        mod.parse_expect_rounds("geometric:2=999,geometric:2=13")
   383	    with pytest.raises(ValueError, match="NO expectations"):
   384	        mod.parse_expect_rounds(",,,")
   385	    # Whitespace-only is a value the caller PASSED; reading it as "no gate"
   386	    # is the same bypass (codex round 3 — the first fix guarded on
   387	    # spec.strip() and let this through).
   388	    for blank in ("   ", "\t", " , , "):
   389	        with pytest.raises(ValueError, match="NO expectations"):
   390	            mod.parse_expect_rounds(blank)
   391	    # A genuinely empty spec is the "no gate requested" default, not an error.
   392	    assert mod.parse_expect_rounds("") == {}
   393	
   394	
   395	@pytest.mark.parametrize("flag, value", [
   396	    ("--lloyd", "-1"), ("--subdivision", "-1"), ("--halo-depth", "-1")])
   397	def test_negative_numeric_args_refused_as_false_provenance(
   398	        monkeypatch, flag, value):
   399	    """All three behave exactly like 0 in the underlying code but would be
   400	    RECORDED under the negative value — a level-0 mesh filed as 'L-1'."""
   401	    argv = ["bench", "--subdivision", "2", "--rank-counts", "2",
   402	            "--methods", "geometric", "--out", "/dev/null"]
   403	    # Replace the flag if already present, else append.
   404	    if flag in argv:
   405	        argv[argv.index(flag) + 1] = value
   406	    else:
   407	        argv += [flag, value]
   408	    monkeypatch.setattr(sys, "argv", argv)
   409	    with pytest.raises(SystemExit, match=r"must be >= 0"):
   410	        mod.main()
   411	
   412	
   413	def test_expect_rounds_fails_when_a_method_is_unavailable(
   414	        tmp_path, monkeypatch):
   415	    """An expectation naming a method that reported UNAVAILABLE must fail.
   416	
   417	    Distinct from omitting the method from --methods: here the sweep asks
   418	    for it and the partitioner is missing, which is exactly how a two-method
   419	    table gets misread as a three-method one.
   420	    """
   421	    monkeypatch.setattr(mod, "method_available",
   422	                        lambda m: m != "metis")
   423	    out = tmp_path / "unavail.json"
   424	    monkeypatch.setattr(sys, "argv", [
   425	        "bench", "--subdivision", "2", "--rank-counts", "2",
   426	        "--methods", "geometric,metis", "--schedule-cost",
   427	        "--expect-rounds", "metis:2=1", "--out", str(out)])
   428	    assert mod.main() == 1
   429	    payload = json.loads(out.read_text())
   430	    fails = payload["expected_rounds_check"]["failures"]
   431	    assert any("NOT SCORED" in f for f in fails), fails
   432	    # Assert the EXPLICIT unavailable row, not just the failure: without
   433	    # this the test would also pass if metis were silently omitted, which
   434	    # is the very substitution this bench refuses to make.
   435	    metis_rows = [r for r in payload["rows"] if r["method"] == "metis"]
   436	    assert metis_rows and metis_rows[0]["available"] is False, payload["rows"]
   437	
   438	
   439	def test_lloyd_rejects_negative(monkeypatch):
   440	    """A negative Lloyd count behaves like 0 in the builder (it relaxes only
   441	    for > 0) but is recorded and cached under its own key — false
   442	    provenance, so the CLI must refuse it rather than run."""
   443	    monkeypatch.setattr(sys, "argv", [
   444	        "bench", "--subdivision", "2", "--rank-counts", "2",
   445	        "--methods", "geometric", "--lloyd", "-1", "--out", "/dev/null"])
   446	    with pytest.raises(SystemExit, match=r"--lloyd must be >= 0"):
   447	        mod.main()

 succeeded in 698ms:
diff --git a/scripts/bench/bench_voronoi_partition_methods.py b/scripts/bench/bench_voronoi_partition_methods.py
index 6aa0f4a40..3f7e61ec5 100644
--- a/scripts/bench/bench_voronoi_partition_methods.py
+++ b/scripts/bench/bench_voronoi_partition_methods.py
@@ -8,6 +8,54 @@ Scaling-audit item 8.  Two independent measurement layers:
    for each method x rank-count on the real icosahedral mesh.  These are
    the numbers ``resolve_partition_method``'s ``auto`` policy must be
    justified by.
+1b. **SPMD halo-schedule depth** (``--schedule-cost``, opt-in because it is
+   the expensive layer): ``n_rounds`` — the number of SEQUENTIAL ppermute
+   rounds one halo fill costs — and ``max_degree``, the communication
+   graph's lower bound on it.  This is the term that binds MPAS GPU strong
+   scaling above ~64 devices, and layer 1 CANNOT stand in for it: the
+   neighbor fan-out above is a 1-ring proxy that reported 8 rounds for
+   every method and rank count while the real depth-3-plus-closure
+   schedule reported 12-14.  Scored by the production
+   ``spmd_schedule_cost`` (which calls the production builders), never a
+   re-derived lookalike.
+
+   The DECISIVE column is ``coloring_gap = n_rounds - max_degree``, read
+   through VIZING'S THEOREM, which bounds what recolouring could ever buy.
+   The schedule is a proper EDGE colouring of the device communication
+   graph (one colour = one ppermute round; ``_build_ppermute_schedule``
+   asserts properness), and ``max_degree`` is that same graph's maximum
+   vertex degree.  So the chromatic index obeys ``Delta <= chi' <=
+   Delta + 1``: the gap is a bound on recolouring headroom, NOT a
+   yes/no flag.
+
+   * ``gap == 0`` -> ``n_rounds == Delta``, and no proper edge colouring
+     can beat ``Delta``.  The colouring is PROVABLY OPTIMAL; recolouring
+     headroom is exactly ZERO.
+   * ``gap == 1`` -> INCONCLUSIVE.  A Class 2 graph genuinely needs
+     ``Delta + 1``, and deciding Class 1 vs Class 2 is NP-complete, so
+     this neither establishes nor excludes a one-round win.
+   * ``gap >= 2`` -> recolouring is guaranteed to remove AT LEAST
+     ``gap - 1`` rounds (the optimum is at worst ``Delta + 1``) and at
+     most ``gap``.
+
+   SCOPE, and it is not a formality: ``Delta`` bounds only a proper
+   UNDIRECTED edge colouring of THIS graph.  ``_build_ppermute_schedule``
+   enters a device pair into ``comm_pairs`` when EITHER direction has a
+   halo dependency and then emits BOTH ppermute directions, even where one
+   send map is empty.  A redesigned DIRECTED schedule that exploits
+   one-way exchanges is therefore not bounded by ``Delta`` at all, so
+   ``gap == 0`` must never be reported as "only ownership can help" — it
+   rules out a better undirected edge colouring of this graph, and
+   nothing more.
+
+   NOTE the halo depth differs by lane and is NOT ``--halo-depth``: that
+   flag is the MPI lane's (default 2), while the schedule is scored at the
+   SPMD production depth ``SPMD_HALO_DEPTH`` (3).  Both are recorded.
+   ``n_rounds`` is per HALO FILL, not per step — multiply by the tendency
+   evaluations of the integrator actually run.  When
+   ``production_strategy`` is ``"allgather"`` (auto-selected below the
+   cells/device threshold) there is no ppermute schedule in production and
+   the round count is COUNTERFACTUAL; the row says so.
 2. **Step time** (optional pointer, NOT run here): drive the existing
    MPI lane with ``bench_ocean_mpas_scaling.py --partition-method <m>``
    (ocean) or ``bench_mpas_spmd_scaling.py --partition-method <m>``
@@ -23,6 +71,11 @@ exactly one rank; owner range valid) before any metric is recorded.
 Run:
   python scripts/bench/bench_voronoi_partition_methods.py \
       --subdivision 6 --rank-counts 2,4,8,16 --out results/partition_quality.json
+
+  # + the SPMD schedule depth (minutes to hours at subdiv>=8 — batch it):
+  python scripts/bench/bench_voronoi_partition_methods.py \
+      --subdivision 9 --rank-counts 64,128 --schedule-cost \
+      --out results/a1/schedule_cost_s9.json
 """
 from __future__ import annotations
 
@@ -128,6 +181,150 @@ def owner_for(mesh, method: str, n_ranks: int) -> np.ndarray:
                      f"expected one of {METHODS}")
 
 
+def parse_expect_rounds(spec: str) -> dict:
+    """Parse ``'sfc:64=12,metis:128=19'`` into ``{("sfc", 64): 12}``.
+
+    Raises on anything malformed rather than skipping it — a typo'd
+    expectation that is silently dropped turns the gate into a no-op, which
+    is exactly the failure this flag exists to prevent.  Three ways that
+    could happen, all rejected here:
+
+    * a malformed item (``geometric:2``, ``geometric=2``);
+    * an unknown method;
+    * a DUPLICATE key — ``geometric:2=999,geometric:2=13`` would otherwise
+      let the second silently overwrite the first and pass;
+    * a non-empty spec that parses to NOTHING (``",,,"``), which would make
+      the caller skip the check while believing it ran.
+    """
+    out: dict[tuple[str, int], int] = {}
+    for item in (s.strip() for s in spec.split(",")):
+        if not item:
+            continue
+        try:
+            lhs, rounds = item.split("=")
+            method, n_ranks = lhs.split(":")
+            key = (method.strip(), int(n_ranks))
+            value = int(rounds)
+        except ValueError as exc:
+            raise ValueError(
+                f"--expect-rounds: cannot parse {item!r}; expected "
+                f"'<method>:<n_ranks>=<n_rounds>'") from exc
+        if key[0] not in METHODS:
+            raise ValueError(
+                f"--expect-rounds: unknown method {key[0]!r} in {item!r}; "
+                f"expected one of {METHODS}")
+        if key in out:
+            raise ValueError(
+                f"--expect-rounds: duplicate expectation for "
+                f"{key[0]}:{key[1]} ({out[key]} then {value}); the later one "
+                f"would silently overwrite the earlier and the gate would "
+                f"pass while discarding a listed expectation.")
+        out[key] = value
+    # Guard on `spec`, NOT `spec.strip()`: a whitespace-only value is a value
+    # the caller PASSED, and silently reading it as "no gate requested" is
+    # the same bypass as ",,," (codex round 3).  Only the default empty
+    # string means "no gate".
+    if spec and not out:
+        raise ValueError(
+            f"--expect-rounds={spec!r} parses to NO expectations; the gate "
+            f"would be skipped while looking like it ran.")
+    return out
+
+
+def check_expected_rounds(rows: list, expect: dict) -> list:
+    """Compare scored rounds against *expect*; return failure strings.
+
+    A listed pair that was never scored is a FAILURE, not a skip: otherwise
+    a sweep that silently dropped a method (unavailable ``pymetis``) or a
+    rank count would still report a clean gate.
+    """
+    scored = {
+        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
+        for r in rows
+        if r.get("available") and "schedule" in r and "n_ranks" in r
+    }
+    failures = []
+    for (method, n_ranks), want in sorted(expect.items()):
+        got = scored.get((method, n_ranks))
+        if got is None:
+            failures.append(
+                f"{method}:{n_ranks} expected rounds={want} but the pair was "
+                f"NOT SCORED (method unavailable, or not in this sweep)")
+        elif got != want:
+            failures.append(
+                f"{method}:{n_ranks} expected rounds={want}, got {got}")
+    return failures
+
+
+def schedule_cost_row(mesh, method: str, n_ranks: int) -> dict:
+    """SPMD halo-schedule depth for one (method, n_ranks) candidate.
+
+    Thin wrapper over the production
+    :func:`legoesm.parallel.sharded_dynamics.spmd_schedule_cost` — it reorders
+    the RAW mesh for ``n_ranks`` with ``method`` and colours the real
+    depth-``SPMD_HALO_DEPTH`` communication graph, so the number is the one
+    production pays, not a 1-ring lookalike.  ``halo_depth`` is deliberately
+    LEFT AT THE SCORER'S DEFAULT: this script's ``--halo-depth`` is the MPI
+    lane's (2), and scoring the SPMD schedule at 2 would colour a different
+    graph.
+
+    Adds ``coloring_gap = n_rounds - max_degree`` and the Vizing reading of
+    it (see the module docstring).  The headroom is reported as an INTERVAL,
+    because Vizing pins the optimum only to ``{Delta, Delta + 1}``:
+
+    * ``coloring_headroom_rounds_min = max(0, gap - 1)`` — rounds a perfect
+      recolouring is GUARANTEED to remove (it beats the ``Delta + 1`` case).
+    * ``coloring_headroom_rounds_max = gap`` — the best case, realized only
+      if the graph is Class 1.
+
+    It is deliberately NOT a "recolour vs ownership" verdict: at
+    ``gap == 1`` the min is 0 and the max is 1, i.e. genuinely inconclusive.
+
+    Errors are NOT caught.  The scorer's one refusal — a mesh padded for a
+    different reorder target, which would mis-slice the owned blocks — is
+    unreachable from here: this passes the raw mesh with the scorer's default
+    ``reorder_target = n_ranks``, and ``reorder_voronoi_for_sharding`` pads
+    ``nCells``/``nEdges`` to be divisible by exactly that target.  Wrapping
+    the call would therefore only swallow *unforeseen* failures into a row
+    that reads like an orderly skip, which is how a missing number turns into
+    a silently wrong table.  Rows already print as the sweep goes, so a raise
+    keeps the completed rungs in the log.
+    """
+    import time
+
+    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
+
+    t0 = time.perf_counter()
+    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
+    gap = int(cost["n_rounds"]) - int(cost["max_degree"])
+    return {
+        "n_rounds": int(cost["n_rounds"]),
+        "n_rounds_greedy": int(cost["n_rounds_greedy"]),
+        "max_degree": int(cost["max_degree"]),
+        # The decisive column, read through Vizing (see module docstring).
+        "coloring_gap": gap,
+        # Rounds a perfect recolouring could remove, as an INTERVAL: Vizing
+        # pins the optimum to {Delta, Delta+1}, so gap-1 is guaranteed and
+        # gap is the best case.  A gap of 1 spans [0, 1] = inconclusive.
+        "coloring_headroom_rounds_min": max(0, gap - 1),
+        "coloring_headroom_rounds_max": gap,
+        "coloring_optimal_proven": gap == 0,
+        # Scorer provenance, carried per row: a copied/flattened row must be
+        # able to show it scored a raw mesh partitioned for THIS device
+        # count, not one reordered for a different target.
+        "reorder_target": cost["reorder_target"],
+        "already_reordered": cost["already_reordered"],
+        "coloring_method": cost["coloring_method"],
+        "resolved_method": cost["resolved_method"],
+        "schedule_halo_depth": int(cost["halo_depth"]),
+        "cells_per_device": int(cost["cells_per_device"]),
+        # "allgather" => production runs no ppermute schedule here, so the
+        # round count above is COUNTERFACTUAL, not a cost production pays.
+        "production_strategy": cost["production_strategy"],
+        "score_seconds": round(time.perf_counter() - t0, 2),
+    }
+
+
 def main() -> int:
     p = argparse.ArgumentParser(
         description=__doc__,
@@ -138,6 +335,30 @@ def main() -> int:
     p.add_argument("--halo-depth", type=int, default=2,
                    help="Halo layers (runtime default 2, del4 support).")
     p.add_argument("--methods", type=str, default=",".join(METHODS))
+    p.add_argument("--schedule-cost", action="store_true",
+                   help="Also score the SPMD ppermute halo-schedule depth "
+                        "(n_rounds vs max_degree) per method x rank count. "
+                        "Uses the production SPMD halo depth, NOT "
+                        "--halo-depth. Expensive: minutes per candidate at "
+                        "subdiv>=8 — run it under batch.")
+    p.add_argument("--lloyd", type=int, default=50,
+                   help="Lloyd relaxation iterations for the mesh. 50 = the "
+                        "production SCVT key; 0 = the LABELLED synthetic "
+                        "scaling mesh. Recorded so a lloyd=0 mesh can never "
+                        "masquerade as a production receipt, and it must "
+                        "match the prewarmed cache key at subdiv>=9. "
+                        "Must be >= 0: the builder relaxes only when this is "
+                        "> 0, so a negative behaves exactly like 0 while "
+                        "being recorded (and cached) under a different key — "
+                        "false provenance.")
+    p.add_argument("--expect-rounds", type=str, default="",
+                   help="Instrument check, MECHANICAL. Comma-separated "
+                        "'<method>:<n_ranks>=<n_rounds>' expectations (e.g. "
+                        "'sfc:64=12,sfc:128=14'). Every listed pair must be "
+                        "scored and match, or main() returns 1 — so a caller "
+                        "that reproduces a known census can GATE on it "
+                        "instead of asserting agreement in a comment. "
+                        "Requires --schedule-cost.")
     p.add_argument("--out", type=str,
                    default="results/a1/voronoi_partition_quality.json")
     args = p.parse_args()
@@ -149,12 +370,39 @@ def main() -> int:
             raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
     if not rank_counts or any(n < 2 for n in rank_counts):
         raise SystemExit("--rank-counts needs integers >= 2")
+    # Negative values in these three are all the SAME false-provenance bug:
+    # the underlying code treats them exactly like 0 (no relaxation, no
+    # bisection, `range(-1)` is empty), but the run is recorded under the
+    # negative value, so a level-0 mesh gets filed as "L-1".
+    if args.lloyd < 0:
+        raise SystemExit(
+            f"--lloyd must be >= 0, got {args.lloyd}: the mesh builder "
+            f"relaxes only for > 0, so a negative is indistinguishable from "
+            f"0 in the mesh but is recorded and cached under its own key.")
+    if args.subdivision < 0:
+        raise SystemExit(
+            f"--subdivision must be >= 0, got {args.subdivision}: the mesh "
+            f"builder bisects only for > 0, so a negative silently yields "
+            f"the level-0 base mesh while being recorded as "
+            f"L{args.subdivision}.")
+    if args.halo_depth < 0:
+        raise SystemExit(
+            f"--halo-depth must be >= 0, got {args.halo_depth}: the halo "
+            f"loop is `range(depth)`, so a negative behaves exactly like 0 "
+            f"while being recorded as {args.halo_depth}.")
+    expect = parse_expect_rounds(args.expect_rounds)
+    if expect and not args.schedule_cost:
+        raise SystemExit(
+            "--expect-rounds compares scored round counts, so it needs "
+            "--schedule-cost; without it nothing is scored and the gate "
+            "would pass vacuously.")
 
 
     from legoesm.grids.voronoi import create_voronoi_mesh
     from legoesm.parallel.voronoi_partition import resolve_partition_method
 
-    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
+    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
+                               lloyd_iterations=args.lloyd)
     if max(rank_counts) > int(mesh.nCells):
         raise SystemExit(
             f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
@@ -185,6 +433,23 @@ def main() -> int:
                   f"mean={q['halo_cells_mean']:8.1f} | "
                   f"halo/owned max={q['halo_owned_ratio_max']:.3f} | "
                   f"nbrs max={q['neighbor_ranks_max']}")
+            if args.schedule_cost:
+                sc = schedule_cost_row(mesh, method, n_ranks)
+                row["schedule"] = sc
+                note = ("  [COUNTERFACTUAL: production auto-selects "
+                        "allgather here, no ppermute schedule]"
+                        if sc["production_strategy"] == "allgather" else "")
+                verdict = ("colouring PROVABLY OPTIMAL (Vizing floor)"
+                           if sc["coloring_optimal_proven"] else
+                           f"recolour headroom "
+                           f"{sc['coloring_headroom_rounds_min']}-"
+                           f"{sc['coloring_headroom_rounds_max']} round(s)")
+                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
+                      f"rounds={sc['n_rounds']:3d} "
+                      f"max_degree={sc['max_degree']:3d} "
+                      f"gap={sc['coloring_gap']:+d} "
+                      f"-> {verdict} "
+                      f"({sc['score_seconds']:.1f}s){note}", flush=True)
 
     payload = {
         "rows": rows,
@@ -204,7 +469,9 @@ def main() -> int:
             scaling_kind="partition-quality",
             transport="none",
             extra={"rank_counts": rank_counts, "methods": methods,
-                   "halo_depth": args.halo_depth},
+                   "halo_depth": args.halo_depth,
+                   "schedule_cost": bool(args.schedule_cost),
+                   "lloyd_iterations": args.lloyd},
         )),
     }
     outdir = os.path.dirname(args.out)
@@ -213,6 +480,24 @@ def main() -> int:
     with open(args.out, "w") as f:
         json.dump(payload, f, indent=2)
     print(f"JSON: {args.out}")
+
+    if expect:
+        failures = check_expected_rounds(rows, expect)
+        payload["expected_rounds_check"] = {
+            "expected": {f"{m}:{n}": v for (m, n), v in sorted(expect.items())},
+            "failures": failures,
+            "passed": not failures,
+        }
+        with open(args.out, "w") as f:
+            json.dump(payload, f, indent=2)
+        if failures:
+            print(f"INSTRUMENT CHECK FAILED ({len(failures)}):")
+            for line in failures:
+                print(f"  {line}")
+            print("The scorer did NOT reproduce the known census — treat every "
+                  "unknown row in this run as UNTRUSTED.")
+            return 1
+        print(f"INSTRUMENT CHECK PASSED ({len(expect)} expectations)")
     return 0
 
 
diff --git a/tests/bench/test_bench_voronoi_partition_methods.py b/tests/bench/test_bench_voronoi_partition_methods.py
index 21f861fd7..ff4d93433 100644
--- a/tests/bench/test_bench_voronoi_partition_methods.py
+++ b/tests/bench/test_bench_voronoi_partition_methods.py
@@ -146,3 +146,302 @@ def test_halo_matches_runtime_partition():
         halos.append(int(part.n_local_cells) - int(part.n_owned_cells))
     assert q["halo_cells_max"] == max(halos)
     assert q["halo_cells_mean"] == pytest.approx(float(np.mean(halos)))
+
+
+# --- SPMD halo-schedule depth (--schedule-cost) ---------------------------
+
+
+def test_schedule_cost_row_reports_rounds_against_their_lower_bound():
+    """``schedule_cost_row`` returns the REAL schedule depth and the bound it
+    must be read against."""
+    mesh = _mesh()
+    sc = mod.schedule_cost_row(mesh, "geometric", 2)
+    # max_degree is the graph's own lower bound on a proper edge colouring,
+    # so a schedule can never beat it.  -1 is the scorer's "not reported"
+    # sentinel and would make the gap meaningless.
+    assert sc["max_degree"] >= 1
+    assert sc["n_rounds"] >= sc["max_degree"]
+    assert sc["coloring_gap"] == sc["n_rounds"] - sc["max_degree"]
+    assert sc["n_rounds_greedy"] >= sc["n_rounds"]
+    assert sc["score_seconds"] >= 0.0
+
+
+@pytest.mark.parametrize(
+    "n_rounds, max_degree, gap, hmin, hmax, proven",
+    [
+        # Provably optimal: no proper edge colouring beats max_degree.
+        (12, 12, 0, 0, 0, True),
+        # Vizing allows the true optimum to BE max_degree+1, so a gap of 1
+        # spans [0, 1] — inconclusive.  This is the case that a naive
+        # "gap > 0 means recolour" rule would over-claim.
+        (13, 12, 1, 0, 1, False),
+        # Guaranteed to remove at least gap-1 = 3, at most gap = 4.
+        (14, 10, 4, 3, 4, False),
+    ])
+def test_coloring_gap_and_headroom_are_derived_not_assumed(
+        monkeypatch, n_rounds, max_degree, gap, hmin, hmax, proven):
+    """Gap and the Vizing-bounded headroom must be COMPUTED, not assumed.
+
+    Non-vacuity, the hard way: on every mesh small enough to test quickly the
+    real gap is 0 (measured L2/L3/L4 x {geometric,sfc} x nd 2-16, and s6
+    lloyd=0 at np8/np16 — the colourer lands exactly on ``max_degree`` every
+    time), so a real-mesh assertion cannot tell a correct subtraction from a
+    hardcoded ``0``; that exact mutation passed the first version of this
+    test.  Stubbing the production scorer with KNOWN values is what makes
+    the assertion able to fail.
+
+    The ``gap == 1`` row is the one that matters: the schedule is a proper
+    EDGE colouring and ``max_degree`` is that graph's max vertex degree, so
+    Vizing gives ``Delta <= chi' <= Delta + 1``.  A gap of 1 is therefore
+    indistinguishable from optimal (Class 2), and claiming recolouring
+    headroom there would be an over-claim.
+    """
+    stub = {
+        "n_rounds": n_rounds, "max_degree": max_degree,
+        "n_rounds_greedy": n_rounds + 1, "coloring_method": "stub",
+        "resolved_method": "geometric", "halo_depth": 3,
+        "cells_per_device": 99_999, "production_strategy": "ppermute",
+        "reorder_target": 8, "already_reordered": False,
+    }
+    import legoesm.parallel.sharded_dynamics as sd
+    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
+
+    sc = mod.schedule_cost_row(object(), "geometric", 8)
+    assert sc["coloring_gap"] == gap
+    # The headroom is an INTERVAL: Vizing pins the optimum to
+    # {Delta, Delta+1}, so gap-1 is guaranteed and gap is the best case.
+    assert sc["coloring_headroom_rounds_min"] == hmin
+    assert sc["coloring_headroom_rounds_max"] == hmax
+    assert sc["coloring_optimal_proven"] is proven
+
+
+def test_schedule_rounds_never_beat_the_vizing_floor_on_a_real_mesh():
+    """Real-mesh sanity on the bound itself: a proper edge colouring can
+    never use fewer rounds than the graph's max degree, and the multi-start
+    search should not overshoot Vizing's ``Delta + 1`` either.  If this ever
+    fires, ``max_degree`` is not the degree of the graph being coloured and
+    every gap-based conclusion built on it is void."""
+    for method in ("geometric", "sfc"):
+        for n_ranks in (2, 4, 8):
+            sc = mod.schedule_cost_row(_mesh(level=3), method, n_ranks)
+            assert sc["max_degree"] <= sc["n_rounds"] <= sc["max_degree"] + 1, (
+                f"{method} np={n_ranks}: rounds={sc['n_rounds']} "
+                f"max_degree={sc['max_degree']}")
+
+
+def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
+    """The schedule is scored at the SPMD production halo depth, NOT this
+    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
+    a different graph and quietly report the wrong lane's cost."""
+    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH
+
+    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
+    assert sc["schedule_halo_depth"] == SPMD_HALO_DEPTH
+
+
+def test_schedule_cost_matches_the_production_scorer_exactly():
+    """Lock: the wrapper reports what the production scorer returns — it is
+    a passthrough, not a re-derivation (the whole point: a 1-ring lookalike
+    reports 8 rounds where the real depth-3 graph reports 12-14)."""
+    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
+
+    mesh = _mesh()
+    ref = spmd_schedule_cost(mesh, 2, method="sfc")
+    sc = mod.schedule_cost_row(mesh, "sfc", 2)
+    for key in ("n_rounds", "max_degree", "n_rounds_greedy",
+                "coloring_method", "resolved_method", "cells_per_device",
+                "production_strategy"):
+        assert sc[key] == ref[key], key
+
+
+def test_schedule_cost_flag_is_opt_in_and_lands_in_the_json(
+        tmp_path, monkeypatch):
+    """Off by default (it is the expensive layer); on, every scored row
+    carries the schedule block and the run records that it ran."""
+    out = tmp_path / "off.json"
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", "geometric", "--out", str(out)])
+    assert mod.main() == 0
+    payload = json.loads(out.read_text())
+    assert "schedule" not in payload["rows"][0]
+    assert payload["metadata"]["extra"]["schedule_cost"] is False
+
+    out2 = tmp_path / "on.json"
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
+    assert mod.main() == 0
+    payload2 = json.loads(out2.read_text())
+    row = payload2["rows"][0]
+    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
+    assert payload2["metadata"]["extra"]["schedule_cost"] is True
+    # Scorer provenance per row: a copied row must show it scored a mesh
+    # partitioned for THIS device count, not one reordered for another.
+    assert row["schedule"]["reorder_target"] == row["n_ranks"]
+    assert row["schedule"]["already_reordered"] is False
+
+
+def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
+    """``--lloyd`` must actually select the mesh, not just be recorded.
+
+    Non-vacuity: asserting only the recorded default (50) passes even if the
+    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
+    that.  This spies on the builder, so dropping ``lloyd_iterations=
+    args.lloyd`` fails here, and it checks a NON-default value so the
+    assertion cannot be satisfied by the default.
+    """
+    import legoesm.grids.voronoi as vor
+
+    seen = {}
+    real = vor.create_voronoi_mesh
+
+    def spy(*a, **k):
+        seen.update(k)
+        # lloyd=0 is cheap and is what the scaling meshes actually use.
+        return real(*a, **k)
+
+    monkeypatch.setattr(vor, "create_voronoi_mesh", spy)
+    out = tmp_path / "lloyd0.json"
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", "geometric", "--lloyd", "0", "--out", str(out)])
+    assert mod.main() == 0
+    assert seen.get("lloyd_iterations") == 0, seen
+    # And it is recorded, so a synthetic mesh cannot be read back as a
+    # production SCVT receipt.
+    payload = json.loads(out.read_text())
+    assert payload["metadata"]["extra"]["lloyd_iterations"] == 0
+
+
+def test_expect_rounds_gate_fails_loudly_and_never_vacuously(
+        tmp_path, monkeypatch):
+    """The instrument check must FAIL on a wrong expectation and on a pair
+    that was never scored — a gate that can only pass is not a gate."""
+    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
+            "--methods", "geometric", "--schedule-cost"]
+
+    # Truth first: read what this configuration really scores.
+    out = tmp_path / "truth.json"
+    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
+    assert mod.main() == 0
+    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
+
+    # Matching expectation -> pass, and the check is recorded.
+    ok = tmp_path / "ok.json"
+    monkeypatch.setattr(sys, "argv", base + [
+        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
+    assert mod.main() == 0
+    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
+
+    # Wrong expectation -> non-zero exit.
+    bad = tmp_path / "bad.json"
+    monkeypatch.setattr(sys, "argv", base + [
+        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
+    assert mod.main() == 1
+    assert json.loads(bad.read_text())["expected_rounds_check"]["failures"]
+
+    # A pair that was never scored is a FAILURE, not a silent skip —
+    # otherwise a sweep missing a method still reports a clean gate.
+    missing = tmp_path / "missing.json"
+    monkeypatch.setattr(sys, "argv", base + [
+        "--expect-rounds", "sfc:2=3", "--out", str(missing)])
+    assert mod.main() == 1
+    fails = json.loads(missing.read_text())["expected_rounds_check"]["failures"]
+    assert any("NOT SCORED" in f for f in fails), fails
+
+
+def test_expect_rounds_refuses_to_pass_vacuously_without_scoring(monkeypatch):
+    """Without --schedule-cost nothing is scored, so the gate would pass on
+    an empty comparison. It must refuse instead."""
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", "geometric", "--expect-rounds", "geometric:2=1",
+        "--out", "/dev/null"])
+    with pytest.raises(SystemExit, match="needs --schedule-cost"):
+        mod.main()
+
+
+@pytest.mark.parametrize("spec", ["geometric:2", "geometric=2", "voodoo:2=3"])
+def test_expect_rounds_rejects_malformed_specs(spec):
+    """A typo'd expectation must raise, never be dropped — a silently
+    skipped expectation turns the gate into a no-op."""
+    with pytest.raises(ValueError, match="expect-rounds"):
+        mod.parse_expect_rounds(spec)
+
+
+def test_expect_rounds_rejects_duplicate_and_empty_specs():
+    """Two more ways the gate could be silently bypassed (codex round 2).
+
+    A duplicate key would let the later expectation overwrite the earlier,
+    so a listed-but-wrong expectation is discarded and the gate passes. A
+    non-empty spec that parses to nothing (``",,,"``) would make main() skip
+    the check entirely while the caller believes it ran.
+    """
+    with pytest.raises(ValueError, match="duplicate expectation"):
+        mod.parse_expect_rounds("geometric:2=999,geometric:2=13")
+    with pytest.raises(ValueError, match="NO expectations"):
+        mod.parse_expect_rounds(",,,")
+    # Whitespace-only is a value the caller PASSED; reading it as "no gate"
+    # is the same bypass (codex round 3 — the first fix guarded on
+    # spec.strip() and let this through).
+    for blank in ("   ", "\t", " , , "):
+        with pytest.raises(ValueError, match="NO expectations"):
+            mod.parse_expect_rounds(blank)
+    # A genuinely empty spec is the "no gate requested" default, not an error.
+    assert mod.parse_expect_rounds("") == {}
+
+
+@pytest.mark.parametrize("flag, value", [
+    ("--lloyd", "-1"), ("--subdivision", "-1"), ("--halo-depth", "-1")])
+def test_negative_numeric_args_refused_as_false_provenance(
+        monkeypatch, flag, value):
+    """All three behave exactly like 0 in the underlying code but would be
+    RECORDED under the negative value — a level-0 mesh filed as 'L-1'."""
+    argv = ["bench", "--subdivision", "2", "--rank-counts", "2",
+            "--methods", "geometric", "--out", "/dev/null"]
+    # Replace the flag if already present, else append.
+    if flag in argv:
+        argv[argv.index(flag) + 1] = value
+    else:
+        argv += [flag, value]
+    monkeypatch.setattr(sys, "argv", argv)
+    with pytest.raises(SystemExit, match=r"must be >= 0"):
+        mod.main()
+
+
+def test_expect_rounds_fails_when_a_method_is_unavailable(
+        tmp_path, monkeypatch):
+    """An expectation naming a method that reported UNAVAILABLE must fail.
+
+    Distinct from omitting the method from --methods: here the sweep asks
+    for it and the partitioner is missing, which is exactly how a two-method
+    table gets misread as a three-method one.
+    """
+    monkeypatch.setattr(mod, "method_available",
+                        lambda m: m != "metis")
+    out = tmp_path / "unavail.json"
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", "geometric,metis", "--schedule-cost",
+        "--expect-rounds", "metis:2=1", "--out", str(out)])
+    assert mod.main() == 1
+    payload = json.loads(out.read_text())
+    fails = payload["expected_rounds_check"]["failures"]
+    assert any("NOT SCORED" in f for f in fails), fails
+    # Assert the EXPLICIT unavailable row, not just the failure: without
+    # this the test would also pass if metis were silently omitted, which
+    # is the very substitution this bench refuses to make.
+    metis_rows = [r for r in payload["rows"] if r["method"] == "metis"]
+    assert metis_rows and metis_rows[0]["available"] is False, payload["rows"]
+
+
+def test_lloyd_rejects_negative(monkeypatch):
+    """A negative Lloyd count behaves like 0 in the builder (it relaxes only
+    for > 0) but is recorded and cached under its own key — false
+    provenance, so the CLI must refuse it rather than run."""
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", "geometric", "--lloyd", "-1", "--out", "/dev/null"])
+    with pytest.raises(SystemExit, match=r"--lloyd must be >= 0"):
+        mod.main()
 M scripts/bench/bench_voronoi_partition_methods.py
 M tests/bench/test_bench_voronoi_partition_methods.py
?? .physics-validator/flag_reachability_audit/
?? .physics-validator/homogeneous_ice_nucleation/
?? .physics-validator/scaling_campaign/codex_consult_hundreds_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r2_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r3_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r4_2026-08-07.md
?? config/aimip/clubb_bechtold/
?? config/aimip/scale/suite_sfno_full_longtrain.yaml
?? config/aimip/scale/suite_sfno_full_maxdata.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed0.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed1.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed2.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed3.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed4.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed5.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed6.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed7.yaml
?? config/aimip/scale/suite_sfno_full_ood_finetune.yaml
?? config/aimip/scale/suite_sfno_full_res_t106.yaml
?? config/aimip/scale/suite_sfno_full_res_t21.yaml
?? config/aimip/scale/suite_sfno_full_res_t42.yaml
?? config/aimip/scale/suite_sfno_full_res_t63.yaml
?? config/aimip/scale/suite_sfno_full_t63_ace2.yaml
?? config/aimip/scale/suite_sfno_full_t63_allyears_ace2.yaml
?? config/aimip/scale/suite_sfno_full_t63_allyears_gc.yaml
?? config/aimip/scale/suite_sfno_full_ucast_dropout.yaml
?? config/aimip/scale/suite_sfno_full_ucast_mae.yaml
?? config/aimip/scale/suite_sfno_full_ucast_muon.yaml
?? config/aimip/scale/suite_sfno_full_ucast_muondrop.yaml
?? config/aimip/scale/suite_sfno_full_ucast_muonlr.yaml
?? config/wb/loss_presets/ucast.yaml
?? docs/dev-notes/ace2_learning_strategy_gap.md
?? docs/dev-notes/ucast_learning_strategy_gap.md
?? drift.txt
?? nat.txt
?? scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
?? scripts/cluster/unified_training/bench_wb2_levante.sbatch
?? scripts/cluster/unified_training/score_ucast_arms.sbatch
?? scripts/cluster/unified_training/score_ucast_probabilistic.sbatch
?? tests/bench/test_mpas_schedule_cost_scan_sbatch.py
?? tests/distributed/test_sfno_full_dp_mpi.py
?? tests/unit/test_ace2_sfno_suites.py
?? tests/unit/test_aimip_eval_train_gate.py
?? tests/unit/test_aimip_generic_scheme_params.py
?? tests/unit/test_apply_params_to_pipeline.py
?? tests/unit/test_ensemble_crps.py
?? tests/unit/test_loss_level_weighting.py
?? tests/unit/test_multiseed_ensemble_cli.py
?? tests/unit/test_muon_partitioned_groups.py
?? tests/unit/test_persistence_normalized_loss.py
?? tests/unit/test_sfno_full_data_parallel.py
?? tests/unit/test_sfno_full_pmap_devices.py
?? tests/unit/test_sfno_pe_rollout_stability.py
?? tests/unit/test_ucast_learning_strategy.py
?? tests/unit/test_wb_climatology_reference.py
?? tests/unit/test_wb_scorecard_sota_units.py
     1	#!/bin/bash -l
     2	#SBATCH --job-name=mpas_sched_cost
     3	#SBATCH --account=bb1596
     4	#SBATCH --partition=shared
     5	#SBATCH --ntasks=1
     6	#SBATCH --cpus-per-task=8
     7	#SBATCH --mem=200G
     8	#SBATCH --time=24:00:00
     9	#SBATCH --output=mpas_schedule_cost_scan.%j.log
    10	# ===========================================================================
    11	# Does RECOLOURING still have room on the MPAS halo schedule?  CPU-only,
    12	# ZERO GPU hours.
    13	#
    14	# Note this is NOT a two-way choice between colouring and ownership: a
    15	# redesigned DIRECTED schedule (see the SCOPE note below) is a third path,
    16	# and it is not bounded by the quantity measured here.  What this run
    17	# settles is only whether recolouring THIS undirected graph is exhausted.
    18	#
    19	# WHY THIS RUN EXISTS
    20	# MPAS GPU is the worst-scaling lane we have: measured/modelled-bound 3.16x
    21	# (s8@16) to 4.47x (s9@64), and one halo fill costs 12-14 SEQUENTIAL ppermute
    22	# rounds.  Both independent reviews (codex + GLM, 2026-08-07) ranked cutting
    23	# that round count as the top structural lever, at 300-800 LOC and 7-14 days
    24	# for a partitioner with a new objective.  That estimate is only worth
    25	# spending if recolouring is genuinely exhausted, and the colourer's own
    26	# lower bound decides it.
    27	#
    28	# The schedule is a proper EDGE colouring of the device communication graph
    29	# (one colour = one ppermute round) and max_degree is that graph's maximum
    30	# vertex degree, so VIZING bounds the chromatic index: Delta <= chi' <=
    31	# Delta + 1.  Read coloring_gap = n_rounds - max_degree through that:
    32	#     gap == 0 -> PROVABLY OPTIMAL; recolouring headroom is exactly zero.
    33	#     gap == 1 -> indistinguishable from optimal (a Class 2 graph really
    34	#                 needs Delta+1, and deciding Class 1/2 is NP-complete);
    35	#                 nothing provable to win.
    36	#     gap >= 2 -> at least gap-1 rounds of genuine recolouring headroom.
    37	# So recolouring is capped at ~1 round out of 12-14 (<= ~7%) wherever the
    38	# multi-start search lands on Delta or Delta+1 — which it does on every
    39	# configuration probed so far.  This run measures whether the production
    40	# working points are in that regime.
    41	#
    42	# (a) NUMBER PRODUCED: n_rounds, max_degree and their gap per
    43	#     (method x n_dev) at the production working points.
    44	# (b) CONFIRMS a cheap fix: gap >= 2 at any production (ppermute) row
    45	#     (recolouring then removes at least gap-1 rounds, guaranteed).
    46	#     gap == 0: recolouring is provably worthless AT THIS OWNERSHIP.
    47	#     gap == 1: INCONCLUSIVE — Class 1 vs Class 2 is NP-complete, so this
    48	#     neither confirms nor refutes a one-round win.
    49	#     Note what a gap of 0 does NOT prove: Delta bounds only a proper
    50	#     UNDIRECTED edge colouring of this graph, and the builder forces BOTH
    51	#     ppermute directions per pair even when one send map is empty, so a
    52	#     redesigned DIRECTED schedule is not bounded by Delta.  "Ownership is
    53	#     the only path" would overclaim; the honest statement is "recolouring
    54	#     this undirected graph is exhausted".
    55	# (c) WHY NOT CHEAPER: this IS the cheap test — no GPU, no MPI, no model
    56	#     step.  It cannot be shrunk further onto small meshes: measured
    57	#     2026-08-07, L2/L4 x {geometric,sfc} x nd 2-16 all report gap == 0,
    58	#     but EVERY one of those rows auto-selects the ALLGATHER strategy
    59	#     (cells/device below the threshold), so production runs no ppermute
    60	#     schedule there and the number is counterfactual.  Only meshes big
    61	#     enough to keep cells/device above the threshold answer the question.
    62	#     s6 lloyd=0 also gave gap == 0 on all three methods — rounds 7/7/6 at
    63	#     np8 and 13/10/10 at np16 for geometric/sfc/metis.  Both are genuinely
    64	#     ppermute (L6 = 40,962 cells; padded, np8 gives 40,968/8 = 5,121 and
    65	#     np16 gives 40,976/16 = 2,561 cells/device, each above the 2,000
    66	#     threshold).  But np8's 7 == n_dev-1 is complete-graph saturation and
    67	#     says nothing; only np16 (13/10/10 < 15) escapes it.  Still a SMALL
    68	#     working point; s8-s10 are the production ones.
    69	#
    70	# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
    71	# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
    72	#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
    73	#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
    74	# Arms 1-2 must REPRODUCE those before arm 3's unknown s10 number is quoted.
    75	# A scan that misses the known answer is a broken instrument, and its s10
    76	# row must not be believed.
    77	#
    78	# lloyd=0 throughout: it is what the reference census used AND the cache key
    79	# the prewarmed s8/s9/s10 meshes were written under.  It is the LABELLED
    80	# synthetic scaling mesh, recorded in every row's metadata so it can never be
    81	# read back as a production SCVT receipt.
    82	#
    83	# Arms run cheapest-first and each writes its own JSON, so a later arm that
    84	# runs out of time or memory cannot lose an earlier arm's result.  s10@128 is
    85	# last and is the one genuinely at risk: the scorer is known not to have
    86	# finished at subdiv-8@128 on a laptop, which is why this asks for 24 h.
    87	#
    88	# SUBMIT (from the repo root):
    89	#   sbatch scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
    90	# ===========================================================================
    91	set -uo pipefail
    92	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    93	export JAX_PLATFORMS=cpu
    94	export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
    95	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    96	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    97	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    98	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    99	
   100	OUT=results/a1/mpas_schedule_cost
   101	mkdir -p "$OUT"
   102	# Overridable ONLY so the exit-status logic below can be regression-tested
   103	# with a stub bench (tests/bench/test_mpas_schedule_cost_scan_sbatch.py).
   104	# A wrong exit code here is the blocker class this gate exists to prevent:
   105	# SLURM would file a scan with no valid result as COMPLETED.
   106	BENCH="${LEGOESM_SCHEDULE_COST_BENCH:-scripts/bench/bench_voronoi_partition_methods.py}"
   107	
   108	# pymetis is present in .venv but was ABSENT from .venv-mpi for a while, and a
   109	# missing partitioner is reported as "unavailable" rather than substituted.
   110	# Say which python and whether metis is really there, so a two-method table
   111	# cannot be misread as a three-method one.
   112	echo "[scan] python=$PY"
   113	"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
   114	  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"
   115	
   116	run_arm () {  # $1=level  $2=rank-counts  $3=label  $4=expect-rounds (may be "")
   117	  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
   118	  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
   119	  "$PY" "$BENCH" \
   120	      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
   121	      --methods geometric,sfc,metis --schedule-cost \
   122	      ${4:+--expect-rounds "$4"} \
   123	      --out "$OUT/schedule_cost_s$1.json"
   124	  local rc=$?   # capture BEFORE anything else runs, or `date` overwrites it
   125	  echo "[scan] arm $3 exit=$rc"
   126	  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
   127	  return $rc
   128	}
   129	
   130	# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
   131	# makes the bench exit non-zero unless it reproduces the census below, and a
   132	# listed pair that never got scored (e.g. pymetis missing) counts as a
   133	# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
   134	# pass — an instrument that misses the known answer cannot be trusted on the
   135	# unknown one.
   136	S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
   137	S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"
   138	
   139	run_arm 8 64,128 "1/3 VALIDATION s8" "$S8_EXPECT"; RC8=$?
   140	run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?
   141	
   142	if [ "$RC8" -ne 0 ] || [ "$RC9" -ne 0 ]; then
   143	  echo "[scan] INSTRUMENT CHECK FAILED (s8 rc=$RC8, s9 rc=$RC9)."
   144	  echo "[scan] The scorer does not reproduce the reference census, so an s10"
   145	  echo "[scan] number from it would be untrustworthy. NOT running arm 3."
   146	  echo "SCAN_ABORTED_VALIDATION"
   147	  exit 1
   148	fi
   149	
   150	# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
   151	run_arm 10 128 "3/3 UNKNOWN s10" ""; RC10=$?
   152	echo "[scan] arm 3 rc=$RC10"
   153	
   154	if [ "$RC10" -ne 0 ]; then
   155	  # Do NOT let SLURM record this as COMPLETED: there is no valid s10 row,
   156	  # and an exit-0 job with a missing result reads as a successful scan.
   157	  echo "SCAN_FAILED_ARM3"
   158	  exit "$RC10"
   159	fi
   160	
   161	echo "SCAN_DONE"
     1	"""Exit-status regression for the MPAS schedule-cost scan launcher.
     2	
     3	The failure this guards is not hypothetical: the first version of the script
     4	captured arm 3's status into ``RC10`` and then ended on a successful ``echo``,
     5	so a scan that produced NO valid s10 result exited 0 and SLURM filed it as
     6	COMPLETED (codex round 2, BLOCKER).  "Tool status is not evidence" cuts both
     7	ways — a launcher that cannot report failure makes every downstream reading
     8	of `sacct` a lie.
     9	
    10	The script is driven with a STUB bench (``LEGOESM_SCHEDULE_COST_BENCH``) whose
    11	exit codes are scripted per arm, so all three paths are exercised without a
    12	mesh, a partitioner, or a compute node.
    13	"""
    14	from __future__ import annotations
    15	
    16	import os
    17	import subprocess
    18	from pathlib import Path
    19	
    20	import pytest
    21	
    22	_REPO = Path(__file__).resolve().parents[2]
    23	_SBATCH = (_REPO / "scripts" / "cluster" / "scaling_levante"
    24	           / "mpas_schedule_cost_scan.sbatch")
    25	
    26	
    27	def _run(tmp_path, codes):
    28	    """Run the launcher with a stub bench that exits ``codes`` per call.
    29	
    30	    ``codes`` is one exit status per invocation, in order (arm 1, arm 2,
    31	    arm 3).  The stub also records the ``--subdivision`` of each call so a
    32	    test can assert an arm did NOT run.
    33	    """
    34	    stub = tmp_path / "stub_bench.py"
    35	    calls = tmp_path / "calls.txt"
    36	    stub.write_text(
    37	        "import sys\n"
    38	        f"codes = {list(codes)!r}\n"
    39	        f"log = {str(calls)!r}\n"
    40	        "sub = sys.argv[sys.argv.index('--subdivision') + 1]\n"
    41	        "with open(log, 'a') as f:\n"
    42	        "    f.write(sub + '\\n')\n"
    43	        "n = sum(1 for _ in open(log))\n"
    44	        "sys.exit(codes[n - 1] if n <= len(codes) else 0)\n"
    45	    )
    46	    env = dict(os.environ)
    47	    env["LEGOESM_SCHEDULE_COST_BENCH"] = str(stub)
    48	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
    49	    env["LEGOESM_REPO"] = str(_REPO)
    50	    # _env.sh resolves PY from here; the stub is plain python, and _env.sh
    51	    # verifies PY can import jax, so use the repo venv.
    52	    env.setdefault("LEGOESM_PYTHON", str(_REPO / ".venv" / "bin" / "python"))
    53	    proc = subprocess.run(
    54	        ["bash", str(_SBATCH)], env=env, cwd=str(_REPO),
    55	        capture_output=True, text=True, timeout=600)
    56	    ran = calls.read_text().split() if calls.exists() else []
    57	    return proc, ran
    58	
    59	
    60	@pytest.mark.skipif(not (_REPO / ".venv" / "bin" / "python").exists(),
    61	                    reason="needs the repo venv for _env.sh's jax check")
    62	def test_validation_failure_aborts_before_the_unknown_arm(tmp_path):
    63	    """Arm 1 failing must abort with non-zero AND must not run arm 3.
    64	
    65	    An instrument that misses the known census cannot be trusted on the
    66	    unknown one, so producing an s10 number anyway would be worse than
    67	    producing none.
    68	    """
    69	    proc, ran = _run(tmp_path, [1, 0, 0])
    70	    assert proc.returncode != 0, proc.stdout[-2000:]
    71	    assert "SCAN_ABORTED_VALIDATION" in proc.stdout
    72	    assert "10" not in ran, f"arm 3 ran despite failed validation: {ran}"
    73	
    74	
    75	@pytest.mark.skipif(not (_REPO / ".venv" / "bin" / "python").exists(),
    76	                    reason="needs the repo venv for _env.sh's jax check")
    77	def test_arm3_failure_is_not_reported_as_success(tmp_path):
    78	    """The original BLOCKER: arm 3 fails, job must NOT exit 0."""
    79	    proc, ran = _run(tmp_path, [0, 0, 3])
    80	    assert proc.returncode == 3, (
    81	        f"arm-3 status not propagated (got {proc.returncode})\n"
    82	        f"{proc.stdout[-2000:]}")
    83	    assert "SCAN_FAILED_ARM3" in proc.stdout
    84	    assert "SCAN_DONE" not in proc.stdout
    85	    assert ran == ["8", "9", "10"], ran
    86	
    87	
    88	@pytest.mark.skipif(not (_REPO / ".venv" / "bin" / "python").exists(),
    89	                    reason="needs the repo venv for _env.sh's jax check")
    90	def test_full_success_exits_zero_and_runs_every_arm(tmp_path):
    91	    proc, ran = _run(tmp_path, [0, 0, 0])
    92	    assert proc.returncode == 0, proc.stdout[-2000:]
    93	    assert "SCAN_DONE" in proc.stdout
    94	    assert ran == ["8", "9", "10"], ran

 succeeded in 1336ms:
tests/bench/test_scaling_metadata.py:199:        extra={"note": "metis"},
tests/bench/test_scaling_metadata.py:202:    assert rec["extra"]["note"] == "metis"
docs/validation/cmip_readiness.md:130:| Voronoi mesh decomposition | Done | `auto` (METIS when `pymetis` present, else RCB) + Hilbert SFC, 2-layer halo, owned-first entity ownership |
docs/architecture/composability.md:44:spectral truncation, number of latitudes, or icosahedral subdivision level
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:10:The script is driven with a STUB bench (``LEGOESM_SCHEDULE_COST_BENCH``) whose
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:24:           / "mpas_schedule_cost_scan.sbatch")
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:31:    arm 3).  The stub also records the ``--subdivision`` of each call so a
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:40:        "sub = sys.argv[sys.argv.index('--subdivision') + 1]\n"
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:47:    env["LEGOESM_SCHEDULE_COST_BENCH"] = str(stub)
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:83:    assert "SCAN_FAILED_ARM3" in proc.stdout
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:84:    assert "SCAN_DONE" not in proc.stdout
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:93:    assert "SCAN_DONE" in proc.stdout
scripts/bench/bench_mpas_spmd_scaling.py:12:  strong: fixed subdivision level, vary n_devices -> speedup = t(1)/t(n).
scripts/bench/bench_mpas_spmd_scaling.py:13:  (Weak scaling rides the subdivision ladder: one level = 4x the cells, so
scripts/bench/bench_mpas_spmd_scaling.py:50:      --subdivision 3 --nlev 4 --n-devices 2 --steps 4 --parity-gate
scripts/bench/bench_mpas_spmd_scaling.py:99:def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
scripts/bench/bench_mpas_spmd_scaling.py:120:    mesh = create_voronoi_mesh(subdivision_level=subdivision,
scripts/bench/bench_mpas_spmd_scaling.py:179:    p.add_argument("--subdivision", type=int, default=5,
scripts/bench/bench_mpas_spmd_scaling.py:180:                   help="icosahedral subdivision level L "
scripts/bench/bench_mpas_spmd_scaling.py:195:                   choices=["auto", "geometric", "metis", "sfc"],
scripts/bench/bench_mpas_spmd_scaling.py:306:        dt = max(600.0 * 4.0 ** (4 - args.subdivision), 30.0)
scripts/bench/bench_mpas_spmd_scaling.py:315:        args.subdivision, args.nlev, reorder_for, nd, args.partition_method,
scripts/bench/bench_mpas_spmd_scaling.py:498:        subdivision=args.subdivision, n_devices=nd,
scripts/bench/bench_mpas_spmd_scaling.py:528:    # twin): resolution = subdivision level, matching run_cpu_mpi_scaling's
scripts/bench/bench_mpas_spmd_scaling.py:532:        resolution=args.subdivision,
scripts/bench/bench_mpas_spmd_scaling.py:545:        resolution=f"L{args.subdivision}",
scripts/bench/bench_mpas_spmd_scaling.py:574:        print(f"[mpas nd={nd} L{args.subdivision} nCells={mesh.nCells} "
docs/performance/scaling/scaling_theoretical_limit_report_2026-06-15.md:105:| MPAS METIS partitioning | +2.6 % np16 (rank-growing, opt-in, pymetis dep) — real but "not a new mechanism" | 8491002 |
scripts/bench/bench_ocean_mpas_scaling.py:9:  strong: fixed subdivision level, vary np  -> speedup = t(1)/t(np).
scripts/bench/bench_ocean_mpas_scaling.py:47:      --mode strong --subdivision 4 --nlev 10 --steps 12
scripts/bench/bench_ocean_mpas_scaling.py:93:#: Icosahedral subdivision levels this lane will consider for weak mode.
scripts/bench/bench_ocean_mpas_scaling.py:171:def build_global_problem(subdivision: int, nlev: int, seed: int = 0,
scripts/bench/bench_ocean_mpas_scaling.py:186:    mesh = create_voronoi_mesh(subdivision_level=subdivision)
scripts/bench/bench_ocean_mpas_scaling.py:307:    p.add_argument("--subdivision", type=int, default=4,
scripts/bench/bench_ocean_mpas_scaling.py:329:                   choices=["auto", "geometric", "metis", "sfc"],
scripts/bench/bench_ocean_mpas_scaling.py:443:    subdivision = args.subdivision
scripts/bench/bench_ocean_mpas_scaling.py:445:        subdivision = weak_level_for(args.cells_per_rank, n_ranks)
scripts/bench/bench_ocean_mpas_scaling.py:519:        subdivision, args.nlev, barotropic_solver=args.barotropic_solver,
scripts/bench/bench_ocean_mpas_scaling.py:794:        mode=args.mode, subdivision=subdivision, n_ranks=n_ranks,
scripts/bench/bench_ocean_mpas_scaling.py:836:        resolution=f"L{subdivision}",
scripts/bench/bench_ocean_mpas_scaling.py:895:        print(f"[mpas-ocean np={n_ranks} L{subdivision} "
scripts/bench/aggregate_bcw_scaling.py:67:        n_cells = 10.0 * 4.0 ** r + 2.0  # SCVT cell count at subdivision r
scripts/bench/aggregate_scaling_results.py:74:                # discrete 4x subdivision levels — so normalize by the
scripts/bench/bench_halo_exchange.py:193:                        help="Voronoi subdivision level")
tests/bench/test_bench_ocean_mpas_scaling.py:179:        "bench", "--subdivision", "2", "--nlev", "3", "--steps", "3",
tests/bench/test_bench_ocean_mpas_scaling.py:236:        "bench", "--subdivision", "2", "--nlev", "3", "--steps", "2",
tests/bench/test_bench_voronoi_partition_methods.py:22:    return create_voronoi_mesh(subdivision_level=level)
tests/bench/test_bench_voronoi_partition_methods.py:54:    # geometric/sfc are dependency-free; metis truthfully reports.
tests/bench/test_bench_voronoi_partition_methods.py:58:        import pymetis  # noqa: F401
tests/bench/test_bench_voronoi_partition_methods.py:60:        assert mod.method_available("metis") is True
tests/bench/test_bench_voronoi_partition_methods.py:62:        assert mod.method_available("metis") is False
tests/bench/test_bench_voronoi_partition_methods.py:68:        "bench", "--subdivision", "2", "--rank-counts", "2,4",
tests/bench/test_bench_voronoi_partition_methods.py:75:    assert payload["auto_resolves_to"] in ("metis", "geometric")
tests/bench/test_bench_voronoi_partition_methods.py:85:    monkeypatch.setattr(sys, "argv", ["bench", "--rank-counts", "1"])
tests/bench/test_bench_voronoi_partition_methods.py:234:    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
tests/bench/test_bench_voronoi_partition_methods.py:263:        "bench", "--subdivision", "2", "--rank-counts", "2",
tests/bench/test_bench_voronoi_partition_methods.py:272:        "bench", "--subdivision", "2", "--rank-counts", "2",
tests/bench/test_bench_voronoi_partition_methods.py:307:        "bench", "--subdivision", "2", "--rank-counts", "2",
tests/bench/test_bench_voronoi_partition_methods.py:321:    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
tests/bench/test_bench_voronoi_partition_methods.py:333:        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
tests/bench/test_bench_voronoi_partition_methods.py:340:        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
tests/bench/test_bench_voronoi_partition_methods.py:348:        "--expect-rounds", "sfc:2=3", "--out", str(missing)])
tests/bench/test_bench_voronoi_partition_methods.py:358:        "bench", "--subdivision", "2", "--rank-counts", "2",
tests/bench/test_bench_voronoi_partition_methods.py:359:        "--methods", "geometric", "--expect-rounds", "geometric:2=1",
tests/bench/test_bench_voronoi_partition_methods.py:369:    with pytest.raises(ValueError, match="expect-rounds"):
tests/bench/test_bench_voronoi_partition_methods.py:396:    ("--lloyd", "-1"), ("--subdivision", "-1"), ("--halo-depth", "-1")])
tests/bench/test_bench_voronoi_partition_methods.py:401:    argv = ["bench", "--subdivision", "2", "--rank-counts", "2",
tests/bench/test_bench_voronoi_partition_methods.py:422:                        lambda m: m != "metis")
tests/bench/test_bench_voronoi_partition_methods.py:425:        "bench", "--subdivision", "2", "--rank-counts", "2",
tests/bench/test_bench_voronoi_partition_methods.py:426:        "--methods", "geometric,metis", "--schedule-cost",
tests/bench/test_bench_voronoi_partition_methods.py:427:        "--expect-rounds", "metis:2=1", "--out", str(out)])
tests/bench/test_bench_voronoi_partition_methods.py:433:    # this the test would also pass if metis were silently omitted, which
tests/bench/test_bench_voronoi_partition_methods.py:435:    metis_rows = [r for r in payload["rows"] if r["method"] == "metis"]
tests/bench/test_bench_voronoi_partition_methods.py:436:    assert metis_rows and metis_rows[0]["available"] is False, payload["rows"]
tests/bench/test_bench_voronoi_partition_methods.py:444:        "bench", "--subdivision", "2", "--rank-counts", "2",
tests/bench/test_bench_mpas_spmd_gates.py:76:    def _spy(subdivision_level, **kwargs):
tests/bench/test_bench_mpas_spmd_gates.py:77:        seen["level"] = subdivision_level
tests/bench/test_bench_mpas_spmd_gates.py:107:        create_voronoi_mesh(subdivision_level=3), 2, method="sfc")
tests/bench/test_bench_mpas_spmd_gates.py:139:            "--subdivision", "3", "--nlev", "4",
tests/bench/test_bench_mpas_spmd_gates.py:174:            "--subdivision", "3", "--nlev", "4",
tests/bench/test_bench_mpas_spmd_gates.py:197:            "--n-devices", "2", "--subdivision", "3", "--nlev", "4",
scripts/bench/run_levante_gpu_scaling.py:363:# Icosahedral weak scaling: subdivision level as base.
scripts/bench/run_levante_gpu_scaling.py:377:    """Compute icosahedral subdivision level for weak scaling.
scripts/bench/run_levante_gpu_scaling.py:381:    subdivision level whose cells-per-GPU ratio is nearest to ``base_cells``
scripts/bench/run_levante_gpu_scaling.py:860:        # Voronoi SCVT: n_grid is subdivision level, nCells = 10*4^level + 2
scripts/bench/run_levante_gpu_scaling.py:890:        # For icosahedral, n_grid is a subdivision level.  Scale the
scripts/bench/run_levante_gpu_scaling.py:1518:        grid = create_voronoi_mesh(subdivision_level=n_grid)
scripts/bench/run_levante_gpu_scaling.py:2082:    # For icosahedral grids the subdivision level jumps by 4× in cell
scripts/bench/bench_voronoi_partition_methods.py:51:   NOTE the halo depth differs by lane and is NOT ``--halo-depth``: that
scripts/bench/bench_voronoi_partition_methods.py:65:Guards: methods that are unavailable (``metis`` without ``pymetis``) are
scripts/bench/bench_voronoi_partition_methods.py:73:      --subdivision 6 --rank-counts 2,4,8,16 --out results/partition_quality.json
scripts/bench/bench_voronoi_partition_methods.py:77:      --subdivision 9 --rank-counts 64,128 --schedule-cost \
scripts/bench/bench_voronoi_partition_methods.py:94:METHODS = ("geometric", "sfc", "metis")
scripts/bench/bench_voronoi_partition_methods.py:98:    if method != "metis":
scripts/bench/bench_voronoi_partition_methods.py:101:        import pymetis  # noqa: F401
scripts/bench/bench_voronoi_partition_methods.py:170:        partition_cells_metis,
scripts/bench/bench_voronoi_partition_methods.py:178:    if method == "metis":
scripts/bench/bench_voronoi_partition_methods.py:179:        return np.asarray(partition_cells_metis(mesh, n_ranks))
scripts/bench/bench_voronoi_partition_methods.py:185:    """Parse ``'sfc:64=12,metis:128=19'`` into ``{("sfc", 64): 12}``.
scripts/bench/bench_voronoi_partition_methods.py:210:                f"--expect-rounds: cannot parse {item!r}; expected "
scripts/bench/bench_voronoi_partition_methods.py:214:                f"--expect-rounds: unknown method {key[0]!r} in {item!r}; "
scripts/bench/bench_voronoi_partition_methods.py:218:                f"--expect-rounds: duplicate expectation for "
scripts/bench/bench_voronoi_partition_methods.py:229:            f"--expect-rounds={spec!r} parses to NO expectations; the gate "
scripts/bench/bench_voronoi_partition_methods.py:238:    a sweep that silently dropped a method (unavailable ``pymetis``) or a
scripts/bench/bench_voronoi_partition_methods.py:267:    LEFT AT THE SCORER'S DEFAULT: this script's ``--halo-depth`` is the MPI
scripts/bench/bench_voronoi_partition_methods.py:332:    p.add_argument("--subdivision", type=int, default=5,
scripts/bench/bench_voronoi_partition_methods.py:334:    p.add_argument("--rank-counts", type=str, default="2,4,8,16")
scripts/bench/bench_voronoi_partition_methods.py:335:    p.add_argument("--halo-depth", type=int, default=2,
scripts/bench/bench_voronoi_partition_methods.py:342:                        "--halo-depth. Expensive: minutes per candidate at "
scripts/bench/bench_voronoi_partition_methods.py:354:    p.add_argument("--expect-rounds", type=str, default="",
scripts/bench/bench_voronoi_partition_methods.py:372:        raise SystemExit("--rank-counts needs integers >= 2")
scripts/bench/bench_voronoi_partition_methods.py:382:    if args.subdivision < 0:
scripts/bench/bench_voronoi_partition_methods.py:384:            f"--subdivision must be >= 0, got {args.subdivision}: the mesh "
scripts/bench/bench_voronoi_partition_methods.py:387:            f"L{args.subdivision}.")
scripts/bench/bench_voronoi_partition_methods.py:390:            f"--halo-depth must be >= 0, got {args.halo_depth}: the halo "
scripts/bench/bench_voronoi_partition_methods.py:396:            "--expect-rounds compares scored round counts, so it needs "
scripts/bench/bench_voronoi_partition_methods.py:404:    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
scripts/bench/bench_voronoi_partition_methods.py:408:            f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
scripts/bench/bench_voronoi_partition_methods.py:410:    print(f"mesh L{args.subdivision}: {int(mesh.nCells)} cells, "
scripts/bench/bench_voronoi_partition_methods.py:417:            print(f"  {method:9s}: UNAVAILABLE (pymetis not importable) — "
scripts/bench/bench_voronoi_partition_methods.py:464:            resolution=f"L{args.subdivision}",
scripts/bench/profile_mpas_ocean.py:4:``scripts/bench/bench_ocean_gpu_scaling.py`` does (subdivision_level=6,
scripts/bench/profile_mpas_ocean.py:129:    subdivision_level = 6
scripts/bench/profile_mpas_ocean.py:140:        f"profile_mpas_ocean: I{subdivision_level}/L{n_levels}, "
scripts/bench/profile_mpas_ocean.py:150:        subdivision_level=subdivision_level,
scripts/bench/run_cpu_mpi_scaling.py:322:WEAK_BASE_ICO = 4  # subdivision level
scripts/bench/run_cpu_mpi_scaling.py:356:    needs for its discrete 4x subdivision-level jumps (see
scripts/bench/run_cpu_mpi_scaling.py:1015:            mesh = create_voronoi_mesh(subdivision_level=resolution)  # build + cache
scripts/bench/run_cpu_mpi_scaling.py:1018:            mesh = create_voronoi_mesh(subdivision_level=resolution)  # load from cache
scripts/bench/run_cpu_mpi_scaling.py:1020:        mesh = create_voronoi_mesh(subdivision_level=resolution)
scripts/bench/run_cpu_mpi_scaling.py:1064:        # "geometric" (RCB, default) | "metis" (pymetis k-way edge-cut min).
scripts/bench/run_levante_gpu_scaling.sh:56:STRONG_RESOLUTIONS="4,5,6"     # icosahedral subdivision levels
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md:67:ACROSS FIELDS: `jnp.stack` the same-halo-depth fields of a stage → ONE
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md:98:`pymetis` is present else RCB; `method="sfc"` gives the dependency-free Hilbert
docs/performance/scaling/scaling_indicators.csv:43:2026-06-15,2786274b,voronoi_metis,atm_icosahedral,mpi,speedup,1.019,voronoi_lever_speedup,8488136,METIS vs RCB partition (audit #3) — I6 f64 np16 1.019x (59.93 vs 58.82); np8 neutral; edge-cut-min win GROWS with rank; OPT-IN (needs pymetis), RCB default fine on near-uniform mesh
docs/performance/scaling/scaling_indicators.csv:55:2026-06-15,da443723,metis_np16_ico,atm_ico,mpi,strong,1.026,metis_speedup,8491002,METIS (edge-cut min) vs RCB voronoi partition I6 np16 4-node: RCB 38.52 vs METIS 37.55 ms/step = +2.6% (consistent with prior +1.9%; cut-reduction win grows with rank). np32 RCB 34.35-36.64 (node variance); np32 metis leg slow-compiling. Closes codex's last named measurable A/B; OPT-IN (pymetis not in prod venv), not a new lever
docs/performance/scaling/SCALING_SUMMARY.md:91:| atmosphere| **icosahedral** | ✅ `auto` domain decomp (METIS graph-partition when `pymetis` present, else RCB; `sfc` Hilbert option) | 1.24× @2 ranks, plateaus @4 — sync-barrier/jitter-bound on a shared node; correct vs serial to ~1e-9 |
docs/performance/scaling/levante_campaign_2026-07-24.md:118:  np4/np8 = 17.22/7.00 ms (sfc), 17.20/6.97 (metis), 19.35/6.26
docs/performance/scaling/levante_campaign_2026-07-24.md:290:campaign. Efficiency t1/(n*tn) by subdivision:
docs/performance/scaling/levante_campaign_2026-07-24.md:318:size or ico subdivision also changes the global problem, so tile size is
docs/performance/scaling/levante_campaign_2026-07-24.md:336:held constant cleanly across that sweep's subdivision steps, so no weak
docs/performance/scaling/levante_campaign_2026-07-24.md:950:    ValueError: subdivision_level=9 would create 2.62e+06 cells.
docs/performance/scaling/levante_campaign_2026-07-24.md:1262:  sfc is actively harmful at scale. pymetis is absent from `.venv-mpi`,
docs/performance/scaling/levante_campaign_2026-07-24.md:1601:meaning when pymetis appeared in `.venv-mpi` on 2026-07-31), f64,
docs/performance/scaling/levante_campaign_2026-07-24.md:1604:5,120–5,121 at BOTH scales; metis 5,100–5,145 @32 and 5,093–5,144 @128
docs/performance/scaling/levante_campaign_2026-07-24.md:1605:(±0.5 %). WET load is looser still under metis — per-rank wet
docs/performance/scaling/levante_campaign_2026-07-24.md:1606:cell-levels min/max: geometric@32 100,740–102,420; metis@32
docs/performance/scaling/levante_campaign_2026-07-24.md:1607:96,800–102,720; metis@128 **78,000–102,880** (one rank 24 % under the
docs/performance/scaling/levante_campaign_2026-07-24.md:1615:| C | s7 np32 metis, block:cyclic | 191.99 |
docs/performance/scaling/levante_campaign_2026-07-24.md:1616:| D | s8 np128 metis, block:cyclic | 333.39 |
docs/performance/scaling/levante_campaign_2026-07-24.md:1617:| E | s8 np128 metis, block:block | 537.91 |
docs/performance/scaling/levante_campaign_2026-07-24.md:1623:  rank-count isolate): geometric 1.628, metis 1.736. The offline-quality
docs/performance/scaling/levante_campaign_2026-07-24.md:1624:  -> step-time inference FAILS on this lane; part of metis's loss is
docs/performance/scaling/levante_campaign_2026-07-24.md:1790:  decay with parts; the metis receipt argues against pure
docs/science/legoesm_scientific_guide.tex:3398:optional \texttt{pymetis} dependency, \texttt{pip install -e ".[mesh]"}) when
docs/science/specification.html:127:<p><strong>Voronoi mesh domain decomposition</strong> (<code>parallel/voronoi_partition.py</code>): Full domain decomposition for unstructured MPAS/Voronoi meshes. The partitioner is selected by a capability-aware <code>method="auto"</code> default: METIS k-way graph partitioning (via the optional <code>pymetis</code> dependency — the <code>[mesh]</code> extra) when available, else the dependency-free Recursive Coordinate Bisection (RCB) geometric partitioner using 3D cell-center coordinates on the unit sphere. Graph partitioning minimizes the edge cut → better load balance and smaller halos on variable-resolution meshes; <code>auto</code> is byte-identical to RCB where <code>pymetis</code> is absent. A third <code>method="sfc"</code> orders cells along a Hilbert space-filling curve into balanced contiguous chunks (dependency-free, locality-preserving), and <code>reorder_voronoi_for_sharding()</code> Hilbert-orders cells within each shard for <code>NamedSharding</code> locality. Local arrays use owned-first indexing (owned cells before halo). 2-layer halo depth for biharmonic (del4) stencils. Entity ownership rules: edges owned by rank of <code>min(cellsOnEdge)</code>, vertices by <code>min(cellsOnVertex)</code>. Communication schedules with send/recv lists sorted by global index for deterministic MPI matching. See §6.2.3.</p>
docs/science/specification.html:2625:<p><strong>Partitioning strategies</strong> (<code>method="auto"</code> default via <code>resolve_partition_method()</code>: METIS when <code>pymetis</code> importable, else RCB):
docs/science/specification.html:2627:- <code>partition_cells_metis()</code> — k-way graph partitioning via pymetis (<code>[mesh]</code> extra). Minimizes edge cut → better load balance + smaller halos on variable-resolution meshes.
scripts/bench/bench_ocean_gpu_scaling.py:102:    mesh = create_voronoi_mesh(subdivision_level=level, lloyd_iterations=5)
docs/performance/scaling/scaling_levers_audit_2026-06-14.md:28:   (METIS when `pymetis` present, else RCB) + new Hilbert `method="sfc"`; see the
scripts/bench/bcw_scaling_ledger.py:97:    icosahedral subdivision jumps cell count 4x per level, so cells/rank is NOT
tests/test_mpas_conservation.py:139:    mesh = create_voronoi_mesh(subdivision_level=3)
tests/test_mpas_conservation.py:205:    mesh = create_voronoi_mesh(subdivision_level=2)
tests/test_mpas_conservation.py:323:    mesh = create_voronoi_mesh(subdivision_level=2)
docs/performance/scaling/fig_scaling_caption.md:29:subdivision-7 on CPU–MPI to 64 ranks; **e**, ocean lat–lon 576×1152 L20,
docs/performance/scaling/fig_scaling_caption.md:32:grid; **g**, MPAS Voronoi ocean, subdivision-7, CPU–MPI.
tests/distributed/test_voronoi_mpi.py:84:    return create_voronoi_mesh(subdivision_level=SUBDIVISION_LEVEL, lloyd_iterations=5)
docs/science/specs/SPECIFICATION.md:99:1. **Voronoi mesh domain decomposition** (`parallel/voronoi_partition.py`): Full domain decomposition for unstructured MPAS/Voronoi meshes. The partitioner is selected by a capability-aware `method="auto"` default: METIS k-way graph partitioning (via the optional `pymetis` dependency — the `[mesh]` extra) when available, else the dependency-free Recursive Coordinate Bisection (RCB) geometric partitioner using 3D cell-center coordinates on the unit sphere. Graph partitioning minimizes the edge cut → better load balance and smaller halos on variable-resolution meshes; `auto` is byte-identical to RCB where `pymetis` is absent. A third `method="sfc"` orders cells along a Hilbert space-filling curve into balanced contiguous chunks (dependency-free, locality-preserving), and `reorder_voronoi_for_sharding()` Hilbert-orders cells within each shard for `NamedSharding` locality. Local arrays use owned-first indexing (owned cells before halo). 2-layer halo depth for biharmonic (del4) stencils. Entity ownership rules: edges owned by rank of `min(cellsOnEdge)`, vertices by `min(cellsOnVertex)`. Communication schedules with send/recv lists sorted by global index for deterministic MPI matching. See §6.2.3.
docs/science/specs/SPECIFICATION.md:1823:**Partitioning strategies** (`method="auto"` default via `resolve_partition_method()`: METIS when `pymetis` importable, else RCB):
docs/science/specs/SPECIFICATION.md:1825:- `partition_cells_metis()` — k-way graph partitioning via pymetis (`[mesh]` extra). Minimizes edge cut → better load balance + smaller halos on variable-resolution meshes.
tests/ocean/unit/test_mle_mpas.py:36:    return create_voronoi_mesh(subdivision_level=3)
docs/performance/scaling/scaling.md:1467:  and ``partition_cells_metis`` were imported inline at two call
tests/test_atmosphere_cross_grid_plots.py:1876:        mesh = create_voronoi_mesh(subdivision_level=1)
tests/test_atmosphere_cross_grid_plots.py:1946:        mesh = create_voronoi_mesh(subdivision_level=1)
tests/test_atmosphere_cross_grid_plots.py:1995:        mesh = create_voronoi_mesh(subdivision_level=1)
docs/performance/scaling/scaling_levers_audit_2026-06-15.md:38:   `pymetis` is importable, else falls back to geometric RCB. The win is now
docs/performance/scaling/scaling_levers_audit_2026-06-15.md:39:   realized automatically wherever pymetis is present (the `[mesh]` extra) with NO
tests/distributed/test_voronoi_halo.py:49:    return create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/distributed/test_voronoi_batched_halo.py:60:    return create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/distributed/test_mpi_differentiability.py:346:        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
docs/dev-notes/implementation_summary.md:204:- Capability-aware `method="auto"` default: METIS graph partitioner when `pymetis` (the `[mesh]` extra) is present, else Recursive Coordinate Bisection (RCB)
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:49:      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:9:#SBATCH --output=mpas_schedule_cost_scan.%j.log
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:63:#     np8 and 13/10/10 at np16 for geometric/sfc/metis.  Both are genuinely
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:72:#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:73:#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:89:#   sbatch scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:103:# with a stub bench (tests/bench/test_mpas_schedule_cost_scan_sbatch.py).
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:106:BENCH="${LEGOESM_SCHEDULE_COST_BENCH:-scripts/bench/bench_voronoi_partition_methods.py}"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:108:# pymetis is present in .venv but was ABSENT from .venv-mpi for a while, and a
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:110:# Say which python and whether metis is really there, so a two-method table
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:113:"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:114:  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:116:run_arm () {  # $1=level  $2=rank-counts  $3=label  $4=expect-rounds (may be "")
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:120:      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:121:      --methods geometric,sfc,metis --schedule-cost \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:122:      ${4:+--expect-rounds "$4"} \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:130:# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:132:# listed pair that never got scored (e.g. pymetis missing) counts as a
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:136:S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:137:S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:157:  echo "SCAN_FAILED_ARM3"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:161:echo "SCAN_DONE"
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:36:    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
docs/dev-notes/GPU_SCALING_BRANCH.md:301:63e75704 tests: extend voronoi sharded equivalence to subdivision_level=5
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:209:        --subdivision 4 --nlev 8 --steps 4 --warmup 1 \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:216:        --subdivision "$ICO_LEVEL" --nlev "$NLEV" \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:71:      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
tests/unit/test_voronoi_big_mesh_policy.py:5:hard cap. All run at tiny subdivision levels by lowering the policy
tests/ocean/unit/test_mpas_tke.py:12:``subdivision_level=1`` Voronoi mesh (42 cells / 120 edges):
tests/ocean/unit/test_mpas_tke.py:68:    return create_voronoi_mesh(subdivision_level=1)
tests/ocean/unit/test_gm_redi_mpas_seafloor_slope.py:52:    return create_voronoi_mesh(subdivision_level=2)
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:36:    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch:31:    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_bound_base.sbatch:31:    --n-devices 1 --subdivision 6 --nlev 26 --steps 12 --warmup 3 \
docs/ocean/experiments/mpas_etopo_progression_findings.md:591:  --subdivision 5 --cases a,b,c,d,e,f,f2,g
docs/ocean/experiments/mpas_etopo_progression_findings.md:596:  --subdivision 5 --cases h,i \
docs/ocean/experiments/density_jacobian_pgf_mpas.md:1010:Per the §8f recommendation we ran Option D — ico-5 (subdivision_level=5,
docs/ocean/experiments/density_jacobian_pgf_mpas.md:1017:* mesh: subdivision_level=5 (10242 cells, ~240 km, ~half ico-4 dx)
docs/ocean/experiments/density_jacobian_pgf_mpas.md:1878:ico-5 (subdivision_level=5, 10242 cells, ~240 km, dt=150s,
docs/ocean/experiments/realistic_geometry_mpas_plan.md:367:    production ``--subdivision 7 --years 5`` for the 5-yr headline.
tests/atmosphere/test_kessler_forcing_mpas.py:130:    mesh = create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_partial_cells_mpas.py:38:    return create_voronoi_mesh(subdivision_level=2)
tests/parallel/test_spmd_schedule_cost.py:18:    return create_voronoi_mesh(subdivision_level=4)
tests/parallel/test_spmd_schedule_cost.py:103:def test_sfc_beats_metis_and_geometric_on_rounds():
tests/parallel/test_spmd_schedule_cost.py:105:    production sizes: subdiv-8 sfc 12/14 rounds at 64/128 devices vs metis
tests/parallel/test_spmd_schedule_cost.py:113:    pytest.importorskip("pymetis", reason="metis arm needs pymetis")
tests/parallel/test_spmd_schedule_cost.py:114:    big = create_voronoi_mesh(subdivision_level=5)
tests/parallel/test_spmd_schedule_cost.py:116:         for m in ("sfc", "metis", "geometric")}
tests/parallel/test_spmd_schedule_cost.py:120:    assert r["sfc"] <= r["metis"] and r["sfc"] <= r["geometric"], r
tests/parallel/test_ppermute_edge_coloring.py:112:    mesh = create_voronoi_mesh(subdivision_level=3)
tests/ocean/unit/test_mpas_physics.py:42:    return create_voronoi_mesh(subdivision_level=2)
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:81:ICO_LEVEL="${ICO_LEVEL:-7}"         # lane E subdivision (L7 = 163842 cells
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:230:            --subdivision 4 --nlev 8 --steps 4 --warmup 1 \
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:237:            --subdivision "$ICO_LEVEL" --nlev "$NLEV" \
tests/parallel/test_mpas_partitionlocal_build.py:51:    mesh = create_voronoi_mesh(subdivision_level=SUBDIVISION)
tests/parallel/test_voronoi_schedule_symmetry.py:68:# (subdivision_level, n_ranks): include a fine-partition case (small cells/rank)
tests/parallel/test_voronoi_schedule_symmetry.py:73:    mesh = create_voronoi_mesh(subdivision_level=level)
tests/parallel/test_voronoi_sharded_equivalence.py:59:             n_steps: int = 5, subdivision_level: int = 4,
tests/parallel/test_voronoi_sharded_equivalence.py:86:        # Lower dt at higher subdivision to stay CFL-stable (dx scales
tests/parallel/test_voronoi_sharded_equivalence.py:88:        dt = 600.0 if subdivision_level == 4 else 200.0
tests/parallel/test_voronoi_sharded_equivalence.py:90:        mesh = create_voronoi_mesh(subdivision_level=subdivision_level)
tests/parallel/test_voronoi_sharded_equivalence.py:161:        self._run_equivalence(devices=devices, subdivision_level=4)
tests/parallel/test_voronoi_sharded_equivalence.py:165:        (subdivision_level=5, 10242 cells).  Same envelope as subdiv=4.
tests/parallel/test_voronoi_sharded_equivalence.py:167:        self._run_equivalence(devices=2, subdivision_level=5)
tests/parallel/test_voronoi_sharded_equivalence.py:192:        common = dict(reorder_for=2, n_steps=2, subdivision_level=4)
tests/parallel/test_voronoi_sharded_equivalence.py:236:    def _run_equivalence(self, *, devices: int, subdivision_level: int):
tests/parallel/test_voronoi_sharded_equivalence.py:258:            subdivision_level=subdivision_level,
tests/parallel/test_voronoi_sharded_equivalence.py:262:            subdivision_level=subdivision_level,
tests/unit/test_cdgrid_fv3_regression.py:1350:        """Iter-128 (Priority 3): guard the halo-depth invariant that
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:52:            "--subdivision", "3", "--nlev", "4",
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:54:            "--partition-method", "sfc",  # deterministic, no pymetis dep
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:107:            "--subdivision", "3", "--nlev", "4",
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:109:            "--partition-method", "sfc",  # deterministic, no pymetis dep
tests/parallel/test_scaling_operators.py:256:        mesh = create_voronoi_mesh(subdivision_level=2)
tests/parallel/test_scaling_operators.py:396:        mesh = create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_omip_prognostic_ice.py:151:    return create_voronoi_mesh(subdivision_level=2)
tests/parallel/test_tiled_mass_divergence.py:97:    # not algorithmic.  Tight enough to catch a halo-depth / index / upwind /
tests/parallel/test_mpas_atm_native_step.py:55:_DT = 200.0  # CFL-stable at subdivision 3 (test_voronoi_sharded_equivalence)
tests/parallel/test_mpas_atm_native_step.py:87:    mesh = create_voronoi_mesh(subdivision_level=_SUBDIV)
tests/parallel/test_mpas_atm_native_step.py:557:        mesh = create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_freshwater.py:44:    return create_voronoi_mesh(subdivision_level=2)
tests/unit/test_pr_d_fixes.py:35:    mesh = create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_backscatter.py:57:    return create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_vmix_mpas_integration.py:5:factories and the two private kernels directly on a small ``subdivision_level=1``
tests/ocean/unit/test_vmix_mpas_integration.py:50:    return create_voronoi_mesh(subdivision_level=1)
tests/unit/test_sea_ice_new_physics.py:1085:        return create_voronoi_mesh(subdivision_level=level, lloyd_iterations=5)
tests/ocean/unit/test_bathymetry.py:710:    return create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_mpas_ocean.py:59:    return create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_smagorinsky.py:61:    return create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_ocean_differentiability.py:31:        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=10)
tests/ocean/unit/test_ocean_differentiability.py:515:        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=10)
tests/unit/test_conservative_regrid_unstructured.py:32:        cls.coarse = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)  # 162
tests/unit/test_conservative_regrid_unstructured.py:33:        cls.fine = create_voronoi_mesh(subdivision_level=3, lloyd_iterations=5)    # 642
tests/ocean/unit/test_barotropic_implicit_mpas.py:54:    return create_voronoi_mesh(subdivision_level=2)
tests/unit/test_voronoi_partition_method.py:4:runs prefer graph partitioning (METIS) when ``pymetis`` is available — the MPAS
tests/unit/test_voronoi_partition_method.py:26:    partition_cells_metis,
tests/unit/test_voronoi_partition_method.py:39:    return create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
tests/unit/test_voronoi_partition_method.py:58:        assert resolve_partition_method("metis") == "metis"
tests/unit/test_voronoi_partition_method.py:64:    def test_auto_prefers_metis_when_available(self, monkeypatch):
tests/unit/test_voronoi_partition_method.py:65:        monkeypatch.setattr(vp, "_metis_available", lambda: True)
tests/unit/test_voronoi_partition_method.py:66:        assert resolve_partition_method("auto") == "metis"
tests/unit/test_voronoi_partition_method.py:68:    def test_auto_falls_back_without_metis(self, monkeypatch):
tests/unit/test_voronoi_partition_method.py:69:        monkeypatch.setattr(vp, "_metis_available", lambda: False)
tests/unit/test_voronoi_partition_method.py:72:    def test_metis_available_matches_importlib(self):
tests/unit/test_voronoi_partition_method.py:74:        assert vp._metis_available() == (importlib.util.find_spec("pymetis") is not None)
tests/unit/test_voronoi_partition_method.py:91:    def test_metis_owner_valid_if_available(self, mesh):
tests/unit/test_voronoi_partition_method.py:92:        pytest.importorskip("pymetis")
tests/unit/test_voronoi_partition_method.py:93:        owner = partition_cells_metis(mesh, N_RANKS)
tests/unit/test_voronoi_partition_method.py:180:    def test_partition_auto_equals_geometric_without_metis(self, mesh, monkeypatch):
tests/unit/test_voronoi_partition_method.py:181:        monkeypatch.setattr(vp, "_metis_available", lambda: False)
tests/unit/test_voronoi_partition_method.py:188:    def test_reorder_auto_runs_without_metis(self, mesh, monkeypatch):
tests/unit/test_voronoi_partition_method.py:189:        monkeypatch.setattr(vp, "_metis_available", lambda: False)
tests/unit/test_voronoi_partition_method.py:244:def test_sharding_reorder_auto_is_sfc_not_metis(monkeypatch):
tests/unit/test_voronoi_partition_method.py:247:    The global ``resolve_partition_method`` policy prefers METIS when pymetis
tests/unit/test_voronoi_partition_method.py:251:    those objectives move oppositely -- subdiv-8 @128: sfc 14 rounds, metis 19
tests/unit/test_voronoi_partition_method.py:262:    if vp.resolve_partition_method("auto") != "metis":
tests/unit/test_voronoi_partition_method.py:263:        pytest.skip("global auto policy is not METIS here (pymetis absent) — "
tests/unit/test_voronoi_partition_method.py:267:    for name in ("partition_cells_sfc", "partition_cells_metis",
tests/unit/test_voronoi_partition_method.py:277:    vp.reorder_voronoi_for_sharding(create_voronoi_mesh(subdivision_level=3), 4)
tests/unit/test_voronoi_partition_method.py:287:    pins metis/geometric (or a future census that prefers one) still gets it."""
tests/unit/test_voronoi_partition_method.py:291:    names = ("partition_cells_sfc", "partition_cells_metis",
tests/unit/test_voronoi_partition_method.py:296:    mesh = create_voronoi_mesh(subdivision_level=3)
tests/unit/test_voronoi_partition_method.py:298:    for method, expect in (("metis", "partition_cells_metis"),
tests/ocean/unit/test_bbl_drag.py:217:    mesh = create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_freesurface_helmholtz_adjoint_mpas.py:48:    mesh = create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_freesurface_helmholtz_adjoint_mpas.py:227:    mesh = create_voronoi_mesh(subdivision_level=2)
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:73:    # Smallest viable mesh: subdivision_level=1 → 42 cells.  Level 2
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:76:    return create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
tests/ocean/unit/test_advection_grad_underflow.py:176:        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/ocean/unit/test_leith.py:66:    return create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_mpas_partial_cell_helpers.py:47:    return create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_gm_resolution_function.py:573:    mesh = create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_static_rho_ref_mpas.py:57:    mesh = create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_pgf_ahh08_phase2_mpas.py:47:    mesh = create_voronoi_mesh(subdivision_level=sub)
tests/ocean/unit/test_pgf_ahh08_phase2_mpas.py:67:    mesh = create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_kpp_mpas_conservation.py:99:    return create_voronoi_mesh(subdivision_level=2)
tests/unit/test_sea_ice_dynamics.py:3064:        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/unit/test_sea_ice_dynamics.py:3079:        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/ocean/unit/test_sponge_gamma_3d.py:219:    mesh = create_voronoi_mesh(subdivision_level=1)
tests/ocean/unit/test_pgf_smc03_mpas.py:63:    return create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_gm_redi_mpas.py:44:    return create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_gm_redi_mpas.py:573:    mesh_local = create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_gm_redi_mpas.py:632:    mesh_local = create_voronoi_mesh(subdivision_level=2)
tests/ocean/unit/test_cross_grid_parity.py:65:    return create_voronoi_mesh(subdivision_level=2)
tests/unit/test_voronoi_mesh_cache.py:4:``(subdivision_level, radius, lloyd_iterations, omega)`` when ``density_fn`` is
tests/unit/test_cfl.py:66:        """Higher subdivision level → finer grid."""
tests/unit/test_mpas_land_boundary.py:140:    return create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
scripts/matrix/check_conservation_all.py:168:    mesh = create_voronoi_mesh(subdivision_level=4)
scripts/matrix/check_conservation_all.py:327:    mesh = create_voronoi_mesh(subdivision_level=4)
scripts/matrix/check_conservation_all.py:497:    mesh = create_voronoi_mesh(subdivision_level=3)
tests/unit/test_plot_amip_mpas_checkpoint.py:4:wrong figure: picking the icosahedral subdivision level from the cell count
tests/unit/test_plot_amip_mpas_checkpoint.py:16:    subdivisions_for,
tests/unit/test_plot_amip_mpas_checkpoint.py:35:        assert subdivisions_for(n) == k
tests/unit/test_plot_amip_mpas_checkpoint.py:40:            subdivisions_for(10000)
scripts/run/run_baroclinic_wave_benchmark.py:289:        return "5"  # subdivision level 5 → 10242 cells (~120 km)
scripts/run/run_baroclinic_wave_benchmark.py:333:             "(e.g. C48), subdivision level for icosahedral (e.g. 5).  "
scripts/run/run_baroclinic_wave_benchmark.py:423:        _ico_mesh = create_voronoi_mesh(subdivision_level=N_GRID)
scripts/run/run_lmip_biophys.py:93:        return create_voronoi_mesh(resolution)       # N = SCVT subdivision level
scripts/run/run_omip_core2.py:3599:                   help="MPAS Voronoi subdivision level (nCells=10*4^level+2): "
scripts/run/mpas_realistic_geometry/run_mpas_seamount_rest.py:27:    JAX_ENABLE_X64=1 python scripts/run/mpas_realistic_geometry/run_mpas_seamount_rest.py --subdivision 4 --hours 24
scripts/run/mpas_realistic_geometry/run_mpas_seamount_rest.py:83:    print(f"[mpas-seamount] mesh subdivision={args.subdivision}, "
scripts/run/mpas_realistic_geometry/run_mpas_seamount_rest.py:85:    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
scripts/run/mpas_realistic_geometry/run_mpas_seamount_rest.py:168:    p.add_argument("--subdivision", type=int, default=2,
scripts/run/mpas_realistic_geometry/run_mpas_seamount_rest.py:169:                   help="Voronoi mesh subdivision level (2=162 cells, 4=2562, 5=10242)")
scripts/run/run_lmip_smoke.py:71:      * voronoi/mpas → SCVT mesh at subdivision level ``N`` (nCells columns)
scripts/run/run_lmip_smoke.py:91:        return create_voronoi_mesh(resolution)       # N = SCVT subdivision level
scripts/run/mpas_realistic_geometry/README.md:28:(`--subdivision 7 --years 5`):
scripts/run/mpas_realistic_geometry/diagnose_vertex_thickness_hybrid.py:17:Mirrors the §8f run config: subdivision_level=4 (2562 cells, ~480 km),
scripts/run/mpas_realistic_geometry/diagnose_vertex_thickness_hybrid.py:57:    print(f"[diag-vth] subdivision={args.subdivision} "
scripts/run/mpas_realistic_geometry/diagnose_vertex_thickness_hybrid.py:60:    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
scripts/run/mpas_realistic_geometry/diagnose_vertex_thickness_hybrid.py:279:    p.add_argument("--subdivision", type=int, default=4)
scripts/run/mpas_realistic_geometry/run_mpas_etopo_spinup.py:15:Default config is **smoke-test scale** (subdivision=4, 30 days) so
scripts/run/mpas_realistic_geometry/run_mpas_etopo_spinup.py:21:        --subdivision 7 --years 5 --diag-every-days 30
scripts/run/mpas_realistic_geometry/run_mpas_etopo_spinup.py:87:    print(f"[mpas-etopo] subdivision={args.subdivision}, "
scripts/run/mpas_realistic_geometry/run_mpas_etopo_spinup.py:89:    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
scripts/run/mpas_realistic_geometry/run_mpas_etopo_spinup.py:262:    p.add_argument("--subdivision", type=int, default=4,
scripts/run/mpas_realistic_geometry/run_mpas_etopo_spinup.py:263:                   help="Voronoi subdivision level (4=2562, 7=163842 cells)")
scripts/run/global_overturning/plot_mpas_baseline_snapshot.py:81:    sub_level = int(npz.get("mpas_subdivision_level", 4))
scripts/run/run_held_suarez_icos_0p5deg.py:3:subdivision_level=7 → 163,842 cells, mean cell spacing ≈ 62 km (~0.56°).
scripts/run/run_held_suarez_icos_0p5deg.py:87:                "(%d days, subdivision_level=%d, dt=%.0fs, radiation=none)",
scripts/validate/verify_scan_steps_equivalence.py:81:    mesh = create_voronoi_mesh(subdivision_level=4)
scripts/plot/plot_amip_mpas_checkpoint.py:57:def subdivisions_for(n_cells: int) -> int:
scripts/plot/plot_amip_mpas_checkpoint.py:58:    """Icosahedral subdivision level giving ``n_cells`` (10*4^k + 2).
scripts/plot/plot_amip_mpas_checkpoint.py:125:    mesh = create_voronoi_mesh(subdivisions_for(T.shape[0]))
scripts/run/global_overturning/run_global_overturning_mpas_baseline.py:91:           "mpas_subdivision_level": MPAS_SUBDIVISION_LEVEL}
scripts/run/global_overturning/diagnose_mpas_baro_noise.py:225:    sub_level = int(sample["mpas_subdivision_level"])
scripts/run/global_overturning/diagnose_mpas_baro_noise.py:234:    mesh = create_voronoi_mesh(subdivision_level=sub_level)
scripts/run/global_overturning/compare_etopo_idealized_sections.py:141:def _mpas_subdivision(n_cells):
scripts/run/global_overturning/compare_etopo_idealized_sections.py:145:        raise ValueError(f"Cannot infer MPAS subdivision: {n_cells} cells")
scripts/run/global_overturning/compare_etopo_idealized_sections.py:155:    raw_sub = int(npz.get("subdivision", -1))
scripts/run/global_overturning/compare_etopo_idealized_sections.py:156:    sub_level = raw_sub if raw_sub > 0 else _mpas_subdivision(T.shape[0])
scripts/run/global_overturning/compare_etopo_idealized_4runs.py:177:def _mpas_subdivision(n_cells: int) -> int:
scripts/run/global_overturning/compare_etopo_idealized_4runs.py:178:    """Infer subdivision level from cell count: nCells = 10·4^k + 2."""
scripts/run/global_overturning/compare_etopo_idealized_4runs.py:182:        raise ValueError(f"Cannot infer MPAS subdivision: {n_cells} cells")
scripts/run/global_overturning/compare_etopo_idealized_4runs.py:192:    # Prefer stored 'subdivision' field, but infer from cell count if missing.
scripts/run/global_overturning/compare_etopo_idealized_4runs.py:193:    raw_sub = int(npz.get("subdivision", -1))
scripts/run/global_overturning/compare_etopo_idealized_4runs.py:194:    sub_level = raw_sub if raw_sub > 0 else _mpas_subdivision(T.shape[0])
scripts/plot/plot_mpas_omip_snapshot.py:204:                   help="Icosahedral subdivision level (default 5)")
scripts/run/global_overturning/run_comparison_mpas.py:146:               "grid_type": "mpas", "subdivision": SUBDIVISION}
scripts/run/global_overturning/run_comparison_mpas.py:354:    mesh = create_voronoi_mesh(subdivision_level=SUBDIVISION)
scripts/run/global_overturning/run_comparison_mpas.py:504:        "subdivision": SUBDIVISION,
scripts/run/global_overturning/compare_moc_and_w_mpas_vs_latlon.py:181:    sub_level = int(d.get("mpas_subdivision_level", 4))
scripts/run/global_overturning/run_global_overturning_mpas_etopo.py:104:               "grid_type": "mpas", "subdivision": SUBDIVISION}
scripts/run/global_overturning/run_global_overturning_mpas_etopo.py:212:    mesh = create_voronoi_mesh(subdivision_level=SUBDIVISION)
scripts/run/global_overturning/run_global_overturning_mpas_50yr_implicit.py:71:           "mpas_subdivision_level": MPAS_SUBDIVISION_LEVEL}
scripts/data/prewarm_voronoi_mesh.py:1:"""Prewarm the on-disk Voronoi mesh cache for big subdivision levels.
scripts/data/prewarm_voronoi_mesh.py:36:                    help="Icosahedral subdivision level (9 or 10 need this "

exec
/usr/bin/bash -lc "nl -ba tests/bench/test_bench_voronoi_partition_methods.py | sed -n '1,520p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc 'nl -ba scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc 'nl -ba tests/bench/test_mpas_schedule_cost_scan_sbatch.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '300,530p'; git status --short; git diff --check; git diff --numstat -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 282ms:
     1	"""Direct tests for the Voronoi partition-method quality bench (item 8)."""
     2	from __future__ import annotations
     3	
     4	import importlib.util
     5	import json
     6	import sys
     7	from pathlib import Path
     8	
     9	import numpy as np
    10	import pytest
    11	
    12	_BENCH = (Path(__file__).resolve().parents[2]
    13	          / "scripts" / "bench" / "bench_voronoi_partition_methods.py")
    14	_spec = importlib.util.spec_from_file_location("bench_vor_part", _BENCH)
    15	mod = importlib.util.module_from_spec(_spec)
    16	_spec.loader.exec_module(mod)
    17	
    18	
    19	def _mesh(level=2):
    20	    from legoesm.grids.voronoi import create_voronoi_mesh
    21	
    22	    return create_voronoi_mesh(subdivision_level=level)
    23	
    24	
    25	def test_partition_quality_metrics_shape_and_sanity():
    26	    mesh = _mesh()
    27	    owner = mod.owner_for(mesh, "geometric", 4)
    28	    q = mod.partition_quality(mesh, owner, 4)
    29	    assert q["cells_per_rank_min"] >= 1
    30	    assert q["cells_per_rank_max"] >= q["cells_per_rank_min"]
    31	    assert q["load_imbalance_max_over_mean"] >= 1.0
    32	    assert 0 < q["edge_cut"] < int(mesh.nEdges)
    33	    assert 0.0 < q["edge_cut_fraction"] < 1.0
    34	    assert q["halo_cells_max"] >= q["halo_cells_mean"] > 0
    35	    assert 1 <= q["neighbor_ranks_max"] < 4
    36	
    37	
    38	def test_partition_quality_rejects_bad_owner():
    39	    mesh = _mesh()
    40	    n = int(mesh.nCells)
    41	    with pytest.raises(AssertionError, match="out of range"):
    42	        mod.partition_quality(mesh, np.full(n, 7), 4)
    43	    with pytest.raises(AssertionError, match="shape"):
    44	        mod.partition_quality(mesh, np.zeros(n - 1, dtype=int), 4)
    45	
    46	
    47	def test_owner_for_unknown_method_raises():
    48	    mesh = _mesh()
    49	    with pytest.raises(ValueError, match="unknown partition method"):
    50	        mod.owner_for(mesh, "voodoo", 4)
    51	
    52	
    53	def test_method_available_never_substitutes():
    54	    # geometric/sfc are dependency-free; metis truthfully reports.
    55	    assert mod.method_available("geometric") is True
    56	    assert mod.method_available("sfc") is True
    57	    try:
    58	        import pymetis  # noqa: F401
    59	
    60	        assert mod.method_available("metis") is True
    61	    except Exception:
    62	        assert mod.method_available("metis") is False
    63	
    64	
    65	def test_main_writes_quality_table(tmp_path, monkeypatch):
    66	    out = tmp_path / "q.json"
    67	    monkeypatch.setattr(sys, "argv", [
    68	        "bench", "--subdivision", "2", "--rank-counts", "2,4",
    69	        "--methods", "geometric,sfc", "--out", str(out)])
    70	    assert mod.main() == 0
    71	    payload = json.loads(out.read_text())
    72	    rows = [r for r in payload["rows"] if r.get("available")]
    73	    assert {(r["method"], r["n_ranks"]) for r in rows} == {
    74	        ("geometric", 2), ("geometric", 4), ("sfc", 2), ("sfc", 4)}
    75	    assert payload["auto_resolves_to"] in ("metis", "geometric")
    76	    md = payload["metadata"]
    77	    assert md["transport"] == "none"
    78	    assert "_incomplete" not in md
    79	
    80	
    81	def test_main_rejects_bad_args(monkeypatch):
    82	    monkeypatch.setattr(sys, "argv", ["bench", "--methods", "voodoo"])
    83	    with pytest.raises(SystemExit):
    84	        mod.main()
    85	    monkeypatch.setattr(sys, "argv", ["bench", "--rank-counts", "1"])
    86	    with pytest.raises(SystemExit):
    87	        mod.main()
    88	
    89	
    90	class _SyntheticMesh:
    91	    """4-cell ring: cells 0-1-2-3 cyclic (each cell has 2 neighbors).
    92	
    93	    Edges: (0,1) (1,2) (2,3) (3,0) + one INVALID edge (-1,-1) to lock the
    94	    valid-edge masking in the edge-cut denominator.
    95	    """
    96	    nCells = 4
    97	    nEdges = 5
    98	    maxEdges = 2
    99	    cellsOnEdge = np.array([[0, 1, 2, 3, -1],
   100	                            [1, 2, 3, 0, -1]])
   101	    cellsOnCell = np.array([[1, 2, 3, 0],    # neighbor k=0
   102	                            [3, 0, 1, 2]])   # neighbor k=1
   103	
   104	
   105	def test_metric_definitions_locked_on_synthetic_mesh():
   106	    """Exact edge cut / halo / neighbor values on a hand-built ring —
   107	    a denominator or halo-construction drift fails HERE, not in a range
   108	    check (codex finding 3)."""
   109	    mesh = _SyntheticMesh()
   110	    owner = np.array([0, 0, 1, 1])  # cells 0,1 -> rank0; 2,3 -> rank1
   111	    q = mod.partition_quality(mesh, owner, 2, halo_depth=1)
   112	    # Cut edges: (1,2) and (3,0) -> 2 of 4 VALID edges (invalid edge
   113	    # excluded from the denominator).
   114	    assert q["edge_cut"] == 2
   115	    assert q["edge_cut_fraction"] == pytest.approx(0.5)
   116	    # halo_depth=1: each rank's halo = the 2 cells of the other rank that
   117	    # touch it (ring: both of them).
   118	    assert q["halo_cells_max"] == 2
   119	    assert q["halo_cells_mean"] == pytest.approx(2.0)
   120	    assert q["halo_owned_ratio_max"] == pytest.approx(1.0)
   121	    assert q["neighbor_ranks_max"] == 1
   122	    assert q["cells_per_rank_min"] == q["cells_per_rank_max"] == 2
   123	    assert q["load_imbalance_max_over_mean"] == pytest.approx(1.0)
   124	
   125	
   126	def test_empty_rank_rejected():
   127	    mesh = _SyntheticMesh()
   128	    owner = np.array([0, 0, 0, 0])  # rank 1 skipped
   129	    with pytest.raises(AssertionError, match="empty rank"):
   130	        mod.partition_quality(mesh, owner, 2)
   131	
   132	
   133	def test_halo_matches_runtime_partition():
   134	    """Real-mesh lock: the bench's halo size equals the RUNTIME partition's
   135	    (n_local - n_owned) for the same owner array — the bench reports the
   136	    runtime's halos, not an estimate (codex finding 3)."""
   137	    from legoesm.parallel.voronoi_partition import partition_voronoi_mesh
   138	
   139	    mesh = _mesh()
   140	    owner = mod.owner_for(mesh, "geometric", 4)
   141	    q = mod.partition_quality(mesh, owner, 4, halo_depth=2)
   142	    halos = []
   143	    for r in range(4):
   144	        part = partition_voronoi_mesh(
   145	            mesh, 4, r, method="geometric", halo_depth=2, cell_owner=owner)
   146	        halos.append(int(part.n_local_cells) - int(part.n_owned_cells))
   147	    assert q["halo_cells_max"] == max(halos)
   148	    assert q["halo_cells_mean"] == pytest.approx(float(np.mean(halos)))
   149	
   150	
   151	# --- SPMD halo-schedule depth (--schedule-cost) ---------------------------
   152	
   153	
   154	def test_schedule_cost_row_reports_rounds_against_their_lower_bound():
   155	    """``schedule_cost_row`` returns the REAL schedule depth and the bound it
   156	    must be read against."""
   157	    mesh = _mesh()
   158	    sc = mod.schedule_cost_row(mesh, "geometric", 2)
   159	    # max_degree is the graph's own lower bound on a proper edge colouring,
   160	    # so a schedule can never beat it.  -1 is the scorer's "not reported"
   161	    # sentinel and would make the gap meaningless.
   162	    assert sc["max_degree"] >= 1
   163	    assert sc["n_rounds"] >= sc["max_degree"]
   164	    assert sc["coloring_gap"] == sc["n_rounds"] - sc["max_degree"]
   165	    assert sc["n_rounds_greedy"] >= sc["n_rounds"]
   166	    assert sc["score_seconds"] >= 0.0
   167	
   168	
   169	@pytest.mark.parametrize(
   170	    "n_rounds, max_degree, gap, hmin, hmax, proven",
   171	    [
   172	        # Provably optimal: no proper edge colouring beats max_degree.
   173	        (12, 12, 0, 0, 0, True),
   174	        # Vizing allows the true optimum to BE max_degree+1, so a gap of 1
   175	        # spans [0, 1] — inconclusive.  This is the case that a naive
   176	        # "gap > 0 means recolour" rule would over-claim.
   177	        (13, 12, 1, 0, 1, False),
   178	        # Guaranteed to remove at least gap-1 = 3, at most gap = 4.
   179	        (14, 10, 4, 3, 4, False),
   180	    ])
   181	def test_coloring_gap_and_headroom_are_derived_not_assumed(
   182	        monkeypatch, n_rounds, max_degree, gap, hmin, hmax, proven):
   183	    """Gap and the Vizing-bounded headroom must be COMPUTED, not assumed.
   184	
   185	    Non-vacuity, the hard way: on every mesh small enough to test quickly the
   186	    real gap is 0 (measured L2/L3/L4 x {geometric,sfc} x nd 2-16, and s6
   187	    lloyd=0 at np8/np16 — the colourer lands exactly on ``max_degree`` every
   188	    time), so a real-mesh assertion cannot tell a correct subtraction from a
   189	    hardcoded ``0``; that exact mutation passed the first version of this
   190	    test.  Stubbing the production scorer with KNOWN values is what makes
   191	    the assertion able to fail.
   192	
   193	    The ``gap == 1`` row is the one that matters: the schedule is a proper
   194	    EDGE colouring and ``max_degree`` is that graph's max vertex degree, so
   195	    Vizing gives ``Delta <= chi' <= Delta + 1``.  A gap of 1 is therefore
   196	    indistinguishable from optimal (Class 2), and claiming recolouring
   197	    headroom there would be an over-claim.
   198	    """
   199	    stub = {
   200	        "n_rounds": n_rounds, "max_degree": max_degree,
   201	        "n_rounds_greedy": n_rounds + 1, "coloring_method": "stub",
   202	        "resolved_method": "geometric", "halo_depth": 3,
   203	        "cells_per_device": 99_999, "production_strategy": "ppermute",
   204	        "reorder_target": 8, "already_reordered": False,
   205	    }
   206	    import legoesm.parallel.sharded_dynamics as sd
   207	    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
   208	
   209	    sc = mod.schedule_cost_row(object(), "geometric", 8)
   210	    assert sc["coloring_gap"] == gap
   211	    # The headroom is an INTERVAL: Vizing pins the optimum to
   212	    # {Delta, Delta+1}, so gap-1 is guaranteed and gap is the best case.
   213	    assert sc["coloring_headroom_rounds_min"] == hmin
   214	    assert sc["coloring_headroom_rounds_max"] == hmax
   215	    assert sc["coloring_optimal_proven"] is proven
   216	
   217	
   218	def test_schedule_rounds_never_beat_the_vizing_floor_on_a_real_mesh():
   219	    """Real-mesh sanity on the bound itself: a proper edge colouring can
   220	    never use fewer rounds than the graph's max degree, and the multi-start
   221	    search should not overshoot Vizing's ``Delta + 1`` either.  If this ever
   222	    fires, ``max_degree`` is not the degree of the graph being coloured and
   223	    every gap-based conclusion built on it is void."""
   224	    for method in ("geometric", "sfc"):
   225	        for n_ranks in (2, 4, 8):
   226	            sc = mod.schedule_cost_row(_mesh(level=3), method, n_ranks)
   227	            assert sc["max_degree"] <= sc["n_rounds"] <= sc["max_degree"] + 1, (
   228	                f"{method} np={n_ranks}: rounds={sc['n_rounds']} "
   229	                f"max_degree={sc['max_degree']}")
   230	
   231	
   232	def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
   233	    """The schedule is scored at the SPMD production halo depth, NOT this
   234	    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
   235	    a different graph and quietly report the wrong lane's cost."""
   236	    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH
   237	
   238	    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
   239	    assert sc["schedule_halo_depth"] == SPMD_HALO_DEPTH
   240	
   241	
   242	def test_schedule_cost_matches_the_production_scorer_exactly():
   243	    """Lock: the wrapper reports what the production scorer returns — it is
   244	    a passthrough, not a re-derivation (the whole point: a 1-ring lookalike
   245	    reports 8 rounds where the real depth-3 graph reports 12-14)."""
   246	    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
   247	
   248	    mesh = _mesh()
   249	    ref = spmd_schedule_cost(mesh, 2, method="sfc")
   250	    sc = mod.schedule_cost_row(mesh, "sfc", 2)
   251	    for key in ("n_rounds", "max_degree", "n_rounds_greedy",
   252	                "coloring_method", "resolved_method", "cells_per_device",
   253	                "production_strategy"):
   254	        assert sc[key] == ref[key], key
   255	
   256	
   257	def test_schedule_cost_flag_is_opt_in_and_lands_in_the_json(
   258	        tmp_path, monkeypatch):
   259	    """Off by default (it is the expensive layer); on, every scored row
   260	    carries the schedule block and the run records that it ran."""
   261	    out = tmp_path / "off.json"
   262	    monkeypatch.setattr(sys, "argv", [
   263	        "bench", "--subdivision", "2", "--rank-counts", "2",
   264	        "--methods", "geometric", "--out", str(out)])
   265	    assert mod.main() == 0
   266	    payload = json.loads(out.read_text())
   267	    assert "schedule" not in payload["rows"][0]
   268	    assert payload["metadata"]["extra"]["schedule_cost"] is False
   269	
   270	    out2 = tmp_path / "on.json"
   271	    monkeypatch.setattr(sys, "argv", [
   272	        "bench", "--subdivision", "2", "--rank-counts", "2",
   273	        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
   274	    assert mod.main() == 0
   275	    payload2 = json.loads(out2.read_text())
   276	    row = payload2["rows"][0]
   277	    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
   278	    assert payload2["metadata"]["extra"]["schedule_cost"] is True
   279	    # Scorer provenance per row: a copied row must show it scored a mesh
   280	    # partitioned for THIS device count, not one reordered for another.
   281	    assert row["schedule"]["reorder_target"] == row["n_ranks"]
   282	    assert row["schedule"]["already_reordered"] is False
   283	
   284	
   285	def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
   286	    """``--lloyd`` must actually select the mesh, not just be recorded.
   287	
   288	    Non-vacuity: asserting only the recorded default (50) passes even if the
   289	    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
   290	    that.  This spies on the builder, so dropping ``lloyd_iterations=
   291	    args.lloyd`` fails here, and it checks a NON-default value so the
   292	    assertion cannot be satisfied by the default.
   293	    """
   294	    import legoesm.grids.voronoi as vor
   295	
   296	    seen = {}
   297	    real = vor.create_voronoi_mesh
   298	
   299	    def spy(*a, **k):
   300	        seen.update(k)
   301	        # lloyd=0 is cheap and is what the scaling meshes actually use.
   302	        return real(*a, **k)
   303	
   304	    monkeypatch.setattr(vor, "create_voronoi_mesh", spy)
   305	    out = tmp_path / "lloyd0.json"
   306	    monkeypatch.setattr(sys, "argv", [
   307	        "bench", "--subdivision", "2", "--rank-counts", "2",
   308	        "--methods", "geometric", "--lloyd", "0", "--out", str(out)])
   309	    assert mod.main() == 0
   310	    assert seen.get("lloyd_iterations") == 0, seen
   311	    # And it is recorded, so a synthetic mesh cannot be read back as a
   312	    # production SCVT receipt.
   313	    payload = json.loads(out.read_text())
   314	    assert payload["metadata"]["extra"]["lloyd_iterations"] == 0
   315	
   316	
   317	def test_expect_rounds_gate_fails_loudly_and_never_vacuously(
   318	        tmp_path, monkeypatch):
   319	    """The instrument check must FAIL on a wrong expectation and on a pair
   320	    that was never scored — a gate that can only pass is not a gate."""
   321	    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
   322	            "--methods", "geometric", "--schedule-cost"]
   323	
   324	    # Truth first: read what this configuration really scores.
   325	    out = tmp_path / "truth.json"
   326	    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
   327	    assert mod.main() == 0
   328	    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
   329	
   330	    # Matching expectation -> pass, and the check is recorded.
   331	    ok = tmp_path / "ok.json"
   332	    monkeypatch.setattr(sys, "argv", base + [
   333	        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
   334	    assert mod.main() == 0
   335	    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
   336	
   337	    # Wrong expectation -> non-zero exit.
   338	    bad = tmp_path / "bad.json"
   339	    monkeypatch.setattr(sys, "argv", base + [
   340	        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
   341	    assert mod.main() == 1
   342	    assert json.loads(bad.read_text())["expected_rounds_check"]["failures"]
   343	
   344	    # A pair that was never scored is a FAILURE, not a silent skip —
   345	    # otherwise a sweep missing a method still reports a clean gate.
   346	    missing = tmp_path / "missing.json"
   347	    monkeypatch.setattr(sys, "argv", base + [
   348	        "--expect-rounds", "sfc:2=3", "--out", str(missing)])
   349	    assert mod.main() == 1
   350	    fails = json.loads(missing.read_text())["expected_rounds_check"]["failures"]
   351	    assert any("NOT SCORED" in f for f in fails), fails
   352	
   353	
   354	def test_expect_rounds_refuses_to_pass_vacuously_without_scoring(monkeypatch):
   355	    """Without --schedule-cost nothing is scored, so the gate would pass on
   356	    an empty comparison. It must refuse instead."""
   357	    monkeypatch.setattr(sys, "argv", [
   358	        "bench", "--subdivision", "2", "--rank-counts", "2",
   359	        "--methods", "geometric", "--expect-rounds", "geometric:2=1",
   360	        "--out", "/dev/null"])
   361	    with pytest.raises(SystemExit, match="needs --schedule-cost"):
   362	        mod.main()
   363	
   364	
   365	@pytest.mark.parametrize("spec", ["geometric:2", "geometric=2", "voodoo:2=3"])
   366	def test_expect_rounds_rejects_malformed_specs(spec):
   367	    """A typo'd expectation must raise, never be dropped — a silently
   368	    skipped expectation turns the gate into a no-op."""
   369	    with pytest.raises(ValueError, match="expect-rounds"):
   370	        mod.parse_expect_rounds(spec)
   371	
   372	
   373	def test_expect_rounds_rejects_duplicate_and_empty_specs():
   374	    """Two more ways the gate could be silently bypassed (codex round 2).
   375	
   376	    A duplicate key would let the later expectation overwrite the earlier,
   377	    so a listed-but-wrong expectation is discarded and the gate passes. A
   378	    non-empty spec that parses to nothing (``",,,"``) would make main() skip
   379	    the check entirely while the caller believes it ran.
   380	    """
   381	    with pytest.raises(ValueError, match="duplicate expectation"):
   382	        mod.parse_expect_rounds("geometric:2=999,geometric:2=13")
   383	    with pytest.raises(ValueError, match="NO expectations"):
   384	        mod.parse_expect_rounds(",,,")
   385	    # Whitespace-only is a value the caller PASSED; reading it as "no gate"
   386	    # is the same bypass (codex round 3 — the first fix guarded on
   387	    # spec.strip() and let this through).
   388	    for blank in ("   ", "\t", " , , "):
   389	        with pytest.raises(ValueError, match="NO expectations"):
   390	            mod.parse_expect_rounds(blank)
   391	    # A genuinely empty spec is the "no gate requested" default, not an error.
   392	    assert mod.parse_expect_rounds("") == {}
   393	
   394	
   395	@pytest.mark.parametrize("flag, value", [
   396	    ("--lloyd", "-1"), ("--subdivision", "-1"), ("--halo-depth", "-1")])
   397	def test_negative_numeric_args_refused_as_false_provenance(
   398	        monkeypatch, flag, value):
   399	    """All three behave exactly like 0 in the underlying code but would be
   400	    RECORDED under the negative value — a level-0 mesh filed as 'L-1'."""
   401	    argv = ["bench", "--subdivision", "2", "--rank-counts", "2",
   402	            "--methods", "geometric", "--out", "/dev/null"]
   403	    # Replace the flag if already present, else append.
   404	    if flag in argv:
   405	        argv[argv.index(flag) + 1] = value
   406	    else:
   407	        argv += [flag, value]
   408	    monkeypatch.setattr(sys, "argv", argv)
   409	    with pytest.raises(SystemExit, match=r"must be >= 0"):
   410	        mod.main()
   411	
   412	
   413	def test_expect_rounds_fails_when_a_method_is_unavailable(
   414	        tmp_path, monkeypatch):
   415	    """An expectation naming a method that reported UNAVAILABLE must fail.
   416	
   417	    Distinct from omitting the method from --methods: here the sweep asks
   418	    for it and the partitioner is missing, which is exactly how a two-method
   419	    table gets misread as a three-method one.
   420	    """
   421	    monkeypatch.setattr(mod, "method_available",
   422	                        lambda m: m != "metis")
   423	    out = tmp_path / "unavail.json"
   424	    monkeypatch.setattr(sys, "argv", [
   425	        "bench", "--subdivision", "2", "--rank-counts", "2",
   426	        "--methods", "geometric,metis", "--schedule-cost",
   427	        "--expect-rounds", "metis:2=1", "--out", str(out)])
   428	    assert mod.main() == 1
   429	    payload = json.loads(out.read_text())
   430	    fails = payload["expected_rounds_check"]["failures"]
   431	    assert any("NOT SCORED" in f for f in fails), fails
   432	    # Assert the EXPLICIT unavailable row, not just the failure: without
   433	    # this the test would also pass if metis were silently omitted, which
   434	    # is the very substitution this bench refuses to make.
   435	    metis_rows = [r for r in payload["rows"] if r["method"] == "metis"]
   436	    assert metis_rows and metis_rows[0]["available"] is False, payload["rows"]
   437	
   438	
   439	def test_lloyd_rejects_negative(monkeypatch):
   440	    """A negative Lloyd count behaves like 0 in the builder (it relaxes only
   441	    for > 0) but is recorded and cached under its own key — false
   442	    provenance, so the CLI must refuse it rather than run."""
   443	    monkeypatch.setattr(sys, "argv", [
   444	        "bench", "--subdivision", "2", "--rank-counts", "2",
   445	        "--methods", "geometric", "--lloyd", "-1", "--out", "/dev/null"])
   446	    with pytest.raises(SystemExit, match=r"--lloyd must be >= 0"):
   447	        mod.main()

 succeeded in 280ms:
     1	#!/bin/bash -l
     2	#SBATCH --job-name=mpas_sched_cost
     3	#SBATCH --account=bb1596
     4	#SBATCH --partition=shared
     5	#SBATCH --ntasks=1
     6	#SBATCH --cpus-per-task=8
     7	#SBATCH --mem=200G
     8	#SBATCH --time=24:00:00
     9	#SBATCH --output=mpas_schedule_cost_scan.%j.log
    10	# ===========================================================================
    11	# Does RECOLOURING still have room on the MPAS halo schedule?  CPU-only,
    12	# ZERO GPU hours.
    13	#
    14	# Note this is NOT a two-way choice between colouring and ownership: a
    15	# redesigned DIRECTED schedule (see the SCOPE note below) is a third path,
    16	# and it is not bounded by the quantity measured here.  What this run
    17	# settles is only whether recolouring THIS undirected graph is exhausted.
    18	#
    19	# WHY THIS RUN EXISTS
    20	# MPAS GPU is the worst-scaling lane we have: measured/modelled-bound 3.16x
    21	# (s8@16) to 4.47x (s9@64), and one halo fill costs 12-14 SEQUENTIAL ppermute
    22	# rounds.  Both independent reviews (codex + GLM, 2026-08-07) ranked cutting
    23	# that round count as the top structural lever, at 300-800 LOC and 7-14 days
    24	# for a partitioner with a new objective.  That estimate is only worth
    25	# spending if recolouring is genuinely exhausted, and the colourer's own
    26	# lower bound decides it.
    27	#
    28	# The schedule is a proper EDGE colouring of the device communication graph
    29	# (one colour = one ppermute round) and max_degree is that graph's maximum
    30	# vertex degree, so VIZING bounds the chromatic index: Delta <= chi' <=
    31	# Delta + 1.  Read coloring_gap = n_rounds - max_degree through that:
    32	#     gap == 0 -> PROVABLY OPTIMAL; recolouring headroom is exactly zero.
    33	#     gap == 1 -> indistinguishable from optimal (a Class 2 graph really
    34	#                 needs Delta+1, and deciding Class 1/2 is NP-complete);
    35	#                 nothing provable to win.
    36	#     gap >= 2 -> at least gap-1 rounds of genuine recolouring headroom.
    37	# So recolouring is capped at ~1 round out of 12-14 (<= ~7%) wherever the
    38	# multi-start search lands on Delta or Delta+1 — which it does on every
    39	# configuration probed so far.  This run measures whether the production
    40	# working points are in that regime.
    41	#
    42	# (a) NUMBER PRODUCED: n_rounds, max_degree and their gap per
    43	#     (method x n_dev) at the production working points.
    44	# (b) CONFIRMS a cheap fix: gap >= 2 at any production (ppermute) row
    45	#     (recolouring then removes at least gap-1 rounds, guaranteed).
    46	#     gap == 0: recolouring is provably worthless AT THIS OWNERSHIP.
    47	#     gap == 1: INCONCLUSIVE — Class 1 vs Class 2 is NP-complete, so this
    48	#     neither confirms nor refutes a one-round win.
    49	#     Note what a gap of 0 does NOT prove: Delta bounds only a proper
    50	#     UNDIRECTED edge colouring of this graph, and the builder forces BOTH
    51	#     ppermute directions per pair even when one send map is empty, so a
    52	#     redesigned DIRECTED schedule is not bounded by Delta.  "Ownership is
    53	#     the only path" would overclaim; the honest statement is "recolouring
    54	#     this undirected graph is exhausted".
    55	# (c) WHY NOT CHEAPER: this IS the cheap test — no GPU, no MPI, no model
    56	#     step.  It cannot be shrunk further onto small meshes: measured
    57	#     2026-08-07, L2/L4 x {geometric,sfc} x nd 2-16 all report gap == 0,
    58	#     but EVERY one of those rows auto-selects the ALLGATHER strategy
    59	#     (cells/device below the threshold), so production runs no ppermute
    60	#     schedule there and the number is counterfactual.  Only meshes big
    61	#     enough to keep cells/device above the threshold answer the question.
    62	#     s6 lloyd=0 also gave gap == 0 on all three methods — rounds 7/7/6 at
    63	#     np8 and 13/10/10 at np16 for geometric/sfc/metis.  Both are genuinely
    64	#     ppermute (L6 = 40,962 cells; padded, np8 gives 40,968/8 = 5,121 and
    65	#     np16 gives 40,976/16 = 2,561 cells/device, each above the 2,000
    66	#     threshold).  But np8's 7 == n_dev-1 is complete-graph saturation and
    67	#     says nothing; only np16 (13/10/10 < 15) escapes it.  Still a SMALL
    68	#     working point; s8-s10 are the production ones.
    69	#
    70	# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
    71	# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
    72	#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
    73	#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
    74	# Arms 1-2 must REPRODUCE those before arm 3's unknown s10 number is quoted.
    75	# A scan that misses the known answer is a broken instrument, and its s10
    76	# row must not be believed.
    77	#
    78	# lloyd=0 throughout: it is what the reference census used AND the cache key
    79	# the prewarmed s8/s9/s10 meshes were written under.  It is the LABELLED
    80	# synthetic scaling mesh, recorded in every row's metadata so it can never be
    81	# read back as a production SCVT receipt.
    82	#
    83	# Arms run cheapest-first and each writes its own JSON, so a later arm that
    84	# runs out of time or memory cannot lose an earlier arm's result.  s10@128 is
    85	# last and is the one genuinely at risk: the scorer is known not to have
    86	# finished at subdiv-8@128 on a laptop, which is why this asks for 24 h.
    87	#
    88	# SUBMIT (from the repo root):
    89	#   sbatch scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
    90	# ===========================================================================
    91	set -uo pipefail
    92	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    93	export JAX_PLATFORMS=cpu
    94	export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
    95	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    96	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    97	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    98	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    99	
   100	OUT=results/a1/mpas_schedule_cost
   101	mkdir -p "$OUT"
   102	# Overridable ONLY so the exit-status logic below can be regression-tested
   103	# with a stub bench (tests/bench/test_mpas_schedule_cost_scan_sbatch.py).
   104	# A wrong exit code here is the blocker class this gate exists to prevent:
   105	# SLURM would file a scan with no valid result as COMPLETED.
   106	BENCH="${LEGOESM_SCHEDULE_COST_BENCH:-scripts/bench/bench_voronoi_partition_methods.py}"
   107	
   108	# pymetis is present in .venv but was ABSENT from .venv-mpi for a while, and a
   109	# missing partitioner is reported as "unavailable" rather than substituted.
   110	# Say which python and whether metis is really there, so a two-method table
   111	# cannot be misread as a three-method one.
   112	echo "[scan] python=$PY"
   113	"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
   114	  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"
   115	
   116	run_arm () {  # $1=level  $2=rank-counts  $3=label  $4=expect-rounds (may be "")
   117	  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
   118	  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
   119	  "$PY" "$BENCH" \
   120	      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
   121	      --methods geometric,sfc,metis --schedule-cost \
   122	      ${4:+--expect-rounds "$4"} \
   123	      --out "$OUT/schedule_cost_s$1.json"
   124	  local rc=$?   # capture BEFORE anything else runs, or `date` overwrites it
   125	  echo "[scan] arm $3 exit=$rc"
   126	  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
   127	  return $rc
   128	}
   129	
   130	# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
   131	# makes the bench exit non-zero unless it reproduces the census below, and a
   132	# listed pair that never got scored (e.g. pymetis missing) counts as a
   133	# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
   134	# pass — an instrument that misses the known answer cannot be trusted on the
   135	# unknown one.
   136	S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
   137	S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"
   138	
   139	run_arm 8 64,128 "1/3 VALIDATION s8" "$S8_EXPECT"; RC8=$?
   140	run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?
   141	
   142	if [ "$RC8" -ne 0 ] || [ "$RC9" -ne 0 ]; then
   143	  echo "[scan] INSTRUMENT CHECK FAILED (s8 rc=$RC8, s9 rc=$RC9)."
   144	  echo "[scan] The scorer does not reproduce the reference census, so an s10"
   145	  echo "[scan] number from it would be untrustworthy. NOT running arm 3."
   146	  echo "SCAN_ABORTED_VALIDATION"
   147	  exit 1
   148	fi
   149	
   150	# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
   151	run_arm 10 128 "3/3 UNKNOWN s10" ""; RC10=$?
   152	echo "[scan] arm 3 rc=$RC10"
   153	
   154	if [ "$RC10" -ne 0 ]; then
   155	  # Do NOT let SLURM record this as COMPLETED: there is no valid s10 row,
   156	  # and an exit-0 job with a missing result reads as a successful scan.
   157	  echo "SCAN_FAILED_ARM3"
   158	  exit "$RC10"
   159	fi
   160	
   161	echo "SCAN_DONE"

 succeeded in 338ms:
     1	"""Exit-status regression for the MPAS schedule-cost scan launcher.
     2	
     3	The failure this guards is not hypothetical: the first version of the script
     4	captured arm 3's status into ``RC10`` and then ended on a successful ``echo``,
     5	so a scan that produced NO valid s10 result exited 0 and SLURM filed it as
     6	COMPLETED (codex round 2, BLOCKER).  "Tool status is not evidence" cuts both
     7	ways — a launcher that cannot report failure makes every downstream reading
     8	of `sacct` a lie.
     9	
    10	The script is driven with a STUB bench (``LEGOESM_SCHEDULE_COST_BENCH``) whose
    11	exit codes are scripted per arm, so all three paths are exercised without a
    12	mesh, a partitioner, or a compute node.
    13	"""
    14	from __future__ import annotations
    15	
    16	import os
    17	import subprocess
    18	from pathlib import Path
    19	
    20	import pytest
    21	
    22	_REPO = Path(__file__).resolve().parents[2]
    23	_SBATCH = (_REPO / "scripts" / "cluster" / "scaling_levante"
    24	           / "mpas_schedule_cost_scan.sbatch")
    25	
    26	
    27	def _run(tmp_path, codes):
    28	    """Run the launcher with a stub bench that exits ``codes`` per call.
    29	
    30	    ``codes`` is one exit status per invocation, in order (arm 1, arm 2,
    31	    arm 3).  The stub also records the ``--subdivision`` of each call so a
    32	    test can assert an arm did NOT run.
    33	    """
    34	    stub = tmp_path / "stub_bench.py"
    35	    calls = tmp_path / "calls.txt"
    36	    stub.write_text(
    37	        "import sys\n"
    38	        f"codes = {list(codes)!r}\n"
    39	        f"log = {str(calls)!r}\n"
    40	        "sub = sys.argv[sys.argv.index('--subdivision') + 1]\n"
    41	        "with open(log, 'a') as f:\n"
    42	        "    f.write(sub + '\\n')\n"
    43	        "n = sum(1 for _ in open(log))\n"
    44	        "sys.exit(codes[n - 1] if n <= len(codes) else 0)\n"
    45	    )
    46	    env = dict(os.environ)
    47	    env["LEGOESM_SCHEDULE_COST_BENCH"] = str(stub)
    48	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
    49	    env["LEGOESM_REPO"] = str(_REPO)
    50	    # _env.sh resolves PY from here; the stub is plain python, and _env.sh
    51	    # verifies PY can import jax, so use the repo venv.
    52	    env.setdefault("LEGOESM_PYTHON", str(_REPO / ".venv" / "bin" / "python"))
    53	    proc = subprocess.run(
    54	        ["bash", str(_SBATCH)], env=env, cwd=str(_REPO),
    55	        capture_output=True, text=True, timeout=600)
    56	    ran = calls.read_text().split() if calls.exists() else []
    57	    return proc, ran
    58	
    59	
    60	@pytest.mark.skipif(not (_REPO / ".venv" / "bin" / "python").exists(),
    61	                    reason="needs the repo venv for _env.sh's jax check")
    62	def test_validation_failure_aborts_before_the_unknown_arm(tmp_path):
    63	    """Arm 1 failing must abort with non-zero AND must not run arm 3.
    64	
    65	    An instrument that misses the known census cannot be trusted on the
    66	    unknown one, so producing an s10 number anyway would be worse than
    67	    producing none.
    68	    """
    69	    proc, ran = _run(tmp_path, [1, 0, 0])
    70	    assert proc.returncode != 0, proc.stdout[-2000:]
    71	    assert "SCAN_ABORTED_VALIDATION" in proc.stdout
    72	    assert "10" not in ran, f"arm 3 ran despite failed validation: {ran}"
    73	
    74	
    75	@pytest.mark.skipif(not (_REPO / ".venv" / "bin" / "python").exists(),
    76	                    reason="needs the repo venv for _env.sh's jax check")
    77	def test_arm3_failure_is_not_reported_as_success(tmp_path):
    78	    """The original BLOCKER: arm 3 fails, job must NOT exit 0."""
    79	    proc, ran = _run(tmp_path, [0, 0, 3])
    80	    assert proc.returncode == 3, (
    81	        f"arm-3 status not propagated (got {proc.returncode})\n"
    82	        f"{proc.stdout[-2000:]}")
    83	    assert "SCAN_FAILED_ARM3" in proc.stdout
    84	    assert "SCAN_DONE" not in proc.stdout
    85	    assert ran == ["8", "9", "10"], ran
    86	
    87	
    88	@pytest.mark.skipif(not (_REPO / ".venv" / "bin" / "python").exists(),
    89	                    reason="needs the repo venv for _env.sh's jax check")
    90	def test_full_success_exits_zero_and_runs_every_arm(tmp_path):
    91	    proc, ran = _run(tmp_path, [0, 0, 0])
    92	    assert proc.returncode == 0, proc.stdout[-2000:]
    93	    assert "SCAN_DONE" in proc.stdout
    94	    assert ran == ["8", "9", "10"], ran

 succeeded in 567ms:
   300	    return {
   301	        "n_rounds": int(cost["n_rounds"]),
   302	        "n_rounds_greedy": int(cost["n_rounds_greedy"]),
   303	        "max_degree": int(cost["max_degree"]),
   304	        # The decisive column, read through Vizing (see module docstring).
   305	        "coloring_gap": gap,
   306	        # Rounds a perfect recolouring could remove, as an INTERVAL: Vizing
   307	        # pins the optimum to {Delta, Delta+1}, so gap-1 is guaranteed and
   308	        # gap is the best case.  A gap of 1 spans [0, 1] = inconclusive.
   309	        "coloring_headroom_rounds_min": max(0, gap - 1),
   310	        "coloring_headroom_rounds_max": gap,
   311	        "coloring_optimal_proven": gap == 0,
   312	        # Scorer provenance, carried per row: a copied/flattened row must be
   313	        # able to show it scored a raw mesh partitioned for THIS device
   314	        # count, not one reordered for a different target.
   315	        "reorder_target": cost["reorder_target"],
   316	        "already_reordered": cost["already_reordered"],
   317	        "coloring_method": cost["coloring_method"],
   318	        "resolved_method": cost["resolved_method"],
   319	        "schedule_halo_depth": int(cost["halo_depth"]),
   320	        "cells_per_device": int(cost["cells_per_device"]),
   321	        # "allgather" => production runs no ppermute schedule here, so the
   322	        # round count above is COUNTERFACTUAL, not a cost production pays.
   323	        "production_strategy": cost["production_strategy"],
   324	        "score_seconds": round(time.perf_counter() - t0, 2),
   325	    }
   326	
   327	
   328	def main() -> int:
   329	    p = argparse.ArgumentParser(
   330	        description=__doc__,
   331	        formatter_class=argparse.RawDescriptionHelpFormatter)
   332	    p.add_argument("--subdivision", type=int, default=5,
   333	                   help="Icosahedral level (L5=10,242 cells; L6=40,962).")
   334	    p.add_argument("--rank-counts", type=str, default="2,4,8,16")
   335	    p.add_argument("--halo-depth", type=int, default=2,
   336	                   help="Halo layers (runtime default 2, del4 support).")
   337	    p.add_argument("--methods", type=str, default=",".join(METHODS))
   338	    p.add_argument("--schedule-cost", action="store_true",
   339	                   help="Also score the SPMD ppermute halo-schedule depth "
   340	                        "(n_rounds vs max_degree) per method x rank count. "
   341	                        "Uses the production SPMD halo depth, NOT "
   342	                        "--halo-depth. Expensive: minutes per candidate at "
   343	                        "subdiv>=8 — run it under batch.")
   344	    p.add_argument("--lloyd", type=int, default=50,
   345	                   help="Lloyd relaxation iterations for the mesh. 50 = the "
   346	                        "production SCVT key; 0 = the LABELLED synthetic "
   347	                        "scaling mesh. Recorded so a lloyd=0 mesh can never "
   348	                        "masquerade as a production receipt, and it must "
   349	                        "match the prewarmed cache key at subdiv>=9. "
   350	                        "Must be >= 0: the builder relaxes only when this is "
   351	                        "> 0, so a negative behaves exactly like 0 while "
   352	                        "being recorded (and cached) under a different key — "
   353	                        "false provenance.")
   354	    p.add_argument("--expect-rounds", type=str, default="",
   355	                   help="Instrument check, MECHANICAL. Comma-separated "
   356	                        "'<method>:<n_ranks>=<n_rounds>' expectations (e.g. "
   357	                        "'sfc:64=12,sfc:128=14'). Every listed pair must be "
   358	                        "scored and match, or main() returns 1 — so a caller "
   359	                        "that reproduces a known census can GATE on it "
   360	                        "instead of asserting agreement in a comment. "
   361	                        "Requires --schedule-cost.")
   362	    p.add_argument("--out", type=str,
   363	                   default="results/a1/voronoi_partition_quality.json")
   364	    args = p.parse_args()
   365	
   366	    rank_counts = [int(x) for x in args.rank_counts.split(",") if x]
   367	    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
   368	    for m in methods:
   369	        if m not in METHODS:
   370	            raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
   371	    if not rank_counts or any(n < 2 for n in rank_counts):
   372	        raise SystemExit("--rank-counts needs integers >= 2")
   373	    # Negative values in these three are all the SAME false-provenance bug:
   374	    # the underlying code treats them exactly like 0 (no relaxation, no
   375	    # bisection, `range(-1)` is empty), but the run is recorded under the
   376	    # negative value, so a level-0 mesh gets filed as "L-1".
   377	    if args.lloyd < 0:
   378	        raise SystemExit(
   379	            f"--lloyd must be >= 0, got {args.lloyd}: the mesh builder "
   380	            f"relaxes only for > 0, so a negative is indistinguishable from "
   381	            f"0 in the mesh but is recorded and cached under its own key.")
   382	    if args.subdivision < 0:
   383	        raise SystemExit(
   384	            f"--subdivision must be >= 0, got {args.subdivision}: the mesh "
   385	            f"builder bisects only for > 0, so a negative silently yields "
   386	            f"the level-0 base mesh while being recorded as "
   387	            f"L{args.subdivision}.")
   388	    if args.halo_depth < 0:
   389	        raise SystemExit(
   390	            f"--halo-depth must be >= 0, got {args.halo_depth}: the halo "
   391	            f"loop is `range(depth)`, so a negative behaves exactly like 0 "
   392	            f"while being recorded as {args.halo_depth}.")
   393	    expect = parse_expect_rounds(args.expect_rounds)
   394	    if expect and not args.schedule_cost:
   395	        raise SystemExit(
   396	            "--expect-rounds compares scored round counts, so it needs "
   397	            "--schedule-cost; without it nothing is scored and the gate "
   398	            "would pass vacuously.")
   399	
   400	
   401	    from legoesm.grids.voronoi import create_voronoi_mesh
   402	    from legoesm.parallel.voronoi_partition import resolve_partition_method
   403	
   404	    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
   405	                               lloyd_iterations=args.lloyd)
   406	    if max(rank_counts) > int(mesh.nCells):
   407	        raise SystemExit(
   408	            f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
   409	            f"{int(mesh.nCells)} cells (empty ranks are meaningless).")
   410	    print(f"mesh L{args.subdivision}: {int(mesh.nCells)} cells, "
   411	          f"{int(mesh.nEdges)} edges; auto -> "
   412	          f"{resolve_partition_method('auto')!r}")
   413	
   414	    rows = []
   415	    for method in methods:
   416	        if not method_available(method):
   417	            print(f"  {method:9s}: UNAVAILABLE (pymetis not importable) — "
   418	                  f"column omitted, never substituted")
   419	            rows.append({"method": method, "available": False})
   420	            continue
   421	        for n_ranks in rank_counts:
   422	            q = partition_quality(
   423	                mesh, owner_for(mesh, method, n_ranks), n_ranks,
   424	                halo_depth=args.halo_depth)
   425	            row = {"method": method, "available": True,
   426	                   "n_ranks": n_ranks, **q}
   427	            rows.append(row)
   428	            print(f"  {method:9s} np={n_ranks:3d} | "
   429	                  f"imbalance={q['load_imbalance_max_over_mean']:.3f} | "
   430	                  f"edge_cut={q['edge_cut']:6d} "
   431	                  f"({100 * q['edge_cut_fraction']:.2f}%) | "
   432	                  f"halo max={q['halo_cells_max']:5d} "
   433	                  f"mean={q['halo_cells_mean']:8.1f} | "
   434	                  f"halo/owned max={q['halo_owned_ratio_max']:.3f} | "
   435	                  f"nbrs max={q['neighbor_ranks_max']}")
   436	            if args.schedule_cost:
   437	                sc = schedule_cost_row(mesh, method, n_ranks)
   438	                row["schedule"] = sc
   439	                note = ("  [COUNTERFACTUAL: production auto-selects "
   440	                        "allgather here, no ppermute schedule]"
   441	                        if sc["production_strategy"] == "allgather" else "")
   442	                verdict = ("colouring PROVABLY OPTIMAL (Vizing floor)"
   443	                           if sc["coloring_optimal_proven"] else
   444	                           f"recolour headroom "
   445	                           f"{sc['coloring_headroom_rounds_min']}-"
   446	                           f"{sc['coloring_headroom_rounds_max']} round(s)")
   447	                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
   448	                      f"rounds={sc['n_rounds']:3d} "
   449	                      f"max_degree={sc['max_degree']:3d} "
   450	                      f"gap={sc['coloring_gap']:+d} "
   451	                      f"-> {verdict} "
   452	                      f"({sc['score_seconds']:.1f}s){note}", flush=True)
   453	
   454	    payload = {
   455	        "rows": rows,
   456	        "auto_resolves_to": resolve_partition_method("auto"),
   457	        "step_time_pointer": (
   458	            "step-time per method: bench_ocean_mpas_scaling.py / "
   459	            "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
   460	            "per launch, same case otherwise)"),
   461	        "metadata": annotate_incomplete(scaling_metadata(
   462	            grid="voronoi",
   463	            component="partitioning",
   464	            resolution=f"L{args.subdivision}",
   465	            n_levels=0,
   466	            precision="n/a",
   467	            decomposition="cell_partition",
   468	            solver_variant="n/a",
   469	            scaling_kind="partition-quality",
   470	            transport="none",
   471	            extra={"rank_counts": rank_counts, "methods": methods,
   472	                   "halo_depth": args.halo_depth,
   473	                   "schedule_cost": bool(args.schedule_cost),
   474	                   "lloyd_iterations": args.lloyd},
   475	        )),
   476	    }
   477	    outdir = os.path.dirname(args.out)
   478	    if outdir:
   479	        os.makedirs(outdir, exist_ok=True)
   480	    with open(args.out, "w") as f:
   481	        json.dump(payload, f, indent=2)
   482	    print(f"JSON: {args.out}")
   483	
   484	    if expect:
   485	        failures = check_expected_rounds(rows, expect)
   486	        payload["expected_rounds_check"] = {
   487	            "expected": {f"{m}:{n}": v for (m, n), v in sorted(expect.items())},
   488	            "failures": failures,
   489	            "passed": not failures,
   490	        }
   491	        with open(args.out, "w") as f:
   492	            json.dump(payload, f, indent=2)
   493	        if failures:
   494	            print(f"INSTRUMENT CHECK FAILED ({len(failures)}):")
   495	            for line in failures:
   496	                print(f"  {line}")
   497	            print("The scorer did NOT reproduce the known census — treat every "
   498	                  "unknown row in this run as UNTRUSTED.")
   499	            return 1
   500	        print(f"INSTRUMENT CHECK PASSED ({len(expect)} expectations)")
   501	    return 0
   502	
   503	
   504	if __name__ == "__main__":
   505	    raise SystemExit(main())
 M scripts/bench/bench_voronoi_partition_methods.py
 M tests/bench/test_bench_voronoi_partition_methods.py
?? .physics-validator/flag_reachability_audit/
?? .physics-validator/homogeneous_ice_nucleation/
?? .physics-validator/scaling_campaign/codex_consult_hundreds_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r2_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r3_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r4_2026-08-07.md
?? config/aimip/clubb_bechtold/
?? config/aimip/scale/suite_sfno_full_longtrain.yaml
?? config/aimip/scale/suite_sfno_full_maxdata.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed0.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed1.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed2.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed3.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed4.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed5.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed6.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed7.yaml
?? config/aimip/scale/suite_sfno_full_ood_finetune.yaml
?? config/aimip/scale/suite_sfno_full_res_t106.yaml
?? config/aimip/scale/suite_sfno_full_res_t21.yaml
?? config/aimip/scale/suite_sfno_full_res_t42.yaml
?? config/aimip/scale/suite_sfno_full_res_t63.yaml
?? config/aimip/scale/suite_sfno_full_t63_ace2.yaml
?? config/aimip/scale/suite_sfno_full_t63_allyears_ace2.yaml
?? config/aimip/scale/suite_sfno_full_t63_allyears_gc.yaml
?? config/aimip/scale/suite_sfno_full_ucast_dropout.yaml
?? config/aimip/scale/suite_sfno_full_ucast_mae.yaml
?? config/aimip/scale/suite_sfno_full_ucast_muon.yaml
?? config/aimip/scale/suite_sfno_full_ucast_muondrop.yaml
?? config/aimip/scale/suite_sfno_full_ucast_muonlr.yaml
?? config/wb/loss_presets/ucast.yaml
?? docs/dev-notes/ace2_learning_strategy_gap.md
?? docs/dev-notes/ucast_learning_strategy_gap.md
?? drift.txt
?? nat.txt
?? scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
?? scripts/cluster/unified_training/bench_wb2_levante.sbatch
?? scripts/cluster/unified_training/score_ucast_arms.sbatch
?? scripts/cluster/unified_training/score_ucast_probabilistic.sbatch
?? tests/bench/test_mpas_schedule_cost_scan_sbatch.py
?? tests/distributed/test_sfno_full_dp_mpi.py
?? tests/unit/test_ace2_sfno_suites.py
?? tests/unit/test_aimip_eval_train_gate.py
?? tests/unit/test_aimip_generic_scheme_params.py
?? tests/unit/test_apply_params_to_pipeline.py
?? tests/unit/test_ensemble_crps.py
?? tests/unit/test_loss_level_weighting.py
?? tests/unit/test_multiseed_ensemble_cli.py
?? tests/unit/test_muon_partitioned_groups.py
?? tests/unit/test_persistence_normalized_loss.py
?? tests/unit/test_sfno_full_data_parallel.py
?? tests/unit/test_sfno_full_pmap_devices.py
?? tests/unit/test_sfno_pe_rollout_stability.py
?? tests/unit/test_ucast_learning_strategy.py
?? tests/unit/test_wb_climatology_reference.py
?? tests/unit/test_wb_scorecard_sota_units.py
287	2	scripts/bench/bench_voronoi_partition_methods.py
299	0	tests/bench/test_bench_voronoi_partition_methods.py

exec
/usr/bin/bash -lc "nl -ba tests/bench/test_bench_voronoi_partition_methods.py | sed -n '130,360p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "git diff -- tests/bench/test_bench_voronoi_partition_methods.py | sed -n '1,520p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "sed -n '1,220p' scripts/cluster/scaling_levante/_env.sh; rg -n \"def spmd_schedule_cost|SPMD_HALO_DEPTH|def _build_ppermute_schedule|production_strategy|cells_per_device|allgather\" legoesm/parallel" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "bash -c 'f() { printf \"<%s>\\n\" \""'$@"; }; x="sfc:64=12,sfc:128=14"; f ${x:+--expect-rounds "$x"}'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 230ms:
<--expect-rounds>
<sfc:64=12,sfc:128=14>

 succeeded in 247ms:
   130	        mod.partition_quality(mesh, owner, 2)
   131	
   132	
   133	def test_halo_matches_runtime_partition():
   134	    """Real-mesh lock: the bench's halo size equals the RUNTIME partition's
   135	    (n_local - n_owned) for the same owner array — the bench reports the
   136	    runtime's halos, not an estimate (codex finding 3)."""
   137	    from legoesm.parallel.voronoi_partition import partition_voronoi_mesh
   138	
   139	    mesh = _mesh()
   140	    owner = mod.owner_for(mesh, "geometric", 4)
   141	    q = mod.partition_quality(mesh, owner, 4, halo_depth=2)
   142	    halos = []
   143	    for r in range(4):
   144	        part = partition_voronoi_mesh(
   145	            mesh, 4, r, method="geometric", halo_depth=2, cell_owner=owner)
   146	        halos.append(int(part.n_local_cells) - int(part.n_owned_cells))
   147	    assert q["halo_cells_max"] == max(halos)
   148	    assert q["halo_cells_mean"] == pytest.approx(float(np.mean(halos)))
   149	
   150	
   151	# --- SPMD halo-schedule depth (--schedule-cost) ---------------------------
   152	
   153	
   154	def test_schedule_cost_row_reports_rounds_against_their_lower_bound():
   155	    """``schedule_cost_row`` returns the REAL schedule depth and the bound it
   156	    must be read against."""
   157	    mesh = _mesh()
   158	    sc = mod.schedule_cost_row(mesh, "geometric", 2)
   159	    # max_degree is the graph's own lower bound on a proper edge colouring,
   160	    # so a schedule can never beat it.  -1 is the scorer's "not reported"
   161	    # sentinel and would make the gap meaningless.
   162	    assert sc["max_degree"] >= 1
   163	    assert sc["n_rounds"] >= sc["max_degree"]
   164	    assert sc["coloring_gap"] == sc["n_rounds"] - sc["max_degree"]
   165	    assert sc["n_rounds_greedy"] >= sc["n_rounds"]
   166	    assert sc["score_seconds"] >= 0.0
   167	
   168	
   169	@pytest.mark.parametrize(
   170	    "n_rounds, max_degree, gap, hmin, hmax, proven",
   171	    [
   172	        # Provably optimal: no proper edge colouring beats max_degree.
   173	        (12, 12, 0, 0, 0, True),
   174	        # Vizing allows the true optimum to BE max_degree+1, so a gap of 1
   175	        # spans [0, 1] — inconclusive.  This is the case that a naive
   176	        # "gap > 0 means recolour" rule would over-claim.
   177	        (13, 12, 1, 0, 1, False),
   178	        # Guaranteed to remove at least gap-1 = 3, at most gap = 4.
   179	        (14, 10, 4, 3, 4, False),
   180	    ])
   181	def test_coloring_gap_and_headroom_are_derived_not_assumed(
   182	        monkeypatch, n_rounds, max_degree, gap, hmin, hmax, proven):
   183	    """Gap and the Vizing-bounded headroom must be COMPUTED, not assumed.
   184	
   185	    Non-vacuity, the hard way: on every mesh small enough to test quickly the
   186	    real gap is 0 (measured L2/L3/L4 x {geometric,sfc} x nd 2-16, and s6
   187	    lloyd=0 at np8/np16 — the colourer lands exactly on ``max_degree`` every
   188	    time), so a real-mesh assertion cannot tell a correct subtraction from a
   189	    hardcoded ``0``; that exact mutation passed the first version of this
   190	    test.  Stubbing the production scorer with KNOWN values is what makes
   191	    the assertion able to fail.
   192	
   193	    The ``gap == 1`` row is the one that matters: the schedule is a proper
   194	    EDGE colouring and ``max_degree`` is that graph's max vertex degree, so
   195	    Vizing gives ``Delta <= chi' <= Delta + 1``.  A gap of 1 is therefore
   196	    indistinguishable from optimal (Class 2), and claiming recolouring
   197	    headroom there would be an over-claim.
   198	    """
   199	    stub = {
   200	        "n_rounds": n_rounds, "max_degree": max_degree,
   201	        "n_rounds_greedy": n_rounds + 1, "coloring_method": "stub",
   202	        "resolved_method": "geometric", "halo_depth": 3,
   203	        "cells_per_device": 99_999, "production_strategy": "ppermute",
   204	        "reorder_target": 8, "already_reordered": False,
   205	    }
   206	    import legoesm.parallel.sharded_dynamics as sd
   207	    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
   208	
   209	    sc = mod.schedule_cost_row(object(), "geometric", 8)
   210	    assert sc["coloring_gap"] == gap
   211	    # The headroom is an INTERVAL: Vizing pins the optimum to
   212	    # {Delta, Delta+1}, so gap-1 is guaranteed and gap is the best case.
   213	    assert sc["coloring_headroom_rounds_min"] == hmin
   214	    assert sc["coloring_headroom_rounds_max"] == hmax
   215	    assert sc["coloring_optimal_proven"] is proven
   216	
   217	
   218	def test_schedule_rounds_never_beat_the_vizing_floor_on_a_real_mesh():
   219	    """Real-mesh sanity on the bound itself: a proper edge colouring can
   220	    never use fewer rounds than the graph's max degree, and the multi-start
   221	    search should not overshoot Vizing's ``Delta + 1`` either.  If this ever
   222	    fires, ``max_degree`` is not the degree of the graph being coloured and
   223	    every gap-based conclusion built on it is void."""
   224	    for method in ("geometric", "sfc"):
   225	        for n_ranks in (2, 4, 8):
   226	            sc = mod.schedule_cost_row(_mesh(level=3), method, n_ranks)
   227	            assert sc["max_degree"] <= sc["n_rounds"] <= sc["max_degree"] + 1, (
   228	                f"{method} np={n_ranks}: rounds={sc['n_rounds']} "
   229	                f"max_degree={sc['max_degree']}")
   230	
   231	
   232	def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
   233	    """The schedule is scored at the SPMD production halo depth, NOT this
   234	    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
   235	    a different graph and quietly report the wrong lane's cost."""
   236	    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH
   237	
   238	    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
   239	    assert sc["schedule_halo_depth"] == SPMD_HALO_DEPTH
   240	
   241	
   242	def test_schedule_cost_matches_the_production_scorer_exactly():
   243	    """Lock: the wrapper reports what the production scorer returns — it is
   244	    a passthrough, not a re-derivation (the whole point: a 1-ring lookalike
   245	    reports 8 rounds where the real depth-3 graph reports 12-14)."""
   246	    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
   247	
   248	    mesh = _mesh()
   249	    ref = spmd_schedule_cost(mesh, 2, method="sfc")
   250	    sc = mod.schedule_cost_row(mesh, "sfc", 2)
   251	    for key in ("n_rounds", "max_degree", "n_rounds_greedy",
   252	                "coloring_method", "resolved_method", "cells_per_device",
   253	                "production_strategy"):
   254	        assert sc[key] == ref[key], key
   255	
   256	
   257	def test_schedule_cost_flag_is_opt_in_and_lands_in_the_json(
   258	        tmp_path, monkeypatch):
   259	    """Off by default (it is the expensive layer); on, every scored row
   260	    carries the schedule block and the run records that it ran."""
   261	    out = tmp_path / "off.json"
   262	    monkeypatch.setattr(sys, "argv", [
   263	        "bench", "--subdivision", "2", "--rank-counts", "2",
   264	        "--methods", "geometric", "--out", str(out)])
   265	    assert mod.main() == 0
   266	    payload = json.loads(out.read_text())
   267	    assert "schedule" not in payload["rows"][0]
   268	    assert payload["metadata"]["extra"]["schedule_cost"] is False
   269	
   270	    out2 = tmp_path / "on.json"
   271	    monkeypatch.setattr(sys, "argv", [
   272	        "bench", "--subdivision", "2", "--rank-counts", "2",
   273	        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
   274	    assert mod.main() == 0
   275	    payload2 = json.loads(out2.read_text())
   276	    row = payload2["rows"][0]
   277	    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
   278	    assert payload2["metadata"]["extra"]["schedule_cost"] is True
   279	    # Scorer provenance per row: a copied row must show it scored a mesh
   280	    # partitioned for THIS device count, not one reordered for another.
   281	    assert row["schedule"]["reorder_target"] == row["n_ranks"]
   282	    assert row["schedule"]["already_reordered"] is False
   283	
   284	
   285	def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
   286	    """``--lloyd`` must actually select the mesh, not just be recorded.
   287	
   288	    Non-vacuity: asserting only the recorded default (50) passes even if the
   289	    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
   290	    that.  This spies on the builder, so dropping ``lloyd_iterations=
   291	    args.lloyd`` fails here, and it checks a NON-default value so the
   292	    assertion cannot be satisfied by the default.
   293	    """
   294	    import legoesm.grids.voronoi as vor
   295	
   296	    seen = {}
   297	    real = vor.create_voronoi_mesh
   298	
   299	    def spy(*a, **k):
   300	        seen.update(k)
   301	        # lloyd=0 is cheap and is what the scaling meshes actually use.
   302	        return real(*a, **k)
   303	
   304	    monkeypatch.setattr(vor, "create_voronoi_mesh", spy)
   305	    out = tmp_path / "lloyd0.json"
   306	    monkeypatch.setattr(sys, "argv", [
   307	        "bench", "--subdivision", "2", "--rank-counts", "2",
   308	        "--methods", "geometric", "--lloyd", "0", "--out", str(out)])
   309	    assert mod.main() == 0
   310	    assert seen.get("lloyd_iterations") == 0, seen
   311	    # And it is recorded, so a synthetic mesh cannot be read back as a
   312	    # production SCVT receipt.
   313	    payload = json.loads(out.read_text())
   314	    assert payload["metadata"]["extra"]["lloyd_iterations"] == 0
   315	
   316	
   317	def test_expect_rounds_gate_fails_loudly_and_never_vacuously(
   318	        tmp_path, monkeypatch):
   319	    """The instrument check must FAIL on a wrong expectation and on a pair
   320	    that was never scored — a gate that can only pass is not a gate."""
   321	    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
   322	            "--methods", "geometric", "--schedule-cost"]
   323	
   324	    # Truth first: read what this configuration really scores.
   325	    out = tmp_path / "truth.json"
   326	    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
   327	    assert mod.main() == 0
   328	    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
   329	
   330	    # Matching expectation -> pass, and the check is recorded.
   331	    ok = tmp_path / "ok.json"
   332	    monkeypatch.setattr(sys, "argv", base + [
   333	        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
   334	    assert mod.main() == 0
   335	    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
   336	
   337	    # Wrong expectation -> non-zero exit.
   338	    bad = tmp_path / "bad.json"
   339	    monkeypatch.setattr(sys, "argv", base + [
   340	        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
   341	    assert mod.main() == 1
   342	    assert json.loads(bad.read_text())["expected_rounds_check"]["failures"]
   343	
   344	    # A pair that was never scored is a FAILURE, not a silent skip —
   345	    # otherwise a sweep missing a method still reports a clean gate.
   346	    missing = tmp_path / "missing.json"
   347	    monkeypatch.setattr(sys, "argv", base + [
   348	        "--expect-rounds", "sfc:2=3", "--out", str(missing)])
   349	    assert mod.main() == 1
   350	    fails = json.loads(missing.read_text())["expected_rounds_check"]["failures"]
   351	    assert any("NOT SCORED" in f for f in fails), fails
   352	
   353	
   354	def test_expect_rounds_refuses_to_pass_vacuously_without_scoring(monkeypatch):
   355	    """Without --schedule-cost nothing is scored, so the gate would pass on
   356	    an empty comparison. It must refuse instead."""
   357	    monkeypatch.setattr(sys, "argv", [
   358	        "bench", "--subdivision", "2", "--rank-counts", "2",
   359	        "--methods", "geometric", "--expect-rounds", "geometric:2=1",
   360	        "--out", "/dev/null"])

 succeeded in 311ms:
diff --git a/tests/bench/test_bench_voronoi_partition_methods.py b/tests/bench/test_bench_voronoi_partition_methods.py
index 21f861fd7..ff4d93433 100644
--- a/tests/bench/test_bench_voronoi_partition_methods.py
+++ b/tests/bench/test_bench_voronoi_partition_methods.py
@@ -146,3 +146,302 @@ def test_halo_matches_runtime_partition():
         halos.append(int(part.n_local_cells) - int(part.n_owned_cells))
     assert q["halo_cells_max"] == max(halos)
     assert q["halo_cells_mean"] == pytest.approx(float(np.mean(halos)))
+
+
+# --- SPMD halo-schedule depth (--schedule-cost) ---------------------------
+
+
+def test_schedule_cost_row_reports_rounds_against_their_lower_bound():
+    """``schedule_cost_row`` returns the REAL schedule depth and the bound it
+    must be read against."""
+    mesh = _mesh()
+    sc = mod.schedule_cost_row(mesh, "geometric", 2)
+    # max_degree is the graph's own lower bound on a proper edge colouring,
+    # so a schedule can never beat it.  -1 is the scorer's "not reported"
+    # sentinel and would make the gap meaningless.
+    assert sc["max_degree"] >= 1
+    assert sc["n_rounds"] >= sc["max_degree"]
+    assert sc["coloring_gap"] == sc["n_rounds"] - sc["max_degree"]
+    assert sc["n_rounds_greedy"] >= sc["n_rounds"]
+    assert sc["score_seconds"] >= 0.0
+
+
+@pytest.mark.parametrize(
+    "n_rounds, max_degree, gap, hmin, hmax, proven",
+    [
+        # Provably optimal: no proper edge colouring beats max_degree.
+        (12, 12, 0, 0, 0, True),
+        # Vizing allows the true optimum to BE max_degree+1, so a gap of 1
+        # spans [0, 1] — inconclusive.  This is the case that a naive
+        # "gap > 0 means recolour" rule would over-claim.
+        (13, 12, 1, 0, 1, False),
+        # Guaranteed to remove at least gap-1 = 3, at most gap = 4.
+        (14, 10, 4, 3, 4, False),
+    ])
+def test_coloring_gap_and_headroom_are_derived_not_assumed(
+        monkeypatch, n_rounds, max_degree, gap, hmin, hmax, proven):
+    """Gap and the Vizing-bounded headroom must be COMPUTED, not assumed.
+
+    Non-vacuity, the hard way: on every mesh small enough to test quickly the
+    real gap is 0 (measured L2/L3/L4 x {geometric,sfc} x nd 2-16, and s6
+    lloyd=0 at np8/np16 — the colourer lands exactly on ``max_degree`` every
+    time), so a real-mesh assertion cannot tell a correct subtraction from a
+    hardcoded ``0``; that exact mutation passed the first version of this
+    test.  Stubbing the production scorer with KNOWN values is what makes
+    the assertion able to fail.
+
+    The ``gap == 1`` row is the one that matters: the schedule is a proper
+    EDGE colouring and ``max_degree`` is that graph's max vertex degree, so
+    Vizing gives ``Delta <= chi' <= Delta + 1``.  A gap of 1 is therefore
+    indistinguishable from optimal (Class 2), and claiming recolouring
+    headroom there would be an over-claim.
+    """
+    stub = {
+        "n_rounds": n_rounds, "max_degree": max_degree,
+        "n_rounds_greedy": n_rounds + 1, "coloring_method": "stub",
+        "resolved_method": "geometric", "halo_depth": 3,
+        "cells_per_device": 99_999, "production_strategy": "ppermute",
+        "reorder_target": 8, "already_reordered": False,
+    }
+    import legoesm.parallel.sharded_dynamics as sd
+    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
+
+    sc = mod.schedule_cost_row(object(), "geometric", 8)
+    assert sc["coloring_gap"] == gap
+    # The headroom is an INTERVAL: Vizing pins the optimum to
+    # {Delta, Delta+1}, so gap-1 is guaranteed and gap is the best case.
+    assert sc["coloring_headroom_rounds_min"] == hmin
+    assert sc["coloring_headroom_rounds_max"] == hmax
+    assert sc["coloring_optimal_proven"] is proven
+
+
+def test_schedule_rounds_never_beat_the_vizing_floor_on_a_real_mesh():
+    """Real-mesh sanity on the bound itself: a proper edge colouring can
+    never use fewer rounds than the graph's max degree, and the multi-start
+    search should not overshoot Vizing's ``Delta + 1`` either.  If this ever
+    fires, ``max_degree`` is not the degree of the graph being coloured and
+    every gap-based conclusion built on it is void."""
+    for method in ("geometric", "sfc"):
+        for n_ranks in (2, 4, 8):
+            sc = mod.schedule_cost_row(_mesh(level=3), method, n_ranks)
+            assert sc["max_degree"] <= sc["n_rounds"] <= sc["max_degree"] + 1, (
+                f"{method} np={n_ranks}: rounds={sc['n_rounds']} "
+                f"max_degree={sc['max_degree']}")
+
+
+def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
+    """The schedule is scored at the SPMD production halo depth, NOT this
+    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
+    a different graph and quietly report the wrong lane's cost."""
+    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH
+
+    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
+    assert sc["schedule_halo_depth"] == SPMD_HALO_DEPTH
+
+
+def test_schedule_cost_matches_the_production_scorer_exactly():
+    """Lock: the wrapper reports what the production scorer returns — it is
+    a passthrough, not a re-derivation (the whole point: a 1-ring lookalike
+    reports 8 rounds where the real depth-3 graph reports 12-14)."""
+    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
+
+    mesh = _mesh()
+    ref = spmd_schedule_cost(mesh, 2, method="sfc")
+    sc = mod.schedule_cost_row(mesh, "sfc", 2)
+    for key in ("n_rounds", "max_degree", "n_rounds_greedy",
+                "coloring_method", "resolved_method", "cells_per_device",
+                "production_strategy"):
+        assert sc[key] == ref[key], key
+
+
+def test_schedule_cost_flag_is_opt_in_and_lands_in_the_json(
+        tmp_path, monkeypatch):
+    """Off by default (it is the expensive layer); on, every scored row
+    carries the schedule block and the run records that it ran."""
+    out = tmp_path / "off.json"
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", "geometric", "--out", str(out)])
+    assert mod.main() == 0
+    payload = json.loads(out.read_text())
+    assert "schedule" not in payload["rows"][0]
+    assert payload["metadata"]["extra"]["schedule_cost"] is False
+
+    out2 = tmp_path / "on.json"
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
+    assert mod.main() == 0
+    payload2 = json.loads(out2.read_text())
+    row = payload2["rows"][0]
+    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
+    assert payload2["metadata"]["extra"]["schedule_cost"] is True
+    # Scorer provenance per row: a copied row must show it scored a mesh
+    # partitioned for THIS device count, not one reordered for another.
+    assert row["schedule"]["reorder_target"] == row["n_ranks"]
+    assert row["schedule"]["already_reordered"] is False
+
+
+def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
+    """``--lloyd`` must actually select the mesh, not just be recorded.
+
+    Non-vacuity: asserting only the recorded default (50) passes even if the
+    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
+    that.  This spies on the builder, so dropping ``lloyd_iterations=
+    args.lloyd`` fails here, and it checks a NON-default value so the
+    assertion cannot be satisfied by the default.
+    """
+    import legoesm.grids.voronoi as vor
+
+    seen = {}
+    real = vor.create_voronoi_mesh
+
+    def spy(*a, **k):
+        seen.update(k)
+        # lloyd=0 is cheap and is what the scaling meshes actually use.
+        return real(*a, **k)
+
+    monkeypatch.setattr(vor, "create_voronoi_mesh", spy)
+    out = tmp_path / "lloyd0.json"
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", "geometric", "--lloyd", "0", "--out", str(out)])
+    assert mod.main() == 0
+    assert seen.get("lloyd_iterations") == 0, seen
+    # And it is recorded, so a synthetic mesh cannot be read back as a
+    # production SCVT receipt.
+    payload = json.loads(out.read_text())
+    assert payload["metadata"]["extra"]["lloyd_iterations"] == 0
+
+
+def test_expect_rounds_gate_fails_loudly_and_never_vacuously(
+        tmp_path, monkeypatch):
+    """The instrument check must FAIL on a wrong expectation and on a pair
+    that was never scored — a gate that can only pass is not a gate."""
+    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
+            "--methods", "geometric", "--schedule-cost"]
+
+    # Truth first: read what this configuration really scores.
+    out = tmp_path / "truth.json"
+    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
+    assert mod.main() == 0
+    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
+
+    # Matching expectation -> pass, and the check is recorded.
+    ok = tmp_path / "ok.json"
+    monkeypatch.setattr(sys, "argv", base + [
+        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
+    assert mod.main() == 0
+    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
+
+    # Wrong expectation -> non-zero exit.
+    bad = tmp_path / "bad.json"
+    monkeypatch.setattr(sys, "argv", base + [
+        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
+    assert mod.main() == 1
+    assert json.loads(bad.read_text())["expected_rounds_check"]["failures"]
+
+    # A pair that was never scored is a FAILURE, not a silent skip —
+    # otherwise a sweep missing a method still reports a clean gate.
+    missing = tmp_path / "missing.json"
+    monkeypatch.setattr(sys, "argv", base + [
+        "--expect-rounds", "sfc:2=3", "--out", str(missing)])
+    assert mod.main() == 1
+    fails = json.loads(missing.read_text())["expected_rounds_check"]["failures"]
+    assert any("NOT SCORED" in f for f in fails), fails
+
+
+def test_expect_rounds_refuses_to_pass_vacuously_without_scoring(monkeypatch):
+    """Without --schedule-cost nothing is scored, so the gate would pass on
+    an empty comparison. It must refuse instead."""
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", "geometric", "--expect-rounds", "geometric:2=1",
+        "--out", "/dev/null"])
+    with pytest.raises(SystemExit, match="needs --schedule-cost"):
+        mod.main()
+
+
+@pytest.mark.parametrize("spec", ["geometric:2", "geometric=2", "voodoo:2=3"])
+def test_expect_rounds_rejects_malformed_specs(spec):
+    """A typo'd expectation must raise, never be dropped — a silently
+    skipped expectation turns the gate into a no-op."""
+    with pytest.raises(ValueError, match="expect-rounds"):
+        mod.parse_expect_rounds(spec)
+
+
+def test_expect_rounds_rejects_duplicate_and_empty_specs():
+    """Two more ways the gate could be silently bypassed (codex round 2).
+
+    A duplicate key would let the later expectation overwrite the earlier,
+    so a listed-but-wrong expectation is discarded and the gate passes. A
+    non-empty spec that parses to nothing (``",,,"``) would make main() skip
+    the check entirely while the caller believes it ran.
+    """
+    with pytest.raises(ValueError, match="duplicate expectation"):
+        mod.parse_expect_rounds("geometric:2=999,geometric:2=13")
+    with pytest.raises(ValueError, match="NO expectations"):
+        mod.parse_expect_rounds(",,,")
+    # Whitespace-only is a value the caller PASSED; reading it as "no gate"
+    # is the same bypass (codex round 3 — the first fix guarded on
+    # spec.strip() and let this through).
+    for blank in ("   ", "\t", " , , "):
+        with pytest.raises(ValueError, match="NO expectations"):
+            mod.parse_expect_rounds(blank)
+    # A genuinely empty spec is the "no gate requested" default, not an error.
+    assert mod.parse_expect_rounds("") == {}
+
+
+@pytest.mark.parametrize("flag, value", [
+    ("--lloyd", "-1"), ("--subdivision", "-1"), ("--halo-depth", "-1")])
+def test_negative_numeric_args_refused_as_false_provenance(
+        monkeypatch, flag, value):
+    """All three behave exactly like 0 in the underlying code but would be
+    RECORDED under the negative value — a level-0 mesh filed as 'L-1'."""
+    argv = ["bench", "--subdivision", "2", "--rank-counts", "2",
+            "--methods", "geometric", "--out", "/dev/null"]
+    # Replace the flag if already present, else append.
+    if flag in argv:
+        argv[argv.index(flag) + 1] = value
+    else:
+        argv += [flag, value]
+    monkeypatch.setattr(sys, "argv", argv)
+    with pytest.raises(SystemExit, match=r"must be >= 0"):
+        mod.main()
+
+
+def test_expect_rounds_fails_when_a_method_is_unavailable(
+        tmp_path, monkeypatch):
+    """An expectation naming a method that reported UNAVAILABLE must fail.
+
+    Distinct from omitting the method from --methods: here the sweep asks
+    for it and the partitioner is missing, which is exactly how a two-method
+    table gets misread as a three-method one.
+    """
+    monkeypatch.setattr(mod, "method_available",
+                        lambda m: m != "metis")
+    out = tmp_path / "unavail.json"
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", "geometric,metis", "--schedule-cost",
+        "--expect-rounds", "metis:2=1", "--out", str(out)])
+    assert mod.main() == 1
+    payload = json.loads(out.read_text())
+    fails = payload["expected_rounds_check"]["failures"]
+    assert any("NOT SCORED" in f for f in fails), fails
+    # Assert the EXPLICIT unavailable row, not just the failure: without
+    # this the test would also pass if metis were silently omitted, which
+    # is the very substitution this bench refuses to make.
+    metis_rows = [r for r in payload["rows"] if r["method"] == "metis"]
+    assert metis_rows and metis_rows[0]["available"] is False, payload["rows"]
+
+
+def test_lloyd_rejects_negative(monkeypatch):
+    """A negative Lloyd count behaves like 0 in the builder (it relaxes only
+    for > 0) but is recorded and cached under its own key — false
+    provenance, so the CLI must refuse it rather than run."""
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", "geometric", "--lloyd", "-1", "--out", "/dev/null"])
+    with pytest.raises(SystemExit, match=r"--lloyd must be >= 0"):
+        mod.main()

 exited 2 in 313ms:
# Shared environment for DKRZ Levante GPU scaling jobs (sourced by the SLURM
# scripts here).  SLURM/OpenMPI twin of scripts/cluster/scaling_derecho/_env.sh
# (which is PBS/Cray-MPICH).
# ---------------------------------------------------------------------------
# EDIT the marked values (account / repo / conda env / module versions) before
# the first submit.  Everything is overridable from the sbatch environment, e.g.
#   sbatch --export=ALL,LEGOESM_CONDA_ENV=my-jax-env gpu_moist_scaling.slurm
#
# Levante GPU partition (partition `gpu`): 60 nodes, each 2x AMD EPYC 7763 +
# 4x NVIDIA A100 (56 nodes 80GB, 4 nodes 40GB), InfiniBand HDR200.  MPI stack is
# OpenMPI over UCX with CUDA-aware transports -- NOT Cray MPICH.
# ---------------------------------------------------------------------------

# --- (1) Project allocation (matches SBATCH --account in the job scripts) -----
#     Levante GPU jobs bill a *_gpu sub-account (run_levante_gpu_scaling.sh uses
#     bd1083_gpu, matching the SBATCH --account in the .slurm, which is the
#     source of truth).  This default is only for interactive sourcing.
export LEGOESM_SLURM_ACCOUNT="${LEGOESM_SLURM_ACCOUNT:-bd1083_gpu}"

# --- (2) Repo location on Levante -- EDIT to where you cloned legoESM ---------
# The default is a GUESS at a per-user clone path. When it is wrong the job
# does not fail here — it fails ~60 lines later with a bare
# "cd: <path>: No such file or directory" plus "_chain_body.sh: No such file",
# 7 seconds in, which reads like a broken launcher rather than an unset
# variable (three U-Cast arms lost this way, 2026-08-01). Say it plainly.
REPO="${LEGOESM_REPO:-/work/bd1083/$USER/legoESM}"
if [ ! -d "$REPO" ]; then
  echo "[_env.sh] REPO='$REPO' does not exist." >&2
  if [ -z "${LEGOESM_REPO:-}" ]; then
    echo "[_env.sh] LEGOESM_REPO is unset, so this is the per-user DEFAULT" >&2
    echo "[_env.sh] guess, not a configured path. Submit with" >&2
    echo "[_env.sh]   sbatch --export=ALL,LEGOESM_REPO=\$PWD,... " >&2
    echo "[_env.sh] (--export=ALL alone does NOT carry it if your shell" >&2
    echo "[_env.sh]  never exported it)." >&2
  fi
  exit 1
fi
export REPO

# --- (3) Conda env with a CUDA jaxlib AND a CUDA-aware mpi4jax (see README) ---
CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"

# --- Federation PYTHONPATH (belt-and-braces; `pip install -e .` makes it
#     redundant but harmless) ------------------------------------------------
PP="$REPO/src"
for p in atmosphere core coupler ice land ml ocean tools; do
  PP="$PP:$REPO/packages/$p"
done
export PYTHONPATH="$PP:${PYTHONPATH:-}"

# --- Modules + conda -- EDIT the module versions to the Levante stack you built
#     mpi4py / mpi4jax against (README Step 1); pinned versions matter because
#     the runtime libmpi ABI must match the build ABI ------------------------
module load python3 2>/dev/null || true      # EDIT: e.g. python3/2023.01-gcc-11.2.0
module load openmpi 2>/dev/null || true       # EDIT: the CUDA-aware openmpi you built against
module load cuda    2>/dev/null || true       # EDIT: matching cuda toolkit
if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV" 2>/dev/null || true
fi
# Prefer the repo's own uv venv when it exists — that is the interpreter every
# dev/test workflow uses, and the bare `python` on a Levante compute node has
# no jax (three U-Cast arms died at `import jax` inside 7 s, 2026-08-02).
if [ -z "${LEGOESM_PYTHON:-}" ] && [ -x "$REPO/.venv/bin/python" ]; then
  LEGOESM_PYTHON="$REPO/.venv/bin/python"
fi
PY="${LEGOESM_PYTHON:-$(command -v python)}"
export PY
# Fail at source time, not 4 GPU-hours in: the launcher's first real work is
# `$PY scripts/run/run_aimip.py`, which imports jax immediately.
if ! "$PY" -c "import jax" >/dev/null 2>&1; then
  echo "[_env.sh] PY='$PY' cannot import jax." >&2
  echo "[_env.sh] Set LEGOESM_PYTHON=<repo>/.venv/bin/python (uv venv) or" >&2
  echo "[_env.sh] LEGOESM_CONDA_ENV=<env with a CUDA jaxlib>." >&2
  exit 1
fi

# --- JAX / runtime knobs -----------------------------------------------------
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export MPI4JAX_NO_WARN_JAX_VERSION=1
export MPLBACKEND="${MPLBACKEND:-Agg}"          # headless plotting
# DKRZ scratch is /scratch/<first-letter-of-user>/<user>.
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
# Persistent JIT cache reuses compiles across runs; set empty to force a cold
# compile (true compile_time_s).  On SCRATCH so it survives between jobs.
export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-$SCRATCH/legoesm_jit_cache}"

# --- OpenMPI + UCX CUDA-aware fabric (GPU route-A) ---------------------------
# Route-A hands the on-device sendrecv buffer straight to MPI (the whole point:
# no device->host->device staging, which would erase multi-GPU scaling).  On
# Levante that path is OpenMPI-over-UCX; the pml/osc + UCX transports below turn
# on GPU-direct: cuda_copy + cuda_ipc intra-node, gdr_copy over InfiniBand HDR
# inter-node.  Requires a CUDA-aware mpi4jax (README) + MPI4JAX_USE_CUDA_MPI=1
# (set in the job script).  UCX_MEMTYPE_CACHE=n avoids a stale device/host
# memtype-cache hang that CUDA-aware sendrecv is prone to.
export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
export OMPI_MCA_osc="${OMPI_MCA_osc:-ucx}"
export UCX_TLS="${UCX_TLS:-rc,cuda_copy,cuda_ipc,gdr_copy,sm,self}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export UCX_RNDV_SCHEME="${UCX_RNDV_SCHEME:-put_zcopy}"

# --- NCCL over InfiniBand (route-B: jax.distributed multi-node lanes) --------
# NCCL (shard_map/ppermute collectives under jax.distributed) uses its own
# IB-verbs stack — independent of the UCX/MPI settings above; the two configs
# coexist. Bootstrap ring runs over IPoIB: verify the interface name once with
# `ip addr` on a gpu node (a wrong NCCL_SOCKET_IFNAME is the #1 cause of
# multi-node NCCL bootstrap timeouts on IB clusters).
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
# Prefix-match BOTH HCAs (mlx5_0/mlx5_1 — one per socket on Levante nodes).
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
# GPUDirect RDMA when NIC and GPU share a NUMA/PCIe root.
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"

# --- XLA overlap defaults for the lat-lon SPMD lanes (2026-08-04) --------
# Latency-hiding scheduler + pipelined p2p: -8.4% at LL2048@64 (job
# 26677602) and -8.3% at @128 (26677668), A/A2 drift 0.3-0.4% — twice-
# reproduced, parity suites green with flags on. MPAS lane: null (0.0%,
# 26677669 — its edge-coloured schedule does not benefit; harmless).
# Below the pre-registered 10% bar AND other lanes are unvalidated
# (cube_tiled_step.sbatch force-disables latency hiding for a known
# comm-init sensitivity; MPAS is null) — so this is strictly OPT-IN
# (codex r23): set LEGOESM_XLA_OVERLAP=1 in validated lat-lon
# launchers; never a shared default, and A/B control arms must keep
# REPLACING XLA_FLAGS, not appending.
if [ "${LEGOESM_XLA_OVERLAP:-0}" = 1 ]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_p2p=true"
fi

export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
mkdir -p "$TMPDIR" 2>/dev/null || true

# #1361 memory preflight: target device whose HBM the benches gate against
# (`--device-hbm`). Set in the SHARED env so the gate is on for every launcher
# that sources this file — codex found the Derecho-only export left every
# Levante bench ungated. Levante's GPU jobs request `--constraint=a100_80`.
export LEGOESM_DEVICE_HBM="${LEGOESM_DEVICE_HBM:-a100-80}"
rg: legoesm/parallel: IO error for operation on legoesm/parallel: No such file or directory (os error 2)

codex
The whitespace and negative-value checks are present and behave as claimed. I’ve also found one new empty-input path and am checking whether it can yield a misleading “successful” artifact, plus whether the batch test’s hook creates an integrity risk.
exec
/usr/bin/bash -lc "rg --files | rg 'sharded_dynamics\\.py"'$|voronoi_partition'"\\.py"'$|test_spmd_schedule_cost'"\\.py"'$'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc 'rg -n "def spmd_schedule_cost|SPMD_HALO_DEPTH|_build_ppermute_schedule|production_strategy|cells_per_device" packages src tests/parallel/test_spmd_schedule_cost.py 2>/dev/null' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc 'bash -n scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && bash -n scripts/cluster/scaling_levante/_env.sh && PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/bench/test_bench_voronoi_partition_methods.py tests/bench/test_mpas_schedule_cost_scan_sbatch.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "git diff -- scripts/bench/bench_voronoi_partition_methods.py | sed -n '1,600p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 375ms:
tests/parallel/test_spmd_schedule_cost.py
packages/core/legoesm/parallel/voronoi_partition.py
packages/core/legoesm/parallel/sharded_dynamics.py
tests/unit/test_sharded_dynamics.py
tests/unit/test_scale_sharded_dynamics.py

 succeeded in 369ms:
diff --git a/scripts/bench/bench_voronoi_partition_methods.py b/scripts/bench/bench_voronoi_partition_methods.py
index 6aa0f4a40..3f7e61ec5 100644
--- a/scripts/bench/bench_voronoi_partition_methods.py
+++ b/scripts/bench/bench_voronoi_partition_methods.py
@@ -8,6 +8,54 @@ Scaling-audit item 8.  Two independent measurement layers:
    for each method x rank-count on the real icosahedral mesh.  These are
    the numbers ``resolve_partition_method``'s ``auto`` policy must be
    justified by.
+1b. **SPMD halo-schedule depth** (``--schedule-cost``, opt-in because it is
+   the expensive layer): ``n_rounds`` — the number of SEQUENTIAL ppermute
+   rounds one halo fill costs — and ``max_degree``, the communication
+   graph's lower bound on it.  This is the term that binds MPAS GPU strong
+   scaling above ~64 devices, and layer 1 CANNOT stand in for it: the
+   neighbor fan-out above is a 1-ring proxy that reported 8 rounds for
+   every method and rank count while the real depth-3-plus-closure
+   schedule reported 12-14.  Scored by the production
+   ``spmd_schedule_cost`` (which calls the production builders), never a
+   re-derived lookalike.
+
+   The DECISIVE column is ``coloring_gap = n_rounds - max_degree``, read
+   through VIZING'S THEOREM, which bounds what recolouring could ever buy.
+   The schedule is a proper EDGE colouring of the device communication
+   graph (one colour = one ppermute round; ``_build_ppermute_schedule``
+   asserts properness), and ``max_degree`` is that same graph's maximum
+   vertex degree.  So the chromatic index obeys ``Delta <= chi' <=
+   Delta + 1``: the gap is a bound on recolouring headroom, NOT a
+   yes/no flag.
+
+   * ``gap == 0`` -> ``n_rounds == Delta``, and no proper edge colouring
+     can beat ``Delta``.  The colouring is PROVABLY OPTIMAL; recolouring
+     headroom is exactly ZERO.
+   * ``gap == 1`` -> INCONCLUSIVE.  A Class 2 graph genuinely needs
+     ``Delta + 1``, and deciding Class 1 vs Class 2 is NP-complete, so
+     this neither establishes nor excludes a one-round win.
+   * ``gap >= 2`` -> recolouring is guaranteed to remove AT LEAST
+     ``gap - 1`` rounds (the optimum is at worst ``Delta + 1``) and at
+     most ``gap``.
+
+   SCOPE, and it is not a formality: ``Delta`` bounds only a proper
+   UNDIRECTED edge colouring of THIS graph.  ``_build_ppermute_schedule``
+   enters a device pair into ``comm_pairs`` when EITHER direction has a
+   halo dependency and then emits BOTH ppermute directions, even where one
+   send map is empty.  A redesigned DIRECTED schedule that exploits
+   one-way exchanges is therefore not bounded by ``Delta`` at all, so
+   ``gap == 0`` must never be reported as "only ownership can help" — it
+   rules out a better undirected edge colouring of this graph, and
+   nothing more.
+
+   NOTE the halo depth differs by lane and is NOT ``--halo-depth``: that
+   flag is the MPI lane's (default 2), while the schedule is scored at the
+   SPMD production depth ``SPMD_HALO_DEPTH`` (3).  Both are recorded.
+   ``n_rounds`` is per HALO FILL, not per step — multiply by the tendency
+   evaluations of the integrator actually run.  When
+   ``production_strategy`` is ``"allgather"`` (auto-selected below the
+   cells/device threshold) there is no ppermute schedule in production and
+   the round count is COUNTERFACTUAL; the row says so.
 2. **Step time** (optional pointer, NOT run here): drive the existing
    MPI lane with ``bench_ocean_mpas_scaling.py --partition-method <m>``
    (ocean) or ``bench_mpas_spmd_scaling.py --partition-method <m>``
@@ -23,6 +71,11 @@ exactly one rank; owner range valid) before any metric is recorded.
 Run:
   python scripts/bench/bench_voronoi_partition_methods.py \
       --subdivision 6 --rank-counts 2,4,8,16 --out results/partition_quality.json
+
+  # + the SPMD schedule depth (minutes to hours at subdiv>=8 — batch it):
+  python scripts/bench/bench_voronoi_partition_methods.py \
+      --subdivision 9 --rank-counts 64,128 --schedule-cost \
+      --out results/a1/schedule_cost_s9.json
 """
 from __future__ import annotations
 
@@ -128,6 +181,150 @@ def owner_for(mesh, method: str, n_ranks: int) -> np.ndarray:
                      f"expected one of {METHODS}")
 
 
+def parse_expect_rounds(spec: str) -> dict:
+    """Parse ``'sfc:64=12,metis:128=19'`` into ``{("sfc", 64): 12}``.
+
+    Raises on anything malformed rather than skipping it — a typo'd
+    expectation that is silently dropped turns the gate into a no-op, which
+    is exactly the failure this flag exists to prevent.  Three ways that
+    could happen, all rejected here:
+
+    * a malformed item (``geometric:2``, ``geometric=2``);
+    * an unknown method;
+    * a DUPLICATE key — ``geometric:2=999,geometric:2=13`` would otherwise
+      let the second silently overwrite the first and pass;
+    * a non-empty spec that parses to NOTHING (``",,,"``), which would make
+      the caller skip the check while believing it ran.
+    """
+    out: dict[tuple[str, int], int] = {}
+    for item in (s.strip() for s in spec.split(",")):
+        if not item:
+            continue
+        try:
+            lhs, rounds = item.split("=")
+            method, n_ranks = lhs.split(":")
+            key = (method.strip(), int(n_ranks))
+            value = int(rounds)
+        except ValueError as exc:
+            raise ValueError(
+                f"--expect-rounds: cannot parse {item!r}; expected "
+                f"'<method>:<n_ranks>=<n_rounds>'") from exc
+        if key[0] not in METHODS:
+            raise ValueError(
+                f"--expect-rounds: unknown method {key[0]!r} in {item!r}; "
+                f"expected one of {METHODS}")
+        if key in out:
+            raise ValueError(
+                f"--expect-rounds: duplicate expectation for "
+                f"{key[0]}:{key[1]} ({out[key]} then {value}); the later one "
+                f"would silently overwrite the earlier and the gate would "
+                f"pass while discarding a listed expectation.")
+        out[key] = value
+    # Guard on `spec`, NOT `spec.strip()`: a whitespace-only value is a value
+    # the caller PASSED, and silently reading it as "no gate requested" is
+    # the same bypass as ",,," (codex round 3).  Only the default empty
+    # string means "no gate".
+    if spec and not out:
+        raise ValueError(
+            f"--expect-rounds={spec!r} parses to NO expectations; the gate "
+            f"would be skipped while looking like it ran.")
+    return out
+
+
+def check_expected_rounds(rows: list, expect: dict) -> list:
+    """Compare scored rounds against *expect*; return failure strings.
+
+    A listed pair that was never scored is a FAILURE, not a skip: otherwise
+    a sweep that silently dropped a method (unavailable ``pymetis``) or a
+    rank count would still report a clean gate.
+    """
+    scored = {
+        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
+        for r in rows
+        if r.get("available") and "schedule" in r and "n_ranks" in r
+    }
+    failures = []
+    for (method, n_ranks), want in sorted(expect.items()):
+        got = scored.get((method, n_ranks))
+        if got is None:
+            failures.append(
+                f"{method}:{n_ranks} expected rounds={want} but the pair was "
+                f"NOT SCORED (method unavailable, or not in this sweep)")
+        elif got != want:
+            failures.append(
+                f"{method}:{n_ranks} expected rounds={want}, got {got}")
+    return failures
+
+
+def schedule_cost_row(mesh, method: str, n_ranks: int) -> dict:
+    """SPMD halo-schedule depth for one (method, n_ranks) candidate.
+
+    Thin wrapper over the production
+    :func:`legoesm.parallel.sharded_dynamics.spmd_schedule_cost` — it reorders
+    the RAW mesh for ``n_ranks`` with ``method`` and colours the real
+    depth-``SPMD_HALO_DEPTH`` communication graph, so the number is the one
+    production pays, not a 1-ring lookalike.  ``halo_depth`` is deliberately
+    LEFT AT THE SCORER'S DEFAULT: this script's ``--halo-depth`` is the MPI
+    lane's (2), and scoring the SPMD schedule at 2 would colour a different
+    graph.
+
+    Adds ``coloring_gap = n_rounds - max_degree`` and the Vizing reading of
+    it (see the module docstring).  The headroom is reported as an INTERVAL,
+    because Vizing pins the optimum only to ``{Delta, Delta + 1}``:
+
+    * ``coloring_headroom_rounds_min = max(0, gap - 1)`` — rounds a perfect
+      recolouring is GUARANTEED to remove (it beats the ``Delta + 1`` case).
+    * ``coloring_headroom_rounds_max = gap`` — the best case, realized only
+      if the graph is Class 1.
+
+    It is deliberately NOT a "recolour vs ownership" verdict: at
+    ``gap == 1`` the min is 0 and the max is 1, i.e. genuinely inconclusive.
+
+    Errors are NOT caught.  The scorer's one refusal — a mesh padded for a
+    different reorder target, which would mis-slice the owned blocks — is
+    unreachable from here: this passes the raw mesh with the scorer's default
+    ``reorder_target = n_ranks``, and ``reorder_voronoi_for_sharding`` pads
+    ``nCells``/``nEdges`` to be divisible by exactly that target.  Wrapping
+    the call would therefore only swallow *unforeseen* failures into a row
+    that reads like an orderly skip, which is how a missing number turns into
+    a silently wrong table.  Rows already print as the sweep goes, so a raise
+    keeps the completed rungs in the log.
+    """
+    import time
+
+    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
+
+    t0 = time.perf_counter()
+    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
+    gap = int(cost["n_rounds"]) - int(cost["max_degree"])
+    return {
+        "n_rounds": int(cost["n_rounds"]),
+        "n_rounds_greedy": int(cost["n_rounds_greedy"]),
+        "max_degree": int(cost["max_degree"]),
+        # The decisive column, read through Vizing (see module docstring).
+        "coloring_gap": gap,
+        # Rounds a perfect recolouring could remove, as an INTERVAL: Vizing
+        # pins the optimum to {Delta, Delta+1}, so gap-1 is guaranteed and
+        # gap is the best case.  A gap of 1 spans [0, 1] = inconclusive.
+        "coloring_headroom_rounds_min": max(0, gap - 1),
+        "coloring_headroom_rounds_max": gap,
+        "coloring_optimal_proven": gap == 0,
+        # Scorer provenance, carried per row: a copied/flattened row must be
+        # able to show it scored a raw mesh partitioned for THIS device
+        # count, not one reordered for a different target.
+        "reorder_target": cost["reorder_target"],
+        "already_reordered": cost["already_reordered"],
+        "coloring_method": cost["coloring_method"],
+        "resolved_method": cost["resolved_method"],
+        "schedule_halo_depth": int(cost["halo_depth"]),
+        "cells_per_device": int(cost["cells_per_device"]),
+        # "allgather" => production runs no ppermute schedule here, so the
+        # round count above is COUNTERFACTUAL, not a cost production pays.
+        "production_strategy": cost["production_strategy"],
+        "score_seconds": round(time.perf_counter() - t0, 2),
+    }
+
+
 def main() -> int:
     p = argparse.ArgumentParser(
         description=__doc__,
@@ -138,6 +335,30 @@ def main() -> int:
     p.add_argument("--halo-depth", type=int, default=2,
                    help="Halo layers (runtime default 2, del4 support).")
     p.add_argument("--methods", type=str, default=",".join(METHODS))
+    p.add_argument("--schedule-cost", action="store_true",
+                   help="Also score the SPMD ppermute halo-schedule depth "
+                        "(n_rounds vs max_degree) per method x rank count. "
+                        "Uses the production SPMD halo depth, NOT "
+                        "--halo-depth. Expensive: minutes per candidate at "
+                        "subdiv>=8 — run it under batch.")
+    p.add_argument("--lloyd", type=int, default=50,
+                   help="Lloyd relaxation iterations for the mesh. 50 = the "
+                        "production SCVT key; 0 = the LABELLED synthetic "
+                        "scaling mesh. Recorded so a lloyd=0 mesh can never "
+                        "masquerade as a production receipt, and it must "
+                        "match the prewarmed cache key at subdiv>=9. "
+                        "Must be >= 0: the builder relaxes only when this is "
+                        "> 0, so a negative behaves exactly like 0 while "
+                        "being recorded (and cached) under a different key — "
+                        "false provenance.")
+    p.add_argument("--expect-rounds", type=str, default="",
+                   help="Instrument check, MECHANICAL. Comma-separated "
+                        "'<method>:<n_ranks>=<n_rounds>' expectations (e.g. "
+                        "'sfc:64=12,sfc:128=14'). Every listed pair must be "
+                        "scored and match, or main() returns 1 — so a caller "
+                        "that reproduces a known census can GATE on it "
+                        "instead of asserting agreement in a comment. "
+                        "Requires --schedule-cost.")
     p.add_argument("--out", type=str,
                    default="results/a1/voronoi_partition_quality.json")
     args = p.parse_args()
@@ -149,12 +370,39 @@ def main() -> int:
             raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
     if not rank_counts or any(n < 2 for n in rank_counts):
         raise SystemExit("--rank-counts needs integers >= 2")
+    # Negative values in these three are all the SAME false-provenance bug:
+    # the underlying code treats them exactly like 0 (no relaxation, no
+    # bisection, `range(-1)` is empty), but the run is recorded under the
+    # negative value, so a level-0 mesh gets filed as "L-1".
+    if args.lloyd < 0:
+        raise SystemExit(
+            f"--lloyd must be >= 0, got {args.lloyd}: the mesh builder "
+            f"relaxes only for > 0, so a negative is indistinguishable from "
+            f"0 in the mesh but is recorded and cached under its own key.")
+    if args.subdivision < 0:
+        raise SystemExit(
+            f"--subdivision must be >= 0, got {args.subdivision}: the mesh "
+            f"builder bisects only for > 0, so a negative silently yields "
+            f"the level-0 base mesh while being recorded as "
+            f"L{args.subdivision}.")
+    if args.halo_depth < 0:
+        raise SystemExit(
+            f"--halo-depth must be >= 0, got {args.halo_depth}: the halo "
+            f"loop is `range(depth)`, so a negative behaves exactly like 0 "
+            f"while being recorded as {args.halo_depth}.")
+    expect = parse_expect_rounds(args.expect_rounds)
+    if expect and not args.schedule_cost:
+        raise SystemExit(
+            "--expect-rounds compares scored round counts, so it needs "
+            "--schedule-cost; without it nothing is scored and the gate "
+            "would pass vacuously.")
 
 
     from legoesm.grids.voronoi import create_voronoi_mesh
     from legoesm.parallel.voronoi_partition import resolve_partition_method
 
-    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
+    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
+                               lloyd_iterations=args.lloyd)
     if max(rank_counts) > int(mesh.nCells):
         raise SystemExit(
             f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
@@ -185,6 +433,23 @@ def main() -> int:
                   f"mean={q['halo_cells_mean']:8.1f} | "
                   f"halo/owned max={q['halo_owned_ratio_max']:.3f} | "
                   f"nbrs max={q['neighbor_ranks_max']}")
+            if args.schedule_cost:
+                sc = schedule_cost_row(mesh, method, n_ranks)
+                row["schedule"] = sc
+                note = ("  [COUNTERFACTUAL: production auto-selects "
+                        "allgather here, no ppermute schedule]"
+                        if sc["production_strategy"] == "allgather" else "")
+                verdict = ("colouring PROVABLY OPTIMAL (Vizing floor)"
+                           if sc["coloring_optimal_proven"] else
+                           f"recolour headroom "
+                           f"{sc['coloring_headroom_rounds_min']}-"
+                           f"{sc['coloring_headroom_rounds_max']} round(s)")
+                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
+                      f"rounds={sc['n_rounds']:3d} "
+                      f"max_degree={sc['max_degree']:3d} "
+                      f"gap={sc['coloring_gap']:+d} "
+                      f"-> {verdict} "
+                      f"({sc['score_seconds']:.1f}s){note}", flush=True)
 
     payload = {
         "rows": rows,
@@ -204,7 +469,9 @@ def main() -> int:
             scaling_kind="partition-quality",
             transport="none",
             extra={"rank_counts": rank_counts, "methods": methods,
-                   "halo_depth": args.halo_depth},
+                   "halo_depth": args.halo_depth,
+                   "schedule_cost": bool(args.schedule_cost),
+                   "lloyd_iterations": args.lloyd},
         )),
     }
     outdir = os.path.dirname(args.out)
@@ -213,6 +480,24 @@ def main() -> int:
     with open(args.out, "w") as f:
         json.dump(payload, f, indent=2)
     print(f"JSON: {args.out}")
+
+    if expect:
+        failures = check_expected_rounds(rows, expect)
+        payload["expected_rounds_check"] = {
+            "expected": {f"{m}:{n}": v for (m, n), v in sorted(expect.items())},
+            "failures": failures,
+            "passed": not failures,
+        }
+        with open(args.out, "w") as f:
+            json.dump(payload, f, indent=2)
+        if failures:
+            print(f"INSTRUMENT CHECK FAILED ({len(failures)}):")
+            for line in failures:
+                print(f"  {line}")
+            print("The scorer did NOT reproduce the known census — treat every "
+                  "unknown row in this run as UNTRUSTED.")
+            return 1
+        print(f"INSTRUMENT CHECK PASSED ({len(expect)} expectations)")
     return 0
 
 

 exited 127 in 379ms:
/usr/bin/bash: pytest: command not found

 succeeded in 1223ms:
tests/parallel/test_spmd_schedule_cost.py:55:                              ppermute_cells_per_device_threshold=10**9)
tests/parallel/test_spmd_schedule_cost.py:56:    assert c["production_strategy"] == "allgather"
tests/parallel/test_spmd_schedule_cost.py:58:                                ppermute_cells_per_device_threshold=1)
tests/parallel/test_spmd_schedule_cost.py:59:    assert big["production_strategy"] == "ppermute"
tests/parallel/test_spmd_schedule_cost.py:159:    assert sd.SPMD_HALO_DEPTH == 3
tests/parallel/test_spmd_schedule_cost.py:161:            .parameters["halo_depth"].default == sd.SPMD_HALO_DEPTH)
tests/parallel/test_spmd_schedule_cost.py:163:    assert "halo_depth=SPMD_HALO_DEPTH" in prod, (
tests/parallel/test_spmd_schedule_cost.py:169:    assert sd.spmd_schedule_cost(mesh, 1)["production_strategy"] is None
packages/core/legoesm/parallel/sharded_dynamics.py:1247:SPMD_HALO_DEPTH = 3
packages/core/legoesm/parallel/sharded_dynamics.py:1250:def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
packages/core/legoesm/parallel/sharded_dynamics.py:1251:                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
packages/core/legoesm/parallel/sharded_dynamics.py:1252:                       ppermute_cells_per_device_threshold=2_000):
packages/core/legoesm/parallel/sharded_dynamics.py:1263:    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
packages/core/legoesm/parallel/sharded_dynamics.py:1274:      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
packages/core/legoesm/parallel/sharded_dynamics.py:1279:      cells/device is below ``ppermute_cells_per_device_threshold``, in which
packages/core/legoesm/parallel/sharded_dynamics.py:1281:      see the returned ``production_strategy``.
packages/core/legoesm/parallel/sharded_dynamics.py:1310:    ppermute_cells_per_device_threshold : int
packages/core/legoesm/parallel/sharded_dynamics.py:1312:        ``production_strategy``.
packages/core/legoesm/parallel/sharded_dynamics.py:1320:        ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
packages/core/legoesm/parallel/sharded_dynamics.py:1372:    sched = _build_ppermute_schedule(
packages/core/legoesm/parallel/sharded_dynamics.py:1393:        "production_strategy": (
packages/core/legoesm/parallel/sharded_dynamics.py:1395:            ("allgather" if cells_per < ppermute_cells_per_device_threshold
packages/core/legoesm/parallel/sharded_dynamics.py:1397:        "cells_per_device": cells_per,
packages/core/legoesm/parallel/sharded_dynamics.py:1762:def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
packages/core/legoesm/parallel/sharded_dynamics.py:2117:    ppermute_cells_per_device_threshold: int = 2_000,
packages/core/legoesm/parallel/sharded_dynamics.py:2164:        *ppermute_cells_per_device_threshold*.
packages/core/legoesm/parallel/sharded_dynamics.py:2169:    ppermute_cells_per_device_threshold : int
packages/core/legoesm/parallel/sharded_dynamics.py:2263:        if cells_per < ppermute_cells_per_device_threshold:
packages/core/legoesm/parallel/sharded_dynamics.py:2266:                "Auto-selected allgather strategy: cells_per_device=%d < "
packages/core/legoesm/parallel/sharded_dynamics.py:2268:                cells_per, ppermute_cells_per_device_threshold,
packages/core/legoesm/parallel/sharded_dynamics.py:2273:                "Auto-selected ppermute strategy: cells_per_device=%d >= "
packages/core/legoesm/parallel/sharded_dynamics.py:2275:                cells_per, ppermute_cells_per_device_threshold,
packages/core/legoesm/parallel/sharded_dynamics.py:2298:                                       halo_depth=SPMD_HALO_DEPTH)
packages/core/legoesm/parallel/sharded_dynamics.py:2331:        pp_sched = _build_ppermute_schedule(

exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1230,1410p'; nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1750,1930p'; nl -ba tests/parallel/test_spmd_schedule_cost.py | sed -n '1,210p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/bench/test_bench_voronoi_partition_methods.py tests/bench/test_mpas_schedule_cost_scan_sbatch.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "git log --oneline -5 -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py; rg -n \"mpas_schedule_cost_scan\" -g '"'!scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch'"' . 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/parallel/voronoi_partition.py | sed -n '1,190p'; rg -n \"def create_voronoi_mesh|if .*subdivision|range\\(.*subdivision|lloyd_iterations\" packages/core/legoesm/grids/voronoi.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 300ms:
     1	"""Domain decomposition for Voronoi (MPAS-style) meshes.
     2	
     3	Partitions an unstructured Voronoi mesh across MPI ranks or JAX devices
     4	and constructs local sub-meshes with halo (ghost) entities for parallel
     5	stencil computation.
     6	
     7	Two partitioning methods:
     8	
     9	1. **Geometric (RCB)**: Recursive Coordinate Bisection on cell-center
    10	   Cartesian coordinates.  No external dependencies.
    11	2. **METIS** (optional): k-way graph partitioning via ``pymetis``.
    12	
    13	After partitioning, each rank holds owned + halo entities.  The halo
    14	exchange (:mod:`legoesm.parallel.halo_exchange_voronoi`) updates halo
    15	values from their owning ranks between timesteps.
    16	
    17	Usage
    18	-----
    19	::
    20	
    21	    partition = partition_voronoi_mesh(mesh, n_ranks=4, rank=0)
    22	    local_mesh = build_local_mesh(mesh, partition)
    23	
    24	    # In the time loop, exchange halo data:
    25	    from legoesm.parallel.halo_exchange_voronoi import VoronoiHaloExchange
    26	    halo = VoronoiHaloExchange(partition, backend="mpi")
    27	    h_local = halo.exchange_cell_field(h_local)
    28	"""
    29	
    30	from __future__ import annotations
    31	
    32	from typing import NamedTuple
    33	
    34	import importlib.util
    35	import logging
    36	
    37	import numpy as np
    38	import jax.numpy as jnp
    39	
    40	from legoesm.grids.voronoi import VoronoiMesh
    41	
    42	logger = logging.getLogger("legoesm.parallel.voronoi_partition")
    43	
    44	# One-time log guard so a per-rank/per-call "auto" resolution does not spam.
    45	_AUTO_METHOD_LOGGED = False
    46	
    47	# Hilbert space-filling-curve resolution: a 2^order x 2^order (lat, lon) grid.
    48	# order=10 -> 1024^2 ~ 1.05e6 buckets, finer than any production Voronoi mesh
    49	# (level-9 SCVT ~2.6e6 cells is the practical ceiling; ties break by stable
    50	# sort), so distinct cells almost never collide. Module constant, not config:
    51	# it is a numerics resolution knob, not a tunable.
    52	_DEFAULT_HILBERT_ORDER = 10
    53	
    54	
    55	def _metis_available() -> bool:
    56	    """True if the optional ``pymetis`` graph-partitioning package is importable."""
    57	    return importlib.util.find_spec("pymetis") is not None
    58	
    59	
    60	def resolve_partition_method(method: str) -> str:
    61	    """Resolve a partition method, expanding ``"auto"`` by available capability.
    62	
    63	    ``"auto"`` (the default) selects ``"metis"`` when ``pymetis`` is importable —
    64	    graph partitioning minimizes the edge cut, giving better load balance and
    65	    smaller halos on irregular/variable-resolution meshes (the MPAS lesson:
    66	    geometric RCB leaves lopsided cell counts and fat halos at scale) — and
    67	    otherwise falls back to ``"geometric"`` (RCB, no dependency).
    68	
    69	    ``"geometric"``, ``"metis"``, and any unknown value pass through UNCHANGED so
    70	    the caller's own dispatch guard still raises on an unknown method. Returns the
    71	    concrete method name.
    72	    """
    73	    global _AUTO_METHOD_LOGGED
    74	    if method != "auto":
    75	        return method
    76	    chosen = "metis" if _metis_available() else "geometric"
    77	    if not _AUTO_METHOD_LOGGED:
    78	        _AUTO_METHOD_LOGGED = True
    79	        if chosen == "metis":
    80	            logger.info(
    81	                "Voronoi partition method='auto' -> 'metis' (pymetis available; "
    82	                "graph partitioning for load balance + smaller halos)."
    83	            )
    84	        else:
    85	            logger.info(
    86	                "Voronoi partition method='auto' -> 'geometric' RCB (pymetis not "
    87	                "installed; `pip install pymetis` for better load balance at scale)."
    88	            )
    89	    return chosen
    90	
    91	
    92	# ============================================================================
    93	# Data structures
    94	# ============================================================================
    95	
    96	class HaloCommSchedule(NamedTuple):
    97	    """Communication schedule for halo exchange of one entity type.
    98	
    99	    For neighbor rank ``neighbor_ranks[i]``:
   100	
   101	    - Send ``send_counts[i]`` values starting at cumulative offset in
   102	      ``send_idx``.
   103	    - Recv ``recv_counts[i]`` values starting at cumulative offset in
   104	      ``recv_idx``.
   105	    """
   106	    neighbor_ranks: tuple[int, ...]
   107	    send_counts: tuple[int, ...]
   108	    recv_counts: tuple[int, ...]
   109	    send_idx: jnp.ndarray   # (total_send,) local indices to pack
   110	    recv_idx: jnp.ndarray   # (total_recv,) local indices to fill
   111	
   112	
   113	class BatchedHaloSchedule(NamedTuple):
   114	    """Union-neighbor comm schedule joining the cell + edge index spaces.
   115	
   116	    Built once at layout-build time by :func:`build_batched_halo_schedule`
   117	    from a partition's ``cell_comm`` and ``edge_comm``.  Lets the MPAS
   118	    state exchange send ONE message per neighbor per dtype group (u edges
   119	    + T/p_s cells + tracer cells packed into a single flat buffer) instead
   120	    of one message per neighbor per entity exchange.
   121	
   122	    ``neighbor_ranks`` is the sorted union of the cell and edge neighbor
   123	    lists.  A rank present in only one of the two entity schedules gets
   124	    zero counts for the other entity (zero-length pack segments).  The
   125	    union relation is symmetric across ranks whenever the underlying
   126	    entity schedules are (rank A lists B iff B lists A) — see
   127	    ``tests/distributed/test_voronoi_batched_halo.py`` for the mechanical
   128	    cross-rank check.
   129	
   130	    For union neighbor ``i``:
   131	
   132	    - cell send rows: ``cell_send_idx[sum(cell_send_counts[:i]) : ... +
   133	      cell_send_counts[i]]`` (local OWNED cell indices to pack);
   134	    - cell recv rows: same slicing of ``cell_recv_idx`` (local HALO cell
   135	      indices to fill);
   136	    - edge send/recv rows: identical layout in ``edge_send_idx`` /
   137	      ``edge_recv_idx``.
   138	
   139	    All counts are Python ints (layout constants) so every pack/unpack
   140	    slice has a static shape under JIT.  Concatenating the per-neighbor
   141	    recv rows in union order yields exactly ``cell_recv_idx`` /
   142	    ``edge_recv_idx``, so the unpack can do a single functional scatter
   143	    per field.
   144	    """
   145	    neighbor_ranks: tuple[int, ...]
   146	    cell_send_counts: tuple[int, ...]
   147	    cell_recv_counts: tuple[int, ...]
   148	    edge_send_counts: tuple[int, ...]
   149	    edge_recv_counts: tuple[int, ...]
   150	    cell_send_idx: jnp.ndarray   # (total_cell_send,) local indices to pack
   151	    cell_recv_idx: jnp.ndarray   # (total_cell_recv,) local indices to fill
   152	    edge_send_idx: jnp.ndarray   # (total_edge_send,)
   153	    edge_recv_idx: jnp.ndarray   # (total_edge_recv,)
   154	
   155	    def messages_per_exchange(self, n_dtype_groups: int = 1) -> int:
   156	        """Messages one batched state exchange posts per rank.
   157	
   158	        Pure schedule math for the homogeneous case where every dtype
   159	        group touches both index spaces (the expected production case:
   160	        all prognostic fields share one dtype, so ``n_dtype_groups=1``).
   161	        For heterogeneous groups (e.g. a cell-only dtype group facing an
   162	        edge-only neighbor) the exact count is
   163	        :func:`legoesm.parallel.halo_exchange_voronoi.count_batched_messages`,
   164	        which never exceeds this bound.
   165	        """
   166	        return len(self.neighbor_ranks) * n_dtype_groups
   167	
   168	
   169	class VoronoiPartition(NamedTuple):
   170	    """Domain decomposition descriptor for one rank of a Voronoi mesh.
   171	
   172	    Entities are ordered: owned first (sorted by global index), then
   173	    halo (sorted by global index).
   174	    """
   175	    rank: int
   176	    n_ranks: int
   177	
   178	    # Global counts
   179	    nCells_global: int
   180	    nEdges_global: int
   181	    nVertices_global: int
   182	
   183	    # Owned counts
   184	    n_owned_cells: int
   185	    n_owned_edges: int
   186	    n_owned_vertices: int
   187	
   188	    # Local counts (owned + halo)
   189	    n_local_cells: int
   190	    n_local_edges: int
1095:# DETERMINISTIC in (subdivision_level, radius, lloyd_iterations, omega) when
1143:    subdivision_level: int, radius: float, lloyd_iterations: int, omega: float
1154:        f"_r{radius!r}_omega{omega!r}_lloyd{lloyd_iterations}"
1224:    lloyd_iterations: int = 50,
1236:    path = _voronoi_cache_path(subdivision_level, radius, lloyd_iterations, omega)
1239:        lloyd_iterations=lloyd_iterations, omega=omega)
1243:def create_voronoi_mesh(
1246:    lloyd_iterations: int = 50,
1263:    lloyd_iterations : int
1278:        ``lloyd_iterations``; for strong/precise variable resolution prefer a
1285:    if subdivision_level > _BIG_MESH_MAX_LEVEL:
1308:            subdivision_level, radius, lloyd_iterations, omega)
1362:    if subdivision_level > 0:
1366:    if lloyd_iterations > 0 and subdivision_level > 0:
1368:            verts, n_iter=lloyd_iterations, density_fn=density_fn)
1371:            "density_fn requires lloyd_iterations > 0 and subdivision_level > 0 "

 succeeded in 324ms:
  1230	            np.isin(cellsOnEdge_np[0], cur_local_arr)
  1231	            | np.isin(cellsOnEdge_np[1], cur_local_arr)
  1232	        )
  1233	        cand_edges = np.flatnonzero(edge_one_in)
  1234	        cand_cells = cellsOnEdge_np[:, cand_edges].reshape(-1)
  1235	        cand_cells = np.unique(cand_cells[cand_cells >= 0])
  1236	        # Set difference: cells not yet in local set.
  1237	        new_cells = cand_cells[~np.isin(cand_cells, cur_local_arr)]
  1238	        if new_cells.size == 0:
  1239	            return
  1240	        halo_cells_set.update(new_cells.tolist())
  1241	
  1242	
  1243	#: Halo depth the SPMD Voronoi partition infra is built at.  ONE definition
  1244	#: consumed by both the production step factory and ``spmd_schedule_cost``:
  1245	#: a score computed at a different depth describes a different comm graph, and
  1246	#: two independently hardcoded 3s let production drift unnoticed.
  1247	SPMD_HALO_DEPTH = 3
  1248	
  1249	
  1250	def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
  1251	                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
  1252	                       ppermute_cells_per_device_threshold=2_000):
  1253	    """How much halo communication one ownership choice costs, computed offline.
  1254	
  1255	    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
  1256	    ROUNDS one halo exchange needs -- the sequential collective launches that
  1257	    dominate MPAS strong scaling above ~64 devices.  Runs on a laptop: no GPU,
  1258	    no MPI, no benchmark job, so a split can be compared before it costs an
  1259	    allocation.
  1260	
  1261	    It calls the SAME builders production calls
  1262	    (:func:`_build_voronoi_partition_infra` then
  1263	    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
  1264	    different question: a 1-ring ``cellsOnEdge`` adjacency graph reports 8
  1265	    rounds where the real depth-3-plus-closure graph reports 12-14.
  1266	
  1267	    WHAT THE NUMBER IS NOT
  1268	    ----------------------
  1269	    * ``n_rounds`` is per HALO FILL, not per model step.  A step costs
  1270	      ``n_rounds`` x (tendency evaluations per step), which depends on the
  1271	      configured integrator -- SSP-RK3 evaluates 3 times, but the MPAS default
  1272	      is ``ssp_rk54_scan``.  Multiply with the integrator you actually run.
  1273	    * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
  1274	      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
  1275	      keeps the best; equality is MEASURED (compare the returned
  1276	      ``max_degree``), never assumed.  Do not claim "the colouring is already
  1277	      optimal so only ownership can help" from this function.
  1278	    * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
  1279	      cells/device is below ``ppermute_cells_per_device_threshold``, in which
  1280	      case there is no ppermute schedule and this number is counterfactual --
  1281	      see the returned ``production_strategy``.
  1282	
  1283	    MESH STATE -- the one thing that silently invalidates the score
  1284	    --------------------------------------------------------------
  1285	    Production does NOT reorder inside ``make_voronoi_sharded_step``; it
  1286	    consumes an already-reordered ``model.mesh``.  The scaling bench reorders
  1287	    ONCE for a ``reorder_target`` device count and then runs at a possibly
  1288	    DIFFERENT device count.  So pass what you actually have:
  1289	
  1290	    * raw mesh, scoring a run at ``n_dev``: defaults are right.
  1291	    * raw mesh, but the run reorders for a different target: pass
  1292	      ``reorder_target=<that target>``; the split is built for the target and
  1293	      scored at ``n_dev``.
  1294	    * already-reordered mesh (what production holds): pass
  1295	      ``already_reordered=True``; ``method`` is then ignored and reported as
  1296	      ``"pre-reordered"``, because the ownership is already baked in.
  1297	
  1298	    Parameters
  1299	    ----------
  1300	    mesh : VoronoiMesh
  1301	    n_dev : int
  1302	        Device count the run uses.  Must be >= 1.
  1303	    method : str
  1304	        Ownership for the reorder; ignored when *already_reordered*.
  1305	    reorder_target : int | None
  1306	        Device count the reorder targets, when it differs from *n_dev*.
  1307	    already_reordered : bool
  1308	    halo_depth : int
  1309	        Must match production (3) or the graph is a different graph.
  1310	    ppermute_cells_per_device_threshold : int
  1311	        Mirror of the production auto-select threshold, only used to report
  1312	        ``production_strategy``.
  1313	
  1314	    Returns
  1315	    -------
  1316	    dict
  1317	        ``n_rounds`` (the cost), ``max_degree`` (the lower bound to compare
  1318	        it against), ``n_rounds_greedy``, ``coloring_method``,
  1319	        ``resolved_method`` (concrete, never ``"auto"``),
  1320	        ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
  1321	        ``max_local_cells``, ``max_local_edges``, and the echoed inputs.
  1322	
  1323	    Reference census on the unrelaxed mesh, which any change here must still
  1324	    reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
  1325	    geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
  1326	    """
  1327	    from legoesm.parallel.voronoi_partition import (
  1328	        reorder_voronoi_for_sharding, resolve_sharding_partition_method,
  1329	    )
  1330	
  1331	    if int(n_dev) != n_dev or int(n_dev) < 1:
  1332	        # int() would silently truncate 3.9 -> 3 and score the wrong split.
  1333	        raise ValueError(
  1334	            f"spmd_schedule_cost: n_dev must be an integer >= 1, got {n_dev!r}")
  1335	    n_dev = int(n_dev)
  1336	
  1337	    if already_reordered:
  1338	        if reorder_target is not None:
  1339	            raise ValueError(
  1340	                "spmd_schedule_cost: reorder_target is meaningless with "
  1341	                "already_reordered=True — the ownership is already baked into "
  1342	                "the mesh.")
  1343	        prepared, resolved = mesh, "pre-reordered"
  1344	    else:
  1345	        target = n_dev if reorder_target is None else int(reorder_target)
  1346	        prepared = reorder_voronoi_for_sharding(mesh, target, method=method)
  1347	        # Report the CONCRETE ownership: "auto" hides which partitioner ran.
  1348	        # Uses the SAME resolver the reorder used, so the label cannot drift
  1349	        # from the policy.
  1350	        resolved = resolve_sharding_partition_method(method)
  1351	
  1352	    # The builder assigns residual entities to the LAST owner but excludes them
  1353	    # from every owned contiguous block, so schedule send indices can exceed a
  1354	    # device's shard length -- a number that looks fine and is not.  Reachable
  1355	    # via reorder_target: a mesh padded for 3 devices is not divisible by 4.
  1356	    # The scaling bench rejects that pairing; so does this.
  1357	    n_cells, n_edges = int(prepared.nCells), int(prepared.nEdges)
  1358	    if n_cells % n_dev or n_edges % n_dev:
  1359	        raise ValueError(
  1360	            f"spmd_schedule_cost: prepared mesh has nCells={n_cells}, "
  1361	            f"nEdges={n_edges}, neither divisible by n_dev={n_dev}. The mesh "
  1362	            f"is padded for its reorder target"
  1363	            f"{'' if already_reordered else f' ({target})'}, so scoring it at "
  1364	            f"a device count that does not divide it silently mis-slices the "
  1365	            f"owned blocks. Score at a device count that divides the prepared "
  1366	            f"mesh.")
  1367	    (
  1368	        _stacked, _gc, _ge, _noc, _noe, max_lc, max_le, partitions, cell_owner,
  1369	    ) = _build_voronoi_partition_infra(prepared, n_dev, halo_depth=halo_depth)
  1370	    cells_per = n_cells // n_dev
  1371	    edges_per = n_edges // n_dev
  1372	    sched = _build_ppermute_schedule(
  1373	        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
  1374	    )
  1375	    return {
  1376	        "method": method,
  1377	        "resolved_method": resolved,
  1378	        "n_dev": n_dev,
  1379	        # Unknown for a pre-reordered mesh: the ownership is baked in and the
  1380	        # target that produced it is not recoverable from the mesh. Reporting
  1381	        # n_dev there would assert something we did not verify.
  1382	        "reorder_target": (None if already_reordered else
  1383	                           (n_dev if reorder_target is None
  1384	                            else int(reorder_target))),
  1385	        "already_reordered": bool(already_reordered),
  1386	        "halo_depth": halo_depth,
  1387	        "n_rounds": int(sched["n_rounds"]),
  1388	        "n_rounds_greedy": int(sched["n_rounds_greedy"]),
  1389	        "max_degree": int(sched.get("max_degree", -1)),
  1390	        "coloring_method": sched["coloring_method"],
  1391	        # Production returns before selecting a strategy at n_dev==1, and a
  1392	        # caller may force halo_strategy; this reports what AUTO would pick.
  1393	        "production_strategy": (
  1394	            None if n_dev == 1 else
  1395	            ("allgather" if cells_per < ppermute_cells_per_device_threshold
  1396	             else "ppermute")),
  1397	        "cells_per_device": cells_per,
  1398	        "max_local_cells": int(max_lc),
  1399	        "max_local_edges": int(max_le),
  1400	    }
  1401	
  1402	
  1403	def _build_voronoi_partition_infra(global_mesh, n_dev, halo_depth=2):
  1404	    """Pre-compute per-device local meshes and gather/scatter indices.
  1405	
  1406	    After ``reorder_voronoi_for_sharding`` the global mesh has *both*
  1407	    cells and edges ordered by contiguous device blocks.  We compute
  1408	    partitions whose owned-entity boundaries exactly match the shard
  1409	    boundaries (``cells_per = nCells // n_dev``, ``edges_per = nEdges //
  1410	    n_dev``), then build local meshes with remapped connectivity.
  1750	    best_colors: dict[tuple[int, int], int] | None = None
  1751	    best_rounds = None
  1752	    for order in orders:
  1753	        ec = _greedy_edge_coloring_ordered(comm_pairs, order)
  1754	        rounds = max(ec.values(), default=-1) + 1
  1755	        if best_rounds is None or rounds < best_rounds:
  1756	            best_rounds, best_colors = rounds, ec
  1757	            if best_rounds <= max_degree:
  1758	                break            # hit the chromatic-index floor — optimal
  1759	    return best_colors, max_degree
  1760	
  1761	
  1762	def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
  1763	                             edges_per, max_lc, max_le):
  1764	    """Build a ppermute-based halo exchange schedule.
  1765	
  1766	    Instead of all-gathering the full state (O(N) communication),
  1767	    this schedule uses ``jax.lax.ppermute`` to exchange only halo data
  1768	    between neighboring devices.  The communication graph is edge-colored
  1769	    so that each round of ppermute moves data between non-conflicting
  1770	    pairs simultaneously.
  1771	
  1772	    Parameters
  1773	    ----------
  1774	    partitions : list[VoronoiPartition]
  1775	    cell_owner : np.ndarray, (nCells,)
  1776	    n_dev, cells_per, edges_per : int
  1777	    max_lc, max_le : int
  1778	        Maximum local cell/edge counts (owned + halo) across devices.
  1779	
  1780	    Returns
  1781	    -------
  1782	    dict with keys:
  1783	        n_rounds, n_rounds_greedy, max_degree, coloring_method,
  1784	        ppermute_perms, send_cell_idx, recv_cell_pos,
  1785	        send_edge_idx, recv_edge_pos, halo_cells_per_round,
  1786	        halo_edges_per_round.
  1787	    """
  1788	    from collections import defaultdict
  1789	
  1790	    import numpy as np
  1791	
  1792	    # ------------------------------------------------------------------
  1793	    # 1. For each device pair, find which cells/edges cross the boundary
  1794	    # ------------------------------------------------------------------
  1795	    # halo_cells_from[d][d'] = global indices of d's halo cells owned by d'
  1796	    halo_cells_from: dict[int, dict[int, list[int]]] = defaultdict(
  1797	        lambda: defaultdict(list))
  1798	    halo_edges_from: dict[int, dict[int, list[int]]] = defaultdict(
  1799	        lambda: defaultdict(list))
  1800	
  1801	    for d, part in enumerate(partitions):
  1802	        for h_idx in range(part.n_owned_cells, part.n_local_cells):
  1803	            g = int(part.local_cells[h_idx])
  1804	            owner = int(cell_owner[g])
  1805	            halo_cells_from[d][owner].append(g)
  1806	
  1807	        for h_idx in range(part.n_owned_edges, part.n_local_edges):
  1808	            g = int(part.local_edges[h_idx])
  1809	            owner = min(g // edges_per, n_dev - 1)
  1810	            halo_edges_from[d][owner].append(g)
  1811	
  1812	    # ------------------------------------------------------------------
  1813	    # 2. Build undirected communication graph
  1814	    # ------------------------------------------------------------------
  1815	    comm_pairs: set[tuple[int, int]] = set()
  1816	    for d in range(n_dev):
  1817	        for d_prime in halo_cells_from[d]:
  1818	            if d != d_prime:
  1819	                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
  1820	        for d_prime in halo_edges_from[d]:
  1821	            if d != d_prime:
  1822	                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
  1823	
  1824	    if not comm_pairs:
  1825	        return {
  1826	            'n_rounds': 0,
  1827	            'n_rounds_greedy': 0,
  1828	            'max_degree': 0,
  1829	            'coloring_method': 'none',
  1830	            'ppermute_perms': [],
  1831	            'send_cell_idx': [],
  1832	            'recv_cell_pos': [],
  1833	            'send_edge_idx': [],
  1834	            'recv_edge_pos': [],
  1835	            'halo_cells_per_round': [],
  1836	            'halo_edges_per_round': [],
  1837	        }
  1838	
  1839	    # ------------------------------------------------------------------
  1840	    # 3. Edge-color the graph: each color = one bidirectional ppermute
  1841	    #    ROUND, and the route-B lane is round-latency-bound (#1113), so
  1842	    #    fewer colors = directly less wall-clock. First-fit greedy is
  1843	    #    order-sensitive; the multi-start coloring reaches the
  1844	    #    chromatic-index floor (= max_degree) on every probed MPAS config
  1845	    #    where the legacy sorted greedy overshoots (up to 3 rounds at 16
  1846	    #    devices). It can never regress: the legacy sorted order is one of
  1847	    #    its candidates and it takes the min. Both are verified proper.
  1848	    # ------------------------------------------------------------------
  1849	    greedy_colors = _greedy_edge_coloring(comm_pairs)
  1850	    n_rounds_greedy = max(greedy_colors.values()) + 1
  1851	    multi_colors, max_degree = _multi_ordering_edge_coloring(comm_pairs)
  1852	    n_rounds_multi = max(multi_colors.values()) + 1
  1853	    # Adopt the multi-start coloring ONLY when it STRICTLY reduces rounds;
  1854	    # on a tie keep the exact legacy sorted-greedy coloring so the produced
  1855	    # schedule is byte-identical to before wherever there is no round win
  1856	    # (the win only appears at high device counts — >=16 on the probed
  1857	    # MPAS meshes). Both colorings are proper.
  1858	    if n_rounds_multi < n_rounds_greedy:
  1859	        edge_colors, n_rounds, coloring_method = (
  1860	            multi_colors, n_rounds_multi, "multi_greedy")
  1861	    else:
  1862	        edge_colors, n_rounds, coloring_method = (
  1863	            greedy_colors, n_rounds_greedy, "greedy")
  1864	    assert _check_proper_edge_coloring(edge_colors, comm_pairs), (
  1865	        "improper ppermute edge coloring — two same-round exchanges "
  1866	        "would collide at a device")
  1867	    rounds: dict[int, list[tuple[int, int]]] = defaultdict(list)
  1868	    for (u, v), color in edge_colors.items():
  1869	        rounds[color].append((u, v))
  1870	
  1871	    # ------------------------------------------------------------------
  1872	    # 4. Build directed send/recv maps for each device pair
  1873	    # ------------------------------------------------------------------
  1874	    # cell_send_map[(src, dst)] = list of owned-local indices in src to send
  1875	    # cell_recv_map[(dst, src)] = list of local positions in dst to place data
  1876	    cell_send_map: dict[tuple[int, int], list[int]] = {}
  1877	    cell_recv_map: dict[tuple[int, int], list[int]] = {}
  1878	    edge_send_map: dict[tuple[int, int], list[int]] = {}
  1879	    edge_recv_map: dict[tuple[int, int], list[int]] = {}
  1880	
  1881	    for d in range(n_dev):
  1882	        for d_prime, cells_g in halo_cells_from[d].items():
  1883	            if d_prime == d:
  1884	                continue
  1885	            # d_prime sends its owned cells that d needs as halo
  1886	            cell_send_map[(d_prime, d)] = [
  1887	                g - d_prime * cells_per for g in cells_g]
  1888	            cell_recv_map[(d, d_prime)] = [
  1889	                int(partitions[d].cell_g2l[g]) for g in cells_g]
  1890	
  1891	        for d_prime, edges_g in halo_edges_from[d].items():
  1892	            if d_prime == d:
  1893	                continue
  1894	            edge_send_map[(d_prime, d)] = [
  1895	                g - d_prime * edges_per for g in edges_g]
  1896	            edge_recv_map[(d, d_prime)] = [
  1897	                int(partitions[d].edge_g2l[g]) for g in edges_g]
  1898	
  1899	    # ------------------------------------------------------------------
  1900	    # 5. Assemble per-round ppermute patterns and index arrays
  1901	    # ------------------------------------------------------------------
  1902	    ppermute_perms_out: list[list[tuple[int, int]]] = []
  1903	    send_cell_idx_out: list[jnp.ndarray] = []
  1904	    recv_cell_pos_out: list[jnp.ndarray] = []
  1905	    send_edge_idx_out: list[jnp.ndarray] = []
  1906	    recv_edge_pos_out: list[jnp.ndarray] = []
  1907	    halo_cells_per_round: list[int] = []
  1908	    halo_edges_per_round: list[int] = []
  1909	
  1910	    for r in range(n_rounds):
  1911	        # Max halo size across all pairs in this round
  1912	        max_c = 0
  1913	        max_e = 0
  1914	        for u, v in rounds[r]:
  1915	            for src, dst in [(u, v), (v, u)]:
  1916	                max_c = max(max_c, len(cell_send_map.get((src, dst), [])))
  1917	                max_e = max(max_e, len(edge_send_map.get((src, dst), [])))
  1918	        max_c = max(max_c, 1)  # at least 1 for array shape
  1919	        max_e = max(max_e, 1)
  1920	        halo_cells_per_round.append(max_c)
  1921	        halo_edges_per_round.append(max_e)
  1922	
  1923	        # Bidirectional ppermute pattern
  1924	        perm: list[tuple[int, int]] = []
  1925	        partner: dict[int, int] = {}
  1926	        for u, v in rounds[r]:
  1927	            perm.append((u, v))
  1928	            perm.append((v, u))
  1929	            partner[u] = v
  1930	            partner[v] = u
     1	"""Direct tests for ``sharded_dynamics.spmd_schedule_cost``.
     2	
     3	It scores a mesh split by how many sequential halo exchanges it needs. The
     4	number only means something if it comes from the same builders, the same halo
     5	depth, and the same mesh state production uses -- so that is what these pin.
     6	"""
     7	from __future__ import annotations
     8	
     9	import pytest
    10	
    11	from legoesm.grids.voronoi import create_voronoi_mesh
    12	from legoesm.parallel import sharded_dynamics as sd
    13	from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
    14	
    15	
    16	@pytest.fixture(scope="module")
    17	def mesh():
    18	    return create_voronoi_mesh(subdivision_level=4)
    19	
    20	
    21	def test_returns_sane_cost(mesh):
    22	    c = sd.spmd_schedule_cost(mesh, 4)
    23	    assert c["n_dev"] == 4
    24	    assert c["halo_depth"] == 3
    25	    assert c["resolved_method"] == "sfc", "auto must resolve concretely"
    26	    # A round exchanges data between disjoint device PAIRS, so a proper
    27	    # schedule on n_dev devices needs between 1 and n_dev-1 rounds.
    28	    assert 1 <= c["n_rounds"] <= 3, c
    29	    assert c["n_rounds"] <= c["n_rounds_greedy"]
    30	    assert c["max_degree"] >= 1
    31	    assert c["n_rounds"] >= c["max_degree"], (
    32	        "a proper edge colouring can never use FEWER rounds than max_degree")
    33	    assert c["max_local_cells"] > mesh.nCells // 4, "must include the halo"
    34	
    35	
    36	def test_reports_max_degree_so_optimality_is_measured_not_assumed(mesh):
    37	    """The colouring is a best-of-a-few-orders search, NOT a proof of the
    38	    max_degree lower bound. Callers must be able to check equality rather than
    39	    assume it, so max_degree is returned alongside n_rounds."""
    40	    c = sd.spmd_schedule_cost(mesh, 8)
    41	    assert "max_degree" in c and c["max_degree"] > 0
    42	    assert c["n_rounds"] >= c["max_degree"]
    43	
    44	
    45	def test_single_device_is_zero_rounds_and_zero_is_refused(mesh):
    46	    assert sd.spmd_schedule_cost(mesh, 1)["n_rounds"] == 0
    47	    with pytest.raises(ValueError, match="integer >= 1"):
    48	        sd.spmd_schedule_cost(mesh, 0)
    49	
    50	
    51	def test_flags_allgather_when_production_would_not_use_ppermute(mesh):
    52	    """Production auto-selects allgather below a cells/device threshold; the
    53	    ppermute round count is then counterfactual and must say so."""
    54	    c = sd.spmd_schedule_cost(mesh, 8,
    55	                              ppermute_cells_per_device_threshold=10**9)
    56	    assert c["production_strategy"] == "allgather"
    57	    big = sd.spmd_schedule_cost(mesh, 8,
    58	                                ppermute_cells_per_device_threshold=1)
    59	    assert big["production_strategy"] == "ppermute"
    60	
    61	
    62	def test_already_reordered_mesh_is_not_reordered_again(mesh):
    63	    """Production holds an already-reordered mesh. Re-splitting it would score
    64	    a mesh no run uses, so that path must be expressible and must agree with
    65	    scoring the raw mesh once."""
    66	    prepared = reorder_voronoi_for_sharding(mesh, 8, method="sfc")
    67	    pre = sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
    68	    raw = sd.spmd_schedule_cost(mesh, 8, method="sfc")
    69	    assert pre["resolved_method"] == "pre-reordered"
    70	    assert pre["reorder_target"] is None, (
    71	        "the target that produced a pre-reordered mesh is not recoverable "
    72	        "from it — reporting n_dev would assert something unverified")
    73	    assert pre["n_rounds"] == raw["n_rounds"], (
    74	        "scoring a pre-reordered mesh must match scoring the raw mesh with "
    75	        "the same ownership")
    76	    with pytest.raises(ValueError, match="reorder_target is meaningless"):
    77	        sd.spmd_schedule_cost(prepared, 8, already_reordered=True,
    78	                              reorder_target=16)
    79	
    80	
    81	def test_reorder_target_differing_from_run_device_count(mesh):
    82	    """The scaling bench reorders once for a target and then runs at a
    83	    different device count. That combination must be expressible, and must
    84	    differ from reordering for the run count."""
    85	    same = sd.spmd_schedule_cost(mesh, 4, method="sfc")
    86	    split_for_16 = sd.spmd_schedule_cost(mesh, 4, method="sfc",
    87	                                         reorder_target=16)
    88	    assert split_for_16["reorder_target"] == 16
    89	    assert split_for_16["n_dev"] == 4
    90	    # Non-vacuity: a split built for 16 devices really is a different split.
    91	    assert split_for_16["n_rounds"] != same["n_rounds"] or (
    92	        split_for_16["max_local_cells"] != same["max_local_cells"]), (
    93	        "reorder_target had no effect — the argument would be decorative")
    94	
    95	
    96	def test_repeatable(mesh):
    97	    a = sd.spmd_schedule_cost(mesh, 8, method="sfc")
    98	    b = sd.spmd_schedule_cost(mesh, 8, method="sfc")
    99	    assert a == b, "same inputs must give the same score"
   100	
   101	
   102	@pytest.mark.slow
   103	def test_sfc_beats_metis_and_geometric_on_rounds():
   104	    """The finding this function exists to make measurable. Census at
   105	    production sizes: subdiv-8 sfc 12/14 rounds at 64/128 devices vs metis
   106	    13/19, geometric 16/21. Same ordering here at the SMALLEST size that can
   107	    still tell the methods apart.
   108	
   109	    subdiv-4@8 and subdiv-5@8 score all three methods identically (7 rounds) —
   110	    too coarse to discriminate — so this uses subdiv-5@16 and asserts the
   111	    scores actually differ before asserting their order. ~100 s, hence slow.
   112	    """
   113	    pytest.importorskip("pymetis", reason="metis arm needs pymetis")
   114	    big = create_voronoi_mesh(subdivision_level=5)
   115	    r = {m: sd.spmd_schedule_cost(big, 16, method=m)["n_rounds"]
   116	         for m in ("sfc", "metis", "geometric")}
   117	    assert len(set(r.values())) > 1, (
   118	        f"all methods scored identically ({r}) — the comparison is vacuous "
   119	        f"at this mesh size; use a finer mesh or more devices")
   120	    assert r["sfc"] <= r["metis"] and r["sfc"] <= r["geometric"], r
   121	
   122	
   123	def test_unknown_method_raises(mesh):
   124	    with pytest.raises(ValueError, match="Unknown partitioning method"):
   125	        sd.spmd_schedule_cost(mesh, 4, method="not_a_method")
   126	
   127	
   128	def test_already_reordered_does_not_reorder(mesh, monkeypatch):
   129	    """Spy, not inference: an accidental second reorder could still produce a
   130	    coincidentally equal round count, so assert the call never happens."""
   131	    from legoesm.parallel import voronoi_partition as vp
   132	
   133	    prepared = reorder_voronoi_for_sharding(mesh, 8, method="sfc")
   134	    calls = []
   135	    monkeypatch.setattr(
   136	        vp, "reorder_voronoi_for_sharding",
   137	        lambda *a, **k: calls.append(1) or prepared)
   138	    sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
   139	    assert not calls, "already_reordered=True must not reorder the mesh"
   140	
   141	
   142	def test_indivisible_reorder_target_is_refused(mesh):
   143	    """A mesh padded for 3 devices is not divisible by 4. The builder would
   144	    assign residual cells to the last owner while excluding them from every
   145	    owned block — a plausible-looking, wrong number."""
   146	    with pytest.raises(ValueError, match="divisible by n_dev"):
   147	        sd.spmd_schedule_cost(mesh, 4, method="sfc", reorder_target=3)
   148	
   149	
   150	def test_non_integer_device_count_is_refused(mesh):
   151	    with pytest.raises(ValueError, match="integer >= 1"):
   152	        sd.spmd_schedule_cost(mesh, 3.9)
   153	
   154	
   155	def test_halo_depth_is_one_shared_constant():
   156	    """Production and the scorer must read the SAME depth. Two independently
   157	    hardcoded 3s let production drift without the score noticing."""
   158	    import inspect
   159	    assert sd.SPMD_HALO_DEPTH == 3
   160	    assert (inspect.signature(sd.spmd_schedule_cost)
   161	            .parameters["halo_depth"].default == sd.SPMD_HALO_DEPTH)
   162	    prod = inspect.getsource(sd.make_voronoi_sharded_step)
   163	    assert "halo_depth=SPMD_HALO_DEPTH" in prod, (
   164	        "production stopped consuming the shared constant")
   165	
   166	
   167	def test_single_device_reports_no_strategy(mesh):
   168	    """Production returns before choosing a halo strategy at one device."""
   169	    assert sd.spmd_schedule_cost(mesh, 1)["production_strategy"] is None

 exited 1 in 1934ms:
Traceback (most recent call last):
  File "<frozen runpy>", line 203, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/pytest/__main__.py", line 9, in <module>
    raise SystemExit(pytest.console_main())
                     ~~~~~~~~~~~~~~~~~~~^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/_pytest/config/__init__.py", line 223, in console_main
    code = main()
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/_pytest/config/__init__.py", line 193, in main
    config = _prepareconfig(new_args, plugins)
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/_pytest/config/__init__.py", line 361, in _prepareconfig
    config: Config = pluginmanager.hook.pytest_cmdline_parse(
                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        pluginmanager=pluginmanager, args=args
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/pluggy/_callers.py", line 167, in _multicall
    raise exception
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/pluggy/_callers.py", line 139, in _multicall
    teardown.throw(exception)
    ~~~~~~~~~~~~~~^^^^^^^^^^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/_pytest/helpconfig.py", line 124, in pytest_cmdline_parse
    config = yield
             ^^^^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/_pytest/config/__init__.py", line 1192, in pytest_cmdline_parse
    self.parse(args)
    ~~~~~~~~~~^^^^^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/_pytest/config/__init__.py", line 1562, in parse
    self.hook.pytest_load_initial_conftests(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        early_config=self, args=args, parser=self._parser
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/pluggy/_callers.py", line 167, in _multicall
    raise exception
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/pluggy/_callers.py", line 139, in _multicall
    teardown.throw(exception)
    ~~~~~~~~~~~~~~^^^^^^^^^^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/_pytest/warnings.py", line 128, in pytest_load_initial_conftests
    return (yield)
            ^^^^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/pluggy/_callers.py", line 116, in _multicall
    next(function_gen)  # first yield
    ~~~~^^^^^^^^^^^^^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/_pytest/capture.py", line 170, in pytest_load_initial_conftests
    capman.start_global_capturing()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/_pytest/capture.py", line 773, in start_global_capturing
    self._global_capturing = _get_multicapture(self._method)
                             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/_pytest/capture.py", line 714, in _get_multicapture
    return MultiCapture(in_=FDCapture(0), out=FDCapture(1), err=FDCapture(2))
                                              ~~~~~~~~~^^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/_pytest/capture.py", line 493, in __init__
    TemporaryFile(buffering=0),
    ~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/home/b/b381103/.local/share/uv/python/cpython-3.14.6-linux-x86_64-gnu/lib/python3.14/tempfile.py", line 652, in TemporaryFile
    prefix, suffix, dir, output_type = _sanitize_params(prefix, suffix, dir)
                                       ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/home/b/b381103/.local/share/uv/python/cpython-3.14.6-linux-x86_64-gnu/lib/python3.14/tempfile.py", line 127, in _sanitize_params
    dir = gettempdir()
  File "/home/b/b381103/.local/share/uv/python/cpython-3.14.6-linux-x86_64-gnu/lib/python3.14/tempfile.py", line 312, in gettempdir
    return _os.fsdecode(_gettempdir())
                        ~~~~~~~~~~~^^
  File "/home/b/b381103/.local/share/uv/python/cpython-3.14.6-linux-x86_64-gnu/lib/python3.14/tempfile.py", line 305, in _gettempdir
    tempdir = _get_default_tempdir()
  File "/home/b/b381103/.local/share/uv/python/cpython-3.14.6-linux-x86_64-gnu/lib/python3.14/tempfile.py", line 222, in _get_default_tempdir
    raise FileNotFoundError(_errno.ENOENT,
                            "No usable temporary directory found in %s" %
                            dirlist)
FileNotFoundError: [Errno 2] No usable temporary directory found in ['/tmp', '/var/tmp', '/usr/tmp', '/work/bd1083/b309178/diffESM/legoesm_pg/legoESM']

 succeeded in 15240ms:
06c7c7e8c feat(parallel): route-B NCCL SPMD launch hardening — fallback guard, launcher-family GPU binding, transport report (audit item 6) (#879)
./tests/bench/test_mpas_schedule_cost_scan_sbatch.py:24:           / "mpas_schedule_cost_scan.sbatch")

exec
/usr/bin/bash -lc "nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '170,250p'; nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '360,505p'; nl -ba tests/bench/test_mpas_schedule_cost_scan_sbatch.py | sed -n '25,110p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "rg -n \"\\.venv|pytest .*tests|uv run pytest|python -m pytest\" .github pyproject.toml tox.ini setup.cfg Makefile justfile 2>/dev/null | head -240" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "rg -n \"def scaling_metadata|def annotate_incomplete\" packages scripts/bench/bench_voronoi_partition_methods.py; nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '80,185p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c \"import ast,pathlib; paths=['scripts/bench/bench_voronoi_partition_methods.py','tests/bench/test_bench_voronoi_partition_methods.py','tests/bench/test_mpas_schedule_cost_scan_sbatch.py']; [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in paths]; print('Python syntax OK')\"; bash -n scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && echo 'shell syntax OK'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 225ms:
   170	        partition_cells_metis,
   171	        partition_cells_sfc,
   172	    )
   173	
   174	    if method == "geometric":
   175	        return np.asarray(partition_cells_geometric(mesh, n_ranks))
   176	    if method == "sfc":
   177	        return np.asarray(partition_cells_sfc(mesh, n_ranks))
   178	    if method == "metis":
   179	        return np.asarray(partition_cells_metis(mesh, n_ranks))
   180	    raise ValueError(f"unknown partition method {method!r}; "
   181	                     f"expected one of {METHODS}")
   182	
   183	
   184	def parse_expect_rounds(spec: str) -> dict:
   185	    """Parse ``'sfc:64=12,metis:128=19'`` into ``{("sfc", 64): 12}``.
   186	
   187	    Raises on anything malformed rather than skipping it — a typo'd
   188	    expectation that is silently dropped turns the gate into a no-op, which
   189	    is exactly the failure this flag exists to prevent.  Three ways that
   190	    could happen, all rejected here:
   191	
   192	    * a malformed item (``geometric:2``, ``geometric=2``);
   193	    * an unknown method;
   194	    * a DUPLICATE key — ``geometric:2=999,geometric:2=13`` would otherwise
   195	      let the second silently overwrite the first and pass;
   196	    * a non-empty spec that parses to NOTHING (``",,,"``), which would make
   197	      the caller skip the check while believing it ran.
   198	    """
   199	    out: dict[tuple[str, int], int] = {}
   200	    for item in (s.strip() for s in spec.split(",")):
   201	        if not item:
   202	            continue
   203	        try:
   204	            lhs, rounds = item.split("=")
   205	            method, n_ranks = lhs.split(":")
   206	            key = (method.strip(), int(n_ranks))
   207	            value = int(rounds)
   208	        except ValueError as exc:
   209	            raise ValueError(
   210	                f"--expect-rounds: cannot parse {item!r}; expected "
   211	                f"'<method>:<n_ranks>=<n_rounds>'") from exc
   212	        if key[0] not in METHODS:
   213	            raise ValueError(
   214	                f"--expect-rounds: unknown method {key[0]!r} in {item!r}; "
   215	                f"expected one of {METHODS}")
   216	        if key in out:
   217	            raise ValueError(
   218	                f"--expect-rounds: duplicate expectation for "
   219	                f"{key[0]}:{key[1]} ({out[key]} then {value}); the later one "
   220	                f"would silently overwrite the earlier and the gate would "
   221	                f"pass while discarding a listed expectation.")
   222	        out[key] = value
   223	    # Guard on `spec`, NOT `spec.strip()`: a whitespace-only value is a value
   224	    # the caller PASSED, and silently reading it as "no gate requested" is
   225	    # the same bypass as ",,," (codex round 3).  Only the default empty
   226	    # string means "no gate".
   227	    if spec and not out:
   228	        raise ValueError(
   229	            f"--expect-rounds={spec!r} parses to NO expectations; the gate "
   230	            f"would be skipped while looking like it ran.")
   231	    return out
   232	
   233	
   234	def check_expected_rounds(rows: list, expect: dict) -> list:
   235	    """Compare scored rounds against *expect*; return failure strings.
   236	
   237	    A listed pair that was never scored is a FAILURE, not a skip: otherwise
   238	    a sweep that silently dropped a method (unavailable ``pymetis``) or a
   239	    rank count would still report a clean gate.
   240	    """
   241	    scored = {
   242	        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
   243	        for r in rows
   244	        if r.get("available") and "schedule" in r and "n_ranks" in r
   245	    }
   246	    failures = []
   247	    for (method, n_ranks), want in sorted(expect.items()):
   248	        got = scored.get((method, n_ranks))
   249	        if got is None:
   250	            failures.append(
   360	                        "instead of asserting agreement in a comment. "
   361	                        "Requires --schedule-cost.")
   362	    p.add_argument("--out", type=str,
   363	                   default="results/a1/voronoi_partition_quality.json")
   364	    args = p.parse_args()
   365	
   366	    rank_counts = [int(x) for x in args.rank_counts.split(",") if x]
   367	    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
   368	    for m in methods:
   369	        if m not in METHODS:
   370	            raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
   371	    if not rank_counts or any(n < 2 for n in rank_counts):
   372	        raise SystemExit("--rank-counts needs integers >= 2")
   373	    # Negative values in these three are all the SAME false-provenance bug:
   374	    # the underlying code treats them exactly like 0 (no relaxation, no
   375	    # bisection, `range(-1)` is empty), but the run is recorded under the
   376	    # negative value, so a level-0 mesh gets filed as "L-1".
   377	    if args.lloyd < 0:
   378	        raise SystemExit(
   379	            f"--lloyd must be >= 0, got {args.lloyd}: the mesh builder "
   380	            f"relaxes only for > 0, so a negative is indistinguishable from "
   381	            f"0 in the mesh but is recorded and cached under its own key.")
   382	    if args.subdivision < 0:
   383	        raise SystemExit(
   384	            f"--subdivision must be >= 0, got {args.subdivision}: the mesh "
   385	            f"builder bisects only for > 0, so a negative silently yields "
   386	            f"the level-0 base mesh while being recorded as "
   387	            f"L{args.subdivision}.")
   388	    if args.halo_depth < 0:
   389	        raise SystemExit(
   390	            f"--halo-depth must be >= 0, got {args.halo_depth}: the halo "
   391	            f"loop is `range(depth)`, so a negative behaves exactly like 0 "
   392	            f"while being recorded as {args.halo_depth}.")
   393	    expect = parse_expect_rounds(args.expect_rounds)
   394	    if expect and not args.schedule_cost:
   395	        raise SystemExit(
   396	            "--expect-rounds compares scored round counts, so it needs "
   397	            "--schedule-cost; without it nothing is scored and the gate "
   398	            "would pass vacuously.")
   399	
   400	
   401	    from legoesm.grids.voronoi import create_voronoi_mesh
   402	    from legoesm.parallel.voronoi_partition import resolve_partition_method
   403	
   404	    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
   405	                               lloyd_iterations=args.lloyd)
   406	    if max(rank_counts) > int(mesh.nCells):
   407	        raise SystemExit(
   408	            f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
   409	            f"{int(mesh.nCells)} cells (empty ranks are meaningless).")
   410	    print(f"mesh L{args.subdivision}: {int(mesh.nCells)} cells, "
   411	          f"{int(mesh.nEdges)} edges; auto -> "
   412	          f"{resolve_partition_method('auto')!r}")
   413	
   414	    rows = []
   415	    for method in methods:
   416	        if not method_available(method):
   417	            print(f"  {method:9s}: UNAVAILABLE (pymetis not importable) — "
   418	                  f"column omitted, never substituted")
   419	            rows.append({"method": method, "available": False})
   420	            continue
   421	        for n_ranks in rank_counts:
   422	            q = partition_quality(
   423	                mesh, owner_for(mesh, method, n_ranks), n_ranks,
   424	                halo_depth=args.halo_depth)
   425	            row = {"method": method, "available": True,
   426	                   "n_ranks": n_ranks, **q}
   427	            rows.append(row)
   428	            print(f"  {method:9s} np={n_ranks:3d} | "
   429	                  f"imbalance={q['load_imbalance_max_over_mean']:.3f} | "
   430	                  f"edge_cut={q['edge_cut']:6d} "
   431	                  f"({100 * q['edge_cut_fraction']:.2f}%) | "
   432	                  f"halo max={q['halo_cells_max']:5d} "
   433	                  f"mean={q['halo_cells_mean']:8.1f} | "
   434	                  f"halo/owned max={q['halo_owned_ratio_max']:.3f} | "
   435	                  f"nbrs max={q['neighbor_ranks_max']}")
   436	            if args.schedule_cost:
   437	                sc = schedule_cost_row(mesh, method, n_ranks)
   438	                row["schedule"] = sc
   439	                note = ("  [COUNTERFACTUAL: production auto-selects "
   440	                        "allgather here, no ppermute schedule]"
   441	                        if sc["production_strategy"] == "allgather" else "")
   442	                verdict = ("colouring PROVABLY OPTIMAL (Vizing floor)"
   443	                           if sc["coloring_optimal_proven"] else
   444	                           f"recolour headroom "
   445	                           f"{sc['coloring_headroom_rounds_min']}-"
   446	                           f"{sc['coloring_headroom_rounds_max']} round(s)")
   447	                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
   448	                      f"rounds={sc['n_rounds']:3d} "
   449	                      f"max_degree={sc['max_degree']:3d} "
   450	                      f"gap={sc['coloring_gap']:+d} "
   451	                      f"-> {verdict} "
   452	                      f"({sc['score_seconds']:.1f}s){note}", flush=True)
   453	
   454	    payload = {
   455	        "rows": rows,
   456	        "auto_resolves_to": resolve_partition_method("auto"),
   457	        "step_time_pointer": (
   458	            "step-time per method: bench_ocean_mpas_scaling.py / "
   459	            "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
   460	            "per launch, same case otherwise)"),
   461	        "metadata": annotate_incomplete(scaling_metadata(
   462	            grid="voronoi",
   463	            component="partitioning",
   464	            resolution=f"L{args.subdivision}",
   465	            n_levels=0,
   466	            precision="n/a",
   467	            decomposition="cell_partition",
   468	            solver_variant="n/a",
   469	            scaling_kind="partition-quality",
   470	            transport="none",
   471	            extra={"rank_counts": rank_counts, "methods": methods,
   472	                   "halo_depth": args.halo_depth,
   473	                   "schedule_cost": bool(args.schedule_cost),
   474	                   "lloyd_iterations": args.lloyd},
   475	        )),
   476	    }
   477	    outdir = os.path.dirname(args.out)
   478	    if outdir:
   479	        os.makedirs(outdir, exist_ok=True)
   480	    with open(args.out, "w") as f:
   481	        json.dump(payload, f, indent=2)
   482	    print(f"JSON: {args.out}")
   483	
   484	    if expect:
   485	        failures = check_expected_rounds(rows, expect)
   486	        payload["expected_rounds_check"] = {
   487	            "expected": {f"{m}:{n}": v for (m, n), v in sorted(expect.items())},
   488	            "failures": failures,
   489	            "passed": not failures,
   490	        }
   491	        with open(args.out, "w") as f:
   492	            json.dump(payload, f, indent=2)
   493	        if failures:
   494	            print(f"INSTRUMENT CHECK FAILED ({len(failures)}):")
   495	            for line in failures:
   496	                print(f"  {line}")
   497	            print("The scorer did NOT reproduce the known census — treat every "
   498	                  "unknown row in this run as UNTRUSTED.")
   499	            return 1
   500	        print(f"INSTRUMENT CHECK PASSED ({len(expect)} expectations)")
   501	    return 0
   502	
   503	
   504	if __name__ == "__main__":
   505	    raise SystemExit(main())
    25	
    26	
    27	def _run(tmp_path, codes):
    28	    """Run the launcher with a stub bench that exits ``codes`` per call.
    29	
    30	    ``codes`` is one exit status per invocation, in order (arm 1, arm 2,
    31	    arm 3).  The stub also records the ``--subdivision`` of each call so a
    32	    test can assert an arm did NOT run.
    33	    """
    34	    stub = tmp_path / "stub_bench.py"
    35	    calls = tmp_path / "calls.txt"
    36	    stub.write_text(
    37	        "import sys\n"
    38	        f"codes = {list(codes)!r}\n"
    39	        f"log = {str(calls)!r}\n"
    40	        "sub = sys.argv[sys.argv.index('--subdivision') + 1]\n"
    41	        "with open(log, 'a') as f:\n"
    42	        "    f.write(sub + '\\n')\n"
    43	        "n = sum(1 for _ in open(log))\n"
    44	        "sys.exit(codes[n - 1] if n <= len(codes) else 0)\n"
    45	    )
    46	    env = dict(os.environ)
    47	    env["LEGOESM_SCHEDULE_COST_BENCH"] = str(stub)
    48	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
    49	    env["LEGOESM_REPO"] = str(_REPO)
    50	    # _env.sh resolves PY from here; the stub is plain python, and _env.sh
    51	    # verifies PY can import jax, so use the repo venv.
    52	    env.setdefault("LEGOESM_PYTHON", str(_REPO / ".venv" / "bin" / "python"))
    53	    proc = subprocess.run(
    54	        ["bash", str(_SBATCH)], env=env, cwd=str(_REPO),
    55	        capture_output=True, text=True, timeout=600)
    56	    ran = calls.read_text().split() if calls.exists() else []
    57	    return proc, ran
    58	
    59	
    60	@pytest.mark.skipif(not (_REPO / ".venv" / "bin" / "python").exists(),
    61	                    reason="needs the repo venv for _env.sh's jax check")
    62	def test_validation_failure_aborts_before_the_unknown_arm(tmp_path):
    63	    """Arm 1 failing must abort with non-zero AND must not run arm 3.
    64	
    65	    An instrument that misses the known census cannot be trusted on the
    66	    unknown one, so producing an s10 number anyway would be worse than
    67	    producing none.
    68	    """
    69	    proc, ran = _run(tmp_path, [1, 0, 0])
    70	    assert proc.returncode != 0, proc.stdout[-2000:]
    71	    assert "SCAN_ABORTED_VALIDATION" in proc.stdout
    72	    assert "10" not in ran, f"arm 3 ran despite failed validation: {ran}"
    73	
    74	
    75	@pytest.mark.skipif(not (_REPO / ".venv" / "bin" / "python").exists(),
    76	                    reason="needs the repo venv for _env.sh's jax check")
    77	def test_arm3_failure_is_not_reported_as_success(tmp_path):
    78	    """The original BLOCKER: arm 3 fails, job must NOT exit 0."""
    79	    proc, ran = _run(tmp_path, [0, 0, 3])
    80	    assert proc.returncode == 3, (
    81	        f"arm-3 status not propagated (got {proc.returncode})\n"
    82	        f"{proc.stdout[-2000:]}")
    83	    assert "SCAN_FAILED_ARM3" in proc.stdout
    84	    assert "SCAN_DONE" not in proc.stdout
    85	    assert ran == ["8", "9", "10"], ran
    86	
    87	
    88	@pytest.mark.skipif(not (_REPO / ".venv" / "bin" / "python").exists(),
    89	                    reason="needs the repo venv for _env.sh's jax check")
    90	def test_full_success_exits_zero_and_runs_every_arm(tmp_path):
    91	    proc, ran = _run(tmp_path, [0, 0, 0])
    92	    assert proc.returncode == 0, proc.stdout[-2000:]
    93	    assert "SCAN_DONE" in proc.stdout
    94	    assert ran == ["8", "9", "10"], ran

 succeeded in 271ms:
pyproject.toml:335:# Run: `.venv/bin/lint-imports`
.github/workflows/ci.yml:97:        run: python -m pytest --collect-only tests/ 2>&1
.github/workflows/ci.yml:121:          python -m pytest tests/unit/ -x --timeout=300 \
.github/workflows/ci.yml:182:          python -m pytest "${files[@]}" --timeout=300 \
.github/workflows/ci.yml:196:          python -m pytest tests/grids/test_fv3_native_grid_phase1.py \
.github/workflows/ci.yml:213:          python -m pytest tests/grids/test_fv3_native_swcore_phase4.py \
.github/workflows/ci.yml:229:          python -m pytest tests/unit/test_cmor_experiments_restart.py \
.github/workflows/ci.yml:245:          python -m pytest tests/atmosphere/ -x --timeout=600 \
.github/workflows/mpi-distributed.yml:52:              python -m pytest -q tests/distributed/test_halo_mpi.py
.github/workflows/mpi-distributed.yml:62:              python -m pytest -q tests/distributed/test_plane_pencil_mpi.py
.github/workflows/mpi-distributed.yml:75:              python -m pytest -q tests/distributed/test_latlon_2d_pad_wall_mpi.py
.github/workflows/mpi-distributed.yml:88:              python -m pytest -q tests/distributed/test_latlon_transpose_ad_mpi.py
.github/workflows/mpi-nightly.yml:47:            python -m pytest -q tests/distributed/test_ocean_mpi_conservation.py -k longrun

 succeeded in 321ms:
    80	from __future__ import annotations
    81	
    82	import argparse
    83	import json
    84	import os
    85	import sys
    86	from pathlib import Path
    87	
    88	import numpy as np
    89	
    90	sys.path.insert(0, str(Path(__file__).resolve().parent))
    91	
    92	from metadata import annotate_incomplete, scaling_metadata  # noqa: E402
    93	
    94	METHODS = ("geometric", "sfc", "metis")
    95	
    96	
    97	def method_available(method: str) -> bool:
    98	    if method != "metis":
    99	        return True
   100	    try:
   101	        import pymetis  # noqa: F401
   102	
   103	        return True
   104	    except Exception:
   105	        return False
   106	
   107	
   108	def partition_quality(mesh, cell_owner: np.ndarray, n_ranks: int,
   109	                      halo_depth: int = 2) -> dict:
   110	    """Exact partition-quality metrics from a global owner array.
   111	
   112	    Serial enumeration of every rank (no MPI): the same halo construction
   113	    the runtime uses (``compute_halo_cells``), so the reported halo sizes
   114	    are the runtime's, not an estimate.
   115	    """
   116	    from legoesm.parallel.voronoi_partition import compute_halo_cells
   117	
   118	    n_cells = int(mesh.nCells)
   119	    # Correctness: every cell owned exactly once, owners in range, no
   120	    # empty rank (an empty rank would silently deflate the halo/owned
   121	    # ratio through the max(counts, 1) guard).
   122	    if n_cells == 0 or cell_owner.size == 0:
   123	        raise AssertionError("empty mesh / owner array")
   124	    if cell_owner.shape != (n_cells,):
   125	        raise AssertionError(f"owner shape {cell_owner.shape} != ({n_cells},)")
   126	    if cell_owner.min() < 0 or cell_owner.max() >= n_ranks:
   127	        raise AssertionError("owner out of range")
   128	    counts = np.bincount(cell_owner, minlength=n_ranks).astype(float)
   129	    if int(counts.sum()) != n_cells:
   130	        raise AssertionError("ownership does not cover the mesh")
   131	    if counts.min() <= 0:
   132	        raise AssertionError(
   133	            f"empty rank in partition (counts.min()={counts.min():.0f}) — "
   134	            f"a skipped rank corrupts every per-rank metric")
   135	
   136	    # Edge cut: edges whose two cells have different owners.
   137	    c1, c2 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
   138	    valid = (c1 >= 0) & (c2 >= 0)
   139	    edge_cut = int((cell_owner[c1[valid]] != cell_owner[c2[valid]]).sum())
   140	
   141	    halo_sizes = []
   142	    neighbor_counts = []
   143	    cells_on_cell = np.asarray(mesh.cellsOnCell)
   144	    max_edges = int(cells_on_cell.shape[0]) if cells_on_cell.ndim == 2 else 0
   145	    for r in range(n_ranks):
   146	        halo = compute_halo_cells(
   147	            cell_owner, mesh.cellsOnCell, mesh.maxEdges, r, halo_depth)
   148	        halo_sizes.append(len(halo))
   149	        neighbor_counts.append(
   150	            len(set(int(cell_owner[c]) for c in halo) - {r}))
   151	    _ = max_edges
   152	    halo_sizes = np.array(halo_sizes, dtype=float)
   153	    return {
   154	        "cells_per_rank_min": int(counts.min()),
   155	        "cells_per_rank_max": int(counts.max()),
   156	        "load_imbalance_max_over_mean": float(counts.max() / counts.mean()),
   157	        "edge_cut": edge_cut,
   158	        "edge_cut_fraction": float(edge_cut / max(int(valid.sum()), 1)),
   159	        "halo_cells_max": int(halo_sizes.max()),
   160	        "halo_cells_mean": float(halo_sizes.mean()),
   161	        "halo_owned_ratio_max": float(
   162	            (halo_sizes / np.maximum(counts, 1.0)).max()),
   163	        "neighbor_ranks_max": int(max(neighbor_counts)),
   164	    }
   165	
   166	
   167	def owner_for(mesh, method: str, n_ranks: int) -> np.ndarray:
   168	    from legoesm.parallel.voronoi_partition import (
   169	        partition_cells_geometric,
   170	        partition_cells_metis,
   171	        partition_cells_sfc,
   172	    )
   173	
   174	    if method == "geometric":
   175	        return np.asarray(partition_cells_geometric(mesh, n_ranks))
   176	    if method == "sfc":
   177	        return np.asarray(partition_cells_sfc(mesh, n_ranks))
   178	    if method == "metis":
   179	        return np.asarray(partition_cells_metis(mesh, n_ranks))
   180	    raise ValueError(f"unknown partition method {method!r}; "
   181	                     f"expected one of {METHODS}")
   182	
   183	
   184	def parse_expect_rounds(spec: str) -> dict:
   185	    """Parse ``'sfc:64=12,metis:128=19'`` into ``{("sfc", 64): 12}``.

 succeeded in 319ms:
Python syntax OK
shell syntax OK

exec
/usr/bin/bash -lc "nl -ba .github/workflows/ci.yml | sed -n '50,135p'; nl -ba .github/workflows/ci.yml | sed -n '145,195p'; nl -ba pyproject.toml | sed -n '300,345p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc 'rg -n "tests/bench|test_bench_voronoi_partition_methods|bench/" .github/workflows pyproject.toml tox.ini 2>/dev/null' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "git diff --check; git status --short scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch tests/bench/test_mpas_schedule_cost_scan_sbatch.py; rg -n \"LEGOESM_SCHEDULE_COST_BENCH\" . --glob '"'!*.md'"' --glob '"'!*.pyc'"' 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 212ms:
    50	        with:
    51	          python-version: "3.11"
    52	      # dev extra (not bare ruff): lint-imports/grimp must be able to import the
    53	      # installed legoesm namespace to walk the dependency graph.
    54	      - run: pip install -e ".[dev]"
    55	      # Scope is the WHOLE tree, not just src/legoesm: the federation carve left
    56	      # only 6 modules under src/legoesm, so the old scope linted ~0.6% of the
    57	      # code and never saw the F821s that lived in packages/ and scripts/.
    58	      - name: Ruff check (errors only)
    59	        run: ruff check src packages scripts tests --select E9,F63,F7,F82 --statistics
    60	      - name: Import-boundary contracts (import-linter)
    61	        run: lint-imports
    62	
    63	  type-check:
    64	    name: Type Check (core)
    65	    runs-on: ubuntu-latest
    66	    steps:
    67	      - uses: actions/checkout@v4
    68	      - uses: actions/setup-python@v5
    69	        with:
    70	          python-version: "3.11"
    71	      - run: pip install -e ".[dev]"
    72	      # The federation carve moved these four out of src/legoesm/; the stale
    73	      # paths meant mypy exited "file not found" and `|| true` swallowed it, so
    74	      # this job typechecked NOTHING. mypy itself stays advisory (the tree is
    75	      # not clean yet), but a missing target is now a hard failure so the next
    76	      # move cannot silently blank the job again.
    77	      - name: mypy on core modules
    78	        run: |
    79	          set -eu
    80	          dirs="packages/core/legoesm/core packages/core/legoesm/io \
    81	                packages/core/legoesm/parallel packages/coupler/legoesm/driver"
    82	          for d in $dirs; do
    83	            test -d "$d" || { echo "ERROR: mypy target '$d' does not exist (moved?)"; exit 1; }
    84	          done
    85	          mypy $dirs --ignore-missing-imports || true
    86	
    87	  test-collect:
    88	    name: Test Collection
    89	    runs-on: ubuntu-latest
    90	    steps:
    91	      - uses: actions/checkout@v4
    92	      - uses: actions/setup-python@v5
    93	        with:
    94	          python-version: "3.11"
    95	      - run: pip install -e ".[dev]"
    96	      - name: Verify all tests collect
    97	        run: python -m pytest --collect-only tests/ 2>&1
    98	
    99	  unit-tests:
   100	    name: Unit Tests
   101	    runs-on: ubuntu-latest
   102	    needs: [install-smoke, test-collect]
   103	    strategy:
   104	      matrix:
   105	        python-version: ["3.11"]
   106	    steps:
   107	      - uses: actions/checkout@v4
   108	      - uses: actions/setup-python@v5
   109	        with:
   110	          python-version: ${{ matrix.python-version }}
   111	      - run: pip install -e ".[dev]"
   112	      # Deliberately NO JAX_ENABLE_X64 here: the unit tier is fp32-by-default
   113	      # (CLAUDE.md: "finite-volume can float32"); unit tests that need x64
   114	      # self-enable it at module scope (e.g. test_spectral_plane_ops.py,
   115	      # test_precision_modes.py). Setting it job-wide would change what the
   116	      # fp32 paths test. x64 jobs: top-level-fidelity-tests, visual-regression.
   117	      # Known wart: a module-level jax.config.update("jax_enable_x64", True)
   118	      # leaks across the pytest session for subsequently-imported modules.
   119	      - name: Run unit tests
   120	        run: |
   121	          python -m pytest tests/unit/ -x --timeout=300 \
   122	            -k "not test_coupler_with_3d_ocean_fc_gram" \
   123	            --tb=short -q
   124	        timeout-minutes: 30
   125	
   126	  top-level-fidelity-tests:
   127	    # Iter-874 (Codex iter-873 stop-time fix): the iter chain's
   128	    # Fortran-fidelity tests live at top-level `tests/test_*.py`
   129	    # (not under `tests/unit/`).  The pre-iter-874 CI ran only
   130	    # `tests/unit/` and `tests/atmosphere/`, so top-level tests
   131	    # would not catch regressions despite passing locally.  This
   132	    # job explicitly executes the top-level tests/test_*.py files
   133	    # so the iter-862-873 chain regression sentinels actually run
   134	    # on every push.
   135	    #
   145	        python-version: ["3.11"]
   146	    steps:
   147	      - uses: actions/checkout@v4
   148	      - uses: actions/setup-python@v5
   149	        with:
   150	          python-version: ${{ matrix.python-version }}
   151	      - run: pip install -e ".[dev]"
   152	      - name: Run top-level tests/test_*.py
   153	        env:
   154	          JAX_ENABLE_X64: "1"
   155	        # Use shell glob to restrict to top-level files only.
   156	        # `bash -c` is required so the glob expands before pytest
   157	        # sees the arguments; otherwise pytest gets the literal
   158	        # `tests/test_*.py` and reports "no such file".
   159	        run: |
   160	          shopt -s nullglob
   161	          files=( tests/test_*.py )
   162	          if [ ${#files[@]} -eq 0 ]; then
   163	            echo "ERROR: no top-level tests/test_*.py files matched"
   164	            exit 1
   165	          fi
   166	          echo "Running ${#files[@]} top-level test files:"
   167	          printf '  %s\n' "${files[@]}"
   168	          # No -x: these files are the AST ratchets (constants, param specs,
   169	          # dispatch hardening, scripts layout, ...). Stopping at the first
   170	          # failure reports one violation per run and hides the rest, which is
   171	          # how ~30 of them accumulated unnoticed.
   172	          #
   173	          # -n 4 --dist loadfile: spread the files over four WORKER PROCESSES.
   174	          # Run as one process, this set reliably dies partway through with
   175	          # "Fatal Python error: Segmentation fault" inside XLA compilation of a
   176	          # reverse-mode HLO (see docs/dev-notes/xla-compile-crash.md).  Every
   177	          # subset passes on its own, so breaking the sequence up avoids it —
   178	          # and it is ~4x faster.  NOTE this is a mitigation, not a guarantee:
   179	          # loadfile keeps a worker alive across the files it is handed, so
   180	          # process state still accumulates within a worker.  If the crash
   181	          # returns, the deterministic fallback is one pytest process per file.
   182	          python -m pytest "${files[@]}" --timeout=300 \
   183	            --tb=short -q -n 4 --dist loadfile
   184	        shell: bash
   185	        timeout-minutes: 30
   186	      - name: FV3-native grid fidelity tests (phase 1)
   187	        # Source-derived FV3 gnomonic_ed oracle + static grid-provenance
   188	        # gates. tests/grids/ has no other CI job; without this step the
   189	        # phase-1 fidelity tests are collected but never executed.
   190	        # iter62 (fixed create-layout centres) + iter73 (seam no-collapse)
   191	        # protect the _GNOMONIC_ED_FACE_PERM/_ROT remap TOPOLOGY, which the
   192	        # unordered point-cloud oracle test deliberately cannot see.
   193	        env:
   194	          JAX_ENABLE_X64: "1"
   195	        run: |
   300	
   301	[tool.mypy]
   302	python_version = "3.11"
   303	warn_return_any = true
   304	warn_unused_configs = true
   305	disallow_untyped_defs = false
   306	# Vendored 3rd-party CLM-ML-JAX backend (BSD-3) — not type-annotated to legoESM's
   307	# standard; excluded from type-checking here (audited upstream).
   308	exclude = ['packages/land/legoesm/land/canopy/clm_ml_backend/']
   309	
   310	[tool.pytest.ini_options]
   311	testpaths = ["tests"]
   312	addopts = "-v --tb=short -m 'not slow'"
   313	markers = [
   314	    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
   315	    # Complexity-tier ladder (docs/validation/TESTING.md). Opt-in selectors: run a rung
   316	    # with e.g. `-m 'tier1 and not slow'`. The default addopts stays `not slow`
   317	    # (does NOT tier-gate) so untagged tests still run until the suite is tagged.
   318	    "tier0: unit/operator tests — kernels, no model integration (numerical invariants)",
   319	    "tier1: research tier — idealized/column/shallow-water; mass+energy+AAM gates, analytic benchmarks",
   320	    "tier2: intermediate tier — hydrostatic 3D + slab; mass+energy+moisture gates (often slow)",
   321	    "tier3: operational tier — full complexity + real forcing (AMIP/OMIP/ERA5); budget closure (slow)",
   322	]
   323	
   324	# ---------------------------------------------------------------------------
   325	# Import-boundary enforcement (the path to independent self-running packages).
   326	# These contracts encode the TARGET layered architecture (master plan D7):
   327	#   core (substrate)  <  {atmosphere, ocean, land, ice}  <  coupler  <  driver
   328	# Each Earth-system component must depend only on `core` (+ truly-shared bricks),
   329	# never on another component or on the coupler/driver — that one-way dependency
   330	# is exactly what lets `pip install legoesm-ocean` run standalone (Stage D carve).
   331	#
   332	# `ignore_imports` is the CURRENT-violation BASELINE: the contracts pass today,
   333	# and every entry removed is a real step toward independence (a ratchet).  The
   334	# goal is an empty ignore list = full component independence, enforced in CI.
   335	# Run: `.venv/bin/lint-imports`
   336	# ---------------------------------------------------------------------------
   337	[tool.importlinter]
   338	root_package = "legoesm"
   339	
   340	# === ROOT INVERSION — the highest-priority independence fix ===
   341	# A handful of low-level / shared modules import UP into atmosphere/ocean
   342	# component dynamics; because `grids.halo_latlon -> parallel.latlon_mpi`, that
   343	# single inversion cascades into nearly every cross-component and core→component
   344	# chain.  Fixing these collapses most of the graph:
   345	#   parallel.latlon_mpi  -> atmosphere.dynamics.primitive_eq_latlon_cgrid, ocean.state

 exited 2 in 238ms:
 succeeded in 492ms:
 M scripts/bench/bench_voronoi_partition_methods.py
 M tests/bench/test_bench_voronoi_partition_methods.py
?? scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
?? tests/bench/test_mpas_schedule_cost_scan_sbatch.py
./tests/bench/test_mpas_schedule_cost_scan_sbatch.py:10:The script is driven with a STUB bench (``LEGOESM_SCHEDULE_COST_BENCH``) whose
./tests/bench/test_mpas_schedule_cost_scan_sbatch.py:47:    env["LEGOESM_SCHEDULE_COST_BENCH"] = str(stub)
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:106:BENCH="${LEGOESM_SCHEDULE_COST_BENCH:-scripts/bench/bench_voronoi_partition_methods.py}"

codex
## Findings

1. **HIGH — the test hook can silently replace the production instrument.**  
   [`BENCH` trusts `LEGOESM_SCHEDULE_COST_BENCH`](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:102) and executes it as the benchmark at [line 119](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:119). A stray exported value can make all three stub calls return zero, print `SCAN_DONE`, and leave no real scan results. This directly reintroduces the “SLURM COMPLETED is a lie” failure class. Remove the production override; test through an interpreter wrapper or another test-only mechanism.

2. **HIGH — the required shell regression test is not protected by CI.**  
   The test is skipped unless a repository-local `.venv/bin/python` exists ([lines 60–61](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:60)), while CI installs into the Actions Python environment ([ci.yml:91–97](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:91)) and only collects all tests ([line 97](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:97)); its test execution shown here is `tests/unit/` ([line 121](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:121)). Thus this new regression guard will not run in current CI.

3. **MEDIUM — `--methods` can still produce a successful empty benchmark.**  
   [`--methods " , , "` becomes `[]`](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:367), has no non-empty validation, skips the measurement loop ([lines 414–421](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:414)), writes `rows: []`, and returns zero. `--rank-counts` is correctly guarded, but methods need the same non-empty requirement and a regression test.

4. **MEDIUM — the new shell test is real, but still partly vacuous.**  
   It does execute the actual `.sbatch` file, not a copy ([lines 53–55](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:53)); its status sequencing is sound because each serial call appends before counting ([lines 40–44](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:40)). The arm-3 test would fail under the original lost-exit-status regression: it demands exact exit `3` and no `SCAN_DONE` ([lines 79–85](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:79)).

   But validation failure is tested only for arm 1 ([line 69](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:69)), not arm 2; and the stub records only `--subdivision` ([line 40](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:40)). Dropping `--expect-rounds` or `--schedule-cost` from validation arms would still pass these tests. Add an arm-2 failure case, assert validation ran exactly `["8", "9"]`, and assert each arm’s required arguments.

Verified fixes:

- Whitespace-only `--expect-rounds` is now rejected: the raw-string guard is correct ([lines 223–230](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:223)). I found no remaining whitespace bypass of that gate.
- Negative Lloyd, subdivision, and halo depth are all rejected before mesh construction ([lines 377–392](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:377)).
- The unavailable-METIS test now requires the explicit unavailable row ([lines 421–436](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:421)).
- Shell and Python syntax checks pass. Full pytest could not run in this read-only environment because no temporary directory is writable.

The single s8 geometric@64 match is a useful spot check, not instrument validation. The “reference census” is embedded in the scorer’s own docstring ([sharded_dynamics.py:1323–1325](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1323)); one matching tuple can arise from a wrong graph, hard-coded behavior, or an ignored option. Treat it as validated only after the full independent six-row arm checks complete and the real benchmark JSON records a passing gate.

**NOT READY**
tokens used
90,119
## Findings

1. **HIGH — the test hook can silently replace the production instrument.**  
   [`BENCH` trusts `LEGOESM_SCHEDULE_COST_BENCH`](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:102) and executes it as the benchmark at [line 119](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:119). A stray exported value can make all three stub calls return zero, print `SCAN_DONE`, and leave no real scan results. This directly reintroduces the “SLURM COMPLETED is a lie” failure class. Remove the production override; test through an interpreter wrapper or another test-only mechanism.

2. **HIGH — the required shell regression test is not protected by CI.**  
   The test is skipped unless a repository-local `.venv/bin/python` exists ([lines 60–61](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:60)), while CI installs into the Actions Python environment ([ci.yml:91–97](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:91)) and only collects all tests ([line 97](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:97)); its test execution shown here is `tests/unit/` ([line 121](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:121)). Thus this new regression guard will not run in current CI.

3. **MEDIUM — `--methods` can still produce a successful empty benchmark.**  
   [`--methods " , , "` becomes `[]`](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:367), has no non-empty validation, skips the measurement loop ([lines 414–421](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:414)), writes `rows: []`, and returns zero. `--rank-counts` is correctly guarded, but methods need the same non-empty requirement and a regression test.

4. **MEDIUM — the new shell test is real, but still partly vacuous.**  
   It does execute the actual `.sbatch` file, not a copy ([lines 53–55](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:53)); its status sequencing is sound because each serial call appends before counting ([lines 40–44](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:40)). The arm-3 test would fail under the original lost-exit-status regression: it demands exact exit `3` and no `SCAN_DONE` ([lines 79–85](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:79)).

   But validation failure is tested only for arm 1 ([line 69](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:69)), not arm 2; and the stub records only `--subdivision` ([line 40](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:40)). Dropping `--expect-rounds` or `--schedule-cost` from validation arms would still pass these tests. Add an arm-2 failure case, assert validation ran exactly `["8", "9"]`, and assert each arm’s required arguments.

Verified fixes:

- Whitespace-only `--expect-rounds` is now rejected: the raw-string guard is correct ([lines 223–230](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:223)). I found no remaining whitespace bypass of that gate.
- Negative Lloyd, subdivision, and halo depth are all rejected before mesh construction ([lines 377–392](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:377)).
- The unavailable-METIS test now requires the explicit unavailable row ([lines 421–436](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:421)).
- Shell and Python syntax checks pass. Full pytest could not run in this read-only environment because no temporary directory is writable.

The single s8 geometric@64 match is a useful spot check, not instrument validation. The “reference census” is embedded in the scorer’s own docstring ([sharded_dynamics.py:1323–1325](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1323)); one matching tuple can arise from a wrong graph, hard-coded behavior, or an ignored option. Treat it as validated only after the full independent six-row arm checks complete and the real benchmark JSON records a passing gate.

**NOT READY**
