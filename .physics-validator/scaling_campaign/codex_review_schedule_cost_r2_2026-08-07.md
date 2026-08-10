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
session id: 019fdbcc-b97e-76f3-90c0-5d389e968b77
--------
user
ADVERSARIAL REVIEW, ROUND 2. You reviewed this change in round 1 and found defects. I applied fixes. Verify each fix is REAL and complete, then hunt for NEW defects introduced by the fixes. Prefer refutation; do not summarize what changed.

Files: `git diff` on scripts/bench/bench_voronoi_partition_methods.py and tests/bench/test_bench_voronoi_partition_methods.py, plus the untracked scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch.

ROUND-1 FINDINGS AND WHAT I CLAIM I DID:
1. BLOCKER 'validation is only a comment' -> added --expect-rounds: parse_expect_rounds() + check_expected_rounds(), main() returns 1 on mismatch AND on a listed pair that was never scored. sbatch now captures rc per arm (local rc=$? immediately after the command), gates arm 3 behind RC8/RC9, exits 1 with SCAN_ABORTED_VALIDATION. VERIFY: is the rc capture actually correct given `local rc=$?` on a line where `local` itself sets $?; does ${4:+--expect-rounds "$4"} quote correctly under set -u; can arm 3 still run after a failure by any path.
2. 'gap<=1 => ownership only is false' -> sbatch decision rule rewritten: gap==0 provably worthless AT THIS OWNERSHIP, gap==1 INCONCLUSIVE, gap>=2 confirms. VERIFY the sbatch text no longer asserts the false claim anywhere.
3. 'coloring_headroom_rounds documented and printed BACKWARDS' -> you were right, gap-1 is a LOWER bound on the achievable reduction, not the max. Replaced with an interval: coloring_headroom_rounds_min = max(0, gap-1), coloring_headroom_rounds_max = gap. Print is now 'recolour headroom MIN-MAX round(s)'. VERIFY the direction is now correct in code, docstring, print AND test, and that (14,10) really means at-least-3 at-most-4.
4. 'Delta is not a lower bound for a redesigned DIRECTED schedule' -> added an explicit SCOPE paragraph to the module docstring and the sbatch saying gap==0 rules out only a better UNDIRECTED edge colouring of this graph, because comm_pairs enters a pair when EITHER direction has a dependency and then emits both directions. VERIFY I stated it correctly and did not re-introduce the overclaim elsewhere.
5. 'provenance false at row level' -> schedule_cost_row now carries reorder_target and already_reordered per row. VERIFY that is sufficient to establish a row's provenance, or name what is still missing (note lloyd_iterations is still only in enclosing metadata - is that actually a defect, given the JSON is one file per mesh?).
6. '--lloyd test is vacuous' -> new test_lloyd_flag_reaches_the_mesh_builder monkeypatches legoesm.grids.voronoi.create_voronoi_mesh and asserts lloyd_iterations==0 at a NON-default value. VERIFY the spy actually intercepts the call the bench makes (check the import style in the bench - if it imported the symbol directly rather than the module, the patch may not take effect and the test would be vacuous AGAIN).
7. '2560 vs 2561 cells/device' -> corrected in the sbatch comment, and I added that np8's rounds==n_dev-1 is complete-graph saturation.

ALSO CHECK:
- The new --expect-rounds census values I hardcoded in the sbatch (S8_EXPECT/S9_EXPECT) against spmd_schedule_cost's docstring census. A transcription error there turns the gate into a false alarm or, worse, a false pass.
- Whether check_expected_rounds can pass VACUOUSLY in any path (empty expect, rows missing 'schedule', method reported unavailable).
- Any test that passes for the wrong reason, and any NEW vacuity I introduced.
- Whether returning 1 from main() but still writing the JSON is the right failure semantics.

Cite file:line. Be blunt.
codex
I’ll adversarially inspect the exact diff and untracked batch script, trace the execution/error paths, and cross-check the census against the cost-function documentation.
exec
/usr/bin/bash -lc "git status --short && git diff -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py && sed -n '1,260p' scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && rg -n -C 4 \"spmd_schedule_cost|census|rounds|expect-rounds|check_expected_rounds|parse_expect_rounds|schedule_cost_row|coloring_headroom|lloyd\" scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 522ms:
 M scripts/bench/bench_voronoi_partition_methods.py
 M tests/bench/test_bench_voronoi_partition_methods.py
?? .physics-validator/flag_reachability_audit/
?? .physics-validator/homogeneous_ice_nucleation/
?? .physics-validator/scaling_campaign/codex_consult_hundreds_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r2_2026-08-07.md
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
diff --git a/scripts/bench/bench_voronoi_partition_methods.py b/scripts/bench/bench_voronoi_partition_methods.py
index 6aa0f4a40..4da0bd0a8 100644
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
 
@@ -128,6 +181,127 @@ def owner_for(mesh, method: str, n_ranks: int) -> np.ndarray:
                      f"expected one of {METHODS}")
 
 
+def parse_expect_rounds(spec: str) -> dict:
+    """Parse ``'sfc:64=12,metis:128=19'`` into ``{("sfc", 64): 12}``.
+
+    Raises on anything malformed rather than skipping it — a typo'd
+    expectation that is silently dropped turns the gate into a no-op, which
+    is exactly the failure this flag exists to prevent.
+    """
+    out: dict[tuple[str, int], int] = {}
+    for item in (s.strip() for s in spec.split(",")):
+        if not item:
+            continue
+        try:
+            lhs, rounds = item.split("=")
+            method, n_ranks = lhs.split(":")
+            key = (method.strip(), int(n_ranks))
+            out[key] = int(rounds)
+        except ValueError as exc:
+            raise ValueError(
+                f"--expect-rounds: cannot parse {item!r}; expected "
+                f"'<method>:<n_ranks>=<n_rounds>'") from exc
+        if key[0] not in METHODS:
+            raise ValueError(
+                f"--expect-rounds: unknown method {key[0]!r} in {item!r}; "
+                f"expected one of {METHODS}")
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
@@ -138,6 +312,26 @@ def main() -> int:
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
+                        "match the prewarmed cache key at subdiv>=9.")
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
@@ -149,12 +343,19 @@ def main() -> int:
             raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
     if not rank_counts or any(n < 2 for n in rank_counts):
         raise SystemExit("--rank-counts needs integers >= 2")
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
@@ -185,6 +386,23 @@ def main() -> int:
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
@@ -204,7 +422,9 @@ def main() -> int:
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
@@ -213,6 +433,24 @@ def main() -> int:
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
index 21f861fd7..3a0549233 100644
--- a/tests/bench/test_bench_voronoi_partition_methods.py
+++ b/tests/bench/test_bench_voronoi_partition_methods.py
@@ -146,3 +146,225 @@ def test_halo_matches_runtime_partition():
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
#!/bin/bash -l
#SBATCH --job-name=mpas_sched_cost
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=mpas_schedule_cost_scan.%j.log
# ===========================================================================
# Does RECOLOURING still have room on the MPAS halo schedule, or is only a
# new OWNERSHIP objective left?  CPU-only, ZERO GPU hours.
#
# WHY THIS RUN EXISTS
# MPAS GPU is the worst-scaling lane we have: measured/modelled-bound 3.16x
# (s8@16) to 4.47x (s9@64), and one halo fill costs 12-14 SEQUENTIAL ppermute
# rounds.  Both independent reviews (codex + GLM, 2026-08-07) ranked cutting
# that round count as the top structural lever, at 300-800 LOC and 7-14 days
# for a partitioner with a new objective.  That estimate is only worth
# spending if recolouring is genuinely exhausted, and the colourer's own
# lower bound decides it.
#
# The schedule is a proper EDGE colouring of the device communication graph
# (one colour = one ppermute round) and max_degree is that graph's maximum
# vertex degree, so VIZING bounds the chromatic index: Delta <= chi' <=
# Delta + 1.  Read coloring_gap = n_rounds - max_degree through that:
#     gap == 0 -> PROVABLY OPTIMAL; recolouring headroom is exactly zero.
#     gap == 1 -> indistinguishable from optimal (a Class 2 graph really
#                 needs Delta+1, and deciding Class 1/2 is NP-complete);
#                 nothing provable to win.
#     gap >= 2 -> at least gap-1 rounds of genuine recolouring headroom.
# So recolouring is capped at ~1 round out of 12-14 (<= ~7%) wherever the
# multi-start search lands on Delta or Delta+1 — which it does on every
# configuration probed so far.  This run measures whether the production
# working points are in that regime.
#
# (a) NUMBER PRODUCED: n_rounds, max_degree and their gap per
#     (method x n_dev) at the production working points.
# (b) CONFIRMS a cheap fix: gap >= 2 at any production (ppermute) row
#     (recolouring then removes at least gap-1 rounds, guaranteed).
#     gap == 0: recolouring is provably worthless AT THIS OWNERSHIP.
#     gap == 1: INCONCLUSIVE — Class 1 vs Class 2 is NP-complete, so this
#     neither confirms nor refutes a one-round win.
#     Note what a gap of 0 does NOT prove: Delta bounds only a proper
#     UNDIRECTED edge colouring of this graph, and the builder forces BOTH
#     ppermute directions per pair even when one send map is empty, so a
#     redesigned DIRECTED schedule is not bounded by Delta.  "Ownership is
#     the only path" would overclaim; the honest statement is "recolouring
#     this undirected graph is exhausted".
# (c) WHY NOT CHEAPER: this IS the cheap test — no GPU, no MPI, no model
#     step.  It cannot be shrunk further onto small meshes: measured
#     2026-08-07, L2/L4 x {geometric,sfc} x nd 2-16 all report gap == 0,
#     but EVERY one of those rows auto-selects the ALLGATHER strategy
#     (cells/device below the threshold), so production runs no ppermute
#     schedule there and the number is counterfactual.  Only meshes big
#     enough to keep cells/device above the threshold answer the question.
#     s6 lloyd=0 np8/np16 (2561 cells/device after padding, above the 2000
#     ppermute threshold, so genuinely ppermute) also gave gap == 0 on all
#     three methods — rounds 7/7/6 at np8 and 13/10/10 at np16 for
#     geometric/sfc/metis.  np8's 7 == n_dev-1 is complete-graph
#     saturation and says nothing; np16 (13/10/10 < 15) does escape it.
#     Still a SMALL working point; s8-s10 are the production ones.
#
# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
# Arms 1-2 must REPRODUCE those before arm 3's unknown s10 number is quoted.
# A scan that misses the known answer is a broken instrument, and its s10
# row must not be believed.
#
# lloyd=0 throughout: it is what the reference census used AND the cache key
# the prewarmed s8/s9/s10 meshes were written under.  It is the LABELLED
# synthetic scaling mesh, recorded in every row's metadata so it can never be
# read back as a production SCVT receipt.
#
# Arms run cheapest-first and each writes its own JSON, so a later arm that
# runs out of time or memory cannot lose an earlier arm's result.  s10@128 is
# last and is the one genuinely at risk: the scorer is known not to have
# finished at subdiv-8@128 on a laptop, which is why this asks for 24 h.
#
# SUBMIT (from the repo root):
#   sbatch scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

OUT=results/a1/mpas_schedule_cost
mkdir -p "$OUT"
BENCH=scripts/bench/bench_voronoi_partition_methods.py

# pymetis is present in .venv but was ABSENT from .venv-mpi for a while, and a
# missing partitioner is reported as "unavailable" rather than substituted.
# Say which python and whether metis is really there, so a two-method table
# cannot be misread as a three-method one.
echo "[scan] python=$PY"
"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"

run_arm () {  # $1=level  $2=rank-counts  $3=label  $4=expect-rounds (may be "")
  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
  "$PY" "$BENCH" \
      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
      --methods geometric,sfc,metis --schedule-cost \
      ${4:+--expect-rounds "$4"} \
      --out "$OUT/schedule_cost_s$1.json"
  local rc=$?   # capture BEFORE anything else runs, or `date` overwrites it
  echo "[scan] arm $3 exit=$rc"
  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
  return $rc
}

# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
# makes the bench exit non-zero unless it reproduces the census below, and a
# listed pair that never got scored (e.g. pymetis missing) counts as a
# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
# pass — an instrument that misses the known answer cannot be trusted on the
# unknown one.
S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"

run_arm 8 64,128 "1/3 VALIDATION s8" "$S8_EXPECT"; RC8=$?
run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?

if [ "$RC8" -ne 0 ] || [ "$RC9" -ne 0 ]; then
  echo "[scan] INSTRUMENT CHECK FAILED (s8 rc=$RC8, s9 rc=$RC9)."
  echo "[scan] The scorer does not reproduce the reference census, so an s10"
  echo "[scan] number from it would be untrustworthy. NOT running arm 3."
  echo "SCAN_ABORTED_VALIDATION"
  exit 1
fi

# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
run_arm 10 128 "3/3 UNKNOWN s10" ""; RC10=$?
echo "[scan] arm 3 rc=$RC10"

echo "SCAN_DONE"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-13-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-14-# WHY THIS RUN EXISTS
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-15-# MPAS GPU is the worst-scaling lane we have: measured/modelled-bound 3.16x
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-16-# (s8@16) to 4.47x (s9@64), and one halo fill costs 12-14 SEQUENTIAL ppermute
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:17:# rounds.  Both independent reviews (codex + GLM, 2026-08-07) ranked cutting
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-18-# that round count as the top structural lever, at 300-800 LOC and 7-14 days
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-19-# for a partitioner with a new objective.  That estimate is only worth
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-20-# spending if recolouring is genuinely exhausted, and the colourer's own
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-21-# lower bound decides it.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-22-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-23-# The schedule is a proper EDGE colouring of the device communication graph
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-24-# (one colour = one ppermute round) and max_degree is that graph's maximum
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-25-# vertex degree, so VIZING bounds the chromatic index: Delta <= chi' <=
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:26:# Delta + 1.  Read coloring_gap = n_rounds - max_degree through that:
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-27-#     gap == 0 -> PROVABLY OPTIMAL; recolouring headroom is exactly zero.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-28-#     gap == 1 -> indistinguishable from optimal (a Class 2 graph really
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-29-#                 needs Delta+1, and deciding Class 1/2 is NP-complete);
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-30-#                 nothing provable to win.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:31:#     gap >= 2 -> at least gap-1 rounds of genuine recolouring headroom.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-32-# So recolouring is capped at ~1 round out of 12-14 (<= ~7%) wherever the
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-33-# multi-start search lands on Delta or Delta+1 — which it does on every
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-34-# configuration probed so far.  This run measures whether the production
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-35-# working points are in that regime.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-36-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:37:# (a) NUMBER PRODUCED: n_rounds, max_degree and their gap per
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-38-#     (method x n_dev) at the production working points.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-39-# (b) CONFIRMS a cheap fix: gap >= 2 at any production (ppermute) row
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:40:#     (recolouring then removes at least gap-1 rounds, guaranteed).
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-41-#     gap == 0: recolouring is provably worthless AT THIS OWNERSHIP.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-42-#     gap == 1: INCONCLUSIVE — Class 1 vs Class 2 is NP-complete, so this
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-43-#     neither confirms nor refutes a one-round win.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-44-#     Note what a gap of 0 does NOT prove: Delta bounds only a proper
--
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-53-#     but EVERY one of those rows auto-selects the ALLGATHER strategy
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-54-#     (cells/device below the threshold), so production runs no ppermute
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-55-#     schedule there and the number is counterfactual.  Only meshes big
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-56-#     enough to keep cells/device above the threshold answer the question.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:57:#     s6 lloyd=0 np8/np16 (2561 cells/device after padding, above the 2000
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-58-#     ppermute threshold, so genuinely ppermute) also gave gap == 0 on all
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:59:#     three methods — rounds 7/7/6 at np8 and 13/10/10 at np16 for
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-60-#     geometric/sfc/metis.  np8's 7 == n_dev-1 is complete-graph
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-61-#     saturation and says nothing; np16 (13/10/10 < 15) does escape it.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-62-#     Still a SMALL working point; s8-s10 are the production ones.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-63-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:64:# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:65:# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:66:#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-67-#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-68-# Arms 1-2 must REPRODUCE those before arm 3's unknown s10 number is quoted.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-69-# A scan that misses the known answer is a broken instrument, and its s10
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-70-# row must not be believed.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-71-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:72:# lloyd=0 throughout: it is what the reference census used AND the cache key
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-73-# the prewarmed s8/s9/s10 meshes were written under.  It is the LABELLED
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-74-# synthetic scaling mesh, recorded in every row's metadata so it can never be
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-75-# read back as a production SCVT receipt.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-76-#
--
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-102-echo "[scan] python=$PY"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-103-"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-104-  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-105-
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:106:run_arm () {  # $1=level  $2=rank-counts  $3=label  $4=expect-rounds (may be "")
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:107:  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-108-  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-109-  "$PY" "$BENCH" \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:110:      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-111-      --methods geometric,sfc,metis --schedule-cost \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:112:      ${4:+--expect-rounds "$4"} \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-113-      --out "$OUT/schedule_cost_s$1.json"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-114-  local rc=$?   # capture BEFORE anything else runs, or `date` overwrites it
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-115-  echo "[scan] arm $3 exit=$rc"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-116-  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-117-  return $rc
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-118-}
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-119-
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:120:# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:121:# makes the bench exit non-zero unless it reproduces the census below, and a
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-122-# listed pair that never got scored (e.g. pymetis missing) counts as a
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-123-# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-124-# pass — an instrument that misses the known answer cannot be trusted on the
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-125-# unknown one.
--
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-130-run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-131-
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-132-if [ "$RC8" -ne 0 ] || [ "$RC9" -ne 0 ]; then
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-133-  echo "[scan] INSTRUMENT CHECK FAILED (s8 rc=$RC8, s9 rc=$RC9)."
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:134:  echo "[scan] The scorer does not reproduce the reference census, so an s10"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-135-  echo "[scan] number from it would be untrustworthy. NOT running arm 3."
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-136-  echo "SCAN_ABORTED_VALIDATION"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-137-  exit 1
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-138-fi
--
tests/bench/test_bench_voronoi_partition_methods.py-150-
tests/bench/test_bench_voronoi_partition_methods.py-151-# --- SPMD halo-schedule depth (--schedule-cost) ---------------------------
tests/bench/test_bench_voronoi_partition_methods.py-152-
tests/bench/test_bench_voronoi_partition_methods.py-153-
tests/bench/test_bench_voronoi_partition_methods.py:154:def test_schedule_cost_row_reports_rounds_against_their_lower_bound():
tests/bench/test_bench_voronoi_partition_methods.py:155:    """``schedule_cost_row`` returns the REAL schedule depth and the bound it
tests/bench/test_bench_voronoi_partition_methods.py-156-    must be read against."""
tests/bench/test_bench_voronoi_partition_methods.py-157-    mesh = _mesh()
tests/bench/test_bench_voronoi_partition_methods.py:158:    sc = mod.schedule_cost_row(mesh, "geometric", 2)
tests/bench/test_bench_voronoi_partition_methods.py-159-    # max_degree is the graph's own lower bound on a proper edge colouring,
tests/bench/test_bench_voronoi_partition_methods.py-160-    # so a schedule can never beat it.  -1 is the scorer's "not reported"
tests/bench/test_bench_voronoi_partition_methods.py-161-    # sentinel and would make the gap meaningless.
tests/bench/test_bench_voronoi_partition_methods.py-162-    assert sc["max_degree"] >= 1
tests/bench/test_bench_voronoi_partition_methods.py:163:    assert sc["n_rounds"] >= sc["max_degree"]
tests/bench/test_bench_voronoi_partition_methods.py:164:    assert sc["coloring_gap"] == sc["n_rounds"] - sc["max_degree"]
tests/bench/test_bench_voronoi_partition_methods.py:165:    assert sc["n_rounds_greedy"] >= sc["n_rounds"]
tests/bench/test_bench_voronoi_partition_methods.py-166-    assert sc["score_seconds"] >= 0.0
tests/bench/test_bench_voronoi_partition_methods.py-167-
tests/bench/test_bench_voronoi_partition_methods.py-168-
tests/bench/test_bench_voronoi_partition_methods.py-169-@pytest.mark.parametrize(
tests/bench/test_bench_voronoi_partition_methods.py:170:    "n_rounds, max_degree, gap, hmin, hmax, proven",
tests/bench/test_bench_voronoi_partition_methods.py-171-    [
tests/bench/test_bench_voronoi_partition_methods.py-172-        # Provably optimal: no proper edge colouring beats max_degree.
tests/bench/test_bench_voronoi_partition_methods.py-173-        (12, 12, 0, 0, 0, True),
tests/bench/test_bench_voronoi_partition_methods.py-174-        # Vizing allows the true optimum to BE max_degree+1, so a gap of 1
--
tests/bench/test_bench_voronoi_partition_methods.py-178-        # Guaranteed to remove at least gap-1 = 3, at most gap = 4.
tests/bench/test_bench_voronoi_partition_methods.py-179-        (14, 10, 4, 3, 4, False),
tests/bench/test_bench_voronoi_partition_methods.py-180-    ])
tests/bench/test_bench_voronoi_partition_methods.py-181-def test_coloring_gap_and_headroom_are_derived_not_assumed(
tests/bench/test_bench_voronoi_partition_methods.py:182:        monkeypatch, n_rounds, max_degree, gap, hmin, hmax, proven):
tests/bench/test_bench_voronoi_partition_methods.py-183-    """Gap and the Vizing-bounded headroom must be COMPUTED, not assumed.
tests/bench/test_bench_voronoi_partition_methods.py-184-
tests/bench/test_bench_voronoi_partition_methods.py-185-    Non-vacuity, the hard way: on every mesh small enough to test quickly the
tests/bench/test_bench_voronoi_partition_methods.py-186-    real gap is 0 (measured L2/L3/L4 x {geometric,sfc} x nd 2-16, and s6
tests/bench/test_bench_voronoi_partition_methods.py:187:    lloyd=0 at np8/np16 — the colourer lands exactly on ``max_degree`` every
tests/bench/test_bench_voronoi_partition_methods.py-188-    time), so a real-mesh assertion cannot tell a correct subtraction from a
tests/bench/test_bench_voronoi_partition_methods.py-189-    hardcoded ``0``; that exact mutation passed the first version of this
tests/bench/test_bench_voronoi_partition_methods.py-190-    test.  Stubbing the production scorer with KNOWN values is what makes
tests/bench/test_bench_voronoi_partition_methods.py-191-    the assertion able to fail.
--
tests/bench/test_bench_voronoi_partition_methods.py-196-    indistinguishable from optimal (Class 2), and claiming recolouring
tests/bench/test_bench_voronoi_partition_methods.py-197-    headroom there would be an over-claim.
tests/bench/test_bench_voronoi_partition_methods.py-198-    """
tests/bench/test_bench_voronoi_partition_methods.py-199-    stub = {
tests/bench/test_bench_voronoi_partition_methods.py:200:        "n_rounds": n_rounds, "max_degree": max_degree,
tests/bench/test_bench_voronoi_partition_methods.py:201:        "n_rounds_greedy": n_rounds + 1, "coloring_method": "stub",
tests/bench/test_bench_voronoi_partition_methods.py-202-        "resolved_method": "geometric", "halo_depth": 3,
tests/bench/test_bench_voronoi_partition_methods.py-203-        "cells_per_device": 99_999, "production_strategy": "ppermute",
tests/bench/test_bench_voronoi_partition_methods.py-204-        "reorder_target": 8, "already_reordered": False,
tests/bench/test_bench_voronoi_partition_methods.py-205-    }
tests/bench/test_bench_voronoi_partition_methods.py-206-    import legoesm.parallel.sharded_dynamics as sd
tests/bench/test_bench_voronoi_partition_methods.py:207:    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
tests/bench/test_bench_voronoi_partition_methods.py-208-
tests/bench/test_bench_voronoi_partition_methods.py:209:    sc = mod.schedule_cost_row(object(), "geometric", 8)
tests/bench/test_bench_voronoi_partition_methods.py-210-    assert sc["coloring_gap"] == gap
tests/bench/test_bench_voronoi_partition_methods.py-211-    # The headroom is an INTERVAL: Vizing pins the optimum to
tests/bench/test_bench_voronoi_partition_methods.py-212-    # {Delta, Delta+1}, so gap-1 is guaranteed and gap is the best case.
tests/bench/test_bench_voronoi_partition_methods.py:213:    assert sc["coloring_headroom_rounds_min"] == hmin
tests/bench/test_bench_voronoi_partition_methods.py:214:    assert sc["coloring_headroom_rounds_max"] == hmax
tests/bench/test_bench_voronoi_partition_methods.py-215-    assert sc["coloring_optimal_proven"] is proven
tests/bench/test_bench_voronoi_partition_methods.py-216-
tests/bench/test_bench_voronoi_partition_methods.py-217-
tests/bench/test_bench_voronoi_partition_methods.py:218:def test_schedule_rounds_never_beat_the_vizing_floor_on_a_real_mesh():
tests/bench/test_bench_voronoi_partition_methods.py-219-    """Real-mesh sanity on the bound itself: a proper edge colouring can
tests/bench/test_bench_voronoi_partition_methods.py:220:    never use fewer rounds than the graph's max degree, and the multi-start
tests/bench/test_bench_voronoi_partition_methods.py-221-    search should not overshoot Vizing's ``Delta + 1`` either.  If this ever
tests/bench/test_bench_voronoi_partition_methods.py-222-    fires, ``max_degree`` is not the degree of the graph being coloured and
tests/bench/test_bench_voronoi_partition_methods.py-223-    every gap-based conclusion built on it is void."""
tests/bench/test_bench_voronoi_partition_methods.py-224-    for method in ("geometric", "sfc"):
tests/bench/test_bench_voronoi_partition_methods.py-225-        for n_ranks in (2, 4, 8):
tests/bench/test_bench_voronoi_partition_methods.py:226:            sc = mod.schedule_cost_row(_mesh(level=3), method, n_ranks)
tests/bench/test_bench_voronoi_partition_methods.py:227:            assert sc["max_degree"] <= sc["n_rounds"] <= sc["max_degree"] + 1, (
tests/bench/test_bench_voronoi_partition_methods.py:228:                f"{method} np={n_ranks}: rounds={sc['n_rounds']} "
tests/bench/test_bench_voronoi_partition_methods.py-229-                f"max_degree={sc['max_degree']}")
tests/bench/test_bench_voronoi_partition_methods.py-230-
tests/bench/test_bench_voronoi_partition_methods.py-231-
tests/bench/test_bench_voronoi_partition_methods.py-232-def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
--
tests/bench/test_bench_voronoi_partition_methods.py-234-    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
tests/bench/test_bench_voronoi_partition_methods.py-235-    a different graph and quietly report the wrong lane's cost."""
tests/bench/test_bench_voronoi_partition_methods.py-236-    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH
tests/bench/test_bench_voronoi_partition_methods.py-237-
tests/bench/test_bench_voronoi_partition_methods.py:238:    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
tests/bench/test_bench_voronoi_partition_methods.py-239-    assert sc["schedule_halo_depth"] == SPMD_HALO_DEPTH
tests/bench/test_bench_voronoi_partition_methods.py-240-
tests/bench/test_bench_voronoi_partition_methods.py-241-
tests/bench/test_bench_voronoi_partition_methods.py-242-def test_schedule_cost_matches_the_production_scorer_exactly():
tests/bench/test_bench_voronoi_partition_methods.py-243-    """Lock: the wrapper reports what the production scorer returns — it is
tests/bench/test_bench_voronoi_partition_methods.py-244-    a passthrough, not a re-derivation (the whole point: a 1-ring lookalike
tests/bench/test_bench_voronoi_partition_methods.py:245:    reports 8 rounds where the real depth-3 graph reports 12-14)."""
tests/bench/test_bench_voronoi_partition_methods.py:246:    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
tests/bench/test_bench_voronoi_partition_methods.py-247-
tests/bench/test_bench_voronoi_partition_methods.py-248-    mesh = _mesh()
tests/bench/test_bench_voronoi_partition_methods.py:249:    ref = spmd_schedule_cost(mesh, 2, method="sfc")
tests/bench/test_bench_voronoi_partition_methods.py:250:    sc = mod.schedule_cost_row(mesh, "sfc", 2)
tests/bench/test_bench_voronoi_partition_methods.py:251:    for key in ("n_rounds", "max_degree", "n_rounds_greedy",
tests/bench/test_bench_voronoi_partition_methods.py-252-                "coloring_method", "resolved_method", "cells_per_device",
tests/bench/test_bench_voronoi_partition_methods.py-253-                "production_strategy"):
tests/bench/test_bench_voronoi_partition_methods.py-254-        assert sc[key] == ref[key], key
tests/bench/test_bench_voronoi_partition_methods.py-255-
--
tests/bench/test_bench_voronoi_partition_methods.py-273-        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
tests/bench/test_bench_voronoi_partition_methods.py-274-    assert mod.main() == 0
tests/bench/test_bench_voronoi_partition_methods.py-275-    payload2 = json.loads(out2.read_text())
tests/bench/test_bench_voronoi_partition_methods.py-276-    row = payload2["rows"][0]
tests/bench/test_bench_voronoi_partition_methods.py:277:    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
tests/bench/test_bench_voronoi_partition_methods.py-278-    assert payload2["metadata"]["extra"]["schedule_cost"] is True
tests/bench/test_bench_voronoi_partition_methods.py-279-    # Scorer provenance per row: a copied row must show it scored a mesh
tests/bench/test_bench_voronoi_partition_methods.py-280-    # partitioned for THIS device count, not one reordered for another.
tests/bench/test_bench_voronoi_partition_methods.py-281-    assert row["schedule"]["reorder_target"] == row["n_ranks"]
tests/bench/test_bench_voronoi_partition_methods.py-282-    assert row["schedule"]["already_reordered"] is False
tests/bench/test_bench_voronoi_partition_methods.py-283-
tests/bench/test_bench_voronoi_partition_methods.py-284-
tests/bench/test_bench_voronoi_partition_methods.py:285:def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
tests/bench/test_bench_voronoi_partition_methods.py:286:    """``--lloyd`` must actually select the mesh, not just be recorded.
tests/bench/test_bench_voronoi_partition_methods.py-287-
tests/bench/test_bench_voronoi_partition_methods.py-288-    Non-vacuity: asserting only the recorded default (50) passes even if the
tests/bench/test_bench_voronoi_partition_methods.py-289-    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
tests/bench/test_bench_voronoi_partition_methods.py:290:    that.  This spies on the builder, so dropping ``lloyd_iterations=
tests/bench/test_bench_voronoi_partition_methods.py:291:    args.lloyd`` fails here, and it checks a NON-default value so the
tests/bench/test_bench_voronoi_partition_methods.py-292-    assertion cannot be satisfied by the default.
tests/bench/test_bench_voronoi_partition_methods.py-293-    """
tests/bench/test_bench_voronoi_partition_methods.py-294-    import legoesm.grids.voronoi as vor
tests/bench/test_bench_voronoi_partition_methods.py-295-
--
tests/bench/test_bench_voronoi_partition_methods.py-297-    real = vor.create_voronoi_mesh
tests/bench/test_bench_voronoi_partition_methods.py-298-
tests/bench/test_bench_voronoi_partition_methods.py-299-    def spy(*a, **k):
tests/bench/test_bench_voronoi_partition_methods.py-300-        seen.update(k)
tests/bench/test_bench_voronoi_partition_methods.py:301:        # lloyd=0 is cheap and is what the scaling meshes actually use.
tests/bench/test_bench_voronoi_partition_methods.py-302-        return real(*a, **k)
tests/bench/test_bench_voronoi_partition_methods.py-303-
tests/bench/test_bench_voronoi_partition_methods.py-304-    monkeypatch.setattr(vor, "create_voronoi_mesh", spy)
tests/bench/test_bench_voronoi_partition_methods.py:305:    out = tmp_path / "lloyd0.json"
tests/bench/test_bench_voronoi_partition_methods.py-306-    monkeypatch.setattr(sys, "argv", [
tests/bench/test_bench_voronoi_partition_methods.py-307-        "bench", "--subdivision", "2", "--rank-counts", "2",
tests/bench/test_bench_voronoi_partition_methods.py:308:        "--methods", "geometric", "--lloyd", "0", "--out", str(out)])
tests/bench/test_bench_voronoi_partition_methods.py-309-    assert mod.main() == 0
tests/bench/test_bench_voronoi_partition_methods.py:310:    assert seen.get("lloyd_iterations") == 0, seen
tests/bench/test_bench_voronoi_partition_methods.py-311-    # And it is recorded, so a synthetic mesh cannot be read back as a
tests/bench/test_bench_voronoi_partition_methods.py-312-    # production SCVT receipt.
tests/bench/test_bench_voronoi_partition_methods.py-313-    payload = json.loads(out.read_text())
tests/bench/test_bench_voronoi_partition_methods.py:314:    assert payload["metadata"]["extra"]["lloyd_iterations"] == 0
tests/bench/test_bench_voronoi_partition_methods.py-315-
tests/bench/test_bench_voronoi_partition_methods.py-316-
tests/bench/test_bench_voronoi_partition_methods.py:317:def test_expect_rounds_gate_fails_loudly_and_never_vacuously(
tests/bench/test_bench_voronoi_partition_methods.py-318-        tmp_path, monkeypatch):
tests/bench/test_bench_voronoi_partition_methods.py-319-    """The instrument check must FAIL on a wrong expectation and on a pair
tests/bench/test_bench_voronoi_partition_methods.py-320-    that was never scored — a gate that can only pass is not a gate."""
tests/bench/test_bench_voronoi_partition_methods.py-321-    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
--
tests/bench/test_bench_voronoi_partition_methods.py-324-    # Truth first: read what this configuration really scores.
tests/bench/test_bench_voronoi_partition_methods.py-325-    out = tmp_path / "truth.json"
tests/bench/test_bench_voronoi_partition_methods.py-326-    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
tests/bench/test_bench_voronoi_partition_methods.py-327-    assert mod.main() == 0
tests/bench/test_bench_voronoi_partition_methods.py:328:    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
tests/bench/test_bench_voronoi_partition_methods.py-329-
tests/bench/test_bench_voronoi_partition_methods.py-330-    # Matching expectation -> pass, and the check is recorded.
tests/bench/test_bench_voronoi_partition_methods.py-331-    ok = tmp_path / "ok.json"
tests/bench/test_bench_voronoi_partition_methods.py-332-    monkeypatch.setattr(sys, "argv", base + [
tests/bench/test_bench_voronoi_partition_methods.py:333:        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
tests/bench/test_bench_voronoi_partition_methods.py-334-    assert mod.main() == 0
tests/bench/test_bench_voronoi_partition_methods.py:335:    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
tests/bench/test_bench_voronoi_partition_methods.py-336-
tests/bench/test_bench_voronoi_partition_methods.py-337-    # Wrong expectation -> non-zero exit.
tests/bench/test_bench_voronoi_partition_methods.py-338-    bad = tmp_path / "bad.json"
tests/bench/test_bench_voronoi_partition_methods.py-339-    monkeypatch.setattr(sys, "argv", base + [
tests/bench/test_bench_voronoi_partition_methods.py:340:        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
tests/bench/test_bench_voronoi_partition_methods.py-341-    assert mod.main() == 1
tests/bench/test_bench_voronoi_partition_methods.py:342:    assert json.loads(bad.read_text())["expected_rounds_check"]["failures"]
tests/bench/test_bench_voronoi_partition_methods.py-343-
tests/bench/test_bench_voronoi_partition_methods.py-344-    # A pair that was never scored is a FAILURE, not a silent skip —
tests/bench/test_bench_voronoi_partition_methods.py-345-    # otherwise a sweep missing a method still reports a clean gate.
tests/bench/test_bench_voronoi_partition_methods.py-346-    missing = tmp_path / "missing.json"
tests/bench/test_bench_voronoi_partition_methods.py-347-    monkeypatch.setattr(sys, "argv", base + [
tests/bench/test_bench_voronoi_partition_methods.py:348:        "--expect-rounds", "sfc:2=3", "--out", str(missing)])
tests/bench/test_bench_voronoi_partition_methods.py-349-    assert mod.main() == 1
tests/bench/test_bench_voronoi_partition_methods.py:350:    fails = json.loads(missing.read_text())["expected_rounds_check"]["failures"]
tests/bench/test_bench_voronoi_partition_methods.py-351-    assert any("NOT SCORED" in f for f in fails), fails
tests/bench/test_bench_voronoi_partition_methods.py-352-
tests/bench/test_bench_voronoi_partition_methods.py-353-
tests/bench/test_bench_voronoi_partition_methods.py:354:def test_expect_rounds_refuses_to_pass_vacuously_without_scoring(monkeypatch):
tests/bench/test_bench_voronoi_partition_methods.py-355-    """Without --schedule-cost nothing is scored, so the gate would pass on
tests/bench/test_bench_voronoi_partition_methods.py-356-    an empty comparison. It must refuse instead."""
tests/bench/test_bench_voronoi_partition_methods.py-357-    monkeypatch.setattr(sys, "argv", [
tests/bench/test_bench_voronoi_partition_methods.py-358-        "bench", "--subdivision", "2", "--rank-counts", "2",
tests/bench/test_bench_voronoi_partition_methods.py:359:        "--methods", "geometric", "--expect-rounds", "geometric:2=1",
tests/bench/test_bench_voronoi_partition_methods.py-360-        "--out", "/dev/null"])
tests/bench/test_bench_voronoi_partition_methods.py-361-    with pytest.raises(SystemExit, match="needs --schedule-cost"):
tests/bench/test_bench_voronoi_partition_methods.py-362-        mod.main()
tests/bench/test_bench_voronoi_partition_methods.py-363-
tests/bench/test_bench_voronoi_partition_methods.py-364-
tests/bench/test_bench_voronoi_partition_methods.py-365-@pytest.mark.parametrize("spec", ["geometric:2", "geometric=2", "voodoo:2=3"])
tests/bench/test_bench_voronoi_partition_methods.py:366:def test_expect_rounds_rejects_malformed_specs(spec):
tests/bench/test_bench_voronoi_partition_methods.py-367-    """A typo'd expectation must raise, never be dropped — a silently
tests/bench/test_bench_voronoi_partition_methods.py-368-    skipped expectation turns the gate into a no-op."""
tests/bench/test_bench_voronoi_partition_methods.py:369:    with pytest.raises(ValueError, match="expect-rounds"):
tests/bench/test_bench_voronoi_partition_methods.py:370:        mod.parse_expect_rounds(spec)
--
scripts/bench/bench_voronoi_partition_methods.py-8-   for each method x rank-count on the real icosahedral mesh.  These are
scripts/bench/bench_voronoi_partition_methods.py-9-   the numbers ``resolve_partition_method``'s ``auto`` policy must be
scripts/bench/bench_voronoi_partition_methods.py-10-   justified by.
scripts/bench/bench_voronoi_partition_methods.py-11-1b. **SPMD halo-schedule depth** (``--schedule-cost``, opt-in because it is
scripts/bench/bench_voronoi_partition_methods.py:12:   the expensive layer): ``n_rounds`` — the number of SEQUENTIAL ppermute
scripts/bench/bench_voronoi_partition_methods.py:13:   rounds one halo fill costs — and ``max_degree``, the communication
scripts/bench/bench_voronoi_partition_methods.py-14-   graph's lower bound on it.  This is the term that binds MPAS GPU strong
scripts/bench/bench_voronoi_partition_methods.py-15-   scaling above ~64 devices, and layer 1 CANNOT stand in for it: the
scripts/bench/bench_voronoi_partition_methods.py:16:   neighbor fan-out above is a 1-ring proxy that reported 8 rounds for
scripts/bench/bench_voronoi_partition_methods.py-17-   every method and rank count while the real depth-3-plus-closure
scripts/bench/bench_voronoi_partition_methods.py-18-   schedule reported 12-14.  Scored by the production
scripts/bench/bench_voronoi_partition_methods.py:19:   ``spmd_schedule_cost`` (which calls the production builders), never a
scripts/bench/bench_voronoi_partition_methods.py-20-   re-derived lookalike.
scripts/bench/bench_voronoi_partition_methods.py-21-
scripts/bench/bench_voronoi_partition_methods.py:22:   The DECISIVE column is ``coloring_gap = n_rounds - max_degree``, read
scripts/bench/bench_voronoi_partition_methods.py-23-   through VIZING'S THEOREM, which bounds what recolouring could ever buy.
scripts/bench/bench_voronoi_partition_methods.py-24-   The schedule is a proper EDGE colouring of the device communication
scripts/bench/bench_voronoi_partition_methods.py-25-   graph (one colour = one ppermute round; ``_build_ppermute_schedule``
scripts/bench/bench_voronoi_partition_methods.py-26-   asserts properness), and ``max_degree`` is that same graph's maximum
scripts/bench/bench_voronoi_partition_methods.py-27-   vertex degree.  So the chromatic index obeys ``Delta <= chi' <=
scripts/bench/bench_voronoi_partition_methods.py-28-   Delta + 1``: the gap is a bound on recolouring headroom, NOT a
scripts/bench/bench_voronoi_partition_methods.py-29-   yes/no flag.
scripts/bench/bench_voronoi_partition_methods.py-30-
scripts/bench/bench_voronoi_partition_methods.py:31:   * ``gap == 0`` -> ``n_rounds == Delta``, and no proper edge colouring
scripts/bench/bench_voronoi_partition_methods.py-32-     can beat ``Delta``.  The colouring is PROVABLY OPTIMAL; recolouring
scripts/bench/bench_voronoi_partition_methods.py-33-     headroom is exactly ZERO.
scripts/bench/bench_voronoi_partition_methods.py-34-   * ``gap == 1`` -> INCONCLUSIVE.  A Class 2 graph genuinely needs
scripts/bench/bench_voronoi_partition_methods.py-35-     ``Delta + 1``, and deciding Class 1 vs Class 2 is NP-complete, so
scripts/bench/bench_voronoi_partition_methods.py-36-     this neither establishes nor excludes a one-round win.
scripts/bench/bench_voronoi_partition_methods.py-37-   * ``gap >= 2`` -> recolouring is guaranteed to remove AT LEAST
scripts/bench/bench_voronoi_partition_methods.py:38:     ``gap - 1`` rounds (the optimum is at worst ``Delta + 1``) and at
scripts/bench/bench_voronoi_partition_methods.py-39-     most ``gap``.
scripts/bench/bench_voronoi_partition_methods.py-40-
scripts/bench/bench_voronoi_partition_methods.py-41-   SCOPE, and it is not a formality: ``Delta`` bounds only a proper
scripts/bench/bench_voronoi_partition_methods.py-42-   UNDIRECTED edge colouring of THIS graph.  ``_build_ppermute_schedule``
--
scripts/bench/bench_voronoi_partition_methods.py-50-
scripts/bench/bench_voronoi_partition_methods.py-51-   NOTE the halo depth differs by lane and is NOT ``--halo-depth``: that
scripts/bench/bench_voronoi_partition_methods.py-52-   flag is the MPI lane's (default 2), while the schedule is scored at the
scripts/bench/bench_voronoi_partition_methods.py-53-   SPMD production depth ``SPMD_HALO_DEPTH`` (3).  Both are recorded.
scripts/bench/bench_voronoi_partition_methods.py:54:   ``n_rounds`` is per HALO FILL, not per step — multiply by the tendency
scripts/bench/bench_voronoi_partition_methods.py-55-   evaluations of the integrator actually run.  When
scripts/bench/bench_voronoi_partition_methods.py-56-   ``production_strategy`` is ``"allgather"`` (auto-selected below the
scripts/bench/bench_voronoi_partition_methods.py-57-   cells/device threshold) there is no ppermute schedule in production and
scripts/bench/bench_voronoi_partition_methods.py-58-   the round count is COUNTERFACTUAL; the row says so.
--
scripts/bench/bench_voronoi_partition_methods.py-180-    raise ValueError(f"unknown partition method {method!r}; "
scripts/bench/bench_voronoi_partition_methods.py-181-                     f"expected one of {METHODS}")
scripts/bench/bench_voronoi_partition_methods.py-182-
scripts/bench/bench_voronoi_partition_methods.py-183-
scripts/bench/bench_voronoi_partition_methods.py:184:def parse_expect_rounds(spec: str) -> dict:
scripts/bench/bench_voronoi_partition_methods.py-185-    """Parse ``'sfc:64=12,metis:128=19'`` into ``{("sfc", 64): 12}``.
scripts/bench/bench_voronoi_partition_methods.py-186-
scripts/bench/bench_voronoi_partition_methods.py-187-    Raises on anything malformed rather than skipping it — a typo'd
scripts/bench/bench_voronoi_partition_methods.py-188-    expectation that is silently dropped turns the gate into a no-op, which
--
scripts/bench/bench_voronoi_partition_methods.py-192-    for item in (s.strip() for s in spec.split(",")):
scripts/bench/bench_voronoi_partition_methods.py-193-        if not item:
scripts/bench/bench_voronoi_partition_methods.py-194-            continue
scripts/bench/bench_voronoi_partition_methods.py-195-        try:
scripts/bench/bench_voronoi_partition_methods.py:196:            lhs, rounds = item.split("=")
scripts/bench/bench_voronoi_partition_methods.py-197-            method, n_ranks = lhs.split(":")
scripts/bench/bench_voronoi_partition_methods.py-198-            key = (method.strip(), int(n_ranks))
scripts/bench/bench_voronoi_partition_methods.py:199:            out[key] = int(rounds)
scripts/bench/bench_voronoi_partition_methods.py-200-        except ValueError as exc:
scripts/bench/bench_voronoi_partition_methods.py-201-            raise ValueError(
scripts/bench/bench_voronoi_partition_methods.py:202:                f"--expect-rounds: cannot parse {item!r}; expected "
scripts/bench/bench_voronoi_partition_methods.py:203:                f"'<method>:<n_ranks>=<n_rounds>'") from exc
scripts/bench/bench_voronoi_partition_methods.py-204-        if key[0] not in METHODS:
scripts/bench/bench_voronoi_partition_methods.py-205-            raise ValueError(
scripts/bench/bench_voronoi_partition_methods.py:206:                f"--expect-rounds: unknown method {key[0]!r} in {item!r}; "
scripts/bench/bench_voronoi_partition_methods.py-207-                f"expected one of {METHODS}")
scripts/bench/bench_voronoi_partition_methods.py-208-    return out
scripts/bench/bench_voronoi_partition_methods.py-209-
scripts/bench/bench_voronoi_partition_methods.py-210-
scripts/bench/bench_voronoi_partition_methods.py:211:def check_expected_rounds(rows: list, expect: dict) -> list:
scripts/bench/bench_voronoi_partition_methods.py:212:    """Compare scored rounds against *expect*; return failure strings.
scripts/bench/bench_voronoi_partition_methods.py-213-
scripts/bench/bench_voronoi_partition_methods.py-214-    A listed pair that was never scored is a FAILURE, not a skip: otherwise
scripts/bench/bench_voronoi_partition_methods.py-215-    a sweep that silently dropped a method (unavailable ``pymetis``) or a
scripts/bench/bench_voronoi_partition_methods.py-216-    rank count would still report a clean gate.
scripts/bench/bench_voronoi_partition_methods.py-217-    """
scripts/bench/bench_voronoi_partition_methods.py-218-    scored = {
scripts/bench/bench_voronoi_partition_methods.py:219:        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
scripts/bench/bench_voronoi_partition_methods.py-220-        for r in rows
scripts/bench/bench_voronoi_partition_methods.py-221-        if r.get("available") and "schedule" in r and "n_ranks" in r
scripts/bench/bench_voronoi_partition_methods.py-222-    }
scripts/bench/bench_voronoi_partition_methods.py-223-    failures = []
scripts/bench/bench_voronoi_partition_methods.py-224-    for (method, n_ranks), want in sorted(expect.items()):
scripts/bench/bench_voronoi_partition_methods.py-225-        got = scored.get((method, n_ranks))
scripts/bench/bench_voronoi_partition_methods.py-226-        if got is None:
scripts/bench/bench_voronoi_partition_methods.py-227-            failures.append(
scripts/bench/bench_voronoi_partition_methods.py:228:                f"{method}:{n_ranks} expected rounds={want} but the pair was "
scripts/bench/bench_voronoi_partition_methods.py-229-                f"NOT SCORED (method unavailable, or not in this sweep)")
scripts/bench/bench_voronoi_partition_methods.py-230-        elif got != want:
scripts/bench/bench_voronoi_partition_methods.py-231-            failures.append(
scripts/bench/bench_voronoi_partition_methods.py:232:                f"{method}:{n_ranks} expected rounds={want}, got {got}")
scripts/bench/bench_voronoi_partition_methods.py-233-    return failures
scripts/bench/bench_voronoi_partition_methods.py-234-
scripts/bench/bench_voronoi_partition_methods.py-235-
scripts/bench/bench_voronoi_partition_methods.py:236:def schedule_cost_row(mesh, method: str, n_ranks: int) -> dict:
scripts/bench/bench_voronoi_partition_methods.py-237-    """SPMD halo-schedule depth for one (method, n_ranks) candidate.
scripts/bench/bench_voronoi_partition_methods.py-238-
scripts/bench/bench_voronoi_partition_methods.py-239-    Thin wrapper over the production
scripts/bench/bench_voronoi_partition_methods.py:240:    :func:`legoesm.parallel.sharded_dynamics.spmd_schedule_cost` — it reorders
scripts/bench/bench_voronoi_partition_methods.py-241-    the RAW mesh for ``n_ranks`` with ``method`` and colours the real
scripts/bench/bench_voronoi_partition_methods.py-242-    depth-``SPMD_HALO_DEPTH`` communication graph, so the number is the one
scripts/bench/bench_voronoi_partition_methods.py-243-    production pays, not a 1-ring lookalike.  ``halo_depth`` is deliberately
scripts/bench/bench_voronoi_partition_methods.py-244-    LEFT AT THE SCORER'S DEFAULT: this script's ``--halo-depth`` is the MPI
scripts/bench/bench_voronoi_partition_methods.py-245-    lane's (2), and scoring the SPMD schedule at 2 would colour a different
scripts/bench/bench_voronoi_partition_methods.py-246-    graph.
scripts/bench/bench_voronoi_partition_methods.py-247-
scripts/bench/bench_voronoi_partition_methods.py:248:    Adds ``coloring_gap = n_rounds - max_degree`` and the Vizing reading of
scripts/bench/bench_voronoi_partition_methods.py-249-    it (see the module docstring).  The headroom is reported as an INTERVAL,
scripts/bench/bench_voronoi_partition_methods.py-250-    because Vizing pins the optimum only to ``{Delta, Delta + 1}``:
scripts/bench/bench_voronoi_partition_methods.py-251-
scripts/bench/bench_voronoi_partition_methods.py:252:    * ``coloring_headroom_rounds_min = max(0, gap - 1)`` — rounds a perfect
scripts/bench/bench_voronoi_partition_methods.py-253-      recolouring is GUARANTEED to remove (it beats the ``Delta + 1`` case).
scripts/bench/bench_voronoi_partition_methods.py:254:    * ``coloring_headroom_rounds_max = gap`` — the best case, realized only
scripts/bench/bench_voronoi_partition_methods.py-255-      if the graph is Class 1.
scripts/bench/bench_voronoi_partition_methods.py-256-
scripts/bench/bench_voronoi_partition_methods.py-257-    It is deliberately NOT a "recolour vs ownership" verdict: at
scripts/bench/bench_voronoi_partition_methods.py-258-    ``gap == 1`` the min is 0 and the max is 1, i.e. genuinely inconclusive.
--
scripts/bench/bench_voronoi_partition_methods.py-268-    keeps the completed rungs in the log.
scripts/bench/bench_voronoi_partition_methods.py-269-    """
scripts/bench/bench_voronoi_partition_methods.py-270-    import time
scripts/bench/bench_voronoi_partition_methods.py-271-
scripts/bench/bench_voronoi_partition_methods.py:272:    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
scripts/bench/bench_voronoi_partition_methods.py-273-
scripts/bench/bench_voronoi_partition_methods.py-274-    t0 = time.perf_counter()
scripts/bench/bench_voronoi_partition_methods.py:275:    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
scripts/bench/bench_voronoi_partition_methods.py:276:    gap = int(cost["n_rounds"]) - int(cost["max_degree"])
scripts/bench/bench_voronoi_partition_methods.py-277-    return {
scripts/bench/bench_voronoi_partition_methods.py:278:        "n_rounds": int(cost["n_rounds"]),
scripts/bench/bench_voronoi_partition_methods.py:279:        "n_rounds_greedy": int(cost["n_rounds_greedy"]),
scripts/bench/bench_voronoi_partition_methods.py-280-        "max_degree": int(cost["max_degree"]),
scripts/bench/bench_voronoi_partition_methods.py-281-        # The decisive column, read through Vizing (see module docstring).
scripts/bench/bench_voronoi_partition_methods.py-282-        "coloring_gap": gap,
scripts/bench/bench_voronoi_partition_methods.py-283-        # Rounds a perfect recolouring could remove, as an INTERVAL: Vizing
scripts/bench/bench_voronoi_partition_methods.py-284-        # pins the optimum to {Delta, Delta+1}, so gap-1 is guaranteed and
scripts/bench/bench_voronoi_partition_methods.py-285-        # gap is the best case.  A gap of 1 spans [0, 1] = inconclusive.
scripts/bench/bench_voronoi_partition_methods.py:286:        "coloring_headroom_rounds_min": max(0, gap - 1),
scripts/bench/bench_voronoi_partition_methods.py:287:        "coloring_headroom_rounds_max": gap,
scripts/bench/bench_voronoi_partition_methods.py-288-        "coloring_optimal_proven": gap == 0,
scripts/bench/bench_voronoi_partition_methods.py-289-        # Scorer provenance, carried per row: a copied/flattened row must be
scripts/bench/bench_voronoi_partition_methods.py-290-        # able to show it scored a raw mesh partitioned for THIS device
scripts/bench/bench_voronoi_partition_methods.py-291-        # count, not one reordered for a different target.
--
scripts/bench/bench_voronoi_partition_methods.py-313-                   help="Halo layers (runtime default 2, del4 support).")
scripts/bench/bench_voronoi_partition_methods.py-314-    p.add_argument("--methods", type=str, default=",".join(METHODS))
scripts/bench/bench_voronoi_partition_methods.py-315-    p.add_argument("--schedule-cost", action="store_true",
scripts/bench/bench_voronoi_partition_methods.py-316-                   help="Also score the SPMD ppermute halo-schedule depth "
scripts/bench/bench_voronoi_partition_methods.py:317:                        "(n_rounds vs max_degree) per method x rank count. "
scripts/bench/bench_voronoi_partition_methods.py-318-                        "Uses the production SPMD halo depth, NOT "
scripts/bench/bench_voronoi_partition_methods.py-319-                        "--halo-depth. Expensive: minutes per candidate at "
scripts/bench/bench_voronoi_partition_methods.py-320-                        "subdiv>=8 — run it under batch.")
scripts/bench/bench_voronoi_partition_methods.py:321:    p.add_argument("--lloyd", type=int, default=50,
scripts/bench/bench_voronoi_partition_methods.py-322-                   help="Lloyd relaxation iterations for the mesh. 50 = the "
scripts/bench/bench_voronoi_partition_methods.py-323-                        "production SCVT key; 0 = the LABELLED synthetic "
scripts/bench/bench_voronoi_partition_methods.py:324:                        "scaling mesh. Recorded so a lloyd=0 mesh can never "
scripts/bench/bench_voronoi_partition_methods.py-325-                        "masquerade as a production receipt, and it must "
scripts/bench/bench_voronoi_partition_methods.py-326-                        "match the prewarmed cache key at subdiv>=9.")
scripts/bench/bench_voronoi_partition_methods.py:327:    p.add_argument("--expect-rounds", type=str, default="",
scripts/bench/bench_voronoi_partition_methods.py-328-                   help="Instrument check, MECHANICAL. Comma-separated "
scripts/bench/bench_voronoi_partition_methods.py:329:                        "'<method>:<n_ranks>=<n_rounds>' expectations (e.g. "
scripts/bench/bench_voronoi_partition_methods.py-330-                        "'sfc:64=12,sfc:128=14'). Every listed pair must be "
scripts/bench/bench_voronoi_partition_methods.py-331-                        "scored and match, or main() returns 1 — so a caller "
scripts/bench/bench_voronoi_partition_methods.py:332:                        "that reproduces a known census can GATE on it "
scripts/bench/bench_voronoi_partition_methods.py-333-                        "instead of asserting agreement in a comment. "
scripts/bench/bench_voronoi_partition_methods.py-334-                        "Requires --schedule-cost.")
scripts/bench/bench_voronoi_partition_methods.py-335-    p.add_argument("--out", type=str,
scripts/bench/bench_voronoi_partition_methods.py-336-                   default="results/a1/voronoi_partition_quality.json")
--
scripts/bench/bench_voronoi_partition_methods.py-342-        if m not in METHODS:
scripts/bench/bench_voronoi_partition_methods.py-343-            raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
scripts/bench/bench_voronoi_partition_methods.py-344-    if not rank_counts or any(n < 2 for n in rank_counts):
scripts/bench/bench_voronoi_partition_methods.py-345-        raise SystemExit("--rank-counts needs integers >= 2")
scripts/bench/bench_voronoi_partition_methods.py:346:    expect = parse_expect_rounds(args.expect_rounds)
scripts/bench/bench_voronoi_partition_methods.py-347-    if expect and not args.schedule_cost:
scripts/bench/bench_voronoi_partition_methods.py-348-        raise SystemExit(
scripts/bench/bench_voronoi_partition_methods.py:349:            "--expect-rounds compares scored round counts, so it needs "
scripts/bench/bench_voronoi_partition_methods.py-350-            "--schedule-cost; without it nothing is scored and the gate "
scripts/bench/bench_voronoi_partition_methods.py-351-            "would pass vacuously.")
scripts/bench/bench_voronoi_partition_methods.py-352-
scripts/bench/bench_voronoi_partition_methods.py-353-
scripts/bench/bench_voronoi_partition_methods.py-354-    from legoesm.grids.voronoi import create_voronoi_mesh
scripts/bench/bench_voronoi_partition_methods.py-355-    from legoesm.parallel.voronoi_partition import resolve_partition_method
scripts/bench/bench_voronoi_partition_methods.py-356-
scripts/bench/bench_voronoi_partition_methods.py-357-    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
scripts/bench/bench_voronoi_partition_methods.py:358:                               lloyd_iterations=args.lloyd)
scripts/bench/bench_voronoi_partition_methods.py-359-    if max(rank_counts) > int(mesh.nCells):
scripts/bench/bench_voronoi_partition_methods.py-360-        raise SystemExit(
scripts/bench/bench_voronoi_partition_methods.py-361-            f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
scripts/bench/bench_voronoi_partition_methods.py-362-            f"{int(mesh.nCells)} cells (empty ranks are meaningless).")
--
scripts/bench/bench_voronoi_partition_methods.py-386-                  f"mean={q['halo_cells_mean']:8.1f} | "
scripts/bench/bench_voronoi_partition_methods.py-387-                  f"halo/owned max={q['halo_owned_ratio_max']:.3f} | "
scripts/bench/bench_voronoi_partition_methods.py-388-                  f"nbrs max={q['neighbor_ranks_max']}")
scripts/bench/bench_voronoi_partition_methods.py-389-            if args.schedule_cost:
scripts/bench/bench_voronoi_partition_methods.py:390:                sc = schedule_cost_row(mesh, method, n_ranks)
scripts/bench/bench_voronoi_partition_methods.py-391-                row["schedule"] = sc
scripts/bench/bench_voronoi_partition_methods.py-392-                note = ("  [COUNTERFACTUAL: production auto-selects "
scripts/bench/bench_voronoi_partition_methods.py-393-                        "allgather here, no ppermute schedule]"
scripts/bench/bench_voronoi_partition_methods.py-394-                        if sc["production_strategy"] == "allgather" else "")
scripts/bench/bench_voronoi_partition_methods.py-395-                verdict = ("colouring PROVABLY OPTIMAL (Vizing floor)"
scripts/bench/bench_voronoi_partition_methods.py-396-                           if sc["coloring_optimal_proven"] else
scripts/bench/bench_voronoi_partition_methods.py-397-                           f"recolour headroom "
scripts/bench/bench_voronoi_partition_methods.py:398:                           f"{sc['coloring_headroom_rounds_min']}-"
scripts/bench/bench_voronoi_partition_methods.py:399:                           f"{sc['coloring_headroom_rounds_max']} round(s)")
scripts/bench/bench_voronoi_partition_methods.py-400-                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
scripts/bench/bench_voronoi_partition_methods.py:401:                      f"rounds={sc['n_rounds']:3d} "
scripts/bench/bench_voronoi_partition_methods.py-402-                      f"max_degree={sc['max_degree']:3d} "
scripts/bench/bench_voronoi_partition_methods.py-403-                      f"gap={sc['coloring_gap']:+d} "
scripts/bench/bench_voronoi_partition_methods.py-404-                      f"-> {verdict} "
scripts/bench/bench_voronoi_partition_methods.py-405-                      f"({sc['score_seconds']:.1f}s){note}", flush=True)
--
scripts/bench/bench_voronoi_partition_methods.py-423-            transport="none",
scripts/bench/bench_voronoi_partition_methods.py-424-            extra={"rank_counts": rank_counts, "methods": methods,
scripts/bench/bench_voronoi_partition_methods.py-425-                   "halo_depth": args.halo_depth,
scripts/bench/bench_voronoi_partition_methods.py-426-                   "schedule_cost": bool(args.schedule_cost),
scripts/bench/bench_voronoi_partition_methods.py:427:                   "lloyd_iterations": args.lloyd},
scripts/bench/bench_voronoi_partition_methods.py-428-        )),
scripts/bench/bench_voronoi_partition_methods.py-429-    }
scripts/bench/bench_voronoi_partition_methods.py-430-    outdir = os.path.dirname(args.out)
scripts/bench/bench_voronoi_partition_methods.py-431-    if outdir:
--
scripts/bench/bench_voronoi_partition_methods.py-434-        json.dump(payload, f, indent=2)
scripts/bench/bench_voronoi_partition_methods.py-435-    print(f"JSON: {args.out}")
scripts/bench/bench_voronoi_partition_methods.py-436-
scripts/bench/bench_voronoi_partition_methods.py-437-    if expect:
scripts/bench/bench_voronoi_partition_methods.py:438:        failures = check_expected_rounds(rows, expect)
scripts/bench/bench_voronoi_partition_methods.py:439:        payload["expected_rounds_check"] = {
scripts/bench/bench_voronoi_partition_methods.py-440-            "expected": {f"{m}:{n}": v for (m, n), v in sorted(expect.items())},
scripts/bench/bench_voronoi_partition_methods.py-441-            "failures": failures,
scripts/bench/bench_voronoi_partition_methods.py-442-            "passed": not failures,
scripts/bench/bench_voronoi_partition_methods.py-443-        }
--
scripts/bench/bench_voronoi_partition_methods.py-446-        if failures:
scripts/bench/bench_voronoi_partition_methods.py-447-            print(f"INSTRUMENT CHECK FAILED ({len(failures)}):")
scripts/bench/bench_voronoi_partition_methods.py-448-            for line in failures:
scripts/bench/bench_voronoi_partition_methods.py-449-                print(f"  {line}")
scripts/bench/bench_voronoi_partition_methods.py:450:            print("The scorer did NOT reproduce the known census — treat every "
scripts/bench/bench_voronoi_partition_methods.py-451-                  "unknown row in this run as UNTRUSTED.")
scripts/bench/bench_voronoi_partition_methods.py-452-            return 1
scripts/bench/bench_voronoi_partition_methods.py-453-        print(f"INSTRUMENT CHECK PASSED ({len(expect)} expectations)")
scripts/bench/bench_voronoi_partition_methods.py-454-    return 0

exec
/usr/bin/bash -lc 'nl -ba scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && nl -ba tests/bench/test_bench_voronoi_partition_methods.py && rg -n -C 10 "def spmd_schedule_cost|SPMD_HALO_DEPTH|reorder_target|already_reordered|comm_pairs|_build_ppermute_schedule|n_rounds|census" legoesm tests scripts | head -n 1200' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 483ms:
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
    11	# Does RECOLOURING still have room on the MPAS halo schedule, or is only a
    12	# new OWNERSHIP objective left?  CPU-only, ZERO GPU hours.
    13	#
    14	# WHY THIS RUN EXISTS
    15	# MPAS GPU is the worst-scaling lane we have: measured/modelled-bound 3.16x
    16	# (s8@16) to 4.47x (s9@64), and one halo fill costs 12-14 SEQUENTIAL ppermute
    17	# rounds.  Both independent reviews (codex + GLM, 2026-08-07) ranked cutting
    18	# that round count as the top structural lever, at 300-800 LOC and 7-14 days
    19	# for a partitioner with a new objective.  That estimate is only worth
    20	# spending if recolouring is genuinely exhausted, and the colourer's own
    21	# lower bound decides it.
    22	#
    23	# The schedule is a proper EDGE colouring of the device communication graph
    24	# (one colour = one ppermute round) and max_degree is that graph's maximum
    25	# vertex degree, so VIZING bounds the chromatic index: Delta <= chi' <=
    26	# Delta + 1.  Read coloring_gap = n_rounds - max_degree through that:
    27	#     gap == 0 -> PROVABLY OPTIMAL; recolouring headroom is exactly zero.
    28	#     gap == 1 -> indistinguishable from optimal (a Class 2 graph really
    29	#                 needs Delta+1, and deciding Class 1/2 is NP-complete);
    30	#                 nothing provable to win.
    31	#     gap >= 2 -> at least gap-1 rounds of genuine recolouring headroom.
    32	# So recolouring is capped at ~1 round out of 12-14 (<= ~7%) wherever the
    33	# multi-start search lands on Delta or Delta+1 — which it does on every
    34	# configuration probed so far.  This run measures whether the production
    35	# working points are in that regime.
    36	#
    37	# (a) NUMBER PRODUCED: n_rounds, max_degree and their gap per
    38	#     (method x n_dev) at the production working points.
    39	# (b) CONFIRMS a cheap fix: gap >= 2 at any production (ppermute) row
    40	#     (recolouring then removes at least gap-1 rounds, guaranteed).
    41	#     gap == 0: recolouring is provably worthless AT THIS OWNERSHIP.
    42	#     gap == 1: INCONCLUSIVE — Class 1 vs Class 2 is NP-complete, so this
    43	#     neither confirms nor refutes a one-round win.
    44	#     Note what a gap of 0 does NOT prove: Delta bounds only a proper
    45	#     UNDIRECTED edge colouring of this graph, and the builder forces BOTH
    46	#     ppermute directions per pair even when one send map is empty, so a
    47	#     redesigned DIRECTED schedule is not bounded by Delta.  "Ownership is
    48	#     the only path" would overclaim; the honest statement is "recolouring
    49	#     this undirected graph is exhausted".
    50	# (c) WHY NOT CHEAPER: this IS the cheap test — no GPU, no MPI, no model
    51	#     step.  It cannot be shrunk further onto small meshes: measured
    52	#     2026-08-07, L2/L4 x {geometric,sfc} x nd 2-16 all report gap == 0,
    53	#     but EVERY one of those rows auto-selects the ALLGATHER strategy
    54	#     (cells/device below the threshold), so production runs no ppermute
    55	#     schedule there and the number is counterfactual.  Only meshes big
    56	#     enough to keep cells/device above the threshold answer the question.
    57	#     s6 lloyd=0 np8/np16 (2561 cells/device after padding, above the 2000
    58	#     ppermute threshold, so genuinely ppermute) also gave gap == 0 on all
    59	#     three methods — rounds 7/7/6 at np8 and 13/10/10 at np16 for
    60	#     geometric/sfc/metis.  np8's 7 == n_dev-1 is complete-graph
    61	#     saturation and says nothing; np16 (13/10/10 < 15) does escape it.
    62	#     Still a SMALL working point; s8-s10 are the production ones.
    63	#
    64	# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
    65	# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
    66	#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
    67	#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
    68	# Arms 1-2 must REPRODUCE those before arm 3's unknown s10 number is quoted.
    69	# A scan that misses the known answer is a broken instrument, and its s10
    70	# row must not be believed.
    71	#
    72	# lloyd=0 throughout: it is what the reference census used AND the cache key
    73	# the prewarmed s8/s9/s10 meshes were written under.  It is the LABELLED
    74	# synthetic scaling mesh, recorded in every row's metadata so it can never be
    75	# read back as a production SCVT receipt.
    76	#
    77	# Arms run cheapest-first and each writes its own JSON, so a later arm that
    78	# runs out of time or memory cannot lose an earlier arm's result.  s10@128 is
    79	# last and is the one genuinely at risk: the scorer is known not to have
    80	# finished at subdiv-8@128 on a laptop, which is why this asks for 24 h.
    81	#
    82	# SUBMIT (from the repo root):
    83	#   sbatch scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
    84	# ===========================================================================
    85	set -uo pipefail
    86	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    87	export JAX_PLATFORMS=cpu
    88	export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
    89	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    90	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    91	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    92	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    93	
    94	OUT=results/a1/mpas_schedule_cost
    95	mkdir -p "$OUT"
    96	BENCH=scripts/bench/bench_voronoi_partition_methods.py
    97	
    98	# pymetis is present in .venv but was ABSENT from .venv-mpi for a while, and a
    99	# missing partitioner is reported as "unavailable" rather than substituted.
   100	# Say which python and whether metis is really there, so a two-method table
   101	# cannot be misread as a three-method one.
   102	echo "[scan] python=$PY"
   103	"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
   104	  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"
   105	
   106	run_arm () {  # $1=level  $2=rank-counts  $3=label  $4=expect-rounds (may be "")
   107	  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
   108	  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
   109	  "$PY" "$BENCH" \
   110	      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
   111	      --methods geometric,sfc,metis --schedule-cost \
   112	      ${4:+--expect-rounds "$4"} \
   113	      --out "$OUT/schedule_cost_s$1.json"
   114	  local rc=$?   # capture BEFORE anything else runs, or `date` overwrites it
   115	  echo "[scan] arm $3 exit=$rc"
   116	  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
   117	  return $rc
   118	}
   119	
   120	# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
   121	# makes the bench exit non-zero unless it reproduces the census below, and a
   122	# listed pair that never got scored (e.g. pymetis missing) counts as a
   123	# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
   124	# pass — an instrument that misses the known answer cannot be trusted on the
   125	# unknown one.
   126	S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
   127	S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"
   128	
   129	run_arm 8 64,128 "1/3 VALIDATION s8" "$S8_EXPECT"; RC8=$?
   130	run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?
   131	
   132	if [ "$RC8" -ne 0 ] || [ "$RC9" -ne 0 ]; then
   133	  echo "[scan] INSTRUMENT CHECK FAILED (s8 rc=$RC8, s9 rc=$RC9)."
   134	  echo "[scan] The scorer does not reproduce the reference census, so an s10"
   135	  echo "[scan] number from it would be untrustworthy. NOT running arm 3."
   136	  echo "SCAN_ABORTED_VALIDATION"
   137	  exit 1
   138	fi
   139	
   140	# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
   141	run_arm 10 128 "3/3 UNKNOWN s10" ""; RC10=$?
   142	echo "[scan] arm 3 rc=$RC10"
   143	
   144	echo "SCAN_DONE"
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
rg: legoesm: No such file or directory (os error 2)
tests/bench/test_scaling_metadata.py-356-    out = md.tidy_throughput_fields(
tests/bench/test_scaling_metadata.py-357-        dt_seconds=600.0, time_per_step_ms=None, total_cells=10)
tests/bench/test_scaling_metadata.py-358-    assert out["time_per_step_ms"] is None
tests/bench/test_scaling_metadata.py-359-    assert out["sypd"] is None
tests/bench/test_scaling_metadata.py-360-    assert out["mcells_per_s"] is None
tests/bench/test_scaling_metadata.py-361-    assert out["dt_seconds"] == 600.0
tests/bench/test_scaling_metadata.py-362-    assert out["total_cells"] == 10
tests/bench/test_scaling_metadata.py-363-
tests/bench/test_scaling_metadata.py-364-
tests/bench/test_scaling_metadata.py-365-def test_count_collective_permutes_matches_hyphen_and_underscore():
tests/bench/test_scaling_metadata.py:366:    """Canonical CP census (#1113): counts StableHLO underscore + optimized
tests/bench/test_scaling_metadata.py-367-    hyphen spellings, and drops the async ``-done`` companion so one logical
tests/bench/test_scaling_metadata.py-368-    exchange counts once."""
tests/bench/test_scaling_metadata.py-369-    hlo = "\n".join([
tests/bench/test_scaling_metadata.py-370-        "  %a = collective-permute(%x)",          # optimized HLO
tests/bench/test_scaling_metadata.py-371-        "  %b = collective_permute(%y)",          # StableHLO
tests/bench/test_scaling_metadata.py-372-        "  %c = collective-permute-done(%a)",     # async companion -> excluded
tests/bench/test_scaling_metadata.py-373-        "  %d = collective_permute_done(%b)",     # async companion -> excluded
tests/bench/test_scaling_metadata.py-374-        "  %e = all-gather(%z)",                  # different collective
tests/bench/test_scaling_metadata.py-375-    ])
tests/bench/test_scaling_metadata.py-376-    assert md.count_collective_permutes(hlo) == 2
tests/bench/test_scaling_metadata.py-377-    assert md.count_collective_permutes("no collectives here") == 0
tests/bench/test_scaling_metadata.py-378-
tests/bench/test_scaling_metadata.py-379-
tests/bench/test_scaling_metadata.py-380-def _cpu_compile():
tests/bench/test_scaling_metadata.py-381-    """Pin the HLO-probe compiles to a CPU device.
tests/bench/test_scaling_metadata.py-382-
tests/bench/test_scaling_metadata.py-383-    The probes read ``compile().as_text()``, and some backends return None for
tests/bench/test_scaling_metadata.py-384-    it (the experimental Apple ``mps`` plugin does) — which would make the
tests/bench/test_scaling_metadata.py:385:    "collective-free fn -> all-zero census" assertion vacuously unreachable on
tests/bench/test_scaling_metadata.py-386-    a dev laptop.  CPU is also the reproducible-by-construction reference the
tests/bench/test_scaling_metadata.py:387:    census docstring names (the DEFAULT schedule, no GPU collective-combining),
tests/bench/test_scaling_metadata.py-388-    so this keeps the assertion meaningful on every machine.  Production probes
tests/bench/test_scaling_metadata.py:389:    deliberately do NOT pin: cluster jobs must census the REAL on-device
tests/bench/test_scaling_metadata.py-390-    executable.
tests/bench/test_scaling_metadata.py-391-    """
tests/bench/test_scaling_metadata.py-392-    import jax
tests/bench/test_scaling_metadata.py-393-    return jax.default_device(jax.devices("cpu")[0])
tests/bench/test_scaling_metadata.py-394-
tests/bench/test_scaling_metadata.py-395-
tests/bench/test_scaling_metadata.py-396-def test_hlo_collective_permutes_lowers_counts_and_is_error_safe():
tests/bench/test_scaling_metadata.py-397-    """The best-effort probe lowers a fn and counts its collective-permutes: a
tests/bench/test_scaling_metadata.py-398-    fn with none -> 0; an unlowerable fn -> None (never raises). Real ppermute
tests/bench/test_scaling_metadata.py-399-    counting is exercised by the MPAS/cube bench gates and
tests/bench/test_scaling_metadata.py-400-    count_collective_permutes' synthetic HLO test above."""
tests/bench/test_scaling_metadata.py-401-    import jax.numpy as jnp
tests/bench/test_scaling_metadata.py-402-    with _cpu_compile():
tests/bench/test_scaling_metadata.py-403-        assert md.hlo_collective_permutes(lambda x: x + 1, jnp.arange(4.0)) == 0
tests/bench/test_scaling_metadata.py-404-
tests/bench/test_scaling_metadata.py-405-        def _boom(x):
tests/bench/test_scaling_metadata.py-406-            raise RuntimeError("unlowerable")
tests/bench/test_scaling_metadata.py-407-        assert md.hlo_collective_permutes(_boom, jnp.arange(4.0)) is None
tests/bench/test_scaling_metadata.py-408-
tests/bench/test_scaling_metadata.py-409-
tests/bench/test_scaling_metadata.py:410:def test_count_collectives_full_census_all_families():
tests/bench/test_scaling_metadata.py:411:    """Full census counts every collective family with the same op-call-form
tests/bench/test_scaling_metadata.py-412-    discipline: async ``-start`` once (``-done`` excluded), StableHLO
tests/bench/test_scaling_metadata.py-413-    underscore + optimized hyphen, and a config-header flag echo that merely
tests/bench/test_scaling_metadata.py-414-    CONTAINS an op name never inflates the count."""
tests/bench/test_scaling_metadata.py-415-    hlo = "\n".join([
tests/bench/test_scaling_metadata.py-416-        # config-header echo of XLA_FLAGS -> must NOT match (no op-call paren)
tests/bench/test_scaling_metadata.py-417-        "  // xla_gpu_collective_permute_combine_threshold_bytes=33554432",
tests/bench/test_scaling_metadata.py-418-        "  %a = collective-permute(%x)",             # permute (optimized)
tests/bench/test_scaling_metadata.py-419-        "  %b = collective_permute(%y)",             # permute (StableHLO)
tests/bench/test_scaling_metadata.py-420-        "  %c = collective-permute-done(%a)",        # async companion -> drop
tests/bench/test_scaling_metadata.py-421-        "  %r1 = all-reduce(%p)",                    # reduction (hyphen)
--
tests/bench/test_scaling_metadata.py-434-    assert c["total"] == 7
tests/bench/test_scaling_metadata.py-435-    # permute family stays bit-identical to the canonical scalar helper
tests/bench/test_scaling_metadata.py-436-    assert c["collective_permute"] == md.count_collective_permutes(hlo)
tests/bench/test_scaling_metadata.py-437-
tests/bench/test_scaling_metadata.py-438-    empty = md.count_collectives("no collectives here")
tests/bench/test_scaling_metadata.py-439-    assert empty["total"] == 0
tests/bench/test_scaling_metadata.py-440-    assert set(empty) == {"collective_permute", "all_reduce", "all_gather",
tests/bench/test_scaling_metadata.py-441-                          "all_to_all", "reduce_scatter", "total"}
tests/bench/test_scaling_metadata.py-442-
tests/bench/test_scaling_metadata.py-443-
tests/bench/test_scaling_metadata.py:444:def test_census_does_not_false_drop_ops_with_done_in_metadata():
tests/bench/test_scaling_metadata.py-445-    """Regression (codex): the ``-done`` async COMPANION is excluded by the
tests/bench/test_scaling_metadata.py-446-    regex structurally (``op-done(`` never matches ``op(?:[_-]start)?\\(``), so
tests/bench/test_scaling_metadata.py-447-    a line that merely CONTAINS the substring "done" elsewhere — an XLA
tests/bench/test_scaling_metadata.py-448-    metadata op_name, a ``%done_*`` SSA name — must STILL be counted.  A blunt
tests/bench/test_scaling_metadata.py-449-    ``"done" not in line`` filter would false-drop these to zero."""
tests/bench/test_scaling_metadata.py-450-    hlo = "\n".join([
tests/bench/test_scaling_metadata.py-451-        '  %r = all-reduce(%p), metadata={op_name="jit(step)/done_stage/psum"}',
tests/bench/test_scaling_metadata.py-452-        '  %done_mass = f32[] collective-permute(%q)',
tests/bench/test_scaling_metadata.py-453-        '  %g = all-gather(%z), metadata={op_name="reduce_done/x"}',
tests/bench/test_scaling_metadata.py-454-    ])
tests/bench/test_scaling_metadata.py-455-    c = md.count_collectives(hlo)
tests/bench/test_scaling_metadata.py-456-    assert c["all_reduce"] == 1        # NOT dropped despite "done" in metadata
tests/bench/test_scaling_metadata.py-457-    assert c["collective_permute"] == 1  # NOT dropped despite %done_ SSA name
tests/bench/test_scaling_metadata.py-458-    assert c["all_gather"] == 1
tests/bench/test_scaling_metadata.py-459-    # canonical permute helper is fixed by the same shared counter
tests/bench/test_scaling_metadata.py-460-    assert md.count_collective_permutes(hlo) == 1
tests/bench/test_scaling_metadata.py-461-
tests/bench/test_scaling_metadata.py-462-
tests/bench/test_scaling_metadata.py:463:def test_hlo_collective_census_lowers_and_is_error_safe():
tests/bench/test_scaling_metadata.py:464:    """Best-effort full-census probe: a collective-free fn -> all-zero dict;
tests/bench/test_scaling_metadata.py-465-    an unlowerable fn -> None (never raises)."""
tests/bench/test_scaling_metadata.py-466-    import jax.numpy as jnp
tests/bench/test_scaling_metadata.py-467-    with _cpu_compile():
tests/bench/test_scaling_metadata.py:468:        census = md.hlo_collective_census(lambda x: x + 1, jnp.arange(4.0))
tests/bench/test_scaling_metadata.py:469:        assert census is not None and census["total"] == 0
tests/bench/test_scaling_metadata.py-470-
tests/bench/test_scaling_metadata.py-471-        def _boom(x):
tests/bench/test_scaling_metadata.py-472-            raise RuntimeError("unlowerable")
tests/bench/test_scaling_metadata.py:473:        assert md.hlo_collective_census(_boom, jnp.arange(4.0)) is None
--
scripts/run/run_correction_campaign.py-1668-
scripts/run/run_correction_campaign.py-1669-    if x is None:
scripts/run/run_correction_campaign.py-1670-        return None
scripts/run/run_correction_campaign.py-1671-    v = float(x)
scripts/run/run_correction_campaign.py-1672-    return v if math.isfinite(v) else None
scripts/run/run_correction_campaign.py-1673-
scripts/run/run_correction_campaign.py-1674-
scripts/run/run_correction_campaign.py-1675-def _summary_to_json(summary):
scripts/run/run_correction_campaign.py-1676-    """JSON-serializable form of a :class:`CampaignSummary` for the output file."""
scripts/run/run_correction_campaign.py-1677-    return {
scripts/run/run_correction_campaign.py:1678:        "n_rounds": summary.n_rounds, "n_accepted": summary.n_accepted,
scripts/run/run_correction_campaign.py-1679-        "acceptance_rate": summary.acceptance_rate, "stop_reason": summary.stop_reason,
scripts/run/run_correction_campaign.py-1680-        "initial_bias": _json_finite(summary.initial_bias),
scripts/run/run_correction_campaign.py-1681-        "final_bias": _json_finite(summary.final_bias),
scripts/run/run_correction_campaign.py-1682-        "absolute_reduction": _json_finite(summary.absolute_reduction),
scripts/run/run_correction_campaign.py-1683-        "fractional_reduction": _json_finite(summary.fractional_reduction),
scripts/run/run_correction_campaign.py-1684-        "n_diagnosed_total": summary.n_diagnosed_total,
scripts/run/run_correction_campaign.py-1685-        "n_diagnoses_valid_total": summary.n_diagnoses_valid_total,
scripts/run/run_correction_campaign.py-1686-        # NB: the per-round trajectory is NOT re-serialized here — the campaign output
scripts/run/run_correction_campaign.py-1687-        # already carries it at the TOP LEVEL as ``biases`` (per-round [baseline,
scripts/run/run_correction_campaign.py-1688-        # updated, …], NaN-sanitized) + ``accepted``, which the plotter reads. The
--
tests/bench/test_m1_accounting.py-125-        halo_messages_per_step=0,
tests/bench/test_m1_accounting.py-126-        halo_bytes_per_step=0,
tests/bench/test_m1_accounting.py-127-        n_reductions_per_step=0,
tests/bench/test_m1_accounting.py-128-        rank_imbalance=1.0,
tests/bench/test_m1_accounting.py-129-    )
tests/bench/test_m1_accounting.py-130-    assert out["t_bound_ms"] == pytest.approx(5.0)
tests/bench/test_m1_accounting.py-131-    assert out["bound_calibrated"] is False
tests/bench/test_m1_accounting.py-132-    assert out["bound_incomplete_reason"] is None
tests/bench/test_m1_accounting.py-133-
tests/bench/test_m1_accounting.py-134-
tests/bench/test_m1_accounting.py:135:def test_bound_missing_comm_census_is_null_and_named():
tests/bench/test_m1_accounting.py-136-    out = calibrated_bound(
tests/bench/test_m1_accounting.py-137-        measured_fused_step_ms=8.0,
tests/bench/test_m1_accounting.py-138-        single_device_fused_step_ms=4.0,
tests/bench/test_m1_accounting.py:139:        halo_messages_per_step=None,            # no census for this config
tests/bench/test_m1_accounting.py-140-        halo_bytes_per_step=None,
tests/bench/test_m1_accounting.py-141-        n_reductions_per_step=None,
tests/bench/test_m1_accounting.py-142-        rank_imbalance=1.05,
tests/bench/test_m1_accounting.py-143-    )
tests/bench/test_m1_accounting.py-144-    assert out["t_bound_ms"] is None
tests/bench/test_m1_accounting.py-145-    for name in ("halo_messages_per_step", "halo_bytes_per_step",
tests/bench/test_m1_accounting.py-146-                 "n_reductions_per_step"):
tests/bench/test_m1_accounting.py-147-        assert name in out["bound_incomplete_reason"]
tests/bench/test_m1_accounting.py-148-
tests/bench/test_m1_accounting.py-149-
--
tests/bench/test_m1_accounting.py-205-
tests/bench/test_m1_accounting.py-206-
tests/bench/test_m1_accounting.py-207-def test_comm_3d_slab_levels():
tests/bench/test_m1_accounting.py-208-    out = comm_accounting(
tests/bench/test_m1_accounting.py-209-        halo_messages_per_step=4, n_lon=100, nlev=20, dtype_bytes=4,
tests/bench/test_m1_accounting.py-210-    )
tests/bench/test_m1_accounting.py-211-    assert out["halo_bytes_per_message"] == 100 * 20 * 4
tests/bench/test_m1_accounting.py-212-    assert out["halo_bytes_per_step"] == 4 * 8000
tests/bench/test_m1_accounting.py-213-
tests/bench/test_m1_accounting.py-214-
tests/bench/test_m1_accounting.py:215:def test_comm_unknown_census_is_all_null_never_fabricated():
tests/bench/test_m1_accounting.py-216-    out = comm_accounting(halo_messages_per_step=None,
tests/bench/test_m1_accounting.py:217:                          scope_note="no census for this config")
tests/bench/test_m1_accounting.py-218-    assert out["halo_messages_per_step"] is None
tests/bench/test_m1_accounting.py-219-    assert out["halo_bytes_per_message"] is None
tests/bench/test_m1_accounting.py-220-    assert out["halo_bytes_per_step"] is None
tests/bench/test_m1_accounting.py:221:    assert out["comm_scope_note"] == "no census for this config"
tests/bench/test_m1_accounting.py-222-
tests/bench/test_m1_accounting.py-223-
tests/bench/test_m1_accounting.py-224-def test_comm_zero_messages_zero_bytes():
tests/bench/test_m1_accounting.py-225-    out = comm_accounting(halo_messages_per_step=0, bytes_per_message=0)
tests/bench/test_m1_accounting.py-226-    assert out["halo_messages_per_step"] == 0
tests/bench/test_m1_accounting.py-227-    assert out["halo_bytes_per_step"] == 0
tests/bench/test_m1_accounting.py-228-
tests/bench/test_m1_accounting.py-229-
tests/bench/test_m1_accounting.py-230-def test_comm_message_count_without_size_raises():
tests/bench/test_m1_accounting.py-231-    with pytest.raises(ValueError, match="payload size"):
tests/bench/test_m1_accounting.py-232-        comm_accounting(halo_messages_per_step=5)
tests/bench/test_m1_accounting.py-233-
tests/bench/test_m1_accounting.py-234-
tests/bench/test_m1_accounting.py-235-def test_comm_lower_bound_flag_machine_readable():
tests/bench/test_m1_accounting.py:236:    """The partial-census fact must survive as a FIELD, not only prose
tests/bench/test_m1_accounting.py-237-    (codex batch4): tri-state True (undercount) / False (exact) / None
tests/bench/test_m1_accounting.py-238-    (unknown), default None."""
tests/bench/test_m1_accounting.py-239-    out = comm_accounting(halo_messages_per_step=4, bytes_per_message=8,
tests/bench/test_m1_accounting.py-240-                          bytes_are_lower_bound=True)
tests/bench/test_m1_accounting.py-241-    assert out["halo_bytes_is_lower_bound"] is True
tests/bench/test_m1_accounting.py-242-    out = comm_accounting(halo_messages_per_step=0, bytes_per_message=0,
tests/bench/test_m1_accounting.py-243-                          bytes_are_lower_bound=False)
tests/bench/test_m1_accounting.py-244-    assert out["halo_bytes_is_lower_bound"] is False
tests/bench/test_m1_accounting.py-245-    out = comm_accounting(halo_messages_per_step=None)
tests/bench/test_m1_accounting.py-246-    assert out["halo_bytes_is_lower_bound"] is None
--
scripts/bench/bench_ocean_mpi_scaling.py-2610-            results=[result],
scripts/bench/bench_ocean_mpi_scaling.py-2611-            backend=jax.default_backend().upper(),
scripts/bench/bench_ocean_mpi_scaling.py-2612-            hostname=os.environ.get("HOSTNAME", "unknown"),
scripts/bench/bench_ocean_mpi_scaling.py-2613-        )
scripts/bench/bench_ocean_mpi_scaling.py-2614-        # component="ocean": without it write_json stamps its atmosphere
scripts/bench/bench_ocean_mpi_scaling.py-2615-        # default on every row (mislabel); n_ranks_true: jax cannot see the
scripts/bench/bench_ocean_mpi_scaling.py-2616-        # mpirun world (process_count()==1 per rank), so the real rank count
scripts/bench/bench_ocean_mpi_scaling.py-2617-        # must be recorded explicitly — it also resolves transport="mpi4jax".
scripts/bench/bench_ocean_mpi_scaling.py-2618-        # solver_variant records the wide-halo/local-clamp levers so an A/B
scripts/bench/bench_ocean_mpi_scaling.py-2619-        # pair can never be conflated with the baseline in aggregation.
scripts/bench/bench_ocean_mpi_scaling.py:2620:        # (The analytic barotropic halo-message census lives in the SPMD
scripts/bench/bench_ocean_mpi_scaling.py:2621:        # bench's records — model.config is in scope there; here the census is
scripts/bench/bench_ocean_mpi_scaling.py-2622-        # derivable offline from solver_variant + n_barotropic_substeps, so it
scripts/bench/bench_ocean_mpi_scaling.py-2623-        # is deliberately not recomputed. Pre-merge codex note.)
scripts/bench/bench_ocean_mpi_scaling.py-2624-        _grid_label = "tripole" if args.tripole else "latlon"
scripts/bench/bench_ocean_mpi_scaling.py-2625-        _variant = args.baro_solver
scripts/bench/bench_ocean_mpi_scaling.py-2626-        if os.environ.get("LEGOESM_BARO_LOCAL_CLAMP", "0") == "1":
scripts/bench/bench_ocean_mpi_scaling.py-2627-            _variant += "+local_clamp"
scripts/bench/bench_ocean_mpi_scaling.py-2628-        if os.environ.get("LEGOESM_BARO_WIDE_HALO", "0") == "1":
scripts/bench/bench_ocean_mpi_scaling.py-2629-            _variant += "+wide_halo"
scripts/bench/bench_ocean_mpi_scaling.py-2630-        _md_over = {"solver_variant": _variant}
scripts/bench/bench_ocean_mpi_scaling.py-2631-        if part_metrics is not None:
--
tests/bench/test_scaling_diagnosis_device_gate.py-1-"""Direct tests for the ``--expect-devices`` anti-fake-scaling gate.
tests/bench/test_scaling_diagnosis_device_gate.py-2-
tests/bench/test_scaling_diagnosis_device_gate.py:3:A scaling/census row is only meaningful if the run actually spanned the device
tests/bench/test_scaling_diagnosis_device_gate.py-4-count it claims.  Rank count is NOT a device count: under route-A MPI each rank
tests/bench/test_scaling_diagnosis_device_gate.py-5-builds its mesh from its own local devices, so ``-np 2`` can span four device
tests/bench/test_scaling_diagnosis_device_gate.py-6-slots (both ranks inheriting ``CUDA_VISIBLE_DEVICES=0,1``) or one (both ranks
tests/bench/test_scaling_diagnosis_device_gate.py-7-pinned to the same GPU) — and a ``world_size``-based check passes in both cases
tests/bench/test_scaling_diagnosis_device_gate.py-8-while the recorded row is fake.  The gate therefore counts DISTINCT physical
tests/bench/test_scaling_diagnosis_device_gate.py-9-devices, and these tests lock both failure modes so neither can be silently
tests/bench/test_scaling_diagnosis_device_gate.py-10-weakened back into a rank-count check.
tests/bench/test_scaling_diagnosis_device_gate.py-11-"""
tests/bench/test_scaling_diagnosis_device_gate.py-12-
tests/bench/test_scaling_diagnosis_device_gate.py-13-from __future__ import annotations
--
scripts/bench/run_scaling_diagnosis.py-91-    # when the script is launched as ``python scripts/...`` (no
scripts/bench/run_scaling_diagnosis.py-92-    # ``PYTHONPATH=$PWD``).
scripts/bench/run_scaling_diagnosis.py-93-    import sys
scripts/bench/run_scaling_diagnosis.py-94-    from pathlib import Path
scripts/bench/run_scaling_diagnosis.py-95-    _repo_root = Path(__file__).resolve().parents[2]
scripts/bench/run_scaling_diagnosis.py-96-    if str(_repo_root) not in sys.path:
scripts/bench/run_scaling_diagnosis.py-97-        sys.path.insert(0, str(_repo_root))
scripts/bench/run_scaling_diagnosis.py-98-
scripts/bench/run_scaling_diagnosis.py-99-    # GPU affinity for MPI: pin each rank to its node-local GPU.  ONLY when the
scripts/bench/run_scaling_diagnosis.py-100-    # caller has not already set CUDA_VISIBLE_DEVICES — otherwise a single-
scripts/bench/run_scaling_diagnosis.py:101:    # process census/scaling ladder that exports e.g. CUDA_VISIBLE_DEVICES=0,1
scripts/bench/run_scaling_diagnosis.py-102-    # to shard over 2 GPUs would be silently narrowed to GPU 0 (SLURM sets
scripts/bench/run_scaling_diagnosis.py-103-    # SLURM_LOCALID=0 even for a 1-task batch step), collapsing the run to one
scripts/bench/run_scaling_diagnosis.py:104:    # device and recording a FAKE serial census under the multi-device dir.
scripts/bench/run_scaling_diagnosis.py-105-    # A real multi-task srun/mpirun step leaves CVD unset per rank, so the
scripts/bench/run_scaling_diagnosis.py-106-    # local-rank pin still applies there.
scripts/bench/run_scaling_diagnosis.py-107-    if "CUDA_VISIBLE_DEVICES" not in os.environ:
scripts/bench/run_scaling_diagnosis.py-108-        local_rank = (
scripts/bench/run_scaling_diagnosis.py-109-            os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
scripts/bench/run_scaling_diagnosis.py-110-            or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK")
scripts/bench/run_scaling_diagnosis.py-111-            or os.environ.get("SLURM_LOCALID")
scripts/bench/run_scaling_diagnosis.py-112-        )
scripts/bench/run_scaling_diagnosis.py-113-        if local_rank is not None:
scripts/bench/run_scaling_diagnosis.py-114-            os.environ["CUDA_VISIBLE_DEVICES"] = local_rank
--
scripts/bench/run_scaling_diagnosis.py-467-
scripts/bench/run_scaling_diagnosis.py-468-    return {
scripts/bench/run_scaling_diagnosis.py-469-        "compile_time_s": round(compile_time, 3),
scripts/bench/run_scaling_diagnosis.py-470-        "scan_total_s": round(total_time, 4),
scripts/bench/run_scaling_diagnosis.py-471-        "time_per_step_ms": round(time_per_step_ms, 3),
scripts/bench/run_scaling_diagnosis.py-472-        "sypd": round(sypd, 4),
scripts/bench/run_scaling_diagnosis.py-473-        "n_timing": n_timing,
scripts/bench/run_scaling_diagnosis.py-474-    }
scripts/bench/run_scaling_diagnosis.py-475-
scripts/bench/run_scaling_diagnosis.py-476-
scripts/bench/run_scaling_diagnosis.py:477:def run_collective_census(step_fn, state, dt: float) -> dict:
scripts/bench/run_scaling_diagnosis.py:478:    """Static HLO collective census of the compiled per-step function.
scripts/bench/run_scaling_diagnosis.py-479-
scripts/bench/run_scaling_diagnosis.py-480-    The remaining strong-scaling loss at small per-device tiles is
scripts/bench/run_scaling_diagnosis.py:481:    ``per-step message count x per-message latency`` (2026-07-08 census note):
scripts/bench/run_scaling_diagnosis.py-482-    f64 and f32 GPU strong-scaling curves coincide => LATENCY-bound, not
scripts/bench/run_scaling_diagnosis.py-483-    bandwidth-bound, so the message COUNT — not the byte volume — is the lever.
scripts/bench/run_scaling_diagnosis.py-484-    This makes that count a first-class, tracked diagnostic instead of a
scripts/bench/run_scaling_diagnosis.py-485-    one-off ``scripts/tmp`` probe.
scripts/bench/run_scaling_diagnosis.py-486-
scripts/bench/run_scaling_diagnosis.py-487-    STATIC: the count is fixed by the partition / edge-coloring schedule, not
scripts/bench/run_scaling_diagnosis.py-488-    the data.  A CPU virtual-device compile reproduces the DEFAULT-flags count,
scripts/bench/run_scaling_diagnosis.py-489-    which is why this same probe runs cheaply in local CI; but it is NOT fully
scripts/bench/run_scaling_diagnosis.py-490-    backend-independent — GPU-only XLA collective combining / pipelined-p2p
scripts/bench/run_scaling_diagnosis.py-491-    (lane T) can LOWER the optimized count, so the cluster jobs run this against
scripts/bench/run_scaling_diagnosis.py-492-    the real on-device executable to capture what the GPU actually launches.
scripts/bench/run_scaling_diagnosis.py-493-    Counts XLA collectives only — route-A ``mpi4jax`` sendrecv are opaque
scripts/bench/run_scaling_diagnosis.py-494-    custom-calls (invisible here AND to the XLA latency-hiding scheduler; that
scripts/bench/run_scaling_diagnosis.py:495:    opacity is itself the route-A ceiling), so a ~0 census on an MPI rank is the
scripts/bench/run_scaling_diagnosis.py-496-    expected reading, not a bug.
scripts/bench/run_scaling_diagnosis.py-497-
scripts/bench/run_scaling_diagnosis.py:498:    Returns the per-family census plus the device count it was taken at (the
scripts/bench/run_scaling_diagnosis.py-499-    schedule is device-count dependent: more shards => more exchange rounds).
scripts/bench/run_scaling_diagnosis.py-500-    """
scripts/bench/run_scaling_diagnosis.py-501-    import jax
scripts/bench/run_scaling_diagnosis.py-502-    sys.path.insert(0, str(Path(__file__).resolve().parent))
scripts/bench/run_scaling_diagnosis.py:503:    from metadata import hlo_collective_census  # sibling scripts/bench module
scripts/bench/run_scaling_diagnosis.py-504-
scripts/bench/run_scaling_diagnosis.py:505:    census = hlo_collective_census(step_fn, state, dt)
scripts/bench/run_scaling_diagnosis.py-506-    return {
scripts/bench/run_scaling_diagnosis.py-507-        "n_devices": jax.local_device_count(),
scripts/bench/run_scaling_diagnosis.py:508:        "census": census,  # None if the step could not be lowered/compiled
scripts/bench/run_scaling_diagnosis.py-509-    }
scripts/bench/run_scaling_diagnosis.py-510-
scripts/bench/run_scaling_diagnosis.py-511-
scripts/bench/run_scaling_diagnosis.py-512-# ---------------------------------------------------------------------------
scripts/bench/run_scaling_diagnosis.py-513-# Main
scripts/bench/run_scaling_diagnosis.py-514-# ---------------------------------------------------------------------------
scripts/bench/run_scaling_diagnosis.py-515-
scripts/bench/run_scaling_diagnosis.py-516-def build_parser() -> argparse.ArgumentParser:
scripts/bench/run_scaling_diagnosis.py-517-    p = argparse.ArgumentParser(
scripts/bench/run_scaling_diagnosis.py-518-        description="Comprehensive scaling diagnostics for legoESM",
scripts/bench/run_scaling_diagnosis.py-519-        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
scripts/bench/run_scaling_diagnosis.py-520-    )
scripts/bench/run_scaling_diagnosis.py-521-    p.add_argument("--mode", choices=["full", "quick", "halo-only",
scripts/bench/run_scaling_diagnosis.py-522-                                       "reduction", "roofline", "profile",
scripts/bench/run_scaling_diagnosis.py:523:                                       "census"],
scripts/bench/run_scaling_diagnosis.py-524-                   default="full")
scripts/bench/run_scaling_diagnosis.py-525-    p.add_argument("--grid", choices=["cubed-sphere"], default="cubed-sphere",
scripts/bench/run_scaling_diagnosis.py-526-                   help="Grid type (cubed-sphere supported for now)")
scripts/bench/run_scaling_diagnosis.py-527-    p.add_argument("--n", type=int, default=48, help="Per-face resolution")
scripts/bench/run_scaling_diagnosis.py-528-    p.add_argument("--nlev", type=int, default=40, help="Vertical levels")
scripts/bench/run_scaling_diagnosis.py-529-    p.add_argument("--dt", type=float, default=600.0, help="Timestep (s)")
scripts/bench/run_scaling_diagnosis.py-530-    p.add_argument("--precision", choices=["float32", "float64"],
scripts/bench/run_scaling_diagnosis.py-531-                   default="float64")
scripts/bench/run_scaling_diagnosis.py-532-    p.add_argument("--physics", choices=["none", "held_suarez"],
scripts/bench/run_scaling_diagnosis.py-533-                   default="none")
--
scripts/bench/run_scaling_diagnosis.py-540-                   help="Capture XLA/TensorBoard profile traces")
scripts/bench/run_scaling_diagnosis.py-541-    p.add_argument("--output-dir", default="results/scaling_diagnosis")
scripts/bench/run_scaling_diagnosis.py-542-    p.add_argument("--no-timestamp", action="store_true")
scripts/bench/run_scaling_diagnosis.py-543-    p.add_argument("--expect-devices", type=int, default=None,
scripts/bench/run_scaling_diagnosis.py-544-                   help="Anti-fake-scaling gate: fail LOUDLY unless the run "
scripts/bench/run_scaling_diagnosis.py-545-                        "actually shards over exactly this many DISTINCT "
scripts/bench/run_scaling_diagnosis.py-546-                        "physical devices (local devices, gathered across MPI "
scripts/bench/run_scaling_diagnosis.py-547-                        "ranks and de-duplicated, so neither a rank pinned to "
scripts/bench/run_scaling_diagnosis.py-548-                        "a shared GPU nor a rank holding extra GPUs passes). "
scripts/bench/run_scaling_diagnosis.py-549-                        "Used by the cluster ladders so a rung narrowed to "
scripts/bench/run_scaling_diagnosis.py:550:                        "1 GPU can never record a fake census.")
scripts/bench/run_scaling_diagnosis.py-551-    return p
scripts/bench/run_scaling_diagnosis.py-552-
scripts/bench/run_scaling_diagnosis.py-553-
scripts/bench/run_scaling_diagnosis.py-554-def device_identity_keys(local_positions, cvd: str | None, hostname: str):
scripts/bench/run_scaling_diagnosis.py-555-    """Physical-device identity key per LOCAL device slot: ``(host, token)``.
scripts/bench/run_scaling_diagnosis.py-556-
scripts/bench/run_scaling_diagnosis.py-557-    ``local_positions`` are the 0-based positions of this process's devices
scripts/bench/run_scaling_diagnosis.py-558-    within ``jax.local_devices()`` (that ordering IS the CUDA ordinal order).
scripts/bench/run_scaling_diagnosis.py-559-
scripts/bench/run_scaling_diagnosis.py-560-    A device id is CUDA_VISIBLE_DEVICES-RELATIVE, so the raw ordinal is not an
--
scripts/bench/run_scaling_diagnosis.py-614-    # Environment setup
scripts/bench/run_scaling_diagnosis.py-615-    # ---------------------------------------------------------------
scripts/bench/run_scaling_diagnosis.py-616-    _configure_env(args.precision)
scripts/bench/run_scaling_diagnosis.py-617-
scripts/bench/run_scaling_diagnosis.py-618-    import jax
scripts/bench/run_scaling_diagnosis.py-619-    import jax.numpy as jnp
scripts/bench/run_scaling_diagnosis.py-620-
scripts/bench/run_scaling_diagnosis.py-621-    rank, world_size = _init_distributed(args.grid, global_n=args.n)
scripts/bench/run_scaling_diagnosis.py-622-    is_rank0 = (rank == 0)
scripts/bench/run_scaling_diagnosis.py-623-
scripts/bench/run_scaling_diagnosis.py:624:    # Anti-fake-scaling gate (see --expect-devices): a census/scaling row is
scripts/bench/run_scaling_diagnosis.py-625-    # meaningless if the process did not actually shard over the intended
scripts/bench/run_scaling_diagnosis.py-626-    # device count.  Fail LOUDLY here, before any measurement.
scripts/bench/run_scaling_diagnosis.py-627-    #
scripts/bench/run_scaling_diagnosis.py-628-    # Rank count is NOT a device count (codex P1): under MPI each rank builds
scripts/bench/run_scaling_diagnosis.py-629-    # its mesh from its OWN local devices, so `-np 2` with both ranks
scripts/bench/run_scaling_diagnosis.py-630-    # inheriting CUDA_VISIBLE_DEVICES=0,1 spans four device slots, while two
scripts/bench/run_scaling_diagnosis.py-631-    # ranks oversubscribed onto one physical GPU span one.  Both would pass a
scripts/bench/run_scaling_diagnosis.py-632-    # world_size check while recording a fake row.  So count DISTINCT physical
scripts/bench/run_scaling_diagnosis.py-633-    # devices, identified by device_identity_keys() — under route-A MPI the
scripts/bench/run_scaling_diagnosis.py-634-    # ranks are independent JAX processes whose local device ordinals both
--
scripts/bench/run_scaling_diagnosis.py-726-    # ---------------------------------------------------------------
scripts/bench/run_scaling_diagnosis.py-727-    # Create harness
scripts/bench/run_scaling_diagnosis.py-728-    # ---------------------------------------------------------------
scripts/bench/run_scaling_diagnosis.py-729-    harness = DiagnosticHarness(
scripts/bench/run_scaling_diagnosis.py-730-        rank=rank, world_size=world_size,
scripts/bench/run_scaling_diagnosis.py-731-        grid_type=args.grid, config=config_info,
scripts/bench/run_scaling_diagnosis.py-732-    )
scripts/bench/run_scaling_diagnosis.py-733-    harness.memory_snapshot("initial")
scripts/bench/run_scaling_diagnosis.py-734-
scripts/bench/run_scaling_diagnosis.py-735-    # ---------------------------------------------------------------
scripts/bench/run_scaling_diagnosis.py:736:    # [1b] Static collective census (message-count = latency-bound lever)
scripts/bench/run_scaling_diagnosis.py-737-    # ---------------------------------------------------------------
scripts/bench/run_scaling_diagnosis.py:738:    census_info = None
scripts/bench/run_scaling_diagnosis.py:739:    if args.mode in ("full", "quick", "census"):
scripts/bench/run_scaling_diagnosis.py-740-        if is_rank0:
scripts/bench/run_scaling_diagnosis.py:741:            print("\n[1b] Collective census (static HLO message count)...",
scripts/bench/run_scaling_diagnosis.py-742-                  flush=True)
scripts/bench/run_scaling_diagnosis.py:743:        census_info = run_collective_census(step_fn, state, args.dt)
scripts/bench/run_scaling_diagnosis.py-744-        if is_rank0:
scripts/bench/run_scaling_diagnosis.py:745:            c = census_info.get("census")
scripts/bench/run_scaling_diagnosis.py-746-            if c is None:
scripts/bench/run_scaling_diagnosis.py:747:                print("  census: unavailable (step not lowerable on this "
scripts/bench/run_scaling_diagnosis.py-748-                      "backend)")
scripts/bench/run_scaling_diagnosis.py-749-            else:
scripts/bench/run_scaling_diagnosis.py:750:                print(f"  {census_info['n_devices']} device(s): "
scripts/bench/run_scaling_diagnosis.py-751-                      f"{c['collective_permute']} collective-permute, "
scripts/bench/run_scaling_diagnosis.py-752-                      f"{c['all_reduce']} all-reduce, "
scripts/bench/run_scaling_diagnosis.py-753-                      f"{c['all_gather']} all-gather / step")
scripts/bench/run_scaling_diagnosis.py:754:                if census_info["n_devices"] == 1:
scripts/bench/run_scaling_diagnosis.py-755-                    print("  (single device: no inter-device schedule — run "
scripts/bench/run_scaling_diagnosis.py-756-                          "with XLA_FLAGS=--xla_force_host_platform_device_"
scripts/bench/run_scaling_diagnosis.py-757-                          "count=N or on multi-GPU for the real count)")
scripts/bench/run_scaling_diagnosis.py-758-
scripts/bench/run_scaling_diagnosis.py:759:    # ``census`` mode is census-only: emit the report and stop before the
scripts/bench/run_scaling_diagnosis.py-760-    # timing phases (each of which is gated on its own mode below and so is
scripts/bench/run_scaling_diagnosis.py-761-    # already skipped — this early return just avoids the empty setup churn).
scripts/bench/run_scaling_diagnosis.py:762:    if args.mode == "census":
scripts/bench/run_scaling_diagnosis.py-763-        harness.dump(output_dir / f"diag_rank{rank}.json",
scripts/bench/run_scaling_diagnosis.py:764:                     extra={"collective_census": census_info})
scripts/bench/run_scaling_diagnosis.py-765-        if is_rank0:
scripts/bench/run_scaling_diagnosis.py-766-            summary = {"world_size": world_size, "config": config_info,
scripts/bench/run_scaling_diagnosis.py:767:                       "collective_census": census_info}
scripts/bench/run_scaling_diagnosis.py-768-            with open(output_dir / "summary.json", "w") as f:
scripts/bench/run_scaling_diagnosis.py-769-                json.dump(summary, f, indent=2, default=str)
scripts/bench/run_scaling_diagnosis.py-770-            print(f"\n  Census written to {output_dir}/")
scripts/bench/run_scaling_diagnosis.py-771-        return 0
scripts/bench/run_scaling_diagnosis.py-772-
scripts/bench/run_scaling_diagnosis.py-773-    # ---------------------------------------------------------------
scripts/bench/run_scaling_diagnosis.py-774-    # [2] Per-step profiling
scripts/bench/run_scaling_diagnosis.py-775-    # ---------------------------------------------------------------
scripts/bench/run_scaling_diagnosis.py-776-    if args.mode in ("full", "quick"):
scripts/bench/run_scaling_diagnosis.py-777-        if is_rank0:
--
scripts/bench/run_scaling_diagnosis.py-889-            step_fn, state_xla, args.dt,
scripts/bench/run_scaling_diagnosis.py-890-            str(output_dir), n_steps=10, rank=rank,
scripts/bench/run_scaling_diagnosis.py-891-        )
scripts/bench/run_scaling_diagnosis.py-892-        if is_rank0:
scripts/bench/run_scaling_diagnosis.py-893-            print(f"  Profile: {xla_info.get('status', 'unknown')}")
scripts/bench/run_scaling_diagnosis.py-894-
scripts/bench/run_scaling_diagnosis.py-895-    # ---------------------------------------------------------------
scripts/bench/run_scaling_diagnosis.py-896-    # Build and save reports
scripts/bench/run_scaling_diagnosis.py-897-    # ---------------------------------------------------------------
scripts/bench/run_scaling_diagnosis.py-898-    extra = {}
scripts/bench/run_scaling_diagnosis.py:899:    if census_info:
scripts/bench/run_scaling_diagnosis.py:900:        extra["collective_census"] = census_info
scripts/bench/run_scaling_diagnosis.py-901-    if scan_info:
scripts/bench/run_scaling_diagnosis.py-902-        extra["scan_throughput"] = scan_info
scripts/bench/run_scaling_diagnosis.py-903-    if halo_info:
scripts/bench/run_scaling_diagnosis.py-904-        extra["halo_profile"] = halo_info
scripts/bench/run_scaling_diagnosis.py-905-    if reduction_info:
scripts/bench/run_scaling_diagnosis.py-906-        extra["reduction_profile"] = reduction_info
scripts/bench/run_scaling_diagnosis.py-907-    if roofline_info:
scripts/bench/run_scaling_diagnosis.py-908-        extra["roofline"] = roofline_info
scripts/bench/run_scaling_diagnosis.py-909-    if overlap_info:
scripts/bench/run_scaling_diagnosis.py-910-        extra["overlap_estimate"] = overlap_info
--
scripts/bench/run_scaling_diagnosis.py-941-                "rank": rank,
scripts/bench/run_scaling_diagnosis.py-942-                "phase_timing": harness.timer.summary(),
scripts/bench/run_scaling_diagnosis.py-943-            }
scripts/bench/run_scaling_diagnosis.py-944-            all_data = comm.gather(local_data, root=0)
scripts/bench/run_scaling_diagnosis.py-945-            if is_rank0 and all_data:
scripts/bench/run_scaling_diagnosis.py-946-                summary = {
scripts/bench/run_scaling_diagnosis.py-947-                    "world_size": world_size,
scripts/bench/run_scaling_diagnosis.py-948-                    "config": config_info,
scripts/bench/run_scaling_diagnosis.py-949-                    "per_rank_timing": all_data,
scripts/bench/run_scaling_diagnosis.py-950-                    "scan_throughput": scan_info,
scripts/bench/run_scaling_diagnosis.py:951:                    "collective_census": census_info,
scripts/bench/run_scaling_diagnosis.py-952-                }
scripts/bench/run_scaling_diagnosis.py-953-                with open(output_dir / "summary.json", "w") as f:
scripts/bench/run_scaling_diagnosis.py-954-                    json.dump(summary, f, indent=2, default=str)
scripts/bench/run_scaling_diagnosis.py-955-        except Exception as e:
scripts/bench/run_scaling_diagnosis.py-956-            if is_rank0:
scripts/bench/run_scaling_diagnosis.py-957-                print(f"  Warning: cross-rank summary failed: {e}")
scripts/bench/run_scaling_diagnosis.py-958-    elif is_rank0:
scripts/bench/run_scaling_diagnosis.py-959-        summary = {
scripts/bench/run_scaling_diagnosis.py-960-            "world_size": 1,
scripts/bench/run_scaling_diagnosis.py-961-            "config": config_info,
scripts/bench/run_scaling_diagnosis.py-962-            "scan_throughput": scan_info,
scripts/bench/run_scaling_diagnosis.py:963:            "collective_census": census_info,
scripts/bench/run_scaling_diagnosis.py-964-        }
scripts/bench/run_scaling_diagnosis.py-965-        with open(output_dir / "summary.json", "w") as f:
scripts/bench/run_scaling_diagnosis.py-966-            json.dump(summary, f, indent=2, default=str)
scripts/bench/run_scaling_diagnosis.py-967-
scripts/bench/run_scaling_diagnosis.py-968-    if is_rank0:
scripts/bench/run_scaling_diagnosis.py-969-        print(f"\n{'='*72}")
scripts/bench/run_scaling_diagnosis.py-970-        print(f"  Diagnostics complete. Results in: {output_dir}/")
scripts/bench/run_scaling_diagnosis.py-971-        print(f"  Files:")
scripts/bench/run_scaling_diagnosis.py-972-        for p in sorted(output_dir.glob("*.json")):
scripts/bench/run_scaling_diagnosis.py-973-            print(f"    {p.name} ({p.stat().st_size / 1024:.1f} KB)")
--
scripts/bench/bench_mpas_spmd_scaling.py-66-# baroclinic-wave IC the icosahedral lanes of run_levante_gpu_scaling use).
scripts/bench/bench_mpas_spmd_scaling.py-67-sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
scripts/bench/bench_mpas_spmd_scaling.py-68-# Bench dir for the shared metadata module (sibling-script import pattern).
scripts/bench/bench_mpas_spmd_scaling.py-69-sys.path.insert(0, str(Path(__file__).resolve().parent))
scripts/bench/bench_mpas_spmd_scaling.py-70-
scripts/bench/bench_mpas_spmd_scaling.py-71-# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
scripts/bench/bench_mpas_spmd_scaling.py-72-# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
scripts/bench/bench_mpas_spmd_scaling.py-73-# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
scripts/bench/bench_mpas_spmd_scaling.py-74-# imports JAX lazily, so this is safe before jax.distributed.initialize.
scripts/bench/bench_mpas_spmd_scaling.py-75-from metadata import (  # noqa: E402
scripts/bench/bench_mpas_spmd_scaling.py:76:    annotate_incomplete, hlo_collective_census, scaling_metadata,
scripts/bench/bench_mpas_spmd_scaling.py-77-    tidy_throughput_fields)
scripts/bench/bench_mpas_spmd_scaling.py-78-
scripts/bench/bench_mpas_spmd_scaling.py-79-# SPMD full-step parity tolerances — the FLOATING-POINT RE-ASSOCIATION floor
scripts/bench/bench_mpas_spmd_scaling.py-80-# of the sharded step (ppermute halo + mass-fix psum reduction-order change),
scripts/bench/bench_mpas_spmd_scaling.py-81-# NOT a bug margin; a real halo/partition regression shows up orders of
scripts/bench/bench_mpas_spmd_scaling.py-82-# magnitude above these.  Values extend the 1-step envelope of
scripts/bench/bench_mpas_spmd_scaling.py-83-# tests/parallel/test_voronoi_sharded_equivalence.py (u/T atol 1e-6, p_s
scripts/bench/bench_mpas_spmd_scaling.py-84-# atol 1e-1) to the smoke window; the floor grows with steps, hence the cap.
scripts/bench/bench_mpas_spmd_scaling.py-85-MPAS_PARITY_TOLS = {  # precision -> field -> (rtol, atol)
scripts/bench/bench_mpas_spmd_scaling.py-86-    "float64": {"u": (1.0e-5, 1.0e-5), "T": (1.0e-6, 1.0e-5),
--
scripts/bench/bench_mpas_spmd_scaling.py-89-                "p_s": (1.0e-3, 50.0)},
scripts/bench/bench_mpas_spmd_scaling.py-90-}
scripts/bench/bench_mpas_spmd_scaling.py-91-MPAS_PARITY_MAX_STEPS = 8
scripts/bench/bench_mpas_spmd_scaling.py-92-
scripts/bench/bench_mpas_spmd_scaling.py-93-# Conservation gate default: with fix_mass=True the step restores the global
scripts/bench/bench_mpas_spmd_scaling.py-94-# dry mass to the pre-step value each step, so the drift over a smoke window
scripts/bench/bench_mpas_spmd_scaling.py-95-# is the allreduce rounding floor, not scheme drift.
scripts/bench/bench_mpas_spmd_scaling.py-96-MASS_RTOL_DEFAULTS = {"float64": 1.0e-11, "float32": 1.0e-5}
scripts/bench/bench_mpas_spmd_scaling.py-97-
scripts/bench/bench_mpas_spmd_scaling.py-98-
scripts/bench/bench_mpas_spmd_scaling.py:99:def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
scripts/bench/bench_mpas_spmd_scaling.py-100-                          moist=False, lloyd_iterations=50):
scripts/bench/bench_mpas_spmd_scaling.py-101-    """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.
scripts/bench/bench_mpas_spmd_scaling.py-102-
scripts/bench/bench_mpas_spmd_scaling.py:103:    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
scripts/bench/bench_mpas_spmd_scaling.py-104-    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
scripts/bench/bench_mpas_spmd_scaling.py-105-    device count of THIS run's mesh/model (the two differ for the
scripts/bench/bench_mpas_spmd_scaling.py-106-    single-device reference leg of a ladder, via ``--reorder-for``).
scripts/bench/bench_mpas_spmd_scaling.py-107-    ``moist=True`` attaches the q_v/q_c/q_r tracers (moist baroclinic
scripts/bench/bench_mpas_spmd_scaling.py-108-    wave) so the sharded step's packed tracer halo exchange + RK tracer
scripts/bench/bench_mpas_spmd_scaling.py-109-    advection sit on the timed/gated path.
scripts/bench/bench_mpas_spmd_scaling.py-110-    """
scripts/bench/bench_mpas_spmd_scaling.py-111-    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
scripts/bench/bench_mpas_spmd_scaling.py-112-        MPASPrimitiveEquationConfig,
scripts/bench/bench_mpas_spmd_scaling.py-113-        MPASPrimitiveEquationModel,
scripts/bench/bench_mpas_spmd_scaling.py-114-    )
scripts/bench/bench_mpas_spmd_scaling.py-115-    from legoesm.grids.vertical import create_sigma_coordinate
scripts/bench/bench_mpas_spmd_scaling.py-116-    from legoesm.grids.voronoi import create_voronoi_mesh
scripts/bench/bench_mpas_spmd_scaling.py-117-    from legoesm.parallel.mesh import create_voronoi_device_mesh
scripts/bench/bench_mpas_spmd_scaling.py-118-    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
scripts/bench/bench_mpas_spmd_scaling.py-119-
scripts/bench/bench_mpas_spmd_scaling.py-120-    mesh = create_voronoi_mesh(subdivision_level=subdivision,
scripts/bench/bench_mpas_spmd_scaling.py-121-                               lloyd_iterations=lloyd_iterations)
scripts/bench/bench_mpas_spmd_scaling.py:122:    mesh = reorder_voronoi_for_sharding(mesh, reorder_target, method=method)
scripts/bench/bench_mpas_spmd_scaling.py-123-    if run_nd > 1 and (mesh.nCells % run_nd or mesh.nEdges % run_nd):
scripts/bench/bench_mpas_spmd_scaling.py:124:        # Padding only guarantees divisibility for reorder_target.
scripts/bench/bench_mpas_spmd_scaling.py-125-        raise SystemExit(
scripts/bench/bench_mpas_spmd_scaling.py-126-            f"padded mesh (nCells={mesh.nCells}, nEdges={mesh.nEdges}) not "
scripts/bench/bench_mpas_spmd_scaling.py-127-            f"divisible by --n-devices {run_nd}; use a ladder where every "
scripts/bench/bench_mpas_spmd_scaling.py:128:            f"count divides --reorder-for ({reorder_target}).")
scripts/bench/bench_mpas_spmd_scaling.py-129-    sigma = create_sigma_coordinate(nlev)
scripts/bench/bench_mpas_spmd_scaling.py-130-    # Same recipe as the icosahedral lane of run_levante_gpu_scaling /
scripts/bench/bench_mpas_spmd_scaling.py-131-    # tests/parallel/test_voronoi_sharded_equivalence.py: del4 hyperdiffusion,
scripts/bench/bench_mpas_spmd_scaling.py-132-    # energy-conserving PV flux, SSP-RK3, global mass fixer.
scripts/bench/bench_mpas_spmd_scaling.py-133-    cfg = MPASPrimitiveEquationConfig(
scripts/bench/bench_mpas_spmd_scaling.py-134-        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
scripts/bench/bench_mpas_spmd_scaling.py-135-        pv_scheme="energy", time_integrator="ssp_rk3",
scripts/bench/bench_mpas_spmd_scaling.py-136-    )
scripts/bench/bench_mpas_spmd_scaling.py-137-    dev_config = create_voronoi_device_mesh(
scripts/bench/bench_mpas_spmd_scaling.py-138-        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
--
scripts/bench/bench_mpas_spmd_scaling.py-408-            s = step(s, dt, physics_fn=physics_fn)
scripts/bench/bench_mpas_spmd_scaling.py-409-        else:
scripts/bench/bench_mpas_spmd_scaling.py-410-            s = step(s, dt)
scripts/bench/bench_mpas_spmd_scaling.py-411-        _block(s)
scripts/bench/bench_mpas_spmd_scaling.py-412-        per_step_ms.append((time.perf_counter() - t0) * 1e3)
scripts/bench/bench_mpas_spmd_scaling.py-413-
scripts/bench/bench_mpas_spmd_scaling.py-414-    if jax.process_count() > 1:
scripts/bench/bench_mpas_spmd_scaling.py-415-        from jax.experimental import multihost_utils
scripts/bench/bench_mpas_spmd_scaling.py-416-        multihost_utils.sync_global_devices("mpas_spmd_bench_end")
scripts/bench/bench_mpas_spmd_scaling.py-417-
scripts/bench/bench_mpas_spmd_scaling.py:418:    # HLO collective-permute census (#1113 ask 2): a STATIC compile property of
scripts/bench/bench_mpas_spmd_scaling.py-419-    # the sharded step — the ppermute ROUND count that decomposes multi-node
scripts/bench/bench_mpas_spmd_scaling.py-420-    # overhead (overhead ~= CPs/step * ~0.11 ms launch floor). The cube benches
scripts/bench/bench_mpas_spmd_scaling.py:421:    # record this; the MPAS row did not, forcing an out-of-band census. Counted
scripts/bench/bench_mpas_spmd_scaling.py:422:    # AFTER the timed loop so the census compile can't perturb per_step_ms[0]'s
scripts/bench/bench_mpas_spmd_scaling.py-423-    # compile timing (the executable is already cached — this re-lower/compile
scripts/bench/bench_mpas_spmd_scaling.py-424-    # is a cache hit; the count is data-independent, static in the partition).
scripts/bench/bench_mpas_spmd_scaling.py-425-    # Best-effort (None if compilation is unsupported); the serial n=1 leg has
scripts/bench/bench_mpas_spmd_scaling.py-426-    # no ppermute halo -> 0.
scripts/bench/bench_mpas_spmd_scaling.py-427-    if physics_fn is not None:
scripts/bench/bench_mpas_spmd_scaling.py:428:        _census_fn = lambda st: step(st, dt, physics_fn=physics_fn)  # noqa: E731
scripts/bench/bench_mpas_spmd_scaling.py-429-    else:
scripts/bench/bench_mpas_spmd_scaling.py:430:        _census_fn = lambda st: step(st, dt)  # noqa: E731
scripts/bench/bench_mpas_spmd_scaling.py:431:    # ONE compile → full per-family census; the CP scalar (the #1113 round-count
scripts/bench/bench_mpas_spmd_scaling.py-432-    # wall) is the collective_permute member, so no second compile for it.
scripts/bench/bench_mpas_spmd_scaling.py:433:    hlo_census = hlo_collective_census(_census_fn, s)
scripts/bench/bench_mpas_spmd_scaling.py:434:    hlo_cp = hlo_census["collective_permute"] if hlo_census else None
scripts/bench/bench_mpas_spmd_scaling.py-435-
scripts/bench/bench_mpas_spmd_scaling.py-436-    # --- Correctness gates (before any timing is reported) -----------------
scripts/bench/bench_mpas_spmd_scaling.py-437-    if args.parity_gate or args.check_conservation:
scripts/bench/bench_mpas_spmd_scaling.py-438-        final_global = (gather_voronoi_state_spmd(s, dev_config)
scripts/bench/bench_mpas_spmd_scaling.py-439-                        if dev_config.n_devices > 1 else s)
scripts/bench/bench_mpas_spmd_scaling.py-440-        prec = "float64" if jax.config.jax_enable_x64 else "float32"
scripts/bench/bench_mpas_spmd_scaling.py-441-        rank0 = jax.process_index() == 0
scripts/bench/bench_mpas_spmd_scaling.py-442-        if args.check_conservation:
scripts/bench/bench_mpas_spmd_scaling.py-443-            mass_after = _global_dry_mass(final_global, mesh)
scripts/bench/bench_mpas_spmd_scaling.py-444-            tol = (args.mass_rtol if args.mass_rtol is not None
--
scripts/bench/bench_mpas_spmd_scaling.py-513-        multicontroller=bool(args.multicontroller),
scripts/bench/bench_mpas_spmd_scaling.py-514-        compile_ms=round(per_step_ms[0], 1),
scripts/bench/bench_mpas_spmd_scaling.py-515-        steady_median_ms=round(med, 2),
scripts/bench/bench_mpas_spmd_scaling.py-516-        steady_min_ms=round(float(np.min(steady)), 2),
scripts/bench/bench_mpas_spmd_scaling.py-517-        per_step_ms=[round(x, 1) for x in per_step_ms],
scripts/bench/bench_mpas_spmd_scaling.py-518-        cells=int(mesh.nCells) * args.nlev,
scripts/bench/bench_mpas_spmd_scaling.py-519-        # ppermute round count/step (static compile property; #1113) — the
scripts/bench/bench_mpas_spmd_scaling.py-520-        # multi-node ceiling is this count x the ~0.11 ms launch floor, so it
scripts/bench/bench_mpas_spmd_scaling.py-521-        # belongs on every row like the cube benches.
scripts/bench/bench_mpas_spmd_scaling.py-522-        hlo_collective_permutes=hlo_cp,
scripts/bench/bench_mpas_spmd_scaling.py:523:        # full per-family census (permute + all-reduce + all-gather + ...) on
scripts/bench/bench_mpas_spmd_scaling.py-524-        # the SAME compile: exposes any reduction the ico step introduces.
scripts/bench/bench_mpas_spmd_scaling.py:525:        hlo_collectives=hlo_census,
scripts/bench/bench_mpas_spmd_scaling.py-526-    )
scripts/bench/bench_mpas_spmd_scaling.py-527-    # Flat aggregator-compatible identity + metric fields (see the latlon
scripts/bench/bench_mpas_spmd_scaling.py-528-    # twin): resolution = subdivision level, matching run_cpu_mpi_scaling's
scripts/bench/bench_mpas_spmd_scaling.py-529-    # icosahedral convention so both lanes land on the same plot curves.
scripts/bench/bench_mpas_spmd_scaling.py-530-    rec.update(
scripts/bench/bench_mpas_spmd_scaling.py-531-        grid_type="icosahedral",
scripts/bench/bench_mpas_spmd_scaling.py-532-        resolution=args.subdivision,
scripts/bench/bench_mpas_spmd_scaling.py-533-        n_levels=args.nlev,
scripts/bench/bench_mpas_spmd_scaling.py-534-        mode="strong",  # this bench fixes the mesh and sweeps devices
scripts/bench/bench_mpas_spmd_scaling.py-535-        precision="float64" if jax.config.jax_enable_x64 else "float32",
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-362-    # honest nulls (its flat throughput twins are nulled after assembly).
scripts/bench/bench_atm_latlon_spmd_scaling.py-363-    _measured_med = med if valid else None
scripts/bench/bench_atm_latlon_spmd_scaling.py-364-    # Honest per-device geometry residency (from the real band-grid shapes):
scripts/bench/bench_atm_latlon_spmd_scaling.py-365-    # the default lane replicates all-band stacks; the segment lane shards.
scripts/bench/bench_atm_latlon_spmd_scaling.py-366-    geom_bytes = (atm_latlon_geometry_bytes(model.grid, nd) if nd > 1
scripts/bench/bench_atm_latlon_spmd_scaling.py-367-                  else None)
scripts/bench/bench_atm_latlon_spmd_scaling.py-368-
scripts/bench/bench_atm_latlon_spmd_scaling.py-369-    # Communication accounting (audit item 4) + calibrated T_bound (item 8).
scripts/bench/bench_atm_latlon_spmd_scaling.py-370-    # nd=1: zero inter-device traffic is a FACT (recorded as 0), so the
scripts/bench/bench_atm_latlon_spmd_scaling.py-371-    # bound is complete and trivially equals the measured compute.  nd>1:
scripts/bench/bench_atm_latlon_spmd_scaling.py:372:    # there is no analytic halo-message census for the atm latlon step yet
scripts/bench/bench_atm_latlon_spmd_scaling.py-373-    # (the ocean twin derives one from its barotropic solver) — the comm
scripts/bench/bench_atm_latlon_spmd_scaling.py-374-    # ingredients are recorded null with this reason and the bound is
scripts/bench/bench_atm_latlon_spmd_scaling.py-375-    # emitted incomplete rather than fabricated.
scripts/bench/bench_atm_latlon_spmd_scaling.py-376-    if nd <= 1:
scripts/bench/bench_atm_latlon_spmd_scaling.py-377-        _msgs, _bytes_msg, _nred = 0, 0, 0
scripts/bench/bench_atm_latlon_spmd_scaling.py-378-        _bytes_lower = False   # zero traffic is exact, not an undercount
scripts/bench/bench_atm_latlon_spmd_scaling.py-379-        _comm_note = "single device: no inter-device halo/reduction traffic"
scripts/bench/bench_atm_latlon_spmd_scaling.py-380-    else:
scripts/bench/bench_atm_latlon_spmd_scaling.py-381-        _msgs, _bytes_msg, _nred = None, None, None
scripts/bench/bench_atm_latlon_spmd_scaling.py-382-        _bytes_lower = None
scripts/bench/bench_atm_latlon_spmd_scaling.py:383:        _comm_note = ("no analytic halo-message census for the atm latlon "
scripts/bench/bench_atm_latlon_spmd_scaling.py-384-                      "step yet (audit item 4 follow-up) — comm fields null, "
scripts/bench/bench_atm_latlon_spmd_scaling.py-385-                      "not fabricated")
scripts/bench/bench_atm_latlon_spmd_scaling.py-386-    comm_rec = comm_accounting(
scripts/bench/bench_atm_latlon_spmd_scaling.py-387-        halo_messages_per_step=_msgs,
scripts/bench/bench_atm_latlon_spmd_scaling.py-388-        bytes_per_message=_bytes_msg,
scripts/bench/bench_atm_latlon_spmd_scaling.py-389-        full_state_gathers_per_step=0,   # fused scan/segment: no per-step gather
scripts/bench/bench_atm_latlon_spmd_scaling.py-390-        scope_note=_comm_note,
scripts/bench/bench_atm_latlon_spmd_scaling.py-391-        bytes_are_lower_bound=_bytes_lower,
scripts/bench/bench_atm_latlon_spmd_scaling.py-392-    )
scripts/bench/bench_atm_latlon_spmd_scaling.py-393-    bound_rec = calibrated_bound(
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-256-                        "dominates this step above its roofline. Lowering "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-257-                        "it trades solver convergence for sync points — "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-258-                        "check zero_forcing_probe_residual in the output "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-259-                        "before believing any speedup.")
scripts/bench/bench_ocean_latlon_spmd_scaling.py-260-    p.add_argument("--pcg-variant", choices=["standard", "single_reduce"],
scripts/bench/bench_ocean_latlon_spmd_scaling.py-261-                   default="standard",
scripts/bench/bench_ocean_latlon_spmd_scaling.py-262-                   help="Fixed-M PCG recurrence for the implicit_cn "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-263-                        "barotropic solve: standard = 2 dependent reduction "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-264-                        "batches/iter; single_reduce = Chronopoulos-Gear, "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-265-                        "ONE batched reduction/iter (halves the per-step "
scripts/bench/bench_ocean_latlon_spmd_scaling.py:266:                        "reduction count the census reports).")
scripts/bench/bench_ocean_latlon_spmd_scaling.py-267-    p.add_argument("--dt", type=float, default=600.0)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-268-    p.add_argument("--single-dev-fused-ms", type=float, default=None,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-269-                   help="fused_step_ms of the nd=1 row at the SAME per-device "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-270-                        "size (the compute ingredient of the calibrated "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-271-                        "T_bound, audit item 8). Omitted at nd>1 -> the bound "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-272-                        "is emitted null + flagged incomplete (never "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-273-                        "fabricated); nd=1 rows use their own measurement.")
scripts/bench/bench_ocean_latlon_spmd_scaling.py-274-    p.add_argument("--comm-latency-us", type=float, default=None,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-275-                   help="MEASURED per-message latency [us] of THIS machine's "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-276-                        "fabric (ping-pong microbenchmark). Default: the "
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-677-                barotropic=_probe_cfg.barotropic._replace(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-678-                    barotropic_implicit_force_pcg=True))
scripts/bench/bench_ocean_latlon_spmd_scaling.py-679-        _probe_out = barotropic_implicit_latlon_cgrid(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-680-            final_global, args.dt, model.grid, model.z_coord, _probe_cfg,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-681-            return_residual=True)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-682-        zero_forcing_probe_residual = float(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-683-            jax.block_until_ready(_probe_out[2]))
scripts/bench/bench_ocean_latlon_spmd_scaling.py-684-        zero_forcing_probe_measured = True
scripts/bench/bench_ocean_latlon_spmd_scaling.py-685-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-686-    # --- Communication accounting (audit item 4) + calibrated bound (8) ----
scripts/bench/bench_ocean_latlon_spmd_scaling.py:687:    # Analytic INTER-DEVICE census, barotropic-solver scope ONLY (the
scripts/bench/bench_ocean_latlon_spmd_scaling.py:688:    # baroclinic 3-D pads are not counted -> bytes/comm are a LOWER census,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-689-    # flagged machine-readably via halo_bytes_is_lower_bound; T_bound is a
scripts/bench/bench_ocean_latlon_spmd_scaling.py-690-    # heuristic model, see calibrated_bound's docstring).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-691-    # implicit_cn PCG: each Helmholtz apply pads eta N+S (gradient stencil)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-692-    # + the v-face flux row (divergence) ~= 2 exchanges/apply, applied
scripts/bench/bench_ocean_latlon_spmd_scaling.py-693-    # iters + 1 times (incl. the initial residual); reductions = the dot
scripts/bench/bench_ocean_latlon_spmd_scaling.py-694-    # batches (2/iter standard, 1/iter single_reduce) + the initial batch
scripts/bench/bench_ocean_latlon_spmd_scaling.py-695-    # + the mass-projection psum + the eta-floor clamp psum.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-696-    # The split-explicit substep-pad ESTIMATOR is only meaningful for
scripts/bench/bench_ocean_latlon_spmd_scaling.py-697-    # explicit_substep — implicit_cn has no substep loop, so publishing its
scripts/bench/bench_ocean_latlon_spmd_scaling.py-698-    # numbers on implicit rows would mislabel their traffic (codex batch4).
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-713-                      "traffic")
scripts/bench/bench_ocean_latlon_spmd_scaling.py-714-    elif _pcg_fixed_path:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-715-        _msgs = 2 * (solver_iters + 1)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-716-        _per_iter = (1 if _baro_cfg.barotropic_implicit_pcg_variant
scripts/bench/bench_ocean_latlon_spmd_scaling.py-717-                     == "single_reduce" else 2)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-718-        _n_reductions = _per_iter * solver_iters + 3
scripts/bench/bench_ocean_latlon_spmd_scaling.py-719-        _bytes_lower = True
scripts/bench/bench_ocean_latlon_spmd_scaling.py-720-        _comm_note = ("analytic, barotropic implicit-CN PCG scope only; "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-721-                      "2-D eta row slabs (nlev=1), one row per direction "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-722-                      "(rows_per_message=2); baroclinic 3-D pads NOT "
scripts/bench/bench_ocean_latlon_spmd_scaling.py:723:                      "counted — bytes are a lower census")
scripts/bench/bench_ocean_latlon_spmd_scaling.py-724-    elif args.wide_halo:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-725-        _msgs = None
scripts/bench/bench_ocean_latlon_spmd_scaling.py-726-        _bytes_lower = None
scripts/bench/bench_ocean_latlon_spmd_scaling.py-727-        _comm_note = ("wide-halo arm: per-chunk exchange count depends on "
scripts/bench/bench_ocean_latlon_spmd_scaling.py:728:                      "the auto chunk size; census in "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-729-                      "extra.barotropic_halo_messages — bytes not derived "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-730-                      "(not fabricated)")
scripts/bench/bench_ocean_latlon_spmd_scaling.py-731-    else:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-732-        # explicit_substep standard path: the analytic per-substep pad
scripts/bench/bench_ocean_latlon_spmd_scaling.py:733:        # census (also recorded verbatim in extra.barotropic_halo_messages).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-734-        _msgs = int(_halo_est["standard_messages"])
scripts/bench/bench_ocean_latlon_spmd_scaling.py-735-        _bytes_lower = True
scripts/bench/bench_ocean_latlon_spmd_scaling.py-736-        _comm_note = ("analytic, explicit-substep barotropic scope only "
scripts/bench/bench_ocean_latlon_spmd_scaling.py:737:                      "(2-D slabs, one row per direction); reduction census "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-738-                      "not derived for this path; baroclinic 3-D pads NOT "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-739-                      "counted")
scripts/bench/bench_ocean_latlon_spmd_scaling.py-740-    comm_rec = comm_accounting(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-741-        halo_messages_per_step=_msgs,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-742-        n_lon=args.n_lon, nlev=1, dtype_bytes=_dtype_bytes,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-743-        rows_per_message=2,   # a pad exchange moves one row N + one row S
scripts/bench/bench_ocean_latlon_spmd_scaling.py-744-        full_state_gathers_per_step=0,   # fused scan: no per-step gather
scripts/bench/bench_ocean_latlon_spmd_scaling.py-745-        scope_note=_comm_note,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-746-        bytes_are_lower_bound=_bytes_lower,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-747-    )
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-849-            "fused_halo": os.environ.get(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-850-                "LEGOESM_LATLON_SPMD_FUSED_HALO", "1") != "0",
scripts/bench/bench_ocean_latlon_spmd_scaling.py-851-            # Route-B transport facts (socket-fallback flag): a
scripts/bench/bench_ocean_latlon_spmd_scaling.py-852-            # multi-node row without an NCCL net plugin is
scripts/bench/bench_ocean_latlon_spmd_scaling.py-853-            # falsifiable from the record alone.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-854-            "nccl": (_nccl_report if args.multicontroller
scripts/bench/bench_ocean_latlon_spmd_scaling.py-855-                     else None),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-856-            "parity_gate": bool(args.parity_gate),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-857-            "check_conservation": bool(args.check_conservation),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-858-            "cells_per_device": (n_lat // nd) * args.n_lon * args.nlev,
scripts/bench/bench_ocean_latlon_spmd_scaling.py:859:            # Analytic split-explicit substep-pad census (the wide-halo
scripts/bench/bench_ocean_latlon_spmd_scaling.py-860-            # audit item's halo-count metric): standard per-substep pads
scripts/bench/bench_ocean_latlon_spmd_scaling.py-861-            # vs the wide path's fused per-chunk exchanges.  None on
scripts/bench/bench_ocean_latlon_spmd_scaling.py-862-            # implicit_cn rows — the estimator describes a substep loop
scripts/bench/bench_ocean_latlon_spmd_scaling.py-863-            # implicit-CN does not run (codex batch4); PCG traffic is the
scripts/bench/bench_ocean_latlon_spmd_scaling.py:864:            # flat halo_messages_per_step census.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-865-            "barotropic_halo_messages": _halo_est,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-866-            # Solver-iteration facts next to the probe (audit item 6).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-867-            "solver_iters": solver_iters,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-868-            "solver_iters_mode": solver_iters_mode,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-869-            "zero_forcing_probe_measured": zero_forcing_probe_measured,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-870-        },
scripts/bench/bench_ocean_latlon_spmd_scaling.py-871-    ))
scripts/bench/bench_ocean_latlon_spmd_scaling.py-872-    # Multi-controller: every process times the same program; process 0 owns
scripts/bench/bench_ocean_latlon_spmd_scaling.py-873-    # the JSONL + stdout (others would duplicate/corrupt the append).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-874-    if jax.process_index() == 0:
--
tests/bench/test_bench_voronoi_partition_methods.py-153-
tests/bench/test_bench_voronoi_partition_methods.py-154-def test_schedule_cost_row_reports_rounds_against_their_lower_bound():
tests/bench/test_bench_voronoi_partition_methods.py-155-    """``schedule_cost_row`` returns the REAL schedule depth and the bound it
tests/bench/test_bench_voronoi_partition_methods.py-156-    must be read against."""
tests/bench/test_bench_voronoi_partition_methods.py-157-    mesh = _mesh()
tests/bench/test_bench_voronoi_partition_methods.py-158-    sc = mod.schedule_cost_row(mesh, "geometric", 2)
tests/bench/test_bench_voronoi_partition_methods.py-159-    # max_degree is the graph's own lower bound on a proper edge colouring,
tests/bench/test_bench_voronoi_partition_methods.py-160-    # so a schedule can never beat it.  -1 is the scorer's "not reported"
tests/bench/test_bench_voronoi_partition_methods.py-161-    # sentinel and would make the gap meaningless.
tests/bench/test_bench_voronoi_partition_methods.py-162-    assert sc["max_degree"] >= 1
tests/bench/test_bench_voronoi_partition_methods.py:163:    assert sc["n_rounds"] >= sc["max_degree"]
tests/bench/test_bench_voronoi_partition_methods.py:164:    assert sc["coloring_gap"] == sc["n_rounds"] - sc["max_degree"]
tests/bench/test_bench_voronoi_partition_methods.py:165:    assert sc["n_rounds_greedy"] >= sc["n_rounds"]
tests/bench/test_bench_voronoi_partition_methods.py-166-    assert sc["score_seconds"] >= 0.0
tests/bench/test_bench_voronoi_partition_methods.py-167-
tests/bench/test_bench_voronoi_partition_methods.py-168-
tests/bench/test_bench_voronoi_partition_methods.py-169-@pytest.mark.parametrize(
tests/bench/test_bench_voronoi_partition_methods.py:170:    "n_rounds, max_degree, gap, hmin, hmax, proven",
tests/bench/test_bench_voronoi_partition_methods.py-171-    [
tests/bench/test_bench_voronoi_partition_methods.py-172-        # Provably optimal: no proper edge colouring beats max_degree.
tests/bench/test_bench_voronoi_partition_methods.py-173-        (12, 12, 0, 0, 0, True),
tests/bench/test_bench_voronoi_partition_methods.py-174-        # Vizing allows the true optimum to BE max_degree+1, so a gap of 1
tests/bench/test_bench_voronoi_partition_methods.py-175-        # spans [0, 1] — inconclusive.  This is the case that a naive
tests/bench/test_bench_voronoi_partition_methods.py-176-        # "gap > 0 means recolour" rule would over-claim.
tests/bench/test_bench_voronoi_partition_methods.py-177-        (13, 12, 1, 0, 1, False),
tests/bench/test_bench_voronoi_partition_methods.py-178-        # Guaranteed to remove at least gap-1 = 3, at most gap = 4.
tests/bench/test_bench_voronoi_partition_methods.py-179-        (14, 10, 4, 3, 4, False),
tests/bench/test_bench_voronoi_partition_methods.py-180-    ])
tests/bench/test_bench_voronoi_partition_methods.py-181-def test_coloring_gap_and_headroom_are_derived_not_assumed(
tests/bench/test_bench_voronoi_partition_methods.py:182:        monkeypatch, n_rounds, max_degree, gap, hmin, hmax, proven):
tests/bench/test_bench_voronoi_partition_methods.py-183-    """Gap and the Vizing-bounded headroom must be COMPUTED, not assumed.
tests/bench/test_bench_voronoi_partition_methods.py-184-
tests/bench/test_bench_voronoi_partition_methods.py-185-    Non-vacuity, the hard way: on every mesh small enough to test quickly the
tests/bench/test_bench_voronoi_partition_methods.py-186-    real gap is 0 (measured L2/L3/L4 x {geometric,sfc} x nd 2-16, and s6
tests/bench/test_bench_voronoi_partition_methods.py-187-    lloyd=0 at np8/np16 — the colourer lands exactly on ``max_degree`` every
tests/bench/test_bench_voronoi_partition_methods.py-188-    time), so a real-mesh assertion cannot tell a correct subtraction from a
tests/bench/test_bench_voronoi_partition_methods.py-189-    hardcoded ``0``; that exact mutation passed the first version of this
tests/bench/test_bench_voronoi_partition_methods.py-190-    test.  Stubbing the production scorer with KNOWN values is what makes
tests/bench/test_bench_voronoi_partition_methods.py-191-    the assertion able to fail.
tests/bench/test_bench_voronoi_partition_methods.py-192-
tests/bench/test_bench_voronoi_partition_methods.py-193-    The ``gap == 1`` row is the one that matters: the schedule is a proper
tests/bench/test_bench_voronoi_partition_methods.py-194-    EDGE colouring and ``max_degree`` is that graph's max vertex degree, so
tests/bench/test_bench_voronoi_partition_methods.py-195-    Vizing gives ``Delta <= chi' <= Delta + 1``.  A gap of 1 is therefore
tests/bench/test_bench_voronoi_partition_methods.py-196-    indistinguishable from optimal (Class 2), and claiming recolouring
tests/bench/test_bench_voronoi_partition_methods.py-197-    headroom there would be an over-claim.
tests/bench/test_bench_voronoi_partition_methods.py-198-    """
tests/bench/test_bench_voronoi_partition_methods.py-199-    stub = {
tests/bench/test_bench_voronoi_partition_methods.py:200:        "n_rounds": n_rounds, "max_degree": max_degree,
tests/bench/test_bench_voronoi_partition_methods.py:201:        "n_rounds_greedy": n_rounds + 1, "coloring_method": "stub",
tests/bench/test_bench_voronoi_partition_methods.py-202-        "resolved_method": "geometric", "halo_depth": 3,
tests/bench/test_bench_voronoi_partition_methods.py-203-        "cells_per_device": 99_999, "production_strategy": "ppermute",
tests/bench/test_bench_voronoi_partition_methods.py:204:        "reorder_target": 8, "already_reordered": False,
tests/bench/test_bench_voronoi_partition_methods.py-205-    }
tests/bench/test_bench_voronoi_partition_methods.py-206-    import legoesm.parallel.sharded_dynamics as sd
tests/bench/test_bench_voronoi_partition_methods.py-207-    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
tests/bench/test_bench_voronoi_partition_methods.py-208-
tests/bench/test_bench_voronoi_partition_methods.py-209-    sc = mod.schedule_cost_row(object(), "geometric", 8)
tests/bench/test_bench_voronoi_partition_methods.py-210-    assert sc["coloring_gap"] == gap
tests/bench/test_bench_voronoi_partition_methods.py-211-    # The headroom is an INTERVAL: Vizing pins the optimum to
tests/bench/test_bench_voronoi_partition_methods.py-212-    # {Delta, Delta+1}, so gap-1 is guaranteed and gap is the best case.
tests/bench/test_bench_voronoi_partition_methods.py-213-    assert sc["coloring_headroom_rounds_min"] == hmin
tests/bench/test_bench_voronoi_partition_methods.py-214-    assert sc["coloring_headroom_rounds_max"] == hmax
--
tests/bench/test_bench_voronoi_partition_methods.py-217-
tests/bench/test_bench_voronoi_partition_methods.py-218-def test_schedule_rounds_never_beat_the_vizing_floor_on_a_real_mesh():
tests/bench/test_bench_voronoi_partition_methods.py-219-    """Real-mesh sanity on the bound itself: a proper edge colouring can
tests/bench/test_bench_voronoi_partition_methods.py-220-    never use fewer rounds than the graph's max degree, and the multi-start
tests/bench/test_bench_voronoi_partition_methods.py-221-    search should not overshoot Vizing's ``Delta + 1`` either.  If this ever
tests/bench/test_bench_voronoi_partition_methods.py-222-    fires, ``max_degree`` is not the degree of the graph being coloured and
tests/bench/test_bench_voronoi_partition_methods.py-223-    every gap-based conclusion built on it is void."""
tests/bench/test_bench_voronoi_partition_methods.py-224-    for method in ("geometric", "sfc"):
tests/bench/test_bench_voronoi_partition_methods.py-225-        for n_ranks in (2, 4, 8):
tests/bench/test_bench_voronoi_partition_methods.py-226-            sc = mod.schedule_cost_row(_mesh(level=3), method, n_ranks)
tests/bench/test_bench_voronoi_partition_methods.py:227:            assert sc["max_degree"] <= sc["n_rounds"] <= sc["max_degree"] + 1, (
tests/bench/test_bench_voronoi_partition_methods.py:228:                f"{method} np={n_ranks}: rounds={sc['n_rounds']} "
tests/bench/test_bench_voronoi_partition_methods.py-229-                f"max_degree={sc['max_degree']}")
tests/bench/test_bench_voronoi_partition_methods.py-230-
tests/bench/test_bench_voronoi_partition_methods.py-231-
tests/bench/test_bench_voronoi_partition_methods.py-232-def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
tests/bench/test_bench_voronoi_partition_methods.py-233-    """The schedule is scored at the SPMD production halo depth, NOT this
tests/bench/test_bench_voronoi_partition_methods.py-234-    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
tests/bench/test_bench_voronoi_partition_methods.py-235-    a different graph and quietly report the wrong lane's cost."""
tests/bench/test_bench_voronoi_partition_methods.py:236:    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH
tests/bench/test_bench_voronoi_partition_methods.py-237-
tests/bench/test_bench_voronoi_partition_methods.py-238-    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
tests/bench/test_bench_voronoi_partition_methods.py:239:    assert sc["schedule_halo_depth"] == SPMD_HALO_DEPTH
tests/bench/test_bench_voronoi_partition_methods.py-240-
tests/bench/test_bench_voronoi_partition_methods.py-241-
tests/bench/test_bench_voronoi_partition_methods.py-242-def test_schedule_cost_matches_the_production_scorer_exactly():
tests/bench/test_bench_voronoi_partition_methods.py-243-    """Lock: the wrapper reports what the production scorer returns — it is
tests/bench/test_bench_voronoi_partition_methods.py-244-    a passthrough, not a re-derivation (the whole point: a 1-ring lookalike
tests/bench/test_bench_voronoi_partition_methods.py-245-    reports 8 rounds where the real depth-3 graph reports 12-14)."""
tests/bench/test_bench_voronoi_partition_methods.py-246-    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
tests/bench/test_bench_voronoi_partition_methods.py-247-
tests/bench/test_bench_voronoi_partition_methods.py-248-    mesh = _mesh()
tests/bench/test_bench_voronoi_partition_methods.py-249-    ref = spmd_schedule_cost(mesh, 2, method="sfc")
tests/bench/test_bench_voronoi_partition_methods.py-250-    sc = mod.schedule_cost_row(mesh, "sfc", 2)
tests/bench/test_bench_voronoi_partition_methods.py:251:    for key in ("n_rounds", "max_degree", "n_rounds_greedy",
tests/bench/test_bench_voronoi_partition_methods.py-252-                "coloring_method", "resolved_method", "cells_per_device",
tests/bench/test_bench_voronoi_partition_methods.py-253-                "production_strategy"):
tests/bench/test_bench_voronoi_partition_methods.py-254-        assert sc[key] == ref[key], key
tests/bench/test_bench_voronoi_partition_methods.py-255-
tests/bench/test_bench_voronoi_partition_methods.py-256-
tests/bench/test_bench_voronoi_partition_methods.py-257-def test_schedule_cost_flag_is_opt_in_and_lands_in_the_json(
tests/bench/test_bench_voronoi_partition_methods.py-258-        tmp_path, monkeypatch):
tests/bench/test_bench_voronoi_partition_methods.py-259-    """Off by default (it is the expensive layer); on, every scored row
tests/bench/test_bench_voronoi_partition_methods.py-260-    carries the schedule block and the run records that it ran."""
tests/bench/test_bench_voronoi_partition_methods.py-261-    out = tmp_path / "off.json"
--
tests/bench/test_bench_voronoi_partition_methods.py-267-    assert "schedule" not in payload["rows"][0]
tests/bench/test_bench_voronoi_partition_methods.py-268-    assert payload["metadata"]["extra"]["schedule_cost"] is False
tests/bench/test_bench_voronoi_partition_methods.py-269-
tests/bench/test_bench_voronoi_partition_methods.py-270-    out2 = tmp_path / "on.json"
tests/bench/test_bench_voronoi_partition_methods.py-271-    monkeypatch.setattr(sys, "argv", [
tests/bench/test_bench_voronoi_partition_methods.py-272-        "bench", "--subdivision", "2", "--rank-counts", "2",
tests/bench/test_bench_voronoi_partition_methods.py-273-        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
tests/bench/test_bench_voronoi_partition_methods.py-274-    assert mod.main() == 0
tests/bench/test_bench_voronoi_partition_methods.py-275-    payload2 = json.loads(out2.read_text())
tests/bench/test_bench_voronoi_partition_methods.py-276-    row = payload2["rows"][0]
tests/bench/test_bench_voronoi_partition_methods.py:277:    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
tests/bench/test_bench_voronoi_partition_methods.py-278-    assert payload2["metadata"]["extra"]["schedule_cost"] is True
tests/bench/test_bench_voronoi_partition_methods.py-279-    # Scorer provenance per row: a copied row must show it scored a mesh
tests/bench/test_bench_voronoi_partition_methods.py-280-    # partitioned for THIS device count, not one reordered for another.
tests/bench/test_bench_voronoi_partition_methods.py:281:    assert row["schedule"]["reorder_target"] == row["n_ranks"]
tests/bench/test_bench_voronoi_partition_methods.py:282:    assert row["schedule"]["already_reordered"] is False
tests/bench/test_bench_voronoi_partition_methods.py-283-
tests/bench/test_bench_voronoi_partition_methods.py-284-
tests/bench/test_bench_voronoi_partition_methods.py-285-def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
tests/bench/test_bench_voronoi_partition_methods.py-286-    """``--lloyd`` must actually select the mesh, not just be recorded.
tests/bench/test_bench_voronoi_partition_methods.py-287-
tests/bench/test_bench_voronoi_partition_methods.py-288-    Non-vacuity: asserting only the recorded default (50) passes even if the
tests/bench/test_bench_voronoi_partition_methods.py-289-    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
tests/bench/test_bench_voronoi_partition_methods.py-290-    that.  This spies on the builder, so dropping ``lloyd_iterations=
tests/bench/test_bench_voronoi_partition_methods.py-291-    args.lloyd`` fails here, and it checks a NON-default value so the
tests/bench/test_bench_voronoi_partition_methods.py-292-    assertion cannot be satisfied by the default.
--
tests/bench/test_bench_voronoi_partition_methods.py-318-        tmp_path, monkeypatch):
tests/bench/test_bench_voronoi_partition_methods.py-319-    """The instrument check must FAIL on a wrong expectation and on a pair
tests/bench/test_bench_voronoi_partition_methods.py-320-    that was never scored — a gate that can only pass is not a gate."""
tests/bench/test_bench_voronoi_partition_methods.py-321-    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
tests/bench/test_bench_voronoi_partition_methods.py-322-            "--methods", "geometric", "--schedule-cost"]
tests/bench/test_bench_voronoi_partition_methods.py-323-
tests/bench/test_bench_voronoi_partition_methods.py-324-    # Truth first: read what this configuration really scores.
tests/bench/test_bench_voronoi_partition_methods.py-325-    out = tmp_path / "truth.json"
tests/bench/test_bench_voronoi_partition_methods.py-326-    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
tests/bench/test_bench_voronoi_partition_methods.py-327-    assert mod.main() == 0
tests/bench/test_bench_voronoi_partition_methods.py:328:    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
tests/bench/test_bench_voronoi_partition_methods.py-329-
tests/bench/test_bench_voronoi_partition_methods.py-330-    # Matching expectation -> pass, and the check is recorded.
tests/bench/test_bench_voronoi_partition_methods.py-331-    ok = tmp_path / "ok.json"
tests/bench/test_bench_voronoi_partition_methods.py-332-    monkeypatch.setattr(sys, "argv", base + [
tests/bench/test_bench_voronoi_partition_methods.py-333-        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
tests/bench/test_bench_voronoi_partition_methods.py-334-    assert mod.main() == 0
tests/bench/test_bench_voronoi_partition_methods.py-335-    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
tests/bench/test_bench_voronoi_partition_methods.py-336-
tests/bench/test_bench_voronoi_partition_methods.py-337-    # Wrong expectation -> non-zero exit.
tests/bench/test_bench_voronoi_partition_methods.py-338-    bad = tmp_path / "bad.json"
--
tests/bench/test_bench_cube_tiled_step_scaling.py-1-"""Direct tests for the sub-face tiled cube step scaling lane (item 5).
tests/bench/test_bench_cube_tiled_step_scaling.py-2-
tests/bench/test_bench_cube_tiled_step_scaling.py-3-The END-TO-END lane needs >=24 devices and a tiled-stage compile that is
tests/bench/test_bench_cube_tiled_step_scaling.py-4-cluster-scale (the tiled module's own note: not laptop-benchable; the
tests/bench/test_bench_cube_tiled_step_scaling.py-5-pre-existing adapter parity gate itself runs ~10 min on a laptop CPU) — so
tests/bench/test_bench_cube_tiled_step_scaling.py:6:these tests lock the fast layers: CLI rejections, the anti-fake HLO census
tests/bench/test_bench_cube_tiled_step_scaling.py-7-logic, and the parity-tolerance source; the end-to-end smoke is device-count
tests/bench/test_bench_cube_tiled_step_scaling.py-8-gated for cluster CI.
tests/bench/test_bench_cube_tiled_step_scaling.py-9-"""
tests/bench/test_bench_cube_tiled_step_scaling.py-10-from __future__ import annotations
tests/bench/test_bench_cube_tiled_step_scaling.py-11-
tests/bench/test_bench_cube_tiled_step_scaling.py-12-import importlib.util
tests/bench/test_bench_cube_tiled_step_scaling.py-13-import sys
tests/bench/test_bench_cube_tiled_step_scaling.py-14-from pathlib import Path
tests/bench/test_bench_cube_tiled_step_scaling.py-15-
tests/bench/test_bench_cube_tiled_step_scaling.py-16-import pytest
--
tests/bench/test_bench_cube_tiled_step_scaling.py-47-
tests/bench/test_bench_cube_tiled_step_scaling.py-48-    if len(jax.devices()) >= 24:
tests/bench/test_bench_cube_tiled_step_scaling.py-49-        pytest.skip("host exposes >=24 devices; guard not reachable")
tests/bench/test_bench_cube_tiled_step_scaling.py-50-    monkeypatch.setattr(sys, "argv", ["bench", "--kt", "2",
tests/bench/test_bench_cube_tiled_step_scaling.py-51-                                      "--resolution", "8", "--steps", "2",
tests/bench/test_bench_cube_tiled_step_scaling.py-52-                                      "--warmup", "1"])
tests/bench/test_bench_cube_tiled_step_scaling.py-53-    with pytest.raises(SystemExit, match="need 24 devices"):
tests/bench/test_bench_cube_tiled_step_scaling.py-54-        mod.main()
tests/bench/test_bench_cube_tiled_step_scaling.py-55-
tests/bench/test_bench_cube_tiled_step_scaling.py-56-
tests/bench/test_bench_cube_tiled_step_scaling.py:57:def test_collective_census_helper():
tests/bench/test_bench_cube_tiled_step_scaling.py-58-    hlo = "\n".join([
tests/bench/test_bench_cube_tiled_step_scaling.py-59-        "%x = collective-permute(...)",
tests/bench/test_bench_cube_tiled_step_scaling.py-60-        "%y = collective-permute-start(...)",
tests/bench/test_bench_cube_tiled_step_scaling.py-61-        "%z = collective-permute-done(...)",  # not counted (done)
tests/bench/test_bench_cube_tiled_step_scaling.py-62-        "%w = add(...)",
tests/bench/test_bench_cube_tiled_step_scaling.py-63-    ])
tests/bench/test_bench_cube_tiled_step_scaling.py-64-    assert mod._count_collective_permutes(hlo) == 2
tests/bench/test_bench_cube_tiled_step_scaling.py-65-
tests/bench/test_bench_cube_tiled_step_scaling.py-66-
tests/bench/test_bench_cube_tiled_step_scaling.py-67-def test_parity_tolerances_match_adapter_gate():
--
tests/bench/test_bench_ocean_latlon_spmd_gates.py-95-    assert rec["solver_iters_mode"].startswith("fixed_pcg")
tests/bench/test_bench_ocean_latlon_spmd_gates.py-96-    assert rec["solver_residual"] is None
tests/bench/test_bench_ocean_latlon_spmd_gates.py-97-    assert rec["residual_reason"]
tests/bench/test_bench_ocean_latlon_spmd_gates.py-98-    assert rec["zero_forcing_probe_measured"] is True
tests/bench/test_bench_ocean_latlon_spmd_gates.py-99-    assert rec["zero_forcing_probe_residual"] is not None
tests/bench/test_bench_ocean_latlon_spmd_gates.py-100-    assert rec["zero_forcing_probe_residual"] < 1e-6
tests/bench/test_bench_ocean_latlon_spmd_gates.py-101-    assert rec["metadata"]["solver_residual"] is None
tests/bench/test_bench_ocean_latlon_spmd_gates.py-102-    assert (rec["metadata"]["extra"]["zero_forcing_probe_measured"]
tests/bench/test_bench_ocean_latlon_spmd_gates.py-103-            is True)
tests/bench/test_bench_ocean_latlon_spmd_gates.py-104-    # item 4: bytes arithmetic consistent; fused scan never gathers; the
tests/bench/test_bench_ocean_latlon_spmd_gates.py:105:    # partial (barotropic-only) census is flagged machine-readably; the
tests/bench/test_bench_ocean_latlon_spmd_gates.py-106-    # split-explicit substep estimator must NOT be published on an
tests/bench/test_bench_ocean_latlon_spmd_gates.py-107-    # implicit-CN row (codex batch4).
tests/bench/test_bench_ocean_latlon_spmd_gates.py-108-    assert rec["full_state_gathers_per_step"] == 0
tests/bench/test_bench_ocean_latlon_spmd_gates.py-109-    assert rec["halo_messages_per_step"] > 0
tests/bench/test_bench_ocean_latlon_spmd_gates.py-110-    assert (rec["halo_bytes_per_step"]
tests/bench/test_bench_ocean_latlon_spmd_gates.py-111-            == rec["halo_messages_per_step"] * rec["halo_bytes_per_message"])
tests/bench/test_bench_ocean_latlon_spmd_gates.py-112-    assert rec["halo_bytes_is_lower_bound"] is True
tests/bench/test_bench_ocean_latlon_spmd_gates.py-113-    assert rec["metadata"]["extra"]["barotropic_halo_messages"] is None
tests/bench/test_bench_ocean_latlon_spmd_gates.py-114-    # item 8: nd=2 without --single-dev-fused-ms AND with the placeholder
tests/bench/test_bench_ocean_latlon_spmd_gates.py-115-    # (uncalibrated) fabric -> the bound must be NULL + named-incomplete
--
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-19-RECIPE = sys.argv[1]; OUT = sys.argv[2]; DT = 2700.0
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-20-NSTEPS = 2880; ACC0 = 1920  # 90-day screen
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-21-RUN = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ"
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-22-g = read_nemo_mesh_mask(f"{RUN}/mesh_mask.nc", nn_hls=2)
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-23-s = read_nemo_restart(f"{RUN}/DINO_00000320_restart.nc", nn_hls=2)  # geometry donor only
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-24-br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-25-ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else None
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-26-cfg = dataclasses.replace(dino_config_for_recipe(RECIPE),
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-27-    lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)   # bridge-frame lon fix
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-28-st = dino_lat_lon_state(br.geometry, br.z_coord, cfg, land_mask_override=br.land_mask)
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py:29:# --- HARD topo census gate: legoESM wet cells must equal NEMO tmask exactly ---
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-30-import netCDF4 as _nc
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-31-_mm = _nc.Dataset(f"{RUN}/mesh_mask.nc"); _H = 2
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-32-_tm = (np.moveaxis(np.asarray(_mm["tmask"][0]).squeeze(), 0, -1)[_H:-_H, _H:-_H] > 0.5)
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-33-_wet = np.asarray(br.z_coord.is_active) & (np.asarray(st.land_mask.data) > 0.5)[:, :, None]
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-34-if not np.array_equal(_wet, _tm):
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-35-    _d = int(np.sum(_wet != _tm))
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-36-    raise SystemExit(f"TOPO CENSUS FAIL: {_d} cells differ from NEMO tmask — refusing to run")
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py:37:print(f"topo census OK: {int(_tm.sum())} wet cells == NEMO tmask")          # analytic IC == NEMO usrdef_istate
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-38-mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-39-model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-40-forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-41-sf = dino_step_surface_forcing(forcing)   # WIND: tau_x/taum into the dycore external-tau block
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-42-print(f"slope_scheme={mc.gm_redi.slope_scheme} kappa_GM_max={float(jnp.max(jnp.abs(mc.gm_redi.kappa_GM))):.1f}")
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-43-print(f"tau_x[Pa] min/max = {float(jnp.min(sf.tau_x)):.3f}/{float(jnp.max(sf.tau_x)):.3f}")
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-44-
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-45-dyn = jax.jit(lambda st: model.step(st, DT, surface_forcing=sf))  # sf constant (annual tau)
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-46-
scripts/validate/ocean_fidelity/dino_1226/dino_90d_screen.py-47-acc = {k: jnp.zeros_like(getattr(st, k).data) for k in ("T", "S", "eta", "u", "v")}
--
scripts/validate/ocean_fidelity/dino_1226/README.md-7-- `RUN_TRAJ/DINO_00000320_restart.nc` — geometry-donor restart
scripts/validate/ocean_fidelity/dino_1226/README.md-8-- `RUN_1Y/DINO_1y_00010101_00011230_grid_{T,U}.nc` — NEMO year-1 annual mean (stitched)
scripts/validate/ocean_fidelity/dino_1226/README.md-9-- `RUN_5Y/DINO_1y_00050101_00051230_grid_{T,V,U}.nc` — NEMO year-5 mean (stitched)
scripts/validate/ocean_fidelity/dino_1226/README.md-10-- `RUN_STEPDUMP/` — per-step restarts kt=5761-5790 + MY_SRC trddyn/trdtra per-term
scripts/validate/ocean_fidelity/dino_1226/README.md-11-  trend dumps (day ~180) — the instrument for the channel momentum-budget diff
scripts/validate/ocean_fidelity/dino_1226/README.md-12-- NEMO 5.0.2 source: `~/oracle-builds/nemo5/nemo_5.0.2/src/OCE/` (READ THIS, per
scripts/validate/ocean_fidelity/dino_1226/README.md-13-  the #1226 philosophy: transcribe, never diagnose from first principles)
scripts/validate/ocean_fidelity/dino_1226/README.md-14-
scripts/validate/ocean_fidelity/dino_1226/README.md-15-## Screens (protocol: 1-year matched window for screening; 5-year ONLY for final certification)
scripts/validate/ocean_fidelity/dino_1226/README.md-16-
scripts/validate/ocean_fidelity/dino_1226/README.md:17:    # 1-year from-rest, census-gated NEMO topo, year-1 mean accumulated:
scripts/validate/ocean_fidelity/dino_1226/README.md-18-    CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python dino_year_screen.py nemo_dino_kamm_mlf out.npz
scripts/validate/ocean_fidelity/dino_1226/README.md-19-    # 90-day stability probe (day 60-90 mean):
scripts/validate/ocean_fidelity/dino_1226/README.md-20-    CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python dino_90d_screen.py nemo_dino_kamm_mlf out.npz
scripts/validate/ocean_fidelity/dino_1226/README.md-21-
scripts/validate/ocean_fidelity/dino_1226/README.md-22-Both drivers refuse to run unless the legoESM 3-D wet mask equals NEMO tmask
scripts/validate/ocean_fidelity/dino_1226/README.md:23:exactly (topo census gate). ~40 min / ~12 min on one V100S.
scripts/validate/ocean_fidelity/dino_1226/README.md-24-
scripts/validate/ocean_fidelity/dino_1226/README.md-25-## Metrics (year-1 reference values, 2026-07-20, branch aabadc4df)
scripts/validate/ocean_fidelity/dino_1226/README.md-26-
scripts/validate/ocean_fidelity/dino_1226/README.md-27-ACC = median over lon sections of net zonal transport (e3t_1d ladder + e2u,
scripts/validate/ocean_fidelity/dino_1226/README.md-28-model's own wet cells — NO cross-model mask). SSH small-scale ratio = std of
scripts/validate/ocean_fidelity/dino_1226/README.md-29-(field - 5pt boxcar) over wet cells, lego/NEMO. Current: lego ACC 14.5 Sv vs
scripts/validate/ocean_fidelity/dino_1226/README.md-30-NEMO 69.1; SSH ratio 0.51. `dino_year5_compare.py` = the 9-panel climate
scripts/validate/ocean_fidelity/dino_1226/README.md-31-comparison (edit paths at top; written for the year-5 pair).
scripts/validate/ocean_fidelity/dino_1226/README.md-32-
scripts/validate/ocean_fidelity/dino_1226/README.md-33-## Twin instruments (promoted 2026-07-24)
--
scripts/cluster/scaling_derecho/README.md-478-## Ocean + multi-node GPU additions (2026-07)
scripts/cluster/scaling_derecho/README.md-479-
scripts/cluster/scaling_derecho/README.md-480-Alongside the cube pair above, three further jobs + a build script:
scripts/cluster/scaling_derecho/README.md-481-
scripts/cluster/scaling_derecho/README.md-482-| File | What |
scripts/cluster/scaling_derecho/README.md-483-|---|---|
scripts/cluster/scaling_derecho/README.md-484-| `ocean_gpu_scaling.pbs` | OCEAN weak+strong on one GPU node via `scripts/bench/bench_ocean_latlon_spmd_scaling.py` (full lat-lon C-grid step sharded over 1/2/4 A100; fail-fast `--parity-gate` + `--check-conservation` smoke first). Plain GPU env — no mpi4jax. |
scripts/cluster/scaling_derecho/README.md-485-| `ocean_cpu_scaling.pbs` | OCEAN weak+strong CPU-MPI rank ladder (`bench_ocean_mpi_scaling.py`, `legoesm-mpi` env), with a 2-rank parity+conservation smoke. |
scripts/cluster/scaling_derecho/README.md-486-| `gpu_multinode_scaling.pbs` | MULTI-NODE GPU lanes over jax.distributed + NCCL: A = cube `--cs-spmd` (6 GPU / 2 nodes), C = atm lat-lon `--multicontroller` (8 GPU), D = ocean `--multicontroller` (8 GPU); plus the optional route-A CUDA-aware mpi4jax lane (`RUN_ROUTEA=1`, needs the overlay env) and the comm-tuning A/B ladder (`RUN_TUNE=1`, lane T below). |
scripts/cluster/scaling_derecho/README.md-487-| `build_nccl_ofi.sh` | Login-node build of **aws-ofi-nccl** against Derecho's Cray libfabric (no NCCL build dep — the plugin vendors the net-API headers and is dlopen'd by the jax-wheel NCCL). |
scripts/cluster/scaling_derecho/README.md:488:| `diagnosis.pbs` | BOTTLENECK diagnosis via `scripts/bench/run_scaling_diagnosis.py` — per-phase halo bandwidth, reduction latency, roofline, compute/comm overlap, and the **static collective census** (collective-permute + all-reduce + all-gather per step) that the throughput jobs above do NOT capture. Climbs the cube face-shard `1 2 3` ladder (must divide 6) so the message-count-vs-shard curve — the LATENCY-bound anti-scaling signal — is recorded. `qsub -v MODE=census` for counts only. |
scripts/cluster/scaling_derecho/README.md-489-
scripts/cluster/scaling_derecho/README.md-490-NCCL on Slingshot-11 has NO native CXI support: without the plugin the
scripts/cluster/scaling_derecho/README.md-491-multi-node lanes fall back to TCP sockets over `hsn` (correct, 2-3x slower
scripts/cluster/scaling_derecho/README.md-492-comm — loud warning, fine for shakeout). For production numbers:
scripts/cluster/scaling_derecho/README.md-493-
scripts/cluster/scaling_derecho/README.md-494-```bash
scripts/cluster/scaling_derecho/README.md-495-bash scripts/cluster/scaling_derecho/build_nccl_ofi.sh
scripts/cluster/scaling_derecho/README.md-496-qsub -v LEGOESM_NCCL_OFI_LIB=/glade/work/$USER/nccl-ofi/<tag>/lib \
scripts/cluster/scaling_derecho/README.md-497-     scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs
scripts/cluster/scaling_derecho/README.md-498-```
--
scripts/cluster/scaling_derecho/README.md-515-every level. A subdiv-4 smoke with `--parity-gate --check-conservation`
scripts/cluster/scaling_derecho/README.md-516-runs before the timed `ICO_LEVEL` (default L7 = 163842 cells, ~27k
scripts/cluster/scaling_derecho/README.md-517-cells/GPU at np=6) case. 2-process CPU federation gate:
scripts/cluster/scaling_derecho/README.md-518-`tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py`.
scripts/cluster/scaling_derecho/README.md-519-
scripts/cluster/scaling_derecho/README.md-520-
scripts/cluster/scaling_derecho/README.md-521-## 2026-07 lane T: comm-tuning A/B ladder (`RUN_TUNE=1`)
scripts/cluster/scaling_derecho/README.md-522-
scripts/cluster/scaling_derecho/README.md-523-Once the route-B lanes are green on this machine, the remaining strong-
scripts/cluster/scaling_derecho/README.md-524-scaling headroom at small tiles is **per-step message count × latency**
scripts/cluster/scaling_derecho/README.md:525:(census: cube 46 collective-permutes/step at 6 devices with field packing
scripts/cluster/scaling_derecho/README.md-526-already at floor; atm latlon 41 → 29 behind the fused-halo flag — see
scripts/cluster/scaling_derecho/README.md:527:`docs/performance/scaling/spmd_message_census_2026-07-08.md`). Lane T runs
scripts/cluster/scaling_derecho/README.md-528-the ranked rungs as same-allocation A/B arms (a fresh `base` control arm is
scripts/cluster/scaling_derecho/README.md-529-re-run in the same job — never compare against an earlier job's numbers):
scripts/cluster/scaling_derecho/README.md-530-
scripts/cluster/scaling_derecho/README.md-531-1. `fused` — `LEGOESM_LATLON_SPMD_FUSED_HALO=1`.  **MEASURED 2026-07-09
scripts/cluster/scaling_derecho/README.md-532-   (8×A100): latlon −12 %, ocean +1 % (noise) — the default stays OFF**
scripts/cluster/scaling_derecho/README.md-533-   (bigger messages + pack/unpack copies outweigh the −29 % message count
scripts/cluster/scaling_derecho/README.md-534-   at LL512/np8; re-A/B at higher rank counts before discarding).
scripts/cluster/scaling_derecho/README.md-535-2. `xla` — CP-combine 32 MiB + pipelined p2p.  **MEASURED: −10 % — not
scripts/cluster/scaling_derecho/README.md-536-   recommended combined; split the two flags in a follow-up arm.**
scripts/cluster/scaling_derecho/README.md-537-3. `pgle` — `JAX_ENABLE_PGLE=true JAX_PGLE_PROFILING_RUNS=3`.  **MEASURED:
scripts/cluster/scaling_derecho/README.md-538-   +8.5 % (5.97 vs 6.48 ms/step) — the winner; recommend per-run on
scripts/cluster/scaling_derecho/README.md-539-   route-B latlon lanes.**  Stays per-run opt-in (recompiles after the
scripts/cluster/scaling_derecho/README.md-540-   profiling runs — AOT-incompatible, never a `backend.py` default).
scripts/cluster/scaling_derecho/README.md-541-4. If still send/recv-bound: sweep `NCCL_NCHANNELS_PER_NET_PEER` 4→8/16.
scripts/cluster/scaling_derecho/README.md:542:   Full numbers: `docs/performance/scaling/spmd_message_census_2026-07-08.md`.
scripts/cluster/scaling_derecho/README.md-543-
scripts/cluster/scaling_derecho/README.md-544-```bash
scripts/cluster/scaling_derecho/README.md-545-qsub -v RUN_TUNE=1,RUN_NCCL=0,RUN_LATLON=0,RUN_OCEAN=0,RUN_MPAS=0 \
scripts/cluster/scaling_derecho/README.md-546-     scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs
scripts/cluster/scaling_derecho/README.md-547-```
scripts/cluster/scaling_derecho/README.md-548-
scripts/cluster/scaling_derecho/README.md-549-Outputs land under `$OUTDIR/_ab_tuning/` — a path the tidy-CSV aggregator
scripts/cluster/scaling_derecho/README.md-550-deliberately skips, so A/B receipts never contaminate the scaling curves;
scripts/cluster/scaling_derecho/README.md-551-read the per-arm `steady_median_ms` / `sypd` straight from the JSONL rows.
--
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-19-RECIPE = sys.argv[1]; OUT = sys.argv[2]; DT = 2700.0
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-20-NSTEPS = 11520; ACC0 = 0  # 1-year screen: full year-1 mean (matched to NEMO annual mean)
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-21-RUN = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ"
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-22-g = read_nemo_mesh_mask(f"{RUN}/mesh_mask.nc", nn_hls=2)
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-23-s = read_nemo_restart(f"{RUN}/DINO_00000320_restart.nc", nn_hls=2)  # geometry donor only
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-24-br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-25-ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else None
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-26-cfg = dataclasses.replace(dino_config_for_recipe(RECIPE),
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-27-    lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)   # bridge-frame lon fix
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-28-st = dino_lat_lon_state(br.geometry, br.z_coord, cfg, land_mask_override=br.land_mask)
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py:29:# --- HARD topo census gate: legoESM wet cells must equal NEMO tmask exactly ---
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-30-import netCDF4 as _nc
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-31-_mm = _nc.Dataset(f"{RUN}/mesh_mask.nc"); _H = 2
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-32-_tm = (np.moveaxis(np.asarray(_mm["tmask"][0]).squeeze(), 0, -1)[_H:-_H, _H:-_H] > 0.5)
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-33-_wet = np.asarray(br.z_coord.is_active) & (np.asarray(st.land_mask.data) > 0.5)[:, :, None]
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-34-if not np.array_equal(_wet, _tm):
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-35-    _d = int(np.sum(_wet != _tm))
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-36-    raise SystemExit(f"TOPO CENSUS FAIL: {_d} cells differ from NEMO tmask — refusing to run")
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py:37:print(f"topo census OK: {int(_tm.sum())} wet cells == NEMO tmask")          # analytic IC == NEMO usrdef_istate
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-38-mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-39-model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-40-forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-41-sf = dino_step_surface_forcing(forcing)   # WIND: tau_x/taum into the dycore external-tau block
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-42-print(f"slope_scheme={mc.gm_redi.slope_scheme} kappa_GM_max={float(jnp.max(jnp.abs(mc.gm_redi.kappa_GM))):.1f}")
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-43-print(f"tau_x[Pa] min/max = {float(jnp.min(sf.tau_x)):.3f}/{float(jnp.max(sf.tau_x)):.3f}")
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-44-
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-45-dyn = jax.jit(lambda st: model.step(st, DT, surface_forcing=sf))  # sf constant (annual tau)
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-46-
scripts/validate/ocean_fidelity/dino_1226/dino_year_screen.py-47-acc = {k: jnp.zeros_like(getattr(st, k).data) for k in ("T", "S", "eta", "u", "v")}
--
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-8-already built the population decomposition (commit d2d680d8d) -- the
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-9-headline err_norm's mask is the SURFACE u-mask broadcast to depth, which
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-10-includes below-seafloor cells NEMO's own dump carries as leftover Krhs
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-11-(``dynzad.F90:86`` has no per-level umask guard; NEMO discards it later at
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-12-``dynzdf.F90:121``). This script re-runs that exact decomposition (does not
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-13-re-derive it) and prints the corrected active-only row value + per-level
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-14-profile for the record.
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-15-
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-16-TASK B (the real target): at Python level 29 there are ZERO masked/inactive
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-17-cells, yet S_act jumps ~2000x in RMS from level 28 to 29. This script:
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py:18:  1. censuses the jk 29/30 transition (bathymetry shelf count, e3 ratio,
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-19-     gdepw, mi96 stretch parameters, any DINO bathymetry feature at ~1900 m
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-20-     besides the ~625 m sill),
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-21-  2. maps the level-29 error spatially (few columns vs basin-wide),
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-22-  3. decomposes ZAD's own inputs (ww, shear, e3u(Kmm), area factors, the
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-23-     assembled flux zWdzU) AT LEVEL 29 ONLY against NEMO's own dumps,
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-24-  4. requires magnitude arithmetic predicting the ~2000x jump before
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-25-     naming a cause.
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-26-
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-27-Run::
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-28-
--
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-190-    for k in range(24, min(n_lev_d, 36)):
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-191-        print(f"  lev {k:2d} (jk {k+1:2d}): n_active={cnt_act[k]:5d}  n_inactive_wet2d={cnt_inact[k]:5d}  "
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-192-              f"err_by_level(active)={ebl_act[k]:.4e}")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-193-    print(f"\n  TASK A HEADLINE (active-only, harness-corrected): {en_act:.4e}")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-194-    print("  This is a HARNESS fix only -- production legoESM already zeros")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-195-    print("  below-seafloor u-faces correctly (max|vertadv_u|=0.0 there, prior")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-196-    print("  finding); the ORIGINAL union number mixed in cells NEMO itself")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-197-    print("  discards before use. Not recorded to fidelity_bar_gate.py here.")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-198-
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-199-    # =========================================================================
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py:200:    # TASK B STEP 1: census the jk 29/30 transition.
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-201-    # =========================================================================
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-202-    print("\n" + "=" * 78)
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py:203:    print("TASK B STEP 1: census of Python level 28 -> 29 -> 30 -> 31 transition")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-204-    print("=" * 78)
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-205-    with nc.Dataset(os.path.join(RUN_DIR, "mesh_mask.nc")) as ds:
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-206-        gdepw_1d = np.asarray(ds.variables["gdepw_1d"][0], dtype=np.float64)
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-207-        gdept_1d = np.asarray(ds.variables["gdept_1d"][0], dtype=np.float64)
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-208-        e3t_1d = np.asarray(ds.variables["e3t_1d"][0], dtype=np.float64)
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-209-        mbathy = np.asarray(ds.variables["mbathy"][0], dtype=np.int64)  # NEMO 1-indexed bottom T-level count
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-210-        e1t = np.asarray(ds.variables["e1t"][0], dtype=np.float64)
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-211-        e2t = np.asarray(ds.variables["e2t"][0], dtype=np.float64)
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-212-        e1u = np.asarray(ds.variables["e1u"][0], dtype=np.float64)
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-213-        e2v = np.asarray(ds.variables["e2v"][0], dtype=np.float64)
--
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-215-    print("  1-D reference ladder (levels 27-32, python idx, gdepw/gdept in m, e3t_1d in m):")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-216-    for k in range(26, 32):
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-217-        print(f"    lev {k:2d} (jk {k+1:2d}): gdepw_1d={gdepw_1d[k]:8.2f}  gdept_1d={gdept_1d[k]:8.2f}  e3t_1d={e3t_1d[k]:.3f}")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-218-    ratio_2928 = float(e3t_1d[29] / e3t_1d[28]) if e3t_1d[28] != 0 else float("nan")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-219-    ratio_3029 = float(e3t_1d[30] / e3t_1d[29]) if e3t_1d[29] != 0 else float("nan")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-220-    print(f"  e3t_1d[29]/e3t_1d[28] = {ratio_2928:.4f}   e3t_1d[30]/e3t_1d[29] = {ratio_3029:.4f}")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-221-    print("  (a smooth tanh stretch predicts a ratio close to 1 here -- no")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-222-    print("  discontinuity in the 1-D reference ladder itself at this level;")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-223-    print("  any per-level jump must come from a 2-D/3-D field, not the ladder.)")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-224-
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py:225:    # Bathymetry shelf census: how many T-columns have their WET COLUMN
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-226-    # BOTTOM exactly at level 29/30 (a shelf edge at this depth)?
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-227-    bottom_level_t = np.asarray(br.z_coord.bottom_level)  # 0-indexed python bottom T-level per column
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-228-    tmask2d = np.asarray(g.tmask)[..., 0] > 0.5
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-229-    for lev in (27, 28, 29, 30, 31, 32):
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-230-        n_shelf = int(np.sum((bottom_level_t == lev) & tmask2d))
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-231-        print(f"  columns whose WET-COLUMN BOTTOM is exactly level {lev:2d} (jk {lev+1}): {n_shelf}")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-232-    n_wet_total = int(tmask2d.sum())
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-233-    n_wet_at_29_or_deeper = int(np.sum((bottom_level_t >= 29) & tmask2d))
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-234-    print(f"  total wet T-columns: {n_wet_total}; columns reaching level >=29 (still wet AT level 29): {n_wet_at_29_or_deeper}")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-235-    print("  (DINO's documented sill is ~625 m -- levels 27-32 span ~1140-2350 m,")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py:236:    print("  well below the sill; this census checks for any SECOND bathymetry")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-237-    print("  feature -- ridge/Drake-sill parameters live in usrdef_nam, not")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-238-    print("  reconstructed here since br.state.H_bathy is read directly from")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py:239:    print("  NEMO's own mesh_mask, not re-derived -- so this census uses the")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-240-    print("  ACTUAL bathymetry NEMO ran with, not a legoESM reconstruction.)")
scripts/validate/ocean_fidelity/dino_1226/zad_level29_onset_walk.py-241-

exec
/usr/bin/bash -lc "sed -n '220,520p' tests/bench/test_bench_voronoi_partition_methods.py | nl -ba -v220
rg -l \"def spmd_schedule_cost\" legoesm
rg -n -A 220 -B 30 \"def spmd_schedule_cost\" legoesm/parallel/sharded_dynamics.py
bash -u -x -c 'f() { printf \"argc=%d\\n\" \""'$#"; for x in "$@"; do printf "<%s>'"\\n\" \""'$x"; done; }; g() { f A ${4:+--expect-rounds "$4"} Z; local rc=$?; printf "rc=%s'"\\n\" \""'$rc"; }; g 1 2 3 "x y"; g 1 2 3 ""'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 170ms:
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
rg: legoesm: IO error for operation on legoesm: No such file or directory (os error 2)
rg: legoesm/parallel/sharded_dynamics.py: IO error for operation on legoesm/parallel/sharded_dynamics.py: No such file or directory (os error 2)
+ g 1 2 3 'x y'
argc=4
+ f A --expect-rounds 'x y' Z
+ printf 'argc=%d\n' 4
+ for x in "$@"
<A>
+ printf '<%s>\n' A
+ for x in "$@"
+ printf '<%s>\n' --expect-rounds
<--expect-rounds>
<x y>
<Z>
+ for x in "$@"
+ printf '<%s>\n' 'x y'
+ for x in "$@"
+ printf '<%s>\n' Z
+ local rc=0
+ printf 'rc=%s\n' 0
rc=0
+ g 1 2 3 ''
+ f A Z
+ printf 'argc=%d\n' 2
+ for x in "$@"
+ printf '<%s>\n' A
+ for x in "$@"
argc=2
<A>
<Z>
rc=0
+ printf '<%s>\n' Z
+ local rc=0
+ printf 'rc=%s\n' 0

exec
/usr/bin/bash -lc "rg -l \"def spmd_schedule_cost\" . --glob '*.py' --glob '"'!*.pyc'"' && rg -n -A 260 -B 40 \"def spmd_schedule_cost\" . --glob '*.py' --glob '"'!*.pyc'"' && rg -n -C 10 \"comm_pairs|_build_ppermute_schedule\" . --glob '*.py' --glob '"'!*.pyc'"' | head -n 800 && nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '1,475p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 2320ms:
./packages/core/legoesm/parallel/sharded_dynamics.py
./packages/core/legoesm/parallel/sharded_dynamics.py-1210-    halo_cells_set : set[int]
./packages/core/legoesm/parallel/sharded_dynamics.py-1211-        Initial halo (cellsOnCell ring + iter-23 owned-edge other-cells).
./packages/core/legoesm/parallel/sharded_dynamics.py-1212-    cellsOnEdge_np : np.ndarray, shape (2, nEdges), int
./packages/core/legoesm/parallel/sharded_dynamics.py-1213-        Global cellsOnEdge connectivity.
./packages/core/legoesm/parallel/sharded_dynamics.py-1214-    n_passes : int
./packages/core/legoesm/parallel/sharded_dynamics.py-1215-        Number of augmentation iterations.  2 is the minimum that
./packages/core/legoesm/parallel/sharded_dynamics.py-1216-        closes the dycore's depth-2 operator chain; use a higher
./packages/core/legoesm/parallel/sharded_dynamics.py-1217-        value only if a future operator extends the chain depth.
./packages/core/legoesm/parallel/sharded_dynamics.py-1218-    """
./packages/core/legoesm/parallel/sharded_dynamics.py-1219-    for _ in range(n_passes):
./packages/core/legoesm/parallel/sharded_dynamics.py-1220-        cur_local_arr = np.concatenate([
./packages/core/legoesm/parallel/sharded_dynamics.py-1221-            owned_cells_arr,
./packages/core/legoesm/parallel/sharded_dynamics.py-1222-            np.fromiter(
./packages/core/legoesm/parallel/sharded_dynamics.py-1223-                halo_cells_set,
./packages/core/legoesm/parallel/sharded_dynamics.py-1224-                dtype=np.int64,
./packages/core/legoesm/parallel/sharded_dynamics.py-1225-                count=len(halo_cells_set),
./packages/core/legoesm/parallel/sharded_dynamics.py-1226-            ),
./packages/core/legoesm/parallel/sharded_dynamics.py-1227-        ])
./packages/core/legoesm/parallel/sharded_dynamics.py-1228-        # Edges where AT LEAST one ``cellsOnEdge`` is currently local.
./packages/core/legoesm/parallel/sharded_dynamics.py-1229-        edge_one_in = (
./packages/core/legoesm/parallel/sharded_dynamics.py-1230-            np.isin(cellsOnEdge_np[0], cur_local_arr)
./packages/core/legoesm/parallel/sharded_dynamics.py-1231-            | np.isin(cellsOnEdge_np[1], cur_local_arr)
./packages/core/legoesm/parallel/sharded_dynamics.py-1232-        )
./packages/core/legoesm/parallel/sharded_dynamics.py-1233-        cand_edges = np.flatnonzero(edge_one_in)
./packages/core/legoesm/parallel/sharded_dynamics.py-1234-        cand_cells = cellsOnEdge_np[:, cand_edges].reshape(-1)
./packages/core/legoesm/parallel/sharded_dynamics.py-1235-        cand_cells = np.unique(cand_cells[cand_cells >= 0])
./packages/core/legoesm/parallel/sharded_dynamics.py-1236-        # Set difference: cells not yet in local set.
./packages/core/legoesm/parallel/sharded_dynamics.py-1237-        new_cells = cand_cells[~np.isin(cand_cells, cur_local_arr)]
./packages/core/legoesm/parallel/sharded_dynamics.py-1238-        if new_cells.size == 0:
./packages/core/legoesm/parallel/sharded_dynamics.py-1239-            return
./packages/core/legoesm/parallel/sharded_dynamics.py-1240-        halo_cells_set.update(new_cells.tolist())
./packages/core/legoesm/parallel/sharded_dynamics.py-1241-
./packages/core/legoesm/parallel/sharded_dynamics.py-1242-
./packages/core/legoesm/parallel/sharded_dynamics.py-1243-#: Halo depth the SPMD Voronoi partition infra is built at.  ONE definition
./packages/core/legoesm/parallel/sharded_dynamics.py-1244-#: consumed by both the production step factory and ``spmd_schedule_cost``:
./packages/core/legoesm/parallel/sharded_dynamics.py-1245-#: a score computed at a different depth describes a different comm graph, and
./packages/core/legoesm/parallel/sharded_dynamics.py-1246-#: two independently hardcoded 3s let production drift unnoticed.
./packages/core/legoesm/parallel/sharded_dynamics.py-1247-SPMD_HALO_DEPTH = 3
./packages/core/legoesm/parallel/sharded_dynamics.py-1248-
./packages/core/legoesm/parallel/sharded_dynamics.py-1249-
./packages/core/legoesm/parallel/sharded_dynamics.py:1250:def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
./packages/core/legoesm/parallel/sharded_dynamics.py-1251-                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
./packages/core/legoesm/parallel/sharded_dynamics.py-1252-                       ppermute_cells_per_device_threshold=2_000):
./packages/core/legoesm/parallel/sharded_dynamics.py-1253-    """How much halo communication one ownership choice costs, computed offline.
./packages/core/legoesm/parallel/sharded_dynamics.py-1254-
./packages/core/legoesm/parallel/sharded_dynamics.py-1255-    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
./packages/core/legoesm/parallel/sharded_dynamics.py-1256-    ROUNDS one halo exchange needs -- the sequential collective launches that
./packages/core/legoesm/parallel/sharded_dynamics.py-1257-    dominate MPAS strong scaling above ~64 devices.  Runs on a laptop: no GPU,
./packages/core/legoesm/parallel/sharded_dynamics.py-1258-    no MPI, no benchmark job, so a split can be compared before it costs an
./packages/core/legoesm/parallel/sharded_dynamics.py-1259-    allocation.
./packages/core/legoesm/parallel/sharded_dynamics.py-1260-
./packages/core/legoesm/parallel/sharded_dynamics.py-1261-    It calls the SAME builders production calls
./packages/core/legoesm/parallel/sharded_dynamics.py-1262-    (:func:`_build_voronoi_partition_infra` then
./packages/core/legoesm/parallel/sharded_dynamics.py-1263-    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
./packages/core/legoesm/parallel/sharded_dynamics.py-1264-    different question: a 1-ring ``cellsOnEdge`` adjacency graph reports 8
./packages/core/legoesm/parallel/sharded_dynamics.py-1265-    rounds where the real depth-3-plus-closure graph reports 12-14.
./packages/core/legoesm/parallel/sharded_dynamics.py-1266-
./packages/core/legoesm/parallel/sharded_dynamics.py-1267-    WHAT THE NUMBER IS NOT
./packages/core/legoesm/parallel/sharded_dynamics.py-1268-    ----------------------
./packages/core/legoesm/parallel/sharded_dynamics.py-1269-    * ``n_rounds`` is per HALO FILL, not per model step.  A step costs
./packages/core/legoesm/parallel/sharded_dynamics.py-1270-      ``n_rounds`` x (tendency evaluations per step), which depends on the
./packages/core/legoesm/parallel/sharded_dynamics.py-1271-      configured integrator -- SSP-RK3 evaluates 3 times, but the MPAS default
./packages/core/legoesm/parallel/sharded_dynamics.py-1272-      is ``ssp_rk54_scan``.  Multiply with the integrator you actually run.
./packages/core/legoesm/parallel/sharded_dynamics.py-1273-    * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
./packages/core/legoesm/parallel/sharded_dynamics.py-1274-      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
./packages/core/legoesm/parallel/sharded_dynamics.py-1275-      keeps the best; equality is MEASURED (compare the returned
./packages/core/legoesm/parallel/sharded_dynamics.py-1276-      ``max_degree``), never assumed.  Do not claim "the colouring is already
./packages/core/legoesm/parallel/sharded_dynamics.py-1277-      optimal so only ownership can help" from this function.
./packages/core/legoesm/parallel/sharded_dynamics.py-1278-    * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
./packages/core/legoesm/parallel/sharded_dynamics.py-1279-      cells/device is below ``ppermute_cells_per_device_threshold``, in which
./packages/core/legoesm/parallel/sharded_dynamics.py-1280-      case there is no ppermute schedule and this number is counterfactual --
./packages/core/legoesm/parallel/sharded_dynamics.py-1281-      see the returned ``production_strategy``.
./packages/core/legoesm/parallel/sharded_dynamics.py-1282-
./packages/core/legoesm/parallel/sharded_dynamics.py-1283-    MESH STATE -- the one thing that silently invalidates the score
./packages/core/legoesm/parallel/sharded_dynamics.py-1284-    --------------------------------------------------------------
./packages/core/legoesm/parallel/sharded_dynamics.py-1285-    Production does NOT reorder inside ``make_voronoi_sharded_step``; it
./packages/core/legoesm/parallel/sharded_dynamics.py-1286-    consumes an already-reordered ``model.mesh``.  The scaling bench reorders
./packages/core/legoesm/parallel/sharded_dynamics.py-1287-    ONCE for a ``reorder_target`` device count and then runs at a possibly
./packages/core/legoesm/parallel/sharded_dynamics.py-1288-    DIFFERENT device count.  So pass what you actually have:
./packages/core/legoesm/parallel/sharded_dynamics.py-1289-
./packages/core/legoesm/parallel/sharded_dynamics.py-1290-    * raw mesh, scoring a run at ``n_dev``: defaults are right.
./packages/core/legoesm/parallel/sharded_dynamics.py-1291-    * raw mesh, but the run reorders for a different target: pass
./packages/core/legoesm/parallel/sharded_dynamics.py-1292-      ``reorder_target=<that target>``; the split is built for the target and
./packages/core/legoesm/parallel/sharded_dynamics.py-1293-      scored at ``n_dev``.
./packages/core/legoesm/parallel/sharded_dynamics.py-1294-    * already-reordered mesh (what production holds): pass
./packages/core/legoesm/parallel/sharded_dynamics.py-1295-      ``already_reordered=True``; ``method`` is then ignored and reported as
./packages/core/legoesm/parallel/sharded_dynamics.py-1296-      ``"pre-reordered"``, because the ownership is already baked in.
./packages/core/legoesm/parallel/sharded_dynamics.py-1297-
./packages/core/legoesm/parallel/sharded_dynamics.py-1298-    Parameters
./packages/core/legoesm/parallel/sharded_dynamics.py-1299-    ----------
./packages/core/legoesm/parallel/sharded_dynamics.py-1300-    mesh : VoronoiMesh
./packages/core/legoesm/parallel/sharded_dynamics.py-1301-    n_dev : int
./packages/core/legoesm/parallel/sharded_dynamics.py-1302-        Device count the run uses.  Must be >= 1.
./packages/core/legoesm/parallel/sharded_dynamics.py-1303-    method : str
./packages/core/legoesm/parallel/sharded_dynamics.py-1304-        Ownership for the reorder; ignored when *already_reordered*.
./packages/core/legoesm/parallel/sharded_dynamics.py-1305-    reorder_target : int | None
./packages/core/legoesm/parallel/sharded_dynamics.py-1306-        Device count the reorder targets, when it differs from *n_dev*.
./packages/core/legoesm/parallel/sharded_dynamics.py-1307-    already_reordered : bool
./packages/core/legoesm/parallel/sharded_dynamics.py-1308-    halo_depth : int
./packages/core/legoesm/parallel/sharded_dynamics.py-1309-        Must match production (3) or the graph is a different graph.
./packages/core/legoesm/parallel/sharded_dynamics.py-1310-    ppermute_cells_per_device_threshold : int
./packages/core/legoesm/parallel/sharded_dynamics.py-1311-        Mirror of the production auto-select threshold, only used to report
./packages/core/legoesm/parallel/sharded_dynamics.py-1312-        ``production_strategy``.
./packages/core/legoesm/parallel/sharded_dynamics.py-1313-
./packages/core/legoesm/parallel/sharded_dynamics.py-1314-    Returns
./packages/core/legoesm/parallel/sharded_dynamics.py-1315-    -------
./packages/core/legoesm/parallel/sharded_dynamics.py-1316-    dict
./packages/core/legoesm/parallel/sharded_dynamics.py-1317-        ``n_rounds`` (the cost), ``max_degree`` (the lower bound to compare
./packages/core/legoesm/parallel/sharded_dynamics.py-1318-        it against), ``n_rounds_greedy``, ``coloring_method``,
./packages/core/legoesm/parallel/sharded_dynamics.py-1319-        ``resolved_method`` (concrete, never ``"auto"``),
./packages/core/legoesm/parallel/sharded_dynamics.py-1320-        ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
./packages/core/legoesm/parallel/sharded_dynamics.py-1321-        ``max_local_cells``, ``max_local_edges``, and the echoed inputs.
./packages/core/legoesm/parallel/sharded_dynamics.py-1322-
./packages/core/legoesm/parallel/sharded_dynamics.py-1323-    Reference census on the unrelaxed mesh, which any change here must still
./packages/core/legoesm/parallel/sharded_dynamics.py-1324-    reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
./packages/core/legoesm/parallel/sharded_dynamics.py-1325-    geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
./packages/core/legoesm/parallel/sharded_dynamics.py-1326-    """
./packages/core/legoesm/parallel/sharded_dynamics.py-1327-    from legoesm.parallel.voronoi_partition import (
./packages/core/legoesm/parallel/sharded_dynamics.py-1328-        reorder_voronoi_for_sharding, resolve_sharding_partition_method,
./packages/core/legoesm/parallel/sharded_dynamics.py-1329-    )
./packages/core/legoesm/parallel/sharded_dynamics.py-1330-
./packages/core/legoesm/parallel/sharded_dynamics.py-1331-    if int(n_dev) != n_dev or int(n_dev) < 1:
./packages/core/legoesm/parallel/sharded_dynamics.py-1332-        # int() would silently truncate 3.9 -> 3 and score the wrong split.
./packages/core/legoesm/parallel/sharded_dynamics.py-1333-        raise ValueError(
./packages/core/legoesm/parallel/sharded_dynamics.py-1334-            f"spmd_schedule_cost: n_dev must be an integer >= 1, got {n_dev!r}")
./packages/core/legoesm/parallel/sharded_dynamics.py-1335-    n_dev = int(n_dev)
./packages/core/legoesm/parallel/sharded_dynamics.py-1336-
./packages/core/legoesm/parallel/sharded_dynamics.py-1337-    if already_reordered:
./packages/core/legoesm/parallel/sharded_dynamics.py-1338-        if reorder_target is not None:
./packages/core/legoesm/parallel/sharded_dynamics.py-1339-            raise ValueError(
./packages/core/legoesm/parallel/sharded_dynamics.py-1340-                "spmd_schedule_cost: reorder_target is meaningless with "
./packages/core/legoesm/parallel/sharded_dynamics.py-1341-                "already_reordered=True — the ownership is already baked into "
./packages/core/legoesm/parallel/sharded_dynamics.py-1342-                "the mesh.")
./packages/core/legoesm/parallel/sharded_dynamics.py-1343-        prepared, resolved = mesh, "pre-reordered"
./packages/core/legoesm/parallel/sharded_dynamics.py-1344-    else:
./packages/core/legoesm/parallel/sharded_dynamics.py-1345-        target = n_dev if reorder_target is None else int(reorder_target)
./packages/core/legoesm/parallel/sharded_dynamics.py-1346-        prepared = reorder_voronoi_for_sharding(mesh, target, method=method)
./packages/core/legoesm/parallel/sharded_dynamics.py-1347-        # Report the CONCRETE ownership: "auto" hides which partitioner ran.
./packages/core/legoesm/parallel/sharded_dynamics.py-1348-        # Uses the SAME resolver the reorder used, so the label cannot drift
./packages/core/legoesm/parallel/sharded_dynamics.py-1349-        # from the policy.
./packages/core/legoesm/parallel/sharded_dynamics.py-1350-        resolved = resolve_sharding_partition_method(method)
./packages/core/legoesm/parallel/sharded_dynamics.py-1351-
./packages/core/legoesm/parallel/sharded_dynamics.py-1352-    # The builder assigns residual entities to the LAST owner but excludes them
./packages/core/legoesm/parallel/sharded_dynamics.py-1353-    # from every owned contiguous block, so schedule send indices can exceed a
./packages/core/legoesm/parallel/sharded_dynamics.py-1354-    # device's shard length -- a number that looks fine and is not.  Reachable
./packages/core/legoesm/parallel/sharded_dynamics.py-1355-    # via reorder_target: a mesh padded for 3 devices is not divisible by 4.
./packages/core/legoesm/parallel/sharded_dynamics.py-1356-    # The scaling bench rejects that pairing; so does this.
./packages/core/legoesm/parallel/sharded_dynamics.py-1357-    n_cells, n_edges = int(prepared.nCells), int(prepared.nEdges)
./packages/core/legoesm/parallel/sharded_dynamics.py-1358-    if n_cells % n_dev or n_edges % n_dev:
./packages/core/legoesm/parallel/sharded_dynamics.py-1359-        raise ValueError(
./packages/core/legoesm/parallel/sharded_dynamics.py-1360-            f"spmd_schedule_cost: prepared mesh has nCells={n_cells}, "
./packages/core/legoesm/parallel/sharded_dynamics.py-1361-            f"nEdges={n_edges}, neither divisible by n_dev={n_dev}. The mesh "
./packages/core/legoesm/parallel/sharded_dynamics.py-1362-            f"is padded for its reorder target"
./packages/core/legoesm/parallel/sharded_dynamics.py-1363-            f"{'' if already_reordered else f' ({target})'}, so scoring it at "
./packages/core/legoesm/parallel/sharded_dynamics.py-1364-            f"a device count that does not divide it silently mis-slices the "
./packages/core/legoesm/parallel/sharded_dynamics.py-1365-            f"owned blocks. Score at a device count that divides the prepared "
./packages/core/legoesm/parallel/sharded_dynamics.py-1366-            f"mesh.")
./packages/core/legoesm/parallel/sharded_dynamics.py-1367-    (
./packages/core/legoesm/parallel/sharded_dynamics.py-1368-        _stacked, _gc, _ge, _noc, _noe, max_lc, max_le, partitions, cell_owner,
./packages/core/legoesm/parallel/sharded_dynamics.py-1369-    ) = _build_voronoi_partition_infra(prepared, n_dev, halo_depth=halo_depth)
./packages/core/legoesm/parallel/sharded_dynamics.py-1370-    cells_per = n_cells // n_dev
./packages/core/legoesm/parallel/sharded_dynamics.py-1371-    edges_per = n_edges // n_dev
./packages/core/legoesm/parallel/sharded_dynamics.py-1372-    sched = _build_ppermute_schedule(
./packages/core/legoesm/parallel/sharded_dynamics.py-1373-        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
./packages/core/legoesm/parallel/sharded_dynamics.py-1374-    )
./packages/core/legoesm/parallel/sharded_dynamics.py-1375-    return {
./packages/core/legoesm/parallel/sharded_dynamics.py-1376-        "method": method,
./packages/core/legoesm/parallel/sharded_dynamics.py-1377-        "resolved_method": resolved,
./packages/core/legoesm/parallel/sharded_dynamics.py-1378-        "n_dev": n_dev,
./packages/core/legoesm/parallel/sharded_dynamics.py-1379-        # Unknown for a pre-reordered mesh: the ownership is baked in and the
./packages/core/legoesm/parallel/sharded_dynamics.py-1380-        # target that produced it is not recoverable from the mesh. Reporting
./packages/core/legoesm/parallel/sharded_dynamics.py-1381-        # n_dev there would assert something we did not verify.
./packages/core/legoesm/parallel/sharded_dynamics.py-1382-        "reorder_target": (None if already_reordered else
./packages/core/legoesm/parallel/sharded_dynamics.py-1383-                           (n_dev if reorder_target is None
./packages/core/legoesm/parallel/sharded_dynamics.py-1384-                            else int(reorder_target))),
./packages/core/legoesm/parallel/sharded_dynamics.py-1385-        "already_reordered": bool(already_reordered),
./packages/core/legoesm/parallel/sharded_dynamics.py-1386-        "halo_depth": halo_depth,
./packages/core/legoesm/parallel/sharded_dynamics.py-1387-        "n_rounds": int(sched["n_rounds"]),
./packages/core/legoesm/parallel/sharded_dynamics.py-1388-        "n_rounds_greedy": int(sched["n_rounds_greedy"]),
./packages/core/legoesm/parallel/sharded_dynamics.py-1389-        "max_degree": int(sched.get("max_degree", -1)),
./packages/core/legoesm/parallel/sharded_dynamics.py-1390-        "coloring_method": sched["coloring_method"],
./packages/core/legoesm/parallel/sharded_dynamics.py-1391-        # Production returns before selecting a strategy at n_dev==1, and a
./packages/core/legoesm/parallel/sharded_dynamics.py-1392-        # caller may force halo_strategy; this reports what AUTO would pick.
./packages/core/legoesm/parallel/sharded_dynamics.py-1393-        "production_strategy": (
./packages/core/legoesm/parallel/sharded_dynamics.py-1394-            None if n_dev == 1 else
./packages/core/legoesm/parallel/sharded_dynamics.py-1395-            ("allgather" if cells_per < ppermute_cells_per_device_threshold
./packages/core/legoesm/parallel/sharded_dynamics.py-1396-             else "ppermute")),
./packages/core/legoesm/parallel/sharded_dynamics.py-1397-        "cells_per_device": cells_per,
./packages/core/legoesm/parallel/sharded_dynamics.py-1398-        "max_local_cells": int(max_lc),
./packages/core/legoesm/parallel/sharded_dynamics.py-1399-        "max_local_edges": int(max_le),
./packages/core/legoesm/parallel/sharded_dynamics.py-1400-    }
./packages/core/legoesm/parallel/sharded_dynamics.py-1401-
./packages/core/legoesm/parallel/sharded_dynamics.py-1402-
./packages/core/legoesm/parallel/sharded_dynamics.py-1403-def _build_voronoi_partition_infra(global_mesh, n_dev, halo_depth=2):
./packages/core/legoesm/parallel/sharded_dynamics.py-1404-    """Pre-compute per-device local meshes and gather/scatter indices.
./packages/core/legoesm/parallel/sharded_dynamics.py-1405-
./packages/core/legoesm/parallel/sharded_dynamics.py-1406-    After ``reorder_voronoi_for_sharding`` the global mesh has *both*
./packages/core/legoesm/parallel/sharded_dynamics.py-1407-    cells and edges ordered by contiguous device blocks.  We compute
./packages/core/legoesm/parallel/sharded_dynamics.py-1408-    partitions whose owned-entity boundaries exactly match the shard
./packages/core/legoesm/parallel/sharded_dynamics.py-1409-    boundaries (``cells_per = nCells // n_dev``, ``edges_per = nEdges //
./packages/core/legoesm/parallel/sharded_dynamics.py-1410-    n_dev``), then build local meshes with remapped connectivity.
./packages/core/legoesm/parallel/sharded_dynamics.py-1411-
./packages/core/legoesm/parallel/sharded_dynamics.py-1412-    Using ``partition_voronoi_mesh`` directly is unsuitable because it
./packages/core/legoesm/parallel/sharded_dynamics.py-1413-    derives edge ownership from cell ownership, producing an uneven edge
./packages/core/legoesm/parallel/sharded_dynamics.py-1414-    split that mismatches the even shard split.  Instead we construct the
./packages/core/legoesm/parallel/sharded_dynamics.py-1415-    :class:`VoronoiPartition` objects manually with contiguous-block
./packages/core/legoesm/parallel/sharded_dynamics.py-1416-    ownership for both cells **and** edges.
./packages/core/legoesm/parallel/sharded_dynamics.py-1417-
./packages/core/legoesm/parallel/sharded_dynamics.py-1418-    Returns
./packages/core/legoesm/parallel/sharded_dynamics.py-1419-    -------
./packages/core/legoesm/parallel/sharded_dynamics.py-1420-    stacked_meshes : VoronoiMesh
./packages/core/legoesm/parallel/sharded_dynamics.py-1421-        Each leaf has shape ``(n_dev, max_local_*)``.
./packages/core/legoesm/parallel/sharded_dynamics.py-1422-    gather_cells : jnp.ndarray, (n_dev, max_local_cells)
./packages/core/legoesm/parallel/sharded_dynamics.py-1423-    gather_edges : jnp.ndarray, (n_dev, max_local_edges)
./packages/core/legoesm/parallel/sharded_dynamics.py-1424-    n_owned_cells : list[int]
./packages/core/legoesm/parallel/sharded_dynamics.py-1425-    n_owned_edges : list[int]
./packages/core/legoesm/parallel/sharded_dynamics.py-1426-    max_lc : int
./packages/core/legoesm/parallel/sharded_dynamics.py-1427-    max_le : int
./packages/core/legoesm/parallel/sharded_dynamics.py-1428-    partitions : list[VoronoiPartition]
./packages/core/legoesm/parallel/sharded_dynamics.py-1429-        Per-device partition descriptors (for ppermute schedule building).
./packages/core/legoesm/parallel/sharded_dynamics.py-1430-    cell_owner : np.ndarray, (nCells,)
./packages/core/legoesm/parallel/sharded_dynamics.py-1431-        Cell ownership array.
./packages/core/legoesm/parallel/sharded_dynamics.py-1432-    """
./packages/core/legoesm/parallel/sharded_dynamics.py-1433-    import numpy as np
./packages/core/legoesm/parallel/sharded_dynamics.py-1434-    from legoesm.parallel.voronoi_partition import (
./packages/core/legoesm/parallel/sharded_dynamics.py-1435-        HaloCommSchedule,
./packages/core/legoesm/parallel/sharded_dynamics.py-1436-        VoronoiPartition,
./packages/core/legoesm/parallel/sharded_dynamics.py-1437-        build_local_mesh,
./packages/core/legoesm/parallel/sharded_dynamics.py-1438-        compute_halo_cells,
./packages/core/legoesm/parallel/sharded_dynamics.py-1439-    )
./packages/core/legoesm/parallel/sharded_dynamics.py-1440-
./packages/core/legoesm/parallel/sharded_dynamics.py-1441-    nCells = global_mesh.nCells
./packages/core/legoesm/parallel/sharded_dynamics.py-1442-    nEdges = global_mesh.nEdges
./packages/core/legoesm/parallel/sharded_dynamics.py-1443-    nVertices = global_mesh.nVertices
./packages/core/legoesm/parallel/sharded_dynamics.py-1444-    cells_per = nCells // n_dev
./packages/core/legoesm/parallel/sharded_dynamics.py-1445-    edges_per = nEdges // n_dev
./packages/core/legoesm/parallel/sharded_dynamics.py-1446-    verts_per = nVertices // n_dev
./packages/core/legoesm/parallel/sharded_dynamics.py-1447-
./packages/core/legoesm/parallel/sharded_dynamics.py-1448-    # Convert mesh arrays to numpy for setup.
./packages/core/legoesm/parallel/sharded_dynamics.py-1449-    gm_np = jax.tree.map(
./packages/core/legoesm/parallel/sharded_dynamics.py-1450-        lambda x: np.asarray(x) if hasattr(x, "shape") else x,
./packages/core/legoesm/parallel/sharded_dynamics.py-1451-        global_mesh,
./packages/core/legoesm/parallel/sharded_dynamics.py-1452-    )
./packages/core/legoesm/parallel/sharded_dynamics.py-1453-    cellsOnCell_np = np.asarray(gm_np.cellsOnCell)       # (maxEdges, nCells)
./packages/core/legoesm/parallel/sharded_dynamics.py-1454-    cellsOnEdge_np = np.asarray(gm_np.cellsOnEdge)       # (2, nEdges)
./packages/core/legoesm/parallel/sharded_dynamics.py-1455-    np.asarray(gm_np.cellsOnVertex)    # (vDeg, nVerts)
./packages/core/legoesm/parallel/sharded_dynamics.py-1456-    verticesOnCell_np = np.asarray(gm_np.verticesOnCell)   # (maxEdges, nCells)
./packages/core/legoesm/parallel/sharded_dynamics.py-1457-    verticesOnEdge_np = np.asarray(gm_np.verticesOnEdge)   # (2, nEdges)
./packages/core/legoesm/parallel/sharded_dynamics.py-1458-    maxEdges = int(gm_np.maxEdges)
./packages/core/legoesm/parallel/sharded_dynamics.py-1459-    int(gm_np.vertexDegree)
./packages/core/legoesm/parallel/sharded_dynamics.py-1460-
./packages/core/legoesm/parallel/sharded_dynamics.py-1461-    # Contiguous-block cell ownership (matches shard layout).
./packages/core/legoesm/parallel/sharded_dynamics.py-1462-    cell_owner = np.repeat(np.arange(n_dev, dtype=np.int32), cells_per)
./packages/core/legoesm/parallel/sharded_dynamics.py-1463-    if len(cell_owner) < nCells:
./packages/core/legoesm/parallel/sharded_dynamics.py-1464-        cell_owner = np.concatenate([
./packages/core/legoesm/parallel/sharded_dynamics.py-1465-            cell_owner,
./packages/core/legoesm/parallel/sharded_dynamics.py-1466-            np.full(nCells - len(cell_owner), n_dev - 1, dtype=np.int32),
./packages/core/legoesm/parallel/sharded_dynamics.py-1467-        ])
./packages/core/legoesm/parallel/sharded_dynamics.py-1468-
./packages/core/legoesm/parallel/sharded_dynamics.py-1469-    # Dummy comm schedule (not needed for shard_map path).
./packages/core/legoesm/parallel/sharded_dynamics.py-1470-    _dummy_comm = HaloCommSchedule(
./packages/core/legoesm/parallel/sharded_dynamics.py-1471-        neighbor_ranks=(), send_counts=(), recv_counts=(),
./packages/core/legoesm/parallel/sharded_dynamics.py-1472-        send_idx=jnp.empty(0, dtype=jnp.int32),
./packages/core/legoesm/parallel/sharded_dynamics.py-1473-        recv_idx=jnp.empty(0, dtype=jnp.int32),
./packages/core/legoesm/parallel/sharded_dynamics.py-1474-    )
./packages/core/legoesm/parallel/sharded_dynamics.py-1475-
./packages/core/legoesm/parallel/sharded_dynamics.py-1476-    partitions = []
./packages/core/legoesm/parallel/sharded_dynamics.py-1477-    local_meshes_raw = []
./packages/core/legoesm/parallel/sharded_dynamics.py-1478-
./packages/core/legoesm/parallel/sharded_dynamics.py-1479-    for rank in range(n_dev):
./packages/core/legoesm/parallel/sharded_dynamics.py-1480-        # ----- Owned entities (contiguous blocks) ----- #
./packages/core/legoesm/parallel/sharded_dynamics.py-1481-        c_start, c_end = rank * cells_per, (rank + 1) * cells_per
./packages/core/legoesm/parallel/sharded_dynamics.py-1482-        e_start, e_end = rank * edges_per, (rank + 1) * edges_per
./packages/core/legoesm/parallel/sharded_dynamics.py-1483-        v_start, v_end = rank * verts_per, (rank + 1) * verts_per
./packages/core/legoesm/parallel/sharded_dynamics.py-1484-
./packages/core/legoesm/parallel/sharded_dynamics.py-1485-        owned_cells = np.arange(c_start, c_end, dtype=np.int64)
./packages/core/legoesm/parallel/sharded_dynamics.py-1486-        owned_edges = np.arange(e_start, e_end, dtype=np.int64)
./packages/core/legoesm/parallel/sharded_dynamics.py-1487-        owned_vertices = np.arange(v_start, v_end, dtype=np.int64)
./packages/core/legoesm/parallel/sharded_dynamics.py-1488-        owned_cells_set = set(owned_cells.tolist())
./packages/core/legoesm/parallel/sharded_dynamics.py-1489-        set(owned_edges.tolist())
./packages/core/legoesm/parallel/sharded_dynamics.py-1490-
./packages/core/legoesm/parallel/sharded_dynamics.py-1491-        # ----- Halo cells: k-ring neighbours of owned cells ----- #
./packages/core/legoesm/parallel/sharded_dynamics.py-1492-        halo_cells_set = compute_halo_cells(
./packages/core/legoesm/parallel/sharded_dynamics.py-1493-            cell_owner, cellsOnCell_np, maxEdges, rank, halo_depth,
./packages/core/legoesm/parallel/sharded_dynamics.py-1494-        )
./packages/core/legoesm/parallel/sharded_dynamics.py-1495-        # Augment with the OTHER cell of every owned edge: on Voronoi
./packages/core/legoesm/parallel/sharded_dynamics.py-1496-        # SCVT meshes the cellsOnCell adjacency *should* match the
./packages/core/legoesm/parallel/sharded_dynamics.py-1497-        # cellsOnEdge connectivity, but the k-ring construction in
./packages/core/legoesm/parallel/sharded_dynamics.py-1498-        # ``compute_halo_cells`` can miss a handful of cells at the
./packages/core/legoesm/parallel/sharded_dynamics.py-1499-        # mesh boundary or near pentagons (owned edges whose far cell
./packages/core/legoesm/parallel/sharded_dynamics.py-1500-        # is reachable via cellsOnEdge but whose hop chain through
./packages/core/legoesm/parallel/sharded_dynamics.py-1501-        # cellsOnCell at depth <= halo_depth is broken by a -1 slot
./packages/core/legoesm/parallel/sharded_dynamics.py-1502-        # or pentagon irregularity).  Without this augmentation, the
./packages/core/legoesm/parallel/sharded_dynamics.py-1503-        # local mesh's cellsOnEdge has -1 entries for those edges
./packages/core/legoesm/parallel/sharded_dynamics.py-1504-        # after remap → ``gradient_edge`` does ``phi[-1]`` (Python
./packages/core/legoesm/parallel/sharded_dynamics.py-1505-        # last-element indexing!) and produces 1e76 garbage
./packages/core/legoesm/parallel/sharded_dynamics.py-1506-        # within one SSP-RK3 step.  Iter-23 root-cause analysis.
./packages/core/legoesm/parallel/sharded_dynamics.py-1507-        owned_edge_cells = cellsOnEdge_np[:, owned_edges].reshape(-1)
./packages/core/legoesm/parallel/sharded_dynamics.py-1508-        owned_edge_cells = owned_edge_cells[owned_edge_cells >= 0]
./packages/core/legoesm/parallel/sharded_dynamics.py-1509-        for c in owned_edge_cells:
./packages/core/legoesm/parallel/sharded_dynamics.py-1510-            ic = int(c)
./scripts/bench/bench_voronoi_partition_methods.py-15-   scaling above ~64 devices, and layer 1 CANNOT stand in for it: the
./scripts/bench/bench_voronoi_partition_methods.py-16-   neighbor fan-out above is a 1-ring proxy that reported 8 rounds for
./scripts/bench/bench_voronoi_partition_methods.py-17-   every method and rank count while the real depth-3-plus-closure
./scripts/bench/bench_voronoi_partition_methods.py-18-   schedule reported 12-14.  Scored by the production
./scripts/bench/bench_voronoi_partition_methods.py-19-   ``spmd_schedule_cost`` (which calls the production builders), never a
./scripts/bench/bench_voronoi_partition_methods.py-20-   re-derived lookalike.
./scripts/bench/bench_voronoi_partition_methods.py-21-
./scripts/bench/bench_voronoi_partition_methods.py-22-   The DECISIVE column is ``coloring_gap = n_rounds - max_degree``, read
./scripts/bench/bench_voronoi_partition_methods.py-23-   through VIZING'S THEOREM, which bounds what recolouring could ever buy.
./scripts/bench/bench_voronoi_partition_methods.py-24-   The schedule is a proper EDGE colouring of the device communication
./scripts/bench/bench_voronoi_partition_methods.py:25:   graph (one colour = one ppermute round; ``_build_ppermute_schedule``
./scripts/bench/bench_voronoi_partition_methods.py-26-   asserts properness), and ``max_degree`` is that same graph's maximum
./scripts/bench/bench_voronoi_partition_methods.py-27-   vertex degree.  So the chromatic index obeys ``Delta <= chi' <=
./scripts/bench/bench_voronoi_partition_methods.py-28-   Delta + 1``: the gap is a bound on recolouring headroom, NOT a
./scripts/bench/bench_voronoi_partition_methods.py-29-   yes/no flag.
./scripts/bench/bench_voronoi_partition_methods.py-30-
./scripts/bench/bench_voronoi_partition_methods.py-31-   * ``gap == 0`` -> ``n_rounds == Delta``, and no proper edge colouring
./scripts/bench/bench_voronoi_partition_methods.py-32-     can beat ``Delta``.  The colouring is PROVABLY OPTIMAL; recolouring
./scripts/bench/bench_voronoi_partition_methods.py-33-     headroom is exactly ZERO.
./scripts/bench/bench_voronoi_partition_methods.py-34-   * ``gap == 1`` -> INCONCLUSIVE.  A Class 2 graph genuinely needs
./scripts/bench/bench_voronoi_partition_methods.py-35-     ``Delta + 1``, and deciding Class 1 vs Class 2 is NP-complete, so
./scripts/bench/bench_voronoi_partition_methods.py-36-     this neither establishes nor excludes a one-round win.
./scripts/bench/bench_voronoi_partition_methods.py-37-   * ``gap >= 2`` -> recolouring is guaranteed to remove AT LEAST
./scripts/bench/bench_voronoi_partition_methods.py-38-     ``gap - 1`` rounds (the optimum is at worst ``Delta + 1``) and at
./scripts/bench/bench_voronoi_partition_methods.py-39-     most ``gap``.
./scripts/bench/bench_voronoi_partition_methods.py-40-
./scripts/bench/bench_voronoi_partition_methods.py-41-   SCOPE, and it is not a formality: ``Delta`` bounds only a proper
./scripts/bench/bench_voronoi_partition_methods.py:42:   UNDIRECTED edge colouring of THIS graph.  ``_build_ppermute_schedule``
./scripts/bench/bench_voronoi_partition_methods.py:43:   enters a device pair into ``comm_pairs`` when EITHER direction has a
./scripts/bench/bench_voronoi_partition_methods.py-44-   halo dependency and then emits BOTH ppermute directions, even where one
./scripts/bench/bench_voronoi_partition_methods.py-45-   send map is empty.  A redesigned DIRECTED schedule that exploits
./scripts/bench/bench_voronoi_partition_methods.py-46-   one-way exchanges is therefore not bounded by ``Delta`` at all, so
./scripts/bench/bench_voronoi_partition_methods.py-47-   ``gap == 0`` must never be reported as "only ownership can help" — it
./scripts/bench/bench_voronoi_partition_methods.py-48-   rules out a better undirected edge colouring of this graph, and
./scripts/bench/bench_voronoi_partition_methods.py-49-   nothing more.
./scripts/bench/bench_voronoi_partition_methods.py-50-
./scripts/bench/bench_voronoi_partition_methods.py-51-   NOTE the halo depth differs by lane and is NOT ``--halo-depth``: that
./scripts/bench/bench_voronoi_partition_methods.py-52-   flag is the MPI lane's (default 2), while the schedule is scored at the
./scripts/bench/bench_voronoi_partition_methods.py-53-   SPMD production depth ``SPMD_HALO_DEPTH`` (3).  Both are recorded.
--
./tests/parallel/test_ppermute_edge_coloring.py-1-"""Edge-coloring correctness + round-count for the route-B MPAS ppermute
./tests/parallel/test_ppermute_edge_coloring.py:2:schedule (:func:`legoesm.parallel.sharded_dynamics._build_ppermute_schedule`).
./tests/parallel/test_ppermute_edge_coloring.py-3-
./tests/parallel/test_ppermute_edge_coloring.py-4-Each color is one bidirectional ppermute ROUND and the route-B lane is
./tests/parallel/test_ppermute_edge_coloring.py-5-round-latency-bound (#1113), so the schedule builder picks the coloring
./tests/parallel/test_ppermute_edge_coloring.py-6-with the fewest rounds across several deterministic first-fit orderings
./tests/parallel/test_ppermute_edge_coloring.py-7-(``_multi_ordering_edge_coloring``). These tests pin:
./tests/parallel/test_ppermute_edge_coloring.py-8-
./tests/parallel/test_ppermute_edge_coloring.py-9-- every candidate coloring is PROPER (no device sends/receives twice in a
./tests/parallel/test_ppermute_edge_coloring.py-10-  round) — a wrong coloring would collide two exchanges at a device;
./tests/parallel/test_ppermute_edge_coloring.py-11-- the multi-start coloring never uses MORE rounds than the legacy sorted
./tests/parallel/test_ppermute_edge_coloring.py-12-  greedy (never-regress), and reaches the chromatic-index floor
--
./tests/parallel/test_ppermute_edge_coloring.py-94-
./tests/parallel/test_ppermute_edge_coloring.py-95-def test_schedule_reaches_floor_on_real_mesh():
./tests/parallel/test_ppermute_edge_coloring.py-96-    """End-to-end through the schedule builder on a real reordered MPAS
./tests/parallel/test_ppermute_edge_coloring.py-97-    mesh where sorted greedy overshoots: the produced schedule must reach
./tests/parallel/test_ppermute_edge_coloring.py-98-    the max_degree floor and stay proper (the round count feeds the
./tests/parallel/test_ppermute_edge_coloring.py-99-    ppermute loop directly)."""
./tests/parallel/test_ppermute_edge_coloring.py-100-    pytest.importorskip("jax")
./tests/parallel/test_ppermute_edge_coloring.py-101-
./tests/parallel/test_ppermute_edge_coloring.py-102-    from legoesm.grids.voronoi import create_voronoi_mesh
./tests/parallel/test_ppermute_edge_coloring.py-103-    from legoesm.parallel.sharded_dynamics import (
./tests/parallel/test_ppermute_edge_coloring.py:104:        _build_ppermute_schedule,
./tests/parallel/test_ppermute_edge_coloring.py-105-        _build_voronoi_partition_infra,
./tests/parallel/test_ppermute_edge_coloring.py-106-    )
./tests/parallel/test_ppermute_edge_coloring.py-107-    from legoesm.parallel.voronoi_partition import (
./tests/parallel/test_ppermute_edge_coloring.py-108-        reorder_voronoi_for_sharding,
./tests/parallel/test_ppermute_edge_coloring.py-109-    )
./tests/parallel/test_ppermute_edge_coloring.py-110-
./tests/parallel/test_ppermute_edge_coloring.py-111-    n_dev = 16
./tests/parallel/test_ppermute_edge_coloring.py-112-    mesh = create_voronoi_mesh(subdivision_level=3)
./tests/parallel/test_ppermute_edge_coloring.py-113-    mesh = reorder_voronoi_for_sharding(mesh, n_dev, method="auto")
./tests/parallel/test_ppermute_edge_coloring.py-114-    if mesh.nCells % n_dev or mesh.nEdges % n_dev:
./tests/parallel/test_ppermute_edge_coloring.py-115-        pytest.skip("mesh not divisible by n_dev")
./tests/parallel/test_ppermute_edge_coloring.py-116-    cells_per, edges_per = mesh.nCells // n_dev, mesh.nEdges // n_dev
./tests/parallel/test_ppermute_edge_coloring.py-117-    (_sm, _gc, _ge, _noc, _noe, max_lc, max_le, partitions,
./tests/parallel/test_ppermute_edge_coloring.py-118-     cell_owner) = _build_voronoi_partition_infra(mesh, n_dev, halo_depth=3)
./tests/parallel/test_ppermute_edge_coloring.py:119:    sched = _build_ppermute_schedule(
./tests/parallel/test_ppermute_edge_coloring.py-120-        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
./tests/parallel/test_ppermute_edge_coloring.py-121-    )
./tests/parallel/test_ppermute_edge_coloring.py-122-    # sorted greedy overshoots here (17); multi-start reaches the floor (14).
./tests/parallel/test_ppermute_edge_coloring.py-123-    assert sched["n_rounds"] == sched["max_degree"], (
./tests/parallel/test_ppermute_edge_coloring.py-124-        f"expected chromatic-index floor {sched['max_degree']}, "
./tests/parallel/test_ppermute_edge_coloring.py-125-        f"got {sched['n_rounds']}")
./tests/parallel/test_ppermute_edge_coloring.py-126-    assert sched["n_rounds"] < sched["n_rounds_greedy"], (
./tests/parallel/test_ppermute_edge_coloring.py-127-        "multi-start coloring should beat sorted greedy on this mesh")
./tests/parallel/test_ppermute_edge_coloring.py-128-    assert sched["coloring_method"] == "multi_greedy"
./tests/parallel/test_ppermute_edge_coloring.py-129-    # ppermute perms are proper: no device appears twice as a source or
--
./tests/distributed/test_voronoi_halo.py-406-    """Tests for the ppermute-based halo exchange schedule builder."""
./tests/distributed/test_voronoi_halo.py-407-
./tests/distributed/test_voronoi_halo.py-408-    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
./tests/distributed/test_voronoi_halo.py-409-    def test_schedule_covers_all_halo_cells(self, mesh, n_ranks):
./tests/distributed/test_voronoi_halo.py-410-        """Every halo cell is covered by exactly one ppermute round."""
./tests/distributed/test_voronoi_halo.py-411-        from legoesm.parallel.voronoi_partition import (
./tests/distributed/test_voronoi_halo.py-412-            reorder_voronoi_for_sharding,
./tests/distributed/test_voronoi_halo.py-413-        )
./tests/distributed/test_voronoi_halo.py-414-        from legoesm.parallel.sharded_dynamics import (
./tests/distributed/test_voronoi_halo.py-415-            _build_voronoi_partition_infra,
./tests/distributed/test_voronoi_halo.py:416:            _build_ppermute_schedule,
./tests/distributed/test_voronoi_halo.py-417-        )
./tests/distributed/test_voronoi_halo.py-418-
./tests/distributed/test_voronoi_halo.py-419-        reordered = reorder_voronoi_for_sharding(mesh, n_ranks)
./tests/distributed/test_voronoi_halo.py-420-        nCells = reordered.nCells
./tests/distributed/test_voronoi_halo.py-421-        nEdges = reordered.nEdges
./tests/distributed/test_voronoi_halo.py-422-        cells_per = nCells // n_ranks
./tests/distributed/test_voronoi_halo.py-423-        edges_per = nEdges // n_ranks
./tests/distributed/test_voronoi_halo.py-424-
./tests/distributed/test_voronoi_halo.py-425-        (_, _, _, _, _, max_lc, max_le, partitions, cell_owner,
./tests/distributed/test_voronoi_halo.py-426-         ) = _build_voronoi_partition_infra(reordered, n_ranks, halo_depth=2)
./tests/distributed/test_voronoi_halo.py-427-
./tests/distributed/test_voronoi_halo.py:428:        sched = _build_ppermute_schedule(
./tests/distributed/test_voronoi_halo.py-429-            partitions, cell_owner, n_ranks,
./tests/distributed/test_voronoi_halo.py-430-            cells_per, edges_per, max_lc, max_le,
./tests/distributed/test_voronoi_halo.py-431-        )
./tests/distributed/test_voronoi_halo.py-432-
./tests/distributed/test_voronoi_halo.py-433-        # For each device, collect all halo cell positions reached
./tests/distributed/test_voronoi_halo.py-434-        for d in range(n_ranks):
./tests/distributed/test_voronoi_halo.py-435-            part = partitions[d]
./tests/distributed/test_voronoi_halo.py-436-            expected_halo_positions = set(range(
./tests/distributed/test_voronoi_halo.py-437-                part.n_owned_cells, part.n_local_cells))
./tests/distributed/test_voronoi_halo.py-438-
--
./tests/distributed/test_voronoi_halo.py-452-
./tests/distributed/test_voronoi_halo.py-453-    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
./tests/distributed/test_voronoi_halo.py-454-    def test_ppermute_halo_matches_simulated_exchange(self, mesh, n_ranks):
./tests/distributed/test_voronoi_halo.py-455-        """ppermute schedule produces same halo values as simulated
./tests/distributed/test_voronoi_halo.py-456-        exchange."""
./tests/distributed/test_voronoi_halo.py-457-        from legoesm.parallel.voronoi_partition import (
./tests/distributed/test_voronoi_halo.py-458-            reorder_voronoi_for_sharding,
./tests/distributed/test_voronoi_halo.py-459-        )
./tests/distributed/test_voronoi_halo.py-460-        from legoesm.parallel.sharded_dynamics import (
./tests/distributed/test_voronoi_halo.py-461-            _build_voronoi_partition_infra,
./tests/distributed/test_voronoi_halo.py:462:            _build_ppermute_schedule,
./tests/distributed/test_voronoi_halo.py-463-        )
./tests/distributed/test_voronoi_halo.py-464-
./tests/distributed/test_voronoi_halo.py-465-        reordered = reorder_voronoi_for_sharding(mesh, n_ranks)
./tests/distributed/test_voronoi_halo.py-466-        nCells = reordered.nCells
./tests/distributed/test_voronoi_halo.py-467-        cells_per = nCells // n_ranks
./tests/distributed/test_voronoi_halo.py-468-        edges_per = reordered.nEdges // n_ranks
./tests/distributed/test_voronoi_halo.py-469-
./tests/distributed/test_voronoi_halo.py-470-        (_, _, _, _, _, max_lc, max_le, partitions, cell_owner,
./tests/distributed/test_voronoi_halo.py-471-         ) = _build_voronoi_partition_infra(reordered, n_ranks, halo_depth=2)
./tests/distributed/test_voronoi_halo.py-472-
./tests/distributed/test_voronoi_halo.py:473:        sched = _build_ppermute_schedule(
./tests/distributed/test_voronoi_halo.py-474-            partitions, cell_owner, n_ranks,
./tests/distributed/test_voronoi_halo.py-475-            cells_per, edges_per, max_lc, max_le,
./tests/distributed/test_voronoi_halo.py-476-        )
./tests/distributed/test_voronoi_halo.py-477-
./tests/distributed/test_voronoi_halo.py-478-        # Use global index as value for easy verification
./tests/distributed/test_voronoi_halo.py-479-        phi_global = jnp.arange(nCells, dtype=jnp.float64)
./tests/distributed/test_voronoi_halo.py-480-
./tests/distributed/test_voronoi_halo.py-481-        for d in range(n_ranks):
./tests/distributed/test_voronoi_halo.py-482-            part = partitions[d]
./tests/distributed/test_voronoi_halo.py-483-            owned_data = phi_global[d * cells_per:(d + 1) * cells_per]
--
./tests/distributed/test_voronoi_halo.py-519-                )
./tests/distributed/test_voronoi_halo.py-520-
./tests/distributed/test_voronoi_halo.py-521-    @pytest.mark.parametrize("n_ranks", [2, 4])
./tests/distributed/test_voronoi_halo.py-522-    def test_edge_coloring_valid(self, mesh, n_ranks):
./tests/distributed/test_voronoi_halo.py-523-        """Each ppermute round has no device appearing as sender twice."""
./tests/distributed/test_voronoi_halo.py-524-        from legoesm.parallel.voronoi_partition import (
./tests/distributed/test_voronoi_halo.py-525-            reorder_voronoi_for_sharding,
./tests/distributed/test_voronoi_halo.py-526-        )
./tests/distributed/test_voronoi_halo.py-527-        from legoesm.parallel.sharded_dynamics import (
./tests/distributed/test_voronoi_halo.py-528-            _build_voronoi_partition_infra,
./tests/distributed/test_voronoi_halo.py:529:            _build_ppermute_schedule,
./tests/distributed/test_voronoi_halo.py-530-        )
./tests/distributed/test_voronoi_halo.py-531-
./tests/distributed/test_voronoi_halo.py-532-        reordered = reorder_voronoi_for_sharding(mesh, n_ranks)
./tests/distributed/test_voronoi_halo.py-533-        cells_per = reordered.nCells // n_ranks
./tests/distributed/test_voronoi_halo.py-534-        edges_per = reordered.nEdges // n_ranks
./tests/distributed/test_voronoi_halo.py-535-
./tests/distributed/test_voronoi_halo.py-536-        (_, _, _, _, _, max_lc, max_le, partitions, cell_owner,
./tests/distributed/test_voronoi_halo.py-537-         ) = _build_voronoi_partition_infra(reordered, n_ranks, halo_depth=2)
./tests/distributed/test_voronoi_halo.py-538-
./tests/distributed/test_voronoi_halo.py:539:        sched = _build_ppermute_schedule(
./tests/distributed/test_voronoi_halo.py-540-            partitions, cell_owner, n_ranks,
./tests/distributed/test_voronoi_halo.py-541-            cells_per, edges_per, max_lc, max_le,
./tests/distributed/test_voronoi_halo.py-542-        )
./tests/distributed/test_voronoi_halo.py-543-
./tests/distributed/test_voronoi_halo.py-544-        for r in range(sched['n_rounds']):
./tests/distributed/test_voronoi_halo.py-545-            senders = [s for s, _ in sched['ppermute_perms'][r]]
./tests/distributed/test_voronoi_halo.py-546-            assert len(senders) == len(set(senders)), (
./tests/distributed/test_voronoi_halo.py-547-                f"Round {r}: duplicate senders in ppermute perm"
./tests/distributed/test_voronoi_halo.py-548-            )
--
./tests/parallel/test_mpas_atm_native_step.py-536-        inserted in REVERSED order but packed in sorted wire order)
./tests/parallel/test_mpas_atm_native_step.py-537-        fails loudly.  ``n_dev=3`` exercises a multi-round schedule
./tests/parallel/test_mpas_atm_native_step.py-538-        (edge-colored comm graph); bases stay below 2^24 so the test
./tests/parallel/test_mpas_atm_native_step.py-539-        is exact in float32 too."""
./tests/parallel/test_mpas_atm_native_step.py-540-        _need_multi_device(n_dev)
./tests/parallel/test_mpas_atm_native_step.py-541-        from jax.sharding import Mesh, NamedSharding
./tests/parallel/test_mpas_atm_native_step.py-542-        from jax.sharding import PartitionSpec as P
./tests/parallel/test_mpas_atm_native_step.py-543-        from legoesm.grids.voronoi import create_voronoi_mesh
./tests/parallel/test_mpas_atm_native_step.py-544-        from legoesm.parallel.shard_map_compat import shard_map
./tests/parallel/test_mpas_atm_native_step.py-545-        from legoesm.parallel.sharded_dynamics import (
./tests/parallel/test_mpas_atm_native_step.py:546:            _build_ppermute_schedule,
./tests/parallel/test_mpas_atm_native_step.py-547-            _build_voronoi_partition_infra,
./tests/parallel/test_mpas_atm_native_step.py-548-            _pack_cell_state,
./tests/parallel/test_mpas_atm_native_step.py-549-            _ppermute_halo_fill,
./tests/parallel/test_mpas_atm_native_step.py-550-            _unpack_cell_state,
./tests/parallel/test_mpas_atm_native_step.py-551-        )
./tests/parallel/test_mpas_atm_native_step.py-552-        from legoesm.parallel.voronoi_partition import (
./tests/parallel/test_mpas_atm_native_step.py-553-            reorder_voronoi_for_sharding,
./tests/parallel/test_mpas_atm_native_step.py-554-        )
./tests/parallel/test_mpas_atm_native_step.py-555-
./tests/parallel/test_mpas_atm_native_step.py-556-        nlev = 3
./tests/parallel/test_mpas_atm_native_step.py-557-        mesh = create_voronoi_mesh(subdivision_level=2)
./tests/parallel/test_mpas_atm_native_step.py-558-        mesh = reorder_voronoi_for_sharding(mesh, n_dev)
./tests/parallel/test_mpas_atm_native_step.py-559-        nCells, nEdges = mesh.nCells, mesh.nEdges
./tests/parallel/test_mpas_atm_native_step.py-560-        cells_per, edges_per = nCells // n_dev, nEdges // n_dev
./tests/parallel/test_mpas_atm_native_step.py-561-
./tests/parallel/test_mpas_atm_native_step.py-562-        (_sm, _gc, _ge, _noc, _noe, max_lc, max_le, partitions,
./tests/parallel/test_mpas_atm_native_step.py-563-         cell_owner) = _build_voronoi_partition_infra(
./tests/parallel/test_mpas_atm_native_step.py-564-            mesh, n_dev, halo_depth=3)
./tests/parallel/test_mpas_atm_native_step.py:565:        sched = _build_ppermute_schedule(
./tests/parallel/test_mpas_atm_native_step.py-566-            partitions, cell_owner, n_dev, cells_per, edges_per,
./tests/parallel/test_mpas_atm_native_step.py-567-            max_lc, max_le)
./tests/parallel/test_mpas_atm_native_step.py-568-        assert sched["n_rounds"] >= 1, "no comm rounds — test is vacuous"
./tests/parallel/test_mpas_atm_native_step.py-569-        if n_dev >= 3:
./tests/parallel/test_mpas_atm_native_step.py-570-            # 3 mutually-adjacent partitions edge-color to >= 2 rounds:
./tests/parallel/test_mpas_atm_native_step.py-571-            # the multi-round scatter path is genuinely exercised.
./tests/parallel/test_mpas_atm_native_step.py-572-            assert sched["n_rounds"] >= 2, (
./tests/parallel/test_mpas_atm_native_step.py-573-                "expected a multi-round schedule at 3 devices")
./tests/parallel/test_mpas_atm_native_step.py-574-
./tests/parallel/test_mpas_atm_native_step.py-575-        # Named per-field sentinels: value[g, lev] = base + lev*1e4 + g.
--
./packages/core/legoesm/parallel/sharded_dynamics.py-1253-    """How much halo communication one ownership choice costs, computed offline.
./packages/core/legoesm/parallel/sharded_dynamics.py-1254-
./packages/core/legoesm/parallel/sharded_dynamics.py-1255-    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
./packages/core/legoesm/parallel/sharded_dynamics.py-1256-    ROUNDS one halo exchange needs -- the sequential collective launches that
./packages/core/legoesm/parallel/sharded_dynamics.py-1257-    dominate MPAS strong scaling above ~64 devices.  Runs on a laptop: no GPU,
./packages/core/legoesm/parallel/sharded_dynamics.py-1258-    no MPI, no benchmark job, so a split can be compared before it costs an
./packages/core/legoesm/parallel/sharded_dynamics.py-1259-    allocation.
./packages/core/legoesm/parallel/sharded_dynamics.py-1260-
./packages/core/legoesm/parallel/sharded_dynamics.py-1261-    It calls the SAME builders production calls
./packages/core/legoesm/parallel/sharded_dynamics.py-1262-    (:func:`_build_voronoi_partition_infra` then
./packages/core/legoesm/parallel/sharded_dynamics.py:1263:    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
./packages/core/legoesm/parallel/sharded_dynamics.py-1264-    different question: a 1-ring ``cellsOnEdge`` adjacency graph reports 8
./packages/core/legoesm/parallel/sharded_dynamics.py-1265-    rounds where the real depth-3-plus-closure graph reports 12-14.
./packages/core/legoesm/parallel/sharded_dynamics.py-1266-
./packages/core/legoesm/parallel/sharded_dynamics.py-1267-    WHAT THE NUMBER IS NOT
./packages/core/legoesm/parallel/sharded_dynamics.py-1268-    ----------------------
./packages/core/legoesm/parallel/sharded_dynamics.py-1269-    * ``n_rounds`` is per HALO FILL, not per model step.  A step costs
./packages/core/legoesm/parallel/sharded_dynamics.py-1270-      ``n_rounds`` x (tendency evaluations per step), which depends on the
./packages/core/legoesm/parallel/sharded_dynamics.py-1271-      configured integrator -- SSP-RK3 evaluates 3 times, but the MPAS default
./packages/core/legoesm/parallel/sharded_dynamics.py-1272-      is ``ssp_rk54_scan``.  Multiply with the integrator you actually run.
./packages/core/legoesm/parallel/sharded_dynamics.py-1273-    * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
./packages/core/legoesm/parallel/sharded_dynamics.py:1274:      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
./packages/core/legoesm/parallel/sharded_dynamics.py-1275-      keeps the best; equality is MEASURED (compare the returned
./packages/core/legoesm/parallel/sharded_dynamics.py-1276-      ``max_degree``), never assumed.  Do not claim "the colouring is already
./packages/core/legoesm/parallel/sharded_dynamics.py-1277-      optimal so only ownership can help" from this function.
./packages/core/legoesm/parallel/sharded_dynamics.py-1278-    * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
./packages/core/legoesm/parallel/sharded_dynamics.py-1279-      cells/device is below ``ppermute_cells_per_device_threshold``, in which
./packages/core/legoesm/parallel/sharded_dynamics.py-1280-      case there is no ppermute schedule and this number is counterfactual --
./packages/core/legoesm/parallel/sharded_dynamics.py-1281-      see the returned ``production_strategy``.
./packages/core/legoesm/parallel/sharded_dynamics.py-1282-
./packages/core/legoesm/parallel/sharded_dynamics.py-1283-    MESH STATE -- the one thing that silently invalidates the score
./packages/core/legoesm/parallel/sharded_dynamics.py-1284-    --------------------------------------------------------------
--
./packages/core/legoesm/parallel/sharded_dynamics.py-1362-            f"is padded for its reorder target"
./packages/core/legoesm/parallel/sharded_dynamics.py-1363-            f"{'' if already_reordered else f' ({target})'}, so scoring it at "
./packages/core/legoesm/parallel/sharded_dynamics.py-1364-            f"a device count that does not divide it silently mis-slices the "
./packages/core/legoesm/parallel/sharded_dynamics.py-1365-            f"owned blocks. Score at a device count that divides the prepared "
./packages/core/legoesm/parallel/sharded_dynamics.py-1366-            f"mesh.")
./packages/core/legoesm/parallel/sharded_dynamics.py-1367-    (
./packages/core/legoesm/parallel/sharded_dynamics.py-1368-        _stacked, _gc, _ge, _noc, _noe, max_lc, max_le, partitions, cell_owner,
./packages/core/legoesm/parallel/sharded_dynamics.py-1369-    ) = _build_voronoi_partition_infra(prepared, n_dev, halo_depth=halo_depth)
./packages/core/legoesm/parallel/sharded_dynamics.py-1370-    cells_per = n_cells // n_dev
./packages/core/legoesm/parallel/sharded_dynamics.py-1371-    edges_per = n_edges // n_dev
./packages/core/legoesm/parallel/sharded_dynamics.py:1372:    sched = _build_ppermute_schedule(
./packages/core/legoesm/parallel/sharded_dynamics.py-1373-        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
./packages/core/legoesm/parallel/sharded_dynamics.py-1374-    )
./packages/core/legoesm/parallel/sharded_dynamics.py-1375-    return {
./packages/core/legoesm/parallel/sharded_dynamics.py-1376-        "method": method,
./packages/core/legoesm/parallel/sharded_dynamics.py-1377-        "resolved_method": resolved,
./packages/core/legoesm/parallel/sharded_dynamics.py-1378-        "n_dev": n_dev,
./packages/core/legoesm/parallel/sharded_dynamics.py-1379-        # Unknown for a pre-reordered mesh: the ownership is baked in and the
./packages/core/legoesm/parallel/sharded_dynamics.py-1380-        # target that produced it is not recoverable from the mesh. Reporting
./packages/core/legoesm/parallel/sharded_dynamics.py-1381-        # n_dev there would assert something we did not verify.
./packages/core/legoesm/parallel/sharded_dynamics.py-1382-        "reorder_target": (None if already_reordered else
--
./packages/core/legoesm/parallel/sharded_dynamics.py-1650-        jnp.array(gather_edges),
./packages/core/legoesm/parallel/sharded_dynamics.py-1651-        n_owned_cells,
./packages/core/legoesm/parallel/sharded_dynamics.py-1652-        n_owned_edges,
./packages/core/legoesm/parallel/sharded_dynamics.py-1653-        max_lc,
./packages/core/legoesm/parallel/sharded_dynamics.py-1654-        max_le,
./packages/core/legoesm/parallel/sharded_dynamics.py-1655-        partitions,
./packages/core/legoesm/parallel/sharded_dynamics.py-1656-        cell_owner,
./packages/core/legoesm/parallel/sharded_dynamics.py-1657-    )
./packages/core/legoesm/parallel/sharded_dynamics.py-1658-
./packages/core/legoesm/parallel/sharded_dynamics.py-1659-
./packages/core/legoesm/parallel/sharded_dynamics.py:1660:def _greedy_edge_coloring_ordered(comm_pairs, order):
./packages/core/legoesm/parallel/sharded_dynamics.py-1661-    """First-fit edge coloring visiting ``order`` (a list of normalized
./packages/core/legoesm/parallel/sharded_dynamics.py-1662-    ``(min,max)`` pairs). Always a PROPER coloring; the color count depends
./packages/core/legoesm/parallel/sharded_dynamics.py-1663-    on the visitation order.
./packages/core/legoesm/parallel/sharded_dynamics.py-1664-    """
./packages/core/legoesm/parallel/sharded_dynamics.py-1665-    from collections import defaultdict
./packages/core/legoesm/parallel/sharded_dynamics.py-1666-
./packages/core/legoesm/parallel/sharded_dynamics.py-1667-    vertex_colors: dict[int, set[int]] = defaultdict(set)
./packages/core/legoesm/parallel/sharded_dynamics.py-1668-    edge_colors: dict[tuple[int, int], int] = {}
./packages/core/legoesm/parallel/sharded_dynamics.py-1669-    for u, v in order:
./packages/core/legoesm/parallel/sharded_dynamics.py-1670-        used = vertex_colors[u] | vertex_colors[v]
./packages/core/legoesm/parallel/sharded_dynamics.py-1671-        color = 0
./packages/core/legoesm/parallel/sharded_dynamics.py-1672-        while color in used:
./packages/core/legoesm/parallel/sharded_dynamics.py-1673-            color += 1
./packages/core/legoesm/parallel/sharded_dynamics.py-1674-        edge_colors[(u, v)] = color
./packages/core/legoesm/parallel/sharded_dynamics.py-1675-        vertex_colors[u].add(color)
./packages/core/legoesm/parallel/sharded_dynamics.py-1676-        vertex_colors[v].add(color)
./packages/core/legoesm/parallel/sharded_dynamics.py-1677-    return edge_colors
./packages/core/legoesm/parallel/sharded_dynamics.py-1678-
./packages/core/legoesm/parallel/sharded_dynamics.py-1679-
./packages/core/legoesm/parallel/sharded_dynamics.py:1680:def _greedy_edge_coloring(comm_pairs):
./packages/core/legoesm/parallel/sharded_dynamics.py-1681-    """Legacy first-fit coloring on sorted pairs (the reference/never-regress
./packages/core/legoesm/parallel/sharded_dynamics.py-1682-    baseline for :func:`_multi_ordering_edge_coloring`). Worst case
./packages/core/legoesm/parallel/sharded_dynamics.py-1683-    ``2*max_degree - 1`` colors — each color is one ppermute ROUND, and the
./packages/core/legoesm/parallel/sharded_dynamics.py-1684-    route-B MPAS lane is round-latency-bound (#1113), so excess colors are
./packages/core/legoesm/parallel/sharded_dynamics.py-1685-    pure wall-clock.
./packages/core/legoesm/parallel/sharded_dynamics.py-1686-    """
./packages/core/legoesm/parallel/sharded_dynamics.py:1687:    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
./packages/core/legoesm/parallel/sharded_dynamics.py:1688:    return _greedy_edge_coloring_ordered(comm_pairs, edges)
./packages/core/legoesm/parallel/sharded_dynamics.py-1689-
./packages/core/legoesm/parallel/sharded_dynamics.py-1690-
./packages/core/legoesm/parallel/sharded_dynamics.py:1691:def _check_proper_edge_coloring(edge_colors, comm_pairs):
./packages/core/legoesm/parallel/sharded_dynamics.py-1692-    """Every pair colored, and no vertex sees a color twice."""
./packages/core/legoesm/parallel/sharded_dynamics.py-1693-    from collections import defaultdict
./packages/core/legoesm/parallel/sharded_dynamics.py-1694-
./packages/core/legoesm/parallel/sharded_dynamics.py:1695:    if set(edge_colors) != {tuple(sorted(p)) for p in comm_pairs}:
./packages/core/legoesm/parallel/sharded_dynamics.py-1696-        return False
./packages/core/legoesm/parallel/sharded_dynamics.py-1697-    seen: dict[int, set[int]] = defaultdict(set)
./packages/core/legoesm/parallel/sharded_dynamics.py-1698-    for (u, v), c in edge_colors.items():
./packages/core/legoesm/parallel/sharded_dynamics.py-1699-        if c in seen[u] or c in seen[v]:
./packages/core/legoesm/parallel/sharded_dynamics.py-1700-            return False
./packages/core/legoesm/parallel/sharded_dynamics.py-1701-        seen[u].add(c)
./packages/core/legoesm/parallel/sharded_dynamics.py-1702-        seen[v].add(c)
./packages/core/legoesm/parallel/sharded_dynamics.py-1703-    return True
./packages/core/legoesm/parallel/sharded_dynamics.py-1704-
./packages/core/legoesm/parallel/sharded_dynamics.py-1705-
./packages/core/legoesm/parallel/sharded_dynamics.py-1706-# Fixed shuffle seeds for the multi-start greedy edge coloring below —
./packages/core/legoesm/parallel/sharded_dynamics.py-1707-# a constant so every MPI rank / process builds the byte-identical
./packages/core/legoesm/parallel/sharded_dynamics.py-1708-# schedule (the coloring must agree across ranks or the ppermute pattern
./packages/core/legoesm/parallel/sharded_dynamics.py-1709-# desynchronises). NOT Math.random / device randomness: this is host-side
./packages/core/legoesm/parallel/sharded_dynamics.py-1710-# schedule construction, deterministic by seed.
./packages/core/legoesm/parallel/sharded_dynamics.py-1711-_COLORING_SHUFFLE_SEEDS = tuple(range(16))
./packages/core/legoesm/parallel/sharded_dynamics.py-1712-
./packages/core/legoesm/parallel/sharded_dynamics.py-1713-
./packages/core/legoesm/parallel/sharded_dynamics.py:1714:def _multi_ordering_edge_coloring(comm_pairs):
./packages/core/legoesm/parallel/sharded_dynamics.py-1715-    """Proper edge coloring via multi-start first-fit; returns the coloring
./packages/core/legoesm/parallel/sharded_dynamics.py-1716-    using the FEWEST colors (= ppermute rounds) across several deterministic
./packages/core/legoesm/parallel/sharded_dynamics.py-1717-    visitation orders.
./packages/core/legoesm/parallel/sharded_dynamics.py-1718-
./packages/core/legoesm/parallel/sharded_dynamics.py-1719-    First-fit greedy is order-sensitive: on the reordered MPAS comm graphs
./packages/core/legoesm/parallel/sharded_dynamics.py-1720-    the sorted order can overshoot the chromatic index by up to 3 rounds at
./packages/core/legoesm/parallel/sharded_dynamics.py-1721-    16 devices, while a degree-descending or shuffled order reaches the
./packages/core/legoesm/parallel/sharded_dynamics.py-1722-    ``max_degree`` lower bound (verified optimal on ico subdivisions 3–5 ×
./packages/core/legoesm/parallel/sharded_dynamics.py-1723-    {4,8,16} devices, auto/sfc partitions). Every candidate is a proper
./packages/core/legoesm/parallel/sharded_dynamics.py-1724-    coloring by construction, so taking the min can NEVER produce an
./packages/core/legoesm/parallel/sharded_dynamics.py-1725-    invalid schedule and can never regress below the legacy sorted greedy.
./packages/core/legoesm/parallel/sharded_dynamics.py-1726-
./packages/core/legoesm/parallel/sharded_dynamics.py-1727-    Deterministic across ranks (sorted + degree orders + fixed-seed
./packages/core/legoesm/parallel/sharded_dynamics.py-1728-    shuffles). Returns ``(edge_colors, max_degree)``.
./packages/core/legoesm/parallel/sharded_dynamics.py-1729-    """
./packages/core/legoesm/parallel/sharded_dynamics.py-1730-    import random
./packages/core/legoesm/parallel/sharded_dynamics.py-1731-    from collections import defaultdict
./packages/core/legoesm/parallel/sharded_dynamics.py-1732-
./packages/core/legoesm/parallel/sharded_dynamics.py:1733:    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
./packages/core/legoesm/parallel/sharded_dynamics.py-1734-    deg: dict[int, int] = defaultdict(int)
./packages/core/legoesm/parallel/sharded_dynamics.py-1735-    for u, v in edges:
./packages/core/legoesm/parallel/sharded_dynamics.py-1736-        deg[u] += 1
./packages/core/legoesm/parallel/sharded_dynamics.py-1737-        deg[v] += 1
./packages/core/legoesm/parallel/sharded_dynamics.py-1738-    max_degree = max(deg.values(), default=0)
./packages/core/legoesm/parallel/sharded_dynamics.py-1739-
./packages/core/legoesm/parallel/sharded_dynamics.py-1740-    orders = [
./packages/core/legoesm/parallel/sharded_dynamics.py-1741-        edges,                                                   # sorted
./packages/core/legoesm/parallel/sharded_dynamics.py-1742-        sorted(edges, key=lambda e: -(deg[e[0]] + deg[e[1]])),   # sum-deg desc
./packages/core/legoesm/parallel/sharded_dynamics.py-1743-        sorted(edges, key=lambda e: -max(deg[e[0]], deg[e[1]])),  # max-deg desc
./packages/core/legoesm/parallel/sharded_dynamics.py-1744-    ]
./packages/core/legoesm/parallel/sharded_dynamics.py-1745-    for seed in _COLORING_SHUFFLE_SEEDS:
./packages/core/legoesm/parallel/sharded_dynamics.py-1746-        shuffled = edges[:]
./packages/core/legoesm/parallel/sharded_dynamics.py-1747-        random.Random(seed).shuffle(shuffled)
./packages/core/legoesm/parallel/sharded_dynamics.py-1748-        orders.append(shuffled)
./packages/core/legoesm/parallel/sharded_dynamics.py-1749-
./packages/core/legoesm/parallel/sharded_dynamics.py-1750-    best_colors: dict[tuple[int, int], int] | None = None
./packages/core/legoesm/parallel/sharded_dynamics.py-1751-    best_rounds = None
./packages/core/legoesm/parallel/sharded_dynamics.py-1752-    for order in orders:
./packages/core/legoesm/parallel/sharded_dynamics.py:1753:        ec = _greedy_edge_coloring_ordered(comm_pairs, order)
./packages/core/legoesm/parallel/sharded_dynamics.py-1754-        rounds = max(ec.values(), default=-1) + 1
./packages/core/legoesm/parallel/sharded_dynamics.py-1755-        if best_rounds is None or rounds < best_rounds:
./packages/core/legoesm/parallel/sharded_dynamics.py-1756-            best_rounds, best_colors = rounds, ec
./packages/core/legoesm/parallel/sharded_dynamics.py-1757-            if best_rounds <= max_degree:
./packages/core/legoesm/parallel/sharded_dynamics.py-1758-                break            # hit the chromatic-index floor — optimal
./packages/core/legoesm/parallel/sharded_dynamics.py-1759-    return best_colors, max_degree
./packages/core/legoesm/parallel/sharded_dynamics.py-1760-
./packages/core/legoesm/parallel/sharded_dynamics.py-1761-
./packages/core/legoesm/parallel/sharded_dynamics.py:1762:def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
./packages/core/legoesm/parallel/sharded_dynamics.py-1763-                             edges_per, max_lc, max_le):
./packages/core/legoesm/parallel/sharded_dynamics.py-1764-    """Build a ppermute-based halo exchange schedule.
./packages/core/legoesm/parallel/sharded_dynamics.py-1765-
./packages/core/legoesm/parallel/sharded_dynamics.py-1766-    Instead of all-gathering the full state (O(N) communication),
./packages/core/legoesm/parallel/sharded_dynamics.py-1767-    this schedule uses ``jax.lax.ppermute`` to exchange only halo data
./packages/core/legoesm/parallel/sharded_dynamics.py-1768-    between neighboring devices.  The communication graph is edge-colored
./packages/core/legoesm/parallel/sharded_dynamics.py-1769-    so that each round of ppermute moves data between non-conflicting
./packages/core/legoesm/parallel/sharded_dynamics.py-1770-    pairs simultaneously.
./packages/core/legoesm/parallel/sharded_dynamics.py-1771-
./packages/core/legoesm/parallel/sharded_dynamics.py-1772-    Parameters
--
./packages/core/legoesm/parallel/sharded_dynamics.py-1805-            halo_cells_from[d][owner].append(g)
./packages/core/legoesm/parallel/sharded_dynamics.py-1806-
./packages/core/legoesm/parallel/sharded_dynamics.py-1807-        for h_idx in range(part.n_owned_edges, part.n_local_edges):
./packages/core/legoesm/parallel/sharded_dynamics.py-1808-            g = int(part.local_edges[h_idx])
./packages/core/legoesm/parallel/sharded_dynamics.py-1809-            owner = min(g // edges_per, n_dev - 1)
./packages/core/legoesm/parallel/sharded_dynamics.py-1810-            halo_edges_from[d][owner].append(g)
./packages/core/legoesm/parallel/sharded_dynamics.py-1811-
./packages/core/legoesm/parallel/sharded_dynamics.py-1812-    # ------------------------------------------------------------------
./packages/core/legoesm/parallel/sharded_dynamics.py-1813-    # 2. Build undirected communication graph
./packages/core/legoesm/parallel/sharded_dynamics.py-1814-    # ------------------------------------------------------------------
./packages/core/legoesm/parallel/sharded_dynamics.py:1815:    comm_pairs: set[tuple[int, int]] = set()
./packages/core/legoesm/parallel/sharded_dynamics.py-1816-    for d in range(n_dev):
./packages/core/legoesm/parallel/sharded_dynamics.py-1817-        for d_prime in halo_cells_from[d]:
./packages/core/legoesm/parallel/sharded_dynamics.py-1818-            if d != d_prime:
./packages/core/legoesm/parallel/sharded_dynamics.py:1819:                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
./packages/core/legoesm/parallel/sharded_dynamics.py-1820-        for d_prime in halo_edges_from[d]:
./packages/core/legoesm/parallel/sharded_dynamics.py-1821-            if d != d_prime:
./packages/core/legoesm/parallel/sharded_dynamics.py:1822:                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
./packages/core/legoesm/parallel/sharded_dynamics.py-1823-
./packages/core/legoesm/parallel/sharded_dynamics.py:1824:    if not comm_pairs:
./packages/core/legoesm/parallel/sharded_dynamics.py-1825-        return {
./packages/core/legoesm/parallel/sharded_dynamics.py-1826-            'n_rounds': 0,
./packages/core/legoesm/parallel/sharded_dynamics.py-1827-            'n_rounds_greedy': 0,
./packages/core/legoesm/parallel/sharded_dynamics.py-1828-            'max_degree': 0,
./packages/core/legoesm/parallel/sharded_dynamics.py-1829-            'coloring_method': 'none',
./packages/core/legoesm/parallel/sharded_dynamics.py-1830-            'ppermute_perms': [],
./packages/core/legoesm/parallel/sharded_dynamics.py-1831-            'send_cell_idx': [],
./packages/core/legoesm/parallel/sharded_dynamics.py-1832-            'recv_cell_pos': [],
./packages/core/legoesm/parallel/sharded_dynamics.py-1833-            'send_edge_idx': [],
./packages/core/legoesm/parallel/sharded_dynamics.py-1834-            'recv_edge_pos': [],
--
./packages/core/legoesm/parallel/sharded_dynamics.py-1839-    # ------------------------------------------------------------------
./packages/core/legoesm/parallel/sharded_dynamics.py-1840-    # 3. Edge-color the graph: each color = one bidirectional ppermute
./packages/core/legoesm/parallel/sharded_dynamics.py-1841-    #    ROUND, and the route-B lane is round-latency-bound (#1113), so
./packages/core/legoesm/parallel/sharded_dynamics.py-1842-    #    fewer colors = directly less wall-clock. First-fit greedy is
./packages/core/legoesm/parallel/sharded_dynamics.py-1843-    #    order-sensitive; the multi-start coloring reaches the
./packages/core/legoesm/parallel/sharded_dynamics.py-1844-    #    chromatic-index floor (= max_degree) on every probed MPAS config
./packages/core/legoesm/parallel/sharded_dynamics.py-1845-    #    where the legacy sorted greedy overshoots (up to 3 rounds at 16
./packages/core/legoesm/parallel/sharded_dynamics.py-1846-    #    devices). It can never regress: the legacy sorted order is one of
./packages/core/legoesm/parallel/sharded_dynamics.py-1847-    #    its candidates and it takes the min. Both are verified proper.
./packages/core/legoesm/parallel/sharded_dynamics.py-1848-    # ------------------------------------------------------------------
./packages/core/legoesm/parallel/sharded_dynamics.py:1849:    greedy_colors = _greedy_edge_coloring(comm_pairs)
./packages/core/legoesm/parallel/sharded_dynamics.py-1850-    n_rounds_greedy = max(greedy_colors.values()) + 1
./packages/core/legoesm/parallel/sharded_dynamics.py:1851:    multi_colors, max_degree = _multi_ordering_edge_coloring(comm_pairs)
./packages/core/legoesm/parallel/sharded_dynamics.py-1852-    n_rounds_multi = max(multi_colors.values()) + 1
./packages/core/legoesm/parallel/sharded_dynamics.py-1853-    # Adopt the multi-start coloring ONLY when it STRICTLY reduces rounds;
./packages/core/legoesm/parallel/sharded_dynamics.py-1854-    # on a tie keep the exact legacy sorted-greedy coloring so the produced
./packages/core/legoesm/parallel/sharded_dynamics.py-1855-    # schedule is byte-identical to before wherever there is no round win
./packages/core/legoesm/parallel/sharded_dynamics.py-1856-    # (the win only appears at high device counts — >=16 on the probed
./packages/core/legoesm/parallel/sharded_dynamics.py-1857-    # MPAS meshes). Both colorings are proper.
./packages/core/legoesm/parallel/sharded_dynamics.py-1858-    if n_rounds_multi < n_rounds_greedy:
./packages/core/legoesm/parallel/sharded_dynamics.py-1859-        edge_colors, n_rounds, coloring_method = (
./packages/core/legoesm/parallel/sharded_dynamics.py-1860-            multi_colors, n_rounds_multi, "multi_greedy")
./packages/core/legoesm/parallel/sharded_dynamics.py-1861-    else:
./packages/core/legoesm/parallel/sharded_dynamics.py-1862-        edge_colors, n_rounds, coloring_method = (
./packages/core/legoesm/parallel/sharded_dynamics.py-1863-            greedy_colors, n_rounds_greedy, "greedy")
./packages/core/legoesm/parallel/sharded_dynamics.py:1864:    assert _check_proper_edge_coloring(edge_colors, comm_pairs), (
./packages/core/legoesm/parallel/sharded_dynamics.py-1865-        "improper ppermute edge coloring — two same-round exchanges "
./packages/core/legoesm/parallel/sharded_dynamics.py-1866-        "would collide at a device")
./packages/core/legoesm/parallel/sharded_dynamics.py-1867-    rounds: dict[int, list[tuple[int, int]]] = defaultdict(list)
./packages/core/legoesm/parallel/sharded_dynamics.py-1868-    for (u, v), color in edge_colors.items():
./packages/core/legoesm/parallel/sharded_dynamics.py-1869-        rounds[color].append((u, v))
./packages/core/legoesm/parallel/sharded_dynamics.py-1870-
./packages/core/legoesm/parallel/sharded_dynamics.py-1871-    # ------------------------------------------------------------------
./packages/core/legoesm/parallel/sharded_dynamics.py-1872-    # 4. Build directed send/recv maps for each device pair
./packages/core/legoesm/parallel/sharded_dynamics.py-1873-    # ------------------------------------------------------------------
./packages/core/legoesm/parallel/sharded_dynamics.py-1874-    # cell_send_map[(src, dst)] = list of owned-local indices in src to send
--
./packages/core/legoesm/parallel/sharded_dynamics.py-2321-
./packages/core/legoesm/parallel/sharded_dynamics.py-2322-    # ------------------------------------------------------------------
./packages/core/legoesm/parallel/sharded_dynamics.py-2323-    # Strategy dispatch: ppermute (O(halo)) vs allgather (O(N))
./packages/core/legoesm/parallel/sharded_dynamics.py-2324-    # ------------------------------------------------------------------
./packages/core/legoesm/parallel/sharded_dynamics.py-2325-
./packages/core/legoesm/parallel/sharded_dynamics.py-2326-    use_ppermute = halo_strategy == "ppermute"
./packages/core/legoesm/parallel/sharded_dynamics.py-2327-
./packages/core/legoesm/parallel/sharded_dynamics.py-2328-    if use_ppermute:
./packages/core/legoesm/parallel/sharded_dynamics.py-2329-        # Build ppermute schedule: neighbor-only halo exchange
./packages/core/legoesm/parallel/sharded_dynamics.py-2330-        t1 = time.time()
./packages/core/legoesm/parallel/sharded_dynamics.py:2331:        pp_sched = _build_ppermute_schedule(
./packages/core/legoesm/parallel/sharded_dynamics.py-2332-            partitions_out, cell_owner_out, n_dev,
./packages/core/legoesm/parallel/sharded_dynamics.py-2333-            cells_per, edges_per, max_lc, max_le,
./packages/core/legoesm/parallel/sharded_dynamics.py-2334-        )
./packages/core/legoesm/parallel/sharded_dynamics.py-2335-        n_rounds = pp_sched['n_rounds']
./packages/core/legoesm/parallel/sharded_dynamics.py-2336-        ppermute_perms = pp_sched['ppermute_perms']
./packages/core/legoesm/parallel/sharded_dynamics.py-2337-
./packages/core/legoesm/parallel/sharded_dynamics.py-2338-        # LOCAL-ONLY metadata: shard the per-round index arrays on the
./packages/core/legoesm/parallel/sharded_dynamics.py-2339-        # leading device axis (each device holds only its own schedule
./packages/core/legoesm/parallel/sharded_dynamics.py-2340-        # rows) and thread them as shard_map ARGUMENTS — see the stacked
./packages/core/legoesm/parallel/sharded_dynamics.py-2341-        # meshes above for why args, not closures.
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
   189	    is exactly the failure this flag exists to prevent.
   190	    """
   191	    out: dict[tuple[str, int], int] = {}
   192	    for item in (s.strip() for s in spec.split(",")):
   193	        if not item:
   194	            continue
   195	        try:
   196	            lhs, rounds = item.split("=")
   197	            method, n_ranks = lhs.split(":")
   198	            key = (method.strip(), int(n_ranks))
   199	            out[key] = int(rounds)
   200	        except ValueError as exc:
   201	            raise ValueError(
   202	                f"--expect-rounds: cannot parse {item!r}; expected "
   203	                f"'<method>:<n_ranks>=<n_rounds>'") from exc
   204	        if key[0] not in METHODS:
   205	            raise ValueError(
   206	                f"--expect-rounds: unknown method {key[0]!r} in {item!r}; "
   207	                f"expected one of {METHODS}")
   208	    return out
   209	
   210	
   211	def check_expected_rounds(rows: list, expect: dict) -> list:
   212	    """Compare scored rounds against *expect*; return failure strings.
   213	
   214	    A listed pair that was never scored is a FAILURE, not a skip: otherwise
   215	    a sweep that silently dropped a method (unavailable ``pymetis``) or a
   216	    rank count would still report a clean gate.
   217	    """
   218	    scored = {
   219	        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
   220	        for r in rows
   221	        if r.get("available") and "schedule" in r and "n_ranks" in r
   222	    }
   223	    failures = []
   224	    for (method, n_ranks), want in sorted(expect.items()):
   225	        got = scored.get((method, n_ranks))
   226	        if got is None:
   227	            failures.append(
   228	                f"{method}:{n_ranks} expected rounds={want} but the pair was "
   229	                f"NOT SCORED (method unavailable, or not in this sweep)")
   230	        elif got != want:
   231	            failures.append(
   232	                f"{method}:{n_ranks} expected rounds={want}, got {got}")
   233	    return failures
   234	
   235	
   236	def schedule_cost_row(mesh, method: str, n_ranks: int) -> dict:
   237	    """SPMD halo-schedule depth for one (method, n_ranks) candidate.
   238	
   239	    Thin wrapper over the production
   240	    :func:`legoesm.parallel.sharded_dynamics.spmd_schedule_cost` — it reorders
   241	    the RAW mesh for ``n_ranks`` with ``method`` and colours the real
   242	    depth-``SPMD_HALO_DEPTH`` communication graph, so the number is the one
   243	    production pays, not a 1-ring lookalike.  ``halo_depth`` is deliberately
   244	    LEFT AT THE SCORER'S DEFAULT: this script's ``--halo-depth`` is the MPI
   245	    lane's (2), and scoring the SPMD schedule at 2 would colour a different
   246	    graph.
   247	
   248	    Adds ``coloring_gap = n_rounds - max_degree`` and the Vizing reading of
   249	    it (see the module docstring).  The headroom is reported as an INTERVAL,
   250	    because Vizing pins the optimum only to ``{Delta, Delta + 1}``:
   251	
   252	    * ``coloring_headroom_rounds_min = max(0, gap - 1)`` — rounds a perfect
   253	      recolouring is GUARANTEED to remove (it beats the ``Delta + 1`` case).
   254	    * ``coloring_headroom_rounds_max = gap`` — the best case, realized only
   255	      if the graph is Class 1.
   256	
   257	    It is deliberately NOT a "recolour vs ownership" verdict: at
   258	    ``gap == 1`` the min is 0 and the max is 1, i.e. genuinely inconclusive.
   259	
   260	    Errors are NOT caught.  The scorer's one refusal — a mesh padded for a
   261	    different reorder target, which would mis-slice the owned blocks — is
   262	    unreachable from here: this passes the raw mesh with the scorer's default
   263	    ``reorder_target = n_ranks``, and ``reorder_voronoi_for_sharding`` pads
   264	    ``nCells``/``nEdges`` to be divisible by exactly that target.  Wrapping
   265	    the call would therefore only swallow *unforeseen* failures into a row
   266	    that reads like an orderly skip, which is how a missing number turns into
   267	    a silently wrong table.  Rows already print as the sweep goes, so a raise
   268	    keeps the completed rungs in the log.
   269	    """
   270	    import time
   271	
   272	    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
   273	
   274	    t0 = time.perf_counter()
   275	    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
   276	    gap = int(cost["n_rounds"]) - int(cost["max_degree"])
   277	    return {
   278	        "n_rounds": int(cost["n_rounds"]),
   279	        "n_rounds_greedy": int(cost["n_rounds_greedy"]),
   280	        "max_degree": int(cost["max_degree"]),
   281	        # The decisive column, read through Vizing (see module docstring).
   282	        "coloring_gap": gap,
   283	        # Rounds a perfect recolouring could remove, as an INTERVAL: Vizing
   284	        # pins the optimum to {Delta, Delta+1}, so gap-1 is guaranteed and
   285	        # gap is the best case.  A gap of 1 spans [0, 1] = inconclusive.
   286	        "coloring_headroom_rounds_min": max(0, gap - 1),
   287	        "coloring_headroom_rounds_max": gap,
   288	        "coloring_optimal_proven": gap == 0,
   289	        # Scorer provenance, carried per row: a copied/flattened row must be
   290	        # able to show it scored a raw mesh partitioned for THIS device
   291	        # count, not one reordered for a different target.
   292	        "reorder_target": cost["reorder_target"],
   293	        "already_reordered": cost["already_reordered"],
   294	        "coloring_method": cost["coloring_method"],
   295	        "resolved_method": cost["resolved_method"],
   296	        "schedule_halo_depth": int(cost["halo_depth"]),
   297	        "cells_per_device": int(cost["cells_per_device"]),
   298	        # "allgather" => production runs no ppermute schedule here, so the
   299	        # round count above is COUNTERFACTUAL, not a cost production pays.
   300	        "production_strategy": cost["production_strategy"],
   301	        "score_seconds": round(time.perf_counter() - t0, 2),
   302	    }
   303	
   304	
   305	def main() -> int:
   306	    p = argparse.ArgumentParser(
   307	        description=__doc__,
   308	        formatter_class=argparse.RawDescriptionHelpFormatter)
   309	    p.add_argument("--subdivision", type=int, default=5,
   310	                   help="Icosahedral level (L5=10,242 cells; L6=40,962).")
   311	    p.add_argument("--rank-counts", type=str, default="2,4,8,16")
   312	    p.add_argument("--halo-depth", type=int, default=2,
   313	                   help="Halo layers (runtime default 2, del4 support).")
   314	    p.add_argument("--methods", type=str, default=",".join(METHODS))
   315	    p.add_argument("--schedule-cost", action="store_true",
   316	                   help="Also score the SPMD ppermute halo-schedule depth "
   317	                        "(n_rounds vs max_degree) per method x rank count. "
   318	                        "Uses the production SPMD halo depth, NOT "
   319	                        "--halo-depth. Expensive: minutes per candidate at "
   320	                        "subdiv>=8 — run it under batch.")
   321	    p.add_argument("--lloyd", type=int, default=50,
   322	                   help="Lloyd relaxation iterations for the mesh. 50 = the "
   323	                        "production SCVT key; 0 = the LABELLED synthetic "
   324	                        "scaling mesh. Recorded so a lloyd=0 mesh can never "
   325	                        "masquerade as a production receipt, and it must "
   326	                        "match the prewarmed cache key at subdiv>=9.")
   327	    p.add_argument("--expect-rounds", type=str, default="",
   328	                   help="Instrument check, MECHANICAL. Comma-separated "
   329	                        "'<method>:<n_ranks>=<n_rounds>' expectations (e.g. "
   330	                        "'sfc:64=12,sfc:128=14'). Every listed pair must be "
   331	                        "scored and match, or main() returns 1 — so a caller "
   332	                        "that reproduces a known census can GATE on it "
   333	                        "instead of asserting agreement in a comment. "
   334	                        "Requires --schedule-cost.")
   335	    p.add_argument("--out", type=str,
   336	                   default="results/a1/voronoi_partition_quality.json")
   337	    args = p.parse_args()
   338	
   339	    rank_counts = [int(x) for x in args.rank_counts.split(",") if x]
   340	    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
   341	    for m in methods:
   342	        if m not in METHODS:
   343	            raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
   344	    if not rank_counts or any(n < 2 for n in rank_counts):
   345	        raise SystemExit("--rank-counts needs integers >= 2")
   346	    expect = parse_expect_rounds(args.expect_rounds)
   347	    if expect and not args.schedule_cost:
   348	        raise SystemExit(
   349	            "--expect-rounds compares scored round counts, so it needs "
   350	            "--schedule-cost; without it nothing is scored and the gate "
   351	            "would pass vacuously.")
   352	
   353	
   354	    from legoesm.grids.voronoi import create_voronoi_mesh
   355	    from legoesm.parallel.voronoi_partition import resolve_partition_method
   356	
   357	    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
   358	                               lloyd_iterations=args.lloyd)
   359	    if max(rank_counts) > int(mesh.nCells):
   360	        raise SystemExit(
   361	            f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
   362	            f"{int(mesh.nCells)} cells (empty ranks are meaningless).")
   363	    print(f"mesh L{args.subdivision}: {int(mesh.nCells)} cells, "
   364	          f"{int(mesh.nEdges)} edges; auto -> "
   365	          f"{resolve_partition_method('auto')!r}")
   366	
   367	    rows = []
   368	    for method in methods:
   369	        if not method_available(method):
   370	            print(f"  {method:9s}: UNAVAILABLE (pymetis not importable) — "
   371	                  f"column omitted, never substituted")
   372	            rows.append({"method": method, "available": False})
   373	            continue
   374	        for n_ranks in rank_counts:
   375	            q = partition_quality(
   376	                mesh, owner_for(mesh, method, n_ranks), n_ranks,
   377	                halo_depth=args.halo_depth)
   378	            row = {"method": method, "available": True,
   379	                   "n_ranks": n_ranks, **q}
   380	            rows.append(row)
   381	            print(f"  {method:9s} np={n_ranks:3d} | "
   382	                  f"imbalance={q['load_imbalance_max_over_mean']:.3f} | "
   383	                  f"edge_cut={q['edge_cut']:6d} "
   384	                  f"({100 * q['edge_cut_fraction']:.2f}%) | "
   385	                  f"halo max={q['halo_cells_max']:5d} "
   386	                  f"mean={q['halo_cells_mean']:8.1f} | "
   387	                  f"halo/owned max={q['halo_owned_ratio_max']:.3f} | "
   388	                  f"nbrs max={q['neighbor_ranks_max']}")
   389	            if args.schedule_cost:
   390	                sc = schedule_cost_row(mesh, method, n_ranks)
   391	                row["schedule"] = sc
   392	                note = ("  [COUNTERFACTUAL: production auto-selects "
   393	                        "allgather here, no ppermute schedule]"
   394	                        if sc["production_strategy"] == "allgather" else "")
   395	                verdict = ("colouring PROVABLY OPTIMAL (Vizing floor)"
   396	                           if sc["coloring_optimal_proven"] else
   397	                           f"recolour headroom "
   398	                           f"{sc['coloring_headroom_rounds_min']}-"
   399	                           f"{sc['coloring_headroom_rounds_max']} round(s)")
   400	                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
   401	                      f"rounds={sc['n_rounds']:3d} "
   402	                      f"max_degree={sc['max_degree']:3d} "
   403	                      f"gap={sc['coloring_gap']:+d} "
   404	                      f"-> {verdict} "
   405	                      f"({sc['score_seconds']:.1f}s){note}", flush=True)
   406	
   407	    payload = {
   408	        "rows": rows,
   409	        "auto_resolves_to": resolve_partition_method("auto"),
   410	        "step_time_pointer": (
   411	            "step-time per method: bench_ocean_mpas_scaling.py / "
   412	            "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
   413	            "per launch, same case otherwise)"),
   414	        "metadata": annotate_incomplete(scaling_metadata(
   415	            grid="voronoi",
   416	            component="partitioning",
   417	            resolution=f"L{args.subdivision}",
   418	            n_levels=0,
   419	            precision="n/a",
   420	            decomposition="cell_partition",
   421	            solver_variant="n/a",
   422	            scaling_kind="partition-quality",
   423	            transport="none",
   424	            extra={"rank_counts": rank_counts, "methods": methods,
   425	                   "halo_depth": args.halo_depth,
   426	                   "schedule_cost": bool(args.schedule_cost),
   427	                   "lloyd_iterations": args.lloyd},
   428	        )),
   429	    }
   430	    outdir = os.path.dirname(args.out)
   431	    if outdir:
   432	        os.makedirs(outdir, exist_ok=True)
   433	    with open(args.out, "w") as f:
   434	        json.dump(payload, f, indent=2)
   435	    print(f"JSON: {args.out}")
   436	
   437	    if expect:
   438	        failures = check_expected_rounds(rows, expect)
   439	        payload["expected_rounds_check"] = {
   440	            "expected": {f"{m}:{n}": v for (m, n), v in sorted(expect.items())},
   441	            "failures": failures,
   442	            "passed": not failures,
   443	        }
   444	        with open(args.out, "w") as f:
   445	            json.dump(payload, f, indent=2)
   446	        if failures:
   447	            print(f"INSTRUMENT CHECK FAILED ({len(failures)}):")
   448	            for line in failures:
   449	                print(f"  {line}")
   450	            print("The scorer did NOT reproduce the known census — treat every "
   451	                  "unknown row in this run as UNTRUSTED.")
   452	            return 1
   453	        print(f"INSTRUMENT CHECK PASSED ({len(expect)} expectations)")
   454	    return 0
   455	
   456	
   457	if __name__ == "__main__":
   458	    raise SystemExit(main())

exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1240,1420p'
rg -n \"def _build_ppermute_schedule\" packages/core/legoesm/parallel/sharded_dynamics.py
nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '800,1100p'
rg -n -C 12 \"assert.*proper|proper.*color|comm_pairs|pair.*send|ppermute\" packages/core/legoesm/parallel/sharded_dynamics.py | head -n 800" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 1372ms:
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
  1411	
  1412	    Using ``partition_voronoi_mesh`` directly is unsuitable because it
  1413	    derives edge ownership from cell ownership, producing an uneven edge
  1414	    split that mismatches the even shard split.  Instead we construct the
  1415	    :class:`VoronoiPartition` objects manually with contiguous-block
  1416	    ownership for both cells **and** edges.
  1417	
  1418	    Returns
  1419	    -------
  1420	    stacked_meshes : VoronoiMesh
1762:def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
   800	            and "face" in getattr(config.mesh, 'axis_names', ())):
   801	        from legoesm.parallel.cubesphere_exchange import (
   802	            activate_spmd_halo_backend,
   803	        )
   804	        activate_spmd_halo_backend(config.mesh, n=n, nlev=nlev)
   805	        logger.info(
   806	            "make_sharded_step: activated SPMD halo backend "
   807	            "(%d devices, %s, n=%d, nlev=%d)",
   808	            config.n_devices,
   809	            "face-sharded" if _face_ok else f"tiled {_tiling}",
   810	            n, nlev,
   811	        )
   812	
   813	    logger.info(
   814	        "make_sharded_step: %d-device mode, tiling=%s",
   815	        config.n_devices, config.tiling,
   816	    )
   817	    return CompiledShardedStep(model, config, halo_exchange_fn)
   818	
   819	
   820	def sharded_step_with_halo(
   821	    model,
   822	    state,
   823	    dt: float,
   824	    config: DeviceConfig,
   825	    halo_exchange_fn=None,
   826	    physics_fn=None,
   827	):
   828	    """Execute one dynamics step with explicit halo exchange.
   829	
   830	    This is a functional (non-cached) variant of :func:`make_sharded_step`
   831	    useful for one-off calls or when the model/config may change between
   832	    steps.
   833	
   834	    Steps:
   835	
   836	    1. Run the dynamics step on each device's partition.
   837	    2. Apply halo exchange between partitions (either custom or
   838	       built-in via ``pad_halo``).
   839	    3. Return the updated sharded state.
   840	
   841	    Parameters
   842	    ----------
   843	    model
   844	        Dynamics model with ``.step()`` method.
   845	    state
   846	        Model state (should already be sharded via :func:`shard_state`).
   847	    dt : float
   848	        Time step [seconds].
   849	    config : DeviceConfig
   850	        Device configuration.
   851	    halo_exchange_fn : callable, optional
   852	        Custom halo exchange ``f(state) -> state``.
   853	    physics_fn : callable, optional
   854	        Physics forcing function.
   855	
   856	    Returns
   857	    -------
   858	    Updated state (same sharding as input).
   859	    """
   860	    if physics_fn is not None:
   861	        _refuse_stateful_physics_unthreaded_wrapper(physics_fn)
   862	
   863	    if config.mesh is None:
   864	        # Single device — no sharding needed
   865	        if physics_fn is not None and hasattr(model, "step_with_physics"):
   866	            return model.step_with_physics(state, dt, physics_fn)
   867	        return model.step(state, dt)
   868	
   869	    out_shardings = create_output_shardings(state, config)
   870	
   871	    @partial(jax.jit, out_shardings=out_shardings)
   872	    def _step(s):
   873	        if physics_fn is not None and hasattr(model, "step_with_physics"):
   874	            new_state = model.step_with_physics(s, dt, physics_fn)
   875	        else:
   876	            new_state = model.step(s, dt)
   877	
   878	        if halo_exchange_fn is not None:
   879	            new_state = halo_exchange_fn(new_state)
   880	
   881	        return new_state
   882	
   883	    return _step(state)
   884	
   885	
   886	# ======================================================================
   887	# Halo exchange utilities for cubed-sphere face boundaries
   888	# ======================================================================
   889	
   890	def make_face_halo_exchange(grid, config: DeviceConfig):
   891	    """Create a halo exchange function for cubed-sphere face boundaries.
   892	
   893	    The returned function operates on a full model state pytree and
   894	    applies halo exchange to all face-dimensioned arrays.
   895	
   896	    For face-only sharding (<=6 devices), this function is typically
   897	    not needed because the built-in ``pad_halo`` already handles
   898	    cross-face communication within the JIT'd step function.  It is
   899	    provided for explicit control when needed (e.g., in custom
   900	    time-stepping loops).
   901	
   902	    For sub-face tiling (>6 devices), this additionally exchanges
   903	    tile boundary data within each face.
   904	
   905	    Parameters
   906	    ----------
   907	    grid : CubedSphereGrid
   908	        The cubed-sphere grid (provides connectivity and metric info).
   909	    config : DeviceConfig
   910	        Device configuration.
   911	
   912	    Returns
   913	    -------
   914	    callable
   915	        ``exchange(state) -> state`` that applies halo exchange to
   916	        all face-dimensioned fields in the state pytree.
   917	    """
   918	    def _exchange(state):
   919	        """Apply halo exchange to face-dimensioned arrays.
   920	
   921	        This function pads each 2D face field ``(6, n, n)`` with halo
   922	        data from neighbors, then strips the halos back to ``(6, n, n)``.
   923	        This ensures boundary values are fresh after a dynamics step.
   924	
   925	        For 3D fields ``(6, n, n, nlev)`` the exchange uses the native
   926	        4D halo path (``pad_halo_4d``) which fetches halos for every
   927	        level in one MPI message — see CLAUDE.md ``Parallel and HPC
   928	        Rules``.  The previous ``vmap(pad_halo)`` per level pattern is
   929	        forbidden because it issues ``nlev`` separate messages.
   930	        """
   931	        def _exchange_leaf(leaf):
   932	            if not isinstance(leaf, (jax.Array, jnp.ndarray)):
   933	                return leaf
   934	            if leaf.ndim < 3 or leaf.shape[0] != N_FACES:
   935	                return leaf
   936	
   937	            if leaf.ndim == 3:
   938	                # 2D field: (6, n, n)
   939	                padded = pad_halo(leaf)
   940	                return padded[:, 1:-1, 1:-1]
   941	
   942	            elif leaf.ndim == 4:
   943	                # 3D field: (6, n, n, nlev) — single 4D halo exchange.
   944	                padded = pad_halo_4d(leaf)
   945	                return padded[:, 1:-1, 1:-1, :]
   946	
   947	            return leaf
   948	
   949	        return jax.tree.map(_exchange_leaf, state)
   950	
   951	    return _exchange
   952	
   953	
   954	# ======================================================================
   955	# Multi-step integration with sharding
   956	# ======================================================================
   957	
   958	def sharded_integrate(
   959	    model,
   960	    state,
   961	    n_steps: int,
   962	    dt: float,
   963	    config: DeviceConfig,
   964	    halo_exchange_fn=None,
   965	    physics_fn=None,
   966	    save_every: int = 0,
   967	    step_fn=None,
   968	):
   969	    """Integrate a sharded model forward for multiple steps.
   970	
   971	    This is a convenience wrapper that creates (or reuses) a sharded
   972	    step function and runs it in a loop, optionally saving intermediate
   973	    states.
   974	
   975	    Parameters
   976	    ----------
   977	    model
   978	        Dynamics model with ``.step()`` method.
   979	    state
   980	        Initial state (will be sharded if not already).
   981	    n_steps : int
   982	        Number of time steps.
   983	    dt : float
   984	        Time step [seconds].
   985	    config : DeviceConfig
   986	        Device configuration.
   987	    halo_exchange_fn : callable, optional
   988	        Custom halo exchange function.
   989	    physics_fn : callable, optional
   990	        Physics forcing function.
   991	    save_every : int
   992	        Save state every N steps.  0 means don't save intermediates.
   993	    step_fn : CompiledShardedStep, optional
   994	        Pre-built step function.  If ``None``, one is created via
   995	        :func:`make_sharded_step`.  Passing an existing ``step_fn``
   996	        avoids redundant object construction when calling
   997	        ``sharded_integrate`` multiple times with the same model.
   998	
   999	    Returns
  1000	    -------
  1001	    final_state : pytree
  1002	        Final state (still sharded).
  1003	    trajectory : list of pytree
  1004	        Saved intermediate states (sharded).  Empty if ``save_every=0``.
  1005	    """
  1006	    if step_fn is None:
  1007	        step_fn = make_sharded_step(model, config, halo_exchange_fn)
  1008	
  1009	    # Ensure state is sharded
  1010	    state = shard_state(state, config)
  1011	
  1012	    trajectory = []
  1013	    for i in range(n_steps):
  1014	        state = step_fn(state, dt, physics_fn)
  1015	        if save_every > 0 and (i + 1) % save_every == 0:
  1016	            trajectory.append(state)
  1017	
  1018	    return state, trajectory
  1019	
  1020	
  1021	def sharded_integrate_scan(
  1022	    model,
  1023	    state,
  1024	    n_steps: int,
  1025	    dt: float,
  1026	    config: DeviceConfig,
  1027	):
  1028	    """Integrate using ``jax.lax.scan`` for XLA fusion and differentiation.
  1029	
  1030	    This provides a fully JIT-compiled, differentiable integration loop.
  1031	    No intermediate states are saved (use ``sharded_integrate`` with
  1032	    ``save_every`` for that).
  1033	
  1034	    The entire loop is compiled as a single XLA program, enabling
  1035	    fusion across time steps and efficient reverse-mode differentiation.
  1036	
  1037	    Parameters
  1038	    ----------
  1039	    model
  1040	        Dynamics model with ``.step()`` method.
  1041	    state
  1042	        Initial state (will be sharded if not already).
  1043	    n_steps : int
  1044	        Number of time steps.
  1045	    dt : float
  1046	        Time step [seconds].
  1047	    config : DeviceConfig
  1048	        Device configuration.
  1049	
  1050	    Returns
  1051	    -------
  1052	    final_state : pytree
  1053	        Final state (sharded).
  1054	    """
  1055	    state = shard_state(state, config)
  1056	
  1057	    # Build a dtype-preserving step: some models promote float32 -> float64
  1058	    # when jax_enable_x64 is True, which breaks jax.lax.scan's type-matching
  1059	    # requirement.  We cast the output back to the input dtype tree.
  1060	    input_dtypes = jax.tree.map(
  1061	        lambda x: x.dtype if hasattr(x, "dtype") else None,
  1062	        state,
  1063	    )
  1064	
  1065	    def _dtype_safe_step(carry, _):
  1066	        new = model.step(carry, dt)
  1067	        # Cast each leaf back to its original dtype
  1068	        new = jax.tree.map(
  1069	            lambda x, d: x.astype(d) if d is not None and hasattr(x, "astype") else x,
  1070	            new,
  1071	            input_dtypes,
  1072	        )
  1073	        return new, None
  1074	
  1075	    if config.mesh is None:
  1076	        final, _ = jax.lax.scan(_dtype_safe_step, state, None, length=n_steps)
  1077	        return final
  1078	
  1079	    out_shardings = create_output_shardings(state, config)
  1080	
  1081	    @partial(jax.jit, out_shardings=out_shardings)
  1082	    def _scan_integrate(s):
  1083	        final, _ = jax.lax.scan(_dtype_safe_step, s, None, length=n_steps)
  1084	        return final
  1085	
  1086	    return _scan_integrate(state)
  1087	
  1088	
  1089	# ======================================================================
  1090	# Voronoi (MPAS) multi-GPU sharded step
  1091	# ======================================================================
  1092	
  1093	def _pad_local_mesh_to(mesh, target_nCells, target_nEdges, target_nVertices):
  1094	    """Pad a local VoronoiMesh to target dimensions with inert ghost entities.
  1095	
  1096	    Ghost cells have ``areaCell=1``, zero signs/weights, and connectivity
  1097	    pointing to index 0.  Ghost edges have ``dvEdge=0`` (zero flux),
  1098	    ``dcEdge=1``, and ``cellsOnEdge=[0,0]``.
  1099	    """
  1100	    from legoesm.grids.voronoi import VoronoiMesh
2-
3-Uses ``jax.shard_map`` to partition cubed-sphere faces across
4-devices, with explicit halo exchange at partition boundaries.
5-
6-Design overview
7----------------
8-The cubed-sphere has 6 faces, each carrying an (n, n) or (n, n, nlev)
9-grid.  On a multi-device system the natural decomposition is:
10-
11-1. **Face sharding** (<=6 devices): assign one or more faces per device.
12-   Each device computes the dynamics for its assigned faces, and halo
13-   exchange (ghost-zone fill from neighbor faces) happens as a
14:   ``jax.lax.ppermute``-like collective inside shard_map.
15-
16-2. **Sub-face tiling** (>6 devices): each face is further split into
17-   a (tx x ty) tile grid, giving up to 6*tx*ty devices.  Halo exchange
18-   happens both between tiles on the same face and across face boundaries.
19-
20-Both modes are handled transparently by the functions in this module.
21-
22-Compilation strategy
23---------------------
24-Compiled executables are cached in :class:`CompiledShardedStep`.  The
25-cache key (:class:`StepCacheKey`) captures every static input that
26-affects the XLA program:
--
689-    ----------
690-    model
691-        A dynamics model with a ``.step(state, dt)`` method.  Typically
692-        a ``PrimitiveEquationModel`` or ``ShallowWaterModel``.
693-    config : DeviceConfig
694-        Device configuration from :func:`create_device_mesh`.
695-    halo_exchange_fn : callable, optional
696-        Custom halo exchange function ``f(state) -> state`` applied
697-        after each dynamics step.  If ``None``, the model's built-in
698-        halo exchange (via ``pad_halo``) is used.
699-    n : int, optional
700-        Per-face resolution — logging/prewarm metadata forwarded to
701:        ``activate_spmd_halo_backend``.  The old ppermute-vs-all_gather
702:        volume auto-selection is RETIRED: ppermute is always selected;
703-        all_gather only via explicit ``LEGOESM_SPMD_FORCE_ALLGATHER=1``.
704-    nlev : int, optional
705-        Number of vertical levels (logging only, see ``n``).
706-
707-    Returns
708-    -------
709-    CompiledShardedStep or _SingleDeviceStep
710-        Callable with ``(state, dt, physics_fn=None) -> state``.
711-
712-    Notes
713-    -----
714-    For face-only sharding (1/2/3/6 devices) this function ACTIVATES
715-    the explicit SPMD halo backend
716-    (``cubesphere_exchange.activate_spmd_halo_backend``): ``pad_halo``
717:    then routes through shard_map ppermute kernels (multiface; one-face
718-    at halo=1 with 6 devices) instead of relying on XLA's implicit
719-    cross-shard reads.  Letting GSPMD auto-insert collectives for the
720-    cross-face reads — the pre-activation behavior this Notes section
721-    used to describe — replicates ALL compute per device (HLO probe job
722-    8456476); the all_gather kernels survive only as the explicit
723-    ``LEGOESM_SPMD_FORCE_ALLGATHER=1`` diagnostic.
724-
725-    For sub-face tiling (>6 devices), additional tile-boundary
726-    exchange is needed; the SPMD halo backend is NOT activated there
727-    (the tiled path is unvalidated — bench guards exclude it).
728-    """
729-    if config.mesh is None:
730-        logger.info("make_sharded_step: single-device mode, using plain JIT")
731-        return _SingleDeviceStep(model)
732-
733-    # Activate explicit SPMD halo exchange for face-sharded cubed-sphere.
734-    # This replaces implicit cross-shard reads with explicit
735-    # collective-permute rounds, producing much better XLA communication
736-    # patterns.
737-    #
738-    # Iter-49 generalised activation from "exactly 6 devices" to "any
739-    # divisor of 6" (1, 2, 3, 6) on the allgather kernels; the
740:    # ppermute-multiface refit then made ppermute the DEFAULT exchange
741-    # for every face-sharded count and at halo=2 — the allgather
742-    # variant provably replicated ALL compute per device (HLO probe job
743-    # 8456476, per-device FLOPs ratio 1.00 at 2 devices) and is now an
744-    # explicit diagnostic opt-in only (LEGOESM_SPMD_FORCE_ALLGATHER=1).
745-    _n = config.n_devices
746-    _tiling = getattr(config, 'tiling', (1, 1))
747-    _face_ok = (_n in (1, 2, 3, 6) and _tiling == (1, 1))
748:    # 6*kt^2 sub-face tiling: the tiled ppermute EXCHANGE is serial-
749-    # exact (h1+h2, offsets+raw, corners — probe job 8464648), but the
750-    # DYCORE is not yet tile-aware: consumers slice padded arrays with
751-    # full-face (n+2h) indexing (operators_cdgrid.py:555/638/766,
752-    # operators_3d.py:85, fv_tp_2d.py:1024, fv3_sw_core.py:1358 —
753-    # codex review) and staggered (n+1) leaves cannot shard over tile
754-    # axes (IndivisibleError, probe job 8464703).  Activation is
755-    # therefore EXPERIMENTAL and opt-in only; the P4 milestone
756-    # (tile-aware consumers + staggered-leaf ownership layout) flips
757-    # the default.
758-    # UPDATE (2026-06-14): the tile-aware CONSUMERS now EXIST and are
759-    # bit-identity-validated standalone — the full tiled production SW
760-    # tendency ``make_tiled_fv3_sw_tendencies_stage_2d`` (momentum + mass-PPM,
--
1240-        halo_cells_set.update(new_cells.tolist())
1241-
1242-
1243-#: Halo depth the SPMD Voronoi partition infra is built at.  ONE definition
1244-#: consumed by both the production step factory and ``spmd_schedule_cost``:
1245-#: a score computed at a different depth describes a different comm graph, and
1246-#: two independently hardcoded 3s let production drift unnoticed.
1247-SPMD_HALO_DEPTH = 3
1248-
1249-
1250-def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
1251-                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
1252:                       ppermute_cells_per_device_threshold=2_000):
1253-    """How much halo communication one ownership choice costs, computed offline.
1254-
1255:    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
1256-    ROUNDS one halo exchange needs -- the sequential collective launches that
1257-    dominate MPAS strong scaling above ~64 devices.  Runs on a laptop: no GPU,
1258-    no MPI, no benchmark job, so a split can be compared before it costs an
1259-    allocation.
1260-
1261-    It calls the SAME builders production calls
1262-    (:func:`_build_voronoi_partition_infra` then
1263:    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
1264-    different question: a 1-ring ``cellsOnEdge`` adjacency graph reports 8
1265-    rounds where the real depth-3-plus-closure graph reports 12-14.
1266-
1267-    WHAT THE NUMBER IS NOT
1268-    ----------------------
1269-    * ``n_rounds`` is per HALO FILL, not per model step.  A step costs
1270-      ``n_rounds`` x (tendency evaluations per step), which depends on the
1271-      configured integrator -- SSP-RK3 evaluates 3 times, but the MPAS default
1272-      is ``ssp_rk54_scan``.  Multiply with the integrator you actually run.
1273-    * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
1274:      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
1275-      keeps the best; equality is MEASURED (compare the returned
1276-      ``max_degree``), never assumed.  Do not claim "the colouring is already
1277-      optimal so only ownership can help" from this function.
1278:    * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
1279:      cells/device is below ``ppermute_cells_per_device_threshold``, in which
1280:      case there is no ppermute schedule and this number is counterfactual --
1281-      see the returned ``production_strategy``.
1282-
1283-    MESH STATE -- the one thing that silently invalidates the score
1284-    --------------------------------------------------------------
1285-    Production does NOT reorder inside ``make_voronoi_sharded_step``; it
1286-    consumes an already-reordered ``model.mesh``.  The scaling bench reorders
1287-    ONCE for a ``reorder_target`` device count and then runs at a possibly
1288-    DIFFERENT device count.  So pass what you actually have:
1289-
1290-    * raw mesh, scoring a run at ``n_dev``: defaults are right.
1291-    * raw mesh, but the run reorders for a different target: pass
1292-      ``reorder_target=<that target>``; the split is built for the target and
--
1298-    Parameters
1299-    ----------
1300-    mesh : VoronoiMesh
1301-    n_dev : int
1302-        Device count the run uses.  Must be >= 1.
1303-    method : str
1304-        Ownership for the reorder; ignored when *already_reordered*.
1305-    reorder_target : int | None
1306-        Device count the reorder targets, when it differs from *n_dev*.
1307-    already_reordered : bool
1308-    halo_depth : int
1309-        Must match production (3) or the graph is a different graph.
1310:    ppermute_cells_per_device_threshold : int
1311-        Mirror of the production auto-select threshold, only used to report
1312-        ``production_strategy``.
1313-
1314-    Returns
1315-    -------
1316-    dict
1317-        ``n_rounds`` (the cost), ``max_degree`` (the lower bound to compare
1318-        it against), ``n_rounds_greedy``, ``coloring_method``,
1319-        ``resolved_method`` (concrete, never ``"auto"``),
1320:        ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
1321-        ``max_local_cells``, ``max_local_edges``, and the echoed inputs.
1322-
1323-    Reference census on the unrelaxed mesh, which any change here must still
1324-    reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
1325-    geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
1326-    """
1327-    from legoesm.parallel.voronoi_partition import (
1328-        reorder_voronoi_for_sharding, resolve_sharding_partition_method,
1329-    )
1330-
1331-    if int(n_dev) != n_dev or int(n_dev) < 1:
1332-        # int() would silently truncate 3.9 -> 3 and score the wrong split.
--
1360-            f"spmd_schedule_cost: prepared mesh has nCells={n_cells}, "
1361-            f"nEdges={n_edges}, neither divisible by n_dev={n_dev}. The mesh "
1362-            f"is padded for its reorder target"
1363-            f"{'' if already_reordered else f' ({target})'}, so scoring it at "
1364-            f"a device count that does not divide it silently mis-slices the "
1365-            f"owned blocks. Score at a device count that divides the prepared "
1366-            f"mesh.")
1367-    (
1368-        _stacked, _gc, _ge, _noc, _noe, max_lc, max_le, partitions, cell_owner,
1369-    ) = _build_voronoi_partition_infra(prepared, n_dev, halo_depth=halo_depth)
1370-    cells_per = n_cells // n_dev
1371-    edges_per = n_edges // n_dev
1372:    sched = _build_ppermute_schedule(
1373-        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
1374-    )
1375-    return {
1376-        "method": method,
1377-        "resolved_method": resolved,
1378-        "n_dev": n_dev,
1379-        # Unknown for a pre-reordered mesh: the ownership is baked in and the
1380-        # target that produced it is not recoverable from the mesh. Reporting
1381-        # n_dev there would assert something we did not verify.
1382-        "reorder_target": (None if already_reordered else
1383-                           (n_dev if reorder_target is None
1384-                            else int(reorder_target))),
1385-        "already_reordered": bool(already_reordered),
1386-        "halo_depth": halo_depth,
1387-        "n_rounds": int(sched["n_rounds"]),
1388-        "n_rounds_greedy": int(sched["n_rounds_greedy"]),
1389-        "max_degree": int(sched.get("max_degree", -1)),
1390-        "coloring_method": sched["coloring_method"],
1391-        # Production returns before selecting a strategy at n_dev==1, and a
1392-        # caller may force halo_strategy; this reports what AUTO would pick.
1393-        "production_strategy": (
1394-            None if n_dev == 1 else
1395:            ("allgather" if cells_per < ppermute_cells_per_device_threshold
1396:             else "ppermute")),
1397-        "cells_per_device": cells_per,
1398-        "max_local_cells": int(max_lc),
1399-        "max_local_edges": int(max_le),
1400-    }
1401-
1402-
1403-def _build_voronoi_partition_infra(global_mesh, n_dev, halo_depth=2):
1404-    """Pre-compute per-device local meshes and gather/scatter indices.
1405-
1406-    After ``reorder_voronoi_for_sharding`` the global mesh has *both*
1407-    cells and edges ordered by contiguous device blocks.  We compute
1408-    partitions whose owned-entity boundaries exactly match the shard
--
1417-
1418-    Returns
1419-    -------
1420-    stacked_meshes : VoronoiMesh
1421-        Each leaf has shape ``(n_dev, max_local_*)``.
1422-    gather_cells : jnp.ndarray, (n_dev, max_local_cells)
1423-    gather_edges : jnp.ndarray, (n_dev, max_local_edges)
1424-    n_owned_cells : list[int]
1425-    n_owned_edges : list[int]
1426-    max_lc : int
1427-    max_le : int
1428-    partitions : list[VoronoiPartition]
1429:        Per-device partition descriptors (for ppermute schedule building).
1430-    cell_owner : np.ndarray, (nCells,)
1431-        Cell ownership array.
1432-    """
1433-    import numpy as np
1434-    from legoesm.parallel.voronoi_partition import (
1435-        HaloCommSchedule,
1436-        VoronoiPartition,
1437-        build_local_mesh,
1438-        compute_halo_cells,
1439-    )
1440-
1441-    nCells = global_mesh.nCells
--
1648-        stacked_meshes,
1649-        jnp.array(gather_cells),
1650-        jnp.array(gather_edges),
1651-        n_owned_cells,
1652-        n_owned_edges,
1653-        max_lc,
1654-        max_le,
1655-        partitions,
1656-        cell_owner,
1657-    )
1658-
1659-
1660:def _greedy_edge_coloring_ordered(comm_pairs, order):
1661-    """First-fit edge coloring visiting ``order`` (a list of normalized
1662-    ``(min,max)`` pairs). Always a PROPER coloring; the color count depends
1663-    on the visitation order.
1664-    """
1665-    from collections import defaultdict
1666-
1667-    vertex_colors: dict[int, set[int]] = defaultdict(set)
1668-    edge_colors: dict[tuple[int, int], int] = {}
1669-    for u, v in order:
1670-        used = vertex_colors[u] | vertex_colors[v]
1671-        color = 0
1672-        while color in used:
1673-            color += 1
1674-        edge_colors[(u, v)] = color
1675-        vertex_colors[u].add(color)
1676-        vertex_colors[v].add(color)
1677-    return edge_colors
1678-
1679-
1680:def _greedy_edge_coloring(comm_pairs):
1681-    """Legacy first-fit coloring on sorted pairs (the reference/never-regress
1682-    baseline for :func:`_multi_ordering_edge_coloring`). Worst case
1683:    ``2*max_degree - 1`` colors — each color is one ppermute ROUND, and the
1684-    route-B MPAS lane is round-latency-bound (#1113), so excess colors are
1685-    pure wall-clock.
1686-    """
1687:    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
1688:    return _greedy_edge_coloring_ordered(comm_pairs, edges)
1689-
1690-
1691:def _check_proper_edge_coloring(edge_colors, comm_pairs):
1692-    """Every pair colored, and no vertex sees a color twice."""
1693-    from collections import defaultdict
1694-
1695:    if set(edge_colors) != {tuple(sorted(p)) for p in comm_pairs}:
1696-        return False
1697-    seen: dict[int, set[int]] = defaultdict(set)
1698-    for (u, v), c in edge_colors.items():
1699-        if c in seen[u] or c in seen[v]:
1700-            return False
1701-        seen[u].add(c)
1702-        seen[v].add(c)
1703-    return True
1704-
1705-
1706-# Fixed shuffle seeds for the multi-start greedy edge coloring below —
1707-# a constant so every MPI rank / process builds the byte-identical
1708:# schedule (the coloring must agree across ranks or the ppermute pattern
1709-# desynchronises). NOT Math.random / device randomness: this is host-side
1710-# schedule construction, deterministic by seed.
1711-_COLORING_SHUFFLE_SEEDS = tuple(range(16))
1712-
1713-
1714:def _multi_ordering_edge_coloring(comm_pairs):
1715-    """Proper edge coloring via multi-start first-fit; returns the coloring
1716:    using the FEWEST colors (= ppermute rounds) across several deterministic
1717-    visitation orders.
1718-
1719-    First-fit greedy is order-sensitive: on the reordered MPAS comm graphs
1720-    the sorted order can overshoot the chromatic index by up to 3 rounds at
1721-    16 devices, while a degree-descending or shuffled order reaches the
1722-    ``max_degree`` lower bound (verified optimal on ico subdivisions 3–5 ×
1723-    {4,8,16} devices, auto/sfc partitions). Every candidate is a proper
1724-    coloring by construction, so taking the min can NEVER produce an
1725-    invalid schedule and can never regress below the legacy sorted greedy.
1726-
1727-    Deterministic across ranks (sorted + degree orders + fixed-seed
1728-    shuffles). Returns ``(edge_colors, max_degree)``.
1729-    """
1730-    import random
1731-    from collections import defaultdict
1732-
1733:    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
1734-    deg: dict[int, int] = defaultdict(int)
1735-    for u, v in edges:
1736-        deg[u] += 1
1737-        deg[v] += 1
1738-    max_degree = max(deg.values(), default=0)
1739-
1740-    orders = [
1741-        edges,                                                   # sorted
1742-        sorted(edges, key=lambda e: -(deg[e[0]] + deg[e[1]])),   # sum-deg desc
1743-        sorted(edges, key=lambda e: -max(deg[e[0]], deg[e[1]])),  # max-deg desc
1744-    ]
1745-    for seed in _COLORING_SHUFFLE_SEEDS:
1746-        shuffled = edges[:]
1747-        random.Random(seed).shuffle(shuffled)
1748-        orders.append(shuffled)
1749-
1750-    best_colors: dict[tuple[int, int], int] | None = None
1751-    best_rounds = None
1752-    for order in orders:
1753:        ec = _greedy_edge_coloring_ordered(comm_pairs, order)
1754-        rounds = max(ec.values(), default=-1) + 1
1755-        if best_rounds is None or rounds < best_rounds:
1756-            best_rounds, best_colors = rounds, ec
1757-            if best_rounds <= max_degree:
1758-                break            # hit the chromatic-index floor — optimal
1759-    return best_colors, max_degree
1760-
1761-
1762:def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
1763-                             edges_per, max_lc, max_le):
1764:    """Build a ppermute-based halo exchange schedule.
1765-
1766-    Instead of all-gathering the full state (O(N) communication),
1767:    this schedule uses ``jax.lax.ppermute`` to exchange only halo data
1768-    between neighboring devices.  The communication graph is edge-colored
1769:    so that each round of ppermute moves data between non-conflicting
1770-    pairs simultaneously.
1771-
1772-    Parameters
1773-    ----------
1774-    partitions : list[VoronoiPartition]
1775-    cell_owner : np.ndarray, (nCells,)
1776-    n_dev, cells_per, edges_per : int
1777-    max_lc, max_le : int
1778-        Maximum local cell/edge counts (owned + halo) across devices.
1779-
1780-    Returns
1781-    -------
1782-    dict with keys:
1783-        n_rounds, n_rounds_greedy, max_degree, coloring_method,
1784:        ppermute_perms, send_cell_idx, recv_cell_pos,
1785-        send_edge_idx, recv_edge_pos, halo_cells_per_round,
1786-        halo_edges_per_round.
1787-    """
1788-    from collections import defaultdict
1789-
1790-    import numpy as np
1791-
1792-    # ------------------------------------------------------------------
1793-    # 1. For each device pair, find which cells/edges cross the boundary
1794-    # ------------------------------------------------------------------
1795-    # halo_cells_from[d][d'] = global indices of d's halo cells owned by d'
1796-    halo_cells_from: dict[int, dict[int, list[int]]] = defaultdict(
--
1803-            g = int(part.local_cells[h_idx])
1804-            owner = int(cell_owner[g])
1805-            halo_cells_from[d][owner].append(g)
1806-
1807-        for h_idx in range(part.n_owned_edges, part.n_local_edges):
1808-            g = int(part.local_edges[h_idx])
1809-            owner = min(g // edges_per, n_dev - 1)
1810-            halo_edges_from[d][owner].append(g)
1811-
1812-    # ------------------------------------------------------------------
1813-    # 2. Build undirected communication graph
1814-    # ------------------------------------------------------------------
1815:    comm_pairs: set[tuple[int, int]] = set()
1816-    for d in range(n_dev):
1817-        for d_prime in halo_cells_from[d]:
1818-            if d != d_prime:
1819:                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
1820-        for d_prime in halo_edges_from[d]:
1821-            if d != d_prime:
1822:                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
1823-
1824:    if not comm_pairs:
1825-        return {
1826-            'n_rounds': 0,
1827-            'n_rounds_greedy': 0,
1828-            'max_degree': 0,
1829-            'coloring_method': 'none',
1830:            'ppermute_perms': [],
1831-            'send_cell_idx': [],
1832-            'recv_cell_pos': [],
1833-            'send_edge_idx': [],
1834-            'recv_edge_pos': [],
1835-            'halo_cells_per_round': [],
1836-            'halo_edges_per_round': [],
1837-        }
1838-
1839-    # ------------------------------------------------------------------
1840:    # 3. Edge-color the graph: each color = one bidirectional ppermute
1841-    #    ROUND, and the route-B lane is round-latency-bound (#1113), so
1842-    #    fewer colors = directly less wall-clock. First-fit greedy is
1843-    #    order-sensitive; the multi-start coloring reaches the
1844-    #    chromatic-index floor (= max_degree) on every probed MPAS config
1845-    #    where the legacy sorted greedy overshoots (up to 3 rounds at 16
1846-    #    devices). It can never regress: the legacy sorted order is one of
1847-    #    its candidates and it takes the min. Both are verified proper.
1848-    # ------------------------------------------------------------------
1849:    greedy_colors = _greedy_edge_coloring(comm_pairs)
1850-    n_rounds_greedy = max(greedy_colors.values()) + 1
1851:    multi_colors, max_degree = _multi_ordering_edge_coloring(comm_pairs)
1852-    n_rounds_multi = max(multi_colors.values()) + 1
1853-    # Adopt the multi-start coloring ONLY when it STRICTLY reduces rounds;
1854-    # on a tie keep the exact legacy sorted-greedy coloring so the produced
1855-    # schedule is byte-identical to before wherever there is no round win
1856-    # (the win only appears at high device counts — >=16 on the probed
1857-    # MPAS meshes). Both colorings are proper.
1858-    if n_rounds_multi < n_rounds_greedy:
1859-        edge_colors, n_rounds, coloring_method = (
1860-            multi_colors, n_rounds_multi, "multi_greedy")
1861-    else:
1862-        edge_colors, n_rounds, coloring_method = (
1863-            greedy_colors, n_rounds_greedy, "greedy")
1864:    assert _check_proper_edge_coloring(edge_colors, comm_pairs), (
1865:        "improper ppermute edge coloring — two same-round exchanges "
1866-        "would collide at a device")
1867-    rounds: dict[int, list[tuple[int, int]]] = defaultdict(list)
1868-    for (u, v), color in edge_colors.items():
1869-        rounds[color].append((u, v))
1870-
1871-    # ------------------------------------------------------------------
1872-    # 4. Build directed send/recv maps for each device pair
1873-    # ------------------------------------------------------------------
1874-    # cell_send_map[(src, dst)] = list of owned-local indices in src to send
1875-    # cell_recv_map[(dst, src)] = list of local positions in dst to place data
1876-    cell_send_map: dict[tuple[int, int], list[int]] = {}
1877-    cell_recv_map: dict[tuple[int, int], list[int]] = {}
--
1888-            cell_recv_map[(d, d_prime)] = [
1889-                int(partitions[d].cell_g2l[g]) for g in cells_g]
1890-
1891-        for d_prime, edges_g in halo_edges_from[d].items():
1892-            if d_prime == d:
1893-                continue
1894-            edge_send_map[(d_prime, d)] = [
1895-                g - d_prime * edges_per for g in edges_g]
1896-            edge_recv_map[(d, d_prime)] = [
1897-                int(partitions[d].edge_g2l[g]) for g in edges_g]
1898-
1899-    # ------------------------------------------------------------------
1900:    # 5. Assemble per-round ppermute patterns and index arrays
1901-    # ------------------------------------------------------------------
1902:    ppermute_perms_out: list[list[tuple[int, int]]] = []
1903-    send_cell_idx_out: list[jnp.ndarray] = []
1904-    recv_cell_pos_out: list[jnp.ndarray] = []
1905-    send_edge_idx_out: list[jnp.ndarray] = []
1906-    recv_edge_pos_out: list[jnp.ndarray] = []
1907-    halo_cells_per_round: list[int] = []
1908-    halo_edges_per_round: list[int] = []
1909-
1910-    for r in range(n_rounds):
1911-        # Max halo size across all pairs in this round
1912-        max_c = 0
1913-        max_e = 0
1914-        for u, v in rounds[r]:
1915-            for src, dst in [(u, v), (v, u)]:
1916-                max_c = max(max_c, len(cell_send_map.get((src, dst), [])))
1917-                max_e = max(max_e, len(edge_send_map.get((src, dst), [])))
1918-        max_c = max(max_c, 1)  # at least 1 for array shape
1919-        max_e = max(max_e, 1)
1920-        halo_cells_per_round.append(max_c)
1921-        halo_edges_per_round.append(max_e)
1922-
1923:        # Bidirectional ppermute pattern
1924-        perm: list[tuple[int, int]] = []
1925-        partner: dict[int, int] = {}
1926-        for u, v in rounds[r]:
1927-            perm.append((u, v))
1928-            perm.append((v, u))
1929-            partner[u] = v
1930-            partner[v] = u
1931:        ppermute_perms_out.append(perm)
1932-
1933-        # Per-device index arrays (padded with safe defaults)
1934-        sc = np.zeros((n_dev, max_c), dtype=np.int64)
1935-        # Garbage slot: writes go to max_lc (trimmed off later)
1936-        rc = np.full((n_dev, max_c), max_lc, dtype=np.int64)
1937-        se = np.zeros((n_dev, max_e), dtype=np.int64)
1938-        re = np.full((n_dev, max_e), max_le, dtype=np.int64)
1939-
1940-        for d in range(n_dev):
1941-            if d not in partner:
1942-                continue
1943-            dp = partner[d]
--
1959-                re[d, j] = pos
1960-
1961-        send_cell_idx_out.append(jnp.array(sc))
1962-        recv_cell_pos_out.append(jnp.array(rc))
1963-        send_edge_idx_out.append(jnp.array(se))
1964-        recv_edge_pos_out.append(jnp.array(re))
1965-
1966-    return {
1967-        'n_rounds': n_rounds,
1968-        'n_rounds_greedy': n_rounds_greedy,
1969-        'max_degree': max_degree,
1970-        'coloring_method': coloring_method,
1971:        'ppermute_perms': ppermute_perms_out,
1972-        'send_cell_idx': send_cell_idx_out,
1973-        'recv_cell_pos': recv_cell_pos_out,
1974-        'send_edge_idx': send_edge_idx_out,
1975-        'recv_edge_pos': recv_edge_pos_out,
1976-        'halo_cells_per_round': halo_cells_per_round,
1977-        'halo_edges_per_round': halo_edges_per_round,
1978-    }
1979-
1980-
1981-# Schema-drift tripwire (mirrors the M3d ocean twin
1982-# ``voronoi_mpi.exchange_state_mpas_ocean``): a NEW HydrostaticState field
1983-# would silently ride through the packed SPMD halo exchange UNEXCHANGED
--
2054-    omitted field or a swapped slot cannot hide in a hand-rolled copy.
2055-    """
2056-    return jnp.concatenate(
2057-        [T, p_s[:, jnp.newaxis], phis[:, jnp.newaxis], q_flat], axis=-1)
2058-
2059-
2060-def _unpack_cell_state(cell_buf, nlev):
2061-    """Inverse of :func:`_pack_cell_state`: ``(T, p_s, phis, q_flat)``."""
2062-    return (cell_buf[:, :nlev], cell_buf[:, nlev],
2063-            cell_buf[:, nlev + 1], cell_buf[:, nlev + 2:])
2064-
2065-
2066:def _ppermute_halo_fill(cell_pack, u_shard, halo_sl, ppermute_perms,
2067-                        max_lc, max_le):
2068:    """Fill (owned + halo) local buffers from owned shards via ppermute.
2069-
2070-    Runs INSIDE ``shard_map``.  ``cell_pack`` ``(cells_per, W)`` is the
2071-    packed owned-cell buffer — ALL cell-centred prognostics (T | p_s |
2072-    phis | tracers) concatenated on the trailing axis; ``u_shard``
2073-    ``(edges_per, nlev)`` the owned-edge buffer.  Each edge-colored round
2074:    posts ONE flat ppermute carrying BOTH entity classes for ALL packed
2075-    fields — the SPMD mirror of route-A's batched union-neighbor exchange
2076-    (one message per neighbor per dtype group; the compute-precision cast
2077-    upstream guarantees a single dtype group here).
2078-
2079-    ``halo_sl`` is a tuple of per-round ``(send_cell_idx, recv_cell_pos,
2080-    send_edge_idx, recv_edge_pos)`` tuples whose arrays are ALREADY
2081-    device-local ``(1, n_round)`` shard_map arguments (``P("device")``
2082-    specs) — per-rank LOCAL metadata; no device materializes the global
2083:    schedule.  ``ppermute_perms`` is the static per-round permutation.
2084-
2085-    Returns ``(cell_local, u_local)`` of shapes ``(max_lc, W)`` /
2086-    ``(max_le, nlev)``; ghost tail rows stay zero.
2087-    """
2088-    cells_per = cell_pack.shape[0]
2089-    edges_per = u_shard.shape[0]
2090-    # +1 garbage slot for padded scatter targets (trimmed at the end):
2091-    # schedule rows are padded to the round's max halo count, and padding
2092-    # entries target position max_lc / max_le.
2093-    cell_local = jnp.pad(cell_pack, ((0, max_lc + 1 - cells_per), (0, 0)))
2094-    u_local = jnp.pad(u_shard, ((0, max_le + 1 - edges_per), (0, 0)))
2095-
2096-    for r, (sc, rc, se, re) in enumerate(halo_sl):
2097-        send_c = cell_pack[sc[0]]             # (hc_r, W)
2098-        send_e = u_shard[se[0]]               # (he_r, nlev)
2099-        send_c_flat = send_c.ravel()
2100-        send_packed = jnp.concatenate([send_c_flat, send_e.ravel()])
2101:        recv_packed = jax.lax.ppermute(
2102:            send_packed, "device", perm=ppermute_perms[r])
2103-        split_at = send_c_flat.shape[0]       # static
2104-        recv_c = recv_packed[:split_at].reshape(send_c.shape)
2105-        recv_e = recv_packed[split_at:].reshape(send_e.shape)
2106-        cell_local = cell_local.at[rc[0]].set(recv_c)
2107-        u_local = u_local.at[re[0]].set(recv_e)
2108-
2109-    return cell_local[:max_lc], u_local[:max_le]
2110-
2111-
2112-def make_voronoi_sharded_step(
2113-    model,
2114-    dev_config: DeviceConfig,
2115-    *,
2116-    halo_strategy: str = "auto",
2117:    ppermute_cells_per_device_threshold: int = 2_000,
2118-    return_phys_state: bool = False,
2119-):
2120-    """Create a halo-partitioned multi-GPU step for Voronoi (MPAS/TRiSK) grids.
2121-
2122-    Instead of replicating the full state and redundantly computing the
2123-    full step on every device, this implementation:
2124-
2125-    1. Pre-computes per-device local meshes (owned cells/edges + halo)
2126-       at setup time via domain decomposition.
2127-    2. At each RK stage (``config.time_integrator`` via
2128-       ``dispatch_integrator`` — same integrator code as the serial
2129-       ``_step_jit``), exchanges only halo data between neighboring
2130-       devices (not the full state), then computes tendencies on the
2131-       local mesh.  The packed exchange carries the FULL prognostic
2132-       state: u (edge) plus T, p_s, phis and every tracer (cell) in one
2133:       flat ppermute payload per neighbor round.
2134-    3. Applies operator-split physics ONCE on the post-dynamics state
2135-       (traced ``forcing`` + prognostic ``phys_state`` carry threaded
2136-       through), then the temperature/tracer floors and the global mass
2137-       fix — mirroring the serial ``MPASPrimitiveEquationModel._step_jit``
2138-       operator ordering exactly.
2139-
2140-    Local-only metadata: the per-device local meshes (stacked with a
2141:    leading device axis), the ppermute schedule index arrays, and the
2142-    mass-fix ``areaCell`` are ``P("device")``-sharded and passed as
2143-    ARGUMENTS into the jitted step (multi-controller-safe: sharded jit
2144-    args are legal where sharded closure constants raise at trace time)
2145-    — each device holds ONLY its own local mesh + schedule rows, never
2146-    the global connectivity.  The one remaining NON-local metadata is
2147-    the global-mesh closure handed to the operator-split physics term:
2148-    column-local physics runs OUTSIDE shard_map on the GSPMD-sharded
2149-    global arrays, reading only replicated 1-D cell fields (latCell
2150-    etc. — O(nCells) scalars, not the 2-D connectivity).  See the
2151-    physics block below and
2152-    ``docs/performance/scaling/mpas_atm_native_step_audit.md``.
2153-
2154-    Parameters
2155-    ----------
2156-    model
2157-        ``MPASPrimitiveEquationModel`` with ``.mesh``, ``.sigma_coord``,
2158-        ``.config``.
2159-    dev_config : DeviceConfig
2160-        From :func:`~legoesm.parallel.mesh.create_voronoi_device_mesh`.
2161-    halo_strategy : str
2162:        ``"auto"`` (default) selects ``"ppermute"`` for large grids and
2163-        ``"allgather"`` for small ones based on
2164:        *ppermute_cells_per_device_threshold*.
2165:        ``"ppermute"`` forces neighbor-only exchange via
2166:        ``jax.lax.ppermute`` — O(halo) communication.
2167-        ``"allgather"`` forces the full-state all-gather —
2168-        O(N) communication.
2169:    ppermute_cells_per_device_threshold : int
2170:        When ``halo_strategy="auto"``, use ppermute only if each device
2171-        owns at least this many cells.  Below this threshold the
2172:        per-round packing/scatter overhead of ppermute exceeds the
2173-        communication savings over allgather.  Default: 2 000.
2174-        (Lowered from 25 000 to avoid the O(N) allgather bottleneck
2175-        on moderate icosahedral grids like I5 with 2–4 GPUs.)
2176-    return_phys_state : bool
2177-        ``False`` (default, backward-compatible): the returned step is
2178-        ``step(state, dt, physics_fn=None, forcing=None, phys_state=None)
2179-        -> state`` — the physics carry is dropped, so a STATEFUL
2180-        physics_fn is refused loudly (issue #405/#413).  ``True``: the
2181-        step returns ``(state, phys_state_out)`` — full operator-split
2182-        production parity with ``make_voronoi_mpi_step(
2183-        return_phys_state=True)``; the prognostic physics carry (TKE /
2184-        convection state) and the traced per-step ``forcing`` (e.g.
--
2251-    if mpas_hydrostatic_tendencies is None:
2252-        raise TypeError(
2253-            f"{type(model).__name__} does not expose a 'sharded_tendency_fn' "
2254-            f"staticmethod; the multi-device Voronoi sharder needs the free "
2255-            f"tendency RHS (state, mesh, sigma_coord, config, *, dt=...) to run "
2256-            f"on a rank-local mesh without importing the dycore's component."
2257-        )
2258-
2259-    # ------------------------------------------------------------------
2260-    # Auto-select halo strategy based on grid size per device
2261-    # ------------------------------------------------------------------
2262-    if halo_strategy == "auto":
2263:        if cells_per < ppermute_cells_per_device_threshold:
2264-            halo_strategy = "allgather"
2265-            logger.info(
2266-                "Auto-selected allgather strategy: cells_per_device=%d < "
2267:                "threshold=%d — ppermute packing overhead would dominate.",
2268:                cells_per, ppermute_cells_per_device_threshold,
2269-            )
2270-        else:
2271:            halo_strategy = "ppermute"
2272-            logger.info(
2273:                "Auto-selected ppermute strategy: cells_per_device=%d >= "
2274-                "threshold=%d.",
2275:                cells_per, ppermute_cells_per_device_threshold,
2276-            )
2277-
2278-    # ------------------------------------------------------------------
2279-    # Setup: build per-device local meshes and gather indices
2280-    # ------------------------------------------------------------------
2281-    logger.info(
2282-        "Building halo-partitioned infrastructure for %d device(s) "
2283-        "(nCells=%d, nEdges=%d, halo_depth=3, strategy=%s) ...",
2284-        n_dev, nCells, nEdges, halo_strategy,
2285-    )
2286-    t0 = time.time()
2287-    (
2288-        stacked_meshes,   # VoronoiMesh pytree with (n_dev, max_l*) leaves
2289-        gather_cells,     # (n_dev, max_lc)
2290-        gather_edges,     # (n_dev, max_le)
2291-        _n_owned_cells,
2292-        _n_owned_edges,
2293-        max_lc,
2294-        max_le,
2295:        partitions_out,   # list[VoronoiPartition] (for ppermute schedule)
2296-        cell_owner_out,   # np.ndarray (nCells,) cell ownership
2297-    ) = _build_voronoi_partition_infra(global_mesh, n_dev,
2298-                                       halo_depth=SPMD_HALO_DEPTH)
2299-    logger.info(
2300-        "  partition setup done in %.2fs  "
2301-        "(max_local_cells=%d, max_local_edges=%d, cells_per=%d, edges_per=%d)",
2302-        time.time() - t0, max_lc, max_le, cells_per, edges_per,
2303-    )
2304-
2305-    # LOCAL-ONLY metadata: shard the stacked local meshes on the leading
2306-    # device axis — device i holds ONLY its own local mesh (leaf slice
2307-    # [i]), never the other devices' connectivity.  The mesh rides into
--
2311-    # ``multiprocess_safe_device_put`` builds the global array from each
2312-    # process's local copy).  All leaves are arrays after the jnp.stack
2313-    # in _build_voronoi_partition_infra (ints become (n_dev,) arrays).
2314-    dev_sharding = dev_config.face_sharding  # P("device") on axis 0
2315-    stacked_meshes = jax.tree.map(
2316-        lambda x: multiprocess_safe_device_put(x, dev_sharding),
2317-        stacked_meshes,
2318-    )
2319-
2320-    nlev = model.sigma_coord.n_levels
2321-
2322-    # ------------------------------------------------------------------
2323:    # Strategy dispatch: ppermute (O(halo)) vs allgather (O(N))
2324-    # ------------------------------------------------------------------
2325-
2326:    use_ppermute = halo_strategy == "ppermute"
2327-
2328:    if use_ppermute:
2329:        # Build ppermute schedule: neighbor-only halo exchange
2330-        t1 = time.time()
2331:        pp_sched = _build_ppermute_schedule(
2332-            partitions_out, cell_owner_out, n_dev,
2333-            cells_per, edges_per, max_lc, max_le,
2334-        )
2335-        n_rounds = pp_sched['n_rounds']
2336:        ppermute_perms = pp_sched['ppermute_perms']
2337-
2338-        # LOCAL-ONLY metadata: shard the per-round index arrays on the
2339-        # leading device axis (each device holds only its own schedule
2340-        # rows) and thread them as shard_map ARGUMENTS — see the stacked
2341-        # meshes above for why args, not closures.
2342-        halo_args = tuple(
2343-            (
2344-                multiprocess_safe_device_put(
2345-                    pp_sched['send_cell_idx'][r], dev_sharding),
2346-                multiprocess_safe_device_put(

exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1730,1990p'
nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1660,1735p'
rg -n -A 100 -B 20 \"def _multi_ordering_edge_coloring|def _check_proper_edge_coloring|def _greedy_edge_coloring\" packages/core/legoesm/parallel/sharded_dynamics.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 229ms:
  1730	    import random
  1731	    from collections import defaultdict
  1732	
  1733	    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
  1734	    deg: dict[int, int] = defaultdict(int)
  1735	    for u, v in edges:
  1736	        deg[u] += 1
  1737	        deg[v] += 1
  1738	    max_degree = max(deg.values(), default=0)
  1739	
  1740	    orders = [
  1741	        edges,                                                   # sorted
  1742	        sorted(edges, key=lambda e: -(deg[e[0]] + deg[e[1]])),   # sum-deg desc
  1743	        sorted(edges, key=lambda e: -max(deg[e[0]], deg[e[1]])),  # max-deg desc
  1744	    ]
  1745	    for seed in _COLORING_SHUFFLE_SEEDS:
  1746	        shuffled = edges[:]
  1747	        random.Random(seed).shuffle(shuffled)
  1748	        orders.append(shuffled)
  1749	
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
  1931	        ppermute_perms_out.append(perm)
  1932	
  1933	        # Per-device index arrays (padded with safe defaults)
  1934	        sc = np.zeros((n_dev, max_c), dtype=np.int64)
  1935	        # Garbage slot: writes go to max_lc (trimmed off later)
  1936	        rc = np.full((n_dev, max_c), max_lc, dtype=np.int64)
  1937	        se = np.zeros((n_dev, max_e), dtype=np.int64)
  1938	        re = np.full((n_dev, max_e), max_le, dtype=np.int64)
  1939	
  1940	        for d in range(n_dev):
  1941	            if d not in partner:
  1942	                continue
  1943	            dp = partner[d]
  1944	
  1945	            cs = cell_send_map.get((d, dp), [])
  1946	            for j, idx in enumerate(cs):
  1947	                sc[d, j] = idx
  1948	
  1949	            cr = cell_recv_map.get((d, dp), [])
  1950	            for j, pos in enumerate(cr):
  1951	                rc[d, j] = pos
  1952	
  1953	            es = edge_send_map.get((d, dp), [])
  1954	            for j, idx in enumerate(es):
  1955	                se[d, j] = idx
  1956	
  1957	            er = edge_recv_map.get((d, dp), [])
  1958	            for j, pos in enumerate(er):
  1959	                re[d, j] = pos
  1960	
  1961	        send_cell_idx_out.append(jnp.array(sc))
  1962	        recv_cell_pos_out.append(jnp.array(rc))
  1963	        send_edge_idx_out.append(jnp.array(se))
  1964	        recv_edge_pos_out.append(jnp.array(re))
  1965	
  1966	    return {
  1967	        'n_rounds': n_rounds,
  1968	        'n_rounds_greedy': n_rounds_greedy,
  1969	        'max_degree': max_degree,
  1970	        'coloring_method': coloring_method,
  1971	        'ppermute_perms': ppermute_perms_out,
  1972	        'send_cell_idx': send_cell_idx_out,
  1973	        'recv_cell_pos': recv_cell_pos_out,
  1974	        'send_edge_idx': send_edge_idx_out,
  1975	        'recv_edge_pos': recv_edge_pos_out,
  1976	        'halo_cells_per_round': halo_cells_per_round,
  1977	        'halo_edges_per_round': halo_edges_per_round,
  1978	    }
  1979	
  1980	
  1981	# Schema-drift tripwire (mirrors the M3d ocean twin
  1982	# ``voronoi_mpi.exchange_state_mpas_ocean``): a NEW HydrostaticState field
  1983	# would silently ride through the packed SPMD halo exchange UNEXCHANGED
  1984	# (stale halos on every RK stage) — fail loudly so the cell-pack layout,
  1985	# the physics application and this set are extended deliberately.
  1986	# Workload signatures — (n_devices, global edge rows, global cell rows,
  1987	# nlev), all trace-time-static — where a tendency-output
  1988	# optimization_barrier is measured to pay. OBSERVED CORRELATION, not a
  1989	# proven XLA cost-model account: at ico-L8 np4 (and only there among
  1990	# np2/4/8) the compiled step carries three once-per-step 3.3-3.6 ms
  1660	def _greedy_edge_coloring_ordered(comm_pairs, order):
  1661	    """First-fit edge coloring visiting ``order`` (a list of normalized
  1662	    ``(min,max)`` pairs). Always a PROPER coloring; the color count depends
  1663	    on the visitation order.
  1664	    """
  1665	    from collections import defaultdict
  1666	
  1667	    vertex_colors: dict[int, set[int]] = defaultdict(set)
  1668	    edge_colors: dict[tuple[int, int], int] = {}
  1669	    for u, v in order:
  1670	        used = vertex_colors[u] | vertex_colors[v]
  1671	        color = 0
  1672	        while color in used:
  1673	            color += 1
  1674	        edge_colors[(u, v)] = color
  1675	        vertex_colors[u].add(color)
  1676	        vertex_colors[v].add(color)
  1677	    return edge_colors
  1678	
  1679	
  1680	def _greedy_edge_coloring(comm_pairs):
  1681	    """Legacy first-fit coloring on sorted pairs (the reference/never-regress
  1682	    baseline for :func:`_multi_ordering_edge_coloring`). Worst case
  1683	    ``2*max_degree - 1`` colors — each color is one ppermute ROUND, and the
  1684	    route-B MPAS lane is round-latency-bound (#1113), so excess colors are
  1685	    pure wall-clock.
  1686	    """
  1687	    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
  1688	    return _greedy_edge_coloring_ordered(comm_pairs, edges)
  1689	
  1690	
  1691	def _check_proper_edge_coloring(edge_colors, comm_pairs):
  1692	    """Every pair colored, and no vertex sees a color twice."""
  1693	    from collections import defaultdict
  1694	
  1695	    if set(edge_colors) != {tuple(sorted(p)) for p in comm_pairs}:
  1696	        return False
  1697	    seen: dict[int, set[int]] = defaultdict(set)
  1698	    for (u, v), c in edge_colors.items():
  1699	        if c in seen[u] or c in seen[v]:
  1700	            return False
  1701	        seen[u].add(c)
  1702	        seen[v].add(c)
  1703	    return True
  1704	
  1705	
  1706	# Fixed shuffle seeds for the multi-start greedy edge coloring below —
  1707	# a constant so every MPI rank / process builds the byte-identical
  1708	# schedule (the coloring must agree across ranks or the ppermute pattern
  1709	# desynchronises). NOT Math.random / device randomness: this is host-side
  1710	# schedule construction, deterministic by seed.
  1711	_COLORING_SHUFFLE_SEEDS = tuple(range(16))
  1712	
  1713	
  1714	def _multi_ordering_edge_coloring(comm_pairs):
  1715	    """Proper edge coloring via multi-start first-fit; returns the coloring
  1716	    using the FEWEST colors (= ppermute rounds) across several deterministic
  1717	    visitation orders.
  1718	
  1719	    First-fit greedy is order-sensitive: on the reordered MPAS comm graphs
  1720	    the sorted order can overshoot the chromatic index by up to 3 rounds at
  1721	    16 devices, while a degree-descending or shuffled order reaches the
  1722	    ``max_degree`` lower bound (verified optimal on ico subdivisions 3–5 ×
  1723	    {4,8,16} devices, auto/sfc partitions). Every candidate is a proper
  1724	    coloring by construction, so taking the min can NEVER produce an
  1725	    invalid schedule and can never regress below the legacy sorted greedy.
  1726	
  1727	    Deterministic across ranks (sorted + degree orders + fixed-seed
  1728	    shuffles). Returns ``(edge_colors, max_degree)``.
  1729	    """
  1730	    import random
  1731	    from collections import defaultdict
  1732	
  1733	    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
  1734	    deg: dict[int, int] = defaultdict(int)
  1735	    for u, v in edges:
1640-    n_owned_edges = []
1641-    for rank, part in enumerate(partitions):
1642-        gather_cells[rank, : part.n_local_cells] = part.local_cells
1643-        gather_edges[rank, : part.n_local_edges] = part.local_edges
1644-        n_owned_cells.append(part.n_owned_cells)
1645-        n_owned_edges.append(part.n_owned_edges)
1646-
1647-    return (
1648-        stacked_meshes,
1649-        jnp.array(gather_cells),
1650-        jnp.array(gather_edges),
1651-        n_owned_cells,
1652-        n_owned_edges,
1653-        max_lc,
1654-        max_le,
1655-        partitions,
1656-        cell_owner,
1657-    )
1658-
1659-
1660:def _greedy_edge_coloring_ordered(comm_pairs, order):
1661-    """First-fit edge coloring visiting ``order`` (a list of normalized
1662-    ``(min,max)`` pairs). Always a PROPER coloring; the color count depends
1663-    on the visitation order.
1664-    """
1665-    from collections import defaultdict
1666-
1667-    vertex_colors: dict[int, set[int]] = defaultdict(set)
1668-    edge_colors: dict[tuple[int, int], int] = {}
1669-    for u, v in order:
1670-        used = vertex_colors[u] | vertex_colors[v]
1671-        color = 0
1672-        while color in used:
1673-            color += 1
1674-        edge_colors[(u, v)] = color
1675-        vertex_colors[u].add(color)
1676-        vertex_colors[v].add(color)
1677-    return edge_colors
1678-
1679-
1680:def _greedy_edge_coloring(comm_pairs):
1681-    """Legacy first-fit coloring on sorted pairs (the reference/never-regress
1682-    baseline for :func:`_multi_ordering_edge_coloring`). Worst case
1683-    ``2*max_degree - 1`` colors — each color is one ppermute ROUND, and the
1684-    route-B MPAS lane is round-latency-bound (#1113), so excess colors are
1685-    pure wall-clock.
1686-    """
1687-    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
1688-    return _greedy_edge_coloring_ordered(comm_pairs, edges)
1689-
1690-
1691:def _check_proper_edge_coloring(edge_colors, comm_pairs):
1692-    """Every pair colored, and no vertex sees a color twice."""
1693-    from collections import defaultdict
1694-
1695-    if set(edge_colors) != {tuple(sorted(p)) for p in comm_pairs}:
1696-        return False
1697-    seen: dict[int, set[int]] = defaultdict(set)
1698-    for (u, v), c in edge_colors.items():
1699-        if c in seen[u] or c in seen[v]:
1700-            return False
1701-        seen[u].add(c)
1702-        seen[v].add(c)
1703-    return True
1704-
1705-
1706-# Fixed shuffle seeds for the multi-start greedy edge coloring below —
1707-# a constant so every MPI rank / process builds the byte-identical
1708-# schedule (the coloring must agree across ranks or the ppermute pattern
1709-# desynchronises). NOT Math.random / device randomness: this is host-side
1710-# schedule construction, deterministic by seed.
1711-_COLORING_SHUFFLE_SEEDS = tuple(range(16))
1712-
1713-
1714:def _multi_ordering_edge_coloring(comm_pairs):
1715-    """Proper edge coloring via multi-start first-fit; returns the coloring
1716-    using the FEWEST colors (= ppermute rounds) across several deterministic
1717-    visitation orders.
1718-
1719-    First-fit greedy is order-sensitive: on the reordered MPAS comm graphs
1720-    the sorted order can overshoot the chromatic index by up to 3 rounds at
1721-    16 devices, while a degree-descending or shuffled order reaches the
1722-    ``max_degree`` lower bound (verified optimal on ico subdivisions 3–5 ×
1723-    {4,8,16} devices, auto/sfc partitions). Every candidate is a proper
1724-    coloring by construction, so taking the min can NEVER produce an
1725-    invalid schedule and can never regress below the legacy sorted greedy.
1726-
1727-    Deterministic across ranks (sorted + degree orders + fixed-seed
1728-    shuffles). Returns ``(edge_colors, max_degree)``.
1729-    """
1730-    import random
1731-    from collections import defaultdict
1732-
1733-    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
1734-    deg: dict[int, int] = defaultdict(int)
1735-    for u, v in edges:
1736-        deg[u] += 1
1737-        deg[v] += 1
1738-    max_degree = max(deg.values(), default=0)
1739-
1740-    orders = [
1741-        edges,                                                   # sorted
1742-        sorted(edges, key=lambda e: -(deg[e[0]] + deg[e[1]])),   # sum-deg desc
1743-        sorted(edges, key=lambda e: -max(deg[e[0]], deg[e[1]])),  # max-deg desc
1744-    ]
1745-    for seed in _COLORING_SHUFFLE_SEEDS:
1746-        shuffled = edges[:]
1747-        random.Random(seed).shuffle(shuffled)
1748-        orders.append(shuffled)
1749-
1750-    best_colors: dict[tuple[int, int], int] | None = None
1751-    best_rounds = None
1752-    for order in orders:
1753-        ec = _greedy_edge_coloring_ordered(comm_pairs, order)
1754-        rounds = max(ec.values(), default=-1) + 1
1755-        if best_rounds is None or rounds < best_rounds:
1756-            best_rounds, best_colors = rounds, ec
1757-            if best_rounds <= max_degree:
1758-                break            # hit the chromatic-index floor — optimal
1759-    return best_colors, max_degree
1760-
1761-
1762-def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
1763-                             edges_per, max_lc, max_le):
1764-    """Build a ppermute-based halo exchange schedule.
1765-
1766-    Instead of all-gathering the full state (O(N) communication),
1767-    this schedule uses ``jax.lax.ppermute`` to exchange only halo data
1768-    between neighboring devices.  The communication graph is edge-colored
1769-    so that each round of ppermute moves data between non-conflicting
1770-    pairs simultaneously.
1771-
1772-    Parameters
1773-    ----------
1774-    partitions : list[VoronoiPartition]
1775-    cell_owner : np.ndarray, (nCells,)
1776-    n_dev, cells_per, edges_per : int
1777-    max_lc, max_le : int
1778-        Maximum local cell/edge counts (owned + halo) across devices.
1779-
1780-    Returns
1781-    -------
1782-    dict with keys:
1783-        n_rounds, n_rounds_greedy, max_degree, coloring_method,
1784-        ppermute_perms, send_cell_idx, recv_cell_pos,
1785-        send_edge_idx, recv_edge_pos, halo_cells_per_round,
1786-        halo_edges_per_round.
1787-    """
1788-    from collections import defaultdict
1789-
1790-    import numpy as np
1791-
1792-    # ------------------------------------------------------------------
1793-    # 1. For each device pair, find which cells/edges cross the boundary
1794-    # ------------------------------------------------------------------
1795-    # halo_cells_from[d][d'] = global indices of d's halo cells owned by d'
1796-    halo_cells_from: dict[int, dict[int, list[int]]] = defaultdict(
1797-        lambda: defaultdict(list))
1798-    halo_edges_from: dict[int, dict[int, list[int]]] = defaultdict(
1799-        lambda: defaultdict(list))
1800-
1801-    for d, part in enumerate(partitions):
1802-        for h_idx in range(part.n_owned_cells, part.n_local_cells):
1803-            g = int(part.local_cells[h_idx])
1804-            owner = int(cell_owner[g])
1805-            halo_cells_from[d][owner].append(g)
1806-
1807-        for h_idx in range(part.n_owned_edges, part.n_local_edges):
1808-            g = int(part.local_edges[h_idx])
1809-            owner = min(g // edges_per, n_dev - 1)
1810-            halo_edges_from[d][owner].append(g)
1811-
1812-    # ------------------------------------------------------------------
1813-    # 2. Build undirected communication graph
1814-    # ------------------------------------------------------------------

codex
The mechanics are mostly behaving as claimed: Bash expands `$?` before executing `local`, and the conditional argument expansion preserves the expectation as one argument. I’m now checking the remaining gate loopholes and whether the tests actually exercise the advertised failure paths.
exec
/usr/bin/bash -lc "bash -n scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
rg -n -C 6 \"def scaling_metadata|git|revision|commit|metadata\" scripts/bench/metadata.py scripts/bench 2>/dev/null | head -n 500
rg -n \"REPO=|LEGOESM_REPO|PY=\" scripts/cluster/scaling_levante/_env.sh scripts/cluster/scaling_levante 2>/dev/null | head -n 400
rg -n \"expect_rounds|expected_rounds_check|lloyd_flag|reorder_target|already_reordered\" tests scripts --glob '*.py' --glob '*.sbatch'
ls -d .venv 2>/dev/null; command -v pytest || true; command -v python" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 1 in 431ms:
scripts/bench/metadata.py:1:"""Shared, self-describing metadata for scaling-benchmark outputs (roadmap item 9).
scripts/bench/metadata.py-2-
scripts/bench/metadata.py:3:SINGLE source of the metadata block that every scaling JSON carries, so that
scripts/bench/metadata.py-4-each bench driver stops rolling its own record.  The goal is that a scaling row
scripts/bench/metadata.py-5-is comparable and *falsifiable* from the record ALONE — a CPU fallback, a
scripts/bench/metadata.py-6-host-staged halo, a replicated (non-SPMD) run, or an f32 ablation can no longer
scripts/bench/metadata.py-7-masquerade as a valid GPU-direct f64 scaling point.
scripts/bench/metadata.py-8-
scripts/bench/metadata.py-9-Every record answers the roadmap's item-9 questions:
--
scripts/bench/metadata.py-16-mode) are auto-detected from the LIVE process so they cannot be mislabeled.
scripts/bench/metadata.py-17-
scripts/bench/metadata.py-18-Consumers: ``run_levante_gpu_scaling.py``, ``run_cpu_mpi_scaling.py``,
scripts/bench/metadata.py-19-``bench_atm_latlon_spmd_scaling.py``, ``bench_ocean_latlon_spmd_scaling.py``,
scripts/bench/metadata.py-20-``bench_mpas_spmd_scaling.py``, ``bench_ocean_mpi_scaling.py``,
scripts/bench/metadata.py-21-``bench_ocean_gpu_scaling.py`` (and any future bench driver) merge
scripts/bench/metadata.py:22:``scaling_metadata(...)`` under the ``"metadata"`` key of their JSON payload.
scripts/bench/metadata.py:23:Aggregators read ``payload["metadata"]``.
scripts/bench/metadata.py-24-"""
scripts/bench/metadata.py-25-from __future__ import annotations
scripts/bench/metadata.py-26-
scripts/bench/metadata.py-27-import os
scripts/bench/metadata.py-28-import re
scripts/bench/metadata.py-29-from datetime import datetime, timezone
--
scripts/bench/metadata.py-81-    "transport",
scripts/bench/metadata.py-82-    "virtual_cpu_devices",
scripts/bench/metadata.py-83-    "launcher",
scripts/bench/metadata.py-84-)
scripts/bench/metadata.py-85-
scripts/bench/metadata.py-86-#: Keys that MUST be PRESENT (the roadmap requires the field) but whose value
scripts/bench/metadata.py:87:#: may legitimately be ``None`` — an atmosphere run has no barotropic-solver
scripts/bench/metadata.py-88-#: residual; a single-device run has no partition metrics or cells-per-rank.
scripts/bench/metadata.py-89-PRESENT_KEYS: tuple[str, ...] = (
scripts/bench/metadata.py-90-    "devices_per_rank",
scripts/bench/metadata.py-91-    "cells_per_rank",
scripts/bench/metadata.py-92-    "solver_variant",
scripts/bench/metadata.py-93-    "solver_residual",
--
scripts/bench/metadata.py-112-    """Count lines of ``hlo_text`` matching the op-call regex ``rx``.
scripts/bench/metadata.py-113-
scripts/bench/metadata.py-114-    The ``-done``/``_done`` async companion is excluded STRUCTURALLY by the
scripts/bench/metadata.py-115-    regex — ``op(?:[_-]start)?\\(`` cannot match ``op-done(`` (the char after
scripts/bench/metadata.py-116-    the op name is ``-``/``_``, not ``(`` or a ``start`` suffix) — so no
scripts/bench/metadata.py-117-    substring ``"done" not in line`` filter is used: that filter would
scripts/bench/metadata.py:118:    false-drop a legitimate collective whose line merely CONTAINS "done"
scripts/bench/metadata.py:119:    elsewhere (an XLA ``metadata={op_name="…/done_stage/…"}`` tag, a
scripts/bench/metadata.py-120-    ``%done_mass`` SSA name).  The ``\\(`` op-call anchor still keeps a
scripts/bench/metadata.py-121-    config-header ``XLA_FLAGS`` echo (a flag name, no paren) from inflating."""
scripts/bench/metadata.py-122-    return sum(1 for line in hlo_text.splitlines() if rx.search(line))
scripts/bench/metadata.py-123-
scripts/bench/metadata.py-124-
scripts/bench/metadata.py-125-def count_collective_permutes(hlo_text: str) -> int:
--
scripts/bench/metadata.py-224-        return None
scripts/bench/metadata.py-225-
scripts/bench/metadata.py-226-
scripts/bench/metadata.py-227-def _is_empty(v: Any) -> bool:
scripts/bench/metadata.py-228-    """True for a non-informative required value: ``None``, ``""``, or an EMPTY
scripts/bench/metadata.py-229-    container (e.g. ``precision_knobs={}`` — which would hide an f32/TF32
scripts/bench/metadata.py:230:    ablation).  Scalars ``0`` / ``0.0`` / ``False`` are NOT empty (a legitimate
scripts/bench/metadata.py-231-    ``n_gpus=0`` / ``gpu_direct_active=False`` must pass)."""
scripts/bench/metadata.py-232-    if v is None or v == "":
scripts/bench/metadata.py-233-        return True
scripts/bench/metadata.py-234-    if isinstance(v, (dict, list, tuple, set, frozenset)) and len(v) == 0:
scripts/bench/metadata.py-235-        return True
scripts/bench/metadata.py-236-    return False
scripts/bench/metadata.py-237-
scripts/bench/metadata.py-238-
scripts/bench/metadata.py:239:def git_sha(short: bool = True) -> str:
scripts/bench/metadata.py:240:    """Best-effort git commit SHA of THIS source tree (repro signature).
scripts/bench/metadata.py-241-
scripts/bench/metadata.py-242-    Every scaling row must be attributable to an exact code state — a
scripts/bench/metadata.py-243-    measurement without a SHA cannot be reproduced or compared across
scripts/bench/metadata.py-244-    branches (audit item 7: no harness recorded one).  A dirty working
scripts/bench/metadata.py-245-    tree is marked ``<sha>-dirty`` so the SHA never over-claims
scripts/bench/metadata.py:246:    reproducibility.  Fail-open (``"unknown"``): a missing git binary or
scripts/bench/metadata.py-247-    a non-repo run directory must never kill a benchmark.
scripts/bench/metadata.py-248-    """
scripts/bench/metadata.py-249-    import subprocess
scripts/bench/metadata.py-250-
scripts/bench/metadata.py-251-    cwd = os.path.dirname(os.path.abspath(__file__))
scripts/bench/metadata.py-252-    try:
scripts/bench/metadata.py:253:        cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
scripts/bench/metadata.py-254-        out = subprocess.run(cmd, capture_output=True, text=True,
scripts/bench/metadata.py-255-                             timeout=5, cwd=cwd)
scripts/bench/metadata.py-256-        sha = out.stdout.strip()
scripts/bench/metadata.py-257-        if out.returncode != 0 or not sha:
scripts/bench/metadata.py-258-            return "unknown"
scripts/bench/metadata.py-259-        dirty = subprocess.run(
scripts/bench/metadata.py:260:            ["git", "status", "--porcelain", "--untracked-files=no"],
scripts/bench/metadata.py-261-            capture_output=True, text=True, timeout=5, cwd=cwd)
scripts/bench/metadata.py-262-        if dirty.returncode == 0 and dirty.stdout.strip():
scripts/bench/metadata.py-263-            return sha + "-dirty"
scripts/bench/metadata.py-264-        return sha
scripts/bench/metadata.py-265-    except Exception:
scripts/bench/metadata.py-266-        return "unknown"
--
scripts/bench/metadata.py-407-        "mpi4jax_cuda_support": cuda,
scripts/bench/metadata.py-408-        "gpu_direct_active": bool(on_gpu and mpi4jax_halo and device_direct),
scripts/bench/metadata.py-409-        "host_staged_halo": bool(on_gpu and mpi4jax_halo and not device_direct),
scripts/bench/metadata.py-410-    }
scripts/bench/metadata.py-411-
scripts/bench/metadata.py-412-
scripts/bench/metadata.py:413:def scaling_metadata(
scripts/bench/metadata.py-414-    *,
scripts/bench/metadata.py-415-    grid: str,
scripts/bench/metadata.py-416-    component: str,
scripts/bench/metadata.py-417-    resolution: Any,
scripts/bench/metadata.py-418-    n_levels: int,
scripts/bench/metadata.py-419-    precision: str,
--
scripts/bench/metadata.py-428-    scaling_kind: str | None = None,
scripts/bench/metadata.py-429-    transport: str | None = None,
scripts/bench/metadata.py-430-    partition_metrics: dict[str, Any] | None = None,
scripts/bench/metadata.py-431-    timestamp_utc: str | None = None,
scripts/bench/metadata.py-432-    extra: dict[str, Any] | None = None,
scripts/bench/metadata.py-433-) -> dict[str, Any]:
scripts/bench/metadata.py:434:    """Return the complete self-describing metadata block for one scaling row.
scripts/bench/metadata.py-435-
scripts/bench/metadata.py-436-    Parameters
scripts/bench/metadata.py-437-    ----------
scripts/bench/metadata.py-438-    grid, component, resolution, n_levels, precision, decomposition
scripts/bench/metadata.py-439-        Scientific descriptors of the case (caller-supplied).
scripts/bench/metadata.py-440-    n_ranks, n_gpus, devices_per_rank, cells_per_rank
--
scripts/bench/metadata.py-511-        "transport": resolved_transport,
scripts/bench/metadata.py-512-        "virtual_cpu_devices": detect_virtual_cpu_devices(backend),
scripts/bench/metadata.py-513-        "launcher": detect_launcher(),
scripts/bench/metadata.py-514-        "hostname": os.environ.get("HOSTNAME")
scripts/bench/metadata.py-515-        or os.environ.get("SLURMD_NODENAME", ""),
scripts/bench/metadata.py-516-        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
scripts/bench/metadata.py:517:        "git_sha": git_sha(),
scripts/bench/metadata.py-518-    }
scripts/bench/metadata.py-519-    md.update(gpu_direct_mode(backend, transport=resolved_transport))
scripts/bench/metadata.py-520-    if partition_metrics:
scripts/bench/metadata.py-521-        md["partition_metrics"] = partition_metrics
scripts/bench/metadata.py-522-    if extra:
scripts/bench/metadata.py-523-        md["extra"] = extra
scripts/bench/metadata.py-524-    return md
scripts/bench/metadata.py-525-
scripts/bench/metadata.py-526-
scripts/bench/metadata.py:527:def validate_scaling_metadata(
scripts/bench/metadata.py-528-    md: dict[str, Any], *, strict: bool = True
scripts/bench/metadata.py-529-) -> list[str]:
scripts/bench/metadata.py-530-    """Return the list of comparability problems (empty = self-describing).
scripts/bench/metadata.py-531-
scripts/bench/metadata.py-532-    A REQUIRED key that is ``None``/``""`` and a PRESENT key that is absent
scripts/bench/metadata.py-533-    both count.  Raises ``ValueError`` when ``strict`` and any problem exists —
--
scripts/bench/metadata.py-537-    """
scripts/bench/metadata.py-538-    missing = [k for k in REQUIRED_KEYS if _is_empty(md.get(k))]
scripts/bench/metadata.py-539-    absent = [f"{k}(absent)" for k in PRESENT_KEYS if k not in md]
scripts/bench/metadata.py-540-    problems = missing + absent
scripts/bench/metadata.py-541-    if strict and problems:
scripts/bench/metadata.py-542-        raise ValueError(
scripts/bench/metadata.py:543:            f"scaling metadata not self-describing: {problems}; a record that "
scripts/bench/metadata.py-544-            "cannot be compared must not be written (roadmap benchmark hygiene)."
scripts/bench/metadata.py-545-        )
scripts/bench/metadata.py-546-    return problems
scripts/bench/metadata.py-547-
scripts/bench/metadata.py-548-
scripts/bench/metadata.py-549-def annotate_incomplete(md: dict[str, Any], *, warn: bool = True) -> dict[str, Any]:
--
scripts/bench/metadata.py-552-    A benchmark record is built AFTER an expensive run has already completed,
scripts/bench/metadata.py-553-    so a hard raise here would throw away real data.  Instead: validate
scripts/bench/metadata.py-554-    non-strictly, and if the record is not self-describing, embed the problem
scripts/bench/metadata.py-555-    list under ``md["_incomplete"]`` and emit a ``RuntimeWarning`` so the row
scripts/bench/metadata.py-556-    is LOUDLY flagged and a downstream aggregator can skip/annotate it.
scripts/bench/metadata.py-557-    Returns ``md`` (mutated) for chaining.  Use this at JSON-write time; use
scripts/bench/metadata.py:558:    :func:`validate_scaling_metadata` (strict) where aborting is acceptable.
scripts/bench/metadata.py-559-    """
scripts/bench/metadata.py:560:    problems = validate_scaling_metadata(md, strict=False)
scripts/bench/metadata.py-561-    if problems:
scripts/bench/metadata.py-562-        md["_incomplete"] = problems
scripts/bench/metadata.py-563-        if warn:
scripts/bench/metadata.py-564-            import warnings
scripts/bench/metadata.py-565-
scripts/bench/metadata.py-566-            warnings.warn(
--
scripts/bench/sweep_Ah_long.py-101-
scripts/bench/sweep_Ah_long.py-102-for i, (d, eta_masked) in enumerate(eta_snapshots):
scripts/bench/sweep_Ah_long.py-103-    ax = axes[i // ncols, i % ncols]
scripts/bench/sweep_Ah_long.py-104-    im = ax.pcolormesh(lon, lat, eta_masked, cmap="RdBu_r",
scripts/bench/sweep_Ah_long.py-105-                      vmin=-0.08, vmax=0.08, shading="auto")
scripts/bench/sweep_Ah_long.py-106-    ax.set_title(f"Day {d:.0f}")
scripts/bench/sweep_Ah_long.py:107:    ax.set_xlabel("Longitude")
scripts/bench/sweep_Ah_long.py-108-    ax.set_ylabel("Latitude")
scripts/bench/sweep_Ah_long.py-109-    plt.colorbar(im, ax=ax, label="SSH (m)", shrink=0.8)
scripts/bench/sweep_Ah_long.py-110-
scripts/bench/sweep_Ah_long.py-111-for i in range(n_snaps, nrows * ncols):
scripts/bench/sweep_Ah_long.py-112-    axes[i // ncols, i % ncols].set_visible(False)
scripts/bench/sweep_Ah_long.py-113-
--
scripts/bench/bench_cube_tiled_step_scaling.py-14-production stepping is measured TODAY.
scripts/bench/bench_cube_tiled_step_scaling.py-15-
scripts/bench/bench_cube_tiled_step_scaling.py-16-Anti-fake-scaling guards:
scripts/bench/bench_cube_tiled_step_scaling.py-17-  * compiled-HLO census: the step must contain collective-permutes and NO
scripts/bench/bench_cube_tiled_step_scaling.py-18-    full-cube all-gather (``find_fullcube_allgathers`` — an all-gather means
scripts/bench/bench_cube_tiled_step_scaling.py-19-    replicated, not tiled, execution): the row is REFUSED otherwise;
scripts/bench/bench_cube_tiled_step_scaling.py:20:  * shared metadata v2 rows (virtual-CPU devices flagged; transport
scripts/bench/bench_cube_tiled_step_scaling.py-21-    auto-resolves) — a CPU smoke row can never masquerade as GPU scaling;
scripts/bench/bench_cube_tiled_step_scaling.py-22-  * --parity-gate: ONE tiled step vs one serial untiled step at the adapter
scripts/bench/bench_cube_tiled_step_scaling.py-23-    gate's f32-honest tolerances (the adapter is single-shot — tile-replicated
scripts/bench/bench_cube_tiled_step_scaling.py-24-    in, tile-sharded out; single-process only; smoke windows).
scripts/bench/bench_cube_tiled_step_scaling.py-25-
scripts/bench/bench_cube_tiled_step_scaling.py-26-Launch:
--
scripts/bench/bench_cube_tiled_step_scaling.py-54-from pathlib import Path
scripts/bench/bench_cube_tiled_step_scaling.py-55-
scripts/bench/bench_cube_tiled_step_scaling.py-56-import numpy as np
scripts/bench/bench_cube_tiled_step_scaling.py-57-
scripts/bench/bench_cube_tiled_step_scaling.py-58-sys.path.insert(0, str(Path(__file__).resolve().parent))
scripts/bench/bench_cube_tiled_step_scaling.py-59-
scripts/bench/bench_cube_tiled_step_scaling.py:60:from metadata import (  # noqa: E402
scripts/bench/bench_cube_tiled_step_scaling.py-61-    annotate_incomplete, count_collective_permutes, count_collectives,
scripts/bench/bench_cube_tiled_step_scaling.py:62:    scaling_metadata, tidy_throughput_fields)
scripts/bench/bench_cube_tiled_step_scaling.py-63-
scripts/bench/bench_cube_tiled_step_scaling.py-64-#: Parity tolerances vs the serial untiled step — the adapter gate's
scripts/bench/bench_cube_tiled_step_scaling.py-65-#: f32-honest bounds (exact f32 ulps of the field scales; a real stage
scripts/bench/bench_cube_tiled_step_scaling.py-66-#: regression is 2e-5-abs class).  These bound the SINGLE tiled-vs-serial
scripts/bench/bench_cube_tiled_step_scaling.py-67-#: step the parity gate checks (measured ~4e-6 u, ~3e-5 T, ~8e-3 p_s), and
scripts/bench/bench_cube_tiled_step_scaling.py-68-#: stay the adapter gate's own tolerances so a silent loosening here can't let
scripts/bench/bench_cube_tiled_step_scaling.py-69-#: a stage regression pass the lane.
scripts/bench/bench_cube_tiled_step_scaling.py-70-TILED_PARITY_ATOL = {"u": 2e-5, "v": 2e-5, "T": 1e-4, "p_s": 0.06}
scripts/bench/bench_cube_tiled_step_scaling.py-71-
scripts/bench/bench_cube_tiled_step_scaling.py-72-
scripts/bench/bench_cube_tiled_step_scaling.py:73:# Shared canonical CP census (metadata.count_collective_permutes); kept as a
scripts/bench/bench_cube_tiled_step_scaling.py-74-# module-level name for the existing test + call site.
scripts/bench/bench_cube_tiled_step_scaling.py-75-_count_collective_permutes = count_collective_permutes
scripts/bench/bench_cube_tiled_step_scaling.py-76-
scripts/bench/bench_cube_tiled_step_scaling.py-77-
scripts/bench/bench_cube_tiled_step_scaling.py-78-def main() -> int:
scripts/bench/bench_cube_tiled_step_scaling.py-79-    p = argparse.ArgumentParser(
--
scripts/bench/bench_cube_tiled_step_scaling.py-381-        **tidy_throughput_fields(
scripts/bench/bench_cube_tiled_step_scaling.py-382-            dt_seconds=args.dt, time_per_step_ms=med,
scripts/bench/bench_cube_tiled_step_scaling.py-383-            total_cells=total_cells),
scripts/bench/bench_cube_tiled_step_scaling.py-384-    )
scripts/bench/bench_cube_tiled_step_scaling.py-385-    from legoesm.parallel.early_init import nccl_transport_report
scripts/bench/bench_cube_tiled_step_scaling.py-386-    _nccl_report = nccl_transport_report()
scripts/bench/bench_cube_tiled_step_scaling.py:387:    rec["metadata"] = annotate_incomplete(scaling_metadata(
scripts/bench/bench_cube_tiled_step_scaling.py-388-        grid="cubed-sphere",
scripts/bench/bench_cube_tiled_step_scaling.py-389-        component="atmosphere",
scripts/bench/bench_cube_tiled_step_scaling.py-390-        resolution=f"C{args.resolution}",
scripts/bench/bench_cube_tiled_step_scaling.py-391-        n_levels=args.nlev,
scripts/bench/bench_cube_tiled_step_scaling.py-392-        precision=("float64" if jax.config.jax_enable_x64 else "float32"),
scripts/bench/bench_cube_tiled_step_scaling.py-393-        n_gpus=(n_devices if jax.default_backend() in ("gpu", "cuda",
--
scripts/bench/bench_cube_tiled_step_scaling.py-426-        print(json.dumps(rec))
scripts/bench/bench_cube_tiled_step_scaling.py-427-        print(f"[cube-tiled kt={args.kt} nd={n_devices} "
scripts/bench/bench_cube_tiled_step_scaling.py-428-              f"C{args.resolution}x{args.nlev}] "
scripts/bench/bench_cube_tiled_step_scaling.py-429-              f"compile={rec['compile_ms']}ms "
scripts/bench/bench_cube_tiled_step_scaling.py-430-              f"steady_median={med:.2f}ms/step "
scripts/bench/bench_cube_tiled_step_scaling.py-431-              f"ppermutes={n_ppermute}")
scripts/bench/bench_cube_tiled_step_scaling.py:432:        if rec["metadata"]["virtual_cpu_devices"]:
scripts/bench/bench_cube_tiled_step_scaling.py-433-            print("[virtual-cpu] forced host-platform CPU devices: this row "
scripts/bench/bench_cube_tiled_step_scaling.py-434-                  "is a communication-overhead / correctness proxy, NOT "
scripts/bench/bench_cube_tiled_step_scaling.py-435-                  "hardware scaling — do not report it as a speedup.")
scripts/bench/bench_cube_tiled_step_scaling.py-436-    return 0
scripts/bench/bench_cube_tiled_step_scaling.py-437-
scripts/bench/bench_cube_tiled_step_scaling.py-438-
--
scripts/bench/bench_cube_shardmap_halo.py-554-                    "SLURM_STEP_NUM_TASKS", "SLURM_NTASKS")
scripts/bench/bench_cube_shardmap_halo.py-555-
scripts/bench/bench_cube_shardmap_halo.py-556-
scripts/bench/bench_cube_shardmap_halo.py-557-def _launcher_world_size() -> int:
scripts/bench/bench_cube_shardmap_halo.py-558-    """Largest world size any launcher env var reports (1 if none present)."""
scripts/bench/bench_cube_shardmap_halo.py-559-    sizes = [int(v) for v in (os.environ.get(k) for k in _WORLD_SIZE_VARS)
scripts/bench/bench_cube_shardmap_halo.py:560:             if v is not None and v.isdigit()]
scripts/bench/bench_cube_shardmap_halo.py-561-    return max(sizes) if sizes else 1
scripts/bench/bench_cube_shardmap_halo.py-562-
scripts/bench/bench_cube_shardmap_halo.py-563-
scripts/bench/bench_cube_shardmap_halo.py-564-def _multihost_barrier(tag: str) -> None:
scripts/bench/bench_cube_shardmap_halo.py-565-    """Synchronise all processes (no-op for a single process)."""
scripts/bench/bench_cube_shardmap_halo.py-566-    import jax
--
scripts/bench/scaling_summary.py-1-"""Walk every ``output/baroclinic_wave_diagnostics_*.npz`` file and
scripts/bench/scaling_summary.py-2-emit a tidy CSV + Markdown table of throughput per (tag, grid, res,
scripts/bench/scaling_summary.py-3-backend, dt, days, sps).  The npz format added ``steps_per_sec``,
scripts/bench/scaling_summary.py-4-``wall_time_s``, and ``backend`` in iter-202 so this runs purely from
scripts/bench/scaling_summary.py:5:the npz metadata — no log-scraping required.
scripts/bench/scaling_summary.py-6-
scripts/bench/scaling_summary.py-7-Usage:
scripts/bench/scaling_summary.py-8-    PYTHONPATH=. .venv/bin/python scripts/scaling_summary.py [--tag-glob iter204]
scripts/bench/scaling_summary.py-9-"""
scripts/bench/scaling_summary.py-10-
scripts/bench/scaling_summary.py-11-from __future__ import annotations
--
scripts/bench/probe_w5_metrics.py-9-  eddy_rms        RMS of the zonal-asymmetry h' = h - zonalmean(h) over the
scripts/bench/probe_w5_metrics.py-10-                  whole sphere (area-weighted by cos lat).
scripts/bench/probe_w5_metrics.py-11-  eddy_rms_down   same, restricted to the *downstream* sector (east of the
scripts/bench/probe_w5_metrics.py-12-                  mountain at 90 E .. 270 E i.e. lon in [-90, +90] here after
scripts/bench/probe_w5_metrics.py-13-                  the W5 mountain is centred at lon = -90 / 270E) and the
scripts/bench/probe_w5_metrics.py-14-                  20 S .. 60 N band where the wave train lives.
scripts/bench/probe_w5_metrics.py:15:  east_reach_deg  furthest eastward longitude (deg from the mountain) at which
scripts/bench/probe_w5_metrics.py-16-                  |h'| zonal profile exceeds 20% of its global-max — a proxy
scripts/bench/probe_w5_metrics.py-17-                  for how far the train has propagated.
scripts/bench/probe_w5_metrics.py-18-  speed_max       max wind speed (m/s).
scripts/bench/probe_w5_metrics.py-19-
scripts/bench/probe_w5_metrics.py-20-Usage:
scripts/bench/probe_w5_metrics.py-21-  JAX_PLATFORMS=cpu .venv/bin/python scripts/probe_w5_metrics.py \
--
scripts/bench/bench_ocean_mpas_scaling.py-15-  --parity-gate         gathered owned cells vs the serial trajectory at the
scripts/bench/bench_ocean_mpas_scaling.py-16-                        re-association floor (smoke windows only).
scripts/bench/bench_ocean_mpas_scaling.py-17-  --check-conservation  global volume/heat/salt drift over the run.
scripts/bench/bench_ocean_mpas_scaling.py-18-
scripts/bench/bench_ocean_mpas_scaling.py-19-Anti-fake-scaling guards: multi-rank REQUIRES an armed VoronoiPartitionLayout
scripts/bench/bench_ocean_mpas_scaling.py-20-(a replicated global-mesh run cannot masquerade as decomposed); rows carry
scripts/bench/bench_ocean_mpas_scaling.py:21:the shared self-describing metadata (transport resolves to mpi4jax via
scripts/bench/bench_ocean_mpas_scaling.py-22-n_ranks > process_count) + voronoi partition-quality metrics.
scripts/bench/bench_ocean_mpas_scaling.py-23-
scripts/bench/bench_ocean_mpas_scaling.py-24-M1 measurement contract (scaling-M3d increment-1): the headline number is a
scripts/bench/bench_ocean_mpas_scaling.py:25:fused ``lax.scan`` block (``metadata.timed_scan_blocks``; per-step
scripts/bench/bench_ocean_mpas_scaling.py-26-dispatch latency probed SEPARATELY), cross-rank MAX-reduced, and it is what
scripts/bench/bench_ocean_mpas_scaling.py-27-the aggregator-facing ``steady_median_ms`` carries (the same deliberate
scripts/bench/bench_ocean_mpas_scaling.py-28-naming as ``bench_ocean_latlon_spmd_scaling``); the host-synced gate-loop
scripts/bench/bench_ocean_mpas_scaling.py-29-median is dispatch+sync LATENCY and is recorded only under
scripts/bench/bench_ocean_mpas_scaling.py-30-``step_latency_gate_loop_ms``, never as the headline.  Rows also carry
scripts/bench/bench_ocean_mpas_scaling.py-31-``wet_cell_metrics``, solver-iteration mode + a post-run zero-forcing
--
scripts/bench/bench_ocean_mpas_scaling.py-58-import sys
scripts/bench/bench_ocean_mpas_scaling.py-59-import time
scripts/bench/bench_ocean_mpas_scaling.py-60-from pathlib import Path
scripts/bench/bench_ocean_mpas_scaling.py-61-
scripts/bench/bench_ocean_mpas_scaling.py-62-import numpy as np
scripts/bench/bench_ocean_mpas_scaling.py-63-
scripts/bench/bench_ocean_mpas_scaling.py:64:# Bench dir for the shared metadata module (sibling-script import pattern).
scripts/bench/bench_ocean_mpas_scaling.py-65-sys.path.insert(0, str(Path(__file__).resolve().parent))
scripts/bench/bench_ocean_mpas_scaling.py-66-
scripts/bench/bench_ocean_mpas_scaling.py:67:from metadata import (  # noqa: E402
scripts/bench/bench_ocean_mpas_scaling.py-68-    annotate_incomplete,
scripts/bench/bench_ocean_mpas_scaling.py:69:    scaling_metadata,
scripts/bench/bench_ocean_mpas_scaling.py-70-    timed_scan_blocks,
scripts/bench/bench_ocean_mpas_scaling.py-71-    wet_cell_metrics,
scripts/bench/bench_ocean_mpas_scaling.py-72-)
scripts/bench/bench_ocean_mpas_scaling.py-73-
scripts/bench/bench_ocean_mpas_scaling.py-74-#: Parity tolerances (gathered MPI vs serial, f64/f32) — the re-association
scripts/bench/bench_ocean_mpas_scaling.py-75-#: floor of the rank-local step + owned-masked reductions over a SMOKE
--
scripts/bench/bench_ocean_mpas_scaling.py-144-                           if np.isfinite(imb).any() else float("nan")),
scripts/bench/bench_ocean_mpas_scaling.py-145-        "fused_step_ms": round(float(np.median(par)) / block_steps, 4),
scripts/bench/bench_ocean_mpas_scaling.py-146-    }
scripts/bench/bench_ocean_mpas_scaling.py-147-
scripts/bench/bench_ocean_mpas_scaling.py-148-
scripts/bench/bench_ocean_mpas_scaling.py-149-def stage_halo_note_for(n_ranks: int, halo_refresh: str):
scripts/bench/bench_ocean_mpas_scaling.py:150:    """Row-metadata companion to ``stage_halo_correct``.
scripts/bench/bench_ocean_mpas_scaling.py-151-
scripts/bench/bench_ocean_mpas_scaling.py-152-    Must describe the ACTUAL refresh selection (codex finding 5): a
scripts/bench/bench_ocean_mpas_scaling.py-153-    ``--halo-refresh none`` row suffers ACROSS-STEP halo rot on top of
scripts/bench/bench_ocean_mpas_scaling.py-154-    the within-step staleness every multi-rank row has — labeling it
scripts/bench/bench_ocean_mpas_scaling.py-155-    "per-step packed refresh" would misdescribe the evidence.
scripts/bench/bench_ocean_mpas_scaling.py-156-    """
--
scripts/bench/bench_ocean_mpas_scaling.py-410-            local = (os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
scripts/bench/bench_ocean_mpas_scaling.py-411-                     or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK")
scripts/bench/bench_ocean_mpas_scaling.py-412-                     or os.environ.get("PALS_LOCAL_RANKID"))
scripts/bench/bench_ocean_mpas_scaling.py-413-            if local is None:
scripts/bench/bench_ocean_mpas_scaling.py-414-                slid = os.environ.get("SLURM_LOCALID")
scripts/bench/bench_ocean_mpas_scaling.py-415-                nt = os.environ.get("SLURM_NTASKS", "1")
scripts/bench/bench_ocean_mpas_scaling.py:416:                if slid is not None and nt.isdigit() and int(nt) > 1:
scripts/bench/bench_ocean_mpas_scaling.py-417-                    local = slid
scripts/bench/bench_ocean_mpas_scaling.py-418-            os.environ["CUDA_VISIBLE_DEVICES"] = local or "0" 
scripts/bench/bench_ocean_mpas_scaling.py-419-        os.environ["JAX_PLATFORMS"] = "cuda"
scripts/bench/bench_ocean_mpas_scaling.py-420-        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
scripts/bench/bench_ocean_mpas_scaling.py-421-    if args.precision == "float64":
scripts/bench/bench_ocean_mpas_scaling.py-422-        import jax
--
scripts/bench/bench_ocean_mpas_scaling.py-827-        solver_iters=solver_iters,
scripts/bench/bench_ocean_mpas_scaling.py-828-        solver_iters_mode=solver_iters_mode,
scripts/bench/bench_ocean_mpas_scaling.py-829-        zero_forcing_probe_residual=zero_forcing_probe_residual,
scripts/bench/bench_ocean_mpas_scaling.py-830-        zero_forcing_probe_measured=zero_forcing_probe_measured,
scripts/bench/bench_ocean_mpas_scaling.py-831-        residual_reason=residual_reason,
scripts/bench/bench_ocean_mpas_scaling.py-832-    )
scripts/bench/bench_ocean_mpas_scaling.py:833:    rec["metadata"] = annotate_incomplete(scaling_metadata(
scripts/bench/bench_ocean_mpas_scaling.py-834-        grid="voronoi",
scripts/bench/bench_ocean_mpas_scaling.py-835-        component="ocean",
scripts/bench/bench_ocean_mpas_scaling.py-836-        resolution=f"L{subdivision}",
scripts/bench/bench_ocean_mpas_scaling.py-837-        n_levels=args.nlev,
scripts/bench/bench_ocean_mpas_scaling.py-838-        precision=args.precision,
scripts/bench/bench_ocean_mpas_scaling.py-839-        n_ranks=n_ranks,
--
scripts/bench/bench_ocean_mpas_scaling.py-894-            else "n/a (--block-steps 0: fused measurement disabled)")
scripts/bench/bench_ocean_mpas_scaling.py-895-        print(f"[mpas-ocean np={n_ranks} L{subdivision} "
scripts/bench/bench_ocean_mpas_scaling.py-896-              f"nCells={mesh.nCells} nlev={args.nlev} "
scripts/bench/bench_ocean_mpas_scaling.py-897-              f"solver={args.barotropic_solver} halo={halo_refresh}] "
scripts/bench/bench_ocean_mpas_scaling.py-898-              f"compile={rec['compile_ms']}ms fused={_fused_txt} "
scripts/bench/bench_ocean_mpas_scaling.py-899-              f"gate_loop_latency={gate_loop_med:.2f}ms/step")
scripts/bench/bench_ocean_mpas_scaling.py:900:        if rec["metadata"]["virtual_cpu_devices"]:
scripts/bench/bench_ocean_mpas_scaling.py-901-            print("[virtual-cpu] forced host-platform CPU devices: this row "
scripts/bench/bench_ocean_mpas_scaling.py-902-                  "is a communication-overhead / correctness proxy, NOT "
scripts/bench/bench_ocean_mpas_scaling.py-903-                  "hardware scaling — do not report it as a speedup.")
scripts/bench/bench_ocean_mpas_scaling.py-904-    return 0
scripts/bench/bench_ocean_mpas_scaling.py-905-
scripts/bench/bench_ocean_mpas_scaling.py-906-
--
scripts/bench/bcw_scaling_ledger.py-7-
scripts/bench/bcw_scaling_ledger.py-8-* ``snapshot`` — read the tidy CSV (``aggregate_bcw_scaling.py``), compute, for
scripts/bench/bcw_scaling_ledger.py-9-  every (backend, grid, case, precision) strong-scaling series, the strong
scripts/bench/bcw_scaling_ledger.py-10-  efficiency at the largest device count, the peak SYPD, the peak per-device
scripts/bench/bcw_scaling_ledger.py-11-  throughput (Mcells/s), and the max device count reached; append one ledger row
scripts/bench/bcw_scaling_ledger.py-12-  per series with an auto-incremented ``iteration`` index, a UTC timestamp, the
scripts/bench/bcw_scaling_ledger.py:13:  git commit, and a free-text ``--note``.
scripts/bench/bcw_scaling_ledger.py-14-* ``plot`` — render scaling metric vs iteration (strong efficiency -> 1.0 ideal;
scripts/bench/bcw_scaling_ledger.py-15-  per-device throughput -> roofline), one line per icosahedral series.
scripts/bench/bcw_scaling_ledger.py-16-
scripts/bench/bcw_scaling_ledger.py-17-Strong efficiency E = [SYPD(n_max)/SYPD(n_min)] / (n_max/n_min) on the resolution
scripts/bench/bcw_scaling_ledger.py-18-with the most device points in the series (ideal E=1).  Pure stdlib +
scripts/bench/bcw_scaling_ledger.py-19-matplotlib; no JAX -> runs on a login node.
--
scripts/bench/bcw_scaling_ledger.py-29-import math
scripts/bench/bcw_scaling_ledger.py-30-import subprocess
scripts/bench/bcw_scaling_ledger.py-31-from collections import defaultdict
scripts/bench/bcw_scaling_ledger.py-32-from pathlib import Path
scripts/bench/bcw_scaling_ledger.py-33-
scripts/bench/bcw_scaling_ledger.py-34-LEDGER_FIELDS = [
scripts/bench/bcw_scaling_ledger.py:35:    "iteration", "timestamp", "commit", "note",
scripts/bench/bcw_scaling_ledger.py-36-    "backend", "grid", "case", "precision",
scripts/bench/bcw_scaling_ledger.py-37-    "max_ndev", "peak_sypd", "strong_eff", "weak_eff", "peak_mcells_per_s",
scripts/bench/bcw_scaling_ledger.py-38-]
scripts/bench/bcw_scaling_ledger.py-39-
scripts/bench/bcw_scaling_ledger.py-40-
scripts/bench/bcw_scaling_ledger.py:41:def _git_commit() -> str:
scripts/bench/bcw_scaling_ledger.py-42-    try:
scripts/bench/bcw_scaling_ledger.py-43-        return subprocess.check_output(
scripts/bench/bcw_scaling_ledger.py:44:            ["git", "rev-parse", "--short", "HEAD"], text=True,
scripts/bench/bcw_scaling_ledger.py-45-            stderr=subprocess.DEVNULL).strip()
scripts/bench/bcw_scaling_ledger.py-46-    except Exception:
scripts/bench/bcw_scaling_ledger.py-47-        return ""
scripts/bench/bcw_scaling_ledger.py-48-
scripts/bench/bcw_scaling_ledger.py-49-
scripts/bench/bcw_scaling_ledger.py-50-def _f(x, default=0.0):
--
scripts/bench/bcw_scaling_ledger.py-145-        return list(csv.DictReader(f))
scripts/bench/bcw_scaling_ledger.py-146-
scripts/bench/bcw_scaling_ledger.py-147-
scripts/bench/bcw_scaling_ledger.py-148-def snapshot(tidy_csv: Path, ledger: Path, note: str, timestamp: str) -> int:
scripts/bench/bcw_scaling_ledger.py-149-    existing = _read_ledger(ledger)
scripts/bench/bcw_scaling_ledger.py-150-    its = [int(r["iteration"]) for r in existing
scripts/bench/bcw_scaling_ledger.py:151:           if str(r.get("iteration", "")).isdigit()]
scripts/bench/bcw_scaling_ledger.py-152-    it = (max(its) + 1) if its else 1
scripts/bench/bcw_scaling_ledger.py:153:    commit = _git_commit()
scripts/bench/bcw_scaling_ledger.py-154-    metric_rows = compute_rows(tidy_csv)
scripts/bench/bcw_scaling_ledger.py:155:    new = [{"iteration": it, "timestamp": timestamp, "commit": commit,
scripts/bench/bcw_scaling_ledger.py-156-            "note": note, **m} for m in metric_rows]
scripts/bench/bcw_scaling_ledger.py-157-    # Rewrite the WHOLE file with the current schema (append-mode would keep a
scripts/bench/bcw_scaling_ledger.py-158-    # stale header if LEDGER_FIELDS ever grows -> misaligned rows).
scripts/bench/bcw_scaling_ledger.py-159-    # extrasaction="ignore" drops any legacy/restkey columns; restval=""
scripts/bench/bcw_scaling_ledger.py-160-    # back-fills fields absent from older rows.
scripts/bench/bcw_scaling_ledger.py-161-    ledger.parent.mkdir(parents=True, exist_ok=True)
--
scripts/bench/bench_spectral_transform_micro.py-1-"""Spectral-transform microbenchmark (feasibility inputs, audit item 9).
scripts/bench/bench_spectral_transform_micro.py-2-
scripts/bench/bench_spectral_transform_micro.py:3:Times the spherical-harmonic ANALYSIS/SYNTHESIS pair (FFT-in-longitude +
scripts/bench/bench_spectral_transform_micro.py-4-dense Legendre GEMM) in isolation at several truncations, and reports the
scripts/bench/bench_spectral_transform_micro.py-5-GEMM problem SHAPES + FLOP counts + arithmetic intensity — the numbers the
scripts/bench/bench_spectral_transform_micro.py-6-GPU-native-transform go/no-go note
scripts/bench/bench_spectral_transform_micro.py-7-(docs/performance/scaling/spectral_gpu_feasibility.md) is grounded in.
scripts/bench/bench_spectral_transform_micro.py-8-
scripts/bench/bench_spectral_transform_micro.py-9-MICROBENCHMARK ONLY: no dycore, no multi-device, no rewrite.  Honest about
scripts/bench/bench_spectral_transform_micro.py:10:its backend: rows carry the live backend + the shared metadata, so a
scripts/bench/bench_spectral_transform_micro.py-11-CPU-only laptop row can never masquerade as the missing GPU measurement
scripts/bench/bench_spectral_transform_micro.py-12-(the note lists the exact command to reproduce on a CUDA node).
scripts/bench/bench_spectral_transform_micro.py-13-
scripts/bench/bench_spectral_transform_micro.py-14-Run:
scripts/bench/bench_spectral_transform_micro.py-15-  JAX_ENABLE_X64=1 python scripts/bench/bench_spectral_transform_micro.py \
scripts/bench/bench_spectral_transform_micro.py-16-      --truncations 42,85,170 --nlev 30 --out results/spectral_micro.json
--
scripts/bench/bench_spectral_transform_micro.py-25-from pathlib import Path
scripts/bench/bench_spectral_transform_micro.py-26-
scripts/bench/bench_spectral_transform_micro.py-27-import numpy as np
scripts/bench/bench_spectral_transform_micro.py-28-
scripts/bench/bench_spectral_transform_micro.py-29-sys.path.insert(0, str(Path(__file__).resolve().parent))
scripts/bench/bench_spectral_transform_micro.py-30-
scripts/bench/bench_spectral_transform_micro.py:31:from metadata import annotate_incomplete, scaling_metadata  # noqa: E402
scripts/bench/bench_spectral_transform_micro.py-32-
scripts/bench/bench_spectral_transform_micro.py-33-
scripts/bench/bench_spectral_transform_micro.py-34-def transform_flops(n_lat: int, n_lon: int, n_sh: int, nlev: int) -> dict:
scripts/bench/bench_spectral_transform_micro.py-35-    """Analytic cost model of ONE analysis+synthesis round trip.
scripts/bench/bench_spectral_transform_micro.py-36-
scripts/bench/bench_spectral_transform_micro.py-37-    Legendre leg = dense GEMM pair, ``(n_lat x n_sh) @ (n_sh x
scripts/bench/bench_spectral_transform_micro.py:38:    nlev-batch)`` per longitudinal wavenumber block (one einsum over the
scripts/bench/bench_spectral_transform_micro.py-39-    packed ``(n_lat, n_sh, nlev)`` tensor).  The Legendre matrices are
scripts/bench/bench_spectral_transform_micro.py-40-    REAL and the spectral fields complex: a real x complex MAC is 4 real
scripts/bench/bench_spectral_transform_micro.py-41-    FLOPs (2 mul + 2 add), not 8 (codex).  FFT leg: ``5 * n * log2(n)``
scripts/bench/bench_spectral_transform_micro.py-42-    per real transform row.
scripts/bench/bench_spectral_transform_micro.py-43-    Arithmetic intensity vs the ``(n_lat, n_sh, nlev)`` working set is the
scripts/bench/bench_spectral_transform_micro.py-44-    GPU-viability number (fp64 GEMM runs at tensor-core-less rates on most
--
scripts/bench/bench_spectral_transform_micro.py-139-            cost = transform_flops(n_lat, n_lon, n_sh, args.nlev)
scripts/bench/bench_spectral_transform_micro.py-140-            achieved = cost["gemm_flops"] / med_s / 1e9
scripts/bench/bench_spectral_transform_micro.py-141-            mode_name = "gemm" if gemm_mode else "legacy_segment_sum"
scripts/bench/bench_spectral_transform_micro.py-142-            row = {
scripts/bench/bench_spectral_transform_micro.py-143-                "truncation": T,
scripts/bench/bench_spectral_transform_micro.py-144-                "legendre_path": mode_name,
scripts/bench/bench_spectral_transform_micro.py:145:                # Live backend ON THE ROW (not only in the shared metadata
scripts/bench/bench_spectral_transform_micro.py-146-                # block): a CPU row must be self-labeling even when a
scripts/bench/bench_spectral_transform_micro.py-147-                # consumer copies the rows table alone (codex).
scripts/bench/bench_spectral_transform_micro.py-148-                "backend": str(jax.default_backend()),
scripts/bench/bench_spectral_transform_micro.py-149-                "device": str(jax.devices()[0]),
scripts/cluster/scaling_levante/_env.sh:26:REPO="${LEGOESM_REPO:-/work/bd1083/$USER/legoESM}"
scripts/cluster/scaling_levante/_env.sh:28:  echo "[_env.sh] REPO='$REPO' does not exist." >&2
scripts/cluster/scaling_levante/_env.sh:29:  if [ -z "${LEGOESM_REPO:-}" ]; then
scripts/cluster/scaling_levante/_env.sh:30:    echo "[_env.sh] LEGOESM_REPO is unset, so this is the per-user DEFAULT" >&2
scripts/cluster/scaling_levante/_env.sh:32:    echo "[_env.sh]   sbatch --export=ALL,LEGOESM_REPO=\$PWD,... " >&2
scripts/cluster/scaling_levante/_env.sh:66:PY="${LEGOESM_PYTHON:-$(command -v python)}"
scripts/cluster/scaling_levante/_env.sh:71:  echo "[_env.sh] PY='$PY' cannot import jax." >&2
scripts/cluster/scaling_levante/mpas_s10_192.sbatch:20:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/README.md:36:Set `LEGOESM_REPO`, `LEGOESM_CONDA_ENV`, and pin the `module load` versions to
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:32:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:52:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/_env.sh:26:REPO="${LEGOESM_REPO:-/work/bd1083/$USER/legoESM}"
scripts/cluster/scaling_levante/_env.sh:28:  echo "[_env.sh] REPO='$REPO' does not exist." >&2
scripts/cluster/scaling_levante/_env.sh:29:  if [ -z "${LEGOESM_REPO:-}" ]; then
scripts/cluster/scaling_levante/_env.sh:30:    echo "[_env.sh] LEGOESM_REPO is unset, so this is the per-user DEFAULT" >&2
scripts/cluster/scaling_levante/_env.sh:32:    echo "[_env.sh]   sbatch --export=ALL,LEGOESM_REPO=\$PWD,... " >&2
scripts/cluster/scaling_levante/_env.sh:66:PY="${LEGOESM_PYTHON:-$(command -v python)}"
scripts/cluster/scaling_levante/_env.sh:71:  echo "[_env.sh] PY='$PY' cannot import jax." >&2
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:89:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:30:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch:21:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:32:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:35:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_bound_base.sbatch:22:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/ocean_latlon_hundreds.sbatch:22:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:25:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:17:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_latlon_144.sbatch:30:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/prewarm_s10.sbatch:18:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:25:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/bench/bench_mpas_spmd_scaling.py:99:def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
scripts/bench/bench_mpas_spmd_scaling.py:103:    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
scripts/bench/bench_mpas_spmd_scaling.py:122:    mesh = reorder_voronoi_for_sharding(mesh, reorder_target, method=method)
scripts/bench/bench_mpas_spmd_scaling.py:124:        # Padding only guarantees divisibility for reorder_target.
scripts/bench/bench_mpas_spmd_scaling.py:128:            f"count divides --reorder-for ({reorder_target}).")
scripts/bench/bench_voronoi_partition_methods.py:184:def parse_expect_rounds(spec: str) -> dict:
scripts/bench/bench_voronoi_partition_methods.py:263:    ``reorder_target = n_ranks``, and ``reorder_voronoi_for_sharding`` pads
scripts/bench/bench_voronoi_partition_methods.py:292:        "reorder_target": cost["reorder_target"],
scripts/bench/bench_voronoi_partition_methods.py:293:        "already_reordered": cost["already_reordered"],
scripts/bench/bench_voronoi_partition_methods.py:346:    expect = parse_expect_rounds(args.expect_rounds)
scripts/bench/bench_voronoi_partition_methods.py:439:        payload["expected_rounds_check"] = {
tests/bench/test_bench_voronoi_partition_methods.py:204:        "reorder_target": 8, "already_reordered": False,
tests/bench/test_bench_voronoi_partition_methods.py:281:    assert row["schedule"]["reorder_target"] == row["n_ranks"]
tests/bench/test_bench_voronoi_partition_methods.py:282:    assert row["schedule"]["already_reordered"] is False
tests/bench/test_bench_voronoi_partition_methods.py:285:def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
tests/bench/test_bench_voronoi_partition_methods.py:317:def test_expect_rounds_gate_fails_loudly_and_never_vacuously(
tests/bench/test_bench_voronoi_partition_methods.py:335:    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
tests/bench/test_bench_voronoi_partition_methods.py:342:    assert json.loads(bad.read_text())["expected_rounds_check"]["failures"]
tests/bench/test_bench_voronoi_partition_methods.py:350:    fails = json.loads(missing.read_text())["expected_rounds_check"]["failures"]
tests/bench/test_bench_voronoi_partition_methods.py:354:def test_expect_rounds_refuses_to_pass_vacuously_without_scoring(monkeypatch):
tests/bench/test_bench_voronoi_partition_methods.py:366:def test_expect_rounds_rejects_malformed_specs(spec):
tests/bench/test_bench_voronoi_partition_methods.py:370:        mod.parse_expect_rounds(spec)
tests/bench/test_bench_mpas_spmd_gates.py:63:def test_lloyd_flag_reaches_the_mesh_builder(monkeypatch):
tests/parallel/test_spmd_schedule_cost.py:62:def test_already_reordered_mesh_is_not_reordered_again(mesh):
tests/parallel/test_spmd_schedule_cost.py:67:    pre = sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
tests/parallel/test_spmd_schedule_cost.py:70:    assert pre["reorder_target"] is None, (
tests/parallel/test_spmd_schedule_cost.py:76:    with pytest.raises(ValueError, match="reorder_target is meaningless"):
tests/parallel/test_spmd_schedule_cost.py:77:        sd.spmd_schedule_cost(prepared, 8, already_reordered=True,
tests/parallel/test_spmd_schedule_cost.py:78:                              reorder_target=16)
tests/parallel/test_spmd_schedule_cost.py:81:def test_reorder_target_differing_from_run_device_count(mesh):
tests/parallel/test_spmd_schedule_cost.py:87:                                         reorder_target=16)
tests/parallel/test_spmd_schedule_cost.py:88:    assert split_for_16["reorder_target"] == 16
tests/parallel/test_spmd_schedule_cost.py:93:        "reorder_target had no effect — the argument would be decorative")
tests/parallel/test_spmd_schedule_cost.py:128:def test_already_reordered_does_not_reorder(mesh, monkeypatch):
tests/parallel/test_spmd_schedule_cost.py:138:    sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
tests/parallel/test_spmd_schedule_cost.py:139:    assert not calls, "already_reordered=True must not reorder the mesh"
tests/parallel/test_spmd_schedule_cost.py:142:def test_indivisible_reorder_target_is_refused(mesh):
tests/parallel/test_spmd_schedule_cost.py:147:        sd.spmd_schedule_cost(mesh, 4, method="sfc", reorder_target=3)
tests/parallel/test_ppermute_multiface.py:116:    "k, expect_rounds, expect_slots",
tests/parallel/test_ppermute_multiface.py:119:def test_multiface_schedule_static_properties(k, expect_rounds, expect_slots):
tests/parallel/test_ppermute_multiface.py:129:    assert len(t.perms) == expect_rounds
.venv

exec
/usr/bin/bash -lc 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider tests/bench/test_bench_voronoi_partition_methods.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 1 in 2828ms:
Traceback (most recent call last):
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/pytest", line 10, in <module>
    sys.exit(console_main())
             ~~~~~~~~~~~~^^
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

exec
/usr/bin/bash -lc 'rg -n -A 120 -B 20 "def create_voronoi_mesh" packages/core/legoesm/grids/voronoi.py
rg -n "lloyd_iterations" packages/core/legoesm/grids/voronoi.py | head -n 100
rg -n -C 6 "only a|ownership.*only|only.*ownership|recolour|recolor|DIRECTED|UNDIRECTED|gap == 0" scripts/bench/bench_voronoi_partition_methods.py scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch packages/core/legoesm/parallel/sharded_dynamics.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 199ms:
1223-    radius: float = constants.R_earth,
1224-    lloyd_iterations: int = 50,
1225-    omega: float = constants.Omega,
1226-) -> str:
1227-    """Build (if absent) and cache the uniform SCVT mesh; return the cache path.
1228-
1229-    Call ONCE from a single process before an MPI launch so every rank loads the
1230-    mesh instead of redundantly rebuilding it.  Cheap no-op when already cached
1231-    (create_voronoi_mesh returns the cached mesh without rebuilding).
1232-    """
1233-    if _voronoi_cache_disabled():
1234-        raise RuntimeError(
1235-            f"{_MESH_CACHE_DISABLE_ENV} is set; cannot pre-warm a disabled cache.")
1236-    path = _voronoi_cache_path(subdivision_level, radius, lloyd_iterations, omega)
1237-    create_voronoi_mesh(
1238-        subdivision_level, radius=radius,
1239-        lloyd_iterations=lloyd_iterations, omega=omega)
1240-    return path
1241-
1242-
1243:def create_voronoi_mesh(
1244-    subdivision_level: int,
1245-    radius: float = constants.R_earth,
1246-    lloyd_iterations: int = 50,
1247-    omega: float = constants.Omega,
1248-    density_fn=None,
1249-) -> VoronoiMesh:
1250-    """Create a centroidal Voronoi tessellation (SCVT) on the sphere.
1251-
1252-    Starts from an icosahedral triangulation, bisects to the desired
1253-    level, applies Lloyd relaxation, then builds the full MPAS-compatible
1254-    mesh with all connectivity and geometric arrays.
1255-
1256-    Parameters
1257-    ----------
1258-    subdivision_level : int
1259-        Number of bisection levels. nCells = 10*4^level + 2.
1260-        level=3: 642 cells, level=4: 2562, level=5: 10242.
1261-    radius : float
1262-        Sphere radius [m]. Default: Earth radius.
1263-    lloyd_iterations : int
1264-        Number of Lloyd relaxation iterations. Default: 50.
1265-    omega : float
1266-        Rotation rate [rad/s]. Default: Earth rotation.
1267-    density_fn : callable(lat, lon) -> float, optional
1268-        Relative mesh-density function (``lat``, ``lon`` in radians; larger =>
1269-        finer cells).  When provided, the Lloyd relaxation is **density-weighted**
1270-        (Du–Faber–Gunzburger SCVT): generators concentrate where the density is
1271-        high, yielding a VARIABLE-RESOLUTION mesh refined over the high-density
1272-        region (the MPAS variable-resolution capability).  Cell area relaxes toward
1273-        ``~ 1/density`` (weighting by ``density**2``), so a region with relative
1274-        density ``d`` gets roughly ``d`` x smaller cells.  ``None`` (default) =>
1275-        the uniform quasi-uniform SCVT (unchanged).  Host-side mesh generation —
1276-        the callable is plain NumPy, never traced.  NOTE: Lloyd relaxation converges
1277-        linearly, so the achieved refinement contrast grows with
1278-        ``lloyd_iterations``; for strong/precise variable resolution prefer a
1279-        JIGSAW-built mesh loaded via :func:`load_mpas_mesh`.
1280-
1281-    Returns
1282-    -------
1283-    VoronoiMesh
1284-    """
1285-    if subdivision_level > _BIG_MESH_MAX_LEVEL:
1286-        n_cells = 10 * 4 ** subdivision_level + 2
1287-        raise ValueError(
1288-            f"subdivision_level={subdivision_level} would create {n_cells:.2e} "
1289-            f"cells — unsupported (hard cap {_BIG_MESH_MAX_LEVEL}; the scipy "
1290-            f"SphericalVoronoi build path does not scale there). For higher "
1291-            f"resolutions, use load_mpas_mesh() with a pre-built mesh file."
1292-        )
1293-    # Levels above the routine cap are CACHE-OR-PREWARM only (codex round-19
1294-    # design): a valid cache hit is always admissible; a MISS is refused
1295-    # unless this process is the designated prewarmer — otherwise an N-rank
1296-    # MPI launch would have every rank silently rebuild for hours (build is
1297-    # ~87 s/Lloyd-iteration at subdiv-8, ~4x that at 9). The prewarm path is
1298-    # single-builder: an exclusive lockfile serialises concurrent opt-ins.
1299-    _big = subdivision_level > _BIG_MESH_ROUTINE_LEVEL
1300-
1301-    # Disk cache: a uniform SCVT mesh is deterministic in these args, so skip the
1302-    # expensive rebuild on a hit.  density_fn meshes are NOT cached (a callable
1303-    # has no stable key); the env switch lets a run force a fresh build.
1304-    use_cache = density_fn is None and not _voronoi_cache_disabled()
1305-    cache_path = None
1306-    if use_cache:
1307-        cache_path = _voronoi_cache_path(
1308-            subdivision_level, radius, lloyd_iterations, omega)
1309-        cached = _load_voronoi_cache(cache_path)
1310-        if cached is not None:
1311-            return cached
1312-
1313-    _lock_path = None
1314-    if _big:
1315-        if not use_cache:
1316-            raise ValueError(
1317-                f"subdivision_level={subdivision_level} needs the mesh disk "
1318-                f"cache (density_fn=None and {_MESH_CACHE_DISABLE_ENV} unset)"
1319-                f" — an uncached big-mesh build would repeat per rank.")
1320-        if os.environ.get(_BIG_MESH_BUILD_ENV, "") != "1":
1321-            raise ValueError(
1322-                f"subdivision_level={subdivision_level}: no cached mesh at "
1323-                f"{cache_path} and this process is not the designated "
1324-                f"prewarmer. Build the cache ONCE via scripts/data/"
1325-                f"prewarm_voronoi_mesh.py (or set {_BIG_MESH_BUILD_ENV}=1 in "
1326-                f"a SINGLE-process job), then rerun.")
1327-        # Single-builder lock (codex round-19: the opt-in alone is a
1328-        # thundering herd — N authorized ranks could all miss). O_EXCL
1329-        # lockfile beside the cache; losers wait for the winner's atomic
1330-        # os.replace to land and then load it. Stale locks (builder died)
1331-        # are stolen after _BIG_MESH_LOCK_STALE_S with a warning.
1332-        _lock_path = f"{cache_path}.lock"
1333-        import time as _time
1334-        while True:
1335-            try:
1336-                _fd = os.open(_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
1337-                os.write(_fd, str(os.getpid()).encode())
1338-                os.close(_fd)
1339-                break                      # we are the builder
1340-            except FileExistsError:
1341-                for _ in range(int(_BIG_MESH_LOCK_STALE_S)):
1342-                    _time.sleep(1.0)
1343-                    cached = _load_voronoi_cache(cache_path)
1344-                    if cached is not None:
1345-                        return cached      # winner finished; use its mesh
1346-                    if not os.path.exists(_lock_path):
1347-                        break              # lock released without a cache?!
1348-                else:
1349-                    try:                   # stale: builder likely died
1350-                        os.unlink(_lock_path)
1351-                    except FileNotFoundError:
1352-                        pass
1353-                    import warnings
1354-                    warnings.warn(
1355-                        f"stale big-mesh lock {_lock_path} removed after "
1356-                        f"{_BIG_MESH_LOCK_STALE_S}s; taking over the build.")
1357-
1358-    # Step 1: Icosahedral base
1359-    verts, triangles = _icosahedral_base()
1360-
1361-    # Step 2: Bisect to desired resolution
1362-    if subdivision_level > 0:
1363-        verts, triangles = _bisect_mesh(verts, triangles, subdivision_level)
1095:# DETERMINISTIC in (subdivision_level, radius, lloyd_iterations, omega) when
1143:    subdivision_level: int, radius: float, lloyd_iterations: int, omega: float
1154:        f"_r{radius!r}_omega{omega!r}_lloyd{lloyd_iterations}"
1224:    lloyd_iterations: int = 50,
1236:    path = _voronoi_cache_path(subdivision_level, radius, lloyd_iterations, omega)
1239:        lloyd_iterations=lloyd_iterations, omega=omega)
1246:    lloyd_iterations: int = 50,
1263:    lloyd_iterations : int
1278:        ``lloyd_iterations``; for strong/precise variable resolution prefer a
1308:            subdivision_level, radius, lloyd_iterations, omega)
1366:    if lloyd_iterations > 0 and subdivision_level > 0:
1368:            verts, n_iter=lloyd_iterations, density_fn=density_fn)
1371:            "density_fn requires lloyd_iterations > 0 and subdivision_level > 0 "
packages/core/legoesm/parallel/sharded_dynamics.py-157-      level axis is reserved for downstream column-wise physics
packages/core/legoesm/parallel/sharded_dynamics.py-158-      to shard.  Returning ``P()`` for both 3D and 4D specs
packages/core/legoesm/parallel/sharded_dynamics.py-159-      makes ``shard_state`` produce a fully-replicated dycore
packages/core/legoesm/parallel/sharded_dynamics.py-160-      state, which is the correct behavior on the level mesh.
packages/core/legoesm/parallel/sharded_dynamics.py-161-    """
packages/core/legoesm/parallel/sharded_dynamics.py-162-    if getattr(config, "grid_type", None) == "cubed_sphere_level":
packages/core/legoesm/parallel/sharded_dynamics.py:163:        # Replicated-dycore path.  Mesh has only a ``'level'`` axis;
packages/core/legoesm/parallel/sharded_dynamics.py-164-        # any attempt to address ``'face'`` would crash with
packages/core/legoesm/parallel/sharded_dynamics.py-165-        # ``unmatched mesh axis`` from JAX.
packages/core/legoesm/parallel/sharded_dynamics.py-166-        return ShardingSpec(
packages/core/legoesm/parallel/sharded_dynamics.py-167-            face_3d=P(),
packages/core/legoesm/parallel/sharded_dynamics.py-168-            face_2d=P(),
packages/core/legoesm/parallel/sharded_dynamics.py-169-            replicated=P(),
--
packages/core/legoesm/parallel/sharded_dynamics.py-716-    (``cubesphere_exchange.activate_spmd_halo_backend``): ``pad_halo``
packages/core/legoesm/parallel/sharded_dynamics.py-717-    then routes through shard_map ppermute kernels (multiface; one-face
packages/core/legoesm/parallel/sharded_dynamics.py-718-    at halo=1 with 6 devices) instead of relying on XLA's implicit
packages/core/legoesm/parallel/sharded_dynamics.py-719-    cross-shard reads.  Letting GSPMD auto-insert collectives for the
packages/core/legoesm/parallel/sharded_dynamics.py-720-    cross-face reads — the pre-activation behavior this Notes section
packages/core/legoesm/parallel/sharded_dynamics.py-721-    used to describe — replicates ALL compute per device (HLO probe job
packages/core/legoesm/parallel/sharded_dynamics.py:722:    8456476); the all_gather kernels survive only as the explicit
packages/core/legoesm/parallel/sharded_dynamics.py-723-    ``LEGOESM_SPMD_FORCE_ALLGATHER=1`` diagnostic.
packages/core/legoesm/parallel/sharded_dynamics.py-724-
packages/core/legoesm/parallel/sharded_dynamics.py-725-    For sub-face tiling (>6 devices), additional tile-boundary
packages/core/legoesm/parallel/sharded_dynamics.py-726-    exchange is needed; the SPMD halo backend is NOT activated there
packages/core/legoesm/parallel/sharded_dynamics.py-727-    (the tiled path is unvalidated — bench guards exclude it).
packages/core/legoesm/parallel/sharded_dynamics.py-728-    """
--
packages/core/legoesm/parallel/sharded_dynamics.py-781-    #    limit (63820333056)"  (job 26495955, C768/L60 f32 at kt=4/96 devices)
packages/core/legoesm/parallel/sharded_dynamics.py-782-    # -- which reads as an OOM, not as "this tiling is not supported". That is
packages/core/legoesm/parallel/sharded_dynamics.py-783-    # the silent-fallback pattern the dispatch-hardening rule exists to kill:
packages/core/legoesm/parallel/sharded_dynamics.py-784-    # refuse loudly instead, naming what IS validated.
packages/core/legoesm/parallel/sharded_dynamics.py-785-    if _tiled_requested and _tiling[0] not in VALIDATED_TILE_FACTORS:
packages/core/legoesm/parallel/sharded_dynamics.py-786-        raise ValueError(
packages/core/legoesm/parallel/sharded_dynamics.py:787:            f"tiled cube SPMD is bit-identity-validated only at kt in "
packages/core/legoesm/parallel/sharded_dynamics.py-788-            f"{sorted(VALIDATED_TILE_FACTORS)} (6*kt^2 = "
packages/core/legoesm/parallel/sharded_dynamics.py-789-            f"{[6 * k * k for k in sorted(VALIDATED_TILE_FACTORS)]} devices); "
packages/core/legoesm/parallel/sharded_dynamics.py-790-            f"got kt={_tiling[0]} ({_n} devices). Running it would NOT shard: "
packages/core/legoesm/parallel/sharded_dynamics.py-791-            f"the step falls back to replicating the global state on every "
packages/core/legoesm/parallel/sharded_dynamics.py-792-            f"device and dies with an XLA argument-size error that looks like "
packages/core/legoesm/parallel/sharded_dynamics.py-793-            f"an OOM (#1360). Validate that kt the way kt=2/3 were "
--
packages/core/legoesm/parallel/sharded_dynamics.py-1271-      configured integrator -- SSP-RK3 evaluates 3 times, but the MPAS default
packages/core/legoesm/parallel/sharded_dynamics.py-1272-      is ``ssp_rk54_scan``.  Multiply with the integrator you actually run.
packages/core/legoesm/parallel/sharded_dynamics.py-1273-    * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
packages/core/legoesm/parallel/sharded_dynamics.py-1274-      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
packages/core/legoesm/parallel/sharded_dynamics.py-1275-      keeps the best; equality is MEASURED (compare the returned
packages/core/legoesm/parallel/sharded_dynamics.py-1276-      ``max_degree``), never assumed.  Do not claim "the colouring is already
packages/core/legoesm/parallel/sharded_dynamics.py:1277:      optimal so only ownership can help" from this function.
packages/core/legoesm/parallel/sharded_dynamics.py-1278-    * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
packages/core/legoesm/parallel/sharded_dynamics.py-1279-      cells/device is below ``ppermute_cells_per_device_threshold``, in which
packages/core/legoesm/parallel/sharded_dynamics.py-1280-      case there is no ppermute schedule and this number is counterfactual --
packages/core/legoesm/parallel/sharded_dynamics.py-1281-      see the returned ``production_strategy``.
packages/core/legoesm/parallel/sharded_dynamics.py-1282-
packages/core/legoesm/parallel/sharded_dynamics.py-1283-    MESH STATE -- the one thing that silently invalidates the score
--
packages/core/legoesm/parallel/sharded_dynamics.py-1850-    n_rounds_greedy = max(greedy_colors.values()) + 1
packages/core/legoesm/parallel/sharded_dynamics.py-1851-    multi_colors, max_degree = _multi_ordering_edge_coloring(comm_pairs)
packages/core/legoesm/parallel/sharded_dynamics.py-1852-    n_rounds_multi = max(multi_colors.values()) + 1
packages/core/legoesm/parallel/sharded_dynamics.py-1853-    # Adopt the multi-start coloring ONLY when it STRICTLY reduces rounds;
packages/core/legoesm/parallel/sharded_dynamics.py-1854-    # on a tie keep the exact legacy sorted-greedy coloring so the produced
packages/core/legoesm/parallel/sharded_dynamics.py-1855-    # schedule is byte-identical to before wherever there is no round win
packages/core/legoesm/parallel/sharded_dynamics.py:1856:    # (the win only appears at high device counts — >=16 on the probed
packages/core/legoesm/parallel/sharded_dynamics.py-1857-    # MPAS meshes). Both colorings are proper.
packages/core/legoesm/parallel/sharded_dynamics.py-1858-    if n_rounds_multi < n_rounds_greedy:
packages/core/legoesm/parallel/sharded_dynamics.py-1859-        edge_colors, n_rounds, coloring_method = (
packages/core/legoesm/parallel/sharded_dynamics.py-1860-            multi_colors, n_rounds_multi, "multi_greedy")
packages/core/legoesm/parallel/sharded_dynamics.py-1861-    else:
packages/core/legoesm/parallel/sharded_dynamics.py-1862-        edge_colors, n_rounds, coloring_method = (
--
scripts/bench/bench_voronoi_partition_methods.py-17-   every method and rank count while the real depth-3-plus-closure
scripts/bench/bench_voronoi_partition_methods.py-18-   schedule reported 12-14.  Scored by the production
scripts/bench/bench_voronoi_partition_methods.py-19-   ``spmd_schedule_cost`` (which calls the production builders), never a
scripts/bench/bench_voronoi_partition_methods.py-20-   re-derived lookalike.
scripts/bench/bench_voronoi_partition_methods.py-21-
scripts/bench/bench_voronoi_partition_methods.py-22-   The DECISIVE column is ``coloring_gap = n_rounds - max_degree``, read
scripts/bench/bench_voronoi_partition_methods.py:23:   through VIZING'S THEOREM, which bounds what recolouring could ever buy.
scripts/bench/bench_voronoi_partition_methods.py-24-   The schedule is a proper EDGE colouring of the device communication
scripts/bench/bench_voronoi_partition_methods.py-25-   graph (one colour = one ppermute round; ``_build_ppermute_schedule``
scripts/bench/bench_voronoi_partition_methods.py-26-   asserts properness), and ``max_degree`` is that same graph's maximum
scripts/bench/bench_voronoi_partition_methods.py-27-   vertex degree.  So the chromatic index obeys ``Delta <= chi' <=
scripts/bench/bench_voronoi_partition_methods.py:28:   Delta + 1``: the gap is a bound on recolouring headroom, NOT a
scripts/bench/bench_voronoi_partition_methods.py-29-   yes/no flag.
scripts/bench/bench_voronoi_partition_methods.py-30-
scripts/bench/bench_voronoi_partition_methods.py:31:   * ``gap == 0`` -> ``n_rounds == Delta``, and no proper edge colouring
scripts/bench/bench_voronoi_partition_methods.py:32:     can beat ``Delta``.  The colouring is PROVABLY OPTIMAL; recolouring
scripts/bench/bench_voronoi_partition_methods.py-33-     headroom is exactly ZERO.
scripts/bench/bench_voronoi_partition_methods.py-34-   * ``gap == 1`` -> INCONCLUSIVE.  A Class 2 graph genuinely needs
scripts/bench/bench_voronoi_partition_methods.py-35-     ``Delta + 1``, and deciding Class 1 vs Class 2 is NP-complete, so
scripts/bench/bench_voronoi_partition_methods.py-36-     this neither establishes nor excludes a one-round win.
scripts/bench/bench_voronoi_partition_methods.py:37:   * ``gap >= 2`` -> recolouring is guaranteed to remove AT LEAST
scripts/bench/bench_voronoi_partition_methods.py-38-     ``gap - 1`` rounds (the optimum is at worst ``Delta + 1``) and at
scripts/bench/bench_voronoi_partition_methods.py-39-     most ``gap``.
scripts/bench/bench_voronoi_partition_methods.py-40-
scripts/bench/bench_voronoi_partition_methods.py:41:   SCOPE, and it is not a formality: ``Delta`` bounds only a proper
scripts/bench/bench_voronoi_partition_methods.py:42:   UNDIRECTED edge colouring of THIS graph.  ``_build_ppermute_schedule``
scripts/bench/bench_voronoi_partition_methods.py-43-   enters a device pair into ``comm_pairs`` when EITHER direction has a
scripts/bench/bench_voronoi_partition_methods.py-44-   halo dependency and then emits BOTH ppermute directions, even where one
scripts/bench/bench_voronoi_partition_methods.py:45:   send map is empty.  A redesigned DIRECTED schedule that exploits
scripts/bench/bench_voronoi_partition_methods.py-46-   one-way exchanges is therefore not bounded by ``Delta`` at all, so
scripts/bench/bench_voronoi_partition_methods.py:47:   ``gap == 0`` must never be reported as "only ownership can help" — it
scripts/bench/bench_voronoi_partition_methods.py-48-   rules out a better undirected edge colouring of this graph, and
scripts/bench/bench_voronoi_partition_methods.py-49-   nothing more.
scripts/bench/bench_voronoi_partition_methods.py-50-
scripts/bench/bench_voronoi_partition_methods.py-51-   NOTE the halo depth differs by lane and is NOT ``--halo-depth``: that
scripts/bench/bench_voronoi_partition_methods.py-52-   flag is the MPI lane's (default 2), while the schedule is scored at the
scripts/bench/bench_voronoi_partition_methods.py-53-   SPMD production depth ``SPMD_HALO_DEPTH`` (3).  Both are recorded.
--
scripts/bench/bench_voronoi_partition_methods.py-247-
scripts/bench/bench_voronoi_partition_methods.py-248-    Adds ``coloring_gap = n_rounds - max_degree`` and the Vizing reading of
scripts/bench/bench_voronoi_partition_methods.py-249-    it (see the module docstring).  The headroom is reported as an INTERVAL,
scripts/bench/bench_voronoi_partition_methods.py-250-    because Vizing pins the optimum only to ``{Delta, Delta + 1}``:
scripts/bench/bench_voronoi_partition_methods.py-251-
scripts/bench/bench_voronoi_partition_methods.py-252-    * ``coloring_headroom_rounds_min = max(0, gap - 1)`` — rounds a perfect
scripts/bench/bench_voronoi_partition_methods.py:253:      recolouring is GUARANTEED to remove (it beats the ``Delta + 1`` case).
scripts/bench/bench_voronoi_partition_methods.py-254-    * ``coloring_headroom_rounds_max = gap`` — the best case, realized only
scripts/bench/bench_voronoi_partition_methods.py-255-      if the graph is Class 1.
scripts/bench/bench_voronoi_partition_methods.py-256-
scripts/bench/bench_voronoi_partition_methods.py:257:    It is deliberately NOT a "recolour vs ownership" verdict: at
scripts/bench/bench_voronoi_partition_methods.py-258-    ``gap == 1`` the min is 0 and the max is 1, i.e. genuinely inconclusive.
scripts/bench/bench_voronoi_partition_methods.py-259-
scripts/bench/bench_voronoi_partition_methods.py-260-    Errors are NOT caught.  The scorer's one refusal — a mesh padded for a
scripts/bench/bench_voronoi_partition_methods.py-261-    different reorder target, which would mis-slice the owned blocks — is
scripts/bench/bench_voronoi_partition_methods.py-262-    unreachable from here: this passes the raw mesh with the scorer's default
scripts/bench/bench_voronoi_partition_methods.py-263-    ``reorder_target = n_ranks``, and ``reorder_voronoi_for_sharding`` pads
--
scripts/bench/bench_voronoi_partition_methods.py-277-    return {
scripts/bench/bench_voronoi_partition_methods.py-278-        "n_rounds": int(cost["n_rounds"]),
scripts/bench/bench_voronoi_partition_methods.py-279-        "n_rounds_greedy": int(cost["n_rounds_greedy"]),
scripts/bench/bench_voronoi_partition_methods.py-280-        "max_degree": int(cost["max_degree"]),
scripts/bench/bench_voronoi_partition_methods.py-281-        # The decisive column, read through Vizing (see module docstring).
scripts/bench/bench_voronoi_partition_methods.py-282-        "coloring_gap": gap,
scripts/bench/bench_voronoi_partition_methods.py:283:        # Rounds a perfect recolouring could remove, as an INTERVAL: Vizing
scripts/bench/bench_voronoi_partition_methods.py-284-        # pins the optimum to {Delta, Delta+1}, so gap-1 is guaranteed and
scripts/bench/bench_voronoi_partition_methods.py-285-        # gap is the best case.  A gap of 1 spans [0, 1] = inconclusive.
scripts/bench/bench_voronoi_partition_methods.py-286-        "coloring_headroom_rounds_min": max(0, gap - 1),
scripts/bench/bench_voronoi_partition_methods.py-287-        "coloring_headroom_rounds_max": gap,
scripts/bench/bench_voronoi_partition_methods.py:288:        "coloring_optimal_proven": gap == 0,
scripts/bench/bench_voronoi_partition_methods.py-289-        # Scorer provenance, carried per row: a copied/flattened row must be
scripts/bench/bench_voronoi_partition_methods.py-290-        # able to show it scored a raw mesh partitioned for THIS device
scripts/bench/bench_voronoi_partition_methods.py-291-        # count, not one reordered for a different target.
scripts/bench/bench_voronoi_partition_methods.py-292-        "reorder_target": cost["reorder_target"],
scripts/bench/bench_voronoi_partition_methods.py-293-        "already_reordered": cost["already_reordered"],
scripts/bench/bench_voronoi_partition_methods.py-294-        "coloring_method": cost["coloring_method"],
--
scripts/bench/bench_voronoi_partition_methods.py-391-                row["schedule"] = sc
scripts/bench/bench_voronoi_partition_methods.py-392-                note = ("  [COUNTERFACTUAL: production auto-selects "
scripts/bench/bench_voronoi_partition_methods.py-393-                        "allgather here, no ppermute schedule]"
scripts/bench/bench_voronoi_partition_methods.py-394-                        if sc["production_strategy"] == "allgather" else "")
scripts/bench/bench_voronoi_partition_methods.py-395-                verdict = ("colouring PROVABLY OPTIMAL (Vizing floor)"
scripts/bench/bench_voronoi_partition_methods.py-396-                           if sc["coloring_optimal_proven"] else
scripts/bench/bench_voronoi_partition_methods.py:397:                           f"recolour headroom "
scripts/bench/bench_voronoi_partition_methods.py-398-                           f"{sc['coloring_headroom_rounds_min']}-"
scripts/bench/bench_voronoi_partition_methods.py-399-                           f"{sc['coloring_headroom_rounds_max']} round(s)")
scripts/bench/bench_voronoi_partition_methods.py-400-                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
scripts/bench/bench_voronoi_partition_methods.py-401-                      f"rounds={sc['n_rounds']:3d} "
scripts/bench/bench_voronoi_partition_methods.py-402-                      f"max_degree={sc['max_degree']:3d} "
scripts/bench/bench_voronoi_partition_methods.py-403-                      f"gap={sc['coloring_gap']:+d} "
--
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-5-#SBATCH --ntasks=1
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-6-#SBATCH --cpus-per-task=8
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-7-#SBATCH --mem=200G
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-8-#SBATCH --time=24:00:00
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-9-#SBATCH --output=mpas_schedule_cost_scan.%j.log
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-10-# ===========================================================================
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:11:# Does RECOLOURING still have room on the MPAS halo schedule, or is only a
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-12-# new OWNERSHIP objective left?  CPU-only, ZERO GPU hours.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-13-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-14-# WHY THIS RUN EXISTS
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-15-# MPAS GPU is the worst-scaling lane we have: measured/modelled-bound 3.16x
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-16-# (s8@16) to 4.47x (s9@64), and one halo fill costs 12-14 SEQUENTIAL ppermute
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-17-# rounds.  Both independent reviews (codex + GLM, 2026-08-07) ranked cutting
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-18-# that round count as the top structural lever, at 300-800 LOC and 7-14 days
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-19-# for a partitioner with a new objective.  That estimate is only worth
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:20:# spending if recolouring is genuinely exhausted, and the colourer's own
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-21-# lower bound decides it.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-22-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-23-# The schedule is a proper EDGE colouring of the device communication graph
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-24-# (one colour = one ppermute round) and max_degree is that graph's maximum
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-25-# vertex degree, so VIZING bounds the chromatic index: Delta <= chi' <=
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-26-# Delta + 1.  Read coloring_gap = n_rounds - max_degree through that:
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:27:#     gap == 0 -> PROVABLY OPTIMAL; recolouring headroom is exactly zero.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-28-#     gap == 1 -> indistinguishable from optimal (a Class 2 graph really
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-29-#                 needs Delta+1, and deciding Class 1/2 is NP-complete);
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-30-#                 nothing provable to win.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:31:#     gap >= 2 -> at least gap-1 rounds of genuine recolouring headroom.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:32:# So recolouring is capped at ~1 round out of 12-14 (<= ~7%) wherever the
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-33-# multi-start search lands on Delta or Delta+1 — which it does on every
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-34-# configuration probed so far.  This run measures whether the production
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-35-# working points are in that regime.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-36-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-37-# (a) NUMBER PRODUCED: n_rounds, max_degree and their gap per
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-38-#     (method x n_dev) at the production working points.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-39-# (b) CONFIRMS a cheap fix: gap >= 2 at any production (ppermute) row
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:40:#     (recolouring then removes at least gap-1 rounds, guaranteed).
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:41:#     gap == 0: recolouring is provably worthless AT THIS OWNERSHIP.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-42-#     gap == 1: INCONCLUSIVE — Class 1 vs Class 2 is NP-complete, so this
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-43-#     neither confirms nor refutes a one-round win.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:44:#     Note what a gap of 0 does NOT prove: Delta bounds only a proper
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:45:#     UNDIRECTED edge colouring of this graph, and the builder forces BOTH
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-46-#     ppermute directions per pair even when one send map is empty, so a
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:47:#     redesigned DIRECTED schedule is not bounded by Delta.  "Ownership is
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:48:#     the only path" would overclaim; the honest statement is "recolouring
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-49-#     this undirected graph is exhausted".
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-50-# (c) WHY NOT CHEAPER: this IS the cheap test — no GPU, no MPI, no model
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-51-#     step.  It cannot be shrunk further onto small meshes: measured
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:52:#     2026-08-07, L2/L4 x {geometric,sfc} x nd 2-16 all report gap == 0,
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-53-#     but EVERY one of those rows auto-selects the ALLGATHER strategy
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-54-#     (cells/device below the threshold), so production runs no ppermute
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-55-#     schedule there and the number is counterfactual.  Only meshes big
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-56-#     enough to keep cells/device above the threshold answer the question.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-57-#     s6 lloyd=0 np8/np16 (2561 cells/device after padding, above the 2000
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:58:#     ppermute threshold, so genuinely ppermute) also gave gap == 0 on all
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-59-#     three methods — rounds 7/7/6 at np8 and 13/10/10 at np16 for
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-60-#     geometric/sfc/metis.  np8's 7 == n_dev-1 is complete-graph
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-61-#     saturation and says nothing; np16 (13/10/10 < 15) does escape it.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-62-#     Still a SMALL working point; s8-s10 are the production ones.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-63-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-64-# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's

exec
/usr/bin/bash -lc "bash -u -c 'f() { false; local rc="'$?; printf "false rc=%s'"\\n\" \""'$rc"; }; f'"'
bash -u -c 'f() { printf \"argc=%d %s|%s\\n\" \""'$#" "$1" "$2"; }; g() { f ${4:+--expect-rounds "$4"}; }; g a b c "sfc:64=12,sfc:128=14"; g a b c ""'"'
.venv/bin/python -B - <<'PY'
import importlib.util
from pathlib import Path
p=Path('scripts/bench/bench_voronoi_partition_methods.py')
s=importlib.util.spec_from_file_location('bench_review', p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(m.parse_expect_rounds('geometric:2=9,geometric:2=1'))
print(m.parse_expect_rounds(',,,'))
print(m.check_expected_rounds([], {}))
PY" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 1 in 265ms:
false rc=1
argc=2 --expect-rounds|sfc:64=12,sfc:128=14
environment: $1: unbound variable
/usr/bin/bash: line 2: cannot create temp file for here-document: Read-only file system

codex
Not ready. These defects remain:

- **BLOCKER:** Arm 3 failure is reported as job success. After `run_arm 10` fails, the script assigns `RC10`, prints `SCAN_DONE`, then exits with that successful `echo` status—zero. Slurm will mark the scan successful despite no valid s10 result. [mpas_schedule_cost_scan.sbatch:141](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:141) [mpas_schedule_cost_scan.sbatch:144](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:144)

- **HIGH:** `--expect-rounds` is still bypassable. Duplicate keys overwrite silently, so `geometric:2=999,geometric:2=<actual>` passes while discarding the bad listed expectation. [bench_voronoi_partition_methods.py:198](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:198) [bench_voronoi_partition_methods.py:199](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:199)  
  Also, `--expect-rounds ",,,"` becomes `{}` because empty items are skipped, and `main()` silently omits the check. The helper itself returns success for `{}`. [bench_voronoi_partition_methods.py:192](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:192) [bench_voronoi_partition_methods.py:347](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:347)  
  This does not compromise the current sbatch literals—they are nonempty and unique—but it disproves the claimed non-vacuous general gate.

- **MEDIUM:** The 2561 correction is still wrong for half of the stated cases. L6 has 40,962 cells: after padding, np8 has 40,968 / 8 = **5,121** cells/device; only np16 has 40,976 / 16 = **2,561**. The comment claims 2,561 for “np8/np16.” [mpas_schedule_cost_scan.sbatch:57](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:57)

- **MEDIUM:** `--lloyd` accepts negative values. The builder treats `-1` identically to zero relaxation but it is recorded/cached as `-1`, creating false provenance and a distinct cache key. [bench_voronoi_partition_methods.py:321](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:321) The builder relaxes only when iterations are `> 0`. [voronoi.py:1366](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/voronoi.py:1366)

- The sbatch still opens with the false ownership-only dichotomy, despite correctly disclaiming it later. A directed schedule is a third path, not an ownership objective. [mpas_schedule_cost_scan.sbatch:11](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:11)

The claimed fixes that are real:

- `local rc=$?` captures the preceding Python command correctly: expansion occurs before `local` runs. The current fourth arguments are always supplied, so `${4:+--expect-rounds "$4"}` is correctly quoted and emits either two arguments or none. Omitting `$4` in a future call would abort under `set -u`; current calls do not. [mpas_schedule_cost_scan.sbatch:106](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:106) [mpas_schedule_cost_scan.sbatch:112](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:112)

- A real validation-arm nonzero cannot reach arm 3: both statuses are captured and the branch exits before it. [mpas_schedule_cost_scan.sbatch:129](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:129) [mpas_schedule_cost_scan.sbatch:132](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:132)

- The hardcoded S8/S9 census exactly matches the scorer’s reference census. [mpas_schedule_cost_scan.sbatch:126](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:126) [sharded_dynamics.py:1323](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1323)

- Nonempty expectations do fail for unavailable, omitted, or schedule-less rows. [bench_voronoi_partition_methods.py:218](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:218)

- The headroom interval is now correct everywhere. `(14, 10)` gives gap 4, hence guaranteed reduction 3 and possible reduction 4. [bench_voronoi_partition_methods.py:252](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:252) [bench_voronoi_partition_methods.py:286](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:286) [test_bench_voronoi_partition_methods.py:178](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:178)

- The directed/undirected scope is correctly stated in the bench, and matches the builder: it builds an undirected pair for either dependency, then emits both directions. [bench_voronoi_partition_methods.py:41](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:41) [sharded_dynamics.py:1815](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1815) [sharded_dynamics.py:1926](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1926)

- The Lloyd spy is real: `main()` imports the builder after the test patches its module. [test_bench_voronoi_partition_methods.py:304](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:304) [bench_voronoi_partition_methods.py:354](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:354)

- Per-file provenance is sufficient: every row comes from the one mesh built before the loop, while level and Lloyd count live in enclosing metadata. Flattened standalone rows would still lack mesh identity, but that is not a defect in this one-mesh-per-JSON format. [bench_voronoi_partition_methods.py:357](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:357) [bench_voronoi_partition_methods.py:424](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:424)

Writing the JSON then returning 1 is the right validation-failure behavior: it preserves the failed receipt with `passed: false`, and the caller gets a nonzero status. [bench_voronoi_partition_methods.py:437](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:437)

Test gap: the “missing method” test only omits `sfc` from the sweep; it does not exercise an unavailable `metis` row. [test_bench_voronoi_partition_methods.py:321](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:321) [test_bench_voronoi_partition_methods.py:344](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:344) It also misses duplicate and all-empty expectation specs, and there is no test for arm-3 exit propagation.
tokens used
106,996
Not ready. These defects remain:

- **BLOCKER:** Arm 3 failure is reported as job success. After `run_arm 10` fails, the script assigns `RC10`, prints `SCAN_DONE`, then exits with that successful `echo` status—zero. Slurm will mark the scan successful despite no valid s10 result. [mpas_schedule_cost_scan.sbatch:141](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:141) [mpas_schedule_cost_scan.sbatch:144](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:144)

- **HIGH:** `--expect-rounds` is still bypassable. Duplicate keys overwrite silently, so `geometric:2=999,geometric:2=<actual>` passes while discarding the bad listed expectation. [bench_voronoi_partition_methods.py:198](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:198) [bench_voronoi_partition_methods.py:199](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:199)  
  Also, `--expect-rounds ",,,"` becomes `{}` because empty items are skipped, and `main()` silently omits the check. The helper itself returns success for `{}`. [bench_voronoi_partition_methods.py:192](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:192) [bench_voronoi_partition_methods.py:347](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:347)  
  This does not compromise the current sbatch literals—they are nonempty and unique—but it disproves the claimed non-vacuous general gate.

- **MEDIUM:** The 2561 correction is still wrong for half of the stated cases. L6 has 40,962 cells: after padding, np8 has 40,968 / 8 = **5,121** cells/device; only np16 has 40,976 / 16 = **2,561**. The comment claims 2,561 for “np8/np16.” [mpas_schedule_cost_scan.sbatch:57](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:57)

- **MEDIUM:** `--lloyd` accepts negative values. The builder treats `-1` identically to zero relaxation but it is recorded/cached as `-1`, creating false provenance and a distinct cache key. [bench_voronoi_partition_methods.py:321](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:321) The builder relaxes only when iterations are `> 0`. [voronoi.py:1366](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/voronoi.py:1366)

- The sbatch still opens with the false ownership-only dichotomy, despite correctly disclaiming it later. A directed schedule is a third path, not an ownership objective. [mpas_schedule_cost_scan.sbatch:11](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:11)

The claimed fixes that are real:

- `local rc=$?` captures the preceding Python command correctly: expansion occurs before `local` runs. The current fourth arguments are always supplied, so `${4:+--expect-rounds "$4"}` is correctly quoted and emits either two arguments or none. Omitting `$4` in a future call would abort under `set -u`; current calls do not. [mpas_schedule_cost_scan.sbatch:106](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:106) [mpas_schedule_cost_scan.sbatch:112](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:112)

- A real validation-arm nonzero cannot reach arm 3: both statuses are captured and the branch exits before it. [mpas_schedule_cost_scan.sbatch:129](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:129) [mpas_schedule_cost_scan.sbatch:132](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:132)

- The hardcoded S8/S9 census exactly matches the scorer’s reference census. [mpas_schedule_cost_scan.sbatch:126](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:126) [sharded_dynamics.py:1323](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1323)

- Nonempty expectations do fail for unavailable, omitted, or schedule-less rows. [bench_voronoi_partition_methods.py:218](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:218)

- The headroom interval is now correct everywhere. `(14, 10)` gives gap 4, hence guaranteed reduction 3 and possible reduction 4. [bench_voronoi_partition_methods.py:252](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:252) [bench_voronoi_partition_methods.py:286](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:286) [test_bench_voronoi_partition_methods.py:178](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:178)

- The directed/undirected scope is correctly stated in the bench, and matches the builder: it builds an undirected pair for either dependency, then emits both directions. [bench_voronoi_partition_methods.py:41](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:41) [sharded_dynamics.py:1815](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1815) [sharded_dynamics.py:1926](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1926)

- The Lloyd spy is real: `main()` imports the builder after the test patches its module. [test_bench_voronoi_partition_methods.py:304](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:304) [bench_voronoi_partition_methods.py:354](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:354)

- Per-file provenance is sufficient: every row comes from the one mesh built before the loop, while level and Lloyd count live in enclosing metadata. Flattened standalone rows would still lack mesh identity, but that is not a defect in this one-mesh-per-JSON format. [bench_voronoi_partition_methods.py:357](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:357) [bench_voronoi_partition_methods.py:424](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:424)

Writing the JSON then returning 1 is the right validation-failure behavior: it preserves the failed receipt with `passed: false`, and the caller gets a nonzero status. [bench_voronoi_partition_methods.py:437](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:437)

Test gap: the “missing method” test only omits `sfc` from the sweep; it does not exercise an unavailable `metis` row. [test_bench_voronoi_partition_methods.py:321](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:321) [test_bench_voronoi_partition_methods.py:344](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:344) It also misses duplicate and all-empty expectation specs, and there is no test for arm-3 exit propagation.
