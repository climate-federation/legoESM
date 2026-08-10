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
session id: 019fdbfc-9237-7841-8206-d01e686610d4
--------
user
ADVERSARIAL REVIEW, ROUND 6 (final). Verify the round-5 fixes; hunt for new defects. Refute, do not summarize. End with SHIP or NOT READY.

Files: `git diff` on scripts/bench/bench_voronoi_partition_methods.py and tests/bench/test_bench_voronoi_partition_methods.py; untracked scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch and tests/bench/test_mpas_schedule_cost_scan_sbatch.py.

ROUND-5 FINDINGS, AND WHAT I DID — including TWO I deliberately did NOT fix. Judge those two on whether my reasoning is sound, not on whether the residual risk is zero:

FIXED:
(a) 'empty-successful benchmark: --methods metis without pymetis exits 0 having measured nothing' -> main() now raises SystemExit when no row is both available and scored. VERIFY, and check it cannot fire spuriously on a legitimate run.
(b) 'the six-row gate regression test is vacuous on cardinality/content' -> new test parses the launcher's own S8_EXPECT/S9_EXPECT literals with the PRODUCTION parse_expect_rounds and asserts the key set is exactly {geometric,sfc,metis} x {64,128}. VERIFY deleting a row or corrupting the literal now fails.
(c) 'a failed rerun leaves a stale s10 JSON that looks current' -> run_arm now rm -f's the arm's JSON BEFORE running, and after a zero exit requires the JSON to contain "n_ranks" or returns 90. New tests cover both (stale file removed on failure; exit-0-without-artifact caught). VERIFY the artifact check cannot pass on a JSON that has the substring but no real rows, and that returning 90 composes correctly with the validation/arm-3 gates.

NOT FIXED, with reasoning — tell me if I am wrong:
(d) 'the shell regression is not executed in required CI (tests/bench is not in the CI test jobs)'. TRUE, but tests/bench has never been in CI for ANY test in it, and this repo's GitHub Actions have been disabled repo-wide since 2026-05-27, so no test here runs in CI today. Wiring tests/bench into CI is a repo-wide change well outside a scaling-measurement change. I intend to REPORT it, not do it. Is that the right call, or does this specific test justify the CI change?
(e) 'the production instrument is still redirectable via LEGOESM_PYTHON / LEGOESM_REPO'. TRUE, but that is the deliberate, pre-existing contract of _env.sh, shared by every launcher in scripts/cluster/scaling_levante/ (see prewarm_s10.sbatch). My change did not introduce it; it removed the seam I had added. Hardening _env.sh's contract is a change to shared infrastructure used by many launchers. My mitigation is the artifact check in (c), which makes 'succeeded without measuring' DETECTABLE rather than impossible, and I say exactly that in the comment. Is that the right scope call?

ALSO: re-check anything previously verified in case a round-5 edit broke it, and look for new vacuity in the 3 tests added since round 4. Note the test that asserts _levels(arms) == ['8','9'] on an artifact failure: I initially wrote ['8'], the test failed, and I concluded the EXPECTATION was wrong because the launcher intentionally runs both validation arms before aborting. Confirm that reading of the control flow.

Cite file:line.
codex
I’ll inspect the exact diff and untracked files, trace the launcher’s control flow against the new tests, and look for a concrete refutation rather than recap.
exec
/usr/bin/bash -lc 'git status --short && git diff -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py && git diff --no-index /dev/null scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch; git diff --no-index /dev/null tests/bench/test_mpas_schedule_cost_scan_sbatch.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 1 in 672ms:
 M scripts/bench/bench_voronoi_partition_methods.py
 M tests/bench/test_bench_voronoi_partition_methods.py
?? .physics-validator/flag_reachability_audit/
?? .physics-validator/homogeneous_ice_nucleation/
?? .physics-validator/scaling_campaign/codex_consult_hundreds_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r2_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r3_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r4_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r5_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md
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
diff --git a/scripts/bench/bench_voronoi_partition_methods.py b/scripts/bench/bench_voronoi_partition_methods.py
index 6aa0f4a40..e7355ed5a 100644
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
@@ -149,12 +370,46 @@ def main() -> int:
             raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
     if not rank_counts or any(n < 2 for n in rank_counts):
         raise SystemExit("--rank-counts needs integers >= 2")
+    if not methods:
+        # Same class as an empty --expect-rounds: the loop would be skipped,
+        # `rows: []` written, and 0 returned — an empty run that reads as a
+        # successful one.
+        raise SystemExit(
+            f"--methods={args.methods!r} selects NO methods; the benchmark "
+            f"would measure nothing and still exit 0. Choose from {METHODS}.")
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
@@ -185,6 +440,33 @@ def main() -> int:
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
+
+    # An "empty but successful" run is the same failure class as a skipped
+    # gate: --methods metis on a box without pymetis appends only an
+    # unavailable row and would otherwise exit 0 having measured nothing
+    # (codex round 5).
+    if not any(r.get("available") and "n_ranks" in r for r in rows):
+        raise SystemExit(
+            f"no method was actually measured (requested {methods}); every "
+            f"one was unavailable, so this run has no results and must not "
+            f"report success.")
 
     payload = {
         "rows": rows,
@@ -204,7 +486,9 @@ def main() -> int:
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
@@ -213,6 +497,24 @@ def main() -> int:
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
index 21f861fd7..66a433f06 100644
--- a/tests/bench/test_bench_voronoi_partition_methods.py
+++ b/tests/bench/test_bench_voronoi_partition_methods.py
@@ -146,3 +146,312 @@ def test_halo_matches_runtime_partition():
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
+
+
+def test_empty_methods_refused_instead_of_measuring_nothing(monkeypatch):
+    """``--methods " , , "`` used to select nothing, write ``rows: []`` and
+    exit 0 — an empty run that reads as a successful one (codex round 4)."""
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", " , , ", "--out", "/dev/null"])
+    with pytest.raises(SystemExit, match="selects NO methods"):
+        mod.main()
diff --git a/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch b/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
new file mode 100644
index 000000000..d89778ba0
--- /dev/null
+++ b/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
@@ -0,0 +1,180 @@
+#!/bin/bash -l
+#SBATCH --job-name=mpas_sched_cost
+#SBATCH --account=bb1596
+#SBATCH --partition=shared
+#SBATCH --ntasks=1
+#SBATCH --cpus-per-task=8
+#SBATCH --mem=200G
+#SBATCH --time=24:00:00
+#SBATCH --output=mpas_schedule_cost_scan.%j.log
+# ===========================================================================
+# Does RECOLOURING still have room on the MPAS halo schedule?  CPU-only,
+# ZERO GPU hours.
+#
+# Note this is NOT a two-way choice between colouring and ownership: a
+# redesigned DIRECTED schedule (see the SCOPE note below) is a third path,
+# and it is not bounded by the quantity measured here.  What this run
+# settles is only whether recolouring THIS undirected graph is exhausted.
+#
+# WHY THIS RUN EXISTS
+# MPAS GPU is the worst-scaling lane we have: measured/modelled-bound 3.16x
+# (s8@16) to 4.47x (s9@64), and one halo fill costs 12-14 SEQUENTIAL ppermute
+# rounds.  Both independent reviews (codex + GLM, 2026-08-07) ranked cutting
+# that round count as the top structural lever, at 300-800 LOC and 7-14 days
+# for a partitioner with a new objective.  That estimate is only worth
+# spending if recolouring is genuinely exhausted, and the colourer's own
+# lower bound decides it.
+#
+# The schedule is a proper EDGE colouring of the device communication graph
+# (one colour = one ppermute round) and max_degree is that graph's maximum
+# vertex degree, so VIZING bounds the chromatic index: Delta <= chi' <=
+# Delta + 1.  Read coloring_gap = n_rounds - max_degree through that:
+#     gap == 0 -> PROVABLY OPTIMAL; recolouring headroom is exactly zero.
+#     gap == 1 -> indistinguishable from optimal (a Class 2 graph really
+#                 needs Delta+1, and deciding Class 1/2 is NP-complete);
+#                 nothing provable to win.
+#     gap >= 2 -> at least gap-1 rounds of genuine recolouring headroom.
+# So recolouring is capped at ~1 round out of 12-14 (<= ~7%) wherever the
+# multi-start search lands on Delta or Delta+1 — which it does on every
+# configuration probed so far.  This run measures whether the production
+# working points are in that regime.
+#
+# (a) NUMBER PRODUCED: n_rounds, max_degree and their gap per
+#     (method x n_dev) at the production working points.
+# (b) CONFIRMS a cheap fix: gap >= 2 at any production (ppermute) row
+#     (recolouring then removes at least gap-1 rounds, guaranteed).
+#     gap == 0: recolouring is provably worthless AT THIS OWNERSHIP.
+#     gap == 1: INCONCLUSIVE — Class 1 vs Class 2 is NP-complete, so this
+#     neither confirms nor refutes a one-round win.
+#     Note what a gap of 0 does NOT prove: Delta bounds only a proper
+#     UNDIRECTED edge colouring of this graph, and the builder forces BOTH
+#     ppermute directions per pair even when one send map is empty, so a
+#     redesigned DIRECTED schedule is not bounded by Delta.  "Ownership is
+#     the only path" would overclaim; the honest statement is "recolouring
+#     this undirected graph is exhausted".
+# (c) WHY NOT CHEAPER: this IS the cheap test — no GPU, no MPI, no model
+#     step.  It cannot be shrunk further onto small meshes: measured
+#     2026-08-07, L2/L4 x {geometric,sfc} x nd 2-16 all report gap == 0,
+#     but EVERY one of those rows auto-selects the ALLGATHER strategy
+#     (cells/device below the threshold), so production runs no ppermute
+#     schedule there and the number is counterfactual.  Only meshes big
+#     enough to keep cells/device above the threshold answer the question.
+#     s6 lloyd=0 also gave gap == 0 on all three methods — rounds 7/7/6 at
+#     np8 and 13/10/10 at np16 for geometric/sfc/metis.  Both are genuinely
+#     ppermute (L6 = 40,962 cells; padded, np8 gives 40,968/8 = 5,121 and
+#     np16 gives 40,976/16 = 2,561 cells/device, each above the 2,000
+#     threshold).  But np8's 7 == n_dev-1 is complete-graph saturation and
+#     says nothing; only np16 (13/10/10 < 15) escapes it.  Still a SMALL
+#     working point; s8-s10 are the production ones.
+#
+# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
+# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
+#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
+#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
+# Arms 1-2 must REPRODUCE those before arm 3's unknown s10 number is quoted.
+# A scan that misses the known answer is a broken instrument, and its s10
+# row must not be believed.
+#
+# lloyd=0 throughout: it is what the reference census used AND the cache key
+# the prewarmed s8/s9/s10 meshes were written under.  It is the LABELLED
+# synthetic scaling mesh, recorded in every row's metadata so it can never be
+# read back as a production SCVT receipt.
+#
+# Arms run cheapest-first and each writes its own JSON, so a later arm that
+# runs out of time or memory cannot lose an earlier arm's result.  s10@128 is
+# last and is the one genuinely at risk: the scorer is known not to have
+# finished at subdiv-8@128 on a laptop, which is why this asks for 24 h.
+#
+# SUBMIT (from the repo root):
+#   sbatch scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
+# ===========================================================================
+set -uo pipefail
+SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
+export JAX_PLATFORMS=cpu
+export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
+export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
+export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
+source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
+cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
+
+OUT=results/a1/mpas_schedule_cost
+mkdir -p "$OUT"
+# NOT overridable, deliberately.  An earlier revision made this settable
+# from the environment so the exit-status logic could be regression-tested;
+# that reintroduced exactly the failure class this script exists to prevent
+# (a stray exported variable redirects a real scan to something else, every
+# arm exits 0, SLURM files it COMPLETED with no results).  The shell test
+# drives the interpreter instead — see
+# tests/bench/test_mpas_schedule_cost_scan_sbatch.py.
+BENCH=scripts/bench/bench_voronoi_partition_methods.py
+
+# pymetis is present in .venv but was ABSENT from .venv-mpi for a while, and a
+# missing partitioner is reported as "unavailable" rather than substituted.
+# Say which python and whether metis is really there, so a two-method table
+# cannot be misread as a three-method one.
+echo "[scan] python=$PY"
+"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
+  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"
+
+run_arm () {  # $1=level  $2=rank-counts  $3=label  $4=expect-rounds (may be "")
+  local json="$OUT/schedule_cost_s$1.json"
+  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
+  # Delete any previous artifact FIRST. A failed rerun that leaves the
+  # earlier run's JSON in place produces a stale file with a current-looking
+  # mtime story, and a stale receipt read as current is worse than a missing
+  # one (codex round 5).
+  rm -f "$json"
+  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
+  "$PY" "$BENCH" \
+      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
+      --methods geometric,sfc,metis --schedule-cost \
+      ${4:+--expect-rounds "$4"} \
+      --out "$json"
+  local rc=$?   # capture BEFORE anything else runs, or `date` overwrites it
+  echo "[scan] arm $3 exit=$rc"
+  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
+  # An exit code alone is not evidence the arm produced anything (CLAUDE.md:
+  # "tool status is not evidence"). Require the artifact to exist and carry
+  # rows, so a $PY that silently does nothing cannot yield SCAN_DONE with no
+  # results. This does not make the launcher unspoofable — $PY and $REPO come
+  # from _env.sh and remain configurable, as every launcher here is — but it
+  # does make "succeeded without measuring" DETECTABLE.
+  if [ "$rc" -eq 0 ] && ! grep -q '"n_ranks"' "$json" 2>/dev/null; then
+    echo "[scan] arm $3 exited 0 but $json is missing or has no scored rows"
+    return 90
+  fi
+  return $rc
+}
+
+# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
+# makes the bench exit non-zero unless it reproduces the census below, and a
+# listed pair that never got scored (e.g. pymetis missing) counts as a
+# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
+# pass — an instrument that misses the known answer cannot be trusted on the
+# unknown one.
+S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
+S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"
+
+run_arm 8 64,128 "1/3 VALIDATION s8" "$S8_EXPECT"; RC8=$?
+run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?
+
+if [ "$RC8" -ne 0 ] || [ "$RC9" -ne 0 ]; then
+  echo "[scan] INSTRUMENT CHECK FAILED (s8 rc=$RC8, s9 rc=$RC9)."
+  echo "[scan] The scorer does not reproduce the reference census, so an s10"
+  echo "[scan] number from it would be untrustworthy. NOT running arm 3."
+  echo "SCAN_ABORTED_VALIDATION"
+  exit 1
+fi
+
+# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
+run_arm 10 128 "3/3 UNKNOWN s10" ""; RC10=$?
+echo "[scan] arm 3 rc=$RC10"
+
+if [ "$RC10" -ne 0 ]; then
+  # Do NOT let SLURM record this as COMPLETED: there is no valid s10 row,
+  # and an exit-0 job with a missing result reads as a successful scan.
+  echo "SCAN_FAILED_ARM3"
+  exit "$RC10"
+fi
+
+echo "SCAN_DONE"
diff --git a/tests/bench/test_mpas_schedule_cost_scan_sbatch.py b/tests/bench/test_mpas_schedule_cost_scan_sbatch.py
new file mode 100644
index 000000000..eb87ede5f
--- /dev/null
+++ b/tests/bench/test_mpas_schedule_cost_scan_sbatch.py
@@ -0,0 +1,227 @@
+"""Exit-status regression for the MPAS schedule-cost scan launcher.
+
+The failure this guards is not hypothetical: the first version of the script
+captured arm 3's status into ``RC10`` and then ended on a successful ``echo``,
+so a scan that produced NO valid s10 result exited 0 and SLURM filed it as
+COMPLETED (codex round 2, BLOCKER).  "Tool status is not evidence" cuts both
+ways — a launcher that cannot report failure makes every downstream reading
+of ``sacct`` a lie.
+
+HOW THIS IS DRIVEN, and why not the obvious way: the bench path in the
+launcher is deliberately NOT overridable from the environment.  Making it
+overridable (the first attempt) handed a stray exported variable the power to
+redirect a real scan to something that exits 0 — reintroducing the very
+failure class under test (codex round 4).  Instead this substitutes the
+INTERPRETER via ``LEGOESM_PYTHON``, which is an existing production knob that
+``_env.sh`` already reads, so the launcher itself carries no test-only seam.
+
+The stub interpreter answers ``_env.sh``'s jax probe, then exits with a
+scripted status per arm, recording each arm's full argv so a test can assert
+both WHICH arms ran and that each carried its required flags.
+"""
+from __future__ import annotations
+
+import os
+import shutil
+import subprocess
+import sys
+from pathlib import Path
+
+import pytest
+
+_REPO = Path(__file__).resolve().parents[2]
+_SBATCH = (_REPO / "scripts" / "cluster" / "scaling_levante"
+           / "mpas_schedule_cost_scan.sbatch")
+
+pytestmark = pytest.mark.skipif(
+    shutil.which("bash") is None, reason="needs bash to run the launcher")
+
+
+def _run(tmp_path, codes, write_artifact=True):
+    """Run the launcher with a stub interpreter that exits ``codes`` per arm.
+
+    ``codes`` is one exit status per bench invocation, in order (arm 1, arm
+    2, arm 3).  When ``write_artifact`` the stub also writes a minimal JSON
+    to the arm's ``--out`` containing an ``n_ranks`` key, which is what the
+    launcher's post-arm artifact check looks for; setting it False simulates
+    an interpreter that exits 0 having measured nothing.
+
+    Returns ``(proc, arms)`` where ``arms`` is the recorded argv of each
+    bench call.
+    """
+    log = tmp_path / "calls.txt"
+    stub = tmp_path / "stub_python"
+    stub.write_text(
+        "#!/usr/bin/env python3\n"
+        "import sys, json\n"
+        f"codes = {list(codes)!r}\n"
+        f"log = {str(log)!r}\n"
+        f"write_artifact = {bool(write_artifact)!r}\n"
+        "argv = sys.argv[1:]\n"
+        # _env.sh probes the interpreter with `-c 'import jax...'`; answer it
+        # without counting it as a bench call.
+        "if argv and argv[0] == '-c':\n"
+        "    sys.exit(0)\n"
+        "with open(log, 'a') as f:\n"
+        "    f.write(json.dumps(argv) + '\\n')\n"
+        "n = sum(1 for _ in open(log))\n"
+        "rc = codes[n - 1] if n <= len(codes) else 0\n"
+        "if write_artifact and rc == 0:\n"
+        "    out = argv[argv.index('--out') + 1]\n"
+        "    with open(out, 'w') as f:\n"
+        "        json.dump({'rows': [{'n_ranks': 64}]}, f)\n"
+        "sys.exit(rc)\n"
+    )
+    stub.chmod(0o755)
+
+    env = dict(os.environ)
+    env["SLURM_SUBMIT_DIR"] = str(_REPO)
+    env["LEGOESM_REPO"] = str(_REPO)
+    env["LEGOESM_PYTHON"] = str(stub)
+    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
+    proc = subprocess.run(
+        ["bash", str(_SBATCH)], env=env, cwd=str(_REPO),
+        capture_output=True, text=True, timeout=600)
+
+    arms = []
+    if log.exists():
+        import json
+        arms = [json.loads(line) for line in log.read_text().splitlines()
+                if line.strip()]
+    return proc, arms
+
+
+def _levels(arms):
+    return [a[a.index("--subdivision") + 1] for a in arms]
+
+
+def test_launcher_is_not_redirectable_from_the_environment():
+    """The bench path must be hardcoded.
+
+    If it were env-overridable, a stray exported variable could point every
+    arm at something that exits 0 and the job would report SCAN_DONE with no
+    results — the exact failure this module guards.
+    """
+    text = _SBATCH.read_text()
+    assert "BENCH=scripts/bench/bench_voronoi_partition_methods.py" in text
+    assert "LEGOESM_SCHEDULE_COST_BENCH" not in text
+
+
+@pytest.mark.parametrize("codes, failing_arm", [([1, 0, 0], 1), ([0, 1, 0], 2)])
+def test_validation_failure_aborts_before_the_unknown_arm(
+        tmp_path, codes, failing_arm):
+    """EITHER validation arm failing must abort non-zero and skip arm 3.
+
+    An instrument that misses the known census cannot be trusted on the
+    unknown one, so producing an s10 number anyway is worse than none.
+    Both arms are exercised: guarding only arm 1 leaves arm 2 unchecked.
+    """
+    proc, arms = _run(tmp_path, codes)
+    assert proc.returncode != 0, proc.stdout[-2000:]
+    assert "SCAN_ABORTED_VALIDATION" in proc.stdout
+    assert "SCAN_DONE" not in proc.stdout
+    assert _levels(arms) == ["8", "9"], (
+        f"arm 3 must not run after validation arm {failing_arm} failed: "
+        f"{_levels(arms)}")
+
+
+def test_arm3_failure_is_not_reported_as_success(tmp_path):
+    """The original BLOCKER: arm 3 fails, the job must NOT exit 0, and must
+    propagate the exact status."""
+    proc, arms = _run(tmp_path, [0, 0, 3])
+    assert proc.returncode == 3, (
+        f"arm-3 status not propagated (got {proc.returncode})\n"
+        f"{proc.stdout[-2000:]}")
+    assert "SCAN_FAILED_ARM3" in proc.stdout
+    assert "SCAN_DONE" not in proc.stdout
+    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
+
+
+def test_full_success_exits_zero_and_runs_every_arm(tmp_path):
+    proc, arms = _run(tmp_path, [0, 0, 0])
+    assert proc.returncode == 0, proc.stdout[-2000:]
+    assert "SCAN_DONE" in proc.stdout
+    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
+
+
+def test_each_arm_carries_the_flags_its_claim_depends_on(tmp_path):
+    """The exit plumbing being right is worthless if an arm silently stops
+    scoring or stops gating.
+
+    Without this, dropping ``--schedule-cost`` (nothing is scored) or
+    ``--expect-rounds`` (the validation arms assert nothing) would leave
+    every other test in this module green.
+    """
+    _, arms = _run(tmp_path, [0, 0, 0])
+    assert len(arms) == 3, arms
+    for argv in arms:
+        assert "--schedule-cost" in argv, argv
+        assert "--lloyd" in argv and argv[argv.index("--lloyd") + 1] == "0"
+    # Validation arms must actually gate; the unknown arm must not pretend to.
+    for argv in arms[:2]:
+        assert "--expect-rounds" in argv, argv
+        assert argv[argv.index("--expect-rounds") + 1].strip(), argv
+    assert "--expect-rounds" not in arms[2], arms[2]
+
+
+def test_arm_that_exits_zero_without_producing_results_is_caught(tmp_path):
+    """An exit code is not evidence an arm measured anything.
+
+    If the interpreter succeeds but writes no JSON (or one with no scored
+    rows), the launcher must NOT report SCAN_DONE — otherwise a misconfigured
+    run files as COMPLETED with nothing in it, the exact class this module
+    exists to prevent.
+    """
+    proc, arms = _run(tmp_path, [0, 0, 0], write_artifact=False)
+    assert proc.returncode != 0, proc.stdout[-2000:]
+    assert "has no scored rows" in proc.stdout
+    assert "SCAN_DONE" not in proc.stdout
+    # Both validation arms still RUN (they are independent measurements and
+    # running both reports more before aborting); what matters is that the
+    # unknown arm 3 is skipped.
+    assert _levels(arms) == ["8", "9"], _levels(arms)
+
+
+def test_stale_artifact_from_a_previous_run_is_removed_before_each_arm(
+        tmp_path):
+    """A failed rerun must not leave the previous run's JSON in place: a
+    stale receipt read as current is worse than a missing one."""
+    out_dir = _REPO / "results" / "a1" / "mpas_schedule_cost"
+    out_dir.mkdir(parents=True, exist_ok=True)
+    stale = out_dir / "schedule_cost_s8.json"
+    stale.write_text('{"rows": [{"n_ranks": 999, "STALE": true}]}')
+    try:
+        # Arm 1 fails and writes nothing; the stale file must be gone, not
+        # left behind looking like this run's result.
+        proc, _ = _run(tmp_path, [1, 0, 0], write_artifact=False)
+        assert proc.returncode != 0
+        assert not stale.exists(), (
+            "stale s8 JSON survived a failed arm and would read as current")
+    finally:
+        stale.unlink(missing_ok=True)
+
+
+def test_launcher_expectations_are_the_full_six_row_census():
+    """The gate's strength is its CARDINALITY and content, not its presence.
+
+    Asserting only that --expect-rounds is non-empty (the previous test)
+    would still pass if five of the six rows were deleted, because the stub
+    never runs the real parser.  Parse the launcher's own literals with the
+    production parser and check every method x rank pair is present.
+    """
+    import importlib.util
+
+    spec = importlib.util.spec_from_file_location(
+        "bench_vor_part_gate",
+        _REPO / "scripts" / "bench" / "bench_voronoi_partition_methods.py")
+    bench = importlib.util.module_from_spec(spec)
+    spec.loader.exec_module(bench)
+
+    text = _SBATCH.read_text()
+    for var in ("S8_EXPECT", "S9_EXPECT"):
+        line = next(ln for ln in text.splitlines()
+                    if ln.startswith(f"{var}="))
+        parsed = bench.parse_expect_rounds(line.split("=", 1)[1].strip('"'))
+        assert set(parsed) == {
+            (m, n) for m in ("geometric", "sfc", "metis") for n in (64, 128)
+        }, f"{var} is not the full six-row census: {sorted(parsed)}"

exec
/usr/bin/bash -lc 'git diff --unified=80 -- scripts/bench/bench_voronoi_partition_methods.py && git diff --unified=80 -- tests/bench/test_bench_voronoi_partition_methods.py && nl -ba scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && nl -ba tests/bench/test_mpas_schedule_cost_scan_sbatch.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 241ms:
diff --git a/scripts/bench/bench_voronoi_partition_methods.py b/scripts/bench/bench_voronoi_partition_methods.py
index 6aa0f4a40..e7355ed5a 100644
--- a/scripts/bench/bench_voronoi_partition_methods.py
+++ b/scripts/bench/bench_voronoi_partition_methods.py
@@ -1,220 +1,522 @@
 """Benchmark MPAS/Voronoi partition methods: RCB vs Hilbert-SFC vs METIS.
 
 Scaling-audit item 8.  Two independent measurement layers:
 
 1. **Offline partition quality** (no MPI, exact, every rank enumerated
    serially): edge cut, halo cells (max/mean, halo/owned ratio), load
    balance (cells/rank min/max, imbalance max/mean), neighbor-rank fan-out —
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
    (atmosphere SPMD) — one method per launch, same case otherwise
    (controlled comparison).
 
 Guards: methods that are unavailable (``metis`` without ``pymetis``) are
 reported as ``"unavailable"`` — never silently substituted, so a table
 column can never claim METIS numbers that actually came from the RCB
 fallback.  Partition CORRECTNESS is asserted per row (every cell owned by
 exactly one rank; owner range valid) before any metric is recorded.
 
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
 
 import argparse
 import json
 import os
 import sys
 from pathlib import Path
 
 import numpy as np
 
 sys.path.insert(0, str(Path(__file__).resolve().parent))
 
 from metadata import annotate_incomplete, scaling_metadata  # noqa: E402
 
 METHODS = ("geometric", "sfc", "metis")
 
 
 def method_available(method: str) -> bool:
     if method != "metis":
         return True
     try:
         import pymetis  # noqa: F401
 
         return True
     except Exception:
         return False
 
 
 def partition_quality(mesh, cell_owner: np.ndarray, n_ranks: int,
                       halo_depth: int = 2) -> dict:
     """Exact partition-quality metrics from a global owner array.
 
     Serial enumeration of every rank (no MPI): the same halo construction
     the runtime uses (``compute_halo_cells``), so the reported halo sizes
     are the runtime's, not an estimate.
     """
     from legoesm.parallel.voronoi_partition import compute_halo_cells
 
     n_cells = int(mesh.nCells)
     # Correctness: every cell owned exactly once, owners in range, no
     # empty rank (an empty rank would silently deflate the halo/owned
     # ratio through the max(counts, 1) guard).
     if n_cells == 0 or cell_owner.size == 0:
         raise AssertionError("empty mesh / owner array")
     if cell_owner.shape != (n_cells,):
         raise AssertionError(f"owner shape {cell_owner.shape} != ({n_cells},)")
     if cell_owner.min() < 0 or cell_owner.max() >= n_ranks:
         raise AssertionError("owner out of range")
     counts = np.bincount(cell_owner, minlength=n_ranks).astype(float)
     if int(counts.sum()) != n_cells:
         raise AssertionError("ownership does not cover the mesh")
     if counts.min() <= 0:
         raise AssertionError(
             f"empty rank in partition (counts.min()={counts.min():.0f}) — "
             f"a skipped rank corrupts every per-rank metric")
 
     # Edge cut: edges whose two cells have different owners.
     c1, c2 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
     valid = (c1 >= 0) & (c2 >= 0)
     edge_cut = int((cell_owner[c1[valid]] != cell_owner[c2[valid]]).sum())
 
     halo_sizes = []
     neighbor_counts = []
     cells_on_cell = np.asarray(mesh.cellsOnCell)
     max_edges = int(cells_on_cell.shape[0]) if cells_on_cell.ndim == 2 else 0
     for r in range(n_ranks):
         halo = compute_halo_cells(
             cell_owner, mesh.cellsOnCell, mesh.maxEdges, r, halo_depth)
         halo_sizes.append(len(halo))
         neighbor_counts.append(
             len(set(int(cell_owner[c]) for c in halo) - {r}))
     _ = max_edges
     halo_sizes = np.array(halo_sizes, dtype=float)
     return {
         "cells_per_rank_min": int(counts.min()),
         "cells_per_rank_max": int(counts.max()),
         "load_imbalance_max_over_mean": float(counts.max() / counts.mean()),
         "edge_cut": edge_cut,
         "edge_cut_fraction": float(edge_cut / max(int(valid.sum()), 1)),
         "halo_cells_max": int(halo_sizes.max()),
         "halo_cells_mean": float(halo_sizes.mean()),
         "halo_owned_ratio_max": float(
             (halo_sizes / np.maximum(counts, 1.0)).max()),
         "neighbor_ranks_max": int(max(neighbor_counts)),
     }
 
 
 def owner_for(mesh, method: str, n_ranks: int) -> np.ndarray:
     from legoesm.parallel.voronoi_partition import (
         partition_cells_geometric,
         partition_cells_metis,
         partition_cells_sfc,
     )
 
     if method == "geometric":
         return np.asarray(partition_cells_geometric(mesh, n_ranks))
     if method == "sfc":
         return np.asarray(partition_cells_sfc(mesh, n_ranks))
     if method == "metis":
         return np.asarray(partition_cells_metis(mesh, n_ranks))
     raise ValueError(f"unknown partition method {method!r}; "
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
         formatter_class=argparse.RawDescriptionHelpFormatter)
     p.add_argument("--subdivision", type=int, default=5,
                    help="Icosahedral level (L5=10,242 cells; L6=40,962).")
     p.add_argument("--rank-counts", type=str, default="2,4,8,16")
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
 
     rank_counts = [int(x) for x in args.rank_counts.split(",") if x]
     methods = [m.strip() for m in args.methods.split(",") if m.strip()]
     for m in methods:
         if m not in METHODS:
             raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
     if not rank_counts or any(n < 2 for n in rank_counts):
         raise SystemExit("--rank-counts needs integers >= 2")
+    if not methods:
+        # Same class as an empty --expect-rounds: the loop would be skipped,
+        # `rows: []` written, and 0 returned — an empty run that reads as a
+        # successful one.
+        raise SystemExit(
+            f"--methods={args.methods!r} selects NO methods; the benchmark "
+            f"would measure nothing and still exit 0. Choose from {METHODS}.")
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
             f"{int(mesh.nCells)} cells (empty ranks are meaningless).")
     print(f"mesh L{args.subdivision}: {int(mesh.nCells)} cells, "
           f"{int(mesh.nEdges)} edges; auto -> "
           f"{resolve_partition_method('auto')!r}")
 
     rows = []
     for method in methods:
         if not method_available(method):
             print(f"  {method:9s}: UNAVAILABLE (pymetis not importable) — "
                   f"column omitted, never substituted")
             rows.append({"method": method, "available": False})
             continue
         for n_ranks in rank_counts:
             q = partition_quality(
                 mesh, owner_for(mesh, method, n_ranks), n_ranks,
                 halo_depth=args.halo_depth)
             row = {"method": method, "available": True,
                    "n_ranks": n_ranks, **q}
             rows.append(row)
             print(f"  {method:9s} np={n_ranks:3d} | "
                   f"imbalance={q['load_imbalance_max_over_mean']:.3f} | "
                   f"edge_cut={q['edge_cut']:6d} "
                   f"({100 * q['edge_cut_fraction']:.2f}%) | "
                   f"halo max={q['halo_cells_max']:5d} "
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
+
+    # An "empty but successful" run is the same failure class as a skipped
+    # gate: --methods metis on a box without pymetis appends only an
+    # unavailable row and would otherwise exit 0 having measured nothing
+    # (codex round 5).
+    if not any(r.get("available") and "n_ranks" in r for r in rows):
+        raise SystemExit(
+            f"no method was actually measured (requested {methods}); every "
+            f"one was unavailable, so this run has no results and must not "
+            f"report success.")
 
     payload = {
         "rows": rows,
         "auto_resolves_to": resolve_partition_method("auto"),
         "step_time_pointer": (
             "step-time per method: bench_ocean_mpas_scaling.py / "
             "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
             "per launch, same case otherwise)"),
         "metadata": annotate_incomplete(scaling_metadata(
             grid="voronoi",
             component="partitioning",
             resolution=f"L{args.subdivision}",
             n_levels=0,
             precision="n/a",
             decomposition="cell_partition",
             solver_variant="n/a",
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
     if outdir:
         os.makedirs(outdir, exist_ok=True)
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
 
 
 if __name__ == "__main__":
     raise SystemExit(main())
diff --git a/tests/bench/test_bench_voronoi_partition_methods.py b/tests/bench/test_bench_voronoi_partition_methods.py
index 21f861fd7..66a433f06 100644
--- a/tests/bench/test_bench_voronoi_partition_methods.py
+++ b/tests/bench/test_bench_voronoi_partition_methods.py
@@ -69,80 +69,389 @@ def test_main_writes_quality_table(tmp_path, monkeypatch):
         "--methods", "geometric,sfc", "--out", str(out)])
     assert mod.main() == 0
     payload = json.loads(out.read_text())
     rows = [r for r in payload["rows"] if r.get("available")]
     assert {(r["method"], r["n_ranks"]) for r in rows} == {
         ("geometric", 2), ("geometric", 4), ("sfc", 2), ("sfc", 4)}
     assert payload["auto_resolves_to"] in ("metis", "geometric")
     md = payload["metadata"]
     assert md["transport"] == "none"
     assert "_incomplete" not in md
 
 
 def test_main_rejects_bad_args(monkeypatch):
     monkeypatch.setattr(sys, "argv", ["bench", "--methods", "voodoo"])
     with pytest.raises(SystemExit):
         mod.main()
     monkeypatch.setattr(sys, "argv", ["bench", "--rank-counts", "1"])
     with pytest.raises(SystemExit):
         mod.main()
 
 
 class _SyntheticMesh:
     """4-cell ring: cells 0-1-2-3 cyclic (each cell has 2 neighbors).
 
     Edges: (0,1) (1,2) (2,3) (3,0) + one INVALID edge (-1,-1) to lock the
     valid-edge masking in the edge-cut denominator.
     """
     nCells = 4
     nEdges = 5
     maxEdges = 2
     cellsOnEdge = np.array([[0, 1, 2, 3, -1],
                             [1, 2, 3, 0, -1]])
     cellsOnCell = np.array([[1, 2, 3, 0],    # neighbor k=0
                             [3, 0, 1, 2]])   # neighbor k=1
 
 
 def test_metric_definitions_locked_on_synthetic_mesh():
     """Exact edge cut / halo / neighbor values on a hand-built ring —
     a denominator or halo-construction drift fails HERE, not in a range
     check (codex finding 3)."""
     mesh = _SyntheticMesh()
     owner = np.array([0, 0, 1, 1])  # cells 0,1 -> rank0; 2,3 -> rank1
     q = mod.partition_quality(mesh, owner, 2, halo_depth=1)
     # Cut edges: (1,2) and (3,0) -> 2 of 4 VALID edges (invalid edge
     # excluded from the denominator).
     assert q["edge_cut"] == 2
     assert q["edge_cut_fraction"] == pytest.approx(0.5)
     # halo_depth=1: each rank's halo = the 2 cells of the other rank that
     # touch it (ring: both of them).
     assert q["halo_cells_max"] == 2
     assert q["halo_cells_mean"] == pytest.approx(2.0)
     assert q["halo_owned_ratio_max"] == pytest.approx(1.0)
     assert q["neighbor_ranks_max"] == 1
     assert q["cells_per_rank_min"] == q["cells_per_rank_max"] == 2
     assert q["load_imbalance_max_over_mean"] == pytest.approx(1.0)
 
 
 def test_empty_rank_rejected():
     mesh = _SyntheticMesh()
     owner = np.array([0, 0, 0, 0])  # rank 1 skipped
     with pytest.raises(AssertionError, match="empty rank"):
         mod.partition_quality(mesh, owner, 2)
 
 
 def test_halo_matches_runtime_partition():
     """Real-mesh lock: the bench's halo size equals the RUNTIME partition's
     (n_local - n_owned) for the same owner array — the bench reports the
     runtime's halos, not an estimate (codex finding 3)."""
     from legoesm.parallel.voronoi_partition import partition_voronoi_mesh
 
     mesh = _mesh()
     owner = mod.owner_for(mesh, "geometric", 4)
     q = mod.partition_quality(mesh, owner, 4, halo_depth=2)
     halos = []
     for r in range(4):
         part = partition_voronoi_mesh(
             mesh, 4, r, method="geometric", halo_depth=2, cell_owner=owner)
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
+
+
+def test_empty_methods_refused_instead_of_measuring_nothing(monkeypatch):
+    """``--methods " , , "`` used to select nothing, write ``rows: []`` and
+    exit 0 — an empty run that reads as a successful one (codex round 4)."""
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", " , , ", "--out", "/dev/null"])
+    with pytest.raises(SystemExit, match="selects NO methods"):
+        mod.main()
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
   102	# NOT overridable, deliberately.  An earlier revision made this settable
   103	# from the environment so the exit-status logic could be regression-tested;
   104	# that reintroduced exactly the failure class this script exists to prevent
   105	# (a stray exported variable redirects a real scan to something else, every
   106	# arm exits 0, SLURM files it COMPLETED with no results).  The shell test
   107	# drives the interpreter instead — see
   108	# tests/bench/test_mpas_schedule_cost_scan_sbatch.py.
   109	BENCH=scripts/bench/bench_voronoi_partition_methods.py
   110	
   111	# pymetis is present in .venv but was ABSENT from .venv-mpi for a while, and a
   112	# missing partitioner is reported as "unavailable" rather than substituted.
   113	# Say which python and whether metis is really there, so a two-method table
   114	# cannot be misread as a three-method one.
   115	echo "[scan] python=$PY"
   116	"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
   117	  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"
   118	
   119	run_arm () {  # $1=level  $2=rank-counts  $3=label  $4=expect-rounds (may be "")
   120	  local json="$OUT/schedule_cost_s$1.json"
   121	  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
   122	  # Delete any previous artifact FIRST. A failed rerun that leaves the
   123	  # earlier run's JSON in place produces a stale file with a current-looking
   124	  # mtime story, and a stale receipt read as current is worse than a missing
   125	  # one (codex round 5).
   126	  rm -f "$json"
   127	  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
   128	  "$PY" "$BENCH" \
   129	      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
   130	      --methods geometric,sfc,metis --schedule-cost \
   131	      ${4:+--expect-rounds "$4"} \
   132	      --out "$json"
   133	  local rc=$?   # capture BEFORE anything else runs, or `date` overwrites it
   134	  echo "[scan] arm $3 exit=$rc"
   135	  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
   136	  # An exit code alone is not evidence the arm produced anything (CLAUDE.md:
   137	  # "tool status is not evidence"). Require the artifact to exist and carry
   138	  # rows, so a $PY that silently does nothing cannot yield SCAN_DONE with no
   139	  # results. This does not make the launcher unspoofable — $PY and $REPO come
   140	  # from _env.sh and remain configurable, as every launcher here is — but it
   141	  # does make "succeeded without measuring" DETECTABLE.
   142	  if [ "$rc" -eq 0 ] && ! grep -q '"n_ranks"' "$json" 2>/dev/null; then
   143	    echo "[scan] arm $3 exited 0 but $json is missing or has no scored rows"
   144	    return 90
   145	  fi
   146	  return $rc
   147	}
   148	
   149	# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
   150	# makes the bench exit non-zero unless it reproduces the census below, and a
   151	# listed pair that never got scored (e.g. pymetis missing) counts as a
   152	# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
   153	# pass — an instrument that misses the known answer cannot be trusted on the
   154	# unknown one.
   155	S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
   156	S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"
   157	
   158	run_arm 8 64,128 "1/3 VALIDATION s8" "$S8_EXPECT"; RC8=$?
   159	run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?
   160	
   161	if [ "$RC8" -ne 0 ] || [ "$RC9" -ne 0 ]; then
   162	  echo "[scan] INSTRUMENT CHECK FAILED (s8 rc=$RC8, s9 rc=$RC9)."
   163	  echo "[scan] The scorer does not reproduce the reference census, so an s10"
   164	  echo "[scan] number from it would be untrustworthy. NOT running arm 3."
   165	  echo "SCAN_ABORTED_VALIDATION"
   166	  exit 1
   167	fi
   168	
   169	# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
   170	run_arm 10 128 "3/3 UNKNOWN s10" ""; RC10=$?
   171	echo "[scan] arm 3 rc=$RC10"
   172	
   173	if [ "$RC10" -ne 0 ]; then
   174	  # Do NOT let SLURM record this as COMPLETED: there is no valid s10 row,
   175	  # and an exit-0 job with a missing result reads as a successful scan.
   176	  echo "SCAN_FAILED_ARM3"
   177	  exit "$RC10"
   178	fi
   179	
   180	echo "SCAN_DONE"
     1	"""Exit-status regression for the MPAS schedule-cost scan launcher.
     2	
     3	The failure this guards is not hypothetical: the first version of the script
     4	captured arm 3's status into ``RC10`` and then ended on a successful ``echo``,
     5	so a scan that produced NO valid s10 result exited 0 and SLURM filed it as
     6	COMPLETED (codex round 2, BLOCKER).  "Tool status is not evidence" cuts both
     7	ways — a launcher that cannot report failure makes every downstream reading
     8	of ``sacct`` a lie.
     9	
    10	HOW THIS IS DRIVEN, and why not the obvious way: the bench path in the
    11	launcher is deliberately NOT overridable from the environment.  Making it
    12	overridable (the first attempt) handed a stray exported variable the power to
    13	redirect a real scan to something that exits 0 — reintroducing the very
    14	failure class under test (codex round 4).  Instead this substitutes the
    15	INTERPRETER via ``LEGOESM_PYTHON``, which is an existing production knob that
    16	``_env.sh`` already reads, so the launcher itself carries no test-only seam.
    17	
    18	The stub interpreter answers ``_env.sh``'s jax probe, then exits with a
    19	scripted status per arm, recording each arm's full argv so a test can assert
    20	both WHICH arms ran and that each carried its required flags.
    21	"""
    22	from __future__ import annotations
    23	
    24	import os
    25	import shutil
    26	import subprocess
    27	import sys
    28	from pathlib import Path
    29	
    30	import pytest
    31	
    32	_REPO = Path(__file__).resolve().parents[2]
    33	_SBATCH = (_REPO / "scripts" / "cluster" / "scaling_levante"
    34	           / "mpas_schedule_cost_scan.sbatch")
    35	
    36	pytestmark = pytest.mark.skipif(
    37	    shutil.which("bash") is None, reason="needs bash to run the launcher")
    38	
    39	
    40	def _run(tmp_path, codes, write_artifact=True):
    41	    """Run the launcher with a stub interpreter that exits ``codes`` per arm.
    42	
    43	    ``codes`` is one exit status per bench invocation, in order (arm 1, arm
    44	    2, arm 3).  When ``write_artifact`` the stub also writes a minimal JSON
    45	    to the arm's ``--out`` containing an ``n_ranks`` key, which is what the
    46	    launcher's post-arm artifact check looks for; setting it False simulates
    47	    an interpreter that exits 0 having measured nothing.
    48	
    49	    Returns ``(proc, arms)`` where ``arms`` is the recorded argv of each
    50	    bench call.
    51	    """
    52	    log = tmp_path / "calls.txt"
    53	    stub = tmp_path / "stub_python"
    54	    stub.write_text(
    55	        "#!/usr/bin/env python3\n"
    56	        "import sys, json\n"
    57	        f"codes = {list(codes)!r}\n"
    58	        f"log = {str(log)!r}\n"
    59	        f"write_artifact = {bool(write_artifact)!r}\n"
    60	        "argv = sys.argv[1:]\n"
    61	        # _env.sh probes the interpreter with `-c 'import jax...'`; answer it
    62	        # without counting it as a bench call.
    63	        "if argv and argv[0] == '-c':\n"
    64	        "    sys.exit(0)\n"
    65	        "with open(log, 'a') as f:\n"
    66	        "    f.write(json.dumps(argv) + '\\n')\n"
    67	        "n = sum(1 for _ in open(log))\n"
    68	        "rc = codes[n - 1] if n <= len(codes) else 0\n"
    69	        "if write_artifact and rc == 0:\n"
    70	        "    out = argv[argv.index('--out') + 1]\n"
    71	        "    with open(out, 'w') as f:\n"
    72	        "        json.dump({'rows': [{'n_ranks': 64}]}, f)\n"
    73	        "sys.exit(rc)\n"
    74	    )
    75	    stub.chmod(0o755)
    76	
    77	    env = dict(os.environ)
    78	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
    79	    env["LEGOESM_REPO"] = str(_REPO)
    80	    env["LEGOESM_PYTHON"] = str(stub)
    81	    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    82	    proc = subprocess.run(
    83	        ["bash", str(_SBATCH)], env=env, cwd=str(_REPO),
    84	        capture_output=True, text=True, timeout=600)
    85	
    86	    arms = []
    87	    if log.exists():
    88	        import json
    89	        arms = [json.loads(line) for line in log.read_text().splitlines()
    90	                if line.strip()]
    91	    return proc, arms
    92	
    93	
    94	def _levels(arms):
    95	    return [a[a.index("--subdivision") + 1] for a in arms]
    96	
    97	
    98	def test_launcher_is_not_redirectable_from_the_environment():
    99	    """The bench path must be hardcoded.
   100	
   101	    If it were env-overridable, a stray exported variable could point every
   102	    arm at something that exits 0 and the job would report SCAN_DONE with no
   103	    results — the exact failure this module guards.
   104	    """
   105	    text = _SBATCH.read_text()
   106	    assert "BENCH=scripts/bench/bench_voronoi_partition_methods.py" in text
   107	    assert "LEGOESM_SCHEDULE_COST_BENCH" not in text
   108	
   109	
   110	@pytest.mark.parametrize("codes, failing_arm", [([1, 0, 0], 1), ([0, 1, 0], 2)])
   111	def test_validation_failure_aborts_before_the_unknown_arm(
   112	        tmp_path, codes, failing_arm):
   113	    """EITHER validation arm failing must abort non-zero and skip arm 3.
   114	
   115	    An instrument that misses the known census cannot be trusted on the
   116	    unknown one, so producing an s10 number anyway is worse than none.
   117	    Both arms are exercised: guarding only arm 1 leaves arm 2 unchecked.
   118	    """
   119	    proc, arms = _run(tmp_path, codes)
   120	    assert proc.returncode != 0, proc.stdout[-2000:]
   121	    assert "SCAN_ABORTED_VALIDATION" in proc.stdout
   122	    assert "SCAN_DONE" not in proc.stdout
   123	    assert _levels(arms) == ["8", "9"], (
   124	        f"arm 3 must not run after validation arm {failing_arm} failed: "
   125	        f"{_levels(arms)}")
   126	
   127	
   128	def test_arm3_failure_is_not_reported_as_success(tmp_path):
   129	    """The original BLOCKER: arm 3 fails, the job must NOT exit 0, and must
   130	    propagate the exact status."""
   131	    proc, arms = _run(tmp_path, [0, 0, 3])
   132	    assert proc.returncode == 3, (
   133	        f"arm-3 status not propagated (got {proc.returncode})\n"
   134	        f"{proc.stdout[-2000:]}")
   135	    assert "SCAN_FAILED_ARM3" in proc.stdout
   136	    assert "SCAN_DONE" not in proc.stdout
   137	    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
   138	
   139	
   140	def test_full_success_exits_zero_and_runs_every_arm(tmp_path):
   141	    proc, arms = _run(tmp_path, [0, 0, 0])
   142	    assert proc.returncode == 0, proc.stdout[-2000:]
   143	    assert "SCAN_DONE" in proc.stdout
   144	    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
   145	
   146	
   147	def test_each_arm_carries_the_flags_its_claim_depends_on(tmp_path):
   148	    """The exit plumbing being right is worthless if an arm silently stops
   149	    scoring or stops gating.
   150	
   151	    Without this, dropping ``--schedule-cost`` (nothing is scored) or
   152	    ``--expect-rounds`` (the validation arms assert nothing) would leave
   153	    every other test in this module green.
   154	    """
   155	    _, arms = _run(tmp_path, [0, 0, 0])
   156	    assert len(arms) == 3, arms
   157	    for argv in arms:
   158	        assert "--schedule-cost" in argv, argv
   159	        assert "--lloyd" in argv and argv[argv.index("--lloyd") + 1] == "0"
   160	    # Validation arms must actually gate; the unknown arm must not pretend to.
   161	    for argv in arms[:2]:
   162	        assert "--expect-rounds" in argv, argv
   163	        assert argv[argv.index("--expect-rounds") + 1].strip(), argv
   164	    assert "--expect-rounds" not in arms[2], arms[2]
   165	
   166	
   167	def test_arm_that_exits_zero_without_producing_results_is_caught(tmp_path):
   168	    """An exit code is not evidence an arm measured anything.
   169	
   170	    If the interpreter succeeds but writes no JSON (or one with no scored
   171	    rows), the launcher must NOT report SCAN_DONE — otherwise a misconfigured
   172	    run files as COMPLETED with nothing in it, the exact class this module
   173	    exists to prevent.
   174	    """
   175	    proc, arms = _run(tmp_path, [0, 0, 0], write_artifact=False)
   176	    assert proc.returncode != 0, proc.stdout[-2000:]
   177	    assert "has no scored rows" in proc.stdout
   178	    assert "SCAN_DONE" not in proc.stdout
   179	    # Both validation arms still RUN (they are independent measurements and
   180	    # running both reports more before aborting); what matters is that the
   181	    # unknown arm 3 is skipped.
   182	    assert _levels(arms) == ["8", "9"], _levels(arms)
   183	
   184	
   185	def test_stale_artifact_from_a_previous_run_is_removed_before_each_arm(
   186	        tmp_path):
   187	    """A failed rerun must not leave the previous run's JSON in place: a
   188	    stale receipt read as current is worse than a missing one."""
   189	    out_dir = _REPO / "results" / "a1" / "mpas_schedule_cost"
   190	    out_dir.mkdir(parents=True, exist_ok=True)
   191	    stale = out_dir / "schedule_cost_s8.json"
   192	    stale.write_text('{"rows": [{"n_ranks": 999, "STALE": true}]}')
   193	    try:
   194	        # Arm 1 fails and writes nothing; the stale file must be gone, not
   195	        # left behind looking like this run's result.
   196	        proc, _ = _run(tmp_path, [1, 0, 0], write_artifact=False)
   197	        assert proc.returncode != 0
   198	        assert not stale.exists(), (
   199	            "stale s8 JSON survived a failed arm and would read as current")
   200	    finally:
   201	        stale.unlink(missing_ok=True)
   202	
   203	
   204	def test_launcher_expectations_are_the_full_six_row_census():
   205	    """The gate's strength is its CARDINALITY and content, not its presence.
   206	
   207	    Asserting only that --expect-rounds is non-empty (the previous test)
   208	    would still pass if five of the six rows were deleted, because the stub
   209	    never runs the real parser.  Parse the launcher's own literals with the
   210	    production parser and check every method x rank pair is present.
   211	    """
   212	    import importlib.util
   213	
   214	    spec = importlib.util.spec_from_file_location(
   215	        "bench_vor_part_gate",
   216	        _REPO / "scripts" / "bench" / "bench_voronoi_partition_methods.py")
   217	    bench = importlib.util.module_from_spec(spec)
   218	    spec.loader.exec_module(bench)
   219	
   220	    text = _SBATCH.read_text()
   221	    for var in ("S8_EXPECT", "S9_EXPECT"):
   222	        line = next(ln for ln in text.splitlines()
   223	                    if ln.startswith(f"{var}="))
   224	        parsed = bench.parse_expect_rounds(line.split("=", 1)[1].strip('"'))
   225	        assert set(parsed) == {
   226	            (m, n) for m in ("geometric", "sfc", "metis") for n in (64, 128)
   227	        }, f"{var} is not the full six-row census: {sorted(parsed)}"

exec
/usr/bin/bash -lc "nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '300,540p'; nl -ba tests/bench/test_bench_voronoi_partition_methods.py | sed -n '1,520p'; nl -ba scripts/cluster/scaling_levante/_env.sh | sed -n '1,240p'; rg -n \"spmd_schedule_cost|def create_voronoi_mesh|def scaling_metadata\" legoesm scripts tests | head -100" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 1173ms:
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
   373	    if not methods:
   374	        # Same class as an empty --expect-rounds: the loop would be skipped,
   375	        # `rows: []` written, and 0 returned — an empty run that reads as a
   376	        # successful one.
   377	        raise SystemExit(
   378	            f"--methods={args.methods!r} selects NO methods; the benchmark "
   379	            f"would measure nothing and still exit 0. Choose from {METHODS}.")
   380	    # Negative values in these three are all the SAME false-provenance bug:
   381	    # the underlying code treats them exactly like 0 (no relaxation, no
   382	    # bisection, `range(-1)` is empty), but the run is recorded under the
   383	    # negative value, so a level-0 mesh gets filed as "L-1".
   384	    if args.lloyd < 0:
   385	        raise SystemExit(
   386	            f"--lloyd must be >= 0, got {args.lloyd}: the mesh builder "
   387	            f"relaxes only for > 0, so a negative is indistinguishable from "
   388	            f"0 in the mesh but is recorded and cached under its own key.")
   389	    if args.subdivision < 0:
   390	        raise SystemExit(
   391	            f"--subdivision must be >= 0, got {args.subdivision}: the mesh "
   392	            f"builder bisects only for > 0, so a negative silently yields "
   393	            f"the level-0 base mesh while being recorded as "
   394	            f"L{args.subdivision}.")
   395	    if args.halo_depth < 0:
   396	        raise SystemExit(
   397	            f"--halo-depth must be >= 0, got {args.halo_depth}: the halo "
   398	            f"loop is `range(depth)`, so a negative behaves exactly like 0 "
   399	            f"while being recorded as {args.halo_depth}.")
   400	    expect = parse_expect_rounds(args.expect_rounds)
   401	    if expect and not args.schedule_cost:
   402	        raise SystemExit(
   403	            "--expect-rounds compares scored round counts, so it needs "
   404	            "--schedule-cost; without it nothing is scored and the gate "
   405	            "would pass vacuously.")
   406	
   407	
   408	    from legoesm.grids.voronoi import create_voronoi_mesh
   409	    from legoesm.parallel.voronoi_partition import resolve_partition_method
   410	
   411	    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
   412	                               lloyd_iterations=args.lloyd)
   413	    if max(rank_counts) > int(mesh.nCells):
   414	        raise SystemExit(
   415	            f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
   416	            f"{int(mesh.nCells)} cells (empty ranks are meaningless).")
   417	    print(f"mesh L{args.subdivision}: {int(mesh.nCells)} cells, "
   418	          f"{int(mesh.nEdges)} edges; auto -> "
   419	          f"{resolve_partition_method('auto')!r}")
   420	
   421	    rows = []
   422	    for method in methods:
   423	        if not method_available(method):
   424	            print(f"  {method:9s}: UNAVAILABLE (pymetis not importable) — "
   425	                  f"column omitted, never substituted")
   426	            rows.append({"method": method, "available": False})
   427	            continue
   428	        for n_ranks in rank_counts:
   429	            q = partition_quality(
   430	                mesh, owner_for(mesh, method, n_ranks), n_ranks,
   431	                halo_depth=args.halo_depth)
   432	            row = {"method": method, "available": True,
   433	                   "n_ranks": n_ranks, **q}
   434	            rows.append(row)
   435	            print(f"  {method:9s} np={n_ranks:3d} | "
   436	                  f"imbalance={q['load_imbalance_max_over_mean']:.3f} | "
   437	                  f"edge_cut={q['edge_cut']:6d} "
   438	                  f"({100 * q['edge_cut_fraction']:.2f}%) | "
   439	                  f"halo max={q['halo_cells_max']:5d} "
   440	                  f"mean={q['halo_cells_mean']:8.1f} | "
   441	                  f"halo/owned max={q['halo_owned_ratio_max']:.3f} | "
   442	                  f"nbrs max={q['neighbor_ranks_max']}")
   443	            if args.schedule_cost:
   444	                sc = schedule_cost_row(mesh, method, n_ranks)
   445	                row["schedule"] = sc
   446	                note = ("  [COUNTERFACTUAL: production auto-selects "
   447	                        "allgather here, no ppermute schedule]"
   448	                        if sc["production_strategy"] == "allgather" else "")
   449	                verdict = ("colouring PROVABLY OPTIMAL (Vizing floor)"
   450	                           if sc["coloring_optimal_proven"] else
   451	                           f"recolour headroom "
   452	                           f"{sc['coloring_headroom_rounds_min']}-"
   453	                           f"{sc['coloring_headroom_rounds_max']} round(s)")
   454	                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
   455	                      f"rounds={sc['n_rounds']:3d} "
   456	                      f"max_degree={sc['max_degree']:3d} "
   457	                      f"gap={sc['coloring_gap']:+d} "
   458	                      f"-> {verdict} "
   459	                      f"({sc['score_seconds']:.1f}s){note}", flush=True)
   460	
   461	    # An "empty but successful" run is the same failure class as a skipped
   462	    # gate: --methods metis on a box without pymetis appends only an
   463	    # unavailable row and would otherwise exit 0 having measured nothing
   464	    # (codex round 5).
   465	    if not any(r.get("available") and "n_ranks" in r for r in rows):
   466	        raise SystemExit(
   467	            f"no method was actually measured (requested {methods}); every "
   468	            f"one was unavailable, so this run has no results and must not "
   469	            f"report success.")
   470	
   471	    payload = {
   472	        "rows": rows,
   473	        "auto_resolves_to": resolve_partition_method("auto"),
   474	        "step_time_pointer": (
   475	            "step-time per method: bench_ocean_mpas_scaling.py / "
   476	            "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
   477	            "per launch, same case otherwise)"),
   478	        "metadata": annotate_incomplete(scaling_metadata(
   479	            grid="voronoi",
   480	            component="partitioning",
   481	            resolution=f"L{args.subdivision}",
   482	            n_levels=0,
   483	            precision="n/a",
   484	            decomposition="cell_partition",
   485	            solver_variant="n/a",
   486	            scaling_kind="partition-quality",
   487	            transport="none",
   488	            extra={"rank_counts": rank_counts, "methods": methods,
   489	                   "halo_depth": args.halo_depth,
   490	                   "schedule_cost": bool(args.schedule_cost),
   491	                   "lloyd_iterations": args.lloyd},
   492	        )),
   493	    }
   494	    outdir = os.path.dirname(args.out)
   495	    if outdir:
   496	        os.makedirs(outdir, exist_ok=True)
   497	    with open(args.out, "w") as f:
   498	        json.dump(payload, f, indent=2)
   499	    print(f"JSON: {args.out}")
   500	
   501	    if expect:
   502	        failures = check_expected_rounds(rows, expect)
   503	        payload["expected_rounds_check"] = {
   504	            "expected": {f"{m}:{n}": v for (m, n), v in sorted(expect.items())},
   505	            "failures": failures,
   506	            "passed": not failures,
   507	        }
   508	        with open(args.out, "w") as f:
   509	            json.dump(payload, f, indent=2)
   510	        if failures:
   511	            print(f"INSTRUMENT CHECK FAILED ({len(failures)}):")
   512	            for line in failures:
   513	                print(f"  {line}")
   514	            print("The scorer did NOT reproduce the known census — treat every "
   515	                  "unknown row in this run as UNTRUSTED.")
   516	            return 1
   517	        print(f"INSTRUMENT CHECK PASSED ({len(expect)} expectations)")
   518	    return 0
   519	
   520	
   521	if __name__ == "__main__":
   522	    raise SystemExit(main())
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
   448	
   449	
   450	def test_empty_methods_refused_instead_of_measuring_nothing(monkeypatch):
   451	    """``--methods " , , "`` used to select nothing, write ``rows: []`` and
   452	    exit 0 — an empty run that reads as a successful one (codex round 4)."""
   453	    monkeypatch.setattr(sys, "argv", [
   454	        "bench", "--subdivision", "2", "--rank-counts", "2",
   455	        "--methods", " , , ", "--out", "/dev/null"])
   456	    with pytest.raises(SystemExit, match="selects NO methods"):
   457	        mod.main()
     1	# Shared environment for DKRZ Levante GPU scaling jobs (sourced by the SLURM
     2	# scripts here).  SLURM/OpenMPI twin of scripts/cluster/scaling_derecho/_env.sh
     3	# (which is PBS/Cray-MPICH).
     4	# ---------------------------------------------------------------------------
     5	# EDIT the marked values (account / repo / conda env / module versions) before
     6	# the first submit.  Everything is overridable from the sbatch environment, e.g.
     7	#   sbatch --export=ALL,LEGOESM_CONDA_ENV=my-jax-env gpu_moist_scaling.slurm
     8	#
     9	# Levante GPU partition (partition `gpu`): 60 nodes, each 2x AMD EPYC 7763 +
    10	# 4x NVIDIA A100 (56 nodes 80GB, 4 nodes 40GB), InfiniBand HDR200.  MPI stack is
    11	# OpenMPI over UCX with CUDA-aware transports -- NOT Cray MPICH.
    12	# ---------------------------------------------------------------------------
    13	
    14	# --- (1) Project allocation (matches SBATCH --account in the job scripts) -----
    15	#     Levante GPU jobs bill a *_gpu sub-account (run_levante_gpu_scaling.sh uses
    16	#     bd1083_gpu, matching the SBATCH --account in the .slurm, which is the
    17	#     source of truth).  This default is only for interactive sourcing.
    18	export LEGOESM_SLURM_ACCOUNT="${LEGOESM_SLURM_ACCOUNT:-bd1083_gpu}"
    19	
    20	# --- (2) Repo location on Levante -- EDIT to where you cloned legoESM ---------
    21	# The default is a GUESS at a per-user clone path. When it is wrong the job
    22	# does not fail here — it fails ~60 lines later with a bare
    23	# "cd: <path>: No such file or directory" plus "_chain_body.sh: No such file",
    24	# 7 seconds in, which reads like a broken launcher rather than an unset
    25	# variable (three U-Cast arms lost this way, 2026-08-01). Say it plainly.
    26	REPO="${LEGOESM_REPO:-/work/bd1083/$USER/legoESM}"
    27	if [ ! -d "$REPO" ]; then
    28	  echo "[_env.sh] REPO='$REPO' does not exist." >&2
    29	  if [ -z "${LEGOESM_REPO:-}" ]; then
    30	    echo "[_env.sh] LEGOESM_REPO is unset, so this is the per-user DEFAULT" >&2
    31	    echo "[_env.sh] guess, not a configured path. Submit with" >&2
    32	    echo "[_env.sh]   sbatch --export=ALL,LEGOESM_REPO=\$PWD,... " >&2
    33	    echo "[_env.sh] (--export=ALL alone does NOT carry it if your shell" >&2
    34	    echo "[_env.sh]  never exported it)." >&2
    35	  fi
    36	  exit 1
    37	fi
    38	export REPO
    39	
    40	# --- (3) Conda env with a CUDA jaxlib AND a CUDA-aware mpi4jax (see README) ---
    41	CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
    42	
    43	# --- Federation PYTHONPATH (belt-and-braces; `pip install -e .` makes it
    44	#     redundant but harmless) ------------------------------------------------
    45	PP="$REPO/src"
    46	for p in atmosphere core coupler ice land ml ocean tools; do
    47	  PP="$PP:$REPO/packages/$p"
    48	done
    49	export PYTHONPATH="$PP:${PYTHONPATH:-}"
    50	
    51	# --- Modules + conda -- EDIT the module versions to the Levante stack you built
    52	#     mpi4py / mpi4jax against (README Step 1); pinned versions matter because
    53	#     the runtime libmpi ABI must match the build ABI ------------------------
    54	module load python3 2>/dev/null || true      # EDIT: e.g. python3/2023.01-gcc-11.2.0
    55	module load openmpi 2>/dev/null || true       # EDIT: the CUDA-aware openmpi you built against
    56	module load cuda    2>/dev/null || true       # EDIT: matching cuda toolkit
    57	if command -v conda >/dev/null 2>&1; then
    58	  conda activate "$CONDA_ENV" 2>/dev/null || true
    59	fi
    60	# Prefer the repo's own uv venv when it exists — that is the interpreter every
    61	# dev/test workflow uses, and the bare `python` on a Levante compute node has
    62	# no jax (three U-Cast arms died at `import jax` inside 7 s, 2026-08-02).
    63	if [ -z "${LEGOESM_PYTHON:-}" ] && [ -x "$REPO/.venv/bin/python" ]; then
    64	  LEGOESM_PYTHON="$REPO/.venv/bin/python"
    65	fi
    66	PY="${LEGOESM_PYTHON:-$(command -v python)}"
    67	export PY
    68	# Fail at source time, not 4 GPU-hours in: the launcher's first real work is
    69	# `$PY scripts/run/run_aimip.py`, which imports jax immediately.
    70	if ! "$PY" -c "import jax" >/dev/null 2>&1; then
    71	  echo "[_env.sh] PY='$PY' cannot import jax." >&2
    72	  echo "[_env.sh] Set LEGOESM_PYTHON=<repo>/.venv/bin/python (uv venv) or" >&2
    73	  echo "[_env.sh] LEGOESM_CONDA_ENV=<env with a CUDA jaxlib>." >&2
    74	  exit 1
    75	fi
    76	
    77	# --- JAX / runtime knobs -----------------------------------------------------
    78	export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
    79	export MPI4JAX_NO_WARN_JAX_VERSION=1
    80	export MPLBACKEND="${MPLBACKEND:-Agg}"          # headless plotting
    81	# DKRZ scratch is /scratch/<first-letter-of-user>/<user>.
    82	export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
    83	# Persistent JIT cache reuses compiles across runs; set empty to force a cold
    84	# compile (true compile_time_s).  On SCRATCH so it survives between jobs.
    85	export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-$SCRATCH/legoesm_jit_cache}"
    86	
    87	# --- OpenMPI + UCX CUDA-aware fabric (GPU route-A) ---------------------------
    88	# Route-A hands the on-device sendrecv buffer straight to MPI (the whole point:
    89	# no device->host->device staging, which would erase multi-GPU scaling).  On
    90	# Levante that path is OpenMPI-over-UCX; the pml/osc + UCX transports below turn
    91	# on GPU-direct: cuda_copy + cuda_ipc intra-node, gdr_copy over InfiniBand HDR
    92	# inter-node.  Requires a CUDA-aware mpi4jax (README) + MPI4JAX_USE_CUDA_MPI=1
    93	# (set in the job script).  UCX_MEMTYPE_CACHE=n avoids a stale device/host
    94	# memtype-cache hang that CUDA-aware sendrecv is prone to.
    95	export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
    96	export OMPI_MCA_osc="${OMPI_MCA_osc:-ucx}"
    97	export UCX_TLS="${UCX_TLS:-rc,cuda_copy,cuda_ipc,gdr_copy,sm,self}"
    98	export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
    99	export UCX_RNDV_SCHEME="${UCX_RNDV_SCHEME:-put_zcopy}"
   100	
   101	# --- NCCL over InfiniBand (route-B: jax.distributed multi-node lanes) --------
   102	# NCCL (shard_map/ppermute collectives under jax.distributed) uses its own
   103	# IB-verbs stack — independent of the UCX/MPI settings above; the two configs
   104	# coexist. Bootstrap ring runs over IPoIB: verify the interface name once with
   105	# `ip addr` on a gpu node (a wrong NCCL_SOCKET_IFNAME is the #1 cause of
   106	# multi-node NCCL bootstrap timeouts on IB clusters).
   107	export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
   108	export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
   109	# Prefix-match BOTH HCAs (mlx5_0/mlx5_1 — one per socket on Levante nodes).
   110	export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
   111	# GPUDirect RDMA when NIC and GPU share a NUMA/PCIe root.
   112	export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
   113	export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
   114	
   115	# --- XLA overlap defaults for the lat-lon SPMD lanes (2026-08-04) --------
   116	# Latency-hiding scheduler + pipelined p2p: -8.4% at LL2048@64 (job
   117	# 26677602) and -8.3% at @128 (26677668), A/A2 drift 0.3-0.4% — twice-
   118	# reproduced, parity suites green with flags on. MPAS lane: null (0.0%,
   119	# 26677669 — its edge-coloured schedule does not benefit; harmless).
   120	# Below the pre-registered 10% bar AND other lanes are unvalidated
   121	# (cube_tiled_step.sbatch force-disables latency hiding for a known
   122	# comm-init sensitivity; MPAS is null) — so this is strictly OPT-IN
   123	# (codex r23): set LEGOESM_XLA_OVERLAP=1 in validated lat-lon
   124	# launchers; never a shared default, and A/B control arms must keep
   125	# REPLACING XLA_FLAGS, not appending.
   126	if [ "${LEGOESM_XLA_OVERLAP:-0}" = 1 ]; then
   127	  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_p2p=true"
   128	fi
   129	
   130	export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
   131	mkdir -p "$TMPDIR" 2>/dev/null || true
   132	
   133	# #1361 memory preflight: target device whose HBM the benches gate against
   134	# (`--device-hbm`). Set in the SHARED env so the gate is on for every launcher
   135	# that sources this file — codex found the Derecho-only export left every
   136	# Levante bench ungated. Levante's GPU jobs request `--constraint=a100_80`.
   137	export LEGOESM_DEVICE_HBM="${LEGOESM_DEVICE_HBM:-a100-80}"
rg: legoesm: No such file or directory (os error 2)
scripts/bench/metadata.py:413:def scaling_metadata(
scripts/bench/bench_voronoi_partition_methods.py:19:   ``spmd_schedule_cost`` (which calls the production builders), never a
scripts/bench/bench_voronoi_partition_methods.py:263:    :func:`legoesm.parallel.sharded_dynamics.spmd_schedule_cost` — it reorders
scripts/bench/bench_voronoi_partition_methods.py:295:    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
scripts/bench/bench_voronoi_partition_methods.py:298:    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
tests/bench/test_bench_voronoi_partition_methods.py:207:    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
tests/bench/test_bench_voronoi_partition_methods.py:246:    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
tests/bench/test_bench_voronoi_partition_methods.py:249:    ref = spmd_schedule_cost(mesh, 2, method="sfc")
tests/parallel/test_spmd_schedule_cost.py:1:"""Direct tests for ``sharded_dynamics.spmd_schedule_cost``.
tests/parallel/test_spmd_schedule_cost.py:22:    c = sd.spmd_schedule_cost(mesh, 4)
tests/parallel/test_spmd_schedule_cost.py:40:    c = sd.spmd_schedule_cost(mesh, 8)
tests/parallel/test_spmd_schedule_cost.py:46:    assert sd.spmd_schedule_cost(mesh, 1)["n_rounds"] == 0
tests/parallel/test_spmd_schedule_cost.py:48:        sd.spmd_schedule_cost(mesh, 0)
tests/parallel/test_spmd_schedule_cost.py:54:    c = sd.spmd_schedule_cost(mesh, 8,
tests/parallel/test_spmd_schedule_cost.py:57:    big = sd.spmd_schedule_cost(mesh, 8,
tests/parallel/test_spmd_schedule_cost.py:67:    pre = sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
tests/parallel/test_spmd_schedule_cost.py:68:    raw = sd.spmd_schedule_cost(mesh, 8, method="sfc")
tests/parallel/test_spmd_schedule_cost.py:77:        sd.spmd_schedule_cost(prepared, 8, already_reordered=True,
tests/parallel/test_spmd_schedule_cost.py:85:    same = sd.spmd_schedule_cost(mesh, 4, method="sfc")
tests/parallel/test_spmd_schedule_cost.py:86:    split_for_16 = sd.spmd_schedule_cost(mesh, 4, method="sfc",
tests/parallel/test_spmd_schedule_cost.py:97:    a = sd.spmd_schedule_cost(mesh, 8, method="sfc")
tests/parallel/test_spmd_schedule_cost.py:98:    b = sd.spmd_schedule_cost(mesh, 8, method="sfc")
tests/parallel/test_spmd_schedule_cost.py:115:    r = {m: sd.spmd_schedule_cost(big, 16, method=m)["n_rounds"]
tests/parallel/test_spmd_schedule_cost.py:125:        sd.spmd_schedule_cost(mesh, 4, method="not_a_method")
tests/parallel/test_spmd_schedule_cost.py:138:    sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
tests/parallel/test_spmd_schedule_cost.py:147:        sd.spmd_schedule_cost(mesh, 4, method="sfc", reorder_target=3)
tests/parallel/test_spmd_schedule_cost.py:152:        sd.spmd_schedule_cost(mesh, 3.9)
tests/parallel/test_spmd_schedule_cost.py:160:    assert (inspect.signature(sd.spmd_schedule_cost)
tests/parallel/test_spmd_schedule_cost.py:169:    assert sd.spmd_schedule_cost(mesh, 1)["production_strategy"] is None
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:70:# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's

exec
/usr/bin/bash -lc "nl -ba tests/bench/test_bench_voronoi_partition_methods.py | sed -n '145,330p'; find . -path '*sharded_dynamics.py' -o -path '*voronoi.py' | head -40; rg -n -A150 -B15 \"def spmd_schedule_cost\" . --glob '*.py'; rg -n -A100 -B15 \"def create_voronoi_mesh\" . --glob '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "bash -n scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && bash -c 'f(){ printf \"<%s>\\n\" "'${4:+--expect-rounds "$4"}; }; f 1 2 3 "sfc:64=12,sfc:128=14"'"' && git diff --check && git ls-files .github/workflows && rg -n \"tests/bench|pytest|bench\" .github/workflows .github 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 274ms:
<--expect-rounds>
<sfc:64=12,sfc:128=14>
.github/workflows/ci.yml
.github/workflows/claude-code-review.yml
.github/workflows/claude.yml
.github/workflows/mpi-distributed.yml
.github/workflows/mpi-nightly.yml
.github/workflows/mpi-distributed.yml:52:              python -m pytest -q tests/distributed/test_halo_mpi.py
.github/workflows/mpi-distributed.yml:62:              python -m pytest -q tests/distributed/test_plane_pencil_mpi.py
.github/workflows/mpi-distributed.yml:75:              python -m pytest -q tests/distributed/test_latlon_2d_pad_wall_mpi.py
.github/workflows/mpi-distributed.yml:88:              python -m pytest -q tests/distributed/test_latlon_transpose_ad_mpi.py
.github/workflows/ci.yml:97:        run: python -m pytest --collect-only tests/ 2>&1
.github/workflows/ci.yml:118:      # leaks across the pytest session for subsequently-imported modules.
.github/workflows/ci.yml:121:          python -m pytest tests/unit/ -x --timeout=300 \
.github/workflows/ci.yml:156:        # `bash -c` is required so the glob expands before pytest
.github/workflows/ci.yml:157:        # sees the arguments; otherwise pytest gets the literal
.github/workflows/ci.yml:181:          # returns, the deterministic fallback is one pytest process per file.
.github/workflows/ci.yml:182:          python -m pytest "${files[@]}" --timeout=300 \
.github/workflows/ci.yml:196:          python -m pytest tests/grids/test_fv3_native_grid_phase1.py \
.github/workflows/ci.yml:213:          python -m pytest tests/grids/test_fv3_native_swcore_phase4.py \
.github/workflows/ci.yml:229:          python -m pytest tests/unit/test_cmor_experiments_restart.py \
.github/workflows/ci.yml:245:          python -m pytest tests/atmosphere/ -x --timeout=600 \
.github/workflows/mpi-nightly.yml:47:            python -m pytest -q tests/distributed/test_ocean_mpi_conservation.py -k longrun
.github/workflows/ci.yml:97:        run: python -m pytest --collect-only tests/ 2>&1
.github/workflows/ci.yml:118:      # leaks across the pytest session for subsequently-imported modules.
.github/workflows/ci.yml:121:          python -m pytest tests/unit/ -x --timeout=300 \
.github/workflows/ci.yml:156:        # `bash -c` is required so the glob expands before pytest
.github/workflows/ci.yml:157:        # sees the arguments; otherwise pytest gets the literal
.github/workflows/ci.yml:181:          # returns, the deterministic fallback is one pytest process per file.
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

codex
I’ve confirmed the new “no artifact” path reaches both validation arms, then blocks arm 3 as intended. I also found the artifact predicate is still only a text-substring check, so I’m checking its test coverage and the expectation literals for bypasses.
exec
/usr/bin/bash -lc "nl -ba .github/workflows/ci.yml | sed -n '1,280p'; git log -1 --format='%H%n%ad%n%s' --date=iso -- .github/workflows/ci.yml scripts/cluster/scaling_levante/_env.sh scripts/cluster/scaling_levante/prewarm_s10.sbatch; nl -ba scripts/cluster/scaling_levante/prewarm_s10.sbatch | sed -n '80,180p'; rg -n \"schedule_cost_s|mpas_schedule_cost\" .gitignore .  --glob '"'!tests/bench/test_mpas_schedule_cost_scan_sbatch.py'"' --glob '"'!scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch'"' | head -200" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 3088ms:
     1	name: CI
     2	
     3	on:
     4	  push:
     5	    branches: [main]
     6	  pull_request:
     7	    branches: [main]
     8	  schedule:
     9	    # Nightly (03:17 UTC) — runs the non-blocking visual-regression diagnostic.
    10	    - cron: "17 3 * * *"
    11	  workflow_dispatch: {}
    12	
    13	concurrency:
    14	  group: ci-${{ github.ref }}
    15	  cancel-in-progress: true
    16	
    17	jobs:
    18	  install-smoke:
    19	    name: Install & Import Smoke
    20	    runs-on: ubuntu-latest
    21	    strategy:
    22	      matrix:
    23	        python-version: ["3.11", "3.12"]
    24	    steps:
    25	      - uses: actions/checkout@v4
    26	      - uses: actions/setup-python@v5
    27	        with:
    28	          python-version: ${{ matrix.python-version }}
    29	      - name: Install package
    30	        run: pip install -e ".[dev]"
    31	      - name: Import smoke test
    32	        run: |
    33	          python -c "import legoesm; print(f'legoESM {legoesm.__version__}')"
    34	          python -c "from legoesm.config import Config"
    35	          python -c "from legoesm.core.hardware import detect_devices"
    36	          python -c "from legoesm.driver.config import ExperimentConfig"
    37	          python -c "from legoesm.io.restart import save_restart, load_restart"
    38	          python -c "import legoesm.coupler"
    39	          python -c "import legoesm.ocean"
    40	          python -c "import legoesm.ml"
    41	          python -c "import legoesm.da"
    42	          echo "All imports successful"
    43	
    44	  lint:
    45	    name: Lint
    46	    runs-on: ubuntu-latest
    47	    steps:
    48	      - uses: actions/checkout@v4
    49	      - uses: actions/setup-python@v5
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
   136	    # Scope: only files matching `tests/test_*.py` at the top level
   137	    # of `tests/` — not subdirectories.  Subdirectory tests
   138	    # (`tests/atmosphere/`, `tests/distributed/`, etc.) have their
   139	    # own CI jobs or are deliberately excluded for cost.
   140	    name: Top-level Fortran-fidelity Tests
   141	    runs-on: ubuntu-latest
   142	    needs: [install-smoke, test-collect]
   143	    strategy:
   144	      matrix:
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
   196	          python -m pytest tests/grids/test_fv3_native_grid_phase1.py \
   197	            tests/grids/test_fv3_native_metrics_phase2.py \
   198	            tests/grids/test_fv3_native_halos_phase3.py \
   199	            tests/grids/test_gnomonic_ed_gate_iter68.py \
   200	            tests/grids/test_gnomonic_ed_centers_iter62.py \
   201	            tests/grids/test_gnomonic_ed_halo_nocollapse_iter73.py \
   202	            -x --timeout=600 --tb=short -q
   203	      - name: FV3 phase-4 c_sw oracle (Fortran rebuild REQUIRED)
   204	        # codex r2 P1-3a: the verbatim-Fortran rebuild-and-compare test
   205	        # must EXECUTE in CI, not sit behind a skipif.  gfortran is
   206	        # asserted so a toolchain regression fails loudly instead of
   207	        # silently skipping the oracle.
   208	        env:
   209	          JAX_ENABLE_X64: "1"
   210	        run: |
   211	          sudo apt-get update -qq && sudo apt-get install -y -qq gfortran
   212	          gfortran --version
   213	          python -m pytest tests/grids/test_fv3_native_swcore_phase4.py \
   214	            -x --timeout=900 --tb=short -q
   215	        timeout-minutes: 30
   216	
   217	  restart-smoke:
   218	    name: Restart Roundtrip
   219	    runs-on: ubuntu-latest
   220	    needs: [install-smoke]
   221	    steps:
   222	      - uses: actions/checkout@v4
   223	      - uses: actions/setup-python@v5
   224	        with:
   225	          python-version: "3.11"
   226	      - run: pip install -e ".[dev]"
   227	      - name: Restart roundtrip test
   228	        run: |
   229	          python -m pytest tests/unit/test_cmor_experiments_restart.py \
   230	            tests/unit/test_zarr_checkpoint.py -v --tb=short
   231	        timeout-minutes: 15
   232	
   233	  integration-smoke:
   234	    name: Integration Smoke
   235	    runs-on: ubuntu-latest
   236	    needs: [unit-tests]
   237	    steps:
   238	      - uses: actions/checkout@v4
   239	      - uses: actions/setup-python@v5
   240	        with:
   241	          python-version: "3.11"
   242	      - run: pip install -e ".[dev]"
   243	      - name: Atmosphere integration smoke
   244	        run: |
   245	          python -m pytest tests/atmosphere/ -x --timeout=600 \
   246	            -k "test_williamson or test_held_suarez" \
   247	            --tb=short -q 2>&1 | head -50
   248	        timeout-minutes: 30
   249	
   250	  # Cube-artifact visual-regression diagnostic. NON-BLOCKING (continue-on-error)
   251	  # and scheduled/manual only: the deterministic SSIM/perceptual-hash metrics are
   252	  # already gated in CI by tests/test_visual_regression_metrics.py (in unit-tests),
   253	  # but the full cube shallow-water --check tolerances still need calibration
   254	  # across CI hardware / JAX versions before it can block a build. Promote to a
   255	  # required job once it has demonstrated low false-fail on this runner.
   256	  visual-regression:
   257	    name: Visual Regression (cube imprint, nightly, non-blocking)
   258	    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
   259	    runs-on: ubuntu-latest
   260	    continue-on-error: true
   261	    steps:
   262	      - uses: actions/checkout@v4
   263	      - uses: actions/setup-python@v5
   264	        with:
   265	          python-version: "3.11"
   266	      - run: pip install -e ".[dev]"
   267	      - name: W2 cube v-wind visual-regression check
   268	        env:
   269	          JAX_ENABLE_X64: "1"
   270	          JAX_PLATFORMS: "cpu"
   271	        run: |
   272	          python scripts/validate/visual_regression.py --check 2>&1 | tail -20
   273	        timeout-minutes: 20
86842d5fb0f57407147e8fd511656a7cb4dbc83d
2026-08-07 10:27:29 +0200
docs(claude): plain language first — the communication rule that worked
./tests/bench/test_bench_voronoi_partition_methods.py:232:def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
./scripts/bench/bench_voronoi_partition_methods.py:78:      --out results/a1/schedule_cost_s9.json

exec
/usr/bin/bash -lc "rg --files scripts/cluster/scaling_levante | sort; rg -n \"LEGOESM_PYTHON|LEGOESM_REPO|source .*_env\" scripts/cluster/scaling_levante --glob '*.sbatch' --glob '*.slurm' | head -220; git check-ignore -v results/a1/mpas_schedule_cost/schedule_cost_s8.json || true; git status --short -- results/a1/mpas_schedule_cost; git diff --numstat -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 432ms:
scripts/cluster/scaling_levante/README.md
scripts/cluster/scaling_levante/_env.sh
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch
scripts/cluster/scaling_levante/atm_latlon_144.sbatch
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch
scripts/cluster/scaling_levante/cpu_scaling.sbatch
scripts/cluster/scaling_levante/cube_tiled_step.sbatch
scripts/cluster/scaling_levante/diagnosis.sbatch
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch
scripts/cluster/scaling_levante/gpu_scaling.sbatch
scripts/cluster/scaling_levante/mpas_bound_base.sbatch
scripts/cluster/scaling_levante/mpas_s10_128.sbatch
scripts/cluster/scaling_levante/mpas_s10_192.sbatch
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
scripts/cluster/scaling_levante/ocean_latlon_hundreds.sbatch
scripts/cluster/scaling_levante/prewarm_s10.sbatch
scripts/cluster/scaling_levante/atm_latlon_144.sbatch:30:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_latlon_144.sbatch:31:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_latlon_144.sbatch:32:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:30:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:31:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:32:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:25:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:26:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:27:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:35:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:36:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:37:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:38:source "${SCRIPT_DIR}/_env.sh"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:95:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:96:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:97:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:32:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:33:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:34:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/cpu_scaling.sbatch:46:source "${SCRIPT_DIR}/_env.sh"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:55:source "${SCRIPT_DIR}/_env.sh"
scripts/cluster/scaling_levante/prewarm_s10.sbatch:18:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/prewarm_s10.sbatch:19:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/prewarm_s10.sbatch:20:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch:21:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch:22:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch:23:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:17:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:18:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/bin/python}"
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:19:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/ocean_latlon_hundreds.sbatch:22:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/ocean_latlon_hundreds.sbatch:23:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/ocean_latlon_hundreds.sbatch:24:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/diagnosis.sbatch:53:source "${SCRIPT_DIR}/_env.sh"
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:32:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:33:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:34:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:52:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:53:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:54:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/gpu_scaling.sbatch:46:source "${SCRIPT_DIR}/_env.sh"
scripts/cluster/scaling_levante/mpas_bound_base.sbatch:22:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_bound_base.sbatch:23:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_bound_base.sbatch:24:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm:67:source "${SCRIPT_DIR}/_env.sh"
scripts/cluster/scaling_levante/mpas_s10_192.sbatch:20:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_s10_192.sbatch:21:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_s10_192.sbatch:22:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:25:export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:26:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:27:source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
.gitignore:29:results/	results/a1/mpas_schedule_cost/schedule_cost_s8.json
304	2	scripts/bench/bench_voronoi_partition_methods.py
309	0	tests/bench/test_bench_voronoi_partition_methods.py

exec
/usr/bin/bash -lc "rg --files packages | rg 'sharded_dynamics\\.py"'$|grids/voronoi'"\\.py"'$'"' | head -40; nl -ba scripts/cluster/scaling_levante/prewarm_s10.sbatch | sed -n '1,100p'; nl -ba .gitignore | sed -n '20,35p'; rg -n -A210 -B10 '"'^def spmd_schedule_cost'"' packages --glob 'sharded_dynamics.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 385ms:
packages/core/legoesm/grids/voronoi.py
packages/core/legoesm/parallel/sharded_dynamics.py
     1	#!/bin/bash -l
     2	#SBATCH --job-name=prewarm_s10
     3	#SBATCH --account=bb1596
     4	#SBATCH --partition=shared
     5	#SBATCH --ntasks=1
     6	#SBATCH --cpus-per-task=8
     7	#SBATCH --mem=120G
     8	#SBATCH --time=10:00:00
     9	#SBATCH --output=prewarm_s10.%j.log
    10	# Prewarm subdiv-10 lloyd=0 (10.5M cells, ~7 GB npz) into the shared
    11	# mesh cache — unlocks MPAS at 128-224 GPUs ABOVE the ~30k tile floor
    12	# (np128 = 81.9k, np224 = 46.8k cells/GPU).  s9 (2.62M) built in 67 min;
    13	# s10 estimated ~4-5 h.
    14	set -uo pipefail
    15	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    16	export JAX_PLATFORMS=cpu
    17	export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
    18	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    19	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    20	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    21	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    22	"$PY" scripts/data/prewarm_voronoi_mesh.py --level 10 --lloyd 0
    20	*.out
    21	*.toc
    22	*.synctex.gz
    23	
    24	# Data and results artifacts
    25	*.zarr
    26	*.zarr/
    27	*.nc
    28	*.nc4
    29	results/
    30	checkpoints/
    31	data/
    32	# ...but scripts/data/ is a version-controlled scripts bucket, not runtime data.
    33	!scripts/data/
    34	!scripts/data/**
    35	wandb/
packages/core/legoesm/parallel/sharded_dynamics.py-1240-        halo_cells_set.update(new_cells.tolist())
packages/core/legoesm/parallel/sharded_dynamics.py-1241-
packages/core/legoesm/parallel/sharded_dynamics.py-1242-
packages/core/legoesm/parallel/sharded_dynamics.py-1243-#: Halo depth the SPMD Voronoi partition infra is built at.  ONE definition
packages/core/legoesm/parallel/sharded_dynamics.py-1244-#: consumed by both the production step factory and ``spmd_schedule_cost``:
packages/core/legoesm/parallel/sharded_dynamics.py-1245-#: a score computed at a different depth describes a different comm graph, and
packages/core/legoesm/parallel/sharded_dynamics.py-1246-#: two independently hardcoded 3s let production drift unnoticed.
packages/core/legoesm/parallel/sharded_dynamics.py-1247-SPMD_HALO_DEPTH = 3
packages/core/legoesm/parallel/sharded_dynamics.py-1248-
packages/core/legoesm/parallel/sharded_dynamics.py-1249-
packages/core/legoesm/parallel/sharded_dynamics.py:1250:def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
packages/core/legoesm/parallel/sharded_dynamics.py-1251-                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
packages/core/legoesm/parallel/sharded_dynamics.py-1252-                       ppermute_cells_per_device_threshold=2_000):
packages/core/legoesm/parallel/sharded_dynamics.py-1253-    """How much halo communication one ownership choice costs, computed offline.
packages/core/legoesm/parallel/sharded_dynamics.py-1254-
packages/core/legoesm/parallel/sharded_dynamics.py-1255-    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
packages/core/legoesm/parallel/sharded_dynamics.py-1256-    ROUNDS one halo exchange needs -- the sequential collective launches that
packages/core/legoesm/parallel/sharded_dynamics.py-1257-    dominate MPAS strong scaling above ~64 devices.  Runs on a laptop: no GPU,
packages/core/legoesm/parallel/sharded_dynamics.py-1258-    no MPI, no benchmark job, so a split can be compared before it costs an
packages/core/legoesm/parallel/sharded_dynamics.py-1259-    allocation.
packages/core/legoesm/parallel/sharded_dynamics.py-1260-
packages/core/legoesm/parallel/sharded_dynamics.py-1261-    It calls the SAME builders production calls
packages/core/legoesm/parallel/sharded_dynamics.py-1262-    (:func:`_build_voronoi_partition_infra` then
packages/core/legoesm/parallel/sharded_dynamics.py-1263-    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
packages/core/legoesm/parallel/sharded_dynamics.py-1264-    different question: a 1-ring ``cellsOnEdge`` adjacency graph reports 8
packages/core/legoesm/parallel/sharded_dynamics.py-1265-    rounds where the real depth-3-plus-closure graph reports 12-14.
packages/core/legoesm/parallel/sharded_dynamics.py-1266-
packages/core/legoesm/parallel/sharded_dynamics.py-1267-    WHAT THE NUMBER IS NOT
packages/core/legoesm/parallel/sharded_dynamics.py-1268-    ----------------------
packages/core/legoesm/parallel/sharded_dynamics.py-1269-    * ``n_rounds`` is per HALO FILL, not per model step.  A step costs
packages/core/legoesm/parallel/sharded_dynamics.py-1270-      ``n_rounds`` x (tendency evaluations per step), which depends on the
packages/core/legoesm/parallel/sharded_dynamics.py-1271-      configured integrator -- SSP-RK3 evaluates 3 times, but the MPAS default
packages/core/legoesm/parallel/sharded_dynamics.py-1272-      is ``ssp_rk54_scan``.  Multiply with the integrator you actually run.
packages/core/legoesm/parallel/sharded_dynamics.py-1273-    * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
packages/core/legoesm/parallel/sharded_dynamics.py-1274-      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
packages/core/legoesm/parallel/sharded_dynamics.py-1275-      keeps the best; equality is MEASURED (compare the returned
packages/core/legoesm/parallel/sharded_dynamics.py-1276-      ``max_degree``), never assumed.  Do not claim "the colouring is already
packages/core/legoesm/parallel/sharded_dynamics.py-1277-      optimal so only ownership can help" from this function.
packages/core/legoesm/parallel/sharded_dynamics.py-1278-    * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
packages/core/legoesm/parallel/sharded_dynamics.py-1279-      cells/device is below ``ppermute_cells_per_device_threshold``, in which
packages/core/legoesm/parallel/sharded_dynamics.py-1280-      case there is no ppermute schedule and this number is counterfactual --
packages/core/legoesm/parallel/sharded_dynamics.py-1281-      see the returned ``production_strategy``.
packages/core/legoesm/parallel/sharded_dynamics.py-1282-
packages/core/legoesm/parallel/sharded_dynamics.py-1283-    MESH STATE -- the one thing that silently invalidates the score
packages/core/legoesm/parallel/sharded_dynamics.py-1284-    --------------------------------------------------------------
packages/core/legoesm/parallel/sharded_dynamics.py-1285-    Production does NOT reorder inside ``make_voronoi_sharded_step``; it
packages/core/legoesm/parallel/sharded_dynamics.py-1286-    consumes an already-reordered ``model.mesh``.  The scaling bench reorders
packages/core/legoesm/parallel/sharded_dynamics.py-1287-    ONCE for a ``reorder_target`` device count and then runs at a possibly
packages/core/legoesm/parallel/sharded_dynamics.py-1288-    DIFFERENT device count.  So pass what you actually have:
packages/core/legoesm/parallel/sharded_dynamics.py-1289-
packages/core/legoesm/parallel/sharded_dynamics.py-1290-    * raw mesh, scoring a run at ``n_dev``: defaults are right.
packages/core/legoesm/parallel/sharded_dynamics.py-1291-    * raw mesh, but the run reorders for a different target: pass
packages/core/legoesm/parallel/sharded_dynamics.py-1292-      ``reorder_target=<that target>``; the split is built for the target and
packages/core/legoesm/parallel/sharded_dynamics.py-1293-      scored at ``n_dev``.
packages/core/legoesm/parallel/sharded_dynamics.py-1294-    * already-reordered mesh (what production holds): pass
packages/core/legoesm/parallel/sharded_dynamics.py-1295-      ``already_reordered=True``; ``method`` is then ignored and reported as
packages/core/legoesm/parallel/sharded_dynamics.py-1296-      ``"pre-reordered"``, because the ownership is already baked in.
packages/core/legoesm/parallel/sharded_dynamics.py-1297-
packages/core/legoesm/parallel/sharded_dynamics.py-1298-    Parameters
packages/core/legoesm/parallel/sharded_dynamics.py-1299-    ----------
packages/core/legoesm/parallel/sharded_dynamics.py-1300-    mesh : VoronoiMesh
packages/core/legoesm/parallel/sharded_dynamics.py-1301-    n_dev : int
packages/core/legoesm/parallel/sharded_dynamics.py-1302-        Device count the run uses.  Must be >= 1.
packages/core/legoesm/parallel/sharded_dynamics.py-1303-    method : str
packages/core/legoesm/parallel/sharded_dynamics.py-1304-        Ownership for the reorder; ignored when *already_reordered*.
packages/core/legoesm/parallel/sharded_dynamics.py-1305-    reorder_target : int | None
packages/core/legoesm/parallel/sharded_dynamics.py-1306-        Device count the reorder targets, when it differs from *n_dev*.
packages/core/legoesm/parallel/sharded_dynamics.py-1307-    already_reordered : bool
packages/core/legoesm/parallel/sharded_dynamics.py-1308-    halo_depth : int
packages/core/legoesm/parallel/sharded_dynamics.py-1309-        Must match production (3) or the graph is a different graph.
packages/core/legoesm/parallel/sharded_dynamics.py-1310-    ppermute_cells_per_device_threshold : int
packages/core/legoesm/parallel/sharded_dynamics.py-1311-        Mirror of the production auto-select threshold, only used to report
packages/core/legoesm/parallel/sharded_dynamics.py-1312-        ``production_strategy``.
packages/core/legoesm/parallel/sharded_dynamics.py-1313-
packages/core/legoesm/parallel/sharded_dynamics.py-1314-    Returns
packages/core/legoesm/parallel/sharded_dynamics.py-1315-    -------
packages/core/legoesm/parallel/sharded_dynamics.py-1316-    dict
packages/core/legoesm/parallel/sharded_dynamics.py-1317-        ``n_rounds`` (the cost), ``max_degree`` (the lower bound to compare
packages/core/legoesm/parallel/sharded_dynamics.py-1318-        it against), ``n_rounds_greedy``, ``coloring_method``,
packages/core/legoesm/parallel/sharded_dynamics.py-1319-        ``resolved_method`` (concrete, never ``"auto"``),
packages/core/legoesm/parallel/sharded_dynamics.py-1320-        ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
packages/core/legoesm/parallel/sharded_dynamics.py-1321-        ``max_local_cells``, ``max_local_edges``, and the echoed inputs.
packages/core/legoesm/parallel/sharded_dynamics.py-1322-
packages/core/legoesm/parallel/sharded_dynamics.py-1323-    Reference census on the unrelaxed mesh, which any change here must still
packages/core/legoesm/parallel/sharded_dynamics.py-1324-    reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
packages/core/legoesm/parallel/sharded_dynamics.py-1325-    geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
packages/core/legoesm/parallel/sharded_dynamics.py-1326-    """
packages/core/legoesm/parallel/sharded_dynamics.py-1327-    from legoesm.parallel.voronoi_partition import (
packages/core/legoesm/parallel/sharded_dynamics.py-1328-        reorder_voronoi_for_sharding, resolve_sharding_partition_method,
packages/core/legoesm/parallel/sharded_dynamics.py-1329-    )
packages/core/legoesm/parallel/sharded_dynamics.py-1330-
packages/core/legoesm/parallel/sharded_dynamics.py-1331-    if int(n_dev) != n_dev or int(n_dev) < 1:
packages/core/legoesm/parallel/sharded_dynamics.py-1332-        # int() would silently truncate 3.9 -> 3 and score the wrong split.
packages/core/legoesm/parallel/sharded_dynamics.py-1333-        raise ValueError(
packages/core/legoesm/parallel/sharded_dynamics.py-1334-            f"spmd_schedule_cost: n_dev must be an integer >= 1, got {n_dev!r}")
packages/core/legoesm/parallel/sharded_dynamics.py-1335-    n_dev = int(n_dev)
packages/core/legoesm/parallel/sharded_dynamics.py-1336-
packages/core/legoesm/parallel/sharded_dynamics.py-1337-    if already_reordered:
packages/core/legoesm/parallel/sharded_dynamics.py-1338-        if reorder_target is not None:
packages/core/legoesm/parallel/sharded_dynamics.py-1339-            raise ValueError(
packages/core/legoesm/parallel/sharded_dynamics.py-1340-                "spmd_schedule_cost: reorder_target is meaningless with "
packages/core/legoesm/parallel/sharded_dynamics.py-1341-                "already_reordered=True — the ownership is already baked into "
packages/core/legoesm/parallel/sharded_dynamics.py-1342-                "the mesh.")
packages/core/legoesm/parallel/sharded_dynamics.py-1343-        prepared, resolved = mesh, "pre-reordered"
packages/core/legoesm/parallel/sharded_dynamics.py-1344-    else:
packages/core/legoesm/parallel/sharded_dynamics.py-1345-        target = n_dev if reorder_target is None else int(reorder_target)
packages/core/legoesm/parallel/sharded_dynamics.py-1346-        prepared = reorder_voronoi_for_sharding(mesh, target, method=method)
packages/core/legoesm/parallel/sharded_dynamics.py-1347-        # Report the CONCRETE ownership: "auto" hides which partitioner ran.
packages/core/legoesm/parallel/sharded_dynamics.py-1348-        # Uses the SAME resolver the reorder used, so the label cannot drift
packages/core/legoesm/parallel/sharded_dynamics.py-1349-        # from the policy.
packages/core/legoesm/parallel/sharded_dynamics.py-1350-        resolved = resolve_sharding_partition_method(method)
packages/core/legoesm/parallel/sharded_dynamics.py-1351-
packages/core/legoesm/parallel/sharded_dynamics.py-1352-    # The builder assigns residual entities to the LAST owner but excludes them
packages/core/legoesm/parallel/sharded_dynamics.py-1353-    # from every owned contiguous block, so schedule send indices can exceed a
packages/core/legoesm/parallel/sharded_dynamics.py-1354-    # device's shard length -- a number that looks fine and is not.  Reachable
packages/core/legoesm/parallel/sharded_dynamics.py-1355-    # via reorder_target: a mesh padded for 3 devices is not divisible by 4.
packages/core/legoesm/parallel/sharded_dynamics.py-1356-    # The scaling bench rejects that pairing; so does this.
packages/core/legoesm/parallel/sharded_dynamics.py-1357-    n_cells, n_edges = int(prepared.nCells), int(prepared.nEdges)
packages/core/legoesm/parallel/sharded_dynamics.py-1358-    if n_cells % n_dev or n_edges % n_dev:
packages/core/legoesm/parallel/sharded_dynamics.py-1359-        raise ValueError(
packages/core/legoesm/parallel/sharded_dynamics.py-1360-            f"spmd_schedule_cost: prepared mesh has nCells={n_cells}, "
packages/core/legoesm/parallel/sharded_dynamics.py-1361-            f"nEdges={n_edges}, neither divisible by n_dev={n_dev}. The mesh "
packages/core/legoesm/parallel/sharded_dynamics.py-1362-            f"is padded for its reorder target"
packages/core/legoesm/parallel/sharded_dynamics.py-1363-            f"{'' if already_reordered else f' ({target})'}, so scoring it at "
packages/core/legoesm/parallel/sharded_dynamics.py-1364-            f"a device count that does not divide it silently mis-slices the "
packages/core/legoesm/parallel/sharded_dynamics.py-1365-            f"owned blocks. Score at a device count that divides the prepared "
packages/core/legoesm/parallel/sharded_dynamics.py-1366-            f"mesh.")
packages/core/legoesm/parallel/sharded_dynamics.py-1367-    (
packages/core/legoesm/parallel/sharded_dynamics.py-1368-        _stacked, _gc, _ge, _noc, _noe, max_lc, max_le, partitions, cell_owner,
packages/core/legoesm/parallel/sharded_dynamics.py-1369-    ) = _build_voronoi_partition_infra(prepared, n_dev, halo_depth=halo_depth)
packages/core/legoesm/parallel/sharded_dynamics.py-1370-    cells_per = n_cells // n_dev
packages/core/legoesm/parallel/sharded_dynamics.py-1371-    edges_per = n_edges // n_dev
packages/core/legoesm/parallel/sharded_dynamics.py-1372-    sched = _build_ppermute_schedule(
packages/core/legoesm/parallel/sharded_dynamics.py-1373-        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
packages/core/legoesm/parallel/sharded_dynamics.py-1374-    )
packages/core/legoesm/parallel/sharded_dynamics.py-1375-    return {
packages/core/legoesm/parallel/sharded_dynamics.py-1376-        "method": method,
packages/core/legoesm/parallel/sharded_dynamics.py-1377-        "resolved_method": resolved,
packages/core/legoesm/parallel/sharded_dynamics.py-1378-        "n_dev": n_dev,
packages/core/legoesm/parallel/sharded_dynamics.py-1379-        # Unknown for a pre-reordered mesh: the ownership is baked in and the
packages/core/legoesm/parallel/sharded_dynamics.py-1380-        # target that produced it is not recoverable from the mesh. Reporting
packages/core/legoesm/parallel/sharded_dynamics.py-1381-        # n_dev there would assert something we did not verify.
packages/core/legoesm/parallel/sharded_dynamics.py-1382-        "reorder_target": (None if already_reordered else
packages/core/legoesm/parallel/sharded_dynamics.py-1383-                           (n_dev if reorder_target is None
packages/core/legoesm/parallel/sharded_dynamics.py-1384-                            else int(reorder_target))),
packages/core/legoesm/parallel/sharded_dynamics.py-1385-        "already_reordered": bool(already_reordered),
packages/core/legoesm/parallel/sharded_dynamics.py-1386-        "halo_depth": halo_depth,
packages/core/legoesm/parallel/sharded_dynamics.py-1387-        "n_rounds": int(sched["n_rounds"]),
packages/core/legoesm/parallel/sharded_dynamics.py-1388-        "n_rounds_greedy": int(sched["n_rounds_greedy"]),
packages/core/legoesm/parallel/sharded_dynamics.py-1389-        "max_degree": int(sched.get("max_degree", -1)),
packages/core/legoesm/parallel/sharded_dynamics.py-1390-        "coloring_method": sched["coloring_method"],
packages/core/legoesm/parallel/sharded_dynamics.py-1391-        # Production returns before selecting a strategy at n_dev==1, and a
packages/core/legoesm/parallel/sharded_dynamics.py-1392-        # caller may force halo_strategy; this reports what AUTO would pick.
packages/core/legoesm/parallel/sharded_dynamics.py-1393-        "production_strategy": (
packages/core/legoesm/parallel/sharded_dynamics.py-1394-            None if n_dev == 1 else
packages/core/legoesm/parallel/sharded_dynamics.py-1395-            ("allgather" if cells_per < ppermute_cells_per_device_threshold
packages/core/legoesm/parallel/sharded_dynamics.py-1396-             else "ppermute")),
packages/core/legoesm/parallel/sharded_dynamics.py-1397-        "cells_per_device": cells_per,
packages/core/legoesm/parallel/sharded_dynamics.py-1398-        "max_local_cells": int(max_lc),
packages/core/legoesm/parallel/sharded_dynamics.py-1399-        "max_local_edges": int(max_le),
packages/core/legoesm/parallel/sharded_dynamics.py-1400-    }
packages/core/legoesm/parallel/sharded_dynamics.py-1401-
packages/core/legoesm/parallel/sharded_dynamics.py-1402-
packages/core/legoesm/parallel/sharded_dynamics.py-1403-def _build_voronoi_partition_infra(global_mesh, n_dev, halo_depth=2):
packages/core/legoesm/parallel/sharded_dynamics.py-1404-    """Pre-compute per-device local meshes and gather/scatter indices.
packages/core/legoesm/parallel/sharded_dynamics.py-1405-
packages/core/legoesm/parallel/sharded_dynamics.py-1406-    After ``reorder_voronoi_for_sharding`` the global mesh has *both*
packages/core/legoesm/parallel/sharded_dynamics.py-1407-    cells and edges ordered by contiguous device blocks.  We compute
packages/core/legoesm/parallel/sharded_dynamics.py-1408-    partitions whose owned-entity boundaries exactly match the shard
packages/core/legoesm/parallel/sharded_dynamics.py-1409-    boundaries (``cells_per = nCells // n_dev``, ``edges_per = nEdges //
packages/core/legoesm/parallel/sharded_dynamics.py-1410-    n_dev``), then build local meshes with remapped connectivity.
packages/core/legoesm/parallel/sharded_dynamics.py-1411-
packages/core/legoesm/parallel/sharded_dynamics.py-1412-    Using ``partition_voronoi_mesh`` directly is unsuitable because it
packages/core/legoesm/parallel/sharded_dynamics.py-1413-    derives edge ownership from cell ownership, producing an uneven edge
packages/core/legoesm/parallel/sharded_dynamics.py-1414-    split that mismatches the even shard split.  Instead we construct the
packages/core/legoesm/parallel/sharded_dynamics.py-1415-    :class:`VoronoiPartition` objects manually with contiguous-block
packages/core/legoesm/parallel/sharded_dynamics.py-1416-    ownership for both cells **and** edges.
packages/core/legoesm/parallel/sharded_dynamics.py-1417-
packages/core/legoesm/parallel/sharded_dynamics.py-1418-    Returns
packages/core/legoesm/parallel/sharded_dynamics.py-1419-    -------
packages/core/legoesm/parallel/sharded_dynamics.py-1420-    stacked_meshes : VoronoiMesh
packages/core/legoesm/parallel/sharded_dynamics.py-1421-        Each leaf has shape ``(n_dev, max_local_*)``.
packages/core/legoesm/parallel/sharded_dynamics.py-1422-    gather_cells : jnp.ndarray, (n_dev, max_local_cells)
packages/core/legoesm/parallel/sharded_dynamics.py-1423-    gather_edges : jnp.ndarray, (n_dev, max_local_edges)
packages/core/legoesm/parallel/sharded_dynamics.py-1424-    n_owned_cells : list[int]
packages/core/legoesm/parallel/sharded_dynamics.py-1425-    n_owned_edges : list[int]
packages/core/legoesm/parallel/sharded_dynamics.py-1426-    max_lc : int
packages/core/legoesm/parallel/sharded_dynamics.py-1427-    max_le : int
packages/core/legoesm/parallel/sharded_dynamics.py-1428-    partitions : list[VoronoiPartition]
packages/core/legoesm/parallel/sharded_dynamics.py-1429-        Per-device partition descriptors (for ppermute schedule building).
packages/core/legoesm/parallel/sharded_dynamics.py-1430-    cell_owner : np.ndarray, (nCells,)
packages/core/legoesm/parallel/sharded_dynamics.py-1431-        Cell ownership array.
packages/core/legoesm/parallel/sharded_dynamics.py-1432-    """
packages/core/legoesm/parallel/sharded_dynamics.py-1433-    import numpy as np
packages/core/legoesm/parallel/sharded_dynamics.py-1434-    from legoesm.parallel.voronoi_partition import (
packages/core/legoesm/parallel/sharded_dynamics.py-1435-        HaloCommSchedule,
packages/core/legoesm/parallel/sharded_dynamics.py-1436-        VoronoiPartition,
packages/core/legoesm/parallel/sharded_dynamics.py-1437-        build_local_mesh,
packages/core/legoesm/parallel/sharded_dynamics.py-1438-        compute_halo_cells,
packages/core/legoesm/parallel/sharded_dynamics.py-1439-    )
packages/core/legoesm/parallel/sharded_dynamics.py-1440-
packages/core/legoesm/parallel/sharded_dynamics.py-1441-    nCells = global_mesh.nCells
packages/core/legoesm/parallel/sharded_dynamics.py-1442-    nEdges = global_mesh.nEdges
packages/core/legoesm/parallel/sharded_dynamics.py-1443-    nVertices = global_mesh.nVertices
packages/core/legoesm/parallel/sharded_dynamics.py-1444-    cells_per = nCells // n_dev
packages/core/legoesm/parallel/sharded_dynamics.py-1445-    edges_per = nEdges // n_dev
packages/core/legoesm/parallel/sharded_dynamics.py-1446-    verts_per = nVertices // n_dev
packages/core/legoesm/parallel/sharded_dynamics.py-1447-
packages/core/legoesm/parallel/sharded_dynamics.py-1448-    # Convert mesh arrays to numpy for setup.
packages/core/legoesm/parallel/sharded_dynamics.py-1449-    gm_np = jax.tree.map(
packages/core/legoesm/parallel/sharded_dynamics.py-1450-        lambda x: np.asarray(x) if hasattr(x, "shape") else x,
packages/core/legoesm/parallel/sharded_dynamics.py-1451-        global_mesh,
packages/core/legoesm/parallel/sharded_dynamics.py-1452-    )
packages/core/legoesm/parallel/sharded_dynamics.py-1453-    cellsOnCell_np = np.asarray(gm_np.cellsOnCell)       # (maxEdges, nCells)
packages/core/legoesm/parallel/sharded_dynamics.py-1454-    cellsOnEdge_np = np.asarray(gm_np.cellsOnEdge)       # (2, nEdges)
packages/core/legoesm/parallel/sharded_dynamics.py-1455-    np.asarray(gm_np.cellsOnVertex)    # (vDeg, nVerts)
packages/core/legoesm/parallel/sharded_dynamics.py-1456-    verticesOnCell_np = np.asarray(gm_np.verticesOnCell)   # (maxEdges, nCells)
packages/core/legoesm/parallel/sharded_dynamics.py-1457-    verticesOnEdge_np = np.asarray(gm_np.verticesOnEdge)   # (2, nEdges)
packages/core/legoesm/parallel/sharded_dynamics.py-1458-    maxEdges = int(gm_np.maxEdges)
packages/core/legoesm/parallel/sharded_dynamics.py-1459-    int(gm_np.vertexDegree)
packages/core/legoesm/parallel/sharded_dynamics.py-1460-

exec
/usr/bin/bash -lc "git diff --unified=3 -- tests/bench/test_bench_voronoi_partition_methods.py | sed -n '1,420p'; git diff --unified=3 -- scripts/bench/bench_voronoi_partition_methods.py | sed -n '1,420p'; git diff --check; bash -n scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch; python - <<'PY'
import ast
from pathlib import Path
for path in [
    'scripts/bench/bench_voronoi_partition_methods.py',
    'tests/bench/test_bench_voronoi_partition_methods.py',
    'tests/bench/test_mpas_schedule_cost_scan_sbatch.py',
]:
    ast.parse(Path(path).read_text(), filename=path)
    print(f'AST OK: {path}')
PY" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 1 in 526ms:
diff --git a/tests/bench/test_bench_voronoi_partition_methods.py b/tests/bench/test_bench_voronoi_partition_methods.py
index 21f861fd7..66a433f06 100644
--- a/tests/bench/test_bench_voronoi_partition_methods.py
+++ b/tests/bench/test_bench_voronoi_partition_methods.py
@@ -146,3 +146,312 @@ def test_halo_matches_runtime_partition():
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
+
+
+def test_empty_methods_refused_instead_of_measuring_nothing(monkeypatch):
+    """``--methods " , , "`` used to select nothing, write ``rows: []`` and
+    exit 0 — an empty run that reads as a successful one (codex round 4)."""
+    monkeypatch.setattr(sys, "argv", [
+        "bench", "--subdivision", "2", "--rank-counts", "2",
+        "--methods", " , , ", "--out", "/dev/null"])
+    with pytest.raises(SystemExit, match="selects NO methods"):
+        mod.main()
diff --git a/scripts/bench/bench_voronoi_partition_methods.py b/scripts/bench/bench_voronoi_partition_methods.py
index 6aa0f4a40..e7355ed5a 100644
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
@@ -149,12 +370,46 @@ def main() -> int:
             raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
     if not rank_counts or any(n < 2 for n in rank_counts):
         raise SystemExit("--rank-counts needs integers >= 2")
+    if not methods:
+        # Same class as an empty --expect-rounds: the loop would be skipped,
+        # `rows: []` written, and 0 returned — an empty run that reads as a
+        # successful one.
+        raise SystemExit(
+            f"--methods={args.methods!r} selects NO methods; the benchmark "
+            f"would measure nothing and still exit 0. Choose from {METHODS}.")
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
@@ -185,6 +440,33 @@ def main() -> int:
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
+
+    # An "empty but successful" run is the same failure class as a skipped
+    # gate: --methods metis on a box without pymetis appends only an
+    # unavailable row and would otherwise exit 0 having measured nothing
+    # (codex round 5).
+    if not any(r.get("available") and "n_ranks" in r for r in rows):
+        raise SystemExit(
+            f"no method was actually measured (requested {methods}); every "
+            f"one was unavailable, so this run has no results and must not "
+            f"report success.")
 
     payload = {
         "rows": rows,
@@ -204,7 +486,9 @@ def main() -> int:
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
@@ -213,6 +497,24 @@ def main() -> int:
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
 
 
/usr/bin/bash: cannot create temp file for here-document: Read-only file system

exec
/usr/bin/bash -lc "python -c \"import ast; from pathlib import Path; paths=('scripts/bench/bench_voronoi_partition_methods.py','tests/bench/test_bench_voronoi_partition_methods.py','tests/bench/test_mpas_schedule_cost_scan_sbatch.py'); [ast.parse(Path(p).read_text(), filename=p) for p in paths]; print('AST OK:', ', '.join(paths))\"" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 127 in 301ms:
/usr/bin/bash: python: command not found

exec
/usr/bin/bash -lc 'for p in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python /usr/bin/python3 /bin/python3; do if [ -x "$p" ]; then "$p" -c "import ast; from pathlib import Path; paths=('"'scripts/bench/bench_voronoi_partition_methods.py','tests/bench/test_bench_voronoi_partition_methods.py','tests/bench/test_mpas_schedule_cost_scan_sbatch.py'); [ast.parse(Path(x).read_text(), filename=x) for x in paths]; print('AST OK')\"; exit "'$?; fi; done; echo '"'No Python interpreter available'; exit 127" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 411ms:
AST OK

exec
/usr/bin/bash -lc "if printf '%s\\n' '{\"rows\": [], \"note\": \"n_ranks\"}' | grep -q '\"n_ranks\"'; then echo 'grep predicate ACCEPTS empty rows'; else echo 'grep predicate rejects empty rows'; fi
PYTHONDONTWRITEBYTECODE=1 /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python -c \"import importlib.util; from pathlib import Path; p=Path('scripts/bench/bench_voronoi_partition_methods.py'); s=importlib.util.spec_from_file_location('b',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); x=m.parse_expect_rounds('sfc:64=999,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21'); print(sorted(x)); print(set(x)=={(k,n) for k in ('geometric','sfc','metis') for n in (64,128)})\"" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 653ms:
grep predicate ACCEPTS empty rows
[('geometric', 64), ('geometric', 128), ('metis', 64), ('metis', 128), ('sfc', 64), ('sfc', 128)]
True

 succeeded in 199779ms:
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
./.venv-mpi/lib/python3.13/site-packages/scipy/spatial/_spherical_voronoi.py
./.venv-mpi/lib/python3.13/site-packages/scipy/spatial/tests/test_spherical_voronoi.py
./tests/atmosphere/test_column_large_scale_extract_voronoi.py
./tests/unit/test_sharded_dynamics.py
./tests/unit/test_scale_sharded_dynamics.py
./tests/unit/test_regional_voronoi.py
./.venv/lib/python3.14/site-packages/scipy/spatial/_spherical_voronoi.py
./.venv/lib/python3.14/site-packages/scipy/spatial/tests/test_spherical_voronoi.py
./.venv-gpumpi/lib/python3.14/site-packages/scipy/spatial/_spherical_voronoi.py
./.venv-gpumpi/lib/python3.14/site-packages/scipy/spatial/tests/test_spherical_voronoi.py
./packages/core/legoesm/grids/voronoi.py
./packages/core/legoesm/core/operators_voronoi.py
./packages/core/legoesm/parallel/halo_exchange_voronoi.py
./packages/core/legoesm/parallel/sharded_dynamics.py
./.claude/worktrees/levante-ocean/tests/atmosphere/test_column_large_scale_extract_voronoi.py
./.claude/worktrees/levante-ocean/tests/unit/test_sharded_dynamics.py
./.claude/worktrees/levante-ocean/tests/unit/test_scale_sharded_dynamics.py
./.claude/worktrees/levante-ocean/tests/unit/test_regional_voronoi.py
./.claude/worktrees/levante-ocean/.venv/lib/python3.14/site-packages/scipy/spatial/_spherical_voronoi.py
./.claude/worktrees/levante-ocean/.venv/lib/python3.14/site-packages/scipy/spatial/tests/test_spherical_voronoi.py
./.claude/worktrees/levante-ocean/packages/core/legoesm/grids/voronoi.py
./.claude/worktrees/levante-ocean/packages/core/legoesm/core/operators_voronoi.py
./.claude/worktrees/levante-ocean/packages/core/legoesm/parallel/halo_exchange_voronoi.py
./.claude/worktrees/levante-ocean/packages/core/legoesm/parallel/sharded_dynamics.py
./.claude/worktrees/scaling-campaign/tests/atmosphere/test_column_large_scale_extract_voronoi.py
./.claude/worktrees/scaling-campaign/tests/unit/test_sharded_dynamics.py
./.claude/worktrees/scaling-campaign/tests/unit/test_scale_sharded_dynamics.py
./.claude/worktrees/scaling-campaign/tests/unit/test_regional_voronoi.py
./.claude/worktrees/scaling-campaign/packages/core/legoesm/grids/voronoi.py
./.claude/worktrees/scaling-campaign/packages/core/legoesm/core/operators_voronoi.py
./.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/halo_exchange_voronoi.py
./.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/sharded_dynamics.py
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
./packages/core/legoesm/grids/voronoi.py-1228-
./packages/core/legoesm/grids/voronoi.py-1229-    Call ONCE from a single process before an MPI launch so every rank loads the
./packages/core/legoesm/grids/voronoi.py-1230-    mesh instead of redundantly rebuilding it.  Cheap no-op when already cached
./packages/core/legoesm/grids/voronoi.py-1231-    (create_voronoi_mesh returns the cached mesh without rebuilding).
./packages/core/legoesm/grids/voronoi.py-1232-    """
./packages/core/legoesm/grids/voronoi.py-1233-    if _voronoi_cache_disabled():
./packages/core/legoesm/grids/voronoi.py-1234-        raise RuntimeError(
./packages/core/legoesm/grids/voronoi.py-1235-            f"{_MESH_CACHE_DISABLE_ENV} is set; cannot pre-warm a disabled cache.")
./packages/core/legoesm/grids/voronoi.py-1236-    path = _voronoi_cache_path(subdivision_level, radius, lloyd_iterations, omega)
./packages/core/legoesm/grids/voronoi.py-1237-    create_voronoi_mesh(
./packages/core/legoesm/grids/voronoi.py-1238-        subdivision_level, radius=radius,
./packages/core/legoesm/grids/voronoi.py-1239-        lloyd_iterations=lloyd_iterations, omega=omega)
./packages/core/legoesm/grids/voronoi.py-1240-    return path
./packages/core/legoesm/grids/voronoi.py-1241-
./packages/core/legoesm/grids/voronoi.py-1242-
./packages/core/legoesm/grids/voronoi.py:1243:def create_voronoi_mesh(
./packages/core/legoesm/grids/voronoi.py-1244-    subdivision_level: int,
./packages/core/legoesm/grids/voronoi.py-1245-    radius: float = constants.R_earth,
./packages/core/legoesm/grids/voronoi.py-1246-    lloyd_iterations: int = 50,
./packages/core/legoesm/grids/voronoi.py-1247-    omega: float = constants.Omega,
./packages/core/legoesm/grids/voronoi.py-1248-    density_fn=None,
./packages/core/legoesm/grids/voronoi.py-1249-) -> VoronoiMesh:
./packages/core/legoesm/grids/voronoi.py-1250-    """Create a centroidal Voronoi tessellation (SCVT) on the sphere.
./packages/core/legoesm/grids/voronoi.py-1251-
./packages/core/legoesm/grids/voronoi.py-1252-    Starts from an icosahedral triangulation, bisects to the desired
./packages/core/legoesm/grids/voronoi.py-1253-    level, applies Lloyd relaxation, then builds the full MPAS-compatible
./packages/core/legoesm/grids/voronoi.py-1254-    mesh with all connectivity and geometric arrays.
./packages/core/legoesm/grids/voronoi.py-1255-
./packages/core/legoesm/grids/voronoi.py-1256-    Parameters
./packages/core/legoesm/grids/voronoi.py-1257-    ----------
./packages/core/legoesm/grids/voronoi.py-1258-    subdivision_level : int
./packages/core/legoesm/grids/voronoi.py-1259-        Number of bisection levels. nCells = 10*4^level + 2.
./packages/core/legoesm/grids/voronoi.py-1260-        level=3: 642 cells, level=4: 2562, level=5: 10242.
./packages/core/legoesm/grids/voronoi.py-1261-    radius : float
./packages/core/legoesm/grids/voronoi.py-1262-        Sphere radius [m]. Default: Earth radius.
./packages/core/legoesm/grids/voronoi.py-1263-    lloyd_iterations : int
./packages/core/legoesm/grids/voronoi.py-1264-        Number of Lloyd relaxation iterations. Default: 50.
./packages/core/legoesm/grids/voronoi.py-1265-    omega : float
./packages/core/legoesm/grids/voronoi.py-1266-        Rotation rate [rad/s]. Default: Earth rotation.
./packages/core/legoesm/grids/voronoi.py-1267-    density_fn : callable(lat, lon) -> float, optional
./packages/core/legoesm/grids/voronoi.py-1268-        Relative mesh-density function (``lat``, ``lon`` in radians; larger =>
./packages/core/legoesm/grids/voronoi.py-1269-        finer cells).  When provided, the Lloyd relaxation is **density-weighted**
./packages/core/legoesm/grids/voronoi.py-1270-        (Du–Faber–Gunzburger SCVT): generators concentrate where the density is
./packages/core/legoesm/grids/voronoi.py-1271-        high, yielding a VARIABLE-RESOLUTION mesh refined over the high-density
./packages/core/legoesm/grids/voronoi.py-1272-        region (the MPAS variable-resolution capability).  Cell area relaxes toward
./packages/core/legoesm/grids/voronoi.py-1273-        ``~ 1/density`` (weighting by ``density**2``), so a region with relative
./packages/core/legoesm/grids/voronoi.py-1274-        density ``d`` gets roughly ``d`` x smaller cells.  ``None`` (default) =>
./packages/core/legoesm/grids/voronoi.py-1275-        the uniform quasi-uniform SCVT (unchanged).  Host-side mesh generation —
./packages/core/legoesm/grids/voronoi.py-1276-        the callable is plain NumPy, never traced.  NOTE: Lloyd relaxation converges
./packages/core/legoesm/grids/voronoi.py-1277-        linearly, so the achieved refinement contrast grows with
./packages/core/legoesm/grids/voronoi.py-1278-        ``lloyd_iterations``; for strong/precise variable resolution prefer a
./packages/core/legoesm/grids/voronoi.py-1279-        JIGSAW-built mesh loaded via :func:`load_mpas_mesh`.
./packages/core/legoesm/grids/voronoi.py-1280-
./packages/core/legoesm/grids/voronoi.py-1281-    Returns
./packages/core/legoesm/grids/voronoi.py-1282-    -------
./packages/core/legoesm/grids/voronoi.py-1283-    VoronoiMesh
./packages/core/legoesm/grids/voronoi.py-1284-    """
./packages/core/legoesm/grids/voronoi.py-1285-    if subdivision_level > _BIG_MESH_MAX_LEVEL:
./packages/core/legoesm/grids/voronoi.py-1286-        n_cells = 10 * 4 ** subdivision_level + 2
./packages/core/legoesm/grids/voronoi.py-1287-        raise ValueError(
./packages/core/legoesm/grids/voronoi.py-1288-            f"subdivision_level={subdivision_level} would create {n_cells:.2e} "
./packages/core/legoesm/grids/voronoi.py-1289-            f"cells — unsupported (hard cap {_BIG_MESH_MAX_LEVEL}; the scipy "
./packages/core/legoesm/grids/voronoi.py-1290-            f"SphericalVoronoi build path does not scale there). For higher "
./packages/core/legoesm/grids/voronoi.py-1291-            f"resolutions, use load_mpas_mesh() with a pre-built mesh file."
./packages/core/legoesm/grids/voronoi.py-1292-        )
./packages/core/legoesm/grids/voronoi.py-1293-    # Levels above the routine cap are CACHE-OR-PREWARM only (codex round-19
./packages/core/legoesm/grids/voronoi.py-1294-    # design): a valid cache hit is always admissible; a MISS is refused
./packages/core/legoesm/grids/voronoi.py-1295-    # unless this process is the designated prewarmer — otherwise an N-rank
./packages/core/legoesm/grids/voronoi.py-1296-    # MPI launch would have every rank silently rebuild for hours (build is
./packages/core/legoesm/grids/voronoi.py-1297-    # ~87 s/Lloyd-iteration at subdiv-8, ~4x that at 9). The prewarm path is
./packages/core/legoesm/grids/voronoi.py-1298-    # single-builder: an exclusive lockfile serialises concurrent opt-ins.
./packages/core/legoesm/grids/voronoi.py-1299-    _big = subdivision_level > _BIG_MESH_ROUTINE_LEVEL
./packages/core/legoesm/grids/voronoi.py-1300-
./packages/core/legoesm/grids/voronoi.py-1301-    # Disk cache: a uniform SCVT mesh is deterministic in these args, so skip the
./packages/core/legoesm/grids/voronoi.py-1302-    # expensive rebuild on a hit.  density_fn meshes are NOT cached (a callable
./packages/core/legoesm/grids/voronoi.py-1303-    # has no stable key); the env switch lets a run force a fresh build.
./packages/core/legoesm/grids/voronoi.py-1304-    use_cache = density_fn is None and not _voronoi_cache_disabled()
./packages/core/legoesm/grids/voronoi.py-1305-    cache_path = None
./packages/core/legoesm/grids/voronoi.py-1306-    if use_cache:
./packages/core/legoesm/grids/voronoi.py-1307-        cache_path = _voronoi_cache_path(
./packages/core/legoesm/grids/voronoi.py-1308-            subdivision_level, radius, lloyd_iterations, omega)
./packages/core/legoesm/grids/voronoi.py-1309-        cached = _load_voronoi_cache(cache_path)
./packages/core/legoesm/grids/voronoi.py-1310-        if cached is not None:
./packages/core/legoesm/grids/voronoi.py-1311-            return cached
./packages/core/legoesm/grids/voronoi.py-1312-
./packages/core/legoesm/grids/voronoi.py-1313-    _lock_path = None
./packages/core/legoesm/grids/voronoi.py-1314-    if _big:
./packages/core/legoesm/grids/voronoi.py-1315-        if not use_cache:
./packages/core/legoesm/grids/voronoi.py-1316-            raise ValueError(
./packages/core/legoesm/grids/voronoi.py-1317-                f"subdivision_level={subdivision_level} needs the mesh disk "
./packages/core/legoesm/grids/voronoi.py-1318-                f"cache (density_fn=None and {_MESH_CACHE_DISABLE_ENV} unset)"
./packages/core/legoesm/grids/voronoi.py-1319-                f" — an uncached big-mesh build would repeat per rank.")
./packages/core/legoesm/grids/voronoi.py-1320-        if os.environ.get(_BIG_MESH_BUILD_ENV, "") != "1":
./packages/core/legoesm/grids/voronoi.py-1321-            raise ValueError(
./packages/core/legoesm/grids/voronoi.py-1322-                f"subdivision_level={subdivision_level}: no cached mesh at "
./packages/core/legoesm/grids/voronoi.py-1323-                f"{cache_path} and this process is not the designated "
./packages/core/legoesm/grids/voronoi.py-1324-                f"prewarmer. Build the cache ONCE via scripts/data/"
./packages/core/legoesm/grids/voronoi.py-1325-                f"prewarm_voronoi_mesh.py (or set {_BIG_MESH_BUILD_ENV}=1 in "
./packages/core/legoesm/grids/voronoi.py-1326-                f"a SINGLE-process job), then rerun.")
./packages/core/legoesm/grids/voronoi.py-1327-        # Single-builder lock (codex round-19: the opt-in alone is a
./packages/core/legoesm/grids/voronoi.py-1328-        # thundering herd — N authorized ranks could all miss). O_EXCL
./packages/core/legoesm/grids/voronoi.py-1329-        # lockfile beside the cache; losers wait for the winner's atomic
./packages/core/legoesm/grids/voronoi.py-1330-        # os.replace to land and then load it. Stale locks (builder died)
./packages/core/legoesm/grids/voronoi.py-1331-        # are stolen after _BIG_MESH_LOCK_STALE_S with a warning.
./packages/core/legoesm/grids/voronoi.py-1332-        _lock_path = f"{cache_path}.lock"
./packages/core/legoesm/grids/voronoi.py-1333-        import time as _time
./packages/core/legoesm/grids/voronoi.py-1334-        while True:
./packages/core/legoesm/grids/voronoi.py-1335-            try:
./packages/core/legoesm/grids/voronoi.py-1336-                _fd = os.open(_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
./packages/core/legoesm/grids/voronoi.py-1337-                os.write(_fd, str(os.getpid()).encode())
./packages/core/legoesm/grids/voronoi.py-1338-                os.close(_fd)
./packages/core/legoesm/grids/voronoi.py-1339-                break                      # we are the builder
./packages/core/legoesm/grids/voronoi.py-1340-            except FileExistsError:
./packages/core/legoesm/grids/voronoi.py-1341-                for _ in range(int(_BIG_MESH_LOCK_STALE_S)):
./packages/core/legoesm/grids/voronoi.py-1342-                    _time.sleep(1.0)
./packages/core/legoesm/grids/voronoi.py-1343-                    cached = _load_voronoi_cache(cache_path)

codex
NOT READY.

- The round-5 artifact guard is bypassable. [`run_arm`]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:142) accepts any file containing the literal `"n_ranks"`. Valid empty JSON such as `{"rows": [], "note": "n_ranks"}` passes, returns 0, and can reach `SCAN_DONE`. It neither parses JSON nor verifies a scored row. The new test only covers a missing file ([test:175]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:175)); replacing the guard with a mere existence test would still pass it. Require parsed JSON with a non-empty `rows` list containing real schedule rows.

- The stale-artifact test is destructive to real ignored results. `_run` explicitly runs in the repository and sets the repo there ([test:77]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:77)), while the launcher writes the fixed `results/a1/mpas_schedule_cost` path ([launcher:100]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:100)). The test overwrites and finally deletes `schedule_cost_s8.json` without preserving an existing receipt ([test:189]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:189)). All shell tests leave or delete artifacts in that shared directory.

- The six-row test catches a deleted/key-corrupted literal, but not a corrupted expected number. It only asserts `set(parsed)` ([test:224]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:224)); changing `sfc:64=12` to `sfc:64=999` still passes. The real benchmark gate would abort on that wrong value, but the claimed regression test does not. Assert each complete expected mapping.

Verified correct:

- The no-successful-measurement guard cannot fire on a legitimate completed row: every successful loop iteration has `available` and `n_ranks`; only all-unavailable runs abort ([bench:421]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:421), [bench:465]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:465)).

- Your `['8', '9']` expectation is correct. Both validation arms run before the failure gate is evaluated ([launcher:158]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:158)); arm 3 is then skipped ([launcher:161]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:161)). A validation artifact failure becomes final exit 1; an arm-3 artifact failure propagates 90 ([launcher:173]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:173)).

- (d) Report, don’t broaden CI here. The configured workflow collects all tests but executes `tests/unit/` and selected paths, not `tests/bench/` ([ci.yml:97]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:97), [ci.yml:121]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:121)). Given the stated repo-wide Actions disablement, a targeted CI wiring change is inert and out of scope.

- (e) Keeping the shared `_env.sh` contract is the right scope call; `prewarm_s10` uses the same override pattern ([prewarm:18]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/prewarm_s10.sbatch:18)). But the claimed mitigation is false until the artifact check is semantic: it currently does not make “succeeded without measuring” detectable.

`bash -n` and `git diff --check` passed.
tokens used
96,896
NOT READY.

- The round-5 artifact guard is bypassable. [`run_arm`]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:142) accepts any file containing the literal `"n_ranks"`. Valid empty JSON such as `{"rows": [], "note": "n_ranks"}` passes, returns 0, and can reach `SCAN_DONE`. It neither parses JSON nor verifies a scored row. The new test only covers a missing file ([test:175]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:175)); replacing the guard with a mere existence test would still pass it. Require parsed JSON with a non-empty `rows` list containing real schedule rows.

- The stale-artifact test is destructive to real ignored results. `_run` explicitly runs in the repository and sets the repo there ([test:77]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:77)), while the launcher writes the fixed `results/a1/mpas_schedule_cost` path ([launcher:100]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:100)). The test overwrites and finally deletes `schedule_cost_s8.json` without preserving an existing receipt ([test:189]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:189)). All shell tests leave or delete artifacts in that shared directory.

- The six-row test catches a deleted/key-corrupted literal, but not a corrupted expected number. It only asserts `set(parsed)` ([test:224]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:224)); changing `sfc:64=12` to `sfc:64=999` still passes. The real benchmark gate would abort on that wrong value, but the claimed regression test does not. Assert each complete expected mapping.

Verified correct:

- The no-successful-measurement guard cannot fire on a legitimate completed row: every successful loop iteration has `available` and `n_ranks`; only all-unavailable runs abort ([bench:421]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:421), [bench:465]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:465)).

- Your `['8', '9']` expectation is correct. Both validation arms run before the failure gate is evaluated ([launcher:158]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:158)); arm 3 is then skipped ([launcher:161]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:161)). A validation artifact failure becomes final exit 1; an arm-3 artifact failure propagates 90 ([launcher:173]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:173)).

- (d) Report, don’t broaden CI here. The configured workflow collects all tests but executes `tests/unit/` and selected paths, not `tests/bench/` ([ci.yml:97]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:97), [ci.yml:121]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:121)). Given the stated repo-wide Actions disablement, a targeted CI wiring change is inert and out of scope.

- (e) Keeping the shared `_env.sh` contract is the right scope call; `prewarm_s10` uses the same override pattern ([prewarm:18]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/prewarm_s10.sbatch:18)). But the claimed mitigation is false until the artifact check is semantic: it currently does not make “succeeded without measuring” detectable.

`bash -n` and `git diff --check` passed.
