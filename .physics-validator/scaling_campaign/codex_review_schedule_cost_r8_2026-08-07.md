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
session id: 019fdc1a-1b05-71e3-948e-873949ab586b
--------
user
ADVERSARIAL REVIEW, ROUND 8. Verify the round-7 fixes; hunt for new defects. Refute, do not summarize. End with SHIP or NOT READY.

Files: `git diff` on scripts/bench/bench_voronoi_partition_methods.py and tests/bench/test_bench_voronoi_partition_methods.py; untracked scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch and tests/bench/test_mpas_schedule_cost_scan_sbatch.py.

ROUND-7 FINDINGS AND FIXES:
1. BLOCKER 'the parse guard accepts {"rows": {"n_ranks": 1}} (iterating a dict yields keys) and {"rows": [{"n_ranks": 1}]} (no schedule score)' -> the guard now requires rows to be a LIST, each candidate row to be a dict, and to carry BOTH 'n_ranks' and a dict 'schedule' containing 'n_rounds'. A new parametrized test drives the launcher with four decoys — rows-as-dict, row-without-schedule, schedule-without-n_rounds, and unparseable text — and asserts non-zero + 'no scored rows' + no SCAN_DONE for each. VERIFY the guard logic is right and that each decoy would pass under the PREVIOUS version (i.e. the test is a real regression net, not decoration).
2. 'the stub writes only {n_ranks: 64}; tests never assert per-arm rank counts or methods' -> the stub now writes a realistic scored row (n_ranks + schedule.n_rounds/max_degree), and test_each_arm_carries_the_flags_its_claim_depends_on now asserts --rank-counts is exactly ['64,128','64,128','128'] and --methods is 'geometric,sfc,metis' on every arm. VERIFY those are the right working points and that the assertions would fail if an arm's rank count or method list were changed.

Full tests/bench suite: 239 passed, 1 skipped.

ALSO: re-check anything previously verified. Look for new vacuity in the tests added since round 6. If you find only cosmetic or speculative issues, say SHIP — I have run 7 rounds and want to know whether anything REAL remains, not whether further polish is conceivable.

Cite file:line.
codex
I’ll independently trace the launcher contract and the regression tests, specifically checking whether the new cases would have slipped through the prior guard and whether the per-arm assertions actually constrain each invocation.
exec
/usr/bin/bash -lc "git status --short && git diff -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py && sed -n '1,260p' scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && sed -n '1,320p' tests/bench/test_mpas_schedule_cost_scan_sbatch.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 422ms:
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
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_r8_2026-08-07.md
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
# Does RECOLOURING still have room on the MPAS halo schedule?  CPU-only,
# ZERO GPU hours.
#
# Note this is NOT a two-way choice between colouring and ownership: a
# redesigned DIRECTED schedule (see the SCOPE note below) is a third path,
# and it is not bounded by the quantity measured here.  What this run
# settles is only whether recolouring THIS undirected graph is exhausted.
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
#     s6 lloyd=0 also gave gap == 0 on all three methods — rounds 7/7/6 at
#     np8 and 13/10/10 at np16 for geometric/sfc/metis.  Both are genuinely
#     ppermute (L6 = 40,962 cells; padded, np8 gives 40,968/8 = 5,121 and
#     np16 gives 40,976/16 = 2,561 cells/device, each above the 2,000
#     threshold).  But np8's 7 == n_dev-1 is complete-graph saturation and
#     says nothing; only np16 (13/10/10 < 15) escapes it.  Still a SMALL
#     working point; s8-s10 are the production ones.
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
# NOT overridable, deliberately.  An earlier revision made this settable
# from the environment so the exit-status logic could be regression-tested;
# that reintroduced exactly the failure class this script exists to prevent
# (a stray exported variable redirects a real scan to something else, every
# arm exits 0, SLURM files it COMPLETED with no results).  The shell test
# drives the interpreter instead — see
# tests/bench/test_mpas_schedule_cost_scan_sbatch.py.
BENCH=scripts/bench/bench_voronoi_partition_methods.py

# pymetis is present in .venv but was ABSENT from .venv-mpi for a while, and a
# missing partitioner is reported as "unavailable" rather than substituted.
# Say which python and whether metis is really there, so a two-method table
# cannot be misread as a three-method one.
echo "[scan] python=$PY"
"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"

run_arm () {  # $1=level  $2=rank-counts  $3=label  $4=expect-rounds (may be "")
  local json="$OUT/schedule_cost_s$1.json"
  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
  # Delete any previous artifact FIRST. A failed rerun that leaves the
  # earlier run's JSON in place produces a stale file with a current-looking
  # mtime story, and a stale receipt read as current is worse than a missing
  # one (codex round 5).
  rm -f "$json"
  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
  "$PY" "$BENCH" \
      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
      --methods geometric,sfc,metis --schedule-cost \
      ${4:+--expect-rounds "$4"} \
      --out "$json"
  local rc=$?   # capture BEFORE anything else runs, or `date` overwrites it
  echo "[scan] arm $3 exit=$rc"
  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
  # An exit code alone is not evidence the arm produced anything (CLAUDE.md:
  # "tool status is not evidence"). Require the artifact to exist and carry
  # rows, so a $PY that silently does nothing cannot yield SCAN_DONE with no
  # results. This does not make the launcher unspoofable — $PY and $REPO come
  # from _env.sh and remain configurable, as every launcher here is — but it
  # does make "succeeded without measuring" DETECTABLE.
  # PARSE the JSON — a substring grep for "n_ranks" passes on
  # {"rows": [], "note": "n_ranks"}, which is exactly the empty-but-
  # successful artifact this is supposed to catch (codex round 6).
  if [ "$rc" -eq 0 ] && ! "$PY" -c '
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
rows = doc.get("rows")
# `rows` MUST be a list of mappings, and a row only counts as SCORED if it
# carries a schedule with a round count.  Checking `"n_ranks" in r` alone
# passes on {"rows": {"n_ranks": 1}} (iterating a dict yields its keys) and
# on a quality-only row with no schedule at all (codex round 7).
if not isinstance(rows, list):
    sys.exit(1)
ok = any(
    isinstance(r, dict)
    and "n_ranks" in r
    and isinstance(r.get("schedule"), dict)
    and "n_rounds" in r["schedule"]
    for r in rows
)
sys.exit(0 if ok else 1)
' "$json"; then
    echo "[scan] arm $3 exited 0 but $json is missing, unparseable, or has"
    echo "[scan] no scored rows — refusing to call that a result"
    return 90
  fi
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

if [ "$RC10" -ne 0 ]; then
  # Do NOT let SLURM record this as COMPLETED: there is no valid s10 row,
  # and an exit-0 job with a missing result reads as a successful scan.
  echo "SCAN_FAILED_ARM3"
  exit "$RC10"
fi

echo "SCAN_DONE"
"""Exit-status regression for the MPAS schedule-cost scan launcher.

The failure this guards is not hypothetical: the first version of the script
captured arm 3's status into ``RC10`` and then ended on a successful ``echo``,
so a scan that produced NO valid s10 result exited 0 and SLURM filed it as
COMPLETED (codex round 2, BLOCKER).  "Tool status is not evidence" cuts both
ways — a launcher that cannot report failure makes every downstream reading
of ``sacct`` a lie.

HOW THIS IS DRIVEN, and why not the obvious way: the bench path in the
launcher is deliberately NOT overridable from the environment.  Making it
overridable (the first attempt) handed a stray exported variable the power to
redirect a real scan to something that exits 0 — reintroducing the very
failure class under test (codex round 4).  Instead this substitutes the
INTERPRETER via ``LEGOESM_PYTHON``, which is an existing production knob that
``_env.sh`` already reads, so the launcher itself carries no test-only seam.

The stub interpreter answers ``_env.sh``'s jax probe, then exits with a
scripted status per arm, recording each arm's full argv so a test can assert
both WHICH arms ran and that each carried its required flags.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SBATCH = (_REPO / "scripts" / "cluster" / "scaling_levante"
           / "mpas_schedule_cost_scan.sbatch")

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="needs bash to run the launcher")


def _run(tmp_path, codes, write_artifact=True):
    """Run the launcher with a stub interpreter that exits ``codes`` per arm.

    ``codes`` is one exit status per bench invocation, in order (arm 1, arm
    2, arm 3).  When ``write_artifact`` the stub also writes a minimal JSON
    to the arm's ``--out`` containing an ``n_ranks`` key, which is what the
    launcher's post-arm artifact check looks for; setting it False simulates
    an interpreter that exits 0 having measured nothing.

    Returns ``(proc, arms)`` where ``arms`` is the recorded argv of each
    bench call.
    """
    log = tmp_path / "calls.txt"
    stub = tmp_path / "stub_python"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        f"codes = {list(codes)!r}\n"
        f"log = {str(log)!r}\n"
        f"write_artifact = {bool(write_artifact)!r}\n"
        "argv = sys.argv[1:]\n"
        # `-c` must behave like a REAL interpreter, not a rubber stamp: both
        # _env.sh's jax probe and the launcher's JSON artifact check go
        # through it, and stubbing -c to exit 0 would make that check pass
        # vacuously in every test here.
        "if argv and argv[0] == '-c':\n"
        "    src = argv[1]\n"
        "    if 'import jax' in src:\n"
        "        sys.exit(0)\n"
        "    sys.argv = ['-c'] + argv[2:]\n"
        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
        "    sys.exit(0)\n"
        "with open(log, 'a') as f:\n"
        "    f.write(json.dumps(argv) + '\\n')\n"
        "n = sum(1 for _ in open(log))\n"
        "rc = codes[n - 1] if n <= len(codes) else 0\n"
        "if write_artifact and rc == 0:\n"
        "    out = argv[argv.index('--out') + 1]\n"
        "    with open(out, 'w') as f:\n"
        "        json.dump({'rows': [{'n_ranks': 64,\n"
        "                       'schedule': {'n_rounds': 12,\n"
        "                                    'max_degree': 12}}]}, f)\n"
        "sys.exit(rc)\n"
    )
    stub.chmod(0o755)

    # Point REPO at a throwaway tree that only SYMLINKS the real scripts.
    # The launcher cd's to $REPO and writes results/a1/mpas_schedule_cost
    # RELATIVE to it, so running these tests against the real repo would
    # create — and, in the stale-artifact test, DELETE — files in the same
    # directory a live scan writes its receipts to (codex round 6).
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir(exist_ok=True)
    link = fake_repo / "scripts"
    if not link.exists():
        link.symlink_to(_REPO / "scripts")

    env = dict(os.environ)
    env["SLURM_SUBMIT_DIR"] = str(_REPO)   # _env.sh is sourced from the real tree
    env["LEGOESM_REPO"] = str(fake_repo)
    env["LEGOESM_PYTHON"] = str(stub)
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    proc = subprocess.run(
        ["bash", str(_SBATCH)], env=env, cwd=str(fake_repo),
        capture_output=True, text=True, timeout=600)
    out_dir = fake_repo / "results" / "a1" / "mpas_schedule_cost"

    arms = []
    if log.exists():
        import json
        arms = [json.loads(line) for line in log.read_text().splitlines()
                if line.strip()]
    return proc, arms, out_dir


def _levels(arms):
    return [a[a.index("--subdivision") + 1] for a in arms]


def test_launcher_is_not_redirectable_from_the_environment():
    """The bench path must be hardcoded.

    If it were env-overridable, a stray exported variable could point every
    arm at something that exits 0 and the job would report SCAN_DONE with no
    results — the exact failure this module guards.
    """
    text = _SBATCH.read_text()
    assert "BENCH=scripts/bench/bench_voronoi_partition_methods.py" in text
    assert "LEGOESM_SCHEDULE_COST_BENCH" not in text


@pytest.mark.parametrize("codes, failing_arm", [([1, 0, 0], 1), ([0, 1, 0], 2)])
def test_validation_failure_aborts_before_the_unknown_arm(
        tmp_path, codes, failing_arm):
    """EITHER validation arm failing must abort non-zero and skip arm 3.

    An instrument that misses the known census cannot be trusted on the
    unknown one, so producing an s10 number anyway is worse than none.
    Both arms are exercised: guarding only arm 1 leaves arm 2 unchecked.
    """
    proc, arms, _out = _run(tmp_path, codes)
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert "SCAN_ABORTED_VALIDATION" in proc.stdout
    assert "SCAN_DONE" not in proc.stdout
    assert _levels(arms) == ["8", "9"], (
        f"arm 3 must not run after validation arm {failing_arm} failed: "
        f"{_levels(arms)}")


def test_arm3_failure_is_not_reported_as_success(tmp_path):
    """The original BLOCKER: arm 3 fails, the job must NOT exit 0, and must
    propagate the exact status."""
    proc, arms, _out = _run(tmp_path, [0, 0, 3])
    assert proc.returncode == 3, (
        f"arm-3 status not propagated (got {proc.returncode})\n"
        f"{proc.stdout[-2000:]}")
    assert "SCAN_FAILED_ARM3" in proc.stdout
    assert "SCAN_DONE" not in proc.stdout
    assert _levels(arms) == ["8", "9", "10"], _levels(arms)


def test_full_success_exits_zero_and_runs_every_arm(tmp_path):
    proc, arms, _out = _run(tmp_path, [0, 0, 0])
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert "SCAN_DONE" in proc.stdout
    assert _levels(arms) == ["8", "9", "10"], _levels(arms)


def test_each_arm_carries_the_flags_its_claim_depends_on(tmp_path):
    """The exit plumbing being right is worthless if an arm silently stops
    scoring or stops gating.

    Without this, dropping ``--schedule-cost`` (nothing is scored) or
    ``--expect-rounds`` (the validation arms assert nothing) would leave
    every other test in this module green.
    """
    _, arms, _out = _run(tmp_path, [0, 0, 0])
    assert len(arms) == 3, arms
    for argv in arms:
        assert "--schedule-cost" in argv, argv
        assert "--lloyd" in argv and argv[argv.index("--lloyd") + 1] == "0"
    # Validation arms must actually gate; the unknown arm must not pretend to.
    for argv in arms[:2]:
        assert "--expect-rounds" in argv, argv
        assert argv[argv.index("--expect-rounds") + 1].strip(), argv
    assert "--expect-rounds" not in arms[2], arms[2]
    # The WORKING POINTS are part of the claim: dropping 128, or a method,
    # would leave every other launcher test green and only surface as a
    # failed gate in a real (hours-long) run.
    assert [a[a.index("--rank-counts") + 1] for a in arms] == [
        "64,128", "64,128", "128"]
    for argv in arms:
        assert argv[argv.index("--methods") + 1] == "geometric,sfc,metis", argv


def test_arm_that_exits_zero_without_producing_results_is_caught(tmp_path):
    """An exit code is not evidence an arm measured anything.

    If the interpreter succeeds but writes no JSON (or one with no scored
    rows), the launcher must NOT report SCAN_DONE — otherwise a misconfigured
    run files as COMPLETED with nothing in it, the exact class this module
    exists to prevent.
    """
    proc, arms, _out = _run(tmp_path, [0, 0, 0], write_artifact=False)
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert "no scored rows" in proc.stdout
    assert "SCAN_DONE" not in proc.stdout
    # Both validation arms still RUN (they are independent measurements and
    # running both reports more before aborting); what matters is that the
    # unknown arm 3 is skipped.
    assert _levels(arms) == ["8", "9"], _levels(arms)


def test_stale_artifact_from_a_previous_run_is_removed_before_each_arm(
        tmp_path):
    """A failed rerun must not leave the previous run's JSON in place: a
    stale receipt read as current is worse than a missing one.

    Runs entirely inside the throwaway repo tree (see ``_run``), so it can
    never disturb the directory a live scan writes real receipts to.
    """
    out_dir = tmp_path / "repo" / "results" / "a1" / "mpas_schedule_cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "schedule_cost_s8.json"
    stale.write_text('{"rows": [{"n_ranks": 999, "STALE": true}]}')

    # Arm 1 fails and writes nothing; the stale file must be GONE, not left
    # behind looking like this run's result.
    proc, _, _out = _run(tmp_path, [1, 0, 0], write_artifact=False)
    assert proc.returncode != 0
    assert not stale.exists(), (
        "stale s8 JSON survived a failed arm and would read as current")


def test_artifact_check_parses_json_and_rejects_an_empty_rows_list(tmp_path):
    """The guard must PARSE, not grep.

    A substring check for "n_ranks" passes on {"rows": [], "note":
    "n_ranks"} — an empty result that mentions the key — which is exactly
    the empty-but-successful artifact the guard exists to catch (codex
    round 6).
    """
    log = tmp_path / "calls.txt"
    stub = tmp_path / "stub_python"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        f"log = {str(log)!r}\n"
        "argv = sys.argv[1:]\n"
        "if argv and argv[0] == '-c':\n"
        "    src = argv[1]\n"
        "    if 'import jax' in src:\n"
        "        sys.exit(0)\n"
        "    sys.argv = ['-c'] + argv[2:]\n"
        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
        "    sys.exit(0)\n"
        "with open(log, 'a') as f:\n"
        "    f.write(json.dumps(argv) + '\\n')\n"
        # Exit 0 having written a decoy: valid JSON, mentions the key, but
        # carries no scored row.
        "out = argv[argv.index('--out') + 1]\n"
        "with open(out, 'w') as f:\n"
        "    json.dump({'rows': [], 'note': 'n_ranks'}, f)\n"
        "sys.exit(0)\n"
    )
    stub.chmod(0o755)
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir(exist_ok=True)
    if not (fake_repo / "scripts").exists():
        (fake_repo / "scripts").symlink_to(_REPO / "scripts")

    env = dict(os.environ)
    env["SLURM_SUBMIT_DIR"] = str(_REPO)
    env["LEGOESM_REPO"] = str(fake_repo)
    env["LEGOESM_PYTHON"] = str(stub)
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    proc = subprocess.run(["bash", str(_SBATCH)], env=env, cwd=str(fake_repo),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert "no scored rows" in proc.stdout
    assert "SCAN_DONE" not in proc.stdout


def test_launcher_expectations_are_the_full_six_row_census():
    """The gate's strength is its CARDINALITY and content, not its presence.

    Asserting only that --expect-rounds is non-empty (the previous test)
    would still pass if five of the six rows were deleted, because the stub
    never runs the real parser.  Parse the launcher's own literals with the
    production parser and check every method x rank pair is present.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bench_vor_part_gate",
        _REPO / "scripts" / "bench" / "bench_voronoi_partition_methods.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    # The scorer's own docstring census, transcribed once here. Asserting
    # only the KEY SET would let sfc:64=12 rot into sfc:64=999 unnoticed
    # (codex round 6) — the values are the whole point of the gate.
    expected = {
        "S8_EXPECT": {("sfc", 64): 12, ("sfc", 128): 14,
                      ("metis", 64): 13, ("metis", 128): 19,
                      ("geometric", 64): 16, ("geometric", 128): 21},
        "S9_EXPECT": {("sfc", 64): 11, ("sfc", 128): 13,
                      ("metis", 64): 14, ("metis", 128): 18,
                      ("geometric", 64): 14, ("geometric", 128): 18},
    }
    text = _SBATCH.read_text()
    for var in ("S8_EXPECT", "S9_EXPECT"):
        line = next(ln for ln in text.splitlines()
                    if ln.startswith(f"{var}="))
        parsed = bench.parse_expect_rounds(line.split("=", 1)[1].strip('"'))
        assert parsed == expected[var], (
            f"{var} does not match the scorer's reference census "
            f"(spmd_schedule_cost docstring): {sorted(parsed.items())}")


@pytest.mark.parametrize("decoy, why", [

exec
/usr/bin/bash -lc 'git diff --unified=80 -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 132ms:
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

exec
/usr/bin/bash -lc "nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '320,540p'; nl -ba scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch | sed -n '1,280p'; nl -ba tests/bench/test_mpas_schedule_cost_scan_sbatch.py | sed -n '1,420p'; nl -ba tests/bench/test_bench_voronoi_partition_methods.py | sed -n '1,540p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 116ms:
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
   142	  # PARSE the JSON — a substring grep for "n_ranks" passes on
   143	  # {"rows": [], "note": "n_ranks"}, which is exactly the empty-but-
   144	  # successful artifact this is supposed to catch (codex round 6).
   145	  if [ "$rc" -eq 0 ] && ! "$PY" -c '
   146	import json, sys
   147	try:
   148	    doc = json.load(open(sys.argv[1]))
   149	except Exception:
   150	    sys.exit(1)
   151	rows = doc.get("rows")
   152	# `rows` MUST be a list of mappings, and a row only counts as SCORED if it
   153	# carries a schedule with a round count.  Checking `"n_ranks" in r` alone
   154	# passes on {"rows": {"n_ranks": 1}} (iterating a dict yields its keys) and
   155	# on a quality-only row with no schedule at all (codex round 7).
   156	if not isinstance(rows, list):
   157	    sys.exit(1)
   158	ok = any(
   159	    isinstance(r, dict)
   160	    and "n_ranks" in r
   161	    and isinstance(r.get("schedule"), dict)
   162	    and "n_rounds" in r["schedule"]
   163	    for r in rows
   164	)
   165	sys.exit(0 if ok else 1)
   166	' "$json"; then
   167	    echo "[scan] arm $3 exited 0 but $json is missing, unparseable, or has"
   168	    echo "[scan] no scored rows — refusing to call that a result"
   169	    return 90
   170	  fi
   171	  return $rc
   172	}
   173	
   174	# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
   175	# makes the bench exit non-zero unless it reproduces the census below, and a
   176	# listed pair that never got scored (e.g. pymetis missing) counts as a
   177	# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
   178	# pass — an instrument that misses the known answer cannot be trusted on the
   179	# unknown one.
   180	S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
   181	S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"
   182	
   183	run_arm 8 64,128 "1/3 VALIDATION s8" "$S8_EXPECT"; RC8=$?
   184	run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?
   185	
   186	if [ "$RC8" -ne 0 ] || [ "$RC9" -ne 0 ]; then
   187	  echo "[scan] INSTRUMENT CHECK FAILED (s8 rc=$RC8, s9 rc=$RC9)."
   188	  echo "[scan] The scorer does not reproduce the reference census, so an s10"
   189	  echo "[scan] number from it would be untrustworthy. NOT running arm 3."
   190	  echo "SCAN_ABORTED_VALIDATION"
   191	  exit 1
   192	fi
   193	
   194	# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
   195	run_arm 10 128 "3/3 UNKNOWN s10" ""; RC10=$?
   196	echo "[scan] arm 3 rc=$RC10"
   197	
   198	if [ "$RC10" -ne 0 ]; then
   199	  # Do NOT let SLURM record this as COMPLETED: there is no valid s10 row,
   200	  # and an exit-0 job with a missing result reads as a successful scan.
   201	  echo "SCAN_FAILED_ARM3"
   202	  exit "$RC10"
   203	fi
   204	
   205	echo "SCAN_DONE"
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
    61	        # `-c` must behave like a REAL interpreter, not a rubber stamp: both
    62	        # _env.sh's jax probe and the launcher's JSON artifact check go
    63	        # through it, and stubbing -c to exit 0 would make that check pass
    64	        # vacuously in every test here.
    65	        "if argv and argv[0] == '-c':\n"
    66	        "    src = argv[1]\n"
    67	        "    if 'import jax' in src:\n"
    68	        "        sys.exit(0)\n"
    69	        "    sys.argv = ['-c'] + argv[2:]\n"
    70	        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
    71	        "    sys.exit(0)\n"
    72	        "with open(log, 'a') as f:\n"
    73	        "    f.write(json.dumps(argv) + '\\n')\n"
    74	        "n = sum(1 for _ in open(log))\n"
    75	        "rc = codes[n - 1] if n <= len(codes) else 0\n"
    76	        "if write_artifact and rc == 0:\n"
    77	        "    out = argv[argv.index('--out') + 1]\n"
    78	        "    with open(out, 'w') as f:\n"
    79	        "        json.dump({'rows': [{'n_ranks': 64,\n"
    80	        "                       'schedule': {'n_rounds': 12,\n"
    81	        "                                    'max_degree': 12}}]}, f)\n"
    82	        "sys.exit(rc)\n"
    83	    )
    84	    stub.chmod(0o755)
    85	
    86	    # Point REPO at a throwaway tree that only SYMLINKS the real scripts.
    87	    # The launcher cd's to $REPO and writes results/a1/mpas_schedule_cost
    88	    # RELATIVE to it, so running these tests against the real repo would
    89	    # create — and, in the stale-artifact test, DELETE — files in the same
    90	    # directory a live scan writes its receipts to (codex round 6).
    91	    fake_repo = tmp_path / "repo"
    92	    fake_repo.mkdir(exist_ok=True)
    93	    link = fake_repo / "scripts"
    94	    if not link.exists():
    95	        link.symlink_to(_REPO / "scripts")
    96	
    97	    env = dict(os.environ)
    98	    env["SLURM_SUBMIT_DIR"] = str(_REPO)   # _env.sh is sourced from the real tree
    99	    env["LEGOESM_REPO"] = str(fake_repo)
   100	    env["LEGOESM_PYTHON"] = str(stub)
   101	    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
   102	    proc = subprocess.run(
   103	        ["bash", str(_SBATCH)], env=env, cwd=str(fake_repo),
   104	        capture_output=True, text=True, timeout=600)
   105	    out_dir = fake_repo / "results" / "a1" / "mpas_schedule_cost"
   106	
   107	    arms = []
   108	    if log.exists():
   109	        import json
   110	        arms = [json.loads(line) for line in log.read_text().splitlines()
   111	                if line.strip()]
   112	    return proc, arms, out_dir
   113	
   114	
   115	def _levels(arms):
   116	    return [a[a.index("--subdivision") + 1] for a in arms]
   117	
   118	
   119	def test_launcher_is_not_redirectable_from_the_environment():
   120	    """The bench path must be hardcoded.
   121	
   122	    If it were env-overridable, a stray exported variable could point every
   123	    arm at something that exits 0 and the job would report SCAN_DONE with no
   124	    results — the exact failure this module guards.
   125	    """
   126	    text = _SBATCH.read_text()
   127	    assert "BENCH=scripts/bench/bench_voronoi_partition_methods.py" in text
   128	    assert "LEGOESM_SCHEDULE_COST_BENCH" not in text
   129	
   130	
   131	@pytest.mark.parametrize("codes, failing_arm", [([1, 0, 0], 1), ([0, 1, 0], 2)])
   132	def test_validation_failure_aborts_before_the_unknown_arm(
   133	        tmp_path, codes, failing_arm):
   134	    """EITHER validation arm failing must abort non-zero and skip arm 3.
   135	
   136	    An instrument that misses the known census cannot be trusted on the
   137	    unknown one, so producing an s10 number anyway is worse than none.
   138	    Both arms are exercised: guarding only arm 1 leaves arm 2 unchecked.
   139	    """
   140	    proc, arms, _out = _run(tmp_path, codes)
   141	    assert proc.returncode != 0, proc.stdout[-2000:]
   142	    assert "SCAN_ABORTED_VALIDATION" in proc.stdout
   143	    assert "SCAN_DONE" not in proc.stdout
   144	    assert _levels(arms) == ["8", "9"], (
   145	        f"arm 3 must not run after validation arm {failing_arm} failed: "
   146	        f"{_levels(arms)}")
   147	
   148	
   149	def test_arm3_failure_is_not_reported_as_success(tmp_path):
   150	    """The original BLOCKER: arm 3 fails, the job must NOT exit 0, and must
   151	    propagate the exact status."""
   152	    proc, arms, _out = _run(tmp_path, [0, 0, 3])
   153	    assert proc.returncode == 3, (
   154	        f"arm-3 status not propagated (got {proc.returncode})\n"
   155	        f"{proc.stdout[-2000:]}")
   156	    assert "SCAN_FAILED_ARM3" in proc.stdout
   157	    assert "SCAN_DONE" not in proc.stdout
   158	    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
   159	
   160	
   161	def test_full_success_exits_zero_and_runs_every_arm(tmp_path):
   162	    proc, arms, _out = _run(tmp_path, [0, 0, 0])
   163	    assert proc.returncode == 0, proc.stdout[-2000:]
   164	    assert "SCAN_DONE" in proc.stdout
   165	    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
   166	
   167	
   168	def test_each_arm_carries_the_flags_its_claim_depends_on(tmp_path):
   169	    """The exit plumbing being right is worthless if an arm silently stops
   170	    scoring or stops gating.
   171	
   172	    Without this, dropping ``--schedule-cost`` (nothing is scored) or
   173	    ``--expect-rounds`` (the validation arms assert nothing) would leave
   174	    every other test in this module green.
   175	    """
   176	    _, arms, _out = _run(tmp_path, [0, 0, 0])
   177	    assert len(arms) == 3, arms
   178	    for argv in arms:
   179	        assert "--schedule-cost" in argv, argv
   180	        assert "--lloyd" in argv and argv[argv.index("--lloyd") + 1] == "0"
   181	    # Validation arms must actually gate; the unknown arm must not pretend to.
   182	    for argv in arms[:2]:
   183	        assert "--expect-rounds" in argv, argv
   184	        assert argv[argv.index("--expect-rounds") + 1].strip(), argv
   185	    assert "--expect-rounds" not in arms[2], arms[2]
   186	    # The WORKING POINTS are part of the claim: dropping 128, or a method,
   187	    # would leave every other launcher test green and only surface as a
   188	    # failed gate in a real (hours-long) run.
   189	    assert [a[a.index("--rank-counts") + 1] for a in arms] == [
   190	        "64,128", "64,128", "128"]
   191	    for argv in arms:
   192	        assert argv[argv.index("--methods") + 1] == "geometric,sfc,metis", argv
   193	
   194	
   195	def test_arm_that_exits_zero_without_producing_results_is_caught(tmp_path):
   196	    """An exit code is not evidence an arm measured anything.
   197	
   198	    If the interpreter succeeds but writes no JSON (or one with no scored
   199	    rows), the launcher must NOT report SCAN_DONE — otherwise a misconfigured
   200	    run files as COMPLETED with nothing in it, the exact class this module
   201	    exists to prevent.
   202	    """
   203	    proc, arms, _out = _run(tmp_path, [0, 0, 0], write_artifact=False)
   204	    assert proc.returncode != 0, proc.stdout[-2000:]
   205	    assert "no scored rows" in proc.stdout
   206	    assert "SCAN_DONE" not in proc.stdout
   207	    # Both validation arms still RUN (they are independent measurements and
   208	    # running both reports more before aborting); what matters is that the
   209	    # unknown arm 3 is skipped.
   210	    assert _levels(arms) == ["8", "9"], _levels(arms)
   211	
   212	
   213	def test_stale_artifact_from_a_previous_run_is_removed_before_each_arm(
   214	        tmp_path):
   215	    """A failed rerun must not leave the previous run's JSON in place: a
   216	    stale receipt read as current is worse than a missing one.
   217	
   218	    Runs entirely inside the throwaway repo tree (see ``_run``), so it can
   219	    never disturb the directory a live scan writes real receipts to.
   220	    """
   221	    out_dir = tmp_path / "repo" / "results" / "a1" / "mpas_schedule_cost"
   222	    out_dir.mkdir(parents=True, exist_ok=True)
   223	    stale = out_dir / "schedule_cost_s8.json"
   224	    stale.write_text('{"rows": [{"n_ranks": 999, "STALE": true}]}')
   225	
   226	    # Arm 1 fails and writes nothing; the stale file must be GONE, not left
   227	    # behind looking like this run's result.
   228	    proc, _, _out = _run(tmp_path, [1, 0, 0], write_artifact=False)
   229	    assert proc.returncode != 0
   230	    assert not stale.exists(), (
   231	        "stale s8 JSON survived a failed arm and would read as current")
   232	
   233	
   234	def test_artifact_check_parses_json_and_rejects_an_empty_rows_list(tmp_path):
   235	    """The guard must PARSE, not grep.
   236	
   237	    A substring check for "n_ranks" passes on {"rows": [], "note":
   238	    "n_ranks"} — an empty result that mentions the key — which is exactly
   239	    the empty-but-successful artifact the guard exists to catch (codex
   240	    round 6).
   241	    """
   242	    log = tmp_path / "calls.txt"
   243	    stub = tmp_path / "stub_python"
   244	    stub.write_text(
   245	        "#!/usr/bin/env python3\n"
   246	        "import sys, json\n"
   247	        f"log = {str(log)!r}\n"
   248	        "argv = sys.argv[1:]\n"
   249	        "if argv and argv[0] == '-c':\n"
   250	        "    src = argv[1]\n"
   251	        "    if 'import jax' in src:\n"
   252	        "        sys.exit(0)\n"
   253	        "    sys.argv = ['-c'] + argv[2:]\n"
   254	        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
   255	        "    sys.exit(0)\n"
   256	        "with open(log, 'a') as f:\n"
   257	        "    f.write(json.dumps(argv) + '\\n')\n"
   258	        # Exit 0 having written a decoy: valid JSON, mentions the key, but
   259	        # carries no scored row.
   260	        "out = argv[argv.index('--out') + 1]\n"
   261	        "with open(out, 'w') as f:\n"
   262	        "    json.dump({'rows': [], 'note': 'n_ranks'}, f)\n"
   263	        "sys.exit(0)\n"
   264	    )
   265	    stub.chmod(0o755)
   266	    fake_repo = tmp_path / "repo"
   267	    fake_repo.mkdir(exist_ok=True)
   268	    if not (fake_repo / "scripts").exists():
   269	        (fake_repo / "scripts").symlink_to(_REPO / "scripts")
   270	
   271	    env = dict(os.environ)
   272	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
   273	    env["LEGOESM_REPO"] = str(fake_repo)
   274	    env["LEGOESM_PYTHON"] = str(stub)
   275	    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
   276	    proc = subprocess.run(["bash", str(_SBATCH)], env=env, cwd=str(fake_repo),
   277	                          capture_output=True, text=True, timeout=600)
   278	    assert proc.returncode != 0, proc.stdout[-2000:]
   279	    assert "no scored rows" in proc.stdout
   280	    assert "SCAN_DONE" not in proc.stdout
   281	
   282	
   283	def test_launcher_expectations_are_the_full_six_row_census():
   284	    """The gate's strength is its CARDINALITY and content, not its presence.
   285	
   286	    Asserting only that --expect-rounds is non-empty (the previous test)
   287	    would still pass if five of the six rows were deleted, because the stub
   288	    never runs the real parser.  Parse the launcher's own literals with the
   289	    production parser and check every method x rank pair is present.
   290	    """
   291	    import importlib.util
   292	
   293	    spec = importlib.util.spec_from_file_location(
   294	        "bench_vor_part_gate",
   295	        _REPO / "scripts" / "bench" / "bench_voronoi_partition_methods.py")
   296	    bench = importlib.util.module_from_spec(spec)
   297	    spec.loader.exec_module(bench)
   298	
   299	    # The scorer's own docstring census, transcribed once here. Asserting
   300	    # only the KEY SET would let sfc:64=12 rot into sfc:64=999 unnoticed
   301	    # (codex round 6) — the values are the whole point of the gate.
   302	    expected = {
   303	        "S8_EXPECT": {("sfc", 64): 12, ("sfc", 128): 14,
   304	                      ("metis", 64): 13, ("metis", 128): 19,
   305	                      ("geometric", 64): 16, ("geometric", 128): 21},
   306	        "S9_EXPECT": {("sfc", 64): 11, ("sfc", 128): 13,
   307	                      ("metis", 64): 14, ("metis", 128): 18,
   308	                      ("geometric", 64): 14, ("geometric", 128): 18},
   309	    }
   310	    text = _SBATCH.read_text()
   311	    for var in ("S8_EXPECT", "S9_EXPECT"):
   312	        line = next(ln for ln in text.splitlines()
   313	                    if ln.startswith(f"{var}="))
   314	        parsed = bench.parse_expect_rounds(line.split("=", 1)[1].strip('"'))
   315	        assert parsed == expected[var], (
   316	            f"{var} does not match the scorer's reference census "
   317	            f"(spmd_schedule_cost docstring): {sorted(parsed.items())}")
   318	
   319	
   320	@pytest.mark.parametrize("decoy, why", [
   321	    ('{"rows": {"n_ranks": 1}}',
   322	     "rows is a DICT: iterating it yields keys, so a naive `in` test passes"),
   323	    ('{"rows": [{"n_ranks": 64}]}',
   324	     "a quality-only row with no schedule block is not a SCORED row"),
   325	    ('{"rows": [{"n_ranks": 64, "schedule": {}}]}',
   326	     "a schedule block with no n_rounds carries no score"),
   327	    ('not json at all',
   328	     "unparseable output is not a result"),
   329	])
   330	def test_artifact_guard_rejects_every_non_result_shape(tmp_path, decoy, why):
   331	    """Each decoy is valid-looking output that carries NO schedule score.
   332	
   333	    The guard must reject all of them; the first two are the shapes that
   334	    slipped past the earlier substring and `in`-based versions (codex
   335	    rounds 6 and 7).
   336	    """
   337	    stub = tmp_path / "stub_python"
   338	    stub.write_text(
   339	        "#!/usr/bin/env python3\n"
   340	        "import sys, json\n"
   341	        f"decoy = {decoy!r}\n"
   342	        "argv = sys.argv[1:]\n"
   343	        "if argv and argv[0] == '-c':\n"
   344	        "    src = argv[1]\n"
   345	        "    if 'import jax' in src:\n"
   346	        "        sys.exit(0)\n"
   347	        "    sys.argv = ['-c'] + argv[2:]\n"
   348	        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
   349	        "    sys.exit(0)\n"
   350	        "open(argv[argv.index('--out') + 1], 'w').write(decoy)\n"
   351	        "sys.exit(0)\n"
   352	    )
   353	    stub.chmod(0o755)
   354	    fake_repo = tmp_path / "repo"
   355	    fake_repo.mkdir(exist_ok=True)
   356	    if not (fake_repo / "scripts").exists():
   357	        (fake_repo / "scripts").symlink_to(_REPO / "scripts")
   358	
   359	    env = dict(os.environ)
   360	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
   361	    env["LEGOESM_REPO"] = str(fake_repo)
   362	    env["LEGOESM_PYTHON"] = str(stub)
   363	    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
   364	    proc = subprocess.run(["bash", str(_SBATCH)], env=env, cwd=str(fake_repo),
   365	                          capture_output=True, text=True, timeout=600)
   366	    assert proc.returncode != 0, f"{why}\n{proc.stdout[-1500:]}"
   367	    assert "no scored rows" in proc.stdout, why
   368	    assert "SCAN_DONE" not in proc.stdout, why
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

exec
/usr/bin/bash -lc "nl -ba scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch | sed -n '119,230p'; nl -ba tests/bench/test_mpas_schedule_cost_scan_sbatch.py | sed -n '1,400p'; nl -ba tests/bench/test_bench_voronoi_partition_methods.py | sed -n '450,620p'; git diff --numstat -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 208ms:
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
   142	  # PARSE the JSON — a substring grep for "n_ranks" passes on
   143	  # {"rows": [], "note": "n_ranks"}, which is exactly the empty-but-
   144	  # successful artifact this is supposed to catch (codex round 6).
   145	  if [ "$rc" -eq 0 ] && ! "$PY" -c '
   146	import json, sys
   147	try:
   148	    doc = json.load(open(sys.argv[1]))
   149	except Exception:
   150	    sys.exit(1)
   151	rows = doc.get("rows")
   152	# `rows` MUST be a list of mappings, and a row only counts as SCORED if it
   153	# carries a schedule with a round count.  Checking `"n_ranks" in r` alone
   154	# passes on {"rows": {"n_ranks": 1}} (iterating a dict yields its keys) and
   155	# on a quality-only row with no schedule at all (codex round 7).
   156	if not isinstance(rows, list):
   157	    sys.exit(1)
   158	ok = any(
   159	    isinstance(r, dict)
   160	    and "n_ranks" in r
   161	    and isinstance(r.get("schedule"), dict)
   162	    and "n_rounds" in r["schedule"]
   163	    for r in rows
   164	)
   165	sys.exit(0 if ok else 1)
   166	' "$json"; then
   167	    echo "[scan] arm $3 exited 0 but $json is missing, unparseable, or has"
   168	    echo "[scan] no scored rows — refusing to call that a result"
   169	    return 90
   170	  fi
   171	  return $rc
   172	}
   173	
   174	# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
   175	# makes the bench exit non-zero unless it reproduces the census below, and a
   176	# listed pair that never got scored (e.g. pymetis missing) counts as a
   177	# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
   178	# pass — an instrument that misses the known answer cannot be trusted on the
   179	# unknown one.
   180	S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
   181	S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"
   182	
   183	run_arm 8 64,128 "1/3 VALIDATION s8" "$S8_EXPECT"; RC8=$?
   184	run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?
   185	
   186	if [ "$RC8" -ne 0 ] || [ "$RC9" -ne 0 ]; then
   187	  echo "[scan] INSTRUMENT CHECK FAILED (s8 rc=$RC8, s9 rc=$RC9)."
   188	  echo "[scan] The scorer does not reproduce the reference census, so an s10"
   189	  echo "[scan] number from it would be untrustworthy. NOT running arm 3."
   190	  echo "SCAN_ABORTED_VALIDATION"
   191	  exit 1
   192	fi
   193	
   194	# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
   195	run_arm 10 128 "3/3 UNKNOWN s10" ""; RC10=$?
   196	echo "[scan] arm 3 rc=$RC10"
   197	
   198	if [ "$RC10" -ne 0 ]; then
   199	  # Do NOT let SLURM record this as COMPLETED: there is no valid s10 row,
   200	  # and an exit-0 job with a missing result reads as a successful scan.
   201	  echo "SCAN_FAILED_ARM3"
   202	  exit "$RC10"
   203	fi
   204	
   205	echo "SCAN_DONE"
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
    61	        # `-c` must behave like a REAL interpreter, not a rubber stamp: both
    62	        # _env.sh's jax probe and the launcher's JSON artifact check go
    63	        # through it, and stubbing -c to exit 0 would make that check pass
    64	        # vacuously in every test here.
    65	        "if argv and argv[0] == '-c':\n"
    66	        "    src = argv[1]\n"
    67	        "    if 'import jax' in src:\n"
    68	        "        sys.exit(0)\n"
    69	        "    sys.argv = ['-c'] + argv[2:]\n"
    70	        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
    71	        "    sys.exit(0)\n"
    72	        "with open(log, 'a') as f:\n"
    73	        "    f.write(json.dumps(argv) + '\\n')\n"
    74	        "n = sum(1 for _ in open(log))\n"
    75	        "rc = codes[n - 1] if n <= len(codes) else 0\n"
    76	        "if write_artifact and rc == 0:\n"
    77	        "    out = argv[argv.index('--out') + 1]\n"
    78	        "    with open(out, 'w') as f:\n"
    79	        "        json.dump({'rows': [{'n_ranks': 64,\n"
    80	        "                       'schedule': {'n_rounds': 12,\n"
    81	        "                                    'max_degree': 12}}]}, f)\n"
    82	        "sys.exit(rc)\n"
    83	    )
    84	    stub.chmod(0o755)
    85	
    86	    # Point REPO at a throwaway tree that only SYMLINKS the real scripts.
    87	    # The launcher cd's to $REPO and writes results/a1/mpas_schedule_cost
    88	    # RELATIVE to it, so running these tests against the real repo would
    89	    # create — and, in the stale-artifact test, DELETE — files in the same
    90	    # directory a live scan writes its receipts to (codex round 6).
    91	    fake_repo = tmp_path / "repo"
    92	    fake_repo.mkdir(exist_ok=True)
    93	    link = fake_repo / "scripts"
    94	    if not link.exists():
    95	        link.symlink_to(_REPO / "scripts")
    96	
    97	    env = dict(os.environ)
    98	    env["SLURM_SUBMIT_DIR"] = str(_REPO)   # _env.sh is sourced from the real tree
    99	    env["LEGOESM_REPO"] = str(fake_repo)
   100	    env["LEGOESM_PYTHON"] = str(stub)
   101	    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
   102	    proc = subprocess.run(
   103	        ["bash", str(_SBATCH)], env=env, cwd=str(fake_repo),
   104	        capture_output=True, text=True, timeout=600)
   105	    out_dir = fake_repo / "results" / "a1" / "mpas_schedule_cost"
   106	
   107	    arms = []
   108	    if log.exists():
   109	        import json
   110	        arms = [json.loads(line) for line in log.read_text().splitlines()
   111	                if line.strip()]
   112	    return proc, arms, out_dir
   113	
   114	
   115	def _levels(arms):
   116	    return [a[a.index("--subdivision") + 1] for a in arms]
   117	
   118	
   119	def test_launcher_is_not_redirectable_from_the_environment():
   120	    """The bench path must be hardcoded.
   121	
   122	    If it were env-overridable, a stray exported variable could point every
   123	    arm at something that exits 0 and the job would report SCAN_DONE with no
   124	    results — the exact failure this module guards.
   125	    """
   126	    text = _SBATCH.read_text()
   127	    assert "BENCH=scripts/bench/bench_voronoi_partition_methods.py" in text
   128	    assert "LEGOESM_SCHEDULE_COST_BENCH" not in text
   129	
   130	
   131	@pytest.mark.parametrize("codes, failing_arm", [([1, 0, 0], 1), ([0, 1, 0], 2)])
   132	def test_validation_failure_aborts_before_the_unknown_arm(
   133	        tmp_path, codes, failing_arm):
   134	    """EITHER validation arm failing must abort non-zero and skip arm 3.
   135	
   136	    An instrument that misses the known census cannot be trusted on the
   137	    unknown one, so producing an s10 number anyway is worse than none.
   138	    Both arms are exercised: guarding only arm 1 leaves arm 2 unchecked.
   139	    """
   140	    proc, arms, _out = _run(tmp_path, codes)
   141	    assert proc.returncode != 0, proc.stdout[-2000:]
   142	    assert "SCAN_ABORTED_VALIDATION" in proc.stdout
   143	    assert "SCAN_DONE" not in proc.stdout
   144	    assert _levels(arms) == ["8", "9"], (
   145	        f"arm 3 must not run after validation arm {failing_arm} failed: "
   146	        f"{_levels(arms)}")
   147	
   148	
   149	def test_arm3_failure_is_not_reported_as_success(tmp_path):
   150	    """The original BLOCKER: arm 3 fails, the job must NOT exit 0, and must
   151	    propagate the exact status."""
   152	    proc, arms, _out = _run(tmp_path, [0, 0, 3])
   153	    assert proc.returncode == 3, (
   154	        f"arm-3 status not propagated (got {proc.returncode})\n"
   155	        f"{proc.stdout[-2000:]}")
   156	    assert "SCAN_FAILED_ARM3" in proc.stdout
   157	    assert "SCAN_DONE" not in proc.stdout
   158	    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
   159	
   160	
   161	def test_full_success_exits_zero_and_runs_every_arm(tmp_path):
   162	    proc, arms, _out = _run(tmp_path, [0, 0, 0])
   163	    assert proc.returncode == 0, proc.stdout[-2000:]
   164	    assert "SCAN_DONE" in proc.stdout
   165	    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
   166	
   167	
   168	def test_each_arm_carries_the_flags_its_claim_depends_on(tmp_path):
   169	    """The exit plumbing being right is worthless if an arm silently stops
   170	    scoring or stops gating.
   171	
   172	    Without this, dropping ``--schedule-cost`` (nothing is scored) or
   173	    ``--expect-rounds`` (the validation arms assert nothing) would leave
   174	    every other test in this module green.
   175	    """
   176	    _, arms, _out = _run(tmp_path, [0, 0, 0])
   177	    assert len(arms) == 3, arms
   178	    for argv in arms:
   179	        assert "--schedule-cost" in argv, argv
   180	        assert "--lloyd" in argv and argv[argv.index("--lloyd") + 1] == "0"
   181	    # Validation arms must actually gate; the unknown arm must not pretend to.
   182	    for argv in arms[:2]:
   183	        assert "--expect-rounds" in argv, argv
   184	        assert argv[argv.index("--expect-rounds") + 1].strip(), argv
   185	    assert "--expect-rounds" not in arms[2], arms[2]
   186	    # The WORKING POINTS are part of the claim: dropping 128, or a method,
   187	    # would leave every other launcher test green and only surface as a
   188	    # failed gate in a real (hours-long) run.
   189	    assert [a[a.index("--rank-counts") + 1] for a in arms] == [
   190	        "64,128", "64,128", "128"]
   191	    for argv in arms:
   192	        assert argv[argv.index("--methods") + 1] == "geometric,sfc,metis", argv
   193	
   194	
   195	def test_arm_that_exits_zero_without_producing_results_is_caught(tmp_path):
   196	    """An exit code is not evidence an arm measured anything.
   197	
   198	    If the interpreter succeeds but writes no JSON (or one with no scored
   199	    rows), the launcher must NOT report SCAN_DONE — otherwise a misconfigured
   200	    run files as COMPLETED with nothing in it, the exact class this module
   201	    exists to prevent.
   202	    """
   203	    proc, arms, _out = _run(tmp_path, [0, 0, 0], write_artifact=False)
   204	    assert proc.returncode != 0, proc.stdout[-2000:]
   205	    assert "no scored rows" in proc.stdout
   206	    assert "SCAN_DONE" not in proc.stdout
   207	    # Both validation arms still RUN (they are independent measurements and
   208	    # running both reports more before aborting); what matters is that the
   209	    # unknown arm 3 is skipped.
   210	    assert _levels(arms) == ["8", "9"], _levels(arms)
   211	
   212	
   213	def test_stale_artifact_from_a_previous_run_is_removed_before_each_arm(
   214	        tmp_path):
   215	    """A failed rerun must not leave the previous run's JSON in place: a
   216	    stale receipt read as current is worse than a missing one.
   217	
   218	    Runs entirely inside the throwaway repo tree (see ``_run``), so it can
   219	    never disturb the directory a live scan writes real receipts to.
   220	    """
   221	    out_dir = tmp_path / "repo" / "results" / "a1" / "mpas_schedule_cost"
   222	    out_dir.mkdir(parents=True, exist_ok=True)
   223	    stale = out_dir / "schedule_cost_s8.json"
   224	    stale.write_text('{"rows": [{"n_ranks": 999, "STALE": true}]}')
   225	
   226	    # Arm 1 fails and writes nothing; the stale file must be GONE, not left
   227	    # behind looking like this run's result.
   228	    proc, _, _out = _run(tmp_path, [1, 0, 0], write_artifact=False)
   229	    assert proc.returncode != 0
   230	    assert not stale.exists(), (
   231	        "stale s8 JSON survived a failed arm and would read as current")
   232	
   233	
   234	def test_artifact_check_parses_json_and_rejects_an_empty_rows_list(tmp_path):
   235	    """The guard must PARSE, not grep.
   236	
   237	    A substring check for "n_ranks" passes on {"rows": [], "note":
   238	    "n_ranks"} — an empty result that mentions the key — which is exactly
   239	    the empty-but-successful artifact the guard exists to catch (codex
   240	    round 6).
   241	    """
   242	    log = tmp_path / "calls.txt"
   243	    stub = tmp_path / "stub_python"
   244	    stub.write_text(
   245	        "#!/usr/bin/env python3\n"
   246	        "import sys, json\n"
   247	        f"log = {str(log)!r}\n"
   248	        "argv = sys.argv[1:]\n"
   249	        "if argv and argv[0] == '-c':\n"
   250	        "    src = argv[1]\n"
   251	        "    if 'import jax' in src:\n"
   252	        "        sys.exit(0)\n"
   253	        "    sys.argv = ['-c'] + argv[2:]\n"
   254	        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
   255	        "    sys.exit(0)\n"
   256	        "with open(log, 'a') as f:\n"
   257	        "    f.write(json.dumps(argv) + '\\n')\n"
   258	        # Exit 0 having written a decoy: valid JSON, mentions the key, but
   259	        # carries no scored row.
   260	        "out = argv[argv.index('--out') + 1]\n"
   261	        "with open(out, 'w') as f:\n"
   262	        "    json.dump({'rows': [], 'note': 'n_ranks'}, f)\n"
   263	        "sys.exit(0)\n"
   264	    )
   265	    stub.chmod(0o755)
   266	    fake_repo = tmp_path / "repo"
   267	    fake_repo.mkdir(exist_ok=True)
   268	    if not (fake_repo / "scripts").exists():
   269	        (fake_repo / "scripts").symlink_to(_REPO / "scripts")
   270	
   271	    env = dict(os.environ)
   272	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
   273	    env["LEGOESM_REPO"] = str(fake_repo)
   274	    env["LEGOESM_PYTHON"] = str(stub)
   275	    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
   276	    proc = subprocess.run(["bash", str(_SBATCH)], env=env, cwd=str(fake_repo),
   277	                          capture_output=True, text=True, timeout=600)
   278	    assert proc.returncode != 0, proc.stdout[-2000:]
   279	    assert "no scored rows" in proc.stdout
   280	    assert "SCAN_DONE" not in proc.stdout
   281	
   282	
   283	def test_launcher_expectations_are_the_full_six_row_census():
   284	    """The gate's strength is its CARDINALITY and content, not its presence.
   285	
   286	    Asserting only that --expect-rounds is non-empty (the previous test)
   287	    would still pass if five of the six rows were deleted, because the stub
   288	    never runs the real parser.  Parse the launcher's own literals with the
   289	    production parser and check every method x rank pair is present.
   290	    """
   291	    import importlib.util
   292	
   293	    spec = importlib.util.spec_from_file_location(
   294	        "bench_vor_part_gate",
   295	        _REPO / "scripts" / "bench" / "bench_voronoi_partition_methods.py")
   296	    bench = importlib.util.module_from_spec(spec)
   297	    spec.loader.exec_module(bench)
   298	
   299	    # The scorer's own docstring census, transcribed once here. Asserting
   300	    # only the KEY SET would let sfc:64=12 rot into sfc:64=999 unnoticed
   301	    # (codex round 6) — the values are the whole point of the gate.
   302	    expected = {
   303	        "S8_EXPECT": {("sfc", 64): 12, ("sfc", 128): 14,
   304	                      ("metis", 64): 13, ("metis", 128): 19,
   305	                      ("geometric", 64): 16, ("geometric", 128): 21},
   306	        "S9_EXPECT": {("sfc", 64): 11, ("sfc", 128): 13,
   307	                      ("metis", 64): 14, ("metis", 128): 18,
   308	                      ("geometric", 64): 14, ("geometric", 128): 18},
   309	    }
   310	    text = _SBATCH.read_text()
   311	    for var in ("S8_EXPECT", "S9_EXPECT"):
   312	        line = next(ln for ln in text.splitlines()
   313	                    if ln.startswith(f"{var}="))
   314	        parsed = bench.parse_expect_rounds(line.split("=", 1)[1].strip('"'))
   315	        assert parsed == expected[var], (
   316	            f"{var} does not match the scorer's reference census "
   317	            f"(spmd_schedule_cost docstring): {sorted(parsed.items())}")
   318	
   319	
   320	@pytest.mark.parametrize("decoy, why", [
   321	    ('{"rows": {"n_ranks": 1}}',
   322	     "rows is a DICT: iterating it yields keys, so a naive `in` test passes"),
   323	    ('{"rows": [{"n_ranks": 64}]}',
   324	     "a quality-only row with no schedule block is not a SCORED row"),
   325	    ('{"rows": [{"n_ranks": 64, "schedule": {}}]}',
   326	     "a schedule block with no n_rounds carries no score"),
   327	    ('not json at all',
   328	     "unparseable output is not a result"),
   329	])
   330	def test_artifact_guard_rejects_every_non_result_shape(tmp_path, decoy, why):
   331	    """Each decoy is valid-looking output that carries NO schedule score.
   332	
   333	    The guard must reject all of them; the first two are the shapes that
   334	    slipped past the earlier substring and `in`-based versions (codex
   335	    rounds 6 and 7).
   336	    """
   337	    stub = tmp_path / "stub_python"
   338	    stub.write_text(
   339	        "#!/usr/bin/env python3\n"
   340	        "import sys, json\n"
   341	        f"decoy = {decoy!r}\n"
   342	        "argv = sys.argv[1:]\n"
   343	        "if argv and argv[0] == '-c':\n"
   344	        "    src = argv[1]\n"
   345	        "    if 'import jax' in src:\n"
   346	        "        sys.exit(0)\n"
   347	        "    sys.argv = ['-c'] + argv[2:]\n"
   348	        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
   349	        "    sys.exit(0)\n"
   350	        "open(argv[argv.index('--out') + 1], 'w').write(decoy)\n"
   351	        "sys.exit(0)\n"
   352	    )
   353	    stub.chmod(0o755)
   354	    fake_repo = tmp_path / "repo"
   355	    fake_repo.mkdir(exist_ok=True)
   356	    if not (fake_repo / "scripts").exists():
   357	        (fake_repo / "scripts").symlink_to(_REPO / "scripts")
   358	
   359	    env = dict(os.environ)
   360	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
   361	    env["LEGOESM_REPO"] = str(fake_repo)
   362	    env["LEGOESM_PYTHON"] = str(stub)
   363	    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
   364	    proc = subprocess.run(["bash", str(_SBATCH)], env=env, cwd=str(fake_repo),
   365	                          capture_output=True, text=True, timeout=600)
   366	    assert proc.returncode != 0, f"{why}\n{proc.stdout[-1500:]}"
   367	    assert "no scored rows" in proc.stdout, why
   368	    assert "SCAN_DONE" not in proc.stdout, why
   450	def test_empty_methods_refused_instead_of_measuring_nothing(monkeypatch):
   451	    """``--methods " , , "`` used to select nothing, write ``rows: []`` and
   452	    exit 0 — an empty run that reads as a successful one (codex round 4)."""
   453	    monkeypatch.setattr(sys, "argv", [
   454	        "bench", "--subdivision", "2", "--rank-counts", "2",
   455	        "--methods", " , , ", "--out", "/dev/null"])
   456	    with pytest.raises(SystemExit, match="selects NO methods"):
   457	        mod.main()
304	2	scripts/bench/bench_voronoi_partition_methods.py
309	0	tests/bench/test_bench_voronoi_partition_methods.py

exec
/usr/bin/bash -lc "nl -ba scripts/cluster/scaling_levante/_env.sh | sed -n '1,280p'; rg -n \"def spmd_schedule_cost|spmd_schedule_cost\\(\" legoesm tests scripts | head -80; git log --oneline --all -- scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch tests/bench/test_mpas_schedule_cost_scan_sbatch.py; git log --oneline -12 -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "rg --files | rg 'sharded_dynamics\\.py"'$|test_spmd_schedule_cost'"\\.py"'$'"'; rg -n -A180 -B35 '"'^def spmd_schedule_cost'"' packages src tests 2>/dev/null | head -300; git status --short --untracked-files=no; git diff --check" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 1053ms:
tests/parallel/test_spmd_schedule_cost.py
packages/core/legoesm/parallel/sharded_dynamics.py
tests/unit/test_sharded_dynamics.py
tests/unit/test_scale_sharded_dynamics.py
packages/core/legoesm/parallel/sharded_dynamics.py-1215-        Number of augmentation iterations.  2 is the minimum that
packages/core/legoesm/parallel/sharded_dynamics.py-1216-        closes the dycore's depth-2 operator chain; use a higher
packages/core/legoesm/parallel/sharded_dynamics.py-1217-        value only if a future operator extends the chain depth.
packages/core/legoesm/parallel/sharded_dynamics.py-1218-    """
packages/core/legoesm/parallel/sharded_dynamics.py-1219-    for _ in range(n_passes):
packages/core/legoesm/parallel/sharded_dynamics.py-1220-        cur_local_arr = np.concatenate([
packages/core/legoesm/parallel/sharded_dynamics.py-1221-            owned_cells_arr,
packages/core/legoesm/parallel/sharded_dynamics.py-1222-            np.fromiter(
packages/core/legoesm/parallel/sharded_dynamics.py-1223-                halo_cells_set,
packages/core/legoesm/parallel/sharded_dynamics.py-1224-                dtype=np.int64,
packages/core/legoesm/parallel/sharded_dynamics.py-1225-                count=len(halo_cells_set),
packages/core/legoesm/parallel/sharded_dynamics.py-1226-            ),
packages/core/legoesm/parallel/sharded_dynamics.py-1227-        ])
packages/core/legoesm/parallel/sharded_dynamics.py-1228-        # Edges where AT LEAST one ``cellsOnEdge`` is currently local.
packages/core/legoesm/parallel/sharded_dynamics.py-1229-        edge_one_in = (
packages/core/legoesm/parallel/sharded_dynamics.py-1230-            np.isin(cellsOnEdge_np[0], cur_local_arr)
packages/core/legoesm/parallel/sharded_dynamics.py-1231-            | np.isin(cellsOnEdge_np[1], cur_local_arr)
packages/core/legoesm/parallel/sharded_dynamics.py-1232-        )
packages/core/legoesm/parallel/sharded_dynamics.py-1233-        cand_edges = np.flatnonzero(edge_one_in)
packages/core/legoesm/parallel/sharded_dynamics.py-1234-        cand_cells = cellsOnEdge_np[:, cand_edges].reshape(-1)
packages/core/legoesm/parallel/sharded_dynamics.py-1235-        cand_cells = np.unique(cand_cells[cand_cells >= 0])
packages/core/legoesm/parallel/sharded_dynamics.py-1236-        # Set difference: cells not yet in local set.
packages/core/legoesm/parallel/sharded_dynamics.py-1237-        new_cells = cand_cells[~np.isin(cand_cells, cur_local_arr)]
packages/core/legoesm/parallel/sharded_dynamics.py-1238-        if new_cells.size == 0:
packages/core/legoesm/parallel/sharded_dynamics.py-1239-            return
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
 M scripts/bench/bench_voronoi_partition_methods.py
 M tests/bench/test_bench_voronoi_partition_methods.py

exec
/usr/bin/bash -lc "set -- 8 64,128 label 'sfc:64=12, sfc:128=14'; printf '<%s>\\n' "'${4:+--expect-rounds "$4"}; bash -n scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch; pytest -q tests/bench/test_bench_voronoi_partition_methods.py tests/bench/test_mpas_schedule_cost_scan_sbatch.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 127 in 183ms:
<--expect-rounds>
<sfc:64=12, sfc:128=14>
/usr/bin/bash: pytest: command not found

exec
/usr/bin/bash -lc "if [ -x .venv/bin/python ]; then .venv/bin/python -m pytest -q tests/bench/test_bench_voronoi_partition_methods.py tests/bench/test_mpas_schedule_cost_scan_sbatch.py; else echo 'no .venv python'; fi" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 1 in 1465ms:
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

 succeeded in 50261ms:
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
tests/bench/test_bench_voronoi_partition_methods.py:249:    ref = spmd_schedule_cost(mesh, 2, method="sfc")
scripts/bench/bench_voronoi_partition_methods.py:298:    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
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
tests/parallel/test_spmd_schedule_cost.py:169:    assert sd.spmd_schedule_cost(mesh, 1)["production_strategy"] is None
06c7c7e8c feat(parallel): route-B NCCL SPMD launch hardening — fallback guard, launcher-family GPU binding, transport report (audit item 6) (#879)

exec
/usr/bin/bash -lc "rg -n -A12 -B8 'rows.*n_ranks|unparseable|round.?7|schedule.*n_rounds|artifact guard' .physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md .physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md 2>/dev/null || true; git show HEAD:scripts/bench/bench_voronoi_partition_methods.py | nl -ba | sed -n '120,250p'; git show HEAD:tests/bench/test_bench_voronoi_partition_methods.py | nl -ba | sed -n '1,180p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 273ms:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12---------
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13-user
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14-ADVERSARIAL REVIEW, ROUND 7. Verify the round-6 fixes; hunt for new defects. Refute, do not summarize. End with SHIP or NOT READY.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-15-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-16-Files: `git diff` on scripts/bench/bench_voronoi_partition_methods.py and tests/bench/test_bench_voronoi_partition_methods.py; untracked scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch and tests/bench/test_mpas_schedule_cost_scan_sbatch.py.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-17-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-18-You endorsed my two scope calls in round 6 ((d) report the CI gap rather than wire tests/bench into a disabled CI; (e) keep the shared _env.sh contract). Those stand. Round-6's three defects and what I did:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-19-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:20:1. 'the artifact guard is a substring grep — {"rows": [], "note": "n_ranks"} passes' -> run_arm now PARSES with "$PY" -c: json.load, then requires any row in doc['rows'] to contain 'n_ranks'; unparseable or empty -> return 90. New test test_artifact_check_parses_json_and_rejects_an_empty_rows_list drives the launcher with a stub that writes exactly your decoy and asserts non-zero + no SCAN_DONE. NOTE the stub's -c branch now EXECUTES the code (it only short-circuits the 'import jax' probe), so the guard is really exercised rather than rubber-stamped. VERIFY the parse is correct, that using $PY for the check is sound given $PY is the same interpreter that ran the arm, and that the decoy test would fail if the guard reverted to grep.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-21-2. 'the stale-artifact test is destructive to real ignored results in results/a1/mpas_schedule_cost' -> ALL shell tests now run with LEGOESM_REPO pointed at a throwaway tmp tree that only symlinks scripts/, and cwd set there, so the launcher's relative results/ path lands in tmp. The stale test builds its decoy inside that tmp tree. VERIFY no test can now write or delete anything under the real repo's results/, and that the symlinked-scripts tree does not change what is being tested.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-22-3. 'the six-row test only asserts the key set; sfc:64=12 -> 999 still passes' -> it now asserts the FULL mapping (keys AND values) against the census transcribed in the test. VERIFY, and check the transcription against spmd_schedule_cost's docstring independently — if my transcription is wrong in BOTH the launcher and the test, they would agree with each other and still be wrong.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-23-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-24-ALSO: the launcher message now spans two echo lines ('...or has' / 'no scored rows — refusing...'); one test asserted the old single-line substring and failed, and I aligned the test. Confirm that was the right call rather than reflowing the message. Re-check anything previously verified. Look for new vacuity in the tests added since round 5.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-25-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-26-Cite file:line.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-27-codex
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-28-I’ll audit the actual diff and untracked additions as an adversarial review, including running the focused tests if the environment permits.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-29-exec
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-30-/usr/bin/bash -lc 'nl -ba scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && nl -ba tests/bench/test_mpas_schedule_cost_scan_sbatch.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-31-exec
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-32-/usr/bin/bash -lc 'git status --short && git diff -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-170-   135	  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-171-   136	  # An exit code alone is not evidence the arm produced anything (CLAUDE.md:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-172-   137	  # "tool status is not evidence"). Require the artifact to exist and carry
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-173-   138	  # rows, so a $PY that silently does nothing cannot yield SCAN_DONE with no
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-174-   139	  # results. This does not make the launcher unspoofable — $PY and $REPO come
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-175-   140	  # from _env.sh and remain configurable, as every launcher here is — but it
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-176-   141	  # does make "succeeded without measuring" DETECTABLE.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-177-   142	  # PARSE the JSON — a substring grep for "n_ranks" passes on
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:178:   143	  # {"rows": [], "note": "n_ranks"}, which is exactly the empty-but-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-179-   144	  # successful artifact this is supposed to catch (codex round 6).
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-180-   145	  if [ "$rc" -eq 0 ] && ! "$PY" -c '
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-181-   146	import json, sys
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-182-   147	try:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-183-   148	    doc = json.load(open(sys.argv[1]))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-184-   149	except Exception:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-185-   150	    sys.exit(1)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-186-   151	rows = doc.get("rows") or []
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-187-   152	sys.exit(0 if any("n_ranks" in r for r in rows) else 1)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-188-   153	' "$json"; then
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:189:   154	    echo "[scan] arm $3 exited 0 but $json is missing, unparseable, or has"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-190-   155	    echo "[scan] no scored rows — refusing to call that a result"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-191-   156	    return 90
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-192-   157	  fi
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-193-   158	  return $rc
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-194-   159	}
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-195-   160	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-196-   161	# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-197-   162	# makes the bench exit non-zero unless it reproduces the census below, and a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-198-   163	# listed pair that never got scored (e.g. pymetis missing) counts as a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-199-   164	# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-200-   165	# pass — an instrument that misses the known answer cannot be trusted on the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-201-   166	# unknown one.
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-298-    71	        "    sys.exit(0)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-299-    72	        "with open(log, 'a') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-300-    73	        "    f.write(json.dumps(argv) + '\\n')\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-301-    74	        "n = sum(1 for _ in open(log))\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-302-    75	        "rc = codes[n - 1] if n <= len(codes) else 0\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-303-    76	        "if write_artifact and rc == 0:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-304-    77	        "    out = argv[argv.index('--out') + 1]\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-305-    78	        "    with open(out, 'w') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:306:    79	        "        json.dump({'rows': [{'n_ranks': 64}]}, f)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-307-    80	        "sys.exit(rc)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-308-    81	    )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-309-    82	    stub.chmod(0o755)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-310-    83	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-311-    84	    # Point REPO at a throwaway tree that only SYMLINKS the real scripts.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-312-    85	    # The launcher cd's to $REPO and writes results/a1/mpas_schedule_cost
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-313-    86	    # RELATIVE to it, so running these tests against the real repo would
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-314-    87	    # create — and, in the stale-artifact test, DELETE — files in the same
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-315-    88	    # directory a live scan writes its receipts to (codex round 6).
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-316-    89	    fake_repo = tmp_path / "repo"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-317-    90	    fake_repo.mkdir(exist_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-318-    91	    link = fake_repo / "scripts"
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-434-   207	    stale receipt read as current is worse than a missing one.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-435-   208	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-436-   209	    Runs entirely inside the throwaway repo tree (see ``_run``), so it can
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-437-   210	    never disturb the directory a live scan writes real receipts to.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-438-   211	    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-439-   212	    out_dir = tmp_path / "repo" / "results" / "a1" / "mpas_schedule_cost"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-440-   213	    out_dir.mkdir(parents=True, exist_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-441-   214	    stale = out_dir / "schedule_cost_s8.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:442:   215	    stale.write_text('{"rows": [{"n_ranks": 999, "STALE": true}]}')
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-443-   216	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-444-   217	    # Arm 1 fails and writes nothing; the stale file must be GONE, not left
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-445-   218	    # behind looking like this run's result.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-446-   219	    proc, _, _out = _run(tmp_path, [1, 0, 0], write_artifact=False)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-447-   220	    assert proc.returncode != 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-448-   221	    assert not stale.exists(), (
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-449-   222	        "stale s8 JSON survived a failed arm and would read as current")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-450-   223	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-451-   224	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-452-   225	def test_artifact_check_parses_json_and_rejects_an_empty_rows_list(tmp_path):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-453-   226	    """The guard must PARSE, not grep.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-454-   227	
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-472-   245	        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-473-   246	        "    sys.exit(0)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-474-   247	        "with open(log, 'a') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-475-   248	        "    f.write(json.dumps(argv) + '\\n')\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-476-   249	        # Exit 0 having written a decoy: valid JSON, mentions the key, but
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-477-   250	        # carries no scored row.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-478-   251	        "out = argv[argv.index('--out') + 1]\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-479-   252	        "with open(out, 'w') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:480:   253	        "    json.dump({'rows': [], 'note': 'n_ranks'}, f)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-481-   254	        "sys.exit(0)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-482-   255	    )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-483-   256	    stub.chmod(0o755)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-484-   257	    fake_repo = tmp_path / "repo"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-485-   258	    fake_repo.mkdir(exist_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-486-   259	    if not (fake_repo / "scripts").exists():
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-487-   260	        (fake_repo / "scripts").symlink_to(_REPO / "scripts")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-488-   261	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-489-   262	    env = dict(os.environ)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-490-   263	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-491-   264	    env["LEGOESM_REPO"] = str(fake_repo)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-492-   265	    env["LEGOESM_PYTHON"] = str(stub)
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-725-+def check_expected_rounds(rows: list, expect: dict) -> list:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-726-+    """Compare scored rounds against *expect*; return failure strings.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-727-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-728-+    A listed pair that was never scored is a FAILURE, not a skip: otherwise
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-729-+    a sweep that silently dropped a method (unavailable ``pymetis``) or a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-730-+    rank count would still report a clean gate.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-731-+    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-732-+    scored = {
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:733:+        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-734-+        for r in rows
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-735-+        if r.get("available") and "schedule" in r and "n_ranks" in r
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-736-+    }
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-737-+    failures = []
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-738-+    for (method, n_ranks), want in sorted(expect.items()):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-739-+        got = scored.get((method, n_ranks))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-740-+        if got is None:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-741-+            failures.append(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-742-+                f"{method}:{n_ranks} expected rounds={want} but the pair was "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-743-+                f"NOT SCORED (method unavailable, or not in this sweep)")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-744-+        elif got != want:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-745-+            failures.append(
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1099-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1100-+    out2 = tmp_path / "on.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1101-+    monkeypatch.setattr(sys, "argv", [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1102-+        "bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1103-+        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1104-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1105-+    payload2 = json.loads(out2.read_text())
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1106-+    row = payload2["rows"][0]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:1107:+    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1108-+    assert payload2["metadata"]["extra"]["schedule_cost"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1109-+    # Scorer provenance per row: a copied row must show it scored a mesh
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1110-+    # partitioned for THIS device count, not one reordered for another.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1111-+    assert row["schedule"]["reorder_target"] == row["n_ranks"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1112-+    assert row["schedule"]["already_reordered"] is False
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1113-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1114-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1115-+def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1116-+    """``--lloyd`` must actually select the mesh, not just be recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1117-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1118-+    Non-vacuity: asserting only the recorded default (50) passes even if the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1119-+    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1150-+    that was never scored — a gate that can only pass is not a gate."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1151-+    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1152-+            "--methods", "geometric", "--schedule-cost"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1153-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1154-+    # Truth first: read what this configuration really scores.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1155-+    out = tmp_path / "truth.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1156-+    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1157-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:1158:+    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1159-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1160-+    # Matching expectation -> pass, and the check is recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1161-+    ok = tmp_path / "ok.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1162-+    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1163-+        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1164-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1165-+    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1166-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1167-+    # Wrong expectation -> non-zero exit.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1168-+    bad = tmp_path / "bad.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1169-+    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1170-+        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1692-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-66-        "    src = argv[1]\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1693-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-67-        "    if 'import jax' in src:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1694---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1695-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-74-        "n = sum(1 for _ in open(log))\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1696-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-75-        "rc = codes[n - 1] if n <= len(codes) else 0\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1697-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-76-        "if write_artifact and rc == 0:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1698-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-77-        "    out = argv[argv.index('--out') + 1]\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1699-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-78-        "    with open(out, 'w') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:1700:tests/bench/test_mpas_schedule_cost_scan_sbatch.py:79:        "        json.dump({'rows': [{'n_ranks': 64}]}, f)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1701-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-80-        "sys.exit(rc)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1702-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-81-    )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1703-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-82-    stub.chmod(0o755)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1704-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-83-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1705-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-84-    # Point REPO at a throwaway tree that only SYMLINKS the real scripts.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1706-tests/bench/test_mpas_schedule_cost_scan_sbatch.py:85:    # The launcher cd's to $REPO and writes results/a1/mpas_schedule_cost
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1707-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-86-    # RELATIVE to it, so running these tests against the real repo would
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1708-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-87-    # create — and, in the stale-artifact test, DELETE — files in the same
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1709-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-88-    # directory a live scan writes its receipts to (codex round 6).
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1710-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-89-    fake_repo = tmp_path / "repo"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1711-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-90-    fake_repo.mkdir(exist_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1712-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-91-    link = fake_repo / "scripts"
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1793-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-207-    stale receipt read as current is worse than a missing one.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1794-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-208-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1795-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-209-    Runs entirely inside the throwaway repo tree (see ``_run``), so it can
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1796-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-210-    never disturb the directory a live scan writes real receipts to.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1797-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-211-    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1798-tests/bench/test_mpas_schedule_cost_scan_sbatch.py:212:    out_dir = tmp_path / "repo" / "results" / "a1" / "mpas_schedule_cost"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1799-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-213-    out_dir.mkdir(parents=True, exist_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1800-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-214-    stale = out_dir / "schedule_cost_s8.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:1801:tests/bench/test_mpas_schedule_cost_scan_sbatch.py:215:    stale.write_text('{"rows": [{"n_ranks": 999, "STALE": true}]}')
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1802-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-216-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1803-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-217-    # Arm 1 fails and writes nothing; the stale file must be GONE, not left
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1804-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-218-    # behind looking like this run's result.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1805-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-219-    proc, _, _out = _run(tmp_path, [1, 0, 0], write_artifact=False)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1806-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-220-    assert proc.returncode != 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1807---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1808-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-223-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1809-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-224-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1810-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-225-def test_artifact_check_parses_json_and_rejects_an_empty_rows_list(tmp_path):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1811-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-226-    """The guard must PARSE, not grep.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1812-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-227-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1813-tests/bench/test_mpas_schedule_cost_scan_sbatch.py:228:    A substring check for "n_ranks" passes on {"rows": [], "note":
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1818-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-233-    log = tmp_path / "calls.txt"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1819-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-234-    stub = tmp_path / "stub_python"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1820---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1821-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-248-        "    f.write(json.dumps(argv) + '\\n')\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1822-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-249-        # Exit 0 having written a decoy: valid JSON, mentions the key, but
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1823-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-250-        # carries no scored row.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1824-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-251-        "out = argv[argv.index('--out') + 1]\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1825-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-252-        "with open(out, 'w') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:1826:tests/bench/test_mpas_schedule_cost_scan_sbatch.py:253:        "    json.dump({'rows': [], 'note': 'n_ranks'}, f)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1827-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-254-        "sys.exit(0)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1828-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-255-    )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1829-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-256-    stub.chmod(0o755)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1830-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-257-    fake_repo = tmp_path / "repo"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1831-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-258-    fake_repo.mkdir(exist_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1832-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-259-    if not (fake_repo / "scripts").exists():
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1833-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-260-        (fake_repo / "scripts").symlink_to(_REPO / "scripts")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1834-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-261-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1835-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-262-    env = dict(os.environ)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1836-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-263-    env["SLURM_SUBMIT_DIR"] = str(_REPO)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1837-tests/bench/test_mpas_schedule_cost_scan_sbatch.py:264:    env["LEGOESM_REPO"] = str(fake_repo)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-1838-tests/bench/test_mpas_schedule_cost_scan_sbatch.py-265-    env["LEGOESM_PYTHON"] = str(stub)
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2095-tests/bench/test_bench_voronoi_partition_methods.py:249:    ref = spmd_schedule_cost(mesh, 2, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2096-tests/bench/test_bench_voronoi_partition_methods.py-250-    sc = mod.schedule_cost_row(mesh, "sfc", 2)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2097-tests/bench/test_bench_voronoi_partition_methods.py-251-    for key in ("n_rounds", "max_degree", "n_rounds_greedy",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2098-tests/bench/test_bench_voronoi_partition_methods.py-252-                "coloring_method", "resolved_method", "cells_per_device",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2099-tests/bench/test_bench_voronoi_partition_methods.py-253-                "production_strategy"):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2100-tests/bench/test_bench_voronoi_partition_methods.py-254-        assert sc[key] == ref[key], key
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2101---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2102-tests/bench/test_bench_voronoi_partition_methods.py-276-    row = payload2["rows"][0]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:2103:tests/bench/test_bench_voronoi_partition_methods.py-277-    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2104-tests/bench/test_bench_voronoi_partition_methods.py-278-    assert payload2["metadata"]["extra"]["schedule_cost"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2105-tests/bench/test_bench_voronoi_partition_methods.py-279-    # Scorer provenance per row: a copied row must show it scored a mesh
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2106-tests/bench/test_bench_voronoi_partition_methods.py-280-    # partitioned for THIS device count, not one reordered for another.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2107-tests/bench/test_bench_voronoi_partition_methods.py:281:    assert row["schedule"]["reorder_target"] == row["n_ranks"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2108-tests/bench/test_bench_voronoi_partition_methods.py-282-    assert row["schedule"]["already_reordered"] is False
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2109-tests/bench/test_bench_voronoi_partition_methods.py-283-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2110-tests/bench/test_bench_voronoi_partition_methods.py-284-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2111-tests/bench/test_bench_voronoi_partition_methods.py-285-def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2112-tests/bench/test_bench_voronoi_partition_methods.py-286-    """``--lloyd`` must actually select the mesh, not just be recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2113---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2114-tests/bench/test_latlon_mpi_wiring.py-113-    state = baroclinic_wave_init_latlon(grid, sigma, perturbed=True, moist=False)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2115-tests/bench/test_latlon_mpi_wiring.py-114-
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2939-scripts/bench/run_levante_gpu_scaling.py-420-    at very large rank counts ``base_n * sqrt(ng) / ng`` shrinks below the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2940-scripts/bench/run_levante_gpu_scaling.py-421-    halo, which would make ``exchange_halo_latlon`` raise mid-sweep.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2941-scripts/bench/run_levante_gpu_scaling.py-422-    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2942-scripts/bench/run_levante_gpu_scaling.py-423-    n_raw = base_n * math.sqrt(n_gpus)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2943-scripts/bench/run_levante_gpu_scaling.py-424-    n_rounded = max(8, 2 * round(n_raw / 2))  # even number, min 8
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2944-scripts/bench/run_levante_gpu_scaling.py:425:    if n_ranks > 1:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2945-scripts/bench/run_levante_gpu_scaling.py:426:        while n_rounded % n_ranks != 0:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2946-scripts/bench/run_levante_gpu_scaling.py-427-            n_rounded += 2
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:2947:scripts/bench/run_levante_gpu_scaling.py:428:        # Floor at halo rows per band (n_ranks * _LATLON_MPI_HALO is even and
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2948-scripts/bench/run_levante_gpu_scaling.py:429:        # divisible by n_ranks, so it preserves both invariants above).
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2949-scripts/bench/run_levante_gpu_scaling.py:430:        n_rounded = max(n_rounded, n_ranks * _LATLON_MPI_HALO)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2950-scripts/bench/run_levante_gpu_scaling.py-431-    return n_rounded
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2951-scripts/bench/run_levante_gpu_scaling.py-432-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2952-scripts/bench/run_levante_gpu_scaling.py-433-# Strong scaling: fixed resolutions, sweep GPU counts.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2953-scripts/bench/run_levante_gpu_scaling.py-434-STRONG_RESOLUTIONS_CS = [48, 96, 192]   # cubed-sphere: ~200, ~100, ~50 km
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2954-scripts/bench/run_levante_gpu_scaling.py-435-STRONG_RESOLUTIONS_SP = [42, 85, 170]   # spectral: T42, T85, T170
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2955---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2956-scripts/bench/run_levante_gpu_scaling.py-1289-    jax.block_until_ready(jax.tree.leaves(carry))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2957-scripts/bench/run_levante_gpu_scaling.py-1290-    warmup_time = time.perf_counter() - t_warmup_start
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2958-scripts/bench/run_levante_gpu_scaling.py-1291-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-2959-scripts/bench/run_levante_gpu_scaling.py-1292-    # MPI detection (mirrors the dry path): the shared runner's sharded
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3354-scripts/bench/bench_voronoi_partition_methods.py-215-                f"expected one of {METHODS}")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3355-scripts/bench/bench_voronoi_partition_methods.py-216-        if key in out:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3356---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3357-scripts/bench/bench_voronoi_partition_methods.py-237-    A listed pair that was never scored is a FAILURE, not a skip: otherwise
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3358-scripts/bench/bench_voronoi_partition_methods.py-238-    a sweep that silently dropped a method (unavailable ``pymetis``) or a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3359-scripts/bench/bench_voronoi_partition_methods.py-239-    rank count would still report a clean gate.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3360-scripts/bench/bench_voronoi_partition_methods.py-240-    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3361-scripts/bench/bench_voronoi_partition_methods.py-241-    scored = {
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:3362:scripts/bench/bench_voronoi_partition_methods.py:242:        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3363-scripts/bench/bench_voronoi_partition_methods.py-243-        for r in rows
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3364-scripts/bench/bench_voronoi_partition_methods.py:244:        if r.get("available") and "schedule" in r and "n_ranks" in r
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3365-scripts/bench/bench_voronoi_partition_methods.py-245-    }
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3366-scripts/bench/bench_voronoi_partition_methods.py-246-    failures = []
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3367-scripts/bench/bench_voronoi_partition_methods.py:247:    for (method, n_ranks), want in sorted(expect.items()):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3368-scripts/bench/bench_voronoi_partition_methods.py:248:        got = scored.get((method, n_ranks))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3369-scripts/bench/bench_voronoi_partition_methods.py-249-        if got is None:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3370-scripts/bench/bench_voronoi_partition_methods.py-250-            failures.append(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3371-scripts/bench/bench_voronoi_partition_methods.py:251:                f"{method}:{n_ranks} expected rounds={want} but the pair was "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3372-scripts/bench/bench_voronoi_partition_methods.py-252-                f"NOT SCORED (method unavailable, or not in this sweep)")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3373-scripts/bench/bench_voronoi_partition_methods.py-253-        elif got != want:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-3374-scripts/bench/bench_voronoi_partition_methods.py-254-            failures.append(
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4362-scripts/bench/run_scaling_iter222_weak.sh-71-                 --output-dir '${outdir}' --tag '${tag}'" \
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4363-scripts/bench/run_scaling_iter222_weak.sh-72-      > "${logfile}" 2>&1; then
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4364---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4365-scripts/bench/bench_ocean_mpi_scaling.py-62-Weak scaling convention (``--weak-style``)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4366-scripts/bench/bench_ocean_mpi_scaling.py-63-------------------------------------------
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4367-scripts/bench/bench_ocean_mpi_scaling.py-64-``--resolution`` is the per-rank base in weak mode; two styles:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4368-scripts/bench/bench_ocean_mpi_scaling.py-65-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4369-scripts/bench/bench_ocean_mpi_scaling.py-66-``band`` (default — latitude-band algorithmic weak scaling)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:4370:scripts/bench/bench_ocean_mpi_scaling.py:67:    ``n_lat = rows * n_ranks`` with ``n_lon = 2 * rows`` held FIXED
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4371-scripts/bench/bench_ocean_mpi_scaling.py-68-    across the ladder, so cells/rank AND halo bytes/rank are exactly
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4372-scripts/bench/bench_ocean_mpi_scaling.py-69-    constant (each cut exchanges the same ``n_lon``-wide rows).  The
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4373-scripts/bench/bench_ocean_mpi_scaling.py-70-    domain aspect ratio distorts as np grows (bands get tall and
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4374-scripts/bench/bench_ocean_mpi_scaling.py-71-    narrow relative to a real production grid) — this ladder isolates
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4375-scripts/bench/bench_ocean_mpi_scaling.py-72-    the ALGORITHMIC weak-scaling cost.  ``--mode weak --resolution 64``
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4376-scripts/bench/bench_ocean_mpi_scaling.py-73-    at np=1 is exactly the LL64 case (n_lat=64, n_lon=128) of
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4377-scripts/bench/bench_ocean_mpi_scaling.py-74-    ``bench_ocean_gpu_scaling.py``.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4378-scripts/bench/bench_ocean_mpi_scaling.py-75-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4379-scripts/bench/bench_ocean_mpi_scaling.py-76-``aspect`` (production-realism cross-check)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4380-scripts/bench/bench_ocean_mpi_scaling.py:77:    ``n_lat ~= resolution * sqrt(n_ranks)`` rounded to a multiple of
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4381-scripts/bench/bench_ocean_mpi_scaling.py:78:    ``n_ranks`` (equal rows/band), with the standard square-cell
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4382-scripts/bench/bench_ocean_mpi_scaling.py-79-    ``n_lon = 2 * n_lat``.  The grid keeps the production aspect ratio,
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4390-scripts/bench/bench_ocean_mpi_scaling.py-265-# ===========================================================================
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4391-scripts/bench/bench_ocean_mpi_scaling.py-266-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4392-scripts/bench/bench_ocean_mpi_scaling.py-267-def resolve_grid_size(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4393-scripts/bench/bench_ocean_mpi_scaling.py:268:    mode: str, resolution: int, n_ranks: int, weak_style: str = "band",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4394-scripts/bench/bench_ocean_mpi_scaling.py-269-) -> tuple[int, int]:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4395-scripts/bench/bench_ocean_mpi_scaling.py-270-    """Return (n_lat, n_lon) for a case.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4396-scripts/bench/bench_ocean_mpi_scaling.py-271-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4397-scripts/bench/bench_ocean_mpi_scaling.py-272-    strong: ``resolution`` = global n_lat, n_lon = 2*n_lat.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:4398:scripts/bench/bench_ocean_mpi_scaling.py:273:    weak/band:   ``resolution`` = rows/rank; n_lat = rows*n_ranks and
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4399-scripts/bench/bench_ocean_mpi_scaling.py-274-            n_lon = 2*rows held fixed across the ladder (cells/rank and
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4400-scripts/bench/bench_ocean_mpi_scaling.py-275-            halo bytes/rank exactly constant; aspect ratio distorts).
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4401-scripts/bench/bench_ocean_mpi_scaling.py:276:    weak/aspect: n_lat ~= resolution*sqrt(n_ranks) rounded to a multiple
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4402-scripts/bench/bench_ocean_mpi_scaling.py:277:            of n_ranks (equal rows/band), n_lon = 2*n_lat (square-cell
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4403-scripts/bench/bench_ocean_mpi_scaling.py-278-            production aspect; cells/rank carries integer-rounding
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4404-scripts/bench/bench_ocean_mpi_scaling.py-279-            drift — the row records the ACTUAL cells/rank).
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4405-scripts/bench/bench_ocean_mpi_scaling.py-280-    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4406-scripts/bench/bench_ocean_mpi_scaling.py-281-    if mode == "strong":
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4407-scripts/bench/bench_ocean_mpi_scaling.py-282-        return resolution, 2 * resolution
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4408-scripts/bench/bench_ocean_mpi_scaling.py-283-    if weak_style == "band":
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4409-scripts/bench/bench_ocean_mpi_scaling.py:284:        return resolution * n_ranks, 2 * resolution
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4410-scripts/bench/bench_ocean_mpi_scaling.py-285-    if weak_style == "aspect":
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4411-scripts/bench/bench_ocean_mpi_scaling.py-286-        import math
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4412-scripts/bench/bench_ocean_mpi_scaling.py-287-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4413-scripts/bench/bench_ocean_mpi_scaling.py-288-        rows_per_rank = max(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4414-scripts/bench/bench_ocean_mpi_scaling.py-289-            MIN_ROWS_PER_RANK,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4415-scripts/bench/bench_ocean_mpi_scaling.py:290:            round(resolution * math.sqrt(n_ranks) / n_ranks),
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4416-scripts/bench/bench_ocean_mpi_scaling.py-291-        )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:4417:scripts/bench/bench_ocean_mpi_scaling.py:292:        n_lat = rows_per_rank * n_ranks
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4418-scripts/bench/bench_ocean_mpi_scaling.py-293-        return n_lat, 2 * n_lat
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4419-scripts/bench/bench_ocean_mpi_scaling.py-294-    raise ValueError(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4420-scripts/bench/bench_ocean_mpi_scaling.py-295-        f"Unknown weak_style {weak_style!r}; expected one of "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4421-scripts/bench/bench_ocean_mpi_scaling.py-296-        f"{WEAK_STYLE_CHOICES}."
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4422-scripts/bench/bench_ocean_mpi_scaling.py-297-    )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4423---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4424-scripts/bench/bench_ocean_mpi_scaling.py-443-def build_case(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4425-scripts/bench/bench_ocean_mpi_scaling.py-444-    *,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4426-scripts/bench/bench_ocean_mpi_scaling.py-445-    n_lat: int,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4427-scripts/bench/bench_ocean_mpi_scaling.py-446-    n_lon: int,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4428-scripts/bench/bench_ocean_mpi_scaling.py-447-    nlev: int,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4429-scripts/bench/bench_ocean_mpi_scaling.py:448:    n_ranks: int,
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4457-scripts/bench/bench_ocean_mpi_scaling.py-499-            scatter_state_latlon_cgrid_ocean,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4458-scripts/bench/bench_ocean_mpi_scaling.py-500-            slice_cgrid_geometry_to_band,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4459---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4460-scripts/bench/bench_ocean_mpi_scaling.py-510-            import numpy as np
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4461-scripts/bench/bench_ocean_mpi_scaling.py-511-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4462-scripts/bench/bench_ocean_mpi_scaling.py-512-            from legoesm.parallel.latlon_mpi import wet_band_boundaries
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4463-scripts/bench/bench_ocean_mpi_scaling.py-513-            wet_rows = np.asarray(state_global.land_mask.data).sum(axis=1)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4464-scripts/bench/bench_ocean_mpi_scaling.py-514-            band_boundaries = wet_band_boundaries(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:4465:scripts/bench/bench_ocean_mpi_scaling.py:515:                wet_rows, n_ranks, min_rows=MIN_ROWS_PER_RANK)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4466-scripts/bench/bench_ocean_mpi_scaling.py-516-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4467-scripts/bench/bench_ocean_mpi_scaling.py-517-        layout = initialize_distributed_latlon(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4468-scripts/bench/bench_ocean_mpi_scaling.py-518-            global_n_lat=n_lat, global_n_lon=n_lon,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4469-scripts/bench/bench_ocean_mpi_scaling.py-519-            band_boundaries=band_boundaries,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4470-scripts/bench/bench_ocean_mpi_scaling.py-520-            # Fold-aware layout: the north rank applies the ORCA fold at
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4471---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4472-scripts/bench/bench_ocean_mpi_scaling.py-527-        # about.  Computed host-side from the global mask on every rank
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4473-scripts/bench/bench_ocean_mpi_scaling.py-528-        # (deterministic), recorded on rank 0's JSON row.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4474-scripts/bench/bench_ocean_mpi_scaling.py-529-        import numpy as np
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4475-scripts/bench/bench_ocean_mpi_scaling.py-530-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4476-scripts/bench/bench_ocean_mpi_scaling.py-531-        _bounds = (band_boundaries if band_boundaries is not None
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4477-scripts/bench/bench_ocean_mpi_scaling.py:532:                   else _even_boundaries(n_lat, n_ranks))
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4636-scripts/bench/bench_ocean_mpi_scaling.py-1898-    if not is_rank0:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4637-scripts/bench/bench_ocean_mpi_scaling.py-1899-        return
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4638---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4639-scripts/bench/bench_ocean_mpi_scaling.py-2125-    )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4640-scripts/bench/bench_ocean_mpi_scaling.py-2126-    p.add_argument(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4641-scripts/bench/bench_ocean_mpi_scaling.py-2127-        "--mode", choices=["strong", "weak"], default="strong",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4642-scripts/bench/bench_ocean_mpi_scaling.py-2128-        help="strong: --resolution is global n_lat (n_lon=2*n_lat). "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4643-scripts/bench/bench_ocean_mpi_scaling.py-2129-             "weak: --resolution is lat rows per rank "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:4644:scripts/bench/bench_ocean_mpi_scaling.py:2130:             "(n_lat=rows*n_ranks, n_lon=2*rows fixed).",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4645-scripts/bench/bench_ocean_mpi_scaling.py-2131-    )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4646-scripts/bench/bench_ocean_mpi_scaling.py-2132-    p.add_argument(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4647-scripts/bench/bench_ocean_mpi_scaling.py-2133-        "--resolution", type=int, default=64,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4648-scripts/bench/bench_ocean_mpi_scaling.py-2134-        help="Global n_lat (strong) or per-rank base (weak).",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4649-scripts/bench/bench_ocean_mpi_scaling.py-2135-    )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4650---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4651-scripts/bench/bench_ocean_mpi_scaling.py-2171-    p.add_argument(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4652-scripts/bench/bench_ocean_mpi_scaling.py-2172-        "--wet-balance", action="store_true",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4653-scripts/bench/bench_ocean_mpi_scaling.py-2173-        help="Wet-cell-aware latitude bands: band boundaries equalize OCEAN "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4654-scripts/bench/bench_ocean_mpi_scaling.py-2174-             "cells per rank (wet_band_boundaries on the global land_mask) "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4655-scripts/bench/bench_ocean_mpi_scaling.py-2175-             "instead of row counts, so land-heavy bands stop idling. "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4656-scripts/bench/bench_ocean_mpi_scaling.py:2176:             "MPI-band path only (n_ranks > 1); every rank computes the "
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4692-scripts/bench/bench_ocean_mpi_scaling.py-2367-                f"{MIN_ROWS_PER_RANK}-row minimum.",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4693-scripts/bench/bench_ocean_mpi_scaling.py-2368-                flush=True,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4694-scripts/bench/bench_ocean_mpi_scaling.py-2369-            )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4695-scripts/bench/bench_ocean_mpi_scaling.py-2370-        return 2
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4696-scripts/bench/bench_ocean_mpi_scaling.py:2371:    if n_ranks > 1 and (n_lat // n_ranks) < MIN_ROWS_PER_RANK:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4697-scripts/bench/bench_ocean_mpi_scaling.py-2372-        if is_rank0:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4698-scripts/bench/bench_ocean_mpi_scaling.py-2373-            print(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4699-scripts/bench/bench_ocean_mpi_scaling.py-2374-                f"ERROR: lat-lon band MPI needs >={MIN_ROWS_PER_RANK} lat "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:4700:scripts/bench/bench_ocean_mpi_scaling.py:2375:                f"rows per rank; n_lat={n_lat} on {n_ranks} ranks gives "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4701-scripts/bench/bench_ocean_mpi_scaling.py:2376:                f"{n_lat // n_ranks} rows/rank. Increase --resolution or "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4702-scripts/bench/bench_ocean_mpi_scaling.py-2377-                f"reduce ranks.",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4703-scripts/bench/bench_ocean_mpi_scaling.py-2378-                flush=True,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4704-scripts/bench/bench_ocean_mpi_scaling.py-2379-            )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4705-scripts/bench/bench_ocean_mpi_scaling.py-2380-        return 2
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4706-scripts/bench/bench_ocean_mpi_scaling.py-2381-    # implicit_cn IS MPI-safe under the band decomposition as of the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4707---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4708-scripts/bench/bench_ocean_mpi_scaling.py-2396-        print("=" * 72)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4709-scripts/bench/bench_ocean_mpi_scaling.py-2397-        print("  legoESM Ocean CPU MPI Scaling Benchmark (lat-lon C-grid)")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4710-scripts/bench/bench_ocean_mpi_scaling.py-2398-        print("=" * 72)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4711-scripts/bench/bench_ocean_mpi_scaling.py-2399-        print(f"  Mode:        {mode_label}")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-4712-scripts/bench/bench_ocean_mpi_scaling.py-2400-        print(f"  Grid:        n_lat={n_lat} n_lon={n_lon} L{args.n_levels}")
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5066-scripts/bench/run_dd_scaling_sweep.sh-64-            --nx "$NX" --ny "$NY" --nlev "$NLEV" \
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5067-scripts/bench/run_dd_scaling_sweep.sh-65-            --dt "$DT" --n-acoustic-substeps "$N_ACOUSTIC" \
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5068---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5069-scripts/bench/run_dd_scaling_sweep.sh-81-    by_mode.setdefault(row["mode"], []).append(row)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5070-scripts/bench/run_dd_scaling_sweep.sh-82-print()
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5071-scripts/bench/run_dd_scaling_sweep.sh-83-for mode, mode_rows in by_mode.items():
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5072-scripts/bench/run_dd_scaling_sweep.sh-84-    print(f"=== {mode} scaling ===")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5073-scripts/bench/run_dd_scaling_sweep.sh-85-    print(f"{'ranks':>6} {'wall/step [s]':>14} {'steps/s':>10} {'efficiency':>12}")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:5074:scripts/bench/run_dd_scaling_sweep.sh:86:    mode_rows.sort(key=lambda r: int(r["n_ranks"]))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5075-scripts/bench/run_dd_scaling_sweep.sh-87-    baseline_wall = float(mode_rows[0]["wall_per_step_s"])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5076-scripts/bench/run_dd_scaling_sweep.sh-88-    for r in mode_rows:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5077-scripts/bench/run_dd_scaling_sweep.sh:89:        n = int(r["n_ranks"])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5078-scripts/bench/run_dd_scaling_sweep.sh-90-        wall = float(r["wall_per_step_s"])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5079-scripts/bench/run_dd_scaling_sweep.sh-91-        rate = float(r["steps_per_s"])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5080-scripts/bench/run_dd_scaling_sweep.sh-92-        if mode == "strong":
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5081-scripts/bench/run_dd_scaling_sweep.sh-93-            eff = baseline_wall / (n * wall) if n > 0 else 0.0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5082-scripts/bench/run_dd_scaling_sweep.sh-94-        else:  # weak
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5083---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5084-tests/atmosphere/nonhydrostatic/integration/test_summarize_rce_trajectory.py-1261-def _write_log_txt(path: Path, max_w_values: list[float]) -> None:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5085-tests/atmosphere/nonhydrostatic/integration/test_summarize_rce_trajectory.py-1262-    """Build a minimal log.txt matching run_rce_mpi_long.py's
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-5086-tests/atmosphere/nonhydrostatic/integration/test_summarize_rce_trajectory.py-1263-    schema. ``max_w_values`` populates the max|w| column row by
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7858-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-135-  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7859-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-136-  # An exit code alone is not evidence the arm produced anything (CLAUDE.md:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7860-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-137-  # "tool status is not evidence"). Require the artifact to exist and carry
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7861-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:138:  # rows, so a $PY that silently does nothing cannot yield SCAN_DONE with no
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7862-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-139-  # results. This does not make the launcher unspoofable — $PY and $REPO come
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7863-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:140:  # from _env.sh and remain configurable, as every launcher here is — but it
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7864-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-141-  # does make "succeeded without measuring" DETECTABLE.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7865-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:142:  # PARSE the JSON — a substring grep for "n_ranks" passes on
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:7866:scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:143:  # {"rows": [], "note": "n_ranks"}, which is exactly the empty-but-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7867-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-144-  # successful artifact this is supposed to catch (codex round 6).
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7868-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-145-  if [ "$rc" -eq 0 ] && ! "$PY" -c '
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7869-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-146-import json, sys
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7870-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-147-try:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7871-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-148-    doc = json.load(open(sys.argv[1]))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7872-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-149-except Exception:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7873-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-150-    sys.exit(1)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7874-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-151-rows = doc.get("rows") or []
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7875-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:152:sys.exit(0 if any("n_ranks" in r for r in rows) else 1)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7876-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-153-' "$json"; then
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:7877:scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-154-    echo "[scan] arm $3 exited 0 but $json is missing, unparseable, or has"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7878-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-155-    echo "[scan] no scored rows — refusing to call that a result"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7879-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-156-    return 90
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7880-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-157-  fi
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7881---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7882-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-187-  # and an exit-0 job with a missing result reads as a successful scan.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7883-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-188-  echo "SCAN_FAILED_ARM3"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7884-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-189-  exit "$RC10"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7885-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-190-fi
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7886-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-191-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7887-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:192:echo "SCAN_DONE"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7888---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-7889-scripts/cluster/scaling_levante/_env.sh-1-# Shared environment for DKRZ Levante GPU scaling jobs (sourced by the SLURM
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8802---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8803-tests/ocean/unit/test_multigrid_preconditioner.py-266-def test_coarse_band_hierarchy_distributed_lockstep():
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8804-tests/ocean/unit/test_multigrid_preconditioner.py-267-    """4-rank 64-row decomposition: every rank produces the SAME depth and stays
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8805-tests/ocean/unit/test_multigrid_preconditioner.py-268-    even-aligned at every level (the lock-step requirement so the band-local
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8806-tests/ocean/unit/test_multigrid_preconditioner.py-269-    restriction equals the global one).  Depth is set by the GLOBAL row count
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8807-tests/ocean/unit/test_multigrid_preconditioner.py-270-    (coarsen until global < 2*min_coarse_rows): 64->32->16->8 (depth 4)."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8808-tests/ocean/unit/test_multigrid_preconditioner.py:271:    n_ranks, n_lat = 4, 64
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8809-tests/ocean/unit/test_multigrid_preconditioner.py:272:    hs = [_coarse_band_hierarchy(make_latlon_band_layout(r, n_ranks, n_lat, N_LON),
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:8810:tests/ocean/unit/test_multigrid_preconditioner.py:273:                                 min_coarse_rows=8) for r in range(n_ranks)]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8811-tests/ocean/unit/test_multigrid_preconditioner.py-274-    depths = {len(h) for h in hs}
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8812-tests/ocean/unit/test_multigrid_preconditioner.py-275-    assert depths == {4}, f"ranks disagree on depth: {depths}"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8813-tests/ocean/unit/test_multigrid_preconditioner.py-276-    for h in hs:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8814-tests/ocean/unit/test_multigrid_preconditioner.py-277-        assert [lvl.n_lat_local for lvl in h] == [16, 8, 4, 2]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8815-tests/ocean/unit/test_multigrid_preconditioner.py-278-        # global lat halves with the band; depth driven by global, not local.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8816---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8817-tests/ocean/unit/test_multigrid_preconditioner.py-350-    assert float(jnp.min(np.asarray(vm[-1]))) > 0.5       # north cut OPEN (wet)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8818-tests/ocean/unit/test_multigrid_preconditioner.py-351-    assert float(jnp.min(np.asarray(Hv[-1]))) > 0.0       # north cut carries depth
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8819-tests/ocean/unit/test_multigrid_preconditioner.py-352-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8820-tests/ocean/unit/test_multigrid_preconditioner.py-353-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8821-tests/ocean/unit/test_multigrid_preconditioner.py-354-def test_banded_mg_reduces_to_serial_on_global_band():
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8822-tests/ocean/unit/test_multigrid_preconditioner.py:355:    """The BANDED V-cycle on a GLOBAL single band (n_ranks=1) matches the SERIAL
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8830-tests/ocean/unit/test_multigrid_preconditioner.py-372-        f"{np.max(np.abs(out_b - out_s)):.2e}")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8831-tests/ocean/unit/test_multigrid_preconditioner.py-373-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8832-tests/ocean/unit/test_multigrid_preconditioner.py-374-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8833-tests/ocean/unit/test_multigrid_preconditioner.py-375-def test_banded_mg_refuses_unequal_bands():
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8834-tests/ocean/unit/test_multigrid_preconditioner.py:376:    """Lock-step guard: an UNEQUAL decomposition (n_lat_global % n_ranks != 0)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8835-tests/ocean/unit/test_multigrid_preconditioner.py-377-    would coarsen ranks to different depths, so the banded factory FAILS LOUD
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8836-tests/ocean/unit/test_multigrid_preconditioner.py-378-    rather than deadlock on mismatched halo schedules."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8837-tests/ocean/unit/test_multigrid_preconditioner.py-379-    grid, coeff, mask, H_cell, _, _ = _setup()
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:8838:tests/ocean/unit/test_multigrid_preconditioner.py:380:    # 48 rows over 5 ranks = 9/10/10/... unequal; rank-0 layout carries n_ranks.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8839-tests/ocean/unit/test_multigrid_preconditioner.py-381-    layout = make_latlon_band_layout(0, 5, N_LAT, N_LON)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8840-tests/ocean/unit/test_multigrid_preconditioner.py-382-    with pytest.raises(ValueError, match="EQUAL bands"):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8841-tests/ocean/unit/test_multigrid_preconditioner.py-383-        _make_multigrid_preconditioner_banded(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8842-tests/ocean/unit/test_multigrid_preconditioner.py-384-            H_cell, coeff, grid, mask, layout)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8843-tests/ocean/unit/test_multigrid_preconditioner.py-385-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8844---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8845-tests/parallel/test_spmd_schedule_cost.py:1:"""Direct tests for ``sharded_dynamics.spmd_schedule_cost``.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8846-tests/parallel/test_spmd_schedule_cost.py-2-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8847-tests/parallel/test_spmd_schedule_cost.py-3-It scores a mesh split by how many sequential halo exchanges it needs. The
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8848-tests/parallel/test_spmd_schedule_cost.py-4-number only means something if it comes from the same builders, the same halo
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8849-tests/parallel/test_spmd_schedule_cost.py-5-depth, and the same mesh state production uses -- so that is what these pin.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8850-tests/parallel/test_spmd_schedule_cost.py-6-"""
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8860-tests/parallel/test_spmd_schedule_cost.py-25-    assert c["resolved_method"] == "sfc", "auto must resolve concretely"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8861-tests/parallel/test_spmd_schedule_cost.py-26-    # A round exchanges data between disjoint device PAIRS, so a proper
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8862-tests/parallel/test_spmd_schedule_cost.py-27-    # schedule on n_dev devices needs between 1 and n_dev-1 rounds.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8863---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8864-tests/parallel/test_spmd_schedule_cost.py-35-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8865-tests/parallel/test_spmd_schedule_cost.py-36-def test_reports_max_degree_so_optimality_is_measured_not_assumed(mesh):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8866-tests/parallel/test_spmd_schedule_cost.py-37-    """The colouring is a best-of-a-few-orders search, NOT a proof of the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8867-tests/parallel/test_spmd_schedule_cost.py-38-    max_degree lower bound. Callers must be able to check equality rather than
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:8868:tests/parallel/test_spmd_schedule_cost.py-39-    assume it, so max_degree is returned alongside n_rounds."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8869-tests/parallel/test_spmd_schedule_cost.py:40:    c = sd.spmd_schedule_cost(mesh, 8)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8870-tests/parallel/test_spmd_schedule_cost.py-41-    assert "max_degree" in c and c["max_degree"] > 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:8871:tests/parallel/test_spmd_schedule_cost.py-42-    assert c["n_rounds"] >= c["max_degree"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8872-tests/parallel/test_spmd_schedule_cost.py-43-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8873-tests/parallel/test_spmd_schedule_cost.py-44-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8874-tests/parallel/test_spmd_schedule_cost.py-45-def test_single_device_is_zero_rounds_and_zero_is_refused(mesh):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:8875:tests/parallel/test_spmd_schedule_cost.py:46:    assert sd.spmd_schedule_cost(mesh, 1)["n_rounds"] == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8876-tests/parallel/test_spmd_schedule_cost.py-47-    with pytest.raises(ValueError, match="integer >= 1"):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8877-tests/parallel/test_spmd_schedule_cost.py:48:        sd.spmd_schedule_cost(mesh, 0)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8878-tests/parallel/test_spmd_schedule_cost.py-49-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8879-tests/parallel/test_spmd_schedule_cost.py-50-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8880-tests/parallel/test_spmd_schedule_cost.py-51-def test_flags_allgather_when_production_would_not_use_ppermute(mesh):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8881-tests/parallel/test_spmd_schedule_cost.py-52-    """Production auto-selects allgather below a cells/device threshold; the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8882-tests/parallel/test_spmd_schedule_cost.py-53-    ppermute round count is then counterfactual and must say so."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8883-tests/parallel/test_spmd_schedule_cost.py:54:    c = sd.spmd_schedule_cost(mesh, 8,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8884-tests/parallel/test_spmd_schedule_cost.py-55-                              ppermute_cells_per_device_threshold=10**9)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8885-tests/parallel/test_spmd_schedule_cost.py-56-    assert c["production_strategy"] == "allgather"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8886-tests/parallel/test_spmd_schedule_cost.py:57:    big = sd.spmd_schedule_cost(mesh, 8,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8887-tests/parallel/test_spmd_schedule_cost.py-58-                                ppermute_cells_per_device_threshold=1)
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8894-tests/parallel/test_spmd_schedule_cost.py-65-    scoring the raw mesh once."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8895-tests/parallel/test_spmd_schedule_cost.py-66-    prepared = reorder_voronoi_for_sharding(mesh, 8, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8896-tests/parallel/test_spmd_schedule_cost.py:67:    pre = sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8897-tests/parallel/test_spmd_schedule_cost.py:68:    raw = sd.spmd_schedule_cost(mesh, 8, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8898-tests/parallel/test_spmd_schedule_cost.py-69-    assert pre["resolved_method"] == "pre-reordered"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8899-tests/parallel/test_spmd_schedule_cost.py-70-    assert pre["reorder_target"] is None, (
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8900-tests/parallel/test_spmd_schedule_cost.py-71-        "the target that produced a pre-reordered mesh is not recoverable "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8901-tests/parallel/test_spmd_schedule_cost.py-72-        "from it — reporting n_dev would assert something unverified")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:8902:tests/parallel/test_spmd_schedule_cost.py-73-    assert pre["n_rounds"] == raw["n_rounds"], (
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8903-tests/parallel/test_spmd_schedule_cost.py-74-        "scoring a pre-reordered mesh must match scoring the raw mesh with "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8904-tests/parallel/test_spmd_schedule_cost.py-75-        "the same ownership")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8905-tests/parallel/test_spmd_schedule_cost.py-76-    with pytest.raises(ValueError, match="reorder_target is meaningless"):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8906-tests/parallel/test_spmd_schedule_cost.py:77:        sd.spmd_schedule_cost(prepared, 8, already_reordered=True,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8907-tests/parallel/test_spmd_schedule_cost.py-78-                              reorder_target=16)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8908-tests/parallel/test_spmd_schedule_cost.py-79-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8909-tests/parallel/test_spmd_schedule_cost.py-80-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8910-tests/parallel/test_spmd_schedule_cost.py-81-def test_reorder_target_differing_from_run_device_count(mesh):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8911-tests/parallel/test_spmd_schedule_cost.py-82-    """The scaling bench reorders once for a target and then runs at a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8912-tests/parallel/test_spmd_schedule_cost.py-83-    different device count. That combination must be expressible, and must
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8913-tests/parallel/test_spmd_schedule_cost.py-84-    differ from reordering for the run count."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8914-tests/parallel/test_spmd_schedule_cost.py:85:    same = sd.spmd_schedule_cost(mesh, 4, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8915-tests/parallel/test_spmd_schedule_cost.py:86:    split_for_16 = sd.spmd_schedule_cost(mesh, 4, method="sfc",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8916-tests/parallel/test_spmd_schedule_cost.py-87-                                         reorder_target=16)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8917-tests/parallel/test_spmd_schedule_cost.py-88-    assert split_for_16["reorder_target"] == 16
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8918-tests/parallel/test_spmd_schedule_cost.py-89-    assert split_for_16["n_dev"] == 4
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8919-tests/parallel/test_spmd_schedule_cost.py-90-    # Non-vacuity: a split built for 16 devices really is a different split.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:8920:tests/parallel/test_spmd_schedule_cost.py-91-    assert split_for_16["n_rounds"] != same["n_rounds"] or (
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8921-tests/parallel/test_spmd_schedule_cost.py-92-        split_for_16["max_local_cells"] != same["max_local_cells"]), (
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8922-tests/parallel/test_spmd_schedule_cost.py-93-        "reorder_target had no effect — the argument would be decorative")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8923-tests/parallel/test_spmd_schedule_cost.py-94-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8924-tests/parallel/test_spmd_schedule_cost.py-95-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8925-tests/parallel/test_spmd_schedule_cost.py-96-def test_repeatable(mesh):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8926-tests/parallel/test_spmd_schedule_cost.py:97:    a = sd.spmd_schedule_cost(mesh, 8, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8927-tests/parallel/test_spmd_schedule_cost.py:98:    b = sd.spmd_schedule_cost(mesh, 8, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8928-tests/parallel/test_spmd_schedule_cost.py-99-    assert a == b, "same inputs must give the same score"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8929-tests/parallel/test_spmd_schedule_cost.py-100-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8930-tests/parallel/test_spmd_schedule_cost.py-101-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8931-tests/parallel/test_spmd_schedule_cost.py-102-@pytest.mark.slow
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:8932:tests/parallel/test_spmd_schedule_cost.py-103-def test_sfc_beats_metis_and_geometric_on_rounds():
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8933---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8934-tests/parallel/test_spmd_schedule_cost.py-110-    too coarse to discriminate — so this uses subdiv-5@16 and asserts the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8935-tests/parallel/test_spmd_schedule_cost.py-111-    scores actually differ before asserting their order. ~100 s, hence slow.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8936-tests/parallel/test_spmd_schedule_cost.py-112-    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8937-tests/parallel/test_spmd_schedule_cost.py-113-    pytest.importorskip("pymetis", reason="metis arm needs pymetis")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8938-tests/parallel/test_spmd_schedule_cost.py-114-    big = create_voronoi_mesh(subdivision_level=5)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:8939:tests/parallel/test_spmd_schedule_cost.py:115:    r = {m: sd.spmd_schedule_cost(big, 16, method=m)["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8940-tests/parallel/test_spmd_schedule_cost.py-116-         for m in ("sfc", "metis", "geometric")}
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8941-tests/parallel/test_spmd_schedule_cost.py-117-    assert len(set(r.values())) > 1, (
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8942-tests/parallel/test_spmd_schedule_cost.py-118-        f"all methods scored identically ({r}) — the comparison is vacuous "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8943-tests/parallel/test_spmd_schedule_cost.py-119-        f"at this mesh size; use a finer mesh or more devices")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8944-tests/parallel/test_spmd_schedule_cost.py-120-    assert r["sfc"] <= r["metis"] and r["sfc"] <= r["geometric"], r
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8945-tests/parallel/test_spmd_schedule_cost.py-121-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8946-tests/parallel/test_spmd_schedule_cost.py-122-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8947-tests/parallel/test_spmd_schedule_cost.py-123-def test_unknown_method_raises(mesh):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8948-tests/parallel/test_spmd_schedule_cost.py-124-    with pytest.raises(ValueError, match="Unknown partitioning method"):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8949-tests/parallel/test_spmd_schedule_cost.py:125:        sd.spmd_schedule_cost(mesh, 4, method="not_a_method")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8950-tests/parallel/test_spmd_schedule_cost.py-126-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-8951-tests/parallel/test_spmd_schedule_cost.py-127-
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9195---
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9196-tests/parallel/test_wet_band_boundaries.py-242-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9197-tests/parallel/test_wet_band_boundaries.py-243-    grid = create_latlon_grid(n_lat=48, n_lon=96)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9198-tests/parallel/test_wet_band_boundaries.py-244-    _, mask = load_bathymetry_latlon_cgrid(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9199-tests/parallel/test_wet_band_boundaries.py-245-        grid, BathymetryConfig(source="file", path=str(bathy)))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9200-tests/parallel/test_wet_band_boundaries.py-246-    wet_rows = np.asarray(mask).sum(axis=1)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9201-tests/parallel/test_wet_band_boundaries.py:247:    n_ranks = 8
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9202-tests/parallel/test_wet_band_boundaries.py-248-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:9203:tests/parallel/test_wet_band_boundaries.py:249:    b1 = wet_band_boundaries(wet_rows, n_ranks, min_rows=2)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:9204:tests/parallel/test_wet_band_boundaries.py:250:    b2 = wet_band_boundaries(wet_rows, n_ranks, min_rows=2)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9205-tests/parallel/test_wet_band_boundaries.py-251-    assert b1 == b2, "wet boundaries must be deterministic"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9206-tests/parallel/test_wet_band_boundaries.py:252:    validate_band_boundaries(b1, n_ranks, 48)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9207-tests/parallel/test_wet_band_boundaries.py-253-    rows = np.diff(b1)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9208-tests/parallel/test_wet_band_boundaries.py-254-    assert rows.min() >= 2, "halo floor violated"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9209-tests/parallel/test_wet_band_boundaries.py-255-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9210-tests/parallel/test_wet_band_boundaries.py-256-    def imbalance(bounds):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9211-tests/parallel/test_wet_band_boundaries.py-257-        wet = np.array([wet_rows[bounds[r]:bounds[r + 1]].sum()
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9212-tests/parallel/test_wet_band_boundaries.py:258:                        for r in range(n_ranks)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9213-tests/parallel/test_wet_band_boundaries.py-259-        return float(wet.max() / max(wet.mean(), 1.0))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9214-tests/parallel/test_wet_band_boundaries.py-260-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9215-tests/parallel/test_wet_band_boundaries.py:261:    base, rem = divmod(48, n_ranks)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-9216-tests/parallel/test_wet_band_boundaries.py-262-    even = tuple(np.concatenate(
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10414-    71	        "    sys.exit(0)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10415-    72	        "with open(log, 'a') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10416-    73	        "    f.write(json.dumps(argv) + '\\n')\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10417-    74	        "n = sum(1 for _ in open(log))\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10418-    75	        "rc = codes[n - 1] if n <= len(codes) else 0\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10419-    76	        "if write_artifact and rc == 0:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10420-    77	        "    out = argv[argv.index('--out') + 1]\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10421-    78	        "    with open(out, 'w') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:10422:    79	        "        json.dump({'rows': [{'n_ranks': 64}]}, f)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10423-    80	        "sys.exit(rc)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10424-    81	    )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10425-    82	    stub.chmod(0o755)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10426-    83	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10427-    84	    # Point REPO at a throwaway tree that only SYMLINKS the real scripts.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10428-    85	    # The launcher cd's to $REPO and writes results/a1/mpas_schedule_cost
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10429-    86	    # RELATIVE to it, so running these tests against the real repo would
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10430-    87	    # create — and, in the stale-artifact test, DELETE — files in the same
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10431-    88	    # directory a live scan writes its receipts to (codex round 6).
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10432-    89	    fake_repo = tmp_path / "repo"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10433-    90	    fake_repo.mkdir(exist_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10434-    91	    link = fake_repo / "scripts"
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10550-   207	    stale receipt read as current is worse than a missing one.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10551-   208	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10552-   209	    Runs entirely inside the throwaway repo tree (see ``_run``), so it can
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10553-   210	    never disturb the directory a live scan writes real receipts to.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10554-   211	    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10555-   212	    out_dir = tmp_path / "repo" / "results" / "a1" / "mpas_schedule_cost"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10556-   213	    out_dir.mkdir(parents=True, exist_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10557-   214	    stale = out_dir / "schedule_cost_s8.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:10558:   215	    stale.write_text('{"rows": [{"n_ranks": 999, "STALE": true}]}')
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10559-   216	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10560-   217	    # Arm 1 fails and writes nothing; the stale file must be GONE, not left
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10561-   218	    # behind looking like this run's result.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10562-   219	    proc, _, _out = _run(tmp_path, [1, 0, 0], write_artifact=False)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10563-   220	    assert proc.returncode != 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10564-   221	    assert not stale.exists(), (
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10565-   222	        "stale s8 JSON survived a failed arm and would read as current")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10566-   223	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10567-   224	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10568-   225	def test_artifact_check_parses_json_and_rejects_an_empty_rows_list(tmp_path):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10569-   226	    """The guard must PARSE, not grep.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10570-   227	
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10588-   245	        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10589-   246	        "    sys.exit(0)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10590-   247	        "with open(log, 'a') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10591-   248	        "    f.write(json.dumps(argv) + '\\n')\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10592-   249	        # Exit 0 having written a decoy: valid JSON, mentions the key, but
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10593-   250	        # carries no scored row.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10594-   251	        "out = argv[argv.index('--out') + 1]\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10595-   252	        "with open(out, 'w') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:10596:   253	        "    json.dump({'rows': [], 'note': 'n_ranks'}, f)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10597-   254	        "sys.exit(0)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10598-   255	    )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10599-   256	    stub.chmod(0o755)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10600-   257	    fake_repo = tmp_path / "repo"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10601-   258	    fake_repo.mkdir(exist_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10602-   259	    if not (fake_repo / "scripts").exists():
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10603-   260	        (fake_repo / "scripts").symlink_to(_REPO / "scripts")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10604-   261	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10605-   262	    env = dict(os.environ)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10606-   263	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10607-   264	    env["LEGOESM_REPO"] = str(fake_repo)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10608-   265	    env["LEGOESM_PYTHON"] = str(stub)
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10788-   135	  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10789-   136	  # An exit code alone is not evidence the arm produced anything (CLAUDE.md:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10790-   137	  # "tool status is not evidence"). Require the artifact to exist and carry
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10791-   138	  # rows, so a $PY that silently does nothing cannot yield SCAN_DONE with no
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10792-   139	  # results. This does not make the launcher unspoofable — $PY and $REPO come
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10793-   140	  # from _env.sh and remain configurable, as every launcher here is — but it
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10794-   141	  # does make "succeeded without measuring" DETECTABLE.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10795-   142	  # PARSE the JSON — a substring grep for "n_ranks" passes on
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:10796:   143	  # {"rows": [], "note": "n_ranks"}, which is exactly the empty-but-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10797-   144	  # successful artifact this is supposed to catch (codex round 6).
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10798-   145	  if [ "$rc" -eq 0 ] && ! "$PY" -c '
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10799-   146	import json, sys
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10800-   147	try:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10801-   148	    doc = json.load(open(sys.argv[1]))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10802-   149	except Exception:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10803-   150	    sys.exit(1)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10804-   151	rows = doc.get("rows") or []
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10805-   152	sys.exit(0 if any("n_ranks" in r for r in rows) else 1)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10806-   153	' "$json"; then
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:10807:   154	    echo "[scan] arm $3 exited 0 but $json is missing, unparseable, or has"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10808-   155	    echo "[scan] no scored rows — refusing to call that a result"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10809-   156	    return 90
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10810-   157	  fi
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10811-   158	  return $rc
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10812-   159	}
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10813-   160	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10814-   161	# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10815-   162	# makes the bench exit non-zero unless it reproduces the census below, and a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10816-   163	# listed pair that never got scored (e.g. pymetis missing) counts as a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10817-   164	# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10818-   165	# pass — an instrument that misses the known answer cannot be trusted on the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-10819-   166	# unknown one.
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11086-+def check_expected_rounds(rows: list, expect: dict) -> list:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11087-+    """Compare scored rounds against *expect*; return failure strings.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11088-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11089-+    A listed pair that was never scored is a FAILURE, not a skip: otherwise
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11090-+    a sweep that silently dropped a method (unavailable ``pymetis``) or a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11091-+    rank count would still report a clean gate.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11092-+    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11093-+    scored = {
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:11094:+        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11095-+        for r in rows
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11096-+        if r.get("available") and "schedule" in r and "n_ranks" in r
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11097-+    }
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11098-+    failures = []
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11099-+    for (method, n_ranks), want in sorted(expect.items()):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11100-+        got = scored.get((method, n_ranks))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11101-+        if got is None:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11102-+            failures.append(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11103-+                f"{method}:{n_ranks} expected rounds={want} but the pair was "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11104-+                f"NOT SCORED (method unavailable, or not in this sweep)")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11105-+        elif got != want:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11106-+            failures.append(
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11582-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11583-+    out2 = tmp_path / "on.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11584-+    monkeypatch.setattr(sys, "argv", [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11585-+        "bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11586-+        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11587-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11588-+    payload2 = json.loads(out2.read_text())
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11589-+    row = payload2["rows"][0]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:11590:+    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11591-+    assert payload2["metadata"]["extra"]["schedule_cost"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11592-+    # Scorer provenance per row: a copied row must show it scored a mesh
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11593-+    # partitioned for THIS device count, not one reordered for another.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11594-+    assert row["schedule"]["reorder_target"] == row["n_ranks"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11595-+    assert row["schedule"]["already_reordered"] is False
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11596-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11597-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11598-+def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11599-+    """``--lloyd`` must actually select the mesh, not just be recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11600-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11601-+    Non-vacuity: asserting only the recorded default (50) passes even if the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11602-+    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11633-+    that was never scored — a gate that can only pass is not a gate."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11634-+    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11635-+            "--methods", "geometric", "--schedule-cost"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11636-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11637-+    # Truth first: read what this configuration really scores.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11638-+    out = tmp_path / "truth.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11639-+    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11640-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:11641:+    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11642-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11643-+    # Matching expectation -> pass, and the check is recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11644-+    ok = tmp_path / "ok.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11645-+    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11646-+        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11647-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11648-+    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11649-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11650-+    # Wrong expectation -> non-zero exit.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11651-+    bad = tmp_path / "bad.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11652-+    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11653-+        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11831-   207	    stale receipt read as current is worse than a missing one.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11832-   208	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11833-   209	    Runs entirely inside the throwaway repo tree (see ``_run``), so it can
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11834-   210	    never disturb the directory a live scan writes real receipts to.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11835-   211	    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11836-   212	    out_dir = tmp_path / "repo" / "results" / "a1" / "mpas_schedule_cost"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11837-   213	    out_dir.mkdir(parents=True, exist_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11838-   214	    stale = out_dir / "schedule_cost_s8.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:11839:   215	    stale.write_text('{"rows": [{"n_ranks": 999, "STALE": true}]}')
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11840-   216	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11841-   217	    # Arm 1 fails and writes nothing; the stale file must be GONE, not left
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11842-   218	    # behind looking like this run's result.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11843-   219	    proc, _, _out = _run(tmp_path, [1, 0, 0], write_artifact=False)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11844-   220	    assert proc.returncode != 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11845-   221	    assert not stale.exists(), (
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11846-   222	        "stale s8 JSON survived a failed arm and would read as current")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11847-   223	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11848-   224	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11849-   225	def test_artifact_check_parses_json_and_rejects_an_empty_rows_list(tmp_path):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11850-   226	    """The guard must PARSE, not grep.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11851-   227	
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11869-   245	        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11870-   246	        "    sys.exit(0)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11871-   247	        "with open(log, 'a') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11872-   248	        "    f.write(json.dumps(argv) + '\\n')\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11873-   249	        # Exit 0 having written a decoy: valid JSON, mentions the key, but
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11874-   250	        # carries no scored row.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11875-   251	        "out = argv[argv.index('--out') + 1]\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11876-   252	        "with open(out, 'w') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:11877:   253	        "    json.dump({'rows': [], 'note': 'n_ranks'}, f)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11878-   254	        "sys.exit(0)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11879-   255	    )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11880-   256	    stub.chmod(0o755)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11881-   257	    fake_repo = tmp_path / "repo"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11882-   258	    fake_repo.mkdir(exist_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11883-   259	    if not (fake_repo / "scripts").exists():
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11884-   260	        (fake_repo / "scripts").symlink_to(_REPO / "scripts")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11885-   261	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11886-   262	    env = dict(os.environ)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11887-   263	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11888-   264	    env["LEGOESM_REPO"] = str(fake_repo)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-11889-   265	    env["LEGOESM_PYTHON"] = str(stub)
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12413-   234	def check_expected_rounds(rows: list, expect: dict) -> list:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12414-   235	    """Compare scored rounds against *expect*; return failure strings.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12415-   236	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12416-   237	    A listed pair that was never scored is a FAILURE, not a skip: otherwise
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12417-   238	    a sweep that silently dropped a method (unavailable ``pymetis``) or a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12418-   239	    rank count would still report a clean gate.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12419-   240	    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12420-   241	    scored = {
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:12421:   242	        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12422-   243	        for r in rows
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12423-   244	        if r.get("available") and "schedule" in r and "n_ranks" in r
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12424-   245	    }
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12425-   246	    failures = []
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12426-   247	    for (method, n_ranks), want in sorted(expect.items()):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12427-   248	        got = scored.get((method, n_ranks))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12428-   249	        if got is None:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12429-   250	            failures.append(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12430-   251	                f"{method}:{n_ranks} expected rounds={want} but the pair was "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12431-   252	                f"NOT SCORED (method unavailable, or not in this sweep)")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12432-   253	        elif got != want:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12433-   254	            failures.append(
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12685-./packages/ocean/legoesm/ocean/coupler/omip2_applicator.py:330:    q_sfc = np.asarray(_bolton_q_sat(jnp.asarray(T_sfc_K)), dtype=np.float64)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12686-./packages/ocean/legoesm/ocean/coupler/omip2_applicator.py:485:        T_sfc_K = np.asarray(state.T.data, dtype=np.float64)[..., 0] + constants.T_freeze
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12687-./packages/ocean/legoesm/ocean/coupler/omip2_applicator.py:538:        T_sfc_K = np.asarray(state.T.data, dtype=np.float64)[:, 0] + constants.T_freeze
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12688-./packages/ocean/legoesm/ocean/coupler/omip2_applicator.py:697:    T_sfc_K = np.asarray(state.T.data[..., 0], dtype=np.float64) + float(constants.T_freeze)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12689-./packages/ocean/legoesm/ocean/coupler/omip2_applicator.py:964:    T_sfc_K = np.asarray(state.T.data[..., 0], dtype=np.float64) + T_freeze
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12690-./tests/parallel/test_spmd_schedule_cost.py:1:"""Direct tests for ``sharded_dynamics.spmd_schedule_cost``.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12691-./tests/parallel/test_spmd_schedule_cost.py:22:    c = sd.spmd_schedule_cost(mesh, 4)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12692-./tests/parallel/test_spmd_schedule_cost.py:40:    c = sd.spmd_schedule_cost(mesh, 8)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:12693:./tests/parallel/test_spmd_schedule_cost.py:46:    assert sd.spmd_schedule_cost(mesh, 1)["n_rounds"] == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12694-./tests/parallel/test_spmd_schedule_cost.py:48:        sd.spmd_schedule_cost(mesh, 0)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12695-./tests/parallel/test_spmd_schedule_cost.py:54:    c = sd.spmd_schedule_cost(mesh, 8,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12696-./tests/parallel/test_spmd_schedule_cost.py:57:    big = sd.spmd_schedule_cost(mesh, 8,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12697-./tests/parallel/test_spmd_schedule_cost.py:67:    pre = sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12698-./tests/parallel/test_spmd_schedule_cost.py:68:    raw = sd.spmd_schedule_cost(mesh, 8, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12699-./tests/parallel/test_spmd_schedule_cost.py:77:        sd.spmd_schedule_cost(prepared, 8, already_reordered=True,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12700-./tests/parallel/test_spmd_schedule_cost.py:85:    same = sd.spmd_schedule_cost(mesh, 4, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12701-./tests/parallel/test_spmd_schedule_cost.py:86:    split_for_16 = sd.spmd_schedule_cost(mesh, 4, method="sfc",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12702-./tests/parallel/test_spmd_schedule_cost.py:97:    a = sd.spmd_schedule_cost(mesh, 8, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12703-./tests/parallel/test_spmd_schedule_cost.py:98:    b = sd.spmd_schedule_cost(mesh, 8, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12704-./tests/parallel/test_spmd_schedule_cost.py:105:    production sizes: subdiv-8 sfc 12/14 rounds at 64/128 devices vs metis
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:12705:./tests/parallel/test_spmd_schedule_cost.py:115:    r = {m: sd.spmd_schedule_cost(big, 16, method=m)["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12706-./tests/parallel/test_spmd_schedule_cost.py:125:        sd.spmd_schedule_cost(mesh, 4, method="not_a_method")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12707-./tests/parallel/test_spmd_schedule_cost.py:138:    sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12708-./tests/parallel/test_spmd_schedule_cost.py:147:        sd.spmd_schedule_cost(mesh, 4, method="sfc", reorder_target=3)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12709-./tests/parallel/test_spmd_schedule_cost.py:152:        sd.spmd_schedule_cost(mesh, 3.9)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12710-./tests/parallel/test_spmd_schedule_cost.py:160:    assert (inspect.signature(sd.spmd_schedule_cost)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12711-./tests/parallel/test_spmd_schedule_cost.py:169:    assert sd.spmd_schedule_cost(mesh, 1)["production_strategy"] is None
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12712-./tests/parallel/test_operator_split_spmd_carry_aux_export.py:130:    ser_sw = np.asarray(ser._carry_aux["held_sw_net_sfc"], dtype=np.float64)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12713-./scripts/matrix/ocean_test_matrix/experiments.py:1547:    S_init_sfc = np.asarray(state.S.data[..., 0], dtype=np.float64)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12714-./scripts/matrix/ocean_test_matrix/experiments.py:1563:        S_sfc = np.asarray(s.S.data[..., 0], dtype=np.float64)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12715-./scripts/matrix/ocean_test_matrix/diagnostic_io.py:344:                    u_raw = np.asarray(snapshots[step]["u_sfc"], dtype=np.float64)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12716-./scripts/matrix/ocean_test_matrix/diagnostic_io.py:345:                    v_raw = np.asarray(snapshots[step]["v_sfc"], dtype=np.float64)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-12717-./scripts/matrix/ocean_test_matrix/extraction.py:21:    u_sfc = np.asarray(state.u.data[..., 0], dtype=np.float64)
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13226-    38	    max_degree lower bound. Callers must be able to check equality rather than
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13227-    39	    assume it, so max_degree is returned alongside n_rounds."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13228-    40	    c = sd.spmd_schedule_cost(mesh, 8)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13229-    41	    assert "max_degree" in c and c["max_degree"] > 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13230-    42	    assert c["n_rounds"] >= c["max_degree"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13231-    43	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13232-    44	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13233-    45	def test_single_device_is_zero_rounds_and_zero_is_refused(mesh):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:13234:    46	    assert sd.spmd_schedule_cost(mesh, 1)["n_rounds"] == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13235-    47	    with pytest.raises(ValueError, match="integer >= 1"):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13236-    48	        sd.spmd_schedule_cost(mesh, 0)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13237-    49	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13238-    50	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13239-    51	def test_flags_allgather_when_production_would_not_use_ppermute(mesh):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13240-    52	    """Production auto-selects allgather below a cells/device threshold; the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13241-    53	    ppermute round count is then counterfactual and must say so."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13242-    54	    c = sd.spmd_schedule_cost(mesh, 8,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13243-    55	                              ppermute_cells_per_device_threshold=10**9)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13244-    56	    assert c["production_strategy"] == "allgather"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13245-    57	    big = sd.spmd_schedule_cost(mesh, 8,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13246-    58	                                ppermute_cells_per_device_threshold=1)
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13295-   107	    still tell the methods apart.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13296-   108	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13297-   109	    subdiv-4@8 and subdiv-5@8 score all three methods identically (7 rounds) —
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13298-   110	    too coarse to discriminate — so this uses subdiv-5@16 and asserts the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13299-   111	    scores actually differ before asserting their order. ~100 s, hence slow.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13300-   112	    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13301-   113	    pytest.importorskip("pymetis", reason="metis arm needs pymetis")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13302-   114	    big = create_voronoi_mesh(subdivision_level=5)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:13303:   115	    r = {m: sd.spmd_schedule_cost(big, 16, method=m)["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13304-   116	         for m in ("sfc", "metis", "geometric")}
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13305-   117	    assert len(set(r.values())) > 1, (
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13306-   118	        f"all methods scored identically ({r}) — the comparison is vacuous "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13307-   119	        f"at this mesh size; use a finer mesh or more devices")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13308-   120	    assert r["sfc"] <= r["metis"] and r["sfc"] <= r["geometric"], r
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13309-   121	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13310-   122	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13311-   123	def test_unknown_method_raises(mesh):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13312-   124	    with pytest.raises(ValueError, match="Unknown partitioning method"):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13313-   125	        sd.spmd_schedule_cost(mesh, 4, method="not_a_method")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13314-   126	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-13315-   127	
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14046-+    that was never scored — a gate that can only pass is not a gate."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14047-+    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14048-+            "--methods", "geometric", "--schedule-cost"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14049-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14050-+    # Truth first: read what this configuration really scores.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14051-+    out = tmp_path / "truth.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14052-+    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14053-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:14054:+    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14055-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14056-+    # Matching expectation -> pass, and the check is recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14057-+    ok = tmp_path / "ok.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14058-+    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14059-+        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14060-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14061-+    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14062-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14063-+    # Wrong expectation -> non-zero exit.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14064-+    bad = tmp_path / "bad.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14065-+    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14066-+        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14599-    tempdir = _get_default_tempdir()
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14600-  File "/home/b/b381103/.local/share/uv/python/cpython-3.14.6-linux-x86_64-gnu/lib/python3.14/tempfile.py", line 222, in _get_default_tempdir
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14601-    raise FileNotFoundError(_errno.ENOENT,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14602-                            "No usable temporary directory found in %s" %
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14603-                            dirlist)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14604-FileNotFoundError: [Errno 2] No usable temporary directory found in ['/tmp', '/var/tmp', '/usr/tmp', '/work/bd1083/b309178/diffESM/legoesm_pg/legoESM']
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14605-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14606-codex
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:14607:- Blocker: the artifact “parse” guard still accepts malformed/non-results JSON. `{"rows": {"n_ranks": 1}}` passes because the check iterates dictionary keys; `{"rows": [{"n_ranks": 1}]}` also passes despite containing no schedule score. That contradicts “no scored rows” and can yield `SCAN_DONE` without a score. Require `rows` to be a list and a row to be a mapping containing at least `n_ranks` plus `schedule.n_rounds`. Add mutations for both cases. [mpas_schedule_cost_scan.sbatch:145](</work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:145>) [test_mpas_schedule_cost_scan_sbatch.py:225](</work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:225>)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14608-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14609-- Test gap: the launcher stub writes only `{"n_ranks": 64}` for every successful arm, and the tests never assert each arm’s exact rank-counts or method set. Thus a mutation that drops `128` or a method still passes the launcher tests; only a real benchmark run would fail its expectation gate. Assert `--rank-counts` is `64,128`, `64,128`, `128` and `--methods geometric,sfc,metis`. [test_mpas_schedule_cost_scan_sbatch.py:76](</work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:76>) [test_mpas_schedule_cost_scan_sbatch.py:166](</work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:166>)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14610-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14611-Round-6 checks otherwise hold: the decoy test genuinely executes the JSON `-c` branch and would fail under a grep regression; `$PY` is the same interpreter used for each arm; the fake repo redirects relative `results/` writes/deletes; and the exact six-row mappings match the independent scorer census. [sharded_dynamics.py:1323](</work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1323>) The two-line diagnostic and its updated substring assertion are correct. [mpas_schedule_cost_scan.sbatch:154](</work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:154>)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14612-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14613-Focused pytest could not run because this read-only environment has no usable temporary directory; shell syntax passed.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14614-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14615-NOT READY
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14616-tokens used
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14617-92,714
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md:14618:- Blocker: the artifact “parse” guard still accepts malformed/non-results JSON. `{"rows": {"n_ranks": 1}}` passes because the check iterates dictionary keys; `{"rows": [{"n_ranks": 1}]}` also passes despite containing no schedule score. That contradicts “no scored rows” and can yield `SCAN_DONE` without a score. Require `rows` to be a list and a row to be a mapping containing at least `n_ranks` plus `schedule.n_rounds`. Add mutations for both cases. [mpas_schedule_cost_scan.sbatch:145](</work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:145>) [test_mpas_schedule_cost_scan_sbatch.py:225](</work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:225>)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14619-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14620-- Test gap: the launcher stub writes only `{"n_ranks": 64}` for every successful arm, and the tests never assert each arm’s exact rank-counts or method set. Thus a mutation that drops `128` or a method still passes the launcher tests; only a real benchmark run would fail its expectation gate. Assert `--rank-counts` is `64,128`, `64,128`, `128` and `--methods geometric,sfc,metis`. [test_mpas_schedule_cost_scan_sbatch.py:76](</work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:76>) [test_mpas_schedule_cost_scan_sbatch.py:166](</work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:166>)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14621-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14622-Round-6 checks otherwise hold: the decoy test genuinely executes the JSON `-c` branch and would fail under a grep regression; `$PY` is the same interpreter used for each arm; the fake repo redirects relative `results/` writes/deletes; and the exact six-row mappings match the independent scorer census. [sharded_dynamics.py:1323](</work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1323>) The two-line diagnostic and its updated substring assertion are correct. [mpas_schedule_cost_scan.sbatch:154](</work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:154>)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14623-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14624-Focused pytest could not run because this read-only environment has no usable temporary directory; shell syntax passed.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14625-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r7_2026-08-07.md-14626-NOT READY
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-223-+def check_expected_rounds(rows: list, expect: dict) -> list:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-224-+    """Compare scored rounds against *expect*; return failure strings.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-225-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-226-+    A listed pair that was never scored is a FAILURE, not a skip: otherwise
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-227-+    a sweep that silently dropped a method (unavailable ``pymetis``) or a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-228-+    rank count would still report a clean gate.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-229-+    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-230-+    scored = {
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:231:+        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-232-+        for r in rows
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-233-+        if r.get("available") and "schedule" in r and "n_ranks" in r
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-234-+    }
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-235-+    failures = []
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-236-+    for (method, n_ranks), want in sorted(expect.items()):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-237-+        got = scored.get((method, n_ranks))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-238-+        if got is None:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-239-+            failures.append(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-240-+                f"{method}:{n_ranks} expected rounds={want} but the pair was "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-241-+                f"NOT SCORED (method unavailable, or not in this sweep)")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-242-+        elif got != want:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-243-+            failures.append(
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-597-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-598-+    out2 = tmp_path / "on.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-599-+    monkeypatch.setattr(sys, "argv", [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-600-+        "bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-601-+        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-602-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-603-+    payload2 = json.loads(out2.read_text())
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-604-+    row = payload2["rows"][0]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:605:+    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-606-+    assert payload2["metadata"]["extra"]["schedule_cost"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-607-+    # Scorer provenance per row: a copied row must show it scored a mesh
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-608-+    # partitioned for THIS device count, not one reordered for another.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-609-+    assert row["schedule"]["reorder_target"] == row["n_ranks"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-610-+    assert row["schedule"]["already_reordered"] is False
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-611-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-612-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-613-+def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-614-+    """``--lloyd`` must actually select the mesh, not just be recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-615-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-616-+    Non-vacuity: asserting only the recorded default (50) passes even if the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-617-+    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-648-+    that was never scored — a gate that can only pass is not a gate."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-649-+    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-650-+            "--methods", "geometric", "--schedule-cost"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-651-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-652-+    # Truth first: read what this configuration really scores.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-653-+    out = tmp_path / "truth.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-654-+    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-655-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:656:+    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-657-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-658-+    # Matching expectation -> pass, and the check is recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-659-+    ok = tmp_path / "ok.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-660-+    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-661-+        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-662-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-663-+    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-664-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-665-+    # Wrong expectation -> non-zero exit.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-666-+    bad = tmp_path / "bad.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-667-+    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-668-+        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1041-+        "    sys.exit(0)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1042-+        "with open(log, 'a') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1043-+        "    f.write(json.dumps(argv) + '\\n')\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1044-+        "n = sum(1 for _ in open(log))\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1045-+        "rc = codes[n - 1] if n <= len(codes) else 0\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1046-+        "if write_artifact and rc == 0:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1047-+        "    out = argv[argv.index('--out') + 1]\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1048-+        "    with open(out, 'w') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:1049:+        "        json.dump({'rows': [{'n_ranks': 64}]}, f)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1050-+        "sys.exit(rc)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1051-+    )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1052-+    stub.chmod(0o755)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1053-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1054-+    env = dict(os.environ)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1055-+    env["SLURM_SUBMIT_DIR"] = str(_REPO)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1056-+    env["LEGOESM_REPO"] = str(_REPO)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1057-+    env["LEGOESM_PYTHON"] = str(stub)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1058-+    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1059-+    proc = subprocess.run(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1060-+        ["bash", str(_SBATCH)], env=env, cwd=str(_REPO),
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1061-+        capture_output=True, text=True, timeout=600)
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1161-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1162-+def test_stale_artifact_from_a_previous_run_is_removed_before_each_arm(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1163-+        tmp_path):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1164-+    """A failed rerun must not leave the previous run's JSON in place: a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1165-+    stale receipt read as current is worse than a missing one."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1166-+    out_dir = _REPO / "results" / "a1" / "mpas_schedule_cost"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1167-+    out_dir.mkdir(parents=True, exist_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1168-+    stale = out_dir / "schedule_cost_s8.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:1169:+    stale.write_text('{"rows": [{"n_ranks": 999, "STALE": true}]}')
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1170-+    try:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1171-+        # Arm 1 fails and writes nothing; the stale file must be gone, not
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1172-+        # left behind looking like this run's result.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1173-+        proc, _ = _run(tmp_path, [1, 0, 0], write_artifact=False)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1174-+        assert proc.returncode != 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1175-+        assert not stale.exists(), (
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1176-+            "stale s8 JSON survived a failed arm and would read as current")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1177-+    finally:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1178-+        stale.unlink(missing_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1179-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1180-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1181-+def test_launcher_expectations_are_the_full_six_row_census():
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1447-+def check_expected_rounds(rows: list, expect: dict) -> list:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1448-+    """Compare scored rounds against *expect*; return failure strings.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1449-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1450-+    A listed pair that was never scored is a FAILURE, not a skip: otherwise
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1451-+    a sweep that silently dropped a method (unavailable ``pymetis``) or a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1452-+    rank count would still report a clean gate.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1453-+    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1454-+    scored = {
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:1455:+        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1456-+        for r in rows
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1457-+        if r.get("available") and "schedule" in r and "n_ranks" in r
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1458-+    }
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1459-+    failures = []
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1460-+    for (method, n_ranks), want in sorted(expect.items()):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1461-+        got = scored.get((method, n_ranks))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1462-+        if got is None:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1463-+            failures.append(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1464-+                f"{method}:{n_ranks} expected rounds={want} but the pair was "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1465-+                f"NOT SCORED (method unavailable, or not in this sweep)")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1466-+        elif got != want:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1467-+            failures.append(
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1943-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1944-+    out2 = tmp_path / "on.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1945-+    monkeypatch.setattr(sys, "argv", [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1946-+        "bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1947-+        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1948-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1949-+    payload2 = json.loads(out2.read_text())
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1950-+    row = payload2["rows"][0]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:1951:+    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1952-+    assert payload2["metadata"]["extra"]["schedule_cost"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1953-+    # Scorer provenance per row: a copied row must show it scored a mesh
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1954-+    # partitioned for THIS device count, not one reordered for another.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1955-+    assert row["schedule"]["reorder_target"] == row["n_ranks"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1956-+    assert row["schedule"]["already_reordered"] is False
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1957-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1958-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1959-+def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1960-+    """``--lloyd`` must actually select the mesh, not just be recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1961-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1962-+    Non-vacuity: asserting only the recorded default (50) passes even if the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1963-+    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1994-+    that was never scored — a gate that can only pass is not a gate."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1995-+    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1996-+            "--methods", "geometric", "--schedule-cost"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1997-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1998-+    # Truth first: read what this configuration really scores.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-1999-+    out = tmp_path / "truth.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2000-+    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2001-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:2002:+    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2003-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2004-+    # Matching expectation -> pass, and the check is recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2005-+    ok = tmp_path / "ok.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2006-+    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2007-+        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2008-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2009-+    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2010-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2011-+    # Wrong expectation -> non-zero exit.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2012-+    bad = tmp_path / "bad.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2013-+    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2014-+        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2375-    64	        "    sys.exit(0)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2376-    65	        "with open(log, 'a') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2377-    66	        "    f.write(json.dumps(argv) + '\\n')\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2378-    67	        "n = sum(1 for _ in open(log))\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2379-    68	        "rc = codes[n - 1] if n <= len(codes) else 0\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2380-    69	        "if write_artifact and rc == 0:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2381-    70	        "    out = argv[argv.index('--out') + 1]\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2382-    71	        "    with open(out, 'w') as f:\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:2383:    72	        "        json.dump({'rows': [{'n_ranks': 64}]}, f)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2384-    73	        "sys.exit(rc)\n"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2385-    74	    )
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2386-    75	    stub.chmod(0o755)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2387-    76	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2388-    77	    env = dict(os.environ)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2389-    78	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2390-    79	    env["LEGOESM_REPO"] = str(_REPO)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2391-    80	    env["LEGOESM_PYTHON"] = str(stub)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2392-    81	    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2393-    82	    proc = subprocess.run(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2394-    83	        ["bash", str(_SBATCH)], env=env, cwd=str(_REPO),
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2395-    84	        capture_output=True, text=True, timeout=600)
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2495-   184	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2496-   185	def test_stale_artifact_from_a_previous_run_is_removed_before_each_arm(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2497-   186	        tmp_path):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2498-   187	    """A failed rerun must not leave the previous run's JSON in place: a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2499-   188	    stale receipt read as current is worse than a missing one."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2500-   189	    out_dir = _REPO / "results" / "a1" / "mpas_schedule_cost"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2501-   190	    out_dir.mkdir(parents=True, exist_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2502-   191	    stale = out_dir / "schedule_cost_s8.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:2503:   192	    stale.write_text('{"rows": [{"n_ranks": 999, "STALE": true}]}')
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2504-   193	    try:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2505-   194	        # Arm 1 fails and writes nothing; the stale file must be gone, not
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2506-   195	        # left behind looking like this run's result.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2507-   196	        proc, _ = _run(tmp_path, [1, 0, 0], write_artifact=False)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2508-   197	        assert proc.returncode != 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2509-   198	        assert not stale.exists(), (
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2510-   199	            "stale s8 JSON survived a failed arm and would read as current")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2511-   200	    finally:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2512-   201	        stale.unlink(missing_ok=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2513-   202	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2514-   203	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-2515-   204	def test_launcher_expectations_are_the_full_six_row_census():
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3034-   269	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3035-   270	    out2 = tmp_path / "on.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3036-   271	    monkeypatch.setattr(sys, "argv", [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3037-   272	        "bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3038-   273	        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3039-   274	    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3040-   275	    payload2 = json.loads(out2.read_text())
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3041-   276	    row = payload2["rows"][0]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:3042:   277	    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3043-   278	    assert payload2["metadata"]["extra"]["schedule_cost"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3044-   279	    # Scorer provenance per row: a copied row must show it scored a mesh
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3045-   280	    # partitioned for THIS device count, not one reordered for another.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3046-   281	    assert row["schedule"]["reorder_target"] == row["n_ranks"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3047-   282	    assert row["schedule"]["already_reordered"] is False
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3048-   283	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3049-   284	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3050-   285	def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3051-   286	    """``--lloyd`` must actually select the mesh, not just be recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3052-   287	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3053-   288	    Non-vacuity: asserting only the recorded default (50) passes even if the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3054-   289	    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3085-   320	    that was never scored — a gate that can only pass is not a gate."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3086-   321	    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3087-   322	            "--methods", "geometric", "--schedule-cost"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3088-   323	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3089-   324	    # Truth first: read what this configuration really scores.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3090-   325	    out = tmp_path / "truth.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3091-   326	    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3092-   327	    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:3093:   328	    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3094-   329	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3095-   330	    # Matching expectation -> pass, and the check is recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3096-   331	    ok = tmp_path / "ok.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3097-   332	    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3098-   333	        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3099-   334	    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3100-   335	    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3101-   336	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3102-   337	    # Wrong expectation -> non-zero exit.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3103-   338	    bad = tmp_path / "bad.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3104-   339	    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3105-   340	        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3364-scripts/bench/bench_voronoi_partition_methods.py:295:    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3365-scripts/bench/bench_voronoi_partition_methods.py:298:    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3366-tests/bench/test_bench_voronoi_partition_methods.py:207:    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3367-tests/bench/test_bench_voronoi_partition_methods.py:246:    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3368-tests/bench/test_bench_voronoi_partition_methods.py:249:    ref = spmd_schedule_cost(mesh, 2, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3369-tests/parallel/test_spmd_schedule_cost.py:1:"""Direct tests for ``sharded_dynamics.spmd_schedule_cost``.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3370-tests/parallel/test_spmd_schedule_cost.py:22:    c = sd.spmd_schedule_cost(mesh, 4)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3371-tests/parallel/test_spmd_schedule_cost.py:40:    c = sd.spmd_schedule_cost(mesh, 8)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:3372:tests/parallel/test_spmd_schedule_cost.py:46:    assert sd.spmd_schedule_cost(mesh, 1)["n_rounds"] == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3373-tests/parallel/test_spmd_schedule_cost.py:48:        sd.spmd_schedule_cost(mesh, 0)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3374-tests/parallel/test_spmd_schedule_cost.py:54:    c = sd.spmd_schedule_cost(mesh, 8,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3375-tests/parallel/test_spmd_schedule_cost.py:57:    big = sd.spmd_schedule_cost(mesh, 8,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3376-tests/parallel/test_spmd_schedule_cost.py:67:    pre = sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3377-tests/parallel/test_spmd_schedule_cost.py:68:    raw = sd.spmd_schedule_cost(mesh, 8, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3378-tests/parallel/test_spmd_schedule_cost.py:77:        sd.spmd_schedule_cost(prepared, 8, already_reordered=True,
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3379-tests/parallel/test_spmd_schedule_cost.py:85:    same = sd.spmd_schedule_cost(mesh, 4, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3380-tests/parallel/test_spmd_schedule_cost.py:86:    split_for_16 = sd.spmd_schedule_cost(mesh, 4, method="sfc",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3381-tests/parallel/test_spmd_schedule_cost.py:97:    a = sd.spmd_schedule_cost(mesh, 8, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3382-tests/parallel/test_spmd_schedule_cost.py:98:    b = sd.spmd_schedule_cost(mesh, 8, method="sfc")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:3383:tests/parallel/test_spmd_schedule_cost.py:115:    r = {m: sd.spmd_schedule_cost(big, 16, method=m)["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3384-tests/parallel/test_spmd_schedule_cost.py:125:        sd.spmd_schedule_cost(mesh, 4, method="not_a_method")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3385-tests/parallel/test_spmd_schedule_cost.py:138:    sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3386-tests/parallel/test_spmd_schedule_cost.py:147:        sd.spmd_schedule_cost(mesh, 4, method="sfc", reorder_target=3)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3387-tests/parallel/test_spmd_schedule_cost.py:152:        sd.spmd_schedule_cost(mesh, 3.9)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3388-tests/parallel/test_spmd_schedule_cost.py:160:    assert (inspect.signature(sd.spmd_schedule_cost)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3389-tests/parallel/test_spmd_schedule_cost.py:169:    assert sd.spmd_schedule_cost(mesh, 1)["production_strategy"] is None
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3390-scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:70:# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3391-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3392-exec
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3393-/usr/bin/bash -lc "nl -ba tests/bench/test_bench_voronoi_partition_methods.py | sed -n '145,330p'; find . -path '*sharded_dynamics.py' -o -path '*voronoi.py' | head -40; rg -n -A150 -B15 \"def spmd_schedule_cost\" . --glob '*.py'; rg -n -A100 -B15 \"def create_voronoi_mesh\" . --glob '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3394-exec
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-3395-/usr/bin/bash -lc "bash -n scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && bash -c 'f(){ printf \"<%s>\\n\" "'${4:+--expect-rounds "$4"}; }; f 1 2 3 "sfc:64=12,sfc:128=14"'"' && git diff --check && git ls-files .github/workflows && rg -n \"tests/bench|pytest|bench\" .github/workflows .github 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4208-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4209-+    out2 = tmp_path / "on.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4210-+    monkeypatch.setattr(sys, "argv", [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4211-+        "bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4212-+        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4213-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4214-+    payload2 = json.loads(out2.read_text())
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4215-+    row = payload2["rows"][0]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:4216:+    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4217-+    assert payload2["metadata"]["extra"]["schedule_cost"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4218-+    # Scorer provenance per row: a copied row must show it scored a mesh
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4219-+    # partitioned for THIS device count, not one reordered for another.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4220-+    assert row["schedule"]["reorder_target"] == row["n_ranks"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4221-+    assert row["schedule"]["already_reordered"] is False
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4222-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4223-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4224-+def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4225-+    """``--lloyd`` must actually select the mesh, not just be recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4226-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4227-+    Non-vacuity: asserting only the recorded default (50) passes even if the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4228-+    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4259-+    that was never scored — a gate that can only pass is not a gate."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4260-+    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4261-+            "--methods", "geometric", "--schedule-cost"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4262-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4263-+    # Truth first: read what this configuration really scores.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4264-+    out = tmp_path / "truth.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4265-+    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4266-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:4267:+    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4268-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4269-+    # Matching expectation -> pass, and the check is recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4270-+    ok = tmp_path / "ok.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4271-+    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4272-+        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4273-+    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4274-+    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4275-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4276-+    # Wrong expectation -> non-zero exit.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4277-+    bad = tmp_path / "bad.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4278-+    monkeypatch.setattr(sys, "argv", base + [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4279-+        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4522-+def check_expected_rounds(rows: list, expect: dict) -> list:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4523-+    """Compare scored rounds against *expect*; return failure strings.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4524-+
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4525-+    A listed pair that was never scored is a FAILURE, not a skip: otherwise
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4526-+    a sweep that silently dropped a method (unavailable ``pymetis``) or a
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4527-+    rank count would still report a clean gate.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4528-+    """
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4529-+    scored = {
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:4530:+        (r["method"], r["n_ranks"]): r["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4531-+        for r in rows
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4532-+        if r.get("available") and "schedule" in r and "n_ranks" in r
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4533-+    }
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4534-+    failures = []
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4535-+    for (method, n_ranks), want in sorted(expect.items()):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4536-+        got = scored.get((method, n_ranks))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4537-+        if got is None:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4538-+            failures.append(
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4539-+                f"{method}:{n_ranks} expected rounds={want} but the pair was "
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4540-+                f"NOT SCORED (method unavailable, or not in this sweep)")
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4541-+        elif got != want:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4542-+            failures.append(
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4773-/usr/bin/bash: python: command not found
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4774-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4775-exec
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4776-/usr/bin/bash -lc 'for p in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python /usr/bin/python3 /bin/python3; do if [ -x "$p" ]; then "$p" -c "import ast; from pathlib import Path; paths=('"'scripts/bench/bench_voronoi_partition_methods.py','tests/bench/test_bench_voronoi_partition_methods.py','tests/bench/test_mpas_schedule_cost_scan_sbatch.py'); [ast.parse(Path(x).read_text(), filename=x) for x in paths]; print('AST OK')\"; exit "'$?; fi; done; echo '"'No Python interpreter available'; exit 127" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4777- succeeded in 411ms:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4778-AST OK
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4779-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4780-exec
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:4781:/usr/bin/bash -lc "if printf '%s\\n' '{\"rows\": [], \"note\": \"n_ranks\"}' | grep -q '\"n_ranks\"'; then echo 'grep predicate ACCEPTS empty rows'; else echo 'grep predicate rejects empty rows'; fi
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4782-PYTHONDONTWRITEBYTECODE=1 /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python -c \"import importlib.util; from pathlib import Path; p=Path('scripts/bench/bench_voronoi_partition_methods.py'); s=importlib.util.spec_from_file_location('b',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); x=m.parse_expect_rounds('sfc:64=999,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21'); print(sorted(x)); print(set(x)=={(k,n) for k in ('geometric','sfc','metis') for n in (64,128)})\"" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4783- succeeded in 653ms:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4784-grep predicate ACCEPTS empty rows
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4785-[('geometric', 64), ('geometric', 128), ('metis', 64), ('metis', 128), ('sfc', 64), ('sfc', 128)]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4786-True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4787-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4788- succeeded in 199779ms:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4789-   145	            mesh, 4, r, method="geometric", halo_depth=2, cell_owner=owner)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4790-   146	        halos.append(int(part.n_local_cells) - int(part.n_owned_cells))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4791-   147	    assert q["halo_cells_max"] == max(halos)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4792-   148	    assert q["halo_cells_mean"] == pytest.approx(float(np.mean(halos)))
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4793-   149	
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4913-   269	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4914-   270	    out2 = tmp_path / "on.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4915-   271	    monkeypatch.setattr(sys, "argv", [
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4916-   272	        "bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4917-   273	        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4918-   274	    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4919-   275	    payload2 = json.loads(out2.read_text())
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4920-   276	    row = payload2["rows"][0]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:4921:   277	    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4922-   278	    assert payload2["metadata"]["extra"]["schedule_cost"] is True
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4923-   279	    # Scorer provenance per row: a copied row must show it scored a mesh
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4924-   280	    # partitioned for THIS device count, not one reordered for another.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4925-   281	    assert row["schedule"]["reorder_target"] == row["n_ranks"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4926-   282	    assert row["schedule"]["already_reordered"] is False
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4927-   283	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4928-   284	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4929-   285	def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4930-   286	    """``--lloyd`` must actually select the mesh, not just be recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4931-   287	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4932-   288	    Non-vacuity: asserting only the recorded default (50) passes even if the
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4933-   289	    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4964-   320	    that was never scored — a gate that can only pass is not a gate."""
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4965-   321	    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4966-   322	            "--methods", "geometric", "--schedule-cost"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4967-   323	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4968-   324	    # Truth first: read what this configuration really scores.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4969-   325	    out = tmp_path / "truth.json"
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4970-   326	    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4971-   327	    assert mod.main() == 0
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:4972:   328	    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4973-   329	
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4974-   330	    # Matching expectation -> pass, and the check is recorded.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4975-./.venv-mpi/lib/python3.13/site-packages/scipy/spatial/_spherical_voronoi.py
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4976-./.venv-mpi/lib/python3.13/site-packages/scipy/spatial/tests/test_spherical_voronoi.py
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4977-./tests/atmosphere/test_column_large_scale_extract_voronoi.py
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4978-./tests/unit/test_sharded_dynamics.py
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4979-./tests/unit/test_scale_sharded_dynamics.py
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4980-./tests/unit/test_regional_voronoi.py
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4981-./.venv/lib/python3.14/site-packages/scipy/spatial/_spherical_voronoi.py
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4982-./.venv/lib/python3.14/site-packages/scipy/spatial/tests/test_spherical_voronoi.py
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4983-./.venv-gpumpi/lib/python3.14/site-packages/scipy/spatial/_spherical_voronoi.py
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-4984-./.venv-gpumpi/lib/python3.14/site-packages/scipy/spatial/tests/test_spherical_voronoi.py
--
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5285-./packages/core/legoesm/grids/voronoi.py-1340-            except FileExistsError:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5286-./packages/core/legoesm/grids/voronoi.py-1341-                for _ in range(int(_BIG_MESH_LOCK_STALE_S)):
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5287-./packages/core/legoesm/grids/voronoi.py-1342-                    _time.sleep(1.0)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5288-./packages/core/legoesm/grids/voronoi.py-1343-                    cached = _load_voronoi_cache(cache_path)
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5289-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5290-codex
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5291-NOT READY.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5292-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:5293:- The round-5 artifact guard is bypassable. [`run_arm`]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:142) accepts any file containing the literal `"n_ranks"`. Valid empty JSON such as `{"rows": [], "note": "n_ranks"}` passes, returns 0, and can reach `SCAN_DONE`. It neither parses JSON nor verifies a scored row. The new test only covers a missing file ([test:175]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:175)); replacing the guard with a mere existence test would still pass it. Require parsed JSON with a non-empty `rows` list containing real schedule rows.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5294-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5295-- The stale-artifact test is destructive to real ignored results. `_run` explicitly runs in the repository and sets the repo there ([test:77]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:77)), while the launcher writes the fixed `results/a1/mpas_schedule_cost` path ([launcher:100]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:100)). The test overwrites and finally deletes `schedule_cost_s8.json` without preserving an existing receipt ([test:189]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:189)). All shell tests leave or delete artifacts in that shared directory.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5296-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5297-- The six-row test catches a deleted/key-corrupted literal, but not a corrupted expected number. It only asserts `set(parsed)` ([test:224]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:224)); changing `sfc:64=12` to `sfc:64=999` still passes. The real benchmark gate would abort on that wrong value, but the claimed regression test does not. Assert each complete expected mapping.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5298-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5299-Verified correct:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5300-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5301-- The no-successful-measurement guard cannot fire on a legitimate completed row: every successful loop iteration has `available` and `n_ranks`; only all-unavailable runs abort ([bench:421]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:421), [bench:465]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:465)).
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5302-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5303-- Your `['8', '9']` expectation is correct. Both validation arms run before the failure gate is evaluated ([launcher:158]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:158)); arm 3 is then skipped ([launcher:161]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:161)). A validation artifact failure becomes final exit 1; an arm-3 artifact failure propagates 90 ([launcher:173]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:173)).
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5304-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5305-- (d) Report, don’t broaden CI here. The configured workflow collects all tests but executes `tests/unit/` and selected paths, not `tests/bench/` ([ci.yml:97]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:97), [ci.yml:121]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:121)). Given the stated repo-wide Actions disablement, a targeted CI wiring change is inert and out of scope.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5306-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5307-- (e) Keeping the shared `_env.sh` contract is the right scope call; `prewarm_s10` uses the same override pattern ([prewarm:18]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/prewarm_s10.sbatch:18)). But the claimed mitigation is false until the artifact check is semantic: it currently does not make “succeeded without measuring” detectable.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5308-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5309-`bash -n` and `git diff --check` passed.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5310-tokens used
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5311-96,896
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5312-NOT READY.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5313-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md:5314:- The round-5 artifact guard is bypassable. [`run_arm`]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:142) accepts any file containing the literal `"n_ranks"`. Valid empty JSON such as `{"rows": [], "note": "n_ranks"}` passes, returns 0, and can reach `SCAN_DONE`. It neither parses JSON nor verifies a scored row. The new test only covers a missing file ([test:175]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:175)); replacing the guard with a mere existence test would still pass it. Require parsed JSON with a non-empty `rows` list containing real schedule rows.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5315-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5316-- The stale-artifact test is destructive to real ignored results. `_run` explicitly runs in the repository and sets the repo there ([test:77]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:77)), while the launcher writes the fixed `results/a1/mpas_schedule_cost` path ([launcher:100]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:100)). The test overwrites and finally deletes `schedule_cost_s8.json` without preserving an existing receipt ([test:189]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:189)). All shell tests leave or delete artifacts in that shared directory.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5317-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5318-- The six-row test catches a deleted/key-corrupted literal, but not a corrupted expected number. It only asserts `set(parsed)` ([test:224]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:224)); changing `sfc:64=12` to `sfc:64=999` still passes. The real benchmark gate would abort on that wrong value, but the claimed regression test does not. Assert each complete expected mapping.
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5319-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5320-Verified correct:
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5321-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5322-- The no-successful-measurement guard cannot fire on a legitimate completed row: every successful loop iteration has `available` and `n_ranks`; only all-unavailable runs abort ([bench:421]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:421), [bench:465]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:465)).
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5323-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5324-- Your `['8', '9']` expectation is correct. Both validation arms run before the failure gate is evaluated ([launcher:158]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:158)); arm 3 is then skipped ([launcher:161]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:161)). A validation artifact failure becomes final exit 1; an arm-3 artifact failure propagates 90 ([launcher:173]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:173)).
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5325-
.physics-validator/scaling_campaign/codex_review_schedule_cost_r6_2026-08-07.md-5326-- (d) Report, don’t broaden CI here. The configured workflow collects all tests but executes `tests/unit/` and selected paths, not `tests/bench/` ([ci.yml:97]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:97), [ci.yml:121]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:121)). Given the stated repo-wide Actions disablement, a targeted CI wiring change is inert and out of scope.
   120	
   121	    if method == "geometric":
   122	        return np.asarray(partition_cells_geometric(mesh, n_ranks))
   123	    if method == "sfc":
   124	        return np.asarray(partition_cells_sfc(mesh, n_ranks))
   125	    if method == "metis":
   126	        return np.asarray(partition_cells_metis(mesh, n_ranks))
   127	    raise ValueError(f"unknown partition method {method!r}; "
   128	                     f"expected one of {METHODS}")
   129	
   130	
   131	def main() -> int:
   132	    p = argparse.ArgumentParser(
   133	        description=__doc__,
   134	        formatter_class=argparse.RawDescriptionHelpFormatter)
   135	    p.add_argument("--subdivision", type=int, default=5,
   136	                   help="Icosahedral level (L5=10,242 cells; L6=40,962).")
   137	    p.add_argument("--rank-counts", type=str, default="2,4,8,16")
   138	    p.add_argument("--halo-depth", type=int, default=2,
   139	                   help="Halo layers (runtime default 2, del4 support).")
   140	    p.add_argument("--methods", type=str, default=",".join(METHODS))
   141	    p.add_argument("--out", type=str,
   142	                   default="results/a1/voronoi_partition_quality.json")
   143	    args = p.parse_args()
   144	
   145	    rank_counts = [int(x) for x in args.rank_counts.split(",") if x]
   146	    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
   147	    for m in methods:
   148	        if m not in METHODS:
   149	            raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
   150	    if not rank_counts or any(n < 2 for n in rank_counts):
   151	        raise SystemExit("--rank-counts needs integers >= 2")
   152	
   153	
   154	    from legoesm.grids.voronoi import create_voronoi_mesh
   155	    from legoesm.parallel.voronoi_partition import resolve_partition_method
   156	
   157	    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
   158	    if max(rank_counts) > int(mesh.nCells):
   159	        raise SystemExit(
   160	            f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
   161	            f"{int(mesh.nCells)} cells (empty ranks are meaningless).")
   162	    print(f"mesh L{args.subdivision}: {int(mesh.nCells)} cells, "
   163	          f"{int(mesh.nEdges)} edges; auto -> "
   164	          f"{resolve_partition_method('auto')!r}")
   165	
   166	    rows = []
   167	    for method in methods:
   168	        if not method_available(method):
   169	            print(f"  {method:9s}: UNAVAILABLE (pymetis not importable) — "
   170	                  f"column omitted, never substituted")
   171	            rows.append({"method": method, "available": False})
   172	            continue
   173	        for n_ranks in rank_counts:
   174	            q = partition_quality(
   175	                mesh, owner_for(mesh, method, n_ranks), n_ranks,
   176	                halo_depth=args.halo_depth)
   177	            row = {"method": method, "available": True,
   178	                   "n_ranks": n_ranks, **q}
   179	            rows.append(row)
   180	            print(f"  {method:9s} np={n_ranks:3d} | "
   181	                  f"imbalance={q['load_imbalance_max_over_mean']:.3f} | "
   182	                  f"edge_cut={q['edge_cut']:6d} "
   183	                  f"({100 * q['edge_cut_fraction']:.2f}%) | "
   184	                  f"halo max={q['halo_cells_max']:5d} "
   185	                  f"mean={q['halo_cells_mean']:8.1f} | "
   186	                  f"halo/owned max={q['halo_owned_ratio_max']:.3f} | "
   187	                  f"nbrs max={q['neighbor_ranks_max']}")
   188	
   189	    payload = {
   190	        "rows": rows,
   191	        "auto_resolves_to": resolve_partition_method("auto"),
   192	        "step_time_pointer": (
   193	            "step-time per method: bench_ocean_mpas_scaling.py / "
   194	            "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
   195	            "per launch, same case otherwise)"),
   196	        "metadata": annotate_incomplete(scaling_metadata(
   197	            grid="voronoi",
   198	            component="partitioning",
   199	            resolution=f"L{args.subdivision}",
   200	            n_levels=0,
   201	            precision="n/a",
   202	            decomposition="cell_partition",
   203	            solver_variant="n/a",
   204	            scaling_kind="partition-quality",
   205	            transport="none",
   206	            extra={"rank_counts": rank_counts, "methods": methods,
   207	                   "halo_depth": args.halo_depth},
   208	        )),
   209	    }
   210	    outdir = os.path.dirname(args.out)
   211	    if outdir:
   212	        os.makedirs(outdir, exist_ok=True)
   213	    with open(args.out, "w") as f:
   214	        json.dump(payload, f, indent=2)
   215	    print(f"JSON: {args.out}")
   216	    return 0
   217	
   218	
   219	if __name__ == "__main__":
   220	    raise SystemExit(main())
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

exec
/usr/bin/bash -lc "rg -n -A260 -B40 '"'^def _build_ppermute_schedule'"' packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1,340p'; nl -ba tests/parallel/test_spmd_schedule_cost.py | sed -n '1,240p'; rg -n \"max_degree|edge colori|ppermute_schedule\" packages/core/legoesm/parallel/sharded_dynamics.py tests/parallel/test_spmd_schedule_cost.py | head -200" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 187ms:
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
1762:def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
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
1815-    comm_pairs: set[tuple[int, int]] = set()
1816-    for d in range(n_dev):
1817-        for d_prime in halo_cells_from[d]:
1818-            if d != d_prime:
1819-                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
1820-        for d_prime in halo_edges_from[d]:
1821-            if d != d_prime:
1822-                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
1823-
1824-    if not comm_pairs:
1825-        return {
1826-            'n_rounds': 0,
1827-            'n_rounds_greedy': 0,
1828-            'max_degree': 0,
1829-            'coloring_method': 'none',
1830-            'ppermute_perms': [],
1831-            'send_cell_idx': [],
1832-            'recv_cell_pos': [],
1833-            'send_edge_idx': [],
1834-            'recv_edge_pos': [],
1835-            'halo_cells_per_round': [],
1836-            'halo_edges_per_round': [],
1837-        }
1838-
1839-    # ------------------------------------------------------------------
1840-    # 3. Edge-color the graph: each color = one bidirectional ppermute
1841-    #    ROUND, and the route-B lane is round-latency-bound (#1113), so
1842-    #    fewer colors = directly less wall-clock. First-fit greedy is
1843-    #    order-sensitive; the multi-start coloring reaches the
1844-    #    chromatic-index floor (= max_degree) on every probed MPAS config
1845-    #    where the legacy sorted greedy overshoots (up to 3 rounds at 16
1846-    #    devices). It can never regress: the legacy sorted order is one of
1847-    #    its candidates and it takes the min. Both are verified proper.
1848-    # ------------------------------------------------------------------
1849-    greedy_colors = _greedy_edge_coloring(comm_pairs)
1850-    n_rounds_greedy = max(greedy_colors.values()) + 1
1851-    multi_colors, max_degree = _multi_ordering_edge_coloring(comm_pairs)
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
1864-    assert _check_proper_edge_coloring(edge_colors, comm_pairs), (
1865-        "improper ppermute edge coloring — two same-round exchanges "
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
1878-    edge_send_map: dict[tuple[int, int], list[int]] = {}
1879-    edge_recv_map: dict[tuple[int, int], list[int]] = {}
1880-
1881-    for d in range(n_dev):
1882-        for d_prime, cells_g in halo_cells_from[d].items():
1883-            if d_prime == d:
1884-                continue
1885-            # d_prime sends its owned cells that d needs as halo
1886-            cell_send_map[(d_prime, d)] = [
1887-                g - d_prime * cells_per for g in cells_g]
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
1900-    # 5. Assemble per-round ppermute patterns and index arrays
1901-    # ------------------------------------------------------------------
1902-    ppermute_perms_out: list[list[tuple[int, int]]] = []
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
1923-        # Bidirectional ppermute pattern
1924-        perm: list[tuple[int, int]] = []
1925-        partner: dict[int, int] = {}
1926-        for u, v in rounds[r]:
1927-            perm.append((u, v))
1928-            perm.append((v, u))
1929-            partner[u] = v
1930-            partner[v] = u
1931-        ppermute_perms_out.append(perm)
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
1944-
1945-            cs = cell_send_map.get((d, dp), [])
1946-            for j, idx in enumerate(cs):
1947-                sc[d, j] = idx
1948-
1949-            cr = cell_recv_map.get((d, dp), [])
1950-            for j, pos in enumerate(cr):
1951-                rc[d, j] = pos
1952-
1953-            es = edge_send_map.get((d, dp), [])
1954-            for j, idx in enumerate(es):
1955-                se[d, j] = idx
1956-
1957-            er = edge_recv_map.get((d, dp), [])
1958-            for j, pos in enumerate(er):
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
1971-        'ppermute_perms': ppermute_perms_out,
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
1984-# (stale halos on every RK stage) — fail loudly so the cell-pack layout,
1985-# the physics application and this set are extended deliberately.
1986-# Workload signatures — (n_devices, global edge rows, global cell rows,
1987-# nlev), all trace-time-static — where a tendency-output
1988-# optimization_barrier is measured to pay. OBSERVED CORRELATION, not a
1989-# proven XLA cost-model account: at ico-L8 np4 (and only there among
1990-# np2/4/8) the compiled step carries three once-per-step 3.3-3.6 ms
1991-# serialized loop-fusion kernels that the HLO frame table resolves to the
1992-# RK pytree_axpy (pytree_ops.py), ~10.9 ms/step in total (nsys 26479922,
1993-# HLO 26480096), and the barrier removes most of that: same-day ladder
1994-# np4 17.78 -> 14.10 ms (-20.7%) with np2 +1.2% (noise) and np8 0.0%
1995-# (jobs 26486123 dead-gate vs 26486163). An UNgated barrier regressed np2
1996-# by 10.7% in the earlier experiment (26480310), hence the gate. The
1997-# signature includes cell rows + nlev because L8's edge count
1998-# (1,966,080, unpadded — padding pads CELLS, e.g. 655,362 -> 655,376 for
1999-# 16) is divisible several ways and edge rows alone would fire on
2000-# unmeasured workloads (codex round-12). The dtype is part of the
2001-# signature for the same reason — fusion decisions depend on element
2002-# type, and the receipt is f32-only (f64 unmeasured as of 2026-07-27).
2003-# Grow ONLY with a measured receipt for the exact signature.
2004-# PROVISIONAL f64 np8 entry under test (job 26493638: f64 np4 is HEALTHY
2005-# at eff 0.95 while np8 ANTI-scales 20.10 -> 21.42 — the candidate
2006-# pathological shape shifts one rung with the doubled element size).
2007-# Receipt job decides whether this entry stays.
2008-_FUSION_BARRIER_WORKLOADS = frozenset({
2009-    (4, 1_966_080, 655_376, 26, "float32"),
2010-    (8, 1_966_080, 655_376, 26, "float64"),
2011-})
2012-
2013-_VORONOI_SPMD_STATE_FIELDS = frozenset(
2014-    {"u", "T", "p_s", "phis", "v", "tracers"})
2015-
2016-
2017-def check_voronoi_spmd_state_schema(state) -> tuple:
2018-    """Validate the MPAS state schema for the sharded SPMD step.
2019-
2020-    Raises on (a) a ``HydrostaticState`` field-set drift (a new field
2021-    must be threaded through the packed exchange deliberately) and
2022-    (b) a non-None ``v`` (MPAS carries the wind as edge-normal ``u``
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
tests/parallel/test_spmd_schedule_cost.py:30:    assert c["max_degree"] >= 1
tests/parallel/test_spmd_schedule_cost.py:31:    assert c["n_rounds"] >= c["max_degree"], (
tests/parallel/test_spmd_schedule_cost.py:32:        "a proper edge colouring can never use FEWER rounds than max_degree")
tests/parallel/test_spmd_schedule_cost.py:36:def test_reports_max_degree_so_optimality_is_measured_not_assumed(mesh):
tests/parallel/test_spmd_schedule_cost.py:38:    max_degree lower bound. Callers must be able to check equality rather than
tests/parallel/test_spmd_schedule_cost.py:39:    assume it, so max_degree is returned alongside n_rounds."""
tests/parallel/test_spmd_schedule_cost.py:41:    assert "max_degree" in c and c["max_degree"] > 0
tests/parallel/test_spmd_schedule_cost.py:42:    assert c["n_rounds"] >= c["max_degree"]
packages/core/legoesm/parallel/sharded_dynamics.py:1263:    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
packages/core/legoesm/parallel/sharded_dynamics.py:1273:    * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
packages/core/legoesm/parallel/sharded_dynamics.py:1274:      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
packages/core/legoesm/parallel/sharded_dynamics.py:1276:      ``max_degree``), never assumed.  Do not claim "the colouring is already
packages/core/legoesm/parallel/sharded_dynamics.py:1317:        ``n_rounds`` (the cost), ``max_degree`` (the lower bound to compare
packages/core/legoesm/parallel/sharded_dynamics.py:1372:    sched = _build_ppermute_schedule(
packages/core/legoesm/parallel/sharded_dynamics.py:1389:        "max_degree": int(sched.get("max_degree", -1)),
packages/core/legoesm/parallel/sharded_dynamics.py:1661:    """First-fit edge coloring visiting ``order`` (a list of normalized
packages/core/legoesm/parallel/sharded_dynamics.py:1683:    ``2*max_degree - 1`` colors — each color is one ppermute ROUND, and the
packages/core/legoesm/parallel/sharded_dynamics.py:1706:# Fixed shuffle seeds for the multi-start greedy edge coloring below —
packages/core/legoesm/parallel/sharded_dynamics.py:1715:    """Proper edge coloring via multi-start first-fit; returns the coloring
packages/core/legoesm/parallel/sharded_dynamics.py:1722:    ``max_degree`` lower bound (verified optimal on ico subdivisions 3–5 ×
packages/core/legoesm/parallel/sharded_dynamics.py:1728:    shuffles). Returns ``(edge_colors, max_degree)``.
packages/core/legoesm/parallel/sharded_dynamics.py:1738:    max_degree = max(deg.values(), default=0)
packages/core/legoesm/parallel/sharded_dynamics.py:1757:            if best_rounds <= max_degree:
packages/core/legoesm/parallel/sharded_dynamics.py:1759:    return best_colors, max_degree
packages/core/legoesm/parallel/sharded_dynamics.py:1762:def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
packages/core/legoesm/parallel/sharded_dynamics.py:1783:        n_rounds, n_rounds_greedy, max_degree, coloring_method,
packages/core/legoesm/parallel/sharded_dynamics.py:1828:            'max_degree': 0,
packages/core/legoesm/parallel/sharded_dynamics.py:1844:    #    chromatic-index floor (= max_degree) on every probed MPAS config
packages/core/legoesm/parallel/sharded_dynamics.py:1851:    multi_colors, max_degree = _multi_ordering_edge_coloring(comm_pairs)
packages/core/legoesm/parallel/sharded_dynamics.py:1865:        "improper ppermute edge coloring — two same-round exchanges "
packages/core/legoesm/parallel/sharded_dynamics.py:1969:        'max_degree': max_degree,
packages/core/legoesm/parallel/sharded_dynamics.py:2331:        pp_sched = _build_ppermute_schedule(

codex
The round-7 structural guard is genuinely exercised by the stub’s executed `-c` path, and the exact arm argument assertions bind the intended 64/128 and method matrix. I found no production-facing escape; I’m doing the final predecessor-versus-current check now.
codex
No material defect found. **SHIP.**

- The guard rejects the three round-7 bypass shapes: only a list containing a mapping with `n_ranks` and `schedule.n_rounds` qualifies. The stub executes that exact inline check, so the test is not rubber-stamped. [launcher](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:145) [test](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:320)

- Strictly, the unparseable-text decoy did not pass the immediate prior guard—it was already rejected by `json.load`. It is retained round-6 coverage; the other three would have passed the prior `any("n_ranks" in r ...)` predicate. This is not vacuous coverage.

- The scan’s working points are correct: S8/S9 run `64,128`, S10 runs `128`; every arm uses all three methods. The test asserts those exact argv values, so changing either an arm’s rank counts or its method list fails. The validation census matches the scorer’s documented reference values. [launcher](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:180) [test](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:186) [scorer census](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1323)

- Rechecked failure propagation: invalid zero-exit artifacts become nonzero; either validation failure prevents arm 3; arm-3 failure is propagated rather than reaching `SCAN_DONE`. [launcher](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:166) [launcher](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:186)

`bash -n` and `git diff --check` passed. I could not re-run pytest in this read-only sandbox because it has no writable temporary directory.
tokens used
96,699
No material defect found. **SHIP.**

- The guard rejects the three round-7 bypass shapes: only a list containing a mapping with `n_ranks` and `schedule.n_rounds` qualifies. The stub executes that exact inline check, so the test is not rubber-stamped. [launcher](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:145) [test](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:320)

- Strictly, the unparseable-text decoy did not pass the immediate prior guard—it was already rejected by `json.load`. It is retained round-6 coverage; the other three would have passed the prior `any("n_ranks" in r ...)` predicate. This is not vacuous coverage.

- The scan’s working points are correct: S8/S9 run `64,128`, S10 runs `128`; every arm uses all three methods. The test asserts those exact argv values, so changing either an arm’s rank counts or its method list fails. The validation census matches the scorer’s documented reference values. [launcher](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:180) [test](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:186) [scorer census](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1323)

- Rechecked failure propagation: invalid zero-exit artifacts become nonzero; either validation failure prevents arm 3; arm-3 failure is propagated rather than reaching `SCAN_DONE`. [launcher](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:166) [launcher](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:186)

`bash -n` and `git diff --check` passed. I could not re-run pytest in this read-only sandbox because it has no writable temporary directory.
