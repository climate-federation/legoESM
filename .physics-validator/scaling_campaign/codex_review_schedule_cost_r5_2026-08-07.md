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
session id: 019fdbf0-3c83-7140-bf8b-aeeec43f8b60
--------
user
ADVERSARIAL REVIEW, ROUND 5. Rounds 1-4 all ended NOT READY. Verify the round-4 fixes and hunt for new defects. Refute, do not summarize. End with SHIP or NOT READY.

Files: `git diff` on scripts/bench/bench_voronoi_partition_methods.py and tests/bench/test_bench_voronoi_partition_methods.py; untracked scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch and tests/bench/test_mpas_schedule_cost_scan_sbatch.py.

ROUND-4 FINDINGS AND CLAIMED FIXES:
1. HIGH 'the test hook LEGOESM_SCHEDULE_COST_BENCH can silently replace the production instrument' -> AGREED, removed. BENCH is hardcoded again, with a comment saying why it must not be overridable. The shell test now substitutes the INTERPRETER via LEGOESM_PYTHON (an existing production knob _env.sh already reads) with a stub that answers _env.sh's `-c` probe and then exits with scripted per-arm codes. A new test asserts the sbatch contains the hardcoded BENCH line and does NOT contain the old env var name. VERIFY there is no remaining test-only seam in the launcher, and that swapping the interpreter is not itself a hazard I have merely moved.
2. HIGH 'the shell regression test is skipped in CI because it required a repo-local .venv' -> the venv requirement is GONE (the stub is a standalone script); the module now skips only if bash is absent. VERIFY it would actually run under a CI python.
3. MEDIUM '--methods " , , " produces a successful empty benchmark' -> main() now raises SystemExit('selects NO methods'), with a test. VERIFY, and look for any remaining input that yields an empty-but-successful run.
4. MEDIUM 'shell test partly vacuous: only arm 1 validation failure tested; stub records only --subdivision' -> now parametrized over BOTH validation arms failing ([1,0,0] and [0,1,0]), asserts the arms that ran are exactly ['8','9'], the stub records FULL argv, and a new test asserts each arm carries --schedule-cost and --lloyd 0, that both validation arms carry a non-empty --expect-rounds, and that arm 3 does NOT. VERIFY these would fail if the corresponding flag were dropped.

ALSO:
- Re-check everything you previously called verified, in case a round-4 edit broke it.
- Hunt for NEW vacuity in the tests added since round 3.
- You said the single s8 geometric@64 census match is a spot check, not validation, and that validation requires the full six-row gate to pass in the real JSON. I agree and have not claimed otherwise; the run is what produces that. Confirm the gate as written would actually catch a wrong graph (i.e. that all six rows are compared, not just the ones that happen to be scored).

Cite file:line.
codex
I’ll re-audit the actual diffs and the new launcher/test pair as a hostile reviewer, tracing every claimed safeguard through the code and checking the tests for ways they could pass vacuously.
exec
/usr/bin/bash -lc "git status --short && git diff -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py && printf '\\n--- untracked launcher ---\\n' && nl -ba scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && printf '\\n--- untracked launcher test ---\\n' && nl -ba tests/bench/test_mpas_schedule_cost_scan_sbatch.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 417ms:
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
index 6aa0f4a40..ed60dc80f 100644
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
@@ -185,6 +440,23 @@ def main() -> int:
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
@@ -204,7 +476,9 @@ def main() -> int:
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
@@ -213,6 +487,24 @@ def main() -> int:
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

--- untracked launcher ---
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
   120	  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
   121	  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
   122	  "$PY" "$BENCH" \
   123	      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
   124	      --methods geometric,sfc,metis --schedule-cost \
   125	      ${4:+--expect-rounds "$4"} \
   126	      --out "$OUT/schedule_cost_s$1.json"
   127	  local rc=$?   # capture BEFORE anything else runs, or `date` overwrites it
   128	  echo "[scan] arm $3 exit=$rc"
   129	  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
   130	  return $rc
   131	}
   132	
   133	# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
   134	# makes the bench exit non-zero unless it reproduces the census below, and a
   135	# listed pair that never got scored (e.g. pymetis missing) counts as a
   136	# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
   137	# pass — an instrument that misses the known answer cannot be trusted on the
   138	# unknown one.
   139	S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
   140	S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"
   141	
   142	run_arm 8 64,128 "1/3 VALIDATION s8" "$S8_EXPECT"; RC8=$?
   143	run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?
   144	
   145	if [ "$RC8" -ne 0 ] || [ "$RC9" -ne 0 ]; then
   146	  echo "[scan] INSTRUMENT CHECK FAILED (s8 rc=$RC8, s9 rc=$RC9)."
   147	  echo "[scan] The scorer does not reproduce the reference census, so an s10"
   148	  echo "[scan] number from it would be untrustworthy. NOT running arm 3."
   149	  echo "SCAN_ABORTED_VALIDATION"
   150	  exit 1
   151	fi
   152	
   153	# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
   154	run_arm 10 128 "3/3 UNKNOWN s10" ""; RC10=$?
   155	echo "[scan] arm 3 rc=$RC10"
   156	
   157	if [ "$RC10" -ne 0 ]; then
   158	  # Do NOT let SLURM record this as COMPLETED: there is no valid s10 row,
   159	  # and an exit-0 job with a missing result reads as a successful scan.
   160	  echo "SCAN_FAILED_ARM3"
   161	  exit "$RC10"
   162	fi
   163	
   164	echo "SCAN_DONE"

--- untracked launcher test ---
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
    40	def _run(tmp_path, codes):
    41	    """Run the launcher with a stub interpreter that exits ``codes`` per arm.
    42	
    43	    ``codes`` is one exit status per bench invocation, in order (arm 1, arm
    44	    2, arm 3).  Returns ``(proc, arms)`` where ``arms`` is the recorded argv
    45	    of each bench call.
    46	    """
    47	    log = tmp_path / "calls.txt"
    48	    stub = tmp_path / "stub_python"
    49	    stub.write_text(
    50	        "#!/usr/bin/env python3\n"
    51	        "import sys, json\n"
    52	        f"codes = {list(codes)!r}\n"
    53	        f"log = {str(log)!r}\n"
    54	        "argv = sys.argv[1:]\n"
    55	        # _env.sh probes the interpreter with `-c 'import jax...'`; answer it
    56	        # without counting it as a bench call.
    57	        "if argv and argv[0] == '-c':\n"
    58	        "    sys.exit(0)\n"
    59	        "with open(log, 'a') as f:\n"
    60	        "    f.write(json.dumps(argv) + '\\n')\n"
    61	        "n = sum(1 for _ in open(log))\n"
    62	        "sys.exit(codes[n - 1] if n <= len(codes) else 0)\n"
    63	    )
    64	    stub.chmod(0o755)
    65	
    66	    env = dict(os.environ)
    67	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
    68	    env["LEGOESM_REPO"] = str(_REPO)
    69	    env["LEGOESM_PYTHON"] = str(stub)
    70	    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    71	    proc = subprocess.run(
    72	        ["bash", str(_SBATCH)], env=env, cwd=str(_REPO),
    73	        capture_output=True, text=True, timeout=600)
    74	
    75	    arms = []
    76	    if log.exists():
    77	        import json
    78	        arms = [json.loads(line) for line in log.read_text().splitlines()
    79	                if line.strip()]
    80	    return proc, arms
    81	
    82	
    83	def _levels(arms):
    84	    return [a[a.index("--subdivision") + 1] for a in arms]
    85	
    86	
    87	def test_launcher_is_not_redirectable_from_the_environment():
    88	    """The bench path must be hardcoded.
    89	
    90	    If it were env-overridable, a stray exported variable could point every
    91	    arm at something that exits 0 and the job would report SCAN_DONE with no
    92	    results — the exact failure this module guards.
    93	    """
    94	    text = _SBATCH.read_text()
    95	    assert "BENCH=scripts/bench/bench_voronoi_partition_methods.py" in text
    96	    assert "LEGOESM_SCHEDULE_COST_BENCH" not in text
    97	
    98	
    99	@pytest.mark.parametrize("codes, failing_arm", [([1, 0, 0], 1), ([0, 1, 0], 2)])
   100	def test_validation_failure_aborts_before_the_unknown_arm(
   101	        tmp_path, codes, failing_arm):
   102	    """EITHER validation arm failing must abort non-zero and skip arm 3.
   103	
   104	    An instrument that misses the known census cannot be trusted on the
   105	    unknown one, so producing an s10 number anyway is worse than none.
   106	    Both arms are exercised: guarding only arm 1 leaves arm 2 unchecked.
   107	    """
   108	    proc, arms = _run(tmp_path, codes)
   109	    assert proc.returncode != 0, proc.stdout[-2000:]
   110	    assert "SCAN_ABORTED_VALIDATION" in proc.stdout
   111	    assert "SCAN_DONE" not in proc.stdout
   112	    assert _levels(arms) == ["8", "9"], (
   113	        f"arm 3 must not run after validation arm {failing_arm} failed: "
   114	        f"{_levels(arms)}")
   115	
   116	
   117	def test_arm3_failure_is_not_reported_as_success(tmp_path):
   118	    """The original BLOCKER: arm 3 fails, the job must NOT exit 0, and must
   119	    propagate the exact status."""
   120	    proc, arms = _run(tmp_path, [0, 0, 3])
   121	    assert proc.returncode == 3, (
   122	        f"arm-3 status not propagated (got {proc.returncode})\n"
   123	        f"{proc.stdout[-2000:]}")
   124	    assert "SCAN_FAILED_ARM3" in proc.stdout
   125	    assert "SCAN_DONE" not in proc.stdout
   126	    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
   127	
   128	
   129	def test_full_success_exits_zero_and_runs_every_arm(tmp_path):
   130	    proc, arms = _run(tmp_path, [0, 0, 0])
   131	    assert proc.returncode == 0, proc.stdout[-2000:]
   132	    assert "SCAN_DONE" in proc.stdout
   133	    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
   134	
   135	
   136	def test_each_arm_carries_the_flags_its_claim_depends_on(tmp_path):
   137	    """The exit plumbing being right is worthless if an arm silently stops
   138	    scoring or stops gating.
   139	
   140	    Without this, dropping ``--schedule-cost`` (nothing is scored) or
   141	    ``--expect-rounds`` (the validation arms assert nothing) would leave
   142	    every other test in this module green.
   143	    """
   144	    _, arms = _run(tmp_path, [0, 0, 0])
   145	    assert len(arms) == 3, arms
   146	    for argv in arms:
   147	        assert "--schedule-cost" in argv, argv
   148	        assert "--lloyd" in argv and argv[argv.index("--lloyd") + 1] == "0"
   149	    # Validation arms must actually gate; the unknown arm must not pretend to.
   150	    for argv in arms[:2]:
   151	        assert "--expect-rounds" in argv, argv
   152	        assert argv[argv.index("--expect-rounds") + 1].strip(), argv
   153	    assert "--expect-rounds" not in arms[2], arms[2]

exec
/usr/bin/bash -lc "git diff --unified=80 -- scripts/bench/bench_voronoi_partition_methods.py && printf '\\n--- bench test full, numbered ---\\n' && nl -ba tests/bench/test_bench_voronoi_partition_methods.py && printf '\\n--- launcher syntax / expansion probe ---\\n' && bash -n scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && spec='sfc:64=12,sfc:128=14'; set -- "'${spec:+--expect-rounds "$spec"}; printf '"'<%s>\\n' \""'$@"' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 199ms:
diff --git a/scripts/bench/bench_voronoi_partition_methods.py b/scripts/bench/bench_voronoi_partition_methods.py
index 6aa0f4a40..ed60dc80f 100644
--- a/scripts/bench/bench_voronoi_partition_methods.py
+++ b/scripts/bench/bench_voronoi_partition_methods.py
@@ -1,220 +1,512 @@
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

--- bench test full, numbered ---
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

--- launcher syntax / expansion probe ---
<--expect-rounds>
<sfc:64=12,sfc:128=14>

exec
/usr/bin/bash -lc "nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '320,540p'
printf '\\n--- bench test lines 1-160 ---\\n'
nl -ba tests/bench/test_bench_voronoi_partition_methods.py | sed -n '1,160p'
printf '\\n--- bench test lines 160-360 ---\\n'
nl -ba tests/bench/test_bench_voronoi_partition_methods.py | sed -n '160,360p'
printf '\\n--- _env production knob ---\\n'
nl -ba scripts/cluster/scaling_levante/_env.sh | sed -n '1,240p'
printf '\\n--- matching references ---\\n'
rg -n --glob '"'!*.pyc'"' 'LEGOESM_PYTHON|expect-rounds|spmd_schedule_cost|S8_EXPECT|S9_EXPECT|schedule_cost' scripts tests legoesm | head -n 400" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 825ms:
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
   461	    payload = {
   462	        "rows": rows,
   463	        "auto_resolves_to": resolve_partition_method("auto"),
   464	        "step_time_pointer": (
   465	            "step-time per method: bench_ocean_mpas_scaling.py / "
   466	            "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
   467	            "per launch, same case otherwise)"),
   468	        "metadata": annotate_incomplete(scaling_metadata(
   469	            grid="voronoi",
   470	            component="partitioning",
   471	            resolution=f"L{args.subdivision}",
   472	            n_levels=0,
   473	            precision="n/a",
   474	            decomposition="cell_partition",
   475	            solver_variant="n/a",
   476	            scaling_kind="partition-quality",
   477	            transport="none",
   478	            extra={"rank_counts": rank_counts, "methods": methods,
   479	                   "halo_depth": args.halo_depth,
   480	                   "schedule_cost": bool(args.schedule_cost),
   481	                   "lloyd_iterations": args.lloyd},
   482	        )),
   483	    }
   484	    outdir = os.path.dirname(args.out)
   485	    if outdir:
   486	        os.makedirs(outdir, exist_ok=True)
   487	    with open(args.out, "w") as f:
   488	        json.dump(payload, f, indent=2)
   489	    print(f"JSON: {args.out}")
   490	
   491	    if expect:
   492	        failures = check_expected_rounds(rows, expect)
   493	        payload["expected_rounds_check"] = {
   494	            "expected": {f"{m}:{n}": v for (m, n), v in sorted(expect.items())},
   495	            "failures": failures,
   496	            "passed": not failures,
   497	        }
   498	        with open(args.out, "w") as f:
   499	            json.dump(payload, f, indent=2)
   500	        if failures:
   501	            print(f"INSTRUMENT CHECK FAILED ({len(failures)}):")
   502	            for line in failures:
   503	                print(f"  {line}")
   504	            print("The scorer did NOT reproduce the known census — treat every "
   505	                  "unknown row in this run as UNTRUSTED.")
   506	            return 1
   507	        print(f"INSTRUMENT CHECK PASSED ({len(expect)} expectations)")
   508	    return 0
   509	
   510	
   511	if __name__ == "__main__":
   512	    raise SystemExit(main())

--- bench test lines 1-160 ---
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

--- bench test lines 160-360 ---
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

--- _env production knob ---
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

--- matching references ---
rg: legoesm: No such file or directory (os error 2)
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:15:INTERPRETER via ``LEGOESM_PYTHON``, which is an existing production knob that
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:34:           / "mpas_schedule_cost_scan.sbatch")
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:69:    env["LEGOESM_PYTHON"] = str(stub)
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:141:    ``--expect-rounds`` (the validation arms assert nothing) would leave
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:151:        assert "--expect-rounds" in argv, argv
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:152:        assert argv[argv.index("--expect-rounds") + 1].strip(), argv
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:153:    assert "--expect-rounds" not in arms[2], arms[2]
tests/bench/test_bench_voronoi_partition_methods.py:154:def test_schedule_cost_row_reports_rounds_against_their_lower_bound():
tests/bench/test_bench_voronoi_partition_methods.py:155:    """``schedule_cost_row`` returns the REAL schedule depth and the bound it
tests/bench/test_bench_voronoi_partition_methods.py:158:    sc = mod.schedule_cost_row(mesh, "geometric", 2)
tests/bench/test_bench_voronoi_partition_methods.py:207:    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
tests/bench/test_bench_voronoi_partition_methods.py:209:    sc = mod.schedule_cost_row(object(), "geometric", 8)
tests/bench/test_bench_voronoi_partition_methods.py:226:            sc = mod.schedule_cost_row(_mesh(level=3), method, n_ranks)
tests/bench/test_bench_voronoi_partition_methods.py:232:def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
tests/bench/test_bench_voronoi_partition_methods.py:238:    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
tests/bench/test_bench_voronoi_partition_methods.py:242:def test_schedule_cost_matches_the_production_scorer_exactly():
tests/bench/test_bench_voronoi_partition_methods.py:246:    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
tests/bench/test_bench_voronoi_partition_methods.py:249:    ref = spmd_schedule_cost(mesh, 2, method="sfc")
tests/bench/test_bench_voronoi_partition_methods.py:250:    sc = mod.schedule_cost_row(mesh, "sfc", 2)
tests/bench/test_bench_voronoi_partition_methods.py:257:def test_schedule_cost_flag_is_opt_in_and_lands_in_the_json(
tests/bench/test_bench_voronoi_partition_methods.py:268:    assert payload["metadata"]["extra"]["schedule_cost"] is False
tests/bench/test_bench_voronoi_partition_methods.py:278:    assert payload2["metadata"]["extra"]["schedule_cost"] is True
tests/bench/test_bench_voronoi_partition_methods.py:333:        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
tests/bench/test_bench_voronoi_partition_methods.py:340:        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
tests/bench/test_bench_voronoi_partition_methods.py:348:        "--expect-rounds", "sfc:2=3", "--out", str(missing)])
tests/bench/test_bench_voronoi_partition_methods.py:359:        "--methods", "geometric", "--expect-rounds", "geometric:2=1",
tests/bench/test_bench_voronoi_partition_methods.py:369:    with pytest.raises(ValueError, match="expect-rounds"):
tests/bench/test_bench_voronoi_partition_methods.py:427:        "--expect-rounds", "metis:2=1", "--out", str(out)])
scripts/bench/bench_voronoi_partition_methods.py:19:   ``spmd_schedule_cost`` (which calls the production builders), never a
scripts/bench/bench_voronoi_partition_methods.py:78:      --out results/a1/schedule_cost_s9.json
scripts/bench/bench_voronoi_partition_methods.py:210:                f"--expect-rounds: cannot parse {item!r}; expected "
scripts/bench/bench_voronoi_partition_methods.py:214:                f"--expect-rounds: unknown method {key[0]!r} in {item!r}; "
scripts/bench/bench_voronoi_partition_methods.py:218:                f"--expect-rounds: duplicate expectation for "
scripts/bench/bench_voronoi_partition_methods.py:229:            f"--expect-rounds={spec!r} parses to NO expectations; the gate "
scripts/bench/bench_voronoi_partition_methods.py:259:def schedule_cost_row(mesh, method: str, n_ranks: int) -> dict:
scripts/bench/bench_voronoi_partition_methods.py:263:    :func:`legoesm.parallel.sharded_dynamics.spmd_schedule_cost` — it reorders
scripts/bench/bench_voronoi_partition_methods.py:295:    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
scripts/bench/bench_voronoi_partition_methods.py:298:    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
scripts/bench/bench_voronoi_partition_methods.py:354:    p.add_argument("--expect-rounds", type=str, default="",
scripts/bench/bench_voronoi_partition_methods.py:374:        # Same class as an empty --expect-rounds: the loop would be skipped,
scripts/bench/bench_voronoi_partition_methods.py:401:    if expect and not args.schedule_cost:
scripts/bench/bench_voronoi_partition_methods.py:403:            "--expect-rounds compares scored round counts, so it needs "
scripts/bench/bench_voronoi_partition_methods.py:443:            if args.schedule_cost:
scripts/bench/bench_voronoi_partition_methods.py:444:                sc = schedule_cost_row(mesh, method, n_ranks)
scripts/bench/bench_voronoi_partition_methods.py:480:                   "schedule_cost": bool(args.schedule_cost),
tests/land/test_init_experiment.py:143:    assert 'export LEGOESM_PYTHON="$(which python)"' in run_sh
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:36:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:9:#SBATCH --output=mpas_schedule_cost_scan.%j.log
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:70:# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:89:#   sbatch scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:96:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:100:OUT=results/a1/mpas_schedule_cost
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:108:# tests/bench/test_mpas_schedule_cost_scan_sbatch.py.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:119:run_arm () {  # $1=level  $2=rank-counts  $3=label  $4=expect-rounds (may be "")
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:125:      ${4:+--expect-rounds "$4"} \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:126:      --out "$OUT/schedule_cost_s$1.json"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:133:# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:139:S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:140:S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:142:run_arm 8 64,128 "1/3 VALIDATION s8" "$S8_EXPECT"; RC8=$?
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:143:run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?
scripts/cluster/scaling_levante/_env.sh:63:if [ -z "${LEGOESM_PYTHON:-}" ] && [ -x "$REPO/.venv/bin/python" ]; then
scripts/cluster/scaling_levante/_env.sh:64:  LEGOESM_PYTHON="$REPO/.venv/bin/python"
scripts/cluster/scaling_levante/_env.sh:66:PY="${LEGOESM_PYTHON:-$(command -v python)}"
scripts/cluster/scaling_levante/_env.sh:72:  echo "[_env.sh] Set LEGOESM_PYTHON=<repo>/.venv/bin/python (uv venv) or" >&2
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:26:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_latlon_144.sbatch:31:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:18:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/bin/python}"
scripts/cluster/scaling_levante/ocean_latlon_hundreds.sbatch:23:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:53:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:31:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:33:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/prewarm_s10.sbatch:19:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:26:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_s10_192.sbatch:21:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_bound_base.sbatch:23:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch:22:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:33:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
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
scripts/cluster/unified_training/score_ucast_probabilistic.sbatch:41:#LEGOESM_PYTHON=$PWD/.venv/bin/python \
scripts/cluster/unified_training/bench_wb2_levante.sbatch:32:PY="${LEGOESM_PYTHON:-$REPO/.venv/bin/python}"
scripts/cluster/unified_training/score_ucast_arms.sbatch:28:#LEGOESM_PYTHON=$PWD/.venv/bin/python \
scripts/run/init_experiment.py:182:    # (--machine), env_setup overrides LEGOESM_PYTHON via `which python` after
scripts/run/init_experiment.py:207:        f'PYTHON_BIN="${{LEGOESM_PYTHON:-{sys.executable}}}"\n'

exec
/usr/bin/bash -lc "bash -c 'x(){ set -- PRE "'${4:+--expect-rounds "$4"} POST; printf "<%s>'"\\\\n\" \""'$@"; }; x a b c "sfc:64=12,sfc:128=14"; x a b c ""'"'
printf '\\n--- production scorer and direct tests ---\\n'
nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | rg -n -C 8 'def spmd_schedule_cost|_build_ppermute_schedule|comm_pairs|production_strategy|max_degree' 
printf '\\n--- direct schedule-cost tests ---\\n'
nl -ba tests/parallel/test_spmd_schedule_cost.py | sed -n '1,230p'
printf '\\n--- git diff name/status and check syntax ---\\n'
git diff --check && git diff --numstat -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py && python -m py_compile scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py tests/bench/test_mpas_schedule_cost_scan_sbatch.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 127 in 184ms:
<PRE>
<--expect-rounds>
<sfc:64=12,sfc:128=14>
<POST>
<PRE>
<POST>

--- production scorer and direct tests ---
1242-  1242	
1243-  1243	#: Halo depth the SPMD Voronoi partition infra is built at.  ONE definition
1244-  1244	#: consumed by both the production step factory and ``spmd_schedule_cost``:
1245-  1245	#: a score computed at a different depth describes a different comm graph, and
1246-  1246	#: two independently hardcoded 3s let production drift unnoticed.
1247-  1247	SPMD_HALO_DEPTH = 3
1248-  1248	
1249-  1249	
1250:  1250	def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
1251-  1251	                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
1252-  1252	                       ppermute_cells_per_device_threshold=2_000):
1253-  1253	    """How much halo communication one ownership choice costs, computed offline.
1254-  1254	
1255-  1255	    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
1256-  1256	    ROUNDS one halo exchange needs -- the sequential collective launches that
1257-  1257	    dominate MPAS strong scaling above ~64 devices.  Runs on a laptop: no GPU,
1258-  1258	    no MPI, no benchmark job, so a split can be compared before it costs an
1259-  1259	    allocation.
1260-  1260	
1261-  1261	    It calls the SAME builders production calls
1262-  1262	    (:func:`_build_voronoi_partition_infra` then
1263:  1263	    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
1264-  1264	    different question: a 1-ring ``cellsOnEdge`` adjacency graph reports 8
1265-  1265	    rounds where the real depth-3-plus-closure graph reports 12-14.
1266-  1266	
1267-  1267	    WHAT THE NUMBER IS NOT
1268-  1268	    ----------------------
1269-  1269	    * ``n_rounds`` is per HALO FILL, not per model step.  A step costs
1270-  1270	      ``n_rounds`` x (tendency evaluations per step), which depends on the
1271-  1271	      configured integrator -- SSP-RK3 evaluates 3 times, but the MPAS default
1272-  1272	      is ``ssp_rk54_scan``.  Multiply with the integrator you actually run.
1273:  1273	    * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
1274:  1274	      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
1275-  1275	      keeps the best; equality is MEASURED (compare the returned
1276:  1276	      ``max_degree``), never assumed.  Do not claim "the colouring is already
1277-  1277	      optimal so only ownership can help" from this function.
1278-  1278	    * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
1279-  1279	      cells/device is below ``ppermute_cells_per_device_threshold``, in which
1280-  1280	      case there is no ppermute schedule and this number is counterfactual --
1281:  1281	      see the returned ``production_strategy``.
1282-  1282	
1283-  1283	    MESH STATE -- the one thing that silently invalidates the score
1284-  1284	    --------------------------------------------------------------
1285-  1285	    Production does NOT reorder inside ``make_voronoi_sharded_step``; it
1286-  1286	    consumes an already-reordered ``model.mesh``.  The scaling bench reorders
1287-  1287	    ONCE for a ``reorder_target`` device count and then runs at a possibly
1288-  1288	    DIFFERENT device count.  So pass what you actually have:
1289-  1289	
--
1304-  1304	        Ownership for the reorder; ignored when *already_reordered*.
1305-  1305	    reorder_target : int | None
1306-  1306	        Device count the reorder targets, when it differs from *n_dev*.
1307-  1307	    already_reordered : bool
1308-  1308	    halo_depth : int
1309-  1309	        Must match production (3) or the graph is a different graph.
1310-  1310	    ppermute_cells_per_device_threshold : int
1311-  1311	        Mirror of the production auto-select threshold, only used to report
1312:  1312	        ``production_strategy``.
1313-  1313	
1314-  1314	    Returns
1315-  1315	    -------
1316-  1316	    dict
1317:  1317	        ``n_rounds`` (the cost), ``max_degree`` (the lower bound to compare
1318-  1318	        it against), ``n_rounds_greedy``, ``coloring_method``,
1319-  1319	        ``resolved_method`` (concrete, never ``"auto"``),
1320:  1320	        ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
1321-  1321	        ``max_local_cells``, ``max_local_edges``, and the echoed inputs.
1322-  1322	
1323-  1323	    Reference census on the unrelaxed mesh, which any change here must still
1324-  1324	    reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
1325-  1325	    geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
1326-  1326	    """
1327-  1327	    from legoesm.parallel.voronoi_partition import (
1328-  1328	        reorder_voronoi_for_sharding, resolve_sharding_partition_method,
--
1364-  1364	            f"a device count that does not divide it silently mis-slices the "
1365-  1365	            f"owned blocks. Score at a device count that divides the prepared "
1366-  1366	            f"mesh.")
1367-  1367	    (
1368-  1368	        _stacked, _gc, _ge, _noc, _noe, max_lc, max_le, partitions, cell_owner,
1369-  1369	    ) = _build_voronoi_partition_infra(prepared, n_dev, halo_depth=halo_depth)
1370-  1370	    cells_per = n_cells // n_dev
1371-  1371	    edges_per = n_edges // n_dev
1372:  1372	    sched = _build_ppermute_schedule(
1373-  1373	        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
1374-  1374	    )
1375-  1375	    return {
1376-  1376	        "method": method,
1377-  1377	        "resolved_method": resolved,
1378-  1378	        "n_dev": n_dev,
1379-  1379	        # Unknown for a pre-reordered mesh: the ownership is baked in and the
1380-  1380	        # target that produced it is not recoverable from the mesh. Reporting
1381-  1381	        # n_dev there would assert something we did not verify.
1382-  1382	        "reorder_target": (None if already_reordered else
1383-  1383	                           (n_dev if reorder_target is None
1384-  1384	                            else int(reorder_target))),
1385-  1385	        "already_reordered": bool(already_reordered),
1386-  1386	        "halo_depth": halo_depth,
1387-  1387	        "n_rounds": int(sched["n_rounds"]),
1388-  1388	        "n_rounds_greedy": int(sched["n_rounds_greedy"]),
1389:  1389	        "max_degree": int(sched.get("max_degree", -1)),
1390-  1390	        "coloring_method": sched["coloring_method"],
1391-  1391	        # Production returns before selecting a strategy at n_dev==1, and a
1392-  1392	        # caller may force halo_strategy; this reports what AUTO would pick.
1393:  1393	        "production_strategy": (
1394-  1394	            None if n_dev == 1 else
1395-  1395	            ("allgather" if cells_per < ppermute_cells_per_device_threshold
1396-  1396	             else "ppermute")),
1397-  1397	        "cells_per_device": cells_per,
1398-  1398	        "max_local_cells": int(max_lc),
1399-  1399	        "max_local_edges": int(max_le),
1400-  1400	    }
1401-  1401	
--
1652-  1652	        n_owned_edges,
1653-  1653	        max_lc,
1654-  1654	        max_le,
1655-  1655	        partitions,
1656-  1656	        cell_owner,
1657-  1657	    )
1658-  1658	
1659-  1659	
1660:  1660	def _greedy_edge_coloring_ordered(comm_pairs, order):
1661-  1661	    """First-fit edge coloring visiting ``order`` (a list of normalized
1662-  1662	    ``(min,max)`` pairs). Always a PROPER coloring; the color count depends
1663-  1663	    on the visitation order.
1664-  1664	    """
1665-  1665	    from collections import defaultdict
1666-  1666	
1667-  1667	    vertex_colors: dict[int, set[int]] = defaultdict(set)
1668-  1668	    edge_colors: dict[tuple[int, int], int] = {}
--
1672-  1672	        while color in used:
1673-  1673	            color += 1
1674-  1674	        edge_colors[(u, v)] = color
1675-  1675	        vertex_colors[u].add(color)
1676-  1676	        vertex_colors[v].add(color)
1677-  1677	    return edge_colors
1678-  1678	
1679-  1679	
1680:  1680	def _greedy_edge_coloring(comm_pairs):
1681-  1681	    """Legacy first-fit coloring on sorted pairs (the reference/never-regress
1682-  1682	    baseline for :func:`_multi_ordering_edge_coloring`). Worst case
1683:  1683	    ``2*max_degree - 1`` colors — each color is one ppermute ROUND, and the
1684-  1684	    route-B MPAS lane is round-latency-bound (#1113), so excess colors are
1685-  1685	    pure wall-clock.
1686-  1686	    """
1687:  1687	    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
1688:  1688	    return _greedy_edge_coloring_ordered(comm_pairs, edges)
1689-  1689	
1690-  1690	
1691:  1691	def _check_proper_edge_coloring(edge_colors, comm_pairs):
1692-  1692	    """Every pair colored, and no vertex sees a color twice."""
1693-  1693	    from collections import defaultdict
1694-  1694	
1695:  1695	    if set(edge_colors) != {tuple(sorted(p)) for p in comm_pairs}:
1696-  1696	        return False
1697-  1697	    seen: dict[int, set[int]] = defaultdict(set)
1698-  1698	    for (u, v), c in edge_colors.items():
1699-  1699	        if c in seen[u] or c in seen[v]:
1700-  1700	            return False
1701-  1701	        seen[u].add(c)
1702-  1702	        seen[v].add(c)
1703-  1703	    return True
--
1706-  1706	# Fixed shuffle seeds for the multi-start greedy edge coloring below —
1707-  1707	# a constant so every MPI rank / process builds the byte-identical
1708-  1708	# schedule (the coloring must agree across ranks or the ppermute pattern
1709-  1709	# desynchronises). NOT Math.random / device randomness: this is host-side
1710-  1710	# schedule construction, deterministic by seed.
1711-  1711	_COLORING_SHUFFLE_SEEDS = tuple(range(16))
1712-  1712	
1713-  1713	
1714:  1714	def _multi_ordering_edge_coloring(comm_pairs):
1715-  1715	    """Proper edge coloring via multi-start first-fit; returns the coloring
1716-  1716	    using the FEWEST colors (= ppermute rounds) across several deterministic
1717-  1717	    visitation orders.
1718-  1718	
1719-  1719	    First-fit greedy is order-sensitive: on the reordered MPAS comm graphs
1720-  1720	    the sorted order can overshoot the chromatic index by up to 3 rounds at
1721-  1721	    16 devices, while a degree-descending or shuffled order reaches the
1722:  1722	    ``max_degree`` lower bound (verified optimal on ico subdivisions 3–5 ×
1723-  1723	    {4,8,16} devices, auto/sfc partitions). Every candidate is a proper
1724-  1724	    coloring by construction, so taking the min can NEVER produce an
1725-  1725	    invalid schedule and can never regress below the legacy sorted greedy.
1726-  1726	
1727-  1727	    Deterministic across ranks (sorted + degree orders + fixed-seed
1728:  1728	    shuffles). Returns ``(edge_colors, max_degree)``.
1729-  1729	    """
1730-  1730	    import random
1731-  1731	    from collections import defaultdict
1732-  1732	
1733:  1733	    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
1734-  1734	    deg: dict[int, int] = defaultdict(int)
1735-  1735	    for u, v in edges:
1736-  1736	        deg[u] += 1
1737-  1737	        deg[v] += 1
1738:  1738	    max_degree = max(deg.values(), default=0)
1739-  1739	
1740-  1740	    orders = [
1741-  1741	        edges,                                                   # sorted
1742-  1742	        sorted(edges, key=lambda e: -(deg[e[0]] + deg[e[1]])),   # sum-deg desc
1743-  1743	        sorted(edges, key=lambda e: -max(deg[e[0]], deg[e[1]])),  # max-deg desc
1744-  1744	    ]
1745-  1745	    for seed in _COLORING_SHUFFLE_SEEDS:
1746-  1746	        shuffled = edges[:]
1747-  1747	        random.Random(seed).shuffle(shuffled)
1748-  1748	        orders.append(shuffled)
1749-  1749	
1750-  1750	    best_colors: dict[tuple[int, int], int] | None = None
1751-  1751	    best_rounds = None
1752-  1752	    for order in orders:
1753:  1753	        ec = _greedy_edge_coloring_ordered(comm_pairs, order)
1754-  1754	        rounds = max(ec.values(), default=-1) + 1
1755-  1755	        if best_rounds is None or rounds < best_rounds:
1756-  1756	            best_rounds, best_colors = rounds, ec
1757:  1757	            if best_rounds <= max_degree:
1758-  1758	                break            # hit the chromatic-index floor — optimal
1759:  1759	    return best_colors, max_degree
1760-  1760	
1761-  1761	
1762:  1762	def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
1763-  1763	                             edges_per, max_lc, max_le):
1764-  1764	    """Build a ppermute-based halo exchange schedule.
1765-  1765	
1766-  1766	    Instead of all-gathering the full state (O(N) communication),
1767-  1767	    this schedule uses ``jax.lax.ppermute`` to exchange only halo data
1768-  1768	    between neighboring devices.  The communication graph is edge-colored
1769-  1769	    so that each round of ppermute moves data between non-conflicting
1770-  1770	    pairs simultaneously.
--
1775-  1775	    cell_owner : np.ndarray, (nCells,)
1776-  1776	    n_dev, cells_per, edges_per : int
1777-  1777	    max_lc, max_le : int
1778-  1778	        Maximum local cell/edge counts (owned + halo) across devices.
1779-  1779	
1780-  1780	    Returns
1781-  1781	    -------
1782-  1782	    dict with keys:
1783:  1783	        n_rounds, n_rounds_greedy, max_degree, coloring_method,
1784-  1784	        ppermute_perms, send_cell_idx, recv_cell_pos,
1785-  1785	        send_edge_idx, recv_edge_pos, halo_cells_per_round,
1786-  1786	        halo_edges_per_round.
1787-  1787	    """
1788-  1788	    from collections import defaultdict
1789-  1789	
1790-  1790	    import numpy as np
1791-  1791	
--
1807-  1807	        for h_idx in range(part.n_owned_edges, part.n_local_edges):
1808-  1808	            g = int(part.local_edges[h_idx])
1809-  1809	            owner = min(g // edges_per, n_dev - 1)
1810-  1810	            halo_edges_from[d][owner].append(g)
1811-  1811	
1812-  1812	    # ------------------------------------------------------------------
1813-  1813	    # 2. Build undirected communication graph
1814-  1814	    # ------------------------------------------------------------------
1815:  1815	    comm_pairs: set[tuple[int, int]] = set()
1816-  1816	    for d in range(n_dev):
1817-  1817	        for d_prime in halo_cells_from[d]:
1818-  1818	            if d != d_prime:
1819:  1819	                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
1820-  1820	        for d_prime in halo_edges_from[d]:
1821-  1821	            if d != d_prime:
1822:  1822	                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
1823-  1823	
1824:  1824	    if not comm_pairs:
1825-  1825	        return {
1826-  1826	            'n_rounds': 0,
1827-  1827	            'n_rounds_greedy': 0,
1828:  1828	            'max_degree': 0,
1829-  1829	            'coloring_method': 'none',
1830-  1830	            'ppermute_perms': [],
1831-  1831	            'send_cell_idx': [],
1832-  1832	            'recv_cell_pos': [],
1833-  1833	            'send_edge_idx': [],
1834-  1834	            'recv_edge_pos': [],
1835-  1835	            'halo_cells_per_round': [],
1836-  1836	            'halo_edges_per_round': [],
1837-  1837	        }
1838-  1838	
1839-  1839	    # ------------------------------------------------------------------
1840-  1840	    # 3. Edge-color the graph: each color = one bidirectional ppermute
1841-  1841	    #    ROUND, and the route-B lane is round-latency-bound (#1113), so
1842-  1842	    #    fewer colors = directly less wall-clock. First-fit greedy is
1843-  1843	    #    order-sensitive; the multi-start coloring reaches the
1844:  1844	    #    chromatic-index floor (= max_degree) on every probed MPAS config
1845-  1845	    #    where the legacy sorted greedy overshoots (up to 3 rounds at 16
1846-  1846	    #    devices). It can never regress: the legacy sorted order is one of
1847-  1847	    #    its candidates and it takes the min. Both are verified proper.
1848-  1848	    # ------------------------------------------------------------------
1849:  1849	    greedy_colors = _greedy_edge_coloring(comm_pairs)
1850-  1850	    n_rounds_greedy = max(greedy_colors.values()) + 1
1851:  1851	    multi_colors, max_degree = _multi_ordering_edge_coloring(comm_pairs)
1852-  1852	    n_rounds_multi = max(multi_colors.values()) + 1
1853-  1853	    # Adopt the multi-start coloring ONLY when it STRICTLY reduces rounds;
1854-  1854	    # on a tie keep the exact legacy sorted-greedy coloring so the produced
1855-  1855	    # schedule is byte-identical to before wherever there is no round win
1856-  1856	    # (the win only appears at high device counts — >=16 on the probed
1857-  1857	    # MPAS meshes). Both colorings are proper.
1858-  1858	    if n_rounds_multi < n_rounds_greedy:
1859-  1859	        edge_colors, n_rounds, coloring_method = (
1860-  1860	            multi_colors, n_rounds_multi, "multi_greedy")
1861-  1861	    else:
1862-  1862	        edge_colors, n_rounds, coloring_method = (
1863-  1863	            greedy_colors, n_rounds_greedy, "greedy")
1864:  1864	    assert _check_proper_edge_coloring(edge_colors, comm_pairs), (
1865-  1865	        "improper ppermute edge coloring — two same-round exchanges "
1866-  1866	        "would collide at a device")
1867-  1867	    rounds: dict[int, list[tuple[int, int]]] = defaultdict(list)
1868-  1868	    for (u, v), color in edge_colors.items():
1869-  1869	        rounds[color].append((u, v))
1870-  1870	
1871-  1871	    # ------------------------------------------------------------------
1872-  1872	    # 4. Build directed send/recv maps for each device pair
--
1961-  1961	        send_cell_idx_out.append(jnp.array(sc))
1962-  1962	        recv_cell_pos_out.append(jnp.array(rc))
1963-  1963	        send_edge_idx_out.append(jnp.array(se))
1964-  1964	        recv_edge_pos_out.append(jnp.array(re))
1965-  1965	
1966-  1966	    return {
1967-  1967	        'n_rounds': n_rounds,
1968-  1968	        'n_rounds_greedy': n_rounds_greedy,
1969:  1969	        'max_degree': max_degree,
1970-  1970	        'coloring_method': coloring_method,
1971-  1971	        'ppermute_perms': ppermute_perms_out,
1972-  1972	        'send_cell_idx': send_cell_idx_out,
1973-  1973	        'recv_cell_pos': recv_cell_pos_out,
1974-  1974	        'send_edge_idx': send_edge_idx_out,
1975-  1975	        'recv_edge_pos': recv_edge_pos_out,
1976-  1976	        'halo_cells_per_round': halo_cells_per_round,
1977-  1977	        'halo_edges_per_round': halo_edges_per_round,
--
2323-  2323	    # Strategy dispatch: ppermute (O(halo)) vs allgather (O(N))
2324-  2324	    # ------------------------------------------------------------------
2325-  2325	
2326-  2326	    use_ppermute = halo_strategy == "ppermute"
2327-  2327	
2328-  2328	    if use_ppermute:
2329-  2329	        # Build ppermute schedule: neighbor-only halo exchange
2330-  2330	        t1 = time.time()
2331:  2331	        pp_sched = _build_ppermute_schedule(
2332-  2332	            partitions_out, cell_owner_out, n_dev,
2333-  2333	            cells_per, edges_per, max_lc, max_le,
2334-  2334	        )
2335-  2335	        n_rounds = pp_sched['n_rounds']
2336-  2336	        ppermute_perms = pp_sched['ppermute_perms']
2337-  2337	
2338-  2338	        # LOCAL-ONLY metadata: shard the per-round index arrays on the
2339-  2339	        # leading device axis (each device holds only its own schedule

--- direct schedule-cost tests ---
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

--- git diff name/status and check syntax ---
294	2	scripts/bench/bench_voronoi_partition_methods.py
309	0	tests/bench/test_bench_voronoi_partition_methods.py
/usr/bin/bash: line 6: python: command not found

codex
I’ve confirmed the hardcoded bench path is back, but the test’s interpreter substitution is still a production-controlled redirection path. I’m now checking the exact environment behavior and whether the six-row validation is genuinely enforced rather than merely present.
exec
/usr/bin/bash -lc "command -v python3 || true
python3 --version || true
rg -n --glob 'pyproject.toml' --glob 'pytest.ini' --glob 'tox.ini' --glob 'setup.cfg' --glob '.github/**' 'pytest|python-version|python3' . | head -n 300
printf '\\n--- source exact bench reference census / implementation ---\\n'
nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1320,1410p'
printf '\\n--- launcher env attack surface and test assertions ---\\n'
nl -ba scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch | sed -n '88,165p'
nl -ba tests/bench/test_mpas_schedule_cost_scan_sbatch.py | sed -n '32,170p'
printf '\\n--- arguments parse behavior candidates (read-only) ---\\n'
python3 -B - <<'PY'
import importlib.util
from pathlib import Path
p=Path('scripts/bench/bench_voronoi_partition_methods.py')
s=importlib.util.spec_from_file_location('b',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for spec in ('',',,,','   ','geometric:2=3','geometric:2=3,geometric:4=3'):
    try: print(repr(spec), m.parse_expect_rounds(spec))
    except Exception as e: print(repr(spec), type(e).__name__, str(e))
PY" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 1 in 186ms:
/usr/bin/python3
Python 3.6.8
./pyproject.toml:108:    "pytest>=8.0",
./pyproject.toml:109:    "pytest-xdist",
./pyproject.toml:110:    "pytest-timeout",
./pyproject.toml:310:[tool.pytest.ini_options]

--- source exact bench reference census / implementation ---
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

--- launcher env attack surface and test assertions ---
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
   120	  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
   121	  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
   122	  "$PY" "$BENCH" \
   123	      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
   124	      --methods geometric,sfc,metis --schedule-cost \
   125	      ${4:+--expect-rounds "$4"} \
   126	      --out "$OUT/schedule_cost_s$1.json"
   127	  local rc=$?   # capture BEFORE anything else runs, or `date` overwrites it
   128	  echo "[scan] arm $3 exit=$rc"
   129	  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
   130	  return $rc
   131	}
   132	
   133	# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
   134	# makes the bench exit non-zero unless it reproduces the census below, and a
   135	# listed pair that never got scored (e.g. pymetis missing) counts as a
   136	# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
   137	# pass — an instrument that misses the known answer cannot be trusted on the
   138	# unknown one.
   139	S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
   140	S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"
   141	
   142	run_arm 8 64,128 "1/3 VALIDATION s8" "$S8_EXPECT"; RC8=$?
   143	run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?
   144	
   145	if [ "$RC8" -ne 0 ] || [ "$RC9" -ne 0 ]; then
   146	  echo "[scan] INSTRUMENT CHECK FAILED (s8 rc=$RC8, s9 rc=$RC9)."
   147	  echo "[scan] The scorer does not reproduce the reference census, so an s10"
   148	  echo "[scan] number from it would be untrustworthy. NOT running arm 3."
   149	  echo "SCAN_ABORTED_VALIDATION"
   150	  exit 1
   151	fi
   152	
   153	# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
   154	run_arm 10 128 "3/3 UNKNOWN s10" ""; RC10=$?
   155	echo "[scan] arm 3 rc=$RC10"
   156	
   157	if [ "$RC10" -ne 0 ]; then
   158	  # Do NOT let SLURM record this as COMPLETED: there is no valid s10 row,
   159	  # and an exit-0 job with a missing result reads as a successful scan.
   160	  echo "SCAN_FAILED_ARM3"
   161	  exit "$RC10"
   162	fi
   163	
   164	echo "SCAN_DONE"
    32	_REPO = Path(__file__).resolve().parents[2]
    33	_SBATCH = (_REPO / "scripts" / "cluster" / "scaling_levante"
    34	           / "mpas_schedule_cost_scan.sbatch")
    35	
    36	pytestmark = pytest.mark.skipif(
    37	    shutil.which("bash") is None, reason="needs bash to run the launcher")
    38	
    39	
    40	def _run(tmp_path, codes):
    41	    """Run the launcher with a stub interpreter that exits ``codes`` per arm.
    42	
    43	    ``codes`` is one exit status per bench invocation, in order (arm 1, arm
    44	    2, arm 3).  Returns ``(proc, arms)`` where ``arms`` is the recorded argv
    45	    of each bench call.
    46	    """
    47	    log = tmp_path / "calls.txt"
    48	    stub = tmp_path / "stub_python"
    49	    stub.write_text(
    50	        "#!/usr/bin/env python3\n"
    51	        "import sys, json\n"
    52	        f"codes = {list(codes)!r}\n"
    53	        f"log = {str(log)!r}\n"
    54	        "argv = sys.argv[1:]\n"
    55	        # _env.sh probes the interpreter with `-c 'import jax...'`; answer it
    56	        # without counting it as a bench call.
    57	        "if argv and argv[0] == '-c':\n"
    58	        "    sys.exit(0)\n"
    59	        "with open(log, 'a') as f:\n"
    60	        "    f.write(json.dumps(argv) + '\\n')\n"
    61	        "n = sum(1 for _ in open(log))\n"
    62	        "sys.exit(codes[n - 1] if n <= len(codes) else 0)\n"
    63	    )
    64	    stub.chmod(0o755)
    65	
    66	    env = dict(os.environ)
    67	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
    68	    env["LEGOESM_REPO"] = str(_REPO)
    69	    env["LEGOESM_PYTHON"] = str(stub)
    70	    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    71	    proc = subprocess.run(
    72	        ["bash", str(_SBATCH)], env=env, cwd=str(_REPO),
    73	        capture_output=True, text=True, timeout=600)
    74	
    75	    arms = []
    76	    if log.exists():
    77	        import json
    78	        arms = [json.loads(line) for line in log.read_text().splitlines()
    79	                if line.strip()]
    80	    return proc, arms
    81	
    82	
    83	def _levels(arms):
    84	    return [a[a.index("--subdivision") + 1] for a in arms]
    85	
    86	
    87	def test_launcher_is_not_redirectable_from_the_environment():
    88	    """The bench path must be hardcoded.
    89	
    90	    If it were env-overridable, a stray exported variable could point every
    91	    arm at something that exits 0 and the job would report SCAN_DONE with no
    92	    results — the exact failure this module guards.
    93	    """
    94	    text = _SBATCH.read_text()
    95	    assert "BENCH=scripts/bench/bench_voronoi_partition_methods.py" in text
    96	    assert "LEGOESM_SCHEDULE_COST_BENCH" not in text
    97	
    98	
    99	@pytest.mark.parametrize("codes, failing_arm", [([1, 0, 0], 1), ([0, 1, 0], 2)])
   100	def test_validation_failure_aborts_before_the_unknown_arm(
   101	        tmp_path, codes, failing_arm):
   102	    """EITHER validation arm failing must abort non-zero and skip arm 3.
   103	
   104	    An instrument that misses the known census cannot be trusted on the
   105	    unknown one, so producing an s10 number anyway is worse than none.
   106	    Both arms are exercised: guarding only arm 1 leaves arm 2 unchecked.
   107	    """
   108	    proc, arms = _run(tmp_path, codes)
   109	    assert proc.returncode != 0, proc.stdout[-2000:]
   110	    assert "SCAN_ABORTED_VALIDATION" in proc.stdout
   111	    assert "SCAN_DONE" not in proc.stdout
   112	    assert _levels(arms) == ["8", "9"], (
   113	        f"arm 3 must not run after validation arm {failing_arm} failed: "
   114	        f"{_levels(arms)}")
   115	
   116	
   117	def test_arm3_failure_is_not_reported_as_success(tmp_path):
   118	    """The original BLOCKER: arm 3 fails, the job must NOT exit 0, and must
   119	    propagate the exact status."""
   120	    proc, arms = _run(tmp_path, [0, 0, 3])
   121	    assert proc.returncode == 3, (
   122	        f"arm-3 status not propagated (got {proc.returncode})\n"
   123	        f"{proc.stdout[-2000:]}")
   124	    assert "SCAN_FAILED_ARM3" in proc.stdout
   125	    assert "SCAN_DONE" not in proc.stdout
   126	    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
   127	
   128	
   129	def test_full_success_exits_zero_and_runs_every_arm(tmp_path):
   130	    proc, arms = _run(tmp_path, [0, 0, 0])
   131	    assert proc.returncode == 0, proc.stdout[-2000:]
   132	    assert "SCAN_DONE" in proc.stdout
   133	    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
   134	
   135	
   136	def test_each_arm_carries_the_flags_its_claim_depends_on(tmp_path):
   137	    """The exit plumbing being right is worthless if an arm silently stops
   138	    scoring or stops gating.
   139	
   140	    Without this, dropping ``--schedule-cost`` (nothing is scored) or
   141	    ``--expect-rounds`` (the validation arms assert nothing) would leave
   142	    every other test in this module green.
   143	    """
   144	    _, arms = _run(tmp_path, [0, 0, 0])
   145	    assert len(arms) == 3, arms
   146	    for argv in arms:
   147	        assert "--schedule-cost" in argv, argv
   148	        assert "--lloyd" in argv and argv[argv.index("--lloyd") + 1] == "0"
   149	    # Validation arms must actually gate; the unknown arm must not pretend to.
   150	    for argv in arms[:2]:
   151	        assert "--expect-rounds" in argv, argv
   152	        assert argv[argv.index("--expect-rounds") + 1].strip(), argv
   153	    assert "--expect-rounds" not in arms[2], arms[2]

--- arguments parse behavior candidates (read-only) ---
/usr/bin/bash: line 9: cannot create temp file for here-document: Read-only file system

exec
/usr/bin/bash -lc "sed -n '1,150p' pyproject.toml && sed -n '295,335p' pyproject.toml && find .github -maxdepth 3 -type f -print 2>/dev/null | sort | head -n 200 && rg -n 'python-version|setup-python|pytest' .github 2>/dev/null || true
printf '\\n--- bench changes only (compact) ---\\n'
git diff --unified=8 -- scripts/bench/bench_voronoi_partition_methods.py | sed -n '1,500p'
printf '\\n--- find test-only seams/new env identifiers ---\\n'
rg -n 'SCHEDULE_COST_BENCH|stub_python|LEGOESM_PYTHON|BENCH=' scripts/cluster/scaling_levante tests/bench | sed -n '1,260p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 297ms:
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "legoesm"
version = "0.1.0"
description = "legoESM: A Differentiable Earth System Model in JAX"
readme = "README.md"
license = "LicenseRef-PolyForm-Noncommercial-1.0.0"
license-files = ["LICENSE"]
requires-python = ">=3.11"
authors = [
    {name = "Pierre Gentine"},
]
keywords = ["earth-system-model", "jax", "differentiable", "weather", "climate"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
]
dependencies = [
    # The substrate, carved into its own installable member (uv workspace).
    # Everything still imports as ``legoesm.<subpkg>`` via the shared PEP-420
    # namespace; this is what lets ``legoesm-atmosphere`` / ``-ocean`` / ... be
    # shipped independently on top of the same core (see FEDERATION.md).  Pinned
    # to the matching minor (single-sourced version; members bump in lock-step)
    # so a non-workspace ``pip install legoesm`` cannot resolve a mismatched core.
    "legoesm-core~=0.1.0",
    # The four Earth-system components, each its own installable member on top of
    # legoesm-core (mutually independent — import-linter contract #2).  The meta
    # ``legoesm`` pulls them all; ``pip install legoesm-ocean`` alone is enough to
    # run the ocean standalone.
    "legoesm-atmosphere~=0.1.0",
    "legoesm-ocean~=0.1.0",
    "legoesm-land~=0.1.0",
    "legoesm-ice~=0.1.0",
    # The orchestration cluster (coupler/driver, ml/training/da, tools): the meta
    # package's own modules (cli, config, dycore_factory, taxonomy) import these,
    # so the meta-member pulls the whole stack.  They are mutually interdependent
    # (a uv-workspace cycle), unlike the four independent components above.
    "legoesm-coupler~=0.1.0",
    "legoesm-ml~=0.1.0",
    "legoesm-tools~=0.1.0",
    "jax>=0.4.35",              # install minimum; tested with 0.8–0.9
    "jaxlib>=0.4.35",           # must match jax
    "equinox>=0.11",
    "optax>=0.2",
    "xarray>=2024.0",
    "zarr>=2.18",
    "matplotlib>=3.9",
    "cartopy>=0.23",
    "pyyaml>=6.0",
    "numpy>=1.26",
    "netCDF4>=1.6",
    "scipy>=1.10",
    "pandas>=2.0",
]

[project.optional-dependencies]
# clm-ml-jax is published on PyPI (https://pypi.org/project/clm-ml-jax/).
# The 0.1.0 release ships the DIFFERENTIABLE canopy capability: its
# `_CanopyFluxesDiagnostics` accepts `grid=`, so the canopy-integrated flux
# diagnostics run on the jax.grad tape (CLMMLCanopyConfig(differentiable=True)).
# `clm_ml_interface` still PROBES this capability at runtime and raises a clear,
# actionable error if an older/local build lacks it (so there is never a silent
# wrong answer); forward-only mode (the default) needs only the base 0.1.0.
# Editable local checkouts (`pip install -e ./clm-ml-jax`) still shadow the
# published wheel for development.
canopy = ["clm-ml-jax>=0.1.0"]
# gcsfs/fsspec included: the ML training lane's DEFAULT ERA5 path is a gs://
# zarr store (load_era5_samples / era5_to_state), so the ml extra must open it
# without also requesting [data] (#817 papercut: the WB runbook env built
# --extras dev only and hit ModuleNotFoundError: gcsfs at first data load).
ml = ["imageio>=2.30", "equinox>=0.11", "optax>=0.2",
      "gcsfs>=2024.2", "fsspec>=2024.2"]
mesh = ["pymetis>=1.2", "scipy>=1.10"]
viz = ["matplotlib>=3.9", "cartopy>=0.23"]
data = [
    "gcsfs>=2024.2",
    "fsspec>=2024.2",
]
# Apple-Silicon local-GPU backend via MLX (PJRT plugin). Select with
# ``JAX_PLATFORMS=mps``.
# float32 ONLY — MLX has no float64, so ``JAX_ENABLE_X64=1`` arrays raise and
# complex128 downcasts to complex64; spectral/x64/conservation-grade and any
# mpi4jax-collective run must stay on ``JAX_PLATFORMS=cpu``. Single MpsDevice:
# no SPMD/MPI multi-device. Good for single-device float32 FV dycore / ML work.
mps = ["jax-mps"]
# Interactive configuration wizard (scripts/experiment/wizard.py, `legoesm wizard`).
wizard = ["questionary>=2.0"]
# Tested jax envelope for MPI is >=0.8,<0.10 (runtime-guarded in
# parallel/reductions.py).  jax 0.10 removed the legacy CustomCallV1 that
# mpi4jax 0.8.x used; mpi4jax 0.9.0 (the FFI rewrite, mpi4jax#289) switched to
# JAX's FFI mechanism and works on jax 0.10+ — so pip resolves the FFI 0.9.x by
# default here.  Do NOT cap jax here: uv resolves one universal lock across all
# extras, so a cap would downgrade the default (non-MPI) workspace install.
# For a working MPI stack install `-r requirements_mpi.txt` (pinned combo).
mpi = [
    "mpi4py>=4.1,<5",
    "mpi4jax>=0.8,<0.10",
]
dev = [
    "pytest>=8.0",
    "pytest-xdist",
    "pytest-timeout",
    "ruff",
    "mypy",
    "pre-commit",
    "import-linter>=2.0",
    # Federation packaging validation (scripts/validate_federation_packaging.py):
    # build the per-member wheels + root-absent install in CI.
    "build>=1.0",
    "hatchling",
]
docs = ["sphinx", "myst-parser", "furo"]
all = [
    "legoesm[ml,mesh,viz,data,dev,docs,wizard]",
]

[project.scripts]
legoesm = "legoesm.cli:main"

# NOTE: the "legoesm.sw_barotropics" entry points (the FV3 shallow-water dycore
# providers the ocean barotropic solver resolves by name) are declared by their
# PROVIDER, legoesm-atmosphere, NOT here — so `pip install legoesm-ocean
# legoesm-atmosphere` registers them without the root meta-package.  See
# packages/atmosphere/pyproject.toml.

[tool.hatch.build.targets.wheel]
packages = ["src/legoesm"]

# === uv workspace (federation carve) ===
# The repo is a uv workspace; the substrate lives in its own member
# (packages/core, dist name legoesm-core) and the root ``legoesm`` is the
# meta-member that depends on it.  Each member's folder drops the legoesm- prefix
# and has no src/ level (packages/<m>/legoesm/<subpkg>); the root depends on them
# via [tool.uv.sources].  All members ship into the
# one ``legoesm`` PEP-420 namespace, so imports are unchanged.
[tool.uv.workspace]
members = ["packages/*"]

# Every inter-member dependency resolves to the in-tree workspace member (never
# PyPI).  All members must be listed here, or uv would try to fetch e.g.
# legoesm-atmosphere~=0.1.0 from an index that does not publish it yet.
[tool.uv.sources]
# Coupled latlon-band slab step (C10): ``T`` for the atm temperature field
# (T = atm_state.T, N806) and the CLAUDE.md-mandated ``_K`` unit suffix on the
# freezing-floor config field (t_freeze_ocean_K, N815) read clearer than
# snake_case — same rationale as the physical-symbol entries above.
"packages/coupler/legoesm/coupler/coupled_latlon_band.py" = ["N806", "N815"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
# Vendored 3rd-party CLM-ML-JAX backend (BSD-3) — not type-annotated to legoESM's
# standard; excluded from type-checking here (audited upstream).
exclude = ['packages/land/legoesm/land/canopy/clm_ml_backend/']

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short -m 'not slow'"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    # Complexity-tier ladder (docs/validation/TESTING.md). Opt-in selectors: run a rung
    # with e.g. `-m 'tier1 and not slow'`. The default addopts stays `not slow`
    # (does NOT tier-gate) so untagged tests still run until the suite is tagged.
    "tier0: unit/operator tests — kernels, no model integration (numerical invariants)",
    "tier1: research tier — idealized/column/shallow-water; mass+energy+AAM gates, analytic benchmarks",
    "tier2: intermediate tier — hydrostatic 3D + slab; mass+energy+moisture gates (often slow)",
    "tier3: operational tier — full complexity + real forcing (AMIP/OMIP/ERA5); budget closure (slow)",
]

# ---------------------------------------------------------------------------
# Import-boundary enforcement (the path to independent self-running packages).
# These contracts encode the TARGET layered architecture (master plan D7):
#   core (substrate)  <  {atmosphere, ocean, land, ice}  <  coupler  <  driver
# Each Earth-system component must depend only on `core` (+ truly-shared bricks),
# never on another component or on the coupler/driver — that one-way dependency
# is exactly what lets `pip install legoesm-ocean` run standalone (Stage D carve).
#
# `ignore_imports` is the CURRENT-violation BASELINE: the contracts pass today,
# and every entry removed is a real step toward independence (a ratchet).  The
# goal is an empty ignore list = full component independence, enforced in CI.
# Run: `.venv/bin/lint-imports`
.github/workflows/ci.yml
.github/workflows/claude-code-review.yml
.github/workflows/claude.yml
.github/workflows/mpi-distributed.yml
.github/workflows/mpi-nightly.yml
.github/workflows/mpi-distributed.yml:20:        uses: actions/setup-python@v5
.github/workflows/mpi-distributed.yml:22:          python-version: "3.12"
.github/workflows/mpi-distributed.yml:52:              python -m pytest -q tests/distributed/test_halo_mpi.py
.github/workflows/mpi-distributed.yml:62:              python -m pytest -q tests/distributed/test_plane_pencil_mpi.py
.github/workflows/mpi-distributed.yml:75:              python -m pytest -q tests/distributed/test_latlon_2d_pad_wall_mpi.py
.github/workflows/mpi-distributed.yml:88:              python -m pytest -q tests/distributed/test_latlon_transpose_ad_mpi.py
.github/workflows/mpi-nightly.yml:19:        uses: actions/setup-python@v5
.github/workflows/mpi-nightly.yml:21:          python-version: "3.12"
.github/workflows/mpi-nightly.yml:47:            python -m pytest -q tests/distributed/test_ocean_mpi_conservation.py -k longrun
.github/workflows/ci.yml:23:        python-version: ["3.11", "3.12"]
.github/workflows/ci.yml:26:      - uses: actions/setup-python@v5
.github/workflows/ci.yml:28:          python-version: ${{ matrix.python-version }}
.github/workflows/ci.yml:49:      - uses: actions/setup-python@v5
.github/workflows/ci.yml:51:          python-version: "3.11"
.github/workflows/ci.yml:68:      - uses: actions/setup-python@v5
.github/workflows/ci.yml:70:          python-version: "3.11"
.github/workflows/ci.yml:92:      - uses: actions/setup-python@v5
.github/workflows/ci.yml:94:          python-version: "3.11"
.github/workflows/ci.yml:97:        run: python -m pytest --collect-only tests/ 2>&1
.github/workflows/ci.yml:105:        python-version: ["3.11"]
.github/workflows/ci.yml:108:      - uses: actions/setup-python@v5
.github/workflows/ci.yml:110:          python-version: ${{ matrix.python-version }}
.github/workflows/ci.yml:118:      # leaks across the pytest session for subsequently-imported modules.
.github/workflows/ci.yml:121:          python -m pytest tests/unit/ -x --timeout=300 \
.github/workflows/ci.yml:145:        python-version: ["3.11"]
.github/workflows/ci.yml:148:      - uses: actions/setup-python@v5
.github/workflows/ci.yml:150:          python-version: ${{ matrix.python-version }}
.github/workflows/ci.yml:156:        # `bash -c` is required so the glob expands before pytest
.github/workflows/ci.yml:157:        # sees the arguments; otherwise pytest gets the literal
.github/workflows/ci.yml:181:          # returns, the deterministic fallback is one pytest process per file.
.github/workflows/ci.yml:182:          python -m pytest "${files[@]}" --timeout=300 \
.github/workflows/ci.yml:196:          python -m pytest tests/grids/test_fv3_native_grid_phase1.py \
.github/workflows/ci.yml:213:          python -m pytest tests/grids/test_fv3_native_swcore_phase4.py \
.github/workflows/ci.yml:223:      - uses: actions/setup-python@v5
.github/workflows/ci.yml:225:          python-version: "3.11"
.github/workflows/ci.yml:229:          python -m pytest tests/unit/test_cmor_experiments_restart.py \
.github/workflows/ci.yml:239:      - uses: actions/setup-python@v5
.github/workflows/ci.yml:241:          python-version: "3.11"
.github/workflows/ci.yml:245:          python -m pytest tests/atmosphere/ -x --timeout=600 \
.github/workflows/ci.yml:263:      - uses: actions/setup-python@v5
.github/workflows/ci.yml:265:          python-version: "3.11"

--- bench changes only (compact) ---
diff --git a/scripts/bench/bench_voronoi_partition_methods.py b/scripts/bench/bench_voronoi_partition_methods.py
index 6aa0f4a40..ed60dc80f 100644
--- a/scripts/bench/bench_voronoi_partition_methods.py
+++ b/scripts/bench/bench_voronoi_partition_methods.py
@@ -3,31 +3,84 @@
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
@@ -123,43 +176,245 @@ def owner_for(mesh, method: str, n_ranks: int) -> np.ndarray:
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
 
@@ -180,16 +435,33 @@ def main() -> int:
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
 
     payload = {
         "rows": rows,
         "auto_resolves_to": resolve_partition_method("auto"),
         "step_time_pointer": (
             "step-time per method: bench_ocean_mpas_scaling.py / "
             "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
             "per launch, same case otherwise)"),
@@ -199,22 +471,42 @@ def main() -> int:
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

--- find test-only seams/new env identifiers ---
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:15:INTERPRETER via ``LEGOESM_PYTHON``, which is an existing production knob that
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:48:    stub = tmp_path / "stub_python"
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:69:    env["LEGOESM_PYTHON"] = str(stub)
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:95:    assert "BENCH=scripts/bench/bench_voronoi_partition_methods.py" in text
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:96:    assert "LEGOESM_SCHEDULE_COST_BENCH" not in text
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:53:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_s10_192.sbatch:21:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:36:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:26:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:31:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:33:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/ocean_latlon_hundreds.sbatch:23:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/prewarm_s10.sbatch:19:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:18:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/bin/python}"
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:26:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:33:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:96:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:109:BENCH=scripts/bench/bench_voronoi_partition_methods.py
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch:22:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/atm_latlon_144.sbatch:31:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/_env.sh:63:if [ -z "${LEGOESM_PYTHON:-}" ] && [ -x "$REPO/.venv/bin/python" ]; then
scripts/cluster/scaling_levante/_env.sh:64:  LEGOESM_PYTHON="$REPO/.venv/bin/python"
scripts/cluster/scaling_levante/_env.sh:66:PY="${LEGOESM_PYTHON:-$(command -v python)}"
scripts/cluster/scaling_levante/_env.sh:72:  echo "[_env.sh] Set LEGOESM_PYTHON=<repo>/.venv/bin/python (uv venv) or" >&2
scripts/cluster/scaling_levante/mpas_bound_base.sbatch:23:export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"

exec
/usr/bin/bash -lc "nl -ba .github/workflows/ci.yml | sed -n '1,310p'
printf '\\n--- test selection sources ---\\n'
rg -n 'tests/bench|tests/|pytest' .github/workflows scripts Makefile pyproject.toml | sed -n '1,500p'
printf '\\n--- zero-scorable-run control flow lines ---\\n'
nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '360,510p'
printf '\\n--- all test function names ---\\n'
rg -n '"'^def test_'"' tests/bench/test_bench_voronoi_partition_methods.py tests/bench/test_mpas_schedule_cost_scan_sbatch.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 306ms:
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

--- test selection sources ---
rg: Makefile: No such file or directory (os error 2)
pyproject.toml:108:    "pytest>=8.0",
pyproject.toml:109:    "pytest-xdist",
pyproject.toml:110:    "pytest-timeout",
pyproject.toml:191:"tests/grids/test_fv3_native_duo_phase4c.py" = ["N806"]
pyproject.toml:192:"tests/grids/test_fv3_native_geopk_pgrad.py" = ["N806"]
pyproject.toml:193:"tests/grids/test_fv3_native_pgrad.py" = ["N803", "N806"]
pyproject.toml:216:"tests/atmosphere/test_column_large_scale_extract*.py" = ["E402", "N806"]
pyproject.toml:221:"tests/unit/test_compare_reanalysis.py" = ["N803", "N806"]
pyproject.toml:224:"tests/run/test_correction_e2e_integration.py" = ["N806"]
pyproject.toml:231:"tests/unit/test_les_closure_diagnosis.py" = ["N803", "N806"]
pyproject.toml:234:"tests/unit/test_prognostic_aerosol_faithful.py" = ["N803", "N806"]
pyproject.toml:237:"tests/ocean/unit/test_ice_shelf_three_equation_faithful.py" = ["N803", "N806"]
pyproject.toml:240:"tests/ocean/unit/test_catke_faithful.py" = ["N803", "N806"]
pyproject.toml:243:"tests/ocean/unit/test_backscatter_reservoir_faithful.py" = ["N803", "N806"]
pyproject.toml:246:"tests/ocean/unit/test_enhanced_diffusion_faithful.py" = ["N806"]
pyproject.toml:249:"tests/ocean/unit/test_plume_convection_faithful.py" = ["N803", "N806"]
pyproject.toml:252:"tests/atmosphere/hydrostatic/unit/test_pbl_height_faithful.py" = ["N803", "N806"]
pyproject.toml:261:"tests/atmosphere/hydrostatic/unit/test_semi_implicit_gw_stability.py" = ["N803", "N806"]
pyproject.toml:262:"tests/unit/test_check_cross_grid_deploy.py" = ["N803"]
pyproject.toml:266:"tests/ocean/unit/test_nemo_roquet_eos.py" = ["E402", "N803", "N806"]
pyproject.toml:274:"tests/unit/test_era5_mpas_carry.py" = ["N806"]
pyproject.toml:279:"tests/unit/test_clubb_*.py" = ["E402", "N802", "N803", "N806"]
pyproject.toml:283:"tests/unit/test_warm_rain_fp32_saturation.py" = ["E402", "I001", "N803"]
pyproject.toml:289:"tests/parallel/test_cube_face_scatter.py" = ["E402"]
pyproject.toml:290:"tests/distributed/test_cube_face_scatter_mpi.py" = ["E402"]
pyproject.toml:294:"tests/parallel/test_mpas_atm_native_step.py" = ["N806"]
pyproject.toml:310:[tool.pytest.ini_options]
scripts/README.md:45:their dominant function (enforced by `tests/test_scripts_layout.py`):
.github/workflows/ci.yml:97:        run: python -m pytest --collect-only tests/ 2>&1
.github/workflows/ci.yml:118:      # leaks across the pytest session for subsequently-imported modules.
.github/workflows/ci.yml:121:          python -m pytest tests/unit/ -x --timeout=300 \
.github/workflows/ci.yml:128:    # Fortran-fidelity tests live at top-level `tests/test_*.py`
.github/workflows/ci.yml:129:    # (not under `tests/unit/`).  The pre-iter-874 CI ran only
.github/workflows/ci.yml:130:    # `tests/unit/` and `tests/atmosphere/`, so top-level tests
.github/workflows/ci.yml:132:    # job explicitly executes the top-level tests/test_*.py files
.github/workflows/ci.yml:136:    # Scope: only files matching `tests/test_*.py` at the top level
.github/workflows/ci.yml:137:    # of `tests/` — not subdirectories.  Subdirectory tests
.github/workflows/ci.yml:138:    # (`tests/atmosphere/`, `tests/distributed/`, etc.) have their
.github/workflows/ci.yml:152:      - name: Run top-level tests/test_*.py
.github/workflows/ci.yml:156:        # `bash -c` is required so the glob expands before pytest
.github/workflows/ci.yml:157:        # sees the arguments; otherwise pytest gets the literal
.github/workflows/ci.yml:158:        # `tests/test_*.py` and reports "no such file".
.github/workflows/ci.yml:161:          files=( tests/test_*.py )
.github/workflows/ci.yml:163:            echo "ERROR: no top-level tests/test_*.py files matched"
.github/workflows/ci.yml:181:          # returns, the deterministic fallback is one pytest process per file.
.github/workflows/ci.yml:182:          python -m pytest "${files[@]}" --timeout=300 \
.github/workflows/ci.yml:188:        # gates. tests/grids/ has no other CI job; without this step the
.github/workflows/ci.yml:196:          python -m pytest tests/grids/test_fv3_native_grid_phase1.py \
.github/workflows/ci.yml:197:            tests/grids/test_fv3_native_metrics_phase2.py \
.github/workflows/ci.yml:198:            tests/grids/test_fv3_native_halos_phase3.py \
.github/workflows/ci.yml:199:            tests/grids/test_gnomonic_ed_gate_iter68.py \
.github/workflows/ci.yml:200:            tests/grids/test_gnomonic_ed_centers_iter62.py \
.github/workflows/ci.yml:201:            tests/grids/test_gnomonic_ed_halo_nocollapse_iter73.py \
.github/workflows/ci.yml:213:          python -m pytest tests/grids/test_fv3_native_swcore_phase4.py \
.github/workflows/ci.yml:229:          python -m pytest tests/unit/test_cmor_experiments_restart.py \
.github/workflows/ci.yml:230:            tests/unit/test_zarr_checkpoint.py -v --tb=short
.github/workflows/ci.yml:245:          python -m pytest tests/atmosphere/ -x --timeout=600 \
.github/workflows/ci.yml:252:  # already gated in CI by tests/test_visual_regression_metrics.py (in unit-tests),
scripts/run/run_rce_convection_sweep.py:69:- No conservation tests in ``tests/unit/test_{zm,emanuel,bechtold,
scripts/run/run_rce_convection_sweep.py:408:    # schemes (``tests/unit/test_tiedtke.py``, ``test_zhang_mcfarlane.py``,
.github/workflows/mpi-distributed.yml:52:              python -m pytest -q tests/distributed/test_halo_mpi.py
.github/workflows/mpi-distributed.yml:62:              python -m pytest -q tests/distributed/test_plane_pencil_mpi.py
.github/workflows/mpi-distributed.yml:75:              python -m pytest -q tests/distributed/test_latlon_2d_pad_wall_mpi.py
.github/workflows/mpi-distributed.yml:88:              python -m pytest -q tests/distributed/test_latlon_transpose_ad_mpi.py
.github/workflows/mpi-nightly.yml:47:            python -m pytest -q tests/distributed/test_ocean_mpi_conservation.py -k longrun
scripts/cluster/les_scm/test_bridge.sbatch:24:$PY -m pytest tests/unit/test_les_record.py -q -p no:cacheprovider --noconftest
scripts/cluster/les_scm/test_bridge.sbatch:28:$PY -m pytest tests/unit/test_les_reference.py -q -p no:cacheprovider
scripts/cluster/les_scm/test_bridge.sbatch:32:$PY -m pytest tests/atmosphere/hydrostatic/unit/test_sam_case_scm.py -q \
scripts/cluster/les_scm/test_bridge.sbatch:37:$PY -m pytest tests/atmosphere/hydrostatic/test_run_scm_les_turbulence_tuning.py \
scripts/cluster/les_scm/test_bridge.sbatch:42:$PY -m pytest tests/atmosphere/hydrostatic/unit/test_scm.py \
scripts/cluster/les_scm/test_bridge.sbatch:43:             tests/atmosphere/hydrostatic/unit/test_scm_forcing.py \
scripts/cluster/les_scm/test_bridge.sbatch:44:             tests/unit/test_sam_case_forcing.py \
scripts/cluster/les_scm/test_bridge.sbatch:45:             tests/unit/test_scm_rce_metrics.py -q -p no:cacheprovider
scripts/bench/bench_mpas_spmd_scaling.py:83:# tests/parallel/test_voronoi_sharded_equivalence.py (u/T atol 1e-6, p_s
scripts/bench/bench_mpas_spmd_scaling.py:131:    # tests/parallel/test_voronoi_sharded_equivalence.py: del4 hyperdiffusion,
scripts/bench/bench_mpas_spmd_scaling.py:152:    # tests/parallel/test_mpas_partitionlocal_build.py).  The GLOBAL
scripts/bench/bench_mpas_spmd_scaling.py:362:    # sharded s0 per tests/parallel/test_mpas_partitionlocal_build.py).
scripts/experiment/smoke_compare_reanalysis.py:34:# pytest the repo root is already on the path, so the guard makes this a no-op there.
scripts/cluster/les_scm/codex_review.sbatch:39:  tests/atmosphere/hydrostatic/unit/test_sam_case_scm.py
scripts/cluster/les_scm/codex_review.sbatch:40:  tests/unit/test_les_reference.py
scripts/cluster/les_scm/codex_review.sbatch:41:  tests/atmosphere/hydrostatic/test_run_scm_les_turbulence_tuning.py
scripts/cluster/dchain_oracle.sbatch:32:  $PY -m pytest $REPO/tests/grids/test_fv3_native_duo_chain.py -q
scripts/experiment/ck_sensitivity_vs_era5.py:30:# when this file is run as a standalone CLI; under pytest the repo root is already on the
scripts/cluster/divduo_oracle.sbatch:35:  -m pytest $REPO/tests/grids/test_fv3_native_duo_phase4c.py -q
scripts/experiment/run_mpi_tests.sh:10:# guarded by a HARD wall-clock ``timeout`` (pytest's signal-timeout can't
scripts/experiment/run_mpi_tests.sh:27:mapfile -t FILES < <(ls tests/distributed/test_*.py | sort)
scripts/experiment/run_mpi_tests.sh:34:  timeout -s KILL "$TMO" mpirun -np "$NP" "$PY" -m pytest "$f" \
scripts/run/run_correction_campaign.py:48:# script-path invocation works (it already worked via ``-m`` / pytest, which put the CWD on path).
scripts/experiment/wizard_core.py:24:  built against — and ``tests/unit/test_wizard_core.py`` asserts the curated
scripts/cluster/les_scm/rcemip_oracle_codex.sbatch:41:that oracle, (c) adds tests/unit/test_rcemip_gsam_oracle_ic.py plus a tiny
scripts/cluster/les_scm/rcemip_oracle_codex.sbatch:42:vendored baseline tests/oracle_baselines/gsam_rcemip300_snd.json.
scripts/cluster/les_scm/rcemip_oracle_codex.sbatch:60:3. TEST VACUITY. tests/unit/test_rcemip_gsam_oracle_ic.py claims to fail for the
scripts/cluster/les_scm/rcemip_oracle_codex.sbatch:73:   (tests/validation/test_rcemip_plane_smoke.py,
scripts/cluster/les_scm/rcemip_oracle_codex.sbatch:74:   tests/atmosphere/hydrostatic/test_build_rcemip1_small_reference.py,
scripts/cluster/les_scm/rcemip_oracle_codex.sbatch:75:   tests/unit/test_rcemip_wing_ic.py,
scripts/cluster/les_scm/rcemip_oracle_codex.sbatch:76:   tests/unit/test_compute_reference_state_p_sfc_iter95.py). Were any anchors
scripts/experiment/setup_mpi_local.sh:52:echo "== installing legoESM (editable) + pytest, re-pinning jax 0.9.2 =="
scripts/experiment/setup_mpi_local.sh:53:uv pip install --python "$VENV/bin/python" -e . pytest pytest-timeout
scripts/experiment/setup_mpi_local.sh:66:mpirun -np 2 "$VENV/bin/python" -m pytest \
scripts/experiment/setup_mpi_local.sh:67:    tests/distributed/test_mpi_bootstrap.py tests/distributed/test_halo_mpi.py \
scripts/experiment/setup_mpi_local.sh:74:  mpirun -np 2 .venv-mpi/bin/python -m pytest tests/distributed/ -q
scripts/data/extract_gsam_rcemip_baseline.py:12:``tests/oracle_baselines/gsam_rcemip300_snd.json`` with FULL provenance (upstream
scripts/data/extract_gsam_rcemip_baseline.py:37:       --out tests/oracle_baselines/gsam_rcemip300_snd.json
scripts/data/extract_gsam_rcemip_baseline.py:207:                        f"--out tests/oracle_baselines/gsam_rcemip300_snd.json "
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:58:  "$PY" -m pytest -q --no-header --tb=short \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:59:      tests/unit/test_rcemip_wing_ic.py \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:60:      tests/unit/test_compute_reference_state_p_sfc_iter95.py \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:61:      tests/validation/test_rcemip_plane_smoke.py \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:62:      tests/atmosphere/hydrostatic/test_build_rcemip1_small_reference.py \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:84:  "$PY" -m pytest -q --tb=short \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:85:      tests/unit/test_rcemip_gsam_oracle_ic.py \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:86:      tests/unit/test_rcemip_wing_ic.py \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:87:      tests/unit/test_compute_reference_state_p_sfc_iter95.py \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:88:      tests/validation/test_rcemip_plane_smoke.py \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:89:      tests/atmosphere/hydrostatic/test_build_rcemip1_small_reference.py \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:90:      tests/atmosphere/nonhydrostatic/integration/test_plane_crm_end_to_end_smoke.py
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:93:  # CONTROL: the plane-CRM end-to-end smoke writes into /local pytest tmpdirs
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:105:    PYTHONPATH="${_mp}:${MAINWT}/src" "$PY" -m pytest -q --no-header --tb=line \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:106:        tests/atmosphere/nonhydrostatic/integration/test_plane_crm_end_to_end_smoke.py \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:112:  env -u LEGOESM_GSAM_ROOT "$PY" -m pytest -q \
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:113:      tests/unit/test_rcemip_gsam_oracle_ic.py
scripts/cluster/les_scm/rcemip_oracle_gate.sbatch:126:sys.path.insert(0, "tests/unit")
scripts/cluster/d2a2c_oracle.sbatch:33:  $PY -m pytest $REPO/tests/grids/test_fv3_native_d2a2c_duo.py -q
scripts/bench/bench_ocean_mpas_scaling.py:460:        # tests/ocean/distributed/test_barotropic_pcg_mpas_mpi.py and the
scripts/cluster/scaling_derecho/README.md:518:`tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py`.
scripts/cluster/scm_rce_paper/test_phase2_full.sbatch:18:JAX_ENABLE_X64=1 $PY -m pytest tests/unit/test_convection_subsidence_solve_threading.py \
scripts/cluster/scm_rce_paper/test_phase2_full.sbatch:19:  tests/unit/test_scm_rce_subsidence_solve_override.py -q 2>&1 | tail -30
scripts/cluster/scm_rce_paper/test_phase2_full.sbatch:21:JAX_ENABLE_X64=1 $PY -m pytest tests/unit/test_tiedtke.py tests/unit/test_emanuel.py \
scripts/cluster/scm_rce_paper/test_phase2_full.sbatch:22:  tests/unit/test_zhang_mcfarlane.py tests/unit/test_bechtold.py \
scripts/cluster/scm_rce_paper/test_phase2_full.sbatch:23:  tests/unit/test_bechtold_implicit_flux.py tests/unit/test_kain_fritsch.py tests/atmosphere/hydrostatic/unit/test_edmf_convection_824.py \
scripts/cluster/scm_rce_paper/test_phase2_full.sbatch:24:  tests/atmosphere/hydrostatic/unit/test_convection.py -q 2>&1 | tail -25
scripts/cluster/scm_rce_paper/test_phase2_full.sbatch:26:JAX_ENABLE_X64=1 $PY -m pytest tests/test_param_specs.py tests/test_no_inline_physics_coeffs.py \
scripts/cluster/scm_rce_paper/test_phase2_full.sbatch:27:  tests/test_dispatch_hardening.py tests/test_validate_strict_coverage.py \
scripts/cluster/scm_rce_paper/test_phase2_full.sbatch:28:  tests/test_physics_contracts.py tests/test_no_private_cross_imports.py \
scripts/cluster/scm_rce_paper/test_phase2_full.sbatch:29:  tests/test_no_hardcoded_constants.py tests/test_no_saturation_reimpl.py -q 2>&1 | tail -25
scripts/data/generate_aerobulk_reference.py:9:    tests/unit/baselines/aerobulk_noskin_v1.npz
scripts/data/generate_aerobulk_reference.py:10:    tests/unit/baselines/aerobulk_noskin_v1.json   (provenance metadata)
scripts/data/generate_aerobulk_reference.py:12:Consumed by ``tests/unit/test_aerobulk_oracle.py`` — the parity test needs
scripts/cluster/scm_rce_paper/codex_review_phase2.sbatch:36:  tests/unit/test_convection_subsidence_solve_threading.py
scripts/cluster/scm_rce_paper/codex_review_phase2.sbatch:69:   (tests/test_param_specs.py requires every :float-annotated field to be
scripts/cluster/scm_rce_paper/codex_review_phase2.sbatch:72:   tests/_param_spec_baseline.py and tests/_inline_coeff_baseline.py.
scripts/validate/validate_restart_continuation.py:13:``tests/unit/test_persist_physics_state.py::test_*_bitexact_restart_continuation``
scripts/cluster/scm_rce_paper/validator_baseline.sbatch:22:JAX_ENABLE_X64=1 $PY -m pytest tests/test_no_hardcoded_constants.py -q 2>&1 | tail -8
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs:20:# Correctness gates: tests/parallel/test_latlon_ocean_spmd_step.py +
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs:21:# tests/bench/test_bench_ocean_latlon_spmd_gates.py; the job
scripts/cluster/scm_rce_paper/isolate_nonhydro_failure.sbatch:12:T=tests/atmosphere/hydrostatic/unit/test_convection.py::TestIntegration::test_nonhydrostatic_nonzero_heating
scripts/cluster/scm_rce_paper/isolate_nonhydro_failure.sbatch:17:  JAX_ENABLE_X64=1 $PY -m pytest "$T" -q -p no:cacheprovider 2>&1 | tail -18
scripts/matrix/scm/gabls1.py:22:NetCDF in ``tests/validation/scm_oracle/`` rather than profile-by-
scripts/matrix/scm/gabls1.py:240:    # tests/validation/test_scm_gabls1.py::test_gabls1_lowest_cell_in_stable_band):
scripts/run/run_plane_rising_thermal.py:3:Nightly companion to ``tests/validation/test_plane_nh_rising_thermal.py``.
scripts/data/setup_jax_scm_oracle.sh:4:# the SCM oracle NetCDF outputs under ``tests/validation/scm_oracle/``.
scripts/data/setup_jax_scm_oracle.sh:11:#   benchmarks in ``tests/validation/test_scm_*.py`` load the resulting
scripts/data/stage_jra55_do_fesom_pool.py:384:    # late if something imported jax first (a pytest conftest, say).
scripts/cluster/scm_rce_paper/test_drivers.sbatch:17:JAX_ENABLE_X64=1 $PY -m pytest \
scripts/cluster/scm_rce_paper/test_drivers.sbatch:18:  tests/unit/test_scm_rce_convection_intercomparison_cli.py \
scripts/cluster/scm_rce_paper/test_drivers.sbatch:19:  tests/unit/test_train_scm_rce_params_cli.py -q 2>&1 | tail -35
scripts/cluster/scm_rce_paper/test_drivers.sbatch:21:JAX_ENABLE_X64=1 $PY -m pytest tests/ -q -k "scm_rce" 2>&1 | tail -20
scripts/cluster/scm_rce_paper/test_phase2.sbatch:19:JAX_ENABLE_X64=1 $PY -m pytest tests/unit/test_convection_subsidence_solve_threading.py -q 2>&1 | tail -30
scripts/cluster/scm_rce_paper/test_phase2.sbatch:22:JAX_ENABLE_X64=1 $PY -m pytest \
scripts/cluster/scm_rce_paper/test_phase2.sbatch:23:  tests/unit/test_tiedtke.py tests/unit/test_emanuel.py \
scripts/cluster/scm_rce_paper/test_phase2.sbatch:24:  tests/unit/test_zhang_mcfarlane.py tests/unit/test_bechtold_implicit_flux.py \
scripts/cluster/scm_rce_paper/test_phase2.sbatch:25:  tests/atmosphere/hydrostatic/unit/test_convection.py \
scripts/cluster/scm_rce_paper/test_phase2.sbatch:29:JAX_ENABLE_X64=1 $PY -m pytest \
scripts/cluster/scm_rce_paper/test_phase2.sbatch:30:  tests/test_param_specs.py tests/test_no_inline_physics_coeffs.py \
scripts/cluster/scm_rce_paper/test_phase2.sbatch:31:  tests/test_dispatch_hardening.py tests/test_validate_strict_coverage.py \
scripts/cluster/scm_rce_paper/test_phase2.sbatch:32:  tests/test_physics_contracts.py tests/test_no_private_cross_imports.py \
scripts/cluster/scm_rce_paper/test_phase2.sbatch:36:JAX_ENABLE_X64=1 $PY -m pytest tests/ -q -k "ml_parameterization" 2>&1 | tail -12
scripts/bench/bench_fv3_sw_fb_vs_production.py:219:    # policy makes later tests/callers order-dependent).
scripts/validate/sweep_ocean_tests.py:4:#1387: a plain ``pytest tests/ocean`` aborts at ~2% and dies under xdist. The
scripts/validate/sweep_ocean_tests.py:23:    python scripts/validate/sweep_ocean_tests.py --paths tests/ocean/unit
scripts/validate/sweep_ocean_tests.py:41:DEFAULT_PATHS = ("tests/ocean",)
scripts/validate/sweep_ocean_tests.py:72:    cmd += [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
scripts/validate/sweep_ocean_tests.py:93:        # a non-obvious status, so keep pytest's OWN summary line.
scripts/validate/sweep_ocean_tests.py:94:        "summary": tail[-1].strip() if tail else "(no pytest summary line)",
scripts/validate/sweep_ocean_tests.py:105:    ap.add_argument("-k", dest="k", default=None, help="pytest -k expression")
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py:4:Runs STANDALONE (not pytest): the root ``tests/conftest.py`` initializes the XLA
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py:58:# Grid / run sizing (module-level so the pytest wrapper can import without MPI).
scripts/cluster/scm_rce_paper/test_regression_split.sbatch:19:for f in tests/unit/test_tiedtke.py tests/unit/test_emanuel.py \
scripts/cluster/scm_rce_paper/test_regression_split.sbatch:20:         tests/unit/test_zhang_mcfarlane.py tests/unit/test_kain_fritsch.py \
scripts/cluster/scm_rce_paper/test_regression_split.sbatch:21:         tests/unit/test_bechtold_implicit_flux.py \
scripts/cluster/scm_rce_paper/test_regression_split.sbatch:22:         tests/atmosphere/hydrostatic/unit/test_edmf_convection_824.py \
scripts/cluster/scm_rce_paper/test_regression_split.sbatch:23:         tests/atmosphere/hydrostatic/unit/test_convection.py \
scripts/cluster/scm_rce_paper/test_regression_split.sbatch:24:         tests/unit/test_bechtold.py ; do
scripts/cluster/scm_rce_paper/test_regression_split.sbatch:26:  JAX_ENABLE_X64=1 $PY -m pytest $f -q -p no:cacheprovider 2>&1 | tail -6
scripts/validate/summarize_rce_trajectory.py:302:# update (locked by tests/unit/test_dod_doc_code_consistency.py).
scripts/data/build_legoesm_deck.py:1485:        ["Dycore progression suite",     "tests/validation/run_dycore_progression_suite.py",
scripts/data/build_legoesm_deck.py:1489:        ["MPI differentiability tests",  "tests/distributed/test_mpi_differentiability.py",
scripts/validate/validate_omip_precision.py:4:The cheap unit gate (tests/validation/test_precision_omip.py) proves the
scripts/validate/validate_omip_precision.py:58:    Same construction the ocean unit tests use (tests/ocean/unit/
scripts/cluster/scaling_derecho/mc_nccl_canary.sh:68:# rank id to the OMPI env contract the probe/tests/benches read. The world
scripts/cluster/scaling_derecho/mc_nccl_canary.sh:93:run_stage B-atm "$PORT_B1" "$PY" -m pytest -x -q \
scripts/cluster/scaling_derecho/mc_nccl_canary.sh:94:    tests/parallel/test_atm_latlon_spmd_multicontroller.py
scripts/cluster/scaling_derecho/mc_nccl_canary.sh:95:run_stage B-ocean "$PORT_B2" "$PY" -m pytest -x -q \
scripts/cluster/scaling_derecho/mc_nccl_canary.sh:96:    tests/parallel/test_latlon_ocean_spmd_multicontroller.py
scripts/matrix/scm/oracle.py:5:resulting trajectories under ``tests/validation/scm_oracle/<case>.nc``
scripts/matrix/scm/oracle.py:16:Each case writes a single NetCDF to ``tests/validation/scm_oracle/``.
scripts/matrix/scm/oracle.py:136:                   default=pathlib.Path("tests/validation/scm_oracle"),
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:168:# Gate: tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:190:# Gate: tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:214:# Gate: tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py.
scripts/cluster/csw_oracle.sbatch:33:  $PY -m pytest $REPO/tests/grids/test_fv3_native_c_sw_oracle.py -q
scripts/bench/validate_scaling.sh:42:    bash -c "JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python -m pytest \
scripts/bench/validate_scaling.sh:43:        tests/atmosphere/hydrostatic/unit/test_primitive_eq.py::TestHydrostaticToFV3VectorHalo \
scripts/bench/validate_scaling.sh:65:    bash -c "JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python -m pytest \
scripts/bench/validate_scaling.sh:66:        tests/test_bcw_benchmark_scan_steps.py \
scripts/bench/validate_scaling.sh:72:    bash -c "JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python -m pytest \
scripts/bench/validate_scaling.sh:73:        tests/test_pe_dycore_inline_imports.py \
scripts/run/run_scm_rce_convection_intercomparison.py:121:# tests/unit/test_scm_rce_convection_intercomparison_cli.py.
scripts/data/build_land_forcing_climatology.py:44:(``tests/land/unit/test_build_land_forcing_climatology.py``).
scripts/bench/run_levante_gpu_scaling.py:280:#       ``tests/distributed/test_latlon_mpi_step.py`` (np=2/4).  Still
scripts/bench/run_levante_gpu_scaling.py:290:# reference in ``tests/distributed/test_cube_face_scatter_mpi.py``,
scripts/bench/run_levante_gpu_scaling.py:1612:            # (tests/distributed/test_latlon_mpi_step.py: MPI == serial after
scripts/bench/run_levante_gpu_scaling.py:1764:    # single-process reference in tests/distributed/test_cube_face_scatter_mpi.py.
scripts/matrix/run_sea_ice_test_matrix.py:466:    # tests/unit/test_land_ice_sea_ice_thermo.py::
scripts/validate/validate_tiled_fv3_sw_multinode.py:9:device_count=N`` virtual-device path the pytest gate uses.  Each process then
scripts/cluster/scaling_derecho/cube_scaling_cpu_routeb.sh:27:# resolution) so the two lanes never overlay (tests/bench/
scripts/cluster/scaling_levante/README.md:98:`tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py`.
scripts/run/run_amip.py:274:    # tests/timestepping/test_ssp_rk3_scan_bit_equivalence.py).
scripts/run/run_amip.py:3015:        # tests/test_scripts_layout.py).
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:108:# tests/bench/test_mpas_schedule_cost_scan_sbatch.py.
scripts/bench/run_cpu_mpi_scaling.py:685:    tests/parallel/test_tiled_blocked_loop.py (np24, vs serial model.step).
scripts/bench/run_cpu_mpi_scaling.py:1551:             "tests/distributed/test_latlon_2d_mpi_step.py (mass<1e-12 + "
scripts/data/generate_mitgcm_barotropic_gyre_reference.py:39:``tests/ocean/fidelity/test_generate_mitgcm_gyre_reference.py``.
scripts/validate/validate_federation_packaging.py:51:#: tests/test_federation_plan.FEDERATION_MEMBERS (several members bundle more than
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:160:# mpi4jax >=2-node ceiling. Gate: tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:175:# Gate: tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:197:# Gate: tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py.
scripts/validate/validate_driver_tiled_dycore_parity.py:22:(tests/parallel/test_tiled_cc_step_adapter.py) measured the tiled-vs-
scripts/run/run_omip_core2.py:2528:    tests/parallel/test_persistent_sharded_ocean_loop.py).
scripts/run/run_omip_core2.py:3387:    (tests/unit/test_run_omip_core2_spmd_persistent_cli.py — supplementary
scripts/run/run_omip_core2.py:5979:    # tests/unit/test_run_omip_core2_spmd_persistent_cli.py); main() only
scripts/run/run_omip_core2.py:6659:            # contract, locked by tests/unit/test_sss_apply.py) — counted.
scripts/validate/fv3_native/gen_dswcore_oracle.sh:23:"$PY" "$REPO/tests/grids/fixtures/build_dsw_fixture.py" \
scripts/validate/fv3_native/gen_dswcore_oracle.sh:26:echo "install into tests/grids/fixtures/ to update the committed pair"
scripts/validate/fv3_native/gen_boundedgs_oracle.py:45:        REPO / "tests/grids/fixtures/fv3_boundedgs_oracle.npz"))
scripts/run/train_carbon_params.py:98:# which keeps it -- see tests/land/validation/test_carbon_calibration_gradient.py).
scripts/bench/run_scaling_diagnosis.py:83:    ``pytest`` from the repo root.
scripts/bench/run_scaling_diagnosis.py:251:    # lives under tests/test_cases — same routine the GPU/CPU scaling
scripts/run/run_w2_w5_cosine_bell_iter1030.py:7:Equivalent to `pytest tests/test_iter1032_dual_target_full_matrix.py`
scripts/run/run_w2_w5_cosine_bell_iter1030.py:8:but runnable directly without pytest, useful for first-time setup
scripts/run/run_w2_w5_cosine_bell_iter1030.py:22:# `tests/` is importable.
scripts/matrix/run_atmosphere_test_matrix.py:111:    (tests/distributed/test_latlon_mpi_polar_filter.py).
scripts/matrix/run_atmosphere_test_matrix.py:693:# Sentinel tests under ``tests/test_iter9*`` keep their own bit-identical
scripts/matrix/run_atmosphere_test_matrix.py:2994:        # ``tests/test_cases/williamson_extended.py`` and rotates
scripts/validate/fv3_native/gen_extproj_oracle.py:6:and projections into a fixture the pytest gate compares against
scripts/validate/fv3_native/gen_extproj_oracle.py:49:        REPO / "tests/grids/fixtures/fv3_extproj_oracle.npz"))
scripts/matrix/run_ocean_test_matrix.py:6283:            "ocean fidelity layer (tests/ocean/fidelity, "
scripts/cluster/omip_nemo/run_test_prescribed_flow.sbatch:34:$PY -m pytest tests/ocean/unit/test_prescribed_flow.py -v -x --tb=short
scripts/cluster/omip_nemo/run_test_prescribed_flow.sbatch:38:$PY -m pytest tests/ocean/unit/test_ab2_scope.py -v --tb=short
scripts/bench/bench_cube_shardmap_halo.py:165:    ``tests/parallel/test_cubed_sphere_spmd_step.py`` so the timed integrand is
scripts/bench/bench_coupled_latlon_scaling.py:120:    failure.  Delegates to the same logic as the pytest gate."""
scripts/run/run_rcemip_plane.py:867:    # convention used by tests/validation/test_plane_nh_rising_thermal.py).
scripts/run/run_les_plane.py:176:# with ``_CASES`` (enforced by tests/unit/test_run_les_plane_cli.py) so a new
scripts/cluster/fv3_native/k2e_auth_reverify.sbatch:20:$PY -m pytest $REPO/tests/grids/test_fv3_native_ext_vector.py -q -x \
scripts/cluster/fv3_native/dsw6_duo_oracle.sbatch:37:  $PY -m pytest $REPO/tests/grids/test_fv3_native_dsw6_duo.py -q
scripts/cluster/fv3_native/dsw2_duo_oracle.sbatch:33:  $PY -m pytest $REPO/tests/grids/test_fv3_native_dsw2_duo.py -q
scripts/cluster/omip_nemo/rerun_dino_l2_r2.sbatch:24:# dispatch test : tests/ocean/unit/test_dino_experiment.py::TestDINORecipes
scripts/cluster/omip_nemo/rerun_dino_l2_r2.sbatch:61:$PY -m pytest -q \
scripts/cluster/omip_nemo/rerun_dino_l2_r2.sbatch:62:  tests/ocean/unit/test_dino_experiment.py::TestDINORecipes \
scripts/cluster/omip_nemo/rerun_dino_l2_r2.sbatch:63:  "tests/ocean/unit/test_dino_experiment.py::TestDINOConfig::test_vmix_scheme_default_and_dispatch" \
scripts/cluster/omip_nemo/validate_changes.sbatch:19:JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -q \
scripts/cluster/omip_nemo/validate_changes.sbatch:20:  tests/ocean/unit/test_omip2_applicator.py \
scripts/cluster/omip_nemo/validate_changes.sbatch:21:  tests/ocean/unit/test_pgf_smc03_phase1.py \
scripts/cluster/omip_nemo/validate_changes.sbatch:22:  tests/ocean/unit/test_pgf_smc03_phase2.py \
scripts/cluster/omip_nemo/validate_changes.sbatch:23:  tests/ocean/unit/test_pgf_smc03_phase3.py 2>&1 | tail -30
scripts/cluster/omip_nemo/validate_changes.sbatch:24:echo "pytest_rc=${PIPESTATUS[0]}"
scripts/bench/bench_ocean_mpi_scaling.py:104:tests/ocean/distributed/test_ocean_mpi_tvd_parity.py).
scripts/bench/bench_ocean_mpi_scaling.py:202:# perturbation in tests/ocean/distributed/test_ocean_mpi_conservation.py.
scripts/bench/bench_ocean_mpi_scaling.py:705:    tests/ocean/distributed/test_ocean_mpi_conservation.py: cell-centred
scripts/bench/bench_ocean_mpi_scaling.py:2320:        # MPI-vs-serial (tests/ocean/distributed/
scripts/cluster/omip_nemo/_test_zdf_probe.sbatch:15:$PY -m pytest tests/ocean/unit/test_tendency_probe.py -x -q 2>&1 \
scripts/bench/bench_ocean_latlon_spmd_scaling.py:85:# mirror the equivalence gate (tests/parallel/test_latlon_ocean_spmd_step.py,
scripts/bench/bench_ocean_latlon_spmd_scaling.py:120:        # is gated by tests/parallel/test_latlon_ocean_spmd_tripole.py;
scripts/plot/plot_atmosphere_term_by_term_analytic.py:4:``tests/atmosphere/shallow_water/unit/test_term_by_term_analytic.py``.
scripts/plot/plot_atmosphere_term_by_term_analytic.py:317:        "Test surfaced as documented pytest.skip\n"
scripts/cluster/omip_nemo/test_applicator.sbatch:16:JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -x -q \
scripts/cluster/omip_nemo/test_applicator.sbatch:17:  tests/ocean/unit/test_omip2_applicator.py 2>&1 | tail -45
scripts/cluster/omip_nemo/test_applicator.sbatch:18:echo "pytest_rc=${PIPESTATUS[0]}"
scripts/cluster/omip_nemo/_test_gw_cumulative.sbatch:11:# Runs in the PINNED PR worktree; read-only (pytest writes only to --basetemp).
scripts/cluster/omip_nemo/_test_gw_cumulative.sbatch:26:$PY -m pytest \
scripts/cluster/omip_nemo/_test_gw_cumulative.sbatch:27:  tests/unit/test_run_omip_core2_gateway_transports.py \
scripts/cluster/omip_nemo/_test_gw_cumulative.sbatch:28:  tests/ocean/unit/test_diagnostics_sections.py \
scripts/cluster/omip_nemo/_test_gw_cumulative.sbatch:34:$PY -m pytest \
scripts/cluster/omip_nemo/_test_gw_cumulative.sbatch:35:  tests/ocean/unit/test_mass_flux_store.py \
scripts/cluster/omip_nemo/run_faithful_trp2_tke_ab.sbatch:59:JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -q \
scripts/cluster/omip_nemo/run_faithful_trp2_tke_ab.sbatch:60:  tests/ocean/unit/test_tripole_vmix_cli.py 2>&1 | tail -15
scripts/cluster/fv3_native/geopk_pgrad_oracle.sbatch:83:  test -s "$REPO/tests/grids/fixtures/geopk_pgrad_oracle_c12_km${KM}.npz"
scripts/cluster/fv3_native/geopk_pgrad_oracle.sbatch:91:  echo "  tests/grids/fixtures/geopk_pgrad_oracle_c12_km${KM}.npz"
scripts/cluster/fv3_native/geopk_pgrad_oracle.sbatch:93:git -C "$REPO" status --short -- tests/grids/fixtures/ || true
scripts/cluster/fv3_native/geopk_pgrad_oracle.sbatch:97:  $PY -m pytest \
scripts/cluster/fv3_native/geopk_pgrad_oracle.sbatch:98:    "$REPO/tests/grids/test_fv3_native_geopk_pgrad.py" \
scripts/cluster/fv3_native/geopk_pgrad_oracle.sbatch:99:    "$REPO/tests/grids/test_fv3_native_pgrad.py" \
scripts/cluster/fv3_native/geopk_pgrad_oracle.sbatch:100:    "$REPO/tests/grids/test_fv3_native_duo_stepper.py" \
scripts/cluster/omip_nemo/run_dino_r1_kpp.sbatch:29:  --nemo-gridt /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/tests/DINO_R1/EXP00/DINO_1m_grid_T.nc \
scripts/cluster/omip_nemo/run_dino_r1_kpp.sbatch:30:  --nemo-gridu /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/tests/DINO_R1/EXP00/DINO_1m_grid_U.nc \
scripts/cluster/fv3_native/cornerlag_oracle.sbatch:21:$PY -m pytest $REPO/tests/grids/test_fv3_native_ext_vector.py -q -x \
scripts/cluster/fv3_native/dsw1_duo_oracle.sbatch:32:  $PY -m pytest $REPO/tests/grids/test_fv3_native_dsw1_duo.py -q
scripts/cluster/fv3_native/ext_bundle_gate.sbatch:22:    $REPO/tests/grids/test_fv3_native_ext_vector.py \
scripts/cluster/fv3_native/ext_bundle_gate.sbatch:26:$PY -m pytest $REPO/tests/grids/test_fv3_native_ext_vector.py -x -q
scripts/cluster/fv3_native/ext_bundle_gate.sbatch:29:$PY -m pytest $REPO/tests/grids -k "fv3_native and (duo or dsw or ext)" -q
scripts/validate/fv3_native/gen_oracle.sh:7:#   tests/grids/fixtures/fv3_gnomonic_ed_oracle.npz
scripts/validate/fv3_native/gen_oracle.sh:23:mkdir -p "$REPO_ROOT/tests/grids/fixtures"
scripts/validate/fv3_native/gen_oracle.sh:24:python - "$WORKDIR" "$REPO_ROOT/tests/grids/fixtures/fv3_gnomonic_ed_oracle.npz" <<'EOF'
scripts/cluster/omip_nemo/run_scm_twins.sbatch:40:echo "=== unit tests: tests/ocean/unit/test_scm_column_twins.py ==="
scripts/cluster/omip_nemo/run_scm_twins.sbatch:41:$PY -u -m pytest tests/ocean/unit/test_scm_column_twins.py -q
scripts/validate/fv3_native/gen_dsw3_duo_oracle.py:7:``tests/grids/fixtures/dswcore_input.npz``.  The fixture additionally
scripts/validate/fv3_native/gen_dsw3_duo_oracle.py:12:                                      write tests/grids/fixtures/
scripts/cluster/omip_nemo/_mutation_gw_cumulative.sbatch:30:DRV_T=tests/unit/test_run_omip_core2_gateway_transports.py
scripts/cluster/omip_nemo/_mutation_gw_cumulative.sbatch:31:SECT_T=tests/ocean/unit/test_diagnostics_sections.py
scripts/cluster/omip_nemo/_mutation_gw_cumulative.sbatch:32:MFS_T=tests/ocean/unit/test_mass_flux_store.py
scripts/cluster/omip_nemo/_mutation_gw_cumulative.sbatch:58:# ENFORCEMENT (codex round-1 YELLOW 7).  Printing a pytest status is not a
scripts/cluster/omip_nemo/_mutation_gw_cumulative.sbatch:69:run() {      # run <label> <pytest args...>
scripts/cluster/omip_nemo/_mutation_gw_cumulative.sbatch:72:  $PY -m pytest "$@" -q --no-header -p no:cacheprovider --basetemp="$BT/$label" 2>&1 \
scripts/cluster/omip_nemo/_mutation_gw_cumulative.sbatch:310:MFS_T=tests/ocean/unit/test_mass_flux_store.py
scripts/cluster/omip_nemo/_mutation_gw_cumulative.sbatch:388:$PY -m pytest "$DRV_T" "$SECT_T" -q --no-header -p no:cacheprovider \
scripts/cluster/omip_nemo/_mutation_gw_cumulative.sbatch:393:$PY -m pytest "$MFS_T" -q --no-header -p no:cacheprovider \
scripts/cluster/wb_forecast/run_pytest.sbatch:4:#SBATCH --job-name=wb_pytest
scripts/cluster/wb_forecast/run_pytest.sbatch:8:#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM_wbforecast/scripts/cluster/wb_forecast/logs/wb_pytest_%j.out
scripts/cluster/wb_forecast/run_pytest.sbatch:10:# Usage: sbatch --export=ALL,WB_PYTEST_TARGET="tests/evaluations/..." run_pytest.sbatch
scripts/cluster/wb_forecast/run_pytest.sbatch:17:echo "target: ${WB_PYTEST_TARGET:-tests/evaluations/}"
scripts/cluster/wb_forecast/run_pytest.sbatch:19:"${WB_PYTHON}" -m pytest ${WB_PYTEST_TARGET:-tests/evaluations/} -v -p no:cacheprovider
scripts/validate/fv3_native/gen_swcore_oracle.sh:43:      f"(install into tests/grids/fixtures/ alongside swcore_input.npz)")
scripts/cluster/wb_forecast/run_dp_test.sbatch:17:"${WB_PYTHON}" -m pytest tests/ml/test_data_parallel.py -v -p no:cacheprovider
scripts/cluster/fv3_native/dsw4_duo_oracle.sbatch:33:  $PY -m pytest $REPO/tests/grids/test_fv3_native_dsw4_duo.py -q
scripts/run/run_coupled.py:124:    (tests/unit/test_params_reachability_audit.py) walks THIS bundle, so the
scripts/run/run_coupled.py:291:    # tests/unit/test_air_sea_scheme_consistency.py pinned; that test now
scripts/run/run_coupled.py:315:    (tests/unit/test_run_coupled_config_yaml.py)."""
scripts/cluster/fv3_native/extproj_oracle.sbatch:21:# post-flight: the committed pytest gate must pass on the fresh fixture
scripts/cluster/fv3_native/extproj_oracle.sbatch:22:$PY -m pytest $REPO/tests/grids/test_fv3_native_ext_vector.py -q -x
scripts/cluster/omip_nemo/run_dino_r1_zco.sbatch:29:  --nemo-gridt /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/tests/DINO_R1/EXP00/DINO_1m_grid_T.nc \
scripts/cluster/omip_nemo/run_dino_r1_zco.sbatch:30:  --nemo-gridu /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/tests/DINO_R1/EXP00/DINO_1m_grid_U.nc \
scripts/validate/ocean_fidelity/dino_1226/deep_box_heat_budget.py:376:# breaks -- not a pytest suite, this is a scratch #1226 probe script)
scripts/cluster/fv3_native/dsw3_duo_oracle.sbatch:33:  $PY -m pytest $REPO/tests/grids/test_fv3_native_dsw3_duo.py $REPO/tests/grids/test_fv3_native_dswcore_phase4b.py -q
scripts/run/run_omip.py:1553:        # ``omip_nemo_match_mpas_v1`` by tests/ocean/unit/test_recipes.py), then
scripts/run/run_omip.py:1645:        # ``omip_nemo_match_tripole_v1`` by tests/ocean/unit/test_recipes.py),
scripts/cluster/fv3_native/dsw5_duo_oracle.sbatch:35:  $PY -m pytest $REPO/tests/grids/test_fv3_native_dsw5_duo.py -q
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:38:T_UNITS=tests/unit/test_seg_precip_units.py
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:39:T_AD=tests/ice/unit/test_ice_float32_grad_safety.py
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:40:T_C2=tests/coupler/unit/test_coupled_lane_carry_aux.py
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:41:T_XG=tests/coupler/unit/test_grid_remap_wet_mask.py
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:42:T_F4=tests/coupler/unit/test_segment_callback_elapsed_dt.py
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:43:T_H5=tests/grids/test_tripole_internals.py::TestRotationAngleSignAndSeam
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:44:T_F5=tests/unit/test_coupled_ocean_ice_forcing.py
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:45:T_F2=tests/unit/test_diagnostics_water_budget.py
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:46:T_OMIP=tests/unit/test_omip_sea_ice_coupling.py
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:47:T_M8=tests/unit/test_conservative_regrid.py
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:48:T_M8B=tests/coupler/unit/test_grid_remap_coupling.py
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:63:mkdir -p "$PREFIX/tests/unit" "$PREFIX/tests/ice/unit"
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:78:  # Build node ids from the FIXED source by grep, NOT pytest --collect-only:
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:79:  # collection depends on the sourced env / conftest / pytest version and once
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:99:    JAX_ENABLE_X64="$x64" "$PY" -m pytest -q --tb=no "$id" >/dev/null 2>&1
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:118:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q --tb=line "$T_UNITS"
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:159:_h5f="tests/grids/test_tripole_internals.py"
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:165:  JAX_ENABLE_X64=1 "$PY" -m pytest -q --tb=no \
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:178:_f5f="tests/unit/test_coupled_ocean_ice_forcing.py"
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:180:JAX_ENABLE_X64=1 "$PY" -m pytest -q --tb=no \
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:187:JAX_ENABLE_X64=1 "$PY" -m pytest -q --tb=no \
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:200:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q --tb=line "$T_F2" >/dev/null 2>&1
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:207:_omipf="tests/unit/test_omip_sea_ice_coupling.py"
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:209:JAX_ENABLE_X64=1 "$PY" -m pytest -q --tb=no \
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:216:_m8f="tests/unit/test_conservative_regrid.py"
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:219:  JAX_ENABLE_X64=1 "$PY" -m pytest -q --tb=no "${_m8f}::${_m}" >/dev/null 2>&1
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:234:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q \
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:235:  tests/coupler/unit/test_mpas_surface_flux_export.py
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:247:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_UNITS";      RC=$((RC + $?))
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:249:JAX_ENABLE_X64=0 "$PY" -u -m pytest -q "$T_AD";         RC=$((RC + $?))
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:251:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_C2";         RC=$((RC + $?))
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:253:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_XG";         RC=$((RC + $?))
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:255:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_F4";         RC=$((RC + $?))
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:257:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_H5";         RC=$((RC + $?))
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:259:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_F5";         RC=$((RC + $?))
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:261:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_F2";         RC=$((RC + $?))
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:263:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_OMIP";       RC=$((RC + $?))
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:265:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_M8" "$T_M8B"; RC=$((RC + $?))
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:273:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q \
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:274:  tests/parallel/test_operator_split_spmd_driver_parity.py \
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:275:  tests/parallel/test_operator_split_tiled_cube_driver_parity.py
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:281:JAX_ENABLE_X64=1 "$PY" -u -m pytest -q \
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:282:  tests/coupler/ \
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:283:  tests/ice/ \
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:284:  tests/unit/test_conservative_regrid.py \
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:285:  tests/unit/test_conservative_regrid_curvilinear.py \
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch:286:  tests/unit/test_conservative_regrid_unstructured.py
scripts/cluster/omip_nemo/run_dino_r1_tke.sbatch:29:  --nemo-gridt /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/tests/DINO_R1/EXP00/DINO_1m_grid_T.nc \
scripts/cluster/omip_nemo/run_dino_r1_tke.sbatch:30:  --nemo-gridu /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/tests/DINO_R1/EXP00/DINO_1m_grid_U.nc \
scripts/cluster/carbon_calibration/validate_fast_analytic.sbatch:33:$PY -m pytest tests/land/unit/test_fast_analytic.py -q -s; R1=$?
scripts/cluster/carbon_calibration/validate_fast_analytic.sbatch:36:$PY -m pytest tests/land/unit/test_train_carbon_params.py -q -s; R2=$?
scripts/cluster/carbon_calibration/test_chunked_grad.sbatch:10:# Verify the chunked EXACT-gradient accumulation.  Two ISOLATED pytest processes
scripts/cluster/carbon_calibration/test_chunked_grad.sbatch:33:$PY -m pytest tests/land/unit/test_train_carbon_params.py -q -s \
scripts/cluster/carbon_calibration/test_chunked_grad.sbatch:38:$PY -m pytest tests/land/unit/test_train_carbon_params.py -q -s \
scripts/cluster/carbon_calibration/retest_fast.sbatch:19:$PY -m pytest tests/land/unit/test_soc_observations.py \
scripts/cluster/carbon_calibration/retest_fast.sbatch:20:    tests/land/unit/test_train_carbon_params.py -q -k "not quick_dry_run"
scripts/validate/fv3_native/gen_dsw4_duo_oracle.py:11:                                      write tests/grids/fixtures/
scripts/cluster/carbon_calibration/validate_carbon_trainer.sbatch:31:$PY -m pytest tests/land/unit/test_soc_observations.py -q; R1=$?
scripts/cluster/carbon_calibration/validate_carbon_trainer.sbatch:34:$PY -m pytest tests/land/unit/test_train_carbon_params.py -q -s; R2=$?
scripts/cluster/carbon_calibration/validate_carbon_trainer.sbatch:37:$PY -m pytest tests/land/validation/test_carbon_calibration_gradient.py -q -s; R3=$?
scripts/plot/plot_les_dycore_analytic.py:2:verification (companion to ``tests/unit/test_spectral_les_analytic.py``).
scripts/plot/plot_term_by_term_analytic.py:3:Mirrors the scenarios in ``tests/ocean/unit/test_term_by_term_analytic.py``
scripts/plot/plot_term_by_term_analytic.py:528:    # `tests/ocean/unit/test_term_by_term_analytic.py`.
scripts/plot/plot_crm_dycore_analytic.py:3:``tests/unit/test_compressible_euler_plane_analytic.py``).
scripts/cluster/sw_dissipation/validate_sw_dissipation_fix.sbatch:33:$P -m pytest -x -q \
scripts/cluster/sw_dissipation/validate_sw_dissipation_fix.sbatch:34:  tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py \
scripts/cluster/sw_dissipation/validate_sw_dissipation_fix.sbatch:35:  tests/atmosphere/shallow_water/unit/test_spectral_sw_state_truncation.py \
scripts/cluster/sw_dissipation/validate_sw_dissipation_fix.sbatch:36:  tests/atmosphere/shallow_water/unit/test_spectral.py \
scripts/cluster/sw_dissipation/validate_sw_dissipation_fix.sbatch:37:  tests/atmosphere/shallow_water/unit/test_nesting_latlon.py \
scripts/cluster/sw_dissipation/validate_sw_dissipation_fix.sbatch:38:  tests/atmosphere/shallow_water/unit/test_colliding_modons_ic.py \
scripts/cluster/sw_dissipation/validate_sw_dissipation_fix.sbatch:39:  tests/atmosphere/shallow_water/unit/test_williamson_extended.py \
scripts/validate/fv3_native/fv3_gnomonic_oracle.f90:66:! tests/grids/fixtures/fv3_gnomonic_ed_oracle.npz.
scripts/validate/ocean_fidelity/build_recipe_comparison.py:15:tests/ocean/fidelity/test_recipe_comparison.py (``--check``).
scripts/validate/ocean_fidelity/build_recipe_comparison.py:115:        "enforced by `tests/ocean/fidelity/test_recipe_comparison.py`.",
scripts/validate/fv3_native/gen_duogrid_oracle_n2.sh:12:#   tests/grids/fixtures/fv3_duogrid_oracle_n2.npz
scripts/validate/fv3_native/gen_duogrid_oracle_n2.sh:44:mkdir -p "$REPO_ROOT/tests/grids/fixtures"
scripts/validate/fv3_native/gen_duogrid_oracle_n2.sh:45:python3 - "$WORKDIR" "$REPO_ROOT/tests/grids/fixtures/fv3_duogrid_oracle_n2.npz" << 'PACKEOF'
scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py:75:``tests/ocean/unit/test_vface_metric_consistency_mercator.py``) and cell/face
scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py:600:                                                              "tests/ocean/unit/test_nemo_mld_criterion.py). RE-MEASURED at "
scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py:629:                                                              "coordinate (tests/ocean/unit/test_nemo_mld_criterion.py) -- "
scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py:1837:                                                              "tests/ocean/unit/test_nemo_ldf_lap_viscosity_e3.py."),
scripts/validate/ocean_fidelity/build_recipe_case_board.py:7:fresh by tests/ocean/fidelity/test_recipe_case_board.py (which calls ``--check``).
scripts/validate/ocean_fidelity/build_recipe_case_board.py:63:        "`tests/ocean/fidelity/test_recipe_case_board.py`.",
scripts/validate/ocean_fidelity/run_scm_column_twins.py:304:    unit test (tests/ocean/unit/test_scm_column_twins.py).
scripts/cluster/unified_training/run_pytest.sbatch:4:#SBATCH --job-name=ut_pytest
scripts/cluster/unified_training/run_pytest.sbatch:8:#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/logs/ut_pytest_%j.out
scripts/cluster/unified_training/run_pytest.sbatch:10:# Usage: sbatch --export=ALL,UT_PYTEST_TARGET="tests/unit/test_curriculum.py" run_pytest.sbatch
scripts/cluster/unified_training/run_pytest.sbatch:17:echo "target: ${UT_PYTEST_TARGET:-tests/unit/test_curriculum.py}"
scripts/cluster/unified_training/run_pytest.sbatch:19:"${UT_PYTHON}" -m pytest ${UT_PYTEST_TARGET:-tests/unit/test_curriculum.py} -v -p no:cacheprovider
scripts/validate/ocean_fidelity/dino_1226/box_budget_run.py:16:tests/ocean/unit/test_box_heat_budget.py::test_accumulator_does_not_alter_trajectory
scripts/validate/ocean_fidelity/dino_1226/hpg_tendency_compare.py:80:  analytically instead by ``tests/ocean/unit/test_nemo_sco_step_pgf.py``
scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py:19:``tests/ocean/unit/test_dino_1226_instruments.py``): a rest-state start must be
scripts/validate/ocean_fidelity/dino_1226/acc_momentum_budget.py:92:``tests/ocean/unit/test_momentum_diagnostics_closure.py``) -- no re-derived
scripts/validate/ocean_fidelity/dino_1226/acc_momentum_budget.py:467:    # (closure-tested in tests/ocean/unit/test_momentum_diagnostics_closure.py:
scripts/validate/fv3_native/gen_dsw2_duo_oracle.py:7:``tests/grids/fixtures/dswcore_input.npz``; the same sha256 is stored
scripts/validate/fv3_native/gen_dsw2_duo_oracle.py:12:                                      write tests/grids/fixtures/
scripts/validate/fv3_native/gen_duogrid_oracle.sh:9:#   tests/grids/fixtures/fv3_duogrid_oracle.npz
scripts/validate/fv3_native/gen_duogrid_oracle.sh:34:mkdir -p "$REPO_ROOT/tests/grids/fixtures"
scripts/validate/fv3_native/gen_duogrid_oracle.sh:35:python - "$WORKDIR" "$REPO_ROOT/tests/grids/fixtures/fv3_duogrid_oracle.npz" < "$HERE/_pack_duogrid_oracle.py"
scripts/validate/fv3_native/gen_dsw1_duo_oracle.py:5:``tests/grids/fixtures/dswcore_input.npz`` into the ``dswcore_input.txt``
scripts/validate/fv3_native/gen_dsw1_duo_oracle.py:12:                                      write tests/grids/fixtures/
scripts/validate/fv3_native/gen_dsw6_duo_oracle.py:15:                                      write tests/grids/fixtures/
scripts/validate/fv3_native/gen_dsw5_duo_oracle.py:16:                                      write tests/grids/fixtures/
scripts/validate/fv3_native/gen_cornerlag_oracle.py:77:        REPO / "tests/grids/fixtures/fv3_cornerlag_oracle.npz"))
scripts/validate/fv3_native/run_duo_stepper_modon.py:7:IC: the faithful case-8 port from tests/test_cases/colliding_modons
scripts/validate/visual_regression.py:10:     cubed sphere — reusing the canonical ``tests/williamson_diagnostic.py``
scripts/validate/visual_regression.py:32:``tests/test_visual_regression_metrics.py``; the full sim runs in a nightly CI job.
scripts/validate/run_perfect_model_osse.py:25:# already worked via ``-m`` / pytest, which put the CWD on path). MUST precede the ``scripts.*``
scripts/validate/fv3_native/gen_duogrid_oracle_auth.sh:7:#   tests/grids/fixtures/fv3_duogrid_oracle_auth.npz
scripts/validate/fv3_native/gen_duogrid_oracle_auth.sh:39:mkdir -p "$REPO_ROOT/tests/grids/fixtures"
scripts/validate/fv3_native/gen_duogrid_oracle_auth.sh:41:    "$REPO_ROOT/tests/grids/fixtures/fv3_duogrid_oracle_auth.npz" \
scripts/validate/compare_ml_bigleaf_ec.py:56:``tests/land/integration/test_ml_bigleaf_ec_compare.py`` without the heavy model.

--- zero-scorable-run control flow lines ---
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
   461	    payload = {
   462	        "rows": rows,
   463	        "auto_resolves_to": resolve_partition_method("auto"),
   464	        "step_time_pointer": (
   465	            "step-time per method: bench_ocean_mpas_scaling.py / "
   466	            "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
   467	            "per launch, same case otherwise)"),
   468	        "metadata": annotate_incomplete(scaling_metadata(
   469	            grid="voronoi",
   470	            component="partitioning",
   471	            resolution=f"L{args.subdivision}",
   472	            n_levels=0,
   473	            precision="n/a",
   474	            decomposition="cell_partition",
   475	            solver_variant="n/a",
   476	            scaling_kind="partition-quality",
   477	            transport="none",
   478	            extra={"rank_counts": rank_counts, "methods": methods,
   479	                   "halo_depth": args.halo_depth,
   480	                   "schedule_cost": bool(args.schedule_cost),
   481	                   "lloyd_iterations": args.lloyd},
   482	        )),
   483	    }
   484	    outdir = os.path.dirname(args.out)
   485	    if outdir:
   486	        os.makedirs(outdir, exist_ok=True)
   487	    with open(args.out, "w") as f:
   488	        json.dump(payload, f, indent=2)
   489	    print(f"JSON: {args.out}")
   490	
   491	    if expect:
   492	        failures = check_expected_rounds(rows, expect)
   493	        payload["expected_rounds_check"] = {
   494	            "expected": {f"{m}:{n}": v for (m, n), v in sorted(expect.items())},
   495	            "failures": failures,
   496	            "passed": not failures,
   497	        }
   498	        with open(args.out, "w") as f:
   499	            json.dump(payload, f, indent=2)
   500	        if failures:
   501	            print(f"INSTRUMENT CHECK FAILED ({len(failures)}):")
   502	            for line in failures:
   503	                print(f"  {line}")
   504	            print("The scorer did NOT reproduce the known census — treat every "
   505	                  "unknown row in this run as UNTRUSTED.")
   506	            return 1
   507	        print(f"INSTRUMENT CHECK PASSED ({len(expect)} expectations)")
   508	    return 0
   509	
   510	

--- all test function names ---
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:87:def test_launcher_is_not_redirectable_from_the_environment():
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:100:def test_validation_failure_aborts_before_the_unknown_arm(
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:117:def test_arm3_failure_is_not_reported_as_success(tmp_path):
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:129:def test_full_success_exits_zero_and_runs_every_arm(tmp_path):
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:136:def test_each_arm_carries_the_flags_its_claim_depends_on(tmp_path):
tests/bench/test_bench_voronoi_partition_methods.py:25:def test_partition_quality_metrics_shape_and_sanity():
tests/bench/test_bench_voronoi_partition_methods.py:38:def test_partition_quality_rejects_bad_owner():
tests/bench/test_bench_voronoi_partition_methods.py:47:def test_owner_for_unknown_method_raises():
tests/bench/test_bench_voronoi_partition_methods.py:53:def test_method_available_never_substitutes():
tests/bench/test_bench_voronoi_partition_methods.py:65:def test_main_writes_quality_table(tmp_path, monkeypatch):
tests/bench/test_bench_voronoi_partition_methods.py:81:def test_main_rejects_bad_args(monkeypatch):
tests/bench/test_bench_voronoi_partition_methods.py:105:def test_metric_definitions_locked_on_synthetic_mesh():
tests/bench/test_bench_voronoi_partition_methods.py:126:def test_empty_rank_rejected():
tests/bench/test_bench_voronoi_partition_methods.py:133:def test_halo_matches_runtime_partition():
tests/bench/test_bench_voronoi_partition_methods.py:154:def test_schedule_cost_row_reports_rounds_against_their_lower_bound():
tests/bench/test_bench_voronoi_partition_methods.py:181:def test_coloring_gap_and_headroom_are_derived_not_assumed(
tests/bench/test_bench_voronoi_partition_methods.py:218:def test_schedule_rounds_never_beat_the_vizing_floor_on_a_real_mesh():
tests/bench/test_bench_voronoi_partition_methods.py:232:def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
tests/bench/test_bench_voronoi_partition_methods.py:242:def test_schedule_cost_matches_the_production_scorer_exactly():
tests/bench/test_bench_voronoi_partition_methods.py:257:def test_schedule_cost_flag_is_opt_in_and_lands_in_the_json(
tests/bench/test_bench_voronoi_partition_methods.py:285:def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
tests/bench/test_bench_voronoi_partition_methods.py:317:def test_expect_rounds_gate_fails_loudly_and_never_vacuously(
tests/bench/test_bench_voronoi_partition_methods.py:354:def test_expect_rounds_refuses_to_pass_vacuously_without_scoring(monkeypatch):
tests/bench/test_bench_voronoi_partition_methods.py:366:def test_expect_rounds_rejects_malformed_specs(spec):
tests/bench/test_bench_voronoi_partition_methods.py:373:def test_expect_rounds_rejects_duplicate_and_empty_specs():
tests/bench/test_bench_voronoi_partition_methods.py:397:def test_negative_numeric_args_refused_as_false_provenance(
tests/bench/test_bench_voronoi_partition_methods.py:413:def test_expect_rounds_fails_when_a_method_is_unavailable(
tests/bench/test_bench_voronoi_partition_methods.py:439:def test_lloyd_rejects_negative(monkeypatch):
tests/bench/test_bench_voronoi_partition_methods.py:450:def test_empty_methods_refused_instead_of_measuring_nothing(monkeypatch):

codex
The CI workflow exposes another blocker: it collects `tests/bench` but only executes `tests/unit`, selected top-level files, grids, and atmosphere tests. The new shell test is therefore never run by the required CI jobs.
exec
/usr/bin/bash -lc "nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '176,258p'
printf '\\n--- CI execution boundaries ---\\n'
nl -ba .github/workflows/ci.yml | sed -n '87,125p;126,185p;233,249p'
printf '\\n--- launcher artifact and repo/interpreter routing ---\\n'
nl -ba scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch | sed -n '91,131p;139,164p'
printf '\\n--- shell test tests only argv / no results ---\\n'
nl -ba tests/bench/test_mpas_schedule_cost_scan_sbatch.py | sed -n '40,80p;87,154p'
printf '\\n--- old hook absent; hardcoded word evidence ---\\n'
if rg -n 'LEGOESM_SCHEDULE_COST_BENCH' scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch tests/bench; then exit 3; else echo 'old hook absent'; fi
bash -n scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && echo 'bash syntax OK'
git diff --check && echo 'diff whitespace OK'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 3 in 152ms:
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

--- CI execution boundaries ---
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

--- launcher artifact and repo/interpreter routing ---
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
   120	  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
   121	  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
   122	  "$PY" "$BENCH" \
   123	      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
   124	      --methods geometric,sfc,metis --schedule-cost \
   125	      ${4:+--expect-rounds "$4"} \
   126	      --out "$OUT/schedule_cost_s$1.json"
   127	  local rc=$?   # capture BEFORE anything else runs, or `date` overwrites it
   128	  echo "[scan] arm $3 exit=$rc"
   129	  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
   130	  return $rc
   131	}
   139	S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
   140	S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"
   141	
   142	run_arm 8 64,128 "1/3 VALIDATION s8" "$S8_EXPECT"; RC8=$?
   143	run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?
   144	
   145	if [ "$RC8" -ne 0 ] || [ "$RC9" -ne 0 ]; then
   146	  echo "[scan] INSTRUMENT CHECK FAILED (s8 rc=$RC8, s9 rc=$RC9)."
   147	  echo "[scan] The scorer does not reproduce the reference census, so an s10"
   148	  echo "[scan] number from it would be untrustworthy. NOT running arm 3."
   149	  echo "SCAN_ABORTED_VALIDATION"
   150	  exit 1
   151	fi
   152	
   153	# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
   154	run_arm 10 128 "3/3 UNKNOWN s10" ""; RC10=$?
   155	echo "[scan] arm 3 rc=$RC10"
   156	
   157	if [ "$RC10" -ne 0 ]; then
   158	  # Do NOT let SLURM record this as COMPLETED: there is no valid s10 row,
   159	  # and an exit-0 job with a missing result reads as a successful scan.
   160	  echo "SCAN_FAILED_ARM3"
   161	  exit "$RC10"
   162	fi
   163	
   164	echo "SCAN_DONE"

--- shell test tests only argv / no results ---
    40	def _run(tmp_path, codes):
    41	    """Run the launcher with a stub interpreter that exits ``codes`` per arm.
    42	
    43	    ``codes`` is one exit status per bench invocation, in order (arm 1, arm
    44	    2, arm 3).  Returns ``(proc, arms)`` where ``arms`` is the recorded argv
    45	    of each bench call.
    46	    """
    47	    log = tmp_path / "calls.txt"
    48	    stub = tmp_path / "stub_python"
    49	    stub.write_text(
    50	        "#!/usr/bin/env python3\n"
    51	        "import sys, json\n"
    52	        f"codes = {list(codes)!r}\n"
    53	        f"log = {str(log)!r}\n"
    54	        "argv = sys.argv[1:]\n"
    55	        # _env.sh probes the interpreter with `-c 'import jax...'`; answer it
    56	        # without counting it as a bench call.
    57	        "if argv and argv[0] == '-c':\n"
    58	        "    sys.exit(0)\n"
    59	        "with open(log, 'a') as f:\n"
    60	        "    f.write(json.dumps(argv) + '\\n')\n"
    61	        "n = sum(1 for _ in open(log))\n"
    62	        "sys.exit(codes[n - 1] if n <= len(codes) else 0)\n"
    63	    )
    64	    stub.chmod(0o755)
    65	
    66	    env = dict(os.environ)
    67	    env["SLURM_SUBMIT_DIR"] = str(_REPO)
    68	    env["LEGOESM_REPO"] = str(_REPO)
    69	    env["LEGOESM_PYTHON"] = str(stub)
    70	    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    71	    proc = subprocess.run(
    72	        ["bash", str(_SBATCH)], env=env, cwd=str(_REPO),
    73	        capture_output=True, text=True, timeout=600)
    74	
    75	    arms = []
    76	    if log.exists():
    77	        import json
    78	        arms = [json.loads(line) for line in log.read_text().splitlines()
    79	                if line.strip()]
    80	    return proc, arms
    87	def test_launcher_is_not_redirectable_from_the_environment():
    88	    """The bench path must be hardcoded.
    89	
    90	    If it were env-overridable, a stray exported variable could point every
    91	    arm at something that exits 0 and the job would report SCAN_DONE with no
    92	    results — the exact failure this module guards.
    93	    """
    94	    text = _SBATCH.read_text()
    95	    assert "BENCH=scripts/bench/bench_voronoi_partition_methods.py" in text
    96	    assert "LEGOESM_SCHEDULE_COST_BENCH" not in text
    97	
    98	
    99	@pytest.mark.parametrize("codes, failing_arm", [([1, 0, 0], 1), ([0, 1, 0], 2)])
   100	def test_validation_failure_aborts_before_the_unknown_arm(
   101	        tmp_path, codes, failing_arm):
   102	    """EITHER validation arm failing must abort non-zero and skip arm 3.
   103	
   104	    An instrument that misses the known census cannot be trusted on the
   105	    unknown one, so producing an s10 number anyway is worse than none.
   106	    Both arms are exercised: guarding only arm 1 leaves arm 2 unchecked.
   107	    """
   108	    proc, arms = _run(tmp_path, codes)
   109	    assert proc.returncode != 0, proc.stdout[-2000:]
   110	    assert "SCAN_ABORTED_VALIDATION" in proc.stdout
   111	    assert "SCAN_DONE" not in proc.stdout
   112	    assert _levels(arms) == ["8", "9"], (
   113	        f"arm 3 must not run after validation arm {failing_arm} failed: "
   114	        f"{_levels(arms)}")
   115	
   116	
   117	def test_arm3_failure_is_not_reported_as_success(tmp_path):
   118	    """The original BLOCKER: arm 3 fails, the job must NOT exit 0, and must
   119	    propagate the exact status."""
   120	    proc, arms = _run(tmp_path, [0, 0, 3])
   121	    assert proc.returncode == 3, (
   122	        f"arm-3 status not propagated (got {proc.returncode})\n"
   123	        f"{proc.stdout[-2000:]}")
   124	    assert "SCAN_FAILED_ARM3" in proc.stdout
   125	    assert "SCAN_DONE" not in proc.stdout
   126	    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
   127	
   128	
   129	def test_full_success_exits_zero_and_runs_every_arm(tmp_path):
   130	    proc, arms = _run(tmp_path, [0, 0, 0])
   131	    assert proc.returncode == 0, proc.stdout[-2000:]
   132	    assert "SCAN_DONE" in proc.stdout
   133	    assert _levels(arms) == ["8", "9", "10"], _levels(arms)
   134	
   135	
   136	def test_each_arm_carries_the_flags_its_claim_depends_on(tmp_path):
   137	    """The exit plumbing being right is worthless if an arm silently stops
   138	    scoring or stops gating.
   139	
   140	    Without this, dropping ``--schedule-cost`` (nothing is scored) or
   141	    ``--expect-rounds`` (the validation arms assert nothing) would leave
   142	    every other test in this module green.
   143	    """
   144	    _, arms = _run(tmp_path, [0, 0, 0])
   145	    assert len(arms) == 3, arms
   146	    for argv in arms:
   147	        assert "--schedule-cost" in argv, argv
   148	        assert "--lloyd" in argv and argv[argv.index("--lloyd") + 1] == "0"
   149	    # Validation arms must actually gate; the unknown arm must not pretend to.
   150	    for argv in arms[:2]:
   151	        assert "--expect-rounds" in argv, argv
   152	        assert argv[argv.index("--expect-rounds") + 1].strip(), argv
   153	    assert "--expect-rounds" not in arms[2], arms[2]

--- old hook absent; hardcoded word evidence ---
tests/bench/test_mpas_schedule_cost_scan_sbatch.py:96:    assert "LEGOESM_SCHEDULE_COST_BENCH" not in text

codex
## Findings

- **HIGH — the shell regression is not executed in required CI.** CI only collects all tests with `--collect-only`; its actual test jobs run `tests/unit`, top-level files, and selected atmosphere tests—not `tests/bench`. The claimed CI fix therefore has no CI enforcement. [ci.yml:87](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:87) [ci.yml:119](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:119)

- **HIGH — the production instrument is still redirectable; the seam was moved to interpreter/repo selection.** `LEGOESM_PYTHON` remains environment-overridable and `$PY` executes every arm; `LEGOESM_REPO` changes the working directory before the supposedly hardcoded *relative* bench path is resolved. The regression test demonstrates the exact bypass: inject an arbitrary executable that passes the `-c` probe and exits zero per arm. A stray exported knob can still yield `SCAN_DONE` with no benchmark output. [mpas_schedule_cost_scan.sbatch:95](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:95) [mpas_schedule_cost_scan.sbatch:122](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:122) [test_mpas_schedule_cost_scan_sbatch.py:49](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:49)

- **MEDIUM — an empty-successful benchmark remains.** `--methods metis` on an installation without `pymetis` appends only an unavailable row and returns zero when no `--expect-rounds` gate is supplied: zero scored methods, successful command. The new unavailable-method test supplies an expectation, so it cannot catch this case. [bench_voronoi_partition_methods.py:421](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:421) [bench_voronoi_partition_methods.py:491](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:491) [test_bench_voronoi_partition_methods.py:413](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:413)

- **MEDIUM — the six-row gate is currently correct, but its regression test is vacuous on cardinality/content.** Each launcher expectation presently has six entries, and the checker iterates every expected key and fails missing scores. But the shell test only asserts a nonempty `--expect-rounds`; deleting five rows or replacing it with malformed content still passes because the stub never invokes the benchmark parser. [mpas_schedule_cost_scan.sbatch:139](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:139) [bench_voronoi_partition_methods.py:241](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:241) [test_mpas_schedule_cost_scan_sbatch.py:149](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:149)

- **MEDIUM — failed reruns can leave a stale `s10` result that looks current.** All jobs reuse the fixed output directory and never clear or namespace `schedule_cost_s10.json`. If arm 3 fails before writing, the launcher exits nonzero but a prior valid-looking s10 JSON remains; the stub test never creates or validates result artifacts. [mpas_schedule_cost_scan.sbatch:100](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:100) [mpas_schedule_cost_scan.sbatch:154](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:154) [test_mpas_schedule_cost_scan_sbatch.py:75](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:75)

NOT READY
tokens used
105,064
## Findings

- **HIGH — the shell regression is not executed in required CI.** CI only collects all tests with `--collect-only`; its actual test jobs run `tests/unit`, top-level files, and selected atmosphere tests—not `tests/bench`. The claimed CI fix therefore has no CI enforcement. [ci.yml:87](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:87) [ci.yml:119](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.github/workflows/ci.yml:119)

- **HIGH — the production instrument is still redirectable; the seam was moved to interpreter/repo selection.** `LEGOESM_PYTHON` remains environment-overridable and `$PY` executes every arm; `LEGOESM_REPO` changes the working directory before the supposedly hardcoded *relative* bench path is resolved. The regression test demonstrates the exact bypass: inject an arbitrary executable that passes the `-c` probe and exits zero per arm. A stray exported knob can still yield `SCAN_DONE` with no benchmark output. [mpas_schedule_cost_scan.sbatch:95](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:95) [mpas_schedule_cost_scan.sbatch:122](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:122) [test_mpas_schedule_cost_scan_sbatch.py:49](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:49)

- **MEDIUM — an empty-successful benchmark remains.** `--methods metis` on an installation without `pymetis` appends only an unavailable row and returns zero when no `--expect-rounds` gate is supplied: zero scored methods, successful command. The new unavailable-method test supplies an expectation, so it cannot catch this case. [bench_voronoi_partition_methods.py:421](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:421) [bench_voronoi_partition_methods.py:491](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:491) [test_bench_voronoi_partition_methods.py:413](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:413)

- **MEDIUM — the six-row gate is currently correct, but its regression test is vacuous on cardinality/content.** Each launcher expectation presently has six entries, and the checker iterates every expected key and fails missing scores. But the shell test only asserts a nonempty `--expect-rounds`; deleting five rows or replacing it with malformed content still passes because the stub never invokes the benchmark parser. [mpas_schedule_cost_scan.sbatch:139](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:139) [bench_voronoi_partition_methods.py:241](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:241) [test_mpas_schedule_cost_scan_sbatch.py:149](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:149)

- **MEDIUM — failed reruns can leave a stale `s10` result that looks current.** All jobs reuse the fixed output directory and never clear or namespace `schedule_cost_s10.json`. If arm 3 fails before writing, the launcher exits nonzero but a prior valid-looking s10 JSON remains; the stub test never creates or validates result artifacts. [mpas_schedule_cost_scan.sbatch:100](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:100) [mpas_schedule_cost_scan.sbatch:154](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:154) [test_mpas_schedule_cost_scan_sbatch.py:75](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_mpas_schedule_cost_scan_sbatch.py:75)

NOT READY
